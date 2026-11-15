import math
import unittest
import torch
from p3b_time_link import (N,CP,PROFILES,ALL_CONDITIONS,complex_rand,qpsk,draw_base,tx_cp,
                           causal_linear_filter,link,rapp,quantize_complex,build_batch,PAYLOAD_MASK)
from p3b_receivers import (DataResidualHead,new_models,logits_from_channel,phase_from_anchors,
                           channels,classical_logits,bit_count,load_frozen_d)
from pathlib import Path

class TimeLinkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
    def frame(self,batch=12,snr=10):
        gen=torch.Generator().manual_seed(123456)
        return draw_base(batch,PROFILES['long12'],snr,gen)

    def test_clean_zero_impairment_matches_frequency_domain(self):
        b=self.frame(32)
        b.noise_p=torch.zeros_like(b.noise_p)
        b.noise_d=torch.zeros_like(b.noise_d)
        got=link(b)
        self.assertLess(float((got.yp-b.h).abs().max()),1e-6)
        self.assertLess(float((got.yd-b.h*b.xdata).abs().max()),1e-6)

    def test_known_pilot_and_public_anchors(self):
        b=self.frame(16)
        self.assertLess(float((b.xpilot.abs()-1).abs().max()),1e-6)
        self.assertTrue(torch.equal(b.bits[:,0],torch.ones_like(b.bits[:,0])))
        self.assertTrue(torch.equal(b.bits[:,8],torch.ones_like(b.bits[:,8])))
        self.assertEqual(int(PAYLOAD_MASK.sum()),30)
        self.assertFalse(bool(PAYLOAD_MASK[0]));self.assertFalse(bool(PAYLOAD_MASK[8]))

    def test_cp_condition_and_freq_convolution(self):
        b=self.frame(17)
        tx=tx_cp(b.xdata)
        y=causal_linear_filter(tx,b.h_time)[:,CP:]
        self.assertLess(float((torch.fft.fft(y,dim=-1)-b.h*b.xdata).abs().max()),1e-6)
        self.assertGreater(CP,len(PROFILES['long12'])-2)

    def test_thermal_noise_spectral_power(self):
        g=torch.Generator().manual_seed(971)
        b=draw_base(5000,PROFILES['train3'],10,g)
        n=torch.fft.fft(b.noise_p,dim=-1)
        self.assertLess(abs(float(n.abs().square().mean())-.1),.003)
        got=link(b)
        resid=got.yp-b.h
        self.assertLess(abs(float(resid.abs().square().mean())-.1),.003)

    def test_pilot_papr_not_all_ones(self):
        b=self.frame(256)
        t=torch.fft.ifft(b.xpilot,dim=-1)
        papr=t.abs().square().max(dim=-1).values/t.abs().square().mean(dim=-1)
        self.assertGreater(float(papr.mean()),2.0)
        self.assertLess(float(papr.mean()),10.0)

    def test_rapp_compression_preserves_phase_and_bounds(self):
        x=torch.complex(torch.tensor([0.01,.2,.8]),torch.tensor([.02,.1,.3]))
        a=rapp(x,'strong')
        self.assertTrue(torch.all(a.abs()<=x.abs()+1e-8))
        self.assertLess(float((x*a.conj()).imag.abs().max()),1e-6)
        self.assertLess(float(a.abs().max()),.25)

    def test_impulse_noise_power_and_non_gaussian_tail(self):
        # Do not confuse receiver random impulses with active signal generation.
        b=self.frame(5000)
        base=link(b,'clean')
        imp=link(b,'impulse_strong',extra_seed=777)
        self.assertAlmostEqual(float((imp.noise_var/base.noise_var).mean()),4,places=5)
        self.assertGreater(imp.stats['burst_rate'],.035)
        self.assertLess(imp.stats['burst_rate'],.065)
        self.assertGreater(float((imp.yp-base.yp).abs().square().mean()),.20)

    def test_cpe_is_data_only_and_anchor_identifies_angle(self):
        b=self.frame(1024,30)
        clean=link(b,'clean')
        cpe=link(b,'cpe_strong',extra_seed=17)
        self.assertTrue(torch.equal(clean.yp,cpe.yp))
        _,phi=phase_from_anchors(cpe.yd,b.h)
        corrected=cpe.yd*torch.polar(torch.ones_like(phi),-phi)
        e0=(clean.yd-b.h*b.xdata).abs().square().mean()
        e1=(corrected-b.h*b.xdata).abs().square().mean()
        self.assertLess(float(e1),float(e0)+.002)

    def test_adc_reports_real_saturation(self):
        b=self.frame(256)
        mild=link(b,'adc_mild');strong=link(b,'adc_strong')
        self.assertGreaterEqual(strong.stats['adc_saturation'],mild.stats['adc_saturation'])
        self.assertGreater(strong.stats['adc_saturation'],0)
        self.assertTrue(torch.isfinite(strong.yp).all())

    def test_clean_vs_nonideal_controls_and_noise_axes(self):
        b=self.frame(32)
        vals=[link(b,c,extra_seed=44) for c in ALL_CONDITIONS]
        self.assertEqual(len(vals),9)
        for got in vals:
            self.assertTrue(torch.isfinite(got.yd).all())
            self.assertEqual(got.bits.shape,(32,32,2))
            self.assertEqual(got.yp.shape,(32,32))
        self.assertTrue(torch.all(vals[0].noise_var==vals[1].noise_var))

    def test_neural_initialization_finite_gradients(self):
        models=new_models(48101)
        self.assertEqual(len(models),6)
        base=list(models.values())[0].state_dict()
        for v in models.values():
            for k in base:self.assertTrue(torch.equal(base[k],v.state_dict()[k]))
        r=self.frame(8)
        recv=link(r)
        # Use true channel only in this *unit test* to probe numeric gradients,
        # not in training or model-inference input code.
        for m in models.values():
            m.zero_grad()
            y=m(recv.yd,recv.yp,recv.noise_var)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(
                y[:,PAYLOAD_MASK],recv.bits[:,PAYLOAD_MASK].float())
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))
            self.assertLess(sum(x.numel() for x in m.parameters()),5000)

    def test_all_references_receive_only_observed_data(self):
        # Structural check: changing inaccessible true-H cannot affect a
        # classical receiver's forward results.
        b=self.frame(5)
        recv=link(b)
        dpath=Path(__file__).parent/'p3a2_checkpoints'/'model_D_no_pos_10_seed47101.pt'
        if not dpath.exists():self.skipTest('Frozen P3-A2 D checkpoint unavailable')
        m=load_frozen_d(dpath)
        a,_=classical_logits(recv,m)
        recv.h=torch.full_like(recv.h,999.+0j)
        c,_=classical_logits(recv,m)
        for k in a:self.assertTrue(torch.equal(a[k],c[k]),k)

if __name__=='__main__':unittest.main()
