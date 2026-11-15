"""Small white-box numerical sanity checks for frozen P3-C benchmark."""
import unittest
import torch
from p3b_time_link import PAYLOAD_MASK,PROFILES
from p3c_precision import (SEEDS,EVAL_MODES,STORAGE_MODES,ensure_protocol,inspect_capabilities,
  models_for,get_observed,quantize_received,autocast_for,transform_observed,safe_predictions,
  forward_methods,representation_stress,profile_operators,validate_checkpoints)

class PrecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.seed=SEEDS[0]
        cls.frozen,cls.heads=models_for(cls.seed)
        cls.obs=get_observed(cls.seed,'train3',10,'clean',0,frames=4)

    def test_protocol_frozen(self):
        self.assertEqual(len(ensure_protocol()),64)

    def test_checkpoint_inventory(self):
        self.assertEqual(len(validate_checkpoints()),35)

    def test_cpu_autocast_conv_native(self):
        c=inspect_capabilities()
        self.assertEqual(c['float16_cpu_autocast_conv_dtype'],'torch.float16')
        self.assertEqual(c['bfloat16_cpu_autocast_conv_dtype'],'torch.bfloat16')

    def test_cpu_complex_fft_not_falsely_labeled_supported(self):
        c=inspect_capabilities()
        self.assertFalse(c['float16_cpu_fft']['supported'])
        self.assertFalse(c['bfloat16_cpu_fft']['supported'])

    def test_no_unexpected_conv_fallback(self):
        for v in profile_operators():
            expected={'fp32':'torch.float32','fp16_realconv':'torch.float16','bf16_realconv':'torch.bfloat16'}[v['mode']]
            self.assertTrue(all(x==[expected] for x in v['conv_dtypes'].values()))
            self.assertEqual(v['fft_output_dtype'],'torch.complex64')

    def test_input_rounding_has_exact_target_dtype(self):
        for dt in (torch.float16,torch.bfloat16):
            q=quantize_received(self.obs,dt)
            self.assertEqual(q.yp.dtype,torch.complex64)
            self.assertTrue(torch.equal(q.yp.real,self.obs.yp.real.to(dt).float()))
            self.assertTrue(torch.equal(q.yd.imag,self.obs.yd.imag.to(dt).float()))

    def test_reported_noise_is_not_quantized(self):
        q=quantize_received(self.obs,torch.float16)
        self.assertTrue(torch.equal(q.noise_var,self.obs.noise_var))
        self.assertTrue(torch.equal(q.bits,self.obs.bits))

    def test_precision_mode_roundtrip(self):
        for mode in EVAL_MODES:
            p,h=forward_methods(self.obs,self.frozen,self.heads,mode)
            self.assertEqual(len(p),6)
            self.assertTrue(torch.isfinite(h).all())
            for name,y in p.items():
                self.assertEqual(y.shape,(4,32,2))
                self.assertTrue(torch.isfinite(y).all(),name)

    def test_storage_and_mixed_modes_finite(self):
        for mode in STORAGE_MODES:
            p,h=forward_methods(transform_observed(self.obs,mode),self.frozen,self.heads,mode)
            self.assertEqual(len(p),6)
            self.assertTrue(torch.isfinite(h).all(),mode)

    def test_no_true_h_or_bits_leakage(self):
        import dataclasses
        corrupt=dataclasses.replace(self.obs,h=self.obs.h*100,bits=1-self.obs.bits)
        a,_=forward_methods(self.obs,self.frozen,self.heads,'fp32')
        b,_=forward_methods(corrupt,self.frozen,self.heads,'fp32')
        for name in a:self.assertTrue(torch.equal(a[name],b[name]))

    def test_paired_seed_replay(self):
        q=get_observed(self.seed,'train3',10,'clean',0,frames=4)
        self.assertTrue(torch.equal(self.obs.yp,q.yp))
        self.assertTrue(torch.equal(self.obs.yd,q.yd))
        self.assertTrue(torch.equal(self.obs.bits,q.bits))

    def test_qpsk_unknown_tones_only(self):
        self.assertEqual(PAYLOAD_MASK.sum().item(),30)
        p,h=forward_methods(self.obs,self.frozen,self.heads,'fp32')
        for y in p.values():
            err,nan,pred=safe_predictions(y,self.obs.bits)
            self.assertEqual(nan,0)
            self.assertEqual(pred.numel(),4*60)
            self.assertGreaterEqual(err,0)

    def test_nonfinite_reports_failure_not_fake_ber(self):
        x=torch.zeros((4,32,2))
        x[0,3,0]=float('nan')
        error,nonfinite,pred=safe_predictions(x,self.obs.bits)
        self.assertIsNone(error)
        self.assertIsNone(pred)
        self.assertEqual(nonfinite,1)

    def test_high_dynamic_encoding_stress(self):
        x=representation_stress()
        self.assertEqual(len(x),6)
        raw16=next(r for r in x if r['encoding']=='raw_cartesian' and r['storage']=='float16')
        log16=next(r for r in x if r['encoding']=='log_phase' and r['storage']=='float16')
        self.assertGreater(raw16['nonfinite'],0)
        self.assertEqual(log16['nonfinite'],0)

if __name__=='__main__':unittest.main(verbosity=2)
