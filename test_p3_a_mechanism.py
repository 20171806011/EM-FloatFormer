#!/usr/bin/env python3
"""Pretraining structural tests: never consume P3-A2 heldout evaluation streams."""
import math
import unittest
import torch
from torch.nn import functional as F
from ofdm_baseline import make_batch, estimate_channel, decisions_from_channel
from p3_models import EquivDelayConv
from p3_a import dft_threshold
from p3_a_mechanism import (FactorialDelayConv, init_shared, ARMS, PROFILES,
                             test_data_seed, N, sha256,EXPECTED_PROTOCOL_SHA256)
from pathlib import Path

class MechanismTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(8831)
        self.b=make_batch(7,N,10,10,'cpu')

    def test_protocol_frozen(self):
        self.assertEqual(sha256(Path(__file__).with_name('P3_A_MECHANISM_PROTOCOL.md')),
                         EXPECTED_PROTOCOL_SHA256)

    def test_same_parameter_shapes_and_pair_init(self):
        models=init_shared(47101)
        keys=[set(m.state_dict()) for m in models.values()]
        self.assertTrue(all(k==keys[0] for k in keys))
        first=list(models.values())[0]
        self.assertEqual(sum(p.numel() for p in first.parameters()),985)
        for other in list(models.values())[1:]:
            for a,b in zip(first.parameters(),other.parameters()):
                self.assertTrue(torch.equal(a,b))

    def test_original_a_bit_identical(self):
        torch.manual_seed(102)
        old=EquivDelayConv().eval()
        new=FactorialDelayConv(True,0.0).eval()
        new.load_state_dict(old.state_dict())
        with torch.inference_mode():
            c_old=old.channel(self.b.yp,self.b.yd,self.b.noise_var)
            c_new=new.channel(self.b.yp,self.b.yd,self.b.noise_var)
        self.assertTrue(torch.equal(c_old,c_new))

    def test_shapes_finite_and_gradients(self):
        for name,(pos,beta) in ARMS.items():
            model=FactorialDelayConv(pos,beta)
            logits=model(self.b.yp,self.b.yd,self.b.noise_var)
            self.assertEqual(logits.shape,self.b.bits.shape,name)
            self.assertTrue(torch.isfinite(logits).all(),name)
            loss=F.binary_cross_entropy_with_logits(logits,self.b.bits.float())
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),name)
            self.assertTrue(model.stem.weight.grad.abs().sum()>0,name)

    def test_phase_rotation_equivariance(self):
        torch.manual_seed(224)
        for pos,beta in ARMS.values():
            model=FactorialDelayConv(pos,beta).eval()
            h=model.channel(self.b.yp,self.b.yd,self.b.noise_var)
            log=model(self.b.yp,self.b.yd,self.b.noise_var)
            for phi in (.3,1.23,2.76):
                u=torch.polar(torch.ones_like(self.b.yp.real),torch.full_like(self.b.yp.real,phi))
                hr=model.channel(self.b.yp*u,self.b.yd*u,self.b.noise_var)
                lr=model(self.b.yp*u,self.b.yd*u,self.b.noise_var)
                self.assertLess(float((hr-h*u).abs().max()),1e-5)
                self.assertLess(float((lr-log).abs().max()),1e-5)
                self.assertTrue(torch.equal(lr>0,log>0))

    def test_no_position_circular_delay_equivariance(self):
        for beta in (0.,.10):
            model=FactorialDelayConv(False,beta).eval()
            original_t=torch.fft.ifft(self.b.yp,dim=-1)
            expected_t=torch.roll(original_t,shifts=7,dims=-1)
            shifted_yp=torch.fft.fft(expected_t,dim=-1)
            h=model.channel(self.b.yp,self.b.yd,self.b.noise_var)
            got=model.channel(shifted_yp,self.b.yd,self.b.noise_var)
            expected=torch.fft.fft(torch.roll(torch.fft.ifft(h,dim=-1),7,dims=-1),dim=-1)
            self.assertLess(float((got-expected).abs().max()),1e-5)

    def test_removed_pos_channels_have_no_effect(self):
        for beta in (0.,.10):
            model=FactorialDelayConv(False,beta).eval()
            h0=model.channel(self.b.yp,self.b.yd,self.b.noise_var)
            with torch.no_grad():model.stem.weight[:,4:6].fill_(1000.)
            h1=model.channel(self.b.yp,self.b.yd,self.b.noise_var)
            self.assertTrue(torch.equal(h0,h1))

    def test_floor_bounds(self):
        for pos,beta in ARMS.values():
            model=FactorialDelayConv(pos,beta)
            g=model.effective_gate(self.b.yp,self.b.noise_var)
            self.assertGreaterEqual(float(g.min()),beta-1e-6)
            self.assertLessEqual(float(g.max()),1.0+1e-6)

    def test_data_not_used_as_pilot(self):
        for pos,beta in ARMS.values():
            model=FactorialDelayConv(pos,beta)
            a=model.channel(self.b.yp,self.b.yd,self.b.noise_var)
            b=model.channel(self.b.yp,torch.randn_like(self.b.yd),self.b.noise_var)
            self.assertTrue(torch.equal(a,b))

    def test_fixed_baseline_matches_p3(self):
        h=dft_threshold(self.b.yp,self.b.noise_var,'hard')
        ref=torch.fft.ifft(self.b.yp,dim=-1)
        t=(self.b.noise_var/N*math.log(N/.05)).sqrt()
        hand=torch.fft.fft(ref*(ref.abs()>t),dim=-1)
        self.assertLess(float((hand-h).abs().max()),1e-6)
        d8=estimate_channel(self.b.yp,self.b.noise_var,'dft',taps=8)
        d12=estimate_channel(self.b.yp,self.b.noise_var,'dft',taps=12)
        self.assertEqual(d8.shape,d12.shape)

    def test_evaluation_seed_ranges_disjoint(self):
        for seed in (47101,47102,47103,47104,47105):
            for profile in PROFILES:
                self.assertGreaterEqual(test_data_seed(seed,profile,10),361000000)
                self.assertNotEqual(test_data_seed(seed,profile,10),84300000+seed)
        self.assertEqual(len({test_data_seed(seed,p,s) for seed in (47101,47102,47103,47104,47105)
                              for p in PROFILES for s in (0,10,20)}),75)


def run_tests():
    s=unittest.TestLoader().loadTestsFromTestCase(MechanismTests)
    o=unittest.TextTestRunner(verbosity=2).run(s)
    if not o.wasSuccessful():raise AssertionError('Mechanism tests failed')
    return o.testsRun

if __name__=='__main__':run_tests()
