#!/usr/bin/env python3
"""Minimal deterministic checks requiring only PyTorch and local source files."""
import math
import unittest
import torch
from ofdm_baseline import make_batch, decisions_from_channel
from p1b_representation import iq_features
from p1c_architecture import model_factory, TRAIN_PROFILE
from p1d_robustness import Scenario, observed_batch, scenarios, to_complex_storage


class CivilianOFDMInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(min(4,torch.get_num_threads()))

    def test_scenario_matrix_exactly_25_unique(self):
        sc=scenarios()
        self.assertEqual(len(sc),25)
        self.assertEqual(len({s.name for s in sc}),25)

    def test_signal_energy_recovery_noiseless(self):
        torch.manual_seed(3)
        b=make_batch(24,32,300,300,'cpu',TRAIN_PROFILE)
        pred=decisions_from_channel(b.yd,b.yp,b.noise_var)
        self.assertTrue(torch.equal(pred,b.bits))

    def test_model_shapes_and_finite(self):
        torch.manual_seed(5)
        b=make_batch(2,32,10,10,'cpu',TRAIN_PROFILE)
        for arch in ['token_mlp','mlp_mixer','circular_cnn','transformer']:
            for enc in ['raw_iq_magnitude','logamp_phase']:
                model=model_factory(arch,enc,32)
                y=model(b.yp,b.yd,b.noise_var)
                self.assertEqual(tuple(y.shape),(2,32,2))
                self.assertTrue(bool(torch.isfinite(y).all()))

    def test_float_storage_has_finite_values_at_normal_amplitude(self):
        z=torch.complex(torch.tensor([0.,0.5,1.2]),torch.tensor([-1.,0.,0.25]))
        for dtype in ('fp32','fp16_storage','bf16_storage'):
            v=to_complex_storage(z,dtype)
            self.assertTrue(bool(torch.isfinite(v).all()))
            self.assertEqual(v.dtype,torch.complex64)

    def test_noise_report_and_phase_drift(self):
        torch.manual_seed(5)
        b=make_batch(2,32,10,10,'cpu',TRAIN_PROFILE)
        with_noise=Scenario('nv6','train3',10,supplied_noise_db_error=6)
        yp,yd,nv=observed_batch(b,with_noise,torch.Generator().manual_seed(42))
        self.assertTrue(torch.allclose(nv,b.noise_var*(10**.6)))
        self.assertTrue(torch.equal(yp,b.yp))
        self.assertTrue(torch.equal(yd,b.yd))
        with_phase=Scenario('p','train3',10,max_phase_drift_rad=.35)
        _,yd2,_=observed_batch(b,with_phase,torch.Generator().manual_seed(42))
        self.assertTrue(torch.allclose(yd2.abs(),b.yd.abs(),rtol=1e-6,atol=1e-6))

    def test_log_phase_is_invertible_in_real_arithmetic(self):
        z=torch.complex(torch.tensor([0.,-0.01,1.],dtype=torch.float64),
                        torch.tensor([0.,1.,-0.2],dtype=torch.float64))
        feat=iq_features(z,'logamp_phase',sigma=1.,eps=1e-5)
        radius=torch.expm1(feat[...,0])
        rec=(radius+1e-5)*torch.complex(feat[...,1],feat[...,2])
        self.assertLess(float((rec-z).abs().max()),1e-12)

if __name__=='__main__':
    unittest.main(verbosity=2)
