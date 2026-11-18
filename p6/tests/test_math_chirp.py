import unittest
from unittest.mock import patch

import torch

from p6 import axial_em_transformer as solver
from p6 import math_chirp
from p6.math_chirp_fisher import reliability
from p6.data import dev_frame


class ConditionedChirpTests(unittest.TestCase):
    def test_pilot_aperture_fisher_tracks_centered_time_information(self):
        mask = torch.zeros(20, 4, dtype=torch.bool)
        mask[[0, 9, 10, 19], ::2] = True
        y = torch.full((1, 2, 20, 4), 1+1j, dtype=torch.complex128)
        noise = torch.ones(1, dtype=torch.float64)
        levels = []
        for attenuation in (1., .95, .9, .8, 0.):
            observed = y.clone()
            observed[:, :, 0] *= attenuation
            levels.append(float(reliability(observed, noise, mask)))
            g = .6 + .4j
            torch.testing.assert_close(reliability(observed*g, noise*abs(g)**2, mask),
                                       reliability(observed, noise, mask), atol=3.e-15, rtol=0)
        self.assertTrue(all(a > b for a, b in zip(levels, levels[1:])))
        self.assertGreater(levels[0], .999999)
        self.assertLess(levels[-1], .01)
        self.assertEqual(float(reliability(torch.zeros_like(y), noise, mask)), 0.)

    def test_tangent_recovers_local_chirp_and_profiles_complex_gain(self):
        basis = math_chirp._chirp_basis(20)
        target = basis @ torch.tensor([.16, -.11], dtype=torch.float64)
        rot = torch.ones(1, 20, 32, 1, dtype=torch.complex128)
        amplitude = torch.tensor([.7+.1j, -.3+.9j], dtype=torch.complex128)
        taps = amplitude[None, :, None, None] * (1j * target).exp()
        got = math_chirp._tangent(taps, rot).flatten()
        phase = torch.tensor([-.4, 1.3], dtype=torch.float64)
        changed = math_chirp._tangent(taps * (1j * phase)[None, :, None, None].exp(), rot)
        torch.testing.assert_close(got, changed.flatten(), rtol=0, atol=1.e-14)
        self.assertLess(float((got-target).norm()/target.norm()), .03)
        self.assertLess(float((got - basis @ (basis.mT @ got)).abs().max()), 1.e-15)
        zero = torch.zeros_like(taps, requires_grad=True)
        correction = math_chirp._tangent(zero, rot)
        self.assertFalse(correction.count_nonzero())
        correction.sum().backward()
        self.assertTrue(zero.grad.isfinite().all())

    def test_first_bank_is_original_and_late_candidates_stay_finite(self):
        frame = dev_frame(2, 5, 901, 2)
        obs = frame.obs
        y, _, _ = solver._gauge(obs, True)
        geometry = solver._dictionary(2, y.shape[-1])
        _, rot, _, _, _ = solver._initialize(obs, y, geometry)
        args = (y, obs.pilots, geometry[0], rot, None, .05)
        base = solver._soft_chirp_residual(*args)
        for name in ("tempered", "tangent", "hybrid", "conditional", "conditional_gn", "fisher", "fisher_trust"):
            first = math_chirp.residual(*args, 1, name)
            self.assertTrue(torch.equal(base, first))
            late = math_chirp.residual(*args, 2, name, torch.zeros(2, dtype=torch.float64))
            self.assertTrue(late.isfinite().all())
            self.assertGreater(float((late - base).abs().max()), .001)
            erased = math_chirp.residual(torch.zeros_like(y), *args[1:], 2, name,
                                         torch.zeros(2, dtype=torch.float64))
            self.assertTrue(erased.isfinite().all())

    @torch.no_grad()
    def test_trust_normal_entry_uses_observed_aperture(self):
        from p6 import math_chirp_trust
        from p5b.diagnostics import erased
        model=solver.AxialEMTransformer("parity", steps=3)
        model.chirp_mode="fisher_trust"
        obs=erased(dev_frame(2,6,901,2).obs,"first_pilot_row")
        with patch.object(math_chirp_trust,"residual",wraps=math_chirp_trust.residual) as calls:
            llr,channel=model(obs)
        self.assertTrue(torch.isfinite(llr).all() and torch.isfinite(channel).all())
        self.assertEqual([c.args[-2] for c in calls.call_args_list],[1,2])
        self.assertIsNone(calls.call_args_list[0].args[-1])
        self.assertTrue((calls.call_args_list[1].args[-1]<.01).all())
