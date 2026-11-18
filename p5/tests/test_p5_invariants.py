"""Receiver and link invariants, including tests away from initialization."""
from dataclasses import replace
import math
import unittest

import torch
from torch.nn import functional as F

from p4.classical_baselines import calibrated_16qam_llrs
from p4.simulator import complex_randn, q16
from p4.slot_2d import decode_min_sum, encode_ldpc
from p5.classical import receive
from p5.unrolled_em_transformer import UnrolledEMTransformer, fourier_anchor
from p5.wavefield_sim import simulate, steer


class Invariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(54193)

    @staticmethod
    def loss(model, frame):
        logits, _ = model(frame.obs)
        return F.binary_cross_entropy_with_logits(logits[:, ~frame.obs.pilot_mask], frame.bits.double())

    def test_zero_step_physics_and_fourier(self):
        for r in (1, 2, 3):
            frame = simulate(r, 2, 11, 54193)
            for steps in (3, 4):
                model = UnrolledEMTransformer(steps=steps)
                ref = UnrolledEMTransformer(variant="physics", steps=steps)
                for a, b in zip(model(frame.obs), ref(frame.obs)):
                    torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-13)
        obs = simulate(1, 2, 11, 54193).obs
        ideal = torch.exp(-2j * math.pi * 4 * torch.fft.fftfreq(32, dtype=torch.float64))
        y = ideal * obs.pilots
        h = fourier_anchor(replace(obs, y=y.expand_as(obs.y)))
        torch.testing.assert_close(h, ideal.expand_as(h), atol=2e-13, rtol=0)

    def test_paired_arms_and_first_step_gradients(self):
        for r in (1, 2, 3):
            frame = simulate(r, 4, 11, 54193)
            sizes = []
            for arm in ("full", "local", "no_gauge", "no_unroll"):
                model = UnrolledEMTransformer(variant=arm)
                pars = [p for p in model.parameters() if p.requires_grad]
                sizes.append(sum(p.numel() * (1 + p.is_complex()) for p in pars))
                self.loss(model, frame).backward()
                for p in pars:
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(torch.isfinite(p.grad).all(), (r, arm))
                    self.assertGreater(p.grad.abs().max().item(), 0., (r, arm))
            self.assertEqual(len(set(sizes)), 1, sizes)

    def test_cstar_learned_map_and_extremes(self):
        # Train once first. The centered network at initialization would only test the solver.
        g = torch.Generator().manual_seed(54194)
        for r in (1, 2, 3):
            frame = simulate(r, 3, 13, 54193)
            model = UnrolledEMTransformer()
            opt = torch.optim.Adam(model.parameters(), lr=.01)
            self.loss(model, frame).backward()
            opt.step()
            with torch.no_grad():
                ref_llr, ref_h = model(frame.obs)
                for scale in [1e-35, 1e20] + [2.**i for i in range(-12, 13)]:
                    phases = torch.rand(3, dtype=torch.float64, generator=g) * 2 * math.pi
                    gain = torch.polar(torch.full_like(phases, scale), phases)[:, None, None, None]
                    obs = replace(frame.obs, y=gain * frame.obs.y, noise_var=scale**2 * frame.obs.noise_var)
                    llr, h = model(obs)
                    self.assertTrue(torch.isfinite(llr).all() and torch.isfinite(h).all())
                    self.assertEqual(int(((llr > 0) != (ref_llr > 0)).sum()), 0)
                    self.assertLess((llr - ref_llr).abs().max().item(), 1e-5)
                    torch.testing.assert_close(h / gain, ref_h, atol=1e-10, rtol=1e-10)

    def test_steering_and_alias_geometry(self):
        a = steer(torch.tensor([0., math.pi/6], dtype=torch.float64), 4)
        torch.testing.assert_close(a.abs().square().sum(-1), torch.ones(2, dtype=torch.float64))
        torch.testing.assert_close(a[1, 1] / a[1, 0], torch.tensor(1j, dtype=torch.complex128))
        f = simulate(2, 4, 11, 54195)
        rows = f.obs.pilot_mask.any(-1).nonzero().flatten()
        spacing = int(rows[1] - rows[0]) * 48 / 32
        wall = .5 / spacing
        self.assertAlmostEqual(wall, 1/27)
        for endpoints in (f.truth.nu_start, f.truth.nu_end):
            self.assertTrue(((endpoints.abs() >= .045) & (endpoints.abs() <= .080)).all())
            self.assertTrue((endpoints.abs() > wall).all())
        self.assertTrue((f.truth.nu_start.sign() == f.truth.nu_end.sign()).all())

    def test_backward_at_extreme_gains(self):
        frame = simulate(3, 2, 11., 54196)
        model = UnrolledEMTransformer()
        # Check the physical input Jacobian and trainable gradients away from initialization.
        opt = torch.optim.Adam(model.parameters(), lr=.006)
        self.loss(model, frame).backward()
        opt.step()
        for scale in (1e-35, 1., 1e20):
            model.zero_grad(set_to_none=True)
            y = (frame.obs.y * complex(.6*scale, .8*scale)).requires_grad_()
            obs = replace(frame.obs, y=y, noise_var=frame.obs.noise_var * scale**2)
            self.loss(model, replace(frame, obs=obs)).backward()
            for grad in [y.grad, *(p.grad for p in model.parameters() if p.requires_grad)]:
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(float(grad.abs().max()), 0.)

    def test_exact_pilot_erasure_and_no_information(self):
        for r in (1, 2, 3):
            frame = simulate(r, 2, 12, 55139)
            model = UnrolledEMTransformer()
            opt = torch.optim.Adam(model.parameters(), lr=.006)
            self.loss(model, frame).backward()
            opt.step()
            with torch.no_grad():
                for erased in ("first", "last", "all_pilots", "whole_signal"):
                    y = frame.obs.y.clone()
                    rows = frame.obs.pilot_mask.any(-1).nonzero().flatten()
                    if erased == "whole_signal":
                        y.zero_()
                    elif erased == "all_pilots":
                        y[:, :, frame.obs.pilot_mask] = 0
                    else:
                        row = rows[0 if erased == "first" else -1]
                        y[:, :, row, frame.obs.pilot_mask[row]] = 0
                    obs = replace(frame.obs, y=y)
                    llr, h = model(obs)
                    if erased == "whole_signal":
                        self.assertEqual(int(torch.count_nonzero(llr)), 0)
                    for gain in (2j, complex(-.7, 1.1), 1e-35j, -1e20):
                        other = replace(obs, y=y * gain, noise_var=abs(gain)**2 * obs.noise_var)
                        now, channel = model(other)
                        torch.testing.assert_close(llr, now, rtol=0., atol=1e-5)
                        torch.testing.assert_close(h, channel/gain, rtol=1e-10, atol=1e-10)
                        self.assertEqual(int(((llr > 0) != (now > 0)).sum()), 0)

    def test_ldpc_waterfall_sanity(self):
        gen = torch.Generator().manual_seed(989066)
        info = torch.randint(2, (256, 64), generator=gen)
        code, hmat = encode_ldpc(info)
        x = q16(code.reshape(256, 32, 4))
        noise = complex_randn(x.shape, gen)
        rates = []
        for snr in (4., 8., 10.):
            var = torch.tensor(10.**(-snr/10))
            llr = calibrated_16qam_llrs(x + var.sqrt() * noise, var).reshape(256, 128)
            errors = (decode_min_sum(llr, hmat, max_iter=12) != info).any(-1).sum()
            rates.append(float(errors) / 256)
        self.assertGreater(rates[0], .35)
        self.assertTrue(.05 <= rates[1] <= .35)
        self.assertLess(rates[2], .05)

    def test_classical_privilege_boundary(self):
        f = simulate(1, 2, 12., 54195)
        with self.assertRaises(ValueError):
            receive(f.obs, "nominal_idd_pa", genie_pa=f.truth.pa)
        with self.assertRaises(ValueError):
            receive(f.obs, "genie_pa")
        self.assertEqual(set(vars(f.obs)), {"pilots", "pilot_mask", "y", "noise_var", "regime"})


if __name__ == "__main__":
    unittest.main()
