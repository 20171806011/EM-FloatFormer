"""Synthetic checks of the shared R2 and R3 physical updates."""
import unittest
from unittest.mock import patch

import torch

from p5.wavefield_sim import Observation
from p5b.operator_em_transformer import _ray_bank
from p6 import axial_em_transformer as solver


class PhysicalUpdates(unittest.TestCase):
    def test_spatial_readout_projects_a_neutral_strength_for_each_ray(self):
        t, n = 5, 32
        basis, _ = solver._dictionary(3, n)
        p = len(basis)
        gen = torch.Generator().manual_seed(319)
        amp = torch.randn(2, 4, p, dtype=basis.dtype, generator=gen)
        rot = torch.polar(torch.ones(2, t, p, dtype=torch.float64),
                          torch.randn(2, t, p, dtype=torch.float64, generator=gen))
        cpe = rot[:, :, 0]
        modulation = rot * cpe[..., None]
        h = (amp[:, :, None] * modulation[:, None]) @ basis
        fields = torch.zeros(*h.shape, 4, dtype=h.dtype)
        x = torch.ones(len(h), t, n, dtype=h.dtype)

        def read():
            return solver._physical_handles(fields, h, amp, rot, cpe, basis, 3, x).spatial

        operator = torch.ones(4, 4, dtype=amp.dtype) / 4
        self.assertTrue(torch.equal(solver._apply_spatial(amp, (operator, read())), amp))
        fields[:, 3, :, :, 3] = modulation[:, :, :1] * basis[0]
        strength = read()
        self.assertEqual(strength.shape, (len(h), p))
        self.assertFalse(strength.is_complex())
        self.assertTrue((strength >= 0).all() and (strength <= .18).all())
        self.assertGreater(strength[:, 0].min().item(), 0.)
        updated = solver._apply_spatial(amp, (operator, strength))
        torch.testing.assert_close(updated, (1-strength[:, None])*amp
                                   + strength[:, None]*amp.mean(1, keepdim=True))
        phase = .6 + .8j
        fields *= phase
        amp *= phase
        h *= phase
        torch.testing.assert_close(read(), strength)
        torch.testing.assert_close(solver._apply_spatial(amp, (operator, read())), phase * updated)

    def test_public_bank_moves_opposite_sign_prior_toward_synthetic_path(self):
        t, n = 20, 32
        basis, _ = solver._dictionary(2, n)
        bank, means, _ = _ray_bank(2, t, n)
        # Each target is an observed positive chirp. Seed the receiver with
        # the corresponding negative chirp for each of the three delay taps.
        ids = torch.tensor([4, 40, 76])
        target = means[ids].T[None, None]
        initial = bank[ids + len(bank) // 2].permute(1, 2, 0)[None]
        generator = torch.Generator().manual_seed(317)
        x = torch.polar(torch.ones(1, t, n, dtype=torch.float64),
                        torch.rand(1, t, n, dtype=torch.float64, generator=generator) * 6)
        clean = (x[:, None] * (target @ basis)).detach().requires_grad_()
        update = solver._soft_chirp_residual(clean, x, basis, initial)
        actual = solver._centered_angles(bank[ids, :, n // 2]).T[None]
        start = solver._centered_angles(initial[:, :, n // 2].mT).mT
        self.assertLess((actual - start - update).norm(), .2 * (actual - start).norm())
        gradient, = torch.autograd.grad(update.square().mean(), clean)
        self.assertTrue(bool(gradient.isfinite().all()))
        self.assertGreater(gradient.abs().max().item(), 0.)

    @torch.no_grad()
    def test_bank_starts_only_after_the_first_waveform_solve(self):
        generator = torch.Generator().manual_seed(318)
        t, n = 5, 32
        y = torch.randn(1, 1, t, n, dtype=torch.complex128, generator=generator)
        mask = torch.zeros(t, n, dtype=torch.bool)
        mask[[0, -1], ::2] = True
        obs = Observation(y, mask, mask.to(y.dtype), torch.tensor([.2], dtype=torch.float64), 2)
        for arm in ("pure", "local"):
            model = solver.AxialEMTransformer(arm, steps=3)
            with patch.object(solver, "_physical_handles", wraps=solver._physical_handles) as handles:
                model(obs)
            residuals = [call.args[-1] for call in handles.call_args_list]
            self.assertEqual(len(residuals), 3)
            self.assertIsNone(residuals[0])
            for residual in residuals[1:]:
                self.assertEqual(residual.shape, (1, t, 3))
                self.assertGreater(residual.abs().max().item(), 0.)

    def test_oscillator_update_obeys_public_regime(self):
        for regime in (1, 2, 3):
            t, n = 5, 32
            basis, _ = solver._dictionary(regime, n)
            p = len(basis)
            r = 4 if regime == 3 else 1
            mean_rot = torch.ones(1, t, p, dtype=basis.dtype)
            cpe = torch.ones(1, t, dtype=basis.dtype)
            amp = torch.ones(1, r, p, dtype=basis.dtype)
            h = (amp[:, :, None] * mean_rot[:, None]) @ basis
            fields = torch.zeros(*h.shape, 4, dtype=h.dtype)
            fields[..., 2] = 1j * h
            handle = solver._physical_handles(fields, h, amp, mean_rot, cpe, basis, regime,
                                              torch.ones(1, t, n, dtype=h.dtype))
            if regime == 1:
                self.assertGreater(handle.cpe.abs().max().item(), .1)
            else:
                self.assertEqual(handle.cpe.count_nonzero().item(), 0)
