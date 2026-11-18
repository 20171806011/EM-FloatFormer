"""Fourth-iteration PA projection, path gating and first-pass observations."""
import unittest
from unittest.mock import patch

import torch

from p4.simulator import oversample_frequency_to_time
from p5.study import setup
from p6 import axial_em_transformer as solver
from p6.data import dev_frame


class ReceiverProjections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup()

    def test_emitted_pa_residual_is_orthogonal_on_each_symbol(self):
        gen = torch.Generator().manual_seed(901)
        x = torch.randn(2, 5, 32, dtype=torch.complex128, generator=gen)
        radial = 3 * torch.randn(2, 5, 2, dtype=x.dtype, generator=gen)
        x[0, 2] = 0
        for regime in (1, 3):
            zero = solver._emit(x, regime, torch.zeros_like(radial))
            change = solver._emit(x, regime, radial) - zero
            self.assertGreater(float(change.abs().max()), .01)
            inner = (x.conj() * change).sum(-1)
            torch.testing.assert_close(inner, torch.zeros_like(inner), atol=1.e-14, rtol=0)
        u = oversample_frequency_to_time(x, 2)
        nominal = solver._nominal_pa(u.abs().square()) / solver._nominal_gain()
        delta = solver._pa_distortion(u, nominal, radial)
        inner = (u.conj() * delta).sum(-1)
        torch.testing.assert_close(inner, torch.zeros_like(inner), atol=2.e-14, rtol=0)
        self.assertFalse(delta[0, 2].count_nonzero())

    def test_quintic_contrast_rejects_affine_power_and_two_radius_rows(self):
        g = torch.Generator().manual_seed(902)
        power = torch.rand(3, 5, 32, dtype=torch.float64, generator=g) + .1
        x = power.sqrt().to(torch.complex128)
        field = ((1.3 - .4j) + (.2 + .5j) * power)[:, None]
        radial = solver._radial_read(field, x)
        self.assertGreater(radial[..., 0].abs().min().item(), .01)
        self.assertLess(radial[..., 1].abs().max().item(), 1.e-14)
        two_radii = torch.where(power > .5, .2, 1.8).to(torch.float64).sqrt().to(x.dtype)
        radial = solver._radial_read(field, two_radii)
        self.assertLess(radial[..., 1].abs().max().item(), 1.e-14)

    @torch.no_grad()
    def test_pilot_only_residual_on_first_pass_and_phase_only_in_r2(self):
        for regime in (1, 2, 3):
            obs = dev_frame(regime, 5, 901, 2).obs
            model = solver.AxialEMTransformer("parity")
            model.head.read.add_(.2 + .1j)
            with (patch.object(solver, "full_frame_features",
                               wraps=solver.full_frame_features) as features,
                  patch.object(solver, "_physical_handles",
                               wraps=solver._physical_handles) as handles):
                model(obs)
            first, second = [call.args[3][:, :, ~obs.pilot_mask] for call in features.call_args_list]
            self.assertFalse(first.count_nonzero())
            self.assertTrue(second.count_nonzero())
            first_handle = solver._physical_handles(*handles.call_args_list[0].args)
            self.assertEqual(bool(first_handle.phase.count_nonzero()), regime == 2)
            self.assertLess(float(first_handle.delta_h.abs().max()), .06)
