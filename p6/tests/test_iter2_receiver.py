"""Checks for the waveform-domain PA inverse and the first learned E-step."""
import unittest
from unittest.mock import patch

import torch

from p4.simulator import decimate_time_to_frequency, oversample_frequency_to_time
from p5.study import setup
from p5b.operator_em_transformer import _emit as archived_emit
from p6 import axial_em_transformer as solver
from p6.data import dev_frame


class ReceiverUpdates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup()

    def test_laguerre_inverse_uses_forward_waveform_basis(self):
        # All time-domain powers sit below the deep-fade safety cap.
        generator = torch.Generator().manual_seed(301)
        z = .4 * torch.randn(2, 5, 32, generator=generator, dtype=torch.complex128)
        radial = torch.randn(2, 5, 2, generator=generator, dtype=torch.complex128)
        zero = torch.zeros_like(radial)
        self.assertTrue(torch.equal(solver._radial_inverse(z, zero), z))
        u = oversample_frequency_to_time(z, 2)
        for regime in (1, 2, 3):
            self.assertTrue(torch.equal(solver._emit(z, regime, zero), archived_emit(z, regime, zero)))
        multiplier = solver._pa_residual_gain(u, 1.e8 * radial)
        self.assertTrue(torch.isfinite(multiplier).all())
        self.assertGreaterEqual(float(multiplier.abs().min()), .85 - 1.e-14)
        self.assertLessEqual(float(multiplier.abs().max()), 1.15 + 1.e-14)
        self.assertLessEqual(float(multiplier.angle().abs().max()), .20 + 1.e-14)
        torch.testing.assert_close(u * multiplier * multiplier.reciprocal(), u,
                                   atol=1.e-15, rtol=1.e-14)
        strong = solver._radial_inverse(z, 1.e8 * radial)
        torch.testing.assert_close(strong, decimate_time_to_frequency(u / multiplier, 32, 2),
                                   atol=1.e-15, rtol=1.e-14)
        self.assertLessEqual(float(strong.norm() / z.norm()), 1 / .85)

    def test_step0_redemaps_and_later_phases_accumulate(self):
        obs = dev_frame(2, 5, 903, 2).obs
        receiver = solver.AxialEMTransformer("pure", steps=3)
        original_handles = solver._physical_handles
        original_demap = solver._demap_input
        events, means, channels = [], [], []

        def angular_update(*args):
            handles = original_handles(*args)
            means.append(args[3].clone())
            events.append("phase")
            return handles._replace(phase=torch.full_like(handles.phase, .7))

        def demap(y, h, nv, radial):
            channels.append(h.clone())
            events.append("demap")
            return original_demap(y, h, nv, radial)

        with (torch.no_grad(), patch.object(solver, "_physical_handles", angular_update),
              patch.object(solver, "_demap_input", demap)):
            receiver(obs)
        self.assertEqual(events[:3], ["demap", "phase", "demap"])
        turns = torch.exp(torch.tensor(.7j, dtype=torch.complex128))
        torch.testing.assert_close(channels[1], channels[0] * turns, atol=2.e-15, rtol=1.e-14)
        self.assertEqual(len(means), 3)
        for step in (1, 2):
            torch.testing.assert_close(means[step], means[0] * turns**step,
                                       atol=2.e-15, rtol=1.e-14)


if __name__ == "__main__":
    unittest.main()
