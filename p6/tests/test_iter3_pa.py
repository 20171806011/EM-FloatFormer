"""Full-grid PA readout and conservative likelihood correction."""

import unittest
from unittest.mock import patch

import torch

from p5.unrolled_em_transformer import _equalize
from p6 import axial_em_transformer as solver


class PAUpdates(unittest.TestCase):
    def test_contrast_rejects_symbol_common_phase(self):
        powers = torch.tensor([.2, 1., 1.8, .2, 1., 1.8, 1., 1.], dtype=torch.float64)
        x = powers.sqrt().to(torch.complex128).expand(1, 5, -1)
        phase = torch.tensor([.1, -.7, .4, -.2, .9], dtype=torch.float64)
        pa = (1j * phase)[None, None, :, None].expand(1, 2, 5, 8)
        radial = solver._radial_read(pa, x)
        torch.testing.assert_close(radial, torch.zeros_like(radial), atol=1.e-15, rtol=0)

    def test_interior_payload_tone_contributes_to_both_endpoints(self):
        powers = torch.ones(1, 5, 8, dtype=torch.float64)
        powers[..., 1] = .2
        powers[..., 3] = 1.8
        x = powers.sqrt().to(torch.complex128)
        pa = torch.zeros(1, 2, 5, 8, dtype=torch.complex128)
        pa[:, :, 1:3, 3] = 1 + .5j
        radial = solver._radial_read(pa, x)
        self.assertTrue((radial[:, (0, -1)].abs() > .0001).all())
        self.assertTrue((radial[:, 0].abs() > radial[:, -1].abs()).all())
        torch.testing.assert_close(radial[:, 2], (radial[:, 0] + radial[:, -1]) / 2)

    def test_zero_residual_preserves_likelihood_and_dewarp_is_bounded(self):
        generator = torch.Generator().manual_seed(428)
        shape = (2, 2, 5, 32)
        h = torch.randn(shape, generator=generator, dtype=torch.complex128)
        h[..., 0] = 0.
        y = torch.randn(shape, generator=generator, dtype=torch.complex128)
        y[..., 7] *= 1.e4
        nv = torch.tensor([.01, .2], dtype=torch.float64)
        radial = torch.randn(2, 5, 2, generator=generator, dtype=torch.complex128) * 5
        with patch.object(solver, "_radial_inverse", side_effect=AssertionError("archived inverse")):
            baseline = _equalize(y, h, nv)
            restored = solver._demap_input(y, h, nv, torch.zeros_like(radial))
            for expected, actual in zip(baseline, restored):
                self.assertTrue(torch.equal(actual, expected))
            z, variance, informed = solver._demap_input(y, h, nv, radial)

        linear, linear_variance, valid = baseline
        self.assertTrue(torch.equal(variance, linear_variance))
        self.assertTrue(torch.equal(informed, valid))
        correction = (linear - z).abs()
        self.assertGreater(correction.max().item(), 1.e-4)
        self.assertLess(correction.max().item(), .04)


if __name__ == "__main__":
    unittest.main()
