"""Direct ray coordinates and spatial reach of the centered receiver."""
import unittest
from unittest.mock import patch

import torch

from p5.study import setup
from p6 import axial_em_transformer as receiver
from p6.axial_operator import AxialOperatorBlock
from p6.data import dev_frame
from p6.experiment import model_for
from p6.run_dev_iter4 import objective


class RayCoordinates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup()

    @torch.no_grad()
    def test_changing_one_ray_moves_only_its_own_query_and_key_coordinate(self):
        generator = torch.Generator().manual_seed(901)
        observed = torch.randn(1, 4, 5, 8, 9, generator=generator, dtype=torch.complex128)
        rays = torch.randn(1, 4, 5, 8, 4, generator=generator, dtype=torch.complex128)
        head = AxialOperatorBlock()
        physical_inputs = []
        original = head._global

        def record(q, k, v, axis, gains):
            if axis == 1:
                physical_inputs.append((q.movedim(-2, 1), k.movedim(-2, 1)))
            return original(q, k, v, axis, gains)

        altered = rays.clone()
        altered[..., 2] *= 1j
        with patch.object(head, "_global", record):
            head(torch.cat((observed, rays), dim=-1))
            head(torch.cat((observed, altered), dim=-1))
        expected = torch.zeros_like(physical_inputs[0][0])
        expected[..., 2] = (
            head.width**.5 * (altered[..., 2] - rays[..., 2]) / rays.norm(dim=-1)
        )
        for before, after in zip(*physical_inputs):
            torch.testing.assert_close(after - before, expected, atol=2.e-15, rtol=1.e-14)

    @torch.no_grad()
    def test_spatial_covariance_preserves_dc_and_obeys_operator_reach(self):
        frame = dev_frame(3, 5, 902, 1)
        y = frame.obs.y
        b, r, t, n = y.shape
        nominal, _ = receiver._dictionary(3, n)
        p = nominal.shape[0]
        generator = torch.Generator().manual_seed(902)
        inputs = torch.randn(b, r, t, n, 9 + p, generator=generator, dtype=y.dtype)
        remote = inputs.clone()
        remote[:, 3, :, :, 0] *= 1.25j
        amp = torch.randn(b, r, p, generator=generator, dtype=y.dtype)
        rot = y.new_ones(b, t, p)
        cpe = y.new_ones(b, t)
        x = y.new_ones(b, t, n)
        for arm in ("pure", "local"):
            model = model_for(3, 902, arm, 3)
            model.head.read[:, 0] += .03

            def spatial(features):
                fields, _, operator = model._fields(features, y.real.new_ones(b), False, False)
                strength = receiver._physical_handles(fields, y, amp, rot, cpe, nominal, 3, x).spatial
                return operator.expand(b, r, r), strength

            matrix, strength = spatial(inputs)
            self.assertTrue(matrix.is_complex())
            torch.testing.assert_close(matrix, matrix.mH, rtol=0, atol=2.e-15)
            torch.testing.assert_close(matrix.sum(-1), y.new_ones(b, r), rtol=0, atol=2.e-15)
            eigenvalues = torch.linalg.eigvalsh(matrix)
            self.assertTrue((eigenvalues >= -2.e-15).all() and (eigenvalues <= 1+2.e-15).all())
            changed = receiver._apply_spatial(amp, (matrix, strength))
            self.assertGreater((changed - amp).abs().max().item(), 0.)
            constant = torch.ones_like(amp)
            torch.testing.assert_close(receiver._apply_spatial(constant, (matrix, strength)), constant)
            if arm == "pure":
                self.assertGreater(matrix[:, 0, 3].abs().min().item(), 1.e-8)
                altered, _ = spatial(remote)
                self.assertGreater((altered - matrix).abs().max().item(), 1.e-8)
            else:
                off_band = (torch.arange(r)[:, None] - torch.arange(r)[None]).abs() > 1
                self.assertFalse(matrix[:, off_band].count_nonzero())
                self.assertGreater(matrix[:, 0, 1].abs().min().item(), .01)
                model.head.kernel[1].add_(.5)
                altered, _ = spatial(inputs)
                self.assertGreater((altered - matrix).abs().max().item(), .01)

    def test_first_training_batch_has_exact_anchor_and_every_real_gradient(self):
        for regime in (1, 2, 3):
            frame = dev_frame(regime, 5, 902, 4)
            fixed = model_for(regime, 902, "physics_parity", 3)
            with torch.no_grad():
                expected = fixed(frame.obs)
            for arm in ("parity", "local_parity"):
                with self.subTest(regime=regime, arm=arm):
                    model = model_for(regime, 902, arm, 3)
                    with torch.no_grad():
                        for actual, target in zip(model(frame.obs), expected, strict=True):
                            self.assertTrue(torch.equal(actual, target))
                    objective(frame, model)[0].backward()
                    for name, param in model.head.named_parameters():
                        gradient = torch.view_as_real(param.grad) if param.is_complex() else param.grad
                        self.assertTrue(gradient.isfinite().all(), name)
                        self.assertEqual(gradient.count_nonzero(), gradient.numel(), name)


if __name__ == "__main__":
    unittest.main()
