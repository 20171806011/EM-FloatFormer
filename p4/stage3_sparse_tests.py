"""Focused invariance, physical-layer and connectivity tests for the S05 receiver."""
import unittest

import torch

from p4.next_pilots import joint_soft_channel, sparse_slot
from p4.next_sparse_neural import cross_disabled
from p4.slot_2d import JointPhysicalLoss2D
from p4.stage3_sparse import SparseUnrolledEM


class TestSparseUnrolledEM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.slot = sparse_slot(848888, 4, 2, 20.)
        cls.args = cls.slot.y_grid, cls.slot.pilot_tx, cls.slot.pilot_mask, cls.slot.noise_var

    def setUp(self):
        torch.manual_seed(848889)
        self.receiver = SparseUnrolledEM()

    @torch.no_grad()
    def test_null_correction_is_independent_physical_solver(self):
        y, pilots, mask, nv = self.args
        scale = (y.to(torch.complex128)[:, mask]*pilots.conj()).abs().square().mean(-1, keepdim=True).sqrt()
        expected = joint_soft_channel(y.to(torch.complex128)/scale.unsqueeze(-1), pilots, mask,
                                      nv/scale.square(), "train3", .01, 16) * scale.unsqueeze(-1)
        physical = self.receiver(*self.args, learned=False)
        torch.testing.assert_close(expected, physical.h_hat.to(expected.dtype), atol=1e-5, rtol=1e-5)
        for layer in self.receiver.readouts:
            layer.weight_r.zero_()
            layer.weight_i.zero_()
        full = self.receiver(*self.args)
        torch.testing.assert_close(full.h_hat, physical.h_hat, rtol=0, atol=0)
        torch.testing.assert_close(full.llrs, physical.llrs, rtol=0, atol=0)

    @torch.no_grad()
    def test_posteriors_fix_pilots_and_uniform_missing_information(self):
        y, pilots, mask, nv = self.args
        y = torch.zeros_like(y, dtype=torch.complex128)
        grid = y.clone()
        grid[:, mask] = pilots.to(grid.dtype)
        mean, power = self.receiver._beliefs(y, y, grid, mask, nv)
        torch.testing.assert_close(mean[:, mask], grid[:, mask], rtol=0, atol=0)
        torch.testing.assert_close(power[:, mask], grid[:, mask].abs().square(), rtol=0, atol=0)
        torch.testing.assert_close(mean[:, ~mask], torch.zeros_like(mean[:, ~mask]), rtol=0, atol=1e-14)
        torch.testing.assert_close(power[:, ~mask], torch.ones_like(power[:, ~mask]), rtol=0, atol=1e-14)

    @torch.no_grad()
    def test_channel_scaling_and_llr_invariance(self):
        y, p, mask, nv = self.args
        c = torch.polar(torch.tensor(.37, dtype=torch.float64), torch.tensor(1.13, dtype=torch.float64))
        y = y.to(torch.complex128)
        base = self.receiver(y, p, mask, nv.double())
        transformed = self.receiver(c*y, p, mask, c.abs().square()*nv.double())
        torch.testing.assert_close(transformed.h_hat, c*base.h_hat, rtol=1e-7, atol=1e-8)
        torch.testing.assert_close(transformed.llrs, base.llrs, rtol=1e-7, atol=3e-7)

    def test_both_cross_directions_affect_loss(self):
        full, stages = self.receiver(*self.args, trace=True)
        self.assertEqual(list(stages), ["pilot","map_1","map_2","map_3","learned_3","map_4",
                                       "map_5","map_6","learned_6"])
        loss = JointPhysicalLoss2D()(full,self.slot)
        loss.backward()
        for block in self.receiver.blocks:
            for direction in (block.cross_g2p, block.cross_p2g):
                for p in direction.parameters():
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(torch.isfinite(p.grad).all())
                    self.assertGreater(float(p.grad.abs().max()), 0.)
        with torch.no_grad(), cross_disabled(self.receiver):
            cut = self.receiver(*self.args)
        self.assertGreater(float((full.h_hat.detach()-cut.h_hat).abs().max()),1e-5)


if __name__ == "__main__":
    unittest.main()
