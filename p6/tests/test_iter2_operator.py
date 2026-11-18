"""Iteration-2 parallel operator checks using synthetic receive fields."""

import unittest

import torch

from p6.axial_operator import AxialOperatorBlock


class ParallelMixing(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(901)
        self.fields = torch.randn(2, 4, 5, 12, 13, dtype=torch.complex128)
        self.global_head = AxialOperatorBlock()
        self.local_head = AxialOperatorBlock("local")
        self.local_head.load_state_dict(self.global_head.state_dict())

    @torch.no_grad()
    def test_spatial_output_reads_only_clean_spatial_sites(self):
        original = self.fields
        # Site (0,1,3) sees antenna 3 only in the all-antenna branch.
        remote = original.clone()
        remote[:, 3, 1, 3, 0] *= -.7 + 1.2j
        other_time = original.clone()
        other_time[:, 3, 4, 3, 0] *= -.7 + 1.2j
        other_freq = original.clone()
        other_freq[:, 3, 1, 5, 0] *= -.7 + 1.2j
        for head in (self.global_head, self.local_head):
            def spatial_at_query(fields):
                return head(fields)[0][:, 0, 1, 3, 3]
            initial = spatial_at_query(original)
            for off_line in (other_time, other_freq):
                torch.testing.assert_close(spatial_at_query(off_line), initial, rtol=0, atol=0)
            difference = float((spatial_at_query(remote) - initial).abs().max())
            if head.local:
                self.assertEqual(difference, 0.)
            else:
                self.assertGreater(difference, 1.e-8)
            head.disable_spatial = True
            self.assertEqual(int(spatial_at_query(original).count_nonzero()), 0)

    @torch.no_grad()
    def test_parallel_transport_has_axis_line_support(self):
        _, (coords, product, axes) = self.global_head(self.fields, probe=True)
        parallel = axes["parallel_joint"]
        same = coords[:, None] == coords[None, :]
        expected = (axes["space"] * same[..., 1:].all(-1)
                    + axes["time"] * same[..., [0, 2]].all(-1)
                    + axes["frequency"] * same[..., :2].all(-1)) / 3
        torch.testing.assert_close(parallel, expected)
        off_support = same.sum(-1) < 2
        self.assertEqual(int(parallel[:, off_support].count_nonzero()), 0)
        self.assertTrue(bool((product[:, off_support] > 0).all()))

    def test_shared_paths_have_live_single_antenna_gradients(self):
        fields = self.fields[:, :1, :, :, :-1]
        target = torch.randn_like(fields[..., :3])
        for head in (self.global_head, self.local_head):
            core = head(fields)[0][..., :3]
            (core * target.conj()).real.mean().backward()
            for parameter in head.parameters():
                grad = parameter.grad
                if grad.is_complex():
                    grad = torch.view_as_real(grad)
                self.assertTrue(bool(torch.isfinite(grad).all()))
                self.assertEqual(int(grad.count_nonzero()), grad.numel())


if __name__ == "__main__":
    unittest.main()
