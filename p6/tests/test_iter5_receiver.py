import unittest
from unittest.mock import patch

import torch

from p6 import axial_em_transformer as solver
from p6.data import dev_frame


class IterationFiveReceiver(unittest.TestCase):
    @torch.no_grad()
    def test_deferred_r1_only_uses_learned_pa_after_first_pass(self):
        obs = dev_frame(1, 5, 901, 2).obs
        torch.manual_seed(901)
        model = solver.AxialEMTransformer("parity", steps=3)
        model.defer_r1_pa = True
        model.head.read.add_(.25 + .1j)
        with (patch.object(solver, "_demap_input", wraps=solver._demap_input) as demap,
              patch.object(solver, "_emit", wraps=solver._emit) as emit,
              patch.object(solver, "soft_parity_symbols", wraps=solver.soft_parity_symbols) as parity):
            model(obs)
        self.assertEqual([call.args[-1] is None for call in demap.call_args_list],
                         [True, True, False, False, False])
        self.assertEqual([bool(call.args[-1].count_nonzero()) for call in emit.call_args_list],
                         [False, True, True])
        self.assertEqual(parity.call_count, 2)

    @torch.no_grad()
    def test_zero_chirp_weight_uses_unperturbed_ls_taps(self):
        frame = dev_frame(2, 5, 902, 2)
        obs = frame.obs
        y, _, _ = solver._gauge(obs, True)
        geometry = solver._dictionary(2, y.shape[-1])
        _, rot, _, _, _ = solver._initialize(obs, y, geometry)
        baseline = solver._soft_chirp_residual(y, obs.pilots, geometry[0], rot)
        g = torch.Generator().manual_seed(902)
        cue = torch.randn(y.shape, generator=g, dtype=y.dtype)
        off = solver._soft_chirp_residual(y, obs.pilots, geometry[0], rot, cue, 0.)
        small = solver._soft_chirp_residual(y, obs.pilots, geometry[0], rot, cue, .05)
        self.assertTrue(torch.equal(baseline, off))
        self.assertGreater(float((small - off).abs().max()), 0.)
