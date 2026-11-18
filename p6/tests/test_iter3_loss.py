"""The training cap and the anchored penalty act only on the fit objective."""
import unittest

import torch
from torch.nn import functional as F

from p5.study import payload, setup
from p6.data import dev_frame
from p6.experiment import model_for
from p6.training_iter3 import ANCHOR, AUX_NMSE, CAP, objective


class TrainingLoss(unittest.TestCase):
    @torch.no_grad()
    def test_iter4_selected_objective_uses_reduced_nmse_for_every_regime(self):
        from p6.run_dev_iter4 import objective as selected_loss

        setup()
        for regime in (1, 2, 3):
            frame = dev_frame(regime, 5, 901, 2)
            model = model_for(regime, 901, "parity", 3)
            next(model.head.parameters()).add_(.02j)
            llr, channel = model(frame.obs)
            z = payload(frame, llr)
            bits = frame.bits.double()
            bce = F.binary_cross_entropy_with_logits(CAP * torch.tanh(z / CAP), bits)
            nmse = (channel - frame.h_true).abs().square().sum() / frame.h_true.abs().square().sum()
            distance = sum((p - q).abs().square().sum() for p, q in
                           zip(model.head.parameters(), model.initial_map.parameters(), strict=True))
            selected, raw = selected_loss(frame, model)
            torch.testing.assert_close(selected, bce + .005 * nmse + .5e-4 * distance,
                                       atol=1.e-14, rtol=0.)
            torch.testing.assert_close(raw, F.binary_cross_entropy_with_logits(z, bits), atol=0., rtol=0.)

    @torch.no_grad()
    def test_cap_control_anchor_and_raw_scoring(self):
        setup()
        frame = dev_frame(1, 5, 901, 2)
        model = model_for(1, 901, "parity", 3)
        next(model.head.parameters()).add_(0.037j)
        llr, channel = model(frame.obs)
        bits = frame.bits.double()
        z = payload(frame, llr)
        self.assertGreater(float(z.abs().max()), CAP)
        plain = F.binary_cross_entropy_with_logits(z, bits)
        capped = F.binary_cross_entropy_with_logits(CAP * torch.tanh(z / CAP), bits)
        nmse = (channel - frame.h_true).abs().square().sum() / frame.h_true.abs().square().sum()
        penalty = AUX_NMSE * nmse + ANCHOR / 2 * next(model.head.parameters()).numel() * .037 ** 2
        for cap, expected in ((CAP, capped), (None, plain)):
            loss, raw = objective(frame, model, cap=cap)
            torch.testing.assert_close(loss, expected + penalty, atol=1.e-14, rtol=0.)
            torch.testing.assert_close(raw, plain, atol=0., rtol=0.)
        self.assertTrue(torch.equal(model(frame.obs)[0], llr))
