"""Shared Iteration-3 objective. Inference and validation use the raw receiver logits."""

import torch.nn.functional as F

from p5.study import payload


CAP = 10.0
AUX_NMSE = 0.04
ANCHOR = 1.e-4


def objective(frame, model, *, cap=CAP, aux_nmse=AUX_NMSE):
    llrs, channel = model(frame.obs)
    logits = payload(frame, llrs)
    bits = frame.bits.double()
    raw_bce = F.binary_cross_entropy_with_logits(logits, bits)
    if cap is None:
        training_bce = raw_bce
    else:
        training_bce = F.binary_cross_entropy_with_logits(cap * (logits / cap).tanh(), bits)
    nmse = (channel - frame.h_true).abs().square().sum() / frame.h_true.abs().square().sum()
    displacement = sum((p - p0).abs().square().sum() for p, p0 in
                       zip(model.head.parameters(), model.initial_map.parameters(), strict=True))
    return training_bce + aux_nmse * nmse + (ANCHOR / 2) * displacement, raw_bce
