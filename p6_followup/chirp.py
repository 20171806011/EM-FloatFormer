"""A C1 sign-chart blend for the final, weak-information R2 chirp update.

This module is a follow-up to the frozen P6 implementation.  It adds no
parameters, observed inputs, temperatures, or simulator-truth dependencies.
Only the weak-information term at the final R2 pass changes.  The C1 claim
concerns the sign-chart switch, not every operation in the complete receiver.
"""

import torch

from p6 import math_chirp as chirp
from p6.axial_em_transformer import _centered_angles


def chart_coefficients(positive_logits, negative_logits):
    """Return [..., 3] weights in tangent, positive, negative order.

    Each group must contain at least one finite logit.  The existing public
    R2 bank supplies equally sized sign groups.  Squared ReLU makes the
    selected chart's coefficient and its first derivative vanish at a tie.
    These are score-confidence weights, not calibrated posterior confidence.
    """
    difference = positive_logits.logsumexp(-1) - negative_logits.logsumexp(-1)
    signed_mass = torch.tanh(0.5 * difference)
    positive = signed_mass.clamp_min(0).square()
    negative = (-signed_mass).clamp_min(0).square()
    tangent = 1.0 - signed_mass.square()
    return torch.stack((tangent, positive, negative), dim=-1)


def confidence_chart_mix(positive_logits, negative_logits, positive_angles,
                         negative_angles, current, tangent):
    """Blend conditional phase innovations with the local tangent.

    Logits have shape [..., J_sign], angle tables [J_sign, T], and current
    and tangent [..., T].  For finite logits this helper is C1 across equal
    positive/negative group masses, conditional on its other inputs.
    """
    coefficients = chart_coefficients(positive_logits, negative_logits)
    positive_mean = positive_logits.softmax(-1) @ positive_angles
    negative_mean = negative_logits.softmax(-1) @ negative_angles
    return (coefficients[..., 0, None] * tangent
            + coefficients[..., 1, None] * (positive_mean - current)
            + coefficients[..., 2, None] * (negative_mean - current))


def residual(clean, x, basis, rot, field_ray, cue_weight, step, pilot_reliability):
    """Drop-in signature for ``p6.math_chirp_trust.residual``.

    The first bank call and q=1 final-pass rule are unchanged.  Global
    complex-gain compatibility relies on the complete receiver's existing
    gauge normalization and the corresponding transformation of noise.
    """
    if step == 1:
        return chirp.coarse_bank(clean, x, basis, rot, field_ray, cue_weight)
    if step != 2 or pilot_reliability is None:
        raise ValueError("The chart update requires third-pass pilot information")

    taps = chirp._soft_taps(clean, x, basis, field_ray, cue_weight)
    coherence, angles = chirp._bank_geometry(taps, rot)
    current = _centered_angles(rot[:, :, rot.shape[2] // 2].mT)
    information = pilot_reliability[:, None, None]

    # Preserve the existing bank formula and operation order.  Sharing its
    # coherence calculation avoids another template correlation product.
    bank = ((40.0 + 120.0 * information) * coherence).softmax(-1) @ angles - current
    tangent = chirp._tangent(taps, rot)
    increasing = angles[:, -1] > angles[:, 0]
    logits = 160.0 * coherence
    weak = confidence_chart_mix(logits[..., increasing], logits[..., ~increasing],
                                angles[increasing], angles[~increasing],
                                current, tangent)
    return (information * bank + (1.0 - information) * weak).mT
