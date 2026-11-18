"""Blend coarse global and local continuous innovations by observed pilot information."""
from p6 import math_chirp as chirp


def residual(clean, x, basis, rot, field_ray, cue_weight, step, pilot_reliability):
    if step == 1:
        return chirp.coarse_bank(clean, x, basis, rot, field_ray, cue_weight)
    if step != 2 or pilot_reliability is None:
        raise ValueError("The trust update requires third-pass pilot information")
    taps = chirp._soft_taps(clean, x, basis, field_ray, cue_weight)
    information = pilot_reliability[:, None, None]
    bank = chirp._bank(taps, rot, 40. + 120. * information)
    tangent = chirp._tangent(taps, rot)
    return (information * bank + (1. - information) * tangent).mT
