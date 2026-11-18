"""Coarse R2 bank followed by a conditioned phase or continuous chirp update."""
from functools import lru_cache

import torch

from p5b.operator_em_transformer import _bound
from p6.axial_em_transformer import _bank_templates, _centered_angles
from p6.axial_em_transformer import _soft_chirp_residual as coarse_bank


@lru_cache(maxsize=1)
def _chirp_basis(t):
    clock = torch.linspace(-1., 1., t, dtype=torch.float64)
    quadratic = clock.square()
    columns = torch.stack((clock, quadratic - quadratic.mean()), -1)
    return columns / columns.norm(dim=0)


def _soft_taps(clean, x, basis, field_ray, cue_weight):
    n = clean.shape[-1]
    design = x[..., None] * basis.mT
    normal = design.mH @ design
    ridge = .02 * n * torch.eye(normal.shape[-1], dtype=normal.dtype, device=normal.device)
    rhs = design.mH @ clean.permute(0, 2, 3, 1)
    taps = torch.linalg.solve(normal + ridge, rhs).permute(0, 3, 2, 1)
    if field_ray is not None and cue_weight:
        taps = taps + cue_weight * _bound((field_ray @ basis.mH).mT / n)
    return taps


def _bank_geometry(taps, rot):
    t, n = rot.shape[1:3]
    means, angles = _bank_templates(t, n)
    corr = (taps @ means.mH).abs().square().sum(1)
    norm = taps.abs().square().sum((1, 3))[..., None] * means.abs().square().sum(-1)
    coherence = corr / (.001 + norm)
    return coherence, angles


def _bank(taps, rot, beta):
    coherence, angles = _bank_geometry(taps, rot)
    weights = (beta * coherence).softmax(-1)
    return weights @ angles - _centered_angles(rot[:, :, rot.shape[2] // 2].mT)


def _tangent(taps, rot):
    """Variable-projection Gauss–Newton step in two centered chirp coordinates."""
    return _tangent_at(taps, rot[:, :, rot.shape[2] // 2].mT, 1.57)


def _tangent_at(taps, phase, radius):
    dephased = taps * phase[:, None].conj()
    amplitude = dephased.mean(-1, keepdim=True)
    score = (dephased * amplitude.conj()).imag.sum(1)
    # The constant phase is absorbed by a free complex amplitude. The same
    # scalar Fisher information multiplies the two orthonormal columns.
    information = amplitude.abs().square().sum(1)
    ridge = .001 + .02 * taps.abs().square().mean(-1, keepdim=True).sum(1)
    basis = _chirp_basis(taps.shape[-1])
    correction = ((score / (information + ridge)) @ basis) @ basis.mT
    # Scale the entire polynomial so its peak remains in the local phase chart.
    return correction / (1 + (correction.abs().amax(-1, keepdim=True) / radius).square()).sqrt()


def _conditional(taps, rot, refine):
    coherence, angles = _bank_geometry(taps, rot)
    logits = 160. * coherence
    increasing = angles[:, -1] > angles[:, 0]
    positive = logits[..., increasing].logsumexp(-1)
    negative = logits[..., ~increasing].logsumexp(-1)
    support = increasing == (positive >= negative)[..., None]
    weights = logits.masked_fill(~support, -torch.inf).softmax(-1)
    estimate = weights @ angles
    if refine:
        # Half a cell in the public Cartesian 9x9 endpoint-frequency grid.
        radius = .5 * ((angles[1]-angles[0]).abs().max() + (angles[9]-angles[0]).abs().max())
        estimate = estimate + _tangent_at(taps, (1j * estimate).exp(), radius)
    return estimate - _centered_angles(rot[:, :, rot.shape[2] // 2].mT)


def residual(clean, x, basis, rot, field_ray, cue_weight, step, mode, pilot_reliability=None):
    """Residual at the ordinary solver bank hook; no parameters or simulator truth."""
    if mode == "fisher_trust":
        from p6.math_chirp_trust import residual as trust_update
        return trust_update(clean, x, basis, rot, field_ray, cue_weight, step, pilot_reliability)
    if step == 1:
        return coarse_bank(clean, x, basis, rot, field_ray, cue_weight)
    if step != 2:
        raise ValueError("The second chirp update requires three EM passes")
    taps = _soft_taps(clean, x, basis, field_ray, cue_weight)
    if mode == "tempered":
        return _bank(taps, rot, 40.).mT
    if mode == "fisher":
        if pilot_reliability is None:
            raise ValueError("Fisher conditioning requires the observed pilot reliability")
        return _bank(taps, rot, (40. + 120. * pilot_reliability)[:, None, None]).mT
    if mode in ("conditional", "conditional_gn"):
        return _conditional(taps, rot, mode == "conditional_gn").mT
    tangent = _tangent(taps, rot)
    if mode == "tangent":
        return tangent.mT
    if mode == "hybrid":
        return (.25 * _bank(taps, rot, 160.) + .75 * tangent).mT
    raise ValueError(f"Unknown R2 update {mode}")
