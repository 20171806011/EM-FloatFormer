"""Pilot-innovation covariance shrinkage and a precision-weighted R3 equalizer."""
import math

import torch


def precision_correction(y, h, nv, pilot_mask, pilots, transport, strength):
    """Estimate the dimensionless covariance increment K in Sigma = nv (I + K).

    M=transport is the Hermitian PSD contraction from the current receiver arm.
    For spatially white pilot innovations E[C]=nv I, this estimated increment
    has zero expectation for fixed M and gate. Residuals and learned gates can
    be correlated in practice. Bounded shrinkage supplies the eigenvalue floor.
    """
    residual = y[:, :, pilot_mask] - h[:, :, pilot_mask] * pilots[pilot_mask]
    count = residual.shape[-1]
    covariance = (residual @ residual.mH) / count
    eye = torch.eye(h.shape[1], dtype=h.dtype, device=h.device)
    safe_noise = torch.where(nv > 0, nv, 1.)
    difference = covariance / safe_noise[:, None, None] - eye
    normalized_strength = strength.mean(-1) / .18
    gate = .5 * normalized_strength / (normalized_strength + 1 / math.sqrt(count))
    gate = torch.where(nv > 0, gate, 0.)
    correction = gate[:, None, None] * (transport @ difference @ transport)
    return (correction + correction.mH) / 2, gate


def innovation_equalize(y, h, nv, pilot_mask, pilots, transport, strength):
    """Precision-weighted unbiased linear QAM estimate and modeled variance."""
    increment, _ = precision_correction(y, h, nv, pilot_mask, pilots, transport, strength)
    identity = torch.eye(h.shape[1], dtype=h.dtype, device=h.device)
    # Solve for the change from MRC. A zero handle gives literal zero increments
    # and preserves the original power/inner-product reductions bit for bit.
    delta = torch.linalg.solve(identity + increment, -(increment @ h.flatten(2))).reshape_as(h)
    power = h.abs().square().sum(1) + (h.conj() * delta).sum(1).real
    informed = power > 0
    safe_power = torch.where(informed, power, 1.)
    numerator = (h.conj() * y).sum(1) + (delta.conj() * y).sum(1)
    z = numerator / safe_power
    variance = (nv[:, None, None] / safe_power).clamp_min(torch.finfo(torch.float64).eps)
    return z, variance, informed


def apply_in_ray_frame(amp, handle, *, pilot_count):
    transport, strength = handle
    if transport is None:
        return amp
    adjacent = amp[:, 1:] * amp[:, :-1].conj()
    lag = adjacent.sum(1)
    magnitude = lag.abs()
    phasor = torch.where(
        magnitude > 0, lag / torch.where(magnitude > 0, magnitude, 1.0), 1.0
    )
    power = 0.5 * (amp[:, 1:].abs().square() + amp[:, :-1].abs().square()).sum(1)
    coherence = magnitude / torch.where(power > 0, power, 1.0)
    # Integer multiplication avoids complex log(phasor) at the negative real axis.
    rays = torch.stack([phasor**i for i in range(amp.shape[1])], dim=1)
    demod = amp * rays.conj()
    filtered = rays * (transport @ (transport @ demod))
    q = strength / 0.18
    gate = coherence * q / (q + 1 / math.sqrt(pilot_count))
    return amp + gate[:, None] * (filtered - amp)
