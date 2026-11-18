"""Pilot-referenced nonlinear emission and centered sample-clock R1 phase bridge."""
import math

import torch

from p4.simulator import decimate_time_to_frequency, oversample_frequency_to_time
from p5.wavefield_sim import CP


INNOVATION_VARIANCE = (.012**2 + .012 * .032 + .032**2) / 3.


def _confidence(learned_row_phase):
    signal = learned_row_phase.square().mean(-1, keepdim=True)
    return signal / (signal + INNOVATION_VARIANCE)


def bridge(cpe, learned_row_phase, n):
    """Conditional Brownian mean at sample times given row-midpoint phase estimates."""
    edges = cpe[:, 1:] * cpe[:, :-1].conj()
    # Half-turn differences have two equally short lifts. Pick one deterministically.
    tied = (edges.real < 0) & (edges.imag.abs() <= 32 * torch.finfo(edges.real.dtype).eps * edges.abs())
    differences = torch.where(tied, math.pi, edges.angle())
    flat = torch.zeros_like(differences[:, :1])
    left = torch.cat((flat, differences), dim=1)
    right = torch.cat((differences, flat), dim=1)
    fraction = (torch.arange(n, device=cpe.device, dtype=cpe.real.dtype) - (n - 1)/2) / (n + CP)
    excursion = torch.where(fraction < 0, left[..., None], right[..., None]) * fraction
    # Prior variance of one Brownian increment; sigma is uniform on [.012,.032].
    return torch.exp(1j * _confidence(learned_row_phase)[..., None] * excursion)


def tangent_bridge(sample_bridge, cpe, learned_row_phase, wave, amp, received, nv):
    """Take one posterior phase-slope tangent step, eliminating unknown row phase."""
    n = wave.shape[2]
    tau = (torch.arange(n, device=cpe.device, dtype=cpe.real.dtype) - (n - 1)/2) / (n + CP)
    prediction = torch.einsum("btnp,brp->brtn", wave, amp) * (sample_bridge * cpe[..., None])[:, None]
    power = prediction.abs().square().sum(1)
    power_row = power.sum(-1, keepdim=True)
    offset = tau - (power * tau).sum(-1, keepdim=True) / torch.where(power_row > 0, power_row, 1.)
    score = (offset * (prediction.conj() * received).imag.sum(1)).sum(-1)
    information = (offset.square() * power).sum(-1)
    prior_variance = (n + CP) * INNOVATION_VARIANCE
    precision = information + nv[:, None] / (2 * prior_variance)
    slope = score / torch.where(precision > 0, precision, 1.)
    # A linearization is local; cap the correction at three prior standard deviations.
    trust = 3 * math.sqrt(prior_variance)
    step = trust * torch.tanh(slope / trust) * _confidence(learned_row_phase)
    return sample_bridge * torch.exp(1j * step[..., None] * tau)


def differential_emission(x, radial, pilot_mask):
    """Restore the differential coherent gain removed by row-wise orthogonalization."""
    from p6.axial_em_transformer import _nominal_gain, _nominal_pa, _pa_residual_gain

    u = oversample_frequency_to_time(x, 2)
    gain = _nominal_pa(u.abs().square()) / _nominal_gain()
    residual = u * gain * (_pa_residual_gain(u, radial) - 1)
    energy = u.abs().square().sum(-1)
    row_gain = (u.conj() * residual).sum(-1) / torch.where(energy > 0, energy, 1.)
    grid_residual = decimate_time_to_frequency(residual, x.shape[-1], 2)
    pilots = x[:, pilot_mask]
    pilot_energy = pilots.abs().square().sum(-1, keepdim=True)
    pilot_gain = (pilots.conj() * grid_residual[:, pilot_mask]).sum(-1, keepdim=True)
    pilot_gain = pilot_gain / torch.where(pilot_energy > 0, pilot_energy, 1.)
    return x * (row_gain - pilot_gain)[..., None]
