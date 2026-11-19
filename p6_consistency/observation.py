"""Known receive corruption, kept separate from known transmit pilots.

The archived intervention multiplies the complete received sample (signal
AND noise) by a real gain. Positive known gains can therefore be inverted;
zero gains remove receive equations. This is a privileged diagnostic, not
a blind erasure detector or a physical attenuation/noise simulation.
"""
from types import SimpleNamespace

import torch

from p5.unrolled_em_transformer import _gauge
from p5b.operator_em_transformer import _initialize, _ray_bank
from p6.math_chirp_fisher import reliability


def observation_gain(mask, condition):
    gain = torch.ones_like(mask, dtype=torch.float64)
    rows = torch.arange(mask.shape[0], device=mask.device)[:, None]
    tones = torch.arange(mask.shape[1], device=mask.device)[None] // 2
    if condition == "intact":
        return gain
    if condition in ("first_attenuated", "first_erased"):
        selected = mask & (rows == 0)
    elif condition == "interior_erased":
        selected = mask & (rows == 9)
    elif condition == "scattered_erased":
        selected = mask & ((rows + tones) % 2 == 0)
    else:
        raise ValueError(condition)
    gain[selected] = .4 if condition == "first_attenuated" else 0.
    return gain


def canonicalize(obs, gain=None):
    """Return observable-only copy and a separate shared receive validity mask."""
    if gain is None:
        gain = torch.ones_like(obs.pilot_mask, dtype=torch.float64)
    if gain.shape != obs.pilot_mask.shape or gain.is_complex():
        raise ValueError("This study requires a shared real T,N gain map")
    if bool((~torch.isfinite(gain) | (gain < 0)).any()):
        raise ValueError("Gain must be finite and nonnegative")
    valid = gain > 0
    denominator = torch.where(valid, gain, 1.)
    y = torch.where(valid, obs.y / denominator, 0.)
    copied = SimpleNamespace(y=y, noise_var=obs.noise_var, pilot_mask=obs.pilot_mask,
                             pilots=obs.pilots, regime=obs.regime)
    return copied, valid


def gauge(obs, valid):
    if bool(valid.all()):
        return _gauge(obs, True)
    y = obs.y.to(torch.complex128)
    nv = obs.noise_var.double().reshape(-1)
    known = obs.pilot_mask & valid
    if bool(known.any()):
        samples = y[:, :, known].flatten(1)
        peak = samples.abs().amax(-1)
        active = peak > 0
        safe = torch.where(active, peak, 1.)
        norm = (samples / safe[:, None]).abs().square().mean(-1)
        pilot_rms = safe * torch.where(active, norm, 1.).sqrt()
    else:
        active = torch.zeros(y.shape[0], dtype=torch.bool, device=y.device)
        pilot_rms = y.real.new_ones(y.shape[0])
    field_peak = y.abs().flatten(1).amax(-1)
    fallback = torch.where(nv > 0, torch.where(nv > 0, nv, 1.).sqrt(),
                           torch.where(field_peak > 0, field_peak, 1.))
    scale = torch.where(active, pilot_rms, fallback)
    return y / scale[:, None, None, None], nv / scale.square(), scale


def initialize(obs, y, geometry, valid):
    """Mask-aware version of the original approximate pilot-tap initializer.

    Each row uses its observed pilot subdictionary, and ray-bank matching is
    weighted by the retained fraction of that row's pilot tones. The public
    bank, ambiguity rule and diagonal pilot approximation are retained.
    """
    if bool(valid.all()):
        return _initialize(obs, y, geometry)
    b, r, t, n = y.shape
    nominal = geometry[0]
    rows = obs.pilot_mask.any(-1).nonzero().flatten()
    taps, presence, fractions = [], [], []
    for row in rows:
        mask = obs.pilot_mask[row] & valid[row]
        count = int(mask.sum())
        fractions.append(count / max(int(obs.pilot_mask[row].sum()), 1))
        if count:
            design = nominal[:, mask].mT * obs.pilots[row, mask, None]
            samples = y[:, :, row, mask]
            taps.append(samples @ torch.linalg.pinv(design).mT)
            presence.append(samples.abs().sum((1, 2)) > 0)
        else:
            taps.append(y.new_zeros(b, r, nominal.shape[0]))
            presence.append(torch.zeros(b, dtype=torch.bool, device=y.device))
    taps = torch.stack(taps, 2)
    present = torch.stack(presence, -1)
    row_weight = present * y.real.new_tensor(fractions)
    bank, means, default = _ray_bank(2, t, n)
    sampled = means[:, rows]
    corr = (taps * row_weight[:, None, :, None]).transpose(-1, -2) @ sampled.mH
    energy = row_weight @ sampled.abs().square().T
    score = corr.abs().square().sum(1) / torch.where(energy > 0, energy, 1.)[:, None]
    top = score.amax(-1, keepdim=True)
    error = 64 * torch.finfo(y.real.dtype).eps * top.amax(1, keepdim=True).sqrt()
    tolerance = 2 * top.sqrt() * error + error.square()
    best = ((top - score) <= tolerance).int().argmax(-1)
    best = torch.where(present.sum(-1, keepdim=True) >= 2, best, default)
    rot = bank[best].permute(0, 2, 3, 1)
    mean_rot = means[best].transpose(-1, -2)
    at_rows = mean_rot[:, rows]
    norm = (at_rows.abs().square() * row_weight[..., None]).sum(1)
    amp = (taps * at_rows[:, None].conj() * row_weight[:, None, :, None]).sum(2)
    amp = amp / torch.where(norm > 0, norm, 1.)[:, None]
    return amp, rot, mean_rot, y.new_ones(b, t), present.any(-1)


def pilot_reliability(y, noise, pilot_mask, valid):
    """Retain the nominal aperture as reference when observations are missing."""
    if bool(valid.all()):
        return reliability(y, noise, pilot_mask)
    nominal_count = pilot_mask.sum(-1).to(y.real.dtype)
    actual_mask = (pilot_mask & valid).to(y.real.dtype)
    actual_count = actual_mask.sum(-1)
    time = torch.linspace(-1., 1., pilot_mask.shape[0], dtype=y.real.dtype, device=y.device)
    design = torch.stack((time, time.square()), -1)
    center = (nominal_count @ design) / nominal_count.sum().clamp_min(1.)
    centered = design - center
    full = centered.mT @ (nominal_count[:, None] * centered)
    if int(torch.linalg.matrix_rank(full)) < 2:
        return y.real.new_zeros(y.shape[0])
    factor = torch.linalg.cholesky(full)
    rows = torch.linalg.solve_triangular(factor, centered.mT, upper=False).mT
    power = (y.abs().square().mean(1) * actual_mask).sum(-1) / actual_count.clamp_min(1.)
    nv = noise[:, None]
    usable = torch.where(nv > 0, (power / torch.where(nv > 0, nv, 1.) - 1).clamp(0., 1.),
                         (power > 0).to(y.real.dtype))
    weight = actual_count * usable
    total = weight.sum(-1, keepdim=True)
    mean = (weight @ rows) / torch.where(total > 0, total, 1.)
    local = rows - mean[:, None]
    actual = local.mT @ (weight[..., None] * local)
    return torch.linalg.eigvalsh(actual)[:, 0].clamp(0., 1.)
