"""Physics-anchored, weight-tied EM receiver. All receiver inputs are observable.

The network learns step sizes in a projected soft decision-directed solver and
the precision of the final demapper. It cannot write an arbitrary channel.
"""

import copy
import math
from functools import lru_cache

import torch
from torch import nn

from p4.em_floatformer import ComplexRMSNorm, RadialActivation
from p5.classical import NOMINAL_PA
from p5.wavefield_sim import DELAYS, pa_grid


@lru_cache(maxsize=16)
def _geometry(regime, n, t, r):
    """CPU-only public pilot geometry, delay dictionary and coarse token grid."""
    delays = DELAYS[regime]
    freq = torch.fft.fftfreq(n, dtype=torch.float64)
    basis = torch.exp(-2j * math.pi * freq[:, None] * torch.tensor(delays, dtype=torch.float64))
    pilot_inverse = torch.linalg.pinv(basis[::2]).T.contiguous()
    full_inverse = torch.linalg.pinv(basis).T.contiguous()
    if regime == 2:
        times = torch.tensor([0, 3, 6, 9, 10, 13, 16, 19])
    else:
        times = torch.arange(0, t, 2)
    tones = torch.arange(2, n, 8)
    coords = torch.cartesian_prod(torch.arange(r), times, tones)
    t_index = (torch.arange(t)[:, None] - times).abs().argmin(-1)
    f_index = (torch.arange(n)[:, None] - tones).abs().argmin(-1)
    # A separate rotary phase coordinate for array element, symbol and tone.
    phase = coords.double() * torch.tensor([0.71, 0.39, 0.17], dtype=torch.float64)
    rope = torch.polar(torch.ones_like(phase), phase)
    # Fixed local three-site filters on each physical axis for the convolution control.
    tile = torch.arange(len(coords)).reshape(r, len(times), len(tones))
    neighbors = []
    for axis, size in enumerate(tile.shape):
        prev = (torch.arange(size) - 1).clamp_min(0)
        nxt = (torch.arange(size) + 1).clamp_max(size - 1)
        neighbors.append((tile.index_select(axis, prev).reshape(-1),
                          tile.index_select(axis, nxt).reshape(-1)))
    return basis.T.contiguous(), pilot_inverse, full_inverse, times, tones, coords, t_index, f_index, rope, neighbors


@lru_cache(maxsize=8)
def _chirps(t):
    """Coherent alias hypotheses spanning the *public* signed trajectory support."""
    speeds = torch.linspace(0.045, 0.080, 7, dtype=torch.float64)
    pairs = torch.cartesian_prod(speeds, speeds)
    endpoints = torch.cat((pairs, -pairs))
    # The CP is 16 samples for N=32. Absolute phase has an unknown complex gain.
    duration = 1.5 * t - 1 / 32
    time = 1.5 * torch.arange(t, dtype=torch.float64) + .5 + 15.5 / 32
    phase = 2 * math.pi * (endpoints[:, :1] * time +
                          (endpoints[:, 1:] - endpoints[:, :1]) * time.square() / (2 * duration))
    return torch.polar(torch.ones_like(phase), phase)


def _fourier_anchor(obs, y, geom):
    basis, pinv = geom[:2]
    rows = torch.where(obs.pilot_mask.any(-1))[0]
    pilot_ls = y[:, :, rows, ::2] / obs.pilots[rows, ::2]
    taps = pilot_ls @ pinv
    if obs.regime == 2:
        dictionary = _chirps(y.shape[2])
        correlations = taps.transpose(-2, -1) @ dictionary[:, rows].conj().T
        best = correlations.abs().argmax(-1)
        gain = correlations.gather(-1, best[..., None]) / len(rows)
        interpolated = (gain * dictionary[best]).transpose(-2, -1)
    else:
        position = (torch.arange(y.shape[2], dtype=torch.float64) / (y.shape[2] - 1))[None, None, :, None]
        first, last = taps[:, :, :1], taps[:, :, -1:]
        cross = last * first.conj()
        angle = torch.angle(cross)
        # Choose the same semicircle when roundoff puts an antipodal pair on
        # opposite sides of the angle branch cut under a common phase rotation.
        antipodal = (cross.real < 0) & (cross.imag.abs() <= 32 * torch.finfo(torch.float64).eps * cross.abs())
        angle = torch.where(antipodal, math.pi, angle)
        # atan2 distinguishes signed zero. A missing endpoint provides no turn.
        angle = torch.where((first.abs() > 0) & (last.abs() > 0), angle, 0.)
        amplitude = (1 - position) * first.abs() + position * last.abs()
        reference = torch.where(first.abs() > 0, first, last)
        interpolated = torch.polar(amplitude, torch.angle(reference) + position * angle)
    return interpolated @ basis


def fourier_anchor(obs):
    """Physical CFR in observation units, preserving every public support basis vector."""
    _, r, t, n = obs.y.shape
    return _fourier_anchor(obs, obs.y.to(torch.complex128), _geometry(obs.regime, n, t, r))


def _gauge(obs, enabled):
    y = obs.y.to(torch.complex128)
    nv = obs.noise_var.double().reshape(-1)
    if enabled:
        pilots = y[:, :, obs.pilot_mask].flatten(1)
        peak = pilots.abs().amax(-1)
        active = peak > 0
        peak_safe = torch.where(active, peak, 1.0)
        norm = (pilots / peak_safe[:, None]).abs().square().mean(-1)
        pilot_rms = peak_safe * torch.where(active, norm, 1.0).sqrt()
        field_peak = y.abs().flatten(1).amax(-1)
        zero_noise_fallback = torch.where(field_peak > 0, field_peak, 1.0)
        noise_std = torch.where(nv > 0, nv, 1.0).sqrt()
        fallback = torch.where(nv > 0, noise_std, zero_noise_fallback)
        s0 = torch.where(active, pilot_rms, fallback)
    else:
        s0 = torch.ones(y.shape[0], dtype=torch.float64, device=y.device)
    return y / s0[:, None, None, None], nv / s0.square(), s0


def _soft_symbols(z, variance):
    # Exact, separable four-point posterior on each axis. Pilot values are inserted by caller.
    points = z.real.new_tensor([-3., -1., 1., 3.]) / math.sqrt(10)
    axes = torch.view_as_real(z).unsqueeze(-1)
    logits = -(axes - points).square() / variance[..., None, None]
    mean = (torch.softmax(logits, -1) * points).sum(-1)
    return torch.complex(mean[..., 0], mean[..., 1])


def _llrs(z, variance):
    # Max-log Gray 16-QAM, positive for bit one. No logit clipping or hard floor on error rates.
    boundary = 2 / math.sqrt(10)
    axes = torch.view_as_real(z)
    magnitude = axes.abs()
    sign = axes + axes.sign() * (magnitude - boundary).relu()
    inner = boundary - magnitude
    pair = torch.stack((sign, inner), -1)
    return (pair * (4 / math.sqrt(10) / variance)[..., None, None]).flatten(-2)


def _equalize(y, h, nv):
    power = h.abs().square().sum(1)
    informed = power > 0
    safe_power = torch.where(informed, power, 1.0)
    z = (h.conj() * y).sum(1) / safe_power
    variance = nv[:, None, None] / safe_power
    # Physical zero noise would define an infinite LLR, outside the finite-noise study.
    variance = variance.clamp_min(torch.finfo(torch.float64).eps)
    return z, variance, informed


def _nominal_distortion(x, regime):
    if regime == 2:
        return torch.zeros_like(x)
    # Rapp p=2; shared nominal PA with public receiver coefficients, no burst metadata.
    # Normalize the nominal PA by its complex Gaussian Bussgang gain.
    return pa_grid(x, *NOMINAL_PA) / _nominal_gain() - x


def _doppler_ici(h, x):
    """First-order within-symbol CFR variation, from neighboring estimated CFRs."""
    slope = torch.cat((h[:, :, 1:2] - h[:, :, :1],
                       (h[:, :, 2:] - h[:, :, :-2]) / 2,
                       h[:, :, -1:] - h[:, :, -2:-1]), dim=2)
    n = h.shape[-1]
    offset = (torch.arange(n, dtype=torch.float64) - (n - 1) / 2) / (1.5 * n)
    return torch.fft.fft(offset * torch.fft.ifft(slope * x[:, None], dim=-1), dim=-1)


@lru_cache(maxsize=1)
def _nominal_gain():
    # Midpoint quadrature of E[|u|^2 f(u)/u], with |u|^2 exponentially distributed.
    pwr = (torch.arange(2048, dtype=torch.float64) + .5) / 64
    a, _, alpha = NOMINAL_PA
    gain = (1 + (pwr / a**2).square()).pow(-0.25) * torch.exp(1j * alpha * pwr / (1 + pwr))
    return complex((pwr * (-pwr).exp() * gain).sum() / 64)


class _Propagation(nn.Module):
    """One three-coordinate Hermitian head, or a matched local separable convolution."""

    def __init__(self, variant):
        super().__init__()
        self.variant = variant
        self.norm = ComplexRMSNorm(4).double()
        self.radial = RadialActivation("silu")
        self.project = nn.Parameter(torch.randn(4, 9, dtype=torch.complex128) * 0.45)
        self.read = nn.Parameter(torch.randn(3, 3, dtype=torch.complex128) * 0.4)

    def forward(self, features, geom):
        normalized = (features * self.norm.gamma if self.variant == "no_gauge"
                      else self.norm(features))
        q, k, v = (normalized @ self.project).chunk(3, -1)
        if self.variant == "local":
            q_pos, k_pos, v_pos = geom[9]
            mixed = sum((a[:, prev] + 2 * a + a[:, nxt]) / 4
                        for a, (prev, nxt) in zip((q, k, v), (q_pos, k_pos, v_pos))) / math.sqrt(3)
            weights = None
        else:
            q, k = q * geom[8], k * geom[8]
            weights = torch.softmax((q @ k.conj().transpose(-2, -1)).real / math.sqrt(3), dim=-1)
            mixed = weights.to(v.dtype) @ v
        # Inner product with the local phase reference yields invariant real solver coefficients.
        control = (self.radial(mixed) @ self.read) * normalized[..., :1].conj()
        return control.real, weights


class UnrolledEMTransformer(nn.Module):
    """Three solver controls are shared across unrolling iterations.

    Returns full-grid positive-one LLRs [B,T,F,4], estimated pre-PA CFR [B,R,T,F],
    and optionally a list of (antenna/time/frequency coordinates, attention mass).
    """

    def __init__(self, variant="full", steps=3):
        super().__init__()
        if variant not in ("full", "local", "no_gauge", "no_unroll", "physics"):
            raise ValueError(f"unknown variant {variant}")
        if steps not in (3, 4):
            raise ValueError("steps must be 3 or 4")
        self.variant = variant
        self.steps = 1 if variant == "no_unroll" else steps
        if variant != "physics":
            self.propagation = _Propagation(variant)
            self.initial_map = copy.deepcopy(self.propagation).requires_grad_(False)

    def forward(self, obs, probe=False):
        y, nv, s0 = _gauge(obs, self.variant != "no_gauge")
        b, r, t, n = y.shape
        geom = _geometry(obs.regime, n, t, r)
        basis, _, support_inverse, times, tones, coords, t_index, f_index = geom[:8]
        anchor = _fourier_anchor(obs, y, geom)
        h = anchor
        clean = y
        records = []
        for _ in range(self.steps):
            z, variance, informed = _equalize(clean, h, nv)
            # The likelihood is constant across the alphabet at a zero channel.
            # Its uniform-prior mean is exactly zero, including in feedback.
            soft = torch.where(informed, _soft_symbols(z, variance), 0.)
            x = torch.where(obs.pilot_mask, obs.pilots, soft)
            distortion = _nominal_distortion(x, obs.regime)
            predicted_tx = x + distortion
            ici = _doppler_ici(h, predicted_tx)
            residual = y - h * predicted_tx[:, None] - ici
            if self.variant == "physics":
                h_step, pa_step, log_precision = .35, .8, 0.
            else:
                a, j, k = coords.T
                x_tokens = x[:, j, k].conj()
                feat = torch.stack((h[:, a, j, k], y[:, a, j, k] * x_tokens,
                                    residual[:, a, j, k] * x_tokens, anchor[:, a, j, k]), -1)
                learned, weights = self.propagation(feat, geom)
                # A frozen copy defines an honest function difference. Its input Jacobian
                # participates in autograd, including after the first optimizer step.
                reference, _ = self.initial_map(feat, geom)
                delta = (learned - reference).reshape(b, r, len(times), len(tones), 3)
                delta = delta[:, :, t_index][:, :, :, f_index]
                d_h, d_pa, d_precision = delta.unbind(-1)
                h_step = .35 * (1 + d_h.tanh())
                pa_step = .8 * (1 + d_pa.tanh())
                log_precision = d_precision.mean(1).tanh()
                if probe and weights is not None:
                    records.append((coords, weights))
            # Project the preconditioned adjoint residual back onto the delay support.
            norm = predicted_tx.abs().square().mean(-1, keepdim=True) + .1
            update = h_step * (residual * (predicted_tx.conj() / norm)[:, None])
            h = h + (update @ support_inverse) @ basis
            clean = y - (h * distortion[:, None] + ici) * pa_step
        z, variance, informed = _equalize(clean, h, nv)
        if isinstance(log_precision, torch.Tensor):
            variance = variance * (-log_precision).exp()
        # The pilot estimate includes the PA's coherent gain. De-embed only the
        # public nominal value when scoring against the simulator's pre-PA target.
        physical_h = h * s0[:, None, None, None]
        if obs.regime != 2:
            physical_h = physical_h / _nominal_gain()
        logits = torch.where(informed[..., None], _llrs(z, variance), 0.)
        output = (logits, physical_h)
        return (*output, records) if probe else output
