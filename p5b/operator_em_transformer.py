"""Gauge-equivariant complex operators inside a two-pass waveform receiver.

Only a P5 Observation crosses the inference boundary. Complex fields at the
operator head live in receive-gauge units. Transmit symbols and the weights of
the radial PA expansion are dimensionless. All arithmetic is CPU float64 or
complex128. Positive LLR means bit one.
"""

import copy
import math
from functools import lru_cache

import torch
from torch import nn

from p4.simulator import decimate_time_to_frequency, oversample_frequency_to_time
from p5.classical import NOMINAL_PA, _time, _trajectory_bank
from p5.unrolled_em_transformer import (
    _equalize, _gauge, _geometry, _llrs, _nominal_gain, _soft_symbols,
)
from p5.wavefield_sim import CP, POWERS
from p5b.parity import soft_parity_symbols


class _ComplexOperatorHead(nn.Module):
    """Read five invariant complex coefficients and one covariant complex field."""

    width = 12

    def __init__(self, variant="pure"):
        super().__init__()
        self.local = variant in ("local", "local_parity")
        self.gauge = variant != "no_gauge"
        self.gamma = nn.Parameter(torch.ones(6, dtype=torch.float64))

        def matrix(n_in, n_out):
            return nn.Parameter(torch.randn(n_in, n_out, dtype=torch.complex128) / math.sqrt(n_in))

        self.project = matrix(6, 3 * self.width)
        self.mix = matrix(2 * self.width, self.width)
        self.read = matrix(self.width, 6)

    def forward(self, features, geom):
        if self.gauge:
            energy = features.abs().square().mean(-1, keepdim=True)
            features = features / torch.where(energy > 0, energy, 1.).sqrt()
        z = features * self.gamma
        q, k, v = (z @ self.project).chunk(3, -1)
        if self.local:
            # Depthwise three-tap convolution along space, time and frequency.
            mixed = sum((a[:, left] + 2 * a + a[:, right]) / 4
                        for a, (left, right) in zip((q, k, v), geom[9])) / math.sqrt(3)
            weights = None
        else:
            rope = geom[8].repeat(1, self.width // 3)
            score = ((q * rope) @ (k * rope).mH).real / math.sqrt(self.width)
            weights = score.softmax(-1)
            mixed = weights.to(v.dtype) @ v
        latent = torch.cat((v, mixed), -1) @ self.mix
        readout = (latent * latent.abs().sigmoid()) @ self.read
        invariant = readout[..., :5] * z[..., :1].conj()
        return torch.cat((invariant, readout[..., 5:]), -1), weights


@lru_cache(maxsize=8)
def _grid_weights(t, n, time_knots, freq_knots):
    """Adjacent knots for symbol interpolation and circular frequency interpolation."""
    times = torch.tensor(time_knots)
    idx = torch.arange(t)
    right = torch.searchsorted(times, idx).clamp(1, len(times) - 1)
    left = right - 1
    wt = (idx - times[left]).double() / (times[right] - times[left])
    tones = torch.tensor(freq_knots)
    idx = torch.arange(n)
    right_f = torch.searchsorted(tones, idx) % len(tones)
    left_f = (right_f - 1) % len(tones)
    wf = ((idx - tones[left_f]) % n).double() / ((tones[right_f] - tones[left_f]) % n)
    return left, right, wt, left_f, right_f, wf


def _synthesize(raw, geom, t, n):
    times, tones = geom[3:5]
    lt, rt, wt, lf, rf, wf = _grid_weights(t, n, tuple(times.tolist()), tuple(tones.tolist()))
    tiles = raw.reshape(raw.shape[0], -1, len(times), len(tones), raw.shape[-1])
    by_time = tiles[:, :, lt] * (1 - wt[:, None, None]) + tiles[:, :, rt] * wt[:, None, None]
    return by_time[:, :, :, lf] * (1 - wf[:, None]) + by_time[:, :, :, rf] * wf[:, None]


@lru_cache(maxsize=4)
def _ray_bank(regime, t, n):
    """Public mean and sample phasors, computed once for all bursts."""
    ts = _time(t, n)
    duration = (t * (n + CP) - 1) / n
    start, end = _trajectory_bank(regime).T[:, :, None, None]
    phase = 2 * math.pi * (ts * start + .5 * ts.square() * (end - start) / duration)
    rotation = torch.polar(torch.ones_like(phase), phase)
    default = int((start.abs() + end.abs()).flatten().argmin())
    return rotation, rotation.mean(-1), default


def _unit(z):
    mag = z.abs()
    return torch.where(mag > 0, z / torch.where(mag > 0, mag, 1.), 1.)


def _initialize(obs, y, geom):
    b, _, t, n = y.shape
    rows = obs.pilot_mask.any(-1).nonzero().flatten()
    pilots = y[:, :, rows, ::2] / obs.pilots[rows, ::2]
    taps = pilots @ geom[1]
    present = pilots.abs().sum((1, 3)) > 0
    cpe = y.new_ones(b, t)
    if obs.regime == 1:
        cross = (taps[:, :, -1] * taps[:, :, 0].conj()).sum((1, 2))
        angle = torch.angle(torch.where(cross.abs() > 0, cross, 1.))
        # Two antipodal endpoint phasors admit either semicircle. Near the
        # negative real branch cut, input roundoff must not choose the path.
        antipodal = (cross.real < 0) & (cross.imag.abs() <= 32 * torch.finfo(torch.float64).eps * cross.abs())
        angle = torch.where(antipodal, math.pi, angle)
        pos = torch.linspace(0., 1., t, dtype=torch.float64)
        cpe = torch.exp(1j * angle[:, None] * pos)
        amp = (taps * cpe[:, None, rows, None].conj()).sum(2)
        amp = amp / present.sum(-1).clamp_min(1)[:, None, None]
        rot = None
        mean_rot = y.new_ones(b, t, 1)
    else:
        bank, means, default = _ray_bank(obs.regime, t, n)
        sampled = means[:, rows]
        corr = taps.transpose(-1, -2) @ sampled.mH
        energy = present.double() @ sampled.abs().square().T
        safe_energy = torch.where(energy > 0, energy, 1.)
        score = corr.abs().square().sum(1) / safe_energy[:, None]
        top = score.amax(-1, keepdim=True)
        # Equal-scoring chirps can predict different payloads. Include backward
        # error of the pilot tap projection: an amplitude error e changes power
        # by at most 2|a|e+e², including when a tap is a numerical null. Break
        # indistinguishable scores in public bank order.
        err_amp = 64 * torch.finfo(torch.float64).eps * top.amax(1, keepdim=True).sqrt()
        tol = 2 * top.sqrt() * err_amp + err_amp.square()
        best = ((top - score) <= tol).int().argmax(-1)
        # A single row cannot identify Doppler when ray gain is unknown.
        best = torch.where(present.sum(-1, keepdim=True) >= 2, best, default)
        rot = bank[best].permute(0, 2, 3, 1)
        mean_rot = means[best].transpose(-1, -2)
        at_rows = mean_rot[:, rows] * present[..., None]
        norm = at_rows.abs().square().sum(1)
        amp = (taps * at_rows[:, None].conj()).sum(2) / torch.where(norm > 0, norm, 1.)[:, None]
    return amp, rot, mean_rot, cpe, present.any(-1)


def _emit(x, regime, radial):
    """Oversampled odd polynomial residual around the public nominal transmit map."""
    u = oversample_frequency_to_time(x, 2)
    p = u.abs().square()
    c3, c5 = radial.unbind(-1)
    # Orthogonalized cubic/quintic bases under the unit-variance complex Gaussian
    # transmit reference. Centering reduces confounding with coherent channel gain.
    correction = .12 * (c3[..., None] * (p - 2) / 2 + c5[..., None] * (p.square() - 6*p + 6) / 6)
    if regime == 2:
        gain = 1.
    else:
        a, knee, alpha = NOMINAL_PA
        gain = (1 + (p / a**2).pow(knee)).pow(-.5 / knee) * torch.exp(1j * alpha * p / (1+p))
        gain = gain / _nominal_gain()
    return decimate_time_to_frequency(u * (gain + correction), x.shape[-1], 2)


def _bound(z):
    """Bound a complex amplitude radially while retaining its full phase."""
    return z / (1 + z.abs())


class OperatorEMTransformer(nn.Module):
    """Two weight-tied EM steps with optional inter-step soft public parity."""

    def __init__(self, variant="pure", steps=2):
        super().__init__()
        if variant not in {"pure", "parity", "local", "local_parity", "no_gauge",
                           "physics", "physics_parity"}:
            raise ValueError(f"unknown operator receiver {variant}")
        if steps not in (2, 3):
            raise ValueError("use two or three waveform steps")
        self.variant = variant
        self.steps = steps
        self.parity = variant in {"parity", "local_parity", "physics_parity"}
        if variant not in {"physics", "physics_parity"}:
            self.head = _ComplexOperatorHead(variant)
            self.initial_map = copy.deepcopy(self.head).requires_grad_(False)

    def _operators(self, h, y, x, residual, anchor, geom):
        b, r, t, n = y.shape
        if not hasattr(self, "head"):
            return y.new_zeros(b, r, t, n, 6), None
        ar, ti, fr = geom[5].T
        ref = x[:, ti, fr].conj()
        ht, yt, err = (a[:, ar, ti, fr] for a in (h, y, residual))
        features = torch.stack((ht, yt * ref, err * ref, anchor[:, ar, ti, fr], err, ht * ref), -1)
        learned, weights = self.head(features, geom)
        # The frozen map remains differentiable with respect to its input.
        initial, _ = self.initial_map(features, geom)
        return _synthesize(learned - initial, geom, t, n), weights

    def forward(self, obs, probe=False):
        y, nv, s0 = _gauge(obs, self.variant != "no_gauge")
        b, r, t, n = y.shape
        geom = _geometry(obs.regime, n, t, r)
        basis = geom[0]
        amp, rot, mean_rot, cpe, has_pilot = _initialize(obs, y, geom)
        anchor = ((amp[:, :, None] * mean_rot[:, None]) @ basis) * cpe[:, None, :, None]
        h, clean = anchor, y
        records = []
        pwr = y[:, :, obs.pilot_mask].abs().square().mean((1, 2))
        # Noise-to-observed-pilot-energy ratio keeps the field LS prior homogeneous.
        ref_pwr = torch.where(pwr > 0, pwr, torch.where(nv > 0, nv, 1.))
        ridge = torch.diag_embed((nv / ref_pwr)[:, None] / nv.new_tensor(POWERS[obs.regime]))
        for step in range(self.steps):
            z, variance, informed = _equalize(clean, h, nv)
            if step and self.parity:
                soft = soft_parity_symbols(z, variance, informed, obs.pilot_mask)
            else:
                soft = torch.where(informed, _soft_symbols(z, variance), 0.)
            x = torch.where(obs.pilot_mask, obs.pilots, soft)
            residual = y - h * x[:, None]
            grid, weights = self._operators(h, y, x, residual, anchor, geom)
            if probe and weights is not None:
                records.append((geom[5], weights))
            coeff = _bound(grid[..., :5])
            symbol = coeff[..., :3].mean((1, 3))
            rotation = torch.exp(.30j * symbol[..., 2].imag)
            log_precision = .45 * coeff[..., 2].real.mean(1)
            ici = .16 * (coeff[..., 3] * residual.roll(1, -1) +
                         coeff[..., 4] * residual.roll(-1, -1))
            delta_h = .10 * _bound(grid[..., 5])
            adjusted = y - ici
            emitted = _emit(x, obs.regime, symbol[..., :2])
            wave = torch.fft.ifft(emitted[..., None] * basis.T, dim=2, norm="ortho")
            if rot is not None:
                wave = wave * rot
            design = (wave * cpe[:, :, None, None]).reshape(b, t*n, -1)
            rx = torch.fft.ifft(adjusted, norm="ortho")
            rhs = design.mH @ rx.reshape(b, r, t*n).mT
            amp = torch.linalg.solve(design.mH @ design + ridge, rhs).mT
            prediction = (amp @ wave.reshape(b, t*n, -1).mT).reshape(b, r, t, n)
            if obs.regime == 1:
                rotation = rotation * _unit((rx * prediction.conj()).sum((1, 3)))
            cpe = rotation
            diagonal = ((amp[:, :, None] * mean_rot[:, None]) @ basis) * cpe[:, None, :, None]
            h = diagonal + delta_h
            received = torch.fft.fft(prediction * cpe[:, None, :, None], norm="ortho")
            clean = adjusted - (received - diagonal * x[:, None])
        z, variance, informed = _equalize(clean, h, nv)
        llr = _llrs(z, variance * (-log_precision).exp())
        llr = torch.where((informed & has_pilot[:, None, None])[..., None], llr, 0.)
        physical_h = torch.where(has_pilot[:, None, None, None], h * s0[:, None, None, None], 0.)
        if obs.regime != 2:
            physical_h = physical_h / _nominal_gain()
        return (llr, physical_h, records) if probe else (llr, physical_h)
