"""Full-grid waveform EM receiver with a centered, weight-tied axial operator."""

import copy
import math
from functools import lru_cache
from typing import NamedTuple

import torch
from torch import nn

from p4.simulator import decimate_time_to_frequency, oversample_frequency_to_time
from p5.classical import NOMINAL_PA
from p5.unrolled_em_transformer import _equalize, _gauge, _llrs, _nominal_gain, _soft_symbols
from p5.wavefield_sim import CP, DELAYS, POWERS
from p5b.operator_em_transformer import _bound, _initialize, _ray_bank, _unit
from p6.axial_operator import AxialOperatorBlock, full_frame_features
from p6.math_spatial import apply_in_ray_frame, innovation_equalize
from p6.parity import soft_parity_symbols


@lru_cache(maxsize=3)
def _dictionary(regime, n):
    delays = torch.tensor(DELAYS[regime], dtype=torch.float64)
    freq = torch.fft.fftfreq(n, dtype=torch.float64)
    phase = -2j * math.pi * freq[:, None] * delays
    columns = phase.exp()
    return columns.T.contiguous(), torch.linalg.pinv(columns[::2]).T.contiguous()


class Handles(NamedTuple):
    radial: torch.Tensor    # B,T,2 complex forward PA residual coefficients
    cpe: torch.Tensor       # B,T common phase increment
    phase: torch.Tensor     # B,T,N,P residual sample-time trajectory increment
    delays: torch.Tensor    # B,P offset from nominal delay in Nyquist samples
    spatial: torch.Tensor   # B,P real convex shrinkage strength on each ray
    delta_h: torch.Tensor   # B,R,T,N general learned CFR residual


def _sample_phase(row_phase, n):
    """Lift per-symbol phase to sample time by local, centered linear interpolation."""
    center = (torch.arange(n, dtype=row_phase.dtype) - (n - 1) / 2) / (n + CP)
    difference = row_phase[:, 1:] - row_phase[:, :-1]
    slope = torch.cat((difference[:, :1], (difference[:, 1:] + difference[:, :-1]) / 2,
                       difference[:, -1:]), dim=1)
    return row_phase[:, :, None] + center[None, None, :, None] * slope[:, :, None]


def _radial_read(pa, x):
    """Fit linear burst drift to two orthogonal, zero-DC constellation contrasts."""
    t = x.shape[1]
    power = x.abs().square()
    first = power - power.mean(-1, keepdim=True)
    second = (power - 1).square()
    second = second - second.mean(-1, keepdim=True)
    variance = first.square().mean(-1, keepdim=True)
    cross = (second * first).mean(-1, keepdim=True)
    second = second - first * cross / torch.where(variance > 0, variance, 1.)
    contrasts = torch.stack((first, second), -1)
    contrasts = contrasts / contrasts.square().mean(-2, keepdim=True).clamp_min(.04).sqrt()
    field = pa.mean(1)
    field = field - field.mean(-1, keepdim=True)
    row = (field[..., None] * contrasts).mean(-2)
    progress = torch.linspace(0., 1., t, dtype=x.real.dtype, device=x.device)[None, :, None]
    centered = progress - .5
    slope = (centered * row).mean(1, keepdim=True) / centered.square().mean().clamp_min(1.e-12)
    midpoint = row.mean(1, keepdim=True)
    endpoints = torch.cat((midpoint - .5 * slope, midpoint + .5 * slope), dim=1)
    endpoints = 2 * torch.complex(endpoints.real.tanh(), endpoints.imag.tanh())
    return endpoints[:, :1] + progress * (endpoints[:, 1:] - endpoints[:, :1])


def _centered_angles(phasors):
    """Remove the constant phase absorbed by the fitted complex gain."""
    increments = (phasors[..., 1:] * phasors[..., :-1].conj()).angle()
    angles = torch.cat((torch.zeros_like(phasors[..., :1].real), increments), -1).cumsum(-1)
    return angles - angles.mean(-1, keepdim=True)


@lru_cache(maxsize=1)
def _bank_templates(t, n):
    rotations, means, _ = _ray_bank(2, t, n)
    return means, _centered_angles(rotations[:, :, n // 2])


def _soft_chirp_residual(clean, x, basis, rot, field_ray=None, cue_weight=.05):
    """Score the public R2 chirps with soft-symbol LS taps and a learned path-time residual."""
    _, _, t, n = clean.shape
    design = x[..., None] * basis.mT
    normal = design.mH @ design
    ridge = .02 * n * torch.eye(normal.shape[-1], dtype=normal.dtype)
    rhs = design.mH @ clean.permute(0, 2, 3, 1)
    taps = torch.linalg.solve(normal + ridge, rhs).permute(0, 3, 2, 1)
    if field_ray is not None and cue_weight:
        cue = (field_ray @ basis.mH).mT / n
        taps = taps + cue_weight * _bound(cue)
    means, angles = _bank_templates(t, n)
    corr = (taps @ means.mH).abs().square().sum(1)
    norm = taps.abs().square().sum((1, 3))[..., None] * means.abs().square().sum(-1)
    coherence = corr / (.001 + norm)
    weights = (160 * coherence).softmax(-1)
    current = _centered_angles(rot[:, :, n // 2].mT)
    target = weights @ angles
    return (target - current).mT


def _apply_spatial(amp, handle):
    """Filter the current, possibly newly solved ray amplitudes through the antenna map."""
    if len(handle) == 3:
        return apply_in_ray_frame(amp, handle[:2], pilot_count=handle[2])
    operator, strength = handle
    if operator is None:
        return amp
    return amp + strength[:, None] * (operator @ amp - amp)


def _physical_handles(fields, h, amp, mean_rot, cpe, nominal_basis, regime, x, bank_residual=None):
    """Read observable, charge-neutral PA, CPE, path phase and delay corrections."""
    b, r, t, n = h.shape
    field_h, field_ray, field_pa, field_space = fields.unbind(-1)
    reference = h.conj() / (.10 + h.abs().square())
    pa = field_pa * reference
    ray = field_ray * reference
    radial = _radial_read(pa, x)
    cpe_increment = (.7 * (pa[..., n // 4].imag + .35 * ray[..., n // 4].real).mean(1).tanh()
                     if regime == 1 else h.real.new_zeros(b, t))

    # Project to the public P-dimensional delay dictionary independently at
    # each symbol. The complex reference removes each ray's current angle and
    # scales the update in radians.
    ray_state = amp[:, :, None] * mean_rot[:, None] * cpe[:, None, :, None]
    ray_energy = ray_state.abs().square().mean(1)
    if regime == 2:
        projection = ((field_ray + .35 * field_pa) @ nominal_basis.mH) / n
        phase_score = (projection * ray_state.conj()).imag.mean(1) / (.04 + ray_energy)
        row_phase = 3.2 * torch.asinh(phase_score)
        if bank_residual is not None:
            # Centered fields give an exactly zero bank update at initialization.
            # A learned signed readout opens the gate at first order. As the PA
            # field departs from zero, repeated passes can commit to one bank.
            signed = (4 * pa.real.mean((1, 2, 3))).tanh()
            energy = pa.abs().square().mean((1, 2, 3))
            gate = 1 - (-24 * energy).exp() * (1 - signed)
            row_phase = row_phase + gate[:, None, None] * bank_residual
        phase = _sample_phase(row_phase, n)
    else:
        phase = h.real.new_zeros(b, t, n, len(DELAYS[regime]))

    if regime == 3:
        freq = torch.fft.fftfreq(n, dtype=h.real.dtype)
        derivative = nominal_basis * (-2j * math.pi * freq)
        delay_projection = (field_ray @ derivative.mH) / n
        numerator = (delay_projection * ray_state.conj()).real.mean((1, 2))
        delay_score = numerator / (.04 + ray_energy.mean(1))
        movable = torch.tensor(DELAYS[regime], dtype=h.real.dtype) != 0
        delays = .30 * (2.5 * delay_score).tanh() * movable
        antenna_taps = (field_space @ nominal_basis.mH) / n
        alignment = (antenna_taps * ray_state.conj()).mean((1, 2))
        spatial = .18 * (alignment.abs() / (.04 + ray_energy.mean(1))).tanh()
    else:
        delays = h.real.new_zeros(b, len(DELAYS[regime]))
        spatial = h.real.new_zeros(b, len(DELAYS[regime]))
    return Handles(radial, cpe_increment, phase, delays, spatial, .06 * _bound(field_h))


def _pa_residual_gain(u, radial):
    """Bounded multiplicative AM-AM/AM-PM correction at each waveform sample."""
    p = u.abs().square().clamp_max(8.)
    c3, c5 = radial.unbind(-1)
    residual = .12 * (c3[..., None] * (p - 2) / 2 +
                      c5[..., None] * (p.square() - 6*p + 6) / 6)
    amplitude = 1 + .15 * torch.tanh(residual.real / .15)
    angle = .20 * torch.tanh(residual.imag / .20)
    return amplitude * torch.exp(1j * angle)


def _nominal_pa(power):
    a, knee, alpha = NOMINAL_PA
    amplitude = (1 + (power / a**2).pow(knee)).pow(-.5 / knee)
    return amplitude * torch.exp(1j * alpha * power / (1 + power))


def _pa_distortion(u, nominal_gain, radial):
    """Remove each symbol's complex linear projection from its learned PA residual."""
    delta = u * nominal_gain * (_pa_residual_gain(u, radial) - 1)
    energy = u.abs().square().sum(-1, keepdim=True)
    linear = (u.conj() * delta).sum(-1, keepdim=True) / torch.where(energy > 0, energy, 1.)
    return delta - linear * u


def _emit(x, regime, radial):
    u = oversample_frequency_to_time(x, 2)
    if regime != 2:
        nominal = _nominal_pa(u.abs().square()) / _nominal_gain()
        u = u * nominal + _pa_distortion(u, nominal, radial)
    return decimate_time_to_frequency(u, x.shape[-1], 2)


def _radial_inverse(z, radial):
    """Archived Iteration 2 inverse, retained for its regression test."""
    u = oversample_frequency_to_time(z, 2)
    inverse = _pa_residual_gain(u, radial).reciprocal()
    # Transform only the residual, so zero coefficients preserve z exactly.
    return z + decimate_time_to_frequency(u * (inverse - 1), z.shape[-1], 2)


def _demap_input(y, h, nv, radial, equalized=None):
    z, variance, informed = _equalize(y, h, nv) if equalized is None else equalized
    if radial is None:
        return z, variance, informed

    # Only the within-symbol nonlinear component enters the payload correction.
    x = torch.where(informed, _soft_symbols(z, variance), 0.)
    u = oversample_frequency_to_time(x, 2)
    nominal = _nominal_pa(u.abs().square()) / _nominal_gain()
    distortion = decimate_time_to_frequency(_pa_distortion(u, nominal, radial), z.shape[-1], 2)
    return z - .04 * _bound(distortion / .20), variance, informed


class AxialEMTransformer(nn.Module):
    """Weight-tied waveform EM with full-grid attention or matched local mixing."""

    def __init__(self, variant="pure", steps=2):
        super().__init__()
        if variant not in {"pure", "attention", "parity", "local", "local_parity", "no_gauge",
                           "physics", "physics_parity"}:
            raise ValueError(f"Unknown axial receiver variant {variant!r}")
        if steps not in (2, 3):
            raise ValueError("Expected two or three waveform steps")
        self.steps = steps
        self.pa_dewarp = True
        self.spatial_irc = False
        self.spatial_ula = False
        self.defer_r1_pa = False
        self.cue_weight = .05
        self.chirp_mode = "baseline"
        self.variant = "pure" if variant == "attention" else variant
        self.parity = self.variant in {"parity", "local_parity", "physics_parity"}
        if self.variant not in {"physics", "physics_parity"}:
            self.head = AxialOperatorBlock(variant=self.variant)
            self.initial_map = copy.deepcopy(self.head).requires_grad_(False)

    def _fields(self, feat, scale, at_initial_map, probe):
        if self.variant == "no_gauge":
            feat = feat * scale[:, None, None, None, None]
        fields, record, spatial_operator = self.head(feat, probe=bool(probe), return_spatial=True)
        initial, _ = self.initial_map(feat)
        fields = fields - initial
        if at_initial_map:
            # Identical GEMMs can choose different floating-point reductions.
            # Restore the exact zero while retaining the centered map Jacobian.
            fields = fields - fields.detach()
        return fields, record, spatial_operator

    def forward(self, obs, probe=False):
        y, nv, scale = _gauge(obs, True)
        b, r, t, n = y.shape
        geometry = _dictionary(obs.regime, n)
        nominal = geometry[0]
        amp, rot, mean_rot, cpe, has_pilot = _initialize(obs, y, geometry)
        basis = nominal.expand(b, -1, -1)
        anchor = ((amp[:, :, None] * mean_rot[:, None]) @ nominal) * cpe[:, None, :, None]
        pwr = y[:, :, obs.pilot_mask].abs().square().mean((1, 2))
        ref_pwr = torch.where(pwr > 0, pwr, torch.where(nv > 0, nv, 1.))
        ridge = torch.diag_embed((nv / ref_pwr)[:, None] / nv.new_tensor(POWERS[obs.regime]))
        freq = torch.fft.fftfreq(n, dtype=torch.float64)
        rx = torch.fft.ifft(y, norm="ortho")
        pa_present = obs.regime != 2
        h, clean, radial = anchor, y, None
        records = []
        learned = hasattr(self, "head")
        at_initial_map = learned and all(
            torch.equal(p, p0) for p, p0 in zip(self.head.parameters(), self.initial_map.parameters())
        )
        last_spatial = None
        def demap(clean, h, nv, radial):
            if last_spatial is None:
                return _demap_input(clean, h, nv, radial)
            weighted = innovation_equalize(clean, h, nv, obs.pilot_mask, obs.pilots, *last_spatial)
            return _demap_input(clean, h, nv, radial, weighted)

        for step in range(self.steps):
            z, var, informed = demap(clean, h, nv, radial)
            if self.parity and step:
                soft = soft_parity_symbols(z, var, informed, obs.pilot_mask)
            else:
                soft = torch.where(informed, _soft_symbols(z, var), 0.)
            x = torch.where(obs.pilot_mask, obs.pilots, soft)
            if learned:
                residual = y - h * x[:, None]
                if step == 0:
                    residual = residual * obs.pilot_mask
                ray_amp = amp[:, :, None] * mean_rot[:, None] * cpe[:, None, :, None]
                paths = ray_amp.unsqueeze(-2) * basis.mT[:, None, None]
                feat = full_frame_features(h, y, x, residual, anchor, obs.pilot_mask, paths)
                fields, record, spatial_operator = self._fields(feat, scale, at_initial_map, probe)
                if record is not None:
                    coords, joint, axes = record
                    axes = dict(axes)
                    if "space" in axes:
                        axes["spatial"] = axes.pop("space")
                    records.append((coords, joint, axes) if probe == "axes" else (coords, joint))
            else:
                fields = y.new_zeros(b, r, t, n, 4)
                spatial_operator = None
            bank_residual = None
            if learned and obs.regime == 2 and step >= 1:
                if self.chirp_mode == "baseline":
                    bank_residual = _soft_chirp_residual(clean, x, nominal, rot, fields[..., 1], self.cue_weight)
                else:
                    from p6.math_chirp import residual as chirp_update
                    pilot_reliability = None
                    if self.chirp_mode in ("fisher", "fisher_trust") and step == 2:
                        from p6.math_chirp_fisher import reliability
                        pilot_reliability = reliability(y, nv, obs.pilot_mask)
                    bank_residual = chirp_update(clean, x, nominal, rot, fields[..., 1],
                                                 self.cue_weight, step, self.chirp_mode, pilot_reliability)
            handles = _physical_handles(fields, h, amp, mean_rot, cpe, nominal, obs.regime, x,
                                        bank_residual)
            spatial_handle = (spatial_operator, handles.spatial)
            if self.spatial_ula and obs.regime == 3:
                spatial_handle += (int(obs.pilot_mask.sum()),)
            radial = handles.radial if pa_present and self.pa_dewarp else None
            basis = nominal * torch.exp(-2j * math.pi * handles.delays[..., None] * freq)
            rotation = torch.exp(1j * handles.cpe)
            design_cpe = cpe * rotation
            if obs.regime == 2:
                turn = torch.exp(1j * handles.phase)
                # Keep _initialize's exact mean when the centered update is
                # zero. Both sample and row rotations accumulate across steps.
                mean_rot = mean_rot + (rot * (turn - 1)).mean(2)
                rot = rot * turn
            if step == 0:
                amp = _apply_spatial(amp, spatial_handle)
                diagonal = ((amp[:, :, None] * mean_rot[:, None]) @ basis[:, None])
                h = diagonal * design_cpe[:, None, :, None] + handles.delta_h
                early_pa = None if obs.regime == 1 and self.defer_r1_pa else radial
                z, var, informed = _demap_input(clean, h, nv, early_pa)
                soft = torch.where(informed, _soft_symbols(z, var), 0.)
                x = torch.where(obs.pilot_mask, obs.pilots, soft)

            learned_pa = pa_present and not (obs.regime == 1 and self.defer_r1_pa and step == 0)
            coefficients = handles.radial if learned_pa else torch.zeros_like(handles.radial)
            emitted = _emit(x, obs.regime, coefficients)
            if obs.regime == 1 and getattr(self, "r1_differential", False):
                from p6.math_pa import differential_emission
                emitted = emitted + differential_emission(x, coefficients, obs.pilot_mask)
            wave = torch.fft.ifft(emitted[..., None] * basis.mT[:, None], dim=2, norm="ortho")
            if rot is not None:
                wave = wave * rot
            sample_bridge = None
            if obs.regime == 1 and getattr(self, "r1_bridge", False):
                from p6.math_pa import bridge
                sample_bridge = bridge(design_cpe, handles.cpe, n)
                if getattr(self, "r1_tangent", False):
                    from p6.math_pa import tangent_bridge
                    sample_bridge = tangent_bridge(sample_bridge, design_cpe, handles.cpe, wave, amp, rx, nv)
                wave = wave * sample_bridge[..., None]
            design = (wave * design_cpe[:, :, None, None]).reshape(b, t*n, -1)
            rhs = design.mH @ rx.reshape(b, r, t*n).mT
            amp = torch.linalg.solve(design.mH @ design + ridge, rhs).mT
            amp = _apply_spatial(amp, spatial_handle)
            prediction = (amp @ wave.reshape(b, t*n, -1).mT).reshape(b, r, t, n)
            cpe = (rotation * _unit((rx * prediction.conj()).sum((1, 3)))
                   if rot is None else design_cpe)
            diagonal = ((amp[:, :, None] * mean_rot[:, None]) @ basis[:, None]) * cpe[:, None, :, None]
            if sample_bridge is not None:
                diagonal = diagonal * sample_bridge.mean(-1)[:, None, :, None]
            h = diagonal + handles.delta_h
            received = torch.fft.fft(prediction * cpe[:, None, :, None], norm="ortho")
            clean = y - (received - diagonal * x[:, None])
            if self.spatial_irc and obs.regime == 3 and spatial_operator is not None:
                last_spatial = spatial_operator, handles.spatial

        z, variance, informed = demap(clean, h, nv, radial)
        llr = _llrs(z, variance)
        noiseless = -math.log(torch.finfo(torch.float64).eps) * llr.sign()
        llr = torch.where((nv == 0)[:, None, None, None], noiseless, llr)
        llr = torch.where((informed & has_pilot[:, None, None])[..., None], llr, 0.)
        physical_h = torch.where(has_pilot[:, None, None, None], h * scale[:, None, None, None], 0.)
        if pa_present:
            physical_h = physical_h / _nominal_gain()
        return (llr, physical_h, records) if probe else (llr, physical_h)


class AxialLocalConv(AxialEMTransformer):
    """The same physical receiver with three-tap mixing on each axis."""

    def __init__(self, parity=False, steps=2):
        super().__init__(variant="local_parity" if parity else "local", steps=steps)


def real_parameters(model):
    return sum(p.numel() * (2 if p.is_complex() else 1)
               for p in model.parameters() if p.requires_grad)
