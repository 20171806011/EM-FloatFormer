"""Isolated R2 receiver with posterior-variance and full-waveform ablations.

The R2 forward path is explicitly adapted from the frozen P6 receiver. No
weights or archived sources are changed. ``use_variance=False, refine=False``
is checked against that receiver bit for bit. Other regimes are rejected.
"""
import math

import torch
from torch import nn

from p5.unrolled_em_transformer import _gauge, _llrs, _soft_symbols
from p5.wavefield_sim import POWERS
from p6.axial_operator import full_frame_features
from p6.axial_em_transformer import (
    _apply_spatial, _demap_input, _dictionary, _emit, _initialize,
    _physical_handles, _soft_chirp_residual,
)
from p6.parity import soft_parity_symbols
from p6_followup.common import intervention
from p6_softinfo.moments import symbol_moments, waveform_gram_correction


class SoftInformationReceiver(nn.Module):
    def __init__(self, model, *, use_variance=True, refine=False, chirp="trust"):
        super().__init__()
        self.model = model
        self.use_variance, self.refine, self.chirp = use_variance, refine, chirp
        self.last_diagnostics = None

    def forward(self, obs, probe=False):
        if obs.regime != 2:
            raise ValueError("This study implements the linear R2 waveform only")
        with intervention(self.chirp):
            return self._forward_r2(obs, probe)

    def _moments(self, z, var, informed, obs, parity):
        if self.use_variance:
            return symbol_moments(z, var, informed, pilot_mask=obs.pilot_mask,
                                  pilots=obs.pilots, parity=parity)
        soft = (soft_parity_symbols(z, var, informed, obs.pilot_mask) if parity
                else torch.where(informed, _soft_symbols(z, var), 0.))
        return torch.where(obs.pilot_mask, obs.pilots, soft), None

    def _forward_r2(self, obs, probe):
        model = self.model
        y, nv, scale = _gauge(obs, True)
        b, r, t, n = y.shape
        geometry = _dictionary(2, n)
        nominal = geometry[0]
        amp, rot, mean_rot, cpe, has_pilot = _initialize(obs, y, geometry)
        basis = nominal.expand(b, -1, -1)
        anchor = ((amp[:, :, None] * mean_rot[:, None]) @ nominal) * cpe[:, None, :, None]
        pwr = y[:, :, obs.pilot_mask].abs().square().mean((1, 2))
        ref_pwr = torch.where(pwr > 0, pwr, torch.where(nv > 0, nv, 1.))
        ridge = torch.diag_embed((nv / ref_pwr)[:, None] / nv.new_tensor(POWERS[2]))
        freq = torch.fft.fftfreq(n, dtype=torch.float64)
        rx = torch.fft.ifft(y, norm="ortho")
        h, clean = anchor, y
        records, moment_records = [], []
        learned = hasattr(model, "head")
        at_initial_map = learned and all(
            torch.equal(p, p0) for p, p0 in zip(model.head.parameters(), model.initial_map.parameters())
        )
        profile_record = None
        for step in range(model.steps):
            z, var, informed = _demap_input(clean, h, nv, None)
            x, symbol_var = self._moments(z, var, informed, obs, model.parity and step > 0)
            if learned:
                residual = y - h * x[:, None]
                if step == 0:
                    residual = residual * obs.pilot_mask
                ray_amp = amp[:, :, None] * mean_rot[:, None] * cpe[:, None, :, None]
                paths = ray_amp.unsqueeze(-2) * basis.mT[:, None, None]
                feat = full_frame_features(h, y, x, residual, anchor, obs.pilot_mask, paths)
                fields, record, spatial_operator = model._fields(feat, scale, at_initial_map, probe)
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
            if learned and step >= 1:
                if model.chirp_mode == "baseline":
                    bank_residual = _soft_chirp_residual(clean, x, nominal, rot, fields[..., 1], model.cue_weight)
                else:
                    from p6.math_chirp import residual as chirp_update
                    q = None
                    if model.chirp_mode in ("fisher", "fisher_trust") and step == 2:
                        from p6.math_chirp_fisher import reliability
                        q = reliability(y, nv, obs.pilot_mask)
                    bank_residual = chirp_update(clean, x, nominal, rot, fields[..., 1],
                                                model.cue_weight, step, model.chirp_mode, q)
            handles = _physical_handles(fields, h, amp, mean_rot, cpe, nominal, 2, x, bank_residual)
            spatial_handle = spatial_operator, handles.spatial
            basis = nominal * torch.exp(-2j * math.pi * handles.delays[..., None] * freq)
            design_cpe = cpe * torch.exp(1j * handles.cpe)
            turn = torch.exp(1j * handles.phase)
            mean_rot = mean_rot + (rot * (turn - 1)).mean(2)
            rot = rot * turn
            if step == 0:
                amp = _apply_spatial(amp, spatial_handle)
                diagonal = (amp[:, :, None] * mean_rot[:, None]) @ basis[:, None]
                h = diagonal * design_cpe[:, None, :, None] + handles.delta_h
                z, var, informed = _demap_input(clean, h, nv, None)
                # Step zero re-demaps; variance must accompany this new mean.
                x, symbol_var = self._moments(z, var, informed, obs, False)
            emitted = _emit(x, 2, torch.zeros_like(handles.radial))
            wave = torch.fft.ifft(emitted[..., None] * basis.mT[:, None], dim=2, norm="ortho") * rot
            design = (wave * design_cpe[:, :, None, None]).reshape(b, t*n, -1)
            rhs = design.mH @ rx.reshape(b, r, t*n).mT
            normal = design.mH @ design + ridge
            if self.use_variance:
                correction = waveform_gram_correction(symbol_var, basis, rot, design_cpe)
                normal = normal + correction
                moment_records.append(dict(step=step, mean_variance=float(symbol_var.detach().mean()),
                    covariance_trace_mean=float(correction.detach().diagonal(dim1=-2, dim2=-1).real.sum(-1).mean())))
            amp = torch.linalg.solve(normal, rhs).mT
            if self.refine and step == model.steps - 1:
                from p6_softinfo.profile import profile_refine
                updated = profile_refine(emitted, torch.zeros_like(x.real) if symbol_var is None else symbol_var,
                                         basis, rot, design_cpe, rx.reshape(b, r, t*n).mT, ridge)
                rot, amp, profile_record = updated[0], updated[1].mT, updated[2]
                mean_rot = rot.mean(2)
                wave = torch.fft.ifft(emitted[..., None] * basis.mT[:, None], dim=2, norm="ortho") * rot
            amp = _apply_spatial(amp, spatial_handle)
            prediction = (amp @ wave.reshape(b, t*n, -1).mT).reshape(b, r, t, n)
            cpe = design_cpe
            diagonal = ((amp[:, :, None] * mean_rot[:, None]) @ basis[:, None]) * cpe[:, None, :, None]
            h = diagonal + handles.delta_h
            received = torch.fft.fft(prediction * cpe[:, None, :, None], norm="ortho")
            clean = y - (received - diagonal * x[:, None])
        z, variance, informed = _demap_input(clean, h, nv, None)
        llr = _llrs(z, variance)
        noiseless = -math.log(torch.finfo(torch.float64).eps) * llr.sign()
        llr = torch.where((nv == 0)[:, None, None, None], noiseless, llr)
        llr = torch.where((informed & has_pilot[:, None, None])[..., None], llr, 0.)
        physical_h = torch.where(has_pilot[:, None, None, None], h * scale[:, None, None, None], 0.)
        self.last_diagnostics = dict(moments=moment_records, profile=profile_record)
        return (llr, physical_h, records) if probe else (llr, physical_h)
