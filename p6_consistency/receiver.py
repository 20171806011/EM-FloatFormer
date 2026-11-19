"""R2 observation-aware mechanism ablations, with frozen learned weights.

No simulation truth enters this module. A supplied receive-gain map is
privileged side information. Transmitted pilots remain known at erased
receive positions. Full consistency refers to the fixed linear effective
operator and approximate diagonal moments, not exact joint Bayesian EM.
"""
import math

import torch
from torch import nn

from p5.unrolled_em_transformer import _llrs
from p5.wavefield_sim import POWERS
from p6.axial_operator import full_frame_features
from p6.axial_em_transformer import (
    _apply_spatial, _demap_input, _dictionary, _physical_handles,
)
from p6_softinfo.moments import symbol_moments
from p6_softinfo.receiver import SoftInformationReceiver
from p6_consistency.observation import canonicalize, gauge, initialize, pilot_reliability
from p6_consistency.reliability import cavity_demap, make_prior
from p6_consistency.weighted import frequency_operators, weighted_amplitude_update


class LegacyTraceReceiver(SoftInformationReceiver):
    """Capture moments without changing the archived receiver's mean/outputs."""
    def _moments(self, z, var, informed, obs, parity):
        mean, variance = symbol_moments(z, var, informed, pilot_mask=obs.pilot_mask,
                                        pilots=obs.pilots, parity=parity)
        self._captured.append((mean.detach(), variance.detach()))
        return mean, variance if self.use_variance else None

    def forward(self, obs, probe=False):
        self._captured = []
        result = super().forward(obs, probe)
        if len(self._captured) != self.model.steps + 1:
            raise ValueError("Unexpected archived moment-call layout")
        # First call precedes the step-zero learned update and is not an M-step input.
        self.last_trace = [dict(step=i, mean=m, variance=v)
                           for i, (m, v) in enumerate(self._captured[1:])]
        return result


class ObservationAwareReceiver(nn.Module):
    def __init__(self, model, *, use_variance=False, effective=False, cavity=False):
        super().__init__()
        self.model = model
        self.use_variance, self.effective, self.cavity = use_variance, effective, cavity
        if cavity and not effective:
            raise ValueError("Cavity arms require the full effective physical operator")
        self.last_trace, self.last_diagnostics = [], None

    def forward(self, observation, observation_gain=None, probe=False):
        if observation.regime != 2 or observation.y.shape[1] != 1:
            raise ValueError("This study is restricted to single-antenna linear R2")
        obs, valid = canonicalize(observation, observation_gain)
        y, nv, scale = gauge(obs, valid)
        b, r, t, n = y.shape
        weights = valid.to(y.real.dtype).expand(b, -1, -1)
        pilot_observed = obs.pilot_mask & valid
        model = self.model
        geometry = _dictionary(2, n)
        nominal = geometry[0]
        amp, rot, mean_rot, cpe, has_pilot = initialize(obs, y, geometry, valid)
        basis = nominal.expand(b, -1, -1)
        anchor = ((amp[:, :, None] * mean_rot[:, None]) @ nominal) * cpe[:, None, :, None]
        pwr = (y[:, :, pilot_observed].abs().square().mean((1, 2)) if bool(pilot_observed.any())
               else nv.new_zeros(b))
        ref = torch.where(pwr > 0, pwr, torch.where(nv > 0, nv, 1.))
        ridge = torch.diag_embed((nv / ref)[:, None] / nv.new_tensor(POWERS[2]))
        freq = torch.fft.fftfreq(n, dtype=torch.float64)
        h, clean = anchor, y
        propagation = self._operator(amp, basis, rot, cpe, torch.zeros_like(h)) if self.cavity else None
        learned = hasattr(model, "head")
        at_initial = learned and all(torch.equal(p, q) for p, q in
                                    zip(model.head.parameters(), model.initial_map.parameters()))
        traces, moments, reliability_records, records = [], [], [], []
        previous_llr = None
        for step in range(model.steps):
            prior = (make_prior(y[:, 0], obs.pilot_mask, obs.pilots, previous_llr,
                                parity=model.parity and step > 0) if self.cavity else None)

            def demap():
                if self.cavity:
                    result = cavity_demap(y[:, 0], propagation[:, 0], nv, weights, prior,
                                          obs.pilot_mask, obs.pilots)
                    return result.mean, result.posterior_variance, result.channel_llr, result.diagnostics
                z, variance, informed = _demap_input(clean, h, nv, None)
                informed = informed & valid
                mean, sv = symbol_moments(z, variance, informed, pilot_mask=obs.pilot_mask,
                                          pilots=obs.pilots, parity=model.parity and step > 0)
                return mean, sv, None, None

            x, symbol_var, channel_llr, demap_record = demap()
            if learned:
                # A missing feature is imputed by the current diagonal mean;
                # its innovation is zero. Imputed values NEVER enter likelihoods.
                feature_y = torch.where(valid, y, h * x[:, None])
                residual = feature_y - h * x[:, None]
                if step == 0:
                    residual = residual * pilot_observed
                ray_amp = amp[:, :, None] * mean_rot[:, None] * cpe[:, None, :, None]
                paths = ray_amp.unsqueeze(-2) * basis.mT[:, None, None]
                feat = full_frame_features(h, feature_y, x, residual, anchor, pilot_observed, paths)
                fields, record, spatial_operator = model._fields(feat, scale, at_initial, probe)
                if record is not None:
                    coords, joint, axes = record
                    axes = dict(axes)
                    if "space" in axes:
                        axes["spatial"] = axes.pop("space")
                    records.append((coords, joint, axes) if probe == "axes" else (coords, joint))
            else:
                fields, spatial_operator = y.new_zeros(b, r, t, n, 4), None
            bank_residual = None
            if learned and step >= 1:
                from p6.math_chirp import residual as chirp_update
                q = (pilot_reliability(y, nv, obs.pilot_mask, valid)
                     if model.chirp_mode in ("fisher", "fisher_trust") and step == 2 else None)
                bank_residual = chirp_update(clean * valid, x * valid, nominal, rot,
                    fields[..., 1] * valid, model.cue_weight, step, model.chirp_mode, q)
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
                if self.cavity:
                    propagation = self._operator(amp, basis, rot, design_cpe, handles.delta_h)
                # Reuse the SAME prior for the within-pass re-demap.
                x, symbol_var, channel_llr, demap_record = demap()
            traces.append(dict(step=step, mean=x.detach(), variance=symbol_var.detach()))
            fit_variance = symbol_var if self.use_variance else torch.zeros_like(symbol_var)
            fitted = weighted_amplitude_update(x, fit_variance, y, basis, rot, design_cpe,
                weights, ridge, handles.delta_h if self.effective else None)
            amp = _apply_spatial(fitted.amplitudes.mT, spatial_handle)
            cpe = design_cpe
            physical = self._operator(amp, basis, rot, cpe, torch.zeros_like(h))
            diagonal = physical.diagonal(dim1=-2, dim2=-1)
            h = diagonal + handles.delta_h
            prediction = torch.einsum("brtfk,btk->brtf", physical, x)
            clean = y - (prediction - diagonal * x[:, None])
            if self.cavity:
                propagation = physical + torch.diag_embed(handles.delta_h)
                previous_llr = channel_llr
                reliability_records.append(dict(step=step, **demap_record))
            moments.append(dict(step=step, mean_variance=float(symbol_var.mean()),
                covariance_trace_mean=float(fitted.gram_correction.diagonal(dim1=-2, dim2=-1).real.sum(-1).mean()),
                covariance_rhs_norm=float(torch.linalg.vector_norm(fitted.covariance_rhs)),
                objective_sum=float(fitted.objective.sum()),
                uncertainty_energy_sum=float(fitted.uncertainty_energy.sum()),
                pinv_batches=int(fitted.used_pinv.sum())))
        if self.cavity:
            final = cavity_demap(y[:, 0], propagation[:, 0], nv, weights, prior,
                                 obs.pilot_mask, obs.pilots)
            llr = final.channel_llr
            reliability_records.append(dict(step="final", **final.diagnostics))
            informed = final.informed
        else:
            z, variance, informed = _demap_input(clean, h, nv, None)
            llr = _llrs(z, variance)
            llr = torch.where((nv == 0)[:, None, None, None],
                              -math.log(torch.finfo(torch.float64).eps) * llr.sign(), llr)
            informed = informed & valid
        llr = torch.where((informed & has_pilot[:, None, None])[..., None], llr, 0.)
        physical_h = torch.where(has_pilot[:, None, None, None], h * scale[:, None, None, None], 0.)
        self.last_trace = traces
        self.last_diagnostics = dict(moments=moments, cavity=reliability_records,
            observed_pilots=int(pilot_observed.sum()), transmitted_pilots=int(obs.pilot_mask.sum()),
            observed_frequency_bins=int(valid.sum()))
        return (llr, physical_h, records) if probe else (llr, physical_h)

    @staticmethod
    def _operator(amp, basis, rot, cpe, delta_h):
        return (torch.einsum("btpfk,brp->brtfk", frequency_operators(basis, rot, cpe), amp)
                + torch.diag_embed(delta_h))
