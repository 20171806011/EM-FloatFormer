"""Unified Phase 7 EM7HypothesisReceiver answering sent bits, held-out waveforms, and disagreement."""
import copy
import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from p5.wavefield_sim import POWERS
from p6.axial_em_transformer import _centered_angles, _dictionary, _sample_phase, _bank_templates
from p6.math_chirp import _bank_geometry, _tangent
from p6_consistency.reliability import SymbolPrior, cavity_demap
from p7.hypotheses import (
    branch_evidence, canonicalize, diagonal_ici_demap, frequency_operators_fast,
    initialize_hypotheses, make_code_prior, mixture_posterior, pilot_gauge,
    pilot_reliability_masked, weighted_amplitude_solve,
)
from p7.routing_operator import EvidenceRoutingBlock, routing_features


REAL = torch.float64
COMPLEX = torch.complex128


class EM7Output(NamedTuple):
    bit_llr: torch.Tensor
    h_hat: torch.Tensor
    pi: torch.Tensor
    log_pi: torch.Tensor
    symbol_mean: torch.Tensor
    symbol_var_total: torch.Tensor
    symbol_var_within: torch.Tensor
    symbol_var_between: torch.Tensor
    bit_disagreement: torch.Tensor
    pred_y: torch.Tensor
    pred_var: torch.Tensor
    branch_pred_y: torch.Tensor
    branch_pred_var: torch.Tensor
    traces: list
    ell_phys: torch.Tensor
    ell_const: torch.Tensor
    delta_route: torch.Tensor


def _soft_taps_masked(clean, x, basis, valid, field_ray, cue_weight):
    """Estimate per-symbol path taps ignoring unobserved frequency bins."""
    n = clean.shape[-1]
    v_c = valid.to(clean.dtype)[..., None]  # BM, T, N, 1
    design = x[..., None] * basis.mT * v_c  # BM, T, N, P (masked)
    normal = design.mH @ design
    ridge = .02 * n * torch.eye(normal.shape[-1], dtype=normal.dtype, device=normal.device)
    rhs = design.mH @ (clean * valid).permute(0, 2, 3, 1)  # BM, T, P, 1
    taps = torch.linalg.solve(normal + ridge, rhs).permute(0, 3, 2, 1)  # BM, 1, P, T
    if field_ray is not None and cue_weight:
        bound = field_ray / (1. + field_ray.abs().square()).sqrt()
        taps = taps + cue_weight * ((bound @ basis.mH).mT / n)
    return taps


def _sector_chirp_update(clean, x, basis, rot, valid, field_ray, cue_weight, step, q, multi_hyp):
    """Smooth within-sector bank refinement + Fisher-trust tangent update preventing collapse."""
    t, n = rot.shape[1:3]
    taps = _soft_taps_masked(clean, x, basis, valid, field_ray, cue_weight)
    means, angles = _bank_templates(t, n)
    coherence, _ = _bank_geometry(taps, rot)  # BM, P, 162
    n_pos = coherence.shape[-1] // 2  # 81

    if multi_hyp:
        # Preserve each branch's assigned Doppler sector on dominant rays 0 & 1:
        # m=0:(+,+), m=1:(+,-), m=2:(-,+), m=3:(-,-), ray 2 unrestricted.
        b = clean.shape[0] // 4
        sector = torch.ones((4, 3, coherence.shape[-1]), dtype=torch.bool, device=rot.device)
        sector[0, 0, n_pos:] = False
        sector[0, 1, n_pos:] = False
        sector[1, 0, n_pos:] = False
        sector[1, 1, :n_pos] = False
        sector[2, 0, :n_pos] = False
        sector[2, 1, n_pos:] = False
        sector[3, 0, :n_pos] = False
        sector[3, 1, :n_pos] = False
        mask = sector[None].expand(b, -1, -1, -1).reshape(b * 4, 3, -1)
    else:
        mask = torch.ones_like(coherence, dtype=torch.bool)

    beta = (40. if step == 1 else (20. + 60. * q)[:, None, None])
    logits = (beta * coherence).masked_fill(~mask, -torch.inf)
    weights = logits.softmax(-1)
    bank_res = (weights @ angles - _centered_angles(rot[:, :, n // 2].mT)).mT
    if step == 1:
        return bank_res
    tangent = _tangent(taps, rot).mT
    info = q[:, None, None]
    return info * bank_res + (1. - info) * tangent


def _phase_step(fields, h, amp, mean_rot, cpe, nominal, bank_res):
    b, r, t, n = h.shape
    field_h, field_ray, field_pa, _ = fields.unbind(-1)
    ref = h.conj() / (.10 + h.abs().square())
    pa = field_pa * ref
    ray_state = amp[:, :, None] * mean_rot[:, None] * cpe[:, None, :, None]
    ray_energy = ray_state.abs().square().mean(1)
    projection = ((field_ray + .35 * field_pa) @ nominal.mH) / n
    phase_score = (projection * ray_state.conj()).imag.mean(1) / (.04 + ray_energy)
    row_phase = 3.2 * torch.asinh(phase_score)
    if bank_res is not None:
        signed = (4. * pa.real.mean((1, 2, 3))).tanh()
        energy = pa.abs().square().mean((1, 2, 3))
        gate = 1. - (-24. * energy).exp() * (1. - signed)
        row_phase = row_phase + gate[:, None, None] * bank_res
    delta_h = .06 * field_h / (1. + field_h.abs().square()).sqrt()
    return _sample_phase(row_phase, n), delta_h


class EM7HypothesisReceiver(nn.Module):
    """Multi-hypothesis C*-equivariant EM receiver answering bits, waveforms & disagreement."""

    def __init__(self, variant="axial", steps=3, width=8, cue_weight=0.05, gamma_init=12.0):
        super().__init__()
        if variant not in ("axial", "local", "fixed", "single_axial"):
            raise ValueError(f"Unknown EM7 variant: {variant}")
        self.variant = variant
        self.steps = steps
        self.width = width
        self.cue_weight = cue_weight
        self.multi_hyp = variant != "single_axial"
        self.learned = variant != "fixed"
        self.gamma_init = float(gamma_init)

        if self.learned:
            block_kind = "local" if variant == "local" else "axial"
            self.head = EvidenceRoutingBlock(block_kind, width=width)
            self.initial_map = copy.deepcopy(self.head)
            for p in self.initial_map.parameters():
                p.requires_grad_(False)
            self.fusion_scale = nn.Parameter(torch.zeros(2, dtype=REAL))

    def real_parameters(self):
        return sum(p.numel() * (2 if p.is_complex() else 1) for p in self.parameters() if p.requires_grad)

    def _fields(self, feat, at_initial, return_attn=False):
        if not self.learned:
            z = feat.new_zeros(*feat.shape[:-1], 4)
            r = feat.new_zeros(*feat.shape[:-1], 2)
            return z, r, None
        phys_live, route_live, *attn = self.head(feat, return_attn=return_attn)
        if at_initial:
            phys_init, route_init = phys_live.detach(), route_live.detach()
        else:
            with torch.no_grad():
                phys_init, route_init = self.initial_map(feat.detach(), return_attn=False)
        return (phys_live - phys_init, route_live - route_init, attn[0] if return_attn else None)

    def forward(self, obs, gain=None, return_attn=False):
        if obs.regime != 2 or obs.y.shape[1] != 1:
            raise ValueError("EM7 targets single-antenna R2 (20x32 16-QAM)")
        obs_c, valid = canonicalize(obs, gain)
        y, nv, scale = pilot_gauge(obs_c, valid)
        b, _, t, n = y.shape
        m = 4 if self.multi_hyp else 1
        pilot_observed = obs_c.pilot_mask[None] & valid  # B, T, N
        weights_bm = valid[:, None].expand(b, m, t, n).reshape(b * m, t, n).to(REAL)
        valid_bm = valid[:, None].expand(b, m, t, n).reshape(b * m, t, n)

        geometry = _dictionary(2, n)
        nominal = geometry[0]
        amp, rot, mean_rot, basis, cpe, has_pilot = initialize_hypotheses(
            obs_c, y, geometry, valid, multi_hyp=self.multi_hyp
        )
        # Per-burst pilot power for scale-covariant amplitude ridge
        po_f = pilot_observed[:, None].to(REAL)
        po_cnt = po_f.sum((-2, -1)).clamp_min(1.)
        pwr_b = (y.abs().square() * po_f).sum((-2, -1))[:, 0] / po_cnt[:, 0]
        has_obs_b = pilot_observed.any((-2, -1))
        pwr_b = torch.where(has_obs_b, pwr_b, 0.)
        ref_b = torch.where(pwr_b > 0, pwr_b, torch.where(nv > 0, nv, 1.))
        ridge_b = torch.diag_embed((nv / ref_b)[:, None] / nv.new_tensor(POWERS[2]))
        ridge_bm = ridge_b[:, None].expand(b, m, -1, -1).reshape(b * m, 3, 3)

        amp_bm = amp.reshape(b * m, 1, 3)
        rot_bm = rot.reshape(b * m, t, n, 3)
        mean_rot_bm = mean_rot.reshape(b * m, t, 3)
        basis_bm = basis.reshape(b * m, 3, n)
        cpe_bm = cpe.reshape(b * m, t)

        anchor_bm = ((amp_bm[:, :, None] * mean_rot_bm[:, None]) @ nominal) * cpe_bm[:, None, :, None]
        ops_bm = frequency_operators_fast(basis_bm, rot_bm, cpe_bm)
        G_bm = torch.einsum("btpfk,brp->brtfk", ops_bm, amp_bm).reshape(b * m, t, n, n)
        h_bm = G_bm.diagonal(dim1=-2, dim2=-1)[:, None]  # BM, 1, T, N
        delta_h_bm = torch.zeros_like(h_bm)

        y_bm = y[:, 0:1].expand(b, m, t, n).reshape(b * m, t, n)
        nv_bm = nv[:, None].expand(b, m).reshape(b * m)
        valid_f = valid_bm[:, None]  # BM, 1, T, N

        pi = torch.full((b, m), 1.0 / m, dtype=REAL, device=y.device)
        at_initial = (
            not self.learned
            or (
                all(torch.equal(p, q) for p, q in zip(self.head.parameters(), self.initial_map.parameters()))
                and bool((self.fusion_scale == 0).all())
            )
        )
        gamma_phys = self.gamma_init * (self.fusion_scale[0].exp() if self.learned else 1.0)
        gamma_const = self.gamma_init * (self.fusion_scale[1].exp() if self.learned else 1.0)

        q_b = pilot_reliability_masked(y, nv, obs_c.pilot_mask, valid)
        q_bm = q_b[:, None].expand(b, m).reshape(b * m)
        freq = torch.fft.fftfreq(n, dtype=REAL, device=y.device)

        traces = []
        prev_mix_llr = None
        last_attn = None
        delta_route_acc = y.real.new_zeros(b, m)

        for step in range(self.steps):
            prior_b = make_code_prior(y[:, 0], obs_c.pilot_mask, obs_c.pilots, prev_mix_llr, parity=(step > 0))
            prior_bm = SymbolPrior(
                prior_b.mean[:, None].expand(b, m, t, n).reshape(b * m, t, n),
                prior_b.variance[:, None].expand(b, m, t, n).reshape(b * m, t, n),
                prior_b.bit_logits[:, None].expand(b, m, t, n, 4).reshape(b * m, t, n, 4),
            )
            z_bm, tau_bm, inf_bm, x_bm, v_bm, cllr_bm = diagonal_ici_demap(
                y_bm, G_bm, nv_bm, valid_bm, prior_bm, obs_c.pilot_mask, obs_c.pilots
            )

            # Build full-grid routing + disagreement features from current branch belief
            pred_full_bm = torch.einsum("btfk,btk->btf", G_bm, x_bm)[:, None]  # BM, 1, T, N
            y_obs_bm = torch.where(valid_f, y_bm[:, None], 0.)
            pred_obs_bm = torch.where(valid_f, pred_full_bm, 0.)
            h_view = h_bm.reshape(b, m, 1, t, n)
            pred_view = pred_obs_bm.reshape(b, m, 1, t, n)
            h_bar = (pi[:, :, None, None, None] * h_view).sum(1, keepdim=True).expand_as(h_view).reshape(b * m, 1, t, n)
            y_bar = (pi[:, :, None, None, None] * pred_view).sum(1, keepdim=True).expand_as(pred_view).reshape(b * m, 1, t, n)
            res_obs_bm = y_obs_bm - pred_obs_bm
            if step == 0:
                res_obs_bm = res_obs_bm * (obs_c.pilot_mask[None, None] & valid_f)
            ray_amp = amp_bm[:, :, None] * mean_rot_bm[:, None] * cpe_bm[:, None, :, None]
            path_fields = ray_amp.unsqueeze(-2) * basis_bm.mT[:, None, None]

            feat = routing_features(
                h_bm, h_bar, y_obs_bm, x_bm, pred_obs_bm, y_bar, res_obs_bm, anchor_bm, path_fields
            )
            fields, route_fields, step_attn = self._fields(
                feat, at_initial=at_initial, return_attn=(return_attn and step == self.steps - 1)
            )
            if step_attn is not None:
                last_attn = step_attn

            # Strictly C*-invariant linear routing evidence readout (0 at step-0 init, nonzero grad!)
            du, dw = route_fields[:, 0].unbind(-1)  # BM, T, N
            h_energy = h_bm[:, 0].abs().square().mean((-2, -1)).clamp_min(1e-300)
            route_step = self.gamma_init * (
                (h_bm[:, 0].conj() * du + res_obs_bm[:, 0].conj() * dw).real.mean((-2, -1))
                / h_energy
            ).reshape(b, m)
            delta_route_acc = delta_route_acc + route_step

            # Chirp update (passes 1 & 2) using within-sector smooth fisher_trust
            clean_bm = y_bm - (pred_full_bm[:, 0] - h_bm[:, 0] * x_bm)
            bank_res = None
            if step >= 1:
                bank_res = _sector_chirp_update(
                    clean_bm[:, None], x_bm, nominal, rot_bm, valid_bm,
                    fields[..., 1] * valid_f, self.cue_weight, step, q_bm, self.multi_hyp
                )

            phase_step, delta_h_bm = _phase_step(fields, h_bm, amp_bm, mean_rot_bm, cpe_bm, nominal, bank_res)
            turn = torch.exp(1j * phase_step)
            mean_rot_bm = mean_rot_bm + (rot_bm * (turn - 1.)).mean(2)
            rot_bm = rot_bm * turn
            ops_bm = frequency_operators_fast(basis_bm, rot_bm, cpe_bm)

            # Update mixture posterior & trace before M-step amplitude refit (M-step moments)
            ell_p, ell_c = branch_evidence(
                y[:, 0], G_bm.reshape(b, m, t, n, n), x_bm.reshape(b, m, t, n), v_bm.reshape(b, m, t, n),
                z_bm.reshape(b, m, t, n), tau_bm.reshape(b, m, t, n),
                prior_b.bit_logits[:, None].expand(b, m, t, n, 4), nv, obs_c.pilot_mask, valid
            )
            logits_pi = gamma_phys * ell_p + gamma_const * ell_c + delta_route_acc
            if m > 1:
                log_pi = F.log_softmax(logits_pi, dim=-1)
                pi = log_pi.exp()
            else:
                log_pi = logits_pi.new_zeros((b, 1))
                pi = torch.ones((b, 1), dtype=REAL, device=y.device)

            mix_llr, _, sym_mean, total_var, _, _ = mixture_posterior(
                pi, log_pi, cllr_bm.reshape(b, m, t, n, 4),
                x_bm.reshape(b, m, t, n), v_bm.reshape(b, m, t, n)
            )
            traces.append(dict(step=step, mean=sym_mean.detach(), variance=total_var.detach()))

            amp_sol = weighted_amplitude_solve(
                x_bm, v_bm, y_bm[:, None], ops_bm, weights_bm, ridge_bm, delta_h_bm
            )
            amp_bm = amp_sol.transpose(-1, -2)  # BM, 1, 3
            G_bm = (torch.einsum("btpfk,brp->brtfk", ops_bm, amp_bm).reshape(b * m, t, n, n)
                    + torch.diag_embed(delta_h_bm[:, 0]))
            h_bm = G_bm.diagonal(dim1=-2, dim2=-1)[:, None]
            prev_mix_llr = mix_llr

        # Exact full-frequency Gaussian cavity demap on final effective propagation G_bm
        # Reuses step-2 code prior (prior_bm) cleanly without double-counting own-symbol prior
        cav = cavity_demap(
            y_bm, G_bm, nv_bm, weights_bm, prior_bm, obs_c.pilot_mask, obs_c.pilots
        )
        z_fin, tau_fin, inf_fin = cav.z, cav.variance, cav.informed
        x_fin, v_fin, cllr_fin = cav.mean, cav.posterior_variance, cav.channel_llr
        final_llr_m = cllr_fin.reshape(b, m, t, n, 4)
        final_z_m = z_fin.reshape(b, m, t, n)
        final_tau_m = tau_fin.reshape(b, m, t, n)
        final_inf_m = inf_fin.reshape(b, m, t, n)
        final_x_m = x_fin.reshape(b, m, t, n)
        final_v_m = v_fin.reshape(b, m, t, n)

        G_view = G_bm.reshape(b, m, t, n, n)
        ell_p, ell_c = branch_evidence(
            y[:, 0], G_view, final_x_m, final_v_m, final_z_m, final_tau_m,
            prior_b.bit_logits[:, None].expand(b, m, t, n, 4), nv, obs_c.pilot_mask, valid
        )
        logits_pi = gamma_phys * ell_p + gamma_const * ell_c + delta_route_acc
        if m > 1:
            log_pi = F.log_softmax(logits_pi, dim=-1)
            pi = log_pi.exp()
        else:
            log_pi = logits_pi.new_zeros((b, 1))
            pi = torch.ones((b, 1), dtype=REAL, device=y.device)

        bit_llr, bit_disagree, sym_mean, sym_var_tot, sym_var_in, sym_var_bet = mixture_posterior(
            pi, log_pi, final_llr_m, final_x_m, final_v_m
        )
        informed_any = final_inf_m.any(1) & has_pilot[:, None, None]
        bit_llr = torch.where(informed_any[..., None] & (~obs_c.pilot_mask)[None, ..., None], bit_llr, 0.)
        bit_disagree = torch.where(informed_any[..., None] & (~obs_c.pilot_mask)[None, ..., None], bit_disagree, 0.)

        h_hat = (pi[:, :, None, None] * G_view.diagonal(dim1=-2, dim2=-1)).sum(1)[:, None] * scale[:, None, None, None]
        h_hat = torch.where(has_pilot[:, None, None, None], h_hat, 0.)

        # Forward predictive consistency distribution in receive units (valid on ALL bins including held-out pilots!)
        pred_y_norm = torch.einsum("bmtfk,bmtk->bmtf", G_view, final_x_m)
        row_ref = G_view.abs().square().sum(-1).amax(-1, keepdim=True).amax(-2, keepdim=True)
        eps = torch.finfo(REAL).eps
        nv_safe = torch.maximum(nv[:, None, None, None], 64. * n * eps * torch.maximum(row_ref, nv[:, None, None, None]))
        pred_var_norm = nv_safe + torch.einsum("bmtfk,bmtk->bmtf", G_view.abs().square(), final_v_m)

        branch_pred_y = pred_y_norm * scale[:, None, None, None]
        branch_pred_var = pred_var_norm * scale.abs().square()[:, None, None, None]
        pred_y = (pi[:, :, None, None] * branch_pred_y).sum(1)
        pred_var = (pi[:, :, None, None] * (branch_pred_var + (branch_pred_y - pred_y[:, None]).abs().square())).sum(1)

        out = EM7Output(
            bit_llr=bit_llr, h_hat=h_hat, pi=pi, log_pi=log_pi,
            symbol_mean=sym_mean, symbol_var_total=sym_var_tot,
            symbol_var_within=sym_var_in, symbol_var_between=sym_var_bet,
            bit_disagreement=bit_disagree,
            pred_y=pred_y, pred_var=pred_var,
            branch_pred_y=branch_pred_y, branch_pred_var=branch_pred_var,
            traces=traces, ell_phys=ell_p, ell_const=ell_c, delta_route=delta_route_acc,
        )
        return (out, last_attn) if return_attn else out
