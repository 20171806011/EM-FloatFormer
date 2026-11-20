"""Stage 8.0: Direction-1 mathematical fixes on frozen Phase 7 R2 (C2 tangent-Laplace variance,
C5 true Bayesian sequential assimilation) + Off-Grid banded-ICI energy diagnostic."""
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from p4.simulator import q16, qpsk
from p4.slot_2d import encode_ldpc
from p5.study import counts, setup
from p5.wavefield_sim import CP, N, POWERS, frequency_basis, propagate, ray_phase
from p6 import experiment as ex
from p6.axial_em_transformer import _dictionary
from p6.data import dev_frame, dev_seed
from p6_consistency.calibration import calibration
from p6_consistency.reliability import SymbolPrior, _qam_moments, cavity_demap
from p7.conditions import apply_observation_gain, mask_for_condition
from p7.hypotheses import (
    branch_evidence, canonicalize, diagonal_ici_demap, frequency_operators_fast,
    initialize_hypotheses, make_code_prior, mixture_posterior, pilot_gauge,
    pilot_reliability_masked, weighted_amplitude_solve,
)
from p7.metrics import decoding_metrics, sequential_update_metrics, true_propagation_operator
from p7.model import EM7HypothesisReceiver, EM7Output, _phase_step, _sector_chirp_update
from p7.routing_operator import routing_features
from p7.train import load_checkpoint


REAL = torch.float64
COMPLEX = torch.complex128
REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
DEV_PAIRS = (901, 902, 903)


@torch.no_grad()
def _forward_with_state(model, obs, gain, use_tangent_laplace=False, prior_phi_reg=4.0):
    """Run frozen EM7HypothesisReceiver while returning internal branch state and optional
    in-branch Gauss-Newton / von-Mises tangent-Laplace epistemic variance (Direction 1A)."""
    obs_c, valid = canonicalize(obs, gain)
    y, nv, scale = pilot_gauge(obs_c, valid)
    b, _, t, n = y.shape
    m = 4 if model.multi_hyp else 1
    pilot_observed = obs_c.pilot_mask[None] & valid  # B, T, N
    weights_bm = valid[:, None].expand(b, m, t, n).reshape(b * m, t, n).to(REAL)
    valid_bm = valid[:, None].expand(b, m, t, n).reshape(b * m, t, n)

    geometry = _dictionary(2, n)
    nominal = geometry[0]
    amp, rot, mean_rot, basis, cpe, has_pilot = initialize_hypotheses(
        obs_c, y, geometry, valid, multi_hyp=model.multi_hyp
    )
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
    h_bm = G_bm.diagonal(dim1=-2, dim2=-1)[:, None]
    delta_h_bm = torch.zeros_like(h_bm)

    y_bm = y[:, 0:1].expand(b, m, t, n).reshape(b * m, t, n)
    nv_bm = nv[:, None].expand(b, m).reshape(b * m)
    valid_f = valid_bm[:, None]

    pi = torch.full((b, m), 1.0 / m, dtype=REAL, device=y.device)
    gamma_phys = model.gamma_init * (model.fusion_scale[0].exp() if model.learned else 1.0)
    gamma_const = model.gamma_init * (model.fusion_scale[1].exp() if model.learned else 1.0)
    q_b = pilot_reliability_masked(y, nv, obs_c.pilot_mask, valid)
    q_bm = q_b[:, None].expand(b, m).reshape(b * m)

    # Normalized temporal quadratic basis psi_t = [1, tau, tau^2] for phase-trajectory curvature
    tau_t = torch.linspace(-1.0, 1.0, t, dtype=REAL, device=y.device)
    psi_t = torch.stack((torch.ones_like(tau_t), tau_t, tau_t.square()), dim=-1)  # T, 3

    def _compute_epistemic_var(amp_cur, mean_rot_cur, basis_cur, cpe_cur, ops_cur, x_cur, v_cur, w_cur):
        """Exact C*-equivariant Gauss-Newton amplitude + quadratic phase-extrapolation variance [B, M, T, N]."""
        # 1. Amplitude Fisher covariance Sigma_a = N0 * (D^H W D + C + ridge)^{-1}
        p_cnt = 3
        wc = w_cur.to(COMPLEX)
        design = torch.einsum("btpfk,btk->btfp", ops_cur, x_cur)
        d_flat = design.reshape(b * m, t * n, p_cnt)
        d_w = (design * wc[..., None]).reshape(b * m, t * n, p_cnt)
        mean_norm = d_w.mH @ d_flat
        wv = (w_cur[:, :, :, None] * v_cur[:, :, None, :]).to(COMPLEX)
        op_flat = ops_cur.reshape(b * m, t, p_cnt, n * n).transpose(-1, -2).reshape(b * m, t * n * n, p_cnt)
        op_w = (ops_cur * wv[:, :, None]).reshape(b * m, t, p_cnt, n * n).transpose(-1, -2).reshape(b * m, t * n * n, p_cnt)
        normal = mean_norm + op_w.mH @ op_flat + ridge_bm.to(COMPLEX)
        normal = 0.5 * (normal + normal.mH)
        sigma_a = nv_bm[:, None, None] * torch.linalg.pinv(normal, hermitian=True)  # BM, 3, 3
        # Diagonal ray basis j_a[bm, t, n, p] = mean_rot[bm, t, p] * basis[bm, p, n] * cpe[bm, t]
        j_a = mean_rot_cur[:, :, None, :] * basis_cur.transpose(-1, -2)[:, None, :, :] * cpe_cur[:, :, None, None]
        v_amp = torch.einsum("btnp,bpq,btnq->btn", j_a.conj(), sigma_a, j_a).real.clamp_min(0.0)

        # 2. Temporal phase-trajectory Fisher covariance per ray from observed pilots + confident symbols
        # h_ray[bm, t, n, p] = amp[bm, 0, p] * j_a[bm, t, n, p]
        h_ray = amp_cur[:, 0, None, None, :] * j_a  # BM, T, N, 3
        # Weight each bin by observed reliability (1.0 on observed pilots, attenuated by 1/(1+v/0.2) on payload)
        po_bm = pilot_observed[:, None].expand(b, m, t, n).reshape(b * m, t, n)
        sym_rel = torch.where(po_bm, torch.ones_like(w_cur), w_cur * (0.25 / (0.25 + v_cur)))
        ray_snr_t = ((h_ray.abs().square() * sym_rel[..., None]).sum(dim=2)
                     / nv_bm[:, None, None].clamp_min(1e-12))  # BM, T, 3
        # Per-ray 3x3 temporal Fisher matrix I_phi = 2 * sum_t snr_t * psi_t psi_t^T + prior_phi_reg * diag([0.2, 1, 1])
        prior_diag = torch.tensor([0.2, 1.0, 1.0], dtype=REAL, device=y.device) * prior_phi_reg
        # Also penalize missing pilot aperture explicitly on non-pilot extrapolated rows
        po_t = po_bm.to(REAL).sum(dim=-1)  # BM, T
        ray_snr_pilot_t = ((h_ray.abs().square() * po_bm.to(REAL)[..., None]).sum(dim=2)
                           / nv_bm[:, None, None].clamp_min(1e-12))
        # Combine pilot Fisher (exact known symbols) + conservative payload Fisher (0.08 factor for loopy/unknown-symbol correlation)
        eff_snr_t = 2.0 * (ray_snr_pilot_t + 0.08 * (ray_snr_t - ray_snr_pilot_t).clamp_min(0.0))  # BM, T, 3
        I_phi = torch.einsum("btp,ti,tj->bpij", eff_snr_t, psi_t, psi_t) + torch.diag_embed(prior_diag)[None, None]
        Sigma_phi = torch.linalg.inv(I_phi)  # BM, 3, 3, 3
        sigma2_phi_t = torch.einsum("ti,bpij,tj->btp", psi_t, Sigma_phi, psi_t).clamp_min(0.0)  # BM, T, 3
        # Exact von Mises circular variance: |h_ray|^2 * (1 - exp(-sigma2_phi))
        v_phase = (h_ray.abs().square() * (1.0 - (-sigma2_phi_t[:, :, None, :]).exp())).sum(-1)  # BM, T, N
        return (v_amp + v_phase).reshape(b, m, t, n)

    traces = []
    prev_mix_llr = None
    delta_route_acc = y.real.new_zeros(b, m)

    for step in range(model.steps):
        prior_b = make_code_prior(y[:, 0], obs_c.pilot_mask, obs_c.pilots, prev_mix_llr, parity=(step > 0))
        prior_bm = SymbolPrior(
            prior_b.mean[:, None].expand(b, m, t, n).reshape(b * m, t, n),
            prior_b.variance[:, None].expand(b, m, t, n).reshape(b * m, t, n),
            prior_b.bit_logits[:, None].expand(b, m, t, n, 4).reshape(b * m, t, n, 4),
        )
        z_bm, tau_bm, inf_bm, x_bm, v_bm, cllr_bm = diagonal_ici_demap(
            y_bm, G_bm, nv_bm, valid_bm, prior_bm, obs_c.pilot_mask, obs_c.pilots
        )

        # Optional Direction 1A: inject in-branch tangent-Laplace epistemic variance into tau & posterior
        if use_tangent_laplace:
            v_epi_m = _compute_epistemic_var(amp_bm, mean_rot_bm, basis_bm, cpe_bm, ops_bm, x_bm, v_bm, weights_bm)
            diag_pow = G_bm.diagonal(dim1=-2, dim2=-1).abs().square().clamp_min(1e-12)
            # Propagate epistemic CFR variance into symbol cavity variance tau_eff = tau + v_epi / |h|^2 * E|x|^2
            ex2 = (prior_bm.mean.abs().square() + prior_bm.variance)
            tau_epi = tau_bm + (v_epi_m.reshape(b * m, t, n) * ex2 / diag_pow)
            x_epi, v_epi_post = _qam_moments(prior_bm.bit_logits, z_bm, tau_epi, inf_bm)
            x_epi = torch.where(obs_c.pilot_mask, obs_c.pilots, x_epi)
            v_epi_post = torch.where(obs_c.pilot_mask, 0.0, v_epi_post)
            from p5.unrolled_em_transformer import _llrs
            cllr_epi = torch.where(inf_bm[..., None] & (~obs_c.pilot_mask)[..., None], _llrs(z_bm, tau_epi), 0.0)
        else:
            v_epi_m = torch.zeros((b, m, t, n), dtype=REAL, device=y.device)
            x_epi, v_epi_post, cllr_epi = x_bm, v_bm, cllr_bm

        pred_full_bm = torch.einsum("btfk,btk->btf", G_bm, x_bm)[:, None]
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

        feat = routing_features(h_bm, h_bar, y_obs_bm, x_bm, pred_obs_bm, y_bar, res_obs_bm, anchor_bm, path_fields)
        fields, route_fields, _ = model._fields(feat, at_initial=(not model.learned), return_attn=False)
        du, dw = route_fields[:, 0].unbind(-1)
        h_energy = h_bm[:, 0].abs().square().mean((-2, -1)).clamp_min(1e-300)
        route_step = model.gamma_init * (
            (h_bm[:, 0].conj() * du + res_obs_bm[:, 0].conj() * dw).real.mean((-2, -1)) / h_energy
        ).reshape(b, m)
        delta_route_acc = delta_route_acc + route_step

        clean_bm = y_bm - (pred_full_bm[:, 0] - h_bm[:, 0] * x_bm)
        bank_res = None
        if step >= 1:
            bank_res = _sector_chirp_update(
                clean_bm[:, None], x_bm, nominal, rot_bm, valid_bm,
                fields[..., 1] * valid_f, model.cue_weight, step, q_bm, model.multi_hyp
            )

        phase_step, delta_h_bm = _phase_step(fields, h_bm, amp_bm, mean_rot_bm, cpe_bm, nominal, bank_res)
        turn = torch.exp(1j * phase_step)
        mean_rot_bm = mean_rot_bm + (rot_bm * (turn - 1.)).mean(2)
        rot_bm = rot_bm * turn
        ops_bm = frequency_operators_fast(basis_bm, rot_bm, cpe_bm)

        ell_p, ell_c = branch_evidence(
            y[:, 0], G_bm.reshape(b, m, t, n, n), x_bm.reshape(b, m, t, n), v_bm.reshape(b, m, t, n),
            z_bm.reshape(b, m, t, n), tau_bm.reshape(b, m, t, n),
            prior_b.bit_logits[:, None].expand(b, m, t, n, 4), nv, obs_c.pilot_mask, valid
        )
        logits_pi = gamma_phys * ell_p + gamma_const * ell_c + delta_route_acc
        log_pi = F.log_softmax(logits_pi, dim=-1) if m > 1 else logits_pi.new_zeros((b, 1))
        pi = log_pi.exp() if m > 1 else torch.ones((b, 1), dtype=REAL, device=y.device)

        mix_llr, _, sym_mean, total_var, _, _ = mixture_posterior(
            pi, log_pi, cllr_epi.reshape(b, m, t, n, 4),
            x_epi.reshape(b, m, t, n), v_epi_post.reshape(b, m, t, n)
        )
        traces.append(dict(step=step, mean=sym_mean.detach(), variance=total_var.detach()))

        amp_sol = weighted_amplitude_solve(x_bm, v_bm, y_bm[:, None], ops_bm, weights_bm, ridge_bm, delta_h_bm)
        amp_bm = amp_sol.transpose(-1, -2)
        G_bm = (torch.einsum("btpfk,brp->brtfk", ops_bm, amp_bm).reshape(b * m, t, n, n)
                + torch.diag_embed(delta_h_bm[:, 0]))
        h_bm = G_bm.diagonal(dim1=-2, dim2=-1)[:, None]
        prev_mix_llr = mix_llr

    # Final exact cavity readout
    cav = cavity_demap(y_bm, G_bm, nv_bm, weights_bm, prior_bm, obs_c.pilot_mask, obs_c.pilots)
    z_fin, tau_fin, inf_fin = cav.z, cav.variance, cav.informed
    x_fin, v_fin, cllr_fin = cav.mean, cav.posterior_variance, cav.channel_llr

    if use_tangent_laplace:
        v_epi_fin = _compute_epistemic_var(amp_bm, mean_rot_bm, basis_bm, cpe_bm, ops_bm, x_fin, v_fin, weights_bm)
        diag_pow = G_bm.diagonal(dim1=-2, dim2=-1).abs().square().clamp_min(1e-12)
        ex2 = prior_bm.mean.abs().square() + prior_bm.variance
        tau_fin_eff = tau_fin + (v_epi_fin.reshape(b * m, t, n) * ex2 / diag_pow)
        x_fin_eff, v_fin_eff = _qam_moments(prior_bm.bit_logits, z_fin, tau_fin_eff, inf_fin)
        x_fin_eff = torch.where(obs_c.pilot_mask, obs_c.pilots, x_fin_eff)
        v_fin_eff = torch.where(obs_c.pilot_mask, 0.0, v_fin_eff)
        from p5.unrolled_em_transformer import _llrs
        cllr_fin_eff = torch.where(inf_fin[..., None] & (~obs_c.pilot_mask)[..., None], _llrs(z_fin, tau_fin_eff), 0.0)
    else:
        x_fin_eff, v_fin_eff, cllr_fin_eff = x_fin, v_fin, cllr_fin

    G_view = G_bm.reshape(b, m, t, n, n)
    ell_p, ell_c = branch_evidence(
        y[:, 0], G_view, x_fin.reshape(b, m, t, n), v_fin.reshape(b, m, t, n),
        z_fin.reshape(b, m, t, n), tau_fin.reshape(b, m, t, n),
        prior_b.bit_logits[:, None].expand(b, m, t, n, 4), nv, obs_c.pilot_mask, valid
    )
    logits_pi = gamma_phys * ell_p + gamma_const * ell_c + delta_route_acc
    log_pi = F.log_softmax(logits_pi, dim=-1) if m > 1 else logits_pi.new_zeros((b, 1))
    pi = log_pi.exp() if m > 1 else torch.ones((b, 1), dtype=REAL, device=y.device)

    bit_llr, bit_disagree, sym_mean, sym_var_tot, sym_var_in, sym_var_bet = mixture_posterior(
        pi, log_pi, cllr_fin_eff.reshape(b, m, t, n, 4),
        x_fin_eff.reshape(b, m, t, n), v_fin_eff.reshape(b, m, t, n)
    )
    informed_any = inf_fin.reshape(b, m, t, n).any(1) & has_pilot[:, None, None]
    bit_llr = torch.where(informed_any[..., None] & (~obs_c.pilot_mask)[None, ..., None], bit_llr, 0.)
    bit_disagree = torch.where(informed_any[..., None] & (~obs_c.pilot_mask)[None, ..., None], bit_disagree, 0.)
    h_hat = (pi[:, :, None, None] * G_view.diagonal(dim1=-2, dim2=-1)).sum(1)[:, None] * scale[:, None, None, None]
    h_hat = torch.where(has_pilot[:, None, None, None], h_hat, 0.)

    pred_y_norm = torch.einsum("bmtfk,bmtk->bmtf", G_view, x_fin.reshape(b, m, t, n))
    nv_safe = nv[:, None, None, None].clamp_min(1e-12)
    pred_var_norm = nv_safe + torch.einsum("bmtfk,bmtk->bmtf", G_view.abs().square(), v_fin.reshape(b, m, t, n))
    branch_pred_y = pred_y_norm * scale[:, None, None, None]
    branch_pred_var = pred_var_norm * scale.abs().square()[:, None, None, None]
    pred_y = (pi[:, :, None, None] * branch_pred_y).sum(1)
    pred_var = (pi[:, :, None, None] * (branch_pred_var + (branch_pred_y - pred_y[:, None]).abs().square())).sum(1)

    out = EM7Output(
        bit_llr=bit_llr, h_hat=h_hat, pi=pi, log_pi=log_pi,
        symbol_mean=sym_mean, symbol_var_total=sym_var_tot,
        symbol_var_within=sym_var_in, symbol_var_between=sym_var_bet,
        bit_disagreement=bit_disagree, pred_y=pred_y, pred_var=pred_var,
        branch_pred_y=branch_pred_y, branch_pred_var=branch_pred_var,
        traces=traces, ell_phys=ell_p, ell_const=ell_c, delta_route=delta_route_acc,
    )
    state = dict(
        y_norm=y, nv_norm=nv, scale=scale, valid=valid, has_pilot=has_pilot,
        ops_bm=ops_bm, G_bm=G_bm, amp_bm=amp_bm, delta_h_bm=delta_h_bm,
        ridge_bm=ridge_bm, prior_b=prior_b, prior_bm=prior_bm,
        x_fin=x_fin, v_fin=v_fin, logits_pi=logits_pi,
        delta_route_acc=delta_route_acc, gamma_phys=gamma_phys, gamma_const=gamma_const,
    )
    return out, state


@torch.no_grad()
def assimilate_probes(model, state, obs_full, gain2, probe_mask, probe_weight=1.0):
    """Direction 1B: True Bayesian sequential assimilation of newly revealed probes into Pass-1 state.
    Zero restart, zero noisy 4-pilot pinv re-initialization."""
    obs_c2, valid2 = canonicalize(obs_full, gain2)
    scale = state["scale"]
    y2_norm = (obs_c2.y.to(COMPLEX) / scale.abs()[:, None, None, None]) * (scale / scale.abs()).conj()[:, None, None, None]
    y2_norm = torch.where(valid2[:, None], y2_norm, 0.)
    nv = state["nv_norm"]
    b, _, t, n = y2_norm.shape
    m = 4 if model.multi_hyp else 1

    G_view = state["G_bm"].reshape(b, m, t, n, n)
    x_m = state["x_fin"].reshape(b, m, t, n)
    v_m = state["v_fin"].reshape(b, m, t, n)

    # 1. Exact Bayesian marginal log-likelihood increment on newly revealed probe bins ONLY
    pred_p = torch.einsum("bmtfk,bmtk->bmtf", G_view, x_m)
    resid2 = (y2_norm[:, 0][:, None] - pred_p).abs().square()
    S_var = nv[:, None, None, None].clamp_min(1e-12) + torch.einsum("bmtfk,bmtk->bmtf", G_view.abs().square(), v_m)
    pm = probe_mask[:, None].to(REAL)
    delta_log_lik = -((resid2 / S_var + torch.log(S_var / nv[:, None, None, None].clamp_min(1e-12))) * pm).sum((-2, -1))

    # 2. Assimilate new observations into amplitude fit (keeping Pass 1 converged rot/ops & symbol moments!)
    weights2_bm = valid2[:, None].expand(b, m, t, n).reshape(b * m, t, n).to(REAL)
    y2_bm = y2_norm[:, 0:1].expand(b, m, t, n).reshape(b * m, t, n)
    nv_bm = nv[:, None].expand(b, m).reshape(b * m)

    amp_sol = weighted_amplitude_solve(
        state["x_fin"], state["v_fin"], y2_bm[:, None], state["ops_bm"],
        weights2_bm, state["ridge_bm"], state["delta_h_bm"]
    )
    amp2_bm = amp_sol.transpose(-1, -2)
    G2_bm = (torch.einsum("btpfk,brp->brtfk", state["ops_bm"], amp2_bm).reshape(b * m, t, n, n)
             + torch.diag_embed(state["delta_h_bm"][:, 0]))
    G2_view = G2_bm.reshape(b, m, t, n, n)

    # 3. Exact cavity demap on assimilated G2 and valid2 with Pass-1 extrinsic prior
    cav2 = cavity_demap(y2_bm, G2_bm, nv_bm, weights2_bm, state["prior_bm"], obs_c2.pilot_mask, obs_c2.pilots)
    x2_m = cav2.mean.reshape(b, m, t, n)
    v2_m = cav2.posterior_variance.reshape(b, m, t, n)
    cllr2_m = cav2.channel_llr.reshape(b, m, t, n, 4)

    # Combine Pass-1 evidence + exact probe marginal log-likelihood increment (posterior proportional to prior * lik_new)
    logits_pi2 = state["logits_pi"] + probe_weight * delta_log_lik
    log_pi2 = F.log_softmax(logits_pi2, dim=-1) if m > 1 else logits_pi2.new_zeros((b, 1))
    pi2 = log_pi2.exp() if m > 1 else torch.ones((b, 1), dtype=REAL, device=y2_norm.device)

    bit_llr, bit_disagree, sym_mean, sym_var_tot, sym_var_in, sym_var_bet = mixture_posterior(
        pi2, log_pi2, cllr2_m, x2_m, v2_m
    )
    informed_any = cav2.informed.reshape(b, m, t, n).any(1) & state["has_pilot"][:, None, None]
    bit_llr = torch.where(informed_any[..., None] & (~obs_c2.pilot_mask)[None, ..., None], bit_llr, 0.)
    bit_disagree = torch.where(informed_any[..., None] & (~obs_c2.pilot_mask)[None, ..., None], bit_disagree, 0.)
    h_hat = (pi2[:, :, None, None] * G2_view.diagonal(dim1=-2, dim2=-1)).sum(1)[:, None] * scale[:, None, None, None]
    h_hat = torch.where(state["has_pilot"][:, None, None, None], h_hat, 0.)

    z = torch.zeros(1, dtype=REAL)
    return EM7Output(
        bit_llr=bit_llr, h_hat=h_hat, pi=pi2, log_pi=log_pi2,
        symbol_mean=sym_mean, symbol_var_total=sym_var_tot,
        symbol_var_within=sym_var_in, symbol_var_between=sym_var_bet,
        bit_disagreement=bit_disagree, pred_y=z, pred_var=z,
        branch_pred_y=z, branch_pred_var=z, traces=[], ell_phys=z, ell_const=z, delta_route=z,
    )


def _banded_ici_diagnostic():
    """Measure exact Frobenius energy fraction captured by subcarrier bandwidth B_ici on Off-Grid R2."""
    b, t, n = 32, 20, N
    gen = torch.Generator().manual_seed(808001)
    p_max = 5
    delays = torch.zeros((b, p_max), dtype=REAL)
    gains = torch.zeros((b, p_max), dtype=COMPLEX)
    nu_s = torch.zeros((b, p_max), dtype=REAL)
    nu_e = torch.zeros((b, p_max), dtype=REAL)
    for i in range(b):
        p_cnt = int(torch.randint(2, 6, (1,), generator=gen).item())
        d = torch.empty(p_cnt, dtype=REAL)
        d[0] = 1.5 * float(torch.rand(1, generator=gen).item())
        # Sorted separated continuous delays in [0.5, 11.5]
        pts = torch.sort(0.5 + 11.0 * torch.rand(p_cnt - 1, dtype=REAL, generator=gen)).values
        d[1:] = pts
        pw = torch.exp(-0.45 * torch.arange(p_cnt, dtype=REAL))
        pw = pw / pw.sum()
        g_real = torch.randn(p_cnt, dtype=REAL, generator=gen)
        g_imag = torch.randn(p_cnt, dtype=REAL, generator=gen)
        g = torch.complex(g_real, g_imag) * (pw / 2.0).sqrt()
        g[0] = pw[0].sqrt() * torch.polar(torch.tensor(1.0, dtype=REAL), (2 * math.pi * torch.rand(1, dtype=REAL, generator=gen) - math.pi)[0])
        signs = 2.0 * torch.randint(2, (p_cnt,), generator=gen).double() - 1.0
        ns = signs * (0.035 + 0.050 * torch.rand(p_cnt, dtype=REAL, generator=gen))
        ne = signs * (0.035 + 0.050 * torch.rand(p_cnt, dtype=REAL, generator=gen))
        delays[i, :p_cnt] = d
        gains[i, :p_cnt] = g
        nu_s[i, :p_cnt] = ns
        nu_e[i, :p_cnt] = ne

    clock = (torch.arange(t, dtype=REAL)[:, None] * (n + CP) + CP + torch.arange(n, dtype=REAL)) / n
    duration = (t * (n + CP) - 1) / n
    phase = ray_phase(nu_s, nu_e, clock.flatten(), duration)
    rot = torch.polar(torch.ones_like(phase), phase).reshape(b, p_max, t, n).permute(0, 2, 3, 1)
    basis = frequency_basis(delays, n)
    ops = frequency_operators_fast(basis, rot, torch.ones((b, t), dtype=COMPLEX))
    G_true = torch.einsum("btpfk,bp->btfk", ops, gains)  # B, T, N, N

    f_idx = torch.arange(n)[:, None]
    k_idx = torch.arange(n)[None, :]
    circ_dist = torch.minimum((f_idx - k_idx) % n, (k_idx - f_idx) % n)
    tot_e = float(G_true.abs().square().sum())
    by_bw = {}
    for bw in (0, 1, 2, 3, 4, 5):
        mask = circ_dist <= bw
        cap = float((G_true[:, :, mask]).abs().square().sum())
        by_bw[f"B_ici={bw}"] = cap / tot_e
    return by_bw


def main():
    setup()
    t0 = time.perf_counter()
    fixed_model = EM7HypothesisReceiver("fixed").eval()

    # Track C2 tangent-Laplace before vs after & C5 restart vs true assimilation
    c2_stats = {k: dict(blks=0, err=0, bce=0.0, bits=0, b0_e=0.0, b0_v=0.0, t_e=0.0, t_v=0.0)
                for k in ("EM7-Axial_base", "EM7-Axial_laplace_reg4", "EM7-Axial_laplace_reg1")}
    c5_stats = {k: dict(p1_err=0, p2_err=0, p1_bce=0.0, p2_bce=0.0, bits=0, hi_ps=0.0, lo_ps=0.0, hi_fl=0.0, lo_fl=0.0, hi_n=0, lo_n=0)
                for k in ("EM7-Fixed_restart", "EM7-Fixed_assimilate", "EM7-Axial_restart", "EM7-Axial_assimilate")}

    for pair in DEV_PAIRS:
        ax_model, _ = load_checkpoint(REPO / f"p7/results/dev/pair{pair}/axial.pt")
        for b_idx in range(4):
            frame = dev_frame(2, 6, pair, 8, batch=b_idx)
            cond_seed = dev_seed(2, 6, pair, 100 + b_idx)

            # 1A: C2_row0_erased
            g_c2, _, _ = mask_for_condition(frame.obs.pilot_mask, "C2_row0_erased", 8, seed=cond_seed)
            obs_c2 = apply_observation_gain(frame, g_c2).obs
            for name, use_lap, reg in (
                ("EM7-Axial_base", False, 4.0),
                ("EM7-Axial_laplace_reg4", True, 4.0),
                ("EM7-Axial_laplace_reg1", True, 1.0),
            ):
                out, _ = _forward_with_state(ax_model, obs_c2, g_c2, use_tangent_laplace=use_lap, prior_phi_reg=reg)
                sc = decoding_metrics(frame, out.bit_llr, out.h_hat)
                cal = calibration(frame, out.bit_llr, out.traces)
                s2 = cal["steps"][2]
                st = c2_stats[name]
                st["blks"] += sc["blocks"]
                st["err"] += sc["block_errors"]
                st["bce"] += sc["bce_sum"]
                st["bits"] += sc["bit_count"]
                st["b0_e"] += s2["bins"][0]["sum_squared_error"]
                st["b0_v"] += s2["bins"][0]["sum_predicted_variance"]
                st["t_e"] += s2["total"]["sum_squared_error"]
                st["t_v"] += s2["total"]["sum_predicted_variance"]

            # 1B: C5_holdout_update (Pass 1 -> Pass 2 restart vs true assimilation)
            g1, g2, probe = mask_for_condition(frame.obs.pilot_mask, "C5_holdout_update", 8, seed=cond_seed)
            obs1 = apply_observation_gain(frame, g1).obs
            obs2 = apply_observation_gain(frame, g2).obs
            for tag, mdl in (("EM7-Fixed", fixed_model), ("EM7-Axial", ax_model)):
                out1, st1 = _forward_with_state(mdl, obs1, g1, use_tangent_laplace=False)
                out2_rst, _ = _forward_with_state(mdl, obs2, g2, use_tangent_laplace=False)
                out2_asm = assimilate_probes(mdl, st1, obs2, g2, probe, probe_weight=1.0)
                for mode, o2 in ((f"{tag}_restart", out2_rst), (f"{tag}_assimilate", out2_asm)):
                    seq = sequential_update_metrics(frame, out1, o2)
                    st = c5_stats[mode]
                    st["p1_err"] += seq["pass1_block_errors"]
                    st["p2_err"] += seq["pass2_block_errors"]
                    st["p1_bce"] += seq["pass1_score"]["bce_sum"]
                    st["p2_bce"] += seq["pass2_score"]["bce_sum"]
                    st["bits"] += seq["pass1_score"]["bit_count"]
                    st["hi_ps"] += seq["high_disagreement_prob_shift_sum"]
                    st["lo_ps"] += seq["low_disagreement_prob_shift_sum"]
                    st["hi_fl"] += seq["high_disagreement_flip_sum"]
                    st["lo_fl"] += seq["low_disagreement_flip_sum"]
                    st["hi_n"] += seq["high_count"]
                    st["lo_n"] += seq["low_count"]

    c2_summary = {}
    for k, st in c2_stats.items():
        c2_summary[k] = dict(
            block_errors=st["err"], blocks=st["blks"], bler=st["err"] / st["blks"],
            bit_bce=st["bce"] / st["bits"],
            bin0_ratio=st["b0_e"] / max(st["b0_v"], 1e-300),
            total_ratio=st["t_e"] / max(st["t_v"], 1e-300),
        )

    c5_summary = {}
    for k, st in c5_stats.items():
        ps_hi = st["hi_ps"] / max(st["hi_n"], 1)
        ps_lo = st["lo_ps"] / max(st["lo_n"], 1)
        fl_hi = st["hi_fl"] / max(st["hi_n"], 1)
        fl_lo = st["lo_fl"] / max(st["lo_n"], 1)
        c5_summary[k] = dict(
            pass1_errors=st["p1_err"], pass2_errors=st["p2_err"],
            delta_blocks=st["p1_err"] - st["p2_err"],
            pass1_bce=st["p1_bce"] / st["bits"], pass2_bce=st["p2_bce"] / st["bits"],
            delta_bce=(st["p1_bce"] - st["p2_bce"]) / st["bits"],
            prob_shift_ratio=ps_hi / max(ps_lo, 1e-12),
            flip_ratio=fl_hi / max(fl_lo, 1e-12),
        )

    banded = _banded_ici_diagnostic()
    report = dict(
        elapsed_seconds=time.perf_counter() - t0,
        c2_tangent_laplace=c2_summary,
        c5_sequential_assimilation=c5_summary,
        offgrid_banded_ici_energy=banded,
    )
    ex.save(RESULTS / "stage80_r2_fixes.json", report)

    print("\n=== Stage 8.0A: C2_row0_erased Tangent-Laplace Continuous Uncertainty (dev 1728 blks) ===", flush=True)
    for k, v in c2_summary.items():
        print(f"  {k:26s} BLER={v['block_errors']:4d}/{v['blocks']} ({100*v['bler']:.2f}%) "
              f"BCE={v['bit_bce']:.4f} | <0.01={v['bin0_ratio']:.2f}x | total={v['total_ratio']:.3f}x", flush=True)

    print("\n=== Stage 8.0B: C5_holdout_update Restart vs True Bayesian Assimilation (dev 1728 blks) ===", flush=True)
    for k, v in c5_summary.items():
        print(f"  {k:26s} Pass1={v['pass1_errors']:4d} -> Pass2={v['pass2_errors']:4d} "
              f"(delta={v['delta_blocks']:+4d} blks, dBCE={v['delta_bce']:+.5f}) | "
              f"prob_ratio={v['prob_shift_ratio']:.2f}x flip_ratio={v['flip_ratio']:.2f}x", flush=True)

    print("\n=== Stage 8.0C: Off-Grid Banded-ICI Frobenius Energy Capture ===", flush=True)
    for k, val in banded.items():
        print(f"  {k}: {100*val:.3f}%", flush=True)


if __name__ == "__main__":
    main()
