"""Unified Phase 8 EM8ManifoldReceiver: Dictionary-Free Multi-Chart C*-Equivariant Receiver with
Unitary Parallel-Transport Attention, Cross-Hypothesis Jeffreys Routing, Blind Student-t Precision,
Continuous Epistemic Variance, and True Bayesian Sequential Assimilation."""
import copy
import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from p6_consistency.reliability import SymbolPrior
from p7.hypotheses import mixture_posterior
from p8_manifold.manifold_charts import (
    B_ICI,
    assemble_banded_operator,
    grid_coordinates,
    init_blind_pilot_weights,
    init_offgrid_charts,
    kalman_gp_assimilate_charts,
    make_offgrid_code_prior,
    offgrid_branch_evidence,
    offgrid_cavity_demap,
    offgrid_diagonal_demap,
    offgrid_pilot_gauge,
    student_t_pilot_weights,
)
from p8_manifold.transport_attention import (
    CoherentTransportLayer,
    CrossHypothesisJeffreysLayer,
    _cmat,
    _safe_rms,
    jeffreys_divergence,
)


REAL = torch.float64
COMPLEX = torch.complex128
C_IN = 8


class EM8Output(NamedTuple):
    bit_llr: torch.Tensor               # [B, T, N, 4]
    h_hat: torch.Tensor                 # [B, 1, T, N] complex128
    pi: torch.Tensor                    # [B, M]
    log_pi: torch.Tensor                # [B, M]
    symbol_mean: torch.Tensor           # [B, T, N] complex128
    symbol_var_total: torch.Tensor      # [B, T, N] float64
    symbol_var_within: torch.Tensor     # [B, T, N] float64
    symbol_var_between: torch.Tensor    # [B, T, N] float64
    bit_disagreement: torch.Tensor      # [B, T, N, 4] float64
    pred_y: torch.Tensor                # [B, T, N] complex128
    pred_var: torch.Tensor              # [B, T, N] float64 (total = aleatoric + epistemic + between)
    pred_var_aleatoric: torch.Tensor    # [B, T, N] float64
    pred_var_epistemic: torch.Tensor    # [B, T, N] float64
    pred_var_between: torch.Tensor      # [B, T, N] float64
    branch_pred_y: torch.Tensor         # [B, M, T, N] complex128
    branch_pred_var: torch.Tensor       # [B, M, T, N] float64
    pilot_weights: torch.Tensor         # [B, T, N] float64 inferred observation precision
    traces: list
    state: dict


class _EM8Backbone(nn.Module):
    """Centered C*-equivariant backbone combining Construction A (Coherent Transport) & B (Cross-Hyp Jeffreys)."""

    def __init__(self, variant="transport", d_model=16, n_layers=2, n_heads=4):
        super().__init__()
        self.embed = _cmat(C_IN, d_model)
        self.layers = nn.ModuleList([CoherentTransportLayer(variant, d_model, n_heads) for _ in range(n_layers)])
        self.cross_hyp = CrossHypothesisJeffreysLayer(d_model, n_heads)
        self.read_h = _cmat(d_model, 1 + 2 * B_ICI)  # diagonal delta_H + 4 banded off-diagonal ICI taps
        self.read_epi = _cmat(d_model, 1)            # scale-invariant epistemic log-variance refinement

    def forward(self, feat_bm, coords_grid, p_t_idx, p_f_idx, p_coords, p_w, pred_y_m, pred_var_m, h_m, res_obs_m, w_grid, b, m):
        bm, t, n, _ = feat_bm.shape
        unit_f, rms_f = _safe_rms(feat_bm, dim=-1, keepdim=True)
        state = (unit_f @ self.embed) * rms_f  # BM, T, N, D (1-homogeneous)

        # Compute per-site Jeffreys divergence between each chart and mixture predictive distribution
        mu_bar = pred_y_m.mean(dim=1, keepdim=True)
        v_bar = (pred_var_m + (pred_y_m - mu_bar).abs().square()).mean(dim=1, keepdim=True)
        dj_grid = torch.log1p(jeffreys_divergence(pred_y_m, pred_var_m, mu_bar, v_bar))  # B, M, T, N
        dj_bm = dj_grid.reshape(bm, t, n)
        w_bm = w_grid[:, None].expand(b, m, t, n).reshape(bm, t, n)
        grid_ev_bias = torch.log(w_bm.clamp_min(1e-6)) + dj_bm  # BM, T, N (held-out bins strongly negative)

        # Gather observed pilot states & Jeffreys divergences (zeroed where p_w == 0 -> ZERO LEAKAGE)
        p_t_bm = p_t_idx[:, None].expand(b, m, -1).reshape(bm, -1)
        p_f_bm = p_f_idx[:, None].expand(b, m, -1).reshape(bm, -1)
        p_c_bm = p_coords[:, None].expand(b, m, -1, -1).reshape(bm, -1, 2)
        p_w_bm = p_w[:, None].expand(b, m, -1).reshape(bm, -1)
        bm_idx = torch.arange(bm, device=state.device)[:, None]

        p_dj_bm = dj_bm[bm_idx, p_t_bm, p_f_bm] * (p_w_bm > 0).to(REAL)
        for layer in self.layers:
            p_state = state[bm_idx, p_t_bm, p_f_bm] * (p_w_bm > 0).to(COMPLEX)[..., None]
            state = layer(state, coords_grid, p_state, p_c_bm, p_w_bm, p_dj_bm, grid_ev_bias=grid_ev_bias)

        # Construction B: Cross-Hypothesis Jeffreys routing across M charts
        state_m = state.reshape(b, m, t, n, -1)
        state_m, route_delta, _ = self.cross_hyp(state_m, pred_y_m, pred_var_m, h_m, res_obs_m, w_grid)
        state = state_m.reshape(bm, t, n, -1)

        u_s, r_s = _safe_rms(state, dim=-1, keepdim=True)
        raw_op = (u_s @ self.read_h) * r_s  # BM, T, N, 1 + 2*B_ICI (1-homogeneous!)
        raw_epi_c = u_s @ self.read_epi     # unit-scale complex -> .real is NOT U(1)-invariant unless inner product!
        # Strictly C*-invariant real epistemic log-delta via Hermitian inner product with h_m!
        h_flat = h_m.reshape(bm, t, n)
        h_u, _ = _safe_rms(h_flat[..., None], dim=-1, keepdim=True)
        epi_delta = (h_u.conj() * raw_epi_c).real.squeeze(-1)  # BM, T, N (strictly C*-invariant!)
        return raw_op, epi_delta, route_delta


class EM8ManifoldReceiver(nn.Module):
    """Unified Off-Grid C*-equivariant multi-chart EM receiver."""

    def __init__(self, variant="transport", n_hyp=4, d_model=16, n_layers=2, n_heads=4, steps=2, gamma_init=10.0):
        super().__init__()
        if variant not in ("transport", "abs_no_transport", "local", "fixed", "single_transport"):
            raise ValueError(f"Unknown EM8 variant: {variant}")
        self.variant = variant
        self.n_hyp = 1 if variant == "single_transport" else int(n_hyp)
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.steps = steps
        self.learned = variant != "fixed"
        self.gamma_init = float(gamma_init)

        if self.learned:
            block_var = "transport" if variant == "single_transport" else variant
            self.backbone = _EM8Backbone(block_var, d_model=d_model, n_layers=n_layers, n_heads=n_heads)
            self.initial_map = copy.deepcopy(self.backbone)
            for p in self.initial_map.parameters():
                p.requires_grad_(False)
            self.fusion_scale = nn.Parameter(torch.zeros(3, dtype=REAL))  # phys, const, route scales

    def real_parameters(self):
        return sum(p.numel() * (2 if p.is_complex() else 1) for p in self.parameters() if p.requires_grad)

    def _run_backbone(self, *args, at_initial=False):
        if not self.learned:
            feat_bm = args[0]
            b, m = args[-2], args[-1]
            bm, t, n, _ = feat_bm.shape
            z_op = feat_bm.new_zeros(bm, t, n, 1 + 2 * B_ICI)
            z_ep = feat_bm.real.new_zeros(bm, t, n)
            z_rt = feat_bm.real.new_zeros(b, m)
            return z_op, z_ep, z_rt
        op_live, ep_live, rt_live = self.backbone(*args)
        if at_initial:
            return op_live - op_live.detach(), ep_live - ep_live.detach(), rt_live - rt_live.detach()
        with torch.no_grad():
            det_args = [a.detach() if torch.is_tensor(a) else a for a in args]
            op_0, ep_0, rt_0 = self.initial_map(*det_args)
        return op_live - op_0, ep_live - ep_0, rt_live - rt_0

    def _build_features(self, H_m, G_view, y_norm, x_m, pilot_mask, pilots, w_grid, pi):
        """Construct C_in=8 strictly 1-homogeneous complex grid features [BM, T, N, 8]."""
        b, m, t, n = H_m.shape
        pred_full = torch.einsum("bmtfk,bmtk->bmtf", G_view, x_m)
        w_m = w_grid[:, None]  # B, 1, T, N (0 on held-out probes!)
        y_obs = (y_norm[:, 0][:, None] * w_m).expand(b, m, t, n)
        pred_obs = pred_full * w_m
        res_obs = y_obs - pred_obs

        # Explicit observed pilot CFR feature (0 on payload & held-out probes)
        p_c = (pilot_mask.to(REAL) * w_grid)[:, None]  # B, 1, T, N
        h_pilot_obs = (y_norm[:, 0] * pilots.to(COMPLEX).conj())[:, None] * p_c
        h_pilot_obs = h_pilot_obs.expand(b, m, t, n)

        h_bar = (pi[:, :, None, None] * H_m).sum(1, keepdim=True)
        y_bar = (pi[:, :, None, None] * pred_obs).sum(1, keepdim=True)
        dis_h = H_m - h_bar
        dis_y = pred_obs - y_bar
        h_x = H_m * x_m

        feat = torch.stack((H_m, y_obs, h_pilot_obs, pred_obs, res_obs, dis_h, dis_y, h_x), dim=-1)
        return feat.reshape(b * m, t, n, C_IN), res_obs, pred_full

    def _apply_refinement(self, H_m, ici_m, v_epi_0, raw_op, epi_delta):
        """Apply bounded, strictly C*-equivariant operator & epistemic variance refinement."""
        b, m, t, n = H_m.shape
        h_scale = H_m.abs().mean((-2, -1), keepdim=True).clamp_min(1e-12)  # B, M, 1, 1
        dH_raw = raw_op[..., 0].reshape(b, m, t, n)
        dI_raw = raw_op[..., 1:].reshape(b, m, t, n, 2 * B_ICI)
        # Smooth complex soft-clip preserving exact phase & 1-homogeneity
        # Off-diagonal ICI cap 0.05 matches physical Doppler ICI rms (~0.032) and prevents hallucinated interference
        dH = 0.65 * dH_raw / (1.0 + (dH_raw.abs() / h_scale).square()).sqrt()
        dI = 0.05 * dI_raw / (1.0 + (dI_raw.abs() / h_scale[..., None]).square()).sqrt()
        H_new = H_m + dH
        ici_new = ici_m + dI
        # Cap epistemic shrinkage at 2x (2^tanh(delta) in [0.5, 2.0]) to prevent <0.01 overconfidence collapse
        epi_mult = (math.log(2.0) * torch.tanh(epi_delta.reshape(b, m, t, n))).exp()
        v_epi_new = (v_epi_0 * epi_mult).clamp_min(1e-6)
        return H_new, ici_new, v_epi_new

    def _finalize_readout(
        self, y_norm, nv_norm, scale_c, obs_pilot_mask, obs_pilots, w_grid,
        H_m, ici_m, v_epi_m, prior_b, prior_bm, logits_route, gamma_phys, gamma_const, gamma_route,
        traces, p_coords, p_t_idx, p_f_idx, p_h_obs, p_w, blind, override_logits_pi=None,
    ):
        b, m, t, n = H_m.shape
        bm = b * m
        G_bm = assemble_banded_operator(H_m.reshape(bm, t, n), ici_m.reshape(bm, t, n, 2 * B_ICI))
        G_view = G_bm.reshape(b, m, t, n, n)
        y_bm = y_norm[:, 0:1].expand(b, m, t, n).reshape(bm, t, n)
        nv_bm = nv_norm[:, None].expand(b, m).reshape(bm)
        w_bm = w_grid[:, None].expand(b, m, t, n).reshape(bm, t, n)
        pm_bm = obs_pilot_mask[:, None].expand(b, m, t, n).reshape(bm, t, n)
        pil_bm = obs_pilots.to(COMPLEX)[:, None].expand(b, m, t, n).reshape(bm, t, n)

        z_f, tau_f, inf_f, x_f, v_f, cllr_f = offgrid_cavity_demap(
            y_bm, G_bm, nv_bm, v_epi_m.reshape(bm, t, n), w_bm, prior_bm, pm_bm, pil_bm
        )
        z_m = z_f.reshape(b, m, t, n)
        tau_m = tau_f.reshape(b, m, t, n)
        inf_m = inf_f.reshape(b, m, t, n)
        x_m = x_f.reshape(b, m, t, n)
        v_m = v_f.reshape(b, m, t, n)
        cllr_m = cllr_f.reshape(b, m, t, n, 4)

        if override_logits_pi is not None:
            logits_pi = override_logits_pi
        else:
            ell_p, ell_c = offgrid_branch_evidence(
                y_norm, G_view, x_m, v_m, v_epi_m, z_m, tau_m,
                prior_b.bit_logits[:, None].expand(b, m, t, n, 4), nv_norm, obs_pilot_mask, w_grid
            )
            logits_pi = gamma_phys * ell_p + gamma_const * ell_c + gamma_route * logits_route
        if m > 1:
            log_pi = F.log_softmax(logits_pi, dim=-1)
            pi = log_pi.exp()
        else:
            log_pi = logits_pi.new_zeros((b, 1))
            pi = torch.ones((b, 1), dtype=REAL, device=y_norm.device)

        bit_llr, bit_dis, sym_mean, sym_var_tot, sym_var_in, sym_var_bet = mixture_posterior(
            pi, log_pi, cllr_m, x_m, v_m
        )
        informed_any = inf_m.any(1)
        bit_llr = torch.where(informed_any[..., None] & (~obs_pilot_mask)[..., None], bit_llr, 0.0)
        bit_dis = torch.where(informed_any[..., None] & (~obs_pilot_mask)[..., None], bit_dis, 0.0)

        h_hat = (pi[:, :, None, None] * H_m).sum(1)[:, None] * scale_c[:, None, None, None]

        # Exact 3-term predictive variance decomposition in receive units: total == aleatoric + epistemic + between
        pred_y_norm = torch.einsum("bmtfk,bmtk->bmtf", G_view, x_m)
        nv_safe = nv_norm[:, None, None, None].clamp_min(1e-300)
        ex2_m = x_m.abs().square() + v_m
        v_alea_norm = nv_safe + torch.einsum("bmtfk,bmtk->bmtf", G_view.abs().square(), v_m)
        v_epi_pred_norm = v_epi_m * ex2_m
        branch_pred_var_norm = v_alea_norm + v_epi_pred_norm

        s_sq = scale_c.abs().square()[:, None, None, None]
        branch_pred_y = pred_y_norm * scale_c[:, None, None, None]
        branch_pred_var = branch_pred_var_norm * s_sq

        pred_y = (pi[:, :, None, None] * branch_pred_y).sum(1)
        pred_var_aleatoric = (pi[:, :, None, None] * (v_alea_norm * s_sq)).sum(1)
        pred_var_epistemic = (pi[:, :, None, None] * (v_epi_pred_norm * s_sq)).sum(1)
        pred_var_between = (pi[:, :, None, None] * (branch_pred_y - pred_y[:, None]).abs().square()).sum(1)
        pred_var = pred_var_aleatoric + pred_var_epistemic + pred_var_between

        traces_out = list(traces) + [dict(step=len(traces), mean=sym_mean.detach(), variance=sym_var_tot.detach())]
        state = dict(
            y_norm=y_norm, nv_norm=nv_norm, scale_c=scale_c,
            pilot_mask=obs_pilot_mask, pilots=obs_pilots, w_grid=w_grid, blind=blind,
            H_m=H_m, ici_m=ici_m, v_epi_0=v_epi_m, v_epi_m=v_epi_m,
            G_view=G_view, x_m=x_m, v_m=v_m, prior_b=prior_b, prior_bm=prior_bm,
            pi=pi, log_pi=log_pi, logits_pi=logits_pi, logits_route=logits_route,
            p_coords=p_coords, p_t_idx=p_t_idx, p_f_idx=p_f_idx, p_h_obs=p_h_obs, p_w=p_w,
            traces=traces_out,
        )
        return EM8Output(
            bit_llr=bit_llr, h_hat=h_hat, pi=pi, log_pi=log_pi,
            symbol_mean=sym_mean, symbol_var_total=sym_var_tot,
            symbol_var_within=sym_var_in, symbol_var_between=sym_var_bet,
            bit_disagreement=bit_dis,
            pred_y=pred_y, pred_var=pred_var,
            pred_var_aleatoric=pred_var_aleatoric,
            pred_var_epistemic=pred_var_epistemic,
            pred_var_between=pred_var_between,
            branch_pred_y=branch_pred_y, branch_pred_var=branch_pred_var,
            pilot_weights=w_grid, traces=traces_out, state=state,
        )

    def forward(self, obs):
        y_norm, nv_norm, scale_c = offgrid_pilot_gauge(obs)
        b, _, t, n = y_norm.shape
        m = self.n_hyp
        bm = b * m
        coords_grid = grid_coordinates(t, n, y_norm.device)

        w_grid = init_blind_pilot_weights(y_norm, nv_norm, obs.pilot_mask, obs.valid_mask, blind=obs.blind)
        H_0, v_epi_0, p_coords, p_t_idx, p_f_idx, p_h_obs, p_w = init_offgrid_charts(
            y_norm, nv_norm, obs.pilot_mask, obs.pilots, w_grid, n_hyp=m
        )
        H_m = H_0
        v_epi_m = v_epi_0
        ici_m = torch.zeros((b, m, t, n, 2 * B_ICI), dtype=COMPLEX, device=y_norm.device)

        at_initial = (
            not self.learned
            or (
                all(torch.equal(p, q) for p, q in zip(self.backbone.parameters(), self.initial_map.parameters()))
                and bool((self.fusion_scale == 0).all())
            )
        )
        gamma_phys = self.gamma_init * (self.fusion_scale[0].exp() if self.learned else 1.0)
        gamma_const = self.gamma_init * (self.fusion_scale[1].exp() if self.learned else 1.0)
        gamma_route = self.gamma_init * (self.fusion_scale[2].exp() if self.learned else 1.0)

        pi = torch.full((b, m), 1.0 / m, dtype=REAL, device=y_norm.device)
        y_bm = y_norm[:, 0:1].expand(b, m, t, n).reshape(bm, t, n)
        nv_bm = nv_norm[:, None].expand(b, m).reshape(bm)
        pm_bm = obs.pilot_mask[:, None].expand(b, m, t, n).reshape(bm, t, n)
        pil_bm = obs.pilots.to(COMPLEX)[:, None].expand(b, m, t, n).reshape(bm, t, n)

        prev_mix_llr = None
        logits_route = y_norm.real.new_zeros(b, m)
        traces = []

        for step in range(self.steps):
            prior_b = make_offgrid_code_prior(y_norm[:, 0], obs.pilot_mask, obs.pilots, prev_mix_llr, parity=(step > 0))
            prior_bm = SymbolPrior(
                prior_b.mean[:, None].expand(b, m, t, n).reshape(bm, t, n),
                prior_b.variance[:, None].expand(b, m, t, n).reshape(bm, t, n),
                prior_b.bit_logits[:, None].expand(b, m, t, n, 4).reshape(bm, t, n, 4),
            )
            G_bm = assemble_banded_operator(H_m.reshape(bm, t, n), ici_m.reshape(bm, t, n, 2 * B_ICI))
            G_view = G_bm.reshape(b, m, t, n, n)
            w_bm = w_grid[:, None].expand(b, m, t, n).reshape(bm, t, n)

            z_bm, tau_bm, inf_bm, x_bm, v_bm, cllr_bm = offgrid_diagonal_demap(
                y_bm, G_bm, nv_bm, v_epi_m.reshape(bm, t, n), w_bm, prior_bm, pm_bm, pil_bm
            )
            x_m = x_bm.reshape(b, m, t, n)
            v_m = v_bm.reshape(b, m, t, n)
            z_m = z_bm.reshape(b, m, t, n)
            tau_m = tau_bm.reshape(b, m, t, n)
            cllr_m = cllr_bm.reshape(b, m, t, n, 4)

            feat_bm, res_obs_m, pred_full_m = self._build_features(
                H_m, G_view, y_norm, x_m, obs.pilot_mask, obs.pilots, w_grid, pi
            )
            ex2_m = x_m.abs().square() + v_m
            pred_var_m = (
                nv_norm[:, None, None, None].clamp_min(1e-12)
                + v_epi_m * ex2_m
                + torch.einsum("bmtfk,bmtk->bmtf", G_view.abs().square(), v_m)
            )
            raw_op, epi_delta, route_step = self._run_backbone(
                feat_bm, coords_grid, p_t_idx, p_f_idx, p_coords, p_w,
                pred_full_m, pred_var_m, H_m, res_obs_m, w_grid, b, m,
                at_initial=at_initial,
            )
            logits_route = logits_route + route_step
            H_m, ici_m, v_epi_m = self._apply_refinement(H_m, ici_m, v_epi_0, raw_op, epi_delta)

            # Update branch posterior & LDPC extrinsic feedback
            ell_p, ell_c = offgrid_branch_evidence(
                y_norm, G_view, x_m, v_m, v_epi_m, z_m, tau_m,
                prior_b.bit_logits[:, None].expand(b, m, t, n, 4), nv_norm, obs.pilot_mask, w_grid
            )
            logits_pi = gamma_phys * ell_p + gamma_const * ell_c + gamma_route * logits_route
            log_pi = F.log_softmax(logits_pi, dim=-1) if m > 1 else logits_pi.new_zeros((b, 1))
            pi = log_pi.exp() if m > 1 else torch.ones((b, 1), dtype=REAL, device=y_norm.device)

            mix_llr, _, sym_mean, sym_var_tot, _, _ = mixture_posterior(pi, log_pi, cllr_m, x_m, v_m)
            traces.append(dict(step=step, mean=sym_mean.detach(), variance=sym_var_tot.detach()))
            prev_mix_llr = mix_llr

            # Direction 1.3: Blind Student-t E-step updates pilot precision weights from mixture prediction!
            if obs.blind:
                pred_y_mix = (pi[:, :, None, None] * pred_full_m).sum(1)
                pred_var_mix = (pi[:, :, None, None] * (pred_var_m + (pred_full_m - pred_y_mix[:, None]).abs().square())).sum(1)
                w_grid = student_t_pilot_weights(
                    y_norm, pred_y_mix.detach(), pred_var_mix.detach(), nv_norm,
                    obs.pilot_mask, obs.valid_mask, blind=True, nu=3.0
                )
                # Refresh p_w from updated w_grid
                b_idx = torch.arange(b, device=y_norm.device)[:, None]
                p_w = w_grid[b_idx, p_t_idx, p_f_idx]

        # Final exact cavity readout with step-(steps-1) extrinsic code prior
        prior_fin = make_offgrid_code_prior(y_norm[:, 0], obs.pilot_mask, obs.pilots, prev_mix_llr, parity=True)
        prior_fin_bm = SymbolPrior(
            prior_fin.mean[:, None].expand(b, m, t, n).reshape(bm, t, n),
            prior_fin.variance[:, None].expand(b, m, t, n).reshape(bm, t, n),
            prior_fin.bit_logits[:, None].expand(b, m, t, n, 4).reshape(bm, t, n, 4),
        )
        return self._finalize_readout(
            y_norm, nv_norm, scale_c, obs.pilot_mask, obs.pilots, w_grid,
            H_m, ici_m, v_epi_m, prior_fin, prior_fin_bm, logits_route,
            gamma_phys, gamma_const, gamma_route, traces,
            p_coords, p_t_idx, p_f_idx, p_h_obs, p_w, obs.blind,
        )

    def assimilate(self, state, y_new_full, reveal_mask, probe_weight=0.10):
        """True Bayesian sequential assimilation of newly revealed probe observations into Pass-1 state.

        Zero restart & zero out-of-distribution 3rd neural step: preserves Pass-1 canonical gauge scale_c
        and converged neural chart operators (H_m, ici_m, v_epi_m), then applies ONE unified Gaussian
        observation model (accounting for off-diagonal ICI) for BOTH:
          1. Exact Schur-product PSD Kalman/GP conditional update on (H_m, v_epi_m),
          2. Exact Bayes-rule posterior update log pi_m^+ = log_softmax(logits_pi^(1) + probe_weight * delta_ll_m),
          3. Exact epistemic-aware Gaussian cavity readout.
        Exact bit-identity when reveal_mask is empty for BOTH learned and fixed receivers.
        """
        scale_c = state["scale_c"]
        nv_norm = state["nv_norm"]
        pilot_mask = state["pilot_mask"]
        pilots = state["pilots"]
        H_m = state["H_m"]
        ici_m = state["ici_m"]
        v_epi_m = state["v_epi_m"]
        G_view = state["G_view"]
        x_m = state["x_m"]
        v_m = state["v_m"]
        w_grid_1 = state["w_grid"]
        b = H_m.shape[0]

        # Normalize ONLY the revealed probe observations in Pass-1's canonical C* gauge (zero leakage!)
        s_abs = scale_c.abs()[:, None, None, None]
        u_conj = (scale_c / scale_c.abs()).conj()[:, None, None, None]
        y_rev_norm = (y_new_full.to(COMPLEX) / s_abs) * u_conj
        y_norm_2 = torch.where(reveal_mask[:, None], y_rev_norm, state["y_norm"])
        w_grid_2 = torch.where(reveal_mask, torch.ones_like(w_grid_1), w_grid_1)

        # Unified ICI-aware exact GP posterior Schur-complement Kalman update + joint probe log-likelihood
        p_t_idx, p_f_idx, p_coords = state["p_t_idx"], state["p_f_idx"], state["p_coords"]
        H_2, v_epi_2, delta_ll = kalman_gp_assimilate_charts(
            H_m, v_epi_m, G_view, x_m, v_m, y_norm_2, nv_norm, pilots, reveal_mask,
            p_t_idx, p_f_idx, state["p_w"], state["p_h_obs"],
        )

        b_idx = torch.arange(b, device=y_norm_2.device)[:, None]
        p_w_2 = w_grid_2[b_idx, p_t_idx, p_f_idx]
        p_h_obs_2 = torch.where(
            p_w_2 > 0,
            y_norm_2[:, 0][b_idx, p_t_idx, p_f_idx] * pilots.to(COMPLEX)[b_idx, p_t_idx, p_f_idx].conj(),
            torch.zeros_like(state["p_h_obs"]),
        )

        # Exact Bayes rule: log pi^+ = log_softmax(logits_pi^(1) + probe_weight * delta_ll)
        logits_pi_2 = state["logits_pi"] + probe_weight * delta_ll

        gamma_phys = self.gamma_init * (self.fusion_scale[0].exp() if self.learned else 1.0)
        gamma_const = self.gamma_init * (self.fusion_scale[1].exp() if self.learned else 1.0)
        gamma_route = self.gamma_init * (self.fusion_scale[2].exp() if self.learned else 1.0)

        return self._finalize_readout(
            y_norm_2, nv_norm, scale_c, pilot_mask, pilots, w_grid_2,
            H_2, ici_m, v_epi_2, state["prior_b"], state["prior_bm"],
            state["logits_route"],
            gamma_phys, gamma_const, gamma_route, state["traces"][:-1],
            p_coords, p_t_idx, p_f_idx, p_h_obs_2, p_w_2, state["blind"],
            override_logits_pi=logits_pi_2,
        )
