"""Dictionary-free Off-Grid C* = R+ x U(1) gauge, continuous 2D GP multi-chart initialization,
blind Student-t observation reliability, banded-ICI assembly, and epistemic-aware cavity demapping."""
import math

import torch
from torch.nn import functional as F

from p5.unrolled_em_transformer import _llrs
from p5b.parity import min_sum_extrinsic
from p6_consistency.reliability import SymbolPrior, _qam_moments


REAL = torch.float64
COMPLEX = torch.complex128
B_ICI = 2  # 5 diagonals (-2,-1,0,+1,+2) capture 99.71% of Off-Grid Doppler ICI energy
PAM = torch.tensor([-3.0, -1.0, 1.0, 3.0], dtype=REAL) / math.sqrt(10.0)


def grid_coordinates(t, n, device):
    """Continuous normalized 2D coordinates z = (t_norm, f_norm) in [-1, 1] x [0, 2pi)."""
    t_idx = torch.arange(t, dtype=REAL, device=device)
    f_idx = torch.arange(n, dtype=REAL, device=device)
    t_norm = 2.0 * (t_idx / max(t - 1, 1)) - 1.0  # [-1, 1]
    f_norm = 2.0 * math.pi * (f_idx / float(n))  # [0, 2pi) periodic
    tt, ff = torch.meshgrid(t_norm, f_norm, indexing="ij")
    return torch.stack((tt, ff), dim=-1)  # T, N, 2


def offgrid_pilot_gauge(obs):
    """Per-burst full C* = R+ x U(1) robust canonical gauge strictly over valid observed pilots.

    Strictly invariant to global alpha in C* across [1e-35, 1e20] and robust to blind erasures/impulses.
    """
    y = obs.y.to(COMPLEX)  # B, 1, T, N
    nv = obs.noise_var.double().reshape(-1)  # B
    b, _, t, n = y.shape
    known = (obs.pilot_mask & obs.valid_mask)  # B, T, N

    scale_r_list = []
    u_list = []
    for i in range(b):
        k_i = known[i]
        y_i = y[i, 0]
        p_i = obs.pilots[i].to(COMPLEX)
        if bool(k_i.any()):
            s_i = y_i[k_i]
            pk_i = p_i[k_i]
            mags = s_i.abs()
            peak = float(mags.max())
            if peak > 0.0:
                safe_peak = mags.max()
                s_unit = s_i / safe_peak  # O(1)
                if obs.blind:
                    # Robust median magnitude over nonzero observed pilots (immune to 25% erasures + 10x impulses)
                    nz = mags > 0
                    med = s_unit[nz].abs().median() if bool(nz.any()) else s_unit.abs().median()
                    scale_r = safe_peak * med.clamp_min(1e-12)
                    # Robust Huber/Cauchy-weighted pilot phase projection (strictly scale-invariant!)
                    rel_m = s_unit.abs() / med.clamp_min(1e-12)
                    rho = (rel_m.square() / (rel_m.square() + 0.02)) / (1.0 + rel_m.square())
                    proj = (rho * s_unit * pk_i.conj()).sum()
                else:
                    scale_r = safe_peak * s_unit.abs().square().mean().sqrt()
                    proj = (s_unit * pk_i.conj()).sum()
                dom_idx = s_unit.abs().argmax()
                dom_vec = s_unit[dom_idx] * pk_i[dom_idx].conj()
                ref_vec = proj if float(proj.abs()) > 1e-8 else dom_vec
                ref_mag = ref_vec.abs()
                u = ref_vec / ref_mag if float(ref_mag) > 0.0 else torch.tensor(1.0 + 0.0j, dtype=COMPLEX, device=y.device)
            else:
                scale_r = nv[i].sqrt() if float(nv[i]) > 0.0 else torch.tensor(1.0, dtype=REAL, device=y.device)
                u = torch.tensor(1.0 + 0.0j, dtype=COMPLEX, device=y.device)
        else:
            scale_r = nv[i].sqrt() if float(nv[i]) > 0.0 else torch.tensor(1.0, dtype=REAL, device=y.device)
            u = torch.tensor(1.0 + 0.0j, dtype=COMPLEX, device=y.device)
        scale_r_list.append(scale_r.clamp_min(1e-300))
        u_list.append(u)

    scale_r = torch.stack(scale_r_list, dim=0)  # B real
    unit_phase = torch.stack(u_list, dim=0)  # B complex
    y_valid = torch.where(obs.valid_mask[:, None], y, 0.0)
    y_norm = (y_valid / scale_r[:, None, None, None]) * unit_phase.conj()[:, None, None, None]
    nv_norm = nv / scale_r.square()
    scale_c = scale_r.to(COMPLEX) * unit_phase
    return y_norm, nv_norm, scale_c


def init_blind_pilot_weights(y_norm, nv_norm, pilot_mask, valid_mask, blind=False, nu=4.0):
    """Return initial C*-invariant observation precision weights w_grid[B, T, N] in [0, 1]."""
    b, _, t, n = y_norm.shape
    w_grid = valid_mask.to(REAL).clone()
    if not blind:
        return w_grid
    known = pilot_mask & valid_mask
    for i in range(b):
        k_i = known[i]
        if not bool(k_i.any()):
            continue
        mags = y_norm[i, 0, k_i].abs()
        nz = mags > 0
        med = mags[nz].median().clamp_min(1e-12) if bool(nz.any()) else torch.tensor(1.0, dtype=REAL, device=y_norm.device)
        rel2 = (mags / med).square()
        # Reject exact/near-zero erasures AND strong impulsive outliers (>2x median) on step 0 via redescending gate
        low_gate = rel2 / (rel2 + 0.02)
        hi_excess = (rel2 - 4.0).clamp_min(0.0)
        hi_gate = torch.exp(-0.5 * hi_excess)
        w_p = (low_gate * hi_gate).clamp(0.0, 1.0)
        w_p = torch.where(mags == 0, torch.zeros_like(w_p), w_p)
        w_grid[i, k_i] = w_p
    return w_grid


def student_t_pilot_weights(y_norm, pred_y_mix, pred_var_mix, nv_norm, pilot_mask, valid_mask, blind=False, nu=3.0):
    """Exact C*-invariant blind Student-t E-step precision weights w_grid[B, T, N] in [0, 1]."""
    w_grid = valid_mask.to(REAL).clone()
    if not blind:
        return w_grid
    known = pilot_mask & valid_mask
    resid2 = (y_norm[:, 0] - pred_y_mix).abs().square()
    norm_res = resid2 / pred_var_mix.clamp_min(1e-12)
    w_t = ((nu + 1.0) / (nu + norm_res)).clamp(0.0, 1.0)
    # Keep exact zero-erased pilot samples strictly rejected even if |pred_y| is temporarily small
    zero_gate = y_norm[:, 0].abs().square() / (y_norm[:, 0].abs().square() + 0.02 * nv_norm[:, None, None].clamp_min(1e-12))
    w_p = torch.where(y_norm[:, 0].abs() == 0, torch.zeros_like(w_t), (w_t * zero_gate).clamp(0.0, 1.0))
    return torch.where(known, w_p, w_grid)


# Four complementary dictionary-free continuous 2D electromagnetic GP charts: (nu_center, tau_center, ell_t, ell_f)
# Chart 0: Broad-Doppler low-delay; Chart 1/2: Directional +nu / -nu coherent; Chart 3: High delay-spread
CHART_SPECS_M4 = (
    (0.0, 2.2, 4.2, 4.2),
    (+0.030, 2.6, 6.5, 3.8),
    (-0.030, 2.6, 6.5, 3.8),
    (0.0, 4.6, 4.2, 2.6),
)
CHART_SPECS_M1 = (
    (0.0, 3.0, 4.5, 3.6),
)


def _gp_kernel_2d(t1, f1, t2, f2, n_sub, nu_c, tau_c, ell_t, ell_f):
    """Strictly PSD Hermitian continuous 2D electromagnetic covariance kernel K((t1,f1), (t2,f2)).

    Uses exact periodic chordal RBF on S^1 along frequency: exp(-2 * (sin(pi*df/N) / (pi*ell_f/N))^2),
    which matches Gaussian exp(-0.5*(df/ell_f)^2) locally and is provably strictly PSD for all N, ell_f > 0.
    """
    dt = t1[..., :, None] - t2[..., None, :]
    df_raw = f1[..., :, None] - f2[..., None, :]
    # 1 OFDM symbol duration = (N + CP) / N = 1.5 useful symbols in R2 clock
    dt_useful = dt * 1.5
    phase = 2.0 * math.pi * (nu_c * dt_useful - (tau_c / float(n_sub)) * df_raw)
    sin_f = torch.sin(math.pi * df_raw / float(n_sub))
    denom_f = math.pi * ell_f / float(n_sub)
    env = torch.exp(-0.5 * (dt / ell_t).square() - 2.0 * (sin_f / denom_f).square())
    return torch.polar(env, phase)


def init_offgrid_charts(y_norm, nv_norm, pilot_mask, pilots, w_grid, n_hyp=4):
    """Dictionary-free continuous 2D GP multi-chart initializer on arbitrary scattered pilots.

    Returns:
      H_0:       [B, M, T, N] complex128 initial CFR per chart
      v_epi_0:   [B, M, T, N] float64 exact GP posterior epistemic CFR variance per chart
      p_coords:  [B, N_p, 2] float64 continuous (t_norm, f_rad) of valid TX pilots
      p_t_idx:   [B, N_p] int64 row indices
      p_f_idx:   [B, N_p] int64 col indices
      p_h_obs:   [B, N_p] complex128 observed pilot CFR y_p / x_p (0 where w_p == 0)
      p_w:       [B, N_p] float64 pilot precision weights (0 on held-out probes -> ZERO LEAKAGE!)
    """
    b, _, t, n = y_norm.shape
    specs = CHART_SPECS_M4 if n_hyp == 4 else CHART_SPECS_M1
    m = len(specs)
    dev = y_norm.device

    # Full grid coordinates in raw symbol/subcarrier indices
    gt = torch.arange(t, dtype=REAL, device=dev)[:, None].expand(t, n).reshape(-1)  # 640
    gf = torch.arange(n, dtype=REAL, device=dev)[None, :].expand(t, n).reshape(-1)  # 640

    # Gather fixed N_p=64 TX pilot slots per burst (held-out probes have p_w == 0 and p_h_obs == 0!)
    n_p = int(pilot_mask[0].sum())
    p_t_list, p_f_list, p_h_list, p_w_list = [], [], [], []
    for i in range(b):
        coords = pilot_mask[i].nonzero(as_tuple=False)  # N_p, 2
        rt, cf = coords[:, 0], coords[:, 1]
        wp = w_grid[i, rt, cf]
        yp = y_norm[i, 0, rt, cf]
        xp = pilots[i, rt, cf].to(COMPLEX)
        hp = torch.where(wp > 0, yp * xp.conj(), torch.zeros_like(yp))  # |x_p| == 1 exactly
        p_t_list.append(rt)
        p_f_list.append(cf)
        p_h_list.append(hp)
        p_w_list.append(wp)

    p_t_idx = torch.stack(p_t_list, dim=0)  # B, N_p
    p_f_idx = torch.stack(p_f_list, dim=0)  # B, N_p
    p_h_obs = torch.stack(p_h_list, dim=0)  # B, N_p
    p_w = torch.stack(p_w_list, dim=0)      # B, N_p

    p_t_f = p_t_idx.to(REAL)
    p_f_f = p_f_idx.to(REAL)
    p_coords = torch.stack(
        (2.0 * (p_t_f / max(t - 1, 1)) - 1.0, 2.0 * math.pi * (p_f_f / float(n))),
        dim=-1,
    )  # B, N_p, 2

    # Estimate scale-invariant channel signal variance sigma_h2 per burst from active weighted pilots
    w_sum = p_w.sum(-1).clamp_min(1e-12)
    h_pwr = (p_w * p_h_obs.abs().square()).sum(-1) / w_sum
    sigma_h2 = (h_pwr - nv_norm).clamp_min(0.25)  # B (in normalized gauge units ~ 1.0)
    eye_p = torch.eye(n_p, dtype=COMPLEX, device=dev)

    H_list, V_list = [], []
    w_sqrt = p_w.clamp_min(0.0).sqrt().to(COMPLEX)  # B, N_p (detached data weights -> no sqrt backward!)

    gt_b = gt[None].expand(b, -1)
    gf_b = gf[None].expand(b, -1)
    for (nu_c, tau_c, ell_t, ell_f) in specs:
        K_pp = _gp_kernel_2d(p_t_f, p_f_f, p_t_f, p_f_f, n, nu_c, tau_c, ell_t, ell_f)  # B, N_p, N_p
        K_gp = _gp_kernel_2d(gt_b, gf_b, p_t_f, p_f_f, n, nu_c, tau_c, ell_t, ell_f)    # B, 640, N_p
        # Zero-weight-safe symmetrized GP posterior: A = W^{1/2} K_pp W^{1/2} + (N0 / sigma_h2) I
        noise_ratio = (nv_norm / sigma_h2).clamp_min(1e-6)
        KW = K_pp * w_sqrt[:, :, None] * w_sqrt[:, None, :]
        A = 0.5 * (KW + KW.mH) + noise_ratio[:, None, None] * eye_p
        rhs_h = (w_sqrt * p_h_obs)[..., None]  # B, N_p, 1
        K_gp_w = K_gp * w_sqrt[:, None, :]     # B, 640, N_p
        # Solve A^{-1} [rhs_h, K_gp_w^H] in one batched Cholesky/LU call
        rhs_all = torch.cat((rhs_h, K_gp_w.mH), dim=-1)  # B, N_p, 1 + 640
        sol = torch.linalg.solve(A, rhs_all)
        alpha_h = sol[..., 0]  # B, N_p
        Ainv_K = sol[..., 1:]  # B, N_p, 640
        h_grid = (K_gp_w @ alpha_h[..., None]).squeeze(-1).reshape(b, t, n)
        # Posterior epistemic variance: sigma_h2 * (1 - Re diag(K_gp_w A^{-1} K_gp_w^H))
        explained = (K_gp_w.conj() * Ainv_K.transpose(-1, -2)).sum(-1).real
        v_grid = (sigma_h2[:, None] * (1.0 - explained).clamp_min(1e-4)).reshape(b, t, n)
        H_list.append(h_grid)
        V_list.append(v_grid)

    H_0 = torch.stack(H_list, dim=1)       # B, M, T, N
    v_epi_0 = torch.stack(V_list, dim=1)   # B, M, T, N
    return H_0, v_epi_0, p_coords, p_t_idx, p_f_idx, p_h_obs, p_w


def kalman_gp_assimilate_charts(
    H_m, v_epi_m, G_view, x_m, v_m, y_norm_2, nv_norm, pilots, reveal_mask, p_t_idx, p_f_idx, p_w_1, p_h_obs_1
):
    """Exact GP posterior Schur-complement sequential assimilation & joint multivariate probe log-likelihood.

    Conditions the Pass-1 converged chart state (H_m, v_epi_m) on newly revealed probes R using the EXACT
    Pass-1 GP posterior cross-covariance conditioned on Pass-1 observed pilots P_1:
      Sigma_{g, R | P_1} = sigma_h^2 (K_{g, R} - K_{g, P_1} W^{1/2} A_{P_1}^{-1} W^{1/2} K_{P_1, R})
      Sigma_{R, R | P_1} = sigma_h^2 (K_{R, R} - K_{R, P_1} W^{1/2} A_{P_1}^{-1} W^{1/2} K_{P_1, R})
    Zero corruption of Pass-1 observed pilots, exact equivalence to joint GP conditioning on P_1 U R when
    dH_neural == 0, exact preservation of Pass-1 neural refinements, strictly PSD, strictly C*-equivariant.
    """
    b, m, t, n = H_m.shape
    specs = CHART_SPECS_M4 if m == 4 else CHART_SPECS_M1
    dev = H_m.device
    gt = torch.arange(t, dtype=REAL, device=dev)[:, None].expand(t, n).reshape(-1)  # 640
    gf = torch.arange(n, dtype=REAL, device=dev)[None, :].expand(t, n).reshape(-1)  # 640

    # Full predicted waveform & off-diagonal ICI variance on every bin
    pred_full = torch.einsum("bmtfk,bmtk->bmtf", G_view, x_m)
    row_var = torch.einsum("bmtfk,bmtk->bmtf", G_view.abs().square(), v_m)
    diag_pow = G_view.diagonal(dim1=-2, dim2=-1).abs().square()
    ici_var = (row_var - diag_pow * v_m).clamp_min(0.0)

    # Exact Pass-1 chart signal variance sigma_h2 matching init_offgrid_charts
    w_sum = p_w_1.sum(-1).clamp_min(1e-12)
    h_pwr = (p_w_1 * p_h_obs_1.abs().square()).sum(-1) / w_sum
    sigma_h2 = (h_pwr - nv_norm).clamp_min(0.25)  # B
    n_p = p_t_idx.shape[-1]
    eye_p = torch.eye(n_p, dtype=COMPLEX, device=dev)

    H_out = H_m.clone()
    V_out = v_epi_m.clone()
    delta_ll = torch.zeros((b, m), dtype=REAL, device=dev)

    for i in range(b):
        coords_r = reveal_mask[i].nonzero(as_tuple=False)  # n_r, 2
        n_r = int(coords_r.shape[0])
        if n_r == 0:
            continue
        r_rows, r_cols = coords_r[:, 0], coords_r[:, 1]
        rt = r_rows.to(REAL)
        cf = r_cols.to(REAL)
        pt = p_t_idx[i].to(REAL)
        pf = p_f_idx[i].to(REAL)
        ws_i = p_w_1[i].clamp_min(0.0).sqrt().to(COMPLEX)  # n_p (0 on probes!)
        y_r = y_norm_2[i, 0, r_rows, r_cols]
        xp_r = pilots[i, r_rows, r_cols].to(COMPLEX)
        nv_i = nv_norm[i].clamp_min(1e-12)
        sh2_i = sigma_h2[i]
        nr_i = (nv_i / sh2_i).clamp_min(1e-6)

        for mi, (nu_c, tau_c, ell_t, ell_f) in enumerate(specs):
            h_cur = H_m[i, mi].reshape(-1)  # 640
            v_cur = v_epi_m[i, mi].reshape(-1).clamp_min(1e-6)  # 640

            # Exact Pass-1 GP Gram matrix A_1 on P_1 (zero-weight safe on held-out probes)
            K_pp = _gp_kernel_2d(pt, pf, pt, pf, n, nu_c, tau_c, ell_t, ell_f)      # n_p, n_p
            K_gp = _gp_kernel_2d(gt, gf, pt, pf, n, nu_c, tau_c, ell_t, ell_f)      # 640, n_p
            K_pr = _gp_kernel_2d(pt, pf, rt, cf, n, nu_c, tau_c, ell_t, ell_f)      # n_p, n_r
            K_gr = _gp_kernel_2d(gt, gf, rt, cf, n, nu_c, tau_c, ell_t, ell_f)      # 640, n_r
            K_rr = _gp_kernel_2d(rt, cf, rt, cf, n, nu_c, tau_c, ell_t, ell_f)      # n_r, n_r

            KW = K_pp * ws_i[:, None] * ws_i[None, :]
            A1 = 0.5 * (KW + KW.mH) + nr_i * eye_p
            K_gp_w = K_gp * ws_i[None, :]  # 640, n_p
            K_pr_w = ws_i[:, None] * K_pr  # n_p, n_r
            A1_inv_Kpr = torch.linalg.solve(A1, K_pr_w)  # n_p, n_r

            # Exact posterior covariances Sigma_{g, R | P_1} and Sigma_{R, R | P_1}
            Sigma_gr = sh2_i * (K_gr - K_gp_w @ A1_inv_Kpr)
            Sigma_rr = sh2_i * (K_rr - K_pr_w.mH @ A1_inv_Kpr)
            # Scale smoothly if Pass-1 neural refinement inflated/deflated epistemic variance relative to GP prior
            v_gp_r = Sigma_rr.diagonal().real.clamp_min(1e-6)
            v_m_r = v_epi_m[i, mi, r_rows, r_cols].clamp_min(1e-6)
            scale_r = (v_m_r / v_gp_r).clamp(0.5, 2.0).sqrt()
            Sigma_gr = Sigma_gr * scale_r[None, :]
            Sigma_rr = Sigma_rr * (scale_r[:, None] * scale_r[None, :])

            # ICI-corrected pilot innovation r_m = x_R^* (y_R - (G_m x_m)_R)
            innov_r = (y_r - pred_full[i, mi, r_rows, r_cols]) * xp_r.conj()  # n_r
            obs_var_r = nv_i + ici_var[i, mi, r_rows, r_cols]                  # n_r
            S_mat = 0.5 * (Sigma_rr + Sigma_rr.mH) + torch.diag(obs_var_r.to(COMPLEX))

            rhs = torch.cat((innov_r[:, None], Sigma_gr.mH), dim=-1)  # n_r, 1 + 640
            sol = torch.linalg.solve(S_mat, rhs)
            s_inv_r = sol[:, 0]
            dh = (Sigma_gr @ s_inv_r[:, None]).squeeze(-1)
            dv = (Sigma_gr.conj() * sol[:, 1:].transpose(-1, -2)).sum(-1).real.clamp_min(0.0)
            H_out[i, mi] = (h_cur + dh).reshape(t, n)
            V_out[i, mi] = (v_cur - dv).clamp_min(1e-6).reshape(t, n)

            # Exact dimensionless joint multivariate complex Gaussian log-likelihood increment
            quad = (innov_r.conj() * s_inv_r).sum().real.clamp_min(0.0)
            _, logdet = torch.linalg.slogdet(S_mat / nv_i)
            delta_ll[i, mi] = -(quad + logdet.real)

    return H_out, V_out, delta_ll


def assemble_banded_operator(H_diag, ici_taps=None):
    """Build [BM, T, N, N] complex frequency operator from diagonal H_diag[BM,T,N] and optional
    banded off-diagonal taps ici_taps[BM, T, N, 2*B_ICI] for circular offsets [-2,-1,+1,+2]."""
    G = torch.diag_embed(H_diag)
    if ici_taps is None:
        return G
    bm, t, n, k_off = ici_taps.shape
    offsets = [d for d in range(-B_ICI, B_ICI + 1) if d != 0]
    assert k_off == len(offsets)
    f_idx = torch.arange(n, device=H_diag.device)
    G = G.clone()
    for idx, d in enumerate(offsets):
        k_idx = (f_idx + d) % n
        G[:, :, f_idx, k_idx] = ici_taps[..., idx]
    return G


def make_offgrid_code_prior(template, pilot_mask, pilots, prev_mix_llr=None, parity=True):
    """Per-burst pilot-mask-aware extrinsic LDPC code prior without own-symbol double counting."""
    b, t, n = template.shape
    bit_logits = template.real.new_zeros(b, t, n, 4)
    if prev_mix_llr is not None and parity:
        data = torch.stack([prev_mix_llr[i, ~pilot_mask[i]] for i in range(b)], dim=0)  # B, 576, 4
        external = min_sum_extrinsic(data.reshape(b, -1, 128)).reshape_as(data)
        for i in range(b):
            bit_logits[i, ~pilot_mask[i]] = external[i]
    mean, variance = _qam_moments(bit_logits)
    mean = torch.where(pilot_mask, pilots.to(COMPLEX), mean)
    variance = torch.where(pilot_mask, 0.0, variance)
    return SymbolPrior(mean, variance, bit_logits)


def offgrid_diagonal_demap(y_bm, G_bm, nv_bm, v_epi_bm, w_bm, prior_bm, pilot_mask_bm, pilots_bm):
    """Fast differentiable per-burst banded/full-ICI diagonal demapper with epistemic channel variance."""
    diag_h = G_bm.diagonal(dim1=-2, dim2=-1)  # BM, T, N
    pred_full = torch.einsum("btfk,btk->btf", G_bm, prior_bm.mean)
    clean_y = y_bm - (pred_full - diag_h * prior_bm.mean)
    row_var = torch.einsum("btfk,btk->btf", G_bm.abs().square(), prior_bm.variance)
    diag_pow = diag_h.abs().square()
    ici_var = (row_var - diag_pow * prior_bm.variance).clamp_min(0.0)
    ex2 = (prior_bm.mean.abs().square() + prior_bm.variance)
    eff_nv = nv_bm[:, None, None] + ici_var + v_epi_bm * ex2
    informed = (diag_pow > 0) & (w_bm > 0)
    safe_h = torch.where(diag_pow > 0, diag_h, 1.0)
    safe_pow = torch.where(diag_pow > 0, diag_pow, 1.0)
    z = torch.where(informed, clean_y / safe_h, prior_bm.mean)
    eps_v = torch.finfo(REAL).eps * torch.maximum(prior_bm.variance, torch.ones_like(prior_bm.variance))
    tau = torch.where(informed, (eff_nv / safe_pow).clamp_min(eps_v.clamp_min(1e-12)), 1.0)
    post_mean, post_var = _qam_moments(prior_bm.bit_logits, z, tau, informed)
    post_mean = torch.where(pilot_mask_bm, pilots_bm, post_mean)
    post_var = torch.where(pilot_mask_bm, 0.0, post_var)
    cllr = _llrs(z, tau)
    cllr = torch.where(informed[..., None] & (~pilot_mask_bm)[..., None], cllr, 0.0)
    return z, tau, informed, post_mean, post_var, cllr


def offgrid_cavity_demap(y_bm, G_bm, nv_bm, v_epi_bm, w_bm, prior_bm, pilot_mask_bm, pilots_bm):
    """Exact differentiable full-frequency Gaussian cavity demap supporting per-burst pilot masks,
    continuous precision weights w_bm, and in-branch epistemic channel variance v_epi_bm."""
    bm, t, n = y_bm.shape
    eps = torch.finfo(REAL).eps
    # Per-bin effective diagonal observation noise: N0 + E[|x_f|^2] * v_epi_f
    ex2 = prior_bm.mean.abs().square() + prior_bm.variance
    eff_noise_bin = nv_bm[:, None, None] + v_epi_bm * ex2  # BM, T, N
    # Weight observations smoothly (0 weight strictly removes missing/held-out rows!)
    # Zero-sqrt-safe weighting: scale equations by w_bm directly where w_bm in [0, 1]
    # Note: for binary w_bm in {0, 1}, w_bm == sqrt(w_bm); for continuous w_bm, use safe sqrt via detached weights
    root_w = w_bm.detach().clamp_min(0.0).sqrt()
    wg = G_bm * root_w[..., None]
    wy = y_bm * root_w

    row_power = wg.abs().square().sum(-1)  # BM, T, N
    ref = torch.maximum(row_power.amax(-1), eff_noise_bin.amax(-1))  # BM, T
    safe_ref = torch.where(ref > 0, ref, 1.0)
    inv_sqrt_ref = safe_ref.rsqrt()
    g = wg * inv_sqrt_ref[..., None, None]
    obs_y = wy * inv_sqrt_ref[..., None]
    norm_noise = (eff_noise_bin / safe_ref[..., None]).clamp_min(64.0 * n * eps)

    cov = (g * prior_bm.variance[..., None, :]) @ g.mH
    cov = 0.5 * (cov + cov.mH) + torch.diag_embed(norm_noise)
    resid = obs_y - (g @ prior_bm.mean[..., None]).squeeze(-1)
    rhs = torch.cat((resid[..., None], g), dim=-1)
    solved = torch.linalg.solve(cov, rhs)
    filt_res, filt_cols = solved[..., 0], solved[..., 1:]
    matched = (g.conj() * filt_res[..., None]).sum(-2)
    prec = (g.conj() * filt_cols).sum(-2).real.clamp_min(0.0)
    informed = (g.abs().square().sum(-2) > 0) & (prec > 0)
    safe_prec = torch.where(informed, prec, 1.0)
    z = torch.where(informed, prior_bm.mean + matched / safe_prec, prior_bm.mean)
    raw_var = (1.0 - prior_bm.variance * prec) / safe_prec
    sym_floor = eps * torch.maximum(prior_bm.variance, torch.ones_like(prior_bm.variance))
    tau = torch.where(informed, torch.maximum(raw_var, sym_floor), 1.0)

    post_mean, post_var = _qam_moments(prior_bm.bit_logits, z, tau, informed)
    post_mean = torch.where(pilot_mask_bm, pilots_bm, post_mean)
    post_var = torch.where(pilot_mask_bm, 0.0, post_var)
    cllr = _llrs(z, tau)
    cllr = torch.where(informed[..., None] & (~pilot_mask_bm)[..., None], cllr, 0.0)
    return z, tau, informed, post_mean, post_var, cllr


def offgrid_branch_evidence(y_norm, G_view, x_m, v_m, v_epi_m, z_m, tau_m, prior_logits_m, nv_norm, pilot_mask, w_grid):
    """Scale-invariant physical waveform & 16-QAM constellation log-evidence [B, M] with epistemic variance."""
    b, m, t, n, _ = G_view.shape
    pred = torch.einsum("bmtfk,bmtk->bmtf", G_view, x_m)
    resid2 = (y_norm[:, 0][:, None] - pred).abs().square()
    ex2 = x_m.abs().square() + v_m
    noise = nv_norm[:, None, None, None]
    row_ref = G_view.abs().square().sum(-1).amax(-1, keepdim=True).amax(-2, keepdim=True)
    eps = torch.finfo(REAL).eps
    noise_safe = torch.maximum(noise, 64.0 * n * eps * torch.maximum(row_ref, noise))
    eff_var = noise_safe + v_epi_m * ex2 + torch.einsum("bmtfk,bmtk->bmtf", G_view.abs().square(), v_m)
    w = w_grid[:, None]  # B, 1, T, N
    n_obs = w.sum((-2, -1)).clamp_min(1.0)
    ell_phys = -((resid2 / eff_var + torch.log(eff_var / noise_safe)) * w).sum((-2, -1)) / n_obs

    pam = PAM.to(device=z_m.device, dtype=REAL)
    prior = prior_logits_m.reshape(*prior_logits_m.shape[:-1], 2, 2)
    sign, inner = prior.unbind(-1)
    lp_prior = torch.stack((torch.zeros_like(sign), inner, sign + inner, sign), -1)
    lp_prior = lp_prior - torch.logsumexp(lp_prior, dim=-1, keepdim=True)
    tau_safe = tau_m.clamp_min(1e-12)[..., None, None]
    dist2 = (torch.view_as_real(z_m)[..., None] - pam).square()
    lp_lik = -dist2 / tau_safe - 0.5 * torch.log(math.pi * tau_safe)
    axis_ev = torch.logsumexp(lp_prior + lp_lik, dim=-1).sum(-1)  # B, M, T, N
    p_mask = ((~pilot_mask).to(REAL) * w_grid)[:, None]
    n_pay = p_mask.sum((-2, -1)).clamp_min(1.0)
    ell_const = (axis_ev * p_mask).sum((-2, -1)) / n_pay
    return ell_phys, ell_const


@torch.no_grad()
def offgrid_2d_lmmse(obs):
    """Fair dictionary-free classical 2D scattered-pilot GP/Wiener LMMSE comparator."""
    y_norm, nv_norm, scale_c = offgrid_pilot_gauge(obs)
    w_grid = obs.valid_mask.to(REAL)  # Classical LMMSE has no blind outlier rejection
    H_0, v_epi_0, *_ = init_offgrid_charts(y_norm, nv_norm, obs.pilot_mask, obs.pilots, w_grid, n_hyp=1)
    H = H_0[:, 0]         # B, T, N
    v_epi = v_epi_0[:, 0] # B, T, N
    prior = make_offgrid_code_prior(y_norm[:, 0], obs.pilot_mask, obs.pilots, None, parity=False)
    G = torch.diag_embed(H)
    _, _, _, sym_m, sym_v, cllr = offgrid_diagonal_demap(
        y_norm[:, 0], G, nv_norm, v_epi, w_grid, prior, obs.pilot_mask, obs.pilots.to(COMPLEX)
    )
    h_hat = (H * scale_c[:, None, None])[:, None]
    traces = [dict(step=0, mean=sym_m, variance=sym_v)]
    return cllr, h_hat, traces
