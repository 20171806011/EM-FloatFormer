"""Shared zero-label-leakage physical multi-hypothesis frontend for Group B and Group C.

Constructs an identical R = 17 physical channel hypothesis bank across all 14 OFDM symbols
and 101 active subcarriers using ONLY:
  1. The current TTI's received complex IQ grid [14, 128]
  2. The visible pilot subcarrier indices (14 or 7) and known pilot sequence
  3. A frozen Train-only historical pilot-channel reference bank [M=16, 101] (strictly no Dev/Test data!)

Hypotheses (R = 17, condition-agnostic, same receiver configuration across C0..C4):
  - r = 0      : Classical single-TTI LS-linear static channel H_ls
  - r = 1..8   : 8 continuous linear phase-drift (CFO/Doppler) trajectories anchored at pilot sym 2:
                 omega_r in {-0.36, -0.27, -0.18, -0.09, +0.09, +0.18, +0.27, +0.36} rad/symbol
  - r = 9..15  : 7 discrete post-pilot phase-jump hypotheses:
                 phi_r in {-135, -90, -45, +45, +90, +135, 180} deg applied to symbols != pilot sym 2
  - r = 16     : Train-only historical content-addressed regularized channel H_ca

Also computes:
  - Strictly C*-invariant local excess-energy NBI burst variance guard nv_grid [14, 101]
  - Exact blind symbol-level marginalized 16-QAM log-likelihoods sym_ll [17, 14]
  - Strictly U(1)-equivariant complex features [14, 17, 4] and C*-invariant scalar features [14, 17, 10]
    shared byte-for-byte by Group C Role A (Physics Parallel-Transport Attention) and Role B (Non-Physics Baselines).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ofdm_data import (
    BIT_ONE_IDX,
    BIT_ZERO_IDX,
    CONST16,
    DATA_MASK_101,
    N_SYM,
    PILOT_SCS_14,
    PILOT_SYM,
    USED_SCS,
    estimate_noise_var_from_grid,
    ls_linear_channel,
    pilot_tx_for_scs,
)

# Shared 17-hypothesis physical bank constants
OMEGA_GRID: np.ndarray = np.array(
    [-0.36, -0.27, -0.18, -0.09, 0.09, 0.18, 0.27, 0.36], dtype=np.float64
)  # 8 CFO/Doppler slope hypotheses (rad/sym)
JUMP_DEG_GRID: np.ndarray = np.array(
    [-135.0, -90.0, -45.0, 45.0, 90.0, 135.0, 180.0], dtype=np.float64
)  # 7 discrete phase-jump hypotheses
JUMP_RAD_GRID: np.ndarray = np.deg2rad(JUMP_DEG_GRID)

R_HYP: int = 1 + len(OMEGA_GRID) + len(JUMP_RAD_GRID) + 1  # 17

# Prospective fixed physical prior over hypotheses (sum = 1.0, condition-agnostic)
HYP_PRIOR: np.ndarray = np.array(
    [0.28]
    + [0.28 / len(OMEGA_GRID)] * len(OMEGA_GRID)
    + [0.28 / len(JUMP_RAD_GRID)] * len(JUMP_RAD_GRID)
    + [0.16],
    dtype=np.float64,
)
HYP_LOG_PRIOR: np.ndarray = np.log(HYP_PRIOR)

# Per-hypothesis physical coordinate metadata for features
HYP_OMEGA: np.ndarray = np.concatenate(
    ([0.0], OMEGA_GRID, np.zeros(len(JUMP_RAD_GRID)), [0.0])
).astype(np.float64)
HYP_JUMP_RAD: np.ndarray = np.concatenate(
    ([0.0], np.zeros(len(OMEGA_GRID)), JUMP_RAD_GRID, [0.0])
).astype(np.float64)


def build_train_reference_bank(
    iq_train: np.ndarray,
    sinr_train: np.ndarray,
    m_refs: int = 16,
) -> np.ndarray:
    """Build a strictly Train-only M=16 historical pilot-channel reference bank [M, 101].

    Uses ONLY the 14 pilot observations of the highest-SINR, temporally stratified Train TTIs.
    NEVER touches data symbols, labels, Dev, or Test!
    """
    n_tr = len(sinr_train)
    edges = np.linspace(0, n_tr, m_refs + 1, dtype=int)
    refs = []
    for i in range(m_refs):
        s, e = int(edges[i]), int(edges[i + 1])
        # Pick highest-SINR TTI within each contiguous train segment (using only public sinr & pilot IQ)
        best_local = s + int(np.argmax(sinr_train[s:e]))
        h_ref = ls_linear_channel(iq_train[best_local], PILOT_SCS_14)
        # Canonicalize global phase to first subcarrier for clean storage (aligns per-query anyway)
        ph0 = np.angle(np.mean(h_ref))
        refs.append(h_ref * np.exp(-1j * ph0))
    return np.stack(refs, axis=0).astype(np.complex128)  # [M, 101]


def content_addressed_pilot_channel(
    y_grid: np.ndarray,
    pilot_scs: np.ndarray,
    ref_bank: np.ndarray,
    *,
    tau_ca: float = 1.0,
    lam_shrink: float = 0.55,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Global content-addressed channel estimate using U(1) phase-aligned Train reference bank.

    Returns:
      h_ca: [101] complex128 combined channel estimate
      alpha: [M] float64 softmax attention weights
      scores: [M] float64 pilot subspace match scores
    """
    h_ls = ls_linear_channel(y_grid, pilot_scs)
    p_tx = pilot_tx_for_scs(pilot_scs)
    hp_obs = y_grid[PILOT_SYM, pilot_scs] / p_tx  # [P]
    used_p_idx = np.searchsorted(USED_SCS, pilot_scs)
    ref_p = ref_bank[:, used_p_idx]               # [M, P]

    # Exact U(1) complex gain/phase alignment per reference j on visible pilots ONLY
    # Optimal complex scalar c_j = <ref_p[j], hp_obs> / ||ref_p[j]||^2 (gauge-equivariant!)
    # We use unit-modulus phase alignment + mild amplitude ratio for physical channel scaling
    inner = np.sum(np.conj(ref_p) * hp_obs[None, :], axis=1)  # [M]
    ref_norm2 = np.maximum(np.sum(np.abs(ref_p) ** 2, axis=1), 1e-12)
    c_opt = inner / ref_norm2  # [M] complex optimal alignment factor
    aligned_refs = c_opt[:, None] * ref_bank                  # [M, 101]
    resid_p = hp_obs[None, :] - aligned_refs[:, used_p_idx]   # [M, P]
    mse_p = np.mean(np.abs(resid_p) ** 2, axis=1)             # [M]
    n0_scale = max(float(np.mean(np.abs(hp_obs) ** 2)) * 0.05, 1e-6)
    scores = -mse_p / n0_scale                                # [M] higher is better

    scaled = scores / max(float(tau_ca), 1e-6)
    scaled -= float(np.max(scaled))
    w = np.exp(scaled)
    alpha = w / float(np.sum(w))
    h_ref_blend = np.sum(alpha[:, None] * aligned_refs, axis=0)
    h_ca = (1.0 - lam_shrink) * h_ls + lam_shrink * h_ref_blend
    return h_ca, alpha, scores


def compute_nbi_burst_variance_guard(
    y_used: np.ndarray,    # [14, 101]
    h_base: np.ndarray,    # [101]
    n0_base: float,
    *,
    trigger_ratio: float = 1.75,
    max_mult: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Strictly blind, C*-invariant localized narrowband burst interference (NBI) variance guard.

    Detects localized excess energy in a 3x3 (symbol x subcarrier) neighborhood relative to
    expected unit-energy 16-QAM signal + baseline thermal noise energy: E_exp(k) = |h_k|^2 + n0_base.
    Returns:
      nv_grid: [14, 101] float64 local effective noise variance per RE
      excess_log_ratio: [14, 101] float64 C*-invariant log(E_obs / E_exp) diagnostic feature
    """
    e_re = y_used.real ** 2 + y_used.imag ** 2  # [14, 101]
    # 3x3 boxcar average with edge reflection (captures 8sc x 5sym bursts cleanly while smoothing 16-QAM crest)
    pad = np.pad(e_re, ((1, 1), (1, 1)), mode="edge")
    e_loc = (
        pad[0:-2, 0:-2] + pad[0:-2, 1:-1] + pad[0:-2, 2:]
        + pad[1:-1, 0:-2] + pad[1:-1, 1:-1] + pad[1:-1, 2:]
        + pad[2:, 0:-2] + pad[2:, 1:-1] + pad[2:, 2:]
    ) / 9.0
    sig_exp = (h_base.real ** 2 + h_base.imag ** 2)[None, :]  # [1, 101]
    e_exp = np.maximum(sig_exp + float(n0_base), 1e-12)
    ratio = e_loc / e_exp
    excess_log = np.log(np.maximum(ratio, 1e-6))

    nv_grid = np.full_like(e_re, float(n0_base), dtype=np.float64)
    burst = ratio > trigger_ratio
    if np.any(burst):
        est_nv = np.maximum(float(n0_base), e_loc - sig_exp)
        est_nv = np.minimum(est_nv, float(max_mult) * max(float(n0_base), float(np.median(sig_exp)) * 0.5))
        nv_grid = np.where(burst, est_nv, nv_grid)
    return nv_grid, excess_log


@dataclass(frozen=True)
class SharedHypothesisBundle:
    y_used: np.ndarray           # [14, 101] complex128
    h_ls: np.ndarray             # [101] complex128
    h_ca: np.ndarray             # [101] complex128
    h_bank: np.ndarray           # [17, 14, 101] complex128
    nv_base: float
    nv_grid: np.ndarray          # [14, 101] float64 (with NBI variance guard)
    sym_ll: np.ndarray           # [17, 14] float64 blind marginalized 16-QAM symbol log-likelihood
    prior_logw: np.ndarray       # [17, 14] float64 prior + likelihood calibrated log-weights
    # Byte-exact shared features for Group C Role A and Role B:
    cplx_feats: np.ndarray       # [14, 17, 4] complex128 (strictly U(1)-equivariant in residual phase)
    scal_feats: np.ndarray       # [14, 17, 10] float64 (strictly C*-invariant)
    base_logw_sr: np.ndarray     # [14, 17] float64 normalized base log-posterior per symbol


def build_shared_hypothesis_bundle(
    y_grid: np.ndarray,
    pilot_scs: np.ndarray,
    train_ref_bank: np.ndarray,
    *,
    ll_temp: float = 1.0,
    tau_ca: float = 1.0,
    lam_ca: float = 0.55,
    use_nbi_guard: bool = True,
) -> SharedHypothesisBundle:
    """Build the complete shared 17-hypothesis physical bundle & features for one TTI."""
    y_used = y_grid[:, USED_SCS].astype(np.complex128)  # [14, 101]
    h_ls = ls_linear_channel(y_grid, pilot_scs)
    h_ca, _, _ = content_addressed_pilot_channel(
        y_grid, pilot_scs, train_ref_bank, tau_ca=tau_ca, lam_shrink=lam_ca
    )
    nv_base = estimate_noise_var_from_grid(y_grid, h_ls, pilot_scs)
    if use_nbi_guard:
        nv_grid, excess_log = compute_nbi_burst_variance_guard(y_used, h_ls, nv_base)
    else:
        nv_grid = np.full((N_SYM, len(USED_SCS)), nv_base, dtype=np.float64)
        e_re = y_used.real ** 2 + y_used.imag ** 2
        e_exp = np.maximum((h_ls.real ** 2 + h_ls.imag ** 2)[None, :] + nv_base, 1e-12)
        excess_log = np.log(np.maximum(e_re / e_exp, 1e-6))

    # Construct [17, 14, 101] channel bank
    s_idx = np.arange(N_SYM, dtype=np.float64)
    dt_s = s_idx - float(PILOT_SYM)  # [-2, -1, 0, +1, ..., +11]
    non_pilot_s = (np.arange(N_SYM) != PILOT_SYM).astype(np.float64)

    h_bank = np.empty((R_HYP, N_SYM, len(USED_SCS)), dtype=np.complex128)
    hyp_phase = np.zeros((R_HYP, N_SYM), dtype=np.float64)
    # r = 0: static LS
    h_bank[0] = h_ls[None, :]
    # r = 1..8: linear CFO/Doppler slope
    for i, om in enumerate(OMEGA_GRID):
        ph = om * dt_s
        hyp_phase[1 + i] = ph
        h_bank[1 + i] = h_ls[None, :] * np.exp(1j * ph)[:, None]
    # r = 9..15: discrete post-pilot phase jumps
    off_j = 1 + len(OMEGA_GRID)
    for i, jr in enumerate(JUMP_RAD_GRID):
        ph = jr * non_pilot_s
        hyp_phase[off_j + i] = ph
        h_bank[off_j + i] = h_ls[None, :] * np.exp(1j * ph)[:, None]
    # r = 16: Train-only content-addressed regularized static channel
    h_bank[R_HYP - 1] = h_ca[None, :]

    # Blind marginalized 16-QAM symbol log-likelihood & soft-decision feedback features per (r, s)
    # Vectorized over [R=17, S=14, K=101, M=16] -> ~380k elements, takes ~2 ms in numpy!
    nv_safe = np.maximum(nv_grid, 1e-12)  # [14, 101]
    diff = y_used[None, :, :, None] - h_bank[:, :, :, None] * CONST16[None, None, None, :]  # [17, 14, 101, 16]
    dist2 = diff.real ** 2 + diff.imag ** 2
    logp_m = -np.log(math.pi * nv_safe)[None, :, :, None] - dist2 / nv_safe[None, :, :, None]  # [17, 14, 101, 16]
    m_max = np.max(logp_m, axis=-1, keepdims=True)
    exp_m = np.exp(logp_m - m_max)
    sum_exp = np.sum(exp_m, axis=-1)  # [17, 14, 101]
    log_marg_re = m_max[..., 0] + np.log(sum_exp) - math.log(16.0)  # [17, 14, 101]

    # Mask out pilot REs on sym 2 (only score data REs!)
    dmask = DATA_MASK_101.astype(np.float64)  # [14, 101]
    dcount = np.sum(dmask, axis=1)            # [14] (101 except sym 2 has 87)
    sym_ll = np.sum(log_marg_re * dmask[None, :, :], axis=-1)  # [17, 14]

    # Posterior soft symbol probabilities p(x_m | y, H_r) -> [17, 14, 101, 16]
    pm = exp_m / np.maximum(sum_exp[..., None], 1e-300)
    x_soft = np.sum(pm * CONST16[None, None, None, :], axis=-1)  # [17, 14, 101] complex soft symbol
    # Symbol posterior entropy (normalized by log(16) -> in [0, 1])
    ent_re = -np.sum(pm * np.log(np.maximum(pm, 1e-300)), axis=-1) / math.log(16.0)
    sym_conf = 1.0 - np.sum(ent_re * dmask[None, :, :], axis=-1) / dcount[None, :]  # [17, 14]

    # Equalized symbol y / H_r and soft residual features (strictly blind!)
    h_abs2 = np.maximum(h_bank.real ** 2 + h_bank.imag ** 2, 1e-12)
    snr_re = h_abs2 / nv_safe[None, :, :]
    w_re = (snr_re / (1.0 + snr_re)) * dmask[None, :, :]  # reliability weight per RE [17, 14, 101]
    w_sum = np.maximum(np.sum(w_re, axis=-1), 1e-12)      # [17, 14]

    x_eq = (y_used[None, :, :] * np.conj(h_bank)) / h_abs2  # [17, 14, 101]
    # 1. Coherent soft-DF phasor: arg(z_corr) directly measures residual off-grid phase error!
    z_corr = np.sum(w_re * x_eq * np.conj(x_soft), axis=-1) / w_sum  # [17, 14]
    # 2. Soft residual phasor
    z_res = np.sum(w_re * (x_eq - x_soft) * np.conj(x_soft), axis=-1) / w_sum  # [17, 14]
    # 3. Blind 4th-power constellation phasor (cancels 90-deg QAM symmetry, invariant to modulation!)
    x_eq_unit = x_eq / np.maximum(np.abs(x_eq), 0.25)
    z_quad = np.sum(w_re * (x_eq_unit ** 4), axis=-1) / w_sum  # [17, 14]
    # 4. Hypothesis phase state phasor exp(j * hyp_phase)
    z_hyp = np.exp(1j * hyp_phase)  # [17, 14]

    # Soft normalized residual MSE per (r, s)
    soft_mse = np.sum(w_re * (np.abs(x_eq - x_soft) ** 2), axis=-1) / w_sum  # [17, 14]

    # Base calibrated log-posterior per symbol (with optional mild ll_temp scaling)
    raw_logw = HYP_LOG_PRIOR[:, None] + sym_ll / max(float(ll_temp), 1e-6)  # [17, 14]
    raw_logw -= np.max(raw_logw, axis=0, keepdims=True)
    logw_norm = raw_logw - np.log(np.sum(np.exp(raw_logw), axis=0, keepdims=True))  # [17, 14]

    # Assemble [14, 17, 4] complex features and [14, 17, 10] scalar features
    cplx_feats = np.stack([z_corr.T, z_res.T, z_quad.T, z_hyp.T], axis=-1).astype(np.complex128)

    ll_centered = (sym_ll - np.max(sym_ll, axis=0, keepdims=True)) / 100.0  # [17, 14]
    ll_frame = np.mean(sym_ll, axis=1, keepdims=True)                       # [17, 1]
    ll_frame_centered = np.broadcast_to((ll_frame - np.max(ll_frame)) / 100.0, (R_HYP, N_SYM))
    prior_b = np.broadcast_to(HYP_LOG_PRIOR[:, None], (R_HYP, N_SYM))
    t_norm = np.broadcast_to(((s_idx - float(PILOT_SYM)) / 11.0)[None, :], (R_HYP, N_SYM))
    om_norm = np.broadcast_to((HYP_OMEGA / 0.40)[:, None], (R_HYP, N_SYM))
    cos_j = np.broadcast_to(np.cos(HYP_JUMP_RAD)[:, None], (R_HYP, N_SYM))
    sin_j = np.broadcast_to(np.sin(HYP_JUMP_RAD)[:, None], (R_HYP, N_SYM))
    # Per-symbol NBI excess energy summary + estimated wideband log-SNR
    sym_excess = np.broadcast_to(np.mean(np.maximum(excess_log, 0.0), axis=1)[None, :], (R_HYP, N_SYM))
    snr_est_log = np.full(
        (R_HYP, N_SYM),
        math.log1p(float(np.mean(np.abs(h_ls) ** 2)) / max(nv_base, 1e-12)),
        dtype=np.float64,
    )
    # 10 scalar features: [14, 17, 10]
    scal_feats = np.stack(
        [
            ll_centered.T,
            ll_frame_centered.T,
            prior_b.T,
            sym_conf.T,
            np.log1p(np.maximum(soft_mse.T, 0.0)),
            t_norm.T,
            om_norm.T,
            cos_j.T,
            sin_j.T,
            (sym_excess + 0.1 * snr_est_log).T,
        ],
        axis=-1,
    ).astype(np.float64)

    return SharedHypothesisBundle(
        y_used=y_used,
        h_ls=h_ls,
        h_ca=h_ca,
        h_bank=h_bank,
        nv_base=float(nv_base),
        nv_grid=nv_grid,
        sym_ll=sym_ll,
        prior_logw=logw_norm,
        cplx_feats=cplx_feats,
        scal_feats=scal_feats,
        base_logw_sr=logw_norm.T.copy(),  # [14, 17]
    )


def hmm_smooth_log_posterior(
    sym_ll: np.ndarray,
    log_prior: np.ndarray = HYP_LOG_PRIOR,
    ll_temp: float = 2.0,
    p_stay: float = 0.88,
) -> np.ndarray:
    """Classical physical Forward-Backward HMM smoother across the 14 OFDM symbols.

    Enforces temporal continuity of phase state while cleanly permitting abrupt mid-TTI jumps.
    Returns normalized log-posterior [17, 14].
    """
    r_cnt, s_cnt = sym_ll.shape
    emit = sym_ll / max(float(ll_temp), 1e-6)  # [R, S]
    # Transition matrix T[i, j] = P(r_s = j | r_{s-1} = i)
    t_mat = np.full((r_cnt, r_cnt), (1.0 - p_stay) / float(r_cnt - 1), dtype=np.float64)
    np.fill_diagonal(t_mat, p_stay)
    log_t = np.log(t_mat)

    # Forward pass
    log_f = np.empty((r_cnt, s_cnt), dtype=np.float64)
    log_f[:, 0] = log_prior + emit[:, 0]
    log_f[:, 0] -= float(np.max(log_f[:, 0]) + math.log(np.sum(np.exp(log_f[:, 0] - np.max(log_f[:, 0])))))
    for s in range(1, s_cnt):
        m = log_f[:, s - 1][:, None] + log_t  # [R_prev, R_curr]
        mx = np.max(m, axis=0)
        pred = mx + np.log(np.sum(np.exp(m - mx[None, :]), axis=0))
        lf = pred + emit[:, s]
        mx2 = float(np.max(lf))
        log_f[:, s] = lf - (mx2 + math.log(float(np.sum(np.exp(lf - mx2)))))

    # Backward pass
    log_b = np.zeros((r_cnt, s_cnt), dtype=np.float64)
    for s in range(s_cnt - 2, -1, -1):
        m = log_t + (emit[:, s + 1] + log_b[:, s + 1])[None, :]  # [R_curr, R_next]
        mx = np.max(m, axis=1)
        lb = mx + np.log(np.sum(np.exp(m - mx[:, None]), axis=1))
        mx2 = float(np.max(lb))
        log_b[:, s] = lb - (mx2 + math.log(float(np.sum(np.exp(lb - mx2)))))

    log_post = log_f + log_b
    mx = np.max(log_post, axis=0, keepdims=True)
    log_post = log_post - (mx + np.log(np.sum(np.exp(log_post - mx), axis=0, keepdims=True)))
    return log_post
