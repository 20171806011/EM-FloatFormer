#!/usr/bin/env python3
"""Classical baselines, Group A (Global Reference & Content Addressing), and Group B
(Hard Selection vs Posterior Mean Fusion vs Full Mixture Likelihood) receivers for
ADALM-PLUTO SDR QPSK continuous frames.

Strict information hygiene & zero leakage invariants:
  - Online estimation reads ONLY:
      * current frame online pilots y[13:21] (8 symbols, known Barker-13 subset)
      * causal history library M_t = {t - L_lib, ..., t - 1} built strictly from blind
        soft-decision-directed updates of past frames (ZERO ground truth Tx_data!)
      * (for Group B3b joint mixture and classical VV-CFO baseline only): unlabeled payload
        observations y[26:126] (never ground truth!)
  - Held-out pilot probe y[21:26] (5 symbols) is STRICTLY MASKED during all online estimation
    and candidate scoring.
  - All arms in Group A share the EXACT same causal history library M_t; all arms in Group B
    share the EXACT same 84-branch physical candidate bank and prior weights alpha_r.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping

import numpy as np
from scipy.special import logsumexp

from evaluator import evaluate_frame_record
from waveform_frontend import (
    FULL_HDR_26,
    N_SYM,
    PAYLOAD_SLICE,
    PILOT_SLICE,
    PROBE_SLICE,
    QPSK_BITS,
    QPSK_CONST,
    SYNC_SLICE,
    nearest_qpsk_symbols,
)

# Known online pilot waveform (8 symbols at indices [13:21])
S_PIL = FULL_HDR_26[PILOT_SLICE]                  # shape (8,)
S_PIL_EN = float(np.sum(np.abs(S_PIL) ** 2))      # ~7.9976
N_PIL = len(S_PIL)                                # 8
K_PIL = np.arange(13, 21, dtype=np.float64)
K_ALL = np.arange(N_SYM, dtype=np.float64)
K_PAY = np.arange(26, N_SYM, dtype=np.float64)
K_CENTER = 16.5                                   # Pilot segment center index

# Full-header reference (indices [1:21] = 20 symbols, excluding transient k=0 and held-out probe [21:26])
S_HDR20 = FULL_HDR_26[1:21]
X_PIL_DC = np.column_stack([S_PIL, np.ones(N_PIL, dtype=np.complex128)])
X_HDR20_DC = np.column_stack([S_HDR20, np.ones(20, dtype=np.complex128)])

# Known physical AC-coupling mean symbol offset: mean(Barker26 + random 100 QPSK) ~ (18-8)/126 * (0.707+0.707j)
PRIOR_DC_RATIO = -((18.0 - 8.0) / 126.0) * (0.707 + 0.707j)  # ~ -0.05611 - 0.05611j
PRIOR_DC_REL_VAR = 0.0085  # variance of random 100-QPSK sample mean around 0: 1/(126) ~ 0.0079

# Group B shared physical candidate bank: 4 quadrant phases x 7 fine CFOs x 3 channel prior families = 84 branches
QUAD_PHASES = np.array([0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi], dtype=np.float64)
CFO_GRID = np.array([-0.030, -0.020, -0.010, 0.0, 0.010, 0.020, 0.030], dtype=np.float64)
PRIOR_FAMILIES = ("self_clear", "local_neighbor", "content_global")
N_CANDIDATES_B = len(QUAD_PHASES) * len(CFO_GRID) * len(PRIOR_FAMILIES)  # 84

# Precompute phase factors for (4, 7, 126)
_PHASE_ROT_PIL = np.exp(1j * (QUAD_PHASES[:, None, None] + CFO_GRID[None, :, None] * (K_PIL[None, None, :] - K_CENTER)))
_PHASE_ROT_ALL = np.exp(1j * (QUAD_PHASES[:, None, None] + CFO_GRID[None, :, None] * (K_ALL[None, None, :] - K_CENTER)))
_PHASE_ROT_PAY = _PHASE_ROT_ALL[:, :, 26:]  # (4, 7, 100)


@dataclass
class ReceiverHyperparams:
    """Dev-tuned hyperparameters (frozen prior to test evaluation)."""
    # Global empirical channel prior (estimated from train split ONLY!)
    h_mean_prior: complex = 0.021 + 0.0j
    h_var_prior: float = 1.0e-4
    # Neighbor window (A2)
    neighbor_k: int = 5
    neighbor_decay: float = 0.45
    neighbor_drift_rel_var: float = 0.015
    # Global uniform & reliability (A3, A4)
    global_drift_rel_var: float = 0.02
    # Content addressing (A5)
    attn_temp: float = 1.0
    attn_drift_rel_var: float = 0.004
    clear_prior_log_bias: float = -1.5
    age_penalty: float = 0.0  # Pure content addressing by default!
    # Group B posterior fusion temperature & family priors
    b_temp: float = 1.0
    b_cfo_zero_bonus: float = 1.2
    b_quad_zero_bonus: float = 0.0  # Uniform over 4 quadrants in prior (let pilots & history resolve!)
    b_var_scale: float = 1.0        # Variance calibration multiplier tuned on dev


def estimate_train_channel_prior(y_sym_train: np.ndarray) -> tuple[complex, float]:
    """Estimate empirical (m0, v0) from train split using ONLY online pilots [13:21] (zero ground truth!)."""
    s_eff = S_PIL + PRIOR_DC_RATIO
    denom = float(np.sum(np.abs(s_eff) ** 2))
    h_pils = (y_sym_train[:, PILOT_SLICE] @ np.conj(s_eff)) / denom
    m0 = complex(np.mean(h_pils))
    v0 = float(np.mean(np.abs(h_pils - m0) ** 2))
    # Add generous safety margin so prior stays well-calibrated across fading regimes
    return m0, max(v0 * 1.5, 1e-6)


def estimate_single_frame_pilot_stats(
    y_pil: np.ndarray,
    h_prior_mean: complex,
    h_prior_var: float,
) -> dict[str, Any]:
    """Extract pilot LS/LMMSE channel, DC offset, and robust noise variance from 8 online pilots [13:21]."""
    # Effective zero-DC-adjusted pilot vector s_eff = s_pil + PRIOR_DC_RATIO
    s_eff = S_PIL + PRIOR_DC_RATIO
    se_en = float(np.sum(np.abs(s_eff) ** 2))

    # 1. Naive 1-tap LS (no DC)
    h_ls_1tap = complex(np.vdot(S_PIL, y_pil) / S_PIL_EN)
    res_1tap = y_pil - h_ls_1tap * S_PIL
    sig2_1tap = float(np.sum(np.abs(res_1tap) ** 2) / max(N_PIL - 1, 1))

    # 2. Unregularized 2-param (h, d) LS on 8 pilots
    w_ls_dc, _, _, _ = np.linalg.lstsq(X_PIL_DC, y_pil, rcond=None)
    h_ls_dc = complex(w_ls_dc[0])
    d_ls_dc = complex(w_ls_dc[1])
    res_dc = y_pil - (h_ls_dc * S_PIL + d_ls_dc)
    sig2_dc = float(np.sum(np.abs(res_dc) ** 2) / max(N_PIL - 2, 1))

    # 3. Physics-regularized AC-coupled model: y_p = h * s_eff + delta_d + w
    # Using s_eff absorbs the deterministic mean Barker DC while allowing small random payload-mean delta_d!
    h_ac_ls = complex(np.vdot(s_eff, y_pil) / se_en)
    res_ac = y_pil - h_ac_ls * s_eff
    # Robust noise variance estimate: combine pilot residual with lower floor from hardware ISI floor (~0.015 * |h|^2)
    h_pow = max(abs(h_ac_ls) ** 2, abs(h_prior_mean) ** 2 * 0.1, 1e-10)
    raw_var = float(np.sum(np.abs(res_ac) ** 2) / max(N_PIL - 1, 1))
    # Shrink noisy 7-dof variance estimate slightly toward realistic floor to avoid catastrophic under-estimation
    sigma2_est = max(raw_var, 0.015 * h_pow)

    # Conjugate Gaussian LMMSE update for h under prior CN(h_prior_mean, h_prior_var)
    prec_prior = 1.0 / max(h_prior_var, 1e-12)
    prec_lik = se_en / sigma2_est
    post_var_h = 1.0 / (prec_prior + prec_lik)
    h_mmse = post_var_h * (prec_prior * h_prior_mean + (np.vdot(s_eff, y_pil) / sigma2_est))

    # Empirical Bayes DC update around h_mmse * PRIOR_DC_RATIO
    d_prior_mean = h_mmse * PRIOR_DC_RATIO
    d_prior_var = max(abs(h_mmse) ** 2 * PRIOR_DC_REL_VAR, 1e-10)
    res_after_h = y_pil - (h_mmse * S_PIL + d_prior_mean)
    prec_d_prior = 1.0 / d_prior_var
    prec_d_lik = float(N_PIL) / sigma2_est
    post_var_d = 1.0 / (prec_d_prior + prec_d_lik)
    delta_d = post_var_d * (np.sum(res_after_h) / sigma2_est)
    d_mmse = d_prior_mean + delta_d

    # Refined residual variance including small uncertainty
    res_mmse = y_pil - (h_mmse * S_PIL + d_mmse)
    sigma2_ref = max(float(np.mean(np.abs(res_mmse) ** 2)) * (N_PIL / max(N_PIL - 1.5, 1.0)), 0.015 * max(abs(h_mmse) ** 2, 1e-10))

    return {
        "h_ls_1tap": h_ls_1tap,
        "sig2_1tap": max(sig2_1tap, 1e-10),
        "h_ls_dc": h_ls_dc,
        "d_ls_dc": d_ls_dc,
        "sig2_dc": max(sig2_dc, 1e-10),
        "h_ac_ls": h_ac_ls,
        "h_mmse": complex(h_mmse),
        "d_mmse": complex(d_mmse),
        "var_h_mmse": float(post_var_h),
        "var_d_mmse": float(post_var_d),
        "sigma2_est": float(sigma2_ref),
    }


def viterbi_viterbi_cfo_est(y_pay: np.ndarray, d_est: complex) -> float:
    """Classical 4th-power Viterbi-Viterbi blind CFO estimator on QPSK payload (covers [-0.04, +0.04] rad/sym)."""
    z = y_pay - d_est
    # Normalize amplitude lightly to prevent outliers from dominating 4th power
    amp = np.abs(z) + 1e-12
    z4 = (z / np.sqrt(amp)) ** 4  # M-th power with soft normalization
    # Fine grid search over [-0.038, +0.038] rad/sym (step 0.001 -> 77 points)
    omegas = np.linspace(-0.038, 0.038, 77, dtype=np.float64)
    k = np.arange(len(z), dtype=np.float64)
    spec = np.abs(np.exp(-1j * 4.0 * omegas[:, None] * k[None, :]) @ z4)
    idx = int(np.argmax(spec))
    # Parabolic refinement
    if 0 < idx < len(omegas) - 1:
        yl, y0, yr = float(spec[idx - 1]), float(spec[idx]), float(spec[idx + 1])
        den = yl - 2.0 * y0 + yr
        frac = float(np.clip(0.5 * (yl - yr) / den, -0.5, 0.5)) if abs(den) > 1e-12 else 0.0
        step = omegas[1] - omegas[0]
        return float(omegas[idx] + frac * step)
    return float(omegas[idx])


def build_causal_history_library(
    y_sym_seq: np.ndarray,
    h_mean_prior: complex,
    h_var_prior: float,
    lib_size: int = 200,
) -> dict[str, np.ndarray]:
    """Build strictly causal, ZERO-LEAKAGE blind soft-decision-directed channel states for all frames.

    For each frame j, estimates (h_dd[j], d_dd[j], var_h_dd[j], sigma2_dd[j], cfo_dd[j], quality[j])
    using ONLY online pilots y_j[13:21] + high-confidence soft-demodulated payload y_j[26:126].
    NEVER touches held-out probe y_j[21:26] or ground-truth Tx_data!
    All downstream arms (A, B, C) read exclusively from causal past indices [t-lib_size : t].
    """
    n = len(y_sym_seq)
    h_dd = np.empty(n, dtype=np.complex128)
    d_dd = np.empty(n, dtype=np.complex128)
    var_h_dd = np.empty(n, dtype=np.float64)
    sigma2_dd = np.empty(n, dtype=np.float64)
    cfo_dd = np.empty(n, dtype=np.float64)
    qual_dd = np.empty(n, dtype=np.float64)

    for j in range(n):
        yj = np.asarray(y_sym_seq[j], dtype=np.complex128)
        st = estimate_single_frame_pilot_stats(yj[PILOT_SLICE], h_mean_prior, h_var_prior)
        h0, d0, sig2 = st["h_mmse"], st["d_mmse"], st["sigma2_est"]

        # Quick blind CFO check via 4-point grid / VV on payload to prevent CFO-corrupted entries
        # Use pilot + payload soft-EM 1-step refinement over 7 CFO grid values
        y_pay = yj[PAYLOAD_SLICE]
        best_ll = -1e30
        best_cfo = 0.0
        best_h = h0
        best_d = d0
        best_s_soft = None
        best_w_soft = None

        for g_idx, w_g in enumerate(CFO_GRID):
            rot_p = np.exp(1j * w_g * (K_PIL - K_CENTER))
            s_eff_g = S_PIL * rot_p + PRIOR_DC_RATIO
            en_g = float(np.sum(np.abs(s_eff_g) ** 2))
            prec_p = 1.0 / max(h_var_prior, 1e-12)
            v_g = 1.0 / (prec_p + en_g / sig2)
            h_g = v_g * (prec_p * h_mean_prior + np.vdot(s_eff_g, yj[PILOT_SLICE]) / sig2)
            d_g = h_g * PRIOR_DC_RATIO

            # Payload marginal log-likelihood under QPSK constellation (100% blind to truth!)
            h_pay_g = h_g * np.exp(1j * w_g * (K_PAY - K_CENTER))
            pred = h_pay_g[:, None] * QPSK_CONST[None, :] + d_g  # (100, 4)
            v_obs = sig2 + v_g * 1.0
            d2 = (y_pay[:, None].real - pred.real) ** 2 + (y_pay[:, None].imag - pred.imag) ** 2
            logp = -d2 / v_obs - math.log(math.pi * v_obs) - math.log(4.0)
            lse = logsumexp(logp, axis=1)
            # Pilot log-likelihood + weak zero-CFO prior
            res_p = yj[PILOT_SLICE] - (h_g * S_PIL * rot_p + d_g)
            ll_pil = -float(np.sum(np.abs(res_p) ** 2)) / v_obs
            ll_tot = float(np.sum(lse)) + ll_pil + (1.0 if abs(w_g) < 1e-9 else 0.0)
            if ll_tot > best_ll:
                best_ll = ll_tot
                best_cfo = float(w_g)
                best_h = complex(h_g)
                best_d = complex(d_g)
                probs = np.exp(logp - lse[:, None])
                best_s_soft = probs @ QPSK_CONST
                best_w_soft = np.clip((np.abs(best_s_soft) ** 2) / 0.9997, 0.0, 1.0)

        # Weighted soft-DD refinement combining 8 known pilots (weight 1.0) + confident payload (weight w^2)
        rot_pil = np.exp(1j * best_cfo * (K_PIL - K_CENTER))
        rot_pay = np.exp(1j * best_cfo * (K_PAY - K_CENTER))
        w_pay = np.where(best_w_soft > 0.35, best_w_soft ** 2, 0.0)
        n_eff_pay = float(np.sum(w_pay))

        if n_eff_pay > 5.0:
            # Joint weighted LS on [s*rot, 1] over pilots [13:21] and soft payload [26:126] (NEVER probe [21:26]!)
            s_comb = np.concatenate([S_PIL * rot_pil, best_s_soft * rot_pay])
            y_comb = np.concatenate([yj[PILOT_SLICE], y_pay])
            wt = np.concatenate([np.ones(N_PIL, dtype=np.float64), w_pay])
            sw = np.sqrt(wt)
            X = np.column_stack([s_comb * sw, sw])
            yw = y_comb * sw
            # Tiny ridge towards pilot prior for complete numerical stability
            XtX = X.conj().T @ X
            Xty = X.conj().T @ yw
            reg = np.diag([sig2 / max(h_var_prior, 1e-10), sig2 / max(abs(best_h) ** 2 * PRIOR_DC_REL_VAR, 1e-10)])
            rhs_prior = np.array([h_mean_prior * reg[0, 0], (best_h * PRIOR_DC_RATIO) * reg[1, 1]], dtype=np.complex128)
            w_dd = np.linalg.solve(XtX + reg, Xty + rhs_prior)
            hj, dj = complex(w_dd[0]), complex(w_dd[1])
            res_comb = y_comb - (hj * s_comb + dj)
            sig2_j = max(float(np.sum(wt * (np.abs(res_comb) ** 2)) / max(float(np.sum(wt)) - 2.0, 1.0)), 0.012 * max(abs(hj) ** 2, 1e-10))
            eff_n = float(N_PIL + 0.7 * n_eff_pay)
            var_hj = sig2_j / max(eff_n, 1.0)
            qual_j = float(n_eff_pay / 100.0)
        else:
            hj, dj, sig2_j = best_h, best_d, sig2
            var_hj = st["var_h_mmse"]
            qual_j = 0.0

        h_dd[j] = hj
        d_dd[j] = dj
        var_h_dd[j] = var_hj
        sigma2_dd[j] = sig2_j
        cfo_dd[j] = best_cfo
        qual_dd[j] = qual_j

    return {
        "h_dd": h_dd,
        "d_dd": d_dd,
        "var_h_dd": var_h_dd,
        "sigma2_dd": sigma2_dd,
        "cfo_dd": cfo_dd,
        "qual_dd": qual_dd,
    }


def _fuse_prior_with_pilots(
    y_pil: np.ndarray,
    h_prior: complex,
    var_h_prior: float,
    d_prior: complex,
    sigma2: float,
    rot_pil: np.ndarray | None = None,
) -> tuple[complex, complex, float, float, float]:
    """Conjugate Gaussian fusion of prior CN(h_prior, var_h_prior) with 8 online pilots.

    Returns (h_post, d_post, var_h_post, var_d_post, marginal_log_evidence).
    """
    sp = S_PIL if rot_pil is None else (S_PIL * rot_pil)
    # Center DC around physics-coupled prior d_prior
    y_adj = y_pil - d_prior
    sp_en = float(np.sum(np.abs(sp) ** 2))
    v_pr = max(float(var_h_prior), 1e-12)
    s2 = max(float(sigma2), 1e-12)

    # Exact marginal log-evidence log N(y_adj; h_prior * sp, s2*I + v_pr * sp sp^H) via matrix determinant lemma!
    # Sigma = s2 I + v_pr sp sp^H
    # det(Sigma) = s2^N * (1 + (v_pr / s2) * ||sp||^2)
    # r^H Sigma^{-1} r = (1/s2) ||r||^2 - (v_pr / (s2 * (s2 + v_pr * ||sp||^2))) |sp^H r|^2
    r = y_adj - h_prior * sp
    r2 = float(np.sum(r.real ** 2 + r.imag ** 2))
    sp_r = np.vdot(sp, r)
    denom = s2 + v_pr * sp_en
    log_det = N_PIL * math.log(math.pi * s2) + math.log(denom / s2)
    quad = (r2 / s2) - (v_pr / (s2 * denom)) * float(sp_r.real ** 2 + sp_r.imag ** 2)
    log_ev = -(log_det + quad)

    # Posterior for h
    prec_pr = 1.0 / v_pr
    prec_lk = sp_en / s2
    var_h_post = 1.0 / (prec_pr + prec_lk)
    h_post = var_h_post * (prec_pr * h_prior + np.vdot(sp, y_adj) / s2)

    # Empirical Bayes tiny refinement of DC offset around d_prior
    d_rel_var = max(abs(h_post) ** 2 * PRIOR_DC_REL_VAR, 1e-10)
    res_d = y_pil - (h_post * sp + d_prior)
    var_d_post = 1.0 / (1.0 / d_rel_var + float(N_PIL) / s2)
    d_post = d_prior + var_d_post * (np.sum(res_d) / s2)

    return complex(h_post), complex(d_post), float(var_h_post), float(var_d_post), float(log_ev)


def compute_content_prior(
    y_pil: np.ndarray,
    lib_h: np.ndarray,
    lib_d: np.ndarray,
    lib_vh: np.ndarray,
    sigma2: float,
    hp: ReceiverHyperparams,
) -> dict[str, Any]:
    """EM-FloatFormer physical content-addressing attention over causal library M_t + self/clear branch."""
    m = len(lib_h)
    # Predictive variance for each library entry evaluating current pilots
    h_scale2 = np.maximum(np.abs(lib_h) ** 2, 1e-10)
    nu = sigma2 + lib_vh + hp.attn_drift_rel_var * h_scale2  # (M,)
    # Residual ||y_pil - (h_j * S_PIL + d_j)||^2 for all M entries vectorized!
    pred = lib_h[:, None] * S_PIL[None, :] + lib_d[:, None]  # (M, 8)
    diff = y_pil[None, :] - pred
    res2 = np.sum(diff.real ** 2 + diff.imag ** 2, axis=1)  # (M,)
    ll_lib = -(N_PIL * np.log(np.pi * nu) + res2 / nu)
    if hp.age_penalty > 0.0:
        ages = np.arange(m, 0, -1, dtype=np.float64) / float(m)
        ll_lib = ll_lib - hp.age_penalty * ages

    # Self/clear branch marginal log-evidence under broad empirical prior
    _, _, _, _, ll_clear = _fuse_prior_with_pilots(
        y_pil, hp.h_mean_prior, hp.h_var_prior, hp.h_mean_prior * PRIOR_DC_RATIO, sigma2
    )
    scores = np.empty(m + 1, dtype=np.float64)
    scores[:m] = (ll_lib - math.log(float(m))) / max(hp.attn_temp, 1e-3)
    scores[m] = (ll_clear + hp.clear_prior_log_bias) / max(hp.attn_temp, 1e-3)
    lse = float(logsumexp(scores))
    w = np.exp(scores - lse)

    # Mixture prior moments (preserving BOTH within-branch and between-branch variance!)
    w_lib = w[:m]
    w_clr = float(w[m])
    h_all = np.append(lib_h, hp.h_mean_prior)
    d_all = np.append(lib_d, hp.h_mean_prior * PRIOR_DC_RATIO)
    v_all = np.append(lib_vh + hp.attn_drift_rel_var * h_scale2, hp.h_var_prior)

    h_mix = complex(np.sum(w * h_all))
    d_mix = complex(np.sum(w * d_all))
    # Law of total variance (posterior_fusion §4.1!)
    v_within = float(np.sum(w * v_all))
    v_between = float(np.sum(w * (np.abs(h_all - h_mix) ** 2)))
    v_tot = max(v_within + v_between, 1e-10)
    return {
        "h_prior": h_mix,
        "d_prior": d_mix,
        "var_h_prior": v_tot,
        "weights": w,
        "clear_weight": w_clr,
        "eff_n_refs": float(1.0 / max(float(np.sum(w ** 2)), 1e-12)),
    }


def build_shared_b_candidates(
    y_pil: np.ndarray,
    sigma2: float,
    priors_3: list[tuple[complex, float, complex]],  # [self_clear, local_neighbor, content_global]
    hp: ReceiverHyperparams,
    extra_logit_bias: np.ndarray | None = None,
) -> dict[str, Any]:
    """Construct the exact shared 84-branch physical candidate bank (4 quad x 7 CFO x 3 prior families).

    Computes exact pilot-conditioned posterior (h_r, d_r, var_h_r, var_d_r) and marginal log-evidence
    for every branch r in 0..83 using ONLY online pilots y_pil and causal priors.
    """
    n_q = len(QUAD_PHASES)
    n_g = len(CFO_GRID)
    n_c = len(priors_3)
    r_tot = n_q * n_g * n_c

    h_post = np.empty(r_tot, dtype=np.complex128)
    d_post = np.empty(r_tot, dtype=np.complex128)
    var_h_post = np.empty(r_tot, dtype=np.float64)
    var_d_post = np.empty(r_tot, dtype=np.float64)
    log_ev = np.empty(r_tot, dtype=np.float64)
    log_pi = np.empty(r_tot, dtype=np.float64)
    q_idx_arr = np.empty(r_tot, dtype=np.int32)
    g_idx_arr = np.empty(r_tot, dtype=np.int32)
    c_idx_arr = np.empty(r_tot, dtype=np.int32)

    idx = 0
    for c_i, (h_pr, v_pr, d_pr) in enumerate(priors_3):
        for q_i in range(n_q):
            # Quadrant rotation applies to the reference prior phase hypothesis:
            # rotating the channel hypothesis by e^{j*q*pi/2} tests all 4 QPSK phase ambiguity states!
            rot_q = np.exp(1j * QUAD_PHASES[q_i])
            h_pr_q = h_pr * rot_q
            d_pr_q = d_pr * rot_q
            for g_i in range(n_g):
                rot_g_pil = np.exp(1j * CFO_GRID[g_i] * (K_PIL - K_CENTER))
                hp_r, dp_r, vhp_r, vdp_r, lev = _fuse_prior_with_pilots(
                    y_pil, h_pr_q, v_pr, d_pr_q, sigma2, rot_pil=rot_g_pil
                )
                h_post[idx] = hp_r
                d_post[idx] = dp_r
                var_h_post[idx] = vhp_r
                var_d_post[idx] = vdp_r
                log_ev[idx] = lev
                # Prospective prior: slight bonus for 0-CFO (parsimony), uniform over 3 families
                lp = (hp.b_cfo_zero_bonus if g_i == 3 else 0.0) + (hp.b_quad_zero_bonus if q_i == 0 else 0.0)
                log_pi[idx] = lp
                q_idx_arr[idx] = q_i
                g_idx_arr[idx] = g_i
                c_idx_arr[idx] = c_i
                idx += 1

    logits = (log_pi + log_ev) / max(hp.b_temp, 1e-3)
    if extra_logit_bias is not None:
        logits = logits + extra_logit_bias
    alpha = np.exp(logits - float(logsumexp(logits)))

    # Build per-symbol predicted channel h_all[r, k] = h_post[r] * exp(j * omega_g[r] * (k - K_CENTER))
    # Note: the quadrant phase q_i is ALREADY absorbed into h_post[r] via h_pr_q!
    cfo_per_r = CFO_GRID[g_idx_arr]
    rot_all = np.exp(1j * cfo_per_r[:, None] * (K_ALL[None, :] - K_CENTER))  # (84, 126)
    h_all = h_post[:, None] * rot_all                                       # (84, 126)
    # Predictive observation variance per branch (at symbol energy ~1.0)
    obs_var_r = (sigma2 + var_h_post + var_d_post) * hp.b_var_scale         # (84,)

    return {
        "alpha": alpha,
        "logits_raw": log_pi + log_ev,
        "h_post": h_post,
        "d_post": d_post,
        "var_h_post": var_h_post,
        "var_d_post": var_d_post,
        "h_all": h_all,
        "obs_var_r": obs_var_r,
        "q_idx": q_idx_arr,
        "g_idx": g_idx_arr,
        "c_idx": c_idx_arr,
    }


def decode_group_b_variants(
    y_sym: np.ndarray,
    bank: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Run all 4 Group B decision rules on the EXACT same 84-branch candidate bank & alpha_r!

    Variants:
      - B1_hard_map: single argmax_r alpha_r branch
      - B2_posterior_mean_plugin: posterior mean channel + total variance (within + between) plug-in QPSK
      - B3a_sym_mixture: exact per-symbol mixture marginalization over all 84 branches using alpha_r
      - B3b_joint_mixture: exact full-frame payload mixture posterior beta_r (posterior_fusion §5.2)
                           marginalized over all 84 branches for symbol-MAP and bit-LLR-MAP decisions!
    """
    alpha = bank["alpha"]       # (84,)
    h_all = bank["h_all"]       # (84, 126)
    d_post = bank["d_post"]     # (84,)
    obs_var = bank["obs_var_r"] # (84,)
    y_pay = y_sym[PAYLOAD_SLICE]  # (100,)
    h_pay = h_all[:, PAYLOAD_SLICE]  # (84, 100)

    # --- B1: Hard MAP ---
    t0 = time.perf_counter()
    r_hat = int(np.argmax(alpha))
    h_b1 = h_all[r_hat]
    d_b1 = complex(d_post[r_hat])
    v_b1 = np.full(N_SYM, float(obs_var[r_hat]), dtype=np.float64)
    eq_b1 = (y_pay - d_b1) / np.where(np.abs(h_b1[PAYLOAD_SLICE]) > 1e-12, h_b1[PAYLOAD_SLICE], 1e-12)
    sdec_b1, idx_b1 = nearest_qpsk_symbols(eq_b1)
    bits_b1 = QPSK_BITS[idx_b1]
    t_b1 = (time.perf_counter() - t0) * 1000.0

    # --- B2: Posterior Mean Plug-in (with Law of Total Variance!) ---
    t0 = time.perf_counter()
    h_b2 = np.sum(alpha[:, None] * h_all, axis=0)                  # (126,)
    d_b2 = complex(np.sum(alpha * d_post))
    diff_h = h_all - h_b2[None, :]                                 # (84, 126)
    diff_d = d_post - d_b2                                         # (84,)
    v_between = np.sum(alpha[:, None] * (diff_h.real ** 2 + diff_h.imag ** 2), axis=0) + float(
        np.sum(alpha * (diff_d.real ** 2 + diff_d.imag ** 2))
    )
    v_within = float(np.sum(alpha * obs_var))
    v_b2 = v_within + v_between                                    # (126,)
    eq_b2 = (y_pay - d_b2) / np.where(np.abs(h_b2[PAYLOAD_SLICE]) > 1e-12, h_b2[PAYLOAD_SLICE], 1e-12)
    sdec_b2, idx_b2 = nearest_qpsk_symbols(eq_b2)
    bits_b2 = QPSK_BITS[idx_b2]
    t_b2 = (time.perf_counter() - t0) * 1000.0

    # Shared exact branch-symbol likelihood tensor: log p(y_k, S_k=s | I, r) for (84, 100, 4)
    t_mix0 = time.perf_counter()
    # pred: (84, 100, 4)
    pred = h_pay[:, :, None] * QPSK_CONST[None, None, :] + d_post[:, None, None]
    err = y_pay[None, :, None] - pred
    d2 = err.real ** 2 + err.imag ** 2
    v_r = obs_var[:, None, None]
    log_p_y_s_given_r = -d2 / v_r - np.log(np.pi * v_r) - math.log(4.0)  # (84, 100, 4)
    log_alpha = np.log(np.maximum(alpha, 1e-300))                        # (84,)
    t_shared_mix = (time.perf_counter() - t_mix0) * 1000.0

    # --- B3a: Per-symbol Exact Mixture Likelihood (prior alpha_r only) ---
    t0 = time.perf_counter()
    log_joint_sym = log_alpha[:, None, None] + log_p_y_s_given_r         # (84, 100, 4)
    log_marg_s_a = logsumexp(log_joint_sym, axis=0)                      # (100, 4)
    p_s_a = np.exp(log_marg_s_a - logsumexp(log_marg_s_a, axis=1, keepdims=True))
    idx_b3a = np.argmax(p_s_a, axis=1)
    sdec_b3a = QPSK_CONST[idx_b3a]
    # Exact bit-wise MAP from marginal bit probabilities (b0 in {2,3}, b1 in {1,3})
    p_b0_a = p_s_a[:, 2] + p_s_a[:, 3]
    p_b1_a = p_s_a[:, 1] + p_s_a[:, 3]
    bits_b3a = np.stack([(p_b0_a > 0.5).astype(np.int8), (p_b1_a > 0.5).astype(np.int8)], axis=-1)
    # Soft MMSE symbol reconstruction E[S_k | I, y_k]
    eq_b3a = p_s_a @ QPSK_CONST
    t_b3a = t_shared_mix + (time.perf_counter() - t0) * 1000.0

    # --- B3b: Exact Full-Frame Joint Mixture Posterior (posterior_fusion §5.2!) ---
    t0 = time.perf_counter()
    # Exact per-symbol marginal log p(y_k | I, r) = logsumexp_s log p(y_k, S_k=s | I, r)
    log_p_y_given_r = logsumexp(log_p_y_s_given_r, axis=2)               # (84, 100)
    # Full-frame payload marginal log-likelihood per branch
    log_beta_unnorm = log_alpha + np.sum(log_p_y_given_r, axis=1)        # (84,)
    beta = np.exp(log_beta_unnorm - float(logsumexp(log_beta_unnorm)))   # (84,)
    # Exact conditional symbol posterior within each branch: P(S_k=s | I, y_k, r)
    p_s_given_r = np.exp(log_p_y_s_given_r - log_p_y_given_r[:, :, None])  # (84, 100, 4)
    # Exact mixture marginalization (ZERO double counting of y_k!): P(S_k=s | O) = sum_r beta_r P(S_k=s | y_k, r)
    p_s_b = np.tensordot(beta, p_s_given_r, axes=([0], [0]))             # (100, 4)
    idx_b3b = np.argmax(p_s_b, axis=1)
    sdec_b3b = QPSK_CONST[idx_b3b]
    p_b0_b = p_s_b[:, 2] + p_s_b[:, 3]
    p_b1_b = p_s_b[:, 1] + p_s_b[:, 3]
    bits_b3b = np.stack([(p_b0_b > 0.5).astype(np.int8), (p_b1_b > 0.5).astype(np.int8)], axis=-1)
    # Posterior-weighted equalized symbol (MMSE across beta-weighted branches, linear equalization for fair EVM!)
    eq_branch = (y_pay[None, :] - d_post[:, None]) / np.where(np.abs(h_pay) > 1e-12, h_pay, 1e-12)
    eq_b3b = np.sum(beta[:, None] * eq_branch, axis=0)
    t_b3b = t_shared_mix + (time.perf_counter() - t0) * 1000.0

    # Notice: for held-out probe [21:26] prediction, B3a and B3b use the EXACT pre-payload posterior mean (h_b2, d_b2, v_b2)
    # so probe evaluation NEVER leaks payload observations or probe observations!
    return {
        "B1_hard_map": {
            "h_all": h_b1, "d_pred": d_b1, "v_all": v_b1,
            "s_dec": sdec_b1, "bits_dec": bits_b1, "s_eq": eq_b1, "extra_ms": t_b1,
        },
        "B2_posterior_mean_plugin": {
            "h_all": h_b2, "d_pred": d_b2, "v_all": v_b2,
            "s_dec": sdec_b2, "bits_dec": bits_b2, "s_eq": eq_b2, "extra_ms": t_b2,
        },
        "B3a_sym_mixture": {
            "h_all": h_b2, "d_pred": d_b2, "v_all": v_b2,
            "s_dec": sdec_b3a, "bits_dec": bits_b3a, "s_eq": eq_b2, "extra_ms": t_b3a,
        },
        "B3b_joint_mixture": {
            "h_all": h_b2, "d_pred": d_b2, "v_all": v_b2,
            "s_dec": sdec_b3b, "bits_dec": bits_b3b, "s_eq": eq_b3b, "extra_ms": t_b3b,
            "beta": beta,
        },
    }


def run_all_classical_and_ab_arms(
    bundle: Mapping[str, np.ndarray],
    tx_mm: np.ndarray,
    hp: ReceiverHyperparams,
    lib_size: int = 200,
    cached_lib: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, np.ndarray]]:
    """Run all Classical Baselines, Group A (5 arms), and Group B (4 arms) paired on one split-condition bundle."""
    y_all = np.asarray(bundle["y_sym"], dtype=np.complex128)
    frame_indices = np.asarray(bundle["frame_indices"], dtype=np.int32)
    eval_offset = int(bundle["eval_offset"])
    h_ora_all = np.asarray(bundle["h_ora"], dtype=np.complex128)
    d_ora_all = np.asarray(bundle["d_ora"], dtype=np.complex128)
    sync_ok_all = np.asarray(bundle["sync_ok"], dtype=np.bool_)

    if cached_lib is None:
        lib = build_causal_history_library(y_all, hp.h_mean_prior, hp.h_var_prior, lib_size=lib_size)
    else:
        lib = cached_lib

    arm_names = [
        "base_ls_1tap",
        "base_ls_dc",
        "base_mmse_dc_cfo",
        "base_full20_ls_dc",
        "A1_local_single",
        "A2_neighbor_k5",
        "A3_global_uniform",
        "A4_global_reliability",
        "A5_global_content_attn",
        "B1_hard_map",
        "B2_posterior_mean_plugin",
        "B3a_sym_mixture",
        "B3b_joint_mixture",
    ]
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in arm_names}

    # Precompute neighbor decay weights for k=5
    nb_k = hp.neighbor_k
    nb_w_raw = np.exp(-hp.neighbor_decay * np.arange(nb_k, 0, -1, dtype=np.float64))
    nb_w_raw /= np.sum(nb_w_raw)

    n_tot = len(y_all)
    for t in range(eval_offset, n_tot):
        fidx = int(frame_indices[t])
        y = y_all[t]
        y_pil = y[PILOT_SLICE]
        s_true = np.asarray(tx_mm[fidx], dtype=np.complex128)
        h_ora = complex(h_ora_all[t])
        d_ora = complex(d_ora_all[t])
        sync_ok = bool(sync_ok_all[t])

        # Strictly causal library slice [t - lib_size : t]
        l_start = max(0, t - lib_size)
        lib_h = lib["h_dd"][l_start:t]
        lib_d = lib["d_dd"][l_start:t]
        lib_vh = lib["var_h_dd"][l_start:t]

        # --- Common single-frame pilot statistics ---
        t0 = time.perf_counter()
        st = estimate_single_frame_pilot_stats(y_pil, hp.h_mean_prior, hp.h_var_prior)
        t_st = (time.perf_counter() - t0) * 1000.0
        sig2 = st["sigma2_est"]

        # 1. base_ls_1tap
        h1 = st["h_ls_1tap"]
        v1 = np.full(N_SYM, st["sig2_1tap"] * (1.0 + 1.0 / N_PIL), dtype=np.float64)
        records["base_ls_1tap"].append(evaluate_frame_record(
            fidx, y, s_true, h_ora, d_ora, sync_ok,
            np.full(N_SYM, h1, dtype=np.complex128), 0.0j, v1, latency_ms=t_st, n_candidates=1,
        ))

        # 2. base_ls_dc
        h_dc, d_dc = st["h_ls_dc"], st["d_ls_dc"]
        v_dc = np.full(N_SYM, st["sig2_dc"] * (1.0 + 2.0 / N_PIL), dtype=np.float64)
        records["base_ls_dc"].append(evaluate_frame_record(
            fidx, y, s_true, h_ora, d_ora, sync_ok,
            np.full(N_SYM, h_dc, dtype=np.complex128), d_dc, v_dc, latency_ms=t_st, n_candidates=1,
        ))

        # 3. base_mmse_dc_cfo (Mature classical single-frame receiver: LMMSE + DC + Viterbi-Viterbi blind CFO!)
        t0 = time.perf_counter()
        w_vv = viterbi_viterbi_cfo_est(y[PAYLOAD_SLICE], st["d_mmse"])
        rot_vv_pil = np.exp(1j * w_vv * (K_PIL - K_CENTER))
        h_vv, d_vv, vh_vv, vd_vv, _ = _fuse_prior_with_pilots(
            y_pil, hp.h_mean_prior, hp.h_var_prior, hp.h_mean_prior * PRIOR_DC_RATIO, sig2, rot_pil=rot_vv_pil
        )
        h_vv_all = h_vv * np.exp(1j * w_vv * (K_ALL - K_CENTER))
        v_vv_all = np.full(N_SYM, (sig2 + vh_vv + vd_vv) * hp.b_var_scale, dtype=np.float64)
        t_vv = t_st + (time.perf_counter() - t0) * 1000.0
        records["base_mmse_dc_cfo"].append(evaluate_frame_record(
            fidx, y, s_true, h_ora, d_ora, sync_ok,
            h_vv_all, d_vv, v_vv_all, latency_ms=t_vv, n_candidates=1,
        ))

        # 4. base_full20_ls_dc (Reference classical upper bound using [1:21], 20 symbols, still masking probe [21:26]!)
        t0 = time.perf_counter()
        w20, _, _, _ = np.linalg.lstsq(X_HDR20_DC, y[1:21], rcond=None)
        h20, d20 = complex(w20[0]), complex(w20[1])
        r20 = y[1:21] - (h20 * S_HDR20 + d20)
        s2_20 = max(float(np.sum(np.abs(r20) ** 2) / 18.0), 0.015 * max(abs(h20) ** 2, 1e-10))
        v20 = np.full(N_SYM, s2_20 * (1.0 + 2.0 / 20.0), dtype=np.float64)
        t_20 = (time.perf_counter() - t0) * 1000.0
        records["base_full20_ls_dc"].append(evaluate_frame_record(
            fidx, y, s_true, h_ora, d_ora, sync_ok,
            np.full(N_SYM, h20, dtype=np.complex128), d20, v20, latency_ms=t_20, n_candidates=1,
        ))

        # --- Group A: 5 strictly paired arms separating Reference Scope vs Content Addressing ---
        # A1: Local single frame LMMSE (M = 0)
        t0 = time.perf_counter()
        h_a1, d_a1, vh_a1, vd_a1, _ = _fuse_prior_with_pilots(
            y_pil, hp.h_mean_prior, hp.h_var_prior, hp.h_mean_prior * PRIOR_DC_RATIO, sig2
        )
        t_a1 = t_st + (time.perf_counter() - t0) * 1000.0
        records["A1_local_single"].append(evaluate_frame_record(
            fidx, y, s_true, h_ora, d_ora, sync_ok,
            np.full(N_SYM, h_a1, dtype=np.complex128), d_a1,
            np.full(N_SYM, (sig2 + vh_a1 + vd_a1) * hp.b_var_scale, dtype=np.float64),
            latency_ms=t_a1, n_candidates=1,
        ))

        # A2: Fixed local neighbor window (k = 5)
        t0 = time.perf_counter()
        nh = lib_h[-nb_k:]
        nd = lib_d[-nb_k:]
        nvh = lib_vh[-nb_k:]
        w_nb = nb_w_raw[-len(nh):]
        w_nb = w_nb / np.sum(w_nb)
        h_pr_nb = complex(np.sum(w_nb * nh))
        d_pr_nb = complex(np.sum(w_nb * nd))
        v_pr_nb = float(np.sum(w_nb * (nvh + np.abs(nh - h_pr_nb) ** 2 + hp.neighbor_drift_rel_var * np.abs(nh) ** 2)))
        h_a2, d_a2, vh_a2, vd_a2, _ = _fuse_prior_with_pilots(y_pil, h_pr_nb, v_pr_nb, d_pr_nb, sig2)
        t_a2 = t_st + (time.perf_counter() - t0) * 1000.0
        records["A2_neighbor_k5"].append(evaluate_frame_record(
            fidx, y, s_true, h_ora, d_ora, sync_ok,
            np.full(N_SYM, h_a2, dtype=np.complex128), d_a2,
            np.full(N_SYM, (sig2 + vh_a2 + vd_a2) * hp.b_var_scale, dtype=np.float64),
            latency_ms=t_a2, n_candidates=nb_k,
        ))

        # A3: Global uniform average (M = 200) -> isolates scope expansion WITHOUT content selection
        t0 = time.perf_counter()
        h_pr_u = complex(np.mean(lib_h))
        d_pr_u = complex(np.mean(lib_d))
        v_pr_u = float(np.mean(lib_vh + np.abs(lib_h - h_pr_u) ** 2 + hp.global_drift_rel_var * np.abs(lib_h) ** 2))
        h_a3, d_a3, vh_a3, vd_a3, _ = _fuse_prior_with_pilots(y_pil, h_pr_u, v_pr_u, d_pr_u, sig2)
        t_a3 = t_st + (time.perf_counter() - t0) * 1000.0
        records["A3_global_uniform"].append(evaluate_frame_record(
            fidx, y, s_true, h_ora, d_ora, sync_ok,
            np.full(N_SYM, h_a3, dtype=np.complex128), d_a3,
            np.full(N_SYM, (sig2 + vh_a3 + vd_a3) * hp.b_var_scale, dtype=np.float64),
            latency_ms=t_a3, n_candidates=len(lib_h),
        ))

        # A4: Global classical reliability weights (M = 200, inverse-variance only, content-blind)
        t0 = time.perf_counter()
        w_rel = 1.0 / (lib_vh + 1e-10)
        w_rel /= np.sum(w_rel)
        h_pr_r = complex(np.sum(w_rel * lib_h))
        d_pr_r = complex(np.sum(w_rel * lib_d))
        v_pr_r = float(np.sum(w_rel * (lib_vh + np.abs(lib_h - h_pr_r) ** 2 + hp.global_drift_rel_var * np.abs(lib_h) ** 2)))
        h_a4, d_a4, vh_a4, vd_a4, _ = _fuse_prior_with_pilots(y_pil, h_pr_r, v_pr_r, d_pr_r, sig2)
        t_a4 = t_st + (time.perf_counter() - t0) * 1000.0
        records["A4_global_reliability"].append(evaluate_frame_record(
            fidx, y, s_true, h_ora, d_ora, sync_ok,
            np.full(N_SYM, h_a4, dtype=np.complex128), d_a4,
            np.full(N_SYM, (sig2 + vh_a4 + vd_a4) * hp.b_var_scale, dtype=np.float64),
            latency_ms=t_a4, n_candidates=len(lib_h),
        ))

        # A5: EM-FloatFormer Physical Content-Addressing Attention (M = 200 + clear branch)
        t0 = time.perf_counter()
        ca = compute_content_prior(y_pil, lib_h, lib_d, lib_vh, sig2, hp)
        h_a5, d_a5, vh_a5, vd_a5, _ = _fuse_prior_with_pilots(
            y_pil, ca["h_prior"], ca["var_h_prior"], ca["d_prior"], sig2
        )
        t_a5 = t_st + (time.perf_counter() - t0) * 1000.0
        records["A5_global_content_attn"].append(evaluate_frame_record(
            fidx, y, s_true, h_ora, d_ora, sync_ok,
            np.full(N_SYM, h_a5, dtype=np.complex128), d_a5,
            np.full(N_SYM, (sig2 + vh_a5 + vd_a5) * hp.b_var_scale, dtype=np.float64),
            latency_ms=t_a5, n_candidates=len(lib_h) + 1,
        ))

        # --- Group B: Shared 84-branch Physical Candidate Bank (4 quad x 7 CFO x 3 prior families) ---
        t_b0 = time.perf_counter()
        priors_3 = [
            (hp.h_mean_prior, hp.h_var_prior, hp.h_mean_prior * PRIOR_DC_RATIO),
            (h_pr_nb, v_pr_nb, d_pr_nb),
            (ca["h_prior"], ca["var_h_prior"], ca["d_prior"]),
        ]
        bank = build_shared_b_candidates(y_pil, sig2, priors_3, hp)
        t_bank = t_a5 + (time.perf_counter() - t_b0) * 1000.0
        b_out = decode_group_b_variants(y, bank)

        for bname in ("B1_hard_map", "B2_posterior_mean_plugin", "B3a_sym_mixture", "B3b_joint_mixture"):
            bo = b_out[bname]
            records[bname].append(evaluate_frame_record(
                fidx, y, s_true, h_ora, d_ora, sync_ok,
                bo["h_all"], bo["d_pred"], bo["v_all"],
                s_dec_payload=bo["s_dec"],
                bits_dec_payload=bo["bits_dec"],
                s_eq_payload=bo["s_eq"],
                latency_ms=t_bank + bo["extra_ms"],
                n_candidates=N_CANDIDATES_B,
            ))

    return records, lib
