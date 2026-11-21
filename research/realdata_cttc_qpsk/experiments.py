"""Controlled comparative experiments A (Global Content Addressing), B (Hard vs Posterior Fusion
vs Mixture Likelihood), and C (Lightweight Trainable Attention / Weight & Residual Network).

Strictly aligns with:
  - research/content_addressing_scalar/REPORT.md (Eqs. 1-9)
  - research/posterior_fusion/pages/01-08.md (orthogonal projection, total covariance, mixture LLR)
  - p12/P12_CONSISTENCY_FIX.md (single-count pilot evidence, zero target/holdout leakage)
"""
from __future__ import annotations

import math
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from .adapter import (
    FS_SYM,
    HOLDOUT_IDX,
    KNOWN_IDX,
    PAD_SYMS,
    PAYLOAD_IDX,
    PILOT_IDX,
    PKT_LEN,
    SPS_NOM,
    SYNC_IDX,
    PacketSample,
)
from .baselines import (
    ClassicalConfig,
    ReceiverOutput,
    acquire_timing_and_cfo,
    extract_symbol_stream,
)
from .evaluator import (
    QPSK_CONSTELLATION,
    bits_to_qpsk,
    joint_mixture_qpsk_posterior,
    mixture_qpsk_posterior,
    qpsk_bit_llrs_gaussian,
    qpsk_hard_symbols,
    qpsk_to_bits,
)


# =============================================================================
# Shared Reference Pool & Pilot-Only Reference Extraction (Zero Leakage)
# =============================================================================

@dataclass
class RefEntry:
    """Pilot-only physical summary of one reference packet (never touches ref payload/holdout)."""
    unit_id: str
    capture_id: str  # evaluator-only audit metadata; NEVER read by receiver scoring!
    freq_mhz: int    # evaluator-only audit metadata
    nom_snr_db: int  # evaluator-only audit metadata
    time_sec: float
    est_timing_rx: float
    est_cfo_hz: float
    h_pow: float
    nv_est: float
    est_snr_lin: float
    est_snr_db: float
    slope_ratio: complex


def extract_reference_entry(pkt: PacketSample, cfg: ClassicalConfig) -> RefEntry:
    """Extract pilot-only physical statistics from a reference packet."""
    t_rx, cfo_hz, y_sym, _ = acquire_timing_and_cfo(pkt.y_iq_4x, pkt.x_known, cfg)
    n_idx = np.arange(PKT_LEN, dtype=np.float64)
    y_derot = y_sym * np.exp(-2j * math.pi * cfo_hz * n_idx / FS_SYM)
    yk = y_derot[KNOWN_IDX]
    xk = pkt.x_known
    n_pil = len(KNOWN_IDX)

    h_ls = complex(np.vdot(xk, yk) / max(float(np.vdot(xk, xk).real), 1e-18))
    resid = yk - h_ls * xk
    nv_resid = float(np.sum(np.abs(resid) ** 2) / max(n_pil - 2, 1))
    nv_bg = max(float(pkt.bg_noise_var), 1e-18)
    nv_est = max(cfg.noise_floor_mult * nv_bg, cfg.noise_blend * nv_resid + (1.0 - cfg.noise_blend) * nv_bg)
    h_pow = max(float(abs(h_ls) ** 2) - nv_est / float(n_pil), 0.05 * nv_est)
    snr_lin = h_pow / max(nv_est, 1e-18)

    # Normalized linear envelope slope from pilots
    t_k = (KNOWN_IDX.astype(np.float64) - (PKT_LEN - 1) / 2.0) / (PKT_LEN / 2.0)
    A = np.column_stack([xk, xk * t_k])
    reg = np.array([nv_est / max(h_pow, 1e-18), 4.0 * nv_est / max(h_pow, 1e-18)], dtype=np.float64)
    coef = np.linalg.solve(A.conj().T @ A + np.diag(reg), A.conj().T @ yk)
    slope_ratio = complex(coef[1] / (coef[0] + 1e-18))

    return RefEntry(
        unit_id=pkt.unit_id,
        capture_id=pkt.capture_id,
        freq_mhz=pkt.freq_mhz,
        nom_snr_db=pkt.nom_snr_db,
        time_sec=pkt.time_sec,
        est_timing_rx=t_rx,
        est_cfo_hz=cfo_hz,
        h_pow=h_pow,
        nv_est=nv_est,
        est_snr_lin=snr_lin,
        est_snr_db=10.0 * math.log10(max(snr_lin, 1e-12)),
        slope_ratio=slope_ratio,
    )


def build_cross_session_pool_for_target(
    target: PacketSample,
    refs_by_capture: dict[str, list[RefEntry]],
    *,
    per_capture: int = 4,
    rng: np.random.Generator,
) -> list[RefEntry]:
    """Build a balanced, shuffled global reference pool of M = 9 * per_capture references.

    Pool order is uniformly shuffled using `rng` independent of target packet content, so a
    fixed local window S = {0..k-1} satisfies the exact assumptions of content_addressing_scalar §2.
    """
    pool: list[RefEntry] = []
    for cap_id in sorted(refs_by_capture.keys()):
        entries = refs_by_capture[cap_id]
        if len(entries) <= per_capture:
            chosen = list(entries)
        else:
            idx = rng.choice(len(entries), size=per_capture, replace=False)
            chosen = [entries[int(i)] for i in idx]
        pool.extend(chosen)
    perm = rng.permutation(len(pool))
    return [pool[int(i)] for i in perm]


def build_intra_session_pool_for_target(
    target: PacketSample,
    refs_same_capture: list[RefEntry],
    *,
    pool_size: int = 24,
) -> list[RefEntry]:
    """Build an intra-session temporal pool sorted by time distance |t_ref - t_target|.

    Thus local window S = {0..k-1} selects the k temporally nearest reference packets,
    while global methods access all M=pool_size packets spanning distant loops/times.
    """
    ordered = sorted(refs_same_capture, key=lambda r: abs(r.time_sec - target.time_sec))
    return ordered[:pool_size]


# =============================================================================
# Experiment A: Global Reference Pool & Content Addressing
# =============================================================================

def pilot_marginal_loglik_for_ref(
    y_iq_4x: np.ndarray,
    x_known: np.ndarray,
    ref: RefEntry,
    *,
    y_sym_cache: dict[float, np.ndarray] | None = None,
) -> tuple[float, float]:
    """Exact phase-marginalized complex Gaussian log-likelihood of target pilots under ref prior.

    Model under ref j:
        y_k = h * x_k * exp(j 2pi f_j k / fsym) + n_k,  h ~ CN(0, P_j),  n_k ~ CN(0, nv_j).
    Marginalizing out complex h yields exact closed-form log p(y_known | x_known, ref_j):
        ll = -N_p*log(pi*nv_j) - log(1 + N_p*P_j/nv_j) - ||y||^2/nv_j + |<z, 1>|^2 / (nv_j*(N_p + nv_j/P_j))
    Returns (log_lik, coherent_corr_power).
    """
    t_key = round(ref.est_timing_rx, 2)
    if y_sym_cache is not None and t_key in y_sym_cache:
        y_sym = y_sym_cache[t_key]
    else:
        y_sym = extract_symbol_stream(y_iq_4x, ref.est_timing_rx)
        if y_sym_cache is not None:
            y_sym_cache[t_key] = y_sym

    yk = y_sym[KNOWN_IDX]
    n_p = float(len(KNOWN_IDX))
    rot = np.exp(-2j * math.pi * ref.est_cfo_hz * KNOWN_IDX.astype(np.float64) / FS_SYM)
    z = yk * np.conj(x_known) * rot
    coh = abs(complex(np.sum(z))) ** 2
    y_e = float(np.sum(np.abs(yk) ** 2))
    p_j = max(ref.h_pow, 1e-18)
    nv_j = max(ref.nv_est, 1e-18)
    ll = (
        -n_p * math.log(math.pi * nv_j)
        - math.log(1.0 + n_p * p_j / nv_j)
        - y_e / nv_j
        + coh / (nv_j * (n_p + nv_j / p_j))
    )
    return float(ll), float(coh / (n_p * n_p * p_j + 1e-30))


def _softmax(scores: np.ndarray, temp: float = 1.0) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64) / max(float(temp), 1e-6)
    s = s - float(np.max(s))
    e = np.exp(np.clip(s, -500.0, 0.0))
    return e / float(np.sum(e))


def run_prior_conditioned_receiver(
    pkt: PacketSample,
    cfg: ClassicalConfig,
    *,
    cfo_prior_mean: float,
    cfo_prior_var: float,
    h_pow_prior: float,
    nv_prior: float,
    timing_prior: float,
    prior_strength: float = 1.0,
    extra: dict[str, Any] | None = None,
) -> ReceiverOutput:
    """Common downstream MAP-CFO + MMSE receiver conditioned on reference-aggregated prior.

    All arms in Experiment A (local window, global uniform, global SNR weight, global content
    matching, learned attention) use this EXACT same downstream estimator so differences isolate
    purely the reference scope and weighting mechanism.
    """
    t0 = time.perf_counter()
    # Timing: blend target pilot timing search lightly around prior or full grid
    t_rx_local, cfo_local, y_sym_local, _ = acquire_timing_and_cfo(pkt.y_iq_4x, pkt.x_known, cfg)
    # Decide timing: if prior variance on CFO is tight (coherent match), test timing_prior vs t_rx_local
    y_sym_pr = extract_symbol_stream(pkt.y_iq_4x, timing_prior)
    # Evaluate MAP objective over fine CFO grid combining pilot likelihood + Gaussian CFO prior:
    #   J(f) = |sum_k y_k x_k^* e^{-j2pi f k/fsym}|^2 / (nv_eff * (N_p + nv_eff/P_h)) - (f - f_pr)^2 / (2 * v_f)
    nv_bg = max(float(pkt.bg_noise_var), 1e-18)
    nv_eff_prior = max(0.5 * nv_bg, 0.65 * nv_prior + 0.35 * nv_bg)
    p_h_eff = max(h_pow_prior, 0.05 * nv_eff_prior)

    # Choose between local timing and prior timing by MAP objective
    known_f = KNOWN_IDX.astype(np.float64)
    n_p = float(len(KNOWN_IDX))
    v_f = max(cfo_prior_var, 15.0 ** 2) / max(prior_strength, 1e-6)

    # Search grid covering both prior neighborhood and local peak
    f_cands = np.unique(
        np.concatenate(
            [
                np.linspace(cfo_prior_mean - 450.0, cfo_prior_mean + 450.0, 91),
                np.linspace(cfo_local - 350.0, cfo_local + 350.0, 71),
            ]
        )
    )
    rot_mat = np.exp(-2j * math.pi * f_cands[:, None] * known_f[None, :] / FS_SYM)
    prior_pen = -0.5 * ((f_cands - cfo_prior_mean) ** 2) / v_f

    best_obj = -1e300
    best_cfo = cfo_prior_mean
    best_t_rx = t_rx_local
    best_y_sym = y_sym_local
    for cand_t, cand_y in ((t_rx_local, y_sym_local), (timing_prior, y_sym_pr)):
        zk = cand_y[KNOWN_IDX] * np.conj(pkt.x_known)
        coh2 = np.abs(rot_mat @ zk) ** 2
        ll_f = coh2 / (nv_eff_prior * (n_p + nv_eff_prior / p_h_eff))
        obj = ll_f + prior_pen
        idx_b = int(np.argmax(obj))
        if float(obj[idx_b]) > best_obj:
            best_obj = float(obj[idx_b])
            best_cfo = float(f_cands[idx_b])
            best_t_rx = cand_t
            best_y_sym = cand_y

    # MMSE channel estimate at best_cfo using reference-informed signal power & noise variance
    n_idx = np.arange(PKT_LEN, dtype=np.float64)
    rot_all = np.exp(2j * math.pi * best_cfo * n_idx / FS_SYM)
    y_derot = best_y_sym * np.conj(rot_all)
    yk = y_derot[KNOWN_IDX]
    xk = pkt.x_known
    h_ls = complex(np.vdot(xk, yk) / n_p)
    resid = yk - h_ls * xk
    nv_resid = float(np.sum(np.abs(resid) ** 2) / max(int(n_p) - 2, 1))
    nv_est = max(
        cfg.noise_floor_mult * nv_bg,
        0.25 * nv_resid + 0.45 * nv_prior + 0.30 * nv_bg,
    )
    lam = p_h_eff / (p_h_eff + nv_est / n_p)
    h_mmse = lam * h_ls
    h_var_post = (p_h_eff * (nv_est / n_p)) / (p_h_eff + nv_est / n_p)
    # Include residual CFO variance contribution to effective predictive variance across packet
    dt_center = (n_idx - (PKT_LEN - 1) / 2.0) / FS_SYM
    # Posterior CFO variance combines Fisher information from pilots + prior precision
    snr_pil = p_h_eff / max(nv_est, 1e-18)
    t_rms2 = float(np.mean(((known_f - np.mean(known_f)) / FS_SYM) ** 2))
    fisher_f = 2.0 * (2.0 * math.pi) ** 2 * n_p * snr_pil * t_rms2
    post_var_f = 1.0 / (fisher_f + 1.0 / v_f)
    phase_var_n = ((2.0 * math.pi * dt_center) ** 2) * post_var_f
    nv_eff = (nv_est + h_var_post + (abs(h_mmse) ** 2) * np.minimum(phase_var_n, 0.5)) * cfg.var_calib_scale

    h_hat = np.full(PKT_LEN, h_mmse, dtype=np.complex128) * rot_all
    x_eq = np.conj(h_hat) * best_y_sym / (np.abs(h_hat) ** 2 + nv_eff)
    bit_llrs = qpsk_bit_llrs_gaussian(best_y_sym, h_hat, nv_eff)
    hard_syms = qpsk_hard_symbols(x_eq)

    out_extra = {
        "cfo_prior_mean": cfo_prior_mean,
        "cfo_prior_std": math.sqrt(max(cfo_prior_var, 0.0)),
        "h_pow_prior": h_pow_prior,
    }
    if extra:
        out_extra.update(extra)
    return ReceiverOutput(
        y_sym=best_y_sym,
        h_hat=h_hat,
        nv_hat=nv_eff,
        x_eq=x_eq,
        bit_llrs=bit_llrs,
        hard_syms=hard_syms,
        est_timing_rx=best_t_rx,
        est_cfo_hz=best_cfo,
        est_noise_var=nv_est,
        elapsed_ms=(time.perf_counter() - t0) * 1000.0,
        extra=out_extra,
    )


def score_reference_pool(
    pkt: PacketSample,
    pool: Sequence[RefEntry],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact pilot marginal log-likelihood scores and normalized correlations for all M refs."""
    cache: dict[float, np.ndarray] = {}
    scores = np.empty(len(pool), dtype=np.float64)
    corrs = np.empty(len(pool), dtype=np.float64)
    for j, ref in enumerate(pool):
        ll, cr = pilot_marginal_loglik_for_ref(pkt.y_iq_4x, pkt.x_known, ref, y_sym_cache=cache)
        scores[j] = ll
        corrs[j] = cr
    return scores, corrs


def aggregate_ref_weights(
    pool: Sequence[RefEntry],
    weights: np.ndarray,
    *,
    cfo_var_floor: float = 40.0 ** 2,
) -> dict[str, float]:
    """Aggregate reference pool values using probability weights alpha (summing to 1)."""
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 0.0)
    w /= max(float(np.sum(w)), 1e-18)
    cfos = np.array([r.est_cfo_hz for r in pool], dtype=np.float64)
    pows = np.array([r.h_pow for r in pool], dtype=np.float64)
    nvs = np.array([r.nv_est for r in pool], dtype=np.float64)
    tims = np.array([r.est_timing_rx for r in pool], dtype=np.float64)

    f_mean = float(np.sum(w * cfos))
    f_var = float(np.sum(w * ((cfos - f_mean) ** 2))) + cfo_var_floor
    # Geometric mean for positive power/noise spans across 20 dB dynamic range
    p_mean = float(np.exp(np.sum(w * np.log(np.maximum(pows, 1e-18)))))
    nv_mean = float(np.exp(np.sum(w * np.log(np.maximum(nvs, 1e-18)))))
    t_mean = float(np.sum(w * tims))
    return {
        "cfo_prior_mean": f_mean,
        "cfo_prior_var": f_var,
        "h_pow_prior": p_mean,
        "nv_prior": nv_mean,
        "timing_prior": t_mean,
    }


def run_experiment_a_arms(
    pkt: PacketSample,
    pool: Sequence[RefEntry],
    cfg: ClassicalConfig,
    *,
    local_k: int = 4,
    content_temp: float = 1.5,
    snr_temp: float = 5.0,
) -> tuple[dict[str, ReceiverOutput], dict[str, Any]]:
    """Run all paired arms of Experiment A on one packet with identical reference pool."""
    M = len(pool)
    scores, _ = score_reference_pool(pkt, pool)
    snr_dbs = np.array([r.est_snr_db for r in pool], dtype=np.float64)

    # Weight vectors for each arm (all sum to 1)
    w_local_k = np.zeros(M, dtype=np.float64)
    w_local_k[: min(local_k, M)] = 1.0 / float(min(local_k, M))

    w_unif = np.full(M, 1.0 / float(M), dtype=np.float64)
    w_snr = _softmax(snr_dbs, temp=snr_temp)
    w_content = _softmax(scores, temp=content_temp)
    w_cold = np.zeros(M, dtype=np.float64)
    w_cold[int(np.argmax(scores))] = 1.0
    w_hot = _softmax(scores, temp=content_temp * 15.0)

    # Audit metrics (offline only: same-capture weight mass & content addressing theory terms)
    same_cap_mask = np.array([1.0 if r.capture_id == pkt.capture_id else 0.0 for r in pool], dtype=np.float64)
    same_freq_mask = np.array([1.0 if r.freq_mhz == pkt.freq_mhz else 0.0 for r in pool], dtype=np.float64)
    ref_cfos = np.array([r.est_cfo_hz for r in pool], dtype=np.float64)
    # Effective gap between best same-capture score and best mismatched-capture score
    if np.any(same_cap_mask > 0.5) and np.any(same_cap_mask < 0.5):
        s_match = float(np.max(scores[same_cap_mask > 0.5]))
        s_mismatch = float(np.max(scores[same_cap_mask < 0.5]))
        score_margin = s_match - s_mismatch
    else:
        score_margin = float("nan")

    arms_spec = [
        ("A1_local_window_k4", w_local_k, 1.0),
        ("A2_global_uniform", w_unif, 1.0),
        ("A3_global_snr_weight", w_snr, 1.0),
        ("A4_global_content_attn", w_content, 1.0),
        ("A5_content_cold_hard", w_cold, 1.0),
        ("A6_content_hot", w_hot, 1.0),
    ]

    outputs: dict[str, ReceiverOutput] = {}
    audit: dict[str, Any] = {
        "unit_id": pkt.unit_id,
        "capture_id": pkt.capture_id,
        "nom_snr_db": pkt.nom_snr_db,
        "freq_mhz": pkt.freq_mhz,
        "offline_cfo_hz": pkt.offline_cfo_hz,
        "score_margin_match_vs_mismatch": score_margin,
        "pool_M": M,
        "local_k": local_k,
        "same_cap_in_local_k": int(np.sum(same_cap_mask[:local_k])),
        "same_cap_total": int(np.sum(same_cap_mask)),
    }

    for name, w, p_str in arms_spec:
        agg = aggregate_ref_weights(pool, w)
        hit_cap = float(np.sum(w * same_cap_mask))
        hit_freq = float(np.sum(w * same_freq_mask))
        cfo_prior_err2 = (agg["cfo_prior_mean"] - pkt.offline_cfo_hz) ** 2
        # L2 weight distance to ideal uniform mass over same-capture refs (for Eq. 2 check)
        n_sc = float(np.sum(same_cap_mask))
        e_ideal = same_cap_mask / max(n_sc, 1.0)
        w_l2_err2 = float(np.sum((w - e_ideal) ** 2))
        w_var_sum = float(np.sum((w - 1.0 / M) ** 2))
        out = run_prior_conditioned_receiver(
            pkt,
            cfg,
            cfo_prior_mean=agg["cfo_prior_mean"],
            cfo_prior_var=agg["cfo_prior_var"],
            h_pow_prior=agg["h_pow_prior"],
            nv_prior=agg["nv_prior"],
            timing_prior=agg["timing_prior"],
            prior_strength=p_str,
            extra={
                "same_cap_weight": hit_cap,
                "same_freq_weight": hit_freq,
                "cfo_prior_abs_err": abs(agg["cfo_prior_mean"] - pkt.offline_cfo_hz),
            },
        )
        outputs[name] = out
        audit[f"{name}__hit_cap"] = hit_cap
        audit[f"{name}__hit_freq"] = hit_freq
        audit[f"{name}__cfo_prior_err2"] = cfo_prior_err2
        audit[f"{name}__w_l2_err2"] = w_l2_err2
        audit[f"{name}__w_var_sum"] = w_var_sum
        audit[f"{name}__ref_cfo_est"] = float(np.sum(w * ref_cfos))
    return outputs, audit


# =============================================================================
# Experiment B: Common Physical Candidate Bank — Hard vs Linear Fusion vs Mixture
# =============================================================================

@dataclass
class CandidateBank:
    """Shared L-hypothesis physical candidate bank for one packet (strictly pilot-only)."""
    y_sym: np.ndarray        # (256,) complex128
    timing_rx: float
    h_cands: np.ndarray      # (L, 256) complex128 per-candidate channel predictions
    nv_cands: np.ndarray     # (L, 256) float64 per-candidate effective predictive variance
    log_ev: np.ndarray       # (L,) float64 exact single-count pilot log-evidence + log-prior
    cand_cfos: np.ndarray    # (L,) float64 per-candidate CFO
    cand_feats: np.ndarray   # (L, 12) float64 strictly pilot-observable features for Exp C
    cand_names: tuple[str, ...]
    build_ms: float


def _fit_scalar_cfo_branch(
    y_sym: np.ndarray,
    x_known: np.ndarray,
    cfo_hz: float,
    p_h_prior: float,
    nv_est: float,
    var_calib: float,
    *,
    extra_cfo_var: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    """Fit complex scalar MMSE channel at fixed CFO hypothesis and return exact marginal log-evidence.

    Exact complex Gaussian marginal log-likelihood of y_known given x_known & h ~ CN(0, p_h_prior):
        log p(y_k | x_k, f) = -N_p log(pi nv) - log(1 + N_p p_h / nv) - ||y_k||^2/nv + |z_sum|^2 / (nv*(N_p + nv/p_h))
    Single-count: pilots are scored ONCE via exact marginal evidence (no double counting).
    """
    n_idx = np.arange(PKT_LEN, dtype=np.float64)
    rot_all = np.exp(2j * math.pi * cfo_hz * n_idx / FS_SYM)
    y_derot = y_sym * np.conj(rot_all)
    yk = y_derot[KNOWN_IDX]
    n_p = float(len(KNOWN_IDX))
    z_sum = complex(np.vdot(x_known, yk))
    y_e = float(np.sum(np.abs(yk) ** 2))
    p_h = max(p_h_prior, 1e-18)
    nv = max(nv_est, 1e-18)

    log_ev = (
        -n_p * math.log(math.pi * nv)
        - math.log(1.0 + n_p * p_h / nv)
        - y_e / nv
        + (abs(z_sum) ** 2) / (nv * (n_p + nv / p_h))
    )
    post_var_h = (p_h * (nv / n_p)) / (p_h + nv / n_p)
    h_post = (p_h / (p_h * n_p + nv)) * z_sum
    h_vec = np.full(PKT_LEN, h_post, dtype=np.complex128) * rot_all

    # Separate sync vs mid+tail residuals (strictly pilot-observable diagnostic features)
    n_sync = len(SYNC_IDX)
    res_all = yk - h_post * x_known
    res_sync = float(np.mean(np.abs(res_all[:n_sync]) ** 2) / nv)
    res_tail = float(np.mean(np.abs(res_all[n_sync:]) ** 2) / nv)
    # Phase consistency between first half (sync) and second half (mid+tail)
    h_pre = complex(np.vdot(x_known[:n_sync], yk[:n_sync]) / float(n_sync))
    h_post_half = complex(np.vdot(x_known[n_sync:], yk[n_sync:]) / float(len(KNOWN_IDX) - n_sync))
    ph_diff = abs(math.atan2((h_post_half * np.conj(h_pre)).imag, (h_post_half * np.conj(h_pre)).real))

    dt = (n_idx - (PKT_LEN - 1) / 2.0) / FS_SYM
    nv_vec = (nv + post_var_h + (abs(h_post) ** 2) * ((2.0 * math.pi * dt) ** 2) * extra_cfo_var) * var_calib
    feats = {
        "res_sync": res_sync,
        "res_tail": res_tail,
        "ph_diff": ph_diff,
        "h_abs": abs(h_post),
        "post_var_ratio": post_var_h / nv,
    }
    return h_vec, nv_vec, float(log_ev), feats


def _fit_slope_branch(
    y_sym: np.ndarray,
    x_known: np.ndarray,
    cfo_hz: float,
    p_h_prior: float,
    nv_est: float,
    slope_reg: float,
    var_calib: float,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    """Fit 2-DoF complex affine channel h_env(t) = c0 + c1*t with exact marginal log-evidence."""
    n_idx = np.arange(PKT_LEN, dtype=np.float64)
    rot_all = np.exp(2j * math.pi * cfo_hz * n_idx / FS_SYM)
    y_derot = y_sym * np.conj(rot_all)
    yk = y_derot[KNOWN_IDX]
    t_all = (n_idx - (PKT_LEN - 1) / 2.0) / (PKT_LEN / 2.0)
    t_k = t_all[KNOWN_IDX]
    A = np.column_stack([x_known, x_known * t_k])  # (16, 2)
    p_h = max(p_h_prior, 1e-18)
    nv = max(nv_est, 1e-18)
    v_prior = np.array([p_h, p_h / max(slope_reg, 0.2)], dtype=np.float64)
    P_inv = np.diag(1.0 / v_prior)
    # Posterior precision M = P_inv + A^H A / nv
    M_prec = P_inv + (A.conj().T @ A) / nv
    cov_c = np.linalg.inv(M_prec)
    rhs = (A.conj().T @ yk) / nv
    coef = cov_c @ rhs
    # Exact complex Gaussian marginal log-likelihood via matrix determinant & Woodbury:
    # log p(y) = -N_p log(pi nv) - log det(I + diag(v_prior) A^H A / nv) - ||y||^2/nv + rhs^H cov_c rhs
    sign, logdet_m = np.linalg.slogdet(M_prec)
    logdet_p_inv = float(np.sum(np.log(1.0 / v_prior)))
    log_ev = (
        -len(KNOWN_IDX) * math.log(math.pi * nv)
        - float(logdet_m - logdet_p_inv)
        - float(np.sum(np.abs(yk) ** 2)) / nv
        + float(np.real(np.vdot(rhs, coef)))
    )
    B_all = np.column_stack([np.ones(PKT_LEN), t_all])
    h_env = B_all @ coef
    ch_var = np.maximum(np.real(np.einsum("ni,ij,nj->n", B_all, cov_c, B_all)), 0.0)
    h_vec = h_env * rot_all
    nv_vec = (nv + ch_var) * var_calib

    n_sync = len(SYNC_IDX)
    res_all = yk - (A @ coef)
    feats = {
        "res_sync": float(np.mean(np.abs(res_all[:n_sync]) ** 2) / nv),
        "res_tail": float(np.mean(np.abs(res_all[n_sync:]) ** 2) / nv),
        "ph_diff": float(abs(coef[1]) / (abs(coef[0]) + 1e-18)),
        "h_abs": float(abs(coef[0])),
        "post_var_ratio": float(np.mean(ch_var) / nv),
    }
    return h_vec, nv_vec, float(log_ev), feats


def _fit_piecewise_jump_branch(
    y_sym: np.ndarray,
    x_known: np.ndarray,
    cfo_hz: float,
    p_h_prior: float,
    nv_est: float,
    split_sym: int,
    var_calib: float,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float]]:
    """Fit piecewise channel allowing a mid-packet phase/gain jump at `split_sym` (exact marginal evidence)."""
    n_idx = np.arange(PKT_LEN, dtype=np.float64)
    rot_all = np.exp(2j * math.pi * cfo_hz * n_idx / FS_SYM)
    y_derot = y_sym * np.conj(rot_all)
    yk = y_derot[KNOWN_IDX]
    m_pre = KNOWN_IDX < split_sym
    m_post = ~m_pre
    p_h = max(p_h_prior, 1e-18)
    nv = max(nv_est, 1e-18)

    def _seg_fit(ys: np.ndarray, xs: np.ndarray) -> tuple[complex, float, float]:
        ns = float(len(xs))
        zs = complex(np.vdot(xs, ys))
        ye = float(np.sum(np.abs(ys) ** 2))
        ev = -ns * math.log(math.pi * nv) - math.log(1.0 + ns * p_h / nv) - ye / nv + (abs(zs) ** 2) / (nv * (ns + nv / p_h))
        h_s = (p_h / (p_h * ns + nv)) * zs
        v_s = (p_h * (nv / ns)) / (p_h + nv / ns)
        return h_s, v_s, float(ev)

    h1, v1, ev1 = _seg_fit(yk[m_pre], x_known[m_pre])
    h2, v2, ev2 = _seg_fit(yk[m_post], x_known[m_post])
    # Exact joint marginal evidence of independent pre/post channel gains (naturally pays Occam penalty for 2nd gain!)
    log_ev = ev1 + ev2
    h_env = np.where(n_idx < split_sym, h1, h2)
    ch_var = np.where(n_idx < split_sym, v1, v2)
    h_vec = h_env * rot_all
    nv_vec = (nv + ch_var) * var_calib

    res_pre = float(np.mean(np.abs(yk[m_pre] - h1 * x_known[m_pre]) ** 2) / nv)
    res_post = float(np.mean(np.abs(yk[m_post] - h2 * x_known[m_post]) ** 2) / nv)
    ph_jump = abs(math.atan2((h2 * np.conj(h1)).imag, (h2 * np.conj(h1)).real))
    feats = {
        "res_sync": res_pre,
        "res_tail": res_post,
        "ph_diff": ph_jump,
        "h_abs": 0.5 * (abs(h1) + abs(h2)),
        "post_var_ratio": 0.5 * (v1 + v2) / nv,
    }
    return h_vec, nv_vec, float(log_ev), feats


def build_common_candidate_bank(
    pkt: PacketSample,
    pool: Sequence[RefEntry],
    cfg: ClassicalConfig,
    *,
    content_temp: float = 1.5,
) -> CandidateBank:
    """Build the unified L=12 physical candidate bank shared identically by Exp B & Exp C.

    Strictly pilot-only: reads ONLY pkt.y_iq_4x, pkt.x_known, pkt.bg_noise_var, and pilot-only RefEntries.
    Candidates (L = 12):
      0..6 : 7 multi-hypothesis CFO grid around local pilot ML peak: {-900,-600,-300,0,+300,+600,+900} Hz
      7    : Global content-matched Top-1 cluster CFO + power/noise prior branch
      8    : Global content-matched Top-2 distinct-CFO cluster prior branch
      9    : Linear intra-packet drift (2-DoF affine slope) branch at best content/local CFO
      10   : Piecewise phase-jump branch (split at sym 96, between sync and mid-pilot) at local CFO
      11   : Piecewise phase-jump branch (split at sym 96) at content-matched prior CFO
    """
    t0 = time.perf_counter()
    timing_rx, cfo_local, y_sym, _ = acquire_timing_and_cfo(pkt.y_iq_4x, pkt.x_known, cfg)
    n_idx = np.arange(PKT_LEN, dtype=np.float64)

    # Local pilot power & noise baseline
    yk_raw = y_sym[KNOWN_IDX]
    nv_bg = max(float(pkt.bg_noise_var), 1e-18)
    y_derot_loc = y_sym * np.exp(-2j * math.pi * cfo_local * n_idx / FS_SYM)
    h_loc = complex(np.vdot(pkt.x_known, y_derot_loc[KNOWN_IDX]) / len(KNOWN_IDX))
    nv_res_loc = float(np.sum(np.abs(y_derot_loc[KNOWN_IDX] - h_loc * pkt.x_known) ** 2) / (len(KNOWN_IDX) - 2))
    nv_loc = max(cfg.noise_floor_mult * nv_bg, cfg.noise_blend * nv_res_loc + (1.0 - cfg.noise_blend) * nv_bg)
    p_h_loc = max(float(np.mean(np.abs(yk_raw) ** 2)) - nv_loc, 0.05 * nv_loc)

    # Reference pool content scores for global-cluster candidates (7, 8, 9, 11)
    if len(pool) > 0:
        ref_scores, _ = score_reference_pool(pkt, pool)
        w_ref = _softmax(ref_scores, temp=content_temp)
        agg1 = aggregate_ref_weights(pool, w_ref)
        cfo_ref1 = agg1["cfo_prior_mean"]
        p_h_ref1 = agg1["h_pow_prior"]
        nv_ref1 = max(0.5 * nv_bg, 0.5 * agg1["nv_prior"] + 0.5 * nv_bg)
        # Top-2 distinct CFO cluster (mask references within 180 Hz of cfo_ref1)
        ref_cfos = np.array([r.est_cfo_hz for r in pool], dtype=np.float64)
        far_mask = np.abs(ref_cfos - cfo_ref1) > 180.0
        if np.any(far_mask):
            s2 = np.where(far_mask, ref_scores, -1e9)
            w_ref2 = _softmax(s2, temp=content_temp)
            agg2 = aggregate_ref_weights(pool, w_ref2)
            cfo_ref2 = agg2["cfo_prior_mean"]
            p_h_ref2 = agg2["h_pow_prior"]
            nv_ref2 = max(0.5 * nv_bg, 0.5 * agg2["nv_prior"] + 0.5 * nv_bg)
        else:
            cfo_ref2, p_h_ref2, nv_ref2 = cfo_local + 250.0, p_h_loc, nv_loc
    else:
        cfo_ref1, p_h_ref1, nv_ref1 = cfo_local, p_h_loc, nv_loc
        cfo_ref2, p_h_ref2, nv_ref2 = cfo_local + 250.0, p_h_loc, nv_loc

    # Shared noise scale `nv_common` for apples-to-apples evidence comparison across all 12 candidates
    nv_common = max(cfg.noise_floor_mult * nv_bg, 0.30 * nv_res_loc + 0.35 * nv_ref1 + 0.35 * nv_bg)
    p_h_common = max(0.5 * p_h_loc + 0.5 * p_h_ref1, 0.05 * nv_common)

    # Extract top-3 genuinely competing local CFO spectral peaks from pilot periodogram (min separation 240 Hz)
    f_scan = np.linspace(cfg.cfo_search_min_hz, cfg.cfo_search_max_hz, 161)
    known_f = KNOWN_IDX.astype(np.float64)
    zk_raw = y_sym[KNOWN_IDX] * np.conj(pkt.x_known)
    spec_p = np.abs(np.exp(-2j * math.pi * f_scan[:, None] * known_f[None, :] / FS_SYM) @ zk_raw)
    peak_cfos: list[float] = [cfo_local]
    spec_work = spec_p.copy()
    spec_work[np.abs(f_scan - cfo_local) < 240.0] = -1.0
    for _ in range(2):
        pi = int(np.argmax(spec_work))
        if spec_work[pi] > 0.0:
            fc = float(f_scan[pi])
            peak_cfos.append(fc)
            spec_work[np.abs(f_scan - fc) < 240.0] = -1.0
        else:
            peak_cfos.append(cfo_local + 350.0 * len(peak_cfos))

    # Build L=12 genuinely competing physical hypotheses:
    #   0..2 : Top-3 local pilot spectral CFO peaks (captures sidelobe/grating-lobe ambiguity in low SNR!)
    #   3..4 : Fine +-180 Hz neighborhood around primary local peak (tracks phase drift uncertainty)
    #   5..6 : Global content Top-1 & Top-2 distinct-CFO cluster priors (from cross-session pool)
    #   7    : Global content Top-3 distinct-CFO cluster prior (or local + 450 Hz fallback)
    #   8    : 2-DoF intra-packet linear drift (affine slope) branch at content-matched CFO (tracks f708 1510Hz beat)
    #   9    : 2-DoF affine slope branch at local ML CFO
    #  10..11: Piecewise mid-packet phase-jump branches (split at sym=96) at local CFO & content prior CFO
    if len(pool) > 0:
        ref_cfos_all = np.array([r.est_cfo_hz for r in pool], dtype=np.float64)
        far3 = (np.abs(ref_cfos_all - cfo_ref1) > 180.0) & (np.abs(ref_cfos_all - cfo_ref2) > 180.0)
        if np.any(far3):
            s3 = np.where(far3, ref_scores, -1e9)
            w_ref3 = _softmax(s3, temp=content_temp)
            agg3 = aggregate_ref_weights(pool, w_ref3)
            cfo_ref3, p_h_ref3 = agg3["cfo_prior_mean"], agg3["h_pow_prior"]
        else:
            cfo_ref3, p_h_ref3 = cfo_ref1 - 220.0, p_h_ref1
    else:
        cfo_ref3, p_h_ref3 = cfo_local - 220.0, p_h_loc

    cand_h_list: list[np.ndarray] = []
    cand_nv_list: list[np.ndarray] = []
    cand_ev_list: list[float] = []
    cand_cfo_list: list[float] = []
    cand_names: list[str] = []
    raw_feats: list[dict[str, float]] = []
    family_ids: list[int] = []

    # Family 0: 5 multi-peak local CFO branches (total family prior 0.35 -> 0.07 each)
    local_cfo_specs = [
        ("cfo_peak1", peak_cfos[0]),
        ("cfo_peak2", peak_cfos[1]),
        ("cfo_peak3", peak_cfos[2]),
        ("cfo_loc_m180", cfo_local - 180.0),
        ("cfo_loc_p180", cfo_local + 180.0),
    ]
    for nm, f_c in local_cfo_specs:
        h_v, nv_v, ev, ft = _fit_scalar_cfo_branch(
            y_sym, pkt.x_known, f_c, p_h_common, nv_common, cfg.var_calib_scale, extra_cfo_var=50.0 ** 2
        )
        cand_h_list.append(h_v)
        cand_nv_list.append(nv_v)
        cand_ev_list.append(ev + math.log(0.35 / len(local_cfo_specs)))
        cand_cfo_list.append(f_c)
        cand_names.append(nm)
        raw_feats.append(ft)
        family_ids.append(0)

    # Family 1: Global content Top-1, Top-2, Top-3 reference cluster branches (total prior 0.30)
    for nm, f_c, p_c, pr_w in (
        ("global_ref_top1", cfo_ref1, p_h_ref1, 0.16),
        ("global_ref_top2", cfo_ref2, p_h_ref2, 0.08),
        ("global_ref_top3", cfo_ref3, p_h_ref3, 0.06),
    ):
        h_v, nv_v, ev, ft = _fit_scalar_cfo_branch(
            y_sym, pkt.x_known, f_c, max(p_c, 0.05 * nv_common), nv_common, cfg.var_calib_scale, extra_cfo_var=35.0 ** 2
        )
        cand_h_list.append(h_v)
        cand_nv_list.append(nv_v)
        cand_ev_list.append(ev + math.log(pr_w))
        cand_cfo_list.append(f_c)
        cand_names.append(nm)
        raw_feats.append(ft)
        family_ids.append(1)

    # Family 2: Intra-packet linear slope branches (total prior 0.15 -> 0.075 each)
    for nm, f_c in (("slope_drift_ref", cfo_ref1), ("slope_drift_loc", cfo_local)):
        h_sl, nv_sl, ev_sl, ft_sl = _fit_slope_branch(
            y_sym, pkt.x_known, f_c, p_h_common, nv_common, cfg.slope_reg, cfg.var_calib_scale
        )
        cand_h_list.append(h_sl)
        cand_nv_list.append(nv_sl)
        cand_ev_list.append(ev_sl + math.log(0.075))
        cand_cfo_list.append(f_c)
        cand_names.append(nm)
        raw_feats.append(ft_sl)
        family_ids.append(2)

    # Family 3: Piecewise mid-packet phase-jump branches (total prior 0.20 -> 0.10 each)
    for nm, f_c in (("pw_jump_local", cfo_local), ("pw_jump_ref", cfo_ref1)):
        h_pw, nv_pw, ev_pw, ft_pw = _fit_piecewise_jump_branch(
            y_sym, pkt.x_known, f_c, p_h_common, nv_common, 96, cfg.var_calib_scale
        )
        cand_h_list.append(h_pw)
        cand_nv_list.append(nv_pw)
        cand_ev_list.append(ev_pw + math.log(0.10))
        cand_cfo_list.append(f_c)
        cand_names.append(nm)
        raw_feats.append(ft_pw)
        family_ids.append(3)

    h_stack = np.stack(cand_h_list, axis=0)      # (12, 256)
    nv_stack = np.stack(cand_nv_list, axis=0)    # (12, 256)
    log_ev = np.array(cand_ev_list, dtype=np.float64)
    cfo_arr = np.array(cand_cfo_list, dtype=np.float64)

    # Build normalized, scale-invariant, strictly pilot-observable feature matrix (12, 12) for Exp C
    ev_rel = log_ev - float(np.max(log_ev))
    w_init = _softmax(log_ev, temp=1.0)
    h_bar = np.sum(w_init[:, None] * h_stack, axis=0)
    h_scale = math.sqrt(max(p_h_common, 1e-18))
    snr_db_est = 10.0 * math.log10(max(p_h_common / nv_common, 1e-6))

    feats_mat = np.zeros((len(cand_names), 12), dtype=np.float64)
    for r in range(len(cand_names)):
        ft = raw_feats[r]
        dist_to_mean = float(np.sqrt(np.mean(np.abs(h_stack[r] - h_bar) ** 2)) / (h_scale + 1e-18))
        fam_oh = [1.0 if family_ids[r] == f else 0.0 for f in range(4)]
        feats_mat[r] = [
            float(np.clip(ev_rel[r] / 10.0, -5.0, 0.0)),
            math.log1p(max(ft["res_sync"], 0.0)),
            math.log1p(max(ft["res_tail"], 0.0)),
            ft["ph_diff"] / math.pi,
            (cfo_arr[r] - cfo_ref1) / 500.0,
            (cfo_arr[r] - cfo_local) / 500.0,
            dist_to_mean,
            snr_db_est / 20.0,
            *fam_oh,
        ]

    return CandidateBank(
        y_sym=y_sym,
        timing_rx=timing_rx,
        h_cands=h_stack,
        nv_cands=nv_stack,
        log_ev=log_ev,
        cand_cfos=cfo_arr,
        cand_feats=feats_mat,
        cand_names=tuple(cand_names),
        build_ms=(time.perf_counter() - t0) * 1000.0,
    )


_PAYLOAD_MASK = np.zeros(PKT_LEN, dtype=np.float64)
_PAYLOAD_MASK[PAYLOAD_IDX] = 1.0


def decode_from_candidate_bank(
    bank: CandidateBank,
    weights: np.ndarray,
    *,
    mode: str,
    block_temp: float = 0.25,
    h_override: np.ndarray | None = None,
    nv_override: np.ndarray | None = None,
) -> ReceiverOutput:
    """Decode QPSK packet from a fixed candidate bank and posterior weights using specified fusion rule.

    Modes (paired on identical bank & pilot-only prior weights `weights`):
      - `hard_select`: MAP single candidate r* = argmax_r alpha_r
      - `linear_fusion_plugin`: posterior mean h_mix = sum_r alpha_r h_r + TOTAL variance
        (within + between, posterior_fusion §4.1), then Gaussian plug-in QPSK demodulation
      - `linear_within_only`: posterior mean h_mix with WITHIN variance only (tests §4.1 under-variance)
      - `sym_mixture`: per-symbol independent mixture likelihood sum_r alpha_r CN(y_n | h_r x, nv_r)
      - `mixture_likelihood`: EXACT block-joint LOO mixture posterior (posterior_fusion §5.2):
        updates branch posterior beta_r^{(-n)} using unlabeled constellation likelihood across the
        shared payload block, then marginalizes mixture into exact QPSK MAP & log-sum-exp bit LLR
      - `avg_llr_antipattern`: linear average of per-branch LLRs sum_r alpha_r LLR_r (negative control §5.2)
    """
    t0 = time.perf_counter()
    h_c = bank.h_cands if h_override is None else h_override
    nv_c = bank.nv_cands if nv_override is None else nv_override
    w = np.asarray(weights, dtype=np.float64)
    w = np.maximum(w, 0.0)
    w /= max(float(np.sum(w)), 1e-18)

    # Exact posterior mean & within/between covariance decomposition (posterior_fusion §4.1)
    h_mix = np.sum(w[:, None] * h_c, axis=0)
    v_within = np.sum(w[:, None] * nv_c, axis=0)
    v_between = np.sum(w[:, None] * (np.abs(h_c - h_mix[None, :]) ** 2), axis=0)
    v_total = np.maximum(v_within + v_between, 1e-18)

    r_hat = int(np.argmax(w))
    cfo_mix = float(np.sum(w * bank.cand_cfos))
    entropy = float(-np.sum(w * np.log(np.maximum(w, 1e-300))))

    if mode == "hard_select":
        h_out = h_c[r_hat]
        nv_out = nv_c[r_hat]
        x_eq = np.conj(h_out) * bank.y_sym / (np.abs(h_out) ** 2 + nv_out)
        bit_llrs = qpsk_bit_llrs_gaussian(bank.y_sym, h_out, nv_out)
        hard_syms = qpsk_hard_symbols(x_eq)
        est_cfo = float(bank.cand_cfos[r_hat])
    elif mode == "linear_fusion_plugin":
        h_out = h_mix
        nv_out = v_total
        x_eq = np.conj(h_out) * bank.y_sym / (np.abs(h_out) ** 2 + nv_out)
        bit_llrs = qpsk_bit_llrs_gaussian(bank.y_sym, h_out, nv_out)
        hard_syms = qpsk_hard_symbols(x_eq)
        est_cfo = cfo_mix
    elif mode == "linear_within_only":
        h_out = h_mix
        nv_out = np.maximum(v_within, 1e-18)
        x_eq = np.conj(h_out) * bank.y_sym / (np.abs(h_out) ** 2 + nv_out)
        bit_llrs = qpsk_bit_llrs_gaussian(bank.y_sym, h_out, nv_out)
        hard_syms = qpsk_hard_symbols(x_eq)
        est_cfo = cfo_mix
    elif mode == "sym_mixture":
        h_out = h_mix
        nv_out = v_total
        hard_syms, bit_llrs, sym_ll = mixture_qpsk_posterior(bank.y_sym, h_c, nv_c, w)
        sym_ll_s = sym_ll - np.max(sym_ll, axis=1, keepdims=True)
        sym_p = np.exp(sym_ll_s)
        sym_p /= np.sum(sym_p, axis=1, keepdims=True)
        x_eq = sym_p @ QPSK_CONSTELLATION
        est_cfo = cfo_mix
    elif mode == "mixture_likelihood":
        # Full block-joint LOO posterior mixture (strictly unlabeled payload constellation marginalization)
        hard_syms, bit_llrs, sym_ll, beta_full = joint_mixture_qpsk_posterior(
            bank.y_sym, h_c, nv_c, w, payload_mask=_PAYLOAD_MASK, block_temp=block_temp
        )
        # Keep holdout channel readout strictly pilot-only (h_mix, v_total) so holdout evaluation
        # never uses payload observations, OR note exact posterior readout! Using pilot-only (h_mix, v_total)
        # keeps channel estimation strictly comparable on identical pilot information I!
        h_out = h_mix
        nv_out = v_total
        sym_ll_s = sym_ll - np.max(sym_ll, axis=1, keepdims=True)
        sym_p = np.exp(sym_ll_s)
        sym_p /= np.sum(sym_p, axis=1, keepdims=True)
        x_eq = sym_p @ QPSK_CONSTELLATION
        est_cfo = float(np.sum(beta_full * bank.cand_cfos))
    elif mode == "avg_llr_antipattern":
        h_out = h_mix
        nv_out = v_total
        llrs_each = np.stack([qpsk_bit_llrs_gaussian(bank.y_sym, h_c[r], nv_c[r]) for r in range(len(w))], axis=0)
        bit_llrs = np.sum(w[:, None] * llrs_each, axis=0)
        dec_bits = (bit_llrs > 0.0).astype(np.uint8)
        hard_syms = bits_to_qpsk(dec_bits)
        x_eq = np.conj(h_out) * bank.y_sym / (np.abs(h_out) ** 2 + nv_out)
        est_cfo = cfo_mix
    else:
        raise ValueError(f"Unknown Experiment B decode mode: {mode}")

    dec_ms = (time.perf_counter() - t0) * 1000.0
    proj_gap = float(np.mean(np.abs(h_c[r_hat] - h_mix) ** 2))
    return ReceiverOutput(
        y_sym=bank.y_sym,
        h_hat=h_out,
        nv_hat=nv_out,
        x_eq=x_eq,
        bit_llrs=bit_llrs,
        hard_syms=hard_syms,
        est_timing_rx=bank.timing_rx,
        est_cfo_hz=est_cfo,
        est_noise_var=float(np.median(nv_out)),
        elapsed_ms=bank.build_ms + dec_ms,
        extra={
            "decode_only_ms": dec_ms,
            "bank_build_ms": bank.build_ms,
            "top_cand_idx": r_hat,
            "top_cand_weight": float(w[r_hat]),
            "weight_entropy": entropy,
            "proj_gap_mse": proj_gap,
            "between_var_mean": float(np.mean(v_between)),
            "within_var_mean": float(np.mean(v_within)),
        },
    )


def run_experiment_b_arms(
    bank: CandidateBank,
    *,
    post_temp: float = 1.5,
    block_temp: float = 0.25,
) -> dict[str, ReceiverOutput]:
    """Run all 6 Experiment B arms on the EXACT same CandidateBank and pilot posterior weights."""
    w = _softmax(bank.log_ev, temp=post_temp)
    return {
        "B1_hard_select": decode_from_candidate_bank(bank, w, mode="hard_select"),
        "B2_linear_fusion_plugin": decode_from_candidate_bank(bank, w, mode="linear_fusion_plugin"),
        "B2b_linear_within_only": decode_from_candidate_bank(bank, w, mode="linear_within_only"),
        "B3a_sym_mixture": decode_from_candidate_bank(bank, w, mode="sym_mixture"),
        "B3_mixture_likelihood": decode_from_candidate_bank(bank, w, mode="mixture_likelihood", block_temp=block_temp),
        "B4_avg_llr_antipattern": decode_from_candidate_bank(bank, w, mode="avg_llr_antipattern"),
    }


# =============================================================================
# Experiment C: Lightweight Trainable PyTorch Attention / Weight + Residual Net
# =============================================================================

def _require_torch():
    import torch
    torch.set_num_threads(1)
    return torch


def build_c_sample_arrays(
    pkt: PacketSample,
    bank: CandidateBank,
) -> dict[str, np.ndarray]:
    """Pack compact float32 tensors for fast single-threaded PyTorch training/inference on fixed CandidateBank."""
    # Scale normalize complex signals per packet by rms(|y_sym|) so neural inputs/losses are well-conditioned
    scale = float(np.sqrt(np.mean(np.abs(bank.y_sym) ** 2)) + 1e-18)
    y_pay = bank.y_sym[PAYLOAD_IDX] / scale
    h_pay = bank.h_cands[:, PAYLOAD_IDX] / scale
    nv_pay = bank.nv_cands[:, PAYLOAD_IDX] / (scale * scale)
    y_hold = bank.y_sym[HOLDOUT_IDX] / scale
    h_hold = bank.h_cands[:, HOLDOUT_IDX] / scale
    nv_hold = bank.nv_cands[:, HOLDOUT_IDX] / (scale * scale)
    x_hold = pkt.x_all_true[HOLDOUT_IDX]
    bits_pay = qpsk_to_bits(pkt.x_all_true[PAYLOAD_IDX]).astype(np.float32)

    return {
        "feats": bank.cand_feats.astype(np.float32),                 # (L, 12)
        "log_ev": (bank.log_ev - float(np.max(bank.log_ev))).astype(np.float32),  # (L,)
        "y_pay_re": y_pay.real.astype(np.float32),                   # (224,)
        "y_pay_im": y_pay.imag.astype(np.float32),
        "h_pay_re": h_pay.real.astype(np.float32),                   # (L, 224)
        "h_pay_im": h_pay.imag.astype(np.float32),
        "nv_pay": np.maximum(nv_pay, 1e-8).astype(np.float32),       # (L, 224)
        "bits_pay": bits_pay,                                        # (448,)
        "y_hold_re": y_hold.real.astype(np.float32),                 # (16,)
        "y_hold_im": y_hold.imag.astype(np.float32),
        "h_hold_re": h_hold.real.astype(np.float32),                 # (L, 16)
        "h_hold_im": h_hold.imag.astype(np.float32),
        "nv_hold": np.maximum(nv_hold, 1e-8).astype(np.float32),     # (L, 16)
        "x_hold_re": x_hold.real.astype(np.float32),
        "x_hold_im": x_hold.imag.astype(np.float32),
    }


def train_and_infer_experiment_c(
    train_pkts: Sequence[PacketSample],
    train_banks: Sequence[CandidateBank],
    dev_pkts: Sequence[PacketSample],
    dev_banks: Sequence[CandidateBank],
    test_pkts: Sequence[PacketSample],
    test_banks: Sequence[CandidateBank],
    *,
    init_temp: float = 1.5,
    block_temp: float = 0.25,
    seeds: Sequence[int] = (20261121, 20261122, 20261123),
    epochs: int = 45,
    batch_size: int = 64,
    lr: float = 4e-3,
) -> dict[str, Any]:
    """Train two lightweight single-thread PyTorch modules over the FIXED candidate bank:

      - C1 (`C1_attn_weight_net`): Cross-candidate self-attention weight calibration network
        predicting residual logits over fixed physical log-evidence (NO channel modification).
      - C2 (`C2_attn_weight_resid_net`): Same attention weight network + bounded per-candidate
        complex gain/phase residual & variance calibration head.

    Both feed the EXACT SAME fixed mixture-likelihood QPSK decoder (`decode_from_candidate_bank`).
    Strictly trained on `train`, model-selected & temperature-calibrated on `dev`, evaluated on `test`.
    """
    torch = _require_torch()
    import torch.nn as nn
    import torch.nn.functional as F

    class LightweightFusionNet(nn.Module):
        def __init__(self, in_dim: int = 12, d_model: int = 24, use_resid: bool = False, init_t: float = 1.5):
            super().__init__()
            self.use_resid = use_resid
            self.enc = nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            # Single-head cross-candidate scaled dot-product self-attention
            self.wq = nn.Linear(d_model, d_model, bias=False)
            self.wk = nn.Linear(d_model, d_model, bias=False)
            self.wv = nn.Linear(d_model, d_model, bias=False)
            self.norm = nn.LayerNorm(d_model)
            self.logit_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
            )
            # Learnable log inverse-temperature on physical evidence baseline
            self.log_inv_temp = nn.Parameter(torch.tensor(math.log(1.0 / max(init_t, 0.1)), dtype=torch.float32))
            if use_resid:
                # Bounded per-candidate (d_amp, d_phase, d_logvar) residual head initialized near zero
                self.resid_head = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Linear(d_model // 2, 3),
                )
                nn.init.zeros_(self.resid_head[-1].weight)
                nn.init.zeros_(self.resid_head[-1].bias)
            nn.init.zeros_(self.logit_head[-1].weight)
            nn.init.zeros_(self.logit_head[-1].bias)

        def forward(self, feats: torch.Tensor, log_ev: torch.Tensor):
            # feats: (B, L, D), log_ev: (B, L)
            x = self.enc(feats)
            q = self.wq(x)
            k = self.wk(x)
            v = self.wv(x)
            attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
            attn_w = torch.softmax(attn_logits, dim=-1)
            h = self.norm(x + torch.matmul(attn_w, v))
            delta_s = self.logit_head(h).squeeze(-1)  # (B, L)
            inv_temp = torch.exp(self.log_inv_temp).clamp(0.05, 5.0)
            logits = inv_temp * log_ev + delta_s
            weights = torch.softmax(logits, dim=-1)
            if self.use_resid:
                r_raw = self.resid_head(h)  # (B, L, 3)
                d_amp = 0.15 * torch.tanh(r_raw[..., 0])
                d_ph = 0.18 * torch.tanh(r_raw[..., 1])
                d_lv = 0.60 * torch.tanh(r_raw[..., 2])
                return weights, d_amp, d_ph, d_lv
            return weights, None, None, None

    # Differentiable exact mixture QPSK bit LLR loss in PyTorch (matches evaluator.mixture_qpsk_posterior)
    c_re = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=torch.float32) / math.sqrt(2.0)
    c_im = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float32) / math.sqrt(2.0)

    def batch_loss(model: LightweightFusionNet, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        w, d_amp, d_ph, d_lv = model(batch["feats"], batch["log_ev"])
        h_re = batch["h_pay_re"]
        h_im = batch["h_pay_im"]
        nv = batch["nv_pay"]
        if d_amp is not None:
            g_re = (1.0 + d_amp) * torch.cos(d_ph)
            g_im = (1.0 + d_amp) * torch.sin(d_ph)
            h_re_new = h_re * g_re.unsqueeze(-1) - h_im * g_im.unsqueeze(-1)
            h_im_new = h_re * g_im.unsqueeze(-1) + h_im * g_re.unsqueeze(-1)
            h_re, h_im = h_re_new, h_im_new
            nv = nv * torch.exp(d_lv).unsqueeze(-1)

        # (B, L, N, 4)
        diff_re = batch["y_pay_re"][:, None, :, None] - (
            h_re[..., None] * c_re[None, None, None, :] - h_im[..., None] * c_im[None, None, None, :]
        )
        diff_im = batch["y_pay_im"][:, None, :, None] - (
            h_re[..., None] * c_im[None, None, None, :] + h_im[..., None] * c_re[None, None, None, :]
        )
        log_g = -torch.log(math.pi * nv[..., None]) - (diff_re ** 2 + diff_im ** 2) / nv[..., None]
        # Exact differentiable leave-one-out block-joint posterior matching fixed backend decoder
        log_py_sym = torch.logsumexp(log_g, dim=-1) - math.log(4.0)  # (B, L, N)
        tot_pay_ll = torch.sum(log_py_sym, dim=-1, keepdim=True)     # (B, L, 1)
        loo_ll = tot_pay_ll - log_py_sym                             # (B, L, N)
        log_w = torch.log(w.clamp_min(1e-12))[:, :, None]            # (B, L, 1)
        log_beta_loo = torch.log_softmax(log_w + float(block_temp) * loo_ll, dim=1)  # (B, L, N)
        sym_ll = torch.logsumexp(log_beta_loo[..., None] + log_g, dim=1)  # (B, N, 4)

        ll_b0_0 = torch.logaddexp(sym_ll[..., 0], sym_ll[..., 1])
        ll_b0_1 = torch.logaddexp(sym_ll[..., 2], sym_ll[..., 3])
        ll_b1_0 = torch.logaddexp(sym_ll[..., 0], sym_ll[..., 2])
        ll_b1_1 = torch.logaddexp(sym_ll[..., 1], sym_ll[..., 3])
        llr0 = ll_b0_1 - ll_b0_0
        llr1 = ll_b1_1 - ll_b1_0
        llrs = torch.stack([llr0, llr1], dim=-1).reshape(sym_ll.shape[0], -1)
        bce = F.binary_cross_entropy_with_logits(llrs, batch["bits_pay"])

        # Auxiliary channel prediction MSE on training payload/holdout? Note: in train split, holdout pilots
        # are available for supervised channel calibration (never touched in dev/test!)
        if d_amp is not None:
            hh_re = batch["h_hold_re"] * g_re.unsqueeze(-1) - batch["h_hold_im"] * g_im.unsqueeze(-1)
            hh_im = batch["h_hold_re"] * g_im.unsqueeze(-1) + batch["h_hold_im"] * g_re.unsqueeze(-1)
        else:
            hh_re, hh_im = batch["h_hold_re"], batch["h_hold_im"]
        hm_re = torch.sum(w.unsqueeze(-1) * hh_re, dim=1)
        hm_im = torch.sum(w.unsqueeze(-1) * hh_im, dim=1)
        yh_re = hm_re * batch["x_hold_re"] - hm_im * batch["x_hold_im"]
        yh_im = hm_re * batch["x_hold_im"] + hm_im * batch["x_hold_re"]
        ch_mse = torch.mean((batch["y_hold_re"] - yh_re) ** 2 + (batch["y_hold_im"] - yh_im) ** 2)
        return bce + 0.10 * ch_mse

    def stack_split(pkts: Sequence[PacketSample], banks: Sequence[CandidateBank]) -> dict[str, torch.Tensor]:
        arrs = [build_c_sample_arrays(p, b) for p, b in zip(pkts, banks)]
        return {k: torch.from_numpy(np.stack([a[k] for a in arrs], axis=0)) for k in arrs[0].keys()}

    tr_data = stack_split(train_pkts, train_banks)
    dv_data = stack_split(dev_pkts, dev_banks)
    n_tr = len(train_pkts)

    results_c: dict[str, Any] = {}
    for variant_name, use_resid in (("C1_attn_weight_net", False), ("C2_attn_weight_resid_net", True)):
        seed_runs: list[dict[str, Any]] = []
        test_outputs_by_seed: dict[int, list[ReceiverOutput]] = {}
        dev_outputs_by_seed: dict[int, list[ReceiverOutput]] = {}

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            tracemalloc.start()
            t_tr0 = time.perf_counter()

            model = LightweightFusionNet(in_dim=12, d_model=24, use_resid=use_resid, init_t=init_temp)
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

            best_dev_loss = float("inf")
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = 0

            gen = torch.Generator().manual_seed(seed)
            for ep in range(1, epochs + 1):
                model.train()
                perm = torch.randperm(n_tr, generator=gen)
                for i0 in range(0, n_tr, batch_size):
                    idx = perm[i0 : i0 + batch_size]
                    mb = {k: v[idx] for k, v in tr_data.items()}
                    opt.zero_grad()
                    loss = batch_loss(model, mb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    opt.step()
                sched.step()

                model.eval()
                with torch.no_grad():
                    dv_loss = float(batch_loss(model, dv_data).item())
                if dv_loss < best_dev_loss:
                    best_dev_loss = dv_loss
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    best_epoch = ep

            train_sec = time.perf_counter() - t_tr0
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            model.load_state_dict(best_state)
            model.eval()

            def _infer_split(banks: Sequence[CandidateBank]) -> list[ReceiverOutput]:
                outs: list[ReceiverOutput] = []
                for bk in banks:
                    t_i0 = time.perf_counter()
                    ft = torch.from_numpy(bk.cand_feats.astype(np.float32))[None, :, :]
                    lev = torch.from_numpy((bk.log_ev - float(np.max(bk.log_ev))).astype(np.float32))[None, :]
                    with torch.no_grad():
                        w_t, da_t, dp_t, dlv_t = model(ft, lev)
                    w_np = w_t[0].numpy().astype(np.float64)
                    if da_t is not None:
                        da = da_t[0].numpy().astype(np.float64)
                        dp = dp_t[0].numpy().astype(np.float64)
                        dlv = dlv_t[0].numpy().astype(np.float64)
                        gain = (1.0 + da) * np.exp(1j * dp)
                        h_mod = bk.h_cands * gain[:, None]
                        nv_mod = bk.nv_cands * np.exp(dlv)[:, None]
                    else:
                        h_mod, nv_mod = None, None
                    nn_ms = (time.perf_counter() - t_i0) * 1000.0
                    ro = decode_from_candidate_bank(
                        bk, w_np, mode="mixture_likelihood", block_temp=block_temp, h_override=h_mod, nv_override=nv_mod
                    )
                    ro.elapsed_ms += nn_ms
                    ro.extra["nn_infer_ms"] = nn_ms
                    ro.extra["seed"] = seed
                    outs.append(ro)
                return outs

            dev_outs = _infer_split(dev_banks)
            test_outs = _infer_split(test_banks)
            dev_outputs_by_seed[seed] = dev_outs
            test_outputs_by_seed[seed] = test_outs
            seed_runs.append(
                {
                    "seed": seed,
                    "n_params": n_params,
                    "best_epoch": best_epoch,
                    "best_dev_loss": best_dev_loss,
                    "train_time_sec": train_sec,
                    "train_peak_py_mem_mb": peak_bytes / (1024.0 * 1024.0),
                    "learned_inv_temp": float(torch.exp(model.log_inv_temp).item()),
                }
            )

        # Select primary seed by lowest dev loss (zero test peeking!) and also keep all seeds
        best_seed_rec = min(seed_runs, key=lambda r: r["best_dev_loss"])
        results_c[variant_name] = {
            "seed_runs": seed_runs,
            "dev_selected_seed": best_seed_rec["seed"],
            "dev_outputs_by_seed": dev_outputs_by_seed,
            "test_outputs_by_seed": test_outputs_by_seed,
        }
    return results_c
