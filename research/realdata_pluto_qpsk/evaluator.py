#!/usr/bin/env python3
"""Unified rigorous evaluator for ADALM-PLUTO QPSK real-data experiments.

Reports metrics partitioned cleanly by:
  - Sync header [0:13]
  - Online pilot [13:21]
  - Held-out pilot probe [21:26] (zero-leakage reference prediction & uncertainty calibration)
  - Payload [26:126] (100 QPSK symbols = 200 Gray bits)

Metrics:
  - SER, symbol reconstruction RMSE/NMSE, EVM (%)
  - BER (explicit Gray QPSK mapping: b0 = Re<0, b1 = Im<0)
  - FER (explicit definition: >=1 symbol error among 100 payload symbols; denominator reported)
  - Channel NMSE (vs offline 126-sym oracle), held-out probe prediction NMSE
  - Uncertainty calibration: proper complex Gaussian NLL, mean z^2, empirical 90% & 95% coverage
  - Coverage & reliability: sync success / sync failure / numeric anomaly counts & rates
  - Engineering cost: single-thread median/mean latency (ms/frame), peak RSS (MB), candidate count, train cost
  - Paired cluster statistics (G=16 contiguous temporal clusters on test, G=10 on dev):
      exact 2^G paired sign-flip permutation p-value, Student-t 95% CI, Holm-Bonferroni correction.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from waveform_frontend import (
    PAYLOAD_SLICE,
    PILOT_SLICE,
    PROBE_SLICE,
    SYNC_SLICE,
    nearest_qpsk_symbols,
    symbols_to_gray_bits,
)


def evaluate_segment_predictions(
    s_hat_eq: np.ndarray,
    s_dec: np.ndarray,
    s_true: np.ndarray,
    bits_dec: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute SER, symbol reconstruction RMSE/NMSE, EVM (%), BER (Gray), and frame error flag for one segment."""
    err_sym = s_hat_eq - s_true
    mse_sym = float(np.mean(err_sym.real ** 2 + err_sym.imag ** 2))
    sig_pow = float(np.mean(s_true.real ** 2 + s_true.imag ** 2)) + 1e-30
    nmse_sym = mse_sym / sig_pow
    rmse_sym = math.sqrt(mse_sym)
    evm_pct = 100.0 * math.sqrt(nmse_sym)

    sym_wrong = np.abs(s_dec - s_true) > 1e-3
    ser = float(np.mean(sym_wrong))

    bits_true = symbols_to_gray_bits(s_true)
    if bits_dec is None:
        bits_dec = symbols_to_gray_bits(s_dec)
    ber = float(np.mean(bits_dec != bits_true))
    fer = 1.0 if bool(np.any(sym_wrong)) else 0.0
    return {
        "ser": ser,
        "ber": ber,
        "fer": fer,
        "sym_rmse": rmse_sym,
        "sym_nmse": nmse_sym,
        "evm_pct": evm_pct,
        "n_syms": int(len(s_true)),
        "n_bits": int(bits_true.size),
    }


def evaluate_probe_uncertainty(
    y_probe: np.ndarray,
    y_pred: np.ndarray,
    pred_var: np.ndarray,
) -> dict[str, float]:
    """Evaluate held-out reference probe prediction NMSE and complex circular Gaussian uncertainty calibration.

    For proper circular complex Gaussian y ~ CN(mu, v):
      - pdf: (1 / (pi * v)) * exp(-|y - mu|^2 / v)
      - NLL: log(pi * v) + |y - mu|^2 / v
      - Normalized squared error z^2 = |y - mu|^2 / v ~ Exp(mean=1)
      - P(z^2 <= -ln(1 - alpha)) = alpha -> 90% threshold = -ln(0.1) = 2.302585, 95% threshold = -ln(0.05) = 2.995732
    """
    err = y_probe - y_pred
    err2 = err.real ** 2 + err.imag ** 2
    y2 = y_probe.real ** 2 + y_probe.imag ** 2
    nmse = float(np.sum(err2) / max(float(np.sum(y2)), 1e-30))
    v = np.maximum(np.asarray(pred_var, dtype=np.float64), 1e-18)
    z2 = err2 / v
    nll = float(np.mean(np.log(np.pi * v) + z2))
    cov90 = float(np.mean(z2 <= 2.302585092994046))
    cov95 = float(np.mean(z2 <= 2.995732273553991))
    return {
        "probe_nmse": nmse,
        "probe_nll": nll,
        "probe_z2_mean": float(np.mean(z2)),
        "probe_cov90": cov90,
        "probe_cov95": cov95,
    }


def evaluate_frame_record(
    frame_idx: int,
    y_sym: np.ndarray,
    s_true: np.ndarray,
    h_oracle: complex,
    d_oracle: complex,
    sync_ok: bool,
    h_pred_per_sym: np.ndarray,    # shape (126,) complex predicted channel at each symbol k
    d_pred: complex,               # scalar predicted DC offset
    var_pred_per_sym: np.ndarray,  # shape (126,) total predictive observation variance at each symbol k
    s_dec_payload: np.ndarray | None = None,
    bits_dec_payload: np.ndarray | None = None,
    s_eq_payload: np.ndarray | None = None,
    latency_ms: float = 0.0,
    n_candidates: int = 1,
) -> dict[str, Any]:
    """Produce a complete, leakage-free per-frame evaluation record across all segments."""
    # Check numeric finiteness
    is_finite = bool(
        np.all(np.isfinite(h_pred_per_sym))
        and np.isfinite(d_pred)
        and np.all(np.isfinite(var_pred_per_sym))
        and np.all(var_pred_per_sym > 0.0)
    )
    if not is_finite:
        return {
            "frame_idx": int(frame_idx),
            "sync_ok": bool(sync_ok),
            "numeric_ok": False,
            "latency_ms": float(latency_ms),
            "n_candidates": int(n_candidates),
        }

    # Safe equalization across all 126 symbols for segment-wise diagnostics
    h_safe = np.where(np.abs(h_pred_per_sym) > 1e-12, h_pred_per_sym, 1e-12 + 0j)
    s_eq_all = (y_sym - d_pred) / h_safe

    # Sync header [0:13]
    s_dec_sync, _ = nearest_qpsk_symbols(s_eq_all[SYNC_SLICE])
    m_sync = evaluate_segment_predictions(s_eq_all[SYNC_SLICE], s_dec_sync, s_true[SYNC_SLICE])

    # Online pilot [13:21]
    s_dec_pil, _ = nearest_qpsk_symbols(s_eq_all[PILOT_SLICE])
    m_pil = evaluate_segment_predictions(s_eq_all[PILOT_SLICE], s_dec_pil, s_true[PILOT_SLICE])

    # Held-out probe [21:26]
    s_dec_prb, _ = nearest_qpsk_symbols(s_eq_all[PROBE_SLICE])
    m_prb = evaluate_segment_predictions(s_eq_all[PROBE_SLICE], s_dec_prb, s_true[PROBE_SLICE])
    y_prb_pred = h_pred_per_sym[PROBE_SLICE] * s_true[PROBE_SLICE] + d_pred
    m_cal = evaluate_probe_uncertainty(y_sym[PROBE_SLICE], y_prb_pred, var_pred_per_sym[PROBE_SLICE])

    # Payload [26:126] (allows custom mixture-MAP symbol/bit decisions!)
    eq_pay = s_eq_all[PAYLOAD_SLICE] if s_eq_payload is None else s_eq_payload
    if s_dec_payload is None:
        s_dec_pay, _ = nearest_qpsk_symbols(eq_pay)
    else:
        s_dec_pay = s_dec_payload
    m_pay = evaluate_segment_predictions(eq_pay, s_dec_pay, s_true[PAYLOAD_SLICE], bits_dec=bits_dec_payload)

    # Channel NMSE vs offline 126-sym oracle (measured at payload center k=75 or mean across frame)
    h_est_center = complex(np.mean(h_pred_per_sym[PILOT_SLICE]))
    ch_nmse = float(abs(h_est_center - h_oracle) ** 2 / max(abs(h_oracle) ** 2, 1e-30))
    # Full waveform prediction NMSE vs oracle clean signal h_ora*s + d_ora
    y_ora_clean = h_oracle * s_true + d_oracle
    y_pred_all = h_pred_per_sym * s_true + d_pred
    clean_pred_nmse = float(
        np.sum(np.abs(y_pred_all - y_ora_clean) ** 2) / max(float(np.sum(np.abs(y_ora_clean) ** 2)), 1e-30)
    )

    return {
        "frame_idx": int(frame_idx),
        "is_burst_boundary": bool(int(frame_idx) % 20 == 0),
        "sync_ok": bool(sync_ok),
        "numeric_ok": True,
        "latency_ms": float(latency_ms),
        "n_candidates": int(n_candidates),
        "ch_nmse": ch_nmse,
        "clean_pred_nmse": clean_pred_nmse,
        "probe_nmse": m_cal["probe_nmse"],
        "probe_nll": m_cal["probe_nll"],
        "probe_z2_mean": m_cal["probe_z2_mean"],
        "probe_cov90": m_cal["probe_cov90"],
        "probe_cov95": m_cal["probe_cov95"],
        "sync_ser": m_sync["ser"],
        "sync_ber": m_sync["ber"],
        "sync_evm_pct": m_sync["evm_pct"],
        "pilot_ser": m_pil["ser"],
        "pilot_ber": m_pil["ber"],
        "pilot_evm_pct": m_pil["evm_pct"],
        "probe_ser": m_prb["ser"],
        "probe_evm_pct": m_prb["evm_pct"],
        "payload_ser": m_pay["ser"],
        "payload_ber": m_pay["ber"],
        "payload_fer": m_pay["fer"],
        "payload_sym_rmse": m_pay["sym_rmse"],
        "payload_sym_nmse": m_pay["sym_nmse"],
        "payload_evm_pct": m_pay["evm_pct"],
    }


def aggregate_records(
    records: Sequence[Mapping[str, Any]],
    n_clusters: int = 16,
    peak_rss_mb: float = 0.0,
    train_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate per-frame records into summary statistics + cluster means for CI & paired tests."""
    n_tot = len(records)
    sync_ok_flags = np.array([bool(r["sync_ok"]) for r in records], dtype=np.bool_)
    num_ok_flags = np.array([bool(r["numeric_ok"]) for r in records], dtype=np.bool_)
    valid_recs = [r for r in records if r["numeric_ok"]]
    sync_valid_recs = [r for r in records if r["numeric_ok"] and r["sync_ok"]]

    def _mean_field(recs: Sequence[Mapping[str, Any]], key: str) -> float:
        if not recs:
            return float("nan")
        return float(np.mean([float(r[key]) for r in recs]))

    def _median_field(recs: Sequence[Mapping[str, Any]], key: str) -> float:
        if not recs:
            return float("nan")
        return float(np.median([float(r[key]) for r in recs]))

    # Burst boundary vs in-burst breakdown
    bb_recs = [r for r in valid_recs if r.get("is_burst_boundary", False)]
    in_recs = [r for r in valid_recs if not r.get("is_burst_boundary", False)]

    # Contiguous temporal clusters for cluster-level inference
    clusters: dict[str, list[float]] = {}
    cluster_keys = [
        "payload_ser", "payload_ber", "payload_fer", "payload_evm_pct", "payload_sym_nmse",
        "ch_nmse", "probe_nmse", "probe_nll", "probe_cov90", "probe_cov95",
        "sync_ser", "pilot_ser",
    ]
    if n_tot >= n_clusters and len(valid_recs) == n_tot:
        edges = np.linspace(0, n_tot, n_clusters + 1, dtype=int)
        for k in cluster_keys:
            clusters[k] = [
                float(np.mean([float(records[i][k]) for i in range(edges[g], edges[g + 1])]))
                for g in range(n_clusters)
            ]

    def _cluster_ci95(vals: Sequence[float]) -> dict[str, float]:
        arr = np.asarray(vals, dtype=np.float64)
        g = len(arr)
        if g < 2:
            m = float(arr[0]) if g == 1 else float("nan")
            return {"mean": m, "se": 0.0, "ci95_lo": m, "ci95_hi": m}
        m = float(np.mean(arr))
        se = float(np.std(arr, ddof=1) / math.sqrt(g))
        tcrit = float(student_t.ppf(0.975, df=g - 1))
        return {"mean": m, "se": se, "ci95_lo": m - tcrit * se, "ci95_hi": m + tcrit * se}

    summary: dict[str, Any] = {
        "n_frames_total": n_tot,
        "n_sync_ok": int(np.sum(sync_ok_flags)),
        "n_sync_fail": int(n_tot - np.sum(sync_ok_flags)),
        "sync_ok_rate": float(np.mean(sync_ok_flags)) if n_tot else 0.0,
        "n_numeric_ok": int(np.sum(num_ok_flags)),
        "n_numeric_fail": int(n_tot - np.sum(num_ok_flags)),
        "numeric_ok_rate": float(np.mean(num_ok_flags)) if n_tot else 0.0,
        "bit_definition": "Gray QPSK: b0=(Re<0), b1=(Im<0); 2 bits/symbol, 200 payload bits/frame",
        "fer_definition": "Frame error iff >=1 of 100 payload symbols is wrong; denominator = n_frames_total",
        "payload": {
            "ser": _mean_field(valid_recs, "payload_ser"),
            "ber": _mean_field(valid_recs, "payload_ber"),
            "fer": _mean_field(valid_recs, "payload_fer"),
            "fer_sync_ok_only": _mean_field(sync_valid_recs, "payload_fer"),
            "ser_sync_ok_only": _mean_field(sync_valid_recs, "payload_ser"),
            "evm_pct": _mean_field(valid_recs, "payload_evm_pct"),
            "evm_median_pct": _median_field(valid_recs, "payload_evm_pct"),
            "sym_rmse": _mean_field(valid_recs, "payload_sym_rmse"),
            "sym_nmse": _mean_field(valid_recs, "payload_sym_nmse"),
            "ser_ci95": _cluster_ci95(clusters["payload_ser"]) if "payload_ser" in clusters else None,
            "ber_ci95": _cluster_ci95(clusters["payload_ber"]) if "payload_ber" in clusters else None,
            "fer_ci95": _cluster_ci95(clusters["payload_fer"]) if "payload_fer" in clusters else None,
            "evm_ci95": _cluster_ci95(clusters["payload_evm_pct"]) if "payload_evm_pct" in clusters else None,
        },
        "sync_header": {
            "ser": _mean_field(valid_recs, "sync_ser"),
            "ber": _mean_field(valid_recs, "sync_ber"),
            "evm_pct": _mean_field(valid_recs, "sync_evm_pct"),
        },
        "pilot": {
            "ser": _mean_field(valid_recs, "pilot_ser"),
            "ber": _mean_field(valid_recs, "pilot_ber"),
            "evm_pct": _mean_field(valid_recs, "pilot_evm_pct"),
        },
        "channel_and_probe": {
            "ch_nmse": _mean_field(valid_recs, "ch_nmse"),
            "ch_nmse_db": float(10.0 * np.log10(max(_mean_field(valid_recs, "ch_nmse"), 1e-30))),
            "clean_pred_nmse": _mean_field(valid_recs, "clean_pred_nmse"),
            "probe_nmse": _mean_field(valid_recs, "probe_nmse"),
            "probe_nmse_db": float(10.0 * np.log10(max(_mean_field(valid_recs, "probe_nmse"), 1e-30))),
            "probe_nll": _mean_field(valid_recs, "probe_nll"),
            "probe_z2_mean": _mean_field(valid_recs, "probe_z2_mean"),
            "probe_cov90": _mean_field(valid_recs, "probe_cov90"),
            "probe_cov95": _mean_field(valid_recs, "probe_cov95"),
            "probe_nmse_ci95": _cluster_ci95(clusters["probe_nmse"]) if "probe_nmse" in clusters else None,
            "ch_nmse_ci95": _cluster_ci95(clusters["ch_nmse"]) if "ch_nmse" in clusters else None,
        },
        "regime_breakdown": {
            "burst_boundary_n": len(bb_recs),
            "burst_boundary_ser": _mean_field(bb_recs, "payload_ser"),
            "burst_boundary_evm_pct": _mean_field(bb_recs, "payload_evm_pct"),
            "burst_boundary_probe_nmse": _mean_field(bb_recs, "probe_nmse"),
            "burst_boundary_ch_nmse": _mean_field(bb_recs, "ch_nmse"),
            "in_burst_n": len(in_recs),
            "in_burst_ser": _mean_field(in_recs, "payload_ser"),
            "in_burst_evm_pct": _mean_field(in_recs, "payload_evm_pct"),
            "in_burst_probe_nmse": _mean_field(in_recs, "probe_nmse"),
            "in_burst_ch_nmse": _mean_field(in_recs, "ch_nmse"),
        },
        "engineering": {
            "latency_mean_ms": _mean_field(valid_recs, "latency_ms"),
            "latency_median_ms": _median_field(valid_recs, "latency_ms"),
            "peak_rss_mb": round(float(peak_rss_mb), 2),
            "n_candidates": int(valid_recs[0]["n_candidates"]) if valid_recs else 0,
            "train_meta": dict(train_meta) if train_meta is not None else None,
        },
        "cluster_series": clusters,
    }
    return summary


def paired_cluster_signflip(
    cand_clusters: Sequence[float],
    comp_clusters: Sequence[float],
    higher_is_better: bool = False,
) -> dict[str, Any]:
    """Exact 2^G paired cluster sign-flip permutation test (G<=16 -> exact enumeration up to 65,536 states).

    Returns diff = (comp - cand) when higher_is_better=False (so positive = candidate improves over comparator).
    """
    a = np.asarray(cand_clusters, dtype=np.float64)
    b = np.asarray(comp_clusters, dtype=np.float64)
    assert len(a) == len(b) and len(a) >= 2
    d = (a - b) if higher_is_better else (b - a)
    g = len(d)
    obs = float(np.mean(d))
    sd = float(np.std(d, ddof=1))
    se = sd / math.sqrt(float(g))
    t_crit = float(student_t.ppf(0.975, df=g - 1))
    tol = 1e-12

    if g <= 16:
        n_states = 1 << g
        bits = ((np.arange(n_states, dtype=np.int64)[:, None] >> np.arange(g, dtype=np.int64)[None, :]) & 1)
        signs = np.where(bits == 1, 1.0, -1.0)
        means = (signs @ d) / float(g)
        p_two = float(np.mean(np.abs(means) >= abs(obs) - tol))
        p_one = float(np.mean(means >= obs - tol))
        method = f"exact_2^{g}_signflip"
    else:
        rng = np.random.default_rng(20261121)
        signs = rng.choice([-1.0, 1.0], size=(65536, g))
        means = (signs @ d) / float(g)
        p_two = float((1 + np.sum(np.abs(means) >= abs(obs) - tol)) / (len(means) + 1))
        p_one = float((1 + np.sum(means >= obs - tol)) / (len(means) + 1))
        method = "mc_65536_signflip"

    return {
        "n_clusters": int(g),
        "improvement_mean": obs,
        "sd": sd,
        "se": se,
        "ci95_lo": obs - t_crit * se,
        "ci95_hi": obs + t_crit * se,
        "wins": int(np.sum(d > tol)),
        "ties": int(np.sum(np.abs(d) <= tol)),
        "losses": int(np.sum(d < -tol)),
        "p_two_sided": p_two,
        "p_one_sided_pos": p_one,
        "test_method": method,
    }


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: float(p_values[i]))
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, float(p_values[idx]) * float(m - rank))
        running = max(running, val)
        adj[idx] = running
    return adj
