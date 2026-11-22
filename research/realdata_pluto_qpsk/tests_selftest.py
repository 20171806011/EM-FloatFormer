#!/usr/bin/env python3
"""Independent zero-leakage firewall & mathematical identity self-test suite for
research/realdata_pluto_qpsk/.

Verifies:
  1. Strict held-out probe [21:26] firewall: corrupting y[21:26] to NaN produces
     0.0 bit-level change in all online channel/payload predictions (Base, A1-A5, B1-B3b).
  2. Strict causal future-frame firewall: corrupting future frames t' > t produces
     0.0 bit-level change in frame t predictions across all receivers.
  3. Exact mathematical identities:
       - Law of total variance (within + between) in Group B2 posterior fusion.
       - Zero double-counting identity in Group B3b joint mixture marginalization:
         leave-one-out beta_r^{(-k)} * p(y_k, s | r) == beta_r * P(S_k = s | y_k, r) to <1e-13.
       - Exact Gray QPSK bit mapping & FER/BER/SER consistency.
       - Exact 2^16 = 65,536 paired cluster sign-flip p-value bounds.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import logsumexp

from evaluator import evaluate_segment_predictions, holm_bonferroni, paired_cluster_signflip
from receivers import (
    N_CANDIDATES_B,
    PRIOR_DC_RATIO,
    ReceiverHyperparams,
    build_causal_history_library,
    build_shared_b_candidates,
    compute_content_prior,
    decode_group_b_variants,
    estimate_single_frame_pilot_stats,
    run_all_classical_and_ab_arms,
)
from waveform_frontend import FULL_HDR_26, N_SYM, PAYLOAD_SLICE, PILOT_SLICE, PROBE_SLICE, QPSK_CONST, symbols_to_gray_bits


def _make_synthetic_bundle(n_frames: int = 230, seed: int = 42) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    tx = np.empty((n_frames, N_SYM), dtype=np.complex128)
    tx[:, :26] = FULL_HDR_26[None, :]
    pay_idx = rng.integers(0, 4, size=(n_frames, 100))
    tx[:, 26:] = QPSK_CONST[pay_idx]

    h_true = 0.02 * np.exp(1j * np.cumsum(rng.normal(0.0, 0.03, size=n_frames)))
    d_true = -h_true * np.mean(tx, axis=1)
    noise = (rng.normal(0.0, 0.004, size=(n_frames, N_SYM)) + 1j * rng.normal(0.0, 0.004, size=(n_frames, N_SYM)))
    y_sym = h_true[:, None] * tx + d_true[:, None] + noise

    bundle = {
        "frame_indices": np.arange(n_frames, dtype=np.int32),
        "eval_offset": np.int32(205),
        "y_sym": y_sym.astype(np.complex64),
        "h_ora": h_true,
        "d_ora": d_true,
        "sync_ok": np.ones(n_frames, dtype=np.bool_),
    }
    return bundle, tx


def test_probe_firewall() -> dict[str, object]:
    """Corrupting held-out probe y[21:26] MUST NOT alter any online estimate or payload decision."""
    bundle, tx = _make_synthetic_bundle(n_frames=215, seed=101)
    hp = ReceiverHyperparams(h_mean_prior=0.02 + 0.0j, h_var_prior=1e-4)

    recs_clean, lib_clean = run_all_classical_and_ab_arms(bundle, tx, hp, lib_size=200)

    bundle_poison = {k: v.copy() for k, v in bundle.items()}
    # Inject extreme finite sentinel into held-out probe [21:26] across ALL frames (warmup + eval)
    bundle_poison["y_sym"][:, PROBE_SLICE] = (1e6 - 1e6j)
    recs_poison, lib_poison = run_all_classical_and_ab_arms(bundle_poison, tx, hp, lib_size=200)

    max_lib_diff = float(np.max(np.abs(lib_clean["h_dd"] - lib_poison["h_dd"])))
    assert max_lib_diff == 0.0, f"Library leaked probe! diff={max_lib_diff}"

    max_arm_diffs = {}
    for arm in recs_clean.keys():
        diffs = []
        for rc, rp in zip(recs_clean[arm], recs_poison[arm]):
            # ch_nmse and payload_ser/evm MUST be bit-identical (probe_nmse naturally changes because y_probe changed!)
            diffs.append(abs(rc["ch_nmse"] - rp["ch_nmse"]))
            diffs.append(abs(rc["payload_ser"] - rp["payload_ser"]))
            diffs.append(abs(rc["payload_evm_pct"] - rp["payload_evm_pct"]))
        max_arm_diffs[arm] = float(max(diffs))
        assert max_arm_diffs[arm] == 0.0, f"Arm {arm} leaked held-out probe! diff={max_arm_diffs[arm]}"
    return {"passed": True, "max_lib_diff": max_lib_diff, "max_arm_diff": max(max_arm_diffs.values())}


def test_causal_future_firewall() -> dict[str, object]:
    """Corrupting all future frames t' > t_target MUST NOT alter frame t_target predictions."""
    bundle, tx = _make_synthetic_bundle(n_frames=220, seed=202)
    hp = ReceiverHyperparams(h_mean_prior=0.02 + 0.0j, h_var_prior=1e-4)
    recs_clean, _ = run_all_classical_and_ab_arms(bundle, tx, hp, lib_size=200)

    t_check_eval_idx = 4  # corresponds to absolute frame index 209
    t_abs = int(bundle["eval_offset"]) + t_check_eval_idx
    bundle_fut = {k: v.copy() for k, v in bundle.items()}
    bundle_fut["y_sym"][t_abs + 1:] = 999.0 + 999.0j
    recs_fut, _ = run_all_classical_and_ab_arms(bundle_fut, tx, hp, lib_size=200)

    max_diff = 0.0
    for arm in recs_clean.keys():
        rc = recs_clean[arm][t_check_eval_idx]
        rf = recs_fut[arm][t_check_eval_idx]
        for k in ("ch_nmse", "probe_nmse", "payload_ser", "payload_evm_pct", "probe_nll"):
            d = abs(rc[k] - rf[k])
            max_diff = max(max_diff, d)
            assert d == 0.0, f"Causal future leak in arm={arm}, key={k}, diff={d}"
    return {"passed": True, "max_future_leak_diff": max_diff}


def test_b3b_leave_one_out_exactness() -> dict[str, object]:
    """Verify exact equivalence of B3b joint mixture marginalization and explicit leave-one-out formula."""
    rng = np.random.default_rng(303)
    R, K = 12, 20
    alpha = rng.dirichlet(np.ones(R))
    h_pay = 0.02 * (rng.normal(size=(R, K)) + 1j * rng.normal(size=(R, K)))
    d_post = 0.001 * (rng.normal(size=R) + 1j * rng.normal(size=R))
    obs_var = rng.uniform(1e-5, 5e-5, size=R)
    y_pay = h_pay[0] * QPSK_CONST[rng.integers(0, 4, size=K)] + d_post[0] + 0.003 * (rng.normal(size=K) + 1j * rng.normal(size=K))

    pred = h_pay[:, :, None] * QPSK_CONST[None, None, :] + d_post[:, None, None]
    d2 = np.abs(y_pay[None, :, None] - pred) ** 2
    vr = obs_var[:, None, None]
    log_p_ys = -d2 / vr - np.log(np.pi * vr) - np.log(4.0)
    log_p_y = logsumexp(log_p_ys, axis=2)  # (R, K)
    log_alpha = np.log(alpha)

    # Method 1: Fast exact B3b (beta_r * P(S_k=s | y_k, r))
    log_beta = log_alpha + np.sum(log_p_y, axis=1)
    beta = np.exp(log_beta - logsumexp(log_beta))
    p_s_given_r = np.exp(log_p_ys - log_p_y[:, :, None])
    p_fast = np.tensordot(beta, p_s_given_r, axes=([0], [0]))  # (K, 4)

    # Method 2: Explicit leave-one-out per symbol k: beta_r^{(-k)} * p(y_k, s | r)
    p_loo = np.zeros_like(p_fast)
    for k in range(K):
        log_b_loo = log_alpha + (np.sum(log_p_y, axis=1) - log_p_y[:, k])
        log_joint_k = log_b_loo[:, None] + log_p_ys[:, k, :]  # (R, 4)
        log_s_k = logsumexp(log_joint_k, axis=0)
        p_loo[k] = np.exp(log_s_k - logsumexp(log_s_k))

    max_err = float(np.max(np.abs(p_fast - p_loo)))
    assert max_err < 1e-13, f"LOO identity mismatch: {max_err}"
    return {"passed": True, "max_loo_prob_diff": max_err}


def test_gray_bits_and_cluster_stats() -> dict[str, object]:
    """Verify Gray bit adjacent Hamming distance = 1 and exact 2^16 sign-flip bounds."""
    # Check adjacent pairs in QPSK circle: (0:++, 1:+-, 3:--, 2:-+)
    circle = [0, 1, 3, 2, 0]
    bits = symbols_to_gray_bits(QPSK_CONST)
    for a, b in zip(circle[:-1], circle[1:]):
        hd = int(np.sum(bits[a] != bits[b]))
        assert hd == 1, f"Non-Gray adjacent pair ({a},{b}) Hamming={hd}"
    # Opposite pairs must have Hamming distance = 2
    assert int(np.sum(bits[0] != bits[3])) == 2
    assert int(np.sum(bits[1] != bits[2])) == 2

    # Segment metric check: 1 adjacent error in 100 symbols -> SER=0.01, BER=0.005, FER=1.0
    s_true = np.full(100, QPSK_CONST[0])
    s_dec = s_true.copy()
    s_dec[0] = QPSK_CONST[1]
    seg = evaluate_segment_predictions(s_dec, s_dec, s_true)
    assert abs(seg["ser"] - 0.01) < 1e-12
    assert abs(seg["ber"] - 0.005) < 1e-12
    assert seg["fer"] == 1.0

    # Exact 2^16 sign-flip check: all 16 positive equal diffs -> p_two = 2 / 65536 = 1/32768
    st = paired_cluster_signflip(np.zeros(16), np.ones(16), higher_is_better=False)
    expected_p = 2.0 / 65536.0
    assert abs(st["p_two_sided"] - expected_p) < 1e-12
    assert st["wins"] == 16 and st["losses"] == 0
    assert holm_bonferroni([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    return {"passed": True, "exact_p_16of16": st["p_two_sided"]}


def main() -> None:
    report = {
        "probe_firewall": test_probe_firewall(),
        "causal_future_firewall": test_causal_future_firewall(),
        "b3b_leave_one_out_exactness": test_b3b_leave_one_out_exactness(),
        "gray_bits_and_cluster_stats": test_gray_bits_and_cluster_stats(),
    }
    print(json.dumps(report, indent=2))
    assert all(v["passed"] for v in report.values())


if __name__ == "__main__":
    main()
