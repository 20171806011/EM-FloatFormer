"""Phase 12 Stage 1A: Primary-Axis Causal Diagnostics Runner (32 fresh full_reset dev links, 9 dB).

Executes four paired first-principles diagnostic modules strictly on SEED_P12_DIAG:
  1. Four-arm causal sentinel geometry intervention on locked ResetBase:
     T=(2,5,14,17), A=(1,4,7,13,16), arms={(8,9),(9,12),(11,12),(8,11)} with
     true_words_n0, true_words_vfloor, blind_decoded, and true_on_accepted.
  2. Full 16-core-target P11 chained error decomposition:
     P11 (11+5) vs forced_true_branch (reset) vs post-chain common-phase oracle
     vs P12 symmetric bilateral K=2 (8+8, identical 10 helper attempts) vs K=3 (6+5+5).
  3. Exact 5,832-node GL3 covariance posterior vs 160-node compression + continuous 6D
     (start_freq, end_freq) Top-5 mode refinement vs ground-truth initialization (8 links).
  4. Summary JSON + per-link JSONL persisted cleanly under `p12/results/`.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from p9.channel import simulate_link
from p9.contract import CORE_ROWS, PRIMARY_SNR_DB, PROBE_ROW
from p11.contract import enforce_determinism
from p11.wavefront_pipeline import AUX_FOLDS as P11_AUX_FOLDS, FOLD_TARGETS as P11_FOLD_TARGETS, _sanitize_y_b, warmup_p11_numba
from p12.contract import (
    P12_FOLD2_AUX,
    P12_FOLD2_TARGETS,
    P12_FOLD3_AUX,
    P12_FOLD3_TARGETS,
    SEED_P12_DIAG,
    SENTINEL_ARMS,
    SENTINEL_AUX,
    SENTINEL_POST_TARGETS,
    SENTINEL_PRE_TARGETS,
    SENTINEL_TARGETS,
    assert_p12_seed_disjoint,
)
from p12.primary_folds import (
    CORE_ROW_TO_INFO_IDX,
    SPARSE_ROW_TO_INFO_IDX,
    decode_rows_strong_l,
    evaluate_post_chain_common_phase_oracle,
    row_channel_metrics,
    run_gl3_exact_and_refinement_diag,
    run_k_fold_wavefront,
    run_sentinel_arms_on_link,
)

PRIMARY_DIAG_SEED_OFFSET = 100_000


def _worker_init(require_env: bool) -> None:
    enforce_determinism(require_env=require_env)
    warmup_p11_numba()


def _score_core_partition_result(
    res: dict[str, Any],
    truth: Any,
    y_b_safe: np.ndarray,
    n0: float,
    *,
    run_cp_oracle: bool = False,
) -> dict[str, Any]:
    """Score a K-fold wavefront run on all 16 CORE_ROWS with per-fold & pre/post breakdown."""
    per_row: dict[str, Any] = {}
    tot_err = 0
    pre_err = 0
    post_err = 0
    fold_errs = [0] * int(res["k_folds"])
    fold_blocks = [0] * int(res["k_folds"])

    for r in CORE_ROWS:
        idx_c = CORE_ROW_TO_INFO_IDX[int(r)]
        f_idx = int(res["owner_fold_idx"][int(r)])
        err = int(not np.array_equal(res["owner_decoded"][int(r)], truth.b_info[idx_c]))
        met = row_channel_metrics(res["owner_mean"][int(r)], truth.h_b_true[:, int(r), :])
        tot_err += err
        fold_errs[f_idx] += err
        fold_blocks[f_idx] += 1
        if int(r) < 10:
            pre_err += err
        else:
            post_err += err
        per_row[str(r)] = {
            "fold": f_idx,
            "block_error": err,
            "ray0_phase_err_deg": met["ray0_phase_err_deg"],
            "common_phase_err_deg": met["common_phase_err_deg"],
            "rel_frob_err": met["rel_frob_err"],
            "rel_frob_after_cp_oracle": met["rel_frob_after_cp_oracle"],
        }

    # Ledger false assimilation stats
    tot_acc = 0
    false_acc = 0
    branches: list[str] = []
    for f_obj in res["folds"]:
        branches.append(str(f_obj["branch"]))
        for rec in f_obj["ledger"]:
            tot_acc += 1
            r_inj = int(rec["row"])
            if not np.array_equal(rec["injected_info"], truth.b_info[SPARSE_ROW_TO_INFO_IDX[r_inj]]):
                false_acc += 1

    # Probe clean NMSE
    p_clean = truth.y_b_probe_clean
    p_err = res["probe_pred"] - p_clean
    probe_clean_nmse = float(np.sum(p_err.real ** 2 + p_err.imag ** 2) / max(float(np.sum(p_clean.real ** 2 + p_clean.imag ** 2)), 1e-30))

    out: dict[str, Any] = {
        "blocks": len(CORE_ROWS),
        "block_errors": int(tot_err),
        "pre_anchor_errors": int(pre_err),
        "post_anchor_errors": int(post_err),
        "fold_errors": fold_errs,
        "fold_blocks": fold_blocks,
        "branches": branches,
        "helpers_accepted_total": int(tot_acc),
        "false_assimilations": int(false_acc),
        "probe_clean_nmse": probe_clean_nmse,
        "elapsed_ms": float(res["elapsed_ms"]),
        "per_row": per_row,
    }

    if run_cp_oracle:
        dec_cp, _ = evaluate_post_chain_common_phase_oracle(
            res["owner_mean"], res["owner_var"], y_b_safe, truth.h_b_true, CORE_ROWS, n0
        )
        cp_tot = 0
        cp_fold = [0] * int(res["k_folds"])
        cp_pre = 0
        cp_post = 0
        for r in CORE_ROWS:
            idx_c = CORE_ROW_TO_INFO_IDX[int(r)]
            f_idx = int(res["owner_fold_idx"][int(r)])
            e_cp = int(not np.array_equal(dec_cp[int(r)], truth.b_info[idx_c]))
            cp_tot += e_cp
            cp_fold[f_idx] += e_cp
            if int(r) < 10:
                cp_pre += e_cp
            else:
                cp_post += e_cp
        out["cp_oracle_block_errors"] = int(cp_tot)
        out["cp_oracle_fold_errors"] = cp_fold
        out["cp_oracle_pre_errors"] = int(cp_pre)
        out["cp_oracle_post_errors"] = int(cp_post)

    return out


def _eval_one_primary_link(args: tuple[int, int, float, bool]) -> dict[str, Any]:
    link_idx, seed, snr_db, run_gl3_full = args
    t0 = time.perf_counter()
    obs, truth = simulate_link(seed, snr_db=snr_db, regime="full_reset", budget="sparse")
    y_b_safe = _sanitize_y_b(obs.y_b, PROBE_ROW)

    # 1) Four-arm causal sentinel geometry intervention
    sentinel = run_sentinel_arms_on_link(
        obs,
        truth.b_symbols,
        truth.b_info,
        truth.h_b_true,
        sentinel_targets=SENTINEL_TARGETS,
        sentinel_aux=SENTINEL_AUX,
        sentinel_arms=SENTINEL_ARMS,
    )

    # 2) Full 16-target comparison & decomposition
    res_p11 = run_k_fold_wavefront(obs, P11_FOLD_TARGETS, P11_AUX_FOLDS)
    sc_p11 = _score_core_partition_result(res_p11, truth, y_b_safe, obs.n0, run_cp_oracle=True)

    res_p11_forced = run_k_fold_wavefront(obs, P11_FOLD_TARGETS, P11_AUX_FOLDS, forced_branch="reset")
    sc_p11_forced = _score_core_partition_result(res_p11_forced, truth, y_b_safe, obs.n0, run_cp_oracle=True)

    res_p12_k2 = run_k_fold_wavefront(obs, P12_FOLD2_TARGETS, P12_FOLD2_AUX)
    sc_p12_k2 = _score_core_partition_result(res_p12_k2, truth, y_b_safe, obs.n0, run_cp_oracle=True)

    res_p12_k3 = run_k_fold_wavefront(obs, P12_FOLD3_TARGETS, P12_FOLD3_AUX)
    sc_p12_k3 = _score_core_partition_result(res_p12_k3, truth, y_b_safe, obs.n0, run_cp_oracle=False)

    # True CSI reference on all 16 CORE_ROWS
    v_zero = np.zeros((3, len(CORE_ROWS), 32), dtype=np.float64)
    dec_oracle, _ = decode_rows_strong_l(
        truth.h_b_true[:, list(CORE_ROWS), :], v_zero, y_b_safe[list(CORE_ROWS)], CORE_ROWS, obs.n0
    )
    oracle_core_errs = sum(
        int(not np.array_equal(dec_oracle[int(r)], truth.b_info[CORE_ROW_TO_INFO_IDX[int(r)]]))
        for r in CORE_ROWS
    )

    # 3) Optional full 5,832 GL3 exact & continuous refinement diagnostic (first 8 links)
    gl3_diag: Any = None
    if run_gl3_full:
        gl3_diag = run_gl3_exact_and_refinement_diag(
            obs,
            truth.channel_b,
            truth.b_symbols,
            truth.b_info,
            truth.h_b_true,
            aux_rows=SENTINEL_AUX,
            target_rows=SENTINEL_TARGETS,
            bilateral_helpers=(9, 12),
            top_k_refine=5,
        )

    return {
        "link_idx": int(link_idx),
        "seed": int(seed),
        "snr_db": float(snr_db),
        "regime": "full_reset",
        "sentinel": sentinel,
        "p11_asymmetric_2fold": sc_p11,
        "p11_forced_reset_2fold": sc_p11_forced,
        "p12_symmetric_2fold": sc_p12_k2,
        "p12_symmetric_3fold": sc_p12_k3,
        "oracle_core_errors": int(oracle_core_errs),
        "gl3_diag": gl3_diag,
        "wall_ms": float((time.perf_counter() - t0) * 1000.0),
    }


def aggregate_primary_diag(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n_links = len(records)

    def _agg_sentinel_block(extractor) -> dict[str, Any]:
        pre_e = pre_n = post_e = post_n = 0
        pre_ph: list[float] = []
        post_ph: list[float] = []
        pre_frob: list[float] = []
        post_frob: list[float] = []
        for rec in records:
            pr = extractor(rec)["per_row"]
            for r in SENTINEL_PRE_TARGETS:
                d = pr[str(r)]
                pre_e += int(d["block_error"])
                pre_n += 1
                if "ray0_phase_err_deg" in d:
                    pre_ph.append(float(d["ray0_phase_err_deg"]))
                    pre_frob.append(float(d["rel_frob_err"]))
            for r in SENTINEL_POST_TARGETS:
                d = pr[str(r)]
                post_e += int(d["block_error"])
                post_n += 1
                if "ray0_phase_err_deg" in d:
                    post_ph.append(float(d["ray0_phase_err_deg"]))
                    post_frob.append(float(d["rel_frob_err"]))
        tot_e = pre_e + post_e
        tot_n = pre_n + post_n
        out = {
            "total_bler_pct": 100.0 * tot_e / max(tot_n, 1),
            "pre_bler_pct": 100.0 * pre_e / max(pre_n, 1),
            "post_bler_pct": 100.0 * post_e / max(post_n, 1),
            "total_errors": int(tot_e),
            "total_blocks": int(tot_n),
        }
        if pre_ph:
            out["pre_mean_ray0_phase_deg"] = float(np.mean(pre_ph))
            out["post_mean_ray0_phase_deg"] = float(np.mean(post_ph))
            out["all_mean_ray0_phase_deg"] = float(np.mean(pre_ph + post_ph))
            out["pre_mean_rel_frob"] = float(np.mean(pre_frob))
            out["post_mean_rel_frob"] = float(np.mean(post_frob))
            out["all_mean_rel_frob"] = float(np.mean(pre_frob + post_frob))
        return out

    sentinel_summary: dict[str, Any] = {
        "payload_only": _agg_sentinel_block(lambda r: r["sentinel"]["payload_only"]),
        "true_csi": _agg_sentinel_block(lambda r: r["sentinel"]["true_csi"]),
        "arms": {},
    }
    for arm_name in SENTINEL_ARMS:
        arm_modes: dict[str, Any] = {}
        for mode in ("true_words_n0", "true_words_vfloor", "blind_decoded", "true_on_accepted"):
            arm_modes[mode] = _agg_sentinel_block(lambda r, a=arm_name, m=mode: r["sentinel"]["arms"][a][m])
        # Helper decode/acceptance diagnostics
        tot_h = acc_h = corr_h = false_acc_h = 0
        both_acc_links = 0
        for rec in records:
            hlist = rec["sentinel"]["arms"][arm_name]["helper_diag"]
            if all(h["accepted"] for h in hlist):
                both_acc_links += 1
            for h in hlist:
                tot_h += 1
                corr_h += int(h["decoded_correct"])
                acc_h += int(h["accepted"])
                false_acc_h += int(h["false_accepted"])
        arm_modes["helper_stats"] = {
            "total_helpers": tot_h,
            "decoded_correct_pct": 100.0 * corr_h / max(tot_h, 1),
            "accepted_pct": 100.0 * acc_h / max(tot_h, 1),
            "false_accepted_count": int(false_acc_h),
            "false_accepted_pct_of_accepted": 100.0 * false_acc_h / max(acc_h, 1),
            "both_helpers_accepted_links": int(both_acc_links),
        }
        sentinel_summary["arms"][arm_name] = arm_modes

    # Paired bilateral minus one-sided contrast on true_words_n0 and blind_decoded
    def _paired_arm_contrast(mode: str, arm_bilat: str, arm_one: str) -> dict[str, float]:
        diffs_bler = []
        diffs_phase = []
        for rec in records:
            pb = rec["sentinel"]["arms"][arm_bilat][mode]["per_row"]
            po = rec["sentinel"]["arms"][arm_one][mode]["per_row"]
            eb = sum(pb[str(r)]["block_error"] for r in SENTINEL_TARGETS) / len(SENTINEL_TARGETS)
            eo = sum(po[str(r)]["block_error"] for r in SENTINEL_TARGETS) / len(SENTINEL_TARGETS)
            phb = np.mean([pb[str(r)]["ray0_phase_err_deg"] for r in SENTINEL_TARGETS])
            pho = np.mean([po[str(r)]["ray0_phase_err_deg"] for r in SENTINEL_TARGETS])
            diffs_bler.append(100.0 * (eo - eb))  # positive = bilateral wins
            diffs_phase.append(float(pho - phb))
        arr_b = np.asarray(diffs_bler, dtype=np.float64)
        arr_p = np.asarray(diffs_phase, dtype=np.float64)
        return {
            "one_sided_minus_bilateral_bler_pp": float(np.mean(arr_b)),
            "bler_se_pp": float(np.std(arr_b, ddof=1) / math.sqrt(len(arr_b))) if len(arr_b) > 1 else 0.0,
            "one_sided_minus_bilateral_phase_deg": float(np.mean(arr_p)),
            "phase_se_deg": float(np.std(arr_p, ddof=1) / math.sqrt(len(arr_p))) if len(arr_p) > 1 else 0.0,
        }

    sentinel_summary["paired_contrasts"] = {
        "true_words_n0_left_vs_9_12": _paired_arm_contrast("true_words_n0", "bilateral_9_12", "one_sided_left"),
        "true_words_n0_right_vs_8_11": _paired_arm_contrast("true_words_n0", "bilateral_8_11", "one_sided_right"),
        "blind_decoded_left_vs_9_12": _paired_arm_contrast("blind_decoded", "bilateral_9_12", "one_sided_left"),
        "blind_decoded_right_vs_8_11": _paired_arm_contrast("blind_decoded", "bilateral_8_11", "one_sided_right"),
    }

    # Aggregate 16-target partition arms
    def _agg_partition(key: str) -> dict[str, Any]:
        tot_e = sum(r[key]["block_errors"] for r in records)
        tot_b = sum(r[key]["blocks"] for r in records)
        pre_e = sum(r[key]["pre_anchor_errors"] for r in records)
        post_e = sum(r[key]["post_anchor_errors"] for r in records)
        k_f = len(records[0][key]["fold_errors"])
        f_errs = [sum(r[key]["fold_errors"][i] for r in records) for i in range(k_f)]
        f_blks = [sum(r[key]["fold_blocks"][i] for r in records) for i in range(k_f)]
        acc_tot = sum(r[key]["helpers_accepted_total"] for r in records)
        false_tot = sum(r[key]["false_assimilations"] for r in records)
        branch_counts: dict[str, int] = {}
        for r in records:
            for b in r[key]["branches"]:
                branch_counts[b] = branch_counts.get(b, 0) + 1
        out: dict[str, Any] = {
            "core_bler_pct": 100.0 * tot_e / max(tot_b, 1),
            "core_errors": int(tot_e),
            "core_blocks": int(tot_b),
            "pre_anchor_bler_pct": 100.0 * pre_e / max(n_links * sum(1 for r in CORE_ROWS if r < 10), 1),
            "post_anchor_bler_pct": 100.0 * post_e / max(n_links * sum(1 for r in CORE_ROWS if r > 10), 1),
            "fold_bler_pct": [100.0 * fe / max(fb, 1) for fe, fb in zip(f_errs, f_blks)],
            "fold_errors": f_errs,
            "fold_blocks": f_blks,
            "branch_counts": branch_counts,
            "helpers_accepted_total": int(acc_tot),
            "false_assimilations": int(false_tot),
            "false_assimilation_pct": 100.0 * false_tot / max(acc_tot, 1),
            "mean_probe_clean_nmse": float(np.mean([r[key]["probe_clean_nmse"] for r in records])),
            "mean_elapsed_ms": float(np.mean([r[key]["elapsed_ms"] for r in records])),
        }
        if "cp_oracle_block_errors" in records[0][key]:
            cp_e = sum(r[key]["cp_oracle_block_errors"] for r in records)
            cp_f = [sum(r[key]["cp_oracle_fold_errors"][i] for r in records) for i in range(k_f)]
            out["cp_oracle_bler_pct"] = 100.0 * cp_e / max(tot_b, 1)
            out["cp_oracle_fold_bler_pct"] = [100.0 * fe / max(fb, 1) for fe, fb in zip(cp_f, f_blks)]
            out["cp_oracle_gain_pp"] = out["core_bler_pct"] - out["cp_oracle_bler_pct"]
        return out

    part_summary = {
        "p11_asymmetric_2fold": _agg_partition("p11_asymmetric_2fold"),
        "p11_forced_reset_2fold": _agg_partition("p11_forced_reset_2fold"),
        "p12_symmetric_2fold": _agg_partition("p12_symmetric_2fold"),
        "p12_symmetric_3fold": _agg_partition("p12_symmetric_3fold"),
        "oracle_core_bler_pct": 100.0 * sum(r["oracle_core_errors"] for r in records) / max(n_links * len(CORE_ROWS), 1),
    }

    # Paired 16-target contrasts (P11 -> P12 K=2, P12 K=2 -> P12 K=3)
    d_p11_k2 = np.array([
        100.0 * (r["p11_asymmetric_2fold"]["block_errors"] - r["p12_symmetric_2fold"]["block_errors"]) / 16.0
        for r in records
    ])
    d_k2_k3 = np.array([
        100.0 * (r["p12_symmetric_2fold"]["block_errors"] - r["p12_symmetric_3fold"]["block_errors"]) / 16.0
        for r in records
    ])
    part_summary["paired_contrasts"] = {
        "p11_minus_p12_k2_pp": float(np.mean(d_p11_k2)),
        "p11_minus_p12_k2_se_pp": float(np.std(d_p11_k2, ddof=1) / math.sqrt(len(d_p11_k2))) if len(d_p11_k2) > 1 else 0.0,
        "p12_k2_minus_k3_pp": float(np.mean(d_k2_k3)),
        "p12_k2_minus_k3_se_pp": float(np.std(d_k2_k3, ddof=1) / math.sqrt(len(d_k2_k3))) if len(d_k2_k3) > 1 else 0.0,
        "k3_over_k2_latency_ratio": float(
            part_summary["p12_symmetric_3fold"]["mean_elapsed_ms"]
            / max(part_summary["p12_symmetric_2fold"]["mean_elapsed_ms"], 1e-9)
        ),
    }

    # Aggregate 8-link full GL3 exact & continuous refinement
    gl3_recs = [r["gl3_diag"] for r in records if r["gl3_diag"] is not None]
    gl3_summary: dict[str, Any] = {"n_links": len(gl3_recs)}
    if gl3_recs:
        gl3_summary.update({
            "mean_omitted_exact_mass": float(np.mean([g["omitted_exact_mass"] for g in gl3_recs])),
            "max_omitted_exact_mass": float(np.max([g["omitted_exact_mass"] for g in gl3_recs])),
            "map_retained_in_160_rate": float(np.mean([float(g["map_retained_in_160"]) for g in gl3_recs])),
            "mean_abs_log_ev_diff_160_vs_5832": float(np.mean([abs(g["log_ev_diff_160_minus_5832"]) for g in gl3_recs])),
            "mean_p_true_sign_160": float(np.mean([g["p_true_sign_160"] for g in gl3_recs])),
            "mean_p_true_sign_5832": float(np.mean([g["p_true_sign_5832"] for g in gl3_recs])),
            "no_helper_bler_160_pct": 100.0 * sum(g["eval_160"]["no_helper"]["block_errors"] for g in gl3_recs) / sum(g["eval_160"]["no_helper"]["blocks"] for g in gl3_recs),
            "no_helper_bler_5832_pct": 100.0 * sum(g["eval_5832"]["no_helper"]["block_errors"] for g in gl3_recs) / sum(g["eval_5832"]["no_helper"]["blocks"] for g in gl3_recs),
            "bilat_helper_bler_160_pct": 100.0 * sum(g["eval_160"]["true_bilateral_helpers"]["block_errors"] for g in gl3_recs) / sum(g["eval_160"]["true_bilateral_helpers"]["blocks"] for g in gl3_recs),
            "bilat_helper_bler_5832_pct": 100.0 * sum(g["eval_5832"]["true_bilateral_helpers"]["block_errors"] for g in gl3_recs) / sum(g["eval_5832"]["true_bilateral_helpers"]["blocks"] for g in gl3_recs),
            "mean_compression_peak_gap_nats": float(np.mean([g["compression_peak_gap"] for g in gl3_recs])),
            "mean_grid_discretization_gap_nats": float(np.mean([g["grid_discretization_gap"] for g in gl3_recs])),
            "mean_basin_search_gap_nats": float(np.mean([g["basin_search_gap"] for g in gl3_recs])),
            "max_basin_search_gap_nats": float(np.max([g["basin_search_gap"] for g in gl3_recs])),
            "n_links_truth_beats_topk_by_1nat": int(sum(1 for g in gl3_recs if g["basin_search_gap"] > 1.0)),
            "truth_sign_matched_by_topk_rate": float(np.mean([float(g["truth_sign_matched_by_topk"]) for g in gl3_recs])),
            "mean_full_5832_score_ms": float(np.mean([g["full_5832_score_ms"] for g in gl3_recs])),
        })

    return {
        "n_links": n_links,
        "snr_db": float(records[0]["snr_db"]) if records else float(PRIMARY_SNR_DB),
        "sentinel": sentinel_summary,
        "partitions": part_summary,
        "gl3_exact_and_refinement": gl3_summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P12 Stage 1A Primary-Axis Causal Diagnostics")
    parser.add_argument("--n-links", type=int, default=32)
    parser.add_argument("--n-gl3-links", type=int, default=8)
    parser.add_argument("--snr-db", type=float, default=PRIMARY_SNR_DB)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="p12/results")
    parser.add_argument("--require-env", action="store_true", default=False)
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    enforce_determinism(require_env=args.require_env)

    seeds = assert_p12_seed_disjoint(
        [SEED_P12_DIAG + PRIMARY_DIAG_SEED_OFFSET + i for i in range(args.n_links)],
        expected_namespace="p12_diag",
    )
    tasks = [(i, seeds[i], float(args.snr_db), i < int(args.n_gl3_links)) for i in range(args.n_links)]

    if args.workers <= 1:
        warmup_p11_numba()
        records = [_eval_one_primary_link(t) for t in tasks]
    else:
        with ProcessPoolExecutor(
            max_workers=int(args.workers),
            initializer=_worker_init,
            initargs=(bool(args.require_env),),
        ) as pool:
            records = list(pool.map(_eval_one_primary_link, tasks))

    records.sort(key=lambda d: d["link_idx"])
    summary = aggregate_primary_diag(records)
    summary["total_wall_s"] = float(time.perf_counter() - t0)
    summary["workers"] = int(args.workers)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "stage1_primary_links.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for r in records:
            fh.write(json.dumps(r, allow_nan=False) + "\n")
    with open(out_dir / "stage1_primary_summary.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2, allow_nan=False)
        fh.write("\n")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
