#!/usr/bin/env python3
"""Run Stage 1 (Dev hyperparameter tuning & uncertainty calibration) and Stage 2 (Test closed-book
evaluation across all 8 conditions) for Classical Baselines, Group A (5 arms), and Group B (4 arms).

Saves:
  - dev_tuning_summary.json   (dev grid search log + frozen ReceiverHyperparams)
  - ab_test_summary.json      (all 8 conditions x 13 arms complete partitioned metrics & engineering costs)
  - ab_paired_stats.json      (G=16 exact 65,536-state sign-flip tests + Holm-Bonferroni corrections)
  - plot_data_ab.npz          (lightweight arrays for publication figures: channel trajectory,
                               constellations, attention weights, per-frame paired curves)
  - test_per_frame_summary.csv.gz (compact per-frame summary for key arms across conditions)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np

from evaluator import aggregate_records, holm_bonferroni, paired_cluster_signflip
from receivers import (
    ReceiverHyperparams,
    build_causal_history_library,
    compute_content_prior,
    estimate_single_frame_pilot_stats,
    estimate_train_channel_prior,
    run_all_classical_and_ab_arms,
)
from waveform_frontend import PAYLOAD_SLICE, PILOT_SLICE, STRESS_SPECS


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def load_bundle(bench_dir: Path, cond: str, split: str) -> dict[str, np.ndarray]:
    z = np.load(bench_dir / f"{cond}_{split}.npz")
    return {k: z[k] for k in z.files}


def tune_hyperparams_on_dev(
    bench_dir: Path,
    tx_mm: np.ndarray,
    h_mean_prior: complex,
    h_var_prior: float,
) -> tuple[ReceiverHyperparams, dict[str, Any]]:
    """Tune Group A & Group B hyperparameters strictly on DEV split (1,000 frames)."""
    tuning_conds = ["raw_original", "awgn_snr6db", "cfo_0p020_snr6db", "phase_jump_snr6db"]
    bundles = {c: load_bundle(bench_dir, c, "dev") for c in tuning_conds}
    # Precompute shared causal history library ONCE per dev condition (exact same library across grid!)
    libs = {
        c: build_causal_history_library(
            np.asarray(bundles[c]["y_sym"], dtype=np.complex128), h_mean_prior, h_var_prior, lib_size=200
        )
        for c in tuning_conds
    }

    # Step 1: Tune Group A2 neighbor decay & A5 content attention temperature/clear_bias/drift on dev
    grid_a = [
        {"attn_temp": t, "clear_prior_log_bias": cb, "attn_drift_rel_var": dv, "neighbor_decay": nd}
        for t in (0.75, 1.0, 1.5, 2.0)
        for cb in (-2.5, -1.0, 0.5)
        for dv in (0.002, 0.006)
        for nd in (0.45,)
    ]
    best_a_score = 1e30
    best_a_cfg: dict[str, float] = {}
    a_trials = []
    # Fast evaluation on A5 & A2 across raw_original + awgn_snr6db + phase_jump_snr6db
    for cfg in grid_a:
        hp = ReceiverHyperparams(
            h_mean_prior=h_mean_prior,
            h_var_prior=h_var_prior,
            neighbor_decay=cfg["neighbor_decay"],
            attn_temp=cfg["attn_temp"],
            clear_prior_log_bias=cfg["clear_prior_log_bias"],
            attn_drift_rel_var=cfg["attn_drift_rel_var"],
        )
        # Composite dev loss: mean(ch_nmse_raw + 0.5*ser_snr6 + 0.5*ser_jump + 0.2*probe_nmse_snr6)
        scores_c = {}
        for c in ("raw_original", "awgn_snr6db", "phase_jump_snr6db"):
            recs, _ = run_all_classical_and_ab_arms(bundles[c], tx_mm, hp, lib_size=200, cached_lib=libs[c])
            agg = aggregate_records(recs["A5_global_content_attn"], n_clusters=10)
            scores_c[c] = {
                "ch_nmse": agg["channel_and_probe"]["ch_nmse"],
                "probe_nmse": agg["channel_and_probe"]["probe_nmse"],
                "ser": agg["payload"]["ser"],
                "probe_nll": agg["channel_and_probe"]["probe_nll"],
            }
        comp = (
            scores_c["raw_original"]["ch_nmse"] * 10.0
            + scores_c["awgn_snr6db"]["ser"] * 5.0
            + scores_c["awgn_snr6db"]["ch_nmse"] * 2.0
            + scores_c["phase_jump_snr6db"]["ser"] * 5.0
            + scores_c["phase_jump_snr6db"]["ch_nmse"] * 2.0
        )
        a_trials.append({"cfg": cfg, "composite": comp, "details": scores_c})
        if comp < best_a_score:
            best_a_score = comp
            best_a_cfg = dict(cfg)

    # Step 2: Tune Group B temperature, CFO zero bonus, and variance calibration multiplier on dev
    grid_b = [
        {"b_temp": bt, "b_cfo_zero_bonus": zb}
        for bt in (0.75, 1.0, 1.5, 2.0)
        for zb in (0.0, 1.0, 2.5)
    ]
    best_b_score = 1e30
    best_b_cfg: dict[str, float] = {}
    b_trials = []
    for cfg in grid_b:
        hp = ReceiverHyperparams(
            h_mean_prior=h_mean_prior,
            h_var_prior=h_var_prior,
            neighbor_decay=best_a_cfg["neighbor_decay"],
            attn_temp=best_a_cfg["attn_temp"],
            clear_prior_log_bias=best_a_cfg["clear_prior_log_bias"],
            attn_drift_rel_var=best_a_cfg["attn_drift_rel_var"],
            b_temp=cfg["b_temp"],
            b_cfo_zero_bonus=cfg["b_cfo_zero_bonus"],
        )
        scores_b = {}
        for c in tuning_conds:
            recs, _ = run_all_classical_and_ab_arms(bundles[c], tx_mm, hp, lib_size=200, cached_lib=libs[c])
            agg = aggregate_records(recs["B3b_joint_mixture"], n_clusters=10)
            scores_b[c] = {
                "ser": agg["payload"]["ser"],
                "probe_nmse": agg["channel_and_probe"]["probe_nmse"],
                "probe_nll": agg["channel_and_probe"]["probe_nll"],
                "probe_z2": agg["channel_and_probe"]["probe_z2_mean"],
            }
        comp = (
            scores_b["raw_original"]["probe_nmse"] * 5.0
            + scores_b["awgn_snr6db"]["ser"] * 5.0
            + scores_b["cfo_0p020_snr6db"]["ser"] * 5.0
            + scores_b["phase_jump_snr6db"]["ser"] * 5.0
        )
        b_trials.append({"cfg": cfg, "composite": comp, "details": scores_b})
        if comp < best_b_score:
            best_b_score = comp
            best_b_cfg = dict(cfg)

    # Step 3: Exact closed-form dev variance calibration (set b_var_scale = median/mean z^2 across dev conditions!)
    hp_pre = ReceiverHyperparams(
        h_mean_prior=h_mean_prior,
        h_var_prior=h_var_prior,
        neighbor_decay=best_a_cfg["neighbor_decay"],
        attn_temp=best_a_cfg["attn_temp"],
        clear_prior_log_bias=best_a_cfg["clear_prior_log_bias"],
        attn_drift_rel_var=best_a_cfg["attn_drift_rel_var"],
        b_temp=best_b_cfg["b_temp"],
        b_cfo_zero_bonus=best_b_cfg["b_cfo_zero_bonus"],
        b_var_scale=1.0,
    )
    z2_vals = []
    for c in tuning_conds:
        recs, _ = run_all_classical_and_ab_arms(bundles[c], tx_mm, hp_pre, lib_size=200, cached_lib=libs[c])
        agg = aggregate_records(recs["B3b_joint_mixture"], n_clusters=10)
        z2_vals.append(agg["channel_and_probe"]["probe_z2_mean"])
    # Calibrate variance scale gently within [0.7, 2.5] so z^2 ~ 1.0 across regimes
    b_var_scale = float(np.clip(np.median(z2_vals), 0.7, 2.5))

    hp_frozen = ReceiverHyperparams(
        h_mean_prior=h_mean_prior,
        h_var_prior=h_var_prior,
        neighbor_decay=best_a_cfg["neighbor_decay"],
        attn_temp=best_a_cfg["attn_temp"],
        clear_prior_log_bias=best_a_cfg["clear_prior_log_bias"],
        attn_drift_rel_var=best_a_cfg["attn_drift_rel_var"],
        b_temp=best_b_cfg["b_temp"],
        b_cfo_zero_bonus=best_b_cfg["b_cfo_zero_bonus"],
        b_var_scale=b_var_scale,
    )

    # Also compute full DEV summary across all 8 conditions with frozen hyperparams
    dev_full = {}
    for cname in STRESS_SPECS.keys():
        b_dev = bundles[cname] if cname in bundles else load_bundle(bench_dir, cname, "dev")
        l_dev = libs[cname] if cname in libs else None
        recs, _ = run_all_classical_and_ab_arms(b_dev, tx_mm, hp_frozen, lib_size=200, cached_lib=l_dev)
        dev_full[cname] = {
            arm: {
                "payload_ser": aggregate_records(r, n_clusters=10)["payload"]["ser"],
                "payload_ber": aggregate_records(r, n_clusters=10)["payload"]["ber"],
                "payload_fer": aggregate_records(r, n_clusters=10)["payload"]["fer"],
                "payload_evm_pct": aggregate_records(r, n_clusters=10)["payload"]["evm_pct"],
                "ch_nmse": aggregate_records(r, n_clusters=10)["channel_and_probe"]["ch_nmse"],
                "probe_nmse": aggregate_records(r, n_clusters=10)["channel_and_probe"]["probe_nmse"],
                "probe_nll": aggregate_records(r, n_clusters=10)["channel_and_probe"]["probe_nll"],
                "probe_cov95": aggregate_records(r, n_clusters=10)["channel_and_probe"]["probe_cov95"],
            }
            for arm, r in recs.items()
        }

    tuning_log = {
        "train_channel_prior": {"re": h_mean_prior.real, "im": h_mean_prior.imag, "var": h_var_prior},
        "best_a_cfg": best_a_cfg,
        "best_a_score": best_a_score,
        "best_b_cfg": best_b_cfg,
        "best_b_score": best_b_score,
        "dev_pre_cal_z2_by_cond": dict(zip(tuning_conds, z2_vals)),
        "frozen_hyperparams": {
            "h_mean_prior_re": hp_frozen.h_mean_prior.real,
            "h_mean_prior_im": hp_frozen.h_mean_prior.imag,
            "h_var_prior": hp_frozen.h_var_prior,
            "neighbor_k": hp_frozen.neighbor_k,
            "neighbor_decay": hp_frozen.neighbor_decay,
            "neighbor_drift_rel_var": hp_frozen.neighbor_drift_rel_var,
            "global_drift_rel_var": hp_frozen.global_drift_rel_var,
            "attn_temp": hp_frozen.attn_temp,
            "attn_drift_rel_var": hp_frozen.attn_drift_rel_var,
            "clear_prior_log_bias": hp_frozen.clear_prior_log_bias,
            "age_penalty": hp_frozen.age_penalty,
            "b_temp": hp_frozen.b_temp,
            "b_cfo_zero_bonus": hp_frozen.b_cfo_zero_bonus,
            "b_quad_zero_bonus": hp_frozen.b_quad_zero_bonus,
            "b_var_scale": hp_frozen.b_var_scale,
        },
        "n_a_grid_trials": len(a_trials),
        "n_b_grid_trials": len(b_trials),
        "dev_all8_summary": dev_full,
    }
    return hp_frozen, tuning_log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    t_all = time.perf_counter()
    bench_dir = args.work_dir / "benchmarks"
    tx_mm = np.load(args.work_dir / "tx_c64.npy", mmap_mode="r")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Estimate empirical prior strictly from TRAIN raw_original pilots
    train_raw = load_bundle(bench_dir, "raw_original", "train")
    h_mean_pr, h_var_pr = estimate_train_channel_prior(
        np.asarray(train_raw["y_sym"][int(train_raw["eval_offset"]):], dtype=np.complex128)
    )
    print(f"Train empirical prior (pilots only): m0={h_mean_pr:.6f}, v0={h_var_pr:.3e}")

    # 2. Tune strictly on DEV
    t_dev0 = time.perf_counter()
    hp_frozen, tuning_log = tune_hyperparams_on_dev(bench_dir, tx_mm, h_mean_pr, h_var_pr)
    tuning_log["dev_tuning_elapsed_sec"] = round(time.perf_counter() - t_dev0, 2)
    (args.out_dir / "dev_tuning_summary.json").write_text(
        json.dumps(tuning_log, indent=2) + "\n", encoding="utf-8"
    )
    print(f"DEV tuning complete in {tuning_log['dev_tuning_elapsed_sec']}s. Frozen HP:", tuning_log["frozen_hyperparams"])

    # 3. Closed-book TEST evaluation across ALL 8 conditions (2,240 frames = G=16 clusters x 140 frames)
    test_summary: dict[str, Any] = {
        "split": "test",
        "eval_range": [4800, 7040],
        "n_frames_per_condition": 2240,
        "n_clusters": 16,
        "frames_per_cluster": 140,
        "frozen_hyperparams": tuning_log["frozen_hyperparams"],
        "conditions": {},
    }
    paired_tests: dict[str, Any] = {}
    plot_arrays: dict[str, Any] = {}

    csv_path = args.out_dir / "test_per_frame_summary.csv.gz"
    csv_arms = [
        "base_ls_1tap", "base_mmse_dc_cfo", "A1_local_single", "A2_neighbor_k5",
        "A3_global_uniform", "A4_global_reliability", "A5_global_content_attn",
        "B1_hard_map", "B2_posterior_mean_plugin", "B3a_sym_mixture", "B3b_joint_mixture",
    ]

    with gzip.open(csv_path, "wt", encoding="utf-8", newline="") as gz:
        writer = csv.writer(gz)
        writer.writerow([
            "condition", "arm", "frame_idx", "is_burst_boundary", "sync_ok",
            "payload_ser", "payload_ber", "payload_fer", "payload_evm_pct",
            "ch_nmse", "probe_nmse", "probe_nll", "latency_ms",
        ])

        for cname in STRESS_SPECS.keys():
            t_c0 = time.perf_counter()
            b_test = load_bundle(bench_dir, cname, "test")
            recs, lib = run_all_classical_and_ab_arms(b_test, tx_mm, hp_frozen, lib_size=200)
            rss_now = peak_rss_mb()

            cond_agg: dict[str, Any] = {}
            for arm, rlist in recs.items():
                cond_agg[arm] = aggregate_records(rlist, n_clusters=16, peak_rss_mb=rss_now)
                if arm in csv_arms:
                    for r in rlist:
                        writer.writerow([
                            cname, arm, r["frame_idx"], int(r["is_burst_boundary"]), int(r["sync_ok"]),
                            f"{r['payload_ser']:.6f}", f"{r['payload_ber']:.6f}", int(r["payload_fer"]),
                            f"{r['payload_evm_pct']:.4f}", f"{r['ch_nmse']:.6e}",
                            f"{r['probe_nmse']:.6e}", f"{r['probe_nll']:.4f}", f"{r['latency_ms']:.4f}",
                        ])
            test_summary["conditions"][cname] = cond_agg

            # Paired G=16 exact sign-flip tests (Group A & Group B planned comparisons)
            comparisons = [
                # Group A: Content addressing vs local / neighbor / uniform / reliability
                ("A5_vs_A1_ch_nmse", "A5_global_content_attn", "A1_local_single", "ch_nmse"),
                ("A5_vs_A2_ch_nmse", "A5_global_content_attn", "A2_neighbor_k5", "ch_nmse"),
                ("A5_vs_A3_ch_nmse", "A5_global_content_attn", "A3_global_uniform", "ch_nmse"),
                ("A5_vs_A4_ch_nmse", "A5_global_content_attn", "A4_global_reliability", "ch_nmse"),
                ("A5_vs_A1_ser", "A5_global_content_attn", "A1_local_single", "payload_ser"),
                ("A5_vs_A2_ser", "A5_global_content_attn", "A2_neighbor_k5", "payload_ser"),
                ("A5_vs_A3_ser", "A5_global_content_attn", "A3_global_uniform", "payload_ser"),
                ("A5_vs_A4_ser", "A5_global_content_attn", "A4_global_reliability", "payload_ser"),
                # Group B: Posterior fusion vs Hard MAP vs Mean Plug-in
                ("B2_vs_B1_probe_nmse", "B2_posterior_mean_plugin", "B1_hard_map", "probe_nmse"),
                ("B3b_vs_B1_ser", "B3b_joint_mixture", "B1_hard_map", "payload_ser"),
                ("B3b_vs_B2_ser", "B3b_joint_mixture", "B2_posterior_mean_plugin", "payload_ser"),
                ("B3b_vs_B3a_ser", "B3b_joint_mixture", "B3a_sym_mixture", "payload_ser"),
                ("B3b_vs_base_cfo_ser", "B3b_joint_mixture", "base_mmse_dc_cfo", "payload_ser"),
                ("B3b_vs_B1_fer", "B3b_joint_mixture", "B1_hard_map", "payload_fer"),
                ("B3b_vs_B2_fer", "B3b_joint_mixture", "B2_posterior_mean_plugin", "payload_fer"),
            ]
            c_stats = {}
            p_raws = []
            keys_ord = []
            for label, cand, comp, metric in comparisons:
                st_p = paired_cluster_signflip(
                    cond_agg[cand]["cluster_series"][metric],
                    cond_agg[comp]["cluster_series"][metric],
                    higher_is_better=False,
                )
                c_stats[label] = st_p
                p_raws.append(st_p["p_two_sided"])
                keys_ord.append(label)
            p_holm = holm_bonferroni(p_raws)
            for k_ord, ph in zip(keys_ord, p_holm):
                c_stats[k_ord]["p_holm"] = ph
            paired_tests[cname] = c_stats

            # Save key diagnostic arrays for figures (raw_original, awgn_snr6db, cfo_0p020_snr6db, phase_jump_snr6db)
            if cname in ("raw_original", "awgn_snr6db", "cfo_0p020_snr6db", "phase_jump_snr6db"):
                ev_off = int(b_test["eval_offset"])
                plot_arrays[f"{cname}_h_ora"] = b_test["h_ora"][ev_off:]
                plot_arrays[f"{cname}_frame_idx"] = b_test["frame_indices"][ev_off:]
                for arm in ("base_ls_1tap", "base_mmse_dc_cfo", "A1_local_single", "A2_neighbor_k5", "A5_global_content_attn", "B1_hard_map", "B2_posterior_mean_plugin", "B3b_joint_mixture"):
                    plot_arrays[f"{cname}_{arm}_ch_nmse"] = np.array([r["ch_nmse"] for r in recs[arm]], dtype=np.float32)
                    plot_arrays[f"{cname}_{arm}_ser"] = np.array([r["payload_ser"] for r in recs[arm]], dtype=np.float32)
                    plot_arrays[f"{cname}_{arm}_evm"] = np.array([r["payload_evm_pct"] for r in recs[arm]], dtype=np.float32)

                # Save attention weight heatmap snapshot for 60 contiguous frames across 3 burst boundaries in raw_original & phase_jump_snr6db!
                if cname in ("raw_original", "phase_jump_snr6db"):
                    y_all = np.asarray(b_test["y_sym"], dtype=np.complex128)
                    att_mat = np.zeros((60, 201), dtype=np.float32)
                    for row_i, t_pos in enumerate(range(ev_off, ev_off + 60)):
                        y_pil = y_all[t_pos, PILOT_SLICE]
                        st_p = estimate_single_frame_pilot_stats(y_pil, hp_frozen.h_mean_prior, hp_frozen.h_var_prior)
                        ca = compute_content_prior(
                            y_pil,
                            lib["h_dd"][t_pos - 200:t_pos],
                            lib["d_dd"][t_pos - 200:t_pos],
                            lib["var_h_dd"][t_pos - 200:t_pos],
                            st_p["sigma2_est"],
                            hp_frozen,
                        )
                        att_mat[row_i] = ca["weights"].astype(np.float32)
                    plot_arrays[f"{cname}_attn_weights_60x201"] = att_mat

            print(
                f"[TEST {cname:18s}] ({time.perf_counter()-t_c0:.1f}s) | "
                f"base_cfo SER={cond_agg['base_mmse_dc_cfo']['payload']['ser']*100:.2f}% | "
                f"A1={cond_agg['A1_local_single']['payload']['ser']*100:.2f}% "
                f"A2={cond_agg['A2_neighbor_k5']['payload']['ser']*100:.2f}% "
                f"A3={cond_agg['A3_global_uniform']['payload']['ser']*100:.2f}% "
                f"A5={cond_agg['A5_global_content_attn']['payload']['ser']*100:.2f}% | "
                f"B1={cond_agg['B1_hard_map']['payload']['ser']*100:.2f}% "
                f"B2={cond_agg['B2_posterior_mean_plugin']['payload']['ser']*100:.2f}% "
                f"B3a={cond_agg['B3a_sym_mixture']['payload']['ser']*100:.2f}% "
                f"B3b={cond_agg['B3b_joint_mixture']['payload']['ser']*100:.2f}% | "
                f"chNMSE A2={cond_agg['A2_neighbor_k5']['channel_and_probe']['ch_nmse']:.4f} "
                f"A5={cond_agg['A5_global_content_attn']['channel_and_probe']['ch_nmse']:.4f}"
            )

    test_summary["total_elapsed_sec"] = round(time.perf_counter() - t_all, 2)
    test_summary["peak_rss_mb"] = round(peak_rss_mb(), 2)
    (args.out_dir / "ab_test_summary.json").write_text(json.dumps(test_summary, indent=2) + "\n", encoding="utf-8")
    (args.out_dir / "ab_paired_stats.json").write_text(json.dumps(paired_tests, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.out_dir / "plot_data_ab.npz", **plot_arrays)
    print(f"ALL A/B TEST EXPERIMENTS DONE in {test_summary['total_elapsed_sec']}s, peak RSS {test_summary['peak_rss_mb']} MB")


if __name__ == "__main__":
    main()
