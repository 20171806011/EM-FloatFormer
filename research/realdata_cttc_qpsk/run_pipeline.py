"""End-to-end execution driver for CTTC Indoor Multi-Band/Multi-SNR QPSK empirical study.

Supports:
  1. `--mode smoke`: 32-packet end-to-end verification on high-SNR (f2450_snr20_QPSK) +
     f2450_snr0_QPSK + f708_snr10_QPSK, including raw `file_1.dat` (re-filtered RRC) vs
     official `file_2.dat` cross-check and full A/B/C dry run.
  2. `--mode full`: Complete 9-capture (3 bands x 3 SNR), strictly zero-leakage train/dev/test
     empirical study with dev-only hyperparameter tuning, paired cluster-level statistical
     inference (bootstrap 95% CI + exact/MC sign-flip p-values), controlled physical stress
     perturbation suite, figures, and compact JSON/CSV deliverables.
"""
from __future__ import annotations

import os

# Enforce single-threaded BLAS/OpenMP/PyTorch before numerical imports
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

import argparse
import csv
import json
import math
import platform
try:
    import resource
except ImportError:
    resource = None  # type: ignore[assignment]
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .adapter import (
    ALL_CAPTURES,
    CaptureAnchor,
    PacketSample,
    anchor_capture,
    apply_controlled_stress,
    extract_packets_for_capture,
    load_source_qpsk,
    save_anchor_index,
    verify_zero_leakage,
)
from .baselines import (
    ClassicalConfig,
    ReceiverOutput,
    evaluate_receiver_output,
    run_classical_receiver,
)
from .evaluator import (
    PacketEvalRecord,
    aggregate_records,
    paired_cluster_inference,
)
from .experiments import (
    CandidateBank,
    RefEntry,
    build_common_candidate_bank,
    build_cross_session_pool_for_target,
    build_intra_session_pool_for_target,
    extract_reference_entry,
    run_experiment_a_arms,
    run_experiment_b_arms,
    train_and_infer_experiment_c,
)


def _peak_rss_mb() -> float:
    if resource is None:
        return float("nan")
    # On Linux, ru_maxrss is in kilobytes
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for r in rows[1:]:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _eval_batch(
    pkts: Sequence[PacketSample],
    outs: Sequence[ReceiverOutput],
    method_name: str,
) -> tuple[list[PacketEvalRecord], dict[str, Any]]:
    recs: list[PacketEvalRecord] = []
    probs_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    for p, o in zip(pkts, outs):
        rec, pr, lb = evaluate_receiver_output(p, o, method_name)
        recs.append(rec)
        if len(pr) > 0:
            probs_list.append(pr)
            labels_list.append(lb)
    p_all = np.concatenate(probs_list) if probs_list else None
    l_all = np.concatenate(labels_list) if labels_list else None
    summ = aggregate_records(recs, bit_probs_all=p_all, bit_labels_all=l_all)
    # Also compute stratified breakdown by (capture_id) and (nom_snr_db) and (freq_mhz)
    by_cap: dict[str, Any] = {}
    for cid in sorted({r.capture_id for r in recs}):
        sub = [r for r in recs if r.capture_id == cid]
        by_cap[cid] = aggregate_records(sub)
    by_snr: dict[str, Any] = {}
    for snr in sorted({r.nom_snr_db for r in recs}):
        sub = [r for r in recs if r.nom_snr_db == snr]
        by_snr[f"snr{snr}"] = aggregate_records(sub)
    by_freq: dict[str, Any] = {}
    for fr in sorted({r.freq_mhz for r in recs}):
        sub = [r for r in recs if r.freq_mhz == fr]
        by_freq[f"f{fr}"] = aggregate_records(sub)
    summ["by_capture"] = by_cap
    summ["by_snr"] = by_snr
    summ["by_freq"] = by_freq
    return recs, summ


def run_smoke_check(work_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Run fast 32-packet end-to-end check on f2450_snr20/snr10/snr0 & f708_snr10 + file_1 vs file_2."""
    print("=== SMOKE TEST: 32-packet end-to-end verification ===")
    t0 = time.perf_counter()
    _, sym_even, sym_odd = load_source_qpsk(work_dir / "sources")
    data_dir = work_dir / "data"

    smoke_caps = [c for c in ALL_CAPTURES if c[0] in ("f2450_snr20_QPSK", "f2450_snr0_QPSK", "f708_snr10_QPSK", "f433_snr20_QPSK")]
    anchors: list[CaptureAnchor] = []
    ds_all: dict[str, dict[str, list[PacketSample]]] = {sp: {"target": [], "ref": []} for sp in ("train", "dev", "test")}

    per_loop = {"train": (2, 2), "dev": (2, 2), "test": (4, 2)}  # 16 train, 16 dev, 32 test per capture
    for cap_id, f_mhz, snr_db, cfo_h in smoke_caps:
        anc = anchor_capture(data_dir, cap_id, f_mhz, snr_db, cfo_h, sym_even, sym_odd)
        anchors.append(anc)
        print(
            f"[anchor] {cap_id:17s} align={anc.bit_alignment} corr={anc.lock_norm_corr_mean:.4f} "
            f"SNR={anc.measured_snr_db:+6.2f}dB SFO={anc.sfo_ppm:+.4f}ppm CFO={anc.offline_cfo_med_hz:+.1f}Hz"
        )
        sub = extract_packets_for_capture(data_dir, anc, sym_even, sym_odd, per_loop_counts=per_loop, n_loops_use=8)
        for sp in ("train", "dev", "test"):
            for role in ("target", "ref"):
                ds_all[sp][role].extend(sub[sp][role])

    leak = verify_zero_leakage(ds_all)
    print(f"[leakage] {leak}")

    # Cross-check raw file_1.dat (self-filtered 256-tap RRC) vs official file_2.dat on f2450_snr20
    anc_20 = [a for a in anchors if a.capture_id == "f2450_snr20_QPSK"][0]
    sub_f1 = extract_packets_for_capture(
        data_dir, anc_20, sym_even, sym_odd, per_loop_counts={"train": (0, 0), "dev": (0, 0), "test": (4, 0)},
        use_raw_file1_rrc=True, n_loops_use=8,
    )
    cfg = ClassicalConfig()
    pkts_20_f2 = [p for p in ds_all["test"]["target"] if p.capture_id == "f2450_snr20_QPSK"]
    pkts_20_f1 = sub_f1["test"]["target"]
    outs_f2 = [run_classical_receiver(p, cfg, mode="mmse") for p in pkts_20_f2]
    outs_f1 = [run_classical_receiver(p, cfg, mode="mmse") for p in pkts_20_f1]
    _, s_f2 = _eval_batch(pkts_20_f2, outs_f2, "classical_mmse_file2")
    _, s_f1 = _eval_batch(pkts_20_f1, outs_f1, "classical_mmse_file1_rrc")
    print(
        f"[file1 vs file2 @ f2450_snr20, N={len(pkts_20_f2)}] "
        f"file2 SER={s_f2['payload_ser']:.6f} EVM={s_f2['payload_evm_unbiased_db']:.2f}dB | "
        f"file1+RRC SER={s_f1['payload_ser']:.6f} EVM={s_f1['payload_evm_unbiased_db']:.2f}dB"
    )
    assert s_f2["payload_ser"] < 0.002, f"High-SNR sanity failed on file_2: SER={s_f2['payload_ser']}"
    assert s_f1["payload_ser"] < 0.005, f"High-SNR sanity failed on file_1+RRC: SER={s_f1['payload_ser']}"

    # Run classical + Exp A + Exp B + Exp C dry-run on test packets
    test_pkts = ds_all["test"]["target"]
    ref_entries_by_cap: dict[str, list[RefEntry]] = {}
    for rp in ds_all["test"]["ref"]:
        ref_entries_by_cap.setdefault(rp.capture_id, []).append(extract_reference_entry(rp, cfg))

    rng = np.random.default_rng(20261121)
    a_outs: dict[str, list[ReceiverOutput]] = {}
    b_outs: dict[str, list[ReceiverOutput]] = {}
    cls_outs = [run_classical_receiver(p, cfg, mode="mmse") for p in test_pkts]
    banks_test: list[CandidateBank] = []
    for p in test_pkts:
        pool = build_cross_session_pool_for_target(p, ref_entries_by_cap, per_capture=4, rng=rng)
        ao, _ = run_experiment_a_arms(p, pool, cfg, local_k=4, content_temp=1.5)
        for k, v in ao.items():
            a_outs.setdefault(k, []).append(v)
        bk = build_common_candidate_bank(p, pool, cfg, content_temp=1.5)
        banks_test.append(bk)
        bo = run_experiment_b_arms(bk, post_temp=1.5)
        for k, v in bo.items():
            b_outs.setdefault(k, []).append(v)

    # Dry run Exp C with 1 seed, 8 epochs
    refs_tr: dict[str, list[RefEntry]] = {}
    for rp in ds_all["train"]["ref"]:
        refs_tr.setdefault(rp.capture_id, []).append(extract_reference_entry(rp, cfg))
    refs_dv: dict[str, list[RefEntry]] = {}
    for rp in ds_all["dev"]["ref"]:
        refs_dv.setdefault(rp.capture_id, []).append(extract_reference_entry(rp, cfg))
    banks_tr = [
        build_common_candidate_bank(p, build_cross_session_pool_for_target(p, refs_tr, per_capture=4, rng=rng), cfg)
        for p in ds_all["train"]["target"]
    ]
    banks_dv = [
        build_common_candidate_bank(p, build_cross_session_pool_for_target(p, refs_dv, per_capture=4, rng=rng), cfg)
        for p in ds_all["dev"]["target"]
    ]
    res_c = train_and_infer_experiment_c(
        ds_all["train"]["target"], banks_tr, ds_all["dev"]["target"], banks_dv, test_pkts, banks_test,
        seeds=(20261121,), epochs=8, batch_size=16,
    )

    smoke_summary: dict[str, Any] = {
        "mode": "smoke",
        "elapsed_sec": time.perf_counter() - t0,
        "peak_rss_mb": _peak_rss_mb(),
        "leakage": leak,
        "file2_snr20": {k: s_f2[k] for k in ("payload_ser", "payload_ber", "payload_evm_unbiased_db", "holdout_ref_nmse")},
        "file1_rrc_snr20": {k: s_f1[k] for k in ("payload_ser", "payload_ber", "payload_evm_unbiased_db", "holdout_ref_nmse")},
        "methods": {},
    }
    all_eval_sets: list[tuple[str, list[ReceiverOutput]]] = [("classical_mmse", cls_outs)]
    all_eval_sets.extend(a_outs.items())
    all_eval_sets.extend(b_outs.items())
    for vname, vobj in res_c.items():
        all_eval_sets.append((vname, vobj["test_outputs_by_seed"][20261121]))
    for mname, outs in all_eval_sets:
        _, sm = _eval_batch(test_pkts, outs, mname)
        smoke_summary["methods"][mname] = {
            "payload_ser": sm["payload_ser"],
            "payload_ber": sm["payload_ber"],
            "payload_per": sm["payload_per"],
            "payload_evm_unbiased_db": sm["payload_evm_unbiased_db"],
            "payload_bit_nll_nats": sm["payload_bit_nll_nats"],
            "payload_bit_ece": sm.get("payload_bit_ece"),
            "holdout_obs_nmse": sm["holdout_obs_nmse"],
            "holdout_ref_nmse": sm["holdout_ref_nmse"],
            "holdout_ch_nll_nats": sm["holdout_ch_nll_nats"],
            "elapsed_ms_median": sm["elapsed_ms_median"],
        }
        print(
            f"[smoke] {mname:26s} | SER={sm['payload_ser']:.5f} BER={sm['payload_ber']:.5f} "
            f"PER={sm['payload_per']:.4f} NLL={sm['payload_bit_nll_nats']:.4f} ECE={sm.get('payload_bit_ece', 0):.4f} "
            f"hRefNMSE={sm['holdout_ref_nmse']:.5f}"
        )
    _write_json(out_dir / "smoke_summary.json", smoke_summary)
    return smoke_summary


def tune_hyperparams_on_dev(
    dev_pkts: Sequence[PacketSample],
    dev_refs_by_cap: dict[str, list[RefEntry]],
) -> dict[str, Any]:
    """Tune classical noise blend, variance calibration, content temp, and posterior temp STRICTLY on dev."""
    print("=== DEV-ONLY HYPERPARAMETER CALIBRATION ===")
    # 1) Classical noise blend & slope_reg on dev subset (every 2nd dev packet = 432 packets for fast grid)
    sub_dev = list(dev_pkts[::2])
    best_cls_nll = float("inf")
    best_cls_cfg = ClassicalConfig()
    cls_grid_log = []
    for nb in (0.15, 0.35, 0.55):
        for vcs in (0.85, 1.0, 1.25):
            cfg_t = ClassicalConfig(noise_blend=nb, var_calib_scale=vcs)
            outs = [run_classical_receiver(p, cfg_t, mode="mmse") for p in sub_dev]
            _, sm = _eval_batch(sub_dev, outs, "dev_cls")
            score = sm["payload_bit_nll_nats"] + 0.1 * sm["holdout_ch_nll_nats"]
            cls_grid_log.append({"noise_blend": nb, "var_calib_scale": vcs, "bit_nll": sm["payload_bit_nll_nats"], "ser": sm["payload_ser"], "ch_nll": sm["holdout_ch_nll_nats"]})
            if score < best_cls_nll:
                best_cls_nll = score
                best_cls_cfg = cfg_t
    print(f"[dev-tune] Best classical cfg: noise_blend={best_cls_cfg.noise_blend}, var_calib_scale={best_cls_cfg.var_calib_scale}")

    # 2) Content addressing temperature tau_A on dev
    rng_a = np.random.default_rng(2026112101)
    dev_pools = [build_cross_session_pool_for_target(p, dev_refs_by_cap, per_capture=4, rng=rng_a) for p in sub_dev]
    best_tau_a = 1.5
    best_a_ser = float("inf")
    a_grid_log = []
    for tau_a in (0.20, 0.45, 0.80, 1.50, 2.50):
        outs_a4 = []
        for p, pl in zip(sub_dev, dev_pools):
            ao, _ = run_experiment_a_arms(p, pl, best_cls_cfg, local_k=4, content_temp=tau_a)
            outs_a4.append(ao["A4_global_content_attn"])
        _, sm = _eval_batch(sub_dev, outs_a4, f"dev_a_{tau_a}")
        a_grid_log.append({"tau_a": tau_a, "ser": sm["payload_ser"], "bit_nll": sm["payload_bit_nll_nats"], "holdout_ref_nmse": sm["holdout_ref_nmse"]})
        if sm["payload_bit_nll_nats"] < best_a_ser:
            best_a_ser = sm["payload_bit_nll_nats"]
            best_tau_a = tau_a
    print(f"[dev-tune] Best Exp A content_temp={best_tau_a}")

    # 3) Posterior fusion temperature tau_B & block_temp on dev (using B3 mixture likelihood bit NLL)
    banks_sub = [build_common_candidate_bank(p, pl, best_cls_cfg, content_temp=best_tau_a) for p, pl in zip(sub_dev, dev_pools)]
    best_tau_b = 1.5
    best_blk_t = 0.30
    best_b_nll = float("inf")
    b_grid_log = []
    for tau_b in (0.40, 0.80, 1.50, 2.50):
        for blk_t in (0.15, 0.30, 0.50):
            outs_b3 = [run_experiment_b_arms(bk, post_temp=tau_b, block_temp=blk_t)["B3_mixture_likelihood"] for bk in banks_sub]
            _, sm = _eval_batch(sub_dev, outs_b3, f"dev_b_{tau_b}_{blk_t}")
            b_grid_log.append({
                "tau_b": tau_b,
                "block_temp": blk_t,
                "ser": sm["payload_ser"],
                "bit_nll": sm["payload_bit_nll_nats"],
                "ece": sm.get("payload_bit_ece"),
            })
            if sm["payload_bit_nll_nats"] < best_b_nll:
                best_b_nll = sm["payload_bit_nll_nats"]
                best_tau_b = tau_b
                best_blk_t = blk_t
    print(f"[dev-tune] Best Exp B post_temp={best_tau_b}, block_temp={best_blk_t}")

    return {
        "classical_cfg": best_cls_cfg,
        "content_temp": best_tau_a,
        "post_temp": best_tau_b,
        "block_temp": best_blk_t,
        "cls_grid_log": cls_grid_log,
        "a_grid_log": a_grid_log,
        "b_grid_log": b_grid_log,
    }


def _make_figures(summary: dict[str, Any], a_audit: list[dict[str, Any]], fig_dir: Path) -> list[str]:
    """Generate publication-quality summary figures for REPORT.md."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    plt.rcParams.update({
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 160,
    })

    # Fig 1: Per-capture SNR / SER / EVM overview across all 9 real CTTC captures
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    caps = [a["capture_id"] for a in summary["anchors"]]
    short_caps = [c.replace("_QPSK", "") for c in caps]
    snrs_meas = [a["measured_snr_db"] for a in summary["anchors"]]
    methods_fig1 = [
        ("classical_ls", "Classical LS", "#7f8c8d"),
        ("classical_mmse", "Classical MMSE", "#2980b9"),
        ("A4_global_content_attn", "Exp A: Global Content", "#27ae60"),
        ("B3_mixture_likelihood", "Exp B: Mixture Posterior", "#8e44ad"),
        ("C2_attn_weight_resid_net", "Exp C: Learned Attn+Resid", "#e67e22"),
    ]
    x = np.arange(len(caps))
    w = 0.16
    for i, (mkey, mlab, col) in enumerate(methods_fig1):
        if mkey not in summary["test_original"]["methods"]:
            continue
        by_cap = summary["test_original"]["methods"][mkey]["by_capture"]
        sers = [max(by_cap[c]["payload_ser"], 1e-5) for c in caps]
        axes[0].bar(x + (i - 2) * w, sers, width=w, label=mlab, color=col, alpha=0.9)
    axes[0].set_yscale("log")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_caps, rotation=35, ha="right", fontsize=8.5)
    axes[0].set_ylabel("Payload SER (log scale, floor 1e-5)")
    axes[0].set_title("Test Payload SER across All 9 CTTC Real Captures")
    axes[0].legend(fontsize=8, loc="upper right")

    for i, (mkey, mlab, col) in enumerate(methods_fig1):
        if mkey not in summary["test_original"]["methods"]:
            continue
        by_cap = summary["test_original"]["methods"][mkey]["by_capture"]
        nlls = [by_cap[c]["payload_bit_nll_nats"] for c in caps]
        axes[1].bar(x + (i - 2) * w, nlls, width=w, label=mlab, color=col, alpha=0.9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_caps, rotation=35, ha="right", fontsize=8.5)
    axes[1].set_ylabel("Payload Bit NLL (nats/bit)")
    axes[1].set_title("Test Soft-Decision Calibration (Bit NLL) by Capture")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    p1 = fig_dir / "fig1_9captures_ser_nll.png"
    fig.savefig(p1)
    plt.close(fig)
    saved.append(str(p1.name))

    # Fig 2: Experiment A — Scope vs Content Matching Decomposition (SER, Holdout NMSE, CFO Prior Error)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7))
    a_keys = [
        ("classical_mmse", "Local Single"),
        ("A1_local_window_k4", "Local Win k=4"),
        ("A2_global_uniform", "Global Uniform"),
        ("A3_global_snr_weight", "Global SNR Wt"),
        ("A4_global_content_attn", "Global Content"),
        ("A5_content_cold_hard", "Content Cold"),
        ("A6_content_hot", "Content Hot"),
    ]
    cols_a = ["#95a5a6", "#7f8c8d", "#e74c3c", "#f39c12", "#27ae60", "#2ecc71", "#d35400"]
    m_orig = summary["test_original"]["methods"]
    # Plot SNR=0 (hardest real regime) + All-9 aggregate
    ser_all = [m_orig[k]["payload_ser"] for k, _ in a_keys]
    ser_s0 = [m_orig[k]["by_snr"]["snr0"]["payload_ser"] for k, _ in a_keys]
    nmse_all = [m_orig[k]["holdout_ref_nmse"] for k, _ in a_keys]
    xa = np.arange(len(a_keys))
    axes[0].bar(xa - 0.18, ser_all, width=0.35, label="All 9 Captures", color="#34495e")
    axes[0].bar(xa + 0.18, ser_s0, width=0.35, label="Low-SNR (0 dB nom)", color="#c0392b")
    axes[0].set_xticks(xa)
    axes[0].set_xticklabels([lb for _, lb in a_keys], fontsize=8.5, rotation=28, ha="right")
    axes[0].set_ylabel("Payload SER")
    axes[0].set_title("Exp A: Reference Scope vs Content Matching (SER)")
    axes[0].legend(fontsize=8)

    axes[1].bar(xa, nmse_all, color=cols_a, alpha=0.9)
    axes[1].set_xticks(xa)
    axes[1].set_xticklabels([lb for _, lb in a_keys], fontsize=8.5, rotation=28, ha="right")
    axes[1].set_ylabel("Holdout Reference Channel NMSE")
    axes[1].set_title("Exp A: Holdout Channel NMSE")

    # Right panel: Same-capture reference hit weight & CFO prior RMSE across A1..A6
    a_sub_keys = [k for k, _ in a_keys[1:]]
    hit_means = [float(np.mean([r[f"{k}__hit_cap"] for r in a_audit])) for k in a_sub_keys]
    cfo_rmses = [math.sqrt(float(np.mean([r[f"{k}__cfo_prior_err2"] for r in a_audit]))) for k in a_sub_keys]
    ax2 = axes[2]
    ax2_r = ax2.twinx()
    xb = np.arange(len(a_sub_keys))
    b1 = ax2.bar(xb - 0.18, hit_means, width=0.35, color="#2980b9", label="Matched-Session Wt")
    b2 = ax2_r.bar(xb + 0.18, cfo_rmses, width=0.35, color="#e67e22", label="CFO Prior RMSE (Hz)")
    ax2.set_xticks(xb)
    ax2.set_xticklabels([lb for _, lb in a_keys[1:]], fontsize=8.5, rotation=28, ha="right")
    ax2.set_ylabel("Matched-Session Weight Mass", color="#2980b9")
    ax2_r.set_ylabel("CFO Prior RMSE (Hz)", color="#e67e22")
    ax2.set_title("Exp A: Content Addressing Accuracy")
    ax2.legend([b1, b2], ["Matched-Session Wt", "CFO Prior RMSE (Hz)"], fontsize=7.8, loc="upper right")
    fig.tight_layout()
    p2 = fig_dir / "fig2_expA_content_addressing.png"
    fig.savefig(p2)
    plt.close(fig)
    saved.append(str(p2.name))

    # Fig 3: Experiment B & C — Hard vs Linear Fusion vs Mixture Posterior vs Trainable Attn (Original + Stress)
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7))
    bc_keys = [
        ("B1_hard_select", "B1: Hard Select"),
        ("B2_linear_fusion_plugin", "B2: Linear Plug-in"),
        ("B2b_linear_within_only", "B2b: Within-Only"),
        ("B3_mixture_likelihood", "B3: Mixture Post"),
        ("B4_avg_llr_antipattern", "B4: Avg-LLR"),
        ("C1_attn_weight_net", "C1: Attn Weight"),
        ("C2_attn_weight_resid_net", "C2: Attn + Resid"),
    ]
    m_stress = summary["test_stress"]["methods"]
    xb = np.arange(len(bc_keys))
    axes[0].bar(xb - 0.18, [m_orig[k]["payload_ser"] for k, _ in bc_keys], width=0.35, label="Original Real (9 caps)", color="#2c3e50")
    axes[0].bar(xb + 0.18, [m_stress[k]["payload_ser"] for k, _ in bc_keys], width=0.35, label="Stress Perturbed", color="#e74c3c")
    axes[0].set_xticks(xb)
    axes[0].set_xticklabels([lb for _, lb in bc_keys], fontsize=8.5, rotation=28, ha="right")
    axes[0].set_ylabel("Payload SER")
    axes[0].set_title("Exp B & C: Payload SER (Original vs Stress)")
    axes[0].legend(fontsize=8)

    axes[1].bar(xb - 0.18, [m_orig[k]["payload_bit_nll_nats"] for k, _ in bc_keys], width=0.35, label="Original Real", color="#16a085")
    axes[1].bar(xb + 0.18, [m_stress[k]["payload_bit_nll_nats"] for k, _ in bc_keys], width=0.35, label="Stress Perturbed", color="#d35400")
    axes[1].set_xticks(xb)
    axes[1].set_xticklabels([lb for _, lb in bc_keys], fontsize=8.5, rotation=28, ha="right")
    axes[1].set_ylabel("Payload Bit NLL (nats/bit)")
    axes[1].set_title("Exp B & C: Soft-Decision Bit NLL")
    axes[1].legend(fontsize=8)

    axes[2].bar(xb - 0.18, [m_orig[k].get("payload_bit_ece", 0.0) for k, _ in bc_keys], width=0.35, label="Original Real", color="#8e44ad")
    axes[2].bar(xb + 0.18, [m_stress[k].get("payload_bit_ece", 0.0) for k, _ in bc_keys], width=0.35, label="Stress Perturbed", color="#c0392b")
    axes[2].set_xticks(xb)
    axes[2].set_xticklabels([lb for _, lb in bc_keys], fontsize=8.5, rotation=28, ha="right")
    axes[2].set_ylabel("Bit ECE (15 bins)")
    axes[2].set_title("Exp B & C: Calibration Error (ECE)")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    p3 = fig_dir / "fig3_expBC_posterior_fusion_and_learned.png"
    fig.savefig(p3)
    plt.close(fig)
    saved.append(str(p3.name))

    # Fig 4: Breakdown by Stress Sub-Regime (low_snr, cfo_phase_jump, burst_impairment)
    if "by_stress_mode" in summary["test_stress"]:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
        modes = ["stress_low_snr", "stress_cfo_phase_jump", "stress_burst_impairment"]
        titles = ["Stress 1: Low-SNR AWGN", "Stress 2: CFO + Mid-Pkt Phase Jump", "Stress 3: Burst + CFO + Noise"]
        show_m = [
            ("classical_mmse", "Cls MMSE", "#7f8c8d"),
            ("B1_hard_select", "B1 Hard", "#e74c3c"),
            ("B2_linear_fusion_plugin", "B2 LinPlug", "#f39c12"),
            ("B3_mixture_likelihood", "B3 Mixture", "#8e44ad"),
            ("C2_attn_weight_resid_net", "C2 Learned", "#27ae60"),
        ]
        for ax, md, ttl in zip(axes, modes, titles):
            sub_m = summary["test_stress"]["by_stress_mode"][md]
            xs = np.arange(len(show_m))
            vals = [sub_m[mk]["payload_ser"] for mk, _, _ in show_m]
            ax.bar(xs, vals, color=[c for _, _, c in show_m], alpha=0.9)
            ax.set_xticks(xs)
            ax.set_xticklabels([lb for _, lb, _ in show_m], fontsize=8, rotation=15)
            ax.set_ylabel("Payload SER")
            ax.set_title(ttl)
        fig.tight_layout()
        p4 = fig_dir / "fig4_stress_breakdown.png"
        fig.savefig(p4)
        plt.close(fig)
        saved.append(str(p4.name))

    return saved


def run_full_study(work_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Run complete 9-capture zero-leakage empirical study + stress suite + figures + CSV/JSON exports."""
    t_start = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    _, sym_even, sym_odd = load_source_qpsk(work_dir / "sources")
    data_dir = work_dir / "data"

    # 1) Anchor all 9 captures & extract strictly disjoint train/dev/test target & ref packets
    print("=== STEP 1: ANCHOR ALL 9 CTTC CAPTURES & EXTRACT ZERO-LEAKAGE SPLITS ===")
    anchors: list[CaptureAnchor] = []
    ds_all: dict[str, dict[str, list[PacketSample]]] = {sp: {"target": [], "ref": []} for sp in ("train", "dev", "test")}
    per_loop_counts = {"train": (20, 8), "dev": (12, 8), "test": (20, 8)}  # 1440 / 864 / 1440 targets

    for cap_id, f_mhz, snr_db, cfo_h in ALL_CAPTURES:
        anc = anchor_capture(data_dir, cap_id, f_mhz, snr_db, cfo_h, sym_even, sym_odd)
        anchors.append(anc)
        print(
            f"[anchor] {cap_id:17s} align={anc.bit_alignment} corr={anc.lock_norm_corr_mean:.4f} "
            f"SNR={anc.measured_snr_db:+6.2f}dB SFO={anc.sfo_ppm:+.4f}ppm CFO={anc.offline_cfo_med_hz:+.1f}Hz "
            f"spur={anc.secondary_spur_ratio:.3f}@{anc.secondary_spur_offset_hz:+.0f}Hz"
        )
        sub = extract_packets_for_capture(
            data_dir, anc, sym_even, sym_odd, per_loop_counts=per_loop_counts, n_loops_use=8
        )
        for sp in ("train", "dev", "test"):
            for role in ("target", "ref"):
                ds_all[sp][role].extend(sub[sp][role])

    save_anchor_index(anchors, out_dir / "capture_anchors.json")
    leak_info = verify_zero_leakage(ds_all)
    print(f"[leakage-verified] {leak_info}")

    # Also verify raw file_1.dat + 256-tap RRC vs official file_2.dat on 64 packets of f2450_snr20
    anc_2450_20 = [a for a in anchors if a.capture_id == "f2450_snr20_QPSK"][0]
    sub_f1 = extract_packets_for_capture(
        data_dir, anc_2450_20, sym_even, sym_odd,
        per_loop_counts={"train": (0, 0), "dev": (0, 0), "test": (8, 0)},
        use_raw_file1_rrc=True, n_loops_use=8,
    )
    cfg_default = ClassicalConfig()
    pkts_20_f1 = sub_f1["test"]["target"]
    pkts_20_f2 = [p for p in ds_all["test"]["target"] if p.capture_id == "f2450_snr20_QPSK"][: len(pkts_20_f1)]
    _, sm_f1 = _eval_batch(pkts_20_f1, [run_classical_receiver(p, cfg_default, mode="mmse") for p in pkts_20_f1], "f1_rrc")
    _, sm_f2 = _eval_batch(pkts_20_f2, [run_classical_receiver(p, cfg_default, mode="mmse") for p in pkts_20_f2], "f2_off")
    print(
        f"[file1 vs file2 check, N={len(pkts_20_f1)}] "
        f"file1+RRC SER={sm_f1['payload_ser']:.6f} EVM={sm_f1['payload_evm_unbiased_db']:.2f}dB | "
        f"file2 SER={sm_f2['payload_ser']:.6f} EVM={sm_f2['payload_evm_unbiased_db']:.2f}dB"
    )

    # 2) Extract pilot-only reference pools for train / dev / test (using default cfg for unbiased pool)
    print("=== STEP 2: BUILD PILOT-ONLY REFERENCE POOLS ===")
    refs_by_split: dict[str, dict[str, list[RefEntry]]] = {}
    for sp in ("train", "dev", "test"):
        mp: dict[str, list[RefEntry]] = {}
        for rp in ds_all[sp]["ref"]:
            mp.setdefault(rp.capture_id, []).append(extract_reference_entry(rp, cfg_default))
        refs_by_split[sp] = mp
        print(f"  [{sp}] extracted {sum(len(v) for v in mp.values())} pilot-only reference entries across {len(mp)} captures")

    # 3) Dev-only hyperparameter tuning
    tune_res = tune_hyperparams_on_dev(ds_all["dev"]["target"], refs_by_split["dev"])
    best_cfg: ClassicalConfig = tune_res["classical_cfg"]
    tau_a: float = tune_res["content_temp"]
    tau_b: float = tune_res["post_temp"]
    blk_t: float = tune_res["block_temp"]

    # 4) Build fixed candidate banks for train, dev, and test (using fixed deterministic RNGs per split)
    print("=== STEP 3: EVALUATE CLASSICAL BASELINES + EXP A + EXP B ON FIXED TEST SET ===")
    rng_tr = np.random.default_rng(2026112110)
    rng_dv = np.random.default_rng(2026112120)
    rng_te = np.random.default_rng(2026112130)

    train_pkts = ds_all["train"]["target"]
    dev_pkts = ds_all["dev"]["target"]
    test_pkts = ds_all["test"]["target"]

    train_pools = [build_cross_session_pool_for_target(p, refs_by_split["train"], per_capture=4, rng=rng_tr) for p in train_pkts]
    dev_pools = [build_cross_session_pool_for_target(p, refs_by_split["dev"], per_capture=4, rng=rng_dv) for p in dev_pkts]
    test_pools = [build_cross_session_pool_for_target(p, refs_by_split["test"], per_capture=4, rng=rng_te) for p in test_pkts]

    # Classical baselines on test
    test_outputs: dict[str, list[ReceiverOutput]] = {}
    for cmode in ("ls", "mmse", "mmse_slope", "mmse_3tap"):
        test_outputs[f"classical_{cmode}"] = [run_classical_receiver(p, best_cfg, mode=cmode) for p in test_pkts]
        print(f"  [test] finished classical_{cmode}")

    # Experiment A (Cross-session M=36 global pool) on test
    a_audit_rows: list[dict[str, Any]] = []
    for p, pl in zip(test_pkts, test_pools):
        ao, aud = run_experiment_a_arms(p, pl, best_cfg, local_k=4, content_temp=tau_a)
        for k, v in ao.items():
            test_outputs.setdefault(k, []).append(v)
        a_audit_rows.append(aud)
    print("  [test] finished Experiment A (cross-session M=36 pool)")

    # Experiment A-Intra (Intra-session temporal pool M=24: nearest k=4 vs global M=24) on test
    a_intra_outputs: dict[str, list[ReceiverOutput]] = {}
    for p in test_pkts:
        pl_in = build_intra_session_pool_for_target(p, refs_by_split["test"][p.capture_id], pool_size=24)
        ao_in, _ = run_experiment_a_arms(p, pl_in, best_cfg, local_k=4, content_temp=tau_a)
        for k in ("A1_local_window_k4", "A2_global_uniform", "A4_global_content_attn"):
            a_intra_outputs.setdefault(f"Intra_{k}", []).append(ao_in[k])
    print("  [test] finished Experiment A-Intra (same-session temporal M=24 pool)")

    # Build shared fixed CandidateBanks for train/dev/test
    banks_tr = [build_common_candidate_bank(p, pl, best_cfg, content_temp=tau_a) for p, pl in zip(train_pkts, train_pools)]
    banks_dv = [build_common_candidate_bank(p, pl, best_cfg, content_temp=tau_a) for p, pl in zip(dev_pkts, dev_pools)]
    banks_te = [build_common_candidate_bank(p, pl, best_cfg, content_temp=tau_a) for p, pl in zip(test_pkts, test_pools)]

    # Experiment B on test
    for bk in banks_te:
        bo = run_experiment_b_arms(bk, post_temp=tau_b, block_temp=blk_t)
        for k, v in bo.items():
            test_outputs.setdefault(k, []).append(v)
    print("  [test] finished Experiment B (L=12 shared candidate bank)")

    # 5) Experiment C: Train lightweight PyTorch Attention / Weight + Residual Net (3 seeds, single-thread)
    print("=== STEP 4: TRAIN & EVALUATE EXPERIMENT C (3 SEEDS, TORCH_NUM_THREADS=1) ===")
    res_c_orig = train_and_infer_experiment_c(
        train_pkts, banks_tr, dev_pkts, banks_dv, test_pkts, banks_te,
        init_temp=tau_b, block_temp=blk_t, seeds=(20261121, 20261122, 20261123), epochs=45, batch_size=64, lr=4e-3,
    )
    c_multi_seed_summary: dict[str, Any] = {}
    for vname, vobj in res_c_orig.items():
        sel_seed = vobj["dev_selected_seed"]
        test_outputs[vname] = vobj["test_outputs_by_seed"][sel_seed]
        seed_metrics = []
        for srun in vobj["seed_runs"]:
            sd = srun["seed"]
            _, sm_sd = _eval_batch(test_pkts, vobj["test_outputs_by_seed"][sd], f"{vname}_s{sd}")
            seed_metrics.append({
                **srun,
                "test_ser": sm_sd["payload_ser"],
                "test_ber": sm_sd["payload_ber"],
                "test_per": sm_sd["payload_per"],
                "test_bit_nll": sm_sd["payload_bit_nll_nats"],
                "test_bit_ece": sm_sd.get("payload_bit_ece"),
                "test_holdout_ref_nmse": sm_sd["holdout_ref_nmse"],
                "test_elapsed_ms_median": sm_sd["elapsed_ms_median"],
            })
        c_multi_seed_summary[vname] = {
            "dev_selected_seed": sel_seed,
            "seeds": seed_metrics,
            "test_ser_mean": float(np.mean([m["test_ser"] for m in seed_metrics])),
            "test_ser_std": float(np.std([m["test_ser"] for m in seed_metrics], ddof=1)),
            "test_bit_nll_mean": float(np.mean([m["test_bit_nll"] for m in seed_metrics])),
            "test_bit_nll_std": float(np.std([m["test_bit_nll"] for m in seed_metrics], ddof=1)),
        }
        print(
            f"  [Exp C] {vname}: dev_sel_seed={sel_seed} | "
            f"test_SER={c_multi_seed_summary[vname]['test_ser_mean']:.5f}+-{c_multi_seed_summary[vname]['test_ser_std']:.5f} | "
            f"test_NLL={c_multi_seed_summary[vname]['test_bit_nll_mean']:.4f}+-{c_multi_seed_summary[vname]['test_bit_nll_std']:.4f}"
        )

    # Evaluate all original-test methods & save per-packet CSVs
    orig_recs_by_method: dict[str, list[PacketEvalRecord]] = {}
    orig_summ_by_method: dict[str, Any] = {}
    all_orig_methods = {**test_outputs, **a_intra_outputs}
    for mname, outs in all_orig_methods.items():
        recs, sm = _eval_batch(test_pkts, outs, mname)
        orig_recs_by_method[mname] = recs
        orig_summ_by_method[mname] = sm
        print(
            f"[ORIG-TEST] {mname:26s} | SER={sm['payload_ser']:.5f} BER={sm['payload_ber']:.5f} "
            f"PER={sm['payload_per']:.4f} EVM={sm['payload_evm_unbiased_db']:+6.2f}dB "
            f"NLL={sm['payload_bit_nll_nats']:.4f} ECE={sm.get('payload_bit_ece', 0):.4f} "
            f"hRefNMSE={sm['holdout_ref_nmse']:.5f} t_med={sm['elapsed_ms_median']:.2f}ms"
        )

    # Paired cluster-level statistical comparisons (G = 72 clusters & G = 9 captures) on Original Test
    paired_orig: dict[str, Any] = {}
    comparisons_orig = [
        ("A4_vs_classical_mmse", "classical_mmse", "A4_global_content_attn"),
        ("A4_vs_A1_local_k4", "A1_local_window_k4", "A4_global_content_attn"),
        ("A4_vs_A2_global_unif", "A2_global_uniform", "A4_global_content_attn"),
        ("A4_vs_A3_global_snr", "A3_global_snr_weight", "A4_global_content_attn"),
        ("A4_vs_A5_cold_hard", "A5_content_cold_hard", "A4_global_content_attn"),
        ("A2_unif_vs_A1_local_k4", "A1_local_window_k4", "A2_global_uniform"),
        ("B2_lin_vs_B1_hard", "B1_hard_select", "B2_linear_fusion_plugin"),
        ("B3_mix_vs_B1_hard", "B1_hard_select", "B3_mixture_likelihood"),
        ("B3_mix_vs_B2_lin", "B2_linear_fusion_plugin", "B3_mixture_likelihood"),
        ("B3_mix_vs_B3a_sym", "B3a_sym_mixture", "B3_mixture_likelihood"),
        ("B3_mix_vs_B4_avgllr", "B4_avg_llr_antipattern", "B3_mixture_likelihood"),
        ("B2_tot_vs_B2b_within", "B2b_linear_within_only", "B2_linear_fusion_plugin"),
        ("B3_mix_vs_classical_mmse", "classical_mmse", "B3_mixture_likelihood"),
        ("C1_vs_B3_mix", "B3_mixture_likelihood", "C1_attn_weight_net"),
        ("C2_vs_B3_mix", "B3_mixture_likelihood", "C2_attn_weight_resid_net"),
        ("C2_vs_classical_mmse", "classical_mmse", "C2_attn_weight_resid_net"),
    ]
    for label, ma, mb in comparisons_orig:
        paired_orig[label] = {
            "payload_ser_cluster72": paired_cluster_inference(orig_recs_by_method[ma], orig_recs_by_method[mb], "payload_ser", group_by="cluster_id"),
            "payload_ber_cluster72": paired_cluster_inference(orig_recs_by_method[ma], orig_recs_by_method[mb], "payload_ber", group_by="cluster_id"),
            "payload_bit_nll_cluster72": paired_cluster_inference(orig_recs_by_method[ma], orig_recs_by_method[mb], "payload_bit_nll", group_by="cluster_id"),
            "holdout_ref_nmse_cluster72": paired_cluster_inference(orig_recs_by_method[ma], orig_recs_by_method[mb], "holdout_ref_nmse", group_by="cluster_id"),
            "holdout_ch_nll_cluster72": paired_cluster_inference(orig_recs_by_method[ma], orig_recs_by_method[mb], "holdout_ch_nll", group_by="cluster_id"),
            "payload_ser_session9": paired_cluster_inference(orig_recs_by_method[ma], orig_recs_by_method[mb], "payload_ser", group_by="capture_id"),
        }

    # 6) Controlled Physical Stress-Test Perturbation Suite (separate result track, preserves original intact!)
    print("=== STEP 5: CONTROLLED PHYSICAL STRESS-TEST SUITE (3 PERTURBATION MODES) ===")
    stress_modes = ("stress_low_snr", "stress_cfo_phase_jump", "stress_burst_impairment")
    rng_st = np.random.default_rng(2026112199)

    # Create stress-perturbed train (subset 720), dev (432), and test (1080 = 360 per stress mode across all 9 captures)
    st_train_pkts, st_train_pools = [], []
    for i, (p, pl) in enumerate(zip(train_pkts[::2], train_pools[::2])):
        md = stress_modes[i % 3]
        st_train_pkts.append(apply_controlled_stress(p, rng_st, mode=md))
        st_train_pools.append(pl)

    st_dev_pkts, st_dev_pools = [], []
    for i, (p, pl) in enumerate(zip(dev_pkts[::2], dev_pools[::2])):
        md = stress_modes[i % 3]
        st_dev_pkts.append(apply_controlled_stress(p, rng_st, mode=md))
        st_dev_pools.append(pl)

    st_test_pkts, st_test_pools = [], []
    # Every test packet in a balanced 1,080-packet suite (40/cap x 9 caps x 3 modes -> actually 120/cap = 1080)
    for i, (p, pl) in enumerate(zip(test_pkts, test_pools)):
        if (i % 4) == 3:  # 1,080 packets (120 per capture, 360 per stress mode, balanced across all 72 clusters)
            continue
        md = stress_modes[len(st_test_pkts) % 3]
        st_test_pkts.append(apply_controlled_stress(p, rng_st, mode=md))
        st_test_pools.append(pl)

    st_outputs: dict[str, list[ReceiverOutput]] = {}
    for cmode in ("ls", "mmse", "mmse_slope"):
        st_outputs[f"classical_{cmode}"] = [run_classical_receiver(p, best_cfg, mode=cmode) for p in st_test_pkts]
    for p, pl in zip(st_test_pkts, st_test_pools):
        ao, _ = run_experiment_a_arms(p, pl, best_cfg, local_k=4, content_temp=tau_a)
        for k in ("A1_local_window_k4", "A2_global_uniform", "A3_global_snr_weight", "A4_global_content_attn"):
            st_outputs.setdefault(k, []).append(ao[k])

    st_banks_tr = [build_common_candidate_bank(p, pl, best_cfg, content_temp=tau_a) for p, pl in zip(st_train_pkts, st_train_pools)]
    st_banks_dv = [build_common_candidate_bank(p, pl, best_cfg, content_temp=tau_a) for p, pl in zip(st_dev_pkts, st_dev_pools)]
    st_banks_te = [build_common_candidate_bank(p, pl, best_cfg, content_temp=tau_a) for p, pl in zip(st_test_pkts, st_test_pools)]

    for bk in st_banks_te:
        bo = run_experiment_b_arms(bk, post_temp=tau_b, block_temp=blk_t)
        for k, v in bo.items():
            st_outputs.setdefault(k, []).append(v)

    res_c_stress = train_and_infer_experiment_c(
        st_train_pkts, st_banks_tr, st_dev_pkts, st_banks_dv, st_test_pkts, st_banks_te,
        init_temp=tau_b, block_temp=blk_t, seeds=(20261121, 20261122, 20261123), epochs=45, batch_size=64, lr=4e-3,
    )
    c_stress_multi_seed: dict[str, Any] = {}
    for vname, vobj in res_c_stress.items():
        sel_seed = vobj["dev_selected_seed"]
        st_outputs[vname] = vobj["test_outputs_by_seed"][sel_seed]
        s_mets = []
        for srun in vobj["seed_runs"]:
            sd = srun["seed"]
            _, sm_sd = _eval_batch(st_test_pkts, vobj["test_outputs_by_seed"][sd], f"st_{vname}_s{sd}")
            s_mets.append({**srun, "test_ser": sm_sd["payload_ser"], "test_bit_nll": sm_sd["payload_bit_nll_nats"], "test_bit_ece": sm_sd.get("payload_bit_ece")})
        c_stress_multi_seed[vname] = {
            "dev_selected_seed": sel_seed,
            "seeds": s_mets,
            "test_ser_mean": float(np.mean([m["test_ser"] for m in s_mets])),
            "test_ser_std": float(np.std([m["test_ser"] for m in s_mets], ddof=1)),
            "test_bit_nll_mean": float(np.mean([m["test_bit_nll"] for m in s_mets])),
            "test_bit_nll_std": float(np.std([m["test_bit_nll"] for m in s_mets], ddof=1)),
        }

    st_recs_by_method: dict[str, list[PacketEvalRecord]] = {}
    st_summ_by_method: dict[str, Any] = {}
    by_stress_mode: dict[str, dict[str, Any]] = {md: {} for md in stress_modes}
    for mname, outs in st_outputs.items():
        recs, sm = _eval_batch(st_test_pkts, outs, mname)
        st_recs_by_method[mname] = recs
        st_summ_by_method[mname] = sm
        for md in stress_modes:
            sub_idx = [i for i, p in enumerate(st_test_pkts) if p.perturbation_tag == md]
            sub_recs = [recs[i] for i in sub_idx]
            by_stress_mode[md][mname] = aggregate_records(sub_recs)
        print(
            f"[STRESS-TE] {mname:26s} | SER={sm['payload_ser']:.5f} BER={sm['payload_ber']:.5f} "
            f"PER={sm['payload_per']:.4f} NLL={sm['payload_bit_nll_nats']:.4f} "
            f"ECE={sm.get('payload_bit_ece', 0):.4f} hRefNMSE={sm['holdout_ref_nmse']:.5f}"
        )

    paired_stress: dict[str, Any] = {}
    for label, ma, mb in (
        ("A4_vs_A2_unif", "A2_global_uniform", "A4_global_content_attn"),
        ("B2_lin_vs_B1_hard", "B1_hard_select", "B2_linear_fusion_plugin"),
        ("B3_mix_vs_B1_hard", "B1_hard_select", "B3_mixture_likelihood"),
        ("B3_mix_vs_B2_lin", "B2_linear_fusion_plugin", "B3_mixture_likelihood"),
        ("B3_mix_vs_B3a_sym", "B3a_sym_mixture", "B3_mixture_likelihood"),
        ("B3_mix_vs_B4_avgllr", "B4_avg_llr_antipattern", "B3_mixture_likelihood"),
        ("B2_tot_vs_B2b_within", "B2b_linear_within_only", "B2_linear_fusion_plugin"),
        ("C1_vs_B3_mix", "B3_mixture_likelihood", "C1_attn_weight_net"),
        ("C2_vs_B3_mix", "B3_mixture_likelihood", "C2_attn_weight_resid_net"),
        ("C2_vs_classical_mmse", "classical_mmse", "C2_attn_weight_resid_net"),
    ):
        paired_stress[label] = {
            "payload_ser_cluster72": paired_cluster_inference(st_recs_by_method[ma], st_recs_by_method[mb], "payload_ser", group_by="cluster_id"),
            "payload_bit_nll_cluster72": paired_cluster_inference(st_recs_by_method[ma], st_recs_by_method[mb], "payload_bit_nll", group_by="cluster_id"),
            "holdout_ref_nmse_cluster72": paired_cluster_inference(st_recs_by_method[ma], st_recs_by_method[mb], "holdout_ref_nmse", group_by="cluster_id"),
            "holdout_ch_nll_cluster72": paired_cluster_inference(st_recs_by_method[ma], st_recs_by_method[mb], "holdout_ch_nll", group_by="cluster_id"),
        }

    # 7) Theoretical Risk Check for content_addressing_scalar (Eqs. 1-7 on real CFO estimation)
    cfo_true = np.array([r["offline_cfo_hz"] for r in a_audit_rows], dtype=np.float64)
    cfo_var_prior = float(np.var(cfo_true))
    theory_a_check = {
        "n_test_packets": len(a_audit_rows),
        "target_cfo_total_var_hz2": cfo_var_prior,
        "arms": {},
    }
    for ak in ("A1_local_window_k4", "A2_global_uniform", "A3_global_snr_weight", "A4_global_content_attn", "A5_content_cold_hard", "A6_content_hot"):
        err2 = np.array([r[f"{ak}__cfo_prior_err2"] for r in a_audit_rows], dtype=np.float64)
        hit_c = np.array([r[f"{ak}__hit_cap"] for r in a_audit_rows], dtype=np.float64)
        hit_f = np.array([r[f"{ak}__hit_freq"] for r in a_audit_rows], dtype=np.float64)
        w_l2 = np.array([r[f"{ak}__w_l2_err2"] for r in a_audit_rows], dtype=np.float64)
        w_var = np.array([r[f"{ak}__w_var_sum"] for r in a_audit_rows], dtype=np.float64)
        theory_a_check["arms"][ak] = {
            "cfo_prior_mse_hz2": float(np.mean(err2)),
            "cfo_prior_rmse_hz": float(math.sqrt(np.mean(err2))),
            "mean_same_capture_weight": float(np.mean(hit_c)),
            "mean_same_freq_weight": float(np.mean(hit_f)),
            "mean_w_l2_dist_to_ideal2": float(np.mean(w_l2)),
            "mean_w_var_from_uniform": float(np.mean(w_var)),
        }

    # 8) Assemble full summary JSON, write compact per-packet CSVs, and render figures
    summary = {
        "study": "EM-FloatFormer Real-Data Study: CTTC Indoor Multi-Band/Multi-SNR QPSK",
        "zenodo_doi": "10.5281/zenodo.4895133",
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "torch_num_threads": os.environ.get("TORCH_NUM_THREADS"),
            "total_elapsed_sec": time.perf_counter() - t_start,
            "peak_rss_mb": _peak_rss_mb(),
        },
        "leakage_audit": leak_info,
        "file1_vs_file2_validation": {
            "n_packets": len(pkts_20_f1),
            "file1_self_rrc": {k: sm_f1[k] for k in ("payload_ser", "payload_ber", "payload_evm_unbiased_db", "holdout_ref_nmse")},
            "file2_official": {k: sm_f2[k] for k in ("payload_ser", "payload_ber", "payload_evm_unbiased_db", "holdout_ref_nmse")},
        },
        "anchors": [asdict(a) for a in anchors],
        "dev_tuning": {
            "best_classical_cfg": asdict(best_cfg),
            "best_content_temp": tau_a,
            "best_post_temp": tau_b,
            "best_block_temp": blk_t,
            "cls_grid_log": tune_res["cls_grid_log"],
            "a_grid_log": tune_res["a_grid_log"],
            "b_grid_log": tune_res["b_grid_log"],
        },
        "theory_a_content_addressing_check": theory_a_check,
        "experiment_c_original_multiseed": c_multi_seed_summary,
        "experiment_c_stress_multiseed": c_stress_multi_seed,
        "test_original": {
            "n_packets": len(test_pkts),
            "methods": orig_summ_by_method,
            "paired_inference": paired_orig,
        },
        "test_stress": {
            "n_packets": len(st_test_pkts),
            "methods": st_summ_by_method,
            "by_stress_mode": by_stress_mode,
            "paired_inference": paired_stress,
        },
    }

    # Save compact per-packet CSVs for key representative methods (keeps git repo < 3 MB total!)
    key_orig_csv_methods = (
        "classical_ls",
        "classical_mmse",
        "classical_mmse_slope",
        "A1_local_window_k4",
        "A2_global_uniform",
        "A3_global_snr_weight",
        "A4_global_content_attn",
        "B1_hard_select",
        "B2_linear_fusion_plugin",
        "B3_mixture_likelihood",
        "C1_attn_weight_net",
        "C2_attn_weight_resid_net",
    )
    orig_rows = [r.to_flat_dict() for m in key_orig_csv_methods for r in orig_recs_by_method[m]]
    _write_csv(out_dir / "per_packet_test_original.csv", orig_rows)

    key_stress_csv_methods = (
        "classical_mmse",
        "A4_global_content_attn",
        "B1_hard_select",
        "B2_linear_fusion_plugin",
        "B3_mixture_likelihood",
        "C2_attn_weight_resid_net",
    )
    st_rows = [r.to_flat_dict() for m in key_stress_csv_methods for r in st_recs_by_method[m]]
    _write_csv(out_dir / "per_packet_test_stress.csv", st_rows)
    _write_csv(out_dir / "expA_content_audit_test.csv", a_audit_rows)

    # Save packet index metadata for full reproducibility
    idx_rows = [
        {
            "unit_id": p.unit_id,
            "capture_id": p.capture_id,
            "freq_mhz": p.freq_mhz,
            "nom_snr_db": p.nom_snr_db,
            "cluster_id": p.cluster_id,
            "loop_id": p.loop_id,
            "split": p.split,
            "role": p.role,
            "rx_start_sample": p.rx_start_sample,
            "src_sym_start": p.src_sym_start,
            "time_sec": round(p.time_sec, 6),
        }
        for sp in ("train", "dev", "test")
        for role in ("target", "ref")
        for p in ds_all[sp][role]
    ]
    _write_csv(out_dir / "packet_index.csv", idx_rows)

    figs = _make_figures(summary, a_audit_rows, out_dir / "figures")
    summary["figures"] = figs
    _write_json(out_dir / "full_results_summary.json", summary)
    print(f"=== FULL STUDY COMPLETE in {summary['environment']['total_elapsed_sec']:.1f}s | Peak RSS={summary['environment']['peak_rss_mb']:.1f} MB ===")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CTTC Indoor QPSK empirical study.")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--work-dir", type=Path, default=Path("/tmp/agy_cli1_cttc/work"))
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/agy_cli1_cttc/out"))
    args = parser.parse_args()
    if args.mode == "smoke":
        run_smoke_check(args.work_dir, args.out_dir)
    else:
        run_full_study(args.work_dir, args.out_dir)


if __name__ == "__main__":
    main()
