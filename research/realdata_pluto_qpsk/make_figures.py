#!/usr/bin/env python3
"""Generate publication-quality 300-DPI figures for ADALM-PLUTO SDR QPSK real-data study.

Produces 5 comprehensive figures saved to `--out-dir`:
  1. fig1_pluto_hardware_audit.png         - 7,096-frame channel evolution, AC-coupling proof, burst jumps & splits
  2. fig2_group_a_content_addressing.png   - Group A scope vs content addressing, burst-boundary breakdown & attention map
  3. fig3_group_b_posterior_fusion.png     - Group B Hard MAP vs Mean Plug-in vs Mixture Likelihood across all stress regimes
  4. fig4_stress_constellations.png        - Real equalized QPSK constellation convergence under raw/AWGN/CFO/PhaseJump
  5. fig5_calibration_and_group_c.png      - Held-out probe uncertainty calibration & Group C (3 seeds) G=16 paired forest plot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from receivers import (
    CFO_GRID,
    K_ALL,
    K_CENTER,
    PRIOR_DC_RATIO,
    ReceiverHyperparams,
    build_causal_history_library,
    build_shared_b_candidates,
    compute_content_prior,
    decode_group_b_variants,
    estimate_single_frame_pilot_stats,
)
from waveform_frontend import PAYLOAD_SLICE, PILOT_SLICE, QPSK_CONST

plt.rcParams.update({
    "font.size": 9.5,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.grid": True,
    "grid.alpha": 0.28,
})

COLORS = {
    "base_ls_1tap": "#7f7f7f",
    "base_mmse_dc_cfo": "#333333",
    "A1_local_single": "#8c564b",
    "A2_neighbor_k5": "#ff7f0e",
    "A3_global_uniform": "#9467bd",
    "A4_global_reliability": "#17becf",
    "A5_global_content_attn": "#1f77b4",
    "B1_hard_map": "#d62728",
    "B2_posterior_mean_plugin": "#e377c2",
    "B3a_sym_mixture": "#2ca02c",
    "B3b_joint_mixture": "#006400",
    "C1_neural_content_attn": "#0b3d91",
    "C2_neural_cal_mixture": "#008080",
}


def make_fig1_hardware_audit(work_dir: Path, out_dir: Path) -> None:
    tx = np.load(work_dir / "tx_c64.npy", mmap_mode="r")
    d = np.load(work_dir / "extracted_symbols_7096.npz")
    y1 = d["sym_copy1"].astype(np.complex128)
    s = np.asarray(tx, dtype=np.complex128)
    n = len(y1)

    h = np.empty(n, dtype=np.complex128)
    dc = np.empty(n, dtype=np.complex128)
    for f in range(n):
        X = np.column_stack([s[f], np.ones(126)])
        w, _, _, _ = np.linalg.lstsq(X, y1[f], rcond=None)
        h[f], dc[f] = w[0], w[1]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))

    # (a) Channel amplitude & phase trajectory + temporal splits
    ax = axes[0, 0]
    t_idx = np.arange(n)
    ax.plot(t_idx, np.abs(h) * 1e3, color="#1f77b4", lw=0.7, alpha=0.85, label="|h(t)| (x1e-3)")
    ax2 = ax.twinx()
    ax2.plot(t_idx, np.angle(h), color="#d62728", lw=0.5, alpha=0.55, label="angle h(t) (rad)")
    ax2.set_ylabel("Channel Phase (rad)", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    ax2.grid(False)
    # Shade splits
    ax.axvspan(200, 3200, color="#2ca02c", alpha=0.07, label="Train [200,3200)")
    ax.axvspan(3200, 3300, color="#7f7f7f", alpha=0.22, label="Guard (100)")
    ax.axvspan(3500, 4500, color="#ff7f0e", alpha=0.09, label="Dev [3500,4500)")
    ax.axvspan(4500, 4600, color="#7f7f7f", alpha=0.22)
    ax.axvspan(4800, 7040, color="#1f77b4", alpha=0.09, label="Test [4800,7040) G=16")
    ax.set_xlabel("Frame Index t")
    ax.set_ylabel("Channel Magnitude |h(t)| (x10^-3)", color="#1f77b4")
    ax.set_title("(a) ADALM-PLUTO 7,096-Frame Channel Evolution & Temporal Splits")
    ax.legend(loc="upper left", ncol=2, frameon=True, fontsize=7.2)

    # (b) Hardware Zero-IF AC-Coupling verification: Re(d/h) vs Re(-mean(s))
    ax = axes[0, 1]
    dc_rel = dc / h
    neg_sbar = -np.mean(s, axis=1)
    r_re = float(np.corrcoef(dc_rel.real, neg_sbar.real)[0, 1])
    r_im = float(np.corrcoef(dc_rel.imag, neg_sbar.imag)[0, 1])
    step = 4
    ax.scatter(neg_sbar.real[::step], dc_rel.real[::step], s=5, alpha=0.35, color="#1f77b4", label=f"Real (r={r_re:.4f})")
    ax.scatter(neg_sbar.imag[::step], dc_rel.imag[::step], s=5, alpha=0.35, color="#d62728", label=f"Imag (r={r_im:.4f})")
    lims = [min(neg_sbar.real.min(), neg_sbar.imag.min()) - 0.02, max(neg_sbar.real.max(), neg_sbar.imag.max()) + 0.02]
    ax.plot(lims, lims, "k--", lw=1.1, label="y = x (exact AC coupling)")
    ax.set_xlabel("Predicted -mean(Tx_frame)")
    ax.set_ylabel("Measured Normalized DC Offset d_t / h_t")
    ax.set_title("(b) Zero-IF AD9363 Hardware DC-Blocking Verification")
    ax.legend(loc="upper left", frameon=True)

    # (c) 20-frame burst boundary jumps vs in-burst transitions
    ax = axes[1, 0]
    dh = np.abs(h[1:] - h[:-1]) / np.abs(h[:-1])
    mods = np.arange(1, n) % 20
    meds = [float(np.median(dh[mods == m])) * 100.0 for m in range(20)]
    p90s = [float(np.quantile(dh[mods == m], 0.90)) * 100.0 for m in range(20)]
    xs = np.arange(20)
    bars = ax.bar(xs, p90s, color=["#d62728" if m == 0 else "#aec7e8" for m in xs], edgecolor="#333333", lw=0.6, label="90th percentile |dh|/|h|")
    ax.plot(xs, meds, "ko-", ms=3.5, lw=1.2, label="Median |dh|/|h|")
    ax.set_xticks(xs)
    ax.set_xlabel("Frame Index Modulo 20  (t mod 20)")
    ax.set_ylabel("Frame-to-Frame Relative Jump |dh|/|h| (%)")
    ax.set_title("(c) Periodic 20-Frame Hardware Burst Jumps (p90 = 36.4% at t%20=0)")
    ax.legend(loc="upper right", frameon=True)

    # (d) Non-local historical recurrence vs previous-frame error CDF
    ax = axes[1, 1]
    ts = np.arange(500, n, 5)
    err_prev1 = np.abs(h[ts - 1] - h[ts]) / np.abs(h[ts])
    err_far = np.array([np.min(np.abs(h[max(0, t - 200):t - 20] - h[t])) / abs(h[t]) for t in ts])
    err_bb_prev1 = err_prev1[(ts % 20) == 0]
    err_bb_far = err_far[(ts % 20) == 0]
    for arr, lbl, col, ls in [
        (err_prev1, f"Prev-1 frame (all, med={np.median(err_prev1)*100:.1f}%)", "#ff7f0e", "-"),
        (err_far, f"Best causal [t-200:t-20] (all, med={np.median(err_far)*100:.1f}%)", "#1f77b4", "-"),
        (err_bb_prev1, f"Prev-1 at burst jump t%20=0 (med={np.median(err_bb_prev1)*100:.1f}%)", "#d62728", "--"),
        (err_bb_far, f"Best causal [t-200:t-20] at jump (med={np.median(err_bb_far)*100:.1f}%)", "#006400", "--"),
    ]:
        s_arr = np.sort(arr) * 100.0
        y_cdf = np.linspace(0, 1, len(s_arr))
        ax.plot(s_arr, y_cdf, label=lbl, color=col, ls=ls, lw=1.6)
    ax.set_xlim(0, 60)
    ax.set_xlabel("Relative Channel Distance |h_ref - h_t| / |h_t| (%)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("(d) Causal Non-Local Channel Recurrence vs Local Neighbor")
    ax.legend(loc="lower right", frameon=True, fontsize=7.6)

    fig.tight_layout()
    fig.savefig(out_dir / "fig1_pluto_hardware_audit.png", bbox_inches="tight")
    plt.close(fig)


def make_fig2_group_a(ab_test: dict, c_test: dict, plot_npz: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6))

    # (a) Channel NMSE (dB) across AWGN SNR sweep + raw + phase_jump
    ax = axes[0, 0]
    conds = ["raw_original", "awgn_snr10db", "awgn_snr6db", "awgn_snr2db", "awgn_snr_m2db", "phase_jump_snr6db"]
    labels = ["Raw\n(Pluto)", "AWGN\n10 dB", "AWGN\n6 dB", "AWGN\n2 dB", "AWGN\n-2 dB", "PhaseJump\n6 dB"]
    x = np.arange(len(conds))
    arms_a = [
        ("A1_local_single", "A1 Local Single (M=0)"),
        ("A2_neighbor_k5", "A2 Neighbor (k=5)"),
        ("A3_global_uniform", "A3 Global Uniform (M=200)"),
        ("A4_global_reliability", "A4 Global Reliability (M=200)"),
        ("A5_global_content_attn", "A5 Content Attention (M=200)"),
    ]
    for arm, lbl in arms_a:
        vals = [10.0 * np.log10(max(ab_test["conditions"][c][arm]["channel_and_probe"]["ch_nmse"], 1e-8)) for c in conds]
        ax.plot(x, vals, marker="o", ms=5, lw=1.8, color=COLORS[arm], label=lbl)
    c1_vals = [10.0 * np.log10(max(c_test["conditions"][c]["C1_neural_content_attn"]["mean_std"]["ch_nmse"]["mean"], 1e-8)) for c in conds]
    ax.plot(x, c1_vals, marker="s", ms=5.5, lw=1.8, ls="--", color=COLORS["C1_neural_content_attn"], label="C1 Neural Attn (3-seed)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Channel Estimation NMSE (dB, lower is better)")
    ax.set_title("(a) Group A: Reference Scope vs Content Addressing (Test, n=2,240)")
    ax.legend(loc="upper left", frameon=True, fontsize=7.5)

    # (b) Payload SER (%) across SNR & phase_jump
    ax = axes[0, 1]
    for arm, lbl in arms_a:
        means = [ab_test["conditions"][c][arm]["payload"]["ser"] * 100.0 for c in conds]
        errs = [
            (ab_test["conditions"][c][arm]["payload"]["ser_ci95"]["ci95_hi"] - ab_test["conditions"][c][arm]["payload"]["ser"]) * 100.0
            for c in conds
        ]
        ax.errorbar(x, means, yerr=errs, marker="o", ms=4.5, lw=1.7, capsize=2.5, color=COLORS[arm], label=lbl)
    c1_ser = [c_test["conditions"][c]["C1_neural_content_attn"]["mean_std"]["payload_ser"]["mean"] * 100.0 for c in conds]
    ax.plot(x, c1_ser, marker="s", ms=5, lw=1.7, ls="--", color=COLORS["C1_neural_content_attn"], label="C1 Neural Attn")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Payload QPSK SER (%) ± G=16 95% CI")
    ax.set_title("(b) Group A: Payload SER Separation Across Regimes")
    ax.legend(loc="upper left", frameon=True, fontsize=7.5)

    # (c) Burst-boundary (t%20==0) vs In-burst (t%20!=0) breakdown on raw_original & awgn_snr6db & phase_jump
    ax = axes[1, 0]
    sub_conds = [("raw_original", "Raw Pluto"), ("awgn_snr6db", "AWGN 6dB"), ("phase_jump_snr6db", "PhaseJump 6dB")]
    bar_arms = ["A1_local_single", "A2_neighbor_k5", "A3_global_uniform", "A5_global_content_attn"]
    bar_lbls = ["A1 Local", "A2 Neighbor", "A3 Uniform", "A5 ContentAttn"]
    width = 0.18
    gx = np.arange(len(sub_conds) * 2)
    xlabels = []
    for _, clbl in sub_conds:
        xlabels.extend([f"{clbl}\nIn-Burst", f"{clbl}\nJump (t%20=0)"])
    for bi, (arm, blbl) in enumerate(zip(bar_arms, bar_lbls)):
        vals = []
        for c, _ in sub_conds:
            rb = ab_test["conditions"][c][arm]["regime_breakdown"]
            vals.append(rb["in_burst_ch_nmse"] * 100.0)
            vals.append(rb["burst_boundary_ch_nmse"] * 100.0)
        ax.bar(gx + (bi - 1.5) * width, vals, width=width, color=COLORS[arm], label=blbl, edgecolor="#222", lw=0.4)
    ax.set_xticks(gx)
    ax.set_xticklabels(xlabels, fontsize=7.8)
    ax.set_ylabel("Channel NMSE (%) (lower is better)")
    ax.set_title("(c) In-Burst vs Real 20-Frame Hardware Jump Boundary Breakdown")
    ax.legend(loc="upper left", frameon=True, fontsize=7.8)

    # (d) Attention weight heatmap over 60 contiguous test frames across 3 burst boundaries (phase_jump_snr6db or raw_original)
    ax = axes[1, 1]
    key_hm = "phase_jump_snr6db_attn_weights_60x201" if "phase_jump_snr6db_attn_weights_60x201" in plot_npz else "raw_original_attn_weights_60x201"
    att = plot_npz[key_hm]  # (60, 201): cols 0..199 are lags 200..1, col 200 is clear token!
    # Show lags 1..80 (cols 199 down to 120) + clear token (col 200) for high visual clarity
    show_mat = np.concatenate([att[:, 199:119:-1], att[:, 200:201]], axis=1)  # (60, 81)
    im = ax.imshow(np.log10(np.maximum(show_mat, 1e-4)), aspect="auto", cmap="magma", vmin=-3.5, vmax=0.0)
    for b_row in (0, 20, 40):
        ax.axhline(b_row - 0.5, color="#00ffff", ls="--", lw=0.9, alpha=0.8)
    ax.axvline(79.5, color="#00ff00", ls="-", lw=1.0)
    ax.set_xlabel("Causal Lag (1..80 frames ago)  |  Rightmost Col = Clear Token")
    ax.set_ylabel("Test Frame Offset (Cyan dashes = t%20=0 bursts)")
    ax.set_title("(d) A5 Content-Addressing Attention Weights (log10 w, PhaseJump 6dB)")
    plt.colorbar(im, ax=ax, label="log10(weight)")

    fig.tight_layout()
    fig.savefig(out_dir / "fig2_group_a_content_addressing.png", bbox_inches="tight")
    plt.close(fig)


def make_fig3_group_b(ab_test: dict, c_test: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.6))
    conds = [
        "raw_original", "awgn_snr10db", "awgn_snr6db", "awgn_snr2db",
        "awgn_snr_m2db", "cfo_0p015_raw", "cfo_0p020_snr6db", "phase_jump_snr6db",
    ]
    labels = ["Raw", "10dB", "6dB", "2dB", "-2dB", "CFO\nRaw", "CFO+\n6dB", "Jump+\n6dB"]
    x = np.arange(len(conds))

    b_arms = [
        ("base_mmse_dc_cfo", "Classical MMSE+DC+VV-CFO"),
        ("B1_hard_map", "B1 Hard MAP (84 cand)"),
        ("B2_posterior_mean_plugin", "B2 Posterior Mean Plug-in"),
        ("B3a_sym_mixture", "B3a Per-Sym Mixture"),
        ("B3b_joint_mixture", "B3b Full Joint Mixture (§5.2)"),
    ]

    # (a) Payload SER (%) across all 8 conditions
    ax = axes[0, 0]
    for arm, lbl in b_arms:
        means = [ab_test["conditions"][c][arm]["payload"]["ser"] * 100.0 for c in conds]
        errs = [
            (ab_test["conditions"][c][arm]["payload"]["ser_ci95"]["ci95_hi"] - ab_test["conditions"][c][arm]["payload"]["ser"]) * 100.0
            for c in conds
        ]
        ax.errorbar(x, means, yerr=errs, marker="o", ms=4.5, lw=1.7, capsize=2.5, color=COLORS[arm], label=lbl)
    c2_ser = [c_test["conditions"][c]["C2_neural_cal_mixture"]["mean_std"]["payload_ser"]["mean"] * 100.0 for c in conds]
    ax.plot(x, c2_ser, marker="D", ms=4.8, lw=1.8, ls="--", color=COLORS["C2_neural_cal_mixture"], label="C2 Neural Cal+Mix (3-seed)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Payload SER (%) ± G=16 95% CI")
    ax.set_title("(a) Group B & C: Payload SER Across All 8 Real-Waveform Regimes")
    ax.legend(loc="upper left", frameon=True, fontsize=7.4)

    # (b) Payload FER (%) across all 8 conditions
    ax = axes[0, 1]
    for arm, lbl in b_arms:
        means = [ab_test["conditions"][c][arm]["payload"]["fer"] * 100.0 for c in conds]
        ax.plot(x, means, marker="o", ms=4.5, lw=1.7, color=COLORS[arm], label=lbl)
    c2_fer = [c_test["conditions"][c]["C2_neural_cal_mixture"]["mean_std"]["payload_fer"]["mean"] * 100.0 for c in conds]
    ax.plot(x, c2_fer, marker="D", ms=4.8, lw=1.8, ls="--", color=COLORS["C2_neural_cal_mixture"], label="C2 Neural Cal+Mix")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Payload Frame Error Rate FER (%)")
    ax.set_title("(b) Payload FER (100-Sym Frame, >=1 Err = Frame Error)")
    ax.legend(loc="lower right", frameon=True, fontsize=7.4)

    # (c) Posterior Fusion Theorem 1 Verification: Held-out Probe NMSE (B1 Hard MAP vs B2/B3 Posterior Mean vs C2)
    ax = axes[1, 0]
    w = 0.25
    b1_p = [ab_test["conditions"][c]["B1_hard_map"]["channel_and_probe"]["probe_nmse"] * 100.0 for c in conds]
    b2_p = [ab_test["conditions"][c]["B2_posterior_mean_plugin"]["channel_and_probe"]["probe_nmse"] * 100.0 for c in conds]
    c2_p = [c_test["conditions"][c]["C2_neural_cal_mixture"]["mean_std"]["probe_nmse"]["mean"] * 100.0 for c in conds]
    ax.bar(x - w, b1_p, width=w, color=COLORS["B1_hard_map"], label="B1 Hard MAP", edgecolor="#222", lw=0.4)
    ax.bar(x, b2_p, width=w, color=COLORS["B2_posterior_mean_plugin"], label="B2/B3 Posterior Mean", edgecolor="#222", lw=0.4)
    ax.bar(x + w, c2_p, width=w, color=COLORS["C2_neural_cal_mixture"], label="C2 Neural Calibrated", edgecolor="#222", lw=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Held-Out Pilot Probe [21:26] NMSE (%)")
    ax.set_title("(c) Projection Theorem Verification: Soft Mean Beats Hard MAP on Probe NMSE")
    ax.legend(loc="upper left", frameon=True)

    # (d) Direct SER Gain of B3b Joint Mixture over B1 Hard MAP & B2 Mean Plug-in (pp, positive = B3b better)
    ax = axes[1, 1]
    d_b1 = [(ab_test["conditions"][c]["B1_hard_map"]["payload"]["ser"] - ab_test["conditions"][c]["B3b_joint_mixture"]["payload"]["ser"]) * 100.0 for c in conds]
    d_b2 = [(ab_test["conditions"][c]["B2_posterior_mean_plugin"]["payload"]["ser"] - ab_test["conditions"][c]["B3b_joint_mixture"]["payload"]["ser"]) * 100.0 for c in conds]
    d_vv = [(ab_test["conditions"][c]["base_mmse_dc_cfo"]["payload"]["ser"] - ab_test["conditions"][c]["B3b_joint_mixture"]["payload"]["ser"]) * 100.0 for c in conds]
    ax.bar(x - w, d_b1, width=w, color="#d62728", alpha=0.85, label="B3b Gain vs B1 Hard MAP (pp)")
    ax.bar(x, d_b2, width=w, color="#e377c2", alpha=0.85, label="B3b Gain vs B2 Mean Plug-in (pp)")
    ax.bar(x + w, d_vv, width=w, color="#333333", alpha=0.80, label="B3b Gain vs Classical VV-CFO (pp)")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("SER Reduction by B3b Full Mixture (percentage points)")
    ax.set_title("(d) Decoding Gain of Exact Mixture Marginalization (B3b)")
    ax.legend(loc="upper left", frameon=True, fontsize=7.6)

    fig.tight_layout()
    fig.savefig(out_dir / "fig3_group_b_posterior_fusion.png", bbox_inches="tight")
    plt.close(fig)


def make_fig4_constellations(work_dir: Path, dev_tune: dict, out_dir: Path) -> None:
    """Plot real equalized QPSK constellation scatter across 4 representative conditions on Test frames."""
    bench_dir = work_dir / "benchmarks"
    tx_mm = np.load(work_dir / "tx_c64.npy", mmap_mode="r")
    fhp = dev_tune["frozen_hyperparams"]
    hp = ReceiverHyperparams(
        h_mean_prior=complex(fhp["h_mean_prior_re"], fhp["h_mean_prior_im"]),
        h_var_prior=float(fhp["h_var_prior"]),
        neighbor_k=int(fhp["neighbor_k"]),
        neighbor_decay=float(fhp["neighbor_decay"]),
        neighbor_drift_rel_var=float(fhp["neighbor_drift_rel_var"]),
        global_drift_rel_var=float(fhp["global_drift_rel_var"]),
        attn_temp=float(fhp["attn_temp"]),
        attn_drift_rel_var=float(fhp["attn_drift_rel_var"]),
        clear_prior_log_bias=float(fhp["clear_prior_log_bias"]),
        age_penalty=float(fhp["age_penalty"]),
        b_temp=float(fhp["b_temp"]),
        b_cfo_zero_bonus=float(fhp["b_cfo_zero_bonus"]),
        b_quad_zero_bonus=float(fhp["b_quad_zero_bonus"]),
        b_var_scale=float(fhp["b_var_scale"]),
    )
    cond_list = [
        ("raw_original", "Raw ADALM-PLUTO"),
        ("awgn_snr6db", "AWGN 6 dB"),
        ("cfo_0p020_snr6db", "Residual CFO + 6 dB"),
        ("phase_jump_snr6db", "Burst Phase Jump + 6 dB"),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(11.5, 13.0))
    nb_w = np.exp(-hp.neighbor_decay * np.arange(hp.neighbor_k, 0, -1, dtype=np.float64))
    nb_w /= np.sum(nb_w)
    q_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for row, (cname, title) in enumerate(cond_list):
        z = np.load(bench_dir / f"{cname}_test.npz")
        y_all = np.asarray(z["y_sym"], dtype=np.complex128)
        fids = np.asarray(z["frame_indices"], dtype=np.int32)
        ev_off = int(z["eval_offset"])
        # Build causal library on first 230 frames only (fast!)
        lib = build_causal_history_library(y_all[:ev_off + 25], hp.h_mean_prior, hp.h_var_prior, lib_size=200)

        eq_base, eq_b1, eq_b3b, true_idx = [], [], [], []
        for t in range(ev_off, ev_off + 20):  # 20 frames = 2,000 QPSK payload symbols!
            y = y_all[t]
            y_pil = y[PILOT_SLICE]
            s_pay = np.asarray(tx_mm[fids[t], PAYLOAD_SLICE], dtype=np.complex128)
            idx_t = ((s_pay.real < 0.0).astype(np.int64) << 1) | (s_pay.imag < 0.0).astype(np.int64)
            st = estimate_single_frame_pilot_stats(y_pil, hp.h_mean_prior, hp.h_var_prior)
            # Naive 1-tap LS
            eq_base.append(y[PAYLOAD_SLICE] / st["h_ls_1tap"])

            lh, ld, lvh = lib["h_dd"][t - 200:t], lib["d_dd"][t - 200:t], lib["var_h_dd"][t - 200:t]
            nh, nd, nvh = lh[-hp.neighbor_k:], ld[-hp.neighbor_k:], lvh[-hp.neighbor_k:]
            h_nb = complex(np.sum(nb_w * nh))
            d_nb = complex(np.sum(nb_w * nd))
            v_nb = float(np.sum(nb_w * (nvh + np.abs(nh - h_nb) ** 2 + hp.neighbor_drift_rel_var * np.abs(nh) ** 2)))
            ca = compute_content_prior(y_pil, lh, ld, lvh, st["sigma2_est"], hp)
            priors_3 = [(hp.h_mean_prior, hp.h_var_prior, hp.h_mean_prior * PRIOR_DC_RATIO), (h_nb, v_nb, d_nb), (ca["h_prior"], ca["var_h_prior"], ca["d_prior"])]
            bank = build_shared_b_candidates(y_pil, st["sigma2_est"], priors_3, hp)
            bo = decode_group_b_variants(y, bank)
            eq_b1.append(bo["B1_hard_map"]["s_eq"])
            eq_b3b.append(bo["B3b_joint_mixture"]["s_eq"])
            true_idx.append(idx_t)

        eq_base = np.concatenate(eq_base)
        eq_b1 = np.concatenate(eq_b1)
        eq_b3b = np.concatenate(eq_b3b)
        true_idx = np.concatenate(true_idx)

        for col, (arr, mname) in enumerate([
            (eq_base, "Classical 1-Tap Pilot LS"),
            (eq_b1, "B1 Hard MAP (84-cand)"),
            (eq_b3b, "B3b Full Joint Mixture"),
        ]):
            ax = axes[row, col]
            for qi in range(4):
                m = true_idx == qi
                ax.scatter(arr[m].real, arr[m].imag, s=4, alpha=0.35, color=q_colors[qi])
            ax.scatter(QPSK_CONST.real, QPSK_CONST.imag, s=55, marker="x", color="k", lw=1.5, zorder=5)
            ax.axhline(0, color="k", lw=0.6, alpha=0.5)
            ax.axvline(0, color="k", lw=0.6, alpha=0.5)
            ax.set_xlim(-2.0, 2.0)
            ax.set_ylim(-2.0, 2.0)
            ax.set_aspect("equal")
            if row == 0:
                ax.set_title(mname, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{title}\nImag")
            if row == 3:
                ax.set_xlabel("Real")

    fig.suptitle("Real ADALM-PLUTO Equalized Payload Constellations (20 Test Frames = 2,000 QPSK Symbols)", y=0.995, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_stress_constellations.png", bbox_inches="tight")
    plt.close(fig)


def make_fig5_calibration_and_group_c(ab_test: dict, c_test: dict, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.5))
    conds = [
        "raw_original", "awgn_snr10db", "awgn_snr6db", "awgn_snr2db",
        "awgn_snr_m2db", "cfo_0p015_raw", "cfo_0p020_snr6db", "phase_jump_snr6db",
    ]
    labels = ["Raw", "10dB", "6dB", "2dB", "-2dB", "CFO\nRaw", "CFO+\n6dB", "Jump+\n6dB"]
    x = np.arange(len(conds))

    # (a) Held-out probe [21:26] 95% interval empirical coverage vs ideal 0.95
    ax = axes[0, 0]
    for arm, lbl in [
        ("base_ls_1tap", "Base 1-tap LS"),
        ("A1_local_single", "A1 Local Single"),
        ("A5_global_content_attn", "A5 Content Attn"),
        ("B1_hard_map", "B1 Hard MAP"),
        ("B2_posterior_mean_plugin", "B2/B3 Posterior Mixture"),
    ]:
        vals = [ab_test["conditions"][c][arm]["channel_and_probe"]["probe_cov95"] for c in conds]
        ax.plot(x, vals, marker="o", ms=4.5, lw=1.6, color=COLORS[arm], label=lbl)
    c2_cov = [c_test["conditions"][c]["C2_neural_cal_mixture"]["mean_std"]["probe_cov95"]["mean"] for c in conds]
    ax.plot(x, c2_cov, marker="D", ms=4.8, lw=1.8, ls="--", color=COLORS["C2_neural_cal_mixture"], label="C2 Neural Cal+Mix")
    ax.axhline(0.95, color="k", ls=":", lw=1.3, label="Ideal 95% Coverage")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.55, 1.02)
    ax.set_ylabel("Empirical 95% Complex Circular Coverage")
    ax.set_title("(a) Uncertainty Calibration on Held-Out Probe [21:26]")
    ax.legend(loc="lower left", frameon=True, fontsize=7.4)

    # (b) Held-out probe Gaussian NLL (lower is better)
    ax = axes[0, 1]
    for arm, lbl in [
        ("base_ls_1tap", "Base 1-tap LS"),
        ("A5_global_content_attn", "A5 Content Attn"),
        ("B1_hard_map", "B1 Hard MAP"),
        ("B2_posterior_mean_plugin", "B2/B3 Posterior Mixture"),
    ]:
        vals = [ab_test["conditions"][c][arm]["channel_and_probe"]["probe_nll"] for c in conds]
        ax.plot(x, vals, marker="o", ms=4.5, lw=1.6, color=COLORS[arm], label=lbl)
    c2_nll = [c_test["conditions"][c]["C2_neural_cal_mixture"]["mean_std"]["probe_nll"]["mean"] for c in conds]
    ax.plot(x, c2_nll, marker="D", ms=4.8, lw=1.8, ls="--", color=COLORS["C2_neural_cal_mixture"], label="C2 Neural Cal+Mix")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Held-Out Probe Proper Complex Gaussian NLL (nats)")
    ax.set_title("(b) Probabilistic Predictive Likelihood (NLL, lower is better)")
    ax.legend(loc="upper right", frameon=True, fontsize=7.6)

    # (c) Group C (3 seeds) vs Dev-tuned Classical B3b paired G=16 Forest Plot (SER diff in pp, positive = C2 better)
    ax = axes[1, 0]
    y_pos = np.arange(len(conds))
    means_pp, lo_pp, hi_pp, p_holms = [], [], [], []
    for c in conds:
        st = c_test["paired_vs_classical"][c]["C2_vs_B3b_ser"]
        means_pp.append(st["improvement_mean"] * 100.0)
        lo_pp.append(st["ci95_lo"] * 100.0)
        hi_pp.append(st["ci95_hi"] * 100.0)
        p_holms.append(st["p_holm"])
    means_arr = np.array(means_pp)
    err_lo = means_arr - np.array(lo_pp)
    err_hi = np.array(hi_pp) - means_arr
    ax.axvline(0.0, color="k", ls="--", lw=1.0)
    ax.errorbar(means_arr, y_pos, xerr=[err_lo, err_hi], fmt="o", color="#008080", ecolor="#005555", capsize=3.5, lw=1.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{lbl.replace(chr(10), ' ')} (pH={ph:.3f})" for lbl, ph in zip(labels, p_holms)], fontsize=8.0)
    ax.set_xlabel("SER Improvement: Classical B3b minus Neural C2 (pp, + = C2 better)")
    ax.set_title("(c) Group C (3-Seed Mean) vs Classical B3b (G=16 Paired 95% CI)")

    # (d) Engineering cost vs accuracy trade-off on cfo_0p020_snr6db + phase_jump_snr6db average SER
    ax = axes[1, 1]
    cost_arms = [
        ("base_ls_1tap", "Base 1-tap"),
        ("base_mmse_dc_cfo", "Base VV-CFO"),
        ("A1_local_single", "A1 Local"),
        ("A2_neighbor_k5", "A2 Neighbor"),
        ("A5_global_content_attn", "A5 ContentAttn"),
        ("B1_hard_map", "B1 Hard MAP"),
        ("B2_posterior_mean_plugin", "B2 MeanPlug"),
        ("B3b_joint_mixture", "B3b JointMix"),
    ]
    offsets = {
        "Base 1-tap": (6, -12),
        "Base VV-CFO": (6, -4),
        "A1 Local": (6, -3),
        "A2 Neighbor": (6, 4),
        "A5 ContentAttn": (6, 7),
        "B1 Hard MAP": (6, -3),
        "B2 MeanPlug": (6, -3),
        "B3b JointMix": (-66, 6),
    }
    for arm, lbl in cost_arms:
        lat = float(np.mean([ab_test["conditions"][c][arm]["engineering"]["latency_median_ms"] for c in ("awgn_snr6db", "cfo_0p020_snr6db", "phase_jump_snr6db")]))
        ser = float(np.mean([ab_test["conditions"][c][arm]["payload"]["ser"] * 100.0 for c in ("awgn_snr6db", "cfo_0p020_snr6db", "phase_jump_snr6db")]))
        ax.scatter(lat, ser, s=75, color=COLORS[arm], edgecolor="#111", zorder=4)
        ax.annotate(lbl, (lat, ser), textcoords="offset points", xytext=offsets.get(lbl, (5, 4)), fontsize=7.6)
    # Add C1 & C2
    for ck, lbl, col, oxy in [
        ("C1_neural_content_attn", "C1 NeuralAttn", COLORS["C1_neural_content_attn"], (6, -11)),
        ("C2_neural_cal_mixture", "C2 NeuralMix", COLORS["C2_neural_cal_mixture"], (6, -4)),
    ]:
        lat = float(np.mean([c_test["conditions"][c][ck]["mean_std"]["latency_median_ms"]["mean"] for c in ("awgn_snr6db", "cfo_0p020_snr6db", "phase_jump_snr6db")]))
        ser = float(np.mean([c_test["conditions"][c][ck]["mean_std"]["payload_ser"]["mean"] * 100.0 for c in ("awgn_snr6db", "cfo_0p020_snr6db", "phase_jump_snr6db")]))
        ax.scatter(lat, ser, s=90, marker="D", color=col, edgecolor="#111", zorder=5)
        ax.annotate(lbl, (lat, ser), textcoords="offset points", xytext=oxy, fontsize=7.6, fontweight="bold")
    ax.set_xlim(-0.2, 9.6)
    ax.set_xlabel("Single-Thread Median Inference Latency (ms / frame, excl. 0.78ms frontend)")
    ax.set_ylabel("Mean Payload SER (%) across 3 Stress Regimes (6dB / CFO+6dB / Jump+6dB)")
    ax.set_title("(d) Accuracy vs Single-Thread CPU Compute Frontier")

    fig.tight_layout()
    fig.savefig(out_dir / "fig5_calibration_and_group_c.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ab_test = json.loads((args.results_dir / "ab_test_summary.json").read_text(encoding="utf-8"))
    c_test = json.loads((args.results_dir / "c_neural_test_summary.json").read_text(encoding="utf-8"))
    dev_tune = json.loads((args.results_dir / "dev_tuning_summary.json").read_text(encoding="utf-8"))
    plot_npz = dict(np.load(args.results_dir / "plot_data_ab.npz"))

    make_fig1_hardware_audit(args.work_dir, args.out_dir)
    make_fig2_group_a(ab_test, c_test, plot_npz, args.out_dir)
    make_fig3_group_b(ab_test, c_test, args.out_dir)
    make_fig4_constellations(args.work_dir, dev_tune, args.out_dir)
    make_fig5_calibration_and_group_c(ab_test, c_test, args.out_dir)
    print("Generated all 5 figures in", args.out_dir)


if __name__ == "__main__":
    main()
