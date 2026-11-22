"""Generate publication-quality figures for the EM-FloatFormer real-OTA OFDM theory & stress study."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.grid": True,
        "grid.alpha": 0.28,
        "grid.linestyle": "--",
        "figure.dpi": 200,
    }
)

COND_ORDER = [
    "C0_raw_ota",
    "C1_awgn_stress",
    "C2_cfo_doppler",
    "C3_phase_jump_jitter",
    "C4_nbi_sparse7",
]
COND_LABELS = [
    "C0: Raw OTA",
    "C1: AWGN (-6dB)",
    "C2: CFO/Doppler",
    "C3: Phase Jump",
    "C4: NBI+7Pilots",
]


def plot_fig1_group_a(ga: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))

    # Panel A: Sparse-pilot real OTA (14, 7, 4 pilots) channel NMSE across methods
    a1 = ga["experiment_A1_sparse_pilot_historical_bank"]
    cfgs = ["14_pilots_full", "7_pilots_sparse", "4_pilots_ultrasparse"]
    cfg_labels = ["14 Pilots (Full)", "7 Pilots (Sparse)", "4 Pilots (Ultra-Sparse)"]
    methods = [
        ("single_tti_ls_linear", "Single-TTI LS-Lin", "#7f8c8d"),
        ("local_k4_uniform", "Local k=4 Unif", "#e67e22"),
        ("global_m16_uniform", "Global M=16 Unif", "#95a5a6"),
        ("classical_invdist_weight", "Inv-Dist Weight", "#3498db"),
        ("content_addr_calibrated", "Content-Addr (Cal)", "#1b7837"),
        ("content_addr_cold_hard", "Over-Cold Hard", "#c0392b"),
    ]
    x = np.arange(len(cfgs))
    w = 0.13
    ax = axes[0]
    for i, (m, lbl, col) in enumerate(methods):
        vals = [a1[c]["methods"][m]["ch_nmse"] for c in cfgs]
        ax.bar(x + (i - 2.5) * w, vals, width=w, label=lbl, color=col, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(cfg_labels)
    ax.set_ylim(0.0, 0.42)
    ax.set_ylabel("Channel NMSE (Test 240 TTIs)")
    ax.set_title("(a) Real OTA Sparse-Pilot Historical Bank (A1)")
    ax.legend(ncol=2, loc="upper left", frameon=True, fontsize=7.6)

    # Panel B: Prop A & Prop B exact risk verification vs temperature sweep (independent scoring)
    a2 = ga["experiment_A2_propAB_and_eq9_verification"]
    p = a2["empirical_model_params"]
    reg_ind = a2["regimes"]["A2_bin_indep_exact"]
    sw = reg_ind["temperature_sweep"]
    mults = [s["tau_mult"] for s in sw]
    r_emp = [s["R_emp"] for s in sw]
    r_pred = [s["R_propB_pred"] for s in sw]
    r_loc = [s["R_local_k4_emp"] for s in sw]

    ax = axes[1]
    ax.semilogx(mults, r_emp, "o-", color="#1b7837", lw=2.0, ms=6, label=r"Empirical $R_\alpha^{\mathrm{emp}}$")
    ax.semilogx(mults, r_pred, "s--", color="#2c3e50", lw=1.6, ms=5, label=r"Prop. B $v_1 + D\,\mathbb{E}\|e_J-\alpha\|_2^2$")
    ax.semilogx(mults, r_loc, "^:", color="#e67e22", lw=1.4, ms=5, label="Fixed Local-$k$ ($k=4$) Emp.")
    ax.axhline(p["L_k_propA_lower_bound"], color="#d35400", ls="-.", lw=1.3, label=rf"Prop. A Bound $L_k={p['L_k_propA_lower_bound']:.4f}$")
    ax.axhline(p["v1_oracle_residual_var"], color="#7f8c8d", ls=":", lw=1.2, label=rf"Oracle $v_1={p['v1_oracle_residual_var']:.4f}$")
    ax.set_xlabel(r"Temperature Multiplier $\tau / \tau^*$")
    ax.set_ylabel("Normalized Innovation MSE")
    ax.set_title(r"(b) Prop. A/B Exact Risk Verification ($T \perp (\theta,\varepsilon)\mid J$)")
    ax.legend(loc="best", frameon=True, fontsize=7.5)

    # Panel C: RF Disjoint Subcarriers vs Shared-Obs §5.1 Eq. (9) Correlation Bias
    reg_rf = a2["regimes"]["A2_rf_disjoint_subcarriers"]
    reg_sh = a2["regimes"]["A2_shared_obs_eq9_bias"]
    sw_rf = reg_rf["temperature_sweep"]
    sw_sh = reg_sh["temperature_sweep"]
    ax = axes[2]
    ax.semilogx(
        [s["tau_mult"] for s in sw_rf],
        [s["Delta_corr_emp_minus_pred"] for s in sw_rf],
        "o-",
        color="#2980b9",
        lw=1.8,
        label="Disjoint RF Subcarriers (94 sc)",
    )
    ax.semilogx(
        [s["tau_mult"] for s in sw_sh],
        [s["Delta_corr_emp_minus_pred"] for s in sw_sh],
        "D-",
        color="#c0392b",
        lw=1.8,
        label=r"Shared Noisy Obs (§5.1 Eq. 9 Bias)",
    )
    ax.semilogx(
        mults,
        [s["Delta_corr_emp_minus_pred"] for s in sw],
        "s--",
        color="#1b7837",
        lw=1.4,
        label="Strict Indep. Fingerprint",
    )
    ax.axhline(0.0, color="black", lw=1.0, ls="-")
    ax.set_xlabel(r"Temperature Multiplier $\tau / \tau^*$")
    ax.set_ylabel(r"Discrepancy $R_\alpha^{\mathrm{emp}} - R_\alpha^{\mathrm{nominal}}$")
    ax.set_title(r"(c) Shared-Observation Bias Check (§5.1 Eq. 9)")
    ax.legend(loc="best", frameon=True, fontsize=7.5)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig2_group_b(gb: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))

    # Panel A: 5-condition stress matrix BER across Group B methods
    methods = [
        ("B0_classical_single", "B0: Classical LS", "#7f8c8d"),
        ("B1_hard_select", "B1: MAP Hard", "#e67e22"),
        ("B2a_linear_fusion_within", "B2a: Lin-Fus (Within)", "#9b59b6"),
        ("B2b_linear_fusion_totvar", "B2b: Lin-Fus (TotVar)", "#3498db"),
        ("B3_mixture_likelihood", "B3: Mix-Post LLR", "#1b7837"),
    ]
    x = np.arange(len(COND_ORDER))
    w = 0.15
    ax = axes[0]
    for i, (m, lbl, col) in enumerate(methods):
        vals = [gb["conditions"][c]["methods"][m]["ber"] for c in COND_ORDER]
        ax.bar(x + (i - 2.0) * w, vals, width=w, label=lbl, color=col, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(COND_LABELS, rotation=18, ha="right")
    ax.set_ylabel("Test Bit Error Rate (BER)")
    ax.set_title("(a) 5-Condition RF Stress Matrix (Group B)")
    ax.legend(loc="upper left", fontsize=7.5, frameon=True)

    # Panel B: AWGN SNR-drop sweep (0..12 dB)
    sw = gb["awgn_sweep"]
    drops = [r["awgn_drop_db"] for r in sw]
    ax = axes[1]
    for m, lbl, col in methods:
        ax.plot(drops, [r["ber"][m] for r in sw], "o-", color=col, lw=1.8, ms=5, label=lbl)
    ax.set_xlabel("Added AWGN Stress Level (dB drop)")
    ax.set_ylabel("Test BER")
    ax.set_title("(b) Controlled AWGN Stress Sweep")
    ax.legend(loc="upper left", fontsize=7.5, frameon=True)

    # Panel C: C3 Phase-Jump Ambiguity MSE-vs-BER Decoupling (posterior_fusion §5.1)
    amb = gb["c3_ambiguity_mse_ber_decoupling_check"]
    temps = [r["ll_temp_no_hmm"] for r in amb]
    ax = axes[2]
    ax.plot(temps, [r["ber"]["B1_hard_select"] for r in amb], "s-", color="#e67e22", lw=2.0, label="B1 Hard-Select BER")
    ax.plot(temps, [r["ber"]["B2a_linear_fusion_within"] for r in amb], "^--", color="#9b59b6", lw=1.8, label="B2a Linear Fusion BER")
    ax.plot(temps, [r["ber"]["B2b_linear_fusion_totvar"] for r in amb], "v:", color="#3498db", lw=1.8, label="B2b Lin-Fus TotVar BER")
    ax.plot(temps, [r["ber"]["B3_mixture_likelihood"] for r in amb], "o-", color="#1b7837", lw=2.2, label="B3 Mixture LLR BER")
    ax.set_xlabel("Ambiguity Temperature (No-HMM C3 Phase Jump)")
    ax.set_ylabel("Test BER")
    ax.set_ylim(0.342, 0.376)
    ax.set_title("(c) Phase Ambiguity: Lower MSE ≠ Lower BER (§5.1)")
    ax.legend(loc="upper left", fontsize=7.5, frameon=True)

    # Annotate channel MSE ordering cleanly below curves
    last = amb[-1]
    ax.text(
        0.97,
        0.04,
        f"At T={last['ll_temp_no_hmm']:.0f} (amp ratio={last['amp_collapse_ratio']:.2f}):\n"
        f"Ch MSE: Hard={last['ch_mse_B1_hard']:.3f} > Mix={last['ch_mse_B2_mix']:.3f}\n"
        f"Yet Mix-LLR BER={last['ber']['B3_mixture_likelihood']:.3f} << Lin={last['ber']['B2b_linear_fusion_totvar']:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.3,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#bdc3c7", alpha=0.94),
    )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig3_group_c(gb: dict, gc: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5))
    roles = [
        ("B3", "Group B3: Phys Mix (0p)", "#7f8c8d"),
        ("RoleA_PhysParallelTransport", f"Role A: Phys Transport ({gc['trainable_real_params']['RoleA_PhysParallelTransport']}p)", "#1b7837"),
        ("RoleB1_MatchedTransformer", f"Role B1: Real Transf. ({gc['trainable_real_params']['RoleB1_MatchedTransformer']}p)", "#2980b9"),
        ("RoleB2_MatchedMLP", f"Role B2: Real MLP ({gc['trainable_real_params']['RoleB2_MatchedMLP']}p)", "#c0392b"),
    ]
    x = np.arange(len(COND_ORDER))
    w = 0.19

    # Panel A: BER across 5 conditions (with 3-seed error bars)
    ax = axes[0]
    for i, (rkey, lbl, col) in enumerate(roles):
        if rkey == "B3":
            means = [gb["conditions"][c]["methods"]["B3_mixture_likelihood"]["ber"] for c in COND_ORDER]
            stds = [0.0] * len(COND_ORDER)
        else:
            means = [gc["role_aggregates"][rkey]["conditions"][c]["ber_mean"] for c in COND_ORDER]
            stds = [gc["role_aggregates"][rkey]["conditions"][c]["ber_std"] for c in COND_ORDER]
        ax.bar(
            x + (i - 1.5) * w,
            means,
            yerr=stds,
            capsize=2.5,
            width=w,
            label=lbl,
            color=col,
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(COND_LABELS, rotation=18, ha="right")
    ax.set_ylabel("Test BER (3-Seed Mean ± SD)")
    ax.set_title("(a) Group C Test BER Across 5 RF Stress Conditions")
    ax.legend(loc="upper left", fontsize=7.2, frameon=True)

    # Panel B: Calibrated Bit NLL across 5 conditions
    ax = axes[1]
    for i, (rkey, lbl, col) in enumerate(roles):
        if rkey == "B3":
            means = [gb["conditions"][c]["methods"]["B3_mixture_likelihood"]["nll_cal"] for c in COND_ORDER]
            stds = [0.0] * len(COND_ORDER)
        else:
            means = [gc["role_aggregates"][rkey]["conditions"][c]["nll_cal_mean"] for c in COND_ORDER]
            stds = [gc["role_aggregates"][rkey]["conditions"][c]["nll_cal_std"] for c in COND_ORDER]
        ax.bar(
            x + (i - 1.5) * w,
            means,
            yerr=stds,
            capsize=2.5,
            width=w,
            label=lbl,
            color=col,
            edgecolor="black",
            linewidth=0.4,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(COND_LABELS, rotation=18, ha="right")
    ax.set_ylabel("Calibrated Test Bit NLL (nats/bit)")
    ax.set_title("(b) Dev-Calibrated Test NLL (3-Seed Mean ± SD)")

    # Panel C: Paired improvement (pp BER reduction) of Role A vs Role B1 and Role B2 with 95% cluster CI (G=12)
    ax = axes[2]
    y_pos = np.arange(len(COND_ORDER))
    h_bar = 0.34
    for offset, comp_key, lbl, col in [
        (-h_bar / 2, "RoleA_vs_RoleB1_Transformer_ber", "Role A vs Role B1 (Transformer)", "#2980b9"),
        (+h_bar / 2, "RoleA_vs_RoleB2_MLP_ber", "Role A vs Role B2 (MLP)", "#c0392b"),
    ]:
        means = [100.0 * gc["block_comparisons_seed_averaged"][c][comp_key]["mean_diff"] for c in COND_ORDER]
        lo = [100.0 * gc["block_comparisons_seed_averaged"][c][comp_key]["ci95"][0] for c in COND_ORDER]
        hi = [100.0 * gc["block_comparisons_seed_averaged"][c][comp_key]["ci95"][1] for c in COND_ORDER]
        errs = np.array([[m - l for m, l in zip(means, lo)], [h - m for m, h in zip(means, hi)]])
        ax.barh(y_pos + offset, means, xerr=errs, height=h_bar, capsize=2.5, label=lbl, color=col, alpha=0.85, edgecolor="black", linewidth=0.4)
    ax.axvline(0.0, color="black", lw=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(COND_LABELS)
    ax.set_xlabel("BER Improvement by Role A (pp, + = Role A better, 95% 12-Block CI)")
    ax.set_title("(c) Paired 12-Block Cluster Advantage of Role A")
    ax.legend(loc="best", fontsize=7.5, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fig4_data_and_coherence(audit: dict, splits: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.2))

    # Panel A: Temporal split map + block SINR profile + guard bands + sync burst location
    ax = axes[0]
    for sp_name, col in [("train", "#3498db"), ("dev", "#f39c12"), ("test", "#1b7837")]:
        blks = splits[sp_name]["blocks"]
        xs = [0.5 * (b["start_tti"] + b["end_tti_exclusive"]) for b in blks]
        ys = [b["sinr_mean_db"] for b in blks]
        ws = [b["n_ttis"] * 0.88 for b in blks]
        ax.bar(xs, ys, width=ws, color=col, alpha=0.78, edgecolor="black", linewidth=0.4, label=f"{sp_name.title()} blocks")
    for gname in ("guard_train_dev", "guard_dev_test"):
        r0, r1 = splits[gname]["range"]
        ax.axvspan(r0, r1, color="#7f8c8d", alpha=0.35, hatch="///", label="Guard (20 TTIs)" if gname == "guard_train_dev" else None)
    ax.axvspan(655, 666, color="#e74c3c", alpha=0.35, label="Real OTA Sync-Loss Burst (655–665)")
    ax.set_ylim(0.0, 24.5)
    ax.set_xlabel("Measured TTI Index (0 .. 741)")
    ax.set_ylabel("Block Mean Reported SINR (dB)")
    ax.set_title("(a) Leak-Free Contiguous Temporal Block Split (Guard = 20 TTIs)")
    ax.legend(loc="upper left", fontsize=7.5, frameon=True)

    # Panel B: Temporal autocorrelation decay justifying 20-TTI guard band
    coh = audit["temporal_coherence"]
    lags = [int(k) for k in coh.keys()]
    ax = axes[1]
    ax.plot(lags, [coh[str(l)]["centered_mag_corr"] for l in lags], "o-", color="#1b7837", lw=2.0, label="Centered |H| Multipath Corr.")
    ax.plot(lags, [coh[str(l)]["sinr_corr"] for l in lags], "s--", color="#2980b9", lw=1.8, label="Reported SINR Autocorr.")
    ax.axvline(20, color="#c0392b", ls="-.", lw=1.5, label="Chosen Guard Band (20 TTIs)")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("Temporal Lag (TTIs)")
    ax.set_ylabel("Autocorrelation Coefficient")
    ax.set_title("(b) 435 MHz OTA Residual Coherence Decay vs Guard Band")
    ax.legend(loc="upper right", fontsize=8.0, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--fig-dir", required=True, type=Path)
    args = parser.parse_args()

    args.fig_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads((args.results_dir / "mapping_verification.json").read_text(encoding="utf-8"))
    splits = json.loads((args.results_dir / "splits.json").read_text(encoding="utf-8"))
    ga = json.loads((args.results_dir / "group_a_results.json").read_text(encoding="utf-8"))
    gb = json.loads((args.results_dir / "group_b_results.json").read_text(encoding="utf-8"))
    gc = json.loads((args.results_dir / "group_c_results.json").read_text(encoding="utf-8"))

    plot_fig1_group_a(ga, args.fig_dir / "fig1_group_a_theory_and_sparse_pilots.png")
    plot_fig2_group_b(gb, args.fig_dir / "fig2_group_b_stress_matrix_and_decoupling.png")
    plot_fig3_group_c(gc=gc, gb=gb, out_path=args.fig_dir / "fig3_group_c_transport_vs_baselines.png")
    plot_fig4_data_and_coherence(audit, splits, args.fig_dir / "fig4_ota_splits_and_coherence.png")
    print("Generated all 4 publication figures in", args.fig_dir)


if __name__ == "__main__":
    main()
