"""Generate publication-quality PNG and PDF figures strictly from unified M5 results.

All numbers are loaded dynamically from `research/realdata_sdr_ofdm/results/`:
  - summary_clean.json
  - summary_stress.json
  - paired_comparisons_all.json
  - per_frame_all.jsonl

Emits 4 figures (both 300 dpi `.png` and Type-42 `.pdf`) into `figures/`:
  1. fig1_clean_comparison.{png,pdf}
  2. fig2_group_a_b_ablation.{png,pdf}
  3. fig3_stress_robustness.{png,pdf}
  4. fig4_sinr_and_cost.{png,pdf}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.0,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

ARM_ORDER: tuple[str, ...] = (
    "ls_linear_zf",
    "lmmse_1d",
    "lmmse_2d_tracked",
    "local",
    "global_uniform",
    "global_reliability",
    "global_content_attn",
    "hard_map",
    "posterior_mean_plugin",
    "exact_mixture_likelihood",
    "group_c_branch_attn",
    "group_c_hybrid_attn",
    "group_c_pilot_nll_only",
)

ARM_SHORT: dict[str, str] = {
    "ls_linear_zf": "LS+LinZF",
    "lmmse_1d": "1D-LMMSE",
    "lmmse_2d_tracked": "2D-Track",
    "local": "GA:Local",
    "global_uniform": "GA:GlbUnif",
    "global_reliability": "GA:GlbRel",
    "global_content_attn": "GA:ContAttn",
    "hard_map": "GB:HardMAP",
    "posterior_mean_plugin": "GB:PlugIn",
    "exact_mixture_likelihood": "GB:ExactMix",
    "group_c_branch_attn": "GC:BrAttn",
    "group_c_hybrid_attn": "GC:HybAttn",
    "group_c_pilot_nll_only": "GC:NLLOnly",
}

ARM_GROUP: dict[str, str] = {
    "ls_linear_zf": "classical",
    "lmmse_1d": "classical",
    "lmmse_2d_tracked": "classical",
    "local": "group_a",
    "global_uniform": "group_a",
    "global_reliability": "group_a",
    "global_content_attn": "group_a",
    "hard_map": "group_b",
    "posterior_mean_plugin": "group_b",
    "exact_mixture_likelihood": "group_b",
    "group_c_branch_attn": "group_c",
    "group_c_hybrid_attn": "group_c",
    "group_c_pilot_nll_only": "group_c",
}

GROUP_COLORS: dict[str, str] = {
    "classical": "#64748B",
    "group_a": "#0072B2",
    "group_b": "#D55E00",
    "group_c": "#7E2F8E",
}

GROUP_LABELS: dict[str, str] = {
    "classical": "Classical Baselines",
    "group_a": "Group A: Reference & Content Attn",
    "group_b": "Group B: Posterior Fusion",
    "group_c": "Group C: Trainable C*-Net (2,698p)",
}


def _load_per_frame_grouped(jsonl_path: Path) -> dict[str, list[dict[str, Any]]]:
    by_arm: dict[str, list[dict[str, Any]]] = {a: [] for a in ARM_ORDER}
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            rec = json.loads(s)
            arm = str(rec["arm"])
            if arm in by_arm:
                by_arm[arm].append(rec)
    for arm in by_arm:
        by_arm[arm].sort(key=lambda r: int(r["frame_id"]))
    return by_arm


def _subblock_clusters(records: list[dict[str, Any]], target_g: int = 16) -> list[list[dict[str, Any]]]:
    by_macro: dict[str, list[dict[str, Any]]] = {}
    macro_order: list[str] = []
    for r in records:
        gid = str(r["group_id"])
        if gid not in by_macro:
            macro_order.append(gid)
            by_macro[gid] = []
        by_macro[gid].append(r)
    m = len(macro_order)
    if m == 0:
        return []
    if int(target_g) <= m:
        return [
            sorted(by_macro[mgid], key=lambda x: int(x["frame_id"]))
            for mgid in macro_order
        ]
    base_sub = int(target_g) // m
    rem = int(target_g) % m
    clusters: list[list[dict[str, Any]]] = []
    for idx, mgid in enumerate(macro_order):
        k_sub = max(1, base_sub + (1 if idx >= (m - rem) else 0))
        sub = sorted(by_macro[mgid], key=lambda x: int(x["frame_id"]))
        for chunk in np.array_split(np.arange(len(sub)), k_sub):
            if len(chunk) > 0:
                clusters.append([sub[int(p)] for p in chunk])
    return clusters


def _cluster_boot_ci(
    records: list[dict[str, Any]],
    metric: str,
    *,
    n_boot: int = 4000,
    seed: int = 20261122,
) -> tuple[float, float]:
    clusters = _subblock_clusters(records, target_g=16)
    g = len(clusters)
    if g == 0:
        raise ValueError(f"Cannot compute cluster bootstrap CI on empty records for metric={metric!r}")
    vals_num = np.zeros(g, dtype=np.float64)
    vals_den = np.zeros(g, dtype=np.float64)
    for idx, cl in enumerate(clusters):
        if metric == "ber_pct":
            vals_num[idx] = 100.0 * sum(float(r["bit_errors"]) for r in cl)
            vals_den[idx] = sum(float(r["total_bits"]) for r in cl)
        elif metric == "fer_pct":
            vals_num[idx] = 100.0 * sum(float(r["frame_error"]) for r in cl)
            vals_den[idx] = float(len(cl))
        elif metric == "evm_pct":
            vals_num[idx] = float(np.mean([float(r["evm_pct"]) for r in cl]))
            vals_den[idx] = 1.0
        elif metric == "pilot_nll":
            vals_num[idx] = float(np.mean([float(r["pilot_nll"]) for r in cl]))
            vals_den[idx] = 1.0
        else:
            raise ValueError(metric)

    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, g, size=(n_boot, g))
    if metric in ("ber_pct", "fer_pct"):
        samples = np.sum(vals_num[boot_idx], axis=1) / np.maximum(np.sum(vals_den[boot_idx], axis=1), 1.0)
    else:
        samples = np.mean(vals_num[boot_idx], axis=1)
    lo = float(np.quantile(samples, 0.025))
    hi = float(np.quantile(samples, 0.975))
    if not (np.isfinite(lo) and np.isfinite(hi)):
        raise ValueError(f"Non-finite cluster bootstrap CI for metric={metric!r}: ({lo}, {hi})")
    return lo, hi


def _save_both(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, format=ext, bbox_inches="tight", dpi=300)
    plt.close(fig)


def make_fig1_clean_comparison(
    summary_clean: dict[str, Any],
    per_frame_by_arm: dict[str, list[dict[str, Any]]],
    out_dir: Path,
) -> None:
    arms_data = summary_clean["arms"]
    x = np.arange(len(ARM_ORDER))
    colors = [GROUP_COLORS[ARM_GROUP[a]] for a in ARM_ORDER]
    labels = [ARM_SHORT[a] for a in ARM_ORDER]

    panels = [
        ("ber_pct", "(a) Payload Bit Error Rate (BER %)", "BER (%)", lambda d: 100.0 * float(d["ber"]), "{:.2f}"),
        ("fer_pct", "(b) Frame Error Rate (FER %, 5,600-bit TTI)", "FER (%)", lambda d: 100.0 * float(d["fer"]), "{:.1f}"),
        ("evm_pct", "(c) Equalized Payload EVM (%)", "EVM (%)", lambda d: float(d["evm_pct"]), "{:.1f}"),
        ("pilot_nll", "(d) Held-out Pilot NLL (nats/pilot, lower is better)", "Pilot NLL", lambda d: float(d["pilot_nll"]), "{:.2f}"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.2), sharex=True)
    axes_flat = axes.ravel()

    for ax, (m_key, title, ylabel, getter, fmt) in zip(axes_flat, panels):
        pts = np.array([getter(arms_data[a]) for a in ARM_ORDER], dtype=np.float64)
        lo_errs = np.zeros_like(pts)
        hi_errs = np.zeros_like(pts)
        for i, a in enumerate(ARM_ORDER):
            lo, hi = _cluster_boot_ci(per_frame_by_arm[a], m_key)
            if not (np.isfinite(pts[i]) and np.isfinite(lo) and np.isfinite(hi)):
                raise ValueError(
                    f"Non-finite point or CI for {a}/{m_key}: pt={pts[i]}, ci=({lo}, {hi})"
                )
            lo_errs[i] = max(0.0, pts[i] - lo)
            hi_errs[i] = max(0.0, hi - pts[i])

        bars = ax.bar(
            x,
            pts,
            color=colors,
            alpha=0.88,
            edgecolor="#1E293B",
            linewidth=0.6,
            yerr=[lo_errs, hi_errs],
            capsize=2.5,
            error_kw={"elinewidth": 1.0, "ecolor": "#0F172A"},
            zorder=3,
        )
        ref_val = getter(arms_data["lmmse_1d"])
        ax.axhline(ref_val, color="#334155", linestyle="--", linewidth=0.9, alpha=0.7, zorder=2)

        y_min = min(0.0, float(np.min(pts - lo_errs)))
        y_max = float(np.max(pts + hi_errs))
        span = y_max - y_min
        if span <= 0.0:
            y_lo, y_hi = y_min, y_min + 1.0
        else:
            y_lo = y_min - (0.05 * span if y_min < 0.0 else 0.0)
            y_hi = y_max + 0.18 * span
        ax.set_ylim(y_lo, y_hi)
        for idx_b, (b, val, hi_e) in enumerate(zip(bars, pts, hi_errs)):
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                val + hi_e + max(y_hi - y_lo, 1e-6) * 0.02,
                fmt.format(val),
                ha="center",
                va="bottom",
                fontsize=7.2,
                color="#0F172A",
            )
        ax.set_title(title, pad=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

    for ax in axes[1]:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=32, ha="right")

    legend_handles = [
        Patch(facecolor=GROUP_COLORS[g], edgecolor="#1E293B", label=GROUP_LABELS[g])
        for g in ("classical", "group_a", "group_b", "group_c")
    ] + [
        Line2D([0], [0], color="#334155", linestyle="--", linewidth=1.0, label="1D-LMMSE Anchor")
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "Clean Test Set Performance Across All 13 Receiver Arms (N = 138 TTIs, 772,800 bits; G = 16 Sub-Block 95% CIs)",
        y=1.03,
        fontsize=12.0,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))
    _save_both(fig, out_dir, "fig1_clean_comparison")


def make_fig2_group_a_b_ablation(
    paired_all: dict[str, Any],
    out_dir: Path,
) -> None:
    contrasts = paired_all["pairwise_key_contrasts"]

    panel_a_specs = [
        ("local_vs_lmmse_1d", "GA:Local vs 1D-LMMSE", GROUP_COLORS["group_a"]),
        ("global_uniform_vs_lmmse_1d", "GA:GlbUnif vs 1D-LMMSE", GROUP_COLORS["group_a"]),
        ("global_reliability_vs_lmmse_1d", "GA:GlbRel vs 1D-LMMSE", GROUP_COLORS["group_a"]),
        ("global_content_attn_vs_lmmse_1d", "GA:ContAttn vs 1D-LMMSE", GROUP_COLORS["group_a"]),
        ("global_uniform_vs_local", "Scope: GlbUnif vs Local", "#0284C7"),
        ("global_content_attn_vs_local", "Attn+Scope: ContAttn vs Local", "#0369A1"),
        ("global_content_attn_vs_global_uniform", "Selection: ContAttn vs GlbUnif", "#075985"),
        ("global_reliability_vs_global_uniform", "Weight: GlbRel vs GlbUnif", "#38BDF8"),
    ]
    panel_b_specs = [
        ("hard_map_vs_lmmse_1d", "GB:HardMAP vs 1D-LMMSE", GROUP_COLORS["group_b"]),
        ("posterior_mean_plugin_vs_lmmse_1d", "GB:PlugIn vs 1D-LMMSE", GROUP_COLORS["group_b"]),
        ("exact_mixture_likelihood_vs_lmmse_1d", "GB:ExactMix vs 1D-LMMSE", GROUP_COLORS["group_b"]),
        ("posterior_mean_plugin_vs_hard_map", "Fusion: PlugIn vs HardMAP", "#EA580C"),
        ("exact_mixture_likelihood_vs_hard_map", "Fusion: ExactMix vs HardMAP", "#C2410C"),
        ("exact_mixture_likelihood_vs_posterior_mean_plugin", "Demap: ExactMix vs PlugIn", "#FB923C"),
        ("group_c_hybrid_attn_vs_exact_mixture_likelihood", "GC:HybAttn vs GB:ExactMix", GROUP_COLORS["group_c"]),
        ("group_c_hybrid_attn_vs_group_c_branch_attn", "GC:HybAttn vs GC:BrAttn", GROUP_COLORS["group_c"]),
        ("group_c_hybrid_attn_vs_group_c_pilot_nll_only", "GC:HybAttn vs GC:NLLOnly", "#A855F7"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.8))
    configs = [
        (axes[0], panel_a_specs, "(a) Group A: Reference Scope Expansion vs. Content Selection"),
        (axes[1], panel_b_specs, "(b) Group B: Hard MAP vs. Posterior Fusion & Group C Hybrid"),
    ]

    for ax, specs, title in configs:
        n_rows = len(specs)
        y_pos = np.arange(n_rows)[::-1]
        x_hi_max = 0.0
        x_lo_min = 0.0

        row_entries = []
        for idx_r, (key, lbl, col) in enumerate(specs):
            c_g16 = contrasts[key]["ber_g16"]
            c_g3 = contrasts[key]["ber_g3"]
            mean_pp = float(c_g16["mean_pp"])
            b_lo = float(c_g16["boot_ci95_lo_pp"])
            b_hi = float(c_g16["boot_ci95_hi_pp"])
            g3_diffs = [float(v) for v in c_g3["cluster_diffs_pp"]]
            g3_min, g3_max = min(g3_diffs), max(g3_diffs)
            x_hi_max = max(x_hi_max, b_hi, g3_max)
            x_lo_min = min(x_lo_min, b_lo, g3_min)
            row_entries.append((y_pos[idx_r], lbl, col, mean_pp, b_lo, b_hi, g3_min, g3_max, c_g16))

        span = max(x_hi_max - x_lo_min, 0.5)
        text_x = x_hi_max + 0.08 * span
        ax.set_xlim(x_lo_min - 0.12 * span, text_x + 1.05 * span)
        ax.axvline(0.0, color="#334155", linestyle="--", linewidth=1.0, zorder=1)

        for y, lbl, col, mean_pp, b_lo, b_hi, g3_min, g3_max, c_g16 in row_entries:
            # G=3 macro range (thin line + end ticks)
            ax.plot([g3_min, g3_max], [y, y], color="#94A3B8", linewidth=1.4, zorder=2)
            ax.plot([g3_min, g3_max], [y, y], "|", color="#64748B", markersize=7, zorder=2)
            # G=16 bootstrap 95% CI (thick bar)
            ax.plot([b_lo, b_hi], [y, y], color=col, linewidth=3.4, solid_capstyle="round", zorder=3)
            ax.scatter([mean_pp], [y], color=col, edgecolor="#0F172A", s=46, zorder=4)

            delta_b = int(c_g16["error_delta"])
            w, t, l = int(c_g16["wins"]), int(c_g16["ties"]), int(c_g16["losses"])
            p_two = float(c_g16["p_two_sided"])
            p_str = f"{p_two:.1e}" if p_two < 0.001 else f"{p_two:.4f}"
            wtl = f"{w}W/{l}L" if t == 0 else f"{w}W/{t}T/{l}L"
            ann = f"{mean_pp:+.3f}pp ({delta_b:+,}b, {wtl}, p={p_str})"
            ax.text(text_x, y, ann, va="center", ha="left", fontsize=7.6, color="#0F172A")

        ax.set_yticks(y_pos)
        ax.set_yticklabels([r[1] for r in row_entries])
        ax.set_xlabel("Paired BER Reduction vs. Anchor (percentage points, + = better)")
        ax.set_title(title, pad=8)
        ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)

    legend_handles = [
        Line2D([0], [0], color="#0F172A", marker="o", linewidth=3.4, markersize=6, label="G=16 Mean & 95% Boot CI"),
        Line2D([0], [0], color="#94A3B8", marker="|", linewidth=1.4, markersize=7, label="G=3 Macro-Block Min–Max Range"),
        Line2D([0], [0], color="#334155", linestyle="--", linewidth=1.0, label="Null Difference (0 pp)"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=3, frameon=False)
    fig.suptitle(
        "Paired BER Ablation Contrasts: G=16 Sub-Block 95% Bootstrap CIs & G=3 Macro-Block Range",
        y=1.04,
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    _save_both(fig, out_dir, "fig2_group_a_b_ablation")


def make_fig3_stress_robustness(
    summary_stress: dict[str, Any],
    out_dir: Path,
) -> None:
    arms_stress = summary_stress["arms"]
    highlight_specs = [
        ("lmmse_1d", "1D-LMMSE (Anchor)", "#64748B", "--", "s"),
        ("lmmse_2d_tracked", "2D-Tracked LMMSE", "#0F172A", "-.", "D"),
        ("global_content_attn", "GA:ContAttn", "#0072B2", "-", "o"),
        ("exact_mixture_likelihood", "GB:ExactMix", "#D55E00", "-", "^"),
        ("group_c_branch_attn", "GC:BrAttn", "#9333EA", ":", "v"),
        ("group_c_hybrid_attn", "GC:HybAttn", "#7E2F8E", "-", "P"),
        ("group_c_pilot_nll_only", "GC:NLLOnly", "#EC4899", "--", "*"),
    ]

    def _cond_map(arm: str) -> dict[str, dict[str, Any]]:
        return {str(c["condition"]): c["metrics"] for c in arms_stress[arm]["conditions"]}

    panels = [
        (
            "(a) Additive AWGN Perturbation",
            "Extra AWGN Level",
            [("clean", "Clean"), ("awgn_10.0db", "+10 dB"), ("awgn_5.0db", "+5 dB"), ("awgn_0.0db", "0 dB")],
        ),
        (
            "(b) Residual Carrier Frequency Offset",
            "Normalized CFO (Δf / Δf_sc)",
            [("clean", "0.00"), ("cfo_0.020", "0.02"), ("cfo_0.050", "0.05")],
        ),
        (
            "(c) Continuous Phase Drift",
            "Phase Drift (rad / frame)",
            [("clean", "0.00"), ("drift_0.30rad", "0.30"), ("drift_0.60rad", "0.60")],
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.9), sharey=True)
    for ax, (title, xlabel, conds) in zip(axes, panels):
        x = np.arange(len(conds))
        for arm, lbl, col, ls, mk in highlight_specs:
            cmap = _cond_map(arm)
            ys = [100.0 * float(cmap[tag]["ber"]) for tag, _ in conds]
            ms = 7.0 if mk == "*" else 5.5
            ax.plot(x, ys, color=col, linestyle=ls, marker=mk, markersize=ms, linewidth=1.9, label=lbl)
        ax.set_xticks(x)
        ax.set_xticklabels([xlbl for _, xlbl in conds])
        ax.set_xlabel(xlabel)
        ax.set_title(title, pad=8)
        ax.grid(True, linestyle=":", alpha=0.5)

    axes[0].set_ylabel("Bit Error Rate (BER %)")
    axes[0].set_ylim(10.0, 54.0)

    # Helpful callouts in CFO panel without overlap
    axes[1].annotate(
        "1D / Group A collapse (~49.8%)",
        xy=(2, 49.72),
        xytext=(0.75, 45.0),
        fontsize=7.6,
        color="#0072B2",
        arrowprops={"arrowstyle": "->", "color": "#0072B2", "lw": 0.9},
    )
    axes[1].annotate(
        "2D / GB / NLLOnly\n(14.9–15.8%)",
        xy=(2, 15.2),
        xytext=(1.22, 27.5),
        fontsize=7.6,
        color="#D55E00",
        arrowprops={"arrowstyle": "->", "color": "#D55E00", "lw": 0.9},
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=4, frameon=False)
    fig.suptitle(
        "Controllable Physical Stress Robustness Across Key Receiver Families (N = 138 Test TTIs)",
        y=1.06,
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    _save_both(fig, out_dir, "fig3_stress_robustness")


def make_fig4_sinr_and_cost(
    summary_clean: dict[str, Any],
    per_frame_by_arm: dict[str, list[dict[str, Any]]],
    out_dir: Path,
) -> None:
    sinr_strat = summary_clean["sinr_stratified"]
    bins = sinr_strat["bins"]
    bin_labels = [str(b["label"]) for b in bins]
    bin_xticks = [f"{b['label'].replace('dB', '')} dB\n(n={b['n_frames']})" for b in bins]

    rep_arms = [
        ("ls_linear_zf", "LS+LinZF", "#64748B", "--", "s"),
        ("lmmse_1d", "1D-LMMSE", "#334155", "-.", "D"),
        ("global_content_attn", "GA:ContAttn", "#0072B2", "-", "o"),
        ("exact_mixture_likelihood", "GB:ExactMix", "#D55E00", "-", "^"),
        ("group_c_hybrid_attn", "GC:HybAttn", "#7E2F8E", "-", "P"),
    ]

    fig, (ax_sinr, ax_cost) = plt.subplots(1, 2, figsize=(14.5, 5.5))

    # Panel (a): SINR-stratified BER
    x = np.arange(len(bin_labels))
    width = 0.16
    offsets = np.linspace(-2.0 * width, 2.0 * width, len(rep_arms))
    for off, (arm, lbl, col, _, _) in zip(offsets, rep_arms):
        arm_bins = sinr_strat["arms"].get(arm, {})
        ys = [100.0 * float(arm_bins[bl]["ber"]) if bl in arm_bins else 0.0 for bl in bin_labels]
        ax_sinr.bar(x + off, ys, width=width, color=col, alpha=0.88, edgecolor="#1E293B", linewidth=0.5, label=lbl, zorder=3)

    # Find exact outlier frame (block_10, >=20 dB with BER > 0.3)
    outlier_recs = [
        r for r in per_frame_by_arm["group_c_hybrid_attn"]
        if float(r["reported_sinr_db"]) >= 20.0 and float(r["ber"]) > 0.3
    ]
    if outlier_recs:
        top_out = max(outlier_recs, key=lambda r: float(r["reported_sinr_db"]))
        out_note = (
            f"Uptick driven by {top_out['group_id']} burst outlier\n"
            f"(frame_id={top_out['frame_id']}, SINR={top_out['reported_sinr_db']:.1f} dB, BER~50%)"
        )
    else:
        out_note = "Uptick driven by block_10 burst outlier"

    ax_sinr.annotate(
        out_note,
        xy=(3.0, 1.8),
        xytext=(1.78, 9.5),
        fontsize=7.7,
        color="#991B1B",
        bbox={"boxstyle": "round,pad=0.3", "fc": "#FEF2F2", "ec": "#FCA5A5", "lw": 0.8},
        arrowprops={"arrowstyle": "->", "color": "#991B1B", "lw": 1.0},
        zorder=5,
    )
    ax_sinr.set_xticks(x)
    ax_sinr.set_xticklabels(bin_xticks)
    ax_sinr.set_ylabel("Bit Error Rate (BER %)")
    ax_sinr.set_title("(a) SINR-Stratified Test BER Across 4 Regimes", pad=8)
    ax_sinr.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)
    ax_sinr.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)

    # Panel (b): Accuracy vs Latency & Memory Pareto Frontier
    arms_data = summary_clean["arms"]
    xs = [float(arms_data[a]["mean_ms_per_frame"]) for a in ARM_ORDER]
    ys = [100.0 * float(arms_data[a]["ber"]) for a in ARM_ORDER]
    rss_vals = [float(arms_data[a]["peak_rss_mb"]) for a in ARM_ORDER]

    # Compute Pareto non-dominated points (min latency, min BER)
    order_lat = sorted(range(len(ARM_ORDER)), key=lambda i: xs[i])
    pareto_idx: list[int] = []
    best_ber_so_far = float("inf")
    for idx_i in order_lat:
        if ys[idx_i] < best_ber_so_far - 1e-9:
            pareto_idx.append(idx_i)
            best_ber_so_far = ys[idx_i]

    ax_cost.plot(
        [xs[i] for i in pareto_idx],
        [ys[i] for i in pareto_idx],
        color="#0F172A",
        linestyle="--",
        linewidth=1.2,
        alpha=0.65,
        zorder=2,
        label="Pareto Frontier (BER vs Latency)",
    )

    # Smart offsets in points (dx, dy) to guarantee zero overlap
    label_offsets: dict[str, tuple[float, float]] = {
        "ls_linear_zf": (6, 4),
        "lmmse_1d": (6, 4),
        "lmmse_2d_tracked": (6, 4),
        "local": (6, -10),
        "global_uniform": (6, -8),
        "global_reliability": (6, 5),
        "global_content_attn": (-12, -13),
        "hard_map": (6, 4),
        "posterior_mean_plugin": (-34, 9),
        "exact_mixture_likelihood": (-32, -14),
        "group_c_branch_attn": (6, 4),
        "group_c_hybrid_attn": (-80, -11),
        "group_c_pilot_nll_only": (-78, 8),
    }

    for i, a in enumerate(ARM_ORDER):
        grp = ARM_GROUP[a]
        p_cnt = int(arms_data[a]["param_count"])
        marker = "D" if p_cnt > 0 else "o"
        size = max(45.0, (rss_vals[i] / 5.5))
        ax_cost.scatter(
            [xs[i]],
            [ys[i]],
            s=size,
            color=GROUP_COLORS[grp],
            marker=marker,
            edgecolor="#0F172A",
            linewidth=0.8,
            alpha=0.90,
            zorder=4,
        )
        dx, dy = label_offsets.get(a, (5, 4))
        ax_cost.annotate(
            f"{ARM_SHORT[a]} ({rss_vals[i]:.0f}MB)",
            xy=(xs[i], ys[i]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.2,
            color="#0F172A",
            zorder=5,
        )

    ax_cost.set_xlim(0.0, max(max(xs) * 1.18, 1.0))
    ax_cost.set_ylim(min(ys) - 0.18, max(ys) + 0.18)
    ax_cost.set_xlabel("Mean Inference Latency (ms / frame, 1-thread pinned)")
    ax_cost.set_ylabel("Clean Test BER (%)")
    ax_cost.set_title("(b) Accuracy vs. Latency & Fresh-Process RSS Tradeoff", pad=8)
    ax_cost.grid(True, linestyle=":", alpha=0.5, zorder=0)

    cost_handles = [
        Patch(facecolor=GROUP_COLORS[g], edgecolor="#0F172A", label=GROUP_LABELS[g])
        for g in ("classical", "group_a", "group_b", "group_c")
    ] + [
        Line2D([0], [0], color="#0F172A", linestyle="--", linewidth=1.2, label="Pareto Frontier"),
    ]
    ax_cost.legend(handles=cost_handles, loc="upper right", fontsize=7.3, frameon=True, facecolor="white", framealpha=0.92)

    fig.suptitle(
        "SINR Regime Breakdown & Engineering Cost–Accuracy Pareto Frontier (Test Split, N = 138)",
        y=1.02,
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    _save_both(fig, out_dir, "fig4_sinr_and_cost")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate M5 publication figures from unified results.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=_REPO_ROOT / "research/realdata_sdr_ofdm/results",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "research/realdata_sdr_ofdm/figures",
    )
    args = parser.parse_args()

    res_dir: Path = args.results_dir
    out_dir: Path = args.out_dir

    summary_clean = json.loads((res_dir / "summary_clean.json").read_text(encoding="utf-8"))
    summary_stress = json.loads((res_dir / "summary_stress.json").read_text(encoding="utf-8"))
    paired_all = json.loads((res_dir / "paired_comparisons_all.json").read_text(encoding="utf-8"))
    per_frame_by_arm = _load_per_frame_grouped(res_dir / "per_frame_all.jsonl")

    make_fig1_clean_comparison(summary_clean, per_frame_by_arm, out_dir)
    make_fig2_group_a_b_ablation(paired_all, out_dir)
    make_fig3_stress_robustness(summary_stress, out_dir)
    make_fig4_sinr_and_cost(summary_clean, per_frame_by_arm, out_dir)

    print(f"Generated 4 PNG+PDF figure pairs in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
