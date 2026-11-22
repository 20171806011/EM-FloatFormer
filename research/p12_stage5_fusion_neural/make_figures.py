"""Generate publication-grade figures for CLI-4 Stage 5 report (`research/p12_stage5_fusion_neural/figures/`)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PKG_DIR = Path(__file__).resolve().parent
RES_DIR = PKG_DIR / "results"
FIG_DIR = PKG_DIR / "figures"


def make_all_figures() -> list[Path]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    saved: list[Path] = []

    # 1) Step 1 Dev Physics Ablation
    s1_path = RES_DIR / "step1_dev_summary.json"
    if s1_path.is_file():
        s1 = json.loads(s1_path.read_text(encoding="utf-8"))
        regimes = [
            "offgrid_continuous",
            "offgrid_partial_shift",
            "offgrid_midframe_shift",
            "offgrid_full_reset",
            "offgrid_full_reset_new_delays",
        ]
        short_reg = ["Continuous", "Partial\nShift", "Midframe\nShift", "Full Reset\n(Same τ)", "True Reset\n(New τ)"]
        arms = [
            ("p12_consistent_base", "P12 Base (655de6a)", "#6c757d"),
            ("p12_hard_choice", "Hard MAP Choice", "#e07a5f"),
            ("p12_mixture_llr", "Mixture LLR (Full)", "#2a9d8f"),
            ("p12_mixture_llr_content", "Mixture LLR + Content", "#264653"),
            ("oracle_true_channel", "Oracle True H", "#adb5bd"),
        ]
        x = np.arange(len(regimes))
        w = 0.16
        fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=180)
        for i, (key, label, col) in enumerate(arms):
            vals = [s1["by_regime"][r][key]["bler_pct"] for r in regimes]
            ax.bar(x + (i - 2) * w, vals, width=w, label=label, color=col, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(short_reg)
        ax.set_ylabel("Block Error Rate (BLER %)")
        ax.set_title("Step 1 Dev Ablation: Consistent Off-Grid Posterior Fusion Mechanisms (9 dB)")
        ax.legend(frameon=False, ncol=3, loc="upper left")
        fig.tight_layout()
        p1 = FIG_DIR / "fig1_step1_physics_ablation.png"
        fig.savefig(p1)
        plt.close(fig)
        saved.append(p1)

    # 2) Step 2 Training Curves (Role A vs Role B, 3 seeds)
    s2_path = RES_DIR / "step2_training_summary.json"
    if s2_path.is_file():
        s2 = json.loads(s2_path.read_text(encoding="utf-8"))
        fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), dpi=180)
        metrics = [("total", "Composite Dev Loss"), ("mse", "Normalized Channel MSE"), ("bce", "Mixture Bit-LLR BCE")]
        for ax, (mk, title) in zip(axes, metrics):
            for role, label, col, ls in (
                ("role_a", "Role A: Symm. Transport", "#1d3557", "-"),
                ("role_b", "Role B: Non-Physics Transf.", "#e63946", "--"),
            ):
                runs = s2["runs"][role]
                curves = []
                for r in runs:
                    eps = [h["epoch"] for h in r["history"]]
                    ys = [h["dev"][mk] for h in r["history"]]
                    curves.append(ys)
                    ax.plot(eps, ys, color=col, linestyle=ls, alpha=0.35, linewidth=1.0)
                mean_y = np.mean(np.asarray(curves), axis=0)
                ax.plot(eps, mean_y, color=col, linestyle=ls, linewidth=2.2, label=label)
            ax.set_xlabel("Epoch")
            ax.set_title(title)
            ax.legend(frameon=False, fontsize=8.5)
        fig.suptitle(
            f"Step 2 Stage 5 Fair Training (3 Seeds, Exact Parity: {s2['real_parameters_role_a']} real params each)",
            y=1.02,
        )
        fig.tight_layout()
        p2 = FIG_DIR / "fig2_stage5_training_curves.png"
        fig.savefig(p2, bbox_inches="tight")
        plt.close(fig)
        saved.append(p2)

    # 3) Step 3 Formal Multi-Cluster BLER Across All 7 Regimes
    s3_path = RES_DIR / "s5_formal_audit.json"
    if s3_path.is_file():
        s3 = json.loads(s3_path.read_text(encoding="utf-8"))
        all_regs = [
            ("id", "offgrid_continuous", "Continuous"),
            ("id", "offgrid_partial_shift", "Partial\nShift"),
            ("id", "offgrid_midframe_shift", "Midframe\nShift"),
            ("id", "offgrid_full_reset", "Full Reset\n(Same τ)"),
            ("ood", "offgrid_full_reset_new_delays", "True Reset\n(New τ)"),
            ("ood", "offgrid_ood_burst_and_shift", "OOD:\nBurst+Shift"),
            ("ood", "offgrid_ood_five_ray_fast", "OOD:\n5-Ray Fast"),
        ]
        arms = [
            ("p12_consistent_base", "P12 Base (655de6a)", "#6c757d"),
            ("p12_hard_choice", "Hard MAP", "#e07a5f"),
            ("p12_mixture_llr", "Mixture LLR", "#2a9d8f"),
            ("p12_mixture_llr_content", "Mix + Content", "#264653"),
            ("stage5_role_a_ens", "Stage5 Role A (Symm PT)", "#1d3557"),
            ("stage5_role_b_ens", "Stage5 Role B (Control)", "#e63946"),
        ]
        x = np.arange(len(all_regs))
        w = 0.13
        fig, ax = plt.subplots(figsize=(12.2, 4.8), dpi=180)
        for i, (key, label, col) in enumerate(arms):
            vals = []
            for sp, reg, _ in all_regs:
                tbl = s3["id_by_regime"] if sp == "id" else s3["ood_by_regime"]
                vals.append(tbl[reg][key]["bler_pct"])
            ax.bar(x + (i - 2.5) * w, vals, width=w, label=label, color=col, edgecolor="white", linewidth=0.4)
        ax.axvline(3.5, color="#495057", linestyle=":", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([r[2] for r in all_regs])
        ax.set_ylabel("Closed-Book BLER (%)")
        ax.set_title(f"Closed-Book Formal Evaluation (G={s3['n_clusters']} Clusters, 9 dB): Physics Fusion & Stage 5 Neural")
        ax.legend(frameon=False, ncol=3, loc="upper left")
        fig.tight_layout()
        p3 = FIG_DIR / "fig3_formal_bler_comparison.png"
        fig.savefig(p3)
        plt.close(fig)
        saved.append(p3)

        # 4) Per-Cluster Paired BLER Differences (Holm Contrasts)
        contrasts = [
            ("holm_family_physics", "mix_vs_base_4id", "Mixture LLR vs Base (4-ID)"),
            ("holm_family_physics", "content_vs_base_4id", "Mix+Content vs Base (4-ID)"),
            ("holm_family_neural", "role_a_vs_phys_4id", "Role A vs Upgraded Phys (4-ID)"),
            ("holm_family_neural", "role_a_vs_role_b_4id", "Role A vs Role B (4-ID)"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.8), dpi=180)
        clusters = np.array(s3["clusters"])
        verdict_en = {"支持": "SUPPORTED", "反证": "REFUTED", "证据不足": "INCONCLUSIVE"}
        for ax, (fam, key, title) in zip(axes.ravel(), contrasts):
            st = s3[fam][key]
            diffs = np.array(st["cluster_diffs_pp"])
            cols = ["#2a9d8f" if d > 0 else "#e63946" for d in diffs]
            ax.bar(clusters, diffs, color=cols, alpha=0.85, edgecolor="white")
            ax.axhline(0.0, color="black", linewidth=0.9)
            ax.axhline(st["mean_pp"], color="#1d3557", linestyle="--", linewidth=1.3)
            v_lbl = verdict_en.get(st["verdict"], st["verdict"])
            ax.set_title(
                f"{title}\nmean={st['mean_pp']:+.2f} pp, W/T/L={st['wins']}/{st['ties']}/{st['losses']}, p_Holm={st['p_holm']:.4f} [{v_lbl}]",
                fontsize=9.2,
            )
            ax.set_xlabel("Cluster ID")
            ax.set_ylabel("BLER Improvement (pp, + better)")
            ax.set_xticks(clusters)
        fig.tight_layout()
        p4 = FIG_DIR / "fig4_cluster_paired_diffs.png"
        fig.savefig(p4)
        plt.close(fig)
        saved.append(p4)

    return saved


if __name__ == "__main__":
    paths = make_all_figures()
    print("Generated figures:", [str(p) for p in paths])
