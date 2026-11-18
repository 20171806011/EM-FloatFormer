"""Render the two research figures from the published aggregate, on CPU."""
import json
import os
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory


OUT = Path(__file__).resolve().parent / "results"


def render():
    from matplotlib import pyplot as plt
    from matplotlib.lines import Line2D
    data = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    plt.rcParams.update({"font.size": 10, "axes.spines.right": False, "axes.spines.top": False})
    colors = {"pure": "#16637b", "parity": "#9a3f18", "local": "#16637b",
              "local_parity": "#9a3f18", "physics": "#639498", "physics_parity": "#c49570",
              "p5_full": "#5a5377", "ref": "#232c33"}
    legend = [("pure", "Pure wavefield", "o"), ("local", "Local, pure", "s"),
              ("parity", "Soft parity", "o"), ("local_parity", "Local, parity", "s"),
              ("physics", "Fixed wavefield", "^"), ("physics_parity", "Fixed + parity", "^"),
              ("p5_full", "P5 Unrolled EM", "P"), ("ref", "Classical reference", "*")]
    markers = {k: mark for k, _, mark in legend}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6))
    fig.subplots_adjust(left=.075, right=.98, bottom=.29, top=.84, wspace=.35)
    titles = ["R1   Blind PA and phase noise", "R2   Super-aliased Doppler", "R3   Four-antenna array"]
    for regime, ax in enumerate(axes, start=1):
        ref = "pilot_cpe_dd" if regime == 1 else "waveform_dd"
        for row in data["aggregates"]:
            name = "ref" if row["method"] == ref else row["method"]
            if row["regime"] != regime or name not in markers:
                continue
            square = name in ("local", "local_parity")
            ax.scatter(row["b64_ms"], 100 * row["bler"], marker=markers[name],
                       s=110 if name == "ref" else 61, linewidths=1.3, alpha=.93,
                       facecolor="none" if square else colors[name], edgecolor=colors[name], zorder=3)
        ax.set_title(titles[regime-1], loc="left", fontsize=11, fontweight="bold")
        ax.set_xlabel("Batch-64 receiver latency (ms)")
        if regime == 1:
            ax.set_ylabel("Decoded BLER (%)")
        ax.grid(alpha=.2)
        ax.margins(.12, .14)
    handles = [Line2D([], [], marker=mark, linestyle="none", markeredgecolor=colors[k],
                      markerfacecolor="none" if k.startswith("local") else colors[k],
                      markersize=8, label=label) for k, label, mark in legend]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5,.015), ncol=4, frameon=False)
    fig.suptitle("Receiver error rate and measured CPU time", x=.075, ha="left",
                 fontweight="bold", fontsize=15)
    fig.savefig(OUT / "latency_bler.png", dpi=175)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.1), sharey=True, layout="constrained")
    stages = ("initial", "trained", "trained_shuffled")
    for ax, arm, title in zip(axes, ("pure", "parity"), ("Pure wavefield", "Soft parity")):
        rows = [r for r in data["kernel"] if r["method"] == arm]
        for row in rows:
            ax.plot(range(3), [row[s] for s in stages], "-o", color=colors[arm], alpha=.43,
                    lw=1, markersize=4)
        ax.plot(range(3), [fmean(r[s] for r in rows) for s in stages], "-D", color="#1a2329",
                linewidth=2, markersize=6, label="Five-pair mean")
        ax.axhline(fmean(r["p5_trained"] for r in rows), color=colors["p5_full"],
                   linestyle="--", label="P5 trained mean")
        ax.axhline(0, color="gray", linewidth=.7)
        ax.set_xticks(range(3), ["Initial", "Trained", "Shuffled keys"])
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
        ax.margins(x=.1)
    axes[0].set_ylabel("Off-diagonal Pearson correlation")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("R3 attention alignment with physical covariance", fontsize=14, fontweight="bold")
    fig.savefig(OUT / "r3_kernel.png", dpi=175)
    plt.close(fig)


if __name__ == "__main__":
    with TemporaryDirectory(dir=OUT, prefix="fontcache_") as cache:
        os.environ["MPLCONFIGDIR"] = cache
        os.environ["MPLBACKEND"] = "Agg"
        render()
