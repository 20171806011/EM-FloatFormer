"""Plot empirical P8-C cohort results, including the held-out true-CSI diagnostic."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULTS = Path(__file__).parent / "results"
COLORS = {"sparse": "#007a71", "dense": "#415d91"}


def main():
    total = json.loads((RESULTS / "summary.json").read_text())["totals"]
    methods = ["carry", "reset", "mismatch", "oracle"]
    labels = ["Same-link history", "Reset", "Other-link history", "True CSI ᵃ"]
    y = np.arange(len(methods))
    fig = plt.figure(figsize=(9.6, 5.9), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[2.1, 1])
    axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    for ax, (field, scale, title, xlabel, limit, fmt) in zip(axes, [
        ("bler", 100, "A   Common payload", "LDPC block error rate (%)", 114, ".1f"),
        ("probe_nmse", 1, "B   Unseen final block", "Prediction NMSE (linear)", 1.80, ".3f"),
    ]):
        for layout, offset, marker in [("sparse", -.14, "o"), ("dense", .14, "s")]:
            values = np.array([total[layout][m][field] * scale for m in methods])
            ax.scatter(values, y + offset, marker=marker, color=COLORS[layout], s=30, zorder=3)
            for x, ypos in zip(values, y + offset):
                ax.annotate(format(x, fmt), (x, ypos), xytext=(5, 0), textcoords="offset points",
                            va="center", fontsize=8, color=COLORS[layout])
        ax.axhline(2.5, ls="--", color="#939da6", lw=.7)
        ax.set(yticks=y, yticklabels=labels if ax is axes[0] else [], xlabel=xlabel,
               xlim=(0, limit), ylim=(3.4, -.4))
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
        ax.grid(axis="x", color="#e3e6e8", lw=.7)
        ax.set_axisbelow(True)
    for layout, marker, label in [("sparse", "o", "32 training symbols"),
                                  ("dense", "s", "96 training symbols")]:
        axes[0].scatter([], [], color=COLORS[layout], marker=marker, label=label)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0., -.23), ncol=2, frameon=False,
                   fontsize=9, columnspacing=.9, handletextpad=.3)
    axes[1].text(0., -.27, "ᵃ Receiver supplied with the simulated channel.",
                 transform=axes[1].transAxes, fontsize=9, va="top", color="#444d53")

    ax = fig.add_subplot(grid[1, :])
    cases = [("sparse", "carry", "32 / same-link history"), ("dense", "reset", "96 / reset")]
    for i, (layout, method, _) in enumerate(cases):
        c = total[layout][method]
        common = c["blocks"] - c["block_errors"]
        extra = c["extra_blocks"] - c["extra_errors"]
        ax.barh(i, common, height=.52, color=COLORS[layout])
        if extra:
            ax.barh(i, extra, left=common, height=.52, color="#d8ede8", hatch="///",
                    edgecolor=COLORS[layout], linewidth=.6)
            ax.text(common+extra/2, i, f"+{extra:,}", color="#062e2b", ha="center", va="center",
                    fontsize=9, fontweight="bold", bbox=dict(facecolor="#d8ede8", edgecolor="none", pad=.5))
        ax.text(common+extra+26, i, f"{common+extra:,}", va="center", fontsize=10, fontweight="bold")
        ax.text(common/2, i, f"{common:,} common", ha="center", va="center", color="white", fontsize=9)
    ax.set(yticks=[0, 1], yticklabels=[item[2] for item in cases], ylim=(1.5, -.5),
           xlim=(0, 3000), xlabel="Successfully decoded words over 320 equal-duration frames")
    ax.set_title("C   Delivered payload including additional words", loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="x", lw=.7, color="#e3e6e8")
    ax.set_axisbelow(True)
    for panel in (*axes, ax):
        panel.spines[["top", "right"]].set_visible(False)
    fig.savefig(RESULTS / "p8c_transfer.png", dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
