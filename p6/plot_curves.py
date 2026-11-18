"""Render the development BCE trajectories from the completed JSON report."""
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def render():
    report = json.loads((RESULTS / "dev_iter1.json").read_text(encoding="utf-8"))
    with TemporaryDirectory(dir=RESULTS) as cache:
        os.environ["MPLCONFIGDIR"] = cache
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        fig, axes = plt.subplots(2, 3, figsize=(11.2, 5.9), sharex=True)
        styles = [
            ("pure", "Attention, pure", "#b15d0a", "-", "o"),
            ("local", "Local, pure", "#b15d0a", "--", "s"),
            ("parity", "Attention, parity", "#006b88", "-", "o"),
            ("local_parity", "Local, parity", "#006b88", "--", "s"),
            ("no_gauge", "No-gauge", "#605578", ":", "^"),
        ]
        for col, title in enumerate(("R1 · PA and phase noise", "R2 · chirp Doppler", "R3 · antenna array")):
            axes[0, col].set_title(title, fontsize=12)
            for row, (key, ylabel) in enumerate((("train_mean_bce", "Training BCE"),
                                                ("validation_bce", "Validation BCE"))):
                ax = axes[row, col]
                for arm, label, color, line, marker in styles:
                    pts = [p for p in report["curves"][f"r{col+1}/main/{arm}"] if p[key] is not None]
                    ax.plot([p["step"] for p in pts], [p[key] for p in pts], label=label,
                            color=color, ls=line, marker=marker, lw=1.6, markersize=3.4)
                ax.grid(axis="y", color="#e1e3e6", lw=.7)
                ax.axvline(64, color="#85888a", lw=.8, alpha=.65, zorder=0)
                ax.spines[["right", "top"]].set_visible(False)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                ax.set_xticks([0, 32, 64, 96, 128])
                if col == 0:
                    ax.set_ylabel(ylabel)
                if row == 1:
                    ax.set_xlabel("Adam updates")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, ncol=5, loc="upper center", bbox_to_anchor=(.5, .965),
                   frameon=False, fontsize=9)
        fig.suptitle("Development learning curves", fontsize=16, y=1)
        fig.subplots_adjust(top=.82, bottom=.11, hspace=.32, wspace=.31)
        fig.savefig(RESULTS / "dev_curves.png", dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    render()
