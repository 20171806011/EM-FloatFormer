"""Plot the fresh pilot-placement cohort from its stored cluster measurements."""
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from .study import RESULTS


def main():
    data = json.loads((RESULTS / "placement_summary.json").read_text())
    groups = data["clusters"]
    fig, (rates, gap) = plt.subplots(1, 2, figsize=(9.4, 3.9), layout="constrained",
                                     gridspec_kw={"width_ratios": [1.3, 1.]})
    for offset, method, label, color in [
        (-.18, "carry", "Same link", "#08796f"), (0., "reset", "Reset", "#4b527f"),
        (.18, "mismatch", "Other link", "#ab5e26")
    ]:
        sample = 100*np.array([[g[k][method]["bler"] for g in groups] for k in ("sparse", "split", "dense")])
        half = t.ppf(.975, 9)*sample.std(axis=1, ddof=1)/np.sqrt(10)
        rates.errorbar(np.arange(3)+offset, sample.mean(axis=1), yerr=half, fmt="o", lw=1.8,
                       markersize=6, capsize=3, label=label, color=color)
    rates.set(xticks=range(3), xticklabels=["16 at start", "8 + 8", "48 across\nthree rows"],
              title="Common-word decoding", ylabel="Block error rate (%)", ylim=(-3, 105), xlim=(-.48, 2.48))
    rates.set_xlabel("Acquisition pilots in frame B")
    rates.legend(frameon=False, loc="lower center", ncol=3, fontsize=8, handletextpad=.2,
                 columnspacing=.8)
    ni = data["noninferiority"]
    diff = 100*np.array(ni["differences"])
    gap.axvline(0, color="#87909a", lw=.8)
    gap.axvline(1, color="#9a4628", lw=1.2, ls="--")
    gap.scatter(diff, np.arange(1, 11), s=27, color="#08796f", zorder=3)
    mean, upper = 100*ni["mean_diff"], 100*ni["upper_95"]
    gap.plot([mean, upper], [0, 0], lw=2.1, color="#273449")
    gap.scatter(mean, 0, marker="D", s=33, color="#273449", zorder=4)
    gap.plot(upper, 0, marker="|", markersize=10, color="#273449")
    gap.text(1.2, 10.7, "+1 pp margin", fontsize=8, color="#9a4628")
    gap.set(title="Split / same link minus dense / reset", xlabel="BLER difference (percentage points)",
            yticks=[0, 2, 4, 6, 8, 10], yticklabels=["Mean", "2", "4", "6", "8", "10"],
            ylabel="Independent cluster", ylim=(-.7, 11.5), xlim=(-1, 14))
    for ax in (rates, gap):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e3e5e8", lw=.65)
        ax.set_axisbelow(True)
    fig.savefig(RESULTS / "placement_effect.png", dpi=170, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
