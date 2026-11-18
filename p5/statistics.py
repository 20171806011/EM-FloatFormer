"""Aggregate the frozen Phase 5 design. Run with python -B -m p5.statistics."""
import csv
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
from statistics import fmean
import sys
from tempfile import TemporaryDirectory

RESULTS = Path(__file__).resolve().parent / "results"


COUNT_FIELDS = ("bit_errors", "bits", "info_errors", "info_bits", "block_errors", "blocks")
ENERGY_FIELDS = ("channel_error", "channel_energy")
LABELS = {
    "full": "Unrolled EM", "physics": "Pure physics", "local": "Local conv",
    "no_unroll": "One update", "no_gauge": "No gauge",
    "nominal_bussgang_lmmse": "Nom. Bussgang", "nominal_idd_pa": "Nom. IDD-PA",
    "pilot_cpe_dd": "Pilot CPE-DD", "genie_pa": "Genie PA*",
    "tap_phase_slope": "Tap-phase slope", "lmmse_2d": "2D-LMMSE",
    "waveform_dd": "Waveform-DD",
}


def _rates(row):
    return dict(ber=row["bit_errors"] / row["bits"], info_ber=row["info_errors"] / row["info_bits"],
                bler=row["block_errors"] / row["blocks"],
                nmse_db=10 * math.log10(row["channel_error"] / row["channel_energy"]))


def sign_flip(differences):
    """Two-sided randomization distribution of the paired sum, ties included."""
    if not differences:
        raise ValueError("At least one independent pair is needed")
    obs = abs(math.fsum(differences))
    tail = sum(abs(math.fsum(s*x for s, x in zip(signs, differences))) >= obs
               for signs in itertools.product((-1, 1), repeat=len(differences)))
    total = 2**len(differences)
    return tail, total, tail / total


def comparison(regime, name, differences, effect):
    tail, total, probability = sign_flip(differences)
    return dict(regime=regime, comparison=name, differences=differences, effect=effect,
                wins=sum(d > 0 for d in differences), ties=sum(d == 0 for d in differences),
                losses=sum(d < 0 for d in differences), signflip_tail=tail, signflip_total=total,
                p_raw=probability)


def holm(rows):
    """Step-down familywise adjustment with monotonicity in sorted raw p."""
    maximum = 0.
    for i, row in enumerate(sorted(rows, key=lambda item: item["p_raw"])):
        maximum = max(maximum, min(1., (len(rows)-i) * row["p_raw"]))
        row["p_holm"] = maximum


def kernel_row(pair, probes):
    values = {}
    for key in ("initial", "trained"):
        layers = probes[key]
        values[key] = fmean(fmean(layer["corr"]) for layer in layers)
        values[f"{key}_shuffled"] = fmean(
            fmean(fmean(perm) for perm in layer["shuffled_corr"]) for layer in layers)
    return dict(pair=pair, **values)


def aggregate():
    from p5.study import ARMS, METHODS, PAIRS, REFERENCES, frozen, read
    by_seed, sums, kernel, calibration, runtime = [], [], [], [], []
    families = {"primary": [], "ablation": [], "kernel": []}
    for r, n_pairs in PAIRS.items():
        snr = frozen(r)
        calibration_file = read(RESULTS / f"calibration_r{r}.json")
        validation = calibration_file["validation"]
        calibration.append(dict(regime=r, reference=REFERENCES[r], snr_db=snr,
                                block_errors=validation["block_errors"], blocks=validation["blocks"],
                                bler=validation["bler"]))
        arms = (*ARMS, "physics", *METHODS[r])
        records = [read(RESULTS / f"r{r}/pair{pair:02d}/record.json") for pair in range(1, n_pairs+1)]
        times = {(row["method"], row["batch"]): row
                 for row in read(RESULTS / f"latency_r{r}.json")}
        for pair, rec in enumerate(records, start=1):
            if ((rec["pair"], rec["regime"], rec["snr_db"]) != (pair, r, snr) or
                    set(rec["rows"]) != set(arms)):
                raise ValueError(f"Pair metadata/method mismatch in R{r} pair {pair}")
            for arm in arms:
                row = rec["rows"][arm]
                if any(type(row[k]) is not int for k in COUNT_FIELDS):
                    raise ValueError("Every count must be an exact integer")
                if not (0 <= row["info_errors"] <= row["info_bits"] and
                        0 <= row["block_errors"] <= row["blocks"] and
                        0 <= row["bit_errors"] <= row["bits"] and
                        row["bits"] == row["blocks"] * 128 and row["info_bits"] == row["blocks"] * 64):
                    raise ValueError("Inconsistent coded-bit or LDPC-block count")
                if any(rec["rows"]["full"][k] != row[k] for k in ("bits","blocks","info_bits")):
                    raise ValueError("Paired receiver denominators differ")
                by_seed.append(dict(regime=r, pair=pair, snr_db=snr, method=arm, **row, **_rates(row)))
            if r == 3:
                kernel.append(kernel_row(pair, rec["probe"]))
        runtime.append(dict(regime=r, calibration_seconds=calibration_file["seconds"],
                            sum_training_seconds=sum(rec["train_seconds"] for rec in records),
                            sum_pair_wall_seconds=sum(rec["total_seconds"] for rec in records)))
        for name in arms:
            row = {k: sum(rec["rows"][name][k] for rec in records) for k in (*COUNT_FIELDS, *ENERGY_FIELDS)}
            l1, l64 = times[(name, 1)], times[(name, 64)]
            if l1["params"] != l64["params"]:
                raise ValueError("Batch size changed parameter count")
            sums.append(dict(regime=r, method=name, snr_db=snr, pairs=n_pairs, **row, **_rates(row),
                             params=l1["params"], latency_b1_ms=l1["ms"], latency_b64_ms=l64["ms"]))
        for family, methods in (("primary", [REFERENCES[r]]),
                                ("ablation", ["physics", "local", "no_unroll", "no_gauge"])):
            for arm in methods:
                diffs = [rec["rows"][arm]["block_errors"]-rec["rows"]["full"]["block_errors"]
                         for rec in records]
                denom = sum(rec["rows"][arm]["blocks"] for rec in records)
                families[family].append(comparison(r, arm, diffs, 100 * sum(diffs) / denom))
    for control in ("initial", "trained_shuffled"):
        diffs = [row["trained"] - row[control] for row in kernel]
        families["kernel"].append(comparison(3, control, diffs, fmean(diffs)))
    for rows in families.values():
        holm(rows)
    return by_seed, dict(aggregates=sums, comparisons=families, kernel=kernel,
                         calibration=calibration, runtime=runtime)


def _figure(plt, summary):
    colors = {"full": "#135491", "ref": "#c15e16", "genie": "#72549a", "other": "#5d6974"}
    titles = {1: "Blind PA and phase noise", 2: "Super-aliased Doppler", 3: "Four-antenna wavefield"}
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.6), height_ratios=(1, 1.5), layout="constrained")
    for i, calibration in enumerate(summary["calibration"]):
        r = calibration["regime"]
        cal = json.loads((RESULTS / f"calibration_r{r}.json").read_text(encoding="utf-8"))
        up, down = axes[:, i]
        up.set_title(f"R{r}   {titles[r]}", loc="left", fontweight="bold")
        up.axhspan(5, 35, color="#dceddf", zorder=0)
        for key, label, marker, color in (("coarse","24 bursts","o","#777f86"),
                                           ("fine","64 bursts","s","#506d90")):
            grid = cal[key]
            up.plot([p["snr_db"] for p in grid], [p["bler"] * 100 for p in grid], marker=marker,
                    ms=3.4, lw=1.2, color=color, label=label)
        val = cal["validation"]
        up.plot([val["snr_db"]], [100*val["bler"]], marker="D", ms=6, color=colors["ref"],
                label="96-burst validation")
        up.set_ylim(-3,103)
        up.set_xlabel("Es/N0 (dB)")
        if i == 0:
            up.set_ylabel("Calibration BLER (%)")
            up.legend(loc="upper right", fontsize=7)
        rows = [a for a in summary["aggregates"] if a["regime"] == r]
        down.axvspan(5, 35, color="#dceddf", zorder=0)
        for j, arm in enumerate(rows):
            name = arm["method"]
            role = ("full" if name == "full" else "ref" if name == calibration["reference"] else
                    "genie" if name == "genie_pa" else "other")
            rate = 100 * arm["bler"]
            down.plot([rate], [j], marker="D" if role == "ref" else "o", ms=6.2,
                      color=colors[role], zorder=4)
            down.text(rate+2 if rate < 88 else rate-2, j, f"{rate:.1f}",
                      ha="left" if rate < 88 else "right", va="center", fontsize=8)
        down.set_yticks(range(len(rows)), [LABELS[row["method"]] for row in rows], fontsize=8)
        down.set_ylim(len(rows)-.45,-.65)
        down.set_xlim(-2,102)
        down.set_xlabel("Held-out coded BLER (%)")
        down.set_title(f"{len(rows)} receivers at {cal['snr_db']} dB", loc="left", fontsize=10)
        for ax in (up, down):
            ax.spines[["top","right"]].set_visible(False)
            ax.grid(axis="x", alpha=.17)
    fig.suptitle("Classical reference calibration and paired receiver performance", fontsize=15)
    fig.supxlabel("Green band = 5–35% BLER     •     Orange diamond = calibration reference"
                  "     •     *Privileged PA trajectory", fontsize=9)
    fig.savefig(RESULTS / "waterfall_receivers.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.0), layout="constrained")
    keys = ("initial", "trained", "trained_shuffled")
    for row in summary["kernel"]:
        ax.plot(range(3), [row[k] for k in keys], "-o", alpha=.70, lw=1.0, ms=4)
    ax.plot(range(3), [fmean(row[k] for row in summary["kernel"]) for k in keys],
            "-D", color="black", lw=2.1, ms=6, label="Across-pair mean")
    ax.axhline(0, ls=":", color="grey", lw=.9)
    ax.set_xticks(range(3), ("Initialization","Trained","Trained, shuffled keys"))
    ax.set_ylabel("Off-diagonal Pearson correlation")
    ax.set_title("Attention versus physical covariance in the 4-RX array")
    ax.spines[["top","right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(RESULTS / "green_kernel.png", dpi=160)
    plt.close(fig)


def main():
    # The host's PyTorch and Matplotlib dependencies link different OpenMP
    # runtimes. Keep the renderer in its own process with a temporary font cache.
    if sys.argv[1:] == ["--plot"]:
        with TemporaryDirectory(dir=RESULTS, prefix="mpl_") as cache:
            os.environ["MPLCONFIGDIR"] = cache
            import matplotlib
            matplotlib.use("Agg")
            from matplotlib import pyplot as plt
            summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
            _figure(plt, summary)
        return
    from p5.study import save
    rows, study = aggregate()
    with (RESULTS / "raw_by_seed.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    save(RESULTS / "summary.json", study)
    subprocess.run([sys.executable, "-B", "-m", "p5.statistics", "--plot"], check=True)
    print(f"{len(rows)} per-pair receiver records, {len(study['aggregates'])} pooled rows")


if __name__ == "__main__":
    main()
