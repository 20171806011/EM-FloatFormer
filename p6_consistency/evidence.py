"""Descriptive evidence checks and a standard static calibration figure."""
import math

from p6_consistency.common import (CONDITIONS, METHODS, RESULTS, ROOT, read, save,
                                  sha, verify_freeze)


def main():
    signature = verify_freeze()
    summary = read(RESULTS/"summary.json")
    records = [read(RESULTS/"fresh"/f"pair{i:02d}.json") for i in range(1, 11)]
    rows = [row for record in records for row in record["rows"]]
    counters = []
    for name in METHODS[2:]:
        selected = [row for row in rows if row["method"] == name]
        moment_rows = [moment for row in selected for moment in row["diagnostics"]["moments"]]
        cavity_rows = [item for row in selected for item in row["diagnostics"]["cavity"]]
        counters.append(dict(method=name, evaluated_batches=len(selected),
            m_steps=len(moment_rows), cavity_passes=len(cavity_rows),
            amplitude_pinv_batches=sum(item["pinv_batches"] for item in moment_rows),
            computational_noise_floor_frames=sum(item["computational_noise_floor_frames"] for item in cavity_rows),
            cavity_variance_clamps=sum(item["cavity_variance_clamps"] for item in cavity_rows)))
    controls = []
    for record in records:
        grid = {(r["batch_index"], r["condition"], r["method"]): r for r in record["rows"]}
        for batch in range(4):
            for name in METHODS[2:]:
                first = grid[batch, "intact", name]
                second = grid[batch, "first_attenuated", name]
                counts_equal = all(first["score"][key] == second["score"][key] for key in
                    ("bit_errors", "bits", "info_errors", "info_bits", "block_errors", "blocks"))
                first_energy, second_energy = first["score"]["channel_error"], second["score"]["channel_error"]
                channel_relative = abs(first_energy-second_energy)/max(abs(first_energy), 1e-300)
                bce_difference = abs(first["calibration"]["final"]["bce_sum"]-
                                     second["calibration"]["final"]["bce_sum"])
                controls.append(dict(index=record["index"], batch=batch, method=name,
                    integer_counts_equal=counts_equal, relative_channel_error_difference=channel_relative,
                    bce_sum_difference=bce_difference))
    result = dict(freeze_sha256=signature, fresh_grid_cells=len(rows), numerical_counters=counters,
        attenuation_inverse_controls=controls,
        attenuation_integer_controls_passed=all(row["integer_counts_equal"] for row in controls),
        largest_attenuation_channel_error_relative_difference=max(row["relative_channel_error_difference"] for row in controls),
        largest_attenuation_bce_sum_difference=max(row["bce_sum_difference"] for row in controls),
        source_sha256=sha(ROOT/"evidence.py"),
        scope="Descriptive post-score audit, no fitted calibration or new primary endpoint")
    save(RESULTS/"mechanism_checks.json", result)
    plot(summary)
    print(result["numerical_counters"], flush=True)
    print({key: value for key, value in result.items() if key not in
           ("attenuation_inverse_controls", "numerical_counters")}, flush=True)


def plot(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup = {(row["condition"], row["method"]): row for row in summary["calibration"]}
    conditions = ("intact", "first_erased", "interior_erased", "scattered_erased")
    titles = ("Intact", "First pilot row erased", "Interior pilot row erased", "Half the pilots erased")
    styles = (("legacy_mean", "Legacy mean", "#566573"),
              ("aware_variance", "Aware variance", "#1978a5"),
              ("aware_consistent", "Aware consistent", "#b44b18"))
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8), sharex=True, sharey=True)
    for axis, condition, title in zip(axes.flat, conditions, titles):
        axis.plot([1e-4, 2], [1e-4, 2], "--", color="#aaaaaa", lw=1, label="Mean error = variance")
        for method, label, color in styles:
            bins = lookup[condition, method]["steps"][2]["bins"]
            entries = [row for row in bins if row["count"] and row["sum_predicted_variance"] > 0
                       and row["sum_squared_error"] > 0]
            xs = [row["sum_predicted_variance"]/row["count"] for row in entries]
            ys = [row["sum_squared_error"]/row["count"] for row in entries]
            axis.plot(xs, ys, "o-", ms=4.5, color=color, label=label, lw=1.6)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlim(5e-4, 1.15)
        axis.set_ylim(5e-4, 2)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(True, which="major", color="#e5e7eb", lw=.8)
    axes[1, 0].set_xlabel("Mean predicted complex symbol variance")
    axes[1, 1].set_xlabel("Mean predicted complex symbol variance")
    axes[0, 0].set_ylabel("Actual squared error of soft mean")
    axes[1, 0].set_ylabel("Actual squared error of soft mean")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, -.005))
    fig.suptitle("P6 soft-information calibration at the third physical update", fontsize=15, fontweight="bold", y=.99)
    fig.text(.5, .946, "320 reused-weight fresh bursts per condition; payload only; fixed bins; descriptive", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .075, 1, .93))
    fig.savefig(RESULTS/"calibration.png", dpi=180, bbox_inches="tight")
    fig.savefig(RESULTS/"calibration.svg", bbox_inches="tight")
    vector = RESULTS/"calibration.svg"
    vector.write_text("\n".join(line.rstrip() for line in vector.read_text().splitlines())+"\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
