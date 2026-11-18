"""Render the fourth development run's detailed tables from its JSON records."""

from p6.write_dev_iter2 import ARMS, KERNELS, NAMES, PAIRS, PARITY, REGIMES, delta, methods, num, table
from p6.write_dev_iter3 import INTERVENTIONS, percent, timing


def receiver_tables(report, runs):
    sections = ["## Stream-6 receiver comparison",
                "The pair columns count failed LDPC words. BLER pools failed and total words. "
                "Pre-BER uses hard coded bits and post-BER uses decoded information bits. "
                "NMSE divides summed channel-error energy by summed true-channel energy. "
                "BCE uses raw payload LLRs. The Genie-PA control receives the realized PA parameters."]
    for regime in REGIMES:
        pooled = report["pooled"][regime]
        rows = []
        for arm in methods(pooled):
            scores = [(runs[regime, p, arm]["score"] if arm in ARMS
                       else runs[regime, p, "references"]["scores"][arm]) for p in PAIRS]
            total = pooled[arm]
            counts = [s["block_errors"] for s in scores]
            if sum(counts) != total["block_errors"] or sum(s["blocks"] for s in scores) != total["blocks"]:
                raise ValueError(f"Pooled count disagrees with R{regime} {arm} pair receipts")
            matched = {"pure": "local", "parity": "local_parity"}.get(arm)
            difference = delta(sum(counts) - pooled[matched]["block_errors"]) if matched else "—"
            rows.append([NAMES[arm], *counts, f"{sum(counts)} / {total['blocks']}",
                         percent(total["bler"]), percent(total["pre_ber"]),
                         percent(total["info_errors"] / total["info_bits"]), f"{total['nmse_db']:.2f}",
                         num(total["bce"]), difference])
        sections.extend([f"### R{regime} at {report['config']['snr_db'][regime]:g} dB", table(
            ["Receiver", *PAIRS, "Failed / total", "BLER", "Pre-BER", "Post-BER", "NMSE (dB)",
             "Raw BCE", "Δ failures vs matched Local"], rows)])
    return sections


def learning_tables(runs):
    updates = (0, 24, 32, 48, 64)
    rows, times = [], []
    for regime in REGIMES:
        for arm in ARMS:
            fits = [runs[regime, p, arm] for p in PAIRS]
            points = []
            for update in updates:
                scores = [fit["trace"][str(update)]["score"] for fit in fits]
                failures = sum(s["block_errors"] for s in scores)
                bce = sum(s["bce_sum"] for s in scores) / sum(s["bits"] for s in scores)
                points.append(f"{failures} / {bce:.5f}")
            rows.append([f"R{regime}", NAMES[arm], *points])
            times.append([f"R{regime}", NAMES[arm], *[
                f"{fit['trace']['48']['fit_seconds']:.2f}" for fit in fits]])
    return ["## Validation along the shared learning-rate schedule",
            "Each cell gives pooled failures / raw validation BCE. Each checkpoint uses the same "
            "stream-6 validation bursts as its pair’s primary checkpoint. Update 48 ends the cosine "
            "decay. Updates 49–64 continue from that checkpoint at the terminal learning rate.",
            table(["Regime", "Receiver", *[f"Update {u}" for u in updates]], rows),
            "### Training time by pair",
            "Cumulative wall seconds through update 48 include training-frame generation and optimization. "
            "The timer pauses during validation, probing and checkpoint writes.",
            table(["Regime", "Receiver", *[f"{p} (s)" for p in PAIRS]], times)]


def intervention_tables(records):
    indexed = {(str(row["regime"]), int(row["pair"]), row["arm"]): row for row in records}
    labels = dict(INTERVENTIONS, spatial="Additive antenna/path amplitude correction off")
    sections = ["## Handle interventions on the selected checkpoints",
                "Each pair cell gives its failure count and the change from its full receiver. A positive "
                "change means suppression increased failures. All rows reuse the same weights and frames. "
                "The PA-off intervention retains the nominal PA. The forward-only row disables the "
                "bounded receive-side PA adjustment."]
    for regime in REGIMES:
        rows = []
        for arm in PARITY:
            suppressed = [indexed[regime, p, arm]["suppressed"] for p in PAIRS]
            if any(s.keys() != suppressed[0].keys() for s in suppressed):
                raise ValueError(f"Unmatched R{regime} {arm} intervention labels")
            for key in sorted(suppressed[0], key=list(labels).index):
                scores = [s[key] for s in suppressed]
                full = [s["full"] for s in suppressed]
                counts = [s["block_errors"] for s in scores]
                diffs = [s["block_errors"] - f["block_errors"] for s, f in zip(scores, full, strict=True)]
                bits = sum(s["bits"] for s in scores)
                bce_diff = sum(s["bce_sum"] - f["bce_sum"] for s, f in zip(scores, full, strict=True)) / bits
                rows.append([NAMES[arm], labels[key],
                             *[f"{n} ({delta(d)})" for n, d in zip(counts, diffs, strict=True)],
                             sum(counts), delta(sum(diffs)), f"{bce_diff:+.5f}"])
        sections.extend([f"### R{regime}", table(["Receiver", "Intervention", *PAIRS, "Pooled failures",
                                                 "Δ failures", "Δ raw BCE"], rows)])
    return sections


def chirp_tables(report):
    records = {(int(row["pair"]), row["arm"]): row["physical"]
               for row in report["interventions"] if str(row["regime"]) == "2"}
    stages, pairs, subgroups = [], [], []
    groups = {"all": "All rays", "initial_wrong_sign": "Initially wrong sign",
              "initial_correct_sign": "Initially correct sign",
              "initial_span_over_0.5_through_3": "Initial span (0.5, 3] rad",
              "initial_span_over_3": "Initial span >3 rad",
              "wrong_sign_and_initial_span_over_0.5_through_3": "Initially wrong, span (0.5, 3] rad"}
    for arm in PARITY:
        for pair in PAIRS:
            s = records[pair, arm]["sign_summary"]["all"]
            p = records[pair, arm]["phase_passes"]
            stages.append([NAMES[arm], pair, p["trajectories"],
                           *map(num, p["mean_absolute_wrapped_row_error_initial_then_steps"])])
            pairs.append([NAMES[arm], pair, s["burst_rays"], s["initial_wrong"], s["final_wrong"],
                          s["repaired"], s["regressed"], num(s["initial_aligned_phase_mae_rad"]),
                          num(s["final_aligned_phase_mae_rad"])])
        for key, name in groups.items():
            s = report["r2_signs"][arm][key]
            def transition(start, end):
                return f"{num(s[start])} → {num(s[end])}" if s[start] is not None else "—"
            subgroups.append([NAMES[arm], name, s["burst_rays"],
                              f"{s['initial_wrong']} → {s['final_wrong']}",
                              f"{s['initial_zero_slope']} → {s['final_zero_slope']}",
                              s["repaired"], s["regressed"],
                              transition("initial_net_slope_mae", "final_net_slope_mae"),
                              transition("initial_aligned_phase_mae_rad", "final_aligned_phase_mae_rad")])
    return ["## R2 chirp direction and phase error",
            "Direction uses the sign of the net slope between first and last symbol midpoints. Repairs "
            "count wrong-to-correct signs and regressions count correct-to-wrong signs. Phase MAE removes "
            "one constant angle for each burst and ray. Simulator truth enters these comparisons after inference.",
            table(["Receiver", "Pair", "Burst-rays", "Initial wrong", "Final wrong", "Repairs", "Regressions",
                   "Initial midpoint MAE (rad)", "Final midpoint MAE (rad)"], pairs),
            "Subsets overlap. Net slope is in radians per useful OFDM-symbol duration.",
            table(["Receiver", "Subset", "Burst-rays", "Wrong signs", "Zero slopes", "Repairs",
                   "Regressions", "Net slope MAE", "Midpoint phase MAE (rad)"], subgroups),
            "The per-pass comparison averages the sample-time phasor within each row before measuring "
            "wrapped phase error in radians.",
            table(["Receiver", "Pair", "Burst-rays", "Initial", "Pass 1", "Pass 2", "Pass 3"], stages)]


def kernel_tables(report):
    rows = []
    support = report["uniform_green_support"]
    control = support["parallel_joint"]
    for arm, stages in report["kernels"].items():
        for key, init in stages["initial"].items():
            trained = stages["trained"][key]
            start, end = init["overall"]["mean"], trained["overall"]["mean"]
            rows.append([NAMES[arm], KERNELS[key], num(start), num(end),
                         num(trained["shuffled"]["mean"]), num(control["mean"]) if key == "parallel_joint" else "—",
                         num(init.get("axis_only", {}).get("mean")), num(trained.get("axis_only", {}).get("mean")),
                         *[num(trained["overall"]["by_pair"][str(p)]) for p in PAIRS]])
    return ["## R3 Green-kernel correlations",
            f"Each overall Pearson coefficient uses {support['off_diagonal_entries']} off-diagonal entries. "
            f"The {support['total_bursts']} probe bursts come from stream 8. The shuffled control averages "
            "eight stream-9 key permutations. The direct joint averages the three axis transports at "
            "their physical supports. The historical product composes those transports. Axis-only "
            "correlations restrict both tokens to the same line along the measured axis.",
            table(["Receiver", "Kernel", "Initial", "Trained", "Trained shuffled", "Direct uniform",
                   "Initial axis-only", "Trained axis-only", *[f"Trained {p}" for p in PAIRS]], rows),
            "The historical uniform product is constant, so its Pearson correlation is undefined. "
            "The matched direct-joint uniform control has the following correlations.",
            table([*PAIRS, "Mean"], [[*[num(control["by_pair"][str(p)]["mean"]) for p in PAIRS],
                                      num(control["mean"])]])]


def steering_tables(report):
    rows, edges = [], []
    for arm in ("physics_parity", *PARITY):
        stages = report["spatial"][arm]
        for stage, state in stages.items():
            label = stage.replace("_", " ")
            rows.append([NAMES[arm], label, state["burst_rays"],
                         num(state["steering_cosine_before"]), num(state["steering_cosine_after"]),
                         num(state["remote_phase_cosine_before"]), num(state["remote_phase_cosine_after"]),
                         num(state["true_off_mode_rms_ratio"])])
            if stage in ("waveform_solve_3", "all_waveform_solves"):
                cols = state["remote_edges"]
                edges.append([NAMES[arm], label, *[
                    f"{num(cols[e]['before'])} → {num(cols[e]['after'])}" for e in ("02", "03", "13")]])
    return ["### R3 spatial steering on validation frames",
            "At each fit, steering cosine is the absolute normalized inner product of the fitted "
            "antenna vector and its true ULA direction. Remote phase cosine measures relative-phase "
            "agreement for the three non-adjacent antenna pairs. The off-mode ratio divides the RMS "
            "amplitude outside the true steering mode after the correction by its value before the correction. "
            "The pooled waveform row includes each pass once and excludes the pilot redemap.",
            table(["Receiver", "Stage", "Burst-ray applications", "Steering before", "Steering after",
                   "Remote phase before", "Remote phase after", "Off-mode RMS ratio"], rows),
            table(["Receiver", "Stage", "Antennas 0, 2", "Antennas 0, 3", "Antennas 1, 3"], edges)]


def render(report):
    runs = {(str(row["regime"]), int(row["pair"]), row["arm"]): row for row in report["by_pair"]}
    sections = [*receiver_tables(report, runs), *learning_tables(runs),
                *intervention_tables(report["interventions"]), *chirp_tables(report),
                *kernel_tables(report), *steering_tables(report), *timing(report)]
    return "\n\n".join(sections) + "\n"
