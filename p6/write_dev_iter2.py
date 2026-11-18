"""Render the completed dev study from its JSON receipts. No model is imported."""

import json
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent
PAIRS = (901, 902, 903)
REGIMES = ("1", "2", "3")
ARMS = ("pure", "local", "parity", "local_parity", "no_gauge")
PARITY = ("parity", "local_parity")
NAMES = {
    "physics": "Fixed", "physics_parity": "Fixed + parity",
    "pure": "Axial pure", "local": "Local pure", "parity": "Axial parity",
    "local_parity": "Local parity", "no_gauge": "Axial no-gauge",
    "nominal_bussgang_lmmse": "Nominal Bussgang LMMSE", "nominal_idd_pa": "Nominal IDD PA",
    "pilot_cpe_dd": "Pilot-CPE-DD", "genie_pa": "Genie-PA (privileged)",
    "tap_phase_slope": "Tap phase slope", "lmmse_2d": "2D-LMMSE", "waveform_dd": "Waveform-DD",
    "depth2/parity": "Axial parity, 2 steps", "depth2/local_parity": "Local parity, 2 steps",
}
HANDLE_NAMES = {
    "radial": "PA waveform correction", "cpe": "Common phase", "radial_and_cpe": "PA + common phase",
    "phase": "Per-ray trajectory phase", "delays": "Fractional delay",
    "spatial_operator": "Whole antenna branch", "spatial": "Dedicated antenna CFR correction",
    "delta_h": "General CFR residual", "all": "All handles",
}
KERNELS = {
    "parallel_joint": "Direct parallel joint", "joint": "Historical product",
    "spatial": "Spatial", "time": "Time", "frequency": "Frequency",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def table(headers, rows):
    def line(cells):
        return "| " + " | ".join(map(str, cells)) + " |"
    return "\n".join([line(headers), line(["---"] * len(headers)), *(line(row) for row in rows)])


def delta(value):
    return "0" if value == 0 else f"{value:+d}"


def num(value):
    return "—" if value is None else f"{value:.5f}"


def methods(rows):
    return list(dict.fromkeys(m for m in ("physics", "physics_parity", *ARMS, *rows) if m in rows))


def checkpoint_metrics(records, regime, arm, updates):
    traces = [records[(3, int(regime), p, arm)]["trace"] for p in PAIRS]
    scores = []
    for column, update in enumerate(updates):
        points = [t[column] for t in traces]
        if any(point["update"] != update for point in points):
            raise ValueError("Mismatched checkpoint in learning curve")
        vals = [point["score"] for point in points]
        errors = sum(v["block_errors"] for v in vals)
        bce = sum(v["bce_sum"] for v in vals) / sum(v["bits"] for v in vals)
        scores.append((errors, bce))
    return scores


def decision(out, study, records):
    data = study["pooled"]["3"]
    out.extend(["## Development decision",
                "**Stop before protocol freeze.** Three-step attention with parity improved on fixed "
                "physics with parity in every regime. It tied matched local parity in R1, reduced "
                "failures in R2 and had one additional failure in R3. Several physical and decoding "
                "targets remain open on these three development pairs."])
    rows = []
    for r in REGIMES:
        fixed, attention, local = [data[r][arm]["block_errors"] for arm in ("physics_parity", *PARITY)]
        rows.append([f"R{r}", data[r]["parity"]["blocks"], fixed, attention, local,
                     delta(attention-fixed), delta(attention-local)])
    out.append(table(["Regime", "Total words", "Fixed + parity failures", "Axial parity",
                      "Local parity", "Axial − fixed", "Axial − local"], rows))

    changes = {r: study["suppression"][r]["parity"]["removed_minus_full"] for r in REGIMES}
    r1 = data["1"]["parity"]["block_errors"]
    r2 = data["2"]["parity"]["block_errors"]
    r3 = data["3"]["parity"]["block_errors"]
    out.append(f"**R1.** On the trained attention arm, suppressing the PA correction moved the failure "
               f"count from {r1} to {r1+changes['1']['radial']['block_errors']}. Suppressing common "
               f"phase alone gave {r1+changes['1']['cpe']['block_errors']} and suppressing both gave "
               f"{r1+changes['1']['radial_and_cpe']['block_errors']}. These interventions interact. "
               f"The user's three-step joint true-PHN/true-PA run recorded 19 failures, leaving a "
               f"{r1-19}-word gap for the fully enabled receiver.")
    residuals = [records[(3, 2, p, "parity")]["trajectory_radians"][-1] for p in PAIRS]
    max_correction = max(s["max_abs_accumulated"] for s in residuals)
    over = sum(s["trajectories_over_half_rad"] for s in residuals)
    total = sum(s["trajectories"] for s in residuals)
    out.append(f"**R2.** Suppressing the per-ray phase update moved attention from {r2} to "
               f"{r2+changes['2']['phase']['block_errors']} failures. Suppressing the general CFR "
               f"residual gave {r2+changes['2']['delta_h']['block_errors']}. The phase formula allows "
               f"unbounded accumulation. The largest final absolute correction was {max_correction:.3f} rad "
               f"and {over}/{total} burst–ray trajectories spanned more than 0.5 rad across symbols. "
               f"The supplied three-step "
               f"nearest-bank and true-step-1 diagnostics recorded 362 and 329 failures. The attention "
               f"receiver finished {r2-362} and {r2-329} words above them, respectively. Those oracle "
               f"figures and the R1 oracle come from the user's earlier measurements.")
    kernel = study["kernels"]["parity"]
    init = kernel["initial"]["spatial"]["overall"]["mean"]
    end = kernel["trained"]["spatial"]["overall"]["mean"]
    direct = kernel["trained"]["parallel_joint"]["overall"]["mean"]
    historical = kernel["trained"]["joint"]["overall"]["mean"]
    out.append(f"**R3.** Suppressing fractional delays moved attention from {r3} to "
               f"{r3+changes['3']['delays']['block_errors']} failures. Suppressing the entire antenna "
               f"branch gave {r3+changes['3']['spatial_operator']['block_errors']}. Trained spatial "
               f"Pearson was {end:.5f}, slightly below its {init:.5f} initialization. The trained direct "
               f"joint ({direct:.5f}), historical product ({historical:.5f}) and spatial readouts "
               f"all exceeded 0.084. The strong initialization and the branch suppression results limit "
               f"what the correlations establish about learning and decoded benefit.")
    traces = [checkpoint_metrics(records, "2", arm, study["config"]["checkpoints"])[-2:] for arm in PARITY]
    (a0,b0),(a1,b1) = traces[0]
    (l0,c0),(l1,c1) = traces[1]
    (r30,d0),(r31,d1) = checkpoint_metrics(records, "3", "parity", study["config"]["checkpoints"])[-2:]
    out.append(f"**BCE and decoding.** Between updates 24 and 48 in R2, attention BCE fell from "
               f"{b0:.5f} to {b1:.5f} and failed words moved from {a0} to {a1}. Local BCE also fell "
               f"({c0:.5f} to {c1:.5f}) while its failed words rose from {l0} to {l1}. R3 attention "
               f"BCE fell from {d0:.5f} to {d1:.5f} while failures rose from {r30} to {r31}. These "
               f"divergences persisted after removing free per-resource-element variance. Every primary "
               f"comparison uses the scheduled final update.")


def primary(out, study, depth):
    out.extend(["## Three-step receiver comparison",
                "Each regime pools 96 validation bursts, one draw of 32 for each pair. BLER counts failed "
                "LDPC words after public normalized min-sum decoding. Pre-BER counts coded-bit errors "
                "before decoding. CFR NMSE pools squared channel error and truth energy before "
                "conversion to dB. Payload BCE is the unmodified mean over coded bits. Genie-PA receives "
                "the realized PA parameters. In R1 the channel estimate includes fitted oscillator phase "
                "and removes the nominal coherent PA gain."])
    for r in REGIMES:
        data = study["pooled"][depth][r]
        out.append(f"### R{r} at {study['config']['snr_db'][r]:g} dB")
        rows = []
        for arm in methods(data):
            row = data[arm]
            errs = row["block_errors"]
            local = {"pure": "local", "parity": "local_parity"}.get(arm)
            rows.append([NAMES[arm], f"{errs} / {row['blocks']}", f"{row['bler']:.2%}", f"{row['pre_ber']:.2%}",
                         f"{row['nmse_db']:.2f}", num(row["bce"]), delta(errs - data["physics_parity"]["block_errors"]),
                         delta(errs - data[local]["block_errors"]) if local else "—"])
        out.append(table(["Receiver", "Failed / total", "BLER", "Pre-BER", "NMSE (dB)", "BCE",
                          "Δ errors vs fixed + parity", "Δ errors vs matched local"], rows))
    out.append("The following differences subtract the comparator from the attention receiver. "
               "Rate differences use percentage points.")
    rows = []
    for r in REGIMES:
        for arm, local in (("pure", "local"), ("parity", "local_parity")):
            for ref in ("physics_parity", local):
                d = study["differences"][depth][r][f"{arm}_minus_{ref}"]
                rows.append([f"R{r}", NAMES[arm], NAMES[ref], delta(d["block_errors"]), delta(d["bit_errors"]),
                             f"{100*d['bler']:+.2f}", f"{100*d['pre_ber']:+.2f}",
                             f"{d['nmse_db']:+.3f}", f"{d['bce']:+.5f}"])
    out.append(table(["Regime", "Receiver", "Comparator", "Δ failed words", "Δ coded errors", "Δ BLER (pp)",
                      "Δ Pre-BER (pp)", "Δ NMSE (dB)", "Δ BCE"], rows))


def depths_and_curves(out, study, previous, records):
    out.extend(["## Solver depth and training progress",
                "Each count uses the same stream-6 B32 draw for each pair. Iteration 1 used 64 updates "
                "and constant LR 0.003. The depth comparison within iteration 2 holds the optimizer, "
                "batches and update count fixed. The cross-iteration differences include both model and training changes."])
    rows = []
    for r in REGIMES:
        for arm in ("physics_parity", *PARITY):
            old2 = previous["pooled_main"][r][arm]["block_errors"]
            if arm == "physics_parity":
                old3 = previous["depth3_fixed_reference"]["pooled"][r][arm]["block_errors"]
            else:
                old3 = previous["ablations"][r][f"depth3/{arm}/64"]["block_errors"]
            new2 = study["pooled"]["2"][r][arm]["block_errors"]
            new3 = study["pooled"]["3"][r][arm]["block_errors"]
            rows.append([f"R{r}", NAMES[arm], old2, new2, old3, new3, delta(new3-new2), delta(new3-old3)])
    out.append(table(["Regime", "Receiver", "Iter 1, 2 steps", "Iter 2, 2 steps", "Iter 1, 3 steps",
                      "Iter 2, 3 steps", "Iter 2 Δ (3−2)", "3 steps Δ (Iter 2−1)"], rows))
    rows = []
    checkpoints = study["config"]["checkpoints"]
    for r in REGIMES:
        for arm in PARITY:
            scores = checkpoint_metrics(records, r, arm, checkpoints)
            rows.append([f"R{r}", NAMES[arm], *(f"{e} / {b:.5f}" for e,b in scores)])
    out.append("Three-step parity training traces show pooled validation failures / payload BCE. "
               "Every receiver uses the final scheduled update.")
    out.append(table(["Regime", "Receiver", *(f"Update {u}" for u in checkpoints)], rows))


def interventions(out, study, records):
    out.extend(["## Physical-handle suppression",
                "Each entry is the change in failed words after suppressing one family at every pass of "
                "the trained three-step receiver, without refitting. A positive difference means the "
                "enabled family saved words on these frames. Each full-model replay is checked and "
                "removing all handles is checked against fixed + parity. The whole-antenna intervention "
                "disables the branch in both live and frozen maps and preserves the scaling of the "
                "remaining branches. The dedicated antenna correction intervention leaves antenna "
                "inputs to the other readouts active."])
    for r in REGIMES:
        summary = study["suppression"][r]
        out.append(f"### R{r} paired interventions")
        pairs = {arm: [records[(3, int(r), p, arm)]["suppressions"] for p in PAIRS] for arm in PARITY}
        rows = []
        for handle, label in HANDLE_NAMES.items():
            if handle not in summary["parity"]["rows"]:
                continue
            columns = {}
            for arm in PARITY:
                changes = [x[handle]["block_errors"] - x["full"]["block_errors"] for x in pairs[arm]]
                total = summary[arm]["removed_minus_full"][handle]["block_errors"]
                if sum(changes) != total:
                    raise ValueError("Suppression deltas disagree with receipts")
                columns[arm] = [*changes, total]
            rows.append([label, *(f"{delta(a)} / {delta(l)}" for a,l in zip(*columns.values(), strict=True))])
        out.append(table(["Suppressed family (attention / local)", "901", "902", "903", "Pooled Δ errors"], rows))
    rows = []
    for arm in PARITY:
        trajectories = [records[(3, 2, p, arm)]["trajectory_radians"] for p in PAIRS]
        for step, observations in enumerate(zip(*trajectories, strict=True)):
            over = sum(x["trajectories_over_half_rad"] for x in observations)
            total = sum(x["trajectories"] for x in observations)
            rows.append([NAMES[arm], step, f"{max(x['max_abs_increment'] for x in observations):.3f}",
                         f"{max(x['max_abs_accumulated'] for x in observations):.3f}",
                         f"{max(x['max_symbol_span'] for x in observations):.3f}", f"{over} / {total}"])
    out.extend(["### R2 residual trajectory range",
                "Angles are in radians. A trajectory is one burst and one ray. Its symbol span is "
                "the maximum minus minimum of the accumulated residual phase across symbol centers. "
                "Amplitude and span summarize the learned update. Phase recovery requires comparison "
                "to the simulated trajectory.",
                table(["Receiver", "Pass (0-based)", "Max absolute increment",
                       "Max absolute accumulated", "Max symbol span", "Spans above 0.5 rad"], rows)])


def kernels(out, study, previous):
    out.extend(["## R3 Green-kernel alignment",
                "Pearson correlations compare each probe map to the magnitude of the conditional "
                "propagation covariance. Each arm uses eight stream-8 bursts per pair, all three passes, "
                "48 coordinates and 2,256 off-diagonal entries per map. The shuffle uses eight stream-9 "
                "key permutations per pass. Means are taken within pair and then across pairs. "
                "Axis-only Pearson restricts pairs to the selected axis with both other coordinates fixed.",
                "The direct parallel joint uses the unsigned mean branch weight on its actual axis lines. "
                "The historical product composes the three measured maps in F→T→R order for comparison "
                "with iteration 1. Iteration 2 applies its branches in parallel. The individual-axis overall "
                "columns retain the historical expansion across the full set of coordinate pairs. "
                "These probes omit the value projection, per-channel value gain, residual and nonlinear readout."])
    rows = []
    for arm in ("pure", "parity"):
        for kernel, label in KERNELS.items():
            init = study["kernels"][arm]["initial"][kernel]
            end = study["kernels"][arm]["trained"][kernel]
            old = previous["r3_green_probe"][arm]["trained"].get(kernel)
            axis = end.get("axis_only", {})
            rows.append([NAMES[arm], label, num(old["overall"]["mean"] if old else None),
                         num(init["overall"]["mean"]), num(end["overall"]["mean"]),
                         num(init["shuffled"]["mean"]), num(end["shuffled"]["mean"]), num(axis.get("mean"))])
    out.append(table(["Receiver", "Kernel", "Iter 1 trained", "Iter 2 initial", "Iter 2 trained",
                      "Initial shuffled", "Trained shuffled", "Trained axis-only"], rows))
    support = study["invariants"]["results/iter2_checks/green_support_control.json"]["summary"]
    uniform = support["parallel_joint"]["mean"]
    pair_scores = support["parallel_joint"]["by_pair"]
    direct_pure = study["kernels"]["pure"]["trained"]["parallel_joint"]["overall"]["mean"]
    direct_parity = study["kernels"]["parity"]["trained"]["parallel_joint"]["overall"]["mean"]
    out.append(
        f"Uniform mixing on the same three axis lines gives direct-joint Pearson {uniform:.5f} "
        f"({pair_scores['901']['mean']:.5f}, {pair_scores['902']['mean']:.5f} and "
        f"{pair_scores['903']['mean']:.5f} on pairs 901, 902 and 903 respectively). "
        f"{support['off_diagonal_entries_off_all_axes']:,} of the {support['off_diagonal_entries']:,} "
        f"off-diagonal entries lie outside all three lines and carry zero direct weight. Relative to "
        f"this support-matched control, trained pure attention changes direct-joint Pearson by "
        f"{direct_pure-uniform:+.5f} and trained parity attention by {direct_parity-uniform:+.5f}. "
        f"Uniform mixing gives a constant historical product and hence an undefined Pearson coefficient."
    )


def timings(out, study):
    out.extend(["## Training and inference time",
                "Training seconds include stream-5 simulation, forward/backward and optimizer updates. "
                "Whole-job means also include validation and any applicable probes, suppressions and "
                "checkpoint writes. Each fit worker used one PyTorch thread."])
    rows = []
    for d in ("3", "2"):
        for r in REGIMES:
            for arm in ARMS if d == "3" else PARITY:
                row = study["pooled"][d][r][arm]
                fits = row["fit_seconds_by_pair"]
                rows.append([d, f"R{r}", NAMES[arm], *(f"{fits[str(p)]:.2f}" for p in PAIRS),
                             f"{row['mean_fit_seconds']:.2f}", f"{row['mean_job_seconds']:.2f}"])
    out.append(table(["Steps", "Regime", "Receiver", "Fit 901 (s)", "Fit 902 (s)", "Fit 903 (s)",
                      "Mean fit (s)", "Mean job (s)"], rows))
    rows = []
    methods_timed = list(dict.fromkeys(arm for r in REGIMES for arm in methods(study["latency"][r]["rows"]["1"])))
    for arm in methods_timed:
        values = [NAMES[arm]]
        for r in REGIMES:
            times = study["latency"][r]["rows"]
            if arm in times["1"]:
                values.extend(f"{times[b][arm]['median_ms']:.1f}" for b in ("1","64"))
            else:
                values.extend(["—", "—"])
        rows.append(values)
    timer = study["latency"]["3"]
    out.append(f"Forward latency is median wall time in milliseconds on pair 901 after "
               f"{timer['warmups']} warmup and {timer['repeats']} timed forwards, with method order "
               f"shuffled on every repetition. Torch used {timer['torch_threads']} compute thread and "
               f"{timer['torch_interop_threads']} inter-op thread. Simulation, metric scoring and final "
               "LDPC decoding fall outside the timer.")
    out.append(table(["Receiver", "R1 B1", "R1 B64", "R2 B1", "R2 B64", "R3 B1", "R3 B64"], rows))
    isolation = []
    for r in REGIMES:
        timing = study["latency"][r]
        env = timing["environment"]
        core = env["selected_cpu_set"]
        affinity = (f"{core['logical']} / {core['core']} / {core['efficiency_class']}"
                    if core else "OS scheduling")
        rounds = [x for batch in timing["load_by_round"].values() for x in batch if not x["warmup"]]
        load = [x["background_percent_of_host"] for x in rounds]
        scans = [env["project_scan_before"], *(x["process_scan"] for x in rounds)]
        workers = {pid for x in scans for pid in x["concurrent_project_processes"]}
        unreadable = {pid for x in scans for pid in x["unreadable_python_pids"]}
        isolation.append([f"R{r}", affinity, f"{env['baseline']['background_percent_of_host']:.1f}%",
                          f"{min(load):.1f}–{max(load):.1f}%", f"{median(load):.1f}%", len(workers), len(unreadable)])
    out.append("The timing harness screened for project fits and audits before measurement and before "
               "each round. Background percentages measure total host busy CPU after subtracting the "
               "benchmark process's CPU time, including operating-system and other application "
               "activity. Each range and median covers timed rounds at both batch sizes. "
               "Unreadable Python command lines limit worker identification.")
    out.append(table(["Regime", "Pinned logical CPU / core / efficiency class", "Background before run",
                      "Background range", "Background median", "Identified project workers", "Unreadable Python processes"],
                     isolation))


def quiet_timings(out, supplement):
    if not supplement:
        return
    config = next(iter(supplement.values()))["config"]
    need = config["required_samples_per_cell"]
    repeats = config["forwards_per_timed_group"]
    out.extend(["### Low-background timing supplement",
                f"Both three-step parity receivers used the same pair-901 observations and checkpoints "
                f"as the first timing. The harness admitted a sample only when measured background CPU "
                f"was at most {config['full_host_background_limit_percent']:g}% of the full host during "
                f"both a preceding {config['pre_window_seconds']:g}-second window and the timed group. "
                f"The order of eligible cells was shuffled each cycle. Each sample is mean wall time per "
                f"forward over {repeats['1']} forwards at B1 or {repeats['64']} at B64. Admission "
                f"depends on measured CPU activity and project-process scans. Each regime had a "
                f"{config['time_budget_seconds_per_regime']:g}-second search budget after warmup."])
    cells, windows = [], []
    for r in REGIMES:
        if r not in supplement:
            windows.append([f"R{r}", "—", "—", "—", "—", "—", "No timing receipt"])
            continue
        row = supplement[r]
        attempts = row["attempts"]
        for b in ("1", "64"):
            for arm in PARITY:
                entry = row["rows"][b][arm]
                matching = [x for x in attempts if x["batch"] == int(b) and x["arm"] == arm]
                reasons = [x["reason"] for x in matching if not x["admitted"]]
                pre = reasons.count("pre_window_above_limit")
                during = reasons.count("sample_window_above_limit")
                admitted = entry["accepted"]
                if len(matching) != len(reasons) + admitted or entry["complete"] != (admitted == need):
                    raise ValueError(f"Inconsistent timing admission counts for R{r} B{b} {arm}")
                values = ", ".join(f"{v:.1f}" for v in entry["samples_ms"])
                median_ms = entry["median_ms"]
                cells.append([f"R{r}", f"B{b}", NAMES[arm], len(matching), pre, during, len(reasons)-pre-during,
                              f"{admitted}/{need}", values or "—", "—" if median_ms is None else f"{median_ms:.1f}"])
        admitted_load = [x["sample_window"]["background_percent_of_host"] for x in attempts if x["admitted"]]
        load_text = (f"{min(admitted_load):.2f}–{max(admitted_load):.2f}%" if admitted_load else "—")
        unreadable = {pid for x in attempts for pid in x.get("unreadable_python_pids", [])}
        cpu = row["environment"]["selected_cpu_set"]
        stops = (row["concurrent_stop_reason"] or ("Cell quotas reached" if row["complete"] else "Window search ended"))
        windows.append([f"R{r}", f"{cpu['logical']} / {cpu['core']}", f"{row['measured_search_seconds']:.1f}",
                        len(admitted_load), load_text, len(unreadable), stops])
    out.append(table(["Regime", "Batch", "Receiver", "Attempts", "Pre-window rejected", "Timed group rejected",
                      "Other rejected", "Accepted / required", "Accepted group means (ms)", "Median (ms)"], cells))
    out.append(table(["Regime", "Pinned logical CPU / core", "Search (s)", "Total accepted",
                      "Accepted group background CPU", "Unreadable Python processes", "Search outcome"], windows))
    completed = sum(x["complete"] for r in supplement.values() for cells_at_batch in r["rows"].values()
                    for x in cells_at_batch.values())
    target_cells = len(REGIMES) * len(repeats) * len(PARITY)
    out.append(f"{completed}/{target_cells} cells reached the required {need} accepted groups. A dash in "
               f"the median column marks incomplete coverage. Individual accepted group means are "
               f"retained for those cells. Even a qualifying interval permits background activity up to "
               f"the specified limit. [Per-attempt CPU and timing records](results/iter2_quiet_latency.json)")


def checks(out, receipts):
    check = receipts["results/iter2_checks/summary.json"]["summary"]
    pre = check["preflight"]
    tr = check["trained_gauge"]
    worst = tr["worst"]
    out.extend(["## Invariant and checkpoint checks",
                f"The pre-fit gate passed {check['p6_tests']} P6 tests, {check['p5b_tests']} P5-B tests "
                f"and {check['p5_tests']} P5 tests. The unchanged P5/P5-B tests used redirected "
                f"development fixtures. The P4 publication verifier exited successfully.",
                f"Initialized receivers matched their P6 fixed counterparts bit-for-bit in "
                f"{pre['exact_anchors']}/{pre['anchors_checked']} depth/regime/arm checks. The maximum "
                f"initial absolute LLR or CFR difference from the archived solver was "
                f"{pre['worst_archived_anchor']:.3g}. All {pre['min_live_real']} real parameters had "
                f"finite, nonzero gradient components in each of {pre['regime_arm_cells']} regime/arm "
                f"combinations. All {pre['gradient_directions']} tensor-wise directional finite differences "
                f"passed, with maximum absolute error {pre['max_finite_difference_absolute_error']:.3g} "
                f"at half-width 1e−7.",
                f"Restored three-step gauge-enabled checkpoints passed {tr['passed']}/{tr['checks']} "
                f"gain/condition checks. Across {tr['coded_decisions']:,} coded decisions there were "
                f"{tr['coded_flips']} flips. The required absolute grid LLR difference was <1e−12. The "
                f"largest was {tr['max_abs_llr']:.3g} for R{worst['regime']}, pair {worst['pair']}, "
                f"{NAMES[worst['arm']]}, condition `{worst['condition']}`. Maximum relative CFR L2 "
                f"difference was {tr['max_rel_cfr']:.3g}. Absolute receive gains ranged from 1e−35 to "
                f"1e20 with independent phases. Fixtures covered pilot erasures, all-zero observations, "
                f"zero noise and ambiguous pilot phases. No-gauge is the explicit amplitude ablation."])
    if tr["failures"]:
        out.append(table(["Regime", "Pair", "Arm", "Failed condition", "Max Δ LLR", "Coded flips"],
                         [[f"R{v['regime']}", v["pair"], NAMES[v["arm"]], v["condition"],
                           f"{v['max_grid_llr_diff']:.3g}", v["coded_flips"]] for v in tr["failures"]]))
    rows = []
    for r in (2, 3):
        paths = [x for x in tr["pairwise_physical_estimation"] if x["regime"] == r]
        if {v["pair"] for v in paths} != set(PAIRS):
            raise ValueError(f"Incomplete R{r} trajectory/delay audit")
        for arm in PARITY:
            diagnostics = [x["arms"][arm] for x in paths]
            if r == 2:
                values = [x["mean_absolute_wrapped_row_error_initial_then_steps"] for x in diagnostics]
            else:
                values = [[x["nominal_mae"], *x["fitted_mae_by_step"]] for x in diagnostics]
            rows.append([f"R{r}", NAMES[arm], *(f"{sum(v)/len(v):.4f}" for v in zip(*values, strict=True))])
    out.append("The following post-inference truth comparisons average equally across the "
               "three pairs. R2 measures mean absolute wrapped trajectory error per symbol in "
               "radians after fitting one constant phase origin per burst and ray. R3 measures "
               "mean absolute delay error in Nyquist samples over the three movable rays.")
    out.append(table(["Regime", "Receiver", "Initial", "After pass 0", "After pass 1", "After pass 2"], rows))


def render(study, previous):
    config = study["config"]
    if tuple(config["dev_pairs"]) != PAIRS or config["streams"] != {
            "train": 5, "score": 6, "r3_probe": 8, "r3_permutation": 9}:
        raise ValueError("Unexpected dev scope")
    records = {(x["depth"], x["regime"], x["pair"], x["arm"]): x for x in study["by_pair"]}
    nparams, = {x["parameters"] for x in records.values() if "parameters" in x}
    out = ["# Phase 6 development iteration 2"]
    decision(out, study, records)
    out.extend(["## Receiver and fit",
           f"The primary comparison uses three waveform steps and {config['updates']} Adam updates. "
           f"Pairs 901, 902 and 903 use stream 5 for fitting and stream 6 for validation. "
           f"Stream 8 supplies R3 probe bursts and stream 9 supplies permutation seeds.",
           f"All five learned arms use {nparams} trainable real parameters and a frozen copy of the "
           f"initial operator. Attention and local start with identical tensors. Their width-eight "
           f"branches share full-frame features, estimated per-ray CFR inputs, physical handles, "
           f"objective, training batches and optimizer schedule. The local arm applies three-tap "
           f"depthwise mixing. The attention arm uses Hermitian coherence magnitude, rotary positions "
           f"on time and frequency, and phase-steered values on the antenna branch. The no-gauge arm "
           f"exposes absolute receive units to the learned head.",
           "An initial operator pass uses pilot residuals and the initial channel to refine the first "
           "equalizer. Later trajectory updates accumulate in sample time. The inverse PA correction "
           "uses the reciprocal bounded Laguerre AM-AM/AM-PM gain on the oversampled equalized waveform. "
           "Its coefficients interpolate between learned burst endpoints. R3 uses bounded fractional "
           "delays and a dedicated spatial CFR correction. The receiver uses the physical variance.",
           f"The training loss is payload BCE plus {config['aux_nmse']:g} times training-frame CFR NMSE. "
           f"Batch size is {config['train_batch']}. The global gradient norm limit is {config['clip']:g}. "
           f"Four warmup steps reach LR {config['lr']:g}, followed by cosine decay to one tenth "
           f"of peak LR. Training truth is used for the NMSE loss. Inference receives the observation."])
    primary(out, study, str(config["primary_depth"]))
    depths_and_curves(out, study, previous, records)
    interventions(out, study, records)
    kernels(out, study, previous)
    timings(out, study)
    quiet_timings(out, study.get("quiet_latency"))
    checks(out, study["invariants"])
    out.append("[Iteration 2 measurements and receipts](results/dev_iter2.json) · "
               "[Iteration 1 measurements](results/dev_iter1.json)")
    return "\n\n".join(out) + "\n"


if __name__ == "__main__":
    data = read(ROOT / "results/dev_iter2.json")
    earlier = read(ROOT / "results/dev_iter1.json")
    (ROOT / "DEV_ITER2.md").write_text(render(data, earlier), encoding="utf-8")
