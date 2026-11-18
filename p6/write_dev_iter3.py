"""Render the Iteration 3 report from its completed JSON. No receiver is imported."""

import json
from pathlib import Path
from statistics import median

from p6.write_dev_iter2 import KERNELS, NAMES, delta, methods, num, table


ROOT = Path(__file__).resolve().parent
PARITY = ("parity", "local_parity")
REGIMES = ("1", "2", "3")
START = "<!-- BEGIN SCIENTIFIC DECISION -->"
END = "<!-- END SCIENTIFIC DECISION -->"
INTERVENTIONS = {
    "full": "Full receiver", "radial": "PA residual off", "emit_only": "Forward PA residual only",
    "cpe": "Learned common phase off", "radial_and_cpe": "PA residual and common phase off",
    "phase": "Per-ray trajectory off", "soft_bank": "Soft chirp-bank residual off",
    "third_pass": "Omit third EM pass", "spatial": "Path-amplitude filter off",
    "spatial_operator": "Entire antenna mixing branch off", "delays": "Fractional delays off",
    "delta_h": "General CFR residual off", "all": "All handles off",
}


def percent(value):
    return f"{100 * value:.2f}%"


def method(study):
    c = study["config"]
    pairs = ", ".join(map(str, c["dev_pairs"]))
    streams = c["streams"]
    return [
        "## Design and evaluation",
        f"The study uses development pairs {pairs}. Each pair and regime has its own training seed and "
        f"its own stream-{streams['score']} evaluation draw of {c['validation_batch']} bursts. "
        f"All arms and checkpoints use the same evaluation draw within each pair and regime. Stream "
        f"{streams['train']} supplies training batches. The R3 Green probe uses stream {streams['r3_probe']} "
        f"with {c['probe_batch']} bursts per pair and stream {streams['r3_permutation']} for column shuffles. "
        "This is a repeated evaluation on development data. Formal cohorts remain unevaluated.",
        f"Attention and Local each use {study['trainable_real_parameters']} trainable real parameters, "
        f"width eight, {c['depth']} weight-tied EM passes, the same input features, physical solver and "
        f"training batches. Their head tensors share an initialization within each pair and regime. "
        "The centered head subtracts a frozen copy of its initial map. Attention mixes all positions "
        "on each axis. Local mixes three adjacent positions. The parity arms add the same "
        "code-based soft-symbol feedback. NoGauge exposes absolute receive amplitude to its head.",
        f"Each fit uses {c['optimizer']}, batch size {c['train_batch']}, gradient-norm limit {c['clip']:g}, "
        f"four warmup steps to learning rate {c['lr']:g} and cosine decay to one tenth of peak over "
        f"{c['updates']} updates. The primary comparison uses the prespecified final checkpoint. "
        f"Training computes payload BCE on `{c['cap']:g} × tanh(LLR / {c['cap']:g})`, adds "
        f"{c['aux_nmse']:g} times training-frame CFR NMSE, and adds {c['anchor_coefficient_on_sum_sq']:g} "
        "times the squared parameter displacement from the initial map. The uncapped parity control "
        "changes only the logits used in training BCE. Every validation metric and final LDPC decode uses raw LLRs.",
        "**R1.** Each symbol contributes all-tone DC and centered soft-symbol power-contrast projections. "
        "A linear burst fit yields the bounded PA endpoints. The contrast projection rejects a frequency-constant "
        "field on each row. The DC projection retains that field. The forward waveform incorporates the "
        "nominal PA and learned residual. Equalization "
        "applies 20% of the estimated linear Bussgang residual to the channel and predicts a dewarp from "
        "bounded posterior symbols, with an additive magnitude limit of 0.04 constellation units.",
        "**R2.** The first learned pass sees residuals from soft payload and known pilots. A per-symbol "
        "projection onto the delay dictionary drives a continuous sample-time phase update. Later passes "
        "also fit soft-symbol taps and use their coherence with a public chirp bank to estimate a gated "
        "trajectory residual. Each step accumulates its phase correction into the current rotation.",
        "**R3.** The antenna readout controls a Hermitian projection of each fitted antenna-amplitude vector "
        "onto its estimated ULA steering mode before waveform prediction and CFR reconstruction. A signed "
        "strength allows shrinkage or expansion. Estimated ray features enter through a learned embedding. "
        "R2 and R3 set the learned common-phase increment to zero.",
    ]


def comparisons(study):
    out = ["## Receiver comparison",
           "BLER counts failed LDPC words after normalized min-sum decoding. Pre-BER uses hard "
           "coded-bit decisions before decoding. Post-BER uses decoded information bits. CFR NMSE "
           "pools channel error and truth energy across pairs before conversion to dB. The Genie-PA "
           "baseline receives the realized PA parameters."]
    for r in REGIMES:
        rows = study["pooled"][r]
        fixed = rows["physics_parity"]["block_errors"]
        values = []
        for arm in methods(rows):
            s = rows[arm]
            other = {"pure": "local", "parity": "local_parity"}.get(arm)
            matched = delta(s["block_errors"] - rows[other]["block_errors"]) if other else "—"
            values.append([NAMES[arm], f"{s['block_errors']} / {s['blocks']}", percent(s["bler"]),
                           percent(s["pre_ber"]), percent(s["info_errors"] / s["info_bits"]),
                           f"{s['nmse_db']:.2f}", num(s["bce"]), delta(s["block_errors"] - fixed), matched])
        out.extend([f"### R{r} at {study['config']['snr_db'][r]:g} dB",
                    table(["Receiver", "Failed / total", "BLER", "Pre-BER", "Post-BER", "NMSE (dB)",
                           "Raw BCE", "Δ words vs fixed + parity", "Δ words vs matched Local"], values)])
    return out


def curves(study):
    updates = study["config"]["checkpoints"]
    out = ["## Training trajectory and the BCE cap",
           "Each checkpoint cell gives failed words / validation BLER / raw validation BCE on the "
           "same frame draw. A positive 24→48 change means additional failed words. All checkpoints "
           "belong to one continuous schedule per fit."]
    confidence = []
    for r in REGIMES:
        rows = []
        for loss in ("capped", "uncapped"):
            for arm, points in study["curves"][loss][r].items():
                vals = [points[str(u)] for u in updates]
                cells = [f"{s['block_errors']} / {percent(s['bler'])} / {num(s['bce'])}" for s in vals]
                late = points["48"]["block_errors"] - points["24"]["block_errors"]
                rows.append([NAMES[arm], loss, *cells, delta(late)])
                if arm in PARITY:
                    confidence.append([f"R{r}", NAMES[arm], loss, *[
                        f"{v['mean_absolute_llr']:.2f} / {percent(v['safe_margin_over_6'] / v['bits'])} / "
                        f"{percent(v['wrong_margin_below_minus_6'] / v['bits'])}" for v in vals]])
        out.extend([f"### R{r}", table(["Receiver", "Training BCE", *[f"Update {u}" for u in updates],
                                         "Δ words 24→48"], rows)])
    out.extend(["### Validation confidence", "Each cell gives mean absolute raw LLR, percentage of coded "
                "bits with signed correct margin above 6, and percentage with signed margin below −6. "
                "Individual decoder failure mechanisms would require codeword-level interventions.",
                table(["Regime", "Receiver", "Training BCE", *[f"Update {u}" for u in updates]], confidence)])
    return out


def suppressions(study):
    pairs = list(map(str, study["config"]["dev_pairs"]))
    records = {(str(j["regime"]), str(j["pair"]), j["arm"]): j for j in study["by_pair"]
               if j["loss"] == "capped" and j["arm"] in PARITY}
    out = ["## Handle interventions on all development pairs",
           "Each row intervenes on the final capped checkpoint without retraining. Pair cells give failed "
           "words and the change from the same pair’s full receiver in parentheses. A positive change means "
           "the enabled component prevented failures in this intervention. Interacting effects need not add. "
           "PA residual off retains the public nominal PA. Forward PA residual only keeps learned emission "
           "and disables the learned receive-side gain and dewarp. The third-pass intervention evaluates "
           "the first two passes of the trained three-pass receiver."]
    labels = list(INTERVENTIONS)
    for r in REGIMES:
        values = []
        for arm in PARITY:
            data = study["suppression"][r][arm]
            for name in sorted(data["rows"], key=labels.index):
                row = data["rows"][name]
                d = data["removed_minus_full"][name]
                cells = []
                for pair in pairs:
                    n = records[r, pair, arm]["suppressions"][name]["block_errors"]
                    cells.append(f"{n} ({delta(data['block_delta_by_pair'][name][pair])})")
                values.append([NAMES[arm], INTERVENTIONS[name], *cells, row["block_errors"],
                               delta(d["block_errors"]), f"{d['bce']:+.5f}"])
        out.extend([f"### R{r}", table(["Receiver", "Intervention", *pairs, "Pooled failures",
                                         "Δ failures", "Δ raw BCE"], values)])
    return out


def paths(study):
    out = ["## R2 trajectory depth",
           "The correction span is the range across symbol rows of accumulated sample-averaged "
           "phase correction for one burst and ray. The largest span and counts pool all three "
           "pairs. The p95 column lists each pair’s 95th percentile in pair order."]
    span_rows, truth_rows, subgroup_rows = [], [], []
    pairs = list(map(str, study["config"]["dev_pairs"]))
    for arm in PARITY:
        paths_by_pair = [study["paths"]["2"][arm][p] for p in pairs]
        for i, s in enumerate(zip(*(v["trajectory_radians"] for v in paths_by_pair), strict=True), 1):
            span_rows.append([NAMES[arm], i, f"{max(v['max_abs_increment'] for v in s):.4f}",
                              f"{max(v['max_abs_accumulated'] for v in s):.4f}",
                              f"{max(v['max_symbol_span'] for v in s):.4f}",
                              " / ".join(f"{v['p95_symbol_span']:.4f}" for v in s),
                              f"{sum(v['trajectories_over_half_rad'] for v in s)} / "
                              f"{sum(v['trajectories'] for v in s)}"])
        truth = [v["fitted_truth"] for v in paths_by_pair]
        for p, s in zip(pairs, truth, strict=True):
            truth_rows.append([NAMES[arm], p, s["trajectories"],
                               *map(num, s["mean_absolute_wrapped_row_error_initial_then_steps"])])
        for group, label in (("over_0.5", ">0.5"), ("over_0.5_through_3", "(0.5, 3]"), ("over_3", ">3")):
            cells = [v["groups_by_initial_unwrapped_truth_minus_bank_span"][group] for v in truth]
            counts = [v["burst_rays"] for v in cells]
            n = sum(counts)
            errors = zip(*(v["mean_absolute_wrapped_row_error_initial_then_steps"] for v in cells), strict=True)
            means = [sum(m * count for m, count in zip(step, counts, strict=True) if count) / n if n else None
                     for step in errors]
            subgroup_rows.append([NAMES[arm], label, " / ".join(map(str, counts)), n,
                                  *map(num, means), sum(v["improved_final_vs_initial"] for v in cells)])
    out.extend([table(["Receiver", "Pass", "Max increment magnitude", "Max accumulated magnitude",
                       "Max symbol span", f"p95 span ({'/'.join(pairs)})", "Spans >0.5 / total"], span_rows),
                "Truth comparisons use the simulator after receiver inference. Mean absolute wrapped "
                "row error is in radians and allows one constant phase per burst and ray.",
                table(["Receiver", "Pair", "Burst-rays", "Initial", "Pass 1", "Pass 2", "Pass 3"], truth_rows),
                "The initial mismatch span is the range of the unwrapped phase difference between the "
                "initial bank trajectory and truth. The (0.5, 3] and >3 groups partition the >0.5 group. "
                "A large mismatch can arise from an off-grid or wrong-sign trajectory. The diagnostic "
                "does not independently label their causes. Subgroup means weight each burst-ray equally.",
                table(["Receiver", "Initial mismatch span (rad)", f"Counts ({'/'.join(pairs)})", "Total",
                       "Initial error", "Pass 1", "Pass 2", "Pass 3", "Lower final error than initial"],
                      subgroup_rows)])
    return out


def chirp_sign(detail):
    heading = "### Chirp direction against simulator truth"
    if detail is None:
        return [heading, "The aggregate JSON contains no checkpoint replay for the chirp-sign diagnostic."]
    groups = {
        "all": "All rays", "initial_wrong_sign": "Initially wrong sign",
        "initial_correct_sign": "Initially correct sign",
        "initial_span_over_0.5_through_3": "Initial span (0.5, 3] rad",
        "initial_span_over_3": "Initial span >3 rad",
        "wrong_sign_and_initial_span_over_0.5_through_3": "Initially wrong sign and span (0.5, 3] rad",
    }
    rows, by_pair = [], []
    for arm in PARITY:
        record = detail["r2"][arm]
        for key, label in groups.items():
            s = record["pooled"][key]
            rows.append([NAMES[arm], label, s["burst_rays"], f"{s['initial_wrong']} → {s['final_wrong']}",
                         f"{s['initial_zero_slope']} → {s['final_zero_slope']}", s["repaired"], s["regressed"],
                         f"{num(s['initial_net_slope_mae'])} → {num(s['final_net_slope_mae'])}",
                         f"{num(s['initial_aligned_phase_mae_rad'])} → {num(s['final_aligned_phase_mae_rad'])}"])
        for pair in detail["pair_ids"]:
            s = record["by_pair"][str(pair)]["all"]
            by_pair.append([NAMES[arm], pair, s["burst_rays"], s["initial_wrong"], s["final_wrong"],
                            s["repaired"], s["regressed"]])
    return [heading, detail["r2_definition"], "The arrows compare initialization with the final pass. "
            "Repairs count wrong-to-correct sign changes. Regressions count correct-to-wrong changes. "
            "The zero-slope column accounts for indeterminate signs. Groups overlap. Mid-sample errors "
            "and spans here use the midpoint of each symbol. The preceding table uses symbol-averaged phasors.",
            table(["Receiver", "Subset", "Burst-rays", "Wrong signs", "Zero slope", "Repaired",
                   "Regressed", "Net slope MAE", "Aligned phase MAE (rad)"], rows),
            table(["Receiver", "Pair", "Burst-rays", "Initial wrong", "Final wrong", "Repaired",
                   "Regressed"], by_pair)]


def antenna_filter(detail):
    heading = "### Signed antenna filter and off-mode residuals"
    if detail is None:
        return [heading, "The aggregate JSON contains no checkpoint replay for the antenna-mode diagnostic."]
    stage_names = {"pilot_redemap": "Pilot redemap", "all_waveform_solves": "All waveform solves"}
    rows, by_pair = [], []
    for arm in PARITY:
        record = detail["r3"][arm]
        for stage, s in record["pooled"].items():
            label = stage_names.get(stage, stage.replace("_", " ").capitalize())
            rows.append([NAMES[arm], label, s["burst_ray_filter_applications"],
                         f"{s['positive']} / {s['negative']} / {s['zero']}", num(s["mean_gain"]),
                         " / ".join(num(s[k]) for k in ("p05_gain", "median_gain", "p95_gain")),
                         num(s["estimated_mode_off_rms_ratio_after_over_before"]),
                         num(s["true_mode_off_rms_ratio_after_over_before"])])
        for pair in detail["pair_ids"]:
            s = record["by_pair"][str(pair)]["all_waveform_solves"]
            by_pair.append([NAMES[arm], pair, s["burst_ray_filter_applications"],
                            f"{s['positive_percent']:.2f}%", num(s["mean_gain"]),
                            num(s["estimated_mode_off_rms_ratio_after_over_before"]),
                            num(s["true_mode_off_rms_ratio_after_over_before"])])
    return [heading, detail["r3_definition"], "Positive gain moves amplitudes toward the estimated mode. "
            "Negative gain moves them away. An off-mode RMS ratio below one means that the residual "
            "decreased relative to the indicated steering mode. The pooled waveform line counts all "
            "passes for each burst-ray and excludes the initial redemap.",
            table(["Receiver", "Filter application", "Applications", "Positive / negative / zero",
                   "Mean gain", "Gain p05 / p50 / p95", "Estimated-mode RMS ratio",
                   "True-mode RMS ratio"], rows),
            table(["Receiver", "Pair, all waveform solves", "Applications", "Positive gain",
                   "Mean gain", "Estimated-mode RMS ratio", "True-mode RMS ratio"], by_pair)]


def spatial(study):
    support = study["uniform_green_support"]
    base = support["parallel_joint"]
    pairs = list(map(str, study["config"]["dev_pairs"]))
    delay_rows, kernel_rows = [], []
    for arm in PARITY:
        for p in pairs:
            fit = study["paths"]["3"][arm][p]["fitted_truth"]
            delay_rows.append([NAMES[arm], p, num(fit["nominal_mae"]), *map(num, fit["fitted_mae_by_step"])])
    for arm, stages in study["kernels"].items():
        for name, init in stages["initial"].items():
            final = stages["trained"][name]
            raw = final["overall"]
            a, b = init["overall"]["mean"], raw["mean"]
            kernel_rows.append([NAMES[arm], KERNELS[name], num(a), num(b),
                                "—" if a is None or b is None else f"{b-a:+.5f}",
                                num(final["shuffled"]["mean"]),
                                num(init.get("axis_only", {}).get("mean")),
                                num(final.get("axis_only", {}).get("mean")),
                                " / ".join(num(raw["by_pair"][p]) for p in pairs)])
    return ["## R3 fractional delay and Green-kernel measurements",
            "Fractional-delay MAE uses the movable rays and is expressed in Nyquist samples. "
            "The nominal value measures the error at zero learned offset.",
            table(["Receiver", "Pair", "Nominal delay MAE", "Pass 1", "Pass 2", "Pass 3"], delay_rows),
            *antenna_filter(study.get("verification", {}).get("summary", {}).get("physical_detail")),
            "### Green-kernel correlations",
            "Pearson correlation compares the measured attention weights with the evaluation-only "
            "conditional propagation Green kernel. Overall values use off-diagonal token pairs. "
            "Axis-only values restrict both tokens to one physical axis. Pair means average finite "
            "burst/pass coefficients, then the overall mean weights each pair equally. Shuffled values "
            "average eight stream-9 key permutations. The direct parallel joint uses the measured line "
            "supports of the added branches. The historical product composes their weights.",
            table(["Receiver", "Kernel", "Initial", "Trained", "Δ", "Trained shuffled",
                   "Initial axis-only", "Trained axis-only", f"Trained ({'/'.join(pairs)})"], kernel_rows),
            f"Uniform weights on the same three axial lines give direct-joint Pearson {num(base['mean'])} "
            f"over {support['total_bursts']} probe bursts ({', '.join(p + ' ' + num(base['by_pair'][p]['mean']) for p in pairs)}). "
            f"Each burst contributes {support['off_diagonal_entries']} off-diagonal entries, including "
            f"{support['off_diagonal_entries_off_all_axes']} outside all three lines. The uniform historical "
            "product is constant and its Pearson coefficient is undefined. The uniform-support control "
            "quantifies alignment available from geometry alone. The antenna interventions measure decoding impact."]


def timing(study):
    latency = study["latency"]
    out = ["## Single-thread receiver timing",
           "Cells are median wall milliseconds per forward on the timing pair’s stream-6 inputs. "
           "The harness shuffles method order within each round and excludes its warmup. "
           "Simulation, final LDPC decoding and scoring occur outside the timed forward."]
    methods_present = dict.fromkeys(k for v in latency.values() for k in v["rows"]["1"])
    times, loads = [], []
    for arm in methods(methods_present):
        cells = []
        for r in REGIMES:
            for batch in ("1", "64"):
                value = latency[r]["rows"][batch].get(arm)
                cells.append(f"{value['median_ms']:.2f}" if value else "—")
        times.append([NAMES[arm], *cells])
    for r in REGIMES:
        rec = latency[r]
        for batch, rounds in rec["load_by_round"].items():
            timed = [x for x in rounds if not x["warmup"]]
            host = [x["host_busy_percent"] for x in timed]
            back = [x["background_percent_of_host"] for x in timed]
            affinity = rec["environment"]["effective_affinity"]
            loads.append([f"R{r}", batch, rec["pair"], rec["repeats"], rec["torch_threads"],
                          rec["torch_interop_threads"], str(affinity) if affinity else "OS-selected",
                          f"{min(host):.1f}–{max(host):.1f}%", f"{min(back):.1f}–{max(back):.1f}%",
                          f"{median(back):.1f}%", rec["quiet_wait_seconds"]])
    out.extend([table(["Receiver", "R1 B1", "R1 B64", "R2 B1", "R2 B64", "R3 B1", "R3 B64"], times),
                "Host busy is the percentage of total host CPU capacity in use across all processes. "
                "Background subtracts benchmark-process CPU from host busy. Each range covers the timed "
                "rounds for the specified batch. The load was observed during measurement.",
                table(["Regime", "Batch", "Pair", "Timed rounds", "Torch threads", "Interop threads",
                       "Logical affinity", "Host busy range", "Background range", "Background median",
                       "Quiet wait (s)"], loads)])
    return out


def verification(study):
    out = ["## Verification"]
    if "verification" not in study:
        return out + ["The aggregate JSON contains no verification receipt. The numerical and regression "
                      "checks are unverified in this report."]
    rec = study["verification"]
    v = rec["summary"]
    gauge = v["trained_gauge"]
    case = gauge["worst_llr_case"]
    live = v["live_real_per_regime_arm"]
    out.extend([
        f"The combined verification receipt records **{'PASS' if v['passed'] else 'FAIL'}**. "
        f"P6 ran {v['p6_tests']} tests with {v['p6_skipped']} skipped. The archived P5 and P5-B "
        f"suites ran {v['p5_tests']} and {v['p5b_tests']} tests. Their original test bodies used "
        f"{len(v['legacy_dev_draws'])} distinct development frame configurations through the fixture "
        f"adapter. `publication.verify_p4` returned exit code {v['p4']['exit']}.",
        f"Initialization matched the P6 fixed solver bit-for-bit in {v['exact_anchors']}/{v['initialized_anchors']} "
        f"comparisons. The maximum archived initial LLR/CFR difference was {v['max_archived_init_diff']:.3g}. "
        f"Across {len(live)} regime/arm combinations, the active real gradient count ranged from "
        f"{min(live)} to {max(live)} of {study['trainable_real_parameters']} trainable components. "
        f"The audit checked {v['finite_difference_directions']} tensor-wise finite-difference directions. "
        f"The largest absolute directional error was {v['max_directional_error']:.3g}.",
        f"The final gauge-enabled checkpoints passed {gauge['passed_checks']}/{gauge['checks']} condition checks "
        f"at complex gains spanning 1e−35 to 1e20. There were {gauge['coded_flips']:,} coded flips across "
        f"{gauge['decisions']:,} decisions. Maximum absolute grid ΔLLR was {gauge['max_grid_llr_diff']:.4g} "
        f"(R{case['regime']}, pair {case['pair']}, {NAMES[case['arm']]}, `{case['condition']}`). "
        f"Maximum relative CFR L2 difference was {gauge['max_relative_cfr_l2']:.4g}. "
        "The gauge test excludes the intentional NoGauge control.",
        f"Implementation signature `{study['implementation']}`. [Verification receipt]({rec['path']}). "
        "[Full metrics, checkpoints, per-pair measurements and timing rounds](results/dev_iter3.json).",
    ])
    return out


def render(study, decision=""):
    blocks = ["# Phase 6 development study — Iteration 3", "## Scientific decision",
              START + "\n" + decision.strip() + "\n" + END]
    for section in (method, comparisons, curves, suppressions, paths):
        blocks.extend(section(study))
    blocks.extend(chirp_sign(study.get("verification", {}).get("summary", {}).get("physical_detail")))
    for section in (spatial, timing, verification):
        blocks.extend(section(study))
    return "\n\n".join(blocks) + "\n"


def main():
    study = json.loads((ROOT / "results" / "dev_iter3.json").read_text(encoding="utf-8"))
    dest = ROOT / "DEV_ITER3.md"
    analysis = ""
    if dest.exists():
        prior = dest.read_text(encoding="utf-8")
        if prior.count(START) != 1 or prior.count(END) != 1:
            raise ValueError("Existing report has no unique decision markers")
        analysis = prior.split(START, 1)[1].split(END, 1)[0]
    dest.write_text(render(study, analysis), encoding="utf-8")


if __name__ == "__main__":
    main()
