"""Aggregate the selected development checkpoints with replay and provenance checks."""

from dataclasses import asdict

import torch

from p6 import experiment as ex
from p6 import inspect_iter3_paths as signs
from p6 import iter4_spatial
from p6 import run_dev_iter2 as prior
from p6 import run_dev_iter4 as study
from p6.data import PAIRS, SNR, dev_seed
from p6.green_support_control import run as support_control
from p6.handle_ablation import assert_replay


def aggregate():
    signature = study.implementation()
    records, interventions, pooled, latency, spatial, kernels, r2 = [], [], {}, {}, {}, {}, {}
    for regime in SNR:
        records_by_arm = {arm: [] for arm in ("references", *ex.ARMS)}
        diagnostics = {arm: [] for arm in ex.PARITY_ARMS}
        for pair in PAIRS:
            inits = set()
            for arm, group in records_by_arm.items():
                dest = study.location(regime, pair, arm)
                record = ex.read(dest)
                if (record["implementation"] != signature or record["score_seed"] != dev_seed(regime, 6, pair)
                        or (record["regime"], record["pair"], record["arm"]) != (regime, pair, arm)):
                    raise ValueError(f"Mismatched fit receipt {dest}")
                group.append(record)
                records.append(record)
                if arm == "references":
                    continue
                if (record["checkpoint_update"] != study.PRIMARY or record["parameters"] != 856
                        or set(record["trace"]) != {str(s) for s in study.CHECKPOINTS}
                        or record["checkpoint_sha256"] != prior.digest(dest.with_suffix(".pt"))):
                    raise ValueError(f"Invalid model or trace in {dest}")
                inits.add(record["initial_weights_sha256"])
                references = records_by_arm["references"][-1]["scores"]
                ref = references["physics_parity" if arm in ex.PARITY_ARMS else "physics"]
                assert_replay(record["trace"]["0"]["score"], ref)
                assert_replay(record["trace"][str(study.PRIMARY)]["score"], record["score"])
                if arm in ex.PARITY_ARMS:
                    diag = ex.read(dest.with_name(f"{arm}_interventions.json"))
                    if (diag["checkpoint_sha256"] != record["checkpoint_sha256"] or
                            diag["implementation"] != signature):
                        raise ValueError(f"Stale interventions for {dest}")
                    assert_replay(diag["suppressed"]["all"], ref)
                    assert_replay(diag["suppressed"]["full"], record["score"])
                    diagnostics[arm].append(diag)
                    interventions.append(diag)
            if len(inits) != 1:
                raise ValueError(f"Different initial weights in R{regime}/{pair}")
        group = records_by_arm["references"]
        rows = {arm: prior.pooled([r["scores"][arm] for r in group]) for arm in group[0]["scores"]}
        for arm in ex.ARMS:
            rows[arm] = prior.pooled([r["score"] for r in records_by_arm[arm]])
        if {r["blocks"] for r in rows.values()} != {1728 if regime == 2 else 384}:
            raise ValueError("Mismatched validation denominators")
        pooled[str(regime)] = rows
        if regime == 2:
            for arm, group in diagnostics.items():
                values = [{k: torch.tensor(v, dtype=torch.float64) for k, v in
                           r["physical"]["ray_values"].items()} for r in group]
                r2[arm] = signs.sign_summary(values)
        if regime == 3:
            for arm in ("pure", "parity"):
                kernels[arm] = prior.probe_summary(records_by_arm[arm])
            for arm, group in diagnostics.items():
                spatial[arm] = iter4_spatial.pool([r["physical"]["spatial"] for r in group])
            for group_a, group_l in zip(*diagnostics.values(), strict=True):
                if group_a["physical"]["fixed_spatial"] != group_l["physical"]["fixed_spatial"]:
                    raise ValueError("Fixed steering replays disagree")
            spatial["physics_parity"] = iter4_spatial.pool(
                [r["physical"]["fixed_spatial"] for r in diagnostics["parity"]])
        timing = ex.read(study.JOBS / f"latency_r{regime}.json")
        if (timing["implementation"] != signature or timing["pair"] != PAIRS[0]
                or (timing["torch_threads"], timing["torch_interop_threads"]) != (1, 1)
                or set(timing["rows"]) != {"1", "64"}
                or len(timing["environment"]["effective_affinity"]) != 1
                or any(set(v) != set(rows) for v in timing["rows"].values())):
            raise ValueError(f"Incomplete latency record in R{regime}")
        latency[str(regime)] = timing
    checks = ex.RESULTS / "iter4_checks" / "summary.json"
    verification = ex.read(checks)
    if verification["implementation"] != signature:
        raise ValueError("Validation and training source digests disagree")
    cue = ex.read(ex.RESULTS / "iter4_checks" / "chirp_cue.json")
    if cue["implementation"] != signature:
        raise ValueError("Cue intervention and training implementations differ")
    for arm, observations in cue["arms"].items():
        for pair in PAIRS:
            fit = next(r for r in records if (r["regime"], r["pair"], r["arm"]) == (2, pair, arm))
            changed = observations["by_pair"][str(pair)]
            if changed["checkpoint_sha256"] != fit["checkpoint_sha256"]:
                raise ValueError("Cue-only intervention used different weights")
            assert_replay(changed["full"]["score"], fit["score"])
    return dict(
        scope="Iteration 4 development only", implementation=signature,
        config=dict(**asdict(study.CONFIG), optimizer="Adam", updates=study.PRIMARY,
                    checkpoints=study.CHECKPOINTS, extension="49 to 64 fixed at 0.1 times peak",
                    cap=10., displacement_coefficient=.00005, train_batch=study.TRAIN_BATCH,
                    validation_batch=study.EVAL_BATCH, green_probe_batch=study.PROBE_BATCH,
                    pairs=PAIRS, snr_db=SNR, streams=dict(training=5, validation=6, probe=8, shuffle=9)),
        pooled=pooled, by_pair=records, interventions=interventions, r2_signs=r2, kernels=kernels,
        spatial=spatial, spatial_definition=iter4_spatial.DEFINITION, r2_cue_ablation=cue,
        uniform_green_support=support_control(),
        latency=latency, verification=verification, verification_sha256=prior.digest(checks))


def ablated(report, regime, arm, label):
    return sum(r["suppressed"][label]["block_errors"] for r in report["interventions"]
               if r["arm"] == arm and r["regime"] == regime)


def add_decision(report):
    rows = report["pooled"]
    def failures(r, arm):
        return rows[str(r)][arm]["block_errors"]

    r3green = report["kernels"]["parity"]
    uniform = report["uniform_green_support"]["parallel_joint"]["mean"]
    r3steer = report["spatial"]
    def steering(arm):
        return r3steer[arm]["waveform_solve_3"]["steering_cosine_after"]

    checks = dict(
        r1_at_most_23=failures(1, "parity") <= 23,
        r1_at_most_20=failures(1, "parity") <= 20,
        r1_pa_off_no_better=ablated(report, 1, "parity", "radial") >= failures(1, "parity"),
        r2_at_most_355=failures(2, "parity") <= 355,
        r2_seven_of_ten_repaired=(report["r2_signs"]["parity"]["all"]["initial_wrong"] == 10
                                  and report["r2_signs"]["parity"]["all"]["repaired"] >= 7),
        r2_pure_beats_local=failures(2, "pure") < failures(2, "local"),
        r2_parity_beats_local=failures(2, "parity") < failures(2, "local_parity"),
        r3_at_most_53_reference=failures(3, "parity") <= 53,
        r3_pure_beats_local=failures(3, "pure") < failures(3, "local"),
        r3_parity_beats_local=failures(3, "parity") < failures(3, "local_parity"),
        r3_final_steering_beats_local=steering("parity") > steering("local_parity"),
        r3_initial_green_joint_at_least_034=r3green["initial"]["parallel_joint"]["overall"]["mean"] >= .34,
        r3_initial_green_spatial_at_least_015=r3green["initial"]["spatial"]["overall"]["mean"] >= .15,
        r3_trained_green_joint_above_uniform=r3green["trained"]["parallel_joint"]["overall"]["mean"] > uniform,
        r3_trained_green_spatial_above_0084=r3green["trained"]["spatial"]["overall"]["mean"] > .084,
        numerical_and_regression_checks=report["verification"]["passed"])
    report["decision"] = dict(selected_update=study.PRIMARY, checks=checks,
                              all_listed_comparisons_met=all(checks.values()), formal_run=False)


def narrative(report):
    rows = report["pooled"]
    def errors(regime, arm="parity"):
        return rows[str(regime)][arm]["block_errors"]

    v = report["verification"]
    g = v["trained_gauge"]
    sg = report["r2_signs"]["parity"]["all"]
    green = report["kernels"]["parity"]
    def pearson(stage, axis):
        return green[stage][axis]["overall"]["mean"]

    uniform = report["uniform_green_support"]["parallel_joint"]["mean"]
    r3steer = report["spatial"]
    def cosine(arm):
        return r3steer[arm]["waveform_solve_3"]["steering_cosine_after"]

    decision = "Development targets were met." if report["decision"]["all_listed_comparisons_met"] else (
        "Keep the formal protocol open. The common 48-update recipe misses development targets.")
    cells = []
    for r, old in enumerate((29, 378, 57), 1):
        cells.append(f"| R{r} | {rows[str(r)]['parity']['blocks']:,} | {errors(r, 'physics_parity')} | "
                     f"{old} | {errors(r)} | {errors(r, 'local_parity')} |")
    lines = [
        "# Phase 6 development study — Iteration 4", "## Scientific decision",
        f"**{decision}** The primary comparison uses the same update for all five fitted receivers in all three regimes.",
        "\n".join(["| Regime | LDPC words | Fixed + parity | Iteration 3 Axial-Parity (80) | "
                    "Iteration 4 Axial-Parity (48) | Iteration 4 Local-Parity (48) |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |", *cells]),
        f"**R1.** Axial-Parity gave {errors(1)}/384 failures. Suppressing both learned PA coefficients "
        f"gave {ablated(report,1,'parity','radial')}, keeping only their forward contribution gave "
        f"{ablated(report,1,'parity','emit_only')}, and suppressing learned common phase gave "
        f"{ablated(report,1,'parity','cpe')}. The full model exceeded both the 23-word upper target "
        "and the stricter 20-word target. The PA intervention tied the full receiver on each pair. "
        "The same Axial-Parity trace reached 22 at update 24 and 24 at updates 32 and 64. "
        "Only the preselected update 48 counts toward the primary comparison.",
        f"**R2.** Axial-Parity reached {errors(2)}/1,728. Matched Local-Parity gave "
        f"{errors(2,'local_parity')}. Without parity, Attention gave {errors(2,'pure')} and Local "
        f"{errors(2,'local')}. Both pure arms regressed from Iteration 3’s 388 and 401. "
        f"Attention corrected {sg['repaired']} of {sg['initial_wrong']} initially wrong signs and "
        f"introduced {sg['regressed']} wrong signs. Aligned midpoint phase MAE went from "
        f"{sg['initial_aligned_phase_mae_rad']:.4f} to {sg['final_aligned_phase_mae_rad']:.4f} rad. "
        f"Suppressing the bank gave {ablated(report,2,'parity','soft_bank')} failed words. "
        f"Suppressing the general CFR residual gave {ablated(report,2,'parity','delta_h')}. "
        "The parity advantage was six words on pair 901, a tie on 902 and a five-word loss "
        "on 903. Attention rose from 340 at update 32 to 345 at 64. Local had fewer failures at "
        "both of those checkpoints.",
        f"**R3.** Axial-Parity gave {errors(3)}/384 failures against {errors(3,'local_parity')} "
        f"for Local-Parity and the supplied 53-word post-Iteration-3 reference. Suppressing the "
        f"complex spatial amplitude correction gave {ablated(report,3,'parity','spatial')}, "
        "improving the count on every pair. Suppressing fractional delays gave "
        f"{ablated(report,3,'parity','delays')}. Without parity, Attention and Local gave "
        f"{errors(3,'pure')} and {errors(3,'local')} failures. "
        f"The direct joint Green correlation went from {pearson('initial','parallel_joint'):.5f} to "
        f"{pearson('trained','parallel_joint'):.5f} against the {uniform:.5f} uniform control. "
        f"The spatial correlation went from {pearson('initial','spatial'):.5f} to "
        f"{pearson('trained','spatial'):.5f}. Final steering cosine was {cosine('parity'):.5f} "
        f"for Attention and {cosine('local_parity'):.5f} for Local. The structured ray prior "
        "already accounts for most of the trained joint correlation. The new spatial correction "
        "showed a decoding cost and Local retained a slight lead in final steering agreement.",
        "## Design and scope",
        "Pairs 901, 902 and 903 each supply one independent fit per arm and regime. Within a pair "
        "all arms share training batches and one stream-6 validation draw of 32 bursts at the same "
        "SNR (12, 9 and 7 dB for R1, R2 and R3). Stream 5 supplies initialization and training. "
        "Each R3 Green probe uses eight stream-8 bursts and stream-9 column permutations. "
        "The same development draws appeared in prior iterations. Differences in pooled counts "
        "alone cannot establish generalization to unseen pairs.",
        "The two mixers use the same 856 trainable real scalars, input fields, initialization, "
        "three tied waveform passes, centered physical readouts, Adam updates and minibatches. "
        "Attention uses global axis mixing with Hermitian affinity. Local uses three-tap axial "
        "convolutions. Width is eight. The four warmup updates reach 0.0045 and cosine decay reaches "
        "0.00045 at update 48. Updates 49–64 continue at 0.00045 on the same optimizer state. "
        "Training uses B4, unit gradient-norm clipping, BCE on 10 tanh(LLR/10), 0.005 times raw "
        "CFR NMSE and 0.00005 times squared displacement from initialization. Every validation "
        "score and final decoder consumes raw logits. A common 48-update endpoint was chosen "
        "before the fits. The 32- and 64-update rows provide within-trajectory comparisons.",
        "Training times are per-fit wall times with concurrent CPU workers. The separate latency "
        "process used one logical CPU, with intra-op and inter-op thread counts both set to one. "
        "It ran after the project's training, intervention and gauge workers exited. Other host "
        "activity was present during measurement. The timing tables give the measured load and "
        "the JSON retains individual wall-time and process-CPU samples.",
        "The first pass masks the input residual to known pilots. Each PA coefficient uses a "
        "frequency-centered constellation-power contrast. The quintic contrast also removes its "
        "linear projection onto the cubic contrast. The learned time-domain PA increment is "
        "orthogonal to each symbol waveform in the complex inner product. The receive-side "
        "adjustment is bounded at 0.04 constellation units. Each regime bounds the general "
        "CFR residual at 0.06. R2 alone enables trajectory updates. Its later passes refine "
        "soft-symbol LS taps with a bounded backbone path cue, use 160 times normalized coherence "
        "in the chirp softmax and accumulate the gated phase residual. R3 retains fixed per-ray "
        "Doppler and learned fractional delays. Its additive complex antenna correction comes "
        "from the spatial branch projected onto the delay dictionary and demodulated across time. "
        "Both Q and K receive the same normalized, unscrambled physical ray coordinates plus "
        "separate learned residuals. The additive R3 correction precedes waveform prediction "
        "and CFR reconstruction.",
        "The architectural and training changes were applied together. Handle suppressions and "
        "the cue-only replay use the selected weights, so causal attribution is limited to those "
        "interventions. Earlier 48-step spot-check results supplied in the request are historical comparators.",
        "## Verification",
        f"P6 passed {v['p6_tests']}/{v['p6_tests']} with {v['p6_skipped']} skipped. The unchanged P5 "
        f"and P5-B test bodies passed {v['p5_tests']}/15 and {v['p5b_tests']}/13 through the dev "
        f"fixture adapter. The default publication P4 verifier exited {v['p4']['exit']} after checking "
        "archived receipts. The adapter preserved the archived assertions and redirected their "
        "waveform draws to allowed development seeds. P6 assertions that previously required a "
        "DC response in the cubic PA readout and the external ULA projector now exercise "
        "zero-DC PA coefficients and the additive branch-derived antenna correction. New tests "
        "cover waveform orthogonality, same-coordinate ray injection and one-pass antenna reach.",
        f"All {v['first_update_arms']} actual first minibatches had {v['active_real_scalar_min']}/"
        f"{v['trainable_real_scalars']} active real scalar gradients. All {v['exact_depth3_anchors']} "
        "selected-depth initialization checks matched their corresponding fixed receiver bit-for-bit. "
        f"The {v['finite_difference_directions']} tensor-direction checks had maximum absolute error "
        f"{v['max_directional_error']:.3e}.",
        f"The four gauge-enabled trained variants passed {g['passed_checks']}/{g['checks']} "
        f"condition checks over gains spanning 1e−35 to 1e20, with {g['coded_flips']} changed "
        f"payload coded-bit decisions out of {g['decisions']:,}. Maximum absolute ΔLLR was "
        f"{g['max_grid_llr_diff']:.3e} and maximum relative CFR L2 error was "
        f"{g['max_relative_cfr_l2']:.3e}. Each intact, erased or zero-noise condition uses "
        "the full 27-magnitude grid with seeded phases. NoGauge is the intentional amplitude-sensitive control. "
        "[Verification receipt](results/iter4_checks/summary.json).",
        f"Training implementation `{report['implementation']}`. The machine-readable result includes "
        "per-pair scores, interventions, checkpoints’ digests, probe observations and timing samples. "
        "[Full development results](results/dev_iter4.json). [Iteration 3](DEV_ITER3.md).",
    ]
    cue = report["r2_cue_ablation"]["arms"]
    cue_rows = ["| Receiver | Full failures | Cue off | Full sign repairs | Cue off |",
                "| --- | ---: | ---: | ---: | ---: |"]
    for arm, name in (("parity", "Axial-Parity"), ("local_parity", "Local-Parity")):
        full, off = (cue[arm]["pooled"][k] for k in ("full", "no_cue"))
        cue_rows.append(f"| {name} | {full['score']['block_errors']} | {off['score']['block_errors']} "
                        f"| {full['sign_summary']['all']['repaired']} | {off['sign_summary']['all']['repaired']} |")
    lines.extend(["## R2 backbone cue intervention",
                  "Only the learned path cue is removed from chirp scoring. The soft taps, sharp "
                  "bank, phase gate and other learned handles continue to operate.",
                  "\n".join(cue_rows),
                  "Removing the cue preserved all nine sign repairs in each receiver. In Attention, "
                  "adding the cue reduced midpoint phase MAE on the initially wrong rays from "
                  f"{cue['parity']['pooled']['no_cue']['sign_summary']['initial_wrong_sign']['final_aligned_phase_mae_rad']:.4f} "
                  f"to {cue['parity']['pooled']['full']['sign_summary']['initial_wrong_sign']['final_aligned_phase_mae_rad']:.4f} "
                  "radians and added two failed words. The sign repairs survived this intervention "
                  "in the shared solver."])
    return "\n\n".join(lines) + "\n"


def main():
    from p6.iter4_tables import render

    ex.setup()
    report = aggregate()
    add_decision(report)
    ex.save(ex.RESULTS / "dev_iter4.json", report)
    # Render exactly the serialized object used by programmatic consumers.
    report = ex.read(ex.RESULTS / "dev_iter4.json")
    path = ex.ROOT / "DEV_ITER4.md"
    path.write_text(narrative(report) + "\n" + render(report).strip() + "\n", encoding="utf-8")
    print(f"Wrote {path}", flush=True)


if __name__ == "__main__":
    main()
