"""Run the bounded Iteration-1 development study from the repository root."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import math
import statistics

from p6 import experiment as ex


def selected_jobs(args):
    return [j for j in ex.jobs()
            if (args.regime is None or j[0] in args.regime)
            and (args.pair is None or j[1] in args.pair)
            and (args.design is None or j[2] in args.design)
            and (args.arm is None or j[3] in args.arm)]


def execute(jobs, workers):
    if workers == 1:
        for job in jobs:
            ex.run_job(job)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=ex.setup) as pool:
            for _ in pool.map(ex.run_job, jobs, chunksize=1):
                pass


def metrics(row):
    return dict(**row, bler=row["block_errors"] / row["blocks"],
                pre_ber=row["bit_errors"] / row["bits"],
                nmse_db=10 * math.log10(row["channel_error"] / row["channel_energy"]),
                bce=row["bce_sum"] / row["bits"])


def mean_curves(records):
    # Each contributor has the same B4 train and B8 validation sample/bit counts.
    out = []
    for points in zip(*(r["trace"] for r in records), strict=True):
        row = {"step": points[0]["step"]}
        if any(point["step"] != row["step"] for point in points):
            raise ValueError("Unmatched learning curve checkpoint")
        for k in ("train_mean_bce", "train_mean_objective", "validation_bce",
                  "mean_preclip_norm", "max_preclip_norm", "lr", "cumulative_fit_seconds"):
            values = [p[k] for p in points if p[k] is not None]
            row[k] = statistics.mean(values) if values else None
        out.append(row)
    return out


def summarize_probes(records):
    result = {}
    for stage in ("initial", "trained"):
        per_axis = {}
        for axis in ("joint", "spatial", "time", "frequency"):
            per_metric = {}
            for metric in ("overall", "shuffled", "axis_only"):
                if axis == "joint" and metric == "axis_only":
                    continue
                pairs = {}
                for r in records:
                    values = [v for layer in r["probe"][stage]
                              for v in layer["maps"][axis][metric] if v is not None]
                    pairs[str(r["pair"])] = statistics.mean(values) if values else None
                finite = [v for v in pairs.values() if v is not None]
                per_metric[metric] = dict(by_pair=pairs,
                                          mean=statistics.mean(finite) if finite else None)
            per_axis[axis] = per_metric
        result[stage] = per_axis
    return result


def budget_selection(main, exploratory):
    selected = {}
    for regime in main:
        selected[regime] = {}
        for arm in ex.PARITY_ARMS:
            candidates = {64: main[regime][arm]}
            for step in (96, 128):
                candidates[step] = exploratory[regime][f"main/{arm}/{step}"]
            by_metric = {}
            for metric in ("bler", "bce"):
                step = min(candidates, key=lambda n: (candidates[n][metric], n))
                by_metric[metric] = dict(step=step, value=candidates[step][metric],
                                         fit_seconds=candidates[step]["mean_fit_seconds_per_pair"])
            selected[regime][arm] = by_metric
    return selected


def batch_partition_check():
    path = ex.RESULTS / "dev_batch32_check.json"
    if path.exists():
        return ex.read(path)
    rows = {}
    for regime in ex.SNR:
        model = ex.OperatorEMTransformer("physics_parity", steps=2).eval()
        pooled, per_pair = {}, {}
        for pair in ex.PAIRS:
            frame = ex.dev_frame(regime, 6, pair, 32, 0)
            row = ex.score([frame], lambda f: model(f.obs))
            per_pair[str(pair)] = dict(errors=row["block_errors"], blocks=row["blocks"],
                                       seed=ex.dev_seed(regime, 6, pair, 0))
            ex.add(pooled, row)
        rows[str(regime)] = dict(per_pair=per_pair, pooled=pooled)
    data = dict(purpose="Single agreed frame-partition provenance check. Baseline only.",
                batch_size=32, batches_per_pair=1, stream=6,
                model="p5b.OperatorEMTransformer physics_parity depth2", results=rows)
    ex.save(path, data)
    return data


def partition_comparison(main):
    supplied = {"1": (32, 384), "2": (439, 1728), "3": (66, 384)}
    one_batch = batch_partition_check()["results"]
    comparison = {}
    for regime, (user_errors, user_blocks) in supplied.items():
        row = main[regime]["physics_parity"]
        check = one_batch[regime]["pooled"]
        if row["block_errors"] != check["block_errors"] or row["blocks"] != check["blocks"]:
            raise ValueError(f"Primary B32 scoring disagrees with development batch check in R{regime}")
        comparison[regime] = dict(user_errors=user_errors, user_blocks=user_blocks,
                                  measured_errors=row["block_errors"], measured_blocks=row["blocks"],
                                  matches_provided_counts=(row["block_errors"], row["blocks"]) ==
                                                          (user_errors, user_blocks))
    return comparison


def depth_reference():
    """Measure the third fixed EM pass on the same public validation draw."""
    path = ex.RESULTS / "depth3_fixed.json"
    if path.exists():
        return ex.read(path)
    pooled, by_pair = {}, {}
    for regime in ex.SNR:
        key = str(regime)
        pooled[key], by_pair[key] = {}, {}
        for pair in ex.PAIRS:
            frames = ex.valid_frames(regime, pair)
            sample = {}
            for name in ex.FIXED:
                fixed = ex.OperatorEMTransformer(name, steps=3).eval()
                row = ex.score(frames, lambda frame: fixed(frame.obs))
                ex.add(pooled[key].setdefault(name, {}), row)
                sample[name] = row
            by_pair[key][str(pair)] = dict(rows=sample, seed=ex.dev_seed(regime, 6, pair))
        pooled[key] = {method: metrics(row) for method, row in pooled[key].items()}
    out = dict(steps=3, batch_size=32, pooled=pooled, by_pair=by_pair)
    ex.save(path, out)
    return out


def print_diagnostics(report):
    for regime, rows in sorted(report["ablations"].items()):
        print(f"R{regime} parity / matched local at each budget, ordered as "
              "block errors, pooled BCE, mean fit seconds per pair")
        for design, step in (("main", 64), ("main", 96), ("main", 128), ("depth3", 64)):
            details = []
            for arm in ex.PARITY_ARMS:
                row = (report["pooled_main"][regime][arm] if (design, step) == ("main", 64)
                       else rows[f"{design}/{arm}/{step}"])
                details.append(f"{row['block_errors']}/{row['blocks']}, {row['bce']:.5f}, "
                               f"{row['mean_fit_seconds_per_pair']:.1f}s")
            print(f"  {design:8} {step:3}    " + " | ".join(details))
        print(f"  Pair 901 optimizer contrasts, parity / local block errors")
        for design in ("main_901_only", "lr005", "warmcos", "aux", "unclipped"):
            vals = [rows[f"{design}/{arm}/64"] for arm in ex.PARITY_ARMS]
            print(f"  {design:16} " + " | ".join(
                f"{v['block_errors']}/{v['blocks']}, BCE {v['bce']:.5f}" for v in vals))
        print("  Receiver latency in ms (B1 / B64)")
        times = report["latency_ms"][regime]
        for key in ("main/pure", "main/local", "main/parity", "main/local_parity",
                    "depth3/parity", "depth3/local_parity"):
            print(f"  {key:22} {times['1'][key]['median_ms']:.2f} / "
                  f"{times['64'][key]['median_ms']:.2f}")
    for arm, stages in report["r3_green_probe"].items():
        print(f"R3 {arm} Pearson, initial / trained / shuffled trained")
        for axis in stages["trained"]:
            vals = (stages["initial"][axis]["overall"]["mean"],
                    stages["trained"][axis]["overall"]["mean"],
                    stages["trained"][axis]["shuffled"]["mean"])
            print(f"  {axis:12} " + " / ".join("undefined" if v is None else f"{v:.5f}" for v in vals))


def summarize():
    all_jobs = list(ex.jobs())
    grouped = {}
    manifest = []
    missing = []
    signatures = {d: ex.implementation_id(d) for d in ("main", "references")}
    for regime, pair, design, arm in all_jobs:
        path = ex.job_dir(regime, pair, design) / f"{arm}.json"
        if not path.exists():
            missing.append(path)
            continue
        record = ex.read(path)
        signature = signatures["references" if design == "references" else "main"]
        if record["implementation"] != signature:
            raise ValueError(f"Outdated implementation in {path}")
        grouped.setdefault((regime, design, arm), []).append(record)
        manifest.append(str(path.relative_to(ex.ROOT).as_posix()))
    timings = {}
    for regime in ex.SNR:
        path = ex.RESULTS / "jobs" / f"latency_r{regime}.json"
        if not path.exists():
            missing.append(path)
            continue
        record = ex.read(path)
        if record["implementation"] != signatures["main"]:
            raise ValueError(f"Outdated implementation in {path}")
        timings[str(regime)] = record["rows"]
    if missing:
        raise RuntimeError(f"{len(missing)} missing records; first missing: {missing[0]}")
    primary, exploratory, curve, per_pair, probes = {}, {}, {}, {}, {}
    capacities = set()
    for (regime, design, arm), records in grouped.items():
        key = str(regime)
        if design == "references":
            for name in records[0]["rows"]:
                total = {}
                for record in records:
                    ex.add(total, record["rows"][name])
                primary.setdefault(key, {})[name] = metrics(total)
            continue
        capacities.update(r["parameters"] for r in records)
        curvename = f"r{regime}/{design}/{arm}"
        curve[curvename] = mean_curves(records)
        if records[0]["probe"].get("trained"):
            probes[arm] = summarize_probes(records)
        for step in records[0]["rows"]:
            total = {}
            pair_rows = {}
            for record in records:
                ex.add(total, record["rows"][step])
                elapsed = next(p["cumulative_fit_seconds"] for p in record["trace"]
                               if p["step"] == int(step))
                pair_rows[str(record["pair"])] = dict(scores=metrics(record["rows"][step]),
                                                       fit_seconds=elapsed)
            per_pair[f"{curvename}/{step}"] = pair_rows
            entry = metrics(total)
            entry["mean_fit_seconds_per_pair"] = statistics.mean(
                p["fit_seconds"] for p in pair_rows.values())
            if design == "main" and step == "64":
                primary.setdefault(key, {})[arm] = entry
            else:
                exploratory.setdefault(key, {})[f"{design}/{arm}/{step}"] = entry
    if len(capacities) != 1:
        raise ValueError(f"Unmatched capacity across development arms: {capacities}")

    # Restrict optimizer ablation comparators to pair 901; depth/budget use 901..903.
    for regime in ex.SNR:
        for arm in ex.PARITY_ARMS:
            row = per_pair[f"r{regime}/main/{arm}/64"]["901"]
            exploratory[str(regime)][f"main_901_only/{arm}/64"] = {
                **row["scores"], "mean_fit_seconds_per_pair": row["fit_seconds"]}
    report = dict(
        scope="Iteration 1, development only",
        data=dict(pairs=list(ex.PAIRS), snr_db=ex.SNR, train_stream=5, validation_stream=6,
                  kernel_frames_stream=8, kernel_shuffle_seed_stream=9,
                  frames_per_validation_pair=ex.EVAL_BATCH * ex.VAL_BATCHES,
                  scoring_batch_size=ex.EVAL_BATCH, train_batch_size=ex.FIT_BATCH,
                  scoring_batches_per_pair=ex.VAL_BATCHES, validation_curve_frames=ex.CURVE_BATCH,
                  validation_curve_batch_index=0, validation_score_batch_index=0, probe_frames_per_pair=8,
                  shuffled_permutations_per_layer=8),
        methods=dict(
            primary="All five arms: 64 Adam updates, B4, constant LR .003, gradient clip 1, BCE, depth 2.",
            regularization="Aux = BCE + .05 sum(channel squared error)/sum(channel squared magnitude).",
            secondary="Depth3 and 96/128 parity checkpoints use all 3 pairs. LR, warmcos, aux, unclipped use pair 901 in each regime.",
            warmcos="Updates 1-4 warm from .25x to 1x LR. Following updates decay from 1x to .1x over remaining 60 updates.",
            counts="Raw payload coded-bit errors, decoded LDPC block errors, and total squared channel error and energy pooled over frames.",
            classical="genie_pa receives the realized PAState for R1. All other reference methods use the observation.",
            fixed="Archived independent P5-B Fixed and Fixed+Parity receivers at depth 2.",
            curves="At each curve checkpoint, all arms use a separate B8 simulation with the same stream-6 base seed as primary B32 scoring. These B8 draws are not assumed to be a subset of B32. Training values average the preceding 16 B4 updates. Curves average equal-size pair samples.",
            fit_time="One PyTorch thread per worker, frame simulation through optimizer update. Validation, probe, timing and checkpoint I/O excluded. Parallel jobs can share machine resources.",
            latency="Serial, one PyTorch thread. End-to-end receiver forward excluding simulation and final LDPC decode. Use pair 901 step-64 checkpoints; one warmup and five timed repeats per batch/arm.",
            probe="Full Green target for pure and parity attention arms. The local signed depthwise kernels have no scalar softmax attention matrix. Overall Pearson excludes diagonal. Axis-only restricts to coordinate pairs differing only on that axis. Shuffle permutes keys eight times per layer. Each pair is a mean over bursts and EM steps. Null denotes undefined Pearson.",
            baseline_comparison="The initial reference check with 4 batches of 8 gave R1/R2/R3 Fixed+Parity counts 31/438/65. A separate check at the first stream-6 seed with one batch of 32 reproduced all three provided fixed-parity totals. The benchmark adopted that B32 partition across all arms and ablations before inspecting trained results. Equality of aggregate counts does not establish individual frame identity with the supplied oracle run. No oracle outputs were recomputed.",
            budget_selection="For each parity arm separately, minimize pooled development BLER or BCE over checkpoints 64/96/128 of the identical constant-LR trajectory. Ties favor fewer updates. No confirmatory claim or protocol selection."),
        trainable_real_parameters=next(iter(capacities)), implementations=signatures,
        pooled_main=primary, ablations=exploratory, curves=curve, by_pair=per_pair,
        r3_green_probe=probes, latency_ms=timings, job_records=manifest,
        provided_fixed_parity_comparison=partition_comparison(primary),
        depth3_fixed_reference=depth_reference(),
        dev_budget_minima=budget_selection(primary, exploratory),
    )
    handle_path = ex.RESULTS / "handle_ablation.json"
    if handle_path.exists():
        handles = ex.read(handle_path)
        if handles["implementation"] != signatures["main"] or handles["frames_per_regime"] != 32:
            raise ValueError("Physical-handle intervention belongs to a different implementation or frame count")
        report["physical_handle_ablation"] = {
            "file": "results/handle_ablation.json", "pair": handles["pair"],
            "interpretation": handles["interpretation"],
            "removed_minus_full": {r: {a: record["removed_minus_full"] for a, record in arms.items()}
                                   for r, arms in handles["arms"].items()}}
    audit_path = ex.RESULTS / "invariants.json"
    if audit_path.exists():
        audit = ex.read(audit_path)
        tensors = [t for r in audit["rows"] for t in r["gradient"]["tensors"]]
        report["invariants"] = dict(
            file="results/invariants.json", passed=audit["passed"], arms_by_regime=len(audit["rows"]),
            all_p6_anchors_bit_exact=all(a["p6_bit_exact"] for r in audit["rows"] for a in r["anchors"]),
            max_archived_init_llr_diff=max(a["archived_max_llr_cfr"][0] for r in audit["rows"] for a in r["anchors"]),
            max_fd_error_by_halfwidth={str(e["half_width"]): max(t["estimates"][i]["absolute_error"] for t in tensors)
                                      for i, e in enumerate(tensors[0]["estimates"])})
        for name in ("extended_once_updated", "extended_trained"):
            path = ex.RESULTS / f"{name}.json"
            if path.exists():
                report["invariants"][name] = ex.read(path)["summary"]
    ex.save(ex.RESULTS / "dev_iter1.json", report)
    for regime, rows in sorted(primary.items()):
        print(f"R{regime}: method, block errors/blocks, BLER, pre-BER, NMSE dB, BCE")
        for name, row in rows.items():
            print(f"  {name:24} {row['block_errors']}/{row['blocks']}  "
                  f"{row['bler']:.4%}  {row['pre_ber']:.4%}  {row['nmse_db']:.2f}  {row['bce']:.5f}")
    print_diagnostics(report)
    print("Wrote p6/results/dev_iter1.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "job", "latency", "summarize"))
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--regime", type=int, choices=ex.SNR, action="append")
    parser.add_argument("--pair", type=int, choices=ex.PAIRS, action="append")
    parser.add_argument("--design", choices=("references", *ex.DESIGNS), action="append")
    parser.add_argument("--arm", choices=("references", *ex.ARMS), action="append")
    args = parser.parse_args()
    ex.setup()
    if args.action == "summarize":
        summarize()
        return
    if args.action == "latency":
        for regime in args.regime or ex.SNR:
            ex.latency(regime)
        return
    jobs = selected_jobs(args)
    if not jobs:
        parser.error("No jobs match the development design")
    execute(jobs, args.workers)
    if args.action == "run" and not any((args.regime, args.pair, args.design, args.arm)):
        for regime in ex.SNR:
            ex.latency(regime)
        summarize()


if __name__ == "__main__":
    main()
