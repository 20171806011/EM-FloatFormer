"""Iteration-2 development fits, paired interventions and single-thread timing.

From the repo root, run fit, latency, and summarize in that order. Each fit
worker has one torch thread. Run latency after all other workers have exited.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import lru_cache
import hashlib
import inspect
import math
import os
from pathlib import Path
from random import Random
import statistics
import time
from unittest.mock import patch

import torch

from p5.wavefield_sim import green_kernel
from p6 import axial_em_transformer as solver
from p6 import experiment as ex
from p6.data import PAIRS, SNR, dev_frame, dev_seed
from p6.handle_ablation import assert_replay
from p6.study import metrics


JOBS = ex.RESULTS / "iter2_jobs"
UPDATES = 48
TRAIN_BATCH = 4
EVAL_BATCH = 32
PROBE_BATCH = 8
PRIMARY = 3
DEPTHS = (3, 2)
CONFIG = ex.Design(depth=PRIMARY, lr=.0045, schedule="warmcos", clip=1., aux_nmse=.02)
AXES = {"spatial": 0, "time": 1, "frequency": 2}
DIFFERENCES = ("block_errors", "bit_errors", "bler", "pre_ber", "bce", "nmse_db")


def samples(regime, pair):
    return [dev_frame(regime, 6, pair, EVAL_BATCH, 0)]


def contrasts(row, baseline):
    return {name: row[name] - baseline[name] for name in DIFFERENCES}


def pooled(rows):
    total = {}
    for row in rows:
        ex.add(total, row)
    return metrics(total)


def job_keys():
    for depth in DEPTHS:
        for regime in SNR:
            for pair in PAIRS:
                arms = ex.ARMS if depth == PRIMARY else ex.PARITY_ARMS
                for arm in ("references", *arms):
                    yield regime, pair, depth, arm


def location(key):
    regime, pair, depth, arm = key
    if key not in set(job_keys()):
        raise ValueError(f"Outside development experiment: {key}")
    return JOBS / f"r{regime}" / str(pair) / f"depth{depth}" / f"{arm}.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.no_grad()
def probe(model, frame, shuffle_seed):
    _, _, layers = model(frame.obs, probe="axes")
    if not layers:
        raise ValueError("Expected attention matrices for the global receiver")
    output = []
    gen = torch.Generator().manual_seed(shuffle_seed)
    for coords, joint, axes in layers:
        target = green_kernel(frame, coords)
        n = len(coords)
        off = ~torch.eye(n, dtype=torch.bool)
        perms = [torch.randperm(n, generator=gen) for _ in range(8)]
        maps = {}
        kernels = {"joint": joint, **axes}
        if not set(AXES).union({"parallel_joint"}).issubset(kernels):
            raise ValueError(f"Incomplete parallel-probe maps: {kernels.keys()}")
        for name, kernel in kernels.items():
            if kernel.shape != target.shape:
                raise ValueError(f"{name} and Green function shapes disagree")
            raw = ex.correlation(kernel[:, off], target[:, off])
            shuffled = torch.stack([ex.correlation(kernel[:, :, p][:, off], target[:, off])
                                    for p in perms])
            entry = dict(overall=ex.nullable(raw), shuffled=ex.nullable(shuffled.nanmean(0)))
            if name in AXES:
                others = [axis for axis in range(3) if axis != AXES[name]]
                same = (coords[:, None, others] == coords[None, :, others]).all(-1) & off
                entry["axis_only"] = ex.nullable(ex.correlation(kernel[:, same], target[:, same]))
                entry["axis_only_entries"] = int(same.sum())
            maps[name] = entry
        output.append(dict(coords=coords.tolist(), off_diagonal_entries=int(off.sum()), maps=maps))
    return output


def suppressions(regime, frames, model, fixed_row, full_row):
    original = solver._physical_handles
    increments = []

    def observe(*args, **kwargs):
        handles = original(*args, **kwargs)
        if regime == 2:
            increments.append(handles.phase.detach())
        return handles

    with patch.object(solver, "_physical_handles", observe):
        replay = ex.score(frames, lambda f: model(f.obs))
    assert_replay(replay, full_row)
    trajectory = []
    if increments:
        cumulative = torch.stack(increments).cumsum(0)
        for step, phase in enumerate(cumulative):
            row_phase = phase.mean(2)
            span = row_phase.amax(1) - row_phase.amin(1)
            trajectory.append(dict(pass_index=step, max_abs_increment=float(increments[step].abs().max()),
                                   max_abs_accumulated=float(phase.abs().max()),
                                   max_symbol_span=float(span.max()),
                                   p95_symbol_span=float(span.quantile(.95)),
                                   trajectories_over_half_rad=int((span > .5).sum()),
                                   trajectories=span.numel()))
    names = ["cpe", "delta_h"]
    if regime != 2:
        names.append("radial")
    if regime != 1:
        names.append("phase")
    if regime == 3:
        names.extend(("delays", "spatial"))
    masks = {name: (name,) for name in names}
    if regime == 1:
        masks["radial_and_cpe"] = ("radial", "cpe")
    masks["all"] = solver.Handles._fields
    result = {}
    for label, fields in masks.items():
        def masked(*args, **kwargs):
            values = original(*args, **kwargs)
            return values._replace(**{k: torch.zeros_like(getattr(values, k)) for k in fields})

        with patch.object(solver, "_physical_handles", masked):
            result[label] = ex.score(frames, lambda f: model(f.obs))
    if regime == 3:
        with patch.object(model.head, "disable_spatial", True), patch.object(
                model.initial_map, "disable_spatial", True):
            result["spatial_operator"] = ex.score(frames, lambda f: model(f.obs))
    assert_replay(result["all"], fixed_row)
    result["full"] = full_row
    return result, trajectory


def fit(key, signature, dest):
    regime, pair, depth, arm = key
    model = ex.model_for(regime, pair, arm, depth)
    parameters = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(parameters, lr=CONFIG.lr)
    frames = samples(regime, pair)
    init_row = ex.score(frames, lambda f: model(f.obs))
    row = init_row
    trace = [dict(update=0, score=row, train_bce=None, train_objective=None,
                  mean_preclip_norm=None, fit_seconds=0., lr=None)]
    probe_frame = None
    probes = {}
    if depth == PRIMARY and regime == 3 and arm in ("pure", "parity"):
        probe_frame = dev_frame(3, 8, pair, PROBE_BATCH, 0)
        probes["initial"] = probe(model, probe_frame, dev_seed(3, 9, pair))
    seconds = 0.
    losses, bces, norms = [], [], []
    for step in range(UPDATES):
        start = time.perf_counter()
        frame = dev_frame(regime, 5, pair, TRAIN_BATCH, step)
        lr = ex.learning_rate(CONFIG, step, UPDATES)
        opt.param_groups[0]["lr"] = lr
        opt.zero_grad(set_to_none=True)
        obj, bce = ex.objective(frame, model, CONFIG.aux_nmse)
        if not torch.isfinite(obj):
            raise ArithmeticError(f"Nonfinite training objective at {key}, update {step}")
        obj.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, CONFIG.clip, error_if_nonfinite=True)
        opt.step()
        seconds += time.perf_counter() - start
        losses.append(float(obj.detach()))
        bces.append(float(bce.detach()))
        norms.append(float(norm))
        if step + 1 in (UPDATES // 2, UPDATES):
            row = ex.score(frames, lambda f: model(f.obs))
            trace.append(dict(update=step+1, score=row, train_bce=statistics.mean(bces),
                              train_objective=statistics.mean(losses),
                              mean_preclip_norm=statistics.mean(norms), lr=lr, fit_seconds=seconds))
            losses, bces, norms = [], [], []
    model.eval()
    if probe_frame is not None:
        probes["trained"] = probe(model, probe_frame, dev_seed(3, 9, pair))
    ex.save_weights(dest.with_suffix(".pt"), model, signature)
    interventions, trajectory = {}, []
    if depth == PRIMARY and arm in ex.PARITY_ARMS:
        fixed = solver.AxialEMTransformer("physics_parity", steps=depth).eval()
        fixed_row = ex.score(frames, lambda f: fixed(f.obs))
        interventions, trajectory = suppressions(regime, frames, model, fixed_row, row)
    return dict(score=row, initial_score=init_row, trace=trace, probe=probes,
                suppressions=interventions, trajectory_radians=trajectory,
                fit_seconds=seconds, parameters=ex.real_parameters(model),
                init_seed=dev_seed(regime, 5, pair, 900), stream5_batch_range=[0, UPDATES-1],
                first_training_seed=dev_seed(regime, 5, pair, 0),
                last_training_seed=dev_seed(regime, 5, pair, UPDATES-1),
                checkpoint_sha256=digest(dest.with_suffix(".pt")))


def references(regime, pair, depth):
    frames = samples(regime, pair)
    rows = {}
    for name in ex.FIXED:
        model = ex.OperatorEMTransformer(name, steps=depth).eval()
        rows[name] = ex.score(frames, lambda f: model(f.obs))
    if depth == PRIMARY:
        for name in ex.METHODS[regime]:
            rows[name] = ex.score(frames, lambda f: ex.classical(f, name))
    return dict(scores=rows)


@lru_cache(maxsize=1)
def implementation():
    names = ("axial_em_transformer.py", "axial_operator.py", "data.py", "experiment.py",
             "parity.py", "handle_ablation.py")
    parts = {name: digest(ex.ROOT / name) for name in names}
    for func in (samples, job_keys, location, probe, suppressions, fit, references):
        parts[func.__name__] = hashlib.sha256(inspect.getsource(func).encode()).hexdigest()
    parts["config"] = repr((asdict(CONFIG), PRIMARY, DEPTHS, UPDATES, EVAL_BATCH, TRAIN_BATCH,
                            PROBE_BATCH, AXES))
    return hashlib.sha256(repr(sorted(parts.items())).encode()).hexdigest()


def run_job(key):
    dest = location(key)
    signature = implementation()
    regime, pair, depth, arm = key
    if dest.exists():
        old = ex.read(dest)
        if old.get("implementation") == signature:
            if arm == "references" or (dest.with_suffix(".pt").exists() and
                                       old["checkpoint_sha256"] == digest(dest.with_suffix(".pt"))):
                return
    start = time.perf_counter()
    record = references(regime, pair, depth) if arm == "references" else fit(key, signature, dest)
    elapsed = time.perf_counter() - start
    ex.save(dest, dict(regime=regime, pair=pair, depth=depth, arm=arm, implementation=signature,
                       score_seed=dev_seed(regime, 6, pair, 0), wall_seconds=elapsed, **record))
    print(f"R{regime}/{pair} depth{depth} {arm}: {elapsed:.2f}s", flush=True)


def restored(regime, pair, depth, arm):
    path = location((regime, pair, depth, arm))
    ck = path.with_suffix(".pt")
    receipt = ex.read(path)
    if receipt["implementation"] != implementation() or receipt["checkpoint_sha256"] != digest(ck):
        raise ValueError(f"Checkpoint provenance mismatch: {ck}")
    state = torch.load(ck, map_location="cpu", weights_only=True)
    if state["implementation"] != implementation():
        raise ValueError(f"Checkpoint source mismatch: {ck}")
    model = ex.model_for(regime, pair, arm, depth).eval()
    model.load_state_dict(state["state"])
    return model


def clock_methods(regime):
    methods = {}
    for name in ex.ARMS:
        model = restored(regime, PAIRS[0], PRIMARY, name)
        methods[name] = lambda f, m=model: m(f.obs)
    for name in ex.PARITY_ARMS:
        model = restored(regime, PAIRS[0], 2, name)
        methods[f"depth2/{name}"] = lambda f, m=model: m(f.obs)
    for name in ex.FIXED:
        model = ex.OperatorEMTransformer(name, steps=PRIMARY).eval()
        methods[name] = lambda f, m=model: m(f.obs)
    for name in ex.METHODS[regime]:
        methods[name] = lambda f, key=name: ex.classical(f, key)
    return methods


@torch.inference_mode()
def latency(regime, repeats, environment):
    from p6.latency_environment import interval, project_workers, snapshot

    methods = clock_methods(regime)
    pair = PAIRS[0]
    output, loads = {}, {}
    for batch in (1, 64):
        frame = dev_frame(regime, 6, pair, batch, 950 + batch // 64)
        wall = {arm: [] for arm in methods}
        cpu = {arm: [] for arm in methods}
        user = {arm: [] for arm in methods}
        system = {arm: [] for arm in methods}
        loads[batch] = []
        order = list(methods)
        rng = Random(dev_seed(regime, 6, pair, 959))
        for rep in range(repeats+1):
            scan = project_workers()
            start = snapshot()
            rng.shuffle(order)
            for arm in order:
                os_start = os.times()
                cpu0 = time.process_time_ns()
                t0 = time.perf_counter_ns()
                methods[arm](frame)
                ms = (time.perf_counter_ns() - t0) / 1e6
                cpu_ms = (time.process_time_ns() - cpu0) / 1e6
                os_end = os.times()
                if rep:
                    wall[arm].append(ms)
                    cpu[arm].append(cpu_ms)
                    user[arm].append(1000 * (os_end.user - os_start.user))
                    system[arm].append(1000 * (os_end.system - os_start.system))
            loads[batch].append(dict(rep=rep, warmup=not rep, process_scan=scan,
                                     **interval(start, snapshot())))
        project_workers()
        output[batch] = {k: dict(median_ms=statistics.median(v), wall_ms=v, process_cpu_ms=cpu[k],
                                 native_user_ms=user[k], native_system_ms=system[k])
                         for k,v in wall.items()}
    timing_hash = hashlib.sha256((inspect.getsource(clock_methods) + inspect.getsource(latency)).encode())
    timing_hash.update((ex.ROOT / "latency_environment.py").read_bytes())
    ex.save(JOBS / f"latency_r{regime}.json",
            dict(regime=regime, pair=pair, depth=PRIMARY, implementation=implementation(),
                 timing_implementation_sha256=timing_hash.hexdigest(),
                 warmups=1, repeats=repeats, rows=output, torch_threads=torch.get_num_threads(),
                 torch_interop_threads=torch.get_num_interop_threads(),
                 environment=environment, load_by_round=loads,
                 includes="receiver forward only; excludes simulation, metric scoring and final LDPC decode"))
    print(f"Measured R{regime} single-thread B1/B64 latency", flush=True)


def probe_summary(records):
    result = {}
    for stage in ("initial", "trained"):
        layers = records[0]["probe"][stage]
        result[stage] = {}
        for name in layers[0]["maps"]:
            cols = ["overall", "shuffled"] + (["axis_only"] if name in AXES else [])
            result[stage][name] = {}
            for col in cols:
                pair_values = {}
                for r in records:
                    values = [v for p in r["probe"][stage] for v in p["maps"][name][col]
                              if v is not None]
                    pair_values[str(r["pair"])] = statistics.mean(values) if values else None
                finite = [v for v in pair_values.values() if v is not None]
                result[stage][name][col] = dict(mean=statistics.mean(finite) if finite else None,
                                                by_pair=pair_values)
    return result


def summarize(invariant_paths=(), quiet_paths=()):
    signature = implementation()
    by_pair = {}
    digests = {}
    for key in job_keys():
        regime, pair, depth, arm = key
        path = location(key)
        rec = ex.read(path)
        if (rec["implementation"] != signature or rec["score_seed"] != dev_seed(regime, 6, pair)
                or (rec["regime"], rec["pair"], rec["depth"], rec["arm"]) != key):
            raise ValueError(f"Stale development job {key}")
        if arm != "references" and digest(path.with_suffix(".pt")) != rec["checkpoint_sha256"]:
            raise ValueError(f"Stale checkpoint {key}")
        by_pair[key] = rec
        digests[path.relative_to(ex.ROOT).as_posix()] = digest(path)

    results, comparisons, interventions, probes = {}, {}, {}, {}
    for depth in DEPTHS:
        results[depth], comparisons[depth] = {}, {}
        for regime in SNR:
            refs = [by_pair[(regime, p, depth, "references")] for p in PAIRS]
            rows = {m: pooled([r["scores"][m] for r in refs]) for m in refs[0]["scores"]}
            for arm in ex.ARMS if depth == PRIMARY else ex.PARITY_ARMS:
                records = [by_pair[(regime, p, depth, arm)] for p in PAIRS]
                fixed = "physics_parity" if arm in ex.PARITY_ARMS else "physics"
                for r, ref in zip(records, refs, strict=True):
                    assert_replay(r["initial_score"], ref["scores"][fixed])
                row = pooled([r["score"] for r in records])
                row["fit_seconds_by_pair"] = {r["pair"]: r["fit_seconds"] for r in records}
                row["mean_fit_seconds"] = statistics.mean(row["fit_seconds_by_pair"].values())
                row["mean_job_seconds"] = statistics.mean(r["wall_seconds"] for r in records)
                rows[arm] = row
                if depth == PRIMARY and arm in ex.PARITY_ARMS:
                    keys = records[0]["suppressions"]
                    summary = {k: pooled([r["suppressions"][k] for r in records]) for k in keys}
                    summary_deltas = {k: contrasts(v, summary["full"]) for k,v in summary.items()}
                    interventions.setdefault(regime,{})[arm] = dict(rows=summary, removed_minus_full=summary_deltas)
                if records[0]["probe"]:
                    probes[arm] = probe_summary(records)
            results[depth][regime] = rows
            diffs = {f"{m}_minus_physics_parity": contrasts(v, rows["physics_parity"])
                     for m,v in rows.items() if m != "physics_parity"}
            for global_arm, local_arm in (("pure","local"), ("parity","local_parity")):
                if global_arm in rows:
                    diffs[f"{global_arm}_minus_{local_arm}"] = contrasts(rows[global_arm], rows[local_arm])
            comparisons[depth][regime] = diffs
    latencies = {}
    for regime in SNR:
        path = JOBS / f"latency_r{regime}.json"
        latencies[regime] = ex.read(path)
        if latencies[regime]["implementation"] != signature:
            raise ValueError(f"Stale latency timing in R{regime}")
        digests[path.relative_to(ex.ROOT).as_posix()] = digest(path)
    config = dict(**asdict(CONFIG), updates=UPDATES, train_batch=TRAIN_BATCH,
                  validation_batch=EVAL_BATCH, primary_depth=PRIMARY, comparison_depth=2,
                  clipping="global gradient norm", optimizer="Adam",
                  checkpoints=[0,UPDATES//2,UPDATES], selection="fixed final update for every arm",
                  dev_pairs=PAIRS, streams=dict(train=5, score=6, r3_probe=8, r3_permutation=9),
                  probe_batch=PROBE_BATCH, snr_db=SNR)
    report = dict(config=config, implementation=signature, pooled=results, differences=comparisons,
                  suppression=interventions, kernels=probes, latency=latencies,
                  by_pair=list(by_pair.values()), receipt_sha256=digests)
    report["invariants"] = {}
    for path in invariant_paths:
        path = path.resolve()
        if not path.is_relative_to(ex.RESULTS) or path.suffix != ".json":
            raise ValueError(f"Expected an invariant JSON receipt under p6/results: {path}")
        receipt = ex.read(path)
        name = path.relative_to(ex.ROOT).as_posix()
        report["invariants"][name] = dict(sha256=digest(path), summary=receipt.get("summary", receipt))
    if quiet_paths:
        report["quiet_latency"] = {}
        for path in quiet_paths:
            path = path.resolve()
            if not path.is_relative_to(ex.RESULTS) or path.suffix != ".json":
                raise ValueError(f"Expected a quiet latency JSON receipt under p6/results: {path}")
            data = ex.read(path)
            if (data["implementation"] != signature or
                    data["supplement_sha256"] != digest(ex.ROOT / "quiet_latency.py") or
                    data["environment_sha256"] != digest(ex.ROOT / "latency_environment.py")):
                raise ValueError(f"Quiet latency source signature mismatch: {path}")
            for regime, row in data["regimes"].items():
                if regime in report["quiet_latency"] or regime not in ("1", "2", "3"):
                    raise ValueError(f"Duplicate or unknown quiet timing regime {regime}")
                report["quiet_latency"][regime] = dict(**row, config=data["config"], environment=data["environment"],
                                                      receipt=path.relative_to(ex.ROOT).as_posix(),
                                                      receipt_sha256=digest(path))
    dest = ex.RESULTS / "dev_iter2.json"
    ex.save(dest, report)
    print(f"{dest} SHA256={digest(dest)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(dest="action", required=True)
    fits = cmds.add_parser("fit")
    fits.add_argument("--workers", default=2, type=int)
    fits.add_argument("--regime", nargs="+", choices=tuple(SNR), type=int, default=tuple(SNR))
    fits.add_argument("--pair", nargs="+", choices=PAIRS, default=PAIRS, type=int)
    fits.add_argument("--depth", nargs="+", choices=DEPTHS, default=DEPTHS, type=int)
    fits.add_argument("--arm", nargs="+", choices=("references",*ex.ARMS), default=("references",*ex.ARMS))
    timer = cmds.add_parser("latency")
    timer.add_argument("--repeats", default=5, type=int)
    timer.add_argument("--regime", nargs="+", choices=tuple(SNR), type=int, default=tuple(SNR))
    summary = cmds.add_parser("summarize")
    summary.add_argument("--invariants", nargs="+", type=Path, default=())
    summary.add_argument("--quiet-latency", nargs="+", type=Path, default=())
    args = parser.parse_args()
    ex.setup()
    if args.action == "summarize":
        summarize(args.invariants, args.quiet_latency)
    elif args.action == "latency":
        from p6.latency_environment import benchmark_environment

        if args.repeats < 1:
            parser.error("latency requires at least one repeat")
        with benchmark_environment() as environment:
            for regime in args.regime:
                latency(regime, args.repeats, environment)
    else:
        if args.workers < 1:
            parser.error("workers must be positive")
        jobs = [k for k in job_keys() if k[0] in args.regime and k[1] in args.pair
                and k[2] in args.depth and k[3] in args.arm]
        if not jobs:
            parser.error("no development job selected")
        if args.workers == 1:
            for key in jobs:
                run_job(key)
        else:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=ex.setup) as workers:
                for _ in workers.map(run_job, jobs, chunksize=1):
                    pass


if __name__ == "__main__":
    main()
