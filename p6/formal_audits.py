"""Predeclared held-out symmetry, physical interventions and isolated serial timings."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
import math
import os
from pathlib import Path
from random import Random
import statistics
import time
from unittest.mock import patch

import psutil
import torch

from p5 import study as p5
from p5.statistics import COUNT_FIELDS
from p5b.diagnostics import SCALES, erased, gain_check
from p6 import axial_em_transformer as solver, experiment as ex, formal_study as formal
from p6.diagnostics import degenerate
from p6.extended_audit import summarize
from p6.handle_ablation import assert_replay
from p6.latency_environment import benchmark_environment, interval, snapshot
from p6.study import metrics


GAUGE_ARMS = ("pure", "parity")
GAUGE_BATCH = 950
WARMUPS, ROUNDS = 2, 7


def gauge_pair(r, p, fitted):
    obs = formal.draw(r, p, 7, 2, GAUGE_BATCH).obs
    zero_noise = replace(obs, noise_var=torch.zeros_like(obs.noise_var))
    edge = {name: erased(obs, name) for name in (
        "intact", "first_pilot_row", "last_pilot_row", "all_pilots", "whole_field")}
    edge.update(degenerate_pilots=degenerate(obs), zero_noise=zero_noise,
                physical_noiseless=formal.draw(r, p, 7, 2, GAUGE_BATCH, snr=math.inf).obs,
                all_pilots_zero_noise=erased(zero_noise, "all_pilots"),
                whole_zero_noise=erased(zero_noise, "whole_field"))
    rows = []
    for arm in GAUGE_ARMS:
        for i, (name, observation) in enumerate(edge.items()):
            seed = p5.rng(r, 7, p, GAUGE_BATCH + 1 + i)
            result = gain_check(fitted[arm], observation, SCALES, seed)
            rows.append(dict(regime=r, pair=p, arm=arm, design="formal_A", condition=name, **result))
    return dict(observation_seed=p5.rng(r, 7, p, GAUGE_BATCH), rows=rows, summary=summarize(rows))


def suppression_labels(r):
    labels = {1: ["cpe", "radial", "radial_and_cpe", "delta_h", "dewarp", "all"],
              2: ["phase", "bank", "delta_h", "all"],
              3: ["radial", "delays", "spatial", "spatial_operator", "irc", "ula", "delta_h", "all"]}
    return labels[r]


def suppress(model, label, stack):
    if label == "dewarp":
        stack.enter_context(patch.object(model, "pa_dewarp", False))
    elif label == "irc":
        stack.enter_context(patch.object(model, "spatial_irc", False))
    elif label == "ula":
        stack.enter_context(patch.object(model, "spatial_ula", False))
    elif label == "spatial_operator":
        for head in (model.head, model.initial_map):
            stack.enter_context(patch.object(head, "disable_spatial", True))
    else:
        original = solver._physical_handles
        fields = {"all": solver.Handles._fields, "radial_and_cpe": ("radial", "cpe")}.get(label, (label,))

        def masked(fields_arg, h, amp, mean_rot, cpe, nominal, regime, x, bank_residual=None):
            if label == "bank":
                return original(fields_arg, h, amp, mean_rot, cpe, nominal, regime, x, None)
            values = original(fields_arg, h, amp, mean_rot, cpe, nominal, regime, x, bank_residual)
            return values._replace(**{f: torch.zeros_like(getattr(values, f)) for f in fields})

        stack.enter_context(patch.object(solver, "_physical_handles", masked))


def interventions_pair(r, p, fitted, record):
    frames = [formal.draw(r, p, 7, formal.EVAL_BATCH, i) for i in range(formal.TEST_BATCHES)]
    draws = [dict(seed=p5.rng(r, 7, p, i), batch=formal.EVAL_BATCH,
                  field_sha256=formal.p5b.tensor_hash(frame.obs.y),
                  codewords_sha256=formal.p5b.tensor_hash(frame.bits)) for i, frame in enumerate(frames)]
    if draws != record["frames"]:
        raise ValueError("Intervention draws differ from registered scoring")
    fixed = formal.model(r, p, "physics_parity").eval()
    reference = ex.score(frames, lambda f: fixed(f.obs))
    arms = {}
    for arm in ex.PARITY_ARMS:
        trained = fitted[arm]
        full = ex.score(frames, lambda f: trained(f.obs))
        if any(full[k] != record["scores"][arm][k] for k in COUNT_FIELDS):
            raise ValueError(f"Full intervention replay integer mismatch: R{r}/{p}/{arm}")
        scores = {"full": full}
        for label in suppression_labels(r):
            with ExitStack() as stack:
                suppress(trained, label, stack)
                scores[label] = ex.score(frames, lambda f: trained(f.obs))
        assert_replay(scores["all"], reference)
        arms[arm] = scores
    return dict(frames=draws, arms=arms, fixed_parity=reference)


def audit_pair(job):
    action, r, p = job
    signature, _ = formal.contract()
    if p not in formal.COHORTS["A"][r]:
        raise ValueError("Only the primary cohort receives post-fit interventions and symmetry audits")
    folder = formal.folder(r, p)
    fitted = formal.restored(r, p)
    record = ex.read(folder / "record.json")
    if record["freeze_sha256"] != signature:
        raise ValueError("Formal scoring record belongs to a different freeze")
    digests = ex.read(folder / "fit.json")["checkpoints"]
    key = "gauge" if action == "gauge" else "interventions"
    dest = folder / f"audit_{key}.json"
    if dest.exists():
        existing = ex.read(dest)
        if (existing["freeze_sha256"] == signature and existing["checkpoints"] == digests
                and existing["regime"] == r and existing["pair"] == p):
            return
    details = gauge_pair(r, p, fitted) if action == "gauge" else interventions_pair(r, p, fitted, record)
    ex.save(dest, dict(freeze_sha256=signature, regime=r, pair=p, checkpoints=digests, **details))
    print(f"{action} R{r}/{p}", flush=True)


def aggregate(action):
    signature, _ = formal.contract()
    pairs = [(r, p) for r, p in formal.pairs() if p in formal.COHORTS["A"][r]]
    entries = [ex.read(formal.folder(r, p) / f"audit_{action}.json") for r, p in pairs]
    if len(pairs) != 30 or any(
            (row["regime"], row["pair"], row["freeze_sha256"]) != (r,p,signature)
            for (r, p), row in zip(pairs, entries, strict=True)):
        raise ValueError("Incomplete cohort A audit coverage or mismatched freeze")
    result = dict(freeze_sha256=signature, pairs=len(pairs))
    if action == "gauge":
        rows = [row for cell in entries for row in cell["rows"]]
        totals = summarize(rows)
        if len(rows) != 600 or len({(r["regime"],r["pair"],r["arm"],r["condition"]) for r in rows}) != 600:
            raise ValueError("Unexpected gauge coverage")
        result.update(checkpoints=60, arms=list(GAUGE_ARMS), scales=list(SCALES), summary=totals,
                      failed_cases=[r for r in rows if not r["passed"]],
                      by_pair=[{k:c[k] for k in ("regime","pair","summary","checkpoints")} for c in entries])
        ex.save(ex.RESULTS / "trained_gauge.json", result)
    else:
        pooled = {}
        for r in formal.COHORTS["A"]:
            pooled[r] = {}
            for arm in ex.PARITY_ARMS:
                group = [c["arms"][arm] for c in entries if c["regime"] == r]
                scores = {}
                for label in ("full", *suppression_labels(r)):
                    counts = {}
                    for g in group:
                        p5.add(counts, g[label])
                    scores[label] = metrics(counts)
                    scores[label]["delta_errors_vs_full"] = counts["block_errors"] - scores["full"]["block_errors"]
                pooled[r][arm] = scores
        result.update(arms=list(ex.PARITY_ARMS), pooled=pooled, by_pair=entries)
        ex.save(ex.RESULTS / "interventions.json", result)


def project_scan():
    """Check interpreter workdirs as well as module and script paths, including spawned children."""
    root = ex.ROOT.parent.resolve()
    others, unreadable = [], []
    for proc in psutil.process_iter(["name"]):
        if proc.pid == os.getpid() or not (proc.info["name"] or "").lower().startswith(("python", "pypy")):
            continue
        try:
            cwd = Path(proc.cwd()).resolve()
            argv = [a.lower().replace("\\", "/") for a in proc.cmdline()[1:]]
        except (psutil.NoSuchProcess, FileNotFoundError):
            continue
        except (psutil.AccessDenied, PermissionError):
            unreadable.append(proc.pid)
            continue
        if cwd.is_relative_to(root) or any(
                a.startswith(("p6.", "p5.", "p5b.", "publication.verify_p4"))
                or root.as_posix().lower() in a for a in argv):
            others.append(proc.pid)
    if others:
        raise RuntimeError(f"Concurrent project Python processes during latency measurement: {others}")
    return dict(other_project_python=[], unreadable_python_pids=unreadable)


@torch.inference_mode()
def latency():
    signature, _ = formal.contract()
    result = dict(freeze_sha256=signature, threads=torch.get_num_threads(),
                  interop_threads=torch.get_num_interop_threads(), warmup_rounds=WARMUPS,
                  measured_rounds=ROUNDS, regimes={})
    if result["threads"] != 1 or result["interop_threads"] != 1:
        raise ValueError("Latency requires one intra-op and one inter-op thread")
    result["initial_project_scan"] = project_scan()
    with benchmark_environment() as host:
        result["host"] = host
        for r, cohort_pairs in formal.COHORTS["A"].items():
            p = cohort_pairs[0]
            networks = formal.restored(r, p)
            networks.update({a:formal.model(r,p,a).eval() for a in ex.FIXED})
            methods = {name:(lambda f, m=m: m(f.obs)) for name, m in networks.items()}
            for name in ex.METHODS[r]:
                methods[name] = lambda f, name=name: p5.classical(f, name)
            batches = {}
            for index, b in enumerate((1,64)):
                f = formal.draw(r,p,7,b,970+index)
                randomized = Random(p5.rng(r,7,p,972+index))
                timing = {name:[] for name in methods}
                rounds = []
                for rnd in range(WARMUPS+ROUNDS):
                    order = list(methods)
                    randomized.shuffle(order)
                    scan = project_scan()
                    start_host = snapshot()
                    for name in order:
                        start = time.perf_counter_ns()
                        output = methods[name](f)
                        ms = (time.perf_counter_ns()-start) / 1e6
                        del output
                        if rnd >= WARMUPS:
                            timing[name].append(ms)
                    if rnd >= WARMUPS:
                        rounds.append(dict(project_scan=scan, **interval(start_host, snapshot())))
                timing_rows = {}
                for name, values in timing.items():
                    quartiles = statistics.quantiles(values, n=4, method="inclusive")
                    timing_rows[name] = dict(median_ms=statistics.median(values), q1_ms=quartiles[0],
                                             q3_ms=quartiles[2], samples_ms=values)
                batches[b] = dict(seed=p5.rng(r,7,p,970+index), methods=timing_rows, rounds=rounds,
                                  background_percent_median=statistics.median(
                                      row["background_percent_of_host"] for row in rounds))
            result["regimes"][r] = dict(pair=p,batches=batches)
    result["final_project_scan"] = project_scan()
    ex.save(ex.RESULTS / "latency.json", result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("gauge", "interventions", "latency"))
    parser.add_argument("--workers", type=int, choices=range(1,9), default=6)
    args = parser.parse_args()
    ex.setup()
    formal.contract()
    if args.action == "latency":
        latency()
    else:
        jobs = [(args.action,r,p) for r,p in formal.pairs() if p in formal.COHORTS["A"][r]]
        with ProcessPoolExecutor(max_workers=args.workers, initializer=ex.setup) as pool:
            list(pool.map(audit_pair,jobs,chunksize=1))
        aggregate(args.action)
