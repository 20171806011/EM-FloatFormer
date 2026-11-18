"""Low-background supplement to the Iteration-2 parity receiver timings."""

import argparse
import hashlib
import os
from pathlib import Path
import random
import statistics
import time

import psutil
import torch

from p6 import experiment as ex
from p6.data import SNR, dev_frame, dev_seed
from p6.latency_environment import benchmark_environment, interval, project_workers, snapshot
from p6.run_dev_iter2 import implementation, restored


PAIR = 901
REQUIRED = 3
LIMIT = 5.
PRE_WINDOW = .5
WINDOW_SECONDS = 90.
REPEATS = {1: 16, 64: 2}
ARMS = ("parity", "local_parity")
DEST = ex.RESULTS / "iter2_quiet_latency.json"


def sample_cpu(siblings):
    cpus = psutil.cpu_times(percpu=True)
    return snapshot(), {i: (cpus[i].user + cpus[i].system, cpus[i].idle) for i in siblings}


def load(start, end):
    data = interval(start[0], end[0])
    same_core = {}
    for idx, (initial_busy, initial_idle) in start[1].items():
        busy = end[1][idx][0] - initial_busy
        total = busy + end[1][idx][1] - initial_idle
        same_core[str(idx)] = 100 * busy / total if total > 0 else None
    data["same_physical_core_logical_busy_percent"] = same_core
    return data


def scan():
    result = project_workers()
    for p in psutil.process_iter(["name"]):
        if p.pid == os.getpid() or not (p.info["name"] or "").lower().startswith("python"):
            continue
        try:
            args = [a.lower().replace("\\", "/") for a in p.cmdline()[1:]]
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if any(a == "p6.quiet_latency" or a.endswith("/p6/quiet_latency.py")
               or a == "p6/quiet_latency.py" for a in args):
            raise RuntimeError(f"Another low-background benchmark process is active: {p.pid}")
    return result


@torch.inference_mode()
def measure(regime, env):
    choice = env["selected_cpu_set"]
    if not choice or env["effective_affinity"] != [choice["logical"]]:
        raise RuntimeError("Fast-core affinity could not be verified")
    siblings = [c["logical"] for c in env["candidate_cpu_sets"] if c["core"] == choice["core"]]
    frames = {b: dev_frame(regime, 6, PAIR, b, 950 + b // 64) for b in REPEATS}
    models = {arm: restored(regime, PAIR, 3, arm) for arm in ARMS}
    cells = [(b, arm) for b in REPEATS for arm in ARMS]
    accepted = {key: [] for key in cells}
    attempts = []
    for b, arm in cells:
        scan()
        models[arm](frames[b].obs)
    order = random.Random(dev_seed(regime, 6, PAIR, 959))
    start_search = time.monotonic()
    deadline = start_search + WINDOW_SECONDS
    stopped = None
    while (not stopped and any(len(v) < REQUIRED for v in accepted.values())
           and time.monotonic() + PRE_WINDOW < deadline):
        order.shuffle(cells)
        for b, arm in cells:
            if time.monotonic() + PRE_WINDOW >= deadline:
                break
            if len(accepted[(b, arm)]) >= REQUIRED:
                continue
            attempt = dict(batch=b, arm=arm, admitted=False, sample_ms=None, forwards=0)
            attempts.append(attempt)
            try:
                check = scan()
            except RuntimeError as exc:
                stopped = str(exc)
                attempt.update(reason="concurrent_benchmark_or_audit", scan_error=stopped)
                break
            attempt["unreadable_python_pids"] = check["unreadable_python_pids"]
            start = sample_cpu(siblings)
            time.sleep(PRE_WINDOW)
            pre = load(start, sample_cpu(siblings))
            attempt["pre_window"] = pre
            if pre["background_percent_of_host"] > LIMIT:
                attempt["reason"] = "pre_window_above_limit"
                continue
            with_frames = frames[b].obs
            receiver = models[arm]
            repeats = REPEATS[b]
            start = sample_cpu(siblings)
            cpu_start = time.process_time_ns()
            wall_start = time.perf_counter_ns()
            for _ in range(repeats):
                receiver(with_frames)
            elapsed_ms = (time.perf_counter_ns() - wall_start) / 1e6
            cpu_ms = (time.process_time_ns() - cpu_start) / 1e6
            sample = load(start, sample_cpu(siblings))
            try:
                scan()
            except RuntimeError as exc:
                stopped = str(exc)
                attempt.update(reason="concurrent_benchmark_or_audit", scan_error=stopped,
                               sample_window=sample, sample_ms=elapsed_ms / repeats, forwards=repeats)
                break
            attempt.update(sample_window=sample, sample_ms=elapsed_ms / repeats,
                           group_wall_ms=elapsed_ms, group_cpu_ms=cpu_ms, forwards=repeats)
            if sample["background_percent_of_host"] > LIMIT:
                attempt["reason"] = "sample_window_above_limit"
                continue
            attempt.update(admitted=True, reason="both_windows_within_limit")
            accepted[(b, arm)].append(attempt["sample_ms"])
    rows = {}
    for (b, arm), values in accepted.items():
        complete = len(values) == REQUIRED
        rows.setdefault(str(b), {})[arm] = dict(complete=complete, accepted=len(values),
                                                samples_ms=values,
                                                median_ms=statistics.median(values) if complete else None)
    return dict(pair=PAIR, regime=regime, depth=3, stream=6, rows=rows, attempts=attempts,
                measured_search_seconds=time.monotonic()-start_search,
                logical_siblings=siblings, concurrent_stop_reason=stopped,
                complete=all(len(v) == REQUIRED for v in accepted.values()),
                seeds={b: dev_seed(regime, 6, PAIR, 950 + b // 64) for b in REPEATS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", nargs="+", choices=tuple(SNR), type=int, default=tuple(SNR))
    args = parser.parse_args()
    ex.setup()
    scan()
    with benchmark_environment() as env:
        config = dict(full_host_background_limit_percent=LIMIT, pre_window_seconds=PRE_WINDOW,
                      time_budget_seconds_per_regime=WINDOW_SECONDS, required_samples_per_cell=REQUIRED,
                      forwards_per_timed_group=REPEATS, warmups_per_cell=1, torch_threads=torch.get_num_threads(),
                      torch_interop_threads=torch.get_num_interop_threads(), regimes=args.regime,
                      admission="Pre and sample windows must both pass. Wall time is excluded from admission.",
                      incomplete="Incomplete cells include individual admitted samples and omit the median.")
        result = dict(config=config, environment=env, implementation=implementation(),
                      supplement_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      environment_sha256=hashlib.sha256(
                          (ex.ROOT / "latency_environment.py").read_bytes()).hexdigest(), regimes={})
        for regime in args.regime:
            result["regimes"][str(regime)] = measure(regime, env)
            ex.save(DEST, result)
            statuses = {b: {a: v["accepted"] for a,v in arms.items()}
                        for b,arms in result["regimes"][str(regime)]["rows"].items()}
            print(f"R{regime} admitted counts {statuses}", flush=True)


if __name__ == "__main__":
    main()
