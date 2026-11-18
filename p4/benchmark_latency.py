"""Standardized Classical Receiver Latency Benchmark Harness.

Measures CPU (4 threads by default) and GPU (if available) execution latency across:
- Least Squares (LS)
- Truncated Delay Domain (DFT-8)
- Bussgang-LMMSE
- Iterative Decision-Directed PA Cancellation (DD-PA with 1, 2, and 3 iterations)

Outputs:
- p4/results/latency_cpu.csv
- p4/results/timing.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from p4.classical_baselines import (
    bussgang_lmmse,
    delay_est_dft,
    hard_slice_16qam,
    iterative_dd_pa_cancellation,
    ls_est,
)
from p4.simulator import DEFAULT_CP, DEFAULT_N, generate_frame


def benchmark_method(
    fn,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    warmup_runs: int = 30,
    timed_runs: int = 100,
    device: str = "cpu",
) -> Dict[str, float]:
    """Measure warmup and timed iterations, returning latency statistics in milliseconds."""
    # Warmup
    for _ in range(warmup_runs):
        _ = fn(*args, **kwargs)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    times: List[float] = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        _ = fn(*args, **kwargs)
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # Convert to milliseconds

    times.sort()
    n = len(times)
    p95_idx = int(math.ceil(0.95 * n)) - 1
    p95_idx = max(0, min(p95_idx, n - 1))

    return {
        "min_ms": times[0],
        "median_ms": statistics.median(times),
        "mean_ms": statistics.mean(times),
        "p95_ms": times[p95_idx],
        "max_ms": times[-1],
        "std_ms": statistics.stdev(times) if n > 1 else 0.0,
    }


def run_benchmarks(
    batch_sizes: Tuple[int, ...] = (1, 64),
    num_threads: int = 4,
    device: str = "cpu",
    output_dir: str = "p4/results",
    warmup_runs: int = 25,
    timed_runs: int = 100,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Execute latency benchmark matrix across methods and batch sizes."""
    torch.set_num_threads(num_threads)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    methods = [
        ("LS", lambda yp, yd, nv: hard_slice_16qam(yd / (ls_est(yp) + 1e-12))),
        ("DFT-8", lambda yp, yd, nv: hard_slice_16qam(yd / (delay_est_dft(yp, taps=8) + 1e-12))),
        ("Bussgang-LMMSE", lambda yp, yd, nv: bussgang_lmmse(yp, yd, nv, A=1.25, est_method="dft8")),
        ("DD-PA-Iter1", lambda yp, yd, nv: iterative_dd_pa_cancellation(yp, yd, nv, A=1.25, iterations=1, est_method="dft8")),
        ("DD-PA-Iter2", lambda yp, yd, nv: iterative_dd_pa_cancellation(yp, yd, nv, A=1.25, iterations=2, est_method="dft8")),
        ("DD-PA-Iter3", lambda yp, yd, nv: iterative_dd_pa_cancellation(yp, yd, nv, A=1.25, iterations=3, est_method="dft8")),
    ]

    records: List[Dict[str, Any]] = []

    print(f"=== Classical Latency Benchmark (Threads: {num_threads}, Device: {device}) ===")
    for b in batch_sizes:
        print(f"\n--- Batch Size: {b} ---")
        # Prepare sample data
        g = torch.Generator().manual_seed(12345)
        frame = generate_frame(
            batch_size=b,
            n=DEFAULT_N,
            cp=DEFAULT_CP,
            profile="equal4",
            snr_db=10.0,
            pa_condition="strong",
            A=1.25,
            generator=g,
            device=device,
        )
        yp = frame.yp
        yd = frame.yd
        nv = frame.noise_var

        for name, fn in methods:
            stats = benchmark_method(
                fn,
                args=(yp, yd, nv),
                kwargs={},
                warmup_runs=warmup_runs,
                timed_runs=timed_runs,
                device=device,
            )
            rec = {
                "method": name,
                "batch_size": b,
                "threads": num_threads,
                "device": device,
                "warmup_runs": warmup_runs,
                "timed_runs": timed_runs,
                **stats,
            }
            records.append(rec)
            print(
                f"  {name:<16} | Median: {stats['median_ms']:6.3f} ms | "
                f"P95: {stats['p95_ms']:6.3f} ms | Mean: {stats['mean_ms']:6.3f} ms"
            )

    # Save to CSV
    csv_file = out_path / f"latency_{device}.csv"
    fieldnames = [
        "method",
        "batch_size",
        "threads",
        "device",
        "warmup_runs",
        "timed_runs",
        "min_ms",
        "median_ms",
        "mean_ms",
        "p95_ms",
        "max_ms",
        "std_ms",
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
    print(f"\n[OK] Latency CSV saved to {csv_file}")

    # Save to JSON
    json_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "device": device,
        "num_threads": num_threads,
        "results": records,
    }
    json_file = out_path / "timing.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(json_meta, f, indent=2)
    print(f"[OK] Timing JSON saved to {json_file}")

    return records, json_meta


if __name__ == "__main__":
    import math

    parser = argparse.ArgumentParser(description="Stage P4-0 Latency Benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 64], help="Batch sizes to benchmark")
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations")
    parser.add_argument("--runs", type=int, default=100, help="Timed iterations")
    parser.add_argument("--output-dir", type=str, default="p4/results", help="Output directory")

    args = parser.parse_args()
    run_benchmarks(
        batch_sizes=tuple(args.batch_sizes),
        num_threads=args.threads,
        device=args.device,
        output_dir=args.output_dir,
        warmup_runs=args.warmup,
        timed_runs=args.runs,
    )
