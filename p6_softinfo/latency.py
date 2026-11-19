"""Interleaved receiver-only CPU timings; run without other experiment jobs."""
import statistics
import time

import torch

from p5.wavefield_sim import simulate
from p6 import experiment as ex
from p6_softinfo.common import (METHODS, RESULTS, ROOT, SEED_BASE, environment,
    save, sha, tensor_hash, verify_freeze)
from p6_softinfo.study import models_for


@torch.inference_mode()
def main():
    ex.setup()
    destination = RESULTS/"latency.json"
    if destination.exists():
        raise FileExistsError("Do not overwrite latency evidence")
    signature = verify_freeze()
    models, receipts = models_for(6, controls=False)
    names = list(METHODS[:5])
    rows, frames = [], []
    warmup, repeats = 2, 10
    for batch_size in (1, 8, 64):
        frame = simulate(2, batch_size, 9, SEED_BASE+900000+batch_size)
        frames.append(dict(batch_size=batch_size, y_sha256=tensor_hash(frame.obs.y)))
        samples = {name: [] for name in names}
        for repeat in range(warmup+repeats):
            order = names[repeat % len(names):]+names[:repeat % len(names)]
            for name in order:
                start = time.perf_counter_ns()
                models[name](frame.obs)
                elapsed_ms = (time.perf_counter_ns()-start)/1e6
                if repeat >= warmup:
                    samples[name].append(elapsed_ms)
        for name, times in samples.items():
            quartiles = statistics.quantiles(times, n=4, method="inclusive")
            rows.append(dict(batch_size=batch_size, method=name, median_ms=statistics.median(times),
                             p25_ms=quartiles[0], p75_ms=quartiles[2], samples_ms=times))
        print(dict(batch_size=batch_size, median_ms={name: round(statistics.median(times), 3)
                                                    for name, times in samples.items()}), flush=True)
    save(destination, dict(freeze_sha256=signature, checkpoints=receipts,
        environment=environment(), frames=frames, rows=rows, warmup=warmup, repeats=repeats,
        script_sha256=sha(ROOT/"latency.py"),
        scope="Receiver inference including internal parity and profile derivatives; excludes simulation and final LDPC decode",
        comparison="One process/one Torch thread; cyclic interleaving; no concurrent study jobs",
        limitations="One CPU/runtime and short sample; no deployment or accelerator claim"))


if __name__ == "__main__":
    main()
