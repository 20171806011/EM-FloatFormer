"""Interleaved receiver-only CPU timing, run after all competing jobs finish."""
import statistics
import time

import torch

from p6 import experiment as ex
from p6.data import dev_frame
from p6_softinfo.receiver import SoftInformationReceiver
from p6_consistency.common import (METHODS, RESULTS, ROOT, corrupt, environment,
                                 save, sha, tensor_hash, verify_freeze)
from p6_consistency.observation import observation_gain
from p6_consistency.receiver import ObservationAwareReceiver
from p6_consistency.study import models_for


@torch.inference_mode()
def main():
    ex.setup()
    destination = RESULTS/"latency.json"
    if destination.exists():
        raise FileExistsError("Do not overwrite latency evidence")
    signature = verify_freeze()
    models, receipts = models_for(901, development=True)
    actual = models["legacy_mean"].model
    models["legacy_mean"] = actual
    models["legacy_variance"] = SoftInformationReceiver(actual, use_variance=True, refine=False)
    names = list(METHODS)
    rows, frames = [], []
    warmup, repeats = 2, 10
    for batch_size in (1, 8):
        original = dev_frame(2, 6, 901, batch_size)
        frames.append(dict(batch_size=batch_size, y_sha256=tensor_hash(original.obs.y)))
        for condition in ("intact", "scattered_erased"):
            obs = corrupt(original, condition).obs
            gain = observation_gain(obs.pilot_mask, condition)
            samples = {name: [] for name in names}
            for repeat in range(warmup+repeats):
                order = names[repeat % len(names):]+names[:repeat % len(names)]
                for name in order:
                    start = time.perf_counter_ns()
                    candidate = models[name]
                    candidate(obs, gain) if isinstance(candidate, ObservationAwareReceiver) else candidate(obs)
                    elapsed = (time.perf_counter_ns()-start)/1e6
                    if repeat >= warmup:
                        samples[name].append(elapsed)
            for name, times in samples.items():
                quartiles = statistics.quantiles(times, n=4, method="inclusive")
                rows.append(dict(batch_size=batch_size, condition=condition, method=name,
                    median_ms=statistics.median(times), p25_ms=quartiles[0], p75_ms=quartiles[2], samples_ms=times))
            print(dict(batch_size=batch_size, condition=condition,
                median_ms={name: round(statistics.median(times), 3) for name, times in samples.items()}), flush=True)
    save(destination, dict(freeze_sha256=signature, checkpoints=receipts, environment=environment(),
        frames=frames, rows=rows, warmup=warmup, repeats=repeats, script_sha256=sha(ROOT/"latency.py"),
        scope="Receiver inference including internal parity and diagnostics; excludes calibration scoring, simulation and final LDPC decode",
        comparison="One process/one Torch thread; cyclic interleaving; no concurrent study jobs; actual legacy models without trace adapter",
        limitations="Unoptimized dense research implementation, one CPU/runtime and short sample; no deployment or accelerator claim"))


if __name__ == "__main__":
    main()
