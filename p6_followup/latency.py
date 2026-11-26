"""Same-machine interleaved receiver timing; run with other jobs stopped."""
import statistics
import time

import torch

from p5.study import counts
from p5.wavefield_sim import simulate
from p6 import experiment as ex
from p6_followup.common import (AdaptedReceiver, RESULTS, ROOT, SEED_BASE, corrupt,
    environment, load_model, save, sha, tensor_hash, verify_freeze)


@torch.inference_mode()
def main():
    ex.setup()
    signature = verify_freeze()
    model, receipt = load_model(6)
    models = {"trust": AdaptedReceiver(model, "trust"),
              "smooth": AdaptedReceiver(model, "smooth"),
              "smooth_exact": AdaptedReceiver(model, "smooth", exact_emit=True)}
    rows, agreement = [], []
    names = list(models)
    warmup, repeats = 3, 15
    for batch in (1, 8, 64):
        frame = simulate(2, batch, 9, SEED_BASE + 900000 + batch)
        outputs = {name: current(frame.obs) for name, current in models.items()}
        # Compare optional algebraic reduction to the same smooth operator.
        llr, cfr = outputs["smooth"]
        new_llr, new_cfr = outputs["smooth_exact"]
        payload = ~frame.obs.pilot_mask
        agreement.append(dict(batch_size=batch, frame_sha256=tensor_hash(frame.obs.y),
                              max_llr_difference=float((llr-new_llr).abs().max()),
                              max_cfr_difference=float((cfr-new_cfr).abs().max()),
                              payload_hard_flips=int(((llr[:, payload]>0) !=
                                                     (new_llr[:, payload]>0)).sum()),
                              scores={name: counts(frame, *value) for name, value in outputs.items()}))
        samples = {name: [] for name in names}
        for repeat in range(warmup + repeats):
            # Cyclic Latin order balances each method's position in a round.
            order = names[repeat % len(names):] + names[:repeat % len(names)]
            for name in order:
                start = time.perf_counter_ns()
                models[name](frame.obs)
                elapsed = (time.perf_counter_ns()-start)/1e6
                if repeat >= warmup:
                    samples[name].append(elapsed)
        for name, times in samples.items():
            quartiles = statistics.quantiles(times, n=4, method="inclusive")
            rows.append(dict(batch_size=batch, method=name, median_ms=statistics.median(times),
                             p25_ms=quartiles[0], p75_ms=quartiles[2], samples_ms=times))
        print(batch, {name: round(statistics.median(times), 3) for name, times in samples.items()}, flush=True)
    save(RESULTS / "latency.json", dict(freeze_sha256=signature, checkpoint=receipt,
         environment=environment(), warmup=warmup, repeats=repeats, rows=rows,
         numerical_agreement=agreement, script_sha256=sha(ROOT/"latency.py"),
         scope="receiver inference including internal parity; excludes simulation and final external LDPC decode",
         comparison="same process, same observations, same checkpoint, cyclic interleaved order; no other study workers",
         limitations="One CPU/runtime and short timing series; medians are descriptive, not deployment guarantees."))


if __name__ == "__main__":
    main()
