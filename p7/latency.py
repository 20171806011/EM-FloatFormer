"""Isolated single-thread CPU latency benchmark for Phase 7 models vs comparators."""
import json
import statistics
import time

import torch

from p5 import classical
from p5.study import setup
from p5.wavefield_sim import simulate
from p6 import experiment as ex
from p6_followup.common import load_model
from p6_consistency.receiver import ObservationAwareReceiver
from p7.conditions import apply_observation_gain, mask_for_condition
from p7.freeze import REPO, RESULTS, SEED_BASE, verify_freeze
from p7.model import EM7HypothesisReceiver
from p7.train import load_checkpoint


@torch.no_grad()
def main():
    setup()
    freeze_sha = verify_freeze()
    record = json.loads((RESULTS / "formal" / "pair01" / "record.json").read_text(encoding="utf-8"))

    models = {
        "EM7-Axial": load_checkpoint(REPO / record["checkpoints"]["EM7-Axial"]["path"])[0],
        "EM7-Local": load_checkpoint(REPO / record["checkpoints"]["EM7-Local"]["path"])[0],
        "EM7-Fixed": EM7HypothesisReceiver("fixed").eval(),
        "EM7-SingleHyp-Axial": load_checkpoint(REPO / record["checkpoints"]["EM7-SingleHyp-Axial"]["path"])[0],
    }
    p6_model, _ = load_model(6, "parity", development=False)
    p6_rx = ObservationAwareReceiver(p6_model, use_variance=True, effective=True, cavity=True)

    arms = ("EM7-Axial", "EM7-Local", "EM7-Fixed", "EM7-SingleHyp-Axial", "P6-Aware-Consistent", "Waveform-DD")
    rows = []

    for batch_size in (1, 8):
        frame = simulate(2, batch_size, 9, SEED_BASE + 990000 + batch_size)
        for cond in ("C0_intact", "C4a_roving_window"):
            gain, _, _ = mask_for_condition(frame.obs.pilot_mask, cond, batch_size, seed=777 + batch_size)
            obs_c = apply_observation_gain(frame, gain).obs
            # P6-Aware-Consistent uses 2D [T,N] gain on C0, or burst loop on C4a (identical to eval_utils)
            timings = {a: [] for a in arms}
            for round_idx in range(12):  # 2 warmups + 10 interleaved timings
                for arm in arms:
                    t0 = time.perf_counter_ns()
                    if arm in models:
                        models[arm](obs_c, gain)
                    elif arm == "P6-Aware-Consistent":
                        if cond == "C0_intact":
                            p6_rx(obs_c, gain[0])
                        else:
                            from dataclasses import replace
                            for bi in range(batch_size):
                                sub = replace(obs_c, y=obs_c.y[bi:bi + 1], noise_var=obs_c.noise_var[bi:bi + 1])
                                p6_rx(sub, gain[bi])
                    else:
                        classical.receive(obs_c, "waveform_dd")
                    dt_ms = (time.perf_counter_ns() - t0) / 1e6
                    if round_idx >= 2:
                        timings[arm].append(dt_ms)

            for arm in arms:
                xs = timings[arm]
                q1, med, q3 = statistics.quantiles(xs, n=4, method="inclusive")
                rows.append(
                    dict(
                        batch_size=batch_size,
                        condition=cond,
                        arm=arm,
                        median_ms=statistics.median(xs),
                        q1_ms=q1,
                        q3_ms=q3,
                        min_ms=min(xs),
                        max_ms=max(xs),
                        samples_ms=xs,
                    )
                )

    out = dict(
        freeze_sha256=freeze_sha,
        threads=torch.get_num_threads(),
        warmups=2,
        rounds=10,
        scope="Receiver forward pass only (excludes simulation, truth calibration scoring, and final LDPC decode)",
        rows=rows,
    )
    ex.save(RESULTS / "latency.json", out)
    print("Isolated CPU Latency (1 thread, median ms/batch):", flush=True)
    for r in rows:
        print(f"  B={r['batch_size']} {r['condition']:18s} {r['arm']:22s} median={r['median_ms']:7.2f}ms [{r['q1_ms']:.2f}, {r['q3_ms']:.2f}]", flush=True)


if __name__ == "__main__":
    main()
