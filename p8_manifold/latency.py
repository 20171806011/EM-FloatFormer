"""Isolated single-thread CPU latency benchmark for Phase 8 Manifold receivers."""
import json
import statistics
import time

import torch

from p5.study import setup
from p6 import experiment as ex
from p8_manifold.freeze import FORMAL_ARMS, REPO, RESULTS, SEED_BASE, SNR_DB, verify_freeze
from p8_manifold.manifold_charts import offgrid_2d_lmmse
from p8_manifold.model import EM8ManifoldReceiver
from p8_manifold.offgrid_sim import simulate_offgrid
from p8_manifold.train import load_checkpoint


@torch.no_grad()
def main():
    setup()
    freeze_sha = verify_freeze()
    record = json.loads((RESULTS / "formal" / "pair01" / "record.json").read_text(encoding="utf-8"))

    models = {
        "EM8-Transport": load_checkpoint(REPO / record["checkpoints"]["EM8-Transport"]["path"])[0],
        "EM8-AbsNoTransport": load_checkpoint(REPO / record["checkpoints"]["EM8-AbsNoTransport"]["path"])[0],
        "EM8-Local": load_checkpoint(REPO / record["checkpoints"]["EM8-Local"]["path"])[0],
        "EM8-SingleHyp": load_checkpoint(REPO / record["checkpoints"]["EM8-SingleHyp"]["path"])[0],
        "EM8-Fixed": EM8ManifoldReceiver("fixed", d_model=8, n_layers=1, n_heads=2, steps=2).eval(),
    }

    rows = []
    for batch_size in (1, 8):
        for ci, cond in enumerate(("O0_random_sparse", "O1_roving_island")):
            frame = simulate_offgrid(batch_size, snr_db=SNR_DB, seed=SEED_BASE + 990000 + 10 * batch_size + ci, condition=cond)
            timings = {a: [] for a in FORMAL_ARMS}
            for round_idx in range(12):  # 2 warmups + 10 interleaved rounds
                for arm in FORMAL_ARMS:
                    t0 = time.perf_counter_ns()
                    if arm in models:
                        models[arm](frame.obs)
                    else:
                        offgrid_2d_lmmse(frame.obs)
                    dt_ms = (time.perf_counter_ns() - t0) / 1e6
                    if round_idx >= 2:
                        timings[arm].append(dt_ms)

            for arm in FORMAL_ARMS:
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
        scope="Receiver forward pass only (excludes simulation, truth calibration scoring, and LDPC decode)",
        rows=rows,
    )
    ex.save(RESULTS / "latency.json", out)
    print("Isolated CPU Latency (1 thread, median ms/batch):", flush=True)
    for r in rows:
        print(
            f"  B={r['batch_size']} {r['condition']:18s} {r['arm']:20s} "
            f"median={r['median_ms']:7.2f}ms [{r['q1_ms']:.2f}, {r['q3_ms']:.2f}]",
            flush=True,
        )


if __name__ == "__main__":
    main()
