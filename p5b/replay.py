"""Rebuild every test frame, reload final checkpoints, and publish the replay receipt."""

import time
from datetime import datetime, timezone

import torch
from p5.study import PAIRS, save, setup
from p5b.study import RESULTS, replay


def main():
    setup()
    torch.set_num_interop_threads(1)
    start = time.perf_counter()
    rows = [replay(regime, pair) for regime, pairs in PAIRS.items()
            for pair in range(1, pairs + 1)]
    receipt = dict(
        completed_utc=datetime.now(timezone.utc).isoformat(), torch_version=str(torch.__version__),
        seconds=time.perf_counter() - start, threads=torch.get_num_threads(),
        checkpoints=sum(row["checkpoints"] for row in rows),
        score_rows=sum(row["score_rows"] for row in rows), test_batches=sum(row["test_batches"] for row in rows),
        exact_comparison="Saved vs regenerated field and coded-payload digests, complete score "
                         "dictionaries, parameter counts, archive comparisons, and R3 kernel probes.",
        pairs=rows, passed=all(row["passed"] for row in rows),
    )
    save(RESULTS / "checkpoint_replay.json", receipt)
    print(f"Verified {receipt['checkpoints']} checkpoints and {receipt['score_rows']} score rows.")


if __name__ == "__main__":
    main()
