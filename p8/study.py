"""Run the fixed paired cohort. Each JSONL row is an independently replayable link."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from .coding import decode
from .link import B_ROWS, CORE, FRAME_DURATION, N0, PILOT_ROWS, PROBE_ROW, PROBES, row_mask, simulate
from .tracker import advance, observed_taps, prior, readout, update


BASE = 802611190000
CLUSTERS, LINKS = 10, 32
METHODS = ("carry", "reset", "mismatch")
RESULTS = Path(__file__).parent / "results"


def history(y):
    return advance(update(prior(), PILOT_ROWS, observed_taps(y, PILOT_ROWS)), FRAME_DURATION)


def score(llrs, prediction, variance, target, clean, words, rows):
    payload = llrs[CORE].reshape(18, 128)
    wrong = decode(payload) != words.info[:18]
    error = abs(target - prediction)**2
    if not all(np.isfinite(x).all() for x in (llrs, prediction, variance)) or (variance <= 0).any():
        raise ValueError("Nonfinite output or nonpositive probe variance")
    extra = (~CORE) & ~row_mask((*rows, PROBE_ROW))
    extra_errors = 0
    if extra.any():
        extra_errors = int(np.any(decode(llrs[extra]) != words.info[18]))
    return dict(
        blocks=18, block_errors=int(wrong.any(axis=1).sum()),
        first9_errors=int(wrong[:9].any(axis=1).sum()), last9_errors=int(wrong[9:].any(axis=1).sum()),
        coded_bits=payload.size, coded_errors=int(np.count_nonzero((payload > 0) != words.code[:18])),
        info_bits=wrong.size, info_errors=int(wrong.sum()),
        extra_blocks=int(extra.any()), extra_errors=extra_errors,
        probes=len(target), probe_sse=float(error.sum()), probe_energy=float((abs(target)**2).sum()),
        clean_sse=float((abs(clean-prediction)**2).sum()), clean_energy=float((abs(clean)**2).sum()),
        variance_sum=float(variance.sum()), nll_sum=float((np.log(variance/N0) + error/variance).sum()),
    )


def evaluate(job):
    cluster, link = job
    seed = BASE + 1000 * cluster + link
    pair = simulate(seed)
    states = {"carry": history(pair.a), "reset": advance(prior(), FRAME_DURATION),
              "mismatch": history(pair.donor_a)}
    metrics = {}
    for budget, rows in B_ROWS.items():
        y = pair.b[budget]
        # The scoring targets are materialized only on the evaluator side.
        blind = y.copy()
        blind[PROBES] = np.nan
        taps = observed_taps(blind, rows)
        metrics[budget] = {}
        for method, state in states.items():
            posterior = update(state, rows, taps)
            llrs, pred, variance = readout(posterior, blind, rows)
            metrics[budget][method] = score(llrs, pred, variance, y[PROBES],
                                             pair.clean_b[budget][PROBES], pair.b_words, rows)
    return dict(cluster=cluster, link=link, seed=seed, budgets=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "links.jsonl")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--clusters", nargs="+", type=int, default=list(range(1, CLUSTERS+1)))
    args = parser.parse_args()
    if (len(set(args.clusters)) != len(args.clusters) or
            any(i not in range(1, CLUSTERS+1) for i in args.clusters)):
        parser.error("Choose distinct formal clusters between 1 and 10")
    jobs = [(i, j) for i in args.clusters for j in range(LINKS)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".partial")
    with ProcessPoolExecutor(args.workers) as pool, temporary.open("w", encoding="utf-8") as out:
        for record in pool.map(evaluate, jobs, chunksize=4):
            out.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
            if record["link"] == LINKS - 1:
                print(f"Scored cluster {record['cluster']}", flush=True)
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
