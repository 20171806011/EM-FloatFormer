"""Evaluate the frozen cross-waveform cohort; truth is confined to scoring and the CSI diagnostic."""
from pathlib import Path

import numpy as np

from p8.coding import decode
from p8.link import FRAME_DURATION, N0
from p8.study import history, run_cohort
from p8.tracker import advance, prior
from .link import CORE_ROWS, LAYOUTS, PROBE_ROW, extra_rows, simulate
from .receiver import oracle, receive


BASE = 822611190000
METHODS = ("carry", "reset", "mismatch", "oracle")
RESULTS = Path(__file__).parent / "results"


def score(output, target, clean, traffic, training_rows):
    llrs, pred, var = output
    if not all(np.isfinite(x).all() for x in output) or np.any(var <= 0):
        raise ValueError("Invalid receiver output")
    core = llrs[list(CORE_ROWS)].reshape(16, 128)
    wrong = decode(core) != traffic.info[:16]
    errors = wrong.any(axis=1)
    free = list(extra_rows(training_rows))
    extra_errors = 0
    if free:
        extra_wrong = decode(llrs[free].reshape(len(free), 128)) != traffic.info[16:16+len(free)]
        extra_errors = int(extra_wrong.any(axis=1).sum())
    residual = abs(target - pred)**2
    return dict(blocks=len(core), block_errors=int(errors.sum()),
                first8_errors=int(errors[:8].sum()), last8_errors=int(errors[8:].sum()),
                coded_bits=core.size, coded_errors=int(((core > 0) != traffic.code[:16]).sum()),
                info_bits=wrong.size, info_errors=int(wrong.sum()),
                extra_blocks=len(free), extra_errors=extra_errors, probes=len(target),
                probe_sse=float(residual.sum()), probe_energy=float((abs(target)**2).sum()),
                clean_sse=float((abs(clean-pred)**2).sum()), clean_energy=float((abs(clean)**2).sum()),
                variance_sum=float(var.sum()), nll_sum=float((np.log(var/N0)+residual/var).sum()))


def evaluate(job):
    cluster, index = job
    seed = BASE + 1000*cluster + index
    pair = simulate(seed)
    states = dict(carry=history(pair.a), reset=advance(prior(), FRAME_DURATION),
                  mismatch=history(pair.donor_a))
    budgets = {}
    for layout, rows in LAYOUTS.items():
        y = pair.b[layout]
        blind = y.copy()
        blind[PROBE_ROW] = np.nan
        cells = {}
        for arm in METHODS:
            output = oracle(pair.channel, blind, rows) if arm == "oracle" else receive(states[arm], blind, rows)
            cells[arm] = score(output, y[PROBE_ROW], pair.clean_b[layout][PROBE_ROW], pair.b_words, rows)
        budgets[layout] = cells
    return dict(cluster=cluster, link=index, seed=seed, budgets=budgets)


if __name__ == "__main__":
    run_cohort(evaluate, RESULTS / "links.jsonl", __doc__)
