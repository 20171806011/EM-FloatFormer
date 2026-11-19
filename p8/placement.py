"""Fresh paired replication with 16 acquisition pilots split between two B rows."""
import numpy as np

from . import link, tracker
from .study import RESULTS, history, run_cohort, score


PLACEMENT_BASE = 812611190000


def schedules():
    masks = {key: link.row_mask(rows) for key, rows in link.B_ROWS.items()}
    split = np.zeros_like(link.CORE)
    split[0, ::4] = True
    split[10, 2::4] = True
    masks["split"] = split
    return masks


def acquire(y, mask):
    rows = np.flatnonzero(mask.any(axis=1))
    counts = mask[rows].sum(axis=1)
    if np.array_equal(mask, link.row_mask(rows)):
        taps = tracker.observed_taps(y, rows)
    else:
        taps = []
        for row, count in zip(rows, counts):
            columns = np.flatnonzero(mask[row])
            basis = tracker.BASIS[:, columns]
            if not np.allclose(basis @ basis.conj().T / count, np.eye(len(link.DELAYS)), atol=1e-13):
                raise ValueError("This comb must separate the public delays orthogonally")
            taps.append((y[row, columns] / link.PILOTS[row, columns]) @ basis.conj().T / count)
        taps = np.asarray(taps)
    return rows, taps, tracker.TAP_NOISE * (len(link.EVEN) / counts)


def evaluate(job):
    cluster, index = job
    seed = PLACEMENT_BASE + 1000*cluster + index
    masks = schedules()
    pair = link.simulate(seed, masks)
    starts = {"carry": history(pair.a), "reset": tracker.advance(tracker.prior(), link.FRAME_DURATION),
              "mismatch": history(pair.donor_a)}
    metrics = {}
    for layout, mask in masks.items():
        y = pair.b[layout]
        blind = y.copy()
        blind[link.PROBES] = np.nan
        rows, taps, noise = acquire(blind, mask)
        pilots = mask | link.PROBES
        metrics[layout] = {}
        for method, state in starts.items():
            posterior = tracker.update(state, rows, taps, noise)
            llrs, pred, var = tracker.readout_mask(posterior, blind, pilots)
            metrics[layout][method] = score(llrs, pred, var, y[link.PROBES], pair.clean_b[layout][link.PROBES],
                                             pair.b_words, rows, pilot_mask=pilots)
    return dict(cluster=cluster, link=index, seed=seed, budgets=metrics)


if __name__ == "__main__":
    run_cohort(evaluate, RESULTS / "placement_links.jsonl", __doc__)
