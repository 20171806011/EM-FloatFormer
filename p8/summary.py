"""Recompute all pooled measurements and paired contrasts from link-level records."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from p7.statistics import exact_sign_test, holm_adjust
from .study import BASE, B_ROWS, CLUSTERS, LINKS, METHODS, RESULTS


def aggregate(records, budget, method):
    cells = [r["budgets"][budget][method] for r in records]
    v = {key: sum(c[key] for c in cells) for key in cells[0]}
    return dict(**v, bler=v["block_errors"]/v["blocks"],
                coded_ber=v["coded_errors"]/v["coded_bits"], info_ber=v["info_errors"]/v["info_bits"],
                probe_nmse=v["probe_sse"]/v["probe_energy"], clean_nmse=v["clean_sse"]/v["clean_energy"],
                nll=v["nll_sum"]/v["probes"], calibration=v["probe_sse"]/v["variance_sum"])


def summarize(records):
    expected = [(i, j, BASE+1000*i+j) for i in range(1, CLUSTERS+1) for j in range(LINKS)]
    if sorted((r["cluster"], r["link"], r["seed"]) for r in records) != expected:
        raise ValueError("Missing, duplicated or unexpected formal links")
    def table(group):
        return {b: {m: aggregate(group, b, m) for m in METHODS} for b in B_ROWS}
    totals = table(records)
    clusters = [table([r for r in records if r["cluster"] == i]) for i in range(1, CLUSTERS+1)]
    tests = []
    for control in ("reset", "mismatch"):
        for metric in ("bler", "probe_nmse"):
            differences = [g["sparse"][control][metric] - g["sparse"]["carry"][metric] for g in clusters]
            tests.append(dict(control=control, metric=metric, **exact_sign_test(differences)))
    # One-sided paired Student upper bound for the cross-budget noninferiority margin.
    differences = np.array([g["sparse"]["carry"]["bler"] - g["dense"]["reset"]["bler"] for g in clusters])
    upper = float(differences.mean() + t.ppf(.95, CLUSTERS-1) * differences.std(ddof=1) / np.sqrt(CLUSTERS))
    ni = dict(metric="sparse carry minus dense reset BLER", margin=.01, upper_95=upper,
              mean_diff=float(differences.mean()), differences=differences.tolist(),
              passes=upper < .01)
    return dict(totals=totals, tests=holm_adjust(tests), noninferiority=ni, clusters=clusters)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RESULTS / "links.jsonl")
    parser.add_argument("--output", type=Path, default=RESULTS / "summary.json")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text().splitlines()]
    result = summarize(records)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
