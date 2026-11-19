"""Recompute all pooled measurements and paired contrasts from link-level records."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import t

from p7.statistics import exact_sign_test, holm_adjust
from .study import BASE, B_ROWS, CLUSTERS, LINKS, METHODS, RESULTS
from .placement import PLACEMENT_BASE, schedules


def aggregate(records, budget, method):
    cells = [r["budgets"][budget][method] for r in records]
    v = {key: sum(c[key] for c in cells) for key in cells[0]}
    return dict(**v, bler=v["block_errors"]/v["blocks"],
                coded_ber=v["coded_errors"]/v["coded_bits"], info_ber=v["info_errors"]/v["info_bits"],
                probe_nmse=v["probe_sse"]/v["probe_energy"], clean_nmse=v["clean_sse"]/v["clean_energy"],
                nll=v["nll_sum"]/v["probes"], calibration=v["probe_sse"]/v["variance_sum"])


def paired_bound(clusters, budget, method):
    differences = np.array([g[budget][method]["bler"] - g["dense"]["reset"]["bler"] for g in clusters])
    upper = float(differences.mean() + t.ppf(.95, CLUSTERS-1) * differences.std(ddof=1) / np.sqrt(CLUSTERS))
    return dict(metric=f"{budget} {method} minus dense reset BLER", margin=.01, upper_95=upper,
                mean_diff=float(differences.mean()), differences=differences.tolist(), passes=upper < .01)


def summarize(records, placement=False):
    base = PLACEMENT_BASE if placement else BASE
    budgets = schedules() if placement else B_ROWS
    focus = "split" if placement else "sparse"
    expected = [(i, j, base+1000*i+j) for i in range(1, CLUSTERS+1) for j in range(LINKS)]
    if sorted((r["cluster"], r["link"], r["seed"]) for r in records) != expected:
        raise ValueError("Missing, duplicated or unexpected formal links")
    def table(group):
        return {b: {m: aggregate(group, b, m) for m in METHODS} for b in budgets}
    totals = table(records)
    clusters = [table([r for r in records if r["cluster"] == i]) for i in range(1, CLUSTERS+1)]
    tests = []
    controls = [(focus, "reset", "reset"), (focus, "mismatch", "mismatch")]
    if placement:
        controls.append(("sparse", "carry", "front-loaded carry"))
    for budget, method, label in controls:
        for metric in ("bler", "probe_nmse"):
            differences = [g[budget][method][metric] - g[focus]["carry"][metric] for g in clusters]
            tests.append(dict(control=label, metric=metric, **exact_sign_test(differences)))
    result = dict(totals=totals, tests=holm_adjust(tests),
                  noninferiority=paired_bound(clusters, focus, "carry"), clusters=clusters)
    if placement:
        result["reset_cross_budget_descriptive"] = paired_bound(clusters, focus, "reset")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--placement", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prefix = "placement_" if args.placement else ""
    input_path = args.input or RESULTS / (prefix + "links.jsonl")
    output_path = args.output or RESULTS / (prefix + "summary.json")
    records = [json.loads(line) for line in input_path.read_text().splitlines()]
    result = summarize(records, placement=args.placement)
    output_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
