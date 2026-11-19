"""Reduce P8-C link records to paired mechanism tests and the cross-budget comparison."""
import argparse
import json
from pathlib import Path

from p7.statistics import exact_sign_test, holm_adjust
from p8.study import CLUSTERS, LINKS
from p8.summary import aggregate, paired_bound
from .study import BASE, LAYOUTS, METHODS, RESULTS


def summarize(records):
    expected = [(i, j, BASE+1000*i+j) for i in range(1, CLUSTERS+1) for j in range(LINKS)]
    if sorted((r["cluster"], r["link"], r["seed"]) for r in records) != expected:
        raise ValueError("Incomplete or unexpected formal cohort")

    def table(group):
        return {b: {m: aggregate(group, b, m) for m in METHODS} for b in LAYOUTS}

    groups = [table([r for r in records if r["cluster"] == i]) for i in range(1, CLUSTERS+1)]
    tests = []
    for control in ("reset", "mismatch"):
        for metric in ("bler", "probe_nmse"):
            differences = [g["sparse"][control][metric]-g["sparse"]["carry"][metric] for g in groups]
            tests.append(dict(control=control, metric=metric, **exact_sign_test(differences)))
    return dict(totals=table(records), tests=holm_adjust(tests),
                noninferiority=paired_bound(groups, "sparse", "carry"), clusters=groups)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RESULTS/"links.jsonl")
    parser.add_argument("--output", type=Path, default=RESULTS/"summary.json")
    args = parser.parse_args()
    records = [json.loads(line) for line in args.input.read_text().splitlines()]
    args.output.write_text(json.dumps(summarize(records), indent=2, allow_nan=False)+"\n")
