"""Audit the stored cohorts and compare a fresh process replay against the raw records."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.stats import t


ROOT = Path(__file__).parent / "results"


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=2e-12, abs_tol=2e-12):
        raise AssertionError((actual, expected))


def audit(prefix, replay):
    raw = ROOT / (prefix + "links.jsonl")
    summary_file = ROOT / (prefix + "summary.json")
    lines = raw.read_bytes().splitlines()
    rows = [json.loads(line) for line in lines]
    summary = json.loads(summary_file.read_text())
    keys = [(r["cluster"], r["link"]) for r in rows]
    assert keys == list(itertools.product(range(1, 11), range(32)))
    base = 812611190000 if prefix else 802611190000
    assert all(r["seed"] == base + 1000*r["cluster"] + r["link"] for r in rows)
    for r in rows:
        energies = set()
        for layout, arms in r["budgets"].items():
            assert set(arms) == {"carry", "reset", "mismatch"}
            for cell in arms.values():
                assert cell["probes"] == 16 and cell["blocks"] == 18
                assert cell["coded_bits"] == 2304 and cell["info_bits"] == 1152
                assert cell["extra_blocks"] == int(layout != "dense")
                assert 0 <= cell["extra_errors"] <= cell["extra_blocks"]
                assert 0 <= cell["block_errors"] <= min(cell["info_errors"], 18)
                assert cell["info_errors"] <= 64*cell["block_errors"]
                assert cell["first9_errors"] + cell["last9_errors"] == cell["block_errors"]
                energies.add((cell["probe_energy"], cell["clean_energy"]))
        assert len(energies) == 1

    def ratio(cells, numerator, denominator):
        return math.fsum(c[numerator] for c in cells) / math.fsum(c[denominator] for c in cells)

    metrics = dict(bler=("block_errors", "blocks"), probe_nmse=("probe_sse", "probe_energy"),
                   clean_nmse=("clean_sse", "clean_energy"), nll=("nll_sum", "probes"),
                   calibration=("probe_sse", "variance_sum"), coded_ber=("coded_errors", "coded_bits"),
                   info_ber=("info_errors", "info_bits"))
    cluster_rates = {}
    for layout, arms in summary["totals"].items():
        for arm, total in arms.items():
            cells = [r["budgets"][layout][arm] for r in rows]
            for key in cells[0]:
                close(total[key], math.fsum(c[key] for c in cells))
            for name, fields in metrics.items():
                close(total[name], ratio(cells, *fields))
                values = np.array([ratio(cells[i:i+32], *fields) for i in range(0, 320, 32)])
                np.testing.assert_allclose(values, [g[layout][arm][name] for g in summary["clusters"]],
                                           rtol=2e-12, atol=2e-12)
                cluster_rates[layout, arm, name] = values
    focus = "split" if prefix else "sparse"
    flips = np.array(list(itertools.product((-1, 1), repeat=10)))
    raw_p = []
    for test in summary["tests"]:
        label, metric = test["control"], test["metric"]
        layout, arm = ("sparse", "carry") if label == "front-loaded carry" else (focus, label)
        diff = cluster_rates[layout, arm, metric] - cluster_rates[focus, "carry", metric]
        np.testing.assert_allclose(diff, test["differences"], atol=2e-12, rtol=2e-12)
        p = np.mean(abs(flips @ diff) >= abs(diff.sum()) - 1e-12)
        close(p, test["raw_p"])
        raw_p.append(p)
    order = np.argsort(raw_p)
    holm = np.minimum(1., np.maximum.accumulate(np.array(raw_p)[order] * np.arange(len(order), 0, -1)))
    for idx, value in zip(order, holm):
        close(summary["tests"][idx]["holm_p"], value)
    for arm, key in [("carry", "noninferiority"), ("reset", "reset_cross_budget_descriptive")]:
        if key in summary:
            diff = cluster_rates[focus, arm, "bler"] - cluster_rates["dense", "reset", "bler"]
            upper = diff.mean() + t.ppf(.95, 9)*diff.std(ddof=1)/np.sqrt(10)
            close(summary[key]["upper_95"], upper)
            assert bool(upper < .01) == summary[key]["passes"]
    selected = [s for s, r in zip(lines, rows) if r["cluster"] in (1, 10)]
    assert selected == replay.read_bytes().splitlines()
    digest = lambda blob: hashlib.sha256(blob).hexdigest()
    return dict(links=len(rows), cells=len(rows)*len(summary["totals"])*3, contrasts=len(raw_p),
                exact_sign_patterns=len(flips), selected_replay_links=len(selected),
                selected_all_fields_match=True, raw_sha256=digest(raw.read_bytes()),
                summary_sha256=digest(summary_file.read_bytes()),
                selected_records_lf_sha256=digest(b"\n".join(selected)+b"\n"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("primary_replay", type=Path)
    parser.add_argument("placement_replay", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "audit.json")
    args = parser.parse_args()
    result = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  primary=audit("", args.primary_replay), placement=audit("placement_", args.placement_replay))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
