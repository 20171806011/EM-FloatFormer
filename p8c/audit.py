"""Recompute P8-C aggregate counts, paired statistics, and compare fresh replay records."""
import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.stats import t


RESULTS = Path(__file__).parent / "results"
BASE = 822611190000
BUDGETS = ("sparse", "dense")
ARMS = ("carry", "reset", "mismatch", "oracle")
RATIOS = dict(bler=("block_errors", "blocks"), coded_ber=("coded_errors", "coded_bits"),
              info_ber=("info_errors", "info_bits"), probe_nmse=("probe_sse", "probe_energy"),
              clean_nmse=("clean_sse", "clean_energy"), nll=("nll_sum", "probes"),
              calibration=("probe_sse", "variance_sum"))


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def check_cell(cell, budget):
    expected = dict(blocks=16, info_bits=1024, coded_bits=2048, probes=32,
                    extra_blocks=2 if budget == "sparse" else 0)
    require(all(cell[k] == n for k, n in expected.items()), "Incorrect per-link exposure")
    for field, bound in (("first8_errors", 8), ("last8_errors", 8), ("coded_errors", 2048),
                         ("info_errors", 1024), ("extra_errors", expected["extra_blocks"])):
        require(type(cell[field]) is int and 0 <= cell[field] <= bound, f"Invalid {field}")
    failed = cell["block_errors"]
    require(failed == cell["first8_errors"] + cell["last8_errors"], "Half-frame totals disagree")
    require(failed <= cell["info_errors"] <= 64*failed, "Block/bit errors disagree")
    for key in ("probe_sse", "clean_sse", "probe_energy", "clean_energy", "variance_sum"):
        require(math.isfinite(cell[key]) and cell[key] >= 0., f"Invalid {key}")
    require(cell["probe_energy"] > 0 and cell["clean_energy"] > 0 and cell["variance_sum"] > 0,
            "Empty prediction exposure")
    require(math.isfinite(cell["nll_sum"]), "Nonfinite NLL")


def verify_group(records, reported):
    require(set(reported) == set(BUDGETS), "Missing summary budget")
    rates = {}
    for budget, arm in itertools.product(BUDGETS, ARMS):
        require(set(reported[budget]) == set(ARMS), "Missing summary arm")
        cells = [r["budgets"][budget][arm] for r in records]
        sums = {k: math.fsum(c[k] for c in cells) for k in cells[0]}
        metrics = {k: sums[num]/sums[den] for k, (num, den) in RATIOS.items()}
        table = reported[budget][arm]
        for key, value in (sums | metrics).items():
            require(math.isclose(table[key], value, rel_tol=2e-12, abs_tol=2e-12),
                    f"Aggregate mismatch {budget}/{arm}/{key}")
        rates[budget, arm] = metrics
    return rates


def verify_stats(groups, summary):
    def rates(budget, arm, metric):
        return np.array([group[budget, arm][metric] for group in groups])

    tests = {(r["control"], r["metric"]): r for r in summary["tests"]}
    contrasts = list(itertools.product(("reset", "mismatch"), ("bler", "probe_nmse")))
    require(len(summary["tests"]) == 4 and set(tests) == set(contrasts), "Incomplete Holm family")
    signs = np.array(list(itertools.product((-1., 1.), repeat=10)))
    p_values = []
    for control, metric in contrasts:
        test = tests[control, metric]
        diff = rates("sparse", control, metric) - rates("sparse", "carry", metric)
        np.testing.assert_allclose(test["differences"], diff, rtol=2e-12, atol=2e-12)
        threshold = abs(diff.sum()) - 1e-12*max(1., sum(abs(diff)))
        p = Fraction(int((abs(signs @ diff) >= threshold).sum()), len(signs))
        require(Fraction(test["raw_exact"]) == p and test["raw_p"] == float(p), "Raw p-value mismatch")
        p_values.append(p)
        observed = (test["wins"], test["ties"], test["losses"])
        expected = ((diff > 1e-12).sum(), (abs(diff) <= 1e-12).sum(), (diff < -1e-12).sum())
        require(observed == expected and test["n"] == 10, "Cluster sign counts disagree")
        np.testing.assert_allclose(test["mean_diff"], diff.mean(), atol=2e-12)
        half = t.ppf(.975, 9) * diff.std(ddof=1) / np.sqrt(10)
        np.testing.assert_allclose(test["ci95"], [diff.mean()-half, diff.mean()+half], atol=1e-10, rtol=1e-10)
    previous = Fraction(0)
    for rank, i in enumerate(sorted(range(4), key=p_values.__getitem__)):
        previous = max(previous, min(Fraction(1), (4-rank)*p_values[i]))
        test = tests[contrasts[i]]
        require(test["holm_p"] == float(previous) and Fraction(test["holm_exact"]) == previous,
                "Holm adjustment mismatch")
    diff = rates("sparse", "carry", "bler") - rates("dense", "reset", "bler")
    summary_ni = summary["noninferiority"]
    upper = diff.mean() + t.ppf(.95, 9)*diff.std(ddof=1)/np.sqrt(10)
    np.testing.assert_allclose(summary_ni["differences"], diff, atol=2e-12)
    np.testing.assert_allclose([summary_ni["mean_diff"], summary_ni["upper_95"]],
                               [diff.mean(), upper], atol=2e-12)
    require(summary_ni["margin"] == .01 and summary_ni["passes"] == bool(upper < .01), "Incorrect NI gate")


def audit(raw_path, summary_path, replay_path):
    raw = raw_path.read_bytes()
    lines = raw.splitlines()
    records = [json.loads(line) for line in lines]
    keys = [(r["cluster"], r["link"], r["seed"]) for r in records]
    expected = [(i, j, BASE + 1000*i + j) for i in range(1, 11) for j in range(32)]
    require(keys == expected, "Formal cohort missing, duplicated or out of order")
    for r in records:
        energies = set()
        require(set(r["budgets"]) == set(BUDGETS), "Missing raw budget")
        for budget, arms in r["budgets"].items():
            require(set(arms) == set(ARMS), "Incomplete raw arm list")
            for cell in arms.values():
                check_cell(cell, budget)
                energies.add((cell["probe_energy"], cell["clean_energy"]))
        require(len(energies) == 1, "Held-out target energies differ across paired cells")
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    verify_group(records, summary["totals"])
    require(len(summary["clusters"]) == 10, "Incorrect cluster count")
    groups = [verify_group(records[start:start+32], table)
              for start, table in zip(range(0, 320, 32), summary["clusters"])]
    verify_stats(groups, summary)
    selected = [line for line, r in zip(lines, records) if r["cluster"] in (1, 10)]
    require(selected == replay_path.read_bytes().splitlines(), "Fresh replay differs in at least one field")
    return dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                links=320, cells=2560, contrasts=4, exact_sign_patterns=1024,
                replay_clusters=[1, 10], replay_links=len(selected), replay_all_fields_match=True,
                raw_sha256=hashlib.sha256(raw).hexdigest(),
                summary_sha256=hashlib.sha256(summary_bytes).hexdigest(),
                selected_records_lf_sha256=hashlib.sha256(b"\n".join(selected)+b"\n").hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=RESULTS / "links.jsonl")
    parser.add_argument("--summary", type=Path, default=RESULTS / "summary.json")
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=RESULTS / "audit.json")
    args = parser.parse_args()
    result = audit(args.raw, args.summary, args.replay)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
