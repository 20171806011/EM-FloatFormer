"""Run the current P6 suite or validate final dev checkpoints on the gain grid."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from io import StringIO
from pathlib import Path
import unittest

from p5.study import read, save
from p6.data import PAIRS, SNR
from p6.experiment import setup
from p6.extended_audit import ARMS, gauge_rows, summarize


ROOT = Path(__file__).resolve().parent
CHECKS = ROOT / "results" / "iter3_checks"


def suite():
    log = StringIO()
    tests = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(tests)
    print(log.getvalue(), flush=True)
    passed = result.wasSuccessful() and not result.skipped
    save(CHECKS / "p6_suite.json", dict(run=result.testsRun, skipped=len(result.skipped),
                                        passed=passed, log=log.getvalue()))
    return passed


def trained_cell(cell):
    from p6.study_dev import implementation, restored

    regime, pair = cell
    signature = implementation()
    dest = CHECKS / f"trained_r{regime}_{pair}.json"
    if dest.exists():
        data = read(dest)
        if data["implementation"] == signature and data["summary"]["passed"]:
            return data
    rows = []
    for arm in ARMS:
        model = restored(regime, pair, 3, arm)
        rows.extend(gauge_rows(model.eval(), regime, pair, arm, "iter3_capped", 80))
    data = dict(implementation=signature, regime=regime, pair=pair, rows=rows,
                summary=summarize(rows))
    save(dest, data)
    print(f"R{regime}/{pair} {data['summary']}", flush=True)
    return data


def trained(workers):
    cells = [(r, p) for r in SNR for p in PAIRS]
    if workers == 1:
        data = list(map(trained_cell, cells))
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=setup) as pool:
            data = list(pool.map(trained_cell, cells))
    rows = [row for cell in data for row in cell["rows"]]
    totals = dict(**summarize(rows), implementation=data[0]["implementation"],
                  failures=[r for r in rows if not r["passed"]])
    save(CHECKS / "trained_gauge.json", totals)
    return totals["passed"]


def summary():
    from p6.study_dev import implementation

    suite_result = read(CHECKS / "p6_suite.json")
    legacy = read(CHECKS / "legacy.json")
    p4 = read(CHECKS / "p4.json")
    preflight = read(CHECKS / "invariants.json")
    trained_result = read(CHECKS / "trained_gauge.json")
    physical_detail = read(CHECKS / "physical_detail.json")
    cells = preflight["rows"]
    anchors = [a for cell in cells for a in cell["anchors"]]
    directions = [d for cell in cells for d in cell["gradient"]["tensors"]]
    legacy_counts = {k: v["run"] for k, v in legacy["suites"].items()}
    stored = dict(
        implementation=implementation(),
        p6_tests=suite_result["run"], p6_skipped=suite_result["skipped"],
        p5_tests=legacy_counts["p5"], p5b_tests=legacy_counts["p5b"], p4=p4,
        legacy_dev_draws=legacy["unique_frame_draws"],
        capacity_matched=preflight["matched_capacity"],
        live_real_per_regime_arm=[c["active_real_parameters"] for c in cells],
        step1_all_live=all(c["active_real_parameters"] == c["sizes"]["trainable_real"] for c in cells),
        initialized_anchors=len(anchors), exact_anchors=sum(a["p6_bit_exact"] for a in anchors),
        max_archived_init_diff=max(max(a["archived_max_llr_cfr"]) for a in anchors),
        finite_difference_directions=len(directions),
        max_directional_error=max(d["absolute_error"] for d in directions),
        trained_gauge=trained_result, physical_detail=physical_detail,
    )
    stored["passed"] = all((suite_result["passed"], not suite_result["skipped"], legacy["passed"],
                            legacy_counts == {"p5": 15, "p5b": 13}, p4["exit"] == 0,
                            len(cells) == 15, len(anchors) == 30,
                            stored["exact_anchors"] == 30, stored["max_archived_init_diff"] == 0.,
                            stored["step1_all_live"], preflight["passed"],
                            trained_result["implementation"] == stored["implementation"],
                            physical_detail["implementation"] == stored["implementation"],
                            trained_result["checks"] == 360, trained_result["passed"]))
    save(CHECKS / "summary.json", stored)
    print(f"Numerical and regression checks passed: {stored['passed']}", flush=True)
    return stored["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("suite", "trained", "summary"))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    setup()
    functions = {"suite": suite, "trained": lambda: trained(args.workers), "summary": summary}
    raise SystemExit(not functions[args.action]())
