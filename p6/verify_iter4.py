"""Verify the iteration-4 development checkpoints and historical regression suites."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from io import StringIO
import subprocess
import sys
import unittest

import torch

from p5b.diagnostics import SCALES
from p6 import experiment as ex
from p6.data import PAIRS, SNR, dev_frame
from p6.diagnostics import anchor_check
from p6.extended_audit import ARMS, gauge_rows, summarize


CHECKS = ex.RESULTS / "iter4_checks"


def suite():
    log = StringIO()
    tests = unittest.defaultTestLoader.discover(str(ex.ROOT / "tests"))
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(tests)
    passed = result.wasSuccessful() and not result.skipped
    ex.save(CHECKS / "p6_suite.json", dict(run=result.testsRun, skipped=len(result.skipped),
                                          passed=passed, log=log.getvalue()))
    print(log.getvalue(), flush=True)
    return passed


def legacy():
    from p6.legacy_regression import run

    result = run()
    ex.save(CHECKS / "legacy.json", result)
    return result["passed"]


def p4():
    command = [sys.executable, "-B", "-m", "publication.verify_p4"]
    result = subprocess.run(command, cwd=ex.ROOT.parent, capture_output=True, text=True)
    ex.save(CHECKS / "p4.json", dict(command="python -B -m publication.verify_p4",
                                     exit=result.returncode, stdout=result.stdout, stderr=result.stderr))
    print(result.stdout + result.stderr, flush=True)
    return result.returncode == 0


def first_update():
    """Exercise the actual first training minibatch, seed, and loss of every fit."""
    from p6.run_dev_iter4 import DEPTH, implementation, objective

    rows = []
    for regime in SNR:
        for pair in PAIRS:
            train = dev_frame(regime, 5, pair, ex.FIT_BATCH)
            val = dev_frame(regime, 6, pair, 2)
            initial = ex.model_for(regime, pair, "pure", DEPTH).state_dict()
            for arm in ex.ARMS:
                model = ex.model_for(regime, pair, arm, DEPTH)
                state = model.state_dict()
                matched = state.keys() == initial.keys() and all(
                    torch.equal(value, initial[name]) for name, value in state.items())
                anchor = anchor_check(model, val.obs, arm, DEPTH)
                objective(train, model)[0].backward()
                parameters = ex.real_parameters(model)
                counts = {}
                finite = True
                for name, parameter in model.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    grad = parameter.grad
                    if grad is None:
                        counts[name] = 0
                        finite = False
                        continue
                    real = torch.view_as_real(grad) if grad.is_complex() else grad
                    counts[name] = int(real.count_nonzero())
                    finite &= bool(real.isfinite().all())
                live = sum(counts.values())
                passed = (matched and finite and live == parameters and anchor["p6_bit_exact"]
                          and max(anchor["archived_max_llr_cfr"]) == 0.)
                rows.append(dict(regime=regime, pair=pair, arm=arm, matched_initial_state=matched,
                                 first_minibatch_finite_gradients=finite, live=live, parameters=parameters,
                                 active_by_tensor=counts, anchor=anchor, passed=passed))
    result = dict(implementation=implementation(), rows=rows,
                  passed=all(row["passed"] for row in rows) and len({r["parameters"] for r in rows}) == 1)
    ex.save(CHECKS / "first_update.json", result)
    print(f"First update: {sum(row['passed'] for row in rows)}/{len(rows)}", flush=True)
    return result["passed"]


def trained_cell(cell):
    from p6.run_dev_iter4 import DEPTH, PRIMARY, implementation, restored

    regime, pair = cell
    if regime not in SNR or pair not in PAIRS:
        raise ValueError("expected a development regime and pair")
    rows = []
    for arm in ARMS:
        model = restored(regime, pair, DEPTH, arm)
        rows.extend(gauge_rows(model.eval(), regime, pair, arm, "iter4", PRIMARY, full_grid=True))
    totals = summarize(rows)
    result = dict(implementation=implementation(), regime=regime, pair=pair, rows=rows, summary=totals)
    ex.save(CHECKS / f"trained_r{regime}_{pair}.json", result)
    print(f"R{regime}/{pair}: {totals['passed_checks']}/{totals['checks']}, "
          f"{totals['coded_flips']} flips, max delta LLR {totals['max_grid_llr_diff']:.4g}", flush=True)
    return totals["passed"]


def gauge(workers):
    cells = [(r, p) for r in SNR for p in PAIRS]
    if workers == 1:
        return all(list(map(trained_cell, cells)))
    with ProcessPoolExecutor(max_workers=workers, initializer=ex.setup) as pool:
        return all(list(pool.map(trained_cell, cells)))


def summary():
    from p6.run_dev_iter4 import implementation

    signature = implementation()
    test = ex.read(CHECKS / "p6_suite.json")
    archived = ex.read(CHECKS / "legacy.json")
    verifier = ex.read(CHECKS / "p4.json")
    audit = ex.read(CHECKS / "invariants.json")
    first = ex.read(CHECKS / "first_update.json")
    cells = [ex.read(CHECKS / f"trained_r{r}_{p}.json") for r in SNR for p in PAIRS]
    if any(cell["implementation"] != signature for cell in [first, *cells]):
        raise ValueError("verification source differs from the current fit")
    rows = [row for cell in cells for row in cell["rows"]]
    totals = summarize(rows)
    totals["scales"] = list(SCALES)
    totals["failed_cases"] = [row for row in rows if not row["passed"]]
    legacy_counts = {name: result["run"] for name, result in archived["suites"].items()}
    finite_diff = [tensor for row in audit["rows"] for tensor in row["gradient"]["tensors"]]
    anchors = [anchor for row in audit["rows"] for anchor in row["anchors"]]
    expected_cells = {(r, p, a) for r in SNR for p in PAIRS for a in ex.ARMS}
    actual_cells = {(row["regime"], row["pair"], row["arm"]) for row in first["rows"]}
    result = dict(implementation=signature, p6_tests=test["run"], p6_skipped=test["skipped"],
                  p6_passed=test["passed"], p5_tests=legacy_counts["p5"], p5b_tests=legacy_counts["p5b"],
                  legacy_dev_draws=archived["unique_frame_draws"], p4=verifier,
                  first_update_arms=len(first["rows"]), matched_capacity=audit["matched_capacity"],
                  all_initial_states_identical=all(r["matched_initial_state"] for r in first["rows"]),
                  active_real_scalar_min=min(r["live"] for r in first["rows"]),
                  active_real_scalar_max=max(r["live"] for r in first["rows"]),
                  trainable_real_scalars=first["rows"][0]["parameters"], first_update_passed=first["passed"],
                  exact_depth3_anchors=sum(r["anchor"]["p6_bit_exact"] for r in first["rows"]),
                  exact_depth2_and_depth3_anchors=sum(a["p6_bit_exact"] for a in anchors),
                  max_initial_archived_output_error=max(max(a["archived_max_llr_cfr"]) for a in anchors),
                  finite_difference_directions=len(finite_diff),
                  max_directional_error=max(d["absolute_error"] for d in finite_diff),
                  trained_gauge=totals)
    result["passed"] = all((test["passed"], test["skipped"] == 0, archived["passed"],
                            legacy_counts == {"p5": 15, "p5b": 13}, verifier["exit"] == 0,
                            audit["passed"], first["passed"], actual_cells == expected_cells,
                            len(anchors) == result["exact_depth2_and_depth3_anchors"] == 30,
                            len(first["rows"]) == result["exact_depth3_anchors"] == 45,
                            totals["checks"] == 360, totals["passed"],
                            all(row["scales"] == list(SCALES) for row in rows)))
    ex.save(CHECKS / "summary.json", result)
    print(f"Iteration-4 verification passed: {result['passed']}", flush=True)
    return result["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("suite", "legacy", "p4", "first", "gauge", "cell", "summary"))
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--regime", type=int, choices=SNR)
    parser.add_argument("--pair", type=int, choices=PAIRS)
    args = parser.parse_args()
    ex.setup()
    actions = dict(suite=suite, legacy=legacy, p4=p4, first=first_update, summary=summary,
                   gauge=lambda: gauge(args.workers), cell=lambda: trained_cell((args.regime, args.pair)))
    raise SystemExit(not actions[args.action]())
