"""Strict full-grid numerical symmetry audit on reused P6 development frames.

Four receiver modes, three archived checkpoints, ten conditions and 27 complex
gain transformations per condition. Workers are separate processes because
receiver interventions use scoped process-global hooks. Numerical failures
remain evidence: this driver records them without changing the 1e-12 gate.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import multiprocessing
import os
from pathlib import Path
import time
from unittest.mock import patch

from p5b.diagnostics import SCALES
from p6 import experiment as ex, extended_audit
from p6_followup.numerical_audit import THRESHOLD, _json_safe, recorded_gain_check
from p6_softinfo import common
from p6_softinfo.receiver import SoftInformationReceiver


PAIRS = (901, 902, 903)
MODES = (("trust", False, False), ("variance", True, False),
         ("profile", False, True), ("variance_profile", True, True))


def audit_pair(pair):
    started = time.perf_counter()
    model, receipt = common.load_model(pair, development=True)
    fit = common.read((common.REPO / receipt["path"]).with_suffix(".json"))
    rows = []
    for label, variance, refine in MODES:
        # Trust uses the actual frozen module; copy equivalence is separately
        # tested, rather than assumed by making both sides use the new code.
        candidate = model if label == "trust" else SoftInformationReceiver(
            model, use_variance=variance, refine=refine)
        with patch.object(extended_audit, "gain_check", recorded_gain_check):
            measured = extended_audit.gauge_rows(
                candidate, 2, pair, "parity", "archived_final_development",
                fit["recipe"]["steps"], full_grid=True)
        rows.extend(dict(row, mode=label, variance=variance, refinement=refine)
                    for row in measured)
    return dict(pair=pair, checkpoint=receipt, rows=rows,
                environment=common.environment(), process_id=os.getpid(),
                seconds=time.perf_counter() - started)


def sources():
    result = common.source_hashes()
    source = Path(__file__).resolve()
    result[source.relative_to(common.REPO).as_posix()] = common.sha(source)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    destination = common.RESULTS / "numerical_audit.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite numerical evidence: {destination}")
    ex.setup()
    started = time.perf_counter()
    source_before = sources()
    records = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=ex.setup,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(audit_pair, pair): pair for pair in PAIRS}
        for future in as_completed(futures):
            result = future.result()
            records.append(result)
            print(dict(pair=result["pair"], seconds=result["seconds"],
                       condition_checks=len(result["rows"]),
                       strict_failed_conditions=sum(not row["passed"] for row in result["rows"])),
                  flush=True)
    records.sort(key=lambda item: item["pair"])
    rows = [row for record in records for row in record["rows"]]
    summary = {}
    for label, _, _ in MODES:
        selected = [row for row in rows if row["mode"] == label]
        transforms = [item for row in selected for item in row["transformations"]]
        summary[label] = dict(
            extended_audit.summarize(selected), gain_transformations=len(transforms),
            passed_transformations=sum(item["passed"] for item in transforms),
            failed_transformations=sum(not item["passed"] for item in transforms),
            finite=all(row["finite"] for row in selected),
            no_coded_hard_flips=all(row["coded_flips"] == 0 for row in selected))
    source_after = sources()
    report = dict(
        kind="reused_development_numerical_audit_not_new_channel_evidence",
        created_utc=datetime.now(timezone.utc).isoformat(), base_commit=common.BASE,
        environment=common.environment(), workers=args.workers,
        multiprocessing_start_method="spawn", strict_threshold=THRESHOLD,
        strict_comparison="less_than", scales=list(SCALES),
        conditions_per_pair_and_mode=10, pairs=list(PAIRS),
        modes=[item[0] for item in MODES],
        checkpoints={str(record["pair"]): record["checkpoint"] for record in records},
        worker_receipts=[{key: record[key] for key in
                         ("pair", "environment", "process_id", "seconds")} for record in records],
        source_sha256=source_before, source_sha256_after=source_after,
        sources_unchanged_during_audit=source_before == source_after,
        inherited_changes=common.inherited_changes(), rows=rows, summary_by_mode=summary,
        strict_failed_conditions=[{key: row[key] for key in
                                  ("pair", "mode", "condition", "failure_reasons",
                                   "max_grid_llr_diff", "max_relative_cfr_l2", "coded_flips", "finite")}
                                 for row in rows if not row["passed"]],
        no_coded_hard_flips=all(row["coded_flips"] == 0 for row in rows),
        strict_passed=all(row["passed"] for row in rows),
        seconds=time.perf_counter() - started,
        interpretation="Float64/complex128 CPU checks on reused development data; "
                       "no training, no low-precision guarantee or uniform numerical theorem. "
                       "The profile solve is an inference-only detached refinement, "
                       "not an end-to-end differentiable training claim.")
    common.save(destination, _json_safe(report))
    print(dict(path=str(destination), summary_by_mode=summary,
               sources_unchanged_during_audit=report["sources_unchanged_during_audit"],
               seconds=report["seconds"]), flush=True)


if __name__ == "__main__":
    main()
