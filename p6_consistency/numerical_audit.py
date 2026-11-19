"""Strict C* numerical audit of the five known-observation receiver variants.

This uses reused development frames only: three checkpoints, five study
conditions, B1, and the inherited 27 complex-gain transformations.  Known
receive-gain metadata stay fixed during a transformation.  Strict failures
and forward exceptions are retained; the inherited 1e-12 gate is unchanged.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import multiprocessing
import os
from pathlib import Path
import time
import traceback

import torch
from torch import nn

from p5b.diagnostics import SCALES
from p6 import experiment as ex
from p6.data import dev_frame, dev_seed
from p6_followup.numerical_audit import THRESHOLD, _json_safe, recorded_gain_check
from p6_consistency import common
from p6_consistency.observation import observation_gain
from p6_consistency.study import models_for


PAIRS = (901, 902, 903)
MODES = ("aware_mean", "aware_variance", "aware_effective",
         "aware_cavity_mean", "aware_consistent")


class FixedObservationGain(nn.Module):
    """Expose the inherited one-argument audit interface without changing metadata."""

    def __init__(self, candidate, gain):
        super().__init__()
        self.candidate = candidate
        self.register_buffer("known_gain", gain.detach().clone())

    def forward(self, obs):
        return self.candidate(obs, self.known_gain)


def audit_pair(pair):
    signature = common.verify_freeze()
    started = time.perf_counter()
    models, receipts = models_for(pair, development=True)
    frame = dev_frame(2, 6, pair, 1)
    rows, exceptions = [], []
    for condition_index, condition in enumerate(common.CONDITIONS):
        observed = common.corrupt(frame, condition)
        gain = observation_gain(observed.obs.pilot_mask, condition)
        phase_seed = 730000 + pair * 10 + condition_index
        for mode in MODES:
            wrapped = FixedObservationGain(models[mode], gain)
            metadata = dict(pair=pair, mode=mode, condition=condition, batch_size=1,
                            frame_seed=dev_seed(2, 6, pair), phase_seed=phase_seed,
                            input_y_sha256=common.tensor_hash(observed.obs.y),
                            known_gain_sha256=common.tensor_hash(gain),
                            known_gain_fixed_during_transformations=True)
            try:
                measured = recorded_gain_check(wrapped, observed.obs, SCALES, phase_seed)
                rows.append(dict(metadata, **measured))
            except Exception as error:
                exceptions.append(dict(metadata, exception_type=type(error).__name__,
                                       message=str(error), traceback=traceback.format_exc(),
                                       planned_scales=list(SCALES),
                                       interpretation="Condition audit incomplete; never counted as passing"))
    after = common.verify_freeze()
    return dict(pair=pair, checkpoints=receipts, rows=rows, exceptions=exceptions,
                freeze_sha256=signature, freeze_sha256_after=after,
                environment=common.environment(), process_id=os.getpid(),
                seconds=time.perf_counter() - started)


def source_hashes():
    result = common.source_hashes()
    own = Path(__file__).resolve()
    result[own.relative_to(common.REPO).as_posix()] = common.sha(own)
    return result


def summarize(rows, exceptions):
    result = {}
    expected_conditions = len(PAIRS) * len(common.CONDITIONS)
    for mode in MODES:
        selected = [row for row in rows if row["mode"] == mode]
        failed = [row for row in exceptions if row["mode"] == mode]
        transformations = [item for row in selected for item in row["transformations"]]
        llr = max((row["max_grid_llr_diff"] for row in selected), default=0.)
        cfr = max((row["max_relative_cfr_l2"] for row in selected), default=0.)
        complete = len(selected) == expected_conditions and not failed
        finite = complete and all(row["finite"] for row in selected)
        flips = sum(row["coded_flips"] for row in selected)
        passed = complete and all(row["passed"] for row in selected)
        result[mode] = dict(
            planned_condition_checks=expected_conditions, completed_condition_checks=len(selected),
            passed_condition_checks=sum(row["passed"] for row in selected),
            failed_condition_checks=sum(not row["passed"] for row in selected),
            forward_exceptions=len(failed),
            planned_gain_transformations=expected_conditions * len(SCALES),
            gain_transformations=len(transformations),
            passed_transformations=sum(row["passed"] for row in transformations),
            failed_transformations=sum(not row["passed"] for row in transformations),
            max_grid_llr_diff=llr, max_relative_cfr_l2=cfr,
            coded_flips=flips, decisions=sum(row["decisions"] for row in selected),
            finite=finite, no_coded_hard_flips=complete and flips == 0,
            complete=complete, passed=passed)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    destination = common.RESULTS / "numerical_audit.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite numerical evidence: {destination}")
    ex.setup()
    before = common.verify_freeze()
    sources_before = source_hashes()
    started = time.perf_counter()
    records = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=ex.setup,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(audit_pair, pair): pair for pair in PAIRS}
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(dict(pair=record["pair"], seconds=record["seconds"],
                       completed_condition_checks=len(record["rows"]),
                       strict_failed_conditions=sum(not row["passed"] for row in record["rows"]),
                       forward_exceptions=len(record["exceptions"])), flush=True)
    records.sort(key=lambda record: record["pair"])
    rows = [row for record in records for row in record["rows"]]
    exceptions = [row for record in records for row in record["exceptions"]]
    summary = summarize(rows, exceptions)
    after = common.verify_freeze()
    sources_after = source_hashes()
    failures = [{key: row[key] for key in
                 ("pair", "mode", "condition", "failure_reasons", "max_grid_llr_diff",
                  "max_relative_cfr_l2", "coded_flips", "finite")}
                for row in rows if not row["passed"]]
    report = dict(
        kind="reused_development_Cstar_audit_known_observation_metadata",
        created_utc=datetime.now(timezone.utc).isoformat(), base_commit=common.BASE,
        freeze_sha256=before, freeze_sha256_after=after,
        source_sha256=sources_before, source_sha256_after=sources_after,
        sources_unchanged_during_audit=sources_before == sources_after,
        environment=common.environment(), workers=args.workers,
        multiprocessing_start_method="spawn", pairs=list(PAIRS), modes=list(MODES),
        conditions=list(common.CONDITIONS), batch_size=1, scales=list(SCALES),
        phase_seed_formula="730000 + pair * 10 + condition_index",
        strict_threshold=THRESHOLD, strict_comparison="less_than",
        checkpoints={str(record["pair"]): record["checkpoints"] for record in records},
        worker_receipts=[{key: record[key] for key in
                         ("pair", "environment", "process_id", "seconds",
                          "freeze_sha256", "freeze_sha256_after")} for record in records],
        rows=rows, summary_by_mode=summary, strict_failed_conditions=failures,
        forward_exceptions=exceptions,
        strict_passed=all(value["passed"] for value in summary.values()),
        no_coded_hard_flips=all(value["no_coded_hard_flips"] for value in summary.values()),
        seconds=time.perf_counter() - started,
        interpretation="The inherited finite-output, zero-hard-flip, <1e-12 LLR/CFR "
                       "criteria are unchanged. Float64/complex128 CPU checks on "
                       "reused development frames are not a universal numerical "
                       "theorem, low-precision validation, or independent channel evidence.")
    common.save(destination, _json_safe(report))
    print(dict(path=str(destination), summary_by_mode=summary,
               sources_unchanged_during_audit=report["sources_unchanged_during_audit"],
               seconds=report["seconds"]), flush=True)


if __name__ == "__main__":
    main()
