"""Full-grid, reused-development numerical audit of frozen R2 interventions.

Run as ``python -m p6_followup.numerical_audit``.  Workers own independent
models and scoped hooks; no threaded intervention or parameter update occurs.
Strict numerical failures are evidence and are retained in the output file.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
import math
import multiprocessing
import os
from pathlib import Path
import time
from unittest.mock import patch

import torch

from p5.study import bit_loss
from p5b.diagnostics import SCALES, erased, gain_check as original_gain_check
from p6 import experiment as ex, extended_audit
from p6.data import dev_frame, dev_seed
from p6_followup import common


PAIRS = (901, 902, 903)
MODES = (("trust", "trust", False), ("smooth", "smooth", False),
         ("smooth_exact", "smooth", True))
THRESHOLD = 1e-12


def _failure_reasons(row):
    reasons = []
    if not row["finite"]:
        reasons.append("nonfinite_output")
    if row["coded_flips"]:
        reasons.append("coded_hard_decision_flips")
    if not row["max_grid_llr_diff"] < THRESHOLD:
        reasons.append("llr_difference_not_below_1e-12")
    if not row["max_relative_cfr_l2"] < THRESHOLD:
        reasons.append("relative_cfr_l2_not_below_1e-12")
    return reasons


def recorded_gain_check(model, obs, scales, phase_seed):
    """Run the inherited check verbatim while retaining every transformation.

    The forward hook observes its outputs only.  Its separate generator
    reproduces the inherited random phases without advancing any model RNG.
    """
    phases = torch.Generator().manual_seed(phase_seed)
    baseline = None
    records = []

    def capture(module, args, output):
        nonlocal baseline
        llr, channel = output
        if baseline is None:
            baseline = (llr, channel)
            return
        scale = scales[len(records)]
        phase = 2 * math.pi * torch.rand(len(obs.y), dtype=torch.float64, generator=phases)
        gain = torch.polar(torch.full_like(phase, scale), phase)[:, None, None, None]
        base_llr, base_channel = baseline
        finite = bool(torch.isfinite(llr).all() and torch.isfinite(channel).all()
                      and torch.isfinite(base_llr).all() and torch.isfinite(base_channel).all())
        llr_difference = float((base_llr - llr).abs().max())
        error = ((channel / gain) - base_channel).flatten(1).norm(dim=1)
        norm = base_channel.flatten(1).norm(dim=1)
        cfr_difference = float((error / torch.where(norm > 0, norm, 1.)).max())
        changed = ((base_llr[:, ~obs.pilot_mask] > 0)
                   != (llr[:, ~obs.pilot_mask] > 0))
        row = dict(scale=scale, phase_radians=phase.tolist(), finite=finite,
                   max_grid_llr_diff=llr_difference, max_relative_cfr_l2=cfr_difference,
                   coded_flips=int(changed.sum()), decisions=changed.numel())
        row["failure_reasons"] = _failure_reasons(row)
        row["passed"] = not row["failure_reasons"]
        records.append(row)

    with model.register_forward_hook(capture):
        result = original_gain_check(model, obs, scales, phase_seed)
    if len(records) != len(scales):
        raise RuntimeError("Unexpected inherited numerical-audit call count")
    # Preserve the inherited condition summary and its strict pass decision.
    result["transformations"] = records
    result["failure_reasons"] = _failure_reasons(result)
    if result["passed"] != all(row["passed"] for row in records):
        raise RuntimeError("Recorded transformations disagree with inherited audit")
    return result


def gradient_sanity(model, mode, exact_emit, label):
    """One reused erased B2 frame, one backward pass, no optimization step."""
    frame = dev_frame(2, 6, 901, 2)
    frame = replace(frame, obs=erased(frame.obs, "first_pilot_row"))
    before = {name: common.tensor_hash(value) for name, value in model.state_dict().items()}
    model.zero_grad(set_to_none=True)
    loss = bit_loss(frame, common.AdaptedReceiver(model, mode, exact_emit))
    loss.backward()
    tensors = []
    for name, parameter in model.head.named_parameters():
        if not parameter.requires_grad:
            continue
        gradient = parameter.grad
        present = gradient is not None
        finite = present and bool(torch.isfinite(gradient).all())
        tensors.append(dict(name=name, elements=parameter.numel(), present=present,
                            finite=finite,
                            nonzero=present and bool(gradient.count_nonzero()),
                            nonzero_elements=int(gradient.count_nonzero()) if present else 0,
                            norm=float(gradient.norm()) if present else None,
                            max_abs=float(gradient.abs().max()) if present else None))
    after = {name: common.tensor_hash(value) for name, value in model.state_dict().items()}
    model.zero_grad(set_to_none=True)
    return dict(mode=label, kind="descriptive_backward_sanity_not_training", pair=901,
                regime=2, stream=6, seed=dev_seed(2, 6, 901), batch_size=2,
                condition="first_pilot_row", loss="payload_binary_cross_entropy_with_logits",
                loss_value=float(loss.detach()), loss_finite=bool(torch.isfinite(loss)),
                all_head_gradients_present=all(row["present"] for row in tensors),
                all_head_gradients_finite=all(row["finite"] for row in tensors),
                all_head_tensors_nonzero=all(row["nonzero"] for row in tensors),
                parameters_unchanged=before == after, optimizer_steps=0, tensors=tensors)


def audit_pair(pair):
    started = time.perf_counter()
    model, receipt = common.load_model(pair, development=True)
    fit_receipt = common.read((common.REPO / receipt["path"]).with_suffix(".json"))
    updates = fit_receipt["recipe"]["steps"]
    rows, gradients = [], []
    for label, mode, exact_emit in MODES:
        adapted = common.AdaptedReceiver(model, mode, exact_emit)
        with patch.object(extended_audit, "gain_check", recorded_gain_check):
            measured = extended_audit.gauge_rows(adapted, 2, pair, "parity",
                                                "archived_final_development", updates,
                                                full_grid=True)
        rows.extend(dict(row, mode=label, exact_emit=exact_emit) for row in measured)
        if pair == 901:
            gradients.append(gradient_sanity(model, mode, exact_emit, label))
    return dict(pair=pair, checkpoint=receipt, rows=rows, gradient_sanity=gradients,
                environment=common.environment(), process_id=os.getpid(),
                seconds=time.perf_counter() - started)


def sources():
    result = common.source_hashes()
    path = Path(__file__).resolve()
    result[path.relative_to(common.REPO).as_posix()] = common.sha(path)
    return result


def _json_safe(value):
    """Keep nonfinite failures serializable without pretending they are finite."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


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
                       strict_failures=sum(not row["passed"] for row in result["rows"])),
                  flush=True)
    records.sort(key=lambda item: item["pair"])
    rows = [row for record in records for row in record["rows"]]
    summary = {}
    for label, _, _ in MODES:
        selected = [row for row in rows if row["mode"] == label]
        measured = extended_audit.summarize(selected)
        transforms = [item for row in selected for item in row["transformations"]]
        summary[label] = dict(measured, gain_transformations=len(transforms),
                              passed_transformations=sum(item["passed"] for item in transforms),
                              failed_transformations=sum(not item["passed"] for item in transforms),
                              finite=all(row["finite"] for row in selected),
                              no_coded_hard_flips=all(row["coded_flips"] == 0 for row in selected))
    source_after = sources()
    result = dict(kind="reused_development_numerical_audit_not_new_channel_evidence",
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  base_commit=common.BASE, environment=common.environment(),
                  workers=args.workers, multiprocessing_start_method="spawn",
                  strict_threshold=THRESHOLD, strict_comparison="less_than",
                  scales=list(SCALES), conditions_per_pair_and_mode=10,
                  pairs=list(PAIRS), modes=[mode[0] for mode in MODES],
                  checkpoints={str(record["pair"]): record["checkpoint"] for record in records},
                  worker_receipts=[{key: record[key] for key in
                                    ("pair", "environment", "process_id", "seconds")}
                                   for record in records],
                  source_sha256=source_before,
                  sources_unchanged_during_audit=source_before == source_after,
                  source_sha256_after=source_after, rows=rows, summary_by_mode=summary,
                  gradient_sanity=[row for record in records for row in record["gradient_sanity"]],
                  strict_failed_conditions=[{key: row[key] for key in
                                             ("pair", "mode", "condition", "failure_reasons",
                                              "max_grid_llr_diff", "max_relative_cfr_l2",
                                              "coded_flips", "finite")}
                                            for row in rows if not row["passed"]],
                  no_coded_hard_flips=all(row["coded_flips"] == 0 for row in rows),
                  strict_passed=all(row["passed"] for row in rows),
                  seconds=time.perf_counter() - started,
                  interpretation="Float64/complex128 CPU checks on reused development data; "
                                 "no training, no low-precision guarantee, no uniform numerical theorem.")
    common.save(destination, _json_safe(result))
    print(dict(path=str(destination), summary_by_mode=summary,
               sources_unchanged_during_audit=result["sources_unchanged_during_audit"],
               seconds=result["seconds"]), flush=True)


if __name__ == "__main__":
    main()
