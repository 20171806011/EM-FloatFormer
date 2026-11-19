"""Ordered reused-development checks, prospective freeze, paired fresh scoring."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time

import torch

from p5.study import counts
from p6 import experiment as ex
from p6.data import dev_frame
from p6_consistency.calibration import calibration
from p6_consistency.common import (CONDITIONS, METHODS, RESULTS, SEED_BASE, corrupt,
    environment, freeze, fresh_frame, load_model, save, tensor_hash, verify_freeze)
from p6_consistency.observation import observation_gain
from p6_consistency.receiver import LegacyTraceReceiver, ObservationAwareReceiver


def models_for(pair, development=False):
    model, receipt = load_model(pair, development=development)
    models = {
        "legacy_mean": LegacyTraceReceiver(model, use_variance=False, refine=False),
        "legacy_variance": LegacyTraceReceiver(model, use_variance=True, refine=False),
    }
    models.update({name: ObservationAwareReceiver(model, use_variance=variance,
                                                   effective=effective, cavity=cavity)
        for name, variance, effective, cavity in (
            ("aware_mean", False, False, False),
            ("aware_variance", True, False, False),
            ("aware_effective", True, True, False),
            ("aware_cavity_mean", False, True, True),
            ("aware_consistent", True, True, True))})
    return models, {"parity": receipt}


@torch.inference_mode()
def evaluate(models, frame, condition, batch, methods=METHODS):
    observed = corrupt(frame, condition)
    gain = observation_gain(observed.obs.pilot_mask, condition)
    rows = []
    for name in methods:
        model = models[name]
        output = (model(observed.obs, gain) if isinstance(model, ObservationAwareReceiver)
                  else model(observed.obs))
        if not all(bool(torch.isfinite(value).all()) for value in output):
            raise FloatingPointError(f"Nonfinite receiver output: {name}/{condition}")
        rows.append(dict(batch_index=batch, condition=condition, method=name,
            score=counts(observed, *output), calibration=calibration(observed, output[0], model.last_trace),
            diagnostics=model.last_diagnostics))
    return rows


def score_pair(index, replay=False):
    signature = verify_freeze()
    destination = RESULTS/"fresh"/f"pair{index:02d}.json"
    if destination.exists() and not replay:
        raise FileExistsError("Refusing to overwrite fresh evidence")
    started = time.perf_counter()
    pair = index+5
    models, receipts = models_for(pair)
    frames, rows = [], []
    for batch in range(4):
        frame = fresh_frame(index, batch)
        frames.append(dict(seed=SEED_BASE+1000*index+batch, batch_size=8,
            y_sha256=tensor_hash(frame.obs.y), bits_sha256=tensor_hash(frame.bits)))
        for condition in CONDITIONS:
            rows.extend(evaluate(models, frame, condition, batch))
    result = dict(kind="fresh_inference_only", index=index, checkpoint_pair=pair,
        freeze_sha256=signature, checkpoints=receipts, frames=frames, rows=rows,
        seconds=time.perf_counter()-started, environment=environment())
    if not replay:
        save(destination, result)
    return result


def job(index):
    result = score_pair(index)
    return dict(index=index, seconds=result["seconds"], block_errors={condition: {
        name: sum(row["score"]["block_errors"] for row in result["rows"]
                  if row["condition"] == condition and row["method"] == name)
        for name in METHODS} for condition in CONDITIONS})


def development(core=False):
    destination = RESULTS/("development_core.json" if core else "development.json")
    if destination.exists():
        raise FileExistsError("Refusing to overwrite development evidence")
    methods = METHODS[:4] if core else METHODS
    records = []
    for pair in (901, 902, 903):
        models, receipts = models_for(pair, True)
        frame = dev_frame(2, 6, pair, 8)
        rows = []
        for condition in CONDITIONS:
            rows.extend(evaluate(models, frame, condition, 0, methods))
        records.append(dict(pair=pair, checkpoints=receipts, rows=rows))
        print(dict(pair=pair, block_errors={condition: {
            name: next(row["score"]["block_errors"] for row in rows
                       if row["condition"] == condition and row["method"] == name)
            for name in methods} for condition in CONDITIONS}), flush=True)
    save(destination, dict(kind="reused_development_only", methods=list(methods),
                          records=records, environment=environment()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("development_core", "development", "freeze", "score"))
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--index", nargs="+", type=int, choices=range(1, 11), default=list(range(1, 11)))
    args = parser.parse_args()
    if args.action == "score":
        with ProcessPoolExecutor(max_workers=args.workers, initializer=ex.setup,
                mp_context=multiprocessing.get_context("spawn")) as pool:
            for result in pool.map(job, args.index):
                print(result, flush=True)
    else:
        ex.setup()
        if args.action == "freeze":
            print(freeze(), flush=True)
        else:
            development(args.action == "development_core")


if __name__ == "__main__":
    main()
