"""Reused development, prospective freeze, then paired fresh inference."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time

import torch

from p5.study import classical, counts
from p6 import experiment as ex
from p6.data import dev_frame
from p6.math_training import receiver
from p6_followup.common import AdaptedReceiver
from p6_softinfo.common import (CONDITIONS, METHODS, RECIPE, RESULTS, SEED_BASE,
    corrupt, environment, freeze, fresh_frame, load_model, save, tensor_hash, verify_freeze)
from p6_softinfo.receiver import SoftInformationReceiver


def models_for(pair, development=False, controls=True):
    model, receipt = load_model(pair, development=development)
    models = {"trust": model, "smooth": AdaptedReceiver(model, "smooth")}
    models.update({name: SoftInformationReceiver(model, use_variance=variance, refine=refine)
        for name, variance, refine in (("variance", True, False), ("profile", False, True),
                                       ("variance_profile", True, True))})
    receipts = {"parity": receipt}
    if controls:
        local, local_receipt = load_model(pair, "local_parity", development=development)
        models["local_parity"] = local
        models["fixed_parity"] = receiver("physics_parity", 0, RECIPE).eval()
        receipts["local_parity"] = local_receipt
    return models, receipts


@torch.inference_mode()
def evaluate(models, frame, condition, batch, controls=True):
    observed = corrupt(frame, condition)
    rows = []
    for name in (METHODS if controls else METHODS[:5]):
        output = models[name](observed.obs) if name in models else classical(observed, name)
        if not all(bool(torch.isfinite(value).all()) for value in output):
            raise FloatingPointError(f"Nonfinite receiver output: {name}/{condition}")
        rows.append(dict(batch_index=batch, condition=condition, method=name,
            score=counts(observed, *output),
            diagnostics=models[name].last_diagnostics if isinstance(models.get(name), SoftInformationReceiver) else None))
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
        for name in METHODS[:5]} for condition in CONDITIONS})


def development():
    destination = RESULTS/"development.json"
    if destination.exists():
        raise FileExistsError("Refusing to overwrite development evidence")
    records = []
    for pair in (901, 902, 903):
        models, receipts = models_for(pair, True, controls=False)
        frame = dev_frame(2, 6, pair, 8)
        rows = []
        for condition in CONDITIONS:
            rows.extend(evaluate(models, frame, condition, 0, controls=False))
        records.append(dict(pair=pair, checkpoints=receipts, rows=rows))
        print(dict(pair=pair, block_errors={condition: {
            name: next(row["score"]["block_errors"] for row in rows
                       if row["condition"] == condition and row["method"] == name)
            for name in METHODS[:5]} for condition in CONDITIONS}), flush=True)
    save(destination, dict(kind="reused_development_only", records=records, environment=environment()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("development", "freeze", "score"))
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
            development()


if __name__ == "__main__":
    main()
