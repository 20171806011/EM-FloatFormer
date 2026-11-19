"""Run reused development checks, freeze, and fresh paired inference scores."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import time

import torch

from p5.study import classical, counts
from p6 import experiment as ex
from p6.data import dev_frame
from p6.inspect_iter3_paths import r2_observations, sign_summary
from p6.math_chirp_fisher import reliability
from p6.math_training import receiver
from p6_followup.common import (AdaptedReceiver, CONDITIONS, METHODS, RECIPE, RESULTS, ROOT,
    SEED_BASE, corrupt, environment, freeze, fresh_frame, load_model, read, save,
    tensor_hash, verify_freeze)


def models_for(pair, development=False):
    parity, p_receipt = load_model(pair, development=development)
    models = {name: AdaptedReceiver(parity, name) for name in ("trust", "smooth", "hard_chart")}
    receipts = {"parity": p_receipt}
    if not development:
        local, l_receipt = load_model(pair, "local_parity")
        models["local_parity"] = local
        models["fixed_parity"] = receiver("physics_parity", 0, RECIPE).eval()
        receipts["local_parity"] = l_receipt
    return models, receipts


@torch.inference_mode()
def evaluate(models, frame, condition, batch):
    probe = corrupt(frame, condition)
    rows = []
    q = reliability(probe.obs.y, probe.obs.noise_var, probe.obs.pilot_mask)
    for name in (METHODS if "local_parity" in models else ("trust", "smooth", "hard_chart")):
        if name in ("trust", "smooth", "hard_chart"):
            output, details = r2_observations(models[name], probe)
            signs = sign_summary([details])["all"]
        else:
            output = models[name](probe.obs) if name in models else classical(probe, name)
            signs = None
        rows.append(dict(batch_index=batch, condition=condition, method=name,
                         score=counts(probe, *output), signs=signs,
                         pilot_q_min=float(q.min()), pilot_q_mean=float(q.mean()),
                         pilot_q_below_one=int((q < 1).sum())))
    return rows


def score_pair(index, replay=False):
    signature = verify_freeze()
    dest = RESULTS / "fresh" / f"pair{index:02d}.json"
    if dest.exists() and not replay:
        raise FileExistsError(f"Refusing to overwrite fresh result {dest}")
    started = time.perf_counter()
    pair = index + 5
    models, receipts = models_for(pair)
    frames, rows = [], []
    for batch in range(4):
        frame = fresh_frame(index, batch)
        frames.append(dict(seed=SEED_BASE + 1000*index + batch, batch_size=8,
                           y_sha256=tensor_hash(frame.obs.y), bits_sha256=tensor_hash(frame.bits)))
        for condition in CONDITIONS:
            rows.extend(evaluate(models, frame, condition, batch))
    result = dict(kind="fresh_inference_only", index=index, checkpoint_pair=pair,
                  freeze_sha256=signature, checkpoints=receipts, frames=frames, rows=rows,
                  environment=environment(), seconds=time.perf_counter()-started)
    if not replay:
        save(dest, result)
    return result


def job(index):
    result = score_pair(index)
    brief = {condition: {name: sum(row["score"]["block_errors"] for row in result["rows"]
                     if row["condition"] == condition and row["method"] == name)
                         for name in ("trust", "smooth", "hard_chart")} for condition in CONDITIONS}
    return dict(index=index, seconds=result["seconds"], block_errors=brief)


def development():
    path = RESULTS / "development.json"
    if path.exists():
        raise FileExistsError("Development evidence already exists")
    records = []
    for pair in (901, 902, 903):
        models, receipts = models_for(pair, True)
        frame = dev_frame(2, 6, pair, 8)
        rows = []
        for condition in CONDITIONS:
            rows.extend(evaluate(models, frame, condition, 0))
        records.append(dict(pair=pair, checkpoint=receipts, batch_size=8, rows=rows))
    result = dict(kind="reused_development_only", environment=environment(), records=records)
    save(path, result)
    for condition in CONDITIONS:
        print(condition, {name: sum(row["score"]["block_errors"] for record in records
              for row in record["rows"] if row["condition"] == condition and row["method"] == name)
              for name in ("trust", "smooth", "hard_chart")}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("development", "freeze", "score"))
    parser.add_argument("--workers", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--index", nargs="+", type=int, choices=range(1, 11), default=list(range(1, 11)))
    args = parser.parse_args()
    if args.action == "score":
        with ProcessPoolExecutor(max_workers=args.workers, initializer=ex.setup) as pool:
            for row in pool.map(job, args.index):
                print(row, flush=True)
    else:
        ex.setup()
        if args.action == "freeze":
            print(freeze(), flush=True)
        else:
            development()


if __name__ == "__main__":
    main()
