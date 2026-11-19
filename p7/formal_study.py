"""Formal Phase 7 held-out training and 6-condition evaluation across N=10 fresh pairs."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import time

import torch

from p5.study import setup
from p5.wavefield_sim import simulate
from p6 import experiment as ex
from p6_followup.common import load_model
from p7.conditions import CONDITIONS
from p7.eval_utils import (
    evaluate_em7_batch,
    evaluate_p6_consistent_batch,
    evaluate_waveform_dd_batch,
)
from p7.freeze import (
    FORMAL_PAIRS,
    RESULTS,
    SEED_BASE,
    tensor_sha256,
    verify_freeze,
)
from p7.model import EM7HypothesisReceiver
from p7.train import save_checkpoint, train_em7_model


def _run_formal_pair(args):
    pair_idx, freeze_sha = args
    setup()
    t0 = time.perf_counter()
    pair_dir = RESULTS / "formal" / f"pair{pair_idx:02d}"
    record_path = pair_dir / "record.json"
    if record_path.exists():
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if existing.get("freeze_sha256") == freeze_sha and len(existing.get("rows", [])) == 4 * len(CONDITIONS) * 6:
            return existing

    init_seed = SEED_BASE + 900000 + pair_idx
    mask_seed_base = SEED_BASE + 200000 + 1000 * pair_idx
    frame_fn = lambda step, bs: simulate(2, bs, 9, SEED_BASE + 10000 * pair_idx + step)

    models = {"EM7-Fixed": EM7HypothesisReceiver("fixed").eval()}
    checkpoints = {}
    train_traces = {}
    for arm_name, variant in (
        ("EM7-Axial", "axial"),
        ("EM7-Local", "local"),
        ("EM7-SingleHyp-Axial", "single_axial"),
    ):
        ck_path = pair_dir / f"{variant}.pt"
        model, trace = train_em7_model(
            variant,
            init_seed,
            frame_fn,
            steps=40,
            lr=0.004,
            lambda_hold=0.05,
            lambda_nmse=0.005,
            anchor_l2=5e-5,
            batch_size=8,
            grad_clip=1.0,
            mask_seed_base=mask_seed_base,
        )
        ck_sha = save_checkpoint(
            ck_path,
            model,
            dict(pair_idx=pair_idx, init_seed=init_seed, freeze_sha256=freeze_sha),
        )
        models[arm_name] = model
        train_traces[arm_name] = trace
        checkpoints[arm_name] = dict(path=ck_path.relative_to(RESULTS.parent.parent).as_posix(), sha256=ck_sha)

    p6_pair = pair_idx + 5
    p6_model, p6_receipt = load_model(p6_pair, "parity", development=False)

    rows = []
    batch_digests = []
    for batch_idx in range(4):
        eval_seed = SEED_BASE + 500000 + 1000 * pair_idx + batch_idx
        cond_seed = SEED_BASE + 700000 + 1000 * pair_idx + batch_idx
        frame = simulate(2, 8, 9, eval_seed)
        batch_digests.append(
            dict(
                batch_index=batch_idx,
                eval_seed=eval_seed,
                cond_seed=cond_seed,
                obs_y_sha256=tensor_sha256(frame.obs.y),
                bits_sha256=tensor_sha256(frame.bits),
            )
        )
        for cond in CONDITIONS:
            for arm_name, model in models.items():
                rows.append(evaluate_em7_batch(arm_name, model, frame, cond, batch_idx, cond_seed))
            rows.append(evaluate_p6_consistent_batch(p6_model, frame, cond, batch_idx, cond_seed))
            rows.append(evaluate_waveform_dd_batch(frame, cond, batch_idx, cond_seed))

    record = dict(
        pair_idx=pair_idx,
        p6_pair=p6_pair,
        p6_checkpoint=p6_receipt,
        freeze_sha256=freeze_sha,
        init_seed=init_seed,
        checkpoints=checkpoints,
        train_traces=train_traces,
        batch_digests=batch_digests,
        rows=rows,
        elapsed_seconds=time.perf_counter() - t0,
    )
    ex.save(record_path, record)
    return record


def main():
    setup()
    freeze_sha = verify_freeze()
    print(f"Verified prospective freeze sha256={freeze_sha}", flush=True)
    t0 = time.perf_counter()
    tasks = [(pair_idx, freeze_sha) for pair_idx in FORMAL_PAIRS]
    with ProcessPoolExecutor(
        max_workers=5,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=setup,
    ) as pool:
        for rec in pool.map(_run_formal_pair, tasks):
            c4_ax = sum(r["score"]["block_errors"] for r in rec["rows"]
                        if r["arm"] == "EM7-Axial" and r["condition"] == "C4a_roving_window")
            c4_lo = sum(r["score"]["block_errors"] for r in rec["rows"]
                        if r["arm"] == "EM7-Local" and r["condition"] == "C4a_roving_window")
            c1_ax = sum(r["score"]["block_errors"] for r in rec["rows"]
                        if r["arm"] == "EM7-Axial" and r["condition"] == "C1_row9_erased")
            print(
                f"Formal pair {rec['pair_idx']:02d} done in {rec['elapsed_seconds']:.1f}s | "
                f"C1 Axial={c1_ax} | C4a Axial={c4_ax} Local={c4_lo}",
                flush=True,
            )
    print(f"All {len(FORMAL_PAIRS)} formal pairs complete in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
