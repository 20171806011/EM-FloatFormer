"""Formal Phase 8 Manifold held-out training and 4-condition Off-Grid evaluation across N=10 fresh pairs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import time

from p5.study import setup
from p6 import experiment as ex
from p8_manifold.eval_utils import evaluate_em8_batch, evaluate_lmmse_batch
from p8_manifold.freeze import (
    BATCH_SIZE,
    EVAL_BATCHES,
    FORMAL_ARMS,
    FORMAL_PAIRS,
    LEARNED_ARMS,
    RESULTS,
    SEED_BASE,
    SNR_DB,
    tensor_sha256,
    verify_freeze,
)
from p8_manifold.model import EM8ManifoldReceiver
from p8_manifold.offgrid_sim import OFFGRID_CONDITIONS, simulate_offgrid
from p8_manifold.train import save_checkpoint, train_em8_model


def _run_formal_pair(args):
    pair_idx, freeze_sha = args
    setup()
    t0 = time.perf_counter()
    pair_dir = RESULTS / "formal" / f"pair{pair_idx:02d}"
    record_path = pair_dir / "record.json"
    expected_rows = EVAL_BATCHES * len(OFFGRID_CONDITIONS) * len(FORMAL_ARMS)
    if record_path.exists():
        existing = json.loads(record_path.read_text(encoding="utf-8"))
        if existing.get("freeze_sha256") == freeze_sha and len(existing.get("rows", [])) == expected_rows:
            return existing

    init_seed = SEED_BASE + 900000 + pair_idx
    train_seed_base = SEED_BASE + 100000 + 10000 * pair_idx

    models = {"EM8-Fixed": EM8ManifoldReceiver("fixed", d_model=8, n_layers=1, n_heads=2, steps=2).eval()}
    checkpoints = {}
    train_traces = {}
    for arm_name, variant in LEARNED_ARMS:
        ck_path = pair_dir / f"{variant}.pt"
        model, trace = train_em8_model(
            variant,
            init_seed=init_seed,
            train_seed_base=train_seed_base,
            d_model=8,
            n_layers=1,
            n_heads=2,
            steps=60,
            lr=0.0035,
            snr_db=SNR_DB,
            batch_size=BATCH_SIZE,
            lambda_hold=0.08,
            lambda_nmse=0.01,
            lambda_ici=0.05,
            lambda_cal=0.02,
            anchor_l2=1e-4,
            grad_clip=1.0,
            em_steps=2,
        )
        ck_sha = save_checkpoint(
            ck_path,
            model,
            dict(pair_idx=pair_idx, arm=arm_name, init_seed=init_seed, freeze_sha256=freeze_sha),
        )
        models[arm_name] = model
        train_traces[arm_name] = dict(
            elapsed_seconds=trace["elapsed_seconds"],
            steps=trace["steps"],
            real_parameters=model.real_parameters(),
            final_step=trace["history"][-1],
        )
        checkpoints[arm_name] = dict(path=ck_path.relative_to(RESULTS.parent.parent).as_posix(), sha256=ck_sha)

    rows = []
    batch_digests = []
    for c_idx, cond in enumerate(OFFGRID_CONDITIONS):
        for b_idx in range(EVAL_BATCHES):
            eval_seed = SEED_BASE + 500000 + 10000 * pair_idx + 100 * c_idx + b_idx
            frame = simulate_offgrid(BATCH_SIZE, snr_db=SNR_DB, seed=eval_seed, condition=cond)
            batch_digests.append(
                dict(
                    condition=cond,
                    batch_index=b_idx,
                    eval_seed=eval_seed,
                    obs_y_sha256=tensor_sha256(frame.obs.y),
                    bits_sha256=tensor_sha256(frame.bits),
                    pilot_mask_sha256=tensor_sha256(frame.obs.pilot_mask),
                )
            )
            for arm_name, model in models.items():
                rows.append(evaluate_em8_batch(arm_name, model, frame, cond, b_idx))
            rows.append(evaluate_lmmse_batch(frame, cond, b_idx))

    record = dict(
        pair_idx=pair_idx,
        freeze_sha256=freeze_sha,
        init_seed=init_seed,
        train_seed_base=train_seed_base,
        checkpoints=checkpoints,
        train_traces=train_traces,
        batch_digests=batch_digests,
        rows=rows,
        elapsed_seconds=time.perf_counter() - t0,
    )
    ex.save(record_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description="Formal Phase 8 Manifold N=10 Held-Out Study")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    setup()
    freeze_sha = verify_freeze()
    print(f"Verified prospective freeze sha256={freeze_sha}", flush=True)
    t0 = time.perf_counter()
    tasks = [(pair_idx, freeze_sha) for pair_idx in FORMAL_PAIRS]
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=setup,
    ) as pool:
        for rec in pool.map(_run_formal_pair, tasks):
            o1_tr = sum(r["score"]["block_errors"] for r in rec["rows"]
                        if r["arm"] == "EM8-Transport" and r["condition"] == "O1_roving_island")
            o1_ab = sum(r["score"]["block_errors"] for r in rec["rows"]
                        if r["arm"] == "EM8-AbsNoTransport" and r["condition"] == "O1_roving_island")
            o1_lo = sum(r["score"]["block_errors"] for r in rec["rows"]
                        if r["arm"] == "EM8-Local" and r["condition"] == "O1_roving_island")
            o3_d = sum(r["sequential"]["delta_block_errors"] for r in rec["rows"]
                       if r["arm"] == "EM8-Transport" and r["condition"] == "O3_holdout_sequential")
            print(
                f"Formal pair {rec['pair_idx']:02d} done in {rec['elapsed_seconds']:.1f}s | "
                f"O1 Tr={o1_tr} Ab={o1_ab} Lo={o1_lo} | O3 dBlk={o3_d:+d}",
                flush=True,
            )
    print(f"All {len(FORMAL_PAIRS)} formal pairs complete in {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
