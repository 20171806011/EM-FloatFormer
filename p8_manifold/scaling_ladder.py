"""Direction 3: Controlled 3-Rung Scaling-Law Ladder (EM8-Transport vs parameter-matched EM8-Local)."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import time

import torch

from p5.study import setup
from p8_manifold.eval_utils import evaluate_em8_batch, summarize_ratio
from p8_manifold.offgrid_sim import OFFGRID_CONDITIONS, simulate_offgrid
from p8_manifold.train import train_em8_model


LADDER_SEED_BASE = 89000000
LADDER_PAIRS = (901, 902, 903)
EVAL_BATCHES = 4
BATCH_SIZE = 8
SNR_DB = 15.0

RUNGS = (
    dict(name="Small", d_model=8, n_layers=1, n_heads=2),
    dict(name="Medium", d_model=16, n_layers=2, n_heads=4),
    dict(name="Large", d_model=24, n_layers=3, n_heads=4),
)

ARMS = (
    ("EM8-Transport", "transport"),
    ("EM8-Local", "local"),
)


def _seed(pair_id, stream):
    return LADDER_SEED_BASE + int(pair_id) * 10000 + int(stream)


def _run_rung_pair(job):
    rung, pair_id, steps, lr = job
    setup()
    t0 = time.perf_counter()
    init_seed = _seed(pair_id, 1)
    train_seed_base = _seed(pair_id, 100)

    models = {}
    params_map = {}
    train_secs = {}
    for arm_name, variant in ARMS:
        model, info = train_em8_model(
            variant,
            init_seed=init_seed,
            train_seed_base=train_seed_base,
            d_model=rung["d_model"],
            n_layers=rung["n_layers"],
            n_heads=rung["n_heads"],
            steps=steps,
            lr=lr,
            snr_db=SNR_DB,
            batch_size=BATCH_SIZE,
        )
        models[arm_name] = model
        params_map[arm_name] = model.real_parameters()
        train_secs[arm_name] = info["elapsed_seconds"]

    rows = []
    for c_idx, cond in enumerate(OFFGRID_CONDITIONS):
        for b_idx in range(EVAL_BATCHES):
            frame = simulate_offgrid(
                BATCH_SIZE, snr_db=SNR_DB, seed=_seed(pair_id, 5000 + c_idx * 100 + b_idx), condition=cond
            )
            for arm_name, model in models.items():
                rows.append(evaluate_em8_batch(arm_name, model, frame, cond, b_idx))

    return dict(
        rung=rung["name"],
        pair_id=pair_id,
        params=params_map,
        train_secs=train_secs,
        elapsed_seconds=time.perf_counter() - t0,
        rows=rows,
    )


def _loglog_slope(xs, ys):
    """Ordinary least squares slope of log(y) vs log(x) (negative slope = improves with scale)."""
    if any(x <= 0 or y <= 0 for x, y in zip(xs, ys)):
        return None
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx = sum(lx) / len(lx)
    my = sum(ly) / len(ly)
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den if den > 0 else None


def main():
    parser = argparse.ArgumentParser(description="Phase 8 Controlled 3-Rung Scaling-Law Ladder")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.0035)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=str, default="p8_manifold/results/scaling_ladder.json")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    jobs = [(rung, pid, args.steps, args.lr) for rung in RUNGS for pid in LADDER_PAIRS]
    t0 = time.perf_counter()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
            results = list(ex.map(_run_rung_pair, jobs))
    else:
        results = [_run_rung_pair(j) for j in jobs]

    by_rung = {}
    for rung in RUNGS:
        rname = rung["name"]
        sub_res = [r for r in results if r["rung"] == rname]
        all_rows = [row for r in sub_res for row in r["rows"]]
        entry = dict(
            config=rung,
            real_parameters=sub_res[0]["params"]["EM8-Transport"],
            train_seconds_mean={
                arm: sum(r["train_secs"][arm] for r in sub_res) / len(sub_res)
                for arm, _ in ARMS
            },
            conditions={},
        )
        for cond in OFFGRID_CONDITIONS:
            entry["conditions"][cond] = {}
            for arm, _ in ARMS:
                crows = [row for row in all_rows if row["condition"] == cond and row["arm"] == arm]
                blk_err = sum(r["score"]["block_errors"] for r in crows)
                blks = sum(r["score"]["blocks"] for r in crows)
                bce_sum = sum(r["score"]["bce_sum"] for r in crows)
                bit_cnt = sum(r["score"]["bit_count"] for r in crows)
                ch_err = sum(r["score"]["channel_error"] for r in crows)
                ch_eng = sum(r["score"]["channel_energy"] for r in crows)
                b0_var = sum(r["cal_stats"]["bin0_sum_var"] for r in crows)
                b0_mse = sum(r["cal_stats"]["bin0_sum_mse"] for r in crows)
                st = dict(
                    block_errors=blk_err,
                    blocks=blks,
                    bler=blk_err / max(blks, 1),
                    bce=bce_sum / max(bit_cnt, 1),
                    nmse=ch_err / max(ch_eng, 1e-300),
                    bin0_ratio=summarize_ratio(b0_mse, b0_var),
                )
                if any("holdout" in r for r in crows):
                    st["holdout_nll"] = sum(r["holdout"]["nll"] * r["holdout"]["count"] for r in crows) / max(
                        sum(r["holdout"]["count"] for r in crows), 1
                    )
                if any("sequential" in r for r in crows):
                    st["delta_block_errors"] = sum(r["sequential"]["delta_block_errors"] for r in crows)
                entry["conditions"][cond][arm] = st
        by_rung[rname] = entry

    # Compute empirical scaling slopes & widening Transport advantage across rungs
    ps = [by_rung[r["name"]]["real_parameters"] for r in RUNGS]
    slopes = {}
    for cond in OFFGRID_CONDITIONS:
        slopes[cond] = {}
        for arm, _ in ARMS:
            blers = [by_rung[r["name"]]["conditions"][cond][arm]["bler"] for r in RUNGS]
            bces = [by_rung[r["name"]]["conditions"][cond][arm]["bce"] for r in RUNGS]
            nmses = [by_rung[r["name"]]["conditions"][cond][arm]["nmse"] for r in RUNGS]
            slopes[cond][arm] = dict(
                bler_slope=_loglog_slope(ps, blers),
                bce_slope=_loglog_slope(ps, bces),
                nmse_slope=_loglog_slope(ps, nmses),
            )
        slopes[cond]["local_minus_transport_bler"] = [
            by_rung[r["name"]]["conditions"][cond]["EM8-Local"]["bler"]
            - by_rung[r["name"]]["conditions"][cond]["EM8-Transport"]["bler"]
            for r in RUNGS
        ]
        slopes[cond]["local_minus_transport_bce"] = [
            by_rung[r["name"]]["conditions"][cond]["EM8-Local"]["bce"]
            - by_rung[r["name"]]["conditions"][cond]["EM8-Transport"]["bce"]
            for r in RUNGS
        ]

    payload = dict(
        config=dict(
            pairs=list(LADDER_PAIRS),
            steps=args.steps,
            lr=args.lr,
            eval_batches=EVAL_BATCHES,
            batch_size=BATCH_SIZE,
            snr_db=SNR_DB,
        ),
        elapsed_seconds=time.perf_counter() - t0,
        rungs=by_rung,
        scaling_slopes=slopes,
    )
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n=== Phase 8 Scaling-Law Ladder Completed in {payload['elapsed_seconds']:.1f}s ===", flush=True)
    for cond in OFFGRID_CONDITIONS:
        print(f"\n[{cond}]", flush=True)
        for r in RUNGS:
            rn = r["name"]
            p_cnt = by_rung[rn]["real_parameters"]
            tr = by_rung[rn]["conditions"][cond]["EM8-Transport"]
            lo = by_rung[rn]["conditions"][cond]["EM8-Local"]
            print(
                f"  {rn:6s} (P={p_cnt:5d}): "
                f"Tr BLER={100*tr['bler']:5.2f}% BCE={tr['bce']:.4f} | "
                f"Lo BLER={100*lo['bler']:5.2f}% BCE={lo['bce']:.4f} | "
                f"Gap(Lo-Tr)={100*(lo['bler']-tr['bler']):+5.2f}pp / {lo['bce']-tr['bce']:+.4f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
