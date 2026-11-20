"""Formal 40-checkpoint x 3-condition x 27-scale (= 3,240 transforms) strict C* gauge equivariance audit."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import time

from p5.study import setup
from p6 import experiment as ex
from p8_manifold.eval_utils import audit_model_symmetry_p8
from p8_manifold.freeze import (
    AUDIT_CONDITIONS,
    FORMAL_PAIRS,
    LEARNED_ARMS,
    REPO,
    RESULTS,
    SEED_BASE,
    SNR_DB,
    verify_freeze,
)
from p8_manifold.offgrid_sim import simulate_offgrid
from p8_manifold.train import load_checkpoint


def _audit_pair(pair_idx):
    setup()
    t0 = time.perf_counter()
    record = json.loads((RESULTS / "formal" / f"pair{pair_idx:02d}" / "record.json").read_text(encoding="utf-8"))
    frames = {
        cond: simulate_offgrid(1, snr_db=SNR_DB, seed=SEED_BASE + 800000 + 100 * pair_idx + ci, condition=cond)
        for ci, cond in enumerate(AUDIT_CONDITIONS)
    }
    rows = []
    for arm_name, _ in LEARNED_ARMS:
        ck_path = REPO / record["checkpoints"][arm_name]["path"]
        model, _ = load_checkpoint(ck_path)
        for ci, cond in enumerate(AUDIT_CONDITIONS):
            r = audit_model_symmetry_p8(
                model,
                frames[cond],
                seed=SEED_BASE + 850000 + 100 * pair_idx + ci,
                threshold=1e-12,
            )
            r["pair_idx"] = pair_idx
            r["arm"] = arm_name
            rows.append(r)
    return dict(pair_idx=pair_idx, rows=rows, elapsed_seconds=time.perf_counter() - t0)


def main():
    parser = argparse.ArgumentParser(description="Formal Phase 8 Manifold C* Gauge Equivariance Audit")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    setup()
    freeze_sha = verify_freeze()
    t0 = time.perf_counter()
    all_rows = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=setup,
    ) as pool:
        for res in pool.map(_audit_pair, FORMAL_PAIRS):
            all_rows.extend(res["rows"])
            p_rel = sum(1 for r in res["rows"] if r["passed"])
            p_abs = sum(1 for r in res["rows"] if r["passed_strict_abs"])
            print(
                f"Gauge audit pair {res['pair_idx']:02d}: rel={p_rel}/{len(res['rows'])} "
                f"abs={p_abs}/{len(res['rows'])} in {res['elapsed_seconds']:.1f}s",
                flush=True,
            )

    by_arm = {}
    for arm_name, _ in LEARNED_ARMS:
        sub = [r for r in all_rows if r["arm"] == arm_name]
        by_arm[arm_name] = dict(
            passed_rel=sum(1 for r in sub if r["passed"]),
            passed_abs=sum(1 for r in sub if r["passed_strict_abs"]),
            total_conditions=len(sub),
            total_transforms=sum(r["transforms"] for r in sub),
            total_flips=sum(r["coded_flips"] for r in sub),
            all_finite=all(r["all_finite"] for r in sub),
            max_abs_llr_diff=max(r["max_abs_llr_diff"] for r in sub),
            max_rel_llr_diff=max(r["max_rel_llr_diff"] for r in sub),
            max_rel_h=max(r["max_rel_h"] for r in sub),
            max_rel_pred_y=max(r["max_rel_pred_y"] for r in sub),
            max_rel_pred_var=max(r["max_rel_pred_var"] for r in sub),
            max_pi_diff=max(r["max_pi_diff"] for r in sub),
            max_w_diff=max(r["max_w_diff"] for r in sub),
            max_nll_diff=max(r["max_nll_diff"] for r in sub),
            max_p2_rel_llr=max(r["max_p2_rel_llr"] for r in sub),
        )

    out = dict(
        freeze_sha256=freeze_sha,
        threshold=1e-12,
        elapsed_seconds=time.perf_counter() - t0,
        total_conditions=len(all_rows),
        passed_rel_conditions=sum(1 for r in all_rows if r["passed"]),
        passed_abs_conditions=sum(1 for r in all_rows if r["passed_strict_abs"]),
        total_transforms=sum(r["transforms"] for r in all_rows),
        total_flips=sum(r["coded_flips"] for r in all_rows),
        all_passed_rel=(all(r["passed"] for r in all_rows) and sum(r["coded_flips"] for r in all_rows) == 0),
        all_passed_abs=(all(r["passed_strict_abs"] for r in all_rows) and sum(r["coded_flips"] for r in all_rows) == 0),
        by_arm=by_arm,
        rows=all_rows,
    )
    ex.save(RESULTS / "trained_gauge.json", out)
    print(
        f"Formal Gauge Audit Complete: rel={out['passed_rel_conditions']}/{out['total_conditions']} "
        f"abs={out['passed_abs_conditions']}/{out['total_conditions']} "
        f"({out['total_transforms']} transforms), flips={out['total_flips']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
