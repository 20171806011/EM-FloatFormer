"""Formal 30-checkpoint x 5-condition x 27-scale (= 4,050 transforms) strict C* symmetry audit."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import time

from p5.study import setup
from p5.wavefield_sim import simulate
from p6 import experiment as ex
from p7.conditions import PILOT_AUDIT_CONDITIONS
from p7.eval_utils import audit_model_symmetry
from p7.freeze import FORMAL_PAIRS, REPO, RESULTS, SEED_BASE, verify_freeze
from p7.train import load_checkpoint


def _audit_pair(pair_idx):
    setup()
    t0 = time.perf_counter()
    record = json.loads((RESULTS / "formal" / f"pair{pair_idx:02d}" / "record.json").read_text(encoding="utf-8"))
    frame = simulate(2, 1, 9, SEED_BASE + 800000 + pair_idx)
    rows = []
    for arm_name in ("EM7-Axial", "EM7-Local", "EM7-SingleHyp-Axial"):
        ck_path = REPO / record["checkpoints"][arm_name]["path"]
        model, _ = load_checkpoint(ck_path)
        for idx, cond in enumerate(PILOT_AUDIT_CONDITIONS):
            r = audit_model_symmetry(
                model,
                frame,
                cond,
                SEED_BASE + 850000 + 100 * pair_idx + idx,
                threshold=1e-12,
            )
            r["pair_idx"] = pair_idx
            r["arm"] = arm_name
            rows.append(r)
    return dict(pair_idx=pair_idx, rows=rows, elapsed_seconds=time.perf_counter() - t0)


def main():
    setup()
    freeze_sha = verify_freeze()
    t0 = time.perf_counter()
    all_rows = []
    with ProcessPoolExecutor(
        max_workers=5,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=setup,
    ) as pool:
        for res in pool.map(_audit_pair, FORMAL_PAIRS):
            all_rows.extend(res["rows"])
            p_ok = sum(1 for r in res["rows"] if r["passed"])
            print(
                f"Gauge audit pair {res['pair_idx']:02d}: {p_ok}/{len(res['rows'])} passed in {res['elapsed_seconds']:.1f}s",
                flush=True,
            )

    by_arm = {}
    for arm_name in ("EM7-Axial", "EM7-Local", "EM7-SingleHyp-Axial"):
        sub = [r for r in all_rows if r["arm"] == arm_name]
        by_arm[arm_name] = dict(
            passed_conditions=sum(1 for r in sub if r["passed"]),
            total_conditions=len(sub),
            total_transforms=sum(r["transforms"] for r in sub),
            total_flips=sum(r["coded_flips"] for r in sub),
            all_finite=all(r["all_finite"] for r in sub),
            max_llr_diff=max(r["max_llr_diff"] for r in sub),
            max_rel_pred_y=max(r["max_rel_pred_y"] for r in sub),
            max_rel_pred_var=max(r["max_rel_pred_var"] for r in sub),
            max_pi_diff=max(r["max_pi_diff"] for r in sub),
            max_nll_diff=max(r["max_nll_diff"] for r in sub),
            failures=[r for r in sub if not r["passed"]],
        )

    out = dict(
        freeze_sha256=freeze_sha,
        threshold=1e-12,
        elapsed_seconds=time.perf_counter() - t0,
        total_conditions=len(all_rows),
        passed_conditions=sum(1 for r in all_rows if r["passed"]),
        total_transforms=sum(r["transforms"] for r in all_rows),
        total_flips=sum(r["coded_flips"] for r in all_rows),
        all_passed=(all(r["passed"] for r in all_rows) and sum(r["coded_flips"] for r in all_rows) == 0),
        by_arm=by_arm,
        rows=all_rows,
    )
    ex.save(RESULTS / "trained_gauge.json", out)
    print(
        f"Formal Gauge Audit Complete: {out['passed_conditions']}/{out['total_conditions']} "
        f"({out['total_transforms']} transforms), flips={out['total_flips']}, all_passed={out['all_passed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
