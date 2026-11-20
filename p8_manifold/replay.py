"""Independent checkpoint replay verification across all 40 formal Phase 8 Manifold checkpoints."""
import json

from p5.study import setup
from p6 import experiment as ex
from p8_manifold.eval_utils import evaluate_em8_batch
from p8_manifold.freeze import (
    BATCH_SIZE,
    FORMAL_PAIRS,
    LEARNED_ARMS,
    REPO,
    RESULTS,
    SNR_DB,
    sha256_file,
    tensor_sha256,
    verify_freeze,
)
from p8_manifold.offgrid_sim import OFFGRID_CONDITIONS, simulate_offgrid
from p8_manifold.train import load_checkpoint


def main():
    setup()
    freeze_sha = verify_freeze()
    checks = []
    passed = 0
    total = 0

    for pair_idx in FORMAL_PAIRS:
        record = json.loads((RESULTS / "formal" / f"pair{pair_idx:02d}" / "record.json").read_text(encoding="utf-8"))
        if record.get("freeze_sha256") != freeze_sha:
            raise AssertionError(f"Freeze hash mismatch on pair {pair_idx}")

        frames_b0 = {}
        for cond in OFFGRID_CONDITIONS:
            b0 = next(d for d in record["batch_digests"] if d["condition"] == cond and d["batch_index"] == 0)
            frame = simulate_offgrid(BATCH_SIZE, snr_db=SNR_DB, seed=b0["eval_seed"], condition=cond)
            if (
                tensor_sha256(frame.obs.y) != b0["obs_y_sha256"]
                or tensor_sha256(frame.bits) != b0["bits_sha256"]
                or tensor_sha256(frame.obs.pilot_mask) != b0["pilot_mask_sha256"]
            ):
                raise AssertionError(f"Batch 0 digest mismatch on pair {pair_idx} {cond}")
            frames_b0[cond] = frame

        for arm_name, _ in LEARNED_ARMS:
            total += 1
            ck_info = record["checkpoints"][arm_name]
            ck_path = REPO / ck_info["path"]
            ck_sha = sha256_file(ck_path)
            sha_ok = ck_sha == ck_info["sha256"]
            model, _ = load_checkpoint(ck_path)

            arm_ok = sha_ok
            max_bce_diff = 0.0
            max_nll_diff = 0.0
            cond_diffs = {}
            for cond in OFFGRID_CONDITIONS:
                ref = next(
                    r for r in record["rows"]
                    if r["batch_index"] == 0 and r["condition"] == cond and r["arm"] == arm_name
                )
                got = evaluate_em8_batch(arm_name, model, frames_b0[cond], cond, 0)
                blk_diff = abs(got["score"]["block_errors"] - ref["score"]["block_errors"])
                bit_diff = abs(got["score"]["bit_errors"] - ref["score"]["bit_errors"])
                bce_diff = abs(got["score"]["bce_sum"] - ref["score"]["bce_sum"])
                nll_diff = 0.0
                p2_blk_diff = 0
                if cond == "O3_holdout_sequential":
                    nll_diff = abs(got["holdout"]["nll"] - ref["holdout"]["nll"])
                    p2_blk_diff = abs(
                        got["sequential"]["pass2_block_errors"] - ref["sequential"]["pass2_block_errors"]
                    )
                max_bce_diff = max(max_bce_diff, bce_diff)
                max_nll_diff = max(max_nll_diff, nll_diff)
                ok = (blk_diff == 0 and bit_diff == 0 and p2_blk_diff == 0 and bce_diff < 1e-12 and nll_diff < 1e-12)
                if not ok:
                    arm_ok = False
                cond_diffs[cond] = dict(
                    blk_diff=blk_diff, bit_diff=bit_diff, p2_blk_diff=p2_blk_diff,
                    bce_diff=bce_diff, nll_diff=nll_diff,
                )
            if arm_ok:
                passed += 1
            checks.append(
                dict(
                    pair_idx=pair_idx,
                    arm=arm_name,
                    checkpoint=ck_info["path"],
                    sha_ok=sha_ok,
                    max_bce_diff=max_bce_diff,
                    max_nll_diff=max_nll_diff,
                    cond_diffs=cond_diffs,
                    passed=arm_ok,
                )
            )

    summary = dict(
        freeze_sha256=freeze_sha,
        total_checkpoints=total,
        passed_checkpoints=passed,
        passed=(passed == total),
        checks=checks,
    )
    ex.save(RESULTS / "checkpoint_replay.json", summary)
    print(f"Checkpoint Replay: {passed}/{total} PASSED={summary['passed']}", flush=True)
    if not summary["passed"]:
        raise AssertionError("Checkpoint replay failed")


if __name__ == "__main__":
    main()
