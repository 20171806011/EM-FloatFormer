"""Independent checkpoint replay verification across all 30 formal Phase 7 checkpoints."""
import json
from pathlib import Path

from p5.study import setup
from p5.wavefield_sim import simulate
from p6 import experiment as ex
from p7.eval_utils import evaluate_em7_batch
from p7.freeze import FORMAL_PAIRS, REPO, RESULTS, SEED_BASE, sha256_file, tensor_sha256, verify_freeze
from p7.train import load_checkpoint


REPLAY_CONDITIONS = ("C1_row9_erased", "C4a_roving_window", "C5_holdout_update")


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
        b0_meta = next(b for b in record["batch_digests"] if b["batch_index"] == 0)
        frame = simulate(2, 8, 9, b0_meta["eval_seed"])
        if tensor_sha256(frame.obs.y) != b0_meta["obs_y_sha256"] or tensor_sha256(frame.bits) != b0_meta["bits_sha256"]:
            raise AssertionError(f"Batch 0 waveform/bit digest mismatch on pair {pair_idx}")

        for arm_name in ("EM7-Axial", "EM7-Local", "EM7-SingleHyp-Axial"):
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
            for cond in REPLAY_CONDITIONS:
                ref = next(
                    r for r in record["rows"]
                    if r["batch_index"] == 0 and r["condition"] == cond and r["arm"] == arm_name
                )
                got = evaluate_em7_batch(arm_name, model, frame, cond, 0, b0_meta["cond_seed"])
                blk_diff = abs(got["score"]["block_errors"] - ref["score"]["block_errors"])
                bit_diff = abs(got["score"]["bit_errors"] - ref["score"]["bit_errors"])
                bce_diff = abs(got["score"]["bce_sum"] - ref["score"]["bce_sum"])
                nll_diff = 0.0
                if cond == "C5_holdout_update":
                    nll_diff = abs(got["holdout"]["holdout_nll"] - ref["holdout"]["holdout_nll"])
                max_bce_diff = max(max_bce_diff, bce_diff)
                max_nll_diff = max(max_nll_diff, nll_diff)
                ok = (blk_diff == 0 and bit_diff == 0 and bce_diff < 1e-12 and nll_diff < 1e-12)
                if not ok:
                    arm_ok = False
                cond_diffs[cond] = dict(blk_diff=blk_diff, bit_diff=bit_diff, bce_diff=bce_diff, nll_diff=nll_diff)
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
