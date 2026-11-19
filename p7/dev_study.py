"""Stage 2 dev-only training, evaluation, and C* symmetry audit on pairs 901..903."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import time

import torch

from p5.study import setup
from p6 import experiment as ex
from p6.data import dev_frame, dev_seed
from p6_followup.common import load_model
from p7.conditions import CONDITIONS, PILOT_AUDIT_CONDITIONS
from p7.eval_utils import (
    audit_model_symmetry, evaluate_em7_batch, evaluate_p6_consistent_batch,
)
from p7.model import EM7HypothesisReceiver
from p7.train import save_checkpoint, train_em7_model


REPO = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / "results"
DEV_PAIRS = (901, 902, 903)
ARMS = ("EM7-Axial", "EM7-Local", "EM7-Fixed", "EM7-SingleHyp-Axial", "P6-Aware-Consistent")


def _train_and_eval_pair(pair):
    setup()
    t0 = time.perf_counter()
    init_seed = dev_seed(2, 5, pair, 900)
    frame_fn = lambda step, bs: dev_frame(2, 5, pair, bs, batch=step)

    models = {"EM7-Fixed": EM7HypothesisReceiver("fixed").eval()}
    train_traces = {}
    checkpoints = {}
    for arm_name, variant in (
        ("EM7-Axial", "axial"),
        ("EM7-Local", "local"),
        ("EM7-SingleHyp-Axial", "single_axial"),
    ):
        ck_path = RESULTS / "dev" / f"pair{pair}" / f"{variant}.pt"
        model, trace = train_em7_model(
            variant, init_seed, frame_fn,
            steps=40, lr=0.004, lambda_hold=0.05, lambda_nmse=0.005, anchor_l2=5e-5,
            batch_size=8, mask_seed_base=pair * 1000,
        )
        ck_sha = save_checkpoint(ck_path, model, dict(pair=pair, init_seed=init_seed, trace_len=len(trace["history"])))
        models[arm_name] = model
        train_traces[arm_name] = trace
        checkpoints[arm_name] = dict(path=ck_path.relative_to(REPO).as_posix(), sha256=ck_sha)

    p6_model, _ = load_model(pair, "parity", development=True)

    rows = []
    for batch_idx in range(4):
        frame = dev_frame(2, 6, pair, 8, batch=batch_idx)
        cond_seed = dev_seed(2, 6, pair, 100 + batch_idx)
        for cond in CONDITIONS:
            for arm_name, model in models.items():
                rows.append(evaluate_em7_batch(arm_name, model, frame, cond, batch_idx, cond_seed))
            rows.append(evaluate_p6_consistent_batch(p6_model, frame, cond, batch_idx, cond_seed))

    audit_frame = dev_frame(2, 6, pair, 2, batch=0)
    audits = []
    for arm_name in ("EM7-Axial", "EM7-Local", "EM7-SingleHyp-Axial"):
        for idx, cond in enumerate(PILOT_AUDIT_CONDITIONS):
            row = audit_model_symmetry(models[arm_name], audit_frame, cond, dev_seed(2, 6, pair, 900 + idx))
            row["pair"] = pair
            row["arm"] = arm_name
            audits.append(row)

    return dict(
        pair=pair, seconds=time.perf_counter() - t0,
        checkpoints=checkpoints, train_traces=train_traces, rows=rows, audits=audits,
    )


def _aggregate_dev(all_rows):
    agg = {cond: {} for cond in CONDITIONS}
    for cond in CONDITIONS:
        for arm in ARMS:
            sub = [r for r in all_rows if r["condition"] == cond and r["arm"] == arm]
            blk_err = sum(r["score"]["block_errors"] for r in sub)
            blks = sum(r["score"]["blocks"] for r in sub)
            bce = sum(r["score"]["bce_sum"] for r in sub)
            bits = sum(r["score"]["bit_count"] for r in sub)
            entry = dict(
                block_errors=blk_err, blocks=blks, bler=blk_err / blks,
                bit_bce=bce / bits,
            )
            if "cal_stats" in sub[0]:
                b0_n = sum(r["cal_stats"]["bin0_count"] for r in sub)
                b0_v = sum(r["cal_stats"]["bin0_sum_var"] for r in sub)
                b0_e = sum(r["cal_stats"]["bin0_sum_mse"] for r in sub)
                tot_v = sum(r["cal_stats"]["tot_sum_var"] for r in sub)
                tot_e = sum(r["cal_stats"]["tot_sum_mse"] for r in sub)
                entry["bin0_count"] = b0_n
                entry["bin0_ratio"] = (b0_e / b0_v) if b0_v > 0 else None
                entry["total_ratio"] = (tot_e / tot_v) if tot_v > 0 else None
            if cond == "C5_holdout_update" and "holdout" in sub[0]:
                h_nll = sum(r["holdout"]["holdout_nll_sum"] for r in sub)
                h_cnt = sum(r["holdout"]["probe_count"] for r in sub)
                h_sq = sum(r["holdout"]["holdout_sq_err"] for r in sub)
                h_ref = sum(r["holdout"]["holdout_ref_pow"] for r in sub)
                h_var = sum(r["holdout"]["holdout_pred_var_sum"] for r in sub)

                hi_n = sum(r["sequential"]["high_count"] for r in sub)
                lo_n = sum(r["sequential"]["low_count"] for r in sub)
                ps_hi = sum(r["sequential"]["high_disagreement_prob_shift_sum"] for r in sub) / max(hi_n, 1)
                ps_lo = sum(r["sequential"]["low_disagreement_prob_shift_sum"] for r in sub) / max(lo_n, 1)
                fl_hi = sum(r["sequential"]["high_disagreement_flip_sum"] for r in sub) / max(hi_n, 1)
                fl_lo = sum(r["sequential"]["low_disagreement_flip_sum"] for r in sub) / max(lo_n, 1)
                ls_hi = sum(r["sequential"]["high_disagreement_llr_shift_sum"] for r in sub) / max(hi_n, 1)
                ls_lo = sum(r["sequential"]["low_disagreement_llr_shift_sum"] for r in sub) / max(lo_n, 1)

                sp_n = sum(r["sequential"]["split_count"] for r in sub)
                ce_n = sum(r["sequential"]["certain_count"] for r in sub)
                if sp_n > 0 and ce_n > 0:
                    sp_ps = sum(r["sequential"]["split_prob_shift_sum"] for r in sub) / sp_n
                    ce_ps = sum(r["sequential"]["certain_prob_shift_sum"] for r in sub) / ce_n
                    sp_fl = sum(r["sequential"]["split_flip_sum"] for r in sub) / sp_n
                    ce_fl = sum(r["sequential"]["certain_flip_sum"] for r in sub) / ce_n
                    entry["seq_split_prob_ratio"] = sp_ps / max(ce_ps, 1e-12)
                    entry["seq_split_flip_ratio"] = sp_fl / max(ce_fl, 1e-12)
                else:
                    entry["seq_split_prob_ratio"] = None
                    entry["seq_split_flip_ratio"] = None

                p_corrs = [r["sequential"]["prob_shift_corr"] for r in sub]
                l_corrs = [r["sequential"]["disagreement_llr_shift_corr"] for r in sub]
                d_blks = sum(r["sequential"]["delta_block_errors"] for r in sub)
                d_bces = [r["sequential"]["delta_bce"] for r in sub]

                entry["holdout_nll"] = h_nll / h_cnt
                entry["holdout_nmse"] = h_sq / h_ref
                entry["holdout_cal_ratio"] = h_sq / h_var
                entry["seq_prob_shift_high"] = ps_hi
                entry["seq_prob_shift_low"] = ps_lo
                entry["seq_prob_shift_ratio"] = ps_hi / max(ps_lo, 1e-12)
                entry["seq_prob_shift_corr"] = sum(p_corrs) / len(p_corrs)
                entry["seq_flip_high"] = fl_hi
                entry["seq_flip_low"] = fl_lo
                entry["seq_flip_ratio"] = fl_hi / max(fl_lo, 1e-12)
                entry["seq_llr_shift_ratio"] = ls_hi / max(ls_lo, 1e-12)
                entry["seq_llr_shift_corr"] = sum(l_corrs) / len(l_corrs)
                entry["seq_delta_blocks"] = d_blks
                entry["seq_delta_bce"] = sum(d_bces) / len(d_bces)
            elif cond == "C5_holdout_update" and "sequential" in sub[0]:
                entry["seq_delta_blocks"] = sum(r["sequential"]["delta_block_errors"] for r in sub)
            agg[cond][arm] = entry
    return agg


def main():
    setup()
    t0 = time.perf_counter()
    pair_results = []
    with ProcessPoolExecutor(max_workers=3, mp_context=multiprocessing.get_context("spawn"), initializer=setup) as pool:
        for res in pool.map(_train_and_eval_pair, DEV_PAIRS):
            print(f"Finished dev pair {res['pair']} in {res['seconds']:.1f}s", flush=True)
            pair_results.append(res)

    all_rows = [r for pr in pair_results for r in pr["rows"]]
    all_audits = [a for pr in pair_results for a in pr["audits"]]
    summary = _aggregate_dev(all_rows)

    audit_pass = sum(1 for a in all_audits if a["passed"])
    audit_total = len(all_audits)
    audit_transforms = sum(a["transforms"] for a in all_audits)
    audit_flips = sum(a["coded_flips"] for a in all_audits)
    audit_max_llr = max(a["max_llr_diff"] for a in all_audits)

    c1_ax = summary["C1_row9_erased"]["EM7-Axial"]
    c1_sh = summary["C1_row9_erased"]["EM7-SingleHyp-Axial"]
    c1_fx = summary["C1_row9_erased"]["EM7-Fixed"]
    c1_p6 = summary["C1_row9_erased"]["P6-Aware-Consistent"]
    c4_ax = summary["C4a_roving_window"]["EM7-Axial"]
    c4_fx = summary["C4a_roving_window"]["EM7-Fixed"]
    c4_sh = summary["C4a_roving_window"]["EM7-SingleHyp-Axial"]
    c5_ax = summary["C5_holdout_update"]["EM7-Axial"]
    c5_fx = summary["C5_holdout_update"]["EM7-Fixed"]
    c5_sh = summary["C5_holdout_update"]["EM7-SingleHyp-Axial"]

    gate_checks = dict(
        symmetry_all_passed=(audit_pass == audit_total and audit_flips == 0),
        c1_cal_under_4=(c1_ax["bin0_ratio"] is not None and c1_ax["bin0_ratio"] < 4.0),
        c1_cal_beats_single_and_p6=(
            c1_ax["bin0_ratio"] < c1_sh["bin0_ratio"] and c1_ax["bin0_ratio"] < c1_p6["bin0_ratio"]
        ),
        c5_nll_beats_single_and_fixed=(
            c5_ax["holdout_nll"] < c5_sh["holdout_nll"] and c5_ax["holdout_nll"] < c5_fx["holdout_nll"]
        ),
        c5_shift_selective=(
            c5_ax["seq_prob_shift_ratio"] > 2.0
            and c5_ax["seq_flip_ratio"] > 2.0
            and c5_ax["seq_prob_shift_corr"] > 0.0
        ),
        c4_beats_fixed_and_single_bce_or_bler=(
            (c4_ax["block_errors"] <= c4_fx["block_errors"] and c4_ax["bit_bce"] < c4_fx["bit_bce"])
            and (c4_ax["block_errors"] <= c4_sh["block_errors"] and c4_ax["bit_bce"] < c4_sh["bit_bce"])
        ),
        c1_beats_single_and_fixed_bler=(
            c1_ax["block_errors"] <= c1_fx["block_errors"] and c1_ax["block_errors"] <= c1_sh["block_errors"]
        ),
    )
    gate_passed = all(gate_checks.values())

    report = dict(
        elapsed_seconds=time.perf_counter() - t0,
        pairs=list(DEV_PAIRS),
        summary=summary,
        audit_summary=dict(
            passed_conditions=audit_pass, total_conditions=audit_total,
            total_transforms=audit_transforms, total_flips=audit_flips,
            max_llr_diff=audit_max_llr,
        ),
        gate_checks=gate_checks,
        gate_passed=gate_passed,
        pair_details=[{k: v for k, v in pr.items() if k != "rows"} for pr in pair_results],
    )
    ex.save(RESULTS / "dev_gate.json", report)

    print("\n=== Phase 7 Dev Gate Summary (pairs 901..903, stream 6, 1728 blocks/cond) ===", flush=True)
    for cond in CONDITIONS:
        print(f"\nCondition: {cond}", flush=True)
        for arm in ARMS:
            s = summary[cond][arm]
            r0 = f"{s['bin0_ratio']:.2f}x" if s.get("bin0_ratio") is not None else "n/a"
            extra = ""
            if cond == "C5_holdout_update" and "holdout_nll" in s:
                extra = (f" | hold_nll={s['holdout_nll']:.3f} hold_nmse={s['holdout_nmse']:.3f} "
                         f"prob_ratio={s['seq_prob_shift_ratio']:.2f}x flip_ratio={s['seq_flip_ratio']:.2f}x "
                         f"corr={s['seq_prob_shift_corr']:.3f}")
            print(f"  {arm:22s} BLER={s['block_errors']:4d}/{s['blocks']} ({100*s['bler']:.2f}%) "
                  f"BCE={s['bit_bce']:.4f} | <0.01 ratio={r0}{extra}", flush=True)
    print(f"\nDev Symmetry Audit: {audit_pass}/{audit_total} ({audit_transforms} transforms), "
          f"flips={audit_flips}, max_dLLR={audit_max_llr:.3e}", flush=True)
    print("Dev Gate Checks:", gate_checks, "PASSED=" + str(gate_passed), flush=True)


if __name__ == "__main__":
    main()
