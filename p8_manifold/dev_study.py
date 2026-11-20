"""Stage 8.2 Phase 8 Development Pilot & Architectural Gate Study (dev seeds only)."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import torch

from p5.study import setup
from p8_manifold.eval_utils import (
    audit_model_symmetry_p8,
    evaluate_em8_batch,
    evaluate_lmmse_batch,
    summarize_ratio,
)
from p8_manifold.model import EM8ManifoldReceiver
from p8_manifold.offgrid_sim import OFFGRID_CONDITIONS, simulate_offgrid
from p8_manifold.train import save_checkpoint, train_em8_model


DEV_PAIRS = (801, 802, 803)
DEV_SEED_BASE = 88000000
EVAL_BATCHES = 4
BATCH_SIZE = 8
SNR_DB = 15.0

LEARNED_ARMS = (
    ("EM8-Transport", "transport"),
    ("EM8-AbsNoTransport", "abs_no_transport"),
    ("EM8-Local", "local"),
    ("EM8-SingleHyp", "single_transport"),
)


def _pair_seed(pair_id, stream):
    return DEV_SEED_BASE + int(pair_id) * 10000 + int(stream)


def _run_dev_pair(args):
    pair_id, steps, d_model, n_layers, n_heads, lr, ckpt_dir = args
    setup()
    t0 = time.perf_counter()
    init_seed = _pair_seed(pair_id, 1)
    train_seed_base = _pair_seed(pair_id, 100)

    models = {}
    train_info = {}
    ckpt_sha = {}
    for arm_name, variant in LEARNED_ARMS:
        model, info = train_em8_model(
            variant,
            init_seed=init_seed,
            train_seed_base=train_seed_base,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            steps=steps,
            lr=lr,
            snr_db=SNR_DB,
            batch_size=BATCH_SIZE,
        )
        models[arm_name] = model
        train_info[arm_name] = dict(
            elapsed_seconds=info["elapsed_seconds"],
            real_parameters=model.real_parameters(),
            final_bce=info["history"][-1]["bce"],
            final_loss=info["history"][-1]["loss"],
            final_hold_nll=info["history"][-1]["hold_nll"],
        )
        ckpt_path = Path(ckpt_dir) / f"dev_pair{pair_id:03d}_{arm_name}.pt"
        ckpt_sha[arm_name] = save_checkpoint(
            ckpt_path, model, dict(pair_id=pair_id, arm=arm_name, init_seed=init_seed, steps=steps)
        )

    models["EM8-Fixed"] = EM8ManifoldReceiver(
        variant="fixed", d_model=d_model, n_layers=n_layers, n_heads=n_heads, steps=2
    ).eval()

    rows = []
    for c_idx, cond in enumerate(OFFGRID_CONDITIONS):
        for b_idx in range(EVAL_BATCHES):
            eval_seed = _pair_seed(pair_id, 5000 + c_idx * 100 + b_idx)
            frame = simulate_offgrid(BATCH_SIZE, snr_db=SNR_DB, seed=eval_seed, condition=cond)
            for arm_name, model in models.items():
                rows.append(evaluate_em8_batch(arm_name, model, frame, cond, b_idx))
            rows.append(evaluate_lmmse_batch(frame, cond, b_idx))

    # Dev C* symmetry audit on pair_id == 801 across O1, O2, O3
    sym_audits = []
    if pair_id == DEV_PAIRS[0]:
        for c_idx, cond in enumerate(("O1_roving_island", "O2_blind_burst_erasure", "O3_holdout_sequential")):
            a_frame = simulate_offgrid(2, snr_db=SNR_DB, seed=_pair_seed(pair_id, 9000 + c_idx), condition=cond)
            for arm_name in ("EM8-Transport", "EM8-AbsNoTransport", "EM8-Local", "EM8-SingleHyp"):
                res = audit_model_symmetry_p8(models[arm_name], a_frame, seed=_pair_seed(pair_id, 9100 + c_idx))
                res["arm"] = arm_name
                sym_audits.append(res)

    return dict(
        pair_id=pair_id,
        elapsed_seconds=time.perf_counter() - t0,
        train_info=train_info,
        ckpt_sha=ckpt_sha,
        rows=rows,
        sym_audits=sym_audits,
    )


def aggregate_rows(pair_results):
    """Aggregate condition x arm metrics across all dev pairs."""
    all_rows = [r for pr in pair_results for r in pr["rows"]]
    arms = list(dict.fromkeys(r["arm"] for r in all_rows))
    summary = {}
    for cond in OFFGRID_CONDITIONS:
        summary[cond] = {}
        for arm in arms:
            sub = [r for r in all_rows if r["condition"] == cond and r["arm"] == arm]
            blk_err = sum(r["score"]["block_errors"] for r in sub)
            blks = sum(r["score"]["blocks"] for r in sub)
            bce_sum = sum(r["score"]["bce_sum"] for r in sub)
            bit_cnt = sum(r["score"]["bit_count"] for r in sub)
            ch_err = sum(r["score"]["channel_error"] for r in sub)
            ch_eng = sum(r["score"]["channel_energy"] for r in sub)
            b0_var = sum(r["cal_stats"]["bin0_sum_var"] for r in sub)
            b0_mse = sum(r["cal_stats"]["bin0_sum_mse"] for r in sub)
            b0_cnt = sum(r["cal_stats"]["bin0_count"] for r in sub)
            tot_var = sum(r["cal_stats"]["tot_sum_var"] for r in sub)
            tot_mse = sum(r["cal_stats"]["tot_sum_mse"] for r in sub)
            entry = dict(
                block_errors=blk_err,
                blocks=blks,
                bler=blk_err / max(blks, 1),
                bce=bce_sum / max(bit_cnt, 1),
                nmse=ch_err / max(ch_eng, 1e-300),
                bin0_count=b0_cnt,
                bin0_ratio=summarize_ratio(b0_mse, b0_var),
                tot_ratio=summarize_ratio(tot_mse, tot_var),
            )
            if any("blind_weights" in r for r in sub):
                cw = sum(r["blind_weights"]["clean_w_sum"] for r in sub)
                cc = sum(r["blind_weights"]["clean_count"] for r in sub)
                bw = sum(r["blind_weights"]["corrupt_w_sum"] for r in sub)
                bc = sum(r["blind_weights"]["corrupt_count"] for r in sub)
                c_mean = cw / max(cc, 1)
                b_mean = bw / max(bc, 1)
                entry["blind_clean_w"] = c_mean
                entry["blind_corrupt_w"] = b_mean
                entry["blind_rejection_ratio"] = c_mean / max(b_mean, 1e-12)
            if any("holdout" in r for r in sub):
                nll_avg = sum(r["holdout"]["nll"] * r["holdout"]["count"] for r in sub) / max(
                    sum(r["holdout"]["count"] for r in sub), 1
                )
                hmse = sum(r["holdout"]["mse_sum"] for r in sub)
                heng = sum(r["holdout"]["energy_sum"] for r in sub)
                hvar = sum(r["holdout"]["var_sum"] for r in sub)
                entry["holdout_nll"] = nll_avg
                entry["holdout_nmse"] = hmse / max(heng, 1e-300)
                entry["holdout_cal_ratio"] = hmse / max(hvar, 1e-300)
                entry["holdout_var_aleatoric_frac"] = sum(r["holdout"]["var_aleatoric_sum"] for r in sub) / max(hvar, 1e-300)
                entry["holdout_var_epistemic_frac"] = sum(r["holdout"]["var_epistemic_sum"] for r in sub) / max(hvar, 1e-300)
                entry["holdout_var_between_frac"] = sum(r["holdout"]["var_between_sum"] for r in sub) / max(hvar, 1e-300)
            if any("sequential" in r for r in sub):
                p1_err = sum(r["sequential"]["pass1_block_errors"] for r in sub)
                p2_err = sum(r["sequential"]["pass2_block_errors"] for r in sub)
                entry["pass1_block_errors"] = p1_err
                entry["pass2_block_errors"] = p2_err
                entry["delta_block_errors"] = p1_err - p2_err
                entry["delta_bce"] = sum(r["sequential"]["delta_bce"] for r in sub) / len(sub)
                if "prob_shift_ratio" in sub[0]["sequential"]:
                    top_p = sum(r["sequential"]["top25_prob_shift"] for r in sub) / len(sub)
                    bot_p = sum(r["sequential"]["bot25_prob_shift"] for r in sub) / len(sub)
                    top_f = sum(r["sequential"]["top25_flip_rate"] for r in sub) / len(sub)
                    bot_f = sum(r["sequential"]["bot25_flip_rate"] for r in sub) / len(sub)
                    entry["prob_shift_ratio"] = top_p / max(bot_p, 1e-12)
                    entry["flip_rate_ratio"] = top_f / max(bot_f, 1e-12)
                    u_top = sum(r["sequential"]["u_top25_prob_shift"] for r in sub) / len(sub)
                    u_bot = sum(r["sequential"]["u_bot25_prob_shift"] for r in sub) / len(sub)
                    entry["u_prob_shift_ratio"] = u_top / max(u_bot, 1e-12)
            summary[cond][arm] = entry
    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 8.2 Phase 8 Dev Pilot")
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0035)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=str, default="p8_manifold/results/dev_gate.json")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_path.parent / "dev_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    jobs = [
        (pid, args.steps, args.d_model, args.n_layers, args.n_heads, args.lr, str(ckpt_dir))
        for pid in DEV_PAIRS
    ]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
            pair_results = list(ex.map(_run_dev_pair, jobs))
    else:
        pair_results = [_run_dev_pair(j) for j in jobs]

    summary = aggregate_rows(pair_results)
    sym_audits = [a for pr in pair_results for a in pr["sym_audits"]]
    sym_pass_rel = all(a["passed"] for a in sym_audits)
    sym_pass_abs = all(a["passed_strict_abs"] for a in sym_audits)

    # Check non-peeking engineering & convergence gates (never pre-select on the Transport vs Local hypothesis!)
    tr_o1 = summary["O1_roving_island"]["EM8-Transport"]
    lo_o1 = summary["O1_roving_island"]["EM8-Local"]
    ab_o1 = summary["O1_roving_island"]["EM8-AbsNoTransport"]
    sh_o1 = summary["O1_roving_island"]["EM8-SingleHyp"]
    fx_o1 = summary["O1_roving_island"]["EM8-Fixed"]
    tr_o0 = summary["O0_random_sparse"]["EM8-Transport"]
    lo_o0 = summary["O0_random_sparse"]["EM8-Local"]
    ab_o0 = summary["O0_random_sparse"]["EM8-AbsNoTransport"]
    tr_o2 = summary["O2_blind_burst_erasure"]["EM8-Transport"]
    lm_o2 = summary["O2_blind_burst_erasure"]["OffGrid-2D-LMMSE"]
    tr_o3 = summary["O3_holdout_sequential"]["EM8-Transport"]

    gates = dict(
        learned_beats_fixed_o1=(tr_o1["bler"] < fx_o1["bler"] and tr_o1["bce"] < fx_o1["bce"]),
        multichart_beats_singlehyp_o1=(tr_o1["bler"] < sh_o1["bler"] and tr_o1["bce"] < sh_o1["bce"]),
        blind_o2_beats_lmmse=(tr_o2["bler"] < lm_o2["bler"]),
        o3_assimilate_positive_gain=(tr_o3["delta_block_errors"] > 0 and tr_o3["delta_bce"] > 0),
        symmetry_rel_pass=sym_pass_rel,
        symmetry_abs_pass=sym_pass_abs,
        # Descriptive diagnostics only (tested formally on fresh N=10 pairs, never gated on dev!)
        diag_transport_beats_local_o1_bler=(tr_o1["bler"] < lo_o1["bler"]),
        diag_transport_beats_local_o0_bler=(tr_o0["bler"] < lo_o0["bler"]),
        diag_transport_beats_abs_o1_or_o0=(tr_o1["bce"] < ab_o1["bce"] or tr_o0["bce"] < ab_o0["bce"]),
    )
    gate_passed = (
        gates["learned_beats_fixed_o1"]
        and gates["multichart_beats_singlehyp_o1"]
        and gates["blind_o2_beats_lmmse"]
        and gates["o3_assimilate_positive_gain"]
        and gates["symmetry_rel_pass"]
    )

    payload = dict(
        config=dict(
            pairs=list(DEV_PAIRS),
            steps=args.steps,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_heads=args.n_heads,
            lr=args.lr,
            eval_batches=EVAL_BATCHES,
            batch_size=BATCH_SIZE,
            snr_db=SNR_DB,
        ),
        elapsed_seconds=time.perf_counter() - t0,
        train_info={f"pair_{pr['pair_id']}": pr["train_info"] for pr in pair_results},
        summary=summary,
        sym_audits=sym_audits,
        gates=gates,
        gate_passed=gate_passed,
    )
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n=== Stage 8.2 Dev Pilot Completed in {payload['elapsed_seconds']:.1f}s ===", flush=True)
    for cond in OFFGRID_CONDITIONS:
        print(f"\n[{cond}]", flush=True)
        for arm, st in summary[cond].items():
            line = (
                f"  {arm:18s}: BLER={st['block_errors']:4d}/{st['blocks']} ({100*st['bler']:5.2f}%) "
                f"BCE={st['bce']:.4f} NMSE={st['nmse']:.4f} "
                f"<0.01={st['bin0_ratio'] if st['bin0_ratio'] is not None else -1:.2f}x "
                f"tot={st['tot_ratio'] if st['tot_ratio'] is not None else -1:.2f}x"
            )
            if "blind_rejection_ratio" in st:
                line += f" w_clean={st['blind_clean_w']:.3f} w_bad={st['blind_corrupt_w']:.3f}"
            if "holdout_nll" in st:
                line += (
                    f" H-NLL={st['holdout_nll']:.3f} H-Cal={st['holdout_cal_ratio']:.2f}x"
                    f" P1->P2={st['pass1_block_errors']}->{st['pass2_block_errors']} ({st['delta_block_errors']:+d})"
                )
            if "prob_shift_ratio" in st:
                line += f" selP={st['prob_shift_ratio']:.2f}x selF={st['flip_rate_ratio']:.2f}x"
            print(line, flush=True)
    print(f"\nSymmetry audits: rel={sym_pass_rel}, abs={sym_pass_abs}", flush=True)
    print(f"Gates: {gates} => gate_passed={gate_passed}", flush=True)


if __name__ == "__main__":
    main()
