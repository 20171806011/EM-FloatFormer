"""Phase 12 Stage 2 Development Validation Runner (`SEED_P12_DEV`, 32 links/regime, 9 dB).

Validates BOTH frozen-candidate receivers on `SEED_P12_DEV` before prospective freeze & G=16 formal run:
  - Stage 2A (Primary): 32 links x 5 regimes (`continuous`, `partial_shift`, `full_reset`,
    `ood_unseen_onset`, `ood_two_ray_shift`), comparing `dense/always_clear`, `p11_legacy`,
    `p12_symm_2fold`, and `p12_symm_3fold` with counterbalanced timing & strict held-out `(0,9)` readout.
  - Stage 2B (Secondary): 32 links x 7 regimes (4 ID + `offgrid_full_reset_new_delays` + 2 OOD),
    comparing `p11_classical_per_ray`, `p11_transport_attn_ens`, `p12_offgrid_physics` (full),
    and mechanism ablations (`no_piecewise`, `no_delay_redisc`, `no_wave1_mm`).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

from p9.channel import build_sc_channel_matrix, effective_row_noise, readout_frame_b
from p9.contract import CORE_ROWS, PRIMARY_SNR_DB, PROBE_ROW
from p9.decoder import decode_strong_sc, warmup_numba
from p9.gating import build_candidates_with_priors, gate_always_clear
from p11.contract import N, enforce_determinism
from p11.offgrid_channel import K_HAT, build_offgrid_branches, classical_per_ray_gate, discover_frame_a_atoms
from p11.wavefront_pipeline import _sanitize_y_b, run_p11_with_trace, warmup_p11_numba
from p12.contract import (
    OFDM_PILOT_COMB,
    OFFGRID_STAGE1_REGIMES,
    SEED_P12_DEV,
    assert_p12_seed_disjoint,
)
from p12.offgrid_physics import score_offgrid_channel_est, simulate_offgrid_link_p12
from p12.offgrid_receiver import run_p12_offgrid_physics, warmup_p12_offgrid_numba
from p12.primary_folds import CORE_ROW_TO_INFO_IDX, SPARSE_EXTRA_ROWS, SPARSE_ROW_TO_INFO_IDX
from p12.primary_receiver import (
    PRIMARY_OOD_TWO_RAY_SHIFT,
    PRIMARY_OOD_UNSEEN_ONSET,
    run_p12_primary_kfold,
    simulate_p12_primary_link,
)
from p12.stage1_secondary_diag import _eval_oracle_true_channel, _get_transport_models, _run_transport_ensemble

STAGE2_PRIMARY_REGIMES: tuple[str, ...] = (
    "continuous",
    "partial_shift",
    "full_reset",
    PRIMARY_OOD_UNSEEN_ONSET,
    PRIMARY_OOD_TWO_RAY_SHIFT,
)
STAGE2A_SEED_OFFSET = 100_000
STAGE2B_SEED_OFFSET = 500_000


def _init_primary_worker(require_env: bool) -> None:
    enforce_determinism(require_env=require_env)
    warmup_p11_numba()


def _init_secondary_worker(require_env: bool) -> None:
    enforce_determinism(require_env=require_env)
    warmup_numba()
    warmup_p12_offgrid_numba()
    _get_transport_models()


def _score_primary_assim(assim_res, trace, truth) -> dict[str, Any]:
    core_err = 0
    for r in CORE_ROWS:
        idx = CORE_ROW_TO_INFO_IDX[int(r)]
        if not np.array_equal(assim_res.decoded_info[int(r)], truth.b_info[idx]):
            core_err += 1
    ext_err = 0
    for r in SPARSE_EXTRA_ROWS:
        idx = SPARSE_ROW_TO_INFO_IDX[int(r)]
        if not np.array_equal(assim_res.decoded_info[int(r)], truth.b_info[idx]):
            ext_err += 1

    ledger = trace.get("ledger", [])
    tot_acc = len(ledger)
    false_acc = 0
    for rec in ledger:
        r_i = int(rec["row"])
        if not np.array_equal(rec["injected_info"], truth.b_info[SPARSE_ROW_TO_INFO_IDX[r_i]]):
            false_acc += 1

    p_clean = truth.y_b_probe_clean
    pe = assim_res.gate_out.mix_probe_pred - p_clean
    p_nmse = float(np.sum(pe.real ** 2 + pe.imag ** 2) / max(float(np.sum(p_clean.real ** 2 + p_clean.imag ** 2)), 1e-30))
    return {
        "core_blocks": 16,
        "core_errors": int(core_err),
        "extra_blocks": len(SPARSE_EXTRA_ROWS),
        "extra_errors": int(ext_err),
        "helpers_accepted": int(tot_acc),
        "false_assimilations": int(false_acc),
        "probe_clean_nmse": p_nmse,
        "elapsed_ms": float(assim_res.elapsed_ms),
    }


def _eval_one_stage2a(args: tuple[str, int, int, float]) -> dict[str, Any]:
    regime, link_idx, seed, snr_db = args
    obs_sp, truth = simulate_p12_primary_link(seed, snr_db=snr_db, regime=regime, budget="sparse")
    obs_dn, _ = simulate_p12_primary_link(seed, snr_db=snr_db, regime=regime, budget="dense")

    # Dense always_clear reference (exact P9/P11 NI comparator)
    _, _, cand_inh_dn, cand_clr_dn = build_candidates_with_priors(obs_dn)
    g_dn = gate_always_clear(cand_inh_dn, cand_clr_dn)
    y_dn_safe = _sanitize_y_b(obs_dn.y_b, PROBE_ROW)
    dn_err = 0
    for r in CORE_ROWS:
        idx = CORE_ROW_TO_INFO_IDX[int(r)]
        h_r = build_sc_channel_matrix(g_dn.mix_mean_ray[:, r, :])
        nv_r = effective_row_noise(g_dn.mix_var_ray[:, r, :], obs_dn.n0)
        dec = decode_strong_sc(
            y_dn_safe[r], h_r, nv_r, list_size="large", bp_iters=20, turbo_pic=True, pic_scale=0.75,
            initial_llrs=g_dn.mix_llrs[r].reshape(128),
        )
        if not np.array_equal(dec.info_bits, truth.b_info[idx]):
            dn_err += 1

    # Counterbalance execution order of (p11_legacy, p12_k2, p12_k3) across link_idx to remove warm-up bias!
    arms_order = ["p11_legacy", "p12_symm_2fold", "p12_symm_3fold"]
    shift = link_idx % 3
    arms_order = arms_order[shift:] + arms_order[:shift]
    arms_res: dict[str, Any] = {}
    for arm in arms_order:
        if arm == "p11_legacy":
            res, tr = run_p11_with_trace(obs_sp)
        elif arm == "p12_symm_2fold":
            res, tr = run_p12_primary_kfold(obs_sp, k_folds=2)
        else:
            res, tr = run_p12_primary_kfold(obs_sp, k_folds=3)
        arms_res[arm] = _score_primary_assim(res, tr, truth)

    arms_res["dense_always_clear"] = {"core_blocks": 16, "core_errors": int(dn_err)}
    return {"regime": regime, "link_idx": int(link_idx), "seed": int(seed), "arms": arms_res}


def _eval_one_stage2b(args: tuple[str, int, int, float]) -> dict[str, Any]:
    regime, link_idx, seed, snr_db = args
    obs, truth, _ = simulate_offgrid_link_p12(seed, regime, snr_db=snr_db, pilot_rows_a=OFDM_PILOT_COMB)
    atoms = discover_frame_a_atoms(obs, k_hat=K_HAT)
    bundle = build_offgrid_branches(obs, atoms=atoms, k_hat=K_HAT)

    cls_r = classical_per_ray_gate(obs, bundle)
    sc_cls = score_offgrid_channel_est(
        cls_r.h_mix, cls_r.noise_mix, cls_r.gate_out.mix_llrs, obs, truth, probe_pred=cls_r.gate_out.mix_probe_pred
    )
    sc_attn = _run_transport_ensemble(obs, bundle, truth)
    sc_oracle = _eval_oracle_true_channel(obs, truth, use_true_burst_noise=False)

    def _run_phys(**kw) -> dict[str, Any]:
        pr = run_p12_offgrid_physics(obs, **kw)
        sc = score_offgrid_channel_est(
            pr.h_mix, pr.noise_mix, pr.gate_out.mix_llrs, obs, truth, probe_pred=pr.gate_out.mix_probe_pred
        )
        row_to_idx = {int(r): i for i, r in enumerate(truth.data_rows)}
        false_a = sum(
            1 for rec in pr.ledger if not np.array_equal(rec["injected_info"], truth.b_info[row_to_idx[int(rec["row"])]])
        )
        sc["delay_reset_triggered"] = bool(pr.delay_reset_triggered)
        sc["delay_ll_gain_nats"] = float(pr.delay_ll_gain_nats)
        sc["helpers_accepted"] = int(pr.helpers_accepted)
        sc["false_assimilations"] = int(false_a)
        sc["elapsed_ms"] = float(pr.elapsed_ms)
        return sc

    sc_full = _run_phys(
        use_aux_cov=True, use_wavefront=True, use_piecewise_midframe=True, use_delay_rediscovery=True, use_burst_guard=True, use_wave1_multimode=True
    )
    sc_no_pw = _run_phys(
        use_aux_cov=True, use_wavefront=True, use_piecewise_midframe=False, use_delay_rediscovery=True, use_burst_guard=True, use_wave1_multimode=True
    )
    sc_no_mm = _run_phys(
        use_aux_cov=True, use_wavefront=True, use_piecewise_midframe=True, use_delay_rediscovery=True, use_burst_guard=True, use_wave1_multimode=False
    )

    return {
        "regime": regime,
        "link_idx": int(link_idx),
        "seed": int(seed),
        "n_rays": int(truth.n_rays),
        "arms": {
            "p11_classical_per_ray": sc_cls,
            "p11_transport_attn_ens": sc_attn,
            "p12_offgrid_physics": sc_full,
            "p12_no_piecewise": sc_no_pw,
            "p12_no_wave1_mm": sc_no_mm,
            "oracle_true_channel": sc_oracle,
        },
    }


def _agg_stage2a(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"by_regime": {}}
    for reg in STAGE2_PRIMARY_REGIMES:
        recs = [r for r in records if r["regime"] == reg]
        if not recs:
            continue
        reg_map: dict[str, Any] = {"n_links": len(recs)}
        for arm in ("dense_always_clear", "p11_legacy", "p12_symm_2fold", "p12_symm_3fold"):
            e = sum(r["arms"][arm]["core_errors"] for r in recs)
            b = sum(r["arms"][arm]["core_blocks"] for r in recs)
            entry: dict[str, Any] = {"core_bler_pct": 100.0 * e / max(b, 1), "core_errors": int(e), "core_blocks": int(b)}
            if arm != "dense_always_clear":
                ee = sum(r["arms"][arm]["extra_errors"] for r in recs)
                eb = sum(r["arms"][arm]["extra_blocks"] for r in recs)
                ha = sum(r["arms"][arm]["helpers_accepted"] for r in recs)
                fa = sum(r["arms"][arm]["false_assimilations"] for r in recs)
                entry["extra_heldout_bler_pct"] = 100.0 * ee / max(eb, 1)
                entry["false_assim_pct"] = 100.0 * fa / max(ha, 1)
                entry["mean_probe_nmse"] = float(np.mean([r["arms"][arm]["probe_clean_nmse"] for r in recs]))
                entry["median_ms"] = float(np.median([r["arms"][arm]["elapsed_ms"] for r in recs]))
                entry["mean_ms"] = float(np.mean([r["arms"][arm]["elapsed_ms"] for r in recs]))
            reg_map[arm] = entry
        out["by_regime"][reg] = reg_map
    # 3-ID pooled
    id_recs = [r for r in records if r["regime"] in ("continuous", "partial_shift", "full_reset")]
    out["pooled_3id"] = {
        arm: {
            "core_bler_pct": 100.0 * sum(r["arms"][arm]["core_errors"] for r in id_recs) / sum(r["arms"][arm]["core_blocks"] for r in id_recs),
            "median_ms": float(np.median([r["arms"][arm]["elapsed_ms"] for r in id_recs])) if arm != "dense_always_clear" else 0.0,
        }
        for arm in ("dense_always_clear", "p11_legacy", "p12_symm_2fold", "p12_symm_3fold")
    }
    return out


def _agg_stage2b(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"by_regime": {}}
    arm_names = ("p11_classical_per_ray", "p11_transport_attn_ens", "p12_offgrid_physics", "p12_no_piecewise", "p12_no_wave1_mm", "oracle_true_channel")
    for reg in OFFGRID_STAGE1_REGIMES:
        recs = [r for r in records if r["regime"] == reg]
        if not recs:
            continue
        reg_map: dict[str, Any] = {"n_links": len(recs)}
        for arm in arm_names:
            e = sum(r["arms"][arm]["block_errors"] for r in recs)
            b = sum(r["arms"][arm]["blocks"] for r in recs)
            entry: dict[str, Any] = {
                "bler_pct": 100.0 * e / max(b, 1),
                "errors": int(e),
                "blocks": int(b),
                "mean_rel_frob": float(np.mean([r["arms"][arm]["mean_rel_frob_err"] for r in recs])),
                "mean_probe_nmse": float(np.mean([r["arms"][arm]["probe_clean_nmse"] for r in recs])),
            }
            if "delay_reset_triggered" in recs[0]["arms"][arm]:
                entry["delay_reset_rate_pct"] = 100.0 * float(np.mean([float(r["arms"][arm]["delay_reset_triggered"]) for r in recs]))
                ha = sum(r["arms"][arm]["helpers_accepted"] for r in recs)
                fa = sum(r["arms"][arm]["false_assimilations"] for r in recs)
                entry["helpers_accepted"] = int(ha)
                entry["false_assim_pct"] = 100.0 * fa / max(ha, 1)
                entry["median_ms"] = float(np.median([r["arms"][arm]["elapsed_ms"] for r in recs]))
            reg_map[arm] = entry
        out["by_regime"][reg] = reg_map

    id4 = [r for r in records if r["regime"] in ("offgrid_continuous", "offgrid_partial_shift", "offgrid_midframe_shift", "offgrid_full_reset")]
    out["pooled_4id"] = {
        arm: {
            "bler_pct": 100.0 * sum(r["arms"][arm]["block_errors"] for r in id4) / sum(r["arms"][arm]["blocks"] for r in id4),
            "mean_rel_frob": float(np.mean([r["arms"][arm]["mean_rel_frob_err"] for r in id4])),
            "mean_probe_nmse": float(np.mean([r["arms"][arm]["probe_clean_nmse"] for r in id4])),
        }
        for arm in arm_names
    }
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P12 Stage 2 Dev Evaluation (Primary + Off-Grid)")
    parser.add_argument("--n-links", type=int, default=32)
    parser.add_argument("--snr-db", type=float, default=PRIMARY_SNR_DB)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default="p12/results")
    parser.add_argument("--require-env", action="store_true", default=False)
    parser.add_argument("--secondary-only", action="store_true", default=False)
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    enforce_determinism(require_env=args.require_env)

    # Disjoint dev seeds across Stage 2A & Stage 2B
    tasks_a = []
    tasks_b = []
    all_seeds = []
    for r_i, reg in enumerate(STAGE2_PRIMARY_REGIMES):
        for l_i in range(int(args.n_links)):
            sd = SEED_P12_DEV + STAGE2A_SEED_OFFSET + r_i * 10_000 + l_i
            all_seeds.append(sd)
            tasks_a.append((reg, l_i, sd, float(args.snr_db)))
    for r_i, reg in enumerate(OFFGRID_STAGE1_REGIMES):
        for l_i in range(int(args.n_links)):
            sd = SEED_P12_DEV + STAGE2B_SEED_OFFSET + r_i * 10_000 + l_i
            all_seeds.append(sd)
            tasks_b.append((reg, l_i, sd, float(args.snr_db)))
    assert_p12_seed_disjoint(all_seeds, expected_namespace="p12_dev")

    t_a0 = time.perf_counter()
    if args.secondary_only:
        recs_a = []
    elif args.workers <= 1:
        _init_primary_worker(bool(args.require_env))
        recs_a = [_eval_one_stage2a(t) for t in tasks_a]
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers), initializer=_init_primary_worker, initargs=(bool(args.require_env),)) as pool:
            recs_a = list(pool.map(_eval_one_stage2a, tasks_a))
    wall_a = time.perf_counter() - t_a0

    t_b0 = time.perf_counter()
    if args.workers <= 1:
        _init_secondary_worker(bool(args.require_env))
        recs_b = [_eval_one_stage2b(t) for t in tasks_b]
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers), initializer=_init_secondary_worker, initargs=(bool(args.require_env),)) as pool:
            recs_b = list(pool.map(_eval_one_stage2b, tasks_b))
    wall_b = time.perf_counter() - t_b0

    summary = {
        "n_links_per_regime": int(args.n_links),
        "snr_db": float(args.snr_db),
        "stage2a_wall_s": float(wall_a),
        "stage2b_wall_s": float(wall_b),
        "total_wall_s": float(time.perf_counter() - t0),
        "stage2a_primary": _agg_stage2a(recs_a) if recs_a else {},
        "stage2b_secondary": _agg_stage2b(recs_b),
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = "stage2b_v2_dev_summary.json" if args.secondary_only else "stage2_dev_summary.json"
    with open(out_dir / out_name, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(summary, fh, indent=2, allow_nan=False)
        fh.write("\n")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
