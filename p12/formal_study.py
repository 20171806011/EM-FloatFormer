"""Phase 12 Formal G=16 Closed-Book Evaluation Runner (Stage 3 Primary + Stage 4 Secondary + ID/OOD Exact Replay).

Strictly enforces:
  - Prospective freeze verification (`verify_freeze`) & `--confirm-frozen` before touching any
    `SEED_P12_FORMAL_ID` / `SEED_P12_FORMAL_OOD` seed.
  - Strictly disjoint 3,584 unique formal seeds verified via `planned_formal_seeds()`.
  - Counterbalanced timing order across primary arms.
  - Mandatory single-worker exact replay on BOTH ID and OOD for `REPLAY_CLUSTERS = (1, 16)` (`max_float_diff == 0.0`).
  - Automatic independent audit (`p12_audit.json`) and report generation (`p12/P12_REPORT.md`).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np

from p9.channel import build_sc_channel_matrix, effective_row_noise
from p9.contract import CORE_ROWS, PRIMARY_SNR_DB, PROBE_ROW
from p9.decoder import decode_strong_sc, warmup_numba
from p9.gating import build_candidates_with_priors, gate_always_clear
from p11.contract import enforce_determinism
from p11.offgrid_channel import K_HAT, build_offgrid_branches, classical_per_ray_gate, discover_frame_a_atoms
from p11.wavefront_pipeline import _sanitize_y_b, run_p11_with_trace, warmup_p11_numba
from p12.audit import audit_p12_formal
from p12.contract import OFDM_PILOT_COMB, SEED_P12_FORMAL_ID, SEED_P12_FORMAL_OOD
from p12.freeze import (
    DEFAULT_MANIFEST_PATH,
    FORMAL_CLUSTERS,
    OFFGRID_FORMAL_OFFSET,
    OFFGRID_ID_REGIMES,
    OFFGRID_LINKS_PER_CLUSTER_ID,
    OFFGRID_LINKS_PER_CLUSTER_OOD,
    OFFGRID_OOD_REGIMES,
    PRIMARY_FORMAL_OFFSET,
    PRIMARY_ID_REGIMES,
    PRIMARY_LINKS_PER_CLUSTER_ID,
    PRIMARY_LINKS_PER_CLUSTER_OOD,
    PRIMARY_OOD_REGIMES,
    REPLAY_CLUSTERS,
    RESULTS_DIR,
    planned_formal_seeds,
    verify_freeze,
)
from p12.offgrid_physics import score_offgrid_channel_est, simulate_offgrid_link_p12
from p12.offgrid_receiver import run_p12_offgrid_physics, warmup_p12_offgrid_numba
from p12.primary_folds import CORE_ROW_TO_INFO_IDX, SPARSE_EXTRA_ROWS, SPARSE_ROW_TO_INFO_IDX, decode_rows_strong_l
from p12.primary_receiver import run_p12_primary_kfold, simulate_p12_primary_link
from p12.report import write_p12_report
from p12.stage1_secondary_diag import _eval_oracle_true_channel, _get_transport_models, _run_transport_ensemble


def _init_primary_worker(require_env: bool) -> None:
    enforce_determinism(require_env=require_env)
    warmup_p11_numba()


def _init_secondary_worker(require_env: bool) -> None:
    enforce_determinism(require_env=require_env)
    warmup_numba()
    warmup_p12_offgrid_numba()
    _get_transport_models()


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            for r in records:
                fh.write(json.dumps(r, allow_nan=False) + "\n")
            fh.flush()
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=2, allow_nan=False)
            fh.write("\n")
            fh.flush()
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


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


def _eval_one_primary(task: tuple[str, int, str, int, int, float]) -> dict[str, Any]:
    split, cluster, regime, link_idx, seed, snr_db = task
    obs_sp, truth = simulate_p12_primary_link(seed, snr_db=snr_db, regime=regime, budget="sparse")
    obs_dn, _ = simulate_p12_primary_link(seed, snr_db=snr_db, regime=regime, budget="dense")

    # Dense always_clear NI reference
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

    # True CSI oracle reference
    y_sp_safe = _sanitize_y_b(obs_sp.y_b, PROBE_ROW)
    v_zero = np.zeros((3, len(CORE_ROWS), 32), dtype=np.float64)
    dec_or, _ = decode_rows_strong_l(
        truth.h_b_true[:, list(CORE_ROWS), :], v_zero, y_sp_safe[list(CORE_ROWS)], CORE_ROWS, obs_sp.n0
    )
    or_err = sum(
        int(not np.array_equal(dec_or[int(r)], truth.b_info[CORE_ROW_TO_INFO_IDX[int(r)]]))
        for r in CORE_ROWS
    )

    # Counterbalanced execution order of sparse arms
    arms_order = ["p11_legacy", "p12_symm_2fold", "p12_symm_3fold"]
    shift = (cluster + link_idx) % 3
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
    arms_res["oracle_true_csi"] = {"core_blocks": 16, "core_errors": int(or_err)}

    return {
        "split": split,
        "cluster": int(cluster),
        "regime": regime,
        "link": int(link_idx),
        "seed": int(seed),
        "snr_db": float(snr_db),
        "arms": arms_res,
    }


def _eval_one_secondary(task: tuple[str, int, str, int, int, float]) -> dict[str, Any]:
    split, cluster, regime, link_idx, seed, snr_db = task
    obs, truth, extra = simulate_offgrid_link_p12(seed, regime, snr_db=snr_db, pilot_rows_a=OFDM_PILOT_COMB)
    atoms = discover_frame_a_atoms(obs, k_hat=K_HAT)
    bundle = build_offgrid_branches(obs, atoms=atoms, k_hat=K_HAT)

    cls_r = classical_per_ray_gate(obs, bundle)
    sc_cls = score_offgrid_channel_est(
        cls_r.h_mix, cls_r.noise_mix, cls_r.gate_out.mix_llrs, obs, truth, probe_pred=cls_r.gate_out.mix_probe_pred
    )
    sc_attn = _run_transport_ensemble(obs, bundle, truth)
    sc_oracle = _eval_oracle_true_channel(obs, truth, use_true_burst_noise=False)

    # Mask-matched B-refit-off control via identical P12 entrypoint
    pr_off = run_p12_offgrid_physics(
        obs, use_aux_cov=False, use_wavefront=False, use_delay_rediscovery=False, use_burst_guard=False
    )
    sc_off = score_offgrid_channel_est(
        pr_off.h_mix, pr_off.noise_mix, pr_off.gate_out.mix_llrs, obs, truth, probe_pred=pr_off.gate_out.mix_probe_pred
    )

    # Full frozen P12 off-grid pure-physics receiver
    pr_full = run_p12_offgrid_physics(
        obs,
        use_aux_cov=True,
        use_wavefront=True,
        use_piecewise_midframe=True,
        use_delay_rediscovery=True,
        use_burst_guard=True,
        use_wave1_multimode=True,
    )
    sc_full = score_offgrid_channel_est(
        pr_full.h_mix, pr_full.noise_mix, pr_full.gate_out.mix_llrs, obs, truth, probe_pred=pr_full.gate_out.mix_probe_pred
    )
    row_to_idx = {int(r): i for i, r in enumerate(truth.data_rows)}
    false_a = sum(
        1 for rec in pr_full.ledger if not np.array_equal(rec["injected_info"], truth.b_info[row_to_idx[int(rec["row"])]])
    )
    sc_full["delay_reset_triggered"] = bool(pr_full.delay_reset_triggered)
    sc_full["delay_ll_gain_nats"] = float(pr_full.delay_ll_gain_nats)
    sc_full["helpers_accepted"] = int(pr_full.helpers_accepted)
    sc_full["false_assimilations"] = int(false_a)
    sc_full["elapsed_ms"] = float(pr_full.elapsed_ms)

    return {
        "split": split,
        "cluster": int(cluster),
        "regime": regime,
        "link": int(link_idx),
        "seed": int(seed),
        "snr_db": float(snr_db),
        "n_rays": int(truth.n_rays),
        "anchor_row": int(extra["anchor_row"]),
        "arms": {
            "p11_classical_per_ray": sc_cls,
            "p11_transport_attn_ens": sc_attn,
            "p12_brefit_off_control": sc_off,
            "p12_offgrid_physics": sc_full,
            "oracle_true_channel": sc_oracle,
        },
    }


def _build_tasks(snr_db: float) -> tuple[list, list, list, list]:
    p_id = [
        ("id", c, reg, l_i, int(SEED_P12_FORMAL_ID + PRIMARY_FORMAL_OFFSET + r_i * 50_000 + 1000 * c + l_i), snr_db)
        for c in FORMAL_CLUSTERS
        for r_i, reg in enumerate(PRIMARY_ID_REGIMES)
        for l_i in range(PRIMARY_LINKS_PER_CLUSTER_ID)
    ]
    p_ood = [
        ("ood", c, reg, l_i, int(SEED_P12_FORMAL_OOD + PRIMARY_FORMAL_OFFSET + r_i * 50_000 + 1000 * c + l_i), snr_db)
        for c in FORMAL_CLUSTERS
        for r_i, reg in enumerate(PRIMARY_OOD_REGIMES)
        for l_i in range(PRIMARY_LINKS_PER_CLUSTER_OOD)
    ]
    s_id = [
        ("id", c, reg, l_i, int(SEED_P12_FORMAL_ID + OFFGRID_FORMAL_OFFSET + r_i * 50_000 + 1000 * c + l_i), snr_db)
        for c in FORMAL_CLUSTERS
        for r_i, reg in enumerate(OFFGRID_ID_REGIMES)
        for l_i in range(OFFGRID_LINKS_PER_CLUSTER_ID)
    ]
    s_ood = [
        ("ood", c, reg, l_i, int(SEED_P12_FORMAL_OOD + OFFGRID_FORMAL_OFFSET + r_i * 50_000 + 1000 * c + l_i), snr_db)
        for c in FORMAL_CLUSTERS
        for r_i, reg in enumerate(OFFGRID_OOD_REGIMES)
        for l_i in range(OFFGRID_LINKS_PER_CLUSTER_OOD)
    ]
    return p_id, p_ood, s_id, s_ood


def _run_pool(tasks: list, fn, init_fn, workers: int, require_env: bool, label: str) -> list[dict[str, Any]]:
    t0 = time.perf_counter()
    if workers <= 1:
        init_fn(require_env)
        out = [fn(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=int(workers), initializer=init_fn, initargs=(require_env,)) as pool:
            out = list(pool.map(fn, tasks))
    print(f"[{label}] completed {len(out)} links in {time.perf_counter() - t0:.1f}s (workers={workers})", flush=True)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P12 Formal G=16 Closed-Book Runner (Stages 3 & 4)")
    parser.add_argument("--confirm-frozen", action="store_true", required=True)
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out-dir", type=str, default=str(RESULTS_DIR))
    parser.add_argument("--require-env", action="store_true", default=False)
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    enforce_determinism(require_env=args.require_env)
    manifest = verify_freeze(Path(args.manifest))
    planned_formal_seeds()
    snr_db = float(manifest["snr_db"])

    p_id_t, p_ood_t, s_id_t, s_ood_t = _build_tasks(snr_db)
    p_rep_t = [t for t in (p_id_t + p_ood_t) if int(t[1]) in REPLAY_CLUSTERS]
    s_rep_t = [t for t in (s_id_t + s_ood_t) if int(t[1]) in REPLAY_CLUSTERS]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Stage 3: Primary Axis ID + OOD (multi-worker) + ID/OOD exact replay (single-worker)
    p_id_recs = _run_pool(p_id_t, _eval_one_primary, _init_primary_worker, args.workers, args.require_env, "Primary-ID")
    _write_jsonl(out_dir / "p12_primary_id_links.jsonl", p_id_recs)

    p_ood_recs = _run_pool(p_ood_t, _eval_one_primary, _init_primary_worker, args.workers, args.require_env, "Primary-OOD")
    _write_jsonl(out_dir / "p12_primary_ood_links.jsonl", p_ood_recs)

    p_rep_recs = _run_pool(p_rep_t, _eval_one_primary, _init_primary_worker, 1, args.require_env, "Primary-Replay-1w")
    _write_jsonl(out_dir / "p12_primary_replay_links.jsonl", p_rep_recs)

    # 2) Stage 4: Secondary Axis ID + OOD (multi-worker) + ID/OOD exact replay (single-worker)
    s_id_recs = _run_pool(s_id_t, _eval_one_secondary, _init_secondary_worker, args.workers, args.require_env, "Secondary-ID")
    _write_jsonl(out_dir / "p12_secondary_id_links.jsonl", s_id_recs)

    s_ood_recs = _run_pool(s_ood_t, _eval_one_secondary, _init_secondary_worker, args.workers, args.require_env, "Secondary-OOD")
    _write_jsonl(out_dir / "p12_secondary_ood_links.jsonl", s_ood_recs)

    s_rep_recs = _run_pool(s_rep_t, _eval_one_secondary, _init_secondary_worker, 1, args.require_env, "Secondary-Replay-1w")
    _write_jsonl(out_dir / "p12_secondary_replay_links.jsonl", s_rep_recs)

    # 3) Full independent statistical audit & report generation
    audit = audit_p12_formal(p_id_recs, p_ood_recs, p_rep_recs, s_id_recs, s_ood_recs, s_rep_recs)
    audit["freeze_bundle_sha256"] = manifest["bundle_sha256"]
    audit["total_formal_wall_s"] = float(time.perf_counter() - t0)
    audit["workers"] = int(args.workers)
    _write_json(out_dir / "p12_audit.json", audit)

    report_path = out_dir.parent / "P12_REPORT.md"
    write_p12_report(audit, manifest, report_path)
    print(f"=== P12 FORMAL STUDY COMPLETE in {audit['total_formal_wall_s']:.1f}s ===", flush=True)
    print(json.dumps({
        "gate1_passed": audit["fixed_sequence_gate1_passed"],
        "primary_3id": audit["primary"]["pooled_3id_bler_pct"],
        "primary_full_reset_k3_sub10": {k: audit["primary"]["sub10_k3_full_reset"][k] for k in ("mean_pct", "u95_pct", "verdict")},
        "primary_ni_k3": {k: audit["primary"]["ni_k3_continuous"][k] for k in ("mean_excess_pp", "u95_pp", "verdict")},
        "primary_replay": audit["primary"]["replay_audit"],
        "secondary_4id": audit["secondary"]["pooled_4id_bler_pct"],
        "secondary_replay": audit["secondary"]["replay_audit"],
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
