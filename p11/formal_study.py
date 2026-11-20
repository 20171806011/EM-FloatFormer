"""Formal G=16 closed-book evaluation runner (Primary Axis + Secondary Axis + single-worker exact replay).

Strictly enforces:
  - Pre-experiment freeze verification (`verify_freeze`) & `--confirm-frozen` before touching
    any `SEED_P11_FORMAL_ID` / `SEED_P11_FORMAL_OOD` seed.
  - `--mode dry_run` uses strictly `SEED_P11_DEV` (`enforce_determinism(require_env=False)`).
  - Lazy M2 imports (`p11.wavefront_pipeline`, `p11.payload_rebuild`) inside Primary execution only.
  - Disjoint Primary vs Secondary seed sub-ranges verified via `check_p11_seed_disjointness(..., strict=True)`.
  - Single-worker Primary Axis + mandatory single-worker exact replay of `REPLAY_CLUSTERS = (1, 16)`.
  - Owner-row `11 + 5 = 16` pooling for Primary, `18`-block off-grid scoring for Secondary,
    and `p11.audit` verification producing `p11_audit.json`.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import inspect
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from p8.coding import decode as p8_decode
from p8c.link import extra_rows
from p9.channel import (
    build_sc_channel_matrix,
    effective_row_noise,
    oracle_readout_b,
    readout_frame_b,
    simulate_link,
)
from p9.contract import (
    CORE_ROWS,
    PRIMARY_SNR_DB,
    PROBE_ROW,
    REGIMES_ID,
    REGIMES_OOD,
    REGIMES_SANITY,
)
from p9.decoder import decode_strong_sc, warmup_numba
from p9.gating import (
    build_candidates_with_priors,
    gate_always_clear,
    gate_always_inherit,
    gate_jeffreys_student_t,
)
from p11.audit import (
    _audit_single_holm_family,
    _aggregate_nested_budget_arm,
    aggregate_p11_cells,
    audit_engineering_noninferiority,
    audit_p11_offgrid,
    audit_p11_primary,
    summarize_common_phase_blindspot,
    verify_replay_exact,
)
from p11.contract import (
    ANCHOR,
    N,
    OFFGRID_ALL_REGIMES,
    OFFGRID_REGIMES_ID,
    OFFGRID_REGIMES_OOD,
    SEED_P11_DEV,
    SEED_P11_E2E,
    SEED_P11_FORMAL_ID,
    SEED_P11_FORMAL_OOD,
    SEED_P11_TRAIN,
    T,
    GateOutput,
    enforce_determinism,
)
from p11.freeze import (
    CKPT_DIR,
    DEFAULT_MANIFEST_PATH,
    FORMAL_CLUSTERS,
    LINKS_PER_CLUSTER_ID,
    LINKS_PER_CLUSTER_OOD,
    OFFGRID_LINKS_PER_CLUSTER_ID,
    OFFGRID_LINKS_PER_CLUSTER_OOD,
    OFFGRID_SEEDS,
    OFFGRID_VARIANTS,
    REPLAY_CLUSTERS,
    RESULTS_DIR,
    verify_freeze,
)
from p11.independent_ref import check_p11_seed_disjointness


AUX1 = (2, 5, 8, 13, 16)
AUX2 = (1, 4, 7, 12, 15)
FOLD0_TARGETS = tuple(r for r in CORE_ROWS if r not in AUX1)  # 11 rows
FOLD1_TARGETS = AUX1                                         # 5 rows
CORE_ROW_TO_INFO_IDX = {r: idx for idx, r in enumerate(CORE_ROWS)}

_LOADED_OFFGRID_MODELS: Optional[dict[str, list[tuple[int, Any]]]] = None


def _init_secondary_worker(require_env: bool) -> None:
    enforce_determinism(require_env=require_env)
    warmup_numba()


def _get_offgrid_models() -> dict[str, list[tuple[int, Any]]]:
    global _LOADED_OFFGRID_MODELS
    if _LOADED_OFFGRID_MODELS is None:
        import torch
        from p11.manifold_gate import ManifoldTrustGate

        models: dict[str, list[tuple[int, Any]]] = {}
        for var in OFFGRID_VARIANTS:
            models[var] = []
            for seed in OFFGRID_SEEDS:
                ckpt_path = CKPT_DIR / f"{var}_seed{seed}.pt"
                payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                model = ManifoldTrustGate(variant=var, **payload["arch"])
                model.load_state_dict(payload["state_dict"], strict=True)
                model.eval()
                if model.real_parameters() != 3153:
                    raise ValueError(
                        f"Checkpoint {ckpt_path} expected 3153 params, got {model.real_parameters()}"
                    )
                models[var].append((seed, model))
        _LOADED_OFFGRID_MODELS = models
    return _LOADED_OFFGRID_MODELS


def _reject_nonstandard_json_constant(const: str) -> Any:
    raise ValueError(f"Non-standard JSON constant {const!r} not allowed")


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8", newline="\n") as fh:
            for r in records:
                fh.write(json.dumps(r, allow_nan=False) + "\n")
            fh.flush()
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
        os.replace(tmp_path, path)
    except BaseException:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _load_resumed_jsonl(
    path: Path,
    valid_keys: set[tuple[str, int, str, int, int]],
) -> dict[tuple[str, int, str, int, int], dict[str, Any]]:
    loaded: dict[tuple[str, int, str, int, int], dict[str, Any]] = {}
    if not path.is_file():
        return loaded
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line, parse_constant=_reject_nonstandard_json_constant)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(rec, dict):
            continue
        sp = rec.get("split")
        cl = rec.get("cluster")
        rg = rec.get("regime")
        lk = rec.get("link")
        sd = rec.get("seed")
        budgets = rec.get("budgets")
        if (
            isinstance(sp, str)
            and type(cl) is int
            and isinstance(rg, str)
            and type(lk) is int
            and type(sd) is int
            and isinstance(budgets, dict)
            and "sparse" in budgets
            and isinstance(budgets["sparse"], dict)
            and len(budgets["sparse"]) > 0
        ):
            key = (sp, cl, rg, lk, sd)
            if key in valid_keys:
                loaded[key] = rec
    return loaded


def _planned_primary_seeds(
    clusters: Sequence[int],
    n_id: int,
    n_ood: int,
    id_base: int,
    ood_base: int,
    id_offset: int,
    ood_offset: int,
) -> tuple[list[int], list[int]]:
    id_seeds = [
        int(id_base + id_offset + r_idx * 50_000 + 1000 * c + l_idx)
        for c in clusters
        for r_idx in range(len(REGIMES_ID))
        for l_idx in range(n_id)
    ]
    ood_regimes = REGIMES_OOD + REGIMES_SANITY
    ood_seeds = [
        int(ood_base + ood_offset + r_idx * 50_000 + 1000 * c + l_idx)
        for c in clusters
        for r_idx in range(len(ood_regimes))
        for l_idx in range(n_ood)
    ]
    return id_seeds, ood_seeds


def _planned_secondary_seeds(
    clusters: Sequence[int],
    n_id: int,
    n_ood: int,
    id_base: int,
    ood_base: int,
    id_offset: int,
    ood_offset: int,
) -> tuple[list[int], list[int]]:
    id_seeds = [
        int(id_base + id_offset + r_idx * 50_000 + 1000 * c + l_idx)
        for c in clusters
        for r_idx in range(len(OFFGRID_REGIMES_ID))
        for l_idx in range(n_id)
    ]
    ood_seeds = [
        int(ood_base + ood_offset + r_idx * 50_000 + 1000 * c + l_idx)
        for c in clusters
        for r_idx in range(len(OFFGRID_REGIMES_OOD))
        for l_idx in range(n_ood)
    ]
    return id_seeds, ood_seeds


# ---------------------------------------------------------------------------
# Primary Axis Scoring (Lazy M2 Imports, 11+5 Pooling, Injected-Word FA)
# ---------------------------------------------------------------------------

def _probe_stats(truth_probe_noisy: np.ndarray, truth_probe_clean: np.ndarray, pred: np.ndarray, pvar: np.ndarray) -> dict[str, float]:
    if not (np.isfinite(pred).all() and np.isfinite(pvar).all() and (pvar > 0.0).all()):
        raise ValueError("Non-finite probe prediction or non-positive variance")
    return {
        "probe_sse": float(np.sum(np.abs(truth_probe_noisy - pred) ** 2)),
        "clean_sse": float(np.sum(np.abs(truth_probe_clean - pred) ** 2)),
        "probe_energy": float(np.sum(np.abs(truth_probe_noisy) ** 2)),
        "clean_energy": float(np.sum(np.abs(truth_probe_clean) ** 2)),
        "variance_sum": float(np.sum(pvar)),
        "nll_sum": float(np.sum(np.abs(truth_probe_noisy - pred) ** 2 / pvar + np.log(math.pi * pvar))),
    }


def _score_primary_gate_arm(obs, truth, gate_out: GateOutput, *, run_extra: bool = True) -> dict[str, Any]:
    core_rows = list(CORE_ROWS)
    free_rows = list(extra_rows(obs.training_rows)) if run_extra else []

    t0 = time.perf_counter_ns()
    f0_err = 0
    f1_err = 0
    info_err = 0
    for idx_c, row in enumerate(core_rows):
        h_r = build_sc_channel_matrix(gate_out.mix_mean_ray[:, row])
        noise_r = effective_row_noise(gate_out.mix_var_ray, obs.n0, rows=(row,))[0]
        res = decode_strong_sc(
            obs.y_b[row],
            h_r,
            noise_r,
            list_size="large",
            bp_iters=20,
            turbo_pic=True,
            pic_scale=0.75,
            initial_llrs=gate_out.mix_llrs[row].reshape(128),
        )
        diff = res.info_bits != truth.b_info[idx_c]
        if bool(diff.any()):
            if row in FOLD0_TARGETS:
                f0_err += 1
            else:
                f1_err += 1
            info_err += int(diff.sum())

    ext_err = 0
    for idx_e, row in enumerate(free_rows):
        h_r = build_sc_channel_matrix(gate_out.mix_mean_ray[:, row])
        noise_r = effective_row_noise(gate_out.mix_var_ray, obs.n0, rows=(row,))[0]
        res = decode_strong_sc(
            obs.y_b[row],
            h_r,
            noise_r,
            list_size="large",
            bp_iters=20,
            turbo_pic=True,
            pic_scale=0.75,
            initial_llrs=gate_out.mix_llrs[row].reshape(128),
        )
        if not np.array_equal(res.info_bits, truth.b_info[16 + idx_e]):
            ext_err += 1

    core_llrs = gate_out.mix_llrs[core_rows].reshape(16, 128)
    coded_err = int(((core_llrs > 0.0) != truth.b_code[:16]).sum())
    decode_ms = (time.perf_counter_ns() - t0) / 1e6

    p_stats = _probe_stats(truth.y_b_probe_noisy, truth.y_b_probe_clean, gate_out.mix_probe_pred, gate_out.mix_probe_var)
    return {
        "blocks": 16,
        "block_errors": int(f0_err + f1_err),
        "fold0_blocks": 11,
        "fold1_blocks": 5,
        "fold0_errors": int(f0_err),
        "fold1_errors": int(f1_err),
        "coded_bits": 2048,
        "coded_errors": coded_err,
        "info_bits": 1024,
        "info_errors": int(info_err),
        "extra_blocks": len(free_rows),
        "extra_errors": int(ext_err),
        "probes": 32,
        **p_stats,
        "decode_ms": decode_ms,
    }


def _call_payload_rebuild_fold(
    payload_mod: Any,
    obs: Any,
    aux: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if hasattr(payload_mod, "estimate_payload_posterior_obs"):
        post = payload_mod.estimate_payload_posterior_obs(obs, aux, use_payload=True)
    else:
        fn = payload_mod.estimate_payload_posterior
        sig = inspect.signature(fn)
        params = list(sig.parameters.keys())
        if params and params[0] == "obs":
            post = fn(obs, aux, use_payload=True)
        else:
            post = fn(obs.y_b[ANCHOR], aux, obs.y_b[list(aux)], obs.n0)
    p_pred = (
        np.asarray(post.probe_pred, dtype=np.complex128)
        if hasattr(post, "probe_pred") and post.probe_pred is not None
        else None
    )
    p_var = (
        np.asarray(post.probe_var, dtype=np.float64)
        if hasattr(post, "probe_var") and post.probe_var is not None
        else None
    )
    return np.asarray(post.mean, dtype=np.complex128), np.asarray(post.var, dtype=np.float64), p_pred, p_var


def _score_primary_standalone_rebuild(payload_mod: Any, obs, truth) -> dict[str, Any]:
    t0 = time.perf_counter_ns()
    free_rows = list(extra_rows(obs.training_rows))
    extra_row_to_idx = {r: 16 + i for i, r in enumerate(free_rows)}

    fold_models = []
    for aux in (AUX1, AUX2):
        mu, var, post_pp, post_pv = _call_payload_rebuild_fold(payload_mod, obs, aux)
        llrs, ro_pp, ro_pv = readout_frame_b(mu, var, obs.y_b, obs.training_rows, obs.n0)
        p_pred = post_pp if post_pp is not None else ro_pp
        p_var = post_pv if post_pv is not None else ro_pv
        fold_models.append((mu, var, llrs, p_pred, p_var))

    f_errs = [0, 0]
    info_err = 0
    coded_err = 0
    ext_err = 0
    oracle_cp_errs = 0
    run_oracle_cp = truth.regime == "full_reset"

    for f_idx, targets in enumerate((FOLD0_TARGETS, FOLD1_TARGETS)):
        mu, var, llrs, _, _ = fold_models[f_idx]
        for row in targets:
            idx_c = CORE_ROW_TO_INFO_IDX[row]
            h_r = build_sc_channel_matrix(mu[:, row])
            noise_r = effective_row_noise(var, obs.n0, rows=(row,))[0]
            res = decode_strong_sc(
                obs.y_b[row],
                h_r,
                noise_r,
                list_size="large",
                bp_iters=20,
                turbo_pic=True,
                pic_scale=0.75,
                initial_llrs=llrs[row].reshape(128),
            )
            diff = res.info_bits != truth.b_info[idx_c]
            if bool(diff.any()):
                f_errs[f_idx] += 1
                info_err += int(diff.sum())
            coded_err += int(((llrs[row].reshape(128) > 0.0) != truth.b_code[idx_c]).sum())

            if run_oracle_cp:
                inner = np.vdot(mu[:, row], truth.h_b_true[:, row])
                phase = inner / abs(inner) if abs(inner) > 1e-14 else 1.0 + 0.0j
                mu_rot = mu.copy()
                mu_rot[:, row] *= phase
                llrs_rot, _, _ = readout_frame_b(mu_rot, var, obs.y_b, obs.training_rows, obs.n0)
                h_rot = build_sc_channel_matrix(mu_rot[:, row])
                res_o = decode_strong_sc(
                    obs.y_b[row],
                    h_rot,
                    noise_r,
                    list_size="large",
                    bp_iters=20,
                    turbo_pic=True,
                    pic_scale=0.75,
                    initial_llrs=llrs_rot[row].reshape(128),
                )
                if not np.array_equal(res_o.info_bits, truth.b_info[idx_c]):
                    oracle_cp_errs += 1

    # Extra rows (0, 9) owned by Fold 0
    mu0, var0, llrs0, _, _ = fold_models[0]
    for row in free_rows:
        h_r = build_sc_channel_matrix(mu0[:, row])
        noise_r = effective_row_noise(var0, obs.n0, rows=(row,))[0]
        res = decode_strong_sc(
            obs.y_b[row],
            h_r,
            noise_r,
            list_size="large",
            bp_iters=20,
            turbo_pic=True,
            pic_scale=0.75,
            initial_llrs=llrs0[row].reshape(128),
        )
        if not np.array_equal(res.info_bits, truth.b_info[extra_row_to_idx[row]]):
            ext_err += 1

    # Equal-weight predictive mixture for probe
    p0, v0 = fold_models[0][3], fold_models[0][4]
    p1, v1 = fold_models[1][3], fold_models[1][4]
    p_mix = 0.5 * (p0 + p1)
    v_mix = 0.5 * (v0 + v1) + 0.25 * (np.abs(p0 - p1) ** 2)
    p_stats = _probe_stats(truth.y_b_probe_noisy, truth.y_b_probe_clean, p_mix, v_mix)
    decode_ms = (time.perf_counter_ns() - t0) / 1e6

    out = {
        "blocks": 16,
        "block_errors": int(f_errs[0] + f_errs[1]),
        "fold0_blocks": 11,
        "fold1_blocks": 5,
        "fold0_errors": int(f_errs[0]),
        "fold1_errors": int(f_errs[1]),
        "coded_bits": 2048,
        "coded_errors": int(coded_err),
        "info_bits": 1024,
        "info_errors": int(info_err),
        "extra_blocks": len(free_rows),
        "extra_errors": int(ext_err),
        "probes": 32,
        **p_stats,
        "decode_ms": decode_ms,
    }
    if run_oracle_cp:
        out["oracle_common_phase_errors"] = int(oracle_cp_errs)
    return out


def _score_primary_chained(wf_mod: Any, obs, truth) -> dict[str, Any]:
    t0 = time.perf_counter_ns()
    if hasattr(wf_mod, "run_p11_with_trace"):
        assim_res, trace = wf_mod.run_p11_with_trace(obs)
    else:
        assim_res = wf_mod.run_p11_primary(obs)
        trace = {}

    free_rows = list(extra_rows(obs.training_rows))
    extra_row_to_idx = {r: 16 + i for i, r in enumerate(free_rows)}
    row_to_any_idx = dict(CORE_ROW_TO_INFO_IDX)
    row_to_any_idx.update(extra_row_to_idx)

    f0_err = 0
    f1_err = 0
    info_err = 0
    for row in CORE_ROWS:
        idx_c = CORE_ROW_TO_INFO_IDX[row]
        dec_bits = np.asarray(assim_res.decoded_info[row], dtype=np.uint8)
        diff = dec_bits != truth.b_info[idx_c]
        if bool(diff.any()):
            if row in FOLD0_TARGETS:
                f0_err += 1
            else:
                f1_err += 1
            info_err += int(diff.sum())

    ext_err = 0
    for row in free_rows:
        dec_bits = np.asarray(assim_res.decoded_info[row], dtype=np.uint8)
        if not np.array_equal(dec_bits, truth.b_info[extra_row_to_idx[row]]):
            ext_err += 1

    core_llrs = assim_res.gate_out.mix_llrs[list(CORE_ROWS)].reshape(16, 128)
    coded_err = int(((core_llrs > 0.0) != truth.b_code[:16]).sum())

    def _extract_injected_info(entry_map: Mapping[str, Any]) -> np.ndarray:
        if "injected_info" in entry_map:
            return np.asarray(entry_map["injected_info"], dtype=np.uint8)
        if "injected_word" in entry_map:
            return np.asarray(entry_map["injected_word"], dtype=np.uint8)
        raise KeyError("Ledger entry missing both 'injected_info' and 'injected_word'")

    # Injected-word false assimilation from trace ledger
    ledger = trace.get("ledger", []) if isinstance(trace, Mapping) else []
    tot_assim = 0
    false_assim = 0
    if isinstance(ledger, Sequence) and ledger:
        for entry in ledger:
            if isinstance(entry, Mapping):
                r_inj = int(entry["row"])
                w_inj = _extract_injected_info(entry)
            else:
                # Tuple format (fold, row, injected_word, alpha, margin, version)
                r_inj = int(entry[1])
                w_inj = np.asarray(entry[2], dtype=np.uint8)
            if r_inj not in row_to_any_idx:
                raise ValueError(f"Illegal ledger row {r_inj}: not a valid public data row")
            tot_assim += 1
            if not np.array_equal(w_inj, truth.b_info[row_to_any_idx[r_inj]]):
                false_assim += 1
    else:
        for r_inj in assim_res.assimilated_rows:
            if r_inj not in row_to_any_idx:
                raise ValueError(f"Illegal assimilated row {r_inj}: not a valid public data row")
            tot_assim += 1
            if not np.array_equal(
                np.asarray(assim_res.decoded_info[r_inj], dtype=np.uint8),
                truth.b_info[row_to_any_idx[r_inj]],
            ):
                false_assim += 1

    # Wave-1 unlock status per fold (wave == 0 in M2 trace["ledger"] / trace["folds"][f]["ledger"])
    def _w1_status(fold_idx: int) -> str:
        key = f"wave1_fold{fold_idx}"
        if isinstance(trace, Mapping) and key in trace and trace[key] in ("correct", "incorrect", "absent"):
            return str(trace[key])
        fold_w1_rows = trace.get("wave1_rows", {}).get(fold_idx, []) if isinstance(trace, Mapping) else []
        if fold_w1_rows:
            for item in fold_w1_rows:
                if isinstance(item, Mapping):
                    r_w = int(item["row"])
                    w_w = _extract_injected_info(item)
                else:
                    r_w = int(item[0])
                    w_w = np.asarray(item[1], dtype=np.uint8)
                if r_w not in row_to_any_idx:
                    raise ValueError(f"Illegal Wave-1 row {r_w} in wave1_rows")
                if not np.array_equal(w_w, truth.b_info[row_to_any_idx[r_w]]):
                    return "incorrect"
            return "correct"

        src_ledger = ledger if isinstance(ledger, Sequence) else []
        if (
            (not src_ledger)
            and isinstance(trace, Mapping)
            and isinstance(trace.get("folds"), Sequence)
            and len(trace["folds"]) > fold_idx
        ):
            fold_obj = trace["folds"][fold_idx]
            if isinstance(fold_obj, Mapping):
                src_ledger = fold_obj.get("ledger", [])
        w1_entries = [
            e
            for e in src_ledger
            if isinstance(e, Mapping) and int(e.get("fold", -1)) == fold_idx and int(e.get("wave", -1)) == 0
        ]
        if not w1_entries:
            return "absent"
        for e in w1_entries:
            r_w = int(e["row"])
            w_w = _extract_injected_info(e)
            if r_w not in row_to_any_idx:
                raise ValueError(f"Illegal Wave-1 row {r_w} in ledger")
            if not np.array_equal(w_w, truth.b_info[row_to_any_idx[r_w]]):
                return "incorrect"
        return "correct"

    p_stats = _probe_stats(
        truth.y_b_probe_noisy,
        truth.y_b_probe_clean,
        assim_res.gate_out.mix_probe_pred,
        assim_res.gate_out.mix_probe_var,
    )
    elapsed_wall_ms = (time.perf_counter_ns() - t0) / 1e6
    return {
        "blocks": 16,
        "block_errors": int(f0_err + f1_err),
        "fold0_blocks": 11,
        "fold1_blocks": 5,
        "fold0_errors": int(f0_err),
        "fold1_errors": int(f1_err),
        "coded_bits": 2048,
        "coded_errors": coded_err,
        "info_bits": 1024,
        "info_errors": int(info_err),
        "extra_blocks": len(free_rows),
        "extra_errors": int(ext_err),
        "probes": 32,
        **p_stats,
        "total_assimilations": int(tot_assim),
        "false_assimilations": int(false_assim),
        "wave1_fold0": _w1_status(0),
        "wave1_fold1": _w1_status(1),
        "pipeline_ms": elapsed_wall_ms,
    }


def _run_primary_cohort(
    *,
    clusters: Sequence[int],
    n_id: int,
    n_ood: int,
    id_base: int,
    ood_base: int,
    id_offset: int,
    ood_offset: int,
    out_dir: Path,
    run_full: bool,
    run_replay: bool,
    resume: bool = False,
    seed_pools: Optional[Mapping[str, Sequence[int]]] = None,
) -> dict[str, Any]:
    # Lazy imports of M2 ONLY when Primary Axis executes!
    from p11 import payload_rebuild, wavefront_pipeline

    warmup_numba()
    out_dir.mkdir(parents=True, exist_ok=True)

    def _eval_split(
        split: str,
        regimes: tuple[str, ...],
        n_links: int,
        base: int,
        offset: int,
        cluster_list: Sequence[int],
        dest_path: Path,
    ) -> list[dict[str, Any]]:
        jobs: list[tuple[str, int, str, int, int]] = [
            (split, int(c), reg, int(l_idx), int(base + offset + r_idx * 50_000 + 1000 * c + l_idx))
            for c in cluster_list
            for r_idx, reg in enumerate(regimes)
            for l_idx in range(n_links)
        ]
        existing = _load_resumed_jsonl(dest_path, set(jobs)) if resume else {}
        if existing:
            _write_jsonl(dest_path, [existing[j] for j in jobs if j in existing])
        else:
            _write_jsonl(dest_path, [])

        with dest_path.open("a", encoding="utf-8", newline="\n") as fh:
            for j in jobs:
                if j in existing:
                    continue
                _, c, reg, l_idx, seed = j
                obs_sp, truth_sp = simulate_link(seed, snr_db=PRIMARY_SNR_DB, regime=reg, budget="sparse")
                obs_dn, truth_dn = simulate_link(seed, snr_db=PRIMARY_SNR_DB, regime=reg, budget="dense")

                _, _, c_inh_sp, c_clr_sp = build_candidates_with_priors(obs_sp)
                sparse_arms = {
                    "p11_chained_wavefront": _score_primary_chained(wavefront_pipeline, obs_sp, truth_sp),
                    "p11_payload_only": _score_primary_standalone_rebuild(payload_rebuild, obs_sp, truth_sp),
                    "p9_always_inherit": _score_primary_gate_arm(obs_sp, truth_sp, gate_always_inherit(c_inh_sp, c_clr_sp)),
                    "p9_jeffreys_student_t": _score_primary_gate_arm(obs_sp, truth_sp, gate_jeffreys_student_t(obs_sp, c_inh_sp, c_clr_sp)),
                    "p9_always_clear": _score_primary_gate_arm(obs_sp, truth_sp, gate_always_clear(c_inh_sp, c_clr_sp)),
                }
                _, _, c_inh_dn, c_clr_dn = build_candidates_with_priors(obs_dn)
                dense_arms = {
                    "always_clear": _score_primary_gate_arm(obs_dn, truth_dn, gate_always_clear(c_inh_dn, c_clr_dn), run_extra=False),
                }
                rec = {
                    "split": split,
                    "cluster": int(c),
                    "regime": reg,
                    "link": int(l_idx),
                    "seed": seed,
                    "budgets": {"sparse": sparse_arms, "dense": dense_arms},
                }
                fh.write(json.dumps(rec, allow_nan=False) + "\n")
                fh.flush()
                existing[j] = rec

        ordered = [existing[j] for j in jobs]
        _write_jsonl(dest_path, ordered)
        return ordered

    def _build_primary_summary(recs: list[dict[str, Any]]) -> dict[str, Any]:
        arms_map = {
            "sparse": tuple(sorted(recs[0]["budgets"]["sparse"].keys())),
            "dense": tuple(sorted(recs[0]["budgets"]["dense"].keys())),
        }
        totals = _aggregate_nested_budget_arm(recs, ("sparse", "dense"), arms_map)
        c_ids = sorted({int(r["cluster"]) for r in recs})
        cluster_aggs = [
            _aggregate_nested_budget_arm([r for r in recs if r["cluster"] == c], ("sparse", "dense"), arms_map)
            for c in c_ids
        ]
        by_regime = {
            reg: _aggregate_nested_budget_arm([r for r in recs if r["regime"] == reg], ("sparse", "dense"), arms_map)
            for reg in sorted({r["regime"] for r in recs})
        }

        def _c_vec(bud: str, arm: str, met: str, reg_f: Optional[Sequence[str]] = None) -> np.ndarray:
            return np.asarray([
                aggregate_p11_cells([
                    r["budgets"][bud][arm]
                    for r in recs
                    if r["cluster"] == c and (reg_f is None or r["regime"] in reg_f)
                ])[met]
                for c in c_ids
            ], dtype=np.float64)

        tests = _audit_single_holm_family(
            {
                "chained_vs_p9_jt": {"baseline": _c_vec("sparse", "p9_jeffreys_student_t", "bler"), "candidate": _c_vec("sparse", "p11_chained_wavefront", "bler")},
                "chained_vs_payload_only": {"baseline": _c_vec("sparse", "p11_payload_only", "bler"), "candidate": _c_vec("sparse", "p11_chained_wavefront", "bler")},
            },
            "primary",
        )["contrasts"]
        tests[0].update({"baseline": ["sparse", "p9_jeffreys_student_t"], "candidate": ["sparse", "p11_chained_wavefront"]})
        tests[1].update({"baseline": ["sparse", "p11_payload_only"], "candidate": ["sparse", "p11_chained_wavefront"]})

        ni = audit_engineering_noninferiority(
            _c_vec("sparse", "p11_chained_wavefront", "bler", ["continuous"]),
            _c_vec("dense", "always_clear", "bler", ["continuous"]),
            _c_vec("sparse", "p11_chained_wavefront", "net_delivered_info_bits_per_frame", ["continuous"]),
            _c_vec("dense", "always_clear", "net_delivered_info_bits_per_frame", ["continuous"]),
        )
        return {
            "totals": totals,
            "clusters": cluster_aggs,
            "by_regime": by_regime,
            "tests": tests,
            "noninferiority": {
                **ni["bler_noninferiority"],
                "candidate": ["sparse", "p11_chained_wavefront"],
                "reference": ["dense", "always_clear"],
                "regimes": ["continuous"],
            },
            "noninferiority_full": ni,
            "common_phase_blindspot": summarize_common_phase_blindspot(recs),
        }

    id_path = out_dir / "p11_primary_id_links.jsonl"
    ood_path = out_dir / "p11_primary_ood_links.jsonl"
    sum_path = out_dir / "p11_primary_summary.json"
    rep_path = out_dir / "p11_primary_replay.jsonl"

    out_info: dict[str, Any] = {}
    if run_full:
        id_recs = _eval_split("id", REGIMES_ID, n_id, id_base, id_offset, clusters, id_path)
        ood_recs = _eval_split("ood", REGIMES_OOD + REGIMES_SANITY, n_ood, ood_base, ood_offset, clusters, ood_path)
        summary = _build_primary_summary(id_recs)
        _write_json(sum_path, summary)
        out_info["id_links"] = len(id_recs)
        out_info["ood_links"] = len(ood_recs)
        out_info["audit"] = audit_p11_primary(
            id_path, sum_path, seed_pools=seed_pools, require_canonical_family=True
        )

    if run_replay:
        rep_clusters = tuple(c for c in REPLAY_CLUSTERS if c in set(clusters))
        if not rep_clusters:
            rep_clusters = (clusters[0], clusters[-1])
        rep_recs = _eval_split("id", REGIMES_ID, n_id, id_base, id_offset, rep_clusters, rep_path)
        out_info["replay_links"] = len(rep_recs)
        if id_path.exists():
            out_info["replay_verify"] = verify_replay_exact(id_path, rep_path, float_tol=1e-12, strict=True)

    return out_info


# ---------------------------------------------------------------------------
# Secondary Axis Scoring (Off-Grid Continuous Manifold & 12 Checkpoints)
# ---------------------------------------------------------------------------

def _score_offgrid_gate(obs, truth, gate_res, *, run_strong: bool = True) -> dict[str, Any]:
    drows = list(truth.data_rows)
    n_blocks = len(drows)

    t0_nat = time.perf_counter_ns()
    llrs_d = gate_res.gate_out.mix_llrs[drows].reshape(n_blocks, 128)
    nat_wrong = p8_decode(llrs_d) != truth.b_info
    nat_block_err = int(nat_wrong.any(axis=1).sum())
    nat_info_err = int(nat_wrong.sum())
    coded_err = int(((llrs_d > 0.0) != truth.b_code).sum())
    nat_ms = (time.perf_counter_ns() - t0_nat) / 1e6

    t0_str = time.perf_counter_ns()
    if run_strong:
        stg_block_err = 0
        stg_info_err = 0
        for idx_r, r in enumerate(drows):
            sr = decode_strong_sc(
                obs.y_b[r],
                gate_res.h_mix[r],
                gate_res.noise_mix[r],
                list_size="large",
                bp_iters=20,
                turbo_pic=True,
                pic_scale=0.75,
                initial_llrs=gate_res.gate_out.mix_llrs[r].reshape(128),
            )
            diff = sr.info_bits != truth.b_info[idx_r]
            if bool(diff.any()):
                stg_block_err += 1
                stg_info_err += int(diff.sum())
    else:
        stg_block_err = nat_block_err
        stg_info_err = nat_info_err
    str_ms = (time.perf_counter_ns() - t0_str) / 1e6

    p_stats = _probe_stats(
        truth.y_b_probe_noisy,
        truth.y_b_probe_clean,
        gate_res.gate_out.mix_probe_pred,
        gate_res.gate_out.mix_probe_var,
    )
    return {
        "blocks": n_blocks,
        "block_errors": int(stg_block_err),
        "coded_bits": 128 * n_blocks,
        "coded_errors": coded_err,
        "info_bits": 64 * n_blocks,
        "info_errors": int(stg_info_err),
        "extra_blocks": 0,
        "extra_errors": 0,
        "probes": 32,
        **p_stats,
        "native_block_errors": int(nat_block_err),
        "native_info_errors": int(nat_info_err),
        "native_ms": nat_ms,
        "strong_ms": str_ms,
    }


def _eval_secondary_link_job(job: tuple[str, int, str, int, int]) -> dict[str, Any]:
    import torch
    from p11.manifold_gate import COMPLEX, REAL
    from p11.offgrid_channel import (
        OffGridGateResult,
        _compute_all_row_llrs,
        build_offgrid_branches,
        classical_always_clear,
        classical_always_inherit,
        classical_link_gate,
        classical_per_ray_gate,
        compose_offgrid_gate_output,
        simulate_offgrid_link,
    )

    split, cluster_idx, regime, link_idx, seed = job
    obs, truth = simulate_offgrid_link(seed, regime, snr_db=PRIMARY_SNR_DB)
    bundle = build_offgrid_branches(obs)
    models = _get_offgrid_models()

    arms: dict[str, dict[str, Any]] = {}
    for cname, fn in (
        ("classical_always_inherit", classical_always_inherit),
        ("classical_always_clear", classical_always_clear),
        ("classical_link_gate", classical_link_gate),
        ("classical_per_ray_gate", classical_per_ray_gate),
    ):
        t0 = time.perf_counter_ns()
        res = fn(obs, bundle)
        g_ms = (time.perf_counter_ns() - t0) / 1e6
        cell = _score_offgrid_gate(obs, truth, res)
        cell["gate_ms"] = g_ms
        arms[cname] = cell

    for var in OFFGRID_VARIANTS:
        t0 = time.perf_counter_ns()
        seed_models = models[var]
        feats = seed_models[0][1].extract_features_for_variant(obs, bundle)
        s_t = torch.from_numpy(feats.scalar_feats).to(REAL).unsqueeze(0)
        c_t = torch.from_numpy(feats.complex_feats).to(COMPLEX).unsqueeze(0)
        tau_t = torch.from_numpy(feats.tau_hat).to(REAL).unsqueeze(0)
        nu_t = torch.from_numpy(feats.nu_hat).to(REAL).unsqueeze(0)
        tp_t = torch.from_numpy(feats.t_phys).to(REAL).unsqueeze(0)
        tn_t = torch.from_numpy(feats.t_norm).to(REAL).unsqueeze(0)
        ev_t = torch.from_numpy(feats.ev_base).to(REAL).unsqueeze(0)
        bl_t = torch.from_numpy(feats.base_logit).to(REAL).unsqueeze(0)
        pr_t = torch.from_numpy(feats.is_probe).to(REAL).unsqueeze(0)

        logits_list = []
        nm_list = []
        per_seed_res: list[tuple[int, OffGridGateResult]] = []
        with torch.no_grad():
            for m_seed, m in seed_models:
                w_t, logit_t, nm_t = m.forward_tensors(s_t, c_t, tau_t, nu_t, tp_t, tn_t, ev_t, bl_t, pr_t)
                logits_list.append(logit_t.squeeze(0).cpu().numpy())
                nm_np = nm_t.squeeze(0).cpu().numpy()
                nm_list.append(nm_np)
                per_seed_res.append((m_seed, compose_offgrid_gate_output(w_t.squeeze(0).cpu().numpy(), nm_np, obs, bundle)))

        mean_logit = np.mean(logits_list, axis=0)
        w_ens = 1.0 / (1.0 + np.exp(-np.clip(mean_logit, -40.0, 40.0)))
        nm_ens = np.exp(np.mean(np.log(np.maximum(nm_list, 1e-12)), axis=0))
        ens_res = compose_offgrid_gate_output(w_ens, nm_ens, obs, bundle)
        g_ms = (time.perf_counter_ns() - t0) / 1e6

        ens_cell = _score_offgrid_gate(obs, truth, ens_res)
        ens_cell["gate_ms"] = g_ms
        arms[var] = ens_cell

        for m_seed, s_res in per_seed_res:
            s_cell = _score_offgrid_gate(obs, truth, s_res)
            s_cell["gate_ms"] = g_ms / len(OFFGRID_SEEDS)
            arms[f"{var}_s{m_seed}"] = s_cell

    # Oracle true channel reference
    y_b_safe = np.array(obs.y_b, dtype=np.complex128, copy=True)
    y_b_safe[obs.probe_row] = 0.0 + 0.0j
    o_d_mats = np.stack([np.eye(N, dtype=np.complex128) for _ in range(1)], axis=0)
    # Build exact oracle GateOutput directly from truth.h_b_true_mat
    from p8c.link import PILOTS as SC_PILOTS
    from p8.coding import qam_llrs

    eye_n = np.eye(N, dtype=np.complex128)
    drows = list(truth.data_rows)
    h_d = truth.h_b_true_mat[drows] / math.sqrt(obs.n0)
    y_d = y_b_safe[drows] / math.sqrt(obs.n0)
    adj = h_d.conj().swapaxes(-1, -2)
    cov = np.linalg.inv(eye_n[None, :, :] + adj @ h_d)
    est = (cov @ (adj @ y_d[:, :, None]))[..., 0]
    u_var = np.diagonal(cov, axis1=-2, axis2=-1).real
    o_llrs = np.zeros((T, N, 4), dtype=np.float64)
    o_llrs[drows] = qam_llrs(est.ravel(), u_var.ravel()).reshape(len(drows), N, 4)
    o_probe_pred = truth.h_b_true_mat[obs.probe_row] @ SC_PILOTS[obs.probe_row]
    o_probe_var = np.full(N, float(obs.n0), dtype=np.float64)
    o_gate = GateOutput(
        history_weight=np.ones(T, dtype=np.float64),
        probe_weight=1.0,
        mix_llrs=o_llrs,
        mix_probe_pred=o_probe_pred,
        mix_probe_var=o_probe_var,
        mix_mean_ray=np.zeros((1, T, N), dtype=np.complex128),
        mix_var_ray=np.zeros((1, T, N), dtype=np.float64),
    )
    o_res = OffGridGateResult(
        gate_out=o_gate,
        w_ray=np.ones((1, T), dtype=np.float64),
        noise_mult=np.ones(T, dtype=np.float64),
        h_mix=truth.h_b_true_mat,
        noise_mix=np.full((T, N), float(obs.n0), dtype=np.float64),
    )
    arms["oracle_true_channel"] = _score_offgrid_gate(obs, truth, o_res)

    return {
        "split": split,
        "cluster": int(cluster_idx),
        "regime": regime,
        "link": int(link_idx),
        "seed": int(seed),
        "n_rays": int(truth.n_rays),
        "budgets": {"sparse": arms},
    }


def _run_secondary_cohort(
    *,
    clusters: Sequence[int],
    n_id: int,
    n_ood: int,
    id_base: int,
    ood_base: int,
    id_offset: int,
    ood_offset: int,
    out_dir: Path,
    eval_workers: int,
    require_env: bool,
    run_full: bool,
    run_replay: bool,
    resume: bool = False,
    seed_pools: Optional[Mapping[str, Sequence[int]]] = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    id_path = out_dir / "p11_secondary_id_links.jsonl"
    ood_path = out_dir / "p11_secondary_ood_links.jsonl"
    all_path = out_dir / "p11_secondary_links.jsonl"
    sum_path = out_dir / "p11_secondary_summary.json"
    rep_path = out_dir / "p11_secondary_replay.jsonl"

    def _build_jobs(split: str, regimes: tuple[str, ...], n_links: int, base: int, offset: int, c_list: Sequence[int]) -> list[tuple[str, int, str, int, int]]:
        jobs = []
        for c in c_list:
            for r_idx, reg in enumerate(regimes):
                for l_idx in range(n_links):
                    seed = int(base + offset + r_idx * 50_000 + 1000 * c + l_idx)
                    jobs.append((split, int(c), reg, int(l_idx), seed))
        return jobs

    def _run_jobs(
        jobs: list[tuple[str, int, str, int, int]],
        workers: int,
        dest_path: Path,
    ) -> list[dict[str, Any]]:
        existing = _load_resumed_jsonl(dest_path, set(jobs)) if resume else {}
        if existing:
            _write_jsonl(dest_path, [existing[j] for j in jobs if j in existing])
        else:
            _write_jsonl(dest_path, [])

        remaining = [j for j in jobs if j not in existing]
        if remaining:
            with dest_path.open("a", encoding="utf-8", newline="\n") as fh:
                if workers > 1:
                    with ProcessPoolExecutor(
                        max_workers=workers,
                        initializer=_init_secondary_worker,
                        initargs=(require_env,),
                    ) as pool:
                        for rec in pool.map(_eval_secondary_link_job, remaining, chunksize=1):
                            fh.write(json.dumps(rec, allow_nan=False) + "\n")
                            fh.flush()
                            k = (rec["split"], int(rec["cluster"]), rec["regime"], int(rec["link"]), int(rec["seed"]))
                            existing[k] = rec
                else:
                    _init_secondary_worker(require_env)
                    for j in remaining:
                        rec = _eval_secondary_link_job(j)
                        fh.write(json.dumps(rec, allow_nan=False) + "\n")
                        fh.flush()
                        existing[j] = rec

        ordered = [existing[j] for j in jobs]
        _write_jsonl(dest_path, ordered)
        return ordered

    def _build_secondary_summary(all_recs: list[dict[str, Any]]) -> dict[str, Any]:
        arms_map = {"sparse": tuple(sorted(all_recs[0]["budgets"]["sparse"].keys()))}
        totals = _aggregate_nested_budget_arm(all_recs, ("sparse",), arms_map)
        c_ids = sorted({int(r["cluster"]) for r in all_recs})
        cluster_aggs = [
            _aggregate_nested_budget_arm([r for r in all_recs if r["cluster"] == c], ("sparse",), arms_map)
            for c in c_ids
        ]
        by_regime = {
            reg: _aggregate_nested_budget_arm([r for r in all_recs if r["regime"] == reg], ("sparse",), arms_map)
            for reg in sorted({r["regime"] for r in all_recs})
        }
        id_sub = [r for r in all_recs if r["regime"] in OFFGRID_REGIMES_ID]
        ood_sub = [r for r in all_recs if r["regime"] in OFFGRID_REGIMES_OOD]
        id_equal = _aggregate_nested_budget_arm(id_sub, ("sparse",), arms_map)
        ood_equal = _aggregate_nested_budget_arm(ood_sub, ("sparse",), arms_map)

        def _c_vec(arm: str, regimes: Sequence[str], metric: str = "bler") -> np.ndarray:
            reg_set = set(regimes)
            return np.asarray([
                aggregate_p11_cells([
                    r["budgets"]["sparse"][arm]
                    for r in all_recs
                    if r["cluster"] == c and r["regime"] in reg_set
                ])[metric]
                for c in c_ids
            ], dtype=np.float64)

        tests = _audit_single_holm_family(
            {
                "transport_attn_vs_abs_no_transport": {
                    "baseline": _c_vec("em10_abs_no_transport", OFFGRID_REGIMES_ID),
                    "candidate": _c_vec("em10_transport_attn", OFFGRID_REGIMES_ID),
                    "regimes": list(OFFGRID_REGIMES_ID),
                },
                "transport_attn_vs_local": {
                    "baseline": _c_vec("em10_local", OFFGRID_REGIMES_ID),
                    "candidate": _c_vec("em10_transport_attn", OFFGRID_REGIMES_ID),
                    "regimes": list(OFFGRID_REGIMES_ID),
                },
                "transport_attn_vs_no_history": {
                    "baseline": _c_vec("em10_no_history", OFFGRID_REGIMES_ID),
                    "candidate": _c_vec("em10_transport_attn", OFFGRID_REGIMES_ID),
                    "regimes": list(OFFGRID_REGIMES_ID),
                },
            },
            "offgrid",
        )["contrasts"]
        tests[0].update({"baseline": ["sparse", "em10_abs_no_transport"], "candidate": ["sparse", "em10_transport_attn"]})
        tests[1].update({"baseline": ["sparse", "em10_local"], "candidate": ["sparse", "em10_transport_attn"]})
        tests[2].update({"baseline": ["sparse", "em10_no_history"], "candidate": ["sparse", "em10_transport_attn"]})

        ood_tests = _audit_single_holm_family(
            {
                "ood_transport_attn_vs_abs_no_transport": {
                    "baseline": _c_vec("em10_abs_no_transport", OFFGRID_REGIMES_OOD),
                    "candidate": _c_vec("em10_transport_attn", OFFGRID_REGIMES_OOD),
                    "regimes": list(OFFGRID_REGIMES_OOD),
                },
                "ood_transport_attn_vs_local": {
                    "baseline": _c_vec("em10_local", OFFGRID_REGIMES_OOD),
                    "candidate": _c_vec("em10_transport_attn", OFFGRID_REGIMES_OOD),
                    "regimes": list(OFFGRID_REGIMES_OOD),
                },
                "ood_transport_attn_vs_no_history": {
                    "baseline": _c_vec("em10_no_history", OFFGRID_REGIMES_OOD),
                    "candidate": _c_vec("em10_transport_attn", OFFGRID_REGIMES_OOD),
                    "regimes": list(OFFGRID_REGIMES_OOD),
                },
            },
            "offgrid_ood",
        )["contrasts"]
        ood_tests[0].update({"baseline": ["sparse", "em10_abs_no_transport"], "candidate": ["sparse", "em10_transport_attn"]})
        ood_tests[1].update({"baseline": ["sparse", "em10_local"], "candidate": ["sparse", "em10_transport_attn"]})
        ood_tests[2].update({"baseline": ["sparse", "em10_no_history"], "candidate": ["sparse", "em10_transport_attn"]})

        return {
            "totals": totals,
            "id_equal": id_equal,
            "ood_equal": ood_equal,
            "clusters": cluster_aggs,
            "by_regime": by_regime,
            "tests": tests,
            "ood_tests": ood_tests,
        }

    out_info: dict[str, Any] = {}
    if run_full:
        id_jobs = _build_jobs("id", OFFGRID_REGIMES_ID, n_id, id_base, id_offset, clusters)
        ood_jobs = _build_jobs("ood", OFFGRID_REGIMES_OOD, n_ood, ood_base, ood_offset, clusters)
        id_recs = _run_jobs(id_jobs, eval_workers, id_path)
        ood_recs = _run_jobs(ood_jobs, eval_workers, ood_path)
        all_recs = id_recs + ood_recs
        _write_jsonl(all_path, all_recs)
        summary = _build_secondary_summary(all_recs)
        _write_json(sum_path, summary)
        out_info["id_links"] = len(id_recs)
        out_info["ood_links"] = len(ood_recs)
        out_info["audit"] = audit_p11_offgrid(
            all_path, sum_path, seed_pools=seed_pools, require_canonical_family=True
        )

    if run_replay:
        rep_clusters = tuple(c for c in REPLAY_CLUSTERS if c in set(clusters))
        if not rep_clusters:
            rep_clusters = (clusters[0], clusters[-1])
        rep_jobs = (
            _build_jobs("id", OFFGRID_REGIMES_ID, n_id, id_base, id_offset, rep_clusters)
            + _build_jobs("ood", OFFGRID_REGIMES_OOD, n_ood, ood_base, ood_offset, rep_clusters)
        )
        # Mandatory single-worker replay!
        rep_recs = _run_jobs(rep_jobs, 1, rep_path)
        out_info["replay_links"] = len(rep_recs)
        if all_path.exists():
            out_info["replay_verify"] = verify_replay_exact(all_path, rep_path, float_tol=1e-12, strict=True)

    return out_info


def _merge_audit_bundle(
    audit_path: Path,
    audit_bundle: dict[str, Any],
    *,
    mode: str,
    is_dry: bool,
) -> dict[str, Any]:
    merged = dict(audit_bundle)
    if audit_path.is_file():
        try:
            prev_bundle = json.loads(audit_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev_bundle = None
        if isinstance(prev_bundle, dict) and ((prev_bundle.get("mode") == "dry_run") == is_dry):
            for ax_key in ("primary", "secondary"):
                prev_ax = prev_bundle.get(ax_key)
                if isinstance(prev_ax, dict):
                    if ax_key not in merged:
                        merged[ax_key] = prev_ax
                    elif mode == "replay" and isinstance(merged.get(ax_key), dict):
                        merged[ax_key] = {**prev_ax, **merged[ax_key]}
            if "primary" in merged and "secondary" in merged:
                merged["axis"] = "both"
            if mode == "replay" and prev_bundle.get("mode") in ("primary", "secondary", "all"):
                merged["mode"] = prev_bundle["mode"]
    return merged


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 11 formal closed-book evaluation runner.")
    parser.add_argument(
        "--mode",
        choices=("dry_run", "primary", "secondary", "replay", "all"),
        default="dry_run",
        help="Execution mode (default: dry_run on SEED_P11_DEV).",
    )
    parser.add_argument(
        "--axis",
        choices=("primary", "secondary", "both"),
        default="both",
        help="Which experimental axis to run in dry_run or replay mode.",
    )
    parser.add_argument(
        "--confirm-frozen",
        action="store_true",
        help="Mandatory acknowledgment required to execute any formal/replay run.",
    )
    parser.add_argument(
        "--allow-dirty-agents",
        action="store_true",
        help="Pass through to freeze verification.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to freeze manifest JSON.",
    )
    parser.add_argument(
        "--eval-workers",
        type=int,
        default=1,
        help="Worker count for Secondary Axis bulk scoring (Primary & Replay always enforce 1).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to p11/results or p11/results/dry_run for dry_run).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing valid JSONL records in --out-dir.",
    )
    args = parser.parse_args(argv)

    if args.eval_workers < 1:
        raise ValueError(f"--eval-workers must be >= 1, got {args.eval_workers}")

    is_dry = args.mode == "dry_run"
    if not is_dry:
        if not args.confirm_frozen:
            raise RuntimeError(
                "Formal/replay evaluation requires --confirm-frozen after pre-experiment freeze commit."
            )
        det_info = enforce_determinism(require_env=True)
        freeze_sha = verify_freeze(manifest_path=args.manifest, allow_dirty_agents=args.allow_dirty_agents)
        clusters = FORMAL_CLUSTERS
        pri_n_id, pri_n_ood = LINKS_PER_CLUSTER_ID, LINKS_PER_CLUSTER_OOD
        sec_n_id, sec_n_ood = OFFGRID_LINKS_PER_CLUSTER_ID, OFFGRID_LINKS_PER_CLUSTER_OOD
        id_base, ood_base = SEED_P11_FORMAL_ID, SEED_P11_FORMAL_OOD
        pri_id_off, pri_ood_off = 200_000, 0
        sec_id_off, sec_ood_off = 500_000, 500_000
        out_dir = args.out_dir if args.out_dir is not None else RESULTS_DIR
    else:
        det_info = enforce_determinism(require_env=False)
        freeze_sha = None
        clusters = (1, 2, 3, 4)
        pri_n_id, pri_n_ood = 1, 1
        sec_n_id, sec_n_ood = 1, 1
        id_base, ood_base = SEED_P11_DEV, SEED_P11_DEV
        pri_id_off, pri_ood_off = 200_000, 400_000
        sec_id_off, sec_ood_off = 600_000, 800_000
        out_dir = args.out_dir if args.out_dir is not None else (RESULTS_DIR / "dry_run")

    # Verify strict seed disjointness across Primary, Secondary, and all historical ranges BEFORE running
    p_id_seeds, p_ood_seeds = _planned_primary_seeds(
        clusters, pri_n_id, pri_n_ood, id_base, ood_base, pri_id_off, pri_ood_off
    )
    s_id_seeds, s_ood_seeds = _planned_secondary_seeds(
        clusters, sec_n_id, sec_n_ood, id_base, ood_base, sec_id_off, sec_ood_off
    )
    if is_dry:
        seed_report = check_p11_seed_disjointness(
            p11_train_seeds=[SEED_P11_TRAIN],
            p11_dev_seeds=p_id_seeds + p_ood_seeds + s_id_seeds + s_ood_seeds,
            p11_formal_id_seeds=[SEED_P11_FORMAL_ID],
            p11_formal_ood_seeds=[SEED_P11_FORMAL_OOD],
            p11_e2e_seeds=[SEED_P11_E2E],
            strict=True,
        )
    else:
        seed_report = check_p11_seed_disjointness(
            p11_train_seeds=[SEED_P11_TRAIN],
            p11_dev_seeds=[SEED_P11_DEV],
            p11_formal_id_seeds=p_id_seeds + s_id_seeds,
            p11_formal_ood_seeds=p_ood_seeds + s_ood_seeds,
            p11_e2e_seeds=[SEED_P11_E2E],
            strict=True,
        )

    cohort_seed_pools: Optional[dict[str, list[int]]] = (
        None
        if is_dry
        else {
            "train": [SEED_P11_TRAIN],
            "dev": [SEED_P11_DEV],
            "formal_id": p_id_seeds + s_id_seeds,
            "formal_ood": p_ood_seeds + s_ood_seeds,
            "e2e": [SEED_P11_E2E],
        }
    )

    run_pri = (args.mode in ("primary", "all")) or (args.mode in ("dry_run", "replay") and args.axis in ("primary", "both"))
    run_sec = (args.mode in ("secondary", "all")) or (args.mode in ("dry_run", "replay") and args.axis in ("secondary", "both"))
    run_full = args.mode != "replay"
    run_rep = args.mode in ("dry_run", "replay", "all")

    audit_bundle: dict[str, Any] = {
        "mode": args.mode,
        "axis": args.axis,
        "freeze_sha256": freeze_sha,
        "determinism": det_info,
        "seed_disjointness": seed_report,
    }

    if run_pri:
        audit_bundle["primary"] = _run_primary_cohort(
            clusters=clusters,
            n_id=pri_n_id,
            n_ood=pri_n_ood,
            id_base=id_base,
            ood_base=ood_base,
            id_offset=pri_id_off,
            ood_offset=pri_ood_off,
            out_dir=out_dir,
            run_full=run_full,
            run_replay=run_rep,
            resume=bool(args.resume),
            seed_pools=cohort_seed_pools,
        )

    if run_sec:
        audit_bundle["secondary"] = _run_secondary_cohort(
            clusters=clusters,
            n_id=sec_n_id,
            n_ood=sec_n_ood,
            id_base=id_base,
            ood_base=ood_base,
            id_offset=sec_id_off,
            ood_offset=sec_ood_off,
            out_dir=out_dir,
            eval_workers=int(args.eval_workers),
            require_env=not is_dry,
            run_full=run_full,
            run_replay=run_rep,
            resume=bool(args.resume),
            seed_pools=cohort_seed_pools,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    audit_path = out_dir / "p11_audit.json"
    audit_bundle = _merge_audit_bundle(audit_path, audit_bundle, mode=args.mode, is_dry=is_dry)

    _write_json(audit_path, audit_bundle)
    print(json.dumps({"status": "OK", "mode": args.mode, "out_dir": str(out_dir), "audit_json": str(audit_path)}, indent=2))


__all__ = ["main"]


if __name__ == "__main__":
    main()
