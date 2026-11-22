"""Consolidated fresh-process pinned unified benchmark across all 13 arms (M5 / F11).

Runs each receiver arm in an isolated single-threaded subprocess (`--worker-arm`)
while keeping the parent orchestrator free of heavy scientific imports and
accumulated output tensors during the worker-spawning phase. Because Linux
`fork()` + `execve()` latches the parent's `mm->hiwater_rss` into the child's
`signal->maxrss` (`ru_maxrss`), keeping the parent at ~18 MB RSS during worker
spawning guarantees every worker's `peak_rss_mb` is a true, uncontaminated
fresh-process inference high-water mark.

Parent orchestrator mode collects per-arm payloads AFTER all 13 workers exit,
runs paired statistical comparisons (`G=3` macro-blocks, `G=12`, and `G=16`
sub-blocks with autocorrelation caveat), computes SINR-stratified breakdowns,
verifies 16-frame deterministic replay across all 13 arms (`max_float_diff == 0.0`),
and emits consolidated top-level artifacts directly under
`research/realdata_sdr_ofdm/results/`.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import csv
import gc
import hashlib
import json
import pickle
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if TYPE_CHECKING:
    from research.realdata_sdr_ofdm.adapters.sdr_ofdm_adapter import RealOFDMFrame
    from research.realdata_sdr_ofdm.receivers.classical import (
        ClassicalConfig,
        ReceiverOutput,
    )
    from research.realdata_sdr_ofdm.receivers.group_a_reference import GroupAConfig
    from research.realdata_sdr_ofdm.receivers.group_b_fusion import GroupBConfig

ALL_ARMS: tuple[str, ...] = (
    "ls_linear_zf",
    "lmmse_1d",
    "lmmse_2d_tracked",
    "local",
    "global_uniform",
    "global_reliability",
    "global_content_attn",
    "hard_map",
    "posterior_mean_plugin",
    "exact_mixture_likelihood",
    "group_c_branch_attn",
    "group_c_hybrid_attn",
    "group_c_pilot_nll_only",
)

ARM_GROUP: dict[str, str] = {
    "ls_linear_zf": "classical",
    "lmmse_1d": "classical",
    "lmmse_2d_tracked": "classical",
    "local": "group_a",
    "global_uniform": "group_a",
    "global_reliability": "group_a",
    "global_content_attn": "group_a",
    "hard_map": "group_b",
    "posterior_mean_plugin": "group_b",
    "exact_mixture_likelihood": "group_b",
    "group_c_branch_attn": "group_c",
    "group_c_hybrid_attn": "group_c",
    "group_c_pilot_nll_only": "group_c",
}

SINR_BINS: tuple[tuple[str, float, float, bool], ...] = (
    ("[5,10)dB", 5.0, 10.0, False),
    ("[10,15)dB", 10.0, 15.0, False),
    ("[15,20)dB", 15.0, 20.0, False),
    ("[20,31]dB", 20.0, 31.0, True),
)

SUMMARY_CSV_FIELDS: tuple[str, ...] = (
    "arm",
    "group",
    "condition",
    "extra_awgn_db",
    "cfo_norm",
    "phase_drift_rad",
    "total_frames",
    "success_count",
    "fail_count",
    "coverage_pct",
    "bit_errors",
    "total_bits",
    "ber",
    "frame_errors",
    "fer",
    "sym_errors",
    "ser",
    "sym_mse",
    "evm_pct",
    "pilot_nmse",
    "pilot_cal_ratio",
    "pilot_nll",
    "mean_ms_per_frame",
    "median_ms_per_frame",
    "p95_ms_per_frame",
    "peak_rss_mb",
    "tracemalloc_peak_mb",
    "n_candidates",
    "param_count",
    "train_time_s",
)

PER_FRAME_SCALAR_FIELDS: tuple[str, ...] = (
    "arm",
    "group",
    "condition",
    "frame_id",
    "split",
    "group_id",
    "reported_sinr_db",
    "sinr_bin",
    "est_n0",
    "bit_errors",
    "total_bits",
    "ber",
    "frame_error",
    "sym_errors",
    "total_syms",
    "sym_mse",
    "evm_pct",
    "pilot_nmse",
    "pilot_cal_ratio",
    "pilot_nll",
    "n_candidates",
    "param_count",
    "latency_ms",
    "failed",
    "error",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _sinr_bin_label(sinr_db: float) -> str:
    for label, lo, hi, inc_hi in SINR_BINS:
        if inc_hi:
            if lo <= sinr_db <= hi:
                return label
        else:
            if lo <= sinr_db < hi:
                return label
    return "other"


def _load_classical_config(results_root: Path, arm: str) -> ClassicalConfig:
    from research.realdata_sdr_ofdm.receivers.classical import ClassicalConfig

    defaults = {
        "ls_linear_zf": ClassicalConfig(tau_rms=1.5, n_taps=5, n0_scale=0.7, detrend_timing=False, phase_track=True),
        "lmmse_1d": ClassicalConfig(tau_rms=1.0, n_taps=11, n0_scale=0.7, detrend_timing=True, phase_track=True),
        "lmmse_2d_tracked": ClassicalConfig(
            tau_rms=1.0, n_taps=11, n0_scale=0.7, detrend_timing=True, fd_norm=0.0, phase_track=True
        ),
    }
    dt_path = results_root / "baselines" / "dev_tuning.json"
    if dt_path.exists():
        data = json.loads(dt_path.read_text(encoding="utf-8"))
        cfg_dict = data.get("tuned", {}).get(arm, {}).get("config")
        if isinstance(cfg_dict, dict):
            return ClassicalConfig(**cfg_dict)
    return defaults[arm]


def _load_group_a_config(results_root: Path, arm: str) -> GroupAConfig:
    from research.realdata_sdr_ofdm.receivers.group_a_reference import GroupAConfig

    default = GroupAConfig(
        temperature=0.05,
        local_k=4,
        max_pool_size=64,
        prior_scale=0.5,
        n_taps=11,
        tau_rms=1.0,
        n0_scale=0.7,
        diag_reg=1e-4,
        detrend_timing=True,
        phase_track=False,
    )
    dt_path = results_root / "group_a" / "dev_tuning.json"
    if dt_path.exists():
        data = json.loads(dt_path.read_text(encoding="utf-8"))
        cfg_dict = data.get("tuned", {}).get(arm, {}).get("config")
        if isinstance(cfg_dict, dict):
            return GroupAConfig(**cfg_dict)
    return default


def _load_group_b_config(results_root: Path, arm: str) -> GroupBConfig:
    from research.realdata_sdr_ofdm.receivers.group_b_fusion import GroupBConfig

    default = GroupBConfig(
        pdp_hypotheses=((7, 0.8), (11, 1.0), (11, 2.0)),
        n0_scale=0.7,
        diag_reg=1e-4,
        detrend_timing=True,
        phase_track_max_slope=0.50,
        phase_track_prior_weight=2.0,
        beta=0.25,
        nonzero_om_penalty=8.0,
        omega_min_diff=1e-3,
        include_constellation_var=False,
    )
    dt_path = results_root / "group_b" / "dev_tuning.json"
    if dt_path.exists():
        data = json.loads(dt_path.read_text(encoding="utf-8"))
        cfg_dict = data.get("tuned", {}).get(arm, {}).get("config")
        if isinstance(cfg_dict, dict):
            raw = dict(cfg_dict)
            if "pdp_hypotheses" in raw:
                raw["pdp_hypotheses"] = tuple((int(nt), float(tr)) for nt, tr in raw["pdp_hypotheses"])
            return GroupBConfig(**raw)
    return default


def _resolve_canonical_checkpoint(
    arm: str,
    manifest_path: Path,
    ckpt_dir: Path,
) -> tuple[dict[str, Any], Path]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Group C manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [e for e in manifest.get("checkpoints", []) if e.get("arm") == arm]
    if not entries:
        raise KeyError(f"No checkpoint entries found for arm {arm!r} in {manifest_path}")
    best = min(
        entries,
        key=lambda e: (float(e["dev_ber"]), float(e["dev_pilot_nll"]), int(e["seed"])),
    )
    ckpt_path = ckpt_dir / str(best["file"])
    if not ckpt_path.exists():
        fallback = manifest_path.parent / str(best["file"])
        if fallback.exists():
            ckpt_path = fallback
        else:
            raise FileNotFoundError(f"Missing canonical checkpoint for {arm}: {ckpt_path}")
    actual_sha = _sha256_file(ckpt_path)
    expected_sha = str(best["sha256"])
    if actual_sha != expected_sha:
        raise ValueError(
            f"SHA-256 mismatch for {ckpt_path}: actual={actual_sha} != manifest={expected_sha}"
        )
    return best, ckpt_path


def _build_per_frame_record(
    arm: str,
    group: str,
    fr: RealOFDMFrame,
    out: ReceiverOutput,
    *,
    param_count: int,
) -> dict[str, Any]:
    import numpy as np
    from research.realdata_sdr_ofdm.adapters.sdr_ofdm_adapter import (
        PILOT_FFT_INDICES,
        PILOT_SYMBOL_IDX,
    )

    be = int(np.sum(out.hard_bits != fr.tx_bits))
    se = int(np.sum(np.any(out.hard_bits != fr.tx_bits, axis=-1)))
    sq_err = float(np.sum(np.abs(out.eq_symbols - fr.tx_symbols) ** 2))
    sq_sig = float(np.sum(np.abs(fr.tx_symbols) ** 2))
    sym_mse = sq_err / float(fr.tx_symbols.size)
    evm_pct = 100.0 * float(np.sqrt(sq_err / max(sq_sig, 1e-30)))

    p_nmse = 0.0
    p_cal = 0.0
    p_nll = 0.0
    if out.heldout_pilot_pred is not None and out.heldout_pilot_var is not None:
        idx = out.aux_info.get("heldout_pilot_indices") or []
        if len(idx) == len(out.heldout_pilot_pred) and len(idx) > 0:
            fft_idx = [PILOT_FFT_INDICES[i] for i in idx]
            y_p = fr.rx_grid[PILOT_SYMBOL_IDX, fft_idx]
            p_val = fr.pilot_values[idx]
            z_obs = y_p / p_val
            sig2_obs = float(fr.est_n0) / (np.abs(p_val) ** 2)
            v_eff = np.maximum(out.heldout_pilot_var + sig2_obs, 1e-30)
            err = out.heldout_pilot_pred - z_obs
            p_nmse = float(np.sum(np.abs(err) ** 2) / max(np.sum(np.abs(z_obs) ** 2), 1e-30))
            p_cal = float(np.mean((np.abs(err) ** 2) / v_eff))
            p_nll = float(np.mean(np.log(np.pi * v_eff) + (np.abs(err) ** 2) / v_eff))

    aux = out.aux_info or {}
    rec: dict[str, Any] = {
        "arm": arm,
        "group": group,
        "condition": "clean",
        "frame_id": int(fr.frame_id),
        "split": str(fr.split),
        "group_id": str(fr.group_id),
        "reported_sinr_db": float(fr.reported_sinr_db),
        "sinr_bin": _sinr_bin_label(float(fr.reported_sinr_db)),
        "est_n0": float(fr.est_n0),
        "bit_errors": be,
        "total_bits": int(fr.tx_bits.size),
        "ber": float(be / fr.tx_bits.size),
        "frame_error": int(be > 0),
        "sym_errors": se,
        "total_syms": int(fr.tx_symbols.size),
        "sym_mse": float(sym_mse),
        "evm_pct": float(evm_pct),
        "pilot_nmse": p_nmse,
        "pilot_cal_ratio": p_cal,
        "pilot_nll": p_nll,
        "n_candidates": int(aux.get("n_candidates", 1)),
        "param_count": int(aux.get("param_count", param_count)),
        "latency_ms": float(aux.get("latency_ms", 0.0)),
        "failed": bool(aux.get("failed", False)),
        "error": aux.get("error"),
    }
    if group == "classical":
        rec["omega_t_rad_per_sym"] = float(aux.get("omega_t_rad_per_sym", 0.0))
    elif group == "group_a":
        w_arr = np.asarray(aux.get("weights", []), dtype=np.float64)
        w_pos = w_arr[w_arr > 0.0]
        rec["n_values_accessed"] = int(aux.get("n_values_accessed", 0))
        rec["weight_entropy"] = float(-np.sum(w_pos * np.log(w_pos))) if w_pos.size > 0 else 0.0
        rec["weight_max"] = float(np.max(w_arr)) if w_arr.size > 0 else 0.0
    elif group == "group_b":
        rec["branch_entropy"] = float(aux.get("branch_entropy", 0.0))
        rec["between_var_mean"] = float(aux.get("between_var_mean", 0.0))
        rec["omega_t_candidates"] = list(aux.get("omega_t_candidates", [0.0]))
    elif group == "group_c":
        rec["branch_entropy"] = float(aux.get("branch_entropy", 0.0))
        rec["between_var_mean"] = float(aux.get("between_var_mean", 0.0))
        rec["branch_weights"] = [float(x) for x in aux.get("branch_weights", [])]
        rec["prior_weights"] = [float(x) for x in aux.get("prior_weights", [])]
        rec["omega_t_candidates"] = list(aux.get("omega_t_candidates", [0.0]))
    return rec


def _run_worker_arm(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    try:
        torch.set_num_threads(1)
    except RuntimeError:
        pass

    from research.realdata_sdr_ofdm.adapters.sdr_ofdm_adapter import load_processed_split
    from research.realdata_sdr_ofdm.eval.evaluator import (
        evaluate_receiver,
        profile_receiver_pinned,
        run_stress_suite,
    )
    from research.realdata_sdr_ofdm.receivers.classical import run_frame as run_classical_frame
    from research.realdata_sdr_ofdm.receivers.group_a_reference import build_reference_pool
    from research.realdata_sdr_ofdm.receivers.group_a_reference import run_frame as run_group_a_frame
    from research.realdata_sdr_ofdm.receivers.group_b_fusion import run_frame as run_group_b_frame
    from research.realdata_sdr_ofdm.receivers.group_c_trainable import GroupCTrainableReceiver

    arm = str(args.worker_arm)
    if arm not in ARM_GROUP:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ALL_ARMS}")
    group = ARM_GROUP[arm]

    test_frames = load_processed_split(args.proc_dir, "test")

    param_count = 0
    train_time_s = 0.0
    extra_meta: dict[str, Any] = {"group": group}
    fn: Callable[[RealOFDMFrame], ReceiverOutput]

    if group == "classical":
        cfg = _load_classical_config(args.results_root, arm)
        extra_meta["config"] = cfg.to_dict()
        fn = lambda fr, _m=arm, _c=cfg: run_classical_frame(fr, method=_m, config=_c)
    elif group == "group_a":
        ga_cfg = _load_group_a_config(args.results_root, arm)
        train_frames = load_processed_split(args.proc_dir, "train")
        pool = build_reference_pool(
            train_frames,
            max_pool_size=int(ga_cfg.max_pool_size),
            classical_cfg=ga_cfg.to_classical_config(),
        )
        del train_frames
        gc.collect()
        extra_meta["config"] = ga_cfg.to_dict()
        extra_meta["pool_size"] = len(pool)
        fn = lambda fr, _p=pool, _m=arm, _c=ga_cfg: run_group_a_frame(
            fr, context_pool=_p, mode=_m, config=_c
        )
    elif group == "group_b":
        gb_cfg = _load_group_b_config(args.results_root, arm)
        extra_meta["config"] = gb_cfg.to_dict()
        fn = lambda fr, _m=arm, _c=gb_cfg: run_group_b_frame(fr, mode=_m, config=_c)
    else:
        best_entry, ckpt_path = _resolve_canonical_checkpoint(
            arm, args.manifest_path, args.ckpt_dir
        )
        model = GroupCTrainableReceiver.load(ckpt_path)
        param_count = int(model.param_count)
        train_time_s = float(model.train_time_s)
        extra_meta.update(
            {
                "seed": int(model.seed),
                "best_epoch": int(model.best_epoch),
                "architecture": str(model.config.architecture),
                "checkpoint_file": str(ckpt_path.name),
                "checkpoint_bytes": int(ckpt_path.stat().st_size),
                "checkpoint_sha256": str(best_entry["sha256"]),
                "config": model.config.to_dict(),
            }
        )
        # context_pool=None ensures cached self.pool_summary is used without per-frame recomputation
        fn = lambda fr, _mod=model: _mod.run_frame(fr)

    prof = profile_receiver_pinned(
        test_frames,
        fn,
        warmup_frames=int(args.warmup_frames),
        param_count=param_count,
        train_time_s=train_time_s,
    )

    pairs = [(fr, fn(fr)) for fr in test_frames]
    ev = evaluate_receiver(pairs, param_count=param_count, train_time_s=train_time_s)
    ev["mean_ms_per_frame"] = float(prof["mean_ms_per_frame"])
    ev["median_ms_per_frame"] = float(prof["median_ms_per_frame"])
    ev["p95_ms_per_frame"] = float(prof["p95_ms_per_frame"])
    ev["peak_rss_mb"] = float(prof["peak_rss_mb"])
    ev["tracemalloc_peak_mb"] = float(prof["tracemalloc_peak_mb"])
    ev.update(extra_meta)

    per_frame_records = [
        _build_per_frame_record(arm, group, fr, out, param_count=param_count)
        for fr, out in pairs
    ]

    # 16-frame deterministic replay audit
    n_rep = min(16, len(test_frames))
    max_llr_diff = 0.0
    max_h_diff = 0.0
    bit_mismatches = 0
    for fr, out1 in pairs[:n_rep]:
        out2 = fn(fr)
        max_llr_diff = max(max_llr_diff, float(np.max(np.abs(out1.bit_llrs - out2.bit_llrs))))
        max_h_diff = max(max_h_diff, float(np.max(np.abs(out1.h_est - out2.h_est))))
        bit_mismatches += int(np.sum(out1.hard_bits != out2.hard_bits))
    max_float_diff = max(max_llr_diff, max_h_diff)
    replay = {
        "arm": arm,
        "n_frames": n_rep,
        "max_llr_diff": float(max_llr_diff),
        "max_h_diff": float(max_h_diff),
        "max_float_diff": float(max_float_diff),
        "bit_mismatches": int(bit_mismatches),
        "exact_pass": bool(max_llr_diff == 0.0 and max_h_diff == 0.0 and bit_mismatches == 0),
    }

    stress_suite: dict[str, Any] | None = None
    if not args.skip_stress:
        stress_suite = run_stress_suite(
            test_frames,
            fn,
            extra_awgn_db=(None, 10.0, 5.0, 0.0),
            cfo_norm=(0.0, 0.02, 0.05),
            phase_drift_rad=(0.0, 0.3, 0.6),
            seed=int(args.stress_seed),
            param_count=param_count,
        )

    payload = {
        "arm": arm,
        "group": group,
        "metrics": ev,
        "outputs": [out for _, out in pairs],
        "per_frame_records": per_frame_records,
        "replay": replay,
        "stress_suite": stress_suite,
    }
    out_pkl = Path(args.worker_out)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with out_pkl.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return 0


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})


def _git_rev_parse(ref: str) -> str:
    try:
        res = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", ref],
            capture_output=True,
            text=True,
            check=False,
        )
        return res.stdout.strip() if res.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _compute_sinr_stratified(
    test_frames: list[RealOFDMFrame],
    arm_pairs: dict[str, list[tuple[RealOFDMFrame, ReceiverOutput]]],
    arm_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    from research.realdata_sdr_ofdm.eval.evaluator import evaluate_receiver

    strata_indices: dict[str, list[int]] = {label: [] for label, _, _, _ in SINR_BINS}
    for idx, fr in enumerate(test_frames):
        lbl = _sinr_bin_label(float(fr.reported_sinr_db))
        strata_indices.setdefault(lbl, []).append(idx)

    out: dict[str, Any] = {
        "bins": [
            {
                "label": label,
                "lo_db": lo,
                "hi_db": hi,
                "inclusive_hi": inc_hi,
                "n_frames": len(strata_indices[label]),
            }
            for label, lo, hi, inc_hi in SINR_BINS
        ],
        "arms": {},
    }
    for arm in ALL_ARMS:
        pairs = arm_pairs[arm]
        rss_val = float(arm_metrics[arm]["peak_rss_mb"])
        p_cnt = int(arm_metrics[arm]["param_count"])
        t_sec = float(arm_metrics[arm]["train_time_s"])
        per_bin: dict[str, Any] = {}
        for label, _, _, _ in SINR_BINS:
            idxs = strata_indices[label]
            if not idxs:
                continue
            sub_pairs = [pairs[i] for i in idxs]
            ev = evaluate_receiver(
                sub_pairs,
                param_count=p_cnt,
                train_time_s=t_sec,
                peak_rss_mb=rss_val,
            )
            per_bin[label] = {
                "n_frames": int(ev["total_frames"]),
                "bit_errors": int(ev["bit_errors"]),
                "total_bits": int(ev["total_bits"]),
                "ber": float(ev["ber"]),
                "frame_errors": int(ev["frame_errors"]),
                "fer": float(ev["fer"]),
                "sym_errors": int(ev["sym_errors"]),
                "total_syms": int(ev["total_syms"]),
                "ser": float(ev["ser"]),
                "evm_pct": float(ev["evm_pct"]),
                "pilot_nmse": float(ev["pilot_nmse"]),
                "pilot_nll": float(ev["pilot_nll"]),
            }
        out["arms"][arm] = per_bin
    return out


def _run_parent(args: argparse.Namespace) -> int:
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    t0 = time.perf_counter()
    git_head = _git_rev_parse("HEAD")
    git_main = _git_rev_parse("main")
    log(
        f"Starting M5 Unified Fresh-Process Pinned Benchmark | "
        f"python={sys.version.split()[0]} platform={platform.platform()} "
        f"git_HEAD={git_head} git_main={git_main}"
    )
    log(
        f"Pinned threads: OMP={os.environ.get('OMP_NUM_THREADS')}, "
        f"OPENBLAS={os.environ.get('OPENBLAS_NUM_THREADS')}, "
        f"MKL={os.environ.get('MKL_NUM_THREADS')}"
    )
    log(
        f"Paths: proc_dir={args.proc_dir}, out_dir={out_dir}, "
        f"ckpt_dir={args.ckpt_dir}, manifest_path={args.manifest_path}, "
        f"results_root={args.results_root}"
    )
    log(f"Command: {' '.join(sys.argv)}")

    # Pre-verify canonical Group C checkpoints against manifest (zero torch/numpy import yet)
    for g_arm in ("group_c_branch_attn", "group_c_hybrid_attn", "group_c_pilot_nll_only"):
        best_entry, ckpt_path = _resolve_canonical_checkpoint(
            g_arm, args.manifest_path, args.ckpt_dir
        )
        log(
            f"Verified canonical ckpt {g_arm}: file={ckpt_path.name}, "
            f"seed={best_entry['seed']}, ep={best_entry['best_epoch']}, "
            f"dev_ber={best_entry['dev_ber']:.6f}, sha256={best_entry['sha256']}"
        )

    child_env = dict(os.environ)
    child_env["OMP_NUM_THREADS"] = "1"
    child_env["OPENBLAS_NUM_THREADS"] = "1"
    child_env["MKL_NUM_THREADS"] = "1"

    with tempfile.TemporaryDirectory(prefix="emff_m5_workers_") as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        worker_pkls: dict[str, Path] = {}

        # Phase A: Spawn all 13 isolated workers BEFORE loading numpy/torch/outputs in parent
        for arm in ALL_ARMS:
            pkl_path = tmp_dir / f"{arm}.pkl"
            cmd = [
                sys.executable,
                "-B",
                str(Path(__file__).resolve()),
                "--worker-arm",
                arm,
                "--worker-out",
                str(pkl_path),
                "--proc-dir",
                str(args.proc_dir),
                "--results-root",
                str(args.results_root),
                "--ckpt-dir",
                str(args.ckpt_dir),
                "--manifest-path",
                str(args.manifest_path),
                "--stress-seed",
                str(args.stress_seed),
                "--warmup-frames",
                str(args.warmup_frames),
            ]
            if args.skip_stress:
                cmd.append("--skip-stress")

            t_arm0 = time.perf_counter()
            proc = subprocess.run(cmd, env=child_env, check=False)
            dt_arm = time.perf_counter() - t_arm0
            if proc.returncode != 0 or not pkl_path.exists():
                raise RuntimeError(
                    f"Worker failed for arm={arm} (returncode={proc.returncode})"
                )
            worker_pkls[arm] = pkl_path
            log(f"Worker [{arm}] finished subprocess in {dt_arm:.1f}s (pkl={pkl_path.stat().st_size}B)")

        # Phase B: Now import scientific stack & load payloads for consolidated reporting
        import numpy as np
        import scipy
        import torch

        try:
            torch.set_num_threads(1)
        except RuntimeError:
            pass

        from research.realdata_sdr_ofdm.adapters.sdr_ofdm_adapter import load_processed_split
        from research.realdata_sdr_ofdm.eval.evaluator import (
            compare_arms,
            compare_multiple_arms,
            subblock_group_ids,
        )

        log(
            f"Runtime versions: torch={torch.__version__}, numpy={np.__version__}, "
            f"scipy={scipy.__version__}, torch_num_threads={torch.get_num_threads()}"
        )

        test_frames = load_processed_split(args.proc_dir, "test")
        n_test = len(test_frames)
        total_bits = sum(int(fr.tx_bits.size) for fr in test_frames)
        total_syms = sum(int(fr.tx_symbols.size) for fr in test_frames)
        log(f"Loaded test.npz: {n_test} frames, {total_syms} symbols, {total_bits} bits")

        arm_metrics: dict[str, dict[str, Any]] = {}
        arm_pairs: dict[str, list[tuple[RealOFDMFrame, ReceiverOutput]]] = {}
        all_per_frame_records: list[dict[str, Any]] = []
        replay_per_arm: dict[str, Any] = {}
        stress_arms: dict[str, Any] = {}
        clean_csv_rows: list[dict[str, Any]] = []
        stress_csv_rows: list[dict[str, Any]] = []
        conditions_order: list[str] = []

        for arm in ALL_ARMS:
            with worker_pkls[arm].open("rb") as fh:
                payload = pickle.load(fh)

            ev = payload["metrics"]
            arm_metrics[arm] = ev
            arm_pairs[arm] = list(zip(test_frames, payload["outputs"]))
            all_per_frame_records.extend(payload["per_frame_records"])
            replay_per_arm[arm] = payload["replay"]

            clean_csv_rows.append(
                {
                    "arm": arm,
                    "group": ARM_GROUP[arm],
                    "condition": "clean",
                    "extra_awgn_db": None,
                    "cfo_norm": 0.0,
                    "phase_drift_rad": 0.0,
                    **ev,
                }
            )
            log(
                f"Summary [{arm}]: BER={ev['ber']:.6f} ({ev['bit_errors']}/{ev['total_bits']}) "
                f"FER={ev['fer']:.4f} ({ev['frame_errors']}/{ev['total_frames']}) "
                f"EVM={ev['evm_pct']:.2f}% pilot_nll={ev['pilot_nll']:.4f} "
                f"ms={ev['mean_ms_per_frame']:.2f} (p95={ev['p95_ms_per_frame']:.2f}) "
                f"rss={ev['peak_rss_mb']:.2f}MB tm={ev['tracemalloc_peak_mb']:.2f}MB "
                f"replay_exact={payload['replay']['exact_pass']}"
            )

            suite = payload.get("stress_suite")
            if suite is not None:
                stress_arms[arm] = suite
                if not conditions_order:
                    conditions_order = [str(c["condition"]) for c in suite["conditions"]]
                for cond in suite["conditions"]:
                    met = cond["metrics"]
                    stress_csv_rows.append(
                        {
                            "arm": arm,
                            "group": ARM_GROUP[arm],
                            "condition": cond["condition"],
                            "extra_awgn_db": cond["extra_awgn_db"],
                            "cfo_norm": cond["cfo_norm"],
                            "phase_drift_rad": cond["phase_drift_rad"],
                            **met,
                        }
                    )

    # 1. Replay audit unified
    replay_unified = {
        "n_frames": 16,
        "n_arms": len(ALL_ARMS),
        "max_llr_diff": max(float(r["max_llr_diff"]) for r in replay_per_arm.values()),
        "max_h_diff": max(float(r["max_h_diff"]) for r in replay_per_arm.values()),
        "max_float_diff": max(float(r["max_float_diff"]) for r in replay_per_arm.values()),
        "bit_mismatches": sum(int(r["bit_mismatches"]) for r in replay_per_arm.values()),
        "exact_pass": all(bool(r["exact_pass"]) for r in replay_per_arm.values()),
        "per_arm": replay_per_arm,
    }
    assert replay_unified["exact_pass"], f"Replay audit failed: {replay_unified}"
    assert replay_unified["max_float_diff"] == 0.0
    (out_dir / "replay_audit_unified.json").write_text(
        json.dumps(replay_unified, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    # 2. SINR stratification + summary_clean.{json,csv}
    sinr_stratified = _compute_sinr_stratified(test_frames, arm_pairs, arm_metrics)
    summary_clean = {
        "schema_version": "1.0",
        "split": "test",
        "n_frames": int(n_test),
        "total_bits": int(total_bits),
        "total_syms": int(total_syms),
        "pool_split": "train",
        "pool_size": 64,
        "fresh_process_per_arm": True,
        "warmup_frames": int(args.warmup_frames),
        "thread_env": {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "torch_num_threads": 1,
        },
        "command": " ".join(sys.argv),
        "arms": arm_metrics,
        "sinr_stratified": sinr_stratified,
        "replay_audit": {
            "n_frames": replay_unified["n_frames"],
            "n_arms": replay_unified["n_arms"],
            "max_llr_diff": replay_unified["max_llr_diff"],
            "max_h_diff": replay_unified["max_h_diff"],
            "max_float_diff": replay_unified["max_float_diff"],
            "bit_mismatches": replay_unified["bit_mismatches"],
            "exact_pass": replay_unified["exact_pass"],
        },
    }
    (out_dir / "summary_clean.json").write_text(
        json.dumps(summary_clean, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_csv(out_dir / "summary_clean.csv", SUMMARY_CSV_FIELDS, clean_csv_rows)

    # 3. summary_stress.{json,csv}
    if not args.skip_stress:
        summary_stress = {
            "schema_version": "1.0",
            "split": "test",
            "n_frames": int(n_test),
            "seed": int(args.stress_seed),
            "conditions_order": conditions_order,
            "arms": stress_arms,
        }
        (out_dir / "summary_stress.json").write_text(
            json.dumps(summary_stress, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _write_csv(out_dir / "summary_stress.csv", SUMMARY_CSV_FIELDS, stress_csv_rows)

    # 4. per_frame_all.jsonl + per_frame_summary.csv (1,794 lines/rows)
    jsonl_path = out_dir / "per_frame_all.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in all_per_frame_records:
            fh.write(json.dumps(rec, allow_nan=False) + "\n")
    _write_csv(out_dir / "per_frame_summary.csv", PER_FRAME_SCALAR_FIELDS, all_per_frame_records)

    # 5. paired_comparisons_all.json (G=3 macro, G=12 sub-block, G=16 sub-block)
    sub12_map = subblock_group_ids(test_frames, target_g=12)
    sub16_map = subblock_group_ids(test_frames, target_g=16)

    def _sub12(f: RealOFDMFrame) -> str:
        return sub12_map[int(f.frame_id)]

    def _sub16(f: RealOFDMFrame) -> str:
        return sub16_map[int(f.frame_id)]

    anchors = (
        "ls_linear_zf",
        "lmmse_1d",
        "lmmse_2d_tracked",
        "global_uniform",
        "global_content_attn",
        "hard_map",
        "exact_mixture_likelihood",
    )
    macro_g3 = {
        f"vs_{anc}": compare_multiple_arms(arm_pairs, anchor_name=anc, metric="ber")
        for anc in anchors
    }
    subblock_g12 = {
        f"vs_{anc}": compare_multiple_arms(
            arm_pairs, anchor_name=anc, metric="ber", group_fn=_sub12
        )
        for anc in anchors
    }
    subblock_g16 = {
        f"vs_{anc}": compare_multiple_arms(
            arm_pairs, anchor_name=anc, metric="ber", group_fn=_sub16
        )
        for anc in anchors
    }

    key_pairs = (
        ("lmmse_1d_vs_ls_linear_zf", "ls_linear_zf", "lmmse_1d"),
        ("lmmse_2d_tracked_vs_lmmse_1d", "lmmse_1d", "lmmse_2d_tracked"),
        ("local_vs_lmmse_1d", "lmmse_1d", "local"),
        ("global_uniform_vs_lmmse_1d", "lmmse_1d", "global_uniform"),
        ("global_reliability_vs_lmmse_1d", "lmmse_1d", "global_reliability"),
        ("global_content_attn_vs_lmmse_1d", "lmmse_1d", "global_content_attn"),
        ("global_uniform_vs_local", "local", "global_uniform"),
        ("global_content_attn_vs_local", "local", "global_content_attn"),
        ("global_content_attn_vs_global_uniform", "global_uniform", "global_content_attn"),
        ("global_reliability_vs_global_uniform", "global_uniform", "global_reliability"),
        ("hard_map_vs_lmmse_1d", "lmmse_1d", "hard_map"),
        ("posterior_mean_plugin_vs_lmmse_1d", "lmmse_1d", "posterior_mean_plugin"),
        ("exact_mixture_likelihood_vs_lmmse_1d", "lmmse_1d", "exact_mixture_likelihood"),
        ("posterior_mean_plugin_vs_hard_map", "hard_map", "posterior_mean_plugin"),
        ("exact_mixture_likelihood_vs_hard_map", "hard_map", "exact_mixture_likelihood"),
        (
            "exact_mixture_likelihood_vs_posterior_mean_plugin",
            "posterior_mean_plugin",
            "exact_mixture_likelihood",
        ),
        ("group_c_branch_attn_vs_lmmse_1d", "lmmse_1d", "group_c_branch_attn"),
        ("group_c_hybrid_attn_vs_lmmse_1d", "lmmse_1d", "group_c_hybrid_attn"),
        ("group_c_pilot_nll_only_vs_lmmse_1d", "lmmse_1d", "group_c_pilot_nll_only"),
        ("group_c_hybrid_attn_vs_group_c_branch_attn", "group_c_branch_attn", "group_c_hybrid_attn"),
        (
            "group_c_hybrid_attn_vs_group_c_pilot_nll_only",
            "group_c_pilot_nll_only",
            "group_c_hybrid_attn",
        ),
        (
            "group_c_hybrid_attn_vs_exact_mixture_likelihood",
            "exact_mixture_likelihood",
            "group_c_hybrid_attn",
        ),
        (
            "group_c_hybrid_attn_vs_global_content_attn",
            "global_content_attn",
            "group_c_hybrid_attn",
        ),
    )
    pairwise_key_contrasts: dict[str, Any] = {}
    for label, anc, cand in key_pairs:
        pairwise_key_contrasts[label] = {
            "anchor": anc,
            "candidate": cand,
            "ber_g3": compare_arms(arm_pairs[anc], arm_pairs[cand], metric="ber"),
            "ber_g12": compare_arms(
                arm_pairs[anc], arm_pairs[cand], metric="ber", group_fn=_sub12
            ),
            "ber_g16": compare_arms(
                arm_pairs[anc], arm_pairs[cand], metric="ber", group_fn=_sub16
            ),
            "evm_pct_g16": compare_arms(
                arm_pairs[anc], arm_pairs[cand], metric="evm_pct", group_fn=_sub16
            ),
            "pilot_nmse_g16": compare_arms(
                arm_pairs[anc], arm_pairs[cand], metric="pilot_nmse", group_fn=_sub16
            ),
        }

    paired_all = {
        "schema_version": "1.0",
        "split": "test",
        "n_frames": int(n_test),
        "caveat": (
            "Test split contains G=3 independent macro-blocks (block_01, block_09, block_10; "
            "46 frames each; exact sign-flip has 2^3=8 states and minimum two-sided p=0.25). "
            "G=12 and G=16 contiguous sub-block tests are supplementary and subject to residual "
            "RF channel autocorrelation (h_pilot lag-32 corr ~0.736)."
        ),
        "macro_g3": macro_g3,
        "subblock_g12": subblock_g12,
        "subblock_g16": subblock_g16,
        "pairwise_key_contrasts": pairwise_key_contrasts,
    }
    (out_dir / "paired_comparisons_all.json").write_text(
        json.dumps(paired_all, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    dt_total = time.perf_counter() - t0
    log(f"All 13 arms completed in {dt_total:.2f}s. Output SHA-256 checksums:")
    output_names = [
        "summary_clean.json",
        "summary_clean.csv",
        "paired_comparisons_all.json",
        "per_frame_all.jsonl",
        "per_frame_summary.csv",
        "replay_audit_unified.json",
    ]
    if not args.skip_stress:
        output_names.extend(["summary_stress.json", "summary_stress.csv"])
    for fname in output_names:
        fpath = out_dir / fname
        log(f"  {fname}: bytes={fpath.stat().st_size}, sha256={_sha256_file(fpath)}")

    (out_dir / "unified_benchmark_log.txt").write_text(
        "\n".join(log_lines) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified fresh-process pinned benchmark across all 13 SDR OFDM arms."
    )
    parser.add_argument("--proc-dir", type=Path, default=Path("/tmp/tw_realdata/data_proc"))
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "research/realdata_sdr_ofdm/results",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=_REPO_ROOT / "research/realdata_sdr_ofdm/results",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=Path,
        default=_REPO_ROOT / "research/realdata_sdr_ofdm/checkpoints",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=_REPO_ROOT / "research/realdata_sdr_ofdm/checkpoints/manifest.json",
    )
    parser.add_argument("--stress-seed", type=int, default=20261122)
    parser.add_argument("--warmup-frames", type=int, default=8)
    parser.add_argument("--skip-stress", action="store_true")
    parser.add_argument("--worker-arm", type=str, default=None)
    parser.add_argument("--worker-out", type=Path, default=None)
    args = parser.parse_args()

    if args.worker_arm is not None:
        if args.worker_out is None:
            raise ValueError("--worker-out is required with --worker-arm")
        return _run_worker_arm(args)
    return _run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
