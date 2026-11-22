"""Run Group C (F10: Lightweight C*-Equivariant Branch-Weight Attention Module).

Executes:
1. Load train.npz (336), dev.npz (92), test.npz (138) and build M=64 stratified train-only pool.
2. 32-frame smoke check on dev.npz.
3. Deterministic 3-seed training (20261122, 20261123, 20261124) strictly on train.npz
   with checkpoint/seed selection strictly on dev.npz across:
   - `group_c_branch_attn` (6-branch single-TTI bank, TX-BCE + 0.05 pilot NLL reg)
   - `group_c_hybrid_attn` (7-branch bank adding fixed Group A global_content_attn branch)
   - `group_c_pilot_nll_only` (ablation: 7-branch bank trained with held-out pilot NLL only)
4. Frozen test.npz evaluation via reloaded canonical checkpoints + pinned profiler
   + deterministic 16-frame replay audit + 3-seed test variance summary (post-hoc).
5. Paired statistical comparisons vs M2 (`ls_linear_zf`, `lmmse_1d`, `lmmse_2d_tracked`),
   M3 Group A (`global_content_attn`), and M3 Group B (`exact_mixture_likelihood`)
   across G=3 macro-blocks, G=12, and G=16 sub-blocks + stress paired contrasts.
6. 10-condition stress suite (`run_stress_suite`) -> summary_clean/stress JSON+CSV.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    torch.set_num_threads(1)
except RuntimeError:
    pass

from research.realdata_sdr_ofdm.adapters.sdr_ofdm_adapter import (
    PILOT_FFT_INDICES,
    PILOT_SYMBOL_IDX,
    RealOFDMFrame,
    apply_stress_perturbation,
    load_processed_split,
)
from research.realdata_sdr_ofdm.eval.evaluator import (
    compare_arms,
    compare_multiple_arms,
    evaluate_receiver,
    profile_receiver_pinned,
    run_stress_suite,
    subblock_group_ids,
)
from research.realdata_sdr_ofdm.receivers.classical import (
    ClassicalConfig,
    ReceiverOutput,
)
from research.realdata_sdr_ofdm.receivers.classical import (
    run_frame as run_classical_frame,
)
from research.realdata_sdr_ofdm.receivers.group_a_reference import (
    GroupAConfig,
    build_reference_pool,
)
from research.realdata_sdr_ofdm.receivers.group_a_reference import (
    run_frame as run_group_a_frame,
)
from research.realdata_sdr_ofdm.receivers.group_b_fusion import (
    GroupBConfig,
)
from research.realdata_sdr_ofdm.receivers.group_b_fusion import (
    run_frame as run_group_b_frame,
)
from research.realdata_sdr_ofdm.receivers.group_c_trainable import (
    GroupCConfig,
    GroupCTrainableReceiver,
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_m2_anchor_configs(baseline_dir: Path | None) -> dict[str, ClassicalConfig]:
    defaults = {
        "ls_linear_zf": ClassicalConfig(n0_scale=0.7),
        "lmmse_1d": ClassicalConfig(n_taps=11, tau_rms=1.0, n0_scale=0.7, detrend_timing=True),
        "lmmse_2d_tracked": ClassicalConfig(
            n_taps=11, tau_rms=1.0, n0_scale=0.7, detrend_timing=True, fd_norm=0.0, phase_track=True
        ),
    }
    if baseline_dir is not None:
        dt_path = baseline_dir / "dev_tuning.json"
        if dt_path.exists():
            data = json.loads(dt_path.read_text(encoding="utf-8"))
            tuned = data.get("tuned", {})
            for m in list(defaults.keys()):
                if m in tuned and isinstance(tuned[m].get("config"), dict):
                    defaults[m] = ClassicalConfig(**tuned[m]["config"])
    return defaults


def _load_group_a_cfg(group_a_dir: Path | None) -> GroupAConfig:
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
    if group_a_dir is not None:
        dt_path = group_a_dir / "dev_tuning.json"
        if dt_path.exists():
            data = json.loads(dt_path.read_text(encoding="utf-8"))
            tuned = data.get("tuned", {}).get("global_content_attn", {}).get("config")
            if isinstance(tuned, dict):
                return GroupAConfig(**tuned)
    return default


def _load_group_b_cfg(group_b_dir: Path | None) -> GroupBConfig:
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
    if group_b_dir is not None:
        dt_path = group_b_dir / "dev_tuning.json"
        if dt_path.exists():
            data = json.loads(dt_path.read_text(encoding="utf-8"))
            tuned = data.get("tuned", {}).get("exact_mixture_likelihood", {}).get("config")
            if isinstance(tuned, dict):
                raw = dict(tuned)
                if "pdp_hypotheses" in raw:
                    raw["pdp_hypotheses"] = tuple((int(nt), float(tr)) for nt, tr in raw["pdp_hypotheses"])
                return GroupBConfig(**raw)
    return default


def _write_per_frame_jsonl(
    path: Path,
    pairs: list[tuple[RealOFDMFrame, ReceiverOutput]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for fr, out in pairs:
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

            bw = out.aux_info.get("branch_weights", [])
            rec = {
                "frame_id": int(fr.frame_id),
                "split": str(fr.split),
                "group_id": str(fr.group_id),
                "reported_sinr_db": float(fr.reported_sinr_db),
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
                "n_candidates": int(out.aux_info.get("n_candidates", 1)),
                "branch_entropy": float(out.aux_info.get("branch_entropy", 0.0)),
                "between_var_mean": float(out.aux_info.get("between_var_mean", 0.0)),
                "branch_weights": [float(x) for x in bw],
                "prior_weights": [float(x) for x in out.aux_info.get("prior_weights", [])],
                "omega_t_candidates": out.aux_info.get("omega_t_candidates", [0.0]),
                "param_count": int(out.aux_info.get("param_count", 0)),
                "latency_ms": float(out.aux_info.get("latency_ms", 0.0)),
                "failed": bool(out.aux_info.get("failed", False)),
                "error": out.aux_info.get("error"),
            }
            fh.write(json.dumps(rec, allow_nan=False) + "\n")


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = [
        "arm",
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
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Run M4 Group C lightweight attention experiments.")
    parser.add_argument("--proc-dir", type=Path, default=Path("/tmp/tw_realdata/data_proc"))
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/tw_realdata/outputs/group_c"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("/tmp/tw_realdata/outputs/baselines"))
    parser.add_argument("--group-a-dir", type=Path, default=Path("/tmp/tw_realdata/outputs/group_a"))
    parser.add_argument("--group-b-dir", type=Path, default=Path("/tmp/tw_realdata/outputs/group_b"))
    parser.add_argument("--smoke-n", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seeds", type=str, default="20261122,20261123,20261124")
    parser.add_argument("--stress-seed", type=int, default=20261122)
    parser.add_argument("--skip-stress", action="store_true")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        raise ValueError("At least one seed must be provided")

    log(f"Starting M4 Group C: proc_dir={args.proc_dir}, out_dir={out_dir}, epochs={args.epochs}, seeds={seeds}")
    train_frames = load_processed_split(args.proc_dir, "train")
    dev_frames = load_processed_split(args.proc_dir, "dev")
    test_frames = load_processed_split(args.proc_dir, "test")
    log(f"Loaded train={len(train_frames)}, dev={len(dev_frames)}, test={len(test_frames)}")

    m2_cfgs = _load_m2_anchor_configs(args.baseline_dir)
    ga_cfg = _load_group_a_cfg(args.group_a_dir)
    gb_cfg = _load_group_b_cfg(args.group_b_dir)
    train_pool = build_reference_pool(train_frames, max_pool_size=64, classical_cfg=ga_cfg.to_classical_config())
    log(f"Built M={len(train_pool)} stratified train pool for Group A & Group C hybrid arms.")

    arm_configs: dict[str, GroupCConfig] = {
        "group_c_branch_attn": GroupCConfig(
            include_global_branch=False,
            bce_weight=1.0,
            pilot_nll_weight=0.05,
            epochs=int(args.epochs),
            group_b_cfg=gb_cfg,
            group_a_cfg=ga_cfg,
        ),
        "group_c_hybrid_attn": GroupCConfig(
            include_global_branch=True,
            bce_weight=1.0,
            pilot_nll_weight=0.05,
            epochs=int(args.epochs),
            group_b_cfg=gb_cfg,
            group_a_cfg=ga_cfg,
        ),
        "group_c_pilot_nll_only": GroupCConfig(
            include_global_branch=True,
            bce_weight=0.0,
            pilot_nll_weight=1.0,
            epochs=int(args.epochs),
            group_b_cfg=gb_cfg,
            group_a_cfg=ga_cfg,
        ),
    }

    # 1. 32-frame smoke check on dev.npz
    smoke_frames = dev_frames[: min(args.smoke_n, len(dev_frames))]
    smoke_results: dict[str, Any] = {}
    for arm_name in ("group_c_branch_attn", "group_c_hybrid_attn"):
        smoke_cfg = GroupCConfig.from_dict(arm_configs[arm_name].to_dict())
        smoke_cfg.epochs = 2
        sm_model = GroupCTrainableReceiver(
            config=smoke_cfg,
            seed=seeds[0],
            context_pool=(train_pool if smoke_cfg.include_global_branch else None),
        ).fit(train_frames[:32], dev_frames[:16], epochs=2, context_pool=train_pool)
        pairs = [(fr, sm_model.run_frame(fr)) for fr in smoke_frames]
        res = evaluate_receiver(pairs, param_count=sm_model.param_count, train_time_s=sm_model.train_time_s)
        smoke_results[arm_name] = res
        assert res["fail_count"] == 0, f"Smoke failure in {arm_name}: {res['errors']}"
        log(
            f"Smoke ({len(smoke_frames)} dev frames) {arm_name}: "
            f"BER={res['ber']:.5f}, FER={res['fer']:.4f}, EVM={res['evm_pct']:.2f}%, "
            f"pilot_nll={res['pilot_nll']:.4f}, params={res['param_count']}, cov={res['coverage_pct']:.1f}%"
        )

    # 2. Deterministic 3-seed training strictly on train.npz + dev.npz selection
    t_tune0 = time.perf_counter()
    seed_runs: dict[str, list[dict[str, Any]]] = {}
    dev_tuned: dict[str, Any] = {}
    canonical_ckpt: dict[str, Path] = {}
    manifest_entries: list[dict[str, Any]] = []

    precompute_cache: dict[bool, tuple[list[dict[str, Any]], list[dict[str, Any]], float]] = {}
    for arm_name, cfg in arm_configs.items():
        if cfg.include_global_branch not in precompute_cache:
            pre_rx = GroupCTrainableReceiver(
                config=cfg,
                seed=seeds[0],
                context_pool=(train_pool if cfg.include_global_branch else None),
            )
            tr_it, dv_it, pre_s = pre_rx.precompute_training_splits(
                train_frames,
                dev_frames,
                include_cv_folds=True,
                context_pool=(train_pool if cfg.include_global_branch else None),
            )
            precompute_cache[cfg.include_global_branch] = (tr_it, dv_it, pre_s)
            log(f"Precomputed fixed candidate banks (include_global_branch={cfg.include_global_branch}) in {pre_s:.2f}s")

        tr_it, dv_it, pre_s = precompute_cache[cfg.include_global_branch]
        seed_runs[arm_name] = []
        best_dev_key = (float("inf"), float("inf"), float("inf"))
        best_rec: dict[str, Any] | None = None

        for sd in seeds:
            model = GroupCTrainableReceiver(
                config=cfg,
                seed=sd,
                context_pool=(train_pool if cfg.include_global_branch else None),
            )
            model.fit(
                train_frames,
                dev_frames,
                epochs=int(args.epochs),
                context_pool=(train_pool if cfg.include_global_branch else None),
                precomputed_train=tr_it,
                precomputed_dev=dv_it,
                precompute_time_s=pre_s,
            )
            ckpt_path = ckpt_dir / f"{arm_name}_seed{sd}.pt"
            model.save(ckpt_path)
            ckpt_bytes = int(ckpt_path.stat().st_size)
            assert ckpt_bytes < 1024 * 1024, f"Checkpoint {ckpt_path} exceeds 1 MB: {ckpt_bytes}"
            ckpt_sha = _sha256_file(ckpt_path)

            dev_pairs = [(fr, model.run_frame(fr)) for fr in dev_frames]
            dev_ev = evaluate_receiver(
                dev_pairs,
                param_count=model.param_count,
                train_time_s=model.train_time_s,
            )
            # Post-hoc test evaluation logged per seed for variance transparency (NEVER used for selection!)
            test_pairs_sd = [(fr, model.run_frame(fr)) for fr in test_frames]
            test_ev_sd = evaluate_receiver(
                test_pairs_sd,
                param_count=model.param_count,
                train_time_s=model.train_time_s,
            )
            rec = {
                "arm": arm_name,
                "seed": int(sd),
                "best_epoch": int(model.best_epoch),
                "train_time_s": float(model.train_time_s),
                "param_count": int(model.param_count),
                "architecture": str(cfg.architecture),
                "checkpoint_file": ckpt_path.name,
                "checkpoint_bytes": ckpt_bytes,
                "checkpoint_sha256": ckpt_sha,
                "dev_metrics": dev_ev,
                "test_metrics_posthoc": test_ev_sd,
                "history": model.history,
            }
            seed_runs[arm_name].append(rec)
            manifest_entries.append(
                {
                    "arm": arm_name,
                    "seed": int(sd),
                    "file": ckpt_path.name,
                    "bytes": ckpt_bytes,
                    "sha256": ckpt_sha,
                    "param_count": int(model.param_count),
                    "best_epoch": int(model.best_epoch),
                    "train_time_s": float(model.train_time_s),
                    "dev_ber": float(dev_ev["ber"]),
                    "dev_pilot_nll": float(dev_ev["pilot_nll"]),
                }
            )
            log(
                f"Train {arm_name} seed={sd}: best_ep={model.best_epoch}/{args.epochs}, "
                f"train_s={model.train_time_s:.2f}s, dev_BER={dev_ev['ber']:.5f}, "
                f"dev_nll={dev_ev['pilot_nll']:.4f}, test_BER(posthoc)={test_ev_sd['ber']:.5f}, "
                f"ckpt={ckpt_bytes}B"
            )

            # Canonical seed selection strictly on dev_frames!
            sel_key = (float(dev_ev["ber"]), float(dev_ev["pilot_nll"]), float(sd))
            if sel_key < best_dev_key:
                best_dev_key = sel_key
                best_rec = rec
                canonical_ckpt[arm_name] = ckpt_path

            del model
            gc.collect()

        assert best_rec is not None
        dev_tuned[arm_name] = {
            "config": cfg.to_dict(),
            "canonical_seed": int(best_rec["seed"]),
            "canonical_checkpoint": str(best_rec["checkpoint_file"]),
            "best_epoch": int(best_rec["best_epoch"]),
            "train_time_s": float(best_rec["train_time_s"]),
            "param_count": int(best_rec["param_count"]),
            "metrics": best_rec["dev_metrics"],
        }

    del precompute_cache
    gc.collect()
    tune_time_s = time.perf_counter() - t_tune0
    dev_tuning_payload = {
        "split_used": "dev",
        "train_split": "train",
        "n_train_frames": len(train_frames),
        "n_dev_frames": len(dev_frames),
        "pool_size": len(train_pool),
        "seeds": seeds,
        "epochs": int(args.epochs),
        "tune_time_s": float(tune_time_s),
        "smoke": smoke_results,
        "seed_runs": seed_runs,
        "tuned": dev_tuned,
    }
    (out_dir / "dev_tuning.json").write_text(
        json.dumps(dev_tuning_payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest_payload = {
        "schema_version": "1.0",
        "note": "Track checkpoint metadata in git while keeping .pt binaries untracked/gitignored per T1-F1.4.",
        "checkpoints": manifest_entries,
    }
    (ckpt_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    # 3. Frozen test evaluation via reloaded canonical checkpoints + anchors
    anchor_pairs: dict[str, list[tuple[RealOFDMFrame, ReceiverOutput]]] = {
        m: [(fr, run_classical_frame(fr, method=m, config=m2_cfgs[m])) for fr in test_frames]
        for m in ("ls_linear_zf", "lmmse_1d", "lmmse_2d_tracked")
    }
    anchor_pairs["group_a_global_content_attn"] = [
        (fr, run_group_a_frame(fr, context_pool=train_pool, mode="global_content_attn", config=ga_cfg))
        for fr in test_frames
    ]
    anchor_pairs["group_b_exact_mixture_likelihood"] = [
        (fr, run_group_b_frame(fr, mode="exact_mixture_likelihood", config=gb_cfg))
        for fr in test_frames
    ]

    arm_pairs: dict[str, list[tuple[RealOFDMFrame, ReceiverOutput]]] = {}
    loaded_models: dict[str, GroupCTrainableReceiver] = {}
    clean_arms: dict[str, Any] = {}
    clean_csv_rows: list[dict[str, Any]] = []
    seed_summary: dict[str, Any] = {}

    for arm_name in arm_configs:
        model = GroupCTrainableReceiver.load(canonical_ckpt[arm_name])
        loaded_models[arm_name] = model

        pairs = [(fr, model.run_frame(fr)) for fr in test_frames]
        arm_pairs[arm_name] = pairs
        _write_per_frame_jsonl(out_dir / f"per_frame_{arm_name}.jsonl", pairs)

        prof = profile_receiver_pinned(
            test_frames,
            lambda f, _m=model: _m.run_frame(f),
            warmup_frames=8,
            param_count=model.param_count,
            train_time_s=model.train_time_s,
        )
        ev = evaluate_receiver(pairs, param_count=model.param_count, train_time_s=model.train_time_s)
        ev["mean_ms_per_frame"] = prof["mean_ms_per_frame"]
        ev["median_ms_per_frame"] = prof["median_ms_per_frame"]
        ev["p95_ms_per_frame"] = prof["p95_ms_per_frame"]
        ev["peak_rss_mb"] = prof["peak_rss_mb"]
        ev["tracemalloc_peak_mb"] = prof["tracemalloc_peak_mb"]
        ev["seed"] = int(model.seed)
        ev["best_epoch"] = int(model.best_epoch)
        ev["architecture"] = str(model.config.architecture)
        ev["checkpoint_file"] = canonical_ckpt[arm_name].name
        ev["checkpoint_bytes"] = int(canonical_ckpt[arm_name].stat().st_size)
        ev["config"] = model.config.to_dict()
        clean_arms[arm_name] = ev
        clean_csv_rows.append({"arm": arm_name, "condition": "clean", **ev})

        test_bers = [float(r["test_metrics_posthoc"]["ber"]) for r in seed_runs[arm_name]]
        test_nlls = [float(r["test_metrics_posthoc"]["pilot_nll"]) for r in seed_runs[arm_name]]
        seed_summary[arm_name] = {
            "seeds": [int(r["seed"]) for r in seed_runs[arm_name]],
            "canonical_seed": int(model.seed),
            "test_ber_per_seed": test_bers,
            "test_ber_mean": float(np.mean(test_bers)),
            "test_ber_std": float(np.std(test_bers, ddof=1)) if len(test_bers) > 1 else 0.0,
            "test_pilot_nll_per_seed": test_nlls,
            "test_pilot_nll_mean": float(np.mean(test_nlls)),
        }
        log(
            f"Test clean {arm_name} (seed={model.seed}, {len(test_frames)} frames): "
            f"BER={ev['ber']:.5f} ({ev['bit_errors']}/{ev['total_bits']}), "
            f"FER={ev['fer']:.4f} ({ev['frame_errors']}/{ev['total_frames']}), "
            f"SER={ev['ser']:.5f}, EVM={ev['evm_pct']:.2f}%, "
            f"pilot_nmse={ev['pilot_nmse']:.5f}, cal={ev['pilot_cal_ratio']:.4f}, "
            f"nll={ev['pilot_nll']:.4f}, params={ev['param_count']}, "
            f"ms/frame={ev['mean_ms_per_frame']:.2f}, rss_mb={ev['peak_rss_mb']:.1f}"
        )

    # 4. Deterministic 16-frame replay audit
    replay_arm = "group_c_hybrid_attn"
    replay_model = loaded_models[replay_arm]
    max_llr_diff = 0.0
    max_h_diff = 0.0
    bit_mismatches = 0
    for fr, out1 in arm_pairs[replay_arm][:16]:
        out2 = replay_model.run_frame(fr)
        max_llr_diff = max(max_llr_diff, float(np.max(np.abs(out1.bit_llrs - out2.bit_llrs))))
        max_h_diff = max(max_h_diff, float(np.max(np.abs(out1.h_est - out2.h_est))))
        bit_mismatches += int(np.sum(out1.hard_bits != out2.hard_bits))
    replay_audit = {
        "arm": replay_arm,
        "n_frames": min(16, len(test_frames)),
        "max_llr_diff": max_llr_diff,
        "max_h_diff": max_h_diff,
        "bit_mismatches": bit_mismatches,
        "exact_pass": bool(max_llr_diff == 0.0 and max_h_diff == 0.0 and bit_mismatches == 0),
    }
    log(f"Deterministic replay audit: {replay_audit}")

    # 5. Paired statistical comparisons vs M2 & M3 anchors + intra-group contrasts
    sub12_map = subblock_group_ids(test_frames, target_g=12)
    sub16_map = subblock_group_ids(test_frames, target_g=16)

    def _sub12(f: RealOFDMFrame) -> str:
        return sub12_map[int(f.frame_id)]

    def _sub16(f: RealOFDMFrame) -> str:
        return sub16_map[int(f.frame_id)]

    paired_payload: dict[str, Any] = {
        "caveat": (
            "Test split has G=3 independent macro-blocks (block_01, block_09, block_10; 2^3=8 states, "
            "min two-sided p=0.25). G=12 and G=16 contiguous sub-block analyses are supplementary and "
            "subject to residual RF channel correlation (h_pilot lag-32 corr ~0.736)."
        ),
        "macro_g3_vs_lmmse_1d": compare_multiple_arms(
            {"lmmse_1d": anchor_pairs["lmmse_1d"], **arm_pairs}, anchor_name="lmmse_1d", metric="ber"
        ),
        "subblock_g12_vs_lmmse_1d": compare_multiple_arms(
            {"lmmse_1d": anchor_pairs["lmmse_1d"], **arm_pairs},
            anchor_name="lmmse_1d",
            metric="ber",
            group_fn=_sub12,
        ),
        "subblock_g16_vs_lmmse_1d": compare_multiple_arms(
            {"lmmse_1d": anchor_pairs["lmmse_1d"], **arm_pairs},
            anchor_name="lmmse_1d",
            metric="ber",
            group_fn=_sub16,
        ),
        "macro_g3_vs_ls_linear_zf": compare_multiple_arms(
            {"ls_linear_zf": anchor_pairs["ls_linear_zf"], **arm_pairs}, anchor_name="ls_linear_zf", metric="ber"
        ),
        "subblock_g16_vs_ls_linear_zf": compare_multiple_arms(
            {"ls_linear_zf": anchor_pairs["ls_linear_zf"], **arm_pairs},
            anchor_name="ls_linear_zf",
            metric="ber",
            group_fn=_sub16,
        ),
        "macro_g3_vs_lmmse_2d_tracked": compare_multiple_arms(
            {"lmmse_2d_tracked": anchor_pairs["lmmse_2d_tracked"], **arm_pairs},
            anchor_name="lmmse_2d_tracked",
            metric="ber",
        ),
        "subblock_g16_vs_lmmse_2d_tracked": compare_multiple_arms(
            {"lmmse_2d_tracked": anchor_pairs["lmmse_2d_tracked"], **arm_pairs},
            anchor_name="lmmse_2d_tracked",
            metric="ber",
            group_fn=_sub16,
        ),
        "macro_g3_vs_group_b_mix": compare_multiple_arms(
            {"group_b_exact_mixture_likelihood": anchor_pairs["group_b_exact_mixture_likelihood"], **arm_pairs},
            anchor_name="group_b_exact_mixture_likelihood",
            metric="ber",
        ),
        "subblock_g16_vs_group_b_mix": compare_multiple_arms(
            {"group_b_exact_mixture_likelihood": anchor_pairs["group_b_exact_mixture_likelihood"], **arm_pairs},
            anchor_name="group_b_exact_mixture_likelihood",
            metric="ber",
            group_fn=_sub16,
        ),
        "macro_g3_vs_group_a_attn": compare_multiple_arms(
            {"group_a_global_content_attn": anchor_pairs["group_a_global_content_attn"], **arm_pairs},
            anchor_name="group_a_global_content_attn",
            metric="ber",
        ),
        "subblock_g16_vs_group_a_attn": compare_multiple_arms(
            {"group_a_global_content_attn": anchor_pairs["group_a_global_content_attn"], **arm_pairs},
            anchor_name="group_a_global_content_attn",
            metric="ber",
            group_fn=_sub16,
        ),
        "intra_group": {
            "hybrid_vs_branch_attn_g3": compare_arms(
                arm_pairs["group_c_branch_attn"], arm_pairs["group_c_hybrid_attn"], metric="ber"
            ),
            "hybrid_vs_branch_attn_g16": compare_arms(
                arm_pairs["group_c_branch_attn"], arm_pairs["group_c_hybrid_attn"], metric="ber", group_fn=_sub16
            ),
            "hybrid_vs_pilot_nll_only_g3": compare_arms(
                arm_pairs["group_c_pilot_nll_only"], arm_pairs["group_c_hybrid_attn"], metric="ber"
            ),
            "hybrid_vs_pilot_nll_only_g16": compare_arms(
                arm_pairs["group_c_pilot_nll_only"], arm_pairs["group_c_hybrid_attn"], metric="ber", group_fn=_sub16
            ),
        },
    }

    if not args.skip_stress:
        for cond_tag, awgn_v, cfo_v, drift_v in (
            ("cfo_0.05", None, 0.05, 0.0),
            ("drift_0.60", None, 0.0, 0.6),
            ("awgn_0.0db", 0.0, 0.0, 0.0),
        ):
            stressed = [
                apply_stress_perturbation(
                    fr,
                    extra_awgn_db=awgn_v,
                    cfo_norm=cfo_v,
                    phase_drift_rad=drift_v,
                    seed=int(args.stress_seed) + int(fr.frame_id),
                )
                for fr in test_frames
            ]
            st_c_hyb = [(fs, loaded_models["group_c_hybrid_attn"].run_frame(fs)) for fs in stressed]
            st_c_br = [(fs, loaded_models["group_c_branch_attn"].run_frame(fs)) for fs in stressed]
            st_b = [(fs, run_group_b_frame(fs, mode="exact_mixture_likelihood", config=gb_cfg)) for fs in stressed]
            st_a = [
                (fs, run_group_a_frame(fs, context_pool=train_pool, mode="global_content_attn", config=ga_cfg))
                for fs in stressed
            ]
            paired_payload[f"stress_{cond_tag}_hybrid_vs_group_b_g16"] = compare_arms(
                st_b, st_c_hyb, metric="ber", group_fn=_sub16
            )
            paired_payload[f"stress_{cond_tag}_hybrid_vs_group_a_g16"] = compare_arms(
                st_a, st_c_hyb, metric="ber", group_fn=_sub16
            )
            paired_payload[f"stress_{cond_tag}_branch_vs_group_b_g16"] = compare_arms(
                st_b, st_c_br, metric="ber", group_fn=_sub16
            )

    (out_dir / "paired_comparisons.json").write_text(
        json.dumps(paired_payload, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    summary_clean = {
        "split": "test",
        "n_frames": len(test_frames),
        "thread_env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "torch_num_threads": torch.get_num_threads(),
        },
        "command": " ".join(sys.argv),
        "arms": clean_arms,
        "seed_summary": seed_summary,
        "replay_audit": replay_audit,
        "paired_summary": {
            "g3_vs_lmmse_1d": {
                k: {kk: v[kk] for kk in ("mean_pp", "p_two_sided", "p_holm", "verdict", "wins", "ties", "losses", "error_delta")}
                for k, v in paired_payload["macro_g3_vs_lmmse_1d"].items()
            },
            "g16_vs_lmmse_1d": {
                k: {kk: v[kk] for kk in ("mean_pp", "p_two_sided", "p_holm", "verdict", "wins", "ties", "losses", "error_delta")}
                for k, v in paired_payload["subblock_g16_vs_lmmse_1d"].items()
            },
            "g16_vs_group_b_mix": {
                k: {kk: v[kk] for kk in ("mean_pp", "p_two_sided", "p_holm", "verdict", "wins", "ties", "losses", "error_delta")}
                for k, v in paired_payload["subblock_g16_vs_group_b_mix"].items()
            },
            "g16_vs_group_a_attn": {
                k: {kk: v[kk] for kk in ("mean_pp", "p_two_sided", "p_holm", "verdict", "wins", "ties", "losses", "error_delta")}
                for k, v in paired_payload["subblock_g16_vs_group_a_attn"].items()
            },
        },
    }
    (out_dir / "summary_clean.json").write_text(
        json.dumps(summary_clean, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_summary_csv(out_dir / "summary_clean.csv", clean_csv_rows)

    # 6. 10-condition stress suite
    if not args.skip_stress:
        stress_arms: dict[str, Any] = {}
        stress_csv_rows: list[dict[str, Any]] = []
        for arm_name, model in loaded_models.items():
            suite = run_stress_suite(
                test_frames,
                lambda f, _m=model: _m.run_frame(f),
                extra_awgn_db=(None, 10.0, 5.0, 0.0),
                cfo_norm=(0.0, 0.02, 0.05),
                phase_drift_rad=(0.0, 0.3, 0.6),
                seed=int(args.stress_seed),
                param_count=model.param_count,
            )
            stress_arms[arm_name] = suite
            for cond in suite["conditions"]:
                met = cond["metrics"]
                stress_csv_rows.append(
                    {
                        "arm": arm_name,
                        "condition": cond["condition"],
                        "extra_awgn_db": cond["extra_awgn_db"],
                        "cfo_norm": cond["cfo_norm"],
                        "phase_drift_rad": cond["phase_drift_rad"],
                        **met,
                    }
                )
                log(
                    f"Stress {arm_name} [{cond['condition']}]: "
                    f"BER={met['ber']:.5f}, EVM={met['evm_pct']:.2f}%, cov={met['coverage_pct']:.1f}%"
                )

        summary_stress = {
            "split": "test",
            "n_frames": len(test_frames),
            "seed": int(args.stress_seed),
            "arms": stress_arms,
        }
        (out_dir / "summary_stress.json").write_text(
            json.dumps(summary_stress, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _write_summary_csv(out_dir / "summary_stress.csv", stress_csv_rows)

    (out_dir / "run_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8", newline="\n")
    log("Completed M4 Group C run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
