"""Run M2 classical receiver baselines on processed SDR OFDM splits (F6 & F7).

Executes:
1. 32-frame smoke check on `dev.npz`
2. Strictly dev-only hyperparameter tuning (`dev_tuning.json`)
3. Frozen `test.npz` (138 frames) clean evaluation + pinned profiler + paired audits
   (`G=3` macro-block + `G=12` / `G=16` contiguous sub-block with autocorrelation caveat)
4. Frozen `test.npz` controllable stress sweeps (`extra_awgn_db`, `cfo_norm`, `phase_drift_rad`)
5. Deterministic single-worker replay verification (`max_float_diff == 0.0`)
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    torch.set_num_threads(1)
except RuntimeError:
    pass

from research.realdata_sdr_ofdm.adapters.sdr_ofdm_adapter import (
    PILOT_FFT_INDICES,
    PILOT_SYMBOL_IDX,
    RealOFDMFrame,
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
    CLASSICAL_METHODS,
    ClassicalConfig,
    ReceiverOutput,
    calibrate_on_dev,
    run_frame,
)


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
                "latency_ms": float(out.aux_info.get("latency_ms", 0.0)),
                "omega_t_rad_per_sym": float(out.aux_info.get("omega_t_rad_per_sym", 0.0)),
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
    parser = argparse.ArgumentParser(description="Run M2 classical baselines and unified evaluator.")
    parser.add_argument("--proc-dir", type=Path, default=Path("/tmp/tw_realdata/data_proc"))
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/tw_realdata/outputs/baselines"))
    parser.add_argument("--smoke-n", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--skip-stress", action="store_true")
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {msg}"
        print(line, flush=True)
        log_lines.append(line)

    log(f"Starting M2 classical baselines: proc_dir={args.proc_dir}, out_dir={out_dir}")
    dev_frames = load_processed_split(args.proc_dir, "dev")
    test_frames = load_processed_split(args.proc_dir, "test")
    log(f"Loaded dev={len(dev_frames)} frames, test={len(test_frames)} frames")

    # 1. 32-frame smoke check on dev
    smoke_frames = dev_frames[: min(args.smoke_n, len(dev_frames))]
    smoke_results: dict[str, Any] = {}
    for m in CLASSICAL_METHODS:
        pairs = [(fr, run_frame(fr, method=m)) for fr in smoke_frames]
        res = evaluate_receiver(pairs)
        smoke_results[m] = res
        assert res["fail_count"] == 0, f"Smoke failure in {m}: {res['errors']}"
        log(
            f"Smoke ({len(smoke_frames)} dev frames) {m}: "
            f"BER={res['ber']:.5f}, FER={res['fer']:.4f}, EVM={res['evm_pct']:.2f}%, "
            f"pilot_nll={res['pilot_nll']:.4f}, cov={res['coverage_pct']:.1f}%"
        )

    # 2. Strictly dev-only calibration
    t_tune0 = time.perf_counter()
    tuned_configs = calibrate_on_dev(
        dev_frames,
        methods=CLASSICAL_METHODS,
        n0_scales=(0.7, 1.0, 1.3),
        tau_rms_grid=(1.0, 1.5, 2.5, 4.0),
        n_taps_grid=(5, 7, 9, 11),
        fd_norm_grid=(0.0, 0.002, 0.005),
        cpe_refine_grid=(0.0,),
    )
    # Also check detrend_timing=True on dev for lmmse_1d & lmmse_2d_tracked
    for m in ("lmmse_1d", "lmmse_2d_tracked"):
        base_cfg = tuned_configs[m]
        best_res = evaluate_receiver([(fr, run_frame(fr, method=m, config=base_cfg)) for fr in dev_frames])
        best_cfg = base_cfg
        for nt in (base_cfg.n_taps, 9, 11):
            for trms in (base_cfg.tau_rms, 2.5):
                cand_cfg = ClassicalConfig(
                    **{**base_cfg.to_dict(), "detrend_timing": True, "n_taps": nt, "tau_rms": trms}
                )
                cand_res = evaluate_receiver([(fr, run_frame(fr, method=m, config=cand_cfg)) for fr in dev_frames])
                if (cand_res["ber"], cand_res["sym_mse"], cand_res["pilot_nll"]) < (
                    best_res["ber"],
                    best_res["sym_mse"],
                    best_res["pilot_nll"],
                ):
                    best_res = cand_res
                    best_cfg = cand_cfg
        tuned_configs[m] = best_cfg

    tune_time_s = time.perf_counter() - t_tune0
    dev_eval: dict[str, Any] = {}
    for m, cfg in tuned_configs.items():
        dev_eval[m] = {
            "config": cfg.to_dict(),
            "metrics": evaluate_receiver([(fr, run_frame(fr, method=m, config=cfg)) for fr in dev_frames]),
        }
        log(
            f"Dev tuned {m}: cfg={cfg.to_dict()} -> "
            f"BER={dev_eval[m]['metrics']['ber']:.5f}, "
            f"pilot_nmse={dev_eval[m]['metrics']['pilot_nmse']:.5f}, "
            f"pilot_cal={dev_eval[m]['metrics']['pilot_cal_ratio']:.4f}"
        )

    dev_tuning_payload = {
        "split_used": "dev",
        "n_dev_frames": len(dev_frames),
        "tune_time_s": float(tune_time_s),
        "smoke": smoke_results,
        "tuned": dev_eval,
    }
    (out_dir / "dev_tuning.json").write_text(
        json.dumps(dev_tuning_payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    # 3. Frozen test clean evaluation + pinned profiler
    arm_pairs: dict[str, list[tuple[RealOFDMFrame, ReceiverOutput]]] = {}
    clean_arms: dict[str, Any] = {}
    clean_csv_rows: list[dict[str, Any]] = []

    for m in CLASSICAL_METHODS:
        cfg = tuned_configs[m]
        pairs = [(fr, run_frame(fr, method=m, config=cfg)) for fr in test_frames]
        arm_pairs[m] = pairs
        _write_per_frame_jsonl(out_dir / f"per_frame_{m}.jsonl", pairs)

        prof = profile_receiver_pinned(
            test_frames,
            lambda f, _m=m, _c=cfg: run_frame(f, method=_m, config=_c),
            warmup_frames=8,
            param_count=0,
            train_time_s=0.0,
        )
        # Preserve exact per-frame metrics from canonical run_frame pairs
        ev = evaluate_receiver(pairs, param_count=0, train_time_s=0.0)
        ev["mean_ms_per_frame"] = prof["mean_ms_per_frame"]
        ev["median_ms_per_frame"] = prof["median_ms_per_frame"]
        ev["p95_ms_per_frame"] = prof["p95_ms_per_frame"]
        ev["peak_rss_mb"] = prof["peak_rss_mb"]
        ev["tracemalloc_peak_mb"] = prof["tracemalloc_peak_mb"]
        ev["config"] = cfg.to_dict()
        clean_arms[m] = ev
        clean_csv_rows.append({"arm": m, "condition": "clean", **ev})
        log(
            f"Test clean {m} ({len(test_frames)} frames): "
            f"BER={ev['ber']:.5f} ({ev['bit_errors']}/{ev['total_bits']}), "
            f"FER={ev['fer']:.4f} ({ev['frame_errors']}/{ev['total_frames']}), "
            f"SER={ev['ser']:.5f}, EVM={ev['evm_pct']:.2f}%, "
            f"pilot_nmse={ev['pilot_nmse']:.5f}, cal={ev['pilot_cal_ratio']:.4f}, "
            f"nll={ev['pilot_nll']:.4f}, ms/frame={ev['mean_ms_per_frame']:.2f}, "
            f"rss_mb={ev['peak_rss_mb']:.1f}"
        )

    # Deterministic replay check on first 16 test frames
    replay_m = "lmmse_2d_tracked"
    replay_cfg = tuned_configs[replay_m]
    max_llr_diff = 0.0
    max_h_diff = 0.0
    bit_mismatches = 0
    for fr, out1 in arm_pairs[replay_m][:16]:
        out2 = run_frame(fr, method=replay_m, config=replay_cfg)
        max_llr_diff = max(max_llr_diff, float(np.max(np.abs(out1.bit_llrs - out2.bit_llrs))))
        max_h_diff = max(max_h_diff, float(np.max(np.abs(out1.h_est - out2.h_est))))
        bit_mismatches += int(np.sum(out1.hard_bits != out2.hard_bits))
    replay_audit = {
        "arm": replay_m,
        "n_frames": min(16, len(test_frames)),
        "max_llr_diff": max_llr_diff,
        "max_h_diff": max_h_diff,
        "bit_mismatches": bit_mismatches,
        "exact_pass": bool(max_llr_diff == 0.0 and max_h_diff == 0.0 and bit_mismatches == 0),
    }
    log(f"Deterministic replay audit: {replay_audit}")

    # Paired comparisons: G=3 macro-blocks, G=12 sub-blocks, G=16 sub-blocks
    sub12_map = subblock_group_ids(test_frames, target_g=12)
    sub16_map = subblock_group_ids(test_frames, target_g=16)

    paired_payload: dict[str, Any] = {
        "anchor": "ls_linear_zf",
        "caveat": (
            "Test split has G=3 independent macro-blocks (block_01, block_09, block_10; 2^3=8 states, "
            "min two-sided p=0.25). G=12 and G=16 contiguous sub-block analyses are supplementary and "
            "subject to residual RF channel correlation (h_pilot lag-32 corr ~0.736)."
        ),
        "macro_g3_vs_ls_zf": compare_multiple_arms(arm_pairs, anchor_name="ls_linear_zf", metric="ber"),
        "subblock_g12_vs_ls_zf": compare_multiple_arms(
            arm_pairs,
            anchor_name="ls_linear_zf",
            metric="ber",
            group_fn=lambda f: sub12_map[int(f.frame_id)],
        ),
        "subblock_g16_vs_ls_zf": compare_multiple_arms(
            arm_pairs,
            anchor_name="ls_linear_zf",
            metric="ber",
            group_fn=lambda f: sub16_map[int(f.frame_id)],
        ),
        "lmmse_2d_vs_1d_g3": compare_arms(arm_pairs["lmmse_1d"], arm_pairs["lmmse_2d_tracked"], metric="ber"),
        "lmmse_2d_vs_1d_g16": compare_arms(
            arm_pairs["lmmse_1d"],
            arm_pairs["lmmse_2d_tracked"],
            metric="ber",
            group_fn=lambda f: sub16_map[int(f.frame_id)],
        ),
    }
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
        "replay_audit": replay_audit,
        "paired_summary": {
            "g3_vs_ls_zf": {
                k: {kk: v[kk] for kk in ("mean_pp", "p_two_sided", "p_holm", "verdict", "wins", "ties", "losses")}
                for k, v in paired_payload["macro_g3_vs_ls_zf"].items()
            },
            "g16_vs_ls_zf": {
                k: {kk: v[kk] for kk in ("mean_pp", "p_two_sided", "p_holm", "verdict", "wins", "ties", "losses")}
                for k, v in paired_payload["subblock_g16_vs_ls_zf"].items()
            },
        },
    }
    (out_dir / "summary_clean.json").write_text(
        json.dumps(summary_clean, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_summary_csv(out_dir / "summary_clean.csv", clean_csv_rows)

    # 4. Controllable stress perturbations (saved separately)
    if not args.skip_stress:
        stress_arms: dict[str, Any] = {}
        stress_csv_rows: list[dict[str, Any]] = []
        for m in CLASSICAL_METHODS:
            cfg = tuned_configs[m]
            suite = run_stress_suite(
                test_frames,
                lambda f, _m=m, _c=cfg: run_frame(f, method=_m, config=_c),
                extra_awgn_db=(None, 10.0, 5.0, 0.0),
                cfo_norm=(0.0, 0.02, 0.05),
                phase_drift_rad=(0.0, 0.3, 0.6),
                seed=args.seed,
                param_count=0,
            )
            stress_arms[m] = suite
            for cond in suite["conditions"]:
                met = cond["metrics"]
                stress_csv_rows.append(
                    {
                        "arm": m,
                        "condition": cond["condition"],
                        "extra_awgn_db": cond["extra_awgn_db"],
                        "cfo_norm": cond["cfo_norm"],
                        "phase_drift_rad": cond["phase_drift_rad"],
                        **met,
                    }
                )
                log(
                    f"Stress {m} [{cond['condition']}]: "
                    f"BER={met['ber']:.5f}, EVM={met['evm_pct']:.2f}%, cov={met['coverage_pct']:.1f}%"
                )

        summary_stress = {
            "split": "test",
            "n_frames": len(test_frames),
            "seed": int(args.seed),
            "arms": stress_arms,
        }
        (out_dir / "summary_stress.json").write_text(
            json.dumps(summary_stress, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        _write_summary_csv(out_dir / "summary_stress.csv", stress_csv_rows)

    (out_dir / "run_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8", newline="\n")
    log("Completed M2 classical baseline run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
