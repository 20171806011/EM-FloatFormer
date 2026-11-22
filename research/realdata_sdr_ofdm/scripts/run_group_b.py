"""Run Group B (F9: Hard Selection vs. Posterior Mean Plug-in vs. Exact Mixture Likelihood).

Executes:
1. 32-frame smoke check on `dev.npz`.
2. Strictly dev-only calibration of `(beta, nonzero_om_penalty)` (`dev_tuning.json`).
3. Frozen `test.npz` (138 frames) evaluation across all 3 Group B arms (`hard_map`,
   `posterior_mean_plugin`, `exact_mixture_likelihood`) on an identical shared candidate
   branch bank + pinned profiler + paired audits vs M2 baselines and intra-group contrasts.
4. Frozen `test.npz` controllable stress suite (including paired stress comparisons under
   CFO/phase-drift/AWGN) + 16-frame deterministic replay audit.
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
    apply_stress_perturbation,
    load_processed_split,
)
from research.realdata_sdr_ofdm.eval.evaluator import (
    _frame_s2_scale,
    _is_valid_output,
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
from research.realdata_sdr_ofdm.receivers.group_b_fusion import (
    GROUP_B_MODES,
    GroupBConfig,
    calibrate_group_b_on_dev,
    run_frame,
)


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


def _finite_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if np.isfinite(f) else float(default)
    except (TypeError, ValueError):
        return float(default)


def _write_per_frame_jsonl(
    path: Path,
    pairs: list[tuple[RealOFDMFrame, ReceiverOutput]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for fr, out in pairs:
            aux = (getattr(out, "aux_info", None) or {}) if out is not None else {}
            is_timeout = bool(aux.get("timeout", False))
            ok, err_msg = _is_valid_output(fr, out)
            if is_timeout:
                ok = False
                err_msg = str(aux.get("error") or "timeout")

            if not ok:
                be = 0
                se = 0
                sym_mse = 0.0
                evm_pct = 0.0
                p_nmse = 0.0
                p_cal = 0.0
                p_nll = 0.0
                failed_flag = True
                error_val = str(err_msg or "invalid output")
            else:
                be = int(np.sum(out.hard_bits != fr.tx_bits))
                se = int(np.sum(np.any(out.hard_bits != fr.tx_bits, axis=-1)))
                sq_err = float(np.sum(np.abs(out.eq_symbols - fr.tx_symbols) ** 2))
                sq_sig = float(np.sum(np.abs(fr.tx_symbols) ** 2))
                sym_mse = _finite_float(sq_err / float(fr.tx_symbols.size))
                evm_pct = _finite_float(100.0 * float(np.sqrt(sq_err / max(sq_sig, 1e-30))))

                p_nmse = 0.0
                p_cal = 0.0
                p_nll = 0.0
                if out.heldout_pilot_pred is not None and out.heldout_pilot_var is not None:
                    idx = aux.get("heldout_pilot_indices") or []
                    p_pred = np.asarray(out.heldout_pilot_pred, dtype=np.complex128).ravel()
                    p_var = np.asarray(out.heldout_pilot_var, dtype=np.float64).ravel()
                    if (
                        len(idx) == len(p_pred)
                        and len(idx) == len(p_var)
                        and len(idx) > 0
                        and all(0 <= int(i) < len(PILOT_FFT_INDICES) for i in idx)
                    ):
                        int_idx = [int(i) for i in idx]
                        fft_idx = [PILOT_FFT_INDICES[i] for i in int_idx]
                        y_p = fr.rx_grid[PILOT_SYMBOL_IDX, fft_idx]
                        p_val = fr.pilot_values[int_idx]
                        z_obs = y_p / p_val
                        sig2_obs = float(fr.est_n0) / (np.abs(p_val) ** 2)
                        s2_f = _frame_s2_scale(fr)
                        v_eff = np.maximum(p_var + sig2_obs, max(1e-30 * s2_f, 1e-300))
                        err = p_pred - z_obs
                        p_nmse = _finite_float(
                            np.sum(np.abs(err) ** 2) / max(np.sum(np.abs(z_obs) ** 2), max(1e-30 * s2_f, 1e-300))
                        )
                        p_cal = _finite_float(np.mean((np.abs(err) ** 2) / v_eff))
                        p_nll = _finite_float(np.mean(np.log(np.pi * v_eff) + (np.abs(err) ** 2) / v_eff))
                failed_flag = False
                error_val = aux.get("error")

            nc_raw = aux.get("n_candidates")
            try:
                nc_val = 1 if (nc_raw is None or isinstance(nc_raw, bool)) else int(nc_raw)
            except (TypeError, ValueError):
                nc_val = 1
            om_raw = aux.get("omega_t_candidates")
            om_seq = [0.0] if om_raw is None else om_raw

            rec = {
                "frame_id": int(fr.frame_id),
                "split": str(fr.split),
                "group_id": str(fr.group_id),
                "reported_sinr_db": _finite_float(fr.reported_sinr_db),
                "est_n0": _finite_float(fr.est_n0),
                "bit_errors": be,
                "total_bits": int(fr.tx_bits.size),
                "ber": _finite_float(be / max(int(fr.tx_bits.size), 1)),
                "frame_error": 1 if failed_flag else int(be > 0),
                "sym_errors": se,
                "total_syms": int(fr.tx_symbols.size),
                "sym_mse": sym_mse,
                "evm_pct": evm_pct,
                "pilot_nmse": p_nmse,
                "pilot_cal_ratio": p_cal,
                "pilot_nll": p_nll,
                "n_candidates": nc_val,
                "branch_entropy": _finite_float(aux.get("branch_entropy", 0.0)),
                "between_var_mean": _finite_float(aux.get("between_var_mean", 0.0)),
                "omega_t_candidates": [_finite_float(x) for x in om_seq],
                "latency_ms": _finite_float(aux.get("latency_ms", 0.0)),
                "failed": failed_flag,
                "error": error_val,
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
    parser = argparse.ArgumentParser(description="Run M3 Group B posterior fusion experiments.")
    parser.add_argument("--proc-dir", type=Path, default=Path("/tmp/tw_realdata/data_proc"))
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/tw_realdata/outputs/group_b"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("/tmp/tw_realdata/outputs/baselines"))
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

    log(f"Starting M3 Group B: proc_dir={args.proc_dir}, out_dir={out_dir}")
    dev_frames = load_processed_split(args.proc_dir, "dev")
    test_frames = load_processed_split(args.proc_dir, "test")
    log(f"Loaded dev={len(dev_frames)}, test={len(test_frames)}")

    base_cfg = GroupBConfig()

    # 1. 32-frame smoke check on dev
    smoke_frames = dev_frames[: min(args.smoke_n, len(dev_frames))]
    smoke_results: dict[str, Any] = {}
    for m in GROUP_B_MODES:
        pairs = [(fr, run_frame(fr, mode=m, config=base_cfg)) for fr in smoke_frames]
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
    tuned_cfg, tune_details = calibrate_group_b_on_dev(
        dev_frames,
        betas=(0.25, 0.5, 1.0),
        nonzero_om_penalties=(2.0, 4.0, 8.0),
        n0_scales=(0.7,),
        base_config=base_cfg,
    )
    tune_time_s = time.perf_counter() - t_tune0

    dev_eval: dict[str, Any] = {}
    for m in GROUP_B_MODES:
        dev_eval[m] = {
            "config": tuned_cfg.to_dict(),
            "metrics": evaluate_receiver(
                [(fr, run_frame(fr, mode=m, config=tuned_cfg)) for fr in dev_frames]
            ),
        }
        log(
            f"Dev tuned {m}: BER={dev_eval[m]['metrics']['ber']:.5f}, "
            f"pilot_nmse={dev_eval[m]['metrics']['pilot_nmse']:.5f}, "
            f"pilot_cal={dev_eval[m]['metrics']['pilot_cal_ratio']:.4f}"
        )

    dev_tuning_payload = {
        "split_used": "dev",
        "n_dev_frames": len(dev_frames),
        "tune_time_s": float(tune_time_s),
        "smoke": smoke_results,
        "calibration_details": tune_details,
        "tuned": dev_eval,
    }
    (out_dir / "dev_tuning.json").write_text(
        json.dumps(dev_tuning_payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    # 3. Frozen test evaluation across 3 Group B arms + M2 anchors
    m2_cfgs = _load_m2_anchor_configs(args.baseline_dir)
    m2_pairs: dict[str, list[tuple[RealOFDMFrame, ReceiverOutput]]] = {
        m: [(fr, run_classical_frame(fr, method=m, config=m2_cfgs[m])) for fr in test_frames]
        for m in ("ls_linear_zf", "lmmse_1d", "lmmse_2d_tracked")
    }

    arm_pairs: dict[str, list[tuple[RealOFDMFrame, ReceiverOutput]]] = {}
    clean_arms: dict[str, Any] = {}
    clean_csv_rows: list[dict[str, Any]] = []

    for m in GROUP_B_MODES:
        pairs = [(fr, run_frame(fr, mode=m, config=tuned_cfg)) for fr in test_frames]
        arm_pairs[m] = pairs
        _write_per_frame_jsonl(out_dir / f"per_frame_{m}.jsonl", pairs)

        prof = profile_receiver_pinned(
            test_frames,
            lambda f, _m=m, _c=tuned_cfg: run_frame(f, mode=_m, config=_c),
            warmup_frames=8,
            param_count=0,
            train_time_s=0.0,
        )
        ev = evaluate_receiver(pairs, param_count=0, train_time_s=0.0)
        ev["mean_ms_per_frame"] = prof["mean_ms_per_frame"]
        ev["median_ms_per_frame"] = prof["median_ms_per_frame"]
        ev["p95_ms_per_frame"] = prof["p95_ms_per_frame"]
        ev["peak_rss_mb"] = prof["peak_rss_mb"]
        ev["tracemalloc_peak_mb"] = prof["tracemalloc_peak_mb"]
        ev["config"] = tuned_cfg.to_dict()
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

    # Deterministic 16-frame replay check
    replay_m = "exact_mixture_likelihood"
    max_llr_diff = 0.0
    max_h_diff = 0.0
    bit_mismatches = 0
    for fr, out1 in arm_pairs[replay_m][:16]:
        out2 = run_frame(fr, mode=replay_m, config=tuned_cfg)
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

    # Paired statistical comparisons
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
            {"lmmse_1d": m2_pairs["lmmse_1d"], **arm_pairs}, anchor_name="lmmse_1d", metric="ber"
        ),
        "subblock_g12_vs_lmmse_1d": compare_multiple_arms(
            {"lmmse_1d": m2_pairs["lmmse_1d"], **arm_pairs},
            anchor_name="lmmse_1d",
            metric="ber",
            group_fn=_sub12,
        ),
        "subblock_g16_vs_lmmse_1d": compare_multiple_arms(
            {"lmmse_1d": m2_pairs["lmmse_1d"], **arm_pairs},
            anchor_name="lmmse_1d",
            metric="ber",
            group_fn=_sub16,
        ),
        "macro_g3_vs_lmmse_2d_tracked": compare_multiple_arms(
            {"lmmse_2d_tracked": m2_pairs["lmmse_2d_tracked"], **arm_pairs},
            anchor_name="lmmse_2d_tracked",
            metric="ber",
        ),
        "subblock_g16_vs_lmmse_2d_tracked": compare_multiple_arms(
            {"lmmse_2d_tracked": m2_pairs["lmmse_2d_tracked"], **arm_pairs},
            anchor_name="lmmse_2d_tracked",
            metric="ber",
            group_fn=_sub16,
        ),
        "intra_group": {
            "plugin_vs_hard_map_g3": compare_arms(
                arm_pairs["hard_map"], arm_pairs["posterior_mean_plugin"], metric="ber"
            ),
            "plugin_vs_hard_map_g16": compare_arms(
                arm_pairs["hard_map"], arm_pairs["posterior_mean_plugin"], metric="ber", group_fn=_sub16
            ),
            "mixture_vs_hard_map_g3": compare_arms(
                arm_pairs["hard_map"], arm_pairs["exact_mixture_likelihood"], metric="ber"
            ),
            "mixture_vs_hard_map_g16": compare_arms(
                arm_pairs["hard_map"], arm_pairs["exact_mixture_likelihood"], metric="ber", group_fn=_sub16
            ),
            "mixture_vs_plugin_g3": compare_arms(
                arm_pairs["posterior_mean_plugin"], arm_pairs["exact_mixture_likelihood"], metric="ber"
            ),
            "mixture_vs_plugin_g16": compare_arms(
                arm_pairs["posterior_mean_plugin"],
                arm_pairs["exact_mixture_likelihood"],
                metric="ber",
                group_fn=_sub16,
            ),
        },
    }

    # Stress-condition intra-group paired contrasts (cfo_0.05 & drift_0.60)
    if not args.skip_stress:
        for cond_tag, cfo_v, drift_v in (("cfo_0.05", 0.05, 0.0), ("drift_0.60", 0.0, 0.6)):
            stressed = [
                apply_stress_perturbation(
                    fr, cfo_norm=cfo_v, phase_drift_rad=drift_v, seed=int(args.seed) + int(fr.frame_id)
                )
                for fr in test_frames
            ]
            st_pairs = {
                m: [(fr_s, run_frame(fr_s, mode=m, config=tuned_cfg)) for fr_s in stressed]
                for m in GROUP_B_MODES
            }
            paired_payload[f"stress_{cond_tag}_mixture_vs_hard_map_g3"] = compare_arms(
                st_pairs["hard_map"], st_pairs["exact_mixture_likelihood"], metric="ber"
            )
            paired_payload[f"stress_{cond_tag}_mixture_vs_hard_map_g16"] = compare_arms(
                st_pairs["hard_map"], st_pairs["exact_mixture_likelihood"], metric="ber", group_fn=_sub16
            )
            paired_payload[f"stress_{cond_tag}_mixture_vs_plugin_g16"] = compare_arms(
                st_pairs["posterior_mean_plugin"], st_pairs["exact_mixture_likelihood"], metric="ber", group_fn=_sub16
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
        },
    }
    (out_dir / "summary_clean.json").write_text(
        json.dumps(summary_clean, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_summary_csv(out_dir / "summary_clean.csv", clean_csv_rows)

    # 4. Stress suite
    if not args.skip_stress:
        stress_arms: dict[str, Any] = {}
        stress_csv_rows: list[dict[str, Any]] = []
        for m in GROUP_B_MODES:
            suite = run_stress_suite(
                test_frames,
                lambda f, _m=m, _c=tuned_cfg: run_frame(f, mode=_m, config=_c),
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
    log("Completed M3 Group B run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
