#!/usr/bin/env python3
"""Prepare real-waveform benchmark splits across raw_original and all 7 controllable stress conditions.

Directly injects physical perturbations onto the raw 7,560-sample ADALM-PLUTO complex waveforms
(from /tmp/agy_cli2_pluto/work/rx_c64.npy), executes the blind waveform receiver frontend
(matched correlation, timing recovery, fractional Catmull-Rom cubic resampling, frame alignment),
and saves compact symbol-rate benchmark bundles + frontend sync statistics for:
  - train (warmup [0:200] + train [200:3200], raw + multi-condition training pool)
  - dev   (warmup [3300:3500] + eval [3500:4500], all 8 conditions)
  - test  (warmup [4600:4800] + eval [4800:7040], all 8 conditions)

Strictly single-threaded and memory-safe (<500 MB peak RSS).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from waveform_frontend import (
    FRAME_SAMP,
    FULL_HDR_26,
    N_SYM,
    RX_SAMP,
    SPS,
    STRESS_SPECS,
    PerturbationSpec,
    cubic_catmull_rom_sample,
    inject_waveform_perturbation,
)

# Frozen non-overlapping temporal splits aligned to 20-frame hardware bursts with 100-frame guard bands
SPLITS = {
    "train": {"warmup_start": 0, "eval_start": 200, "eval_end": 3200},
    "dev": {"warmup_start": 3300, "eval_start": 3500, "eval_end": 4500},
    "test": {"warmup_start": 4600, "eval_start": 4800, "eval_end": 7040},
}

_HDR21 = FULL_HDR_26[:21]
_HDR21_CONJ = np.conj(_HDR21)
_HDR21_EN = float(np.sum(np.abs(_HDR21) ** 2))
_N_IDX = np.arange(RX_SAMP, dtype=np.float64)


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def fast_blind_frontend(
    r_wav: np.ndarray,
    single_shot_cfo_mode: bool = False,
) -> dict[str, Any]:
    """Fast vectorized blind waveform frontend (exact match to waveform_frontend.blind_waveform_frontend).

    When `single_shot_cfo_mode=True` (used for residual-CFO stress tests to model single-frame
    transmission without a repeated second copy for lag-2520 CFO cancellation), the frontend
    performs blind frame/timing synchronization using non-coherent segmented header correlation
    and leaves the intra-frame residual CFO intact for downstream carrier tracking / B-group mixture!
    """
    t0 = time.perf_counter()
    r = np.asarray(r_wav, dtype=np.complex128)

    if not single_shot_cfo_mode:
        # Standard periodic lag-2520 coarse CFO + two-copy coherent 21-sym header sync
        a = r[:-FRAME_SAMP]
        b = r[FRAME_SAMP:]
        c_per = np.vdot(a, b)
        en_a = float(np.sum(a.real ** 2 + a.imag ** 2))
        en_b = float(np.sum(b.real ** 2 + b.imag ** 2))
        rho_per = float(np.abs(c_per) / np.sqrt(en_a * en_b + 1e-30))
        omega_samp = float(np.angle(c_per)) / float(FRAME_SAMP)
        r_work = r * np.exp(-1j * omega_samp * _N_IDX)
        cfo_est = omega_samp * SPS

        r_dec = r_work.reshape(378, SPS).T  # (20, 378)
        sw1 = sliding_window_view(r_dec[:, :126 + 20], 21, axis=1)
        sw2 = sliding_window_view(r_dec[:, 126:252 + 20], 21, axis=1)
        ip1 = sw1 @ _HDR21_CONJ
        ip2 = sw2 @ _HDR21_CONJ
        en1 = np.sum(sw1.real ** 2 + sw1.imag ** 2, axis=2)
        en2 = np.sum(sw2.real ** 2 + sw2.imag ** 2, axis=2)
        norm21 = np.abs(ip1 + ip2) / np.sqrt((en1 + en2) * 2.0 * _HDR21_EN + 1e-30)
        score_samp = norm21.T.reshape(FRAME_SAMP)
    else:
        # Single-shot residual CFO mode: coarse sync via short-segment (3x7) non-coherent 21-sym correlation
        # that tolerates up to +-0.05 rad/sym CFO without using Copy-2 periodic cancellation!
        rho_per = 0.0
        cfo_est = 0.0
        r_work = r
        r_dec = r_work.reshape(378, SPS).T  # (20, 378)
        sw1 = sliding_window_view(r_dec[:, :126 + 20], 21, axis=1)  # (20, 126, 21)
        # Split 21-sym template into 3 sub-blocks of 7 symbols (coherent within 7 syms, non-coherent across blocks)
        c0 = sw1[:, :, 0:7] @ _HDR21_CONJ[0:7]
        c1 = sw1[:, :, 7:14] @ _HDR21_CONJ[7:14]
        c2 = sw1[:, :, 14:21] @ _HDR21_CONJ[14:21]
        en1 = np.sum(sw1.real ** 2 + sw1.imag ** 2, axis=2)
        norm21 = (np.abs(c0) + np.abs(c1) + np.abs(c2)) / np.sqrt(en1 * _HDR21_EN + 1e-30)
        score_samp = norm21.T.reshape(FRAME_SAMP)

    idx_s = int(np.argmax(score_samp))
    pk = float(score_samp[idx_s])
    yl = float(score_samp[(idx_s - 1) % FRAME_SAMP])
    yr = float(score_samp[(idx_s + 1) % FRAME_SAMP])
    denom = yl - 2.0 * pk + yr
    frac = float(np.clip(0.5 * (yl - yr) / denom, -0.5, 0.5)) if abs(denom) > 1e-12 else 0.0
    s_exact = float(idx_s) + frac

    pos1 = s_exact + np.arange(N_SYM, dtype=np.float64) * float(SPS)
    y1 = cubic_catmull_rom_sample(r_work, pos1)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "y_sym": y1,
        "sample_start_int": idx_s,
        "frac_offset": frac,
        "sync_corr_peak": pk,
        "rho_period": rho_per,
        "cfo_est": cfo_est,
        "frontend_ms": elapsed_ms,
    }


def process_split_condition(
    rx_mm: np.ndarray,
    tx_mm: np.ndarray,
    split_name: str,
    spec: PerturbationSpec,
    out_dir: Path,
) -> dict[str, Any]:
    sp = SPLITS[split_name]
    w_start = sp["warmup_start"]
    e_start = sp["eval_start"]
    e_end = sp["eval_end"]
    n_tot = e_end - w_start
    n_eval = e_end - e_start

    # Deterministic per-(condition, split) RNG seed (strictly independent across splits!)
    split_offset = {"train": 1000, "dev": 2000, "test": 3000}[split_name]
    rng = np.random.default_rng(spec.seed + split_offset)
    single_shot_cfo = spec.cfo_mag_range is not None

    frame_indices = np.arange(w_start, e_end, dtype=np.int32)
    y_sym_all = np.empty((n_tot, N_SYM), dtype=np.complex64)
    h_ora_all = np.empty(n_tot, dtype=np.complex128)
    d_ora_all = np.empty(n_tot, dtype=np.complex128)
    sync_ok_all = np.empty(n_tot, dtype=np.bool_)
    sample_start_all = np.empty(n_tot, dtype=np.int32)
    sample_start_ref_all = np.empty(n_tot, dtype=np.int32)
    sync_peak_all = np.empty(n_tot, dtype=np.float32)
    frontend_ms_all = np.empty(n_tot, dtype=np.float32)
    inj_cfo_all = np.empty(n_tot, dtype=np.float64)
    inj_jump_all = np.empty(n_tot, dtype=np.float64)
    inj_sigma_all = np.empty(n_tot, dtype=np.float64)

    # Pre-load clean reference sample_start from audit_full_7096 (for offline sync_ok audit ONLY!)
    audit_npz = np.load(out_dir.parent / "extracted_symbols_7096.npz")
    ref_starts = audit_npz["sample_start"]
    ref_y1 = audit_npz["sym_copy1"]

    t0 = time.perf_counter()
    cum_jump = 0.0
    for i, fidx in enumerate(frame_indices):
        r_raw = np.asarray(rx_mm[fidx], dtype=np.complex128)
        s_true = np.asarray(tx_mm[fidx], dtype=np.complex128)
        st_ref = int(ref_starts[fidx])
        sample_start_ref_all[i] = st_ref

        if spec.name == "raw_original":
            r_pert = r_raw
            meta = {
                "injected_cfo_rad_per_sym": 0.0,
                "injected_phase_jump_rad": 0.0,
                "injected_noise_sigma": 0.0,
            }
        else:
            r_pert, meta = inject_waveform_perturbation(r_raw, int(fidx), spec, rng, cum_jump_phase=cum_jump)
            cum_jump = float((cum_jump + meta["injected_phase_jump_rad"] + np.pi) % (2.0 * np.pi) - np.pi)

        fe = fast_blind_frontend(r_pert, single_shot_cfo_mode=single_shot_cfo)
        y1 = fe["y_sym"]
        y_sym_all[i] = y1.astype(np.complex64)
        sample_start_all[i] = fe["sample_start_int"]
        sync_peak_all[i] = fe["sync_corr_peak"]
        frontend_ms_all[i] = fe["frontend_ms"]
        inj_cfo_all[i] = meta["injected_cfo_rad_per_sym"]
        inj_jump_all[i] = meta["injected_phase_jump_rad"]
        inj_sigma_all[i] = meta["injected_noise_sigma"]

        # Offline sync audit: true start is cyclic mod 2520; within +-8 samples (< 0.4 sym) = locked!
        diff_s = abs(int(fe["sample_start_int"]) - st_ref)
        diff_s = min(diff_s, FRAME_SAMP - diff_s)
        sync_ok_all[i] = diff_s <= 8

        # Offline oracle clean channel (h_ora, d_ora) computed on the CLEAN underlying waveform
        # rotated by exact cum_jump and exact injected CFO evaluated at pilot center k=16.5!
        # Wait: compute clean oracle directly from clean ref_y1 * exp(j*cum_jump) + injected CFO at pilot center
        y_clean_rot = np.asarray(ref_y1[fidx], dtype=np.complex128) * np.exp(1j * cum_jump)
        if abs(meta["injected_cfo_rad_per_sym"]) > 0.0:
            # Exact phase of injected CFO at sample position pos1[k] = st_ref + 20*k
            # At pilot center k_ref = 16.5:
            phi_center = (meta["injected_cfo_rad_per_sym"] / float(SPS)) * (float(st_ref) + 16.5 * float(SPS))
            y_clean_rot = y_clean_rot * np.exp(1j * phi_center)
        X_ora = np.column_stack([s_true, np.ones(N_SYM)])
        w_ora, _, _, _ = np.linalg.lstsq(X_ora, y_clean_rot, rcond=None)
        h_ora_all[i] = w_ora[0]
        d_ora_all[i] = w_ora[1]

    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{spec.name}_{split_name}.npz"
    np.savez_compressed(
        npz_path,
        frame_indices=frame_indices,
        eval_offset=np.int32(e_start - w_start),
        y_sym=y_sym_all,
        h_ora=h_ora_all,
        d_ora=d_ora_all,
        sync_ok=sync_ok_all,
        sample_start=sample_start_all,
        sample_start_ref=sample_start_ref_all,
        sync_peak=sync_peak_all,
        frontend_ms=frontend_ms_all,
        inj_cfo=inj_cfo_all,
        inj_jump=inj_jump_all,
        inj_sigma=inj_sigma_all,
    )

    ev_sl = slice(e_start - w_start, n_tot)
    stat = {
        "condition": spec.name,
        "split": split_name,
        "n_total": int(n_tot),
        "n_eval": int(n_eval),
        "single_shot_cfo_mode": bool(single_shot_cfo),
        "eval_sync_ok_rate": float(np.mean(sync_ok_all[ev_sl])),
        "eval_sync_fail_count": int(np.sum(~sync_ok_all[ev_sl])),
        "frontend_median_ms": float(np.median(frontend_ms_all[ev_sl])),
        "frontend_mean_ms": float(np.mean(frontend_ms_all[ev_sl])),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
        "peak_rss_mb": round(peak_rss_mb(), 2),
    }
    print(json.dumps(stat))
    return stat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out-summary", type=Path, required=True)
    args = ap.parse_args()

    rx_mm = np.load(args.work_dir / "rx_c64.npy", mmap_mode="r")
    tx_mm = np.load(args.work_dir / "tx_c64.npy", mmap_mode="r")
    bench_dir = args.work_dir / "benchmarks"

    # Conditions to prepare:
    # - train: raw_original + awgn_snr6db + cfo_0p020_snr6db + phase_jump_snr6db (for clean & multi-condition training!)
    # - dev & test: ALL 8 conditions!
    plan = [
        ("train", "raw_original"),
        ("train", "awgn_snr6db"),
        ("train", "awgn_snr2db"),
        ("train", "cfo_0p020_snr6db"),
        ("train", "phase_jump_snr6db"),
    ]
    for sp in ("dev", "test"):
        for cname in STRESS_SPECS.keys():
            plan.append((sp, cname))

    t0 = time.perf_counter()
    stats = []
    for sp, cname in plan:
        stats.append(process_split_condition(rx_mm, tx_mm, sp, STRESS_SPECS[cname], bench_dir))

    summary = {
        "n_bundles": len(stats),
        "total_elapsed_sec": round(time.perf_counter() - t0, 2),
        "peak_rss_mb": round(peak_rss_mb(), 2),
        "bundles": stats,
    }
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"DONE preparing {len(stats)} real-waveform benchmark bundles in {summary['total_elapsed_sec']}s, peak RSS {summary['peak_rss_mb']} MB")


if __name__ == "__main__":
    main()
