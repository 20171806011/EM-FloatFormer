#!/usr/bin/env python3
"""Full 7,096-frame physical-layer audit of ADALM-PLUTO QPSKlink dataset.

Uses ONLY the known 26-symbol double-Barker-13 header for blind frame synchronization,
timing phase recovery, CFO estimation, and single-tap LS equalization, then audits
against ground-truth Tx_data (offline only) to characterize:
  - Per-frame SNR, EVM, SER, sync lock rate, anomaly/corrupt frame rate
  - Time-series evolution of complex channel gain h(t) = |h(t)| exp(j*phi(t)) across all 7,096 frames
  - Residual fractional timing offset and multi-tap ISI energy fraction
  - Cross-frame autocorrelation and non-local state recurrence (critical for global content addressing!)
  - Natural phase jumps, amplitude fades, or regime shifts in the real hardware capture

Memory-safe: streams from memmap, peak RSS < 500 MB, single-threaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--out-npz", type=Path, required=True)
    args = ap.parse_args()

    t0 = time.perf_counter()
    tx_mm = np.load(args.work_dir / "tx_c64.npy", mmap_mode="r")
    rx_mm = np.load(args.work_dir / "rx_c64.npy", mmap_mode="r")
    n_frames, n_sym = tx_mm.shape
    _, n_samp = rx_mm.shape
    sps = 20
    frame_samp = n_sym * sps  # 2520

    # Known 26-symbol Barker header (constant across all 7096 frames, verified in inspect_mat.py)
    hdr = np.asarray(tx_mm[0, :26], dtype=np.complex128)
    hdr_conj_rev = np.conj(hdr[::-1])
    hdr_en = float(np.sum(np.abs(hdr) ** 2))
    # Upsampled matched template (26 taps spaced by SPS=20 -> length 25*20 + 1 = 501)
    # Using decimated-by-phase vectorized correlation is exact and super fast!
    # QPSK constellation (verified exact from Tx_data)
    qpsk_const = np.array([-0.707 - 0.707j, -0.707 + 0.707j, 0.707 - 0.707j, 0.707 + 0.707j], dtype=np.complex128)
    # Exact magnitude scaling of Tx_data symbols
    sym_scale = float(np.abs(tx_mm[0, 0]))
    qpsk_const = np.array([
        -1.0 - 1.0j, -1.0 + 1.0j, 1.0 - 1.0j, 1.0 + 1.0j
    ], dtype=np.complex128) * (sym_scale / np.sqrt(2.0))

    # Output arrays for all 7,096 frames
    sample_start = np.empty(n_frames, dtype=np.int32)
    sample_start_copy2 = np.empty(n_frames, dtype=np.int32)
    phase_mod20 = np.empty(n_frames, dtype=np.int16)
    frac_offset_est = np.empty(n_frames, dtype=np.float32)
    hdr_corr_peak = np.empty(n_frames, dtype=np.float32)
    hdr_corr_ratio2 = np.empty(n_frames, dtype=np.float32)
    rho_period = np.empty(n_frames, dtype=np.float32)
    cfo_rad_per_sym = np.empty(n_frames, dtype=np.float64)
    h_hdr_est = np.empty(n_frames, dtype=np.complex128)
    h_oracle_126 = np.empty(n_frames, dtype=np.complex128)
    h_copy2_126 = np.empty(n_frames, dtype=np.complex128)
    snr_db_oracle = np.empty(n_frames, dtype=np.float32)
    snr_db_hdr = np.empty(n_frames, dtype=np.float32)
    evm_pct_hdr_ls = np.empty(n_frames, dtype=np.float32)
    ser_hdr_ls = np.empty(n_frames, dtype=np.float32)
    isi_energy_ratio = np.empty(n_frames, dtype=np.float32)
    sync_exact_match = np.empty(n_frames, dtype=np.bool_)

    # Also save extracted symbol-rate complex frames (copy 1 & copy 2) for downstream experiments?
    # Wait: let's keep raw waveforms intact and save audited symbol-rate diagnostic arrays separately!
    sym_copy1 = np.empty((n_frames, n_sym), dtype=np.complex64)
    sym_copy2 = np.empty((n_frames, n_sym), dtype=np.complex64)

    ones26 = np.ones(26, dtype=np.float64)
    n_idx = np.arange(n_samp, dtype=np.float64)

    # Search range for first full frame start: [0, 2520) samples guarantees BOTH copy 1 [s, s+2520)
    # and copy 2 [s+2520, s+5040) fit completely inside [0, 7560)!
    # Wait: max start for 2 full copies is 7560 - 5040 = 2520, so s in [0, 2520) ALWAYS gives 2 full copies!
    # Let's verify: if s in [0, 2520), last sample of copy 2 is s + 2520 + 125*20 = s + 5020 < 7540 < 7560!
    # That means we only need to search the first 126 symbols (indices 0..125) per phase p in 0..19 -> 2520 lags!
    for fidx in range(n_frames):
        r = np.asarray(rx_mm[fidx], dtype=np.complex128)
        s_true = np.asarray(tx_mm[fidx], dtype=np.complex128)

        # 1. Blind lag-2520 periodic autocorrelation -> residual CFO estimate (rad/sample)
        a = r[:-frame_samp]
        b = r[frame_samp:]
        c_per = np.vdot(a, b)
        en_a = float(np.sum(a.real ** 2 + a.imag ** 2))
        en_b = float(np.sum(b.real ** 2 + b.imag ** 2))
        rho = float(np.abs(c_per) / np.sqrt(en_a * en_b + 1e-30))
        omega_samp = float(np.angle(c_per)) / float(frame_samp)
        rho_period[fidx] = rho
        cfo_rad_per_sym[fidx] = omega_samp * sps

        # Derotate by blind lag-2520 CFO estimate
        r_derot = r * np.exp(-1j * omega_samp * n_idx)

        # 2. Blind double-copy coherent header correlation over s in [0, 2520)
        # Folding the two periods [0:5040] -> coherently average copy 1 & copy 2?
        # Even better: evaluate single-copy header correlation on first 126 symbol offsets for all 20 phases
        # r_dec shape: (20, 378)
        r_dec = r_derot.reshape(378, sps).T  # (20, 378)
        # Need windows of length 26 starting at m in 0..125 (needs symbols 0..150)
        # Vectorized across all 20 phases simultaneously!
        # Let's also average with copy 2 at m+126 for ultra-clean blind sync check, AND measure single-copy header corr!
        r_win = r_dec[:, :126 + 25]  # (20, 151)
        # Sliding dot product with hdr (26) for each phase:
        # Build stride view: shape (20, 126, 26)
        from numpy.lib.stride_tricks import sliding_window_view
        sw1 = sliding_window_view(r_dec[:, :126 + 25], 26, axis=1)  # (20, 126, 26)
        sw2 = sliding_window_view(r_dec[:, 126:252 + 25], 26, axis=1)  # (20, 126, 26)
        # Inner products
        ip1 = sw1 @ np.conj(hdr)  # (20, 126)
        ip2 = sw2 @ np.conj(hdr)  # (20, 126)
        en1 = np.sum(sw1.real ** 2 + sw1.imag ** 2, axis=2)  # (20, 126)
        en2 = np.sum(sw2.real ** 2 + sw2.imag ** 2, axis=2)  # (20, 126)
        norm1 = np.abs(ip1) / np.sqrt(en1 * hdr_en + 1e-30)
        # Coherent two-copy normalized correlation (still 100% blind to payload, uses only Barker header!)
        norm12 = np.abs(ip1 + ip2) / np.sqrt((en1 + en2) * 2.0 * hdr_en + 1e-30)

        # Flatten in sample order: lag = m * 20 + p -> shape (126, 20)
        score_samp = norm12.T.reshape(frame_samp)
        score_single = norm1.T.reshape(frame_samp)
        best_s = int(np.argmax(score_samp))
        best_m = best_s // sps
        best_p = best_s % sps
        peak_val = float(score_samp[best_s])

        # Parabolic sub-sample offset around best_s (cyclic mod 2520)
        y_l = float(score_samp[(best_s - 1) % frame_samp])
        y_0 = peak_val
        y_r = float(score_samp[(best_s + 1) % frame_samp])
        denom = (y_l - 2.0 * y_0 + y_r)
        frac = 0.5 * (y_l - y_r) / denom if abs(denom) > 1e-12 else 0.0

        # Peak-to-sidelobe ratio (exclude +/- 40 samples around best_s)
        mask = np.ones(frame_samp, dtype=np.bool_)
        for d in range(-40, 41):
            mask[(best_s + d) % frame_samp] = False
        second_peak = float(np.max(score_samp[mask]))

        sample_start[fidx] = best_s
        sample_start_copy2[fidx] = best_s + frame_samp
        phase_mod20[fidx] = best_p
        frac_offset_est[fidx] = frac
        hdr_corr_peak[fidx] = float(score_single[best_s])
        hdr_corr_ratio2[fidx] = peak_val / max(second_peak, 1e-12)

        # Extract copy 1 and copy 2 at symbol rate
        y1 = r_dec[best_p, best_m:best_m + n_sym]
        y2 = r_dec[best_p, best_m + n_sym:best_m + 2 * n_sym]
        sym_copy1[fidx] = y1.astype(np.complex64)
        sym_copy2[fidx] = y2.astype(np.complex64)

        # Basic single-frame LS from copy-1 header ONLY (26 symbols)
        h_ls = np.vdot(hdr, y1[:26]) / hdr_en
        h_hdr_est[fidx] = h_ls
        res_hdr = y1[:26] - h_ls * hdr
        sig_hdr = float(np.mean(np.abs(h_ls * hdr) ** 2))
        nse_hdr = float(np.mean(np.abs(res_hdr) ** 2))
        snr_db_hdr[fidx] = float(10.0 * np.log10(sig_hdr / max(nse_hdr, 1e-30)))

        # Offline ground-truth audit (payload + full 126-sym oracle channel)
        s_pay = s_true[26:]
        eq_pay = y1[26:] / h_ls
        evm_pct_hdr_ls[fidx] = float(100.0 * np.sqrt(np.mean(np.abs(eq_pay - s_pay) ** 2) / np.mean(np.abs(s_pay) ** 2)))
        dec_idx = np.argmin(np.abs(eq_pay[:, None] - qpsk_const[None, :]), axis=1)
        dec_sym = qpsk_const[dec_idx]
        ser_hdr_ls[fidx] = float(np.mean(np.abs(dec_sym - s_pay) > 1e-3))

        # Oracle full-126 LS channel & SNR on copy 1 and copy 2
        s_en = float(np.sum(np.abs(s_true) ** 2))
        h1_ora = np.vdot(s_true, y1) / s_en
        h2_ora = np.vdot(s_true, y2) / s_en
        h_oracle_126[fidx] = h1_ora
        h_copy2_126[fidx] = h2_ora
        res_ora = y1 - h1_ora * s_true
        sig_ora = float(np.mean(np.abs(h1_ora * s_true) ** 2))
        nse_ora = float(np.mean(np.abs(res_ora) ** 2))
        snr_db_oracle[fidx] = float(10.0 * np.log10(sig_ora / max(nse_ora, 1e-30)))

        # Verify sync lock against full-126 correlation at this phase
        corr126_here = abs(np.vdot(s_true, y1)) / np.sqrt(float(np.sum(np.abs(y1) ** 2)) * s_en + 1e-30)
        sync_exact_match[fidx] = corr126_here > 0.85

        # 5-tap ISI channel check (taps -2, -1, 0, +1, +2) on interior symbols 2..123
        # Solve least squares for taps [-2..+2]
        X = np.column_stack([s_true[2 - k:n_sym - 2 - k] for k in (-2, -1, 0, 1, 2)])
        y_int = y1[2:n_sym - 2]
        taps, _, _, _ = np.linalg.lstsq(X, y_int, rcond=None)
        e_main = float(np.abs(taps[2]) ** 2)
        e_isi = float(np.sum(np.abs(taps[[0, 1, 3, 4]]) ** 2))
        isi_energy_ratio[fidx] = e_isi / max(e_main + e_isi, 1e-30)

    # Save symbol-rate extracted dataset & audit diagnostics to NPZ (tiny: ~15 MB!)
    args.out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_npz,
        sym_copy1=sym_copy1,
        sym_copy2=sym_copy2,
        sample_start=sample_start,
        sample_start_copy2=sample_start_copy2,
        phase_mod20=phase_mod20,
        frac_offset_est=frac_offset_est,
        hdr_corr_peak=hdr_corr_peak,
        hdr_corr_ratio2=hdr_corr_ratio2,
        rho_period=rho_period,
        cfo_rad_per_sym=cfo_rad_per_sym,
        h_hdr_est=h_hdr_est,
        h_oracle_126=h_oracle_126,
        h_copy2_126=h_copy2_126,
        snr_db_oracle=snr_db_oracle,
        snr_db_hdr=snr_db_hdr,
        evm_pct_hdr_ls=evm_pct_hdr_ls,
        ser_hdr_ls=ser_hdr_ls,
        isi_energy_ratio=isi_energy_ratio,
        sync_exact_match=sync_exact_match,
    )

    # Compute channel temporal statistics & cross-frame recurrence!
    h_mag = np.abs(h_oracle_126)
    h_ang = np.angle(h_oracle_126)
    dh_ang = np.angle(h_oracle_126[1:] * np.conj(h_oracle_126[:-1]))
    dh_rel = np.abs(h_oracle_126[1:] - h_oracle_126[:-1]) / (h_mag[:-1] + 1e-12)

    # Lag correlations of complex channel h(t) (normalized)
    h_norm = h_oracle_126 / (np.sqrt(np.mean(h_mag ** 2)) + 1e-30)
    lag_corrs = {}
    for lag in [1, 2, 5, 10, 20, 50, 100, 250, 500, 1000, 2000]:
        c = np.vdot(h_norm[:-lag], h_norm[lag:]) / float(n_frames - lag)
        # Also magnitude correlation
        mc = float(np.corrcoef(h_mag[:-lag], h_mag[lag:])[0, 1])
        lag_corrs[str(lag)] = {
            "complex_corr_abs": float(np.abs(c)),
            "complex_corr_real": float(c.real),
            "mag_corr": mc,
            "mean_rel_diff": float(np.mean(np.abs(h_oracle_126[lag:] - h_oracle_126[:-lag]) / (h_mag[:-lag] + 1e-12))),
        }

    # Check non-local recurrence: for every frame t in [500..7095], compare nearest-past frame vs best frame in past library
    # sampled every 10 frames for speed
    sample_ts = np.arange(500, n_frames, 10)
    d_prev1 = []
    d_best_far = []
    best_far_lags = []
    for t in sample_ts:
        ht = h_oracle_126[t]
        d1 = abs(h_oracle_126[t - 1] - ht) / abs(ht)
        # Far past (> 50 frames ago)
        dists_far = np.abs(h_oracle_126[:t - 50] - ht) / abs(ht)
        j_best = int(np.argmin(dists_far))
        d_prev1.append(float(d1))
        d_best_far.append(float(dists_far[j_best]))
        best_far_lags.append(int(t - j_best))

    # Jump statistics: how often does |angle(h[t] conj(h[t-1]))| exceed 0.3, 0.5, 1.0, pi/2 rad?
    abs_dphi = np.abs(dh_ang)
    # Also check block-wise summary across 10 contiguous time blocks of ~710 frames
    block_summaries = []
    edges = np.linspace(0, n_frames, 11, dtype=int)
    for bi in range(10):
        a_i, b_i = int(edges[bi]), int(edges[bi + 1])
        block_summaries.append({
            "block": bi,
            "start": a_i,
            "end": b_i,
            "n_frames": b_i - a_i,
            "sync_lock_rate": float(np.mean(sync_exact_match[a_i:b_i])),
            "mean_snr_db": float(np.mean(snr_db_oracle[a_i:b_i])),
            "min_snr_db": float(np.min(snr_db_oracle[a_i:b_i])),
            "mean_evm_pct": float(np.mean(evm_pct_hdr_ls[a_i:b_i])),
            "mean_ser_hdr_ls": float(np.mean(ser_hdr_ls[a_i:b_i])),
            "fer_hdr_ls": float(np.mean(ser_hdr_ls[a_i:b_i] > 0)),
            "h_mag_mean": float(np.mean(h_mag[a_i:b_i])),
            "h_mag_std": float(np.std(h_mag[a_i:b_i])),
            "h_ang_circ_r": float(np.abs(np.mean(np.exp(1j * h_ang[a_i:b_i])))),
            "h_ang_mean_rad": float(np.angle(np.mean(np.exp(1j * h_ang[a_i:b_i])))),
            "phase_jumps_gt_0p5rad": int(np.sum(np.abs(np.angle(h_oracle_126[a_i + 1:b_i] * np.conj(h_oracle_126[a_i:b_i - 1]))) > 0.5)),
        })

    # Phase-mod-20 distribution across all 7096 frames
    p_vals, p_cnts = np.unique(phase_mod20, return_counts=True)

    # Check copy1 vs copy2 channel phase difference (measures intra-capture CFO / phase drift)
    dphi_c1_c2 = np.angle(h_copy2_126 * np.conj(h_oracle_126))

    summary: dict[str, Any] = {
        "n_frames": int(n_frames),
        "sync_lock_count": int(np.sum(sync_exact_match)),
        "sync_lock_rate": float(np.mean(sync_exact_match)),
        "failed_sync_frames": np.where(~sync_exact_match)[0].tolist()[:50],
        "hdr_corr_peak_quantiles": {q: float(np.quantile(hdr_corr_peak, float(q))) for q in ["0.0", "0.01", "0.05", "0.5", "0.95", "1.0"]},
        "hdr_corr_ratio2_quantiles": {q: float(np.quantile(hdr_corr_ratio2, float(q))) for q in ["0.0", "0.01", "0.05", "0.5", "0.95", "1.0"]},
        "rho_period_quantiles": {q: float(np.quantile(rho_period, float(q))) for q in ["0.0", "0.01", "0.05", "0.5", "0.95", "1.0"]},
        "cfo_rad_per_sym_quantiles": {q: float(np.quantile(cfo_rad_per_sym, float(q))) for q in ["0.0", "0.01", "0.05", "0.5", "0.95", "0.99", "1.0"]},
        "snr_db_oracle_quantiles": {q: float(np.quantile(snr_db_oracle, float(q))) for q in ["0.0", "0.01", "0.05", "0.25", "0.5", "0.75", "0.95", "0.99", "1.0"]},
        "evm_pct_hdr_ls_quantiles": {q: float(np.quantile(evm_pct_hdr_ls, float(q))) for q in ["0.0", "0.05", "0.5", "0.95", "0.99", "1.0"]},
        "ser_hdr_ls_mean": float(np.mean(ser_hdr_ls)),
        "fer_hdr_ls_mean": float(np.mean(ser_hdr_ls > 0)),
        "nonzero_ser_frame_count": int(np.sum(ser_hdr_ls > 0)),
        "isi_energy_ratio_quantiles": {q: float(np.quantile(isi_energy_ratio, float(q))) for q in ["0.0", "0.25", "0.5", "0.75", "0.95", "0.99", "1.0"]},
        "phase_mod20_counts": {int(k): int(v) for k, v in zip(p_vals, p_cnts)},
        "frac_offset_quantiles": {q: float(np.quantile(frac_offset_est, float(q))) for q in ["0.05", "0.25", "0.5", "0.75", "0.95"]},
        "h_mag_quantiles": {q: float(np.quantile(h_mag, float(q))) for q in ["0.0", "0.01", "0.05", "0.25", "0.5", "0.75", "0.95", "0.99", "1.0"]},
        "h_ang_quantiles": {q: float(np.quantile(h_ang, float(q))) for q in ["0.0", "0.1", "0.25", "0.5", "0.75", "0.9", "1.0"]},
        "step_dphi_abs_quantiles": {q: float(np.quantile(abs_dphi, float(q))) for q in ["0.5", "0.75", "0.9", "0.95", "0.99", "0.999", "1.0"]},
        "step_dphi_jump_counts": {
            "gt_0.2rad": int(np.sum(abs_dphi > 0.2)),
            "gt_0.5rad": int(np.sum(abs_dphi > 0.5)),
            "gt_1.0rad": int(np.sum(abs_dphi > 1.0)),
            "gt_1.5rad": int(np.sum(abs_dphi > 1.5)),
        },
        "step_rel_dh_quantiles": {q: float(np.quantile(dh_rel, float(q))) for q in ["0.5", "0.9", "0.95", "0.99", "1.0"]},
        "copy1_to_copy2_dphi_std_rad": float(np.std(dphi_c1_c2)),
        "copy1_to_copy2_dphi_max_rad": float(np.max(np.abs(dphi_c1_c2))),
        "lag_correlations": lag_corrs,
        "nonlocal_recurrence_vs_prev1": {
            "prev1_median_rel_err": float(np.median(d_prev1)),
            "prev1_p95_rel_err": float(np.quantile(d_prev1, 0.95)),
            "best_far_gt50_median_rel_err": float(np.median(d_best_far)),
            "best_far_gt50_p95_rel_err": float(np.quantile(d_best_far, 0.95)),
            "frac_far_better_than_prev1": float(np.mean(np.array(d_best_far) < np.array(d_prev1))),
        },
        "ten_time_blocks": block_summaries,
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "peak_rss_mb": round(peak_rss_mb(), 2),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
