#!/usr/bin/env python3
"""Deep first-principles structural and signal-processing audit of QPSKlink.mat.

Memory-safe (<2.5 GB peak RSS):
- Supports both MATLAB v7.3 (HDF5 via h5py) and MATLAB v5/v7 (via scipy.io).
- Converts Rx_data and Tx_data to float32/complex64 numpy memmaps on disk if desired,
  so all downstream experiments stream frames with minimal RAM (<300 MB).
- Audits variable names, shapes, dtypes, NaN/Inf, Barker header, QPSK constellation,
  oversampling rate (SPS), periodic frame repetition, timing phase, coarse/fine CFO,
  matched-filter correlation, and cross-frame channel continuity on a dev probe slice.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np
from scipy.io import loadmat, whosmat


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def inspect_hdf5(mat_path: Path, out_dir: Path, convert_memmap: bool) -> dict[str, Any]:
    import h5py

    info: dict[str, Any] = {"format": "matlab_v73_hdf5", "variables": {}}
    with h5py.File(mat_path, "r") as f:
        for k in f.keys():
            obj = f[k]
            if isinstance(obj, h5py.Dataset):
                info["variables"][k] = {
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "chunks": list(obj.chunks) if obj.chunks else None,
                    "compression": obj.compression,
                }
            else:
                info["variables"][k] = {"type": str(type(obj))}

        # MATLAB HDF5 stores arrays transposed relative to MATLAB column-major:
        # MATLAB (126, 7096) -> HDF5 (7096, 126); MATLAB (7560, 7096) -> HDF5 (7096, 7560)
        tx_ds = f["Tx_data"]
        rx_ds = f["Rx_data"]

        def _to_complex(arr: np.ndarray) -> np.ndarray:
            if np.iscomplexobj(arr):
                return arr.astype(np.complex64)
            if arr.dtype.names and "real" in arr.dtype.names and "imag" in arr.dtype.names:
                return (arr["real"].astype(np.float32) + 1j * arr["imag"].astype(np.float32)).astype(np.complex64)
            raise ValueError(f"Unexpected dtype for complex array: {arr.dtype}")

        # Determine orientation: frame axis has length 7096 (or largest common dim)
        tx_shape = tx_ds.shape
        rx_shape = rx_ds.shape
        if tx_shape[0] == rx_shape[0]:
            n_frames = tx_shape[0]
            tx_len = tx_shape[1]
            rx_len = rx_shape[1]
            frame_axis = 0
        elif tx_shape[1] == rx_shape[1]:
            n_frames = tx_shape[1]
            tx_len = tx_shape[0]
            rx_len = rx_shape[0]
            frame_axis = 1
        else:
            raise RuntimeError(f"Cannot infer common frame axis from Tx {tx_shape}, Rx {rx_shape}")

        info["inferred_layout"] = {
            "n_frames": int(n_frames),
            "tx_symbols_per_frame": int(tx_len),
            "rx_samples_per_frame": int(rx_len),
            "hdf5_frame_axis": int(frame_axis),
        }

        if convert_memmap:
            out_dir.mkdir(parents=True, exist_ok=True)
            tx_mm = np.lib.format.open_memmap(out_dir / "tx_c64.npy", mode="w+", dtype=np.complex64, shape=(n_frames, tx_len))
            rx_mm = np.lib.format.open_memmap(out_dir / "rx_c64.npy", mode="w+", dtype=np.complex64, shape=(n_frames, rx_len))
            batch = 256
            for i in range(0, n_frames, batch):
                j = min(i + batch, n_frames)
                if frame_axis == 0:
                    tx_chunk = _to_complex(tx_ds[i:j, :])
                    rx_chunk = _to_complex(rx_ds[i:j, :])
                else:
                    tx_chunk = _to_complex(tx_ds[:, i:j]).T
                    rx_chunk = _to_complex(rx_ds[:, i:j]).T
                tx_mm[i:j] = tx_chunk
                rx_mm[i:j] = rx_chunk
            tx_mm.flush()
            rx_mm.flush()
            del tx_mm, rx_mm
    return info


def inspect_mat_v5(mat_path: Path, out_dir: Path, convert_memmap: bool) -> dict[str, Any]:
    vars_meta = whosmat(mat_path)
    info: dict[str, Any] = {
        "format": "matlab_v5_v7",
        "variables": {name: {"shape": list(shape), "dtype": dtype} for name, shape, dtype in vars_meta},
    }
    # Load Tx_data first (tiny: ~14 MB)
    tx_raw = loadmat(mat_path, variable_names=["Tx_data"])["Tx_data"]
    if convert_memmap:
        out_dir.mkdir(parents=True, exist_ok=True)
    # Check orientation before loading Rx_data
    rx_shape = None
    for name, shape, _ in vars_meta:
        if name == "Rx_data":
            rx_shape = shape
    if rx_shape is None:
        raise RuntimeError("Rx_data not found in mat file")

    if tx_raw.shape[1] == rx_shape[1]:
        n_frames = tx_raw.shape[1]
        tx_len = tx_raw.shape[0]
        rx_len = rx_shape[0]
        transpose = True
    elif tx_raw.shape[0] == rx_shape[0]:
        n_frames = tx_raw.shape[0]
        tx_len = tx_raw.shape[1]
        rx_len = rx_shape[1]
        transpose = False
    else:
        raise RuntimeError(f"Shape mismatch: Tx={tx_raw.shape}, Rx={rx_shape}")

    info["inferred_layout"] = {
        "n_frames": int(n_frames),
        "tx_symbols_per_frame": int(tx_len),
        "rx_samples_per_frame": int(rx_len),
        "mat_frame_axis": 1 if transpose else 0,
    }

    if convert_memmap:
        tx_c64 = (tx_raw.T if transpose else tx_raw).astype(np.complex64, copy=False)
        np.save(out_dir / "tx_c64.npy", np.ascontiguousarray(tx_c64))
        del tx_raw, tx_c64

        # Load Rx_data cleanly (complex128 is ~858 MB -> fits well inside <2.5 GB limit)
        rx_dict = loadmat(mat_path, variable_names=["Rx_data"])
        rx_raw = rx_dict.pop("Rx_data")
        del rx_dict
        rx_mm = np.lib.format.open_memmap(out_dir / "rx_c64.npy", mode="w+", dtype=np.complex64, shape=(n_frames, rx_len))
        batch = 256
        for i in range(0, n_frames, batch):
            j = min(i + batch, n_frames)
            chunk = rx_raw[:, i:j].T if transpose else rx_raw[i:j, :]
            rx_mm[i:j] = chunk.astype(np.complex64, copy=False)
        rx_mm.flush()
        del rx_mm, rx_raw
    return info


def analyze_signal_physics(work_dir: Path, n_probe: int = 64) -> dict[str, Any]:
    tx = np.load(work_dir / "tx_c64.npy", mmap_mode="r")
    rx = np.load(work_dir / "rx_c64.npy", mmap_mode="r")
    n_frames, n_sym = tx.shape
    _, n_samp = rx.shape

    # 1. Tx structure audit across ALL frames (fast, 14 MB total)
    tx_all = np.array(tx, dtype=np.complex128)
    tx_nan = int(np.isnan(tx_all.real).sum() + np.isnan(tx_all.imag).sum())
    tx_inf = int(np.isinf(tx_all.real).sum() + np.isinf(tx_all.imag).sum())
    uniq_vals, uniq_counts = np.unique(np.round(tx_all, 6), return_counts=True)

    # Check header constancy across frames
    hdr_diff_per_pos = np.max(np.abs(tx_all - tx_all[0:1, :]), axis=0)
    const_positions = np.where(hdr_diff_per_pos < 1e-6)[0].tolist()
    header_26 = tx_all[0, :26]
    # Check if Barker-13 repeated or double-Barker-13
    barker13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float64)

    # 2. Rx global amplitude & anomaly audit (streamed in chunks)
    rx_rms = np.empty(n_frames, dtype=np.float64)
    rx_peak = np.empty(n_frames, dtype=np.float64)
    rx_dc = np.empty(n_frames, dtype=np.complex128)
    rx_nan = 0
    rx_inf = 0
    for i in range(0, n_frames, 512):
        chunk = np.asarray(rx[i:min(i + 512, n_frames)], dtype=np.complex128)
        rx_nan += int(np.isnan(chunk.real).sum() + np.isnan(chunk.imag).sum())
        rx_inf += int(np.isinf(chunk.real).sum() + np.isinf(chunk.imag).sum())
        rx_rms[i:i + len(chunk)] = np.sqrt(np.mean(chunk.real ** 2 + chunk.imag ** 2, axis=1))
        rx_peak[i:i + len(chunk)] = np.max(np.abs(chunk), axis=1)
        rx_dc[i:i + len(chunk)] = np.mean(chunk, axis=1)

    # 3. First-principles physical parameter verification on first n_probe frames (dev slice)
    # 3a. Autocorrelation period check (is frame repeated every P=2520 samples?)
    probe_rx = np.asarray(rx[:n_probe], dtype=np.complex128)
    probe_tx = tx_all[:n_probe]

    period_lags = [2518, 2519, 2520, 2521, 2522]
    period_corr = {}
    for lag in period_lags:
        a = probe_rx[:, :-lag]
        b = probe_rx[:, lag:]
        num = np.sum(np.conj(a) * b, axis=1)
        den = np.sqrt(np.sum(np.abs(a) ** 2, axis=1) * np.sum(np.abs(b) ** 2, axis=1)) + 1e-30
        c = num / den
        period_corr[str(lag)] = {
            "mean_abs": float(np.mean(np.abs(c))),
            "median_abs": float(np.median(np.abs(c))),
            "min_abs": float(np.min(np.abs(c))),
            "max_abs": float(np.max(np.abs(c))),
            "mean_angle_rad": float(np.angle(np.mean(c))),
            "std_angle_rad": float(np.std(np.angle(c))),
        }

    # 3b. Cyclostationary / energy-vs-phase check for SPS=20
    # Compute mean |r[k]|^2 modulo 20 across sample indices
    sps = 20
    trimmed = probe_rx[:, :(n_samp // sps) * sps].reshape(n_probe, -1, sps)
    mod20_power = np.mean(np.abs(trimmed) ** 2, axis=(0, 1))
    # Also compute per-frame optimal timing phase (Gardner / max power / kurtosis)
    frame_best_phase = np.argmax(np.mean(np.abs(trimmed) ** 2, axis=1), axis=1)

    # 3c. Cross-correlation of probe frames with Tx_data at SPS=20 across all 20 phases and all frame starts
    # Wait: could there be CFO across the 2520-sample frame?
    # Note: from lag=2520 complex correlation angle `phi_frame = angle(sum(conj(r[:-2520]) * r[2520:]))`,
    # the per-sample CFO in rad/sample is `omega_hat = phi_frame / 2520` (modulo 2*pi/2520)!
    # Let's check both (i) zero-CFO correlation and (ii) lag-2520 CFO-compensated correlation,
    # AND (iii) check whether Tx_data vs conj(Tx_data) or different SPS / frequency offset matches!
    lag2520_c = np.sum(np.conj(probe_rx[:, :-2520]) * probe_rx[:, 2520:], axis=1)
    cfo_rad_per_samp_base = np.angle(lag2520_c) / 2520.0  # modulo 2pi/2520

    # For frame 0..15, do a wide CFO search (integer multiples k * 2pi/2520, k in -30..30, plus fine grid)
    # using the known 126-symbol full frame (and separately 26-symbol header) to find exact physical alignment!
    frame_diag = []
    n_idx = np.arange(n_samp, dtype=np.float64)
    # Check both direct tx and conj(tx), and check whether timing offset is uniform or per-frame
    for fidx in range(min(32, n_probe)):
        r = probe_rx[fidx]
        s_full = probe_tx[fidx]  # 126 symbols
        s_hdr = s_full[:26]
        s_pay = s_full[26:]

        # Estimate fine CFO from 2520-lag autocorrelation: omega = (angle + 2*pi*m)/2520
        # Test m in [-15, 15] to resolve any 2pi wrap across 2520 samples!
        c2520 = np.sum(np.conj(r[:-2520]) * r[2520:])
        ang2520 = float(np.angle(c2520))
        rho2520 = float(np.abs(c2520) / (np.sqrt(np.sum(np.abs(r[:-2520])**2) * np.sum(np.abs(r[2520:])**2)) + 1e-30))

        best = {"score": -1.0}
        # Also test coarse frequency search on 20-sample-decimated symbols using FFT of (r_sym * conj(tx_sym))
        # First check across m in [-8..8] and all 2520 lags (since frame repeats every 2520, start in [0, 5040) covers 2 copies)
        for m_wrap in range(-8, 9):
            omega = (ang2520 + 2.0 * np.pi * m_wrap) / 2520.0
            r_derot = r * np.exp(-1j * omega * n_idx)
            # Decimate by phase p in 0..19 -> shape (20, 378)
            r_dec = r_derot[:378 * 20].reshape(378, 20).T  # (20, 378)
            # Correlate each phase with s_full (126 symbols) for starts 0..252
            # Using FFT or direct matmul: 253 windows of length 126
            for p in range(20):
                seq = r_dec[p]
                # sliding inner product with s_full
                corr = np.correlate(seq, s_full, mode="valid")  # length 378 - 126 + 1 = 253
                # normalized by window energy
                win_en = np.convolve(np.abs(seq) ** 2, np.ones(126), mode="valid")
                tx_en = float(np.sum(np.abs(s_full) ** 2))
                norm_corr = np.abs(corr) / np.sqrt(win_en * tx_en + 1e-30)
                idx_max = int(np.argmax(norm_corr))
                sc = float(norm_corr[idx_max])
                if sc > best["score"]:
                    # Check payload SER using header LS channel estimate
                    seg = seq[idx_max:idx_max + 126]
                    h_hdr = np.vdot(s_hdr, seg[:26]) / np.vdot(s_hdr, s_hdr)
                    eq_pay = seg[26:] / h_hdr
                    # nearest QPSK symbol among unique tx constellation
                    const = np.unique(np.round(s_pay, 6))
                    dec_idx = np.argmin(np.abs(eq_pay[:, None] - const[None, :]), axis=1)
                    dec_sym = const[dec_idx]
                    ser = float(np.mean(np.abs(dec_sym - s_pay) > 1e-3))
                    # Header-only peak correlation
                    hcorr = np.correlate(seq, s_hdr, mode="valid")
                    hwin = np.convolve(np.abs(seq) ** 2, np.ones(26), mode="valid")
                    h_norm = np.abs(hcorr) / np.sqrt(hwin * float(np.sum(np.abs(s_hdr)**2)) + 1e-30)
                    h_at_start = float(h_norm[idx_max])
                    best = {
                        "frame": fidx,
                        "score": sc,
                        "hdr_corr_at_start": h_at_start,
                        "hdr_global_max": float(np.max(h_norm)),
                        "hdr_argmax_matches": bool(int(np.argmax(h_norm)) == idx_max or abs(int(np.argmax(h_norm)) - idx_max) == 126),
                        "m_wrap": m_wrap,
                        "omega_rad_per_samp": omega,
                        "omega_rad_per_sym": omega * 20.0,
                        "rho2520": rho2520,
                        "phase_mod20": p,
                        "sym_start": idx_max,
                        "sample_start": idx_max * 20 + p,
                        "h_hdr_abs": float(np.abs(h_hdr)),
                        "h_hdr_angle": float(np.angle(h_hdr)),
                        "payload_ser_basic_ls": ser,
                    }
        frame_diag.append(best)

    # Also check cross-frame correlation of Rx and Tx!
    # Does channel h_hdr evolve smoothly across consecutive frames?
    h_vals = np.array([d["h_hdr_abs"] * np.exp(1j * d["h_hdr_angle"]) for d in frame_diag])
    omega_vals = np.array([d["omega_rad_per_sym"] for d in frame_diag])

    return {
        "tx_shape": [int(n_frames), int(n_sym)],
        "rx_shape": [int(n_frames), int(n_samp)],
        "tx_nan": tx_nan,
        "tx_inf": tx_inf,
        "rx_nan": rx_nan,
        "rx_inf": rx_inf,
        "tx_unique_symbols": [{"re": float(v.real), "im": float(v.imag), "count": int(c)} for v, c in zip(uniq_vals, uniq_counts)],
        "tx_constant_positions_across_all_frames": const_positions,
        "header_26_preview": [{"re": float(v.real), "im": float(v.imag)} for v in header_26],
        "header_26_vs_barker13": {
            "first13_re_match": bool(np.allclose(np.sign(header_26[:13].real), barker13)) or bool(np.allclose(np.sign(header_26[:13].real), -barker13)),
            "second13_re_match": bool(np.allclose(np.sign(header_26[13:26].real), barker13)) or bool(np.allclose(np.sign(header_26[13:26].real), -barker13)),
        },
        "rx_rms_quantiles": {q: float(np.quantile(rx_rms, float(q))) for q in ["0.0", "0.01", "0.05", "0.25", "0.5", "0.75", "0.95", "0.99", "1.0"]},
        "rx_peak_quantiles": {q: float(np.quantile(rx_peak, float(q))) for q in ["0.0", "0.5", "0.95", "1.0"]},
        "rx_dc_abs_median": float(np.median(np.abs(rx_dc))),
        "period_autocorr_probe64": period_corr,
        "mod20_power_profile": [float(x) for x in mod20_power],
        "probe_best_phase_counts": {int(k): int(v) for k, v in zip(*np.unique(frame_best_phase, return_counts=True))},
        "probe32_alignment_summary": {
            "mean_full_corr": float(np.mean([d["score"] for d in frame_diag])),
            "min_full_corr": float(np.min([d["score"] for d in frame_diag])),
            "mean_hdr_corr": float(np.mean([d["hdr_corr_at_start"] for d in frame_diag])),
            "mean_payload_ser_basic_ls": float(np.mean([d["payload_ser_basic_ls"] for d in frame_diag])),
            "median_payload_ser_basic_ls": float(np.median([d["payload_ser_basic_ls"] for d in frame_diag])),
            "zero_ser_frames_out_of_32": int(sum(d["payload_ser_basic_ls"] == 0.0 for d in frame_diag)),
            "m_wrap_values": sorted(list(set(int(d["m_wrap"]) for d in frame_diag))),
            "omega_rad_per_sym_min_med_max": [float(np.min(omega_vals)), float(np.median(omega_vals)), float(np.max(omega_vals))],
            "h_abs_min_med_max": [float(np.min(np.abs(h_vals))), float(np.median(np.abs(h_vals))), float(np.max(np.abs(h_vals)))],
            "sample_starts_first16": [int(d["sample_start"]) for d in frame_diag[:16]],
            "phases_first16": [int(d["phase_mod20"]) for d in frame_diag[:16]],
            "h_angles_first16": [round(float(d["h_hdr_angle"]), 4) for d in frame_diag[:16]],
            "sers_first16": [round(float(d["payload_ser_basic_ls"]), 4) for d in frame_diag[:16]],
        },
        "probe32_frames": frame_diag,
        "peak_rss_mb": round(peak_rss_mb(), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat-path", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--skip-convert", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    with args.mat_path.open("rb") as f:
        magic = f.read(8)
    is_hdf5 = magic == b"\x89HDF\r\n\x1a\n"

    if not args.skip_convert:
        if is_hdf5:
            mat_info = inspect_hdf5(args.mat_path, args.work_dir, convert_memmap=True)
        else:
            mat_info = inspect_mat_v5(args.mat_path, args.work_dir, convert_memmap=True)
    else:
        mat_info = {"format": "skipped_convert", "is_hdf5": is_hdf5}

    phys_info = analyze_signal_physics(args.work_dir, n_probe=64)
    result = {
        "mat_info": mat_info,
        "physics_audit": phys_info,
        "elapsed_sec": round(time.perf_counter() - t0, 3),
        "final_peak_rss_mb": round(peak_rss_mb(), 2),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Print compact summary
    summary = {k: v for k, v in result["physics_audit"].items() if k != "probe32_frames"}
    print(json.dumps({"mat_info": mat_info, "summary": summary, "elapsed_sec": result["elapsed_sec"], "peak_rss_mb": result["final_peak_rss_mb"]}, indent=2))


if __name__ == "__main__":
    main()
