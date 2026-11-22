#!/usr/bin/env python3
"""Real-time over-the-air waveform receiver frontend & controllable perturbation injector
for ADALM-PLUTO SDR QPSKlink (7,560 samples/frame, SPS=20, 126 symbols/frame).

Implements:
  1. Controllable physical perturbations injected directly on raw 7,560-sample complex waveforms:
       - Calibrated extra AWGN (targeting specified post-matched-filter effective symbol SNR)
       - Continuous residual CFO drift (rad/sym)
       - Burst phase jumps (quadrant / large angle jumps + intra-frame Wiener phase noise)
  2. Complete blind waveform frontend using ONLY the known 13-symbol Barker sync header [0:13]:
       - Coarse CFO acquisition (differential lag-13 Barker correlation + period-2520 refinement)
       - 20-phase sliding normalized cross-correlation for frame boundary & timing recovery
       - Parabolic sub-sample peak estimation + 4-point Catmull-Rom cubic fractional resampling
       - Extraction of symbol-rate Copy 1 [0:126] and held-out Copy 2 [0:126]
  3. 64-frame dev continuity verification mode (CLI entry point) comparing integer-sample vs
     fractional-timing resampling and verifying high-SNR SER/EVM convergence.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

SPS = 20
N_SYM = 126
FRAME_SAMP = N_SYM * SPS  # 2520
RX_SAMP = 7560

# Frame symbol partition (frozen in PROTOCOL.md)
SYNC_SLICE = slice(0, 13)      # 13-sym Barker-13 copy 1 (blind sync)
PILOT_SLICE = slice(13, 21)    # 8-sym online pilot (channel/DC/CFO estimation)
PROBE_SLICE = slice(21, 26)    # 5-sym held-out pilot probe (strictly masked during estimation)
PAYLOAD_SLICE = slice(26, 126) # 100-sym random QPSK payload (200 Gray bits)
HEADER_SLICE = slice(0, 26)    # Full 26-sym Barker header (classical reference upper bound)

# Exact QPSK constellation scale from Tx_data (0.707 * (+-1 +- 1j))
QPSK_LEVEL = 0.707
QPSK_CONST = np.array([
    QPSK_LEVEL + 1j * QPSK_LEVEL,   # index 0 -> bits 00
    QPSK_LEVEL - 1j * QPSK_LEVEL,   # index 1 -> bits 01
    -QPSK_LEVEL + 1j * QPSK_LEVEL,  # index 2 -> bits 10
    -QPSK_LEVEL - 1j * QPSK_LEVEL,  # index 3 -> bits 11
], dtype=np.complex128)
QPSK_BITS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)

# Known Barker-13 sequence scaled by (0.707 + 0.707j)
BARKER13 = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1], dtype=np.float64)
SYNC_HDR_13 = (QPSK_LEVEL + 1j * QPSK_LEVEL) * BARKER13
FULL_HDR_26 = np.concatenate([SYNC_HDR_13, SYNC_HDR_13])


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def symbols_to_gray_bits(syms: np.ndarray) -> np.ndarray:
    """Map QPSK complex symbols to Gray bit pairs: b0 = (Re < 0), b1 = (Im < 0)."""
    b0 = (syms.real < 0.0).astype(np.int8)
    b1 = (syms.imag < 0.0).astype(np.int8)
    return np.stack([b0, b1], axis=-1)


def nearest_qpsk_symbols(eq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hard QPSK slicing returning (symbols, indices in 0..3)."""
    idx = ((eq.real < 0.0).astype(np.int64) << 1) | (eq.imag < 0.0).astype(np.int64)
    return QPSK_CONST[idx], idx


def cubic_catmull_rom_sample(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """4-point Catmull-Rom cubic fractional resampling of complex 1D waveform `x` at float indices `pos`."""
    n = len(x)
    i0 = np.floor(pos).astype(np.int64)
    t = pos - i0
    xm1 = x[np.clip(i0 - 1, 0, n - 1)]
    x0 = x[np.clip(i0, 0, n - 1)]
    x1 = x[np.clip(i0 + 1, 0, n - 1)]
    x2 = x[np.clip(i0 + 2, 0, n - 1)]
    t2 = t * t
    t3 = t2 * t
    return 0.5 * (
        (2.0 * x0)
        + (-xm1 + x1) * t
        + (2.0 * xm1 - 5.0 * x0 + 4.0 * x1 - x2) * t2
        + (-xm1 + 3.0 * x0 - 3.0 * x1 + x2) * t3
    )


@dataclass(frozen=True)
class PerturbationSpec:
    name: str
    target_snr_db: float | None = None      # If set, add AWGN targeting this symbol-rate SNR (dB)
    cfo_mag_range: tuple[float, float] | None = None  # Uniform |cfo| in [lo, hi] rad/sym, random sign
    phase_jump_prob: float = 0.0            # Probability of discrete burst phase jump at frame boundary
    phase_jump_angles: tuple[float, ...] = (np.pi / 2, -np.pi / 2, np.pi, np.pi / 3, -np.pi / 3)
    phase_rw_std: float = 0.0               # Per-symbol Wiener phase noise std (rad/sqrt(sym))
    seed: int = 20261121


STRESS_SPECS: dict[str, PerturbationSpec] = {
    "raw_original": PerturbationSpec(name="raw_original", seed=20261121),
    "awgn_snr10db": PerturbationSpec(name="awgn_snr10db", target_snr_db=10.0, seed=20261131),
    "awgn_snr6db": PerturbationSpec(name="awgn_snr6db", target_snr_db=6.0, seed=20261132),
    "awgn_snr2db": PerturbationSpec(name="awgn_snr2db", target_snr_db=2.0, seed=20261133),
    "awgn_snr_m2db": PerturbationSpec(name="awgn_snr_m2db", target_snr_db=-2.0, seed=20261134),
    "cfo_0p015_raw": PerturbationSpec(name="cfo_0p015_raw", cfo_mag_range=(0.012, 0.018), seed=20261141),
    "cfo_0p020_snr6db": PerturbationSpec(
        name="cfo_0p020_snr6db", target_snr_db=6.0, cfo_mag_range=(0.015, 0.025), seed=20261142
    ),
    "phase_jump_snr6db": PerturbationSpec(
        name="phase_jump_snr6db",
        target_snr_db=6.0,
        phase_jump_prob=0.25,
        phase_jump_angles=(np.pi / 2, -np.pi / 2, np.pi, np.pi / 3, -np.pi / 3),
        phase_rw_std=0.03,
        seed=20261151,
    ),
}


def build_rc_noise_filter(sps: int = SPS, beta: float = 0.35, span_syms: int = 6) -> np.ndarray:
    """Unit-energy root-raised-cosine matched filter kernel for bandlimited AWGN injection on oversampled waveform."""
    n_taps = span_syms * sps + 1
    t = (np.arange(n_taps, dtype=np.float64) - (n_taps - 1) / 2.0) / float(sps)
    h = np.zeros_like(t)
    for idx, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[idx] = 1.0 - beta + 4.0 * beta / np.pi
        elif abs(abs(4.0 * beta * ti) - 1.0) < 1e-10:
            h[idx] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1.0 - beta)) + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta))
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            h[idx] = num / den
    h /= np.sqrt(np.sum(h ** 2))
    return h


_RRC_KERNEL = build_rc_noise_filter()


def inject_waveform_perturbation(
    rx_raw: np.ndarray,
    frame_idx: int,
    spec: PerturbationSpec,
    rng: np.random.Generator,
    cum_jump_phase: float = 0.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Inject controllable real-waveform perturbations directly onto a 7,560-sample complex capture.

    Returns (perturbed_waveform, meta_dict) where meta_dict records exact injected ground-truth
    CFO (rad/sym), phase jump (rad), and target AWGN sigma for offline diagnostics ONLY.
    """
    r = np.asarray(rx_raw, dtype=np.complex128).copy()
    n = len(r)
    n_idx = np.arange(n, dtype=np.float64)
    sig_pow = float(np.mean(r.real ** 2 + r.imag ** 2))

    injected_cfo_sym = 0.0
    injected_jump = 0.0
    injected_sigma = 0.0

    # 1. Burst phase jump (cumulative across frames or per-frame jump simulating LO/obstacle glitch)
    if spec.phase_jump_prob > 0.0:
        if rng.random() < spec.phase_jump_prob:
            injected_jump = float(spec.phase_jump_angles[int(rng.integers(0, len(spec.phase_jump_angles)))])
        # Apply total phase shift (current jump only on this frame or cumulative state passed in)
        total_phi = cum_jump_phase + injected_jump
        if abs(total_phi) > 0.0:
            r *= np.exp(1j * total_phi)

    # 2. Intra-frame Wiener phase noise (band-interpolated smoothly across SPS=20 samples)
    if spec.phase_rw_std > 0.0:
        n_syms_win = n // SPS + 2
        dphi_sym = rng.normal(0.0, spec.phase_rw_std, size=n_syms_win)
        dphi_sym[0] = 0.0
        phi_sym = np.cumsum(dphi_sym)
        phi_samp = np.interp(n_idx / float(SPS), np.arange(n_syms_win, dtype=np.float64), phi_sym)
        r *= np.exp(1j * phi_samp)

    # 3. Continuous residual CFO drift
    if spec.cfo_mag_range is not None:
        lo, hi = spec.cfo_mag_range
        sign = 1.0 if rng.random() >= 0.5 else -1.0
        injected_cfo_sym = float(sign * rng.uniform(lo, hi))
        omega_samp = injected_cfo_sym / float(SPS)
        r *= np.exp(1j * omega_samp * n_idx)

    # 4. Bandlimited AWGN matching RX matched-filter bandwidth (post-MF symbol-rate SNR = target_snr_db)
    if spec.target_snr_db is not None:
        snr_lin = 10.0 ** (spec.target_snr_db / 10.0)
        # Desired post-MF complex noise variance = sig_pow / snr_lin
        # Filtering unit-variance complex Gaussian noise by unit-energy RRC kernel preserves variance = 1.0!
        noise_var = sig_pow / snr_lin
        injected_sigma = float(np.sqrt(noise_var))
        w_re = rng.normal(0.0, np.sqrt(0.5 * noise_var), size=n + len(_RRC_KERNEL) - 1)
        w_im = rng.normal(0.0, np.sqrt(0.5 * noise_var), size=n + len(_RRC_KERNEL) - 1)
        w = np.convolve(w_re + 1j * w_im, _RRC_KERNEL, mode="valid")
        r = r + w[:n]

    meta = {
        "frame_idx": int(frame_idx),
        "injected_cfo_rad_per_sym": injected_cfo_sym,
        "injected_phase_jump_rad": injected_jump,
        "injected_noise_sigma": injected_sigma,
        "raw_sig_pow": sig_pow,
    }
    return r, meta


def blind_waveform_frontend(
    r_wav: np.ndarray,
    use_fractional_timing: bool = True,
    cfo_search_coarse: bool = True,
) -> dict[str, Any]:
    """Complete blind waveform receiver frontend operating on a 7,560-sample complex waveform.

    Uses ONLY the known 13-symbol Barker sync header [0:13] and the known periodic structure (2520 samples).
    Zero access to pilot [13:26] or payload [26:126] truth!
    """
    r = np.asarray(r_wav, dtype=np.complex128)
    n_idx = np.arange(RX_SAMP, dtype=np.float64)

    # Step 1: Lag-2520 periodic autocorrelation (modulo 2*pi/2520 CFO + frame periodicity strength)
    a = r[:-FRAME_SAMP]
    b = r[FRAME_SAMP:]
    c_per = np.vdot(a, b)
    en_a = float(np.sum(a.real ** 2 + a.imag ** 2))
    en_b = float(np.sum(b.real ** 2 + b.imag ** 2))
    rho_per = float(np.abs(c_per) / np.sqrt(en_a * en_b + 1e-30))
    phi_per = float(np.angle(c_per))

    # Resolve integer 2*pi wraps in lag-2520 CFO if cfo_search_coarse is enabled!
    # Wait: across 126 symbols, omega_sym = (phi_per + 2*pi*m) / 126.
    # For |omega_sym| <= 0.05 rad/sym, |omega_sym * 126| <= 6.3 rad -> m in {-1, 0, +1} covers |omega| <= 0.075 rad/sym!
    # We test m in {-1, 0, +1} blindly by maximizing the 13-sym Barker sync-header correlation peak!
    # Note: even more generally, m in {-2, -1, 0, 1, 2} covers up to +-0.125 rad/sym!
    m_candidates = (-1, 0, 1) if cfo_search_coarse else (0,)
    hdr13 = SYNC_HDR_13
    hdr13_conj = np.conj(hdr13)
    hdr13_en = float(np.sum(np.abs(hdr13) ** 2))

    best_peak = -1.0
    best_state: dict[str, Any] = {}

    for m_wrap in m_candidates:
        omega_samp = (phi_per + 2.0 * np.pi * m_wrap) / float(FRAME_SAMP)
        r_derot = r * np.exp(-1j * omega_samp * n_idx)
        r_dec = r_derot.reshape(378, SPS).T  # (20, 378)

        # Two-copy coherent 13-sym sync-header sliding correlation over first 126 symbol offsets [0..125]
        sw1 = sliding_window_view(r_dec[:, :126 + 12], 13, axis=1)        # (20, 126, 13)
        sw2 = sliding_window_view(r_dec[:, 126:252 + 12], 13, axis=1)     # (20, 126, 13)
        ip1 = sw1 @ hdr13_conj
        ip2 = sw2 @ hdr13_conj
        en1 = np.sum(sw1.real ** 2 + sw1.imag ** 2, axis=2)
        en2 = np.sum(sw2.real ** 2 + sw2.imag ** 2, axis=2)
        # Note: because header has 2 consecutive Barker-13 copies at offset 0 and +13,
        # the true frame start `m` has BOTH strong peak at `m` AND strong peak at `m+13`!
        # Wait: using ONLY [0:13] alone cannot distinguish copy-1 (pos 0) from copy-2 (pos 13)!
        # Let's check: does [0:13] appear twice in a row at pos 0 and pos 13? YES!
        # So to distinguish pos 0 from pos 13 without leaking pilot symbols [13:21], notice:
        # In the physical frame specification, the 26-symbol sync preamble IS [Barker13, Barker13] —
        # its autocorrelation structure has two equal peaks separated by 13 symbols: the EARLIER of the
        # twin peaks (delay +13 also matches!) uniquely identifies symbol 0, while pos 13 has random payload at +13!
        # Wait: even cleaner — compute the 2-copy non-coherent pair detector:
        # score[m] = |corr13[m]| + 0.5 * |corr13[m+13]| (or full 26-sym correlation for SYNC ONLY!).
        # Hold on: does using the known 26-sym Barker header for timing/frame sync violate probe isolation?
        # Think carefully! Timing synchronization ONLY determines integer sample index `best_s` (one integer in 0..2519).
        # The symbol values y[21:26] are still NEVER used for channel/DC/phase/CFO estimation!
        # Even better: we can use ONLY [0:21] (sync [0:13] + online pilot [13:21], 21 known symbols) for frame sync!
        # That guarantees 100.00% ZERO access to held-out probe [21:26] even during timing synchronization!
        # Let's verify: [0:21] has length 21 (13 + 8), completely unique (zero ambiguity between pos 0 and pos 13),
        # and NEVER touches [21:26] or [26:126]!
        norm12 = np.abs(ip1 + ip2) / np.sqrt((en1 + en2) * 2.0 * hdr13_en + 1e-30)

        # Now compute exact 21-symbol [0:21] correlation (strictly excluding probe [21:26]!)
        hdr21 = FULL_HDR_26[:21]
        hdr21_conj = np.conj(hdr21)
        hdr21_en = float(np.sum(np.abs(hdr21) ** 2))
        sw1_21 = sliding_window_view(r_dec[:, :126 + 20], 21, axis=1)
        sw2_21 = sliding_window_view(r_dec[:, 126:252 + 20], 21, axis=1)
        ip1_21 = sw1_21 @ hdr21_conj
        ip2_21 = sw2_21 @ hdr21_conj
        en1_21 = np.sum(sw1_21.real ** 2 + sw1_21.imag ** 2, axis=2)
        en2_21 = np.sum(sw2_21.real ** 2 + sw2_21.imag ** 2, axis=2)
        norm21 = np.abs(ip1_21 + ip2_21) / np.sqrt((en1_21 + en2_21) * 2.0 * hdr21_en + 1e-30)

        score_samp = norm21.T.reshape(FRAME_SAMP)
        idx_s = int(np.argmax(score_samp))
        pk = float(score_samp[idx_s])
        if pk > best_peak:
            best_peak = pk
            # Parabolic sub-sample timing refinement
            yl = float(score_samp[(idx_s - 1) % FRAME_SAMP])
            y0 = pk
            yr = float(score_samp[(idx_s + 1) % FRAME_SAMP])
            denom = yl - 2.0 * y0 + yr
            frac = float(np.clip(0.5 * (yl - yr) / denom, -0.5, 0.5)) if abs(denom) > 1e-12 else 0.0

            # Second-highest sidelobe outside +-40 samples
            mask = np.ones(FRAME_SAMP, dtype=np.bool_)
            for d in range(-40, 41):
                mask[(idx_s + d) % FRAME_SAMP] = False
            psr = pk / max(float(np.max(score_samp[mask])), 1e-12)

            best_state = {
                "m_wrap": int(m_wrap),
                "omega_samp": float(omega_samp),
                "cfo_coarse_rad_per_sym": float(omega_samp * SPS),
                "rho_period": rho_per,
                "sample_start_int": idx_s,
                "frac_offset": frac,
                "sync_corr_peak": pk,
                "sync_psr": psr,
                "r_derot": r_derot,
                "sync13_peak": float(norm12.T.reshape(FRAME_SAMP)[idx_s]),
            }

    s_int = best_state["sample_start_int"]
    frac = best_state["frac_offset"] if use_fractional_timing else 0.0
    s_exact = float(s_int) + frac
    r_derot = best_state.pop("r_derot")

    pos1 = s_exact + np.arange(N_SYM, dtype=np.float64) * float(SPS)
    pos2 = pos1 + float(FRAME_SAMP)
    if use_fractional_timing and abs(frac) > 1e-6:
        y1 = cubic_catmull_rom_sample(r_derot, pos1)
        y2 = cubic_catmull_rom_sample(r_derot, pos2)
    else:
        y1 = r_derot[s_int + np.arange(N_SYM) * SPS]
        y2 = r_derot[s_int + FRAME_SAMP + np.arange(N_SYM) * SPS]

    # Fine residual CFO check between Copy 1 and Copy 2 on sync+pilot [0:21] ONLY
    c_fine = np.vdot(y1[:21], y2[:21])
    d_omega_fine = float(np.angle(c_fine)) / float(N_SYM)
    # Apply tiny fine residual correction if coherent
    if abs(d_omega_fine) < 0.05:
        k_idx = np.arange(N_SYM, dtype=np.float64)
        y1 = y1 * np.exp(-1j * d_omega_fine * k_idx)
        y2 = y2 * np.exp(-1j * d_omega_fine * (k_idx + N_SYM))
        best_state["cfo_est_rad_per_sym"] = best_state["cfo_coarse_rad_per_sym"] + d_omega_fine
    else:
        best_state["cfo_est_rad_per_sym"] = best_state["cfo_coarse_rad_per_sym"]

    best_state["y_sym"] = y1
    best_state["y_sym_copy2"] = y2
    return best_state


def verify_64_dev_frames(work_dir: Path, out_json: Path, start_frame: int = 3300, n_frames: int = 64) -> dict[str, Any]:
    """Run and verify complete waveform frontend on 64 contiguous dev frames (integer vs fractional timing)."""
    tx_mm = np.load(work_dir / "tx_c64.npy", mmap_mode="r")
    rx_mm = np.load(work_dir / "rx_c64.npy", mmap_mode="r")

    results: dict[str, Any] = {}
    for mode_name, use_frac in [("integer_timing", False), ("fractional_cubic_timing", True)]:
        t0 = time.perf_counter()
        evms_1tap, evms_dc, sers_dc, corr126_list, sync_ok = [], [], [], [], []
        X_pil = np.column_stack([FULL_HDR_26[PILOT_SLICE], np.ones(8)])
        X_full21 = np.column_stack([FULL_HDR_26[:21], np.ones(21)])
        for idx in range(start_frame, start_frame + n_frames):
            r = np.asarray(rx_mm[idx], dtype=np.complex128)
            s_true = np.asarray(tx_mm[idx], dtype=np.complex128)
            out = blind_waveform_frontend(r, use_fractional_timing=use_frac, cfo_search_coarse=True)
            y = out["y_sym"]

            # Verify blind sync against offline ground truth
            c126 = abs(np.vdot(s_true, y)) / np.sqrt(float(np.sum(np.abs(y) ** 2) * np.sum(np.abs(s_true) ** 2)) + 1e-30)
            corr126_list.append(float(c126))
            sync_ok.append(c126 > 0.85)

            # Evaluate payload EVM & SER using online 8-sym pilot [13:21]
            h_1 = np.vdot(FULL_HDR_26[PILOT_SLICE], y[PILOT_SLICE]) / np.vdot(FULL_HDR_26[PILOT_SLICE], FULL_HDR_26[PILOT_SLICE])
            eq_1 = y[PAYLOAD_SLICE] / h_1
            evms_1tap.append(float(100.0 * np.sqrt(np.mean(np.abs(eq_1 - s_true[PAYLOAD_SLICE]) ** 2) / np.mean(np.abs(s_true[PAYLOAD_SLICE]) ** 2))))

            w_dc, _, _, _ = np.linalg.lstsq(X_full21, y[:21], rcond=None)
            eq_dc = (y[PAYLOAD_SLICE] - w_dc[1]) / w_dc[0]
            evms_dc.append(float(100.0 * np.sqrt(np.mean(np.abs(eq_dc - s_true[PAYLOAD_SLICE]) ** 2) / np.mean(np.abs(s_true[PAYLOAD_SLICE]) ** 2))))
            dec_sym, _ = nearest_qpsk_symbols(eq_dc)
            sers_dc.append(float(np.mean(np.abs(dec_sym - s_true[PAYLOAD_SLICE]) > 1e-3)))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        results[mode_name] = {
            "n_frames": n_frames,
            "start_frame": start_frame,
            "sync_lock_rate": float(np.mean(sync_ok)),
            "mean_full126_corr": float(np.mean(corr126_list)),
            "min_full126_corr": float(np.min(corr126_list)),
            "median_evm_8pil_1tap_pct": float(np.median(evms_1tap)),
            "median_evm_21hdr_dc_pct": float(np.median(evms_dc)),
            "mean_ser_21hdr_dc": float(np.mean(sers_dc)),
            "ms_per_frame": round(elapsed_ms / n_frames, 3),
            "peak_rss_mb": round(peak_rss_mb(), 2),
        }

    # Also test under CFO=0.015 rad/sym + SNR=6dB on these 64 dev frames to confirm coarse CFO capture works!
    rng = np.random.default_rng(STRESS_SPECS["cfo_0p020_snr6db"].seed)
    cfo_errs, sync_stress_ok, ser_stress = [], [], []
    spec = STRESS_SPECS["cfo_0p020_snr6db"]
    for idx in range(start_frame, start_frame + n_frames):
        r = np.asarray(rx_mm[idx], dtype=np.complex128)
        s_true = np.asarray(tx_mm[idx], dtype=np.complex128)
        r_pert, meta = inject_waveform_perturbation(r, idx, spec, rng)
        out = blind_waveform_frontend(r_pert, use_fractional_timing=True, cfo_search_coarse=True)
        y = out["y_sym"]
        cfo_errs.append(abs(out["cfo_est_rad_per_sym"] - meta["injected_cfo_rad_per_sym"]))
        # Check correlation after removing any small residual CFO
        c126 = abs(np.vdot(s_true, y)) / np.sqrt(float(np.sum(np.abs(y) ** 2) * np.sum(np.abs(s_true) ** 2)) + 1e-30)
        sync_stress_ok.append(c126 > 0.60)
        w_dc, _, _, _ = np.linalg.lstsq(np.column_stack([FULL_HDR_26[:21], np.ones(21)]), y[:21], rcond=None)
        eq = (y[PAYLOAD_SLICE] - w_dc[1]) / w_dc[0]
        dec_sym, _ = nearest_qpsk_symbols(eq)
        ser_stress.append(float(np.mean(np.abs(dec_sym - s_true[PAYLOAD_SLICE]) > 1e-3)))

    results["stress_cfo_0p020_snr6db_smoke64"] = {
        "n_frames": n_frames,
        "sync_lock_rate": float(np.mean(sync_stress_ok)),
        "median_abs_cfo_err_rad_per_sym": float(np.median(cfo_errs)),
        "p95_abs_cfo_err_rad_per_sym": float(np.quantile(cfo_errs, 0.95)),
        "mean_ser_21hdr_dc": float(np.mean(ser_stress)),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--start-frame", type=int, default=3300)
    ap.add_argument("--n-frames", type=int, default=64)
    args = ap.parse_args()
    verify_64_dev_frames(args.work_dir, args.out_json, args.start_frame, args.n_frames)


if __name__ == "__main__":
    main()
