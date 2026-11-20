"""Independent blind physics reference, exhaustive mini-code ML verifier, and firewall checkers.

Written completely independently of P9-A/B/C receiver and channel implementations so it serves
as an uncoupled verification oracle for Phase 9.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import numpy as np

from p9.contract import (
    CORE_ROWS,
    CP,
    DELAYS,
    FRAME_DURATION,
    N,
    POWERS,
    PROBE_ROW,
    PublicObservation,
    SC_LAYOUTS,
    T,
    n0_from_snr,
)


BLOCK_LEN = N + CP                     # 48 samples per CP-guarded block
TOTAL_BLOCKS = 2 * T                   # 40 blocks across Frame A (OFDM) + Frame B (CP-SC)
TOTAL_SAMPLES = TOTAL_BLOCKS * BLOCK_LEN  # 1,920 serial samples
PAM_GRAY_16QAM = np.array([-3.0, -1.0, 3.0, 1.0], dtype=np.float64) / math.sqrt(10.0)

# Strictly forbidden hidden-truth attribute names on receiver-visible observations.
FORBIDDEN_HIDDEN_ATTRS = frozenset({
    "regime",
    "channel",
    "channel_a",
    "channel_b",
    "changed_ray",
    "delta_freq",
    "shift_row",
    "b_info",
    "b_code",
    "b_symbols",
    "b_words",
    "y_b_probe_noisy",
    "y_b_probe_clean",
    "y_b_full_clean",
    "clean_b",
    "h_b_true",
})

ALLOWED_PUBLIC_ATTRS = frozenset({
    "y_a",
    "y_b",
    "training_rows",
    "probe_row",
    "n0",
    "budget",
})


@dataclass(frozen=True)
class PiecewiseDopplerShift:
    """Phase-continuous instantaneous Doppler (and optional chirp) shift on a single ray.

    For times t >= t_shift on `ray_index`, the instantaneous frequency (cycles / N samples)
    shifts continuously in phase by:
        f(t) = f_0 + c_0 * t + delta_freq + delta_chirp * (t - t_shift)
        phi(t) = 2*pi * (f_0 * t + 0.5 * c_0 * t^2
                         + delta_freq * (t - t_shift) + 0.5 * delta_chirp * (t - t_shift)^2)
    """
    ray_index: int
    delta_freq: float
    t_shift: float
    delta_chirp: float = 0.0


@dataclass(frozen=True)
class SerialPropagationResult:
    """Full diagnostics from sample-by-sample serial CP time-domain convolution."""
    y_a: np.ndarray            # [T, N] complex128 (freq-domain OFDM if a_is_freq else time-domain)
    y_b: np.ndarray            # [T, N] complex128 (time-domain post-CP CP-SC)
    tx_serial: np.ndarray      # [1920] complex128 transmitted serial stream with CPs
    rx_serial: np.ndarray      # [1920] complex128 received serial stream before CP removal
    t_serial: np.ndarray       # [1920] float64 absolute sample clock t_k = k / N
    g_serial: np.ndarray       # [3, 1920] complex128 exact ray gains at every serial sample
    h_b_useful: np.ndarray     # [3, T, N] complex128 exact ray gains on Frame B useful samples
    phase_continuity: Optional[dict] = None


def row_boundary_time(shift_row: int, *, frame: str = "B", at_useful_start: bool = False) -> float:
    """Absolute sample-clock time (in units of N samples) for a row boundary in Frame A or B.

    - row_boundary_time(0, frame='B') == 30.0 == FRAME_DURATION (exact A/B CP boundary)
    - row_boundary_time(6, frame='B') == 39.0 (exact start of Frame B row 6 CP)
    """
    if frame not in ("A", "B"):
        raise ValueError(f"frame must be 'A' or 'B', got {frame!r}")
    if not (0 <= int(shift_row) <= T):
        raise ValueError(f"shift_row must be in [0, {T}], got {shift_row}")
    abs_row = int(shift_row) if frame == "A" else T + int(shift_row)
    sample_idx = abs_row * BLOCK_LEN + (CP if at_useful_start else 0)
    return float(sample_idx) / float(N)


def _extract_channel_params(channel: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (gain, frequency, chirp) in complex128/float64 from any channel-like object."""
    if isinstance(channel, Mapping):
        gain = np.asarray(channel["gain"], dtype=np.complex128)
        freq = np.asarray(channel["frequency"], dtype=np.float64)
        chirp = np.asarray(channel["chirp"], dtype=np.float64)
    else:
        gain = np.asarray(channel.gain, dtype=np.complex128)
        freq = np.asarray(channel.frequency, dtype=np.float64)
        chirp = np.asarray(channel.chirp, dtype=np.float64)
    if gain.shape != (len(DELAYS),) or freq.shape != (len(DELAYS),) or chirp.shape != (len(DELAYS),):
        raise ValueError(
            f"Expected 3-ray channel arrays, got shapes {gain.shape}, {freq.shape}, {chirp.shape}"
        )
    return gain, freq, chirp


def _normalize_shifts(
    shift: Optional[PiecewiseDopplerShift | Sequence[PiecewiseDopplerShift]],
) -> tuple[PiecewiseDopplerShift, ...]:
    if shift is None:
        return ()
    if isinstance(shift, PiecewiseDopplerShift):
        shifts = (shift,)
    else:
        shifts = tuple(shift)
    for s in shifts:
        if not isinstance(s, PiecewiseDopplerShift):
            raise TypeError(f"Expected PiecewiseDopplerShift, got {type(s)!r}")
        if not (0 <= int(s.ray_index) < len(DELAYS)):
            raise ValueError(f"ray_index must be in [0, {len(DELAYS) - 1}], got {s.ray_index}")
        if not (math.isfinite(s.delta_freq) and math.isfinite(s.t_shift) and math.isfinite(s.delta_chirp)):
            raise ValueError(f"Non-finite shift parameter in {s}")
    return shifts


def instantaneous_phase_and_gain(
    times: np.ndarray,
    channel: Any,
    shift: Optional[PiecewiseDopplerShift | Sequence[PiecewiseDopplerShift]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute exact continuous phase phi_l(t) [rad] and complex ray gain g_l(t) in float64/complex128."""
    t_arr = np.asarray(times, dtype=np.float64)
    gain, freq, chirp = _extract_channel_params(channel)
    shifts = _normalize_shifts(shift)

    n_rays = len(DELAYS)
    phi = np.empty((n_rays,) + t_arr.shape, dtype=np.float64)
    for l_idx in range(n_rays):
        phi[l_idx] = 2.0 * math.pi * (freq[l_idx] * t_arr + 0.5 * chirp[l_idx] * (t_arr ** 2))

    for s in shifts:
        mask = t_arr >= s.t_shift
        dt = np.where(mask, t_arr - s.t_shift, 0.0)
        phi[s.ray_index] += 2.0 * math.pi * (s.delta_freq * dt + 0.5 * s.delta_chirp * (dt ** 2))

    g = gain.reshape((n_rays,) + (1,) * t_arr.ndim) * np.exp(1j * phi)
    return phi, g


def verify_phase_continuity(
    channel: Any,
    shift: PiecewiseDopplerShift,
    *,
    probe_dt: float = 1e-7,
    post_sample_times: Optional[np.ndarray] = None,
) -> dict:
    """Explicitly verify instantaneous phase continuity |phi(t_shift^+) - phi(t_shift^-)| < 1e-13.

    Also cross-checks:
    1. Re-anchored post-shift branch (starting at tau = t - t_shift = 0) vs pre-shift branch at t_shift.
    2. Numerical one-sided frequency derivative jump == delta_freq.
    3. Global-clock equivalent Channel representation over post_sample_times.
    """
    gain, freq, chirp = _extract_channel_params(channel)
    l_idx = int(shift.ray_index)
    t_s = float(shift.t_shift)
    df = float(shift.delta_freq)
    dc = float(shift.delta_chirp)

    # Exact left branch evaluated at t_shift^-
    phi_left = 2.0 * math.pi * (freq[l_idx] * t_s + 0.5 * chirp[l_idx] * (t_s ** 2))
    g_left = gain[l_idx] * np.exp(1j * phi_left)

    # Re-anchored right branch evaluated at tau = t - t_shift = 0^+
    f_post_anchor = freq[l_idx] + chirp[l_idx] * t_s + df
    c_post_anchor = chirp[l_idx] + dc
    # Equivalent global-clock representation for t >= t_shift:
    # f_eq = f_0 + df - dc * t_s, c_eq = c_0 + dc, phase_offset = 2*pi * (-df * t_s + 0.5 * dc * t_s^2)
    f_eq = freq[l_idx] + df - dc * t_s
    c_eq = chirp[l_idx] + dc
    phi_offset = 2.0 * math.pi * (-df * t_s + 0.5 * dc * (t_s ** 2))
    phi_right = phi_offset + 2.0 * math.pi * (f_eq * t_s + 0.5 * c_eq * (t_s ** 2))
    g_right = gain[l_idx] * np.exp(1j * phi_right)

    # Direct piecewise evaluator right at t_shift
    phi_eval, g_eval = instantaneous_phase_and_gain(np.array([t_s], dtype=np.float64), channel, shift)
    phi_direct_jump = float(abs(phi_eval[l_idx, 0] - phi_left))
    phi_branch_jump = float(abs(phi_right - phi_left))
    phase_jump_rad = max(phi_direct_jump, phi_branch_jump)
    gain_jump = float(max(abs(g_eval[l_idx, 0] - g_left), abs(g_right - g_left)))

    # One-sided finite-difference instantaneous frequency jump around t_shift
    h = float(probe_dt)
    t_probe = np.array([t_s - h, t_s, t_s + h], dtype=np.float64)
    phi_probe, _ = instantaneous_phase_and_gain(t_probe, channel, shift)
    f_minus = float((phi_probe[l_idx, 1] - phi_probe[l_idx, 0]) / (2.0 * math.pi * h))
    f_plus = float((phi_probe[l_idx, 2] - phi_probe[l_idx, 1]) / (2.0 * math.pi * h))
    freq_jump_measured = f_plus - f_minus
    freq_jump_err = abs(freq_jump_measured - df)

    # Cross-check post-shift trajectory against re-anchored local-tau expression
    if post_sample_times is None:
        post_sample_times = np.arange(TOTAL_SAMPLES, dtype=np.float64) / float(N)
    post_t = np.asarray(post_sample_times, dtype=np.float64)
    post_t = post_t[post_t >= t_s]
    if post_t.size > 0:
        tau = post_t - t_s
        g_reanchored = g_left * np.exp(
            2j * math.pi * (f_post_anchor * tau + 0.5 * c_post_anchor * (tau ** 2))
        )
        _, g_piecewise = instantaneous_phase_and_gain(post_t, channel, shift)
        reanchored_max_gain_err = float(np.max(np.abs(g_piecewise[l_idx] - g_reanchored)))
    else:
        reanchored_max_gain_err = 0.0

    return {
        "ray_index": l_idx,
        "t_shift": t_s,
        "delta_freq": df,
        "delta_chirp": dc,
        "phase_jump_rad": phase_jump_rad,
        "gain_jump": gain_jump,
        "freq_jump_measured": freq_jump_measured,
        "freq_jump_err": freq_jump_err,
        "reanchored_max_gain_err": reanchored_max_gain_err,
        "continuous_ok": bool(phase_jump_rad < 1e-13 and gain_jump < 1e-13 and reanchored_max_gain_err < 1e-13),
    }


def serial_propagate_ab(
    a_grid: np.ndarray,
    b_grid: np.ndarray,
    channel_a: Any,
    *,
    a_is_freq: bool = True,
    shift: Optional[PiecewiseDopplerShift | Sequence[PiecewiseDopplerShift]] = None,
    channel_b: Optional[Any] = None,
    channel_b_time_origin: str = "global",
    return_details: bool = False,
) -> tuple[np.ndarray, np.ndarray] | SerialPropagationResult:
    """Independent sample-by-sample serial CP time-domain convolution across Frame A + Frame B.

    Operates on the full 40-symbol (A=OFDM + B=CP-SC) serial stream of 40 * 48 = 1,920 samples:
      1. Converts Frame A to useful time-domain samples via unitary IFFT (when a_is_freq=True).
      2. Prepends the 16-sample cyclic prefix to every block (40 blocks -> 1,920 serial samples).
      3. Evaluates exact instantaneous 3-ray complex gains g_l(t_k) at every serial clock t_k = k/32.
      4. Performs explicit sample-by-sample causal linear FIR convolution y[k] = sum_l g_l(t_k) s[k-d_l].
      5. Discards every block's 16-sample CP and applies unitary FFT to Frame A.
    """
    a_arr = np.asarray(a_grid, dtype=np.complex128)
    b_arr = np.asarray(b_grid, dtype=np.complex128)
    if a_arr.shape != (T, N) or b_arr.shape != (T, N):
        raise ValueError(f"Expected (20, 32) grids for A and B, got {a_arr.shape} and {b_arr.shape}")

    if channel_b is not None and shift is not None:
        raise ValueError("Specify either piecewise `shift` or full-reset `channel_b`, not both")

    a_useful = np.fft.ifft(a_arr, norm="ortho") if a_is_freq else a_arr.copy()
    b_useful = b_arr.copy()
    useful_ab = np.vstack((a_useful, b_useful))  # [40, 32]

    # Insert CP for every block m = 0..39 -> [40, 48] -> [1920]
    tx_blocks = np.empty((TOTAL_BLOCKS, BLOCK_LEN), dtype=np.complex128)
    for m in range(TOTAL_BLOCKS):
        tx_blocks[m, :CP] = useful_ab[m, N - CP:]
        tx_blocks[m, CP:] = useful_ab[m, :]
    tx_serial = tx_blocks.reshape(TOTAL_SAMPLES)

    t_serial = np.arange(TOTAL_SAMPLES, dtype=np.float64) / float(N)
    phase_cont = None

    if channel_b is not None:
        split_k = T * BLOCK_LEN  # 960, exact A/B CP boundary (t = 30.0)
        t_a = t_serial[:split_k]
        if channel_b_time_origin == "global":
            t_b = t_serial[split_k:]
        elif channel_b_time_origin == "frame_b":
            t_b = t_serial[split_k:] - FRAME_DURATION
        else:
            raise ValueError(f"Unsupported channel_b_time_origin: {channel_b_time_origin!r}")
        _, g_a = instantaneous_phase_and_gain(t_a, channel_a)
        _, g_b = instantaneous_phase_and_gain(t_b, channel_b)
        g_serial = np.concatenate((g_a, g_b), axis=1)
    else:
        _, g_serial = instantaneous_phase_and_gain(t_serial, channel_a, shift=shift)
        shifts = _normalize_shifts(shift)
        if len(shifts) == 1:
            phase_cont = verify_phase_continuity(channel_a, shifts[0], post_sample_times=t_serial)

    # Explicit sample-by-sample causal linear FIR convolution (no np.roll shortcut!)
    n_rays = len(DELAYS)
    delays_int = [int(d) for d in DELAYS]
    rx_serial = np.zeros(TOTAL_SAMPLES, dtype=np.complex128)
    for k in range(TOTAL_SAMPLES):
        acc = 0.0 + 0.0j
        for l_idx in range(n_rays):
            d = delays_int[l_idx]
            if k >= d:
                acc += g_serial[l_idx, k] * tx_serial[k - d]
        rx_serial[k] = acc

    rx_blocks = rx_serial.reshape(TOTAL_BLOCKS, BLOCK_LEN)
    rx_useful = rx_blocks[:, CP:]  # [40, 32]
    y_a_time = rx_useful[:T]
    y_b_time = rx_useful[T:]
    y_a_out = np.fft.fft(y_a_time, norm="ortho") if a_is_freq else y_a_time

    if not return_details:
        return y_a_out, y_b_time

    h_b_useful = g_serial.reshape(n_rays, TOTAL_BLOCKS, BLOCK_LEN)[:, T:, CP:]
    return SerialPropagationResult(
        y_a=y_a_out,
        y_b=y_b_time,
        tx_serial=tx_serial,
        rx_serial=rx_serial,
        t_serial=t_serial,
        g_serial=g_serial,
        h_b_useful=h_b_useful,
        phase_continuity=phase_cont,
    )


def blockwise_propagate_piecewise(
    grid: np.ndarray,
    channel: Any,
    start_row: int,
    *,
    is_ofdm: bool = False,
    shift: Optional[PiecewiseDopplerShift | Sequence[PiecewiseDopplerShift]] = None,
) -> np.ndarray:
    """Blockwise circular-convolution reference supporting piecewise phase-continuous shifts."""
    arr = np.asarray(grid, dtype=np.complex128)
    if arr.ndim != 2 or arr.shape[1] != N:
        raise ValueError(f"Expected [rows, {N}] grid, got {arr.shape}")
    rows = np.arange(start_row, start_row + arr.shape[0], dtype=np.float64)
    times = (rows[:, None] * BLOCK_LEN + CP + np.arange(N, dtype=np.float64)) / float(N)
    _, gains = instantaneous_phase_and_gain(times, channel, shift=shift)
    tx_time = np.fft.ifft(arr, norm="ortho") if is_ofdm else arr
    rx_time = np.zeros_like(tx_time, dtype=np.complex128)
    for l_idx, delay in enumerate(DELAYS):
        rx_time += gains[l_idx] * np.roll(tx_time, int(delay), axis=-1)
    return np.fft.fft(rx_time, norm="ortho") if is_ofdm else rx_time


# ---------------------------------------------------------------------------
# Independent Exhaustive Mini-Codebook (16, 8, 5) ML Certificate Verifier
# ---------------------------------------------------------------------------

def mini_code_16_8() -> dict:
    """Construct optimal systematic (16, 8, 5) binary linear code (shortened (17, 9, 5) QR code).

    Generator polynomial over GF(2): g(x) = x^8 + x^5 + x^4 + x^3 + 1.
    Maps 8 info bits -> 16 coded bits -> 4 Gray 16-QAM symbols (256 total codewords, d_min = 5).
    """
    gen_poly = np.array([1, 0, 0, 1, 1, 1, 0, 0, 1], dtype=np.uint8)
    parity = np.zeros((8, 8), dtype=np.uint8)
    for i in range(8):
        poly = np.zeros(16, dtype=np.uint8)
        poly[8 + i] = 1
        for deg in range(15, 7, -1):
            if poly[deg]:
                poly[deg - 8 : deg + 1] ^= gen_poly
        parity[i] = poly[:8]

    gen = np.hstack((np.eye(8, dtype=np.uint8), parity))
    h_par = np.hstack((parity.T, np.eye(8, dtype=np.uint8)))
    if np.any((gen @ h_par.T) % 2):
        raise AssertionError("Constructed (16, 8) G and H_parity are not orthogonal over GF(2)")

    info_all = ((np.arange(256, dtype=np.uint32)[:, None] >> np.arange(8, dtype=np.uint32)) & 1).astype(np.uint8)
    code_all = (info_all @ gen) % 2
    if np.any((code_all @ h_par.T) % 2):
        raise AssertionError("Codebook contains invalid codeword")
    d_min = int(code_all[1:].sum(axis=1).min())

    bits_4 = code_all.reshape(256, 4, 4)
    x_all = (
        PAM_GRAY_16QAM[2 * bits_4[..., 0] + bits_4[..., 1]]
        + 1j * PAM_GRAY_16QAM[2 * bits_4[..., 2] + bits_4[..., 3]]
    ).astype(np.complex128)

    return {
        "n": 16,
        "k": 8,
        "d_min": d_min,
        "G": gen,
        "H_parity": h_par,
        "info_all": info_all,
        "code_all": code_all,
        "x_all": x_all,
    }


def _mini_bp_decode(llrs: np.ndarray, h_par: np.ndarray, iterations: int = 12, alpha: float = 0.75) -> np.ndarray:
    """Normalized min-sum belief propagation decoder on the (16, 8) parity-check matrix."""
    # Convention: input llrs > 0 favors bit 1; BP internal metric < 0 favors bit 1.
    ch = -np.asarray(llrs, dtype=np.float64).reshape(16)
    rows, cols = np.nonzero(h_par)
    n_checks = h_par.shape[0]
    v2c = ch[cols].copy()
    decided = (ch < 0.0).astype(np.uint8)

    check_edges = [np.flatnonzero(rows == r) for r in range(n_checks)]
    for _ in range(iterations):
        c2v = np.empty_like(v2c)
        for edges in check_edges:
            vals = v2c[edges]
            signs = np.where(vals < 0.0, -1.0, 1.0)
            mags = np.abs(vals)
            prod_s = float(np.prod(signs))
            min_pos = int(np.argmin(mags))
            min1 = float(mags[min_pos])
            if mags.size > 1:
                tmp = mags.copy()
                tmp[min_pos] = math.inf
                min2 = float(np.min(tmp))
            else:
                min2 = min1
            for local_i, e_idx in enumerate(edges):
                mag = min2 if local_i == min_pos else min1
                c2v[e_idx] = alpha * (prod_s * signs[local_i]) * mag

        post = ch.copy()
        np.add.at(post, cols, c2v)
        hard = (post < 0.0).astype(np.uint8)
        decided = hard
        if not np.any((hard @ h_par.T) % 2):
            break
        v2c = post[cols] - c2v
    return decided


def _mini_lmmse_demap(y: np.ndarray, h_mat: np.ndarray, n0: float) -> np.ndarray:
    """Exact Bayesian LMMSE equalization + Gray 16-QAM soft bit LLRs (positive = bit 1)."""
    gram = h_mat.conj().T @ h_mat + float(n0) * np.eye(4, dtype=np.complex128)
    gram_inv = np.linalg.inv(gram)
    x_hat = gram_inv @ (h_mat.conj().T @ y)
    post_var = np.clip(np.real(float(n0) * np.diag(gram_inv)), 1e-12, 1.0)
    gain = 1.0 - post_var
    axes = np.stack((x_hat.real, x_hat.imag), axis=-1)
    logp = (2.0 * axes[..., None] * PAM_GRAY_16QAM - gain[:, None, None] * (PAM_GRAY_16QAM ** 2)) / post_var[:, None, None]
    sign_llr = np.logaddexp(logp[..., 2], logp[..., 3]) - np.logaddexp(logp[..., 0], logp[..., 1])
    inner_llr = np.logaddexp(logp[..., 1], logp[..., 3]) - np.logaddexp(logp[..., 0], logp[..., 2])
    return np.stack((sign_llr, inner_llr), axis=-1).reshape(16)


def exhaustive_ml_verify_mini_code(
    *,
    n_trials: int = 200,
    snr_db: float = 6.0,
    seed: int = 91001608,
    eps_tie: float = 1e-10,
    bp_iters: int = 12,
) -> dict:
    """Exhaustively verify ML lower-bound certificate on a (16, 8, 5) code over 4x4 time-varying channels.

    Proves across `n_trials` random noisy realizations:
      1. Whenever `exists valid c' != c_true` with `||y - H x(c')||^2 < ||y - H x(c_true)||^2 - eps_tie`,
         exact exhaustive 256-codeword ML 100% errs (0 false positives!).
      2. `cert_rate <= ml_bler <= decoder_bler`.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    code = mini_code_16_8()
    h_par = code["H_parity"]
    info_all = code["info_all"]
    code_all = code["code_all"]
    x_all = code["x_all"]

    n0 = n0_from_snr(snr_db)
    rng = np.random.default_rng(int(seed))
    mini_delays = np.array([0, 1, 2], dtype=np.int64)
    times = np.arange(4, dtype=np.float64) / 4.0

    subset_cert_hits = 0
    neighbors_cert_hits = 0
    full_cert_hits = 0
    ml_errors = 0
    decoder_errors = 0
    subset_false_positives = 0
    neighbors_false_positives = 0
    full_false_positives = 0
    max_conv_err = 0.0

    for _ in range(n_trials):
        # Random 3-ray time-varying channel over 4 samples
        gains = np.sqrt(POWERS) * (rng.normal(size=3) + 1j * rng.normal(size=3)) / math.sqrt(2.0)
        gains[0] = math.sqrt(float(POWERS[0])) * np.exp(1j * rng.uniform(-math.pi, math.pi))
        freqs = rng.uniform(0.045, 0.080, size=3) * rng.choice([-1.0, 1.0], size=3)
        chirps = rng.uniform(-0.002, 0.002, size=3)

        h_mat = np.zeros((4, 4), dtype=np.complex128)
        g_t = np.empty((3, 4), dtype=np.complex128)
        for l_idx in range(3):
            d = int(mini_delays[l_idx])
            g_l = gains[l_idx] * np.exp(2j * math.pi * (freqs[l_idx] * times + 0.5 * chirps[l_idx] * (times ** 2)))
            g_t[l_idx] = g_l
            for n_idx in range(4):
                h_mat[n_idx, (n_idx - d) % 4] += g_l[n_idx]

        j_true = int(rng.integers(0, 256))
        u_true = info_all[j_true]
        c_true = code_all[j_true]
        x_true = x_all[j_true]

        # Independent sample-by-sample circular convolution cross-check against H @ x
        y_clean_direct = np.zeros(4, dtype=np.complex128)
        for n_idx in range(4):
            acc = 0.0 + 0.0j
            for l_idx in range(3):
                acc += g_t[l_idx, n_idx] * x_true[(n_idx - int(mini_delays[l_idx])) % 4]
            y_clean_direct[n_idx] = acc
        y_clean_mat = h_mat @ x_true
        max_conv_err = max(max_conv_err, float(np.max(np.abs(y_clean_mat - y_clean_direct))))

        noise = math.sqrt(n0) * (rng.normal(size=4) + 1j * rng.normal(size=4)) / math.sqrt(2.0)
        y = y_clean_mat + noise

        # Exhaustive exact ML over ALL 256 codewords
        resid = y[None, :] - (x_all @ h_mat.T)
        dists = np.sum(resid.real ** 2 + resid.imag ** 2, axis=1)
        d_true = float(dists[j_true])
        j_ml = int(np.argmin(dists))
        ml_err = bool(j_ml != j_true)

        # Runnable suboptimal decoder: LMMSE + Gray 16-QAM soft demapper + normalized min-sum BP
        llrs = _mini_lmmse_demap(y, h_mat, n0)
        c_dec = _mini_bp_decode(llrs, h_par, iterations=bp_iters)
        dec_err = bool(not np.array_equal(c_dec[:8], u_true))

        # Practical certificate candidates: 8 single-info-bit flips + valid runnable decoder output
        tie_guard = float(eps_tie) * max(1.0, float(np.vdot(y, y).real))
        neighbor_indices = [j_true ^ (1 << bit) for bit in range(8)]
        cand_indices = list(neighbor_indices)
        c_dec_valid = not np.any((c_dec @ h_par.T) % 2)
        if c_dec_valid and not np.array_equal(c_dec, c_true):
            j_dec = int(np.sum(c_dec[:8].astype(np.int64) << np.arange(8, dtype=np.int64)))
            if j_dec not in cand_indices:
                cand_indices.append(j_dec)

        neighbors_cert = any(float(dists[jc]) < d_true - tie_guard for jc in neighbor_indices)
        subset_cert = any(float(dists[jc]) < d_true - tie_guard for jc in cand_indices)

        # Full codebook strict certificate check across all 255 competing codewords
        other_dists = np.delete(dists, j_true)
        full_cert = bool(float(np.min(other_dists)) < d_true - tie_guard)

        if neighbors_cert and not ml_err:
            neighbors_false_positives += 1
        if subset_cert and not ml_err:
            subset_false_positives += 1
        if full_cert and not ml_err:
            full_false_positives += 1

        neighbors_cert_hits += int(neighbors_cert)
        subset_cert_hits += int(subset_cert)
        full_cert_hits += int(full_cert)
        ml_errors += int(ml_err)
        decoder_errors += int(dec_err)

    cert_rate = subset_cert_hits / float(n_trials)
    neighbors_cert_rate = neighbors_cert_hits / float(n_trials)
    full_cert_rate = full_cert_hits / float(n_trials)
    ml_bler = ml_errors / float(n_trials)
    decoder_bler = decoder_errors / float(n_trials)

    return {
        "code": {"n": 16, "k": 8, "d_min": code["d_min"], "codewords": 256, "qam_symbols": 4},
        "n_trials": int(n_trials),
        "snr_db": float(snr_db),
        "n0": float(n0),
        "seed": int(seed),
        "eps_tie": float(eps_tie),
        "max_conv_err": float(max_conv_err),
        "neighbors_cert_hits": int(neighbors_cert_hits),
        "subset_cert_hits": int(subset_cert_hits),
        "full_cert_hits": int(full_cert_hits),
        "ml_errors": int(ml_errors),
        "decoder_errors": int(decoder_errors),
        "neighbors_cert_rate": float(neighbors_cert_rate),
        "cert_rate": float(cert_rate),
        "full_cert_rate": float(full_cert_rate),
        "ml_bler": float(ml_bler),
        "decoder_bler": float(decoder_bler),
        "neighbors_false_positives": int(neighbors_false_positives),
        "false_positives": int(subset_false_positives),
        "full_false_positives": int(full_false_positives),
        "inequality_holds": bool(
            neighbors_cert_rate <= cert_rate <= full_cert_rate <= ml_bler <= decoder_bler
        ),
        "soundness_verified": bool(
            neighbors_false_positives == 0 and subset_false_positives == 0 and full_false_positives == 0
        ),
    }


# ---------------------------------------------------------------------------
# Firewall & Anti-Leakage Static / Runtime Checkers
# ---------------------------------------------------------------------------

def _as_exact_int_list(seeds: Iterable[Any], pool_name: str) -> list[int]:
    out: list[int] = []
    for item in seeds:
        if isinstance(item, (bool, np.bool_)):
            raise TypeError(f"Pool {pool_name!r} contains boolean instead of integer seed: {item!r}")
        if isinstance(item, (int, np.integer)):
            out.append(int(item))
        else:
            raise TypeError(f"Pool {pool_name!r} contains non-integer seed: {item!r} ({type(item).__name__})")
    return out


def check_seed_disjointness(
    train_seeds: Iterable[int],
    dev_seeds: Iterable[int],
    formal_id_seeds: Iterable[int],
    formal_ood_seeds: Iterable[int],
    *,
    extra_pools: Optional[Mapping[str, Iterable[int]]] = None,
    strict: bool = False,
) -> dict:
    """Verify strict pairwise disjointness and intra-pool uniqueness across all seed pools."""
    pools_raw: dict[str, Iterable[int]] = {
        "train": train_seeds,
        "dev": dev_seeds,
        "formal_id": formal_id_seeds,
        "formal_ood": formal_ood_seeds,
    }
    if extra_pools is not None:
        for name, values in extra_pools.items():
            if name in pools_raw:
                raise ValueError(f"Duplicate extra pool name: {name!r}")
            pools_raw[name] = values

    pool_lists = {name: _as_exact_int_list(vals, name) for name, vals in pools_raw.items()}
    pool_sets: dict[str, set[int]] = {}
    intra_duplicates: dict[str, list[int]] = {}

    for name, items in pool_lists.items():
        seen: set[int] = set()
        dups: set[int] = set()
        for val in items:
            if val in seen:
                dups.add(val)
            seen.add(val)
        pool_sets[name] = seen
        if dups:
            intra_duplicates[name] = sorted(dups)

    names = list(pool_sets.keys())
    pairwise_overlaps: dict[str, list[int]] = {}
    total_overlap = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a_name, b_name = names[i], names[j]
            inter = sorted(pool_sets[a_name] & pool_sets[b_name])
            if inter:
                pairwise_overlaps[f"{a_name}__{b_name}"] = inter
                total_overlap += len(inter)

    disjoint = (total_overlap == 0) and (len(intra_duplicates) == 0)
    report = {
        "disjoint": disjoint,
        "counts": {name: len(items) for name, items in pool_lists.items()},
        "unique_counts": {name: len(s) for name, s in pool_sets.items()},
        "intra_pool_duplicates": intra_duplicates,
        "pairwise_overlaps": pairwise_overlaps,
        "total_overlap_count": int(total_overlap),
    }
    if strict and not disjoint:
        raise ValueError(
            f"Seed disjointness violation: overlaps={pairwise_overlaps}, duplicates={intra_duplicates}"
        )
    return report


def _try_eval_const_float(node: ast.AST) -> Optional[float]:
    """Safely evaluate pure numeric constant AST expressions (no names/calls except pow)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _try_eval_const_float(node.operand)
        if val is None:
            return None
        return val if isinstance(node.op, ast.UAdd) else -val
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
        left = _try_eval_const_float(node.left)
        right = _try_eval_const_float(node.right)
        if left is None or right is None:
            return None
        try:
            if isinstance(node.op, ast.Add):
                return float(left + right)
            if isinstance(node.op, ast.Sub):
                return float(left - right)
            if isinstance(node.op, ast.Mult):
                return float(left * right)
            if isinstance(node.op, ast.Div):
                return None if right == 0.0 else float(left / right)
            if isinstance(node.op, ast.Pow):
                if abs(right) > 100.0:
                    return None
                return float(left ** right)
        except (OverflowError, ValueError, ZeroDivisionError):
            return None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "pow" and len(node.args) == 2:
        left = _try_eval_const_float(node.args[0])
        right = _try_eval_const_float(node.args[1])
        if left is not None and right is not None and abs(right) <= 100.0:
            try:
                return float(left ** right)
            except (OverflowError, ValueError):
                return None
    return None


def _matches_legacy_p8_n0(val: Optional[float]) -> bool:
    if val is None or not math.isfinite(val):
        return False
    legacy_ref = n0_from_snr(9.0)
    return abs(val - legacy_ref) <= 1e-4 * legacy_ref


def check_no_hardcoded_n0(
    module_paths: Iterable[str | Path],
    *,
    strict: bool = False,
) -> dict:
    """AST static checker ensuring zero hardcoded N0 constants, legacy p8 N0 imports, or fixed N0 defaults.

    Flags:
      1. `from p8.link import N0` / `from p8c.link import N0` / `from p8.link import noise`.
      2. Attribute access `*.N0` (e.g. `ofdm.N0`, `link.N0`).
      3. Constant assignment `N0 = <const>` / `DEFAULT_N0 = <const>` / `FIXED_N0 = <const>`.
      4. Function default argument `n0=<const>` or `N0=<const>`.
      5. Numeric literal or pure constant power expression equal to P8's fixed 9 dB N0 (~0.12589254).
    """
    hardcoded_var_names = {"N0", "DEFAULT_N0", "FIXED_N0", "HARDCODED_N0"}
    violations: list[dict] = []
    checked: list[str] = []

    for raw_path in module_paths:
        path = Path(raw_path)
        checked.append(str(path))
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))

        # Track aliases imported from p8.link / p8c.link to catch `ofdm.noise(...)`
        p8_link_aliases: set[str] = set()

        for node in ast.walk(tree):
            # 1. ImportFrom checks
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod in ("p8.link", "p8c.link") or (mod == "p8" and any(a.name == "link" for a in node.names)):
                    for alias in node.names:
                        if mod in ("p8.link", "p8c.link") and alias.name == "N0":
                            violations.append({
                                "file": str(path),
                                "line": int(node.lineno),
                                "kind": "legacy_n0_import",
                                "detail": f"from {mod} import N0",
                            })
                        if mod == "p8.link" and alias.name == "noise":
                            violations.append({
                                "file": str(path),
                                "line": int(node.lineno),
                                "kind": "legacy_hardcoded_noise_import",
                                "detail": "from p8.link import noise (uses hardcoded N0)",
                            })
                        if mod == "p8" and alias.name == "link":
                            p8_link_aliases.add(alias.asname or alias.name)
                elif mod == "p8c" and any(a.name == "link" for a in node.names):
                    for alias in node.names:
                        if alias.name == "link":
                            p8_link_aliases.add(alias.asname or alias.name)

            # 2. Attribute access `*.N0` or `<p8_link>.noise(...)`
            elif isinstance(node, ast.Attribute):
                if node.attr == "N0":
                    violations.append({
                        "file": str(path),
                        "line": int(node.lineno),
                        "kind": "legacy_n0_attribute",
                        "detail": f"Attribute access .{node.attr}",
                    })
                elif (
                    node.attr == "noise"
                    and isinstance(node.value, ast.Name)
                    and node.value.id in p8_link_aliases
                ):
                    violations.append({
                        "file": str(path),
                        "line": int(node.lineno),
                        "kind": "legacy_hardcoded_noise_attr",
                        "detail": f"{node.value.id}.noise uses hardcoded P8 N0",
                    })

            # 3. Constant assignment to N0 / DEFAULT_N0 / FIXED_N0
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                rhs = node.value
                if rhs is not None:
                    const_val = _try_eval_const_float(rhs)
                    for tgt in targets:
                        if isinstance(tgt, ast.Name) and tgt.id in hardcoded_var_names and const_val is not None:
                            violations.append({
                                "file": str(path),
                                "line": int(node.lineno),
                                "kind": "hardcoded_n0_assignment",
                                "detail": f"{tgt.id} = {const_val!r}",
                            })

            # 4. Function default arguments `n0=<numeric constant>`
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                pos_args = list(args.posonlyargs) + list(args.args)
                if args.defaults:
                    for arg_node, default_node in zip(pos_args[-len(args.defaults):], args.defaults):
                        if arg_node.arg.lower() == "n0":
                            const_val = _try_eval_const_float(default_node)
                            if const_val is not None:
                                violations.append({
                                    "file": str(path),
                                    "line": int(default_node.lineno),
                                    "kind": "hardcoded_n0_default",
                                    "detail": f"def {node.name}(..., {arg_node.arg}={const_val!r})",
                                })
                for arg_node, default_node in zip(args.kwonlyargs, args.kw_defaults):
                    if default_node is not None and arg_node.arg.lower() == "n0":
                        const_val = _try_eval_const_float(default_node)
                        if const_val is not None:
                            violations.append({
                                "file": str(path),
                                "line": int(default_node.lineno),
                                "kind": "hardcoded_n0_default",
                                "detail": f"def {node.name}(*, {arg_node.arg}={const_val!r})",
                            })

            # 5. Bare magic float literal or constant power expression matching 10**(-0.9)
            elif isinstance(node, ast.Constant) and isinstance(node.value, float):
                if _matches_legacy_p8_n0(node.value):
                    violations.append({
                        "file": str(path),
                        "line": int(node.lineno),
                        "kind": "hardcoded_n0_literal",
                        "detail": f"Numeric literal {node.value!r} matches legacy 9 dB N0",
                    })
            elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                const_val = _try_eval_const_float(node)
                if _matches_legacy_p8_n0(const_val):
                    violations.append({
                        "file": str(path),
                        "line": int(node.lineno),
                        "kind": "hardcoded_n0_power_expr",
                        "detail": f"Constant power expression evaluates to legacy 9 dB N0 ({const_val:.12g})",
                    })

    # Sort deterministically by (file, line, kind, detail) and deduplicate exact repeats
    seen_keys: set[tuple[str, int, str, str]] = set()
    unique_violations: list[dict] = []
    for v in sorted(violations, key=lambda d: (d["file"], d["line"], d["kind"], d["detail"])):
        key = (v["file"], v["line"], v["kind"], v["detail"])
        if key not in seen_keys:
            seen_keys.add(key)
            unique_violations.append(v)

    report = {
        "clean": len(unique_violations) == 0,
        "files_checked": checked,
        "violation_count": len(unique_violations),
        "violations": unique_violations,
    }
    if strict and unique_violations:
        raise ValueError(f"Hardcoded N0 violations detected: {unique_violations}")
    return report


def verify_public_observation(obs: Any, *, strict: bool = False) -> dict:
    """Verify strict PublicObservation schema, probe-row NaN masking, and zero hidden truth fields."""
    errors: list[str] = []

    if not isinstance(obs, PublicObservation):
        errors.append(f"Expected PublicObservation instance, got {type(obs).__name__}")

    # Check for forbidden hidden attributes on the object / __dict__
    present_attrs = set(getattr(obs, "__dict__", {}).keys())
    forbidden_present = sorted(present_attrs & FORBIDDEN_HIDDEN_ATTRS)
    if forbidden_present:
        errors.append(f"Forbidden hidden truth attributes attached: {forbidden_present}")
    unexpected_attrs = sorted(present_attrs - ALLOWED_PUBLIC_ATTRS - FORBIDDEN_HIDDEN_ATTRS)
    if unexpected_attrs:
        errors.append(f"Unexpected non-contract attributes attached: {unexpected_attrs}")

    if not errors:
        y_a = np.asarray(obs.y_a)
        y_b = np.asarray(obs.y_b)
        if y_a.shape != (T, N) or y_a.dtype != np.complex128:
            errors.append(f"y_a must have shape {(T, N)} and dtype complex128, got {y_a.shape}/{y_a.dtype}")
        elif not np.isfinite(y_a).all():
            errors.append("y_a contains non-finite values")

        if y_b.shape != (T, N) or y_b.dtype != np.complex128:
            errors.append(f"y_b must have shape {(T, N)} and dtype complex128, got {y_b.shape}/{y_b.dtype}")
        else:
            if obs.probe_row != PROBE_ROW:
                errors.append(f"probe_row must be {PROBE_ROW}, got {obs.probe_row}")
            probe_slice = y_b[PROBE_ROW]
            if not (np.isnan(probe_slice.real).all() and np.isnan(probe_slice.imag).all()):
                errors.append("Withheld PROBE_ROW in y_b is NOT fully masked to NaN+1j*NaN")
            non_probe = np.delete(y_b, PROBE_ROW, axis=0)
            if not np.isfinite(non_probe).all():
                errors.append("Non-probe rows in y_b contain non-finite values")

        if obs.budget not in SC_LAYOUTS:
            errors.append(f"Unknown budget {obs.budget!r}")
        else:
            if tuple(obs.training_rows) != SC_LAYOUTS[obs.budget]:
                errors.append(
                    f"training_rows {obs.training_rows!r} does not match SC_LAYOUTS[{obs.budget!r}]={SC_LAYOUTS[obs.budget]!r}"
                )
        if PROBE_ROW in tuple(obs.training_rows):
            errors.append("PROBE_ROW appears inside training_rows")

        if not (isinstance(obs.n0, (int, float, np.floating)) and not isinstance(obs.n0, bool)):
            errors.append(f"n0 must be a real float, got {type(obs.n0).__name__}")
        elif not (math.isfinite(float(obs.n0)) and float(obs.n0) > 0.0):
            errors.append(f"n0 must be positive finite, got {obs.n0!r}")

    report = {"valid": len(errors) == 0, "errors": errors}
    if strict and errors:
        raise ValueError(f"PublicObservation firewall check failed: {errors}")
    return report


def _extract_receiver_outputs(out: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize either GateOutput or (llrs, probe_pred, probe_var) tuple for anti-leakage auditing."""
    if hasattr(out, "mix_llrs") and hasattr(out, "mix_probe_pred") and hasattr(out, "mix_probe_var"):
        return (
            np.asarray(out.mix_llrs, dtype=np.float64),
            np.asarray(out.mix_probe_pred, dtype=np.complex128),
            np.asarray(out.mix_probe_var, dtype=np.float64),
        )
    if isinstance(out, tuple) and len(out) == 3:
        return (
            np.asarray(out[0], dtype=np.float64),
            np.asarray(out[1], dtype=np.complex128),
            np.asarray(out[2], dtype=np.float64),
        )
    raise TypeError(
        "receiver_fn must return GateOutput or (llrs, probe_pred, probe_var) tuple"
    )


def audit_receiver_anti_leakage(
    receiver_fn: Callable[[PublicObservation], Any],
    obs: PublicObservation,
    *,
    tol: float = 1e-13,
    check_payload_probe_isolation: bool = True,
    strict: bool = False,
) -> dict:
    """Blind probe-poisoning & payload-isolation anti-leakage audit for any P9 receiver/gate.

    1. Verifies `obs` passes `verify_public_observation`.
    2. Runs receiver on NaN-masked `obs` and verifies all outputs are finite and `probe_var >= n0 - 1e-12`.
    3. Poisons `y_b[probe_row]` with two wildly different finite sentinels (+1e6-1e6j and -5e5+3e5j):
       a non-leaking receiver MUST return outputs identical to the NaN-masked run (within `tol`)!
    4. Optionally perturbs `CORE_ROWS` in `y_b` to check that probe predictions do not leak payload cells.
    """
    obs_check = verify_public_observation(obs, strict=strict)
    issues: list[str] = list(obs_check["errors"])

    base_llrs, base_pred, base_var = _extract_receiver_outputs(receiver_fn(obs))
    if not (np.isfinite(base_llrs).all() and np.isfinite(base_pred).all() and np.isfinite(base_var).all()):
        issues.append("Receiver produced non-finite output on NaN-masked PublicObservation (probe read leak!)")
    if np.isfinite(base_var).all() and np.any(base_var < float(obs.n0) - 1e-12):
        issues.append(f"probe_var fell below noise floor n0={obs.n0}: min={float(np.min(base_var))}")

    max_probe_poison_diff = 0.0
    for sentinel in (1e6 - 1e6j, -5e5 + 3e5j):
        y_poison = obs.y_b.copy()
        y_poison[obs.probe_row] = sentinel
        obs_poison = PublicObservation(
            y_a=obs.y_a.copy(),
            y_b=y_poison,
            training_rows=obs.training_rows,
            probe_row=obs.probe_row,
            n0=obs.n0,
            budget=obs.budget,
        )
        p_llrs, p_pred, p_var = _extract_receiver_outputs(receiver_fn(obs_poison))
        if not (np.isfinite(p_llrs).all() and np.isfinite(p_pred).all() and np.isfinite(p_var).all()):
            issues.append("Receiver produced non-finite output under probe poisoning")
            max_probe_poison_diff = math.inf
        elif np.isfinite(base_llrs).all() and np.isfinite(base_pred).all() and np.isfinite(base_var).all():
            diff = max(
                float(np.max(np.abs(p_llrs - base_llrs))),
                float(np.max(np.abs(p_pred - base_pred))),
                float(np.max(np.abs(p_var - base_var))),
            )
            max_probe_poison_diff = max(max_probe_poison_diff, diff)
            if diff > tol:
                issues.append(f"Probe-row leak detected: output changed by {diff:.3e} under probe sentinel")

    max_payload_probe_diff = 0.0
    if check_payload_probe_isolation:
        y_perturbed = obs.y_b.copy()
        y_perturbed[list(CORE_ROWS)] = 40.0 - 60.0j
        obs_perturbed = PublicObservation(
            y_a=obs.y_a.copy(),
            y_b=y_perturbed,
            training_rows=obs.training_rows,
            probe_row=obs.probe_row,
            n0=obs.n0,
            budget=obs.budget,
        )
        _, q_pred, q_var = _extract_receiver_outputs(receiver_fn(obs_perturbed))
        if np.isfinite(base_pred).all() and np.isfinite(q_pred).all():
            max_payload_probe_diff = max(
                float(np.max(np.abs(q_pred - base_pred))),
                float(np.max(np.abs(q_var - base_var))),
            )
            if max_payload_probe_diff > tol:
                issues.append(
                    f"Payload-to-probe coupling detected: probe output changed by {max_payload_probe_diff:.3e}"
                )

    report = {
        "passed": len(issues) == 0,
        "issues": issues,
        "max_probe_poison_diff": float(max_probe_poison_diff),
        "max_payload_probe_diff": float(max_payload_probe_diff),
    }
    if strict and issues:
        raise ValueError(f"Anti-leakage audit failed: {issues}")
    return report
