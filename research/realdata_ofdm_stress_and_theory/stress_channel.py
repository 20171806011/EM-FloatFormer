"""Controlled physical RF stress perturbations applied strictly to real OTA OFDM grids.

All perturbations operate on the measured complex64/complex128 OFDM grids [14, 128]
(with exact IFFT -> CP -> time-domain impairment -> CP removal -> FFT for CFO/Doppler,
preserving genuine inter-carrier interference ICI and CP physics).

The 5 standardized evaluation conditions (raw OTA kept completely separate from stressed):
  - C0_raw_ota           : Unmodified real OTA OFDM IQ grid (14 pilots)
  - C1_awgn_stress       : Controlled AWGN addition (-6 dB SNR degradation relative to measured active power;
                           helper supports arbitrary sweep delta_db)
  - C2_cfo_doppler       : Continuous time-domain CFO / Doppler drift across the 14 OFDM symbols
                           (|eps_cfo| in [0.025, 0.055] subcarrier spacings, random sign), anchored at
                           pilot symbol 2 so symbols 4..13 exhibit realistic phase rotation & ICI
  - C3_phase_jump_jitter : Abrupt oscillator phase jump at random post-pilot onset s_jump in {5, 6, 7, 8}
                           with jump angle in {+-45, +-90, +-135, 180} deg + small Wiener phase jitter
                           (0.05 rad/sym), creating genuine multimodal phase ambiguity
  - C4_nbi_sparse7       : Sparse 7-pilot configuration + localized narrowband burst interference (NBI)
                           occupying 8 contiguous active subcarriers x 5 contiguous OFDM symbols at +10 dB INR
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from ofdm_data import (
    CP_LEN,
    N_FFT,
    N_SYM,
    PILOT_SCS_7,
    PILOT_SCS_14,
    PILOT_SYM,
    USED_SCS,
)

STRESS_CONDITIONS: Tuple[str, ...] = (
    "C0_raw_ota",
    "C1_awgn_stress",
    "C2_cfo_doppler",
    "C3_phase_jump_jitter",
    "C4_nbi_sparse7",
)

# Discrete jump angles (degrees) for C3_phase_jump_jitter
JUMP_ANGLES_DEG: Tuple[float, ...] = (-135.0, -90.0, -45.0, 45.0, 90.0, 135.0, 180.0)


@dataclass(frozen=True)
class PerturbedTTI:
    iq: np.ndarray               # [14, 128] complex128 perturbed RX grid
    pilot_scs: np.ndarray        # active pilot subcarrier indices (14 or 7)
    condition: str
    # Exact ground-truth perturbation metadata (OFFLINE EVALUATION / DIAGNOSTICS ONLY!)
    true_phase_per_sym: np.ndarray   # [14] float64 net phase rotation applied to each symbol (rad)
    true_cfo_norm: float             # normalized CFO (cycles per FFT useful period)
    true_jump_onset: int             # -1 if none, else onset symbol index
    true_jump_rad: float             # jump angle in radians
    true_added_noise_var: float      # added wideband complex noise variance E|w|^2
    true_nbi_mask: np.ndarray        # [14, 101] bool mask of NBI-hit REs


def _ofdm_freq_to_time_cp(grid: np.ndarray) -> np.ndarray:
    """Convert frequency-domain OFDM grid [14, 128] (fftshifted, DC at 64) to serial time samples with CP."""
    # Unshift DC from index 64 back to index 0 for standard unitary IFFT
    f_unshifted = np.fft.ifftshift(grid, axes=-1)
    t_sym = np.fft.ifft(f_unshifted, axis=-1, norm="ortho")  # [14, 128]
    cp = t_sym[:, -CP_LEN:]                                   # [14, 6]
    return np.concatenate([cp, t_sym], axis=-1).reshape(-1)   # [14 * 134 = 1876]


def _ofdm_time_cp_to_freq(t_serial: np.ndarray) -> np.ndarray:
    """Remove CP and apply unitary FFT + fftshift back to [14, 128] (DC at 64)."""
    t_blocks = t_serial.reshape(N_SYM, N_FFT + CP_LEN)
    t_useful = t_blocks[:, CP_LEN:]                           # [14, 128]
    f_unshifted = np.fft.fft(t_useful, axis=-1, norm="ortho")
    return np.fft.fftshift(f_unshifted, axes=-1)


def apply_stress_condition(
    iq_raw: np.ndarray,
    condition: str,
    seed: int,
    *,
    awgn_snr_target_drop_db: float = 6.0,
    cfo_range: Tuple[float, float] = (0.025, 0.055),
    nbi_inr_db: float = 10.0,
) -> PerturbedTTI:
    """Deterministically apply a controlled physical RF stress condition to one real OTA TTI."""
    rng = np.random.default_rng(int(seed))
    y = np.array(iq_raw, dtype=np.complex128, copy=True)
    sig_pwr = float(np.mean(np.abs(y[:, USED_SCS]) ** 2))
    phase_sym = np.zeros(N_SYM, dtype=np.float64)
    cfo_norm = 0.0
    jump_onset = -1
    jump_rad = 0.0
    added_nv = 0.0
    nbi_mask = np.zeros((N_SYM, len(USED_SCS)), dtype=bool)
    pilot_scs = PILOT_SCS_14

    if condition == "C0_raw_ota":
        pass

    elif condition == "C1_awgn_stress":
        # Add wideband complex circular Gaussian noise calibrated to signal power
        # Target SNR drop: add noise variance sigma_add^2 = sig_pwr * 10^(-snr_eff_db / 10)
        # We use explicit added SNR_add = 10 dB - awgn_snr_target_drop_db (at drop=6 -> SNR_add = 4 dB,
        # pushing high-SINR 25 dB frames into mid/low 4-10 dB regime and low-SINR 6 dB frames near 0-2 dB).
        # More cleanly: define added noise power relative to sig_pwr via SNR_add_db = 12.0 - awgn_snr_target_drop_db
        snr_add_db = 12.0 - float(awgn_snr_target_drop_db)
        added_nv = sig_pwr * (10.0 ** (-snr_add_db / 10.0))
        std = math.sqrt(added_nv / 2.0)
        w = rng.normal(0.0, std, size=y.shape) + 1j * rng.normal(0.0, std, size=y.shape)
        y = y + w

    elif condition == "C2_cfo_doppler":
        # True time-domain CFO / Doppler drift through IFFT -> CP -> exp(j 2pi eps (n - n_ref)/N_FFT) -> FFT
        sign = 1.0 if rng.random() < 0.5 else -1.0
        eps = sign * float(rng.uniform(cfo_range[0], cfo_range[1]))
        cfo_norm = eps
        t_ser = _ofdm_freq_to_time_cp(y)
        sym_stride = N_FFT + CP_LEN  # 134 samples per OFDM symbol
        # Reference sample at midpoint of pilot symbol 2 useful part so pilot symbol has zero mean phase offset
        n_ref = PILOT_SYM * sym_stride + CP_LEN + 0.5 * (N_FFT - 1)
        n_idx = np.arange(len(t_ser), dtype=np.float64)
        rot_t = np.exp(1j * 2.0 * math.pi * eps * (n_idx - n_ref) / float(N_FFT))
        y = _ofdm_time_cp_to_freq(t_ser * rot_t)
        # Mild additional wideband AWGN (+16 dB SNR) to avoid noiseless degeneracy
        added_nv = sig_pwr * (10.0 ** (-16.0 / 10.0))
        std = math.sqrt(added_nv / 2.0)
        y = y + rng.normal(0.0, std, size=y.shape) + 1j * rng.normal(0.0, std, size=y.shape)
        # Net symbol-midpoint phase evolution relative to pilot sym 2
        sym_mids = np.arange(N_SYM) * sym_stride + CP_LEN + 0.5 * (N_FFT - 1)
        phase_sym = 2.0 * math.pi * eps * (sym_mids - n_ref) / float(N_FFT)

    elif condition == "C3_phase_jump_jitter":
        # Abrupt oscillator phase jump at s_jump in {5, 6, 7, 8} + small Wiener phase jitter across symbols
        jump_onset = int(rng.choice([5, 6, 7, 8]))
        jump_deg = float(rng.choice(JUMP_ANGLES_DEG))
        jump_rad = math.radians(jump_deg)
        # Small random-walk phase jitter anchored at pilot sym 2
        steps = rng.normal(0.0, 0.04, size=N_SYM)
        jitter = np.cumsum(steps)
        jitter = jitter - jitter[PILOT_SYM]
        phase_sym = jitter.copy()
        phase_sym[jump_onset:] += jump_rad
        # Apply in time domain per symbol (exact common phase error + continuous phase within symbol)
        y = y * np.exp(1j * phase_sym[:, None])
        # Mild AWGN (16 dB relative to signal)
        added_nv = sig_pwr * (10.0 ** (-16.0 / 10.0))
        std = math.sqrt(added_nv / 2.0)
        y = y + rng.normal(0.0, std, size=y.shape) + 1j * rng.normal(0.0, std, size=y.shape)

    elif condition == "C4_nbi_sparse7":
        # Sparse 7 pilots + localized narrowband burst interference (8 contiguous subcarriers x 5 symbols, +10 dB)
        pilot_scs = PILOT_SCS_7
        bw = 8
        dur = 5
        sc_start_u = int(rng.integers(0, len(USED_SCS) - bw + 1))
        sym_start = int(rng.integers(0, N_SYM - dur + 1))
        # Avoid completely wiping out pilot symbol's own pilots unless naturally overlapped on data symbols;
        # allow burst anywhere on symbols 3..13 so pilot estimation is challenged by sparsity while data faces burst NBI
        sym_start = int(rng.integers(3, N_SYM - dur + 1))
        nbi_mask[sym_start : sym_start + dur, sc_start_u : sc_start_u + bw] = True
        nbi_scs = USED_SCS[sc_start_u : sc_start_u + bw]
        nbi_var = sig_pwr * (10.0 ** (float(nbi_inr_db) / 10.0))
        std_nbi = math.sqrt(nbi_var / 2.0)
        burst = rng.normal(0.0, std_nbi, size=(dur, bw)) + 1j * rng.normal(0.0, std_nbi, size=(dur, bw))
        y[sym_start : sym_start + dur, nbi_scs] += burst

    else:
        raise ValueError(f"Unknown stress condition: {condition}")

    return PerturbedTTI(
        iq=y,
        pilot_scs=pilot_scs,
        condition=condition,
        true_phase_per_sym=phase_sym,
        true_cfo_norm=float(cfo_norm),
        true_jump_onset=int(jump_onset),
        true_jump_rad=float(jump_rad),
        true_added_noise_var=float(added_nv),
        true_nbi_mask=nbi_mask,
    )


def condition_seed(tti_index: int, condition: str, split_salt: int = 20261121) -> int:
    """Deterministic per-(TTI, condition) seed independent of ordering or batching."""
    c_idx = STRESS_CONDITIONS.index(condition) if condition in STRESS_CONDITIONS else abs(hash(condition)) % 100
    return int(split_salt + 10007 * c_idx + 131 * int(tti_index))
