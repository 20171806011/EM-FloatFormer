"""Locked read-only shared interface contract for all parallel Phase 9 modules.

All parallel workstreams (P9-A channel/decoder, P9-B classical gating, P9-C neural gate,
and P9-Audit independent verifier) MUST strictly respect these dataclasses, constants,
disjoint seed namespaces, and public/hidden truth boundaries.
"""
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


# --- Physical & Frame Geometry (locked to P8-C ce2ff50 for direct comparability) ---
N = 32
CP = 16
T = 20
DELAYS = np.array([0, 2, 4], dtype=np.int64)
POWERS = np.array([0.75, 0.17, 0.08], dtype=np.float64)
FRAME_DURATION = T * (N + CP) / N         # 30.0 useful symbols
LINK_DURATION = (2 * T * (N + CP) - 1) / N
PROBE_ROW = 19
OFDM_PILOT_ROWS = (0, 9, 10, 19)
SC_LAYOUTS = {"sparse": (10,), "dense": (0, 9, 10)}
CORE_ROWS = tuple(r for r in range(T) if r not in (*SC_LAYOUTS["dense"], PROBE_ROW))  # 16 common SC codewords

# --- Explicit SNR Grid (ZERO hardcoded N0 anywhere in p9!) ---
SNR_GRID_DB = (6.0, 9.0, 12.0, 15.0)
PRIMARY_SNR_DB = 9.0


def n0_from_snr(snr_db: float) -> float:
    """Explicit complex noise variance N0 = 10^(-snr_db/10) for unit-power symbols & unit-sum path powers."""
    return float(10.0 ** (-float(snr_db) / 10.0))


# --- History Trust Regimes (P9-B & P9-C) ---
REGIMES_ID = ("continuous", "partial_shift", "full_reset")
REGIMES_OOD = ("ood_strong_shift", "ood_midframe_shift")
REGIMES_SANITY = ("donor_mismatch",)
ALL_REGIMES = REGIMES_ID + REGIMES_OOD + REGIMES_SANITY

# --- Strictly Disjoint Seed Namespaces (disjoint from P4..P8, P8-C=822611190000, p8_manifold=802611190000) ---
SEED_P9_DEV = 910000000000
SEED_P9_TRAIN = 920000000000
SEED_P9_FORMAL_ID = 930000000000
SEED_P9_FORMAL_OOD = 940000000000

# --- Baseline Provenance ---
P8C_BASELINE_COMMIT = "ce2ff50e81f6b529ec0311393da0013b0a85b801"
P8_MANIFOLD_COMMIT = "e76c61eef26151985963460b0ab03684cd562e3e"


@dataclass(frozen=True)
class PublicObservation:
    """Strictly receiver-visible observables for one OFDM(A) -> CP-SC(B) link.

    Contains ZERO hidden channel gains, ZERO Doppler/chirp truth, ZERO regime labels,
    ZERO info/code bits, and has B's withheld PROBE_ROW masked to NaN before entering any receiver.
    """
    y_a: np.ndarray              # [T, N] complex128 received OFDM Frame A (frequency domain)
    y_b: np.ndarray              # [T, N] complex128 received CP-SC Frame B (time domain, row 19 = NaN+1j*NaN)
    training_rows: tuple         # e.g. (10,) for sparse or (0, 9, 10) for dense
    probe_row: int               # 19
    n0: float                    # explicit known complex noise variance
    budget: str                  # "sparse" or "dense"


@dataclass(frozen=True)
class HiddenTruth:
    """Evaluator-only hidden ground truth (NEVER passed to receivers at inference time)."""
    regime: str                  # one of ALL_REGIMES
    channel_a: Any               # frame A physical channel parameters
    channel_b: Any               # frame B physical channel parameters (or time-varying segment)
    changed_ray: Optional[int]   # index in {0,1,2} for partial/OOD shift, else None
    delta_freq: float            # applied Doppler shift magnitude (0.0 for continuous)
    shift_row: Optional[int]     # row index in B where shift starts (0 for boundary, 6 for ood_midframe_shift)
    b_info: np.ndarray           # [18, 64] uint8 info bits ([:16] core, [16:] extra)
    b_code: np.ndarray           # [18, 128] uint8 codewords
    b_symbols: np.ndarray        # [18, 32] complex128 16-QAM symbols
    y_b_probe_noisy: np.ndarray  # [N] complex128 withheld noisy row 19
    y_b_probe_clean: np.ndarray  # [N] complex128 withheld noiseless row 19
    y_b_full_clean: np.ndarray   # [T, N] complex128 noiseless full B waveform
    h_b_true: np.ndarray         # [3, T, N] complex128 exact time-domain ray gains g_l(t, n) on Frame B


@dataclass(frozen=True)
class CandidateSummary:
    """Public pre-update & post-update physical statistics for one branch ('inherit' or 'clear')."""
    name: str                    # "inherit" or "clear"
    pre_train_loglik: float      # STRICT pre-update marginal predictive log p(y_train | candidate_prior)
    pre_train_pred: np.ndarray   # [len(training_rows), N] complex128 pre-update prediction on training rows
    pre_train_var: np.ndarray    # [len(training_rows), N] float64 pre-update predictive variance on training rows
    mean_ray: np.ndarray         # [3, T, N] complex128 post-training conditioned ray gain means on Frame B
    var_ray: np.ndarray          # [3, T, N] float64 post-training conditioned ray gain variances on Frame B
    probe_pred: np.ndarray       # [N] complex128 withheld row-19 prediction
    probe_var: np.ndarray        # [N] float64 withheld row-19 predictive variance
    llrs: np.ndarray             # [T, N, 4] float64 per-branch soft bit LLRs (positive = bit 1)


@dataclass(frozen=True)
class GateOutput:
    """Unified output of any P9-B classical gate or P9-C learnable gate."""
    history_weight: np.ndarray   # [T] float64 in [0, 1] per-row weight on 'inherit' branch (1 - w on 'clear')
    probe_weight: float          # scalar in [0, 1] weight on 'inherit' for row-19 probe prediction
    mix_llrs: np.ndarray         # [T, N, 4] float64 exact probability-space mixture bit LLRs
    mix_probe_pred: np.ndarray   # [N] complex128 mixture probe mean
    mix_probe_var: np.ndarray    # [N] float64 mixture total probe variance (within + between!)
    mix_mean_ray: np.ndarray     # [3, T, N] complex128 effective mixture ray mean (for whole-word scoring)
    mix_var_ray: np.ndarray      # [3, T, N] float64 effective mixture ray variance (within + between!)


def jeffreys_complex_gaussian(mu1: np.ndarray, var1: np.ndarray, mu2: np.ndarray, var2: np.ndarray) -> np.ndarray:
    """Exact C*-invariant symmetric Jeffreys divergence between CN(mu1, var1) and CN(mu2, var2)."""
    v1 = np.maximum(np.asarray(var1, dtype=np.float64), 1e-300)
    v2 = np.maximum(np.asarray(var2, dtype=np.float64), 1e-300)
    diff2 = np.abs(np.asarray(mu1, dtype=np.complex128) - np.asarray(mu2, dtype=np.complex128)) ** 2
    mean_term = 0.5 * diff2 * (1.0 / v1 + 1.0 / v2)
    var_term = np.maximum(0.5 * (v1 / v2 + v2 / v1 - 2.0), 0.0)
    return np.maximum(mean_term + var_term, 0.0)


def mix_bit_llrs(w_inherit: np.ndarray, llr_inh: np.ndarray, llr_clr: np.ndarray) -> np.ndarray:
    """Exact probability-space mixture of two branch bit LLRs (positive = bit 1).

    w_inherit: shape broadcastable to llr_inh (e.g. [T, 1, 1]).
    """
    w = np.clip(np.asarray(w_inherit, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    l1 = np.clip(np.asarray(llr_inh, dtype=np.float64), -40.0, 40.0)
    l0 = np.clip(np.asarray(llr_clr, dtype=np.float64), -40.0, 40.0)
    p1 = 1.0 / (1.0 + np.exp(-l1))
    p0 = 1.0 / (1.0 + np.exp(-l0))
    pm = np.clip(w * p1 + (1.0 - w) * p0, 1e-15, 1.0 - 1e-15)
    return np.log(pm / (1.0 - pm))
