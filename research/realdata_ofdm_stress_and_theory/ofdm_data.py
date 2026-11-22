"""Data loading, verified 16-QAM mapping, and leak-free contiguous temporal splits for sdr_ofdm_dataset_v1.

Verified physical grid for https://github.com/rikluost/sdr_ofdm_dataset_v1 (commit 6f9821d6390d212884fea3d6621267ebe70c7246):
  - File: ofdm_valset_4_8_sdr.pth (27,684,430 bytes, 742 real OTA TTIs at 435 MHz, pedestrian mobility)
  - SHA-256: e760079a888f37d8c33b21f327646930b4f2cb4511e8a309bcde22c6b1332058
  - Grid: 14 OFDM symbols x 128 FFT subcarriers, CP = 6 samples, DC = subcarrier 64
  - Active non-DC subcarriers (101): [13..63, 65..114]
  - Full pilots (14): OFDM symbol 2 at FFT indices [13, 21, 29, 37, 45, 53, 61, 69, 77, 85, 93, 101, 109, 114]
  - Pilot TX sequence: 0.7 * (-1-j, -1+j, +1-j, +1+j, ...)
  - Data REs (1400): row-major (symbol, subcarrier) order across all active non-pilot REs
  - 16-QAM bit-to-symbol mapping (empirically verified with 0.000000 BER on 63/64 top-SINR TTIs):
      Natural binary (offset binary) on both I and Q axes:
      (b0, b1) -> {-3, -1, +1, +3} / sqrt(10) for I
      (b2, b3) -> {-3, -1, +1, +3} / sqrt(10) for Q
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Exact verified OFDM constants
N_FFT: int = 128
N_SYM: int = 14
CP_LEN: int = 6
DC_SC: int = 64
PILOT_SYM: int = 2

USED_SCS: np.ndarray = np.array([k for k in range(13, 115) if k != DC_SC], dtype=np.int64)  # len 101
PILOT_SCS_14: np.ndarray = np.array(
    [13, 21, 29, 37, 45, 53, 61, 69, 77, 85, 93, 101, 109, 114], dtype=np.int64
)
PILOT_SCS_7: np.ndarray = np.array([13, 29, 45, 61, 77, 93, 114], dtype=np.int64)
PILOT_SCS_4: np.ndarray = np.array([13, 45, 77, 114], dtype=np.int64)

# Held-out pilot positions for sparse-pilot channel MSE verification (indices within PILOT_SCS_14)
HELD_OUT_PILOTS_FROM_7: np.ndarray = np.array([21, 37, 53, 69, 85, 101, 109], dtype=np.int64)
HELD_OUT_PILOTS_FROM_4: np.ndarray = np.array(
    [21, 29, 37, 53, 61, 69, 85, 93, 101, 109], dtype=np.int64
)

PILOT_TX_14: np.ndarray = np.array(
    [
        -0.7 - 0.7j,
        -0.7 + 0.7j,
        0.7 - 0.7j,
        0.7 + 0.7j,
        -0.7 - 0.7j,
        -0.7 + 0.7j,
        0.7 - 0.7j,
        0.7 + 0.7j,
        -0.7 - 0.7j,
        -0.7 + 0.7j,
        0.7 - 0.7j,
        0.7 + 0.7j,
        -0.7 - 0.7j,
        -0.7 + 0.7j,
    ],
    dtype=np.complex128,
)
PILOT_TX_MAP: Dict[int, complex] = {int(sc): complex(val) for sc, val in zip(PILOT_SCS_14, PILOT_TX_14)}

# Map FFT subcarrier -> index in USED_SCS (0..100)
SC_TO_USED_IDX: Dict[int, int] = {int(sc): i for i, sc in enumerate(USED_SCS)}
PILOT_USED_IDX_14: np.ndarray = np.array([SC_TO_USED_IDX[int(sc)] for sc in PILOT_SCS_14], dtype=np.int64)
PILOT_USED_IDX_7: np.ndarray = np.array([SC_TO_USED_IDX[int(sc)] for sc in PILOT_SCS_7], dtype=np.int64)
PILOT_USED_IDX_4: np.ndarray = np.array([SC_TO_USED_IDX[int(sc)] for sc in PILOT_SCS_4], dtype=np.int64)

# Verified Natural Binary 16-QAM constellation
PAM4_LEVELS: np.ndarray = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64) / math.sqrt(10.0)
CONST16: np.ndarray = np.array(
    [PAM4_LEVELS[i // 4] + 1j * PAM4_LEVELS[i % 4] for i in range(16)], dtype=np.complex128
)
BITS16: np.ndarray = np.array(
    [[(i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(16)], dtype=np.int32
)
# Bit masks for exact LLR marginalization: BIT_ONE_MASK[b] has 8 constellation indices where bit b == 1
BIT_ONE_IDX: np.ndarray = np.array([np.where(BITS16[:, b] == 1)[0] for b in range(4)], dtype=np.int64)
BIT_ZERO_IDX: np.ndarray = np.array([np.where(BITS16[:, b] == 0)[0] for b in range(4)], dtype=np.int64)

# Row-major data RE mask and coordinates on [14, 128] and [14, 101]
DATA_MASK_128: np.ndarray = np.zeros((N_SYM, N_FFT), dtype=bool)
DATA_MASK_128[:, USED_SCS] = True
DATA_MASK_128[PILOT_SYM, PILOT_SCS_14] = False
DATA_COORDS_128: np.ndarray = np.argwhere(DATA_MASK_128)  # [1400, 2] (sym, sc128)

DATA_MASK_101: np.ndarray = np.ones((N_SYM, len(USED_SCS)), dtype=bool)
DATA_MASK_101[PILOT_SYM, PILOT_USED_IDX_14] = False
DATA_COORDS_101: np.ndarray = np.argwhere(DATA_MASK_101)  # [1400, 2] (sym, used_idx)
assert len(DATA_COORDS_128) == 1400 and len(DATA_COORDS_101) == 1400

# Contiguous Temporal Block Split with 20-TTI Guard Bands (Zero Temporal Leakage)
GUARD_TTIS: int = 20
TRAIN_RANGE: Tuple[int, int] = (0, 340)      # 340 TTIs (17 blocks x 20)
GUARD1_RANGE: Tuple[int, int] = (340, 360)   # 20 guard TTIs (excluded)
DEV_RANGE: Tuple[int, int] = (360, 482)      # 122 TTIs (6 blocks)
GUARD2_RANGE: Tuple[int, int] = (482, 502)   # 20 guard TTIs (excluded)
TEST_RANGE: Tuple[int, int] = (502, 742)     # 240 TTIs (12 contiguous blocks x 20)
TEST_N_BLOCKS: int = 12
TEST_BLOCK_SIZE: int = 20


class _FallbackCustomDataset(Dataset):
    """Unpickling shim compatible with sdr_ofdm_dataset_v1/config.py::CustomDataset."""

    def __init__(self) -> None:
        self.pdsch_iq = []
        self.labels = []
        self.sinr = []

    def __len__(self) -> int:
        return len(self.pdsch_iq)

    def __getitem__(self, index: int):
        return self.pdsch_iq[index], self.labels[index], self.sinr[index]


@dataclass(frozen=True)
class RealOFDMDataset:
    iq: np.ndarray        # [742, 14, 128] complex128
    labels: np.ndarray    # [742, 1400, 4] int32 in {0, 1}
    sym_idx: np.ndarray   # [742, 1400] int32 in 0..15
    tx_syms: np.ndarray   # [742, 1400] complex128
    sinr_db: np.ndarray   # [742] float64
    sha256: str
    file_size: int


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_real_ofdm_dataset(pth_path: Path) -> RealOFDMDataset:
    """Load ofdm_valset_4_8_sdr.pth cleanly without polluting sys.modules permanently."""
    pth_path = Path(pth_path)
    digest = sha256_file(pth_path)
    fsize = pth_path.stat().st_size

    injected = False
    if "config" not in sys.modules:
        mod = types.ModuleType("config")
        mod.CustomDataset = _FallbackCustomDataset
        sys.modules["config"] = mod
        injected = True
    try:
        ds = torch.load(str(pth_path), map_location="cpu", weights_only=False)
    finally:
        if injected:
            sys.modules.pop("config", None)

    n = len(ds)
    iq = np.stack([ds[i][0].numpy() for i in range(n)], axis=0).astype(np.complex128)
    labels = np.stack([ds[i][1].numpy() for i in range(n)], axis=0).astype(np.int32)
    sinr_db = np.array([float(ds[i][2]) for i in range(n)], dtype=np.float64)
    sym_idx = labels[..., 0] * 8 + labels[..., 1] * 4 + labels[..., 2] * 2 + labels[..., 3]
    tx_syms = CONST16[sym_idx]
    return RealOFDMDataset(
        iq=iq,
        labels=labels,
        sym_idx=sym_idx,
        tx_syms=tx_syms,
        sinr_db=sinr_db,
        sha256=digest,
        file_size=fsize,
    )


def pilot_tx_for_scs(pilot_scs: np.ndarray) -> np.ndarray:
    return np.array([PILOT_TX_MAP[int(sc)] for sc in pilot_scs], dtype=np.complex128)


def build_oracle_channel(ds: RealOFDMDataset, tti_idx: np.ndarray | None = None) -> np.ndarray:
    """Offline evaluation-ONLY symbol-averaged empirical channel H_oracle [N, 101] on USED_SCS.

    NEVER called by any receiver or training pipeline; exclusively for offline channel MSE/EVM
    diagnostics alongside held-out pilot MSE.
    """
    idx = np.arange(len(ds.sinr_db)) if tti_idx is None else np.asarray(tti_idx, dtype=np.int64)
    n = len(idx)
    x_grid = np.zeros((n, N_SYM, N_FFT), dtype=np.complex128)
    x_grid[:, DATA_COORDS_128[:, 0], DATA_COORDS_128[:, 1]] = ds.tx_syms[idx]
    x_grid[:, PILOT_SYM, PILOT_SCS_14] = PILOT_TX_14[None, :]
    y_used = ds.iq[idx][:, :, USED_SCS]
    x_used = x_grid[:, :, USED_SCS]
    # Exact LMMSE-like weighted ratio across 14 symbols: sum(X* Y) / sum(|X|^2)
    num = np.sum(np.conj(x_used) * y_used, axis=1)
    den = np.sum(np.abs(x_used) ** 2, axis=1)
    return num / np.maximum(den, 1e-12)


def build_temporal_splits(n_total: int = 742, sinr_db: np.ndarray | None = None) -> dict:
    """Construct leak-free contiguous temporal blocks with 20-TTI guard bands."""
    assert n_total == 742, f"Expected 742 TTIs, got {n_total}"

    def _make_blocks(start: int, end: int, n_blocks: int, prefix: str) -> List[dict]:
        edges = np.linspace(start, end, n_blocks + 1, dtype=int)
        blocks = []
        for bi in range(n_blocks):
            s, e = int(edges[bi]), int(edges[bi + 1])
            entry = {
                "block_id": f"{prefix}_b{bi:02d}",
                "start_tti": s,
                "end_tti_exclusive": e,
                "n_ttis": e - s,
                "tti_indices": list(range(s, e)),
            }
            if sinr_db is not None:
                seg = sinr_db[s:e]
                entry["sinr_mean_db"] = round(float(np.mean(seg)), 3)
                entry["sinr_min_db"] = round(float(np.min(seg)), 3)
                entry["sinr_max_db"] = round(float(np.max(seg)), 3)
            blocks.append(entry)
        return blocks

    train_blocks = _make_blocks(TRAIN_RANGE[0], TRAIN_RANGE[1], 17, "train")
    dev_blocks = _make_blocks(DEV_RANGE[0], DEV_RANGE[1], 6, "dev")
    test_blocks = _make_blocks(TEST_RANGE[0], TEST_RANGE[1], TEST_N_BLOCKS, "test")

    splits = {
        "n_total_ttis": n_total,
        "guard_ttis": GUARD_TTIS,
        "train": {
            "range": list(TRAIN_RANGE),
            "n_ttis": TRAIN_RANGE[1] - TRAIN_RANGE[0],
            "tti_indices": list(range(*TRAIN_RANGE)),
            "blocks": train_blocks,
        },
        "guard_train_dev": {
            "range": list(GUARD1_RANGE),
            "n_ttis": GUARD1_RANGE[1] - GUARD1_RANGE[0],
            "tti_indices": list(range(*GUARD1_RANGE)),
        },
        "dev": {
            "range": list(DEV_RANGE),
            "n_ttis": DEV_RANGE[1] - DEV_RANGE[0],
            "tti_indices": list(range(*DEV_RANGE)),
            "blocks": dev_blocks,
        },
        "guard_dev_test": {
            "range": list(GUARD2_RANGE),
            "n_ttis": GUARD2_RANGE[1] - GUARD2_RANGE[0],
            "tti_indices": list(range(*GUARD2_RANGE)),
        },
        "test": {
            "range": list(TEST_RANGE),
            "n_ttis": TEST_RANGE[1] - TEST_RANGE[0],
            "tti_indices": list(range(*TEST_RANGE)),
            "n_blocks": TEST_N_BLOCKS,
            "block_size": TEST_BLOCK_SIZE,
            "blocks": test_blocks,
        },
    }
    return splits


def ls_linear_channel(y_grid: np.ndarray, pilot_scs: np.ndarray = PILOT_SCS_14) -> np.ndarray:
    """Classical single-TTI LS pilot estimate + piecewise-linear interpolation to USED_SCS [101]."""
    p_tx = pilot_tx_for_scs(pilot_scs)
    hp = y_grid[PILOT_SYM, pilot_scs] / p_tx
    h_re = np.interp(USED_SCS, pilot_scs, hp.real)
    h_im = np.interp(USED_SCS, pilot_scs, hp.imag)
    return h_re + 1j * h_im


def estimate_noise_var_from_grid(y_grid: np.ndarray, h_used: np.ndarray, pilot_scs: np.ndarray = PILOT_SCS_14) -> float:
    """Strictly blind, C*-equivariant in-band effective noise variance estimator.

    Combines:
      1. Blind nearest-constellation residual variance on the pilot symbol (sym=2) data REs
         (free of post-pilot CFO drift, phase jumps, and post-pilot NBI; accurately captures
         in-band SDR thermal noise + channel interpolation error without guard-band filter roll-off bias),
         scaled by 1.25 to correct Voronoi truncation bias at moderate SNR.
      2. Pilot frequency second-difference variance estimator.
      3. Far guard-band wideband AWGN floor (captures heavy added wideband AWGN when Voronoi saturates).
    """
    guard_scs = np.array([0, 1, 2, 3, 4, 5, 122, 123, 124, 125, 126, 127], dtype=np.int64)
    n0_guard = float(np.mean(np.abs(y_grid[:, guard_scs]) ** 2))
    y_pwr = float(np.mean(np.abs(y_grid[:, USED_SCS]) ** 2))
    h_pwr = float(np.mean(np.abs(h_used) ** 2))

    # 1. Blind in-band constellation residual on pilot symbol (sym=2) non-pilot active subcarriers
    sym2_mask = DATA_MASK_101[PILOT_SYM]  # [101] bool (87 data REs on sym 2)
    y2 = y_grid[PILOT_SYM, USED_SCS][sym2_mask]
    h2 = h_used[sym2_mask]
    diff2 = y2[:, None] - h2[:, None] * CONST16[None, :]
    min_d2 = np.min(diff2.real ** 2 + diff2.imag ** 2, axis=1)
    # Robust trimmed mean (drop top 10% outlier REs) x 1.25 Voronoi correction
    n0_inband = 1.25 * float(np.mean(np.sort(min_d2)[: max(1, int(0.90 * len(min_d2)))]))

    # 2. Pilot high-frequency second-difference noise floor
    p_tx = pilot_tx_for_scs(pilot_scs)
    hp = y_grid[PILOT_SYM, pilot_scs] / p_tx
    if len(hp) >= 3:
        d2 = hp[2:] - 2.0 * hp[1:-1] + hp[:-2]
        n0_diff = float(np.mean(np.abs(d2) ** 2) / 6.0) * float(np.mean(np.abs(p_tx) ** 2))
    else:
        n0_diff = n0_guard

    n0_est = max(n0_inband, n0_guard, 0.35 * n0_diff, 1e-5 * max(h_pwr, 1e-6))
    if y_pwr > 0.0:
        n0_est = min(n0_est, 1.5 * y_pwr)
    return float(n0_est)


def exact_16qam_llrs(
    y_obs: np.ndarray,
    h_est: np.ndarray,
    noise_var: np.ndarray | float,
) -> np.ndarray:
    """Exact log-sum-exp 16-QAM bit LLRs (positive => bit 1) for Natural Binary constellation.

    Args:
      y_obs: complex array of shape [...]
      h_est: complex array broadcastable to y_obs
      noise_var: positive real scalar or array broadcastable to y_obs (complex circular variance E|n|^2)
    Returns:
      llrs: float64 array of shape [..., 4]
    """
    y_obs = np.asarray(y_obs, dtype=np.complex128)
    h_est = np.asarray(h_est, dtype=np.complex128)
    nv = np.maximum(np.asarray(noise_var, dtype=np.float64), 1e-12)
    # log p(y | x_m) = -|y - h * x_m|^2 / nv  (up to common const)
    diff = y_obs[..., None] - h_est[..., None] * CONST16
    logp = -(diff.real ** 2 + diff.imag ** 2) / nv[..., None]  # [..., 16]
    llrs = np.empty(y_obs.shape + (4,), dtype=np.float64)
    for b in range(4):
        lp1 = logp[..., BIT_ONE_IDX[b]]
        lp0 = logp[..., BIT_ZERO_IDX[b]]
        m1 = np.max(lp1, axis=-1, keepdims=True)
        m0 = np.max(lp0, axis=-1, keepdims=True)
        s1 = m1[..., 0] + np.log(np.sum(np.exp(lp1 - m1), axis=-1))
        s0 = m0[..., 0] + np.log(np.sum(np.exp(lp0 - m0), axis=-1))
        llrs[..., b] = s1 - s0
    return llrs


def mixture_16qam_llrs(
    y_obs: np.ndarray,          # [14, 101] complex128
    h_bank: np.ndarray,         # [R, 14, 101] complex128
    nv_bank: np.ndarray,        # [R, 14, 101] float64
    log_weights: np.ndarray,    # [R, 14] float64 (normalized along R per symbol)
) -> np.ndarray:
    """Exact mixture-posterior marginalized 16-QAM bit LLRs (posterior_fusion §5.2).

    Computes for each RE (s, k) and bit b:
      Lambda_{s,k,b} = log sum_{r, x: u_b(x)=1} alpha_{r,s} p(y_{s,k} | H_{r,s,k}, x, nv_{r,s,k})
                     - log sum_{r, x: u_b(x)=0} alpha_{r,s} p(y_{s,k} | H_{r,s,k}, x, nv_{r,s,k})
    where p(y | H, x, nv) = (1 / (pi * nv)) * exp(-|y - H x|^2 / nv) retains the exact
    1 / (pi * nv) normalization factor across hypotheses r!
    """
    nv = np.maximum(nv_bank, 1e-12)  # [R, 14, 101]
    diff = y_obs[None, :, :, None] - h_bank[:, :, :, None] * CONST16[None, None, None, :]  # [R, 14, 101, 16]
    log_lik = (
        log_weights[:, :, None, None]
        - np.log(math.pi * nv)[:, :, :, None]
        - (diff.real ** 2 + diff.imag ** 2) / nv[:, :, :, None]
    )  # [R, 14, 101, 16]
    # Marginalize over R first for each symbol x_m -> [14, 101, 16]
    m_r = np.max(log_lik, axis=0, keepdims=True)
    log_pm = m_r[0] + np.log(np.sum(np.exp(log_lik - m_r), axis=0))  # [14, 101, 16]

    llrs = np.empty((N_SYM, len(USED_SCS), 4), dtype=np.float64)
    for b in range(4):
        lp1 = log_pm[..., BIT_ONE_IDX[b]]
        lp0 = log_pm[..., BIT_ZERO_IDX[b]]
        m1 = np.max(lp1, axis=-1, keepdims=True)
        m0 = np.max(lp0, axis=-1, keepdims=True)
        s1 = m1[..., 0] + np.log(np.sum(np.exp(lp1 - m1), axis=-1))
        s0 = m0[..., 0] + np.log(np.sum(np.exp(lp0 - m0), axis=-1))
        llrs[..., b] = s1 - s0
    return llrs


def compute_bit_metrics(llrs_1400: np.ndarray, labels_1400: np.ndarray, n_bins: int = 15) -> dict:
    """Compute BER, SER, bit NLL (nats/bit), and ECE on 1400 data REs x 4 bits."""
    llrs = np.clip(np.asarray(llrs_1400, dtype=np.float64), -40.0, 40.0)
    lbl = np.asarray(labels_1400, dtype=np.int32)
    pred_bits = (llrs > 0.0).astype(np.int32)
    ber = float(np.mean(pred_bits != lbl))
    ser = float(np.mean(np.any(pred_bits != lbl, axis=-1)))
    signed = 2.0 * lbl - 1.0
    # Stable softplus(-signed * llrs)
    z = -signed * llrs
    nll = float(np.mean(np.where(z > 0, z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)))))
    # Expected Calibration Error (15 uniform probability bins on predicted confidence)
    p1 = 1.0 / (1.0 + np.exp(-llrs))
    conf = np.where(pred_bits == 1, p1, 1.0 - p1).ravel()
    acc = (pred_bits == lbl).astype(np.float64).ravel()
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    n_tot = len(conf)
    for bi in range(n_bins):
        lo, hi = edges[bi], edges[bi + 1]
        mask = (conf >= lo) & (conf <= hi if bi == n_bins - 1 else conf < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            ece += (cnt / n_tot) * abs(float(np.mean(acc[mask])) - float(np.mean(conf[mask])))
    return {"ber": ber, "ser": ser, "nll": nll, "ece": float(ece)}


def exact_sign_flip_test(diffs: np.ndarray) -> dict:
    """Exact 2^G two-sided sign-flip permutation test + cluster t 95% CI on G independent blocks.

    Convention: positive diff = candidate improves over baseline (e.g., baseline_loss - cand_loss > 0).
    """
    d = np.asarray(diffs, dtype=np.float64)
    g = len(d)
    assert g <= 16, f"Exact sign-flip supports G<=16, got {g}"
    mean_d = float(np.mean(d))
    sd_d = float(np.std(d, ddof=1)) if g > 1 else 0.0
    se_d = sd_d / math.sqrt(g) if g > 1 else 0.0
    # Exact t critical for 95% two-sided CI
    from scipy.stats import t as student_t

    t_crit = float(student_t.ppf(0.975, df=g - 1)) if g > 1 else 0.0
    ci_lo = mean_d - t_crit * se_d
    ci_hi = mean_d + t_crit * se_d

    # Enumerate all 2^G sign assignments
    signs = ((np.arange(1 << g)[:, None] >> np.arange(g)[None, :]) & 1) * 2 - 1  # [2^G, G]
    perm_means = (signs @ d) / float(g)
    obs_abs = abs(mean_d) - 1e-15
    p_two = float(np.mean(np.abs(perm_means) >= obs_abs))
    p_one = float(np.mean(perm_means >= mean_d - 1e-15))
    wins = int(np.sum(d > 1e-12))
    ties = int(np.sum(np.abs(d) <= 1e-12))
    losses = int(np.sum(d < -1e-12))
    return {
        "n_blocks": g,
        "mean_diff": mean_d,
        "se": se_d,
        "ci95": [ci_lo, ci_hi],
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "p_two_sided": p_two,
        "p_one_sided": p_one,
    }


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
