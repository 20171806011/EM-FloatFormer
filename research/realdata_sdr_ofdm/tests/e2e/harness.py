"""
Independent E2E Test Harness for EM-FloatFormer Real-Data SDR OFDM (`F12`).

Provides:
- File-existence-aware `MilestoneResolver`: raises `unittest.SkipTest` ONLY when target
  `.py` modules or artifacts are not yet created on disk, fails loudly on broken
  internal imports inside existing modules, and raises `AssertionError` when an
  existing module omits or mis-exposes a required callable symbol.
- Exact SDR OFDM geometry (`S=14, N_fft=128, F=102, FFT_offset=13`, DC bin `64`,
  14 QPSK pilots in symbol `2`, `1400` payload REs, `5600` bits/TTI) and active-band
  normalizer `extract_active_h`.
- Independent non-Gray natural-binary `mapping_table(4)` modulator, hard slicer,
  exact log-sum-exp LLR oracle, max-log LLR oracle, and exact Gaussian-mixture LLR oracle.
- Synthetic `RealOFDMFrame`, `ReceiverOutput`, and `CustomDataset` `.pth` builders.
- Independent Git, SHA-256, Git-blob-SHA1, and statistical (`exact_sign_flip_p_ref`,
  `holm_adjust_ref`, `one_sided_u95_ref`) oracles.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import unittest

import numpy as np
from scipy.stats import t as student_t
import torch

# ---------------------------------------------------------------------------
# Paths, PEP 420 Bootstrap & Governance Constants
# ---------------------------------------------------------------------------

E2E_DIR: Path = Path(__file__).resolve().parent
TESTS_ROOT: Path = E2E_DIR.parent
RESEARCH_ROOT: Path = TESTS_ROOT.parent
REPO_ROOT: Path = RESEARCH_ROOT.parents[1]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPECTED_MAIN_SHA: str = "63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1"
EXPECTED_P12_BASE_SHA: str = "f566f9b8b0ddfb468a4cb3fd64588bfa46c87212"
EXPECTED_WORK_BRANCH: str = "research/realdata-sdr-ofdm"
E2E_SEED_BASE: int = 1_360_000_000_000

# ---------------------------------------------------------------------------
# SDR OFDM Physical Geometry & Pilot Constants (rikluost/sdr_ofdm_dataset_v1)
# ---------------------------------------------------------------------------

N_SYMB: int = 14
N_FFT: int = 128
N_ACTIVE: int = 102
FFT_OFFSET: int = 13
DC_ACTIVE_IDX: int = 51
DC_FFT_BIN: int = 64
PILOT_SYMBOL_IDX: int = 2
PILOT_ACTIVE_INDICES: Tuple[int, ...] = (0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 101)
PILOT_FFT_BINS: Tuple[int, ...] = tuple(FFT_OFFSET + i for i in PILOT_ACTIVE_INDICES)
N_PILOTS: int = len(PILOT_ACTIVE_INDICES)  # 14
N_DC_RES: int = N_SYMB  # 14
N_GUARD_RES: int = (N_FFT - N_ACTIVE) * N_SYMB  # 364
N_DATA_RES: int = 1400
BITS_PER_SYMBOL: int = 4
BITS_PER_TTI: int = N_DATA_RES * BITS_PER_SYMBOL  # 5600

PILOT_VALUES_REF: np.ndarray = np.tile(
    np.array([-0.7 - 0.7j, -0.7 + 0.7j, 0.7 - 0.7j, 0.7 + 0.7j], dtype=np.complex128),
    4,
)[:N_PILOTS].copy()

EXPECTED_PRIMARY_RAW_SIZE: int = 27684430
EXPECTED_PRIMARY_BLOB_SHA1: str = "9b1aee8673ec48f42f52dc365fb01cc8b3e2b741"
EXPECTED_BACKUP_SHA256: str = "c029532908f9702fea4be3ce37d0e1de9138019e12dd06e228abb1b83525f2b7"


# ---------------------------------------------------------------------------
# File-Existence-Aware Milestone Dependency Resolver
# ---------------------------------------------------------------------------


class MilestoneResolver:
    """
    Resolves `research.realdata_sdr_ofdm.*` modules and deliverables progressively.

    Crucially, `get_module` checks whether the target `.py` file (or package
    `__init__.py`) exists on disk BEFORE importing. Missing files raise
    `unittest.SkipTest`, while broken imports/syntax errors inside an existing
    implementation file propagate loudly and fail the test.
    """

    @staticmethod
    def _module_paths(rel_module: str) -> Tuple[Path, Path]:
        parts = rel_module.split(".")
        mod_file = RESEARCH_ROOT.joinpath(*parts).with_suffix(".py")
        pkg_init = RESEARCH_ROOT.joinpath(*parts, "__init__.py")
        return mod_file, pkg_init

    @staticmethod
    def module_exists(rel_module: str) -> bool:
        mod_file, pkg_init = MilestoneResolver._module_paths(rel_module)
        return mod_file.is_file() or pkg_init.is_file()

    @staticmethod
    def get_module(rel_module: str, required_milestone: str) -> Any:
        if not MilestoneResolver.module_exists(rel_module):
            raise unittest.SkipTest(
                f"[PROGRESSIVE EXECUTION] Module 'research.realdata_sdr_ofdm.{rel_module}' "
                f"not yet implemented on disk. Requires {required_milestone}."
            )
        return importlib.import_module(f"research.realdata_sdr_ofdm.{rel_module}")

    @staticmethod
    def get_callable(rel_module: str, func_name: str, required_milestone: str) -> Any:
        mod = MilestoneResolver.get_module(rel_module, required_milestone)
        if not hasattr(mod, func_name):
            raise AssertionError(
                f"[CONTRACT VIOLATION] Module 'research.realdata_sdr_ofdm.{rel_module}' "
                f"exists on disk for {required_milestone} but does not expose required symbol '{func_name}'."
            )
        obj = getattr(mod, func_name)
        if not callable(obj):
            raise AssertionError(
                f"[CONTRACT VIOLATION] Symbol '{func_name}' in "
                f"'research.realdata_sdr_ofdm.{rel_module}' is not callable: {obj!r}."
            )
        return obj

    @staticmethod
    def get_first_callable(
        rel_module: str,
        candidate_names: Sequence[str],
        required_milestone: str,
    ) -> Any:
        mod = MilestoneResolver.get_module(rel_module, required_milestone)
        for name in candidate_names:
            if hasattr(mod, name):
                obj = getattr(mod, name)
                if not callable(obj):
                    raise AssertionError(
                        f"[CONTRACT VIOLATION] Candidate symbol '{name}' in "
                        f"'research.realdata_sdr_ofdm.{rel_module}' is not callable: {obj!r}."
                    )
                return obj
        raise AssertionError(
            f"[CONTRACT VIOLATION] Module 'research.realdata_sdr_ofdm.{rel_module}' "
            f"exists on disk for {required_milestone} but exposes none of required symbols {tuple(candidate_names)}."
        )

    @staticmethod
    def get_class(rel_module: str, class_name: str, required_milestone: str) -> Any:
        return MilestoneResolver.get_callable(rel_module, class_name, required_milestone)

    @staticmethod
    def require_artifact(rel_path: str | Path, required_milestone: str, *, allow_dir: bool = False) -> Path:
        path = RESEARCH_ROOT / rel_path
        if allow_dir:
            if not path.is_dir() or not any(path.iterdir()):
                raise unittest.SkipTest(
                    f"[PROGRESSIVE EXECUTION] Deliverable directory 'research/realdata_sdr_ofdm/{rel_path}' "
                    f"not yet populated. Requires {required_milestone}."
                )
            return path
        if not path.is_file() or path.stat().st_size == 0:
            raise unittest.SkipTest(
                f"[PROGRESSIVE EXECUTION] Deliverable 'research/realdata_sdr_ofdm/{rel_path}' "
                f"not yet created. Requires {required_milestone}."
            )
        return path

    @staticmethod
    def require_live_realdata(subpath: str = "") -> Path:
        root = Path(os.environ.get("EMFF_REALDATA_ROOT", "/tmp/tw_realdata"))
        target = root / subpath if subpath else root
        if not target.exists():
            raise unittest.SkipTest(
                f"[LIVE DATA OPTIONAL] Remote Codespace data path '{target}' not present on host."
            )
        return target


# ---------------------------------------------------------------------------
# SDR OFDM Grid Masks & Independent Non-Gray 16-QAM Reference Oracles
# ---------------------------------------------------------------------------


def build_sdr_ofdm_masks() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return disjoint `(pilot_mask, dc_mask, data_mask)` boolean `(14, 128)` masks."""
    pilot_mask = np.zeros((N_SYMB, N_FFT), dtype=bool)
    pilot_mask[PILOT_SYMBOL_IDX, np.asarray(PILOT_FFT_BINS, dtype=np.int64)] = True

    dc_mask = np.zeros((N_SYMB, N_FFT), dtype=bool)
    dc_mask[:, DC_FFT_BIN] = True

    data_mask = np.zeros((N_SYMB, N_FFT), dtype=bool)
    data_mask[:, FFT_OFFSET : FFT_OFFSET + N_ACTIVE] = True
    data_mask[dc_mask] = False
    data_mask[pilot_mask] = False

    assert int(pilot_mask.sum()) == N_PILOTS
    assert int(dc_mask.sum()) == N_DC_RES
    assert int(data_mask.sum()) == N_DATA_RES
    assert not np.any(pilot_mask & dc_mask)
    assert not np.any(pilot_mask & data_mask)
    assert not np.any(dc_mask & data_mask)
    return pilot_mask, dc_mask, data_mask


def extract_active_h(arr: np.ndarray) -> np.ndarray:
    """Normalize `(14, 102)` or `(14, 128)` channel/variance grid to `(14, 102)` active bins."""
    a = np.asarray(arr)
    if a.ndim != 2 or a.shape[0] != N_SYMB:
        raise ValueError(f"Expected 2D array with {N_SYMB} OFDM symbols, got {a.shape}")
    if a.shape[1] == N_ACTIVE:
        return a
    if a.shape[1] == N_FFT:
        return a[:, FFT_OFFSET : FFT_OFFSET + N_ACTIVE]
    raise ValueError(f"Expected width {N_ACTIVE} or {N_FFT}, got {a.shape}")


def nongray_16qam_table_ref() -> np.ndarray:
    """
    Independent natural-binary 16-QAM constellation matching `mapping_table(4)`.

    Levels: `[-3, -1, +1, +3] / sqrt(10)` indexed sequentially `0..15` by
    `idx = 8*b0 + 4*b1 + 2*b2 + b3`, where `(b0, b1)` select Re and `(b2, b3)` select Im.
    """
    levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64) / math.sqrt(10.0)
    table = (levels[:, None] + 1j * levels[None, :]).reshape(-1).astype(np.complex128)
    return table


def nongray_16qam_bits_ref() -> np.ndarray:
    """Return `(16, 4)` uint8 bit vectors corresponding to `nongray_16qam_table_ref()`."""
    return np.array(
        [[(i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(16)],
        dtype=np.uint8,
    )


def nongray_16qam_mod_ref(bits: np.ndarray) -> np.ndarray:
    """Modulate `(..., 4)` binary bits into non-Gray 16-QAM `complex128` symbols."""
    b = np.asarray(bits, dtype=np.int64)
    if b.shape[-1] != 4:
        raise ValueError(f"Expected last dim 4, got {b.shape}")
    idx = (b[..., 0] << 3) | (b[..., 1] << 2) | (b[..., 2] << 1) | b[..., 3]
    return nongray_16qam_table_ref()[idx]


def nongray_16qam_hard_demod_ref(z: np.ndarray) -> np.ndarray:
    """Nearest-neighbor hard slicer for non-Gray `mapping_table(4)`."""
    z_arr = np.asarray(z, dtype=np.complex128)
    table = nongray_16qam_table_ref()
    bits_tbl = nongray_16qam_bits_ref()
    d2 = np.abs(z_arr[..., None] - table) ** 2
    idx = np.argmin(d2, axis=-1)
    return bits_tbl[idx].copy()


def _logsumexp_np(a: np.ndarray, axis: int = -1) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    finite_max = np.where(np.isfinite(a_max), a_max, 0.0)
    out = finite_max + np.log(np.sum(np.exp(a - finite_max), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def nongray_16qam_exact_llr_ref(z: np.ndarray, sigma2: np.ndarray | float) -> np.ndarray:
    """
    Independent exact log-sum-exp LLR oracle for non-Gray `mapping_table(4)`.

    Convention: positive LLR <=> bit == 1 (`log P(b=1|z) / P(b=0|z)`).
    """
    z_arr = np.asarray(z, dtype=np.complex128)
    var = np.maximum(np.asarray(sigma2, dtype=np.float64), 1e-12)
    table = nongray_16qam_table_ref()
    bits_tbl = nongray_16qam_bits_ref()

    d2 = np.abs(z_arr[..., None] - table) ** 2
    log_likes = -d2 / var[..., None]

    llrs = np.empty(z_arr.shape + (4,), dtype=np.float64)
    for bit_idx in range(4):
        ones = bits_tbl[:, bit_idx] == 1
        zeros = ~ones
        l1 = _logsumexp_np(log_likes[..., ones], axis=-1)
        l0 = _logsumexp_np(log_likes[..., zeros], axis=-1)
        llrs[..., bit_idx] = l1 - l0
    return llrs


def nongray_16qam_maxlog_llr_ref(z: np.ndarray, sigma2: np.ndarray | float) -> np.ndarray:
    """Independent max-log LLR oracle for non-Gray `mapping_table(4)` (positive = bit 1)."""
    z_arr = np.asarray(z, dtype=np.complex128)
    var = np.maximum(np.asarray(sigma2, dtype=np.float64), 1e-12)
    table = nongray_16qam_table_ref()
    bits_tbl = nongray_16qam_bits_ref()

    d2 = np.abs(z_arr[..., None] - table) ** 2
    log_likes = -d2 / var[..., None]

    llrs = np.empty(z_arr.shape + (4,), dtype=np.float64)
    for bit_idx in range(4):
        ones = bits_tbl[:, bit_idx] == 1
        zeros = ~ones
        llrs[..., bit_idx] = np.max(log_likes[..., ones], axis=-1) - np.max(log_likes[..., zeros], axis=-1)
    return llrs


def nongray_16qam_mixture_llr_ref(
    y: np.ndarray,
    branch_means: Sequence[np.ndarray],
    branch_vars: Sequence[np.ndarray],
    branch_weights: Sequence[float],
    n0: float | np.ndarray,
    *,
    include_constellation_var: bool = True,
) -> np.ndarray:
    """Independent log-mixture oracle `log(sum_r alpha_r g_r(c))` on non-Gray `mapping_table(4)` (positive = bit 1)."""
    y_arr = np.asarray(y, dtype=np.complex128)
    alpha = np.asarray(branch_weights, dtype=np.float64).ravel()
    if len(branch_means) != len(alpha) or len(branch_vars) != len(alpha):
        raise ValueError("Mismatched branch_means / branch_vars / branch_weights length")
    n0_arr = np.maximum(np.asarray(n0, dtype=np.float64), 1e-12)
    table = nongray_16qam_table_ref()
    bits_tbl = nongray_16qam_bits_ref()
    c_pow = np.abs(table) ** 2

    weighted_lls = []
    for m_r, v_r, a_r in zip(branch_means, branch_vars, alpha):
        if float(a_r) <= 0.0:
            continue
        m_arr = np.asarray(m_r, dtype=np.complex128)
        v_arr = np.maximum(np.asarray(v_r, dtype=np.float64), 0.0)
        var = n0_arr[..., None] + (v_arr[..., None] * c_pow if include_constellation_var else v_arr[..., None])
        var = np.maximum(var, 1e-12)
        ll_r = -np.log(math.pi * var) - (np.abs(y_arr[..., None] - m_arr[..., None] * table) ** 2) / var
        weighted_lls.append(math.log(float(a_r)) + ll_r)

    if not weighted_lls:
        raise ValueError("At least one branch must have positive weight")
    log_mix = _logsumexp_np(np.stack(weighted_lls, axis=0), axis=0)
    llrs = np.empty(y_arr.shape + (4,), dtype=np.float64)
    for bit_idx in range(4):
        ones = bits_tbl[:, bit_idx] == 1
        llrs[..., bit_idx] = _logsumexp_np(log_mix[..., ones], axis=-1) - _logsumexp_np(log_mix[..., ~ones], axis=-1)
    return llrs


# ---------------------------------------------------------------------------
# Synthetic RealOFDMFrame, ReceiverOutput & CustomDataset Builders
# ---------------------------------------------------------------------------


def _apply_time_cfo_phase(
    grid_14_128: np.ndarray,
    cfo_norm: float = 0.0,
    phase_drift_rad: float = 0.0,
    cp_len: int = 7,
) -> np.ndarray:
    """Apply physical IFFT -> sample-level CFO / symbol-level phase drift -> FFT."""
    if cfo_norm == 0.0 and phase_drift_rad == 0.0:
        return grid_14_128.copy()
    time_sym = np.fft.ifft(np.fft.ifftshift(grid_14_128, axes=1), axis=1)
    n_idx = np.arange(N_FFT, dtype=np.float64)[None, :]
    s_idx = np.arange(N_SYMB, dtype=np.float64)[:, None]
    sample_offsets = s_idx * (N_FFT + cp_len) + n_idx
    phase = 2.0 * math.pi * cfo_norm * sample_offsets / float(N_FFT) + phase_drift_rad * (s_idx / max(N_SYMB - 1, 1))
    rotated = time_sym * np.exp(1j * phase)
    return np.fft.fftshift(np.fft.fft(rotated, axis=1), axes=1).astype(np.complex128)


def make_synthetic_grid_and_truth(
    seed: int,
    *,
    snr_db: float = 20.0,
    cfo_norm: float = 0.0,
    phase_offset: float = 0.0,
    phase_drift_rad: float = 0.0,
    gauge_scale: complex = 1.0,
    n_taps: int = 4,
    flat_channel: bool = False,
) -> Dict[str, Any]:
    """Generate deterministic `(14, 128)` OFDM RX grid, multipath channel, bits, and symbols."""
    rng = np.random.default_rng(int(seed))
    pilot_mask, dc_mask, data_mask = build_sdr_ofdm_masks()

    tx_bits = rng.integers(0, 2, size=(N_DATA_RES, 4), dtype=np.uint8)
    tx_symbols = nongray_16qam_mod_ref(tx_bits)

    tx_grid = np.zeros((N_SYMB, N_FFT), dtype=np.complex128)
    tx_grid[data_mask] = tx_symbols
    tx_grid[pilot_mask] = PILOT_VALUES_REF

    if flat_channel:
        h_freq_128 = np.full(N_FFT, np.exp(1j * phase_offset), dtype=np.complex128)
    else:
        taps = (rng.standard_normal(n_taps) + 1j * rng.standard_normal(n_taps)) / math.sqrt(2.0)
        decay = np.exp(-0.5 * np.arange(n_taps, dtype=np.float64))
        taps = taps * decay
        taps = taps / math.sqrt(np.sum(np.abs(taps) ** 2))
        h_padded = np.zeros(N_FFT, dtype=np.complex128)
        h_padded[:n_taps] = taps
        h_freq_128 = np.fft.fftshift(np.fft.fft(h_padded)) * np.exp(1j * phase_offset)

    h_true = np.broadcast_to(h_freq_128[None, :], (N_SYMB, N_FFT)).copy()
    clean_rx = h_true * tx_grid
    clean_rx = _apply_time_cfo_phase(clean_rx, cfo_norm=cfo_norm, phase_drift_rad=phase_drift_rad)

    n0 = 10.0 ** (-float(snr_db) / 10.0)
    noise = math.sqrt(n0 / 2.0) * (rng.standard_normal((N_SYMB, N_FFT)) + 1j * rng.standard_normal((N_SYMB, N_FFT)))
    # Preserve realistic guard noise while keeping active bins dominant
    rx_grid = clean_rx + noise

    scale_c = complex(gauge_scale)
    rx_scaled = (rx_grid * scale_c).astype(np.complex128)
    pilots_ref = PILOT_VALUES_REF.copy()
    n0_scaled = float(n0 * (abs(scale_c) ** 2))
    h_true_scaled = (h_true * scale_c).astype(np.complex128)

    y_p = rx_scaled[pilot_mask]
    s0 = float(math.sqrt(max(float(np.mean(np.abs(y_p) ** 2)), 1e-38)))

    return {
        "rx_grid": rx_scaled,
        "pilot_mask": pilot_mask,
        "dc_mask": dc_mask,
        "data_mask": data_mask,
        "pilot_values": pilots_ref,
        "tx_bits": tx_bits,
        "tx_symbols": tx_symbols,
        "h_true": h_true_scaled,
        "n0_true": n0_scaled,
        "s0": s0,
        "reported_sinr_db": float(snr_db),
    }


def _build_heldout_pilot_folds_ref(heldout_idx: Sequence[int]) -> List[Dict[str, Any]]:
    fold0_heldout = [int(i) for i in heldout_idx]
    fold0_set = set(fold0_heldout)
    fold0_obs = [i for i in range(N_PILOTS) if i not in fold0_set]
    folds: List[Dict[str, Any]] = [
        {
            "fold": 0,
            "heldout_pilot_indices": fold0_heldout,
            "obs_pilot_indices": fold0_obs,
            "heldout_fft_indices": [PILOT_FFT_BINS[i] for i in fold0_heldout],
            "obs_fft_indices": [PILOT_FFT_BINS[i] for i in fold0_obs],
        }
    ]
    for f in range(1, 4):
        h_idx = list(range(1 + f, 13, 4))
        h_set = set(h_idx)
        o_idx = [i for i in range(N_PILOTS) if i not in h_set]
        folds.append(
            {
                "fold": f,
                "heldout_pilot_indices": h_idx,
                "obs_pilot_indices": o_idx,
                "heldout_fft_indices": [PILOT_FFT_BINS[i] for i in h_idx],
                "obs_fft_indices": [PILOT_FFT_BINS[i] for i in o_idx],
            }
        )
    return folds


def scale_real_ofdm_frame(frame: Any, gauge_scale: complex) -> Any:
    """Return a copy of `frame` scaled by $c \\in \\mathbb{C}^*$ (`rx_grid *= c`, `est_n0 *= |c|^2`)."""
    c = complex(gauge_scale)
    r2 = abs(c) ** 2
    scaled_grid = (np.asarray(frame.rx_grid) * c).astype(
        frame.rx_grid.dtype if hasattr(frame.rx_grid, "dtype") else np.complex128
    )
    meta_copy = dict(frame.meta) if isinstance(getattr(frame, "meta", None), dict) else {}
    if "s0" in meta_copy:
        meta_copy["s0"] = float(meta_copy["s0"]) * abs(c)
    if "guard_band_power" in meta_copy:
        meta_copy["guard_band_power"] = float(meta_copy["guard_band_power"]) * r2
    if "pilot_rms" in meta_copy:
        meta_copy["pilot_rms"] = float(meta_copy["pilot_rms"]) * abs(c)
    if "dc_power" in meta_copy:
        meta_copy["dc_power"] = float(meta_copy["dc_power"]) * r2
    if "rx_grid_sha256" in meta_copy:
        meta_copy["rx_grid_sha256"] = hashlib.sha256(np.ascontiguousarray(scaled_grid).tobytes()).hexdigest()
    cls = type(frame)
    return cls(
        frame_id=frame.frame_id,
        split=frame.split,
        group_id=frame.group_id,
        rx_grid=scaled_grid,
        pilot_mask=frame.pilot_mask.copy(),
        dc_mask=frame.dc_mask.copy(),
        data_mask=frame.data_mask.copy(),
        pilot_values=frame.pilot_values.copy(),
        tx_bits=frame.tx_bits.copy(),
        tx_symbols=frame.tx_symbols.copy(),
        reported_sinr_db=float(frame.reported_sinr_db),
        est_n0=float(frame.est_n0) * r2,
        meta=meta_copy,
    )


def make_synthetic_real_ofdm_frame(
    seed: int,
    *,
    snr_db: float = 20.0,
    split: str = "dev",
    group_id: str = "blk_00",
    frame_id: int = 0,
    cfo_norm: float = 0.0,
    phase_offset: float = 0.0,
    phase_drift_rad: float = 0.0,
    gauge_scale: complex = 1.0,
    n_taps: int = 4,
    flat_channel: bool = False,
    heldout_pilot_indices: Sequence[int] = (1, 5, 9, 12),
    required_milestone: str = "Milestone M1 (F4)",
) -> Tuple[Any, Dict[str, Any]]:
    """Construct a contract-compliant `RealOFDMFrame` via `MilestoneResolver`."""
    RealOFDMFrame = MilestoneResolver.get_class("adapters.sdr_ofdm_adapter", "RealOFDMFrame", required_milestone)
    sim = make_synthetic_grid_and_truth(
        seed=seed,
        snr_db=snr_db,
        cfo_norm=cfo_norm,
        phase_offset=phase_offset,
        phase_drift_rad=phase_drift_rad,
        gauge_scale=gauge_scale,
        n_taps=n_taps,
        flat_channel=flat_channel,
    )

    pilot_indices_arr = np.asarray(PILOT_FFT_BINS, dtype=np.int64)
    heldout_idx = [int(i) for i in heldout_pilot_indices]
    heldout_mask = np.zeros((N_SYMB, N_FFT), dtype=bool)
    for pi in heldout_idx:
        heldout_mask[PILOT_SYMBOL_IDX, pilot_indices_arr[pi]] = True

    guard_mask = ~(sim["pilot_mask"] | sim["dc_mask"] | sim["data_mask"])
    guard_power = float(np.mean(np.abs(sim["rx_grid"][guard_mask]) ** 2))
    pilot_power = float(np.mean(np.abs(sim["rx_grid"][sim["pilot_mask"]]) ** 2))

    meta = {
        "s0": sim["s0"],
        "guard_band_power": guard_power,
        "pilot_snr_db": float(10.0 * math.log10(max(pilot_power / max(sim["n0_true"], 1e-18), 1e-12))),
        "pilot_rms": float(math.sqrt(max(pilot_power, 0.0))),
        "dc_power": float(np.mean(np.abs(sim["rx_grid"][sim["dc_mask"]]) ** 2)),
        "heldout_pilot_folds": _build_heldout_pilot_folds_ref(heldout_idx),
        "heldout_pilot_indices": heldout_idx,
        "heldout_pilot_mask": heldout_mask,
        "tx_bits_sha256": hashlib.sha256(sim["tx_bits"].tobytes()).hexdigest(),
        "rx_grid_sha256": hashlib.sha256(np.ascontiguousarray(sim["rx_grid"]).tobytes()).hexdigest(),
    }

    frame = RealOFDMFrame(
        frame_id=int(frame_id),
        split=str(split),
        group_id=str(group_id),
        rx_grid=sim["rx_grid"],
        pilot_mask=sim["pilot_mask"],
        dc_mask=sim["dc_mask"],
        data_mask=sim["data_mask"],
        pilot_values=sim["pilot_values"],
        tx_bits=sim["tx_bits"],
        tx_symbols=sim["tx_symbols"],
        reported_sinr_db=float(snr_db),
        est_n0=float(sim["n0_true"]),
        meta=meta,
    )
    truth = {
        "h_true": sim["h_true"],
        "n0_true": sim["n0_true"],
        "s0": sim["s0"],
        "heldout_pilot_true": sim["h_true"][heldout_mask],
    }
    return frame, truth


def make_standalone_frame_proxy(
    seed: int,
    *,
    snr_db: float = 20.0,
    split: str = "test",
    group_id: str = "blk_00",
    frame_id: int = 0,
    heldout_pilot_indices: Sequence[int] = (1, 5, 9, 12),
) -> Tuple[Any, Dict[str, Any]]:
    """
    Build either a real `RealOFDMFrame` (if M1 adapter exists) or a contract-identical
    `SimpleNamespace` so F7 evaluator tests can run independently of M1 if M2 lands first.
    """
    if MilestoneResolver.module_exists("adapters.sdr_ofdm_adapter"):
        mod = MilestoneResolver.get_module("adapters.sdr_ofdm_adapter", "Milestone M1 (F4)")
        if hasattr(mod, "RealOFDMFrame"):
            return make_synthetic_real_ofdm_frame(
                seed,
                snr_db=snr_db,
                split=split,
                group_id=group_id,
                frame_id=frame_id,
                heldout_pilot_indices=heldout_pilot_indices,
            )

    sim = make_synthetic_grid_and_truth(seed=seed, snr_db=snr_db)
    pilot_indices_arr = np.asarray(PILOT_FFT_BINS, dtype=np.int64)
    heldout_idx = [int(i) for i in heldout_pilot_indices]
    heldout_mask = np.zeros((N_SYMB, N_FFT), dtype=bool)
    for pi in heldout_idx:
        heldout_mask[PILOT_SYMBOL_IDX, pilot_indices_arr[pi]] = True

    guard_mask = ~(sim["pilot_mask"] | sim["dc_mask"] | sim["data_mask"])
    guard_power = float(np.mean(np.abs(sim["rx_grid"][guard_mask]) ** 2))
    pilot_power = float(np.mean(np.abs(sim["rx_grid"][sim["pilot_mask"]]) ** 2))

    frame = SimpleNamespace(
        frame_id=int(frame_id),
        split=str(split),
        group_id=str(group_id),
        rx_grid=sim["rx_grid"],
        pilot_mask=sim["pilot_mask"],
        dc_mask=sim["dc_mask"],
        data_mask=sim["data_mask"],
        pilot_values=sim["pilot_values"],
        tx_bits=sim["tx_bits"],
        tx_symbols=sim["tx_symbols"],
        reported_sinr_db=float(snr_db),
        est_n0=float(sim["n0_true"]),
        meta={
            "s0": sim["s0"],
            "guard_band_power": guard_power,
            "pilot_snr_db": float(10.0 * math.log10(max(pilot_power / max(sim["n0_true"], 1e-18), 1e-12))),
            "pilot_rms": float(math.sqrt(max(pilot_power, 0.0))),
            "dc_power": float(np.mean(np.abs(sim["rx_grid"][sim["dc_mask"]]) ** 2)),
            "heldout_pilot_folds": _build_heldout_pilot_folds_ref(heldout_idx),
            "heldout_pilot_indices": heldout_idx,
            "heldout_pilot_mask": heldout_mask,
            "tx_bits_sha256": hashlib.sha256(sim["tx_bits"].tobytes()).hexdigest(),
            "rx_grid_sha256": hashlib.sha256(np.ascontiguousarray(sim["rx_grid"]).tobytes()).hexdigest(),
        },
    )
    truth = {
        "h_true": sim["h_true"],
        "n0_true": sim["n0_true"],
        "heldout_pilot_true": sim["h_true"][heldout_mask],
    }
    return frame, truth


def make_synthetic_receiver_output(
    frame: Any,
    *,
    eq_symbols: Optional[np.ndarray] = None,
    bit_flip_indices: Sequence[Tuple[int, int]] = (),
    pilot_pred: Optional[np.ndarray] = None,
    pilot_var: Optional[np.ndarray] = None,
    failed: bool = False,
    error: Optional[str] = None,
    latency_ms: float = 1.25,
    n_candidates: int = 1,
    param_count: int = 0,
    branch_weights: Optional[Sequence[float]] = None,
) -> Any:
    """Create a contract-compliant `ReceiverOutput` (or `SimpleNamespace` matching its fields)."""
    bits = frame.tx_bits.copy()
    for sym_i, bit_j in bit_flip_indices:
        bits[sym_i, bit_j] ^= 1

    llrs = np.where(bits == 1, 12.0, -12.0).astype(np.float64)
    hard_bits = (llrs > 0.0).astype(np.uint8)
    hard_symbols = nongray_16qam_mod_ref(hard_bits)
    eq = frame.tx_symbols.copy() if eq_symbols is None else np.asarray(eq_symbols, dtype=np.complex128)
    h_est = np.ones((N_SYMB, N_ACTIVE), dtype=np.complex128)
    h_var = np.full((N_SYMB, N_ACTIVE), 0.01, dtype=np.float64)
    aux = {
        "n_candidates": int(n_candidates),
        "branch_weights": list(branch_weights) if branch_weights is not None else [1.0],
        "latency_ms": float(latency_ms),
        "param_count": int(param_count),
        "failed": bool(failed),
        "error": error,
    }

    kwargs = dict(
        h_est=h_est,
        h_var=h_var,
        eq_symbols=eq,
        bit_llrs=llrs,
        hard_bits=hard_bits,
        hard_symbols=hard_symbols,
        heldout_pilot_pred=pilot_pred,
        heldout_pilot_var=pilot_var,
        aux_info=aux,
    )

    for rel_mod in ("receivers", "receivers.classical", "adapters.sdr_ofdm_adapter"):
        if MilestoneResolver.module_exists(rel_mod):
            mod = importlib.import_module(f"research.realdata_sdr_ofdm.{rel_mod}")
            if hasattr(mod, "ReceiverOutput"):
                return getattr(mod, "ReceiverOutput")(**kwargs)
    return SimpleNamespace(**kwargs)


def make_synthetic_custom_dataset_pth(
    dest_path: Path,
    *,
    n_frames: int = 200,
    seed: int = E2E_SEED_BASE + 301,
    snr_min_db: float = 7.0,
    snr_max_db: float = 20.0,
    corrupt_iq_shape: bool = False,
    corrupt_labels_shape: bool = False,
) -> Dict[str, Any]:
    """
    Create a deterministic `CustomDataset` `.pth` file compatible with both
    `scripts/download_and_verify.py` and `adapters.sdr_ofdm_adapter.load_raw_sdr_ofdm`
    for offline F3/F4 tests.
    """
    if MilestoneResolver.module_exists("adapters.sdr_ofdm_adapter"):
        CustomDataset = MilestoneResolver.get_class("adapters.sdr_ofdm_adapter", "CustomDataset", "Milestone M1 (F4)")
    else:
        CustomDataset = MilestoneResolver.get_class("scripts.download_and_verify", "CustomDataset", "Milestone M1 (F3)")
    ds = CustomDataset()
    rng = np.random.default_rng(int(seed))

    sinr_grid = np.linspace(snr_min_db, snr_max_db, max(n_frames, 1), dtype=np.float32)
    for idx in range(n_frames):
        snr_db = float(sinr_grid[idx])
        sim = make_synthetic_grid_and_truth(seed=seed + idx * 17, snr_db=snr_db, n_taps=4)
        iq = sim["rx_grid"]
        lab = sim["tx_bits"].astype(np.float32)
        if corrupt_iq_shape:
            iq = iq[:, :64]
        if corrupt_labels_shape:
            lab = lab[:, :2]
        ds.add_item(
            torch.from_numpy(iq.astype(np.complex64)),
            torch.from_numpy(lab),
            torch.tensor([snr_db], dtype=torch.float32),
        )

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ds, dest_path)
    size_bytes = dest_path.stat().st_size
    return {
        "path": dest_path,
        "size_bytes": int(size_bytes),
        "sha256": compute_file_sha256(dest_path),
        "git_blob_sha1": compute_git_blob_sha1(dest_path),
        "n_frames": int(n_frames),
        "rng_check": int(rng.integers(0, 1000)),
    }


# ---------------------------------------------------------------------------
# Independent Git, Checksum & Statistical Oracles
# ---------------------------------------------------------------------------


def run_git(*args: str, check: bool = True) -> str:
    """Execute a read-only git command inside `REPO_ROOT` and return stripped stdout."""
    proc = subprocess.run(
        ["git", *args],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed (exit {proc.returncode}): {proc.stderr.strip()}"
        )
    return proc.stdout.strip()


def run_python_script(script_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a Python script with `REPO_ROOT` on `PYTHONPATH`."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    return subprocess.run(
        [sys.executable, "-B", str(script_path), *args],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def read_json_file(path: Path) -> Any:
    """Read a JSON file tolerating optional UTF-8 BOM."""
    return json.loads(path.read_text(encoding="utf-8-sig"))


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def exact_sign_flip_p_ref(diffs: Sequence[float]) -> float:
    """Independent general-G (G <= 16) exact two-sided sign-flip permutation p-value oracle."""
    d = np.asarray(diffs, dtype=np.float64).ravel()
    g = len(d)
    if g <= 1:
        return 1.0
    if g > 16:
        raise ValueError(f"Exact enumeration limited to G <= 16, got {g}")
    n_states = 1 << g
    obs = abs(float(np.mean(d)))
    tol = 1e-12
    bits = ((np.arange(n_states, dtype=np.int64)[:, None] >> np.arange(g, dtype=np.int64)[None, :]) & 1)
    signs = np.where(bits == 1, 1.0, -1.0)
    means = (signs @ d) / float(g)
    return float(np.mean(np.abs(means) >= obs - tol))


def holm_adjust_ref(p_values: Sequence[float]) -> List[float]:
    """Independent Holm-Bonferroni step-down adjuster oracle."""
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: float(p_values[i]))
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, float(p_values[idx]) * float(m - rank))
        running = max(running, val)
        adj[idx] = running
    return adj


def one_sided_u95_ref(diffs: Sequence[float]) -> float:
    """Independent one-sided 95% Student-t (df = G - 1) upper bound oracle."""
    d = np.asarray(diffs, dtype=np.float64).ravel()
    g = len(d)
    if g < 2:
        raise ValueError(f"Need at least 2 groups for Student-t U95, got {g}")
    mean_d = float(np.mean(d))
    sd_d = float(np.std(d, ddof=1))
    se_d = sd_d / math.sqrt(float(g))
    t_crit = float(student_t.ppf(0.95, df=g - 1))
    return mean_d + t_crit * se_d


def ece_from_llrs(llrs: np.ndarray, bits: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error from bit LLRs (`positive = bit 1`) and binary bits."""
    probs = 1.0 / (1.0 + np.exp(-np.clip(np.asarray(llrs, dtype=np.float64).ravel(), -40.0, 40.0)))
    y = np.asarray(bits, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        cnt = int(np.sum(mask))
        if cnt > 0:
            ece += (cnt / n) * abs(float(np.mean(y[mask])) - float(np.mean(probs[mask])))
    return float(ece)
