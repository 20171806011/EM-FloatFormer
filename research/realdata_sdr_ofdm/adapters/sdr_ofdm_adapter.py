"""Unified data adapter for rikluost/sdr_ofdm_dataset_v1 (F4 & F5)."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .modulation import bits_to_symbols

# Locked waveform constants from rikluost/sdr_ofdm_dataset_v1 & ofdm-plutosdr-pytorch
N_SYMB = 14
N_FFT = 128
N_ACTIVE = 102
FFT_OFFSET = 13
CP_LEN = 7
SCS_HZ = 15_000.0
FS_HZ = 1_920_000.0

ACTIVE_START = 13
ACTIVE_END = 115
DC_BIN = 64
DC_ACTIVE_IDX = 51
PILOT_SYMBOL_IDX = 2

PILOT_ACTIVE_INDICES: tuple[int, ...] = (0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 101)
PILOT_FFT_INDICES: tuple[int, ...] = tuple(FFT_OFFSET + idx for idx in PILOT_ACTIVE_INDICES)

_PILOT_VALS = np.tile(
    np.array([-0.7 - 0.7j, -0.7 + 0.7j, 0.7 - 0.7j, 0.7 + 0.7j], dtype=np.complex128), 4
)[:14]
_PILOT_VALS.setflags(write=False)
PILOT_VALUES: np.ndarray = _PILOT_VALS

GUARD_FFT_INDICES: tuple[int, ...] = tuple(range(0, 13)) + tuple(range(115, 128))
# Empirically verified 825 kHz Butterworth LPF flat-passband guard bins (2-3 bins from active edges, inside |k-64|<55)
SAFE_GUARD_FFT_INDICES: tuple[int, ...] = (10, 11, 117, 118)
OUTER_GUARD_FFT_INDICES: tuple[int, ...] = tuple(range(0, 9)) + tuple(range(120, 128))

N_DATA_RES = 1400
QM = 4
N_BITS = 5600


class CustomDataset(Dataset):
    """Replica of rikluost/sdr_ofdm_dataset_v1/config.py CustomDataset for safe unpickling."""

    def __init__(self) -> None:
        self.pdsch_iq: list[Any] = []
        self.labels: list[Any] = []
        self.sinr: list[Any] = []

    def __len__(self) -> int:
        return len(self.pdsch_iq)

    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        return self.pdsch_iq[index], self.labels[index], self.sinr[index]

    def add_item(self, new_pdsch_iq: Any, new_label: Any, new_sinr: Any) -> None:
        self.pdsch_iq.append(new_pdsch_iq)
        self.labels.append(new_label)
        self.sinr.append(new_sinr)


_ALLOWED_PICKLE_GLOBALS: frozenset[tuple[str, str]] = frozenset(
    {
        ("collections", "OrderedDict"),
        ("_codecs", "encode"),
        ("builtins", "bytearray"),
        ("builtins", "complex"),
        ("builtins", "set"),
        ("builtins", "slice"),
        ("torch", "Tensor"),
        ("torch", "Size"),
        ("torch", "device"),
        ("torch", "FloatStorage"),
        ("torch", "DoubleStorage"),
        ("torch", "HalfStorage"),
        ("torch", "BFloat16Storage"),
        ("torch", "ComplexFloatStorage"),
        ("torch", "ComplexDoubleStorage"),
        ("torch", "LongStorage"),
        ("torch", "IntStorage"),
        ("torch", "ShortStorage"),
        ("torch", "CharStorage"),
        ("torch", "ByteStorage"),
        ("torch", "BoolStorage"),
        ("torch.storage", "TypedStorage"),
        ("torch.storage", "UntypedStorage"),
        ("torch._utils", "_rebuild_tensor"),
        ("torch._utils", "_rebuild_tensor_v2"),
        ("torch._utils", "_rebuild_tensor_v3"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
    }
)


class _CompatUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if name == "CustomDataset" and (
            module in {"config", "__main__", __name__}
            or module.endswith("sdr_ofdm_adapter")
        ):
            return CustomDataset
        if (module, name) in _ALLOWED_PICKLE_GLOBALS:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"Disallowed pickle global: {module}.{name}")


def _compat_load(file: Any, **kwargs: Any) -> Any:
    return _CompatUnpickler(file, **kwargs).load()


def _compat_loads(data: bytes | bytearray, **kwargs: Any) -> Any:
    return _CompatUnpickler(io.BytesIO(data), **kwargs).load()


class _CompatPickleModule:
    Unpickler = _CompatUnpickler
    load = staticmethod(_compat_load)
    loads = staticmethod(_compat_loads)
    dump = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)
    HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL
    DEFAULT_PROTOCOL = pickle.DEFAULT_PROTOCOL


@dataclass
class RealOFDMFrame:
    """Canonical per-TTI frame contract shared across adapters, receivers, and evaluator."""

    frame_id: int
    split: str
    group_id: str
    rx_grid: np.ndarray
    pilot_mask: np.ndarray
    dc_mask: np.ndarray
    data_mask: np.ndarray
    pilot_values: np.ndarray
    tx_bits: np.ndarray
    tx_symbols: np.ndarray
    reported_sinr_db: float
    est_n0: float
    meta: dict[str, Any] = field(default_factory=dict)


def build_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return canonical disjoint boolean masks (pilot_mask, dc_mask, data_mask) of shape (14, 128)."""
    pilot_mask = np.zeros((N_SYMB, N_FFT), dtype=bool)
    pilot_mask[PILOT_SYMBOL_IDX, list(PILOT_FFT_INDICES)] = True

    dc_mask = np.zeros((N_SYMB, N_FFT), dtype=bool)
    dc_mask[:, DC_BIN] = True

    data_mask = np.zeros((N_SYMB, N_FFT), dtype=bool)
    data_mask[:, ACTIVE_START:ACTIVE_END] = True
    data_mask &= ~dc_mask
    data_mask &= ~pilot_mask

    pilot_mask.setflags(write=False)
    dc_mask.setflags(write=False)
    data_mask.setflags(write=False)
    return pilot_mask, dc_mask, data_mask


def active_rx_grid(frame: RealOFDMFrame) -> np.ndarray:
    """Return active subcarrier slice (14, 102) covering FFT bins 13..114."""
    return frame.rx_grid[:, ACTIVE_START:ACTIVE_END]


def build_heldout_pilot_masks(
    n_folds: int = 4,
) -> list[dict[str, Any]]:
    """Generate deterministic strided CV folds over interior pilots [1..12]; endpoints 0,13 always observed."""
    if n_folds < 2 or n_folds > 12:
        raise ValueError(f"n_folds must be in [2, 12], got {n_folds}")
    folds: list[dict[str, Any]] = []
    for f in range(n_folds):
        heldout = list(range(1 + f, 13, n_folds))
        obs = [i for i in range(14) if i not in heldout]
        folds.append(
            {
                "fold": f,
                "heldout_pilot_indices": heldout,
                "obs_pilot_indices": obs,
                "heldout_fft_indices": [PILOT_FFT_INDICES[i] for i in heldout],
                "obs_fft_indices": [PILOT_FFT_INDICES[i] for i in obs],
            }
        )
    return folds


def pilot_cv_masks(
    pilot_mask: np.ndarray,
    heldout_indices: list[int] | tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Split pilot_mask (14, 128) into disjoint (obs_mask_128, heldout_mask_128)."""
    heldout_set = set(int(i) for i in heldout_indices)
    for i in heldout_set:
        if not (0 <= i < 14):
            raise ValueError(f"Pilot index out of range [0, 13]: {i}")
    heldout_mask = np.zeros_like(pilot_mask, dtype=bool)
    for i in heldout_set:
        heldout_mask[PILOT_SYMBOL_IDX, PILOT_FFT_INDICES[i]] = True
    heldout_mask &= pilot_mask
    obs_mask = pilot_mask & ~heldout_mask
    return obs_mask, heldout_mask


def estimate_blind_n0(
    rx_grid: np.ndarray,
    *,
    safe_guard_indices: tuple[int, ...] = SAFE_GUARD_FFT_INDICES,
    n0_floor: float = 1e-8,
) -> tuple[float, dict[str, Any]]:
    """Blind noise-variance estimator using passband guard bins with pilot 2nd-diff fallback.

    Strictly uses only observable rx_grid guard/pilot REs — ZERO label access.
    """
    n0_floor_f = float(n0_floor)
    if not np.isfinite(n0_floor_f) or n0_floor_f <= 0.0:
        raise ValueError(f"n0_floor must be finite and > 0, got {n0_floor!r}")
    if len(safe_guard_indices) == 0:
        raise ValueError("safe_guard_indices must be non-empty")

    grid = np.asarray(rx_grid, dtype=np.complex128)
    if grid.shape != (N_SYMB, N_FFT):
        raise ValueError(f"Expected rx_grid shape {(N_SYMB, N_FFT)}, got {grid.shape}")
    if not np.all(np.isfinite(grid)):
        raise ValueError("rx_grid contains non-finite values")

    g_safe = grid[:, list(safe_guard_indices)]
    safe_power = np.abs(g_safe) ** 2
    guard_n0_mean = float(np.mean(safe_power))
    guard_n0_robust = float(np.median(safe_power) / np.log(2.0))

    g_all = grid[:, list(GUARD_FFT_INDICES)]
    all_guard_n0_mean = float(np.mean(np.abs(g_all) ** 2))
    g_outer = grid[:, list(OUTER_GUARD_FFT_INDICES)]
    outer_guard_n0_mean = float(np.mean(np.abs(g_outer) ** 2))

    # 2nd-difference over the 13 uniformly spaced pilots (indices 0..12, spacing 8 bins)
    h_p = grid[PILOT_SYMBOL_IDX, list(PILOT_FFT_INDICES[:13])] / PILOT_VALUES[:13]
    d2 = h_p[2:] - 2.0 * h_p[1:-1] + h_p[:-2]
    pilot_diff2_n0 = float(0.98 * np.mean(np.abs(d2) ** 2) / 6.0)

    if guard_n0_mean > 1e-10:
        if guard_n0_mean > 3.0 * max(guard_n0_robust, 1e-12):
            n0_raw = guard_n0_robust
            n0_source = "guard_passband_median"
        else:
            n0_raw = guard_n0_mean
            n0_source = "guard_passband_mean"
    else:
        n0_raw = pilot_diff2_n0
        n0_source = "pilot_diff2_fallback"

    if not (
        np.isfinite(guard_n0_mean)
        and np.isfinite(guard_n0_robust)
        and np.isfinite(all_guard_n0_mean)
        and np.isfinite(outer_guard_n0_mean)
        and np.isfinite(pilot_diff2_n0)
    ):
        raise ValueError("Non-finite intermediate blind noise estimate")

    est_n0 = float(max(n0_raw, n0_floor_f))
    if not np.isfinite(est_n0) or est_n0 <= 0.0:
        raise ValueError(f"Estimated n0 is invalid: {est_n0!r}")
    info = {
        "guard_n0_mean": guard_n0_mean,
        "guard_n0_robust": guard_n0_robust,
        "all_guard_n0_mean": all_guard_n0_mean,
        "outer_guard_n0_mean": outer_guard_n0_mean,
        "pilot_diff2_n0": pilot_diff2_n0,
        "n0_source": n0_source,
    }
    return est_n0, info


def _validate_max_frames(max_frames: Optional[int]) -> Optional[int]:
    if max_frames is None:
        return None
    if isinstance(max_frames, bool):
        raise ValueError(f"max_frames must be a non-negative integer or None, got {max_frames!r}")
    val = int(max_frames)
    if val < 0:
        raise ValueError(f"max_frames must be >= 0, got {max_frames}")
    return val


def load_raw_sdr_ofdm(
    raw_path: str | Path,
    *,
    max_frames: Optional[int] = None,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Load and validate raw CustomDataset .pth without polluting sys.modules['config']."""
    max_frames_int = _validate_max_frames(max_frames)
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {path}")

    dataset = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
        pickle_module=_CompatPickleModule,
    )
    for attr in ("pdsch_iq", "labels", "sinr"):
        if not hasattr(dataset, attr):
            raise ValueError(f"Unpickled object missing required attribute: {attr}")

    n_total = len(dataset.pdsch_iq)
    if len(dataset.labels) != n_total or len(dataset.sinr) != n_total:
        raise ValueError("Mismatched lengths across pdsch_iq, labels, sinr")

    n_take = n_total if max_frames_int is None else min(max_frames_int, n_total)
    iq_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    sinr_arr = np.empty(n_take, dtype=np.float64)

    for i in range(n_take):
        iq_i = (
            dataset.pdsch_iq[i].detach().cpu().numpy()
            if torch.is_tensor(dataset.pdsch_iq[i])
            else np.asarray(dataset.pdsch_iq[i])
        )
        lab_i = (
            dataset.labels[i].detach().cpu().numpy()
            if torch.is_tensor(dataset.labels[i])
            else np.asarray(dataset.labels[i])
        )
        if iq_i.shape != (N_SYMB, N_FFT):
            raise ValueError(f"Frame {i} iq shape {iq_i.shape} != {(N_SYMB, N_FFT)}")
        if not np.all(np.isfinite(iq_i)):
            raise ValueError(f"Frame {i} iq contains non-finite values")
        if lab_i.shape != (N_DATA_RES, QM):
            raise ValueError(f"Frame {i} labels shape {lab_i.shape} != {(N_DATA_RES, QM)}")
        if not np.all(np.isfinite(lab_i)) or not np.all((lab_i == 0) | (lab_i == 1)):
            raise ValueError(f"Frame {i} labels contain non-binary values")
        sinr_val = float(dataset.sinr[i])
        if not np.isfinite(sinr_val):
            raise ValueError(f"Frame {i} sinr is non-finite: {sinr_val}")

        iq_list.append(iq_i.astype(np.complex128))
        label_list.append(lab_i.astype(np.uint8))
        sinr_arr[i] = sinr_val

    return iq_list, label_list, sinr_arr


def build_frame(
    frame_id: int,
    iq_raw: np.ndarray,
    labels_raw: np.ndarray,
    sinr_val: float,
    *,
    split: str = "unassigned",
    group_id: str = "unassigned",
    raw_source_sha256: str = "",
) -> RealOFDMFrame:
    """Construct a normalized RealOFDMFrame from raw (14, 128) IQ and (1400, 4) bits."""
    pilot_mask, dc_mask, data_mask = build_masks()
    iq_c = np.asarray(iq_raw, dtype=np.complex128)
    if iq_c.shape != (N_SYMB, N_FFT):
        raise ValueError(f"Expected iq_raw shape {(N_SYMB, N_FFT)}, got {iq_c.shape}")
    if not np.all(np.isfinite(iq_c)):
        raise ValueError("iq_raw contains non-finite values")

    labels_arr = np.asarray(labels_raw)
    if labels_arr.shape != (N_DATA_RES, QM):
        raise ValueError(f"Expected labels shape {(N_DATA_RES, QM)}, got {labels_arr.shape}")
    if not np.all(np.isfinite(labels_arr)) or not np.all((labels_arr == 0) | (labels_arr == 1)):
        raise ValueError("labels must be binary {0, 1}")
    tx_bits = labels_arr.astype(np.uint8, copy=True)

    sinr_f = float(sinr_val)
    if not np.isfinite(sinr_f):
        raise ValueError(f"sinr_val must be finite, got {sinr_val!r}")

    active_mask = pilot_mask | data_mask
    rms_active = float(np.sqrt(np.mean(np.abs(iq_c[active_mask]) ** 2)))
    if not np.isfinite(rms_active):
        raise ValueError(f"Active RMS power is non-finite: {rms_active!r}")
    s0 = max(rms_active, 1e-12)
    rx_grid = (iq_c / s0).astype(np.complex128)
    if not np.all(np.isfinite(rx_grid)):
        raise ValueError("Normalized rx_grid contains non-finite values")

    est_n0, n0_info = estimate_blind_n0(rx_grid)
    pilot_rms = float(np.sqrt(np.mean(np.abs(rx_grid[pilot_mask]) ** 2)))
    dc_power = float(np.mean(np.abs(rx_grid[dc_mask]) ** 2))
    tx_symbols = bits_to_symbols(tx_bits, qm=QM)

    tx_bits_sha256 = hashlib.sha256(tx_bits.tobytes()).hexdigest()
    rx_grid_sha256 = hashlib.sha256(np.ascontiguousarray(rx_grid).tobytes()).hexdigest()

    meta: dict[str, Any] = {
        "s0": s0,
        **n0_info,
        "pilot_rms": pilot_rms,
        "dc_power": dc_power,
        "heldout_pilot_folds": build_heldout_pilot_masks(4),
        "tx_bits_sha256": tx_bits_sha256,
        "rx_grid_sha256": rx_grid_sha256,
        "raw_source_sha256": raw_source_sha256,
    }
    return RealOFDMFrame(
        frame_id=int(frame_id),
        split=str(split),
        group_id=str(group_id),
        rx_grid=rx_grid,
        pilot_mask=pilot_mask,
        dc_mask=dc_mask,
        data_mask=data_mask,
        pilot_values=PILOT_VALUES,
        tx_bits=tx_bits,
        tx_symbols=tx_symbols,
        reported_sinr_db=sinr_f,
        est_n0=est_n0,
        meta=meta,
    )


def apply_stress_perturbation(
    frame: RealOFDMFrame,
    *,
    extra_awgn_db: Optional[float] = None,
    cfo_norm: float = 0.0,
    phase_drift_rad: float = 0.0,
    seed: Optional[int] = None,
    cp_len: int = CP_LEN,
) -> RealOFDMFrame:
    """Apply reproducible physical time-domain CFO/phase drift and Parseval AWGN without mutating frame.

    Args:
        frame: Input RealOFDMFrame (never mutated; tx_bits and tx_symbols are copied bit-identically).
        extra_awgn_db: Added AWGN SNR in dB (`added_awgn_snr_db`) relative to active-band mean power
            `p_active`. `None` (YAML `null`) applies no AWGN (`sigma2_add = 0.0`, clean baseline);
            numeric values inject complex Gaussian noise with variance
            `sigma2_add = p_active * 10**(-extra_awgn_db / 10.0)` (smaller dB = stronger noise;
            `0.0` adds noise variance equal to `p_active`, not clean).
        cfo_norm: Normalized carrier frequency offset (`0.0` = no CFO).
        phase_drift_rad: Linear common phase drift in radians across symbols (`0.0` = no drift).
        seed: RNG seed for reproducible AWGN draw when `extra_awgn_db is not None`.
        cp_len: Cyclic prefix length in samples used in continuous-time CFO phase progression.
    """
    if not np.all(np.isfinite(frame.rx_grid)):
        raise ValueError("frame.rx_grid contains non-finite values")
    if not np.isfinite(float(cfo_norm)) or not np.isfinite(float(phase_drift_rad)):
        raise ValueError("cfo_norm and phase_drift_rad must be finite")
    if extra_awgn_db is not None and not np.isfinite(float(extra_awgn_db)):
        raise ValueError(f"extra_awgn_db must be finite or None, got {extra_awgn_db!r}")
    if int(cp_len) < 0:
        raise ValueError(f"cp_len must be >= 0, got {cp_len}")

    x_time = np.fft.ifft(np.fft.ifftshift(frame.rx_grid, axes=-1), axis=-1, norm="ortho")

    s_idx = np.arange(N_SYMB, dtype=np.float64)[:, None]
    m_idx = np.arange(N_FFT, dtype=np.float64)[None, :]
    n_idx = s_idx * float(N_FFT + int(cp_len)) + m_idx

    cfo_phase = 2.0 * np.pi * float(cfo_norm) * (n_idx / float(N_FFT))
    drift_phase = float(phase_drift_rad) * ((s_idx - float(PILOT_SYMBOL_IDX)) / float(N_SYMB - 1))
    y_time = x_time * np.exp(1j * (cfo_phase + drift_phase))

    sigma2_add = 0.0
    if extra_awgn_db is not None:
        rng = np.random.default_rng(seed)
        active_mask = frame.pilot_mask | frame.data_mask
        p_active = float(np.mean(np.abs(frame.rx_grid[active_mask]) ** 2))
        sigma2_add = p_active * (10.0 ** (-float(extra_awgn_db) / 10.0))
        w_time = np.sqrt(sigma2_add / 2.0) * (
            rng.standard_normal(y_time.shape) + 1j * rng.standard_normal(y_time.shape)
        )
        y_time = y_time + w_time

    rx_pert = np.fft.fftshift(np.fft.fft(y_time, axis=-1, norm="ortho"), axes=-1).astype(np.complex128)
    est_n0_pert, n0_info_pert = estimate_blind_n0(rx_pert)

    new_meta = copy.deepcopy(frame.meta)
    new_meta.update(n0_info_pert)
    new_meta["rx_grid_sha256"] = hashlib.sha256(np.ascontiguousarray(rx_pert).tobytes()).hexdigest()
    new_meta["perturbation"] = {
        "extra_awgn_db": None if extra_awgn_db is None else float(extra_awgn_db),
        "sigma2_add": float(sigma2_add),
        "cfo_norm": float(cfo_norm),
        "phase_drift_rad": float(phase_drift_rad),
        "seed": seed,
        "cp_len": int(cp_len),
    }

    return RealOFDMFrame(
        frame_id=frame.frame_id,
        split=frame.split,
        group_id=frame.group_id,
        rx_grid=rx_pert,
        pilot_mask=frame.pilot_mask,
        dc_mask=frame.dc_mask,
        data_mask=frame.data_mask,
        pilot_values=frame.pilot_values,
        tx_bits=frame.tx_bits.copy(),
        tx_symbols=frame.tx_symbols.copy(),
        reported_sinr_db=frame.reported_sinr_db,
        est_n0=est_n0_pert,
        meta=new_meta,
    )


def compute_temporal_diagnostics(
    sinr_vals: np.ndarray,
    h_pilots: np.ndarray,
    payload_hashes: list[str],
    *,
    max_lag: int = 32,
    autocorr_threshold: float = 0.10,
    min_guard_frames: int = 4,
    max_guard_frames: int = 16,
) -> dict[str, Any]:
    """Compute payload deduplication stats and lag 1..max_lag autocorrelation of SINR and h_pilot."""
    n = len(sinr_vals)
    if len(h_pilots) != n or len(payload_hashes) != n:
        raise ValueError("Mismatched input lengths for temporal diagnostics")

    hash_to_indices: dict[str, list[int]] = {}
    for idx, h in enumerate(payload_hashes):
        hash_to_indices.setdefault(h, []).append(idx)
    dup_groups = {h: idxs for h, idxs in hash_to_indices.items() if len(idxs) > 1}

    sinr_lags: dict[str, float] = {}
    h_coherent_lags: dict[str, float] = {}
    h_envelope_lags: dict[str, float] = {}

    eff_max_lag = min(int(max_lag), max(1, n - 2))
    sinr_f = np.asarray(sinr_vals, dtype=np.float64)
    h_f = np.asarray(h_pilots, dtype=np.complex128)
    h_norms = np.linalg.norm(h_f, axis=1)

    for k in range(1, eff_max_lag + 1):
        a = sinr_f[:-k]
        b = sinr_f[k:]
        sa = float(np.std(a))
        sb = float(np.std(b))
        if sa > 1e-12 and sb > 1e-12:
            corr_s = float(np.mean((a - np.mean(a)) * (b - np.mean(b))) / (sa * sb))
        else:
            corr_s = 0.0
        sinr_lags[str(k)] = corr_s

        inner = np.sum(h_f[:-k] * np.conj(h_f[k:]), axis=1)
        denom_coh = float(np.mean(h_norms[:-k] * h_norms[k:]))
        corr_h_coh = float(np.abs(np.mean(inner)) / max(denom_coh, 1e-18))
        corr_h_env = float(np.mean(np.abs(inner) / np.maximum(h_norms[:-k] * h_norms[k:], 1e-18)))
        h_coherent_lags[str(k)] = corr_h_coh
        h_envelope_lags[str(k)] = corr_h_env

    tau_c = None
    for k in range(1, eff_max_lag + 1):
        if (
            abs(sinr_lags[str(k)]) < autocorr_threshold
            and h_coherent_lags[str(k)] < autocorr_threshold
        ):
            tau_c = k
            break
    if tau_c is None:
        tau_c = max_guard_frames

    guard_width = max(int(min_guard_frames), min(int(max_guard_frames), int(tau_c)))
    return {
        "n_tti": int(n),
        "n_unique_payloads": int(len(hash_to_indices)),
        "n_duplicate_groups": int(len(dup_groups)),
        "duplicate_groups": dup_groups,
        "sinr_autocorr_lags": sinr_lags,
        "h_pilot_coherent_corr_lags": h_coherent_lags,
        "h_pilot_envelope_corr_lags": h_envelope_lags,
        "tau_c": int(tau_c),
        "guard_width": int(guard_width),
    }


def build_leak_free_splits(
    sinr_vals: np.ndarray,
    h_pilots: np.ndarray,
    payload_hashes: list[str],
    *,
    dataset_sha256: str = "",
    n_macro_blocks: int = 12,
    ratios: Optional[dict[str, float]] = None,
    min_guard_frames: int = 4,
    max_guard_frames: int = 16,
    autocorr_threshold: float = 0.10,
    heldout_pilot_eval_indices: Optional[list[int]] = None,
    seed: int = 20261121,
) -> dict[str, Any]:
    """Build contiguous macro-block Train/Dev/Test splits with guard buffers and zero payload overlap."""
    if ratios is None:
        ratios = {"train": 0.60, "dev": 0.15, "test": 0.25}
    if heldout_pilot_eval_indices is None:
        heldout_pilot_eval_indices = [1, 5, 9, 12]

    n = len(sinr_vals)
    diag = compute_temporal_diagnostics(
        sinr_vals,
        h_pilots,
        payload_hashes,
        max_lag=32,
        autocorr_threshold=autocorr_threshold,
        min_guard_frames=min_guard_frames,
        max_guard_frames=max_guard_frames,
    )
    g = int(diag["guard_width"])

    m = min(int(n_macro_blocks), max(3, n // max(2 * g + 4, 10)))
    if m < 3:
        raise ValueError(f"Dataset too small ({n} frames) for >=3 macro-blocks with guard={g}")

    edges = np.linspace(0, n, m + 1, dtype=int)
    raw_spans = [(int(edges[b]), int(edges[b + 1])) for b in range(m)]

    guard_set: set[int] = set()
    block_core_indices: list[list[int]] = []
    left_trim = g // 2
    right_trim = g - left_trim

    for b, (start, end) in enumerate(raw_spans):
        core_start = start + (right_trim if b > 0 else 0)
        core_end = end - (left_trim if b < m - 1 else 0)
        for idx in range(start, core_start):
            guard_set.add(idx)
        for idx in range(core_end, end):
            guard_set.add(idx)
        if core_end <= core_start:
            raise ValueError(f"Macro-block {b} exhausted after trimming guard={g}: [{start}, {end})")
        block_core_indices.append(list(range(core_start, core_end)))

    # Stratify blocks by block-median SINR
    block_medians = [float(np.median(sinr_vals[idxs])) for idxs in block_core_indices]
    sorted_blocks = sorted(range(m), key=lambda b: (block_medians[b], b))

    n_test = max(1, int(round(float(ratios.get("test", 0.25)) * m)))
    n_dev = max(1, int(round(float(ratios.get("dev", 0.15)) * m)))
    if n_test + n_dev >= m:
        n_test = max(1, m // 4)
        n_dev = max(1, m // 6)
    n_train = m - n_test - n_dev

    # Pick evenly-spaced ranks for test and dev across SINR-sorted blocks
    test_ranks = set(np.linspace(1, m - 2, n_test, dtype=int).tolist()) if n_test > 1 else {m // 2}
    # Ensure exact cardinality for test_ranks
    rank_cursor = 0
    while len(test_ranks) < n_test:
        cand = (rank_cursor * 2 + 1) % m
        test_ranks.add(cand)
        rank_cursor += 1

    avail_ranks = [r for r in range(m) if r not in test_ranks]
    dev_pick_pos = np.linspace(0, len(avail_ranks) - 1, n_dev, dtype=int).tolist() if n_dev > 1 else [len(avail_ranks) // 2]
    dev_ranks = {avail_ranks[p] for p in dev_pick_pos}
    pos_cursor = 0
    while len(dev_ranks) < n_dev:
        dev_ranks.add(avail_ranks[pos_cursor])
        pos_cursor += 1

    block_split: dict[int, str] = {}
    for rank, b in enumerate(sorted_blocks):
        if rank in test_ranks:
            block_split[b] = "test"
        elif rank in dev_ranks:
            block_split[b] = "dev"
        else:
            block_split[b] = "train"
    assert sum(1 for v in block_split.values() if v == "train") == n_train

    # Enforce zero payload-hash overlap across splits (priority: test -> dev -> train, conflicting dups -> guard)
    split_indices: dict[str, list[int]] = {"train": [], "dev": [], "test": []}
    seen_hashes_by_split: dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
    blocks_out: list[dict[str, Any]] = []
    dedup_excluded_to_guard: list[int] = []

    # First pass: collect raw block assignments
    for b in range(m):
        sp = block_split[b]
        for idx in block_core_indices[b]:
            seen_hashes_by_split[sp].add(payload_hashes[idx])

    # Identify hashes appearing in multiple splits; keep in earliest-assigned priority (test, dev, train) -> move duplicates in other splits to guard
    cross_hashes = (
        (seen_hashes_by_split["train"] & seen_hashes_by_split["dev"])
        | (seen_hashes_by_split["train"] & seen_hashes_by_split["test"])
        | (seen_hashes_by_split["dev"] & seen_hashes_by_split["test"])
    )
    kept_cross_owner: dict[str, str] = {}
    for h in sorted(cross_hashes):
        if h in seen_hashes_by_split["test"]:
            kept_cross_owner[h] = "test"
        elif h in seen_hashes_by_split["dev"]:
            kept_cross_owner[h] = "dev"
        else:
            kept_cross_owner[h] = "train"

    for b in range(m):
        sp = block_split[b]
        kept_idxs: list[int] = []
        for idx in block_core_indices[b]:
            h = payload_hashes[idx]
            if h in kept_cross_owner and kept_cross_owner[h] != sp:
                guard_set.add(idx)
                dedup_excluded_to_guard.append(idx)
            else:
                kept_idxs.append(idx)
                split_indices[sp].append(idx)
        blocks_out.append(
            {
                "group_id": f"block_{b:02d}",
                "split": sp,
                "raw_span": list(raw_spans[b]),
                "indices": kept_idxs,
                "n_frames": len(kept_idxs),
                "sinr_median": float(np.median(sinr_vals[kept_idxs])) if kept_idxs else None,
                "sinr_min": float(np.min(sinr_vals[kept_idxs])) if kept_idxs else None,
                "sinr_max": float(np.max(sinr_vals[kept_idxs])) if kept_idxs else None,
            }
        )

    train_idx = sorted(split_indices["train"])
    dev_idx = sorted(split_indices["dev"])
    test_idx = sorted(split_indices["test"])
    guard_idx = sorted(guard_set)

    # Hard assertions
    s_tr, s_dv, s_te, s_gd = set(train_idx), set(dev_idx), set(test_idx), set(guard_idx)
    assert s_tr.isdisjoint(s_dv) and s_tr.isdisjoint(s_te) and s_tr.isdisjoint(s_gd)
    assert s_dv.isdisjoint(s_te) and s_dv.isdisjoint(s_gd) and s_te.isdisjoint(s_gd)
    assert (s_tr | s_dv | s_te | s_gd) == set(range(n))

    h_tr = {payload_hashes[i] for i in train_idx}
    h_dv = {payload_hashes[i] for i in dev_idx}
    h_te = {payload_hashes[i] for i in test_idx}
    assert len(h_tr & h_dv) == 0 and len(h_tr & h_te) == 0 and len(h_dv & h_te) == 0

    # Verify boundary separation >= g + 1 across different active splits
    label_of = {}
    for i in train_idx:
        label_of[i] = "train"
    for i in dev_idx:
        label_of[i] = "dev"
    for i in test_idx:
        label_of[i] = "test"
    active_sorted = sorted(label_of.keys())
    min_cross_dist = n
    for pos in range(len(active_sorted) - 1):
        i0, i1 = active_sorted[pos], active_sorted[pos + 1]
        if label_of[i0] != label_of[i1]:
            dist = i1 - i0
            if dist < min_cross_dist:
                min_cross_dist = dist
            assert dist >= g + 1, f"Cross-split boundary gap {dist} < {g + 1} at ({i0}, {i1})"

    diag["dedup_excluded_to_guard"] = dedup_excluded_to_guard
    diag["min_cross_split_distance"] = int(min_cross_dist)

    return {
        "schema_version": "1.0",
        "seed": int(seed),
        "dataset_sha256": dataset_sha256,
        "diagnostics": diag,
        "counts": {
            "total": int(n),
            "train": len(train_idx),
            "dev": len(dev_idx),
            "test": len(test_idx),
            "guard": len(guard_idx),
        },
        "sinr_by_split": {
            sp: {
                "min": float(np.min(sinr_vals[idxs])) if idxs else None,
                "median": float(np.median(sinr_vals[idxs])) if idxs else None,
                "mean": float(np.mean(sinr_vals[idxs])) if idxs else None,
                "max": float(np.max(sinr_vals[idxs])) if idxs else None,
            }
            for sp, idxs in (("train", train_idx), ("dev", dev_idx), ("test", test_idx), ("guard", guard_idx))
        },
        "blocks": blocks_out,
        "train_indices": train_idx,
        "dev_indices": dev_idx,
        "test_indices": test_idx,
        "guard_indices": guard_idx,
        "heldout_pilot_eval_indices": list(heldout_pilot_eval_indices),
    }


def preprocess_and_save(
    raw_path: str | Path,
    splits_manifest_path: str | Path,
    proc_dir: str | Path,
    metadata_jsonl_path: str | Path,
    proc_checksums_path: str | Path,
    *,
    max_frames: Optional[int] = None,
) -> dict[str, Any]:
    """Preprocess raw dataset into canonical RealOFDMFrames and save split .npz + manifests."""
    raw_p = Path(raw_path)
    raw_sha256 = hashlib.sha256(raw_p.read_bytes()).hexdigest()
    splits_data = json.loads(Path(splits_manifest_path).read_text(encoding="utf-8"))

    idx_to_split: dict[int, str] = {}
    idx_to_group: dict[int, str] = {}
    for blk in splits_data["blocks"]:
        for idx in blk["indices"]:
            idx_to_split[int(idx)] = blk["split"]
            idx_to_group[int(idx)] = blk["group_id"]
    for idx in splits_data["guard_indices"]:
        idx_to_split[int(idx)] = "guard"
        idx_to_group[int(idx)] = "guard"

    iq_list, lab_list, sinr_arr = load_raw_sdr_ofdm(raw_p, max_frames=max_frames)
    n = len(iq_list)
    frames: list[RealOFDMFrame] = []
    for i in range(n):
        fr = build_frame(
            frame_id=i,
            iq_raw=iq_list[i],
            labels_raw=lab_list[i],
            sinr_val=float(sinr_arr[i]),
            split=idx_to_split.get(i, "unassigned"),
            group_id=idx_to_group.get(i, "unassigned"),
            raw_source_sha256=raw_sha256,
        )
        frames.append(fr)

    proc_root = Path(proc_dir)
    proc_root.mkdir(parents=True, exist_ok=True)
    meta_out = Path(metadata_jsonl_path)
    meta_out.parent.mkdir(parents=True, exist_ok=True)

    with meta_out.open("w", encoding="utf-8", newline="\n") as mf:
        for fr in frames:
            rec = {
                "frame_id": fr.frame_id,
                "split": fr.split,
                "group_id": fr.group_id,
                "reported_sinr_db": fr.reported_sinr_db,
                "s0": fr.meta["s0"],
                "est_n0": fr.est_n0,
                "guard_n0_mean": fr.meta["guard_n0_mean"],
                "guard_n0_robust": fr.meta["guard_n0_robust"],
                "pilot_diff2_n0": fr.meta["pilot_diff2_n0"],
                "n0_source": fr.meta["n0_source"],
                "pilot_rms": fr.meta["pilot_rms"],
                "dc_power": fr.meta["dc_power"],
                "tx_bits_sha256": fr.meta["tx_bits_sha256"],
                "rx_grid_sha256": fr.meta["rx_grid_sha256"],
                "raw_source_sha256": fr.meta["raw_source_sha256"],
            }
            mf.write(json.dumps(rec) + "\n")

    def _save_subset(sub_frames: list[RealOFDMFrame], out_path: Path) -> dict[str, Any]:
        np.savez(
            out_path,
            frame_ids=np.array([f.frame_id for f in sub_frames], dtype=np.int64),
            splits=np.array([f.split for f in sub_frames]),
            group_ids=np.array([f.group_id for f in sub_frames]),
            rx_grids=np.stack([f.rx_grid for f in sub_frames], axis=0).astype(np.complex128),
            tx_bits=np.stack([f.tx_bits for f in sub_frames], axis=0).astype(np.uint8),
            tx_symbols=np.stack([f.tx_symbols for f in sub_frames], axis=0).astype(np.complex128),
            reported_sinr_db=np.array([f.reported_sinr_db for f in sub_frames], dtype=np.float64),
            est_n0=np.array([f.est_n0 for f in sub_frames], dtype=np.float64),
            s0=np.array([f.meta["s0"] for f in sub_frames], dtype=np.float64),
            meta_json=np.array([json.dumps(f.meta) for f in sub_frames]),
        )
        raw_bytes = out_path.read_bytes()
        return {
            "filename": out_path.name,
            "local_path": str(out_path),
            "n_frames": len(sub_frames),
            "size_bytes": len(raw_bytes),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        }

    files_manifest: dict[str, Any] = {}
    if max_frames is None:
        files_manifest["all_frames"] = _save_subset(frames, proc_root / "all_frames.npz")
        for sp in ("train", "dev", "test", "guard"):
            sub = [f for f in frames if f.split == sp]
            files_manifest[sp] = _save_subset(sub, proc_root / f"{sp}.npz")
    else:
        files_manifest[f"smoke_{n}"] = _save_subset(frames, proc_root / f"smoke_{n}.npz")

    proc_manifest = {
        "raw_source_sha256": raw_sha256,
        "splits_manifest": str(splits_manifest_path),
        "n_total_processed": len(frames),
        "files": files_manifest,
    }
    proc_chk = Path(proc_checksums_path)
    proc_chk.parent.mkdir(parents=True, exist_ok=True)
    proc_chk.write_text(json.dumps(proc_manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return proc_manifest


def load_processed_split(
    proc_dir: str | Path,
    split: str = "test",
    *,
    max_frames: Optional[int] = None,
) -> list[RealOFDMFrame]:
    """Load preprocessed RealOFDMFrame list from /tmp/tw_realdata/data_proc/{split}.npz."""
    max_frames_int = _validate_max_frames(max_frames)
    npz_path = Path(proc_dir) / f"{split}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Processed split archive not found: {npz_path}")

    pilot_mask, dc_mask, data_mask = build_masks()
    frames: list[RealOFDMFrame] = []
    with np.load(npz_path, allow_pickle=False) as d:
        frame_ids_arr = d["frame_ids"]
        n_total = len(frame_ids_arr)
        n_take = n_total if max_frames_int is None else min(max_frames_int, n_total)
        if n_take == 0:
            del frame_ids_arr
            return frames
        splits_arr = d["splits"]
        group_ids_arr = d["group_ids"]
        rx_grids_arr = d["rx_grids"]
        tx_bits_arr = d["tx_bits"]
        tx_symbols_arr = d["tx_symbols"]
        reported_sinr_arr = d["reported_sinr_db"]
        est_n0_arr = d["est_n0"]
        meta_json_arr = d["meta_json"]

    for i in range(n_take):
        meta = json.loads(str(meta_json_arr[i]))
        frames.append(
            RealOFDMFrame(
                frame_id=int(frame_ids_arr[i]),
                split=str(splits_arr[i]),
                group_id=str(group_ids_arr[i]),
                rx_grid=np.array(rx_grids_arr[i], dtype=np.complex128, copy=True),
                pilot_mask=pilot_mask,
                dc_mask=dc_mask,
                data_mask=data_mask,
                pilot_values=PILOT_VALUES,
                tx_bits=np.array(tx_bits_arr[i], dtype=np.uint8, copy=True),
                tx_symbols=np.array(tx_symbols_arr[i], dtype=np.complex128, copy=True),
                reported_sinr_db=float(reported_sinr_arr[i]),
                est_n0=float(est_n0_arr[i]),
                meta=meta,
            )
        )

    del (
        frame_ids_arr,
        splits_arr,
        group_ids_arr,
        rx_grids_arr,
        tx_bits_arr,
        tx_symbols_arr,
        reported_sinr_arr,
        est_n0_arr,
        meta_json_arr,
    )
    return frames
