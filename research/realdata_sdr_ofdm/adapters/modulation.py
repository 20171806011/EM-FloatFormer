"""Exact non-Gray separable QAM/QPSK modulation and LLR helpers matching sdr_ofdm_dataset_v1."""

from __future__ import annotations

from functools import lru_cache

import numpy as np


def _validate_qm(qm: int) -> None:
    if not isinstance(qm, (int, np.integer)) or int(qm) < 2 or int(qm) % 2 != 0:
        raise ValueError(f"qm must be an even integer >= 2 (e.g. 2 or 4), got {qm!r}")


@lru_cache(maxsize=4)
def constellation_points(qm: int = 4) -> np.ndarray:
    """Return unit-average-energy complex128 constellation matching OFDM_SDR_Functions_torch_v2.mapping_table(qm)."""
    _validate_qm(qm)
    qm_int = int(qm)
    size = 1 << (qm_int // 2)
    a = np.arange(size, dtype=np.float64)
    b = a - a.mean()
    c = (b[:, None] + 1j * b[None, :]).reshape(-1).astype(np.complex128)
    c /= np.sqrt(np.mean(np.abs(c) ** 2))
    c.setflags(write=False)
    return c


@lru_cache(maxsize=4)
def constellation_bits(qm: int = 4) -> np.ndarray:
    """Return read-only uint8 array of shape (2**qm, qm) with big-endian bit tuples."""
    _validate_qm(qm)
    qm_int = int(qm)
    m = 1 << qm_int
    shifts = np.arange(qm_int - 1, -1, -1, dtype=np.int64)
    idx = np.arange(m, dtype=np.int64)[:, None]
    bits = ((idx >> shifts[None, :]) & 1).astype(np.uint8)
    bits.setflags(write=False)
    return bits


@lru_cache(maxsize=4)
def _bit_index_groups(qm: int = 4) -> tuple[np.ndarray, np.ndarray]:
    bit_tbl = constellation_bits(qm)
    idx1 = np.stack([np.flatnonzero(bit_tbl[:, k] == 1) for k in range(qm)], axis=0)
    idx0 = np.stack([np.flatnonzero(bit_tbl[:, k] == 0) for k in range(qm)], axis=0)
    idx1.setflags(write=False)
    idx0.setflags(write=False)
    return idx1, idx0


def mapping_table(qm: int = 4) -> tuple[dict[tuple[int, ...], complex], dict[complex, tuple[int, ...]]]:
    """Return (mapping, demapping) dicts matching OFDM_SDR_Functions_torch_v2.mapping_table(qm)."""
    pts = constellation_points(qm)
    bit_tbl = constellation_bits(qm)
    mapping: dict[tuple[int, ...], complex] = {}
    demapping: dict[complex, tuple[int, ...]] = {}
    for i in range(len(pts)):
        bt = tuple(int(x) for x in bit_tbl[i])
        val = complex(pts[i])
        mapping[bt] = val
        demapping[val] = bt
    return mapping, demapping


def bits_to_symbols(bits: np.ndarray, qm: int = 4) -> np.ndarray:
    """Map binary array of shape (..., qm) to complex128 constellation symbols of shape (...)."""
    _validate_qm(qm)
    bits_arr = np.asarray(bits)
    if bits_arr.ndim < 1 or bits_arr.shape[-1] != qm:
        raise ValueError(f"Expected bits shape (..., {qm}), got {bits_arr.shape}")
    if not np.all((bits_arr == 0) | (bits_arr == 1)):
        raise ValueError("bits must contain only 0 and 1")
    bits_u8 = bits_arr.astype(np.int64, copy=False)
    weights = (1 << np.arange(qm - 1, -1, -1, dtype=np.int64))
    idx = np.sum(bits_u8 * weights, axis=-1)
    return constellation_points(qm)[idx].astype(np.complex128)


def symbols_to_hard_bits(symbols: np.ndarray, qm: int = 4) -> np.ndarray:
    """Minimum-Euclidean-distance hard bit demapper returning uint8 array of shape (*symbols.shape, qm)."""
    _validate_qm(qm)
    sym_arr = np.asarray(symbols, dtype=np.complex128)
    if not np.all(np.isfinite(sym_arr)):
        raise ValueError("symbols contains non-finite values")
    pts = constellation_points(qm)
    sq_dist = np.abs(sym_arr[..., None] - pts) ** 2
    idx = np.argmin(sq_dist, axis=-1)
    return constellation_bits(qm)[idx].copy()


def constellation_log_likelihood(
    z: np.ndarray,
    n0: float | np.ndarray,
    qm: int = 4,
    *,
    min_n0: float = 1e-12,
    max_n0: float = 1e12,
    include_norm: bool = True,
) -> np.ndarray:
    """Compute exact complex AWGN log-likelihood log p(z | s_m, n0) over all 2**qm points.

    Model: z = s_m + w, w ~ CN(0, n0). Output shape: (*z.shape, 2**qm).
    Noise variance `n0` must be finite and >= 0.0 (strictly negative values raise `ValueError`).
    Values in `[0.0, min_n0)` (including exact `0.0`) are clamped to `min_n0` (default `1e-12`)
    and values above `max_n0` are clamped to `max_n0` (default `1e12`).
    """
    _validate_qm(qm)
    z_arr = np.asarray(z, dtype=np.complex128)
    if not np.all(np.isfinite(z_arr)):
        raise ValueError("z contains non-finite values")
    n0_arr = np.broadcast_to(np.asarray(n0, dtype=np.float64), z_arr.shape)
    if not np.all(np.isfinite(n0_arr)):
        raise ValueError("n0 contains non-finite values")
    if np.any(n0_arr < 0.0):
        raise ValueError("n0 must be non-negative (>= 0.0)")
    if isinstance(min_n0, (bool, np.bool_)) or isinstance(max_n0, (bool, np.bool_)):
        raise ValueError(
            f"min_n0 and max_n0 must not be bool, got min_n0={min_n0!r}, max_n0={max_n0!r}"
        )
    min_n0_f = float(min_n0)
    max_n0_f = float(max_n0)
    if (
        not (np.isfinite(min_n0_f) and np.isfinite(max_n0_f))
        or min_n0_f <= 0.0
        or max_n0_f < min_n0_f
    ):
        raise ValueError(f"Expected finite 0 < min_n0 <= max_n0, got min_n0={min_n0}, max_n0={max_n0}")
    n0_eff = np.clip(n0_arr, min_n0_f, max_n0_f)[..., None]
    pts = constellation_points(qm)
    sq_dist = np.abs(z_arr[..., None] - pts) ** 2
    ll = -sq_dist / n0_eff
    if include_norm:
        ll = ll - np.log(np.pi * n0_eff)
    return ll.astype(np.float64)


def exact_llr(
    z: np.ndarray,
    n0: float | np.ndarray,
    qm: int = 4,
    *,
    min_n0: float = 1e-12,
) -> np.ndarray:
    """Exact bit-wise LLRs with sign convention positive = bit 1.

    Requires finite `n0 >= 0.0` (strictly negative values raise `ValueError`); `n0` in
    `[0.0, min_n0)` (including `0.0`) is clamped to `min_n0`.
    """
    ll = constellation_log_likelihood(z, n0, qm=qm, min_n0=min_n0, include_norm=False)
    idx1, idx0 = _bit_index_groups(qm)
    ll1 = ll[..., idx1]
    m1 = np.max(ll1, axis=-1, keepdims=True)
    lse1 = m1[..., 0] + np.log(np.sum(np.exp(np.clip(ll1 - m1, -700.0, 0.0)), axis=-1))

    ll0 = ll[..., idx0]
    m0 = np.max(ll0, axis=-1, keepdims=True)
    lse0 = m0[..., 0] + np.log(np.sum(np.exp(np.clip(ll0 - m0, -700.0, 0.0)), axis=-1))
    return (lse1 - lse0).astype(np.float64)


def maxlog_llr(
    z: np.ndarray,
    n0: float | np.ndarray,
    qm: int = 4,
    *,
    min_n0: float = 1e-12,
) -> np.ndarray:
    """Exact max-log bit-wise LLRs over all 2**qm points with sign convention positive = bit 1.

    Requires finite `n0 >= 0.0` (strictly negative values raise `ValueError`); `n0` in
    `[0.0, min_n0)` (including `0.0`) is clamped to `min_n0`.
    """
    ll = constellation_log_likelihood(z, n0, qm=qm, min_n0=min_n0, include_norm=False)
    idx1, idx0 = _bit_index_groups(qm)
    return (np.max(ll[..., idx1], axis=-1) - np.max(ll[..., idx0], axis=-1)).astype(np.float64)
