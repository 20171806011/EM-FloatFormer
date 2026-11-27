"""M5 Stage 4 16-bit dual-cache wire codec (`m5-c16-v1-le`).

Exact bit layout (16 bits MSB -> LSB, little-endian uint16 `<u2`):
    [2b turn | 2b s17_inner | 6b angle | 4b log_radial | 2b quality]

Invariants:
- `w >> 12` recovers the exact 4-bit legacy `s17` code (`0..15`).
- Quarter-turn `r in {0,1,2,3}` adds `r << 14` mod `2**16`, modifying ONLY the top 2 `turn` bits.
- `radial == 0` indicates erasure/unavailable/zero-matched-sum payload (lower 12 bits are `0`).
- Valid non-zero payloads encode `radial in 1..15` using `WIRE_RADIAL_EDGES`.
"""

from __future__ import annotations

import numpy as np

WIRE_ID = "m5-c16-v1-le"


def radial_edges(bits: int) -> np.ndarray:
    """Return expm1-spaced amplitude-to-sqrt-variance bin edges matching M5 §2."""
    if bits == 0:
        return np.array([0.0, np.inf], dtype=np.float64)
    edges = np.expm1(np.linspace(0.0, np.log(25.0), (1 << bits) + 1))
    edges[-1] = np.inf
    return edges.astype(np.float64)


WIRE_RADIAL_EDGES = np.r_[0.0, radial_edges(4)[2:]]


def quality_edges(bits: int) -> np.ndarray:
    """Return residual-energy-to-variance bin edges matching M5 §2."""
    if bits == 0:
        return np.array([0.0, np.inf], dtype=np.float64)
    if bits == 1:
        return np.array([0.0, 4.0, np.inf], dtype=np.float64)
    return np.r_[0.0, np.geomspace(0.75, 16.0, (1 << bits) - 1), np.inf].astype(np.float64)


WIRE_QUALITY_EDGES = quality_edges(2)


def encode_c16(
    legacy_s17: np.ndarray | int,
    t: np.ndarray | complex,
    physical_residual: np.ndarray | float,
    variance: np.ndarray | float,
    available: np.ndarray | bool = True,
) -> np.ndarray:
    """Encode `(legacy_s17, matched_sum t, residual, noise_variance, available)` into `<u2` words."""
    leg, tv, res, var, avail = np.broadcast_arrays(
        np.asarray(legacy_s17, dtype=np.int64),
        np.asarray(t, dtype=np.complex128),
        np.asarray(physical_residual, dtype=np.float64),
        np.asarray(variance, dtype=np.float64),
        np.asarray(available, dtype=bool),
    )
    if np.any((leg < 0) | (leg > 15)):
        raise ValueError("Legacy S17 prefix must lie in [0, 15].")
    if np.any(
        avail
        & (
            ~np.isfinite(tv)
            | ~np.isfinite(res)
            | (res < 0.0)
            | ~np.isfinite(var)
            | (var <= 0.0)
        )
    ):
        raise ValueError("Available samples must have finite matched statistic, non-negative residual, positive variance.")

    keep = avail & (tv != 0.0)
    turns = leg // 4
    phase = np.angle(np.where(keep, tv * ((1j) ** (-turns)), 1.0))
    angle = np.clip(np.floor(64.0 * (phase + np.pi / 4.0) / (np.pi / 2.0)), 0, 63).astype(np.int64)
    safe_var = np.where(avail, var, 1.0)
    radial = 1 + np.searchsorted(
        WIRE_RADIAL_EDGES[1:-1], np.abs(np.where(keep, tv, 0.0)) / np.sqrt(safe_var)
    )
    quality = np.searchsorted(
        WIRE_QUALITY_EDGES[1:-1], np.where(keep, res, 0.0) / safe_var
    )
    tail = (angle << 6) | (radial << 2) | quality
    words = ((leg << 12) | np.where(keep, tail, 0)).astype("<u2")
    return words


def decode_c16(words: np.ndarray | int) -> dict[str, np.ndarray]:
    """Unpack `<u2` words into dict of numpy arrays (preserving scalar shape if 0-d)."""
    w = np.asarray(words, dtype="<u2")
    s17 = (w >> 12).astype(np.int64)
    turn = (w >> 14).astype(np.int64)
    s17_inner = ((w >> 12) & 0x3).astype(np.int64)
    angle_bin = ((w >> 6) & 0x3F).astype(np.int64)
    radial_code = ((w >> 2) & 0xF).astype(np.int64)
    quality_bin = (w & 0x3).astype(np.int64)
    erased = radial_code == 0
    return {
        "s17": s17,
        "turn": turn,
        "s17_inner": s17_inner,
        "angle_bin": angle_bin,
        "radial_code": radial_code,
        "quality_bin": quality_bin,
        "erased": erased,
    }


def rotate_c16(words: np.ndarray | int, quarter_turns: np.ndarray | int) -> np.ndarray:
    """Apply exact C4 quarter-turn to `m5-c16-v1-le` words without touching lower 14 bits."""
    w = np.asarray(words, dtype="<u2")
    r = np.asarray(quarter_turns, dtype=np.int64) & 0x3
    rotated = (w.astype(np.int64) + (r << 14)) & 0xFFFF
    return rotated.astype("<u2")


def canonicalize_c16_to_features(words: np.ndarray | int) -> np.ndarray:
    """Extract canonicalized rotation-invariant polar/quality features + turn-standardized unit vector.

    Input shapes `(...,)` -> returns `(..., 6)` float32:
      `[erased_flag, is_valid, cos_local_angle, sin_local_angle, norm_log_radial, norm_quality]`
    """
    dec = decode_c16(words)
    is_valid = (~dec["erased"]).astype(np.float32)
    erased_f = dec["erased"].astype(np.float32)
    # Local angle inside canonical sector [-pi/4, pi/4) using bin midpoint
    local_theta = -0.25 * np.pi + (dec["angle_bin"].astype(np.float64) + 0.5) * (0.5 * np.pi / 64.0)
    cos_a = (np.cos(local_theta) * is_valid).astype(np.float32)
    sin_a = (np.sin(local_theta) * is_valid).astype(np.float32)
    # Valid radial_code is in 1..15; map to normalized [0, 1]
    norm_r = (np.clip(dec["radial_code"], 0, 15).astype(np.float32) / 15.0) * is_valid
    norm_q = (dec["quality_bin"].astype(np.float32) / 3.0) * is_valid
    return np.stack([erased_f, is_valid, cos_a, sin_a, norm_r, norm_q], axis=-1).astype(np.float32)
