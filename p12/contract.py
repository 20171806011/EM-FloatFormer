"""Locked shared contract, disjoint seed namespaces, and exact mathematical geometry for Phase 12 (`p12/`).

Inherits physical constants from P9/P11 without modifying any prior phase directory.
All seed ranges are strictly disjoint from P1..P10 (`HISTORICAL_SEED_RANGES`) and P11 (`P11_SEED_RANGES`).
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from p9.contract import (
    ALL_REGIMES,
    CORE_ROWS,
    CP,
    DELAYS,
    FRAME_DURATION,
    LINK_DURATION,
    N,
    OFDM_PILOT_ROWS,
    POWERS,
    PRIMARY_SNR_DB,
    PROBE_ROW,
    REGIMES_ID,
    REGIMES_OOD,
    REGIMES_SANITY,
    SC_LAYOUTS,
    T,
    n0_from_snr,
)
from p11.contract import (
    ANCHOR,
    COMPLEX_DTYPE,
    HISTORICAL_SEED_RANGES,
    OFFGRID_ALL_REGIMES,
    OFFGRID_ANCHOR_ROWS,
    OFFGRID_DELAY_MAX,
    OFFGRID_DELAY_MIN,
    OFFGRID_RAY_COUNTS,
    OFFGRID_REGIMES_ID,
    OFFGRID_REGIMES_OOD,
    P11_SEED_RANGES,
    REAL_DTYPE,
    enforce_determinism,
    is_historical_seed,
)

# --- Provenance ---
P11_BASELINE_COMMIT = "307d62fd89206442960fa0cebdcdc6a0530db366"

# --- Strictly Disjoint Phase 12 Seed Namespaces ---
# Completely disjoint from P1..P10 historical ranges (<= 1.06e12 + 9.102e12) and P11 (1.11e12..1.16e12).
SEED_P12_DIAG = 1_210_000_000_000
SEED_P12_DEV = 1_220_000_000_000
SEED_P12_TRAIN = 1_230_000_000_000
SEED_P12_FORMAL_ID = 1_240_000_000_000
SEED_P12_FORMAL_OOD = 1_250_000_000_000
SEED_P12_E2E = 1_260_000_000_000
P12_NAMESPACE_SPAN = 10_000_000_000

P12_SEED_RANGES: dict[str, tuple[int, int]] = {
    "p12_diag": (SEED_P12_DIAG, SEED_P12_DIAG + P12_NAMESPACE_SPAN),
    "p12_dev": (SEED_P12_DEV, SEED_P12_DEV + P12_NAMESPACE_SPAN),
    "p12_train": (SEED_P12_TRAIN, SEED_P12_TRAIN + P12_NAMESPACE_SPAN),
    "p12_formal_id": (SEED_P12_FORMAL_ID, SEED_P12_FORMAL_ID + P12_NAMESPACE_SPAN),
    "p12_formal_ood": (SEED_P12_FORMAL_OOD, SEED_P12_FORMAL_OOD + P12_NAMESPACE_SPAN),
    "p12_e2e": (SEED_P12_E2E, SEED_P12_E2E + P12_NAMESPACE_SPAN),
}

# --- Stage 1A Four-Arm Causal Sentinel Geometry ---
# Holds targets, AUX, helper count (2), and absolute helper offsets {1, 2} strictly identical across all 4 arms!
SENTINEL_TARGETS: tuple[int, ...] = (2, 5, 14, 17)
SENTINEL_PRE_TARGETS: tuple[int, ...] = (2, 5)
SENTINEL_POST_TARGETS: tuple[int, ...] = (14, 17)
SENTINEL_AUX: tuple[int, ...] = (1, 4, 7, 13, 16)
SENTINEL_ARMS: dict[str, tuple[int, int]] = {
    "one_sided_left": (8, 9),      # offsets {-2, -1}
    "bilateral_9_12": (9, 12),     # offsets {-1, +2}
    "one_sided_right": (11, 12),   # offsets {+1, +2}
    "bilateral_8_11": (8, 11),     # offsets {-2, +1}
}

# Strict disjointness check at import time
_sentinel_union = set(SENTINEL_TARGETS) | set(SENTINEL_AUX) | {ANCHOR, PROBE_ROW}
assert set(SENTINEL_TARGETS).isdisjoint(SENTINEL_AUX)
assert set(SENTINEL_TARGETS).isdisjoint({ANCHOR, PROBE_ROW})
assert set(SENTINEL_AUX).isdisjoint({ANCHOR, PROBE_ROW})
for _arm_name, _arm_rows in SENTINEL_ARMS.items():
    assert len(_arm_rows) == 2 and set(_arm_rows).isdisjoint(_sentinel_union), f"Sentinel overlap in {_arm_name}"
    assert sorted(abs(r - ANCHOR) for r in _arm_rows) == [1, 2], f"Sentinel offset mismatch in {_arm_name}"

# --- Exact Feasible Symmetric Bilateral Fold Designs (D = {0..18}\{10}) ---
# K=2 Symmetric Bilateral (8 + 8, exact same 10 eligible helper attempts as P11, worst extrap 8 -> 1)
P12_FOLD2_TARGETS: tuple[tuple[int, ...], tuple[int, ...]] = (
    (1, 3, 5, 8, 12, 13, 16, 17),
    tuple(r for r in CORE_ROWS if r not in (1, 3, 5, 8, 12, 13, 16, 17)),
)
P12_FOLD2_AUX: tuple[tuple[int, ...], tuple[int, ...]] = (
    (2, 6, 7, 14, 15),
    (1, 5, 9, 13, 16),
)

# K=3 Symmetric Bilateral (6 + 5 + 5, globally disjoint AUX, 23 eligible helper attempts, worst extrap 1)
P12_FOLD3_TARGETS: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] = (
    tuple(CORE_ROWS[i] for i in range(0, len(CORE_ROWS), 3)),
    tuple(CORE_ROWS[i] for i in range(1, len(CORE_ROWS), 3)),
    tuple(CORE_ROWS[i] for i in range(2, len(CORE_ROWS), 3)),
)
P12_FOLD3_AUX: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] = (
    (2, 5, 8, 13, 16),
    (0, 3, 6, 12, 17),
    (1, 4, 7, 9, 15),
)

# Verify exact partition & disjointness invariants for both K=2 and K=3
for _label, _t_folds, _a_folds in (
    ("K=2", P12_FOLD2_TARGETS, P12_FOLD2_AUX),
    ("K=3", P12_FOLD3_TARGETS, P12_FOLD3_AUX),
):
    _flat_t = [r for tf in _t_folds for r in tf]
    assert len(_flat_t) == len(CORE_ROWS) and set(_flat_t) == set(CORE_ROWS), f"{_label} target partition broken"
    _flat_a = [r for af in _a_folds for r in af]
    assert len(_flat_a) == len(set(_flat_a)), f"{_label} AUX must be globally disjoint"
    for _tf, _af in zip(_t_folds, _a_folds):
        assert len(_af) == 5 and set(_tf).isdisjoint(_af), f"{_label} fold target/AUX overlap"
        assert set(_af).isdisjoint({ANCHOR, PROBE_ROW})

# --- Off-Grid Frame-A 4-Pilot Geometries ---
OFDM_PILOT_COMB: tuple[int, ...] = tuple(int(r) for r in OFDM_PILOT_ROWS)  # (0, 9, 10, 19) locked P11 comb
OFDM_PILOT_PARETO: tuple[int, ...] = (0, 4, 6, 19)                        # Endpoint-preserving Chebyshev optimum

# Stage 1B extended off-grid regimes (explicitly separating P11's inherited-delay reset from true new-delay reset)
OFFGRID_REGIME_TRUE_RESET = "offgrid_full_reset_new_delays"
OFFGRID_STAGE1_REGIMES: tuple[str, ...] = (
    "offgrid_continuous",
    "offgrid_partial_shift",
    "offgrid_midframe_shift",
    "offgrid_full_reset",
    OFFGRID_REGIME_TRUE_RESET,
    "offgrid_ood_burst_and_shift",
    "offgrid_ood_five_ray_fast",
)


def is_p11_or_historical_seed(seed: int) -> Optional[str]:
    """Return matching historical or P11 seed namespace if `seed` collides, else None."""
    hist = is_historical_seed(seed)
    if hist is not None:
        return hist
    s = int(seed)
    for name, (lo, hi) in P11_SEED_RANGES.items():
        if lo <= s < hi:
            return name
    return None


def assert_p12_seed_disjoint(seeds: Iterable[int], *, expected_namespace: Optional[str] = None) -> list[int]:
    """Verify every seed is unique, disjoint from all P1..P11 ranges, and optionally in expected_namespace."""
    out: list[int] = []
    seen: set[int] = set()
    lo_hi = P12_SEED_RANGES[expected_namespace] if expected_namespace is not None else None
    for raw in seeds:
        hit = is_p11_or_historical_seed(raw)
        if hit is not None:
            raise ValueError(f"Seed {raw} collides with prior namespace {hit!r}")
        s = int(raw)
        if s in seen:
            raise ValueError(f"Duplicate P12 seed {s}")
        if lo_hi is not None and not (lo_hi[0] <= s < lo_hi[1]):
            raise ValueError(f"Seed {s} outside expected P12 namespace {expected_namespace!r} {lo_hi}")
        seen.add(s)
        out.append(s)
    return out


__all__ = [
    "ALL_REGIMES",
    "ANCHOR",
    "COMPLEX_DTYPE",
    "CORE_ROWS",
    "CP",
    "DELAYS",
    "FRAME_DURATION",
    "LINK_DURATION",
    "N",
    "OFDM_PILOT_COMB",
    "OFDM_PILOT_PARETO",
    "OFDM_PILOT_ROWS",
    "OFFGRID_ALL_REGIMES",
    "OFFGRID_ANCHOR_ROWS",
    "OFFGRID_DELAY_MAX",
    "OFFGRID_DELAY_MIN",
    "OFFGRID_RAY_COUNTS",
    "OFFGRID_REGIMES_ID",
    "OFFGRID_REGIMES_OOD",
    "OFFGRID_REGIME_TRUE_RESET",
    "OFFGRID_STAGE1_REGIMES",
    "P11_BASELINE_COMMIT",
    "P12_FOLD2_AUX",
    "P12_FOLD2_TARGETS",
    "P12_FOLD3_AUX",
    "P12_FOLD3_TARGETS",
    "P12_NAMESPACE_SPAN",
    "P12_SEED_RANGES",
    "POWERS",
    "PRIMARY_SNR_DB",
    "PROBE_ROW",
    "REAL_DTYPE",
    "REGIMES_ID",
    "REGIMES_OOD",
    "REGIMES_SANITY",
    "SC_LAYOUTS",
    "SEED_P12_DEV",
    "SEED_P12_DIAG",
    "SEED_P12_E2E",
    "SEED_P12_FORMAL_ID",
    "SEED_P12_FORMAL_OOD",
    "SEED_P12_TRAIN",
    "SENTINEL_ARMS",
    "SENTINEL_AUX",
    "SENTINEL_POST_TARGETS",
    "SENTINEL_PRE_TARGETS",
    "SENTINEL_TARGETS",
    "T",
    "assert_p12_seed_disjoint",
    "enforce_determinism",
    "is_p11_or_historical_seed",
    "n0_from_snr",
]
