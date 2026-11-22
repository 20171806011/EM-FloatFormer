"""Locked contract and strictly disjoint seed namespaces for CLI-4 P12 Stage 5 study.

All seed ranges are strictly disjoint from P1..P10 (`HISTORICAL_SEED_RANGES`),
P11 (`P11_SEED_RANGES`), and P12 (`P12_SEED_RANGES`, including the 2026-11-21
consistency diagnostic offset inside `p12_e2e`).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Optional

from p11.contract import (
    COMPLEX_DTYPE,
    N,
    OFFGRID_ANCHOR_ROWS,
    OFFGRID_DELAY_MAX,
    OFFGRID_DELAY_MIN,
    PRIMARY_SNR_DB,
    PROBE_ROW,
    REAL_DTYPE,
    T,
    enforce_determinism,
    n0_from_snr,
)
from p12.contract import (
    OFDM_PILOT_COMB,
    OFFGRID_REGIME_TRUE_RESET,
    P12_SEED_RANGES,
    is_p11_or_historical_seed,
)

# --- Provenance ---
STAGE5_BASELINE_COMMIT = "f566f9b8b0ddfb468a4cb3fd64588bfa46c87212"

# --- Strictly Disjoint CLI-4 Stage 5 Seed Namespaces ---
# P12 occupies [1.21e12, 1.27e12). CLI-4 Stage 5 uses [1.41e12, 1.46e12).
S5_NAMESPACE_SPAN = 10_000_000_000
SEED_S5_DEV = 1_410_000_000_000
SEED_S5_TRAIN = 1_420_000_000_000
SEED_S5_TEST_ID = 1_430_000_000_000
SEED_S5_TEST_OOD = 1_440_000_000_000
SEED_S5_UNIT = 1_450_000_000_000

S5_SEED_RANGES: dict[str, tuple[int, int]] = {
    "s5_dev": (SEED_S5_DEV, SEED_S5_DEV + S5_NAMESPACE_SPAN),
    "s5_train": (SEED_S5_TRAIN, SEED_S5_TRAIN + S5_NAMESPACE_SPAN),
    "s5_test_id": (SEED_S5_TEST_ID, SEED_S5_TEST_ID + S5_NAMESPACE_SPAN),
    "s5_test_ood": (SEED_S5_TEST_OOD, SEED_S5_TEST_OOD + S5_NAMESPACE_SPAN),
    "s5_unit": (SEED_S5_UNIT, SEED_S5_UNIT + S5_NAMESPACE_SPAN),
}

# --- Off-Grid Regimes ---
ID_REGIMES: tuple[str, ...] = (
    "offgrid_continuous",
    "offgrid_partial_shift",
    "offgrid_midframe_shift",
    "offgrid_full_reset",
)
TRUE_RESET_REGIME: str = OFFGRID_REGIME_TRUE_RESET
OOD_ONLY_REGIMES: tuple[str, ...] = (
    "offgrid_ood_burst_and_shift",
    "offgrid_ood_five_ray_fast",
)
OOD_AND_RESET_REGIMES: tuple[str, ...] = (
    TRUE_RESET_REGIME,
    "offgrid_ood_burst_and_shift",
    "offgrid_ood_five_ray_fast",
)
# Dev & Train NEVER touch burst_and_shift or five_ray_fast so OOD remains 100% unseen!
DEV_ABLATION_REGIMES: tuple[str, ...] = ID_REGIMES + (TRUE_RESET_REGIME,)
TRAIN_REGIMES: tuple[str, ...] = ID_REGIMES + (TRUE_RESET_REGIME,)

STAGE5_SEEDS: tuple[int, ...] = (5001, 5002, 5003)


def is_prior_p1_to_p12_seed(seed: int) -> Optional[str]:
    """Return colliding prior namespace name across P1..P12 if any, else None."""
    hit = is_p11_or_historical_seed(seed)
    if hit is not None:
        return hit
    s = int(seed)
    for name, (lo, hi) in P12_SEED_RANGES.items():
        if lo <= s < hi:
            return name
    return None


def assert_s5_seed_disjoint(
    seeds: Iterable[int],
    *,
    expected_namespace: Optional[str] = None,
) -> list[int]:
    """Assert uniqueness, zero overlap with P1..P12, and optional namespace membership."""
    out: list[int] = []
    seen: set[int] = set()
    lo_hi = S5_SEED_RANGES[expected_namespace] if expected_namespace is not None else None
    for raw in seeds:
        hit = is_prior_p1_to_p12_seed(raw)
        if hit is not None:
            raise ValueError(f"Seed {raw} collides with prior namespace {hit!r}")
        s = int(raw)
        if s in seen:
            raise ValueError(f"Duplicate Stage 5 seed {s}")
        if lo_hi is not None and not (lo_hi[0] <= s < lo_hi[1]):
            raise ValueError(f"Seed {s} outside expected namespace {expected_namespace!r} {lo_hi}")
        seen.add(s)
        out.append(s)
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = [
    "COMPLEX_DTYPE",
    "DEV_ABLATION_REGIMES",
    "ID_REGIMES",
    "N",
    "OFDM_PILOT_COMB",
    "OFFGRID_ANCHOR_ROWS",
    "OFFGRID_DELAY_MAX",
    "OFFGRID_DELAY_MIN",
    "OOD_AND_RESET_REGIMES",
    "OOD_ONLY_REGIMES",
    "PRIMARY_SNR_DB",
    "PROBE_ROW",
    "REAL_DTYPE",
    "S5_NAMESPACE_SPAN",
    "S5_SEED_RANGES",
    "SEED_S5_DEV",
    "SEED_S5_TEST_ID",
    "SEED_S5_TEST_OOD",
    "SEED_S5_TRAIN",
    "SEED_S5_UNIT",
    "STAGE5_BASELINE_COMMIT",
    "STAGE5_SEEDS",
    "T",
    "TRAIN_REGIMES",
    "TRUE_RESET_REGIME",
    "assert_s5_seed_disjoint",
    "enforce_determinism",
    "is_prior_p1_to_p12_seed",
    "n0_from_snr",
    "sha256_file",
]
