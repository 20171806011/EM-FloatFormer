"""
Independent E2E Test Harness: Reference Signal Generators, Physical Constants & Resolvers.

Provides:
- MilestoneResolver: Gracefully skips pending milestones without dummy facade mocking.
- Reference Gray-coded 16-QAM and QPSK modulators/demodulators for independent oracle verification.
- Reference 1024-state exact sign-flip permutation tester and Holm-Bonferroni adjuster.
- Cryptographic SHA256 file audit helpers for repository governance.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import unittest

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Project Root & Path Discovery
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    # Current file is tests/e2e/harness.py -> project root is 2 levels up
    return Path(__file__).resolve().parent.parent.parent

PROJECT_ROOT = get_project_root()
MANIFEST_PATH = PROJECT_ROOT / "MANIFEST_SHA256.json"

# ---------------------------------------------------------------------------
# Milestone Dependency Resolver
# ---------------------------------------------------------------------------

class MilestoneResolver:
    """
    Safely resolves Stage P4 implementation modules or raises descriptive
    unittest.SkipTest when a milestone is still under construction by parallel workers.
    """
    @staticmethod
    def get_module(module_name: str, required_milestone: str) -> Any:
        try:
            return importlib.import_module(f"p4.{module_name}")
        except (ImportError, ModuleNotFoundError) as e:
            raise unittest.SkipTest(
                f"[PROGRESSIVE EXECUTION] Module 'p4.{module_name}' not available. "
                f"Requires {required_milestone}. Dependency detail: {e}"
            )

    @staticmethod
    def get_callable(module_name: str, func_name: str, required_milestone: str) -> Any:
        mod = MilestoneResolver.get_module(module_name, required_milestone)
        if not hasattr(mod, func_name):
            raise unittest.SkipTest(
                f"[PROGRESSIVE EXECUTION] Callable '{func_name}' not found in 'p4.{module_name}'. "
                f"Requires {required_milestone} completion."
            )
        return getattr(mod, func_name)

    @staticmethod
    def get_class(module_name: str, class_name: str, required_milestone: str) -> Any:
        return MilestoneResolver.get_callable(module_name, class_name, required_milestone)


# ---------------------------------------------------------------------------
# Reference Constellation & Signal Definitions
# ---------------------------------------------------------------------------

SIGMA_16QAM = math.sqrt(10.0)
THRESH_16QAM = 2.0 / SIGMA_16QAM
GRAY_TABLE_16QAM = torch.tensor([-3.0, -1.0, 3.0, 1.0], dtype=torch.float32)

# 16-QAM constellation points indexed by (b0, b1, b2, b3)
# Where b0=sign(I), b1=mag(I), b2=sign(Q), b3=mag(Q)
QAM16_POINTS = torch.zeros(16, dtype=torch.complex64)
for _b0 in range(2):
    for _b1 in range(2):
        for _b2 in range(2):
            for _b3 in range(2):
                _idx = _b0 * 8 + _b1 * 4 + _b2 * 2 + _b3
                _re = GRAY_TABLE_16QAM[_b0 * 2 + _b1]
                _im = GRAY_TABLE_16QAM[_b2 * 2 + _b3]
                QAM16_POINTS[_idx] = torch.complex(_re, _im) / SIGMA_16QAM


def q16_reference(bits: torch.Tensor) -> torch.Tensor:
    """Independent reference Gray 16-QAM modulator."""
    table = GRAY_TABLE_16QAM.to(device=bits.device)
    idx_i = bits[..., 0] * 2 + bits[..., 1]
    idx_q = bits[..., 2] * 2 + bits[..., 3]
    real = table[idx_i]
    imag = table[idx_q]
    return torch.complex(real, imag) / SIGMA_16QAM


def qpsk_reference(bits: torch.Tensor) -> torch.Tensor:
    """Independent reference Gray QPSK modulator."""
    real = (2.0 * bits[..., 0].float() - 1.0) / math.sqrt(2.0)
    imag = (2.0 * bits[..., 1].float() - 1.0) / math.sqrt(2.0)
    return torch.complex(real, imag)


def hard_bits_16qam_reference(z: torch.Tensor, thresh: float = THRESH_16QAM) -> torch.Tensor:
    """Independent reference hard slicing for Gray-coded 16-QAM symbols."""
    b0 = (z.real > 0.0).long()
    b1 = (z.real.abs() < thresh).long()
    b2 = (z.imag > 0.0).long()
    b3 = (z.imag.abs() < thresh).long()
    return torch.stack((b0, b1, b2, b3), dim=-1)


# ---------------------------------------------------------------------------
# Pre-Frozen Confirmatory Seeds (Stage P4D)
# ---------------------------------------------------------------------------

FROZEN_SEEDS_P4D: List[int] = [
    74001, 74002, 74003, 74004, 74005,
    74006, 74007, 74008, 74009, 74010,
]

# Historical seeds from Stages P1, P2, P3 to ensure strict disjointness
HISTORICAL_SEEDS: List[int] = [
    42, 100, 2024, 777, 999, 12345, 54321, 99999,
]

# ---------------------------------------------------------------------------
# Reference Sign-Flip Permutation Test & Holm-Bonferroni Adjustment
# ---------------------------------------------------------------------------

def exact_permutation_test_1024(deltas: Union[np.ndarray, torch.Tensor, List[float]]) -> float:
    """
    Compute exact two-sided p-value for N=10 paired differences by enumerating
    all 2^10 = 1024 sign assignments.

    p = (1/1024) * sum_{k=0}^{1023} I(|sum_{s=1}^{10} (-1)^{b_k(s)} delta_s| >= |sum delta_s| - 1e-10)
    """
    if isinstance(deltas, torch.Tensor):
        d = deltas.detach().cpu().numpy().flatten()
    else:
        d = np.asarray(deltas, dtype=np.float64).flatten()

    n = len(d)
    if n == 0:
        return 1.0
    if n == 1:
        return 1.0

    t_obs = abs(float(np.sum(d)))

    # For N=10: 2^10 = 1024 combinations
    num_states = 1 << n
    count_extreme = 0
    tol = 1e-10

    # Exhaustive enumeration across all 2^N sign states
    for mask in range(num_states):
        # Generate sign vector: +1 if bit is 0, -1 if bit is 1
        signs = np.array([1.0 if not (mask & (1 << i)) else -1.0 for i in range(n)], dtype=np.float64)
        t_perm = abs(float(np.sum(signs * d)))
        if t_perm >= (t_obs - tol):
            count_extreme += 1

    return float(count_extreme) / float(num_states)


def holm_bonferroni_adjustment(p_values: List[float]) -> List[float]:
    """
    Apply Holm-Bonferroni step-down adjustment to enforce strong FWER control.
    p_{(k)}^Holm = min(1.0, max_{j <= k} (M - j + 1) * p_{(j)})
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort p-values while preserving original indices
    sorted_pairs = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted_sorted: List[float] = [0.0] * m

    running_max = 0.0
    for rank, (orig_idx, p_val) in enumerate(sorted_pairs):
        multiplier = m - rank
        raw_adj = multiplier * p_val
        running_max = max(running_max, raw_adj)
        adjusted_sorted[rank] = min(1.0, running_max)

    # Restore original order
    result = [0.0] * m
    for rank, (orig_idx, _) in enumerate(sorted_pairs):
        result[orig_idx] = adjusted_sorted[rank]

    return result


# ---------------------------------------------------------------------------
# Cryptographic Manifest & File Immutability Helpers
# ---------------------------------------------------------------------------

def compute_file_sha256(path: Path) -> str:
    """Compute standard SHA256 of raw file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_file_sha256_normalized(path: Path) -> str:
    """Compute SHA256 with CRLF -> LF normalization for cross-platform Git checkouts."""
    with open(path, "rb") as f:
        data = f.read()
    # Normalize CRLF to LF
    normalized = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(normalized).hexdigest()


def load_manifest() -> List[Dict[str, Any]]:
    """Load the sealed MANIFEST_SHA256.json file."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest file not found: {MANIFEST_PATH}")
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
