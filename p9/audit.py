"""Independent blind statistical auditor, integrity verifier, and CLI for Phase 9.

Recomputes all per-cluster, per-regime, and pooled metrics directly from raw per-link JSONL
records, executes exact 2^G paired sign-flip permutation tests (G <= 16, up to 65,536 patterns)
with exact Fraction step-down Holm-Bonferroni adjustment and directional alignment, verifies
engineering non-inferiority (df = G - 1), enforces paired target-energy and seed-disjointness
invariants, and checks exact field-level replay agreement.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import tempfile
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import scipy
from scipy.stats import t as student_t

from p9.contract import (
    ALL_REGIMES,
    N,
    PROBE_ROW,
    PublicObservation,
    REGIMES_ID,
    SC_LAYOUTS,
    SEED_P9_DEV,
    SEED_P9_FORMAL_ID,
    SEED_P9_FORMAL_OOD,
    SEED_P9_TRAIN,
    T,
    n0_from_snr,
)
from p9.independent_ref import (
    PiecewiseDopplerShift,
    audit_receiver_anti_leakage,
    blockwise_propagate_piecewise,
    check_no_hardcoded_n0,
    check_seed_disjointness,
    exhaustive_ml_verify_mini_code,
    row_boundary_time,
    serial_propagate_ab,
    verify_phase_continuity,
    verify_public_observation,
)


RESULTS_DIR = Path(__file__).parent / "results"
P9_HOLM_FAMILY = (
    "decoder",
    "classical_gate",
    "attention_vs_classical",
    "attention_vs_local",
)
P8C_BASE_SEED = 822611190000
P8C_BUDGETS = ("sparse", "dense")
P8C_ARMS = ("carry", "reset", "mismatch", "oracle")

ADDITIVE_INT_FIELDS = (
    "blocks",
    "block_errors",
    "coded_bits",
    "coded_errors",
    "info_bits",
    "info_errors",
    "extra_blocks",
    "extra_errors",
    "probes",
)
OPTIONAL_INT_FIELDS = (
    "first8_errors",
    "last8_errors",
    "ml_cert_errors",
)
ADDITIVE_FLOAT_FIELDS = (
    "probe_sse",
    "probe_energy",
    "clean_sse",
    "clean_energy",
    "variance_sum",
    "nll_sum",
)


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _exact_student_t_cdf(x: float, df: int) -> float:
    """Exact closed-form Student-t CDF for integer degrees of freedom df >= 1 (full float64 precision)."""
    denom = math.sqrt(float(df) + x * x)
    s = x / denom
    c2 = float(df) / (float(df) + x * x)
    if df % 2 == 1:
        theta = math.atan(x / math.sqrt(float(df)))
        if df == 1:
            return 0.5 + theta / math.pi
        c = math.sqrt(c2)
        term = 1.0
        total = 1.0
        for k in range(2, df - 1, 2):
            term *= (float(k) / float(k + 1)) * c2
            total += term
        return 0.5 + (theta + s * c * total) / math.pi
    term = 1.0
    total = 1.0
    for k in range(1, df - 1, 2):
        term *= (float(k) / float(k + 1)) * c2
        total += term
    return 0.5 + 0.5 * s * total


def _exact_student_t_pdf(x: float, df: int) -> float:
    """Exact Student-t PDF for integer df >= 1."""
    log_norm = (
        math.lgamma(0.5 * (df + 1))
        - math.lgamma(0.5 * df)
        - 0.5 * math.log(float(df) * math.pi)
    )
    return math.exp(log_norm - 0.5 * (df + 1) * math.log1p((x * x) / float(df)))


def exact_student_t_ppf(q: float, df: int) -> float:
    """SciPy-version-independent full-float64 Student-t quantile via exact integer-df CDF Newton polish."""
    if not (0.0 < q < 1.0):
        raise ValueError(f"Quantile q must lie in (0, 1), got {q}")
    if type(df) is not int or df < 1:
        raise ValueError(f"Degrees of freedom df must be an integer >= 1, got {df!r}")
    x = float(student_t.ppf(q, df))
    for _ in range(10):
        err = _exact_student_t_cdf(x, df) - float(q)
        pdf = _exact_student_t_pdf(x, df)
        step = err / pdf
        x -= step
        if abs(step) <= 1e-15 * max(1.0, abs(x)):
            break
    return float(x)


# ---------------------------------------------------------------------------
# Exact Paired Sign-Flip Permutation Test, Holm-Bonferroni & Non-Inferiority
# ---------------------------------------------------------------------------

def exact_sign_flip_test(
    diffs: Sequence[float] | np.ndarray,
    *,
    expected_sign: int = 1,
    max_exact_g: int = 16,
    chunk_size: int = 4096,
) -> dict:
    """Exact paired two-sided sign-flip permutation test over all 2^G sign patterns.

    Uses symmetry (`s_0 = +1`, doubling extreme half-space count) and vectorized NumPy chunking
    with exact rational `Fraction` arithmetic. Never inflates degrees of freedom beyond `df = G - 1`.
    """
    d = np.asarray(diffs, dtype=np.float64)
    if d.ndim != 1:
        raise ValueError(f"Expected 1D cluster difference vector, got shape {d.shape}")
    g = int(d.size)
    if g < 2 or g > int(max_exact_g):
        raise ValueError(f"Cluster count G must satisfy 2 <= G <= {max_exact_g}, got G={g}")
    if not np.isfinite(d).all():
        raise ValueError("Cluster differences contain non-finite values")
    if expected_sign not in (-1, 1):
        raise ValueError(f"expected_sign must be +1 or -1, got {expected_sign}")

    obs_sum = abs(float(math.fsum(d.tolist())))
    abs_sum = float(math.fsum(abs(float(x)) for x in d.tolist()))
    threshold = obs_sum - 1e-12 * max(1.0, abs_sum)

    # Enumerate half-space (s_0 = +1 fixed, s_1..s_{G-1} in {-1, +1}) in vectorized chunks.
    # Because |-s @ d| == |s @ d| exactly in IEEE-754 float64, 2 * extreme_half == extreme_full.
    n_half = 1 << (g - 1)
    shifts = np.arange(g - 1, dtype=np.uint32)
    d_tail = d[1:]
    d0 = float(d[0])
    extreme_half = 0

    for start in range(0, n_half, int(chunk_size)):
        end = min(start + int(chunk_size), n_half)
        idx = np.arange(start, end, dtype=np.uint32)
        bits = (idx[:, None] >> shifts[None, :]) & np.uint32(1)
        signs_tail = bits.astype(np.float64) * 2.0 - 1.0
        sums = d0 + signs_tail @ d_tail
        extreme_half += int(np.count_nonzero(np.abs(sums) >= threshold))

    total_patterns = 1 << g
    extreme_full = 2 * extreme_half
    raw_exact = Fraction(extreme_full, total_patterns)

    mean_diff = float(math.fsum(d.tolist()) / float(g))
    std_diff = float(np.std(d, ddof=1))
    df = g - 1
    t_crit = exact_student_t_ppf(0.975, df)
    half_width = t_crit * std_diff / math.sqrt(float(g))
    ci95 = [float(mean_diff - half_width), float(mean_diff + half_width)]

    if expected_sign > 0:
        wins = int(np.count_nonzero(d > 1e-12))
        ties = int(np.count_nonzero(np.abs(d) <= 1e-12))
        losses = int(np.count_nonzero(d < -1e-12))
        direction_aligned = bool(mean_diff > 1e-12)
    else:
        wins = int(np.count_nonzero(d < -1e-12))
        ties = int(np.count_nonzero(np.abs(d) <= 1e-12))
        losses = int(np.count_nonzero(d > 1e-12))
        direction_aligned = bool(mean_diff < -1e-12)

    return {
        "n": g,
        "df": df,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "ci95": ci95,
        "raw_exact": str(raw_exact),
        "raw_p": float(raw_exact),
        "exact_patterns": int(total_patterns),
        "extreme_patterns": int(extreme_full),
        "expected_sign": int(expected_sign),
        "direction_aligned": direction_aligned,
        "differences": [float(x) for x in d.tolist()],
    }


def holm_bonferroni_adjust(
    contrasts: Sequence[Mapping[str, Any]],
    *,
    alpha: Fraction = Fraction(1, 20),
) -> list[dict]:
    """Apply exact rational step-down Holm-Bonferroni adjustment and directional verdicts."""
    m = len(contrasts)
    if m < 1:
        raise ValueError("Holm family must contain at least one contrast")
    raw_fracs = [Fraction(c["raw_exact"]) for c in contrasts]
    order = sorted(range(m), key=raw_fracs.__getitem__)

    adjusted: dict[int, Fraction] = {}
    running_max = Fraction(0)
    for rank, idx in enumerate(order):
        step_p = min(Fraction(1), (m - rank) * raw_fracs[idx])
        running_max = max(running_max, step_p)
        adjusted[idx] = running_max

    results: list[dict] = []
    for idx, item in enumerate(contrasts):
        out = dict(item)
        holm_f = adjusted[idx]
        sig_two_sided = bool(holm_f < alpha)
        dir_aligned = bool(out.get("direction_aligned", out.get("mean_diff", 0.0) > 1e-12))
        supported = bool(sig_two_sided and dir_aligned)
        if supported:
            verdict = "supported"
        elif sig_two_sided:
            verdict = "refuted"
        else:
            verdict = "insufficient"
        out["holm_exact"] = str(holm_f)
        out["holm_p"] = float(holm_f)
        out["alpha"] = float(alpha)
        out["significant_two_sided"] = sig_two_sided
        out["supported"] = supported
        out["verdict"] = verdict
        results.append(out)
    return results


def noninferiority_test(
    diffs: Sequence[float] | np.ndarray,
    *,
    margin: float = 0.01,
) -> dict:
    """One-sided 95% paired t upper bound non-inferiority test with df = G - 1.

    `diffs` = candidate_bler - reference_bler per independent cluster. Passes iff upper_95 < margin.
    """
    d = np.asarray(diffs, dtype=np.float64)
    if d.ndim != 1 or d.size < 2:
        raise ValueError(f"Expected 1D cluster difference array of length >= 2, got shape {d.shape}")
    if not np.isfinite(d).all():
        raise ValueError("Non-inferiority differences contain non-finite values")
    g = int(d.size)
    df = g - 1
    mean_diff = float(math.fsum(d.tolist()) / float(g))
    std_diff = float(np.std(d, ddof=1))
    upper_95 = float(mean_diff + exact_student_t_ppf(0.95, df) * std_diff / math.sqrt(float(g)))
    margin_f = float(margin)
    return {
        "n": g,
        "df": df,
        "margin": margin_f,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "upper_95": upper_95,
        "differences": [float(x) for x in d.tolist()],
        "passes": bool(upper_95 < margin_f),
    }


# ---------------------------------------------------------------------------
# Per-Link Integrity Verification & Aggregate Metric Recomputation
# ---------------------------------------------------------------------------

def check_cell(
    cell: Mapping[str, Any],
    budget: str,
    *,
    expected_blocks: int = 16,
    expected_probes: int = 32,
) -> None:
    """Strictly verify per-link exposure counts, integer types, physical bounds, and finite metrics."""
    expected_extra = 2 if budget == "sparse" else 0
    expected_fixed = {
        "blocks": expected_blocks,
        "info_bits": 64 * expected_blocks,
        "coded_bits": 128 * expected_blocks,
        "probes": expected_probes,
        "extra_blocks": expected_extra,
    }
    for key, expected_val in expected_fixed.items():
        require(key in cell, f"Missing exposure field {key!r}")
        require(type(cell[key]) is int and cell[key] == expected_val,
                f"Incorrect exposure {key}={cell.get(key)!r} (expected {expected_val})")

    int_bounds = {
        "block_errors": expected_blocks,
        "coded_errors": 128 * expected_blocks,
        "info_errors": 64 * expected_blocks,
        "extra_errors": expected_extra,
    }
    for key, bound in int_bounds.items():
        require(key in cell, f"Missing count field {key!r}")
        val = cell[key]
        require(type(val) is int and 0 <= val <= bound, f"Invalid {key}={val!r} (bound 0..{bound})")

    failed = cell["block_errors"]
    require(failed <= cell["info_errors"] <= 64 * failed,
            f"Block/info bit error inconsistency: block_errors={failed}, info_errors={cell['info_errors']}")

    if "first8_errors" in cell or "last8_errors" in cell:
        require("first8_errors" in cell and "last8_errors" in cell, "Incomplete half-frame error counts")
        half_bound = expected_blocks // 2
        for key in ("first8_errors", "last8_errors"):
            val = cell[key]
            require(type(val) is int and 0 <= val <= half_bound, f"Invalid {key}={val!r}")
        require(cell["first8_errors"] + cell["last8_errors"] == failed, "Half-frame error totals disagree")

    if "ml_cert_errors" in cell:
        cert_err = cell["ml_cert_errors"]
        require(type(cert_err) is int and 0 <= cert_err <= expected_blocks,
                f"Invalid ml_cert_errors={cert_err!r}")

    for key in ("probe_sse", "clean_sse", "probe_energy", "clean_energy", "variance_sum"):
        require(key in cell, f"Missing metric field {key!r}")
        val = cell[key]
        require(isinstance(val, (int, float)) and not isinstance(val, bool)
                and math.isfinite(val) and float(val) >= 0.0,
                f"Invalid {key}={val!r}")
    require(float(cell["probe_energy"]) > 0.0 and float(cell["clean_energy"]) > 0.0
            and float(cell["variance_sum"]) > 0.0,
            "Empty prediction exposure denominator")

    require("nll_sum" in cell, "Missing nll_sum")
    nll_val = cell["nll_sum"]
    require(isinstance(nll_val, (int, float)) and not isinstance(nll_val, bool) and math.isfinite(nll_val),
            f"Non-finite nll_sum={nll_val!r}")


def verify_paired_energies(record: Mapping[str, Any], *, energy_tol: float = 0.0) -> tuple[float, float]:
    """Verify every paired (budget, arm) cell on a single link shares identical held-out target energies."""
    require("budgets" in record and isinstance(record["budgets"], Mapping), "Record missing budgets map")
    ref_pair: Optional[tuple[float, float]] = None
    for budget, arms in record["budgets"].items():
        require(isinstance(arms, Mapping) and len(arms) > 0, f"Empty arms under budget {budget!r}")
        for arm, cell in arms.items():
            pair = (float(cell["probe_energy"]), float(cell["clean_energy"]))
            if ref_pair is None:
                ref_pair = pair
            else:
                if energy_tol == 0.0:
                    require(pair == ref_pair,
                            f"Held-out target energies differ across paired arms on seed={record.get('seed')}: "
                            f"{budget}/{arm} has {pair} vs {ref_pair}")
                else:
                    require(
                        abs(pair[0] - ref_pair[0]) <= energy_tol and abs(pair[1] - ref_pair[1]) <= energy_tol,
                        f"Held-out target energies differ beyond {energy_tol} on seed={record.get('seed')}: "
                        f"{budget}/{arm} has {pair} vs {ref_pair}",
                    )
    require(ref_pair is not None, "No cells found in record")
    return ref_pair


def aggregate_cells(cells: Sequence[Mapping[str, Any]]) -> dict:
    """Independently recompute pooled/cluster sums and all P8-C + P9 derived metrics from raw cells."""
    require(len(cells) > 0, "Cannot aggregate empty cell list")
    n_links = len(cells)
    sums: dict[str, Any] = {}

    for key in ADDITIVE_INT_FIELDS:
        sums[key] = int(sum(int(c[key]) for c in cells))
    for key in OPTIONAL_INT_FIELDS:
        if all(key in c for c in cells):
            sums[key] = int(sum(int(c[key]) for c in cells))
    for key in ADDITIVE_FLOAT_FIELDS:
        sums[key] = float(math.fsum(float(c[key]) for c in cells))

    core_delivered = int(sums["blocks"] - sums["block_errors"])
    extra_delivered = int(sums["extra_blocks"] - sums["extra_errors"])
    total_delivered_blocks = int(core_delivered + extra_delivered)
    bits_per_block = float(sums["info_bits"]) / float(sums["blocks"])
    net_delivered_info_bits_per_frame = (float(total_delivered_blocks) * bits_per_block) / float(n_links)

    bler = float(sums["block_errors"]) / float(sums["blocks"])
    coded_ber = float(sums["coded_errors"]) / float(sums["coded_bits"])
    info_ber = float(sums["info_errors"]) / float(sums["info_bits"])
    probe_nmse = float(sums["probe_sse"]) / float(sums["probe_energy"])
    clean_nmse = float(sums["clean_sse"]) / float(sums["clean_energy"])
    nll = float(sums["nll_sum"]) / float(sums["probes"])
    calibration = float(sums["probe_sse"]) / float(sums["variance_sum"])

    out = dict(sums)
    out.update({
        "links": n_links,
        "bler": bler,
        "coded_ber": coded_ber,
        "pre_ber": coded_ber,
        "info_ber": info_ber,
        "post_ber": info_ber,
        "probe_nmse": probe_nmse,
        "clean_nmse": clean_nmse,
        "nll": nll,
        "calibration": calibration,
        "core_delivered": core_delivered,
        "extra_delivered": extra_delivered,
        "total_delivered_blocks": total_delivered_blocks,
        "net_delivered_info_bits_per_frame": net_delivered_info_bits_per_frame,
    })
    if "ml_cert_errors" in sums:
        out["ml_cert_rate"] = float(sums["ml_cert_errors"]) / float(sums["blocks"])
    return out


# ---------------------------------------------------------------------------
# Field-Level Replay Verifier
# ---------------------------------------------------------------------------

def _compare_json_fields(expected: Any, observed: Any, path: str, float_tol: float) -> float:
    """Recursively verify exact key/int/str/bool equality and bounded float agreement."""
    if type(expected) is dict or type(observed) is dict:
        require(type(expected) is dict and type(observed) is dict, f"Type mismatch at {path}")
        require(set(expected.keys()) == set(observed.keys()),
                f"Key mismatch at {path}: {set(expected) ^ set(observed)}")
        max_err = 0.0
        for key in expected:
            err = _compare_json_fields(expected[key], observed[key], f"{path}.{key}", float_tol)
            if err > max_err:
                max_err = err
        return max_err
    if type(expected) is list or type(observed) is list:
        require(type(expected) is list and type(observed) is list and len(expected) == len(observed),
                f"List mismatch at {path}")
        max_err = 0.0
        for idx, (a_val, b_val) in enumerate(zip(expected, observed)):
            err = _compare_json_fields(a_val, b_val, f"{path}[{idx}]", float_tol)
            if err > max_err:
                max_err = err
        return max_err
    if type(expected) is bool or type(observed) is bool:
        require(type(expected) is bool and type(observed) is bool and expected == observed,
                f"Bool mismatch at {path}: {expected!r} vs {observed!r}")
        return 0.0
    if type(expected) is int and type(observed) is int:
        require(expected == observed, f"Int mismatch at {path}: {expected} vs {observed}")
        return 0.0
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        diff = abs(float(expected) - float(observed))
        require(math.isfinite(diff) and diff <= float_tol,
                f"Float mismatch at {path}: {expected!r} vs {observed!r} (diff={diff:.3e} > tol={float_tol:.3e})")
        return diff
    require(expected == observed, f"Value mismatch at {path}: {expected!r} vs {observed!r}")
    return 0.0


def verify_replay(
    raw_lines: Sequence[bytes],
    raw_records: Sequence[Mapping[str, Any]],
    replay_path: Path,
    *,
    float_tol: float = 0.0,
) -> dict:
    """Verify a fresh single-worker replay JSONL matches raw records field-by-field."""
    replay_bytes = replay_path.read_bytes()
    replay_lines = replay_bytes.splitlines()
    require(len(replay_lines) > 0, "Replay JSONL is empty")
    replay_records = [json.loads(line) for line in replay_lines]

    raw_by_key = {
        (r["cluster"], r["link"], r["seed"]): (line, r)
        for line, r in zip(raw_lines, raw_records)
    }

    seen_replay_keys: set[tuple[int, int, int]] = set()
    replay_clusters: set[int] = set()
    selected_raw_lines: list[bytes] = []
    max_float_diff = 0.0

    for idx, (rep_line, rep_rec) in enumerate(zip(replay_lines, replay_records)):
        key = (rep_rec["cluster"], rep_rec["link"], rep_rec["seed"])
        require(key not in seen_replay_keys, f"Duplicate replay key {key}")
        require(key in raw_by_key, f"Replay record {key} not found in raw cohort")
        seen_replay_keys.add(key)
        replay_clusters.add(int(rep_rec["cluster"]))
        raw_line, raw_rec = raw_by_key[key]
        selected_raw_lines.append(raw_line)
        err = _compare_json_fields(raw_rec, rep_rec, f"replay[{idx}]", float_tol)
        if err > max_float_diff:
            max_float_diff = err

    exact_bytes_match = (selected_raw_lines == replay_lines)
    if float_tol == 0.0:
        require(max_float_diff == 0.0, f"Replay float diff non-zero: {max_float_diff}")

    return {
        "replay_clusters": sorted(replay_clusters),
        "replay_links": len(replay_records),
        "replay_all_fields_match": True,
        "replay_exact_bytes_match": bool(exact_bytes_match),
        "replay_max_float_diff": float(max_float_diff),
        "selected_records_lf_sha256": hashlib.sha256(b"\n".join(selected_raw_lines) + b"\n").hexdigest(),
        "replay_sha256": hashlib.sha256(replay_bytes).hexdigest(),
    }


# ---------------------------------------------------------------------------
# Cohort & Summary Auditors (P8-C Compatible Mode + Full P9 Mode)
# ---------------------------------------------------------------------------

def _verify_table_matches_recomputed(
    recomputed: Mapping[str, Any],
    reported: Mapping[str, Any],
    label: str,
    *,
    tol: float = 2e-12,
) -> float:
    """Compare every key present in `reported` against `recomputed`; return max numerical diff."""
    max_diff = 0.0
    for key, rep_val in reported.items():
        require(key in recomputed, f"Reported key {label}/{key} not produced by independent recomputation")
        rec_val = recomputed[key]
        if type(rec_val) is int and type(rep_val) is int:
            require(rec_val == rep_val, f"Integer aggregate mismatch {label}/{key}: {rep_val} vs {rec_val}")
        else:
            diff = abs(float(rep_val) - float(rec_val))
            require(math.isclose(float(rep_val), float(rec_val), rel_tol=tol, abs_tol=tol),
                    f"Aggregate mismatch {label}/{key}: reported={rep_val!r} vs recomputed={rec_val!r}")
            if diff > max_diff:
                max_diff = diff
    return max_diff


def _aggregate_nested_budget_arm(
    records: Sequence[Mapping[str, Any]],
    budgets: Sequence[str],
    arms_by_budget: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for budget in budgets:
        out[budget] = {}
        for arm in arms_by_budget[budget]:
            cells = [r["budgets"][budget][arm] for r in records]
            out[budget][arm] = aggregate_cells(cells)
    return out


def audit_p8c(
    raw_path: Path,
    summary_path: Path,
    replay_path: Optional[Path] = None,
    *,
    replay_float_tol: float = 0.0,
) -> dict:
    """Independently audit P8-C `links.jsonl` + `summary.json` using the P9 statistical engine."""
    raw_bytes = raw_path.read_bytes()
    raw_lines = raw_bytes.splitlines()
    records = [json.loads(line) for line in raw_lines]

    keys = [(r["cluster"], r["link"], r["seed"]) for r in records]
    expected_keys = [(i, j, P8C_BASE_SEED + 1000 * i + j) for i in range(1, 11) for j in range(32)]
    require(keys == expected_keys, "P8-C formal cohort missing, duplicated or out of order")

    for rec in records:
        require(set(rec["budgets"]) == set(P8C_BUDGETS), "Missing P8-C raw budget")
        for budget in P8C_BUDGETS:
            arms = rec["budgets"][budget]
            require(set(arms) == set(P8C_ARMS), f"Incomplete P8-C arms in budget {budget}")
            for cell in arms.values():
                check_cell(cell, budget)
        verify_paired_energies(rec, energy_tol=0.0)

    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    arms_map = {b: P8C_ARMS for b in P8C_BUDGETS}

    max_agg_diff = 0.0
    totals_recomp = _aggregate_nested_budget_arm(records, P8C_BUDGETS, arms_map)
    for budget, arm in itertools.product(P8C_BUDGETS, P8C_ARMS):
        err = _verify_table_matches_recomputed(
            totals_recomp[budget][arm],
            summary["totals"][budget][arm],
            f"totals/{budget}/{arm}",
        )
        max_agg_diff = max(max_agg_diff, err)

    require(len(summary["clusters"]) == 10, f"Expected 10 clusters, got {len(summary['clusters'])}")
    cluster_recomps: list[dict] = []
    for c_idx in range(10):
        c_recs = records[c_idx * 32 : (c_idx + 1) * 32]
        c_agg = _aggregate_nested_budget_arm(c_recs, P8C_BUDGETS, arms_map)
        cluster_recomps.append(c_agg)
        for budget, arm in itertools.product(P8C_BUDGETS, P8C_ARMS):
            err = _verify_table_matches_recomputed(
                c_agg[budget][arm],
                summary["clusters"][c_idx][budget][arm],
                f"cluster[{c_idx + 1}]/{budget}/{arm}",
            )
            max_agg_diff = max(max_agg_diff, err)

    # Recompute exact 2^10 paired sign-flip + Holm family independently
    contrasts_spec = list(itertools.product(("reset", "mismatch"), ("bler", "probe_nmse")))
    reported_tests = {(t["control"], t["metric"]): t for t in summary["tests"]}
    require(len(summary["tests"]) == 4 and set(reported_tests) == set(contrasts_spec),
            "Incomplete P8-C 4-contrast Holm family")

    raw_tests = []
    for control, metric in contrasts_spec:
        ctrl_rates = np.array([g["sparse"][control][metric] for g in cluster_recomps], dtype=np.float64)
        carry_rates = np.array([g["sparse"]["carry"][metric] for g in cluster_recomps], dtype=np.float64)
        res = exact_sign_flip_test(ctrl_rates - carry_rates, expected_sign=1)
        res["control"] = control
        res["metric"] = metric
        raw_tests.append(res)

    holm_tests = holm_bonferroni_adjust(raw_tests, alpha=Fraction(1, 20))
    max_stat_diff = 0.0
    max_ci95_diff = 0.0
    for rec_t in holm_tests:
        rep_t = reported_tests[(rec_t["control"], rec_t["metric"])]
        require(rep_t["n"] == 10, "Cluster count mismatch")
        require((rep_t["wins"], rep_t["ties"], rep_t["losses"]) == (rec_t["wins"], rec_t["ties"], rec_t["losses"]),
                "Cluster sign counts disagree")
        require(Fraction(rep_t["raw_exact"]) == Fraction(rec_t["raw_exact"])
                and rep_t["raw_p"] == rec_t["raw_p"], "Exact raw p-value mismatch")
        require(Fraction(rep_t["holm_exact"]) == Fraction(rec_t["holm_exact"])
                and rep_t["holm_p"] == rec_t["holm_p"], "Exact Holm p-value mismatch")
        diff_err = float(np.max(np.abs(np.asarray(rep_t["differences"]) - np.asarray(rec_t["differences"]))))
        mean_err = abs(float(rep_t["mean_diff"]) - rec_t["mean_diff"])
        ci_err = float(np.max(np.abs(np.asarray(rep_t["ci95"]) - np.asarray(rec_t["ci95"]))))
        require(diff_err <= 2e-12 and mean_err <= 2e-12 and ci_err <= 1e-10,
                f"Stat mismatch on {rec_t['control']}/{rec_t['metric']}: diff={diff_err}, mean={mean_err}, ci={ci_err}")
        max_stat_diff = max(max_stat_diff, diff_err, mean_err)
        max_ci95_diff = max(max_ci95_diff, ci_err)

    # Non-inferiority test (df = 9)
    ni_diffs = np.array(
        [g["sparse"]["carry"]["bler"] - g["dense"]["reset"]["bler"] for g in cluster_recomps],
        dtype=np.float64,
    )
    ni_recomp = noninferiority_test(ni_diffs, margin=0.01)
    rep_ni = summary["noninferiority"]
    ni_diff_err = float(np.max(np.abs(np.asarray(rep_ni["differences"]) - np.asarray(ni_recomp["differences"]))))
    ni_mean_err = abs(float(rep_ni["mean_diff"]) - ni_recomp["mean_diff"])
    ni_upper_err = abs(float(rep_ni["upper_95"]) - ni_recomp["upper_95"])
    require(ni_diff_err <= 2e-12 and ni_mean_err <= 2e-12 and ni_upper_err <= 2e-12,
            f"NI numerical mismatch: diff={ni_diff_err}, mean={ni_mean_err}, upper={ni_upper_err}")
    require(rep_ni["margin"] == 0.01 and rep_ni["passes"] == ni_recomp["passes"], "Incorrect NI gate")
    max_stat_diff = max(max_stat_diff, ni_diff_err, ni_mean_err, ni_upper_err)

    result: dict[str, Any] = {
        "mode": "p8c",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "clusters": 10,
        "df": 9,
        "links": len(records),
        "cells": len(records) * len(P8C_BUDGETS) * len(P8C_ARMS),
        "contrasts": len(holm_tests),
        "exact_sign_patterns": 1 << 10,
        "max_aggregate_diff": float(max_agg_diff),
        "max_stat_diff": float(max_stat_diff),
        "max_ci95_diff": float(max_ci95_diff),
        "holm_tests": holm_tests,
        "noninferiority": ni_recomp,
        "delivery_summary": {
            "sparse_carry_extra_delivered": totals_recomp["sparse"]["carry"]["extra_delivered"],
            "sparse_carry_net_bits_per_frame": totals_recomp["sparse"]["carry"]["net_delivered_info_bits_per_frame"],
            "dense_reset_net_bits_per_frame": totals_recomp["dense"]["reset"]["net_delivered_info_bits_per_frame"],
            "net_extra_blocks_gain": (
                totals_recomp["sparse"]["carry"]["total_delivered_blocks"]
                - totals_recomp["dense"]["reset"]["total_delivered_blocks"]
            ),
        },
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
    }

    if replay_path is not None:
        result.update(verify_replay(raw_lines, records, replay_path, float_tol=replay_float_tol))
    return result


def _extract_arm_spec(spec: Mapping[str, Any], prefix: str) -> tuple[str, str]:
    """Extract (budget, arm) pair from flexible P9 contrast metadata."""
    val = spec.get(prefix)
    if isinstance(val, Mapping):
        return str(val["budget"]), str(val["arm"])
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return str(val[0]), str(val[1])
    budget_key = f"{prefix}_budget"
    arm_key = f"{prefix}_arm"
    if budget_key in spec and arm_key in spec:
        return str(spec[budget_key]), str(spec[arm_key])
    raise ValueError(f"Cannot resolve (budget, arm) for {prefix!r} in contrast spec {spec!r}")


def audit_p9(
    raw_path: Path,
    summary_path: Path,
    replay_path: Optional[Path] = None,
    *,
    seed_pools: Optional[Mapping[str, Sequence[int]]] = None,
    replay_float_tol: float = 0.0,
    require_canonical_family: bool = True,
) -> dict:
    """Full blind statistical & integrity audit for Phase 9 formal JSONL + summary + replay."""
    raw_bytes = raw_path.read_bytes()
    raw_lines = raw_bytes.splitlines()
    require(len(raw_lines) > 0, "Raw JSONL is empty")
    records = [json.loads(line) for line in raw_lines]

    # 1. Verify cohort structure, integer keys, cluster balance, and intra-cohort seed uniqueness
    clusters_seen: dict[int, list[dict]] = {}
    cohort_seeds: list[int] = []
    seen_seeds: set[int] = set()
    budgets_ref: Optional[tuple[str, ...]] = None
    arms_by_budget_ref: Optional[dict[str, tuple[str, ...]]] = None

    for idx, rec in enumerate(records):
        for field in ("cluster", "link", "seed"):
            require(field in rec and type(rec[field]) is int, f"Record {idx} invalid integer {field!r}")
        c_id = int(rec["cluster"])
        seed = int(rec["seed"])
        require(seed not in seen_seeds, f"Duplicate seed {seed} in raw cohort")
        seen_seeds.add(seed)
        cohort_seeds.append(seed)
        clusters_seen.setdefault(c_id, []).append(rec)

        if "regime" in rec:
            require(rec["regime"] in ALL_REGIMES, f"Unknown regime {rec['regime']!r} on record {idx}")

        require("budgets" in rec and isinstance(rec["budgets"], dict) and len(rec["budgets"]) > 0,
                f"Record {idx} missing budgets")
        cur_budgets = tuple(sorted(rec["budgets"].keys()))
        cur_arms = {b: tuple(sorted(rec["budgets"][b].keys())) for b in cur_budgets}
        if budgets_ref is None:
            budgets_ref = cur_budgets
            arms_by_budget_ref = cur_arms
        else:
            require(cur_budgets == budgets_ref, f"Budget set mismatch on record {idx}: {cur_budgets} vs {budgets_ref}")
            require(cur_arms == arms_by_budget_ref, f"Arm set mismatch on record {idx}")

        for budget, arms in rec["budgets"].items():
            for cell in arms.values():
                check_cell(cell, budget)
        verify_paired_energies(rec, energy_tol=0.0)

    require(budgets_ref is not None and arms_by_budget_ref is not None, "Empty cohort")
    cluster_ids = sorted(clusters_seen.keys())
    g = len(cluster_ids)
    require(2 <= g <= 16, f"Cluster count G must be in 2..16 for exact sign-flip audit, got {g}")
    require(cluster_ids == list(range(1, g + 1)), f"Cluster IDs must be contiguous 1..{g}, got {cluster_ids}")

    links_per_cluster = len(clusters_seen[1])
    require(links_per_cluster >= 1, "Empty cluster")
    for c_id in cluster_ids:
        c_recs = clusters_seen[c_id]
        require(len(c_recs) == links_per_cluster,
                f"Unbalanced cluster {c_id}: {len(c_recs)} vs {links_per_cluster}")
        link_ids = [r["link"] for r in c_recs]
        require(link_ids == list(range(links_per_cluster)),
                f"Cluster {c_id} link IDs not contiguous 0..{links_per_cluster - 1}")

    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)

    # 2. Seed disjointness audit (if provided via CLI manifest or embedded in summary["seed_pools"])
    effective_pools = seed_pools if seed_pools is not None else summary.get("seed_pools")
    seed_audit: Optional[dict] = None
    if effective_pools is not None:
        seed_audit = check_seed_disjointness(
            effective_pools["train"],
            effective_pools["dev"],
            effective_pools["formal_id"],
            effective_pools["formal_ood"],
            strict=True,
        )
        train_dev = set(int(s) for s in effective_pools["train"]) | set(int(s) for s in effective_pools["dev"])
        leak_overlap = sorted(seen_seeds & train_dev)
        require(len(leak_overlap) == 0, f"Formal cohort intersects train/dev seeds: {leak_overlap[:10]}")

    # 3. Independently recompute pooled, per-cluster, and per-regime aggregates
    max_agg_diff = 0.0
    totals_recomp = _aggregate_nested_budget_arm(records, budgets_ref, arms_by_budget_ref)
    for budget in budgets_ref:
        require(budget in summary["totals"], f"Summary totals missing budget {budget!r}")
        for arm in arms_by_budget_ref[budget]:
            require(arm in summary["totals"][budget], f"Summary totals missing {budget}/{arm}")
            err = _verify_table_matches_recomputed(
                totals_recomp[budget][arm],
                summary["totals"][budget][arm],
                f"totals/{budget}/{arm}",
            )
            max_agg_diff = max(max_agg_diff, err)

    require(len(summary["clusters"]) == g, f"Summary clusters length {len(summary['clusters'])} != G={g}")
    cluster_recomps: list[dict] = []
    for c_idx, c_id in enumerate(cluster_ids):
        c_agg = _aggregate_nested_budget_arm(clusters_seen[c_id], budgets_ref, arms_by_budget_ref)
        cluster_recomps.append(c_agg)
        for budget in budgets_ref:
            for arm in arms_by_budget_ref[budget]:
                err = _verify_table_matches_recomputed(
                    c_agg[budget][arm],
                    summary["clusters"][c_idx][budget][arm],
                    f"cluster[{c_id}]/{budget}/{arm}",
                )
                max_agg_diff = max(max_agg_diff, err)

    regime_recomps: dict[str, dict] = {}
    if "by_regime" in summary:
        regimes_in_raw = sorted({r["regime"] for r in records if "regime" in r})
        require(set(summary["by_regime"].keys()) == set(regimes_in_raw),
                f"Summary by_regime keys {set(summary['by_regime'])} != raw regimes {set(regimes_in_raw)}")
        for reg in regimes_in_raw:
            reg_recs = [r for r in records if r.get("regime") == reg]
            reg_agg = _aggregate_nested_budget_arm(reg_recs, budgets_ref, arms_by_budget_ref)
            regime_recomps[reg] = reg_agg
            for budget in budgets_ref:
                for arm in arms_by_budget_ref[budget]:
                    err = _verify_table_matches_recomputed(
                        reg_agg[budget][arm],
                        summary["by_regime"][reg][budget][arm],
                        f"by_regime/{reg}/{budget}/{arm}",
                    )
                    max_agg_diff = max(max_agg_diff, err)

    # Helper to compute cluster-level metric array with optional regime filter directly from raw records
    def cluster_metric_vector(budget: str, arm: str, metric: str, regimes: Optional[Sequence[str]] = None) -> np.ndarray:
        vals = []
        reg_set = set(regimes) if regimes is not None else None
        for c_id in cluster_ids:
            c_recs = clusters_seen[c_id]
            if reg_set is not None:
                c_recs = [r for r in c_recs if r.get("regime") in reg_set]
            require(len(c_recs) > 0, f"Cluster {c_id} has zero records after regime_filter={regimes}")
            agg = aggregate_cells([r["budgets"][budget][arm] for r in c_recs])
            require(metric in agg, f"Unknown metric {metric!r}")
            vals.append(float(agg[metric]))
        return np.asarray(vals, dtype=np.float64)

    # 4. Independently recompute pre-registered 4-contrast Holm-Bonferroni family
    rep_tests = summary["tests"]
    require(len(rep_tests) == 4, f"P9 Holm family must contain exactly 4 contrasts, got {len(rep_tests)}")
    if require_canonical_family:
        names = [t.get("name") for t in rep_tests]
        require(set(names) == set(P9_HOLM_FAMILY),
                f"P9 Holm family names {names!r} must match canonical {P9_HOLM_FAMILY!r}")

    recomputed_raw_tests: list[dict] = []
    for rep_t in rep_tests:
        base_b, base_a = _extract_arm_spec(rep_t, "baseline")
        cand_b, cand_a = _extract_arm_spec(rep_t, "candidate")
        metric = str(rep_t.get("metric", "bler"))
        reg_filter = rep_t.get("regimes")
        expected_sign = int(rep_t.get("expected_sign", 1))
        base_vec = cluster_metric_vector(base_b, base_a, metric, reg_filter)
        cand_vec = cluster_metric_vector(cand_b, cand_a, metric, reg_filter)
        res = exact_sign_flip_test(base_vec - cand_vec, expected_sign=expected_sign)
        res["name"] = rep_t.get("name")
        res["metric"] = metric
        res["baseline"] = {"budget": base_b, "arm": base_a}
        res["candidate"] = {"budget": cand_b, "arm": cand_a}
        if reg_filter is not None:
            res["regimes"] = list(reg_filter)
        recomputed_raw_tests.append(res)

    holm_recomp = holm_bonferroni_adjust(recomputed_raw_tests, alpha=Fraction(1, 20))
    max_stat_diff = 0.0
    for rep_t, rec_t in zip(rep_tests, holm_recomp):
        require(rep_t["n"] == g, f"Contrast {rec_t['name']} n={rep_t['n']} != G={g}")
        if "df" in rep_t:
            require(rep_t["df"] == g - 1, f"Contrast {rec_t['name']} df={rep_t['df']} != G-1={g - 1}")
        require((rep_t["wins"], rep_t["ties"], rep_t["losses"]) == (rec_t["wins"], rec_t["ties"], rec_t["losses"]),
                f"Sign counts disagree on {rec_t['name']}")
        require(Fraction(rep_t["raw_exact"]) == Fraction(rec_t["raw_exact"])
                and rep_t["raw_p"] == rec_t["raw_p"], f"Raw exact p mismatch on {rec_t['name']}")
        require(Fraction(rep_t["holm_exact"]) == Fraction(rec_t["holm_exact"])
                and rep_t["holm_p"] == rec_t["holm_p"], f"Holm exact p mismatch on {rec_t['name']}")
        if "direction_aligned" in rep_t:
            require(bool(rep_t["direction_aligned"]) == rec_t["direction_aligned"],
                    f"direction_aligned mismatch on {rec_t['name']}")
        if "supported" in rep_t:
            require(bool(rep_t["supported"]) == rec_t["supported"], f"supported mismatch on {rec_t['name']}")
        if "verdict" in rep_t:
            require(rep_t["verdict"] == rec_t["verdict"], f"verdict mismatch on {rec_t['name']}")
        diff_err = float(np.max(np.abs(np.asarray(rep_t["differences"]) - np.asarray(rec_t["differences"]))))
        mean_err = abs(float(rep_t["mean_diff"]) - rec_t["mean_diff"])
        ci_err = float(np.max(np.abs(np.asarray(rep_t["ci95"]) - np.asarray(rec_t["ci95"]))))
        require(diff_err <= 2e-12 and mean_err <= 2e-12 and ci_err <= 1e-10,
                f"Stat mismatch on {rec_t['name']}: diff={diff_err}, mean={mean_err}, ci={ci_err}")
        max_stat_diff = max(max_stat_diff, diff_err, mean_err, ci_err)

    # 5. Independently recompute engineering non-inferiority gate (df = G - 1, margin = +0.01)
    rep_ni = summary["noninferiority"]
    ni_cand_b, ni_cand_a = _extract_arm_spec(rep_ni, "candidate")
    ni_ref_b, ni_ref_a = _extract_arm_spec(rep_ni, "reference")
    ni_metric = str(rep_ni.get("metric", "bler"))
    ni_regimes = rep_ni.get("regimes")
    ni_margin = float(rep_ni["margin"])
    require(math.isclose(ni_margin, 0.01, rel_tol=0.0, abs_tol=1e-15),
            f"Pre-registered NI margin must be +0.01, got {ni_margin}")
    ni_cand_vec = cluster_metric_vector(ni_cand_b, ni_cand_a, ni_metric, ni_regimes)
    ni_ref_vec = cluster_metric_vector(ni_ref_b, ni_ref_a, ni_metric, ni_regimes)
    ni_recomp = noninferiority_test(ni_cand_vec - ni_ref_vec, margin=ni_margin)

    if "n" in rep_ni:
        require(rep_ni["n"] == g, f"NI n={rep_ni['n']} != G={g}")
    if "df" in rep_ni:
        require(rep_ni["df"] == g - 1, f"NI df={rep_ni['df']} != G-1={g - 1}")
    ni_diff_err = float(np.max(np.abs(np.asarray(rep_ni["differences"]) - np.asarray(ni_recomp["differences"]))))
    ni_mean_err = abs(float(rep_ni["mean_diff"]) - ni_recomp["mean_diff"])
    ni_upper_err = abs(float(rep_ni["upper_95"]) - ni_recomp["upper_95"])
    require(ni_diff_err <= 2e-12 and ni_mean_err <= 2e-12 and ni_upper_err <= 2e-12,
            f"NI numerical mismatch: diff={ni_diff_err}, mean={ni_mean_err}, upper={ni_upper_err}")
    require(bool(rep_ni["passes"]) == ni_recomp["passes"], "Incorrect non-inferiority pass/fail verdict")
    max_stat_diff = max(max_stat_diff, ni_diff_err, ni_mean_err, ni_upper_err)

    total_cells = len(records) * sum(len(arms_by_budget_ref[b]) for b in budgets_ref)
    result: dict[str, Any] = {
        "mode": "p9",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "clusters": g,
        "df": g - 1,
        "links_per_cluster": links_per_cluster,
        "links": len(records),
        "cells": total_cells,
        "contrasts": len(holm_recomp),
        "exact_sign_patterns": 1 << g,
        "max_aggregate_diff": float(max_agg_diff),
        "max_stat_diff": float(max_stat_diff),
        "seed_disjointness": seed_audit,
        "holm_tests": holm_recomp,
        "noninferiority": ni_recomp,
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
    }

    if replay_path is not None:
        result.update(verify_replay(raw_lines, records, replay_path, float_tol=replay_float_tol))
    return result


def audit(
    raw_path: Path,
    summary_path: Path,
    replay_path: Optional[Path] = None,
    *,
    mode: str = "auto",
    seed_pools: Optional[Mapping[str, Sequence[int]]] = None,
    replay_float_tol: float = 0.0,
) -> dict:
    """Auto-dispatching independent auditor supporting both P9 and P8-C schemas."""
    if mode == "p8c":
        return audit_p8c(raw_path, summary_path, replay_path, replay_float_tol=replay_float_tol)
    if mode == "p9":
        return audit_p9(raw_path, summary_path, replay_path, seed_pools=seed_pools,
                        replay_float_tol=replay_float_tol)
    if mode != "auto":
        raise ValueError(f"Unsupported mode {mode!r}; expected 'auto', 'p9', or 'p8c'")

    summary = json.loads(summary_path.read_bytes())
    tests = summary.get("tests", [])
    is_p8c = bool(tests) and ("control" in tests[0]) and ("baseline" not in tests[0])
    if is_p8c:
        return audit_p8c(raw_path, summary_path, replay_path, replay_float_tol=replay_float_tol)
    return audit_p9(raw_path, summary_path, replay_path, seed_pools=seed_pools,
                    replay_float_tol=replay_float_tol)


# ---------------------------------------------------------------------------
# End-to-End Synthetic & Historical Self-Test Suite (`--selftest`)
# ---------------------------------------------------------------------------

def _make_synthetic_cell(
    rng: np.random.Generator,
    budget: str,
    bler_prob: float,
    probe_energy: float,
    clean_energy: float,
    nmse_scale: float,
    n0: float,
) -> dict:
    blocks = 16
    extra_blocks = 2 if budget == "sparse" else 0
    first8 = int(rng.binomial(8, np.clip(bler_prob, 0.0, 1.0)))
    last8 = int(rng.binomial(8, np.clip(bler_prob, 0.0, 1.0)))
    block_errors = first8 + last8
    # Enforce exact consistency: block_errors <= info_errors <= 64 * block_errors
    info_errors = int(sum(int(rng.integers(1, 20)) for _ in range(block_errors)))
    coded_errors = int(min(2048, info_errors * 2 + int(rng.integers(40, 160))))
    extra_errors = int(rng.binomial(extra_blocks, np.clip(bler_prob, 0.0, 1.0))) if extra_blocks else 0
    clean_sse = float(clean_energy * nmse_scale * rng.uniform(0.85, 1.15))
    probe_sse = float(clean_sse + 32.0 * n0 * rng.uniform(0.85, 1.15))
    variance_sum = float(max(32.0 * n0, probe_sse * rng.uniform(0.92, 1.08)))
    nll_sum = float(32.0 * math.log(variance_sum / 32.0) + probe_sse / (variance_sum / 32.0))
    return {
        "blocks": blocks,
        "block_errors": block_errors,
        "first8_errors": first8,
        "last8_errors": last8,
        "coded_bits": 2048,
        "coded_errors": coded_errors,
        "info_bits": 1024,
        "info_errors": info_errors,
        "extra_blocks": extra_blocks,
        "extra_errors": extra_errors,
        "probes": 32,
        "probe_sse": probe_sse,
        "probe_energy": probe_energy,
        "clean_sse": clean_sse,
        "clean_energy": clean_energy,
        "variance_sum": variance_sum,
        "nll_sum": nll_sum,
    }


def _build_synthetic_p9_dataset(root_dir: Path, *, g: int = 16, links_per_cluster: int = 24) -> tuple[Path, Path, Path]:
    """Generate a deterministic synthetic P9 formal dataset (G=16, 3 ID regimes) + summary + replay."""
    rng = np.random.default_rng(91009999)
    n0 = n0_from_snr(9.0)
    arms_sparse = ("oracle_native", "oracle_strong", "best_fixed", "classical_gate", "attn_gate", "local_gate")
    arms_dense = ("reset",)
    budgets_arms = {"sparse": arms_sparse, "dense": arms_dense}

    # True synthetic BLER profiles across regimes (designed to exercise all 4 Holm contrasts & NI gate)
    profiles = {
        "continuous": {
            ("sparse", "oracle_native"): (0.39, 0.00),
            ("sparse", "oracle_strong"): (0.29, 0.00),
            ("sparse", "best_fixed"): (0.55, 0.09),
            ("sparse", "classical_gate"): (0.48, 0.08),
            ("sparse", "attn_gate"): (0.41, 0.07),
            ("sparse", "local_gate"): (0.47, 0.08),
            ("dense", "reset"): (0.44, 0.09),
        },
        "partial_shift": {
            ("sparse", "oracle_native"): (0.40, 0.00),
            ("sparse", "oracle_strong"): (0.30, 0.00),
            ("sparse", "best_fixed"): (0.68, 0.24),
            ("sparse", "classical_gate"): (0.52, 0.14),
            ("sparse", "attn_gate"): (0.44, 0.11),
            ("sparse", "local_gate"): (0.51, 0.13),
            ("dense", "reset"): (0.48, 0.11),
        },
        "full_reset": {
            ("sparse", "oracle_native"): (0.39, 0.00),
            ("sparse", "oracle_strong"): (0.29, 0.00),
            ("sparse", "best_fixed"): (0.88, 0.55),
            ("sparse", "classical_gate"): (0.64, 0.22),
            ("sparse", "attn_gate"): (0.56, 0.18),
            ("sparse", "local_gate"): (0.63, 0.21),
            ("dense", "reset"): (0.59, 0.17),
        },
    }

    records: list[dict] = []
    formal_id_seeds: list[int] = []
    for c_id in range(1, g + 1):
        for l_idx in range(links_per_cluster):
            seed = SEED_P9_FORMAL_ID + 1000 * c_id + l_idx
            formal_id_seeds.append(seed)
            regime = REGIMES_ID[l_idx % len(REGIMES_ID)]
            clean_energy = float(rng.uniform(24.0, 42.0))
            probe_energy = float(clean_energy + 32.0 * n0 * rng.uniform(0.8, 1.2))
            rec_budgets: dict[str, dict] = {}
            for budget, arm_list in budgets_arms.items():
                rec_budgets[budget] = {}
                for arm in arm_list:
                    bler_p, nmse_s = profiles[regime][(budget, arm)]
                    rec_budgets[budget][arm] = _make_synthetic_cell(
                        rng, budget, bler_p, probe_energy, clean_energy, nmse_s, n0
                    )
            records.append({
                "cluster": c_id,
                "link": l_idx,
                "seed": seed,
                "regime": regime,
                "snr_db": 9.0,
                "budgets": rec_budgets,
            })

    raw_path = root_dir / "p9_synth_links.jsonl"
    raw_lines = [json.dumps(r, separators=(",", ":"), allow_nan=False) for r in records]
    raw_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    # Fresh replay of clusters 1 and G
    replay_path = root_dir / "p9_synth_replay.jsonl"
    replay_lines = [line for line, r in zip(raw_lines, records) if r["cluster"] in (1, g)]
    replay_path.write_text("\n".join(replay_lines) + "\n", encoding="utf-8")

    # Build matching summary independently
    totals = _aggregate_nested_budget_arm(records, ("sparse", "dense"), budgets_arms)
    clusters_agg = [
        _aggregate_nested_budget_arm(
            [r for r in records if r["cluster"] == c_id], ("sparse", "dense"), budgets_arms
        )
        for c_id in range(1, g + 1)
    ]
    by_regime = {
        reg: _aggregate_nested_budget_arm(
            [r for r in records if r["regime"] == reg], ("sparse", "dense"), budgets_arms
        )
        for reg in REGIMES_ID
    }

    contrast_defs = [
        ("decoder", ("sparse", "oracle_native"), ("sparse", "oracle_strong")),
        ("classical_gate", ("sparse", "best_fixed"), ("sparse", "classical_gate")),
        ("attention_vs_classical", ("sparse", "classical_gate"), ("sparse", "attn_gate")),
        ("attention_vs_local", ("sparse", "local_gate"), ("sparse", "attn_gate")),
    ]
    raw_tests = []
    for name, (bb, ba), (cb, ca) in contrast_defs:
        b_vec = np.array([c[bb][ba]["bler"] for c in clusters_agg], dtype=np.float64)
        c_vec = np.array([c[cb][ca]["bler"] for c in clusters_agg], dtype=np.float64)
        res = exact_sign_flip_test(b_vec - c_vec, expected_sign=1)
        res["name"] = name
        res["metric"] = "bler"
        res["baseline"] = {"budget": bb, "arm": ba}
        res["candidate"] = {"budget": cb, "arm": ca}
        raw_tests.append(res)
    holm_tests = holm_bonferroni_adjust(raw_tests, alpha=Fraction(1, 20))

    ni_cand = np.array([c["sparse"]["attn_gate"]["bler"] for c in clusters_agg], dtype=np.float64)
    ni_ref = np.array([c["dense"]["reset"]["bler"] for c in clusters_agg], dtype=np.float64)
    ni = noninferiority_test(ni_cand - ni_ref, margin=0.01)
    ni["metric"] = "bler"
    ni["candidate"] = {"budget": "sparse", "arm": "attn_gate"}
    ni["reference"] = {"budget": "dense", "arm": "reset"}

    summary = {
        "seed_pools": {
            "train": [SEED_P9_TRAIN + i for i in range(64)],
            "dev": [SEED_P9_DEV + i for i in range(64)],
            "formal_id": formal_id_seeds,
            "formal_ood": [SEED_P9_FORMAL_OOD + i for i in range(64)],
        },
        "totals": totals,
        "by_regime": by_regime,
        "tests": holm_tests,
        "noninferiority": ni,
        "clusters": clusters_agg,
    }
    summary_path = root_dir / "p9_synth_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return raw_path, summary_path, replay_path


def run_selftest(output_path: Optional[Path] = None) -> dict:
    """Run complete independent physics, ML certificate, P8-C cross-check, G=16 audit, and firewall self-test."""
    from p8 import link as p8_ofdm
    from p8c import link as p8c_link

    # 1. Independent Physics Reference Check (continuous + phase-continuous piecewise Doppler shifts)
    seed_phys = 3117
    pair = p8c_link.simulate(seed_phys)
    channel = p8_ofdm.draw_channel(p8_ofdm.stream(seed_phys, 0))
    a_grid = p8_ofdm.tx_grid(p8_ofdm.traffic(p8_ofdm.stream(seed_phys, 1)), p8_ofdm.PILOT_ROWS)
    b_grid = p8c_link.tx_grid(pair.b_words, p8c_link.LAYOUTS["sparse"])

    y_a_ser, y_b_ser = serial_propagate_ab(a_grid, b_grid, channel)
    y_a_p8 = p8_ofdm.propagate(a_grid, channel, start_row=0)
    y_b_p8c = p8c_link.propagate(b_grid, channel, start_row=T)
    err_a_vs_p8 = float(np.max(np.abs(y_a_ser - y_a_p8)))
    err_b_vs_p8c = float(np.max(np.abs(y_b_ser - y_b_p8c)))
    require(err_a_vs_p8 < 1e-13 and err_b_vs_p8c < 1e-13,
            f"Continuous serial reference mismatch: A={err_a_vs_p8:.3e}, B={err_b_vs_p8c:.3e}")

    shift_checks = {}
    for shift_row in (0, 6):
        shift = PiecewiseDopplerShift(ray_index=1, delta_freq=0.055, t_shift=row_boundary_time(shift_row, frame="B"))
        cont = verify_phase_continuity(channel, shift)
        ser_res = serial_propagate_ab(a_grid, b_grid, channel, shift=shift, return_details=True)
        blk_a = blockwise_propagate_piecewise(a_grid, channel, start_row=0, is_ofdm=True, shift=shift)
        blk_b = blockwise_propagate_piecewise(b_grid, channel, start_row=T, is_ofdm=False, shift=shift)
        err_a = float(np.max(np.abs(ser_res.y_a - blk_a)))
        err_b = float(np.max(np.abs(ser_res.y_b - blk_b)))
        require(cont["continuous_ok"] and err_a < 1e-13 and err_b < 1e-13,
                f"Piecewise shift check failed at shift_row={shift_row}")
        shift_checks[f"shift_row_{shift_row}"] = {
            "t_shift": shift.t_shift,
            "phase_jump_rad": cont["phase_jump_rad"],
            "gain_jump": cont["gain_jump"],
            "reanchored_max_gain_err": cont["reanchored_max_gain_err"],
            "serial_vs_blockwise_err_a": err_a,
            "serial_vs_blockwise_err_b": err_b,
            "passed": True,
        }

    # 2. Exhaustive (16, 8, 5) mini-codebook ML certificate verifier (200 trials)
    mini_ml_6db = exhaustive_ml_verify_mini_code(n_trials=200, snr_db=6.0, seed=91001608)
    mini_ml_9db = exhaustive_ml_verify_mini_code(n_trials=200, snr_db=9.0, seed=91001609)
    require(mini_ml_6db["soundness_verified"] and mini_ml_6db["inequality_holds"],
            "Mini-codebook 6 dB ML certificate check failed")
    require(mini_ml_9db["soundness_verified"] and mini_ml_9db["inequality_holds"],
            "Mini-codebook 9 dB ML certificate check failed")

    # 3. Historical P8-C G=10 cross-check against p8c/results/links.jsonl & summary.json
    repo_root = Path(__file__).resolve().parent.parent
    p8c_raw = repo_root / "p8c" / "results" / "links.jsonl"
    p8c_summary = repo_root / "p8c" / "results" / "summary.json"
    p8c_report = audit_p8c(p8c_raw, p8c_summary)

    # 4. Synthetic P9 G=16 end-to-end audit (65,536 sign patterns) + timing + directional alignment check
    t0 = time.perf_counter()
    all_pos_16 = exact_sign_flip_test(np.linspace(0.05, 0.20, 16), expected_sign=1)
    g16_elapsed_ms = (time.perf_counter() - t0) * 1000.0
    require(all_pos_16["raw_exact"] == "1/32768" and all_pos_16["exact_patterns"] == 65536,
            f"G=16 all-positive exact p-value mismatch: {all_pos_16['raw_exact']}")

    # Directional-refutation sanity check: significant two-sided p but opposite direction -> verdict == 'refuted'
    opposite_test = exact_sign_flip_test(-np.linspace(0.05, 0.20, 16), expected_sign=1)
    opposite_holm = holm_bonferroni_adjust([opposite_test], alpha=Fraction(1, 20))[0]
    require(opposite_holm["significant_two_sided"] and (not opposite_holm["direction_aligned"])
            and (not opposite_holm["supported"]) and opposite_holm["verdict"] == "refuted",
            "Directional alignment gate failed to reject significant opposite-sign contrast")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        synth_raw, synth_summary, synth_replay = _build_synthetic_p9_dataset(tmp_dir, g=16, links_per_cluster=24)
        t_audit_start = time.perf_counter()
        p9_synth_report = audit_p9(synth_raw, synth_summary, synth_replay)
        p9_audit_elapsed_ms = (time.perf_counter() - t_audit_start) * 1000.0

    # 5. Firewall checks: static no-hardcoded-N0 on P9 audit modules + PublicObservation probe masking
    p9_dir = Path(__file__).resolve().parent
    n0_clean_check = check_no_hardcoded_n0(
        [p9_dir / "contract.py", p9_dir / "independent_ref.py", p9_dir / "audit.py"],
        strict=True,
    )
    n0_legacy_catch = check_no_hardcoded_n0([repo_root / "p8" / "link.py", repo_root / "p8c" / "link.py"])
    require(not n0_legacy_catch["clean"] and n0_legacy_catch["violation_count"] >= 2,
            "Static N0 detector failed to catch legacy hardcoded N0 in p8/link.py & p8c/link.py")

    y_b_masked = pair.b["sparse"].astype(np.complex128).copy()
    y_b_masked[PROBE_ROW] = complex(math.nan, math.nan)
    obs_clean = PublicObservation(
        y_a=pair.a.astype(np.complex128).copy(),
        y_b=y_b_masked,
        training_rows=SC_LAYOUTS["sparse"],
        probe_row=PROBE_ROW,
        n0=n0_from_snr(9.0),
        budget="sparse",
    )
    obs_report = verify_public_observation(obs_clean, strict=True)

    def _clean_dummy_receiver(obs: PublicObservation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        llrs = np.zeros((T, N, 4), dtype=np.float64)
        pred = np.mean(obs.y_b[list(obs.training_rows)], axis=0)
        var = np.full(N, obs.n0 + 0.1, dtype=np.float64)
        return llrs, pred, var

    def _leaky_dummy_receiver(obs: PublicObservation) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        llrs = np.zeros((T, N, 4), dtype=np.float64)
        pred = 0.5 * obs.y_b[obs.probe_row]  # planted probe leak!
        var = np.full(N, obs.n0 + 0.1, dtype=np.float64)
        return llrs, pred, var

    anti_leak_clean = audit_receiver_anti_leakage(_clean_dummy_receiver, obs_clean, strict=True)
    anti_leak_caught = audit_receiver_anti_leakage(_leaky_dummy_receiver, obs_clean, strict=False)
    require(not anti_leak_caught["passed"], "Anti-leakage detector failed to catch planted probe leak")

    selftest_report = {
        "status": "PASS",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "physics_reference": {
            "continuous_err_a_vs_p8": err_a_vs_p8,
            "continuous_err_b_vs_p8c": err_b_vs_p8c,
            "continuous_passed": True,
            "piecewise_doppler_shifts": shift_checks,
        },
        "mini_codebook_ml_verifier": {
            "snr_6db": mini_ml_6db,
            "snr_9db": mini_ml_9db,
        },
        "p8c_g10_crosscheck": {
            "clusters": p8c_report["clusters"],
            "df": p8c_report["df"],
            "exact_sign_patterns": p8c_report["exact_sign_patterns"],
            "max_aggregate_diff": p8c_report["max_aggregate_diff"],
            "max_stat_diff": p8c_report["max_stat_diff"],
            "max_ci95_diff": p8c_report["max_ci95_diff"],
            "holm_exact_values": [
                {"control": t["control"], "metric": t["metric"],
                 "raw_exact": t["raw_exact"], "holm_exact": t["holm_exact"],
                 "verdict": t["verdict"]}
                for t in p8c_report["holm_tests"]
            ],
            "noninferiority": {
                "mean_diff": p8c_report["noninferiority"]["mean_diff"],
                "upper_95": p8c_report["noninferiority"]["upper_95"],
                "margin": p8c_report["noninferiority"]["margin"],
                "passes": p8c_report["noninferiority"]["passes"],
            },
            "delivery_summary": p8c_report["delivery_summary"],
            "passed": True,
        },
        "p9_g16_synthetic_audit": {
            "g16_all_positive_raw_exact": all_pos_16["raw_exact"],
            "g16_exact_patterns": all_pos_16["exact_patterns"],
            "g16_single_test_ms": round(g16_elapsed_ms, 3),
            "g16_full_audit_ms": round(p9_audit_elapsed_ms, 3),
            "opposite_sign_verdict": opposite_holm["verdict"],
            "clusters": p9_synth_report["clusters"],
            "df": p9_synth_report["df"],
            "links": p9_synth_report["links"],
            "max_aggregate_diff": p9_synth_report["max_aggregate_diff"],
            "max_stat_diff": p9_synth_report["max_stat_diff"],
            "replay_all_fields_match": p9_synth_report["replay_all_fields_match"],
            "replay_exact_bytes_match": p9_synth_report["replay_exact_bytes_match"],
            "seed_disjointness": p9_synth_report["seed_disjointness"]["disjoint"],
            "holm_verdicts": [
                {"name": t["name"], "raw_exact": t["raw_exact"], "holm_exact": t["holm_exact"],
                 "mean_diff": t["mean_diff"], "direction_aligned": t["direction_aligned"],
                 "supported": t["supported"], "verdict": t["verdict"]}
                for t in p9_synth_report["holm_tests"]
            ],
            "noninferiority": {
                "df": p9_synth_report["noninferiority"]["df"],
                "mean_diff": p9_synth_report["noninferiority"]["mean_diff"],
                "upper_95": p9_synth_report["noninferiority"]["upper_95"],
                "margin": p9_synth_report["noninferiority"]["margin"],
                "passes": p9_synth_report["noninferiority"]["passes"],
            },
            "passed": True,
        },
        "firewall_checks": {
            "p9_modules_no_hardcoded_n0": n0_clean_check["clean"],
            "legacy_p8_n0_detected_count": n0_legacy_catch["violation_count"],
            "public_observation_valid": obs_report["valid"],
            "clean_receiver_passed": anti_leak_clean["passed"],
            "planted_probe_leak_caught": not anti_leak_caught["passed"],
        },
    }

    target_out = output_path if output_path is not None else (RESULTS_DIR / "p9_audit_selftest.json")
    target_out.parent.mkdir(parents=True, exist_ok=True)
    target_out.write_text(json.dumps(selftest_report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return selftest_report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 9 independent blind statistical auditor & firewall.")
    parser.add_argument("--selftest", action="store_true",
                        help="Run full synthetic & historical self-test and write p9/results/p9_audit_selftest.json.")
    parser.add_argument("--raw", type=Path, help="Raw per-link JSONL records.")
    parser.add_argument("--summary", type=Path, help="Summary JSON to independently verify.")
    parser.add_argument("--replay", type=Path, default=None, help="Fresh single-worker replay JSONL.")
    parser.add_argument("--seeds", type=Path, default=None, help="Optional JSON seed-pool manifest.")
    parser.add_argument("--mode", choices=("auto", "p9", "p8c"), default="auto")
    parser.add_argument("--replay-float-tol", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON report path.")
    args = parser.parse_args(argv)

    if args.selftest:
        out_path = args.output if args.output is not None else (RESULTS_DIR / "p9_audit_selftest.json")
        report = run_selftest(out_path)
        print(json.dumps({
            "status": report["status"],
            "output": str(out_path),
            "p8c_max_stat_diff": report["p8c_g10_crosscheck"]["max_stat_diff"],
            "g16_single_test_ms": report["p9_g16_synthetic_audit"]["g16_single_test_ms"],
        }, indent=2))
        return

    if args.raw is None or args.summary is None:
        parser.error("Provide --raw and --summary (or run --selftest).")

    seed_pools = json.loads(args.seeds.read_text(encoding="utf-8")) if args.seeds is not None else None
    result = audit(
        args.raw,
        args.summary,
        args.replay,
        mode=args.mode,
        seed_pools=seed_pools,
        replay_float_tol=args.replay_float_tol,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
