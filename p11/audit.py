"""Independent Phase 11 statistical auditor, raw-cohort recomputation engine, and replay verifier.

Builds on p9.audit's exact 2^G (G = 16 -> 65,536-state) rational Fraction sign-flip permutation
engine, exact step-down Holm-Bonferroni adjuster, and one-sided 95% paired-t non-inferiority test
(df = G - 1). Strictly independent from M2/M3 receiver, channel, and PyTorch implementations.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import tempfile
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import scipy
from scipy.linalg import eigh

from p9.audit import (
    exact_sign_flip_test,
    exact_student_t_ppf,
    holm_bonferroni_adjust,
    noninferiority_test,
    require,
    verify_paired_energies,
)
from p9.independent_ref import serial_propagate_ab
from p11.contract import (
    ALL_REGIMES,
    CORE_ROWS,
    CP,
    DELAYS,
    FRAME_DURATION,
    GateOutput,
    N,
    OFDM_PILOT_ROWS,
    OFFGRID_ALL_REGIMES,
    OFFGRID_REGIMES_ID,
    OFFGRID_REGIMES_OOD,
    OffGridObservation,
    P10_BASELINE_COMMIT,
    POWERS,
    PROBE_ROW,
    PublicObservation,
    REGIMES_ID,
    REGIMES_OOD,
    SC_LAYOUTS,
    SEED_P11_DEV,
    SEED_P11_E2E,
    SEED_P11_TRAIN,
    T,
    AssimilationResult,
    build_offgrid_sc_channel_matrix,
    n0_from_snr,
)
from p11.independent_ref import (
    P4_STAGE3_LITERAL_SEED,
    P8C_BASE_SEED,
    P8_MANIFOLD_BASE_SEED,
    PiecewiseDopplerShift,
    audit_assimilation_anti_leakage,
    audit_fold_target_isolation,
    check_assimilation_result_adapter,
    check_auditor_independence,
    check_p11_no_hardcoded_n0,
    check_p11_seed_disjointness,
    dirichlet_fractional_delay_matrix,
    independent_complex_gaussian_loglik,
    independent_offgrid_propagate_ab,
    independent_subset_bayes_posteriors,
    row_boundary_time,
    verify_fractional_delay_operator,
    verify_offgrid_observation,
    verify_offgrid_phase_continuity,
    verify_p11_public_observation,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"
VALID_VERDICTS = frozenset({"supported", "refuted", "insufficient"})
VERDICT_ZH: dict[str, str] = {
    "supported": "支持",
    "refuted": "反证",
    "insufficient": "证据不足",
}
VALID_VERDICTS_ZH = frozenset({"支持", "反证", "证据不足"})

P11_PRIMARY_CANONICAL_CONTRASTS = (
    "chained_vs_p9_jt",
    "chained_vs_payload_only",
)
P11_SECONDARY_CANONICAL_CONTRASTS = (
    "transport_attn_vs_abs_no_transport",
    "transport_attn_vs_local",
    "transport_attn_vs_no_history",
)
_P11_PRIMARY_CANONICAL_ARMS: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
    "chained_vs_p9_jt": (("sparse", "p9_jeffreys_student_t"), ("sparse", "p11_chained_wavefront")),
    "chained_vs_payload_only": (("sparse", "p11_payload_only"), ("sparse", "p11_chained_wavefront")),
}
_P11_SECONDARY_CANONICAL_ARMS: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
    "transport_attn_vs_abs_no_transport": (("sparse", "em10_abs_no_transport"), ("sparse", "em10_transport_attn")),
    "transport_attn_vs_local": (("sparse", "em10_local"), ("sparse", "em10_transport_attn")),
    "transport_attn_vs_no_history": (("sparse", "em10_no_history"), ("sparse", "em10_transport_attn")),
}
_P11_SECONDARY_OOD_CONTRAST_ARMS: dict[str, tuple[tuple[str, str], tuple[str, str]]] = {
    "ood_transport_attn_vs_abs_no_transport": (("sparse", "em10_abs_no_transport"), ("sparse", "em10_transport_attn")),
    "ood_transport_attn_vs_local": (("sparse", "em10_local"), ("sparse", "em10_transport_attn")),
    "ood_transport_attn_vs_no_history": (("sparse", "em10_no_history"), ("sparse", "em10_transport_attn")),
}

P10_9E9D93B_REFERENCE: dict[str, Any] = {
    "commit": P10_BASELINE_COMMIT,
    "3id_code_gate_bler": 0.2073,
    "continuous_bler": 0.1045,
    "partial_shift_bler": 0.2288,
    "full_reset_bler": 0.2886,
    "full_reset_rebuild_bler": 0.2900,
    "oracle_common_phase_subsets": [
        {"batch": "main", "links": 32, "blocks": 512, "before_errors": 132, "after_errors": 40},
        {"batch": "rep", "links": 32, "blocks": 512, "before_errors": 154, "after_errors": 26},
    ],
}

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
ADDITIVE_FLOAT_FIELDS = (
    "probe_sse",
    "probe_energy",
    "clean_sse",
    "clean_energy",
    "variance_sum",
    "nll_sum",
)


def verdict_to_zh(verdict: str) -> str:
    require(verdict in VALID_VERDICTS, f"Invalid verdict {verdict!r}")
    return VERDICT_ZH[verdict]


# ---------------------------------------------------------------------------
# 1. Exact G <= 16 (65,536-state) Dual-Axis Holm-Bonferroni Family Auditor
# ---------------------------------------------------------------------------

def _reject_bool_and_nonreal(val: Any, label: str) -> None:
    if isinstance(val, (bool, np.bool_, np.timedelta64, np.datetime64, complex, np.complexfloating, str, bytes)):
        raise TypeError(f"{label} must contain real numbers, not {type(val).__name__}")
    if isinstance(val, np.ndarray):
        if val.dtype == np.bool_ or val.dtype.kind not in ("i", "u", "f", "O"):
            raise TypeError(f"{label} has invalid dtype {val.dtype}")
        if val.dtype == object:
            for elem in val.flat:
                _reject_bool_and_nonreal(elem, label)
        return
    if isinstance(val, (list, tuple)):
        for item in val:
            _reject_bool_and_nonreal(item, label)


def _strict_1d_float_array(val: Any, expected_len: int, label: str) -> np.ndarray:
    if isinstance(val, (str, bytes, Mapping)):
        raise ValueError(f"{label} must be a 1D numeric sequence of length {expected_len}, got {type(val).__name__}")
    _reject_bool_and_nonreal(val, label)
    raw = np.asarray(val)
    if raw.dtype == np.bool_ or raw.ndim != 1 or raw.shape != (expected_len,):
        raise ValueError(
            f"{label} must be a 1D non-bool numeric array of shape ({expected_len},), got shape={raw.shape} dtype={raw.dtype}"
        )
    arr = np.asarray(raw, dtype=np.float64)
    require(np.isfinite(arr).all(), f"{label} contains non-finite values")
    return arr


def _normalize_contrast_spec(
    name: str,
    spec: Any,
    family_name: str,
) -> dict[str, Any]:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Contrast name in {family_name} must be a non-empty string, got {name!r}")

    if isinstance(spec, Mapping):
        if "diffs" in spec:
            _reject_bool_and_nonreal(spec["diffs"], f"{family_name}/{name}.diffs")
            diffs = np.asarray(spec["diffs"], dtype=np.float64)
        elif "baseline" in spec and "candidate" in spec:
            _reject_bool_and_nonreal(spec["baseline"], f"{family_name}/{name}.baseline")
            _reject_bool_and_nonreal(spec["candidate"], f"{family_name}/{name}.candidate")
            base = np.asarray(spec["baseline"], dtype=np.float64)
            cand = np.asarray(spec["candidate"], dtype=np.float64)
            if base.shape != cand.shape or base.ndim != 1:
                raise ValueError(
                    f"Contrast {family_name}/{name} baseline {base.shape} and candidate {cand.shape} must be matching 1D arrays"
                )
            diffs = base - cand
        else:
            raise ValueError(
                f"Contrast {family_name}/{name} mapping must contain 'diffs' or ('baseline', 'candidate')"
            )
        raw_sign = spec.get("expected_sign", 1)
        if isinstance(raw_sign, (bool, np.bool_)) or not isinstance(raw_sign, (int, np.integer)):
            raise TypeError(f"Contrast {family_name}/{name} expected_sign must be int, got {type(raw_sign).__name__}")
        expected_sign = int(raw_sign)
        meta = {
            k: v
            for k, v in spec.items()
            if k not in ("diffs", "baseline", "candidate", "expected_sign")
        }
    else:
        _reject_bool_and_nonreal(spec, f"{family_name}/{name}")
        diffs = np.asarray(spec, dtype=np.float64)
        expected_sign = 1
        meta = {}

    if diffs.ndim != 1:
        raise ValueError(f"Contrast {family_name}/{name} diffs must be 1D, got shape {diffs.shape}")
    return {
        "name": name,
        "family": family_name,
        "diffs": diffs,
        "expected_sign": expected_sign,
        "meta": meta,
    }


def _normalize_family_contrasts(
    contrasts: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    family_name: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    if isinstance(contrasts, Mapping):
        if len(contrasts) == 0:
            raise ValueError(f"Holm family {family_name!r} must contain at least one contrast")
        for cname, cspec in contrasts.items():
            item = _normalize_contrast_spec(str(cname), cspec, family_name)
            if item["name"] in seen_names:
                raise ValueError(f"Duplicate contrast name {item['name']!r} in {family_name}")
            seen_names.add(item["name"])
            normalized.append(item)
    elif isinstance(contrasts, Sequence) and not isinstance(contrasts, (str, bytes)):
        if len(contrasts) == 0:
            raise ValueError(f"Holm family {family_name!r} must contain at least one contrast")
        for idx, entry in enumerate(contrasts):
            if not isinstance(entry, Mapping) or "name" not in entry:
                raise ValueError(f"Sequence entry {idx} in {family_name!r} must be a mapping with a 'name' key")
            cname = str(entry["name"])
            item = _normalize_contrast_spec(cname, entry, family_name)
            if item["name"] in seen_names:
                raise ValueError(f"Duplicate contrast name {item['name']!r} in {family_name}")
            seen_names.add(item["name"])
            normalized.append(item)
    else:
        raise TypeError(
            f"Expected Mapping or Sequence of contrast mappings for {family_name!r}, got {type(contrasts).__name__}"
        )

    return normalized


def _audit_single_holm_family(
    contrasts: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    family_name: str,
    *,
    alpha: Fraction = Fraction(1, 20),
    expected_g: Optional[int] = None,
) -> dict[str, Any]:
    norm_items = _normalize_family_contrasts(contrasts, family_name)
    raw_results: list[dict[str, Any]] = []
    g_ref: Optional[int] = None

    for item in norm_items:
        res = exact_sign_flip_test(item["diffs"], expected_sign=item["expected_sign"])
        g_cur = int(res["n"])
        if g_ref is None:
            g_ref = g_cur
        elif g_cur != g_ref:
            raise ValueError(
                f"Inconsistent cluster count G in {family_name}: {item['name']} has G={g_cur} vs G={g_ref}"
            )
        if expected_g is not None and g_cur != int(expected_g):
            raise ValueError(
                f"Expected G={expected_g} in {family_name}/{item['name']}, got G={g_cur}"
            )
        res["name"] = item["name"]
        res["family"] = family_name
        for mk, mv in item["meta"].items():
            if mk not in res:
                res[mk] = mv
        raw_results.append(res)

    assert g_ref is not None
    holm_results = holm_bonferroni_adjust(raw_results, alpha=alpha)
    for r in holm_results:
        require(r["verdict"] in VALID_VERDICTS, f"Invalid verdict {r['verdict']!r} in {family_name}/{r['name']}")
        r["verdict_zh"] = VERDICT_ZH[r["verdict"]]
        require(Fraction(r["holm_exact"]) >= Fraction(r["raw_exact"]), "Holm p-value smaller than raw p-value")
        require(Fraction(r["holm_exact"]) <= Fraction(1), "Holm p-value exceeded 1")

    return {
        "family": family_name,
        "n_contrasts": len(holm_results),
        "clusters": g_ref,
        "df": g_ref - 1,
        "exact_sign_patterns": 1 << g_ref,
        "alpha_exact": str(alpha),
        "alpha": float(alpha),
        "contrasts": holm_results,
        "by_name": {r["name"]: r for r in holm_results},
        "verdicts": {r["name"]: r["verdict"] for r in holm_results},
        "verdicts_zh": {r["name"]: r["verdict_zh"] for r in holm_results},
        "all_supported": bool(all(r["verdict"] == "supported" for r in holm_results)),
    }


def audit_p11_holm_families(
    primary_contrasts: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    secondary_contrasts: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    alpha: Fraction = Fraction(1, 20),
    expected_g: Optional[int] = None,
) -> dict[str, Any]:
    """Run independent exact 2^G (65,536-state for G=16) Holm-Bonferroni families for Primary & Secondary axes."""
    alpha_frac = alpha if isinstance(alpha, Fraction) else Fraction(alpha).limit_denominator(10**9)
    require(Fraction(0) < alpha_frac < Fraction(1), f"alpha must lie in (0, 1), got {alpha_frac}")

    p1 = _audit_single_holm_family(
        primary_contrasts, "primary", alpha=alpha_frac, expected_g=expected_g
    )
    p2 = _audit_single_holm_family(
        secondary_contrasts, "secondary", alpha=alpha_frac, expected_g=expected_g
    )
    return {
        "alpha_exact": str(alpha_frac),
        "alpha": float(alpha_frac),
        "primary": p1,
        "secondary": p2,
    }


# ---------------------------------------------------------------------------
# 2. Engineering Non-Inferiority + Exact Paired Net-Delivered-Bits Gain Auditor
# ---------------------------------------------------------------------------

def audit_engineering_noninferiority(
    sparse_cluster_bler: Sequence[float] | np.ndarray,
    dense_cluster_bler: Sequence[float] | np.ndarray,
    sparse_cluster_bits: Sequence[float] | np.ndarray,
    dense_cluster_bits: Sequence[float] | np.ndarray,
    margin: float = 0.01,
    *,
    alpha: Fraction = Fraction(1, 20),
) -> dict[str, Any]:
    """Audit sparse vs dense BLER non-inferiority (one-sided 95% paired-t, df=G-1) AND exact net bits gain.

    NI Verdict mapping:
      - upper_95 < margin  -> "supported" ("支持")
      - lower_95 >= margin -> "refuted"   ("反证")
      - otherwise          -> "insufficient" ("证据不足")
    """
    _reject_bool_and_nonreal(sparse_cluster_bler, "sparse_cluster_bler")
    _reject_bool_and_nonreal(dense_cluster_bler, "dense_cluster_bler")
    _reject_bool_and_nonreal(sparse_cluster_bits, "sparse_cluster_bits")
    _reject_bool_and_nonreal(dense_cluster_bits, "dense_cluster_bits")

    sb = np.asarray(sparse_cluster_bler, dtype=np.float64)
    db = np.asarray(dense_cluster_bler, dtype=np.float64)
    sbits = np.asarray(sparse_cluster_bits, dtype=np.float64)
    dbits = np.asarray(dense_cluster_bits, dtype=np.float64)

    for label, arr in (
        ("sparse_cluster_bler", sb),
        ("dense_cluster_bler", db),
        ("sparse_cluster_bits", sbits),
        ("dense_cluster_bits", dbits),
    ):
        require(arr.ndim == 1, f"{label} must be 1D, got shape {arr.shape}")
        require(np.isfinite(arr).all(), f"{label} contains non-finite values")

    g = int(sb.size)
    require(db.size == g and sbits.size == g and dbits.size == g,
            f"All four cluster arrays must share identical length G, got {sb.size}, {db.size}, {sbits.size}, {dbits.size}")
    require(2 <= g <= 16, f"Cluster count G must satisfy 2 <= G <= 16, got {g}")
    require(bool(np.all((sb >= 0.0) & (sb <= 1.0))), "sparse_cluster_bler must lie in [0, 1]")
    require(bool(np.all((db >= 0.0) & (db <= 1.0))), "dense_cluster_bler must lie in [0, 1]")
    require(bool(np.all(sbits >= 0.0) and np.all(dbits >= 0.0)), "Cluster delivered bits must be non-negative")

    if isinstance(margin, (bool, np.bool_)):
        raise TypeError(f"margin must be a real number, not {type(margin).__name__}")
    margin_f = float(margin)
    require(math.isfinite(margin_f) and margin_f > 0.0, f"margin must be positive finite, got {margin!r}")
    if isinstance(alpha, (bool, np.bool_)):
        raise TypeError(f"alpha must be a Fraction or real number in (0, 1), not {type(alpha).__name__}")
    if isinstance(alpha, Fraction):
        alpha_frac = alpha
    else:
        alpha_f = float(alpha)
        require(math.isfinite(alpha_f), f"alpha must be finite, got {alpha!r}")
        alpha_frac = Fraction(alpha_f).limit_denominator(10**9)
    require(Fraction(0) < alpha_frac < Fraction(1), f"alpha must lie in (0, 1), got {alpha!r}")

    diff_bler = sb - db
    bler_ni = dict(noninferiority_test(diff_bler, margin=margin_f))
    se = float(bler_ni["std_diff"]) / math.sqrt(float(g))
    t95 = exact_student_t_ppf(0.95, g - 1)
    lower_95 = float(bler_ni["mean_diff"]) - t95 * se
    upper_95 = float(bler_ni["upper_95"])

    if upper_95 < margin_f:
        ni_verdict = "supported"
    elif lower_95 >= margin_f:
        ni_verdict = "refuted"
    else:
        ni_verdict = "insufficient"

    bler_ni["lower_95"] = float(lower_95)
    bler_ni["verdict"] = ni_verdict
    bler_ni["verdict_zh"] = VERDICT_ZH[ni_verdict]

    bits_gain_raw = exact_sign_flip_test(sbits - dbits, expected_sign=1)
    bits_raw_frac = Fraction(bits_gain_raw["raw_exact"])
    bits_sig = bool(bits_raw_frac < alpha_frac)
    bits_supported = bool(bits_sig and bits_gain_raw["direction_aligned"])
    if bits_supported:
        bits_verdict = "supported"
    elif bits_sig:
        bits_verdict = "refuted"
    else:
        bits_verdict = "insufficient"

    sparse_mean_bler = float(math.fsum(sb.tolist()) / float(g))
    dense_mean_bler = float(math.fsum(db.tolist()) / float(g))
    sparse_mean_bits = float(math.fsum(sbits.tolist()) / float(g))
    dense_mean_bits = float(math.fsum(dbits.tolist()) / float(g))
    rel_bits_gain_pct = (
        float(100.0 * (sparse_mean_bits - dense_mean_bits) / dense_mean_bits)
        if dense_mean_bits > 0.0
        else 0.0
    )

    bits_gain = {
        **bits_gain_raw,
        "alpha_exact": str(alpha_frac),
        "alpha": float(alpha_frac),
        "significant_two_sided": bits_sig,
        "supported": bits_supported,
        "verdict": bits_verdict,
        "verdict_zh": VERDICT_ZH[bits_verdict],
    }

    return {
        "clusters": g,
        "df": g - 1,
        "margin": margin_f,
        "sparse_mean_bler": sparse_mean_bler,
        "dense_mean_bler": dense_mean_bler,
        "bler_noninferiority": bler_ni,
        "passes": bool(bler_ni["passes"]),
        "verdict": ni_verdict,
        "verdict_zh": VERDICT_ZH[ni_verdict],
        "sparse_mean_bits": sparse_mean_bits,
        "dense_mean_bits": dense_mean_bits,
        "mean_bits_gain": float(bits_gain_raw["mean_diff"]),
        "relative_bits_gain_pct": rel_bits_gain_pct,
        "net_bits_gain": bits_gain,
        "engineering_dominance_passes": bool(bler_ni["passes"] and bits_supported),
    }


# ---------------------------------------------------------------------------
# 3. Generic Exact Field-Level Replay Verifier (Ignoring Only Wall-Clock *_ms)
# ---------------------------------------------------------------------------

def _is_wallclock_ms_key(key: Any) -> bool:
    return isinstance(key, str) and key.endswith("_ms")


def _load_records_arg(records_or_path: Sequence[Mapping[str, Any]] | Path) -> list[Mapping[str, Any]]:
    if isinstance(records_or_path, Path):
        lines = records_or_path.read_bytes().splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    if isinstance(records_or_path, (str, bytes)):
        raise TypeError("Pass a sequence of record dicts or a pathlib.Path to a JSONL file, not str/bytes")
    return list(records_or_path)


def _infer_key_fields(
    orig_records: Sequence[Mapping[str, Any]],
    replay_records: Sequence[Mapping[str, Any]],
    explicit_keys: Optional[Sequence[str]],
) -> Optional[tuple[str, ...]]:
    if explicit_keys is not None:
        return tuple(str(k) for k in explicit_keys)

    candidates = (
        ("cluster", "link", "seed"),
        ("seed", "regime"),
        ("seed",),
        ("id",),
    )
    all_recs = list(orig_records) + list(replay_records)
    for cand in candidates:
        if all(isinstance(r, Mapping) and all(k in r for k in cand) for r in all_recs):
            orig_keys = [tuple(r[k] for k in cand) for r in orig_records]
            if len(set(orig_keys)) == len(orig_keys):
                return cand
    return None


def _compare_replay_values(
    expected: Any,
    observed: Any,
    path: str,
    float_tol: float,
    mismatches: list[str],
    ignored_ms_fields: set[str],
    stats: dict[str, Any],
) -> None:
    if isinstance(expected, Mapping) or isinstance(observed, Mapping):
        if not (isinstance(expected, Mapping) and isinstance(observed, Mapping)):
            mismatches.append(f"Type mismatch at {path}: {type(expected).__name__} vs {type(observed).__name__}")
            return
        exp_keys = set(expected.keys())
        obs_keys = set(observed.keys())
        for k in sorted(exp_keys | obs_keys, key=str):
            if _is_wallclock_ms_key(k):
                ignored_ms_fields.add(f"{path}.{k}")
                stats["ignored_ms_count"] += 1
        exp_non_ms = {k for k in exp_keys if not _is_wallclock_ms_key(k)}
        obs_non_ms = {k for k in obs_keys if not _is_wallclock_ms_key(k)}
        if exp_non_ms != obs_non_ms:
            missing = sorted(exp_non_ms - obs_non_ms, key=str)
            extra = sorted(obs_non_ms - exp_non_ms, key=str)
            mismatches.append(f"Key mismatch at {path}: missing={missing}, extra={extra}")
            return
        for key in sorted(exp_non_ms, key=str):
            _compare_replay_values(
                expected[key], observed[key], f"{path}.{key}", float_tol, mismatches, ignored_ms_fields, stats
            )
        return

    if isinstance(expected, np.ndarray) or isinstance(observed, np.ndarray):
        exp_arr = np.asarray(expected)
        obs_arr = np.asarray(observed)
        if exp_arr.shape != obs_arr.shape:
            mismatches.append(f"Array shape mismatch at {path}: {exp_arr.shape} vs {obs_arr.shape}")
            return
        if exp_arr.dtype == np.bool_ or obs_arr.dtype == np.bool_:
            if not (exp_arr.dtype == np.bool_ and obs_arr.dtype == np.bool_ and np.array_equal(exp_arr, obs_arr)):
                mismatches.append(f"Bool array mismatch at {path}")
            return
        if np.issubdtype(exp_arr.dtype, np.integer) and np.issubdtype(obs_arr.dtype, np.integer):
            if not np.array_equal(exp_arr, obs_arr):
                mismatches.append(f"Int array mismatch at {path}")
            return
        if np.issubdtype(exp_arr.dtype, np.number) and np.issubdtype(obs_arr.dtype, np.number):
            if not (np.isfinite(exp_arr).all() and np.isfinite(obs_arr).all()):
                mismatches.append(f"Non-finite numeric array value at {path}")
                return
            diff = float(np.max(np.abs(exp_arr.astype(np.complex128) - obs_arr.astype(np.complex128)))) if exp_arr.size else 0.0
            stats["max_float_diff"] = max(stats["max_float_diff"], diff)
            if diff > float_tol:
                mismatches.append(f"Array numeric mismatch at {path}: max_diff={diff:.3e} > tol={float_tol:.3e}")
            return
        if not np.array_equal(exp_arr, obs_arr):
            mismatches.append(f"Array value mismatch at {path}")
        return

    if isinstance(expected, (list, tuple)) or isinstance(observed, (list, tuple)):
        if not (isinstance(expected, (list, tuple)) and isinstance(observed, (list, tuple))):
            mismatches.append(f"Sequence type mismatch at {path}: {type(expected).__name__} vs {type(observed).__name__}")
            return
        if len(expected) != len(observed):
            mismatches.append(f"Sequence length mismatch at {path}: {len(expected)} vs {len(observed)}")
            return
        for idx, (e_val, o_val) in enumerate(zip(expected, observed)):
            _compare_replay_values(e_val, o_val, f"{path}[{idx}]", float_tol, mismatches, ignored_ms_fields, stats)
        return

    if isinstance(expected, (bool, np.bool_)) or isinstance(observed, (bool, np.bool_)):
        if not (isinstance(expected, (bool, np.bool_)) and isinstance(observed, (bool, np.bool_))
                and bool(expected) == bool(observed)):
            mismatches.append(f"Bool mismatch at {path}: {expected!r} vs {observed!r}")
        return

    if isinstance(expected, (int, np.integer)) and isinstance(observed, (int, np.integer)):
        if int(expected) != int(observed):
            mismatches.append(f"Int mismatch at {path}: {int(expected)} vs {int(observed)}")
        return

    if isinstance(expected, (complex, np.complexfloating)) or isinstance(observed, (complex, np.complexfloating)):
        if not (isinstance(expected, (int, float, complex, np.number))
                and isinstance(observed, (int, float, complex, np.number))):
            mismatches.append(f"Complex type mismatch at {path}: {type(expected).__name__} vs {type(observed).__name__}")
            return
        ec = complex(expected)
        oc = complex(observed)
        if not (math.isfinite(ec.real) and math.isfinite(ec.imag) and math.isfinite(oc.real) and math.isfinite(oc.imag)):
            mismatches.append(f"Non-finite complex value at {path}: {ec!r} vs {oc!r}")
            return
        diff = abs(ec - oc)
        stats["max_float_diff"] = max(stats["max_float_diff"], diff)
        if diff > float_tol:
            mismatches.append(f"Complex mismatch at {path}: {ec!r} vs {oc!r} (diff={diff:.3e} > tol={float_tol:.3e})")
        return

    if isinstance(expected, (int, float, np.floating, np.integer)) and isinstance(observed, (int, float, np.floating, np.integer)):
        if isinstance(expected, (int, np.integer)) != isinstance(observed, (int, np.integer)):
            mismatches.append(f"Numeric type mismatch at {path}: {type(expected).__name__} vs {type(observed).__name__}")
            return
        ef = float(expected)
        of = float(observed)
        if not (math.isfinite(ef) and math.isfinite(of)):
            mismatches.append(f"Non-finite float value at {path}: {ef!r} vs {of!r}")
            return
        diff = abs(ef - of)
        stats["max_float_diff"] = max(stats["max_float_diff"], diff)
        if diff > float_tol:
            mismatches.append(f"Float mismatch at {path}: {ef!r} vs {of!r} (diff={diff:.3e} > tol={float_tol:.3e})")
        return

    if type(expected) is not type(observed) or expected != observed:
        mismatches.append(f"Value mismatch at {path}: {expected!r} vs {observed!r}")


def verify_replay_exact(
    orig_records: Sequence[Mapping[str, Any]] | Path,
    replay_records: Sequence[Mapping[str, Any]] | Path,
    float_tol: float = 1e-12,
    *,
    key_fields: Optional[Sequence[str]] = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Generic exact field-level replay verifier ignoring ONLY wall-clock `*_ms` fields."""
    tol = float(float_tol)
    require(math.isfinite(tol) and tol >= 0.0, f"float_tol must be finite and >= 0, got {float_tol!r}")

    orig_list = _load_records_arg(orig_records)
    rep_list = _load_records_arg(replay_records)
    mismatches: list[str] = []
    ignored_ms_fields: set[str] = set()
    stats: dict[str, Any] = {"max_float_diff": 0.0, "ignored_ms_count": 0}
    matched_records = 0

    if len(orig_list) == 0:
        mismatches.append("orig_records is empty")
    if len(rep_list) == 0:
        mismatches.append("replay_records is empty")

    resolved_keys: Optional[tuple[str, ...]] = None
    if not mismatches:
        resolved_keys = _infer_key_fields(orig_list, rep_list, key_fields)
        if resolved_keys is not None:
            orig_by_key: dict[tuple[Any, ...], Mapping[str, Any]] = {}
            for idx, rec in enumerate(orig_list):
                if not all(k in rec for k in resolved_keys):
                    mismatches.append(f"orig[{idx}] missing key field(s) {resolved_keys}")
                    continue
                k_val = tuple(rec[k] for k in resolved_keys)
                if k_val in orig_by_key:
                    mismatches.append(f"Duplicate key {k_val} in orig_records")
                    continue
                orig_by_key[k_val] = rec

            seen_rep_keys: set[tuple[Any, ...]] = set()
            for idx, rep_rec in enumerate(rep_list):
                if not all(k in rep_rec for k in resolved_keys):
                    mismatches.append(f"replay[{idx}] missing key field(s) {resolved_keys}")
                    continue
                k_val = tuple(rep_rec[k] for k in resolved_keys)
                if k_val in seen_rep_keys:
                    mismatches.append(f"Duplicate replay key {k_val}")
                    continue
                seen_rep_keys.add(k_val)
                if k_val not in orig_by_key:
                    mismatches.append(f"Replay key {k_val} not found in orig_records")
                    continue
                matched_records += 1
                _compare_replay_values(
                    orig_by_key[k_val],
                    rep_rec,
                    f"record{k_val}",
                    tol,
                    mismatches,
                    ignored_ms_fields,
                    stats,
                )
        else:
            if len(orig_list) != len(rep_list):
                mismatches.append(
                    f"Cannot positionally align replay without key fields: len(orig)={len(orig_list)} vs len(replay)={len(rep_list)}"
                )
            else:
                for idx, (o_rec, r_rec) in enumerate(zip(orig_list, rep_list)):
                    matched_records += 1
                    _compare_replay_values(
                        o_rec, r_rec, f"record[{idx}]", tol, mismatches, ignored_ms_fields, stats
                    )

    report = {
        "passed": len(mismatches) == 0,
        "key_fields": list(resolved_keys) if resolved_keys is not None else None,
        "n_orig": len(orig_list),
        "n_replay": len(rep_list),
        "matched_records": int(matched_records),
        "float_tol": tol,
        "max_float_diff": float(stats["max_float_diff"]),
        "ignored_ms_count": int(stats["ignored_ms_count"]),
        "ignored_ms_fields": sorted(ignored_ms_fields),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }
    if strict and mismatches:
        raise ValueError(f"Replay verification failed ({len(mismatches)} mismatches): {mismatches[:5]}")
    return report


# ---------------------------------------------------------------------------
# 4. Raw-JSONL Cohort Auditors (P11 Primary 11+5 Pooling & Secondary Off-Grid)
# ---------------------------------------------------------------------------

def _check_common_cell_fields(
    cell: Mapping[str, Any],
    *,
    expected_blocks: int,
    expected_extra: int,
    expected_probes: int,
) -> None:
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


def check_p11_primary_cell(cell: Mapping[str, Any], budget: str) -> None:
    """Strictly verify one P11 Primary cell (16 core blocks via 11+5 owner-row pooling, Wave-1, injected-word FA)."""
    require(budget in ("sparse", "dense"), f"Unsupported Primary budget {budget!r}")
    expected_extra = 2 if budget == "sparse" else 0
    _check_common_cell_fields(cell, expected_blocks=16, expected_extra=expected_extra, expected_probes=32)

    fold_keys = ("fold0_blocks", "fold1_blocks", "fold0_errors", "fold1_errors")
    if any(k in cell for k in fold_keys):
        for k in fold_keys:
            require(k in cell and type(cell[k]) is int, f"Incomplete/non-int fold field {k!r}")
        require(cell["fold0_blocks"] == 11, f"fold0_blocks must be 11, got {cell['fold0_blocks']}")
        require(cell["fold1_blocks"] == 5, f"fold1_blocks must be 5, got {cell['fold1_blocks']}")
        require(0 <= cell["fold0_errors"] <= 11, f"Invalid fold0_errors={cell['fold0_errors']}")
        require(0 <= cell["fold1_errors"] <= 5, f"Invalid fold1_errors={cell['fold1_errors']}")
        require(
            cell["fold0_errors"] + cell["fold1_errors"] == cell["block_errors"],
            f"Owner-row pooling mismatch: {cell['fold0_errors']} + {cell['fold1_errors']} != {cell['block_errors']}",
        )

    if "total_assimilations" in cell or "false_assimilations" in cell:
        require("total_assimilations" in cell and "false_assimilations" in cell,
                "Both total_assimilations and false_assimilations must be provided together")
        tot_a = cell["total_assimilations"]
        false_a = cell["false_assimilations"]
        require(type(tot_a) is int and tot_a >= 0, f"Invalid total_assimilations={tot_a!r}")
        require(type(false_a) is int and 0 <= false_a <= tot_a, f"Invalid false_assimilations={false_a!r} vs {tot_a}")

    valid_w1 = ("correct", "incorrect", "absent")
    if "wave1_status" in cell:
        require(cell["wave1_status"] in valid_w1, f"Invalid wave1_status={cell['wave1_status']!r}")
    for fk in ("wave1_fold0", "wave1_fold1"):
        if fk in cell:
            require(cell[fk] in valid_w1, f"Invalid {fk}={cell[fk]!r}")

    if "oracle_common_phase_errors" in cell and cell["oracle_common_phase_errors"] is not None:
        oce = cell["oracle_common_phase_errors"]
        require(type(oce) is int and 0 <= oce <= 16, f"Invalid oracle_common_phase_errors={oce!r}")


def check_p11_offgrid_cell(
    cell: Mapping[str, Any],
    *,
    expected_blocks: int = 18,
    expected_probes: int = 32,
) -> None:
    """Strictly verify one P11 Secondary Off-Grid cell (18 data blocks/link, extra_blocks=0)."""
    _check_common_cell_fields(
        cell,
        expected_blocks=expected_blocks,
        expected_extra=0,
        expected_probes=expected_probes,
    )
    if "native_block_errors" in cell:
        nbe = cell["native_block_errors"]
        require(type(nbe) is int and 0 <= nbe <= expected_blocks, f"Invalid native_block_errors={nbe!r}")
    if "native_info_errors" in cell:
        nie = cell["native_info_errors"]
        require(type(nie) is int and 0 <= nie <= 64 * expected_blocks, f"Invalid native_info_errors={nie!r}")


def aggregate_p11_cells(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Pool integer counts and float sums before dividing (strict 11+5 owner-row pooling)."""
    require(len(cells) > 0, "Cannot aggregate empty cell list")
    n_links = len(cells)
    sums: dict[str, Any] = {}

    for key in ADDITIVE_INT_FIELDS:
        sums[key] = int(sum(int(c[key]) for c in cells))
    for key in ADDITIVE_FLOAT_FIELDS:
        sums[key] = float(math.fsum(float(c[key]) for c in cells))

    core_delivered = int(sums["blocks"] - sums["block_errors"])
    extra_delivered = int(sums["extra_blocks"] - sums["extra_errors"])
    total_delivered_blocks = int(core_delivered + extra_delivered)
    bits_per_block = float(sums["info_bits"]) / float(sums["blocks"])
    net_delivered_info_bits_per_frame = (float(total_delivered_blocks) * bits_per_block) / float(n_links)

    out = dict(sums)
    out.update({
        "links": n_links,
        "bler": float(sums["block_errors"]) / float(sums["blocks"]),
        "coded_ber": float(sums["coded_errors"]) / float(sums["coded_bits"]),
        "info_ber": float(sums["info_errors"]) / float(sums["info_bits"]),
        "probe_nmse": float(sums["probe_sse"]) / float(sums["probe_energy"]),
        "clean_nmse": float(sums["clean_sse"]) / float(sums["clean_energy"]),
        "nll": float(sums["nll_sum"]) / float(sums["probes"]),
        "calibration": float(sums["probe_sse"]) / float(sums["variance_sum"]),
        "core_delivered": core_delivered,
        "extra_delivered": extra_delivered,
        "total_delivered_blocks": total_delivered_blocks,
        "net_delivered_info_bits_per_frame": net_delivered_info_bits_per_frame,
    })

    if all(k in c for c in cells for k in ("fold0_blocks", "fold1_blocks", "fold0_errors", "fold1_errors")):
        f0_b = int(sum(int(c["fold0_blocks"]) for c in cells))
        f1_b = int(sum(int(c["fold1_blocks"]) for c in cells))
        f0_e = int(sum(int(c["fold0_errors"]) for c in cells))
        f1_e = int(sum(int(c["fold1_errors"]) for c in cells))
        out["fold0_blocks"] = f0_b
        out["fold1_blocks"] = f1_b
        out["fold0_errors"] = f0_e
        out["fold1_errors"] = f1_e
        out["fold0_bler"] = float(f0_e) / float(f0_b) if f0_b > 0 else 0.0
        out["fold1_bler"] = float(f1_e) / float(f1_b) if f1_b > 0 else 0.0

    if all("total_assimilations" in c and "false_assimilations" in c for c in cells):
        tot_a = int(sum(int(c["total_assimilations"]) for c in cells))
        false_a = int(sum(int(c["false_assimilations"]) for c in cells))
        out["total_assimilations"] = tot_a
        out["false_assimilations"] = false_a
        out["false_assimilation_rate"] = float(false_a) / float(tot_a) if tot_a > 0 else 0.0

    if all("native_block_errors" in c for c in cells):
        nbe = int(sum(int(c["native_block_errors"]) for c in cells))
        out["native_block_errors"] = nbe
        out["native_bler"] = float(nbe) / float(sums["blocks"])
    if all("native_info_errors" in c for c in cells):
        nie = int(sum(int(c["native_info_errors"]) for c in cells))
        out["native_info_errors"] = nie
        out["native_info_ber"] = float(nie) / float(sums["info_bits"])

    return out


def _verify_table_matches_recomputed(
    recomputed: Mapping[str, Any],
    reported: Mapping[str, Any],
    label: str,
    *,
    tol: float = 2e-12,
) -> float:
    require(isinstance(reported, Mapping), f"Reported table {label} must be a mapping, got {type(reported).__name__}")
    expected_keys = {k for k in recomputed if not _is_wallclock_ms_key(k)}
    missing_keys = sorted(expected_keys - set(reported.keys()))
    require(len(missing_keys) == 0, f"Reported table {label} missing required keys: {missing_keys}")

    max_diff = 0.0
    for key, rep_val in reported.items():
        if _is_wallclock_ms_key(key):
            continue
        if key not in recomputed:
            raise ValueError(f"Reported key {label}/{key} not produced by independent recomputation")
        if isinstance(rep_val, (bool, np.bool_)):
            raise TypeError(f"Reported key {label}/{key} must not be bool: {rep_val!r}")
        rec_val = recomputed[key]
        if type(rec_val) is int and type(rep_val) is int:
            require(rec_val == rep_val, f"Integer aggregate mismatch {label}/{key}: {rep_val} vs {rec_val}")
        elif type(rec_val) is int:
            raise ValueError(f"Integer aggregate type mismatch {label}/{key}: got {type(rep_val).__name__}")
        else:
            if not isinstance(rep_val, (int, float, np.integer, np.floating)):
                raise TypeError(f"Reported key {label}/{key} must be numeric, got {type(rep_val).__name__}")
            rep_f = float(rep_val)
            rec_f = float(rec_val)
            require(math.isfinite(rep_f) and math.isfinite(rec_f), f"Non-finite aggregate {label}/{key}")
            diff = abs(rep_f - rec_f)
            require(math.isclose(rep_f, rec_f, rel_tol=tol, abs_tol=tol),
                    f"Aggregate mismatch {label}/{key}: reported={rep_val!r} vs recomputed={rec_val!r}")
            max_diff = max(max_diff, diff)
    return max_diff


def _aggregate_nested_budget_arm(
    records: Sequence[Mapping[str, Any]],
    budgets: Sequence[str],
    arms_by_budget: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for budget in budgets:
        out[budget] = {}
        for arm in arms_by_budget[budget]:
            cells = [r["budgets"][budget][arm] for r in records]
            out[budget][arm] = aggregate_p11_cells(cells)
    return out


def summarize_common_phase_blindspot(
    records: Sequence[Mapping[str, Any]],
    *,
    target_arm: str = "p11_chained_wavefront",
    payload_arm: str = "p11_payload_only",
    budget: str = "sparse",
) -> dict[str, Any]:
    """Compute Wave-1 unlock breakdown, conditional target BLER, injected-word FA, & 9e9d93b comparison."""
    require(len(records) > 0, "Cannot summarize common-phase blindspot on empty records")

    # Fallback gracefully if caller named standalone payload arm p11_standalone_rebuild
    first_arms = records[0]["budgets"][budget]
    resolved_payload_arm: Optional[str] = None
    if payload_arm in first_arms:
        resolved_payload_arm = payload_arm
    elif "p11_standalone_rebuild" in first_arms:
        resolved_payload_arm = "p11_standalone_rebuild"

    ref_by_regime = {
        "continuous": float(P10_9E9D93B_REFERENCE["continuous_bler"]),
        "partial_shift": float(P10_9E9D93B_REFERENCE["partial_shift_bler"]),
        "full_reset": float(P10_9E9D93B_REFERENCE["full_reset_bler"]),
    }

    def _summarize_subset(sub_recs: Sequence[Mapping[str, Any]], ref_bler: Optional[float]) -> dict[str, Any]:
        target_cells = [r["budgets"][budget][target_arm] for r in sub_recs]
        agg_t = aggregate_p11_cells(target_cells)
        res: dict[str, Any] = {
            "links": len(target_cells),
            "blocks": int(agg_t["blocks"]),
            "block_errors": int(agg_t["block_errors"]),
            "target_bler": float(agg_t["bler"]),
        }
        if resolved_payload_arm is not None:
            agg_p = aggregate_p11_cells([r["budgets"][budget][resolved_payload_arm] for r in sub_recs])
            res["payload_arm"] = resolved_payload_arm
            res["payload_bler"] = float(agg_p["bler"])
            res["delta_vs_payload_pp"] = 100.0 * (float(agg_p["bler"]) - float(agg_t["bler"]))
        if ref_bler is not None:
            res["p10_9e9d93b_bler"] = ref_bler
            res["delta_vs_p10_9e9d93b_pp"] = 100.0 * (ref_bler - float(agg_t["bler"]))
        if "false_assimilation_rate" in agg_t:
            res["total_assimilations"] = int(agg_t["total_assimilations"])
            res["false_assimilations"] = int(agg_t["false_assimilations"])
            res["false_assimilation_rate"] = float(agg_t["false_assimilation_rate"])

        # Wave-1 conditional BLER breakdown (fold-level if wave1_fold0/fold1 present, else link-level wave1_status)
        strata = {k: {"units": 0, "blocks": 0, "errors": 0} for k in ("correct", "incorrect", "absent")}
        has_fold_w1 = all("wave1_fold0" in c and "wave1_fold1" in c and "fold0_errors" in c for c in target_cells)
        has_link_w1 = all("wave1_status" in c for c in target_cells)
        if has_fold_w1:
            for c in target_cells:
                for fk, bk, ek in (("wave1_fold0", "fold0_blocks", "fold0_errors"), ("wave1_fold1", "fold1_blocks", "fold1_errors")):
                    st = c[fk]
                    strata[st]["units"] += 1
                    strata[st]["blocks"] += int(c[bk])
                    strata[st]["errors"] += int(c[ek])
            total_units = 2 * len(target_cells)
        elif has_link_w1:
            for c in target_cells:
                st = c["wave1_status"]
                strata[st]["units"] += 1
                strata[st]["blocks"] += int(c["blocks"])
                strata[st]["errors"] += int(c["block_errors"])
            total_units = len(target_cells)
        else:
            total_units = 0

        if total_units > 0:
            res["wave1_breakdown"] = {
                st: {
                    "units": strata[st]["units"],
                    "fraction": float(strata[st]["units"]) / float(total_units),
                    "blocks": strata[st]["blocks"],
                    "errors": strata[st]["errors"],
                    "conditional_bler": (
                        float(strata[st]["errors"]) / float(strata[st]["blocks"])
                        if strata[st]["blocks"] > 0
                        else None
                    ),
                }
                for st in ("correct", "incorrect", "absent")
            }

        oracle_cells = [c for c in target_cells if c.get("oracle_common_phase_errors") is not None]
        if oracle_cells:
            o_blks = int(sum(int(c["blocks"]) for c in oracle_cells))
            o_err = int(sum(int(c["oracle_common_phase_errors"]) for c in oracle_cells))
            t_err_sub = int(sum(int(c["block_errors"]) for c in oracle_cells))
            res["oracle_common_phase_subset"] = {
                "links": len(oracle_cells),
                "blocks": o_blks,
                "target_errors": t_err_sub,
                "oracle_errors": o_err,
                "target_bler": float(t_err_sub) / float(o_blks),
                "oracle_bler": float(o_err) / float(o_blks),
                "residual_common_phase_gap_pp": 100.0 * (float(t_err_sub - o_err) / float(o_blks)),
            }
        return res

    regimes_present = sorted({r.get("regime", "unknown") for r in records})
    by_regime = {
        reg: _summarize_subset([r for r in records if r.get("regime") == reg], ref_by_regime.get(reg))
        for reg in regimes_present
    }
    id_recs = [r for r in records if r.get("regime") in REGIMES_ID]
    pooled_recs = id_recs if len(id_recs) == len(records) or len(id_recs) > 0 else list(records)
    pooled_ref = float(P10_9E9D93B_REFERENCE["3id_code_gate_bler"]) if set(REGIMES_ID).issubset(set(regimes_present)) else None

    return {
        "target_arm": target_arm,
        "payload_arm": resolved_payload_arm,
        "p10_9e9d93b_reference": P10_9E9D93B_REFERENCE,
        "pooled_3id": _summarize_subset(pooled_recs, pooled_ref),
        "by_regime": by_regime,
    }


def _extract_arm_spec(spec: Mapping[str, Any], prefix: str) -> tuple[str, str]:
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


def _audit_cohort_common(
    raw_path: Path,
    summary_path: Path,
    replay_path: Optional[Path],
    *,
    mode: str,
    allowed_regimes: tuple[str, ...],
    seed_pools: Optional[Mapping[str, Sequence[int]]],
    replay_float_tol: float,
    require_canonical_family: bool,
) -> dict[str, Any]:
    raw_bytes = raw_path.read_bytes()
    raw_lines = raw_bytes.splitlines()
    require(len(raw_lines) > 0, "Raw JSONL is empty")
    records = [json.loads(line) for line in raw_lines]

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
            require(rec["regime"] in allowed_regimes, f"Unknown {mode} regime {rec['regime']!r} on record {idx}")

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
                if mode == "primary":
                    check_p11_primary_cell(cell, budget)
                else:
                    check_p11_offgrid_cell(cell)
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
        require(len(c_recs) == links_per_cluster, f"Unbalanced cluster {c_id}: {len(c_recs)} vs {links_per_cluster}")
        link_ids = sorted(r["link"] for r in c_recs)
        expected_links = sorted(range(links_per_cluster))
        # Allow per-regime link numbering [0..n_per_reg-1] OR flat [0..links_per_cluster-1]
        if link_ids != expected_links:
            reg_counts: dict[str, list[int]] = {}
            for r in c_recs:
                reg_counts.setdefault(str(r.get("regime", "")), []).append(int(r["link"]))
            for reg_name, l_ids in reg_counts.items():
                require(sorted(l_ids) == list(range(len(l_ids))),
                        f"Cluster {c_id} regime {reg_name!r} link IDs not contiguous: {l_ids}")

    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    require(isinstance(summary, Mapping), f"Summary JSON must be a mapping, got {type(summary).__name__}")
    require(isinstance(summary.get("totals"), Mapping), "Summary missing valid 'totals' mapping")
    require(
        isinstance(summary.get("clusters"), Sequence) and not isinstance(summary.get("clusters"), (str, bytes)),
        "Summary missing valid 'clusters' list",
    )
    if "by_regime" in summary and summary["by_regime"] is not None:
        require(isinstance(summary["by_regime"], Mapping), "Summary 'by_regime' must be a mapping")

    effective_pools = seed_pools if seed_pools is not None else summary.get("seed_pools")
    seed_audit: Optional[dict[str, Any]] = None
    if effective_pools is not None:
        seed_audit = check_p11_seed_disjointness(
            effective_pools["train"],
            effective_pools["dev"],
            effective_pools["formal_id"],
            effective_pools["formal_ood"],
            effective_pools.get("e2e", ()),
            strict=True,
        )
        train_dev = set(int(s) for s in effective_pools["train"]) | set(int(s) for s in effective_pools["dev"])
        leak_overlap = sorted(seen_seeds & train_dev)
        require(len(leak_overlap) == 0, f"Formal cohort intersects train/dev seeds: {leak_overlap[:10]}")

    max_agg_diff = 0.0
    totals_recomp = _aggregate_nested_budget_arm(records, budgets_ref, arms_by_budget_ref)
    for budget in budgets_ref:
        require(budget in summary["totals"], f"Summary totals missing budget {budget!r}")
        require(isinstance(summary["totals"][budget], Mapping), f"Summary totals[{budget!r}] must be a mapping")
        for arm in arms_by_budget_ref[budget]:
            require(arm in summary["totals"][budget], f"Summary totals missing {budget}/{arm}")
            err = _verify_table_matches_recomputed(
                totals_recomp[budget][arm],
                summary["totals"][budget][arm],
                f"totals/{budget}/{arm}",
            )
            max_agg_diff = max(max_agg_diff, err)

    require(len(summary["clusters"]) == g, f"Summary clusters length {len(summary['clusters'])} != G={g}")
    cluster_recomps: list[dict[str, Any]] = []
    for c_idx, c_id in enumerate(cluster_ids):
        c_agg = _aggregate_nested_budget_arm(clusters_seen[c_id], budgets_ref, arms_by_budget_ref)
        cluster_recomps.append(c_agg)
        rep_c = summary["clusters"][c_idx]
        require(isinstance(rep_c, Mapping), f"Summary clusters[{c_idx}] must be a mapping")
        for budget in budgets_ref:
            require(budget in rep_c and isinstance(rep_c[budget], Mapping),
                    f"Summary clusters[{c_idx}] missing budget mapping {budget!r}")
            for arm in arms_by_budget_ref[budget]:
                require(arm in rep_c[budget], f"Summary clusters[{c_idx}][{budget!r}] missing {arm!r}")
                err = _verify_table_matches_recomputed(
                    c_agg[budget][arm],
                    rep_c[budget][arm],
                    f"cluster[{c_id}]/{budget}/{arm}",
                )
                max_agg_diff = max(max_agg_diff, err)

    regime_recomps: dict[str, Any] = {}
    if "by_regime" in summary and summary["by_regime"] is not None:
        regimes_in_raw = sorted({r["regime"] for r in records if "regime" in r})
        require(set(summary["by_regime"].keys()) == set(regimes_in_raw),
                f"Summary by_regime keys {set(summary['by_regime'])} != raw regimes {set(regimes_in_raw)}")
        for reg in regimes_in_raw:
            reg_recs = [r for r in records if r.get("regime") == reg]
            reg_agg = _aggregate_nested_budget_arm(reg_recs, budgets_ref, arms_by_budget_ref)
            regime_recomps[reg] = reg_agg
            rep_reg = summary["by_regime"][reg]
            require(isinstance(rep_reg, Mapping), f"Summary by_regime[{reg!r}] must be a mapping")
            for budget in budgets_ref:
                require(budget in rep_reg and isinstance(rep_reg[budget], Mapping),
                        f"Summary by_regime[{reg!r}] missing budget mapping {budget!r}")
                for arm in arms_by_budget_ref[budget]:
                    require(arm in rep_reg[budget], f"Summary by_regime[{reg!r}][{budget!r}] missing {arm!r}")
                    err = _verify_table_matches_recomputed(
                        reg_agg[budget][arm],
                        rep_reg[budget][arm],
                        f"by_regime/{reg}/{budget}/{arm}",
                    )
                    max_agg_diff = max(max_agg_diff, err)

    id_regimes = REGIMES_ID if mode == "primary" else OFFGRID_REGIMES_ID
    ood_regimes = REGIMES_OOD if mode == "primary" else OFFGRID_REGIMES_OOD
    for eq_key, reg_tuple in (("id_equal", id_regimes), ("ood_equal", ood_regimes)):
        if eq_key in summary and summary[eq_key] is not None:
            eq_recs = [r for r in records if r.get("regime") in reg_tuple]
            require(len(eq_recs) > 0, f"Summary contains {eq_key} but raw JSONL has zero matching regime records")
            eq_recomp = _aggregate_nested_budget_arm(eq_recs, budgets_ref, arms_by_budget_ref)
            rep_eq = summary[eq_key]
            require(
                isinstance(rep_eq, Mapping) and set(rep_eq.keys()) == set(budgets_ref),
                f"Summary {eq_key} budgets {set(rep_eq.keys()) if isinstance(rep_eq, Mapping) else type(rep_eq).__name__} != {set(budgets_ref)}",
            )
            for budget in budgets_ref:
                require(
                    isinstance(rep_eq[budget], Mapping)
                    and set(rep_eq[budget].keys()) == set(arms_by_budget_ref[budget]),
                    f"Summary {eq_key}/{budget} arms mismatch",
                )
                for arm in arms_by_budget_ref[budget]:
                    err = _verify_table_matches_recomputed(
                        eq_recomp[budget][arm],
                        rep_eq[budget][arm],
                        f"{eq_key}/{budget}/{arm}",
                    )
                    max_agg_diff = max(max_agg_diff, err)

    def cluster_metric_vector(budget: str, arm: str, metric: str, regimes: Optional[Sequence[str]] = None) -> np.ndarray:
        vals = []
        reg_set = set(regimes) if regimes is not None else None
        for c_id in cluster_ids:
            c_recs = clusters_seen[c_id]
            if reg_set is not None:
                c_recs = [r for r in c_recs if r.get("regime") in reg_set]
            require(len(c_recs) > 0, f"Cluster {c_id} has zero records after regime_filter={regimes}")
            agg = aggregate_p11_cells([r["budgets"][budget][arm] for r in c_recs])
            require(metric in agg, f"Unknown metric {metric!r}")
            vals.append(float(agg[metric]))
        return np.asarray(vals, dtype=np.float64)

    max_stat_diff = 0.0

    def _verify_holm_tests_block(
        rep_tests: Any,
        family_name: str,
        *,
        canonical_names: Optional[tuple[str, ...]] = None,
        canonical_arms: Optional[Mapping[str, tuple[tuple[str, str], tuple[str, str]]]] = None,
        canonical_regimes: Optional[Sequence[str]] = None,
        fallback_arms: Optional[Mapping[str, tuple[tuple[str, str], tuple[str, str]]]] = None,
        default_regimes: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        nonlocal max_stat_diff
        require(
            isinstance(rep_tests, Sequence) and not isinstance(rep_tests, (str, bytes)) and len(rep_tests) >= 1,
            f"Summary {family_name} tests must be a non-empty list",
        )
        for idx_t, rep_t in enumerate(rep_tests):
            require(
                isinstance(rep_t, Mapping) and "name" in rep_t,
                f"Invalid test entry {idx_t} in {family_name}: {rep_t!r}",
            )
        if canonical_names is not None:
            names = [t.get("name") for t in rep_tests]
            require(
                set(canonical_names).issubset(set(names)),
                f"Holm family {family_name} names {names!r} missing canonical {canonical_names!r}",
            )

        contrast_inputs: list[dict[str, Any]] = []
        for rep_t in rep_tests:
            cname = str(rep_t["name"])
            has_explicit_arms = (
                ("baseline" in rep_t or ("baseline_budget" in rep_t and "baseline_arm" in rep_t))
                and ("candidate" in rep_t or ("candidate_budget" in rep_t and "candidate_arm" in rep_t))
            )
            if has_explicit_arms:
                base_b, base_a = _extract_arm_spec(rep_t, "baseline")
                cand_b, cand_a = _extract_arm_spec(rep_t, "candidate")
            elif fallback_arms is not None and cname in fallback_arms:
                (base_b, base_a), (cand_b, cand_a) = fallback_arms[cname]
            else:
                raise ValueError(f"Cannot resolve baseline/candidate arms for contrast {cname!r} in {family_name}")

            raw_metric = rep_t.get("metric", "bler")
            require(
                isinstance(raw_metric, str) and bool(raw_metric),
                f"Contrast {cname!r} metric must be a non-empty string, got {raw_metric!r}",
            )
            metric = raw_metric
            reg_filter = rep_t.get("regimes")
            if reg_filter is not None:
                require(
                    isinstance(reg_filter, Sequence) and not isinstance(reg_filter, (str, bytes)),
                    f"Contrast {cname!r} regimes must be a sequence, got {type(reg_filter).__name__}",
                )
            if default_regimes is not None:
                if reg_filter is None:
                    reg_filter = list(default_regimes)
                else:
                    require(
                        set(reg_filter) == set(default_regimes),
                        f"Contrast {cname!r} regimes {reg_filter!r} != expected {default_regimes!r}",
                    )
            raw_sign = rep_t.get("expected_sign", 1)
            require(
                not isinstance(raw_sign, (bool, np.bool_))
                and isinstance(raw_sign, (int, np.integer))
                and int(raw_sign) in (-1, 1),
                f"Contrast {cname!r} expected_sign must be -1 or 1, got {raw_sign!r}",
            )
            expected_sign = int(raw_sign)
            if canonical_arms is not None and cname in canonical_arms:
                exp_base, exp_cand = canonical_arms[cname]
                require(
                    (base_b, base_a) == exp_base,
                    f"Canonical contrast {family_name}/{cname} baseline {(base_b, base_a)!r} != expected {exp_base!r}",
                )
                require(
                    (cand_b, cand_a) == exp_cand,
                    f"Canonical contrast {family_name}/{cname} candidate {(cand_b, cand_a)!r} != expected {exp_cand!r}",
                )
                require(
                    metric == "bler",
                    f"Canonical contrast {family_name}/{cname} metric {metric!r} != 'bler'",
                )
                require(
                    expected_sign == 1,
                    f"Canonical contrast {family_name}/{cname} expected_sign {expected_sign} != 1",
                )
                if canonical_regimes is not None and reg_filter is not None:
                    require(
                        set(reg_filter) == set(canonical_regimes),
                        f"Canonical contrast {family_name}/{cname} regimes {reg_filter!r} != expected {list(canonical_regimes)!r}",
                    )
            base_vec = cluster_metric_vector(base_b, base_a, metric, reg_filter)
            cand_vec = cluster_metric_vector(cand_b, cand_a, metric, reg_filter)
            item: dict[str, Any] = {
                "name": cname,
                "baseline": base_vec,
                "candidate": cand_vec,
                "expected_sign": expected_sign,
                "metric": metric,
                "baseline_ref": {"budget": base_b, "arm": base_a},
                "candidate_ref": {"budget": cand_b, "arm": cand_a},
            }
            if reg_filter is not None:
                item["regimes"] = list(reg_filter)
            contrast_inputs.append(item)

        family_recomp = _audit_single_holm_family(contrast_inputs, family_name, alpha=Fraction(1, 20), expected_g=g)
        holm_recomp_list = family_recomp["contrasts"]

        for rep_t, rec_t in zip(rep_tests, holm_recomp_list):
            require(type(rep_t.get("n")) is int and rep_t["n"] == g,
                    f"Contrast {rec_t['name']} n={rep_t.get('n')!r} != G={g}")
            if "df" in rep_t:
                require(type(rep_t["df"]) is int and rep_t["df"] == g - 1,
                        f"Contrast {rec_t['name']} df={rep_t['df']!r} != G-1={g - 1}")
            require(
                all(type(rep_t.get(k)) is int for k in ("wins", "ties", "losses"))
                and (rep_t["wins"], rep_t["ties"], rep_t["losses"]) == (rec_t["wins"], rec_t["ties"], rec_t["losses"]),
                f"Sign counts disagree on {rec_t['name']}",
            )
            require(
                Fraction(rep_t["raw_exact"]) == Fraction(rec_t["raw_exact"])
                and not isinstance(rep_t["raw_p"], (bool, np.bool_))
                and rep_t["raw_p"] == rec_t["raw_p"],
                f"Raw exact p mismatch on {rec_t['name']}",
            )
            require(
                Fraction(rep_t["holm_exact"]) == Fraction(rec_t["holm_exact"])
                and not isinstance(rep_t["holm_p"], (bool, np.bool_))
                and rep_t["holm_p"] == rec_t["holm_p"],
                f"Holm exact p mismatch on {rec_t['name']}",
            )
            if "verdict" in rep_t:
                require(rep_t["verdict"] == rec_t["verdict"], f"verdict mismatch on {rec_t['name']}")
            if "verdict_zh" in rep_t:
                require(rep_t["verdict_zh"] == rec_t["verdict_zh"], f"verdict_zh mismatch on {rec_t['name']}")
            rep_diffs = _strict_1d_float_array(rep_t.get("differences"), g, f"{family_name}/{rec_t['name']}.differences")
            rep_ci95 = _strict_1d_float_array(rep_t.get("ci95"), 2, f"{family_name}/{rec_t['name']}.ci95")
            require(not isinstance(rep_t.get("mean_diff"), (bool, np.bool_)),
                    f"{family_name}/{rec_t['name']}.mean_diff must not be bool")
            diff_err = float(np.max(np.abs(rep_diffs - np.asarray(rec_t["differences"], dtype=np.float64))))
            mean_err = abs(float(rep_t["mean_diff"]) - rec_t["mean_diff"])
            ci_err = float(np.max(np.abs(rep_ci95 - np.asarray(rec_t["ci95"], dtype=np.float64))))
            require(
                diff_err <= 2e-12 and mean_err <= 2e-12 and ci_err <= 1e-10,
                f"Stat mismatch on {rec_t['name']}: diff={diff_err}, mean={mean_err}, ci={ci_err}",
            )
            max_stat_diff = max(max_stat_diff, diff_err, mean_err, ci_err)

        return family_recomp

    canonical_main = (
        (P11_PRIMARY_CANONICAL_CONTRASTS if mode == "primary" else P11_SECONDARY_CANONICAL_CONTRASTS)
        if require_canonical_family
        else None
    )
    canonical_main_arms = (
        (_P11_PRIMARY_CANONICAL_ARMS if mode == "primary" else _P11_SECONDARY_CANONICAL_ARMS)
        if require_canonical_family
        else None
    )
    canonical_main_regimes = (
        (REGIMES_ID if mode == "primary" else OFFGRID_REGIMES_ID)
        if require_canonical_family
        else None
    )
    holm_family = _verify_holm_tests_block(
        summary["tests"],
        mode,
        canonical_names=canonical_main,
        canonical_arms=canonical_main_arms,
        canonical_regimes=canonical_main_regimes,
        fallback_arms=None,
        default_regimes=list(OFFGRID_REGIMES_ID) if mode == "offgrid" else None,
    )
    holm_recomp = holm_family["contrasts"]

    ood_holm_family: Optional[dict[str, Any]] = None
    if "ood_tests" in summary and summary["ood_tests"] is not None:
        ood_holm_family = _verify_holm_tests_block(
            summary["ood_tests"],
            "offgrid_ood" if mode == "offgrid" else f"{mode}_ood",
            canonical_names=(
                tuple(_P11_SECONDARY_OOD_CONTRAST_ARMS.keys())
                if (require_canonical_family and mode == "offgrid")
                else None
            ),
            canonical_arms=(
                _P11_SECONDARY_OOD_CONTRAST_ARMS
                if (require_canonical_family and mode == "offgrid")
                else None
            ),
            canonical_regimes=(
                OFFGRID_REGIMES_OOD
                if (require_canonical_family and mode == "offgrid")
                else None
            ),
            fallback_arms=_P11_SECONDARY_OOD_CONTRAST_ARMS if mode == "offgrid" else None,
            default_regimes=list(OFFGRID_REGIMES_OOD) if mode == "offgrid" else None,
        )

    ni_recomp: Optional[dict[str, Any]] = None
    if mode == "primary" and require_canonical_family:
        require(
            "noninferiority" in summary and isinstance(summary["noninferiority"], Mapping),
            "Primary canonical audit requires a 'noninferiority' mapping",
        )
    if "noninferiority" in summary and summary["noninferiority"] is not None:
        rep_ni = summary["noninferiority"]
        require(isinstance(rep_ni, Mapping), f"noninferiority must be a mapping, got {type(rep_ni).__name__}")
        ni_cand_b, ni_cand_a = _extract_arm_spec(rep_ni, "candidate")
        ni_ref_b, ni_ref_a = _extract_arm_spec(rep_ni, "reference")
        raw_ni_metric = rep_ni.get("metric", "bler")
        require(
            isinstance(raw_ni_metric, str) and bool(raw_ni_metric),
            f"NI metric must be a non-empty string, got {raw_ni_metric!r}",
        )
        ni_metric = raw_ni_metric
        ni_regimes = rep_ni.get("regimes")
        if mode == "primary" and require_canonical_family:
            require(
                (ni_cand_b, ni_cand_a) == ("sparse", "p11_chained_wavefront"),
                f"Canonical NI candidate must be ['sparse', 'p11_chained_wavefront'], got {[ni_cand_b, ni_cand_a]}",
            )
            require(
                (ni_ref_b, ni_ref_a) == ("dense", "always_clear"),
                f"Canonical NI reference must be ['dense', 'always_clear'], got {[ni_ref_b, ni_ref_a]}",
            )
            require(
                ni_metric == "bler",
                f"Canonical NI metric must be 'bler', got {ni_metric!r}",
            )
            require(
                isinstance(ni_regimes, Sequence)
                and not isinstance(ni_regimes, (str, bytes))
                and list(ni_regimes) == ["continuous"],
                f"Canonical NI regimes must be ['continuous'], got {ni_regimes!r}",
            )
        require(not isinstance(rep_ni.get("margin"), (bool, np.bool_)), "NI margin must not be bool")
        ni_margin = float(rep_ni["margin"])
        require(math.isclose(ni_margin, 0.01, rel_tol=0.0, abs_tol=1e-15),
                f"Pre-registered NI margin must be +0.01, got {ni_margin}")
        ni_cand_bler = cluster_metric_vector(ni_cand_b, ni_cand_a, ni_metric, ni_regimes)
        ni_ref_bler = cluster_metric_vector(ni_ref_b, ni_ref_a, ni_metric, ni_regimes)
        ni_cand_bits = cluster_metric_vector(ni_cand_b, ni_cand_a, "net_delivered_info_bits_per_frame", ni_regimes)
        ni_ref_bits = cluster_metric_vector(ni_ref_b, ni_ref_a, "net_delivered_info_bits_per_frame", ni_regimes)
        ni_recomp = audit_engineering_noninferiority(
            ni_cand_bler, ni_ref_bler, ni_cand_bits, ni_ref_bits, margin=ni_margin
        )
        ni_sub = ni_recomp["bler_noninferiority"]
        rep_ni_diffs = _strict_1d_float_array(rep_ni.get("differences"), g, "noninferiority.differences")
        require(
            not isinstance(rep_ni.get("mean_diff"), (bool, np.bool_))
            and not isinstance(rep_ni.get("upper_95"), (bool, np.bool_)),
            "NI mean_diff/upper_95 must not be bool",
        )
        ni_diff_err = float(np.max(np.abs(rep_ni_diffs - np.asarray(ni_sub["differences"], dtype=np.float64))))
        ni_mean_err = abs(float(rep_ni["mean_diff"]) - ni_sub["mean_diff"])
        ni_upper_err = abs(float(rep_ni["upper_95"]) - ni_sub["upper_95"])
        require(ni_diff_err <= 2e-12 and ni_mean_err <= 2e-12 and ni_upper_err <= 2e-12,
                f"NI numerical mismatch: diff={ni_diff_err}, mean={ni_mean_err}, upper={ni_upper_err}")
        require(type(rep_ni.get("passes")) is bool,
                f"NI 'passes' must be exact bool, got {type(rep_ni.get('passes')).__name__}: {rep_ni.get('passes')!r}")
        require(rep_ni["passes"] is ni_recomp["passes"], "Incorrect NI pass/fail verdict")
        if "lower_95" in rep_ni:
            require(not isinstance(rep_ni["lower_95"], (bool, np.bool_)), "NI lower_95 must not be bool")
            ni_lower_err = abs(float(rep_ni["lower_95"]) - ni_sub["lower_95"])
            require(math.isfinite(ni_lower_err) and ni_lower_err <= 2e-12,
                    f"NI lower_95 mismatch: diff={ni_lower_err}")
            max_stat_diff = max(max_stat_diff, ni_lower_err)
        if "verdict" in rep_ni:
            require(rep_ni["verdict"] == ni_recomp["verdict"],
                    f"Incorrect NI verdict: {rep_ni['verdict']!r} vs {ni_recomp['verdict']!r}")
        if "verdict_zh" in rep_ni:
            require(rep_ni["verdict_zh"] == ni_recomp["verdict_zh"], "Incorrect NI verdict_zh")
        max_stat_diff = max(max_stat_diff, ni_diff_err, ni_mean_err, ni_upper_err)

    if mode == "primary" and "noninferiority_full" in summary and summary["noninferiority_full"] is not None:
        require(ni_recomp is not None, "Summary contains 'noninferiority_full' without 'noninferiority'")
        rep_ni_full = summary["noninferiority_full"]
        require(
            isinstance(rep_ni_full, Mapping),
            f"noninferiority_full must be a mapping, got {type(rep_ni_full).__name__}",
        )
        if "verdict" in rep_ni_full:
            require(
                rep_ni_full["verdict"] == ni_recomp["verdict"],
                f"noninferiority_full verdict mismatch: {rep_ni_full['verdict']!r} vs {ni_recomp['verdict']!r}",
            )
        if "verdict_zh" in rep_ni_full:
            require(
                rep_ni_full["verdict_zh"] == ni_recomp["verdict_zh"],
                f"noninferiority_full verdict_zh mismatch: {rep_ni_full['verdict_zh']!r} vs {ni_recomp['verdict_zh']!r}",
            )
        if "passes" in rep_ni_full:
            require(
                type(rep_ni_full["passes"]) is bool and rep_ni_full["passes"] is ni_recomp["passes"],
                f"noninferiority_full passes mismatch: {rep_ni_full['passes']!r} vs {ni_recomp['passes']!r}",
            )
        rep_bler_ni = rep_ni_full.get("bler_noninferiority")
        require(
            isinstance(rep_bler_ni, Mapping),
            "noninferiority_full missing 'bler_noninferiority' mapping",
        )
        ni_sub_full = ni_recomp["bler_noninferiority"]
        rep_full_diffs = _strict_1d_float_array(
            rep_bler_ni.get("differences"), g, "noninferiority_full.bler_noninferiority.differences"
        )
        for k_num in ("mean_diff", "upper_95", "lower_95"):
            require(
                k_num in rep_bler_ni and not isinstance(rep_bler_ni[k_num], (bool, np.bool_)),
                f"noninferiority_full.bler_noninferiority.{k_num} missing or bool",
            )
        f_diff_err = float(np.max(np.abs(rep_full_diffs - np.asarray(ni_sub_full["differences"], dtype=np.float64))))
        f_mean_err = abs(float(rep_bler_ni["mean_diff"]) - ni_sub_full["mean_diff"])
        f_upper_err = abs(float(rep_bler_ni["upper_95"]) - ni_sub_full["upper_95"])
        f_lower_err = abs(float(rep_bler_ni["lower_95"]) - ni_sub_full["lower_95"])
        require(
            f_diff_err <= 2e-12 and f_mean_err <= 2e-12 and f_upper_err <= 2e-12 and f_lower_err <= 2e-12,
            f"noninferiority_full numerical mismatch: diff={f_diff_err}, mean={f_mean_err}, upper={f_upper_err}, lower={f_lower_err}",
        )
        max_stat_diff = max(max_stat_diff, f_diff_err, f_mean_err, f_upper_err, f_lower_err)

    total_cells = len(records) * sum(len(arms_by_budget_ref[b]) for b in budgets_ref)
    result: dict[str, Any] = {
        "mode": mode,
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
        "totals_recomputed": totals_recomp,
        "by_regime_recomputed": regime_recomps,
        "holm_family": holm_family,
        "noninferiority": ni_recomp,
        "raw_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
    }
    if ood_holm_family is not None:
        result["ood_holm_family"] = ood_holm_family

    if mode == "primary":
        primary_target_arm = str(summary.get("target_arm", "p11_chained_wavefront"))
        primary_payload_arm = str(summary.get("payload_arm", "p11_payload_only"))
        if require_canonical_family:
            require(
                primary_target_arm == "p11_chained_wavefront",
                f"Canonical primary target_arm must be 'p11_chained_wavefront', got {primary_target_arm!r}",
            )
        if primary_target_arm in arms_by_budget_ref.get("sparse", ()):
            result["common_phase_blindspot"] = summarize_common_phase_blindspot(
                records,
                target_arm=primary_target_arm,
                payload_arm=primary_payload_arm,
                budget="sparse",
            )
        if "common_phase_blindspot" in summary and summary["common_phase_blindspot"] is not None:
            require(
                "common_phase_blindspot" in result,
                "Summary contains 'common_phase_blindspot' but target arm is absent from raw records",
            )
            rep_cpb = summary["common_phase_blindspot"]
            rec_cpb = result["common_phase_blindspot"]
            require(isinstance(rep_cpb, Mapping), "Summary 'common_phase_blindspot' must be a mapping")
            if "target_arm" in rep_cpb:
                require(
                    rep_cpb["target_arm"] == rec_cpb["target_arm"],
                    f"common_phase_blindspot.target_arm mismatch: {rep_cpb['target_arm']!r} vs {rec_cpb['target_arm']!r}",
                )
            if "payload_arm" in rep_cpb:
                require(
                    rep_cpb["payload_arm"] == rec_cpb["payload_arm"],
                    f"common_phase_blindspot.payload_arm mismatch: {rep_cpb['payload_arm']!r} vs {rec_cpb['payload_arm']!r}",
                )

            def _verify_cpb_subset(rep_sub: Any, rec_sub: Mapping[str, Any], sub_label: str) -> None:
                nonlocal max_stat_diff
                require(isinstance(rep_sub, Mapping), f"{sub_label} must be a mapping")
                for k_req in ("links", "blocks", "block_errors", "target_bler"):
                    require(k_req in rep_sub, f"{sub_label} missing required key {k_req!r}")
                for k_opt in (
                    "p10_9e9d93b_bler",
                    "delta_vs_p10_9e9d93b_pp",
                    "false_assimilation_rate",
                    "false_assimilations",
                    "total_assimilations",
                    "payload_bler",
                    "delta_vs_payload_pp",
                ):
                    if k_opt in rec_sub:
                        require(k_opt in rep_sub, f"{sub_label} missing key {k_opt!r}")
                for k, rec_v in rec_sub.items():
                    if k not in rep_sub:
                        continue
                    rep_v = rep_sub[k]
                    if isinstance(rec_v, (int, float)) and not isinstance(rec_v, bool):
                        require(not isinstance(rep_v, (bool, np.bool_)), f"{sub_label}.{k} must not be bool")
                        if type(rec_v) is int:
                            require(
                                type(rep_v) is int and rep_v == rec_v,
                                f"{sub_label}.{k} mismatch: {rep_v!r} vs {rec_v!r}",
                            )
                        else:
                            require(
                                isinstance(rep_v, (int, float, np.integer, np.floating)),
                                f"{sub_label}.{k} must be numeric, got {type(rep_v).__name__}",
                            )
                            rep_f = float(rep_v)
                            rec_f = float(rec_v)
                            require(
                                math.isfinite(rep_f) and math.isfinite(rec_f),
                                f"{sub_label}.{k} contains non-finite value",
                            )
                            diff = abs(rep_f - rec_f)
                            require(
                                math.isclose(rep_f, rec_f, rel_tol=2e-12, abs_tol=2e-12),
                                f"{sub_label}.{k} mismatch: {rep_v!r} vs {rec_v!r}",
                            )
                            max_stat_diff = max(max_stat_diff, diff)
                    elif isinstance(rec_v, str):
                        require(rep_v == rec_v, f"{sub_label}.{k} mismatch: {rep_v!r} vs {rec_v!r}")

            require("pooled_3id" in rep_cpb, "common_phase_blindspot missing 'pooled_3id'")
            _verify_cpb_subset(rep_cpb["pooled_3id"], rec_cpb["pooled_3id"], "common_phase_blindspot.pooled_3id")
            require(
                "by_regime" in rep_cpb and isinstance(rep_cpb["by_regime"], Mapping),
                "common_phase_blindspot missing 'by_regime' mapping",
            )
            require(
                set(rep_cpb["by_regime"].keys()) == set(rec_cpb["by_regime"].keys()),
                f"common_phase_blindspot.by_regime keys mismatch: {set(rep_cpb['by_regime'])} vs {set(rec_cpb['by_regime'])}",
            )
            for reg, rec_reg_sub in rec_cpb["by_regime"].items():
                _verify_cpb_subset(rep_cpb["by_regime"][reg], rec_reg_sub, f"common_phase_blindspot.by_regime.{reg}")
        result["max_stat_diff"] = float(max_stat_diff)

    if replay_path is not None:
        result["replay"] = verify_replay_exact(raw_path, replay_path, float_tol=replay_float_tol, strict=True)
    return result


def audit_p11_primary(
    raw_path: Path,
    summary_path: Path,
    replay_path: Optional[Path] = None,
    *,
    seed_pools: Optional[Mapping[str, Sequence[int]]] = None,
    replay_float_tol: float = 1e-12,
    require_canonical_family: bool = False,
) -> dict[str, Any]:
    """Independently recompute and audit P11 Primary Axis JSONL + summary + replay."""
    return _audit_cohort_common(
        raw_path,
        summary_path,
        replay_path,
        mode="primary",
        allowed_regimes=ALL_REGIMES,
        seed_pools=seed_pools,
        replay_float_tol=replay_float_tol,
        require_canonical_family=require_canonical_family,
    )


def audit_p11_offgrid(
    raw_path: Path,
    summary_path: Path,
    replay_path: Optional[Path] = None,
    *,
    seed_pools: Optional[Mapping[str, Sequence[int]]] = None,
    replay_float_tol: float = 1e-12,
    require_canonical_family: bool = False,
) -> dict[str, Any]:
    """Independently recompute and audit P11 Secondary Off-Grid JSONL + summary + replay."""
    return _audit_cohort_common(
        raw_path,
        summary_path,
        replay_path,
        mode="offgrid",
        allowed_regimes=OFFGRID_ALL_REGIMES,
        seed_pools=seed_pools,
        replay_float_tol=replay_float_tol,
        require_canonical_family=require_canonical_family,
    )


# ---------------------------------------------------------------------------
# 5. Synthetic Self-Test Suite (`--selftest`)
# ---------------------------------------------------------------------------

def _make_clean_synthetic_offgrid_obs(rng: np.random.Generator, snr_db: float) -> OffGridObservation:
    n0 = n0_from_snr(snr_db)
    y_a = (rng.normal(size=(T, N)) + 1j * rng.normal(size=(T, N))) / math.sqrt(2.0)
    y_b = (rng.normal(size=(T, N)) + 1j * rng.normal(size=(T, N))) / math.sqrt(2.0)
    y_b[PROBE_ROW] = complex(math.nan, math.nan)
    training_rows = (6,)
    pilot_tx_b = np.exp(1j * rng.uniform(-math.pi, math.pi, size=(len(training_rows), N))).astype(np.complex128)
    pilot_tx_a = np.exp(1j * rng.uniform(-math.pi, math.pi, size=(len(OFDM_PILOT_ROWS), N // 2))).astype(np.complex128)
    return OffGridObservation(
        y_a=y_a.astype(np.complex128),
        y_b=y_b.astype(np.complex128),
        training_rows=training_rows,
        probe_row=PROBE_ROW,
        n0=n0,
        pilot_tx_b=pilot_tx_b,
        pilot_tx_a=pilot_tx_a,
    )


def _make_clean_cstar_assimilator():
    def _assimilate(obs: PublicObservation | OffGridObservation) -> AssimilationResult:
        tr = list(obs.training_rows)
        e_a = float(np.mean(np.abs(obs.y_a[list(OFDM_PILOT_ROWS)]) ** 2))
        power_scale = max(e_a, float(obs.n0), 1e-300)
        proj = complex(np.sum(obs.y_a[list(OFDM_PILOT_ROWS)]))
        unit_phase = proj / abs(proj) if abs(proj) > 1e-12 else 1.0 + 0.0j
        c_scale = complex(math.sqrt(power_scale) * unit_phase)
        n0_canon = float(obs.n0) / power_scale

        valid_data = [r for r in range(T) if r != obs.probe_row and r not in set(tr)]
        chosen = tuple(r for r in (8, 12) if r in valid_data)
        row_weights: dict[int, float] = {}
        row_margins: dict[int, float] = {}
        decoded_info: dict[int, np.ndarray] = {}
        for r in valid_data:
            row_canon = obs.y_b[r] / c_scale
            snr_proxy = float(np.mean(np.abs(row_canon) ** 2)) / max(n0_canon, 1e-12)
            margin = float(snr_proxy / (1.0 + snr_proxy))
            row_margins[r] = margin
            row_weights[r] = float(0.8 * margin) if r in chosen else 0.0
            bits = np.zeros(64, dtype=np.uint8)
            bits[:N] = (row_canon.real >= 0.0).astype(np.uint8)
            bits[N:] = (row_canon.imag >= 0.0).astype(np.uint8)
            decoded_info[r] = bits

        anchor_rows = list(tr) + list(chosen)
        probe_pred = np.mean(obs.y_b[anchor_rows], axis=0).astype(np.complex128)
        probe_var = np.full(N, float(obs.n0) + 0.05 * power_scale, dtype=np.float64)

        llrs = np.zeros((T, N, 4), dtype=np.float64)
        for r in valid_data:
            rc = obs.y_b[r] / c_scale
            llrs[r, :, 0] = rc.real
            llrs[r, :, 1] = rc.imag
            llrs[r, :, 2] = np.abs(rc.real) - 0.5
            llrs[r, :, 3] = np.abs(rc.imag) - 0.5

        hw = np.full(T, 0.75, dtype=np.float64)
        mean_ray = np.zeros((len(DELAYS), T, N), dtype=np.complex128)
        mean_ray[0] = c_scale * 0.5
        var_ray = np.full((len(DELAYS), T, N), 0.1 * power_scale, dtype=np.float64)
        gate_out = GateOutput(
            history_weight=hw,
            probe_weight=0.75,
            mix_llrs=llrs,
            mix_probe_pred=probe_pred,
            mix_probe_var=probe_var,
            mix_mean_ray=mean_ray,
            mix_var_ray=var_ray,
        )
        return AssimilationResult(
            gate_out=gate_out,
            decoded_info=decoded_info,
            assimilated_rows=chosen,
            row_weights=row_weights,
            row_margins=row_margins,
            n_rounds=2,
            elapsed_ms=1.25,
        )
    return _assimilate


def run_selftest(output_path: Optional[Path] = None) -> dict[str, Any]:
    """Run complete independent P11 physics, subset-Bayes, firewall, and G=16 statistical self-test.

    Strictly synthetic — NEVER touches SEED_P11_FORMAL_ID or SEED_P11_FORMAL_OOD on real
    formal simulations.
    """
    rng = np.random.default_rng(11104040)
    p11_dir = Path(__file__).resolve().parent

    # 1. Independent Dirichlet fractional-delay kernel checks (< 1e-14)
    frac_report = verify_fractional_delay_operator(
        delays=(0.0, 0.25, 0.5, 1.37, 2.0, 3.14159265, 4.0, 5.83, 6.5),
        n=N,
        tol=1e-14,
    )
    require(frac_report["passed"], f"Fractional-delay operator verification failed: {frac_report}")

    # 2. Independent off-grid cross-waveform propagator checks
    idx = np.arange(N, dtype=np.int64)
    prod_mod = (idx[:, None] * idx[None, :]) % N
    f_dft = np.exp(-2j * math.pi * prod_mod.astype(np.float64) / float(N)) / math.sqrt(float(N))
    f_idft = f_dft.conj().T
    max_duality_err = 0.0
    max_hmat_err = 0.0
    for n_rays in (2, 3, 4, 5):
        delays = rng.uniform(0.0, 6.5, size=n_rays)
        gains = (rng.normal(size=n_rays) + 1j * rng.normal(size=n_rays)) / math.sqrt(2.0 * n_rays)
        freqs_zero = np.zeros(n_rays, dtype=np.float64)
        x_freq = (rng.normal(size=(T, N)) + 1j * rng.normal(size=(T, N))) / math.sqrt(2.0)
        x_time = x_freq @ f_idft.T
        prop_res = independent_offgrid_propagate_ab(
            x_freq, x_time, gains, delays, freqs_zero, a_is_freq=True, return_details=True
        )
        y_b_freq = prop_res.y_b @ f_dft.T
        max_duality_err = max(max_duality_err, float(np.max(np.abs(prop_res.y_a - y_b_freq))))
        for r in (0, 10, 19):
            h_contract = build_offgrid_sc_channel_matrix(prop_res.h_b_useful[:, r, :], delays, n=N)
            max_hmat_err = max(max_hmat_err, float(np.max(np.abs(prop_res.h_b_true_mat[r] - h_contract))))

    require(max_duality_err < 1e-14 and max_hmat_err < 1e-14,
            f"Off-grid static duality/H_mat check failed: duality={max_duality_err:.3e}, hmat={max_hmat_err:.3e}")

    a_grid = (rng.normal(size=(T, N)) + 1j * rng.normal(size=(T, N))) / math.sqrt(2.0)
    b_grid = (rng.normal(size=(T, N)) + 1j * rng.normal(size=(T, N))) / math.sqrt(2.0)
    gains3 = np.sqrt(POWERS) * (rng.normal(size=3) + 1j * rng.normal(size=3)) / math.sqrt(2.0)
    freqs3 = rng.uniform(0.045, 0.080, size=3) * rng.choice([-1.0, 1.0], size=3)
    chirps3 = rng.uniform(-0.0015, 0.0015, size=3)
    ch3 = {"gain": gains3, "frequency": freqs3, "chirp": chirps3}

    p9_ya_c, p9_yb_c = serial_propagate_ab(a_grid, b_grid, ch3)
    p11_ya_c, p11_yb_c = independent_offgrid_propagate_ab(
        a_grid, b_grid, gains3, DELAYS.astype(np.float64), freqs3, chirps3
    )
    p9_cont_err = max(float(np.max(np.abs(p11_ya_c - p9_ya_c))), float(np.max(np.abs(p11_yb_c - p9_yb_c))))
    require(p9_cont_err < 1e-14, f"P11 vs P9 continuous mismatch: {p9_cont_err:.3e}")

    shift_summary: dict[str, dict[str, Any]] = {}
    for s_row in (0, 6, 11):
        t_s = row_boundary_time(s_row, frame="B")
        shift = PiecewiseDopplerShift(ray_index=1, delta_freq=0.055, t_shift=t_s)
        cont = verify_offgrid_phase_continuity(gains3, freqs3, chirps3, shift)
        p9_ya_s, p9_yb_s = serial_propagate_ab(a_grid, b_grid, ch3, shift=shift)
        p11_ya_s, p11_yb_s = independent_offgrid_propagate_ab(
            a_grid, b_grid, gains3, DELAYS.astype(np.float64), freqs3, chirps3, shift_row=s_row,
            freqs_b=freqs3 + np.array([0.0, 0.055, 0.0], dtype=np.float64),
        )
        err_vs_p9 = max(float(np.max(np.abs(p11_ya_s - p9_ya_s))), float(np.max(np.abs(p11_yb_s - p9_yb_s))))
        require(cont["continuous_ok"] and err_vs_p9 < 1e-14,
                f"Shift check failed at shift_row={s_row}: cont={cont}, err_vs_p9={err_vs_p9:.3e}")
        shift_summary[f"shift_row_{s_row}"] = {
            "t_shift": t_s,
            "phase_jump_rad": cont["phase_jump_rad"],
            "gain_jump": cont["gain_jump"],
            "reanchored_max_gain_err": cont["reanchored_max_gain_err"],
            "err_vs_p9_serial": err_vs_p9,
            "passed": True,
        }

    # 3. Independent Complex Gaussian loglik & exact 2^3 subset-Bayes C*-invariance check
    m_dim = 16
    k_rays = 3
    n0_ref = n0_from_snr(9.0)
    mu_inh = np.zeros((k_rays, m_dim), dtype=np.complex128)
    mu_clr = np.zeros((k_rays, m_dim), dtype=np.complex128)
    cov_inh = np.zeros((k_rays, m_dim, m_dim), dtype=np.complex128)
    cov_clr = np.zeros((k_rays, m_dim, m_dim), dtype=np.complex128)
    for l_idx in range(k_rays):
        a_i = (rng.normal(size=(m_dim, 4)) + 1j * rng.normal(size=(m_dim, 4))) * 0.2
        a_c = (rng.normal(size=(m_dim, 4)) + 1j * rng.normal(size=(m_dim, 4))) * 0.55
        mu_inh[l_idx] = (rng.normal(size=m_dim) + 1j * rng.normal(size=m_dim)) * math.sqrt(float(POWERS[l_idx]))
        cov_inh[l_idx] = 0.5 * (a_i @ a_i.conj().T + (a_i @ a_i.conj().T).conj().T) * float(POWERS[l_idx])
        cov_clr[l_idx] = 0.5 * (a_c @ a_c.conj().T + (a_c @ a_c.conj().T).conj().T) * float(POWERS[l_idx])

    cov_test = n0_ref * np.eye(m_dim, dtype=np.complex128) + np.sum(cov_inh, axis=0)
    mu_test = np.sum(mu_inh, axis=0)
    y_test = mu_test + math.sqrt(n0_ref) * (rng.normal(size=m_dim) + 1j * rng.normal(size=m_dim)) / math.sqrt(2.0)
    ll_chol = independent_complex_gaussian_loglik(y_test, mu_test, cov_test)
    eigvals, eigvecs = eigh(cov_test)
    proj_e = eigvecs.conj().T @ (y_test - mu_test)
    ll_eigh = float(-m_dim * math.log(math.pi) - np.sum(np.log(eigvals)) - np.sum((np.abs(proj_e) ** 2) / eigvals))
    loglik_eigh_err = abs(ll_chol - ll_eigh)
    require(loglik_eigh_err < 1e-12, f"Complex Gaussian loglik Cholesky vs eigh mismatch: {loglik_eigh_err:.3e}")

    y_partial = (
        mu_inh[0] + mu_inh[2]
        + (rng.normal(size=m_dim) + 1j * rng.normal(size=m_dim)) * math.sqrt(n0_ref / 2.0)
    )
    sb_base = independent_subset_bayes_posteriors(y_partial, mu_inh, cov_inh, mu_clr, cov_clr, n0_ref)
    alpha_cstar = 3.25 * complex(math.cos(1.4), math.sin(1.4))
    sb_scaled = independent_subset_bayes_posteriors(
        alpha_cstar * y_partial,
        alpha_cstar * mu_inh,
        (abs(alpha_cstar) ** 2) * cov_inh,
        alpha_cstar * mu_clr,
        (abs(alpha_cstar) ** 2) * cov_clr,
        (abs(alpha_cstar) ** 2) * n0_ref,
    )
    sb_cstar_err = float(np.max(np.abs(sb_scaled["subset_posteriors"] - sb_base["subset_posteriors"])))
    require(sb_cstar_err < 1e-12, f"Subset-Bayes C* invariance error: {sb_cstar_err:.3e}")

    # 4. Runtime Anti-Leakage Firewalls, Strict P11 Binding, timedelta64 Guard, Seed Disjointness & Static Checks
    og_obs = _make_clean_synthetic_offgrid_obs(rng, snr_db=9.0)
    og_valid = verify_offgrid_observation(og_obs, strict=True)

    y_b_pub = (rng.normal(size=(T, N)) + 1j * rng.normal(size=(T, N))) / math.sqrt(2.0)
    y_b_pub[PROBE_ROW] = complex(math.nan, math.nan)
    pub_obs = PublicObservation(
        y_a=((rng.normal(size=(T, N)) + 1j * rng.normal(size=(T, N))) / math.sqrt(2.0)).astype(np.complex128),
        y_b=y_b_pub.astype(np.complex128),
        training_rows=SC_LAYOUTS["sparse"],
        probe_row=PROBE_ROW,
        n0=n0_ref,
        budget="sparse",
    )
    verify_p11_public_observation(pub_obs, strict=True)
    clean_assim = _make_clean_cstar_assimilator()
    assim_clean_rep = audit_assimilation_anti_leakage(clean_assim, pub_obs, strict=True)
    adapter_rep = check_assimilation_result_adapter(clean_assim(pub_obs), strict=True)

    # Fold target-row isolation synthetic test
    def _clean_fold(o: PublicObservation) -> dict[str, Any]:
        aux = (2, 5, 8, 13, 16)
        return {"aux_mean": np.mean(o.y_b[list(aux)], axis=0), "branch": "reset"}

    fold_iso_rep = audit_fold_target_isolation(
        _clean_fold, pub_obs, target_rows=(1, 3, 4, 6, 7, 11, 12, 14, 15, 17, 18), strict=True
    )

    # Synthetic integer-only seed check (synthetic lists inside P11 namespaces; NEVER runs formal sims!)
    seed_clean = check_p11_seed_disjointness(
        p11_train_seeds=[SEED_P11_TRAIN + i for i in range(32)],
        p11_dev_seeds=[SEED_P11_DEV + i for i in range(32)],
        p11_formal_id_seeds=[1_130_000_000_000 + i for i in range(32)],
        p11_formal_ood_seeds=[1_140_000_000_000 + i for i in range(32)],
        p11_e2e_seeds=[SEED_P11_E2E + i for i in range(32)],
        strict=True,
    )
    seed_overlap = check_p11_seed_disjointness(
        p11_train_seeds=[SEED_P11_TRAIN, P4_STAGE3_LITERAL_SEED],
        p11_dev_seeds=[SEED_P11_DEV],
        p11_formal_id_seeds=[1_130_000_000_000, P8_MANIFOLD_BASE_SEED],
        p11_formal_ood_seeds=[1_140_000_000_000],
        strict=False,
    )
    require(not seed_overlap["disjoint"], "P11 seed disjointness checker missed planted historical overlap")

    timedelta_rejected = False
    try:
        check_p11_seed_disjointness([np.timedelta64(1, "ns")], [SEED_P11_DEV], [1_130_000_000_000], [1_140_000_000_000])
    except TypeError:
        timedelta_rejected = True
    require(timedelta_rejected, "np.timedelta64 must be rejected as seed")

    n0_audit = check_p11_no_hardcoded_n0(
        [p11_dir / "contract.py", p11_dir / "independent_ref.py", p11_dir / "audit.py"],
        strict=True,
    )
    indep_audit = check_auditor_independence(
        [p11_dir / "independent_ref.py", p11_dir / "audit.py"],
        strict=True,
    )

    # 5. Exact G = 16 (65,536-state) Holm Families, Engineering Non-Inferiority & Replay Verifier
    t0 = time.perf_counter()
    p1_contrasts = {
        "chained_vs_p9_jt": np.linspace(0.08, 0.16, 16),
        "chained_vs_payload_only": np.linspace(0.03, 0.09, 16),
    }
    p2_contrasts = {
        "transport_attn_vs_abs_no_transport": np.linspace(0.04, 0.10, 16),
        "transport_attn_vs_local": np.linspace(0.03, 0.08, 16),
        "transport_attn_vs_no_history": np.linspace(0.02, 0.06, 16),
    }
    holm_rep = audit_p11_holm_families(p1_contrasts, p2_contrasts, expected_g=16)
    holm_ms = (time.perf_counter() - t0) * 1000.0
    require(holm_rep["primary"]["exact_sign_patterns"] == 65536 and holm_rep["secondary"]["exact_sign_patterns"] == 65536,
            "Expected 65,536 exact sign patterns for G=16")
    require(holm_rep["primary"]["all_supported"] and holm_rep["secondary"]["all_supported"],
            "Synthetic Holm families should be supported")
    require(all(v in VALID_VERDICTS_ZH for v in holm_rep["primary"]["verdicts_zh"].values()),
            "Invalid verdicts_zh in primary Holm family")

    ni_pass_rep = audit_engineering_noninferiority(
        sparse_cluster_bler=np.full(16, 0.086) + rng.normal(0.0, 0.002, size=16),
        dense_cluster_bler=np.full(16, 0.084) + rng.normal(0.0, 0.002, size=16),
        sparse_cluster_bits=np.full(16, 1052.0) + rng.normal(0.0, 4.0, size=16),
        dense_cluster_bits=np.full(16, 938.0) + rng.normal(0.0, 4.0, size=16),
        margin=0.01,
    )
    require(ni_pass_rep["passes"] and ni_pass_rep["verdict_zh"] == "支持" and ni_pass_rep["engineering_dominance_passes"],
            f"Synthetic passing engineering NI failed: {ni_pass_rep}")

    ni_fail_rep = audit_engineering_noninferiority(
        sparse_cluster_bler=np.full(16, 0.105),
        dense_cluster_bler=np.full(16, 0.084),
        sparse_cluster_bits=np.full(16, 1030.0),
        dense_cluster_bits=np.full(16, 938.0),
        margin=0.01,
    )
    require((not ni_fail_rep["passes"]) and ni_fail_rep["verdict_zh"] == "反证", "Synthetic failing NI verdict mismatch")

    # Synthetic raw-JSONL Primary (11+5 pooling) & Off-Grid (18 blocks) audit round-trip
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        p_raw = tmp_dir / "primary.jsonl"
        p_sum = tmp_dir / "primary_summary.json"
        p_rep = tmp_dir / "primary_replay.jsonl"
        synth_recs = []
        for c in range(1, 5):
            for l_idx, reg in enumerate(REGIMES_ID):
                s = SEED_P11_DEV + 1000 * c + l_idx
                cells_sp = {}
                for arm_name, base_e in (("p11_chained_wavefront", 1), ("p11_payload_only", 3), ("p9_jeffreys_student_t", 4)):
                    f0_e = min(11, base_e)
                    f1_e = 0 if arm_name == "p11_chained_wavefront" else 1
                    tot_e = f0_e + f1_e
                    cells_sp[arm_name] = {
                        "blocks": 16,
                        "block_errors": tot_e,
                        "fold0_blocks": 11,
                        "fold1_blocks": 5,
                        "fold0_errors": f0_e,
                        "fold1_errors": f1_e,
                        "coded_bits": 2048,
                        "coded_errors": tot_e * 8,
                        "info_bits": 1024,
                        "info_errors": tot_e * 4,
                        "extra_blocks": 2,
                        "extra_errors": 0,
                        "probes": 32,
                        "probe_sse": 4.0,
                        "clean_sse": 1.0,
                        "probe_energy": 32.0,
                        "clean_energy": 30.0,
                        "variance_sum": 4.0,
                        "nll_sum": -10.0,
                        "total_assimilations": 4,
                        "false_assimilations": 0,
                        "wave1_fold0": "correct",
                        "wave1_fold1": "correct",
                    }
                cells_dn = {
                    "always_clear": {
                        "blocks": 16,
                        "block_errors": 1,
                        "fold0_blocks": 11,
                        "fold1_blocks": 5,
                        "fold0_errors": 1,
                        "fold1_errors": 0,
                        "coded_bits": 2048,
                        "coded_errors": 8,
                        "info_bits": 1024,
                        "info_errors": 4,
                        "extra_blocks": 0,
                        "extra_errors": 0,
                        "probes": 32,
                        "probe_sse": 4.0,
                        "clean_sse": 1.0,
                        "probe_energy": 32.0,
                        "clean_energy": 30.0,
                        "variance_sum": 4.0,
                        "nll_sum": -10.0,
                    }
                }
                synth_recs.append({"cluster": c, "link": l_idx, "seed": s, "regime": reg, "budgets": {"sparse": cells_sp, "dense": cells_dn}})
        p_raw.write_text("\n".join(json.dumps(r) for r in synth_recs) + "\n", encoding="utf-8", newline="\n")
        p_rep.write_text("\n".join(json.dumps(r) for r in synth_recs[:3]) + "\n", encoding="utf-8", newline="\n")

        def _c_vec(bud: str, arm: str, met: str, reg_f: Optional[Sequence[str]] = None) -> np.ndarray:
            out_v = []
            for c in range(1, 5):
                sub = [r["budgets"][bud][arm] for r in synth_recs if r["cluster"] == c and (reg_f is None or r["regime"] in reg_f)]
                out_v.append(float(aggregate_p11_cells(sub)[met]))
            return np.asarray(out_v, dtype=np.float64)

        p_tests = _audit_single_holm_family(
            {
                "chained_vs_p9_jt": {"baseline": _c_vec("sparse", "p9_jeffreys_student_t", "bler"), "candidate": _c_vec("sparse", "p11_chained_wavefront", "bler")},
                "chained_vs_payload_only": {"baseline": _c_vec("sparse", "p11_payload_only", "bler"), "candidate": _c_vec("sparse", "p11_chained_wavefront", "bler")},
            },
            "primary",
        )["contrasts"]
        p_tests[0].update({"baseline": ["sparse", "p9_jeffreys_student_t"], "candidate": ["sparse", "p11_chained_wavefront"]})
        p_tests[1].update({"baseline": ["sparse", "p11_payload_only"], "candidate": ["sparse", "p11_chained_wavefront"]})
        p_ni = audit_engineering_noninferiority(
            _c_vec("sparse", "p11_chained_wavefront", "bler", ["continuous"]),
            _c_vec("dense", "always_clear", "bler", ["continuous"]),
            _c_vec("sparse", "p11_chained_wavefront", "net_delivered_info_bits_per_frame", ["continuous"]),
            _c_vec("dense", "always_clear", "net_delivered_info_bits_per_frame", ["continuous"]),
        )
        p_sum_obj = {
            "totals": _aggregate_nested_budget_arm(synth_recs, ("sparse", "dense"), {"sparse": ("p11_chained_wavefront", "p11_payload_only", "p9_jeffreys_student_t"), "dense": ("always_clear",)}),
            "clusters": [
                _aggregate_nested_budget_arm([r for r in synth_recs if r["cluster"] == c], ("sparse", "dense"), {"sparse": ("p11_chained_wavefront", "p11_payload_only", "p9_jeffreys_student_t"), "dense": ("always_clear",)})
                for c in range(1, 5)
            ],
            "tests": p_tests,
            "noninferiority": {
                **p_ni["bler_noninferiority"],
                "candidate": ("sparse", "p11_chained_wavefront"),
                "reference": ("dense", "always_clear"),
                "regimes": ["continuous"],
            },
        }
        p_sum.write_text(json.dumps(p_sum_obj, indent=2) + "\n", encoding="utf-8", newline="\n")
        cohort_audit_rep = audit_p11_primary(p_raw, p_sum, p_rep)
        require(cohort_audit_rep["replay"]["passed"], "Synthetic cohort replay failed")

    selftest_report: dict[str, Any] = {
        "status": "PASS",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "fractional_delay_kernel": frac_report,
        "offgrid_propagator": {
            "max_static_duality_err": max_duality_err,
            "max_h_b_true_mat_vs_contract_err": max_hmat_err,
            "p9_continuous_match_err": p9_cont_err,
            "piecewise_shifts": shift_summary,
            "passed": True,
        },
        "subset_bayes_reference": {
            "complex_gaussian_cholesky_vs_eigh_err": loglik_eigh_err,
            "partial_shift_dominant_subset": int(np.argmax(sb_base["subset_posteriors"])),
            "partial_shift_ray_posteriors": [float(x) for x in sb_base["ray_posteriors"].tolist()],
            "cstar_invariance_max_err": sb_cstar_err,
            "passed": True,
        },
        "firewall_checks": {
            "offgrid_observation_valid": og_valid["valid"],
            "clean_assimilator_passed": assim_clean_rep["passed"],
            "adapter_valid": adapter_rep["valid"],
            "fold_target_isolation_passed": fold_iso_rep["passed"],
            "clean_assimilator_probe_poison_diff": assim_clean_rep["max_probe_poison_diff"],
            "clean_assimilator_cstar_err": assim_clean_rep["max_cstar_err"],
            "p11_seed_disjointness_clean": seed_clean["disjoint"],
            "planted_historical_seed_overlap_caught": not seed_overlap["disjoint"],
            "timedelta64_rejected": timedelta_rejected,
            "p11_no_hardcoded_n0_clean": n0_audit["clean"],
            "auditor_independence_clean": indep_audit["independent"],
            "passed": True,
        },
        "g16_statistical_verifier": {
            "holm_dual_axis_ms": round(holm_ms, 3),
            "primary_patterns": holm_rep["primary"]["exact_sign_patterns"],
            "primary_verdicts": holm_rep["primary"]["verdicts"],
            "primary_verdicts_zh": holm_rep["primary"]["verdicts_zh"],
            "secondary_patterns": holm_rep["secondary"]["exact_sign_patterns"],
            "secondary_verdicts": holm_rep["secondary"]["verdicts"],
            "secondary_verdicts_zh": holm_rep["secondary"]["verdicts_zh"],
            "engineering_ni_pass_verdict_zh": ni_pass_rep["verdict_zh"],
            "engineering_ni_fail_verdict_zh": ni_fail_rep["verdict_zh"],
            "synthetic_primary_cohort_replay_passed": cohort_audit_rep["replay"]["passed"],
            "passed": True,
        },
    }

    target_out = output_path if output_path is not None else (RESULTS_DIR / "p11_audit_selftest.json")
    target_out.parent.mkdir(parents=True, exist_ok=True)
    target_out.write_text(json.dumps(selftest_report, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return selftest_report


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 11 independent blind statistical auditor & firewall.")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run full synthetic P11 self-test and write JSON report.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON report path.")
    args = parser.parse_args(argv)

    if args.selftest:
        out_path = args.output if args.output is not None else (RESULTS_DIR / "p11_audit_selftest.json")
        report = run_selftest(out_path)
        print(json.dumps({
            "status": report["status"],
            "output": str(out_path),
            "frac_delay_max_err": report["fractional_delay_kernel"]["max_contract_err"],
            "p9_serial_match_err": report["offgrid_propagator"]["p9_continuous_match_err"],
            "subset_bayes_cstar_err": report["subset_bayes_reference"]["cstar_invariance_max_err"],
            "holm_dual_axis_ms": report["g16_statistical_verifier"]["holm_dual_axis_ms"],
        }, indent=2))
        return

    parser.error("Run with --selftest to execute the Phase 11 independent audit self-test.")


__all__ = [
    "P10_9E9D93B_REFERENCE",
    "P11_PRIMARY_CANONICAL_CONTRASTS",
    "P11_SECONDARY_CANONICAL_CONTRASTS",
    "RESULTS_DIR",
    "VALID_VERDICTS",
    "VALID_VERDICTS_ZH",
    "VERDICT_ZH",
    "aggregate_p11_cells",
    "audit_engineering_noninferiority",
    "audit_p11_holm_families",
    "audit_p11_offgrid",
    "audit_p11_primary",
    "check_p11_offgrid_cell",
    "check_p11_primary_cell",
    "exact_sign_flip_test",
    "exact_student_t_ppf",
    "holm_bonferroni_adjust",
    "noninferiority_test",
    "require",
    "run_selftest",
    "summarize_common_phase_blindspot",
    "verdict_to_zh",
    "verify_paired_energies",
    "verify_replay_exact",
]


if __name__ == "__main__":
    main()
