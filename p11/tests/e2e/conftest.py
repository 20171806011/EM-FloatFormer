"""Shared fixtures and opaque-box helpers for P11 E2E tests.

Strictly imports only stdlib, numpy, pytest, p11.contract, and p9/p10 frozen priors at top level
so progressive testability works on M1 commit 7e4f912 without skipping or mocking M2/M3.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pytest

import p9.channel as p9_channel
import p9.contract as p9_contract
import p9.independent_ref as p9_indep
import p11.contract as p11_contract
from p11.contract import (
    ANCHOR,
    CORE_ROWS,
    P10_BASELINE_COMMIT,
    P11_NAMESPACE_SPAN,
    P11_SEED_RANGES,
    PRIMARY_SNR_DB,
    PROBE_ROW,
    PublicObservation,
    HiddenTruth,
    REGIMES_ID,
    SEED_P11_DEV,
    SEED_P11_E2E,
    enforce_determinism,
    is_historical_seed,
    n0_from_snr,
)

E2E_DIR = Path(__file__).resolve().parent
P11_DIR = E2E_DIR.parents[1]
REPO_ROOT = P11_DIR.parent

AUX1: tuple[int, ...] = (2, 5, 8, 13, 16)
AUX2: tuple[int, ...] = (1, 4, 7, 12, 15)
VALID_ROWS: tuple[int, ...] = (0, 9)
FOLD_TARGETS: tuple[tuple[int, ...], tuple[int, ...]] = (
    tuple(int(r) for r in CORE_ROWS if r not in AUX1),
    AUX1,
)
MATH_VERIFICATION_SHA256 = "8be75b4c44adf62f901dc30180310280fe0f12876ef87c0a2c004cfa38d21033"
OFFICIAL_MAIN_COMMIT = "63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1"
EXPECTED_BRANCH = "research/p11-wavefront-payload-assimilation"

OFFGRID_VARIANTS: tuple[str, ...] = (
    "em10_transport_attn",
    "em10_abs_no_transport",
    "em10_local",
    "em10_no_history",
)
OFFGRID_CKPT_SEEDS: tuple[int, ...] = (1001, 1002, 1003)

_FORBIDDEN_SKIP_ATTR = "sk" + "ip"
_FORBIDDEN_SKIP_CLS = "Skip" + "Test"
_FORBIDDEN_INFER_ATTR = "inference_" + "mode"
_FORBIDDEN_MOCK_MOD = "unittest." + "mock"
_FORMAL_ID_ATTR = "SEED_P11_" + "FORMAL_ID"
_FORMAL_OOD_ATTR = "SEED_P11_" + "FORMAL_OOD"


@pytest.fixture(scope="session", autouse=True)
def _enforce_e2e_determinism() -> dict[str, Any]:
    """Session-level single-thread deterministic runtime guard."""
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    return enforce_determinism(require_env=False)


def clean_subprocess_env(
    extra: Optional[dict[str, str]] = None,
    *,
    drop_threads: bool = False,
) -> dict[str, str]:
    """Build deterministic subprocess environment for python -B CLI invocations."""
    env = os.environ.copy()
    if drop_threads:
        for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
            env.pop(var, None)
    else:
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    if extra:
        env.update(extra)
    return env


def run_p11_cli(
    args: Sequence[str],
    *,
    cwd: Path = REPO_ROOT,
    env: Optional[dict[str, str]] = None,
    timeout: int = 600,
) -> subprocess.CompletedProcess[str]:
    """Run `python -B ...` as a clean opaque-box subprocess."""
    cmd = [sys.executable, "-B", *args]
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env if env is not None else clean_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def run_git(args: Sequence[str], *, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    """Execute read-only git command."""
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def assert_non_formal_e2e_seed(seed: Any) -> int:
    """Strictly allow only non-historical, non-formal P11 E2E/DEV integer seeds."""
    if isinstance(seed, (bool, np.bool_, np.timedelta64, np.datetime64)) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"Seed must be plain integer, got {type(seed).__name__}")
    s_int = int(seed)
    hist = is_historical_seed(s_int)
    if hist is not None:
        raise ValueError(f"Seed {s_int} collides with historical namespace {hist!r}")
    formal_id_lo, formal_id_hi = P11_SEED_RANGES["p11_formal_id"]
    formal_ood_lo, formal_ood_hi = P11_SEED_RANGES["p11_formal_ood"]
    if formal_id_lo <= s_int < formal_id_hi or formal_ood_lo <= s_int < formal_ood_hi:
        raise ValueError(f"Seed {s_int} lies inside forbidden closed-book formal namespace")
    dev_lo, dev_hi = P11_SEED_RANGES["p11_dev"]
    e2e_lo, e2e_hi = P11_SEED_RANGES["p11_e2e"]
    if not ((dev_lo <= s_int < dev_hi) or (e2e_lo <= s_int < e2e_hi)):
        raise ValueError(f"Seed {s_int} outside allowed P11 E2E/DEV namespaces")
    return s_int


def make_e2e_public_obs(
    offset: int,
    snr_db: float,
    regime: str,
    budget: str = "sparse",
) -> tuple[PublicObservation, HiddenTruth]:
    """Simulate a closed-book-safe E2E P9/P11 Primary-Axis link using SEED_P11_E2E."""
    seed = assert_non_formal_e2e_seed(SEED_P11_E2E + int(offset))
    obs, truth = p9_channel.simulate_link(seed, float(snr_db), regime, budget)
    assert obs.probe_row == PROBE_ROW
    assert math.isclose(obs.n0, n0_from_snr(float(snr_db)), rel_tol=1e-14, abs_tol=1e-14)
    return obs, truth


def clone_public_obs(
    obs: PublicObservation,
    *,
    y_a: Optional[np.ndarray] = None,
    y_b: Optional[np.ndarray] = None,
    training_rows: Optional[tuple[int, ...]] = None,
    probe_row: Optional[int] = None,
    n0_val: Any = None,
    budget: Optional[str] = None,
) -> PublicObservation:
    """Create a fresh PublicObservation copy with explicit overrides."""
    return PublicObservation(
        y_a=obs.y_a.copy() if y_a is None else y_a,
        y_b=obs.y_b.copy() if y_b is None else y_b,
        training_rows=obs.training_rows if training_rows is None else training_rows,
        probe_row=obs.probe_row if probe_row is None else probe_row,
        n0=float(obs.n0) if n0_val is None else n0_val,
        budget=obs.budget if budget is None else budget,
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scan_infer_guard_ast(paths: Iterable[Path]) -> list[tuple[str, int]]:
    """Detect any AST Attribute or Name referencing torch inference mode."""
    hits: list[tuple[str, int]] = []
    for p in paths:
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == _FORBIDDEN_INFER_ATTR:
                hits.append((str(p), int(node.lineno)))
            elif isinstance(node, ast.Name) and node.id == _FORBIDDEN_INFER_ATTR:
                hits.append((str(p), int(node.lineno)))
    return hits


def scan_skip_ast(paths: Iterable[Path]) -> list[tuple[str, int, str]]:
    """Detect any pytest skip calls/decorators or unittest SkipTest usage."""
    hits: list[tuple[str, int, str]] = []
    for p in paths:
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == _FORBIDDEN_SKIP_ATTR:
                hits.append((str(p), int(node.lineno), _FORBIDDEN_SKIP_ATTR))
            elif isinstance(node, ast.Name) and node.id == _FORBIDDEN_SKIP_CLS:
                hits.append((str(p), int(node.lineno), _FORBIDDEN_SKIP_CLS))
    return hits


def scan_mock_ast(paths: Iterable[Path]) -> list[tuple[str, int, str]]:
    """Detect any unittest mock imports or monkeypatch fixtures."""
    hits: list[tuple[str, int, str]] = []
    for p in paths:
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == _FORBIDDEN_MOCK_MOD or alias.name.startswith(_FORBIDDEN_MOCK_MOD + "."):
                        hits.append((str(p), int(node.lineno), alias.name))
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == _FORBIDDEN_MOCK_MOD or mod.startswith(_FORBIDDEN_MOCK_MOD + "."):
                    hits.append((str(p), int(node.lineno), mod))
                elif mod == "unittest" and any(a.name == "mock" for a in node.names):
                    hits.append((str(p), int(node.lineno), "from unittest import mock"))
            elif isinstance(node, ast.FunctionDef):
                for arg in list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs):
                    if arg.arg == "monkeypatch":
                        hits.append((str(p), int(node.lineno), "monkeypatch"))
    return hits


def scan_private_p11_imports_ast(paths: Iterable[Path]) -> list[tuple[str, int, str]]:
    """Detect any private `_` symbol imports from `p11.*` (opaque-box enforcement)."""
    hits: list[tuple[str, int, str]] = []
    for p in paths:
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                if mod == "p11" or mod.startswith("p11."):
                    for alias in node.names:
                        if alias.name.startswith("_"):
                            hits.append((str(p), int(node.lineno), f"from {mod} import {alias.name}"))
    return hits


def scan_formal_seed_symbols_ast(paths: Iterable[Path]) -> list[tuple[str, int, str]]:
    """Ensure E2E test files never reference formal seed symbol names or numeric constants."""
    forbidden_names = {_FORMAL_ID_ATTR, _FORMAL_OOD_ATTR}
    forbidden_vals = {
        int(getattr(p11_contract, _FORMAL_ID_ATTR)),
        int(getattr(p11_contract, _FORMAL_OOD_ATTR)),
    }
    hits: list[tuple[str, int, str]] = []
    for p in paths:
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in forbidden_names:
                hits.append((str(p), int(node.lineno), node.id))
            elif isinstance(node, ast.Attribute) and node.attr in forbidden_names:
                hits.append((str(p), int(node.lineno), node.attr))
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in forbidden_names:
                        hits.append((str(p), int(node.lineno), alias.name))
            elif isinstance(node, ast.Constant) and type(node.value) is int and node.value in forbidden_vals:
                hits.append((str(p), int(node.lineno), str(node.value)))
    return hits


def make_synthetic_g16_dataset(
    dest_dir: Path,
    seed_base: int,
    *,
    links_per_cluster: int = 1,
    sparse_bler_bias: float = 0.0,
    mean_contrast_diff: float = 0.05,
) -> dict[str, Path]:
    """Generate deterministic G=16 synthetic JSONL + summary + replay + audit files under SEED_P11_E2E."""
    import p11.audit as p11_audit

    assert_non_formal_e2e_seed(seed_base)
    dest_dir.mkdir(parents=True, exist_ok=True)
    raw_path = dest_dir / "p11_primary_id_links.jsonl"
    replay_path = dest_dir / "p11_primary_replay.jsonl"
    summary_path = dest_dir / "p11_primary_summary.json"
    audit_path = dest_dir / "p11_audit.json"

    records: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed_base) % (2**31 - 1))
    n_per_reg = max(1, int(links_per_cluster))

    def _make_cell(err: int, extra_blocks: int, extra_err: int, p_energy: float, c_energy: float, var_sum: float, nll_base: float, ms_key: str, ms_val: float) -> dict[str, Any]:
        f0_err = min(11, max(0, err - (err // 3)))
        f1_err = err - f0_err
        if f1_err > 5:
            f0_err += f1_err - 5
            f1_err = 5
        cell: dict[str, Any] = {
            "blocks": 16,
            "block_errors": int(err),
            "coded_bits": 2048,
            "coded_errors": int(err * 8),
            "info_bits": 1024,
            "info_errors": int(err * 4),
            "extra_blocks": int(extra_blocks),
            "extra_errors": int(extra_err),
            "probes": 32,
            "probe_sse": 1.1 + 0.05 * float(err),
            "clean_sse": 0.8 + 0.04 * float(err),
            "probe_energy": p_energy,
            "clean_energy": c_energy,
            "variance_sum": var_sum,
            "nll_sum": nll_base + 0.05 * float(rng.standard_normal()),
            "fold0_blocks": 11,
            "fold1_blocks": 5,
            "fold0_errors": int(f0_err),
            "fold1_errors": int(f1_err),
            ms_key: ms_val,
        }
        if extra_blocks > 0:
            cell["total_assimilations"] = 4
            cell["false_assimilations"] = 0
            cell["wave1_fold0"] = "correct"
            cell["wave1_fold1"] = "correct"
        return cell

    for c_idx in range(1, 17):
        link_id = 0
        for reg in REGIMES_ID:
            for l_sub in range(n_per_reg):
                seed = assert_non_formal_e2e_seed(int(seed_base) + 1000 * c_idx + link_id)
                p_energy = float(32.0 + 0.5 * link_id)
                c_energy = float(31.0 + 0.5 * link_id)
                if reg == "continuous":
                    dn_err = 1
                    ch_err = int(max(0, min(16, 1 + (1 if sparse_bler_bias > 0.0 and (c_idx % 2 == 0 or mean_contrast_diff < 0.0) else 0))))
                    pl_err = 3
                    jt_err = 4
                elif reg == "partial_shift":
                    dn_err = 1
                    ch_err = 2
                    pl_err = 5
                    jt_err = 6
                else:
                    dn_err = 1
                    ch_err = 2
                    pl_err = 5
                    jt_err = 6

                sparse_arms = {
                    "p11_chained_wavefront": _make_cell(ch_err, 2, 0, p_energy, c_energy, 4.0, -12.0, "pipeline_ms", 11.2 + 0.3 * l_sub),
                    "p11_payload_only": _make_cell(pl_err, 2, 1, p_energy, c_energy, 4.4, -11.0, "pipeline_ms", 8.1 + 0.2 * l_sub),
                    "p9_jeffreys_student_t": _make_cell(jt_err, 2, 1, p_energy, c_energy, 4.8, -10.5, "pipeline_ms", 7.5 + 0.2 * l_sub),
                }
                dense_arms = {
                    "always_clear": _make_cell(dn_err, 0, 0, p_energy, c_energy, 3.8, -12.5, "decode_ms", 9.4 + 0.2 * l_sub),
                }
                records.append({
                    "split": "id",
                    "cluster": c_idx,
                    "link": link_id,
                    "seed": seed,
                    "regime": reg,
                    "budgets": {
                        "sparse": sparse_arms,
                        "dense": dense_arms,
                    },
                    "link_ms": 20.6 + 0.5 * link_id,
                })
                link_id += 1

    raw_text = "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n"
    raw_path.write_text(raw_text, encoding="utf-8", newline="\n")

    # Replay alters only *_ms timing keys while preserving every scientific field
    replay_records = []
    for r in records[: len(REGIMES_ID) * n_per_reg]:
        rc = json.loads(json.dumps(r))
        rc["link_ms"] = rc["link_ms"] + 5.0
        for b_map in rc["budgets"].values():
            for c_map in b_map.values():
                for ms_k in ("pipeline_ms", "decode_ms", "elapsed_ms"):
                    if ms_k in c_map:
                        c_map[ms_k] = c_map[ms_k] + 1.5
        replay_records.append(rc)
    replay_path.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in replay_records) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    arms_map = {
        "sparse": ("p11_chained_wavefront", "p11_payload_only", "p9_jeffreys_student_t"),
        "dense": ("always_clear",),
    }

    def _agg_nested(sub_recs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {
            bud: {
                arm: p11_audit.aggregate_p11_cells([r["budgets"][bud][arm] for r in sub_recs])
                for arm in arms_map[bud]
            }
            for bud in ("sparse", "dense")
        }

    def _c_vec(bud: str, arm: str, met: str, reg_f: Optional[Sequence[str]] = None) -> np.ndarray:
        return np.asarray([
            p11_audit.aggregate_p11_cells([
                r["budgets"][bud][arm]
                for r in records
                if r["cluster"] == c and (reg_f is None or r["regime"] in reg_f)
            ])[met]
            for c in range(1, 17)
        ], dtype=np.float64)

    totals = _agg_nested(records)
    clusters = [_agg_nested([r for r in records if r["cluster"] == c]) for c in range(1, 17)]
    by_regime = {reg: _agg_nested([r for r in records if r["regime"] == reg]) for reg in sorted(REGIMES_ID)}

    holm = p11_audit.audit_p11_holm_families(
        {
            "chained_vs_p9_jt": {
                "baseline": _c_vec("sparse", "p9_jeffreys_student_t", "bler"),
                "candidate": _c_vec("sparse", "p11_chained_wavefront", "bler"),
            },
            "chained_vs_payload_only": {
                "baseline": _c_vec("sparse", "p11_payload_only", "bler"),
                "candidate": _c_vec("sparse", "p11_chained_wavefront", "bler"),
            },
        },
        {
            "dummy_secondary": np.full(16, 0.05, dtype=np.float64),
        },
        expected_g=16,
    )["primary"]["contrasts"]
    holm[0].update({"baseline": ["sparse", "p9_jeffreys_student_t"], "candidate": ["sparse", "p11_chained_wavefront"]})
    holm[1].update({"baseline": ["sparse", "p11_payload_only"], "candidate": ["sparse", "p11_chained_wavefront"]})

    ni = p11_audit.audit_engineering_noninferiority(
        _c_vec("sparse", "p11_chained_wavefront", "bler", ["continuous"]),
        _c_vec("dense", "always_clear", "bler", ["continuous"]),
        _c_vec("sparse", "p11_chained_wavefront", "net_delivered_info_bits_per_frame", ["continuous"]),
        _c_vec("dense", "always_clear", "net_delivered_info_bits_per_frame", ["continuous"]),
    )
    summary_obj = {
        "totals": totals,
        "clusters": clusters,
        "by_regime": by_regime,
        "tests": holm,
        "noninferiority": {
            **ni["bler_noninferiority"],
            "candidate": ["sparse", "p11_chained_wavefront"],
            "reference": ["dense", "always_clear"],
            "regimes": ["continuous"],
        },
        "noninferiority_full": ni,
        "common_phase_blindspot": p11_audit.summarize_common_phase_blindspot(records),
    }
    summary_path.write_text(json.dumps(summary_obj, indent=2) + "\n", encoding="utf-8", newline="\n")

    aud_res = p11_audit.audit_p11_primary(
        raw_path, summary_path, replay_path, require_canonical_family=True
    )
    audit_obj = {
        "mode": "dry_run",
        "axis": "primary",
        "freeze_sha256": None,
        "seed_disjointness": {"disjoint": True},
        "primary": {
            "id_links": len(records),
            "ood_links": 0,
            "audit": aud_res,
            "replay_links": len(replay_records),
            "replay_verify": aud_res["replay"],
        },
    }
    audit_path.write_text(json.dumps(audit_obj, indent=2) + "\n", encoding="utf-8", newline="\n")
    return {
        "dir": dest_dir,
        "raw": raw_path,
        "replay": replay_path,
        "summary": summary_path,
        "audit": audit_path,
    }


def call_estimate_payload_posterior(
    payload_mod: Any,
    obs: PublicObservation,
    aux: tuple[int, ...],
    *,
    use_payload: bool = True,
) -> Any:
    """Opaque-box adapter supporting either PROJECT.md or math_verification.md public signature."""
    if hasattr(payload_mod, "estimate_payload_posterior_obs"):
        return payload_mod.estimate_payload_posterior_obs(obs, aux, use_payload=use_payload)
    fn = payload_mod.estimate_payload_posterior
    try:
        return fn(obs, aux, use_payload=use_payload)
    except TypeError:
        return fn(obs.y_b[ANCHOR], aux, obs.y_b[list(aux)], float(obs.n0))
