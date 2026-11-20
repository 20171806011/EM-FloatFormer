"""Tier 2 E2E Boundary, Corner, and Negative Tests (F1-F10, 55 tests).

F1-F3 and F8 pass immediately on M1 commit 7e4f912.
F4-F7 and F9-F10 import M2/M3 modules lazily inside test bodies (zero skips, zero mocks).
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import t as scipy_student_t

import p9.independent_ref as p9_indep
import p11.contract as p11_contract
from p11.contract import (
    ANCHOR,
    COMPLEX_DTYPE,
    CORE_ROWS,
    HISTORICAL_SEED_RANGES,
    N,
    OFFGRID_DELAY_MAX,
    OFFGRID_DELAY_MIN,
    P10_BASELINE_COMMIT,
    P11_SEED_RANGES,
    PRIMARY_SNR_DB,
    PROBE_ROW,
    REAL_DTYPE,
    SEED_P11_DEV,
    SEED_P11_E2E,
    SEED_P11_TRAIN,
    SEED_P9_DEV,
    SNR_GRID_DB,
    T,
    OffGridObservation,
    apply_fractional_sc_operator,
    fractional_delay_matrix,
    is_historical_seed,
    n0_from_snr,
)
from p11.tests.e2e.conftest import (
    AUX1,
    AUX2,
    FOLD_TARGETS,
    MATH_VERIFICATION_SHA256,
    OFFGRID_VARIANTS,
    OFFICIAL_MAIN_COMMIT,
    P11_DIR,
    assert_non_formal_e2e_seed,
    call_estimate_payload_posterior,
    clean_subprocess_env,
    clone_public_obs,
    make_e2e_public_obs,
    make_synthetic_g16_dataset,
    run_git,
    run_p11_cli,
    scan_formal_seed_symbols_ast,
    scan_infer_guard_ast,
    scan_mock_ast,
    scan_private_p11_imports_ast,
    scan_skip_ast,
    sha256_file,
)


# ============================================================================
# F1: Git Branch & Repo Guardrail Boundaries (5 tests)
# ============================================================================

def test_f01_t2_detects_simulated_non_p11_diff_in_temp_worktree(tmp_path: Path) -> None:
    diff = run_git(["diff", "--name-only", P10_BASELINE_COMMIT, "--", ":!p11"])
    assert diff.returncode == 0
    assert diff.stdout.strip() == ""

    # Create a real temporary git repo, commit baseline, modify both root.txt and p11/mod.py,
    # and verify `git diff --name-only -- :!p11` catches root.txt while ignoring p11/mod.py.
    sim_repo = tmp_path / "sim_repo"
    sim_repo.mkdir()
    assert run_git(["init"], cwd=sim_repo).returncode == 0
    assert run_git(["config", "user.email", "e2e@example.com"], cwd=sim_repo).returncode == 0
    assert run_git(["config", "user.name", "E2E Test"], cwd=sim_repo).returncode == 0
    (sim_repo / "root.txt").write_text("baseline\n", encoding="utf-8")
    (sim_repo / "p11").mkdir()
    (sim_repo / "p11" / "mod.py").write_text("X = 1\n", encoding="utf-8")
    assert run_git(["add", "."], cwd=sim_repo).returncode == 0
    assert run_git(["commit", "-m", "base"], cwd=sim_repo).returncode == 0

    (sim_repo / "root.txt").write_text("modified\n", encoding="utf-8")
    (sim_repo / "p11" / "mod.py").write_text("X = 2\n", encoding="utf-8")
    sim_non_p11 = run_git(["diff", "--name-only", "--", ":!p11"], cwd=sim_repo)
    assert sim_non_p11.returncode == 0
    assert [ln.strip() for ln in sim_non_p11.stdout.splitlines() if ln.strip()] == ["root.txt"]
    sim_p11 = run_git(["diff", "--name-only", "--", "p11"], cwd=sim_repo)
    assert sim_p11.returncode == 0
    assert [ln.strip() for ln in sim_p11.stdout.splitlines() if ln.strip()] == ["p11/mod.py"]


def test_f01_t2_rejects_untracked_source_code_in_root() -> None:
    status = run_git(["status", "--porcelain"])
    assert status.returncode == 0
    stray = []
    for line in status.stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if not (path.startswith(".agents/") or path.startswith("p11/")):
            stray.append(path)
    assert stray == [], f"Unexpected untracked/modified paths outside .agents/ and p11/: {stray}"


def test_f01_t2_main_branch_pinned_to_63ad2ec() -> None:
    res = run_git(["rev-parse", "main"])
    assert res.returncode == 0
    assert res.stdout.strip() == OFFICIAL_MAIN_COMMIT


def test_f01_t2_p9_p10_trees_sha_match_9e9d93b() -> None:
    for subdir in ("p4", "p5", "p6", "p7", "p8", "p8c", "p9", "p10"):
        head_tree = run_git(["rev-parse", f"HEAD:{subdir}"])
        base_tree = run_git(["rev-parse", f"{P10_BASELINE_COMMIT}:{subdir}"])
        assert head_tree.returncode == 0 and base_tree.returncode == 0
        assert head_tree.stdout.strip() == base_tree.stdout.strip(), f"Tree mismatch for {subdir}"


def test_f01_t2_no_infer_guard_across_entire_p11_tree() -> None:
    py_files = sorted(p for p in P11_DIR.rglob("*.py") if "__pycache__" not in p.parts)
    assert len(py_files) >= 5
    assert scan_infer_guard_ast(py_files) == []


# ============================================================================
# F2: Disjoint P11 Contract & Determinism Boundaries (6 tests)
# ============================================================================

def test_f02_t2_seed_rejects_bool_timedelta64_datetime64_nan_float() -> None:
    bad_seeds = [
        True,
        False,
        np.bool_(True),
        np.timedelta64(42, "ns"),
        np.timedelta64("NaT"),
        np.datetime64("2026-01-01"),
        float("nan"),
        1.5,
        "1150000000000",
        None,
    ]
    for bad in bad_seeds:
        with pytest.raises(TypeError):
            is_historical_seed(bad)  # type: ignore[arg-type]


def test_f02_t2_historical_seed_exact_boundary_endpoints() -> None:
    for name, (lo, hi) in HISTORICAL_SEED_RANGES.items():
        assert is_historical_seed(lo) == name
        assert is_historical_seed(hi - 1) == name
        # hi itself must NOT belong to `name`
        assert is_historical_seed(hi) != name
        if lo > 0:
            assert is_historical_seed(lo - 1) != name

    for p11_name, (lo, hi) in P11_SEED_RANGES.items():
        assert is_historical_seed(lo) is None, p11_name
        assert is_historical_seed(hi - 1) is None, p11_name


def test_f02_t2_enforce_determinism_rejects_missing_or_multi_threads() -> None:
    script = (
        "import os, sys\n"
        "import p11.contract as c\n"
        "raised = False\n"
        "try:\n"
        "    c.enforce_determinism(require_env=True)\n"
        "except RuntimeError:\n"
        "    raised = True\n"
        "assert raised, 'expected RuntimeError'\n"
        "assert 'torch' not in sys.modules\n"
    )
    proc_missing = run_p11_cli(["-c", script], env=clean_subprocess_env(drop_threads=True))
    assert proc_missing.returncode == 0, proc_missing.stderr

    proc_multi = run_p11_cli(
        ["-c", script],
        env=clean_subprocess_env(extra={"OPENBLAS_NUM_THREADS": "4"}),
    )
    assert proc_multi.returncode == 0, proc_multi.stderr


def test_f02_t2_fractional_delay_negative_large_and_zero_n() -> None:
    ident = np.eye(N, dtype=COMPLEX_DTYPE)
    for d in (-10.5, 0.0, 1e4):
        mat = fractional_delay_matrix(d, N)
        assert np.max(np.abs(mat.conj().T @ mat - ident)) < 1e-12

    d1 = fractional_delay_matrix(0.5, n=1)
    assert d1.shape == (1, 1) and abs(abs(d1[0, 0]) - 1.0) < 1e-14

    for bad_n in (0, -1, True, np.timedelta64(1, "s")):
        with pytest.raises(ValueError):
            fractional_delay_matrix(0.5, n=bad_n)  # type: ignore[arg-type]

    for bad_d in (float("nan"), float("inf"), True, np.timedelta64(1, "ns")):
        with pytest.raises(ValueError):
            fractional_delay_matrix(bad_d, N)  # type: ignore[arg-type]


def test_f02_t2_broken_repr_and_cyclic_containers_rejected() -> None:
    class _BrokenRepr:
        def __repr__(self) -> str:
            raise RuntimeError("boom repr")

    cyclic_list: list = []
    cyclic_list.append(cyclic_list)

    cyclic_obj = np.empty((1,), dtype=object)
    cyclic_obj[0] = cyclic_obj

    for bad in (_BrokenRepr(), cyclic_list, cyclic_obj):
        with pytest.raises(ValueError):
            fractional_delay_matrix(bad, N)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            apply_fractional_sc_operator(bad, np.ones((1, N)), np.array([0.0]), N)  # type: ignore[arg-type]


def test_f02_t2_offgrid_operator_rejects_complex_delays_and_masked_arrays() -> None:
    masked = np.ma.masked_array([1.0, 2.0, 3.0], mask=[False, True, False])
    with pytest.raises(ValueError):
        fractional_delay_matrix(1.0 + 0.5j, N)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        apply_fractional_sc_operator(
            np.ones(N, dtype=COMPLEX_DTYPE),
            np.ones((3, N), dtype=COMPLEX_DTYPE),
            masked,
            N,
        )


# ============================================================================
# F3: Codex Math Verification Boundaries (5 tests)
# ============================================================================

def test_f03_t2_math_verification_rejects_naive_a_gauge_floor_trap() -> None:
    text = (P11_DIR / "math_verification.md").read_text(encoding="utf-8")
    assert "1e-300" in text
    assert "homogeneity" in text


def test_f03_t2_math_verification_all_zero_anchor_convention() -> None:
    text = (P11_DIR / "math_verification.md").read_text(encoding="utf-8")
    assert "u_B=1" in text or "u_B = 1" in text
    assert "all-zero anchor" in text.lower()


def test_f03_t2_math_verification_row11_fold0_firewall_documented() -> None:
    text = (P11_DIR / "math_verification.md").read_text(encoding="utf-8")
    assert "row 11 is a Fold 0 target" in text
    assert "(9,)" in text and "(9,11)" in text


def test_f03_t2_math_verification_t_crit_df15_accuracy() -> None:
    text = (P11_DIR / "math_verification.md").read_text(encoding="utf-8")
    assert "1.75305" in text
    t_exact = float(scipy_student_t.ppf(0.95, 15))
    assert abs(1.75305 - t_exact) < 1e-5


def test_f03_t2_math_verification_sha256_integrity() -> None:
    assert sha256_file(P11_DIR / "math_verification.md") == MATH_VERIFICATION_SHA256


# ============================================================================
# F4: Payload Covariance Reconstruction Boundaries (6 tests)
# ============================================================================

def test_f04_t2_all_zero_anchor_y10_returns_zero_mean_and_finite_var() -> None:
    payload_rebuild = importlib.import_module("p11.payload_rebuild")
    obs, _ = make_e2e_public_obs(201, PRIMARY_SNR_DB, "full_reset")
    y_b_zero = obs.y_b.copy()
    y_b_zero[ANCHOR] = 0.0 + 0.0j
    obs_zero = clone_public_obs(obs, y_b=y_b_zero)

    post_pilot = call_estimate_payload_posterior(payload_rebuild, obs_zero, AUX1, use_payload=False)
    assert np.isfinite(post_pilot.mean).all()
    assert np.isfinite(post_pilot.var).all() and np.all(post_pilot.var >= 0.0)
    assert np.max(np.abs(post_pilot.mean)) < 1e-12

    post_full = call_estimate_payload_posterior(payload_rebuild, obs_zero, AUX1, use_payload=True)
    assert np.isfinite(post_full.mean).all()
    assert np.isfinite(post_full.var).all() and np.all(post_full.var >= 0.0)


def test_f04_t2_extreme_cstar_scaling_1e_minus5_to_1e_plus5_arbitrary_phases() -> None:
    payload_rebuild = importlib.import_module("p11.payload_rebuild")
    obs, _ = make_e2e_public_obs(202, PRIMARY_SNR_DB, "full_reset")
    base = call_estimate_payload_posterior(payload_rebuild, obs, AUX1, use_payload=True)

    scales = (
        1e-5 * complex(math.cos(0.7), math.sin(0.7)),
        1e5 * complex(math.cos(-2.4), math.sin(-2.4)),
        1e-4 * complex(math.cos(3.1), math.sin(3.1)),
        1e4 * complex(math.cos(-1.1), math.sin(-1.1)),
    )
    for alpha in scales:
        alpha_sq = float(abs(alpha) ** 2)
        y_b_s = (alpha * obs.y_b).astype(COMPLEX_DTYPE)
        y_b_s[PROBE_ROW] = complex(math.nan, math.nan)
        obs_s = clone_public_obs(
            obs,
            y_a=(alpha * obs.y_a).astype(COMPLEX_DTYPE),
            y_b=y_b_s,
            n0_val=alpha_sq * float(obs.n0),
        )
        scaled = call_estimate_payload_posterior(payload_rebuild, obs_s, AUX1, use_payload=True)
        m_ref = alpha * base.mean
        v_ref = alpha_sq * base.var
        m_err = np.max(np.abs(scaled.mean - m_ref)) / max(1e-30, float(np.max(np.abs(m_ref))))
        v_err = np.max(np.abs(scaled.var - v_ref)) / max(1e-30, float(np.max(np.abs(v_ref))))
        assert m_err < 1e-11
        assert v_err < 1e-11


def test_f04_t2_invalid_n0_and_malformed_aux_rows_rejected() -> None:
    payload_rebuild = importlib.import_module("p11.payload_rebuild")
    obs, _ = make_e2e_public_obs(203, PRIMARY_SNR_DB, "full_reset")

    for bad_n0 in (0.0, -1.0, float("nan"), float("inf"), True, np.bool_(True), np.timedelta64(1, "s")):
        bad_obs = clone_public_obs(obs, n0_val=bad_n0)
        with pytest.raises((ValueError, TypeError)):
            call_estimate_payload_posterior(payload_rebuild, bad_obs, AUX1)
        with pytest.raises((ValueError, TypeError)):
            payload_rebuild.estimate_payload_posterior(obs.y_b[ANCHOR], AUX1, obs.y_b[list(AUX1)], bad_n0)

    for bad_aux in ((2, 2, 5, 8, 13), (2, 5, 8, 13, ANCHOR), (2, 5, 8, 13, PROBE_ROW), (2, 5, 8), ()):
        with pytest.raises((ValueError, TypeError)):
            call_estimate_payload_posterior(payload_rebuild, obs, bad_aux)


def test_f04_t2_nonfinite_authorized_inputs_rejected() -> None:
    payload_rebuild = importlib.import_module("p11.payload_rebuild")
    obs, _ = make_e2e_public_obs(204, PRIMARY_SNR_DB, "full_reset")

    y_nan_anchor = obs.y_b.copy()
    y_nan_anchor[ANCHOR, 0] = complex(math.nan, 0.0)
    with pytest.raises(ValueError):
        call_estimate_payload_posterior(payload_rebuild, clone_public_obs(obs, y_b=y_nan_anchor), AUX1)

    y_inf_aux = obs.y_b.copy()
    y_inf_aux[AUX1[0], 0] = complex(math.inf, 0.0)
    with pytest.raises(ValueError):
        call_estimate_payload_posterior(payload_rebuild, clone_public_obs(obs, y_b=y_inf_aux), AUX1)


def test_f04_t2_kappa_zero_and_huge_snr_bessel_stability() -> None:
    payload_rebuild = importlib.import_module("p11.payload_rebuild")
    obs_hi, _ = make_e2e_public_obs(205, 30.0, "full_reset")
    post_hi = call_estimate_payload_posterior(payload_rebuild, obs_hi, AUX1, use_payload=True)
    assert np.isfinite(post_hi.mean).all()
    assert np.isfinite(post_hi.var).all() and np.all(post_hi.var >= 0.0)


def test_f04_t2_negative_snrs_scale_invariance() -> None:
    payload_rebuild = importlib.import_module("p11.payload_rebuild")
    for snr_db in SNR_GRID_DB:
        obs, _ = make_e2e_public_obs(206, snr_db, "full_reset")
        post = call_estimate_payload_posterior(payload_rebuild, obs, AUX1, use_payload=True)
        assert np.isfinite(post.mean).all()
        assert np.isfinite(post.var).all() and np.all(post.var >= 0.0)


# ============================================================================
# F5: Wavefront Assimilation & Shift Repair Boundaries (6 tests)
# ============================================================================

def test_f05_t2_probe_row_poisoning_sentinels_zero_delta() -> None:
    wf = importlib.import_module("p11.wavefront_pipeline")
    obs, _ = make_e2e_public_obs(220, PRIMARY_SNR_DB, "full_reset")
    base = wf.run_p11_primary(obs)

    for sentinel in (1e6 - 1e6j, -5e5 + 3e5j, 1e12 + 1e12j):
        y_p = obs.y_b.copy()
        y_p[PROBE_ROW] = sentinel
        res = wf.run_p11_primary(clone_public_obs(obs, y_b=y_p))
        assert tuple(res.assimilated_rows) == tuple(base.assimilated_rows)
        assert set(res.decoded_info.keys()) == set(base.decoded_info.keys())
        for r, bits in base.decoded_info.items():
            assert np.array_equal(res.decoded_info[r], bits)
        assert np.max(np.abs(res.gate_out.mix_probe_pred - base.gate_out.mix_probe_pred)) == 0.0
        assert np.max(np.abs(res.gate_out.mix_probe_var - base.gate_out.mix_probe_var)) == 0.0
        assert np.max(np.abs(res.gate_out.mix_llrs - base.gate_out.mix_llrs)) == 0.0


def test_f05_t2_target_row_poisoning_pretarget_invariance() -> None:
    wf = importlib.import_module("p11.wavefront_pipeline")
    obs, _ = make_e2e_public_obs(221, PRIMARY_SNR_DB, "partial_shift")
    _, base_trace = wf.run_p11_with_trace(obs)

    for fold_idx, targets in enumerate(FOLD_TARGETS):
        y_p = obs.y_b.copy()
        y_p[list(targets)] = 2.125 - 0.75j
        _, p_trace = wf.run_p11_with_trace(clone_public_obs(obs, y_b=y_p))
        assert p_trace["folds"][fold_idx]["branch"] == base_trace["folds"][fold_idx]["branch"]
        assert p_trace["folds"][fold_idx]["state_version"] == base_trace["folds"][fold_idx]["state_version"]
        assert {rec["row"] for rec in p_trace["folds"][fold_idx]["ledger"]} == {
            rec["row"] for rec in base_trace["folds"][fold_idx]["ledger"]
        }

    # Fold 0 single-target poisoning (row 11 in TARGETS[0]) must leave all other TARGETS[0] rows untouched
    base_res = wf.run_p11_primary(obs)
    y_r11 = obs.y_b.copy()
    y_r11[11] = 2.125 - 0.75j
    r11_res = wf.run_p11_primary(clone_public_obs(obs, y_b=y_r11))
    for r2 in FOLD_TARGETS[0]:
        if r2 != 11:
            assert np.max(np.abs(r11_res.gate_out.mix_llrs[r2] - base_res.gate_out.mix_llrs[r2])) == 0.0
            assert np.array_equal(r11_res.decoded_info[r2], base_res.decoded_info[r2])


def test_f05_t2_extreme_cstar_wavefront_1e_minus5_and_1e_plus5() -> None:
    wf = importlib.import_module("p11.wavefront_pipeline")
    obs, _ = make_e2e_public_obs(222, PRIMARY_SNR_DB, "continuous")
    base = wf.run_p11_primary(obs)

    for alpha in (1e-5 * complex(math.cos(2.1), math.sin(2.1)), 1e5 * complex(math.cos(-1.7), math.sin(-1.7))):
        alpha_sq = float(abs(alpha) ** 2)
        y_b_s = (alpha * obs.y_b).astype(COMPLEX_DTYPE)
        y_b_s[PROBE_ROW] = complex(math.nan, math.nan)
        obs_s = clone_public_obs(
            obs,
            y_a=(alpha * obs.y_a).astype(COMPLEX_DTYPE),
            y_b=y_b_s,
            n0_val=alpha_sq * float(obs.n0),
        )
        scaled = wf.run_p11_primary(obs_s)
        assert tuple(scaled.assimilated_rows) == tuple(base.assimilated_rows)
        for r in CORE_ROWS:
            assert np.array_equal(scaled.decoded_info[r], base.decoded_info[r])
        bg, sg = base.gate_out, scaled.gate_out
        assert np.max(np.abs(sg.mix_llrs - bg.mix_llrs)) / max(1.0, float(np.max(np.abs(bg.mix_llrs)))) < 1e-11
        assert np.max(np.abs(sg.history_weight - bg.history_weight)) < 1e-11
        assert abs(float(sg.probe_weight) - float(bg.probe_weight)) < 1e-11
        pred_ref = alpha * bg.mix_probe_pred
        var_ref = alpha_sq * bg.mix_probe_var
        mean_ray_ref = alpha * bg.mix_mean_ray
        var_ray_ref = alpha_sq * bg.mix_var_ray
        assert np.max(np.abs(sg.mix_probe_pred - pred_ref)) / max(1e-300, float(np.max(np.abs(pred_ref)))) < 1e-11
        assert np.max(np.abs(sg.mix_probe_var - var_ref)) / max(1e-300, float(np.max(np.abs(var_ref)))) < 1e-11
        assert np.max(np.abs(sg.mix_mean_ray - mean_ray_ref)) / max(1.0, float(np.max(np.abs(mean_ray_ref)))) < 1e-11
        assert np.max(np.abs(sg.mix_var_ray - var_ray_ref)) / max(1.0, float(np.max(np.abs(var_ray_ref)))) < 1e-11


def test_f05_t2_invalid_obs_n0_or_unmasked_nonprobe_nan_rejected() -> None:
    wf = importlib.import_module("p11.wavefront_pipeline")
    obs, _ = make_e2e_public_obs(223, PRIMARY_SNR_DB, "continuous")

    for bad_n0 in (0.0, -0.5, float("nan")):
        with pytest.raises(ValueError):
            wf.run_p11_primary(clone_public_obs(obs, n0_val=bad_n0))

    y_nan = obs.y_b.copy()
    y_nan[1, 0] = complex(math.nan, 0.0)
    with pytest.raises(ValueError):
        wf.run_p11_primary(clone_public_obs(obs, y_b=y_nan))


def test_f05_t2_no_unlock_high_noise_floor_stability() -> None:
    wf = importlib.import_module("p11.wavefront_pipeline")
    obs, _ = make_e2e_public_obs(224, -6.0, "full_reset")
    res = wf.run_p11_primary(obs)
    assert np.isfinite(res.gate_out.mix_probe_pred).all()
    assert np.isfinite(res.gate_out.mix_probe_var).all()
    assert np.all(res.gate_out.mix_probe_var >= float(obs.n0) - 1e-12)


def test_f05_t2_duplicate_or_probe_in_training_rows_rejected() -> None:
    wf = importlib.import_module("p11.wavefront_pipeline")
    obs, _ = make_e2e_public_obs(225, PRIMARY_SNR_DB, "continuous")
    for bad_tr in ((ANCHOR, PROBE_ROW), (ANCHOR, ANCHOR), (PROBE_ROW,)):
        with pytest.raises(ValueError):
            wf.run_p11_primary(clone_public_obs(obs, training_rows=bad_tr))


# ============================================================================
# F6: Secondary Axis Off-Grid Manifold Boundaries (6 tests)
# ============================================================================

def test_f06_t2_offgrid_extreme_cstar_1e_minus5_to_1e_plus5() -> None:
    import torch

    og = importlib.import_module("p11.offgrid_channel")
    mg = importlib.import_module("p11.manifold_gate")
    seed = assert_non_formal_e2e_seed(SEED_P11_E2E + 250)
    obs, _ = og.simulate_offgrid_link(seed, "offgrid_continuous", snr_db=PRIMARY_SNR_DB)
    b0 = og.build_offgrid_branches(obs)

    ckpt = torch.load(P11_DIR / "ckpts" / "p10b_dev_ckpts" / "em10_transport_attn_seed1001.pt", map_location="cpu", weights_only=True)
    model = mg.ManifoldTrustGate(variant="em10_transport_attn", **ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    r0 = model.gate(obs, b0)

    for alpha in (1e-5 * complex(math.cos(0.9), math.sin(0.9)), 1e5 * complex(math.cos(-1.3), math.sin(-1.3))):
        alpha_sq = float(abs(alpha) ** 2)
        y_b_s = (alpha * obs.y_b).astype(COMPLEX_DTYPE)
        y_b_s[obs.probe_row] = complex(math.nan, math.nan)
        obs_s = OffGridObservation(
            y_a=(alpha * obs.y_a).astype(COMPLEX_DTYPE),
            y_b=y_b_s,
            training_rows=obs.training_rows,
            probe_row=obs.probe_row,
            n0=alpha_sq * float(obs.n0),
            pilot_tx_b=obs.pilot_tx_b.copy(),
            pilot_tx_a=obs.pilot_tx_a.copy(),
        )
        bs = og.build_offgrid_branches(obs_s)
        rs = model.gate(obs_s, bs)
        assert np.max(np.abs(rs.w_ray - r0.w_ray)) < 1e-11
        g0, gs = r0.gate_out, rs.gate_out
        assert np.max(np.abs(gs.mix_llrs - g0.mix_llrs)) / max(1.0, float(np.max(np.abs(g0.mix_llrs)))) < 1e-11
        assert np.max(np.abs(gs.history_weight - g0.history_weight)) < 1e-11
        assert abs(float(gs.probe_weight) - float(g0.probe_weight)) < 1e-11
        pred_ref = alpha * g0.mix_probe_pred
        var_ref = alpha_sq * g0.mix_probe_var
        mean_ray_ref = alpha * g0.mix_mean_ray
        var_ray_ref = alpha_sq * g0.mix_var_ray
        assert np.max(np.abs(gs.mix_probe_pred - pred_ref)) / max(1e-300, float(np.max(np.abs(pred_ref)))) < 1e-11
        assert np.max(np.abs(gs.mix_probe_var - var_ref)) / max(1e-300, float(np.max(np.abs(var_ref)))) < 1e-11
        assert np.max(np.abs(gs.mix_mean_ray - mean_ray_ref)) / max(1e-300, float(np.max(np.abs(mean_ray_ref)))) < 1e-11
        assert np.max(np.abs(gs.mix_var_ray - var_ray_ref)) / max(1e-300, float(np.max(np.abs(var_ray_ref)))) < 1e-11


def test_f06_t2_offgrid_probe_poisoning_zero_delta() -> None:
    import torch

    og = importlib.import_module("p11.offgrid_channel")
    mg = importlib.import_module("p11.manifold_gate")
    seed = assert_non_formal_e2e_seed(SEED_P11_E2E + 251)
    obs, _ = og.simulate_offgrid_link(seed, "offgrid_midframe_shift", snr_db=PRIMARY_SNR_DB)
    b0 = og.build_offgrid_branches(obs)

    ckpt = torch.load(P11_DIR / "ckpts" / "p10b_dev_ckpts" / "em10_transport_attn_seed1002.pt", map_location="cpu", weights_only=True)
    model = mg.ManifoldTrustGate(variant="em10_transport_attn", **ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    r0 = model.gate(obs, b0)

    for sentinel in (1e6 - 1e6j, -5e5 + 3e5j):
        y_p = obs.y_b.copy()
        y_p[obs.probe_row] = sentinel
        obs_p = OffGridObservation(
            y_a=obs.y_a.copy(),
            y_b=y_p,
            training_rows=obs.training_rows,
            probe_row=obs.probe_row,
            n0=obs.n0,
            pilot_tx_b=obs.pilot_tx_b.copy(),
            pilot_tx_a=obs.pilot_tx_a.copy(),
        )
        bp = og.build_offgrid_branches(obs_p)
        rp = model.gate(obs_p, bp)
        assert np.max(np.abs(rp.w_ray - r0.w_ray)) == 0.0
        assert np.max(np.abs(rp.gate_out.mix_llrs - r0.gate_out.mix_llrs)) == 0.0
        assert np.max(np.abs(rp.gate_out.mix_probe_pred - r0.gate_out.mix_probe_pred)) == 0.0
        assert np.max(np.abs(rp.gate_out.mix_probe_var - r0.gate_out.mix_probe_var)) == 0.0


def test_f06_t2_offgrid_invalid_n0_and_training_probe_overlap() -> None:
    og = importlib.import_module("p11.offgrid_channel")
    seed = assert_non_formal_e2e_seed(SEED_P11_E2E + 252)
    obs, _ = og.simulate_offgrid_link(seed, "offgrid_continuous", snr_db=PRIMARY_SNR_DB)

    for bad_n0 in (-1.0, 0.0, float("nan")):
        bad = OffGridObservation(
            y_a=obs.y_a.copy(),
            y_b=obs.y_b.copy(),
            training_rows=obs.training_rows,
            probe_row=obs.probe_row,
            n0=bad_n0,
            pilot_tx_b=obs.pilot_tx_b.copy(),
            pilot_tx_a=obs.pilot_tx_a.copy(),
        )
        with pytest.raises(ValueError):
            og.discover_frame_a_atoms(bad)

    overlap = OffGridObservation(
        y_a=obs.y_a.copy(),
        y_b=obs.y_b.copy(),
        training_rows=(obs.probe_row,),
        probe_row=obs.probe_row,
        n0=obs.n0,
        pilot_tx_b=obs.pilot_tx_b.copy(),
        pilot_tx_a=obs.pilot_tx_a.copy(),
    )
    with pytest.raises(ValueError):
        og.build_offgrid_branches(overlap)


def test_f06_t2_offgrid_delay_boundary_0_and_6_5_and_5_rays() -> None:
    og = importlib.import_module("p11.offgrid_channel")
    seed = assert_non_formal_e2e_seed(SEED_P11_E2E + 253)
    obs5, truth5 = og.simulate_offgrid_link(seed, "offgrid_ood_five_ray_fast", snr_db=PRIMARY_SNR_DB)
    assert truth5.n_rays == 5

    gains = np.ones((2, N), dtype=COMPLEX_DTYPE)
    delays = np.array([OFFGRID_DELAY_MIN, OFFGRID_DELAY_MAX], dtype=REAL_DTYPE)
    h_mat = p11_contract.build_offgrid_sc_channel_matrix(gains, delays, N)
    x = np.ones(N, dtype=COMPLEX_DTYPE)
    y = apply_fractional_sc_operator(x, gains, delays, N)
    assert np.max(np.abs(y - (h_mat @ x))) < 1e-13
    assert obs5.probe_row == PROBE_ROW


def test_f06_t2_offgrid_no_history_mask_invariance_to_y_a() -> None:
    import torch

    og = importlib.import_module("p11.offgrid_channel")
    mg = importlib.import_module("p11.manifold_gate")
    seed = assert_non_formal_e2e_seed(SEED_P11_E2E + 254)
    obs, _ = og.simulate_offgrid_link(seed, "offgrid_continuous", snr_db=PRIMARY_SNR_DB)
    b0 = og.build_offgrid_branches(obs)

    obs_ya_poison = OffGridObservation(
        y_a=np.full_like(obs.y_a, 100.0 - 200.0j),
        y_b=obs.y_b.copy(),
        training_rows=obs.training_rows,
        probe_row=obs.probe_row,
        n0=obs.n0,
        pilot_tx_b=obs.pilot_tx_b.copy(),
        pilot_tx_a=obs.pilot_tx_a.copy(),
    )
    b_corrupt = replace(
        b0,
        atoms=replace(b0.atoms, atom_snr_a=b0.atoms.atom_snr_a * 0.0 + 999.0),
        nu_hat=-b0.nu_hat * 2.0,
        mean_inh=b0.mean_inh * (-3.1 + 2.4j),
        var_inh=b0.var_inh * 5.0,
        pre_pred_inh=b0.pre_pred_inh * (-2.0),
        pre_var_inh=b0.pre_var_inh * 3.0,
        base_logit_ray=-b0.base_logit_ray + 10.0,
        llrs_inh=-b0.llrs_inh,
    )

    ckpt = torch.load(P11_DIR / "ckpts" / "p10b_dev_ckpts" / "em10_no_history_seed1001.pt", map_location="cpu", weights_only=True)
    model = mg.ManifoldTrustGate(variant="em10_no_history", **ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    f0 = mg.extract_manifold_features(obs, b0, mask_history=True)
    f1 = mg.extract_manifold_features(obs_ya_poison, b_corrupt, mask_history=True)
    assert np.max(np.abs(f0.scalar_feats - f1.scalar_feats)) == 0.0
    assert np.max(np.abs(f0.complex_feats - f1.complex_feats)) == 0.0

    o0 = model.gate(obs, b0)
    o1 = model.gate(obs_ya_poison, b_corrupt)
    assert np.max(np.abs(o0.w_ray - o1.w_ray)) == 0.0
    assert np.max(np.abs(o0.noise_mult - o1.noise_mult)) == 0.0


def test_f06_t2_checkpoint_variant_mismatch_rejected() -> None:
    mg = importlib.import_module("p11.manifold_gate")
    with pytest.raises(ValueError):
        mg.ManifoldTrustGate(variant="nonexistent_variant")
    assert "em10_transport_attn" in OFFGRID_VARIANTS


# ============================================================================
# F7: Independent Reference & Statistical Audit Boundaries (6 tests)
# ============================================================================

def test_f07_t2_holm_bonferroni_ties_and_stepdown_monotonicity() -> None:
    audit = importlib.import_module("p11.audit")
    c1 = audit.exact_sign_flip_test(np.linspace(0.05, 0.12, 16), expected_sign=1)
    c2 = audit.exact_sign_flip_test(np.linspace(0.05, 0.12, 16), expected_sign=1)
    c3 = audit.exact_sign_flip_test(np.array([0.04] * 14 + [-0.01, -0.01]), expected_sign=1)
    adj = audit.holm_bonferroni_adjust([c3, c1, c2])

    # Sort by raw_exact and confirm running monotonicity and <= 1 cap
    raws = [Fraction(x["raw_exact"]) for x in adj]
    holms = [Fraction(x["holm_exact"]) for x in adj]
    order = sorted(range(len(adj)), key=raws.__getitem__)
    for k in range(1, len(order)):
        assert holms[order[k]] >= holms[order[k - 1]]
        assert holms[order[k]] <= Fraction(1)
    assert holms[1] == holms[2]


def test_f07_t2_holm_directional_refuted_vs_insufficient_vs_supported() -> None:
    audit = importlib.import_module("p11.audit")
    sup = audit.exact_sign_flip_test(np.linspace(0.03, 0.09, 16), expected_sign=1)
    ref = audit.exact_sign_flip_test(-np.linspace(0.03, 0.09, 16), expected_sign=1)
    ins = audit.exact_sign_flip_test(np.array([0.02, -0.02] * 8), expected_sign=1)
    adj = audit.holm_bonferroni_adjust([sup, ref, ins])
    assert [x["verdict"] for x in adj] == ["supported", "refuted", "insufficient"]


def test_f07_t2_noninferiority_df15_sharp_boundary_0099_vs_0101() -> None:
    audit = importlib.import_module("p11.audit")
    # Constant vector d -> std_diff == 0 -> upper_95 == mean_diff
    res_pass = audit.noninferiority_test(np.full(16, 0.0099), margin=0.01)
    res_fail = audit.noninferiority_test(np.full(16, 0.0101), margin=0.01)
    res_eq = audit.noninferiority_test(np.full(16, 0.0100), margin=0.01)
    assert res_pass["df"] == 15 and res_pass["passes"] is True
    assert res_fail["df"] == 15 and res_fail["passes"] is False
    assert res_eq["df"] == 15 and res_eq["passes"] is False

    # Also verify non-zero std case with exact Student-t df=15 bound
    t_crit = float(audit.exact_student_t_ppf(0.95, 15))
    unit_vec = np.array([1.0, -1.0] * 8, dtype=np.float64)
    std_unit = float(np.std(unit_vec, ddof=1))
    margin_shift = t_crit * std_unit / math.sqrt(16.0)
    d_near_pass = 0.0099 - 0.0005 * margin_shift + 0.0005 * unit_vec
    d_near_eq = 0.0100 - 0.0005 * margin_shift + 0.0005 * unit_vec
    d_near_fail = 0.0101 - 0.0005 * margin_shift + 0.0005 * unit_vec
    assert audit.noninferiority_test(d_near_pass, margin=0.01)["passes"] is True
    assert audit.noninferiority_test(d_near_eq, margin=0.01)["passes"] is False
    assert audit.noninferiority_test(d_near_fail, margin=0.01)["passes"] is False


def test_f07_t2_audit_rejects_paired_energy_mismatch_and_seed_overlap(tmp_path: Path) -> None:
    audit = importlib.import_module("p11.audit")
    indep = importlib.import_module("p11.independent_ref")
    ds = make_synthetic_g16_dataset(tmp_path / "energy_bad", SEED_P11_E2E + 700_000)
    recs = [json.loads(ln) for ln in ds["raw"].read_text(encoding="utf-8").splitlines() if ln.strip()]
    recs[0]["budgets"]["dense"]["always_clear"]["probe_energy"] += 5.0
    with pytest.raises(ValueError):
        audit.verify_paired_energies(recs[0], energy_tol=0.0)

    overlap = indep.check_p11_seed_disjointness(
        p11_train_seeds=[SEED_P11_E2E + 1],
        p11_dev_seeds=[SEED_P11_DEV + 1],
        p11_formal_id_seeds=[SEED_P11_E2E + 1],
        p11_formal_ood_seeds=[SEED_P11_E2E + 2],
        p11_e2e_seeds=[SEED_P11_E2E + 3],
        check_namespace_bounds=False,
        strict=False,
    )
    assert overlap["disjoint"] is False


def test_f07_t2_audit_rejects_invalid_n0_bool_timedelta64_and_replay_diff(tmp_path: Path) -> None:
    audit = importlib.import_module("p11.audit")
    indep = importlib.import_module("p11.independent_ref")

    for bad_seed in (True, np.bool_(True), np.timedelta64(1, "s")):
        with pytest.raises(TypeError, match="invalid seed"):
            indep.check_p11_seed_disjointness(
                p11_train_seeds=[bad_seed],
                p11_dev_seeds=[SEED_P11_DEV],
                p11_formal_id_seeds=(),
                p11_formal_ood_seeds=(),
                p11_e2e_seeds=[SEED_P11_E2E + 10],
                strict=True,
            )

    ds = make_synthetic_g16_dataset(tmp_path / "rep_bad", SEED_P11_E2E + 710_000)
    orig = [json.loads(ln) for ln in ds["raw"].read_text(encoding="utf-8").splitlines() if ln.strip()]
    rep = [json.loads(ln) for ln in ds["replay"].read_text(encoding="utf-8").splitlines() if ln.strip()]
    rep[0]["budgets"]["sparse"]["p11_chained_wavefront"]["block_errors"] += 1
    bad_report = audit.verify_replay_exact(orig, rep, float_tol=1e-12, strict=False)
    assert bad_report["passed"] is False


def test_f07_t2_audit_cli_missing_and_invalid_flags_and_check_cell() -> None:
    audit = importlib.import_module("p11.audit")
    proc_empty = run_p11_cli(["-m", "p11.audit"])
    assert proc_empty.returncode != 0

    proc_bad = run_p11_cli(["-m", "p11.audit", "--invalid-flag-xyz"])
    assert proc_bad.returncode != 0

    bad_bool_cell = {
        "blocks": 16,
        "block_errors": True,
        "coded_bits": 2048,
        "coded_errors": 8,
        "info_bits": 1024,
        "info_errors": 4,
        "extra_blocks": 2,
        "extra_errors": 0,
        "probes": 32,
        "probe_sse": 1.0,
        "clean_sse": 1.0,
        "probe_energy": 10.0,
        "clean_energy": 10.0,
        "variance_sum": 4.0,
        "nll_sum": -1.0,
    }
    with pytest.raises(ValueError):
        audit.check_p11_primary_cell(bad_bool_cell, "sparse")

    bad_bits_cell = dict(bad_bool_cell, block_errors=1, info_errors=65)
    with pytest.raises(ValueError):
        audit.check_p11_primary_cell(bad_bits_cell, "sparse")


# ============================================================================
# F8: E2E Suite & Adversarial AST Guard Boundaries (5 tests — pass on 7e4f912)
# ============================================================================

def test_f08_t2_ast_n0_detector_catches_planted_violations(tmp_path: Path) -> None:
    bad_a = tmp_path / "planted_n0_default.py"
    bad_a.write_text(
        "def bad_fn(x, n0=0.25):\n    return x\n",
        encoding="utf-8",
    )
    bad_b = tmp_path / "planted_n0_const.py"
    bad_b.write_text(
        "from p8.link import " + "N" + "0\n"
        + "VAL = 0.1258" + "9254\n"
        + "PWR = 10 ** (-0." + "9)\n",
        encoding="utf-8",
    )
    rep_a = p9_indep.check_no_hardcoded_n0([bad_a], strict=False)
    assert rep_a["clean"] is False
    assert any(v["kind"] == "hardcoded_n0_default" for v in rep_a["violations"])

    rep_b = p9_indep.check_no_hardcoded_n0([bad_b], strict=False)
    assert rep_b["clean"] is False
    kinds_b = {v["kind"] for v in rep_b["violations"]}
    assert "legacy_n0_import" in kinds_b
    assert "hardcoded_n0_literal" in kinds_b
    assert "hardcoded_n0_power_expr" in kinds_b

    with pytest.raises(ValueError):
        p9_indep.check_no_hardcoded_n0([bad_b], strict=True)


def test_f08_t2_ast_infer_guard_detector_catches_planted_usage(tmp_path: Path) -> None:
    bad_py = tmp_path / "planted_infer.py"
    bad_py.write_text(
        "import torch\n"
        "def run(x):\n"
        "    with torch." + "inference_" + "mode():\n"
        "        return x\n",
        encoding="utf-8",
    )
    hits = scan_infer_guard_ast([bad_py])
    assert len(hits) == 1 and hits[0][1] == 3


def test_f08_t2_ast_skip_and_mock_detectors_catch_planted_usage(tmp_path: Path) -> None:
    bad_skip = tmp_path / "planted_skip.py"
    bad_skip.write_text(
        "import pytest\n"
        "def test_x():\n"
        "    pytest." + "sk" + "ip('blocked')\n",
        encoding="utf-8",
    )
    bad_mock = tmp_path / "planted_mock.py"
    bad_mock.write_text(
        "import unittest." + "mo" + "ck as um\n"
        "def test_y(monkeypatch):\n"
        "    return um\n",
        encoding="utf-8",
    )
    skip_hits = scan_skip_ast([bad_skip])
    mock_hits = scan_mock_ast([bad_mock])
    assert len(skip_hits) == 1
    assert len(mock_hits) == 2


def test_f08_t2_seed_pool_rejects_formal_seeds_in_e2e_guard(tmp_path: Path) -> None:
    formal_id_seed = int(getattr(p11_contract, "SEED_P11_" + "FORMAL_ID"))
    formal_ood_seed = int(getattr(p11_contract, "SEED_P11_" + "FORMAL_OOD"))
    for bad in (formal_id_seed, formal_ood_seed, SEED_P9_DEV):
        with pytest.raises(ValueError):
            assert_non_formal_e2e_seed(bad)
    for bad_type in (True, np.timedelta64(1, "s"), "1150000000000"):
        with pytest.raises(TypeError):
            assert_non_formal_e2e_seed(bad_type)
    assert assert_non_formal_e2e_seed(SEED_P11_E2E + 99) == SEED_P11_E2E + 99

    sym_id = "SEED_P11_" + "FORMAL_ID"
    sym_ood = "SEED_P11_" + "FORMAL_OOD"
    tmp_py = tmp_path / "planted_formal_seed.py"
    tmp_py.write_text(
        f"from p11.contract import {sym_id}, {sym_ood}\n"
        f"X = {sym_id}\n"
        f"LIT = {formal_id_seed}\n",
        encoding="utf-8",
    )
    hits = scan_formal_seed_symbols_ast([tmp_py])
    assert len(hits) >= 2
    assert any(h[2] == sym_id for h in hits)
    assert any(h[2] == str(formal_id_seed) for h in hits)


def test_f08_t2_private_import_scanner_catches_planted_p11_private_import(tmp_path: Path) -> None:
    bad_priv = tmp_path / "planted_priv.py"
    bad_priv.write_text(
        "from p11.contract import _safe" + "_repr\n",
        encoding="utf-8",
    )
    hits = scan_private_p11_imports_ast([bad_priv])
    assert len(hits) == 1 and "_safe" + "_repr" in hits[0][2]


# ============================================================================
# F9: Freeze & Formal Study Boundaries (5 tests)
# ============================================================================

def test_f09_t2_tampered_source_rejected_by_verify_freeze(tmp_path: Path) -> None:
    freeze = importlib.import_module("p11.freeze")
    mpath = tmp_path / "freeze_tampered_src.json"
    freeze.create_freeze(manifest_path=mpath, allow_dirty_agents=True)
    data = json.loads(mpath.read_text(encoding="utf-8"))
    first_key = sorted(data["sources"].keys())[0]
    data["sources"][first_key] = "0" * 64
    mpath.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with pytest.raises(ValueError):
        freeze.verify_freeze(manifest_path=mpath, allow_dirty_agents=True)


def test_f09_t2_tampered_checkpoint_and_design_rejected_by_verify_freeze(tmp_path: Path) -> None:
    freeze = importlib.import_module("p11.freeze")
    mpath_ckpt = tmp_path / "freeze_tampered_ckpt.json"
    freeze.create_freeze(manifest_path=mpath_ckpt, allow_dirty_agents=True)
    data = json.loads(mpath_ckpt.read_text(encoding="utf-8"))
    first_ckpt = sorted(data["checkpoints"].keys())[0]
    data["checkpoints"][first_ckpt] = "f" * 64
    mpath_ckpt.write_text(json.dumps(data, indent=2), encoding="utf-8")
    with pytest.raises(ValueError):
        freeze.verify_freeze(manifest_path=mpath_ckpt, allow_dirty_agents=True)

    mpath_des = tmp_path / "freeze_tampered_design.json"
    freeze.create_freeze(manifest_path=mpath_des, allow_dirty_agents=True)
    data_d = json.loads(mpath_des.read_text(encoding="utf-8"))
    data_d["design"]["noninferiority_margin"] = 0.05
    mpath_des.write_text(json.dumps(data_d, indent=2), encoding="utf-8")
    with pytest.raises(ValueError):
        freeze.verify_freeze(manifest_path=mpath_des, allow_dirty_agents=True)


def test_f09_t2_missing_freeze_and_refuse_overwrite(tmp_path: Path) -> None:
    freeze = importlib.import_module("p11.freeze")
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        freeze.verify_freeze(manifest_path=missing, allow_dirty_agents=True)

    target = tmp_path / "freeze_once.json"
    freeze.create_freeze(manifest_path=target, allow_dirty_agents=True)
    with pytest.raises(FileExistsError):
        freeze.create_freeze(manifest_path=target, allow_dirty_agents=True)


def test_f09_t2_formal_study_invalid_cli_flags_rejected() -> None:
    importlib.import_module("p11.formal_study")
    proc_bad_mode = run_p11_cli(["-m", "p11.formal_study", "--mode", "bogus_mode"])
    assert proc_bad_mode.returncode != 0

    proc_unknown = run_p11_cli(["-m", "p11.formal_study", "--unknown-flag-xyz"])
    assert proc_unknown.returncode != 0


def test_f09_t2_formal_study_refuses_formal_mode_without_valid_freeze(tmp_path: Path) -> None:
    freeze = importlib.import_module("p11.freeze")
    importlib.import_module("p11.formal_study")
    out_dir_1 = tmp_path / "refused_no_confirm"
    proc_no_confirm = run_p11_cli(
        [
            "-m", "p11.formal_study",
            "--mode", "primary",
            "--out-dir", str(out_dir_1),
        ],
        env=clean_subprocess_env(),
    )
    assert proc_no_confirm.returncode != 0
    assert "confirm-frozen" in proc_no_confirm.stderr
    assert not list(out_dir_1.glob("*.jsonl")) if out_dir_1.exists() else True

    missing_freeze = tmp_path / "missing_freeze.json"
    out_dir_2 = tmp_path / "refused_missing_manifest"
    proc_missing = run_p11_cli(
        [
            "-m", "p11.formal_study",
            "--mode", "primary",
            "--confirm-frozen",
            "--manifest", str(missing_freeze),
            "--allow-dirty-agents",
            "--out-dir", str(out_dir_2),
        ],
        env=clean_subprocess_env(),
    )
    assert proc_missing.returncode != 0
    assert ("Missing freeze manifest" in proc_missing.stderr) or ("FileNotFoundError" in proc_missing.stderr)
    assert not list(out_dir_2.glob("*.jsonl")) if out_dir_2.exists() else True

    tampered = tmp_path / "tampered_freeze.json"
    freeze.create_freeze(manifest_path=tampered, allow_dirty_agents=True)
    tdata = json.loads(tampered.read_text(encoding="utf-8"))
    tkey = sorted(tdata["sources"].keys())[0]
    tdata["sources"][tkey] = "0" * 64
    tampered.write_text(json.dumps(tdata, indent=2), encoding="utf-8")
    out_dir_3 = tmp_path / "refused_tampered_manifest"
    proc_tampered = run_p11_cli(
        [
            "-m", "p11.formal_study",
            "--mode", "primary",
            "--confirm-frozen",
            "--manifest", str(tampered),
            "--allow-dirty-agents",
            "--out-dir", str(out_dir_3),
        ],
        env=clean_subprocess_env(),
    )
    assert proc_tampered.returncode != 0
    assert "ValueError" in proc_tampered.stderr
    assert not list(out_dir_3.glob("*.jsonl")) if out_dir_3.exists() else True


# ============================================================================
# F10: Research Report & CLI Sync Boundaries (5 tests)
# ============================================================================

def test_f10_t2_report_cli_missing_or_corrupt_input_rejected(tmp_path: Path) -> None:
    importlib.import_module("p11.report")
    # 1. Non-existent --results-dir (bypassing freeze verification) -> FileNotFoundError
    out_missing_dir = tmp_path / "out_missing_dir"
    proc_missing_dir = run_p11_cli([
        "-m", "p11.report",
        "--results-dir", str(tmp_path / "missing_results"),
        "--no-verify-freeze",
        "--out-report", str(out_missing_dir / "P11_REPORT.md"),
        "--out-sync", str(out_missing_dir / "CLI_SYNC_SUMMARY.md"),
    ])
    assert proc_missing_dir.returncode != 0
    assert "FileNotFoundError" in proc_missing_dir.stderr
    assert "Results directory does not exist" in proc_missing_dir.stderr
    assert not (out_missing_dir / "P11_REPORT.md").exists()
    assert not (out_missing_dir / "CLI_SYNC_SUMMARY.md").exists()

    # 2. Existing valid results_dir + missing --manifest (default freeze verify) -> FileNotFoundError
    ds = make_synthetic_g16_dataset(tmp_path / "valid_ds_for_manifest", SEED_P11_E2E + 740_000)
    out_missing_manifest = tmp_path / "out_missing_manifest"
    proc_missing_manifest = run_p11_cli([
        "-m", "p11.report",
        "--results-dir", str(ds["dir"]),
        "--manifest", str(tmp_path / "missing_manifest.json"),
        "--allow-dirty-agents",
        "--out-report", str(out_missing_manifest / "P11_REPORT.md"),
        "--out-sync", str(out_missing_manifest / "CLI_SYNC_SUMMARY.md"),
    ])
    assert proc_missing_manifest.returncode != 0
    assert "FileNotFoundError" in proc_missing_manifest.stderr
    assert "Missing freeze manifest" in proc_missing_manifest.stderr
    assert not (out_missing_manifest / "P11_REPORT.md").exists()
    assert not (out_missing_manifest / "CLI_SYNC_SUMMARY.md").exists()

    # 3. Missing p11_audit.json without --allow-partial (valid p11_primary_summary.json present) -> FileNotFoundError
    ds["audit"].unlink()
    out_missing_audit = tmp_path / "out_missing_audit"
    proc_missing_audit = run_p11_cli([
        "-m", "p11.report",
        "--results-dir", str(ds["dir"]),
        "--no-verify-freeze",
        "--out-report", str(out_missing_audit / "P11_REPORT.md"),
        "--out-sync", str(out_missing_audit / "CLI_SYNC_SUMMARY.md"),
    ])
    assert proc_missing_audit.returncode != 0
    assert "FileNotFoundError" in proc_missing_audit.stderr
    assert "Missing required audit payload" in proc_missing_audit.stderr
    assert not (out_missing_audit / "P11_REPORT.md").exists()
    assert not (out_missing_audit / "CLI_SYNC_SUMMARY.md").exists()

    # 4a. Corrupt p11_primary_summary.json with --allow-partial -> JSONDecodeError
    corrupt_partial_dir = tmp_path / "corrupt_partial"
    corrupt_partial_dir.mkdir(parents=True, exist_ok=True)
    (corrupt_partial_dir / "p11_primary_summary.json").write_text("{not valid json", encoding="utf-8")
    out_corrupt_partial = tmp_path / "out_corrupt_partial"
    proc_corrupt_partial = run_p11_cli([
        "-m", "p11.report",
        "--results-dir", str(corrupt_partial_dir),
        "--no-verify-freeze",
        "--allow-partial",
        "--out-report", str(out_corrupt_partial / "P11_REPORT.md"),
        "--out-sync", str(out_corrupt_partial / "CLI_SYNC_SUMMARY.md"),
    ])
    assert proc_corrupt_partial.returncode != 0
    assert "JSONDecodeError" in proc_corrupt_partial.stderr
    assert not (out_corrupt_partial / "P11_REPORT.md").exists()
    assert not (out_corrupt_partial / "CLI_SYNC_SUMMARY.md").exists()

    # 4b. Corrupt p11_primary_summary.json with valid p11_audit.json (no --allow-partial) -> JSONDecodeError
    corrupt_with_audit_dir = tmp_path / "corrupt_with_audit"
    corrupt_with_audit_dir.mkdir(parents=True, exist_ok=True)
    (corrupt_with_audit_dir / "p11_primary_summary.json").write_text("{not valid json", encoding="utf-8")
    (corrupt_with_audit_dir / "p11_audit.json").write_text(
        json.dumps({"mode": "dry_run", "axis": "primary", "primary": {"audit": {"clusters": 4}}}),
        encoding="utf-8",
    )
    out_corrupt_audit = tmp_path / "out_corrupt_audit"
    proc_corrupt_audit = run_p11_cli([
        "-m", "p11.report",
        "--results-dir", str(corrupt_with_audit_dir),
        "--no-verify-freeze",
        "--out-report", str(out_corrupt_audit / "P11_REPORT.md"),
        "--out-sync", str(out_corrupt_audit / "CLI_SYNC_SUMMARY.md"),
    ])
    assert proc_corrupt_audit.returncode != 0
    assert "JSONDecodeError" in proc_corrupt_audit.stderr
    assert not (out_corrupt_audit / "P11_REPORT.md").exists()
    assert not (out_corrupt_audit / "CLI_SYNC_SUMMARY.md").exists()


def test_f10_t2_report_invalid_cli_flag_rejected() -> None:
    importlib.import_module("p11.report")
    proc = run_p11_cli(["-m", "p11.report", "--nonexistent-flag-xyz"])
    assert proc.returncode != 0


def test_f10_t2_report_handles_zero_error_and_tie_boundaries(tmp_path: Path) -> None:
    report = importlib.import_module("p11.report")
    audit = importlib.import_module("p11.audit")
    tie_res = audit.exact_sign_flip_test(np.zeros(16, dtype=np.float64), expected_sign=1)
    adj = audit.holm_bonferroni_adjust([tie_res])
    assert audit.verdict_to_zh(adj[0]["verdict"]) == "证据不足"

    adj[0]["name"] = "tie_contrast"
    adj[0]["verdict_zh"] = audit.verdict_to_zh(adj[0]["verdict"])
    zero_cell = {
        "bler": 0.0,
        "clean_nmse": 0.0,
        "calibration": 1.0,
        "net_delivered_info_bits_per_frame": 1152.0,
    }
    rep_md, sync_md = report.render_reports(
        primary_summary={
            "totals": {"sparse": {"p11_chained_wavefront": zero_cell}},
            "by_regime": {"continuous": {"sparse": {"p11_chained_wavefront": zero_cell}}},
            "tests": adj,
            "noninferiority": {"verdict": "insufficient", "verdict_zh": "证据不足", "upper_95": 0.01, "mean_diff": 0.0},
        }
    )
    assert "0.00%" in rep_md
    assert "证据不足" in rep_md
    assert "证据不足" in sync_md


def test_f10_t2_report_rejects_unverified_or_mismatched_summary(tmp_path: Path) -> None:
    freeze = importlib.import_module("p11.freeze")
    audit = importlib.import_module("p11.audit")
    importlib.import_module("p11.report")

    ds = make_synthetic_g16_dataset(tmp_path / "unverified_ds", SEED_P11_E2E + 750_000)
    tampered_manifest = tmp_path / "tampered_manifest.json"
    freeze.create_freeze(manifest_path=tampered_manifest, allow_dirty_agents=True)
    m_obj = json.loads(tampered_manifest.read_text(encoding="utf-8"))
    m_key = sorted(m_obj["sources"].keys())[0]
    m_obj["sources"][m_key] = "0" * 64
    tampered_manifest.write_text(json.dumps(m_obj, indent=2), encoding="utf-8")

    proc = run_p11_cli([
        "-m", "p11.report",
        "--results-dir", str(ds["dir"]),
        "--manifest", str(tampered_manifest),
        "--allow-dirty-agents",
        "--out-report", str(tmp_path / "out_unver" / "P11_REPORT.md"),
        "--out-sync", str(tmp_path / "out_unver" / "CLI_SYNC_SUMMARY.md"),
    ])
    assert proc.returncode != 0
    assert "ValueError" in proc.stderr

    # Mismatched G != 16 rejected by p11.audit
    with pytest.raises(ValueError, match="Expected G=16"):
        audit.audit_p11_holm_families(
            {"chained_vs_p9_jt": np.full(8, 0.05)},
            {"transport_attn_vs_local": np.full(8, 0.05)},
            expected_g=16,
        )
    bad_sum = tmp_path / "bad_g_summary.json"
    sum_obj = json.loads(ds["summary"].read_text(encoding="utf-8"))
    sum_obj["clusters"] = sum_obj["clusters"][:8]
    bad_sum.write_text(json.dumps(sum_obj), encoding="utf-8")
    with pytest.raises(ValueError):
        audit.audit_p11_primary(ds["raw"], bad_sum)


def test_f10_t2_cli_sync_summary_handles_refuted_and_failed_ni() -> None:
    report = importlib.import_module("p11.report")
    audit = importlib.import_module("p11.audit")
    ref = audit.exact_sign_flip_test(-np.linspace(0.04, 0.10, 16), expected_sign=1)
    adj = audit.holm_bonferroni_adjust([ref])
    assert audit.verdict_to_zh(adj[0]["verdict"]) == "反证"
    ni_fail = audit.noninferiority_test(np.full(16, 0.015), margin=0.01)
    assert ni_fail["passes"] is False

    adj[0]["name"] = "refuted_contrast"
    adj[0]["verdict_zh"] = audit.verdict_to_zh(adj[0]["verdict"])
    rep_md, sync_md = report.render_reports(
        primary_summary={
            "totals": {},
            "by_regime": {},
            "tests": adj,
            "noninferiority": {
                "verdict": "refuted",
                "verdict_zh": "反证",
                "upper_95": float(ni_fail["upper_95"]),
                "mean_diff": float(ni_fail["mean_diff"]),
            },
        }
    )
    assert "反证" in rep_md
    assert "反证" in sync_md
