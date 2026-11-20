"""Tier 1 E2E Feature Coverage Tests (F1-F10, 55 tests).

F1-F3 and F8 pass immediately on M1 commit 7e4f912.
F4-F7 and F9-F10 import M2/M3 modules lazily inside test bodies (zero skips, zero mocks).
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from fractions import Fraction
import importlib
import json
import math
from pathlib import Path
import re

import numpy as np
import pytest

import p9.contract as p9_contract
import p9.independent_ref as p9_indep
import p11
import p11.contract as p11_contract
from p11.contract import (
    ALL_REGIMES,
    ANCHOR,
    COMPLEX_DTYPE,
    CORE_ROWS,
    CP,
    HISTORICAL_SEED_RANGES,
    LINK_DURATION,
    N,
    OFFGRID_ALL_REGIMES,
    OFFGRID_DELAY_MAX,
    OFFGRID_DELAY_MIN,
    OFFGRID_RAY_COUNTS,
    P10_BASELINE_COMMIT,
    P11_SEED_RANGES,
    P9_BASELINE_COMMIT,
    POWERS,
    PRIMARY_SNR_DB,
    PROBE_ROW,
    REAL_DTYPE,
    REGIMES_ID,
    SEED_P11_DEV,
    SEED_P11_E2E,
    SNR_GRID_DB,
    T,
    AssimilationResult,
    GateOutput,
    OffGridObservation,
    OffGridTruth,
    PerRayGateOutput,
    PublicObservation,
    apply_fractional_sc_operator,
    build_fractional_sc_matrix,
    build_offgrid_sc_channel_matrix,
    fractional_delay_freq_response,
    fractional_delay_kernel,
    fractional_delay_matrix,
    is_historical_seed,
    n0_from_snr,
    signed_subcarrier_indices,
    unitary_dft_matrix,
)
from p11.tests.e2e.conftest import (
    AUX1,
    AUX2,
    E2E_DIR,
    EXPECTED_BRANCH,
    FOLD_TARGETS,
    MATH_VERIFICATION_SHA256,
    OFFGRID_CKPT_SEEDS,
    OFFGRID_VARIANTS,
    OFFICIAL_MAIN_COMMIT,
    P11_DIR,
    REPO_ROOT,
    VALID_ROWS,
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
# F1: Git Branch & Repository Guardrail (5 tests)
# ============================================================================

def test_f01_t1_git_branch_and_base_commit() -> None:
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    assert branch.returncode == 0
    assert branch.stdout.strip() == EXPECTED_BRANCH

    ancestor = run_git(["merge-base", "--is-ancestor", P10_BASELINE_COMMIT, "HEAD"])
    assert ancestor.returncode == 0

    main_sha = run_git(["rev-parse", "main"])
    assert main_sha.returncode == 0
    assert main_sha.stdout.strip() == OFFICIAL_MAIN_COMMIT


def test_f01_t1_p4_to_p10_and_root_untouched() -> None:
    diff = run_git(["diff", "--name-only", P10_BASELINE_COMMIT, "--", ":!p11"])
    assert diff.returncode == 0
    assert diff.stdout.strip() == "", f"Unexpected modifications outside p11/: {diff.stdout}"


def test_f01_t1_all_tracked_new_files_under_p11() -> None:
    diff = run_git(["diff", "--name-only", P10_BASELINE_COMMIT])
    assert diff.returncode == 0
    changed = [line.strip() for line in diff.stdout.splitlines() if line.strip()]
    assert len(changed) >= 3
    assert all(path.startswith("p11/") for path in changed)


def test_f01_t1_p11_package_import_and_exports() -> None:
    assert hasattr(p11, "__doc__") and p11.__doc__
    assert len(p11_contract.__all__) >= 44
    for sym in p11_contract.__all__:
        assert hasattr(p11_contract, sym), f"Missing exported contract symbol: {sym}"
    assert PublicObservation is p9_contract.PublicObservation
    assert GateOutput is p9_contract.GateOutput
    assert ANCHOR == 10
    assert PROBE_ROW == 19
    assert len(CORE_ROWS) == 16


def test_f01_t1_baseline_commit_constants_match_git() -> None:
    p10_res = run_git(["rev-parse", P10_BASELINE_COMMIT])
    assert p10_res.returncode == 0
    assert p10_res.stdout.strip() == "9e9d93bcbf6345b8645314b223cd954b2f31e63a"
    assert isinstance(P9_BASELINE_COMMIT, str) and len(P9_BASELINE_COMMIT) == 40


# ============================================================================
# F2: Disjoint P11 Contract & Determinism (5 tests)
# ============================================================================

def test_f02_t1_seed_namespaces_disjoint_from_all_history() -> None:
    assert len(P11_SEED_RANGES) == 5
    assert len(HISTORICAL_SEED_RANGES) >= 24

    all_ranges = list(P11_SEED_RANGES.items()) + list(HISTORICAL_SEED_RANGES.items())
    for i in range(len(all_ranges)):
        n_a, (lo_a, hi_a) = all_ranges[i]
        assert lo_a < hi_a, f"Invalid interval {n_a}"
        for j in range(i + 1, len(all_ranges)):
            n_b, (lo_b, hi_b) = all_ranges[j]
            assert hi_a <= lo_b or hi_b <= lo_a, f"Seed overlap between {n_a} and {n_b}"

    for name, (lo, hi) in P11_SEED_RANGES.items():
        assert is_historical_seed(lo) is None, f"{name} start flagged historical"
        assert is_historical_seed(hi - 1) is None, f"{name} end flagged historical"
    assert assert_non_formal_e2e_seed(SEED_P11_E2E) == SEED_P11_E2E
    assert assert_non_formal_e2e_seed(SEED_P11_DEV) == SEED_P11_DEV


def test_f02_t1_enforce_determinism_strict_and_lazy_torch() -> None:
    script = (
        "import json, sys; "
        "import p11.contract as c; "
        "lazy_ok = ('torch' not in sys.modules); "
        "res = c.enforce_determinism(require_env=True); "
        "print(json.dumps({'lazy_ok': lazy_ok, 'res': res}))"
    )
    proc = run_p11_cli(["-c", script], env=clean_subprocess_env())
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout.strip())
    assert payload["lazy_ok"] is True
    assert payload["res"]["torch_num_threads"] == 1
    assert payload["res"]["deterministic_algorithms"] is True
    assert payload["res"]["openblas_num_threads"] == "1"
    assert payload["res"]["omp_num_threads"] == "1"
    assert payload["res"]["mkl_num_threads"] == "1"


def test_f02_t1_fractional_delay_unitarity_and_group_law() -> None:
    ident = np.eye(N, dtype=COMPLEX_DTYPE)
    f_mat = unitary_dft_matrix(N)
    assert np.max(np.abs(f_mat.conj().T @ f_mat - ident)) < 1e-14

    delays = (0.0, 0.37, 2.75, 6.5)
    for d in delays:
        d_mat = fractional_delay_matrix(d, N)
        d_kern = fractional_delay_kernel(d, N)
        assert d_mat.dtype == COMPLEX_DTYPE and d_mat.shape == (N, N)
        assert np.array_equal(d_mat, d_kern)
        assert np.max(np.abs(d_mat.conj().T @ d_mat - ident)) < 1e-14
        resp = fractional_delay_freq_response(d, N)
        diag_from_mat = np.diag(f_mat @ d_mat @ f_mat.conj().T)
        assert np.max(np.abs(diag_from_mat - resp)) < 1e-14

    d_sum = fractional_delay_matrix(0.37 + 2.75, N)
    d_prod = fractional_delay_matrix(0.37, N) @ fractional_delay_matrix(2.75, N)
    assert np.max(np.abs(d_prod - d_sum)) < 1e-13
    assert signed_subcarrier_indices(N).dtype == REAL_DTYPE


def test_f02_t1_sc_operator_equals_matrix_and_parseval() -> None:
    rng = np.random.default_rng(SEED_P11_E2E + 115)
    delays = np.array([0.0, 1.35, 4.62], dtype=REAL_DTYPE)
    gains = (rng.normal(size=(3, N)) + 1j * rng.normal(size=(3, N))).astype(COMPLEX_DTYPE)
    h_mat = build_offgrid_sc_channel_matrix(gains, delays, N)
    h_alias = build_fractional_sc_matrix(gains, delays, N)
    assert np.array_equal(h_mat, h_alias)

    x_1d = (rng.normal(size=N) + 1j * rng.normal(size=N)).astype(COMPLEX_DTYPE)
    y_op_1d = apply_fractional_sc_operator(x_1d, gains, delays, N)
    assert np.max(np.abs(y_op_1d - (h_mat @ x_1d))) < 1e-13

    x_2d = (rng.normal(size=(5, N)) + 1j * rng.normal(size=(5, N))).astype(COMPLEX_DTYPE)
    y_op_2d = apply_fractional_sc_operator(x_2d, gains, delays, N)
    assert np.max(np.abs(y_op_2d - (x_2d @ h_mat.T))) < 1e-13

    unit_gains = np.ones((1, N), dtype=COMPLEX_DTYPE)
    y_unit = apply_fractional_sc_operator(x_1d, unit_gains, np.array([2.31]), N)
    rel_err = abs(np.sum(np.abs(y_unit) ** 2) - np.sum(np.abs(x_1d) ** 2)) / np.sum(np.abs(x_1d) ** 2)
    assert rel_err < 1e-13


def test_f02_t1_dataclasses_and_dtypes_canonical() -> None:
    assert REAL_DTYPE is np.float64
    assert COMPLEX_DTYPE is np.complex128
    gate = GateOutput(
        history_weight=np.zeros(T, dtype=REAL_DTYPE),
        probe_weight=0.5,
        mix_llrs=np.zeros((T, N, 4), dtype=REAL_DTYPE),
        mix_probe_pred=np.zeros(N, dtype=COMPLEX_DTYPE),
        mix_probe_var=np.ones(N, dtype=REAL_DTYPE),
        mix_mean_ray=np.zeros((3, T, N), dtype=COMPLEX_DTYPE),
        mix_var_ray=np.ones((3, T, N), dtype=REAL_DTYPE),
    )
    assim = AssimilationResult(
        gate_out=gate,
        decoded_info={1: np.zeros(64, dtype=np.uint8)},
        assimilated_rows=(9,),
        row_weights={9: 0.5},
        row_margins={9: 6.2},
        n_rounds=4,
        elapsed_ms=1.0,
    )
    pr = PerRayGateOutput(
        gate_out=gate,
        ray_weights=np.zeros((3, T), dtype=REAL_DTYPE),
        subset_posteriors=None,
        subset_logliks=None,
    )
    og_obs = OffGridObservation(
        y_a=np.zeros((T, N), dtype=COMPLEX_DTYPE),
        y_b=np.zeros((T, N), dtype=COMPLEX_DTYPE),
        training_rows=(10,),
        probe_row=PROBE_ROW,
        n0=n0_from_snr(PRIMARY_SNR_DB),
        pilot_tx_b=np.ones((1, N), dtype=COMPLEX_DTYPE),
        pilot_tx_a=np.ones((4, N), dtype=COMPLEX_DTYPE),
    )
    og_truth = OffGridTruth(
        regime="offgrid_continuous",
        n_rays=3,
        delays=np.array([0.5, 2.0, 4.5]),
        powers=POWERS.copy(),
        freqs_a=np.zeros(3),
        freqs_b=np.zeros(3),
        chirps=np.zeros(3),
        changed_rays=(),
        shift_row=None,
        corrupted_rows=(),
        data_rows=CORE_ROWS,
        b_info=np.zeros((16, 64), dtype=np.uint8),
        b_code=np.zeros((16, 128), dtype=np.uint8),
        y_b_probe_noisy=np.zeros(N, dtype=COMPLEX_DTYPE),
        y_b_probe_clean=np.zeros(N, dtype=COMPLEX_DTYPE),
        h_b_true_mat=np.zeros((T, N, N), dtype=COMPLEX_DTYPE),
    )
    for obj, field in ((assim, "n_rounds"), (pr, "ray_weights"), (og_obs, "n0"), (og_truth, "n_rays")):
        with pytest.raises(FrozenInstanceError):
            setattr(obj, field, None)


# ============================================================================
# F3: Codex Mathematical Verification & Optimization (5 tests)
# ============================================================================

def test_f03_t1_math_verification_exists_and_nonempty() -> None:
    doc = P11_DIR / "math_verification.md"
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8")
    assert len(text.encode("utf-8")) > 30_000


def test_f03_t1_covers_sections_a_through_e() -> None:
    text = (P11_DIR / "math_verification.md").read_text(encoding="utf-8")
    for heading in ("## (a)", "## (b)", "## (c)", "## (d)", "## (e)"):
        assert heading in text, f"Missing section {heading} in math_verification.md"
    assert "c_B" in text
    assert "D_j" in text
    assert "M_a(t)" in text


def test_f03_t1_codex_session_provenance_and_verifiable_equations() -> None:
    text = (P11_DIR / "math_verification.md").read_text(encoding="utf-8")
    assert "J=160" in text
    assert "1.75305" in text
    assert "0.01" in text
    assert "2, 5, 8, 13, 16" in text
    assert "1, 4, 7, 12, 15" in text


def test_f03_t1_probe_variance_floor_and_mixture_law_specified() -> None:
    text = (P11_DIR / "math_verification.md").read_text(encoding="utf-8")
    assert "mix_probe_var >= n0" in text
    assert "v_p=n_0+" in text or "v_p = n_0 +" in text
    assert "|p_0-p_1|^2" in text


def test_f03_t1_math_verification_referenced_in_p11_and_untampered() -> None:
    doc = P11_DIR / "math_verification.md"
    ls = run_git(["ls-files", "--error-unmatch", "p11/math_verification.md"])
    assert ls.returncode == 0
    show = run_git(["show", "HEAD:p11/math_verification.md"])
    assert show.returncode == 0
    assert sha256_file(doc) == MATH_VERIFICATION_SHA256


# ============================================================================
# F4: C*-Equivariant Payload Covariance Reconstruction (6 tests)
# ============================================================================

def test_f04_t1_payload_posterior_shapes_dtypes_and_finite() -> None:
    payload_rebuild = importlib.import_module("p11.payload_rebuild")
    obs, _ = make_e2e_public_obs(1, PRIMARY_SNR_DB, "full_reset")
    post = call_estimate_payload_posterior(payload_rebuild, obs, AUX1, use_payload=True)

    mean = np.asarray(post.mean)
    var = np.asarray(post.var)
    assert mean.shape == (3, T, N) and mean.dtype == COMPLEX_DTYPE
    assert var.shape == (3, T, N) and var.dtype == REAL_DTYPE
    assert np.isfinite(mean).all()
    assert np.isfinite(var).all() and np.all(var >= 0.0)


def test_f04_t1_cstar_gauge_equivariance_random_scales() -> None:
    payload_rebuild = importlib.import_module("p11.payload_rebuild")
    obs, _ = make_e2e_public_obs(2, PRIMARY_SNR_DB, "full_reset")
    base = call_estimate_payload_posterior(payload_rebuild, obs, AUX1, use_payload=True)

    for alpha in (2.5 * np.exp(1j * 1.1), 0.4 * np.exp(-1j * 2.2)):
        alpha_c = complex(alpha)
        alpha_sq = float(abs(alpha_c) ** 2)
        y_b_s = (alpha_c * obs.y_b).astype(COMPLEX_DTYPE)
        y_b_s[PROBE_ROW] = complex(math.nan, math.nan)
        obs_s = clone_public_obs(
            obs,
            y_a=(alpha_c * obs.y_a).astype(COMPLEX_DTYPE),
            y_b=y_b_s,
            n0_val=alpha_sq * float(obs.n0),
        )
        scaled = call_estimate_payload_posterior(payload_rebuild, obs_s, AUX1, use_payload=True)
        mean_err = np.max(np.abs(scaled.mean - alpha_c * base.mean)) / max(1.0, float(np.max(np.abs(alpha_c * base.mean))))
        var_err = np.max(np.abs(scaled.var - alpha_sq * base.var)) / max(1.0, float(np.max(np.abs(alpha_sq * base.var))))
        assert mean_err < 1e-12
        assert var_err < 1e-12


def test_f04_t1_b_only_isolation_from_frame_a_and_targets() -> None:
    payload_rebuild = importlib.import_module("p11.payload_rebuild")
    obs, _ = make_e2e_public_obs(3, PRIMARY_SNR_DB, "full_reset")
    base = call_estimate_payload_posterior(payload_rebuild, obs, AUX1, use_payload=True)

    y_b_poison = obs.y_b.copy()
    for r in FOLD_TARGETS[0] + VALID_ROWS:
        y_b_poison[r] = (100.0 + r) * np.exp(1j * np.arange(N))
    y_b_poison[PROBE_ROW] = 1e6 - 1e6j
    obs_poison = clone_public_obs(
        obs,
        y_a=np.full_like(obs.y_a, 50.0 - 150.0j),
        y_b=y_b_poison,
    )
    poisoned = call_estimate_payload_posterior(payload_rebuild, obs_poison, AUX1, use_payload=True)
    assert np.max(np.abs(poisoned.mean - base.mean)) < 1e-12
    assert np.max(np.abs(poisoned.var - base.var)) < 1e-12


def test_f04_t1_reset_base_refit_identity_and_idempotence() -> None:
    payload_rebuild = importlib.import_module("p11.payload_rebuild")
    obs, truth = make_e2e_public_obs(4, PRIMARY_SNR_DB, "full_reset")
    post = call_estimate_payload_posterior(payload_rebuild, obs, AUX1, use_payload=True)

    assert hasattr(payload_rebuild, "refit_reset")
    j0 = payload_rebuild.refit_reset(post, obs.y_b[ANCHOR], {})
    m0, v0 = payload_rebuild.project_joint_moments(j0)
    assert np.max(np.abs(m0 - post.mean)) < 1e-12
    assert np.max(np.abs(v0 - post.var)) < 1e-12

    ledger = {9: (truth.b_symbols[17], obs.y_b[9], float(obs.n0) / 0.65)}
    refit_1 = payload_rebuild.refit_reset(post, obs.y_b[ANCHOR], ledger)
    refit_2 = payload_rebuild.refit_reset(post, obs.y_b[ANCHOR], dict(ledger))
    m1, v1 = payload_rebuild.project_joint_moments(refit_1)
    m2, v2 = payload_rebuild.project_joint_moments(refit_2)
    assert np.isfinite(m1).all() and np.isfinite(v1).all() and np.all(v1 >= 0.0)
    assert np.max(np.abs(m1 - m2)) < 1e-13
    assert np.max(np.abs(v1 - v2)) < 1e-13


def test_f04_t1_three_branch_common_units_and_shift_identity() -> None:
    payload_rebuild = importlib.import_module("p11.payload_rebuild")
    obs, _ = make_e2e_public_obs(5, PRIMARY_SNR_DB, "partial_shift")
    a_prior = payload_rebuild.build_frame_a_prior_clean(obs.y_a, obs.n0)

    dyn_zero = payload_rebuild.ShiftedBranchDynamics(a_prior, changed_ray=1, delta_freq=0.0, onset_row=0)
    m_zero = dyn_zero.shift_rotation()
    assert m_zero.shape == (T, N)
    assert np.max(np.abs(m_zero - 1.0)) < 1e-14

    dyn_b6 = payload_rebuild.ShiftedBranchDynamics(a_prior, changed_ray=1, delta_freq=0.05, onset_row=6)
    m_b6 = dyn_b6.shift_rotation()
    assert m_b6.shape == (T, N)
    assert np.max(np.abs(m_b6[:6, :] - 1.0)) == 0.0

    post = call_estimate_payload_posterior(payload_rebuild, obs, AUX1, use_payload=True)
    ev = payload_rebuild.score_branches_common_units(
        a_prior,
        post,
        obs.y_b[ANCHOR],
        AUX1,
        obs.y_b[list(AUX1)],
        float(obs.n0),
    )
    assert all(math.isfinite(float(v)) for v in (ev.b_c, ev.b_s, ev.b_r))


def test_f04_t1_state_clr_proj_epoch30_moments_finite() -> None:
    payload_rebuild = importlib.import_module("p11.payload_rebuild")
    obs, _ = make_e2e_public_obs(6, 12.0, "full_reset")
    post = call_estimate_payload_posterior(payload_rebuild, obs, AUX2, use_payload=True)
    st = post.state_clr_proj
    assert st.epoch == 30 or math.isclose(float(st.epoch), 30.0)
    assert st.log_weight.shape == (3, 128)
    assert st.mean.shape == (3, 128) and np.isfinite(st.mean).all()
    assert st.variance.shape == (3, 128) and np.isfinite(st.variance).all()
    assert np.all(st.variance > 0.0)


# ============================================================================
# F5: Chained Wavefront Assimilation & Shift Repair (6 tests)
# ============================================================================

def test_f05_t1_run_p11_primary_returns_valid_assimilation_result() -> None:
    wf = importlib.import_module("p11.wavefront_pipeline")
    for idx, regime in enumerate(REGIMES_ID):
        obs, _ = make_e2e_public_obs(10 + idx, PRIMARY_SNR_DB, regime)
        res = wf.run_p11_primary(obs)
        assert isinstance(res, AssimilationResult)
        assert isinstance(res.gate_out, GateOutput)
        assert np.isfinite(res.gate_out.mix_llrs).all()
        assert np.isfinite(res.gate_out.mix_probe_pred).all()
        assert np.isfinite(res.gate_out.mix_probe_var).all()
        assert np.isfinite(res.gate_out.mix_mean_ray).all()
        assert np.isfinite(res.gate_out.mix_var_ray).all()


def test_f05_t1_fold_isolation_wave1_helpers_disjoint_from_targets() -> None:
    wf = importlib.import_module("p11.wavefront_pipeline")
    obs, _ = make_e2e_public_obs(14, PRIMARY_SNR_DB, "full_reset")
    res, trace = wf.run_p11_with_trace(obs)
    assert isinstance(res, AssimilationResult)
    assert "folds" in trace and len(trace["folds"]) == 2
    assert trace["folds"][0]["branch"] in ("continuous", "shift", "reset")
    assert trace["folds"][1]["branch"] in ("continuous", "shift", "reset")
    f0_rows = {rec["row"] for rec in trace["folds"][0]["ledger"]}
    f1_rows = {rec["row"] for rec in trace["folds"][1]["ledger"]}
    assert f0_rows.isdisjoint(FOLD_TARGETS[0])
    assert 11 not in f0_rows, "Fold 0 MUST NOT use target row 11 as a Wave-1 helper"
    assert f1_rows.isdisjoint(FOLD_TARGETS[1])


def test_f05_t1_probe_exclusion_and_noise_floor_invariant() -> None:
    wf = importlib.import_module("p11.wavefront_pipeline")
    for idx, regime in enumerate(REGIMES_ID):
        obs, _ = make_e2e_public_obs(20 + idx, PRIMARY_SNR_DB, regime)
        res = wf.run_p11_primary(obs)
        assert PROBE_ROW not in res.assimilated_rows
        assert PROBE_ROW not in res.decoded_info
        assert PROBE_ROW not in res.row_weights
        assert PROBE_ROW not in res.row_margins
        assert np.all(res.gate_out.mix_probe_var >= float(obs.n0) - 1e-12)


def test_f05_t1_cstar_equivariance_full_wavefront_pipeline() -> None:
    wf = importlib.import_module("p11.wavefront_pipeline")
    obs, _ = make_e2e_public_obs(25, PRIMARY_SNR_DB, "partial_shift")
    base = wf.run_p11_primary(obs)

    alpha = 2.75 * complex(math.cos(1.17), math.sin(1.17))
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
    assert set(scaled.decoded_info.keys()) == set(base.decoded_info.keys())
    for r, bits in base.decoded_info.items():
        assert np.array_equal(scaled.decoded_info[r], bits)

    bg, sg = base.gate_out, scaled.gate_out
    assert np.max(np.abs(sg.mix_llrs - bg.mix_llrs)) < 1e-12
    assert np.max(np.abs(sg.history_weight - bg.history_weight)) < 1e-12
    assert abs(float(sg.probe_weight) - float(bg.probe_weight)) < 1e-12
    assert np.max(np.abs(sg.mix_probe_pred - alpha * bg.mix_probe_pred)) / max(1.0, float(np.max(np.abs(alpha * bg.mix_probe_pred)))) < 1e-12
    assert np.max(np.abs(sg.mix_probe_var - alpha_sq * bg.mix_probe_var)) / max(1.0, float(np.max(np.abs(alpha_sq * bg.mix_probe_var)))) < 1e-12
    assert np.max(np.abs(sg.mix_mean_ray - alpha * bg.mix_mean_ray)) / max(1.0, float(np.max(np.abs(alpha * bg.mix_mean_ray)))) < 1e-12
    assert np.max(np.abs(sg.mix_var_ray - alpha_sq * bg.mix_var_ray)) / max(1.0, float(np.max(np.abs(alpha_sq * bg.mix_var_ray)))) < 1e-12


def test_f05_t1_p11_dev_benchmark_script_and_artifact_valid() -> None:
    p11_dev = importlib.import_module("p11.p11_dev")
    assert callable(getattr(p11_dev, "main", None))
    dev_json = P11_DIR / "results" / "p11_dev.json"
    assert dev_json.is_file(), f"Missing dev artifact {dev_json}"
    data = json.loads(dev_json.read_text(encoding="utf-8"))
    for k in ("dev_seed_base", "summary", "paired_noninferiority_continuous_vs_dense"):
        assert k in data
    assert assert_non_formal_e2e_seed(int(data["dev_seed_base"])) >= SEED_P11_DEV
    pooled = float(data["summary"]["p11_primary"]["_aggregate"]["id_equal_weight_bler"])
    assert 0.0 <= pooled <= 1.0


def test_f05_t1_decoded_info_all_16_core_rows_and_valid_weights() -> None:
    wf = importlib.import_module("p11.wavefront_pipeline")
    obs, _ = make_e2e_public_obs(28, 12.0, "continuous")
    res = wf.run_p11_primary(obs)
    assert set(CORE_ROWS).issubset(res.decoded_info.keys())
    for r in CORE_ROWS:
        arr = np.asarray(res.decoded_info[r])
        assert arr.shape == (64,)
        assert np.isin(arr, (0, 1)).all()
    for w in res.row_weights.values():
        assert 0.0 <= float(w) <= 1.0


# ============================================================================
# F6: Secondary Axis Off-Grid Manifold & Checkpoints (6 tests)
# ============================================================================

def test_f06_t1_all_12_offgrid_checkpoints_exist_and_3153_params() -> None:
    import torch

    mg = importlib.import_module("p11.manifold_gate")
    ckpt_dir = P11_DIR / "ckpts" / "p10b_dev_ckpts"
    assert ckpt_dir.is_dir()
    for variant in OFFGRID_VARIANTS:
        for seed in OFFGRID_CKPT_SEEDS:
            path = ckpt_dir / f"{variant}_seed{seed}.pt"
            assert path.is_file(), f"Missing checkpoint: {path}"
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            assert ckpt["variant"] == variant
            assert int(ckpt["seed"]) == seed
            assert int(ckpt["real_parameters"]) == 3153
            model = mg.ManifoldTrustGate(variant=variant, **ckpt["arch"])
            model.load_state_dict(ckpt["state_dict"])
            assert model.real_parameters() == 3153


def test_f06_t1_simulate_offgrid_link_all_regimes_valid() -> None:
    og = importlib.import_module("p11.offgrid_channel")
    for idx, regime in enumerate(OFFGRID_ALL_REGIMES):
        seed = assert_non_formal_e2e_seed(SEED_P11_E2E + 100 + idx)
        obs, truth = og.simulate_offgrid_link(seed, regime, snr_db=PRIMARY_SNR_DB)
        assert isinstance(obs, OffGridObservation)
        assert isinstance(truth, OffGridTruth)
        assert obs.probe_row == PROBE_ROW
        assert np.isnan(obs.y_b[PROBE_ROW].real).all() and np.isnan(obs.y_b[PROBE_ROW].imag).all()
        assert truth.n_rays in OFFGRID_RAY_COUNTS
        assert np.all((truth.delays >= OFFGRID_DELAY_MIN) & (truth.delays <= OFFGRID_DELAY_MAX))


def test_f06_t1_offgrid_forward_and_3seed_ensemble_finite() -> None:
    import torch

    og = importlib.import_module("p11.offgrid_channel")
    mg = importlib.import_module("p11.manifold_gate")
    seed = assert_non_formal_e2e_seed(SEED_P11_E2E + 110)
    obs, _ = og.simulate_offgrid_link(seed, "offgrid_partial_shift", snr_db=PRIMARY_SNR_DB)
    bundle = og.build_offgrid_branches(obs)

    ckpt_dir = P11_DIR / "ckpts" / "p10b_dev_ckpts"
    for c_seed in OFFGRID_CKPT_SEEDS:
        ckpt = torch.load(ckpt_dir / f"em10_transport_attn_seed{c_seed}.pt", map_location="cpu", weights_only=True)
        model = mg.ManifoldTrustGate(variant="em10_transport_attn", **ckpt["arch"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        res = model.gate(obs, bundle)
        g = res.gate_out
        assert isinstance(g, GateOutput)
        assert np.isfinite(g.mix_llrs).all()
        assert np.isfinite(g.mix_probe_pred).all()
        assert np.all(g.mix_probe_var >= float(obs.n0) - 1e-12)
        assert np.all((g.history_weight >= 0.0) & (g.history_weight <= 1.0))


def test_f06_t1_offgrid_cstar_invariance_and_equivariance() -> None:
    import torch

    og = importlib.import_module("p11.offgrid_channel")
    mg = importlib.import_module("p11.manifold_gate")
    seed = assert_non_formal_e2e_seed(SEED_P11_E2E + 111)
    obs, _ = og.simulate_offgrid_link(seed, "offgrid_continuous", snr_db=PRIMARY_SNR_DB)
    alpha = 1.9 * complex(math.cos(0.8), math.sin(0.8))
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

    b_base = og.build_offgrid_branches(obs)
    b_scal = og.build_offgrid_branches(obs_s)
    ckpt_dir = P11_DIR / "ckpts" / "p10b_dev_ckpts"
    for variant in OFFGRID_VARIANTS:
        ckpt = torch.load(ckpt_dir / f"{variant}_seed1001.pt", map_location="cpu", weights_only=True)
        model = mg.ManifoldTrustGate(variant=variant, **ckpt["arch"])
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        r0 = model.gate(obs, b_base)
        r1 = model.gate(obs_s, b_scal)
        assert np.max(np.abs(r1.w_ray - r0.w_ray)) < 1e-12
        llr_err = float(np.max(np.abs(r1.gate_out.mix_llrs - r0.gate_out.mix_llrs))) / max(
            1.0, float(np.max(np.abs(r0.gate_out.mix_llrs)))
        )
        assert llr_err < 1e-12
        pred_ref = alpha * r0.gate_out.mix_probe_pred
        pred_err = float(np.max(np.abs(r1.gate_out.mix_probe_pred - pred_ref))) / max(
            1e-300, float(np.max(np.abs(pred_ref)))
        )
        assert pred_err < 1e-12
        var_ref = alpha_sq * r0.gate_out.mix_probe_var
        var_err = float(np.max(np.abs(r1.gate_out.mix_probe_var - var_ref))) / max(
            1e-300, float(np.max(np.abs(var_ref)))
        )
        assert var_err < 1e-12


def test_f06_t1_offgrid_ray_permutation_equivariance() -> None:
    import torch

    og = importlib.import_module("p11.offgrid_channel")
    mg = importlib.import_module("p11.manifold_gate")
    seed = assert_non_formal_e2e_seed(SEED_P11_E2E + 112)
    obs, _ = og.simulate_offgrid_link(seed, "offgrid_partial_shift", snr_db=PRIMARY_SNR_DB)
    b0 = og.build_offgrid_branches(obs)
    k_rays = int(b0.k_hat)
    perm = tuple(reversed(range(k_rays)))
    b_perm = og.permute_branch_bundle(b0, perm)

    ckpt = torch.load(P11_DIR / "ckpts" / "p10b_dev_ckpts" / "em10_transport_attn_seed1001.pt", map_location="cpu", weights_only=True)
    model = mg.ManifoldTrustGate(variant="em10_transport_attn", **ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    r0 = model.gate(obs, b0)
    rp = model.gate(obs, b_perm)
    assert np.max(np.abs(rp.w_ray - r0.w_ray[list(perm)])) < 1e-12
    assert np.max(np.abs(rp.gate_out.mix_llrs - r0.gate_out.mix_llrs)) < 1e-12
    assert np.max(np.abs(rp.gate_out.mix_probe_pred - r0.gate_out.mix_probe_pred)) < 1e-12
    assert np.max(np.abs(rp.gate_out.mix_probe_var - r0.gate_out.mix_probe_var)) < 1e-12


def test_f06_t1_untrained_manifold_gate_matches_classical_per_ray() -> None:
    og = importlib.import_module("p11.offgrid_channel")
    mg = importlib.import_module("p11.manifold_gate")
    seed = assert_non_formal_e2e_seed(SEED_P11_E2E + 113)
    obs, _ = og.simulate_offgrid_link(seed, "offgrid_continuous", snr_db=PRIMARY_SNR_DB)
    bundle = og.build_offgrid_branches(obs)

    classical = og.classical_per_ray_gate(obs, bundle)
    untrained = mg.ManifoldTrustGate(variant="em10_transport_attn")
    untrained.eval()
    neural = untrained.gate(obs, bundle)
    assert np.max(np.abs(neural.w_ray - classical.w_ray)) < 1e-13
    assert np.max(np.abs(neural.gate_out.mix_llrs - classical.gate_out.mix_llrs)) < 1e-13


# ============================================================================
# F7: Independent Reference, Audit & Statistical Engine (6 tests)
# ============================================================================

def test_f07_t1_independent_dirichlet_matches_contract_operators() -> None:
    indep = importlib.import_module("p11.independent_ref")
    for n_val in (31, 32):
        for delay in (0.0, 0.43, 2.81, 6.5):
            d_ref = indep.dirichlet_fractional_delay_matrix(delay, n=n_val)
            d_con = fractional_delay_matrix(delay, n=n_val)
            assert np.max(np.abs(d_ref - d_con)) < 1e-14


def test_f07_t1_audit_assimilation_anti_leakage_passes_primary() -> None:
    indep = importlib.import_module("p11.independent_ref")
    wf = importlib.import_module("p11.wavefront_pipeline")
    obs, _ = make_e2e_public_obs(150, PRIMARY_SNR_DB, "full_reset")
    report = indep.audit_assimilation_anti_leakage(wf.run_p11_primary, obs, strict=True)
    assert report["passed"] is True
    assert report["max_probe_poison_diff"] == 0.0
    assert report["max_cstar_err"] < 1e-12
    assert report["probe_excluded"] is True


def test_f07_t1_exact_g16_sign_flip_and_holm_bonferroni() -> None:
    audit = importlib.import_module("p11.audit")
    diffs_pos = np.linspace(0.05, 0.12, 16)
    res_pos = audit.exact_sign_flip_test(diffs_pos, expected_sign=1)
    assert res_pos["exact_patterns"] == 65536
    assert res_pos["raw_exact"] == "1/32768"

    c2 = audit.exact_sign_flip_test(np.linspace(0.02, 0.08, 16), expected_sign=1)
    adj = audit.holm_bonferroni_adjust([res_pos, c2])
    assert Fraction(adj[0]["holm_exact"]) == Fraction(2, 32768)
    assert Fraction(adj[1]["holm_exact"]) >= Fraction(adj[0]["holm_exact"])


def test_f07_t1_noninferiority_test_df15_and_audit_verdicts() -> None:
    audit = importlib.import_module("p11.audit")
    ni_pass = audit.noninferiority_test(np.full(16, 0.002), margin=0.01)
    assert ni_pass["df"] == 15
    assert ni_pass["passes"] is True

    c_sup = audit.exact_sign_flip_test(np.linspace(0.05, 0.12, 16), expected_sign=1)
    c_ref = audit.exact_sign_flip_test(-np.linspace(0.05, 0.12, 16), expected_sign=1)
    c_ins = audit.exact_sign_flip_test(np.array([0.01, -0.01] * 8), expected_sign=1)
    adj = audit.holm_bonferroni_adjust([c_sup, c_ref, c_ins])
    assert [x["verdict"] for x in adj] == ["supported", "refuted", "insufficient"]


def test_f07_t1_auditor_ast_independence_and_no_hardcoded_n0() -> None:
    indep = importlib.import_module("p11.independent_ref")
    importlib.import_module("p11.audit")
    paths = [P11_DIR / "independent_ref.py", P11_DIR / "audit.py"]
    indep_rep = indep.check_auditor_independence(paths, strict=True)
    assert indep_rep["independent"] is True
    n0_rep = p9_indep.check_no_hardcoded_n0(paths, strict=True)
    assert n0_rep["clean"] is True


def test_f07_t1_audit_selftest_cli_executes_and_passes(tmp_path: Path) -> None:
    importlib.import_module("p11.audit")
    out_json = tmp_path / "p11_audit_selftest.json"
    proc = run_p11_cli(["-m", "p11.audit", "--selftest", "--output", str(out_json)])
    assert proc.returncode == 0, proc.stderr
    assert out_json.is_file()
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report.get("status") == "PASS"


# ============================================================================
# F8: E2E Opaque-Box Suite & Adversarial AST Guards (5 tests)
# ============================================================================

def test_f08_t1_e2e_suite_discovers_all_4_tiers_and_min_counts() -> None:
    proc = run_p11_cli(["-m", "pytest", "-p", "no:cacheprovider", str(E2E_DIR), "--collect-only", "-q"])
    assert proc.returncode == 0, f"Collection failed:\n{proc.stdout}\n{proc.stderr}"

    lines = [ln.strip() for ln in proc.stdout.splitlines() if "::test_" in ln]
    t1 = [ln for ln in lines if "test_tier1_features.py::" in ln]
    t2 = [ln for ln in lines if "test_tier2_boundaries.py::" in ln]
    t3 = [ln for ln in lines if "test_tier3_cross_feature.py::" in ln]
    t4 = [ln for ln in lines if "test_tier4_scenarios.py::" in ln]

    assert len(t1) >= 50, f"Tier 1 count too low: {len(t1)}"
    assert len(t2) >= 50, f"Tier 2 count too low: {len(t2)}"
    assert len(t3) >= 10, f"Tier 3 count too low: {len(t3)}"
    assert len(t4) >= 5, f"Tier 4 count too low: {len(t4)}"
    assert len(lines) >= 115, f"Total E2E tests too low: {len(lines)}"


def test_f08_t1_closed_book_e2e_seeds_disjoint_from_formal() -> None:
    e2e_files = sorted(E2E_DIR.glob("*.py"))
    formal_hits = scan_formal_seed_symbols_ast(e2e_files)
    assert formal_hits == [], f"Forbidden formal seed references in E2E suite: {formal_hits}"

    assert assert_non_formal_e2e_seed(SEED_P11_E2E) == SEED_P11_E2E
    assert assert_non_formal_e2e_seed(SEED_P11_DEV) == SEED_P11_DEV


def test_f08_t1_zero_mock_and_opaque_box_imports_in_e2e() -> None:
    e2e_files = sorted(E2E_DIR.glob("*.py"))
    assert scan_mock_ast(e2e_files) == []
    assert scan_private_p11_imports_ast(e2e_files) == []


def test_f08_t1_e2e_zero_hardcoded_n0_and_zero_infer_guard() -> None:
    all_p11_py = sorted(p for p in P11_DIR.rglob("*.py") if "__pycache__" not in p.parts)
    n0_res = p9_indep.check_no_hardcoded_n0(all_p11_py, strict=True)
    assert n0_res["clean"] is True
    assert scan_infer_guard_ast(all_p11_py) == []


def test_f08_t1_no_pytest_skip_in_e2e_suite() -> None:
    e2e_files = sorted(E2E_DIR.glob("*.py"))
    assert scan_skip_ast(e2e_files) == []


# ============================================================================
# F9: Pre-Experiment Freeze & Formal Dry-Run / Replay (6 tests)
# ============================================================================

def test_f09_t1_freeze_create_and_verify_in_isolated_workdir(tmp_path: Path) -> None:
    freeze = importlib.import_module("p11.freeze")
    manifest_path = tmp_path / "freeze.json"
    digest = freeze.create_freeze(manifest_path=manifest_path, allow_dirty_agents=True)
    verified = freeze.verify_freeze(manifest_path=manifest_path, allow_dirty_agents=True)
    assert digest == verified
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert "sources" in data and "checkpoints" in data and "design" in data


def test_f09_t1_freeze_checks_inherited_clean_and_no_n0() -> None:
    freeze = importlib.import_module("p11.freeze")
    rep = freeze.assert_repo_guardrails(allow_dirty_agents=True)
    assert isinstance(rep, dict)
    assert rep["main_head"] == OFFICIAL_MAIN_COMMIT
    assert rep["branch"] != "main" and rep["branch"] == EXPECTED_BRANCH
    assert rep["p10_baseline_commit"] == P10_BASELINE_COMMIT
    assert rep["non_p11_diff_vs_9e9d93b"] == []
    assert rep["allow_dirty_agents"] is True


def test_f09_t1_formal_study_cli_dry_run_primary_and_secondary(tmp_path: Path) -> None:
    importlib.import_module("p11.formal_study")
    out_dir = tmp_path / "dry_run_out"
    proc = run_p11_cli(["-m", "p11.formal_study", "--mode", "dry_run", "--out-dir", str(out_dir)], timeout=600)
    assert proc.returncode == 0, proc.stderr
    assert out_dir.is_dir()
    assert (out_dir / "p11_primary_id_links.jsonl").is_file()
    assert (out_dir / "p11_secondary_links.jsonl").is_file()
    assert (out_dir / "p11_audit.json").is_file()


def test_f09_t1_formal_study_dry_run_audit_roundtrip(tmp_path: Path) -> None:
    importlib.import_module("p11.formal_study")
    audit = importlib.import_module("p11.audit")
    out_dir = tmp_path / "dry_run_roundtrip"
    proc = run_p11_cli(["-m", "p11.formal_study", "--mode", "dry_run", "--out-dir", str(out_dir)], timeout=600)
    assert proc.returncode == 0, proc.stderr

    pri_audit = audit.audit_p11_primary(
        out_dir / "p11_primary_id_links.jsonl",
        out_dir / "p11_primary_summary.json",
        out_dir / "p11_primary_replay.jsonl",
        require_canonical_family=True,
    )
    assert pri_audit["replay"]["passed"] is True
    assert pri_audit["max_aggregate_diff"] < 1e-12
    assert pri_audit["max_stat_diff"] < 1e-12

    sec_audit = audit.audit_p11_offgrid(
        out_dir / "p11_secondary_links.jsonl",
        out_dir / "p11_secondary_summary.json",
        out_dir / "p11_secondary_replay.jsonl",
        require_canonical_family=True,
    )
    assert sec_audit["replay"]["passed"] is True
    assert sec_audit["max_aggregate_diff"] < 1e-12
    assert sec_audit["max_stat_diff"] < 1e-12


def test_f09_t1_single_worker_replay_deterministic_parity(tmp_path: Path) -> None:
    importlib.import_module("p11.formal_study")
    audit = importlib.import_module("p11.audit")
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    proc_a = run_p11_cli(["-m", "p11.formal_study", "--mode", "dry_run", "--out-dir", str(dir_a)], timeout=600)
    proc_b = run_p11_cli(["-m", "p11.formal_study", "--mode", "dry_run", "--out-dir", str(dir_b)], timeout=600)
    assert proc_a.returncode == 0 and proc_b.returncode == 0

    for jpath_a in sorted(dir_a.glob("*.jsonl")):
        jpath_b = dir_b / jpath_a.name
        assert jpath_b.is_file()
        recs_a = [json.loads(ln) for ln in jpath_a.read_text(encoding="utf-8").splitlines() if ln.strip()]
        recs_b = [json.loads(ln) for ln in jpath_b.read_text(encoding="utf-8").splitlines() if ln.strip()]
        rep = audit.verify_replay_exact(recs_a, recs_b, float_tol=1e-12, strict=True)
        assert rep["passed"] is True


def test_f09_t1_freeze_records_all_12_checkpoints_and_p11_sources(tmp_path: Path) -> None:
    freeze = importlib.import_module("p11.freeze")
    manifest_path = tmp_path / "freeze_ckpts.json"
    freeze.create_freeze(manifest_path=manifest_path, allow_dirty_agents=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ckpts = manifest["checkpoints"]
    assert len(ckpts) == 12
    for variant in OFFGRID_VARIANTS:
        for seed in OFFGRID_CKPT_SEEDS:
            assert any(f"{variant}_seed{seed}.pt" in k for k in ckpts)
    assert any("p11/contract.py" in k for k in manifest["sources"])


# ============================================================================
# F10: Research Report & CLI Synchronization (5 tests)
# ============================================================================

def test_f10_t1_report_cli_generates_p11_report_and_cli_sync(tmp_path: Path) -> None:
    importlib.import_module("p11.report")
    ds = make_synthetic_g16_dataset(tmp_path / "report_in", SEED_P11_E2E + 600_000)
    out_dir = tmp_path / "report_out"
    proc = run_p11_cli([
        "-m", "p11.report",
        "--results-dir", str(ds["dir"]),
        "--out-report", str(out_dir / "P11_REPORT.md"),
        "--out-sync", str(out_dir / "CLI_SYNC_SUMMARY.md"),
        "--no-verify-freeze",
    ])
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "P11_REPORT.md").is_file()
    assert (out_dir / "CLI_SYNC_SUMMARY.md").is_file()


def test_f10_t1_report_emits_verbatim_zh_verdicts(tmp_path: Path) -> None:
    report = importlib.import_module("p11.report")
    audit = importlib.import_module("p11.audit")
    c_sup = audit.exact_sign_flip_test(np.linspace(0.05, 0.12, 16), expected_sign=1)
    c_ref = audit.exact_sign_flip_test(-np.linspace(0.05, 0.12, 16), expected_sign=1)
    c_ins = audit.exact_sign_flip_test(np.array([0.01, -0.01] * 8), expected_sign=1)
    adj = audit.holm_bonferroni_adjust([c_sup, c_ref, c_ins])
    verdicts = [audit.verdict_to_zh(item["verdict"]) for item in adj]
    assert verdicts == ["支持", "反证", "证据不足"]

    for item, vz in zip(adj, verdicts):
        item["name"] = f"contrast_{item['verdict']}"
        item["verdict_zh"] = vz
    rep_md, sync_md = report.render_reports(
        primary_summary={
            "totals": {},
            "by_regime": {},
            "tests": adj,
            "noninferiority": {"verdict": "supported", "verdict_zh": "支持", "upper_95": 0.002, "mean_diff": 0.0},
        }
    )
    for expected in ("支持", "反证", "证据不足"):
        assert expected in rep_md
        assert expected in sync_md


def test_f10_t1_cli_sync_summary_contains_primary_secondary_and_hashes(tmp_path: Path) -> None:
    importlib.import_module("p11.report")
    ds = make_synthetic_g16_dataset(tmp_path / "sync_in", SEED_P11_E2E + 610_000)
    out_dir = tmp_path / "sync_out"
    proc = run_p11_cli([
        "-m", "p11.report",
        "--results-dir", str(ds["dir"]),
        "--out-report", str(out_dir / "P11_REPORT.md"),
        "--out-sync", str(out_dir / "CLI_SYNC_SUMMARY.md"),
        "--no-verify-freeze",
    ])
    assert proc.returncode == 0, proc.stderr
    sync_text = (out_dir / "CLI_SYNC_SUMMARY.md").read_text(encoding="utf-8")
    assert len(sync_text.strip()) > 100
    assert P10_BASELINE_COMMIT in sync_text
    assert "p11_chained_wavefront" in sync_text
    assert any(v in sync_text for v in ("支持", "反证", "证据不足"))


def test_f10_t1_report_deterministic_idempotence(tmp_path: Path) -> None:
    importlib.import_module("p11.report")
    ds = make_synthetic_g16_dataset(tmp_path / "idem_in", SEED_P11_E2E + 620_000)
    out_a = tmp_path / "idem_a"
    out_b = tmp_path / "idem_b"
    for out_dir in (out_a, out_b):
        proc = run_p11_cli([
            "-m", "p11.report",
            "--results-dir", str(ds["dir"]),
            "--out-report", str(out_dir / "P11_REPORT.md"),
            "--out-sync", str(out_dir / "CLI_SYNC_SUMMARY.md"),
            "--no-verify-freeze",
        ])
        assert proc.returncode == 0, proc.stderr
    assert (out_a / "P11_REPORT.md").read_text(encoding="utf-8") == (out_b / "P11_REPORT.md").read_text(encoding="utf-8")
    assert (out_a / "CLI_SYNC_SUMMARY.md").read_text(encoding="utf-8") == (out_b / "CLI_SYNC_SUMMARY.md").read_text(encoding="utf-8")


def test_f10_t1_report_preserves_p9_p10_and_root_cleanliness(tmp_path: Path) -> None:
    importlib.import_module("p11.report")
    ds = make_synthetic_g16_dataset(tmp_path / "clean_in", SEED_P11_E2E + 630_000)
    out_dir = tmp_path / "clean_out"
    proc = run_p11_cli([
        "-m", "p11.report",
        "--results-dir", str(ds["dir"]),
        "--out-report", str(out_dir / "P11_REPORT.md"),
        "--out-sync", str(out_dir / "CLI_SYNC_SUMMARY.md"),
        "--no-verify-freeze",
    ])
    assert proc.returncode == 0, proc.stderr
    diff = run_git(["diff", "--name-only", P10_BASELINE_COMMIT, "--", ":!p11"])
    assert diff.returncode == 0 and diff.stdout.strip() == ""
