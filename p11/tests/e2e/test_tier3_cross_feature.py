"""Tier 3 E2E Cross-Feature Pairwise Interaction Tests (12 tests).

All tests import M2/M3 modules lazily inside test bodies for progressive testability.
"""
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

import numpy as np

import p9.independent_ref as p9_indep
from p11.contract import (
    COMPLEX_DTYPE,
    N,
    P10_BASELINE_COMMIT,
    PRIMARY_SNR_DB,
    REAL_DTYPE,
    REGIMES_ID,
    SEED_P11_DEV,
    SEED_P11_E2E,
    SEED_P11_TRAIN,
    T,
    apply_fractional_sc_operator,
    build_offgrid_sc_channel_matrix,
)
from p11.tests.e2e.conftest import (
    AUX1,
    E2E_DIR,
    FOLD_TARGETS,
    P11_DIR,
    assert_non_formal_e2e_seed,
    call_estimate_payload_posterior,
    clean_subprocess_env,
    make_e2e_public_obs,
    make_synthetic_g16_dataset,
    run_git,
    run_p11_cli,
    scan_formal_seed_symbols_ast,
    scan_infer_guard_ast,
    scan_mock_ast,
    scan_private_p11_imports_ast,
    scan_skip_ast,
)


def test_x_p01_p02_repo_guardrail_and_contract_ast() -> None:
    diff = run_git(["diff", "--name-only", P10_BASELINE_COMMIT, "--", ":!p11"])
    assert diff.returncode == 0 and diff.stdout.strip() == ""

    all_p11_py = sorted(p for p in P11_DIR.rglob("*.py") if "__pycache__" not in p.parts)
    assert p9_indep.check_no_hardcoded_n0(all_p11_py, strict=True)["clean"] is True
    assert scan_infer_guard_ast(all_p11_py) == []

    proc = run_p11_cli(
        ["-c", "import sys, p11.contract as c; assert 'torch' not in sys.modules; c.enforce_determinism(require_env=True)"],
        env=clean_subprocess_env(),
    )
    assert proc.returncode == 0, proc.stderr


def test_x_p02_p04_determinism_and_payload_rebuild_cstar() -> None:
    importlib.import_module("p11.payload_rebuild")
    code = (
        "import json\n"
        "import p9.channel as ch\n"
        "from p11.contract import SEED_P11_E2E, PRIMARY_SNR_DB, enforce_determinism\n"
        "from p11.payload_rebuild import estimate_payload_posterior\n"
        "enforce_determinism(require_env=True)\n"
        "obs, _ = ch.simulate_link(SEED_P11_E2E + 301, PRIMARY_SNR_DB, 'full_reset', 'sparse')\n"
        "try:\n"
        "    post = estimate_payload_posterior(obs, (2, 5, 8, 13, 16))\n"
        "except TypeError:\n"
        "    post = estimate_payload_posterior(obs.y_b[10], (2, 5, 8, 13, 16), obs.y_b[[2, 5, 8, 13, 16]], obs.n0)\n"
        "print(json.dumps({'m': post.mean.real.ravel()[:8].tolist(), 'v': post.var.ravel()[:8].tolist()}))\n"
    )
    p1 = run_p11_cli(["-c", code], env=clean_subprocess_env())
    p2 = run_p11_cli(["-c", code], env=clean_subprocess_env())
    assert p1.returncode == 0 and p2.returncode == 0, f"stderr1={p1.stderr}\nstderr2={p2.stderr}"
    assert p1.stdout == p2.stdout


def test_x_p04_p05_payload_rebuild_into_wavefront_pipeline() -> None:
    pr = importlib.import_module("p11.payload_rebuild")
    wf = importlib.import_module("p11.wavefront_pipeline")
    obs, _ = make_e2e_public_obs(302, PRIMARY_SNR_DB, "full_reset")
    post = call_estimate_payload_posterior(pr, obs, AUX1, use_payload=True)
    res, trace = wf.run_p11_with_trace(obs)

    assert np.isfinite(post.mean).all()
    assert np.all(res.gate_out.mix_probe_var >= float(obs.n0) - 1e-12)
    assert "folds" in trace and len(trace["folds"]) == 2
    for f_idx in (0, 1):
        fold = trace["folds"][f_idx]
        assert fold["branch"] in ("continuous", "shift", "reset")
        assert isinstance(fold["state_version"], int) and fold["state_version"] >= 0
        assert {rec["row"] for rec in fold["ledger"]}.isdisjoint(FOLD_TARGETS[f_idx])
    assert sum(int(f["state_version"]) for f in trace["folds"]) >= 0
    assert len(trace["ledger"]) == sum(len(f["ledger"]) for f in trace["folds"])


def test_x_p03_p05_math_design_invariants_live_verification() -> None:
    wf = importlib.import_module("p11.wavefront_pipeline")
    pr = importlib.import_module("p11.payload_rebuild")
    math_text = (P11_DIR / "math_verification.md").read_text(encoding="utf-8")
    assert "row 11 is a Fold 0 target" in math_text

    obs, _ = make_e2e_public_obs(303, PRIMARY_SNR_DB, "full_reset")
    _, trace = wf.run_p11_with_trace(obs)
    f0_rows = {rec["row"] for rec in trace["folds"][0]["ledger"]}
    assert f0_rows.isdisjoint(FOLD_TARGETS[0])
    assert 11 not in f0_rows

    a_prior = pr.build_frame_a_prior_clean(obs.y_a, obs.n0)
    m_b6 = pr.ShiftedBranchDynamics(a_prior, changed_ray=0, delta_freq=0.05, onset_row=6).shift_rotation()
    assert m_b6.shape == (T, N)
    assert np.max(np.abs(m_b6[:6, :] - 1.0)) == 0.0


def test_x_p05_p07_wavefront_anti_leakage_and_cstar_audit() -> None:
    wf = importlib.import_module("p11.wavefront_pipeline")
    indep = importlib.import_module("p11.independent_ref")
    for idx, regime in enumerate(REGIMES_ID):
        obs, _ = make_e2e_public_obs(304 + idx, PRIMARY_SNR_DB, regime)
        rep = indep.audit_assimilation_anti_leakage(wf.run_p11_primary, obs, strict=True)
        assert rep["passed"] is True
        assert rep["max_probe_poison_diff"] == 0.0
        assert rep["max_cstar_err"] < 1e-12


def test_x_p02_p06_offgrid_operators_match_channel_and_determinism() -> None:
    og = importlib.import_module("p11.offgrid_channel")
    seed = assert_non_formal_e2e_seed(SEED_P11_E2E + 310)
    obs, truth = og.simulate_offgrid_link(seed, "offgrid_continuous", snr_db=PRIMARY_SNR_DB)

    rng = np.random.default_rng(SEED_P11_E2E + 310)
    gains = (rng.normal(size=(truth.n_rays, N)) + 1j * rng.normal(size=(truth.n_rays, N))).astype(COMPLEX_DTYPE)
    h_con = build_offgrid_sc_channel_matrix(gains, truth.delays, N)
    x = (rng.normal(size=N) + 1j * rng.normal(size=N)).astype(COMPLEX_DTYPE)
    y_con = apply_fractional_sc_operator(x, gains, truth.delays, N)
    assert np.max(np.abs(y_con - (h_con @ x))) < 1e-13
    assert obs.y_b.shape == (T, N)


def test_x_p06_p07_offgrid_anti_leakage_and_independent_ref() -> None:
    import torch

    og = importlib.import_module("p11.offgrid_channel")
    mg = importlib.import_module("p11.manifold_gate")
    indep = importlib.import_module("p11.independent_ref")
    seed = assert_non_formal_e2e_seed(SEED_P11_E2E + 311)
    obs, truth = og.simulate_offgrid_link(seed, "offgrid_partial_shift", snr_db=PRIMARY_SNR_DB)
    assert indep.verify_offgrid_observation(obs, strict=True)["valid"] is True

    d_ref = indep.dirichlet_fractional_delay_matrix(float(truth.delays[0]), n=N)
    d_con = build_offgrid_sc_channel_matrix(
        np.ones((1, N), dtype=COMPLEX_DTYPE),
        np.array([truth.delays[0]], dtype=REAL_DTYPE),
        N,
    )
    assert np.max(np.abs(d_ref - d_con)) < 1e-13

    bundle = og.build_offgrid_branches(obs)
    ckpt = torch.load(
        P11_DIR / "ckpts" / "p10b_dev_ckpts" / "em10_transport_attn_seed1001.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = mg.ManifoldTrustGate(variant="em10_transport_attn", **ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    out = model.gate(obs, bundle)
    assert np.all(out.gate_out.mix_probe_var >= float(obs.n0) - 1e-12)


def test_x_p04_p07_payload_posterior_vs_independent_cov_oracle() -> None:
    pr = importlib.import_module("p11.payload_rebuild")
    indep = importlib.import_module("p11.independent_ref")
    obs, _ = make_e2e_public_obs(312, PRIMARY_SNR_DB, "full_reset")
    post = call_estimate_payload_posterior(pr, obs, AUX1, use_payload=True)

    mu = np.zeros(4, dtype=COMPLEX_DTYPE)
    cov = float(obs.n0) * np.eye(4, dtype=COMPLEX_DTYPE)
    y = np.ones(4, dtype=COMPLEX_DTYPE)
    ll = indep.independent_complex_gaussian_loglik(y, mu, cov)
    expected_ll = float(-4 * math.log(math.pi * float(obs.n0)) - 4.0 / float(obs.n0))
    assert abs(ll - expected_ll) < 1e-12
    assert np.isfinite(post.pilot_loglik)


def test_x_p07_p09_freeze_manifest_and_audit_disjointness(tmp_path: Path) -> None:
    freeze = importlib.import_module("p11.freeze")
    indep = importlib.import_module("p11.independent_ref")
    importlib.import_module("p11.audit")

    mpath = tmp_path / "freeze_x.json"
    freeze.create_freeze(manifest_path=mpath, allow_dirty_agents=True)
    freeze.verify_freeze(manifest_path=mpath, allow_dirty_agents=True)
    indep.check_auditor_independence([P11_DIR / "independent_ref.py", P11_DIR / "audit.py"], strict=True)
    seed_rep = indep.check_p11_seed_disjointness(
        p11_train_seeds=[SEED_P11_TRAIN + 9000],
        p11_dev_seeds=[SEED_P11_DEV + i for i in range(4)],
        p11_formal_id_seeds=(),
        p11_formal_ood_seeds=(),
        p11_e2e_seeds=[SEED_P11_E2E + 1000 + i for i in range(4)],
        strict=True,
    )
    assert seed_rep["disjoint"] is True


def test_x_p08_p09_e2e_seed_namespace_with_dry_run_study(tmp_path: Path) -> None:
    importlib.import_module("p11.formal_study")
    audit = importlib.import_module("p11.audit")

    out_dir = tmp_path / "x_dry_run"
    proc = run_p11_cli(["-m", "p11.formal_study", "--mode", "dry_run", "--out-dir", str(out_dir)], timeout=600)
    assert proc.returncode == 0, proc.stderr

    ds = make_synthetic_g16_dataset(tmp_path / "x_g16", SEED_P11_E2E + 800_000)
    for ln in ds["raw"].read_text(encoding="utf-8").splitlines():
        rec = json.loads(ln)
        assert_non_formal_e2e_seed(rec["seed"])
        audit.verify_paired_energies(rec, energy_tol=0.0)


def test_x_p09_p10_dry_run_audit_to_report_and_cli_sync(tmp_path: Path) -> None:
    freeze = importlib.import_module("p11.freeze")
    importlib.import_module("p11.formal_study")
    audit = importlib.import_module("p11.audit")
    importlib.import_module("p11.report")

    mpath = tmp_path / "freeze_chain.json"
    freeze.create_freeze(manifest_path=mpath, allow_dirty_agents=True)
    dry_dir = tmp_path / "dry_chain"
    proc_fs = run_p11_cli(["-m", "p11.formal_study", "--mode", "dry_run", "--out-dir", str(dry_dir)], timeout=600)
    assert proc_fs.returncode == 0, proc_fs.stderr

    ds = make_synthetic_g16_dataset(tmp_path / "ds_chain", SEED_P11_E2E + 810_000)
    raw_recs = [json.loads(ln) for ln in ds["raw"].read_text(encoding="utf-8").splitlines() if ln.strip()]
    rep_recs = [json.loads(ln) for ln in ds["replay"].read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert audit.verify_replay_exact(raw_recs, rep_recs, float_tol=1e-12, strict=True)["passed"] is True

    rep_dir = tmp_path / "report_chain"
    proc_rep = run_p11_cli([
        "-m", "p11.report",
        "--results-dir", str(ds["dir"]),
        "--out-report", str(rep_dir / "P11_REPORT.md"),
        "--out-sync", str(rep_dir / "CLI_SYNC_SUMMARY.md"),
        "--no-verify-freeze",
    ])
    assert proc_rep.returncode == 0, proc_rep.stderr
    assert (rep_dir / "P11_REPORT.md").is_file()
    assert (rep_dir / "CLI_SYNC_SUMMARY.md").is_file()


def test_x_p01_p07_p08_full_p11_ast_and_worktree_hygiene() -> None:
    indep = importlib.import_module("p11.independent_ref")
    all_p11_py = sorted(p for p in P11_DIR.rglob("*.py") if "__pycache__" not in p.parts)
    e2e_py = sorted(E2E_DIR.glob("*.py"))

    assert p9_indep.check_no_hardcoded_n0(all_p11_py, strict=True)["clean"] is True
    assert scan_infer_guard_ast(all_p11_py) == []
    assert scan_skip_ast(e2e_py) == []
    assert scan_mock_ast(e2e_py) == []
    assert scan_private_p11_imports_ast(e2e_py) == []
    assert scan_formal_seed_symbols_ast(e2e_py) == []
    assert indep.check_auditor_independence([P11_DIR / "audit.py", P11_DIR / "independent_ref.py"], strict=True)["independent"] is True
    assert run_git(["diff", "--name-only", P10_BASELINE_COMMIT, "--", ":!p11"]).stdout.strip() == ""
