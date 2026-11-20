"""Tier 4 Realistic End-to-End User Workflow Scenarios (6 tests).

Exercises full Primary Axis, Secondary Axis, Dry-Run Freeze/Audit/Report,
Single-Worker Exact Replay, Repository Compliance, and OOD Stress workflows
strictly on SEED_P11_E2E (and read-only on SEED_P11_DEV outputs).
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
    CORE_ROWS,
    OFFGRID_REGIMES_ID,
    OFFGRID_REGIMES_OOD,
    P10_BASELINE_COMMIT,
    PRIMARY_SNR_DB,
    PROBE_ROW,
    REGIMES_ID,
    REGIMES_OOD,
    SEED_P11_DEV,
    SEED_P11_E2E,
    SEED_P11_TRAIN,
    AssimilationResult,
)
from p11.tests.e2e.conftest import (
    E2E_DIR,
    FOLD_TARGETS,
    OFFGRID_CKPT_SEEDS,
    OFFGRID_VARIANTS,
    P11_DIR,
    assert_non_formal_e2e_seed,
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
)


def test_s01_primary_axis_multi_regime_e2e_workflow() -> None:
    wf = importlib.import_module("p11.wavefront_pipeline")
    for snr_db in (PRIMARY_SNR_DB, 12.0):
        for r_idx, regime in enumerate(REGIMES_ID):
            obs, truth = make_e2e_public_obs(400 + int(snr_db) * 10 + r_idx, snr_db, regime)
            res = wf.run_p11_primary(obs)
            assert isinstance(res, AssimilationResult)
            assert PROBE_ROW not in res.assimilated_rows
            assert PROBE_ROW not in res.decoded_info
            assert np.all(res.gate_out.mix_probe_var >= float(obs.n0) - 1e-12)
            assert np.isfinite(res.gate_out.mix_llrs).all()
            assert set(CORE_ROWS).issubset(res.decoded_info.keys())
            for r in CORE_ROWS:
                bits = np.asarray(res.decoded_info[r])
                assert bits.shape == (64,)
                assert bits.dtype == np.uint8
                assert np.isin(bits, (0, 1)).all()
            core_errors = sum(
                int(not np.array_equal(res.decoded_info[r], truth.b_info[idx]))
                for idx, r in enumerate(CORE_ROWS)
            )
            probe_nll = float(np.mean(
                np.abs(truth.y_b_probe_noisy - res.gate_out.mix_probe_pred) ** 2 / res.gate_out.mix_probe_var
                + np.log(math.pi * res.gate_out.mix_probe_var)
            ))
            assert math.isfinite(probe_nll)
            if snr_db == 12.0 and regime == "continuous":
                assert core_errors <= 4


def test_s02_secondary_axis_four_variant_checkpoint_comparison() -> None:
    import torch

    og = importlib.import_module("p11.offgrid_channel")
    mg = importlib.import_module("p11.manifold_gate")
    ckpt_dir = P11_DIR / "ckpts" / "p10b_dev_ckpts"

    regimes = OFFGRID_REGIMES_ID[:3] + OFFGRID_REGIMES_OOD
    for idx, regime in enumerate(regimes):
        seed = assert_non_formal_e2e_seed(SEED_P11_E2E + 450 + idx)
        obs, _ = og.simulate_offgrid_link(seed, regime, snr_db=PRIMARY_SNR_DB)
        bundle = og.build_offgrid_branches(obs)

        variant_llrs: dict[str, np.ndarray] = {}
        for variant in OFFGRID_VARIANTS:
            per_seed_preds = []
            for c_seed in OFFGRID_CKPT_SEEDS:
                ckpt = torch.load(ckpt_dir / f"{variant}_seed{c_seed}.pt", map_location="cpu", weights_only=True)
                assert int(ckpt["real_parameters"]) == 3153
                model = mg.ManifoldTrustGate(variant=variant, **ckpt["arch"])
                model.load_state_dict(ckpt["state_dict"])
                model.eval()
                out = model.gate(obs, bundle)
                assert np.all(out.gate_out.mix_probe_var >= float(obs.n0) - 1e-12)
                per_seed_preds.append(out.gate_out.mix_llrs)
            variant_llrs[variant] = np.mean(per_seed_preds, axis=0)
            assert np.isfinite(variant_llrs[variant]).all()
        assert set(variant_llrs.keys()) == set(OFFGRID_VARIANTS)


def test_s03_prefreeze_dry_run_audit_and_report_pipeline(tmp_path: Path) -> None:
    freeze = importlib.import_module("p11.freeze")
    importlib.import_module("p11.formal_study")
    audit = importlib.import_module("p11.audit")
    importlib.import_module("p11.report")

    manifest = tmp_path / "freeze.json"
    digest = freeze.create_freeze(manifest_path=manifest, allow_dirty_agents=True)
    assert freeze.verify_freeze(manifest_path=manifest, allow_dirty_agents=True) == digest

    dry_dir = tmp_path / "dry_run"
    proc_dry = run_p11_cli(["-m", "p11.formal_study", "--mode", "dry_run", "--out-dir", str(dry_dir)], timeout=600)
    assert proc_dry.returncode == 0, proc_dry.stderr

    selftest_out = tmp_path / "audit_selftest.json"
    proc_aud = run_p11_cli(["-m", "p11.audit", "--selftest", "--output", str(selftest_out)])
    assert proc_aud.returncode == 0, proc_aud.stderr
    assert json.loads(selftest_out.read_text(encoding="utf-8"))["status"] == "PASS"

    ds = make_synthetic_g16_dataset(tmp_path / "g16_s03", SEED_P11_E2E + 900_000)
    raw_recs = [json.loads(ln) for ln in ds["raw"].read_text(encoding="utf-8").splitlines() if ln.strip()]
    rep_recs = [json.loads(ln) for ln in ds["replay"].read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert audit.verify_replay_exact(raw_recs, rep_recs, float_tol=1e-12, strict=True)["passed"] is True

    rep_dir = tmp_path / "report_out"
    proc_rep = run_p11_cli([
        "-m", "p11.report",
        "--results-dir", str(ds["dir"]),
        "--out-report", str(rep_dir / "P11_REPORT.md"),
        "--out-sync", str(rep_dir / "CLI_SYNC_SUMMARY.md"),
        "--manifest", str(manifest),
        "--allow-dirty-agents",
    ])
    assert proc_rep.returncode == 0, proc_rep.stderr
    report_md = (rep_dir / "P11_REPORT.md").read_text(encoding="utf-8")
    sync_md = (rep_dir / "CLI_SYNC_SUMMARY.md").read_text(encoding="utf-8")
    assert digest in report_md
    assert any(v in report_md for v in ("支持", "反证", "证据不足"))
    assert any(v in sync_md for v in ("支持", "反证", "证据不足"))


def test_s04_single_worker_exact_replay_parity_workflow(tmp_path: Path) -> None:
    importlib.import_module("p11.formal_study")
    audit = importlib.import_module("p11.audit")

    out_1 = tmp_path / "replay_1"
    out_2 = tmp_path / "replay_2"
    p1 = run_p11_cli(["-m", "p11.formal_study", "--mode", "dry_run", "--out-dir", str(out_1)], env=clean_subprocess_env(), timeout=600)
    p2 = run_p11_cli(["-m", "p11.formal_study", "--mode", "dry_run", "--out-dir", str(out_2)], env=clean_subprocess_env(), timeout=600)
    assert p1.returncode == 0 and p2.returncode == 0

    jsonl_1 = sorted(out_1.glob("*.jsonl"))
    assert len(jsonl_1) >= 1
    for path_1 in jsonl_1:
        path_2 = out_2 / path_1.name
        assert path_2.is_file()
        r1 = [json.loads(ln) for ln in path_1.read_text(encoding="utf-8").splitlines() if ln.strip()]
        r2 = [json.loads(ln) for ln in path_2.read_text(encoding="utf-8").splitlines() if ln.strip()]
        res = audit.verify_replay_exact(r1, r2, float_tol=0.0, strict=True)
        assert res["passed"] is True


def test_s05_repository_integrity_and_firewall_audit_workflow() -> None:
    indep = importlib.import_module("p11.independent_ref")
    importlib.import_module("p11.audit")

    diff = run_git(["diff", "--name-only", P10_BASELINE_COMMIT, "--", ":!p11"])
    assert diff.returncode == 0 and diff.stdout.strip() == ""

    all_p11_py = sorted(p for p in P11_DIR.rglob("*.py") if "__pycache__" not in p.parts)
    e2e_py = sorted(E2E_DIR.glob("*.py"))

    assert p9_indep.check_no_hardcoded_n0(all_p11_py, strict=True)["clean"] is True
    assert scan_infer_guard_ast(all_p11_py) == []
    assert scan_skip_ast(e2e_py) == []
    assert scan_mock_ast(e2e_py) == []
    assert scan_private_p11_imports_ast(e2e_py) == []
    assert scan_formal_seed_symbols_ast(e2e_py) == []
    assert indep.check_auditor_independence([P11_DIR / "audit.py", P11_DIR / "independent_ref.py"], strict=True)["independent"] is True

    disjoint = indep.check_p11_seed_disjointness(
        p11_train_seeds=[SEED_P11_TRAIN + 5000],
        p11_dev_seeds=[SEED_P11_DEV + i for i in range(4)],
        p11_formal_id_seeds=(),
        p11_formal_ood_seeds=(),
        p11_e2e_seeds=[SEED_P11_E2E + 5000 + i for i in range(4)],
        strict=True,
    )
    assert disjoint["disjoint"] is True


def test_s06_ood_and_calibration_stress_workflow() -> None:
    wf = importlib.import_module("p11.wavefront_pipeline")
    og = importlib.import_module("p11.offgrid_channel")

    for snr_db in (6.0, 9.0, 12.0):
        for r_idx, regime in enumerate(REGIMES_OOD):
            obs, truth = make_e2e_public_obs(550 + int(snr_db) * 10 + r_idx, snr_db, regime)
            res = wf.run_p11_primary(obs)
            pvar = res.gate_out.mix_probe_var
            pred = res.gate_out.mix_probe_pred
            assert np.isfinite(pred).all() and np.isfinite(pvar).all()
            assert np.all(pvar >= float(obs.n0) - 1e-12)
            nll = float(np.mean(np.abs(truth.y_b_probe_noisy - pred) ** 2 / pvar + np.log(math.pi * pvar)))
            assert math.isfinite(nll)

            # Target-row poisoning check under OOD stress
            y_poison = obs.y_b.copy()
            y_poison[list(FOLD_TARGETS[0])] = 2.125 - 0.75j
            _, t_base = wf.run_p11_with_trace(obs)
            _, t_pois = wf.run_p11_with_trace(clone_public_obs(obs, y_b=y_poison))
            assert t_base["folds"][0]["branch"] == t_pois["folds"][0]["branch"]
            assert {rec["row"] for rec in t_base["folds"][0]["ledger"]} == {
                rec["row"] for rec in t_pois["folds"][0]["ledger"]
            }

    seed_og = assert_non_formal_e2e_seed(SEED_P11_E2E + 599)
    obs_og, truth_og = og.simulate_offgrid_link(seed_og, "offgrid_ood_five_ray_fast", snr_db=PRIMARY_SNR_DB)
    assert truth_og.n_rays == 5
    assert obs_og.probe_row == PROBE_ROW
