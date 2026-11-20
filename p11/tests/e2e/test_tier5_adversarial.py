"""Tier 5 White-Box Adversarial Hardening & Codespace Freeze Readiness E2E Suite.

Covers all M4 challenger findings (G1-G7 & F1-F8) using strictly SEED_P11_E2E / SEED_P11_DEV,
zero mocks, zero skips, zero hardcoded N0, and zero torch.inference_mode().
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import warnings

import numpy as np
import pytest

from p11.contract import (
    ANCHOR,
    COMPLEX_DTYPE,
    CORE_ROWS,
    N,
    OFFGRID_ALL_REGIMES,
    OFFGRID_REGIMES_ID,
    OFFGRID_REGIMES_OOD,
    P10_BASELINE_COMMIT,
    PRIMARY_SNR_DB,
    PROBE_ROW,
    REAL_DTYPE,
    REGIMES_ID,
    SEED_P11_DEV,
    SEED_P11_E2E,
    T,
    AssimilationResult,
    GateOutput,
    fractional_delay_freq_response,
)
from p11.tests.e2e.conftest import (
    EXPECTED_BRANCH,
    MATH_VERIFICATION_SHA256,
    OFFGRID_CKPT_SEEDS,
    OFFGRID_VARIANTS,
    P11_DIR,
    REPO_ROOT,
    assert_non_formal_e2e_seed,
    clean_subprocess_env,
    clone_public_obs,
    make_e2e_public_obs,
    make_synthetic_g16_dataset,
    run_git,
    sha256_file,
)


def _copy_p11_clean(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"),
    )


def _init_temp_bundle_repo(tmp_path: Path) -> tuple[Path, Path]:
    """Create a clean git repo on EXPECTED_BRANCH with current p11/ committed."""
    src_repo = tmp_path / "src_repo"
    clone_proc = subprocess.run(
        [
            "git",
            "clone",
            "--branch",
            EXPECTED_BRANCH,
            "--single-branch",
            str(REPO_ROOT),
            str(src_repo),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert clone_proc.returncode == 0, clone_proc.stderr

    for cfg_key, cfg_val in (
        ("user.name", "P11 Test"),
        ("user.email", "p11@example.local"),
    ):
        subprocess.run(["git", "config", cfg_key, cfg_val], cwd=str(src_repo), check=True)

    _copy_p11_clean(P11_DIR, src_repo / "p11")
    # Remove any non-dev artifacts inside temp copy results/ so temp freeze stays clean
    for item in (src_repo / "p11" / "results").iterdir():
        if item.name != "p11_dev.json":
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    subprocess.run(["git", "add", "p11"], cwd=str(src_repo), check=True)
    subprocess.run(["git", "add", "-f", "p11/ckpts"], cwd=str(src_repo), check=True)
    subprocess.run(["git", "commit", "-m", "p11 pre-freeze snapshot"], cwd=str(src_repo), check=True)
    return src_repo, src_repo / "p11"


def test_t5_01_codespace_git_bundle_eol_and_freeze_roundtrip(tmp_path: Path) -> None:
    p11_freeze = importlib.import_module("p11.freeze")

    # 1. p11/.gitattributes present with exact LF content; root untouched; zero CRLF in p11/
    p11_gitattr = P11_DIR / ".gitattributes"
    assert p11_gitattr.is_file()
    assert p11_gitattr.read_bytes() == b"* -text !eol\n"

    diff_non_p11 = run_git(["diff", "--name-only", P10_BASELINE_COMMIT, "--", ":!p11"])
    assert diff_non_p11.returncode == 0 and diff_non_p11.stdout.strip() == ""

    crlf_files: list[str] = []
    for ext in ("*.py", "*.md", "*.json", ".gitattributes"):
        for fpath in P11_DIR.rglob(ext):
            if "__pycache__" in fpath.parts or ".pytest_cache" in fpath.parts:
                continue
            if b"\r\n" in fpath.read_bytes():
                crlf_files.append(fpath.relative_to(REPO_ROOT).as_posix())
    assert crlf_files == [], f"CRLF line endings found in p11/: {crlf_files}"

    # 2. Static check_ast_no_inference_mode clean across p11/*.py
    py_files = [
        p for p in P11_DIR.rglob("*.py")
        if "__pycache__" not in p.parts and ".pytest_cache" not in p.parts
    ]
    assert p11_freeze.check_ast_no_inference_mode(py_files) == []

    # 3. Real git bundle + Linux EOL single-branch clone roundtrip (no refs/heads/main!)
    src_repo, _ = _init_temp_bundle_repo(tmp_path)
    env = clean_subprocess_env({"PYTHONPATH": str(src_repo)})
    w_proc = subprocess.run(
        [sys.executable, "-B", "-m", "p11.freeze", "--write"],
        cwd=str(src_repo),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert w_proc.returncode == 0, w_proc.stderr
    subprocess.run(["git", "add", "p11/results/freeze_manifest.json"], cwd=str(src_repo), check=True)
    subprocess.run(["git", "commit", "-m", "freeze manifest"], cwd=str(src_repo), check=True)

    bundle_path = tmp_path / "p11.bundle"
    subprocess.run(
        ["git", "bundle", "create", str(bundle_path), "HEAD", EXPECTED_BRANCH],
        cwd=str(src_repo),
        check=True,
    )

    linux_clone = tmp_path / "linux_clone"
    subprocess.run(
        [
            "git",
            "clone",
            "-b",
            EXPECTED_BRANCH,
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.eol=lf",
            str(bundle_path),
            str(linux_clone),
        ],
        check=True,
    )

    # Confirm single-branch bundle clone has neither refs/heads/main nor refs/remotes/origin/main
    for ref in ("refs/heads/main", "refs/remotes/origin/main"):
        r_chk = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", ref],
            cwd=str(linux_clone),
            capture_output=True,
            text=True,
            check=False,
        )
        assert r_chk.returncode != 0

    env_linux = clean_subprocess_env({"PYTHONPATH": str(linux_clone)})
    v_proc = subprocess.run(
        [sys.executable, "-B", "-m", "p11.freeze", "--verify"],
        cwd=str(linux_clone),
        env=env_linux,
        capture_output=True,
        text=True,
        check=False,
    )
    assert v_proc.returncode == 0, v_proc.stderr

    manifest = json.loads((linux_clone / "p11" / "results" / "freeze_manifest.json").read_text(encoding="utf-8"))
    assert "p11/.gitattributes" in manifest["sources"]
    assert "p11/results/p11_audit_selftest.json" not in manifest["sources"]
    assert sha256_file(linux_clone / "p11" / "math_verification.md") == MATH_VERIFICATION_SHA256
    for rel_map in (manifest["sources"], manifest["checkpoints"]):
        for rel_p, expected_sha in rel_map.items():
            assert sha256_file(linux_clone / rel_p) == expected_sha


def test_t5_02_freeze_rejects_stray_files_and_allows_post_freeze_selftest(tmp_path: Path) -> None:
    src_repo, src_p11 = _init_temp_bundle_repo(tmp_path)
    env = clean_subprocess_env({"PYTHONPATH": str(src_repo)})

    w_proc = subprocess.run(
        [sys.executable, "-B", "-m", "p11.freeze", "--write"],
        cwd=str(src_repo),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert w_proc.returncode == 0, w_proc.stderr
    subprocess.run(["git", "add", "p11/results/freeze_manifest.json"], cwd=str(src_repo), check=True)
    subprocess.run(["git", "commit", "-m", "freeze manifest"], cwd=str(src_repo), check=True)

    # Plant rogue .py inside p11/results/ -> verify_freeze must fail
    rogue_py = src_p11 / "results" / "rogue.py"
    rogue_py.write_text("x = 1\n", encoding="utf-8", newline="\n")
    bad_rogue = subprocess.run(
        [sys.executable, "-B", "-m", "p11.freeze", "--verify"],
        cwd=str(src_repo),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_rogue.returncode != 0
    rogue_py.unlink()

    # Tamper p11/.gitattributes -> verify_freeze must fail
    ga_path = src_p11 / ".gitattributes"
    orig_ga = ga_path.read_bytes()
    ga_path.write_bytes(b"* text=auto eol=crlf\n")
    bad_ga = subprocess.run(
        [sys.executable, "-B", "-m", "p11.freeze", "--verify"],
        cwd=str(src_repo),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad_ga.returncode != 0
    ga_path.write_bytes(orig_ga)

    # Run default p11.audit --selftest post-freeze -> verify_freeze must still PASS
    st_proc = subprocess.run(
        [sys.executable, "-B", "-m", "p11.audit", "--selftest"],
        cwd=str(src_repo),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert st_proc.returncode == 0, st_proc.stderr
    assert (src_p11 / "results" / "p11_audit_selftest.json").is_file()

    ok_after_st = subprocess.run(
        [sys.executable, "-B", "-m", "p11.freeze", "--verify"],
        cwd=str(src_repo),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ok_after_st.returncode == 0, ok_after_st.stderr


def test_t5_03_report_edge_schemas_and_nan_rejection(tmp_path: Path) -> None:
    p11_report = importlib.import_module("p11.report")
    ds = make_synthetic_g16_dataset(tmp_path / "g16_rep", SEED_P11_E2E + 10, links_per_cluster=1)
    pri_sum = json.loads(ds["summary"].read_text(encoding="utf-8"))
    aud_obj = json.loads(ds["audit"].read_text(encoding="utf-8"))

    # Case 1: seed_disjointness is None
    rep1, _ = p11_report.render_reports(
        primary_summary=pri_sum,
        audit_payload={**aud_obj, "seed_disjointness": None},
        allow_partial=True,
    )
    assert "支持" in rep1 and "disjoint = None" in rep1

    # Case 2: noninferiority_full omits verdict_zh
    pri_no_vzh = json.loads(json.dumps(pri_sum))
    pri_no_vzh["noninferiority_full"] = {
        "verdict": "supported",
        "bler_noninferiority": {"mean_diff": 0.001, "upper_95": 0.004, "lower_95": -0.002},
    }
    rep2, _ = p11_report.render_reports(
        primary_summary=pri_no_vzh,
        audit_payload=aud_obj,
        allow_partial=True,
    )
    assert "支持" in rep2

    # Case 3: totals={"sparse": None} renders cleanly without TypeError
    pri_tot_none = json.loads(json.dumps(pri_sum))
    pri_tot_none["totals"] = {"sparse": None}
    rep3, sync3 = p11_report.render_reports(
        primary_summary=pri_tot_none,
        audit_payload=aud_obj,
        allow_partial=True,
    )
    assert "支持" in rep3 and "支持" in sync3

    # Case 4: NaN / Inf in clean_nmse or calibration must raise ValueError
    for bad_key, bad_val in (("clean_nmse", float("nan")), ("calibration", float("inf"))):
        pri_bad = json.loads(json.dumps(pri_sum))
        pri_bad["totals"]["sparse"]["p11_chained_wavefront"][bad_key] = bad_val
        with pytest.raises(ValueError):
            p11_report.render_reports(primary_summary=pri_bad, audit_payload=aud_obj, allow_partial=True)

    # Case 5 (R2): common_phase_blindspot None sub-mappings/fields render cleanly without AttributeError/TypeError
    for cpb_edge in (
        {"pooled_3id": None, "by_regime": None},
        {"pooled_3id": {}, "by_regime": {"continuous": None}},
        {"pooled_3id": {"delta_vs_p10_9e9d93b_pp": None}},
    ):
        pri_cpb = json.loads(json.dumps(pri_sum))
        pri_cpb["common_phase_blindspot"] = cpb_edge
        rep_cpb, _ = p11_report.render_reports(primary_summary=pri_cpb, audit_payload=aud_obj, allow_partial=True)
        assert "支持" in rep_cpb

    # Case 6 (R2): NaN / Inf / bool in common_phase_blindspot, audit_payload max_*_diff, and literal NaN on disk
    for mutator in (
        lambda s: s["common_phase_blindspot"]["pooled_3id"].update({"delta_vs_p10_9e9d93b_pp": float("nan")}),
        lambda s: s["common_phase_blindspot"]["by_regime"]["continuous"].update({"delta_vs_p10_9e9d93b_pp": float("inf")}),
        lambda s: s["common_phase_blindspot"]["pooled_3id"].update({"false_assimilations": True}),
    ):
        pri_bad_cpb = json.loads(json.dumps(pri_sum))
        mutator(pri_bad_cpb)
        with pytest.raises(ValueError):
            p11_report.render_reports(primary_summary=pri_bad_cpb, audit_payload=aud_obj, allow_partial=True)

    aud_nan = json.loads(json.dumps(aud_obj))
    aud_nan["primary"]["audit"]["max_aggregate_diff"] = float("nan")
    with pytest.raises(ValueError):
        p11_report.render_reports(primary_summary=pri_sum, audit_payload=aud_nan, allow_partial=True)

    disk_nan_dir = tmp_path / "disk_nan_report"
    disk_nan_dir.mkdir()
    bad_disk_sum = json.loads(json.dumps(pri_sum))
    bad_disk_sum["common_phase_blindspot"]["pooled_3id"]["delta_vs_p10_9e9d93b_pp"] = 123456.789
    raw_disk_text = json.dumps(bad_disk_sum).replace("123456.789", "NaN")
    (disk_nan_dir / "p11_primary_summary.json").write_text(raw_disk_text, encoding="utf-8", newline="\n")
    (disk_nan_dir / "p11_audit.json").write_text(json.dumps(aud_obj), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError):
        p11_report.main([
            "--results-dir", str(disk_nan_dir),
            "--out-report", str(disk_nan_dir / "R.md"),
            "--out-sync", str(disk_nan_dir / "S.md"),
            "--no-verify-freeze",
            "--allow-partial",
        ])


def test_t5_04_anti_leakage_all_11_fields_bool_keys_and_shape_guards() -> None:
    p11_indep = importlib.import_module("p11.independent_ref")
    p11_audit = importlib.import_module("p11.audit")

    obs, _ = make_e2e_public_obs(0, snr_db=2.0, regime="continuous", budget="sparse")
    clean_fn = p11_audit._make_clean_cstar_assimilator()

    def _is_poisoned(o) -> bool:
        return not np.allclose(o.y_b[PROBE_ROW], obs.y_b[PROBE_ROW], equal_nan=True)

    # Leak probe into each of the 6 newly guarded fields individually
    for field_name in (
        "history_weight",
        "probe_weight",
        "mix_mean_ray",
        "mix_var_ray",
        "row_margins",
        "n_rounds",
    ):
        def _make_leaky(fname: str):
            def _fn(o):
                res = clean_fn(o)
                if not _is_poisoned(o):
                    return res
                g = res.gate_out
                if fname == "history_weight":
                    hw = g.history_weight.copy()
                    hw[0] = float(np.clip(abs(o.y_b[PROBE_ROW, 0]) * 1e-6, 0.01, 0.99))
                    g = dataclasses.replace(g, history_weight=hw)
                elif fname == "probe_weight":
                    g = dataclasses.replace(g, probe_weight=float(g.probe_weight) + 0.05)
                elif fname == "mix_mean_ray":
                    mr = g.mix_mean_ray.copy()
                    mr[0, 0, 0] += 1e-4 * complex(o.y_b[PROBE_ROW, 0])
                    g = dataclasses.replace(g, mix_mean_ray=mr)
                elif fname == "mix_var_ray":
                    vr = g.mix_var_ray.copy()
                    vr[0, 0, 0] += 1e-4
                    g = dataclasses.replace(g, mix_var_ray=vr)
                elif fname == "row_margins":
                    rm = dict(res.row_margins)
                    k0 = next(iter(rm))
                    rm[k0] = float(rm[k0]) + 0.1
                    return dataclasses.replace(res, row_margins=rm)
                elif fname == "n_rounds":
                    return dataclasses.replace(res, n_rounds=int(res.n_rounds) + 1)
                return dataclasses.replace(res, gate_out=g)
            return _fn

        rep = p11_indep.audit_assimilation_anti_leakage(_make_leaky(field_name), obs, strict=False)
        assert rep["passed"] is False, f"Missed probe leak on {field_name}"

    # Bool key aliasing (True replacing int 1 on poison call)
    def _bool_key_fn(o):
        res = clean_fn(o)
        if not _is_poisoned(o):
            return res
        rw = { (True if k == 1 else k): v for k, v in res.row_weights.items() }
        return dataclasses.replace(res, row_weights=rw)

    rep_bool = p11_indep.audit_assimilation_anti_leakage(_bool_key_fn, obs, strict=False)
    assert rep_bool["passed"] is False

    # Adapter shape validation rejects malformed GateOutput shapes
    base_res = clean_fn(obs)
    bad_mray = GateOutput(
        history_weight=base_res.gate_out.history_weight,
        probe_weight=base_res.gate_out.probe_weight,
        mix_llrs=base_res.gate_out.mix_llrs,
        mix_probe_pred=base_res.gate_out.mix_probe_pred,
        mix_probe_var=base_res.gate_out.mix_probe_var,
        mix_mean_ray=np.zeros((2, 16), dtype=COMPLEX_DTYPE),
        mix_var_ray=np.ones((2, 16), dtype=REAL_DTYPE),
    )
    ad_rep = p11_indep.check_assimilation_result_adapter(
        dataclasses.replace(base_res, gate_out=bad_mray),
        strict=False,
    )
    assert ad_rep["valid"] is False


def test_t5_05_audit_table_empty_bool_ni_and_statistical_boundaries(tmp_path: Path) -> None:
    p11_audit = importlib.import_module("p11.audit")
    ds = make_synthetic_g16_dataset(tmp_path / "g16_aud", SEED_P11_E2E + 20, links_per_cluster=1)
    pri_sum = json.loads(ds["summary"].read_text(encoding="utf-8"))

    # 1. _verify_table_matches_recomputed rejects empty dict & bool values
    rec_tbl = {"count": 1, "errors": 0, "bler": 0.0}
    with pytest.raises(ValueError):
        p11_audit._verify_table_matches_recomputed(rec_tbl, {}, "empty")
    with pytest.raises((TypeError, ValueError)):
        p11_audit._verify_table_matches_recomputed(rec_tbl, {"count": True, "errors": False, "bler": False}, "bool")

    # 2. Canonical primary audit rejects missing NI, swapped NI, int passes=1, 2D differences, totals=None
    cases: list[tuple[str, dict]] = []

    no_ni = json.loads(json.dumps(pri_sum))
    no_ni.pop("noninferiority", None)
    cases.append(("no_ni", no_ni))

    swapped_ni = json.loads(json.dumps(pri_sum))
    swapped_ni["noninferiority"]["candidate"] = ["dense", "always_clear"]
    swapped_ni["noninferiority"]["reference"] = ["sparse", "p11_chained_wavefront"]
    cases.append(("swapped_ni", swapped_ni))

    int_passes = json.loads(json.dumps(pri_sum))
    int_passes["noninferiority"]["passes"] = 1
    cases.append(("int_passes", int_passes))

    diffs_2d = json.loads(json.dumps(pri_sum))
    diffs_2d["tests"][0]["differences"] = [diffs_2d["tests"][0]["differences"]]
    cases.append(("diffs_2d", diffs_2d))

    totals_none = json.loads(json.dumps(pri_sum))
    totals_none["totals"] = None
    cases.append(("totals_none", totals_none))

    for label, bad_obj in cases:
        bad_p = tmp_path / f"bad_{label}.json"
        bad_p.write_text(json.dumps(bad_obj), encoding="utf-8", newline="\n")
        with pytest.raises(ValueError):
            p11_audit.audit_p11_primary(ds["raw"], bad_p, ds["replay"], require_canonical_family=True)

    # 3. Statistical bool & alpha boundaries
    zeros_bool = np.zeros(16, dtype=bool)
    zeros_f = np.zeros(16, dtype=np.float64)
    plus_1pp = np.full(16, 0.01, dtype=np.float64)
    bits = np.full(16, 100.0, dtype=np.float64)

    with pytest.raises((TypeError, ValueError)):
        p11_audit.audit_engineering_noninferiority(zeros_bool, zeros_bool, bits, bits)
    with pytest.raises((TypeError, ValueError)):
        p11_audit.audit_engineering_noninferiority(zeros_f, zeros_f, bits, bits, alpha=2.5)
    with pytest.raises((TypeError, ValueError)):
        p11_audit.audit_engineering_noninferiority(zeros_f, zeros_f, bits, bits, alpha=True)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        p11_audit.audit_p11_holm_families({"c1": zeros_bool}, {"c2": zeros_bool}, expected_g=16)

    ni_zero = p11_audit.audit_engineering_noninferiority(zeros_f, zeros_f, bits, bits)
    assert ni_zero["passes"] is True and ni_zero["bler_noninferiority"]["upper_95"] == 0.0
    ni_1pp = p11_audit.audit_engineering_noninferiority(plus_1pp, zeros_f, bits, bits)
    assert ni_1pp["passes"] is False and ni_1pp["verdict"] == "refuted"

    # 4. check_p11_offgrid_cell valid & malformed
    good_cell = {
        "blocks": 18, "block_errors": 1, "coded_bits": 2304, "coded_errors": 4,
        "info_bits": 1152, "info_errors": 2, "extra_blocks": 0, "extra_errors": 0,
        "probes": 32, "probe_sse": 1.0, "clean_sse": 0.5, "probe_energy": 10.0,
        "clean_energy": 9.0, "variance_sum": 2.0, "nll_sum": -5.0,
        "native_block_errors": 2, "native_info_errors": 4,
    }
    p11_audit.check_p11_offgrid_cell(good_cell)
    with pytest.raises(ValueError):
        p11_audit.check_p11_offgrid_cell({**good_cell, "blocks": True})


def test_t5_06_payload_and_wavefront_row_and_midpoint_boundary_guards() -> None:
    pr = importlib.import_module("p11.payload_rebuild")
    wf = importlib.import_module("p11.wavefront_pipeline")

    obs, _ = make_e2e_public_obs(50, snr_db=9.0, regime="continuous", budget="sparse")

    # 1. stratified_mode_midpoints budget & -inf guards
    strata4 = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], dtype=np.int64)
    idx_cap, mass_cap, _ = pr.stratified_mode_midpoints(np.zeros(16, dtype=REAL_DTYPE), strata4, budget=2)
    assert len(idx_cap) <= 2 and np.all(mass_cap > 0.0) and float(np.sum(mass_cap)) <= 1.0 + 1e-12

    for bad_budget in (0, -1, True, 2.5):
        with pytest.raises((TypeError, ValueError)):
            pr.stratified_mode_midpoints(np.zeros(16, dtype=REAL_DTYPE), strata4, budget=bad_budget)  # type: ignore[arg-type]

    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        idx_inf, mass_inf, lnorm = pr.stratified_mode_midpoints(np.full(16, -np.inf, dtype=REAL_DTYPE), strata4, budget=4)
    assert not any("invalid value encountered in subtract" in str(w.message) for w in wrec)
    assert len(idx_inf) <= 4 and np.isfinite(mass_inf).all() and lnorm == -math.inf

    # 2. Strict row-index integer guards (reject bool, np.bool_, float, timedelta64)
    bad_aux_tuples = (
        (True, 5, 8, 13, 16),
        (np.bool_(True), 5, 8, 13, 16),
        (2.0, 5, 8, 13, 16),
        (2.9, 5, 8, 13, 16),
        (np.timedelta64(2, "ns"), 5, 8, 13, 16),
    )
    for bad_aux in bad_aux_tuples:
        with pytest.raises((TypeError, ValueError)):
            pr.estimate_payload_posterior(obs, bad_aux)  # type: ignore[arg-type]
        with pytest.raises((TypeError, ValueError)):
            pr.estimate_payload_posterior_raw(obs.y_b[ANCHOR], bad_aux, None, obs.n0, use_payload=False)  # type: ignore[arg-type]

    for bad_probe in (19.9, 19.0, True):
        with pytest.raises((TypeError, ValueError)):
            wf.run_p11_primary(dataclasses.replace(obs, probe_row=bad_probe))  # type: ignore[arg-type]
    for bad_tr in ((10.9,), (10.0,), (True,)):
        with pytest.raises((TypeError, ValueError)):
            wf.run_p11_primary(dataclasses.replace(obs, training_rows=bad_tr))  # type: ignore[arg-type]

    # 3. decode_with_blind_margin on degenerate zero input
    dec0 = wf.decode_with_blind_margin(
        np.zeros(N, dtype=COMPLEX_DTYPE),
        np.zeros((N, N), dtype=COMPLEX_DTYPE),
        np.full(N, 0.1, dtype=REAL_DTYPE),
    )
    assert dec0.margin == 0.0 and dec0.fit_ratio == 0.0
    assert math.isfinite(dec0.d_best) and math.isfinite(dec0.d_second)


def test_t5_07_contract_empty_sequences_and_parabolic_refinement_parity() -> None:
    pr = importlib.import_module("p11.payload_rebuild")

    for empty_seq in ([], (), ((),)):
        with pytest.raises(ValueError):
            fractional_delay_freq_response(empty_seq)  # type: ignore[arg-type]

    obs, _ = make_e2e_public_obs(61, snr_db=9.0, regime="partial_shift", budget="sparse")
    aux = (2, 5, 8, 13, 16)
    n0 = float(obs.n0)
    y10 = obs.y_b[ANCHOR]
    y_aux = obs.y_b[list(aux)]
    a_prior = pr.build_frame_a_prior_clean(obs.y_a, n0)
    reset_base = pr.estimate_payload_posterior(obs, aux, use_payload=True)

    ev_no_para = pr.score_branches_common_units(
        a_prior, reset_base, y10, aux, y_aux, n0, use_parabolic=False
    )
    c_b = complex(reset_base.gauge)
    a_prior_c = pr.scale_tracker_state(a_prior, 1.0 / c_b)
    s_single = pr._score_one_inherited_hypothesis(
        a_prior_c,
        y10 / c_b,
        aux,
        y_aux / c_b,
        changed_ray=ev_no_para.best_shift.ray,
        delta_freq=ev_no_para.best_shift.delta,
        onset_row=ev_no_para.best_shift.onset_row,
        aux_temp=1.0,
    )
    assert abs(ev_no_para.s_s_best - s_single) < 1e-12

    ev_para = pr.score_branches_common_units(
        a_prior, reset_base, y10, aux, y_aux, n0, use_parabolic=True
    )
    assert ev_para.s_s_best >= ev_no_para.s_s_best - 1e-15
    assert math.isfinite(ev_para.b_s)


def test_t5_08_offgrid_degenerate_nan_guards_and_wavefront_zero_anchor() -> None:
    import torch

    og = importlib.import_module("p11.offgrid_channel")
    mg = importlib.import_module("p11.manifold_gate")
    wf = importlib.import_module("p11.wavefront_pipeline")
    p11_indep = importlib.import_module("p11.independent_ref")

    # 1. Off-grid input guards (k_hat, zero pilot_tx_a, NaN y_a/y_b/w_ray/noise_mult)
    obs_og, _ = og.simulate_offgrid_link(int(SEED_P11_E2E) + 70, "offgrid_continuous", snr_db=2.0)
    bundle_og = og.build_offgrid_branches(obs_og)

    for bad_k in (0, -1, True, 5.0):
        with pytest.raises((TypeError, ValueError)):
            og.discover_frame_a_atoms(obs_og, k_hat=bad_k)  # type: ignore[arg-type]

    pt_zero = obs_og.pilot_tx_a.copy()
    pt_zero[0, 0] = 0.0 + 0.0j
    with warnings.catch_warnings(record=True) as wrec:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            og.discover_frame_a_atoms(dataclasses.replace(obs_og, pilot_tx_a=pt_zero))
    assert not any("divide by zero" in str(w.message) for w in wrec)

    ya_nan = obs_og.y_a.copy()
    ya_nan[0, 0] = complex(math.nan, 0.0)
    with pytest.raises(ValueError):
        og.discover_frame_a_atoms(dataclasses.replace(obs_og, y_a=ya_nan))

    yb_nan = obs_og.y_b.copy()
    yb_nan[0, 0] = complex(math.nan, 0.0)
    obs_yb_nan = dataclasses.replace(obs_og, y_b=yb_nan)
    with pytest.raises(ValueError):
        og.build_offgrid_branches(obs_yb_nan)
    with pytest.raises(ValueError):
        mg.extract_manifold_features(obs_yb_nan, bundle_og)

    w_nan = np.ones((bundle_og.k_hat, T), dtype=REAL_DTYPE)
    w_nan[0, 0] = math.nan
    with pytest.raises(ValueError):
        og.compose_offgrid_gate_output(w_nan, np.ones(T, dtype=REAL_DTYPE), obs_og, bundle_og)

    nm_nan = np.ones(T, dtype=REAL_DTYPE)
    nm_nan[0] = math.nan
    with pytest.raises(ValueError):
        og.compose_offgrid_gate_output(np.ones((bundle_og.k_hat, T), dtype=REAL_DTYPE), nm_nan, obs_og, bundle_og)

    # 2. All 4 *seed1001.pt checkpoints preserve C* equivariance (<= 1e-12) across 1e-12..1e8
    ckpt_dir = P11_DIR / "ckpts" / "p10b_dev_ckpts"
    for var in OFFGRID_VARIANTS:
        ckpt = torch.load(ckpt_dir / f"{var}_seed{OFFGRID_CKPT_SEEDS[0]}.pt", map_location="cpu", weights_only=True)
        model = mg.ManifoldTrustGate(variant=var, **ckpt["arch"])
        model.load_state_dict(ckpt["state_dict"], strict=True)
        model.eval()
        base_out = model.gate(obs_og, bundle_og)
        for alpha in (1e-12, 1e-8, 1e8, 1e8 * complex(math.cos(0.77), math.sin(0.77))):
            a_c = complex(alpha)
            a2 = float(abs(a_c) ** 2)
            yb_s = (a_c * obs_og.y_b).astype(COMPLEX_DTYPE)
            yb_s[obs_og.probe_row] = complex(math.nan, math.nan)
            obs_s = dataclasses.replace(obs_og, y_a=(a_c * obs_og.y_a).astype(COMPLEX_DTYPE), y_b=yb_s, n0=a2 * float(obs_og.n0))
            b_s = og.build_offgrid_branches(obs_s)
            s_out = model.gate(obs_s, b_s)
            assert float(np.max(np.abs(s_out.w_ray - base_out.w_ray))) <= 1e-12
            assert float(np.max(np.abs(s_out.noise_mult - base_out.noise_mult))) <= 1e-12
            assert float(np.max(np.abs(s_out.gate_out.mix_llrs - base_out.gate_out.mix_llrs))) <= 1e-12

    # 3. Wavefront zero-anchor + negative SNR (-3 dB) stability & C* anti-leakage
    obs_rst, _ = make_e2e_public_obs(77, snr_db=-3.0, regime="full_reset", budget="sparse")
    yb_zero_anc = obs_rst.y_b.copy()
    yb_zero_anc[ANCHOR] = 0.0 + 0.0j
    obs_zero = clone_public_obs(obs_rst, y_b=yb_zero_anc)
    res_zero = wf.run_p11_primary(obs_zero)
    assert np.isfinite(res_zero.gate_out.mix_probe_pred).all()
    assert np.isfinite(res_zero.gate_out.mix_probe_var).all()
    assert np.all(res_zero.gate_out.mix_probe_var >= float(obs_zero.n0) - 1e-12)
    assert PROBE_ROW not in res_zero.assimilated_rows
    assert PROBE_ROW not in res_zero.decoded_info
    assert PROBE_ROW not in res_zero.row_weights
    assert PROBE_ROW not in res_zero.row_margins
    anti_rep = p11_indep.audit_assimilation_anti_leakage(wf.run_p11_primary, obs_zero, strict=True)
    assert anti_rep["passed"] is True and anti_rep["max_cstar_err"] < 1e-12


def test_t5_09_formal_study_resume_audit_merge_and_worker_parity(tmp_path: Path) -> None:
    fs = importlib.import_module("p11.formal_study")
    p11_audit = importlib.import_module("p11.audit")

    # 1. Primary incremental JSONL + truncated-line recovery + --resume parity (2 clusters, ID only for speed)
    dir_full = tmp_path / "pri_full"
    dir_res = tmp_path / "pri_resume"
    fs._run_primary_cohort(
        clusters=(1, 2),
        n_id=1,
        n_ood=0,
        id_base=SEED_P11_DEV,
        ood_base=SEED_P11_DEV,
        id_offset=200_000,
        ood_offset=400_000,
        out_dir=dir_full,
        run_full=True,
        run_replay=False,
        resume=False,
    )
    shutil.copytree(dir_full, dir_res)
    id_jsonl = dir_res / "p11_primary_id_links.jsonl"
    lines = id_jsonl.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    # Drop last record and append a truncated line to simulate Codespace disconnect
    id_jsonl.write_text("\n".join(lines[:-1] + ['{"split": "id", "truncated']) + "\n", encoding="utf-8", newline="\n")

    fs._run_primary_cohort(
        clusters=(1, 2),
        n_id=1,
        n_ood=0,
        id_base=SEED_P11_DEV,
        ood_base=SEED_P11_DEV,
        id_offset=200_000,
        ood_offset=400_000,
        out_dir=dir_res,
        run_full=True,
        run_replay=False,
        resume=True,
    )
    recs_full = [json.loads(ln) for ln in (dir_full / "p11_primary_id_links.jsonl").read_text(encoding="utf-8").splitlines() if ln.strip()]
    recs_res = [json.loads(ln) for ln in id_jsonl.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert p11_audit.verify_replay_exact(recs_full, recs_res, float_tol=0.0, strict=True)["passed"] is True

    # Strict allow_nan=False check on summary JSON
    sum_text = (dir_res / "p11_primary_summary.json").read_text(encoding="utf-8")
    json.loads(sum_text, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Non-standard JSON constant {x}")))
    assert "\r\n" not in sum_text

    # 2. Secondary workers=2 vs single-worker replay exact parity (with require_env=True)
    dir_w2 = tmp_path / "sec_w2"
    sec_out = fs._run_secondary_cohort(
        clusters=(1, 2),
        n_id=1,
        n_ood=1,
        id_base=SEED_P11_DEV,
        ood_base=SEED_P11_DEV,
        id_offset=600_000,
        ood_offset=800_000,
        out_dir=dir_w2,
        eval_workers=2,
        require_env=True,
        run_full=True,
        run_replay=True,
        resume=False,
    )
    assert sec_out["replay_verify"]["passed"] is True
    assert sec_out["replay_verify"]["max_float_diff"] == 0.0

    # 3. Audit JSON merge across sequential --mode primary -> secondary -> replay
    aud_path = tmp_path / "p11_audit.json"
    fs._write_json(aud_path, {"mode": "primary", "axis": "primary", "primary": {"id_links": 1536, "ood_links": 768}})
    merged_sec = fs._merge_audit_bundle(
        aud_path,
        {"mode": "secondary", "axis": "secondary", "secondary": {"id_links": 1024, "ood_links": 512}},
        mode="secondary",
        is_dry=False,
    )
    assert merged_sec["axis"] == "both"
    assert merged_sec["primary"]["id_links"] == 1536
    assert merged_sec["secondary"]["id_links"] == 1024

    fs._write_json(aud_path, merged_sec)
    merged_rep = fs._merge_audit_bundle(
        aud_path,
        {"mode": "replay", "axis": "primary", "primary": {"replay_links": 192, "replay_verify": {"passed": True}}},
        mode="replay",
        is_dry=False,
    )
    assert merged_rep["primary"]["id_links"] == 1536
    assert merged_rep["primary"]["replay_links"] == 192
    assert merged_rep["secondary"]["id_links"] == 1024

    # 4 (R1). _load_resumed_jsonl rejects NaN/Infinity & empty budgets; _write_jsonl atomic replace preserves prior file on error
    valid_rec = recs_full[0]
    valid_key = (
        valid_rec["split"],
        int(valid_rec["cluster"]),
        valid_rec["regime"],
        int(valid_rec["link"]),
        int(valid_rec["seed"]),
    )
    bad_nan_rec = json.loads(json.dumps(recs_full[1]))
    bad_nan_rec["budgets"]["sparse"]["p11_chained_wavefront"]["probe_sse"] = 98765.4321
    bad_nan_line = json.dumps(bad_nan_rec).replace("98765.4321", "NaN")
    bad_empty_rec = json.loads(json.dumps(recs_full[2]))
    bad_empty_rec["budgets"] = {}
    probe_jsonl = tmp_path / "r1_resume_probe.jsonl"
    probe_jsonl.write_text(
        json.dumps(valid_rec) + "\n" + bad_nan_line + "\n" + json.dumps(bad_empty_rec) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    all_keys = {
        (r["split"], int(r["cluster"]), r["regime"], int(r["link"]), int(r["seed"]))
        for r in recs_full
    }
    resumed_map = fs._load_resumed_jsonl(probe_jsonl, all_keys)
    assert set(resumed_map.keys()) == {valid_key}

    # Simulate serialization failure in _write_jsonl (NaN record) -> existing JSONL must remain intact (not 0 bytes!)
    pre_text = json.dumps(valid_rec, allow_nan=False) + "\n"
    probe_jsonl.write_text(pre_text, encoding="utf-8", newline="\n")
    with pytest.raises(ValueError):
        fs._write_jsonl(probe_jsonl, [valid_rec, {"bad": float("nan")}])
    assert probe_jsonl.read_text(encoding="utf-8") == pre_text
    assert list(tmp_path.glob(".r1_resume_probe.jsonl.*.tmp")) == []


def test_t5_10_audit_canonical_family_and_unaudited_summary_blocks(tmp_path: Path) -> None:
    p11_audit = importlib.import_module("p11.audit")

    # 1. Primary canonical audit checks (C2-1)
    ds = make_synthetic_g16_dataset(tmp_path / "g16_c21", SEED_P11_E2E + 100, links_per_cluster=1)
    pri_sum = json.loads(ds["summary"].read_text(encoding="utf-8"))
    raw_recs = [json.loads(ln) for ln in ds["raw"].read_text(encoding="utf-8").splitlines() if ln.strip()]

    def _c_vec_pri(bud: str, arm: str, met: str, reg_f=None) -> np.ndarray:
        return np.asarray([
            p11_audit.aggregate_p11_cells([
                r["budgets"][bud][arm]
                for r in raw_recs
                if r["cluster"] == c and (reg_f is None or r["regime"] in reg_f)
            ])[met]
            for c in range(1, 17)
        ], dtype=np.float64)

    # (a) Tampered noninferiority_full
    bad_ni_full = json.loads(json.dumps(pri_sum))
    bad_ni_full["noninferiority_full"] = {
        "verdict": "refuted",
        "verdict_zh": "反证",
        "bler_noninferiority": {"mean_diff": 0.99, "upper_95": 0.99, "lower_95": 0.99},
    }

    # (b) Tampered common_phase_blindspot
    bad_cpb = json.loads(json.dumps(pri_sum))
    bad_cpb["common_phase_blindspot"]["pooled_3id"]["target_bler"] = 0.0001
    bad_cpb["common_phase_blindspot"]["pooled_3id"]["delta_vs_p10_9e9d93b_pp"] = 20.72

    # (c) Wrong noninferiority metric ("info_ber" with matching recomputed numbers)
    bad_ni_metric = json.loads(json.dumps(pri_sum))
    ni_info = p11_audit.audit_engineering_noninferiority(
        _c_vec_pri("sparse", "p11_chained_wavefront", "info_ber", ["continuous"]),
        _c_vec_pri("dense", "always_clear", "info_ber", ["continuous"]),
        _c_vec_pri("sparse", "p11_chained_wavefront", "net_delivered_info_bits_per_frame", ["continuous"]),
        _c_vec_pri("dense", "always_clear", "net_delivered_info_bits_per_frame", ["continuous"]),
    )
    bad_ni_metric["noninferiority"] = {
        **ni_info["bler_noninferiority"],
        "candidate": ["sparse", "p11_chained_wavefront"],
        "reference": ["dense", "always_clear"],
        "regimes": ["continuous"],
        "metric": "info_ber",
    }

    # (d) Swapped baseline arm on chained_vs_p9_jt (with matching recomputed numbers)
    bad_holm_arm = json.loads(json.dumps(pri_sum))
    swapped_holm = p11_audit._audit_single_holm_family(
        {
            "chained_vs_p9_jt": {"baseline": _c_vec_pri("sparse", "p11_payload_only", "bler"), "candidate": _c_vec_pri("sparse", "p11_chained_wavefront", "bler")},
            "chained_vs_payload_only": {"baseline": _c_vec_pri("sparse", "p11_payload_only", "bler"), "candidate": _c_vec_pri("sparse", "p11_chained_wavefront", "bler")},
        },
        "primary",
    )["contrasts"]
    swapped_holm[0].update({"baseline": ["sparse", "p11_payload_only"], "candidate": ["sparse", "p11_chained_wavefront"]})
    swapped_holm[1].update({"baseline": ["sparse", "p11_payload_only"], "candidate": ["sparse", "p11_chained_wavefront"]})
    bad_holm_arm["tests"] = swapped_holm

    # (e) Cherry-picked regime / wrong metric / flipped sign on chained_vs_p9_jt
    bad_holm_cherry = json.loads(json.dumps(pri_sum))
    cherry_holm = p11_audit._audit_single_holm_family(
        {
            "chained_vs_p9_jt": {
                "baseline": _c_vec_pri("sparse", "p9_jeffreys_student_t", "info_ber", ["full_reset"]),
                "candidate": _c_vec_pri("sparse", "p11_chained_wavefront", "info_ber", ["full_reset"]),
                "expected_sign": -1,
            },
            "chained_vs_payload_only": {"baseline": _c_vec_pri("sparse", "p11_payload_only", "bler"), "candidate": _c_vec_pri("sparse", "p11_chained_wavefront", "bler")},
        },
        "primary",
    )["contrasts"]
    cherry_holm[0].update({
        "baseline": ["sparse", "p9_jeffreys_student_t"],
        "candidate": ["sparse", "p11_chained_wavefront"],
        "metric": "info_ber",
        "regimes": ["full_reset"],
        "expected_sign": -1,
    })
    cherry_holm[1].update({"baseline": ["sparse", "p11_payload_only"], "candidate": ["sparse", "p11_chained_wavefront"]})
    bad_holm_cherry["tests"] = cherry_holm

    # (f) Malformed tests entry ["not_a_dict"] must raise ValueError (never AttributeError)
    bad_tests_str = json.loads(json.dumps(pri_sum))
    bad_tests_str["tests"] = ["not_a_dict"]

    for label, bad_obj in (
        ("ni_full", bad_ni_full),
        ("cpb", bad_cpb),
        ("ni_metric", bad_ni_metric),
        ("holm_arm", bad_holm_arm),
        ("holm_cherry", bad_holm_cherry),
        ("tests_str", bad_tests_str),
    ):
        bad_p = tmp_path / f"bad_c21_{label}.json"
        bad_p.write_text(json.dumps(bad_obj), encoding="utf-8", newline="\n")
        with pytest.raises(ValueError):
            p11_audit.audit_p11_primary(ds["raw"], bad_p, ds["replay"], require_canonical_family=True)

    # 2. Secondary Off-Grid canonical audit checks (C2-1)
    og_dir = tmp_path / "og_c21"
    og_dir.mkdir()
    og_raw = og_dir / "offgrid.jsonl"
    og_sum = og_dir / "offgrid_sum.json"
    og_arms = ("em10_transport_attn", "em10_abs_no_transport", "em10_local", "em10_no_history")
    og_recs = []
    for c in range(1, 5):
        for r_idx, reg in enumerate(OFFGRID_ALL_REGIMES):
            seed = assert_non_formal_e2e_seed(SEED_P11_E2E + 120 + 1000 * c + r_idx)
            sp = {}
            for arm, errs in zip(og_arms, (2, 1, 4, 5)):
                sp[arm] = {
                    "blocks": 18, "block_errors": errs, "coded_bits": 2304, "coded_errors": errs * 10,
                    "info_bits": 1152, "info_errors": errs * 5, "extra_blocks": 0, "extra_errors": 0,
                    "probes": 32, "probe_sse": 4.0, "clean_sse": 1.0, "probe_energy": 32.0,
                    "clean_energy": 30.0, "variance_sum": 4.0, "nll_sum": -10.0,
                }
            og_recs.append({"cluster": c, "link": r_idx, "seed": seed, "regime": reg, "budgets": {"sparse": sp}})
    og_raw.write_text("\n".join(json.dumps(r) for r in og_recs) + "\n", encoding="utf-8", newline="\n")

    s_map = {"sparse": og_arms}
    sc = lambda a, regs: np.asarray([
        p11_audit.aggregate_p11_cells([r["budgets"]["sparse"][a] for r in og_recs if r["cluster"] == c and r["regime"] in regs])["bler"]
        for c in range(1, 5)
    ], dtype=np.float64)
    og_id_tests = p11_audit._audit_single_holm_family(
        {
            "transport_attn_vs_abs_no_transport": {"baseline": sc("em10_abs_no_transport", OFFGRID_REGIMES_ID), "candidate": sc("em10_transport_attn", OFFGRID_REGIMES_ID), "regimes": list(OFFGRID_REGIMES_ID)},
            "transport_attn_vs_local": {"baseline": sc("em10_local", OFFGRID_REGIMES_ID), "candidate": sc("em10_transport_attn", OFFGRID_REGIMES_ID), "regimes": list(OFFGRID_REGIMES_ID)},
            "transport_attn_vs_no_history": {"baseline": sc("em10_no_history", OFFGRID_REGIMES_ID), "candidate": sc("em10_transport_attn", OFFGRID_REGIMES_ID), "regimes": list(OFFGRID_REGIMES_ID)},
        },
        "offgrid",
    )["contrasts"]
    og_id_tests[0].update({"baseline": ["sparse", "em10_abs_no_transport"], "candidate": ["sparse", "em10_transport_attn"]})
    og_id_tests[1].update({"baseline": ["sparse", "em10_local"], "candidate": ["sparse", "em10_transport_attn"]})
    og_id_tests[2].update({"baseline": ["sparse", "em10_no_history"], "candidate": ["sparse", "em10_transport_attn"]})

    og_ood_tests = p11_audit._audit_single_holm_family(
        {
            "ood_transport_attn_vs_abs_no_transport": {"baseline": sc("em10_abs_no_transport", OFFGRID_REGIMES_OOD), "candidate": sc("em10_transport_attn", OFFGRID_REGIMES_OOD), "regimes": list(OFFGRID_REGIMES_OOD)},
            "ood_transport_attn_vs_local": {"baseline": sc("em10_local", OFFGRID_REGIMES_OOD), "candidate": sc("em10_transport_attn", OFFGRID_REGIMES_OOD), "regimes": list(OFFGRID_REGIMES_OOD)},
            "ood_transport_attn_vs_no_history": {"baseline": sc("em10_no_history", OFFGRID_REGIMES_OOD), "candidate": sc("em10_transport_attn", OFFGRID_REGIMES_OOD), "regimes": list(OFFGRID_REGIMES_OOD)},
        },
        "offgrid_ood",
    )["contrasts"]
    og_ood_tests[0].update({"baseline": ["sparse", "em10_abs_no_transport"], "candidate": ["sparse", "em10_transport_attn"]})
    og_ood_tests[1].update({"baseline": ["sparse", "em10_local"], "candidate": ["sparse", "em10_transport_attn"]})
    og_ood_tests[2].update({"baseline": ["sparse", "em10_no_history"], "candidate": ["sparse", "em10_transport_attn"]})

    og_sum_obj = {
        "totals": p11_audit._aggregate_nested_budget_arm(og_recs, ("sparse",), s_map),
        "id_equal": p11_audit._aggregate_nested_budget_arm([r for r in og_recs if r["regime"] in OFFGRID_REGIMES_ID], ("sparse",), s_map),
        "ood_equal": p11_audit._aggregate_nested_budget_arm([r for r in og_recs if r["regime"] in OFFGRID_REGIMES_OOD], ("sparse",), s_map),
        "clusters": [p11_audit._aggregate_nested_budget_arm([r for r in og_recs if r["cluster"] == c], ("sparse",), s_map) for c in range(1, 5)],
        "tests": og_id_tests,
        "ood_tests": og_ood_tests,
    }
    og_sum.write_text(json.dumps(og_sum_obj, indent=2) + "\n", encoding="utf-8", newline="\n")
    assert p11_audit.audit_p11_offgrid(og_raw, og_sum, require_canonical_family=True)["clusters"] == 4

    # Swap baseline to em10_no_history on ID or OOD contrast with matching recomputed stats -> must raise ValueError
    spoof_id = json.loads(json.dumps(og_sum_obj))
    spoof_id["tests"][0] = {**og_id_tests[2], "name": "transport_attn_vs_abs_no_transport"}
    spoof_id_p = og_dir / "spoof_id.json"
    spoof_id_p.write_text(json.dumps(spoof_id), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError):
        p11_audit.audit_p11_offgrid(og_raw, spoof_id_p, require_canonical_family=True)

    spoof_ood = json.loads(json.dumps(og_sum_obj))
    spoof_ood["ood_tests"][0] = {**og_ood_tests[2], "name": "ood_transport_attn_vs_abs_no_transport"}
    spoof_ood_p = og_dir / "spoof_ood.json"
    spoof_ood_p.write_text(json.dumps(spoof_ood), encoding="utf-8", newline="\n")
    with pytest.raises(ValueError):
        p11_audit.audit_p11_offgrid(og_raw, spoof_ood_p, require_canonical_family=True)


def test_t5_11_anti_leakage_gate_shapes_and_probe_decoded_shape() -> None:
    p11_indep = importlib.import_module("p11.independent_ref")
    p11_audit = importlib.import_module("p11.audit")
    pr = importlib.import_module("p11.payload_rebuild")
    wf = importlib.import_module("p11.wavefront_pipeline")

    obs, _ = make_e2e_public_obs(0, snr_db=2.0, regime="continuous", budget="sparse")
    clean_fn = p11_audit._make_clean_cstar_assimilator()

    # 1 (C2-2). Malformed GateOutput.history_weight shape (1,) rejected with allow_historical_adapter=False
    def _bad_hw_shape(o):
        res = clean_fn(o)
        bad_g = dataclasses.replace(res.gate_out, history_weight=np.array([0.5], dtype=REAL_DTYPE))
        return dataclasses.replace(res, gate_out=bad_g)

    rep_hw = p11_indep.audit_assimilation_anti_leakage(
        _bad_hw_shape, obs, allow_historical_adapter=False, strict=False
    )
    assert rep_hw["passed"] is False

    # 2 (C2-2). Probe poison returning decoded_info[1] with shape (1, 64) rejected (no silent broadcast)
    def _is_poisoned(o) -> bool:
        return not np.allclose(o.y_b[PROBE_ROW], obs.y_b[PROBE_ROW], equal_nan=True)

    def _bad_decoded_shape_on_poison(o):
        res = clean_fn(o)
        if not _is_poisoned(o):
            return res
        di = dict(res.decoded_info)
        di[1] = di[1].reshape(1, 64).copy()
        return dataclasses.replace(res, decoded_info=di)

    rep_shape = p11_indep.audit_assimilation_anti_leakage(_bad_decoded_shape_on_poison, obs, strict=False)
    assert rep_shape["passed"] is False

    # 3 (C2-3). InheritedOrShiftedBranch.refit rejects bool ledger key
    a_prior = pr.build_frame_a_prior_clean(obs.y_a, obs.n0)
    c_b, y10_c, _ = pr.b_only_canonical_gauge(obs.y_b[ANCHOR], obs.n0)
    dyn = pr.ShiftedBranchDynamics(a_prior, changed_ray=None, delta_freq=0.0, onset_row=0)
    branch = wf.InheritedOrShiftedBranch(
        kind="continuous", dynamics_c=dyn, gauge=c_b, n0=float(obs.n0), y10_c=y10_c
    )
    with pytest.raises(TypeError):
        branch.refit({True: (np.ones(N, dtype=COMPLEX_DTYPE), obs.y_b[1].copy(), float(obs.n0))})  # type: ignore[dict-item]
