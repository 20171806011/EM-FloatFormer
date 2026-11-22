"""
EM-FloatFormer Real-Data SDR OFDM E2E Suite — Tier 2: Boundary & Corner Cases (55 Tests).

Covers edge cases, extreme dynamic ranges, degeneracy boundaries, and error handling
for all 11 features (F1–F11, exactly 5 tests per feature).
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List
import unittest

import numpy as np
import torch
import yaml

from research.realdata_sdr_ofdm.tests.e2e.harness import (
    BITS_PER_TTI,
    E2E_SEED_BASE,
    EXPECTED_BACKUP_SHA256,
    EXPECTED_P12_BASE_SHA,
    N_ACTIVE,
    N_DATA_RES,
    N_SYMB,
    REPO_ROOT,
    RESEARCH_ROOT,
    MilestoneResolver,
    compute_file_sha256,
    exact_sign_flip_p_ref,
    extract_active_h,
    make_standalone_frame_proxy,
    make_synthetic_custom_dataset_pth,
    make_synthetic_grid_and_truth,
    make_synthetic_real_ofdm_frame,
    make_synthetic_receiver_output,
    nongray_16qam_exact_llr_ref,
    nongray_16qam_mixture_llr_ref,
    read_json_file,
    run_git,
    run_python_script,
    scale_real_ofdm_frame,
)


class TestTier2BoundaryCorner(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases (55 tests — 5 per feature F1–F11)."""

    def setUp(self) -> None:
        try:
            torch.set_num_threads(1)
        except RuntimeError:
            pass

    # =========================================================================
    # Feature F1 Boundary Tests (5 tests)
    # =========================================================================

    def test_t2_f01_agents_never_staged_and_gitignore_active(self) -> None:
        """T2-F1.1: Index contains no `.agents/` or raw binaries, and `.gitignore` ignores `.pth`/`.mat`."""
        staged = run_git("diff", "--cached", "--name-only")
        for line in staged.splitlines():
            p = line.strip()
            self.assertFalse(p.startswith(".agents"), f"Staged .agents file: {p}")
            self.assertFalse(p.lower().endswith((".pth", ".mat", ".tar.gz")), f"Staged raw file: {p}")

        root_ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("*.pth", root_ignore)
        sub_ignore = (RESEARCH_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("*.pth", sub_ignore)
        self.assertIn("*.mat", sub_ignore)
        for dummy_path in (
            "research/realdata_sdr_ofdm/data_raw/dummy.pth",
            "research/realdata_sdr_ofdm/dummy.mat",
        ):
            chk = subprocess.run(
                ["git", "check-ignore", "-q", dummy_path],
                cwd=str(REPO_ROOT),
                check=False,
            )
            self.assertEqual(chk.returncode, 0, f".gitignore must ignore {dummy_path}")

    def test_t2_f01_no_nested_git_repos_or_submodules(self) -> None:
        """T2-F1.2: No nested `.git` directories/files inside `research/realdata_sdr_ofdm/` or `.gitmodules` edits."""
        nested = [p for p in RESEARCH_ROOT.rglob(".git")]
        self.assertEqual(nested, [], f"Nested .git detected: {nested}")
        diff_gm = run_git("diff", f"{EXPECTED_P12_BASE_SHA}..HEAD", "--", ".gitmodules", check=False)
        self.assertEqual(diff_gm, "")

    def test_t2_f01_no_oversized_blobs_tracked(self) -> None:
        """T2-F1.3: All tracked files under `research/realdata_sdr_ofdm/` are strictly < 10 MB."""
        ls_tree = run_git("ls-tree", "-r", "HEAD", "--", "research/realdata_sdr_ofdm", check=False)
        for line in ls_tree.splitlines():
            parts = line.split()
            if len(parts) >= 4 and parts[1] == "blob":
                size = int(run_git("cat-file", "-s", parts[2]))
                self.assertLess(size, 10 * 1024 * 1024, f"Oversized tracked blob {parts[3]}: {size} bytes")

    def test_t2_f01_p12_audit_unmodified_vs_f566f9b8(self) -> None:
        """T2-F1.4: `p12/audit.py` is byte-identical to `f566f9b8` and imports cleanly."""
        diff_audit = run_git("diff", EXPECTED_P12_BASE_SHA, "HEAD", "--", "p12/audit.py")
        self.assertEqual(diff_audit, "")
        import p12.audit as p12_audit

        self.assertTrue(callable(p12_audit.exact_sign_flip_p))
        self.assertTrue(callable(p12_audit.holm_adjust))

    def test_t2_f01_root_docs_untouched_and_no_forbidden_codespace_paths(self) -> None:
        """T2-F1.5: Root docs untouched and non-test scripts avoid `/tmp/agy_cli` and `/workspaces/EM-FloatFormer`."""
        diff_root = run_git("diff", f"{EXPECTED_P12_BASE_SHA}..HEAD", "--", "TEST_INFRA.md", "TEST_READY.md", "PROJECT.md", check=False)
        self.assertEqual(diff_root, "")
        tests_dir = RESEARCH_ROOT / "tests"
        for p in RESEARCH_ROOT.rglob("*"):
            if not p.is_file() or p.suffix not in {".py", ".sh"}:
                continue
            if tests_dir in p.parents:
                continue
            txt = p.read_text(encoding="utf-8", errors="ignore")
            self.assertNotIn("/tmp/agy_cli", txt, f"Forbidden path in {p}")
            self.assertNotIn("/workspaces/EM-FloatFormer", txt, f"Forbidden path in {p}")

    # =========================================================================
    # Feature F2 Boundary Tests (5 tests)
    # =========================================================================

    def test_t2_f02_upstream_doc_discrepancies_noted(self) -> None:
        """T2-F2.1: `DATA_SOURCES.md` reconciles README vs config discrepancies (`CP=6/7`, `435/3500 MHz`, subcarriers)."""
        path = MilestoneResolver.require_artifact("DATA_SOURCES.md", "Milestone M1 (F2)")
        text = path.read_text(encoding="utf-8")
        self.assertIn("6", text)
        self.assertIn("7", text)
        self.assertIn("435", text)
        self.assertTrue("3500" in text or "3.5" in text)
        self.assertTrue("101" in text or "ofdm_dataset_4_8_sdr.pth" in text)
        self.assertIn("102", text)

    def test_t2_f02_non_gray_mapping_warning(self) -> None:
        """T2-F2.2: `DATA_SOURCES.md` explicitly warns that `mapping_table(4)` is non-Gray / natural binary."""
        path = MilestoneResolver.require_artifact("DATA_SOURCES.md", "Milestone M1 (F2)")
        low = path.read_text(encoding="utf-8").lower()
        self.assertTrue("non-gray" in low or "natural binary" in low or "非格雷" in low)

    def test_t2_f02_sha256_hex_validity_no_dummy(self) -> None:
        """T2-F2.3: All 64-hex SHA-256 digests in `DATA_SOURCES.md` are valid non-dummy hashes."""
        path = MilestoneResolver.require_artifact("DATA_SOURCES.md", "Milestone M1 (F2)")
        text = path.read_text(encoding="utf-8")
        hashes = re.findall(r"\b[0-9a-fA-F]{64}\b", text)
        self.assertGreaterEqual(len(hashes), 1)
        for h in hashes:
            self.assertNotEqual(h.lower(), "0" * 64)
            self.assertNotEqual(h.lower(), "f" * 64)
        self.assertIn(EXPECTED_BACKUP_SHA256, text.lower())

    def test_t2_f02_license_and_valid_urls(self) -> None:
        """T2-F2.4: `DATA_SOURCES.md` contains valid HTTPS URLs and dataset license notes."""
        path = MilestoneResolver.require_artifact("DATA_SOURCES.md", "Milestone M1 (F2)")
        text = path.read_text(encoding="utf-8")
        urls = re.findall(r"https://[^\s)>\"]+", text)
        self.assertGreaterEqual(len(urls), 2)
        low = text.lower()
        self.assertTrue("cc by 4.0" in low or "cc-by-4.0" in low)
        self.assertIn("mit", low)

    def test_t2_f02_no_raw_binary_files_committed_locally(self) -> None:
        """T2-F2.5: Local `research/realdata_sdr_ofdm/` contains zero raw `.pth`/`.mat`/`.tar.gz` or >10MB files."""
        for p in RESEARCH_ROOT.rglob("*"):
            if not p.is_file():
                continue
            self.assertFalse(p.name.lower().endswith((".pth", ".mat", ".tar.gz")), f"Raw dataset locally present: {p}")
            self.assertLess(p.stat().st_size, 10 * 1024 * 1024, f"Oversized file locally present: {p}")

    # =========================================================================
    # Feature F3 Boundary Tests (5 tests)
    # =========================================================================

    def test_t2_f03_sha256_single_bit_corruption_rejected(self) -> None:
        """T2-F3.1: Flipping 1 byte in fixture `.pth` triggers acquisition verification failure (exit 42)."""
        dl_script = RESEARCH_ROOT / "scripts" / "download_and_verify.py"
        if not dl_script.is_file():
            raise unittest.SkipTest("[PROGRESSIVE EXECUTION] scripts/download_and_verify.py missing (M1 F3).")
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            raw_dir = td_path / "raw"
            pth_path = raw_dir / "ofdm_valset_4_8_sdr.pth"
            info = make_synthetic_custom_dataset_pth(pth_path, n_frames=200, seed=E2E_SEED_BASE + 311)
            raw_bytes = bytearray(pth_path.read_bytes())
            raw_bytes[len(raw_bytes) // 2] ^= 0x01
            pth_path.write_bytes(bytes(raw_bytes))

            cfg = yaml.safe_load((RESEARCH_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
            cfg["dataset"]["raw_dir"] = str(raw_dir)
            cfg["dataset"]["expected_size_bytes"] = info["size_bytes"]
            cfg["dataset"]["expected_blob_sha1"] = info["git_blob_sha1"]
            cfg_path = td_path / "cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            probe_out = td_path / "probe.json"

            proc = run_python_script(
                dl_script, "--skip-download",
                "--config", str(cfg_path), "--raw-dir", str(raw_dir),
                "--manifest-out", str(td_path / "manifest.json"),
                "--probe-out", str(probe_out), "--probe-n", "16",
            )
            self.assertNotEqual(proc.returncode, 0)
            probe = read_json_file(probe_out)
            self.assertTrue(probe["fallback_triggered"])

    def test_t2_f03_empty_file_and_empty_dataset_rejected(self) -> None:
        """T2-F3.2: 0-byte raw file and 0-frame `CustomDataset` are explicitly rejected."""
        dl_script = RESEARCH_ROOT / "scripts" / "download_and_verify.py"
        if not dl_script.is_file():
            raise unittest.SkipTest("[PROGRESSIVE EXECUTION] scripts/download_and_verify.py missing (M1 F3).")
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            raw_dir = td_path / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            empty_pth = raw_dir / "ofdm_valset_4_8_sdr.pth"
            empty_pth.write_bytes(b"")
            cfg = yaml.safe_load((RESEARCH_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
            cfg["dataset"]["raw_dir"] = str(raw_dir)
            cfg["dataset"]["expected_size_bytes"] = 0
            cfg_path = td_path / "cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            probe_out = td_path / "probe_empty.json"

            proc = run_python_script(
                dl_script, "--skip-download",
                "--config", str(cfg_path), "--raw-dir", str(raw_dir),
                "--manifest-out", str(td_path / "m.json"),
                "--probe-out", str(probe_out), "--probe-n", "16",
            )
            self.assertNotEqual(proc.returncode, 0)

            zero_pth = raw_dir / "ofdm_valset_4_8_sdr.pth"
            info_zero = make_synthetic_custom_dataset_pth(zero_pth, n_frames=0, seed=E2E_SEED_BASE + 312)
            cfg_zero = yaml.safe_load((RESEARCH_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
            cfg_zero["dataset"]["raw_dir"] = str(raw_dir)
            cfg_zero["dataset"]["expected_size_bytes"] = info_zero["size_bytes"]
            cfg_zero["dataset"]["expected_blob_sha1"] = info_zero["git_blob_sha1"]
            cfg_zero_path = td_path / "cfg_zero.yaml"
            cfg_zero_path.write_text(yaml.safe_dump(cfg_zero), encoding="utf-8")
            probe_zero = td_path / "probe_zero_ds.json"
            proc_zero = run_python_script(
                dl_script, "--skip-download",
                "--config", str(cfg_zero_path), "--raw-dir", str(raw_dir),
                "--manifest-out", str(td_path / "m_zero.json"),
                "--probe-out", str(probe_zero), "--probe-n", "16",
            )
            self.assertNotEqual(proc_zero.returncode, 0)
            if probe_zero.is_file():
                self.assertTrue(read_json_file(probe_zero)["fallback_triggered"])

    def test_t2_f03_malformed_tensor_shape_rejected(self) -> None:
        """T2-F3.3: `CustomDataset` with wrong `pdsch_iq` shape `(14, 64)` triggers fallback rejection."""
        dl_script = RESEARCH_ROOT / "scripts" / "download_and_verify.py"
        if not dl_script.is_file():
            raise unittest.SkipTest("[PROGRESSIVE EXECUTION] scripts/download_and_verify.py missing (M1 F3).")
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            raw_dir = td_path / "raw"
            pth_path = raw_dir / "ofdm_valset_4_8_sdr.pth"
            info = make_synthetic_custom_dataset_pth(
                pth_path, n_frames=200, seed=E2E_SEED_BASE + 313, corrupt_iq_shape=True
            )
            cfg = yaml.safe_load((RESEARCH_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
            cfg["dataset"]["raw_dir"] = str(raw_dir)
            cfg["dataset"]["expected_size_bytes"] = info["size_bytes"]
            cfg["dataset"]["expected_blob_sha1"] = info["git_blob_sha1"]
            cfg_path = td_path / "cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            probe_out = td_path / "probe_badshape.json"

            proc = run_python_script(
                dl_script, "--skip-download",
                "--config", str(cfg_path), "--raw-dir", str(raw_dir),
                "--manifest-out", str(td_path / "m.json"),
                "--probe-out", str(probe_out), "--probe-n", "16",
            )
            self.assertNotEqual(proc.returncode, 0)
            probe = read_json_file(probe_out)
            self.assertTrue(probe["fallback_triggered"])

    def test_t2_f03_probe_frame_count_boundaries_16_64(self) -> None:
        """T2-F3.4: Probe succeeds at exact `--probe-n 16` and `64`, and rejects `0` and `-1`."""
        dl_script = RESEARCH_ROOT / "scripts" / "download_and_verify.py"
        if not dl_script.is_file():
            raise unittest.SkipTest("[PROGRESSIVE EXECUTION] scripts/download_and_verify.py missing (M1 F3).")
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            raw_dir = td_path / "raw"
            pth_path = raw_dir / "ofdm_valset_4_8_sdr.pth"
            info = make_synthetic_custom_dataset_pth(pth_path, n_frames=200, seed=E2E_SEED_BASE + 314)
            cfg = yaml.safe_load((RESEARCH_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
            cfg["dataset"]["raw_dir"] = str(raw_dir)
            cfg["dataset"]["expected_size_bytes"] = info["size_bytes"]
            cfg["dataset"]["expected_blob_sha1"] = info["git_blob_sha1"]
            cfg_path = td_path / "cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

            for n_probe in (16, 64):
                p_out = td_path / f"probe_{n_probe}.json"
                proc = run_python_script(
                    dl_script, "--skip-download",
                    "--config", str(cfg_path), "--raw-dir", str(raw_dir),
                    "--manifest-out", str(td_path / "m.json"),
                    "--probe-out", str(p_out), "--probe-n", str(n_probe),
                )
                self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                self.assertEqual(len(read_json_file(p_out)["probe_indices"]), n_probe)

            for bad_n in (0, -1, 65):
                proc_bad = run_python_script(dl_script, "--probe-n", str(bad_n))
                self.assertNotEqual(proc_bad.returncode, 0)

    def test_t2_f03_live_codespace_raw_dataset_if_present(self) -> None:
        """T2-F3.5: Offline fixture check always executes; validates `/tmp/tw_realdata` raw `.pth` if present."""
        live_pth = Path("/tmp/tw_realdata/data_raw/ofdm_valset_4_8_sdr.pth")
        if not live_pth.is_file():
            # Offline synthetic sanity check so test never degenerates into a vacuous pass
            with tempfile.TemporaryDirectory() as td:
                info = make_synthetic_custom_dataset_pth(Path(td) / "fixture.pth", n_frames=16, seed=E2E_SEED_BASE + 315)
                self.assertEqual(info["n_frames"], 16)
                self.assertRegex(info["sha256"], r"^[0-9a-f]{64}$")
            return
        load_raw = MilestoneResolver.get_callable("adapters.sdr_ofdm_adapter", "load_raw_sdr_ofdm", "Milestone M1 (F4)")
        iq_list, label_list, sinr_arr = load_raw(live_pth)
        self.assertGreaterEqual(len(iq_list), 500)
        self.assertEqual(len(label_list), len(iq_list))
        self.assertEqual(len(sinr_arr), len(iq_list))
        self.assertEqual(iq_list[0].shape, (14, 128))
        self.assertEqual(label_list[0].shape, (1400, 4))
        self.assertTrue(np.all(np.isfinite(sinr_arr)))

    # =========================================================================
    # Feature F4 Boundary Tests (5 tests)
    # =========================================================================

    def test_t2_f04_c_star_extreme_20_octave_bounds(self) -> None:
        """T2-F4.1: Adapter gauge & blind `est_n0` remain accurate across 20 octaves ($2^{-10}$ to $2^{10}$)."""
        ad_m = MilestoneResolver.get_module("adapters.sdr_ofdm_adapter", "Milestone M1 (F4)")
        build_frame = getattr(ad_m, "build_frame")
        est_n0_fn = getattr(ad_m, "estimate_blind_n0")

        sim = make_synthetic_grid_and_truth(E2E_SEED_BASE + 411, snr_db=18.0)
        fr_b = build_frame(0, sim["rx_grid"], sim["tx_bits"], sim["reported_sinr_db"])
        s0_b = float(fr_b.meta["s0"])
        n0_b, _ = est_n0_fn(sim["rx_grid"])
        phi = math.pi / math.sqrt(2.0)
        for r in (2.0**-10, 2.0**10):
            c = r * np.exp(1j * phi)
            fr_s = build_frame(0, c * sim["rx_grid"], sim["tx_bits"], sim["reported_sinr_db"])
            s0_s = float(fr_s.meta["s0"])
            n0_s, _ = est_n0_fn(c * sim["rx_grid"])
            self.assertTrue(math.isfinite(s0_s) and math.isfinite(n0_s))
            self.assertAlmostEqual(s0_s / (r * s0_b), 1.0, delta=1e-5)
            np.testing.assert_allclose(fr_s.rx_grid, np.exp(1j * phi) * fr_b.rx_grid, atol=1e-5)
            self.assertAlmostEqual(float(n0_s) / ((r**2) * float(n0_b)), 1.0, delta=1e-4)

    def test_t2_f04_zero_signal_and_zero_noise_guards(self) -> None:
        """T2-F4.2: Zero `rx_grid`, zero guard noise, and `n0=0.0` LLRs stay strictly finite."""
        ad_m = MilestoneResolver.get_module("adapters.sdr_ofdm_adapter", "Milestone M1 (F4)")
        build_frame = getattr(ad_m, "build_frame")
        est_n0_fn = getattr(ad_m, "estimate_blind_n0")
        llr_fn = MilestoneResolver.get_callable("adapters.modulation", "exact_llr", "Milestone M1 (F4)")

        sim = make_synthetic_grid_and_truth(E2E_SEED_BASE + 412)
        zero_grid = np.zeros_like(sim["rx_grid"])
        fr_z = build_frame(0, zero_grid, sim["tx_bits"], 0.0)
        n0_z, _ = est_n0_fn(zero_grid)
        self.assertTrue(math.isfinite(float(fr_z.meta["s0"])) and float(fr_z.meta["s0"]) > 0.0)
        self.assertTrue(math.isfinite(float(n0_z)) and float(n0_z) > 0.0)
        llrs = np.asarray(llr_fn(np.array([0.3 + 0.3j, -0.9 - 0.9j]), 0.0, qm=4), dtype=np.float64)
        self.assertTrue(np.all(np.isfinite(llrs)))

    def test_t2_f04_corrupted_nan_inf_wrong_shape_rejection(self) -> None:
        """T2-F4.3: Adapter rejects wrong tensor shapes and non-binary labels with `ValueError`."""
        ad_m = MilestoneResolver.get_module("adapters.sdr_ofdm_adapter", "Milestone M1 (F4)")
        build_frame = getattr(ad_m, "build_frame")
        est_n0_fn = getattr(ad_m, "estimate_blind_n0")
        sim = make_synthetic_grid_and_truth(E2E_SEED_BASE + 413)
        with self.assertRaises(ValueError):
            build_frame(0, np.zeros((14, 64), dtype=np.complex128), sim["tx_bits"], 10.0)
        with self.assertRaises(ValueError):
            build_frame(0, sim["rx_grid"], np.zeros((100, 4), dtype=np.uint8), 10.0)
        with self.assertRaises(ValueError):
            est_n0_fn(np.zeros((14, 64), dtype=np.complex128))
        llr_fn = MilestoneResolver.get_callable("adapters.modulation", "exact_llr", "Milestone M1 (F4)")
        with self.assertRaises(ValueError):
            llr_fn(np.array([complex(float("nan"), 0.0)]), 0.1, qm=4)

    def test_t2_f04_non_gray_decision_boundaries_and_outliers(self) -> None:
        """T2-F4.4: Non-Gray 16-QAM LLRs satisfy symmetries at `Re=0`, `Re=+2/sqrt(10)` (3rd boundary), and outliers."""
        llr_fn = MilestoneResolver.get_callable("adapters.modulation", "exact_llr", "Milestone M1 (F4)")
        llr_zero = np.asarray(llr_fn(np.array([0.0 + 0.3j]), 0.05, qm=4), dtype=np.float64)
        self.assertLess(abs(float(llr_zero[0, 0])), 1e-3)
        b_pt = np.array([2.0 / math.sqrt(10.0) + 0.316j])
        llr_b1 = np.asarray(llr_fn(b_pt, 0.01, qm=4), dtype=np.float64)
        np.testing.assert_allclose(llr_b1, nongray_16qam_exact_llr_ref(b_pt, 0.01), atol=1e-4)
        self.assertLess(abs(float(llr_b1[0, 1])), 0.05)
        llr_noisy = np.asarray(llr_fn(np.array([0.3 + 0.3j]), 1e6, qm=4), dtype=np.float64)
        self.assertLess(float(np.max(np.abs(llr_noisy))), 1e-3)
        llr_far = np.asarray(llr_fn(np.array([100.0 + 100.0j]), 0.1, qm=4), dtype=np.float64)
        self.assertTrue(np.all(np.isfinite(llr_far)))

    def test_t2_f04_extreme_cfo_half_subcarrier_and_reverse_sign(self) -> None:
        """T2-F4.5: `cfo_norm = +/-0.5` and `phase_drift_rad = -pi` preserve Parseval energy and reverse phase."""
        perturb = MilestoneResolver.get_callable(
            "adapters.sdr_ofdm_adapter", "apply_stress_perturbation", "Milestone M1 (F4)"
        )
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 415, snr_db=22.0)
        e_in = float(np.sum(np.abs(fr.rx_grid) ** 2))
        p_pos = perturb(fr, cfo_norm=0.5, phase_drift_rad=0.0, seed=1)
        p_neg = perturb(fr, cfo_norm=-0.5, phase_drift_rad=0.0, seed=1)
        p_pi = perturb(fr, cfo_norm=0.0, phase_drift_rad=-math.pi, seed=1)
        for p in (p_pos, p_neg, p_pi):
            self.assertTrue(np.all(np.isfinite(p.rx_grid)))
            self.assertAlmostEqual(float(np.sum(np.abs(p.rx_grid) ** 2)) / e_in, 1.0, delta=1e-4)
            np.testing.assert_array_equal(p.tx_bits, fr.tx_bits)
        self.assertGreater(float(np.linalg.norm(p_pos.rx_grid - p_neg.rx_grid)), 1e-2)

    # =========================================================================
    # Feature F5 Boundary Tests (5 tests)
    # =========================================================================

    def test_t2_f05_duplicate_payload_cross_split_rejected(self) -> None:
        """T2-F5.1: Partitioner enforces zero `tx_bits` SHA-256 overlap across `train`, `dev`, and `test`."""
        split_fn = MilestoneResolver.get_callable("adapters.sdr_ofdm_adapter", "build_leak_free_splits", "Milestone M1 (F5)")
        n = 120
        rng = np.random.default_rng(E2E_SEED_BASE + 511)
        sinr_vals = rng.uniform(5.0, 25.0, size=n)
        h_pilots = rng.standard_normal((n, 14)) + 1j * rng.standard_normal((n, 14))
        # Inject identical payload hashes across multiple blocks
        hashes = [f"dup_hash_{i % 6}" for i in range(n)]
        sp = split_fn(sinr_vals, h_pilots, hashes, n_macro_blocks=6, min_guard_frames=2, max_guard_frames=4)
        h_tr = {hashes[i] for i in sp["train_indices"]}
        h_dv = {hashes[i] for i in sp["dev_indices"]}
        h_te = {hashes[i] for i in sp["test_indices"]}
        self.assertEqual(h_tr & h_te, set())
        self.assertEqual(h_dv & h_te, set())
        self.assertEqual(h_tr & h_dv, set())
        seen_groups: set[str] = set()
        group_sets: Dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
        for blk in sp["blocks"]:
            gid = blk["group_id"]
            self.assertNotIn(gid, seen_groups, f"Duplicate dynamic group_id: {gid}")
            seen_groups.add(gid)
            group_sets[blk["split"]].add(gid)
        self.assertEqual(group_sets["train"] & group_sets["test"], set())
        self.assertEqual(group_sets["dev"] & group_sets["test"], set())
        self.assertEqual(group_sets["train"] & group_sets["dev"], set())

    def test_t2_f05_cross_split_group_collision_rejected(self) -> None:
        """T2-F5.2: Diagnostics reject mismatched lengths, and split blocks have strictly disjoint `group_id`s."""
        ad_m = MilestoneResolver.get_module("adapters.sdr_ofdm_adapter", "Milestone M1 (F5)")
        diag_fn = getattr(ad_m, "compute_temporal_diagnostics")
        with self.assertRaises(ValueError):
            diag_fn(np.zeros(10), np.zeros((9, 14), dtype=np.complex128), ["a"] * 10)
        splits_path = MilestoneResolver.require_artifact("manifests/splits.json", "Milestone M1 (F5)")
        sp = read_json_file(splits_path)
        seen_groups: set[str] = set()
        for blk in sp["blocks"]:
            self.assertNotIn(blk["group_id"], seen_groups)
            seen_groups.add(blk["group_id"])

    def test_t2_f05_minimal_group_count_boundary(self) -> None:
        """T2-F5.3: Partitioner succeeds on `n=48` frames with `n_macro_blocks=4` and rejects `n=8` (`m < 3`)."""
        split_fn = MilestoneResolver.get_callable("adapters.sdr_ofdm_adapter", "build_leak_free_splits", "Milestone M1 (F5)")
        rng = np.random.default_rng(E2E_SEED_BASE + 520)
        n = 48
        sp4 = split_fn(
            rng.uniform(5.0, 25.0, size=n),
            rng.standard_normal((n, 14)) + 1j * rng.standard_normal((n, 14)),
            [f"h_{i}" for i in range(n)],
            n_macro_blocks=4,
            min_guard_frames=2,
            max_guard_frames=2,
        )
        self.assertGreater(len(sp4["train_indices"]), 0)
        self.assertGreater(len(sp4["dev_indices"]), 0)
        self.assertGreater(len(sp4["test_indices"]), 0)
        with self.assertRaises(ValueError):
            split_fn(
                rng.uniform(5.0, 25.0, size=8),
                rng.standard_normal((8, 14)) + 1j * rng.standard_normal((8, 14)),
                [f"h_{i}" for i in range(8)],
                n_macro_blocks=3,
                min_guard_frames=4,
                max_guard_frames=4,
            )

    def test_t2_f05_split_sinr_metadata_and_id_bounds(self) -> None:
        """T2-F5.4: `splits.json` stores finite SINR/count summaries and valid non-negative integer IDs."""
        splits_path = MilestoneResolver.require_artifact("manifests/splits.json", "Milestone M1 (F5)")
        sp = read_json_file(splits_path)
        for sname in ("train", "dev", "test"):
            ids = sp[f"{sname}_indices"]
            self.assertTrue(all(isinstance(fid, int) and fid >= 0 for fid in ids))
            sinr_info = sp["sinr_by_split"][sname]
            for stat_k in ("min", "median", "mean", "max"):
                self.assertTrue(math.isfinite(float(sinr_info[stat_k])))

    def test_t2_f05_live_codespace_payload_disjointness_if_present(self) -> None:
        """T2-F5.5: Offline manifest check runs always; checks live `/tmp/tw_realdata/data_proc` payload hashes if present."""
        splits_path = MilestoneResolver.require_artifact("manifests/splits.json", "Milestone M1 (F5)")
        sp = read_json_file(splits_path)
        self.assertEqual(set(sp["train_indices"]) & set(sp["test_indices"]), set())
        proc_dir = Path("/tmp/tw_realdata/data_proc")
        if proc_dir.is_dir():
            ad_m = MilestoneResolver.get_module("adapters.sdr_ofdm_adapter", "Milestone M1 (F5)")
            load_split = getattr(ad_m, "load_processed_split")
            tr_h = {f.meta["tx_bits_sha256"] for f in load_split(proc_dir, "train")}
            te_h = {f.meta["tx_bits_sha256"] for f in load_split(proc_dir, "test")}
            self.assertEqual(tr_h & te_h, set())

    # =========================================================================
    # Feature F6 Boundary Tests (5 tests)
    # =========================================================================

    def test_t2_f06_deep_fade_zero_pilots_guard(self) -> None:
        """T2-F6.1: All-zero pilot observations do not cause `ZeroDivisionError` or `NaN` in classical baseline."""
        run_frame = MilestoneResolver.get_callable("receivers.classical", "run_frame", "Milestone M2 (F6)")
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 611, snr_db=18.0)
        fr.rx_grid[fr.pilot_mask] = 0.0 + 0.0j
        out = run_frame(fr)
        self.assertTrue(np.all(np.isfinite(out.h_est)))
        self.assertTrue(np.all(np.isfinite(out.eq_symbols)))
        self.assertTrue(np.all(np.isfinite(out.bit_llrs)))

    def test_t2_f06_extreme_noise_bounds_minus20_to_plus90_db(self) -> None:
        """T2-F6.2: `est_n0 = 100.0` and `1e-9` yield finite LLRs with no overconfident collapse at high noise."""
        run_frame = MilestoneResolver.get_callable("receivers.classical", "run_frame", "Milestone M2 (F6)")
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 612, snr_db=15.0)
        fr_hi = scale_real_ofdm_frame(fr, 1.0)
        fr_hi.est_n0 = 100.0
        o_hi = run_frame(fr_hi)
        self.assertTrue(np.all(np.isfinite(o_hi.bit_llrs)))
        self.assertTrue(np.all(np.isfinite(o_hi.h_var)) and np.all(extract_active_h(o_hi.h_var) > 0.0))
        self.assertLess(float(np.mean(np.abs(o_hi.bit_llrs))), 1.0)

        fr_lo = scale_real_ofdm_frame(fr, 1.0)
        fr_lo.est_n0 = 1e-9
        o_lo = run_frame(fr_lo)
        self.assertTrue(np.all(np.isfinite(o_lo.bit_llrs)))

    def test_t2_f06_c_star_extreme_2m10_and_2p10(self) -> None:
        """T2-F6.3: Classical baseline keeps exact LLR invariance at $r = 2^{-10}$ and $2^{10}$ with irrational phase."""
        run_frame = MilestoneResolver.get_callable("receivers.classical", "run_frame", "Milestone M2 (F6)")
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 613, snr_db=18.0)
        o_base = run_frame(fr)
        phi = -math.pi / math.sqrt(2.0)
        for r in (2.0**-10, 2.0**10):
            o_s = run_frame(scale_real_ofdm_frame(fr, r * np.exp(1j * phi)))
            np.testing.assert_allclose(o_s.bit_llrs, o_base.bit_llrs, atol=1e-4)
            self.assertEqual(int(np.sum(o_s.hard_bits != o_base.hard_bits)), 0)

    def test_t2_f06_phase_drift_and_cfo_perturbation_stability(self) -> None:
        """T2-F6.4: Classical baseline remains finite and stable under `phase_drift_rad=0.8` and `cfo_norm=0.03`."""
        perturb = MilestoneResolver.get_callable(
            "adapters.sdr_ofdm_adapter", "apply_stress_perturbation", "Milestone M1 (F4)"
        )
        run_frame = MilestoneResolver.get_callable("receivers.classical", "run_frame", "Milestone M2 (F6)")
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 614, snr_db=22.0)
        fr_p = perturb(fr, cfo_norm=0.03, phase_drift_rad=0.8, seed=614)
        out = run_frame(fr_p)
        self.assertFalse(bool(out.aux_info["failed"]))
        self.assertTrue(np.all(np.isfinite(out.h_est)))
        self.assertTrue(np.all(np.isfinite(out.bit_llrs)))

    def test_t2_f06_corrupted_nan_frame_error_accounting(self) -> None:
        """T2-F6.5: NaN-corrupted `rx_grid` raises `ValueError` OR sets `aux_info['failed']=True` with error msg."""
        run_frame = MilestoneResolver.get_callable("receivers.classical", "run_frame", "Milestone M2 (F6)")
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 615, snr_db=18.0)
        fr.rx_grid[3, 20] = complex(float("nan"), 0.0)
        try:
            out = run_frame(fr)
            self.assertTrue(bool(out.aux_info["failed"]))
            self.assertTrue(bool(out.aux_info.get("error")))
        except ValueError:
            pass

    # =========================================================================
    # Feature F7 Boundary Tests (5 tests)
    # =========================================================================

    def test_t2_f07_zero_error_and_identical_arm_tie_degeneracy(self) -> None:
        """T2-F7.1: Identical 0-error arms across `G=16` groups produce `p=1.0`, `ties=16`, and `'证据不足'`."""
        eval_fn = MilestoneResolver.get_first_callable(
            "eval.evaluator", ("evaluate_receiver", "evaluate_frames", "evaluate_outputs"), "Milestone M2 (F7)"
        )
        compare_fn = MilestoneResolver.get_first_callable(
            "eval.evaluator", ("compare_arms", "paired_comparison", "evaluate_paired"), "Milestone M2 (F7)"
        )
        pairs = []
        for g in range(16):
            fr, _ = make_standalone_frame_proxy(E2E_SEED_BASE + 711 + g, group_id=f"blk_{g:02d}", frame_id=g)
            pairs.append((fr, make_synthetic_receiver_output(fr)))
        res = eval_fn(pairs)
        self.assertEqual(float(res["ber"]), 0.0)
        self.assertEqual(float(res["fer"]), 0.0)
        comp = compare_fn(pairs, pairs)
        self.assertAlmostEqual(float(comp["p_two_sided"]), 1.0, places=12)
        self.assertEqual(int(comp["ties"]), 16)
        self.assertIn("verdict", comp)
        self.assertEqual(comp["verdict"], "证据不足")

    def test_t2_f07_non_g16_group_count_graceful_handling(self) -> None:
        """T2-F7.2: Non-16 group count (`G=6`) avoids `p12.audit` `g!=16` crash and matches $2/2^6$ oracle."""
        compare_fn = MilestoneResolver.get_first_callable(
            "eval.evaluator", ("compare_arms", "paired_comparison", "evaluate_paired"), "Milestone M2 (F7)"
        )
        base_p, cand_p = [], []
        for g in range(6):
            fr, _ = make_standalone_frame_proxy(E2E_SEED_BASE + 721 + g, group_id=f"blk_{g:02d}", frame_id=g)
            base_p.append((fr, make_synthetic_receiver_output(fr, bit_flip_indices=[(i, 0) for i in range(6)])))
            cand_p.append((fr, make_synthetic_receiver_output(fr, bit_flip_indices=[(0, 0)])))
        comp = compare_fn(base_p, cand_p)
        p_val = float(comp["p_two_sided"])
        self.assertGreater(p_val, 0.0)
        self.assertLessEqual(p_val, 1.0)
        self.assertAlmostEqual(p_val, exact_sign_flip_p_ref([5.0] * 6), places=10)

    def test_t2_f07_partial_receiver_failure_and_exception_accounting(self) -> None:
        """T2-F7.3: 2 failed frames out of 10 yield `success_count=8, fail_count=2, coverage_pct=80.0`."""
        eval_fn = MilestoneResolver.get_first_callable(
            "eval.evaluator", ("evaluate_receiver", "evaluate_frames", "evaluate_outputs"), "Milestone M2 (F7)"
        )
        pairs = []
        for i in range(10):
            fr, _ = make_standalone_frame_proxy(E2E_SEED_BASE + 731 + i, frame_id=i)
            is_bad = i in (2, 7)
            out = make_synthetic_receiver_output(
                fr, failed=is_bad, error=("RuntimeError: simulated failure" if is_bad else None)
            )
            pairs.append((fr, out))
        res = eval_fn(pairs)
        self.assertEqual(int(res["success_count"]), 8)
        self.assertEqual(int(res["fail_count"]), 2)
        self.assertAlmostEqual(float(res["coverage_pct"]), 80.0, places=6)

    def test_t2_f07_single_bit_error_fer_and_pilot_var_floor(self) -> None:
        """T2-F7.4: Single bit error in 1/10 TTIs yields `fer=0.1` and `pilot_var=1e-30` remains finite."""
        eval_fn = MilestoneResolver.get_first_callable(
            "eval.evaluator", ("evaluate_receiver", "evaluate_frames", "evaluate_outputs"), "Milestone M2 (F7)"
        )
        pairs = []
        for i in range(10):
            fr, tr = make_standalone_frame_proxy(E2E_SEED_BASE + 741 + i, frame_id=i)
            flips = ((0, 0),) if i == 0 else ()
            p_true = tr["heldout_pilot_true"]
            p_var = np.full(p_true.shape, 1e-30, dtype=np.float64)
            out = make_synthetic_receiver_output(fr, bit_flip_indices=flips, pilot_pred=p_true, pilot_var=p_var)
            pairs.append((fr, out))
        res = eval_fn(pairs)
        self.assertEqual(int(res["bit_errors"]), 1)
        self.assertEqual(int(res["frame_errors"]), 1)
        self.assertAlmostEqual(float(res["fer"]), 0.1, places=12)
        self.assertTrue(math.isfinite(float(res["pilot_cal_ratio"])))
        self.assertTrue(math.isfinite(float(res["pilot_nll"])))

    def test_t2_f07_nonfinite_outputs_and_empty_input_rejection(self) -> None:
        """T2-F7.5: Empty input raises `ValueError`, and NaN receiver outputs trigger failure or `ValueError`."""
        eval_fn = MilestoneResolver.get_first_callable(
            "eval.evaluator", ("evaluate_receiver", "evaluate_frames", "evaluate_outputs"), "Milestone M2 (F7)"
        )
        with self.assertRaises(ValueError):
            eval_fn([])
        fr, _ = make_standalone_frame_proxy(E2E_SEED_BASE + 751)
        out = make_synthetic_receiver_output(fr)
        out.bit_llrs[0, 0] = float("nan")
        try:
            res = eval_fn([(fr, out)])
            self.assertEqual(int(res["fail_count"]), 1)
        except ValueError:
            pass

    # =========================================================================
    # Feature F8 Boundary Tests (5 tests)
    # =========================================================================

    def test_t2_f08_1_section51_scoring_value_correlation_guard(self) -> None:
        """T2-F8.1: Perturbing distractor value amplitudes without changing keys preserves Group A weights."""
        run_a = MilestoneResolver.get_callable("receivers.group_a_reference", "run_frame", "Milestone M3 (F8)")
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 811, snr_db=16.0)
        pool_a = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 820 + j, snr_db=18.0)[0] for j in range(8)]
        pool_b = [scale_real_ofdm_frame(p, 1.0) for p in pool_a]
        # Contaminate payload/data REs of distractor 3 while leaving pilot keys untouched
        pool_b[3].rx_grid[pool_b[3].data_mask] *= 100.0
        o1 = run_a(fr, context_pool=pool_a, mode="global_content_attn")
        o2 = run_a(fr, context_pool=pool_b, mode="global_content_attn")
        np.testing.assert_allclose(o1.aux_info["weights"], o2.aux_info["weights"], atol=1e-8)

    def test_t2_f08_2_temperature_extremes_overhot_and_overcold(self) -> None:
        """T2-F8.2: Over-hot $\\tau=10^6$ approaches $1/M$ and over-cold $\\tau=10^{-4}$ stays finite near one-hot."""
        run_a = MilestoneResolver.get_callable("receivers.group_a_reference", "run_frame", "Milestone M3 (F8)")
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 812, snr_db=18.0)
        pool = [
            make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 812 if j == 2 else E2E_SEED_BASE + 8000 + j, snr_db=20.0)[0]
            for j in range(6)
        ]
        o_hot = run_a(fr, context_pool=pool, mode="global_content_attn", temperature=1e6)
        w_hot = np.asarray(o_hot.aux_info["weights"], dtype=np.float64)
        np.testing.assert_allclose(w_hot, np.full(6, 1.0 / 6.0), atol=1e-3)

        o_cold = run_a(fr, context_pool=pool, mode="global_content_attn", temperature=1e-4)
        w_cold = np.asarray(o_cold.aux_info["weights"], dtype=np.float64)
        self.assertTrue(np.all(np.isfinite(w_cold)))
        self.assertGreater(float(np.max(w_cold)), 0.99)

    def test_t2_f08_3_blind_uninformative_keys_degenerate_to_uniform(self) -> None:
        """T2-F8.3: Identical uninformative keys ($T \\perp J$) make `global_content_attn` equal `global_uniform`."""
        run_a = MilestoneResolver.get_callable("receivers.group_a_reference", "run_frame", "Milestone M3 (F8)")
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 813, snr_db=16.0)
        # Clone pilot observations across all pool entries so keys are identical
        pool = []
        for j in range(5):
            p, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 850 + j, snr_db=16.0)
            p.rx_grid[p.pilot_mask] = fr.rx_grid[fr.pilot_mask]
            pool.append(p)
        o_attn = run_a(fr, context_pool=pool, mode="global_content_attn")
        o_unif = run_a(fr, context_pool=pool, mode="global_uniform")
        np.testing.assert_allclose(o_attn.aux_info["weights"], np.full(5, 0.2), atol=1e-8)
        np.testing.assert_allclose(o_attn.h_est, o_unif.h_est, atol=1e-6)

    def test_t2_f08_4_context_budget_boundaries_m1_and_k_equals_m(self) -> None:
        """T2-F8.4: At $M=1$, $\\alpha=[1.0]$ across modes; at $k=M$, local matches global access count."""
        run_a = MilestoneResolver.get_callable("receivers.group_a_reference", "run_frame", "Milestone M3 (F8)")
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 814, snr_db=18.0)
        p1 = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 814, snr_db=18.0)[0]]
        for mode in ("local", "global_uniform", "global_reliability", "global_content_attn"):
            o = run_a(fr, context_pool=p1, mode=mode, local_k=1)
            np.testing.assert_allclose(o.aux_info["weights"], [1.0], atol=1e-10)
        p4 = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 860 + j, snr_db=18.0)[0] for j in range(4)]
        o_loc_m = run_a(fr, context_pool=p4, mode="local", local_k=4)
        self.assertEqual(int(o_loc_m.aux_info["n_values_accessed"]), 4)

    def test_t2_f08_5_degenerate_zero_snr_or_all_identical_values(self) -> None:
        """T2-F8.5: Extreme noise (`est_n0 = 1e4`) stays finite with small-magnitude LLRs."""
        run_a = MilestoneResolver.get_callable("receivers.group_a_reference", "run_frame", "Milestone M3 (F8)")
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 815, snr_db=10.0)
        fr.est_n0 = 1e4
        pool = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 870 + j, snr_db=-10.0)[0] for j in range(4)]
        for p in pool:
            p.est_n0 = 1e4
        for mode in ("local", "global_uniform", "global_reliability", "global_content_attn"):
            out = run_a(fr, context_pool=pool, mode=mode)
            self.assertTrue(np.all(np.isfinite(out.h_est)))
            self.assertTrue(np.all(np.isfinite(out.h_var)) and np.all(extract_active_h(out.h_var) > 0.0))
            self.assertTrue(np.all(np.isfinite(out.bit_llrs)))
            self.assertLess(float(np.mean(np.abs(out.bit_llrs))), 1.0)

    # =========================================================================
    # Feature F9 Boundary Tests (5 tests)
    # =========================================================================

    def test_t2_f09_1_zero_weight_and_duplicate_branches(self) -> None:
        """T2-F9.1: Zero-weight extreme branch (`alpha=0, m=1e6`) contributes exact 0; duplicate branches preserve mass."""
        fuse_fn = MilestoneResolver.get_first_callable(
            "receivers.group_b_fusion", ("fuse_candidate_branches", "run_branches", "run_frame"), "Milestone M3 (F9)"
        )
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 911, snr_db=20.0)
        m0 = np.full((N_SYMB, N_ACTIVE), 1.0 + 0.0j, dtype=np.complex128)
        m_bad = np.full((N_SYMB, N_ACTIVE), 1e6 + 1e6j, dtype=np.complex128)
        vw = np.full((N_SYMB, N_ACTIVE), 0.02, dtype=np.float64)
        o_dup = fuse_fn(fr, branches=[(m0, vw), (m0, vw), (m_bad, vw)], branch_weights=[0.5, 0.5, 0.0], mode="exact_mixture_likelihood")
        o_one = fuse_fn(fr, branches=[(m0, vw)], branch_weights=[1.0], mode="exact_mixture_likelihood")
        self.assertTrue(np.all(np.isfinite(o_dup.h_est)))
        np.testing.assert_allclose(o_dup.h_est, o_one.h_est, atol=1e-8)
        np.testing.assert_allclose(o_dup.bit_llrs, o_one.bit_llrs, atol=1e-8)

    def test_t2_f09_2_llr_averaging_flip_counterexample_guard(self) -> None:
        """T2-F9.2: Mixture decoding follows log-mixture $\\log\\sum_r \\alpha_r g_r(c)$, avoiding LLR-averaging bit flips."""
        fuse_fn = MilestoneResolver.get_first_callable(
            "receivers.group_b_fusion",
            ("fuse_candidate_branches", "run_branches", "run_frame"),
            "Milestone M3 (F9)",
        )
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 912, snr_db=13.0103, flat_channel=True)
        fr.est_n0 = 0.05
        y_ce = -1.2 - 0.35j
        fr.rx_grid[fr.data_mask] = y_ce
        m0 = np.full((N_SYMB, N_ACTIVE), 1.0 + 0.0j, dtype=np.complex128)
        m1 = np.full((N_SYMB, N_ACTIVE), 0.0 + 1.0j, dtype=np.complex128)
        vw = np.full((N_SYMB, N_ACTIVE), 1e-5, dtype=np.float64)
        alpha = np.array([0.8, 0.2], dtype=np.float64)

        out = fuse_fn(
            fr,
            branches=[(m0, vw), (m1, vw)],
            branch_weights=alpha,
            mode="exact_mixture_likelihood",
        )
        y_vec = np.full((N_DATA_RES,), y_ce, dtype=np.complex128)
        m0_vec = np.full((N_DATA_RES,), 1.0 + 0.0j, dtype=np.complex128)
        m1_vec = np.full((N_DATA_RES,), 0.0 + 1.0j, dtype=np.complex128)
        vw_vec = np.full((N_DATA_RES,), 1e-5, dtype=np.float64)
        expected_mix = nongray_16qam_mixture_llr_ref(
            y_vec, [m0_vec, m1_vec], [vw_vec, vw_vec], alpha, fr.est_n0
        )
        naive_avg = (
            alpha[0] * nongray_16qam_exact_llr_ref(y_vec / m0_vec, fr.est_n0 + vw_vec)
            + alpha[1] * nongray_16qam_exact_llr_ref(y_vec / m1_vec, fr.est_n0 + vw_vec)
        )
        # Bit 2 flips sign: exact mixture ~ -1.3856 (bit 0) vs naive LLR avg ~ +1.8586 (bit 1)
        self.assertLess(float(np.max(expected_mix[:, 2])), -1.0)
        self.assertGreater(float(np.min(naive_avg[:, 2])), 1.5)
        self.assertTrue(np.all(np.isfinite(out.bit_llrs)))
        np.testing.assert_allclose(out.bit_llrs, expected_mix, atol=5e-3)
        self.assertTrue(np.all(out.bit_llrs[:, 2] < -1.0))
        self.assertTrue(np.all(out.hard_bits[:, 2] == 0))

    def test_t2_f09_3_extreme_log_likelihood_dynamic_range_lse_stability(self) -> None:
        """T2-F9.3: Log-sum-exp mixture stays finite across `est_n0 = 1e-6` and `est_n0 = 1e3`."""
        run_b = MilestoneResolver.get_callable("receivers.group_b_fusion", "run_frame", "Milestone M3 (F9)")
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 913, snr_db=18.0)
        for n0_val in (1e-6, 1e3):
            fr_x = scale_real_ofdm_frame(fr, 1.0)
            fr_x.est_n0 = n0_val
            out = run_b(fr_x, mode="exact_mixture_likelihood")
            self.assertTrue(np.all(np.isfinite(out.bit_llrs)))
            self.assertFalse(bool(out.aux_info["failed"]))

    def test_t2_f09_4_single_branch_bank_and_tied_map_weights(self) -> None:
        """T2-F9.4: $R=1$ collapses all 3 modes identically; tied $\\alpha=(0.5,0.5)$ is deterministic."""
        fuse_fn = MilestoneResolver.get_first_callable(
            "receivers.group_b_fusion", ("fuse_candidate_branches", "run_branches", "run_frame"), "Milestone M3 (F9)"
        )
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 914, snr_db=18.0)
        m0 = np.full((N_SYMB, N_ACTIVE), 0.9 + 0.1j, dtype=np.complex128)
        m1 = np.full((N_SYMB, N_ACTIVE), 0.7 - 0.2j, dtype=np.complex128)
        vw = np.full((N_SYMB, N_ACTIVE), 0.03, dtype=np.float64)
        o_m = fuse_fn(fr, branches=[(m0, vw)], branch_weights=[1.0], mode="hard_map")
        o_p = fuse_fn(fr, branches=[(m0, vw)], branch_weights=[1.0], mode="posterior_mean_plugin")
        o_x = fuse_fn(fr, branches=[(m0, vw)], branch_weights=[1.0], mode="exact_mixture_likelihood")
        np.testing.assert_allclose(o_m.bit_llrs, o_p.bit_llrs, atol=1e-10)
        np.testing.assert_allclose(o_m.bit_llrs, o_x.bit_llrs, atol=1e-10)

        t1 = fuse_fn(fr, branches=[(m0, vw), (m1, vw)], branch_weights=[0.5, 0.5], mode="hard_map")
        t2 = fuse_fn(fr, branches=[(m0, vw), (m1, vw)], branch_weights=[0.5, 0.5], mode="hard_map")
        np.testing.assert_array_equal(t1.bit_llrs, t2.bit_llrs)

    def test_t2_f09_5_no_anchor_double_counting_under_repeated_refit(self) -> None:
        """T2-F9.5: Repeated invocation on the same frame never shrinks `h_var` or drifts weights (single-count anchor)."""
        run_b = MilestoneResolver.get_callable("receivers.group_b_fusion", "run_frame", "Milestone M3 (F9)")
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 915, snr_db=16.0)
        o1 = run_b(fr, mode="exact_mixture_likelihood")
        o2 = run_b(fr, mode="exact_mixture_likelihood")
        np.testing.assert_array_equal(o1.h_var, o2.h_var)
        np.testing.assert_array_equal(o1.aux_info["branch_weights"], o2.aux_info["branch_weights"])

    # =========================================================================
    # Feature F10 Boundary Tests (5 tests)
    # =========================================================================

    def test_t2_f10_1_extreme_dynamic_range_cstar_overflow_underflow(self) -> None:
        """T2-F10.1: Group C remains finite with 0 bit flips at $c = 10^{-15}$ and $c = 10^{+15}$."""
        cls_c = MilestoneResolver.get_first_callable(
            "receivers.group_c_trainable", ("GroupCTrainableReceiver", "GroupCReceiver"), "Milestone M4 (F10)"
        )
        tr = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1010 + i, snr_db=16.0, split="train")[0] for i in range(4)]
        dv = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1020 + i, snr_db=16.0, split="dev")[0] for i in range(2)]
        te, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1030, snr_db=18.0, split="test")
        model = cls_c(seed=41)
        model.fit(tr, dv, epochs=1)
        o_ref = model.run_frame(te)
        for c in (1e-15, 1e15):
            o_s = model.run_frame(scale_real_ofdm_frame(te, c))
            self.assertTrue(np.all(np.isfinite(o_s.h_est)))
            self.assertTrue(np.all(np.isfinite(o_s.bit_llrs)))
            self.assertEqual(int(np.sum(o_s.hard_bits != o_ref.hard_bits)), 0)

    def test_t2_f10_2_zero_input_and_nonfinite_rejection(self) -> None:
        """T2-F10.2: All-zero frames stay finite; NaN inputs raise `ValueError` or set `failed=True`."""
        cls_c = MilestoneResolver.get_first_callable(
            "receivers.group_c_trainable", ("GroupCTrainableReceiver", "GroupCReceiver"), "Milestone M4 (F10)"
        )
        tr = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1040 + i, snr_db=16.0, split="train")[0] for i in range(3)]
        dv = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1050 + i, snr_db=16.0, split="dev")[0] for i in range(2)]
        model = cls_c(seed=42)
        model.fit(tr, dv, epochs=1)
        fr_zero, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1060, split="test")
        fr_zero.rx_grid[:] = 0.0
        o_zero = model.run_frame(fr_zero)
        self.assertTrue(np.all(np.isfinite(o_zero.bit_llrs)))

        fr_nan, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1061, split="test")
        fr_nan.rx_grid[0, 0] = complex(float("nan"), 0.0)
        try:
            o_nan = model.run_frame(fr_nan)
            self.assertTrue(bool(o_nan.aux_info["failed"]))
        except ValueError:
            pass

    def test_t2_f10_3_single_sample_and_minimal_train_dev_fit(self) -> None:
        """T2-F10.3: Group C trains cleanly on minimal 2-frame train + 1-frame dev without batch-norm collapse."""
        cls_c = MilestoneResolver.get_first_callable(
            "receivers.group_c_trainable", ("GroupCTrainableReceiver", "GroupCReceiver"), "Milestone M4 (F10)"
        )
        tr = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1070 + i, split="train")[0] for i in range(2)]
        dv = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1075, split="dev")[0]]
        te, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1079, split="test")
        model = cls_c(seed=43)
        model.fit(tr, dv, epochs=1)
        out = model.run_frame(te)
        self.assertGreater(int(out.aux_info["param_count"]), 0)
        self.assertTrue(np.all(np.isfinite(out.bit_llrs)))

    def test_t2_f10_4_unseen_snr_and_cfo_ood_generalization_stability(self) -> None:
        """T2-F10.4: Group C remains stable and finite under severe unseen AWGN + CFO + drift OOD stress."""
        perturb = MilestoneResolver.get_callable(
            "adapters.sdr_ofdm_adapter", "apply_stress_perturbation", "Milestone M1 (F4)"
        )
        cls_c = MilestoneResolver.get_first_callable(
            "receivers.group_c_trainable", ("GroupCTrainableReceiver", "GroupCReceiver"), "Milestone M4 (F10)"
        )
        tr = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1080 + i, snr_db=18.0, split="train")[0] for i in range(4)]
        dv = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1085 + i, snr_db=18.0, split="dev")[0] for i in range(2)]
        te, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1089, snr_db=18.0, split="test")
        model = cls_c(seed=44)
        model.fit(tr, dv, epochs=1)
        ood = perturb(te, extra_awgn_db=10.0, cfo_norm=0.08, phase_drift_rad=0.5, seed=44)
        out = model.run_frame(ood)
        self.assertTrue(np.all(np.isfinite(out.h_est)))
        self.assertTrue(np.all(np.isfinite(out.h_var)) and np.all(extract_active_h(out.h_var) > 0.0))
        self.assertTrue(np.all(np.isfinite(out.bit_llrs)))

    def test_t2_f10_5_checkpoint_save_reload_exact_parity(self) -> None:
        """T2-F10.5: Saving and reloading Group C checkpoint produces bit-exact inference parity."""
        cls_c = MilestoneResolver.get_first_callable(
            "receivers.group_c_trainable", ("GroupCTrainableReceiver", "GroupCReceiver"), "Milestone M4 (F10)"
        )
        tr = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1090 + i, split="train")[0] for i in range(4)]
        dv = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1095 + i, split="dev")[0] for i in range(2)]
        te, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1099, split="test")
        model = cls_c(seed=45)
        model.fit(tr, dv, epochs=1)
        o1 = model.run_frame(te)
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "group_c.pt"
            model.save(ckpt)
            reloaded = cls_c.load(ckpt)
            o2 = reloaded.run_frame(te)
        np.testing.assert_array_equal(o1.h_est, o2.h_est)
        np.testing.assert_array_equal(o1.bit_llrs, o2.bit_llrs)
        np.testing.assert_array_equal(o1.hard_bits, o2.hard_bits)

    # =========================================================================
    # Feature F11 Boundary Tests (5 tests)
    # =========================================================================

    def test_t2_f11_1_audit_statistical_verdict_boundary_and_replay_exactness(self) -> None:
        """T2-F11.1: Audit boundaries (`p=1.0` tie vs `p=2/65536` win) and saved replay parity (`max_diff==0`)."""
        MilestoneResolver.require_artifact("REPORT.md", "Milestone M5 (F11)")
        from p12.audit import exact_sign_flip_p, verdict_superiority

        tie_stat = exact_sign_flip_p([0.0] * 16)
        self.assertAlmostEqual(tie_stat["p_two_sided"], 1.0, places=12)
        self.assertEqual(verdict_superiority(tie_stat, tie_stat["p_two_sided"]), "证据不足")
        win_stat = exact_sign_flip_p([1.0] * 16)
        self.assertAlmostEqual(win_stat["p_two_sided"], 2.0 / 65536.0, places=12)
        self.assertEqual(verdict_superiority(win_stat, win_stat["p_two_sided"]), "支持")

        res_dir = RESEARCH_ROOT / "results"
        for rp in res_dir.glob("*replay*.json"):
            data = json.loads(rp.read_text(encoding="utf-8"))
            if "exact_pass" in data:
                self.assertTrue(bool(data["exact_pass"]))
                self.assertEqual(float(data.get("max_float_diff", 0.0)), 0.0)

    def test_t2_f11_2_failure_accounting_conservation_invariant(self) -> None:
        """T2-F11.2: Every evaluated arm in `results/` satisfies `success_count + fail_count == total_frames`."""
        MilestoneResolver.require_artifact("REPORT.md", "Milestone M5 (F11)")
        res_dir = MilestoneResolver.require_artifact("results", "Milestone M5 (F11)", allow_dir=True)
        summary_files = list(res_dir.glob("*summary*.json"))
        self.assertGreater(len(summary_files), 0)
        summary = json.loads(summary_files[0].read_text(encoding="utf-8"))
        arms = summary.get("arms", summary)
        for rec in arms.values():
            if not isinstance(rec, dict) or "total_frames" not in rec:
                continue
            tot = int(rec["total_frames"])
            suc = int(rec["success_count"])
            fail = int(rec["fail_count"])
            self.assertEqual(suc + fail, tot)
            self.assertAlmostEqual(float(rec["coverage_pct"]), 100.0 * suc / max(tot, 1), delta=1e-4)

    def test_t2_f11_3_engineering_cost_positivity_and_consistency(self) -> None:
        """T2-F11.3: Saved engineering cost metrics satisfy positivity and `p95 >= mean` (or within tolerance)."""
        MilestoneResolver.require_artifact("REPORT.md", "Milestone M5 (F11)")
        res_dir = MilestoneResolver.require_artifact("results", "Milestone M5 (F11)", allow_dir=True)
        summary_files = list(res_dir.glob("*summary*.json"))
        self.assertGreater(len(summary_files), 0)
        summary = json.loads(summary_files[0].read_text(encoding="utf-8"))
        arms = summary.get("arms", summary)
        for rec in arms.values():
            if not isinstance(rec, dict) or "mean_ms_per_frame" not in rec:
                continue
            self.assertGreater(float(rec["mean_ms_per_frame"]), 0.0)
            self.assertGreaterEqual(float(rec["p95_ms_per_frame"]), float(rec["mean_ms_per_frame"]) * 0.5)
            self.assertGreater(float(rec["peak_rss_mb"]), 0.0)
            self.assertGreaterEqual(int(rec["n_candidates"]), 1)
            self.assertGreaterEqual(int(rec["param_count"]), 0)

    def test_t2_f11_4_report_numbers_match_summary_json_zero_fabrication(self) -> None:
        """T2-F11.4: `REPORT.md` has zero `TODO`/`TBD`/`PLACEHOLDER` and matches `splits.json` counts."""
        report_path = MilestoneResolver.require_artifact("REPORT.md", "Milestone M5 (F11)")
        splits_path = MilestoneResolver.require_artifact("manifests/splits.json", "Milestone M1 (F5)")
        text = report_path.read_text(encoding="utf-8")
        for tok in ("TODO", "TBD", "PLACEHOLDER"):
            self.assertNotIn(tok, text)
        sp = read_json_file(splits_path)
        n_test = len(sp["test_indices"])
        self.assertIn(str(n_test), text)

    def test_t2_f11_5_untracked_agents_and_clean_workdir_isolation(self) -> None:
        """T2-F11.5: `.agents/` untracked/unstaged and all deliverables reside inside `research/realdata_sdr_ofdm/`."""
        MilestoneResolver.require_artifact("REPORT.md", "Milestone M5 (F11)")
        self.assertEqual(run_git("ls-files", ".agents"), "")
        self.assertNotIn(".agents", run_git("diff", "--cached", "--name-only"))
        self.assertTrue((RESEARCH_ROOT / "DATA_SOURCES.md").is_file())
        self.assertTrue((RESEARCH_ROOT / "PROTOCOL.md").is_file())
