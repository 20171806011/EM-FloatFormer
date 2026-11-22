"""
EM-FloatFormer Real-Data SDR OFDM E2E Suite — Tier 1: Feature Coverage (55 Tests).

Covers all 11 features (F1–F11, exactly 5 isolated tests per feature):
- F1: Git Sync & Branching (M1)
- F2: Data Source Catalog (M1)
- F3: Remote Acquisition & 16–64 Frame Probe (M1)
- F4: Unified Data Adapter & Non-Gray 16-QAM Modulation (M1)
- F5: Leak-Free Partition & Protocol (M1)
- F6: Classical Receiver Baseline (M2)
- F7: Unified Evaluator & Stress Perturbations (M2)
- F8: Group A — Global Reference & Content Matching (M3)
- F9: Group B — Hard Selection vs Posterior Fusion (M3)
- F10: Group C — Lightweight Trainable / Attention Module (M4)
- F11: Full Codespace Run, Figures, Report & Delivery (M5)
"""

from __future__ import annotations

import hashlib
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
    DC_FFT_BIN,
    E2E_SEED_BASE,
    EXPECTED_BACKUP_SHA256,
    EXPECTED_MAIN_SHA,
    EXPECTED_P12_BASE_SHA,
    EXPECTED_WORK_BRANCH,
    FFT_OFFSET,
    N_ACTIVE,
    N_DATA_RES,
    N_FFT,
    N_GUARD_RES,
    N_PILOTS,
    N_SYMB,
    PILOT_FFT_BINS,
    PILOT_SYMBOL_IDX,
    PILOT_VALUES_REF,
    REPO_ROOT,
    RESEARCH_ROOT,
    MilestoneResolver,
    build_sdr_ofdm_masks,
    compute_file_sha256,
    ece_from_llrs,
    exact_sign_flip_p_ref,
    extract_active_h,
    holm_adjust_ref,
    make_standalone_frame_proxy,
    make_synthetic_custom_dataset_pth,
    make_synthetic_grid_and_truth,
    make_synthetic_real_ofdm_frame,
    make_synthetic_receiver_output,
    nongray_16qam_exact_llr_ref,
    nongray_16qam_hard_demod_ref,
    nongray_16qam_mod_ref,
    nongray_16qam_table_ref,
    read_json_file,
    run_git,
    run_python_script,
    scale_real_ofdm_frame,
)


class TestTier1FeatureCoverage(unittest.TestCase):
    """Tier 1: Feature Coverage (55 tests — 5 per feature F1–F11)."""

    def setUp(self) -> None:
        try:
            torch.set_num_threads(1)
        except RuntimeError:
            pass

    # =========================================================================
    # Feature F1: Git Sync & Branching (5 tests)
    # =========================================================================

    def test_t1_f01_working_branch_and_p12_base_ancestor(self) -> None:
        """T1-F1.1: Active branch is `research/realdata-sdr-ofdm` descended from `f566f9b8`."""
        branch = run_git("branch", "--show-current")
        self.assertEqual(branch, EXPECTED_WORK_BRANCH)
        proc = subprocess.run(
            ["git", "merge-base", "--is-ancestor", EXPECTED_P12_BASE_SHA, "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, "HEAD must contain P12 base commit f566f9b8")

    def test_t1_f01_main_branch_immutable_at_63ad2ec(self) -> None:
        """T1-F1.2: `main` branch remains untouched at `63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1`."""
        main_sha = run_git("rev-parse", "main")
        self.assertEqual(main_sha, EXPECTED_MAIN_SHA)
        diff_proc = subprocess.run(
            ["git", "diff", "--quiet", "main", EXPECTED_MAIN_SHA],
            cwd=str(REPO_ROOT),
            check=False,
        )
        self.assertEqual(diff_proc.returncode, 0)

    def test_t1_f01_p12_and_research_upstream_artifacts_present(self) -> None:
        """T1-F1.3: Required upstream P12 and research reports from `f566f9b8` are present."""
        required = [
            REPO_ROOT / "p12" / "P12_CONSISTENCY_FIX.md",
            REPO_ROOT / "p12" / "P12_REPORT.md",
            REPO_ROOT / "p12" / "audit.py",
            REPO_ROOT / "research" / "content_addressing_scalar" / "REPORT.md",
            REPO_ROOT / "research" / "posterior_fusion" / "README.md",
        ]
        for path in required:
            self.assertTrue(path.is_file(), f"Missing upstream artifact: {path}")
            self.assertGreater(path.stat().st_size, 0, f"Empty upstream artifact: {path}")

    def test_t1_f01_agents_and_raw_binaries_untracked(self) -> None:
        """T1-F1.4: `.agents/` and raw dataset archives are never tracked by git."""
        tracked_agents = run_git("ls-files", ".agents")
        self.assertEqual(tracked_agents, "")
        tracked_research = run_git("ls-files", "research/realdata_sdr_ofdm")
        forbidden_ext = (".pth", ".pt", ".ckpt", ".mat", ".tar.gz", ".zip", ".dat")
        for line in tracked_research.splitlines():
            low = line.strip().lower()
            self.assertFalse(low.endswith(forbidden_ext), f"Forbidden tracked binary: {line}")

    def test_t1_f01_repo_isolation_boundary(self) -> None:
        """T1-F1.5: All committed and working-tree changes since `f566f9b8` stay in `research/realdata_sdr_ofdm/`."""
        committed = run_git("diff", "--name-only", f"{EXPECTED_P12_BASE_SHA}..HEAD")
        for line in committed.splitlines():
            p = line.strip()
            if p:
                self.assertTrue(
                    p.startswith("research/realdata_sdr_ofdm/"),
                    f"Committed change outside isolation boundary: {p}",
                )
        status_out = run_git("status", "--porcelain")
        for line in status_out.splitlines():
            if not line.strip():
                continue
            rel = line[3:].strip().strip('"')
            if rel.startswith(".agents"):
                continue
            self.assertTrue(
                rel.startswith("research/realdata_sdr_ofdm/"),
                f"Uncommitted change outside isolation boundary: {rel}",
            )

    # =========================================================================
    # Feature F2: Data Source Catalog (5 tests)
    # =========================================================================

    def test_t1_f02_primary_and_backup_present(self) -> None:
        """T1-F2.1: `DATA_SOURCES.md` documents primary `sdr_ofdm_dataset_v1` and backup `ADALM-PLUTO QPSK`."""
        path = MilestoneResolver.require_artifact("DATA_SOURCES.md", "Milestone M1 (F2)")
        text = path.read_text(encoding="utf-8")
        self.assertGreater(len(text.encode("utf-8")), 500)
        self.assertIn("rikluost/sdr_ofdm_dataset_v1", text)
        self.assertIn("ofdm_valset_4_8_sdr.pth", text)
        self.assertIn("10.17632/k72b4pnc5g.1", text)
        self.assertIn("QPSKlink.mat", text)
        for token in ("TODO", "TBD", "<placeholder>"):
            self.assertNotIn(token, text)

    def test_t1_f02_r2_provenance_fields_complete(self) -> None:
        """T1-F2.2: `DATA_SOURCES.md` includes all R2 provenance metadata fields."""
        path = MilestoneResolver.require_artifact("DATA_SOURCES.md", "Milestone M1 (F2)")
        text = path.read_text(encoding="utf-8")
        low = text.lower()
        for kw in ("https://", ".pth", ".mat", "sha-256", "doi"):
            self.assertIn(kw, low, f"Missing R2 provenance token '{kw}' in DATA_SOURCES.md")
        self.assertTrue("27,684,430" in text or "27684430" in text)
        self.assertTrue("822,762,822" in text or "822762822" in text)

    def test_t1_f02_sdr_ofdm_waveform_pilot_specs(self) -> None:
        """T1-F2.3: `DATA_SOURCES.md` records exact Primary grid and non-Gray 16-QAM parameters."""
        path = MilestoneResolver.require_artifact("DATA_SOURCES.md", "Milestone M1 (F2)")
        text = path.read_text(encoding="utf-8")
        for token in ("128", "14", "102", "64", "1400", "16-QAM", "mapping_table(4)"):
            self.assertIn(token, text, f"Missing waveform spec '{token}' in DATA_SOURCES.md")

    def test_t1_f02_processing_stage_and_repetitions(self) -> None:
        """T1-F2.4: `DATA_SOURCES.md` documents processing stage, session/device, and backup payload repetitions."""
        path = MilestoneResolver.require_artifact("DATA_SOURCES.md", "Milestone M1 (F2)")
        low = path.read_text(encoding="utf-8").lower()
        self.assertTrue("post-dft" in low or "post-fft" in low or "frequency-domain" in low)
        self.assertTrue("repetit" in low or "duplicate" in low)
        self.assertIn("pluto", low)

    def test_t1_f02_checksums_and_storage_separation(self) -> None:
        """T1-F2.5: `DATA_SOURCES.md` specifies raw vs processed separation and checksum references."""
        path = MilestoneResolver.require_artifact("DATA_SOURCES.md", "Milestone M1 (F2)")
        text = path.read_text(encoding="utf-8")
        self.assertIn("raw_checksums.json", text)
        self.assertIn("/tmp/tw_realdata/data_raw", text)
        self.assertIn("/tmp/tw_realdata/data_proc", text)

    # =========================================================================
    # Feature F3: Remote Acquisition & 16–64 Frame Probe (5 tests)
    # =========================================================================

    def test_t1_f03_raw_checksums_manifest_schema(self) -> None:
        """T1-F3.1: `manifests/raw_checksums.json` has valid schema, SHA-256, and optional live match."""
        manifest_path = MilestoneResolver.require_artifact("manifests/raw_checksums.json", "Milestone M1 (F3)")
        data = read_json_file(manifest_path)
        primary = data["primary"]
        self.assertIn("ofdm_valset_4_8_sdr.pth", primary.get("url", ""))
        self.assertGreater(int(primary["size_bytes"]), 0)
        self.assertRegex(primary["sha256"], r"^[0-9a-fA-F]{64}$")
        self.assertNotIn("0" * 64, primary["sha256"])

        live_raw = Path("/tmp/tw_realdata/data_raw/ofdm_valset_4_8_sdr.pth")
        if live_raw.is_file():
            self.assertEqual(live_raw.stat().st_size, int(primary["size_bytes"]))
            self.assertEqual(compute_file_sha256(live_raw), primary["sha256"])

    def test_t1_f03_sha256_verifier_offline_roundtrip(self) -> None:
        """T1-F3.2: Acquisition CLI checksum verifier matches `hashlib` SHA-256 and git-blob SHA-1 offline."""
        dl_script = RESEARCH_ROOT / "scripts" / "download_and_verify.py"
        if not dl_script.is_file():
            raise unittest.SkipTest("[PROGRESSIVE EXECUTION] scripts/download_and_verify.py missing (M1 F3).")
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            raw_dir = td_path / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            pth_path = raw_dir / "ofdm_valset_4_8_sdr.pth"
            payload = b"EM-FloatFormer-SDR-OFDM-checksum-test-\x00\xff\x13\x64"
            pth_path.write_bytes(payload)
            exp_sha256 = hashlib.sha256(payload).hexdigest()
            exp_blob_sha1 = hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()

            cfg = yaml.safe_load((RESEARCH_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
            cfg["dataset"]["raw_dir"] = str(raw_dir)
            cfg["dataset"]["expected_size_bytes"] = len(payload)
            cfg["dataset"]["expected_blob_sha1"] = exp_blob_sha1
            cfg["dataset"]["expected_sha256"] = exp_sha256
            cfg_path = td_path / "cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            manifest_out = td_path / "manifest.json"
            probe_out = td_path / "probe.json"

            run_python_script(
                dl_script,
                "--skip-download",
                "--config", str(cfg_path),
                "--raw-dir", str(raw_dir),
                "--manifest-out", str(manifest_out),
                "--probe-out", str(probe_out),
                "--probe-n", "16",
            )
            self.assertTrue(manifest_out.is_file(), "download_and_verify.py must write manifest JSON")
            manifest = read_json_file(manifest_out)
            self.assertEqual(int(manifest["primary"]["size_bytes"]), len(payload))
            self.assertEqual(manifest["primary"]["sha256"], exp_sha256)
            self.assertEqual(manifest["primary"]["git_blob_sha1"], exp_blob_sha1)
            probe = read_json_file(probe_out)
            self.assertEqual([r for r in probe["fallback_reasons"] if r.startswith("T1:")], [])

    def test_t1_f03_probe_32_frame_fixture_summary(self) -> None:
        """T1-F3.3: 32-frame probe pipeline computes valid SINR stats and non-Gray BER < 0.15 on clean SNR."""
        dl_script = RESEARCH_ROOT / "scripts" / "download_and_verify.py"
        if not dl_script.is_file():
            raise unittest.SkipTest("[PROGRESSIVE EXECUTION] scripts/download_and_verify.py missing (M1 F3).")
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            raw_dir = td_path / "raw"
            pth_path = raw_dir / "ofdm_valset_4_8_sdr.pth"
            info = make_synthetic_custom_dataset_pth(
                pth_path, n_frames=200, seed=E2E_SEED_BASE + 301, snr_min_db=10.0, snr_max_db=22.0
            )
            cfg = yaml.safe_load((RESEARCH_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
            cfg["dataset"]["raw_dir"] = str(raw_dir)
            cfg["dataset"]["expected_size_bytes"] = info["size_bytes"]
            cfg["dataset"]["expected_blob_sha1"] = info["git_blob_sha1"]
            cfg_path = td_path / "cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            manifest_out = td_path / "manifest.json"
            probe_out = td_path / "probe.json"

            proc = run_python_script(
                dl_script,
                "--skip-download",
                "--config", str(cfg_path),
                "--raw-dir", str(raw_dir),
                "--manifest-out", str(manifest_out),
                "--probe-out", str(probe_out),
                "--probe-n", "32",
            )
            self.assertEqual(proc.returncode, 0, f"Probe failed: {proc.stderr}\n{proc.stdout}")
            probe = read_json_file(probe_out)
            self.assertFalse(probe["fallback_triggered"])
            self.assertEqual(len(probe["probe_indices"]), 32)
            self.assertEqual(probe["n_tti"], 200)
            s_stats = probe["sinr_stats"]
            self.assertLessEqual(s_stats["min"], s_stats["mean"])
            self.assertLessEqual(s_stats["mean"], s_stats["max"])
            self.assertLess(probe["ls_zf_probe"]["nongray_ber"], 0.15)
            self.assertLess(probe["ls_zf_probe"]["nongray_ber"], probe["ls_zf_probe"]["gray_ber"])

    def test_t1_f03_probe_pilot_correlation_and_dc_guard_check(self) -> None:
        """T1-F3.4: 16-frame probe confirms >0.70 pilot coherence and >=10x pilot/guard power ratio."""
        dl_script = RESEARCH_ROOT / "scripts" / "download_and_verify.py"
        if not dl_script.is_file():
            raise unittest.SkipTest("[PROGRESSIVE EXECUTION] scripts/download_and_verify.py missing (M1 F3).")
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            raw_dir = td_path / "raw"
            pth_path = raw_dir / "ofdm_valset_4_8_sdr.pth"
            info = make_synthetic_custom_dataset_pth(
                pth_path, n_frames=200, seed=E2E_SEED_BASE + 302, snr_min_db=16.0, snr_max_db=22.0
            )
            cfg = yaml.safe_load((RESEARCH_ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
            cfg["dataset"]["raw_dir"] = str(raw_dir)
            cfg["dataset"]["expected_size_bytes"] = info["size_bytes"]
            cfg["dataset"]["expected_blob_sha1"] = info["git_blob_sha1"]
            cfg_path = td_path / "cfg.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            probe_out = td_path / "probe.json"

            proc = run_python_script(
                dl_script,
                "--skip-download",
                "--config", str(cfg_path),
                "--raw-dir", str(raw_dir),
                "--manifest-out", str(td_path / "manifest.json"),
                "--probe-out", str(probe_out),
                "--probe-n", "16",
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            probe = read_json_file(probe_out)
            self.assertGreater(probe["pilot_coherence"]["mean_pilot_coherence"], 0.70)
            self.assertGreaterEqual(probe["power_profile"]["pilot_guard_power_ratio"], 10.0)

    def test_t1_f03_acquisition_and_smoke_scripts_cli_paths(self) -> None:
        """T1-F3.5: Acquisition & smoke scripts expose `--help` and avoid forbidden paths."""
        dl_script = RESEARCH_ROOT / "scripts" / "download_and_verify.py"
        self.assertTrue(dl_script.is_file())
        proc_dl = subprocess.run(
            [sys.executable, "-B", str(dl_script), "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc_dl.returncode, 0)
        self.assertNotIn("/tmp/agy_cli", dl_script.read_text(encoding="utf-8"))
        self.assertNotIn("/workspaces/EM-FloatFormer", dl_script.read_text(encoding="utf-8"))

        smoke_script = MilestoneResolver.require_artifact("scripts/run_smoke.py", "Milestone M1 (F3)")
        proc_sm = subprocess.run(
            [sys.executable, "-B", str(smoke_script), "--help"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc_sm.returncode, 0)
        smoke_txt = smoke_script.read_text(encoding="utf-8")
        self.assertNotIn("/tmp/agy_cli", smoke_txt)
        self.assertNotIn("/workspaces/EM-FloatFormer", smoke_txt)

    # =========================================================================
    # Feature F4: Unified Data Adapter & Non-Gray 16-QAM (5 tests)
    # =========================================================================

    def test_t1_f04_grid_masks_partition_and_pilot_constants(self) -> None:
        """T1-F4.1: `RealOFDMFrame` masks partition `14x128=1792` REs into 14+14+364+1400 cleanly."""
        frame, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 401)
        p_m, dc_m, d_m = frame.pilot_mask, frame.dc_mask, frame.data_mask
        self.assertEqual(p_m.shape, (N_SYMB, N_FFT))
        self.assertEqual(int(p_m.sum()), N_PILOTS)
        self.assertEqual(int(p_m[PILOT_SYMBOL_IDX].sum()), N_PILOTS)
        self.assertEqual(tuple(int(x) for x in np.flatnonzero(p_m[PILOT_SYMBOL_IDX])), PILOT_FFT_BINS)
        self.assertEqual(int(dc_m.sum()), N_SYMB)
        self.assertTrue(np.all(dc_m[:, DC_FFT_BIN]))
        self.assertEqual(int(d_m.sum()), N_DATA_RES)
        guard_m = ~(p_m | dc_m | d_m)
        self.assertEqual(int(guard_m.sum()), N_GUARD_RES)
        self.assertFalse(np.any(p_m & dc_m))
        self.assertFalse(np.any(p_m & d_m))
        self.assertFalse(np.any(dc_m & d_m))
        np.testing.assert_allclose(frame.pilot_values, PILOT_VALUES_REF, atol=1e-6)

    def test_t1_f04_non_gray_mapping_table_bijection_and_energy(self) -> None:
        """T1-F4.2: `mapping_table(4)` matches natural-binary order, unit mean energy, and 0-BER roundtrip."""
        mod_m = MilestoneResolver.get_module("adapters.modulation", "Milestone M1 (F4)")
        mapping_table = getattr(mod_m, "mapping_table")
        constellation_points = getattr(mod_m, "constellation_points")
        bits_to_symbols = getattr(mod_m, "bits_to_symbols")
        symbols_to_hard_bits = getattr(mod_m, "symbols_to_hard_bits")

        m_dict, dem_dict = mapping_table(4)
        self.assertEqual(len(m_dict), 16)
        self.assertEqual(len(dem_dict), 16)
        tbl = np.asarray(constellation_points(4), dtype=np.complex128).reshape(-1)
        self.assertEqual(tbl.shape, (16,))
        self.assertAlmostEqual(float(np.mean(np.abs(tbl) ** 2)), 1.0, places=6)
        self.assertEqual(len(np.unique(np.round(tbl, 8))), 16)
        np.testing.assert_allclose(tbl, nongray_16qam_table_ref(), atol=1e-6)
        # Explicit non-Gray binary ordering check
        self.assertAlmostEqual(m_dict[(0, 0, 0, 1)], (-3.0 - 1.0j) / math.sqrt(10.0), places=6)
        self.assertAlmostEqual(m_dict[(0, 0, 1, 0)], (-3.0 + 1.0j) / math.sqrt(10.0), places=6)
        self.assertAlmostEqual(m_dict[(0, 1, 0, 0)], (-1.0 - 3.0j) / math.sqrt(10.0), places=6)
        # Noiseless modulation -> demodulation roundtrip
        bits = np.array([[(i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(16)], dtype=np.uint8)
        syms = bits_to_symbols(bits, qm=4)
        np.testing.assert_allclose(syms, tbl, atol=1e-6)
        rec = symbols_to_hard_bits(syms, qm=4)
        np.testing.assert_array_equal(rec, bits)

    def test_t1_f04_non_gray_llr_sign_and_awgn_ece(self) -> None:
        """T1-F4.3: Non-Gray 16-QAM exact & maxlog LLRs recover all 16 points and achieve AWGN ECE < 0.03."""
        mod_m = MilestoneResolver.get_module("adapters.modulation", "Milestone M1 (F4)")
        llr_fn = getattr(mod_m, "exact_llr")
        maxlog_fn = getattr(mod_m, "maxlog_llr")
        tbl = nongray_16qam_table_ref()
        bits_tbl = np.array([[(i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(16)], dtype=np.uint8)
        llrs_clean = np.asarray(llr_fn(tbl, 0.01), dtype=np.float64)
        np.testing.assert_array_equal((llrs_clean > 0.0).astype(np.uint8), bits_tbl)
        np.testing.assert_allclose(llrs_clean, nongray_16qam_exact_llr_ref(tbl, 0.01), atol=1e-4)

        rng = np.random.default_rng(E2E_SEED_BASE + 403)
        bits = rng.integers(0, 2, size=(4096, 4), dtype=np.uint8)
        x = nongray_16qam_mod_ref(bits)
        sigma2 = 0.10
        z = x + math.sqrt(sigma2 / 2.0) * (rng.standard_normal(4096) + 1j * rng.standard_normal(4096))
        llrs_awgn = np.asarray(llr_fn(z, sigma2), dtype=np.float64)
        llrs_max = np.asarray(maxlog_fn(z, sigma2), dtype=np.float64)
        self.assertLess(ece_from_llrs(llrs_awgn, bits), 0.03)
        self.assertGreaterEqual(float(np.mean((llrs_awgn > 0.0) == (llrs_max > 0.0))), 0.99)

    def test_t1_f04_c_star_gauge_and_blind_est_n0(self) -> None:
        """T1-F4.4: Adapter `s0` & blind `est_n0` obey exact $C^*$ scaling with zero label leakage."""
        ad_m = MilestoneResolver.get_module("adapters.sdr_ofdm_adapter", "Milestone M1 (F4)")
        build_frame = getattr(ad_m, "build_frame")
        est_n0_fn = getattr(ad_m, "estimate_blind_n0")

        sim = make_synthetic_grid_and_truth(E2E_SEED_BASE + 404, snr_db=15.0)
        fr_base = build_frame(0, sim["rx_grid"], sim["tx_bits"], sim["reported_sinr_db"], split="dev", group_id="g0")
        s0_base = float(fr_base.meta["s0"])
        n0_raw_base, _ = est_n0_fn(sim["rx_grid"])
        self.assertGreater(s0_base, 0.0)
        self.assertGreater(float(fr_base.est_n0), 0.0)

        # Label poisoning must produce 0.0 change in s0, rx_grid, and est_n0
        fr_poison = build_frame(0, sim["rx_grid"], 1 - sim["tx_bits"], sim["reported_sinr_db"], split="dev", group_id="g0")
        self.assertEqual(float(fr_poison.meta["s0"]), s0_base)
        self.assertEqual(float(fr_poison.est_n0), float(fr_base.est_n0))
        np.testing.assert_array_equal(fr_poison.rx_grid, fr_base.rx_grid)

        for c in (2.0**-10, 0.25 * np.exp(0.7j), math.pi * np.exp(-1.0j * math.pi / 3.0), 2.0**10):
            fr_c = build_frame(0, c * sim["rx_grid"], sim["tx_bits"], sim["reported_sinr_db"])
            self.assertAlmostEqual(float(fr_c.meta["s0"]) / (abs(c) * s0_base), 1.0, delta=1e-5)
            np.testing.assert_allclose(fr_c.rx_grid, (c / abs(c)) * fr_base.rx_grid, atol=1e-6)
            n0_raw_c, _ = est_n0_fn(c * sim["rx_grid"])
            self.assertAlmostEqual(float(n0_raw_c) / ((abs(c) ** 2) * float(n0_raw_base)), 1.0, delta=1e-4)

        # Accuracy across SNR={5, 15, 25} dB
        for snr_db in (5.0, 15.0, 25.0):
            ratios = []
            for k in range(16):
                s_k = make_synthetic_grid_and_truth(E2E_SEED_BASE + 450 + int(snr_db) * 32 + k, snr_db=snr_db)
                est_k, _ = est_n0_fn(s_k["rx_grid"])
                ratios.append(float(est_k) / s_k["n0_true"])
            mean_ratio = float(np.mean(ratios))
            self.assertGreaterEqual(mean_ratio, 0.7)
            self.assertLessEqual(mean_ratio, 1.4)

    def test_t1_f04_stress_perturbation_physics_and_immutability(self) -> None:
        """T1-F4.5: `apply_stress_perturbation` conserves Parseval energy, avoids mutation, and separates CFO vs drift."""
        perturb = MilestoneResolver.get_callable(
            "adapters.sdr_ofdm_adapter", "apply_stress_perturbation", "Milestone M1 (F4)"
        )
        frame, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 405, snr_db=22.0)
        rx_orig = frame.rx_grid.copy()
        bits_orig = frame.tx_bits.copy()
        syms_orig = frame.tx_symbols.copy()

        noop = perturb(frame, extra_awgn_db=None, cfo_norm=0.0, phase_drift_rad=0.0, seed=1)
        np.testing.assert_allclose(noop.rx_grid, rx_orig, atol=1e-5)

        cfo_a = perturb(frame, extra_awgn_db=None, cfo_norm=0.04, phase_drift_rad=0.0, seed=2)
        cfo_b = perturb(frame, extra_awgn_db=None, cfo_norm=0.04, phase_drift_rad=0.0, seed=2)
        np.testing.assert_array_equal(cfo_a.rx_grid, cfo_b.rx_grid)

        e_in = float(np.sum(np.abs(rx_orig) ** 2))
        e_cfo = float(np.sum(np.abs(cfo_a.rx_grid) ** 2))
        self.assertAlmostEqual(e_cfo / e_in, 1.0, delta=1e-4)

        drift = perturb(frame, extra_awgn_db=None, cfo_norm=0.0, phase_drift_rad=0.4, seed=3)
        e_drift = float(np.sum(np.abs(drift.rx_grid) ** 2))
        self.assertAlmostEqual(e_drift / e_in, 1.0, delta=1e-4)
        # Pure symbol-level phase drift preserves per-RE magnitudes, whereas CFO induces ICI
        self.assertLess(float(np.max(np.abs(np.abs(drift.rx_grid) - np.abs(rx_orig)))), 1e-4)
        self.assertGreater(float(np.mean(np.abs(np.abs(cfo_a.rx_grid) - np.abs(rx_orig)))), 1e-4)

        # Input frame untouched
        np.testing.assert_array_equal(frame.rx_grid, rx_orig)
        np.testing.assert_array_equal(frame.tx_bits, bits_orig)
        np.testing.assert_array_equal(frame.tx_symbols, syms_orig)

    # =========================================================================
    # Feature F5: Leak-Free Partition & Protocol (5 tests)
    # =========================================================================

    def test_t1_f05_splits_json_schema_and_coverage(self) -> None:
        """T1-F5.1: `manifests/splits.json` defines non-empty valid `train`, `dev`, and `test` partitions."""
        splits_path = MilestoneResolver.require_artifact("manifests/splits.json", "Milestone M1 (F5)")
        sp = read_json_file(splits_path)
        total = int(sp["counts"]["total"])
        sum_counts = 0
        for split_name in ("train", "dev", "test", "guard"):
            key = f"{split_name}_indices"
            self.assertIn(key, sp)
            ids = sp[key]
            self.assertGreater(len(ids), 0)
            self.assertTrue(all(isinstance(i, int) and i >= 0 for i in ids))
            self.assertEqual(len(set(ids)), len(ids))
            self.assertEqual(len(ids), int(sp["counts"][split_name]))
            sum_counts += len(ids)
        self.assertEqual(sum_counts, total)

    def test_t1_f05_strict_frame_and_group_disjointness(self) -> None:
        """T1-F5.2: `train`, `dev`, `test` share zero `frame_id`s and zero `group_id`s."""
        splits_path = MilestoneResolver.require_artifact("manifests/splits.json", "Milestone M1 (F5)")
        sp = read_json_file(splits_path)
        frame_sets = {k: set(sp[f"{k}_indices"]) for k in ("train", "dev", "test", "guard")}
        group_sets: Dict[str, set[str]] = {"train": set(), "dev": set(), "test": set()}
        for blk in sp["blocks"]:
            group_sets[blk["split"]].add(blk["group_id"])
        for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
            self.assertEqual(frame_sets[a] & frame_sets[b], set(), f"Frame overlap {a}/{b}")
            self.assertEqual(group_sets[a] & group_sets[b], set(), f"Group overlap {a}/{b}")
        self.assertEqual(frame_sets["train"] & frame_sets["guard"], set())
        self.assertEqual(frame_sets["test"] & frame_sets["guard"], set())

    def test_t1_f05_contiguous_temporal_block_partition(self) -> None:
        """T1-F5.3: Split blocks form contiguous non-interleaved intervals preventing temporal leakage."""
        splits_path = MilestoneResolver.require_artifact("manifests/splits.json", "Milestone M1 (F5)")
        sp = read_json_file(splits_path)
        id_to_split = {}
        for k in ("train", "dev", "test"):
            for fid in sp[f"{k}_indices"]:
                id_to_split[fid] = k
        ordered = [id_to_split[fid] for fid in sorted(id_to_split)]
        transitions = sum(1 for a, b in zip(ordered, ordered[1:]) if a != b)
        self.assertLess(transitions, max(len(ordered) // 8, 16))
        self.assertGreaterEqual(int(sp["diagnostics"]["min_cross_split_distance"]), int(sp["diagnostics"]["guard_width"]) + 1)

    def test_t1_f05_protocol_md_required_r4_sections(self) -> None:
        """T1-F5.4: `PROTOCOL.md` defines metrics, FER event (5600-bit TTI), seeds, failures, and pinned threads."""
        proto_path = MilestoneResolver.require_artifact("PROTOCOL.md", "Milestone M1 (F5)")
        text = proto_path.read_text(encoding="utf-8")
        low = text.lower()
        for tok in ("5600", "fer", "ber", "omp_num_threads", "openblas_num_threads", "seed", "leak"):
            self.assertIn(tok, low, f"Missing protocol spec '{tok}' in PROTOCOL.md")
        self.assertNotIn("TBD", text)
        self.assertNotIn("TODO", text)

    def test_t1_f05_zero_test_leakage_calibration_policy(self) -> None:
        """T1-F5.5: Calibration/normalization metadata excludes `"test"` split IDs completely."""
        splits_path = MilestoneResolver.require_artifact("manifests/splits.json", "Milestone M1 (F5)")
        proto_path = MilestoneResolver.require_artifact("PROTOCOL.md", "Milestone M1 (F5)")
        sp = read_json_file(splits_path)
        test_ids = set(sp["test_indices"])
        cal_ids = set(sp.get("calibration_frame_ids", sp["dev_indices"]))
        self.assertEqual(cal_ids & test_ids, set())
        self.assertIn("zero test leakage", proto_path.read_text(encoding="utf-8").lower())

    # =========================================================================
    # Feature F6: Classical Receiver Baseline (5 tests)
    # =========================================================================

    def test_t1_f06_receiver_output_contract_compliance(self) -> None:
        """T1-F6.1: `classical.run_frame` returns compliant `ReceiverOutput` shapes, dtypes, and bits."""
        run_frame = MilestoneResolver.get_callable("receivers.classical", "run_frame", "Milestone M2 (F6)")
        frame, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 601, snr_db=20.0)
        out = run_frame(frame)
        self.assertIn(out.h_est.shape, ((N_SYMB, N_ACTIVE), (N_SYMB, N_FFT)))
        self.assertEqual(out.h_var.shape, out.h_est.shape)
        self.assertEqual(extract_active_h(out.h_est).shape, (N_SYMB, N_ACTIVE))
        self.assertTrue(np.all(np.isfinite(out.h_est)))
        self.assertTrue(np.all(np.isfinite(out.h_var)) and np.all(extract_active_h(out.h_var) >= 0.0))
        self.assertEqual(out.eq_symbols.shape, (N_DATA_RES,))
        self.assertEqual(out.bit_llrs.shape, (N_DATA_RES, 4))
        self.assertTrue(np.all(np.isfinite(out.bit_llrs)))
        np.testing.assert_array_equal(out.hard_bits, (out.bit_llrs > 0.0).astype(np.uint8))
        np.testing.assert_allclose(out.hard_symbols, nongray_16qam_mod_ref(out.hard_bits), atol=1e-6)
        self.assertGreaterEqual(float(out.aux_info["latency_ms"]), 0.0)
        self.assertFalse(bool(out.aux_info["failed"]))
        self.assertIsNone(out.aux_info["error"])

    def test_t1_f06_high_snr_30db_objective_sanity(self) -> None:
        """T1-F6.2: At 30 dB SNR, classical achieves BER<1e-3/EVM<5% on flat-fading and bounded multipath error."""
        run_frame = MilestoneResolver.get_callable("receivers.classical", "run_frame", "Milestone M2 (F6)")

        # Regime A: Flat-fading with non-trivial phase & C* gauge (defeats h=1 no-op receivers; zero Rayleigh-null floor)
        bit_errs_fl, total_bits_fl, sym_errs_fl, total_syms_fl = 0, 0, 0, 0
        sq_err_fl, sq_sig_fl = 0.0, 0.0
        for i in range(16):
            frame, _ = make_synthetic_real_ofdm_frame(
                E2E_SEED_BASE + 620 + i,
                snr_db=30.0,
                flat_channel=True,
                phase_offset=0.4 + 0.15 * i,
                gauge_scale=2.5 * np.exp(0.7j),
            )
            out = run_frame(frame)
            bit_errs_fl += int(np.sum(out.hard_bits != frame.tx_bits))
            total_bits_fl += frame.tx_bits.size
            sym_errs_fl += int(np.sum(np.any(out.hard_bits != frame.tx_bits, axis=1)))
            total_syms_fl += N_DATA_RES
            sq_err_fl += float(np.sum(np.abs(out.eq_symbols - frame.tx_symbols) ** 2))
            sq_sig_fl += float(np.sum(np.abs(frame.tx_symbols) ** 2))
        self.assertLess(bit_errs_fl / total_bits_fl, 1e-3)
        self.assertLess(sym_errs_fl / total_syms_fl, 5e-3)
        self.assertLess(100.0 * math.sqrt(sq_err_fl / sq_sig_fl), 5.0)

        # Regime B: 4-tap Rayleigh multipath bounded above Perfect-CSI Oracle floor (defeats mean-pilot flat-only receivers)
        bit_errs_mp, total_bits_mp, sym_errs_mp, total_syms_mp = 0, 0, 0, 0
        sq_err_mp, sq_sig_mp = 0.0, 0.0
        for i in range(16):
            frame, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 620 + i, snr_db=30.0, n_taps=4)
            out = run_frame(frame)
            bit_errs_mp += int(np.sum(out.hard_bits != frame.tx_bits))
            total_bits_mp += frame.tx_bits.size
            sym_errs_mp += int(np.sum(np.any(out.hard_bits != frame.tx_bits, axis=1)))
            total_syms_mp += N_DATA_RES
            sq_err_mp += float(np.sum(np.abs(out.eq_symbols - frame.tx_symbols) ** 2))
            sq_sig_mp += float(np.sum(np.abs(frame.tx_symbols) ** 2))
        self.assertLess(bit_errs_mp / total_bits_mp, 0.01)
        self.assertLess(sym_errs_mp / total_syms_mp, 0.025)
        self.assertLess(100.0 * math.sqrt(sq_err_mp / sq_sig_mp), 15.0)

    def test_t1_f06_lmmse_superiority_over_ls_zf_multipath(self) -> None:
        """T1-F6.3: Regularized LMMSE achieves lower/equal symbol MSE and BER < 0.15 vs LS+ZF at 15 dB."""
        run_frame = MilestoneResolver.get_callable("receivers.classical", "run_frame", "Milestone M2 (F6)")
        mse_ls, mse_lmmse = 0.0, 0.0
        err_lmmse, tot_bits = 0, 0
        for i in range(24):
            frame, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 640 + i, snr_db=15.0, n_taps=5)
            out_ls = run_frame(frame, method="ls_zf")
            out_mmse = run_frame(frame, method="lmmse")
            mse_ls += float(np.mean(np.abs(out_ls.eq_symbols - frame.tx_symbols) ** 2))
            mse_lmmse += float(np.mean(np.abs(out_mmse.eq_symbols - frame.tx_symbols) ** 2))
            err_lmmse += int(np.sum(out_mmse.hard_bits != frame.tx_bits))
            tot_bits += frame.tx_bits.size
        self.assertLess(mse_lmmse, mse_ls * 1.05)
        self.assertLess(err_lmmse / tot_bits, 0.15)

    def test_t1_f06_c_star_equivariance_exact(self) -> None:
        """T1-F6.4: Classical receiver satisfies exact $C^*$ channel equivariance and LLR invariance."""
        run_frame = MilestoneResolver.get_callable("receivers.classical", "run_frame", "Milestone M2 (F6)")
        frame, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 604, snr_db=18.0)
        c = 12.5 * np.exp(1.23j)
        scaled = scale_real_ofdm_frame(frame, c)
        out1 = run_frame(frame)
        out2 = run_frame(scaled)
        np.testing.assert_allclose(out2.h_est, c * out1.h_est, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(out2.eq_symbols, out1.eq_symbols, atol=1e-4)
        np.testing.assert_allclose(out2.bit_llrs, out1.bit_llrs, atol=1e-4)
        self.assertEqual(int(np.sum(out2.hard_bits != out1.hard_bits)), 0)

    def test_t1_f06_heldout_pilot_and_zero_label_leakage(self) -> None:
        """T1-F6.5: Held-out pilot predictions are finite and poisoning `tx_bits`/`tx_symbols` changes 0 outputs."""
        run_frame = MilestoneResolver.get_callable("receivers.classical", "run_frame", "Milestone M2 (F6)")
        frame, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 605, snr_db=18.0)
        poisoned = scale_real_ofdm_frame(frame, 1.0)
        poisoned.tx_bits[:] = 1 - poisoned.tx_bits
        poisoned.tx_symbols[:] = -poisoned.tx_symbols

        out1 = run_frame(frame)
        out2 = run_frame(poisoned)
        np.testing.assert_array_equal(out1.h_est, out2.h_est)
        np.testing.assert_array_equal(out1.bit_llrs, out2.bit_llrs)
        np.testing.assert_array_equal(out1.hard_bits, out2.hard_bits)
        self.assertIsNotNone(out1.heldout_pilot_pred)
        self.assertIsNotNone(out1.heldout_pilot_var)
        self.assertTrue(np.all(np.isfinite(out1.heldout_pilot_pred)))
        self.assertTrue(np.all(np.asarray(out1.heldout_pilot_var) > 0.0))

    # =========================================================================
    # Feature F7: Unified Evaluator & Stress Perturbations (5 tests)
    # =========================================================================

    def test_t1_f07_heldout_pilot_nmse_cal_nll_exactness(self) -> None:
        """T1-F7.1: Held-out pilot NMSE, calibration ratio, and NLL match PROTOCOL.md §5 observable formulas."""
        eval_fn = MilestoneResolver.get_first_callable(
            "eval.evaluator", ("evaluate_receiver", "evaluate_frames", "evaluate_outputs"), "Milestone M2 (F7)"
        )
        rng = np.random.default_rng(E2E_SEED_BASE + 701)
        heldout_idx = [1, 5, 9, 12]
        heldout_fft = [PILOT_FFT_BINS[k] for k in heldout_idx]

        def _run_case(snr_db: float, noise_scale: float) -> tuple[dict[str, Any], float, float, float, float, float]:
            pairs = []
            errs_obs, z_all, vars_eff, vars_obs = [], [], [], []
            for i in range(6):
                fr, _ = make_standalone_frame_proxy(E2E_SEED_BASE + 710 + i, snr_db=snr_db)
                y_p = fr.rx_grid[PILOT_SYMBOL_IDX, heldout_fft]
                p_val = fr.pilot_values[heldout_idx]
                z_obs = y_p / p_val
                sigma2_obs = float(fr.est_n0) / (np.abs(p_val) ** 2)
                p_var = np.full(z_obs.shape, 0.04, dtype=np.float64)
                pred_err = noise_scale * (rng.standard_normal(z_obs.shape) + 1j * rng.standard_normal(z_obs.shape))
                p_pred = z_obs + pred_err
                out = make_synthetic_receiver_output(fr, pilot_pred=p_pred, pilot_var=p_var)
                pairs.append((fr, out))
                errs_obs.append(p_pred - z_obs)
                z_all.append(z_obs)
                vars_eff.append(p_var + sigma2_obs)
                vars_obs.append(sigma2_obs)

            res = eval_fn(pairs)
            e_cat = np.concatenate(errs_obs)
            z_cat = np.concatenate(z_all)
            v_eff = np.concatenate(vars_eff)
            v_obs = np.concatenate(vars_obs)

            num_raw = float(np.sum(np.abs(e_cat) ** 2))
            den_raw = float(np.sum(np.abs(z_cat) ** 2))
            sum_sig = float(np.sum(v_obs))
            exp_nmse_raw = num_raw / den_raw
            exp_nmse_corr = max(0.0, num_raw - sum_sig) / max(den_raw - sum_sig, 1e-12)
            exp_nmse_corr_rawden = max(0.0, num_raw - sum_sig) / den_raw
            exp_cal = float(np.mean((np.abs(e_cat) ** 2) / v_eff))
            exp_nll = float(np.mean(np.log(math.pi * v_eff) + (np.abs(e_cat) ** 2) / v_eff))
            return res, exp_nmse_raw, exp_nmse_corr, exp_nmse_corr_rawden, exp_cal, exp_nll

        # Case A: Near-noiseless grid (est_n0 = 1e-14); raw and noise-corrected pilot_nmse coincide (< 1e-12)
        res_clean, nmse_c_raw, _, _, cal_c, nll_c = _run_case(snr_db=140.0, noise_scale=0.10)
        self.assertAlmostEqual(float(res_clean["pilot_nmse"]), nmse_c_raw, delta=1e-5)
        self.assertAlmostEqual(float(res_clean["pilot_cal_ratio"]), cal_c, delta=1e-5)
        self.assertAlmostEqual(float(res_clean["pilot_nll"]), nll_c, delta=1e-4)

        # Case B: Noisy 18 dB grid; sigma2_obs = est_n0 / |p|^2 (~0.01617) materially affects cal & NLL
        res_noisy, nmse_n_raw, nmse_n_corr, nmse_n_corr_rd, cal_n, nll_n = _run_case(snr_db=18.0, noise_scale=0.25)
        self.assertAlmostEqual(float(res_noisy["pilot_cal_ratio"]), cal_n, delta=1e-5)
        self.assertAlmostEqual(float(res_noisy["pilot_nll"]), nll_n, delta=1e-4)
        got_nmse = float(res_noisy["pilot_nmse"])
        self.assertLess(
            min(abs(got_nmse - nmse_n_raw), abs(got_nmse - nmse_n_corr), abs(got_nmse - nmse_n_corr_rd)),
            1e-5,
            f"pilot_nmse={got_nmse} matched neither raw ({nmse_n_raw}) nor noise-corrected ({nmse_n_corr}) PROTOCOL.md §5 formulas",
        )

    def test_t1_f07_payload_exact_integer_counts_and_evm(self) -> None:
        """T1-F7.2: Payload evaluator computes exact integer bit/frame error counts and analytical EVM%."""
        eval_fn = MilestoneResolver.get_first_callable(
            "eval.evaluator", ("evaluate_receiver", "evaluate_frames", "evaluate_outputs"), "Milestone M2 (F7)"
        )
        pairs = []
        flips_per_frame = [(), ((0, 0), (1, 2)), ((5, 1),), (), ((10, 3), (11, 0), (12, 1)), (), (), ((9, 2),)]
        sq_err, sq_sig = 0.0, 0.0
        for i, flips in enumerate(flips_per_frame):
            fr, _ = make_standalone_frame_proxy(E2E_SEED_BASE + 720 + i)
            eq = fr.tx_symbols + (0.05 + 0.03j)
            sq_err += float(np.sum(np.abs(eq - fr.tx_symbols) ** 2))
            sq_sig += float(np.sum(np.abs(fr.tx_symbols) ** 2))
            out = make_synthetic_receiver_output(fr, eq_symbols=eq, bit_flip_indices=flips)
            pairs.append((fr, out))

        res = eval_fn(pairs)
        self.assertIsInstance(res["bit_errors"], int)
        self.assertIsInstance(res["frame_errors"], int)
        self.assertEqual(res["bit_errors"], 7)
        self.assertEqual(res["total_bits"], 8 * BITS_PER_TTI)
        self.assertAlmostEqual(float(res["ber"]), 7.0 / (8 * BITS_PER_TTI), places=12)
        self.assertEqual(res["frame_errors"], 4)
        self.assertEqual(res["total_frames"], 8)
        self.assertAlmostEqual(float(res["fer"]), 4.0 / 8.0, places=12)
        self.assertAlmostEqual(float(res["evm_pct"]), 100.0 * math.sqrt(sq_err / sq_sig), delta=1e-4)

    def test_t1_f07_paired_g16_p12_audit_integration(self) -> None:
        """T1-F7.3: Paired comparison on `G=16` groups matches `p12.audit.exact_sign_flip_p` and `holm_adjust`."""
        compare_fn = MilestoneResolver.get_first_callable(
            "eval.evaluator", ("compare_arms", "paired_comparison", "evaluate_paired"), "Milestone M2 (F7)"
        )
        from p12.audit import exact_sign_flip_p, holm_adjust, verdict_superiority

        base_pairs, cand_pairs = [], []
        for g in range(16):
            fr, _ = make_standalone_frame_proxy(E2E_SEED_BASE + 740 + g, group_id=f"blk_{g:02d}", frame_id=g)
            base_pairs.append((fr, make_synthetic_receiver_output(fr, bit_flip_indices=[(i, 0) for i in range(10)])))
            cand_pairs.append((fr, make_synthetic_receiver_output(fr, bit_flip_indices=[(0, 0)])))

        comp = compare_fn(base_pairs, cand_pairs)
        ref_stat = exact_sign_flip_p([100.0 * 9.0 / BITS_PER_TTI] * 16)
        self.assertAlmostEqual(float(comp["p_two_sided"]), ref_stat["p_two_sided"], places=12)
        self.assertAlmostEqual(float(comp["p_two_sided"]), 2.0 / 65536.0, places=12)
        self.assertIn("verdict", comp)
        self.assertEqual(comp["verdict"], "支持")
        self.assertEqual(verdict_superiority(ref_stat, ref_stat["p_two_sided"]), "支持")
        self.assertEqual(holm_adjust([0.001, 0.04]), holm_adjust_ref([0.001, 0.04]))

    def test_t1_f07_engineering_cost_profiler_and_coverage(self) -> None:
        """T1-F7.4: Unified evaluator logs positive runtime, memory, coverage=100%, and JSON serializability."""
        eval_fn = MilestoneResolver.get_first_callable(
            "eval.evaluator", ("evaluate_receiver", "evaluate_frames", "evaluate_outputs"), "Milestone M2 (F7)"
        )
        pairs = []
        for i in range(8):
            fr, _ = make_standalone_frame_proxy(E2E_SEED_BASE + 760 + i, frame_id=i)
            out = make_synthetic_receiver_output(fr, latency_ms=1.0 + 0.25 * i, n_candidates=3, param_count=128)
            pairs.append((fr, out))
        res = eval_fn(pairs)
        self.assertGreater(float(res["mean_ms_per_frame"]), 0.0)
        self.assertGreaterEqual(float(res["p95_ms_per_frame"]), float(res["mean_ms_per_frame"]) * 0.5)
        self.assertGreater(float(res["peak_rss_mb"]), 0.0)
        self.assertGreaterEqual(int(res["n_candidates"]), 1)
        self.assertGreaterEqual(int(res["param_count"]), 0)
        self.assertEqual(int(res["success_count"]), 8)
        self.assertEqual(int(res["fail_count"]), 0)
        self.assertAlmostEqual(float(res["coverage_pct"]), 100.0, places=6)
        json.dumps(res, allow_nan=False)

    def test_t1_f07_stress_perturbations_nonmutating_and_monotonic(self) -> None:
        """T1-F7.5: Stress sweeps under AWGN and CFO leave input frames untouched and increase baseline error."""
        perturb = MilestoneResolver.get_callable(
            "adapters.sdr_ofdm_adapter", "apply_stress_perturbation", "Milestone M1 (F4)"
        )
        run_frame = MilestoneResolver.get_callable("receivers.classical", "run_frame", "Milestone M2 (F6)")
        eval_fn = MilestoneResolver.get_first_callable(
            "eval.evaluator", ("evaluate_receiver", "evaluate_frames", "evaluate_outputs"), "Milestone M2 (F7)"
        )
        frames = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 780 + i, snr_db=22.0)[0] for i in range(8)]
        snaps = [(f.rx_grid.copy(), f.tx_bits.copy()) for f in frames]

        clean_res = eval_fn([(f, run_frame(f)) for f in frames])
        stressed = [perturb(f, extra_awgn_db=10.0, cfo_norm=0.05, seed=700 + i) for i, f in enumerate(frames)]
        stress_res = eval_fn([(f, run_frame(f)) for f in stressed])

        for f, (rx_s, bits_s) in zip(frames, snaps):
            np.testing.assert_array_equal(f.rx_grid, rx_s)
            np.testing.assert_array_equal(f.tx_bits, bits_s)
        self.assertGreater(float(stress_res["sym_mse"]), float(clean_res["sym_mse"]))
        self.assertGreaterEqual(float(stress_res["ber"]), float(clean_res["ber"]))

    # =========================================================================
    # Feature F8: Group A — Global Reference & Content Matching (5 tests)
    # =========================================================================

    def test_t1_f08_1_shared_context_pool_modes_and_contracts(self) -> None:
        """T1-F8.1: All 4 Group A modes share context pool $C$, report valid weights, and obey `ReceiverOutput`."""
        run_a = MilestoneResolver.get_callable("receivers.group_a_reference", "run_frame", "Milestone M3 (F8)")
        frame, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 801, snr_db=18.0)
        pool = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 810 + j, snr_db=18.0, frame_id=j + 1)[0] for j in range(8)]
        for mode in ("local", "global_uniform", "global_reliability", "global_content_attn"):
            out = run_a(frame, context_pool=pool, mode=mode, local_k=2)
            self.assertTrue(np.all(np.isfinite(out.h_est)))
            self.assertTrue(np.all(np.isfinite(out.h_var)) and np.all(extract_active_h(out.h_var) > 0.0))
            self.assertEqual(out.eq_symbols.shape, (N_DATA_RES,))
            self.assertEqual(out.bit_llrs.shape, (N_DATA_RES, 4))
            self.assertFalse(bool(out.aux_info["failed"]))
            weights = np.asarray(out.aux_info["weights"], dtype=np.float64)
            self.assertTrue(np.all(weights >= -1e-12))
            self.assertAlmostEqual(float(np.sum(weights)), 1.0, delta=1e-8)
            if mode == "local":
                self.assertLess(int(out.aux_info["n_values_accessed"]), len(pool))
            else:
                self.assertEqual(int(out.aux_info["n_values_accessed"]), len(pool))

    def test_t1_f08_2_content_matched_attention_beats_local_and_uniform(self) -> None:
        """T1-F8.2: Calibrated global content attention beats fixed-$k$ local and uniform when index $J \\ge k$."""
        run_a = MilestoneResolver.get_callable("receivers.group_a_reference", "run_frame", "Milestone M3 (F8)")
        mse_loc, mse_unif, mse_attn, j_weights = [], [], [], []
        for idx in range(24):
            fr, tr = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 830 + idx, snr_db=14.0, n_taps=4)
            # Build pool where J=5 shares the exact channel realization, while k=2 local only sees 0,1
            pool = []
            for m in range(8):
                seed_m = (E2E_SEED_BASE + 830 + idx) if m == 5 else (E2E_SEED_BASE + 8000 + idx * 16 + m)
                p_fr, _ = make_synthetic_real_ofdm_frame(seed_m, snr_db=22.0, frame_id=100 + m)
                pool.append(p_fr)
            o_loc = run_a(fr, context_pool=pool, mode="local", local_k=2)
            o_unif = run_a(fr, context_pool=pool, mode="global_uniform", local_k=2)
            o_attn = run_a(fr, context_pool=pool, mode="global_content_attn", local_k=2)
            h_act = tr["h_true"][:, FFT_OFFSET : FFT_OFFSET + N_ACTIVE]
            mse_loc.append(float(np.mean(np.abs(extract_active_h(o_loc.h_est) - h_act) ** 2)))
            mse_unif.append(float(np.mean(np.abs(extract_active_h(o_unif.h_est) - h_act) ** 2)))
            mse_attn.append(float(np.mean(np.abs(extract_active_h(o_attn.h_est) - h_act) ** 2)))
            j_weights.append(float(o_attn.aux_info["weights"][5]))
        self.assertLess(float(np.mean(mse_attn)), float(np.mean(mse_loc)))
        self.assertLess(float(np.mean(mse_attn)), float(np.mean(mse_unif)))
        self.assertGreater(float(np.mean(j_weights)), 0.4)

    def test_t1_f08_3_classical_reliability_vs_uniform_under_heterogeneous_snr(self) -> None:
        """T1-F8.3: `global_reliability` assigns higher weight to low-$n_0$ references than `global_uniform`."""
        run_a = MilestoneResolver.get_callable("receivers.group_a_reference", "run_frame", "Milestone M3 (F8)")
        fr, tr = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 860, snr_db=14.0)
        pool = []
        for m in range(6):
            snr_m = 28.0 if m < 2 else 0.0
            seed_m = (E2E_SEED_BASE + 860) if m < 2 else (E2E_SEED_BASE + 9000 + m)
            p_fr, _ = make_synthetic_real_ofdm_frame(seed_m, snr_db=snr_m, frame_id=m)
            pool.append(p_fr)
        o_unif = run_a(fr, context_pool=pool, mode="global_uniform")
        o_rel = run_a(fr, context_pool=pool, mode="global_reliability")
        w_rel = np.asarray(o_rel.aux_info["weights"], dtype=np.float64)
        self.assertGreater(float(np.mean(w_rel[:2])), 1.0 / 6.0)
        h_act = tr["h_true"][:, FFT_OFFSET : FFT_OFFSET + N_ACTIVE]
        mse_unif = float(np.mean(np.abs(extract_active_h(o_unif.h_est) - h_act) ** 2))
        mse_rel = float(np.mean(np.abs(extract_active_h(o_rel.h_est) - h_act) ** 2))
        self.assertLess(mse_rel, mse_unif)

    def test_t1_f08_4_dev_calibrated_temperature_and_equivalence_to_bayes(self) -> None:
        """T1-F8.4: Dev-calibrated temperature $\\tau^*$ is positive, finite, and yields valid Bayes softmax."""
        cal_or_run = MilestoneResolver.get_first_callable(
            "receivers.group_a_reference",
            ("calibrate_group_a_temperature", "calibrate_temperature", "fit_temperature", "run_frame"),
            "Milestone M3 (F8)",
        )
        run_a = MilestoneResolver.get_callable("receivers.group_a_reference", "run_frame", "Milestone M3 (F8)")
        dev_frames = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 870 + i, snr_db=16.0, split="dev")[0] for i in range(8)]
        pools = [
            [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 880 + j, snr_db=18.0)[0] for j in range(4)]
            for _ in range(8)
        ]
        if cal_or_run is not run_a:
            tau = float(cal_or_run(dev_frames, pools))
            self.assertTrue(math.isfinite(tau) and tau > 0.0)
            out = run_a(dev_frames[0], context_pool=pools[0], mode="global_content_attn", temperature=tau)
            w = np.asarray(out.aux_info["weights"], dtype=np.float64)
            self.assertTrue(np.all(w >= -1e-12))
            self.assertAlmostEqual(float(np.sum(w)), 1.0, delta=1e-8)
        else:
            entropies = []
            for tau in (0.25, 1.0, 4.0):
                out = run_a(dev_frames[0], context_pool=pools[0], mode="global_content_attn", temperature=tau)
                w = np.asarray(out.aux_info["weights"], dtype=np.float64)
                self.assertTrue(np.all(w >= -1e-12))
                self.assertAlmostEqual(float(np.sum(w)), 1.0, delta=1e-8)
                entropies.append(float(-np.sum(w * np.log(np.maximum(w, 1e-15)))))
            self.assertLessEqual(entropies[0], entropies[2] + 1e-8)

    def test_t1_f08_5_cstar_gauge_equivariance_group_a(self) -> None:
        """T1-F8.5: Group A satisfies exact $C^*$ equivariance across all 4 reference modes."""
        run_a = MilestoneResolver.get_callable("receivers.group_a_reference", "run_frame", "Milestone M3 (F8)")
        c = 7.5 * np.exp(1.23j)
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 890, snr_db=18.0)
        pool = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 891 + j, snr_db=18.0)[0] for j in range(4)]
        fr_s = scale_real_ofdm_frame(fr, c)
        pool_s = [scale_real_ofdm_frame(p, c) for p in pool]
        for mode in ("local", "global_uniform", "global_reliability", "global_content_attn"):
            o1 = run_a(fr, context_pool=pool, mode=mode)
            o2 = run_a(fr_s, context_pool=pool_s, mode=mode)
            np.testing.assert_allclose(o2.aux_info["weights"], o1.aux_info["weights"], atol=1e-8)
            np.testing.assert_allclose(o2.h_est, c * o1.h_est, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(o2.h_var, (abs(c) ** 2) * o1.h_var, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(o2.bit_llrs, o1.bit_llrs, atol=1e-4)
            self.assertEqual(int(np.sum(o2.hard_bits != o1.hard_bits)), 0)

    # =========================================================================
    # Feature F9: Group B — Hard Selection vs Posterior Fusion (5 tests)
    # =========================================================================

    def test_t1_f09_1_shared_branch_bank_and_simplex_weights(self) -> None:
        """T1-F9.1: All 3 Group B modes operate on the same $R$-branch bank with valid simplex weights."""
        run_b = MilestoneResolver.get_callable("receivers.group_b_fusion", "run_frame", "Milestone M3 (F9)")
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 901, snr_db=18.0)
        for mode in ("hard_map", "posterior_mean_plugin", "exact_mixture_likelihood"):
            out = run_b(fr, mode=mode)
            self.assertGreaterEqual(int(out.aux_info["n_candidates"]), 2)
            w = np.asarray(out.aux_info["branch_weights"], dtype=np.float64)
            self.assertEqual(len(w), int(out.aux_info["n_candidates"]))
            self.assertTrue(np.all(w >= -1e-12))
            self.assertAlmostEqual(float(np.sum(w)), 1.0, delta=1e-8)

    def test_t1_f09_2_posterior_mean_minimizes_channel_mse_over_hard_map(self) -> None:
        """T1-F9.2: Posterior mean $m = \\sum_r \\alpha_r m_r$ achieves strictly lower/equal MSE than hard MAP."""
        fuse_fn = MilestoneResolver.get_first_callable(
            "receivers.group_b_fusion",
            ("fuse_candidate_branches", "run_branches", "run_frame"),
            "Milestone M3 (F9)",
        )
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 902, snr_db=18.0)
        rng = np.random.default_rng(E2E_SEED_BASE + 903)
        mse_map, mse_mean = 0.0, 0.0
        alpha = np.array([0.55, 0.45], dtype=np.float64)
        for _ in range(64):
            m0 = np.full((N_SYMB, N_ACTIVE), 1.0 + 0.0j, dtype=np.complex128)
            m1 = np.full((N_SYMB, N_ACTIVE), -1.0 + 0.0j, dtype=np.complex128)
            v_w = np.full((N_SYMB, N_ACTIVE), 0.05, dtype=np.float64)
            true_r = int(rng.random() >= alpha[0])
            h_true = m0 if true_r == 0 else m1
            o_map = fuse_fn(fr, branches=[(m0, v_w), (m1, v_w)], branch_weights=alpha, mode="hard_map")
            o_mean = fuse_fn(fr, branches=[(m0, v_w), (m1, v_w)], branch_weights=alpha, mode="posterior_mean_plugin")
            h_map_act = extract_active_h(o_map.h_est)
            h_mean_act = extract_active_h(o_mean.h_est)
            np.testing.assert_allclose(h_mean_act, alpha[0] * m0 + alpha[1] * m1, atol=1e-6)
            mse_map += float(np.mean(np.abs(h_map_act - h_true) ** 2))
            mse_mean += float(np.mean(np.abs(h_mean_act - h_true) ** 2))
        self.assertLess(mse_mean, mse_map)

    def test_t1_f09_3_total_covariance_includes_between_branch_spread(self) -> None:
        """T1-F9.3: Fused predictive variance equals within-branch + between-branch spread (Total Covariance)."""
        fuse_fn = MilestoneResolver.get_first_callable(
            "receivers.group_b_fusion", ("fuse_candidate_branches", "run_branches", "run_frame"), "Milestone M3 (F9)"
        )
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 904, snr_db=18.0)
        m0 = np.full((N_SYMB, N_ACTIVE), 1.0 + 0.2j, dtype=np.complex128)
        m1 = np.full((N_SYMB, N_ACTIVE), -0.5 + 0.8j, dtype=np.complex128)
        vw = np.full((N_SYMB, N_ACTIVE), 0.04, dtype=np.float64)
        alpha = np.array([0.6, 0.4], dtype=np.float64)
        out = fuse_fn(fr, branches=[(m0, vw), (m1, vw)], branch_weights=alpha, mode="posterior_mean_plugin")
        m_bar = alpha[0] * m0 + alpha[1] * m1
        expected_var = vw + alpha[0] * (np.abs(m0 - m_bar) ** 2) + alpha[1] * (np.abs(m1 - m_bar) ** 2)
        h_var_act = extract_active_h(out.h_var)
        np.testing.assert_allclose(h_var_act, expected_var, atol=1e-6)
        self.assertTrue(np.all(h_var_act > vw))

    def test_t1_f09_4_exact_mixture_likelihood_beats_plugin_on_multimodal_channel(self) -> None:
        """T1-F9.4: Exact mixture likelihood $\\sum_r \\alpha_r g_r(c)$ outperforms plug-in on bimodal channels."""
        fuse_fn = MilestoneResolver.get_first_callable(
            "receivers.group_b_fusion", ("fuse_candidate_branches", "run_branches", "run_frame"), "Milestone M3 (F9)"
        )
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 905, snr_db=22.0, flat_channel=True)
        # Bimodal phase/gain hypotheses where plug-in m = 0.6*1.0 + 0.4*(-2.0) = -0.2 distorts constellation
        m0 = np.full((N_SYMB, N_ACTIVE), 1.0 + 0.0j, dtype=np.complex128)
        m1 = np.full((N_SYMB, N_ACTIVE), -2.0 + 0.0j, dtype=np.complex128)
        vw = np.full((N_SYMB, N_ACTIVE), 0.01, dtype=np.float64)
        alpha = np.array([0.6, 0.4], dtype=np.float64)
        o_plug = fuse_fn(fr, branches=[(m0, vw), (m1, vw)], branch_weights=alpha, mode="posterior_mean_plugin")
        o_mix = fuse_fn(fr, branches=[(m0, vw), (m1, vw)], branch_weights=alpha, mode="exact_mixture_likelihood")
        ber_plug = float(np.mean(o_plug.hard_bits != fr.tx_bits))
        ber_mix = float(np.mean(o_mix.hard_bits != fr.tx_bits))
        self.assertLessEqual(ber_mix, ber_plug)

    def test_t1_f09_5_vertex_degeneration_alpha_one_equals_hard_map(self) -> None:
        """T1-F9.5: At simplex vertex $\\alpha = [0, 1, 0, 0]$, all 3 Group B arms match identically."""
        fuse_fn = MilestoneResolver.get_first_callable(
            "receivers.group_b_fusion", ("fuse_candidate_branches", "run_branches", "run_frame"), "Milestone M3 (F9)"
        )
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 906, snr_db=20.0)
        branches = [
            (np.full((N_SYMB, N_ACTIVE), 0.7 + 0.1j * r, dtype=np.complex128), np.full((N_SYMB, N_ACTIVE), 0.02, dtype=np.float64))
            for r in range(4)
        ]
        alpha = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        o_map = fuse_fn(fr, branches=branches, branch_weights=alpha, mode="hard_map")
        o_plug = fuse_fn(fr, branches=branches, branch_weights=alpha, mode="posterior_mean_plugin")
        o_mix = fuse_fn(fr, branches=branches, branch_weights=alpha, mode="exact_mixture_likelihood")
        np.testing.assert_allclose(extract_active_h(o_plug.h_est), extract_active_h(o_map.h_est), atol=1e-10)
        np.testing.assert_allclose(extract_active_h(o_mix.h_est), extract_active_h(o_map.h_est), atol=1e-10)
        np.testing.assert_allclose(o_plug.bit_llrs, o_map.bit_llrs, atol=1e-10)
        np.testing.assert_allclose(o_mix.bit_llrs, o_map.bit_llrs, atol=1e-10)
        np.testing.assert_array_equal(o_plug.hard_bits, o_map.hard_bits)
        np.testing.assert_array_equal(o_mix.hard_bits, o_map.hard_bits)

    # =========================================================================
    # Feature F10: Group C — Lightweight Trainable / Attention Module (5 tests)
    # =========================================================================

    def test_t1_f10_1_train_dev_fit_and_receiver_output_contract(self) -> None:
        """T1-F10.1: Group C trains cleanly on Train/Dev and produces contract-compliant `ReceiverOutput`."""
        cls_c = MilestoneResolver.get_first_callable(
            "receivers.group_c_trainable", ("GroupCTrainableReceiver", "GroupCReceiver"), "Milestone M4 (F10)"
        )
        train_f = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1000 + i, snr_db=16.0, split="train")[0] for i in range(8)]
        dev_f = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1010 + i, snr_db=16.0, split="dev")[0] for i in range(4)]
        test_f, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1020, snr_db=18.0, split="test")
        model = cls_c(seed=42)
        model.fit(train_f, dev_f, epochs=2)
        out = model.run_frame(test_f)
        self.assertTrue(np.all(np.isfinite(out.h_est)))
        self.assertTrue(np.all(np.isfinite(out.h_var)) and np.all(extract_active_h(out.h_var) > 0.0))
        self.assertEqual(out.bit_llrs.shape, (N_DATA_RES, 4))
        self.assertFalse(bool(out.aux_info["failed"]))

    def test_t1_f10_2_strict_no_test_leakage_in_training(self) -> None:
        """T1-F10.2: Group C `.fit()` rejects `"test"` frames and weights never depend on evaluation frames."""
        cls_c = MilestoneResolver.get_first_callable(
            "receivers.group_c_trainable", ("GroupCTrainableReceiver", "GroupCReceiver"), "Milestone M4 (F10)"
        )
        train_f = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1030 + i, snr_db=16.0, split="train")[0] for i in range(6)]
        dev_f = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1040 + i, snr_db=16.0, split="dev")[0] for i in range(3)]
        bad_test, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1050, snr_db=16.0, split="test")

        m1 = cls_c(seed=123)
        m1.fit(train_f, dev_f, epochs=2)
        with self.assertRaises(ValueError):
            cls_c(seed=123).fit(train_f + [bad_test], dev_f, epochs=1)
        with self.assertRaises(ValueError):
            cls_c(seed=123).fit(train_f, dev_f + [bad_test], epochs=1)
        out_a = m1.run_frame(bad_test)
        m2 = cls_c(seed=123)
        m2.fit(train_f, dev_f, epochs=2)
        out_b = m2.run_frame(bad_test)
        np.testing.assert_array_equal(out_a.bit_llrs, out_b.bit_llrs)

    def test_t1_f10_3_architecture_param_seed_version_logging(self) -> None:
        """T1-F10.3: Group C logs architecture, exact `param_count`, `seed`, `train_time_s`, and versions."""
        cls_c = MilestoneResolver.get_first_callable(
            "receivers.group_c_trainable", ("GroupCTrainableReceiver", "GroupCReceiver"), "Milestone M4 (F10)"
        )
        train_f = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1060 + i, snr_db=16.0, split="train")[0] for i in range(4)]
        dev_f = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1070 + i, snr_db=16.0, split="dev")[0] for i in range(2)]
        test_f, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1080, snr_db=18.0, split="test")
        model = cls_c(seed=77)
        model.fit(train_f, dev_f, epochs=1)
        out = model.run_frame(test_f)
        aux = out.aux_info
        self.assertIsInstance(aux["architecture"], str)
        self.assertGreater(len(aux["architecture"]), 0)
        self.assertIsInstance(aux["param_count"], int)
        self.assertGreater(aux["param_count"], 0)
        self.assertEqual(int(aux["seed"]), 77)
        self.assertGreaterEqual(float(aux["train_time_s"]), 0.0)
        self.assertGreaterEqual(int(aux["n_candidates"]), 1)
        self.assertIn("versions", aux)

    def test_t1_f10_4_cstar_equivariance_and_scale_robustness(self) -> None:
        """T1-F10.4: Group C preserves $C^*$ equivariance across $r \\in \\{10^{-5}, 1, 10^5\\}$ and arbitrary phases."""
        cls_c = MilestoneResolver.get_first_callable(
            "receivers.group_c_trainable", ("GroupCTrainableReceiver", "GroupCReceiver"), "Milestone M4 (F10)"
        )
        train_f = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1090 + i, snr_db=16.0, split="train")[0] for i in range(4)]
        dev_f = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1100 + i, snr_db=16.0, split="dev")[0] for i in range(2)]
        test_f, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1110, snr_db=18.0, split="test")
        model = cls_c(seed=99)
        model.fit(train_f, dev_f, epochs=2)
        o_ref = model.run_frame(test_f)
        for c in (1e-5 * np.exp(1.37j), 1e5 * np.exp(-0.5j * math.pi)):
            o_s = model.run_frame(scale_real_ofdm_frame(test_f, c))
            rel = float(np.linalg.norm(o_s.h_est - c * o_ref.h_est) / (abs(c) * np.linalg.norm(o_ref.h_est) + 1e-12))
            self.assertLess(rel, 1e-4)
            np.testing.assert_allclose(o_s.bit_llrs, o_ref.bit_llrs, atol=1e-3)
            self.assertEqual(int(np.sum(o_s.hard_bits != o_ref.hard_bits)), 0)

    def test_t1_f10_5_deterministic_inference_under_eval(self) -> None:
        """T1-F10.5: Repeated Group C inference on same test frame is bit-identical (`max_diff == 0.0`)."""
        cls_c = MilestoneResolver.get_first_callable(
            "receivers.group_c_trainable", ("GroupCTrainableReceiver", "GroupCReceiver"), "Milestone M4 (F10)"
        )
        train_f = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1120 + i, snr_db=16.0, split="train")[0] for i in range(4)]
        dev_f = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1130 + i, snr_db=16.0, split="dev")[0] for i in range(2)]
        test_f, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 1140, snr_db=18.0, split="test")
        model = cls_c(seed=55)
        model.fit(train_f, dev_f, epochs=1)
        o1 = model.run_frame(test_f)
        o2 = model.run_frame(test_f)
        np.testing.assert_array_equal(o1.h_est, o2.h_est)
        np.testing.assert_array_equal(o1.bit_llrs, o2.bit_llrs)
        np.testing.assert_array_equal(o1.hard_bits, o2.hard_bits)

    # =========================================================================
    # Feature F11: Full Codespace Run, Figures, Report & Delivery (5 tests)
    # =========================================================================

    def test_t1_f11_1_results_artifacts_and_per_frame_schema(self) -> None:
        """T1-F11.1: `results/` contains non-empty summary & per-frame artifacts covering all experiment arms."""
        MilestoneResolver.require_artifact("REPORT.md", "Milestone M5 (F11)")
        res_dir = MilestoneResolver.require_artifact("results", "Milestone M5 (F11)", allow_dir=True)
        summary_files = list(res_dir.glob("*summary*.json"))
        self.assertGreater(len(summary_files), 0, "Missing summary JSON in results/")
        per_frame_files = list(res_dir.glob("*per_frame*.jsonl")) + list(res_dir.glob("*per_frame*.csv"))
        self.assertGreater(len(per_frame_files), 0, "Missing per-frame JSONL/CSV in results/")
        summary = json.loads(summary_files[0].read_text(encoding="utf-8"))
        arms = summary.get("arms", summary)
        self.assertGreater(len(arms), 0)
        for arm_name, rec in arms.items():
            if not isinstance(rec, dict) or "bit_errors" not in rec:
                continue
            self.assertIsInstance(rec["bit_errors"], int)
            self.assertIsInstance(rec["total_bits"], int)
            self.assertIsInstance(rec["frame_errors"], int)
            self.assertIsInstance(rec["total_frames"], int)
            self.assertGreater(float(rec["mean_ms_per_frame"]), 0.0)

    def test_t1_f11_2_figures_exist_nonempty_and_valid_headers(self) -> None:
        """T1-F11.2: `figures/` contains at least 2 valid PNG/PDF plots > 1024 bytes."""
        fig_dir = MilestoneResolver.require_artifact("figures", "Milestone M5 (F11)", allow_dir=True)
        figs = [p for p in fig_dir.iterdir() if p.suffix.lower() in {".png", ".pdf"}]
        self.assertGreaterEqual(len(figs), 2)
        for p in figs:
            data = p.read_bytes()
            self.assertGreater(len(data), 1024, f"Figure too small: {p}")
            if p.suffix.lower() == ".png":
                self.assertTrue(data.startswith(b"\x89PNG\r\n\x1a\n"), f"Invalid PNG header: {p}")
            else:
                self.assertTrue(data.startswith(b"%PDF-"), f"Invalid PDF header: {p}")

    def test_t1_f11_3_report_md_structure_citations_and_consistency(self) -> None:
        """T1-F11.3: `REPORT.md` has required sections, academic citations, and consistent dataset references."""
        report_path = MilestoneResolver.require_artifact("REPORT.md", "Milestone M5 (F11)")
        text = report_path.read_text(encoding="utf-8")
        self.assertGreater(len(text), 2000)
        self.assertIn("sdr_ofdm_dataset_v1", text)
        self.assertIn("10.17632/k72b4pnc5g.1", text)
        low = text.lower()
        self.assertTrue("classical" in low or "经典" in text)
        self.assertTrue("group a" in low or "a 组" in low or "a组" in low)
        self.assertTrue("group b" in low or "b 组" in low or "b组" in low)
        self.assertTrue("group c" in low or "c 组" in low or "c组" in low)
        self.assertTrue("vaswani" in low or "1706.03762" in low)
        self.assertTrue(any(tok in low for tok in ("citation", "reference", "参考", "引用")))

    def test_t1_f11_4_governance_branch_and_main_immutability(self) -> None:
        """T1-F11.4: Active branch is not `main`, `main` is pinned at `63ad2ec`, and `.agents/` is untracked."""
        MilestoneResolver.require_artifact("REPORT.md", "Milestone M5 (F11)")
        self.assertNotEqual(run_git("rev-parse", "--abbrev-ref", "HEAD"), "main")
        self.assertEqual(run_git("rev-parse", "main"), EXPECTED_MAIN_SHA)
        self.assertEqual(run_git("ls-files", ".agents"), "")

    def test_t1_f11_5_no_large_raw_binaries_committed(self) -> None:
        """T1-F11.5: Tracked files under `research/realdata_sdr_ofdm` contain zero raw `.pth`/`.mat`/`.tar.gz`."""
        MilestoneResolver.require_artifact("REPORT.md", "Milestone M5 (F11)")
        tracked = run_git("ls-files", "research/realdata_sdr_ofdm")
        for line in tracked.splitlines():
            low = line.strip().lower()
            self.assertFalse(low.endswith((".pth", ".mat", ".tar.gz", ".zip", ".h5")), f"Tracked raw binary: {line}")
