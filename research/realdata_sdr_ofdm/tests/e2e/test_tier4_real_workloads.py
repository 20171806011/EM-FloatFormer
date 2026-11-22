"""
EM-FloatFormer Real-Data SDR OFDM E2E Suite — Tier 4: Real-World Workload Scenarios (7 Tests).

Exercises end-to-end multi-frame pipelines, statistical audits, physical stress sweeps,
controlled mechanism ablations, 24-octave C* swings, fault accounting, and governance:
- T4-S01: 16–64 Frame Smoke & Full Test-Manifest End-to-End Pipeline
- T4-S02: Paired `p12/audit.py` Statistical Inference & Deterministic Replay (`max_float_diff == 0.0`)
- T4-S03: Multi-Axis Physical Stress Sweep (AWGN + CFO + Continuous Phase Drift)
- T4-S04: Controlled Mechanism Contrast (Group A Scope/Content & Group B MAP/Plug-in/Mixture)
- T4-S05: 24-Octave $C^*$ Gauge & Dynamic-Range Swing Workload ($2^{-12}..2^{12}$)
- T4-S06: Engineering Cost, Memory & Fault-Injection Failure Accounting Workload
- T4-S07: Full Repository, Artifact, Citation & Governance Audit (M5 Gate)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import unittest

import numpy as np
import torch

from research.realdata_sdr_ofdm.tests.e2e.harness import (
    BITS_PER_TTI,
    E2E_SEED_BASE,
    EXPECTED_MAIN_SHA,
    EXPECTED_P12_BASE_SHA,
    EXPECTED_WORK_BRANCH,
    FFT_OFFSET,
    N_ACTIVE,
    N_SYMB,
    RESEARCH_ROOT,
    MilestoneResolver,
    extract_active_h,
    make_standalone_frame_proxy,
    make_synthetic_real_ofdm_frame,
    make_synthetic_receiver_output,
    read_json_file,
    run_git,
    scale_real_ofdm_frame,
)


class TestTier4RealWorkloads(unittest.TestCase):
    """Tier 4: Real-World Workload Scenarios (7 full E2E scenarios)."""

    def setUp(self) -> None:
        try:
            torch.set_num_threads(1)
        except RuntimeError:
            pass

    def test_t4_s01_smoke_and_full_pipeline_workload(self) -> None:
        """T4-S01: Full E2E cohort across Classical, Group A, Group B, and Group C beats random guessing."""
        run_c = MilestoneResolver.get_callable("receivers.classical", "run_frame", "Milestone M2 (F6)")
        run_a = MilestoneResolver.get_callable("receivers.group_a_reference", "run_frame", "Milestone M3 (F8)")
        run_b = MilestoneResolver.get_callable("receivers.group_b_fusion", "run_frame", "Milestone M3 (F9)")
        cls_c = MilestoneResolver.get_first_callable(
            "receivers.group_c_trainable", ("GroupCTrainableReceiver", "GroupCReceiver"), "Milestone M4 (F10)"
        )
        eval_fn = MilestoneResolver.get_first_callable(
            "eval.evaluator", ("evaluate_receiver", "evaluate_frames", "evaluate_outputs"), "Milestone M2 (F7)"
        )

        train_f = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 4100 + i, snr_db=18.0, split="train", group_id=f"tr_{i//4}")[0] for i in range(16)]
        dev_f = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 4120 + i, snr_db=18.0, split="dev", group_id=f"dv_{i//4}")[0] for i in range(8)]
        test_f = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 4140 + i, snr_db=18.0, split="test", group_id=f"blk_{i//2:02d}", frame_id=i)[0] for i in range(32)]

        model_c = cls_c(seed=401)
        model_c.fit(train_f, dev_f, epochs=2)

        for name, fn in (
            ("classical", lambda f: run_c(f)),
            ("group_a_attn", lambda f: run_a(f, context_pool=dev_f[:4], mode="global_content_attn")),
            ("group_b_mix", lambda f: run_b(f, mode="exact_mixture_likelihood")),
            ("group_c", lambda f: model_c.run_frame(f)),
        ):
            res = eval_fn([(f, fn(f)) for f in test_f])
            self.assertEqual(int(res["fail_count"]), 0, f"{name} had unexpected failures")
            self.assertEqual(int(res["total_bits"]), 32 * BITS_PER_TTI)
            self.assertLess(float(res["ber"]), 0.15, f"{name} BER too high at 18 dB")

    def test_t4_s02_paired_p12_audit_and_deterministic_replay(self) -> None:
        """T4-S02: Paired `p12/audit.py` G=16 inference and single-worker replay (`max_float_diff == 0.0`)."""
        run_c = MilestoneResolver.get_callable("receivers.classical", "run_frame", "Milestone M2 (F6)")
        run_b = MilestoneResolver.get_callable("receivers.group_b_fusion", "run_frame", "Milestone M3 (F9)")
        from p12.audit import exact_sign_flip_p, holm_adjust, verify_replay_records_exact

        frames = [
            make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 4200 + g, snr_db=16.0, split="test", group_id=f"blk_{g:02d}", frame_id=g)[0]
            for g in range(16)
        ]

        def _collect_records(rx_c, rx_b):
            recs = []
            for g, fr in enumerate(frames):
                oc = rx_c(fr)
                ob = rx_b(fr, mode="exact_mixture_likelihood")
                recs.append(
                    {
                        "split": "test",
                        "cluster": g,
                        "regime": "sdr_ofdm",
                        "link": g,
                        "seed": E2E_SEED_BASE + 4200 + g,
                        "arms": {
                            "classical": {
                                "bit_errors": int(np.sum(oc.hard_bits != fr.tx_bits)),
                                "ber": float(np.mean(oc.hard_bits != fr.tx_bits)),
                                "elapsed_ms": float(oc.aux_info["latency_ms"]),
                            },
                            "group_b": {
                                "bit_errors": int(np.sum(ob.hard_bits != fr.tx_bits)),
                                "ber": float(np.mean(ob.hard_bits != fr.tx_bits)),
                                "elapsed_ms": float(ob.aux_info["latency_ms"]),
                            },
                        },
                    }
                )
            return recs

        recs1 = _collect_records(run_c, run_b)
        recs2 = _collect_records(run_c, run_b)
        replay = verify_replay_records_exact(recs1, recs2)
        self.assertTrue(replay["exact_pass"])
        self.assertEqual(replay["mismatches"], 0)
        self.assertEqual(replay["max_float_diff"], 0.0)

        diffs_pp = [100.0 * (r["arms"]["classical"]["ber"] - r["arms"]["group_b"]["ber"]) for r in recs1]
        stat = exact_sign_flip_p(diffs_pp)
        self.assertEqual(len(holm_adjust([stat["p_two_sided"]])), 1)

    def test_t4_s03_multi_axis_physical_stress_sweep(self) -> None:
        """T4-S03: Multi-axis AWGN + CFO + phase drift sweep degrades gracefully without mutating inputs."""
        perturb = MilestoneResolver.get_callable(
            "adapters.sdr_ofdm_adapter", "apply_stress_perturbation", "Milestone M1 (F4)"
        )
        run_c = MilestoneResolver.get_callable("receivers.classical", "run_frame", "Milestone M2 (F6)")
        run_a = MilestoneResolver.get_callable("receivers.group_a_reference", "run_frame", "Milestone M3 (F8)")
        run_b = MilestoneResolver.get_callable("receivers.group_b_fusion", "run_frame", "Milestone M3 (F9)")

        base_fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 4300, snr_db=22.0, split="test")
        pool = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 4310 + j, snr_db=22.0, split="dev")[0] for j in range(4)]
        rx_snap = base_fr.rx_grid.copy()
        evms_c = []
        for awgn_db in (None, 12.0, 6.0):
            for cfo in (0.0, 0.04):
                fr_p = perturb(base_fr, extra_awgn_db=awgn_db, cfo_norm=cfo, phase_drift_rad=0.2, seed=4300)
                for out in (
                    run_c(fr_p),
                    run_a(fr_p, context_pool=pool, mode="global_content_attn"),
                    run_b(fr_p, mode="exact_mixture_likelihood"),
                ):
                    self.assertFalse(bool(out.aux_info["failed"]))
                    self.assertTrue(np.all(np.isfinite(out.bit_llrs)))
                if cfo == 0.0:
                    evms_c.append(float(np.mean(np.abs(run_c(fr_p).eq_symbols - base_fr.tx_symbols) ** 2)))
        self.assertLess(evms_c[0], evms_c[1])
        self.assertLess(evms_c[1], evms_c[2])
        np.testing.assert_array_equal(base_fr.rx_grid, rx_snap)

    def test_t4_s04_controlled_mechanism_contrast_group_a_and_b(self) -> None:
        """T4-S04: Controlled shared-budget contrasts validate Group A & Group B theoretical ordering."""
        run_a = MilestoneResolver.get_callable("receivers.group_a_reference", "run_frame", "Milestone M3 (F8)")
        fuse_b = MilestoneResolver.get_first_callable(
            "receivers.group_b_fusion", ("fuse_candidate_branches", "run_branches", "run_frame"), "Milestone M3 (F9)"
        )
        # Group A contrast
        mse_loc, mse_unif, mse_attn = [], [], []
        for i in range(16):
            fr, tr = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 4400 + i, snr_db=14.0)
            pool = [
                make_synthetic_real_ofdm_frame(
                    E2E_SEED_BASE + 4400 + i if j == 5 else E2E_SEED_BASE + 9900 + i * 8 + j,
                    snr_db=24.0,
                )[0]
                for j in range(6)
            ]
            h_act = tr["h_true"][:, FFT_OFFSET : FFT_OFFSET + N_ACTIVE]
            mse_loc.append(float(np.mean(np.abs(extract_active_h(run_a(fr, context_pool=pool, mode="local", local_k=2).h_est) - h_act) ** 2)))
            mse_unif.append(float(np.mean(np.abs(extract_active_h(run_a(fr, context_pool=pool, mode="global_uniform", local_k=2).h_est) - h_act) ** 2)))
            mse_attn.append(float(np.mean(np.abs(extract_active_h(run_a(fr, context_pool=pool, mode="global_content_attn", local_k=2).h_est) - h_act) ** 2)))
        self.assertLess(float(np.mean(mse_attn)), float(np.mean(mse_loc)))
        self.assertLess(float(np.mean(mse_attn)), float(np.mean(mse_unif)))

        # Group B contrast
        fr_b, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 4450, snr_db=22.0, flat_channel=True)
        m0 = np.full((N_SYMB, N_ACTIVE), 1.0, dtype=np.complex128)
        m1 = np.full((N_SYMB, N_ACTIVE), -2.0, dtype=np.complex128)
        vw = np.full((N_SYMB, N_ACTIVE), 0.01, dtype=np.float64)
        o_plug = fuse_b(fr_b, branches=[(m0, vw), (m1, vw)], branch_weights=[0.6, 0.4], mode="posterior_mean_plugin")
        o_mix = fuse_b(fr_b, branches=[(m0, vw), (m1, vw)], branch_weights=[0.6, 0.4], mode="exact_mixture_likelihood")
        self.assertLessEqual(float(np.mean(o_mix.hard_bits != fr_b.tx_bits)), float(np.mean(o_plug.hard_bits != fr_b.tx_bits)))

    def test_t4_s05_24_octave_cstar_dynamic_range_swing_workload(self) -> None:
        """T4-S05: 24-octave power swing ($2^{-12}..2^{12}$) yields 0 bit flips across all receiver families."""
        run_c = MilestoneResolver.get_callable("receivers.classical", "run_frame", "Milestone M2 (F6)")
        run_a = MilestoneResolver.get_callable("receivers.group_a_reference", "run_frame", "Milestone M3 (F8)")
        run_b = MilestoneResolver.get_callable("receivers.group_b_fusion", "run_frame", "Milestone M3 (F9)")
        cls_c = MilestoneResolver.get_first_callable(
            "receivers.group_c_trainable", ("GroupCTrainableReceiver", "GroupCReceiver"), "Milestone M4 (F10)"
        )
        tr = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 4500 + i, split="train")[0] for i in range(4)]
        dv = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 4505 + i, split="dev")[0] for i in range(2)]
        model_c = cls_c(seed=405)
        model_c.fit(tr, dv, epochs=1)

        base_fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 4510, snr_db=18.0, split="test")
        pool = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 4520 + j, snr_db=18.0)[0] for j in range(4)]

        arms = {
            "classical": lambda f, p: run_c(f),
            "group_a": lambda f, p: run_a(f, context_pool=p, mode="global_content_attn"),
            "group_b": lambda f, p: run_b(f, mode="exact_mixture_likelihood"),
            "group_c": lambda f, p: model_c.run_frame(f),
        }
        refs = {k: fn(base_fr, pool) for k, fn in arms.items()}
        for idx, p in enumerate([-12, -8, -4, 4, 8, 12]):
            c = (2.0**p) * np.exp(1j * (0.37 * idx + 0.2))
            fr_s = scale_real_ofdm_frame(base_fr, c)
            pool_s = [scale_real_ofdm_frame(x, c) for x in pool]
            for name, fn in arms.items():
                out = fn(fr_s, pool_s)
                self.assertTrue(np.all(np.isfinite(out.bit_llrs)))
                np.testing.assert_allclose(out.bit_llrs, refs[name].bit_llrs, atol=1e-3)
                self.assertEqual(int(np.sum(out.hard_bits != refs[name].hard_bits)), 0)

    def test_t4_s06_engineering_cost_memory_and_fault_accounting(self) -> None:
        """T4-S06: Evaluator profiles mixed valid/corrupted batch with exact failure accounting and cost stats."""
        eval_fn = MilestoneResolver.get_first_callable(
            "eval.evaluator", ("evaluate_receiver", "evaluate_frames", "evaluate_outputs"), "Milestone M2 (F7)"
        )
        pairs = []
        for i in range(10):
            fr, _ = make_standalone_frame_proxy(E2E_SEED_BASE + 4600 + i, frame_id=i)
            bad = i in (1, 5, 8)
            out = make_synthetic_receiver_output(
                fr,
                failed=bad,
                error=(f"Fault #{i}" if bad else None),
                latency_ms=1.5 + 0.1 * i,
                n_candidates=4,
                param_count=256,
            )
            pairs.append((fr, out))
        res = eval_fn(pairs)
        self.assertEqual(int(res["success_count"]), 7)
        self.assertEqual(int(res["fail_count"]), 3)
        self.assertEqual(int(res["total_frames"]), 10)
        self.assertAlmostEqual(float(res["coverage_pct"]), 70.0, places=6)
        self.assertGreater(float(res["mean_ms_per_frame"]), 0.0)
        self.assertGreater(float(res["peak_rss_mb"]), 0.0)
        self.assertEqual(int(res["param_count"]), 256)

    def test_t4_s07_full_repo_and_governance_audit(self) -> None:
        """T4-S07: End-to-end M5 gate validates all deliverables, split isolation, citations, and git governance."""
        self.assertEqual(run_git("branch", "--show-current"), EXPECTED_WORK_BRANCH)
        self.assertEqual(run_git("rev-parse", "main"), EXPECTED_MAIN_SHA)
        self.assertEqual(run_git("ls-files", ".agents"), "")

        ds_path = MilestoneResolver.require_artifact("DATA_SOURCES.md", "Milestone M1 (F2)")
        proto_path = MilestoneResolver.require_artifact("PROTOCOL.md", "Milestone M1 (F5)")
        mf_path = MilestoneResolver.require_artifact("manifests/raw_checksums.json", "Milestone M1 (F3)")
        sp_path = MilestoneResolver.require_artifact("manifests/splits.json", "Milestone M1 (F5)")
        res_dir = MilestoneResolver.require_artifact("results", "Milestone M5 (F11)", allow_dir=True)
        fig_dir = MilestoneResolver.require_artifact("figures", "Milestone M5 (F11)", allow_dir=True)
        rep_path = MilestoneResolver.require_artifact("REPORT.md", "Milestone M5 (F11)")

        self.assertGreater(ds_path.stat().st_size, 500)
        self.assertGreater(proto_path.stat().st_size, 500)
        read_json_file(mf_path)
        sp = read_json_file(sp_path)
        self.assertEqual(set(sp["train_indices"]) & set(sp["test_indices"]), set())
        self.assertGreater(len(list(res_dir.iterdir())), 0)
        self.assertGreaterEqual(len([p for p in fig_dir.iterdir() if p.suffix.lower() in {".png", ".pdf"}]), 2)
        self.assertGreater(rep_path.stat().st_size, 2000)
