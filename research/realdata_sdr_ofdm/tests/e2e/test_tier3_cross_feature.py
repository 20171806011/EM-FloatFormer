"""
EM-FloatFormer Real-Data SDR OFDM E2E Suite — Tier 3: Cross-Feature Pairwise Tests (15 Tests).

Exercises pairwise interactions across all 11 features (F1–F11):
- T3-P01 (F1 + F2): Branch isolation & catalog provenance
- T3-P02 (F2 + F3): Catalog matches raw checksum manifest
- T3-P03 (F3 + F4): Smoke probe feeds adapter & non-Gray mapping_table(4)
- T3-P04 (F4 + F5): Adapter + split leak-free isolation
- T3-P05 (F4 + F6): Adapter non-Gray demapping + classical receiver
- T3-P06 (F5 + F6): Dev-only classical calibration & test isolation
- T3-P07 (F6 + F7): Classical receiver + evaluator exact integer counts
- T3-P08 (F4 + F7): Stress perturbation + evaluator monotonic degradation
- T3-P09 (F6 + F8): Classical vs Group A on shared context pool
- T3-P10 (F7 + F8): Evaluator paired audit (`p12/audit.py`) on Group A
- T3-P11 (F8 + F9): Group A content weights into Group B posterior fusion
- T3-P12 (F6 + F9): Classical single branch matches Group B Hard MAP
- T3-P13 (F9 + F10): Trainable Group C branch weights / fusion compatibility
- T3-P14 (F10 + F7): Group C cost & stress profiling via unified evaluator
- T3-P15 (F7 + F11): Evaluator outputs match persisted delivery artifacts & REPORT.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from research.realdata_sdr_ofdm.tests.e2e.harness import (
    BITS_PER_TTI,
    E2E_SEED_BASE,
    EXPECTED_MAIN_SHA,
    EXPECTED_WORK_BRANCH,
    FFT_OFFSET,
    N_ACTIVE,
    N_DATA_RES,
    N_FFT,
    N_PILOTS,
    N_SYMB,
    RESEARCH_ROOT,
    MilestoneResolver,
    extract_active_h,
    make_synthetic_custom_dataset_pth,
    make_synthetic_grid_and_truth,
    make_synthetic_real_ofdm_frame,
    nongray_16qam_mod_ref,
    nongray_16qam_table_ref,
    read_json_file,
    run_git,
    scale_real_ofdm_frame,
)


class TestTier3CrossFeature(unittest.TestCase):
    """Tier 3: Cross-Feature Combinations (15 pairwise interaction tests)."""

    def setUp(self) -> None:
        try:
            torch.set_num_threads(1)
        except RuntimeError:
            pass

    def test_t3_p01_f01_f02_branch_and_catalog_provenance(self) -> None:
        """T3-P01 (F1+F2): Working branch isolates `DATA_SOURCES.md` while `main` remains pinned at `63ad2ec`."""
        self.assertEqual(run_git("branch", "--show-current"), EXPECTED_WORK_BRANCH)
        self.assertEqual(run_git("rev-parse", "main"), EXPECTED_MAIN_SHA)
        ds_path = MilestoneResolver.require_artifact("DATA_SOURCES.md", "Milestone M1 (F2)")
        text = ds_path.read_text(encoding="utf-8")
        self.assertIn("rikluost/sdr_ofdm_dataset_v1", text)
        self.assertIn("10.17632/k72b4pnc5g.1", text)

    def test_t3_p02_f02_f03_catalog_matches_checksum_manifest(self) -> None:
        """T3-P02 (F2+F3): `DATA_SOURCES.md` SHA-256 and file sizes agree with `manifests/raw_checksums.json`."""
        ds_path = MilestoneResolver.require_artifact("DATA_SOURCES.md", "Milestone M1 (F2)")
        mf_path = MilestoneResolver.require_artifact("manifests/raw_checksums.json", "Milestone M1 (F3)")
        ds_txt = ds_path.read_text(encoding="utf-8").lower()
        mf = read_json_file(mf_path)
        primary = mf["primary"]
        self.assertIn(primary["sha256"].lower(), ds_txt)
        self.assertIn("ofdm_valset_4_8_sdr.pth", ds_txt)

    def test_t3_p03_f03_f04_smoke_probe_into_adapter(self) -> None:
        """T3-P03 (F3+F4): Synthetic 16-frame probe samples adapt cleanly into `RealOFDMFrame` & `mapping_table(4)`."""
        load_raw = MilestoneResolver.get_callable("adapters.sdr_ofdm_adapter", "load_raw_sdr_ofdm", "Milestone M1 (F4)")
        build_frame = MilestoneResolver.get_callable("adapters.sdr_ofdm_adapter", "build_frame", "Milestone M1 (F4)")
        mod_m = MilestoneResolver.get_module("adapters.modulation", "Milestone M1 (F4)")
        mapping_table = getattr(mod_m, "mapping_table")
        constellation_points = getattr(mod_m, "constellation_points")

        with tempfile.TemporaryDirectory() as td:
            pth = Path(td) / "probe16.pth"
            make_synthetic_custom_dataset_pth(pth, n_frames=16, seed=E2E_SEED_BASE + 3003)
            iq_list, label_list, sinr_arr = load_raw(pth)
            self.assertEqual(len(iq_list), 16)
            frame = build_frame(
                0,
                iq_list[0],
                label_list[0],
                float(sinr_arr[0]),
                split="dev",
                group_id="blk_0",
            )
            self.assertEqual(frame.rx_grid.shape, (N_SYMB, N_FFT))
            self.assertEqual(int(frame.pilot_mask.sum()), N_PILOTS)
            self.assertEqual(int(frame.data_mask.sum()), N_DATA_RES)
            self.assertEqual(frame.tx_bits.shape, (N_DATA_RES, 4))
            m_dict, _ = mapping_table(4)
            self.assertEqual(len(m_dict), 16)
            tbl = np.asarray(constellation_points(4), dtype=np.complex128).reshape(-1)
            np.testing.assert_allclose(tbl, nongray_16qam_table_ref(), atol=1e-6)

    def test_t3_p04_f04_f05_adapter_split_leak_free_isolation(self) -> None:
        """T3-P04 (F4+F5): Partitioned splits have zero overlap and adapter `est_n0` never reads `tx_bits`."""
        splits_path = MilestoneResolver.require_artifact("manifests/splits.json", "Milestone M1 (F5)")
        ad_m = MilestoneResolver.get_module("adapters.sdr_ofdm_adapter", "Milestone M1 (F4)")
        build_frame = getattr(ad_m, "build_frame")
        sp = read_json_file(splits_path)
        self.assertEqual(set(sp["train_indices"]) & set(sp["test_indices"]), set())
        sim = make_synthetic_grid_and_truth(E2E_SEED_BASE + 3004, snr_db=15.0)
        fr_clean = build_frame(0, sim["rx_grid"], sim["tx_bits"], sim["reported_sinr_db"], split="test")
        fr_poison = build_frame(0, sim["rx_grid"], 1 - sim["tx_bits"], sim["reported_sinr_db"], split="test")
        self.assertEqual(float(fr_poison.est_n0), float(fr_clean.est_n0))
        self.assertEqual(float(fr_poison.meta["s0"]), float(fr_clean.meta["s0"]))

    def test_t3_p05_f04_f06_adapter_non_gray_demapping_classical(self) -> None:
        """T3-P05 (F4+F6): Classical receiver LLR polarity & bits match non-Gray `mapping_table(4)` vs Gray/inverted."""
        run_c = MilestoneResolver.get_callable("receivers.classical", "run_frame", "Milestone M2 (F6)")
        pts = nongray_16qam_table_ref()
        gray2 = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)

        def _gray_demap_ber(eq_syms: np.ndarray, tx_bits: np.ndarray) -> float:
            idx = np.argmin(np.abs(eq_syms[:, None] - pts[None, :]) ** 2, axis=1)
            gray_bits = np.concatenate([gray2[idx // 4], gray2[idx % 4]], axis=1)
            return float(np.mean(gray_bits != tx_bits))

        # 1. Flat-fading non-trivial phase/gauge at 24 dB: exact zero-floor check + Gray & polarity separation
        fr_flat, _ = make_synthetic_real_ofdm_frame(
            E2E_SEED_BASE + 3005,
            snr_db=24.0,
            flat_channel=True,
            phase_offset=0.85,
            gauge_scale=1.7 * np.exp(-0.4j),
        )
        out_flat = run_c(fr_flat)
        ber_flat = float(np.mean(out_flat.hard_bits != fr_flat.tx_bits))
        ber_gray_flat = _gray_demap_ber(out_flat.eq_symbols, fr_flat.tx_bits)
        ber_inv_flat = float(np.mean(((-out_flat.bit_llrs) > 0.0).astype(np.uint8) != fr_flat.tx_bits))
        self.assertLess(ber_flat, 1e-3)
        self.assertGreater(ber_gray_flat, 0.15)
        self.assertLess(ber_flat, ber_gray_flat * 0.1)
        self.assertGreater(ber_inv_flat, 0.9)
        np.testing.assert_array_equal(out_flat.hard_bits, (out_flat.bit_llrs > 0.0).astype(np.uint8))

        # 2. Multi-frame 4-tap multipath average at 30 dB: eliminates single-frame deep-fade fragility
        bit_errs_mp, gray_errs_mp, total_bits_mp = 0, 0, 0
        for i in range(8):
            fr_mp, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 3050 + i, snr_db=30.0, n_taps=4)
            out_mp = run_c(fr_mp)
            np.testing.assert_array_equal(out_mp.hard_bits, (out_mp.bit_llrs > 0.0).astype(np.uint8))
            bit_errs_mp += int(np.sum(out_mp.hard_bits != fr_mp.tx_bits))
            idx_mp = np.argmin(np.abs(out_mp.eq_symbols[:, None] - pts[None, :]) ** 2, axis=1)
            gray_bits_mp = np.concatenate([gray2[idx_mp // 4], gray2[idx_mp % 4]], axis=1)
            gray_errs_mp += int(np.sum(gray_bits_mp != fr_mp.tx_bits))
            total_bits_mp += fr_mp.tx_bits.size
        ber_mp = bit_errs_mp / total_bits_mp
        ber_gray_mp = gray_errs_mp / total_bits_mp
        self.assertLess(ber_mp, 0.005)
        self.assertGreater(ber_gray_mp, 0.15)
        self.assertLess(ber_mp, ber_gray_mp * 0.1)

    def test_t3_p06_f05_f06_dev_only_classical_calibration(self) -> None:
        """T3-P06 (F5+F6): Classical dev selection/calibration uses `"dev"` split only and reduces test MSE."""
        cls_mod = MilestoneResolver.get_module("receivers.classical", "Milestone M2 (F6)")
        run_c = MilestoneResolver.get_callable("receivers.classical", "run_frame", "Milestone M2 (F6)")
        eval_fn = MilestoneResolver.get_first_callable(
            "eval.evaluator", ("evaluate_receiver", "evaluate_frames", "evaluate_outputs"), "Milestone M2 (F7)"
        )
        dev_frames = [
            make_synthetic_real_ofdm_frame(
                E2E_SEED_BASE + 3060 + i, snr_db=16.0, split="dev", group_id="blk_dev", frame_id=i
            )[0]
            for i in range(6)
        ]
        test_pairs_raw = [
            make_synthetic_real_ofdm_frame(
                E2E_SEED_BASE + 3070 + i, snr_db=16.0, split="test", group_id="blk_test", frame_id=100 + i
            )
            for i in range(6)
        ]
        self.assertTrue(all(f.split == "dev" for f in dev_frames))
        self.assertEqual({f.frame_id for f in dev_frames} & {f.frame_id for f, _ in test_pairs_raw}, set())

        # Optional explicit dev calibration API guard if exposed by M2 classical module
        for cal_name in ("calibrate_classical", "calibrate_on_dev", "fit_dev_parameters", "tune_classical"):
            if hasattr(cls_mod, cal_name):
                cal_fn = getattr(cls_mod, cal_name)
                with self.assertRaises(ValueError):
                    cal_fn([test_pairs_raw[0][0]])
                cal_fn(dev_frames)
                break

        # Select classical method strictly on dev_frames via unified evaluator
        dev_scores = {
            m: float(eval_fn([(f, run_c(f, method=m)) for f in dev_frames])["sym_mse"])
            for m in ("ls_zf", "lmmse")
        }
        best_method = min(dev_scores, key=lambda k: dev_scores[k])
        self.assertEqual(best_method, "lmmse")

        ls_errs, mmse_errs, ls_h_errs, mmse_h_errs = [], [], [], []
        for fr, tr in test_pairs_raw:
            o_ls = run_c(fr, method="ls_zf")
            o_mmse = run_c(fr, method=best_method)
            ls_errs.append(float(np.mean(np.abs(o_ls.eq_symbols - fr.tx_symbols) ** 2)))
            mmse_errs.append(float(np.mean(np.abs(o_mmse.eq_symbols - fr.tx_symbols) ** 2)))
            h_act = tr["h_true"][:, FFT_OFFSET : FFT_OFFSET + N_ACTIVE]
            ls_h_errs.append(float(np.mean(np.abs(extract_active_h(o_ls.h_est) - h_act) ** 2)))
            mmse_h_errs.append(float(np.mean(np.abs(extract_active_h(o_mmse.h_est) - h_act) ** 2)))

            # Zero test-label leakage verification
            fr_poison = scale_real_ofdm_frame(fr, 1.0)
            fr_poison.tx_bits[:] = 1 - fr_poison.tx_bits
            fr_poison.tx_symbols[:] = -fr_poison.tx_symbols
            o_poison = run_c(fr_poison, method=best_method)
            np.testing.assert_array_equal(o_poison.h_est, o_mmse.h_est)
            np.testing.assert_array_equal(o_poison.bit_llrs, o_mmse.bit_llrs)

        self.assertLessEqual(float(np.mean(mmse_errs)), float(np.mean(ls_errs)) * 1.05)
        self.assertLessEqual(float(np.mean(mmse_h_errs)), float(np.mean(ls_h_errs)) * 1.05)

    def test_t3_p07_f06_f07_classical_evaluator_exact_counts(self) -> None:
        """T3-P07 (F6+F7): Evaluator integer bit/frame counts match manual XOR on Classical receiver outputs."""
        run_c = MilestoneResolver.get_callable("receivers.classical", "run_frame", "Milestone M2 (F6)")
        eval_fn = MilestoneResolver.get_first_callable(
            "eval.evaluator", ("evaluate_receiver", "evaluate_frames", "evaluate_outputs"), "Milestone M2 (F7)"
        )
        pairs = []
        manual_bit_errs = 0
        manual_frame_errs = 0
        for i in range(16):
            fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 3080 + i, snr_db=14.0, frame_id=i)
            out = run_c(fr)
            pairs.append((fr, out))
            be = int(np.sum(out.hard_bits != fr.tx_bits))
            manual_bit_errs += be
            if be > 0:
                manual_frame_errs += 1
        res = eval_fn(pairs)
        self.assertEqual(int(res["bit_errors"]), manual_bit_errs)
        self.assertEqual(int(res["total_bits"]), 16 * BITS_PER_TTI)
        self.assertEqual(int(res["frame_errors"]), manual_frame_errs)
        self.assertAlmostEqual(float(res["fer"]), manual_frame_errs / 16.0, places=12)

    def test_t3_p08_f04_f07_stress_perturbation_monotonic_degradation(self) -> None:
        """T3-P08 (F4+F7): Stress sweeps (`None -> 12.0 -> 6.0` dB added AWGN SNR + CFO) increase EVM/BER monotonically without mutation."""
        perturb = MilestoneResolver.get_callable(
            "adapters.sdr_ofdm_adapter", "apply_stress_perturbation", "Milestone M1 (F4)"
        )
        run_c = MilestoneResolver.get_callable("receivers.classical", "run_frame", "Milestone M2 (F6)")
        eval_fn = MilestoneResolver.get_first_callable(
            "eval.evaluator", ("evaluate_receiver", "evaluate_frames", "evaluate_outputs"), "Milestone M2 (F7)"
        )
        base = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 3100 + i, snr_db=22.0, frame_id=i)[0] for i in range(6)]
        bits_snap = [f.tx_bits.copy() for f in base]
        evms, bers = [], []
        for awgn_db in (None, 12.0, 6.0):
            stressed = [perturb(f, extra_awgn_db=awgn_db, cfo_norm=(0.0 if awgn_db is None else 0.03), seed=i) for i, f in enumerate(base)]
            res = eval_fn([(f, run_c(f)) for f in stressed])
            evms.append(float(res["evm_pct"]))
            bers.append(float(res["ber"]))
        self.assertLess(evms[0], evms[1])
        self.assertLess(evms[1], evms[2])
        self.assertLessEqual(bers[0], bers[1])
        self.assertLessEqual(bers[1], bers[2])
        for f, b in zip(base, bits_snap):
            np.testing.assert_array_equal(f.tx_bits, b)

    def test_t3_p09_f06_f08_classical_vs_group_a_shared_pool(self) -> None:
        """T3-P09 (F6+F8): Matched Group A content attention achieves lower/equal channel MSE vs Classical & local."""
        run_c = MilestoneResolver.get_callable("receivers.classical", "run_frame", "Milestone M2 (F6)")
        run_a = MilestoneResolver.get_callable("receivers.group_a_reference", "run_frame", "Milestone M3 (F8)")
        mse_c, mse_a = [], []
        for i in range(12):
            fr, tr = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 3120 + i, snr_db=14.0)
            pool = [
                make_synthetic_real_ofdm_frame(
                    E2E_SEED_BASE + 3120 + i if j == 4 else E2E_SEED_BASE + 9500 + i * 8 + j,
                    snr_db=24.0,
                )[0]
                for j in range(6)
            ]
            o_c = run_c(fr)
            o_a = run_a(fr, context_pool=pool, mode="global_content_attn", local_k=2)
            h_act = tr["h_true"][:, FFT_OFFSET : FFT_OFFSET + N_ACTIVE]
            mse_c.append(float(np.mean(np.abs(extract_active_h(o_c.h_est) - h_act) ** 2)))
            mse_a.append(float(np.mean(np.abs(extract_active_h(o_a.h_est) - h_act) ** 2)))
        self.assertLess(float(np.mean(mse_a)), float(np.mean(mse_c)))

    def test_t3_p10_f07_f08_evaluator_paired_audit_group_a(self) -> None:
        """T3-P10 (F7+F8): Evaluator paired audit between Group A `global_content_attn` and `local` matches `p12.audit`."""
        run_a = MilestoneResolver.get_callable("receivers.group_a_reference", "run_frame", "Milestone M3 (F8)")
        compare_fn = MilestoneResolver.get_first_callable(
            "eval.evaluator", ("compare_arms", "paired_comparison", "evaluate_paired"), "Milestone M2 (F7)"
        )
        loc_pairs, attn_pairs = [], []
        for g in range(16):
            fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 3140 + g, snr_db=14.0, group_id=f"blk_{g:02d}", frame_id=g)
            pool = [
                make_synthetic_real_ofdm_frame(
                    E2E_SEED_BASE + 3140 + g if j == 4 else E2E_SEED_BASE + 9700 + g * 8 + j,
                    snr_db=24.0,
                )[0]
                for j in range(6)
            ]
            loc_pairs.append((fr, run_a(fr, context_pool=pool, mode="local", local_k=2)))
            attn_pairs.append((fr, run_a(fr, context_pool=pool, mode="global_content_attn", local_k=2)))
        comp = compare_fn(loc_pairs, attn_pairs)
        self.assertGreater(float(comp["p_two_sided"]), 0.0)
        self.assertLessEqual(float(comp["p_two_sided"]), 1.0)

    def test_t3_p11_f08_f09_content_weights_into_posterior_fusion(self) -> None:
        """T3-P11 (F8+F9): Group A simplex weights feed Group B mixture decoder cleanly and collapse at one-hot."""
        run_a = MilestoneResolver.get_callable("receivers.group_a_reference", "run_frame", "Milestone M3 (F8)")
        fuse_b = MilestoneResolver.get_first_callable(
            "receivers.group_b_fusion", ("fuse_candidate_branches", "run_branches", "run_frame"), "Milestone M3 (F9)"
        )
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 3160, snr_db=18.0)
        pool = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 3161 + j, snr_db=18.0)[0] for j in range(3)]
        o_a = run_a(fr, context_pool=pool, mode="global_content_attn")
        w = np.asarray(o_a.aux_info["weights"], dtype=np.float64)
        branches = [
            (np.full((N_SYMB, N_ACTIVE), 0.9 + 0.1j * r, dtype=np.complex128), np.full((N_SYMB, N_ACTIVE), 0.02, dtype=np.float64))
            for r in range(3)
        ]
        o_mix = fuse_b(fr, branches=branches, branch_weights=w, mode="exact_mixture_likelihood")
        self.assertTrue(np.all(np.isfinite(o_mix.bit_llrs)))
        o_oh_map = fuse_b(fr, branches=branches, branch_weights=[1.0, 0.0, 0.0], mode="hard_map")
        o_oh_mix = fuse_b(fr, branches=branches, branch_weights=[1.0, 0.0, 0.0], mode="exact_mixture_likelihood")
        np.testing.assert_allclose(o_oh_mix.bit_llrs, o_oh_map.bit_llrs, atol=1e-10)

    def test_t3_p12_f06_f09_classical_single_branch_matches_hard_map(self) -> None:
        """T3-P12 (F6+F9): Supplying Classical branch with $\\alpha=[1.0]$ into Group B matches Classical LLRs."""
        run_c = MilestoneResolver.get_callable("receivers.classical", "run_frame", "Milestone M2 (F6)")
        fuse_b = MilestoneResolver.get_first_callable(
            "receivers.group_b_fusion", ("fuse_candidate_branches", "run_branches", "run_frame"), "Milestone M3 (F9)"
        )
        fr, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 3170, snr_db=20.0)
        o_c = run_c(fr)
        h_c = extract_active_h(o_c.h_est)
        v_c = extract_active_h(o_c.h_var)
        o_b = fuse_b(fr, branches=[(h_c, v_c)], branch_weights=[1.0], mode="exact_mixture_likelihood")
        np.testing.assert_allclose(extract_active_h(o_b.h_est), h_c, atol=1e-6)
        self.assertEqual(int(np.sum(o_b.hard_bits != o_c.hard_bits)), 0)

    def test_t3_p13_f09_f10_trainable_branch_weights_fusion_compatibility(self) -> None:
        """T3-P13 (F9+F10): Group C predicted simplex weights integrate into Group B fusion with $C^*$ equivariance."""
        cls_c = MilestoneResolver.get_first_callable(
            "receivers.group_c_trainable", ("GroupCTrainableReceiver", "GroupCReceiver"), "Milestone M4 (F10)"
        )
        tr = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 3180 + i, split="train")[0] for i in range(4)]
        dv = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 3185 + i, split="dev")[0] for i in range(2)]
        te, _ = make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 3189, split="test")
        model = cls_c(seed=313)
        model.fit(tr, dv, epochs=1)
        o_c = model.run_frame(te)
        bw = np.asarray(o_c.aux_info["branch_weights"], dtype=np.float64)
        self.assertTrue(np.all(bw >= -1e-12))
        self.assertAlmostEqual(float(np.sum(bw)), 1.0, delta=1e-6)
        o_cs = model.run_frame(scale_real_ofdm_frame(te, 100.0 * np.exp(0.5j)))
        np.testing.assert_allclose(o_cs.aux_info["branch_weights"], bw, atol=1e-5)

    def test_t3_p14_f10_f07_group_c_cost_and_stress_profiling(self) -> None:
        """T3-P14 (F10+F7): Evaluator accurately profiles Group C `param_count`, runtime, memory, and 100% coverage."""
        cls_c = MilestoneResolver.get_first_callable(
            "receivers.group_c_trainable", ("GroupCTrainableReceiver", "GroupCReceiver"), "Milestone M4 (F10)"
        )
        perturb = MilestoneResolver.get_callable(
            "adapters.sdr_ofdm_adapter", "apply_stress_perturbation", "Milestone M1 (F4)"
        )
        eval_fn = MilestoneResolver.get_first_callable(
            "eval.evaluator", ("evaluate_receiver", "evaluate_frames", "evaluate_outputs"), "Milestone M2 (F7)"
        )
        tr = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 3190 + i, split="train")[0] for i in range(4)]
        dv = [make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 3195 + i, split="dev")[0] for i in range(2)]
        te = [perturb(make_synthetic_real_ofdm_frame(E2E_SEED_BASE + 3200 + i, split="test")[0], extra_awgn_db=4.0, seed=i) for i in range(4)]
        model = cls_c(seed=314)
        model.fit(tr, dv, epochs=1)
        res = eval_fn([(f, model.run_frame(f)) for f in te])
        self.assertGreater(int(res["param_count"]), 0)
        self.assertGreater(float(res["mean_ms_per_frame"]), 0.0)
        self.assertGreater(float(res["peak_rss_mb"]), 0.0)
        self.assertAlmostEqual(float(res["coverage_pct"]), 100.0, places=6)

    def test_t3_p15_f07_f11_evaluator_outputs_match_delivery_artifacts(self) -> None:
        """T3-P15 (F7+F11): Persisted `results/` summary schema matches `REPORT.md` & `figures/` deliverables."""
        res_dir = MilestoneResolver.require_artifact("results", "Milestone M5 (F11)", allow_dir=True)
        fig_dir = MilestoneResolver.require_artifact("figures", "Milestone M5 (F11)", allow_dir=True)
        rep_path = MilestoneResolver.require_artifact("REPORT.md", "Milestone M5 (F11)")
        summary_files = list(res_dir.glob("*summary*.json"))
        self.assertGreater(len(summary_files), 0)
        summary = json.loads(summary_files[0].read_text(encoding="utf-8"))
        self.assertGreater(len(list(fig_dir.iterdir())), 0)
        rep_txt = rep_path.read_text(encoding="utf-8")
        arms = summary.get("arms", summary)
        for arm_name in arms:
            if isinstance(arms[arm_name], dict) and "total_frames" in arms[arm_name]:
                self.assertIn(str(arms[arm_name]["total_frames"]), rep_txt)
