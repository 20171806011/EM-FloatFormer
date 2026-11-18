"""
Stage P4 E2E Test Suite - Tier 4: Real-World Workload Scenarios (11 Full E2E Scenarios).

Covers mission-profile communication workloads combining simulation, channel distortion,
transceiver processing, soft decoding, and statistical auditing:
- S01: High-Speed Train Channel (HST-Doppler with Severe PA Distortion)
- S02: Urban Macrocell with Severe Multipath Delay Spread (long12 & sparse12)
- S03: Extreme Local Oscillator Drift & Continuous Carrier Frequency Offset (CFO)
- S04: Deep Shadowing & 24-Octave Dynamic Range Swing under Mixed Precision
- S05: Ultra-Sparse Pilot Deployment (<= 5% DMRS Overhead)
- S06: Multi-Impairment Simultaneous Stress Workload (PA + Doppler + CFO + Delay Spread)
- S07: Out-of-Distribution Channel Delay Profile Generalization
- S08: Confirmatory 10-Seed Pre-Frozen Replication Workload (Exact Integer Accounting)
- S09: Full 2^3 Factorial Ablation Matrix Evaluation Workload (Yates' Contrasts)
- S10: Full-Precision vs Mixed-Precision Numerical Audit Workload (FP32/BF16/FP16)
- S11: End-to-End Mission-Critical Coded Transmission Workload (10^5 Bits, LDPC Decoding)
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from tests.e2e.harness import (
    FROZEN_SEEDS_P4D,
    MANIFEST_PATH,
    PROJECT_ROOT,
    MilestoneResolver,
    exact_permutation_test_1024,
    holm_bonferroni_adjustment,
    load_manifest,
)


class TestTier4RealWorkloads(unittest.TestCase):
    """Tier 4: Real-World Workload Scenarios (11 comprehensive end-to-end scenarios)."""

    def setUp(self):
        torch.set_num_threads(4)

    # -------------------------------------------------------------------------
    # Scenario 1: High-Speed Train Workload
    # -------------------------------------------------------------------------
    def test_t4_s01_high_speed_train_workload(self):
        """S01: HST channel (fd=500 Hz), Rapp PA (A=1.25), sparse DMRS, outer LDPC."""
        run_hst_workload = MilestoneResolver.get_callable(
            "slot_2d", "run_hst_workload", "Milestone M3 (S01)"
        )
        res = run_hst_workload(speed_kmh=300.0, fc=3.0e9, pa_A=1.25, snr_db=15.0)
        self.assertLess(res["bler"], 0.05)
        self.assertEqual(res["nonfinite_count"], 0)
        self.assertIsInstance(res["bit_errors"], int)

    # -------------------------------------------------------------------------
    # Scenario 2: Urban Macrocell with Severe Multipath Delay Spread
    # -------------------------------------------------------------------------
    def test_t4_s02_urban_macro_dense_multipath_workload(self):
        """S02: Urban Macro (N_sc=64, 12-tap delay spread long12, mild PA A=2.25, 20 slots)."""
        run_urban_macro_workload = MilestoneResolver.get_callable(
            "slot_2d", "run_urban_macro_workload", "Milestone M3 (S02)"
        )
        res = run_urban_macro_workload(n_sc=64, profile="long12", pa_A=2.25, num_slots=20, snr_db=18.0)
        self.assertLess(res["nmse"], 0.05)
        self.assertLess(res["bler"], 0.01)

    # -------------------------------------------------------------------------
    # Scenario 3: Extreme Local Oscillator Drift & Continuous CFO
    # -------------------------------------------------------------------------
    def test_t4_s03_extreme_cfo_drift_workload(self):
        """S03: Normalized CFO epsilon=0.06 across 14 OFDM symbols with CP correlation."""
        run_cfo_drift_workload = MilestoneResolver.get_callable(
            "slot_2d", "run_cfo_drift_workload", "Milestone M3 (S03)"
        )
        res = run_cfo_drift_workload(cfo=0.06, n_symb=14, snr_db=20.0)
        self.assertLess(res["residual_phase_deg"], 3.0)
        self.assertEqual(res["quadrant_flips"], 0)

    # -------------------------------------------------------------------------
    # Scenario 4: Deep Shadowing & 24-Octave Dynamic Range Swing
    # -------------------------------------------------------------------------
    def test_t4_s04_dynamic_range_shadowing_swing_workload(self):
        """S04: Power swing r in {2^-12, 2^-8, 2^-4, 2^0, 2^4, 2^8, 2^12} under BF16."""
        run_shadowing_swing = MilestoneResolver.get_callable(
            "ablation_probing", "run_shadowing_swing", "Milestone M4 (S04)"
        )
        res = run_shadowing_swing(scales=[2.0 ** p for p in [-12, -8, -4, 0, 4, 8, 12]], dtype=torch.bfloat16)
        self.assertEqual(res["flip_rate"], 0.0)
        self.assertLess(res["max_logit_error"], 1e-4)
        self.assertEqual(res["nan_count"], 0)

    # -------------------------------------------------------------------------
    # Scenario 5: Ultra-Sparse Pilot Deployment (<= 5% Overhead)
    # -------------------------------------------------------------------------
    def test_t4_s05_ultra_sparse_dmrs_deployment_workload(self):
        """S05: Single DMRS symbol at symbol 2 with comb-4 pattern (overhead ~1.78%)."""
        run_ultra_sparse_workload = MilestoneResolver.get_callable(
            "slot_2d", "run_ultra_sparse_workload", "Milestone M3 (S05)"
        )
        res = run_ultra_sparse_workload(comb=4, pilot_symbol=2)
        self.assertTrue(res["stable"])
        self.assertGreaterEqual(res["gain_over_ls_db"], 4.0)

    # -------------------------------------------------------------------------
    # Scenario 6: Multi-Impairment Simultaneous Stress Workload
    # -------------------------------------------------------------------------
    def test_t4_s06_multi_impairment_stress_workload(self):
        """S06: Concurrent Rapp AM-AM (1.25), AM-PM (0.5), Doppler (0.03), CFO (0.04), delay spread."""
        run_multi_impairment_stress = MilestoneResolver.get_callable(
            "slot_2d", "run_multi_impairment_stress", "Milestone M3 (S06)"
        )
        res = run_multi_impairment_stress()
        self.assertTrue(res["outputs_finite"])
        self.assertLess(res["p_value"], 0.05)

    # -------------------------------------------------------------------------
    # Scenario 7: Out-of-Distribution Channel Delay Profile Generalization
    # -------------------------------------------------------------------------
    def test_t4_s07_ood_channel_generalization_workload(self):
        """S07: Evaluate on unseen 16-tap exponentially decaying channel."""
        run_ood_generalization = MilestoneResolver.get_callable(
            "slot_2d", "run_ood_generalization", "Milestone M3 (S07)"
        )
        res = run_ood_generalization()
        self.assertFalse(res["shortcut_collapse"])
        self.assertLessEqual(res["ber_gap"], 0.005)

    # -------------------------------------------------------------------------
    # Scenario 8: Confirmatory 10-Seed Pre-Frozen Replication Workload
    # -------------------------------------------------------------------------
    def test_t4_s08_confirmatory_ten_seed_replication_workload(self):
        """S08: Full replication protocol across 10 pre-frozen seeds 74001–74010."""
        run_confirmatory_protocol = MilestoneResolver.get_callable(
            "confirmatory_protocol", "run_confirmatory_protocol", "Milestone M4 (S08)"
        )
        res = run_confirmatory_protocol(seeds=FROZEN_SEEDS_P4D)
        self.assertEqual(len(res["records"]), 10)
        self.assertIsInstance(res["total_bits"], int)
        self.assertLessEqual(res["exact_p_value"], 1.0)

    # -------------------------------------------------------------------------
    # Scenario 9: Full 2^3 Factorial Ablation Matrix Evaluation Workload
    # -------------------------------------------------------------------------
    def test_t4_s09_factorial_ablation_matrix_workload(self):
        """S09: Evaluate all 8 parameter-matched arms and decompose via Yates' matrix."""
        run_full_factorial_evaluation = MilestoneResolver.get_callable(
            "ablation_probing", "run_full_factorial_evaluation", "Milestone M4 (S09)"
        )
        res = run_full_factorial_evaluation(snr_db=10.0)
        self.assertEqual(len(res["arm_scores"]), 8)
        self.assertTrue(res["decomposition_valid"])
        self.assertGreater(res["main_effects"]["F1_gauge"], 0.0)

    # -------------------------------------------------------------------------
    # Scenario 10: Full-Precision vs Mixed-Precision Numerical Audit Workload
    # -------------------------------------------------------------------------
    def test_t4_s10_mixed_precision_audit_workload(self):
        """S10: 100 slots through FP32, BF16, and FP16 across SNR grid [0, 5, 10, 15, 20] dB."""
        run_mixed_precision_audit = MilestoneResolver.get_callable(
            "ablation_probing", "run_mixed_precision_audit", "Milestone M4 (S10)"
        )
        res = run_mixed_precision_audit(num_slots=100, snr_grid=[0, 5, 10, 15, 20])
        self.assertEqual(res["nan_count"], 0)
        self.assertLess(res["flip_rate"], 0.0005)

    # -------------------------------------------------------------------------
    # Scenario 11: End-to-End Mission-Critical Coded Transmission Workload
    # -------------------------------------------------------------------------
    def test_t4_s11_mission_critical_coded_transmission_workload(self):
        """S11: Complete mission link: 10^5 bits over 50 slots under 16-QAM, Rapp PA, Doppler, LDPC."""
        run_mission_link = MilestoneResolver.get_callable(
            "slot_2d", "run_mission_link", "Milestone M3 (S11)"
        )
        res = run_mission_link(total_bits=100_000, num_slots=50, snr_db=15.0)
        self.assertLess(res["bler"], 0.01)
        self.assertIsInstance(res["total_bit_errors"], int)
        self.assertEqual(res["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
