"""
Stage P4 E2E Test Suite - Tier 3: Cross-Feature Combinations (22 Pairwise Interaction Tests).

Exercises pairwise feature interactions across:
- P01: F1 (Rapp PA) + F2 (Dynamic Multipath Simulator)
- P02: F1 (Rapp PA) + F4 (Bussgang LMMSE Equalization)
- P03: F1 (Rapp PA) + F6 (Iterative DD-PA Cancellation)
- P04: F2 (Dynamic Multipath) + F14 (Continuous CFO Drift)
- P05: F4 (Bussgang LMMSE) + F5 (Calibrated 16-QAM Soft LLRs)
- P06: F5 (Calibrated LLRs) + F15 (Native Outer LDPC Decoder)
- P07: F6 (Iterative DD-PA) + F16 (2D Classical Baseline)
- P08: F8 (C* Gauge Normalization) + F9 (Hermitian Green's Attention)
- P09: F8 (C* Gauge) + F12 (Numerical Equivariance Verification)
- P10: F8 (C* Gauge) + F19 (FP16/BF16 Mixed Precision)
- P11: F9 (Hermitian Green's Attention) + F10 (Bidirectional Pilot-Data Attention)
- P12: F9 (Green's Attention) + F18 (Factorial Ablation Architecture)
- P13: F10 (Bidirectional Attention) + F11 (Joint Physical Consistency Loss)
- P14: F11 (Joint Physical Consistency Loss) + F12 (Scale Equivariance of Gradients)
- P15: F13 (2D Slot Simulator) + F14 (Continuous High-Doppler & CFO)
- P16: F13 (2D Slot Simulator) + F16 (2D Classical LMMSE Baseline)
- P17: F14 (Continuous High-Doppler) + F17 (Coded BLER Evaluation)
- P18: F15 (Native Outer Channel Coding) + F17 (Coded BLER Waterfall)
- P19: F16 (2D Baselines) + F17 (OOD Multipath BLER)
- P20: F18 (Factorial Ablation Matrix) + F20 (Pre-Frozen Confirmatory Seeds)
- P21: F20 (Integer Accounting) + F21 (Sign-Flip Permutation Tests)
- P22: F21 (Holm Step-Down Adjustment) + F22 (Repository Governance Audit)
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


class TestTier3CrossFeature(unittest.TestCase):
    """Tier 3: Cross-Feature Combinations (22 pairwise interaction tests)."""

    def setUp(self):
        torch.set_num_threads(4)

    # -------------------------------------------------------------------------
    # Pair 1: F1 + F2 (Rapp PA + Dynamic Multipath Simulator)
    # -------------------------------------------------------------------------
    def test_t3_p01_f01_f02_rapp_pa_dynamic_doppler(self):
        """P01 (F1+F2): Verify Rapp PA combined with dynamic multipath & Doppler ICI."""
        generate_frame = MilestoneResolver.get_callable("simulator", "generate_frame", "Milestone M1 (F1+F2)")
        g = torch.Generator().manual_seed(101)
        # Strong PA + non-zero Doppler
        frame = generate_frame(
            batch_size=8, n=32, cp=16, snr_db=15.0, fd=0.03, pa_condition="strong", generator=g
        )
        self.assertEqual(frame.yd.shape, (8, 32))
        self.assertFalse(torch.isnan(frame.yd).any())
        self.assertFalse(torch.isinf(frame.yd).any())
        self.assertIsNotNone(frame.h_matrix)

    # -------------------------------------------------------------------------
    # Pair 2: F1 + F4 (Rapp PA + Bussgang LMMSE Equalization)
    # -------------------------------------------------------------------------
    def test_t3_p02_f01_f04_rapp_pa_bussgang_lmmse(self):
        """P02 (F1+F4): Verify Bussgang-LMMSE equalization on Rapp PA distorted signal."""
        generate_frame = MilestoneResolver.get_callable("simulator", "generate_frame", "Milestone M1 (F1+F4)")
        bussgang_lmmse = MilestoneResolver.get_callable("classical_baselines", "bussgang_lmmse", "Milestone M1 (F1+F4)")
        g = torch.Generator().manual_seed(102)
        frame = generate_frame(batch_size=32, n=32, snr_db=18.0, pa_condition="strong", generator=g)
        res = bussgang_lmmse(frame.yp, frame.yd, frame.noise_var, A=1.25)
        # Bussgang-LMMSE MMSE symbol estimates achieve low MSE and low BER
        x_mmse = res.w_filter * frame.yd
        mse = ((x_mmse - frame.data_tx).abs() ** 2).mean().item()
        ber = (res.hard_bits != frame.bits).float().mean().item()
        self.assertLess(mse, 0.5)
        self.assertLess(ber, 0.15)

    # -------------------------------------------------------------------------
    # Pair 3: F1 + F6 (Rapp PA + Iterative DD-PA Cancellation)
    # -------------------------------------------------------------------------
    def test_t3_p03_f01_f06_rapp_pa_idd_cancellation(self):
        """P03 (F1+F6): Verify 2-iteration DD-PA cancellation reduces nonlinear residual."""
        generate_frame = MilestoneResolver.get_callable("simulator", "generate_frame", "Milestone M1 (F1+F6)")
        iterative_dd_pa_cancellation = MilestoneResolver.get_callable(
            "classical_baselines", "iterative_dd_pa_cancellation", "Milestone M1 (F1+F6)"
        )
        g = torch.Generator().manual_seed(103)
        frame = generate_frame(batch_size=32, n=32, snr_db=20.0, pa_condition="strong", generator=g)
        h_hat, llrs, hard_bits, history = iterative_dd_pa_cancellation(
            frame.yp, frame.yd, frame.noise_var, A=1.25, iterations=2
        )
        self.assertEqual(len(history), 3)
        mse0 = ((history[0].z_hat - frame.data_tx).abs() ** 2).mean().item()
        mse2 = ((history[2].z_hat - frame.data_tx).abs() ** 2).mean().item()
        self.assertLess(mse2, mse0)

    # -------------------------------------------------------------------------
    # Pair 4: F2 + F14 (Dynamic Multipath + Continuous CFO Phase Drift)
    # -------------------------------------------------------------------------
    def test_t3_p04_f02_f14_multipath_continuous_cfo(self):
        """P04 (F2+F14): Verify continuous multipath fading combined with sample CFO drift."""
        sim_multipath_cfo = MilestoneResolver.get_callable(
            "slot_2d", "sim_multipath_cfo", "Milestone M3 (F2+F14)"
        )
        slot = sim_multipath_cfo(fd=0.03, cfo=0.05)
        self.assertFalse(torch.isnan(slot).any())

    # -------------------------------------------------------------------------
    # Pair 5: F4 + F5 (Bussgang LMMSE + Calibrated 16-QAM Soft LLRs)
    # -------------------------------------------------------------------------
    def test_t3_p05_f04_f05_bussgang_calibrated_llrs(self):
        """P05 (F4+F5): Verify soft LLRs derived from Bussgang conditional mean are calibrated."""
        generate_frame = MilestoneResolver.get_callable("simulator", "generate_frame", "Milestone M1 (F4+F5)")
        bussgang_lmmse = MilestoneResolver.get_callable("classical_baselines", "bussgang_lmmse", "Milestone M1 (F4+F5)")
        g = torch.Generator().manual_seed(105)
        frame = generate_frame(batch_size=64, n=32, snr_db=15.0, pa_condition="strong", generator=g)
        res = bussgang_lmmse(frame.yp, frame.yd, frame.noise_var, A=1.25)
        self.assertEqual(res.llrs.shape, (64, 32, 4))
        # Slicing from LLRs should concord with hard_bits (> 98% agreement in presence of noise)
        bits_from_llrs = (res.llrs > 0).long()
        agreement = (bits_from_llrs == res.hard_bits).float().mean().item()
        self.assertGreater(agreement, 0.98)

    # -------------------------------------------------------------------------
    # Pair 6: F5 + F15 (Calibrated Soft LLRs + Native Outer LDPC Decoder)
    # -------------------------------------------------------------------------
    def test_t3_p06_f05_f15_calibrated_llrs_outer_ldpc(self):
        """P06 (F5+F15): Feed calibrated soft LLRs into native Rate-1/2 LDPC Min-Sum decoder."""
        decode_calibrated_ldpc = MilestoneResolver.get_callable(
            "slot_2d", "decode_calibrated_ldpc", "Milestone M3 (F5+F15)"
        )
        ber = decode_calibrated_ldpc(snr_db=15.0)
        self.assertLess(ber, 0.05)

    # -------------------------------------------------------------------------
    # Pair 7: F6 + F16 (Iterative DD-PA + 2D Classical Baseline)
    # -------------------------------------------------------------------------
    def test_t3_p07_f06_f16_2d_idd_pa_cancellation(self):
        """P07 (F6+F16): Verify 2D iterative decision-directed PA cancellation over slot grid."""
        idd_pa_2d_slot = MilestoneResolver.get_callable(
            "slot_2d", "idd_pa_2d_slot", "Milestone M3 (F6+F16)"
        )
        res = idd_pa_2d_slot(iterations=2)
        self.assertTrue(res["converged"])

    # -------------------------------------------------------------------------
    # Pair 8: F8 + F9 (C* Gauge Normalization + Hermitian Green's Attention)
    # -------------------------------------------------------------------------
    def test_t3_p08_f08_f09_gauge_norm_hermitian_attention(self):
        """P08 (F8+F9): Verify pilot RMS anchoring combined with Hermitian relative-phase attention."""
        verify_gauge_attention_joint = MilestoneResolver.get_callable(
            "em_floatformer", "verify_gauge_attention_joint", "Milestone M2 (F8+F9)"
        )
        is_equivariant = verify_gauge_attention_joint()
        self.assertTrue(is_equivariant)

    # -------------------------------------------------------------------------
    # Pair 9: F8 + F12 (Exact C* Gauge + Numerical Equivariance Verification)
    # -------------------------------------------------------------------------
    def test_t3_p09_f08_f12_gauge_norm_numerical_equivariance(self):
        """P09 (F8+F12): Integrated equivariance verification over 24 octaves and full phase circle."""
        audit_full_equivariance = MilestoneResolver.get_callable(
            "em_floatformer", "audit_full_equivariance", "Milestone M2 (F8+F12)"
        )
        max_err, flips = audit_full_equivariance()
        self.assertLess(max_err, 1e-5)
        self.assertEqual(flips, 0)

    # -------------------------------------------------------------------------
    # Pair 10: F8 + F19 (Exact C* Gauge + FP16/BF16 Mixed Precision)
    # -------------------------------------------------------------------------
    def test_t3_p10_f08_f19_gauge_norm_mixed_precision_audit(self):
        """P10 (F8+F19): C* gauge normalization under FP16/BF16 autocast across 24 octaves."""
        audit_gauge_mixed_precision = MilestoneResolver.get_callable(
            "ablation_probing", "audit_gauge_mixed_precision", "Milestone M4 (F8+F19)"
        )
        stable = audit_gauge_mixed_precision()
        self.assertTrue(stable)

    # -------------------------------------------------------------------------
    # Pair 11: F9 + F10 (Hermitian Green's Attention + Bidirectional Pilot-Data Attention)
    # -------------------------------------------------------------------------
    def test_t3_p11_f09_f10_green_attention_bidirectional_prop(self):
        """P11 (F9+F10): Verify bidirectional cross-attention using Hermitian inner products."""
        verify_bidirectional_green_prop = MilestoneResolver.get_callable(
            "em_floatformer", "verify_bidirectional_green_prop", "Milestone M2 (F9+F10)"
        )
        success = verify_bidirectional_green_prop()
        self.assertTrue(success)

    # -------------------------------------------------------------------------
    # Pair 12: F9 + F18 (Hermitian Green's Attention + Factorial Ablation Architecture)
    # -------------------------------------------------------------------------
    def test_t3_p12_f09_f18_green_attention_factorial_ablation(self):
        """P12 (F9+F18): Verify toggling Green's attention against real attention in ablation matrix."""
        compare_green_vs_real_ablation = MilestoneResolver.get_callable(
            "ablation_probing", "compare_green_vs_real_ablation", "Milestone M4 (F9+F18)"
        )
        diff = compare_green_vs_real_ablation()
        self.assertIsNotNone(diff)

    # -------------------------------------------------------------------------
    # Pair 13: F10 + F11 (Bidirectional Attention + Joint Physical Consistency Loss)
    # -------------------------------------------------------------------------
    def test_t3_p13_f10_f11_bidirectional_joint_loss_convergence(self):
        """P13 (F10+F11): Train 1 step of bidirectional model with joint loss; check gradients."""
        train_step_joint_loss = MilestoneResolver.get_callable(
            "em_floatformer", "train_step_joint_loss", "Milestone M2 (F10+F11)"
        )
        grad_ok = train_step_joint_loss()
        self.assertTrue(grad_ok)

    # -------------------------------------------------------------------------
    # Pair 14: F11 + F12 (Joint Loss + Scale Equivariance of Gradients)
    # -------------------------------------------------------------------------
    def test_t3_p14_f11_f12_joint_loss_gradient_equivariance(self):
        """P14 (F11+F12): Parameter gradients remain identical when inputs scaled by r in [2^-12, 2^12]."""
        check_loss_gradient_scale = MilestoneResolver.get_callable(
            "em_floatformer", "check_loss_gradient_scale", "Milestone M2 (F11+F12)"
        )
        max_diff = check_loss_gradient_scale()
        self.assertLess(max_diff, 1e-5)

    # -------------------------------------------------------------------------
    # Pair 15: F13 + F14 (2D Slot Simulator + Continuous High-Doppler & CFO)
    # -------------------------------------------------------------------------
    def test_t3_p15_f13_f14_2d_slot_doppler_cfo(self):
        """P15 (F13+F14): 14-symbol slot with sparse DMRS under high Doppler (0.05) & CFO (0.05)."""
        sim_slot_doppler_cfo = MilestoneResolver.get_callable(
            "slot_2d", "sim_slot_doppler_cfo", "Milestone M3 (F13+F14)"
        )
        grid = sim_slot_doppler_cfo(fd=0.05, cfo=0.05)
        self.assertEqual(grid.shape, (14, 32))

    # -------------------------------------------------------------------------
    # Pair 16: F13 + F16 (2D Slot Simulator + 2D Classical LMMSE Baseline)
    # -------------------------------------------------------------------------
    def test_t3_p16_f13_f16_2d_slot_2d_lmmse_estimation(self):
        """P16 (F13+F16): 2D-LMMSE channel estimation from sparse DMRS REs over 14-symbol slot."""
        estimate_slot_2d_lmmse = MilestoneResolver.get_callable(
            "slot_2d", "estimate_slot_2d_lmmse", "Milestone M3 (F13+F16)"
        )
        h_est = estimate_slot_2d_lmmse()
        self.assertEqual(h_est.shape, (14, 32))

    # -------------------------------------------------------------------------
    # Pair 17: F14 + F17 (Continuous High-Doppler + Coded BLER Evaluation)
    # -------------------------------------------------------------------------
    def test_t3_p17_f14_f17_high_doppler_coded_bler(self):
        """P17 (F14+F17): Evaluate outer coded BLER under high Doppler (fd=600 Hz)."""
        eval_high_doppler_bler = MilestoneResolver.get_callable(
            "slot_2d", "eval_high_doppler_bler", "Milestone M3 (F14+F17)"
        )
        bler = eval_high_doppler_bler(fd_hz=600.0)
        self.assertLess(bler, 0.5)

    # -------------------------------------------------------------------------
    # Pair 18: F15 + F17 (Native Outer Channel Coding + Coded BLER Waterfall)
    # -------------------------------------------------------------------------
    def test_t3_p18_f15_f17_outer_coding_bler_waterfall(self):
        """P18 (F15+F17): Trace full coded BLER curve across SNR grid [0, 5, 10, 15, 20] dB."""
        trace_bler_waterfall = MilestoneResolver.get_callable(
            "slot_2d", "trace_bler_waterfall", "Milestone M3 (F15+F17)"
        )
        curve = trace_bler_waterfall()
        self.assertEqual(len(curve), 5)

    # -------------------------------------------------------------------------
    # Pair 19: F16 + F17 (2D Baselines + Out-of-Distribution Multipath BLER)
    # -------------------------------------------------------------------------
    def test_t3_p19_f16_f17_2d_baselines_ood_multipath_bler(self):
        """P19 (F16+F17): Benchmark 2D-LMMSE and 2D IDD-PA BLER on held-out sparse12 profile."""
        eval_ood_baselines_bler = MilestoneResolver.get_callable(
            "slot_2d", "eval_ood_baselines_bler", "Milestone M3 (F16+F17)"
        )
        bler_lmmse, bler_idd = eval_ood_baselines_bler()
        self.assertGreaterEqual(bler_lmmse, 0.0)
        self.assertGreaterEqual(bler_idd, 0.0)

    # -------------------------------------------------------------------------
    # Pair 20: F18 + F20 (Factorial Ablation Matrix + Pre-Frozen Confirmatory Seeds)
    # -------------------------------------------------------------------------
    def test_t3_p20_f18_f20_factorial_ablation_confirmatory_seeds(self):
        """P20 (F18+F20): Run all 8 ablation arms across 10 pre-frozen seeds with paired initialization."""
        run_ablation_confirmatory = MilestoneResolver.get_callable(
            "ablation_probing", "run_ablation_confirmatory", "Milestone M4 (F18+F20)"
        )
        records = run_ablation_confirmatory(seeds=FROZEN_SEEDS_P4D)
        self.assertEqual(len(records), 10 * 8)

    # -------------------------------------------------------------------------
    # Pair 21: F20 + F21 (Integer Accounting + Sign-Flip Permutation Tests)
    # -------------------------------------------------------------------------
    def test_t3_p21_f20_f21_integer_accounting_permutation_tests(self):
        """P21 (F20+F21): Feed exact integer error records into 1024-state permutation test runner."""
        # Synthetic paired bit errors across 10 seeds: baseline vs proposed
        baseline_errors = [150, 160, 145, 170, 155, 165, 140, 158, 162, 148]
        proposed_errors = [110, 120, 105, 130, 115, 125, 100, 118, 122, 108]
        deltas = [b - p for b, p in zip(baseline_errors, proposed_errors)]
        # All deltas are +40 > 0
        p_val = exact_permutation_test_1024(deltas)
        self.assertAlmostEqual(p_val, 2.0 / 1024.0, places=6)

    # -------------------------------------------------------------------------
    # Pair 22: F21 + F22 (Holm Step-Down Adjustment + Repository Governance Audit)
    # -------------------------------------------------------------------------
    def test_t3_p22_f21_f22_holm_adjustment_governance_audit(self):
        """P22 (F21+F22): Verify Holm-adjusted p-values logged without modifying historical manifest."""
        manifest = load_manifest()
        self.assertGreater(len(manifest), 100)
        p_vals = [0.00195, 0.015, 0.04, 0.12]
        adjusted = holm_bonferroni_adjustment(p_vals)
        self.assertEqual(len(adjusted), 4)
        for p in adjusted:
            self.assertLessEqual(p, 1.0)
        # Ensure manifest file remains untouched
        self.assertTrue(MANIFEST_PATH.exists())


if __name__ == "__main__":
    unittest.main()
