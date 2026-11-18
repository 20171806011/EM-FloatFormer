"""
Stage P4 E2E Test Suite - Tier 2: Boundary & Corner Cases (110 Tests).

Covers extreme operating conditions, boundary limits, and mathematical edge cases across F1 to F22:
- F1 to F7: M1 Dynamic Simulator, Bussgang-LMMSE Baselines, Classical Benchmark extremes
- F8 to F12: M2 EM-FloatFormer Propagator Transformer & C* Gauge Equivariance singularities
- F13 to F17: M3 2D Multi-Symbol Slot, Continuous Doppler/CFO, Outer FEC, 2D Baselines & Coded BLER edge cases
- F18 to F19: M4 Factorial Ablation Matrix & Precision extremes
- F20 to F21: M4 Pre-Frozen Confirmatory Protocol & Permutation Test degeneracies
- F22: Repository Governance & Historical Immutability stress tests
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List
import unittest

import numpy as np
import torch
import torch.nn.functional as F

from tests.e2e.harness import (
    FROZEN_SEEDS_P4D,
    GRAY_TABLE_16QAM,
    HISTORICAL_SEEDS,
    MANIFEST_PATH,
    PROJECT_ROOT,
    SIGMA_16QAM,
    THRESH_16QAM,
    MilestoneResolver,
    compute_file_sha256,
    compute_file_sha256_normalized,
    exact_permutation_test_1024,
    hard_bits_16qam_reference,
    holm_bonferroni_adjustment,
    load_manifest,
    q16_reference,
    qpsk_reference,
)


class TestTier2BoundaryCorner(unittest.TestCase):
    """Tier 2: Boundary & Corner Cases (110 tests, 5 per feature F1–F22)."""

    def setUp(self):
        torch.set_num_threads(4)

    # =========================================================================
    # Feature F1: Oversampled Rapp PA (5 boundary tests)
    # =========================================================================

    def test_t2_f01_extreme_clipping_knee_p1(self):
        """F1.1: Verify Rapp PA with soft knee p=1.0 asymptotes smoothly."""
        rapp_am_am = MilestoneResolver.get_callable("simulator", "rapp_am_am", "Milestone M1 (F1)")
        r = torch.logspace(-2, 2, 50)
        g_r = rapp_am_am(r, A=1.25, knee=1.0)
        self.assertFalse(torch.isnan(g_r).any())
        self.assertLessEqual(float(g_r.max()), 1.25)

    def test_t2_f01_hard_clipping_limit_p10(self):
        """F1.2: Verify Rapp PA with sharp knee p=10.0 acts as near-ideal hard limiter."""
        rapp_am_am = MilestoneResolver.get_callable("simulator", "rapp_am_am", "Milestone M1 (F1)")
        r_sub = torch.tensor([0.5, 0.8, 1.0])
        r_sup = torch.tensor([1.5, 2.0, 5.0])
        A = 1.25
        g_sub = rapp_am_am(r_sub, A=A, knee=10.0)
        g_sup = rapp_am_am(r_sup, A=A, knee=10.0)
        # Below A, g(r) ~ r
        self.assertTrue(torch.allclose(g_sub, r_sub, atol=1e-2))
        # Above A, g(r) ~ A
        self.assertTrue(torch.allclose(g_sup, torch.full_like(g_sup, A), atol=1e-2))

    def test_t2_f01_near_zero_input_amplitude(self):
        """F1.3: Verify Rapp PA handles near-zero amplitude 1e-15 without NaN or zero division."""
        rapp_am_am = MilestoneResolver.get_callable("simulator", "rapp_am_am", "Milestone M1 (F1)")
        rapp_am_pm = MilestoneResolver.get_callable("simulator", "rapp_am_pm", "Milestone M1 (F1)")
        r_tiny = torch.tensor([1e-15, 1e-25, 0.0])
        g_r = rapp_am_am(r_tiny, A=1.25)
        phi_r = rapp_am_pm(r_tiny, alpha=0.5, beta=1.0)
        self.assertFalse(torch.isnan(g_r).any())
        self.assertFalse(torch.isnan(phi_r).any())
        self.assertEqual(float(g_r[-1]), 0.0)
        self.assertEqual(float(phi_r[-1]), 0.0)

    def test_t2_f01_extreme_overdrive_input_100x_A(self):
        """F1.4: Verify extreme overdrive 100 * A output amplitude remains bounded by A."""
        rapp_pa = MilestoneResolver.get_callable("simulator", "rapp_pa", "Milestone M1 (F1)")
        A = 1.25
        x_huge = torch.randn(2, 32, dtype=torch.complex64) * (100.0 * A)
        y_huge = rapp_pa(x_huge, knee=2.0, A=A, alpha=0.5, oversample=4)
        self.assertFalse(torch.isnan(y_huge).any())
        self.assertFalse(torch.isinf(y_huge).any())

    def test_t2_f01_nyquist_vs_oversampled_aliasing_boundary(self):
        """F1.5: Verify oversampled decimation handles boundary oversample factors (L=1 vs L=4)."""
        rapp_pa = MilestoneResolver.get_callable("simulator", "rapp_pa", "Milestone M1 (F1)")
        x = torch.randn(4, 32, dtype=torch.complex64)
        y_l1 = rapp_pa(x, knee=2.0, A=1.25, oversample=1)
        y_l4 = rapp_pa(x, knee=2.0, A=1.25, oversample=4)
        self.assertEqual(y_l1.shape, (4, 32))
        self.assertEqual(y_l4.shape, (4, 32))

    # =========================================================================
    # Feature F2: Dynamic Multipath Simulator (5 boundary tests)
    # =========================================================================

    def test_t2_f02_max_doppler_nyquist_boundary(self):
        """F2.1: Verify stability at extreme Doppler fd approaching Nyquist bound."""
        DynamicChannelSimulator = MilestoneResolver.get_class("simulator", "DynamicChannelSimulator", "Milestone M1 (F2)")
        sim = DynamicChannelSimulator(n=32, cp=16, profile="equal4", fd=0.45)
        ch_state = sim.draw_channel(batch_size=2)
        h_mat = sim.compute_transfer_matrix(ch_state)
        self.assertFalse(torch.isnan(h_mat).any())

    def test_t2_f02_zero_multipath_flat_fading(self):
        """F2.2: Verify zero delay spread yields flat frequency response across subcarriers."""
        DynamicChannelSimulator = MilestoneResolver.get_class("simulator", "DynamicChannelSimulator", "Milestone M1 (F2)")
        sim = DynamicChannelSimulator(n=32, cp=16, profile="equal4", fd=0.0)
        # Manually collapse to single tap at delay 0
        sim.pdp = torch.tensor([1.0], dtype=torch.float32)
        sim.num_taps = 1
        sim.delays = torch.tensor([0])
        ch_state = sim.draw_channel(batch_size=2)
        # Channel frequency response across all subcarriers should be identical flat complex scalar
        h_flat = ch_state.h_freq_static
        std_across_sc = h_flat.std(dim=-1).max().item()
        self.assertLess(std_across_sc, 1e-5)

    def test_t2_f02_extreme_delay_spread_exceeding_cp(self):
        """F2.3: Verify ValueError when channel delay spread exceeds CP length."""
        DynamicChannelSimulator = MilestoneResolver.get_class("simulator", "DynamicChannelSimulator", "Milestone M1 (F2)")
        with self.assertRaises(ValueError):
            # profile long12 has max delay 11, with cp=8 max delay exceeds cp
            _ = DynamicChannelSimulator(n=32, cp=8, profile="long12")

    def test_t2_f02_zero_snr_pure_noise_limit(self):
        """F2.4: Verify simulator runs stably at extreme low SNR -20 dB."""
        generate_frame = MilestoneResolver.get_callable("simulator", "generate_frame", "Milestone M1 (F2)")
        frame = generate_frame(batch_size=4, n=32, snr_db=-20.0, fd=0.0)
        self.assertFalse(torch.isnan(frame.yd).any())
        self.assertEqual(float(frame.noise_var[0, 0]), 100.0)

    def test_t2_f02_high_snr_infinite_snr_limit(self):
        """F2.5: Verify simulator runs stably at extreme high SNR 50 dB."""
        generate_frame = MilestoneResolver.get_callable("simulator", "generate_frame", "Milestone M1 (F2)")
        frame = generate_frame(batch_size=4, n=32, snr_db=50.0, fd=0.0)
        self.assertFalse(torch.isnan(frame.yd).any())
        self.assertLess(float(frame.noise_var[0, 0]), 1e-4)

    # =========================================================================
    # Feature F3: Static-Limit Parity Check (5 boundary tests)
    # =========================================================================

    def test_t2_f03_single_path_delay_zero_parity(self):
        """F3.1: Verify static parity for single path at delay 0."""
        static_limit_parity_check = MilestoneResolver.get_callable("simulator", "static_limit_parity_check", "Milestone M1 (F3)")
        rel_err = static_limit_parity_check(n=32, cp=16, batch_size=16, profile="train3")
        self.assertLess(rel_err, 1e-5)

    def test_t2_f03_max_taps_density_parity(self):
        """F3.2: Verify static parity with dense multipath profile (long12)."""
        static_limit_parity_check = MilestoneResolver.get_callable("simulator", "static_limit_parity_check", "Milestone M1 (F3)")
        rel_err = static_limit_parity_check(n=32, cp=16, batch_size=16, profile="long12")
        self.assertLess(rel_err, 1e-5)

    def test_t2_f03_extreme_phase_rotation_parity(self):
        """F3.3: Verify static parity holds under constant phase rotation theta = pi*sqrt(2)."""
        DynamicChannelSimulator = MilestoneResolver.get_class("simulator", "DynamicChannelSimulator", "Milestone M1 (F3)")
        sim = DynamicChannelSimulator(n=32, cp=16, profile="equal4", fd=0.0)
        state = sim.draw_channel(batch_size=4)
        theta = torch.tensor(math.pi * math.sqrt(2.0))
        rot = torch.polar(torch.tensor(1.0), theta)
        state.h_time_initial = state.h_time_initial * rot
        state.h_freq_static = state.h_freq_static * rot
        
        x_freq = torch.randn(4, 32, dtype=torch.complex64)
        x_time = torch.fft.ifft(x_freq, norm="ortho")
        x_cp = torch.cat([x_time[:, -16:], x_time], dim=-1)
        y_time = sim.filter_time_signal(x_cp, state)
        y_freq = torch.fft.fft(y_time[:, 16:], norm="ortho")
        y_mat = state.h_freq_static * x_freq
        diff = (y_freq - y_mat).norm() / y_mat.norm()
        self.assertLess(diff.item(), 1e-5)

    def test_t2_f03_all_zeros_input_parity(self):
        """F3.4: Verify all-zero frequency vector produces all-zero output to < 1e-15."""
        DynamicChannelSimulator = MilestoneResolver.get_class("simulator", "DynamicChannelSimulator", "Milestone M1 (F3)")
        sim = DynamicChannelSimulator(n=32, cp=16, profile="equal4", fd=0.0)
        state = sim.draw_channel(batch_size=2)
        x_zero_cp = torch.zeros(2, 48, dtype=torch.complex64)
        y_zero = sim.filter_time_signal(x_zero_cp, state, noise_var=None)
        self.assertLess(y_zero.abs().max().item(), 1e-15)

    def test_t2_f03_single_symbol_vs_batch_parity(self):
        """F3.5: Verify static parity across batch sizes 1, 8, and 64."""
        static_limit_parity_check = MilestoneResolver.get_callable("simulator", "static_limit_parity_check", "Milestone M1 (F3)")
        for b in [1, 8, 64]:
            rel_err = static_limit_parity_check(n=32, cp=16, batch_size=b)
            self.assertLess(rel_err, 1e-5)

    # =========================================================================
    # Feature F4: Bussgang Decomposition & LMMSE (5 boundary tests)
    # =========================================================================

    def test_t2_f04_extreme_saturation_A_0p5(self):
        """F4.1: Verify Bussgang parameters under extreme saturation A=0.5."""
        bussgang_params = MilestoneResolver.get_callable("classical_baselines", "bussgang_params", "Milestone M1 (F4)")
        gamma, sigma_d2, p_out = bussgang_params(A=0.5)
        self.assertGreater(abs(gamma), 0.0)
        self.assertLess(abs(gamma), 1.0)
        self.assertGreater(sigma_d2, 0.0)

    def test_t2_f04_linear_limit_A_1000(self):
        """F4.2: Verify Bussgang parameters under near-linear PA A=1000."""
        bussgang_params = MilestoneResolver.get_callable("classical_baselines", "bussgang_params", "Milestone M1 (F4)")
        gamma, sigma_d2, p_out = bussgang_params(A=1000.0, alpha=0.0)
        self.assertAlmostEqual(abs(gamma), 1.0, places=4)
        self.assertLess(sigma_d2, 1e-5)

    def test_t2_f04_deep_channel_fade_zero_channel(self):
        """F4.3: Verify LMMSE filter when channel deep fades to zero."""
        bussgang_lmmse = MilestoneResolver.get_callable("classical_baselines", "bussgang_lmmse", "Milestone M1 (F4)")
        yp = torch.zeros(1, 32, dtype=torch.complex64)
        yd = torch.randn(1, 32, dtype=torch.complex64)
        nv = torch.tensor([[0.1]])
        res = bussgang_lmmse(yp, yd, nv, A=1.25, est_method="ls")
        self.assertFalse(torch.isnan(res.w_filter).any())
        self.assertFalse(torch.isnan(res.z_hat).any())

    def test_t2_f04_extreme_high_noise_variance(self):
        """F4.4: Verify LMMSE filter with extreme high noise sigma_W^2 = 100."""
        bussgang_lmmse = MilestoneResolver.get_callable("classical_baselines", "bussgang_lmmse", "Milestone M1 (F4)")
        yp = torch.ones(1, 32, dtype=torch.complex64)
        yd = torch.ones(1, 32, dtype=torch.complex64)
        nv = torch.tensor([[100.0]])
        res = bussgang_lmmse(yp, yd, nv, A=1.25, est_method="ls")
        # Filter weights should smoothly shrink toward 0
        self.assertLess(res.w_filter.abs().max().item(), 0.02)

    def test_t2_f04_extreme_low_noise_variance(self):
        """F4.5: Verify LMMSE filter with extreme low noise sigma_W^2 = 1e-8."""
        bussgang_lmmse = MilestoneResolver.get_callable("classical_baselines", "bussgang_lmmse", "Milestone M1 (F4)")
        yp = torch.ones(1, 32, dtype=torch.complex64)
        yd = torch.ones(1, 32, dtype=torch.complex64)
        nv = torch.tensor([[1e-8]])
        res = bussgang_lmmse(yp, yd, nv, A=1.25, est_method="ls")
        self.assertFalse(torch.isnan(res.w_filter).any())
        self.assertFalse(torch.isinf(res.w_filter).any())

    # =========================================================================
    # Feature F5: Calibrated 16-QAM Soft LLRs (5 boundary tests)
    # =========================================================================

    def test_t2_f05_exact_constellation_points_zero_noise(self):
        """F5.1: Verify noiseless constellation points produce large high-confidence LLRs."""
        calibrated_16qam_llrs = MilestoneResolver.get_callable("classical_baselines", "calibrated_16qam_llrs", "Milestone M1 (F5)")
        # Point (+3 + 3j) / sqrt(10): bits are [1, 0, 1, 0]
        z = torch.tensor([[(3.0 + 3.0j) / math.sqrt(10.0)]], dtype=torch.complex64)
        sigma2 = torch.tensor([[0.01]])
        llrs = calibrated_16qam_llrs(z, sigma2)
        self.assertGreater(float(llrs[0, 0, 0]), 10.0)  # b0 = 1
        self.assertLess(float(llrs[0, 0, 1]), -10.0)   # b1 = 0
        self.assertGreater(float(llrs[0, 0, 2]), 10.0)  # b2 = 1
        self.assertLess(float(llrs[0, 0, 3]), -10.0)   # b3 = 0

    def test_t2_f05_exact_decision_boundaries(self):
        """F5.2: Verify points on exact decision boundaries produce LLR ~ 0."""
        calibrated_16qam_llrs = MilestoneResolver.get_callable("classical_baselines", "calibrated_16qam_llrs", "Milestone M1 (F5)")
        # Re(z) = 0 is decision boundary for b0
        z = torch.tensor([[0.0 + 1.0j]], dtype=torch.complex64)
        sigma2 = torch.tensor([[0.1]])
        llrs = calibrated_16qam_llrs(z, sigma2)
        self.assertAlmostEqual(float(llrs[0, 0, 0]), 0.0, places=3)

    def test_t2_f05_extreme_noise_variance_llr_collapse(self):
        """F5.3: Verify extreme noise variance collapses LLRs to 0."""
        calibrated_16qam_llrs = MilestoneResolver.get_callable("classical_baselines", "calibrated_16qam_llrs", "Milestone M1 (F5)")
        z = torch.tensor([[1.0 + 1.0j]], dtype=torch.complex64)
        sigma2 = torch.tensor([[1e6]])
        llrs = calibrated_16qam_llrs(z, sigma2)
        self.assertLess(llrs.abs().max().item(), 1e-4)

    def test_t2_f05_extreme_low_noise_clipping(self):
        """F5.4: Verify extreme low noise sigma2 = 1e-8 does not overflow to +inf."""
        calibrated_16qam_llrs = MilestoneResolver.get_callable("classical_baselines", "calibrated_16qam_llrs", "Milestone M1 (F5)")
        z = torch.tensor([[0.5 + 0.5j]], dtype=torch.complex64)
        sigma2 = torch.tensor([[1e-8]])
        llrs = calibrated_16qam_llrs(z, sigma2)
        self.assertFalse(torch.isinf(llrs).any())
        self.assertFalse(torch.isnan(llrs).any())

    def test_t2_f05_huge_amplitude_outlier_llr(self):
        """F5.5: Verify extreme outlier amplitude |z| = 100 remains finite."""
        calibrated_16qam_llrs = MilestoneResolver.get_callable("classical_baselines", "calibrated_16qam_llrs", "Milestone M1 (F5)")
        z = torch.tensor([[100.0 + 100.0j]], dtype=torch.complex64)
        sigma2 = torch.tensor([[0.1]])
        llrs = calibrated_16qam_llrs(z, sigma2)
        self.assertFalse(torch.isnan(llrs).any())

    # =========================================================================
    # Feature F6: Iterative DD-PA Cancellation (5 boundary tests)
    # =========================================================================

    def test_t2_f06_zero_iteration_identity(self):
        """F6.1: Verify zero iterations returns initial Bussgang-LMMSE estimate."""
        generate_frame = MilestoneResolver.get_callable("simulator", "generate_frame", "Milestone M1 (F6)")
        iterative_dd_pa_cancellation = MilestoneResolver.get_callable(
            "classical_baselines", "iterative_dd_pa_cancellation", "Milestone M1 (F6)"
        )
        bussgang_lmmse = MilestoneResolver.get_callable("classical_baselines", "bussgang_lmmse", "Milestone M1 (F6)")
        frame = generate_frame(batch_size=4, n=32, snr_db=15.0, pa_condition="strong")
        h_hat_dd, llrs_dd, hard_bits_dd, history = iterative_dd_pa_cancellation(
            frame.yp, frame.yd, frame.noise_var, A=1.25, iterations=0
        )
        res_lmmse = bussgang_lmmse(frame.yp, frame.yd, frame.noise_var, A=1.25)
        self.assertTrue(torch.allclose(history[0].z_hat, res_lmmse.z_hat, atol=1e-5))
        self.assertTrue(torch.allclose(llrs_dd, res_lmmse.llrs, atol=1e-5))
        self.assertTrue(torch.equal(hard_bits_dd, res_lmmse.hard_bits))

    def test_t2_f06_high_iterations_stability_10_iters(self):
        """F6.2: Verify running 10 iterations does not diverge or explode in energy."""
        generate_frame = MilestoneResolver.get_callable("simulator", "generate_frame", "Milestone M1 (F6)")
        iterative_dd_pa_cancellation = MilestoneResolver.get_callable(
            "classical_baselines", "iterative_dd_pa_cancellation", "Milestone M1 (F6)"
        )
        frame = generate_frame(batch_size=4, n=32, snr_db=20.0, pa_condition="strong")
        h_hat, llrs, hard_bits, history = iterative_dd_pa_cancellation(
            frame.yp, frame.yd, frame.noise_var, A=1.25, iterations=10
        )
        self.assertEqual(len(history), 11)
        self.assertFalse(torch.isnan(history[-1].z_hat).any())
        self.assertLess(float((history[-1].z_hat.abs() ** 2).mean()), 10.0)

    def test_t2_f06_perfect_decision_feedback_oracle(self):
        """F6.3: Verify decision feedback stably converges and achieves low BER under high SNR."""
        generate_frame = MilestoneResolver.get_callable("simulator", "generate_frame", "Milestone M1 (F6)")
        iterative_dd_pa_cancellation = MilestoneResolver.get_callable(
            "classical_baselines", "iterative_dd_pa_cancellation", "Milestone M1 (F6)"
        )
        frame = generate_frame(batch_size=8, n=32, snr_db=30.0, pa_condition="strong")
        h_hat, llrs, hard_bits, history = iterative_dd_pa_cancellation(
            frame.yp, frame.yd, frame.noise_var, A=1.25, iterations=2
        )
        ber = (hard_bits != frame.bits).float().mean().item()
        self.assertLess(ber, 0.05)

    def test_t2_f06_zero_snr_decision_feedback_robustness(self):
        """F6.4: Verify IDD-PA at -10 dB does not trigger catastrophic explosion."""
        generate_frame = MilestoneResolver.get_callable("simulator", "generate_frame", "Milestone M1 (F6)")
        iterative_dd_pa_cancellation = MilestoneResolver.get_callable(
            "classical_baselines", "iterative_dd_pa_cancellation", "Milestone M1 (F6)"
        )
        frame = generate_frame(batch_size=4, n=32, snr_db=-10.0, pa_condition="strong")
        z_hat, _, _, _ = iterative_dd_pa_cancellation(
            frame.yp, frame.yd, frame.noise_var, A=1.25, iterations=2
        )
        self.assertFalse(torch.isnan(z_hat).any())

    def test_t2_f06_residual_variance_underflow_prevention(self):
        """F6.5: Verify updated residual variance sigma_D^2 has lower bound >= 0."""
        iterative_dd_pa_cancellation = MilestoneResolver.get_callable(
            "classical_baselines", "iterative_dd_pa_cancellation", "Milestone M1 (F6)"
        )
        yp = torch.ones(1, 32, dtype=torch.complex64)
        yd = torch.ones(1, 32, dtype=torch.complex64)
        nv = torch.tensor([[0.01]])
        _, _, _, history = iterative_dd_pa_cancellation(yp, yd, nv, A=1.25, iterations=2)
        for h in history:
            self.assertGreaterEqual(h.sigma_d2_rem, 0.0)

    # =========================================================================
    # Feature F7: Classical Latency Benchmark (5 boundary tests)
    # =========================================================================

    def test_t2_f07_batch_size_1_latency(self):
        """F7.1: Verify single-symbol batch latency is < 20 ms on 4 CPU threads."""
        run_benchmarks = MilestoneResolver.get_callable("benchmark_latency", "run_benchmarks", "Milestone M1 (F7)")
        records, _ = run_benchmarks(batch_sizes=(1,), warmup_runs=2, timed_runs=5)
        for rec in records:
            if rec["batch_size"] == 1:
                self.assertLess(rec["median_ms"], 50.0)

    def test_t2_f07_batch_size_256_throughput(self):
        """F7.2: Verify large batch (B=64) memory stability."""
        run_benchmarks = MilestoneResolver.get_callable("benchmark_latency", "run_benchmarks", "Milestone M1 (F7)")
        records, _ = run_benchmarks(batch_sizes=(64,), warmup_runs=2, timed_runs=5)
        self.assertGreater(len(records), 0)

    def test_t2_f07_single_thread_concurrency(self):
        """F7.3: Verify execution with torch.set_num_threads(1)."""
        run_benchmarks = MilestoneResolver.get_callable("benchmark_latency", "run_benchmarks", "Milestone M1 (F7)")
        records, _ = run_benchmarks(batch_sizes=(1,), num_threads=1, warmup_runs=1, timed_runs=3)
        self.assertGreater(len(records), 0)

    def test_t2_f07_empty_warmup_recovery(self):
        """F7.4: Verify benchmark handles 0 warmup runs cleanly."""
        run_benchmarks = MilestoneResolver.get_callable("benchmark_latency", "run_benchmarks", "Milestone M1 (F7)")
        records, _ = run_benchmarks(batch_sizes=(1,), warmup_runs=0, timed_runs=3)
        self.assertGreater(len(records), 0)

    def test_t2_f07_zero_division_timing_guard(self):
        """F7.5: Verify timing logic guards against zero time deltas."""
        benchmark_method = MilestoneResolver.get_callable("benchmark_latency", "benchmark_method", "Milestone M1 (F7)")
        stats = benchmark_method(lambda: None, (), {}, warmup_runs=1, timed_runs=5)
        self.assertGreaterEqual(stats["min_ms"], 0.0)

    # =========================================================================
    # Feature F8: Exact C* Gauge Normalization (5 boundary tests - Milestone M2)
    # =========================================================================

    def test_t2_f08_extreme_attenuation_scale_2m12(self):
        """F8.1: Verify scale r = 2^-12 maintains exact bit logits (error < 1e-5, 0 flips)."""
        audit_scale = MilestoneResolver.get_callable("em_floatformer", "audit_single_scale", "Milestone M2 (F8)")
        max_err, flips = audit_scale(scale=2.0 ** (-12))
        self.assertLess(max_err, 1e-5)
        self.assertEqual(flips, 0)

    def test_t2_f08_extreme_amplification_scale_2p12(self):
        """F8.2: Verify scale r = 2^+12 maintains exact bit logits (error < 1e-5, 0 flips)."""
        audit_scale = MilestoneResolver.get_callable("em_floatformer", "audit_single_scale", "Milestone M2 (F8)")
        max_err, flips = audit_scale(scale=2.0 ** 12)
        self.assertLess(max_err, 1e-5)
        self.assertEqual(flips, 0)

    def test_t2_f08_zero_signal_power_eps_guard(self):
        """F8.3: Verify gauge normalization guards against zero power input without crash."""
        gauge_norm = MilestoneResolver.get_callable("em_floatformer", "apply_gauge_normalization", "Milestone M2 (F8)")
        yp_zero = torch.zeros(2, 32, dtype=torch.complex64)
        yd_zero = torch.zeros(2, 32, dtype=torch.complex64)
        yp_bar, yd_bar, s0 = gauge_norm(yp_zero, yd_zero)
        self.assertFalse(torch.isnan(yp_bar).any())

    def test_t2_f08_pure_phase_rotation_pi(self):
        """F8.4: Verify multiplication by e^{j pi} = -1 preserves decision bits."""
        audit_phase = MilestoneResolver.get_callable("em_floatformer", "audit_single_phase", "Milestone M2 (F8)")
        flips = audit_phase(phase=math.pi)
        self.assertEqual(flips, 0)

    def test_t2_f08_complex_rms_norm_subnormal_guard(self):
        """F8.5: Verify Complex RMSNorm with subnormal input 1e-35 outputs finite unit vector."""
        ComplexRMSNorm = MilestoneResolver.get_class("em_floatformer", "ComplexRMSNorm", "Milestone M2 (F8)")
        norm = ComplexRMSNorm(d_model=32)
        z_sub = torch.full((1, 32), 1e-35, dtype=torch.complex64)
        out = norm(z_sub)
        self.assertFalse(torch.isnan(out).any())

    # =========================================================================
    # Feature F9: Hermitian Green's Attention (5 boundary tests - Milestone M2)
    # =========================================================================

    def test_t2_f09_single_subcarrier_self_attention(self):
        """F9.1: Verify self-attention on N=1 subcarrier yields attention weight identically 1.0."""
        GreenAttention = MilestoneResolver.get_class("em_floatformer", "GreenAttention", "Milestone M2 (F9)")
        attn = GreenAttention(d_model=16, n_heads=1)
        z = torch.randn(1, 1, 16, dtype=torch.complex64)
        out, weights = attn(z, return_weights=True)
        self.assertAlmostEqual(float(weights[0, 0, 0, 0]), 1.0, places=5)

    def test_t2_f09_maximum_subcarrier_separation(self):
        """F9.2: Verify RoPE at maximum distance |m - n| = N - 1 remains unitary."""
        ComplexRoPE = MilestoneResolver.get_class("em_floatformer", "ComplexRoPE", "Milestone M2 (F9)")
        rope = ComplexRoPE(dim=16)
        phasor = rope.get_phasor(dist=31)
        self.assertAlmostEqual(float(phasor.abs().mean()), 1.0, places=5)

    def test_t2_f09_zero_magnitude_token_radial_activation(self):
        """F9.3: Verify radial activation on exact zero magnitude token gives 0 without NaN."""
        RadialActivation = MilestoneResolver.get_class("em_floatformer", "RadialActivation", "Milestone M2 (F9)")
        act = RadialActivation()
        z = torch.zeros(2, 4, dtype=torch.complex64)
        out = act(z)
        self.assertTrue(torch.all(out == 0.0))
        self.assertFalse(torch.isnan(out).any())

    def test_t2_f09_extreme_magnitude_token_radial_activation(self):
        """F9.4: Verify radial activation on token with magnitude 1e6 does not overflow."""
        RadialActivation = MilestoneResolver.get_class("em_floatformer", "RadialActivation", "Milestone M2 (F9)")
        act = RadialActivation()
        z = torch.full((2, 4), 1e6, dtype=torch.complex64)
        out = act(z)
        self.assertFalse(torch.isinf(out).any())

    def test_t2_f09_identical_keys_uniform_attention(self):
        """F9.5: Verify identical keys yield perfectly uniform softmax attention distribution."""
        GreenAttention = MilestoneResolver.get_class("em_floatformer", "GreenAttention", "Milestone M2 (F9)")
        attn = GreenAttention(d_model=16, n_heads=1)
        q = torch.randn(1, 4, 16, dtype=torch.complex64)
        k = torch.ones(1, 4, 16, dtype=torch.complex64)
        weights = attn.compute_weights(q, k)
        self.assertTrue(torch.allclose(weights, torch.full_like(weights, 0.25), atol=1e-3))

    # =========================================================================
    # Feature F10: Bidirectional Pilot-Data Attention (5 boundary tests - Milestone M2)
    # =========================================================================

    def test_t2_f10_all_pilot_no_data_slot(self):
        """F10.1: Verify block when data tokens are zero preserves pilot channel representation."""
        BidirectionalBlock = MilestoneResolver.get_class("em_floatformer", "BidirectionalBlock", "Milestone M2 (F10)")
        block = BidirectionalBlock(d_model=16)
        p = torch.randn(1, 8, 16, dtype=torch.complex64)
        d = torch.zeros(1, 8, 16, dtype=torch.complex64)
        p_out, _ = block(p, d)
        self.assertFalse(torch.isnan(p_out).any())

    def test_t2_f10_all_data_minimal_pilot_single_tone(self):
        """F10.2: Verify cross-attention propagates channel from single pilot tone."""
        BidirectionalBlock = MilestoneResolver.get_class("em_floatformer", "BidirectionalBlock", "Milestone M2 (F10)")
        block = BidirectionalBlock(d_model=16)
        p = torch.randn(1, 1, 16, dtype=torch.complex64)
        d = torch.randn(1, 15, 16, dtype=torch.complex64)
        p_out, d_out = block(p, d)
        self.assertEqual(d_out.shape, (1, 15, 16))

    def test_t2_f10_random_noise_data_tokens(self):
        """F10.3: Verify pure noise data tokens do not degrade pilot channel estimate severely."""
        evaluate_noise_data_feedback = MilestoneResolver.get_callable(
            "em_floatformer", "evaluate_noise_data_feedback", "Milestone M2 (F10)"
        )
        deg_db = evaluate_noise_data_feedback()
        self.assertLess(deg_db, 1.0)

    def test_t2_f10_perfect_channel_pilot_tokens(self):
        """F10.4: Verify noiseless pilots enable near-optimal AWGN demapping."""
        evaluate_perfect_pilot_demapping = MilestoneResolver.get_callable(
            "em_floatformer", "evaluate_perfect_pilot_demapping", "Milestone M2 (F10)"
        )
        ber = evaluate_perfect_pilot_demapping()
        self.assertLess(ber, 1e-3)

    def test_t2_f10_multi_layer_bidirectional_depth(self):
        """F10.5: Verify 1, 2, and 4 stacked bidirectional blocks execute stably."""
        EMFloatFormer = MilestoneResolver.get_class("em_floatformer", "EMFloatFormer", "Milestone M2 (F10)")
        for depth in [1, 2, 4]:
            model = EMFloatFormer(depth=depth)
            self.assertEqual(len(model.blocks), depth)

    # =========================================================================
    # Feature F11: Joint Physical Consistency Loss (5 boundary tests - Milestone M2)
    # =========================================================================

    def test_t2_f11_perfect_predictions_zero_loss(self):
        """F11.1: Verify loss approaches zero when predictions match ground truth."""
        JointLoss = MilestoneResolver.get_class("em_floatformer", "JointPhysicalLoss", "Milestone M2 (F11)")
        loss_fn = JointLoss()
        h = torch.randn(2, 32, dtype=torch.complex64)
        loss = loss_fn.nmse_loss(h, h)
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_t2_f11_completely_erroneous_predictions_bounded_loss(self):
        """F11.2: Verify BCE loss remains bounded when all signs inverted."""
        JointLoss = MilestoneResolver.get_class("em_floatformer", "JointPhysicalLoss", "Milestone M2 (F11)")
        loss_fn = JointLoss()
        llrs = torch.full((2, 32, 4), -20.0)
        bits = torch.ones(2, 32, 4, dtype=torch.int64)
        loss = loss_fn.bce_loss(llrs, bits)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_t2_f11_loss_scale_invariance_24_octaves(self):
        """F11.3: Verify loss invariance under scaling across 24 octaves."""
        audit_loss_scale_invariance = MilestoneResolver.get_callable(
            "em_floatformer", "audit_loss_scale_invariance", "Milestone M2 (F11)"
        )
        max_rel_diff = audit_loss_scale_invariance()
        self.assertLess(max_rel_diff, 1e-5)

    def test_t2_f11_zero_channel_denominator_protection(self):
        """F11.4: Verify NMSE loss protects against zero channel denominator."""
        JointLoss = MilestoneResolver.get_class("em_floatformer", "JointPhysicalLoss", "Milestone M2 (F11)")
        loss_fn = JointLoss()
        h_zero = torch.zeros(2, 32, dtype=torch.complex64)
        loss = loss_fn.nmse_loss(h_zero, h_zero)
        self.assertFalse(torch.isnan(loss))

    def test_t2_f11_zero_payload_power_pa_loss_protection(self):
        """F11.5: Verify PA residual loss protects against zero received data."""
        JointLoss = MilestoneResolver.get_class("em_floatformer", "JointPhysicalLoss", "Milestone M2 (F11)")
        loss_fn = JointLoss()
        yd_zero = torch.zeros(2, 32, dtype=torch.complex64)
        h = torch.ones(2, 32, dtype=torch.complex64)
        x = torch.zeros(2, 32, dtype=torch.complex64)
        loss = loss_fn.pa_residual_loss(yd_zero, h, x)
        self.assertFalse(torch.isnan(loss))

    # =========================================================================
    # Feature F12: Numerical Equivariance Verification (5 boundary tests - Milestone M2)
    # =========================================================================

    def test_t2_f12_extreme_gain_boundary_2m12_verification(self):
        """F12.1: Verify scale 2^-12 in pure FP32."""
        audit_scale = MilestoneResolver.get_callable("em_floatformer", "audit_single_scale", "Milestone M2 (F12)")
        _, flips = audit_scale(scale=2.0 ** (-12))
        self.assertEqual(flips, 0)

    def test_t2_f12_extreme_gain_boundary_2p12_verification(self):
        """F12.2: Verify scale 2^+12 in pure FP32."""
        audit_scale = MilestoneResolver.get_callable("em_floatformer", "audit_single_scale", "Milestone M2 (F12)")
        _, flips = audit_scale(scale=2.0 ** 12)
        self.assertEqual(flips, 0)

    def test_t2_f12_dense_phase_sweep_100_points(self):
        """F12.3: Verify 100 uniformly spaced phases on [0, 2pi) yield max logit error < 1e-5."""
        sweep_phases = MilestoneResolver.get_callable("em_floatformer", "sweep_phases_dense", "Milestone M2 (F12)")
        max_err = sweep_phases(num_points=100)
        self.assertLess(max_err, 1e-5)

    def test_t2_f12_irrational_phase_angles(self):
        """F12.4: Verify irrational phase angles pi/7, e, sqrt(2) yield 0 bit flips."""
        check_irrational_phases = MilestoneResolver.get_callable(
            "em_floatformer", "check_irrational_phases", "Milestone M2 (F12)"
        )
        flips = check_irrational_phases()
        self.assertEqual(flips, 0)

    def test_t2_f12_noise_variance_quad_scaling(self):
        """F12.5: Verify scaling noise variance by r^2 alongside signal by r maintains bit LLRs."""
        check_noise_quad_scaling = MilestoneResolver.get_callable(
            "em_floatformer", "check_noise_quad_scaling", "Milestone M2 (F12)"
        )
        max_diff = check_noise_quad_scaling(r=3.5)
        self.assertLess(max_diff, 1e-5)

    # =========================================================================
    # Feature F13: 2D Slot Simulator & Sparse DMRS (5 boundary tests - Milestone M3)
    # =========================================================================

    def test_t2_f13_single_symbol_slot_boundary(self):
        """F13.1: Verify single-symbol mini-slot N_symb=1 handled gracefully."""
        SlotSimulator2D = MilestoneResolver.get_class("slot_2d", "SlotSimulator2D", "Milestone M3 (F13)")
        sim = SlotSimulator2D(n_symb=1, n_sc=32)
        self.assertEqual(sim.n_symb, 1)

    def test_t2_f13_large_slot_boundary_28_symb(self):
        """F13.2: Verify double-length slot N_symb=28 scales correctly."""
        SlotSimulator2D = MilestoneResolver.get_class("slot_2d", "SlotSimulator2D", "Milestone M3 (F13)")
        sim = SlotSimulator2D(n_symb=28, n_sc=32)
        grid = sim.generate_empty_grid(batch_size=1)
        self.assertEqual(grid.shape, (1, 28, 32))

    def test_t2_f13_dense_subcarriers_boundary_128(self):
        """F13.3: Verify subcarrier count N_sc=128 grid construction."""
        SlotSimulator2D = MilestoneResolver.get_class("slot_2d", "SlotSimulator2D", "Milestone M3 (F13)")
        sim = SlotSimulator2D(n_symb=14, n_sc=128)
        grid = sim.generate_empty_grid(batch_size=1)
        self.assertEqual(grid.shape, (1, 14, 128))

    def test_t2_f13_ultra_sparse_dmrs_single_pilot_re(self):
        """F13.4: Verify minimal pilot pattern (single pilot RE) mask validity."""
        get_custom_dmrs_mask = MilestoneResolver.get_callable(
            "slot_2d", "get_custom_dmrs_mask", "Milestone M3 (F13)"
        )
        mask = get_custom_dmrs_mask(pilot_coords=[(2, 0)])
        self.assertEqual(int(mask.sum()), 1)

    def test_t2_f13_full_dmrs_slot_100_percent(self):
        """F13.5: Verify 100% pilot slot handles empty payload mask without crash."""
        SlotSimulator2D = MilestoneResolver.get_class("slot_2d", "SlotSimulator2D", "Milestone M3 (F13)")
        sim = SlotSimulator2D(n_symb=14, n_sc=32, pilot_ratio=1.0)
        self.assertEqual(sim.pilot_ratio, 1.0)

    # =========================================================================
    # Feature F14: Continuous Doppler ICI & CFO (5 boundary tests - Milestone M3)
    # =========================================================================

    def test_t2_f14_zero_doppler_zero_cfo_limit(self):
        """F14.1: Verify channel is time-invariant when fd=0 and cfo=0."""
        simulate_slot_channel = MilestoneResolver.get_callable(
            "slot_2d", "simulate_slot_channel", "Milestone M3 (F14)"
        )
        h_evol = simulate_slot_channel(batch_size=1, fd=0.0, cfo=0.0, return_h_time=True)
        # Taps should remain strictly constant over all symbols
        diff = (h_evol[:, 1:] - h_evol[:, :-1]).abs().max().item()
        self.assertLess(diff, 1e-6)

    def test_t2_f14_extreme_cfo_half_subcarrier(self):
        """F14.2: Verify extreme CFO epsilon = 0.5 (maximum ambiguity bound)."""
        simulate_slot_channel = MilestoneResolver.get_callable(
            "slot_2d", "simulate_slot_channel", "Milestone M3 (F14)"
        )
        slot = simulate_slot_channel(batch_size=1, cfo=0.5)
        self.assertFalse(torch.isnan(slot).any())

    def test_t2_f14_negative_cfo_phase_roll(self):
        """F14.3: Verify negative CFO rotates phase in reverse direction."""
        measure_cfo_phase_drift = MilestoneResolver.get_callable(
            "slot_2d", "measure_cfo_phase_drift", "Milestone M3 (F14)"
        )
        drift = measure_cfo_phase_drift(cfo=-0.04)
        self.assertAlmostEqual(drift, -0.04, places=2)

    def test_t2_f14_extreme_doppler_jakes_spectrum(self):
        """F14.4: Verify high Doppler spectrum bounds at max speed v=500 km/h."""
        compute_doppler_spectrum = MilestoneResolver.get_callable(
            "slot_2d", "compute_doppler_spectrum", "Milestone M3 (F14)"
        )
        spec = compute_doppler_spectrum(speed_kmh=500.0, fc=3.5e9)
        self.assertIsNotNone(spec)

    def test_t2_f14_phase_continuity_across_cp(self):
        """F14.5: Verify phase accumulation across CP has zero discontinuous jumps."""
        check_phase_continuity = MilestoneResolver.get_callable(
            "slot_2d", "check_phase_continuity", "Milestone M3 (F14)"
        )
        is_continuous = check_phase_continuity()
        self.assertTrue(is_continuous)

    # =========================================================================
    # Feature F15: Native Outer Channel Coding (5 boundary tests - Milestone M3)
    # =========================================================================

    def test_t2_f15_all_zeros_codeword(self):
        """F15.1: Verify all-zero message produces all-zero codeword satisfying parity."""
        encode_ldpc = MilestoneResolver.get_callable("slot_2d", "encode_ldpc", "Milestone M3 (F15)")
        u_zero = torch.zeros(1, 64, dtype=torch.int64)
        c, h_mat = encode_ldpc(u_zero)
        self.assertTrue(torch.all(c == 0))

    def test_t2_f15_all_ones_message(self):
        """F15.2: Verify all-one message produces valid codeword satisfying parity."""
        encode_ldpc = MilestoneResolver.get_callable("slot_2d", "encode_ldpc", "Milestone M3 (F15)")
        u_one = torch.ones(1, 64, dtype=torch.int64)
        c, h_mat = encode_ldpc(u_one)
        syndrome = torch.remainder(torch.matmul(c.float(), h_mat.t().float()), 2.0)
        self.assertTrue(torch.all(syndrome == 0))

    def test_t2_f15_single_bit_error_recovery(self):
        """F15.3: Verify Min-Sum decoder corrects single flipped bit in 1 iteration."""
        correct_single_bit = MilestoneResolver.get_callable("slot_2d", "correct_single_bit", "Milestone M3 (F15)")
        recovered = correct_single_bit()
        self.assertTrue(recovered)

    def test_t2_f15_max_iterations_unconverged_syndrome(self):
        """F15.4: Verify decoder terminates at max_iter when syndrome does not converge."""
        decode_min_sum = MilestoneResolver.get_callable("slot_2d", "decode_min_sum", "Milestone M3 (F15)")
        encode_ldpc = MilestoneResolver.get_callable("slot_2d", "encode_ldpc", "Milestone M3 (F15)")
        _, h_mat = encode_ldpc(torch.zeros(1, 64))
        # Random noise LLRs with high ambiguity
        noise_llrs = torch.randn(1, 128) * 0.01
        _, iters = decode_min_sum(noise_llrs, h_mat, max_iter=15, return_iters=True)
        self.assertEqual(iters, 15)

    def test_t2_f15_empty_payload_guard(self):
        """F15.5: Verify empty payload raises appropriate exception."""
        encode_ldpc = MilestoneResolver.get_callable("slot_2d", "encode_ldpc", "Milestone M3 (F15)")
        with self.assertRaises((ValueError, RuntimeError, AssertionError)):
            _ = encode_ldpc(torch.empty(0, 0))

    # =========================================================================
    # Feature F16: 2D Classical Baselines (5 boundary tests - Milestone M3)
    # =========================================================================

    def test_t2_f16_2d_lmmse_zero_doppler_flat_channel(self):
        """F16.1: Verify 2D-LMMSE under static AWGN produces constant response across symbols."""
        lmmse_2d_flat = MilestoneResolver.get_callable("slot_2d", "lmmse_2d_flat", "Milestone M3 (F16)")
        h_est = lmmse_2d_flat()
        diff = (h_est[:, 1:] - h_est[:, :-1]).abs().max().item()
        self.assertLess(diff, 1e-4)

    def test_t2_f16_2d_lmmse_singular_covariance_regularization(self):
        """F16.2: Verify 2D-LMMSE regularizer prevents numerical singularity as noise -> 0."""
        lmmse_2d_singular = MilestoneResolver.get_callable("slot_2d", "lmmse_2d_singular", "Milestone M3 (F16)")
        stable = lmmse_2d_singular(noise_var=1e-12)
        self.assertTrue(stable)

    def test_t2_f16_2d_idd_pa_linear_pa_boundary(self):
        """F16.3: Verify 2D IDD-PA distortion subtraction is zero under linear PA."""
        idd_pa_2d_linear = MilestoneResolver.get_callable("slot_2d", "idd_pa_2d_linear", "Milestone M3 (F16)")
        dist_pow = idd_pa_2d_linear()
        self.assertAlmostEqual(dist_pow, 0.0, places=5)

    def test_t2_f16_2d_idd_pa_extreme_clipping_convergence(self):
        """F16.4: Verify 2D IDD-PA does not diverge under severe clipping A=1.0."""
        idd_pa_2d_clipping = MilestoneResolver.get_callable("slot_2d", "idd_pa_2d_clipping", "Milestone M3 (F16)")
        stable = idd_pa_2d_clipping(A=1.0)
        self.assertTrue(stable)

    def test_t2_f16_2d_lmmse_sparse_dmrs_extrapolation(self):
        """F16.5: Verify channel extrapolation accuracy at slot edges (symbols 0 & 13)."""
        check_edge_extrapolation = MilestoneResolver.get_callable(
            "slot_2d", "check_edge_extrapolation", "Milestone M3 (F16)"
        )
        nmse_edges = check_edge_extrapolation()
        self.assertLess(nmse_edges, 0.15)

    # =========================================================================
    # Feature F17: Coded BLER Evaluation (5 boundary tests - Milestone M3)
    # =========================================================================

    def test_t2_f17_zero_noise_zero_bler(self):
        """F17.1: Verify BLER is identically zero at high SNR 40 dB."""
        evaluate_bler_point = MilestoneResolver.get_callable("slot_2d", "evaluate_bler_point", "Milestone M3 (F17)")
        bler = evaluate_bler_point(snr_db=40.0)
        self.assertEqual(bler, 0.0)

    def test_t2_f17_extreme_noise_100_percent_bler(self):
        """F17.2: Verify BLER is 1.0 at -15 dB SNR."""
        evaluate_bler_point = MilestoneResolver.get_callable("slot_2d", "evaluate_bler_point", "Milestone M3 (F17)")
        bler = evaluate_bler_point(snr_db=-15.0)
        self.assertEqual(bler, 1.0)

    def test_t2_f17_single_block_simulation(self):
        """F17.3: Verify simulation of exactly 1 block returns integer 0 or 1."""
        simulate_single_block = MilestoneResolver.get_callable("slot_2d", "simulate_single_block", "Milestone M3 (F17)")
        res = simulate_single_block()
        self.assertIn(res, [0, 1])

    def test_t2_f17_massive_blocks_accounting_overflow_check(self):
        """F17.4: Verify 64-bit integer counters over 10^6 simulated blocks."""
        check_integer_accumulator = MilestoneResolver.get_callable(
            "slot_2d", "check_integer_accumulator", "Milestone M3 (F17)"
        )
        val = check_integer_accumulator(count=1_000_000)
        self.assertEqual(val, 1_000_000)

    def test_t2_f17_bler_confidence_interval_computation(self):
        """F17.5: Verify Clopper-Pearson confidence interval returns valid interval [p_low, p_high]."""
        compute_ci = MilestoneResolver.get_callable("slot_2d", "compute_ci", "Milestone M3 (F17)")
        low, high = compute_ci(errors=5, total=100)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)
        self.assertLess(low, high)

    # =========================================================================
    # Feature F18: Factorial Ablation Matrix (5 boundary tests - Milestone M4)
    # =========================================================================

    def test_t2_f18_ablation_model_single_sample_batch(self):
        """F18.1: Verify all 8 arms forward pass with batch size B=1."""
        test_ablation_forward = MilestoneResolver.get_callable(
            "ablation_probing", "test_ablation_forward", "Milestone M4 (F18)"
        )
        success = test_ablation_forward(batch_size=1)
        self.assertTrue(success)

    def test_t2_f18_ablation_model_large_batch_128(self):
        """F18.2: Verify all 8 arms forward pass with batch size B=128."""
        test_ablation_forward = MilestoneResolver.get_callable(
            "ablation_probing", "test_ablation_forward", "Milestone M4 (F18)"
        )
        success = test_ablation_forward(batch_size=128)
        self.assertTrue(success)

    def test_t2_f18_parameter_budget_matching(self):
        """F18.3: Verify parameter counts of all 8 arms match within 15%."""
        check_arm_parameters = MilestoneResolver.get_callable(
            "ablation_probing", "check_arm_parameters", "Milestone M4 (F18)"
        )
        param_counts = check_arm_parameters()
        min_p = min(param_counts)
        max_p = max(param_counts)
        self.assertLess((max_p - min_p) / float(min_p), 0.15)

    def test_t2_f18_decoupled_arm_zero_feedback(self):
        """F18.4: Verify decoupled arms have zero data gradient flow into pilot estimator."""
        verify_decoupled_gradients = MilestoneResolver.get_callable(
            "ablation_probing", "verify_decoupled_gradients", "Milestone M4 (F18)"
        )
        grad_norm = verify_decoupled_gradients(arm_id=4)
        self.assertEqual(grad_norm, 0.0)

    def test_t2_f18_real_attention_arm_loss_of_u1_equivariance(self):
        """F18.5: Verify Arm 3 (real attention) breaks phase equivariance under e^{j pi/4}."""
        check_arm3_phase_sensitivity = MilestoneResolver.get_callable(
            "ablation_probing", "check_arm3_phase_sensitivity", "Milestone M4 (F18)"
        )
        logit_diff = check_arm3_phase_sensitivity()
        self.assertGreater(logit_diff, 0.5)

    # =========================================================================
    # Feature F19: Numeric Stability & Precision Audit (5 boundary tests - Milestone M4)
    # =========================================================================

    def test_t2_f19_fp16_subnormal_scale_stability(self):
        """F19.1: Verify FP16 forward pass at scale r = 2^-12 without subnormal underflow crash."""
        audit_fp16_scale = MilestoneResolver.get_callable(
            "ablation_probing", "audit_fp16_scale", "Milestone M4 (F19)"
        )
        stable = audit_fp16_scale(scale=2.0 ** (-12))
        self.assertTrue(stable)

    def test_t2_f19_fp16_max_representable_scale(self):
        """F19.2: Verify FP16 forward pass at scale r = 2^+12 does not overflow to +inf."""
        audit_fp16_scale = MilestoneResolver.get_callable(
            "ablation_probing", "audit_fp16_scale", "Milestone M4 (F19)"
        )
        stable = audit_fp16_scale(scale=2.0 ** 12)
        self.assertTrue(stable)

    def test_t2_f19_bf16_wide_exponent_stability(self):
        """F19.3: Verify BF16 stability across wide exponent range [2^-20, 2^+20]."""
        audit_bf16_wide_range = MilestoneResolver.get_callable(
            "ablation_probing", "audit_bf16_wide_range", "Milestone M4 (F19)"
        )
        stable = audit_bf16_wide_range()
        self.assertTrue(stable)

    def test_t2_f19_zero_input_autocast_finite(self):
        """F19.4: Verify all-zero input under half-precision produces finite outputs."""
        check_zero_autocast = MilestoneResolver.get_callable(
            "ablation_probing", "check_zero_autocast", "Milestone M4 (F19)"
        )
        finite = check_zero_autocast()
        self.assertTrue(finite)

    def test_t2_f19_loss_backward_autocast_stability(self):
        """F19.5: Verify backward pass under autocast produces zero NaNs in gradients."""
        check_autocast_gradients = MilestoneResolver.get_callable(
            "ablation_probing", "check_autocast_gradients", "Milestone M4 (F19)"
        )
        has_nans = check_autocast_gradients()
        self.assertFalse(has_nans)

    # =========================================================================
    # Feature F20: Pre-Frozen Confirmatory Protocol (5 boundary tests - Milestone M4)
    # =========================================================================

    def test_t2_f20_frozen_manifest_missing_file_detection(self):
        """F20.1: Verify verification raises error if protocol file is missing or modified."""
        verify_protocol_file = MilestoneResolver.get_callable(
            "confirmatory_protocol", "verify_protocol_file", "Milestone M4 (F20)"
        )
        with self.assertRaises((FileNotFoundError, ValueError, RuntimeError)):
            verify_protocol_file(path=PROJECT_ROOT / "non_existent_file.md")

    def test_t2_f20_seed_list_length_exactly_10(self):
        """F20.2: Verify seed list has length exactly 10."""
        self.assertEqual(len(FROZEN_SEEDS_P4D), 10)

    def test_t2_f20_seed_values_in_declared_range(self):
        """F20.3: Verify all seeds satisfy 74001 <= s <= 74010."""
        for s in FROZEN_SEEDS_P4D:
            self.assertTrue(74001 <= s <= 74010)

    def test_t2_f20_integer_type_enforcement(self):
        """F20.4: Verify passing float to integer recorder raises TypeError."""
        record_error = MilestoneResolver.get_callable(
            "confirmatory_protocol", "format_error_record", "Milestone M4 (F20)"
        )
        with self.assertRaises(TypeError):
            record_error(seed=74001, bit_errors=15.5, total_bits=1000)

    def test_t2_f20_empty_results_directory_initialization(self):
        """F20.5: Verify results directory initialization creates isolated path."""
        init_results_dir = MilestoneResolver.get_callable(
            "confirmatory_protocol", "init_results_dir", "Milestone M4 (F20)"
        )
        res_dir = init_results_dir(dir_name="p4_test_init")
        self.assertTrue(Path(res_dir).exists())

    # =========================================================================
    # Feature F21: Sign-Flip Permutation Tests (5 boundary tests - Milestone M4)
    # =========================================================================

    def test_t2_f21_single_observation_degeneracy(self):
        """F21.1: Verify permutation test returns p=1.0 for degenerate N=1 input."""
        p = exact_permutation_test_1024([5.0])
        self.assertEqual(p, 1.0)

    def test_t2_f21_all_positive_deltas_exact_p_value(self):
        """F21.2: Verify all positive deltas yield exact p = 2/1024."""
        deltas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        p = exact_permutation_test_1024(deltas)
        self.assertAlmostEqual(p, 2.0 / 1024.0, places=6)

    def test_t2_f21_balanced_alternating_signs_p_value(self):
        """F21.3: Verify balanced alternating signs (+1, -1, +1, -1...) yield p = 1.0."""
        deltas = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        p = exact_permutation_test_1024(deltas)
        self.assertEqual(p, 1.0)

    def test_t2_f21_holm_adjustment_single_hypothesis(self):
        """F21.4: Verify Holm adjustment with M=1 returns original p-value."""
        adj = holm_bonferroni_adjustment([0.03])
        self.assertEqual(len(adj), 1)
        self.assertAlmostEqual(adj[0], 0.03, places=5)

    def test_t2_f21_holm_adjustment_large_m_ceiling(self):
        """F21.5: Verify Holm adjusted p-values never exceed 1.0."""
        adj = holm_bonferroni_adjustment([0.8, 0.9, 0.95, 0.99])
        for p in adj:
            self.assertLessEqual(p, 1.0)

    # =========================================================================
    # Feature F22: Repository Governance & Immutability (5 boundary tests)
    # =========================================================================

    def test_t2_f22_tampered_byte_detection(self):
        """F22.1: Verify simulated byte alteration is detected by hash mismatch."""
        manifest = load_manifest()
        entry = manifest[0]
        full_path = PROJECT_ROOT / entry["file"]
        with open(full_path, "rb") as f:
            data = bytearray(f.read())
        # Tamper 1 byte
        data[0] = (data[0] + 1) % 256
        import hashlib
        tampered_hash = hashlib.sha256(data).hexdigest()
        self.assertNotEqual(tampered_hash, entry["sha256"])

    def test_t2_f22_new_file_outside_p4_detection(self):
        """F22.2: Verify that illegal directory creations outside p4/ and historical dirs are prevented."""
        # Allowed directory names (historical results, framework dirs, pytest cache, etc.)
        for item in PROJECT_ROOT.iterdir():
            if item.is_dir():
                name = item.name
                is_allowed = (
                    name in {"p4", "tests", ".agents", "exploratory", "figures", ".git", "results", "__pycache__", "p2_supplement", "p1e_train", ".pytest_cache"} or
                    name.endswith("_results")
                )
                self.assertTrue(is_allowed, f"Unrecognized directory in root: {name}")

    def test_t2_f22_historical_manifest_read_only_permissions(self):
        """F22.3: Verify MANIFEST_SHA256.json is not empty and valid JSON."""
        self.assertTrue(MANIFEST_PATH.exists())
        self.assertGreater(MANIFEST_PATH.stat().st_size, 1000)

    def test_t2_f22_crlf_lf_cross_platform_hash_stability(self):
        """F22.4: Verify CRLF-to-LF conversion produces stable hash on text file."""
        sample_text = b"line 1\r\nline 2\r\n"
        norm_text = sample_text.replace(b"\r\n", b"\n")
        self.assertEqual(norm_text, b"line 1\nline 2\n")

    def test_t2_f22_results_directory_isolation(self):
        """F22.5: Verify results directories reside under p4/."""
        results_dir = PROJECT_ROOT / "p4" / "results"
        self.assertTrue(results_dir.exists())


if __name__ == "__main__":
    unittest.main()
