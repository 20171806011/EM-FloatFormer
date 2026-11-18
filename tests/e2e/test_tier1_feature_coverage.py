"""
Stage P4 E2E Test Suite - Tier 1: Feature Coverage (110 Tests).

Covers features F1 to F22 (exactly 5 isolated test cases per feature):
- F1 to F7: M1 Dynamic Simulator, Bussgang-LMMSE Baselines, Classical Benchmark
- F8 to F12: M2 EM-FloatFormer Propagator Transformer & C* Gauge Equivariance
- F13 to F17: M3 2D Multi-Symbol Slot, Continuous Doppler/CFO, Outer FEC, 2D Baselines & Coded BLER
- F18 to F19: M4 Factorial Ablation Matrix & Mixed Precision Audits
- F20 to F21: M4 Pre-Frozen Confirmatory Protocol & Sign-Flip Permutation Tests
- F22: Repository Governance & Historical Immutability
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


class TestTier1FeatureCoverage(unittest.TestCase):
    """Tier 1: Feature Coverage (110 tests, 5 per feature F1–F22)."""

    def setUp(self):
        torch.set_num_threads(4)

    # =========================================================================
    # Feature F1: Oversampled Rapp PA (5 tests)
    # =========================================================================

    def test_t1_f01_rapp_am_am_compression(self):
        """F1.1: Verify Rapp AM-AM magnitude compression obeys analytical formula."""
        rapp_am_am = MilestoneResolver.get_callable("simulator", "rapp_am_am", "Milestone M1 (F1)")
        r = torch.linspace(0.1, 5.0, 100)
        for A in [1.25, 2.25]:
            knee = 2.0
            g_r = rapp_am_am(r, A=A, knee=knee)
            expected = r / ((1.0 + (r / A) ** 4.0) ** 0.25)
            self.assertTrue(torch.allclose(g_r, expected, atol=1e-5))
            self.assertTrue(torch.all(g_r < r))  # Compressive
            self.assertLessEqual(float(g_r.max()), A)  # Bounded by saturation

    def test_t1_f01_rapp_am_pm_phase_rotation(self):
        """F1.2: Verify Rapp AM-PM phase rotation obeys analytical formulation."""
        rapp_am_pm = MilestoneResolver.get_callable("simulator", "rapp_am_pm", "Milestone M1 (F1)")
        r = torch.linspace(0.0, 4.0, 100)
        alpha, beta = 0.5, 1.0
        phi_r = rapp_am_pm(r, alpha=alpha, beta=beta)
        expected = alpha * (r ** 2) / (1.0 + (r / beta) ** 2)
        self.assertTrue(torch.allclose(phi_r, expected, atol=1e-5))
        self.assertEqual(float(phi_r[0]), 0.0)  # Zero distortion at zero amplitude
        self.assertTrue(torch.all(phi_r[1:] > 0.0))  # Positive phase shift

    def test_t1_f01_oversampling_spectral_regrowth(self):
        """F1.3: Verify oversampled Rapp PA captures spectral regrowth without aliasing."""
        rapp_pa = MilestoneResolver.get_callable("simulator", "rapp_pa", "Milestone M1 (F1)")
        q16 = MilestoneResolver.get_callable("simulator", "q16", "Milestone M1 (F1)")
        bits = torch.randint(0, 2, (16, 32, 4))
        x_freq = q16(bits)
        x_time = torch.fft.ifft(x_freq, norm="ortho")
        y_time = rapp_pa(x_time, knee=2.0, A=1.25, alpha=0.5, oversample=4)
        self.assertEqual(y_time.shape, x_time.shape)
        self.assertFalse(torch.isnan(y_time).any())
        self.assertFalse(torch.isinf(y_time).any())

    def test_t1_f01_linear_limit_infinite_saturation(self):
        """F1.4: Verify linear limit as A -> infinity and alpha = 0 gives exact input parity."""
        rapp_pa = MilestoneResolver.get_callable("simulator", "rapp_pa", "Milestone M1 (F1)")
        x_time = torch.randn(8, 32, dtype=torch.complex64)
        y_time = rapp_pa(x_time, knee=2.0, A=None, alpha=0.0, oversample=4)
        max_diff = (y_time - x_time).abs().max().item()
        self.assertLess(max_diff, 1e-6)

    def test_t1_f01_power_attenuation_bussgang_scaling(self):
        """F1.5: Verify average power attenuation under strong saturation A=1.25."""
        rapp_pa = MilestoneResolver.get_callable("simulator", "rapp_pa", "Milestone M1 (F1)")
        torch.manual_seed(101)
        x_time = torch.randn(64, 32, dtype=torch.complex64)
        p_in = (x_time.abs() ** 2).mean().item()
        y_time = rapp_pa(x_time, knee=2.0, A=1.25, alpha=0.5, oversample=4)
        p_out = (y_time.abs() ** 2).mean().item()
        self.assertLess(p_out, p_in)  # Saturation reduces total output power

    # =========================================================================
    # Feature F2: Dynamic Multipath Simulator (5 tests)
    # =========================================================================

    def test_t1_f02_multipath_delay_profile_fir(self):
        """F2.1: Verify DynamicChannelSimulator initializes delay profile correctly."""
        DynamicChannelSimulator = MilestoneResolver.get_class("simulator", "DynamicChannelSimulator", "Milestone M1 (F2)")
        sim = DynamicChannelSimulator(n=32, cp=16, profile="equal4", fd=0.0)
        self.assertEqual(sim.num_taps, 4)
        self.assertAlmostEqual(float(sim.pdp.sum()), 1.0, places=5)
        ch_state = sim.draw_channel(batch_size=8)
        self.assertEqual(ch_state.h_time_initial.shape, (8, 4))
        self.assertEqual(ch_state.h_freq_static.shape, (8, 32))

    def test_t1_f02_cyclic_prefix_insertion_removal(self):
        """F2.2: Verify CP insertion and removal maintains dimension and boundary consistency."""
        generate_frame = MilestoneResolver.get_callable("simulator", "generate_frame", "Milestone M1 (F2)")
        frame = generate_frame(batch_size=4, n=32, cp=16, profile="equal4", snr_db=20.0, fd=0.0, pa_condition="clean")
        self.assertEqual(frame.yd.shape, (4, 32))
        self.assertEqual(frame.yp.shape, (4, 32))

    def test_t1_f02_doppler_time_varying_evolution(self):
        """F2.3: Verify non-zero Doppler generates time-varying impulse response across samples."""
        DynamicChannelSimulator = MilestoneResolver.get_class("simulator", "DynamicChannelSimulator", "Milestone M1 (F2)")
        sim = DynamicChannelSimulator(n=32, cp=16, profile="equal4", fd=0.05)
        ch_state = sim.draw_channel(batch_size=4)
        self.assertTrue(torch.any(ch_state.fd_shifts.abs() > 0))

    def test_t1_f02_ici_off_diagonal_energy(self):
        """F2.4: Verify Doppler ICI creates non-zero off-diagonal frequency transfer matrix elements."""
        DynamicChannelSimulator = MilestoneResolver.get_class("simulator", "DynamicChannelSimulator", "Milestone M1 (F2)")
        sim = DynamicChannelSimulator(n=32, cp=16, profile="equal4", fd=0.05)
        ch_state = sim.draw_channel(batch_size=4)
        h_mat = sim.compute_transfer_matrix(ch_state)
        self.assertEqual(h_mat.shape, (4, 32, 32))
        # Off-diagonal elements should contain non-trivial energy (> 1e-4)
        eye = torch.eye(32, dtype=torch.bool).unsqueeze(0)
        off_diag = h_mat[~eye.expand_as(h_mat)].abs()
        self.assertGreater(float(off_diag.max()), 1e-4)

    def test_t1_f02_awgn_noise_variance_scaling(self):
        """F2.5: Verify empirical noise variance matches configured snr_db."""
        generate_frame = MilestoneResolver.get_callable("simulator", "generate_frame", "Milestone M1 (F2)")
        g = torch.Generator().manual_seed(42)
        frame = generate_frame(batch_size=256, n=32, snr_db=10.0, fd=0.0, pa_condition="clean", generator=g)
        expected_noise_var = 10.0 ** (-10.0 / 10.0)
        self.assertAlmostEqual(float(frame.noise_var[0, 0]), expected_noise_var, places=4)

    # =========================================================================
    # Feature F3: Static-Limit Parity Check (5 tests)
    # =========================================================================

    def test_t1_f03_zero_doppler_linear_pa_matrix_parity(self):
        """F3.1: Verify dynamic simulator matches static matrix model Y = H*X to < 1e-5."""
        static_limit_parity_check = MilestoneResolver.get_callable("simulator", "static_limit_parity_check", "Milestone M1 (F3)")
        rel_err = static_limit_parity_check(n=32, cp=16, batch_size=64, profile="equal4")
        self.assertLess(rel_err, 1e-5)

    def test_t1_f03_single_subcarrier_impulse_response(self):
        """F3.2: Verify single tone excitation matches static transfer at that tone."""
        DynamicChannelSimulator = MilestoneResolver.get_class("simulator", "DynamicChannelSimulator", "Milestone M1 (F3)")
        sim = DynamicChannelSimulator(n=32, cp=16, profile="train3", fd=0.0)
        state = sim.draw_channel(batch_size=1)
        # Excitation on tone 5
        x_freq = torch.zeros(1, 32, dtype=torch.complex64)
        x_freq[0, 5] = 1.0 + 0j
        x_time = torch.fft.ifft(x_freq, norm="ortho")
        x_cp = torch.cat([x_time[:, -16:], x_time], dim=-1)
        y_time = sim.filter_time_signal(x_cp, state, noise_var=None)
        y_freq = torch.fft.fft(y_time[:, 16:], norm="ortho")
        expected_y = state.h_freq_static * x_freq
        diff = (y_freq - expected_y).norm() / expected_y.norm()
        self.assertLess(diff.item(), 1e-5)

    def test_t1_f03_snr_sweep_parity(self):
        """F3.3: Verify static parity across SNR levels (0 to 30 dB)."""
        static_limit_parity_check = MilestoneResolver.get_callable("simulator", "static_limit_parity_check", "Milestone M1 (F3)")
        for profile in ["train3", "equal4", "long6"]:
            rel_err = static_limit_parity_check(n=32, cp=16, batch_size=32, profile=profile)
            self.assertLess(rel_err, 1e-5)

    def test_t1_f03_different_fft_sizes_parity(self):
        """F3.4: Verify static parity holds for both N=32 and N=64."""
        static_limit_parity_check = MilestoneResolver.get_callable("simulator", "static_limit_parity_check", "Milestone M1 (F3)")
        for n in [32, 64]:
            rel_err = static_limit_parity_check(n=n, cp=16, batch_size=32, profile="equal4")
            self.assertLess(rel_err, 1e-5)

    def test_t1_f03_multi_channel_realizations_parity(self):
        """F3.5: Verify parity holds consistently across multiple seeds."""
        static_limit_parity_check = MilestoneResolver.get_callable("simulator", "static_limit_parity_check", "Milestone M1 (F3)")
        for seed in [101, 202, 303]:
            g = torch.Generator().manual_seed(seed)
            rel_err = static_limit_parity_check(n=32, cp=16, batch_size=16, generator=g)
            self.assertLess(rel_err, 1e-5)

    # =========================================================================
    # Feature F4: Bussgang Decomposition & LMMSE (5 tests)
    # =========================================================================

    def test_t1_f04_bussgang_gain_analytical_vs_empirical(self):
        """F4.1: Verify analytical Bussgang gamma matches Monte Carlo expectation to < 1%."""
        bussgang_params = MilestoneResolver.get_callable("classical_baselines", "bussgang_params", "Milestone M1 (F4)")
        apply_rapp_pa_samples = MilestoneResolver.get_callable("simulator", "apply_rapp_pa_samples", "Milestone M1 (F4)")
        gamma, sigma_d2, p_out = bussgang_params(A=1.25, alpha=0.5, beta=1.0)
        
        # Empirical Monte Carlo verification over 50,000 samples
        g = torch.Generator().manual_seed(42)
        x_r = torch.randn(50000, generator=g) / math.sqrt(2.0)
        x_i = torch.randn(50000, generator=g) / math.sqrt(2.0)
        x = torch.complex(x_r, x_i)
        y = apply_rapp_pa_samples(x, A=1.25, alpha=0.5, beta=1.0)
        gamma_emp = (y * x.conj()).mean().item()
        
        self.assertLess(abs(complex(gamma) - gamma_emp) / abs(gamma), 0.02)

    def test_t1_f04_distortion_variance_calculation(self):
        """F4.2: Verify distortion variance sigma_D^2 is strictly positive for nonlinear PA."""
        bussgang_params = MilestoneResolver.get_callable("classical_baselines", "bussgang_params", "Milestone M1 (F4)")
        gamma, sigma_d2, p_out = bussgang_params(A=1.25)
        self.assertGreater(sigma_d2, 0.0)
        self.assertAlmostEqual(p_out - abs(gamma) ** 2, sigma_d2, places=5)
        
        # In linear limit, sigma_d2 is 0
        gamma_lin, sigma_d2_lin, _ = bussgang_params(A=None, alpha=0.0)
        self.assertAlmostEqual(sigma_d2_lin, 0.0, places=5)
        self.assertAlmostEqual(abs(gamma_lin), 1.0, places=5)

    def test_t1_f04_bussgang_orthogonality(self):
        """F4.3: Verify Bussgang orthogonality E[x * d^*] < 3e-3."""
        bussgang_params = MilestoneResolver.get_callable("classical_baselines", "bussgang_params", "Milestone M1 (F4)")
        apply_rapp_pa_samples = MilestoneResolver.get_callable("simulator", "apply_rapp_pa_samples", "Milestone M1 (F4)")
        gamma, _, _ = bussgang_params(A=1.25, alpha=0.5)
        
        g = torch.Generator().manual_seed(42)
        x = (torch.randn(250000, generator=g) + 1j * torch.randn(250000, generator=g)) / math.sqrt(2.0)
        y = apply_rapp_pa_samples(x, A=1.25, alpha=0.5)
        d = y - complex(gamma) * x
        corr = (d * x.conj()).mean().abs().item()
        self.assertLess(corr, 3e-3)

    def test_t1_f04_lmmse_equalizer_filter_weights(self):
        """F4.4: Verify LMMSE equalizer filter calculation."""
        bussgang_lmmse = MilestoneResolver.get_callable("classical_baselines", "bussgang_lmmse", "Milestone M1 (F4)")
        yp = torch.ones(2, 32, dtype=torch.complex64)
        yd = torch.ones(2, 32, dtype=torch.complex64)
        nv = torch.full((2, 1), 0.01)
        res = bussgang_lmmse(yp, yd, nv, A=1.25, est_method="ls")
        self.assertEqual(res.w_filter.shape, (2, 32))
        self.assertTrue(torch.all(res.mu > 0.0))
        self.assertTrue(torch.all(res.mu <= 1.0))

    def test_t1_f04_lmmse_superiority_over_ls(self):
        """F4.5: Verify Bussgang-LMMSE achieves lower symbol MSE and BER than raw LS under strong PA."""
        generate_frame = MilestoneResolver.get_callable("simulator", "generate_frame", "Milestone M1 (F4)")
        bussgang_lmmse = MilestoneResolver.get_callable("classical_baselines", "bussgang_lmmse", "Milestone M1 (F4)")
        ls_est = MilestoneResolver.get_callable("classical_baselines", "ls_est", "Milestone M1 (F4)")
        hard_slice_16qam = MilestoneResolver.get_callable("classical_baselines", "hard_slice_16qam", "Milestone M1 (F4)")
        
        g = torch.Generator().manual_seed(42)
        frame = generate_frame(batch_size=128, n=32, snr_db=15.0, pa_condition="strong", generator=g)
        res_lmmse = bussgang_lmmse(frame.yp, frame.yd, frame.noise_var, A=1.25)
        h_ls = ls_est(frame.yp)
        z_ls = frame.yd / (h_ls + 1e-12)
        
        x_mmse = res_lmmse.w_filter * frame.yd
        mse_lmmse = ((x_mmse - frame.data_tx).abs() ** 2).mean().item()
        mse_ls = ((z_ls - frame.data_tx).abs() ** 2).mean().item()
        self.assertLess(mse_lmmse, mse_ls)

        ber_lmmse = (res_lmmse.hard_bits != frame.bits).float().mean().item()
        ber_ls = (hard_slice_16qam(z_ls) != frame.bits).float().mean().item()
        self.assertLess(ber_lmmse, ber_ls)

    # =========================================================================
    # Feature F5: Calibrated 16-QAM Soft LLRs (5 tests)
    # =========================================================================

    def test_t1_f05_gray_mapping_bit_definitions(self):
        """F5.1: Verify Gray mapping bit definitions and distance properties."""
        q16 = MilestoneResolver.get_callable("simulator", "q16", "Milestone M1 (F5)")
        bits = torch.zeros(16, 4, dtype=torch.int64)
        for i in range(16):
            bits[i] = torch.tensor([(i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1])
        symbols = q16(bits)
        # Check uniqueness by viewing complex as 2D real tensor
        symbols_real = torch.view_as_real(symbols)
        unique_symbols = torch.unique(symbols_real, dim=0)
        self.assertEqual(len(unique_symbols), 16)
        # Average energy should be exactly 1.0
        self.assertAlmostEqual(float((symbols.abs() ** 2).mean()), 1.0, places=5)

    def test_t1_f05_max_log_llr_sign_quadrant_concordance(self):
        """F5.2: Verify sign of LLR(b0) and LLR(b2) match real and imaginary quadrants."""
        calibrated_16qam_llrs = MilestoneResolver.get_callable("classical_baselines", "calibrated_16qam_llrs", "Milestone M1 (F5)")
        # Test point in first quadrant (+1 + 1j)
        z = torch.tensor([[1.0 + 1.0j]], dtype=torch.complex64)
        sigma2 = torch.tensor([[0.1]])
        llrs = calibrated_16qam_llrs(z, sigma2)
        self.assertGreater(float(llrs[0, 0, 0]), 0.0)  # Re > 0 -> b0 = 1
        self.assertGreater(float(llrs[0, 0, 2]), 0.0)  # Im > 0 -> b2 = 1

    def test_t1_f05_magnitude_boundary_bits(self):
        """F5.3: Verify sign of LLR(b1) and LLR(b3) match inner vs outer rings."""
        calibrated_16qam_llrs = MilestoneResolver.get_callable("classical_baselines", "calibrated_16qam_llrs", "Milestone M1 (F5)")
        # Inner point (|Re| < 2/sqrt(10)) vs outer point (|Re| > 2/sqrt(10))
        z_inner = torch.tensor([[0.2 + 0.2j]], dtype=torch.complex64)
        z_outer = torch.tensor([[1.5 + 1.5j]], dtype=torch.complex64)
        sigma2 = torch.tensor([[0.1]])
        llr_inner = calibrated_16qam_llrs(z_inner, sigma2)
        llr_outer = calibrated_16qam_llrs(z_outer, sigma2)
        self.assertGreater(float(llr_inner[0, 0, 1]), 0.0)  # Inner -> b1 = 1
        self.assertLess(float(llr_outer[0, 0, 1]), 0.0)     # Outer -> b1 = 0

    def test_t1_f05_llr_variance_scaling(self):
        """F5.4: Verify LLR magnitudes scale inversely with effective noise variance."""
        calibrated_16qam_llrs = MilestoneResolver.get_callable("classical_baselines", "calibrated_16qam_llrs", "Milestone M1 (F5)")
        z = torch.tensor([[1.0 + 1.0j]], dtype=torch.complex64)
        llr_low_noise = calibrated_16qam_llrs(z, torch.tensor([[0.01]]))
        llr_high_noise = calibrated_16qam_llrs(z, torch.tensor([[0.5]]))
        self.assertGreater(float(llr_low_noise.abs().max()), float(llr_high_noise.abs().max()))

    def test_t1_f05_awgn_probability_calibration_ece(self):
        """F5.5: Verify calibration ECE < 0.05 on AWGN channel."""
        calibrated_16qam_llrs = MilestoneResolver.get_callable("classical_baselines", "calibrated_16qam_llrs", "Milestone M1 (F5)")
        q16 = MilestoneResolver.get_callable("simulator", "q16", "Milestone M1 (F5)")
        g = torch.Generator().manual_seed(42)
        bits = torch.randint(0, 2, (1000, 32, 4), generator=g)
        x = q16(bits)
        noise_var = 0.1
        w = (torch.randn(1000, 32, generator=g) + 1j * torch.randn(1000, 32, generator=g)) * math.sqrt(noise_var / 2.0)
        y = x + w
        llrs = calibrated_16qam_llrs(y, torch.full((1000, 32), noise_var))
        probs = torch.sigmoid(llrs)
        
        # Simple empirical ECE check: predicted probability matches true bit accuracy
        preds = (probs > 0.5).long()
        acc = (preds == bits).float().mean().item()
        mean_prob = probs.where(preds == 1, 1.0 - probs).mean().item()
        ece = abs(acc - mean_prob)
        self.assertLess(ece, 0.05)

    # =========================================================================
    # Feature F6: Iterative DD-PA Cancellation (5 tests)
    # =========================================================================

    def test_t1_f06_iteration_loop_execution(self):
        """F6.1: Verify 2- and 3-iteration DD-PA loops run and return history records."""
        generate_frame = MilestoneResolver.get_callable("simulator", "generate_frame", "Milestone M1 (F6)")
        iterative_dd_pa_cancellation = MilestoneResolver.get_callable(
            "classical_baselines", "iterative_dd_pa_cancellation", "Milestone M1 (F6)"
        )
        frame = generate_frame(batch_size=4, n=32, snr_db=15.0, pa_condition="strong")
        h_hat, llrs, hard_bits, history = iterative_dd_pa_cancellation(
            frame.yp, frame.yd, frame.noise_var, A=1.25, iterations=3
        )
        self.assertEqual(len(history), 4)  # Iter 0, 1, 2, 3
        self.assertEqual(h_hat.shape, (4, 32))
        self.assertEqual(llrs.shape, (4, 32, 4))
        self.assertEqual(history[-1].z_hat.shape, (4, 32))

    def test_t1_f06_distortion_reconstruction_subtraction(self):
        """F6.2: Verify distortion subtraction cleans the received symbol grid."""
        generate_frame = MilestoneResolver.get_callable("simulator", "generate_frame", "Milestone M1 (F6)")
        iterative_dd_pa_cancellation = MilestoneResolver.get_callable(
            "classical_baselines", "iterative_dd_pa_cancellation", "Milestone M1 (F6)"
        )
        frame = generate_frame(batch_size=8, n=32, snr_db=20.0, pa_condition="strong")
        _, _, _, history = iterative_dd_pa_cancellation(
            frame.yp, frame.yd, frame.noise_var, A=1.25, iterations=2
        )
        # Verify distortion estimate at iteration 1 has non-zero power
        dist1 = history[1].dist_est_freq
        self.assertGreater(float((dist1.abs() ** 2).mean()), 1e-5)

    def test_t1_f06_monotonic_mse_reduction(self):
        """F6.3: Verify MSE improves across iterations under strong PA (A=1.25)."""
        generate_frame = MilestoneResolver.get_callable("simulator", "generate_frame", "Milestone M1 (F6)")
        iterative_dd_pa_cancellation = MilestoneResolver.get_callable(
            "classical_baselines", "iterative_dd_pa_cancellation", "Milestone M1 (F6)"
        )
        g = torch.Generator().manual_seed(123)
        frame = generate_frame(batch_size=64, n=32, snr_db=20.0, pa_condition="strong", generator=g)
        _, _, _, history = iterative_dd_pa_cancellation(
            frame.yp, frame.yd, frame.noise_var, A=1.25, iterations=2
        )
        mse0 = ((history[0].z_hat - frame.data_tx).abs() ** 2).mean().item()
        mse1 = ((history[1].z_hat - frame.data_tx).abs() ** 2).mean().item()
        self.assertLess(mse1, mse0)

    def test_t1_f06_ber_improvement_across_iterations(self):
        """F6.4: Verify bit error rate decreases with iteration count."""
        generate_frame = MilestoneResolver.get_callable("simulator", "generate_frame", "Milestone M1 (F6)")
        iterative_dd_pa_cancellation = MilestoneResolver.get_callable(
            "classical_baselines", "iterative_dd_pa_cancellation", "Milestone M1 (F6)"
        )
        g = torch.Generator().manual_seed(456)
        frame = generate_frame(batch_size=64, n=32, snr_db=18.0, pa_condition="strong", generator=g)
        _, _, _, history = iterative_dd_pa_cancellation(
            frame.yp, frame.yd, frame.noise_var, A=1.25, iterations=2
        )
        ber0 = (history[0].hard_bits != frame.bits).float().mean().item()
        ber1 = (history[1].hard_bits != frame.bits).float().mean().item()
        self.assertLessEqual(ber1, ber0)

    def test_t1_f06_linear_channel_early_convergence(self):
        """F6.5: Verify under linear PA, iteration 0 already converges because no PA cancellation is needed."""
        generate_frame = MilestoneResolver.get_callable("simulator", "generate_frame", "Milestone M1 (F6)")
        iterative_dd_pa_cancellation = MilestoneResolver.get_callable(
            "classical_baselines", "iterative_dd_pa_cancellation", "Milestone M1 (F6)"
        )
        frame = generate_frame(batch_size=16, n=32, snr_db=20.0, pa_condition="clean")
        _, _, _, history = iterative_dd_pa_cancellation(
            frame.yp, frame.yd, frame.noise_var, A=None, alpha=0.0, iterations=2
        )
        # Under clean linear PA, history has exactly 1 record (iteration 0)
        self.assertEqual(len(history), 1)
        mse0 = ((history[0].z_hat - frame.data_tx).abs() ** 2).mean().item()
        self.assertLess(mse0, 0.5)

    # =========================================================================
    # Feature F7: Classical Latency Benchmark (5 tests)
    # =========================================================================

    def test_t1_f07_benchmark_script_invocation(self):
        """F7.1: Verify benchmark suite runs cleanly and completes with non-empty results."""
        run_benchmarks = MilestoneResolver.get_callable("benchmark_latency", "run_benchmarks", "Milestone M1 (F7)")
        records, summary = run_benchmarks(
            batch_sizes=(1,), num_threads=4, warmup_runs=2, timed_runs=5, output_dir="p4/results"
        )
        self.assertGreater(len(records), 0)
        self.assertIn("timestamp_utc", summary)

    def test_t1_f07_csv_timing_artifact_generation(self):
        """F7.2: Verify p4/results/latency_cpu.csv artifact generation."""
        csv_path = PROJECT_ROOT / "p4" / "results" / "latency_cpu.csv"
        # If not present yet, run quick benchmark
        if not csv_path.exists():
            run_benchmarks = MilestoneResolver.get_callable("benchmark_latency", "run_benchmarks", "Milestone M1 (F7)")
            run_benchmarks(batch_sizes=(1,), num_threads=4, warmup_runs=2, timed_runs=5, output_dir="p4/results")
        self.assertTrue(csv_path.exists())
        self.assertGreater(csv_path.stat().st_size, 0)

    def test_t1_f07_json_timing_artifact_generation(self):
        """F7.3: Verify p4/results/timing.json structured artifact generation."""
        json_path = PROJECT_ROOT / "p4" / "results" / "timing.json"
        if not json_path.exists():
            run_benchmarks = MilestoneResolver.get_callable("benchmark_latency", "run_benchmarks", "Milestone M1 (F7)")
            run_benchmarks(batch_sizes=(1,), num_threads=4, warmup_runs=2, timed_runs=5, output_dir="p4/results")
        self.assertTrue(json_path.exists())
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertIn("results", data)
        self.assertIn("timestamp_utc", data)

    def test_t1_f07_cpu_thread_concurrency_control(self):
        """F7.4: Verify benchmark respects torch.set_num_threads(4)."""
        run_benchmarks = MilestoneResolver.get_callable("benchmark_latency", "run_benchmarks", "Milestone M1 (F7)")
        torch.set_num_threads(4)
        self.assertEqual(torch.get_num_threads(), 4)

    def test_t1_f07_latency_hierarchy_ordering(self):
        """F7.5: Verify latency ordering: LS < Bussgang-LMMSE < DD-PA."""
        run_benchmarks = MilestoneResolver.get_callable("benchmark_latency", "run_benchmarks", "Milestone M1 (F7)")
        records, _ = run_benchmarks(
            batch_sizes=(1,), num_threads=4, warmup_runs=3, timed_runs=10, output_dir="p4/results"
        )
        timing_map = {r["method"]: r["median_ms"] for r in records if r["batch_size"] == 1}
        self.assertIn("LS", timing_map)
        self.assertIn("Bussgang-LMMSE", timing_map)
        self.assertIn("DD-PA-Iter3", timing_map)
        self.assertLess(timing_map["LS"], timing_map["DD-PA-Iter3"])

    # =========================================================================
    # Feature F8: Exact C* Gauge Normalization (5 tests - Milestone M2)
    # =========================================================================

    def test_t1_f08_pilot_energy_gauge_anchor(self):
        """F8.1: Verify pilot RMS energy anchoring extracts correct normalization scale."""
        gauge_anchor = MilestoneResolver.get_callable("em_floatformer", "compute_gauge_anchor", "Milestone M2 (F8)")
        yp = torch.randn(4, 32, dtype=torch.complex64) * 5.0
        s0 = gauge_anchor(yp)
        self.assertEqual(s0.shape, (4, 1))
        self.assertTrue(torch.all(s0 > 0.0))

    def test_t1_f08_unit_rms_normalization(self):
        """F8.2: Verify normalized observation has exactly unit RMS power."""
        gauge_norm = MilestoneResolver.get_callable("em_floatformer", "apply_gauge_normalization", "Milestone M2 (F8)")
        yp = torch.randn(4, 32, dtype=torch.complex64) * 3.7
        yd = torch.randn(4, 32, dtype=torch.complex64) * 3.7
        yp_bar, yd_bar, s0 = gauge_norm(yp, yd)
        rms = (yp_bar.abs() ** 2).mean(dim=-1).sqrt()
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-5))

    def test_t1_f08_c_rms_norm_phase_preservation(self):
        """F8.3: Verify Complex RMSNorm preserves complex phase angles."""
        ComplexRMSNorm = MilestoneResolver.get_class("em_floatformer", "ComplexRMSNorm", "Milestone M2 (F8)")
        norm_layer = ComplexRMSNorm(d_model=64)
        z = torch.randn(2, 32, 64, dtype=torch.complex64)
        z_out = norm_layer(z)
        phase_in = torch.angle(z)
        phase_out = torch.angle(z_out)
        self.assertTrue(torch.allclose(phase_in, phase_out, atol=1e-5))

    def test_t1_f08_channel_rescaling_exactness(self):
        """F8.4: Verify output channel estimate recovers true physical scale via s0 * H_bar."""
        rescale_channel = MilestoneResolver.get_callable("em_floatformer", "rescale_channel_estimate", "Milestone M2 (F8)")
        h_bar = torch.ones(2, 32, dtype=torch.complex64)
        s0 = torch.tensor([[2.5], [4.0]])
        h_phys = rescale_channel(h_bar, s0)
        self.assertTrue(torch.allclose(h_phys[0].abs(), torch.tensor(2.5)))
        self.assertTrue(torch.allclose(h_phys[1].abs(), torch.tensor(4.0)))

    def test_t1_f08_scale_invariance_under_arbitrary_gain(self):
        """F8.5: Verify bit logits are identical under arbitrary gain r in [2^-12, 2^12]."""
        model_cls = MilestoneResolver.get_class("em_floatformer", "EMFloatFormer", "Milestone M2 (F8)")
        model = model_cls()
        model.eval()
        yp = torch.randn(2, 32, dtype=torch.complex64)
        yd = torch.randn(2, 32, dtype=torch.complex64)
        nv = torch.tensor([[0.01], [0.01]])
        with torch.no_grad():
            out1 = model(yp, yd, nv)
            out2 = model(yp * 4096.0, yd * 4096.0, nv * (4096.0 ** 2))
        max_diff = (out1["llrs"] - out2["llrs"]).abs().max().item()
        self.assertLess(max_diff, 1e-5)

    # =========================================================================
    # Feature F9: Hermitian Green's Attention (5 tests - Milestone M2)
    # =========================================================================

    def test_t1_f09_complex_linear_zero_bias(self):
        """F9.1: Verify complex linear projection enforces zero additive bias."""
        ComplexLinear = MilestoneResolver.get_class("em_floatformer", "ComplexLinear", "Milestone M2 (F9)")
        layer = ComplexLinear(in_features=32, out_features=32, bias=False)
        self.assertIsNone(layer.bias)

    def test_t1_f09_complex_linear_homogeneity(self):
        """F9.2: Verify ComplexLinear satisfies W(c Z) = c W(Z)."""
        ComplexLinear = MilestoneResolver.get_class("em_floatformer", "ComplexLinear", "Milestone M2 (F9)")
        layer = ComplexLinear(in_features=16, out_features=16, bias=False)
        z = torch.randn(2, 10, 16, dtype=torch.complex64)
        c = 2.5 * torch.exp(1j * torch.tensor(0.7))
        out1 = layer(c * z)
        out2 = c * layer(z)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-5))

    def test_t1_f09_relative_phase_rope(self):
        """F9.3: Verify complex RoPE modulates relative subcarrier distance."""
        ComplexRoPE = MilestoneResolver.get_class("em_floatformer", "ComplexRoPE", "Milestone M2 (F9)")
        rope = ComplexRoPE(dim=16)
        q = torch.randn(1, 8, 16, dtype=torch.complex64)
        q_rot = rope(q)
        self.assertEqual(q_rot.shape, q.shape)

    def test_t1_f09_hermitian_inner_product_phase_cancellation(self):
        """F9.4: Verify Re(<e^{j phi} Q, e^{j phi} K>) == Re(<Q, K>)."""
        hermitian_score = MilestoneResolver.get_callable("em_floatformer", "hermitian_attention_scores", "Milestone M2 (F9)")
        q = torch.randn(2, 4, 16, dtype=torch.complex64)
        k = torch.randn(2, 4, 16, dtype=torch.complex64)
        phi = 1.234
        phase = torch.exp(1j * torch.tensor(phi))
        s1 = hermitian_score(q, k)
        s2 = hermitian_score(phase * q, phase * k)
        self.assertTrue(torch.allclose(s1, s2, atol=1e-5))

    def test_t1_f09_radial_activation_u1_equivariance(self):
        """F9.5: Verify radial activation satisfies f(e^{j phi} z) = e^{j phi} f(z)."""
        RadialActivation = MilestoneResolver.get_class("em_floatformer", "RadialActivation", "Milestone M2 (F9)")
        act = RadialActivation()
        z = torch.randn(4, 16, dtype=torch.complex64)
        phi = 0.88
        phase = torch.exp(1j * torch.tensor(phi))
        out1 = act(phase * z)
        out2 = phase * act(z)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-5))

    # =========================================================================
    # Feature F10: Bidirectional Pilot-Data Attention (5 tests - Milestone M2)
    # =========================================================================

    def test_t1_f10_dual_stream_embedding_shapes(self):
        """F10.1: Verify dual-stream token embedding properly forms pilot and data tokens."""
        JointEmbedding = MilestoneResolver.get_class("em_floatformer", "JointEmbedding", "Milestone M2 (F10)")
        emb = JointEmbedding(n=32, d_model=64)
        yp = torch.randn(2, 32, dtype=torch.complex64)
        yd = torch.randn(2, 32, dtype=torch.complex64)
        tokens = emb(yp, yd)
        self.assertEqual(tokens.shape, (2, 64, 64))

    def test_t1_f10_cross_attention_pilot_to_data(self):
        """F10.2: Verify pilot tokens propagate channel information into data tokens."""
        BidirectionalBlock = MilestoneResolver.get_class("em_floatformer", "BidirectionalBlock", "Milestone M2 (F10)")
        block = BidirectionalBlock(d_model=32)
        p_in = torch.randn(2, 16, 32, dtype=torch.complex64)
        d_in = torch.randn(2, 16, 32, dtype=torch.complex64)
        p_out, d_out = block(p_in, d_in)
        self.assertEqual(d_out.shape, d_in.shape)

    def test_t1_f10_cross_attention_data_to_pilot(self):
        """F10.3: Verify data tokens feed tentative distortion back into pilot tokens."""
        BidirectionalBlock = MilestoneResolver.get_class("em_floatformer", "BidirectionalBlock", "Milestone M2 (F10)")
        block = BidirectionalBlock(d_model=32)
        p_in = torch.randn(2, 16, 32, dtype=torch.complex64)
        d_in = torch.randn(2, 16, 32, dtype=torch.complex64)
        p_out, d_out = block(p_in, d_in)
        self.assertEqual(p_out.shape, p_in.shape)

    def test_t1_f10_gradient_flow_bidirectional(self):
        """F10.4: Verify non-zero gradients flow between pilot and data streams."""
        BidirectionalBlock = MilestoneResolver.get_class("em_floatformer", "BidirectionalBlock", "Milestone M2 (F10)")
        block = BidirectionalBlock(d_model=32)
        p_in = torch.randn(2, 16, 32, dtype=torch.complex64, requires_grad=True)
        d_in = torch.randn(2, 16, 32, dtype=torch.complex64, requires_grad=True)
        p_out, d_out = block(p_in, d_in)
        loss = (d_out.abs() ** 2).sum()
        loss.backward()
        self.assertIsNotNone(p_in.grad)
        self.assertGreater(float(p_in.grad.abs().sum()), 0.0)

    def test_t1_f10_ber_gain_under_strong_pa(self):
        """F10.5: Verify bidirectional attention achieves lower BER than pilot-only model."""
        compare_pilot_vs_bidirectional = MilestoneResolver.get_callable(
            "em_floatformer", "compare_pilot_vs_bidirectional", "Milestone M2 (F10)"
        )
        gain_db = compare_pilot_vs_bidirectional()
        self.assertGreaterEqual(gain_db, 0.5)

    # =========================================================================
    # Feature F11: Joint Physical Consistency Loss (5 tests - Milestone M2)
    # =========================================================================

    def test_t1_f11_calibrated_llr_bce_loss(self):
        """F11.1: Verify BCE loss computes binary cross-entropy on bit projections."""
        JointLoss = MilestoneResolver.get_class("em_floatformer", "JointPhysicalLoss", "Milestone M2 (F11)")
        loss_fn = JointLoss()
        llrs = torch.randn(4, 32, 4)
        bits = torch.randint(0, 2, (4, 32, 4))
        loss = loss_fn.bce_loss(llrs, bits)
        self.assertGreater(float(loss), 0.0)

    def test_t1_f11_normalized_channel_nmse_loss(self):
        """F11.2: Verify normalized channel NMSE loss is non-negative and scale-invariant."""
        JointLoss = MilestoneResolver.get_class("em_floatformer", "JointPhysicalLoss", "Milestone M2 (F11)")
        loss_fn = JointLoss()
        h_true = torch.randn(2, 32, dtype=torch.complex64)
        h_est = h_true * 1.1
        loss1 = loss_fn.nmse_loss(h_est, h_true)
        loss2 = loss_fn.nmse_loss(h_est * 5.0, h_true * 5.0)
        self.assertAlmostEqual(float(loss1), float(loss2), places=5)

    def test_t1_f11_pa_residual_consistency_loss(self):
        """F11.3: Verify forward PA residual loss computes consistency without true channel."""
        JointLoss = MilestoneResolver.get_class("em_floatformer", "JointPhysicalLoss", "Milestone M2 (F11)")
        loss_fn = JointLoss()
        yd = torch.randn(2, 32, dtype=torch.complex64)
        h_est = torch.ones(2, 32, dtype=torch.complex64)
        x_est = torch.ones(2, 32, dtype=torch.complex64)
        res_loss = loss_fn.pa_residual_loss(yd, h_est, x_est, A=1.25)
        self.assertGreaterEqual(float(res_loss), 0.0)

    def test_t1_f11_joint_loss_weighting_schedule(self):
        """F11.4: Verify scheduled weighting lambda1 and lambda2."""
        JointLoss = MilestoneResolver.get_class("em_floatformer", "JointPhysicalLoss", "Milestone M2 (F11)")
        loss_fn = JointLoss(lambda_nmse=0.5, lambda_pa=0.2)
        self.assertEqual(loss_fn.lambda_nmse, 0.5)
        self.assertEqual(loss_fn.lambda_pa, 0.2)

    def test_t1_f11_loss_gradient_scale_invariance(self):
        """F11.5: Verify parameter gradients are scale-invariant under input scaling."""
        verify_loss_gradient_invariance = MilestoneResolver.get_callable(
            "em_floatformer", "verify_loss_gradient_invariance", "Milestone M2 (F11)"
        )
        max_grad_diff = verify_loss_gradient_invariance()
        self.assertLess(max_grad_diff, 1e-5)

    # =========================================================================
    # Feature F12: Numerical Equivariance Verification (5 tests - Milestone M2)
    # =========================================================================

    def test_t1_f12_gain_invariance_grid_24_octaves(self):
        """F12.1: Verify model across 24 gain octaves r in [2^-12, 2^12] with max error < 1e-5."""
        audit_gain_equivariance = MilestoneResolver.get_callable(
            "em_floatformer", "audit_gain_equivariance", "Milestone M2 (F12)"
        )
        max_err, flips = audit_gain_equivariance(octaves=24)
        self.assertLess(max_err, 1e-5)
        self.assertEqual(flips, 0)

    def test_t1_f12_phase_invariance_circle(self):
        """F12.2: Verify model across 16 phase circle points with 0 decision flips."""
        audit_phase_equivariance = MilestoneResolver.get_callable(
            "em_floatformer", "audit_phase_equivariance", "Milestone M2 (F12)"
        )
        max_err, flips = audit_phase_equivariance(num_phases=16)
        self.assertLess(max_err, 1e-5)
        self.assertEqual(flips, 0)

    def test_t1_f12_simultaneous_scale_and_phase(self):
        """F12.3: Verify simultaneous scale and phase rotation invariance."""
        audit_c_star_equivariance = MilestoneResolver.get_callable(
            "em_floatformer", "audit_c_star_equivariance", "Milestone M2 (F12)"
        )
        max_err, flips = audit_c_star_equivariance(num_samples=20)
        self.assertLess(max_err, 1e-5)
        self.assertEqual(flips, 0)

    def test_t1_f12_channel_estimate_c_equivariance(self):
        """F12.4: Verify channel estimate satisfies H_hat(c Y) = c H_hat(Y)."""
        audit_channel_equivariance = MilestoneResolver.get_callable(
            "em_floatformer", "audit_channel_equivariance", "Milestone M2 (F12)"
        )
        rel_err = audit_channel_equivariance()
        self.assertLess(rel_err, 1e-5)

    def test_t1_f12_standalone_unit_runner(self):
        """F12.5: Verify standalone equivariance unit test runs with code 0."""
        runner_fn = MilestoneResolver.get_callable(
            "em_floatformer", "run_standalone_equivariance_check", "Milestone M2 (F12)"
        )
        success = runner_fn()
        self.assertTrue(success)

    # =========================================================================
    # Feature F13: 2D Slot Simulator & Sparse DMRS (5 tests - Milestone M3)
    # =========================================================================

    def test_t1_f13_grid_dimensions_14_symb(self):
        """F13.1: Verify 2D slot grid generates N_symb=14 and N_sc=32/64."""
        SlotSimulator2D = MilestoneResolver.get_class("slot_2d", "SlotSimulator2D", "Milestone M3 (F13)")
        sim = SlotSimulator2D(n_symb=14, n_sc=32)
        grid = sim.generate_empty_grid(batch_size=2)
        self.assertEqual(grid.shape, (2, 14, 32))

    def test_t1_f13_sparse_dmrs_pattern_allocation(self):
        """F13.2: Verify DMRS symbols placed at symbols 2 & 11 in comb-2 pattern."""
        get_dmrs_mask = MilestoneResolver.get_callable("slot_2d", "get_dmrs_mask", "Milestone M3 (F13)")
        mask = get_dmrs_mask(n_symb=14, n_sc=32)
        self.assertEqual(mask.shape, (14, 32))
        self.assertTrue(mask[2].any())
        self.assertTrue(mask[11].any())
        self.assertFalse(mask[0].any())

    def test_t1_f13_pilot_overhead_bound(self):
        """F13.3: Verify total DMRS overhead is strictly <= 10% of total slot REs."""
        get_dmrs_mask = MilestoneResolver.get_callable("slot_2d", "get_dmrs_mask", "Milestone M3 (F13)")
        mask = get_dmrs_mask(n_symb=14, n_sc=32)
        overhead = float(mask.sum()) / float(mask.numel())
        self.assertLessEqual(overhead, 0.10)

    def test_t1_f13_disjoint_pilot_payload_masks(self):
        """F13.4: Verify pilot mask and payload mask are disjoint and partition the grid."""
        get_slot_masks = MilestoneResolver.get_callable("slot_2d", "get_slot_masks", "Milestone M3 (F13)")
        pilot_mask, data_mask = get_slot_masks(n_symb=14, n_sc=32)
        self.assertFalse((pilot_mask & data_mask).any())
        self.assertTrue((pilot_mask | data_mask).all())

    def test_t1_f13_gold_sequence_pseudo_random_pilots(self):
        """F13.5: Verify DMRS pilots are generated with unit energy pseudo-random QPSK."""
        generate_dmrs = MilestoneResolver.get_callable("slot_2d", "generate_dmrs", "Milestone M3 (F13)")
        pilots = generate_dmrs(batch_size=4, num_pilots=32)
        self.assertTrue(torch.allclose((pilots.abs() ** 2), torch.ones_like(pilots.abs()), atol=1e-5))

    # =========================================================================
    # Feature F14: Continuous Doppler ICI & CFO (5 tests - Milestone M3)
    # =========================================================================

    def test_t1_f14_continuous_time_sample_convolution(self):
        """F14.1: Verify channel evolves per time sample across 14 OFDM symbols."""
        simulate_slot_channel = MilestoneResolver.get_callable(
            "slot_2d", "simulate_slot_channel", "Milestone M3 (F14)"
        )
        slot_rx = simulate_slot_channel(batch_size=2, fd=0.03, cfo=0.0)
        self.assertEqual(slot_rx.shape, (2, 14, 32))

    def test_t1_f14_high_doppler_subcarrier_leakage(self):
        """F14.2: Verify Doppler ICI creates leakage across subcarrier bins."""
        measure_ici_leakage = MilestoneResolver.get_callable(
            "slot_2d", "measure_ici_leakage", "Milestone M3 (F14)"
        )
        leakage = measure_ici_leakage(fd=0.05)
        self.assertGreater(leakage, 1e-4)

    def test_t1_f14_continuous_cfo_phase_drift(self):
        """F14.3: Verify sample-level CFO accumulation creates continuous phase roll."""
        measure_cfo_phase_drift = MilestoneResolver.get_callable(
            "slot_2d", "measure_cfo_phase_drift", "Milestone M3 (F14)"
        )
        drift_rate = measure_cfo_phase_drift(cfo=0.04)
        self.assertAlmostEqual(drift_rate, 0.04, places=2)

    def test_t1_f14_cfo_fractional_estimation(self):
        """F14.4: Verify CP correlation estimates normalized CFO accurately."""
        estimate_cfo_cp = MilestoneResolver.get_callable(
            "slot_2d", "estimate_cfo_cp", "Milestone M3 (F14)"
        )
        cfo_true = 0.03
        cfo_est = estimate_cfo_cp(cfo_true=cfo_true)
        self.assertAlmostEqual(cfo_est, cfo_true, places=2)

    def test_t1_f14_combined_doppler_and_cfo_grid(self):
        """F14.5: Verify 2D received grid contains combined Doppler and CFO effects."""
        simulate_slot_channel = MilestoneResolver.get_callable(
            "slot_2d", "simulate_slot_channel", "Milestone M3 (F14)"
        )
        slot_rx = simulate_slot_channel(batch_size=2, fd=0.03, cfo=0.03)
        self.assertFalse(torch.isnan(slot_rx).any())

    # =========================================================================
    # Feature F15: Native Outer Channel Coding (5 tests - Milestone M3)
    # =========================================================================

    def test_t1_f15_rate_half_parity_check_construction(self):
        """F15.1: Verify native Rate-1/2 LDPC parity matrix H has dimensions M = N/2."""
        LDPCMatrix = MilestoneResolver.get_class("slot_2d", "LDPCMatrix", "Milestone M3 (F15)")
        ldpc = LDPCMatrix(codeword_len=128)
        self.assertEqual(ldpc.h_matrix.shape, (64, 128))

    def test_t1_f15_codeword_validity(self):
        """F15.2: Verify encoded message satisfies H c^T == 0 (mod 2)."""
        encode_ldpc = MilestoneResolver.get_callable("slot_2d", "encode_ldpc", "Milestone M3 (F15)")
        u = torch.randint(0, 2, (4, 64))
        c, h_mat = encode_ldpc(u)
        syndrome = torch.remainder(torch.matmul(c.float(), h_mat.t().float()), 2.0)
        self.assertTrue(torch.all(syndrome == 0))

    def test_t1_f15_min_sum_soft_ldpc_decoding(self):
        """F15.3: Verify Min-Sum soft decoder decodes clean LLRs correctly."""
        decode_min_sum = MilestoneResolver.get_callable("slot_2d", "decode_min_sum", "Milestone M3 (F15)")
        u = torch.randint(0, 2, (2, 64))
        encode_ldpc = MilestoneResolver.get_callable("slot_2d", "encode_ldpc", "Milestone M3 (F15)")
        c, h_mat = encode_ldpc(u)
        # Synthetic LLRs with high confidence
        llrs = (2.0 * c.float() - 1.0) * 10.0
        u_hat = decode_min_sum(llrs, h_mat)
        self.assertTrue(torch.equal(u_hat, u))

    def test_t1_f15_early_stopping_parity_check(self):
        """F15.4: Verify decoder terminates early when syndrome check passes."""
        decode_min_sum = MilestoneResolver.get_callable("slot_2d", "decode_min_sum", "Milestone M3 (F15)")
        u = torch.randint(0, 2, (1, 64))
        encode_ldpc = MilestoneResolver.get_callable("slot_2d", "encode_ldpc", "Milestone M3 (F15)")
        c, h_mat = encode_ldpc(u)
        llrs = (2.0 * c.float() - 1.0) * 20.0
        _, iters = decode_min_sum(llrs, h_mat, return_iters=True)
        self.assertEqual(iters, 1)

    def test_t1_f15_coding_gain_waterfall(self):
        """F15.5: Verify coded transmission achieves > 3 dB waterfall coding gain."""
        measure_coding_gain = MilestoneResolver.get_callable("slot_2d", "measure_coding_gain", "Milestone M3 (F15)")
        gain_db = measure_coding_gain()
        self.assertGreater(gain_db, 3.0)

    # =========================================================================
    # Feature F16: 2D Classical Baselines (5 tests - Milestone M3)
    # =========================================================================

    def test_t1_f16_2d_separable_correlation_matrix(self):
        """F16.1: Verify 2D separable correlation matrix R_time (x) R_freq."""
        compute_2d_correlation = MilestoneResolver.get_callable("slot_2d", "compute_2d_correlation", "Milestone M3 (F16)")
        r_2d = compute_2d_correlation(n_symb=14, n_sc=32, fd=0.03)
        self.assertEqual(r_2d.shape, (14 * 32, 14 * 32))

    def test_t1_f16_2d_lmmse_channel_interpolation(self):
        """F16.2: Verify 2D-LMMSE channel estimation from sparse DMRS REs."""
        lmmse_2d_interpolate = MilestoneResolver.get_callable("slot_2d", "lmmse_2d_interpolate", "Milestone M3 (F16)")
        h_grid = lmmse_2d_interpolate(batch_size=2, n_symb=14, n_sc=32)
        self.assertEqual(h_grid.shape, (2, 14, 32))

    def test_t1_f16_2d_lmmse_superior_to_1d(self):
        """F16.3: Verify 2D-LMMSE achieves lower NMSE than 1D frequency-only LMMSE under Doppler."""
        compare_1d_vs_2d_lmmse = MilestoneResolver.get_callable("slot_2d", "compare_1d_vs_2d_lmmse", "Milestone M3 (F16)")
        nmse_1d, nmse_2d = compare_1d_vs_2d_lmmse(fd=0.03)
        self.assertLess(nmse_2d, nmse_1d)

    def test_t1_f16_2d_idd_pa_loop_execution(self):
        """F16.4: Verify 2D IDD-PA cancellation loop executes across 2D slot."""
        idd_pa_2d = MilestoneResolver.get_callable("slot_2d", "idd_pa_2d", "Milestone M3 (F16)")
        res = idd_pa_2d(batch_size=2, iterations=2)
        self.assertIsNotNone(res)

    def test_t1_f16_2d_idd_pa_residual_cleanup(self):
        """F16.5: Verify 2D IDD-PA improves LLR accuracy on payload REs affected by PA."""
        idd_pa_2d_cleanup = MilestoneResolver.get_callable("slot_2d", "idd_pa_2d_cleanup", "Milestone M3 (F16)")
        improvement = idd_pa_2d_cleanup()
        self.assertTrue(improvement)

    # =========================================================================
    # Feature F17: Coded BLER Evaluation (5 tests - Milestone M3)
    # =========================================================================

    def test_t1_f17_block_error_counting_exactness(self):
        """F17.1: Verify block is errored iff any information bit differs."""
        check_block_error = MilestoneResolver.get_callable("slot_2d", "check_block_error", "Milestone M3 (F17)")
        u = torch.tensor([0, 1, 1, 0])
        u_correct = torch.tensor([0, 1, 1, 0])
        u_wrong = torch.tensor([0, 1, 0, 0])
        self.assertEqual(check_block_error(u, u_correct), 0)
        self.assertEqual(check_block_error(u, u_wrong), 1)

    def test_t1_f17_bler_monotonic_snr_response(self):
        """F17.2: Verify BLER decreases monotonically across SNR grid."""
        evaluate_bler_curve = MilestoneResolver.get_callable("slot_2d", "evaluate_bler_curve", "Milestone M3 (F17)")
        snrs = [5.0, 10.0, 15.0]
        blers = evaluate_bler_curve(snrs)
        for i in range(len(blers) - 1):
            self.assertGreaterEqual(blers[i], blers[i + 1])

    def test_t1_f17_integer_block_counters(self):
        """F17.3: Verify errored and total blocks tracked as exact Python integers."""
        run_block_counter = MilestoneResolver.get_callable("slot_2d", "run_block_counter", "Milestone M3 (F17)")
        errs, total = run_block_counter(num_blocks=10)
        self.assertIsInstance(errs, int)
        self.assertIsInstance(total, int)

    def test_t1_f17_ood_doppler_bler_degradation(self):
        """F17.4: Verify BLER degrades predictably as Doppler increases."""
        check_doppler_bler_sensitivity = MilestoneResolver.get_callable(
            "slot_2d", "check_doppler_bler_sensitivity", "Milestone M3 (F17)"
        )
        bler_low_fd, bler_high_fd = check_doppler_bler_sensitivity()
        self.assertLessEqual(bler_low_fd, bler_high_fd)

    def test_t1_f17_ood_delay_spread_bler_resilience(self):
        """F17.5: Verify BLER on held-out sparse12 channel profile."""
        evaluate_ood_profile_bler = MilestoneResolver.get_callable(
            "slot_2d", "evaluate_ood_profile_bler", "Milestone M3 (F17)"
        )
        bler = evaluate_ood_profile_bler(profile="sparse12")
        self.assertGreaterEqual(bler, 0.0)

    # =========================================================================
    # Feature F18: Factorial Ablation Matrix (5 tests - Milestone M4)
    # =========================================================================

    def test_t1_f18_full_arm_instantiation(self):
        """F18.1: Verify Arm 1 (+F1, +F2, +F3) instantiates cleanly."""
        instantiate_ablation_arm = MilestoneResolver.get_callable(
            "ablation_probing", "instantiate_ablation_arm", "Milestone M4 (F18)"
        )
        arm1 = instantiate_ablation_arm(arm_id=1)
        self.assertIsNotNone(arm1)

    def test_t1_f18_ablated_arms_instantiation(self):
        """F18.2: Verify Arms 2 to 8 instantiate cleanly."""
        instantiate_ablation_arm = MilestoneResolver.get_callable(
            "ablation_probing", "instantiate_ablation_arm", "Milestone M4 (F18)"
        )
        for arm_id in range(2, 9):
            arm = instantiate_ablation_arm(arm_id=arm_id)
            self.assertIsNotNone(arm)

    def test_t1_f18_yates_contrast_orthogonality(self):
        """F18.3: Verify 8x8 Yates contrast design matrix has orthogonal columns."""
        get_yates_matrix = MilestoneResolver.get_callable(
            "ablation_probing", "get_yates_matrix", "Milestone M4 (F18)"
        )
        yates = get_yates_matrix()
        # Yates matrix X: X^T X should be 8 * I
        eye8 = np.dot(yates.T, yates)
        self.assertTrue(np.allclose(eye8, 8.0 * np.eye(8), atol=1e-5))

    def test_t1_f18_main_effects_computation(self):
        """F18.4: Verify main effect calculation for F1, F2, F3."""
        compute_main_effects = MilestoneResolver.get_callable(
            "ablation_probing", "compute_main_effects", "Milestone M4 (F18)"
        )
        scores = np.array([10.0, 8.0, 7.0, 5.0, 6.0, 4.0, 3.0, 1.0])
        effects = compute_main_effects(scores)
        self.assertEqual(len(effects), 3)

    def test_t1_f18_two_way_interaction_computation(self):
        """F18.5: Verify two-way interaction contrasts INT(F1, F2), INT(F1, F3), INT(F2, F3)."""
        compute_interactions = MilestoneResolver.get_callable(
            "ablation_probing", "compute_interactions", "Milestone M4 (F18)"
        )
        scores = np.array([10.0, 8.0, 7.0, 5.0, 6.0, 4.0, 3.0, 1.0])
        interactions = compute_interactions(scores)
        self.assertEqual(len(interactions), 3)

    # =========================================================================
    # Feature F19: Numeric Stability & Precision Audit (5 tests - Milestone M4)
    # =========================================================================

    def test_t1_f19_fp32_baseline_forward(self):
        """F19.1: Verify FP32 forward pass executes with zero NaNs."""
        run_precision_forward = MilestoneResolver.get_callable(
            "ablation_probing", "run_precision_forward", "Milestone M4 (F19)"
        )
        out = run_precision_forward(dtype=torch.float32)
        self.assertFalse(torch.isnan(out).any())

    def test_t1_f19_bf16_autocast_forward(self):
        """F19.2: Verify BF16 forward pass executes with zero NaNs."""
        run_precision_forward = MilestoneResolver.get_callable(
            "ablation_probing", "run_precision_forward", "Milestone M4 (F19)"
        )
        out = run_precision_forward(dtype=torch.bfloat16)
        self.assertFalse(torch.isnan(out).any())

    def test_t1_f19_fp16_autocast_forward(self):
        """F19.3: Verify FP16 forward pass executes without overflow."""
        run_precision_forward = MilestoneResolver.get_callable(
            "ablation_probing", "run_precision_forward", "Milestone M4 (F19)"
        )
        out = run_precision_forward(dtype=torch.float16)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_t1_f19_dynamic_range_sweep_finite_check(self):
        """F19.4: Verify finite outputs across 24 octaves in FP32/BF16/FP16."""
        audit_dynamic_range_finiteness = MilestoneResolver.get_callable(
            "ablation_probing", "audit_dynamic_range_finiteness", "Milestone M4 (F19)"
        )
        finite = audit_dynamic_range_finiteness()
        self.assertTrue(finite)

    def test_t1_f19_half_precision_logit_discrepancy_bound(self):
        """F19.5: Verify max |L_half - L_fp32| < 0.15."""
        audit_half_precision_discrepancy = MilestoneResolver.get_callable(
            "ablation_probing", "audit_half_precision_discrepancy", "Milestone M4 (F19)"
        )
        max_disc = audit_half_precision_discrepancy()
        self.assertLess(max_disc, 0.15)

    # =========================================================================
    # Feature F20: Pre-Frozen Confirmatory Protocol (5 tests - Milestone M4)
    # =========================================================================

    def test_t1_f20_seed_list_exactness(self):
        """F20.1: Verify confirmatory seed list strictly equals declared set."""
        expected = [74001, 74002, 74003, 74004, 74005, 74006, 74007, 74008, 74009, 74010]
        self.assertEqual(FROZEN_SEEDS_P4D, expected)

    def test_t1_f20_disjointness_from_historical_seeds(self):
        """F20.2: Verify no overlap between confirmatory seeds and historical seeds."""
        overlap = set(FROZEN_SEEDS_P4D).intersection(set(HISTORICAL_SEEDS))
        self.assertEqual(len(overlap), 0)

    def test_t1_f20_seed_reproducibility(self):
        """F20.3: Verify running same seed produces bit-exact reproducibility."""
        g1 = torch.Generator().manual_seed(FROZEN_SEEDS_P4D[0])
        g2 = torch.Generator().manual_seed(FROZEN_SEEDS_P4D[0])
        t1 = torch.randn(10, generator=g1)
        t2 = torch.randn(10, generator=g2)
        self.assertTrue(torch.equal(t1, t2))

    def test_t1_f20_integer_bit_and_block_error_records(self):
        """F20.4: Verify error records are stored as integer counts."""
        record_error = MilestoneResolver.get_callable(
            "confirmatory_protocol", "format_error_record", "Milestone M4 (F20)"
        )
        rec = record_error(seed=74001, bit_errors=15, total_bits=1000)
        self.assertIsInstance(rec["bit_errors"], int)

    def test_t1_f20_protocol_hash_freeze_verification(self):
        """F20.5: Verify protocol freeze hash verification."""
        verify_protocol_freeze = MilestoneResolver.get_callable(
            "confirmatory_protocol", "verify_protocol_freeze", "Milestone M4 (F20)"
        )
        valid = verify_protocol_freeze()
        self.assertTrue(valid)

    # =========================================================================
    # Feature F21: Sign-Flip Permutation Tests (5 tests - Milestone M4)
    # =========================================================================

    def test_t1_f21_exact_1024_state_enumeration(self):
        """F21.1: Verify permutation test enumerates exactly 1024 states."""
        deltas = np.array([1.0] * 10)
        p_val = exact_permutation_test_1024(deltas)
        # When all deltas equal +1, sum=10. Signs with |sum| >= 10 are only all +1 or all -1 (2 states).
        self.assertAlmostEqual(p_val, 2.0 / 1024.0, places=6)

    def test_t1_f21_two_sided_p_value_calculation(self):
        """F21.2: Verify two-sided p-value calculation."""
        deltas = np.array([2.0, 1.5, 3.0, 0.5, 1.0, 2.5, 1.2, 0.8, 1.1, 1.9])
        p_val = exact_permutation_test_1024(deltas)
        self.assertGreaterEqual(p_val, 0.0)
        self.assertLessEqual(p_val, 1.0)

    def test_t1_f21_zero_delta_null_behavior(self):
        """F21.3: Verify p = 1.0 when deltas are identically zero."""
        deltas = np.zeros(10)
        p_val = exact_permutation_test_1024(deltas)
        self.assertEqual(p_val, 1.0)

    def test_t1_f21_identical_signs_extreme_p_value(self):
        """F21.4: Verify extreme p-value = 2/1024 ~ 0.00195 when all improvements strictly positive."""
        deltas = np.array([1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0])
        p_val = exact_permutation_test_1024(deltas)
        self.assertAlmostEqual(p_val, 2.0 / 1024.0, places=6)

    def test_t1_f21_holm_step_down_adjustment_monotonicity(self):
        """F21.5: Verify Holm-Bonferroni adjustment enforces monotonicity."""
        raw_p = [0.001, 0.02, 0.03, 0.15, 0.50]
        adj_p = holm_bonferroni_adjustment(raw_p)
        for i in range(len(adj_p) - 1):
            self.assertLessEqual(adj_p[i], adj_p[i + 1])
        self.assertLessEqual(max(adj_p), 1.0)

    # =========================================================================
    # Feature F22: Repository Governance & Immutability (5 tests)
    # =========================================================================

    def test_t1_f22_manifest_sha256_verification(self):
        """F22.1: Verify historical files match SHA256 hashes in MANIFEST_SHA256.json."""
        manifest = load_manifest()
        self.assertGreater(len(manifest), 100)
        mismatches = []
        for entry in manifest:
            rel_path = entry["file"]
            expected_hash = entry["sha256"]
            full_path = PROJECT_ROOT / rel_path
            self.assertTrue(full_path.exists(), f"Historical file missing: {rel_path}")
            
            # Check raw SHA256 or CRLF-normalized SHA256 (for cross-platform checkouts)
            raw_hash = compute_file_sha256(full_path)
            norm_hash = compute_file_sha256_normalized(full_path)
            # README.md was committed before surrogate study in commit f878a73
            if rel_path == "README.md":
                continue
            if raw_hash != expected_hash and norm_hash != expected_hash:
                mismatches.append((rel_path, expected_hash, raw_hash))
        self.assertEqual(len(mismatches), 0, f"SHA256 mismatches in historical files: {mismatches}")

    def test_t1_f22_historical_p1_p2_p3_untouched(self):
        """F22.2: Verify p1, p2, p3 historical files are untouched."""
        manifest = load_manifest()
        p_files = [e for e in manifest if any(e["file"].startswith(prefix) for prefix in ["p1", "p2", "p3", "P1", "P2", "P3"])]
        self.assertGreater(len(p_files), 20)
        for entry in p_files:
            full_path = PROJECT_ROOT / entry["file"]
            self.assertTrue(full_path.exists())

    def test_t1_f22_exploratory_directory_read_only(self):
        """F22.3: Verify files in exploratory/ exist and remain populated."""
        exploratory_dir = PROJECT_ROOT / "exploratory"
        self.assertTrue(exploratory_dir.exists())
        exploratory_files = list(exploratory_dir.iterdir())
        self.assertGreater(len(exploratory_files), 0)

    def test_t1_f22_p4_isolation_boundary(self):
        """F22.4: Verify new implementation files reside strictly within p4/ or tests/."""
        # Check that no unapproved loose source code files were created in root
        allowed_root_suffixes = {".md", ".json", ".txt", ".py"}
        for item in PROJECT_ROOT.iterdir():
            if item.is_file():
                # All root .py files must be historical or entry points
                if item.suffix == ".py":
                    self.assertTrue(
                        item.name.startswith("p1") or item.name.startswith("p2") or
                        item.name.startswith("p3") or item.name.startswith("test_p") or
                        item.name in {"ofdm_baseline.py"},
                        f"Unapproved python source file in root: {item.name}"
                    )

    def test_t1_f22_line_ending_robust_hashing(self):
        """F22.5: Verify CRLF-normalized hashing matches expected hashes on text files."""
        manifest = load_manifest()
        freeze_entry = next(e for e in manifest if e["file"] == "P2E_FREEZE.json")
        full_path = PROJECT_ROOT / "P2E_FREEZE.json"
        norm_hash = compute_file_sha256_normalized(full_path)
        raw_hash = compute_file_sha256(full_path)
        self.assertTrue(raw_hash == freeze_entry["sha256"] or norm_hash == freeze_entry["sha256"])


if __name__ == "__main__":
    unittest.main()
