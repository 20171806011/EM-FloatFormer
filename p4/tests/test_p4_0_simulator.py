"""Comprehensive Unit Tests for Stage P4-0 (Features F1 through F7).

Verifies:
- F1: Oversampled Rapp PA (AM-AM, AM-PM, oversampling, linear limit).
- F2: Dynamic Multipath Simulator (Doppler ICI phase evolution, off-diagonal leakage).
- F3: Static-Limit Parity Check (relative error < 1e-5 under zero Doppler and linear PA).
- F4: Bussgang Decomposition & LMMSE Equalization.
- F5: Calibrated 16-QAM Soft LLRs (calibration bias, ECE, sign agreement).
- F6: Iterative Decision-Directed PA Cancellation (DD-PA error reduction under A=1.25).
- F7: Classical Latency Benchmark Execution and Artifact Validation.
"""

import math
import os
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

import p4
from p4.classical_baselines import (
    bussgang_lmmse,
    bussgang_params,
    calibrated_16qam_llrs,
    delay_est_dft,
    hard_slice_16qam,
    iterative_dd_pa_cancellation,
    ls_est,
    soft_slice_16qam,
)
from p4.simulator import (
    DEFAULT_CP,
    DEFAULT_N,
    PROFILES,
    SIGMA_16QAM,
    THRESH_16QAM,
    DynamicChannelSimulator,
    apply_rapp_pa_samples,
    decimate_time_to_frequency,
    generate_frame,
    oversample_frequency_to_time,
    q16,
    qpsk,
    rapp_am_am,
    rapp_am_pm,
    rapp_pa,
    static_limit_parity_check,
)


class TestP40SimulatorAndBaselines(unittest.TestCase):
    """Test suite covering F1 to F7 for Stage P4-0."""

    def setUp(self):
        self.device = "cpu"
        self.generator = torch.Generator(device=self.device).manual_seed(42)

    # -----------------------------------------------------------------------
    # F1: Oversampled Rapp PA Model
    # -----------------------------------------------------------------------

    def test_f1_rapp_am_am_compression(self):
        """Verify AM-AM amplitude compression behavior at A=2.25 and A=1.25."""
        r = torch.linspace(0.1, 10.0, 100)
        # Monotonically increasing but compressed
        for a in [1.25, 2.25]:
            g_r = rapp_am_am(r, A=a, knee=2.0)
            self.assertTrue(torch.all(g_r <= r), f"g(r) must be <= r for A={a}")
            self.assertTrue(torch.all(g_r < a), f"g(r) must be strictly < A for A={a}")
            # Large r should approach saturation A
            self.assertAlmostEqual(g_r[-1].item(), a, delta=0.05)

    def test_f1_rapp_am_pm_phase_distortion(self):
        """Verify AM-PM phase rotation behavior."""
        r = torch.linspace(0.0, 10.0, 100)
        phi_r = rapp_am_pm(r, alpha=0.5, beta=1.0)
        self.assertAlmostEqual(phi_r[0].item(), 0.0, places=6)
        self.assertTrue(torch.all(phi_r >= 0.0))
        # Asymptote: alpha * beta^2 = 0.5 * 1.0 = 0.5
        self.assertAlmostEqual(phi_r[-1].item(), 0.5, delta=0.05)

        # When alpha=0, phase is zero
        phi_zero = rapp_am_pm(r, alpha=0.0)
        self.assertTrue(torch.all(phi_zero == 0.0))

    def test_f1_rapp_linear_limit_identity(self):
        """Verify that when A is None/inf and alpha=0, PA is an exact identity."""
        x = torch.randn(10, DEFAULT_N, dtype=torch.complex64)
        x_pa = rapp_pa(x, knee=2.0, A=None, alpha=0.0, oversample=4)
        diff = (x_pa - x).abs().max().item()
        self.assertLess(diff, 1e-6, f"Linear PA must preserve signal, max diff={diff}")

    def test_f1_oversampling_spectral_containment(self):
        """Verify that oversampled PA processing preserves in-band tones under linear conditions."""
        X_freq = torch.randn(4, DEFAULT_N, dtype=torch.complex64)
        x_ov = oversample_frequency_to_time(X_freq, oversample=4)
        X_rec = decimate_time_to_frequency(x_ov, n=DEFAULT_N, oversample=4)
        diff = (X_rec - X_freq).abs().max().item()
        self.assertLess(diff, 1e-6, f"Zero-padding oversampling must be perfectly invertible in-band: {diff}")

    # -----------------------------------------------------------------------
    # F2: Dynamic Multipath Simulator & Doppler ICI
    # -----------------------------------------------------------------------

    def test_f2_dynamic_doppler_ici_leakage(self):
        """Verify that continuous Doppler evolution introduces ICI off-diagonal power."""
        sim_static = DynamicChannelSimulator(n=32, cp=16, profile="equal4", fd=0.0)
        sim_low_fd = DynamicChannelSimulator(n=32, cp=16, profile="equal4", fd=0.02)
        sim_high_fd = DynamicChannelSimulator(n=32, cp=16, profile="equal4", fd=0.08)

        g = torch.Generator().manual_seed(999)
        ch_static = sim_static.draw_channel(1, generator=g)
        ch_low = sim_low_fd.draw_channel(1, generator=g)
        ch_high = sim_high_fd.draw_channel(1, generator=g)

        H_mat_static = sim_static.compute_transfer_matrix(ch_static)[0]
        H_mat_low = sim_low_fd.compute_transfer_matrix(ch_low)[0]
        H_mat_high = sim_high_fd.compute_transfer_matrix(ch_high)[0]

        # In static channel, off-diagonal elements must be near zero
        eye_mask = torch.eye(32, dtype=torch.bool)
        off_diag_static = H_mat_static[~eye_mask].abs().square().sum().item()
        self.assertLess(off_diag_static, 1e-6, "Static channel transfer matrix must be strictly diagonal")

        # Under Doppler, off-diagonal ICI energy must be strictly positive
        off_diag_low = H_mat_low[~eye_mask].abs().square().sum().item()
        off_diag_high = H_mat_high[~eye_mask].abs().square().sum().item()

        self.assertGreater(off_diag_low, 1e-4, "Low Doppler must produce non-zero ICI off-diagonal energy")
        self.assertGreater(off_diag_high, off_diag_low, "Higher Doppler must produce greater ICI leakage")

    def test_f2_continuous_time_filter_matches_matrix(self):
        """Verify continuous time-domain FIR convolution matches the transfer matrix output."""
        sim = DynamicChannelSimulator(n=32, cp=16, profile="train3", fd=0.05)
        ch = sim.draw_channel(1, generator=self.generator)
        H_mat = sim.compute_transfer_matrix(ch)[0]

        X = torch.randn(1, 32, dtype=torch.complex64)
        Y_mat = (H_mat @ X.transpose(0, 1)).transpose(0, 1)

        x_time = torch.fft.ifft(X, norm="ortho")
        x_cp = torch.cat([x_time[:, -16:], x_time], dim=-1)
        y_time = sim.filter_time_signal(x_cp, ch, noise_var=None)
        Y_sim = torch.fft.fft(y_time[:, 16:], norm="ortho")

        diff = (Y_sim - Y_mat).abs().max().item()
        self.assertLess(diff, 1e-5, f"Time-domain FIR and analytical transfer matrix must match: {diff}")

    # -----------------------------------------------------------------------
    # F3: Static-Limit Parity Check
    # -----------------------------------------------------------------------

    def test_f3_static_limit_parity_all_profiles(self):
        """Verify exact parity with Y = H * X across all standard channel profiles."""
        for name in PROFILES:
            rel_err = static_limit_parity_check(
                n=DEFAULT_N,
                cp=DEFAULT_CP,
                batch_size=128,
                profile=name,
                generator=self.generator,
            )
            self.assertLess(
                rel_err,
                1e-5,
                f"Static parity check failed for profile '{name}': rel_err={rel_err:.2e} >= 1e-5",
            )

    def test_f3_static_limit_parity_n64(self):
        """Verify static parity check on N=64 subcarriers."""
        rel_err = static_limit_parity_check(
            n=64,
            cp=16,
            batch_size=64,
            profile="equal4",
            generator=self.generator,
        )
        self.assertLess(
            rel_err,
            1e-5,
            f"Static parity check failed for N=64: rel_err={rel_err:.2e} >= 1e-5",
        )

    # -----------------------------------------------------------------------
    # F4: Bussgang Decomposition & LMMSE Equalization
    # -----------------------------------------------------------------------

    def test_f4_bussgang_quadrature_vs_monte_carlo(self):
        """Validate 1D analytical quadrature against 500k-sample empirical Monte Carlo."""
        for a in [2.25, 1.25]:
            gamma_quad, sigma_d2_quad, p_out_quad = bussgang_params(A=a, alpha=0.5, beta=1.0)

            # Empirical Monte Carlo
            g = torch.Generator().manual_seed(1234)
            x = (torch.randn(500000, generator=g) + 1j * torch.randn(500000, generator=g)) / math.sqrt(2.0)
            x_pa = apply_rapp_pa_samples(x, A=a, alpha=0.5, beta=1.0)

            gamma_mc = (x_pa * x.conj()).mean().item()
            p_out_mc = (x_pa.abs() ** 2).mean().item()
            sigma_d2_mc = p_out_mc - abs(gamma_mc) ** 2

            self.assertAlmostEqual(gamma_quad.real, gamma_mc.real, delta=0.01)
            self.assertAlmostEqual(gamma_quad.imag, gamma_mc.imag, delta=0.01)
            self.assertAlmostEqual(sigma_d2_quad, sigma_d2_mc, delta=0.005)

    def test_f4_bussgang_lmmse_beats_linear_under_strong_pa(self):
        """Verify that Bussgang-LMMSE outperforms uncalibrated linear LMMSE in MSE and calibrated LLR BCE under strong PA."""
        frame = generate_frame(
            batch_size=800,
            n=DEFAULT_N,
            cp=DEFAULT_CP,
            profile="equal4",
            snr_db=18.0,
            pa_condition="strong",
            A=1.25,
            generator=self.generator,
        )

        # Baseline 1: Naive linear LMMSE (ignoring PA distortion: assumes noise is only nv)
        h_eff = delay_est_dft(frame.yp, taps=8)
        denom_lin = h_eff.abs() ** 2 + frame.noise_var
        w_lin = h_eff.conj() / denom_lin
        mse_lin = (w_lin * frame.yd - frame.data_tx).abs().square().mean().item()

        # Naive uncalibrated LLRs ignoring distortion variance
        sigma2_naive = frame.noise_var / (h_eff.abs() ** 2 + 1e-12)
        llrs_naive = calibrated_16qam_llrs(frame.yd / h_eff, sigma2_naive)
        bce_lin = F.binary_cross_entropy_with_logits(llrs_naive, frame.bits.float()).item()

        # Baseline 2: Bussgang-LMMSE
        res_buss = bussgang_lmmse(
            frame.yp,
            frame.yd,
            frame.noise_var,
            A=1.25,
            est_method="dft8",
        )
        mse_buss = (res_buss.w_filter * frame.yd - frame.data_tx).abs().square().mean().item()
        bce_buss = F.binary_cross_entropy_with_logits(res_buss.llrs, frame.bits.float()).item()

        # Verify Bussgang achieves strictly lower symbol MSE
        self.assertLess(
            mse_buss,
            mse_lin,
            f"Bussgang-LMMSE symbol MSE ({mse_buss:.5f}) must be strictly lower than naive linear LMMSE ({mse_lin:.5f})",
        )

        # Verify Bussgang achieves strictly lower LLR BCE due to calibrated variance
        self.assertLess(
            bce_buss,
            bce_lin,
            f"Bussgang-LMMSE LLR BCE ({bce_buss:.4f}) must be strictly lower than naive linear LLR BCE ({bce_lin:.4f})",
        )

    # -----------------------------------------------------------------------
    # F5: Calibrated 16-QAM Soft LLRs
    # -----------------------------------------------------------------------

    def test_f5_16qam_llr_calibration_on_awgn(self):
        """Verify empirical calibration bias and Expected Calibration Error (ECE) < 0.02."""
        num_samples = 200000
        bits = torch.randint(0, 2, (num_samples, 1, 4), generator=self.generator)
        symbols = q16(bits).squeeze(1)

        sigma2 = torch.tensor(0.2)  # SNR ~ 7 dB
        noise = (torch.randn(num_samples, generator=self.generator) + 1j * torch.randn(num_samples, generator=self.generator)) * math.sqrt(0.1)
        z = symbols + noise

        llrs = calibrated_16qam_llrs(z.unsqueeze(1), sigma2.unsqueeze(0), method="exact").squeeze(1)
        probs = torch.sigmoid(llrs)

        # Check mean calibration bias across all 4 bits
        for bit_idx in range(4):
            empirical_rate = bits[:, 0, bit_idx].float().mean().item()
            predicted_prob = probs[:, bit_idx].mean().item()
            bias = abs(empirical_rate - predicted_prob)
            self.assertLess(bias, 0.01, f"Bit {bit_idx} calibration bias {bias:.4f} exceeds 0.01")

        # Check ECE
        def compute_ece(p, labels, n_bins=10):
            bins = torch.linspace(0, 1, n_bins + 1)
            ece = 0.0
            for i in range(n_bins):
                mask = (p >= bins[i]) & (p < bins[i + 1])
                if mask.sum() > 0:
                    acc = labels[mask].float().mean()
                    conf = p[mask].mean()
                    ece += (acc - conf).abs() * (mask.sum().float() / p.numel())
            return ece.item()

        for bit_idx in range(4):
            ece = compute_ece(probs[:, bit_idx], bits[:, 0, bit_idx])
            self.assertLess(ece, 0.015, f"Bit {bit_idx} ECE {ece:.4f} exceeds 0.015")

    def test_f5_llr_sign_matches_hard_slicing(self):
        """Verify that sign of maxlog LLRs perfectly agrees with hard bit slicing, and exact agrees >99.9%."""
        z = torch.randn(100, 32, dtype=torch.complex64, generator=self.generator)
        sigma2 = torch.full((100, 32), 0.1)

        # Max-log LLRs strictly adhere to the Voronoi boundary: 100% agreement
        llrs_maxlog = calibrated_16qam_llrs(z, sigma2, method="maxlog")
        hard_from_llr = (llrs_maxlog > 0).long()
        hard_from_slice = hard_slice_16qam(z)
        self.assertTrue(torch.all(hard_from_llr == hard_from_slice))

        # Exact Bayesian posterior LLR incorporates all 4 Gaussian tails,
        # shifting the zero-crossing boundary slightly (~2.6e-5): agreement > 99.9%
        llrs_exact = calibrated_16qam_llrs(z, sigma2, method="exact")
        hard_from_exact = (llrs_exact > 0).long()
        agreement = (hard_from_exact == hard_from_slice).float().mean().item()
        self.assertGreater(agreement, 0.999)

    # -----------------------------------------------------------------------
    # F6: Iterative Decision-Directed PA Cancellation (DD-PA)
    # -----------------------------------------------------------------------

    def test_f6_iterative_dd_pa_error_reduction(self):
        """Verify monotonic or significant BER and MSE reduction across DD-PA iterations under strong PA (A=1.25)."""
        frame = generate_frame(
            batch_size=800,
            n=DEFAULT_N,
            cp=DEFAULT_CP,
            profile="equal4",
            snr_db=20.0,
            pa_condition="strong",
            A=1.25,
            generator=self.generator,
        )

        h_hat, llrs_final, bits_final, history = iterative_dd_pa_cancellation(
            frame.yp,
            frame.yd,
            frame.noise_var,
            A=1.25,
            iterations=3,
            est_method="dft8",
        )

        self.assertEqual(len(history), 4)  # iter 0, 1, 2, 3

        bers = []
        mses = []
        for rec in history:
            ber = (rec.hard_bits != frame.bits).float().mean().item()
            mse = (rec.z_hat - frame.data_tx).abs().square().mean().item()
            bers.append(ber)
            mses.append(mse)

        # DD-PA Iter 1 must reduce BER compared to Iter 0 (Bussgang)
        self.assertLess(
            bers[1],
            bers[0],
            f"DD-PA Iter 1 BER ({bers[1]:.4f}) must be lower than Iter 0 ({bers[0]:.4f})",
        )
        self.assertLess(
            mses[1],
            mses[0],
            f"DD-PA Iter 1 MSE ({mses[1]:.4f}) must be lower than Iter 0 ({mses[0]:.4f})",
        )

        # Iter 2 must improve or match Iter 1
        self.assertLessEqual(
            bers[2],
            bers[1] + 1e-4,
            f"DD-PA Iter 2 BER ({bers[2]:.4f}) should not degrade compared to Iter 1 ({bers[1]:.4f})",
        )

    # -----------------------------------------------------------------------
    # F7: Classical Latency Benchmark
    # -----------------------------------------------------------------------

    def test_f7_benchmark_artifacts_generation(self):
        """Verify that running benchmark harness generates valid latency_cpu.csv and timing.json in an isolated directory."""
        import tempfile
        from p4.benchmark_latency import run_benchmarks

        with tempfile.TemporaryDirectory() as tmp_dir:
            records, meta = run_benchmarks(
                batch_sizes=(1, 64),
                num_threads=4,
                device="cpu",
                output_dir=tmp_dir,
                warmup_runs=5,
                timed_runs=10,
            )

            csv_path = Path(tmp_dir) / "latency_cpu.csv"
            json_path = Path(tmp_dir) / "timing.json"

            self.assertTrue(csv_path.exists(), "latency_cpu.csv must exist")
            self.assertTrue(json_path.exists(), "timing.json must exist")
            self.assertGreater(csv_path.stat().st_size, 0, "latency_cpu.csv must not be empty")
            self.assertGreater(json_path.stat().st_size, 0, "timing.json must not be empty")

            # Check required methods in records
            expected_methods = {"LS", "DFT-8", "Bussgang-LMMSE", "DD-PA-Iter1", "DD-PA-Iter2", "DD-PA-Iter3"}
            recorded_methods = {r["method"] for r in records}
            self.assertTrue(expected_methods.issubset(recorded_methods))

            for r in records:
                self.assertGreater(r["median_ms"], 0.0)
                self.assertGreater(r["p95_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
