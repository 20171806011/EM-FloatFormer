"""Adversarial stress test suite challenging C* = R_{>0} x U(1) gauge equivariance of EM-FloatFormer.

Empirical verification harness implementing:
1. Dynamic Range Scaling Stress (50 scales across [2^-12, 2^12], including irrational and non-power-of-2 scales).
2. Dense Phase Sweep Stress (200 phase angles across [-pi, pi), plus irrational angles).
3. Simultaneous Scale and Phase Stress (c = r * exp(j * phi), channel equivariance and logit invariance).
4. Numerical Singularity & Edge Scaling (subnormal r = 1e-20, massive r = 1e+10, all-zero, no NaNs/Infs).
"""

from __future__ import annotations

import math
import unittest
import numpy as np
import torch

from p4.em_floatformer import EMFloatFormer
from p4.simulator import generate_frame


class TestDynamicRangeScalingStress(unittest.TestCase):
    """Scenario 1: Dynamic Range Scaling Stress across 24-octave span [2^-12, 2^12]."""

    def setUp(self):
        torch.manual_seed(2026)
        self.model = EMFloatFormer(d_model=64, n_heads=2, depth=2, n_sc=32)
        self.model.eval()

    def test_dynamic_range_50_scales(self):
        g = torch.Generator().manual_seed(42)
        batch_size = 8
        n_sc = 32
        yp = torch.randn(batch_size, n_sc, dtype=torch.complex64, generator=g)
        yd = torch.randn(batch_size, n_sc, dtype=torch.complex64, generator=g)
        nv = torch.full((batch_size, 1), 0.01)

        with torch.no_grad():
            base_out = self.model(yp, yd, nv)
            base_llrs = base_out["llrs"]

        # Designated irrational and non-power-of-2 scale factors
        explicit_scales = [
            math.sqrt(2.0),
            math.pi,
            1.0 / math.e,
            3.14159 * (2.0 ** -10),
            1.414 * (2.0 ** 11),
            2.0 ** -12,
            2.0 ** 12,
        ]

        # Uniform log-spacing for remaining scales across 24 octaves [-12, 12]
        num_remaining = 50 - len(explicit_scales)
        octave_powers = np.linspace(-12.0, 12.0, num_remaining)
        sampled_scales = [float(2.0 ** p) for p in octave_powers]

        all_scales = explicit_scales + sampled_scales
        self.assertEqual(len(all_scales), 50)

        max_observed_diff = 0.0
        total_flips = 0
        details = []

        with torch.no_grad():
            for idx, r in enumerate(all_scales):
                yp_scaled = yp * r
                yd_scaled = yd * r
                nv_scaled = nv * (r ** 2)

                out_scaled = self.model(yp_scaled, yd_scaled, nv_scaled)
                scaled_llrs = out_scaled["llrs"]

                diff = (scaled_llrs - base_llrs).abs().max().item()
                flips = ((scaled_llrs > 0) != (base_llrs > 0)).sum().item()

                if diff > max_observed_diff:
                    max_observed_diff = diff
                total_flips += flips
                details.append((r, diff, flips))

                self.assertLess(
                    diff,
                    1e-5,
                    f"Scale r={r:.6e} exceeded 1e-5 logit error threshold with max_diff={diff:.6e}",
                )
                self.assertEqual(
                    flips,
                    0,
                    f"Scale r={r:.6e} produced {flips} bit decision flips",
                )

        print(
            f"\n[Scenario 1 Passed] 50 Scales Tested. Max logit diff: {max_observed_diff:.6e}, Total bit flips: {total_flips}"
        )


class TestDensePhaseSweepStress(unittest.TestCase):
    """Scenario 2: Dense Phase Sweep Stress across 200 finely-spaced and irrational angles."""

    def setUp(self):
        torch.manual_seed(2026)
        self.model = EMFloatFormer(d_model=64, n_heads=2, depth=2, n_sc=32)
        self.model.eval()

    def test_dense_phase_sweep_200_angles(self):
        g = torch.Generator().manual_seed(101)
        batch_size = 8
        n_sc = 32
        yp = torch.randn(batch_size, n_sc, dtype=torch.complex64, generator=g)
        yd = torch.randn(batch_size, n_sc, dtype=torch.complex64, generator=g)
        nv = torch.full((batch_size, 1), 0.01)

        with torch.no_grad():
            base_out = self.model(yp, yd, nv)
            base_llrs = base_out["llrs"]

        # 200 finely spaced angles across [-pi, pi)
        phases_grid = np.linspace(-math.pi, math.pi, 200, endpoint=False).tolist()

        # Add explicit irrational angles
        irrational_angles = [
            math.pi / 7.0,
            math.e,
            math.sqrt(3.0),
            (1.0 + math.sqrt(5.0)) / 2.0,  # Golden ratio phi
            -math.pi / math.sqrt(2.0),
            3.0 * math.pi / 11.0,
        ]

        all_phases = phases_grid + irrational_angles

        max_observed_diff = 0.0
        total_flips = 0

        with torch.no_grad():
            for phi in all_phases:
                c = torch.exp(1j * torch.tensor(phi, dtype=torch.float32))
                out_rot = self.model(yp * c, yd * c, nv)
                rot_llrs = out_rot["llrs"]

                diff = (rot_llrs - base_llrs).abs().max().item()
                flips = ((rot_llrs > 0) != (base_llrs > 0)).sum().item()

                if diff > max_observed_diff:
                    max_observed_diff = diff
                total_flips += flips

                self.assertLess(
                    diff,
                    1e-5,
                    f"Phase phi={phi:.6f} rad exceeded 1e-5 logit error threshold with max_diff={diff:.6e}",
                )
                self.assertEqual(
                    flips,
                    0,
                    f"Phase phi={phi:.6f} rad produced {flips} bit decision flips",
                )

        print(
            f"\n[Scenario 2 Passed] {len(all_phases)} Phase Angles Tested. Max logit diff: {max_observed_diff:.6e}, Total bit flips: {total_flips}"
        )


class TestSimultaneousScaleAndPhaseStress(unittest.TestCase):
    """Scenario 3: Simultaneous Scale and Phase Stress c = r * exp(j * phi)."""

    def setUp(self):
        torch.manual_seed(2026)
        self.model = EMFloatFormer(d_model=64, n_heads=2, depth=2, n_sc=32)
        self.model.eval()

    def test_simultaneous_c_star_equivariance(self):
        g = torch.Generator().manual_seed(202)
        batch_size = 8
        n_sc = 32
        yp = torch.randn(batch_size, n_sc, dtype=torch.complex64, generator=g)
        yd = torch.randn(batch_size, n_sc, dtype=torch.complex64, generator=g)
        nv = torch.full((batch_size, 1), 0.01)

        with torch.no_grad():
            base_out = self.model(yp, yd, nv)
            base_h_hat = base_out["h_hat"]
            base_llrs = base_out["llrs"]
            base_h_norm = base_h_hat.abs().max().item()

        np_rng = np.random.default_rng(777)
        num_trials = 50
        max_h_rel_err = 0.0
        max_llr_diff = 0.0
        total_flips = 0

        with torch.no_grad():
            for i in range(num_trials):
                # Sample r uniformly across 24 octaves [-12, 12] in log2 space
                u = np_rng.uniform(-12.0, 12.0)
                r = float(2.0 ** u)
                # Sample phi uniformly in [-pi, pi)
                phi = float(np_rng.uniform(-math.pi, math.pi))

                c = r * torch.exp(1j * torch.tensor(phi, dtype=torch.float32))

                out_c = self.model(yp * c, yd * c, nv * (r ** 2))
                h_c = out_c["h_hat"]
                llrs_c = out_c["llrs"]

                # Channel estimate equivariance check:
                # ||H_hat(c Y) - c H_hat(Y)||_inf / (|c| * (||H_hat(Y)||_inf + 1e-12)) < 1e-5
                h_err_inf = (h_c - c * base_h_hat).abs().max().item()
                rel_h_err = h_err_inf / (r * (base_h_norm + 1e-12))
                if rel_h_err > max_h_rel_err:
                    max_h_rel_err = rel_h_err

                # Bit logit invariance check:
                # ||LLR(c Y) - LLR(Y)||_inf < 1e-5
                llr_diff = (llrs_c - base_llrs).abs().max().item()
                if llr_diff > max_llr_diff:
                    max_llr_diff = llr_diff

                # Zero bit flips check
                flips = ((llrs_c > 0) != (base_llrs > 0)).sum().item()
                total_flips += flips

                self.assertLess(
                    rel_h_err,
                    1e-5,
                    f"Trial {i} (r={r:.4e}, phi={phi:.4f}): channel rel error {rel_h_err:.4e} >= 1e-5",
                )
                self.assertLess(
                    llr_diff,
                    1e-5,
                    f"Trial {i} (r={r:.4e}, phi={phi:.4f}): LLR diff {llr_diff:.4e} >= 1e-5",
                )
                self.assertEqual(
                    flips,
                    0,
                    f"Trial {i} (r={r:.4e}, phi={phi:.4f}): produced {flips} bit flips",
                )

        print(
            f"\n[Scenario 3 Passed] {num_trials} (r, phi) Pairs Tested. "
            f"Max H rel err: {max_h_rel_err:.6e}, Max LLR diff: {max_llr_diff:.6e}, Bit flips: {total_flips}"
        )


class TestNumericalSingularityEdgeScaling(unittest.TestCase):
    """Scenario 4: Numerical Singularity & Edge Scaling (subnormal r=1e-20, massive r=1e+10, zero inputs)."""

    def setUp(self):
        torch.manual_seed(2026)
        self.model = EMFloatFormer(d_model=64, n_heads=2, depth=2, n_sc=32)
        self.model.eval()

    def test_subnormal_and_massive_scales(self):
        g = torch.Generator().manual_seed(303)
        batch_size = 4
        n_sc = 32
        yp = torch.randn(batch_size, n_sc, dtype=torch.complex64, generator=g)
        yd = torch.randn(batch_size, n_sc, dtype=torch.complex64, generator=g)
        nv = torch.full((batch_size, 1), 0.01)

        extreme_scales = [
            1e-20,   # Subnormal scale
            1e-15,
            1e-10,
            1e+5,
            1e+10,   # Massive scale
        ]

        with torch.no_grad():
            for r in extreme_scales:
                yp_scaled = yp * r
                yd_scaled = yd * r
                nv_scaled = nv * (r ** 2)

                out = self.model(yp_scaled, yd_scaled, nv_scaled)
                h_hat = out["h_hat"]
                llrs = out["llrs"]
                dist_est = out["dist_est"]

                # Assert no NaNs or Infs propagate
                self.assertTrue(
                    torch.isfinite(h_hat).all().item(),
                    f"Scale r={r:.2e} produced non-finite values in h_hat",
                )
                self.assertTrue(
                    torch.isfinite(llrs).all().item(),
                    f"Scale r={r:.2e} produced non-finite values in llrs",
                )
                self.assertTrue(
                    torch.isfinite(dist_est).all().item(),
                    f"Scale r={r:.2e} produced non-finite values in dist_est",
                )

        print(f"\n[Scenario 4 Passed] Extreme scales {extreme_scales} checked without any NaN/Inf propagation.")

    def test_all_zero_and_boundary_inputs(self):
        """Test edge condition: zero input power and pure noise."""
        batch_size = 4
        n_sc = 32
        yp_zero = torch.zeros(batch_size, n_sc, dtype=torch.complex64)
        yd_zero = torch.zeros(batch_size, n_sc, dtype=torch.complex64)
        nv_zero = torch.zeros((batch_size, 1), dtype=torch.float32)

        with torch.no_grad():
            out = self.model(yp_zero, yd_zero, nv_zero)
            self.assertTrue(torch.isfinite(out["h_hat"]).all().item())
            self.assertTrue(torch.isfinite(out["llrs"]).all().item())
            self.assertTrue(torch.isfinite(out["dist_est"]).all().item())

        print("[Scenario 4 Passed] Zero input boundary handled safely without NaNs/Infs.")


if __name__ == "__main__":
    unittest.main()
