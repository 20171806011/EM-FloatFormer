"""Comprehensive independent unit tests for Phase 11 physics, caveats, firewalls, and G=16 auditor."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.linalg import eigh

from p9.contract import (
    SEED_P9_DEV,
    SEED_P9_FORMAL_ID,
    SEED_P9_FORMAL_OOD,
    SEED_P9_TRAIN,
)
from p9.independent_ref import serial_propagate_ab
from p11 import audit as p11_audit
from p11 import freeze as p11_freeze
from p11 import report as p11_report
from p11.contract import (
    CORE_ROWS,
    DELAYS,
    FRAME_DURATION,
    GateOutput,
    N,
    OFDM_PILOT_ROWS,
    OFFGRID_ALL_REGIMES,
    OFFGRID_REGIMES_ID,
    OFFGRID_REGIMES_OOD,
    POWERS,
    PROBE_ROW,
    PublicObservation,
    REGIMES_ID,
    SC_LAYOUTS,
    SEED_P11_DEV,
    SEED_P11_E2E,
    SEED_P11_TRAIN,
    T,
    AssimilationResult,
    OffGridObservation,
    build_offgrid_sc_channel_matrix,
    fractional_delay_freq_response,
    fractional_delay_matrix,
    n0_from_snr,
)
from p11.independent_ref import (
    P4_STAGE3_LITERAL_SEED,
    P8C_BASE_SEED,
    P8_MANIFOLD_BASE_SEED,
    PiecewiseDopplerShift,
    audit_assimilation_anti_leakage,
    audit_fold_target_isolation,
    check_assimilation_result_adapter,
    check_auditor_independence,
    check_p11_no_hardcoded_n0,
    check_p11_seed_disjointness,
    dirichlet_fractional_delay_matrix,
    dirichlet_kernel_closed_form,
    independent_complex_gaussian_loglik,
    independent_offgrid_propagate_ab,
    independent_subset_bayes_posteriors,
    ray_predictive_moments,
    row_boundary_time,
    verify_fractional_delay_operator,
    verify_offgrid_observation,
    verify_offgrid_phase_continuity,
    verify_p11_public_observation,
)


P11_ROOT = Path(__file__).resolve().parents[1]
SYNTH_FORMAL_ID_BASE = 1_130_000_000_000
SYNTH_FORMAL_OOD_BASE = 1_140_000_000_000


class DirichletFractionalDelayTests(unittest.TestCase):
    def test_dirichlet_fractional_delay_matches_contract_unitarity_and_integer_shifts(self):
        delays = (0.0, 0.125, 0.5, 1.37, 2.0, 3.14159265, 4.0, 5.999, 6.5)
        rep = verify_fractional_delay_operator(delays=delays, n=N, tol=1e-14)
        self.assertTrue(rep["passed"], f"Fractional delay verification failed: {rep}")
        self.assertLess(rep["max_contract_err"], 1e-14)
        self.assertLess(rep["max_closed_form_err"], 1e-14)
        self.assertLess(rep["max_unitary_err"], 1e-14)
        self.assertLess(rep["max_duality_err"], 1e-14)
        self.assertLess(rep["max_int_shift_err"], 1e-14)
        self.assertLess(rep["homomorphism_err"], 1e-14)
        self.assertLess(rep["adjoint_err"], 1e-14)

        rng = np.random.default_rng(11001)
        x = rng.normal(size=N) + 1j * rng.normal(size=N)
        for d_int in (0, 1, 2, 4, 6):
            d_mat = dirichlet_fractional_delay_matrix(float(d_int), n=N)
            self.assertLess(float(np.max(np.abs(d_mat @ x - np.roll(x, d_int)))), 1e-14)

        for n_dim in (16, 17):
            eye = np.eye(n_dim, dtype=np.complex128)
            for tau in (0.0, 0.73, 2.5, 4.0):
                d_ind = dirichlet_fractional_delay_matrix(tau, n=n_dim)
                d_cls = dirichlet_kernel_closed_form(tau, n=n_dim)
                d_con = fractional_delay_matrix(tau, n=n_dim)
                self.assertLess(float(np.max(np.abs(d_ind - d_con))), 1e-14)
                self.assertLess(float(np.max(np.abs(d_ind - d_cls))), 1e-14)
                self.assertLess(float(np.max(np.abs(d_ind.conj().T @ d_ind - eye))), 1e-14)


class IndependentOffGridPropagatorTests(unittest.TestCase):
    def test_offgrid_propagator_matches_p9_serial_for_integer_delays(self):
        rng = np.random.default_rng(11002)
        a_grid = (rng.normal(size=(T, N)) + 1j * rng.normal(size=(T, N))) / math.sqrt(2.0)
        b_grid = (rng.normal(size=(T, N)) + 1j * rng.normal(size=(T, N))) / math.sqrt(2.0)
        gains = np.sqrt(POWERS) * (rng.normal(size=3) + 1j * rng.normal(size=3)) / math.sqrt(2.0)
        freqs = rng.uniform(0.045, 0.080, size=3) * rng.choice([-1.0, 1.0], size=3)
        chirps = rng.uniform(-0.002, 0.002, size=3)
        ch_a = {"gain": gains, "frequency": freqs, "chirp": chirps}

        p9_ya, p9_yb = serial_propagate_ab(a_grid, b_grid, ch_a)
        res_c = independent_offgrid_propagate_ab(
            a_grid, b_grid, gains, DELAYS.astype(np.float64), freqs, chirps, return_details=True
        )
        self.assertLess(float(np.max(np.abs(res_c.y_a - p9_ya))), 1e-14)
        self.assertLess(float(np.max(np.abs(res_c.y_b - p9_yb))), 1e-14)
        for r in range(T):
            self.assertLess(float(np.max(np.abs(res_c.h_b_true_mat[r] @ b_grid[r] - res_c.y_b[r]))), 1e-14)
            h_con = build_offgrid_sc_channel_matrix(res_c.h_b_useful[:, r, :], DELAYS.astype(np.float64), n=N)
            self.assertLess(float(np.max(np.abs(res_c.h_b_true_mat[r] - h_con))), 1e-14)

        for t_s in (row_boundary_time(0, frame="B"), row_boundary_time(6, frame="B"), (23 * 48 + 25) / 32.0):
            shift = PiecewiseDopplerShift(ray_index=1, delta_freq=-0.062, t_shift=t_s, delta_chirp=0.001)
            p9_ya_s, p9_yb_s = serial_propagate_ab(a_grid, b_grid, ch_a, shift=shift)
            p11_ya_s, p11_yb_s = independent_offgrid_propagate_ab(
                a_grid, b_grid, gains, DELAYS.astype(np.float64), freqs, chirps, shifts=shift
            )
            self.assertLess(float(np.max(np.abs(p11_ya_s - p9_ya_s))), 1e-14)
            self.assertLess(float(np.max(np.abs(p11_yb_s - p9_yb_s))), 1e-14)

        gains_b = np.sqrt(POWERS) * (rng.normal(size=3) + 1j * rng.normal(size=3)) / math.sqrt(2.0)
        freqs_b = rng.uniform(0.045, 0.080, size=3) * rng.choice([-1.0, 1.0], size=3)
        chirps_b = rng.uniform(-0.002, 0.002, size=3)
        ch_b = {"gain": gains_b, "frequency": freqs_b, "chirp": chirps_b}
        for origin in ("global", "frame_b"):
            p9_ya_r, p9_yb_r = serial_propagate_ab(a_grid, b_grid, ch_a, channel_b=ch_b, channel_b_time_origin=origin)
            p11_ya_r, p11_yb_r = independent_offgrid_propagate_ab(
                a_grid,
                b_grid,
                gains,
                DELAYS.astype(np.float64),
                freqs,
                chirps,
                full_reset_b={
                    "gains": gains_b,
                    "delays": DELAYS.astype(np.float64),
                    "freqs": freqs_b,
                    "chirps": chirps_b,
                },
                channel_b_time_origin=origin,
            )
            self.assertLess(float(np.max(np.abs(p11_ya_r - p9_ya_r))), 1e-14)
            self.assertLess(float(np.max(np.abs(p11_yb_r - p9_yb_r))), 1e-14)

    def test_offgrid_ofdm_cpsc_static_duality_variable_rays(self):
        rng = np.random.default_rng(11003)
        idx = np.arange(N, dtype=np.int64)
        prod_mod = (idx[:, None] * idx[None, :]) % N
        f_dft = np.exp(-2j * math.pi * prod_mod.astype(np.float64) / float(N)) / math.sqrt(float(N))
        f_idft = f_dft.conj().T

        for n_rays in (2, 3, 4, 5):
            delays = rng.uniform(0.0, 6.5, size=n_rays)
            gains = (rng.normal(size=n_rays) + 1j * rng.normal(size=n_rays)) / math.sqrt(2.0 * n_rays)
            freqs_zero = np.zeros(n_rays, dtype=np.float64)
            x_freq = (rng.normal(size=(T, N)) + 1j * rng.normal(size=(T, N))) / math.sqrt(2.0)
            x_time = x_freq @ f_idft.T

            res = independent_offgrid_propagate_ab(
                x_freq, x_time, gains, delays, freqs_zero, a_is_freq=True, return_details=True
            )
            y_b_freq = res.y_b @ f_dft.T
            self.assertLess(float(np.max(np.abs(res.y_a - y_b_freq))), 1e-14)
            h_freq = np.sum(gains[:, None] * fractional_delay_freq_response(delays, n=N), axis=0)
            self.assertLess(float(np.max(np.abs(res.y_a - x_freq * h_freq[None, :]))), 1e-14)
            for r in (0, 7, 19):
                h_con = build_offgrid_sc_channel_matrix(res.h_b_useful[:, r, :], delays, n=N)
                self.assertLess(float(np.max(np.abs(res.h_b_true_mat[r] - h_con))), 1e-14)
                self.assertLess(float(np.max(np.abs(res.h_b_true_mat[r] @ x_time[r] - res.y_b[r]))), 1e-14)

    def test_offgrid_phase_continuity_boundary_and_midframe_shifts(self):
        rng = np.random.default_rng(11004)
        a_grid = (rng.normal(size=(T, N)) + 1j * rng.normal(size=(T, N))) / math.sqrt(2.0)
        b_grid = (rng.normal(size=(T, N)) + 1j * rng.normal(size=(T, N))) / math.sqrt(2.0)

        for n_rays in (4, 5):
            delays = rng.uniform(0.0, 6.5, size=n_rays)
            gains = (rng.normal(size=n_rays) + 1j * rng.normal(size=n_rays)) / math.sqrt(2.0 * n_rays)
            freqs_a = rng.uniform(0.04, 0.085, size=n_rays) * rng.choice([-1.0, 1.0], size=n_rays)
            chirps = rng.uniform(-0.002, 0.002, size=n_rays)
            ya_cont, yb_cont = independent_offgrid_propagate_ab(a_grid, b_grid, gains, delays, freqs_a, chirps)

            for shift_row in (0, 5, 11, 16):
                t_s = row_boundary_time(shift_row, frame="B")
                if shift_row == 0:
                    self.assertAlmostEqual(t_s, FRAME_DURATION, places=15)
                freqs_b = freqs_a.copy()
                freqs_b[1] += 0.055
                freqs_b[-1] -= 0.048
                res = independent_offgrid_propagate_ab(
                    a_grid, b_grid, gains, delays, freqs_a, chirps,
                    freqs_b=freqs_b, shift_row=shift_row, return_details=True,
                )
                self.assertIsNotNone(res.phase_continuity)
                for cont in res.phase_continuity:
                    self.assertTrue(cont["continuous_ok"])
                    self.assertLess(cont["phase_jump_rad"], 1e-14)
                    self.assertLess(cont["gain_jump"], 1e-14)
                    self.assertLess(cont["reanchored_max_gain_err"], 1e-13)
                    self.assertLess(cont["freq_jump_err"], 1e-6)
                self.assertLess(float(np.max(np.abs(res.y_a - ya_cont))), 1e-14)
                if shift_row > 0:
                    self.assertLess(float(np.max(np.abs(res.y_b[:shift_row] - yb_cont[:shift_row]))), 1e-14)
                self.assertGreater(float(np.max(np.abs(res.y_b[shift_row:] - yb_cont[shift_row:]))), 1e-2)


class ComplexGaussianAndSubsetBayesTests(unittest.TestCase):
    def test_independent_complex_gaussian_loglik_accuracy_and_cstar_scaling(self):
        rng = np.random.default_rng(11005)
        y1 = np.array([1.2 - 0.7j], dtype=np.complex128)
        mu1 = np.array([-0.3 + 0.4j], dtype=np.complex128)
        v1 = 2.5
        ll1 = independent_complex_gaussian_loglik(y1, mu1, np.array([[v1]], dtype=np.complex128))
        ll1_exact = -math.log(math.pi) - math.log(v1) - (abs(y1[0] - mu1[0]) ** 2) / v1
        self.assertLess(abs(ll1 - ll1_exact), 1e-14)

        m_dim = 24
        a_mat = (rng.normal(size=(m_dim, m_dim)) + 1j * rng.normal(size=(m_dim, m_dim))) / math.sqrt(2.0 * m_dim)
        cov = a_mat @ a_mat.conj().T + 0.15 * np.eye(m_dim, dtype=np.complex128)
        cov = 0.5 * (cov + cov.conj().T)
        mu = rng.normal(size=m_dim) + 1j * rng.normal(size=m_dim)
        y = rng.normal(size=m_dim) + 1j * rng.normal(size=m_dim)

        ll_chol = independent_complex_gaussian_loglik(y, mu, cov)
        eigvals, eigvecs = eigh(cov)
        proj = eigvecs.conj().T @ (y - mu)
        ll_eigh = float(-m_dim * math.log(math.pi) - np.sum(np.log(eigvals)) - np.sum((np.abs(proj) ** 2) / eigvals))
        self.assertLess(abs(ll_chol - ll_eigh), 1e-12)

        for alpha in (1e6 * complex(math.cos(1.3), math.sin(1.3)), 1e-6 * complex(math.cos(-2.1), math.sin(-2.1))):
            ll_s = independent_complex_gaussian_loglik(alpha * y, alpha * mu, (abs(alpha) ** 2) * cov)
            expected_ll_s = ll_chol - float(m_dim) * math.log(abs(alpha) ** 2)
            self.assertLess(abs(ll_s - expected_ll_s), 1e-12)

        bad_non_herm = cov.copy()
        bad_non_herm[0, 1] += 1.0
        with self.assertRaises(ValueError):
            independent_complex_gaussian_loglik(y, mu, bad_non_herm)
        bad_diag = cov.copy()
        bad_diag[0, 0] = -1.0
        with self.assertRaises(ValueError):
            independent_complex_gaussian_loglik(y, mu, bad_diag)

    def test_subset_bayes_2pow3_correctness_and_cstar_invariance(self):
        rng = np.random.default_rng(11006)
        k_rays, n_hyp, m_dim = 3, 5, N
        n0 = n0_from_snr(12.0)

        design = (rng.normal(size=(k_rays, n_hyp, m_dim)) + 1j * rng.normal(size=(k_rays, n_hyp, m_dim))) / math.sqrt(2.0)
        lw_inh = np.log(np.array([[0.7, 0.1, 0.1, 0.05, 0.05]] * k_rays, dtype=np.float64))
        mn_inh = (rng.normal(size=(k_rays, n_hyp)) + 1j * rng.normal(size=(k_rays, n_hyp))) * np.sqrt(POWERS)[:, None]
        vr_inh = np.full((k_rays, n_hyp), 0.02, dtype=np.float64) * POWERS[:, None]

        lw_clr = np.full((k_rays, n_hyp), -math.log(float(n_hyp)), dtype=np.float64)
        mn_clr = np.zeros((k_rays, n_hyp), dtype=np.complex128)
        vr_clr = np.full((k_rays, n_hyp), 1.0, dtype=np.float64) * POWERS[:, None]

        mu_inh, cov_inh = ray_predictive_moments(lw_inh, mn_inh, vr_inh, design)
        mu_clr, cov_clr = ray_predictive_moments(lw_clr, mn_clr, vr_clr, design)

        noise = math.sqrt(n0 / 2.0) * (rng.normal(size=m_dim) + 1j * rng.normal(size=m_dim))
        y_partial = mu_inh[0] + (-2.5 * mu_inh[1]) + mu_inh[2] + noise
        sb_lsb = independent_subset_bayes_posteriors(y_partial, mu_inh, cov_inh, mu_clr, cov_clr, n0, bit_order="lsb")
        self.assertEqual(int(np.argmax(sb_lsb["subset_posteriors"])), 5)

        for alpha in (4.5 * complex(math.cos(0.9), math.sin(0.9)), 0.12 * complex(math.cos(-2.4), math.sin(-2.4))):
            a2 = abs(alpha) ** 2
            sb_s = independent_subset_bayes_posteriors(
                alpha * y_partial, alpha * mu_inh, a2 * cov_inh, alpha * mu_clr, a2 * cov_clr, a2 * n0
            )
            self.assertLess(float(np.max(np.abs(sb_s["subset_posteriors"] - sb_lsb["subset_posteriors"]))), 1e-12)
            self.assertLess(float(np.max(np.abs(sb_s["ray_posteriors"] - sb_lsb["ray_posteriors"]))), 1e-12)


class P11ContractAndCaveatTests(unittest.TestCase):
    def _make_clean_pub_obs(self) -> PublicObservation:
        rng = np.random.default_rng(11007)
        y_a = ((rng.normal(size=(T, N)) + 1j * rng.normal(size=(T, N))) / math.sqrt(2.0)).astype(np.complex128)
        y_b = ((rng.normal(size=(T, N)) + 1j * rng.normal(size=(T, N))) / math.sqrt(2.0)).astype(np.complex128)
        y_b[PROBE_ROW] = complex(math.nan, math.nan)
        return PublicObservation(
            y_a=y_a, y_b=y_b, training_rows=SC_LAYOUTS["sparse"],
            probe_row=PROBE_ROW, n0=n0_from_snr(9.0), budget="sparse",
        )

    def test_caveat1_strict_p11_assimilation_result_binding_and_adapter(self):
        pub_obs = self._make_clean_pub_obs()
        clean_fn = p11_audit._make_clean_cstar_assimilator()

        @dataclass(frozen=True)
        class ForeignAssimilationResult:
            gate_out: GateOutput
            decoded_info: dict[int, np.ndarray]
            assimilated_rows: tuple[int, ...]
            row_weights: dict[int, float]
            row_margins: dict[int, float]
            n_rounds: int
            elapsed_ms: float

        def foreign_fn(o):
            r = clean_fn(o)
            return ForeignAssimilationResult(
                gate_out=r.gate_out, decoded_info=r.decoded_info, assimilated_rows=r.assimilated_rows,
                row_weights=r.row_weights, row_margins=r.row_margins, n_rounds=r.n_rounds, elapsed_ms=r.elapsed_ms,
            )

        # Default strict binding rejects foreign AssimilationResult
        rep_rejected = audit_assimilation_anti_leakage(foreign_fn, pub_obs, strict=False)
        self.assertFalse(rep_rejected["passed"])
        self.assertTrue(any("p11.contract.AssimilationResult" in issue for issue in rep_rejected["issues"]))

        # Explicit adapter check & allow_historical_adapter=True succeed cleanly
        ad_rep = check_assimilation_result_adapter(foreign_fn(pub_obs), target_cls=AssimilationResult, strict=True)
        self.assertTrue(ad_rep["valid"])
        self.assertFalse(ad_rep["is_p11_type"])
        self.assertIsInstance(ad_rep["adapted_instance"], AssimilationResult)

        rep_adapted = audit_assimilation_anti_leakage(foreign_fn, pub_obs, allow_historical_adapter=True, strict=True)
        self.assertTrue(rep_adapted["passed"])

    def test_caveat2_rejects_timedelta64_datetime64_and_bool_in_seeds_and_obs(self):
        p11_tr = [SEED_P11_TRAIN + i for i in range(8)]
        p11_dv = [SEED_P11_DEV + i for i in range(8)]
        p11_fid = [SYNTH_FORMAL_ID_BASE + i for i in range(8)]
        p11_food = [SYNTH_FORMAL_OOD_BASE + i for i in range(8)]

        for bad_val in (
            True,
            np.bool_(False),
            np.timedelta64(42, "ns"),
            np.timedelta64(1, "s"),
            np.timedelta64("NaT"),
            np.datetime64("2026-11-20"),
            1.5,
        ):
            with self.assertRaises(TypeError):
                check_p11_seed_disjointness([bad_val], p11_dv, p11_fid, p11_food)
            with self.assertRaises(TypeError):
                check_p11_seed_disjointness(p11_tr, p11_dv, p11_fid, p11_food, p11_e2e_seeds=[bad_val])

        # OffGridObservation & PublicObservation timedelta64/datetime64 rejection
        rng = np.random.default_rng(11008)
        y_a = ((rng.normal(size=(T, N)) + 1j * rng.normal(size=(T, N))) / math.sqrt(2.0)).astype(np.complex128)
        y_b = ((rng.normal(size=(T, N)) + 1j * rng.normal(size=(T, N))) / math.sqrt(2.0)).astype(np.complex128)
        y_b[PROBE_ROW] = complex(math.nan, math.nan)
        pt_b = np.exp(1j * rng.uniform(-math.pi, math.pi, size=(1, N))).astype(np.complex128)
        pt_a = np.exp(1j * rng.uniform(-math.pi, math.pi, size=(len(OFDM_PILOT_ROWS), N // 2))).astype(np.complex128)

        bad_og_td = OffGridObservation(
            y_a=y_a, y_b=y_b, training_rows=(np.timedelta64(6, "ns"),),
            probe_row=PROBE_ROW, n0=n0_from_snr(9.0), pilot_tx_b=pt_b, pilot_tx_a=pt_a,
        )
        self.assertFalse(verify_offgrid_observation(bad_og_td)["valid"])

        bad_pub_td = PublicObservation(
            y_a=y_a, y_b=y_b, training_rows=(np.timedelta64(10, "ns"),),
            probe_row=PROBE_ROW, n0=n0_from_snr(9.0), budget="sparse",
        )
        self.assertFalse(verify_p11_public_observation(bad_pub_td)["valid"])

    def test_caveat3_all_24_historical_ranges_and_5_p11_pools_disjointness(self):
        p11_tr = [SEED_P11_TRAIN + i for i in range(32)]
        p11_dv = [SEED_P11_DEV + i for i in range(32)]
        p11_fid = [SYNTH_FORMAL_ID_BASE + i for i in range(32)]
        p11_food = [SYNTH_FORMAL_OOD_BASE + i for i in range(32)]
        p11_e2e = [SEED_P11_E2E + i for i in range(32)]

        clean = check_p11_seed_disjointness(p11_tr, p11_dv, p11_fid, p11_food, p11_e2e, strict=True)
        self.assertTrue(clean["disjoint"])
        self.assertEqual(clean["namespace_collisions"], [])

        # Every historical family & literal seed must be caught
        for bad_seed in (
            42,                                # p1_to_p4_legacy
            50_000_000,                        # p1_to_p4_extended
            95_000_000,                        # p8_manifold_dev_ladder
            P8C_BASE_SEED + 1000,              # p8c
            P8_MANIFOLD_BASE_SEED + 1000,      # p8_manifold
            SEED_P9_TRAIN + 100000,            # p9_train
            SEED_P9_DEV + 1000,                # p9_dev
            SEED_P9_FORMAL_ID + 1000,          # p9_formal_id
            SEED_P9_FORMAL_OOD + 1000,         # p9_formal_ood
            971_120_030_005,                   # p10_9e9d93b
            1_010_000_000_123,                 # p10_external_dev
            1_030_000_000_123,                 # p10_external_formal_id
            1_050_000_000_123,                 # p10_external_reserved_1050
            P4_STAGE3_LITERAL_SEED,            # p4_stage3_link_genie_literal
        ):
            rep = check_p11_seed_disjointness(p11_tr + [bad_seed], p11_dv, p11_fid, p11_food, p11_e2e, strict=False)
            self.assertFalse(rep["disjoint"], f"Missed collision with historical seed {bad_seed}")
            with self.assertRaises(ValueError):
                check_p11_seed_disjointness(p11_tr + [bad_seed], p11_dv, p11_fid, p11_food, p11_e2e, strict=True)

    def test_caveat4_auditor_independence_absolute_relative_and_dynamic(self):
        audit_files = [
            P11_ROOT / "independent_ref.py",
            P11_ROOT / "audit.py",
            P11_ROOT / "tests" / "test_p11_audit.py",
        ]
        rep = check_auditor_independence(audit_files, strict=True)
        self.assertTrue(rep["independent"], f"Unexpected violations: {rep['violations']}")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_p11 = Path(tmp) / "p11"
            (tmp_p11 / "tests").mkdir(parents=True)
            snippets = [
                ("bad_torch.py", "import " + "tor" + "ch\n"),
                ("bad_m2_abs.py", "from p11." + "payload" + "_rebuild import estimate_payload_posterior\n"),
                ("bad_m2_rel.py", "from . import " + "wavefront" + "_pipeline\n"),
                ("bad_m3_pkg.py", "from p11 import " + "offgrid" + "_channel\n"),
                ("bad_m3_rel.py", "from ." + "manifold" + "_gate import ManifoldTrustGate\n"),
                ("bad_p10.py", "from p10." + "turbo" + "_assimilate import assimilate_blind_wavefront\n"),
                ("bad_p9.py", "from p9." + "gat" + "ing import build_candidates\n"),
                ("bad_dyn.py", "import importlib\nimportlib.import_module('p11." + "payload" + "_rebuild')\n"),
                ("bad_dyn_kw.py", "import importlib\nimportlib.import_module(name='p11." + "payload" + "_rebuild')\n"),
                ("bad_dyn_from_alias.py", "from importlib import import_module as im\nim('p11." + "wavefront" + "_pipeline')\n"),
                ("bad_dyn_mod_alias.py", "import importlib as il\nil.import_module('p11." + "manifold" + "_gate')\n"),
                ("bad_dyn_assign_alias.py", "import importlib\nim = importlib.import_module\nim('p11." + "offgrid" + "_channel')\n"),
                ("bad_dyn_rel_kw.py", "import importlib\nimportlib.import_module('." + "payload" + "_rebuild', package='p11')\n"),
                ("bad_dyn_rel_pos.py", "import importlib\nimportlib.import_module('." + "wavefront" + "_pipeline', 'p11')\n"),
                ("bad_p10_top.py", "import p" + "10\n"),
                ("bad_p10_from.py", "from p" + "10 import anything\n"),
                ("bad_dunder_fromlist.py", "__import__('p11', fromlist=('" + "payload" + "_rebuild',))\n"),
            ]
            for fname, code in snippets:
                f = tmp_p11 / fname
                f.write_text(code, encoding="utf-8")
                self.assertFalse(check_auditor_independence([f])["independent"], f"Missed violation in {fname}")

            bad_parent_rel = tmp_p11 / "tests" / "bad_rel2.py"
            bad_parent_rel.write_text("from .. import " + "wavefront" + "_pipeline\n", encoding="utf-8")
            self.assertFalse(check_auditor_independence([bad_parent_rel])["independent"])

            bad_dyn_rel_file = tmp_p11 / "tests" / "bad_dyn_rel2.py"
            bad_dyn_rel_file.write_text(
                "import importlib\nimportlib.import_module('.." + "wavefront" + "_pipeline')\n",
                encoding="utf-8",
            )
            self.assertFalse(check_auditor_independence([bad_dyn_rel_file])["independent"])

    def test_caveat5_no_hardcoded_n0_across_p11_audit_and_contract_files(self):
        files = [
            P11_ROOT / "contract.py",
            P11_ROOT / "independent_ref.py",
            P11_ROOT / "audit.py",
            P11_ROOT / "tests" / "test_p11_contract.py",
            P11_ROOT / "tests" / "test_p11_audit.py",
        ]
        rep = check_p11_no_hardcoded_n0(files, strict=True)
        self.assertTrue(rep["clean"], f"Unexpected N0 violations: {rep['violations']}")
        self.assertTrue(rep["p11_no_hardcoded_n0_passed"])

        with tempfile.TemporaryDirectory() as tmp:
            bad_file = Path(tmp) / "bad_n0.py"
            bad_const = "0.1258" + "9254117941673"
            bad_pow = "10.0 ** (-0." + "9)"
            bad_file.write_text(f"def f(x, n0={bad_const}):\n    return x / ({bad_pow})\n", encoding="utf-8")
            self.assertFalse(check_p11_no_hardcoded_n0([bad_file])["clean"])


class AntiLeakageAndFoldIsolationTests(unittest.TestCase):
    def _make_clean_offgrid_obs(self, snr_db: float = 9.0, *, noiseless: bool = False) -> OffGridObservation:
        rng = np.random.default_rng(11009)
        y_a = ((rng.normal(size=(T, N)) + 1j * rng.normal(size=(T, N))) / math.sqrt(2.0)).astype(np.complex128)
        y_b = ((rng.normal(size=(T, N)) + 1j * rng.normal(size=(T, N))) / math.sqrt(2.0)).astype(np.complex128)
        y_b[PROBE_ROW] = complex(math.nan, math.nan)
        tr = (6, 13)
        pt_b = np.exp(1j * rng.uniform(-math.pi, math.pi, size=(len(tr), N))).astype(np.complex128)
        pt_a = np.exp(1j * rng.uniform(-math.pi, math.pi, size=(len(OFDM_PILOT_ROWS), N // 2))).astype(np.complex128)
        return OffGridObservation(
            y_a=y_a, y_b=y_b, training_rows=tr, probe_row=PROBE_ROW,
            n0=0.0 if noiseless else n0_from_snr(snr_db), pilot_tx_b=pt_b, pilot_tx_a=pt_a,
        )

    def test_verify_offgrid_observation_and_anti_leakage_firewalls(self):
        clean_og = self._make_clean_offgrid_obs()
        self.assertTrue(verify_offgrid_observation(clean_og, strict=True)["valid"])
        zero_n0_og = self._make_clean_offgrid_obs(noiseless=True)
        self.assertFalse(verify_offgrid_observation(zero_n0_og)["valid"])
        with self.assertRaises(ValueError):
            verify_offgrid_observation(zero_n0_og, strict=True)
        neg_n0_og = OffGridObservation(
            y_a=clean_og.y_a,
            y_b=clean_og.y_b,
            training_rows=clean_og.training_rows,
            probe_row=PROBE_ROW,
            n0=-0.1,
            pilot_tx_b=clean_og.pilot_tx_b,
            pilot_tx_a=clean_og.pilot_tx_a,
        )
        self.assertFalse(verify_offgrid_observation(neg_n0_og)["valid"])

        y_bad = clean_og.y_b.copy()
        y_bad[PROBE_ROW] = 1.0 + 0.0j
        bad_og = OffGridObservation(
            y_a=clean_og.y_a, y_b=y_bad, training_rows=clean_og.training_rows,
            probe_row=PROBE_ROW, n0=clean_og.n0, pilot_tx_b=clean_og.pilot_tx_b, pilot_tx_a=clean_og.pilot_tx_a,
        )
        self.assertFalse(verify_offgrid_observation(bad_og)["valid"])

        pub_obs = PublicObservation(
            y_a=clean_og.y_a, y_b=clean_og.y_b, training_rows=SC_LAYOUTS["sparse"],
            probe_row=PROBE_ROW, n0=n0_from_snr(9.0), budget="sparse",
        )
        clean_fn = p11_audit._make_clean_cstar_assimilator()
        rep = audit_assimilation_anti_leakage(clean_fn, pub_obs, strict=True)
        self.assertTrue(rep["passed"])
        self.assertEqual(rep["max_probe_poison_diff"], 0.0)
        self.assertLess(rep["max_cstar_err"], 1e-12)

        def leaky_probe(o):
            r = clean_fn(o)
            g = r.gate_out
            p_val = np.nan_to_num(o.y_b[o.probe_row], nan=0.0)
            bad_g = GateOutput(
                history_weight=g.history_weight, probe_weight=g.probe_weight, mix_llrs=g.mix_llrs,
                mix_probe_pred=g.mix_probe_pred + 1e-9 * p_val,
                mix_probe_var=g.mix_probe_var, mix_mean_ray=g.mix_mean_ray, mix_var_ray=g.mix_var_ray,
            )
            return AssimilationResult(
                gate_out=bad_g, decoded_info=r.decoded_info, assimilated_rows=r.assimilated_rows,
                row_weights=r.row_weights, row_margins=r.row_margins, n_rounds=r.n_rounds, elapsed_ms=r.elapsed_ms,
            )

        self.assertFalse(audit_assimilation_anti_leakage(leaky_probe, pub_obs)["passed"])

    def test_adapter_and_anti_leakage_reject_bool_timedelta_datetime(self):
        clean_og = self._make_clean_offgrid_obs()
        pub_obs = PublicObservation(
            y_a=clean_og.y_a,
            y_b=clean_og.y_b,
            training_rows=SC_LAYOUTS["sparse"],
            probe_row=PROBE_ROW,
            n0=n0_from_snr(9.0),
            budget="sparse",
        )
        clean_fn = p11_audit._make_clean_cstar_assimilator()
        base_res = clean_fn(pub_obs)
        g_ok = base_res.gate_out

        # 1. Adapter pre-coercion checks on scalar fields and GateOutput (even when isinstance(..., GateOutput))
        bad_adapter_cases = [
            AssimilationResult(
                gate_out=g_ok,
                decoded_info=base_res.decoded_info,
                assimilated_rows=base_res.assimilated_rows,
                row_weights=base_res.row_weights,
                row_margins=base_res.row_margins,
                n_rounds=True,  # bool!
                elapsed_ms=base_res.elapsed_ms,
            ),
            AssimilationResult(
                gate_out=g_ok,
                decoded_info=base_res.decoded_info,
                assimilated_rows=base_res.assimilated_rows,
                row_weights=base_res.row_weights,
                row_margins=base_res.row_margins,
                n_rounds=base_res.n_rounds,
                elapsed_ms=True,  # bool!
            ),
            AssimilationResult(
                gate_out=g_ok,
                decoded_info=base_res.decoded_info,
                assimilated_rows=(True,),
                row_weights=base_res.row_weights,
                row_margins=base_res.row_margins,
                n_rounds=base_res.n_rounds,
                elapsed_ms=base_res.elapsed_ms,
            ),
            AssimilationResult(
                gate_out=g_ok,
                decoded_info=base_res.decoded_info,
                assimilated_rows=base_res.assimilated_rows,
                row_weights={1: True},
                row_margins=base_res.row_margins,
                n_rounds=base_res.n_rounds,
                elapsed_ms=base_res.elapsed_ms,
            ),
            AssimilationResult(
                gate_out=g_ok,
                decoded_info=base_res.decoded_info,
                assimilated_rows=base_res.assimilated_rows,
                row_weights=base_res.row_weights,
                row_margins={1: np.timedelta64(1, "ns")},
                n_rounds=base_res.n_rounds,
                elapsed_ms=base_res.elapsed_ms,
            ),
            AssimilationResult(
                gate_out=GateOutput(
                    history_weight=g_ok.history_weight,
                    probe_weight=True,  # bool!
                    mix_llrs=g_ok.mix_llrs,
                    mix_probe_pred=g_ok.mix_probe_pred,
                    mix_probe_var=g_ok.mix_probe_var,
                    mix_mean_ray=g_ok.mix_mean_ray,
                    mix_var_ray=g_ok.mix_var_ray,
                ),
                decoded_info=base_res.decoded_info,
                assimilated_rows=base_res.assimilated_rows,
                row_weights=base_res.row_weights,
                row_margins=base_res.row_margins,
                n_rounds=base_res.n_rounds,
                elapsed_ms=base_res.elapsed_ms,
            ),
            AssimilationResult(
                gate_out=GateOutput(
                    history_weight=np.ones_like(g_ok.history_weight, dtype=bool),
                    probe_weight=g_ok.probe_weight,
                    mix_llrs=g_ok.mix_llrs,
                    mix_probe_pred=g_ok.mix_probe_pred,
                    mix_probe_var=g_ok.mix_probe_var,
                    mix_mean_ray=g_ok.mix_mean_ray,
                    mix_var_ray=g_ok.mix_var_ray,
                ),
                decoded_info=base_res.decoded_info,
                assimilated_rows=base_res.assimilated_rows,
                row_weights=base_res.row_weights,
                row_margins=base_res.row_margins,
                n_rounds=base_res.n_rounds,
                elapsed_ms=base_res.elapsed_ms,
            ),
            AssimilationResult(
                gate_out=GateOutput(
                    history_weight=g_ok.history_weight,
                    probe_weight=g_ok.probe_weight,
                    mix_llrs=g_ok.mix_llrs,
                    mix_probe_pred=g_ok.mix_probe_pred,
                    mix_probe_var=np.full(g_ok.mix_probe_var.shape, np.timedelta64(1, "ns")),
                    mix_mean_ray=g_ok.mix_mean_ray,
                    mix_var_ray=g_ok.mix_var_ray,
                ),
                decoded_info=base_res.decoded_info,
                assimilated_rows=base_res.assimilated_rows,
                row_weights=base_res.row_weights,
                row_margins=base_res.row_margins,
                n_rounds=base_res.n_rounds,
                elapsed_ms=base_res.elapsed_ms,
            ),
            AssimilationResult(
                gate_out=g_ok,
                decoded_info={**base_res.decoded_info, CORE_ROWS[0]: np.ones(64, dtype=bool)},
                assimilated_rows=base_res.assimilated_rows,
                row_weights=base_res.row_weights,
                row_margins=base_res.row_margins,
                n_rounds=base_res.n_rounds,
                elapsed_ms=base_res.elapsed_ms,
            ),
        ]
        for idx, bad_res in enumerate(bad_adapter_cases):
            rep = check_assimilation_result_adapter(bad_res, strict=False)
            self.assertFalse(rep["valid"], f"Adapter missed forbidden bool/timedelta64 case {idx}")

        # 2. audit_assimilation_anti_leakage pre-coercion checks on GateOutput arrays and decoded_info
        def _bool_decoded_fn(o):
            r = clean_fn(o)
            di = dict(r.decoded_info)
            di[CORE_ROWS[0]] = np.ones(64, dtype=bool)
            return AssimilationResult(
                gate_out=r.gate_out,
                decoded_info=di,
                assimilated_rows=r.assimilated_rows,
                row_weights=r.row_weights,
                row_margins=r.row_margins,
                n_rounds=r.n_rounds,
                elapsed_ms=r.elapsed_ms,
            )

        def _mixed_bool_list_decoded_fn(o):
            r = clean_fn(o)
            di = dict(r.decoded_info)
            di[CORE_ROWS[0]] = [True] + [0] * 63
            return AssimilationResult(
                gate_out=r.gate_out,
                decoded_info=di,
                assimilated_rows=r.assimilated_rows,
                row_weights=r.row_weights,
                row_margins=r.row_margins,
                n_rounds=r.n_rounds,
                elapsed_ms=r.elapsed_ms,
            )

        def _datetime_gate_fn(o):
            r = clean_fn(o)
            g = r.gate_out
            bad_g = GateOutput(
                history_weight=np.full(g.history_weight.shape, np.datetime64("2026-01-01")),
                probe_weight=g.probe_weight,
                mix_llrs=g.mix_llrs,
                mix_probe_pred=g.mix_probe_pred,
                mix_probe_var=g.mix_probe_var,
                mix_mean_ray=g.mix_mean_ray,
                mix_var_ray=g.mix_var_ray,
            )
            return AssimilationResult(
                gate_out=bad_g,
                decoded_info=r.decoded_info,
                assimilated_rows=r.assimilated_rows,
                row_weights=r.row_weights,
                row_margins=r.row_margins,
                n_rounds=r.n_rounds,
                elapsed_ms=r.elapsed_ms,
            )

        def _bool_gate_fn(o):
            r = clean_fn(o)
            g = r.gate_out
            bad_g = GateOutput(
                history_weight=g.history_weight,
                probe_weight=g.probe_weight,
                mix_llrs=g.mix_llrs,
                mix_probe_pred=g.mix_probe_pred,
                mix_probe_var=np.ones_like(g.mix_probe_var, dtype=bool),
                mix_mean_ray=g.mix_mean_ray,
                mix_var_ray=g.mix_var_ray,
            )
            return AssimilationResult(
                gate_out=bad_g,
                decoded_info=r.decoded_info,
                assimilated_rows=r.assimilated_rows,
                row_weights=r.row_weights,
                row_margins=r.row_margins,
                n_rounds=r.n_rounds,
                elapsed_ms=r.elapsed_ms,
            )

        for bad_fn in (_bool_decoded_fn, _mixed_bool_list_decoded_fn, _datetime_gate_fn, _bool_gate_fn):
            self.assertFalse(audit_assimilation_anti_leakage(bad_fn, pub_obs, strict=False)["passed"])
            with self.assertRaises(ValueError):
                audit_assimilation_anti_leakage(bad_fn, pub_obs, strict=True)

    def test_cstar_margin_tol_allows_small_margin_ulp_while_rejecting_large_and_gate_drifts(self):
        clean_og = self._make_clean_offgrid_obs()
        pub_obs = PublicObservation(
            y_a=clean_og.y_a,
            y_b=clean_og.y_b,
            training_rows=SC_LAYOUTS["sparse"],
            probe_row=PROBE_ROW,
            n0=n0_from_snr(9.0),
            budget="sparse",
        )
        clean_fn = p11_audit._make_clean_cstar_assimilator()
        base_n0 = float(pub_obs.n0)

        def _make_drift_fn(*, margin_rel_drift: float = 0.0, weight_drift: float = 0.0, llr_drift: float = 0.0):
            def _fn(o: PublicObservation) -> AssimilationResult:
                r = clean_fn(o)
                is_scaled = not math.isclose(float(o.n0), base_n0, rel_tol=1e-15, abs_tol=0.0)
                if not is_scaled:
                    return r
                rm = dict(r.row_margins)
                if rm and margin_rel_drift != 0.0:
                    k0 = sorted(rm.keys())[0]
                    rm[k0] = float(rm[k0]) + margin_rel_drift * max(1.0, abs(float(rm[k0])))
                rw = dict(r.row_weights)
                if rw and weight_drift != 0.0:
                    k0 = sorted(rw.keys())[0]
                    rw[k0] = float(rw[k0]) + weight_drift
                g = r.gate_out
                if llr_drift != 0.0:
                    llr_scale = max(1.0, float(np.max(np.abs(g.mix_llrs))))
                    g = GateOutput(
                        history_weight=g.history_weight,
                        probe_weight=g.probe_weight,
                        mix_llrs=g.mix_llrs + llr_drift * llr_scale,
                        mix_probe_pred=g.mix_probe_pred,
                        mix_probe_var=g.mix_probe_var,
                        mix_mean_ray=g.mix_mean_ray,
                        mix_var_ray=g.mix_var_ray,
                    )
                return AssimilationResult(
                    gate_out=g,
                    decoded_info=r.decoded_info,
                    assimilated_rows=r.assimilated_rows,
                    row_weights=rw,
                    row_margins=rm,
                    n_rounds=r.n_rounds,
                    elapsed_ms=r.elapsed_ms,
                )
            return _fn

        # 1.4e-12 row_margins drift passes default cstar_margin_tol while keeping max_cstar_err < 1e-12
        rep_ok = audit_assimilation_anti_leakage(_make_drift_fn(margin_rel_drift=1.4e-12), pub_obs, strict=True)
        self.assertTrue(rep_ok["passed"])
        self.assertLess(rep_ok["max_cstar_err"], 1e-12)
        self.assertAlmostEqual(rep_ok["max_cstar_margin_err"], 1.4e-12, places=15)

        # Explicit strict cstar_margin_tol=1e-12 rejects 1.4e-12 row_margins drift
        self.assertFalse(
            audit_assimilation_anti_leakage(
                _make_drift_fn(margin_rel_drift=1.4e-12), pub_obs, cstar_margin_tol=1e-12, strict=False
            )["passed"]
        )

        # Large 1e-6 row_margins drift and >1e-12 GateOutput / row_weights drifts must all fail
        for kwargs in (
            {"margin_rel_drift": 1e-6},
            {"weight_drift": 2e-12},
            {"llr_drift": 2e-12},
        ):
            fn = _make_drift_fn(**kwargs)
            self.assertFalse(audit_assimilation_anti_leakage(fn, pub_obs, strict=False)["passed"], f"Missed {kwargs}")
            with self.assertRaises(ValueError):
                audit_assimilation_anti_leakage(fn, pub_obs, strict=True)

    def test_audit_fold_target_isolation(self):
        rng = np.random.default_rng(11010)
        y_a = ((rng.normal(size=(T, N)) + 1j * rng.normal(size=(T, N))) / math.sqrt(2.0)).astype(np.complex128)
        y_b = ((rng.normal(size=(T, N)) + 1j * rng.normal(size=(T, N))) / math.sqrt(2.0)).astype(np.complex128)
        y_b[PROBE_ROW] = complex(math.nan, math.nan)
        pub_obs = PublicObservation(
            y_a=y_a, y_b=y_b, training_rows=SC_LAYOUTS["sparse"],
            probe_row=PROBE_ROW, n0=n0_from_snr(9.0), budget="sparse",
        )
        aux = (2, 5, 8, 13, 16)
        targets = tuple(r for r in range(T) if r not in (*SC_LAYOUTS["sparse"], PROBE_ROW, *aux))

        def clean_fold(o: PublicObservation) -> dict:
            return {"aux": o.y_b[list(aux)].copy(), "branch": "reset"}

        self.assertTrue(audit_fold_target_isolation(clean_fold, pub_obs, targets, check_a_isolation=True, strict=True)["passed"])

        def leaky_fold(o: PublicObservation) -> dict:
            return {"bad": o.y_b[targets[0]].copy(), "branch": "reset"}

        self.assertFalse(audit_fold_target_isolation(leaky_fold, pub_obs, targets, strict=False)["passed"])


class P11StatisticalAndCohortAuditTests(unittest.TestCase):
    def test_p11_holm_families_g16_verdicts_zh_and_ni(self):
        pos16 = np.linspace(0.04, 0.14, 16)
        neg16 = -np.linspace(0.03, 0.11, 16)
        null16 = np.array([0.02, -0.02] * 8, dtype=np.float64)

        p1 = {
            "chained_vs_p9_jt": pos16,
            "chained_vs_payload_only": pos16 * 0.5,
        }
        p2 = {
            "transport_attn_vs_abs_no_transport": pos16,
            "transport_attn_vs_local": neg16,
            "transport_attn_vs_no_history": null16,
        }
        rep = p11_audit.audit_p11_holm_families(p1, p2, expected_g=16)
        self.assertEqual(rep["primary"]["exact_sign_patterns"], 65536)
        self.assertEqual(rep["primary"]["verdicts_zh"]["chained_vs_p9_jt"], "支持")
        self.assertEqual(rep["secondary"]["verdicts_zh"]["transport_attn_vs_abs_no_transport"], "支持")
        self.assertEqual(rep["secondary"]["verdicts_zh"]["transport_attn_vs_local"], "反证")
        self.assertEqual(rep["secondary"]["verdicts_zh"]["transport_attn_vs_no_history"], "证据不足")

        # NI: pass -> "支持", lower_95 >= margin -> "反证", wide SE fail -> "证据不足"
        rng = np.random.default_rng(11011)
        dense_bler = np.full(16, 0.084) + rng.normal(0.0, 0.002, size=16)
        sparse_bler = dense_bler + 0.002 + rng.normal(0.0, 0.001, size=16)
        dense_bits = np.full(16, 938.0) + rng.normal(0.0, 3.0, size=16)
        sparse_bits = dense_bits + 110.0 + rng.normal(0.0, 3.0, size=16)

        ni_ok = p11_audit.audit_engineering_noninferiority(sparse_bler, dense_bler, sparse_bits, dense_bits, margin=0.01)
        self.assertTrue(ni_ok["passes"])
        self.assertEqual(ni_ok["verdict_zh"], "支持")

        ni_ref = p11_audit.audit_engineering_noninferiority(
            dense_bler + 0.030, dense_bler, sparse_bits, dense_bits, margin=0.01
        )
        self.assertFalse(ni_ref["passes"])
        self.assertEqual(ni_ref["verdict_zh"], "反证")

        ni_insuff = p11_audit.audit_engineering_noninferiority(
            dense_bler + np.array([0.03, -0.02] * 8, dtype=np.float64),
            dense_bler, sparse_bits, dense_bits, margin=0.01,
        )
        self.assertFalse(ni_insuff["passes"])
        self.assertEqual(ni_insuff["verdict_zh"], "证据不足")

    def test_audit_p11_primary_11plus5_pooling_blindspot_and_mismatch_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            raw_path = tmp_dir / "primary.jsonl"
            sum_path = tmp_dir / "primary_sum.json"
            rep_path = tmp_dir / "primary_rep.jsonl"

            recs = []
            for c in range(1, 5):
                for l_idx, reg in enumerate(REGIMES_ID):
                    seed = SEED_P11_DEV + 1000 * c + l_idx
                    sp = {}
                    for arm, f0_e, f1_e in (
                        ("p11_chained_wavefront", 1, 0),
                        ("p11_payload_only", 3, 2),
                        ("p9_jeffreys_student_t", 4, 2),
                    ):
                        tot_e = f0_e + f1_e
                        sp[arm] = {
                            "blocks": 16,
                            "block_errors": tot_e,
                            "fold0_blocks": 11,
                            "fold1_blocks": 5,
                            "fold0_errors": f0_e,
                            "fold1_errors": f1_e,
                            "coded_bits": 2048,
                            "coded_errors": tot_e * 8,
                            "info_bits": 1024,
                            "info_errors": tot_e * 4,
                            "extra_blocks": 2,
                            "extra_errors": 0,
                            "probes": 32,
                            "probe_sse": 4.0,
                            "clean_sse": 1.0,
                            "probe_energy": 32.0,
                            "clean_energy": 30.0,
                            "variance_sum": 4.0,
                            "nll_sum": -10.0,
                            "total_assimilations": 4,
                            "false_assimilations": 0,
                            "wave1_fold0": "correct",
                            "wave1_fold1": "correct",
                        }
                    dn = {
                        "always_clear": {
                            "blocks": 16,
                            "block_errors": 1,
                            "fold0_blocks": 11,
                            "fold1_blocks": 5,
                            "fold0_errors": 1,
                            "fold1_errors": 0,
                            "coded_bits": 2048,
                            "coded_errors": 8,
                            "info_bits": 1024,
                            "info_errors": 4,
                            "extra_blocks": 0,
                            "extra_errors": 0,
                            "probes": 32,
                            "probe_sse": 4.0,
                            "clean_sse": 1.0,
                            "probe_energy": 32.0,
                            "clean_energy": 30.0,
                            "variance_sum": 4.0,
                            "nll_sum": -10.0,
                        }
                    }
                    recs.append({"cluster": c, "link": l_idx, "seed": seed, "regime": reg, "budgets": {"sparse": sp, "dense": dn}})

            raw_path.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
            rep_path.write_text("\n".join(json.dumps(r) for r in recs[:3]) + "\n", encoding="utf-8")

            arms_map = {"sparse": ("p11_chained_wavefront", "p11_payload_only", "p9_jeffreys_student_t"), "dense": ("always_clear",)}
            totals = p11_audit._aggregate_nested_budget_arm(recs, ("sparse", "dense"), arms_map)
            clusters = [
                p11_audit._aggregate_nested_budget_arm([r for r in recs if r["cluster"] == c], ("sparse", "dense"), arms_map)
                for c in range(1, 5)
            ]
            # Verify 11+5 pooling: (1+0)/16 = 1/16 = 0.0625 (NOT 0.5*(1/11 + 0/5) = 0.04545!)
            self.assertAlmostEqual(totals["sparse"]["p11_chained_wavefront"]["bler"], 1.0 / 16.0, places=14)

            c_vec = lambda b, a, m, rf=None: np.asarray([
                p11_audit.aggregate_p11_cells([r["budgets"][b][a] for r in recs if r["cluster"] == c and (rf is None or r["regime"] in rf)])[m]
                for c in range(1, 5)
            ], dtype=np.float64)
            tests = p11_audit._audit_single_holm_family(
                {
                    "chained_vs_p9_jt": {"baseline": c_vec("sparse", "p9_jeffreys_student_t", "bler"), "candidate": c_vec("sparse", "p11_chained_wavefront", "bler")},
                    "chained_vs_payload_only": {"baseline": c_vec("sparse", "p11_payload_only", "bler"), "candidate": c_vec("sparse", "p11_chained_wavefront", "bler")},
                },
                "primary",
            )["contrasts"]
            tests[0].update({"baseline": ["sparse", "p9_jeffreys_student_t"], "candidate": ["sparse", "p11_chained_wavefront"]})
            tests[1].update({"baseline": ["sparse", "p11_payload_only"], "candidate": ["sparse", "p11_chained_wavefront"]})
            ni = p11_audit.audit_engineering_noninferiority(
                c_vec("sparse", "p11_chained_wavefront", "bler", ["continuous"]),
                c_vec("dense", "always_clear", "bler", ["continuous"]),
                c_vec("sparse", "p11_chained_wavefront", "net_delivered_info_bits_per_frame", ["continuous"]),
                c_vec("dense", "always_clear", "net_delivered_info_bits_per_frame", ["continuous"]),
            )
            sum_obj = {
                "totals": totals,
                "clusters": clusters,
                "tests": tests,
                "noninferiority": {
                    **ni["bler_noninferiority"],
                    "candidate": ("sparse", "p11_chained_wavefront"),
                    "reference": ("dense", "always_clear"),
                    "regimes": ["continuous"],
                },
            }
            sum_path.write_text(json.dumps(sum_obj, indent=2) + "\n", encoding="utf-8")
            aud = p11_audit.audit_p11_primary(raw_path, sum_path, rep_path, require_canonical_family=True)
            self.assertTrue(aud["replay"]["passed"])
            self.assertIn("common_phase_blindspot", aud)
            self.assertAlmostEqual(aud["common_phase_blindspot"]["pooled_3id"]["target_bler"], 1.0 / 16.0, places=14)

            # Tampered NI verdict or lower_95 must fail audit_p11_primary
            bad_ni_verdict = json.loads(json.dumps(sum_obj))
            bad_ni_verdict["noninferiority"]["verdict"] = "supported" if sum_obj["noninferiority"]["verdict"] != "supported" else "refuted"
            bad_ni_verdict_path = tmp_dir / "bad_ni_verdict.json"
            bad_ni_verdict_path.write_text(json.dumps(bad_ni_verdict), encoding="utf-8")
            with self.assertRaises(ValueError):
                p11_audit.audit_p11_primary(raw_path, bad_ni_verdict_path)

            bad_ni_lower = json.loads(json.dumps(sum_obj))
            bad_ni_lower["noninferiority"]["lower_95"] = float(sum_obj["noninferiority"]["lower_95"]) + 0.05
            bad_ni_lower_path = tmp_dir / "bad_ni_lower.json"
            bad_ni_lower_path.write_text(json.dumps(bad_ni_lower), encoding="utf-8")
            with self.assertRaises(ValueError):
                p11_audit.audit_p11_primary(raw_path, bad_ni_lower_path)

            # Deliberate fold mismatch must fail check_p11_primary_cell
            bad_recs = json.loads(json.dumps(recs))
            bad_recs[0]["budgets"]["sparse"]["p11_chained_wavefront"]["fold1_errors"] = 1
            bad_raw = tmp_dir / "bad_primary.jsonl"
            bad_raw.write_text("\n".join(json.dumps(r) for r in bad_recs) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                p11_audit.audit_p11_primary(bad_raw, sum_path)

    def test_audit_p11_offgrid_18blocks_4id_2ood(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            raw_path = tmp_dir / "offgrid.jsonl"
            sum_path = tmp_dir / "offgrid_sum.json"
            rep_path = tmp_dir / "offgrid_rep.jsonl"

            recs = []
            arms = ("em10_transport_attn", "em10_abs_no_transport", "em10_local", "em10_no_history")
            for c in range(1, 5):
                for r_idx, reg in enumerate(OFFGRID_ALL_REGIMES):
                    seed = SEED_P11_DEV + 20000 + 1000 * c + r_idx
                    sp = {}
                    for arm, errs in zip(arms, (1, 3, 4, 5)):
                        sp[arm] = {
                            "blocks": 18,
                            "block_errors": errs,
                            "coded_bits": 2304,
                            "coded_errors": errs * 10,
                            "info_bits": 1152,
                            "info_errors": errs * 5,
                            "extra_blocks": 0,
                            "extra_errors": 0,
                            "probes": 32,
                            "probe_sse": 4.0,
                            "clean_sse": 1.0,
                            "probe_energy": 32.0,
                            "clean_energy": 30.0,
                            "variance_sum": 4.0,
                            "nll_sum": -10.0,
                        }
                    recs.append({"cluster": c, "link": r_idx, "seed": seed, "regime": reg, "budgets": {"sparse": sp}})

            raw_path.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
            rep_path.write_text("\n".join(json.dumps(r) for r in recs[:4]) + "\n", encoding="utf-8")

            arms_map = {"sparse": arms}
            totals = p11_audit._aggregate_nested_budget_arm(recs, ("sparse",), arms_map)
            clusters = [
                p11_audit._aggregate_nested_budget_arm([r for r in recs if r["cluster"] == c], ("sparse",), arms_map)
                for c in range(1, 5)
            ]
            c_vec = lambda a: np.asarray([
                p11_audit.aggregate_p11_cells([r["budgets"]["sparse"][a] for r in recs if r["cluster"] == c and r["regime"] in OFFGRID_REGIMES_ID])["bler"]
                for c in range(1, 5)
            ], dtype=np.float64)
            tests = p11_audit._audit_single_holm_family(
                {
                    "transport_attn_vs_abs_no_transport": {"baseline": c_vec("em10_abs_no_transport"), "candidate": c_vec("em10_transport_attn"), "regimes": list(OFFGRID_REGIMES_ID)},
                    "transport_attn_vs_local": {"baseline": c_vec("em10_local"), "candidate": c_vec("em10_transport_attn"), "regimes": list(OFFGRID_REGIMES_ID)},
                    "transport_attn_vs_no_history": {"baseline": c_vec("em10_no_history"), "candidate": c_vec("em10_transport_attn"), "regimes": list(OFFGRID_REGIMES_ID)},
                },
                "offgrid",
            )["contrasts"]
            tests[0].update({"baseline": ["sparse", "em10_abs_no_transport"], "candidate": ["sparse", "em10_transport_attn"]})
            tests[1].update({"baseline": ["sparse", "em10_local"], "candidate": ["sparse", "em10_transport_attn"]})
            tests[2].update({"baseline": ["sparse", "em10_no_history"], "candidate": ["sparse", "em10_transport_attn"]})

            id_equal = p11_audit._aggregate_nested_budget_arm(
                [r for r in recs if r["regime"] in OFFGRID_REGIMES_ID], ("sparse",), arms_map
            )
            ood_equal = p11_audit._aggregate_nested_budget_arm(
                [r for r in recs if r["regime"] in OFFGRID_REGIMES_OOD], ("sparse",), arms_map
            )
            c_vec_ood = lambda a: np.asarray([
                p11_audit.aggregate_p11_cells([
                    r["budgets"]["sparse"][a]
                    for r in recs
                    if r["cluster"] == c and r["regime"] in OFFGRID_REGIMES_OOD
                ])["bler"]
                for c in range(1, 5)
            ], dtype=np.float64)
            ood_tests_no_arms = p11_audit._audit_single_holm_family(
                {
                    "ood_transport_attn_vs_abs_no_transport": {"baseline": c_vec_ood("em10_abs_no_transport"), "candidate": c_vec_ood("em10_transport_attn"), "regimes": list(OFFGRID_REGIMES_OOD)},
                    "ood_transport_attn_vs_local": {"baseline": c_vec_ood("em10_local"), "candidate": c_vec_ood("em10_transport_attn"), "regimes": list(OFFGRID_REGIMES_OOD)},
                    "ood_transport_attn_vs_no_history": {"baseline": c_vec_ood("em10_no_history"), "candidate": c_vec_ood("em10_transport_attn"), "regimes": list(OFFGRID_REGIMES_OOD)},
                },
                "offgrid_ood",
            )["contrasts"]

            sum_obj_no_arms = {
                "totals": totals,
                "id_equal": id_equal,
                "ood_equal": ood_equal,
                "clusters": clusters,
                "tests": tests,
                "ood_tests": ood_tests_no_arms,
            }
            sum_path.write_text(json.dumps(sum_obj_no_arms, indent=2) + "\n", encoding="utf-8")

            aud = p11_audit.audit_p11_offgrid(raw_path, sum_path, rep_path, require_canonical_family=True)
            self.assertTrue(aud["replay"]["passed"])
            self.assertEqual(aud["cells"], len(recs) * 4)
            self.assertIn("ood_holm_family", aud)

            # Also verify explicit baseline/candidate metadata on ood_tests
            ood_tests_with_arms = json.loads(json.dumps(ood_tests_no_arms))
            ood_tests_with_arms[0].update({"baseline": ["sparse", "em10_abs_no_transport"], "candidate": ["sparse", "em10_transport_attn"]})
            ood_tests_with_arms[1].update({"baseline": ["sparse", "em10_local"], "candidate": ["sparse", "em10_transport_attn"]})
            ood_tests_with_arms[2].update({"baseline": ["sparse", "em10_no_history"], "candidate": ["sparse", "em10_transport_attn"]})
            sum_obj_with_arms = {**sum_obj_no_arms, "ood_tests": ood_tests_with_arms}
            sum_path.write_text(json.dumps(sum_obj_with_arms, indent=2) + "\n", encoding="utf-8")
            aud2 = p11_audit.audit_p11_offgrid(raw_path, sum_path, rep_path, require_canonical_family=True)
            self.assertTrue(aud2["replay"]["passed"])

            # Tampering id_equal, ood_equal, or ood_tests must raise ValueError
            for tamper_key in ("id_equal", "ood_equal", "ood_tests"):
                tampered = json.loads(json.dumps(sum_obj_with_arms))
                if tamper_key in ("id_equal", "ood_equal"):
                    tampered[tamper_key]["sparse"]["em10_transport_attn"]["bler"] += 0.05
                else:
                    tampered["ood_tests"][0]["mean_diff"] += 0.05
                bad_p = tmp_dir / f"bad_{tamper_key}.json"
                bad_p.write_text(json.dumps(tampered), encoding="utf-8")
                with self.assertRaises(ValueError):
                    p11_audit.audit_p11_offgrid(raw_path, bad_p)

    def test_verify_replay_exact_and_selftest_end_to_end(self):
        orig = [{"cluster": 1, "link": i, "seed": SEED_P11_DEV + i, "elapsed_ms": 10.0 + i, "v": 0.5 + i} for i in range(4)]
        rep_ok = [{"cluster": 1, "link": i, "seed": SEED_P11_DEV + i, "elapsed_ms": 99.0, "v": 0.5 + i + 1e-13} for i in (0, 2)]
        self.assertTrue(p11_audit.verify_replay_exact(orig, rep_ok, float_tol=1e-12, strict=True)["passed"])

        rep_bad = [{"cluster": 1, "link": 0, "seed": SEED_P11_DEV, "elapsed_ms": 99.0, "v": 0.5 + 1e-6}]
        self.assertFalse(p11_audit.verify_replay_exact(orig, rep_bad, float_tol=1e-12)["passed"])

        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "selftest.json"
            rep = p11_audit.run_selftest(output_path=out_path)
            self.assertEqual(rep["status"], "PASS")
            self.assertTrue(out_path.exists())
            parsed = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(parsed["status"], "PASS")
            self.assertTrue(parsed["firewall_checks"]["passed"])
            self.assertTrue(parsed["g16_statistical_verifier"]["passed"])


class FormalStudyPrimaryChainedMockTests(unittest.TestCase):
    def _make_mock_obs_and_truth(self):
        rng = np.random.default_rng(11055)
        n0 = n0_from_snr(9.0)
        y_a = ((rng.normal(size=(T, N)) + 1j * rng.normal(size=(T, N))) / math.sqrt(2.0)).astype(np.complex128)
        y_b = ((rng.normal(size=(T, N)) + 1j * rng.normal(size=(T, N))) / math.sqrt(2.0)).astype(np.complex128)
        y_b[PROBE_ROW] = complex(math.nan, math.nan)
        obs = PublicObservation(
            y_a=y_a,
            y_b=y_b,
            training_rows=SC_LAYOUTS["sparse"],
            probe_row=PROBE_ROW,
            n0=n0,
            budget="sparse",
        )
        b_info = rng.integers(0, 2, size=(18, 64), dtype=np.uint8)
        b_code = rng.integers(0, 2, size=(18, 128), dtype=np.uint8)
        y_clean = ((rng.normal(size=N) + 1j * rng.normal(size=N)) / math.sqrt(2.0)).astype(np.complex128)
        y_noisy = y_clean + (rng.normal(size=N) + 1j * rng.normal(size=N)) * math.sqrt(n0 / 2.0)
        truth = SimpleNamespace(
            b_info=b_info,
            b_code=b_code,
            y_b_probe_noisy=y_noisy,
            y_b_probe_clean=y_clean,
            regime="full_reset",
        )
        decoded_info = {r: b_info[i].copy() for i, r in enumerate(CORE_ROWS)}
        decoded_info[0] = b_info[16].copy()
        decoded_info[9] = b_info[17].copy()
        gate_out = GateOutput(
            history_weight=np.full(T, 0.2, dtype=np.float64),
            probe_weight=0.2,
            mix_llrs=np.zeros((T, N, 4), dtype=np.float64),
            mix_probe_pred=y_clean.copy(),
            mix_probe_var=np.full(N, n0 * 1.2, dtype=np.float64),
            mix_mean_ray=np.zeros((3, T, N), dtype=np.complex128),
            mix_var_ray=np.full((3, T, N), n0, dtype=np.float64),
        )
        assim_res = AssimilationResult(
            gate_out=gate_out,
            decoded_info=decoded_info,
            assimilated_rows=(9, 11, 8),
            row_weights={9: 0.7, 11: 0.6, 8: 0.5},
            row_margins={9: 10.0, 11: 9.0, 8: 8.0},
            n_rounds=4,
            elapsed_ms=1.5,
        )
        return obs, truth, assim_res

    def test_score_primary_chained_m2_ledger_and_wave1_schema(self):
        from p11.formal_study import _score_primary_chained

        obs, truth, assim_res = self._make_mock_obs_and_truth()
        # Case 1: exact M2 trace schema (`injected_info`, `wave: 0`, no `wave1_fold*`/`wave1_rows`/`injected_word`)
        # - Fold 0 Wave 0 row 9 (extra index 17): true injected_info -> wave1_fold0 == "correct"
        # - Fold 1 Wave 0 row 11 (core index CORE_ROWS.index(11)): flipped injected_info -> wave1_fold1 == "incorrect"
        #   even though assim_res.decoded_info[11] matches truth!
        # - Fold 1 Wave 1 row 8 (core index CORE_ROWS.index(8)): flipped injected_info -> false_assimilations == 2
        w9_true = truth.b_info[17].copy()
        w11_bad = truth.b_info[CORE_ROWS.index(11)].copy()
        w11_bad[0] ^= 1
        w8_bad = truth.b_info[CORE_ROWS.index(8)].copy()
        w8_bad[1] ^= 1
        trace_m2 = {
            "ledger": [
                {"fold": 0, "wave": 0, "row": 9, "injected_info": w9_true, "alpha": 0.7, "margin": 10.0, "state_version": 1, "branch": "reset"},
                {"fold": 1, "wave": 0, "row": 11, "injected_info": w11_bad, "alpha": 0.6, "margin": 9.0, "state_version": 1, "branch": "reset"},
                {"fold": 1, "wave": 1, "row": 8, "injected_info": w8_bad, "alpha": 0.5, "margin": 8.0, "state_version": 2, "branch": "reset"},
            ],
            "folds": (
                {"fold": 0, "ledger": []},
                {"fold": 1, "ledger": []},
            ),
            "tau_switch": 3.5,
            "aux_temp": 1.0,
        }

        class _MockWF:
            def run_p11_with_trace(self, _obs):
                return assim_res, trace_m2

        scored = _score_primary_chained(_MockWF(), obs, truth)
        self.assertEqual(scored["wave1_fold0"], "correct")
        self.assertEqual(scored["wave1_fold1"], "incorrect")
        self.assertEqual(scored["total_assimilations"], 3)
        self.assertEqual(scored["false_assimilations"], 2)

        # Case 2: empty top-level ledger, fold 0 has wave=0 in folds[0]["ledger"], fold 1 has only wave=1 -> absent
        trace_fold_only = {
            "ledger": [],
            "folds": (
                {"fold": 0, "ledger": [{"fold": 0, "wave": 0, "row": 9, "injected_info": w9_true}]},
                {"fold": 1, "ledger": [{"fold": 1, "wave": 1, "row": 8, "injected_info": w8_bad}]},
            ),
        }

        class _MockWFFoldOnly:
            def run_p11_with_trace(self, _obs):
                return assim_res, trace_fold_only

        scored2 = _score_primary_chained(_MockWFFoldOnly(), obs, truth)
        self.assertEqual(scored2["wave1_fold0"], "correct")
        self.assertEqual(scored2["wave1_fold1"], "absent")

        # Case 3: illegal row (PROBE_ROW=19) in ledger must raise ValueError
        trace_illegal = {
            "ledger": [{"fold": 0, "wave": 0, "row": PROBE_ROW, "injected_info": w9_true}],
            "folds": ({"fold": 0, "ledger": []}, {"fold": 1, "ledger": []}),
        }

        class _MockWFIllegal:
            def run_p11_with_trace(self, _obs):
                return assim_res, trace_illegal

        with self.assertRaises(ValueError):
            _score_primary_chained(_MockWFIllegal(), obs, truth)


class P11ReportValidationTests(unittest.TestCase):
    def _build_valid_g4_bundle(self) -> tuple[dict, dict, dict]:
        pri_recs = []
        for c in range(1, 5):
            for l_idx, reg in enumerate(REGIMES_ID):
                sp = {
                    arm: {
                        "blocks": 16,
                        "block_errors": err,
                        "fold0_blocks": 11,
                        "fold1_blocks": 5,
                        "fold0_errors": max(0, err - 1),
                        "fold1_errors": min(1, err),
                        "coded_bits": 2048,
                        "coded_errors": err * 8,
                        "info_bits": 1024,
                        "info_errors": err * 4,
                        "extra_blocks": 2,
                        "extra_errors": 0,
                        "probes": 32,
                        "probe_sse": 4.0,
                        "clean_sse": 1.0,
                        "probe_energy": 32.0,
                        "clean_energy": 30.0,
                        "variance_sum": 4.0,
                        "nll_sum": -10.0,
                        "total_assimilations": 4,
                        "false_assimilations": 0,
                        "wave1_fold0": "correct",
                        "wave1_fold1": "correct",
                    }
                    for arm, err in (("p11_chained_wavefront", 1), ("p11_payload_only", 3), ("p9_jeffreys_student_t", 4))
                }
                dn = {
                    "always_clear": {
                        "blocks": 16,
                        "block_errors": 1,
                        "fold0_blocks": 11,
                        "fold1_blocks": 5,
                        "fold0_errors": 1,
                        "fold1_errors": 0,
                        "coded_bits": 2048,
                        "coded_errors": 8,
                        "info_bits": 1024,
                        "info_errors": 4,
                        "extra_blocks": 0,
                        "extra_errors": 0,
                        "probes": 32,
                        "probe_sse": 4.0,
                        "clean_sse": 1.0,
                        "probe_energy": 32.0,
                        "clean_energy": 30.0,
                        "variance_sum": 4.0,
                        "nll_sum": -10.0,
                    }
                }
                pri_recs.append({"cluster": c, "link": l_idx, "seed": SEED_P11_DEV + c * 100 + l_idx, "regime": reg, "budgets": {"sparse": sp, "dense": dn}})

        p_arms = {"sparse": ("p11_chained_wavefront", "p11_payload_only", "p9_jeffreys_student_t"), "dense": ("always_clear",)}
        p_totals = p11_audit._aggregate_nested_budget_arm(pri_recs, ("sparse", "dense"), p_arms)
        p_clusters = [p11_audit._aggregate_nested_budget_arm([r for r in pri_recs if r["cluster"] == c], ("sparse", "dense"), p_arms) for c in range(1, 5)]
        p_by_reg = {reg: p11_audit._aggregate_nested_budget_arm([r for r in pri_recs if r["regime"] == reg], ("sparse", "dense"), p_arms) for reg in REGIMES_ID}
        pc = lambda b, a, m, rf=None: np.asarray([
            p11_audit.aggregate_p11_cells([r["budgets"][b][a] for r in pri_recs if r["cluster"] == c and (rf is None or r["regime"] in rf)])[m]
            for c in range(1, 5)
        ], dtype=np.float64)
        p_tests = p11_audit._audit_single_holm_family(
            {
                "chained_vs_p9_jt": {"baseline": pc("sparse", "p9_jeffreys_student_t", "bler"), "candidate": pc("sparse", "p11_chained_wavefront", "bler")},
                "chained_vs_payload_only": {"baseline": pc("sparse", "p11_payload_only", "bler"), "candidate": pc("sparse", "p11_chained_wavefront", "bler")},
            },
            "primary",
        )["contrasts"]
        p_ni = p11_audit.audit_engineering_noninferiority(
            pc("sparse", "p11_chained_wavefront", "bler", ["continuous"]),
            pc("dense", "always_clear", "bler", ["continuous"]),
            pc("sparse", "p11_chained_wavefront", "net_delivered_info_bits_per_frame", ["continuous"]),
            pc("dense", "always_clear", "net_delivered_info_bits_per_frame", ["continuous"]),
        )
        pri_sum = {
            "totals": p_totals,
            "clusters": p_clusters,
            "by_regime": p_by_reg,
            "tests": p_tests,
            "noninferiority": {**p_ni["bler_noninferiority"], "candidate": ["sparse", "p11_chained_wavefront"], "reference": ["dense", "always_clear"], "regimes": ["continuous"]},
            "noninferiority_full": p_ni,
            "common_phase_blindspot": p11_audit.summarize_common_phase_blindspot(pri_recs),
        }

        sec_recs = []
        s_arms = ("em10_transport_attn", "em10_abs_no_transport", "em10_local", "em10_no_history")
        for c in range(1, 5):
            for r_idx, reg in enumerate(OFFGRID_ALL_REGIMES):
                sp = {
                    arm: {
                        "blocks": 18, "block_errors": err, "coded_bits": 2304, "coded_errors": err * 8,
                        "info_bits": 1152, "info_errors": err * 4, "extra_blocks": 0, "extra_errors": 0,
                        "probes": 32, "probe_sse": 4.0, "clean_sse": 1.0, "probe_energy": 32.0, "clean_energy": 30.0,
                        "variance_sum": 4.0, "nll_sum": -10.0,
                    }
                    for arm, err in zip(s_arms, (1, 3, 4, 5))
                }
                sec_recs.append({"cluster": c, "link": r_idx, "seed": SEED_P11_DEV + 5000 + c * 100 + r_idx, "regime": reg, "budgets": {"sparse": sp}})
        s_map = {"sparse": s_arms}
        s_totals = p11_audit._aggregate_nested_budget_arm(sec_recs, ("sparse",), s_map)
        s_id_eq = p11_audit._aggregate_nested_budget_arm([r for r in sec_recs if r["regime"] in OFFGRID_REGIMES_ID], ("sparse",), s_map)
        s_ood_eq = p11_audit._aggregate_nested_budget_arm([r for r in sec_recs if r["regime"] in OFFGRID_REGIMES_OOD], ("sparse",), s_map)
        s_clusters = [p11_audit._aggregate_nested_budget_arm([r for r in sec_recs if r["cluster"] == c], ("sparse",), s_map) for c in range(1, 5)]
        s_by_reg = {reg: p11_audit._aggregate_nested_budget_arm([r for r in sec_recs if r["regime"] == reg], ("sparse",), s_map) for reg in OFFGRID_ALL_REGIMES}
        sc = lambda a, regs: np.asarray([
            p11_audit.aggregate_p11_cells([r["budgets"]["sparse"][a] for r in sec_recs if r["cluster"] == c and r["regime"] in regs])["bler"]
            for c in range(1, 5)
        ], dtype=np.float64)
        s_tests = p11_audit._audit_single_holm_family(
            {
                "transport_attn_vs_abs_no_transport": {"baseline": sc("em10_abs_no_transport", OFFGRID_REGIMES_ID), "candidate": sc("em10_transport_attn", OFFGRID_REGIMES_ID)},
                "transport_attn_vs_local": {"baseline": sc("em10_local", OFFGRID_REGIMES_ID), "candidate": sc("em10_transport_attn", OFFGRID_REGIMES_ID)},
                "transport_attn_vs_no_history": {"baseline": sc("em10_no_history", OFFGRID_REGIMES_ID), "candidate": sc("em10_transport_attn", OFFGRID_REGIMES_ID)},
            },
            "offgrid",
        )["contrasts"]
        s_ood_tests = p11_audit._audit_single_holm_family(
            {
                "ood_transport_attn_vs_abs_no_transport": {"baseline": sc("em10_abs_no_transport", OFFGRID_REGIMES_OOD), "candidate": sc("em10_transport_attn", OFFGRID_REGIMES_OOD)},
                "ood_transport_attn_vs_local": {"baseline": sc("em10_local", OFFGRID_REGIMES_OOD), "candidate": sc("em10_transport_attn", OFFGRID_REGIMES_OOD)},
                "ood_transport_attn_vs_no_history": {"baseline": sc("em10_no_history", OFFGRID_REGIMES_OOD), "candidate": sc("em10_transport_attn", OFFGRID_REGIMES_OOD)},
            },
            "offgrid_ood",
        )["contrasts"]
        sec_sum = {
            "totals": s_totals,
            "id_equal": s_id_eq,
            "ood_equal": s_ood_eq,
            "clusters": s_clusters,
            "by_regime": s_by_reg,
            "tests": s_tests,
            "ood_tests": s_ood_tests,
        }
        aud_obj = {
            "mode": "dry_run",
            "axis": "both",
            "freeze_sha256": None,
            "seed_disjointness": {"disjoint": True},
            "primary": {"audit": {"clusters": 4, "links": len(pri_recs), "max_aggregate_diff": 0.0, "max_stat_diff": 0.0}, "replay_verify": {"passed": True}},
            "secondary": {"audit": {"clusters": 4, "links": len(sec_recs), "max_aggregate_diff": 0.0, "max_stat_diff": 0.0}, "replay_verify": {"passed": True}},
        }
        return pri_sum, sec_sum, aud_obj

    def test_report_cli_and_schema_validation(self):
        pri_sum, sec_sum, aud_obj = self._build_valid_g4_bundle()

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            out_r = tmp_dir / "R.md"
            out_s = tmp_dir / "S.md"

            # 1. Nonexistent directory
            with self.assertRaises(FileNotFoundError):
                p11_report.main(["--results-dir", str(tmp_dir / "missing"), "--out-report", str(out_r), "--out-sync", str(out_s), "--no-verify-freeze"])

            # 2. Empty directory (default & --allow-partial)
            empty_dir = tmp_dir / "empty"
            empty_dir.mkdir()
            with self.assertRaises(FileNotFoundError):
                p11_report.main(["--results-dir", str(empty_dir), "--out-report", str(out_r), "--out-sync", str(out_s), "--no-verify-freeze"])
            with self.assertRaises(FileNotFoundError):
                p11_report.main(["--results-dir", str(empty_dir), "--out-report", str(out_r), "--out-sync", str(out_s), "--no-verify-freeze", "--allow-partial"])

            # 3. Malformed {"g": 8, "clusters": []} rejected even with --allow-partial --no-verify-freeze
            bad_dir = tmp_dir / "malformed"
            bad_dir.mkdir()
            (bad_dir / "p11_primary_summary.json").write_text(json.dumps({"g": 8, "clusters": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                p11_report.main(["--results-dir", str(bad_dir), "--out-report", str(out_r), "--out-sync", str(out_s), "--no-verify-freeze", "--allow-partial"])
            with self.assertRaises(ValueError):
                p11_report.render_reports(primary_summary={"g": 8, "clusters": []})
            with self.assertRaises(ValueError):
                p11_report.render_reports()

            # 4. Valid dry_run axis="both" renders in default mode
            good_dir = tmp_dir / "good_both"
            good_dir.mkdir()
            (good_dir / "p11_primary_summary.json").write_text(json.dumps(pri_sum), encoding="utf-8")
            (good_dir / "p11_secondary_summary.json").write_text(json.dumps(sec_sum), encoding="utf-8")
            (good_dir / "p11_audit.json").write_text(json.dumps(aud_obj), encoding="utf-8")
            p11_report.main(["--results-dir", str(good_dir), "--out-report", str(out_r), "--out-sync", str(out_s), "--no-verify-freeze"])
            self.assertTrue(out_r.is_file() and out_s.is_file())

            # 5. Missing secondary when audit declares axis="both" -> fails default, passes --allow-partial
            partial_dir = tmp_dir / "partial_missing_sec"
            partial_dir.mkdir()
            (partial_dir / "p11_primary_summary.json").write_text(json.dumps(pri_sum), encoding="utf-8")
            (partial_dir / "p11_audit.json").write_text(json.dumps(aud_obj), encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                p11_report.main(["--results-dir", str(partial_dir), "--out-report", str(out_r), "--out-sync", str(out_s), "--no-verify-freeze"])
            p11_report.main(["--results-dir", str(partial_dir), "--out-report", str(out_r), "--out-sync", str(out_s), "--no-verify-freeze", "--allow-partial"])

            # 6. Missing p11_audit.json -> fails default, passes --allow-partial
            no_aud_dir = tmp_dir / "no_audit"
            no_aud_dir.mkdir()
            (no_aud_dir / "p11_secondary_summary.json").write_text(json.dumps(sec_sum), encoding="utf-8")
            with self.assertRaises(FileNotFoundError):
                p11_report.main(["--results-dir", str(no_aud_dir), "--out-report", str(out_r), "--out-sync", str(out_s), "--no-verify-freeze"])
            p11_report.main(["--results-dir", str(no_aud_dir), "--out-report", str(out_r), "--out-sync", str(out_s), "--no-verify-freeze", "--allow-partial"])

            # 7. Valid dry_run axis="secondary" in default mode
            sec_only_dir = tmp_dir / "sec_only"
            sec_only_dir.mkdir()
            (sec_only_dir / "p11_secondary_summary.json").write_text(json.dumps(sec_sum), encoding="utf-8")
            sec_aud = {"mode": "dry_run", "axis": "secondary", "seed_disjointness": {"disjoint": True}, "secondary": aud_obj["secondary"]}
            (sec_only_dir / "p11_audit.json").write_text(json.dumps(sec_aud), encoding="utf-8")
            p11_report.main(["--results-dir", str(sec_only_dir), "--out-report", str(out_r), "--out-sync", str(out_s), "--no-verify-freeze"])

            # 8. Formal mode with g=4 != 16, non-dict JSON, missing NI lower_95, or missing secondary id_equal
            bad_formal_dir = tmp_dir / "bad_formal"
            bad_formal_dir.mkdir()
            (bad_formal_dir / "p11_primary_summary.json").write_text(json.dumps(pri_sum), encoding="utf-8")
            (bad_formal_dir / "p11_audit.json").write_text(
                json.dumps({"mode": "primary", "axis": "primary", "primary": aud_obj["primary"]}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                p11_report.main(["--results-dir", str(bad_formal_dir), "--out-report", str(out_r), "--out-sync", str(out_s), "--no-verify-freeze"])

            bad_ni_dir = tmp_dir / "bad_ni"
            bad_ni_dir.mkdir()
            pri_bad_ni = json.loads(json.dumps(pri_sum))
            pri_bad_ni.pop("noninferiority_full", None)
            pri_bad_ni["noninferiority"].pop("lower_95", None)
            (bad_ni_dir / "p11_primary_summary.json").write_text(json.dumps(pri_bad_ni), encoding="utf-8")
            (bad_ni_dir / "p11_audit.json").write_text(json.dumps({"mode": "dry_run", "axis": "primary", "primary": aud_obj["primary"]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                p11_report.main(["--results-dir", str(bad_ni_dir), "--out-report", str(out_r), "--out-sync", str(out_s), "--no-verify-freeze"])

            bad_sec_dir = tmp_dir / "bad_sec"
            bad_sec_dir.mkdir()
            sec_bad = json.loads(json.dumps(sec_sum))
            sec_bad.pop("id_equal", None)
            (bad_sec_dir / "p11_secondary_summary.json").write_text(json.dumps(sec_bad), encoding="utf-8")
            (bad_sec_dir / "p11_audit.json").write_text(json.dumps(sec_aud), encoding="utf-8")
            with self.assertRaises(ValueError):
                p11_report.main(["--results-dir", str(bad_sec_dir), "--out-report", str(out_r), "--out-sync", str(out_s), "--no-verify-freeze"])

            non_dict_dir = tmp_dir / "non_dict"
            non_dict_dir.mkdir()
            (non_dict_dir / "p11_primary_summary.json").write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                p11_report.main(["--results-dir", str(non_dict_dir), "--out-report", str(out_r), "--out-sync", str(out_s), "--no-verify-freeze", "--allow-partial"])


class P11FreezeDefaultManifestTests(unittest.TestCase):
    def test_parse_porcelain_entries_preserves_leading_space(self):
        sample = " M p11/freeze.py\n?? p11/results/freeze_manifest.json\n"
        self.assertEqual(
            p11_freeze._parse_porcelain_entries(sample),
            [(" M", "p11/freeze.py"), ("??", "p11/results/freeze_manifest.json")],
        )
        self.assertEqual(
            p11_freeze._parse_porcelain_paths(sample),
            ["p11/freeze.py", "p11/results/freeze_manifest.json"],
        )

    def test_untracked_vs_modified_default_manifest_guardrails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_manifest = Path(tmp) / "freeze_manifest.json"
            p11_freeze.create_freeze(manifest_path=tmp_manifest, allow_dirty_agents=True)
            manifest_dict = json.loads(tmp_manifest.read_text(encoding="utf-8"))

            orig_run_git = p11_freeze._run_git
            fake_status = "?? p11/results/freeze_manifest.json\n"

            def _fake_git(args: list[str], *, strip: bool = True) -> str:
                if "status" in args and "--porcelain" in args:
                    return fake_status.strip() if strip else fake_status.rstrip("\r\n")
                return orig_run_git(args, strip=strip)

            try:
                p11_freeze._run_git = _fake_git
                # Untracked default manifest allowed ONLY when allow_untracked_manifest=True
                p11_freeze.assert_repo_guardrails(
                    allow_dirty_agents=True,
                    manifest=manifest_dict,
                    manifest_path=p11_freeze.DEFAULT_MANIFEST_PATH,
                    allow_untracked_manifest=True,
                )
                with self.assertRaises(ValueError):
                    p11_freeze.assert_repo_guardrails(
                        allow_dirty_agents=True,
                        manifest=manifest_dict,
                        manifest_path=p11_freeze.DEFAULT_MANIFEST_PATH,
                        allow_untracked_manifest=False,
                    )

                # Modified default manifest (leading-space ' M') rejected under BOTH True and False
                fake_status = " M p11/results/freeze_manifest.json\n"
                for flag in (True, False):
                    with self.assertRaises(ValueError):
                        p11_freeze.assert_repo_guardrails(
                            allow_dirty_agents=True,
                            manifest=manifest_dict,
                            manifest_path=p11_freeze.DEFAULT_MANIFEST_PATH,
                            allow_untracked_manifest=flag,
                        )
            finally:
                p11_freeze._run_git = orig_run_git

            # CLI --write and --verify in temp dir
            p11_freeze.main(["--write", "--force", "--manifest", str(tmp_manifest), "--allow-dirty-agents"])
            p11_freeze.main(["--verify", "--manifest", str(tmp_manifest), "--allow-dirty-agents"])


if __name__ == "__main__":
    unittest.main()
