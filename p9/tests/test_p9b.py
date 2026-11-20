"""Unit tests for P9-B statistical inference, strict pre-update gating, and C* invariance."""
from __future__ import annotations

import unittest

import numpy as np

from p8 import coding, link as ofdm
from p9.contract import (
    CORE_ROWS,
    CP,
    DELAYS,
    FRAME_DURATION,
    N,
    PROBE_ROW,
    SEED_P9_DEV,
    T,
    GateOutput,
    PublicObservation,
    jeffreys_complex_gaussian,
    mix_bit_llrs,
    n0_from_snr,
)
from p9.gating import (
    build_candidates,
    build_candidates_with_priors,
    build_prior_states,
    combine_candidates,
    complex_gaussian_loglik,
    gate_always_clear,
    gate_always_inherit,
    gate_jeffreys_student_t,
    gate_pre_bayes_factor,
    gate_split_predictive,
    post_update_jeffreys_rays,
    post_update_jeffreys_rows,
    pre_update_jeffreys,
    predictive_distribution,
    score_codeword_mixture,
    split_block_predictive_scores,
)
from p9.p9b_dev import frame_b_ray_gains, simulate_link


class TestP9BGating(unittest.TestCase):
    def test_strict_pre_update_invariance_no_post_fit_cheating(self):
        """Verify pre_train_pred, pre_train_var, and pre_train_loglik are computed strictly
        BEFORE conditioning on obs.y_b[training_rows] (never using post-fit residuals).
        """
        for budget in ("sparse", "dense"):
            obs, _ = simulate_link(SEED_P9_DEV + 101, "continuous", snr_db=9.0, budget=budget)
            st_inh_prior, st_clr_prior = build_prior_states(obs)
            cand_inh, cand_clr = build_candidates(obs)

            rows = obs.training_rows
            y_tr = obs.y_b[list(rows)].reshape(-1)
            for st_prior, cand in ((st_inh_prior, cand_inh), (st_clr_prior, cand_clr)):
                mu_ref, cov_ref = predictive_distribution(st_prior, rows, n0=obs.n0)
                ll_ref = complex_gaussian_loglik(y_tr, mu_ref, cov_ref)
                np.testing.assert_allclose(cand.pre_train_pred, mu_ref.reshape(len(rows), N), atol=1e-14, rtol=0.0)
                np.testing.assert_allclose(
                    cand.pre_train_var, np.real(np.diag(cov_ref)).reshape(len(rows), N), atol=1e-14, rtol=0.0
                )
                self.assertAlmostEqual(cand.pre_train_loglik, ll_ref, places=12)

            # Mutating obs.y_b[training_rows] must NOT alter pre_train_pred or pre_train_var
            # (proving the predictive distribution never peeked at y_b[training_rows]),
            # while it MUST alter pre_train_loglik and post-update mean_ray.
            y_b_alt = obs.y_b.copy()
            y_b_alt[list(rows)] += 2.5 - 1.7j
            obs_alt = PublicObservation(
                y_a=obs.y_a,
                y_b=y_b_alt,
                training_rows=obs.training_rows,
                probe_row=obs.probe_row,
                n0=obs.n0,
                budget=obs.budget,
            )
            cand_inh_alt, cand_clr_alt = build_candidates(obs_alt)
            np.testing.assert_array_equal(cand_inh.pre_train_pred, cand_inh_alt.pre_train_pred)
            np.testing.assert_array_equal(cand_inh.pre_train_var, cand_inh_alt.pre_train_var)
            np.testing.assert_array_equal(cand_clr.pre_train_pred, cand_clr_alt.pre_train_pred)
            np.testing.assert_array_equal(cand_clr.pre_train_var, cand_clr_alt.pre_train_var)
            self.assertNotAlmostEqual(cand_inh.pre_train_loglik, cand_inh_alt.pre_train_loglik, places=3)
            self.assertNotAlmostEqual(cand_clr.pre_train_loglik, cand_clr_alt.pre_train_loglik, places=3)

            # Verify 'clear' pre_train_pred is identically zero (strictly prior to B fitting!)
            # whereas post-update mean_ray on the training row is non-zero.
            np.testing.assert_array_equal(cand_clr.pre_train_pred, np.zeros_like(cand_clr.pre_train_pred))
            self.assertGreater(float(np.max(np.abs(cand_clr.mean_ray[:, rows[0], :]))), 0.1)

    def test_heldout_probe_isolation_exact_zero_change(self):
        """Poisoning obs.y_b[PROBE_ROW] with 1e9 (or arbitrary huge complex numbers) must
        cause EXACT bit-for-bit zero change in build_candidates and every gate.
        """
        for budget in ("sparse", "dense"):
            obs_nan, _ = simulate_link(SEED_P9_DEV + 202, "partial_shift", snr_db=9.0, budget=budget)
            y_b_poison = obs_nan.y_b.copy()
            y_b_poison[PROBE_ROW] = 1e9 - 1e9j
            obs_poison = PublicObservation(
                y_a=obs_nan.y_a,
                y_b=y_b_poison,
                training_rows=obs_nan.training_rows,
                probe_row=obs_nan.probe_row,
                n0=obs_nan.n0,
                budget=obs_nan.budget,
            )

            st_inh_1, st_clr_1, inh_1, clr_1 = build_candidates_with_priors(obs_nan)
            st_inh_2, st_clr_2, inh_2, clr_2 = build_candidates_with_priors(obs_poison)

            for c1, c2 in ((inh_1, inh_2), (clr_1, clr_2)):
                self.assertEqual(c1.pre_train_loglik, c2.pre_train_loglik)
                np.testing.assert_array_equal(c1.pre_train_pred, c2.pre_train_pred)
                np.testing.assert_array_equal(c1.pre_train_var, c2.pre_train_var)
                np.testing.assert_array_equal(c1.mean_ray, c2.mean_ray)
                np.testing.assert_array_equal(c1.var_ray, c2.var_ray)
                np.testing.assert_array_equal(c1.probe_pred, c2.probe_pred)
                np.testing.assert_array_equal(c1.probe_var, c2.probe_var)
                np.testing.assert_array_equal(c1.llrs, c2.llrs)
                self.assertTrue(np.isfinite(c2.llrs).all())
                self.assertTrue(np.isfinite(c2.probe_pred).all())
                self.assertTrue(np.isfinite(c2.probe_var).all())

            gates_1 = [
                gate_always_inherit(inh_1, clr_1),
                gate_always_clear(inh_1, clr_1),
                gate_pre_bayes_factor(inh_1, clr_1),
                gate_split_predictive(obs_nan, st_inh_1, st_clr_1, inh_1, clr_1),
                gate_jeffreys_student_t(obs_nan, inh_1, clr_1),
            ]
            gates_2 = [
                gate_always_inherit(inh_2, clr_2),
                gate_always_clear(inh_2, clr_2),
                gate_pre_bayes_factor(inh_2, clr_2),
                gate_split_predictive(obs_poison, st_inh_2, st_clr_2, inh_2, clr_2),
                gate_jeffreys_student_t(obs_poison, inh_2, clr_2),
            ]
            for g1, g2 in zip(gates_1, gates_2):
                np.testing.assert_array_equal(g1.history_weight, g2.history_weight)
                self.assertEqual(g1.probe_weight, g2.probe_weight)
                np.testing.assert_array_equal(g1.mix_llrs, g2.mix_llrs)
                np.testing.assert_array_equal(g1.mix_probe_pred, g2.mix_probe_pred)
                np.testing.assert_array_equal(g1.mix_probe_var, g2.mix_probe_var)
                np.testing.assert_array_equal(g1.mix_mean_ray, g2.mix_mean_ray)
                np.testing.assert_array_equal(g1.mix_var_ray, g2.mix_var_ray)

    def test_exact_total_variance_identity_and_mix_bit_llrs_boundaries(self):
        """Verify mix_probe_var == within + between, mix_var_ray == within + between,
        and exact probability-space mix_bit_llrs behavior at w=1, w=0, and interior w.
        """
        obs, _ = simulate_link(SEED_P9_DEV + 303, "full_reset", snr_db=9.0, budget="sparse")
        cand_inh, cand_clr = build_candidates(obs)

        # Interior non-trivial row-varying weights
        w_rows = np.linspace(0.05, 0.95, T, dtype=np.float64)
        w_probe = 0.37
        mixed = combine_candidates(w_rows, w_probe, cand_inh, cand_clr)

        # Exact probe total variance = within + between
        within_probe = w_probe * cand_inh.probe_var + (1.0 - w_probe) * cand_clr.probe_var
        between_probe = w_probe * (1.0 - w_probe) * (np.abs(cand_inh.probe_pred - cand_clr.probe_pred) ** 2)
        np.testing.assert_allclose(mixed.mix_probe_var, within_probe + between_probe, atol=1e-15, rtol=1e-14)
        self.assertGreater(float(np.max(between_probe)), 1e-3)
        self.assertTrue(np.all(mixed.mix_probe_var >= within_probe - 1e-15))

        # Exact ray total variance = within + between
        w3 = w_rows[None, :, None]
        within_ray = w3 * cand_inh.var_ray + (1.0 - w3) * cand_clr.var_ray
        between_ray = w3 * (1.0 - w3) * (np.abs(cand_inh.mean_ray - cand_clr.mean_ray) ** 2)
        np.testing.assert_allclose(mixed.mix_var_ray, within_ray + between_ray, atol=1e-15, rtol=1e-14)

        # Exact probability-space bit LLR mixing identity at interior weight
        l1 = np.clip(cand_inh.llrs, -40.0, 40.0)
        l0 = np.clip(cand_clr.llrs, -40.0, 40.0)
        p_expected = w_rows[:, None, None] * (1.0 / (1.0 + np.exp(-l1))) + (
            1.0 - w_rows[:, None, None]
        ) * (1.0 / (1.0 + np.exp(-l0)))
        p_actual = 1.0 / (1.0 + np.exp(-mixed.mix_llrs))
        np.testing.assert_allclose(p_actual, p_expected, atol=1e-14, rtol=0.0)

        # Boundary behavior: w = 1 -> inherit, w = 0 -> clear
        out_inh = gate_always_inherit(cand_inh, cand_clr)
        out_clr = gate_always_clear(cand_inh, cand_clr)
        np.testing.assert_array_equal(out_inh.mix_probe_pred, cand_inh.probe_pred)
        np.testing.assert_array_equal(out_inh.mix_probe_var, cand_inh.probe_var)
        np.testing.assert_array_equal(out_clr.mix_probe_pred, cand_clr.probe_pred)
        np.testing.assert_array_equal(out_clr.mix_probe_var, cand_clr.probe_var)
        # In probability space, w=1 and w=0 match inherit and clear to ~1e-12 (contract clip)
        p_inh = 1.0 / (1.0 + np.exp(-np.clip(cand_inh.llrs, -40.0, 40.0)))
        p_clr = 1.0 / (1.0 + np.exp(-np.clip(cand_clr.llrs, -40.0, 40.0)))
        np.testing.assert_allclose(1.0 / (1.0 + np.exp(-out_inh.mix_llrs)), p_inh, atol=1e-11, rtol=0.0)
        np.testing.assert_allclose(1.0 / (1.0 + np.exp(-out_clr.mix_llrs)), p_clr, atol=1e-11, rtol=0.0)
        # Direct LLR check on well-conditioned LLR values [-12, 12]
        sample_l1 = np.array([-12.0, -3.5, 0.0, 4.2, 12.0])
        sample_l0 = np.array([10.0, 2.0, -1.5, -6.0, -9.0])
        np.testing.assert_allclose(mix_bit_llrs(1.0, sample_l1, sample_l0), sample_l1, atol=1e-6, rtol=0.0)
        np.testing.assert_allclose(mix_bit_llrs(0.0, sample_l1, sample_l0), sample_l0, atol=1e-6, rtol=0.0)

    def test_cstar_scale_and_phase_invariance(self):
        """Verify <1e-12 C* = R+ x U(1) invariance of jeffreys_complex_gaussian, pre-update
        Bayes factor, split-block predictive Bayes factor, all 5 gates, and codeword scoring
        under (y_a, y_b, n0) -> (alpha * y_a, alpha * y_b, |alpha|^2 * n0).
        """
        rng = np.random.default_rng(20261120)
        mu1 = rng.normal(size=(3, 20, 32)) + 1j * rng.normal(size=(3, 20, 32))
        mu2 = rng.normal(size=(3, 20, 32)) + 1j * rng.normal(size=(3, 20, 32))
        v1 = rng.uniform(0.02, 2.0, size=(3, 20, 32))
        v2 = rng.uniform(0.02, 2.0, size=(3, 20, 32))
        dj_base = jeffreys_complex_gaussian(mu1, v1, mu2, v2)

        alphas = [
            3.7 * np.exp(1j * 1.2345),
            1e-4 * np.exp(-1j * 2.718),
            2.5e4 * np.exp(1j * 0.42),
            np.exp(1j * np.pi / 3.0),
        ]
        for alpha in alphas:
            s2 = float(abs(alpha) ** 2)
            dj_scaled = jeffreys_complex_gaussian(alpha * mu1, s2 * v1, alpha * mu2, s2 * v2)
            np.testing.assert_allclose(dj_scaled, dj_base, atol=1e-12, rtol=1e-12)

        obs, truth = simulate_link(SEED_P9_DEV + 404, "partial_shift", snr_db=9.0, budget="sparse")
        st_inh_0, st_clr_0, inh_0, clr_0 = build_candidates_with_priors(obs)
        bf_0 = inh_0.pre_train_loglik - clr_0.pre_train_loglik
        sp_inh_0, sp_clr_0 = split_block_predictive_scores(obs, st_inh_0, st_clr_0)
        sp_bf_0 = sp_inh_0 - sp_clr_0
        dj_pre_0 = pre_update_jeffreys(inh_0, clr_0)
        dj_post_0 = post_update_jeffreys_rows(inh_0, clr_0, obs.n0)
        g_bf_0 = gate_pre_bayes_factor(inh_0, clr_0)
        g_sp_0 = gate_split_predictive(obs, st_inh_0, st_clr_0, inh_0, clr_0)
        g_jt_0 = gate_jeffreys_student_t(obs, inh_0, clr_0)

        # Codeword scoring candidates on row CORE_ROWS[0]
        row_test = CORE_ROWS[0]
        cw_cands = truth.b_symbols[:4]
        sc_0 = score_codeword_mixture(g_jt_0, obs.y_b[row_test], row_test, obs.n0, cw_cands)
        sc_diff_0 = sc_0 - sc_0[0]

        for alpha in alphas:
            s2 = float(abs(alpha) ** 2)
            obs_s = PublicObservation(
                y_a=alpha * obs.y_a,
                y_b=alpha * obs.y_b,
                training_rows=obs.training_rows,
                probe_row=obs.probe_row,
                n0=s2 * obs.n0,
                budget=obs.budget,
            )
            st_inh_s, st_clr_s, inh_s, clr_s = build_candidates_with_priors(obs_s)
            bf_s = inh_s.pre_train_loglik - clr_s.pre_train_loglik
            self.assertLess(abs(bf_s - bf_0), 1e-12, f"Pre-update Bayes factor drifted for alpha={alpha}")

            sp_inh_s, sp_clr_s = split_block_predictive_scores(obs_s, st_inh_s, st_clr_s)
            sp_bf_s = sp_inh_s - sp_clr_s
            np.testing.assert_allclose(sp_bf_s, sp_bf_0, atol=1e-12, rtol=1e-12)

            np.testing.assert_allclose(pre_update_jeffreys(inh_s, clr_s), dj_pre_0, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(post_update_jeffreys_rows(inh_s, clr_s, obs_s.n0), dj_post_0, atol=1e-12, rtol=1e-12)

            np.testing.assert_allclose(inh_s.llrs, inh_0.llrs, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(clr_s.llrs, clr_0.llrs, atol=1e-12, rtol=1e-12)

            g_bf_s = gate_pre_bayes_factor(inh_s, clr_s)
            g_sp_s = gate_split_predictive(obs_s, st_inh_s, st_clr_s, inh_s, clr_s)
            g_jt_s = gate_jeffreys_student_t(obs_s, inh_s, clr_s)
            for g0, gs in ((g_bf_0, g_bf_s), (g_sp_0, g_sp_s), (g_jt_0, g_jt_s)):
                np.testing.assert_allclose(gs.history_weight, g0.history_weight, atol=1e-12, rtol=1e-12)
                self.assertLess(abs(gs.probe_weight - g0.probe_weight), 1e-12)
                # Probability-space mixture invariance (<1e-12); raw logit near |LLR|>=20 hits 1-ULP of (1-p)
                p_mix_0 = 1.0 / (1.0 + np.exp(-g0.mix_llrs))
                p_mix_s = 1.0 / (1.0 + np.exp(-gs.mix_llrs))
                np.testing.assert_allclose(p_mix_s, p_mix_0, atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(gs.mix_llrs, g0.mix_llrs, atol=1e-6, rtol=1e-7)
                np.testing.assert_allclose(gs.mix_probe_pred, alpha * g0.mix_probe_pred, atol=1e-12 * abs(alpha), rtol=1e-12)
                np.testing.assert_allclose(gs.mix_probe_var, s2 * g0.mix_probe_var, atol=1e-12 * s2, rtol=1e-12)

            sc_s = score_codeword_mixture(g_jt_s, obs_s.y_b[row_test], row_test, obs_s.n0, cw_cands)
            np.testing.assert_allclose(sc_s - sc_s[0], sc_diff_0, atol=1e-12, rtol=1e-12)

    def test_regime_discrimination_sanity(self):
        """On continuous links, gate_pre_bayes_factor and gate_jeffreys_student_t must assign
        high trust w > 0.7; on full_reset and donor_mismatch links, weight must drop toward 0.
        """
        n_trials = 16
        w_bf = {reg: [] for reg in ("continuous", "full_reset", "donor_mismatch")}
        w_jt = {reg: [] for reg in ("continuous", "full_reset", "donor_mismatch")}
        w_sp = {reg: [] for reg in ("continuous", "full_reset", "donor_mismatch")}

        for i in range(n_trials):
            seed = SEED_P9_DEV + 5000 + i
            for reg in ("continuous", "full_reset", "donor_mismatch"):
                obs, _ = simulate_link(seed, reg, snr_db=9.0, budget="sparse")
                st_inh, st_clr, cand_inh, cand_clr = build_candidates_with_priors(obs)
                w_bf[reg].append(float(np.mean(gate_pre_bayes_factor(cand_inh, cand_clr).history_weight)))
                w_jt[reg].append(float(np.mean(gate_jeffreys_student_t(obs, cand_inh, cand_clr).history_weight)))
                w_sp[reg].append(float(np.mean(gate_split_predictive(obs, st_inh, st_clr, cand_inh, cand_clr).history_weight)))

        # Mean weights on continuous must exceed 0.7 (in fact close to 1.0)
        self.assertGreater(float(np.mean(w_bf["continuous"])), 0.7)
        self.assertGreater(float(np.mean(w_jt["continuous"])), 0.7)
        self.assertGreater(float(np.mean(w_sp["continuous"])), 0.7)

        # Mean weights on full_reset and donor_mismatch must drop strongly toward 0
        self.assertLess(float(np.mean(w_bf["full_reset"])), 0.15)
        self.assertLess(float(np.mean(w_bf["donor_mismatch"])), 0.15)
        self.assertLess(float(np.mean(w_jt["full_reset"])), 0.15)
        self.assertLess(float(np.mean(w_jt["donor_mismatch"])), 0.15)
        self.assertLess(float(np.mean(w_sp["full_reset"])), 0.25)
        self.assertLess(float(np.mean(w_sp["donor_mismatch"])), 0.25)

    def test_boundary_phase_continuity_and_oracle_codeword_scoring(self):
        """Verify exact boundary complex phase continuity in partial/OOD shift regimes and
        exact oracle-limit equivalence of score_codeword_mixture.
        """
        ch = ofdm.draw_channel(ofdm.stream(SEED_P9_DEV + 606, 0))
        for shift_row in (0, 6):
            t_shift = float((T + shift_row) * (N + CP) / N)
            t_grid = np.array([[t_shift - 1e-12, t_shift, t_shift + 1e-12]], dtype=np.float64)
            h_cont = frame_b_ray_gains(ch, ch, t_grid, changed_ray=None, delta_freq=0.0, shift_row=None)
            h_shft = frame_b_ray_gains(ch, ch, t_grid, changed_ray=1, delta_freq=0.09, shift_row=shift_row)
            # Exact complex equality at t <= t_shift
            np.testing.assert_allclose(h_shft[:, 0, 0], h_cont[:, 0, 0], atol=1e-14, rtol=0.0)
            np.testing.assert_allclose(h_shft[:, 0, 1], h_cont[:, 0, 1], atol=1e-14, rtol=0.0)
            # Unchanged rays stay identical after t_shift
            np.testing.assert_allclose(h_shft[0, 0, 2], h_cont[0, 0, 2], atol=1e-14, rtol=0.0)
            np.testing.assert_allclose(h_shft[2, 0, 2], h_cont[2, 0, 2], atol=1e-14, rtol=0.0)

        # Verify score_codeword_mixture reduces exactly to -N*log(pi*n0) - ||y - Hx||^2 / n0 when var=0
        obs, truth = simulate_link(SEED_P9_DEV + 707, "continuous", snr_db=15.0, budget="sparse")
        row_idx = CORE_ROWS[3]
        oracle_gate = GateOutput(
            history_weight=np.ones(T, dtype=np.float64),
            probe_weight=1.0,
            mix_llrs=np.zeros((T, N, 4), dtype=np.float64),
            mix_probe_pred=np.zeros(N, dtype=np.complex128),
            mix_probe_var=np.full(N, obs.n0, dtype=np.float64),
            mix_mean_ray=truth.h_b_true,
            mix_var_ray=np.zeros_like(truth.h_b_true.real, dtype=np.float64),
        )
        # Build explicit 32x32 time-varying channel matrix H_t
        h_mat = np.zeros((N, N), dtype=np.complex128)
        for ray_idx, delay in enumerate(DELAYS):
            for n_idx in range(N):
                h_mat[n_idx, (n_idx - int(delay)) % N] += truth.h_b_true[ray_idx, row_idx, n_idx]

        # True codeword vs 5 single-info-bit flipped competing codewords
        true_info = truth.b_info[3]
        gen = coding.generator()
        cands = [truth.b_symbols[3]]
        for bit_i in range(5):
            alt_info = true_info.copy()
            alt_info[bit_i] ^= 1
            alt_code = (alt_info @ gen) % 2
            cands.append(coding.qam(alt_code).reshape(N))
        cands_arr = np.stack(cands, axis=0)

        y_row = obs.y_b[row_idx]
        got_scores = score_codeword_mixture(oracle_gate, y_row, row_idx, obs.n0, cands_arr)
        expected_scores = np.array(
            [
                -N * np.log(np.pi * obs.n0) - np.sum(np.abs(y_row - h_mat @ x) ** 2) / obs.n0
                for x in cands_arr
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(got_scores, expected_scores, atol=1e-12, rtol=1e-12)
        self.assertEqual(int(np.argmax(got_scores)), 0)

    def test_explicit_n0_no_hardcoding(self):
        """Changing obs.n0 must directly alter predictive variances and readout variances."""
        obs_9, _ = simulate_link(SEED_P9_DEV + 808, "continuous", snr_db=9.0, budget="sparse")
        obs_15 = PublicObservation(
            y_a=obs_9.y_a,
            y_b=obs_9.y_b,
            training_rows=obs_9.training_rows,
            probe_row=obs_9.probe_row,
            n0=n0_from_snr(15.0),
            budget=obs_9.budget,
        )
        inh_9, clr_9 = build_candidates(obs_9)
        inh_15, clr_15 = build_candidates(obs_15)
        self.assertGreater(float(np.mean(inh_9.probe_var)), float(np.mean(inh_15.probe_var)))
        self.assertGreater(float(np.mean(clr_9.probe_var)), float(np.mean(clr_15.probe_var)))
        self.assertGreater(float(np.mean(inh_9.pre_train_var)), float(np.mean(inh_15.pre_train_var)))
        self.assertFalse(np.allclose(inh_9.llrs, inh_15.llrs))
        self.assertFalse(np.allclose(clr_9.llrs, clr_15.llrs))


if __name__ == "__main__":
    unittest.main()
