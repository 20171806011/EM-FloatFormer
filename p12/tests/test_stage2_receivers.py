"""Strict mathematical invariant, firewall, C*-equivariance & Numba parity tests for P12 Stage 2 receivers."""
from __future__ import annotations

import math
import unittest

import numpy as np
from scipy.special import logsumexp

from p8.coding import qam
from p9.contract import CORE_ROWS, N, PROBE_ROW, PublicObservation
from p9.decoder import pack_codewords
from p11.contract import OffGridObservation
from p11.wavefront_pipeline import _H_U64, _NIBBLE_IM, _NIBBLE_RE
from p12.contract import (
    OFDM_PILOT_COMB,
    OFFGRID_REGIME_TRUE_RESET,
    P12_FOLD2_AUX,
    P12_FOLD2_TARGETS,
    P12_FOLD3_AUX,
    P12_FOLD3_TARGETS,
    SEED_P12_E2E,
)
from p12.offgrid_physics import simulate_offgrid_link_p12
from p12.offgrid_receiver import (
    _score_offgrid_multimode_u64,
    discover_b_only_delays,
    run_p12_offgrid_physics,
    warmup_p12_offgrid_numba,
)
from p12.primary_receiver import (
    PRIMARY_OOD_TWO_RAY_SHIFT,
    PRIMARY_OOD_UNSEEN_ONSET,
    run_p12_primary_kfold,
    simulate_p12_primary_link,
    validate_fold_partition,
)


class TestP12Stage2Receivers(unittest.TestCase):
    def test_validate_fold_partition_rejects_malformed_inputs(self) -> None:
        validate_fold_partition(P12_FOLD2_TARGETS, P12_FOLD2_AUX)
        validate_fold_partition(P12_FOLD3_TARGETS, P12_FOLD3_AUX)
        # Mismatched length must raise (no silent zip truncation!)
        with self.assertRaises(ValueError):
            validate_fold_partition(P12_FOLD3_TARGETS, P12_FOLD2_AUX)
        # Duplicate target ownership must raise
        bad_t = (P12_FOLD2_TARGETS[0], P12_FOLD2_TARGETS[1] + (1,))
        with self.assertRaises(ValueError):
            validate_fold_partition(bad_t, P12_FOLD2_AUX)
        # Target/AUX overlap must raise
        with self.assertRaises(ValueError):
            validate_fold_partition(P12_FOLD2_TARGETS, ((1, 6, 7, 14, 15), P12_FOLD2_AUX[1]))

    def test_primary_k3_zero_target_and_heldout_extra_leakage(self) -> None:
        seed = SEED_P12_E2E + 201
        obs, _ = simulate_p12_primary_link(seed, snr_db=9.0, regime="full_reset", budget="sparse")
        res_clean, _ = run_p12_primary_kfold(obs, k_folds=3)

        # 1) Tamper Fold 0 targets wildly -> Fold 0 channel predictions MUST be bit-identical (0.0 diff)!
        y_t = obs.y_b.copy()
        y_t[list(P12_FOLD3_TARGETS[0])] += 1e6 - 2e6j
        obs_t = PublicObservation(obs.y_a, y_t, obs.training_rows, obs.probe_row, obs.n0, obs.budget)
        res_t, _ = run_p12_primary_kfold(obs_t, k_folds=3)
        for r in P12_FOLD3_TARGETS[0]:
            self.assertEqual(float(np.max(np.abs(res_clean.gate_out.mix_mean_ray[:, r, :] - res_t.gate_out.mix_mean_ray[:, r, :]))), 0.0)
            self.assertEqual(float(np.max(np.abs(res_clean.gate_out.mix_var_ray[:, r, :] - res_t.gate_out.mix_var_ray[:, r, :]))), 0.0)

        # 2) Tamper reclaimed extra rows (0, 9) -> their held-out channel predictions MUST be bit-identical (0.0 diff)!
        for r_ex in (0, 9):
            y_ex = obs.y_b.copy()
            y_ex[r_ex] += 7e5 + 3e5j
            obs_ex = PublicObservation(obs.y_a, y_ex, obs.training_rows, obs.probe_row, obs.n0, obs.budget)
            res_ex, _ = run_p12_primary_kfold(obs_ex, k_folds=3)
            self.assertEqual(
                float(np.max(np.abs(res_clean.gate_out.mix_mean_ray[:, r_ex, :] - res_ex.gate_out.mix_mean_ray[:, r_ex, :]))),
                0.0,
                f"Held-out extra row {r_ex} leaked into its own channel estimate!",
            )

        # 3) Verify out-of-support OOD generators work cleanly
        for reg in (PRIMARY_OOD_UNSEEN_ONSET, PRIMARY_OOD_TWO_RAY_SHIFT):
            obs_ood, tr_ood = simulate_p12_primary_link(seed + 1, snr_db=9.0, regime=reg, budget="sparse")
            self.assertTrue(np.isnan(obs_ood.y_b[PROBE_ROW]).all())
            self.assertEqual(tr_ood.b_info.shape, (18, 64))

    def test_numba_offgrid_multimode_exact_numpy_parity(self) -> None:
        warmup_p12_offgrid_numba()
        rng = np.random.default_rng(12345)
        from p8.coding import generator
        g_mat = generator()
        cws = [(rng.integers(0, 2, size=64, dtype=np.uint8) @ g_mat) % 2 for _ in range(8)]
        cand_u64 = np.stack([pack_codewords(c) for c in cws], axis=0)
        y = (rng.normal(size=N) + 1j * rng.normal(size=N)).astype(np.complex128)
        h_s = (rng.normal(size=(2, N, N)) + 1j * rng.normal(size=(2, N, N))).astype(np.complex128) * 0.2
        nv = rng.uniform(0.5, 2.0, size=(2, N)).astype(np.float64)
        log_p = np.log(np.array([0.65, 0.35], dtype=np.float64))

        b0, b1, best_ll, _, _, sec_ll, win_m, _, n_val = _score_offgrid_multimode_u64(
            cand_u64, _H_U64, y.real.copy(), y.imag.copy(), log_p,
            h_s.real.copy(), h_s.imag.copy(), nv.copy(), _NIBBLE_RE, _NIBBLE_IM,
        )
        self.assertEqual(n_val, 8)
        # Pure NumPy reference
        ref_lls = []
        for cw in cws:
            x = qam(cw).astype(np.complex128)
            m_lls = []
            for b in range(2):
                diff = y - h_s[b] @ x
                m_lls.append(float(log_p[b] - np.sum(np.log(math.pi * nv[b]) + (diff.real ** 2 + diff.imag ** 2) / nv[b])))
            ref_lls.append(float(logsumexp(m_lls)))
        order = np.argsort(-np.array(ref_lls))
        self.assertAlmostEqual(best_ll, ref_lls[order[0]], places=11)
        self.assertAlmostEqual(sec_ll, ref_lls[order[1]], places=11)

    def test_offgrid_physics_zero_leakage_cstar_and_delay_reset(self) -> None:
        seed = SEED_P12_E2E + 205
        obs, truth, extra = simulate_offgrid_link_p12(seed, "offgrid_continuous", snr_db=9.0, pilot_rows_a=OFDM_PILOT_COMB)
        r_anc = int(obs.training_rows[0])

        # 1) Zero probe-row & target-row leakage (tampering PROBE_ROW -> 0.0 diff; tampering Fold-0 targets -> 0.0 diff on Fold-0 channel!)
        from p12.offgrid_physics import make_roving_bilateral_2fold
        (t0, _), _ = make_roving_bilateral_2fold(r_anc, PROBE_ROW)

        res0 = run_p12_offgrid_physics(obs)
        y_t = obs.y_b.copy()
        y_t[list(t0)] += 1e6 - 1e6j
        y_t[PROBE_ROW] = 9e9 + 9e9j
        obs_t = OffGridObservation(obs.y_a, y_t, obs.training_rows, obs.probe_row, obs.n0, obs.pilot_tx_b, obs.pilot_tx_a)
        res_t = run_p12_offgrid_physics(obs_t, use_burst_guard=False)
        res0_nb = run_p12_offgrid_physics(obs, use_burst_guard=False)
        for r in t0:
            self.assertEqual(float(np.max(np.abs(res0_nb.h_mix[r] - res_t.h_mix[r]))), 0.0, f"Target leakage on off-grid row {r}")
            self.assertEqual(float(np.max(np.abs(res0_nb.noise_mix[r] - res_t.noise_mix[r]))), 0.0)

        # 2) Exact C* equivariance (scale y_a, y_b by alpha = 3.7*exp(1.1j), n0 by |alpha|^2)
        alpha = 3.7 * complex(math.cos(1.1), math.sin(1.1))
        obs_s = OffGridObservation(
            obs.y_a * alpha,
            obs.y_b * alpha,
            obs.training_rows,
            obs.probe_row,
            obs.n0 * (abs(alpha) ** 2),
            obs.pilot_tx_b,
            obs.pilot_tx_a,
        )
        res_s = run_p12_offgrid_physics(obs_s)
        self.assertEqual(res0.delay_reset_triggered, res_s.delay_reset_triggered)
        self.assertLess(float(np.max(np.abs(res_s.h_mix - res0.h_mix * alpha))) / abs(alpha), 1e-10)
        self.assertLess(float(np.max(np.abs(res_s.gate_out.mix_llrs - res0.gate_out.mix_llrs))), 1e-9)

        # 3) B-only delay rediscovery recovers new delays cleanly on true full reset
        obs_nr, tr_nr, _ = simulate_offgrid_link_p12(seed + 7, OFFGRID_REGIME_TRUE_RESET, snr_db=9.0)
        y_anc_u = obs_nr.y_b[obs_nr.training_rows[0]] / math.sqrt(obs_nr.n0)
        tau_b, _, _ = discover_b_only_delays(y_anc_u, obs_nr.pilot_tx_b[0], k_hat=5)
        match_err = float(np.mean([np.min(np.abs(tau_b - td)) for td in tr_nr.delays]))
        self.assertLess(match_err, 0.35, f"B-only delay rediscovery failed to match new delays: {match_err}")


if __name__ == "__main__":
    unittest.main()
