"""Fast deterministic invariant & firewall unit tests for Phase 12 Stage 1 diagnostics."""
from __future__ import annotations

import math
import unittest

import numpy as np

from p9.channel import simulate_link
from p9.contract import CORE_ROWS, PROBE_ROW, PublicObservation
from p11.offgrid_channel import simulate_offgrid_link
from p12.contract import (
    OFDM_PILOT_COMB,
    OFDM_PILOT_PARETO,
    OFFGRID_REGIME_TRUE_RESET,
    P12_FOLD2_AUX,
    P12_FOLD2_TARGETS,
    P12_FOLD3_AUX,
    P12_FOLD3_TARGETS,
    SEED_P12_DIAG,
    SEED_P12_E2E,
    SENTINEL_ARMS,
    SENTINEL_AUX,
    SENTINEL_TARGETS,
    assert_p12_seed_disjoint,
)
from p12.offgrid_physics import (
    audit_parallel_transport_connection,
    dominant_sidelobe_spacing,
    make_roving_bilateral_2fold,
    offgrid_cov32_loglik,
    simulate_offgrid_link_p12,
)
from p12.primary_folds import run_k_fold_wavefront, run_sentinel_arms_on_link


class TestP12Stage1Invariants(unittest.TestCase):
    def test_seed_disjointness(self) -> None:
        seeds = [SEED_P12_DIAG + i for i in range(64)]
        out = assert_p12_seed_disjoint(seeds, expected_namespace="p12_diag")
        self.assertEqual(len(out), 64)
        # Reject historical & P11 seeds
        with self.assertRaises(ValueError):
            assert_p12_seed_disjoint([1_130_000_000_001])
        with self.assertRaises(ValueError):
            assert_p12_seed_disjoint([930_000_000_001])

    def test_symmetric_fold_geometry_and_zero_target_leakage(self) -> None:
        seed = SEED_P12_E2E + 1
        obs, _ = simulate_link(seed, snr_db=9.0, regime="full_reset", budget="sparse")

        # Tamper Fold 0 targets wildly; Fold 0 channel predictions on its targets MUST be bit-identical (0.0 diff)!
        y_b_tampered = obs.y_b.copy()
        y_b_tampered[list(P12_FOLD2_TARGETS[0])] += 1e6 + 1e6j
        obs_tamp = PublicObservation(
            y_a=obs.y_a,
            y_b=y_b_tampered,
            training_rows=obs.training_rows,
            probe_row=obs.probe_row,
            n0=obs.n0,
            budget=obs.budget,
        )

        res_clean = run_k_fold_wavefront(obs, P12_FOLD2_TARGETS, P12_FOLD2_AUX)
        res_tamp = run_k_fold_wavefront(obs_tamp, P12_FOLD2_TARGETS, P12_FOLD2_AUX)

        for r in P12_FOLD2_TARGETS[0]:
            diff_m = float(np.max(np.abs(res_clean["owner_mean"][r] - res_tamp["owner_mean"][r])))
            diff_v = float(np.max(np.abs(res_clean["owner_var"][r] - res_tamp["owner_var"][r])))
            self.assertEqual(diff_m, 0.0, f"Target leakage detected on K=2 Fold 0 row {r}")
            self.assertEqual(diff_v, 0.0, f"Target var leakage detected on K=2 Fold 0 row {r}")

        # Verify K=2 & K=3 fold wave-1 bilateral helpers exist on BOTH sides of anchor 10
        for name, tf, af in (("K2", P12_FOLD2_TARGETS, P12_FOLD2_AUX), ("K3", P12_FOLD3_TARGETS, P12_FOLD3_AUX)):
            all_d = set(range(20)) - {10, PROBE_ROW}
            for f_i, (t_r, a_r) in enumerate(zip(tf, af)):
                helpers = sorted(all_d - set(t_r) - set(a_r))
                self.assertTrue(any(h < 10 for h in helpers) and any(h > 10 for h in helpers), f"{name} fold {f_i} not bilateral")

    def test_sentinel_zero_target_leakage_and_cstar_invariance(self) -> None:
        seed = SEED_P12_E2E + 2
        obs, truth = simulate_link(seed, snr_db=9.0, regime="full_reset", budget="sparse")

        # Tampering sentinel targets MUST NOT change any helper acceptance or channel prediction metrics
        # (we verify via blind helper_diag exact equality!)
        y_b_tamp = obs.y_b.copy()
        y_b_tamp[list(SENTINEL_TARGETS)] += 5e5 - 3e5j
        obs_tamp = PublicObservation(
            y_a=obs.y_a, y_b=y_b_tamp, training_rows=obs.training_rows, probe_row=obs.probe_row, n0=obs.n0, budget=obs.budget
        )
        s_clean = run_sentinel_arms_on_link(
            obs, truth.b_symbols, truth.b_info, truth.h_b_true,
            sentinel_targets=SENTINEL_TARGETS, sentinel_aux=SENTINEL_AUX, sentinel_arms=SENTINEL_ARMS,
        )
        s_tamp = run_sentinel_arms_on_link(
            obs_tamp, truth.b_symbols, truth.b_info, truth.h_b_true,
            sentinel_targets=SENTINEL_TARGETS, sentinel_aux=SENTINEL_AUX, sentinel_arms=SENTINEL_ARMS,
        )
        for arm in SENTINEL_ARMS:
            self.assertEqual(s_clean["arms"][arm]["helper_diag"], s_tamp["arms"][arm]["helper_diag"])
            # Even channel phase & frobenius errors on sentinel targets depend ONLY on ResetBase+helpers, not y_b[targets]!
            for mode in ("true_words_n0", "blind_decoded"):
                for r in SENTINEL_TARGETS:
                    self.assertEqual(
                        s_clean["arms"][arm][mode]["per_row"][str(r)]["ray0_phase_err_deg"],
                        s_tamp["arms"][arm][mode]["per_row"][str(r)]["ray0_phase_err_deg"],
                    )

    def test_offgrid_comb_bit_parity_and_pareto_energy_parity(self) -> None:
        seed = SEED_P12_E2E + 3
        obs_p11, tr_p11 = simulate_offgrid_link(seed, "offgrid_continuous", snr_db=9.0)
        obs_p12, tr_p12, _ = simulate_offgrid_link_p12(seed, "offgrid_continuous", snr_db=9.0, pilot_rows_a=OFDM_PILOT_COMB)
        self.assertTrue(np.array_equal(obs_p11.y_a, obs_p12.y_a))
        self.assertTrue(np.array_equal(obs_p11.y_b[:19], obs_p12.y_b[:19]))
        self.assertTrue(np.array_equal(tr_p11.h_b_true_mat, tr_p12.h_b_true_mat))

        # Pareto (0,4,6,19) MUST have identical Frame-B channel & identical total pilot energy in Frame A
        obs_par, tr_par, _ = simulate_offgrid_link_p12(seed, "offgrid_continuous", snr_db=9.0, pilot_rows_a=OFDM_PILOT_PARETO)
        self.assertTrue(np.array_equal(tr_p12.h_b_true_mat, tr_par.h_b_true_mat))
        e_comb = float(np.sum(np.abs(obs_p12.pilot_tx_a) ** 2))
        e_par = float(np.sum(np.abs(obs_par.pilot_tx_a) ** 2))
        self.assertAlmostEqual(e_comb, e_par, places=12)

        # True full reset MUST draw different delays_b from delays_a
        _, _, ex_keep = simulate_offgrid_link_p12(seed, "offgrid_full_reset", snr_db=9.0)
        _, _, ex_new = simulate_offgrid_link_p12(seed, OFFGRID_REGIME_TRUE_RESET, snr_db=9.0)
        self.assertTrue(np.array_equal(ex_keep["delays_a"], ex_keep["delays_b"]))
        self.assertFalse(np.array_equal(ex_new["delays_a"], ex_new["delays_b"]))

    def test_roving_bilateral_2fold_and_symmetric_connection(self) -> None:
        for r_anc in (3, 6, 10, 13, 16):
            (t0, t1), (a0, a1) = make_roving_bilateral_2fold(r_anc, probe_row=PROBE_ROW)
            self.assertEqual(len(t0), 9)
            self.assertEqual(len(t1), 9)
            self.assertEqual(len(a0), 5)
            self.assertEqual(len(a1), 5)
            self.assertTrue(set(t0).isdisjoint(t1))
            self.assertTrue(set(a0).isdisjoint(a1))
            self.assertTrue(set(t0).isdisjoint(a0))
            self.assertTrue(set(t1).isdisjoint(a1))
            all_d = set(range(20)) - {r_anc, PROBE_ROW}
            for tf, af in ((t0, a0), (t1, a1)):
                h = sorted(all_d - set(tf) - set(af))
                self.assertEqual(len(h), 4)
                self.assertTrue(any(x < r_anc for x in h) and any(x > r_anc for x in h))

        # Symmetric parallel-transport connection exact reverse-edge unitarity (< 1e-15)
        res_conn = audit_parallel_transport_connection(
            np.array([0.4, 2.1, 4.7], dtype=np.float64),
            np.array([0.055, -0.072, 0.041], dtype=np.float64),
        )
        self.assertGreater(res_conn["p11_query_only_max_rev_unitarity_err"], 0.5)
        self.assertLess(res_conn["p12_symmetric_max_rev_unitarity_err"], 1e-14)
        self.assertLess(res_conn["p12_symmetric_max_conj_sym_err"], 1e-14)

        # Sidelobe spacing sanity for Pareto (0,4,6,19) -> ~0.2122
        lobe_p = dominant_sidelobe_spacing(OFDM_PILOT_PARETO)
        self.assertAlmostEqual(lobe_p, 0.2122, delta=0.002)

    def test_stage1_smoke_1link_json_serializable(self) -> None:
        import json
        from p12.stage1_primary_diag import _eval_one_primary_link, aggregate_primary_diag
        from p12.stage1_secondary_diag import _eval_one_secondary_link, aggregate_secondary_diag

        p_rec = _eval_one_primary_link((0, SEED_P12_E2E + 10, 9.0, True))
        p_sum = aggregate_primary_diag([p_rec])
        json.dumps(p_rec, allow_nan=False)
        json.dumps(p_sum, allow_nan=False)

        s_rec = _eval_one_secondary_link(("offgrid_continuous", 0, SEED_P12_E2E + 11, 9.0, True))
        s_sum = aggregate_secondary_diag([s_rec])
        json.dumps(s_rec, allow_nan=False)
        json.dumps(s_sum, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
