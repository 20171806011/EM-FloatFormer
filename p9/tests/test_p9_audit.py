"""Comprehensive unit tests for Phase 9 independent physics oracle, firewall, and blind auditor."""
from __future__ import annotations

from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

from p8 import link as p8_ofdm
from p8c import link as p8c_link
from p9 import audit as p9_audit
from p9.contract import (
    PROBE_ROW,
    PublicObservation,
    SC_LAYOUTS,
    SEED_P9_DEV,
    SEED_P9_FORMAL_ID,
    SEED_P9_FORMAL_OOD,
    SEED_P9_TRAIN,
    T,
    N,
    n0_from_snr,
)
from p9.independent_ref import (
    PiecewiseDopplerShift,
    audit_receiver_anti_leakage,
    blockwise_propagate_piecewise,
    check_no_hardcoded_n0,
    check_seed_disjointness,
    exhaustive_ml_verify_mini_code,
    mini_code_16_8,
    row_boundary_time,
    serial_propagate_ab,
    verify_phase_continuity,
    verify_public_observation,
)


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class SerialPhysicsReferenceTests(unittest.TestCase):
    def test_serial_propagate_ab_matches_p8_and_p8c_across_boundary(self):
        for seed in (3117, 6017, 8035):
            pair = p8c_link.simulate(seed)
            channel = p8_ofdm.draw_channel(p8_ofdm.stream(seed, 0))
            a_grid = p8_ofdm.tx_grid(p8_ofdm.traffic(p8_ofdm.stream(seed, 1)), p8_ofdm.PILOT_ROWS)
            y_a_p8 = p8_ofdm.propagate(a_grid, channel, start_row=0)

            for layout, rows in p8c_link.LAYOUTS.items():
                b_grid = p8c_link.tx_grid(pair.b_words, rows)
                y_b_p8c = p8c_link.propagate(b_grid, channel, start_row=T)

                res = serial_propagate_ab(a_grid, b_grid, channel, return_details=True)
                self.assertEqual(res.tx_serial.shape, (40 * 48,))
                self.assertEqual(res.rx_serial.shape, (40 * 48,))
                self.assertEqual(res.h_b_useful.shape, (3, T, N))

                err_a = float(np.max(np.abs(res.y_a - y_a_p8)))
                err_b = float(np.max(np.abs(res.y_b - y_b_p8c)))
                err_clean_b = float(np.max(np.abs(res.y_b - pair.clean_b[layout])))
                self.assertLess(err_a, 1e-13, f"Frame A mismatch {err_a:.3e} on seed={seed}")
                self.assertLess(err_b, 1e-13, f"Frame B mismatch {err_b:.3e} on seed={seed}/{layout}")
                self.assertLess(err_clean_b, 1e-13, f"clean_b mismatch {err_clean_b:.3e} on seed={seed}/{layout}")

    def test_phase_continuity_at_shift_row_0_and_6_and_blockwise_match(self):
        seed = 4219
        pair = p8c_link.simulate(seed)
        channel = p8_ofdm.draw_channel(p8_ofdm.stream(seed, 0))
        a_grid = p8_ofdm.tx_grid(p8_ofdm.traffic(p8_ofdm.stream(seed, 1)), p8_ofdm.PILOT_ROWS)
        b_grid = p8c_link.tx_grid(pair.b_words, p8c_link.LAYOUTS["sparse"])
        y_a_cont, y_b_cont = serial_propagate_ab(a_grid, b_grid, channel)

        for shift_row in (0, 6):
            for ray_idx, df, dc in ((0, 0.04, 0.0), (1, -0.065, 0.0), (2, 0.09, 0.0015)):
                t_shift = row_boundary_time(shift_row, frame="B")
                if shift_row == 0:
                    self.assertAlmostEqual(t_shift, 30.0, places=15)
                if shift_row == 6:
                    self.assertAlmostEqual(t_shift, 39.0, places=15)

                shift = PiecewiseDopplerShift(ray_index=ray_idx, delta_freq=df, t_shift=t_shift, delta_chirp=dc)
                cont = verify_phase_continuity(channel, shift)
                self.assertTrue(cont["continuous_ok"], f"Phase continuity failed: {cont}")
                self.assertLess(cont["phase_jump_rad"], 1e-13)
                self.assertLess(cont["gain_jump"], 1e-13)
                self.assertLess(cont["reanchored_max_gain_err"], 1e-13)
                self.assertLess(cont["freq_jump_err"], 1e-6)

                res = serial_propagate_ab(a_grid, b_grid, channel, shift=shift, return_details=True)
                blk_a = blockwise_propagate_piecewise(a_grid, channel, start_row=0, is_ofdm=True, shift=shift)
                blk_b = blockwise_propagate_piecewise(b_grid, channel, start_row=T, is_ofdm=False, shift=shift)

                self.assertLess(float(np.max(np.abs(res.y_a - blk_a))), 1e-13)
                self.assertLess(float(np.max(np.abs(res.y_b - blk_b))), 1e-13)
                # Frame A (all samples t < 30.0) MUST be completely untouched by a Frame-B shift
                self.assertLess(float(np.max(np.abs(res.y_a - y_a_cont))), 1e-13)
                # Frame B rows strictly before shift_row MUST match continuous trajectory identically
                if shift_row > 0:
                    self.assertLess(float(np.max(np.abs(res.y_b[:shift_row] - y_b_cont[:shift_row]))), 1e-13)
                # Frame B rows at/after shift_row MUST genuinely diverge from continuous trajectory
                self.assertGreater(float(np.max(np.abs(res.y_b[shift_row:] - y_b_cont[shift_row:]))), 1e-2)

    def test_mid_block_arbitrary_sample_shift_continuity_and_full_reset(self):
        seed = 5501
        pair = p8c_link.simulate(seed)
        ch_a = p8_ofdm.draw_channel(p8_ofdm.stream(seed, 0))
        ch_b = p8_ofdm.draw_channel(p8_ofdm.stream(seed, 5))
        a_grid = p8_ofdm.tx_grid(p8_ofdm.traffic(p8_ofdm.stream(seed, 1)), p8_ofdm.PILOT_ROWS)
        b_grid = p8c_link.tx_grid(pair.b_words, p8c_link.LAYOUTS["sparse"])

        # Arbitrary non-row sample boundary inside Frame B row 3
        t_arb = (23 * 48 + 25) / 32.0
        shift = PiecewiseDopplerShift(ray_index=1, delta_freq=0.05, t_shift=t_arb)
        cont = verify_phase_continuity(ch_a, shift)
        self.assertTrue(cont["continuous_ok"])
        self.assertLess(cont["phase_jump_rad"], 1e-13)
        y_a_s, y_b_s = serial_propagate_ab(a_grid, b_grid, ch_a, shift=shift)
        blk_b = blockwise_propagate_piecewise(b_grid, ch_a, start_row=T, is_ofdm=False, shift=shift)
        self.assertLess(float(np.max(np.abs(y_b_s - blk_b))), 1e-13)
        self.assertLess(float(np.max(np.abs(y_a_s - p8_ofdm.propagate(a_grid, ch_a)))), 1e-13)

        # Full reset mode (Mechanism 3): Frame A uses ch_a, Frame B uses independent ch_b
        y_a_r, y_b_r = serial_propagate_ab(a_grid, b_grid, ch_a, channel_b=ch_b, channel_b_time_origin="global")
        self.assertLess(float(np.max(np.abs(y_a_r - p8_ofdm.propagate(a_grid, ch_a)))), 1e-13)
        self.assertLess(float(np.max(np.abs(y_b_r - p8c_link.propagate(b_grid, ch_b, start_row=T)))), 1e-13)


class ExhaustiveMiniCodebookMLTests(unittest.TestCase):
    def test_mini_code_16_8_structure_and_exact_ml_certificate(self):
        code = mini_code_16_8()
        self.assertEqual(code["d_min"], 5)
        self.assertEqual(code["code_all"].shape, (256, 16))
        self.assertEqual(code["x_all"].shape, (256, 4))
        self.assertFalse(np.any((code["G"] @ code["H_parity"].T) % 2))
        self.assertAlmostEqual(float(np.mean(np.abs(code["x_all"]) ** 2)), 1.0, places=12)

        for snr_db, seed in ((6.0, 91001608), (9.0, 91001609)):
            res = exhaustive_ml_verify_mini_code(n_trials=200, snr_db=snr_db, seed=seed)
            self.assertEqual(res["n_trials"], 200)
            self.assertLess(res["max_conv_err"], 1e-13)
            self.assertEqual(res["false_positives"], 0)
            self.assertEqual(res["neighbors_false_positives"], 0)
            self.assertEqual(res["full_false_positives"], 0)
            self.assertTrue(res["soundness_verified"])
            self.assertTrue(res["inequality_holds"], f"Inequality failed at {snr_db} dB: {res}")
            self.assertLessEqual(res["cert_rate"], res["ml_bler"])
            self.assertLessEqual(res["ml_bler"], res["decoder_bler"])
            # Ensure nontrivial errors occurred so soundness was genuinely exercised
            self.assertGreater(res["ml_errors"], 0)
            self.assertGreater(res["subset_cert_hits"], 0)


class IndependentStatisticalAuditTests(unittest.TestCase):
    def test_reproduces_p8c_summary_exact_stats_within_1e12(self):
        raw_path = REPO_ROOT / "p8c" / "results" / "links.jsonl"
        summary_path = REPO_ROOT / "p8c" / "results" / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        report = p9_audit.audit_p8c(raw_path, summary_path)
        self.assertEqual(report["clusters"], 10)
        self.assertEqual(report["df"], 9)
        self.assertEqual(report["exact_sign_patterns"], 1024)
        self.assertLess(report["max_aggregate_diff"], 1e-12)
        self.assertLess(report["max_stat_diff"], 1e-12)
        # Note: p8c/results/summary.json used p7/statistics.py's hardcoded 11-digit
        # t_crit=2.2621571627409915 vs exact float64 2.262157162798202 (5.72e-11 diff),
        # yielding max CI95 delta 3.72e-12 (well inside p8c/audit.py's 1e-10 tolerance).
        self.assertLess(report["max_ci95_diff"], 1e-11)

        # Direct field-by-field assertion against p8c/results/summary.json
        by_key = {(t["control"], t["metric"]): t for t in report["holm_tests"]}
        for exp_t in summary["tests"]:
            got = by_key[(exp_t["control"], exp_t["metric"])]
            self.assertEqual(Fraction(got["raw_exact"]), Fraction(exp_t["raw_exact"]))
            self.assertEqual(Fraction(got["holm_exact"]), Fraction(exp_t["holm_exact"]))
            self.assertAlmostEqual(got["mean_diff"], exp_t["mean_diff"], delta=1e-12)
            np.testing.assert_allclose(got["differences"], exp_t["differences"], atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(got["ci95"], exp_t["ci95"], atol=1e-11, rtol=1e-10)
            self.assertTrue(got["supported"])
            self.assertEqual(got["verdict"], "supported")

        # Verify exact closed-form Student-t quantile precision (< 1e-15 CDF error across df=1..15)
        for df in (1, 2, 9, 10, 14, 15):
            for q in (0.95, 0.975):
                x = p9_audit.exact_student_t_ppf(q, df)
                self.assertLess(abs(p9_audit._exact_student_t_cdf(x, df) - q), 1e-14)

        exp_ni = summary["noninferiority"]
        got_ni = report["noninferiority"]
        self.assertEqual(got_ni["df"], 9)
        self.assertAlmostEqual(got_ni["mean_diff"], exp_ni["mean_diff"], delta=1e-12)
        self.assertAlmostEqual(got_ni["upper_95"], exp_ni["upper_95"], delta=1e-12)
        self.assertEqual(got_ni["passes"], exp_ni["passes"])
        self.assertFalse(got_ni["passes"])
        self.assertEqual(report["delivery_summary"]["net_extra_blocks_gain"], 175)

    def test_exact_sign_flip_g16_speed_exactness_and_bruteforce_match(self):
        # 1. Cross-check chunked symmetry algorithm against explicit brute-force 2^G enumeration for G=8
        rng = np.random.default_rng(12345)
        for diffs in (
            rng.normal(0.05, 0.12, size=8),
            np.array([0.1, -0.1, 0.2, 0.0, 0.05, 0.05, -0.02, 0.03]),
            np.zeros(8),
        ):
            got = p9_audit.exact_sign_flip_test(diffs, expected_sign=1, chunk_size=17)
            signs = np.array(list(itertools.product((-1.0, 1.0), repeat=8)), dtype=np.float64)
            thr = abs(float(diffs.sum())) - 1e-12 * max(1.0, float(np.sum(np.abs(diffs))))
            brute = Fraction(int(np.sum(np.abs(signs @ diffs) >= thr)), len(signs))
            self.assertEqual(Fraction(got["raw_exact"]), brute)

        # 2. G=16 (65,536 patterns): exact 1/32768 for all-positive two-sided + fast execution
        pos16 = np.linspace(0.02, 0.17, 16)
        t0 = time.perf_counter()
        res16 = p9_audit.exact_sign_flip_test(pos16, expected_sign=1)
        elapsed = time.perf_counter() - t0
        self.assertEqual(res16["exact_patterns"], 65536)
        self.assertEqual(res16["extreme_patterns"], 2)
        self.assertEqual(res16["raw_exact"], "1/32768")
        self.assertEqual(res16["raw_p"], 1.0 / 32768.0)
        self.assertEqual(res16["df"], 15)
        self.assertTrue(res16["direction_aligned"])
        self.assertLess(elapsed, 0.5, f"G=16 exact sign-flip too slow: {elapsed:.3f}s")

        # 3. Step-down Holm-Bonferroni + directional alignment check
        c_pos = p9_audit.exact_sign_flip_test(pos16, expected_sign=1)
        c_neg = p9_audit.exact_sign_flip_test(-pos16, expected_sign=1)  # significant but WRONG direction!
        c_null = p9_audit.exact_sign_flip_test(np.array([1, -1] * 8, dtype=np.float64) * 0.01, expected_sign=1)
        c_mid = p9_audit.exact_sign_flip_test(
            np.array([0.05] * 13 + [-0.02, -0.01, -0.01], dtype=np.float64), expected_sign=1
        )
        holm = p9_audit.holm_bonferroni_adjust([c_pos, c_neg, c_null, c_mid], alpha=Fraction(1, 20))
        self.assertEqual(holm[0]["verdict"], "supported")
        self.assertTrue(holm[0]["supported"])
        self.assertEqual(holm[1]["verdict"], "refuted")
        self.assertFalse(holm[1]["supported"])
        self.assertTrue(holm[1]["significant_two_sided"])
        self.assertFalse(holm[1]["direction_aligned"])
        self.assertEqual(holm[2]["verdict"], "insufficient")
        self.assertFalse(holm[2]["supported"])
        self.assertEqual(Fraction(holm[0]["holm_exact"]), Fraction(4, 32768))

    def test_synthetic_p9_audit_catches_planted_corruptions(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            raw_path, summary_path, replay_path = p9_audit._build_synthetic_p9_dataset(
                tmp_dir, g=16, links_per_cluster=12
            )
            ok = p9_audit.audit_p9(raw_path, summary_path, replay_path)
            self.assertEqual(ok["clusters"], 16)
            self.assertEqual(ok["df"], 15)
            self.assertTrue(ok["replay_all_fields_match"])
            self.assertTrue(ok["replay_exact_bytes_match"])

            # Corruption 1: tamper paired target energy on one arm of one link
            raw_lines = raw_path.read_text(encoding="utf-8").splitlines()
            rec0 = json.loads(raw_lines[0])
            rec0["budgets"]["sparse"]["attn_gate"]["probe_energy"] += 1e-4
            bad_energy_raw = tmp_dir / "bad_energy.jsonl"
            bad_energy_raw.write_text(
                "\n".join([json.dumps(rec0)] + raw_lines[1:]) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Held-out target energies differ"):
                p9_audit.audit_p9(bad_energy_raw, summary_path)

            # Corruption 2: plant a train/test seed overlap in summary seed_pools
            bad_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            bad_summary["seed_pools"]["train"].append(bad_summary["seed_pools"]["formal_id"][0])
            bad_seed_summary = tmp_dir / "bad_seeds.json"
            bad_seed_summary.write_text(json.dumps(bad_summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Seed disjointness violation"):
                p9_audit.audit_p9(raw_path, bad_seed_summary)

            # Corruption 3: tamper one field in replay JSONL
            rep_lines = replay_path.read_text(encoding="utf-8").splitlines()
            rep0 = json.loads(rep_lines[0])
            rep0["budgets"]["sparse"]["attn_gate"]["coded_errors"] ^= 1
            bad_replay = tmp_dir / "bad_replay.jsonl"
            bad_replay.write_text(
                "\n".join([json.dumps(rep0)] + rep_lines[1:]) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Int mismatch"):
                p9_audit.audit_p9(raw_path, summary_path, bad_replay)

            # Corruption 4: inflate df in noninferiority from G-1 to G*L-1
            bad_df_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            bad_df_summary["noninferiority"]["df"] = 16 * 12 - 1
            bad_df_path = tmp_dir / "bad_df.json"
            bad_df_path.write_text(json.dumps(bad_df_summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "NI df="):
                p9_audit.audit_p9(raw_path, bad_df_path)


class FirewallAndAntiLeakageTests(unittest.TestCase):
    def test_seed_disjointness_clean_and_planted_overlaps(self):
        train = list(range(SEED_P9_TRAIN, SEED_P9_TRAIN + 32))
        dev = list(range(SEED_P9_DEV, SEED_P9_DEV + 32))
        fid = list(range(SEED_P9_FORMAL_ID, SEED_P9_FORMAL_ID + 32))
        food = list(range(SEED_P9_FORMAL_OOD, SEED_P9_FORMAL_OOD + 32))

        clean = check_seed_disjointness(train, dev, fid, food, strict=True)
        self.assertTrue(clean["disjoint"])
        self.assertEqual(clean["total_overlap_count"], 0)

        # Plant train vs formal_id overlap
        planted = check_seed_disjointness(train + [fid[5]], dev, fid, food, strict=False)
        self.assertFalse(planted["disjoint"])
        self.assertEqual(planted["pairwise_overlaps"]["train__formal_id"], [fid[5]])
        with self.assertRaises(ValueError):
            check_seed_disjointness(train + [fid[5]], dev, fid, food, strict=True)

        # Plant intra-pool duplicate
        dup = check_seed_disjointness(train, dev + [dev[0]], fid, food, strict=False)
        self.assertFalse(dup["disjoint"])
        self.assertEqual(dup["intra_pool_duplicates"]["dev"], [dev[0]])

        # Plant boolean seed (subclass of int in Python — must be rejected!)
        with self.assertRaises(TypeError):
            check_seed_disjointness([True], dev, fid, food)

    def test_no_hardcoded_n0_static_checker_clean_and_planted_leaks(self):
        p9_audit_files = [
            REPO_ROOT / "p9" / "contract.py",
            REPO_ROOT / "p9" / "independent_ref.py",
            REPO_ROOT / "p9" / "audit.py",
            REPO_ROOT / "p9" / "tests" / "test_p9_audit.py",
        ]
        clean = check_no_hardcoded_n0(p9_audit_files, strict=True)
        self.assertTrue(clean["clean"], f"Unexpected violations in P9 audit files: {clean['violations']}")

        # Must catch legacy hardcoded N0 in p8/link.py and p8c/link.py
        legacy = check_no_hardcoded_n0([REPO_ROOT / "p8" / "link.py", REPO_ROOT / "p8c" / "link.py"])
        self.assertFalse(legacy["clean"])
        kinds = {v["kind"] for v in legacy["violations"]}
        self.assertIn("hardcoded_n0_assignment", kinds)
        self.assertIn("legacy_n0_import", kinds)

        # Planted synthetic bad modules
        with tempfile.TemporaryDirectory() as tmp:
            bad_samples = {
                "bad_import.py": "from p8.link import N0\nx = N0\n",
                "bad_attr.py": "from p8 import link as ofdm\nx = ofdm.N0\n",
                "bad_noise.py": "from p8 import link as ofdm\ndef f(rng): return ofdm.noise(rng)\n",
                "bad_default.py": "def f(y, n0=0.12589254117941673):\n    return y / n0\n",
                "bad_power.py": "def f(y):\n    return y / (10.0 ** (-0.9))\n",
            }
            for fname, code_text in bad_samples.items():
                p = Path(tmp) / fname
                p.write_text(code_text, encoding="utf-8")
                rep = check_no_hardcoded_n0([p])
                self.assertFalse(rep["clean"], f"Static checker missed planted violation in {fname}")

    def test_public_observation_and_receiver_anti_leakage_firewall(self):
        pair = p8c_link.simulate(8035)
        y_b = pair.b["sparse"].astype(np.complex128).copy()
        y_b[PROBE_ROW] = complex(math.nan, math.nan)
        obs = PublicObservation(
            y_a=pair.a.astype(np.complex128).copy(),
            y_b=y_b,
            training_rows=SC_LAYOUTS["sparse"],
            probe_row=PROBE_ROW,
            n0=n0_from_snr(9.0),
            budget="sparse",
        )
        self.assertTrue(verify_public_observation(obs, strict=True)["valid"])

        # Catch unmasked probe row
        unmasked_obs = PublicObservation(
            y_a=obs.y_a,
            y_b=pair.b["sparse"].astype(np.complex128).copy(),
            training_rows=SC_LAYOUTS["sparse"],
            probe_row=PROBE_ROW,
            n0=obs.n0,
            budget="sparse",
        )
        self.assertFalse(verify_public_observation(unmasked_obs)["valid"])

        # Clean receiver passes anti-leakage audit
        def clean_rx(o: PublicObservation):
            llrs = np.zeros((T, N, 4), dtype=np.float64)
            pred = np.mean(o.y_b[list(o.training_rows)], axis=0)
            var = np.full(N, o.n0 + 0.05, dtype=np.float64)
            return llrs, pred, var

        self.assertTrue(audit_receiver_anti_leakage(clean_rx, obs, strict=True)["passed"])

        # Leaky receiver reading probe row is caught
        def probe_leaker(o: PublicObservation):
            llrs = np.zeros((T, N, 4), dtype=np.float64)
            pred = np.nan_to_num(o.y_b[o.probe_row], nan=0.0)
            var = np.full(N, o.n0 + 0.05, dtype=np.float64)
            return llrs, pred, var

        leak_rep = audit_receiver_anti_leakage(probe_leaker, obs, strict=False)
        self.assertFalse(leak_rep["passed"])
        self.assertGreater(leak_rep["max_probe_poison_diff"], 1e5)

        # Receiver coupling payload rows into probe prediction is caught
        def payload_leaker(o: PublicObservation):
            llrs = np.zeros((T, N, 4), dtype=np.float64)
            pred = np.mean(o.y_b[[1, 2]], axis=0)
            var = np.full(N, o.n0 + 0.05, dtype=np.float64)
            return llrs, pred, var

        pay_rep = audit_receiver_anti_leakage(payload_leaker, obs, strict=False)
        self.assertFalse(pay_rep["passed"])
        self.assertGreater(pay_rep["max_payload_probe_diff"], 1.0)


if __name__ == "__main__":
    unittest.main()
