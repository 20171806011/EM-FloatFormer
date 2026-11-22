"""Unit tests for Milestone 2: Classical Receivers (F6) and Unified Evaluator (F7)."""

from __future__ import annotations

import json
import math
import sys
import unittest
from dataclasses import replace

import numpy as np

from p12.audit import exact_sign_flip_p, holm_adjust
from research.realdata_sdr_ofdm.adapters.modulation import bits_to_symbols
from research.realdata_sdr_ofdm.adapters.sdr_ofdm_adapter import (
    N_ACTIVE,
    N_BITS,
    N_DATA_RES,
    N_FFT,
    N_SYMB,
    PILOT_FFT_INDICES,
    PILOT_SYMBOL_IDX,
    PILOT_VALUES,
    RealOFDMFrame,
    apply_stress_perturbation,
    build_frame,
)
from research.realdata_sdr_ofdm.eval import (
    compare_arms,
    compare_multiple_arms,
    evaluate_frames,
    evaluate_outputs,
    evaluate_paired,
    evaluate_receiver,
    paired_comparison,
    profile_receiver_pinned,
    run_stress_suite,
    subblock_group_ids,
)
from research.realdata_sdr_ofdm.eval.evaluator import (
    _get_peak_rss_mb,
    _is_valid_output,
)
from research.realdata_sdr_ofdm.receivers import (
    CLASSICAL_METHODS,
    ClassicalConfig,
    ClassicalReceiver,
    ReceiverOutput,
    calibrate_classical,
    calibrate_on_dev,
    run_frame,
)


def _make_m1_frame(
    seed: int,
    *,
    snr_db: float = 18.0,
    split: str = "dev",
    group_id: str = "blk_00",
    frame_id: int = 0,
    n_taps: int = 4,
    gauge_scale: complex = 1.0,
) -> RealOFDMFrame:
    rng = np.random.default_rng(int(seed))
    from research.realdata_sdr_ofdm.adapters.sdr_ofdm_adapter import build_masks

    pilot_mask, dc_mask, data_mask = build_masks()
    tx_bits = rng.integers(0, 2, size=(N_DATA_RES, 4), dtype=np.uint8)
    tx_sym = bits_to_symbols(tx_bits, qm=4)

    tx_grid = np.zeros((N_SYMB, N_FFT), dtype=np.complex128)
    tx_grid[data_mask] = tx_sym
    tx_grid[pilot_mask] = PILOT_VALUES

    taps = (rng.standard_normal(n_taps) + 1j * rng.standard_normal(n_taps)) / math.sqrt(2.0)
    taps *= np.exp(-0.5 * np.arange(n_taps, dtype=np.float64))
    taps /= math.sqrt(np.sum(np.abs(taps) ** 2))
    h_pad = np.zeros(N_FFT, dtype=np.complex128)
    h_pad[:n_taps] = taps
    h_freq = np.fft.fftshift(np.fft.fft(h_pad))

    n0 = 10.0 ** (-float(snr_db) / 10.0)
    noise = math.sqrt(n0 / 2.0) * (
        rng.standard_normal((N_SYMB, N_FFT)) + 1j * rng.standard_normal((N_SYMB, N_FFT))
    )
    rx_raw = (h_freq[None, :] * tx_grid + noise) * complex(gauge_scale)
    return build_frame(
        frame_id=frame_id,
        iq_raw=rx_raw,
        labels_raw=tx_bits,
        sinr_val=snr_db,
        split=split,
        group_id=group_id,
    )


def _scale_frame(fr: RealOFDMFrame, c: complex) -> RealOFDMFrame:
    r2 = abs(c) ** 2
    meta = dict(fr.meta)
    if "s0" in meta:
        meta["s0"] = float(meta["s0"]) * abs(c)
    return RealOFDMFrame(
        frame_id=fr.frame_id,
        split=fr.split,
        group_id=fr.group_id,
        rx_grid=(fr.rx_grid * complex(c)).astype(np.complex128),
        pilot_mask=fr.pilot_mask,
        dc_mask=fr.dc_mask,
        data_mask=fr.data_mask,
        pilot_values=fr.pilot_values,
        tx_bits=fr.tx_bits.copy(),
        tx_symbols=fr.tx_symbols.copy(),
        reported_sinr_db=fr.reported_sinr_db,
        est_n0=float(fr.est_n0) * r2,
        meta=meta,
    )


class TestM2ClassicalReceivers(unittest.TestCase):
    def test_contracts_aliases_and_unknown_method(self) -> None:
        fr = _make_m1_frame(101, snr_db=20.0)
        for m in ("ls_linear_zf", "ls_zf", "lmmse_1d", "lmmse_2d_tracked", "lmmse"):
            out = run_frame(fr, method=m)
            self.assertIsInstance(out, ReceiverOutput)
            self.assertEqual(out.h_est.shape, (N_SYMB, N_ACTIVE))
            self.assertEqual(out.h_var.shape, (N_SYMB, N_ACTIVE))
            self.assertEqual(out.eq_symbols.shape, (N_DATA_RES,))
            self.assertEqual(out.bit_llrs.shape, (N_DATA_RES, 4))
            np.testing.assert_array_equal(out.hard_bits, (out.bit_llrs > 0.0).astype(np.uint8))
            np.testing.assert_allclose(out.hard_symbols, bits_to_symbols(out.hard_bits, qm=4), atol=1e-12)
            self.assertFalse(out.aux_info["failed"])
            self.assertIsNone(out.aux_info["error"])
            self.assertGreater(out.aux_info["latency_ms"], 0.0)
            # M1 build_frame does not set meta["heldout_pilot_indices"], so 4-fold CV covers [1..12]
            self.assertEqual(out.aux_info["heldout_pilot_indices"], list(range(1, 13)))
            self.assertEqual(out.heldout_pilot_pred.shape, (12,))
            self.assertTrue(np.all(out.heldout_pilot_var > 0.0))

        rx = ClassicalReceiver(method="lmmse")
        out_cls = rx.run_frame(fr)
        self.assertEqual(out_cls.aux_info["method"], "lmmse_2d_tracked")

        with self.assertRaises(ValueError):
            run_frame(fr, method="nonexistent_method")

    def test_explicit_heldout_indices_masking(self) -> None:
        fr = _make_m1_frame(102, snr_db=22.0)
        fr.meta["heldout_pilot_indices"] = [1, 5, 9, 12]
        out = run_frame(fr, method="lmmse_1d")
        self.assertEqual(out.aux_info["heldout_pilot_indices"], [1, 5, 9, 12])
        self.assertEqual(out.heldout_pilot_pred.shape, (4,))
        self.assertTrue(np.all(out.heldout_pilot_var > 0.0))

        # Changing heldout pilot RX observations must NOT change heldout_pilot_pred (strict masking)
        fr_mod = _scale_frame(fr, 1.0)
        for idx in (1, 5, 9, 12):
            fr_mod.rx_grid[PILOT_SYMBOL_IDX, PILOT_FFT_INDICES[idx]] += 50.0 - 20.0j
        out_mod = run_frame(fr_mod, method="lmmse_1d")
        np.testing.assert_allclose(out_mod.heldout_pilot_pred, out.heldout_pilot_pred, atol=1e-10)

    def test_24_octave_cstar_equivariance_all_methods(self) -> None:
        fr = _make_m1_frame(103, snr_db=18.0)
        scales = [
            (2.0 ** -12) * np.exp(0.73j),
            (2.0 ** -6) * np.exp(-1.1j),
            (2.0 ** 6) * np.exp(2.2j),
            (2.0 ** 12) * np.exp(-0.37j),
        ]
        for m in CLASSICAL_METHODS:
            ref = run_frame(fr, method=m)
            for c in scales:
                sc = run_frame(_scale_frame(fr, c), method=m)
                np.testing.assert_allclose(sc.h_est, c * ref.h_est, rtol=1e-4, atol=1e-6)
                np.testing.assert_allclose(sc.h_var, (abs(c) ** 2) * ref.h_var, rtol=1e-4, atol=1e-12)
                np.testing.assert_allclose(sc.eq_symbols, ref.eq_symbols, atol=1e-4)
                np.testing.assert_allclose(sc.bit_llrs, ref.bit_llrs, atol=1e-4)
                self.assertEqual(int(np.sum(sc.hard_bits != ref.hard_bits)), 0)

    def test_edge_resilience_zero_pilots_extreme_noise_nan_and_no_label_leak(self) -> None:
        fr = _make_m1_frame(104, snr_db=18.0)

        # Zero pilots
        fr_zp = _scale_frame(fr, 1.0)
        fr_zp.rx_grid[fr_zp.pilot_mask] = 0.0
        for m in CLASSICAL_METHODS:
            out_zp = run_frame(fr_zp, method=m)
            self.assertTrue(np.all(np.isfinite(out_zp.h_est)))
            self.assertTrue(np.all(np.isfinite(out_zp.bit_llrs)))
            self.assertTrue(np.all(out_zp.h_var > 0.0))

        # Extreme noise bounds
        fr_hi = _scale_frame(fr, 1.0)
        fr_hi.est_n0 = 100.0
        o_hi = run_frame(fr_hi)
        self.assertTrue(np.all(np.isfinite(o_hi.bit_llrs)))
        self.assertLess(float(np.mean(np.abs(o_hi.bit_llrs))), 1.0)

        fr_lo = _scale_frame(fr, 1.0)
        fr_lo.est_n0 = 1e-9
        o_lo = run_frame(fr_lo)
        self.assertTrue(np.all(np.isfinite(o_lo.bit_llrs)))

        # NaN frame
        fr_nan = _scale_frame(fr, 1.0)
        fr_nan.rx_grid[1, 30] = complex(float("nan"), 0.0)
        o_nan = run_frame(fr_nan)
        self.assertTrue(o_nan.aux_info["failed"])
        self.assertTrue(bool(o_nan.aux_info["error"]))
        self.assertTrue(np.all(np.isfinite(o_nan.bit_llrs)))

        # Zero label leakage
        fr_pois = _scale_frame(fr, 1.0)
        fr_pois.tx_bits[:] = 1 - fr_pois.tx_bits
        fr_pois.tx_symbols[:] = -fr_pois.tx_symbols
        o_clean = run_frame(fr)
        o_pois = run_frame(fr_pois)
        np.testing.assert_array_equal(o_clean.h_est, o_pois.h_est)
        np.testing.assert_array_equal(o_clean.bit_llrs, o_pois.bit_llrs)

    def test_phase_tracking_helps_under_cfo_and_drift(self) -> None:
        fr = _make_m1_frame(105, snr_db=22.0)
        fr_p = apply_stress_perturbation(fr, cfo_norm=0.04, phase_drift_rad=0.6, seed=105)
        o_1d = run_frame(fr_p, method="lmmse_1d")
        o_2d = run_frame(fr_p, method="lmmse_2d_tracked")
        ber_1d = float(np.mean(o_1d.hard_bits != fr_p.tx_bits))
        ber_2d = float(np.mean(o_2d.hard_bits != fr_p.tx_bits))
        self.assertLess(ber_2d, ber_1d)
        self.assertLess(ber_2d, 0.05)

    def test_dev_only_calibration_firewall(self) -> None:
        dev_frames = [_make_m1_frame(200 + i, snr_db=16.0, split="dev", frame_id=i) for i in range(4)]
        test_fr = _make_m1_frame(300, snr_db=16.0, split="test", frame_id=10)
        guard_fr = _make_m1_frame(301, snr_db=16.0, split="guard", frame_id=11)

        with self.assertRaises(ValueError):
            calibrate_on_dev([])
        with self.assertRaises(ValueError):
            calibrate_on_dev([test_fr])
        with self.assertRaises(ValueError):
            calibrate_classical([dev_frames[0], guard_fr])

        tuned = calibrate_on_dev(
            dev_frames,
            n0_scales=(0.8, 1.0),
            tau_rms_grid=(1.5,),
            fd_norm_grid=(0.0,),
            cpe_refine_grid=(0.0,),
        )
        self.assertEqual(set(tuned.keys()), set(CLASSICAL_METHODS))
        for cfg in tuned.values():
            self.assertIsInstance(cfg, ClassicalConfig)

    def test_uppercase_split_firewall_and_pilot_values_fallback(self) -> None:
        dev_fr = _make_m1_frame(210, snr_db=16.0, split="dev", frame_id=0)
        for bad_split in ("TEST", "GUARD", " Test ", "Guard"):
            bad_fr = _make_m1_frame(211, snr_db=16.0, split=bad_split, frame_id=1)
            with self.assertRaises(ValueError, msg=f"solo split={bad_split!r}"):
                calibrate_on_dev([bad_fr])
            with self.assertRaises(ValueError, msg=f"mixed split={bad_split!r}"):
                calibrate_classical([dev_fr, bad_fr])

        fr_base = _make_m1_frame(212, snr_db=18.0, split="test", frame_id=2)
        pv_all_zero = np.zeros_like(fr_base.pilot_values)
        pv_one_zero = fr_base.pilot_values.copy()
        pv_one_zero[5] = 0.0 + 0.0j
        pv_nan = fr_base.pilot_values.copy()
        pv_nan[2] = complex(float("nan"), 0.0)
        pv_bad_shape = fr_base.pilot_values[:12].copy()

        for pv_tag, pv_bad in (
            ("all_zero", pv_all_zero),
            ("single_zero", pv_one_zero),
            ("nan", pv_nan),
            ("bad_shape", pv_bad_shape),
        ):
            fr_bad = replace(fr_base, pilot_values=pv_bad)
            for m in CLASSICAL_METHODS:
                out = run_frame(fr_bad, method=m)
                self.assertTrue(out.aux_info["failed"], msg=f"{pv_tag}/{m}")
                self.assertTrue(bool(out.aux_info["error"]), msg=f"{pv_tag}/{m}")
                self.assertEqual(out.h_est.shape, (N_SYMB, N_ACTIVE))
                self.assertEqual(out.h_var.shape, (N_SYMB, N_ACTIVE))
                self.assertEqual(out.eq_symbols.shape, (N_DATA_RES,))
                self.assertEqual(out.bit_llrs.shape, (N_DATA_RES, 4))
                for arr in (out.h_est, out.h_var, out.eq_symbols, out.bit_llrs, out.hard_bits, out.hard_symbols):
                    self.assertTrue(np.all(np.isfinite(arr)), msg=f"{pv_tag}/{m}")
                self.assertTrue(np.all(out.h_var > 0.0), msg=f"{pv_tag}/{m}")
                np.testing.assert_array_equal(out.hard_bits, (out.bit_llrs > 0.0).astype(np.uint8))
                np.testing.assert_allclose(out.hard_symbols, bits_to_symbols(out.hard_bits, qm=4), atol=1e-12)


class TestM2UnifiedEvaluator(unittest.TestCase):
    def test_aliases_empty_rejection_and_exact_counts(self) -> None:
        self.assertIs(evaluate_frames, evaluate_receiver)
        self.assertIs(evaluate_outputs, evaluate_receiver)
        self.assertIs(paired_comparison, compare_arms)
        self.assertIs(evaluate_paired, compare_arms)

        with self.assertRaises(ValueError):
            evaluate_receiver([])

        frames = [_make_m1_frame(400 + i, snr_db=22.0, group_id=f"g{i // 2}", frame_id=i) for i in range(4)]
        pairs = [(fr, run_frame(fr)) for fr in frames]
        res = evaluate_receiver(pairs)
        self.assertIsInstance(res["bit_errors"], int)
        self.assertIsInstance(res["total_bits"], int)
        self.assertIsInstance(res["frame_errors"], int)
        self.assertEqual(res["total_bits"], 4 * N_BITS)
        self.assertEqual(res["total_frames"], 4)
        self.assertEqual(res["coverage_pct"], 100.0)
        self.assertGreater(res["peak_rss_mb"], 0.0)
        self.assertGreater(res["pilot_cal_ratio"], 0.0)
        self.assertTrue(math.isfinite(res["pilot_nll"]))
        json.dumps(res, allow_nan=False)

    def test_paired_comparisons_g16_g6_g3_g20_and_subblocks(self) -> None:
        # G=16 exact match to p12.audit
        base_16, cand_16 = [], []
        for g in range(16):
            fr = _make_m1_frame(500 + g, snr_db=14.0, split="test", group_id=f"b_{g:02d}", frame_id=g)
            base_16.append((fr, run_frame(fr, method="ls_zf")))
            cand_16.append((fr, run_frame(fr, method="lmmse")))
        comp16 = compare_arms(base_16, cand_16)
        ref16 = exact_sign_flip_p(comp16["cluster_diffs_pp"])
        self.assertAlmostEqual(comp16["p_two_sided"], ref16["p_two_sided"], places=12)
        self.assertIn(comp16["verdict"], ("支持", "反证", "证据不足"))
        self.assertLessEqual(comp16["boot_ci95_lo_pp"], comp16["boot_ci95_hi_pp"])

        # G=6, G=3, G=1, G=20 graceful handling
        for g_cnt in (6, 3, 1, 20):
            bp = [( _make_m1_frame(600 + i, group_id=f"g{i}", frame_id=i), None) for i in range(g_cnt)]
            bp_pairs = [(fr, run_frame(fr, method="ls_zf")) for fr, _ in bp]
            cp_pairs = [(fr, run_frame(fr, method="lmmse")) for fr, _ in bp]
            comp = compare_arms(bp_pairs, cp_pairs, n_boot=200)
            self.assertEqual(comp["n_groups"], g_cnt)
            self.assertTrue(0.0 < comp["p_two_sided"] <= 1.0)

        # Multi-arm Holm adjustment
        multi = compare_multiple_arms(
            {"ls_zf": base_16, "lmmse_1d": base_16, "lmmse": cand_16},
            anchor_name="ls_zf",
        )
        expected_holms = holm_adjust([multi["lmmse_1d"]["p_two_sided"], multi["lmmse"]["p_two_sided"]])
        self.assertAlmostEqual(multi["lmmse_1d"]["p_holm"], expected_holms[0], places=12)
        self.assertAlmostEqual(multi["lmmse"]["p_holm"], expected_holms[1], places=12)

        # Subblock group IDs
        frames_3 = [_make_m1_frame(700 + i, group_id=f"block_{i // 16:02d}", frame_id=i) for i in range(48)]
        sub16 = subblock_group_ids(frames_3, target_g=16)
        self.assertEqual(len(set(sub16.values())), 16)

    def test_profiler_and_stress_suite(self) -> None:
        frames = [_make_m1_frame(800 + i, snr_db=22.0, frame_id=i) for i in range(4)]
        snaps = [f.rx_grid.copy() for f in frames]
        prof = profile_receiver_pinned(frames, lambda f: run_frame(f, method="lmmse"), warmup_frames=2)
        self.assertGreater(prof["mean_ms_per_frame"], 0.0)
        self.assertGreater(prof["tracemalloc_peak_mb"], 0.0)

        suite = run_stress_suite(
            frames,
            lambda f: run_frame(f, method="lmmse"),
            extra_awgn_db=(None, 10.0, 0.0),
            cfo_norm=(0.0, 0.03),
            phase_drift_rad=(0.0, 0.4),
        )
        for fr, snap in zip(frames, snaps):
            np.testing.assert_array_equal(fr.rx_grid, snap)
        by_cond = {c["condition"]: c["metrics"] for c in suite["conditions"]}
        self.assertLess(by_cond["clean"]["sym_mse"], by_cond["awgn_10.0db"]["sym_mse"])
        self.assertLess(by_cond["awgn_10.0db"]["sym_mse"], by_cond["awgn_0.0db"]["sym_mse"])

    def test_peak_rss_mb_without_psutil_and_cross_platform(self) -> None:
        fr = _make_m1_frame(850, snr_db=20.0, split="test", frame_id=0)
        out = run_frame(fr)
        sentinel = object()
        prev_psutil = sys.modules.get("psutil", sentinel)
        sys.modules["psutil"] = None  # type: ignore[assignment]
        try:
            rss_mb = _get_peak_rss_mb()
            self.assertIsInstance(rss_mb, float)
            self.assertTrue(math.isfinite(rss_mb))
            self.assertGreater(rss_mb, 10.0)

            ev = evaluate_receiver([(fr, out)])
            self.assertGreater(ev["peak_rss_mb"], 10.0)
        finally:
            if prev_psutil is sentinel:
                sys.modules.pop("psutil", None)
            else:
                sys.modules["psutil"] = prev_psutil

    def test_is_valid_output_rejects_shapes_negative_var_and_bad_heldout(self) -> None:
        fr = _make_m1_frame(860, snr_db=20.0, split="test", frame_id=0)
        out_ok = run_frame(fr)
        self.assertEqual(_is_valid_output(fr, out_ok), (True, None))

        # Keep (None, None) heldout_pilot_* valid
        out_no_ho = replace(out_ok, heldout_pilot_pred=None, heldout_pilot_var=None)
        self.assertEqual(_is_valid_output(fr, out_no_ho), (True, None))

        # 1. Wrong bit_llrs shape
        out_bad_llr = replace(out_ok, bit_llrs=np.zeros((N_DATA_RES, 2), dtype=np.float64))
        ok_llr, msg_llr = _is_valid_output(fr, out_bad_llr)
        self.assertFalse(ok_llr)
        self.assertIn("bit_llrs", str(msg_llr))

        # 2. Negative h_var
        neg_hvar = out_ok.h_var.copy()
        neg_hvar[0, 0] = -1e-6
        out_neg_hvar = replace(out_ok, h_var=neg_hvar)
        ok_var, msg_var = _is_valid_output(fr, out_neg_hvar)
        self.assertFalse(ok_var)
        self.assertIn("h_var", str(msg_var))

        # 3. NaN in heldout_pilot_pred
        bad_ho_pred = out_ok.heldout_pilot_pred.copy()
        bad_ho_pred[0] = complex(float("nan"), 0.0)
        out_bad_ho_pred = replace(out_ok, heldout_pilot_pred=bad_ho_pred)
        ok_hop, msg_hop = _is_valid_output(fr, out_bad_ho_pred)
        self.assertFalse(ok_hop)
        self.assertIn("heldout_pilot_pred", str(msg_hop))

        # 4. NaN / < 0 in heldout_pilot_var
        bad_ho_var_nan = out_ok.heldout_pilot_var.copy()
        bad_ho_var_nan[0] = float("nan")
        out_bad_ho_var_nan = replace(out_ok, heldout_pilot_var=bad_ho_var_nan)
        self.assertFalse(_is_valid_output(fr, out_bad_ho_var_nan)[0])

        bad_ho_var_neg = out_ok.heldout_pilot_var.copy()
        bad_ho_var_neg[0] = -1e-9
        out_bad_ho_var_neg = replace(out_ok, heldout_pilot_var=bad_ho_var_neg)
        ok_hov, msg_hov = _is_valid_output(fr, out_bad_ho_var_neg)
        self.assertFalse(ok_hov)
        self.assertIn("heldout_pilot_var", str(msg_hov))

        # 5. Shape-mismatched heldout_pilot_pred vs heldout_pilot_var
        out_mismatch_ho = replace(
            out_ok,
            heldout_pilot_pred=out_ok.heldout_pilot_pred[:4],
            heldout_pilot_var=out_ok.heldout_pilot_var[:3],
        )
        ok_hom, msg_hom = _is_valid_output(fr, out_mismatch_ho)
        self.assertFalse(ok_hom)
        self.assertIn("heldout_pilot", str(msg_hom))

        for bad_out in (
            out_bad_llr,
            out_neg_hvar,
            out_bad_ho_pred,
            out_bad_ho_var_nan,
            out_bad_ho_var_neg,
            out_mismatch_ho,
        ):
            ev = evaluate_receiver([(fr, bad_out)])
            self.assertEqual(ev["success_count"], 0)
            self.assertEqual(ev["fail_count"], 1)
            self.assertEqual(ev["coverage_pct"], 0.0)

    def test_compare_arms_excludes_failed_frames_consistently_and_rejects_empty_cluster(self) -> None:
        frames = [
            _make_m1_frame(870 + i, snr_db=24.0, split="test", group_id=f"g{i // 2}", frame_id=i)
            for i in range(4)
        ]
        base_pairs = []
        for fr in frames:
            out = run_frame(fr, method="lmmse")
            zero_err_out = replace(
                out,
                hard_bits=fr.tx_bits.copy(),
                hard_symbols=fr.tx_symbols.copy(),
                eq_symbols=fr.tx_symbols.copy(),
            )
            base_pairs.append((fr, zero_err_out))

        hb_1err = frames[0].tx_bits.copy()
        hb_1err[0, 0] ^= 1
        eq_1err = frames[0].tx_symbols.copy()
        eq_1err[0] += 1.0 + 0.0j
        out_cand_0 = replace(
            base_pairs[0][1],
            hard_bits=hb_1err,
            hard_symbols=bits_to_symbols(hb_1err, qm=4),
            eq_symbols=eq_1err,
        )
        out_cand_1 = replace(
            base_pairs[1][1],
            hard_bits=(1 - frames[1].tx_bits).astype(np.uint8),
            eq_symbols=frames[1].tx_symbols + 10.0,
            aux_info={**base_pairs[1][1].aux_info, "failed": True, "error": "injected failure"},
        )
        out_cand_2 = base_pairs[2][1]
        bad_h_est = base_pairs[3][1].h_est.copy()
        bad_h_est[0, 0] = complex(float("nan"), 0.0)
        bad_h_var = base_pairs[3][1].h_var.copy()
        bad_h_var[0, 0] = -1.0
        out_cand_3 = replace(
            base_pairs[3][1],
            h_est=bad_h_est,
            h_var=bad_h_var,
            bit_llrs=np.zeros((N_DATA_RES, 2), dtype=np.float64),
            hard_bits=np.zeros((10, 4), dtype=np.uint8),
            eq_symbols=frames[3].tx_symbols + 10.0,
        )
        cand_pairs = [
            (frames[0], out_cand_0),
            (frames[1], out_cand_1),
            (frames[2], out_cand_2),
            (frames[3], out_cand_3),
        ]

        ev_cand = evaluate_receiver(cand_pairs)
        self.assertEqual(ev_cand["success_count"], 2)
        self.assertEqual(ev_cand["fail_count"], 2)
        self.assertEqual(ev_cand["bit_errors"], 1)
        self.assertEqual(ev_cand["frame_errors"], 1)
        self.assertEqual(ev_cand["sym_errors"], 1)

        metric_to_key = {"ber": "bit_errors", "fer": "frame_errors", "ser": "sym_errors"}
        for metric, err_key in metric_to_key.items():
            comp = compare_arms(base_pairs, cand_pairs, metric=metric, n_boot=50)
            self.assertEqual(comp["base_total_errors"], 0, msg=metric)
            self.assertEqual(comp["cand_total_errors"], ev_cand[err_key], msg=metric)
            self.assertEqual(comp["error_delta"], 1, msg=metric)

        comp_mse = compare_arms(base_pairs, cand_pairs, metric="sym_mse", n_boot=50)
        self.assertAlmostEqual(comp_mse["base_total_errors"], 0.0, places=12)
        self.assertAlmostEqual(comp_mse["cand_total_errors"], 1.0, places=12)
        self.assertTrue(math.isfinite(comp_mse["mean_pp"]))

        # 0-valid-frame cluster must raise ValueError with cluster gid context
        for metric in ("ber", "fer", "ser", "sym_mse"):
            with self.assertRaises(ValueError, msg=metric) as ctx:
                compare_arms(
                    [(frames[0], base_pairs[0][1])],
                    [(frames[0], out_cand_1)],
                    metric=metric,
                    n_boot=20,
                )
            self.assertIn("g0", str(ctx.exception), msg=metric)


if __name__ == "__main__":
    unittest.main()
