"""Unit tests for P9-A channel/waveform generator, strong CP-SC decoder, and ML certificate."""
import ast
from pathlib import Path
import unittest

import numpy as np

from p8 import coding, link as ofdm, tracker
from p8.study import history as p8_history
from p8c import link as p8c_link, receiver as p8c_receiver
from p9 import channel, decoder
from p9.contract import (
    ALL_REGIMES,
    CORE_ROWS,
    CP,
    DELAYS,
    FRAME_DURATION,
    N,
    PROBE_ROW,
    SC_LAYOUTS,
    SEED_P9_DEV,
    SNR_GRID_DB,
    T,
    n0_from_snr,
)


def _serial_reference_piecewise(useful: np.ndarray, ch) -> np.ndarray:
    """Independent sample-by-sample CP-inserted linear convolution using evaluate_ray_gains."""
    stream_tx = np.column_stack((useful[:, -CP:], useful)).reshape(-1)
    sample_times = np.arange(stream_tx.size, dtype=np.float64) / N
    ray_gains = channel.evaluate_ray_gains(ch, sample_times)
    received = np.zeros_like(stream_tx)
    for ray_idx, delay in enumerate(DELAYS):
        d = int(delay)
        delayed = np.zeros_like(stream_tx)
        delayed[d:] = stream_tx[: len(stream_tx) - d]
        received += ray_gains[ray_idx] * delayed
    return received.reshape(len(useful), -1)[:, CP:]


class TestP9AChannel(unittest.TestCase):
    def test_zero_hardcoded_n0_source_and_runtime_invariance(self):
        """Verify p9/channel.py and p9/decoder.py never depend on hardcoded p8.link.N0 or TAP_NOISE."""
        root = Path(__file__).resolve().parents[1]
        forbidden = {"N0", "TAP_NOISE"}
        for fname in ("channel.py", "decoder.py"):
            tree = ast.parse((root / fname).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        self.assertNotIn(alias.name, forbidden, f"Forbidden import {alias.name} in {fname}")
                elif isinstance(node, ast.Name):
                    self.assertNotIn(node.id, forbidden, f"Forbidden name {node.id} in {fname}")
                elif isinstance(node, ast.Attribute):
                    self.assertNotIn(node.attr, forbidden, f"Forbidden attribute {node.attr} in {fname}")

        seed = SEED_P9_DEV + 101
        baseline = {}
        for snr_db in SNR_GRID_DB:
            obs, tr = channel.simulate_link(seed, snr_db=snr_db, regime="continuous", budget="sparse")
            st_a = channel.track_frame_a(obs.y_a, obs.n0)
            llrs, pred, var = channel.receive_frame_b(st_a, obs.y_b, obs.training_rows, obs.n0)
            baseline[snr_db] = (obs.y_a, obs.y_b, llrs, pred, var)

        old_n0 = ofdm.N0
        old_tap_noise = tracker.TAP_NOISE
        try:
            ofdm.N0 = 12345.0
            tracker.TAP_NOISE = 99999.0
            for snr_db in SNR_GRID_DB:
                obs, _ = channel.simulate_link(seed, snr_db=snr_db, regime="continuous", budget="sparse")
                st_a = channel.track_frame_a(obs.y_a, obs.n0)
                llrs, pred, var = channel.receive_frame_b(st_a, obs.y_b, obs.training_rows, obs.n0)
                b_ya, b_yb, b_llrs, b_pred, b_var = baseline[snr_db]
                np.testing.assert_array_equal(obs.y_a, b_ya)
                np.testing.assert_array_equal(obs.y_b[:PROBE_ROW], b_yb[:PROBE_ROW])
                np.testing.assert_array_equal(llrs, b_llrs)
                np.testing.assert_array_equal(pred, b_pred)
                np.testing.assert_array_equal(var, b_var)
        finally:
            ofdm.N0 = old_n0
            tracker.TAP_NOISE = old_tap_noise

    def test_paired_unit_noise_across_snr_grid(self):
        """6/9/12/15 dB share identical true channels, codewords, clean waveforms, and unit noise."""
        seed = SEED_P9_DEV + 202
        sweep = channel.simulate_snr_sweep(seed, SNR_GRID_DB, regime="partial_shift", budget="sparse")
        ref_obs, ref_tr = sweep[9.0]

        non_probe = [r for r in range(T) if r != PROBE_ROW]
        ref_unit_b = (ref_obs.y_b[non_probe] - ref_tr.y_b_full_clean[non_probe]) / np.sqrt(ref_obs.n0)
        ref_unit_probe = (ref_tr.y_b_probe_noisy - ref_tr.y_b_probe_clean) / np.sqrt(ref_obs.n0)

        a_words = ofdm.traffic(ofdm.stream(seed, 1))
        a_clean = ofdm.propagate(ofdm.tx_grid(a_words, ofdm.PILOT_ROWS), ref_tr.channel_a)
        ref_unit_a = (ref_obs.y_a - a_clean) / np.sqrt(ref_obs.n0)

        for snr_db, (obs, tr) in sweep.items():
            self.assertAlmostEqual(obs.n0, n0_from_snr(snr_db), places=15)
            np.testing.assert_array_equal(tr.h_b_true, ref_tr.h_b_true)
            np.testing.assert_array_equal(tr.b_info, ref_tr.b_info)
            np.testing.assert_array_equal(tr.b_code, ref_tr.b_code)
            np.testing.assert_array_equal(tr.b_symbols, ref_tr.b_symbols)
            np.testing.assert_array_equal(tr.y_b_full_clean, ref_tr.y_b_full_clean)
            self.assertEqual(tr.changed_ray, ref_tr.changed_ray)
            self.assertEqual(tr.delta_freq, ref_tr.delta_freq)

            unit_b = (obs.y_b[non_probe] - tr.y_b_full_clean[non_probe]) / np.sqrt(obs.n0)
            unit_probe = (tr.y_b_probe_noisy - tr.y_b_probe_clean) / np.sqrt(obs.n0)
            unit_a = (obs.y_a - a_clean) / np.sqrt(obs.n0)
            np.testing.assert_allclose(unit_b, ref_unit_b, atol=1e-13, rtol=0.0)
            np.testing.assert_allclose(unit_probe, ref_unit_probe, atol=1e-13, rtol=0.0)
            np.testing.assert_allclose(unit_a, ref_unit_a, atol=1e-13, rtol=0.0)

    def test_exact_9db_continuous_parity_with_p8c(self):
        """At 9 dB continuous regime, p9/channel matches p8c waveforms, tracker, and receiver."""
        for seed in (3117, 6017, SEED_P9_DEV + 303):
            pair = p8c_link.simulate(seed)
            p8_state = p8_history(pair.a)
            for budget, rows in SC_LAYOUTS.items():
                obs, tr = channel.simulate_link(seed, snr_db=9.0, regime="continuous", budget=budget)
                # Strict probe masking in PublicObservation
                self.assertTrue(np.isnan(obs.y_b[PROBE_ROW].real).all())
                self.assertTrue(np.isnan(obs.y_b[PROBE_ROW].imag).all())

                non_probe = [r for r in range(T) if r != PROBE_ROW]
                np.testing.assert_allclose(obs.y_a, pair.a, atol=1e-14, rtol=0.0)
                np.testing.assert_allclose(obs.y_b[non_probe], pair.b[budget][non_probe], atol=1e-14, rtol=0.0)
                np.testing.assert_allclose(tr.y_b_probe_noisy, pair.b[budget][PROBE_ROW], atol=1e-14, rtol=0.0)
                np.testing.assert_allclose(tr.y_b_full_clean, pair.clean_b[budget], atol=1e-14, rtol=0.0)
                np.testing.assert_array_equal(tr.b_info, pair.b_words.info)
                np.testing.assert_array_equal(tr.b_code, pair.b_words.code)
                np.testing.assert_array_equal(tr.b_symbols, pair.b_words.symbols)

                # Tracker parity
                p9_state = channel.track_frame_a(obs.y_a, obs.n0)
                self.assertEqual(p9_state.epoch, p8_state.epoch)
                np.testing.assert_allclose(p9_state.log_weight, p8_state.log_weight, atol=1e-14, rtol=0.0)
                np.testing.assert_allclose(p9_state.mean, p8_state.mean, atol=1e-14, rtol=0.0)
                np.testing.assert_allclose(p9_state.variance, p8_state.variance, atol=1e-14, rtol=0.0)

                # Receiver & oracle parity
                blind = pair.b[budget].copy()
                blind[PROBE_ROW] = complex(np.nan, np.nan)
                p8_llrs, p8_pred, p8_var = p8c_receiver.receive(p8_state, blind, rows)
                p9_llrs, p9_pred, p9_var = channel.receive_frame_b(p9_state, obs.y_b, rows, obs.n0)
                np.testing.assert_allclose(p9_llrs, p8_llrs, atol=1e-13, rtol=0.0)
                np.testing.assert_allclose(p9_pred, p8_pred, atol=1e-14, rtol=0.0)
                np.testing.assert_allclose(p9_var, p8_var, atol=1e-14, rtol=0.0)

                o8_llrs, o8_pred, o8_var = p8c_receiver.oracle(pair.channel, blind, rows)
                o9_llrs, o9_pred, o9_var = channel.oracle_readout_b(tr, obs.y_b, rows, obs.n0)
                np.testing.assert_allclose(o9_llrs, o8_llrs, atol=1e-13, rtol=0.0)
                np.testing.assert_allclose(o9_pred, o8_pred, atol=1e-14, rtol=0.0)
                np.testing.assert_allclose(o9_var, o8_var, atol=1e-14, rtol=0.0)

            # donor_mismatch parity on Frame A
            obs_dm, tr_dm = channel.simulate_link(seed, snr_db=9.0, regime="donor_mismatch", budget="sparse")
            np.testing.assert_allclose(obs_dm.y_a, pair.donor_a, atol=1e-14, rtol=0.0)
            np.testing.assert_allclose(tr_dm.y_b_full_clean, pair.clean_b["sparse"], atol=1e-14, rtol=0.0)

    def test_exact_phase_continuity_at_boundary_and_midframe_shifts(self):
        """Verify exact complex phase continuity and frequency jump at shift_row=0 and shift_row=6."""
        for seed in (SEED_P9_DEV + 401, SEED_P9_DEV + 402, SEED_P9_DEV + 403):
            _, tr_cont = channel.simulate_link(seed, snr_db=9.0, regime="continuous", budget="sparse")
            for regime, expected_row, lo, hi in (
                ("partial_shift", 0, 0.035, 0.065),
                ("ood_strong_shift", 0, 0.085, 0.120),
                ("ood_midframe_shift", 6, 0.035, 0.065),
            ):
                obs, tr = channel.simulate_link(seed, snr_db=9.0, regime=regime, budget="sparse")
                self.assertEqual(tr.shift_row, expected_row)
                self.assertIn(tr.changed_ray, (0, 1, 2))
                self.assertGreaterEqual(abs(tr.delta_freq), lo)
                self.assertLessEqual(abs(tr.delta_freq), hi)
                self.assertTrue(np.isnan(obs.y_b[PROBE_ROW].real).all())
                self.assertTrue(np.isnan(obs.y_b[PROBE_ROW].imag).all())

                ch = tr.channel_b
                self.assertIsInstance(ch, channel.PiecewiseChannel)
                t_s = channel.row_start_time_b(expected_row)
                self.assertAlmostEqual(ch.shift_time, t_s, places=15)
                if expected_row == 0:
                    self.assertAlmostEqual(t_s, FRAME_DURATION, places=15)

                # 1. Exact complex gain equality at t_s between pre, post, and piecewise evaluation.
                g_pre = channel.evaluate_ray_gains(ch.pre, np.array([t_s]))[:, 0]
                g_post = channel.evaluate_ray_gains(ch.post, np.array([t_s]))[:, 0]
                g_pw = channel.evaluate_ray_gains(ch, np.array([t_s]))[:, 0]
                np.testing.assert_allclose(g_pw, g_pre, atol=0.0, rtol=0.0)
                np.testing.assert_allclose(g_post, g_pre, atol=1e-13, rtol=0.0)

                # 2. Left/right continuity across t_s and exact instantaneous frequency jump.
                dt = 1e-6
                g_minus = channel.evaluate_ray_gains(ch, np.array([t_s - dt]))[:, 0]
                g_plus = channel.evaluate_ray_gains(ch, np.array([t_s + dt]))[:, 0]
                # Complex gain changes only by O(dt), zero phase discontinuity!
                self.assertLess(float(np.max(np.abs(g_plus - g_minus))), 1e-4)

                # Instantaneous frequency = angle(g(t+dt)/g(t)) / (2*pi*dt)
                f_left = np.angle(g_pw / g_minus) / (2.0 * np.pi * dt)
                f_right = np.angle(g_plus / g_pw) / (2.0 * np.pi * dt)
                f_jump = f_right - f_left
                for ray in range(len(DELAYS)):
                    expected_jump = tr.delta_freq if ray == tr.changed_ray else 0.0
                    self.assertAlmostEqual(float(f_jump[ray]), expected_jump, places=5)

                # 3. For ood_midframe_shift (shift_row=6), rows 0..5 match continuous identically!
                if expected_row > 0:
                    np.testing.assert_array_equal(tr.h_b_true[:, :expected_row], tr_cont.h_b_true[:, :expected_row])
                    # Unchanged rays remain identical across all 20 rows.
                    for ray in range(len(DELAYS)):
                        if ray != tr.changed_ray:
                            np.testing.assert_array_equal(tr.h_b_true[ray], tr_cont.h_b_true[ray])
                        else:
                            self.assertGreater(
                                float(np.max(np.abs(tr.h_b_true[ray, expected_row:] - tr_cont.h_b_true[ray, expected_row:]))),
                                1e-2,
                            )

                # 4. Independent sample-by-sample serial reference across A/B and row-6 boundaries.
                a_grid = ofdm.tx_grid(ofdm.traffic(ofdm.stream(seed, 1)), ofdm.PILOT_ROWS)
                a_time = np.fft.ifft(a_grid, norm="ortho")
                b_words = ofdm.Traffic(tr.b_info, tr.b_code, tr.b_symbols)
                b_grid = p8c_link.tx_grid(b_words, obs.training_rows)
                serial_ref = _serial_reference_piecewise(np.vstack((a_time, b_grid)), ch)
                np.testing.assert_allclose(serial_ref[T:], tr.y_b_full_clean, atol=3e-14, rtol=0.0)

    def test_all_regimes_and_full_reset_independence(self):
        seed = SEED_P9_DEV + 505
        for regime in ALL_REGIMES:
            obs, tr = channel.simulate_link(seed, snr_db=9.0, regime=regime, budget="sparse")
            self.assertEqual(tr.regime, regime)
            self.assertTrue(np.isnan(obs.y_b[PROBE_ROW].real).all())
            self.assertTrue(np.isnan(obs.y_b[PROBE_ROW].imag).all())
            st = channel.track_frame_a(obs.y_a, obs.n0)
            llrs, pred, var = channel.receive_frame_b(st, obs.y_b, obs.training_rows, obs.n0)
            self.assertTrue(np.isfinite(llrs).all())
            self.assertTrue(np.isfinite(pred).all())
            self.assertTrue(np.isfinite(var).all())
            self.assertTrue((var >= obs.n0 - 1e-12).all())

        _, tr_reset = channel.simulate_link(seed, snr_db=9.0, regime="full_reset", budget="sparse")
        self.assertIsNone(tr_reset.changed_ray)
        self.assertEqual(tr_reset.delta_freq, 0.0)
        self.assertIsNone(tr_reset.shift_row)
        self.assertFalse(np.allclose(tr_reset.channel_a.gain, tr_reset.channel_b.gain))


class TestP9ADecoderAndCertificate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        decoder.warmup_numba()

    def test_bit_packing_and_nibble_qam_exact_match(self):
        rng = np.random.default_rng(910001)
        infos = rng.integers(0, 2, size=(16, 64), dtype=np.uint8)
        codes = decoder.native_ldpc_codeword(infos)
        packed = decoder.pack_codewords(codes)
        unpacked = decoder.unpack_codewords(packed)
        np.testing.assert_array_equal(unpacked, codes)

        H = np.eye(N, dtype=np.complex128)
        for i in range(16):
            self.assertTrue(decoder.is_valid_codeword(codes[i]))
            x_ref = coding.qam(codes[i])
            # Evaluate CSR waveform metric with y = x_ref and H = I -> must be exact 0.0
            y_re, y_im, rp, ci, vr, vi = decoder._prepare_weighted_csr(x_ref, H, None)
            x_re = np.empty(32, dtype=np.float64)
            x_im = np.empty(32, dtype=np.float64)
            d = decoder._waveform_metric_csr(
                packed[i, 0], packed[i, 1], y_re, y_im, rp, ci, vr, vi,
                decoder._NIBBLE_RE, decoder._NIBBLE_IM, x_re, x_im,
            )
            self.assertLess(d, 1e-28)

    def test_zero_noise_perfect_decoding_and_awgn_sanity(self):
        seed = SEED_P9_DEV + 606
        obs_z, tr_z = channel.simulate_diagnostic(seed, kind="zero_noise", budget="sparse")
        self.assertEqual(obs_z.n0, 0.0)
        Hs = channel.build_sc_channel_matrices(tr_z.h_b_true, CORE_ROWS)
        zero_noise = np.zeros((len(CORE_ROWS), N), dtype=np.float64)

        llrs_z, pred_z, var_z = channel.oracle_readout_b(tr_z, obs_z.y_b, obs_z.training_rows, obs_z.n0)
        np.testing.assert_allclose(pred_z, tr_z.y_b_probe_clean, atol=1e-13, rtol=0.0)
        np.testing.assert_allclose(var_z, 0.0, atol=1e-15, rtol=0.0)

        native_info = decoder.decode_native_ldpc(llrs_z[list(CORE_ROWS)].reshape(16, 128))
        np.testing.assert_array_equal(native_info, tr_z.b_info[:16])

        for ls in ("small", "large"):
            for k, r in enumerate(CORE_ROWS):
                res = decoder.decode_strong_sc(obs_z.y_b[r], Hs[k], zero_noise[k], list_size=ls, turbo_pic=True)
                np.testing.assert_array_equal(res.info_bits, tr_z.b_info[k])
                np.testing.assert_array_equal(res.codeword, tr_z.b_code[k])
                self.assertLess(res.metric, 1e-24)

        # Single-ray AWGN diagnostic sanity
        obs_a, tr_a = channel.simulate_diagnostic(seed, kind="awgn", snr_db=15.0, budget="sparse")
        Hs_a = channel.build_sc_channel_matrices(tr_a.h_b_true, CORE_ROWS)
        for k in range(len(CORE_ROWS)):
            np.testing.assert_array_equal(Hs_a[k], np.eye(N, dtype=np.complex128))
        noise_a = np.full((len(CORE_ROWS), N), obs_a.n0, dtype=np.float64)
        llrs_a, _, _ = channel.oracle_readout_b(tr_a, obs_a.y_b, obs_a.training_rows, obs_a.n0)
        np.testing.assert_array_equal(
            decoder.decode_native_ldpc(llrs_a[list(CORE_ROWS)].reshape(16, 128)),
            tr_a.b_info[:16],
        )
        for k, r in enumerate(CORE_ROWS):
            res = decoder.decode_strong_sc(obs_a.y_b[r], Hs_a[k], noise_a[k], list_size="small", turbo_pic=True)
            np.testing.assert_array_equal(res.info_bits, tr_a.b_info[k])

    def test_valid_codeword_candidates_and_exact_waveform_metric(self):
        seed = SEED_P9_DEV + 707
        obs, tr = channel.simulate_link(seed, snr_db=9.0, regime="continuous", budget="sparse")
        r = CORE_ROWS[3]
        H_r = channel.build_sc_channel_matrix(tr.h_b_true[:, r])
        noise_r = np.full(N, obs.n0, dtype=np.float64)

        # Verify MMSE-PIC with zero prior equals LMMSE identically
        llr_lmmse, est_lmmse, var_lmmse = decoder.lmmse_equalize_row(obs.y_b[r], H_r, noise_r)
        llr_pic0, est_pic0, var_pic0 = decoder.mmse_pic_equalize_row(obs.y_b[r], H_r, noise_r, np.zeros(128))
        np.testing.assert_allclose(est_pic0, est_lmmse, atol=1e-13, rtol=0.0)
        np.testing.assert_allclose(var_pic0, var_lmmse, atol=1e-13, rtol=0.0)
        np.testing.assert_allclose(llr_pic0, llr_lmmse, atol=1e-12, rtol=0.0)

        for ls, expected_min in (("small", 321), ("large", 2641)):
            res = decoder.decode_strong_sc(obs.y_b[r], H_r, noise_r, list_size=ls, turbo_pic=True)
            self.assertGreaterEqual(res.n_candidates, expected_min)
            self.assertTrue(decoder.is_valid_codeword(res.codeword))
            ref_m = decoder.exact_waveform_metric(obs.y_b[r], H_r, res.codeword, noise_r)
            self.assertAlmostEqual(res.metric, ref_m, places=11)

            # Every single candidate in candidates_u64 must satisfy H_parity @ c % 2 == 0
            cands_bits = decoder.unpack_codewords(res.candidates_u64)
            syndromes = (cands_bits @ coding.parity_check().T) % 2
            self.assertFalse(np.any(syndromes))

            # Spot-check that no candidate has strictly lower exact metric than winner
            check_idx = np.linspace(0, len(cands_bits) - 1, 25, dtype=int)
            for idx in check_idx:
                m_i = decoder.exact_waveform_metric(obs.y_b[r], H_r, cands_bits[idx], noise_r)
                self.assertGreaterEqual(m_i, res.metric - 1e-11)

        # Verify turbo_pic=False and arbitrary dense 32x32 channel matrix H_dense
        res_nopic = decoder.decode_strong_sc(obs.y_b[r], H_r, noise_r, list_size="small", turbo_pic=False)
        self.assertEqual(res_nopic.n_candidates, 321)
        self.assertAlmostEqual(
            res_nopic.metric,
            decoder.exact_waveform_metric(obs.y_b[r], H_r, res_nopic.codeword, noise_r),
            places=11,
        )
        rng = np.random.default_rng(910707)
        H_dense = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) / np.sqrt(2.0 * N)
        y_dense = H_dense @ coding.qam(tr.b_code[3]) + np.sqrt(1e-3) * (
            rng.normal(size=N) + 1j * rng.normal(size=N)
        ) / np.sqrt(2.0)
        res_dense = decoder.decode_strong_sc(
            y_dense, H_dense, np.full(N, 1e-3, dtype=np.float64), list_size="small", turbo_pic=True
        )
        self.assertTrue(decoder.is_valid_codeword(res_dense.codeword))
        self.assertAlmostEqual(
            res_dense.metric,
            decoder.exact_waveform_metric(y_dense, H_dense, res_dense.codeword, np.full(N, 1e-3)),
            places=10,
        )

    def test_ml_lower_bound_certificate_and_invalid_rejection(self):
        seed = SEED_P9_DEV + 808
        obs, tr = channel.simulate_link(seed, snr_db=9.0, regime="continuous", budget="sparse")
        r = CORE_ROWS[0]
        H_r = channel.build_sc_channel_matrix(tr.h_b_true[:, r])
        c_true = tr.b_code[0]
        x_true = coding.qam(c_true)
        G = coding.generator()

        # 1. Zero-noise: cannot certify ML error -> undecided
        y_clean = H_r @ x_true
        cert_clean = decoder.ml_lower_bound_certificate(y_clean, H_r, c_true)
        self.assertFalse(cert_clean.ml_must_err)
        self.assertEqual(cert_clean.status, "undecided")
        self.assertEqual(cert_clean.n_single_flip_checked, 64)
        self.assertLess(cert_clean.d_true, 1e-25)

        # 2. Exact midpoint tie with a single-flip neighbor -> tie guard must keep undecided!
        c_nbr = (c_true ^ G[7]).astype(np.uint8)
        y_tie = 0.5 * (H_r @ x_true + H_r @ coding.qam(c_nbr))
        cert_tie = decoder.ml_lower_bound_certificate(y_tie, H_r, c_true)
        self.assertFalse(cert_tie.ml_must_err)
        self.assertEqual(cert_tie.status, "undecided")

        # 3. Received waveform clearly closer to single-flip neighbor -> certified ml_must_err!
        y_nbr = H_r @ coding.qam(c_nbr)
        cert_nbr = decoder.ml_lower_bound_certificate(y_nbr, H_r, c_true)
        self.assertTrue(cert_nbr.ml_must_err)
        self.assertEqual(cert_nbr.status, "ml_must_err")
        self.assertEqual(cert_nbr.witness_source, "single_flip")
        self.assertLess(cert_nbr.d_best_other, cert_nbr.d_true - cert_nbr.eps_tie)
        self.assertTrue(decoder.is_valid_codeword(cert_nbr.witness_codeword))

        # 4. 2-bit-flip valid codeword passed via extra_candidates -> certified via extra_candidate!
        c_two = (c_true ^ G[3] ^ G[11]).astype(np.uint8)
        self.assertTrue(decoder.is_valid_codeword(c_two))
        y_two = H_r @ coding.qam(c_two)
        cert_two = decoder.ml_lower_bound_certificate(y_two, H_r, c_true, extra_candidates=c_two[None, :])
        self.assertTrue(cert_two.ml_must_err)
        self.assertEqual(cert_two.witness_source, "extra_candidate")
        self.assertLess(cert_two.d_best_other, 1e-24)

        # 5. Invalid non-codeword with zero residual MUST be rejected (no false certification!).
        bad_c = c_true.copy()
        bad_c[64] ^= 1  # flip one parity bit -> guaranteed invalid
        self.assertFalse(decoder.is_valid_codeword(bad_c))
        y_bad = H_r @ coding.qam(bad_c)
        cert_bad = decoder.ml_lower_bound_certificate(y_bad, H_r, c_true, extra_candidates=bad_c[None, :])
        self.assertEqual(cert_bad.n_invalid_rejected, 1)
        self.assertEqual(cert_bad.n_extra_valid_checked, 0)
        if cert_bad.witness_codeword is not None:
            self.assertTrue(decoder.is_valid_codeword(cert_bad.witness_codeword))
            self.assertFalse(np.array_equal(cert_bad.witness_codeword, bad_c))

    def test_mini_16_8_exhaustive_ml_certificate_verification(self):
        report = decoder.verify_mini_codebook_ml_certificate(n_trials=240, snr_db=9.0, seed=910999)
        self.assertTrue(report["all_passed"], f"Mini (16,8) verifier failed: {report}")
        self.assertEqual(report["codebook_size"], 256)
        self.assertEqual(report["false_cert_violations"], 0)
        self.assertEqual(report["full_mismatch_violations"], 0)
        self.assertEqual(report["invalid_rejection_passes"], 240)
        self.assertTrue(report["direction_holds"])
        self.assertGreaterEqual(report["d_min"], 4)


if __name__ == "__main__":
    unittest.main()
