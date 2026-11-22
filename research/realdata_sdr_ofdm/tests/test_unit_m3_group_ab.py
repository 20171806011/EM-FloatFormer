"""Unit tests for Milestone 3: Group A (F8) and Group B (F9)."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

import numpy as np

from research.realdata_sdr_ofdm.adapters.modulation import (
    _bit_index_groups,
    bits_to_symbols,
    constellation_points,
)
from research.realdata_sdr_ofdm.adapters.sdr_ofdm_adapter import (
    ACTIVE_END,
    ACTIVE_START,
    N_ACTIVE,
    N_DATA_RES,
    N_FFT,
    N_SYMB,
    PILOT_FFT_INDICES,
    PILOT_SYMBOL_IDX,
    PILOT_VALUES,
    RealOFDMFrame,
    apply_stress_perturbation,
    build_frame,
    build_masks,
)
from research.realdata_sdr_ofdm.receivers import (
    GROUP_A_MODES,
    GROUP_B_MODES,
    KEY_PILOT_INDICES,
    VAL_PILOT_INDICES,
    CandidateBranch,
    GroupAConfig,
    GroupAReferenceReceiver,
    GroupBConfig,
    GroupBFusionReceiver,
    ReceiverOutput,
    build_candidate_bank,
    build_reference_pool,
    calibrate_group_a_on_dev,
    calibrate_group_a_temperature,
    calibrate_group_b_on_dev,
    fuse_candidate_branches,
    run_frame as run_classical_frame,
    run_group_a_frame,
    run_group_b_frame,
)


def _make_channel(seed: int, *, n_taps: int = 4) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    taps = (rng.standard_normal(n_taps) + 1j * rng.standard_normal(n_taps)) / math.sqrt(2.0)
    taps *= np.exp(-0.5 * np.arange(n_taps, dtype=np.float64))
    taps /= math.sqrt(np.sum(np.abs(taps) ** 2))
    h_pad = np.zeros(N_FFT, dtype=np.complex128)
    h_pad[:n_taps] = taps
    return np.fft.fftshift(np.fft.fft(h_pad))


def _make_frame_with_channel(
    seed: int,
    h_freq: np.ndarray,
    *,
    snr_db: float = 18.0,
    split: str = "dev",
    group_id: str = "blk_00",
    frame_id: int = 0,
    gauge_scale: complex = 1.0,
) -> RealOFDMFrame:
    rng = np.random.default_rng(int(seed))
    pilot_mask, _, data_mask = build_masks()
    tx_bits = rng.integers(0, 2, size=(N_DATA_RES, 4), dtype=np.uint8)
    tx_sym = bits_to_symbols(tx_bits, qm=4)

    tx_grid = np.zeros((N_SYMB, N_FFT), dtype=np.complex128)
    tx_grid[data_mask] = tx_sym
    tx_grid[pilot_mask] = PILOT_VALUES

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


class TestM3GroupA(unittest.TestCase):
    def test_contracts_aliases_empty_pool_and_disjoint_pilots(self) -> None:
        self.assertTrue(set(KEY_PILOT_INDICES).isdisjoint(set(VAL_PILOT_INDICES)))
        h_ch = _make_channel(11)
        fr = _make_frame_with_channel(101, h_ch, snr_db=18.0, frame_id=100)
        pool = [_make_frame_with_channel(200 + j, _make_channel(20 + j), snr_db=20.0, frame_id=j) for j in range(6)]

        for mode in (*GROUP_A_MODES, "local_ref", "global_content_match", "content_match"):
            out = run_group_a_frame(fr, context_pool=pool, mode=mode, local_k=2)
            self.assertIsInstance(out, ReceiverOutput)
            self.assertEqual(out.h_est.shape, (N_SYMB, N_ACTIVE))
            self.assertEqual(out.h_var.shape, (N_SYMB, N_ACTIVE))
            self.assertEqual(out.eq_symbols.shape, (N_DATA_RES,))
            self.assertEqual(out.bit_llrs.shape, (N_DATA_RES, 4))
            self.assertTrue(np.all(np.isfinite(out.h_est)))
            self.assertTrue(np.all(out.h_var > 0.0))
            self.assertFalse(out.aux_info["failed"])
            self.assertEqual(len(out.aux_info["weights"]), len(pool))
            self.assertAlmostEqual(sum(out.aux_info["weights"]), 1.0, places=8)
            self.assertTrue(set(out.aux_info["key_pilot_indices"]).isdisjoint(out.aux_info["val_pilot_indices"]))
            if mode in ("local", "local_ref"):
                self.assertEqual(out.aux_info["n_values_accessed"], 2)
                self.assertTrue(all(w == 0.0 for w in out.aux_info["weights"][2:]))
            else:
                self.assertEqual(out.aux_info["n_values_accessed"], len(pool))

        # Empty / None pool fallback
        for empty_pool in (None, []):
            out_e = run_group_a_frame(fr, context_pool=empty_pool, mode="global_content_attn")
            self.assertFalse(out_e.aux_info["failed"])
            self.assertEqual(out_e.aux_info["weights"], [])
            self.assertEqual(out_e.aux_info["n_values_accessed"], 0)

        rx_wrap = GroupAReferenceReceiver(mode="global_content_attn")
        self.assertEqual(rx_wrap.run_frame(fr, context_pool=pool).aux_info["mode"], "global_content_attn")

        with self.assertRaises(ValueError):
            run_group_a_frame(fr, context_pool=pool, mode="bad_mode")

    def test_section51_data_mask_isolation_and_identical_keys_fallback(self) -> None:
        h_q = _make_channel(31)
        fr = _make_frame_with_channel(301, h_q, snr_db=16.0, frame_id=99)
        pool = [_make_frame_with_channel(310 + j, _make_channel(40 + j), snr_db=20.0, frame_id=j) for j in range(6)]

        out_a = run_group_a_frame(fr, context_pool=pool, mode="global_content_attn", temperature=0.15)
        pool_pert = [_scale_frame(p, 1.0) for p in pool]
        pool_pert[3].rx_grid[pool_pert[3].data_mask] *= 100.0
        out_b = run_group_a_frame(fr, context_pool=pool_pert, mode="global_content_attn", temperature=0.15)
        np.testing.assert_allclose(out_a.aux_info["weights"], out_b.aux_info["weights"], atol=1e-12)
        np.testing.assert_allclose(out_a.h_est, out_b.h_est, atol=1e-12)
        np.testing.assert_allclose(out_a.bit_llrs, out_b.bit_llrs, atol=1e-12)

        # Identical pilot observations with varying est_n0 and data_mask must collapse to uniform weights & h_est
        pool_id = []
        for j in range(5):
            pj = _make_frame_with_channel(330 + j, _make_channel(50 + j), snr_db=12.0 + 3.0 * j, frame_id=j)
            pj.rx_grid[pj.pilot_mask] = fr.rx_grid[fr.pilot_mask]
            pj.est_n0 = float(fr.est_n0) * (1.0 + 0.25 * j)
            pool_id.append(pj)
        o_attn = run_group_a_frame(fr, context_pool=pool_id, mode="global_content_attn", temperature=0.1)
        o_unif = run_group_a_frame(fr, context_pool=pool_id, mode="global_uniform")
        np.testing.assert_allclose(o_attn.aux_info["weights"], [0.2] * 5, atol=1e-8)
        np.testing.assert_allclose(o_attn.h_est, o_unif.h_est, atol=1e-8)

    def test_content_matching_and_reliability_superiority(self) -> None:
        h_target = _make_channel(77)
        fr = _make_frame_with_channel(401, h_target, snr_db=14.0, frame_id=99)
        pool = []
        for j in range(8):
            if j == 5:
                pool.append(_make_frame_with_channel(410 + j, h_target, snr_db=24.0, frame_id=j, gauge_scale=np.exp(1.2j)))
            else:
                pool.append(_make_frame_with_channel(410 + j, _make_channel(100 + j), snr_db=22.0, frame_id=j))

        o_loc = run_group_a_frame(fr, context_pool=pool, mode="local", local_k=2)
        o_unif = run_group_a_frame(fr, context_pool=pool, mode="global_uniform")
        o_attn = run_group_a_frame(fr, context_pool=pool, mode="global_content_attn", temperature=0.15)
        self.assertGreater(o_attn.aux_info["weights"][5], 0.4)

        h_true_act = h_target[ACTIVE_START:ACTIVE_END]
        mse_loc = float(np.mean(np.abs(o_loc.h_est[PILOT_SYMBOL_IDX] - h_true_act) ** 2))
        mse_unif = float(np.mean(np.abs(o_unif.h_est[PILOT_SYMBOL_IDX] - h_true_act) ** 2))
        mse_attn = float(np.mean(np.abs(o_attn.h_est[PILOT_SYMBOL_IDX] - h_true_act) ** 2))
        self.assertLess(mse_attn, mse_loc)
        self.assertLess(mse_attn, mse_unif)

        # Reliability test: m=0,1 at 28 dB vs m=2..5 at 0 dB
        pool_rel = [
            _make_frame_with_channel(450 + j, h_target, snr_db=(28.0 if j < 2 else 0.0), frame_id=j)
            for j in range(6)
        ]
        o_rel = run_group_a_frame(fr, context_pool=pool_rel, mode="global_reliability")
        o_rel_unif = run_group_a_frame(fr, context_pool=pool_rel, mode="global_uniform")
        self.assertGreater(float(np.mean(o_rel.aux_info["weights"][:2])), 1.0 / 6.0)
        mse_rel = float(np.mean(np.abs(o_rel.h_est[PILOT_SYMBOL_IDX] - h_true_act) ** 2))
        mse_rel_unif = float(np.mean(np.abs(o_rel_unif.h_est[PILOT_SYMBOL_IDX] - h_true_act) ** 2))
        self.assertLess(mse_rel, mse_rel_unif)

    def test_temperature_extremes_and_antileakage_firewall(self) -> None:
        h_t = _make_channel(88)
        fr = _make_frame_with_channel(501, h_t, snr_db=16.0, split="dev", frame_id=0)
        pool = [
            _make_frame_with_channel(510 + j, h_t if j == 2 else _make_channel(200 + j), snr_db=20.0, split="train", frame_id=10 + j)
            for j in range(5)
        ]
        o_hot = run_group_a_frame(fr, context_pool=pool, mode="global_content_attn", temperature=1e6)
        o_cold = run_group_a_frame(fr, context_pool=pool, mode="global_content_attn", temperature=1e-4)
        np.testing.assert_allclose(o_hot.aux_info["weights"], [0.2] * 5, atol=1e-3)
        self.assertGreater(max(o_cold.aux_info["weights"]), 0.99)

        # Firewall rejects test/guard in pool builder & temperature calibration
        for bad_sp in ("test", "TEST", "guard", " Guard "):
            bad_fr = _make_frame_with_channel(550, h_t, split=bad_sp, frame_id=99)
            with self.assertRaises(ValueError):
                build_reference_pool([bad_fr])
            with self.assertRaises(ValueError):
                calibrate_group_a_temperature([bad_fr], pool)
            with self.assertRaises(ValueError):
                calibrate_group_a_temperature([fr], [bad_fr])

        # Accepts both flat pool and per-frame pools
        tau_flat = calibrate_group_a_temperature([fr], pool, temps=(0.1, 0.5))
        tau_per = calibrate_group_a_temperature([fr], [pool], temps=(0.1, 0.5))
        self.assertGreater(tau_flat, 0.0)
        self.assertGreater(tau_per, 0.0)

        cfg_a_dev, det_a_dev = calibrate_group_a_on_dev(
            [fr], pool, temps=(0.1, 0.5), prior_scales=(0.75, 1.5)
        )
        self.assertEqual(len(det_a_dev["grid_records"]), 4)
        self.assertIn(cfg_a_dev.temperature, (0.1, 0.5))
        self.assertIn(cfg_a_dev.prior_scale, (0.75, 1.5))
        with self.assertRaises(ValueError):
            calibrate_group_a_on_dev([_make_frame_with_channel(551, h_t, split="test", frame_id=98)], pool)

        # Stratified pool size cap
        train_many = [
            _make_frame_with_channel(600 + i, h_t, split="train", group_id=f"g{i % 3}", frame_id=i)
            for i in range(15)
        ]
        sub_pool = build_reference_pool(train_many, max_pool_size=6)
        self.assertEqual(len(sub_pool), 6)
        self.assertEqual(len({f.group_id for f in sub_pool}), 3)

    def test_24_octave_cstar_extreme_noise_no_leak_and_heldout_masking(self) -> None:
        h_t = _make_channel(91)
        fr = _make_frame_with_channel(701, h_t, snr_db=18.0, frame_id=1)
        fr.meta["heldout_pilot_indices"] = [1, 5, 9, 12]
        pool = [_make_frame_with_channel(710 + j, _make_channel(300 + j), snr_db=20.0, frame_id=10 + j) for j in range(4)]

        ref = run_group_a_frame(fr, context_pool=pool, mode="global_content_attn")
        for c in ((2.0 ** -12) * np.exp(0.7j), (2.0 ** 12) * np.exp(-0.9j)):
            fr_s = _scale_frame(fr, c)
            pool_s = [_scale_frame(p, c) for p in pool]
            out_s = run_group_a_frame(fr_s, context_pool=pool_s, mode="global_content_attn")
            np.testing.assert_allclose(out_s.aux_info["weights"], ref.aux_info["weights"], atol=1e-8)
            np.testing.assert_allclose(out_s.h_est, c * ref.h_est, rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(out_s.h_var, (abs(c) ** 2) * ref.h_var, rtol=1e-4, atol=1e-12)
            np.testing.assert_allclose(out_s.bit_llrs, ref.bit_llrs, atol=1e-4)
            self.assertEqual(int(np.sum(out_s.hard_bits != ref.hard_bits)), 0)

        # Extreme noise est_n0 = 1e4
        fr_ext = _scale_frame(fr, 1.0)
        fr_ext.est_n0 = 1e4
        for m in GROUP_A_MODES:
            o_ext = run_group_a_frame(fr_ext, context_pool=pool, mode=m)
            self.assertTrue(np.all(np.isfinite(o_ext.bit_llrs)))
            self.assertTrue(np.all(o_ext.h_var > 0.0))
            self.assertLess(float(np.mean(np.abs(o_ext.bit_llrs))), 1.0)

        # Held-out pilot CV masking (strict literal 0.0 diff on both pred and var)
        fr_ho = _scale_frame(fr, 1.0)
        for idx in (1, 5, 9, 12):
            fr_ho.rx_grid[PILOT_SYMBOL_IDX, PILOT_FFT_INDICES[idx]] += 40.0 - 15.0j
        out_ho = run_group_a_frame(fr_ho, context_pool=pool, mode="global_content_attn")
        np.testing.assert_array_equal(out_ho.heldout_pilot_pred, ref.heldout_pilot_pred)
        np.testing.assert_array_equal(out_ho.heldout_pilot_var, ref.heldout_pilot_var)

        # Zero label leakage
        fr_pois = _scale_frame(fr, 1.0)
        fr_pois.tx_bits[:] = 1 - fr_pois.tx_bits
        fr_pois.tx_symbols[:] = -fr_pois.tx_symbols
        out_pois = run_group_a_frame(fr_pois, context_pool=pool, mode="global_content_attn")
        np.testing.assert_array_equal(out_pois.h_est, ref.h_est)
        np.testing.assert_array_equal(out_pois.bit_llrs, ref.bit_llrs)

    def test_heldout_pilot_cv_exact_zero_leakage_across_scales_and_modes(self) -> None:
        h_t = _make_channel(93)
        base_fr = _make_frame_with_channel(721, h_t, snr_db=18.0, frame_id=1)
        base_pool = [_make_frame_with_channel(730 + j, _make_channel(320 + j), snr_db=20.0, frame_id=10 + j) for j in range(4)]

        for c in (1.0, 2.0 ** -4):
            fr = _scale_frame(base_fr, c)
            fr.meta["heldout_pilot_indices"] = [1, 5, 9, 12]
            pool = [_scale_frame(p, c) for p in base_pool]
            for mut in (1e4 + 1e4j, 1e6 + 1e6j):
                fr_mut = _scale_frame(fr, 1.0)
                for idx in (1, 5, 9, 12):
                    fr_mut.rx_grid[PILOT_SYMBOL_IDX, PILOT_FFT_INDICES[idx]] += mut
                for mode in GROUP_A_MODES:
                    o_ref = run_group_a_frame(fr, context_pool=pool, mode=mode)
                    o_mut = run_group_a_frame(fr_mut, context_pool=pool, mode=mode)
                    self.assertFalse(o_ref.aux_info["failed"])
                    self.assertFalse(o_mut.aux_info["failed"])
                    self.assertEqual(float(np.max(np.abs(o_ref.heldout_pilot_pred - o_mut.heldout_pilot_pred))), 0.0)
                    self.assertEqual(float(np.max(np.abs(o_ref.heldout_pilot_var - o_mut.heldout_pilot_var))), 0.0)

    def test_heldout_pilot_cv_multifold_zero_rms_and_calib_parity(self) -> None:
        h_t = _make_channel(95)
        pool = [_make_frame_with_channel(750 + j, _make_channel(340 + j), snr_db=20.0, split="train", frame_id=20 + j) for j in range(4)]

        # 1) Multi-fold default CV (no explicit heldout_pilot_indices -> 4-fold CV; fold 0 holds out [1, 5, 9])
        fr_mf = _make_frame_with_channel(741, h_t, snr_db=18.0, split="dev", frame_id=2)
        fr_mf.meta.pop("heldout_pilot_indices", None)
        fr_mf_mut = _scale_frame(fr_mf, 1.0)
        for idx in (1, 5, 9):
            fr_mf_mut.rx_grid[PILOT_SYMBOL_IDX, PILOT_FFT_INDICES[idx]] += 1e6 + 1e6j
        for mode in GROUP_A_MODES:
            o_ref = run_group_a_frame(fr_mf, context_pool=pool, mode=mode)
            o_mut = run_group_a_frame(fr_mf_mut, context_pool=pool, mode=mode)
            eval_idx = list(o_ref.aux_info["heldout_pilot_indices"])
            pos = [eval_idx.index(pi) for pi in (1, 5, 9)]
            self.assertEqual(float(np.max(np.abs(o_ref.heldout_pilot_pred[pos] - o_mut.heldout_pilot_pred[pos]))), 0.0)
            self.assertEqual(float(np.max(np.abs(o_ref.heldout_pilot_var[pos] - o_mut.heldout_pilot_var[pos]))), 0.0)

        # 2) Zero-RMS active fallback
        for n0_val in (0.0, 0.25):
            fr_z = _scale_frame(fr_mf, 1.0)
            fr_z.rx_grid[:] = 0.0
            fr_z.est_n0 = n0_val
            fr_z.meta["heldout_pilot_indices"] = [1, 5, 9, 12]
            fr_z_mut = _scale_frame(fr_z, 1.0)
            for idx in (1, 5, 9, 12):
                fr_z_mut.rx_grid[PILOT_SYMBOL_IDX, PILOT_FFT_INDICES[idx]] += 1e6 + 1e6j
            for mode in GROUP_A_MODES:
                oz_ref = run_group_a_frame(fr_z, context_pool=pool, mode=mode)
                oz_mut = run_group_a_frame(fr_z_mut, context_pool=pool, mode=mode)
                self.assertFalse(oz_ref.aux_info["failed"])
                self.assertFalse(oz_mut.aux_info["failed"])
                self.assertTrue(np.all(np.isfinite(oz_ref.heldout_pilot_pred)))
                self.assertTrue(np.all(oz_ref.heldout_pilot_var > 0.0))
                self.assertEqual(float(np.max(np.abs(oz_ref.heldout_pilot_pred - oz_mut.heldout_pilot_pred))), 0.0)
                self.assertEqual(float(np.max(np.abs(oz_ref.heldout_pilot_var - oz_mut.heldout_pilot_var))), 0.0)

        # 3) calibrate_group_a_on_dev fast-path vs slow-path parity + held-out mutation invariance
        dev_frames = []
        dev_mut = []
        for k in range(2):
            dfr = _make_frame_with_channel(760 + k, _make_channel(360 + k), snr_db=16.0 + 2.0 * k, split="dev", frame_id=30 + k)
            dfr.meta["heldout_pilot_indices"] = [1, 5, 9, 12]
            dfr_m = _scale_frame(dfr, 1.0)
            for idx in (1, 5, 9, 12):
                dfr_m.rx_grid[PILOT_SYMBOL_IDX, PILOT_FFT_INDICES[idx]] += 1e6 + 1e6j
            dev_frames.append(dfr)
            dev_mut.append(dfr_m)

        base_cfg = GroupAConfig(phase_track=False)
        _, det_clean = calibrate_group_a_on_dev(
            dev_frames, pool, temps=(0.1, 0.5), prior_scales=(0.75, 1.5), base_config=base_cfg
        )
        _, det_mut = calibrate_group_a_on_dev(
            dev_mut, pool, temps=(0.1, 0.5), prior_scales=(0.75, 1.5), base_config=base_cfg
        )
        for rec_clean, rec_mut in zip(det_clean["grid_records"], det_mut["grid_records"]):
            cand_cfg = GroupAConfig(
                temperature=rec_clean["temperature"],
                prior_scale=rec_clean["prior_scale"],
                phase_track=False,
            )
            for frames_case, rec in ((dev_frames, rec_clean), (dev_mut, rec_mut)):
                bit_err, tot_bits, sq_sym, tot_sym = 0, 0, 0.0, 0
                nlls = []
                for dfr in frames_case:
                    out = run_group_a_frame(dfr, context_pool=pool, mode="global_content_attn", config=cand_cfg)
                    bit_err += int(np.sum(out.hard_bits != dfr.tx_bits))
                    tot_bits += int(dfr.tx_bits.size)
                    sq_sym += float(np.sum(np.abs(out.eq_symbols - dfr.tx_symbols) ** 2))
                    tot_sym += int(dfr.tx_symbols.size)
                    eval_idx = out.aux_info["heldout_pilot_indices"]
                    fft_idx = [PILOT_FFT_INDICES[i] for i in eval_idx]
                    p_val = dfr.pilot_values[eval_idx]
                    z_obs = dfr.rx_grid[PILOT_SYMBOL_IDX, fft_idx] / p_val
                    sig2_obs = float(dfr.est_n0) / (np.abs(p_val) ** 2)
                    v_eff = np.maximum(out.heldout_pilot_var + sig2_obs, 1e-30)
                    nlls.append(float(np.mean(np.log(np.pi * v_eff) + (np.abs(out.heldout_pilot_pred - z_obs) ** 2) / v_eff)))
                self.assertAlmostEqual(rec["ber"], bit_err / tot_bits, delta=1e-12)
                self.assertAlmostEqual(rec["sym_mse"], sq_sym / tot_sym, delta=1e-12)
                self.assertAlmostEqual(rec["pilot_nll"], float(np.mean(nlls)), delta=1e-12)


class TestM3GroupB(unittest.TestCase):
    def test_contracts_single_branch_identity_with_classical_and_aliases(self) -> None:
        h_ch = _make_channel(105)
        fr = _make_frame_with_channel(801, h_ch, snr_db=18.0, frame_id=0)
        o_cls = run_classical_frame(fr, method="lmmse_2d_tracked")

        for mode in (*GROUP_B_MODES, "weighted_plugin", "mixture_likelihood"):
            o_b = fuse_candidate_branches(
                fr,
                branches=[(o_cls.h_est, o_cls.h_var)],
                branch_weights=[1.0],
                mode=mode,
            )
            np.testing.assert_allclose(o_b.h_est, o_cls.h_est, atol=1e-6)
            np.testing.assert_allclose(o_b.bit_llrs, o_cls.bit_llrs, atol=1e-11)
            self.assertEqual(int(np.sum(o_b.hard_bits != o_cls.hard_bits)), 0)

        # Internal bank run_frame & wrapper
        for mode in GROUP_B_MODES:
            out = run_group_b_frame(fr, mode=mode)
            self.assertFalse(out.aux_info["failed"])
            self.assertGreaterEqual(out.aux_info["n_candidates"], 2)
            self.assertAlmostEqual(sum(out.aux_info["branch_weights"]), 1.0, places=8)
            self.assertTrue(np.all(out.h_var > 0.0))

        rx_b = GroupBFusionReceiver(mode="exact_mixture_likelihood")
        self.assertEqual(rx_b.run_frame(fr).aux_info["mode"], "exact_mixture_likelihood")

        with self.assertRaises(ValueError):
            run_group_b_frame(fr, mode="unknown_mode")

    def test_total_variance_zero_weight_skip_and_no_llr_avg_counterexample(self) -> None:
        h_ch = _make_channel(109)
        fr = _make_frame_with_channel(820, h_ch, snr_db=18.0, frame_id=1)

        m0 = np.full((N_SYMB, N_ACTIVE), 1.0 + 0.0j, dtype=np.complex128)
        m1 = np.full((N_SYMB, N_ACTIVE), 0.0 + 1.0j, dtype=np.complex128)
        m_bad = np.full((N_SYMB, N_ACTIVE), 1e6 + 1e6j, dtype=np.complex128)
        v0 = np.full((N_SYMB, N_ACTIVE), 1e-5, dtype=np.float64)
        v1 = np.full((N_SYMB, N_ACTIVE), 1e-5, dtype=np.float64)
        v_bad = np.full((N_SYMB, N_ACTIVE), 1e8, dtype=np.float64)

        # T1-F9.3 Total Variance + T2-F9.1 alpha=0 skip
        o_plug = fuse_candidate_branches(
            fr,
            branches=[(m0, v0), (m1, v1), (m_bad, v_bad)],
            branch_weights=[0.8, 0.2, 0.0],
            mode="posterior_mean_plugin",
        )
        m_expected = 0.8 * m0 + 0.2 * m1
        v_expected = 0.8 * (v0 + np.abs(m0 - m_expected) ** 2) + 0.2 * (v1 + np.abs(m1 - m_expected) ** 2)
        np.testing.assert_allclose(o_plug.h_est, m_expected, atol=1e-10)
        np.testing.assert_allclose(o_plug.h_var, v_expected, atol=1e-10)

        # Duplicate branch invariance [0.5, 0.5] == [1.0]
        o_dup = fuse_candidate_branches(fr, branches=[(m0, v0), (m0, v0)], branch_weights=[0.5, 0.5])
        o_one = fuse_candidate_branches(fr, branches=[(m0, v0)], branch_weights=[1.0])
        np.testing.assert_allclose(o_dup.bit_llrs, o_one.bit_llrs, atol=1e-8)

        # T2-F9.2 Analytic LLR-averaging counterexample
        fr_ce = _scale_frame(fr, 1.0)
        fr_ce.rx_grid[fr_ce.data_mask] = -1.2 - 0.35j
        fr_ce.est_n0 = 0.05
        o_mix = fuse_candidate_branches(
            fr_ce,
            branches=[(m0, v0), (m1, v1)],
            branch_weights=[0.8, 0.2],
            mode="exact_mixture_likelihood",
        )
        pts = constellation_points(4)
        idx1, idx0 = _bit_index_groups(4)
        var_eff = 0.05 + 1e-5
        y_val = -1.2 - 0.35j
        g0 = -math.log(math.pi * var_eff) - (np.abs(y_val - 1.0 * pts) ** 2) / var_eff
        g1 = -math.log(math.pi * var_eff) - (np.abs(y_val - 1j * pts) ** 2) / var_eff
        l_mix = np.logaddexp(math.log(0.8) + g0, math.log(0.2) + g1)
        ref_llr_2 = float(
            np.logaddexp.reduce(l_mix[idx1[2]]) - np.logaddexp.reduce(l_mix[idx0[2]])
        )
        np.testing.assert_allclose(o_mix.bit_llrs[:, 2], ref_llr_2, atol=1e-10)
        self.assertTrue(np.all(o_mix.bit_llrs[:, 2] < -1.0))
        self.assertTrue(np.all(o_mix.hard_bits[:, 2] == 0))

    def test_24_octave_cstar_extreme_noise_determinism_and_firewall(self) -> None:
        h_ch = _make_channel(115)
        fr = apply_stress_perturbation(
            _make_frame_with_channel(850, h_ch, snr_db=20.0, split="dev", frame_id=3),
            cfo_norm=0.04,
            phase_drift_rad=0.4,
            seed=850,
        )
        ref = run_group_b_frame(fr, mode="exact_mixture_likelihood")
        for c in ((2.0 ** -12) * np.exp(0.6j), (2.0 ** 12) * np.exp(-1.1j)):
            sc = run_group_b_frame(_scale_frame(fr, c), mode="exact_mixture_likelihood")
            np.testing.assert_allclose(sc.h_est, c * ref.h_est, rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(sc.h_var, (abs(c) ** 2) * ref.h_var, rtol=1e-4, atol=1e-12)
            np.testing.assert_allclose(sc.bit_llrs, ref.bit_llrs, atol=1e-4)
            self.assertEqual(int(np.sum(sc.hard_bits != ref.hard_bits)), 0)

        # Deterministic exact replay & zero label leakage
        fr_pois = _scale_frame(fr, 1.0)
        fr_pois.tx_bits[:] = 1 - fr_pois.tx_bits
        fr_pois.tx_symbols[:] = -fr_pois.tx_symbols
        rep = run_group_b_frame(fr_pois, mode="exact_mixture_likelihood")
        np.testing.assert_array_equal(rep.h_est, ref.h_est)
        np.testing.assert_array_equal(rep.h_var, ref.h_var)
        np.testing.assert_array_equal(rep.bit_llrs, ref.bit_llrs)

        # Extreme noise & NaN fallback
        for n0_val in (1e-6, 1e3):
            fr_n = _scale_frame(fr, 1.0)
            fr_n.est_n0 = n0_val
            for mode in GROUP_B_MODES:
                o_n = run_group_b_frame(fr_n, mode=mode)
                self.assertFalse(o_n.aux_info["failed"])
                self.assertTrue(np.all(np.isfinite(o_n.bit_llrs)))

        fr_nan = _scale_frame(fr, 1.0)
        fr_nan.rx_grid[0, 20] = complex(float("nan"), 0.0)
        o_nan = run_group_b_frame(fr_nan)
        self.assertTrue(o_nan.aux_info["failed"])
        self.assertTrue(np.all(np.isfinite(o_nan.bit_llrs)))

        # CandidateBank & dev firewall
        branches, alpha = build_candidate_bank(fr)
        self.assertGreaterEqual(len(branches), 3)
        self.assertAlmostEqual(float(np.sum(alpha)), 1.0, places=8)
        for bad_sp in ("test", "guard"):
            bad_fr = _make_frame_with_channel(860, h_ch, split=bad_sp, frame_id=9)
            with self.assertRaises(ValueError):
                calibrate_group_b_on_dev([bad_fr])

    def test_heldout_pilot_cv_exact_zero_leakage_across_scales_and_modes(self) -> None:
        h_ch = _make_channel(117)
        base_fr = _make_frame_with_channel(871, h_ch, snr_db=18.0, split="test", frame_id=4)

        for c in (1.0, 2.0 ** -4):
            fr = _scale_frame(base_fr, c)
            fr.meta["heldout_pilot_indices"] = [1, 5, 9, 12]
            for mut in (1e4 + 1e4j, 1e6 + 1e6j):
                fr_mut = _scale_frame(fr, 1.0)
                for idx in (1, 5, 9, 12):
                    fr_mut.rx_grid[PILOT_SYMBOL_IDX, PILOT_FFT_INDICES[idx]] += mut
                for mode in GROUP_B_MODES:
                    o_ref = run_group_b_frame(fr, mode=mode)
                    o_mut = run_group_b_frame(fr_mut, mode=mode)
                    self.assertFalse(o_ref.aux_info["failed"])
                    self.assertFalse(o_mut.aux_info["failed"])
                    self.assertEqual(float(np.max(np.abs(o_ref.heldout_pilot_pred - o_mut.heldout_pilot_pred))), 0.0)
                    self.assertEqual(float(np.max(np.abs(o_ref.heldout_pilot_var - o_mut.heldout_pilot_var))), 0.0)

    def test_heldout_pilot_cv_multifold_zero_rms_and_calib_parity(self) -> None:
        h_ch = _make_channel(119)
        fr_mf = _make_frame_with_channel(881, h_ch, snr_db=18.0, split="dev", frame_id=5)
        fr_mf.meta.pop("heldout_pilot_indices", None)
        fr_mf_mut = _scale_frame(fr_mf, 1.0)
        for idx in (1, 5, 9):
            fr_mf_mut.rx_grid[PILOT_SYMBOL_IDX, PILOT_FFT_INDICES[idx]] += 1e6 + 1e6j
        for mode in GROUP_B_MODES:
            o_ref = run_group_b_frame(fr_mf, mode=mode)
            o_mut = run_group_b_frame(fr_mf_mut, mode=mode)
            eval_idx = list(o_ref.aux_info["heldout_pilot_indices"])
            pos = [eval_idx.index(pi) for pi in (1, 5, 9)]
            self.assertEqual(float(np.max(np.abs(o_ref.heldout_pilot_pred[pos] - o_mut.heldout_pilot_pred[pos]))), 0.0)
            self.assertEqual(float(np.max(np.abs(o_ref.heldout_pilot_var[pos] - o_mut.heldout_pilot_var[pos]))), 0.0)

        # Zero-RMS active fallback
        for n0_val in (0.0, 0.25):
            fr_z = _scale_frame(fr_mf, 1.0)
            fr_z.rx_grid[:] = 0.0
            fr_z.est_n0 = n0_val
            fr_z.meta["heldout_pilot_indices"] = [1, 5, 9, 12]
            fr_z_mut = _scale_frame(fr_z, 1.0)
            for idx in (1, 5, 9, 12):
                fr_z_mut.rx_grid[PILOT_SYMBOL_IDX, PILOT_FFT_INDICES[idx]] += 1e6 + 1e6j
            for mode in GROUP_B_MODES:
                oz_ref = run_group_b_frame(fr_z, mode=mode)
                oz_mut = run_group_b_frame(fr_z_mut, mode=mode)
                self.assertFalse(oz_ref.aux_info["failed"])
                self.assertFalse(oz_mut.aux_info["failed"])
                self.assertTrue(np.all(np.isfinite(oz_ref.heldout_pilot_pred)))
                self.assertTrue(np.all(oz_ref.heldout_pilot_var > 0.0))
                self.assertEqual(float(np.max(np.abs(oz_ref.heldout_pilot_pred - oz_mut.heldout_pilot_pred))), 0.0)
                self.assertEqual(float(np.max(np.abs(oz_ref.heldout_pilot_var - oz_mut.heldout_pilot_var))), 0.0)

        # calibrate_group_b_on_dev fast-path vs slow-path parity + held-out mutation invariance
        dev_frames = []
        dev_mut = []
        for k in range(2):
            dfr = _make_frame_with_channel(890 + k, _make_channel(125 + k), snr_db=16.0 + 2.0 * k, split="dev", frame_id=40 + k)
            dfr.meta["heldout_pilot_indices"] = [1, 5, 9, 12]
            dfr_m = _scale_frame(dfr, 1.0)
            for idx in (1, 5, 9, 12):
                dfr_m.rx_grid[PILOT_SYMBOL_IDX, PILOT_FFT_INDICES[idx]] += 1e6 + 1e6j
            dev_frames.append(dfr)
            dev_mut.append(dfr_m)

        _, det_clean = calibrate_group_b_on_dev(
            dev_frames, betas=(0.25, 0.5), nonzero_om_penalties=(2.0, 8.0), n0_scales=(0.7,)
        )
        _, det_mut = calibrate_group_b_on_dev(
            dev_mut, betas=(0.25, 0.5), nonzero_om_penalties=(2.0, 8.0), n0_scales=(0.7,)
        )
        for rec_clean, rec_mut in zip(det_clean["grid_records"], det_mut["grid_records"]):
            cand_cfg = GroupBConfig(
                beta=rec_clean["beta"],
                nonzero_om_penalty=rec_clean["nonzero_om_penalty"],
                n0_scale=rec_clean["n0_scale"],
            )
            for frames_case, rec in ((dev_frames, rec_clean), (dev_mut, rec_mut)):
                bit_err, tot_bits, sq_sym, tot_sym = 0, 0, 0.0, 0
                nlls = []
                for dfr in frames_case:
                    out = run_group_b_frame(dfr, mode="exact_mixture_likelihood", config=cand_cfg)
                    bit_err += int(np.sum(out.hard_bits != dfr.tx_bits))
                    tot_bits += int(dfr.tx_bits.size)
                    sq_sym += float(np.sum(np.abs(out.eq_symbols - dfr.tx_symbols) ** 2))
                    tot_sym += int(dfr.tx_symbols.size)
                    eval_idx = out.aux_info["heldout_pilot_indices"]
                    fft_idx = [PILOT_FFT_INDICES[i] for i in eval_idx]
                    p_val = dfr.pilot_values[eval_idx]
                    z_obs = dfr.rx_grid[PILOT_SYMBOL_IDX, fft_idx] / p_val
                    sig2_obs = float(dfr.est_n0) / (np.abs(p_val) ** 2)
                    v_eff = np.maximum(out.heldout_pilot_var + sig2_obs, 1e-30)
                    nlls.append(float(np.mean(np.log(np.pi * v_eff) + (np.abs(out.heldout_pilot_pred - z_obs) ** 2) / v_eff)))
                self.assertAlmostEqual(rec["ber"], bit_err / tot_bits, delta=1e-12)
                self.assertAlmostEqual(rec["sym_mse"], sq_sym / tot_sym, delta=1e-12)
                self.assertAlmostEqual(rec["pilot_nll"], float(np.mean(nlls)), delta=1e-12)


if __name__ == "__main__":
    unittest.main()
