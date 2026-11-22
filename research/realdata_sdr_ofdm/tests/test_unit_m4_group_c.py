"""Unit tests for Milestone 4: Group C lightweight C*-equivariant attention (F10)."""

from __future__ import annotations

import copy
import math
import sys
import tempfile
import unittest
from pathlib import Path

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
    QM,
    RealOFDMFrame,
    apply_stress_perturbation,
    build_frame,
    build_masks,
)
from research.realdata_sdr_ofdm.receivers import (
    BranchWeightAttentionNet,
    GroupCConfig,
    GroupCReceiver,
    GroupCTrainableReceiver,
    fuse_candidate_branches,
    run_group_b_frame,
    run_group_c_frame,
)
from research.realdata_sdr_ofdm.receivers.group_c_trainable import (
    _compute_log_g_torch,
    differentiable_mixture_llrs_torch,
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
    cfo_rad: float = 0.0,
) -> RealOFDMFrame:
    rng = np.random.default_rng(int(seed))
    pilot_mask, _, data_mask = build_masks()
    tx_bits = rng.integers(0, 2, size=(N_DATA_RES, 4), dtype=np.uint8)
    tx_sym = bits_to_symbols(tx_bits, qm=4)

    tx_grid = np.zeros((N_SYMB, N_FFT), dtype=np.complex128)
    tx_grid[data_mask] = tx_sym
    tx_grid[pilot_mask] = PILOT_VALUES

    sym_rot = np.exp(1j * float(cfo_rad) * (np.arange(N_SYMB, dtype=np.float64) - float(PILOT_SYMBOL_IDX)))
    h_2d = h_freq[None, :] * sym_rot[:, None]

    n0 = 10.0 ** (-float(snr_db) / 10.0)
    noise = math.sqrt(n0 / 2.0) * (
        rng.standard_normal((N_SYMB, N_FFT)) + 1j * rng.standard_normal((N_SYMB, N_FFT))
    )
    rx_raw = h_2d * tx_grid + noise
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


class TestM4GroupC(unittest.TestCase):
    def test_exports_param_count_and_no_p9_p11_coupling(self) -> None:
        self.assertIs(GroupCReceiver, GroupCTrainableReceiver)
        self.assertNotIn("p9.neural_gate", sys.modules)
        self.assertNotIn("p11.manifold_gate", sys.modules)
        net = BranchWeightAttentionNet(d_c=8, d_s=12, n_heads=2)
        p_cnt = net.real_parameters()
        self.assertGreaterEqual(p_cnt, 1500)
        self.assertLessEqual(p_cnt, 4000)

    def test_step0_exact_parity_with_group_b(self) -> None:
        h = _make_channel(101)
        fr = _make_frame_with_channel(102, h, snr_db=16.0, split="test", cfo_rad=0.05)
        cfg = GroupCConfig(include_global_branch=False)
        model = GroupCTrainableReceiver(config=cfg, seed=20261121)

        out_c = model.run_frame(fr)
        out_b = run_group_b_frame(fr, mode="exact_mixture_likelihood", config=cfg.group_b_cfg)

        np.testing.assert_allclose(
            out_c.aux_info["branch_weights"],
            out_b.aux_info["branch_weights"],
            atol=1e-12,
        )
        np.testing.assert_allclose(out_c.h_est, out_b.h_est, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(out_c.bit_llrs, out_b.bit_llrs, atol=1e-12)
        self.assertEqual(int(np.sum(out_c.hard_bits != out_b.hard_bits)), 0)

    def test_differentiable_pytorch_vs_numpy_mixture_llr_parity(self) -> None:
        import torch

        h = _make_channel(201)
        fr = _make_frame_with_channel(202, h, snr_db=14.0, split="dev")
        data_mask_act = np.asarray(fr.data_mask[:, ACTIVE_START:ACTIVE_END], dtype=bool)
        rx_act = fr.rx_grid[:, ACTIVE_START:ACTIVE_END]
        y_data = rx_act[data_mask_act]

        rng = np.random.default_rng(203)
        branches = []
        for _ in range(4):
            mh = h[None, ACTIVE_START:ACTIVE_END] + 0.05 * (
                rng.standard_normal((N_SYMB, N_ACTIVE)) + 1j * rng.standard_normal((N_SYMB, N_ACTIVE))
            )
            vh = rng.uniform(0.005, 0.05, size=(N_SYMB, N_ACTIVE))
            branches.append((mh, vh))
        alpha = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
        n0_scale = 0.7

        out_np = fuse_candidate_branches(
            fr,
            branches=branches,
            branch_weights=alpha,
            mode="exact_mixture_likelihood",
            include_constellation_var=False,
            n0_scale=n0_scale,
            return_heldout_cv=False,
        )

        # Evaluate PyTorch mixture LLRs in same s_scale gauge
        active_mask = np.asarray(fr.pilot_mask | fr.data_mask, dtype=bool)
        s_scale = float(np.sqrt(np.mean(np.abs(fr.rx_grid[active_mask]) ** 2)))
        y_norm = y_data / s_scale
        m_norm = np.stack([b[0][data_mask_act] / s_scale for b in branches], axis=0)
        v_norm = np.stack([b[1][data_mask_act] / (s_scale ** 2) for b in branches], axis=0)
        n0_cal = n0_scale * float(fr.est_n0) / (s_scale ** 2)

        pts_t = torch.from_numpy(np.array(constellation_points(QM), copy=True)).to(torch.complex128)
        idx1_np, idx0_np = _bit_index_groups(QM)
        log_g = _compute_log_g_torch(
            torch.from_numpy(y_norm).unsqueeze(0),
            torch.from_numpy(m_norm).unsqueeze(0),
            torch.from_numpy(v_norm).unsqueeze(0),
            torch.tensor([n0_cal], dtype=torch.float64),
            pts_t,
        )
        llrs_torch = differentiable_mixture_llrs_torch(
            torch.from_numpy(alpha).unsqueeze(0),
            log_g,
            torch.from_numpy(idx1_np).to(torch.long),
            torch.from_numpy(idx0_np).to(torch.long),
        ).squeeze(0).numpy()

        max_diff = float(np.max(np.abs(llrs_torch - out_np.bit_llrs)))
        self.assertLess(max_diff, 1e-12)

    def test_24_octave_and_1e15_cstar_equivariance_both_modes(self) -> None:
        h = _make_channel(301)
        tr = [_make_frame_with_channel(310 + i, h, snr_db=16.0, split="train", frame_id=i) for i in range(4)]
        dv = [_make_frame_with_channel(320 + i, h, snr_db=16.0, split="dev", frame_id=10 + i) for i in range(2)]
        te = _make_frame_with_channel(330, h, snr_db=18.0, split="test", frame_id=20)

        for inc_global in (False, True):
            cfg = GroupCConfig(include_global_branch=inc_global, pilot_nll_weight=0.05)
            model = GroupCTrainableReceiver(config=cfg, seed=77).fit(tr, dv, epochs=2)
            o_ref = model.run_frame(te)
            bw_ref = np.asarray(o_ref.aux_info["branch_weights"], dtype=np.float64)
            self.assertAlmostEqual(float(np.sum(bw_ref)), 1.0, places=10)

            scales = [
                1e-15,
                1e15,
                (2.0 ** -12) * np.exp(1j * 0.73),
                (2.0 ** 12) * np.exp(-1j * 1.91),
                100.0 * np.exp(0.5j),
            ]
            for c in scales:
                o_s = model.run_frame(_scale_frame(te, c))
                self.assertTrue(np.all(np.isfinite(o_s.h_est)))
                self.assertTrue(np.all(np.isfinite(o_s.h_var)))
                self.assertTrue(np.all(o_s.h_var > 0.0))
                self.assertTrue(np.all(np.isfinite(o_s.bit_llrs)))
                rel_h = float(
                    np.linalg.norm(o_s.h_est - c * o_ref.h_est)
                    / (abs(c) * np.linalg.norm(o_ref.h_est) + 1e-30)
                )
                self.assertLess(rel_h, 1e-8)
                np.testing.assert_allclose(o_s.bit_llrs, o_ref.bit_llrs, atol=1e-7)
                self.assertEqual(int(np.sum(o_s.hard_bits != o_ref.hard_bits)), 0)
                np.testing.assert_allclose(o_s.aux_info["branch_weights"], bw_ref, atol=1e-8)

    def test_branch_permutation_equivariance(self) -> None:
        h = _make_channel(401)
        tr = [_make_frame_with_channel(410 + i, h, snr_db=16.0, split="train", frame_id=i) for i in range(4)]
        dv = [_make_frame_with_channel(420 + i, h, snr_db=16.0, split="dev", frame_id=10 + i) for i in range(2)]
        te = _make_frame_with_channel(430, h, snr_db=18.0, split="test", frame_id=20)
        model = GroupCTrainableReceiver(GroupCConfig(include_global_branch=False), seed=19).fit(tr, dv, epochs=2)

        rng = np.random.default_rng(444)
        branches = [
            (
                h[None, ACTIVE_START:ACTIVE_END]
                + 0.08 * (rng.standard_normal((N_SYMB, N_ACTIVE)) + 1j * rng.standard_normal((N_SYMB, N_ACTIVE))),
                rng.uniform(0.01, 0.08, size=(N_SYMB, N_ACTIVE)),
            )
            for _ in range(4)
        ]
        perm = [2, 0, 3, 1]
        o_orig = model.run_frame(te, branches=branches)
        o_perm = model.run_frame(te, branches=[branches[i] for i in perm])

        bw_orig = np.asarray(o_orig.aux_info["branch_weights"], dtype=np.float64)
        bw_perm = np.asarray(o_perm.aux_info["branch_weights"], dtype=np.float64)
        np.testing.assert_allclose(bw_perm, bw_orig[perm], atol=1e-12)
        np.testing.assert_allclose(o_perm.bit_llrs, o_orig.bit_llrs, atol=1e-12)

    def test_heldout_cv_literal_zero_leakage(self) -> None:
        h = _make_channel(501)
        tr = [_make_frame_with_channel(510 + i, h, snr_db=16.0, split="train", frame_id=i) for i in range(4)]
        dv = [_make_frame_with_channel(520 + i, h, snr_db=16.0, split="dev", frame_id=10 + i) for i in range(2)]

        for inc_global in (False, True):
            model = GroupCTrainableReceiver(
                GroupCConfig(include_global_branch=inc_global, pilot_nll_weight=0.05),
                seed=55,
            ).fit(tr, dv, epochs=2)

            # Case A: explicit held-out subset
            te_exp = _make_frame_with_channel(530, h, snr_db=16.0, split="test", frame_id=20)
            te_exp.meta["heldout_pilot_indices"] = [2, 5, 8, 11]
            te_exp_mut = copy.deepcopy(te_exp)
            for pi in [2, 5, 8, 11]:
                te_exp_mut.rx_grid[PILOT_SYMBOL_IDX, PILOT_FFT_INDICES[pi]] += 1e12 - 5e11j
            o1 = model.run_frame(te_exp)
            o2 = model.run_frame(te_exp_mut)
            np.testing.assert_array_equal(o1.heldout_pilot_pred, o2.heldout_pilot_pred)
            np.testing.assert_array_equal(o1.heldout_pilot_var, o2.heldout_pilot_var)

            # Case B: 4-fold default CV (mutate one fold's held-out pilots; verify those pilots' pred unchanged)
            te_cv = _make_frame_with_channel(531, h, snr_db=16.0, split="test", frame_id=21)
            fold0_idx = [1, 5, 9]
            te_cv_mut = copy.deepcopy(te_cv)
            for pi in fold0_idx:
                te_cv_mut.rx_grid[PILOT_SYMBOL_IDX, PILOT_FFT_INDICES[pi]] += 1e12 + 3e11j
            c1 = model.run_frame(te_cv)
            c2 = model.run_frame(te_cv_mut)
            eval_idx = c1.aux_info["heldout_pilot_indices"]
            pos0 = [eval_idx.index(pi) for pi in fold0_idx]
            np.testing.assert_array_equal(c1.heldout_pilot_pred[pos0], c2.heldout_pilot_pred[pos0])
            np.testing.assert_array_equal(c1.heldout_pilot_var[pos0], c2.heldout_pilot_var[pos0])

    def test_fit_and_inference_firewalls(self) -> None:
        h = _make_channel(601)
        tr = [_make_frame_with_channel(610 + i, h, snr_db=16.0, split="train", frame_id=i) for i in range(4)]
        dv = [_make_frame_with_channel(620 + i, h, snr_db=16.0, split="dev", frame_id=10 + i) for i in range(2)]
        bad_test = _make_frame_with_channel(630, h, snr_db=16.0, split="test", frame_id=20)
        bad_guard = _make_frame_with_channel(631, h, snr_db=16.0, split="guard", frame_id=21)

        with self.assertRaises(ValueError):
            GroupCTrainableReceiver(seed=1).fit(tr + [bad_test], dv, epochs=1)
        with self.assertRaises(ValueError):
            GroupCTrainableReceiver(seed=1).fit(tr, dv + [bad_test], epochs=1)
        with self.assertRaises(ValueError):
            GroupCTrainableReceiver(seed=1).fit(tr + [bad_guard], dv, epochs=1)
        with self.assertRaises(ValueError):
            GroupCTrainableReceiver(seed=1).fit(tr, dv + [bad_guard], epochs=1)
        with self.assertRaises(ValueError):
            GroupCTrainableReceiver(seed=1, context_pool=tr + [bad_test])

        # Inference must ignore tx_bits, tx_symbols, and reported_sinr_db completely
        model = GroupCTrainableReceiver(seed=91).fit(tr, dv, epochs=1)
        te_mut = copy.deepcopy(bad_test)
        te_mut.tx_bits[:] = 1 - te_mut.tx_bits
        te_mut.tx_symbols[:] = -te_mut.tx_symbols
        te_mut.reported_sinr_db = -99.0

        o_clean = model.run_frame(bad_test)
        o_mut = model.run_frame(te_mut)
        np.testing.assert_array_equal(o_clean.h_est, o_mut.h_est)
        np.testing.assert_array_equal(o_clean.bit_llrs, o_mut.bit_llrs)
        np.testing.assert_array_equal(o_clean.hard_bits, o_mut.hard_bits)

    def test_save_load_bit_exact_and_minimal_fit_zero_nan(self) -> None:
        h = _make_channel(701)
        tr = [_make_frame_with_channel(710 + i, h, snr_db=16.0, split="train", frame_id=i) for i in range(2)]
        dv = [_make_frame_with_channel(720, h, snr_db=16.0, split="dev", frame_id=10)]
        te = _make_frame_with_channel(730, h, snr_db=16.0, split="test", frame_id=20)

        for inc_global in (False, True):
            cfg = GroupCConfig(include_global_branch=inc_global, pilot_nll_weight=0.05)
            model = GroupCTrainableReceiver(config=cfg, seed=707).fit(tr, dv, epochs=1)
            o1 = model.run_frame(te)

            with tempfile.TemporaryDirectory() as td:
                ckpt = Path(td) / "model_c.pt"
                model.save(ckpt)
                self.assertLess(ckpt.stat().st_size, 100 * 1024)
                reloaded = GroupCTrainableReceiver.load(ckpt)
                o2 = reloaded.run_frame(te)  # no context_pool passed — uses saved pool_summary!

            np.testing.assert_array_equal(o1.h_est, o2.h_est)
            np.testing.assert_array_equal(o1.h_var, o2.h_var)
            np.testing.assert_array_equal(o1.bit_llrs, o2.bit_llrs)
            np.testing.assert_array_equal(o1.hard_bits, o2.hard_bits)
            np.testing.assert_array_equal(o1.aux_info["branch_weights"], o2.aux_info["branch_weights"])

        # Zero input & NaN input & convenience function
        fr_zero = copy.deepcopy(te)
        fr_zero.rx_grid[:] = 0.0
        o_zero = run_group_c_frame(fr_zero, model=model)
        self.assertFalse(bool(o_zero.aux_info["failed"]))
        self.assertTrue(np.all(np.isfinite(o_zero.bit_llrs)))

        fr_nan = copy.deepcopy(te)
        fr_nan.rx_grid[0, 0] = complex(float("nan"), 0.0)
        o_nan = model.run_frame(fr_nan)
        self.assertTrue(bool(o_nan.aux_info["failed"]))
        self.assertIsNotNone(o_nan.aux_info["error"])

        # Severe OOD stress remains finite and strictly h_var > 0.0
        ood = apply_stress_perturbation(te, extra_awgn_db=10.0, cfo_norm=0.08, phase_drift_rad=0.5, seed=99)
        o_ood = model.run_frame(ood)
        self.assertFalse(bool(o_ood.aux_info["failed"]))
        self.assertTrue(np.all(np.isfinite(o_ood.bit_llrs)))
        self.assertTrue(np.all(o_ood.h_var > 0.0))


if __name__ == "__main__":
    unittest.main()
