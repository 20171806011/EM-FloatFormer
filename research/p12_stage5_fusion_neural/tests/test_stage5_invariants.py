"""Comprehensive invariant, consistency, zero-leakage, symmetry & parity tests for CLI-4 Stage 5."""
from __future__ import annotations

import ast
import math
import os
from pathlib import Path
import unittest

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from p11.contract import N, PROBE_ROW, T, OffGridObservation, fractional_delay_freq_response, unitary_dft_matrix
from p12.contract import OFDM_PILOT_COMB, OFFGRID_REGIME_TRUE_RESET
from p12.offgrid_physics import make_roving_bilateral_3fold, score_offgrid_channel_est, simulate_offgrid_link_p12
from p12.offgrid_receiver import run_p12_offgrid_physics, warmup_p12_offgrid_numba

from research.p12_stage5_fusion_neural.consistent_fusion_engine import (
    ContentAddressingConfig,
    MixtureDecodeConfig,
    build_coherent_row_mixture,
    evaluate_decoder_mode_on_folds,
    mix_branch_bit_llrs_exact,
    run_consistent_folds,
)
from research.p12_stage5_fusion_neural.contract import (
    SEED_S5_DEV,
    SEED_S5_TEST_ID,
    SEED_S5_TEST_OOD,
    SEED_S5_TRAIN,
    SEED_S5_UNIT,
    assert_s5_seed_disjoint,
    is_prior_p1_to_p12_seed,
)
from research.p12_stage5_fusion_neural.neural_models import (
    COMPLEX,
    REAL,
    build_matched_stage5_models,
    compute_family_summaries,
    extract_stage5_fold_features,
    infer_stage5_fold_override,
    stage5_differentiable_loss,
)

PKG_DIR = Path(__file__).resolve().parent.parent


class TestStage5Invariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        warmup_p12_offgrid_numba()

    def test_seed_namespace_disjointness(self) -> None:
        # Prior P1..P12 seeds must be rejected
        for bad in (0, 12345, 1_110_000_000_001, 1_240_500_001_000, 1_260_020_000_000, 9_102_379_953_802):
            self.assertIsNotNone(is_prior_p1_to_p12_seed(bad))
            with self.assertRaises(ValueError):
                assert_s5_seed_disjoint([bad])
        # Stage 5 seeds across all 5 splits must pass and be disjoint
        all_s = [
            SEED_S5_DEV + 1,
            SEED_S5_TRAIN + 1,
            SEED_S5_TEST_ID + 1,
            SEED_S5_TEST_OOD + 1,
            SEED_S5_UNIT + 1,
        ]
        assert_s5_seed_disjoint(all_s)
        assert_s5_seed_disjoint([SEED_S5_DEV + 7], expected_namespace="s5_dev")
        with self.assertRaises(ValueError):
            assert_s5_seed_disjoint([SEED_S5_TEST_ID + 1], expected_namespace="s5_dev")

    def test_exact_probability_domain_mixture_llr_counterexample(self) -> None:
        """Verify research/posterior_fusion §5.2 counterexample: (0.8, 0.2), P(b=1|m)=(0.7, 0.001)."""
        p1 = np.array([0.7, 0.001], dtype=np.float64)
        branch_llrs = np.log(p1 / (1.0 - p1))[:, None]  # [2, 1]
        log_beta = np.log(np.array([0.8, 0.2], dtype=np.float64))
        naive_avg = float(0.8 * branch_llrs[0, 0] + 0.2 * branch_llrs[1, 0])
        self.assertLess(naive_avg, 0.0)  # Linear LLR averaging flips the sign!

        exact_mix = float(mix_branch_bit_llrs_exact(branch_llrs, log_beta)[0])
        expected_p1 = 0.8 * 0.7 + 0.2 * 0.001  # 0.5602
        expected_llr = math.log(expected_p1 / (1.0 - expected_p1))
        self.assertGreater(exact_mix, 0.0)
        self.assertAlmostEqual(exact_mix, expected_llr, places=12)

    def test_baseline_wrapper_exact_parity_with_p12_655de6a(self) -> None:
        """`run_consistent_folds(content_cfg=None, wave_mix_cfg=None)` + `mean_channel` matches P12 655de6a identically."""
        seed = SEED_S5_UNIT + 101
        obs, truth, _ = simulate_offgrid_link_p12(seed, "offgrid_partial_shift", snr_db=9.0, pilot_rows_a=OFDM_PILOT_COMB)
        ref = run_p12_offgrid_physics(obs)
        ref_sc = score_offgrid_channel_est(ref.h_mix, ref.noise_mix, ref.gate_out.mix_llrs, obs, truth, probe_pred=ref.gate_out.mix_probe_pred)

        folds = run_consistent_folds(obs, content_cfg=None, wave_mix_cfg=None)
        ours = evaluate_decoder_mode_on_folds(obs, truth, folds, mode="mean_channel", use_burst_guard=True)

        self.assertEqual(ref.delay_reset_triggered, folds.delay_reset_triggered)
        self.assertEqual(ref.helpers_accepted, folds.helpers_accepted)
        self.assertEqual(ref_sc["block_errors"], ours["block_errors"])
        self.assertAlmostEqual(ref_sc["mean_rel_frob_err"], ours["mean_rel_frob_err"], places=12)
        self.assertAlmostEqual(ref_sc["probe_clean_nmse"], ours["probe_clean_nmse"], places=12)

    def test_zero_target_and_probe_leakage_across_all_upgraded_and_neural_arms(self) -> None:
        """Tampering Fold-0 targets and PROBE_ROW produces EXACT 0.0 diff on Fold-0 channel, mixture, and Role A/B outputs."""
        seed = SEED_S5_UNIT + 102
        obs, _, _ = simulate_offgrid_link_p12(seed, "offgrid_midframe_shift", snr_db=9.0, pilot_rows_a=OFDM_PILOT_COMB)
        r_anc = int(obs.training_rows[0])
        (t0, _, _), _ = make_roving_bilateral_3fold(r_anc, PROBE_ROW)

        y_t = obs.y_b.copy()
        y_t[list(t0)] += 5e6 - 3e6j
        y_t[PROBE_ROW] = 9e9 - 7e9j
        obs_t = OffGridObservation(obs.y_a, y_t, obs.training_rows, obs.probe_row, obs.n0, obs.pilot_tx_b, obs.pilot_tx_a)

        content_cfg = ContentAddressingConfig()
        mix_cfg = MixtureDecodeConfig()
        f_clean = run_consistent_folds(obs, content_cfg=content_cfg, wave_mix_cfg=mix_cfg)
        f_tamp = run_consistent_folds(obs_t, content_cfg=content_cfg, wave_mix_cfg=mix_cfg)

        snap_c = f_clean.folds[0]
        snap_t = f_tamp.folds[0]
        self.assertEqual(float(np.max(np.abs(snap_c.lw_cur - snap_t.lw_cur))), 0.0)
        self.assertEqual(float(np.max(np.abs(snap_c.m_u - snap_t.m_u))), 0.0)
        self.assertEqual(float(np.max(np.abs(snap_c.v_u - snap_t.v_u))), 0.0)
        if snap_c.per_target_lw is not None:
            for r in t0:
                self.assertEqual(float(np.max(np.abs(snap_c.per_target_lw[r] - snap_t.per_target_lw[r]))), 0.0)

        # Neural Role A & Role B must ALSO have exact 0.0 leakage on Fold 0!
        torch.manual_seed(5001)
        model_a, model_b = build_matched_stage5_models()
        # Perturb weights away from step 0 so outputs are non-zero!
        with torch.no_grad():
            for m in (model_a, model_b):
                for p in m.parameters():
                    if p.requires_grad:
                        p.add_(0.05)

        for label, model in (("RoleA", model_a), ("RoleB", model_b)):
            df_c, dg_c, dlv_c = infer_stage5_fold_override([model], snap_c)
            df_t, dg_t, dlv_t = infer_stage5_fold_override([model], snap_t)
            self.assertEqual(float(np.max(np.abs(df_c - df_t))), 0.0, f"{label} target/probe leakage in delta_fam")
            self.assertEqual(float(np.max(np.abs(dg_c - dg_t))), 0.0, f"{label} target/probe leakage in delta_g")
            self.assertEqual(float(np.max(np.abs(dlv_c - dlv_t))), 0.0, f"{label} target/probe leakage in delta_lv")

    def test_role_a_symmetric_connection_unitarity_cstar_and_role_b_param_parity(self) -> None:
        torch.manual_seed(5001)
        model_a, model_b = build_matched_stage5_models()
        # 1) Exact integer real-parameter parity
        self.assertEqual(model_a.real_parameters(), model_b.real_parameters())

        # 2) Exact reverse-edge unitarity |P_ij P_ji - 1| < 1e-15 & P_ji == conj(P_ij)
        tau = torch.tensor([[0.4, 1.7, 3.1, 4.9, 6.2]], dtype=REAL)
        nu = torch.tensor([[0.05, -0.07, 0.03, -0.04, 0.08]], dtype=REAL)
        t_phys = torch.arange(T, dtype=REAL).unsqueeze(0) * 1.5
        for layer in model_a.backbone.layers:
            p_conn = layer.compute_symmetric_connection(tau, nu, t_phys)
            rev_err = float(torch.max(torch.abs(p_conn * p_conn.transpose(-1, -2) - 1.0)).item())
            conj_err = float(torch.max(torch.abs(p_conn.transpose(-1, -2) - p_conn.conj())).item())
            self.assertLess(rev_err, 1e-14)
            self.assertLess(conj_err, 1e-14)

        # 3) Step-0 zero-residual centering (< 1e-13) AND 100% nonzero step-0 gradients for both Role A and Role B
        seed = SEED_S5_UNIT + 103
        obs, truth, _ = simulate_offgrid_link_p12(seed, "offgrid_continuous", snr_db=9.0, pilot_rows_a=OFDM_PILOT_COMB)
        folds = run_consistent_folds(obs, content_cfg=ContentAddressingConfig(), wave_mix_cfg=MixtureDecodeConfig())
        snap = folds.folds[0]
        feats = extract_stage5_fold_features(snap)

        s_t = torch.from_numpy(feats.scalar_feats).to(REAL).unsqueeze(0)
        c_t = torch.from_numpy(feats.complex_feats).to(COMPLEX).unsqueeze(0)
        tau_t = torch.from_numpy(feats.tau_hat).to(REAL).unsqueeze(0)
        nu_t = torch.from_numpy(feats.nu_post).to(REAL).unsqueeze(0)
        tp_t = torch.from_numpy(feats.t_phys).to(REAL).unsqueeze(0)
        tn_t = torch.from_numpy(feats.t_norm).to(REAL).unsqueeze(0)
        ev_t = torch.from_numpy(feats.ev_base).to(REAL).unsqueeze(0)

        for label, m in (("RoleA", model_a), ("RoleB", model_b)):
            df, dg, dlv = m.forward_tensors(s_t, c_t, tau_t, nu_t, tp_t, tn_t, ev_t)
            self.assertLess(float(df.abs().max().item()), 1e-13, f"{label} step-0 df nonzero")
            self.assertLess(float(dg.abs().max().item()), 1e-13, f"{label} step-0 dg nonzero")
            self.assertLess(float(dlv.abs().max().item()), 1e-13, f"{label} step-0 dlv nonzero")

        # 4) Verify differentiable loss produces finite nonzero gradients on trainable parameters
        fam_logw, fam_mean_u, fam_var_u, fam_active = compute_family_summaries(snap)
        t_idx = list(snap.t_rows)
        row_to_d = {int(r): i for i, r in enumerate(truth.data_rows)}
        d_idxs = [row_to_d[r] for r in t_idx]
        s_mag = math.sqrt(float(obs.n0))
        dft = unitary_dft_matrix(N)
        y_t_u = obs.y_b[t_idx] / s_mag
        y_t_freq = y_t_u @ dft.T

        phi_f = fractional_delay_freq_response(snap.used_delays, n=N)
        loss_kwargs = dict(
            t_rows_idx=torch.tensor([t_idx], dtype=torch.long),
            probe_row_idx=torch.tensor([snap.probe_row], dtype=torch.long),
            rot_intra_t=torch.from_numpy(feats.rot_intra[:, t_idx, :]).to(COMPLEX).unsqueeze(0),
            rot_intra_p=torch.from_numpy(feats.rot_intra[:, snap.probe_row, :]).to(COMPLEX).unsqueeze(0),
            fam_logw=torch.from_numpy(fam_logw).to(REAL).unsqueeze(0),
            fam_active=torch.from_numpy(fam_active).unsqueeze(0),
            fam_mean_t=torch.from_numpy(fam_mean_u[:, :, t_idx, :]).to(COMPLEX).unsqueeze(0),
            fam_var_t=torch.from_numpy(fam_var_u[:, :, t_idx, :]).to(REAL).unsqueeze(0),
            fam_mean_p=torch.from_numpy(fam_mean_u[:, :, snap.probe_row, :]).to(COMPLEX).unsqueeze(0),
            fam_var_p=torch.from_numpy(fam_var_u[:, :, snap.probe_row, :]).to(REAL).unsqueeze(0),
            d_mats=torch.from_numpy(snap.d_mats).to(COMPLEX).unsqueeze(0),
            phi_f=torch.from_numpy(phi_f).to(COMPLEX).unsqueeze(0),
            h_true_u=torch.from_numpy(truth.h_b_true_mat[t_idx] / s_mag).to(COMPLEX).unsqueeze(0),
            y_target_freq_u=torch.from_numpy(y_t_freq).to(COMPLEX).unsqueeze(0),
            b_code_target=torch.from_numpy(truth.b_code[d_idxs].astype(np.float64)).to(REAL).unsqueeze(0),
            u_probe=torch.from_numpy(snap.u_probe).to(COMPLEX).unsqueeze(0),
            probe_clean_u=torch.from_numpy(truth.y_b_probe_clean / s_mag).to(COMPLEX).unsqueeze(0),
        )
        for label, m in (("RoleA", model_a), ("RoleB", model_b)):
            m.zero_grad()
            df, dg, dlv = m.forward_tensors(s_t, c_t, tau_t, nu_t, tp_t, tn_t, ev_t)
            tot, l_mse, l_bce, l_prb = stage5_differentiable_loss(df, dg, dlv, **loss_kwargs)
            self.assertTrue(torch.isfinite(tot).item())
            tot.backward()
            nonzero_grads = sum(
                1 for p in m.parameters() if p.requires_grad and p.grad is not None and float(p.grad.abs().max().item()) > 0.0
            )
            total_train = sum(1 for p in m.parameters() if p.requires_grad)
            self.assertEqual(nonzero_grads, total_train, f"{label} has zero-grad trainable params at step 0")

        # 5) Exact C* invariance/equivariance of Role A AND upgraded physics engine!
        alpha = 2.9 * complex(math.cos(0.85), math.sin(0.85))
        u_alpha = alpha / abs(alpha)
        obs_s = OffGridObservation(
            obs.y_a * alpha,
            obs.y_b * alpha,
            obs.training_rows,
            obs.probe_row,
            obs.n0 * (abs(alpha) ** 2),
            obs.pilot_tx_b,
            obs.pilot_tx_a,
        )
        folds_s = run_consistent_folds(obs_s, content_cfg=ContentAddressingConfig(), wave_mix_cfg=MixtureDecodeConfig())
        with torch.no_grad():
            for p in model_a.parameters():
                if p.requires_grad:
                    p.add_(0.03)
        df0, dg0, dlv0 = infer_stage5_fold_override([model_a], folds.folds[0])
        df1, dg1, dlv1 = infer_stage5_fold_override([model_a], folds_s.folds[0])
        self.assertLess(float(np.max(np.abs(df1 - df0))), 1e-9)
        self.assertLess(float(np.max(np.abs(dlv1 - dlv0))), 1e-9)
        self.assertLess(float(np.max(np.abs(dg1 - dg0 * u_alpha))), 1e-9)

    def test_audit_math_and_formal_seed_disjointness(self) -> None:
        from research.p12_stage5_fusion_neural.audit import (
            exact_sign_flip_p,
            holm_adjust,
            verdict_superiority,
            verify_replay_records_exact,
        )
        from research.p12_stage5_fusion_neural.freeze import planned_stage5_formal_seeds

        st16 = exact_sign_flip_p([1.0] * 16)
        self.assertEqual(st16["n_perm_states"], 65536)
        self.assertAlmostEqual(st16["p_two_sided"], 2.0 / 65536.0, places=12)
        self.assertEqual(verdict_superiority(st16, 0.001), "支持")

        st8_neg = exact_sign_flip_p([-2.0] * 8)
        self.assertEqual(st8_neg["n_perm_states"], 256)
        self.assertAlmostEqual(st8_neg["p_two_sided"], 2.0 / 256.0, places=12)
        self.assertEqual(verdict_superiority(st8_neg, 0.02), "反证")
        self.assertEqual(verdict_superiority(st8_neg, 0.20), "证据不足")
        self.assertEqual(holm_adjust([0.001, 0.04, 0.02]), [0.003, 0.04, 0.04])

        rec = {"split": "id", "cluster": 1, "regime": "offgrid_continuous", "link": 0, "seed": 1, "arms": {"a": {"x": 1.25, "elapsed_ms": 10.0}}}
        rec2 = {"split": "id", "cluster": 1, "regime": "offgrid_continuous", "link": 0, "seed": 1, "arms": {"a": {"x": 1.25, "elapsed_ms": 99.0}}}
        rep = verify_replay_records_exact([rec], [rec2])
        self.assertTrue(rep["exact_pass"])
        self.assertEqual(rep["max_float_diff"], 0.0)

        seeds = planned_stage5_formal_seeds(n_clusters=16, links_per_cluster_id=8, links_per_cluster_ood=4)
        self.assertEqual(len(seeds["formal_id"]), 512)
        self.assertEqual(len(seeds["formal_ood"]), 192)

    def test_no_inference_mode_and_no_hardcoded_n0_in_stage5_sources(self) -> None:
        for py_path in sorted(PKG_DIR.glob("*.py")):
            src = py_path.read_text(encoding="utf-8")
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == "inference_mode":
                    self.fail(f"Forbidden torch.inference_mode call in {py_path.name}")
                if isinstance(node, ast.Name) and node.id == "inference_mode":
                    self.fail(f"Forbidden inference_mode reference in {py_path.name}")
                if isinstance(node, ast.Constant) and isinstance(node.value, float):
                    # Guard against hardcoded 9 dB N0 = 10^(-0.9) ≈ 0.125892541...
                    self.assertFalse(
                        abs(node.value - 0.12589254117941673) < 1e-6,
                        f"Hardcoded N0 constant found in {py_path.name}",
                    )


if __name__ == "__main__":
    unittest.main()
