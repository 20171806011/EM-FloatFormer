"""Comprehensive unit, physics, equivariance, zero-leakage & gradient invariants for Phase 7."""
from dataclasses import replace
import math
import unittest

import torch
from torch.nn import functional as F

from p5.study import setup
from p6.data import dev_frame, dev_seed
from p6_consistency.weighted import frequency_operators, weighted_amplitude_update
from p7.conditions import apply_observation_gain, mask_for_condition
from p7.hypotheses import frequency_operators_fast, mixture_posterior, weighted_amplitude_solve
from p7.metrics import holdout_nll_loss, sequential_update_metrics
from p7.model import EM7HypothesisReceiver, EM7Output
from p7.routing_operator import EvidenceRoutingBlock, routing_features
from p7.statistics import exact_sign_test, holm_adjust


class TestPhase7Invariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup()

    def test_fast_operators_and_amplitude_match_p6_consistency(self):
        torch.manual_seed(77)
        b, t, n, p = 4, 20, 32, 3
        basis = torch.polar(torch.ones(b, p, n, dtype=torch.float64), torch.randn(b, p, n, dtype=torch.float64))
        rot = torch.polar(torch.ones(b, t, n, p, dtype=torch.float64), torch.randn(b, t, n, p, dtype=torch.float64))
        cpe = torch.polar(torch.ones(b, t, dtype=torch.float64), torch.randn(b, t, dtype=torch.float64))
        op_ref = frequency_operators(basis, rot, cpe)
        op_fast = frequency_operators_fast(basis, rot, cpe)
        torch.testing.assert_close(op_fast, op_ref, rtol=1e-14, atol=1e-14)

        mean = torch.randn(b, t, n, dtype=torch.complex128)
        variance = torch.rand(b, t, n, dtype=torch.float64) * 0.5
        y = torch.randn(b, 1, t, n, dtype=torch.complex128)
        weights = (torch.rand(b, t, n, dtype=torch.float64) > 0.1).double()
        ridge = torch.eye(p, dtype=torch.float64)[None].expand(b, -1, -1) * 0.1
        delta_h = torch.randn(b, 1, t, n, dtype=torch.complex128) * 0.05

        amp_ref = weighted_amplitude_update(mean, variance, y, basis, rot, cpe, weights, ridge, delta_h).amplitudes
        amp_fast = weighted_amplitude_solve(mean, variance, y, op_fast, weights, ridge, delta_h)
        torch.testing.assert_close(amp_fast, amp_ref, rtol=1e-13, atol=1e-13)

    def test_holdout_firewall_zero_leakage(self):
        frame = dev_frame(2, 6, 901, 2, 0)
        torch.manual_seed(dev_seed(2, 5, 901, 900))
        model = EM7HypothesisReceiver("axial").eval()
        # Perturb head away from init so learned path is active
        with torch.no_grad():
            model.head.read.add_(0.02 + 0.01j)
            model.fusion_scale.add_(torch.tensor([0.1, -0.1], dtype=torch.float64))

        for cond in ("C1_row9_erased", "C4a_roving_window", "C5_holdout_update"):
            gain, _, _ = mask_for_condition(frame.obs.pilot_mask, cond, 2, seed=42)
            masked = apply_observation_gain(frame, gain)
            corrupted_obs = replace(masked.obs, y=torch.where(gain[:, None] > 0, masked.obs.y, 1e9 + 1e9j))
            with torch.no_grad():
                out_clean = model(masked.obs, gain)
                out_corrupt = model(corrupted_obs, gain)
            for name in ("bit_llr", "pred_y", "pred_var", "pi", "h_hat", "bit_disagreement"):
                self.assertTrue(
                    torch.equal(getattr(out_clean, name), getattr(out_corrupt, name)),
                    f"Leakage detected in {cond}/{name}",
                )

    def test_cstar_gauge_equivariance_strict(self):
        frame = dev_frame(2, 6, 902, 2, 0)
        gains_complex = (
            1.0 + 0j,
            -1.0 + 0j,
            1e-20 * (0.6 - 0.8j),
            1e15 * (-0.8 + 0.6j),
            1e-10j,
            3.7e8 * torch.exp(torch.tensor(1.9j)),
            -2.3e-6 * torch.exp(torch.tensor(-0.7j)),
            0.05 * (0.3 + 0.95j),
            128.0 * (-0.5 - 0.8660254037844386j),
        )
        for variant in ("axial", "local", "fixed", "single_axial"):
            torch.manual_seed(dev_seed(2, 5, 902, 900))
            model = EM7HypothesisReceiver(variant).eval()
            if model.learned:
                with torch.no_grad():
                    model.head.read.add_(0.015 - 0.02j)
                    model.head.kernel.add_(0.03)
                    model.fusion_scale.add_(torch.tensor([0.05, 0.08], dtype=torch.float64))
            for cond in ("C0_intact", "C1_row9_erased", "C2_row0_erased", "C3_scattered_half", "C4a_roving_window"):
                g_mask, _, _ = mask_for_condition(frame.obs.pilot_mask, cond, 2, seed=99)
                base_frame = apply_observation_gain(frame, g_mask)
                _, _, probe_mask = mask_for_condition(frame.obs.pilot_mask, "C5_holdout_update", 2, seed=99)
                with torch.no_grad():
                    ref = model(base_frame.obs, g_mask)
                    ref_nll = holdout_nll_loss(
                        ref.log_pi, ref.branch_pred_y, ref.branch_pred_var,
                        frame.obs.y[:, 0], frame.obs.noise_var, probe_mask
                    )
                for alpha in gains_complex:
                    a = complex(alpha)
                    scaled_obs = replace(
                        base_frame.obs,
                        y=base_frame.obs.y * a,
                        noise_var=base_frame.obs.noise_var * (abs(a) ** 2),
                    )
                    with torch.no_grad():
                        got = model(scaled_obs, g_mask)
                        got_nll = holdout_nll_loss(
                            got.log_pi, got.branch_pred_y, got.branch_pred_var,
                            frame.obs.y[:, 0] * a, frame.obs.noise_var * (abs(a) ** 2), probe_mask
                        )
                    max_llr_diff = float((got.bit_llr - ref.bit_llr).abs().max())
                    self.assertLess(max_llr_diff, 1e-12, f"{variant}/{cond}/alpha={a}: {max_llr_diff}")
                    flips = int(((got.bit_llr > 0) != (ref.bit_llr > 0)).sum())
                    self.assertEqual(flips, 0, f"Bit flip in {variant}/{cond}")
                    rel_pred_y = float(((got.pred_y / a) - ref.pred_y).abs().max()
                                       / ref.pred_y.abs().max().clamp_min(1e-300))
                    self.assertLess(rel_pred_y, 1e-12)
                    rel_pred_var = float(((got.pred_var / (abs(a) ** 2)) - ref.pred_var).abs().max()
                                         / ref.pred_var.abs().max().clamp_min(1e-300))
                    self.assertLess(rel_pred_var, 1e-12)
                    self.assertLess(float((got.pi - ref.pi).abs().max()), 1e-12)
                    self.assertLess(float((got_nll - ref_nll).abs()), 1e-12)

    def test_parameter_and_init_parity(self):
        torch.manual_seed(12345)
        axial = EM7HypothesisReceiver("axial").eval()
        torch.manual_seed(12345)
        local = EM7HypothesisReceiver("local").eval()
        local.load_state_dict(axial.state_dict())
        fixed = EM7HypothesisReceiver("fixed").eval()

        self.assertEqual(axial.real_parameters(), local.real_parameters())
        self.assertGreaterEqual(axial.real_parameters(), 900)
        self.assertEqual(fixed.real_parameters(), 0)

        frame = dev_frame(2, 6, 903, 2, 0)
        gain, _, _ = mask_for_condition(frame.obs.pilot_mask, "C4a_roving_window", 2, seed=7)
        obs_c = apply_observation_gain(frame, gain).obs
        with torch.no_grad():
            o_ax = axial(obs_c, gain)
            o_lo = local(obs_c, gain)
            o_fx = fixed(obs_c, gain)
        for field in ("bit_llr", "h_hat", "pi", "pred_y", "pred_var", "bit_disagreement"):
            self.assertTrue(torch.equal(getattr(o_ax, field), getattr(o_fx, field)), field)
            self.assertTrue(torch.equal(getattr(o_lo, field), getattr(o_fx, field)), field)

    def test_remote_evidence_reach(self):
        torch.manual_seed(987)
        axial_block = EvidenceRoutingBlock("axial", width=8).eval()
        local_block = EvidenceRoutingBlock("local", width=8).eval()
        local_block.load_state_dict(axial_block.state_dict())

        b_m, t, n = 4, 20, 32
        feat = torch.randn(b_m, 1, t, n, 10, dtype=torch.complex128)
        changed = feat.clone()
        changed[:, :, 9, 28] *= 2.5j  # single remote site far from (t=1, f=2)

        with torch.no_grad():
            p_ax_0, r_ax_0 = axial_block(feat)
            p_ax_1, r_ax_1 = axial_block(changed)
            p_lo_0, r_lo_0 = local_block(feat)
            p_lo_1, r_lo_1 = local_block(changed)

        ax_diff = float((p_ax_1[:, :, 1, 2] - p_ax_0[:, :, 1, 2]).abs().max()
                        + (r_ax_1[:, :, 1, 2] - r_ax_0[:, :, 1, 2]).abs().max())
        lo_diff = float((p_lo_1[:, :, 1, 2] - p_lo_0[:, :, 1, 2]).abs().max()
                        + (r_lo_1[:, :, 1, 2] - r_lo_0[:, :, 1, 2]).abs().max())
        self.assertGreater(ax_diff, 1e-12)
        self.assertEqual(lo_diff, 0.0)

    def test_total_variance_and_mixture_llr_identities(self):
        torch.manual_seed(11)
        b, m, t, n = 3, 4, 2, 4
        pi = torch.softmax(torch.randn(b, m, dtype=torch.float64), dim=-1)
        llrs = torch.randn(b, m, t, n, 4, dtype=torch.float64) * 2.0
        means = torch.randn(b, m, t, n, dtype=torch.complex128)
        vars_m = torch.rand(b, m, t, n, dtype=torch.float64) + 0.05
        llr_mix, disagree, sym_mean, tot_v, in_v, bet_v = mixture_posterior(pi, pi.log(), llrs, means, vars_m)

        # Brute force 2nd moment: E[|X - E[X]|^2] = sum_m pi_m (v_m + |mu_m|^2) - |sum_m pi_m mu_m|^2
        brute_second = (pi[:, :, None, None] * (vars_m + means.abs().square())).sum(1)
        brute_tot = brute_second - sym_mean.abs().square()
        torch.testing.assert_close(tot_v, brute_tot, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(tot_v, in_v + bet_v, rtol=1e-14, atol=1e-14)

        # Specific agreement vs opposition identity checks
        pi2 = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        agree = torch.full((1, 2, 1, 1, 4), math.log(0.999 / 0.001), dtype=torch.float64)
        oppose = torch.tensor([[[[[math.log(0.999 / 0.001)] * 4]],
                                [[[math.log(0.001 / 0.999)] * 4]]]], dtype=torch.float64)
        z = torch.zeros((1, 2, 1, 1), dtype=torch.complex128)
        v = torch.ones((1, 2, 1, 1), dtype=torch.float64)
        llr_ag, dis_ag, *_ = mixture_posterior(pi2, pi2.log(), agree, z, v)
        llr_op, dis_op, *_ = mixture_posterior(pi2, pi2.log(), oppose, z, v)
        self.assertGreater(float(llr_ag.min()), 6.5)
        self.assertLess(float(dis_ag.max()), 1e-6)
        self.assertLess(float(llr_op.abs().max()), 1e-12)
        self.assertGreater(float(dis_op.min()), 0.249)

    def test_full_active_finite_gradients(self):
        frame = dev_frame(2, 6, 901, 2, 0)
        gain, _, _ = mask_for_condition(frame.obs.pilot_mask, "C4a_roving_window", 2, seed=55)
        _, _, probe_mask = mask_for_condition(frame.obs.pilot_mask, "C5_holdout_update", 2, seed=55)
        obs_c = apply_observation_gain(frame, gain).obs

        for variant in ("axial", "local", "single_axial"):
            for mag in (1e-20, 1e15):
                alpha = mag * (0.6 + 0.8j)
                scaled_obs = replace(obs_c, y=obs_c.y * alpha, noise_var=obs_c.noise_var * (abs(alpha) ** 2))
                torch.manual_seed(dev_seed(2, 5, 901, 900))
                model = EM7HypothesisReceiver(variant)
                model.zero_grad(set_to_none=True)
                out = model(scaled_obs, gain)

                payload = out.bit_llr[:, ~frame.obs.pilot_mask]
                signed_bits = 2. * frame.bits.double() - 1.
                bce = F.softplus(-signed_bits * (10. * torch.tanh(payload / 10.))).mean()
                nll = holdout_nll_loss(
                    out.log_pi, out.branch_pred_y, out.branch_pred_var,
                    frame.obs.y[:, 0] * alpha, frame.obs.noise_var * (abs(alpha) ** 2), probe_mask
                )
                nmse = ((out.h_hat[:, 0] - frame.h_true[:, 0] * alpha).abs().square().sum()
                        / (frame.h_true[:, 0] * alpha).abs().square().sum().clamp_min(1e-300))
                anchor = sum((p - q).abs().square().sum()
                             for p, q in zip(model.head.parameters(), model.initial_map.parameters()))
                loss = bce + 0.05 * nll + 0.005 * nmse + 5e-5 * anchor
                loss.backward()

                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    if variant == "single_axial" and name == "fusion_scale":
                        self.assertIsNone(param.grad)
                        continue
                    self.assertIsNotNone(param.grad, f"None grad for {variant}/{name}")
                    self.assertTrue(bool(torch.isfinite(param.grad).all()), f"Nonfinite grad {variant}/{name}")
                    if variant == "single_axial" and name == "head.embed":
                        # Rows 4,5 receive disagreement_h/y which are identically zero when M=1;
                        # all other 8 rows must carry nonzero gradients.
                        self.assertEqual(float(param.grad[4:6].abs().max()), 0.0)
                        non_dis_rows = torch.cat((param.grad[:4], param.grad[6:]), dim=0)
                        self.assertTrue(bool((non_dis_rows.abs().amax(-1) > 0).all()))
                    elif variant == "single_axial" and name == "head.read":
                        # Columns 3,4 feed delta_route which does not affect softmax(M=1)=1
                        self.assertTrue(bool((param.grad[:, :3].abs().amax(0) > 0).all()))
                    else:
                        self.assertGreater(float(param.grad.abs().max()), 0.0,
                                           f"Zero grad for {variant}/{name} at mag={mag}")

    def test_exact_paired_statistics(self):
        ties = exact_sign_test([0.0] * 10)
        self.assertEqual(ties["raw_exact"], "1")
        all_win = exact_sign_test([1.0] * 10)
        self.assertEqual(all_win["raw_exact"], "1/512")
        mixed = exact_sign_test([1.0] * 8 + [0.0, -1.0])
        self.assertEqual((mixed["wins"], mixed["ties"], mixed["losses"]), (8, 1, 1))
        rows = holm_adjust([
            {"raw_exact": "1/32"},
            {"raw_exact": "1/512"},
            {"raw_exact": "1/32"},
            {"raw_exact": "1"},
        ])
        self.assertEqual([r["holm_exact"] for r in rows], ["3/32", "1/128", "3/32", "1"])

    def test_sequential_update_metrics_probability_space(self):
        frame = dev_frame(2, 6, 901, 2, 0)
        mask = frame.obs.pilot_mask
        n_pay = int((~mask).sum())
        llr1 = torch.zeros(2, *mask.shape, 4, dtype=torch.float64)
        llr2 = torch.zeros_like(llr1)
        dis1 = torch.zeros_like(llr1)
        # First 20% of payload bits are split (d=0.15 -> flip from -2 to +2), rest certain (d=0 -> stay at +5 to +5.5)
        split_len = (n_pay * 4) // 5
        flat_d = dis1[:, ~mask].reshape(2, -1)
        flat_l1 = llr1[:, ~mask].reshape(2, -1)
        flat_l2 = llr2[:, ~mask].reshape(2, -1)
        flat_d[:, :split_len] = 0.15
        flat_l1[:, :split_len] = -2.0
        flat_l2[:, :split_len] = 2.0
        flat_d[:, split_len:] = 0.0
        flat_l1[:, split_len:] = 5.0
        flat_l2[:, split_len:] = 5.5
        dis1[:, ~mask] = flat_d.reshape(2, n_pay, 4)
        llr1[:, ~mask] = flat_l1.reshape(2, n_pay, 4)
        llr2[:, ~mask] = flat_l2.reshape(2, n_pay, 4)

        def _dummy(bit_llr, dis):
            z = torch.zeros(1)
            return EM7Output(bit_llr, frame.h_true, z, z, z, z, z, z, dis, z, z, z, z, [], z, z, z)

        seq = sequential_update_metrics(frame, _dummy(llr1, dis1), _dummy(llr2, torch.zeros_like(dis1)))
        exp_high_ps = abs(
            torch.sigmoid(torch.tensor(2.0, dtype=torch.float64)).item()
            - torch.sigmoid(torch.tensor(-2.0, dtype=torch.float64)).item()
        )
        exp_low_ps = abs(
            torch.sigmoid(torch.tensor(5.5, dtype=torch.float64)).item()
            - torch.sigmoid(torch.tensor(5.0, dtype=torch.float64)).item()
        )
        self.assertAlmostEqual(seq["high_disagreement_prob_shift"], exp_high_ps, places=12)
        self.assertAlmostEqual(seq["low_disagreement_prob_shift"], exp_low_ps, places=12)
        self.assertEqual(seq["high_disagreement_flip_rate"], 1.0)
        self.assertEqual(seq["low_disagreement_flip_rate"], 0.0)
        self.assertGreater(seq["top_vs_bottom_prob_shift_ratio"], 200.0)
        self.assertGreater(seq["split_vs_certain_prob_shift_ratio"], 200.0)
        self.assertGreater(seq["prob_shift_corr"], 0.5)

        # Constant/zero disagreement (e.g. SingleHyp) must not yield NaN
        seq_const = sequential_update_metrics(frame, _dummy(llr1, torch.zeros_like(dis1)), _dummy(llr2, torch.zeros_like(dis1)))
        self.assertEqual(seq_const["prob_shift_corr"], 0.0)
        self.assertAlmostEqual(seq_const["top_vs_bottom_prob_shift_ratio"], 1.0, places=12)
        self.assertIsNone(seq_const["split_vs_certain_prob_shift_ratio"])


if __name__ == "__main__":
    unittest.main()
