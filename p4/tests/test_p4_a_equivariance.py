"""Unit Test Suite for Stage P4-A: C*-Gauge-Invariant EM-Native Transformer Architecture.

Covers:
- Feature F8: Exact C* Gauge Normalization & Complex RMSNorm
- Feature F9: Hermitian Green's Propagator Attention, Complex Linear, Complex RoPE & Radial Activation
- Feature F10: Bidirectional Pilot-Data Cross-Attention & EMFloatFormer Receiver
- Feature F11: Joint Physical Consistency Objective & Scale-Invariant Loss Gradients
- Feature F12: Numerical Equivariance Verification across 24 Octaves and Full Phase Circle
"""

from __future__ import annotations

import csv
import json
import math
import unittest
from pathlib import Path

import torch

from p4.em_floatformer import (
    ComplexConv1d,
    ComplexLinear,
    ComplexRMSNorm,
    ComplexRoPE,
    GreenAttention,
    JointEmbedding,
    BidirectionalBlock,
    EMFloatFormer,
    JointPhysicalLoss,
    LocalEquivConv16QAM,
    RadialActivation,
    TinyJoint16QAM,
    apply_gauge_normalization,
    audit_c_star_equivariance,
    audit_channel_equivariance,
    audit_full_equivariance,
    audit_gain_equivariance,
    audit_loss_scale_invariance,
    audit_phase_equivariance,
    audit_single_phase,
    audit_single_scale,
    check_irrational_phases,
    check_loss_gradient_scale,
    check_noise_quad_scaling,
    compare_pilot_vs_bidirectional,
    compute_ece,
    compute_gauge_anchor,
    create_floatformer,
    evaluate_noise_data_feedback,
    evaluate_perfect_pilot_demapping,
    hermitian_attention_scores,
    rescale_channel_estimate,
    run_p4a_comparative_benchmarks,
    run_standalone_equivariance_check,
    sweep_phases_dense,
    train_step_joint_loss,
    verify_bidirectional_green_prop,
    verify_gauge_attention_joint,
    verify_loss_gradient_invariance,
)
from p4.simulator import generate_frame


class TestF8GaugeNormalization(unittest.TestCase):
    """F8: Exact C* Gauge Normalization & Complex RMSNorm."""

    def setUp(self):
        torch.set_num_threads(4)

    def test_pilot_energy_gauge_anchor(self):
        yp = torch.randn(4, 32, dtype=torch.complex64) * 5.0
        s0 = compute_gauge_anchor(yp)
        self.assertEqual(s0.shape, (4, 1))
        self.assertTrue(torch.all(s0 > 0.0))

    def test_unit_rms_normalization(self):
        yp = torch.randn(4, 32, dtype=torch.complex64) * 3.7
        yd = torch.randn(4, 32, dtype=torch.complex64) * 3.7
        yp_bar, yd_bar, s0 = apply_gauge_normalization(yp, yd)
        rms = (yp_bar.abs() ** 2).mean(dim=-1).sqrt()
        self.assertTrue(torch.allclose(rms, torch.ones_like(rms), atol=1e-5))

    def test_complex_rms_norm_phase_preservation(self):
        norm_layer = ComplexRMSNorm(d_model=64)
        z = torch.randn(2, 32, 64, dtype=torch.complex64)
        z_out = norm_layer(z)
        phase_in = torch.angle(z)
        phase_out = torch.angle(z_out)
        self.assertTrue(torch.allclose(phase_in, phase_out, atol=1e-5))

    def test_complex_rms_norm_subnormal_guard(self):
        norm = ComplexRMSNorm(d_model=32)
        z_sub = torch.full((1, 32), 1e-35, dtype=torch.complex64, requires_grad=True)
        out = norm(z_sub)
        self.assertFalse(torch.isnan(out).any())
        self.assertTrue(torch.allclose((out.abs() ** 2).mean().sqrt(), torch.tensor(1.0), atol=1e-4))
        (out.real.sum() + out.imag.sum()).backward()
        self.assertIsNotNone(z_sub.grad)
        self.assertTrue(torch.isfinite(z_sub.grad).all())
        self.assertTrue(torch.isfinite(norm.gamma.grad).all())

        norm.zero_grad()
        z_huge = torch.full((1, 32), 1e20, dtype=torch.complex64, requires_grad=True)
        out_huge = norm(z_huge)
        self.assertTrue(torch.isfinite(out_huge).all())
        self.assertTrue(torch.allclose((out_huge.abs() ** 2).mean().sqrt(), torch.tensor(1.0), atol=1e-4))
        (out_huge.real.sum() + out_huge.imag.sum()).backward()
        self.assertIsNotNone(z_huge.grad)
        self.assertTrue(torch.isfinite(z_huge.grad).all())
        self.assertTrue(torch.isfinite(norm.gamma.grad).all())

    def test_gauge_anchor_subnormal_1e35_no_collapse(self):
        yp = torch.full((2, 32), 1e-35, dtype=torch.complex64)
        yd = torch.full((2, 32), 1e-35, dtype=torch.complex64)
        s0 = compute_gauge_anchor(yp)
        self.assertTrue(torch.isfinite(s0).all())
        self.assertTrue(torch.allclose(s0, torch.full_like(s0, 1e-35), rtol=1e-4))
        yp_bar, yd_bar, _ = apply_gauge_normalization(yp, yd)
        rms_bar = (yp_bar.abs() ** 2).mean(dim=-1).sqrt()
        self.assertTrue(torch.allclose(rms_bar, torch.ones_like(rms_bar), atol=1e-4))
        self.assertTrue(torch.isfinite(yd_bar).all())

    def test_channel_rescaling_exactness(self):
        h_bar = torch.ones(2, 32, dtype=torch.complex64)
        s0 = torch.tensor([[2.5], [4.0]])
        h_phys = rescale_channel_estimate(h_bar, s0)
        self.assertTrue(torch.allclose(h_phys[0].abs(), torch.tensor(2.5)))
        self.assertTrue(torch.allclose(h_phys[1].abs(), torch.tensor(4.0)))

    def test_scale_invariance_arbitrary_gain(self):
        model = EMFloatFormer()
        model.eval()
        yp = torch.randn(2, 32, dtype=torch.complex64)
        yd = torch.randn(2, 32, dtype=torch.complex64)
        nv = torch.tensor([[0.01], [0.01]])
        with torch.no_grad():
            out1 = model(yp, yd, nv)
            out2 = model(yp * 4096.0, yd * 4096.0, nv * (4096.0 ** 2))
        max_diff = (out1["llrs"] - out2["llrs"]).abs().max().item()
        self.assertLess(max_diff, 1e-5)

    def test_zero_power_guard(self):
        yp_zero = torch.zeros(2, 32, dtype=torch.complex64)
        yd_zero = torch.zeros(2, 32, dtype=torch.complex64)
        yp_bar, yd_bar, s0 = apply_gauge_normalization(yp_zero, yd_zero)
        self.assertFalse(torch.isnan(yp_bar).any())
        self.assertFalse(torch.isnan(yd_bar).any())
        self.assertTrue(torch.all(s0 > 0.0))

    def test_audit_single_scale_and_phase(self):
        err, flips = audit_single_scale(scale=2.0 ** (-12))
        self.assertLess(err, 1e-5)
        self.assertEqual(flips, 0)
        flips_pi = audit_single_phase(phase=math.pi)
        self.assertEqual(flips_pi, 0)


class TestF9HermitianGreenAttention(unittest.TestCase):
    """F9: Hermitian Green's Propagator Attention."""

    def setUp(self):
        torch.set_num_threads(4)

    def test_complex_linear_zero_bias(self):
        layer = ComplexLinear(in_features=32, out_features=32, bias=False)
        self.assertIsNone(layer.bias)

    def test_complex_linear_homogeneity(self):
        layer = ComplexLinear(in_features=16, out_features=16, bias=False)
        z = torch.randn(2, 10, 16, dtype=torch.complex64)
        c = 2.5 * torch.exp(1j * torch.tensor(0.7))
        out1 = layer(c * z)
        out2 = c * layer(z)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-5))

    def test_relative_phase_rope(self):
        rope = ComplexRoPE(dim=16)
        q = torch.randn(1, 8, 16, dtype=torch.complex64)
        q_rot = rope(q)
        self.assertEqual(q_rot.shape, q.shape)

    def test_rope_maximum_separation(self):
        rope = ComplexRoPE(dim=16)
        phasor = rope.get_phasor(dist=31)
        self.assertAlmostEqual(float(phasor.abs().mean()), 1.0, places=5)

    def test_hermitian_inner_product_phase_cancellation(self):
        q = torch.randn(2, 4, 16, dtype=torch.complex64)
        k = torch.randn(2, 4, 16, dtype=torch.complex64)
        phi = 1.234
        phase = torch.exp(1j * torch.tensor(phi))
        s1 = hermitian_attention_scores(q, k)
        s2 = hermitian_attention_scores(phase * q, phase * k)
        self.assertTrue(torch.allclose(s1, s2, atol=1e-5))

    def test_radial_activation_u1_equivariance(self):
        act = RadialActivation()
        z = torch.randn(4, 16, dtype=torch.complex64)
        phi = 0.88
        phase = torch.exp(1j * torch.tensor(phi))
        out1 = act(phase * z)
        out2 = phase * act(z)
        self.assertTrue(torch.allclose(out1, out2, atol=1e-5))

    def test_radial_activation_zero_and_extreme(self):
        act = RadialActivation()
        z_zero = torch.zeros(2, 4, dtype=torch.complex64)
        out_zero = act(z_zero)
        self.assertTrue(torch.all(out_zero == 0.0))
        self.assertFalse(torch.isnan(out_zero).any())

        z_huge = torch.full((2, 4), 1e6, dtype=torch.complex64)
        out_huge = act(z_huge)
        self.assertFalse(torch.isinf(out_huge).any())

    def test_green_attention_single_subcarrier(self):
        attn = GreenAttention(d_model=16, n_heads=1)
        z = torch.randn(1, 1, 16, dtype=torch.complex64)
        out, weights = attn(z, return_weights=True)
        self.assertAlmostEqual(float(weights[0, 0, 0, 0].detach()), 1.0, places=5)

    def test_green_attention_identical_keys_uniform(self):
        attn = GreenAttention(d_model=16, n_heads=1)
        q = torch.randn(1, 4, 16, dtype=torch.complex64)
        k = torch.ones(1, 4, 16, dtype=torch.complex64)
        weights = attn.compute_weights(q, k)
        self.assertTrue(torch.allclose(weights, torch.full_like(weights, 0.25), atol=1e-3))

    def test_green_attention_u1_and_cstar_equivariance(self):
        attn = GreenAttention(d_model=16, n_heads=2)
        norm = ComplexRMSNorm(d_model=16)
        z = torch.randn(2, 12, 16, dtype=torch.complex64)
        phi = 0.95
        c_u1 = torch.exp(1j * torch.tensor(phi))
        out1 = attn(z)
        out2 = attn(c_u1 * z)
        self.assertTrue(torch.allclose(out2, c_u1 * out1, atol=1e-5))

        c_star = 3.7 * c_u1
        out_n1 = attn(norm(z))
        out_n2 = attn(norm(c_star * z))
        self.assertTrue(torch.allclose(out_n2, c_u1 * out_n1, atol=1e-5))

    def test_complex_rope_relative_shift_invariance(self):
        rope = ComplexRoPE(dim=16)
        q = torch.randn(2, 8, 16, dtype=torch.complex64)
        k = torch.randn(2, 8, 16, dtype=torch.complex64)
        s0 = hermitian_attention_scores(rope(q, offset=0), rope(k, offset=0))
        s_shifted = hermitian_attention_scores(rope(q, offset=11), rope(k, offset=11))
        self.assertTrue(torch.allclose(s0, s_shifted, atol=1e-5))

    def test_radial_activation_zero_gradient_finite(self):
        for kind in ("silu", "gelu", "relu", "tanh"):
            act = RadialActivation(kind=kind)
            z0 = torch.zeros(2, 4, dtype=torch.complex64, requires_grad=True)
            out0 = act(z0)
            self.assertTrue(torch.isfinite(out0).all(), f"{kind} zero forward non-finite")
            (out0.real.sum() + out0.imag.sum()).backward()
            self.assertIsNotNone(z0.grad)
            self.assertTrue(torch.isfinite(z0.grad).all(), f"{kind} zero grad non-finite")

            z_huge = torch.full((2, 4), 1e20 + 1e20j, dtype=torch.complex64, requires_grad=True)
            out_huge = act(z_huge)
            self.assertTrue(torch.isfinite(out_huge).all(), f"{kind} 1e20 forward non-finite")
            (out_huge.real.sum() + out_huge.imag.sum()).backward()
            self.assertIsNotNone(z_huge.grad)
            self.assertTrue(torch.isfinite(z_huge.grad).all(), f"{kind} 1e20 grad non-finite")


class TestF10BidirectionalAttention(unittest.TestCase):
    """F10: Bidirectional Pilot-Data Cross-Attention."""

    def setUp(self):
        torch.set_num_threads(4)

    def test_joint_embedding_shapes(self):
        emb = JointEmbedding(n=32, d_model=64)
        yp = torch.randn(2, 32, dtype=torch.complex64)
        yd = torch.randn(2, 32, dtype=torch.complex64)
        tokens = emb(yp, yd)
        self.assertEqual(tokens.shape, (2, 64, 64))

    def test_cross_attention_shapes_and_gradient(self):
        block = BidirectionalBlock(d_model=32, n_heads=2)
        p_in = torch.randn(2, 16, 32, dtype=torch.complex64, requires_grad=True)
        d_in = torch.randn(2, 16, 32, dtype=torch.complex64, requires_grad=True)
        p_out, d_out = block(p_in, d_in)
        self.assertEqual(p_out.shape, p_in.shape)
        self.assertEqual(d_out.shape, d_in.shape)

        loss_d = (d_out.abs() ** 2).sum()
        loss_d.backward(retain_graph=True)
        self.assertIsNotNone(p_in.grad)
        self.assertGreater(float(p_in.grad.abs().sum()), 0.0)

        loss_p = (p_out.abs() ** 2).sum()
        loss_p.backward()
        self.assertIsNotNone(d_in.grad)
        self.assertGreater(float(d_in.grad.abs().sum()), 0.0)

    def test_boundary_token_conditions(self):
        block = BidirectionalBlock(d_model=16, n_heads=2)
        # All pilot, no data
        p = torch.randn(1, 8, 16, dtype=torch.complex64)
        d = torch.zeros(1, 8, 16, dtype=torch.complex64)
        p_out, _ = block(p, d)
        self.assertFalse(torch.isnan(p_out).any())

        # Minimal pilot, single tone
        p_single = torch.randn(1, 1, 16, dtype=torch.complex64)
        d_multi = torch.randn(1, 15, 16, dtype=torch.complex64)
        p_out_s, d_out_m = block(p_single, d_multi)
        self.assertEqual(d_out_m.shape, (1, 15, 16))

    def test_multi_layer_depth(self):
        for depth in [1, 2, 4]:
            model = EMFloatFormer(depth=depth)
            self.assertEqual(len(model.blocks), depth)

    def test_pilot_vs_bidirectional_ber_gain(self):
        gain_db = compare_pilot_vs_bidirectional()
        self.assertGreaterEqual(gain_db, 0.5)

    def test_noise_feedback_and_perfect_pilot(self):
        deg_db = evaluate_noise_data_feedback()
        self.assertGreater(deg_db, 0.0)
        self.assertLess(deg_db, 1.0)
        ber = evaluate_perfect_pilot_demapping()
        self.assertLess(ber, 1e-3)

    def test_create_floatformer_factory(self):
        loss_fn = JointPhysicalLoss()
        frame = generate_frame(batch_size=2, n=32, snr_db=18.0, pa_condition="strong",
                               generator=torch.Generator().manual_seed(777))
        for g in [True, False]:
            for h in [True, False]:
                for j in [True, False]:
                    m = create_floatformer(
                        use_gauge_norm=g,
                        use_hermitian_attention=h,
                        use_joint_pilot_data=j,
                    )
                    out = m(frame.yp, frame.yd, frame.noise_var)
                    self.assertEqual(out["llrs"].shape, (2, 32, 4))
                    self.assertTrue(torch.isfinite(out["llrs"]).all())
                    loss = loss_fn(out, frame, A=1.25)
                    loss.backward()
                    bad_grads = [
                        name for name, p in m.named_parameters()
                        if p.grad is None or not torch.isfinite(p.grad).all() or float(p.grad.abs().sum()) == 0.0
                    ]
                    self.assertEqual(bad_grads, [], f"Config ({g},{h},{j}) bad grads: {bad_grads}")


class TestF11JointPhysicalLoss(unittest.TestCase):
    """F11: Joint Physical Consistency Loss."""

    def setUp(self):
        torch.set_num_threads(4)

    def test_calibrated_llr_bce_loss(self):
        loss_fn = JointPhysicalLoss()
        llrs = torch.randn(4, 32, 4)
        bits = torch.randint(0, 2, (4, 32, 4))
        loss = loss_fn.bce_loss(llrs, bits)
        self.assertGreater(float(loss), 0.0)

        # Bounded on extreme negative logits
        llrs_extreme = torch.full((2, 32, 4), -20.0)
        bits_ones = torch.ones(2, 32, 4, dtype=torch.int64)
        loss_bounded = loss_fn.bce_loss(llrs_extreme, bits_ones)
        self.assertFalse(torch.isnan(loss_bounded))
        self.assertFalse(torch.isinf(loss_bounded))

    def test_nmse_loss_scale_invariance(self):
        loss_fn = JointPhysicalLoss()
        h_true = torch.randn(2, 32, dtype=torch.complex64)
        h_est = h_true * 1.1
        loss1 = loss_fn.nmse_loss(h_est, h_true)
        loss2 = loss_fn.nmse_loss(h_est * 5.0, h_true * 5.0)
        self.assertAlmostEqual(float(loss1), float(loss2), places=5)

        loss_zero = loss_fn.nmse_loss(h_true, h_true)
        self.assertAlmostEqual(float(loss_zero), 0.0, places=5)

    def test_pa_residual_consistency_loss(self):
        loss_fn = JointPhysicalLoss()
        yd = torch.randn(2, 32, dtype=torch.complex64)
        h_est = torch.ones(2, 32, dtype=torch.complex64)
        x_est = torch.ones(2, 32, dtype=torch.complex64)
        res_loss = loss_fn.pa_residual_loss(yd, h_est, x_est, A=1.25)
        self.assertGreaterEqual(float(res_loss), 0.0)

        # Zero received data protection
        yd_zero = torch.zeros(2, 32, dtype=torch.complex64)
        res_zero = loss_fn.pa_residual_loss(yd_zero, h_est, torch.zeros(2, 32, dtype=torch.complex64))
        self.assertFalse(torch.isnan(res_zero))

    def test_weighting_schedule(self):
        loss_fn = JointPhysicalLoss(lambda_nmse=0.5, lambda_pa=0.2)
        self.assertEqual(loss_fn.lambda_nmse, 0.5)
        self.assertEqual(loss_fn.lambda_pa, 0.2)

    def test_loss_gradient_invariance(self):
        max_diff = verify_loss_gradient_invariance()
        self.assertLess(max_diff, 1e-5)

    def test_loss_scale_invariance_audit(self):
        max_rel_diff = audit_loss_scale_invariance()
        self.assertLess(max_rel_diff, 1e-5)

    def test_train_step_joint_loss(self):
        success = train_step_joint_loss()
        self.assertTrue(success)

    def test_all_params_nonzero_gradient_step1(self):
        from p4.simulator import generate_frame
        model = EMFloatFormer(depth=2)
        self.assertGreater(float(model.head_h.weight_r.detach().abs().sum()), 0.0)
        self.assertGreater(float(model.head_dist.weight_r.detach().abs().sum()), 0.0)
        frame = generate_frame(batch_size=4, n=32, snr_db=18.0, pa_condition="strong")
        loss = JointPhysicalLoss()(model(frame.yp, frame.yd, frame.noise_var), frame)
        loss.backward()
        zero_grads = [n for n, p in model.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0.0]
        self.assertEqual(len(zero_grads), 0, f"Zero-grad params on step 1: {zero_grads}")

    def test_pa_residual_loss_ortho_fft_parity(self):
        loss_fn = JointPhysicalLoss()
        x_est = torch.ones(2, 32, dtype=torch.complex64)
        # With norm='ortho', ifft(ones) has energy 1.0 across time domain (Parseval)
        x_time = torch.fft.ifft(x_est, dim=-1, norm="ortho")
        self.assertAlmostEqual(float((x_time.abs() ** 2).sum(dim=-1).mean()), 32.0, places=4)
        h_est = torch.ones(2, 32, dtype=torch.complex64)
        r2 = x_time.real ** 2 + x_time.imag ** 2
        r = torch.sqrt(r2 + 1e-24)
        phi = 0.5 * r2 / (1.0 + r2)
        x_pa = x_time * ((1.0 + (r / 1.25) ** 4) ** (-0.25)) * torch.complex(torch.cos(phi), torch.sin(phi))
        yd_exact = h_est * torch.fft.fft(x_pa, dim=-1, norm="ortho")
        self.assertAlmostEqual(float(loss_fn.pa_residual_loss(yd_exact, h_est, x_est, A=1.25)), 0.0, places=5)


class TestF12NumericalEquivariance(unittest.TestCase):
    """F12: Numerical Equivariance Verification."""

    def setUp(self):
        torch.set_num_threads(4)

    def test_non_power_of_two_scale_phase_active_backbone(self):
        model = EMFloatFormer(depth=2)
        model.eval()
        g = torch.Generator().manual_seed(42)
        yp = torch.randn(4, 32, dtype=torch.complex64, generator=g)
        yd = torch.randn(4, 32, dtype=torch.complex64, generator=g)
        nv = torch.full((4, 1), 0.01)
        c = 3.7 * torch.exp(1j * torch.tensor(0.95))
        with torch.no_grad():
            o1 = model(yp, yd, nv)
            o2 = model(yp * c, yd * c, nv * (3.7 ** 2))
        self.assertLess((o1["llrs"] - o2["llrs"]).abs().max().item(), 1e-5)
        self.assertEqual(((o1["llrs"] > 0) != (o2["llrs"] > 0)).sum().item(), 0)
        h_rel = (o2["h_hat"] - c * o1["h_hat"]).abs().max().item() / (3.7 * o1["h_hat"].abs().max().item())
        self.assertLess(h_rel, 1e-5)

    def test_gain_invariance_24_octaves(self):
        max_err, flips = audit_gain_equivariance(octaves=24)
        self.assertLess(max_err, 1e-5)
        self.assertEqual(flips, 0)

    def test_phase_invariance_circle(self):
        max_err, flips = audit_phase_equivariance(num_phases=16)
        self.assertLess(max_err, 1e-5)
        self.assertEqual(flips, 0)

    def test_simultaneous_c_star(self):
        max_err, flips = audit_c_star_equivariance(num_samples=20)
        self.assertLess(max_err, 1e-5)
        self.assertEqual(flips, 0)

    def test_channel_estimate_c_equivariance(self):
        rel_err = audit_channel_equivariance()
        self.assertLess(rel_err, 1e-5)

    def test_dense_phase_sweep(self):
        max_err = sweep_phases_dense(num_points=100)
        self.assertLess(max_err, 1e-5)

    def test_irrational_phases(self):
        flips = check_irrational_phases()
        self.assertEqual(flips, 0)

    def test_noise_variance_quad_scaling(self):
        max_diff = check_noise_quad_scaling(r=3.5)
        self.assertLess(max_diff, 1e-5)

    def test_cross_feature_verifiers(self):
        self.assertTrue(verify_gauge_attention_joint())
        self.assertTrue(verify_bidirectional_green_prop())
        err, flips = audit_full_equivariance()
        self.assertLess(err, 1e-5)
        self.assertEqual(flips, 0)
        self.assertTrue(run_standalone_equivariance_check())

    def test_fresh_seed11_and_trained_depth2_c_star_equivariance(self):
        # 1. Fresh seed=11 EMFloatFormer(depth=2)
        torch.manual_seed(11)
        fresh_model = EMFloatFormer(depth=2)
        fresh_model.eval()
        fr = generate_frame(batch_size=8, n=32, snr_db=20.0, pa_condition="strong",
                            generator=torch.Generator().manual_seed(1101))
        transforms = [
            (2.0 ** -12, 2.7183),
            (0.13, -1.7),
            (3.7, 0.95),
            (1023.5, 2.5356),
            (2.0 ** 12, -0.6076),
        ]
        with torch.no_grad():
            o1 = fresh_model(fr.yp, fr.yd, fr.noise_var)
            for r, phi in transforms:
                c = r * torch.exp(1j * torch.tensor(phi))
                o2 = fresh_model(fr.yp * c, fr.yd * c, fr.noise_var * (r ** 2))
                llr_err = (o1["llrs"] - o2["llrs"]).abs().max().item()
                flips = ((o1["llrs"] > 0) != (o2["llrs"] > 0)).sum().item()
                h_rel = (o2["h_hat"] - c * o1["h_hat"]).abs().max().item() / (
                    r * (o1["h_hat"].abs().max().item() + 1e-12)
                )
                self.assertLess(llr_err, 1e-5)
                self.assertEqual(flips, 0)
                self.assertLess(h_rel, 1e-5)

        # 2. Trained EMFloatFormer(depth=2)
        torch.manual_seed(11)
        trained_model = EMFloatFormer(depth=2)
        opt = torch.optim.Adam([
            {"params": [p for n, p in trained_model.named_parameters() if "gamma_eff" not in n], "lr": 1e-2},
            {"params": trained_model.gamma_eff.parameters(), "lr": 3e-2},
        ])
        lf = JointPhysicalLoss(lambda_nmse=1.0, lambda_pa=0.25)
        g_tr = torch.Generator().manual_seed(202)
        for s in range(40):
            pa = ("strong", "mild", "clean")[s % 3]
            tf = generate_frame(16, 32, snr_db=(10.0, 25.0), pa_condition=pa, generator=g_tr)
            opt.zero_grad()
            lf(trained_model(tf.yp, tf.yd, tf.noise_var), tf, A=pa).backward()
            opt.step()
        trained_model.eval()
        with torch.no_grad():
            o1 = trained_model(fr.yp, fr.yd, fr.noise_var)
            for r, phi in transforms:
                c = r * torch.exp(1j * torch.tensor(phi))
                o2 = trained_model(fr.yp * c, fr.yd * c, fr.noise_var * (r ** 2))
                llr_err = (o1["llrs"] - o2["llrs"]).abs().max().item()
                flips = ((o1["llrs"] > 0) != (o2["llrs"] > 0)).sum().item()
                h_rel = (o2["h_hat"] - c * o1["h_hat"]).abs().max().item() / (
                    r * (o1["h_hat"].abs().max().item() + 1e-12)
                )
                self.assertLess(llr_err, 1e-5)
                self.assertEqual(flips, 0)
                self.assertLess(h_rel, 1e-5)

    def test_challenger_m2_3_50step_seed11_30db_strong_256frames(self):
        torch.manual_seed(11)
        model = EMFloatFormer(depth=2)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        lf = JointPhysicalLoss(lambda_nmse=1.0, lambda_pa=0.25)
        tf = generate_frame(
            16, 32, snr_db=18.0, pa_condition="strong",
            generator=torch.Generator().manual_seed(99),
        )
        for _ in range(50):
            opt.zero_grad()
            lf(model(tf.yp, tf.yd, tf.noise_var), tf).backward()
            opt.step()

        model.eval()
        fr = generate_frame(
            256, 32, snr_db=30.0, pa_condition="strong",
            generator=torch.Generator().manual_seed(31415),
        )
        transforms = [
            (2.0 ** -12, 2.7183),
            (0.13, -1.7),
            (3.7, 0.95),
            (1023.5, 2.5356),
            (4096.0, -0.6076),
        ]
        with torch.no_grad():
            o1_c64 = model(fr.yp, fr.yd, fr.noise_var)
            yp128 = fr.yp.to(torch.complex128)
            yd128 = fr.yd.to(torch.complex128)
            nv128 = fr.noise_var.to(torch.float64)
            o1_c128 = model(yp128, yd128, nv128)

            for r, phi in transforms:
                c64 = r * torch.exp(1j * torch.tensor(phi, dtype=torch.float32))
                o2_c64 = model(fr.yp * c64, fr.yd * c64, fr.noise_var * (r ** 2))
                err_c64 = (o1_c64["llrs"] - o2_c64["llrs"]).abs().max().item()
                flips_c64 = ((o1_c64["llrs"] > 0) != (o2_c64["llrs"] > 0)).sum().item()
                h_rel_c64 = (o2_c64["h_hat"] - c64 * o1_c64["h_hat"]).abs().max().item() / (
                    r * (o1_c64["h_hat"].abs().max().item() + 1e-12)
                )
                self.assertLess(err_c64, 1e-5, f"c64 LLR err too high at r={r}: {err_c64}")
                self.assertEqual(flips_c64, 0, f"c64 bit flips at r={r}: {flips_c64}")
                self.assertLess(h_rel_c64, 1e-5, f"c64 h_rel too high at r={r}: {h_rel_c64}")

                c128 = r * torch.exp(1j * torch.tensor(phi, dtype=torch.float64))
                o2_c128 = model(yp128 * c128, yd128 * c128, nv128 * (r ** 2))
                err_c128 = (o1_c128["llrs"] - o2_c128["llrs"]).abs().max().item()
                flips_c128 = ((o1_c128["llrs"] > 0) != (o2_c128["llrs"] > 0)).sum().item()
                h_rel_c128 = (o2_c128["h_hat"] - c128 * o1_c128["h_hat"]).abs().max().item() / (
                    r * (o1_c128["h_hat"].abs().max().item() + 1e-12)
                )
                self.assertLess(err_c128, 1e-11, f"c128 LLR err too high at r={r}: {err_c128}")
                self.assertEqual(flips_c128, 0, f"c128 bit flips at r={r}: {flips_c128}")
                self.assertLess(h_rel_c128, 1e-11, f"c128 h_rel too high at r={r}: {h_rel_c128}")


class TestM2CalibrationAndBaselines(unittest.TestCase):
    """M2.2 & M2.3: LLR Calibration, Mixed-PA Loss Scale Stability, and Comparative Baselines."""

    def setUp(self):
        torch.set_num_threads(4)

    def test_calibrated_llr_dynamic_range_ece_and_equivariance_high_snr(self):
        torch.manual_seed(11)
        model = EMFloatFormer(depth=2)
        opt = torch.optim.Adam([
            {"params": [p for n, p in model.named_parameters() if "gamma_eff" not in n], "lr": 1e-2},
            {"params": model.gamma_eff.parameters(), "lr": 3e-2},
        ])
        lf = JointPhysicalLoss(lambda_nmse=1.0, lambda_pa=0.25)
        g_tr = torch.Generator().manual_seed(101)
        for s in range(80):
            pa = ("clean", "mild", "strong")[s % 3]
            tf = generate_frame(32, 32, snr_db=(10.0, 25.0), pa_condition=pa, generator=g_tr)
            opt.zero_grad()
            lf(model(tf.yp, tf.yd, tf.noise_var), tf, A=pa).backward()
            opt.step()

        model.eval()
        transforms = [(2.0 ** -12, 2.7183), (0.13, -1.7), (3.7, 0.95), (1023.5, 2.5356), (4096.0, -0.6076)]
        with torch.no_grad():
            for pa in ("clean", "mild", "strong"):
                for snr in (15.0, 20.0, 30.0):
                    fr = generate_frame(128, 32, snr_db=snr, pa_condition=pa,
                                        generator=torch.Generator().manual_seed(31415))
                    o1 = model(fr.yp, fr.yd, fr.noise_var)
                    ece = compute_ece(o1["llrs"], fr.bits, n_bins=10)
                    self.assertLess(ece, 0.05, f"ECE too high for {pa} {snr}dB: {ece}")
                    if pa == "clean" and snr == 30.0:
                        self.assertGreater(float(o1["llrs"].abs().max().item()), 15.0)
                    for r, phi in transforms:
                        c = r * torch.exp(1j * torch.tensor(phi))
                        o2 = model(fr.yp * c, fr.yd * c, fr.noise_var * (r ** 2))
                        err = (o1["llrs"] - o2["llrs"]).abs().max().item()
                        flips = ((o1["llrs"] > 0) != (o2["llrs"] > 0)).sum().item()
                        h_rel = (o2["h_hat"] - c * o1["h_hat"]).abs().max().item() / (
                            r * (o1["h_hat"].abs().max().item() + 1e-12)
                        )
                        self.assertLess(err, 1e-5)
                        self.assertEqual(flips, 0)
                        self.assertLess(h_rel, 1e-5)

    def test_joint_loss_mixed_pa_resolution_and_scale_stability(self):
        loss_fn = JointPhysicalLoss(lambda_nmse=1.0, lambda_pa=0.25)
        g = torch.Generator().manual_seed(55)
        fr_plain = generate_frame(4, 32, snr_db=20.0, pa_condition="strong", generator=g)
        self.assertEqual(loss_fn._resolve_pa_A(fr_plain), 1.25)
        self.assertIsNone(loss_fn._resolve_pa_A(fr_plain, A=float("inf")))
        self.assertIsNone(loss_fn._resolve_pa_A(fr_plain, A="clean"))
        self.assertEqual(loss_fn._resolve_pa_A(fr_plain, A="mild"), 2.25)
        self.assertEqual(loss_fn._resolve_pa_A(fr_plain, A=" Strong "), 1.25)
        self.assertRaises(ValueError, loss_fn._resolve_pa_A, fr_plain, A="unknown")

        fr_clean = generate_frame(4, 32, snr_db=20.0, pa_condition="clean", generator=g)
        fr_clean.pa_condition = "clean"
        fr_mild = generate_frame(4, 32, snr_db=20.0, pa_condition="mild", generator=g)
        fr_mild.A = 2.25
        fr_inf = generate_frame(4, 32, snr_db=20.0, pa_condition="clean", generator=g)
        fr_inf.A = float("inf")
        fr_strong = generate_frame(4, 32, snr_db=20.0, pa_condition="strong", generator=g)
        fr_strong.pa_condition = "strong"
        fr_bad = generate_frame(4, 32, snr_db=20.0, pa_condition="clean", generator=g)
        fr_bad.pa_condition = "bad"
        self.assertIsNone(loss_fn._resolve_pa_A(fr_clean))
        self.assertEqual(loss_fn._resolve_pa_A(fr_mild), 2.25)
        self.assertIsNone(loss_fn._resolve_pa_A(fr_inf))
        self.assertEqual(loss_fn._resolve_pa_A(fr_strong), 1.25)
        self.assertRaises(ValueError, loss_fn._resolve_pa_A, fr_bad)

        x_est = torch.ones(2, 32, dtype=torch.complex64)
        h_est = torch.ones(2, 32, dtype=torch.complex64)
        pa_none = loss_fn.pa_residual_loss(fr_plain.yd[:2], h_est, x_est, A=None)
        pa_inf = loss_fn.pa_residual_loss(fr_plain.yd[:2], h_est, x_est, A=float("inf"))
        self.assertTrue(torch.isfinite(pa_inf))
        self.assertGreaterEqual(float(pa_inf), 0.0)
        self.assertAlmostEqual(float(pa_inf), float(pa_none), places=6)

        torch.manual_seed(7)
        model = EMFloatFormer(depth=2)
        opt = torch.optim.Adam([
            {"params": [p for n, p in model.named_parameters() if "gamma_eff" not in n], "lr": 1e-2},
            {"params": model.gamma_eff.parameters(), "lr": 3e-2},
        ])
        scale_ratios = []
        for s in range(60):
            pa = ("strong", "mild", "clean")[s % 3]
            tf = generate_frame(32, 32, snr_db=(10.0, 25.0), pa_condition=pa, generator=g)
            tf.pa_condition = pa
            opt.zero_grad()
            out = model(tf.yp, tf.yd, tf.noise_var)
            loss_fn(out, tf).backward()
            opt.step()
            with torch.no_grad():
                sr = (out["h_hat"].abs().norm(dim=-1) / tf.h_true.abs().norm(dim=-1).clamp(min=1e-12)).mean().item()
                scale_ratios.append(sr)
        final_sr = sum(scale_ratios[-12:]) / 12.0
        self.assertLess(abs(final_sr - 1.0), 0.08, f"Mixed-PA scale_ratio drifted: {final_sr}")

    def test_comparative_baseline_models_equivariance_and_params(self):
        # 1. ComplexConv1d: bias=None and C* homogeneity
        conv = ComplexConv1d(in_channels=4, out_channels=4, kernel_size=5)
        self.assertIsNone(conv.bias)
        z_conv = torch.randn(2, 4, 32, dtype=torch.complex64, generator=torch.Generator().manual_seed(71))
        c = 3.7 * torch.exp(1j * torch.tensor(0.95))
        self.assertTrue(torch.allclose(conv(c * z_conv), c * conv(z_conv), atol=1e-5))

        # 2. LocalEquivConv16QAM: parameter parity, step-1 non-zero grads, 1D input, trained 24-octave equivariance
        emff_params = sum(p.numel() for p in EMFloatFormer(d_model=12, n_heads=2, depth=2).parameters())
        torch.manual_seed(21)
        local_model = LocalEquivConv16QAM()
        local_params = sum(p.numel() for p in local_model.parameters())
        self.assertLess(abs(local_params - emff_params) / emff_params, 0.05)

        fr = generate_frame(4, 32, snr_db=20.0, pa_condition="strong",
                            generator=torch.Generator().manual_seed(808))
        out_local_1d = local_model(fr.yp[0], fr.yd[0], fr.noise_var[0])
        self.assertEqual(out_local_1d["llrs"].shape, (1, 32, 4))
        self.assertEqual(out_local_1d["h_hat"].shape, (1, 32))

        loss_fn = JointPhysicalLoss(lambda_nmse=1.0, lambda_pa=0.25)
        local_model.train()
        out_local_step1 = local_model(fr.yp, fr.yd, fr.noise_var)
        loss_fn(out_local_step1, fr, A=1.25).backward()
        bad_local_grads = [
            name for name, p in local_model.named_parameters()
            if p.grad is None or not torch.isfinite(p.grad).all() or float(p.grad.abs().sum()) == 0.0
        ]
        self.assertEqual(bad_local_grads, [], f"LocalEquivConv16QAM bad step-1 grads: {bad_local_grads}")

        opt_local = torch.optim.Adam([
            {"params": [p for n, p in local_model.named_parameters() if "gamma_eff" not in n], "lr": 3e-3},
            {"params": local_model.gamma_eff.parameters(), "lr": 1e-2},
        ])
        g_loc = torch.Generator().manual_seed(203)
        for step in range(20):
            pa = ("strong", "mild", "clean")[step % 3]
            tf = generate_frame(16, 32, snr_db=(10.0, 25.0), pa_condition=pa, generator=g_loc)
            opt_local.zero_grad()
            loss_fn(local_model(tf.yp, tf.yd, tf.noise_var), tf, A=pa).backward()
            opt_local.step()

        local_model.eval()
        transforms = [
            (2.0 ** -12, 2.7183),
            (0.13, -1.7),
            (3.7, 0.95),
            (1023.5, 2.5356),
            (2.0 ** 12, -0.6076),
        ]
        with torch.no_grad():
            o1 = local_model(fr.yp, fr.yd, fr.noise_var)
            for r, phi in transforms:
                c_t = r * torch.exp(1j * torch.tensor(phi))
                o2 = local_model(fr.yp * c_t, fr.yd * c_t, fr.noise_var * (r ** 2))
                llr_err = (o1["llrs"] - o2["llrs"]).abs().max().item()
                flips = ((o1["llrs"] > 0) != (o2["llrs"] > 0)).sum().item()
                h_rel = (o2["h_hat"] - c_t * o1["h_hat"]).abs().max().item() / (
                    r * (o1["h_hat"].abs().max().item() + 1e-12)
                )
                self.assertLess(llr_err, 1e-5, f"Trained LocalEquivConv LLR err at r={r}: {llr_err}")
                self.assertEqual(flips, 0, f"Trained LocalEquivConv bit flips at r={r}: {flips}")
                self.assertLess(h_rel, 1e-5, f"Trained LocalEquivConv h_rel at r={r}: {h_rel}")

        # 3. TinyJoint16QAM: step-1 non-zero finite grads, finite h_hat/dist_est, 1D input
        torch.manual_seed(22)
        tiny = TinyJoint16QAM()
        out_tiny_1d = tiny(fr.yp[0], fr.yd[0], fr.noise_var[0])
        self.assertEqual(out_tiny_1d["llrs"].shape, (1, 32, 4))
        self.assertEqual(out_tiny_1d["h_hat"].shape, (1, 32))
        self.assertEqual(out_tiny_1d["dist_est"].shape, (1, 32))

        out_tiny = tiny(fr.yp, fr.yd, fr.noise_var)
        self.assertEqual(out_tiny["llrs"].shape, (4, 32, 4))
        self.assertTrue(torch.isfinite(out_tiny["llrs"]).all())
        self.assertTrue(torch.isfinite(out_tiny["h_hat"]).all())
        self.assertTrue(torch.isfinite(out_tiny["dist_est"]).all())
        loss_fn(out_tiny, fr, A=1.25).backward()
        bad_tiny_grads = [
            name for name, p in tiny.named_parameters()
            if p.grad is None or not torch.isfinite(p.grad).all() or float(p.grad.abs().sum()) == 0.0
        ]
        self.assertEqual(bad_tiny_grads, [], f"TinyJoint16QAM bad step-1 grads: {bad_tiny_grads}")

    def test_rng_perturbation_determinism_benchmarks_and_diagnostics(self):
        torch.randn(17)
        d1 = evaluate_noise_data_feedback()
        p1 = evaluate_perfect_pilot_demapping()
        torch.randn(53)
        d2 = evaluate_noise_data_feedback()
        p2 = evaluate_perfect_pilot_demapping()
        self.assertEqual(d1, d2)
        self.assertEqual(p1, p2)

        r1 = run_p4a_comparative_benchmarks(
            train_steps=2,
            batch_size=4,
            eval_frames=8,
            snrs_db=(15.0,),
            pa_conditions=("strong",),
            seed=2026,
            output_dir=None,
        )
        torch.randn(17)
        r2 = run_p4a_comparative_benchmarks(
            train_steps=2,
            batch_size=4,
            eval_frames=8,
            snrs_db=(15.0,),
            pa_conditions=("strong",),
            seed=2026,
            output_dir=None,
        )
        self.assertEqual(r1["train_final_scale_ratio"], r2["train_final_scale_ratio"])
        self.assertEqual(r1["rows"], r2["rows"])

    def test_saved_p4a_comparative_benchmark_artifacts_parity_and_invariants(self):
        results_dir = Path("p4/results")
        json_path = results_dir / "p4a_comparative_benchmark.json"
        csv_path = results_dir / "p4a_comparative_benchmark.csv"
        self.assertTrue(json_path.exists(), f"Missing {json_path}")
        self.assertTrue(csv_path.exists(), f"Missing {csv_path}")

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["stage"], "P4-A")
        self.assertEqual(payload["profile"], "equal4")
        self.assertEqual(payload["n_sc"], 32)
        self.assertEqual(payload["train_steps"], 240)
        self.assertEqual(payload["batch_size"], 32)
        self.assertEqual(payload["eval_frames"], 256)
        self.assertEqual(payload["seed"], 2026)
        self.assertEqual(payload["num_threads"], 4)

        rows = payload["rows"]
        self.assertEqual(len(rows), 75)

        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            expected_fields = list(rows[0].keys())
            self.assertEqual(reader.fieldnames, expected_fields)
            csv_rows = list(reader)

        self.assertEqual(len(csv_rows), 75)
        for jrow, crow in zip(rows, csv_rows):
            for key in expected_fields:
                self.assertEqual(str(jrow[key]), crow[key], f"JSON/CSV mismatch on {key}")

        calibrated_methods = {"EMFloatFormer", "LocalEquivConv16QAM", "Bussgang-LMMSE", "DD-PA-Cancel"}
        ber_curves = {}
        for row in rows:
            self.assertEqual(row["total_bits"], 32768)
            self.assertGreaterEqual(row["bit_errors"], 0)
            self.assertLessEqual(row["bit_errors"], 32768)
            self.assertAlmostEqual(row["ber"], row["bit_errors"] / row["total_bits"], places=12)
            for metric in ("ber", "nmse", "bce", "ece", "scale_ratio"):
                self.assertTrue(math.isfinite(row[metric]), f"Non-finite {metric} in {row}")
            if row["method"] in calibrated_methods:
                self.assertLess(row["ece"], 0.05, f"ECE >= 0.05 for {row}")
            ber_curves.setdefault((row["pa_condition"], row["method"]), []).append(
                (row["snr_db"], row["ber"])
            )

        for (pa_cond, method), pts in ber_curves.items():
            pts_sorted = sorted(pts, key=lambda x: x[0])
            self.assertLess(
                pts_sorted[-1][1],
                pts_sorted[0][1],
                f"BER did not improve 10->30dB for ({pa_cond}, {method})",
            )
            for (s0, b0), (s1, b1) in zip(pts_sorted, pts_sorted[1:]):
                if method in calibrated_methods:
                    self.assertLessEqual(
                        b1, b0, f"BER non-monotonic for ({pa_cond}, {method}): {s0}dB={b0} -> {s1}dB={b1}"
                    )
                else:
                    # TinyJoint16QAM has a fixed +0.03 regularizer floor at 25-30 dB across independent eval seeds
                    self.assertLessEqual(
                        b1, b0 + 5e-4, f"BER regressed beyond noise floor for ({pa_cond}, {method}): {s0}dB={b0} -> {s1}dB={b1}"
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
