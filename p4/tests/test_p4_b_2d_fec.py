"""Unit Test Suite for Milestone 3 (Stage P4-B: Features F13–F17)."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace

import torch

from p4.slot_2d import (
    EMFloatFormer2D,
    JointPhysicalLoss2D,
    LDPCMatrix,
    SlotSimulator2D,
    _evaluate_slot_bler,
    _synchronize_slot,
    check_block_error,
    check_doppler_bler_sensitivity,
    check_edge_extrapolation,
    check_integer_accumulator,
    check_phase_continuity,
    compare_1d_vs_2d_lmmse,
    compensate_cfo_time,
    compute_2d_correlation,
    compute_ci,
    compute_doppler_spectrum,
    correct_single_bit,
    decode_min_sum,
    encode_ldpc,
    estimate_cfo_cp,
    estimate_cfo_cp_tensor,
    evaluate_bler_curve,
    evaluate_bler_point,
    evaluate_ood_profile_bler,
    generate_dmrs,
    get_custom_dmrs_mask,
    get_dmrs_mask,
    get_slot_masks,
    idd_pa_2d,
    idd_pa_2d_cleanup,
    idd_pa_2d_clipping,
    idd_pa_2d_linear,
    lmmse_2d_flat,
    lmmse_2d_interpolate,
    lmmse_2d_singular,
    measure_cfo_phase_drift,
    measure_coding_gain,
    measure_ici_leakage,
    run_block_counter,
    run_p4b_coded_bler_benchmarks,
    simulate_single_block,
    simulate_slot_channel,
)


class TestF13Slot2DAndDMRS(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)

    def test_dmrs_mask_overhead_and_partition(self):
        for n_sc in (32, 64, 128):
            p_mask, d_mask = get_slot_masks(n_symb=14, n_sc=n_sc)
            self.assertEqual(p_mask.shape, (14, n_sc))
            self.assertFalse((p_mask & d_mask).any())
            self.assertTrue((p_mask | d_mask).all())
            overhead = float(p_mask.sum()) / float(p_mask.numel())
            self.assertLessEqual(overhead, 0.10)
            self.assertTrue(p_mask[2].any())
            self.assertTrue(p_mask[11].any())
            self.assertFalse(p_mask[0].any())

    def test_slot_simulator_boundaries_and_exp16(self):
        for n_symb, n_sc in ((1, 32), (14, 32), (28, 32), (14, 64), (14, 128)):
            sim = SlotSimulator2D(n_symb=n_symb, n_sc=n_sc, profile="exp16")
            grid = sim.generate_empty_grid(batch_size=2)
            self.assertEqual(grid.shape, (2, n_symb, n_sc))
            self.assertEqual(grid.dtype, torch.complex64)

        sim_full = SlotSimulator2D(n_symb=14, n_sc=32, pilot_ratio=1.0)
        slot_full = sim_full.generate_slot(batch_size=2, snr_db=20.0, generator=torch.Generator().manual_seed(7))
        self.assertEqual(slot_full.bits.shape, (2, 0, 4))
        self.assertEqual(slot_full.y_grid.shape, (2, 14, 32))

        mask_single = get_custom_dmrs_mask([(2, 0)], n_symb=14, n_sc=32)
        self.assertEqual(int(mask_single.sum()), 1)
        self.assertTrue(bool(mask_single[2, 0]))

    def test_generate_dmrs_unit_energy(self):
        dmrs = generate_dmrs(batch_size=4, num_pilots=32, seed=123)
        self.assertEqual(dmrs.shape, (4, 32))
        self.assertEqual(dmrs.dtype, torch.complex64)
        self.assertTrue(torch.allclose(dmrs.abs() ** 2, torch.ones_like(dmrs.abs()), atol=1e-5))


class TestF14ContinuousDopplerAndCFO(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)

    def test_static_limit_parity(self):
        gen = torch.Generator().manual_seed(99)
        sim = SlotSimulator2D(n_symb=14, n_sc=32, cp=16, profile="exp16", fd=0.0, cfo=0.0)
        slot = sim.generate_slot(batch_size=3, snr_db=None, pa_condition="clean", generator=gen)
        expected = slot.h_true * slot.x_grid
        rel_err = float(
            torch.linalg.norm(slot.y_grid - expected) / torch.linalg.norm(expected).clamp(min=1e-12)
        )
        self.assertLess(rel_err, 1e-5)

    def test_doppler_ici_and_cfo_continuity(self):
        self.assertLess(measure_ici_leakage(fd=0.0), 1e-10)
        self.assertGreater(measure_ici_leakage(fd=0.05), 1e-4)
        self.assertAlmostEqual(measure_cfo_phase_drift(cfo=0.04), 0.04, places=3)
        self.assertAlmostEqual(measure_cfo_phase_drift(cfo=-0.04), -0.04, places=3)
        self.assertAlmostEqual(estimate_cfo_cp(cfo_true=0.03), 0.03, places=2)
        self.assertTrue(check_phase_continuity(cfo=0.04))

        spec = compute_doppler_spectrum(speed_kmh=500.0, fc=3.5e9)
        self.assertGreater(spec["fd_hz"], 1500.0)

        h_evol = simulate_slot_channel(batch_size=1, fd=0.0, cfo=0.0, return_h_time=True)
        self.assertLess((h_evol[:, 1:] - h_evol[:, :-1]).abs().max().item(), 1e-6)


class TestF15NativeRateHalfLDPC(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)

    def test_girth_and_systematic_parity(self):
        ldpc = LDPCMatrix(codeword_len=128)
        h = ldpc.h_matrix
        g = ldpc.g_matrix
        self.assertEqual(h.shape, (64, 128))
        self.assertEqual(g.shape, (64, 128))

        # Verify H @ G^T == 0 (mod 2)
        hg = torch.remainder(torch.matmul(h.float(), g.t().float()), 2.0)
        self.assertTrue(torch.all(hg == 0))

        # Verify girth >= 6 (zero 4-cycles: off-diagonal entries of H^T @ H <= 1)
        gram = torch.matmul(h.t().float(), h.float())
        gram.fill_diagonal_(0.0)
        self.assertLessEqual(int(gram.max().item()), 1)

    def test_encode_decode_sign_and_coding_gain(self):
        u_zero = torch.zeros(2, 64, dtype=torch.int64)
        c_zero, h_mat = encode_ldpc(u_zero)
        self.assertTrue(torch.all(c_zero == 0))

        with self.assertRaises(ValueError):
            encode_ldpc(torch.empty(0, 0))

        gen = torch.Generator().manual_seed(17)
        u = torch.randint(0, 2, (4, 64), generator=gen, dtype=torch.int64)
        c, h_mat = encode_ldpc(u)
        # Positive LLR => bit 1, Negative LLR => bit 0
        clean_llrs = (2.0 * c.float() - 1.0) * 15.0
        u_hat, iters = decode_min_sum(clean_llrs, h_mat, return_iters=True)
        self.assertTrue(torch.equal(u_hat, u))
        self.assertEqual(iters, 1)

        noise_llrs = torch.randn(1, 128, generator=gen) * 0.01
        _, iters_noise = decode_min_sum(noise_llrs, h_mat, max_iter=15, return_iters=True)
        self.assertEqual(iters_noise, 15)

        self.assertTrue(correct_single_bit())
        self.assertGreater(measure_coding_gain(), 3.0)


class TestF16BaselinesAndEMFloatFormer2D(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)

    def test_2d_classical_baselines(self):
        r_2d = compute_2d_correlation(n_symb=14, n_sc=32, fd=0.03)
        self.assertEqual(r_2d.shape, (448, 448))
        h_grid = lmmse_2d_interpolate(batch_size=2, n_symb=14, n_sc=32)
        self.assertEqual(h_grid.shape, (2, 14, 32))
        nmse_1d, nmse_2d = compare_1d_vs_2d_lmmse(fd=0.03)
        self.assertLess(nmse_2d, nmse_1d)
        self.assertTrue(idd_pa_2d_cleanup())
        self.assertTrue(lmmse_2d_singular(noise_var=1e-12))
        self.assertAlmostEqual(idd_pa_2d_linear(), 0.0, places=5)
        self.assertTrue(idd_pa_2d_clipping(A=1.0))
        self.assertLess(check_edge_extrapolation(), 0.15)
        h_flat = lmmse_2d_flat()
        self.assertLess((h_flat[:, 1:] - h_flat[:, :-1]).abs().max().item(), 1e-4)

    def test_emfloatformer2d_strict_c_star_equivariance_and_loss(self):
        torch.manual_seed(42)
        model = EMFloatFormer2D(d_model=16, n_heads=2, depth=2)
        model.eval()
        sim = SlotSimulator2D(n_symb=14, n_sc=32, cp=16, fd=0.02)
        slot = sim.generate_slot(batch_size=2, snr_db=20.0, generator=torch.Generator().manual_seed(42))

        with torch.no_grad():
            base = model(slot.yp_sparse, slot.y_grid, slot.pilot_mask, slot.noise_var)
            # Check M2 saturation bound <= 20.0
            self.assertLessEqual(float(base["llrs"].abs().max().item()), 20.0001)

            max_llr_err = 0.0
            max_h_rel = 0.0
            total_flips = 0
            for p_exp, phi in [(-12, 0.37), (-5, 1.91), (0, math.pi / 7.0), (6, 2.718), (12, 5.1)]:
                r = 2.0 ** p_exp
                c = r * torch.exp(1j * torch.tensor(phi, dtype=torch.float64))
                yp_c = (slot.yp_sparse.to(torch.complex128) * c).to(torch.complex64)
                yg_c = (slot.y_grid.to(torch.complex128) * c).to(torch.complex64)
                nv_c = slot.noise_var * (r ** 2)
                out_c = model(yp_c, yg_c, slot.pilot_mask, nv_c)

                llr_err = float((out_c["llrs"] - base["llrs"]).abs().max().item())
                flips = int(((out_c["llrs"] > 0) != (base["llrs"] > 0)).sum().item())
                h_target = (base["h_hat"].to(torch.complex128) * c).to(torch.complex64)
                h_rel = float(
                    (torch.linalg.norm(out_c["h_hat"] - h_target)
                     / torch.linalg.norm(h_target).clamp(min=1e-30)).item()
                )
                max_llr_err = max(max_llr_err, llr_err)
                max_h_rel = max(max_h_rel, h_rel)
                total_flips += flips

            self.assertLess(max_llr_err, 1e-5)
            self.assertLess(max_h_rel, 1e-5)
            self.assertEqual(total_flips, 0)

        # Ultra-sparse single pilot RE check
        mask_1 = get_custom_dmrs_mask([(2, 0)], n_symb=14, n_sc=32)
        yp_1 = slot.y_grid[:, mask_1]
        with torch.no_grad():
            out_1 = model(yp_1, slot.y_grid, mask_1, slot.noise_var)
        self.assertTrue(torch.isfinite(out_1["llrs"]).all())

        # JointPhysicalLoss2D non-zero grads & C* gradient invariance
        model.train()
        loss_fn = JointPhysicalLoss2D()
        out1 = model(slot.yp_sparse, slot.y_grid, slot.pilot_mask, slot.noise_var)
        loss1 = loss_fn(out1, slot)
        loss1.backward()
        grads1 = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        self.assertEqual(len(grads1), sum(1 for _ in model.parameters()))
        self.assertEqual(sum(1 for g in grads1 if float(g.abs().sum()) == 0.0), 0)

        model.zero_grad()
        c = 3.7 * torch.exp(1j * torch.tensor(0.95))
        slot_s = SlotSimulator2D(n_symb=14, n_sc=32, cp=16, fd=0.02).generate_slot(
            batch_size=2, snr_db=20.0, generator=torch.Generator().manual_seed(42)
        )
        slot_s.yp_sparse = slot_s.yp_sparse * c
        slot_s.y_grid = slot_s.y_grid * c
        slot_s.h_true = slot_s.h_true * c
        slot_s.noise_var = slot_s.noise_var * (3.7 ** 2)
        out2 = model(slot_s.yp_sparse, slot_s.y_grid, slot_s.pilot_mask, slot_s.noise_var)
        loss2 = loss_fn(out2, slot_s)
        loss2.backward()
        grads2 = [p.grad.clone() for p in model.parameters() if p.grad is not None]
        max_grad_diff = max(float((g1 - g2).abs().max().item()) for g1, g2 in zip(grads1, grads2))
        self.assertLess(max_grad_diff, 1e-5)


class TestF17CodedBLERAndDeterminism(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)

    def test_sync_receiver_inputs_ignore_simulator_labels(self):
        sim = SlotSimulator2D(cfo=0.02, fd=0.03)
        slot = sim.generate_slot(batch_size=2, snr_db=16, coded=True,
                                 generator=torch.Generator().manual_seed(81))
        original = _synchronize_slot(slot)
        relabelled = _synchronize_slot(replace(
            slot, cfo=0.0, fd=0.0, h_true=torch.zeros_like(slot.h_true),
            h_time_evol=torch.zeros_like(slot.h_time_evol), data_tx=torch.zeros_like(slot.data_tx),
            bits=1-slot.bits, info_bits=1-slot.info_bits,
        ))
        self.assertTrue(torch.equal(original.y_grid, relabelled.y_grid))
        self.assertTrue(torch.equal(original.yp_sparse, relabelled.yp_sparse))
        self.assertFalse(torch.equal(original.y_grid, slot.y_grid))
        self.assertFalse(torch.equal(original.h_true, relabelled.h_true))

    def test_fixed_and_matched_priors_agree_for_same_configuration(self):
        for method in ("lmmse_2d", "idd_pa_2d"):
            settings = dict(snr_db=17, num_slots=2, profile="equal4", fd=0.01,
                            pa_condition="mild", seed=190)
            matched = _evaluate_slot_bler(method=method, **settings)
            fixed = _evaluate_slot_bler(method=method + "_fixed", **settings)
            self.assertEqual(matched, fixed)
            self.assertEqual(fixed["total_bits"], 26 * 64)
            self.assertEqual(fixed["total_blocks"], 26)
            self.assertEqual(fixed["total_uncoded_bits"], 26 * 128)

    def test_bler_accounting_and_ci(self):
        u = torch.tensor([0, 1, 1, 0])
        self.assertEqual(check_block_error(u, u), 0)
        self.assertEqual(check_block_error(u, torch.tensor([0, 1, 0, 0])), 1)

        errs, total = run_block_counter(num_blocks=10)
        self.assertIsInstance(errs, int)
        self.assertIsInstance(total, int)
        self.assertEqual(check_integer_accumulator(1_000_000), 1_000_000)

        blers = evaluate_bler_curve([5.0, 10.0, 15.0])
        for i in range(len(blers) - 1):
            self.assertGreaterEqual(blers[i], blers[i + 1])

        lo_fd, hi_fd = check_doppler_bler_sensitivity()
        self.assertLessEqual(lo_fd, hi_fd)
        self.assertEqual(evaluate_bler_point(40.0), 0.0)
        self.assertEqual(evaluate_bler_point(-15.0), 1.0)
        self.assertIn(simulate_single_block(), (0, 1))
        self.assertGreaterEqual(evaluate_ood_profile_bler("exp16"), 0.0)

        ci_lo, ci_hi = compute_ci(errors=5, total=100)
        self.assertAlmostEqual(ci_lo, 0.0164319, places=4)
        self.assertAlmostEqual(ci_hi, 0.1128349, places=4)

    def test_benchmark_rng_determinism(self):
        res1 = run_p4b_coded_bler_benchmarks(train_steps=8, eval_slots=2, output_dir=None)
        _ = torch.randn(31)
        res2 = run_p4b_coded_bler_benchmarks(train_steps=8, eval_slots=2, output_dir=None)
        self.assertEqual(res1["rows"], res2["rows"])


if __name__ == "__main__":
    unittest.main()
