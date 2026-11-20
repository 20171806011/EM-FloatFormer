"""Strict mathematical & architectural invariant unit tests for Phase 8 (p8)."""
from dataclasses import replace
import math
import unittest

import torch

from p4.simulator import q16
from p5.wavefield_sim import CP, N, propagate
from p8_manifold.manifold_charts import (
    B_ICI,
    grid_coordinates,
    init_blind_pilot_weights,
    init_offgrid_charts,
    offgrid_2d_lmmse,
    offgrid_pilot_gauge,
)
from p8_manifold.model import EM8ManifoldReceiver
from p8_manifold.offgrid_sim import (
    N_PILOTS,
    OFFGRID_CONDITIONS,
    T_SYM,
    extract_payload_llrs,
    simulate_offgrid,
    true_offgrid_operator,
)
from p8_manifold.transport_attention import CoherentTransportLayer, CrossHypothesisJeffreysLayer, jeffreys_divergence


REAL = torch.float64
COMPLEX = torch.complex128


def _scale_obs(obs, alpha):
    a = complex(alpha)
    a2 = abs(a) ** 2
    return replace(obs, y=obs.y * a, noise_var=obs.noise_var * a2)


def _perturb_model(model, seed=12345, eps=0.08):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if not p.requires_grad:
                continue
            if p.is_complex():
                dr = torch.randn(p.shape, dtype=REAL, generator=gen) * eps
                di = torch.randn(p.shape, dtype=REAL, generator=gen) * eps
                p.add_(torch.complex(dr, di))
            else:
                p.add_(torch.randn(p.shape, dtype=REAL, generator=gen) * eps)


class TestP8Invariants(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_01_offgrid_true_operator_exact(self):
        """Exact noiseless Off-Grid time-domain propagation matches frequency operator G_true < 1e-12."""
        frame = simulate_offgrid(4, snr_db=14.0, seed=8001, condition="O0_random_sparse")
        tx = frame.obs.pilots.clone()
        q_syms = q16(frame.bits).to(COMPLEX)
        for b in range(4):
            tx[b, ~frame.obs.pilot_mask[b]] = q_syms[b]
            self.assertEqual(int(frame.obs.pilot_mask[b].sum()), N_PILOTS)
            self.assertIn(int(frame.n_rays[b]), (2, 3, 4, 5))

        sig, h_true, _ = propagate(
            tx, frame.truth.delays, frame.truth.gains, frame.truth.angles,
            frame.truth.nu_start, frame.truth.nu_end, n_rx=1, cp=CP
        )
        y_noiseless = torch.fft.fft(sig, norm="ortho")[:, 0]
        G_true = true_offgrid_operator(frame)
        y_from_G = torch.einsum("btfk,btk->btf", G_true, tx)
        err = float((y_noiseless - y_from_G).abs().max())
        diag_err = float((G_true.diagonal(dim1=-2, dim2=-1) - h_true[:, 0]).abs().max())
        self.assertLess(err, 1e-12)
        self.assertLess(diag_err, 1e-12)

    def test_02_parameter_and_init_parity(self):
        """EM8-Transport, EM8-AbsNoTransport, and EM8-Local share exact real param count and exact step-0 output."""
        torch.manual_seed(8002)
        m_tr = EM8ManifoldReceiver(variant="transport", d_model=16, n_layers=2, n_heads=4, steps=2)
        torch.manual_seed(8002)
        m_ab = EM8ManifoldReceiver(variant="abs_no_transport", d_model=16, n_layers=2, n_heads=4, steps=2)
        torch.manual_seed(8002)
        m_lo = EM8ManifoldReceiver(variant="local", d_model=16, n_layers=2, n_heads=4, steps=2)
        m_fx = EM8ManifoldReceiver(variant="fixed", d_model=16, n_layers=2, n_heads=4, steps=2)

        self.assertEqual(m_tr.real_parameters(), m_ab.real_parameters())
        self.assertEqual(m_tr.real_parameters(), m_lo.real_parameters())
        self.assertEqual(m_fx.real_parameters(), 0)
        self.assertGreater(m_tr.real_parameters(), 10000)

        frame = simulate_offgrid(2, snr_db=14.0, seed=8002, condition="O1_roving_island")
        with torch.no_grad():
            o_fx = m_fx(frame.obs)
            o_tr = m_tr(frame.obs)
            o_ab = m_ab(frame.obs)
            o_lo = m_lo(frame.obs)
        for out in (o_tr, o_ab, o_lo):
            self.assertLess(float((out.bit_llr - o_fx.bit_llr).abs().max()), 1e-14)
            self.assertLess(float((out.h_hat - o_fx.h_hat).abs().max()), 1e-14)
            self.assertLess(float((out.pi - o_fx.pi).abs().max()), 1e-14)

    def test_03_signed_interference_and_parallel_transport_coherence(self):
        """Directly prove Construction A:
        1. Signed Re(q'^H k') separates +1 constructive vs -1 destructive phase (abs_no_transport cannot).
        2. Unitary parallel transport aggregates phase-rotated remote pilots coherently (abs_no_transport cancels).
        3. Local 3x3 layer has exact zero sensitivity to remote perturbations beyond 1-hop 3x3 window.
        """
        torch.manual_seed(8003)
        d_model, n_heads = 8, 2
        layer_tr = CoherentTransportLayer("transport", d_model=d_model, n_heads=n_heads)
        layer_ab = CoherentTransportLayer("abs_no_transport", d_model=d_model, n_heads=n_heads)
        layer_lo = CoherentTransportLayer("local", d_model=d_model, n_heads=n_heads)

        # Align weights identically across all 3 layers; set w_q = w_k = w_v = w_pk = w_pv = I and omega = (omega_t, 0)
        omega_val = math.pi  # exp(i * pi * t_norm): over t_norm in [-1, 1], phase flips by pi between t=0 and t=19!
        with torch.no_grad():
            eye = torch.eye(d_model, dtype=COMPLEX)
            for lyr in (layer_tr, layer_ab, layer_lo):
                lyr.w_q.copy_(eye)
                lyr.w_k.copy_(eye)
                lyr.w_v.copy_(eye)
                lyr.w_pk.copy_(eye)
                lyr.w_pv.copy_(eye)
                lyr.w_o.copy_(eye)
                lyr.omega.zero_()
                lyr.omega[:, 0] = omega_val
                lyr.raw_beta.fill_(-25.0)  # disable geodesic decay during pure algebraic phase-transport check
                lyr.gamma_ev.zero_()

        coords = grid_coordinates(T_SYM, N, torch.device("cpu"))  # 20, 32, 2
        # (a) Constructive vs destructive sign discrimination on 1D _attend_1d:
        q = torch.ones((1, 1, n_heads, d_model // n_heads), dtype=COMPLEX)
        k_pos = torch.ones((1, 2, n_heads, d_model // n_heads), dtype=COMPLEX)
        k_pos[:, 1] = -1.0  # Key 0 is +1 (constructive), Key 1 is -1 (destructive!)
        v_two = torch.zeros((1, 2, n_heads, d_model // n_heads), dtype=COMPLEX)
        v_two[:, 0] = 1.0 + 0.0j
        v_two[:, 1] = -1.0 + 0.0j
        z_zero = torch.zeros((1, 1, 2), dtype=REAL)
        z_two = torch.zeros((1, 2, 2), dtype=REAL)
        with torch.no_grad():
            out_tr = layer_tr._attend_1d(q, k_pos, v_two, z_zero, z_two)
            out_ab = layer_ab._attend_1d(q, k_pos, v_two, z_zero, z_two)
        # Signed transport selects Key 0 (+1) over Key 1 (-1) -> large positive real output
        self.assertGreater(float(out_tr.real.mean()), 0.85)
        # Unsigned abs_no_transport gives identical score |+1| == |-1| -> exact cancellation (~0.0)
        self.assertLess(float(out_ab.abs().max()), 1e-12)

        # (b) Coherent parallel-transport aggregation across opposite-phase remote pilots (t_norm=-1 vs t_norm=0):
        # Signal wave follows v(z) = exp(i * omega_val * t_norm) * e_0 -> at t_norm=-1, v=-1; at t_norm=0, v=+1.
        # Target at t_norm=+1 expects exp(i * pi * 1) = -1!
        z_q = torch.tensor([[[1.0, 0.0]]], dtype=REAL)
        z_k = torch.tensor([[[-1.0, 0.0], [0.0, 0.0]]], dtype=REAL)
        q_w = torch.full((1, 1, n_heads, d_model // n_heads), -1.0 + 0.0j, dtype=COMPLEX)
        k_w = torch.ones((1, 2, n_heads, d_model // n_heads), dtype=COMPLEX)
        k_w[:, 0] = -1.0 + 0.0j  # exp(i * pi * -1) = -1
        k_w[:, 1] = +1.0 + 0.0j  # exp(i * pi *  0) = +1
        v_w = k_w.clone()
        with torch.no_grad():
            agg_tr = layer_tr._attend_1d(q_w, k_w, v_w, z_q, z_k)
            agg_ab = layer_ab._attend_1d(q_w, k_w, v_w, z_q, z_k)
        # Parallel transport unwinds both pilots to +1, averages +1, transports to t_norm=+1 -> EXACT -1.0!
        self.assertLess(float((agg_tr - (-1.0 + 0.0j)).abs().max()), 1e-12)
        # Untransported abs_no_transport averages (-1 + 1)/2 -> phase cancellation (< 0.25 residual from pos_bias)
        self.assertGreater(float((agg_ab - (-1.0 + 0.0j)).abs().mean()), 0.5)

        # (c) Strict local 3x3 reach vs global 1-hop reach:
        state_a = torch.ones((1, T_SYM, N, d_model), dtype=COMPLEX)
        state_b = state_a.clone()
        # Perturb far-away corner (t=18, f=20)
        state_b[0, 18, 20] = 5.0 - 3.0j
        p_coords = torch.tensor([[[coords[18, 20, 0], coords[18, 20, 1]]]], dtype=REAL)
        p_w = torch.ones((1, 1), dtype=REAL)
        p_dj = torch.ones((1, 1), dtype=REAL)
        with torch.no_grad():
            y_lo_a = layer_lo(state_a, coords, state_a[:, 18, 20:21], p_coords, p_w, p_dj)
            y_lo_b = layer_lo(state_b, coords, state_b[:, 18, 20:21], p_coords, p_w, p_dj)
            y_tr_a = layer_tr(state_a, coords, state_a[:, 18, 20:21], p_coords, p_w, p_dj)
            y_tr_b = layer_tr(state_b, coords, state_b[:, 18, 20:21], p_coords, p_w, p_dj)
        # At distant origin (t=0, f=0): local 3x3 has EXACT ZERO response; transport responds in 1 hop!
        self.assertEqual(float((y_lo_a[0, 0, 0] - y_lo_b[0, 0, 0]).abs().max()), 0.0)
        self.assertGreater(float((y_tr_a[0, 0, 0] - y_tr_b[0, 0, 0]).abs().max()), 1e-3)

    def test_04_jeffreys_divergence_invariance_and_positivity(self):
        """Jeffreys divergence is symmetric, nonnegative, zero iff identical, and strictly C*-invariant."""
        gen = torch.Generator().manual_seed(8004)
        mu1 = torch.complex(torch.randn((2, 4, 8), dtype=REAL, generator=gen), torch.randn((2, 4, 8), dtype=REAL, generator=gen))
        mu2 = torch.complex(torch.randn((2, 4, 8), dtype=REAL, generator=gen), torch.randn((2, 4, 8), dtype=REAL, generator=gen))
        v1 = torch.rand((2, 4, 8), dtype=REAL, generator=gen) + 0.1
        v2 = torch.rand((2, 4, 8), dtype=REAL, generator=gen) + 0.1

        d12 = jeffreys_divergence(mu1, v1, mu2, v2)
        d21 = jeffreys_divergence(mu2, v2, mu1, v1)
        d11 = jeffreys_divergence(mu1, v1, mu1, v1)
        self.assertLess(float((d12 - d21).abs().max()), 1e-14)
        self.assertEqual(float(d11.abs().max()), 0.0)
        self.assertTrue(bool((d12 > 0).all()))

        for alpha in (1e-20 * complex(math.cos(0.7), math.sin(0.7)), 1e15 * complex(math.cos(-2.1), math.sin(-2.1))):
            d_scaled = jeffreys_divergence(mu1 * alpha, v1 * (abs(alpha) ** 2), mu2 * alpha, v2 * (abs(alpha) ** 2))
            rel = float(((d_scaled - d12).abs() / d12.clamp_min(1e-12)).max())
            self.assertLess(rel, 1e-12)

    def test_05_cstar_gauge_equivariance_strict(self):
        """Strict C* = R+ x U(1) invariance/equivariance < 1e-12 across [1e-20, 1e15] for all EM8 variants & conditions."""
        alphas = (
            1e-20 * complex(math.cos(0.9), math.sin(0.9)),
            1e-8 * complex(math.cos(-1.7), math.sin(-1.7)),
            1e8 * complex(math.cos(2.3), math.sin(2.3)),
            1e15 * complex(math.cos(-0.4), math.sin(-0.4)),
        )
        variants = ("fixed", "transport", "abs_no_transport", "local", "single_transport")
        for v_idx, variant in enumerate(variants):
            torch.manual_seed(8010 + v_idx)
            model = EM8ManifoldReceiver(variant=variant, d_model=8, n_layers=1, n_heads=2, steps=2)
            if model.learned:
                _perturb_model(model, seed=8020 + v_idx, eps=0.12)
            cond = OFFGRID_CONDITIONS[v_idx % len(OFFGRID_CONDITIONS)]
            frame = simulate_offgrid(2, snr_db=14.0, seed=8030 + v_idx, condition=cond)
            with torch.no_grad():
                ref = model(frame.obs)
                for alpha in alphas:
                    out = model(_scale_obs(frame.obs, alpha))
                    llr_err = float((out.bit_llr - ref.bit_llr).abs().max())
                    llr_rel = float(llr_err / max(float(ref.bit_llr.abs().max()), 1.0))
                    pi_err = float((out.pi - ref.pi).abs().max())
                    h_ref_sc = ref.h_hat * alpha
                    py_ref_sc = ref.pred_y * alpha
                    pv_ref_sc = ref.pred_var * (abs(alpha) ** 2)
                    h_rel = float((out.h_hat - h_ref_sc).abs().max() / h_ref_sc.abs().max().clamp_min(1e-300))
                    py_rel = float((out.pred_y - py_ref_sc).abs().max() / py_ref_sc.abs().max().clamp_min(1e-300))
                    pv_rel = float((out.pred_var - pv_ref_sc).abs().max() / pv_ref_sc.abs().max().clamp_min(1e-300))
                    w_err = float((out.pilot_weights - ref.pilot_weights).abs().max())
                    flips = int(((out.bit_llr > 0) != (ref.bit_llr > 0)).sum())
                    self.assertLess(llr_rel, 1e-12, f"{variant} {cond} alpha={alpha} llr_rel={llr_rel}")
                    self.assertLess(llr_err, 5e-12, f"{variant} {cond} alpha={alpha} llr_err={llr_err}")
                    self.assertLess(pi_err, 1e-12, f"{variant} {cond} alpha={alpha} pi_err={pi_err}")
                    self.assertLess(h_rel, 1e-12, f"{variant} {cond} alpha={alpha} h_rel={h_rel}")
                    self.assertLess(py_rel, 1e-12, f"{variant} {cond} alpha={alpha} py_rel={py_rel}")
                    self.assertLess(pv_rel, 1e-12, f"{variant} {cond} alpha={alpha} pv_rel={pv_rel}")
                    self.assertLess(w_err, 1e-12, f"{variant} {cond} alpha={alpha} w_err={w_err}")
                    self.assertEqual(flips, 0, f"{variant} {cond} alpha={alpha} flips={flips}")

    def test_06_holdout_zero_leakage_firewall(self):
        """Poisoning held-out probe locations in obs.y with huge garbage causes EXACT ZERO change in Pass 1."""
        frame = simulate_offgrid(2, snr_db=14.0, seed=8006, condition="O3_holdout_sequential")
        self.assertGreater(int(frame.probe_mask.sum()), 0)
        y_poison = frame.obs.y.clone()
        y_poison[:, 0] = torch.where(frame.probe_mask, torch.full_like(y_poison[:, 0], 1e9 - 7e8j), y_poison[:, 0])
        obs_poison = replace(frame.obs, y=y_poison)

        for variant in ("transport", "abs_no_transport", "local", "fixed"):
            torch.manual_seed(8006)
            model = EM8ManifoldReceiver(variant=variant, d_model=8, n_layers=1, n_heads=2, steps=2)
            if model.learned:
                _perturb_model(model, seed=8006, eps=0.1)
            with torch.no_grad():
                o1 = model(frame.obs)
                o2 = model(obs_poison)
            self.assertTrue(torch.equal(o1.bit_llr, o2.bit_llr), f"Leakage in bit_llr for {variant}")
            self.assertTrue(torch.equal(o1.h_hat, o2.h_hat), f"Leakage in h_hat for {variant}")
            self.assertTrue(torch.equal(o1.pred_y, o2.pred_y), f"Leakage in pred_y for {variant}")
            self.assertTrue(torch.equal(o1.pred_var, o2.pred_var), f"Leakage in pred_var for {variant}")

    def test_07_blind_student_t_outlier_and_erasure_rejection(self):
        """In O2_blind_burst_erasure, blind Student-t assigns near-zero weight to unknown erasures/impulses and high weight to clean pilots."""
        frame = simulate_offgrid(4, snr_db=14.0, seed=8007, condition="O2_blind_burst_erasure")
        self.assertTrue(frame.obs.blind)
        self.assertTrue(bool(frame.obs.valid_mask.all()))  # Receiver is NOT given corrupt_mask!

        model = EM8ManifoldReceiver(variant="fixed", steps=2)
        with torch.no_grad():
            out = model(frame.obs)
        clean_p = frame.obs.pilot_mask & (~frame.corrupt_mask)
        corrupt_p = frame.corrupt_mask
        w_clean = float(out.pilot_weights[clean_p].mean())
        w_corrupt = float(out.pilot_weights[corrupt_p].mean())
        self.assertGreater(w_clean, 0.70, f"Expected clean pilot weight > 0.70, got {w_clean}")
        self.assertLess(w_corrupt, 0.08, f"Expected corrupt pilot weight < 0.08, got {w_corrupt}")

    def test_08_sequential_assimilation_preserves_state_and_no_leakage(self):
        """assimilate() has exact identity on empty reveal, zero leakage of unrevealed probes, and strict C* equivariance."""
        frame = simulate_offgrid(2, snr_db=14.0, seed=8008, condition="O3_holdout_sequential")
        torch.manual_seed(8008)
        m_fx = EM8ManifoldReceiver(variant="fixed", steps=2)
        m_tr = EM8ManifoldReceiver(variant="transport", d_model=8, n_layers=1, n_heads=2, steps=2)
        _perturb_model(m_tr, seed=8008, eps=0.08)

        empty_mask = torch.zeros_like(frame.probe_mask)
        with torch.no_grad():
            p1_fx = m_fx(frame.obs)
            p2_empty = m_fx.assimilate(p1_fx.state, frame.y_clean_full, empty_mask)
            self.assertLess(float((p2_empty.bit_llr - p1_fx.bit_llr).abs().max()), 1e-13)
            self.assertLess(float((p2_empty.h_hat - p1_fx.h_hat).abs().max()), 1e-13)

            p1_tr = m_tr(frame.obs)
            p2_tr_empty = m_tr.assimilate(p1_tr.state, frame.y_clean_full, empty_mask)
            self.assertLess(float((p2_tr_empty.bit_llr - p1_tr.bit_llr).abs().max()), 1e-13)
            self.assertLess(float((p2_tr_empty.h_hat - p1_tr.h_hat).abs().max()), 1e-13)

            # Reveal ONLY half the probes; poison the OTHER unrevealed half -> must be bit-identical!
            half_mask = torch.zeros_like(frame.probe_mask)
            for b in range(2):
                coords = frame.probe_mask[b].nonzero(as_tuple=False)[:4]
                half_mask[b, coords[:, 0], coords[:, 1]] = True
            unrev_mask = frame.probe_mask & (~half_mask)
            y_poison = frame.y_clean_full.clone()
            y_poison[:, 0] = torch.where(unrev_mask, torch.full_like(y_poison[:, 0], -9e8 + 4e8j), y_poison[:, 0])

            a_clean = m_tr.assimilate(p1_tr.state, frame.y_clean_full, half_mask)
            a_poison = m_tr.assimilate(p1_tr.state, y_poison, half_mask)
            self.assertTrue(torch.equal(a_clean.bit_llr, a_poison.bit_llr))
            self.assertTrue(torch.equal(a_clean.h_hat, a_poison.h_hat))

            # Full C* equivariance of Pass 1 -> assimilate() pipeline!
            alpha = 1e-15 * complex(math.cos(1.2), math.sin(1.2))
            p1_sc = m_tr(_scale_obs(frame.obs, alpha))
            a_sc = m_tr.assimilate(p1_sc.state, frame.y_clean_full * alpha, frame.probe_mask)
            a_ref = m_tr.assimilate(p1_tr.state, frame.y_clean_full, frame.probe_mask)
            self.assertLess(float((a_sc.bit_llr - a_ref.bit_llr).abs().max()), 1e-12)
            a_ref_sc = a_ref.h_hat * alpha
            self.assertLess(
                float((a_sc.h_hat - a_ref_sc).abs().max() / a_ref_sc.abs().max().clamp_min(1e-300)),
                1e-12,
            )

    def test_09_three_term_variance_decomposition_exact(self):
        """Exact law of total variance: pred_var == aleatoric + epistemic + between to 1e-14 and all terms >= 0."""
        frame = simulate_offgrid(2, snr_db=14.0, seed=8009, condition="O3_holdout_sequential")
        torch.manual_seed(8009)
        model = EM8ManifoldReceiver(variant="transport", d_model=8, n_layers=1, n_heads=2, steps=2)
        _perturb_model(model, seed=8009, eps=0.08)
        with torch.no_grad():
            out = model(frame.obs)
        recomp = out.pred_var_aleatoric + out.pred_var_epistemic + out.pred_var_between
        self.assertLess(float((out.pred_var - recomp).abs().max()), 1e-14)
        self.assertTrue(bool((out.pred_var_aleatoric > 0).all()))
        self.assertTrue(bool((out.pred_var_epistemic > 0).all()))
        self.assertTrue(bool((out.pred_var_between >= 0).all()))
        # Also verify symbol_var_total == within + between
        sym_err = float((out.symbol_var_total - (out.symbol_var_within + out.symbol_var_between)).abs().max())
        self.assertLess(sym_err, 1e-14)

    def test_10_full_active_finite_gradients_extreme_scales(self):
        """Every trainable parameter receives finite nonzero gradients at normal, 1e-20, and 1e15 input scales."""
        frame = simulate_offgrid(2, snr_db=14.0, seed=8010, condition="O3_holdout_sequential")
        for v_idx, variant in enumerate(("transport", "abs_no_transport", "local", "single_transport")):
            for alpha in (1.0 + 0.0j, 1e-20 * (0.6 - 0.8j), 1e15 * (-0.8 + 0.6j)):
                torch.manual_seed(8050 + v_idx)
                model = EM8ManifoldReceiver(variant=variant, d_model=8, n_layers=1, n_heads=2, steps=2)
                # Slightly perturb away from step-0 zero-crossing so fusion_scale[2] (gamma_route * logits_route) is live
                _perturb_model(model, seed=8060 + v_idx, eps=0.05)
                model.zero_grad()
                obs_sc = _scale_obs(frame.obs, alpha)
                out = model(obs_sc)
                data_llr = extract_payload_llrs(frame.obs.pilot_mask, out.bit_llr)
                signed = 2.0 * frame.bits.to(REAL) - 1.0
                bce = torch.nn.functional.softplus(-signed * data_llr).mean()
                # Include probe NLL in gauge-invariant form so epi_delta & pred_var receive direct gradient
                y_true_sc = frame.y_clean_full[:, 0] * alpha
                pm = frame.probe_mask
                nll = (((y_true_sc[pm] - out.pred_y[pm]).abs().square() / out.pred_var[pm]) + out.pred_var[pm].log()).mean()
                loss = bce + 0.05 * nll
                loss.backward()

                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    if variant == "single_transport" and (
                        name == "fusion_scale"
                        or name.startswith("backbone.cross_hyp.w_q")
                        or name.startswith("backbone.cross_hyp.w_k")
                        or name.startswith("backbone.cross_hyp.dj_scale")
                        or name.startswith("backbone.cross_hyp.route_proj")
                    ):
                        # M=1 has trivial softmax pi=[1], so branch-routing params are structurally unused
                        continue
                    self.assertIsNotNone(p.grad, f"{variant} alpha={alpha} missing grad for {name}")
                    self.assertTrue(torch.isfinite(p.grad).all(), f"{variant} alpha={alpha} nonfinite grad for {name}")
                    gnorm = float(p.grad.abs().max())
                    self.assertGreater(gnorm, 0.0, f"{variant} alpha={alpha} ZERO grad for {name}")

    def test_11_periodic_gp_kernel_strictly_psd(self):
        """Exact chordal periodic S^1 2D GP kernel is strictly Hermitian PSD across all charts."""
        from p8_manifold.manifold_charts import CHART_SPECS_M1, CHART_SPECS_M4, _gp_kernel_2d

        gt = torch.arange(T_SYM, dtype=REAL)[:, None].expand(T_SYM, N).reshape(-1)
        gf = torch.arange(N, dtype=REAL)[None, :].expand(T_SYM, N).reshape(-1)
        for spec in (*CHART_SPECS_M4, *CHART_SPECS_M1):
            K = _gp_kernel_2d(gt, gf, gt, gf, N, *spec)
            herm_err = float((K - K.mH).abs().max())
            min_eig = float(torch.linalg.eigvalsh(0.5 * (K + K.mH)).min())
            self.assertLess(herm_err, 1e-14)
            self.assertGreaterEqual(min_eig, -1e-12, f"Non-PSD kernel for {spec}: min_eig={min_eig}")


if __name__ == "__main__":
    unittest.main()
