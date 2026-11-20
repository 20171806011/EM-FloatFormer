"""Strict mathematical, physical, and architectural invariant unit tests for P9-C."""
from __future__ import annotations

from dataclasses import replace
import math
import os
import unittest

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch.nn import functional as F

from p9.contract import (
    N,
    PROBE_ROW,
    REGIMES_ID,
    SEED_P9_DEV,
    T,
    CandidateSummary,
    PublicObservation,
)
from p9.neural_gate import (
    COMPLEX,
    REAL,
    VARIANTS,
    TemporalTransportLayer,
    TemporalTrustGate,
    classical_pre_update_gate,
    extract_gate_features,
    mix_bit_llrs_torch_from_logit,
)
from p9.p9c_dev import (
    _joint_loss_batch,
    _stack_training_tensors,
    build_candidates,
    build_dataset,
    simulate_p9_link,
)


def _scale_obs_and_cands(
    obs: PublicObservation,
    cand_inh: CandidateSummary,
    cand_clr: CandidateSummary,
    alpha: complex,
):
    """Apply exact physical C* scaling (y -> alpha*y, n0 -> |alpha|^2*n0) to obs & CandidateSummary."""
    a = complex(alpha)
    a2 = abs(a) ** 2
    m_train = len(obs.training_rows) * N
    ll_shift = -m_train * math.log(a2)

    obs_s = replace(
        obs,
        y_a=obs.y_a * a,
        y_b=obs.y_b * a,
        n0=obs.n0 * a2,
    )

    def _sc(c: CandidateSummary) -> CandidateSummary:
        return replace(
            c,
            pre_train_loglik=c.pre_train_loglik + ll_shift,
            pre_train_pred=c.pre_train_pred * a,
            pre_train_var=c.pre_train_var * a2,
            mean_ray=c.mean_ray * a,
            var_ray=c.var_ray * a2,
            probe_pred=c.probe_pred * a,
            probe_var=c.probe_var * a2,
        )

    return obs_s, _sc(cand_inh), _sc(cand_clr)


def _perturb_model(model: TemporalTrustGate, seed: int = 2026, eps: float = 0.15) -> None:
    """Move trainable parameters away from frozen initial_map so delta_logit & log_tau are non-trivial."""
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


class TestP9CInvariants(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_01_exact_integer_parameter_parity(self):
        """transport_attn == abs_no_transport == local == no_history in exact real parameter count."""
        counts = {}
        for v in VARIANTS:
            torch.manual_seed(9101)
            m = TemporalTrustGate(variant=v, d_model=12, n_heads=4, n_layers=1)
            counts[v] = m.real_parameters()

        self.assertEqual(counts["transport_attn"], counts["abs_no_transport"])
        self.assertEqual(counts["transport_attn"], counts["local"])
        self.assertEqual(counts["transport_attn"], counts["no_history"])
        self.assertGreaterEqual(counts["transport_attn"], 1200)
        self.assertLessEqual(counts["transport_attn"], 2500)

    def test_02_step0_classical_identity(self):
        """Untrained TemporalTrustGate matches classical pre-update Bayes gate to < 1e-14."""
        for r_idx, regime in enumerate(REGIMES_ID):
            obs, _ = simulate_p9_link(SEED_P9_DEV + 90100 + r_idx, regime, snr_db=9.0, budget="sparse")
            cand_inh, cand_clr = build_candidates(obs)
            ref = classical_pre_update_gate(obs, cand_inh, cand_clr)

            for v in ("transport_attn", "abs_no_transport", "local"):
                torch.manual_seed(9102)
                model = TemporalTrustGate(variant=v, d_model=12, n_heads=4, n_layers=1)
                out = model.gate(obs, cand_inh, cand_clr)

                self.assertLess(float(np.max(np.abs(out.history_weight - ref.history_weight))), 1e-14)
                self.assertLess(abs(out.probe_weight - ref.probe_weight), 1e-14)
                self.assertLess(float(np.max(np.abs(out.mix_llrs - ref.mix_llrs))), 1e-14)
                self.assertLess(float(np.max(np.abs(out.mix_probe_pred - ref.mix_probe_pred))), 1e-14)
                self.assertLess(float(np.max(np.abs(out.mix_probe_var - ref.mix_probe_var))), 1e-14)
                self.assertLess(float(np.max(np.abs(out.mix_mean_ray - ref.mix_mean_ray))), 1e-14)
                self.assertLess(float(np.max(np.abs(out.mix_var_ray - ref.mix_var_ray))), 1e-14)

    def test_03_strict_cstar_gauge_invariance_and_equivariance(self):
        """Strict C* invariance (< 1e-12) on w[T], mix_llrs & equivariance on probe/ray moments."""
        obs, _ = simulate_p9_link(SEED_P9_DEV + 90201, "partial_shift", snr_db=9.0, budget="sparse")
        cand_inh, cand_clr = build_candidates(obs)
        alphas = (
            1e-12 * np.exp(1j * 0.7),
            1e10 * np.exp(-1j * 2.1),
        )

        for v_idx, v in enumerate(VARIANTS):
            torch.manual_seed(9110 + v_idx)
            model = TemporalTrustGate(variant=v, d_model=12, n_heads=4, n_layers=1)
            _perturb_model(model, seed=9210 + v_idx, eps=0.2)
            base_out = model.gate(obs, cand_inh, cand_clr)

            # Confirm model actually moved away from step-0 identity so test is non-trivial.
            if v != "no_history":
                cls_out = classical_pre_update_gate(obs, cand_inh, cand_clr)
                self.assertGreater(float(np.max(np.abs(base_out.history_weight - cls_out.history_weight))), 1e-4)

            for alpha in alphas:
                a2 = abs(alpha) ** 2
                obs_s, ci_s, cc_s = _scale_obs_and_cands(obs, cand_inh, cand_clr, alpha)
                sc_out = model.gate(obs_s, ci_s, cc_s)

                w_err = float(np.max(np.abs(sc_out.history_weight - base_out.history_weight)))
                wp_err = abs(sc_out.probe_weight - base_out.probe_weight)
                llr_diff = np.abs(sc_out.mix_llrs - base_out.mix_llrs)
                llr_abs = float(np.max(llr_diff))
                llr_rel = llr_abs / max(float(np.max(np.abs(base_out.mix_llrs))), 1.0)
                # Non-saturated bits (|L| <= 8, where 1-pm >= 3.3e-4) avoid p9/contract.py's (1 - pm) ULP loss.
                nonsat = np.abs(base_out.mix_llrs) <= 8.0
                llr_nonsat_abs = float(np.max(llr_diff[nonsat])) if np.any(nonsat) else 0.0
                bit_flips = int(np.count_nonzero((sc_out.mix_llrs > 0.0) != (base_out.mix_llrs > 0.0)))

                pp_ref = alpha * base_out.mix_probe_pred
                pv_ref = a2 * base_out.mix_probe_var
                mr_ref = alpha * base_out.mix_mean_ray
                vr_ref = a2 * base_out.mix_var_ray

                pp_rel = float(np.max(np.abs(sc_out.mix_probe_pred - pp_ref)) / max(float(np.max(np.abs(pp_ref))), 1e-300))
                pv_rel = float(np.max(np.abs(sc_out.mix_probe_var - pv_ref)) / max(float(np.max(np.abs(pv_ref))), 1e-300))
                mr_rel = float(np.max(np.abs(sc_out.mix_mean_ray - mr_ref)) / max(float(np.max(np.abs(mr_ref))), 1e-300))
                vr_rel = float(np.max(np.abs(sc_out.mix_var_ray - vr_ref)) / max(float(np.max(np.abs(vr_ref))), 1e-300))

                self.assertLess(w_err, 1e-12, f"{v} alpha={alpha}: w_err={w_err}")
                self.assertLess(wp_err, 1e-12, f"{v} alpha={alpha}: wp_err={wp_err}")
                self.assertLess(llr_nonsat_abs, 1e-12, f"{v} alpha={alpha}: llr_nonsat_abs={llr_nonsat_abs}")
                self.assertLess(llr_rel, 1e-11, f"{v} alpha={alpha}: llr_rel={llr_rel} (abs={llr_abs})")
                self.assertEqual(bit_flips, 0, f"{v} alpha={alpha}: bit_flips={bit_flips}")
                self.assertLess(pp_rel, 1e-12, f"{v} alpha={alpha}: pp_rel={pp_rel}")
                self.assertLess(pv_rel, 1e-12, f"{v} alpha={alpha}: pv_rel={pv_rel}")
                self.assertLess(mr_rel, 1e-12, f"{v} alpha={alpha}: mr_rel={mr_rel}")
                self.assertLess(vr_rel, 1e-12, f"{v} alpha={alpha}: vr_rel={vr_rel}")

                # Also verify strict absolute < 1e-12 in log-space LLR mixture (avoids 1-w ULP amplification).
                f0 = model.extract_features_for_variant(obs, cand_inh, cand_clr)
                f1 = model.extract_features_for_variant(obs_s, ci_s, cc_s)
                with torch.no_grad():
                    _, l0_t = model.forward_tensors(
                        torch.from_numpy(f0.scalar_feats).to(REAL).unsqueeze(0),
                        torch.from_numpy(f0.complex_feats).to(COMPLEX).unsqueeze(0),
                        torch.from_numpy(f0.t_norm).to(REAL).unsqueeze(0),
                        torch.from_numpy(f0.ev_base).to(REAL).unsqueeze(0),
                        torch.from_numpy(f0.base_logit).to(REAL).unsqueeze(0),
                    )
                    _, l1_t = model.forward_tensors(
                        torch.from_numpy(f1.scalar_feats).to(REAL).unsqueeze(0),
                        torch.from_numpy(f1.complex_feats).to(COMPLEX).unsqueeze(0),
                        torch.from_numpy(f1.t_norm).to(REAL).unsqueeze(0),
                        torch.from_numpy(f1.ev_base).to(REAL).unsqueeze(0),
                        torch.from_numpy(f1.base_logit).to(REAL).unsqueeze(0),
                    )
                    li_t = torch.from_numpy(cand_inh.llrs).to(REAL).unsqueeze(0)
                    lc_t = torch.from_numpy(cand_clr.llrs).to(REAL).unsqueeze(0)
                    m0 = mix_bit_llrs_torch_from_logit(l0_t[..., None, None], li_t, lc_t)
                    m1 = mix_bit_llrs_torch_from_logit(l1_t[..., None, None], li_t, lc_t)
                    self.assertLess(float((m1 - m0).abs().max()), 1e-12)

    def test_04_zero_probe_leakage(self):
        """Poisoning obs.y_b[PROBE_ROW] causes EXACT zero change in features, w[T], and mix_llrs."""
        obs, _ = simulate_p9_link(SEED_P9_DEV + 90301, "continuous", snr_db=9.0, budget="sparse")
        cand_inh, cand_clr = build_candidates(obs)

        y_poison = obs.y_b.copy()
        y_poison[PROBE_ROW] = 1e9 - 7e8j
        obs_poison = replace(obs, y_b=y_poison)

        y_inf = obs.y_b.copy()
        y_inf[PROBE_ROW] = np.inf - 1j * np.inf
        obs_inf = replace(obs, y_b=y_inf)

        f_clean = extract_gate_features(obs, cand_inh, cand_clr)
        f_poison = extract_gate_features(obs_poison, cand_inh, cand_clr)
        f_inf = extract_gate_features(obs_inf, cand_inh, cand_clr)

        for f_bad in (f_poison, f_inf):
            self.assertEqual(float(np.max(np.abs(f_bad.scalar_feats - f_clean.scalar_feats))), 0.0)
            self.assertEqual(float(np.max(np.abs(f_bad.complex_feats - f_clean.complex_feats))), 0.0)
            self.assertEqual(float(np.max(np.abs(f_bad.ev_base - f_clean.ev_base))), 0.0)
            self.assertEqual(float(np.max(np.abs(f_bad.base_logit - f_clean.base_logit))), 0.0)

        for v_idx, v in enumerate(VARIANTS):
            torch.manual_seed(9120 + v_idx)
            model = TemporalTrustGate(variant=v, d_model=12, n_heads=4, n_layers=1)
            _perturb_model(model, seed=9220 + v_idx, eps=0.2)
            out_clean = model.gate(obs, cand_inh, cand_clr)
            out_poison = model.gate(obs_poison, cand_inh, cand_clr)
            out_inf = model.gate(obs_inf, cand_inh, cand_clr)
            for out_bad in (out_poison, out_inf):
                self.assertEqual(float(np.max(np.abs(out_bad.history_weight - out_clean.history_weight))), 0.0)
                self.assertEqual(abs(out_bad.probe_weight - out_clean.probe_weight), 0.0)
                self.assertEqual(float(np.max(np.abs(out_bad.mix_llrs - out_clean.mix_llrs))), 0.0)
                self.assertEqual(float(np.max(np.abs(out_bad.mix_probe_pred - out_clean.mix_probe_pred))), 0.0)
                self.assertEqual(float(np.max(np.abs(out_bad.mix_probe_var - out_clean.mix_probe_var))), 0.0)

    def test_05_locality_vs_global_reach(self):
        """Perturbing row 0 causes EXACT zero change at row 18 in 1-layer local, while transport_attn responds."""
        obs, _ = simulate_p9_link(SEED_P9_DEV + 90201, "partial_shift", snr_db=9.0, budget="sparse")
        cand_inh, cand_clr = build_candidates(obs)
        self.assertNotIn(0, obs.training_rows)

        # End-to-end observation perturbation on row 0 (data row in sparse budget).
        y_pert = obs.y_b.copy()
        y_pert[0] = y_pert[0] + 5.0 - 4.0j
        obs_pert = replace(obs, y_b=y_pert)
        llrs_inh_p = cand_inh.llrs.copy()
        llrs_inh_p[0] = -llrs_inh_p[0] + 3.0
        ci_pert = replace(cand_inh, llrs=llrs_inh_p)

        torch.manual_seed(9130)
        m_local = TemporalTrustGate(variant="local", d_model=12, n_heads=4, n_layers=1)
        _perturb_model(m_local, seed=9230, eps=0.25)

        torch.manual_seed(9130)
        m_trans = TemporalTrustGate(variant="transport_attn", d_model=12, n_heads=4, n_layers=1)
        _perturb_model(m_trans, seed=9230, eps=0.25)

        out_lo_0 = m_local.gate(obs, cand_inh, cand_clr)
        out_lo_1 = m_local.gate(obs_pert, ci_pert, cand_clr)
        out_tr_0 = m_trans.gate(obs, cand_inh, cand_clr)
        out_tr_1 = m_trans.gate(obs_pert, ci_pert, cand_clr)

        # 1-layer local (dt in {-1, 0, +1}): row 0 perturbation has EXACT 0.0 effect on rows 2..19!
        self.assertEqual(float(abs(out_lo_1.history_weight[18] - out_lo_0.history_weight[18])), 0.0)
        self.assertEqual(float(np.max(np.abs(out_lo_1.history_weight[2:] - out_lo_0.history_weight[2:]))), 0.0)
        self.assertGreater(float(abs(out_lo_1.history_weight[0] - out_lo_0.history_weight[0])), 1e-6)

        # Global transport_attn responds immediately at row 18 and probe row 19 (both logit and weight)!
        self.assertGreater(float(abs(out_tr_1.history_weight[18] - out_tr_0.history_weight[18])), 1e-6)
        self.assertGreater(float(abs(out_tr_1.probe_weight - out_tr_0.probe_weight)), 1e-6)
        f_tr_0 = m_trans.extract_features_for_variant(obs, cand_inh, cand_clr)
        f_tr_1 = m_trans.extract_features_for_variant(obs_pert, ci_pert, cand_clr)
        with torch.no_grad():
            _, log_tr_0 = m_trans.forward_tensors(
                torch.from_numpy(f_tr_0.scalar_feats).to(REAL).unsqueeze(0),
                torch.from_numpy(f_tr_0.complex_feats).to(COMPLEX).unsqueeze(0),
                torch.from_numpy(f_tr_0.t_norm).to(REAL).unsqueeze(0),
                torch.from_numpy(f_tr_0.ev_base).to(REAL).unsqueeze(0),
                torch.from_numpy(f_tr_0.base_logit).to(REAL).unsqueeze(0),
            )
            _, log_tr_1 = m_trans.forward_tensors(
                torch.from_numpy(f_tr_1.scalar_feats).to(REAL).unsqueeze(0),
                torch.from_numpy(f_tr_1.complex_feats).to(COMPLEX).unsqueeze(0),
                torch.from_numpy(f_tr_1.t_norm).to(REAL).unsqueeze(0),
                torch.from_numpy(f_tr_1.ev_base).to(REAL).unsqueeze(0),
                torch.from_numpy(f_tr_1.base_logit).to(REAL).unsqueeze(0),
            )
        self.assertGreater(float((log_tr_1[0, 18] - log_tr_0[0, 18]).abs()), 1e-3)

    def test_06_no_history_zero_frame_a_sensitivity(self):
        """Replacing or corrupting cand_inh has EXACT zero effect on no_history's predicted weights w[T]."""
        obs, _ = simulate_p9_link(SEED_P9_DEV + 90501, "continuous", snr_db=9.0, budget="sparse")
        cand_inh, cand_clr = build_candidates(obs)
        cand_inh_corrupt = replace(
            cand_inh,
            pre_train_loglik=cand_inh.pre_train_loglik - 500.0,
            mean_ray=cand_inh.mean_ray * (-3.7 + 2.1j),
            llrs=-cand_inh.llrs,
        )

        torch.manual_seed(9140)
        m_nh = TemporalTrustGate(variant="no_history", d_model=12, n_heads=4, n_layers=1)
        _perturb_model(m_nh, seed=9240, eps=0.2)

        o1 = m_nh.gate(obs, cand_inh, cand_clr)
        o2 = m_nh.gate(obs, cand_inh_corrupt, cand_clr)
        self.assertEqual(float(np.max(np.abs(o1.history_weight - o2.history_weight))), 0.0)
        self.assertEqual(abs(o1.probe_weight - o2.probe_weight), 0.0)

    def test_07_total_variance_decomposition_and_signed_transport(self):
        """Verify exact within+between variance identity and signed U(1) parallel transport coherence."""
        obs, _ = simulate_p9_link(SEED_P9_DEV + 90601, "partial_shift", snr_db=9.0, budget="sparse")
        cand_inh, cand_clr = build_candidates(obs)
        torch.manual_seed(9150)
        model = TemporalTrustGate(variant="transport_attn", d_model=12, n_heads=4, n_layers=1)
        _perturb_model(model, seed=9250, eps=0.25)
        out = model.gate(obs, cand_inh, cand_clr)

        wp = out.probe_weight
        v_within = wp * cand_inh.probe_var + (1.0 - wp) * cand_clr.probe_var
        v_between = wp * (1.0 - wp) * np.abs(cand_inh.probe_pred - cand_clr.probe_pred) ** 2
        self.assertLess(float(np.max(np.abs(out.mix_probe_var - (v_within + v_between)))), 1e-14)
        self.assertTrue(bool(np.all(v_between >= 0.0)))

        # Direct U(1) parallel transport coherence check on a rotating wave v_j = exp(i * omega * t_j).
        torch.manual_seed(9151)
        layer_tr = TemporalTransportLayer("transport_attn", d_model=4, d_scalar=4, n_heads=1, d_ff=4)
        layer_ab = TemporalTransportLayer("abs_no_transport", d_model=4, d_scalar=4, n_heads=1, d_ff=4)
        with torch.no_grad():
            eye_c = torch.eye(4, dtype=COMPLEX)
            for lyr in (layer_tr, layer_ab):
                lyr.w_q.copy_(eye_c)
                lyr.w_k.copy_(eye_c)
                lyr.w_v.copy_(eye_c)
                lyr.w_o.copy_(eye_c)
                lyr.omega.fill_(math.pi)       # 1 full cycle over t_norm in [-1, 1]
                lyr.raw_beta.fill_(-20.0)      # flat global geodesic
                lyr.gamma_ev.fill_(0.0)
                lyr.ev_proj.zero_()
                lyr.ff1.zero_()
                lyr.ff2.zero_()
        t_norm = torch.linspace(-1.0, 1.0, T, dtype=REAL).unsqueeze(0)
        wave = torch.polar(torch.ones((1, T, 4), dtype=REAL), math.pi * t_norm[..., None].expand(1, T, 4))
        s_zero = torch.zeros((1, T, 4), dtype=REAL)
        ev_zero = torch.zeros((1, T), dtype=REAL)
        with torch.no_grad():
            h_tr, _ = layer_tr(wave, s_zero, t_norm, ev_zero)
            h_ab, _ = layer_ab(wave, s_zero, t_norm, ev_zero)
        # Transported residual delta_h aligns constructively with wave -> |h_tr| ~ 2.0,
        # whereas untransported average of exp(i pi t) nearly cancels -> |h_ab| ~ 1.0!
        self.assertGreater(float(h_tr.abs().mean()), 1.95)
        self.assertLess(float(h_ab.abs().mean()), 1.15)

    def test_08_full_active_finite_nonzero_gradients_all_variants(self):
        """Every trainable parameter receives active, finite, nonzero gradients across all 4 variants."""
        pairs = [
            (SEED_P9_DEV + 90701, "continuous"),
            (SEED_P9_DEV + 90702, "partial_shift"),
            (SEED_P9_DEV + 90703, "full_reset"),
            (SEED_P9_DEV + 90704, "partial_shift"),
        ]
        items = build_dataset(pairs, snr_db=9.0, budget="sparse")
        tensors_full = _stack_training_tensors(items, use_nohist=False)
        tensors_nh = _stack_training_tensors(items, use_nohist=True)
        idx = torch.arange(len(items), dtype=torch.long)

        for v_idx, variant in enumerate(VARIANTS):
            tensors = tensors_nh if variant == "no_history" else tensors_full
            for phase_name, do_perturb in (("step0", False), ("perturbed", True)):
                torch.manual_seed(9160 + v_idx)
                model = TemporalTrustGate(variant=variant, d_model=12, n_heads=4, n_layers=1)
                if do_perturb:
                    _perturb_model(model, seed=9260 + v_idx, eps=0.1)

                model.zero_grad(set_to_none=True)
                w, logit = model.forward_tensors(
                    tensors["scalar_feats"],
                    tensors["complex_feats"],
                    tensors["t_norm"],
                    tensors["ev_base"],
                    tensors["base_logit"],
                )
                loss, _, _ = _joint_loss_batch(w, logit, tensors, idx)
                loss.backward()

                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    self.assertIsNotNone(p.grad, f"{variant} ({phase_name}): missing grad for {name}")
                    self.assertTrue(
                        bool(torch.isfinite(p.grad).all()),
                        f"{variant} ({phase_name}): nonfinite grad for {name}",
                    )
                    gnorm = float(p.grad.abs().max())
                    self.assertGreater(
                        gnorm,
                        0.0,
                        f"{variant} ({phase_name}): ZERO grad for {name}",
                    )

    def test_09_pre_update_covariance_exact_and_phase_continuity(self):
        """Verify exact Hermitian covariance orientation (no conjugate transpose error) & exact phase continuity."""
        from p8.link import EVEN, FRAME_DURATION, PILOT_ROWS
        from p8.tracker import advance, observed_taps, prior, update
        from p8c.receiver import _training_dictionary
        from p9.p9c_dev import _pre_update_training_stats

        obs, _ = simulate_p9_link(SEED_P9_DEV + 90801, "continuous", snr_db=9.0, budget="sparse")
        tap_noise_a = (obs.n0 + 1.0 - float(np.sinc(0.0625) ** 2)) / len(EVEN)
        p_inh = advance(update(prior(), PILOT_ROWS, observed_taps(obs.y_a, PILOT_ROWS), row_noise=tap_noise_a), FRAME_DURATION)
        p_clr = advance(prior(), FRAME_DURATION)

        y_safe = obs.y_b.copy()
        y_safe[PROBE_ROW] = 0.0
        for st in (p_inh, p_clr):
            ll_fast, pred_fast, var_fast = _pre_update_training_stats(st, y_safe, obs.training_rows, obs.n0)
            design, _, _ = _training_dictionary(st.epoch, obs.training_rows)
            w = np.exp(st.log_weight)
            m = len(obs.training_rows) * N
            mu_r = np.einsum("lh,lhm->lm", w * st.mean, design)
            mu_y = mu_r.sum(axis=0)
            cov_ref = np.eye(m, dtype=np.complex128) * obs.n0
            for l in range(design.shape[0]):
                for h in range(design.shape[1]):
                    sec = w[l, h] * (abs(st.mean[l, h]) ** 2 + st.variance[l, h])
                    cov_ref += sec * np.outer(design[l, h], design[l, h].conj())
                cov_ref -= np.outer(mu_r[l], mu_r[l].conj())
            chol = np.linalg.cholesky(0.5 * (cov_ref + cov_ref.conj().T))
            y_vec = y_safe[list(obs.training_rows)].reshape(-1)
            z = np.linalg.solve(chol, y_vec - mu_y)
            logdet = 2.0 * float(np.sum(np.log(np.real(np.diag(chol)))))
            ll_ref = float(-m * math.log(math.pi) - logdet - float(np.real(np.vdot(z, z))))
            self.assertLess(abs(ll_fast - ll_ref), 1e-12)
            self.assertLess(float(np.max(np.abs(pred_fast - mu_y.reshape(-1, N)))), 1e-13)
            self.assertLess(float(np.max(np.abs(var_fast - np.real(np.diag(cov_ref)).reshape(-1, N)))), 1e-13)


if __name__ == "__main__":
    unittest.main()
