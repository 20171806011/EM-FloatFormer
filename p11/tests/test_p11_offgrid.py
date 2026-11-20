"""Strict mathematical, physical, symmetry, locality, checkpoint, and anti-leakage unit tests for P11 off-grid Pillar 2."""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import math
import os
from pathlib import Path
from typing import Tuple
import unittest

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from p8 import link as ofdm_link
from p8.link import DELAYS as P8_DELAYS
from p8.link import clock, draw_channel, stream, traffic
from p8c import link as sc_link
from p9.independent_ref import check_no_hardcoded_n0
from p11.contract import (
    N,
    OFFGRID_ALL_REGIMES,
    OFFGRID_ANCHOR_ROWS,
    OFFGRID_DELAY_MAX,
    OFFGRID_DELAY_MIN,
    OFFGRID_REGIMES_ID,
    OFDM_PILOT_ROWS,
    P11_NAMESPACE_SPAN,
    PROBE_ROW,
    SEED_P11_DEV,
    T,
    OffGridObservation,
    build_offgrid_sc_channel_matrix,
    enforce_determinism,
    fractional_delay_freq_response,
    fractional_delay_matrix,
    is_historical_seed,
    unitary_dft_matrix,
)
from p11.manifold_gate import (
    COMPLEX,
    REAL,
    VARIANTS,
    ManifoldTransportLayer,
    ManifoldTrustGate,
    differentiable_offgrid_loss,
    extract_manifold_features,
)
from p11.offgrid_channel import (
    K_HAT,
    MIN_DELAY_SEP,
    ROW_STEP,
    OffGridAtomEstimates,
    OffGridBranchBundle,
    build_offgrid_branches,
    classical_per_ray_gate,
    discover_frame_a_atoms,
    eval_piecewise_ray_phase_and_gain,
    offgrid_ofdm_propagate,
    offgrid_sc_propagate,
    permute_branch_bundle,
    simulate_offgrid_link,
)


EXPECTED_CKPT_SHA256: dict[str, str] = {
    "em10_abs_no_transport_seed1001.pt": "283e83ca0ef58fbe54baf9df3c878b06ced5fb5bd754651c7b253f24e736a386",
    "em10_abs_no_transport_seed1002.pt": "25efbb58964554587dd1c8dde5902c848210afd7c1236ed107150f86f8f867e7",
    "em10_abs_no_transport_seed1003.pt": "b57d1a8dc486d02782382fa95053a4a581d9e4b0fe2b00b2c5895de59d6b38d8",
    "em10_local_seed1001.pt": "4e72265f58ea29e4d52efcafdfff5875c1bdf499302537b0594dcab441495a45",
    "em10_local_seed1002.pt": "d7651ec1ec7870dac7516be5f2a68ff0dbd867ed76cf6dadd8a448750e283a0b",
    "em10_local_seed1003.pt": "70dba0d40470bb6fa9f44f55c114abc7416459ddfa19c34fb4bf3100b784582e",
    "em10_no_history_seed1001.pt": "a67e920ad4b9574cb6d66406ffe16c8c8d8f1638e82e4a704f3d8003c0840d97",
    "em10_no_history_seed1002.pt": "1497473c127396dc09fdf5fae46f7357bcf14452972f18d3ca42da3be0d0ef8d",
    "em10_no_history_seed1003.pt": "703a5f4b3fced7aabe8d57ece6d072ad27a7520ab246887293a6dd5c79f392e5",
    "em10_transport_attn_seed1001.pt": "0dc4f2ab45eb1e5d589adb2a4a25b13fe98d1cc87ee561b7e96749aaa4f7c4de",
    "em10_transport_attn_seed1002.pt": "9a00eddda556e49756db37a8def8a9f57ecdc4cb3750d59f10aee7f216b2c740",
    "em10_transport_attn_seed1003.pt": "b671efe4e95bf1c40d5bc3e2a17974538eabf9a7fc27f765e40ded4efbfba0d1",
}
EXPECTED_CKPT_BYTES = 65_863
EXPECTED_CKPT_ARCH = {"d_model": 12, "d_scalar": 12, "n_heads": 4, "n_layers": 1, "d_ff": 12}
EXPECTED_CKPT_PARAMS = 3153


def _dev_seed(offset: int) -> int:
    seed = int(SEED_P11_DEV + offset)
    if not (SEED_P11_DEV <= seed < SEED_P11_DEV + P11_NAMESPACE_SPAN):
        raise AssertionError(f"Seed {seed} out of P11 DEV namespace")
    if is_historical_seed(seed) is not None:
        raise AssertionError(f"Seed {seed} collided with historical seed namespace")
    return seed


def _scale_obs_and_bundle(
    obs: OffGridObservation,
    bundle: OffGridBranchBundle,
    alpha: complex,
) -> Tuple[OffGridObservation, OffGridBranchBundle]:
    """Apply exact physical C* scaling (y -> alpha*y, n0 -> |alpha|^2*n0) to obs & OffGridBranchBundle."""
    a = complex(alpha)
    a2 = abs(a) ** 2
    m_tr = len(obs.training_rows) * N
    ll_shift = -float(m_tr) * math.log(a2)

    obs_s = replace(
        obs,
        y_a=obs.y_a * a,
        y_b=obs.y_b * a,
        n0=obs.n0 * a2,
    )
    atoms_s = replace(
        bundle.atoms,
        g_a_mid=bundle.atoms.g_a_mid * a,
        v_a_mid=bundle.atoms.v_a_mid * a2,
        p_atom_a=bundle.atoms.p_atom_a * a2,
        p_tot_a=bundle.atoms.p_tot_a * a2,
        mu_bank_a=bundle.atoms.mu_bank_a * a,
        var_bank_a=bundle.atoms.var_bank_a * a2,
    )
    bundle_s = replace(
        bundle,
        atoms=atoms_s,
        mean_inh=bundle.mean_inh * a,
        var_inh=bundle.var_inh * a2,
        mean_clr=bundle.mean_clr * a,
        var_clr=bundle.var_clr * a2,
        pre_pred_inh=bundle.pre_pred_inh * a,
        pre_var_inh=bundle.pre_var_inh * a2,
        pre_pred_clr=bundle.pre_pred_clr * a,
        pre_var_clr=bundle.pre_var_clr * a2,
        pre_ll_inh=bundle.pre_ll_inh + ll_shift,
        pre_ll_clr=bundle.pre_ll_clr + ll_shift,
        subset_logliks=bundle.subset_logliks + ll_shift,
    )
    return obs_s, bundle_s


def _perturb_model(model: ManifoldTrustGate, seed: int = 102026, eps: float = 0.18) -> None:
    """Move trainable parameters away from frozen initial_map so residual outputs are non-trivial."""
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


class TestP11ManifoldAndOffGrid(unittest.TestCase):
    def setUp(self):
        self._prev_threads = torch.get_num_threads()
        self._prev_det = torch.are_deterministic_algorithms_enabled()
        enforce_determinism(require_env=False)

    def tearDown(self):
        torch.set_num_threads(self._prev_threads)
        torch.use_deterministic_algorithms(self._prev_det)

    def test_01_fractional_delay_duality_and_p9_integer_match(self):
        """Exact CP-SC vs OFDM fractional-delay duality (< 1e-14) & exact integer match to P8/P9."""
        f_mat = unitary_dft_matrix(N)
        eye_n = np.eye(N, dtype=np.complex128)
        rng = np.random.default_rng(101001)

        # 1. Exact unitarity & cross-waveform duality for continuous off-grid delays.
        test_delays = np.array([0.0, 0.37, 1.85, 3.42, 6.49], dtype=np.float64)
        for tau in test_delays:
            d_mat = fractional_delay_matrix(float(tau), n=N)
            phase = fractional_delay_freq_response(float(tau), n=N)
            unit_err = float(np.max(np.abs(d_mat.conj().T @ d_mat - eye_n)))
            dual_err = float(np.max(np.abs(f_mat @ d_mat @ f_mat.conj().T - np.diag(phase))))
            self.assertLess(unit_err, 1e-14, f"Unitarity failed for tau={tau}: {unit_err}")
            self.assertLess(dual_err, 1e-14, f"Duality failed for tau={tau}: {dual_err}")

        # 2. Static-gain OFDM duality: offgrid_ofdm_propagate == sum_l g_l * exp(-2j*pi*k_s*tau_l/N) * X
        x_grid = (rng.normal(size=(T, N)) + 1j * rng.normal(size=(T, N))) / math.sqrt(2.0)
        g_const = (rng.normal(size=len(test_delays)) + 1j * rng.normal(size=len(test_delays))) / math.sqrt(2.0)
        g_const = g_const / np.linalg.norm(g_const)
        g_arr = np.broadcast_to(g_const[:, None, None], (len(test_delays), T, N)).copy()
        y_ofdm = offgrid_ofdm_propagate(x_grid, g_arr, test_delays)
        h_freq_exact = np.sum(g_const[:, None] * fractional_delay_freq_response(test_delays, n=N), axis=0)
        y_ofdm_ref = x_grid * h_freq_exact[None, :]
        rel_ofdm = float(np.max(np.abs(y_ofdm - y_ofdm_ref)) / max(float(np.max(np.abs(y_ofdm_ref))), 1.0))
        self.assertLess(rel_ofdm, 1e-14)

        # 3. Exact integer-delay match to P8/P9 for DELAYS = (0, 2, 4) with full time-varying Doppler/chirp!
        s_link = _dev_seed(101)
        ch = draw_channel(stream(s_link, 0))
        words_a = traffic(stream(s_link, 1))
        grid_a = ofdm_link.tx_grid(words_a, OFDM_PILOT_ROWS)
        t_a = clock(T, start_row=0)
        _, gains_a = eval_piecewise_ray_phase_and_gain(t_a, ch.gain, ch.frequency, ch.chirp)
        y_a_offgrid = offgrid_ofdm_propagate(grid_a, gains_a, P8_DELAYS.astype(np.float64))
        y_a_p8 = ofdm_link.propagate(grid_a, ch, start_row=0)
        rel_a = float(np.max(np.abs(y_a_offgrid - y_a_p8)) / max(float(np.max(np.abs(y_a_p8))), 1.0))
        self.assertLess(rel_a, 1e-14)

        words_b = traffic(stream(s_link, 3))
        grid_b = sc_link.tx_grid(
            ofdm_link.Traffic(words_b.info[:18], words_b.code[:18], words_b.symbols[:18]), (10,)
        )
        t_b = clock(T, start_row=T)
        _, gains_b = eval_piecewise_ray_phase_and_gain(t_b, ch.gain, ch.frequency, ch.chirp)
        y_b_offgrid, _ = offgrid_sc_propagate(grid_b, gains_b, P8_DELAYS.astype(np.float64))
        y_b_p8c = sc_link.propagate(grid_b, ch, start_row=T)
        rel_b = float(np.max(np.abs(y_b_offgrid - y_b_p8c)) / max(float(np.max(np.abs(y_b_p8c))), 1.0))
        self.assertLess(rel_b, 1e-14)

    def test_02_exact_phase_continuity_and_all_six_regimes(self):
        """Exact phase continuity across Frame A->B boundary & mid-frame shifts, plus 6-regime schema."""
        gains_a = np.array([0.8 * np.exp(0.4j), 0.45 * np.exp(-1.1j), 0.3 * np.exp(2.2j)], dtype=np.complex128)
        freqs_a = np.array([0.062, -0.051, 0.074], dtype=np.float64)
        chirps_a = np.array([0.0003, -0.0004, 0.0002], dtype=np.float64)
        freqs_post = np.array([0.062, 0.068, -0.055], dtype=np.float64)
        chirps_post = np.array([0.0003, 0.0001, -0.0002], dtype=np.float64)
        changed = (1, 2)

        for shift_row in (0, 8):
            t_s = float(T + shift_row) * ROW_STEP
            h = 1e-7
            t_probe = np.array([t_s - h, t_s, t_s + h], dtype=np.float64)
            phi_p, g_p = eval_piecewise_ray_phase_and_gain(
                t_probe,
                gains_a,
                freqs_a,
                chirps_a,
                changed_rays=changed,
                t_shift=t_s,
                freqs_post=freqs_post,
                chirps_post=chirps_post,
            )
            phi_left = 2.0 * np.pi * (freqs_a * t_s + 0.5 * chirps_a * (t_s ** 2))
            g_left = gains_a * np.exp(1j * phi_left)
            self.assertLess(float(np.max(np.abs(phi_p[:, 1] - phi_left))), 1e-13)
            self.assertLess(float(np.max(np.abs(g_p[:, 1] - g_left))), 1e-13)

            for l_idx in range(3):
                f_minus = float((phi_p[l_idx, 1] - phi_p[l_idx, 0]) / (2.0 * np.pi * h))
                f_plus = float((phi_p[l_idx, 2] - phi_p[l_idx, 1]) / (2.0 * np.pi * h))
                f_at_s = float(freqs_a[l_idx] + chirps_a[l_idx] * t_s)
                expected_plus = float(freqs_post[l_idx]) if l_idx in changed else f_at_s
                self.assertLess(abs(f_minus - f_at_s), 1e-6)
                self.assertLess(abs(f_plus - expected_plus), 1e-6)

        for r_idx, reg in enumerate(OFFGRID_ALL_REGIMES):
            obs, truth = simulate_offgrid_link(_dev_seed(200 + r_idx), reg)
            self.assertEqual(obs.y_a.shape, (T, N))
            self.assertEqual(obs.y_b.shape, (T, N))
            self.assertTrue(bool(np.isnan(obs.y_b[PROBE_ROW].real).all() and np.isnan(obs.y_b[PROBE_ROW].imag).all()))
            self.assertIn(obs.training_rows[0], OFFGRID_ANCHOR_ROWS)
            self.assertEqual(len(truth.data_rows), 18)
            self.assertNotIn(PROBE_ROW, truth.data_rows)
            self.assertNotIn(obs.training_rows[0], truth.data_rows)
            self.assertTrue(bool(np.all(truth.delays >= OFFGRID_DELAY_MIN) and np.all(truth.delays <= OFFGRID_DELAY_MAX)))
            self.assertGreaterEqual(float(np.min(np.diff(truth.delays))), MIN_DELAY_SEP - 1e-12)
            self.assertLess(abs(float(truth.powers.sum()) - 1.0), 1e-14)
            if reg == "offgrid_ood_five_ray_fast":
                self.assertEqual(truth.n_rays, 5)
            if reg == "offgrid_ood_burst_and_shift":
                self.assertEqual(len(truth.corrupted_rows), 2)
                for cr in truth.corrupted_rows:
                    self.assertIn(cr, truth.data_rows)

    def test_03_exact_integer_real_parameter_parity(self):
        """Exact integer real parameter parity across all 4 variants (== 3,153)."""
        counts = {}
        for v in VARIANTS:
            torch.manual_seed(10103)
            m = ManifoldTrustGate(variant=v)
            counts[v] = m.real_parameters()

        self.assertEqual(counts["em10_transport_attn"], EXPECTED_CKPT_PARAMS)
        self.assertEqual(counts["em10_transport_attn"], counts["em10_abs_no_transport"])
        self.assertEqual(counts["em10_transport_attn"], counts["em10_local"])
        self.assertEqual(counts["em10_transport_attn"], counts["em10_no_history"])

    def test_04_step0_identity_to_classical_per_ray_gate(self):
        """Untrained ManifoldTrustGate matches classical_per_ray_gate to < 1e-13 across all ID regimes."""
        for r_idx, reg in enumerate(OFFGRID_REGIMES_ID):
            obs, _ = simulate_offgrid_link(_dev_seed(400 + r_idx), reg)
            bundle = build_offgrid_branches(obs)
            ref = classical_per_ray_gate(obs, bundle)

            for v in ("em10_transport_attn", "em10_abs_no_transport", "em10_local"):
                torch.manual_seed(10104)
                model = ManifoldTrustGate(variant=v)
                out = model.gate(obs, bundle)

                self.assertLess(float(np.max(np.abs(out.w_ray - ref.w_ray))), 1e-13)
                self.assertLess(float(np.max(np.abs(out.noise_mult - ref.noise_mult))), 1e-13)
                self.assertLess(float(np.max(np.abs(out.h_mix - ref.h_mix))), 1e-13)
                self.assertLess(float(np.max(np.abs(out.noise_mix - ref.noise_mix))), 1e-13)
                self.assertLess(float(np.max(np.abs(out.gate_out.mix_llrs - ref.gate_out.mix_llrs))), 1e-13)
                self.assertLess(float(np.max(np.abs(out.gate_out.mix_probe_pred - ref.gate_out.mix_probe_pred))), 1e-13)
                self.assertLess(float(np.max(np.abs(out.gate_out.mix_probe_var - ref.gate_out.mix_probe_var))), 1e-13)

    def test_05_strict_cstar_gauge_invariance_and_equivariance(self):
        """Strict C* = R+ x U(1) invariance (< 1e-12) on w_ray & mix_llrs and equivariance on probe/H."""
        obs, _ = simulate_offgrid_link(_dev_seed(501), "offgrid_partial_shift")
        bundle = build_offgrid_branches(obs)
        alphas = (
            1e-12 * np.exp(1j * 0.7),
            1e10 * np.exp(-1j * 2.1),
        )

        # 1. Perturbed neural gate C* invariance & equivariance across ALL 4 variants.
        for v_idx, v in enumerate(VARIANTS):
            torch.manual_seed(10150 + v_idx)
            model = ManifoldTrustGate(variant=v)
            _perturb_model(model, seed=10250 + v_idx, eps=0.22)
            base = model.gate(obs, bundle)

            if v != "em10_no_history":
                cls_ref = classical_per_ray_gate(obs, bundle)
                self.assertGreater(float(np.max(np.abs(base.w_ray - cls_ref.w_ray))), 1e-4)

            for alpha in alphas:
                a2 = abs(alpha) ** 2
                obs_s, bundle_s = _scale_obs_and_bundle(obs, bundle, alpha)
                sc = model.gate(obs_s, bundle_s)

                self.assertLess(float(np.max(np.abs(sc.w_ray - base.w_ray))), 1e-12)
                self.assertLess(float(np.max(np.abs(sc.noise_mult - base.noise_mult))), 1e-12)
                self.assertLess(float(np.max(np.abs(sc.gate_out.mix_llrs - base.gate_out.mix_llrs))), 1e-12)

                h_ref = alpha * base.h_mix
                nm_ref = a2 * base.noise_mix
                pp_ref = alpha * base.gate_out.mix_probe_pred
                pv_ref = a2 * base.gate_out.mix_probe_var

                self.assertLess(float(np.max(np.abs(sc.h_mix - h_ref)) / max(float(np.max(np.abs(h_ref))), 1e-300)), 1e-12)
                self.assertLess(float(np.max(np.abs(sc.noise_mix - nm_ref)) / max(float(np.max(np.abs(nm_ref))), 1e-300)), 1e-12)
                self.assertLess(float(np.max(np.abs(sc.gate_out.mix_probe_pred - pp_ref)) / max(float(np.max(np.abs(pp_ref))), 1e-300)), 1e-12)
                self.assertLess(float(np.max(np.abs(sc.gate_out.mix_probe_var - pv_ref)) / max(float(np.max(np.abs(pv_ref))), 1e-300)), 1e-12)

        # 2. End-to-end blind OMP+RELAX+branch-builder C* invariance directly from scaled OffGridObservation!
        torch.manual_seed(10159)
        m_e2e = ManifoldTrustGate(variant="em10_transport_attn")
        _perturb_model(m_e2e, seed=10259, eps=0.2)
        base_e2e = m_e2e.gate(obs, bundle)
        for alpha in alphas:
            a2 = abs(alpha) ** 2
            obs_scaled = replace(obs, y_a=obs.y_a * alpha, y_b=obs.y_b * alpha, n0=obs.n0 * a2)
            bundle_e2e = build_offgrid_branches(obs_scaled)
            out_e2e = m_e2e.gate(obs_scaled, bundle_e2e)
            self.assertLess(float(np.max(np.abs(bundle_e2e.tau_hat - bundle.tau_hat))), 1e-12)
            self.assertLess(float(np.max(np.abs(out_e2e.w_ray - base_e2e.w_ray))), 1e-12)
            e2e_nm_rel = float(
                np.max(np.abs(out_e2e.noise_mult - base_e2e.noise_mult))
                / max(float(np.max(np.abs(base_e2e.noise_mult))), 1.0)
            )
            self.assertLess(e2e_nm_rel, 1e-12)
            e2e_llr_diff = np.abs(out_e2e.gate_out.mix_llrs - base_e2e.gate_out.mix_llrs)
            e2e_llr_rel = float(np.max(e2e_llr_diff) / max(float(np.max(np.abs(base_e2e.gate_out.mix_llrs))), 1.0))
            self.assertLess(e2e_llr_rel, 1e-12)
            self.assertEqual(
                int(np.count_nonzero((out_e2e.gate_out.mix_llrs > 0.0) != (base_e2e.gate_out.mix_llrs > 0.0))),
                0,
            )

    def test_06_exact_ray_permutation_equivariance(self):
        """Permuting unordered discovered ray atoms permutes w_ray identically & keeps LLRs/probe invariant (< 1e-12)."""
        obs, _ = simulate_offgrid_link(_dev_seed(601), "offgrid_midframe_shift")
        bundle = build_offgrid_branches(obs)
        perms = ((3, 0, 4, 1, 2), (4, 3, 2, 1, 0))

        for v_idx, v in enumerate(VARIANTS):
            torch.manual_seed(10160 + v_idx)
            model = ManifoldTrustGate(variant=v)
            _perturb_model(model, seed=10260 + v_idx, eps=0.22)
            base = model.gate(obs, bundle)

            for perm in perms:
                p_idx = np.asarray(perm, dtype=np.int64)
                bundle_p = permute_branch_bundle(bundle, perm)
                out_p = model.gate(obs, bundle_p)

                w_err = float(np.max(np.abs(out_p.w_ray - base.w_ray[p_idx])))
                nm_err = float(np.max(np.abs(out_p.noise_mult - base.noise_mult)))
                h_err = float(np.max(np.abs(out_p.h_mix - base.h_mix)))
                nv_err = float(np.max(np.abs(out_p.noise_mix - base.noise_mix)))
                llr_err = float(np.max(np.abs(out_p.gate_out.mix_llrs - base.gate_out.mix_llrs)))
                pp_err = float(np.max(np.abs(out_p.gate_out.mix_probe_pred - base.gate_out.mix_probe_pred)))
                pv_err = float(np.max(np.abs(out_p.gate_out.mix_probe_var - base.gate_out.mix_probe_var)))

                self.assertLess(w_err, 1e-12, f"{v} perm={perm}: w_err={w_err}")
                self.assertLess(nm_err, 1e-12, f"{v} perm={perm}: nm_err={nm_err}")
                self.assertLess(h_err, 1e-12, f"{v} perm={perm}: h_err={h_err}")
                self.assertLess(nv_err, 1e-12, f"{v} perm={perm}: nv_err={nv_err}")
                self.assertLess(llr_err, 1e-12, f"{v} perm={perm}: llr_err={llr_err}")
                self.assertLess(pp_err, 1e-12, f"{v} perm={perm}: pp_err={pp_err}")
                self.assertLess(pv_err, 1e-12, f"{v} perm={perm}: pv_err={pv_err}")

    def test_07_zero_probe_leakage(self):
        """Poisoning obs.y_b[PROBE_ROW] causes EXACT 0.0 change in branches, features, and all 4 gates."""
        obs, _ = simulate_offgrid_link(_dev_seed(701), "offgrid_continuous")
        bundle_clean = build_offgrid_branches(obs)

        y_huge = obs.y_b.copy()
        y_huge[PROBE_ROW] = 1e9 - 7e8j
        obs_huge = replace(obs, y_b=y_huge)

        y_inf = obs.y_b.copy()
        y_inf[PROBE_ROW] = np.inf - 1j * np.inf
        obs_inf = replace(obs, y_b=y_inf)

        for obs_bad in (obs_huge, obs_inf):
            b_bad = build_offgrid_branches(obs_bad)
            self.assertEqual(float(np.max(np.abs(b_bad.mean_inh - bundle_clean.mean_inh))), 0.0)
            self.assertEqual(float(np.max(np.abs(b_bad.mean_clr - bundle_clean.mean_clr))), 0.0)
            self.assertEqual(float(np.max(np.abs(b_bad.w_ray_classical - bundle_clean.w_ray_classical))), 0.0)
            self.assertEqual(float(np.max(np.abs(b_bad.llrs_cls - bundle_clean.llrs_cls))), 0.0)

            f_clean = extract_manifold_features(obs, bundle_clean)
            f_bad = extract_manifold_features(obs_bad, b_bad)
            self.assertEqual(float(np.max(np.abs(f_bad.scalar_feats - f_clean.scalar_feats))), 0.0)
            self.assertEqual(float(np.max(np.abs(f_bad.complex_feats - f_clean.complex_feats))), 0.0)
            self.assertEqual(float(np.max(np.abs(f_bad.ev_base - f_clean.ev_base))), 0.0)

        for v_idx, v in enumerate(VARIANTS):
            torch.manual_seed(10170 + v_idx)
            model = ManifoldTrustGate(variant=v)
            _perturb_model(model, seed=10270 + v_idx, eps=0.22)
            out_clean = model.gate(obs, bundle_clean)
            for obs_bad in (obs_huge, obs_inf):
                out_bad = model.gate(obs_bad, bundle_clean)
                self.assertEqual(float(np.max(np.abs(out_bad.w_ray - out_clean.w_ray))), 0.0)
                self.assertEqual(float(np.max(np.abs(out_bad.noise_mult - out_clean.noise_mult))), 0.0)
                self.assertEqual(float(np.max(np.abs(out_bad.gate_out.mix_llrs - out_clean.gate_out.mix_llrs))), 0.0)
                self.assertEqual(float(np.max(np.abs(out_bad.gate_out.mix_probe_pred - out_clean.gate_out.mix_probe_pred))), 0.0)
                self.assertEqual(float(np.max(np.abs(out_bad.gate_out.mix_probe_var - out_clean.gate_out.mix_probe_var))), 0.0)

    def test_08_locality_vs_global_reach(self):
        """Perturbing roving anchor row 3 affects row 18 in 1-layer transport_attn, EXACT 0.0 in 1-layer local."""
        obs, _ = simulate_offgrid_link(_dev_seed(801), "offgrid_midframe_shift", anchor_row=3)
        self.assertEqual(obs.training_rows, (3,))
        bundle = build_offgrid_branches(obs)

        y_pert = obs.y_b.copy()
        y_pert[3] = y_pert[3] + 6.0 - 5.0j
        obs_pert = replace(obs, y_b=y_pert)

        torch.manual_seed(10180)
        m_local = ManifoldTrustGate(variant="em10_local", n_layers=1)
        _perturb_model(m_local, seed=10280, eps=0.25)

        torch.manual_seed(10180)
        m_trans = ManifoldTrustGate(variant="em10_transport_attn", n_layers=1)
        _perturb_model(m_trans, seed=10280, eps=0.25)

        out_lo_0 = m_local.gate(obs, bundle)
        out_lo_1 = m_local.gate(obs_pert, bundle)
        out_tr_0 = m_trans.gate(obs, bundle)
        out_tr_1 = m_trans.gate(obs_pert, bundle)

        self.assertEqual(float(np.max(np.abs(out_lo_1.w_ray[:, 18] - out_lo_0.w_ray[:, 18]))), 0.0)
        self.assertEqual(float(np.max(np.abs(out_lo_1.w_ray[:, 5:] - out_lo_0.w_ray[:, 5:]))), 0.0)
        self.assertEqual(float(np.max(np.abs(out_lo_1.w_ray[:, :2] - out_lo_0.w_ray[:, :2]))), 0.0)
        self.assertEqual(float(abs(out_lo_1.noise_mult[18] - out_lo_0.noise_mult[18])), 0.0)
        self.assertEqual(float(np.max(np.abs(out_lo_1.gate_out.mix_llrs[18] - out_lo_0.gate_out.mix_llrs[18]))), 0.0)
        self.assertEqual(float(np.max(np.abs(out_lo_1.gate_out.mix_probe_pred - out_lo_0.gate_out.mix_probe_pred))), 0.0)
        self.assertGreater(float(np.max(np.abs(out_lo_1.w_ray[:, 3] - out_lo_0.w_ray[:, 3]))), 1e-5)

        self.assertGreater(float(np.max(np.abs(out_tr_1.w_ray[:, 18] - out_tr_0.w_ray[:, 18]))), 1e-5)
        self.assertGreater(float(np.max(np.abs(out_tr_1.w_ray[:, PROBE_ROW] - out_tr_0.w_ray[:, PROBE_ROW]))), 1e-5)

        f0 = extract_manifold_features(obs, bundle)
        s0 = torch.from_numpy(f0.scalar_feats).to(REAL).unsqueeze(0)
        c0 = torch.from_numpy(f0.complex_feats).to(COMPLEX).unsqueeze(0)
        s1 = s0.clone()
        c1 = c0.clone()
        s1[:, 0, :, :] = s1[:, 0, :, :] + 1.5
        c1[:, 0, :, :] = c1[:, 0, :, :] * (1.5 - 0.8j)
        tau_t = torch.from_numpy(f0.tau_hat).to(REAL).unsqueeze(0)
        nu_t = torch.from_numpy(f0.nu_hat).to(REAL).unsqueeze(0)
        tp_t = torch.from_numpy(f0.t_phys).to(REAL).unsqueeze(0)
        tn_t = torch.from_numpy(f0.t_norm).to(REAL).unsqueeze(0)
        ev_t = torch.from_numpy(f0.ev_base).to(REAL).unsqueeze(0)
        bl_t = torch.from_numpy(f0.base_logit).to(REAL).unsqueeze(0)
        pr_t = torch.from_numpy(f0.is_probe).to(REAL).unsqueeze(0)

        with torch.no_grad():
            w_lo_a, _, _ = m_local.forward_tensors(s0, c0, tau_t, nu_t, tp_t, tn_t, ev_t, bl_t, pr_t)
            w_lo_b, _, _ = m_local.forward_tensors(s1, c1, tau_t, nu_t, tp_t, tn_t, ev_t, bl_t, pr_t)
            w_tr_a, _, _ = m_trans.forward_tensors(s0, c0, tau_t, nu_t, tp_t, tn_t, ev_t, bl_t, pr_t)
            w_tr_b, _, _ = m_trans.forward_tensors(s1, c1, tau_t, nu_t, tp_t, tn_t, ev_t, bl_t, pr_t)
        self.assertEqual(float((w_lo_b[:, 1:] - w_lo_a[:, 1:]).abs().max()), 0.0)
        self.assertGreater(float((w_tr_b[:, 1:] - w_tr_a[:, 1:]).abs().max()), 1e-4)

    def test_09_no_history_masking_total_variance_and_transport_coherence(self):
        """Verify no_history Frame-A isolation, exact within+between probe variance, & signed U(1) coherence."""
        obs, _ = simulate_offgrid_link(_dev_seed(901), "offgrid_partial_shift")
        bundle = build_offgrid_branches(obs)

        bundle_corrupt = replace(
            bundle,
            atoms=replace(bundle.atoms, atom_snr_a=bundle.atoms.atom_snr_a * 0.0 + 999.0),
            nu_hat=-bundle.nu_hat * 2.0,
            mean_inh=bundle.mean_inh * (-3.1 + 2.4j),
            var_inh=bundle.var_inh * 5.0,
            pre_pred_inh=bundle.pre_pred_inh * (-2.0),
            base_logit_ray=-bundle.base_logit_ray + 10.0,
            llrs_inh=-bundle.llrs_inh,
        )
        torch.manual_seed(10190)
        m_nh = ManifoldTrustGate(variant="em10_no_history")
        _perturb_model(m_nh, seed=10290, eps=0.2)
        o1 = m_nh.gate(obs, bundle)
        o2 = m_nh.gate(obs, bundle_corrupt)
        self.assertEqual(float(np.max(np.abs(o1.w_ray - o2.w_ray))), 0.0)
        self.assertEqual(float(np.max(np.abs(o1.noise_mult - o2.noise_mult))), 0.0)

        torch.manual_seed(10191)
        m_tr = ManifoldTrustGate(variant="em10_transport_attn")
        _perturb_model(m_tr, seed=10291, eps=0.25)
        out = m_tr.gate(obs, bundle)
        wp = out.w_ray[:, PROBE_ROW, None]
        v_within_ray = wp * bundle.var_inh[:, PROBE_ROW, :] + (1.0 - wp) * bundle.var_clr[:, PROBE_ROW, :]
        v_between_ray = wp * (1.0 - wp) * (np.abs(bundle.mean_inh[:, PROBE_ROW, :] - bundle.mean_clr[:, PROBE_ROW, :]) ** 2)
        v_probe_ref = obs.n0 + np.sum((v_within_ray + v_between_ray) * (np.abs(bundle.u_probe) ** 2), axis=0)
        self.assertLess(float(np.max(np.abs(out.gate_out.mix_probe_var - v_probe_ref))), 1e-14)
        self.assertTrue(bool(np.all(out.gate_out.mix_probe_var >= obs.n0 - 1e-15)))

        torch.manual_seed(10192)
        lyr_tr = ManifoldTransportLayer("em10_transport_attn", d_model=4, d_scalar=4, n_heads=1, d_ff=4)
        lyr_ab = ManifoldTransportLayer("em10_abs_no_transport", d_model=4, d_scalar=4, n_heads=1, d_ff=4)
        with torch.no_grad():
            eye_c = torch.eye(4, dtype=COMPLEX)
            for lyr in (lyr_tr, lyr_ab):
                lyr.w_q.copy_(eye_c)
                lyr.w_k.copy_(eye_c)
                lyr.w_v.copy_(eye_c)
                lyr.w_o.copy_(eye_c)
                lyr.s_nu.fill_(1.0)
                lyr.omega_t.fill_(0.0)
                lyr.df_eff.fill_(0.25)
                lyr.raw_beta_t.fill_(-20.0)
                lyr.raw_beta_tau.fill_(-20.0)
                lyr.gamma_ev.fill_(0.0)
                lyr.ev_proj.zero_()
                lyr.ff1.zero_()
                lyr.ff2.zero_()

        tau_t = torch.tensor([[0.5, 2.3, 4.7]], dtype=REAL)
        nu_t = torch.tensor([[0.065, 0.065, 0.065]], dtype=REAL)
        tp_t = (torch.arange(T, dtype=REAL) * ROW_STEP).unsqueeze(0)
        tn_t = torch.linspace(-1.0, 1.0, T, dtype=REAL).unsqueeze(0)
        ang = 2.0 * math.pi * (nu_t[:, :, None] * tp_t[:, None, :] - 0.25 * tau_t[:, :, None])
        wave = torch.polar(torch.ones((1, 3, T, 4), dtype=REAL), ang[..., None].expand(1, 3, T, 4))
        s_z = torch.zeros((1, 3, T, 4), dtype=REAL)
        ev_z = torch.zeros((1, 3, T), dtype=REAL)
        with torch.no_grad():
            h_tr, _ = lyr_tr(wave, s_z, tau_t, nu_t, tp_t, tn_t, ev_z)
            h_ab, _ = lyr_ab(wave, s_z, tau_t, nu_t, tp_t, tn_t, ev_z)
        self.assertGreater(float(h_tr.abs().mean()), 1.95)
        self.assertLess(float(h_ab.abs().mean()), 1.20)

    def test_10_active_finite_nonzero_gradients_and_static_audits(self):
        """Active finite nonzero grads across all 4 variants & static N0 / AST / precision / p10-import audits."""
        root = Path(__file__).resolve().parents[1]
        p11_offgrid_files = [
            root / "offgrid_channel.py",
            root / "manifold_gate.py",
            root / "tests" / "test_p11_offgrid.py",
        ]
        n0_rep = check_no_hardcoded_n0(p11_offgrid_files, strict=True)
        self.assertTrue(n0_rep["clean"], f"Hardcoded N0 violations: {n0_rep['violations']}")

        inf_mode_name = "inference_" + "mode"
        bad_dtypes = {"float" + "32", "complex" + "64"}
        seed_p10_prefix = "SEED_" + "P10"
        for fpath in p11_offgrid_files:
            tree = ast.parse(fpath.read_text(encoding="utf-8"), filename=str(fpath))
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    self.assertNotEqual(node.id, inf_mode_name, f"Forbidden {inf_mode_name} in {fpath}")
                    self.assertNotIn(node.id, bad_dtypes, f"Forbidden low-precision dtype {node.id} in {fpath}")
                    self.assertFalse(node.id.startswith(seed_p10_prefix), f"Forbidden {node.id} in {fpath}")
                elif isinstance(node, ast.Attribute):
                    self.assertNotEqual(node.attr, inf_mode_name, f"Forbidden {inf_mode_name} attr in {fpath}")
                    self.assertNotIn(node.attr, bad_dtypes, f"Forbidden low-precision attr {node.attr} in {fpath}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(
                            alias.name == "p10" or alias.name.startswith("p10."),
                            f"Forbidden p10 import {alias.name} in {fpath}",
                        )
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    self.assertFalse(
                        mod == "p10" or mod.startswith("p10."),
                        f"Forbidden p10 import-from {mod} in {fpath}",
                    )

        links = [
            simulate_offgrid_link(_dev_seed(1001), "offgrid_continuous"),
            simulate_offgrid_link(_dev_seed(1002), "offgrid_partial_shift"),
            simulate_offgrid_link(_dev_seed(1003), "offgrid_midframe_shift"),
            simulate_offgrid_link(_dev_seed(1004), "offgrid_full_reset"),
        ]
        bundles = [build_offgrid_branches(obs) for obs, _ in links]

        for v_idx, variant in enumerate(VARIANTS):
            mask_h = variant == "em10_no_history"
            feats = [extract_manifold_features(obs, b, mask_history=mask_h) for (obs, _), b in zip(links, bundles)]

            s_t = torch.from_numpy(np.stack([f.scalar_feats for f in feats])).to(REAL)
            c_t = torch.from_numpy(np.stack([f.complex_feats for f in feats])).to(COMPLEX)
            tau_t = torch.from_numpy(np.stack([f.tau_hat for f in feats])).to(REAL)
            nu_t = torch.from_numpy(np.stack([f.nu_hat for f in feats])).to(REAL)
            tp_t = torch.from_numpy(np.stack([f.t_phys for f in feats])).to(REAL)
            tn_t = torch.from_numpy(np.stack([f.t_norm for f in feats])).to(REAL)
            ev_t = torch.from_numpy(np.stack([f.ev_base for f in feats])).to(REAL)
            bl_t = torch.from_numpy(np.stack([f.base_logit for f in feats])).to(REAL)
            pr_t = torch.from_numpy(np.stack([f.is_probe for f in feats])).to(REAL)

            data_idx = torch.tensor([list(tr.data_rows) for _, tr in links], dtype=torch.long)
            mean_inh_bar = torch.from_numpy(
                np.stack([b.mean_inh[:, list(tr.data_rows), :].mean(axis=-1) for (_, tr), b in zip(links, bundles)])
            ).to(COMPLEX)
            var_inh_bar = torch.from_numpy(
                np.stack([b.var_inh[:, list(tr.data_rows), :].mean(axis=-1) for (_, tr), b in zip(links, bundles)])
            ).to(REAL)
            mean_clr_bar = torch.from_numpy(
                np.stack([b.mean_clr[:, list(tr.data_rows), :].mean(axis=-1) for (_, tr), b in zip(links, bundles)])
            ).to(COMPLEX)
            var_clr_bar = torch.from_numpy(
                np.stack([b.var_clr[:, list(tr.data_rows), :].mean(axis=-1) for (_, tr), b in zip(links, bundles)])
            ).to(REAL)
            phi_f = torch.from_numpy(np.stack([b.phi_f for b in bundles])).to(COMPLEX)
            y_data_freq = torch.from_numpy(
                np.stack([np.fft.fft(obs.y_b[list(tr.data_rows)], axis=-1, norm="ortho") for obs, tr in links])
            ).to(COMPLEX)
            b_code = torch.from_numpy(
                np.stack([tr.b_code.reshape(len(tr.data_rows), N, 4) for _, tr in links])
            ).to(REAL)
            mean_inh_p = torch.from_numpy(np.stack([b.mean_inh[:, PROBE_ROW, :] for b in bundles])).to(COMPLEX)
            var_inh_p = torch.from_numpy(np.stack([b.var_inh[:, PROBE_ROW, :] for b in bundles])).to(REAL)
            mean_clr_p = torch.from_numpy(np.stack([b.mean_clr[:, PROBE_ROW, :] for b in bundles])).to(COMPLEX)
            var_clr_p = torch.from_numpy(np.stack([b.var_clr[:, PROBE_ROW, :] for b in bundles])).to(REAL)
            u_probe = torch.from_numpy(np.stack([b.u_probe for b in bundles])).to(COMPLEX)
            probe_tgt = torch.from_numpy(np.stack([tr.y_b_probe_noisy for _, tr in links])).to(COMPLEX)
            n0_t = torch.tensor([obs.n0 for obs, _ in links], dtype=REAL)

            for phase_name, do_perturb in (("step0", False), ("perturbed", True)):
                torch.manual_seed(10195 + v_idx)
                model = ManifoldTrustGate(variant=variant)
                if do_perturb:
                    _perturb_model(model, seed=10295 + v_idx, eps=0.12)

                model.zero_grad(set_to_none=True)
                w_ray, _, noise_mult = model.forward_tensors(
                    s_t, c_t, tau_t, nu_t, tp_t, tn_t, ev_t, bl_t, pr_t
                )
                loss, _, _ = differentiable_offgrid_loss(
                    w_ray,
                    noise_mult,
                    mean_inh_bar=mean_inh_bar,
                    var_inh_bar=var_inh_bar,
                    mean_clr_bar=mean_clr_bar,
                    var_clr_bar=var_clr_bar,
                    phi_f=phi_f,
                    y_data_freq=y_data_freq,
                    b_code=b_code,
                    data_row_idx=data_idx,
                    mean_inh_probe=mean_inh_p,
                    var_inh_probe=var_inh_p,
                    mean_clr_probe=mean_clr_p,
                    var_clr_probe=var_clr_p,
                    u_probe=u_probe,
                    probe_target=probe_tgt,
                    n0=n0_t,
                )
                loss.backward()

                for name, p in model.named_parameters():
                    if not p.requires_grad:
                        continue
                    self.assertIsNotNone(p.grad, f"{variant} ({phase_name}): missing grad for {name}")
                    self.assertTrue(
                        bool(torch.isfinite(p.grad).all()),
                        f"{variant} ({phase_name}): nonfinite grad for {name}",
                    )
                    self.assertGreater(
                        float(p.grad.abs().max()),
                        0.0,
                        f"{variant} ({phase_name}): ZERO grad for {name}",
                    )

    def test_11_frozen_p10b_dev_checkpoints_integrity_and_determinism(self):
        """Verify all 12 ported checkpoints: SHA-256, weights_only=True, 3,153 params, float64/complex128, determinism, C*, & zero probe leak."""
        root = Path(__file__).resolve().parents[1]
        ckpt_dir = root / "ckpts" / "p10b_dev_ckpts"
        self.assertTrue(ckpt_dir.is_dir(), f"Missing checkpoint directory: {ckpt_dir}")

        found_files = sorted(p.name for p in ckpt_dir.glob("*.pt"))
        self.assertEqual(found_files, sorted(EXPECTED_CKPT_SHA256.keys()))
        self.assertEqual(len(found_files), 12)

        loaded_models: dict[str, ManifoldTrustGate] = {}
        for fname, expected_sha in EXPECTED_CKPT_SHA256.items():
            fpath = ckpt_dir / fname
            raw = fpath.read_bytes()
            self.assertEqual(len(raw), EXPECTED_CKPT_BYTES, f"Size mismatch for {fname}")
            actual_sha = hashlib.sha256(raw).hexdigest()
            self.assertEqual(actual_sha, expected_sha, f"SHA-256 mismatch for {fname}")

            payload = torch.load(fpath, map_location="cpu", weights_only=True)
            self.assertEqual(
                set(payload.keys()),
                {"variant", "seed", "real_parameters", "arch", "state_dict"},
                f"Unexpected payload keys in {fname}",
            )
            variant = str(payload["variant"])
            seed = int(payload["seed"])
            self.assertIn(variant, VARIANTS)
            self.assertIn(seed, (1001, 1002, 1003))
            self.assertEqual(fname, f"{variant}_seed{seed}.pt")
            self.assertEqual(int(payload["real_parameters"]), EXPECTED_CKPT_PARAMS)
            self.assertEqual(dict(payload["arch"]), EXPECTED_CKPT_ARCH)

            for tname, tensor in payload["state_dict"].items():
                self.assertIn(
                    tensor.dtype,
                    (torch.float64, torch.complex128),
                    f"Non-double tensor {tname} ({tensor.dtype}) in {fname}",
                )

            model = ManifoldTrustGate(variant=variant, **payload["arch"])
            model.load_state_dict(payload["state_dict"], strict=True)
            model.eval()
            self.assertEqual(model.real_parameters(), EXPECTED_CKPT_PARAMS)
            loaded_models[fname] = model

        # Deterministic forward parity & output invariants across all 12 checkpoints on SEED_P11_DEV links.
        dev_links = [
            simulate_offgrid_link(_dev_seed(1101 + r_idx), reg)
            for r_idx, reg in enumerate(OFFGRID_ALL_REGIMES)
        ]
        dev_bundles = [build_offgrid_branches(obs) for obs, _ in dev_links]

        for fname, model in loaded_models.items():
            for obs, bundle in zip((obs for obs, _ in dev_links), dev_bundles):
                res1 = model.gate(obs, bundle)
                res2 = model.gate(obs, bundle)
                self.assertEqual(float(np.max(np.abs(res1.w_ray - res2.w_ray))), 0.0, fname)
                self.assertEqual(float(np.max(np.abs(res1.noise_mult - res2.noise_mult))), 0.0, fname)
                self.assertEqual(float(np.max(np.abs(res1.h_mix - res2.h_mix))), 0.0, fname)
                self.assertEqual(float(np.max(np.abs(res1.noise_mix - res2.noise_mix))), 0.0, fname)
                self.assertEqual(float(np.max(np.abs(res1.gate_out.mix_llrs - res2.gate_out.mix_llrs))), 0.0, fname)
                self.assertEqual(res1.w_ray.dtype, np.float64)
                self.assertEqual(res1.h_mix.dtype, np.complex128)
                self.assertTrue(bool(np.all(np.isfinite(res1.w_ray))))
                self.assertTrue(bool(np.all(np.isfinite(res1.gate_out.mix_llrs))))
                self.assertTrue(bool(np.all(res1.gate_out.mix_probe_var >= obs.n0 - 1e-15)))

        # C* invariance/equivariance (< 1e-12) and PROBE_ROW poisoning (== 0.0) on loaded seed1001 models.
        obs0, _ = dev_links[1]  # offgrid_partial_shift
        bundle0 = dev_bundles[1]
        y_poison = obs0.y_b.copy()
        y_poison[PROBE_ROW] = 1e9 - 7e8j
        obs0_poison = replace(obs0, y_b=y_poison)
        alphas = (1e-12 * np.exp(1j * 0.7), 1e10 * np.exp(-1j * 2.1))

        for variant in VARIANTS:
            model = loaded_models[f"{variant}_seed1001.pt"]
            base = model.gate(obs0, bundle0)
            poisoned = model.gate(obs0_poison, bundle0)
            self.assertEqual(float(np.max(np.abs(poisoned.w_ray - base.w_ray))), 0.0)
            self.assertEqual(float(np.max(np.abs(poisoned.noise_mult - base.noise_mult))), 0.0)
            self.assertEqual(float(np.max(np.abs(poisoned.gate_out.mix_llrs - base.gate_out.mix_llrs))), 0.0)
            self.assertEqual(float(np.max(np.abs(poisoned.gate_out.mix_probe_pred - base.gate_out.mix_probe_pred))), 0.0)
            self.assertEqual(float(np.max(np.abs(poisoned.gate_out.mix_probe_var - base.gate_out.mix_probe_var))), 0.0)

            for alpha in alphas:
                a2 = abs(alpha) ** 2
                obs_s, bundle_s = _scale_obs_and_bundle(obs0, bundle0, alpha)
                sc = model.gate(obs_s, bundle_s)
                self.assertLess(float(np.max(np.abs(sc.w_ray - base.w_ray))), 1e-12)
                self.assertLess(float(np.max(np.abs(sc.noise_mult - base.noise_mult))), 1e-12)
                self.assertLess(float(np.max(np.abs(sc.gate_out.mix_llrs - base.gate_out.mix_llrs))), 1e-12)
                pp_ref = alpha * base.gate_out.mix_probe_pred
                pv_ref = a2 * base.gate_out.mix_probe_var
                self.assertLess(
                    float(np.max(np.abs(sc.gate_out.mix_probe_pred - pp_ref)) / max(float(np.max(np.abs(pp_ref))), 1e-300)),
                    1e-12,
                )
                self.assertLess(
                    float(np.max(np.abs(sc.gate_out.mix_probe_var - pv_ref)) / max(float(np.max(np.abs(pv_ref))), 1e-300)),
                    1e-12,
                )


if __name__ == "__main__":
    unittest.main()
