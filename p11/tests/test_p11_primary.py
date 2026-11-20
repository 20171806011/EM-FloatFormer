"""Comprehensive Primary Axis Unit Tests for P11 (Features F4 & F5).

Covers:
  1. Static AST & contract guardrails (`check_no_hardcoded_n0`, zero `inference_mode`,
     no formal-seed references, strict `n0` & 5-aux input validation).
  2. Mathematical & state identities (4096-pt phase quadrature, dense 32x32 covariance,
     large-kappa asymptotic Bessel fallback, `ResetBase` zero-virtual & ledger idempotence,
     epoch-30 sign-preserving projection, `delta=0` shift identity, `b=6` pre-onset
     identity, joint cross-ray probe quadratic & `v_p >= n0` floor).
  3. Whole-fold, `y_a`, target-row, and `PROBE_ROW=19` isolation firewalls + burst rejection.
  4. Full `C* = R_+ x U(1)` equivariance (`< 1e-12`) across no-unlock, shift-active,
     mode-split (`SEED_P11_E2E + 150`), and post-virtual regimes.
  5. 2-run in-process and fresh-subprocess bit-exact determinism & `n0` responsiveness.
  6. `AssimilationResult` contract, P10 compatibility adapter, and dev smoke validation.

Strictly uses `SEED_P11_DEV` and `SEED_P11_E2E` only (zero formal seeds, zero hardcoded N0).
"""
from __future__ import annotations

import ast
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
from scipy.special import logsumexp, softmax

from p8.coding import generator
from p8.tracker import State
from p8c.link import PILOTS as SC_PILOTS
from p9.channel import DELAYED_INDEX, simulate_link
from p9.decoder import is_valid_codeword
from p9.gating import build_prior_states, condition_sc
from p9.independent_ref import check_no_hardcoded_n0
from p11.contract import (
    ANCHOR,
    CORE_ROWS,
    DELAYS,
    FRAME_DURATION,
    N,
    POWERS,
    PRIMARY_SNR_DB,
    PROBE_ROW,
    SEED_P11_DEV,
    SEED_P11_E2E,
    T,
    AssimilationResult,
    GateOutput,
    enforce_determinism,
    is_historical_seed,
    n0_from_snr,
)
from p11.payload_rebuild import (
    ORIGIN,
    ShiftedBranchDynamics,
    b_only_canonical_gauge,
    coarse_payload_score,
    default_scale_quadrature,
    estimate_payload_posterior,
    fit_known_blocks_by_node,
    gl3_frequency_bank,
    parity_block_cov_score,
    project_joint_probe,
    project_to_tracker_state,
    refit_reset,
    sample_moments_b_stable_rows,
    stratified_mode_midpoints,
)
from p11.wavefront_pipeline import (
    AUX_FOLDS,
    FOLD_TARGETS,
    as_p10_compatible_dict,
    run_p11_primary,
    run_p11_with_trace,
    to_p10_assimilation_result,
    warmup_p11_numba,
)


P11_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = P11_DIR.parent
AUX1 = AUX_FOLDS[0]
AUX2 = AUX_FOLDS[1]


def setUpModule() -> None:
    enforce_determinism(require_env=False)
    warmup_p11_numba()


class TestP11PrimaryStaticAndContract(unittest.TestCase):
    def test_ast_no_hardcoded_n0_strict(self) -> None:
        targets = [
            P11_DIR / "__init__.py",
            P11_DIR / "contract.py",
            P11_DIR / "payload_rebuild.py",
            P11_DIR / "wavefront_pipeline.py",
            P11_DIR / "p11_dev.py",
            Path(__file__).resolve(),
        ]
        report = check_no_hardcoded_n0(targets, strict=True)
        self.assertTrue(report["clean"], report)
        self.assertEqual(report["violation_count"], 0)

    def test_ast_zero_inference_mode(self) -> None:
        targets = [
            P11_DIR / "__init__.py",
            P11_DIR / "contract.py",
            P11_DIR / "payload_rebuild.py",
            P11_DIR / "wavefront_pipeline.py",
            P11_DIR / "p11_dev.py",
            Path(__file__).resolve(),
        ]
        hits: list[str] = []
        for path in targets:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == "inference_mode":
                    hits.append(f"{path.name}:{node.lineno}")
                elif isinstance(node, ast.Name) and node.id == "inference_mode":
                    hits.append(f"{path.name}:{node.lineno}")
        self.assertEqual(hits, [])

    def test_no_formal_seed_references_in_primary_or_dev(self) -> None:
        forbidden = ("SEED_P11_" + "FORMAL_ID", "SEED_P11_" + "FORMAL_OOD")
        for fname in ("payload_rebuild.py", "wavefront_pipeline.py", "p11_dev.py"):
            src = (P11_DIR / fname).read_text(encoding="utf-8")
            for sym in forbidden:
                self.assertNotIn(sym, src, f"Forbidden formal seed symbol {sym} in {fname}")
        self.assertIsNone(is_historical_seed(SEED_P11_DEV))
        self.assertIsNone(is_historical_seed(SEED_P11_E2E))

    def test_input_validation_rejects_bool_timedelta_and_invalid_aux(self) -> None:
        obs, _ = simulate_link(SEED_P11_E2E + 1, PRIMARY_SNR_DB, "full_reset", "sparse")
        y10 = obs.y_b[ANCHOR]
        y_aux = obs.y_b[list(AUX1)]

        for bad_n0 in (
            True,
            False,
            np.bool_(True),
            np.timedelta64(1, "s"),
            np.datetime64("2026-01-01"),
            1.0 + 0.0j,
            0.0,
            -0.5,
            float("nan"),
            float("inf"),
        ):
            with self.subTest(bad_n0=repr(bad_n0)):
                with self.assertRaises((TypeError, ValueError)):
                    b_only_canonical_gauge(y10, bad_n0)
                with self.assertRaises((TypeError, ValueError)):
                    estimate_payload_posterior(y10, AUX1, y_aux, bad_n0)
                with self.assertRaises((TypeError, ValueError)):
                    run_p11_primary(replace(obs, n0=bad_n0))

        for bad_aux in ((), (2,), (2, 5, 8, 13), (2, 5, 8, 13, 13), (2, 5, 8, 13, ANCHOR), (2, 5, 8, 13, PROBE_ROW)):
            with self.subTest(bad_aux=bad_aux):
                with self.assertRaises(ValueError):
                    estimate_payload_posterior(obs, bad_aux, use_payload=True)

        # Pilot-only mode explicitly allows empty aux=() and ignores targets/y_a
        pilot_only = estimate_payload_posterior(obs, (), use_payload=False)
        self.assertEqual(len(pilot_only.node_ids), 5832)
        self.assertFalse(pilot_only.use_payload)


class TestP11PayloadAndShiftIdentities(unittest.TestCase):
    def test_pilot_likelihood_and_moments_vs_dense_phase_quadrature(self) -> None:
        obs, _ = simulate_link(SEED_P11_DEV + 11, PRIMARY_SNR_DB, "full_reset", "sparse")
        c_b, y10_c, n_c = b_only_canonical_gauge(obs.y_b[ANCHOR], obs.n0)
        bank = gl3_frequency_bank()
        sub_idx = np.array([0, 317, 2500, len(bank.nodes) - 1], dtype=np.int64)
        sub_nodes = bank.nodes[sub_idx]
        rho_test = 4.25
        fit = fit_known_blocks_by_node(
            sub_nodes,
            rows=(ANCHOR,),
            words=SC_PILOTS[ANCHOR : ANCHOR + 1],
            received=y10_c[None, :],
            noises=np.array([n_c], dtype=np.float64),
            scales=(np.array([rho_test]), np.array([0.0])),
        )

        # Independent 4096-point circular phase quadrature + dense 32x32 solve
        test_phases = np.sqrt(rho_test * POWERS[0]) * np.exp(2j * np.pi * np.arange(4096) / 4096.0)
        p_w = rho_test * POWERS[1:]
        for ix, params in enumerate(sub_nodes):
            d = np.stack(
                [bank.rot[p, ANCHOR] * np.roll(SC_PILOTS[ANCHOR], int(lag)) for p, lag in zip(params, DELAYS)],
                axis=1,
            )
            w_mat = d[:, 1:]
            k_cov = n_c * np.eye(N, dtype=np.complex128) + (w_mat * p_w[None, :]) @ w_mat.conj().T
            resid = y10_c[:, None] - d[:, :1] * test_phases[None, :]
            kr = np.linalg.solve(k_cov, resid)
            ll = -(resid.conj() * kr).sum(axis=0).real - np.linalg.slogdet(k_cov)[1] - N * np.log(np.pi)
            weights = softmax(ll)
            weak_m = (w_mat.conj().T @ kr) * p_w[:, None]
            weak_cov = np.linalg.inv(w_mat.conj().T @ w_mat / n_c + np.diag(1.0 / p_w))
            latent = np.vstack([test_phases, weak_m])
            ind_mean = latent @ weights
            dm = latent - ind_mean[:, None]
            ind_cov = (dm * weights[None, :]) @ dm.conj().T
            ind_cov[1:, 1:] += weak_cov
            ind_ll = float(logsumexp(ll) - np.log(len(ll)))

            self.assertLess(abs(fit.log_evidence[ix] - ind_ll), 1e-11)
            self.assertLess(np.max(np.abs(fit.mean[ix] - ind_mean)), 1e-11)
            self.assertLess(np.max(np.abs(fit.cov[ix] - ind_cov)), 1e-11)

        # Large-kappa asymptotic fallback check (very small noise -> kappa >> 1e7)
        fit_hi = fit_known_blocks_by_node(
            sub_nodes[:1],
            rows=(ANCHOR,),
            words=SC_PILOTS[ANCHOR : ANCHOR + 1],
            received=y10_c[None, :],
            noises=np.array([1e-12], dtype=np.float64),
            scales=(np.array([1.0e12]), np.array([0.0])),
        )
        self.assertTrue(np.isfinite(fit_hi.log_evidence).all())
        self.assertGreater(float(fit_hi.cov[0, 0, 0].real), 0.0)

    def test_exact_and_coarse_payload_cov_vs_dense_32x32(self) -> None:
        obs, _ = simulate_link(SEED_P11_DEV + 12, PRIMARY_SNR_DB, "full_reset", "sparse")
        c_b, y10_c, n_c = b_only_canonical_gauge(obs.y_b[ANCHOR], obs.n0)
        y_aux_c = obs.y_b[list(AUX1)] / c_b
        bank = gl3_frequency_bank()
        sub_idx = np.array([0, 412, 2900, 5800], dtype=np.int64)
        sub_nodes = bank.nodes[sub_idx]
        fit = fit_known_blocks_by_node(
            sub_nodes,
            rows=(ANCHOR,),
            words=SC_PILOTS[ANCHOR : ANCHOR + 1],
            received=y10_c[None, :],
            noises=np.array([n_c], dtype=np.float64),
            scales=default_scale_quadrature(),
        )
        exact = parity_block_cov_score(y_aux_c, AUX1, sub_nodes, fit.second, n_c=n_c)

        # Full bank for coarse check
        fit_all = fit_known_blocks_by_node(
            bank.nodes,
            rows=(ANCHOR,),
            words=SC_PILOTS[ANCHOR : ANCHOR + 1],
            received=y10_c[None, :],
            noises=np.array([n_c], dtype=np.float64),
            scales=default_scale_quadrature(),
        )
        coarse = coarse_payload_score(y_aux_c, AUX1, fit_all.second, n_c=n_c)

        f_dft = np.fft.fft(np.eye(N, dtype=np.complex128), norm="ortho", axis=0)
        for ix, k_global in enumerate(sub_idx):
            params = bank.nodes[k_global]
            exact_ref = 0.0
            coarse_ref = 0.0
            for r_pos, row in enumerate(AUX1):
                h_mats = [
                    bank.rot[p, row, :, None] * np.roll(np.eye(N, dtype=np.complex128), int(lag), axis=0)
                    for p, lag in zip(params, DELAYS)
                ]
                cov = n_c * np.eye(N, dtype=np.complex128) + sum(
                    fit.second[ix, a, b] * (h_a @ h_b.conj().T)
                    for a, h_a in enumerate(h_mats)
                    for b, h_b in enumerate(h_mats)
                )
                z = y_aux_c[r_pos]
                exact_ref += float(
                    -N * np.log(np.pi) - np.linalg.slogdet(cov)[1] - np.vdot(z, np.linalg.solve(cov, z)).real
                )
                diag_f = np.diag(f_dft @ cov @ f_dft.conj().T).real
                coarse_ref += float(-N * np.log(np.pi) - np.sum(np.log(diag_f) + (np.abs(f_dft @ z) ** 2) / diag_f))
            self.assertLess(abs(exact[ix] - exact_ref), 1e-11)
            self.assertLess(abs(coarse[k_global] - coarse_ref), 1e-11)

        # Stratified mode compression mass conservation
        idx_s, rep_mass, _ = stratified_mode_midpoints(bank.log_gl_prior + 0.75 * coarse, bank.strata_4, budget=160)
        self.assertLessEqual(len(idx_s), 160)
        self.assertAlmostEqual(float(np.sum(rep_mass)), 1.0, places=12)

    def test_reset_base_zero_virtual_and_ledger_idempotence(self) -> None:
        obs, truth = simulate_link(SEED_P11_DEV + 13, PRIMARY_SNR_DB, "full_reset", "sparse")
        base = estimate_payload_posterior(obs, AUX1, use_payload=True)
        j0 = refit_reset(base, obs.y_b[ANCHOR], {})
        np.testing.assert_allclose(j0.logw, base.logw0, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(j0.mean_c, base.anchor_mean, rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(j0.cov_c, base.anchor_cov, rtol=1e-13, atol=1e-13)

        # Idempotent replay regardless of ledger insertion order
        x9 = truth.b_symbols[17]
        x11 = truth.b_symbols[CORE_ROWS.index(11)]
        ledger_a = {9: (x9, obs.y_b[9], obs.n0 * 1.5), 11: (x11, obs.y_b[11], obs.n0 * 1.8)}
        ledger_b = {11: (x11, obs.y_b[11], obs.n0 * 1.8), 9: (x9, obs.y_b[9], obs.n0 * 1.5)}
        ja = refit_reset(base, obs.y_b[ANCHOR], ledger_a)
        jb = refit_reset(base, obs.y_b[ANCHOR], ledger_b)
        np.testing.assert_array_equal(ja.logw, jb.logw)
        np.testing.assert_array_equal(ja.mean_c, jb.mean_c)
        np.testing.assert_array_equal(ja.cov_c, jb.cov_c)

    def test_epoch30_tracker_projection_moment_preservation(self) -> None:
        obs, _ = simulate_link(SEED_P11_DEV + 14, PRIMARY_SNR_DB, "full_reset", "sparse")
        base = estimate_payload_posterior(obs, AUX1, use_payload=True)
        st = base.state_clr_proj
        self.assertEqual(st.epoch, float(FRAME_DURATION))
        self.assertEqual(st.mean.shape, (3, 128))
        self.assertTrue(np.all(np.isfinite(st.mean)))
        self.assertTrue(np.all(st.variance > 0.0))

        # Verify exact epoch-30 marginal first and second moment preservation vs joint GL3
        bank = gl3_frequency_bank()
        dt = float(FRAME_DURATION) - ORIGIN
        d_tr = np.exp(2j * np.pi * ((bank.start_freq + bank.slope * ORIGIN) * dt + 0.5 * bank.slope * (dt ** 2)))
        w_joint = np.exp(base.logw0)
        g = complex(base.gauge)
        g2 = float(abs(g) ** 2)
        for ray in range(3):
            m_joint_e = base.anchor_mean[:, ray] * g * d_tr[base.node_ids[:, ray]]
            v_joint_e = base.anchor_cov[:, ray, ray].real * g2
            e1_joint = complex(np.sum(w_joint * m_joint_e))
            e2_joint = float(np.sum(w_joint * (v_joint_e + np.abs(m_joint_e) ** 2)))

            w_proj = np.where(np.isfinite(st.log_weight[ray]), np.exp(st.log_weight[ray]), 0.0)
            e1_proj = complex(np.sum(w_proj * st.mean[ray]))
            e2_proj = float(np.sum(w_proj * (st.variance[ray] + np.abs(st.mean[ray]) ** 2)))

            self.assertLess(abs(e1_proj - e1_joint), 1e-12)
            self.assertLess(abs(e2_proj - e2_joint), 1e-12)

    def test_shift_delta_zero_reproduces_continuous_dynamics(self) -> None:
        obs, _ = simulate_link(SEED_P11_DEV + 15, PRIMARY_SNR_DB, "continuous", "sparse")
        state_inh, _ = build_prior_states(obs)
        ref_post = condition_sc(state_inh, obs.y_b, (ANCHOR,), obs.n0)
        ref_m, ref_v = sample_moments_b_stable_rows(ref_post)

        dyn = ShiftedBranchDynamics(state_inh, changed_ray=1, delta_freq=0.0, onset_row=0)
        p_post = dyn.refit((ANCHOR,), SC_PILOTS[ANCHOR : ANCHOR + 1], obs.y_b[ANCHOR : ANCHOR + 1], np.array([obs.n0]))
        p_m, p_v = dyn.predict(p_post)
        np.testing.assert_allclose(p_m, ref_m, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(p_v, ref_v, rtol=1e-12, atol=1e-12)

    def test_midframe_shift_pre_onset_identity(self) -> None:
        obs, _ = simulate_link(SEED_P11_DEV + 16, PRIMARY_SNR_DB, "ood_midframe_shift", "sparse")
        state_inh, _ = build_prior_states(obs)
        dyn = ShiftedBranchDynamics(state_inh, changed_ray=2, delta_freq=0.055, onset_row=6)
        rot = dyn.shift_rotation()
        np.testing.assert_array_equal(rot[:6], np.ones((6, N), dtype=np.complex128))
        self.assertGreater(float(np.max(np.abs(rot[6:] - 1.0))), 0.1)

    def test_joint_probe_quadratic_nonzero_cross_cov_and_floor(self) -> None:
        obs, _ = simulate_link(SEED_P11_DEV + 17, PRIMARY_SNR_DB, "full_reset", "sparse")
        base = estimate_payload_posterior(obs, AUX1, use_payload=True)
        jstate = base.to_joint_state()
        # Confirm nonzero off-diagonal cross-ray covariance exists
        self.assertGreater(float(np.max(np.abs(jstate.cov_c[:, 0, 1]))), 1e-6)

        p_pred, p_var = project_joint_probe(jstate, probe_row=PROBE_ROW)
        self.assertTrue(np.all(p_var >= obs.n0 - 1e-12))

        # Independent dense verification of v_p = n0 + sum_j w_j [x^T Sigma_j x* + |p_j - p|^2]
        bank = gl3_frequency_bank()
        x = SC_PILOTS[PROBE_ROW, DELAYED_INDEX]  # [3, N]
        w = np.exp(jstate.logw)
        g = complex(jstate.gauge)
        g2 = float(abs(g) ** 2)
        for n in range(N):
            xn = x[:, n]
            pj_list = []
            sig_list = []
            for j in range(len(jstate.node_ids)):
                d_mat = np.diag(bank.rot[jstate.node_ids[j], PROBE_ROW, n])
                m_j = d_mat @ (jstate.mean_c[j] * g)
                s_j = d_mat @ (jstate.cov_c[j] * g2) @ d_mat.conj().T
                pj_list.append(complex(xn @ m_j))
                sig_list.append(complex(xn @ s_j @ xn.conj()).real)
            p_exp = sum(w[j] * pj_list[j] for j in range(len(w)))
            v_exp = obs.n0 + sum(w[j] * (sig_list[j] + abs(pj_list[j] - p_exp) ** 2) for j in range(len(w)))
            self.assertLess(abs(p_pred[n] - p_exp), 1e-12)
            self.assertLess(abs(p_var[n] - v_exp), 1e-12)

        # Also check assembled pipeline probe floor across regimes (version 0 and >= 1)
        res = run_p11_primary(obs)
        self.assertTrue(np.all(res.gate_out.mix_probe_var >= obs.n0 - 1e-12))


class TestP11FirewallsAndIsolation(unittest.TestCase):
    def test_payload_rebuild_isolated_from_ya_targets_valid_and_probe(self) -> None:
        obs, _ = simulate_link(SEED_P11_DEV + 21, PRIMARY_SNR_DB, "full_reset", "sparse")
        y_b_poison = obs.y_b.copy()
        for r in range(T):
            if r not in (ANCHOR, *AUX1):
                y_b_poison[r] = 500.0 - 400.0j if r != PROBE_ROW else complex(1e6, -1e6)
        obs_p = replace(obs, y_a=np.full_like(obs.y_a, 50.0 - 150.0j), y_b=y_b_poison)

        for use_p in (True, False):
            r_clean = estimate_payload_posterior(obs, AUX1, use_payload=use_p)
            r_pois = estimate_payload_posterior(obs_p, AUX1, use_payload=use_p)
            np.testing.assert_array_equal(r_clean.node_ids, r_pois.node_ids)
            np.testing.assert_array_equal(r_clean.logw0, r_pois.logw0)
            self.assertEqual(r_clean.class_log_evidence, r_pois.class_log_evidence)
            np.testing.assert_array_equal(r_clean.mean, r_pois.mean)
            np.testing.assert_array_equal(r_clean.var, r_pois.var)
            np.testing.assert_array_equal(r_clean.probe_pred, r_pois.probe_pred)
            np.testing.assert_array_equal(r_clean.probe_var, r_pois.probe_var)

        # All-zero anchor edge case
        z_res = estimate_payload_posterior(np.zeros(N, dtype=np.complex128), AUX1, obs.y_b[list(AUX1)], obs.n0)
        self.assertTrue(np.all(np.isfinite(z_res.mean)))
        self.assertTrue(np.all(z_res.probe_var >= obs.n0 - 1e-12))

    def test_whole_fold_target_poisoning_firewall(self) -> None:
        for reg in ("continuous", "partial_shift", "full_reset"):
            obs, _ = simulate_link(SEED_P11_DEV + 22, PRIMARY_SNR_DB, reg, "sparse")
            res_base, tr_base = run_p11_with_trace(obs)
            for f_idx, targets in enumerate(FOLD_TARGETS):
                y_p = obs.y_b.copy()
                y_p[list(targets)] = 2.125 - 0.75j
                _, tr_p = run_p11_with_trace(replace(obs, y_b=y_p))

                fb = tr_base["folds"][f_idx]
                fp = tr_p["folds"][f_idx]
                self.assertEqual(fb["branch"], fp["branch"])
                self.assertEqual(fb["t_f"], fp["t_f"])
                self.assertEqual(fb["b_c"], fp["b_c"])
                self.assertEqual(fb["b_s"], fp["b_s"])
                self.assertEqual(fb["b_r"], fp["b_r"])
                np.testing.assert_array_equal(fb["probe_pred"], fp["probe_pred"])
                np.testing.assert_array_equal(fb["probe_var"], fp["probe_var"])

                lb = [rec for rec in tr_base["ledger"] if rec["fold"] == f_idx]
                lp = [rec for rec in tr_p["ledger"] if rec["fold"] == f_idx]
                self.assertEqual(len(lb), len(lp))
                for ra, rb in zip(lb, lp):
                    self.assertEqual(ra["row"], rb["row"])
                    self.assertEqual(ra["alpha"], rb["alpha"])
                    np.testing.assert_array_equal(ra["injected_info"], rb["injected_info"])

    def test_wave1_fold0_excludes_row11_and_no_cross_fold_cert_leak(self) -> None:
        obs, _ = simulate_link(SEED_P11_DEV + 23, PRIMARY_SNR_DB, "full_reset", "sparse")
        _, tr_base = run_p11_with_trace(obs)
        f0_rows = {rec["row"] for rec in tr_base["ledger"] if rec["fold"] == 0}
        self.assertNotIn(11, f0_rows)
        self.assertTrue(f0_rows.issubset({0, 9}))

        # Poison row 11 -> Fold 0 pre-target state, ledger, and Fold 0 probe must be bitwise invariant
        y_p11 = obs.y_b.copy()
        y_p11[11] = 80.0 - 65.0j
        _, tr_p11 = run_p11_with_trace(replace(obs, y_b=y_p11))
        np.testing.assert_array_equal(tr_base["folds"][0]["probe_pred"], tr_p11["folds"][0]["probe_pred"])
        np.testing.assert_array_equal(tr_base["folds"][0]["probe_var"], tr_p11["folds"][0]["probe_var"])
        lb0 = [rec for rec in tr_base["ledger"] if rec["fold"] == 0]
        lp0 = [rec for rec in tr_p11["ledger"] if rec["fold"] == 0]
        self.assertEqual([r["row"] for r in lb0], [r["row"] for r in lp0])

    def test_probe_row_19_exact_zero_leakage_and_exclusion(self) -> None:
        for idx, reg in enumerate(("continuous", "partial_shift", "full_reset")):
            obs, _ = simulate_link(SEED_P11_DEV + 24 + idx, PRIMARY_SNR_DB, reg, "sparse")
            base, _ = run_p11_with_trace(obs)
            self.assertNotIn(PROBE_ROW, base.assimilated_rows)
            self.assertNotIn(PROBE_ROW, base.decoded_info)
            self.assertNotIn(PROBE_ROW, base.row_weights)
            self.assertNotIn(PROBE_ROW, base.row_margins)

            for sentinel in (1e6 - 1e6j, -5e5 + 3e5j):
                y_p = obs.y_b.copy()
                y_p[PROBE_ROW] = sentinel
                res_p, tr_p = run_p11_with_trace(replace(obs, y_b=y_p))
                self.assertEqual(base.assimilated_rows, res_p.assimilated_rows)
                self.assertEqual(base.row_weights, res_p.row_weights)
                self.assertEqual(base.row_margins, res_p.row_margins)
                for r in base.decoded_info:
                    np.testing.assert_array_equal(base.decoded_info[r], res_p.decoded_info[r])
                np.testing.assert_array_equal(base.gate_out.mix_probe_pred, res_p.gate_out.mix_probe_pred)
                np.testing.assert_array_equal(base.gate_out.mix_probe_var, res_p.gate_out.mix_probe_var)
                np.testing.assert_array_equal(base.gate_out.mix_llrs, res_p.gate_out.mix_llrs)
                np.testing.assert_array_equal(base.gate_out.mix_mean_ray, res_p.gate_out.mix_mean_ray)
                np.testing.assert_array_equal(base.gate_out.mix_var_ray, res_p.gate_out.mix_var_ray)
                self.assertEqual(base.gate_out.probe_weight, res_p.gate_out.probe_weight)
                for rec in tr_p["ledger"]:
                    self.assertNotEqual(rec["row"], PROBE_ROW)

        # Direct API guards against PROBE_ROW
        with self.assertRaises(ValueError):
            estimate_payload_posterior(obs, (2, 5, 8, 13, PROBE_ROW), use_payload=True)
        base_r = estimate_payload_posterior(obs, AUX1, use_payload=True)
        with self.assertRaises(ValueError):
            refit_reset(base_r, obs.y_b[ANCHOR], {PROBE_ROW: (SC_PILOTS[PROBE_ROW], np.zeros(N, dtype=np.complex128), obs.n0)})

    def test_assimilated_rows_valid_and_100pct_parity_certified(self) -> None:
        obs, _ = simulate_link(SEED_P11_DEV + 28, PRIMARY_SNR_DB, "full_reset", "sparse")
        res, tr = run_p11_with_trace(obs)
        self.assertEqual(len(res.assimilated_rows), len(set(res.assimilated_rows)))
        for r in res.assimilated_rows:
            self.assertGreaterEqual(r, 0)
            self.assertLess(r, T)
            self.assertNotIn(r, obs.training_rows)
            self.assertNotEqual(r, PROBE_ROW)
        g_mat = generator()
        for r, bits in res.decoded_info.items():
            self.assertEqual(bits.shape, (64,))
            self.assertEqual(bits.dtype, np.uint8)
            self.assertTrue(is_valid_codeword((bits @ g_mat) % 2))
        for rec in tr["ledger"]:
            self.assertTrue(is_valid_codeword(rec["injected_codeword"]))

    def test_burst_poisoning_resistance(self) -> None:
        obs, _ = simulate_link(SEED_P11_DEV + 29, PRIMARY_SNR_DB, "continuous", "sparse")
        y_burst = obs.y_b.copy()
        y_burst[9] = 80.0 - 65.0j
        res, tr = run_p11_with_trace(replace(obs, y_b=y_burst))
        self.assertNotIn(9, res.assimilated_rows)
        self.assertEqual(res.row_weights[9], 0.0)
        self.assertTrue(np.all(np.isfinite(res.gate_out.mix_probe_pred)))
        self.assertTrue(np.all(np.isfinite(res.gate_out.mix_llrs)))
        for rec in tr["ledger"]:
            self.assertNotEqual(rec["row"], 9)


class TestP11CStarEquivariance(unittest.TestCase):
    def _assert_cstar_case(self, obs, scales) -> None:
        base = run_p11_primary(obs)
        max_m_base = max(1.0, float(np.max(np.abs(base.gate_out.mix_mean_ray))))
        max_v_base = max(1.0, float(np.max(np.abs(base.gate_out.mix_var_ray))))
        max_p_base = max(1.0, float(np.max(np.abs(base.gate_out.mix_probe_pred))))
        max_pv_base = max(1.0, float(np.max(np.abs(base.gate_out.mix_probe_var))))

        for alpha in scales:
            a2 = float(abs(alpha) ** 2)
            obs_s = replace(obs, y_a=obs.y_a * alpha, y_b=obs.y_b * alpha, n0=obs.n0 * a2)
            sc = run_p11_primary(obs_s)

            self.assertEqual(sc.assimilated_rows, base.assimilated_rows)
            self.assertEqual(sc.n_rounds, base.n_rounds)
            for r, bits in base.decoded_info.items():
                np.testing.assert_array_equal(sc.decoded_info[r], bits)
            for r in base.row_weights:
                self.assertLess(abs(sc.row_weights[r] - base.row_weights[r]), 1e-12)
            for r in base.row_margins:
                rel_m = abs(sc.row_margins[r] - base.row_margins[r]) / max(1.0, abs(base.row_margins[r]))
                self.assertLess(rel_m, 1e-12, f"row {r} margin drift {rel_m}")

            self.assertLess(abs(sc.gate_out.probe_weight - base.gate_out.probe_weight), 1e-12)
            self.assertLess(float(np.max(np.abs(sc.gate_out.history_weight - base.gate_out.history_weight))), 1e-12)
            self.assertLess(float(np.max(np.abs(sc.gate_out.mix_llrs - base.gate_out.mix_llrs))), 1e-12)
            self.assertLess(float(np.max(np.abs(sc.gate_out.mix_probe_pred / alpha - base.gate_out.mix_probe_pred))) / max_p_base, 1e-12)
            self.assertLess(float(np.max(np.abs(sc.gate_out.mix_probe_var / a2 - base.gate_out.mix_probe_var))) / max_pv_base, 1e-12)
            self.assertLess(float(np.max(np.abs(sc.gate_out.mix_mean_ray / alpha - base.gate_out.mix_mean_ray))) / max_m_base, 1e-12)
            self.assertLess(float(np.max(np.abs(sc.gate_out.mix_var_ray / a2 - base.gate_out.mix_var_ray))) / max_v_base, 1e-12)

    def test_cstar_equivariance_four_regimes_and_e2e_regression(self) -> None:
        scales = (
            2.75 * np.exp(1.17j),
            0.37 * np.exp(-2.09j),
            1e-4 * np.exp(-2.718j),
            2.5e4 * np.exp(0.42j),
        )
        # 1. No-unlock (very low SNR link)
        obs_nounlock, _ = simulate_link(SEED_P11_DEV + 41, -3.0, "continuous", "sparse")
        self._assert_cstar_case(obs_nounlock, scales[:2])

        # 2. Shift-active (partial_shift)
        obs_shift, _ = simulate_link(SEED_P11_DEV + 42, PRIMARY_SNR_DB, "partial_shift", "sparse")
        self._assert_cstar_case(obs_shift, scales[:2])

        # 3. Mode-split + E2E regression (SEED_P11_E2E + 150 full_reset)
        obs_reset, _ = simulate_link(SEED_P11_E2E + 150, PRIMARY_SNR_DB, "full_reset", "sparse")
        self._assert_cstar_case(obs_reset, scales)

        # 4. Post-virtual (continuous with >= 2 assimilated rows)
        obs_cont, _ = simulate_link(SEED_P11_DEV + 43, PRIMARY_SNR_DB, "continuous", "sparse")
        res_cont = run_p11_primary(obs_cont)
        self.assertGreaterEqual(len(res_cont.assimilated_rows), 2)
        self._assert_cstar_case(obs_cont, scales[2:])


class TestP11Determinism(unittest.TestCase):
    def test_two_run_bit_exactness_in_process_and_subprocess(self) -> None:
        obs, _ = simulate_link(SEED_P11_DEV + 51, PRIMARY_SNR_DB, "full_reset", "sparse")
        r1 = run_p11_primary(obs)
        r2 = run_p11_primary(obs)
        self.assertEqual(r1.assimilated_rows, r2.assimilated_rows)
        self.assertEqual(r1.n_rounds, r2.n_rounds)
        self.assertEqual(r1.row_weights, r2.row_weights)
        self.assertEqual(r1.row_margins, r2.row_margins)
        for k in r1.decoded_info:
            np.testing.assert_array_equal(r1.decoded_info[k], r2.decoded_info[k])
        np.testing.assert_array_equal(r1.gate_out.mix_probe_pred, r2.gate_out.mix_probe_pred)
        np.testing.assert_array_equal(r1.gate_out.mix_probe_var, r2.gate_out.mix_probe_var)
        np.testing.assert_array_equal(r1.gate_out.mix_llrs, r2.gate_out.mix_llrs)
        np.testing.assert_array_equal(r1.gate_out.mix_mean_ray, r2.gate_out.mix_mean_ray)
        np.testing.assert_array_equal(r1.gate_out.mix_var_ray, r2.gate_out.mix_var_ray)

        # Fresh subprocess determinism check
        script = (
            "import json, numpy as np; "
            "from p9.channel import simulate_link; "
            "from p11.contract import PRIMARY_SNR_DB, SEED_P11_DEV, enforce_determinism; "
            "from p11.wavefront_pipeline import run_p11_primary; "
            "enforce_determinism(require_env=True); "
            "obs, _ = simulate_link(SEED_P11_DEV + 52, PRIMARY_SNR_DB, 'partial_shift', 'sparse'); "
            "r = run_p11_primary(obs); "
            "print(json.dumps({"
            "'assim': list(r.assimilated_rows), "
            "'p_re': r.gate_out.mix_probe_pred.real.tolist(), "
            "'p_im': r.gate_out.mix_probe_pred.imag.tolist(), "
            "'pv': r.gate_out.mix_probe_var.tolist(), "
            "'bits': {str(k): v.tolist() for k, v in sorted(r.decoded_info.items())}"
            "}))"
        )
        env = os.environ.copy()
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        out_a = subprocess.check_output([sys.executable, "-B", "-c", script], cwd=str(REPO_ROOT), env=env, text=True)
        out_b = subprocess.check_output([sys.executable, "-B", "-c", script], cwd=str(REPO_ROOT), env=env, text=True)
        self.assertEqual(json.loads(out_a), json.loads(out_b))

    def test_n0_scaling_responsiveness(self) -> None:
        obs_lo, _ = simulate_link(SEED_P11_DEV + 53, 6.0, "full_reset", "sparse")
        obs_hi, _ = simulate_link(SEED_P11_DEV + 53, 15.0, "full_reset", "sparse")
        r_lo = run_p11_primary(obs_lo)
        r_hi = run_p11_primary(obs_hi)
        self.assertGreater(float(np.mean(r_lo.gate_out.mix_probe_var)), float(np.mean(r_hi.gate_out.mix_probe_var)))


class TestP11ContractAndAdapter(unittest.TestCase):
    def test_assimilation_result_contract_and_adapter(self) -> None:
        obs, _ = simulate_link(SEED_P11_DEV + 61, PRIMARY_SNR_DB, "continuous", "sparse")
        res = run_p11_primary(obs)
        self.assertIsInstance(res, AssimilationResult)
        self.assertIsInstance(res.gate_out, GateOutput)
        np.testing.assert_array_equal(res.gate_out.mix_llrs[ANCHOR], np.zeros((N, 4), dtype=np.float64))
        np.testing.assert_array_equal(res.gate_out.mix_llrs[PROBE_ROW], np.zeros((N, 4), dtype=np.float64))
        for w in res.row_weights.values():
            self.assertGreaterEqual(w, 0.0)
            self.assertLessEqual(w, 1.0)
        for m in res.row_margins.values():
            self.assertTrue(np.isfinite(m) and m >= 0.0)

        compat = as_p10_compatible_dict(res)
        self.assertEqual(
            set(compat.keys()),
            {"gate_out", "decoded_info", "assimilated_rows", "row_weights", "row_margins", "n_rounds", "elapsed_ms"},
        )
        roundtrip = to_p10_assimilation_result(res, AssimilationResult)
        self.assertIsInstance(roundtrip, AssimilationResult)
        self.assertEqual(roundtrip.assimilated_rows, res.assimilated_rows)

    def test_dev_smoke_common_phase_and_probe_improvement(self) -> None:
        obs, truth = simulate_link(SEED_P11_DEV + 62, PRIMARY_SNR_DB, "full_reset", "sparse")
        pilot_only = estimate_payload_posterior(obs, (), use_payload=False)
        res, _ = run_p11_with_trace(obs)
        self.assertGreater(len(res.assimilated_rows), 0)
        sse_pilot = float(np.sum(np.abs(truth.y_b_probe_clean - pilot_only.probe_pred) ** 2))
        sse_p11 = float(np.sum(np.abs(truth.y_b_probe_clean - res.gate_out.mix_probe_pred) ** 2))
        self.assertLess(sse_p11, sse_pilot)


if __name__ == "__main__":
    unittest.main()
