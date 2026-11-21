"""Regressions for B-anchor evidence recycling in the P12 off-grid receiver.

These tests isolate a fixed support without fitting any formal-result hyperparameters.
The analytic reference integrates a one-ray complex Gaussian gain with full covariance
matrices, independently of the receiver's matched-filter/Jacobi implementation.
"""
from __future__ import annotations

from dataclasses import replace
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp

from p8c.link import PILOTS as SC_PILOTS
from p11.contract import N, T, PROBE_ROW, OffGridObservation, fractional_delay_matrix
from p11.offgrid_channel import OffGridAtomEstimates, _TIME_A_FULL, _TIME_B_FULL
from p12.contract import OFDM_PILOT_COMB
from p12 import offgrid_receiver as receiver


def _one_ray_fixture() -> tuple[OffGridObservation, OffGridAtomEstimates]:
    """A controlled two-mode Frame-A estimate and one physical B-anchor observation."""
    anchor = 10
    tau = np.array([1.2], dtype=np.float64)
    nu = np.array([[0.045, -0.060]], dtype=np.float64)
    chirp = np.array([[0.0004, -0.0002]], dtype=np.float64)
    mu = np.array([[0.7 + 0.2j, -0.4 + 0.3j]], dtype=np.complex128)
    var = np.array([[0.18, 0.27]], dtype=np.float64)
    atoms = OffGridAtomEstimates(
        tau_hat=tau,
        nu_hat=nu[:, 0].copy(),
        g_a_mid=mu[:, 0].copy(),
        v_a_mid=var[:, 0].copy(),
        atom_snr_a=np.array([1.125]),
        p_atom_a=np.array([0.9]),
        p_tot_a=0.9,
        nu_bank_a=nu,
        chirp_bank_a=chirp,
        log_w_bank_a=np.log(np.array([[0.65, 0.35]])),
        mu_bank_a=mu,
        var_bank_a=var,
    )
    d_mat = fractional_delay_matrix(float(tau[0]), n=N)
    dt_anchor = _TIME_B_FULL[anchor] - float(_TIME_B_FULL[anchor].mean())
    y_b = np.zeros((T, N), dtype=np.complex128)
    y_b[anchor] = (
        (0.65 - 0.15j) * np.exp(2j * np.pi * 0.065 * dt_anchor) * (d_mat @ SC_PILOTS[anchor])
        + 0.05 * np.exp(0.37j * np.arange(N))
    )
    y_b[PROBE_ROW] = np.nan + 1j * np.nan
    obs = OffGridObservation(
        y_a=np.zeros((T, N), dtype=np.complex128),
        y_b=y_b,
        training_rows=(anchor,),
        probe_row=PROBE_ROW,
        n0=0.8,
        pilot_tx_b=SC_PILOTS[[anchor]].copy(),
        pilot_tx_a=np.ones((4, N // 2), dtype=np.complex128),
    )
    return obs, atoms


def _run_fixed_support(obs, atoms, *, piecewise=False):
    # The empirical delay selector compares identical supports and therefore cannot reset.
    # Keeping the selector enabled avoids the deliberate legacy B-refit-off control path.
    with (
        patch.object(receiver, "discover_frame_a_atoms", return_value=atoms),
        patch.object(receiver, "discover_b_only_delays", return_value=(
            atoms.tau_hat.copy(), atoms.p_atom_a / obs.n0, atoms.p_tot_a / obs.n0,
        )),
    ):
        return receiver.run_p12_offgrid_physics(
            obs, k_hat=1, use_aux_cov=False, use_wavefront=False,
            use_piecewise_midframe=piecewise, use_delay_rediscovery=True,
            use_burst_guard=False,
        )


def _independent_anchor_posterior(obs, atoms):
    """Full-covariance Gaussian integration for the two enabled physical families."""
    anchor = obs.training_rows[0]
    ta = float(_TIME_A_FULL[list(OFDM_PILOT_COMB)].mean())
    tb = float(_TIME_B_FULL[anchor].mean())
    dt = _TIME_B_FULL - tb
    da = tb - ta
    nu_a = atoms.nu_bank_a[0]
    ch_a = atoms.chirp_bank_a[0]
    rot_inh = np.exp(2j * np.pi * (
        (nu_a + ch_a * da)[:, None, None] * dt[None]
        + 0.5 * ch_a[:, None, None] * dt[None] ** 2
    ))
    rot_clr = np.exp(2j * np.pi * receiver._NU_CLR_EXT_BANK[:, None, None] * dt[None])
    rot = np.concatenate((rot_inh, rot_clr))
    mu_prior = np.concatenate((
        atoms.mu_bank_a[0] * np.exp(2j * np.pi * (nu_a * da + 0.5 * ch_a * da ** 2)),
        np.zeros(len(receiver._NU_CLR_EXT_BANK), dtype=np.complex128),
    ))
    var_prior = np.concatenate((atoms.var_bank_a[0], np.full(len(rot_clr), 0.9)))
    clear_mass = np.array([1.0] * 8 + [0.35, 0.35] + [1.0] * 8 + [0.35, 0.35])
    clear_mass /= clear_mass.sum()
    # Equal family masses are the prospective design choice; mode counts do not change it.
    prior_mass = np.concatenate((0.5 * np.exp(atoms.log_w_bank_a[0]), 0.5 * clear_mass))

    d_mat = fractional_delay_matrix(float(atoms.tau_hat[0]), n=N)
    pilot_delayed = d_mat @ obs.pilot_tx_b[0]
    ll = []
    posterior_mu = []
    posterior_var = []
    for h in range(len(rot)):
        design = rot[h, anchor] * pilot_delayed
        cov = obs.n0 * np.eye(N) + var_prior[h] * np.outer(design, design.conj())
        residual = obs.y_b[anchor] - mu_prior[h] * design
        solved_residual = np.linalg.solve(cov, residual)
        ll.append(float(-np.linalg.slogdet(cov)[1] - np.vdot(residual, solved_residual).real))
        posterior_mu.append(mu_prior[h] + var_prior[h] * np.vdot(design, solved_residual))
        posterior_var.append(float(
            var_prior[h] - var_prior[h] ** 2 * np.vdot(design, np.linalg.solve(cov, design)).real
        ))
    ll = np.asarray(ll)
    log_mass = np.log(prior_mass) + ll
    mass = np.exp(log_mass - logsumexp(log_mass))
    posterior_mu = np.asarray(posterior_mu)
    posterior_var = np.asarray(posterior_var)
    mean = np.einsum("h,htn->tn", mass * posterior_mu, rot)
    second = float(np.sum(mass * (np.abs(posterior_mu) ** 2 + posterior_var)))
    variance = second - np.abs(mean) ** 2

    # A deliberately wrong doubled evidence mixture must be distinguishable in this fixture.
    doubled = np.log(prior_mass) + 2.0 * ll
    doubled_mass = np.exp(doubled - logsumexp(doubled))
    doubled_mean = np.einsum("h,htn->tn", doubled_mass * posterior_mu, rot)
    return mean, variance, d_mat, doubled_mean


class TestAnchorPriorConsistency(unittest.TestCase):
    def test_cached_anchor_gate_weights_do_not_affect_full_receiver(self):
        obs, atoms = _one_ray_fixture()
        d_mat = fractional_delay_matrix(float(atoms.tau_hat[0]), n=N)
        for piecewise in (False, True):
            outputs = []
            for cached_gate in (0.02, 0.98):
                cached = SimpleNamespace(
                    tau_hat=atoms.tau_hat.copy(), d_mats=d_mat[None],
                    u_probe=(d_mat @ SC_PILOTS[PROBE_ROW])[None], k_hat=1,
                    w_ray_classical=np.array([cached_gate]),
                )
                with patch.object(receiver, "build_offgrid_branches", return_value=cached) as builder:
                    result = _run_fixed_support(obs, atoms, piecewise=piecewise)
                    outputs.append(result)
                    # The old implementation built this B-conditioned bundle and used its gate.
                    builder.assert_not_called()
            with self.subTest(piecewise=piecewise):
                np.testing.assert_array_equal(outputs[0].h_mix, outputs[1].h_mix)
                np.testing.assert_array_equal(outputs[0].noise_mix, outputs[1].noise_mix)
                np.testing.assert_array_equal(outputs[0].gate_out.mix_llrs, outputs[1].gate_out.mix_llrs)

    def test_single_ray_matches_one_anchor_gaussian_integral(self):
        obs, atoms = _one_ray_fixture()
        for phase in (0.0, 0.4):
            y_b = obs.y_b.copy()
            y_b[obs.training_rows[0]] *= np.exp(1j * phase)
            current = replace(obs, y_b=y_b)
            expected_mean, expected_var, d_mat, doubled_mean = _independent_anchor_posterior(current, atoms)
            with patch.object(receiver, "build_offgrid_branches", side_effect=AssertionError("B-conditioned prior")):
                actual = _run_fixed_support(current, atoms)
            data_rows = [r for r in range(T) if r not in (obs.training_rows[0], PROBE_ROW)]
            with self.subTest(anchor_phase=phase):
                self.assertFalse(actual.delay_reset_triggered)
                self.assertGreater(float(np.max(np.abs(expected_mean - doubled_mean))), 1e-4)
                np.testing.assert_allclose(
                    actual.h_mix[data_rows], expected_mean[data_rows, :, None] * d_mat,
                    rtol=3e-12, atol=3e-12,
                )
                np.testing.assert_allclose(
                    actual.noise_mix[data_rows], obs.n0 + expected_var[data_rows],
                    rtol=3e-12, atol=3e-12,
                )

    def test_removing_bundle_keeps_public_observation_validation(self):
        obs, atoms = _one_ray_fixture()
        bad_y = obs.y_b.copy()
        bad_y[0, 0] = np.nan
        invalid = (
            replace(obs, y_b=obs.y_b[:, :-1]),
            replace(obs, y_b=bad_y),
            replace(obs, training_rows=()),
            replace(obs, training_rows=(PROBE_ROW,)),
            replace(obs, pilot_tx_b=obs.pilot_tx_b[:, :-1]),
        )
        for i, bad in enumerate(invalid):
            with self.subTest(case=i), self.assertRaises(ValueError):
                _run_fixed_support(bad, atoms)


if __name__ == "__main__":
    unittest.main()
