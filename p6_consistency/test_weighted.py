"""Independent exact identities for receive-frequency weighted R2 updates."""

import itertools
import math
import unittest

import torch

from p6_softinfo.moments import waveform_gram_correction
from p6_consistency.weighted import (frequency_operators, weighted_amplitude_update,
                                      weighted_objective, weighted_terms)


class WeightedWaveformTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(28161119)
        torch.set_num_threads(1)

    def example(self, b=2, t=2, n=4, p=3, r=2):
        mean = torch.randn(b, t, n, dtype=torch.complex128)
        variance = torch.rand(b, t, n, dtype=torch.float64)
        y = torch.randn(b, r, t, n, dtype=torch.complex128)
        basis = torch.exp(1j * torch.randn(b, p, n, dtype=torch.float64))
        rot = torch.exp(1j * torch.randn(b, t, n, p, dtype=torch.float64))
        cpe = torch.exp(1j * torch.randn(b, t, dtype=torch.float64))
        weights = torch.rand(b, t, n, dtype=torch.float64) + .1
        prior = torch.randn(b, p, p, dtype=torch.complex128)
        ridge = .1 * (prior @ prior.mH) + .05 * torch.eye(p, dtype=torch.float64)
        delta = .15 * torch.randn(b, r, t, n, dtype=torch.complex128)
        return mean, variance, y, basis, rot, cpe, weights, ridge, delta

    def test_operator_matches_explicit_dft_matrices(self):
        mean, _, _, basis, rot, cpe, _, _, _ = self.example()
        b, t, n = mean.shape
        operators = frequency_operators(basis, rot, cpe)
        axis = torch.arange(n, dtype=torch.float64)
        fourier = torch.exp(-2j * math.pi * axis[:, None] * axis[None, :] / n) / math.sqrt(n)
        for batch, row, ray in itertools.product(range(b), range(t), range(basis.shape[1])):
            explicit = (cpe[batch, row] * fourier @ torch.diag(rot[batch, row, :, ray])
                        @ fourier.mH @ torch.diag(basis[batch, ray]))
            torch.testing.assert_close(operators[batch, row, ray], explicit,
                                       atol=8e-16, rtol=3e-15)

    def test_uniform_weights_recover_full_time_domain_gram(self):
        mean, variance, y, basis, rot, cpe, _, ridge, _ = self.example()
        weights = torch.ones_like(variance)
        solution = weighted_amplitude_update(mean, variance, y, basis, rot, cpe,
                                             weights, ridge)
        wave = torch.fft.ifft(mean[..., None] * basis.mT[:, None], dim=2, norm="ortho")
        design = (wave * rot * cpe[:, :, None, None]).flatten(1, 2)
        correction = waveform_gram_correction(variance, basis, rot, cpe)
        expected_normal = design.mH @ design + correction + ridge
        received = torch.fft.ifft(y, dim=-1, norm="ortho").flatten(2).mT
        expected_rhs = design.mH @ received
        torch.testing.assert_close(solution.gram_correction, correction, atol=1e-14, rtol=3e-14)
        torch.testing.assert_close(solution.normal, expected_normal, atol=2e-14, rtol=3e-14)
        torch.testing.assert_close(solution.rhs, expected_rhs, atol=1e-14, rtol=3e-14)
        torch.testing.assert_close(solution.amplitudes,
                                   torch.linalg.solve(expected_normal, expected_rhs),
                                   atol=2e-14, rtol=4e-14)

    def test_exact_qam_enumeration_with_known_pilot_weights_and_delta(self):
        # One known transmitted pilot and two independent, nonuniform QAM
        # symbols. Erasing the pilot *observation* does not randomize the pilot.
        b, t, n, p, r = 1, 1, 3, 2, 2
        _, _, y, basis, rot, cpe, _, ridge, delta = self.example(b, t, n, p, r)
        qam = torch.tensor([complex(a, b) / math.sqrt(10)
                            for a, b in itertools.product((-3, -1, 1, 3), repeat=2)],
                           dtype=torch.complex128)
        probabilities = torch.randn(2, 16, dtype=torch.float64).softmax(-1)
        data_means = (probabilities * qam).sum(-1)
        data_vars = (probabilities * (qam - data_means[:, None]).abs().square()).sum(-1)
        pilot = complex(1, 1) / math.sqrt(2)
        mean = torch.tensor([[[pilot, 0, 0]]], dtype=torch.complex128)
        mean[0, 0, 1:] = data_means
        variance = torch.zeros(1, 1, 3, dtype=torch.float64)
        variance[0, 0, 1:] = data_vars
        weights = torch.tensor([[[0., .7, 2.3]]], dtype=torch.float64)
        solution = weighted_amplitude_update(mean, variance, y, basis, rot, cpe,
                                             weights, ridge, delta)
        samples = torch.tensor([[pilot, first.item(), second.item()]
                                for first, second in itertools.product(qam, repeat=2)],
                               dtype=torch.complex128).reshape(-1, 1, 3)
        mass = (probabilities[0, :, None] * probabilities[1, None, :]).flatten()
        # Build every sample's response through FFT/IFFT, without using the
        # operator constructor or moment algebra under test.
        time = torch.fft.ifft(samples[..., None] * basis[0].mT[None, None],
                             dim=2, norm="ortho")
        design = torch.fft.fft(time * rot[0:1] * cpe[0:1, :, None, None],
                              dim=2, norm="ortho").flatten(1, 2)
        w = weights.flatten(1)[0]
        gram = design.mH @ (w[None, :, None] * design)
        residual_data = (y[0, :, 0].mT[None]
                         - delta[0, :, 0].mT[None] * samples[:, 0, :, None])
        rhs = design.mH @ (w[None, :, None] * residual_data)
        expected_normal = (gram * mass[:, None, None]).sum(0)[None] + ridge
        expected_rhs = (rhs * mass[:, None, None]).sum(0)[None]
        torch.testing.assert_close(solution.normal, expected_normal, atol=2e-14, rtol=3e-14)
        torch.testing.assert_close(solution.rhs, expected_rhs, atol=2e-14, rtol=3e-14)
        amplitude = solution.amplitudes[0]
        residuals = residual_data - design @ amplitude
        sampled_energy = (residuals.abs().square() * w[None, :, None]).sum((1, 2))
        prior = (amplitude.conj() * (ridge[0] @ amplitude)).real.sum()
        expected_objective = (sampled_energy * mass).sum() + prior
        torch.testing.assert_close(solution.objective[0], expected_objective,
                                   atol=2e-14, rtol=3e-14)
        self.assertGreater(solution.covariance_rhs.abs().max().item(), 1e-3)
        self.assertEqual(variance[0, 0, 0].item(), 0.)
        self.assertEqual(mean[0, 0, 0].item(), pilot)

    def test_solution_matches_independent_augmented_weighted_least_squares(self):
        mean, variance, y, basis, rot, cpe, weights, ridge, delta = self.example()
        operators = frequency_operators(basis, rot, cpe)
        solution = weighted_amplitude_update(mean, variance, y, basis, rot, cpe,
                                             weights, ridge, delta)
        b, t, n = mean.shape
        p, r = basis.shape[1], y.shape[1]
        for batch in range(b):
            design_rows, targets = [], []
            for row in range(t):
                for out_bin in range(n):
                    weight = weights[batch, row, out_bin].sqrt()
                    design_rows.append((operators[batch, row, :, out_bin] * mean[batch, row]).sum(-1)
                                       * weight)
                    targets.append((y[batch, :, row, out_bin]
                                    - delta[batch, :, row, out_bin] * mean[batch, row, out_bin]) * weight)
                    for in_bin in range(n):
                        scale = weight * variance[batch, row, in_bin].sqrt()
                        design_rows.append(operators[batch, row, :, out_bin, in_bin] * scale)
                        target = (-delta[batch, :, row, out_bin] if in_bin == out_bin
                                  else torch.zeros(r, dtype=torch.complex128))
                        targets.append(target * scale)
            prior_root = torch.linalg.cholesky(ridge[batch]).mH
            design_rows.extend(prior_root.unbind(0))
            targets.extend(torch.zeros(p, r, dtype=torch.complex128).unbind(0))
            expanded_design, expanded_y = torch.stack(design_rows), torch.stack(targets)
            qr_solution = torch.linalg.lstsq(expanded_design, expanded_y, driver="gels").solution
            torch.testing.assert_close(solution.amplitudes[batch], qr_solution,
                                       atol=3e-14, rtol=4e-14)
            objective = (expanded_y - expanded_design @ qr_solution).abs().square().sum()
            torch.testing.assert_close(solution.objective[batch], objective,
                                       atol=3e-13, rtol=3e-14)

    def test_heteroscedastic_noise_precision_equals_explicit_whitening(self):
        mean, variance, y, basis, rot, cpe, _, ridge, delta = self.example(b=1)
        variance = torch.zeros_like(variance)
        noise_variance = torch.tensor([[[.2, .6, 1.2, 2.], [.3, .9, 1.8, 2.7]]], dtype=torch.float64)
        weights = noise_variance.reciprocal()
        operators = frequency_operators(basis, rot, cpe)
        solution = weighted_amplitude_update(mean, variance, y, basis, rot, cpe,
                                             weights, ridge, delta)
        design = torch.einsum("btpfk,btk->btfp", operators, mean).flatten(1, 2)[0]
        residual = (y - delta * mean[:, None]).flatten(2).mT[0]
        root = noise_variance.flatten().sqrt()
        expanded_design = torch.cat((design / root[:, None], torch.linalg.cholesky(ridge[0]).mH))
        expanded_y = torch.cat((residual / root[:, None], torch.zeros_like(solution.amplitudes[0])))
        expected = torch.linalg.lstsq(expanded_design, expanded_y, driver="gels").solution
        torch.testing.assert_close(solution.amplitudes[0], expected, atol=3e-14, rtol=3e-14)

    def test_zero_weighted_observations_cannot_change_solution(self):
        mean, variance, y, basis, rot, cpe, weights, ridge, delta = self.example()
        weights[..., 1] = 0.
        first = weighted_amplitude_update(mean, variance, y, basis, rot, cpe, weights, ridge, delta)
        changed = y.clone()
        changed[..., 1] = 1e8 * (1 + 2j)
        second = weighted_amplitude_update(mean, variance, changed, basis, rot, cpe, weights, ridge, delta)
        self.assertTrue(torch.equal(first.amplitudes, second.amplitudes))
        self.assertTrue(torch.equal(first.objective, second.objective))

    def test_all_zero_weights_have_finite_minimum_norm_solution(self):
        mean, variance, y, basis, rot, cpe, weights, ridge, delta = self.example()
        weights = torch.zeros_like(weights)
        for prior in (ridge, torch.zeros_like(ridge)):
            solved = weighted_amplitude_update(mean, variance, y, basis, rot, cpe, weights, prior, delta)
            self.assertTrue(torch.equal(solved.amplitudes, torch.zeros_like(solved.amplitudes)))
            self.assertTrue(torch.equal(solved.objective, torch.zeros_like(solved.objective)))
            self.assertTrue(torch.isfinite(solved.normal).all())
            self.assertEqual(bool(solved.used_pinv.all()), bool(torch.equal(prior, torch.zeros_like(prior))))

    def test_covariance_is_hermitian_psd_and_zero_variance_removes_cross(self):
        mean, variance, y, basis, rot, cpe, weights, ridge, delta = self.example()
        operators = frequency_operators(basis, rot, cpe)
        _, _, _, covariance, _, _, _, _ = weighted_terms(mean, variance, y, operators,
                                                        weights, ridge, delta)
        torch.testing.assert_close(covariance, covariance.mH, atol=1e-14, rtol=1e-14)
        self.assertGreaterEqual(torch.linalg.eigvalsh(covariance).min().item(), -1e-12)
        zero = weighted_amplitude_update(mean, torch.zeros_like(variance), y, basis, rot,
                                         cpe, weights, ridge, delta)
        self.assertTrue(torch.equal(zero.gram_correction, torch.zeros_like(zero.gram_correction)))
        self.assertTrue(torch.equal(zero.covariance_rhs, torch.zeros_like(zero.covariance_rhs)))
        self.assertTrue(torch.equal(zero.uncertainty_energy, torch.zeros_like(zero.uncertainty_energy)))

    def test_exact_solution_is_stationary_for_direct_objective(self):
        mean, variance, y, basis, rot, cpe, weights, ridge, delta = self.example(b=1)
        operators = frequency_operators(basis, rot, cpe)
        solved = weighted_amplitude_update(mean, variance, y, basis, rot, cpe, weights, ridge, delta)
        amplitude = solved.amplitudes.detach().requires_grad_(True)
        objective = weighted_objective(mean, variance, y, operators, amplitude, weights, ridge, delta)[0]
        gradient = torch.autograd.grad(objective.sum(), amplitude)[0]
        self.assertLess(gradient.abs().max().item(), 2e-13)

    def test_negative_or_nonfinite_weights_and_variances_are_rejected(self):
        mean, variance, y, basis, rot, cpe, weights, ridge, delta = self.example()
        for value in (-.1, float("nan"), float("inf")):
            with self.subTest(value=value):
                bad_weights = weights.clone()
                bad_weights[0, 0, 0] = value
                with self.assertRaises(ValueError):
                    weighted_amplitude_update(mean, variance, y, basis, rot, cpe,
                                              bad_weights, ridge, delta)
                bad_variance = variance.clone()
                bad_variance[0, 0, 0] = value
                with self.assertRaises(ValueError):
                    weighted_amplitude_update(mean, bad_variance, y, basis, rot, cpe,
                                              weights, ridge, delta)


if __name__ == "__main__":
    unittest.main(verbosity=2)
