"""Independent finite-constellation and leave-one-out cavity checks."""
import itertools
import math
import unittest

import torch

from p5.unrolled_em_transformer import _llrs, _soft_symbols
from p5b.parity import min_sum_extrinsic
from p6_consistency.reliability import (SymbolPrior, _qam_moments,
                                        cavity_demap, make_prior)


def complex_random(shape, generator):
    return torch.complex(torch.randn(shape, generator=generator, dtype=torch.float64),
                         torch.randn(shape, generator=generator, dtype=torch.float64))


class ReliabilityTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.generator = torch.Generator().manual_seed(813611190071)

    def fixture(self, batch=2, rows=2, tones=4):
        y = complex_random((batch, rows, tones), self.generator)
        g = complex_random((batch, rows, tones, tones), self.generator) / math.sqrt(tones)
        weights = .2 + .8 * torch.rand(y.shape, generator=self.generator, dtype=torch.float64)
        nv = torch.linspace(.3, .7, batch, dtype=torch.float64)
        bit_logits = .5 * torch.randn(*y.shape, 4, generator=self.generator, dtype=torch.float64)
        mean, variance = _qam_moments(bit_logits)
        prior = SymbolPrior(mean, variance, bit_logits)
        mask = torch.zeros(rows, tones, dtype=torch.bool)
        pilots = torch.zeros(rows, tones, dtype=torch.complex128)
        return y, g, nv, weights, prior, mask, pilots

    def test_prior_and_posterior_against_all_16_symbols(self):
        shape = (1, 2, 3)
        bits = torch.tensor(list(itertools.product((0., 1.), repeat=4)), dtype=torch.float64)
        # Gray mapper, independently enumerated from the four-bit definition.
        real = (2*bits[:, 0]-1) * (3-2*bits[:, 1]) / math.sqrt(10.)
        imag = (2*bits[:, 2]-1) * (3-2*bits[:, 3]) / math.sqrt(10.)
        constellation = torch.complex(real, imag)
        logits = torch.randn(*shape, 4, generator=self.generator, dtype=torch.float64)
        z = complex_random(shape, self.generator)
        variance = .2 + torch.rand(shape, generator=self.generator, dtype=torch.float64)
        informed = torch.tensor([[[True, False, True], [True, True, False]]])
        for use_observation in (False, True):
            got = (_qam_moments(logits, z, variance, informed) if use_observation
                   else _qam_moments(logits))
            score = logits @ bits.T
            if use_observation:
                likelihood = -(z[..., None]-constellation).abs().square() / variance[..., None]
                score = score + torch.where(informed[..., None], likelihood, 0.)
            probability = score.softmax(-1)
            expected_mean = (probability * constellation).sum(-1)
            expected_variance = (probability * (constellation-expected_mean[..., None]).abs().square()).sum(-1)
            torch.testing.assert_close(got[0], expected_mean, atol=5e-16, rtol=2e-14)
            torch.testing.assert_close(got[1], expected_variance, atol=6e-16, rtol=2e-14)

    def test_initial_uniform_prior_and_known_pilots(self):
        template = complex_random((2, 2, 32), self.generator)
        mask = torch.zeros(2, 32, dtype=torch.bool)
        mask[0] = True
        pilots = torch.zeros(2, 32, dtype=template.dtype)
        pilots[mask] = .6 + .8j
        prior = make_prior(template, mask, pilots)
        self.assertTrue(torch.equal(prior.mean[:, ~mask], torch.zeros_like(prior.mean[:, ~mask])))
        torch.testing.assert_close(prior.variance[:, ~mask], torch.ones_like(prior.variance[:, ~mask]))
        self.assertTrue(torch.equal(prior.mean[:, mask], pilots[mask].expand(2, -1)))
        self.assertEqual(float(prior.variance[:, mask].abs().max()), 0.)
        self.assertEqual(float(prior.bit_logits.abs().max()), 0.)

    def test_later_prior_uses_only_extrinsic_messages(self):
        template = complex_random((2, 2, 32), self.generator)
        mask = torch.zeros(2, 32, dtype=torch.bool)
        mask[0] = True
        pilots = torch.zeros_like(template[0])
        pilots[mask] = 1.
        previous = torch.randn(2, 2, 32, 4, generator=self.generator, dtype=torch.float64)
        actual = make_prior(template, mask, pilots, previous)
        expected = min_sum_extrinsic(previous[:, ~mask].reshape(2, -1, 128)).reshape(2, 32, 4)
        self.assertTrue(torch.equal(actual.bit_logits[:, ~mask], expected))
        self.assertFalse(torch.equal(actual.bit_logits[:, ~mask], expected+previous[:, ~mask]))
        disabled = make_prior(template, mask, pilots, previous, parity=False)
        self.assertEqual(float(disabled.bit_logits.abs().max()), 0.)

    def test_cavity_matches_independent_leave_one_symbol_out(self):
        y, g, nv, weights, prior, mask, pilots = self.fixture()
        actual = cavity_demap(y, g, nv, weights, prior, mask, pilots)
        expected_z = torch.empty_like(y)
        expected_variance = torch.empty_like(y.real)
        for batch in range(y.shape[0]):
            for row in range(y.shape[1]):
                weighted_g = weights[batch, row].sqrt()[:, None] * g[batch, row]
                weighted_y = weights[batch, row].sqrt() * y[batch, row]
                for tone in range(y.shape[-1]):
                    other = [index for index in range(y.shape[-1]) if index != tone]
                    interferers = weighted_g[:, other]
                    covariance = (nv[batch] * torch.eye(y.shape[-1], dtype=y.dtype)
                                  + (interferers * prior.variance[batch, row, other]) @ interferers.mH)
                    residual = weighted_y-interferers @ prior.mean[batch, row, other]
                    column = weighted_g[:, tone]
                    solved = torch.linalg.solve(covariance, column)
                    precision = (column.conj() @ solved).real
                    expected_z[batch, row, tone] = (solved.conj() @ residual) / precision
                    expected_variance[batch, row, tone] = 1/precision
        torch.testing.assert_close(actual.z, expected_z, atol=8e-15, rtol=8e-14)
        torch.testing.assert_close(actual.variance, expected_variance, atol=8e-15, rtol=8e-14)
        self.assertEqual(actual.diagnostics["computational_noise_floor_frames"], 0)
        self.assertEqual(actual.diagnostics["cavity_variance_clamps"], 0)

    def test_diagonal_operator_reduces_to_archived_likelihood(self):
        y, _, nv, weights, prior, mask, pilots = self.fixture()
        h = complex_random(y.shape, self.generator) + 1.
        weights = torch.ones_like(weights)
        actual = cavity_demap(y, torch.diag_embed(h), nv, weights, prior, mask, pilots)
        expected_z = y/h
        expected_variance = nv[:, None, None] / h.abs().square()
        torch.testing.assert_close(actual.z, expected_z, atol=3e-15, rtol=3e-14)
        torch.testing.assert_close(actual.variance, expected_variance, atol=3e-15, rtol=3e-14)
        torch.testing.assert_close(actual.channel_llr, _llrs(expected_z, expected_variance), atol=2e-13, rtol=3e-14)
        uniform = make_prior(y, mask, pilots)
        answer = cavity_demap(y, torch.diag_embed(h), nv, weights, uniform, mask, pilots)
        torch.testing.assert_close(answer.mean, _soft_symbols(expected_z, expected_variance), atol=8e-16, rtol=3e-14)

    def test_raw_complex_gain_covariance(self):
        y, g, nv, weights, prior, mask, pilots = self.fixture()
        expected = cavity_demap(y, g, nv, weights, prior, mask, pilots)
        for scale in (1e-35, .13, 1e20):
            gain = scale * complex(math.cos(.73), math.sin(.73))
            with self.subTest(scale=scale):
                actual = cavity_demap(gain*y, gain*g, abs(gain)**2*nv, weights, prior, mask, pilots)
                for name in ("z", "variance", "mean", "posterior_variance", "channel_llr"):
                    torch.testing.assert_close(getattr(actual, name), getattr(expected, name), atol=4e-13, rtol=4e-13)

    def test_all_missing_preserves_prior_without_channel_information(self):
        y, g, nv, weights, prior, mask, pilots = self.fixture()
        actual = cavity_demap(y, g, nv, torch.zeros_like(weights), prior, mask, pilots)
        self.assertFalse(bool(actual.informed.any()))
        self.assertEqual(float(actual.channel_llr.abs().max()), 0.)
        torch.testing.assert_close(actual.mean, prior.mean, atol=0., rtol=0.)
        torch.testing.assert_close(actual.posterior_variance, prior.variance, atol=0., rtol=0.)
        self.assertEqual(actual.diagnostics["valid_observations_by_burst"], [0, 0])

    def test_zero_channel_does_not_manufacture_information(self):
        y, g, nv, weights, prior, mask, pilots = self.fixture()
        actual = cavity_demap(y, torch.zeros_like(g), nv, weights, prior, mask, pilots)
        self.assertFalse(bool(actual.informed.any()))
        self.assertEqual(float(actual.channel_llr.abs().max()), 0.)

    def test_masked_output_rows_are_irrelevant(self):
        y, g, nv, weights, prior, mask, pilots = self.fixture()
        weights[..., ::2] = 0.
        expected = cavity_demap(y, g, nv, weights, prior, mask, pilots)
        changed_y, changed_g = y.clone(), g.clone()
        changed_y[..., ::2] = 1e10 + 3e10j
        changed_g[..., ::2, :] = 2e10 - 1e10j
        actual = cavity_demap(changed_y, changed_g, nv, weights, prior, mask, pilots)
        for name in ("z", "variance", "mean", "posterior_variance", "channel_llr"):
            self.assertTrue(torch.equal(getattr(actual, name), getattr(expected, name)))

    def test_zero_noise_is_explicitly_regularized_and_finite(self):
        y, g, nv, weights, prior, mask, pilots = self.fixture()
        weights[..., 0] = 0.
        actual = cavity_demap(y, g, torch.zeros_like(nv), weights, prior, mask, pilots)
        for name in ("z", "variance", "mean", "posterior_variance", "channel_llr"):
            self.assertTrue(bool(torch.isfinite(getattr(actual, name)).all()))
        self.assertEqual(actual.diagnostics["computational_noise_floor_frames"], 4)
        self.assertLessEqual(float(actual.channel_llr.abs().max()), -math.log(torch.finfo(torch.float64).eps))
        missing = cavity_demap(y, g, torch.zeros_like(nv), torch.zeros_like(weights), prior, mask, pilots)
        self.assertEqual(float(missing.channel_llr.abs().max()), 0.)

    def test_channel_readout_does_not_readd_own_symbol_prior(self):
        y, _, nv, weights, prior, mask, pilots = self.fixture()
        h = complex_random(y.shape, self.generator) + 1.
        g = torch.diag_embed(h)
        uniform = make_prior(y, mask, pilots)
        before = cavity_demap(y, g, nv, weights, uniform, mask, pilots)
        after = cavity_demap(y, g, nv, weights, prior, mask, pilots)
        torch.testing.assert_close(after.channel_llr, before.channel_llr, atol=2e-13, rtol=2e-13)
        self.assertGreater(float((after.mean-before.mean).abs().max()), 1e-4)

    def test_full_operator_cavity_excludes_own_gaussian_moments(self):
        y, g, nv, weights, prior, mask, pilots = self.fixture()
        before = cavity_demap(y, g, nv, weights, prior, mask, pilots)
        changed_mean, changed_variance = prior.mean.clone(), prior.variance.clone()
        changed_mean[..., 1] += .8 - .5j
        changed_variance[..., 1] *= .17
        changed = SymbolPrior(changed_mean, changed_variance, prior.bit_logits)
        after = cavity_demap(y, g, nv, weights, changed, mask, pilots)
        torch.testing.assert_close(after.z[..., 1], before.z[..., 1], atol=1e-13, rtol=1e-13)
        torch.testing.assert_close(after.variance[..., 1], before.variance[..., 1], atol=1e-13, rtol=1e-13)

    def test_all_known_pilots_have_no_payload_messages(self):
        y, g, nv, weights, _, mask, pilots = self.fixture()
        mask = torch.ones_like(mask)
        pilots = torch.ones_like(pilots)
        prior = make_prior(y, mask, pilots)
        actual = cavity_demap(y, g, nv, weights, prior, mask, pilots)
        self.assertEqual(actual.diagnostics["payload_symbols"], 0)
        self.assertEqual(float(actual.channel_llr.abs().max()), 0.)
        self.assertTrue(torch.equal(actual.mean, torch.ones_like(y)))
        self.assertEqual(float(actual.posterior_variance.abs().max()), 0.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
