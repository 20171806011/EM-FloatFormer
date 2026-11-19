"""Independent small-problem identities for the soft-information proposal."""

import itertools
import math
import unittest

import torch

from p5.unrolled_em_transformer import _llrs, _soft_symbols
from p5b.parity import min_sum_extrinsic, soft_parity_symbols
from p6_softinfo.moments import symbol_moments, waveform_gram_correction


class MomentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(26261119)

    def test_no_parity_mean_is_archived_mean(self):
        z = torch.randn(2, 4, 10, dtype=torch.complex128)
        sigma = torch.rand(2, 4, 10, dtype=torch.float64) + .1
        informed = torch.rand(2, 4, 10) > .3
        mean, var = symbol_moments(z, sigma, informed)
        reference = torch.where(informed, _soft_symbols(z, sigma), 0.)
        self.assertTrue(torch.equal(mean, reference))
        self.assertTrue(torch.equal(var[~informed], torch.ones_like(var[~informed])))
        self.assertTrue((var >= 0).all())

    def test_no_parity_matches_joint_constellation_enumeration(self):
        z = torch.randn(2, 3, 4, dtype=torch.complex128)
        sigma = torch.rand(2, 3, 4, dtype=torch.float64) + .1
        mean, var = symbol_moments(z, sigma, torch.ones_like(sigma, dtype=torch.bool))
        qam = torch.tensor([complex(a, b) / math.sqrt(10)
                            for a, b in itertools.product((-3, -1, 1, 3), repeat=2)],
                           dtype=torch.complex128)
        posterior = (-(z[..., None] - qam).abs().square() / sigma[..., None]).softmax(-1)
        expected = (posterior * qam).sum(-1)
        direct_var = (posterior * (qam - expected[..., None]).abs().square()).sum(-1)
        torch.testing.assert_close(mean, expected, rtol=2e-14, atol=5e-16)
        torch.testing.assert_close(var, direct_var, rtol=2e-13, atol=8e-16)

    def test_parity_mean_is_archived_mean_and_pilots_are_exact(self):
        mask = torch.zeros(4, 10, dtype=torch.bool)
        mask[:, :2] = True  # 32 payload RE = one 128-bit codeword.
        z = torch.randn(2, 4, 10, dtype=torch.complex128)
        sigma = torch.rand(2, 4, 10, dtype=torch.float64) + .1
        informed = torch.rand(2, 4, 10) > .3
        pilots = torch.full((4, 10), 1 + 1j, dtype=torch.complex128)
        mean, var = symbol_moments(z, sigma, informed,
                                   pilot_mask=mask, pilots=pilots, parity=True)
        old = soft_parity_symbols(z, sigma, informed, mask)
        reference = torch.where(mask, pilots, old)
        self.assertTrue(torch.equal(mean, reference))
        self.assertTrue(torch.equal(var[:, mask], torch.zeros_like(var[:, mask])))
        self.assertTrue((var >= 0).all())

    def test_parity_variance_matches_joint_qam_enumeration(self):
        mask = torch.zeros(4, 10, dtype=torch.bool)
        mask[:, :2] = True
        z = torch.randn(2, 4, 10, dtype=torch.complex128)
        sigma = torch.rand(2, 4, 10, dtype=torch.float64) + .1
        informed = torch.rand(2, 4, 10) > .3
        mean, var = symbol_moments(z, sigma, informed, pilot_mask=mask, parity=True)
        zd, sd, observed = z[:, ~mask], sigma[:, ~mask], informed[:, ~mask]
        channel = torch.where(observed[..., None], _llrs(zd, sd), 0.)
        extrinsic = min_sum_extrinsic(channel.reshape(2, -1, 128)).reshape_as(channel)
        levels = (-3, -1, 1, 3)
        gray = ((0, 0), (0, 1), (1, 1), (1, 0))
        constellation = torch.tensor([complex(levels[i], levels[j]) / math.sqrt(10)
                                      for i, j in itertools.product(range(4), repeat=2)],
                                     dtype=torch.complex128)
        bits = torch.tensor([gray[i] + gray[j]
                             for i, j in itertools.product(range(4), repeat=2)],
                            dtype=torch.float64)
        likelihood = torch.where(observed[..., None],
                                 -(zd[..., None] - constellation).abs().square() / sd[..., None], 0.)
        joint = (likelihood + extrinsic @ bits.T).softmax(-1)
        expected = (joint * constellation).sum(-1)
        direct_var = (joint * (constellation - expected[..., None]).abs().square()).sum(-1)
        torch.testing.assert_close(mean[:, ~mask], expected, rtol=3e-14, atol=9e-16)
        torch.testing.assert_close(var[:, ~mask], direct_var, rtol=4e-13, atol=2e-15)

    def test_fully_uninformed_parity_is_uniform(self):
        z = torch.randn(1, 4, 10, dtype=torch.complex128)
        mask = torch.zeros(4, 10, dtype=torch.bool)
        mask[:, :2] = True
        mean, var = symbol_moments(z, torch.ones_like(z.real),
                                   torch.zeros_like(z.real, dtype=torch.bool),
                                   pilot_mask=mask, parity=True)
        self.assertTrue(torch.equal(mean, torch.zeros_like(mean)))
        torch.testing.assert_close(var[:, ~mask], torch.ones_like(var[:, ~mask]),
                                   rtol=0, atol=3e-16)

    def test_nonparity_moments_are_differentiable(self):
        z = torch.tensor([[[.13 + .21j, -.32 + .16j]]], dtype=torch.complex128,
                         requires_grad=True)
        sigma = torch.tensor([[[.2, .4]]], dtype=torch.float64, requires_grad=True)
        informed = torch.ones_like(sigma, dtype=torch.bool)
        self.assertTrue(torch.autograd.gradcheck(
            lambda a, b: symbol_moments(a, b, informed), (z, sigma),
            eps=1e-6, atol=2e-7, rtol=2e-5))


class ExpectedGramTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17261119)

    def test_matches_explicit_linear_covariance_propagation(self):
        b, t, n, p = 2, 3, 5, 3
        v = torch.rand(b, t, n, dtype=torch.float64)
        basis = torch.randn(b, p, n, dtype=torch.complex128)
        rot = torch.exp(1j * torch.randn(b, t, n, p, dtype=torch.float64))
        cpe = torch.randn(b, t, dtype=torch.complex128)
        actual = waveform_gram_correction(v, basis, rot, cpe)
        # Explicit response to each frequency-bin unit vector. Unlike the
        # closed form, this retains the full time-domain covariance matrices.
        ifft = torch.fft.ifft(torch.eye(n, dtype=torch.complex128), dim=0, norm="ortho")
        expected = torch.zeros_like(actual)
        for batch in range(b):
            for row in range(t):
                operators = [torch.diag(cpe[batch, row] * rot[batch, row, :, ray])
                             @ ifft @ torch.diag(basis[batch, ray]) for ray in range(p)]
                covariance = torch.diag(v[batch, row].to(torch.complex128))
                for first in range(p):
                    for second in range(p):
                        expected[batch, first, second] += torch.trace(
                            operators[second] @ covariance @ operators[first].mH)
        torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)
        torch.testing.assert_close(actual, actual.mH, rtol=2e-14, atol=2e-14)
        self.assertGreaterEqual(torch.linalg.eigvalsh(actual).min().item(), -1e-12)
        self.assertGreater(actual[:, 0, 1].abs().max().item(), 1e-3)

    def test_exact_discrete_expected_gram_and_loss(self):
        b, t, n, p = 1, 1, 3, 2
        mean = torch.randn(b, t, n, dtype=torch.complex128)
        v = torch.tensor([[[.4, .2, .7]]], dtype=torch.float64)
        basis = torch.randn(b, p, n, dtype=torch.complex128)
        rot = torch.exp(1j * torch.randn(b, t, n, p, dtype=torch.float64))
        cpe = torch.exp(1j * torch.randn(b, t, dtype=torch.float64))
        amp = torch.randn(b, p, 1, dtype=torch.complex128)
        y = torch.randn(b, t*n, 1, dtype=torch.complex128)

        def design(x):
            wave = torch.fft.ifft(x[..., None] * basis.mT[:, None], dim=2, norm="ortho")
            return (wave * rot * cpe[:, :, None, None]).reshape(b, t*n, p)

        # Enumerate an independent zero-mean two-point distribution in each
        # frequency bin. Its chosen variance is exact; no Monte Carlo error.
        mean_design = design(mean)
        grams, losses, right_sides = [], [], []
        for signs in itertools.product((-1., 1.), repeat=n):
            sample = mean + v.sqrt() * torch.tensor(signs, dtype=torch.float64)
            d = design(sample)
            grams.append(d.mH @ d)
            losses.append((y - d @ amp).abs().square().sum())
            right_sides.append(d.mH @ y)
        correction = waveform_gram_correction(v, basis, rot, cpe)
        torch.testing.assert_close(torch.stack(grams).mean(0),
                                   mean_design.mH @ mean_design + correction,
                                   rtol=2e-14, atol=2e-14)
        torch.testing.assert_close(torch.stack(right_sides).mean(0), mean_design.mH @ y,
                                   rtol=2e-14, atol=2e-14)
        proposed = ((y - mean_design @ amp).abs().square().sum()
                    + (amp.mH @ correction @ amp).real.sum())
        torch.testing.assert_close(torch.stack(losses).mean(), proposed,
                                   rtol=2e-14, atol=2e-14)

    def test_zero_variance_is_exact_zero(self):
        basis = torch.randn(2, 3, 7, dtype=torch.complex128)
        rot = torch.exp(1j * torch.randn(2, 4, 7, 3, dtype=torch.float64))
        cpe = torch.randn(2, 4, dtype=torch.complex128)
        result = waveform_gram_correction(torch.zeros(2, 4, 7, dtype=torch.float64),
                                          basis, rot, cpe)
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))

    def test_no_rotation_and_shared_basis_match_explicit_ones(self):
        v = torch.rand(2, 4, 7, dtype=torch.float64)
        basis = torch.randn(3, 7, dtype=torch.complex128)
        cpe = torch.randn(2, 4, dtype=torch.complex128)
        missing = waveform_gram_correction(v, basis, None, cpe)
        explicit = waveform_gram_correction(v, basis.expand(2, -1, -1),
                                            torch.ones(2, 4, 7, 3, dtype=torch.complex128), cpe)
        self.assertTrue(torch.equal(missing, explicit))

    def test_phase_curvature_supports_second_derivatives(self):
        v = torch.rand(1, 2, 4, dtype=torch.float64)
        basis = torch.randn(1, 2, 4, dtype=torch.complex128)
        cpe = torch.ones(1, 2, dtype=torch.complex128)
        amp = torch.tensor([[[.7 + .2j], [.4 - .3j]]], dtype=torch.complex128)
        coordinates = torch.randn(1, 2, 4, 2, 2, dtype=torch.float64)
        theta = torch.tensor([.13, -.21], dtype=torch.float64, requires_grad=True)

        def loss(parameters):
            rot = torch.exp(1j * (coordinates @ parameters))
            correction = waveform_gram_correction(v, basis, rot, cpe)
            return (amp.mH @ correction @ amp).real.sum()

        self.assertTrue(torch.autograd.gradgradcheck(loss, (theta,),
                                                     eps=1e-6, atol=2e-7, rtol=2e-5))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
