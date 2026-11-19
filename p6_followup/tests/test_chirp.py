"""Focused mathematical and integration tests for the follow-up chirp map."""

import unittest

import torch

from p6 import math_chirp, math_chirp_trust
from p6.axial_em_transformer import _dictionary
from p5b.operator_em_transformer import _ray_bank
from p6_followup.chirp import chart_coefficients, confidence_chart_mix, residual


class ChartMathTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.generator = torch.Generator().manual_seed(261119)

    def randn(self, *shape):
        return torch.randn(*shape, generator=self.generator, dtype=torch.float64)

    def test_coefficients_are_convex_and_unchanged_by_common_logit_shift(self):
        positive, negative = self.randn(4, 3, 81), self.randn(4, 3, 81)
        coeff = chart_coefficients(positive, negative)
        self.assertTrue(bool((coeff >= 0).all()))
        self.assertTrue(bool((coeff <= 1).all()))
        torch.testing.assert_close(coeff.sum(-1), torch.ones(4, 3, dtype=torch.float64),
                                   rtol=0, atol=2e-16)
        torch.testing.assert_close(chart_coefficients(positive + 7, negative + 7),
                                   coeff, rtol=0, atol=2e-15)

    def test_tied_signs_return_tangent_with_finite_zero_logit_gradient(self):
        positive = self.randn(3, 5).requires_grad_()
        negative = positive.detach().clone().requires_grad_()
        current, tangent = self.randn(3, 7), self.randn(3, 7)
        angles_p, angles_m = self.randn(5, 7), self.randn(5, 7)
        mixed = confidence_chart_mix(positive, negative, angles_p, angles_m, current, tangent)
        self.assertTrue(torch.equal(mixed, tangent))
        grads = torch.autograd.grad(mixed.square().sum(), (positive, negative))
        for grad in grads:
            self.assertTrue(bool(torch.isfinite(grad).all()))
            self.assertEqual(int(torch.count_nonzero(grad)), 0)

    def test_dominant_signs_select_the_corresponding_conditional_mean(self):
        angles_p = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
        angles_m = -angles_p
        zero = torch.zeros(2, dtype=torch.float64)
        dominant = torch.full((2,), 1000., dtype=torch.float64)
        absent = -dominant
        for lp, lm, expected in ((dominant, absent, angles_p.mean(0)),
                                 (absent, dominant, angles_m.mean(0))):
            mixed = confidence_chart_mix(lp, lm, angles_p, angles_m, zero, zero)
            self.assertTrue(torch.equal(mixed, expected))

    def test_first_derivative_is_continuous_at_sign_boundary(self):
        angles_p = torch.tensor([[1., 2.]], dtype=torch.float64)
        angles_m = -angles_p
        zero = torch.zeros(2, dtype=torch.float64)
        derivatives = []
        for value in (-1e-4, -1e-6, 0., 1e-6, 1e-4):
            delta = torch.tensor(value, dtype=torch.float64, requires_grad=True)
            mixed = confidence_chart_mix(delta.reshape(1) / 2, -delta.reshape(1) / 2,
                                          angles_p, angles_m, zero, zero)
            derivative, = torch.autograd.grad(mixed.sum(), delta)
            derivatives.append(float(derivative))
        self.assertEqual(derivatives[2], 0.)
        self.assertLess(abs(derivatives[1]), 2e-6)
        self.assertLess(abs(derivatives[3]), 2e-6)
        self.assertLess(abs(derivatives[1]), abs(derivatives[0]) / 50)
        self.assertLess(abs(derivatives[3]), abs(derivatives[4]) / 50)

    def test_helper_passes_numerical_gradient_check(self):
        positive = self.randn(2, 3).requires_grad_()
        negative = (self.randn(2, 3) - 1.).requires_grad_()
        angles_p, angles_m = self.randn(3, 4), self.randn(3, 4)
        current, tangent = self.randn(2, 4), self.randn(2, 4)
        def function(p, m):
            return confidence_chart_mix(p, m, angles_p, angles_m, current, tangent)
        self.assertTrue(torch.autograd.gradcheck(function, (positive, negative),
                                                eps=1e-6, atol=1e-5, rtol=1e-4))


class ResidualIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        generator = torch.Generator().manual_seed(261120)
        b, r, t, n = 2, 1, 20, 32
        self.clean = torch.randn(b, r, t, n, generator=generator, dtype=torch.complex128)
        self.x = torch.randn(b, t, n, generator=generator, dtype=torch.complex128)
        self.field = .1 * torch.randn(b, r, t, n, generator=generator, dtype=torch.complex128)
        self.basis = _dictionary(2, n)[0]
        rotations = _ray_bank(2, t, n)[0]
        choices = torch.tensor([[0, 30, 100], [50, 80, 150]])
        self.rot = rotations[choices].permute(0, 2, 3, 1)

    def arguments(self, step=2, information=None):
        if information is None:
            information = torch.tensor([.001842, .664825], dtype=torch.float64)
        return (self.clean, self.x, self.basis, self.rot, self.field, .05,
                step, information)

    def test_first_pass_is_identical_to_frozen_bank(self):
        args = self.arguments(step=1)
        self.assertTrue(torch.equal(residual(*args), math_chirp_trust.residual(*args)))

    def test_full_information_is_identical_to_frozen_trust(self):
        args = self.arguments(information=torch.ones(2, dtype=torch.float64))
        self.assertTrue(torch.equal(residual(*args), math_chirp_trust.residual(*args)))

    def test_zero_taps_and_cue_return_finite_zero_local_innovation(self):
        self.clean = torch.zeros_like(self.clean)
        self.field = torch.zeros_like(self.field)
        result = residual(*self.arguments(information=torch.zeros(2, dtype=torch.float64)))
        self.assertTrue(bool(torch.isfinite(result).all()))
        self.assertEqual(int(torch.count_nonzero(result)), 0)

    def test_common_phase_rotation_preserves_the_residual(self):
        args = self.arguments()
        expected = residual(*args)
        phase = torch.exp(1j * torch.tensor([.731, -2.17], dtype=torch.float64))[:, None, None, None]
        rotated = (self.clean * phase, self.x, self.basis, self.rot,
                   self.field * phase, .05, 2, args[-1])
        torch.testing.assert_close(residual(*rotated), expected, rtol=0, atol=2e-12)

    def test_final_pass_rejects_missing_information(self):
        args = list(self.arguments())
        args[-1] = None
        with self.assertRaisesRegex(ValueError, "third-pass pilot information"):
            residual(*args)


if __name__ == "__main__":
    unittest.main()
