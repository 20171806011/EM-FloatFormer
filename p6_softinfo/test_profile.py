"""Independent algebraic and fixed-q objective checks for profile.py."""
import math
import unittest

import torch

from p6_softinfo.profile import (gaussian_log_evidence, mix_bit_llrs,
                                 phase_derivatives, profile_amplitudes,
                                 profile_refine, _phase_solution)
from p6.math_chirp import _chirp_basis


def complex_random(shape, generator):
    return torch.complex(torch.randn(shape, generator=generator, dtype=torch.float64),
                         torch.randn(shape, generator=generator, dtype=torch.float64))


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.g = torch.Generator().manual_seed(602611193170)

    def fixture(self, batch=2, t=6, n=8, paths=3):
        mean = complex_random((batch, t, n), self.g) / math.sqrt(2.)
        variance = .08 * torch.rand((batch, t, n), generator=self.g, dtype=torch.float64)
        delays = torch.tensor([0., .7, 2.3][:paths], dtype=torch.float64)
        basis = torch.exp(-2j * math.pi * delays[:, None] * torch.fft.fftfreq(n)).expand(batch, -1, -1)
        rot = torch.exp(1j * .2 * torch.randn((batch, t, n, paths), generator=self.g, dtype=torch.float64))
        cpe = torch.exp(1j * .1 * torch.randn((batch, t), generator=self.g, dtype=torch.float64))
        rx = complex_random((batch, t*n, 1), self.g)
        ridge = .07 * torch.eye(paths, dtype=torch.complex128).expand(batch, -1, -1)
        return mean, variance, basis, rot, cpe, rx, ridge

    def test_profile_matches_augmented_qr(self):
        b, length, paths, receivers = 2, 11, 3, 2
        design = complex_random((b, length, paths), self.g)
        rx = complex_random((b, length, receivers), self.g)
        root = complex_random((b, paths, paths), self.g)
        covariance = .1 * (root @ root.mH)
        ridge = .3 * torch.eye(paths, dtype=design.dtype).expand(b, -1, -1)
        actual = profile_amplitudes(design, rx, covariance, ridge)
        penalty_root = torch.linalg.cholesky(covariance + ridge)
        augmented = torch.cat((design, penalty_root.mH), dim=1)
        target = torch.cat((rx, rx.new_zeros(b, paths, receivers)), dim=1)
        expected = torch.linalg.lstsq(augmented, target, driver="gels").solution
        torch.testing.assert_close(actual.amplitudes, expected, rtol=2e-13, atol=2e-13)
        expected_objective = (target - augmented @ expected).abs().square().sum((-2, -1))
        torch.testing.assert_close(actual.objective, expected_objective, rtol=2e-13, atol=2e-13)
        self.assertFalse(bool(actual.used_pinv.any()))

    def test_profile_singular_zero_has_valid_minimum_norm(self):
        design = torch.zeros(2, 4, 3, dtype=torch.complex128)
        rx = complex_random((2, 4, 1), self.g)
        zero = design.new_zeros(2, 3, 3)
        answer = profile_amplitudes(design, rx, zero, zero)
        self.assertTrue(bool(answer.used_pinv.all()))
        self.assertEqual(float(answer.amplitudes.abs().max()), 0.)
        torch.testing.assert_close(answer.objective, rx.abs().square().sum((-2, -1)))

    def test_gaussian_evidence_matches_dense_covariance(self):
        design = complex_random((2, 7, 3), self.g)
        rx = complex_random((2, 7, 2), self.g)
        root = complex_random((2, 3, 3), self.g)
        prior = root @ root.mH + .4 * torch.eye(3, dtype=design.dtype)
        noise = torch.tensor([.2, 1.7], dtype=torch.float64)
        actual = gaussian_log_evidence(design, rx, noise, prior)
        covariance = design @ prior @ design.mH + noise[:, None, None] * torch.eye(7, dtype=design.dtype)
        quadratic = (rx.conj() * torch.linalg.solve(covariance, rx)).real.sum((-2, -1))
        expected = -(quadratic + 2 * torch.linalg.slogdet(covariance).logabsdet
                     + 14 * math.log(math.pi))
        torch.testing.assert_close(actual, expected, rtol=5e-13, atol=5e-13)

    def test_evidence_rejects_zero_noise(self):
        with self.assertRaises(ValueError):
            gaussian_log_evidence(torch.ones(1, 2, 1, dtype=torch.complex128),
                                  torch.ones(1, 2, 1, dtype=torch.complex128),
                                  torch.zeros(1), torch.ones(1, 1, 1, dtype=torch.complex128))

    def test_probability_mixture_is_not_llr_average(self):
        llrs = torch.tensor([[[8., -2.], [-1., 4.]]], dtype=torch.float64)
        weights = torch.tensor([[.2, .8]], dtype=torch.float64)
        actual = mix_bit_llrs(llrs, weights.log())
        probability = (weights[..., None] * llrs.sigmoid()).sum(1)
        expected = probability.log() - (1-probability).log()
        torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
        self.assertGreater(float((actual - (weights[..., None]*llrs).sum(1)).abs().max()), .1)

    def test_probability_mixture_extremes_and_log_offset(self):
        llrs = torch.tensor([[[1000., -1000.], [-1000., 1000.]]], dtype=torch.float64)
        weights = torch.tensor([[math.log(.25), math.log(.75)]], dtype=torch.float64)
        actual = mix_bit_llrs(llrs, weights)
        expected = torch.tensor([[math.log(1/3), math.log(3.)]], dtype=torch.float64)
        torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(mix_bit_llrs(llrs, weights + 1000), actual, atol=2e-13, rtol=2e-13)

    def test_probability_mixture_rejects_invalid_weights(self):
        llrs = torch.tensor([[[2.], [-3.]]], dtype=torch.float64)
        for bad in (math.nan, math.inf):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                mix_bit_llrs(llrs, torch.tensor([[0., bad]], dtype=torch.float64))
        with self.assertRaises(ValueError):
            mix_bit_llrs(llrs, torch.full((1, 2), -math.inf, dtype=torch.float64))
        actual = mix_bit_llrs(llrs, torch.tensor([[0., -math.inf]], dtype=torch.float64))
        torch.testing.assert_close(actual, llrs[:, 0], rtol=1e-14, atol=1e-14)

    def test_joint_profile_derivatives_match_finite_differences(self):
        inputs = self.fixture(batch=1)
        theta = torch.zeros(1, 3, 2, dtype=torch.float64)
        objective, gradient, hessian = phase_derivatives(theta, *inputs)
        finite_gradient = torch.zeros(6, dtype=torch.float64)
        finite_hessian = torch.zeros(6, 6, dtype=torch.float64)
        step = 2e-5
        for index in range(6):
            plus, minus = theta.clone(), theta.clone()
            plus.view(-1)[index] += step
            minus.view(-1)[index] -= step
            upper, gu, _ = phase_derivatives(plus, *inputs)
            lower, gl, _ = phase_derivatives(minus, *inputs)
            finite_gradient[index] = (upper-lower)/(2*step)
            finite_hessian[:, index] = ((gu-gl)/(2*step)).reshape(-1)
        torch.testing.assert_close(gradient.flatten(), finite_gradient, atol=2e-8, rtol=3e-7)
        torch.testing.assert_close(hessian[0], finite_hessian, atol=3e-8, rtol=3e-7)
        self.assertGreater(float(hessian[0, :2, 2:4].abs().max()), 1e-5)

    def test_refinement_reduces_actual_fixed_q_objective(self):
        inputs = self.fixture()
        new_rot, amplitude, audit = profile_refine(*inputs)
        before = torch.tensor(audit["objective_before"], dtype=torch.float64)
        after = torch.tensor(audit["objective_after"], dtype=torch.float64)
        self.assertTrue(bool((after <= before).all()))
        self.assertGreater(float((before-after).max()), 1e-6)
        self.assertTrue(all(value <= .5 + 1e-12 for value in audit["max_total_phase_change"]))
        self.assertEqual(amplitude.shape, (2, 3, 1))
        torch.testing.assert_close(new_rot.abs(), inputs[3].abs(), atol=3e-16, rtol=3e-16)

    def test_refinement_runs_inside_inference_mode(self):
        with torch.inference_mode():
            inputs = tuple(t.clone() for t in self.fixture(batch=1))
            rot, amplitude, audit = profile_refine(*inputs, max_steps=1)
        self.assertTrue(torch.isfinite(amplitude).all())
        self.assertLessEqual(audit["objective_after"][0], audit["objective_before"][0])

    def test_cold_cache_primed_under_inference_mode(self):
        inputs = self.fixture(batch=1)
        t = inputs[0].shape[1]
        _chirp_basis.cache_clear()
        try:
            with torch.inference_mode():
                cached = _chirp_basis(t)
                self.assertTrue(torch.is_inference(cached))
                rot, amplitude, audit = profile_refine(*inputs, max_steps=1)
            # Do not clear the cache: stand-alone derivatives must also cope
            # with the same inference tensor cached by the earlier receiver.
            theta = torch.zeros(1, 3, 2, dtype=torch.float64)
            value, gradient, hessian = phase_derivatives(theta, *inputs)
            self.assertTrue(torch.isfinite(value).all())
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertTrue(torch.isfinite(hessian).all())
            self.assertTrue(torch.isfinite(amplitude).all())
            self.assertLessEqual(audit["objective_after"][0], audit["objective_before"][0])
        finally:
            _chirp_basis.cache_clear()

    def test_refinement_gain_covariance(self):
        inputs = self.fixture(batch=1)
        rot1, amp1, audit1 = profile_refine(*inputs)
        gain = .4 + 1.3j
        scaled = list(inputs)
        scaled[-2] = gain * scaled[-2]
        rot2, amp2, audit2 = profile_refine(*scaled)
        torch.testing.assert_close(rot2, rot1, rtol=3e-11, atol=3e-11)
        torch.testing.assert_close(amp2, gain*amp1, rtol=3e-11, atol=3e-11)
        expected = (abs(gain)**2) * torch.tensor(audit1["objective_after"])
        torch.testing.assert_close(torch.tensor(audit2["objective_after"]), expected, rtol=1e-6, atol=1e-6)

    def test_refinement_skips_singular_bursts(self):
        inputs = list(self.fixture())
        inputs[0].zero_()
        inputs[1].zero_()
        inputs[-1].zero_()
        rot, amplitude, audit = profile_refine(*inputs)
        self.assertTrue(all(audit["used_pinv"]))
        self.assertEqual(audit["accepted_steps"], [0, 0])
        self.assertEqual(float(amplitude.abs().max()), 0.)
        self.assertTrue(torch.equal(rot, inputs[3]))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
