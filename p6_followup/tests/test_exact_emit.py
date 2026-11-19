"""Numerical/Jacobian checks for the optional R2 FFT identity reduction."""

import unittest
from unittest.mock import patch

import torch

from p6 import axial_em_transformer as solver
from p6_followup.exact_emit import emit, use_exact_r2_emit


FROZEN_EMIT = solver._emit


class ExactEmissionTests(unittest.TestCase):
    def inputs(self, n=32, amplitude=1., seed=6201):
        generator = torch.Generator().manual_seed(seed)
        x = amplitude * torch.randn((2, 3, n), dtype=torch.complex128,
                                    generator=generator)
        radial = .1 * torch.randn((2, 3, 2), dtype=torch.complex128,
                                 generator=generator)
        return x, radial

    def test_r2_values_match_frozen_roundtrip(self):
        # Odd/even bin counts exercise both halves of the embedding, while
        # the actual P6 protocol uses N=32 and CPU complex128 throughout.
        for n in (31, 32):
            for amplitude in (0., 1.e-8, 1., 1.e6):
                with self.subTest(n=n, amplitude=amplitude):
                    x, radial = self.inputs(n, amplitude)
                    frozen = FROZEN_EMIT(x, 2, radial)
                    optimized = emit(x, 2, radial)
                    self.assertIs(optimized, x)
                    torch.testing.assert_close(
                        optimized, frozen, rtol=5.e-13,
                        atol=5.e-15 * max(amplitude, 1.e-8),
                    )

    def test_r2_jacobian_is_identity_and_radial_stays_unused(self):
        generator = torch.Generator().manual_seed(6202)
        for amplitude in (0., 1.):
            with self.subTest(amplitude=amplitude):
                x, radial = self.inputs(amplitude=amplitude)
                x.requires_grad_()
                radial.requires_grad_()
                upstream = torch.randn(x.shape, dtype=x.dtype, generator=generator)
                frozen_dx, frozen_dr = torch.autograd.grad(
                    FROZEN_EMIT(x, 2, radial), (x, radial),
                    grad_outputs=upstream, allow_unused=True,
                )
                fast_dx, fast_dr = torch.autograd.grad(
                    emit(x, 2, radial), (x, radial),
                    grad_outputs=upstream, allow_unused=True,
                )
                torch.testing.assert_close(fast_dx, upstream, rtol=0., atol=0.)
                torch.testing.assert_close(fast_dx, frozen_dx, rtol=5.e-13, atol=5.e-14)
                self.assertIsNone(frozen_dr)
                self.assertIsNone(fast_dr)

    def test_r2_does_not_call_oversampled_transforms(self):
        x, radial = self.inputs()
        with (patch.object(solver, "oversample_frequency_to_time",
                           side_effect=AssertionError("unnecessary IFFT")),
              patch.object(solver, "decimate_time_to_frequency",
                           side_effect=AssertionError("unnecessary FFT"))):
            self.assertIs(emit(x, 2, radial), x)

    def test_r1_r3_values_and_gradients_are_unchanged(self):
        generator = torch.Generator().manual_seed(6203)
        for regime in (1, 3):
            with self.subTest(regime=regime):
                x, radial = self.inputs()
                x.requires_grad_()
                radial.requires_grad_()
                upstream = torch.randn(x.shape, dtype=x.dtype, generator=generator)
                frozen = FROZEN_EMIT(x, regime, radial)
                optimized = emit(x, regime, radial)
                frozen_grad = torch.autograd.grad(frozen, (x, radial),
                                                 grad_outputs=upstream)
                optimized_grad = torch.autograd.grad(optimized, (x, radial),
                                                    grad_outputs=upstream)
                torch.testing.assert_close(optimized, frozen, rtol=0., atol=0.)
                for actual, expected in zip(optimized_grad, frozen_grad):
                    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)

    def test_context_restores_previous_binding_on_exception_and_nesting(self):
        previous = solver._emit
        with use_exact_r2_emit():
            self.assertIs(solver._emit, emit)
            with use_exact_r2_emit():
                self.assertIs(solver._emit, emit)
            self.assertIs(solver._emit, emit)
        self.assertIs(solver._emit, previous)
        with self.assertRaisesRegex(RuntimeError, "test failure"):
            with use_exact_r2_emit():
                raise RuntimeError("test failure")
        self.assertIs(solver._emit, previous)
        with use_exact_r2_emit(enabled=False):
            self.assertIs(solver._emit, previous)


if __name__ == "__main__":
    unittest.main()
