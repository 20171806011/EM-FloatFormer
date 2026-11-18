import unittest

import torch

from p5.unrolled_em_transformer import _equalize
from p6.math_spatial import apply_in_ray_frame, innovation_equalize, precision_correction


class PilotInnovationEqualizer(unittest.TestCase):
    def test_ula_leaves_a_single_steered_wave_invariant(self):
        sensor = torch.arange(4, dtype=torch.float64)
        rays = (1j * sensor[:, None] * torch.tensor([[.43, -.85]], dtype=torch.float64)).exp()[None]
        contraction = ((torch.ones(4, 4)/4 + torch.eye(4))/2).to(torch.complex128)
        strength = torch.tensor([[.04, .1]], dtype=torch.float64)
        filtered = apply_in_ray_frame(rays, (contraction, strength), pilot_count=32)
        torch.testing.assert_close(filtered, rays, atol=2e-15, rtol=0)
        noisy = rays.clone()
        noisy[:, 2] += .4j
        suppressed = apply_in_ray_frame(noisy, (contraction, strength*0), pilot_count=32)
        self.assertTrue(torch.equal(suppressed, noisy))
        reference = apply_in_ray_frame(noisy, (contraction, strength), pilot_count=32)
        scaled = apply_in_ray_frame(noisy*(.6-.8j), (contraction, strength), pilot_count=32)
        torch.testing.assert_close(scaled, reference*(.6-.8j), atol=2e-15, rtol=0)

    def test_covariance_floor_direct_gls_and_gauge(self):
        gen = torch.Generator().manual_seed(901)
        shape = (2, 4, 5, 32)
        y, h = [torch.randn(shape, generator=gen, dtype=torch.complex128) for _ in range(2)]
        h[0, :, 2, 3] = 0.
        noise = torch.tensor([.12, .31], dtype=torch.float64)
        mask = torch.zeros(5, 32, dtype=torch.bool)
        mask[(0, -1), ::2] = True
        pilots = torch.ones(5, 32, dtype=h.dtype)
        raw = torch.randn(2, 4, 4, generator=gen, dtype=h.dtype)
        psd = raw @ raw.mH
        transport = psd / torch.linalg.eigvalsh(psd).amax(-1)[:, None, None]
        strength = torch.full((2, 4), .09, dtype=torch.float64)
        params = (mask, pilots, transport, strength)
        correction, gate = precision_correction(y, h, noise, *params)
        covariance = torch.eye(4, dtype=h.dtype) + correction
        self.assertTrue(torch.equal(covariance, covariance.mH))
        self.assertTrue((torch.linalg.eigvalsh(covariance) >= (1 - gate[:, None]) - 1e-14).all())

        with torch.no_grad():
            z, var, active = innovation_equalize(y, h, noise, *params)
            inv = torch.linalg.inv(covariance * noise[:, None, None])
            precision_h = (inv @ h.flatten(2)).reshape_as(h)
            den = (h.conj() * precision_h).sum(1).real
            self.assertTrue(torch.equal(active, den > 0))
            target = (precision_h.conj() * y).sum(1)
            torch.testing.assert_close(z[active], target[active] / den[active], atol=1e-14, rtol=1e-13)
            torch.testing.assert_close(var[active], den[active].reciprocal(), atol=1e-14, rtol=1e-13)
            gain = 1e-25 * (.6 - .8j)
            gz, gv, gi = innovation_equalize(gain*y, gain*h, abs(gain)**2 * noise, *params)
            self.assertTrue(torch.equal(active, gi))
            torch.testing.assert_close(gz, z, atol=1e-14, rtol=1e-13)
            torch.testing.assert_close(gv[active], var[active], atol=1e-14, rtol=1e-13)
            zero = innovation_equalize(y, h, noise, mask, pilots, transport, torch.zeros_like(strength))
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(zero, _equalize(y,h,noise), strict=True)))
