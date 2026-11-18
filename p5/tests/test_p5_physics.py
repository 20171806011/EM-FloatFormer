"""Independent overlap and limiting cases for the P5 waveform generator."""
import dataclasses
import math
import unittest

import torch

from p4.simulator import complex_randn, hard_bits_16qam, q16, rapp_pa
from p4.slot_2d import _apply_continuous_slot_channel, _get_default_ldpc, decode_min_sum
from p5.wavefield_sim import (
    COMPLEX, CP, N, REAL, green_kernel, pa_grid, propagate, simulate, steer,
)


class WavefieldPhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def test_rapp_saleh_amplitude_and_phase_match_p4(self):
        rng = torch.Generator().manual_seed(54)
        x = complex_randn((3, 5, N), generator=rng, dtype=COMPLEX)
        for a, p, alpha in [(1.05, 1.3, -.3), (1.48, 1.6, .38), (2.5, 2.7, .9)]:
            ref = rapp_pa(torch.fft.ifft(x, norm="ortho"), A=a, knee=p, alpha=alpha, oversample=2)
            actual = pa_grid(x, a, p, alpha)
            torch.testing.assert_close(actual, torch.fft.fft(ref, norm="ortho"), atol=5e-15, rtol=0)

    def test_static_and_dynamic_integer_paths_against_continuous_p4_stream(self):
        rng = torch.Generator().manual_seed(54)
        x = complex_randn((3, 5, N), generator=rng, dtype=COMPLEX)
        d = torch.tensor([0, 2, 4])
        gains = complex_randn((3, 3), generator=rng, dtype=COMPLEX)
        x_time = torch.fft.ifft(x, norm="ortho")
        stream = torch.cat((x_time[:, :, -CP:], x_time), dim=-1).flatten(1)
        for shifts in [(0., 0., 0.), (-.074, .048, .06)]:
            nu = torch.tensor(shifts, dtype=REAL).expand(3, 3)
            actual, diag, _ = propagate(x, d.to(REAL).expand(3, 3), gains, torch.zeros_like(nu), nu, nu, 1)
            ref, _, _ = _apply_continuous_slot_channel(stream, gains, d, nu, 5, N, CP)
            # The P4 public interface casts its precise computation to complex64.
            torch.testing.assert_close(actual[:, 0], ref.to(COMPLEX).reshape(3, 5, N+CP)[:, :, CP:],
                                       atol=3e-7, rtol=1e-7)
            if shifts[0] == 0:
                torch.testing.assert_close(torch.fft.fft(actual[:, 0], norm="ortho"), diag[:, 0]*x)

    def test_fractional_delay_steering_and_exact_diagonal(self):
        # Signed subcarriers straddle FFT wrap. A chirping ray mixes bins even without noise.
        x = torch.zeros(1, 5, N, dtype=COMPLEX)
        delay = torch.tensor([[2.36]], dtype=REAL)
        theta = torch.tensor([[.52]], dtype=REAL)
        start = torch.tensor([[.047]], dtype=REAL)
        end = torch.tensor([[.078]], dtype=REAL)
        one = torch.ones(1, 1, dtype=COMPLEX)
        torch.testing.assert_close(steer(theta, 4).abs().square().sum(-1), one.real)
        for k in [3, N-4]:
            x.zero_()
            x[:, :, k] = 1.
            signal, diag, _ = propagate(x, delay, one, theta, start, end, 4)
            spectrum = torch.fft.fft(signal, norm="ortho")
            torch.testing.assert_close(spectrum[..., k], diag[..., k])
            self.assertGreater(float(spectrum[..., (k+1) % N].abs().min()), .01)
            ratio = spectrum[:, 1:, :, k] / spectrum[:, :-1, :, k]
            expected_ratio = torch.exp(1j * math.pi * theta.sin()).expand_as(ratio)
            torch.testing.assert_close(ratio, expected_ratio)
            # Compare the first retained time sample directly, independent of FFT interpolation.
            physical_bin = k if k < N//2 else k-N
            clock = CP / N
            duration = (5*(N+CP)-1) / N
            cycles = start*clock + .5*(end-start)*clock**2/duration - physical_bin*delay/N
            expected = torch.exp(2j*math.pi*cycles) / math.sqrt(N)
            torch.testing.assert_close(signal[:, 0, 0, 0], expected.flatten(), atol=1e-13, rtol=1e-13)

    def test_r2_continuous_chirp_above_within_slot_nyquist(self):
        frame = simulate(2, 2, math.inf, 96001)
        a, b = frame.truth.nu_start, frame.truth.nu_end
        self.assertTrue(bool((a*b > 0).all()))
        self.assertTrue(bool((torch.stack([a, b]).abs() >= .045).all()))
        self.assertTrue(bool((torch.stack([a, b]).abs() <= .080).all()))
        delta = 9 * (N + CP) / N
        nyquist = .5 / delta
        self.assertAlmostEqual(nyquist, 1/27.)
        nu = .061
        observed = float(torch.angle(torch.exp(torch.tensor(2j*math.pi*nu*delta, dtype=COMPLEX))))
        self.assertAlmostEqual(observed / (2*math.pi*delta), nu-1/delta)
        # Independently integrate instantaneous frequency over the useful samples
        # on both sides of the slot boundary, including phase accrued in each CP.
        time = (torch.arange(20, dtype=REAL)[:, None]*(N+CP) + CP + torch.arange(N, dtype=REAL)) / N
        duration = (20*(N+CP)-1) / N
        midpoint_frequency = a[:, :, None, None] + .5*(b-a)[:, :, None, None]*time / duration
        average_phasor = torch.exp(2j*math.pi*midpoint_frequency*time).mean(-1).transpose(1, 2)
        torch.testing.assert_close(frame.truth.ray_evolution, average_phasor)

    def test_blind_observable_fields_and_wiener_ici(self):
        f = simulate(1, 4, math.inf, 96002)
        self.assertEqual({d.name for d in dataclasses.fields(f.obs)},
                         {"y", "pilot_mask", "pilots", "noise_var", "regime"})
        self.assertTrue(bool((f.truth.pa.A[:, 0] != f.truth.pa.A[:, -1]).all()))
        self.assertTrue(bool(((f.truth.pa.A >= 1.05) & (f.truth.pa.A <= 2.5)).all()))
        # With a one-tone input, a changing oscillator produces both a rotated
        # diagonal and off-tone energy. Parseval determines their exact sum.
        phase = torch.exp(1j*f.truth.phn)
        spectrum = torch.fft.fft(phase, norm="forward")
        torch.testing.assert_close(spectrum[..., 0], phase.mean(-1))
        self.assertGreater(float(torch.angle(spectrum[..., 0]).abs().max()), .01)
        self.assertGreater(float(spectrum[..., 1:].abs().square().sum(-1).min()), 0.)
        torch.testing.assert_close(spectrum.abs().square().sum(-1), torch.ones_like(phase.real[..., 0]))
        # Rebuild received samples with P4 continuous, static, integer-delay paths.
        grid = f.obs.pilots.expand(4, -1, -1).clone()
        grid[:, ~f.obs.pilot_mask] = q16(f.bits).to(COMPLEX)
        params = f.truth.pa
        x = torch.fft.ifft(pa_grid(grid, params.A, params.knee, params.alpha), norm="ortho")
        stream = torch.cat([x[:, :, -CP:], x], -1).flatten(1)
        ref, _, _ = _apply_continuous_slot_channel(stream, f.truth.gains, f.truth.delays[0].long(),
                                                   f.truth.nu_start, 5, N, CP)
        received = ref.to(COMPLEX).reshape(4, 5, N+CP)[:, :, CP:] * phase
        torch.testing.assert_close(f.obs.y[:, 0], torch.fft.fft(received, norm="ortho"), atol=3e-7, rtol=1e-7)

    def test_ldpc_p4_word_order_and_perfect_soft_decode(self):
        for regime, nwords in ((1, 4), (2, 18), (3, 4)):
            frame = simulate(regime, 2, 12., 96003)
            self.assertEqual(frame.bits.shape, (2, nwords*32, 4))
            matrix = _get_default_ldpc()
            code = hard_bits_16qam(q16(frame.bits)).reshape(2, nwords, 128)
            torch.testing.assert_close(code[..., :64], frame.info_bits)
            self.assertEqual(int((code @ matrix.h_matrix.T % 2).sum()), 0)
            decoded = decode_min_sum(12*(2*code-1), matrix.h_matrix)
            torch.testing.assert_close(decoded, frame.info_bits)

    def test_green_target_matches_phase_ensemble(self):
        frame = simulate(3, 2, 18., 96004)
        coords = torch.tensor([[0, 0, 0], [1, 4, 29], [3, 1, 3], [2, 2, 19]])
        ray_id = torch.arange(4, dtype=REAL)
        basis = torch.exp(2j*math.pi*ray_id[:, None]*ray_id/4)
        # The orthogonal phase ensemble cancels all cross terms exactly.
        ensemble = []
        for phase in basis:
            truth = frame.truth
            _, h, _ = propagate(torch.ones(2, 5, N, dtype=COMPLEX), truth.delays, truth.gains*phase,
                                truth.angles, truth.nu_start, truth.nu_end, 4)
            ensemble.append(h[:, coords[:, 0], coords[:, 1], coords[:, 2]])
        fields = torch.stack(ensemble, -1)
        cov = (fields @ fields.mH) / 4
        torch.testing.assert_close(green_kernel(frame, coords), cov.abs(), atol=1e-13, rtol=1e-13)


if __name__ == "__main__":
    unittest.main()
