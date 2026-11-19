"""Independent serial-waveform, inference-boundary and known-channel checks."""
import unittest

import numpy as np

from p8 import coding, link as ofdm, tracker
from p8.study import history
from p8c import link, receiver


def serial_reference(useful, channel):
    """Insert every CP, delay the linear stream, rotate, then discard every CP."""
    stream = np.column_stack((useful[:, -ofdm.CP:], useful)).reshape(-1)
    clock = np.arange(stream.size) / ofdm.N
    received = np.zeros_like(stream)
    for delay, amplitude, rate, chirp in zip(ofdm.DELAYS, channel.gain,
                                              channel.frequency, channel.chirp):
        delayed = np.zeros_like(stream)
        delayed[delay:] = stream[:len(stream)-delay]
        received += amplitude * np.exp(2j*np.pi*(rate*clock + chirp*clock**2 / 2)) * delayed
    return received.reshape(len(useful), -1)[:, ofdm.CP:]


class CrossWaveformTests(unittest.TestCase):
    def test_mixed_waveform_pair_matches_serial_oracle_across_boundary(self):
        seed = 3117
        pair = link.simulate(seed)
        channel = ofdm.draw_channel(ofdm.stream(seed, 0))
        a = ofdm.tx_grid(ofdm.traffic(ofdm.stream(seed, 1)), ofdm.PILOT_ROWS)
        a_time = np.fft.ifft(a, norm="ortho")
        for layout, rows in link.LAYOUTS.items():
            b = link.tx_grid(pair.b_words, rows)
            reference = serial_reference(np.vstack((a_time, b)), channel)
            a_noiseless = pair.a - ofdm.noise(ofdm.stream(seed, 2))
            np.testing.assert_allclose(np.fft.fft(reference[:ofdm.T], norm="ortho"),
                                       a_noiseless, atol=3e-14, rtol=0.)
            np.testing.assert_allclose(reference[ofdm.T:], pair.clean_b[layout], atol=3e-14, rtol=0.)
            # Re-starting B's clock changes the output, despite identical input and taps.
            self.assertGreater(np.max(abs(reference[ofdm.T:]-serial_reference(b, channel))), .1)

    def test_physical_pilots_and_common_codewords(self):
        pair = link.simulate(6017)
        dense = link.tx_grid(pair.b_words, link.LAYOUTS["dense"])
        sparse = link.tx_grid(pair.b_words, link.LAYOUTS["sparse"])
        common = list(link.CORE_ROWS)
        self.assertEqual(common, [r for r in range(20) if r not in (0, 9, 10, 19)])
        # Each core block carries one codeword, in direct time-domain symbol order.
        code = pair.b_words.code[:16]
        expected = coding.qam(code).reshape(16, 32)
        np.testing.assert_array_equal(dense[common], expected)
        np.testing.assert_array_equal(sparse[common], expected)
        self.assertFalse(np.any((pair.b_words.code[:18] @ coding.parity_check().T) % 2))
        np.testing.assert_array_equal(pair.b_words.code[:18, :64], pair.b_words.info[:18])
        np.testing.assert_array_equal(sparse[[0, 9]], coding.qam(pair.b_words.code[16:18]).reshape(2, 32))
        for layout, waveform, total in (("dense", dense, 128), ("sparse", sparse, 64)):
            pilots = list(link.LAYOUTS[layout]) + [19]
            self.assertEqual(len(pilots)*ofdm.N, total)
            np.testing.assert_array_equal(waveform[pilots], link.PILOTS[pilots])
            np.testing.assert_allclose(abs(waveform[pilots]), 1., atol=5e-15, rtol=0)
        np.testing.assert_array_equal(dense[[10, 19]], sparse[[10, 19]])

    def test_time_domain_noise_paired_and_reproducible(self):
        seed = 2112
        pair = link.simulate(seed)
        noise = np.sqrt(ofdm.N0) * ofdm.complex_normal(ofdm.stream(seed, 4), (ofdm.T, ofdm.N))
        for layout in link.LAYOUTS:
            np.testing.assert_allclose(pair.b[layout] - pair.clean_b[layout], noise, rtol=0., atol=5e-16)
        np.testing.assert_array_equal(pair.b["dense"][19], pair.b["sparse"][19])
        np.testing.assert_array_equal(pair.clean_b["dense"][19], pair.clean_b["sparse"][19])
        donor = ofdm.draw_channel(ofdm.stream(seed, 5))
        donor_grid = ofdm.tx_grid(ofdm.traffic(ofdm.stream(seed, 6)), ofdm.PILOT_ROWS)
        expected = ofdm.propagate(donor_grid, donor) + ofdm.noise(ofdm.stream(seed, 7))
        np.testing.assert_array_equal(pair.donor_a, expected)
        self.assertFalse(np.array_equal(pair.a, pair.donor_a))

    def test_payload_and_probe_isolation_in_all_layouts(self):
        pair = link.simulate(8035)
        state = history(pair.a)
        for layout, rows in link.LAYOUTS.items():
            y = pair.b[layout]
            reference = receiver.receive(state, y, rows)
            blind = y.copy()
            blind[19] = complex(np.nan, np.nan)
            poisoned = receiver.receive(state, blind, rows)
            for expected, got in zip(reference, poisoned):
                self.assertTrue(np.isfinite(got).all())
                np.testing.assert_array_equal(expected, got)
            self.assertTrue((poisoned[2] >= ofdm.N0-1e-12).all())
            blind[list(link.CORE_ROWS)] = 40-60j
            other = receiver.receive(state, blind, rows)
            np.testing.assert_array_equal(other[1], reference[1])
            np.testing.assert_array_equal(other[2], reference[2])
            self.assertFalse(np.array_equal(other[0], reference[0]))

    def test_oracle_predicts_noiseless_full_heldout_block(self):
        seed = 2204
        pair = link.simulate(seed)
        channel = ofdm.draw_channel(ofdm.stream(seed, 0))
        for layout, rows in link.LAYOUTS.items():
            blind = pair.b[layout].copy()
            blind[19] = np.nan
            _, prediction, variance = receiver.oracle(channel, blind, rows)
            np.testing.assert_allclose(prediction, pair.clean_b[layout][19], rtol=0., atol=4e-14)
            np.testing.assert_allclose(variance, ofdm.N0, rtol=0., atol=2e-14)

    def test_noiseless_unit_channel_demapper_and_code(self):
        pair = link.simulate(1593)
        channel = ofdm.Channel(np.array([1.+0j, 0j, 0j]), np.zeros(3), np.zeros(3))
        for layout, rows in link.LAYOUTS.items():
            observed = link.tx_grid(pair.b_words, rows).copy()
            observed[19] = np.nan
            llrs, _, _ = receiver.oracle(channel, observed, rows)
            word_rows = list(link.CORE_ROWS) + sorted({0, 9, 10} - set(rows))
            bits = llrs[word_rows].reshape(-1, 128)
            size = len(word_rows)
            np.testing.assert_array_equal(bits > 0, pair.b_words.code[:size])
            np.testing.assert_array_equal(coding.decode(bits), pair.b_words.info[:size])

    def test_transported_point_state_matches_absolute_physical_time(self):
        nodes = np.array([3, 51, 98])
        gain = np.array([.73+.24j, -.31+.51j, .21-.16j])
        freq, chirp = tracker.FREQ[nodes], tracker.CHIRP[nodes]
        state = tracker.prior()
        log_weights = np.full_like(state.log_weight, -np.inf)
        log_weights[np.arange(3), nodes] = 0.
        means = np.zeros_like(state.mean)
        means[np.arange(3), nodes] = gain
        at_a = tracker.State(0., log_weights, means, np.zeros_like(state.variance))
        at_b = tracker.advance(at_a, ofdm.FRAME_DURATION)
        ray_mean, ray_var = receiver.sample_moments(at_b)
        sample = np.arange(2*20*(32+16)).reshape(40, 48)[20:, 16:]/32
        reference = gain[:, None, None] * np.exp(2j*np.pi*(freq[:, None, None]*sample
                                              + chirp[:, None, None]*sample**2 / 2))
        np.testing.assert_allclose(ray_mean, reference, rtol=0., atol=2e-14)
        np.testing.assert_allclose(ray_var, 0., rtol=0., atol=2e-14)


if __name__ == "__main__":
    unittest.main()
