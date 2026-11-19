import hashlib
import unittest

import numpy as np

from p8 import link, tracker
from p8.coding import decode, generator, parity_check, qam, qam_llrs


class PairTests(unittest.TestCase):
    def test_native_code_and_llr_sign(self):
        h = parity_check()
        self.assertEqual(hashlib.sha256(h.tobytes()).hexdigest(),
                         "5c8a40b0e9046bae409f7d3446a11ed1ed948de37347be90e4dc325177506e72")
        self.assertFalse(np.any((generator() @ h.T) % 2))
        words = link.traffic(link.stream(50, 1))
        symbols = qam(words.code)
        llrs = qam_llrs(symbols * .99, np.full(len(symbols), .01))
        np.testing.assert_array_equal(llrs.reshape(-1, 128) > 0, words.code)
        llrs.reshape(-1, 128)[:, 5] *= -.1
        np.testing.assert_array_equal(decode(llrs), words.info)
        np.testing.assert_array_equal(qam_llrs(np.zeros(3, complex), np.ones(3)), 0.)

    def test_serial_cp_oracle_and_frame_boundary(self):
        rng = link.stream(115, 4)
        channel = link.draw_channel(rng)
        a = link.tx_grid(link.traffic(rng), link.PILOT_ROWS)
        b = link.tx_grid(link.traffic(rng), (*link.B_ROWS["sparse"], link.PROBE_ROW))
        joined = np.vstack((a, b))
        useful = np.fft.ifft(joined, norm="ortho")
        serial = np.column_stack((useful[:, -link.CP:], useful)).ravel()
        time = np.arange(len(serial)) / link.N
        rx = np.zeros_like(serial)
        for d, g, f, c in zip(link.DELAYS, channel.gain, channel.frequency, channel.chirp):
            delayed = np.pad(serial, (d, 0))[:len(serial)]
            rx += g * delayed * np.exp(2j*np.pi*(f*time + .5*c*time**2))
        reference = np.fft.fft(rx.reshape(-1, link.CP + link.N)[:, link.CP:], norm="ortho")
        split = np.vstack((link.propagate(a, channel), link.propagate(b, channel, link.T)))
        np.testing.assert_allclose(split, reference, rtol=0, atol=3e-14)
        self.assertGreater(np.max(np.abs(link.propagate(b, channel)-split[link.T:])), .1)

    def test_pilot_budget_releases_exactly_one_codeword(self):
        words = link.traffic(link.stream(782, 1))
        dense = link.tx_grid(words, (*link.B_ROWS["dense"], link.PROBE_ROW))
        sparse = link.tx_grid(words, (*link.B_ROWS["sparse"], link.PROBE_ROW))
        released = link.row_mask((9, 10))
        self.assertEqual(int(released.sum()), 32)
        np.testing.assert_array_equal(dense[~released], sparse[~released])
        np.testing.assert_array_equal(sparse[released], words.symbols[18])
        np.testing.assert_array_equal(dense[link.CORE], words.symbols[:18].ravel())
        pair = link.simulate(782)
        np.testing.assert_allclose(pair.b["dense"]-pair.clean_b["dense"],
                                   pair.b["sparse"]-pair.clean_b["sparse"], rtol=0, atol=7e-16)
        self.assertFalse(np.array_equal(pair.a, pair.donor_a))

    def test_sequential_gain_and_evidence_match_batch_gaussian(self):
        seed = 812
        p = link.simulate(seed)
        rows = link.PILOT_ROWS
        z = tracker.observed_taps(p.a, rows)
        state = tracker.update(tracker.prior(), rows, z)
        u = tracker.evolution(0.)[:, rows].mean(axis=-1)
        for ray, power in enumerate(link.POWERS):
            den = 1./power + (abs(u)**2).sum(axis=1)/tracker.TAP_NOISE
            mean = (u.conj() @ z[:, ray]) / (tracker.TAP_NOISE * den)
            np.testing.assert_allclose(state.variance[ray], 1/den, atol=1e-14)
            np.testing.assert_allclose(state.mean[ray], mean, atol=2e-14)
            expected = []
            for v in u:
                cov = tracker.TAP_NOISE*np.eye(len(rows)) + power*np.outer(v, v.conj())
                expected.append(-np.linalg.slogdet(cov)[1] - np.vdot(z[:, ray], np.linalg.solve(cov, z[:, ray])).real)
            np.testing.assert_allclose(state.log_weight[ray]-state.log_weight[ray].max(),
                                       expected-np.max(expected), atol=2e-11)
        one = tracker.advance(state, link.FRAME_DURATION)
        twice = tracker.advance(tracker.advance(state, 7.), link.FRAME_DURATION-7.)
        np.testing.assert_allclose(one.mean, twice.mean, atol=1e-14)
        np.testing.assert_array_equal(one.log_weight, state.log_weight)

    def test_blind_prediction_and_decoding_never_read_withheld_values(self):
        pair = link.simulate(912)
        rows = link.B_ROWS["dense"]
        y = pair.b["dense"]
        original = tracker.advance(tracker.prior(), link.FRAME_DURATION)
        posterior = tracker.update(original, rows, tracker.observed_taps(y, rows))
        expected = tracker.readout(posterior, y, rows)
        poisoned = y.copy()
        poisoned[link.PROBES] = np.nan
        from_blind = tracker.update(original, rows, tracker.observed_taps(poisoned, rows))
        for a, b in zip(expected, tracker.readout(from_blind, poisoned, rows)):
            np.testing.assert_array_equal(a, b)
        # Predictions use only the pilot-conditioned state, even if the payload changes.
        poisoned[link.CORE] = 1000 + 1000j
        changed = tracker.readout(from_blind, poisoned, rows)
        np.testing.assert_array_equal(expected[1], changed[1])
        np.testing.assert_array_equal(expected[2], changed[2])
        self.assertFalse(np.array_equal(expected[0], changed[0]))

    def test_predictive_moments_against_time_domain_draws(self):
        rng = link.stream(2113, 2)
        pair = link.simulate(522)
        rows = link.B_ROWS["sparse"]
        st = tracker.advance(tracker.prior(), link.FRAME_DURATION)
        st = tracker.update(st, rows, tracker.observed_taps(pair.b["sparse"], rows))
        _, mean, variance = tracker.readout(st, pair.b["sparse"], rows)
        draws = 16000
        x = np.broadcast_to(link.PILOTS[link.PROBE_ROW], (draws, link.N)).copy()
        x[:, 1::2] = qam(rng.integers(2, size=(draws, 16, 4))).reshape(draws, 16)
        tx = np.fft.ifft(x, norm="ortho")
        timed = np.sqrt(link.N0) * link.complex_normal(rng, tx.shape)
        for r, d in enumerate(link.DELAYS):
            node = rng.choice(len(tracker.FREQ), size=draws, p=np.exp(st.log_weight[r]))
            gain = st.mean[r, node] + np.sqrt(st.variance[r, node])*link.complex_normal(rng, draws)
            rot = tracker.evolution(st.epoch)[node, link.PROBE_ROW]
            timed += np.roll(tx, d, axis=-1) * rot * gain[:, None]
        sampled = np.fft.fft(timed, norm="ortho")[:, link.EVEN]
        np.testing.assert_allclose(sampled.mean(axis=0), mean, rtol=0, atol=.04)
        np.testing.assert_allclose(np.mean(abs(sampled-mean)**2, axis=0), variance, rtol=.045)


if __name__ == "__main__":
    unittest.main()
