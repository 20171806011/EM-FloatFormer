import unittest

import numpy as np

from p8 import link, placement, tracker


class PlacementTests(unittest.TestCase):
    def test_same_frequencies_same_airtime_same_common_words(self):
        masks = placement.schedules()
        front, split = masks["sparse"], masks["split"]
        np.testing.assert_array_equal(front.sum(axis=0), split.sum(axis=0))
        self.assertEqual(np.count_nonzero(front & ~split), 8)
        self.assertEqual(np.count_nonzero(split & ~front), 8)
        self.assertEqual(split.sum(axis=1)[[0, 10]].tolist(), [8, 8])
        pair = link.simulate(6824, masks)
        for name, mask in masks.items():
            np.testing.assert_array_equal(pair.b[name][link.PROBES], pair.b["dense"][link.PROBES])
            grid = link.map_traffic(pair.b_words, mask | link.PROBES)
            np.testing.assert_array_equal(grid[link.CORE], pair.b_words.symbols[:18].ravel())
            if name != "dense":
                np.testing.assert_array_equal(grid[~(link.CORE | mask | link.PROBES)],
                                              pair.b_words.symbols[18])

    def test_eight_pilot_projection_and_uncertainty(self):
        mask = placement.schedules()["split"]
        gains = np.array([.53-.32j, -.2+.48j, -.33+.12j])
        channel = gains @ tracker.BASIS
        y = np.full((link.T, link.N), np.nan + 1j*np.nan)
        y[mask] = (channel * link.PILOTS)[mask]
        rows, taps, variances = placement.acquire(y, mask)
        np.testing.assert_array_equal(rows, [0, 10])
        np.testing.assert_allclose(taps, np.tile(gains, (2, 1)), atol=1e-14)
        np.testing.assert_array_equal(variances, np.full(2, 2*tracker.TAP_NOISE))

    def test_split_probe_poison_leaves_all_outputs_identical(self):
        masks = placement.schedules()
        pair = link.simulate(77, masks)
        mask = masks["split"]
        y = pair.b["split"]
        rows, taps, noise = placement.acquire(y, mask)
        prior = tracker.advance(tracker.prior(), link.FRAME_DURATION)
        state = tracker.update(prior, rows, taps, noise)
        reference = tracker.readout_mask(state, y, mask | link.PROBES)
        blind = y.copy()
        blind[link.PROBES] = np.nan
        rows, taps, noise = placement.acquire(blind, mask)
        state = tracker.update(prior, rows, taps, noise)
        for a, b in zip(reference, tracker.readout_mask(state, blind, mask | link.PROBES)):
            np.testing.assert_array_equal(a, b)
