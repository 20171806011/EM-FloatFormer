"""Synthetic payload-only checks for the evaluation calibration summaries."""

import json
import math
from types import SimpleNamespace
import unittest

import torch

from p4.simulator import q16
from p6_consistency.calibration import VARIANCE_EDGES, calibration


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def fixture(self, batch_size=1):
        mask = torch.zeros(2, 4, dtype=torch.bool)
        mask[-1, -1] = True
        bits = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 1], [1, 1, 1, 1],
                             [1, 0, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1],
                             [0, 1, 1, 0]], dtype=torch.int64)
        bits = bits[None].repeat(batch_size, 1, 1)
        frame = SimpleNamespace(obs=SimpleNamespace(pilot_mask=mask), bits=bits)
        truth = torch.zeros(batch_size, 2, 4, dtype=torch.complex128)
        truth[:, ~mask] = q16(bits).to(torch.complex128)
        variance = torch.full_like(truth.real, 1000.)
        variance[:, ~mask] = torch.tensor(VARIANCE_EDGES, dtype=torch.float64)
        offset = torch.zeros_like(truth)
        offset[:, ~mask] = (torch.arange(7, dtype=torch.float64) / 10).to(torch.complex128)
        traces = [dict(step=step, mean=truth + (step + 1) * offset, variance=variance.clone())
                  for step in range(3)]
        for trace in traces:
            trace["mean"][:, mask] = 100 + 200j
        llr = torch.zeros(batch_size, 2, 4, 4, dtype=torch.float64)
        llr[:, ~mask] = (2 * bits - 1).double() * math.log(3)
        llr[:, mask] = float("nan")  # Pilot quantities are not payload endpoints.
        return frame, llr, traces

    def test_fixed_bins_include_boundaries_on_the_left_and_exclude_pilots(self):
        frame, llr, traces = self.fixture(batch_size=2)
        result = calibration(frame, llr, traces)
        self.assertEqual(result["payload_symbols"], 14)
        self.assertEqual(result["variance_bin_edges"], [*VARIANCE_EDGES, None])
        for step, row in enumerate(result["steps"]):
            self.assertEqual(row["total"]["count"], 14)
            self.assertAlmostEqual(row["total"]["sum_predicted_variance"],
                                   2 * sum(VARIANCE_EDGES), places=13)
            expected_error = 2 * (step + 1)**2 * sum((index / 10)**2 for index in range(7))
            self.assertAlmostEqual(row["total"]["sum_squared_error"], expected_error, places=13)
            for index, entry in enumerate(row["bins"]):
                self.assertEqual(entry["count"], 2)
                self.assertAlmostEqual(entry["sum_predicted_variance"],
                                       2 * VARIANCE_EDGES[index], places=13)
                self.assertAlmostEqual(entry["sum_squared_error"],
                                       2 * (step + 1)**2 * (index / 10)**2, places=13)
            self.assertEqual(sum(entry["count"] for entry in row["bins"]), row["total"]["count"])
            self.assertAlmostEqual(sum(entry["sum_squared_error"] for entry in row["bins"]),
                                   row["total"]["sum_squared_error"], places=13)
        json.dumps(result, allow_nan=False)

    def test_final_bce_uses_positive_one_convention(self):
        frame, llr, traces = self.fixture(batch_size=2)
        result = calibration(frame, llr, traces)["final"]
        self.assertEqual(result["bit_count"], 56)
        self.assertAlmostEqual(result["bce_sum"], 56 * math.log(4/3), places=13)
        self.assertAlmostEqual(result["mean_abs_llr"], math.log(3), places=14)
        self.assertAlmostEqual(result["sum_abs_llr"], 56 * math.log(3), places=13)
        self.assertAlmostEqual(result["max_abs_llr"], math.log(3), places=14)

    def test_large_wrong_llr_is_not_clipped(self):
        frame, llr, traces = self.fixture()
        data = ~frame.obs.pilot_mask
        values = (2 * frame.bits - 1).double() * 10000
        values[0, 0, 0] *= -1
        llr[:, data] = values
        final = calibration(frame, llr, traces)["final"]
        self.assertEqual(final["bce_sum"], 10000.)
        self.assertEqual(final["max_abs_llr"], 10000.)
        self.assertEqual(final["mean_abs_llr"], 10000.)

    def test_optional_channel_llr_and_diagnostics_are_finite_json(self):
        frame, llr, traces = self.fixture()
        traces[0]["channel_llr"] = llr.clone()
        traces[0]["diagnostics"] = dict(eigenvalues=torch.tensor([.1, .2]),
                                         active=torch.tensor(True), absent=None)
        result = calibration(frame, llr, traces)
        self.assertEqual(result["steps"][0]["channel"], result["final"])
        self.assertEqual(result["steps"][0]["diagnostics"]["active"], True)
        self.assertIsNone(result["steps"][0]["diagnostics"]["absent"])
        json.dumps(result, allow_nan=False)

    def test_batch_sums_are_additive(self):
        frame_one, llr_one, traces_one = self.fixture()
        frame_two, llr_two, traces_two = self.fixture(batch_size=2)
        one, two = calibration(frame_one, llr_one, traces_one), calibration(frame_two, llr_two, traces_two)
        for first, second in zip(one["steps"], two["steps"]):
            for name in ("count", "sum_predicted_variance", "sum_squared_error"):
                self.assertAlmostEqual(2 * first["total"][name], second["total"][name], places=13)
        self.assertAlmostEqual(2 * one["final"]["bce_sum"], two["final"]["bce_sum"], places=13)
        self.assertEqual(2 * one["final"]["bit_count"], two["final"]["bit_count"])

    def test_inputs_and_receiver_boundary_remain_untouched(self):
        frame, llr, traces = self.fixture()
        before = [(trace["mean"].clone(), trace["variance"].clone()) for trace in traces]
        bits = frame.bits.clone()
        # No channel truth or received samples are provided to this evaluator.
        calibration(frame, llr, traces)
        self.assertTrue(torch.equal(frame.bits, bits))
        for trace, (mean, variance) in zip(traces, before):
            self.assertTrue(torch.equal(trace["mean"], mean))
            self.assertTrue(torch.equal(trace["variance"], variance))

    def test_exactly_three_distinct_steps_are_required(self):
        frame, llr, traces = self.fixture()
        with self.assertRaises(ValueError):
            calibration(frame, llr, traces[:2])
        traces[2]["step"] = traces[0]["step"]
        with self.assertRaises(ValueError):
            calibration(frame, llr, traces)

    def test_nonfinite_payload_or_negative_variance_is_rejected(self):
        for invalid in (float("nan"), float("inf"), -.1):
            frame, llr, traces = self.fixture()
            traces[0]["variance"][0, 0, 0] = invalid
            with self.subTest(variance=invalid), self.assertRaises(ValueError):
                calibration(frame, llr, traces)
        frame, llr, traces = self.fixture()
        llr[0, 0, 0, 0] = float("inf")
        with self.assertRaises(ValueError):
            calibration(frame, llr, traces)
        frame, llr, traces = self.fixture()
        traces[0]["mean"][0, 0, 0] = complex(float("nan"), 0)
        with self.assertRaises(ValueError):
            calibration(frame, llr, traces)


if __name__ == "__main__":
    unittest.main(verbosity=2)
