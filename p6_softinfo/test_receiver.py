"""Development-only receiver integration checks; never use fresh-study seeds."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import torch

from p5b.diagnostics import erased
from p6.data import dev_frame
from p6.diagnostics import degenerate
from p6_followup.common import CONDITIONS, corrupt, load_model
from p6_softinfo.receiver import SoftInformationReceiver


def edge_observations(obs):
    zero_noise = replace(obs, noise_var=torch.zeros_like(obs.noise_var))
    return {
        "all_pilots": erased(obs, "all_pilots"),
        "whole_field": erased(obs, "whole_field"),
        "degenerate_pilots": degenerate(obs),
        "zero_noise": zero_noise,
        "all_pilots_zero_noise": erased(zero_noise, "all_pilots"),
        "whole_zero_noise": erased(zero_noise, "whole_field"),
    }


class BaselineEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @torch.no_grad()
    def test_copy_matches_frozen_p6_on_all_development_pairs_and_conditions(self):
        for pair in (901, 902, 903):
            model, _ = load_model(pair, development=True)
            copied = SoftInformationReceiver(model, use_variance=False, refine=False)
            frame = dev_frame(2, 6, pair, 1)
            for condition in CONDITIONS:
                with self.subTest(pair=pair, condition=condition):
                    obs = corrupt(frame, condition).obs
                    expected, actual = model(obs), copied(obs)
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))

    @torch.no_grad()
    def test_copy_matches_frozen_zero_and_degenerate_guards(self):
        model, _ = load_model(901, development=True)
        copied = SoftInformationReceiver(model, use_variance=False, refine=False)
        obs = dev_frame(2, 6, 901, 1).obs
        for name, changed in edge_observations(obs).items():
            with self.subTest(condition=name):
                expected, actual = model(changed), copied(changed)
                self.assertTrue(all(torch.isfinite(x).all() for x in actual))
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))

    @torch.no_grad()
    def test_unsupported_nonlinear_regimes_are_rejected(self):
        model, _ = load_model(901, development=True)
        copied = SoftInformationReceiver(model)
        obs = dev_frame(2, 6, 901, 1).obs
        for regime in (1, 3):
            with self.subTest(regime=regime), self.assertRaises(ValueError):
                copied(replace(obs, regime=regime))


class CandidateIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    @torch.no_grad()
    def test_ablation_modes_are_finite_on_development_and_edge_observations(self):
        model, _ = load_model(902, development=True)
        frame = dev_frame(2, 6, 902, 1)
        observations = {name: corrupt(frame, name).obs for name in CONDITIONS}
        observations.update(edge_observations(frame.obs))
        for variance, refine in ((True, False), (False, True), (True, True)):
            candidate = SoftInformationReceiver(model, use_variance=variance, refine=refine)
            for name, obs in observations.items():
                with self.subTest(variance=variance, refine=refine, condition=name):
                    llr, h = candidate(obs)
                    self.assertEqual(llr.shape, (*obs.y.shape[:1], *obs.y.shape[2:], 4))
                    self.assertEqual(h.shape, obs.y.shape)
                    self.assertTrue(torch.isfinite(llr).all())
                    self.assertTrue(torch.isfinite(h).all())
                    if variance:
                        self.assertEqual(len(candidate.last_diagnostics["moments"]), model.steps)
                        for row in candidate.last_diagnostics["moments"]:
                            self.assertGreaterEqual(row["mean_variance"], 0.)
                            self.assertGreaterEqual(row["covariance_trace_mean"], 0.)
                    if name.startswith("all_pilots") or name.startswith("whole_"):
                        self.assertTrue(torch.equal(llr, torch.zeros_like(llr)))
                        self.assertTrue(torch.equal(h, torch.zeros_like(h)))

    @torch.no_grad()
    def test_forward_requires_only_public_observation_fields(self):
        model, _ = load_model(903, development=True)
        obs = dev_frame(2, 6, 903, 1).obs

        class ObservationOnly:
            def __init__(self, original):
                for name in ("y", "pilot_mask", "pilots", "noise_var", "regime"):
                    setattr(self, name, getattr(original, name))

            def __getattr__(self, name):
                raise AssertionError(f"Receiver requested non-observable field: {name}")

        for variance, refine in ((True, False), (False, True), (True, True)):
            candidate = SoftInformationReceiver(model, use_variance=variance, refine=refine)
            with self.subTest(variance=variance, refine=refine), patch(
                    "p5.wavefield_sim.simulate", side_effect=AssertionError("Simulation inside inference")):
                expected = candidate(obs)
                actual = candidate(ObservationOnly(obs))
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))

    @torch.no_grad()
    def test_inference_does_not_modify_weights_or_observation(self):
        model, _ = load_model(901, development=True)
        obs = dev_frame(2, 6, 901, 1).obs
        state = {name: value.clone() for name, value in model.state_dict().items()}
        original = {name: getattr(obs, name).clone()
                    for name in ("y", "pilot_mask", "pilots", "noise_var")}
        candidate = SoftInformationReceiver(model, use_variance=True, refine=True)
        candidate(obs)
        self.assertTrue(all(torch.equal(value, state[name]) for name, value in model.state_dict().items()))
        self.assertTrue(all(torch.equal(getattr(obs, name), value) for name, value in original.items()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    @torch.no_grad()
    def test_profile_also_works_inside_inference_mode(self):
        model, _ = load_model(903, development=True)
        obs = dev_frame(2, 6, 903, 1).obs
        for variance in (False, True):
            candidate = SoftInformationReceiver(model, use_variance=variance, refine=True)
            expected = candidate(obs)
            with self.subTest(variance=variance), torch.inference_mode():
                actual = candidate(obs)
            # PyTorch's inference-mode dispatch also changes the frozen P6
            # reference at roundoff scale. Keep the established 1e-12 numeric
            # criterion here; baseline-copy equivalence above stays bit exact.
            for first, second in zip(expected, actual):
                self.assertTrue(torch.isfinite(second).all())
                self.assertLess(float((first - second).abs().max()), 1e-12)
            self.assertTrue(torch.equal(expected[0][:, ~obs.pilot_mask] > 0,
                                        actual[0][:, ~obs.pilot_mask] > 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
