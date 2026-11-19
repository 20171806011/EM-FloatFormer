"""Development-only integration checks of receiver/model consistency.

These tests use archived development checkpoints and seeds only.  They do not
select an arm, evaluate fresh-study seeds, or treat known receive masks as a
blind corruption detector.
"""
from dataclasses import replace
import json
import math
import unittest
from unittest.mock import patch

import torch

from p6.data import dev_frame
from p6_followup.common import CONDITIONS, corrupt, load_model
from p6_softinfo.receiver import SoftInformationReceiver
from p6_consistency.observation import observation_gain
from p6_consistency.receiver import LegacyTraceReceiver, ObservationAwareReceiver


CONFIGURATIONS = {
    "aware_mean": (False, False, False),
    "aware_variance": (True, False, False),
    "effective_variance": (True, True, False),
    "cavity_mean": (False, True, True),
    "cavity_variance": (True, True, True),
}


def aware(model, flags):
    return ObservationAwareReceiver(model, use_variance=flags[0],
                                     effective=flags[1], cavity=flags[2])


class ReceiverIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.models = {pair: load_model(pair, development=True)[0] for pair in (901, 902, 903)}
        cls.frames = {pair: dev_frame(2, 6, pair, 1) for pair in cls.models}

    @torch.inference_mode()
    def test_legacy_trace_preserves_frozen_readouts_bitwise(self):
        for pair, model in self.models.items():
            for condition in CONDITIONS:
                obs = corrupt(self.frames[pair], condition).obs
                for variance in (False, True):
                    candidate = LegacyTraceReceiver(model, use_variance=variance)
                    reference = SoftInformationReceiver(model, use_variance=True) if variance else model
                    with self.subTest(pair=pair, condition=condition, variance=variance):
                        before, after = reference(obs), candidate(obs)
                        self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, after)))
                        self.assertEqual(len(candidate.last_trace), model.steps)

    @torch.inference_mode()
    def test_all_valid_mean_arm_matches_old_solver_with_reported_roundoff(self):
        diagnostics = []
        for pair, model in self.models.items():
            frame = self.frames[pair]
            before = model(frame.obs)
            after = aware(model, CONFIGURATIONS["aware_mean"])(frame.obs)
            llr_error = float((before[0]-after[0]).abs().max())
            channel_error = float((before[1]-after[1]).abs().max())
            hard_flips = int(((before[0][:, ~frame.obs.pilot_mask] > 0)
                              != (after[0][:, ~frame.obs.pilot_mask] > 0)).sum())
            diagnostics.append(dict(pair=pair, max_llr_difference=llr_error,
                                    max_channel_difference=channel_error, payload_hard_flips=hard_flips))
            with self.subTest(pair=pair):
                self.assertLess(llr_error, 1e-12)
                self.assertLess(channel_error, 1e-12)
                self.assertEqual(hard_flips, 0)
        print("ALL_VALID_MEAN_EQUIVALENCE " + json.dumps(diagnostics), flush=True)

    @torch.inference_mode()
    def test_known_positive_gain_inversion_recovers_intact_predictions(self):
        for pair, model in self.models.items():
            frame = self.frames[pair]
            changed = corrupt(frame, "first_attenuated").obs
            gain = observation_gain(frame.obs.pilot_mask, "first_attenuated")
            for name, flags in CONFIGURATIONS.items():
                candidate = aware(model, flags)
                before, after = candidate(frame.obs), candidate(changed, gain)
                with self.subTest(pair=pair, method=name):
                    for a, b in zip(before, after):
                        torch.testing.assert_close(a, b, atol=1e-10, rtol=0.)
                    data = ~frame.obs.pilot_mask
                    self.assertTrue(torch.equal(before[0][:, data] > 0, after[0][:, data] > 0))

    @torch.inference_mode()
    def test_invalid_received_samples_cannot_change_output(self):
        frame = self.frames[903]
        gain = observation_gain(frame.obs.pilot_mask, "scattered_erased")
        obs = corrupt(frame, "scattered_erased").obs
        changed_y = obs.y.clone()
        changed_y[:, :, gain == 0] = 27. + 91j
        changed = replace(obs, y=changed_y)
        for name, flags in CONFIGURATIONS.items():
            candidate = aware(self.models[903], flags)
            before, after = candidate(obs, gain), candidate(changed, gain)
            with self.subTest(method=name):
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(before, after)))

    @torch.inference_mode()
    def test_transmitted_pilots_stay_known_when_their_observations_are_missing(self):
        frame = self.frames[901]
        gain = observation_gain(frame.obs.pilot_mask, "first_erased")
        obs = corrupt(frame, "first_erased").obs
        missing_pilots = frame.obs.pilot_mask & (gain == 0)
        self.assertGreater(int(missing_pilots.sum()), 0)
        for name, flags in CONFIGURATIONS.items():
            candidate = aware(self.models[901], flags)
            candidate(obs, gain)
            with self.subTest(method=name):
                for row in candidate.last_trace:
                    self.assertTrue(torch.equal(row["mean"][:, missing_pilots],
                                                frame.obs.pilots[missing_pilots].unsqueeze(0)))
                    self.assertEqual(float(row["variance"][:, missing_pilots].abs().max()), 0.)
                self.assertEqual(candidate.last_diagnostics["transmitted_pilots"],
                                 int(frame.obs.pilot_mask.sum()))
                self.assertEqual(candidate.last_diagnostics["observed_pilots"],
                                 int((frame.obs.pilot_mask & (gain > 0)).sum()))

    @torch.inference_mode()
    def test_no_private_metadata_or_simulator_is_available_to_inference(self):
        frame = self.frames[902]
        gain = observation_gain(frame.obs.pilot_mask, "interior_erased")
        obs = corrupt(frame, "interior_erased").obs

        class ObservableFieldsOnly:
            def __init__(self, original):
                for key in ("y", "noise_var", "pilot_mask", "pilots", "regime"):
                    setattr(self, key, getattr(original, key))

            def __getattr__(self, key):
                raise AssertionError("Receiver requested unavailable metadata: " + key)

        public = ObservableFieldsOnly(obs)
        for name, flags in CONFIGURATIONS.items():
            candidate = aware(self.models[902], flags)
            expected = candidate(obs, gain)
            with self.subTest(method=name), patch("p5.wavefield_sim.simulate",
                    side_effect=AssertionError("Simulation invoked during inference")):
                actual = candidate(public, gain)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual)))

    @torch.inference_mode()
    def test_forward_does_not_mutate_observation_gain_or_checkpoint(self):
        frame = self.frames[902]
        obs = corrupt(frame, "scattered_erased").obs
        gain = observation_gain(obs.pilot_mask, "scattered_erased")
        original = {key: getattr(obs, key).clone() for key in ("y", "noise_var", "pilots", "pilot_mask")}
        original_gain = gain.clone()
        model = self.models[902]
        state = {key: value.clone() for key, value in model.state_dict().items()}
        for name, flags in CONFIGURATIONS.items():
            aware(model, flags)(obs, gain)
            with self.subTest(method=name):
                self.assertTrue(torch.equal(gain, original_gain))
                self.assertTrue(all(torch.equal(getattr(obs, key), value) for key, value in original.items()))
                self.assertTrue(all(torch.equal(model.state_dict()[key], value) for key, value in state.items()))
                self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    @torch.inference_mode()
    def test_all_variants_handle_missing_and_zero_noise_edges(self):
        obs = self.frames[902].obs
        all_valid = torch.ones_like(obs.pilot_mask, dtype=torch.float64)
        all_missing = torch.zeros_like(all_valid)
        no_pilot_observations = torch.where(obs.pilot_mask, 0., all_valid)
        nv0 = replace(obs, noise_var=torch.zeros_like(obs.noise_var))
        zero_field_nv0 = replace(nv0, y=torch.zeros_like(obs.y))
        cases = {
            "all_missing": (obs, all_missing, True),
            "all_missing_zero_noise": (nv0, all_missing, True),
            "zero_noise": (nv0, all_valid, False),
            "zero_field_zero_noise": (zero_field_nv0, all_valid, True),
            "no_observed_pilots": (obs, no_pilot_observations, True),
        }
        for name, flags in CONFIGURATIONS.items():
            candidate = aware(self.models[902], flags)
            for condition, (changed, gain, guarded_zero) in cases.items():
                with self.subTest(method=name, condition=condition):
                    llr, h = candidate(changed, gain)
                    self.assertTrue(torch.isfinite(llr).all())
                    self.assertTrue(torch.isfinite(h).all())
                    if guarded_zero:
                        self.assertEqual(float(llr.abs().max()), 0.)
                        self.assertEqual(float(h.abs().max()), 0.)
                    if flags[2]:
                        for record in candidate.last_diagnostics["cavity"]:
                            self.assertTrue(all(math.isfinite(x) for x in record["predictive_quadratic_by_burst"]))

    @torch.inference_mode()
    def test_representative_complex_gain_transformations(self):
        frame = self.frames[903]
        for condition in ("intact", "scattered_erased"):
            obs = corrupt(frame, condition).obs
            known_gain = observation_gain(obs.pilot_mask, condition)
            for name, flags in CONFIGURATIONS.items():
                candidate = aware(self.models[903], flags)
                expected_llr, expected_h = candidate(obs, known_gain)
                for scale in (1e-35, .37, 1e20):
                    gain = scale * complex(math.cos(.731), math.sin(.731))
                    changed = replace(obs, y=gain*obs.y, noise_var=abs(gain)**2*obs.noise_var)
                    actual_llr, actual_h = candidate(changed, known_gain)
                    with self.subTest(method=name, condition=condition, scale=scale):
                        difference = float((actual_llr-expected_llr).abs().max())
                        self.assertLess(difference, 1e-12)
                        relative = float(torch.linalg.vector_norm(actual_h/gain-expected_h)
                                         / torch.linalg.vector_norm(expected_h))
                        self.assertLess(relative, 1e-12)
                        data = ~obs.pilot_mask
                        self.assertTrue(torch.equal(actual_llr[:, data] > 0, expected_llr[:, data] > 0))

    @torch.inference_mode()
    def test_unsupported_regimes_and_antenna_counts_are_rejected(self):
        obs = self.frames[901].obs
        candidate = aware(self.models[901], CONFIGURATIONS["cavity_variance"])
        for regime in (1, 3):
            with self.subTest(regime=regime), self.assertRaises(ValueError):
                candidate(replace(obs, regime=regime))
        with self.assertRaises(ValueError):
            candidate(replace(obs, y=obs.y.repeat(1, 2, 1, 1)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
