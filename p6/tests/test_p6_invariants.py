"""Physics, capacity, and numerical checks using development pairs only."""
from pathlib import Path
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import torch

from p5.study import bit_loss, save, setup
from p6.axial_em_transformer import AxialEMTransformer, AxialLocalConv, real_parameters
from p6.axial_operator import full_frame_features
from p6.data import dev_frame, dev_seed
from p6 import diagnostics


class Invariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup()

    def test_dev_boundary_rejects_before_simulation(self):
        with patch("p6.data.simulate", side_effect=AssertionError("generator reached")):
            for stream, pair in [(6, 1), (5, 20), (7, 901), (8, 901), (9, 901), (6, 904)]:
                with self.assertRaises(ValueError):
                    dev_frame(1, stream, pair, 2)
            with self.assertRaises(ValueError):
                dev_frame(3, 9, 901, 2)
            with self.assertRaises(ValueError):
                dev_frame(3, 5, 903, 2, 1000)

    def test_features_include_payload_and_erased_charge(self):
        obs = dev_frame(1, 5, 902, 2).obs
        h = obs.y.roll(1, 2)
        x = obs.y[:, 0]
        residual = obs.y - h * x[:, None]
        fields = full_frame_features(h, obs.y, x, residual, h, obs.pilot_mask)
        self.assertEqual(fields.shape, (*h.shape, 9))
        self.assertTrue(torch.equal(fields[..., 3], residual))
        for g in (1.e-35j, -.7 + 1.1j, -1.e20):
            other = full_frame_features(g*h, g*obs.y, x, g*residual, g*h, obs.pilot_mask)
            torch.testing.assert_close(other/g, fields, rtol=5.e-15, atol=5.e-15)
        zero = h * 0
        self.assertFalse(full_frame_features(zero, zero, x, zero, zero, obs.pilot_mask).count_nonzero())

    def test_same_initial_weights_and_remote_time_reach(self):
        obs = dev_frame(2, 5, 903, 1).obs
        features = full_frame_features(obs.y, obs.y, obs.y[:, 0], obs.y, obs.y, obs.pilot_mask)
        torch.manual_seed(dev_seed(2, 5, 903, 900))
        global_net = AxialEMTransformer("pure")
        local_net = AxialLocalConv()
        local_net.load_state_dict(global_net.state_dict())
        self.assertEqual(real_parameters(global_net), real_parameters(local_net))
        # Payload row 8 is more than one temporal convolution tap from row 0.
        changed = features.clone()
        changed[:, :, 8, 15] *= 1.8j
        with torch.no_grad():
            for head, is_global in ((global_net.head, True), (local_net.head, False)):
                original = head(features)[0][:, :, 0]
                altered = head(changed)[0][:, :, 0]
                difference = float((original - altered).abs().max())
                if is_global:
                    self.assertGreater(difference, 1.e-12)
                else:
                    self.assertEqual(difference, 0.)

    def test_unmodified_seeded_state_matches_in_all_five_arms(self):
        states = []
        for arm in diagnostics.ARMS:
            torch.manual_seed(dev_seed(3, 5, 902, 900))
            model = AxialEMTransformer(variant=arm, steps=3)
            self.assertGreaterEqual(real_parameters(model), 800)
            self.assertLessEqual(real_parameters(model), 1800)
            states.append(dict(model.named_parameters()))
        for state in states[1:]:
            self.assertEqual(state.keys(), states[0].keys())
            for name, tensor in state.items():
                self.assertEqual(tensor.requires_grad, states[0][name].requires_grad)
                self.assertTrue(torch.equal(tensor, states[0][name]), name)

    def test_probe_readout_and_parity_step_count(self):
        obs = dev_frame(3, 8, 903, 2).obs
        from p6.parity import soft_parity_symbols
        for arm, depth in (("parity", 2), ("local_parity", 3), ("pure", 2)):
            model = AxialEMTransformer(arm, steps=depth).eval()
            with patch("p6.axial_em_transformer.soft_parity_symbols", wraps=soft_parity_symbols) as spy:
                with torch.no_grad():
                    llr, h, layers = model(obs, probe="axes")
                self.assertEqual(spy.call_count, depth - 1 if model.parity else 0)
            if arm != "local_parity":
                self.assertEqual(len(layers), depth)
                for coords, joint, axes in layers:
                    self.assertEqual(joint.shape, (len(obs.y), len(coords), len(coords)))
                    torch.testing.assert_close(joint, axes["spatial"]*axes["time"]*axes["frequency"])
            with torch.no_grad():
                ordinary = model(obs)
            for a, b in zip((llr, h), ordinary):
                torch.testing.assert_close(a, b, atol=1e-12, rtol=0)

    def test_probe_columns_against_full_grid_transport(self):
        obs = dev_frame(3, 8, 902, 2).obs
        head = AxialEMTransformer("pure").head
        y = obs.y
        features = full_frame_features(y, y, y[:, 0], y, y, obs.pilot_mask)
        original = head._global
        weights = {}

        def capture(q, k, v, axis, gains):
            result = original(q, k, v, axis, gains)
            weights[axis] = result[1]
            return result

        with torch.no_grad(), patch.object(head, "_global", side_effect=capture):
            _, (coords, joint, _) = head(features, probe=True)
            for source in (0, len(coords)//2, len(coords)-1):
                mass = torch.zeros_like(y.real)
                a, t, f = coords[source]
                mass[:, a, t, f] = 1.
                mass = torch.einsum("brtfk,brtk->brtf", weights[3], mass)
                mass = torch.einsum("brftu,bruf->brtf", weights[2], mass)
                mass = torch.einsum("btfrs,bstf->brtf", weights[1], mass)
                target = mass[:, coords[:, 0], coords[:, 1], coords[:, 2]]
                torch.testing.assert_close(joint[:, :, source], target, rtol=1e-14, atol=1e-15)

    def test_zero_noise_anchor_and_extreme_input_gradients(self):
        for regime in (1, 2, 3):
            train = dev_frame(regime, 5, 902, 2)
            validation = dev_frame(regime, 6, 902, 2)
            noiseless = dev_frame(regime, 6, 902, 2, snr=math.inf).obs
            torch.manual_seed(dev_seed(regime, 5, 902, 900))
            model = AxialEMTransformer("parity")
            fixed = AxialEMTransformer("physics_parity")
            with torch.no_grad():
                for tensor, expected in zip(model(noiseless), fixed(noiseless)):
                    self.assertTrue(torch.equal(tensor, expected))
            opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.003)
            bit_loss(train, model).backward()
            opt.step()
            for magnitude in (1e-35, 1e20):
                model.zero_grad(set_to_none=True)
                obs = validation.obs
                y = (magnitude * (.6 + .8j) * obs.y).requires_grad_()
                changed = replace(obs, y=y, noise_var=magnitude**2 * obs.noise_var)
                bit_loss(replace(validation, obs=changed), model).backward()
                for g in [y.grad, *(p.grad for p in model.parameters() if p.requires_grad)]:
                    self.assertTrue(torch.isfinite(g).all())
                    self.assertGreater(float(g.abs().max()), 0.)

    def test_full_numeric_audit(self):
        result = diagnostics.run()
        save(Path(diagnostics.__file__).parent / "results/iter4_checks/invariants.json", result)
        failures = [(r["regime"], r["arm"]) for r in result["rows"] if not r["passed"]]
        self.assertTrue(result["passed"], failures)


if __name__ == "__main__":
    unittest.main(verbosity=2)
