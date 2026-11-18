"""Independent numerical and structural tests of the P5-B receiver."""

from dataclasses import replace
import math
import unittest
from unittest.mock import patch

import torch

from p4.slot_2d import _get_default_ldpc, encode_ldpc
from p5.audit_model import initial_gradient_audit
from p5.study import bit_loss, rng
from p5.unrolled_em_transformer import _geometry
from p5.wavefield_sim import simulate
from p5b.diagnostics import ARMS, SNR, degenerate_pilot_audit, gauge_suite, model_sizes
from p5b.operator_em_transformer import OperatorEMTransformer, _ComplexOperatorHead, _emit, _synthesize
from p5b.parity import min_sum_extrinsic, soft_parity_symbols


class ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_messages_against_scalar_check_node_oracle(self):
        words = torch.randn(2, 128, dtype=torch.float64, generator=torch.Generator().manual_seed(7051))
        # Include a tie, a zero LLR and both check-node degree parities.
        words[0, :4] = torch.tensor([0., .125, -.125, 0.])
        reference = torch.zeros_like(words)
        for h_row in _get_default_ldpc().h_matrix:
            neighbors = torch.where(h_row != 0)[0].tolist()
            for target in neighbors:
                others = words[:, [j for j in neighbors if j != target]]
                parity_one = ((others > 0).sum(-1) % 2) != 0
                confidence = .75 * others.abs().amin(-1)
                reference[:, target] += torch.where(parity_one, confidence, -confidence)
        torch.testing.assert_close(min_sum_extrinsic(words), reference, rtol=0., atol=1e-15)

    def test_extrinsic_jacobian_diagonal_and_cross_word_leakage(self):
        llr = torch.randn(128, dtype=torch.float64, generator=torch.Generator().manual_seed(832))
        jac = torch.autograd.functional.jacobian(min_sum_extrinsic, llr)
        self.assertTrue(torch.isfinite(jac).all())
        self.assertEqual(int(jac.diagonal().count_nonzero()), 0)
        self.assertGreater(int(jac.count_nonzero()), 128)
        direction = torch.randn(128, dtype=llr.dtype, generator=torch.Generator().manual_seed(782))
        eps = 1e-7
        numerical = (min_sum_extrinsic(llr + eps * direction) - min_sum_extrinsic(llr - eps * direction)) / (2 * eps)
        torch.testing.assert_close(jac @ direction, numerical, rtol=1e-7, atol=5e-9)
        joint = torch.stack((llr, 2. * llr.flip((-1,))))
        self.assertTrue(torch.equal(min_sum_extrinsic(joint)[0], min_sum_extrinsic(llr)))

    def test_valid_words_sign_and_zero_information(self):
        info = torch.randint(2, (8, 64), generator=torch.Generator().manual_seed(993))
        bits, _ = encode_ldpc(info)
        llr = (2. * bits.double() - 1.) * 5.
        self.assertTrue(torch.equal(min_sum_extrinsic(llr) > 0, bits.bool()))
        self.assertEqual(int(min_sum_extrinsic(torch.zeros_like(llr)).count_nonzero()), 0)

    def test_pilot_exclusion_symbol_posterior_and_gradients(self):
        frame = simulate(1, 2, 12, 83133)
        mask = frame.obs.pilot_mask
        shape = (2, *mask.shape)
        z = torch.randn(shape, dtype=torch.complex128, generator=torch.Generator().manual_seed(721))
        z.requires_grad_()
        variance = torch.full(shape, .48, dtype=torch.float64)
        observed = torch.ones(shape, dtype=torch.bool)
        soft = soft_parity_symbols(z, variance, observed, mask)
        loss = soft.real.square().sum() + .3 * soft.imag.square().sum()
        grad, = torch.autograd.grad(loss, (z,))
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().max()), 0.)
        self.assertEqual(int(grad[:, mask].count_nonzero()), 0)
        altered = torch.where(mask, 1e6 - 2e6j, z.detach())
        self.assertTrue(torch.equal(soft, soft_parity_symbols(altered, variance, observed, mask)))
        zero = soft_parity_symbols(z.detach(), variance, torch.zeros_like(observed), mask)
        self.assertEqual(int(zero.count_nonzero()), 0)
        # Independent 4-PAM max-log bit demapper and enumerated posterior mean.
        pam = z.real.new_tensor([-3., -1., 1., 3.]) / math.sqrt(10.)
        labels = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=torch.float64)
        logp = -(torch.view_as_real(z[:, ~mask])[..., None] - pam).square() / .48
        channel = torch.stack([logp[..., labels[:, j] == 1].amax(-1)
                               - logp[..., labels[:, j] == 0].amax(-1) for j in range(2)], -1)
        extrinsic = min_sum_extrinsic(channel.reshape(2, -1, 128)).reshape_as(channel)
        expected = ((logp + extrinsic @ labels.T).softmax(-1) * pam).sum(-1)
        torch.testing.assert_close(torch.view_as_real(soft[:, ~mask]), expected, rtol=1e-12, atol=1e-14)


class ReceiverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(782603)

    def test_matched_parameters_and_centered_physics_solver(self):
        for regime, snr in SNR.items():
            obs = simulate(regime, 2, snr, 67168).obs
            sizes, signatures = [], []
            with torch.no_grad():
                for arm in ARMS:
                    torch.manual_seed(67168)
                    model = OperatorEMTransformer(variant=arm)
                    sizes.append(model_sizes(model)["trainable_real"])
                    signatures.append([(n, tuple(p.shape)) for n, p in model.named_parameters()
                                       if p.requires_grad])
                    variant = "physics_parity" if arm in ("parity", "local_parity") else "physics"
                    physical = OperatorEMTransformer(variant=variant)
                    self.assertEqual(model_sizes(physical)["trainable_real"], 0)
                    for actual, expected in zip(model(obs), physical(obs)):
                        if arm == "no_gauge":
                            torch.testing.assert_close(actual, expected, rtol=0., atol=1e-12)
                        else:
                            self.assertTrue(torch.equal(actual, expected), (regime, arm))
            self.assertTrue(800 <= sizes[0] <= 2500, sizes)
            self.assertEqual(len(set(sizes)), 1)
            self.assertTrue(all(signatures[0] == s for s in signatures[1:]))

    def test_all_first_step_gradients_and_directional_finite_differences(self):
        for regime, snr in SNR.items():
            frame = simulate(regime, 4, snr, rng(regime, 5, 1))
            for variant in ARMS:
                torch.manual_seed(rng(regime, 4, 1))
                model = OperatorEMTransformer(variant=variant)
                audit = initial_gradient_audit(model, frame, rng(regime, 12, 1))
                for tensor in audit["tensors"]:
                    self.assertTrue(tensor["passed"], (regime, variant, tensor))

    def test_learned_cstar_and_erasure_cases(self):
        for regime, snr in SNR.items():
            fit = simulate(regime, 4, snr, rng(regime, 5, 1))
            held = simulate(regime, 2, snr, rng(regime, 11, 1))
            for variant in ("pure", "parity", "local", "local_parity"):
                torch.manual_seed(rng(regime, 4, 1))
                model = OperatorEMTransformer(variant=variant)
                before = model(held.obs)[0].detach()
                opt = torch.optim.Adam(model.parameters(), lr=.003)
                bit_loss(fit, model).backward()
                opt.step()
                self.assertGreater(float((before - model(held.obs)[0]).detach().abs().max()), 0.)
                audit = gauge_suite(model, held, rng(regime, 13, 1))
                self.assertTrue(audit["passed"], (regime, variant, audit))

    def test_antipodal_phase_and_tied_doppler_pilots(self):
        for regime in SNR:
            model = OperatorEMTransformer(variant="parity")
            audit = degenerate_pilot_audit(model, regime)
            self.assertTrue(audit["passed"], audit)

    def test_radial_terms_preserve_phase_and_have_independent_effects(self):
        x = torch.randn(2, 5, 32, dtype=torch.complex128)
        coeff = torch.randn(2, 5, 2, dtype=torch.complex128)
        phase = torch.exp(1j * torch.randn(2, 5, 1, dtype=torch.float64))
        for regime in SNR:
            observed = _emit(x, regime, coeff)
            torch.testing.assert_close(_emit(phase * x, regime, coeff), phase * observed,
                                       rtol=1e-13, atol=1e-13)
            zero = _emit(x, regime, torch.zeros_like(coeff))
            radial3, radial5 = coeff.clone(), coeff.clone()
            radial3[..., 1] = 0.
            radial5[..., 0] = 0.
            change3, change5 = _emit(x, regime, radial3) - zero, _emit(x, regime, radial5) - zero
            self.assertGreater(float(change3.abs().max()), 1e-6)
            self.assertGreater(float(change5.abs().max()), 1e-6)
            changed = observed - zero
            torch.testing.assert_close(changed, change3 + change5, rtol=1e-12, atol=1e-14)
            self.assertGreater(float(changed.imag.abs().max()), 1e-6)

    def test_parity_calls_are_confined_to_interstep_feedback(self):
        obs = simulate(2, 1, 9., 128603).obs
        with torch.no_grad():
            for arm in (*ARMS, "physics", "physics_parity"):
                with patch("p5b.operator_em_transformer.soft_parity_symbols", wraps=soft_parity_symbols) as spy:
                    OperatorEMTransformer(variant=arm)(obs)
                self.assertEqual(spy.call_count, int(arm in ("parity", "local_parity", "physics_parity")))
            with patch("p5b.operator_em_transformer.soft_parity_symbols", wraps=soft_parity_symbols) as spy:
                OperatorEMTransformer(variant="parity", steps=3)(obs)
            self.assertEqual(spy.call_count, 2)

    def test_operator_output_charges_and_nonlocal_reach(self):
        geom = _geometry(3, 32, 5, 4)
        coords = geom[5]
        features = torch.randn(2, len(coords), 6, dtype=torch.complex128)
        phases = torch.polar(torch.ones(2, dtype=torch.float64),
                             torch.tensor([-1.21, 2.09], dtype=torch.float64))[:, None, None]
        with torch.no_grad():
            for arm in ("pure", "local"):
                head = _ComplexOperatorHead(arm)
                values, weights = head(features, geom)
                changed, weights2 = head(features * phases, geom)
                self.assertEqual(values.shape[-1], 6)
                torch.testing.assert_close(changed[..., :5], values[..., :5], atol=2e-13, rtol=0.)
                torch.testing.assert_close(changed[..., 5:], phases * values[..., 5:], atol=2e-13, rtol=0.)
                self.assertGreater(float(values[..., :5].imag.abs().max()), 0.)
                if weights is not None:
                    torch.testing.assert_close(weights2, weights, atol=2e-13, rtol=0.)
                remote = features.clone()
                remote[:, -1] *= 1.7j
                delta = (head(remote, geom)[0][:, 0] - values[:, 0]).abs().max()
                if arm == "pure":
                    self.assertGreater(float(delta), 1e-12)
                else:
                    self.assertEqual(float(delta), 0.)

    def test_synthesis_reproduces_knots_and_linear_midpoints(self):
        geom = _geometry(3, 32, 5, 4)
        coords = geom[5]
        raw = torch.randn(2, len(coords), 6, dtype=torch.complex128)
        dense = _synthesize(raw, geom, 5, 32)
        a, t, k = coords.T
        torch.testing.assert_close(dense[:, a, t, k], raw, rtol=0., atol=1e-14)
        for low, high in ((2, 10), (10, 18), (18, 26), (26, 2)):
            mid = (low + 4) % 32
            torch.testing.assert_close(dense[..., mid, :],
                                       (dense[..., low, :] + dense[..., high, :]) / 2,
                                       rtol=0., atol=1e-14)
        torch.testing.assert_close(dense[:, :, 1], (dense[:, :, 0] + dense[:, :, 2]) / 2,
                                   rtol=0., atol=1e-14)

    def test_no_gauge_control_exposes_scale_dependence_after_learning(self):
        frame = simulate(3, 3, 7, 882506)
        control = OperatorEMTransformer(variant="no_gauge")
        optimizer = torch.optim.Adam(control.parameters(), lr=.003)
        bit_loss(frame, control).backward()
        optimizer.step()
        with torch.no_grad():
            reference = control(frame.obs)[0]
            altered = replace(frame.obs, y=3 * frame.obs.y, noise_var=9 * frame.obs.noise_var)
            current = control(altered)[0]
            self.assertTrue(torch.isfinite(current).all())
            self.assertGreater(float((current - reference).abs().max()), 1e-9)


if __name__ == "__main__":
    unittest.main()
