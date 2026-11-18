"""Stage P4 Tier 5 white-box checks, separate from the 253 opaque-box cases."""
import copy
import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import torch

from p4 import confirmatory_protocol as protocol
from p4 import stage3_factorial_confirm as factorial
from p4 import stage3_ood, stage3_precision, stage3_sparse
from p4.em_floatformer import ComplexLinear, compute_gauge_anchor
from p4.next_neural import TapPhaseNoChannel
from p4.next_pilots import sparse_slot
from p4.simulator import rapp_am_am
from p4.slot_2d import EMFloatFormer2D, SlotSimulator2D
from p4.stage3_cfo import phase_and_errors


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "p4/results"
M1_FILES = ("p4/simulator.py", "p4/classical_baselines.py", "p4/benchmark_latency.py",
            "p4/tests/test_p4_0_simulator.py", "p4/results/latency_cpu.csv", "p4/results/timing.json")


def setUpModule():
    previous = torch.get_num_threads()
    state = torch.random.get_rng_state()
    torch.set_num_threads(1)
    unittest.addModuleCleanup(torch.set_num_threads, previous)
    unittest.addModuleCleanup(torch.random.set_rng_state, state)


class Tier5ArchiveAndGradient(unittest.TestCase):
    def test_m1_archive_bytes_and_pre_p4_git_tree(self):
        path = ROOT / "p4/revisions/pre_stage3_p4.zip"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                         "4ebfcf6fb2dd36c7575a3648babd7d4958469e8bc5df9c8fafce8c224371f77a")
        with ZipFile(path) as archive:
            for name in M1_FILES:
                with self.subTest(path=name):
                    self.assertEqual((ROOT/name).read_bytes(), archive.read(name))
            historical = json.loads(archive.read("p4/results/p4d_smoke.json"))["historical_checkout"]
        self.assertEqual(historical["file_count"], 230)
        self.assertEqual(protocol._historical_digest(), historical)

    def test_deep_clipping_against_stable_ratio_form(self):
        # Float32 cannot evaluate (r/A)^4 once r/A exceeds approximately 4e9.
        # Confirm its well-conditioned range and the wider float64 reference.
        r = torch.tensor([0.,1e-12,.1,1.,10.],dtype=torch.float64)
        for dtype,limit in ((torch.float32,1e-6),(torch.float64,1e-6),(torch.float64,1e-12)):
            with self.subTest(dtype=dtype,limit=limit):
                minimum = torch.minimum(r,torch.full_like(r,limit))
                maximum = torch.maximum(r,torch.full_like(r,limit))
                expected = minimum/(1+(minimum/maximum)**4).pow(.25)
                observed = rapp_am_am(r.to(dtype),A=limit,knee=2.).double()
                torch.testing.assert_close(observed,expected,rtol=2e-6,atol=1e-24)

    def test_anchor_derivative_inside_floor_and_nonzero_companion(self):
        values = [[0., 0.], [1e-40j, -1e-40], [3+4j, 0.]]
        for dtype in (torch.complex64, torch.complex128):
            with self.subTest(dtype=dtype):
                x = torch.tensor(values, dtype=dtype, requires_grad=True)
                anchor = compute_gauge_anchor(x)
                grad, = torch.autograd.grad(anchor.sum(), x)
                torch.testing.assert_close(grad[:2], torch.zeros_like(grad[:2]), atol=0, rtol=0)
                rms = 5/math.sqrt(2)
                torch.testing.assert_close(grad[2], x[2].detach()/(2*rms))
                torch.testing.assert_close(anchor.detach().flatten(),
                                           torch.tensor([1e-38,1e-38,rms],dtype=x.real.dtype))

    def test_2d_all_zero_and_mixed_input_and_parameter_gradients(self):
        model = EMFloatFormer2D().eval()
        model.load_state_dict(torch.load(RESULTS/"p4b_seed2026_state.pt", weights_only=True))
        slot = SlotSimulator2D().generate_slot(batch_size=2, snr_db=20.,
                                               generator=torch.Generator().manual_seed(848872))
        params = tuple(model.parameters())
        for dtype in (torch.complex64, torch.complex128):
            for mode in ("all_zero", "mixed", "pilot_erasure"):
                with self.subTest(dtype=dtype, mode=mode):
                    yp, y = slot.yp_sparse.to(dtype).clone(), slot.y_grid.to(dtype).clone()
                    yp[0].zero_()
                    if mode != "pilot_erasure":
                        y[0].zero_()
                    if mode == "all_zero":
                        yp[1].zero_()
                        y[1].zero_()
                    yp.requires_grad_()
                    y.requires_grad_()
                    def objective(pilot, received, nv, factor):
                        out = model(pilot, received, slot.pilot_mask, nv)
                        self.assertTrue(all(x.isfinite().all() for x in out))
                        return ((out.h_hat.real.mean()+.3*out.dist_est.imag.mean())/factor
                                + out.llrs[...,0].mean())
                    value = objective(yp, y, slot.noise_var, 1.)
                    grads = torch.autograd.grad(value, (yp, y, *params))
                    self.assertTrue(all(g.isfinite().all() for g in grads))
                    # On a zero-pilot row the RMS floor breaks scale symmetry for a
                    # nonzero received payload. Test companion-row isolation separately.
                    if mode != "pilot_erasure":
                        moved = objective(2*yp, 2*y, 4*slot.noise_var, 2.)
                        moved_grads = torch.autograd.grad(moved, params)
                        for original, scaled in zip(grads[2:], moved_grads):
                            torch.testing.assert_close(original, scaled, atol=2e-7, rtol=2e-6)
                    if mode != "all_zero":
                        zero_weight = [torch.zeros_like(p) for p in params]
                        for b in range(2):
                            single = objective(yp[b:b+1], y[b:b+1],slot.noise_var[b:b+1],1.)
                            single_grads = torch.autograd.grad(single,(yp,y,*params))
                            for i in (0,1):
                                torch.testing.assert_close(single_grads[i][b]/2, grads[i][b], rtol=1e-5, atol=1e-7)
                            for i,g in enumerate(single_grads[2:]):
                                zero_weight[i] += g/2
                        for aggregate, direct in zip(zero_weight,grads[2:]):
                            torch.testing.assert_close(aggregate, direct, rtol=2e-5, atol=1e-6)


class Tier5CarrierPhase(unittest.TestCase):
    @staticmethod
    def score(z, truth):
        return phase_and_errors(z, truth.flatten(1), torch.ones_like(truth[0],dtype=torch.bool))

    def test_sustained_quadrant_rotation_and_isolated_sign_error(self):
        truth = torch.full((1,14,32),1+1j,dtype=torch.complex128)
        turned = self.score(truth*1j,truth)
        self.assertEqual((turned["wrong_carrier_quadrant_rows"], turned["carrier_quadrant_transitions"]), (14,0))
        point = truth.clone()
        point[0,6,3] = -1+1j
        isolated = self.score(point,truth)
        self.assertEqual((isolated["sign_errors"],isolated["wrong_symbol_quadrants"]), (1,1))
        self.assertEqual((isolated["wrong_carrier_quadrant_rows"],isolated["carrier_quadrant_transitions"]),(0,0))

    def test_symmetric_45_degree_ties_and_both_sides(self):
        # Equal real/imaginary parts generate exact angle ties without polar's rounding.
        truth = torch.ones((1,14,32),dtype=torch.complex128)
        for sign in (-1,1):
            for imag, bad in ((1.-1e-5,0),(1.,0),(1.+1e-5,14)):
                with self.subTest(sign=sign,imag=imag):
                    score = self.score(truth*complex(1,sign*imag),truth)
                    self.assertEqual(score["wrong_carrier_quadrant_rows"],bad)

    def test_zero_correlation_has_no_evidence_of_carrier_lock(self):
        truth = torch.full((1,14,32),1+1j,dtype=torch.complex128)
        z = truth.clone()
        z[:,5,::2] *= -1
        z[:,6].zero_()
        try:
            score = self.score(z,truth)
        except ValueError:
            return
        self.assertGreaterEqual(score["unobservable_rows"],2)


class Tier5ConfirmatoryGuard(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((RESULTS/"stage3_protocol_rehearsal.json").read_bytes())

    def test_missing_freeze_blocks_both_engines_before_any_fit(self):
        fake_ids = tuple(range(918001,918011))
        with tempfile.TemporaryDirectory() as tmp, patch.object(protocol,"OFFICIAL",fake_ids), \
                patch.object(protocol,"FREEZE",Path(tmp)/"missing.json"), \
                patch.object(protocol,"DEST",Path(tmp)), patch.object(protocol,"_train") as coded_fit, \
                patch.object(factorial,"_fit_and_score") as arm_fit:
            for engine in (protocol.run_confirmatory_protocol, factorial.run):
                with self.assertRaises((FileNotFoundError,ValueError)):
                    engine(fake_ids,save=True)
            coded_fit.assert_not_called()
            arm_fit.assert_not_called()
            self.assertEqual(list(Path(tmp).iterdir()),[])

    def test_boolean_only_gate_cannot_authorize_confirmation(self):
        # The destination and all would-be sources are temporary. No real freeze can be created.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gate = root/"gate.json"
            gate.write_text('{"stage":"P4-B","passed":true}',encoding="utf-8")
            for name in protocol.SOURCES:
                source = root/name
                source.parent.mkdir(exist_ok=True,parents=True)
                source.write_bytes((ROOT/name).read_bytes())
            with patch.object(protocol,"ROOT",root), patch.object(protocol,"FREEZE",root/"freeze.json"), \
                    patch.object(protocol,"_historical_digest",return_value={"file_count":230}):
                with self.assertRaises((FileNotFoundError,ValueError,RuntimeError)):
                    protocol.freeze_after_gate(gate)
            self.assertFalse((root/"freeze.json").exists())

    def test_m3_units_reject_missing_or_duplicated_named_case(self):
        from p4.stage3_qualification import check_units, original_ids
        inventory = original_ids(ROOT)["unit"]
        self.assertEqual(len(inventory),13)
        names = sorted(inventory)
        for corrupt in (names[:-1],names[:-1]+[names[0]]):
            with self.subTest(length=len(corrupt)):
                result = {"run":13,"passed_cases":corrupt,"failures":[],"errors":[],"skips":[]}
                with self.assertRaises(ValueError):
                    check_units(result,inventory)

    def test_rejects_incomplete_coded_panel_and_impossible_word_counts(self):
        budget = protocol.Budget(**self.fixture["coded_budget"])
        for kind in ("missing_cell","boolean_bits","bit_without_word","too_many_bits_per_word"):
            with self.subTest(kind=kind):
                rows = copy.deepcopy(self.fixture["coded"]["records"])
                cell = rows[0]["rows"][0]
                if kind == "missing_cell":
                    rows[0]["rows"].pop()
                elif kind == "boolean_bits":
                    cell["bit_errors"] = False
                elif kind == "bit_without_word":
                    cell.update(bit_errors=1,block_errors=0)
                else:
                    cell.update(bit_errors=65,block_errors=1)
                with self.assertRaises((TypeError,ValueError)):
                    protocol._analyze(rows,budget)

    def test_rejects_missing_factorial_cells_and_noninteger_denominators(self):
        for kind in ("missing_arm","missing_gain","float_cell_total","float_pooled_total"):
            with self.subTest(kind=kind):
                rows = copy.deepcopy(self.fixture["factorial_rows"])
                if kind == "missing_arm":
                    rows.pop()
                elif kind == "missing_gain":
                    rows[0]["gains"].pop()
                elif kind == "float_cell_total":
                    cell = rows[0]["gains"][0]
                    cell["total_bits"] = float(cell["total_bits"])
                else:
                    rows[0]["total_bits"] = float(rows[0]["total_bits"])
                with self.assertRaises((TypeError, ValueError)):
                    factorial._check(rows,self.fixture["seeds"])


def poison_labels(slot):
    return replace(slot, h_true=-slot.h_true, h_time_evol=-slot.h_time_evol, bits=1-slot.bits,
                   data_tx=-slot.data_tx, x_grid=-slot.x_grid, fd=.475, cfo=.39, profile="long12")


class Tier5BlindInference(unittest.TestCase):
    @staticmethod
    def capture(buffer):
        def hook(module, args, output):
            buffer.append((output.llrs.clone(),output.h_hat.clone()))
        return hook

    def test_sparse_workload_blind_outputs_with_poisoned_ground_truth(self):
        receiver = stage3_sparse.load_receiver()
        slot = sparse_slot(848911,4,2,20.)
        seen, scores = [], []
        with receiver.register_forward_hook(self.capture(seen)), \
                patch.object(stage3_sparse,"load_receiver", return_value=receiver):
            for data in (slot, poison_labels(slot)):
                with patch.object(stage3_sparse,"sparse_slot",return_value=data):
                    scores.append(stage3_sparse.run_ultra_sparse_workload(seed=848911))
        self.assertEqual(len(seen),2)
        self.assertTrue(all(torch.equal(v,w) for v,w in zip(*seen)))
        self.assertGreater(scores[1]["nmse_2d"],scores[0]["nmse_2d"]+1.)

    def test_fractional_ood_workload_blind_outputs_and_exact_complementary_bit_counts(self):
        receiver = TapPhaseNoChannel().eval()
        receiver.load_state_dict(torch.load(RESULTS/"next_neural_tap_phase_no_channel_73161.pt",
                                            weights_only=True)["weights"])
        original = stage3_ood.make_slot
        seen=[]
        with receiver.register_forward_hook(self.capture(seen)):
            base = stage3_ood.evaluate_fractional(receiver,848912,batches=1)
            with patch.object(stage3_ood,"make_slot",side_effect=lambda *a:poison_labels(original(*a))):
                altered = stage3_ood.evaluate_fractional(receiver,848912,batches=1)
        self.assertEqual(len(seen),2)
        self.assertTrue(all(torch.equal(v,w) for v,w in zip(*seen)))
        for method,n in base["bit_errors"].items():
            self.assertEqual(n+altered["bit_errors"][method],base["uncoded_bits"])
        self.assertGreater(altered["channel_nmse"]["model"],base["channel_nmse"]["model"]+1.)


class Tier5Precision(unittest.TestCase):
    def test_packed_projection_against_independent_rounded_real_dot_products(self):
        rng = torch.Generator().manual_seed(848913)
        layer = ComplexLinear(7,4,bias=True)
        with torch.no_grad():
            for p in layer.parameters():
                p.copy_(torch.randn(p.shape,generator=rng))
            z = torch.randn(3,5,7,generator=rng,dtype=torch.complex128).transpose(0,1)
            self.assertFalse(z.is_contiguous())
            for dtype in (torch.float32,torch.bfloat16,torch.float16):
                with self.subTest(dtype=dtype):
                    engine = stage3_precision.PackedProjection(layer,dtype)
                    with patch.object(stage3_precision.F,"linear",wraps=stage3_precision.F.linear) as matmul:
                        actual = engine(z)
                    input_,weight,bias = matmul.call_args.args
                    self.assertEqual((input_.dtype,weight.dtype,bias.dtype),(dtype,)*3)
                    cast = lambda x:x.to(dtype).double()
                    xr,xi = cast(z.real).unsqueeze(-2),cast(z.imag).unsqueeze(-2)
                    wr,wi = cast(layer.weight_r),cast(layer.weight_i)
                    rr = (xr*wr-xi*wi).sum(-1)+cast(layer.bias_r)
                    ii = (xi*wr+xr*wi).sum(-1)+cast(layer.bias_i)
                    expected = torch.complex(cast(rr),cast(ii))
                    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
                    if dtype != torch.float32:
                        self.assertGreater(float((actual-layer(z)).abs().max()),1e-4)


if __name__ == "__main__":
    unittest.main()
