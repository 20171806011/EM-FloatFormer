"""Offline first-derivative audit of a proposed forward-preserving RMS-anchor fix."""
import json
import math
from unittest.mock import patch

import torch

from p4.em_floatformer import compute_gauge_anchor
from p4.next_neural import ARMS, RESULTS
from p4.slot_2d import SlotSimulator2D, _synchronize_slot


def candidate_gauge_anchor(yp, eps=1e-38):
    yp64 = yp.to(torch.complex128)
    peak = yp64.abs().amax(dim=-1, keepdim=True)
    active = peak > 0
    scaled = yp64 / torch.where(active, peak, torch.ones_like(peak))
    power = (scaled.real.square() + scaled.imag.square()).mean(dim=-1, keepdim=True)
    # The inactive row takes the constant branch below. Keep its unused sqrt
    # differentiable to avoid multiplying a zero adjoint by an infinite derivative.
    rms = torch.where(active, power, torch.ones_like(power)).sqrt()
    return torch.where(active, (peak * rms).clamp_min(eps), torch.full_like(peak, eps)).to(yp.real.dtype)


def anchor_checks():
    results = []
    for dtype in (torch.complex64, torch.complex128):
        base = torch.tensor([1j, 2-1j, -.5+.25j, -.3j], dtype=torch.complex128)
        scales = [0, 1e-40, 1e-36, 1e-8, 1, 1e8, 1e37]
        if dtype == torch.complex128:
            scales += [1e-200, 1e200]
        z = torch.stack([base * r for r in scales]).to(dtype).requires_grad_()
        values = compute_gauge_anchor(z)
        fixed = candidate_gauge_anchor(z)
        grads, = torch.autograd.grad(values.sum(), z)
        fixed_grads, = torch.autograd.grad(fixed.sum(), z)
        if not torch.equal(values, fixed):
            raise AssertionError("Forward values differ on the mixed scale grid.")
        for j, r in enumerate(scales):
            # Derive the exact gradient from z/(N*rms) outside the floor,
            # working in scale-free coordinates to independently avoid overflow.
            row = z[j].detach().to(torch.complex128)
            if r == 0 or (row.abs().max() <= 1e-38):
                analytic = torch.zeros_like(row)
            else:
                unit = row / r
                analytic = unit / (len(row) * torch.linalg.vector_norm(unit) / math.sqrt(len(row)))
            max_error = float((fixed_grads[j].to(torch.complex128) - analytic).abs().max())
            if max_error > (1e-7 if dtype == torch.complex64 else 1e-12):
                raise AssertionError((dtype, r, max_error))
            results.append({"dtype": str(dtype), "scale": r, "anchor": float(fixed[j, 0].detach()),
                            "original_gradient_finite": bool(grads[j].isfinite().all()),
                            "proposed_gradient_max_error_from_analytic": max_error,
                            "nonzero_row_gradient_bit_equal": torch.equal(grads[j], fixed_grads[j]) if r != 0 else None})
        z0 = torch.zeros((3, 4), dtype=dtype, requires_grad=True)
        baseline = candidate_gauge_anchor(z0)
        d0, = torch.autograd.grad(baseline.sum(), z0)
        # A nonzero finite displacement entirely inside the epsilon ball.
        for h in (1e-40, -1e-40, 1e-40j, -1e-40j):
            if not torch.equal(baseline, candidate_gauge_anchor(z0 + h)):
                raise AssertionError("The proposed anchor must be locally constant at the origin.")
        if not torch.equal(d0, torch.zeros_like(d0)):
            raise AssertionError("The derivative at the origin must be exactly zero.")
    return results


def model_pass(model, slot, mode, anchor):
    yp = slot.yp_sparse.to(torch.complex128).clone()
    y = slot.y_grid.to(torch.complex128).clone()
    nv = slot.noise_var.double().clone()
    scale = 1.0
    if mode.startswith("all_zero"):
        yp.zero_()
        y.zero_()
        if mode == "all_zero_no_noise":
            nv.zero_()
    elif mode == "mixed":
        yp[0].zero_()
        y[0].zero_()
    elif mode == "pilot_erasure":
        yp[0].zero_()
    elif mode.startswith("scale"):
        scale = float(mode.split("_")[-1])
        yp *= scale
        y *= scale
        nv *= scale**2
    yp.requires_grad_()
    y.requires_grad_()
    params = dict(model.named_parameters())
    with patch("p4.slot_2d.compute_gauge_anchor", anchor):
        out = model(yp, y, slot.pilot_mask, nv)
        # Fixed projections give nontrivial adjoints even when a head output is zero.
        scalar = (out.h_hat.real.mean() + 0.3*out.dist_est.imag.mean()) / scale + out.llrs[..., 0].mean()
        grads = torch.autograd.grad(scalar, (yp, y, *params.values()), allow_unused=True)
    return (out.h_hat.detach(), out.dist_est.detach(), out.llrs.detach()), grads[:2], dict(zip(params, grads[2:]))


def run():
    torch.set_num_threads(1)
    rows = anchor_checks()
    checks = []
    sim = SlotSimulator2D(profile="long12", fd=.012, cfo=.01)
    slot = _synchronize_slot(sim.generate_slot(batch_size=2, snr_db=24, pa_condition="mild",
                                              generator=torch.Generator().manual_seed(847012)))
    for arm, factory in ARMS.items():
        model = factory().eval()
        ckpt = RESULTS / f"next_neural_{arm}_73161.pt"
        model.load_state_dict(torch.load(ckpt, weights_only=True)["weights"])
        for mode in ("all_zero_no_noise", "all_zero_noise", "mixed", "pilot_erasure",
                     "ordinary", "scale_1e-8", "scale_1e8"):
            original, old_inputs, old_w = model_pass(model, slot, mode, compute_gauge_anchor)
            fixed, new_inputs, new_w = model_pass(model, slot, mode, candidate_gauge_anchor)
            if not all(torch.equal(a, b) for a, b in zip(original, fixed)):
                raise AssertionError((arm, mode, "receiver outputs differ"))
            valid_slice = mode in ("mixed", "pilot_erasure")
            nonzero_rows_identical = all(torch.equal(a[1:], b[1:]) for a, b in zip(old_inputs, new_inputs)) if valid_slice else None
            if valid_slice and not nonzero_rows_identical:
                raise AssertionError((arm, mode, "companion-row gradient changed"))
            bad_old = [n for n,g in old_w.items() if g is None or not torch.isfinite(g).all()]
            bad_new = [n for n,g in new_w.items() if g is None or not torch.isfinite(g).all()]
            all_weights_equal = all(a is not None and torch.equal(a, new_w[n]) for n,a in old_w.items())
            if mode in ("ordinary", "scale_1e-8", "scale_1e8"):
                if not all(torch.equal(a, b) for a,b in zip(old_inputs, new_inputs)) or not all_weights_equal:
                    raise AssertionError((arm, mode, "nonzero reference gradients changed"))
            checks.append({"arm": arm, "case": mode, "outputs_bit_equal": True,
                           "all_outputs_finite": all(torch.isfinite(v).all().item() for v in fixed),
                           "original_nonfinite_input_entries": [int((~torch.isfinite(v)).sum()) for v in old_inputs],
                           "proposed_nonfinite_input_entries": [int((~torch.isfinite(v)).sum()) for v in new_inputs],
                           "nonzero_companion_row_input_gradients_bit_equal": nonzero_rows_identical,
                           "parameter_tensors": len(old_w), "original_bad_or_missing_parameter_grads": bad_old,
                           "proposed_bad_or_missing_parameter_grads": bad_new, "all_parameter_gradients_bit_equal": all_weights_equal})
    return {"execution": "Isolated substitution of p4.slot_2d.compute_gauge_anchor, with automatic restoration after each forward/backward. CPU, complex128 receiver inputs, training seed 73161, weights frozen for measurement. No default implementation altered.",
            "input_gradient_order": ["pilot_tensor", "received_grid"],
            "scope": "An exactly zero input derivative is analytically required for the anchor itself inside its floor. Receiver derivatives also include direct pilot and data paths.",
            "anchor_rows": rows, "receivers": checks}


if __name__ == "__main__":
    result = run()
    (RESULTS / "next_anchor_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    for r in result["receivers"]:
        print(r["arm"], r["case"], r["original_nonfinite_input_entries"], r["proposed_nonfinite_input_entries"],
              len(r["proposed_bad_or_missing_parameter_grads"]), r["all_parameter_gradients_bit_equal"])
