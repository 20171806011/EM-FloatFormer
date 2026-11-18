"""Development-only checks of the initialized and once-updated receiver."""
from dataclasses import replace
import math
from pathlib import Path

import torch

from p5.audit_model import real_view
from p5.study import save, setup
from p5b.diagnostics import erased, gain_check, gauge_suite, model_sizes
from p5b.operator_em_transformer import OperatorEMTransformer
from p6.axial_em_transformer import AxialEMTransformer
from p6.data import SNR, dev_frame, dev_seed

ARMS = ("pure", "local", "parity", "local_parity", "no_gauge")
CENTRED_ATOL = 1.e-12


@torch.no_grad()
def anchor_check(model, obs, variant, steps):
    reference_name = "physics_parity" if variant in ("parity", "local_parity") else "physics"
    p6_fixed = AxialEMTransformer(variant=reference_name, steps=steps)
    archived = OperatorEMTransformer(variant=reference_name, steps=steps)
    learned_outputs = model(obs)
    fixed_outputs = p6_fixed(obs)
    archived_outputs = archived(obs)
    fixed_errors = [float((a - b).abs().max()) for a, b in zip(learned_outputs, fixed_outputs)]
    archived_errors = [float((a - b).abs().max()) for a, b in zip(learned_outputs, archived_outputs)]
    exact = all(torch.equal(a, b) for a, b in zip(learned_outputs, fixed_outputs))
    return dict(steps=steps, p6_bit_exact=exact, fixed_max_llr_cfr=fixed_errors,
                archived_max_llr_cfr=archived_errors,
                passed=exact and max(archived_errors) < CENTRED_ATOL)


def degenerate(obs):
    y = obs.y.clone()
    if obs.regime == 1:
        y[:, :, -1, ::2] = -y[:, :, 0, ::2] / obs.pilots[0, ::2] * obs.pilots[-1, ::2]
    else:
        y[:, :, obs.pilot_mask] = obs.pilots[obs.pilot_mask]
        if obs.regime == 3:
            y[:, :, -1, ::2] *= -1.
    return replace(obs, y=y)


def gradient_audit(model, frame, seed):
    """One fixed random direction in each tensor and a central-difference ladder.

    The radial bound u/(1+|u|) is differentiable at the centered origin,
    but its central-difference derivative there has first-order truncation
    error. Preserve the coarse estimate and use the same small-step limit
    and acceptance tolerances for every arm and regime.
    """
    from p6.run_dev_iter4 import objective

    gen = torch.Generator().manual_seed(seed)
    model.zero_grad(set_to_none=True)
    loss, raw_bce = objective(frame, model)
    loss.backward()
    rows = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grad, values = real_view(param.grad), real_view(param)
        direction = torch.randn(values.shape, dtype=values.dtype, generator=gen)
        direction /= direction.norm()
        analytic = float((grad * direction).sum())
        estimates = []
        with torch.no_grad():
            initial = values.clone()
            try:
                for step in (1.e-5, 1.e-6, 1.e-7):
                    values.copy_(initial + step * direction)
                    plus = float(objective(frame, model)[0])
                    values.copy_(initial - step * direction)
                    minus = float(objective(frame, model)[0])
                    numerical = (plus - minus) / (2 * step)
                    estimates.append(dict(half_width=step, finite_difference=numerical,
                                          absolute_error=abs(numerical - analytic)))
            finally:
                values.copy_(initial)
        fine = estimates[-1]
        rows.append(dict(parameter=name, real_dofs=grad.numel(),
                         real_gradient_entries_nonzero=int(grad.count_nonzero()),
                         gradient_l2=float(grad.norm()), analytic_directional=analytic,
                         estimates=estimates, finite_difference=fine["finite_difference"],
                         absolute_error=fine["absolute_error"],
                         passed=bool(grad.isfinite().all() and grad.count_nonzero() == grad.numel())
                         and math.isclose(analytic, fine["finite_difference"], abs_tol=1.e-7, rel_tol=1.e-5)))
    return dict(objective=float(loss.detach()), bce=float(raw_bce.detach()),
                direction_rng_seed=seed, tensors=rows)


def audit(regime, variant, pair=901):
    train = dev_frame(regime, 5, pair, 4)
    val = dev_frame(regime, 6, pair, 2)
    torch.manual_seed(dev_seed(regime, 5, pair, 900))
    model = AxialEMTransformer(variant=variant, steps=3)
    depth2 = AxialEMTransformer(variant=variant, steps=2)
    depth2.load_state_dict(model.state_dict())
    anchors = [anchor_check(m, val.obs, variant, m.steps) for m in (depth2, model)]
    grad = gradient_audit(model, train, dev_seed(regime, 5, pair, 940))
    active = sum(r["real_gradient_entries_nonzero"] for r in grad["tensors"])
    params = model_sizes(model)
    good_gradient = active == params["trainable_real"] and all(r["passed"] for r in grad["tensors"])
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.003)
    with torch.no_grad():
        before = model(val.obs)[0]
    opt.step()
    model.eval()
    with torch.no_grad():
        after = model(val.obs)[0]
        changed = float((after - before).abs().max())
    if variant == "no_gauge":
        with torch.no_grad():
            scaled = model(replace(val.obs, y=3 * val.obs.y, noise_var=9 * val.obs.noise_var))[0]
        effect = float((scaled - after).abs().max())
        gauge = dict(scale3_max_llr_diff=effect, passed=math.isfinite(effect) and effect > 1.e-9)
    else:
        phase_seed = dev_seed(regime, 6, pair, 945)
        gauge = gauge_suite(model, val, phase_seed)
        # A truly noiseless waveform, and a missing receiver noise estimate on
        # the original noisy observation, exercise different solver conditions.
        noiseless = dev_frame(regime, 6, pair, 2, snr=math.inf).obs
        nv0 = replace(val.obs, noise_var=torch.zeros_like(val.obs.noise_var))
        edge = {"degenerate_pilot": degenerate(val.obs), "noiseless": noiseless,
                "zero_reported_noise": nv0, "all_pilots_erased_zero_noise": erased(nv0, "all_pilots")}
        for i, (name, obs) in enumerate(edge.items()):
            gauge["conditions"][name] = gain_check(model, obs, (1.e-35, 1., 1.e20), phase_seed + 30 + i)
        gauge["passed"] &= all(v["passed"] for v in gauge["conditions"].values())
    return dict(regime=regime, pair=pair, arm=variant, tested_depth=model.steps, sizes=params, anchors=anchors,
                gradient=grad, active_real_parameters=active, first_update_max_llr_change=changed,
                gauge=gauge, passed=bool(all(a["passed"] for a in anchors) and good_gradient
                                        and changed > 0 and gauge["passed"]))


def run():
    setup()
    rows = []
    for regime in SNR:
        for arm in ARMS:
            row = audit(regime, arm)
            print(f"R{regime} {arm} active {row['active_real_parameters']}/{row['sizes']['trainable_real']}"
                  f" passed={row['passed']}", flush=True)
            rows.append(row)
    sizes = {tuple(row["sizes"].values()) for row in rows}
    return dict(rows=rows, matched_capacity=len(sizes) == 1,
                passed=len(sizes) == 1 and all(row["passed"] for row in rows))


if __name__ == "__main__":
    result = run()
    save(Path(__file__).parent / "results/iter4_checks/invariants.json", result)
    raise SystemExit(0 if result["passed"] else 1)
