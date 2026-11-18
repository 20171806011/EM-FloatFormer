"""Independent initial-gradient and trained C* audit for pair 1 in each regime."""
from dataclasses import replace
import math

import torch
from torch.nn import functional as F

from p5.study import ARMS, FIT_BATCH, RESULTS, frozen, rng, save, setup
from p5.unrolled_em_transformer import UnrolledEMTransformer
from p5.wavefield_sim import simulate


def real_view(tensor):
    return torch.view_as_real(tensor) if tensor.is_complex() else tensor


def objective(model, frame):
    llrs, _ = model(frame.obs)
    return F.binary_cross_entropy_with_logits(llrs[:, ~frame.obs.pilot_mask], frame.bits.double())


def initial_gradient_audit(model, frame, direction_seed):
    generator = torch.Generator().manual_seed(direction_seed)
    model.zero_grad(set_to_none=True)
    loss = objective(model, frame)
    loss.backward()
    rows = []
    epsilon = 1.e-5
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        grad = real_view(parameter.grad)
        values = real_view(parameter)
        direction = torch.randn(values.shape, dtype=values.dtype, generator=generator)
        direction /= torch.linalg.vector_norm(direction)
        analytic = float((direction * grad).sum())
        # Probe a preselected random direction in each tensor's entire real parameter space.
        with torch.no_grad():
            initial = values.clone()
            try:
                values.copy_(initial + epsilon * direction)
                plus = float(objective(model, frame))
                values.copy_(initial - epsilon * direction)
                minus = float(objective(model, frame))
            finally:
                values.copy_(initial)
        numerical = (plus - minus) / (2 * epsilon)
        rows.append(dict(parameter=name, real_dofs=grad.numel(),
                         real_gradient_entries_nonzero=int(torch.count_nonzero(grad)),
                         gradient_l2=float(grad.norm()), analytic_directional=analytic,
                         finite_difference=numerical, absolute_error=abs(analytic - numerical),
                         passed=bool(grad.isfinite().all() and grad.abs().max() > 0)
                         and math.isclose(analytic, numerical, abs_tol=1.e-7, rel_tol=1.e-5)))
    return dict(bce=float(loss.detach()), direction_rng_seed=direction_seed,
                difference_half_width=epsilon, tensors=rows)


@torch.no_grad()
def gauge_audit(model, frame, phase_seed):
    generators = torch.Generator().manual_seed(phase_seed)
    scales = torch.tensor([1.e-35, *(2.**e for e in range(-12, 13)), 1.e20], dtype=torch.float64)
    obs = frame.obs
    original, h = model(obs)
    per_burst = torch.arange(len(obs.y))
    payload_mask = ~obs.pilot_mask
    rows = []
    for step in range(len(scales)):
        # Offset each row independently. Every burst experiences every scale exactly once.
        scale_idx = (per_burst + step) % len(scales)
        magnitude = scales[scale_idx]
        phase = torch.rand(len(obs.y), dtype=torch.float64, generator=generators) * (2 * math.pi)
        multiplier = torch.polar(magnitude, phase)
        observed_gain = multiplier[:, None, None, None]
        changed = replace(obs, y=observed_gain * obs.y, noise_var=obs.noise_var * magnitude.square())
        current, predicted_h = model(changed)
        error = predicted_h / observed_gain - h
        flips = ((current[:, payload_mask] > 0) != (original[:, payload_mask] > 0)).flatten(1).sum(-1)
        differences = (current - original).abs().flatten(1).amax(-1)
        channel_errors = error.abs().flatten(1).amax(-1)
        relative_l2 = error.flatten(1).norm(dim=-1) / h.flatten(1).norm(dim=-1)
        for i in range(len(obs.y)):
            rows.append(dict(burst=i, magnitude=float(magnitude[i]), phase_rad=float(phase[i]),
                             coded_bit_flips=int(flips[i]), coded_bits=frame.bits[i].numel(),
                             max_grid_llr_difference=float(differences[i]),
                             max_derotated_descaled_cfr_difference=float(channel_errors[i]),
                             relative_cfr_l2=float(relative_l2[i])))
    return dict(phase_rng_seed=phase_seed, transformations=rows,
                total_coded_bits=sum(row["coded_bits"] for row in rows),
                total_coded_bit_flips=sum(row["coded_bit_flips"] for row in rows),
                max_grid_llr_difference=max(row["max_grid_llr_difference"] for row in rows),
                max_derotated_descaled_cfr_difference=max(
                    row["max_derotated_descaled_cfr_difference"] for row in rows),
                max_relative_cfr_l2=max(row["relative_cfr_l2"] for row in rows))


def audit(regime):
    pair = 1
    seed = rng(regime, 11, pair)
    snr_db = frozen(regime)
    frame = simulate(regime, 8, snr_db, seed)
    training_seed = rng(regime, 5, pair, 0)
    training_frame = simulate(regime, FIT_BATCH, snr_db, training_seed)
    torch.manual_seed(rng(regime, 4, pair))
    model = UnrolledEMTransformer().eval()
    with torch.no_grad():
        physics = UnrolledEMTransformer(variant="physics")(frame.obs)
        initial = model(frame.obs)
        equality = dict(llrs=torch.equal(initial[0], physics[0]), cfr=torch.equal(initial[1], physics[1]))
    gradients = initial_gradient_audit(model, training_frame, rng(regime, 12, pair))
    checkpoint = RESULTS / f"r{regime}" / "pair01" / "full.pt"
    state = torch.load(checkpoint, weights_only=True, map_location="cpu")
    changed_trainable = []
    frozen_reference_unchanged = True
    for name, parameter in model.named_parameters():
        differs = not torch.equal(parameter, state[name])
        if parameter.requires_grad and differs:
            changed_trainable.append(name)
        if not parameter.requires_grad and differs:
            frozen_reference_unchanged = False
    model.load_state_dict(state)
    gauge = gauge_audit(model, frame, rng(regime, 13, pair))
    passed = (all(equality.values()) and all(row["passed"] for row in gradients["tensors"])
              and frozen_reference_unchanged and changed_trainable and gauge["total_coded_bit_flips"] == 0
              and gauge["max_grid_llr_difference"] < 1.e-5 and gauge["max_relative_cfr_l2"] < 1.e-9)
    return dict(regime=regime, test_pair=pair, checkpoint=checkpoint.relative_to(RESULTS.parent).as_posix(),
                frame_seed=seed, first_training_seed=training_seed,
                solver_equal_at_initialization=equality, first_bce_gradient=gradients,
                frozen_reference_unchanged=frozen_reference_unchanged, changed_trainable=changed_trainable,
                trained_gauge=gauge, passed=bool(passed))


def main():
    setup()
    sizes = {}
    for variant in (*ARMS, "physics"):
        params = list(UnrolledEMTransformer(variant=variant).parameters())
        sizes[variant] = {group: sum(real_view(p).numel() for p in params if p.requires_grad == active)
                          for group, active in (("trainable_real_dofs", True), ("frozen_real_dofs", False))}
    rows = [audit(regime) for regime in (1, 2, 3)]
    expected = sizes["full"]["trainable_real_dofs"]
    matched = all(sizes[variant]["trainable_real_dofs"] == expected for variant in ARMS)
    summary = dict(torch_version=torch.__version__, arms=sizes,
                   learned_controls_parameter_matched=matched, regimes=rows,
                   passed=matched and all(row["passed"] for row in rows))
    save(RESULTS / "model_audit.json", summary)
    for row in rows:
        gauge = row["trained_gauge"]
        print(f"R{row['regime']} pass={row['passed']}, flips={gauge['total_coded_bit_flips']}"
              f"/{gauge['total_coded_bits']}, max |dLLR|={gauge['max_grid_llr_difference']:.3g},"
              f" max relative CFR L2={gauge['max_relative_cfr_l2']:.3g}")
    if not summary["passed"]:
        raise RuntimeError("model audit failed; see results/model_audit.json")


if __name__ == "__main__":
    main()
