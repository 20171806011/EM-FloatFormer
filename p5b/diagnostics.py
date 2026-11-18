"""CPU preflight and final checkpoint numerical symmetry audit."""

import argparse
from dataclasses import replace
import hashlib
import math
from pathlib import Path

import torch

from p5.audit_model import initial_gradient_audit
from p5.study import bit_loss, rng, save, setup
from p5.wavefield_sim import simulate
from p5b.operator_em_transformer import OperatorEMTransformer


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SNR = {1: 12, 2: 9, 3: 7}
ARMS = ("pure", "parity", "local", "local_parity", "no_gauge")
SCALES = (1e-35, *(2. ** i for i in range(-12, 13)), 1e20)


def model_sizes(model):
    return {group: sum(p.numel() * (1 + int(p.is_complex())) for p in model.parameters()
                       if p.requires_grad == live)
            for group, live in (("trainable_real", True), ("frozen_real", False))}


def erased(obs, which):
    y = obs.y.clone()
    if which == "all_pilots":
        y[:, :, obs.pilot_mask] = 0.
    elif which == "whole_field":
        y.zero_()
    elif which != "intact":
        rows = torch.where(obs.pilot_mask.any(-1))[0]
        row = rows[0 if which == "first_pilot_row" else -1]
        y[:, :, row, obs.pilot_mask[row]] = 0.
    return replace(obs, y=y)


@torch.no_grad()
def gain_check(model, obs, scales, phase_seed):
    phase_rng = torch.Generator().manual_seed(phase_seed)
    llr, channel = model(obs)
    llr_max = cfr_rel = 0.
    flips = total = 0
    finite = bool(torch.isfinite(llr).all() and torch.isfinite(channel).all())
    for scale in scales:
        phase = 2 * math.pi * torch.rand(len(obs.y), dtype=torch.float64, generator=phase_rng)
        gain = torch.polar(torch.full_like(phase, scale), phase)[:, None, None, None]
        changed = replace(obs, y=gain * obs.y, noise_var=(scale ** 2) * obs.noise_var)
        new_llr, new_channel = model(changed)
        finite &= bool(torch.isfinite(new_llr).all() and torch.isfinite(new_channel).all())
        llr_max = max(llr_max, float((llr - new_llr).abs().max()))
        error = ((new_channel / gain) - channel).flatten(1).norm(dim=1)
        baseline = channel.flatten(1).norm(dim=1)
        cfr_rel = max(cfr_rel, float((error / torch.where(baseline > 0, baseline, 1.)).max()))
        different = (llr[:, ~obs.pilot_mask] > 0) != (new_llr[:, ~obs.pilot_mask] > 0)
        flips += int(different.sum())
        total += different.numel()
    return dict(coded_flips=flips, decisions=total, max_grid_llr_diff=llr_max,
                max_relative_cfr_l2=cfr_rel, phase_seed=phase_seed, scales=list(scales),
                finite=finite, passed=finite and flips == 0 and llr_max < 1e-12 and cfr_rel < 1e-12)


def gauge_suite(model, frame, phase_seed):
    # Full gain grid on intact data and both extremes plus unit radius after erasures.
    result = {name: gain_check(model, erased(frame.obs, name),
                               SCALES if name == "intact" else (1e-35, 1., 1e20),
                               phase_seed + i)
              for i, name in enumerate(("intact", "first_pilot_row", "last_pilot_row",
                                        "all_pilots", "whole_field"))}
    zero = erased(frame.obs, "whole_field")
    with torch.no_grad():
        llr1, h1 = model(zero)
        llr2, h2 = model(replace(zero, noise_var=torch.zeros_like(zero.noise_var)))
    zeros = all(bool(torch.isfinite(tensor).all()) and not bool(tensor.count_nonzero())
                for tensor in (llr1, h1, llr2, h2))
    return dict(conditions=result, exact_zero_with_and_without_noise=zeros,
                passed=zeros and all(x["passed"] for x in result.values()))


@torch.no_grad()
def degenerate_pilot_audit(model, regime):
    # Construct an antipodal R1 phase, mirror-image Dopplers and numerically
    # null delay taps for R2/R3. The payload observations retain ordinary noise.
    obs = simulate(regime, 2, SNR[regime], 22).obs
    y = obs.y.clone()
    if regime == 1:
        y[:, :, -1, ::2] = -y[:, :, 0, ::2] / obs.pilots[0, ::2] * obs.pilots[-1, ::2]
    else:
        y[:, :, obs.pilot_mask] = obs.pilots[obs.pilot_mask]
        if regime == 3:
            y[:, :, -1, ::2] *= -1.
    obs = replace(obs, y=y)
    llr, channel = model(obs)
    rows = []
    for gain in (.36 + .49j, .3 + .9j, 1j, 1e20 * (.6 + .8j)):
        llr2, channel2 = model(replace(obs, y=y * gain, noise_var=obs.noise_var * abs(gain)**2))
        difference = ((llr[:, ~obs.pilot_mask] > 0) != (llr2[:, ~obs.pilot_mask] > 0))
        llr_error = float((llr2 - llr).abs().max())
        cfr_error = float((channel2 / gain - channel).abs().max())
        flips = int(difference.sum())
        rows.append(dict(gain=[gain.real, gain.imag], decisions=difference.numel(), coded_flips=flips,
                         max_grid_llr_diff=llr_error, max_cfr_absolute_diff=cfr_error,
                         passed=flips == 0 and llr_error < 1e-12 and cfr_error < 1e-12))
    return dict(regime=regime, seed=22, transformations=rows, passed=all(r["passed"] for r in rows))


def source_identity():
    return {filename: hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
            for filename in ("operator_em_transformer.py", "parity.py")}


def initial_audit(regime, variant):
    model_seed = rng(regime, 4, 1)
    torch.manual_seed(model_seed)
    model = OperatorEMTransformer(variant=variant)
    train_seed, test_seed = rng(regime, 5, 1, 0), rng(regime, 11, 1)
    train = simulate(regime, 4, SNR[regime], train_seed)
    test = simulate(regime, 2, SNR[regime], test_seed)
    physical_variant = "physics_parity" if variant in ("parity", "local_parity") else "physics"
    anchor = OperatorEMTransformer(variant=physical_variant)
    with torch.no_grad():
        base_llr, base_h = anchor(test.obs)
        llr, h = model(test.obs)
        delta_llr, delta_h = float((llr - base_llr).abs().max()), float((h - base_h).abs().max())
    anchored = torch.equal(llr, base_llr) and torch.equal(h, base_h)
    # The no-gauge physical path processes unscaled numbers. Its mathematically
    # equivalent physical anchor can have floating-point rounding differences.
    anchor_ok = (delta_llr < 1e-12 and delta_h < 1e-12) if variant == "no_gauge" else anchored
    grad = initial_gradient_audit(model, train, rng(regime, 12, 1))
    gradient_ok = all(t["passed"] for t in grad["tensors"])
    opt = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.003)
    # initial_gradient_audit preserves analytical gradients and restores the weights.
    opt.step()
    with torch.no_grad():
        learned_difference = float((model(test.obs)[0] - llr).abs().max())
    gauge = (gauge_suite(model, test, rng(regime, 13, 1)) if variant != "no_gauge"
             else None)
    return dict(regime=regime, variant=variant, model_seed=model_seed,
                training_seed=train_seed, audit_seed=test_seed, sizes=model_sizes(model),
                anchor=physical_variant, bit_equal_at_init=anchored, init_max_llr_diff=delta_llr,
                init_max_cfr_diff=delta_h, first_gradient=grad,
                max_llr_changed_after_one_adam_step=learned_difference, gauge=gauge,
                passed=bool(anchor_ok and gradient_ok and learned_difference > 0
                            and (gauge is None or gauge["passed"])))


def preflight():
    rows = [initial_audit(regime, arm) for regime in SNR for arm in ARMS]
    ties = [degenerate_pilot_audit(OperatorEMTransformer(variant="physics"), regime) for regime in SNR]
    counts = {row["sizes"]["trainable_real"] for row in rows}
    passed = (len(counts) == 1 and 800 <= next(iter(counts)) <= 2500 and
              all(r["passed"] for r in rows) and all(t["passed"] for t in ties))
    return dict(sources=source_identity(), rows=rows, degenerate_pilots=ties, passed=passed)


def trained_checkpoints():
    rows = []
    for regime, pairs in ((1, 10), (2, 5), (3, 5)):
        for pair in range(1, pairs + 1):
            seed = rng(regime, 11, pair)
            frame = simulate(regime, 2, SNR[regime], seed)
            for variant in ("pure", "parity"):
                path = RESULTS / f"r{regime}" / f"pair{pair:02d}" / f"{variant}.pt"
                model = OperatorEMTransformer(variant=variant)
                model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
                gauge = gauge_suite(model.eval(), frame, rng(regime, 13, pair))
                rows.append(dict(regime=regime, pair=pair, variant=variant, audit_seed=seed,
                                 checkpoint=path.relative_to(ROOT).as_posix(), gauge=gauge))
    return dict(sources=source_identity(), rows=rows, passed=all(r["gauge"]["passed"] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", action="store_true")
    args = parser.parse_args()
    setup()
    result = trained_checkpoints() if args.checkpoints else preflight()
    path = RESULTS / ("trained_gauge.json" if args.checkpoints else "preflight.json")
    save(path, result)
    print(f"{path}: passed={result['passed']}")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
