"""Run from the repository root. The optional NPZ retains each four-cell probability cube."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

import numpy as np

sys.path[:0] = [str(Path.cwd()), str(Path.cwd() / "research/matched_cache")]
import learned
from research.stage3_ministral_cache import data

CAL = tuple(range(988001, 988009))
TEST = (989011, 989023, 989039, 989063)
SIGN = np.array([-1, 1])
WEIGHT = Path("research/matched_cache/weights/equivariant_b8_s17.npz")
SHA = "e67d9fbafb90791ab19549de7789798d818286d7f52d7ac566cc39df71e78356"


def worlds(seed, blocks, encoder):
    name = "C4_cfo_ramp"
    condition = data.CONDITION_SPECS[name]
    assert (condition.h, condition.ell, condition.sigma_phi, condition.snr_db) == (2, 2, .1, 2.)
    with patch.dict(data.SPLIT_SEEDS, {"work_f": (seed, blocks)}):
        batches = []
        for sign in SIGN:
            with patch.dict(data.CONDITION_SPECS, {name: replace(condition, omega=.35 * sign)}):
                batches.append(data.generate_mothers("work_f", name))
    neg, pos = batches
    for field in ("x1", "x2", "g", "x3_indep", "y1", "p_indep"):
        np.testing.assert_array_equal(getattr(neg, field), getattr(pos, field))
    for time, phase, window, symbols in (
        (2, "phi_2", "y2", neg.x2),
        (4, "phi_3", "y3_repeat", neg.x1),
        (4, "phi_3", "y3_indep", neg.x3_indep),
        (condition.t_p, "phi_p", "p_valid1", np.zeros_like(neg.g)),
        (condition.t_p, "phi_p", "p_valid2", np.zeros_like(neg.g)),
    ):
        carrier = np.array([getattr(b, phase) for b in batches])
        np.testing.assert_allclose(carrier[1] - carrier[0], .70 * time, rtol=0, atol=2e-14)
        signal = data.ROOTS[(symbols + neg.g) % 4] / np.sqrt(data.N_SAMPLES)
        noise = np.array([getattr(b, window) for b in batches]) - (
            np.exp(1j * carrier) * signal
        )[..., None]
        np.testing.assert_allclose(noise[0], noise[1], rtol=0, atol=2e-14)
    cache = np.array([16 * encoder.encode(b.y1) + encoder.encode(b.y2) for b in batches])
    target = np.array([b.k3_repeat for b in batches])
    return cache, target


def calibrate(encoder):
    counts = np.zeros((2, 64), dtype=int)
    mass = np.zeros((2, 256), dtype=int)
    for seed in CAL:
        caches, labels = worlds(seed, 128, encoder)
        for i, (cache, target) in enumerate(zip(caches, labels)):
            z1, z2 = np.divmod(cache, 16)
            orbit_bin = 16 * (z1 % 4) + 4 * (z2 % 4) + (target - z1 // 4) % 4
            counts[i] += np.bincount(orbit_bin, minlength=64)
            mass[i] += np.bincount(cache, minlength=256)
    z1, z2 = np.divmod(np.arange(256), 16)
    bins = 16 * (z1[:, None] % 4) + 4 * (z2[:, None] % 4)
    bins = bins + (np.arange(4) - z1[:, None] // 4) % 4
    reference = counts[:, bins] + .5
    return reference / reference.sum(-1, keepdims=True), mass / mass.sum(-1, keepdims=True)


def inner(a, b, weights):
    return np.dot(weights, np.einsum("zk,zk->z", a, b))


def audit(table, reference, weights, cache, target, draws):
    # cache and target axes: seed, physical sign, mother. Cube adds told sign and sector.
    cube = np.stack([table[t][cache] for t in range(2)], axis=2)
    np.testing.assert_allclose(cube.sum(-1), 1., rtol=0, atol=1e-12)
    assert (cube > 1e-12).all()
    cell = np.take_along_axis(cube, target[:, :, None, :, None], axis=-1)[..., 0]
    matched = np.stack([cell[:, s, s] for s in range(2)], axis=1)
    swapped = np.stack([cell[:, s, 1 - s] for s in range(2)], axis=1)
    log_ratio = np.log(matched / swapped)
    blocks = log_ratio.mean(1).reshape(len(TEST), 16, 64).mean(-1)
    resamples = blocks[np.arange(len(TEST))[:, None, None], draws].mean(axis=(0, 2))
    interval = np.quantile(resamples, [.025, .975]).tolist()
    ties = cube == cube.max(-1, keepdims=True)
    hit = np.take_along_axis(ties, target[:, :, None, :, None], axis=-1)[..., 0] / ties.sum(-1)

    basis = np.array([np.ones((2, 2)), np.tile(SIGN, (2, 1)),
                      np.tile(SIGN[:, None], (1, 2)), np.outer(SIGN, SIGN)])
    coefficients = np.einsum("ast,istmk->aimk", basis, cube) / 4
    np.testing.assert_allclose(np.einsum("ast,aimk->istmk", basis, coefficients),
                               cube, rtol=0, atol=1e-15)

    contrast = table[1] - table[0]
    direction = reference[1] - reference[0]
    norm = inner(direction, direction, weights)
    gain = inner(contrast, direction, weights) / norm
    rotation = (np.arange(256) + 64) % 256
    rotation_tv = abs(table[:, rotation] - np.roll(table, 1, axis=-1)).sum(-1).mean(0) / 2
    result = {
        "orientation_gain": gain,
        "residual_ratio": np.sqrt(inner(contrast - gain * direction, contrast - gain * direction, weights) / norm),
        "mean_sign_TV": weights @ abs(contrast).sum(-1) / 2,
        "mean_rotation_TV": weights @ rotation_tv,
        "reference_odd_norm_squared": norm,
        "matched_accuracy": np.mean([hit[:, s, s] for s in range(2)]),
        "swapped_accuracy": np.mean([hit[:, s, 1 - s] for s in range(2)]),
        "swapped_minus_matched_nll": [log_ratio.mean(), *interval],
        "factorial_mean_squared_norm": dict(zip(
            ("metadata", "wave", "interaction"), (coefficients[1:] ** 2).sum(-1).mean((1, 2))
        )),
    }
    detail = {
        "matched_nll_by_world": (-np.log(matched).mean((0, 2))).tolist(),
        "swapped_nll_by_world": (-np.log(swapped).mean((0, 2))).tolist(),
        "direct_log_ratio_by_world": log_ratio.mean((0, 2)).tolist(),
        "first_mother_probabilities": cube[0, :, :, 0].tolist(),
        "first_mother_log_ratio_by_world": log_ratio[0, :, 0].tolist(),
    }
    return result, detail, cube


def numeric_leaves(node):
    if isinstance(node, dict):
        for key in sorted(node):
            yield from numeric_leaves(node[key])
    elif isinstance(node, list):
        for entry in node:
            yield from numeric_leaves(entry)
    else:
        yield float(node)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probabilities", type=Path)
    args = parser.parse_args()
    assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest() == SHA
    encoder = learned.load(WEIGHT)
    ref, masses = calibrate(encoder)
    overlap = masses.min(axis=0)
    weights = overlap / overlap.sum()
    cache, target = (np.stack(arrays) for arrays in zip(*(worlds(seed, 16, encoder) for seed in TEST)))
    rng = np.random.default_rng(989997)
    draws = rng.integers(16, size=(len(TEST), 2000, 16))

    direction = ref[1] - ref[0]
    contrast = np.broadcast_to([1., -1., 0., 0.], direction.shape)
    orthogonal = contrast - inner(contrast, direction, weights) / inner(direction, direction, weights) * direction
    orthogonal *= .15 / abs(orthogonal).max()
    bias = np.broadcast_to([.4, .3, .2, .1], ref.shape)
    tables = {
        "signed": ref,
        "ignored": np.broadcast_to(ref.mean(0), ref.shape),
        "reversed": ref[::-1],
        "fixed_bias": bias,
        "bias_plus_sign": .92 * bias + .08 * ref,
        "orthogonal": .25 + SIGN[:, None, None] * orthogonal,
        "uniform": np.full_like(ref, .25),
    }
    np.testing.assert_array_equal(tables["bias_plus_sign"].argmax(-1), bias.argmax(-1))

    frozen = json.loads(Path("research/stage3c/F/results.json").read_text())
    result = {"systems": {}, "details": {}}
    cubes = {}
    for name, table in tables.items():
        metrics, detail, cube = audit(table, ref, weights, cache, target, draws)
        expected = list(numeric_leaves(frozen["systems"][name]))
        actual = list(numeric_leaves(metrics))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
        detail["max_abs_difference_from_frozen"] = float(np.max(np.abs(np.subtract(actual, expected))))
        result["systems"][name] = metrics
        result["details"][name] = detail
        cubes[name] = cube
    np.testing.assert_array_equal(cubes["fixed_bias"].argmax(-1), cubes["bias_plus_sign"].argmax(-1))
    result.update(
        python=platform.python_version(), numpy=np.__version__, weight_sha=SHA,
        calibration_seeds=CAL, test_seeds=TEST, bootstrap_seed=989997, bootstrap_draws=2000,
        calibration_mothers_per_sign=65536, paired_test_mothers=cache.shape[0] * cache.shape[-1],
        overlap_mass=float(overlap.sum()), least_mass_per_cell=float(masses.min()),
        noisy_label_changed_count=int((target[:, 0] != target[:, 1]).sum()),
        cache_changed_count=int((cache[:, 0] != cache[:, 1]).sum()),
        first_mother_cache=cache[0, :, 0].tolist(), first_mother_target=target[0, :, 0].tolist(),
    )
    if args.probabilities:
        np.savez_compressed(args.probabilities, cache=cache, target=target, seeds=TEST,
                            signs=SIGN, **cubes)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
