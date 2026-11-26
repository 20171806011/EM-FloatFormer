"""Behavioral sign diagnostics, Q3-repeat, Stage 3 s17 cache. CPU only."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np


SIGNS = (-1, 1)
RATE = 0.35
WEIGHT_SHA = "e67d9fbafb90791ab19549de7789798d818286d7f52d7ac566cc39df71e78356"
CAL_SEEDS = tuple(range(988001, 988009))
TEST_SEEDS = (989011, 989023, 989039, 989063)


def paired_mothers(data, encoder, seed, blocks):
    """Keep balanced triples, Brownian path innovations and complex additive noises paired."""
    tag, cname = "f_counterfactual", "C4_cfo_ramp"
    worlds = []
    codes = []
    for sign in SIGNS:
        condition = replace(data.CONDITION_SPECS[cname], omega=sign * RATE)
        with patch.dict(data.SPLIT_SEEDS, {tag: (seed, blocks)}), patch.dict(
                data.CONDITION_SPECS, {cname: condition}):
            batch = data.generate_mothers(tag, cname)
        codes.append(16 * encoder.encode(batch.y1) + encoder.encode(batch.y2))
        worlds.append(batch)
    neg, pos = worlds
    assert np.array_equal(neg.y1, pos.y1) and np.array_equal(neg.x2, pos.x2)
    # The target is the noisy RECEIVED sector, evaluated separately in each physical world.
    for time, field, symbols, window in ((2, "phi_2", neg.x2, "y2"), (4, "phi_3", neg.x1, "y3_repeat")):
        phi_lo, phi_hi = getattr(neg, field), getattr(pos, field)
        np.testing.assert_allclose(phi_hi - phi_lo, 2 * RATE * time, atol=1e-12)
        base = data.ROOTS[(symbols + neg.g) % 4, None] / np.sqrt(data.N_SAMPLES)
        noise_lo = getattr(neg, window) - base * np.exp(1j * phi_lo[:, None])
        noise_hi = getattr(pos, window) - base * np.exp(1j * phi_hi[:, None])
        np.testing.assert_allclose(noise_lo, noise_hi, rtol=0, atol=2e-14)
    return np.array(codes), np.array([b.k3_repeat for b in worlds])


def calibrate(data, encoder):
    # Pool two independently rotatable QPSK symbols; the received target follows W1.
    counts = np.zeros((2, 4, 4, 4), dtype=int)
    mass = np.zeros((2, 256))
    for seed in CAL_SEEDS:
        codes, targets = paired_mothers(data, encoder, seed, 128)
        for i in range(2):
            z1, z2 = divmod(codes[i], 16)
            np.add.at(counts[i], (z1 % 4, z2 % 4, (targets[i] - z1 // 4) % 4), 1)
            mass[i] += np.bincount(codes[i], minlength=256)
    full = np.zeros((2, 256, 4))
    for k in range(256):
        z1, z2 = divmod(k, 16)
        full[:, k, :] = np.roll(counts[:, z1 % 4, z2 % 4] + 0.5, z1 // 4, axis=-1)
    return full / full.sum(-1, keepdims=True), mass / mass.sum(-1, keepdims=True)


def validate(table):
    if table.shape != (2, 256, 4) or not np.isfinite(table).all() or (table < 0).any():
        raise ValueError("Expected two signs x 256 codes x four finite nonnegative probabilities")
    np.testing.assert_allclose(table.sum(-1), 1, rtol=0, atol=1e-6)


def fixed_cache(table, reference, weights):
    validate(table)
    contrast = table[1] - table[0]
    target = reference[1] - reference[0]
    weighted_sum = lambda x: (weights[:, None] * x).sum()
    norm = weighted_sum(target**2)
    if norm < 1e-10:
        raise ValueError("Reference sign contrast is uninformative at this design")
    gain = weighted_sum(contrast * target) / norm
    equiv = []
    for z in range(256):
        z1, z2 = divmod(z, 16)
        rotated = 16 * ((z1 + 4) % 16) + z2
        discrepancy = table[:, rotated, :] - np.roll(table[:, z, :], 1, axis=-1)
        equiv.append(np.sum(abs(discrepancy), axis=-1).mean()/2)
    return dict(orientation_gain=float(gain),
                residual_ratio=float(np.sqrt(weighted_sum((contrast-gain*target)**2)/norm)),
                mean_sign_TV=float(weighted_sum(abs(contrast))/2),
                mean_rotation_TV=float(np.dot(weights, equiv)),
                reference_odd_norm_squared=float(norm))


def evaluate_physical(table, cubes):
    matched_nll, swapped_nll, matched_hit, swapped_hit = [], [], [], []
    effects = []
    for codes, targets in cubes:
        # p[world_sign, told_sign, mother, sector], with physical labels fixed across told signs.
        p = np.stack([[table[t][codes[s]] for t in range(2)] for s in range(2)])
        m = (p[0, 1]-p[0, 0]+p[1, 1]-p[1, 0])/4
        w = (p[1, 0]-p[0, 0]+p[1, 1]-p[0, 1])/4
        interaction = (p[1, 1]-p[1, 0]-p[0, 1]+p[0, 0])/4
        effects.append([np.mean(np.sum(x*x, axis=-1)) for x in (m, w, interaction)])
        idx = np.arange(codes.shape[1])
        # Resample source blocks once per seed. Each mother carries BOTH signs and all four cells.
        for nll, hit, flip in ((matched_nll, matched_hit, 0), (swapped_nll, swapped_hit, 1)):
            per_world_nll, per_world_hit = [], []
            for s in range(2):
                probs, y = p[s, s^flip], targets[s]
                # Finite reporting score; exact zeros have infinite loss under ordinary log scoring.
                per_world_nll.append(-np.log(np.maximum(probs[idx, y], 1e-12)))
                ties = probs == probs.max(axis=1, keepdims=True)
                per_world_hit.append(ties[idx, y]/ties.sum(1))
            nll.append(np.mean(per_world_nll, axis=0).reshape(-1, 64).mean(1))
            hit.append(np.mean(per_world_hit, axis=0).reshape(-1, 64).mean(1))
    diff = np.array(swapped_nll) - np.array(matched_nll)
    rng = np.random.default_rng(989997)
    strata = [x[rng.integers(len(x), size=(2000,len(x)))].mean(1) for x in diff]
    lo, hi = np.quantile(np.mean(strata, axis=0), [0.025,0.975])
    return dict(matched_accuracy=float(np.mean(matched_hit)), swapped_accuracy=float(np.mean(swapped_hit)),
                swapped_minus_matched_nll=[float(diff.mean()), float(lo), float(hi)],
                factorial_mean_squared_norm=dict(zip(("metadata","wave","interaction"),np.mean(effects,axis=0).tolist())))


def check(repo, host):
    from importlib import import_module
    sys.path[:0] = [str(repo), str(repo/"research"/"matched_cache")]
    data = import_module("research.stage3_ministral_cache.data")
    encoder_file = repo/"research/matched_cache/weights/equivariant_b8_s17.npz"
    assert hashlib.sha256(encoder_file.read_bytes()).hexdigest() == WEIGHT_SHA
    encoder = import_module("learned").load(encoder_file)
    ref, mass = calibrate(data, encoder)
    overlap = np.minimum(*mass)
    weights = overlap / overlap.sum()
    even = np.repeat(ref.mean(0)[None], 2, axis=0)
    sector_bias = np.broadcast_to([.4,.3,.2,.1], ref.shape).copy()
    # A direction that varies with the sign, but has zero weighted inner product
    # with the physical reference contrast. A scalar orientation score alone misses it.
    d = ref[1]-ref[0]
    v = np.broadcast_to([1.,-1.,0.,0.], d.shape)
    v = v - ((weights[:,None]*d*v).sum()/(weights[:,None]*d*d).sum())*d
    v = .15*v/np.abs(v).max()
    distractor = np.stack((.25-v,.25+v))
    # Exact aggregate-accuracy witness with sign-responsive probabilities: common
    # argmax 0 in every cell while the signed probability difference is nonzero.
    biased_signed = .92 * sector_bias + .08 * ref
    assert (biased_signed.argmax(-1) == sector_bias.argmax(-1)).all()
    systems = dict(signed=ref, ignored=even, reversed=ref[::-1], fixed_bias=sector_bias,
                   bias_plus_sign=biased_signed, orthogonal=distractor,
                   uniform=np.full_like(ref,.25))
    if host:
        out = json.loads(host.read_text())
        if (out["task"], out["encoder_sha"], out["rate_magnitudes"], out["h"], out["ell"],
            out["sigma_phi"], out["snr_db"], out["sign_order"], out["pilot"], out["cache_order"]) != (
                "Q3_repeat",WEIGHT_SHA,[RATE],2,2,0.10,2.0,[-1,1],"none","16*z1+z2"):
            raise ValueError("Incompatible host table contract")
        systems["host"] = np.asarray(out["probabilities"], dtype=float)
    cubes = [paired_mothers(data, encoder, s, 16) for s in TEST_SEEDS]
    result = {k: {**fixed_cache(p,ref,weights), **evaluate_physical(p,cubes)} for k,p in systems.items()}
    for name, expected in (("signed",1), ("ignored",0), ("reversed",-1), ("fixed_bias",0), ("bias_plus_sign",.08)):
        assert np.isclose(result[name]["orientation_gain"], expected)
    for name in ("signed","ignored","reversed"):
        assert result[name]["mean_rotation_TV"] < 1e-10
    assert result["fixed_bias"]["mean_rotation_TV"] > 0
    assert abs(result["orthogonal"]["orientation_gain"]) < 1e-12
    assert result["orthogonal"]["mean_sign_TV"] > 0.1
    assert result["orthogonal"]["residual_ratio"] > .1
    assert result["uniform"]["matched_accuracy"] == 0.25
    assert result["bias_plus_sign"]["matched_accuracy"] == result["fixed_bias"]["matched_accuracy"]
    assert result["signed"]["swapped_minus_matched_nll"][1] > 0
    assert result["reversed"]["swapped_minus_matched_nll"][2] < 0
    return dict(weight_sha=WEIGHT_SHA, calibration_seeds=CAL_SEEDS, test_seeds=TEST_SEEDS,
                mothers_per_sign_calibration=64*128*len(CAL_SEEDS),
                paired_test_mothers=16*64*len(TEST_SEEDS),rate_magnitude=RATE,
                least_mass_per_cell=float(mass.min()),overlap_mass=float(overlap.sum()),
                noisy_label_changed_fraction=float(np.mean([np.mean(y[0]!=y[1]) for _,y in cubes])),
                codes_changed_fraction=float(np.mean([np.mean(z[0]!=z[1]) for z,_ in cubes])),systems=result)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--repo",required=True,type=Path)
    cli.add_argument("--host",type=Path)
    cli.add_argument("--out",type=Path)
    args = cli.parse_args()
    payload = json.dumps(check(args.repo.resolve(),args.host),indent=2,allow_nan=False)+"\n"
    if args.out:
        args.out.write_text(payload,encoding="utf-8")
    else:
        print(payload,end="")
