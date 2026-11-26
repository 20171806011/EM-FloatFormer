"""Run Track B with NumPy from the repository at the frozen Stage 3 base."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


def load_encoder(repo):
    sys.path.insert(0, str(repo.resolve()))
    from research.matched_cache.learned import load

    weights = repo / "research/matched_cache/weights/equivariant_b8_s17.npz"
    raw = weights.read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    assert blob == "d33730bfc5e4013a2272eabdaf100cfc92870fa8", blob
    return load(weights), blob


def windows(angles):
    return np.repeat((np.exp(1j * angles) / np.sqrt(3))[:, None], 3, axis=1)


def count_transitions(before, after):
    return np.bincount(16 * before + after, minlength=256).reshape(16, 16)


def deterministic_collisions(encoder, angle):
    phases = (np.arange(4096) + 0.5) * 2 * np.pi / 4096
    before = encoder.encode(windows(phases))
    after = encoder.encode(windows(phases - angle))
    # Select a pair strictly inside the cells on this one-dimensional slice.
    # Verify assignments also persist on both sides of both input angles.
    stable = np.ones(len(phases), dtype=bool)
    for eps in (-0.01, 0.01):
        stable &= encoder.encode(windows(phases + eps)) == before
        stable &= encoder.encode(windows(phases - angle + eps)) == after
    counts = count_transitions(before[stable], after[stable])
    split = np.flatnonzero((counts > 0).sum(axis=1) >= 2)
    assert len(split) > 0
    code = int(split[0])
    dest = np.flatnonzero(counts[code])[:2]
    witness = []
    for target in dest:
        idx = np.flatnonzero(stable & (before == code) & (after == target))
        witness.append({"angle_rad": float(phases[idx[len(idx) // 2]]),
                        "after_derotation": int(target)})
    return {"common_code": code, "witness": witness}


def simulated_legal_inputs(encoder, angle, time, rho, rng, n=65536):
    # Stage 3 data.py: independent complex component standard deviations
    # 1/sqrt(2*rho), signal at each of the three coordinates of size 1/sqrt(3).
    labels = rng.integers(4, size=n)
    residual = rng.normal(0, 0.1 * math.sqrt(time), size=n)
    y = windows(labels * math.pi/2 + angle + residual)
    y += (rng.standard_normal(y.shape) + 1j*rng.standard_normal(y.shape)) / math.sqrt(2*rho)
    code = encoder.encode(y)
    quarter = encoder.encode(1j * y)
    assert np.array_equal(quarter, (code + 4) % 16)
    counts = count_transitions(code, encoder.encode(y * np.exp(-1j * angle)))
    # Optimistic in-sample agreement of the best deterministic post-cache map.
    # This measures code recovery, not final task accuracy or a confidence bound.
    return {"n": n, "split_source_codes": int(((counts > 0).sum(axis=1) > 1).sum()),
            "minimum_sample_code_errors": int(n - counts.max(axis=1).sum())}


def noise_sector_probs(theta, rho):
    # Sectors centered on k*pi/2. Independent orthogonal Gaussian projections.
    def normal_cdf(x):
        return (1 + math.erf(x / math.sqrt(2))) / 2
    c = math.sqrt(2*rho)
    return np.array([
        normal_cdf(c*math.cos(theta-k*math.pi/2 + math.pi/4))
        * normal_cdf(c*math.cos(theta-k*math.pi/2 - math.pi/4))
        for k in range(4)
    ])


def posterior_with_known_initial_orientation(rho, omega, time, rng):
    # Reveal u1=0 ONLY in this sign check, with no other observation.
    # Use Q3 repeat and the actual drift/noise parameters from the CFO ramp.
    angle, sd = omega * time, 0.1 * math.sqrt(time)
    nodes, weights = np.polynomial.hermite.hermgauss(64)
    def integrated(mu):
        return sum(w * noise_sector_probs(mu + math.sqrt(2)*sd*x, rho)
                   for x, w in zip(nodes, weights)) / math.sqrt(math.pi)
    analytic, wrong_sign = integrated(angle), integrated(-angle)
    n = 250000
    t = np.exp(1j*(angle + rng.normal(0, sd, size=n)))
    t += (rng.standard_normal(n) + 1j*rng.standard_normal(n))/math.sqrt(2*rho)
    sector = np.floor((np.angle(t) + math.pi/4)/(math.pi/2)).astype(int) % 4
    empirical = np.bincount(sector, minlength=4) / n
    assert np.max(np.abs(empirical - analytic)) < 0.005
    assert analytic.argmax() == 1 and wrong_sign.argmax() == 3
    assert np.allclose(integrated(angle + math.pi/2), np.roll(analytic, 1), atol=1e-12)
    return {"known_u1": 0, "n": n, "analytic_sector_probs": analytic.tolist(),
            "empirical_sector_probs": empirical.tolist(),
            "forward_argmax": int(analytic.argmax()), "negated_argmax": int(wrong_sign.argmax())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    encoder, blob = load_encoder(args.repo_root)
    rng = np.random.default_rng(261126)
    rho, omega, h, ell, tp = 10**0.2, 0.35, 2, 2, 4
    result = {"weight_git_blob": blob, "seed": 261126, "omega": omega,
              "two_window_shift_rad": omega*h, "third_window_shift_rad": omega*(h+ell),
              "pilot_shift_rad": omega*tp, "cache_derotation": {}}
    for t in (h, h+ell):
        angle = omega * t
        result["cache_derotation"][str(t)] = {
            "circle_witness": deterministic_collisions(encoder, angle),
            "gaussian_inputs": simulated_legal_inputs(encoder, angle, t, rho, rng)}
    result["oracle_sign_check"] = posterior_with_known_initial_orientation(rho, omega, h+ell, rng)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
