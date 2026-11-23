"""Independent discrete-channel audit. Python 3.9+ standard library."""
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import product
from math import gcd
import json
import random

Q = range(4)
N = 60000
SEED = 410257
# Parameters are whole percentages. An independent pilot ignores the channel.
CASES = [
    ("ideal", 0, 0, 0, 0, 0, False),
    ("independent_pilot", 0, 0, 0, 0, 0, True),
    ("unequal_noise_L0", 7, 21, 12, 9, 0, False),
    ("unequal_noise_L1", 7, 21, 12, 9, 1, False),
    ("unequal_noise_L4", 7, 21, 12, 9, 4, False),
    ("one_uniform_increment", 7, 21, 12, 75, 1, False),
    ("uniform_observations", 75, 75, 75, 0, 0, False),
]
CACHES = ("relative", "relative_parity", "pair")


def kernel(pct):
    weights = (3 * (100 - pct), pct, pct, pct)
    common = gcd(*weights)
    return {k: w // common for k, w in enumerate(weights) if w}


def increment_sum(step, gap):
    counts = Counter()
    # Count every path directly rather than evaluating a matrix power.
    for path in product(step.items(), repeat=gap):
        end, weight = 0, 1
        for offset, multiplicity in path:
            end += offset
            weight *= multiplicity
        counts[end % 4] += weight
    return counts


def cache(y1, y2, mode):
    relative = (y2 - y1) % 4
    if mode == "relative":
        return relative
    if mode == "relative_parity":
        return relative, y1 % 2
    if mode == "pair":
        return y1, y2
    raise ValueError(mode)


def latent_law(e1, e2, ep, drift, gap, independent):
    """Integer atom masses indexed by actually observed data, pilot and targets."""
    joint = Counter()
    data1, data2 = kernel(e1), kernel(e2)
    anchor, offsets = kernel(ep), increment_sum(kernel(drift), gap)
    for x1, x2, g in product(Q, repeat=3):
        r = (x2 - x1) % 4
        for (a, wa), (b, wb) in product(data1.items(), data2.items()):
            y1, y2 = (x1 + g + a) % 4, (x2 + g + b) % 4
            if independent:
                for p in Q:
                    joint[y1, y2, p, r, x1] += wa * wb
            else:
                for (h, wh), (c, wc) in product(offsets.items(), anchor.items()):
                    p = (g + h + c) % 4
                    joint[y1, y2, p, r, x1] += wa * wb * wh * wc
    return joint


def majority(tables):
    # Ties go to the smallest label. Risk is constant across maximizers.
    return {view: max(Q, key=counts.__getitem__) for view, counts in tables.items()}


def fit(joint, mode):
    before, after, before_absolute = (defaultdict(Counter) for _ in range(3))
    for (y1, y2, p, r, x1), w in joint.items():
        z = cache(y1, y2, mode)
        before[z][r] += w
        before_absolute[z][x1] += w
        after[z, p][x1] += w
    assert all(len(set(counts[k] for k in Q)) == 1 for counts in before_absolute.values())
    d1, d2 = majority(before), majority(after)
    correct = [0, 0]
    for (y1, y2, p, r, x1), w in joint.items():
        z = cache(y1, y2, mode)
        correct[0] += w * (d1[z] == r)
        correct[1] += w * (d2[z, p] == x1)
    total = sum(joint.values())
    return (d1, d2), tuple(Fraction(c, total) for c in correct)


def sample_noise(rng, pct):
    return rng.randrange(1, 4) if rng.randrange(100) < pct else 0


def run_case(params, case_idx):
    name, e1, e2, ep, drift, gap, independent = params
    law = latent_law(*params[1:])
    if name == "ideal":
        assert len(law) == 64 and set(law.values()) == {1}
    fitted = {mode: fit(law, mode) for mode in CACHES}
    scores = {mode: [0, 0] for mode in CACHES}
    rng = random.Random(SEED + case_idx)
    for _ in range(N):
        x1, x2, g = (rng.randrange(4) for _ in range(3))
        y1 = (x1 + g + sample_noise(rng, e1)) % 4
        y2 = (x2 + g + sample_noise(rng, e2)) % 4
        if independent:
            p = rng.randrange(4)
        else:
            phase = sum(sample_noise(rng, drift) for _ in range(gap))
            p = (g + phase + sample_noise(rng, ep)) % 4
        for mode, ((d1, d2), _) in fitted.items():
            z = cache(y1, y2, mode)
            scores[mode][0] += d1[z] == (x2 - x1) % 4
            scores[mode][1] += d2[z, p] == x1
    report = {}
    for mode, (_, exact) in fitted.items():
        if mode == "relative" or independent or name in ("uniform_observations", "one_uniform_increment"):
            assert exact[1] == Fraction(1, 4)
        if name == "ideal":
            expected = {"relative": Fraction(1, 4), "relative_parity": Fraction(1, 2), "pair": 1}
            assert exact == (1, expected[mode])
        if name == "uniform_observations":
            assert exact == (Fraction(1, 4), Fraction(1, 4))
        for count, prob in zip(scores[mode], exact):
            assert abs(count / N - prob) < 0.012
        report[mode] = dict(exact_success=[str(p) for p in exact], simulated_correct=scores[mode])
    return report


def main():
    output = {params[0]: run_case(params, idx) for idx, params in enumerate(CASES)}
    for mode in CACHES:
        r0 = output["unequal_noise_L0"][mode]["exact_success"][0]
        for case in ("unequal_noise_L1", "unequal_noise_L4", "one_uniform_increment"):
            assert output[case][mode]["exact_success"][0] == r0
    obj = dict(samples_per_case=N, first_seed=SEED, results=output)
    rendered = json.dumps(obj, indent=2) + "\n"
    print(rendered, end="")


if __name__ == "__main__":
    main()
