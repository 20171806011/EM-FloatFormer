"""Exact, dependency-free verifier for the four-phase delayed-reference game."""

from collections import Counter, defaultdict
from itertools import product
from math import comb
import json


PHASES = range(4)
PAIRS = tuple(product(PHASES, repeat=2))
WEIGHTS = ((1, 0), (0, 1), (1, 1), (2, 1), (1, 2))


def enumerate_codebooks():
    """Optimize the encoder separately at each of 16 inputs for each decoder book."""
    per_target = {}
    for weights in WEIGHTS:
        wr, wa = weights
        scores = []
        for r, u in PAIRS:
            scores.append(tuple(wr * (r == rr) + wa * (u == uu) for rr, uu in PAIRS))
        best = [-1] * 17
        # best_for_mask[s] is the total score of decoder reproduction set s.
        # For one source point, its best encoder selects its highest-payoff entry.
        sums = [0] * (1 << 16)
        for row in scores:
            row_best = [0] * (1 << 16)
            for mask in range(1, 1 << 16):
                last = mask & -mask
                row_best[mask] = max(row[last.bit_length() - 1], row_best[mask ^ last])
                sums[mask] += row_best[mask]
        for mask in range(1, 1 << 16):
            m = mask.bit_count()
            best[m] = max(best[m], sums[mask])
        per_target[weights] = best
    return per_target


def bound(m, wr, wa):
    k = min(m, 4)
    return max(wr, wa) * (4 * k) + min(wr, wa) * (m + (4 - k) * k)


def explicit_encoding(m, weights):
    book = [(i, i) for i in range(min(m, 4))]
    book += [pair for pair in PAIRS if pair not in book][:m - len(book)]
    wr, wa = weights
    return {
        (r, u): max(book, key=lambda pair: wr * (r == pair[0]) + wa * (u == pair[1]))
        for r, u in PAIRS
    }


def enumerate_hidden_states(encoding):
    successes = [0, 0, 0]
    for x1, x2, g in product(PHASES, repeat=3):
        y1, y2 = (x1 + g) % 4, (x2 + g) % 4
        rr, uu = encoding[((y2 - y1) % 4, y1)]
        relative = rr == (x2 - x1) % 4
        absolute = (uu - g) % 4 == x1
        successes[0] += relative
        successes[1] += absolute
        successes[2] += relative and absolute
    return successes


def check_missing_reference():
    views = [defaultdict(Counter) for _ in range(3)]
    for x1, x2, g, independent_p in product(PHASES, repeat=4):
        y = ((x1 + g) % 4, (x2 + g) % 4)
        r = (y[1] - y[0]) % 4
        for histogram, observation in zip(views, (y, (y, independent_p), (r, g))):
            histogram[observation][x1] += 1
    for histogram in views:
        assert all(len(counts) == 4 and len(set(counts.values())) == 1
                   for counts in histogram.values())
        assert sum(max(counts.values()) for counts in histogram.values()) == 64


def main():
    check_missing_reference()
    best = enumerate_codebooks()
    for m in range(1, 17):
        for wr, wa in WEIGHTS:
            assert best[wr, wa][m] == bound(m, wr, wa)
        high, low = 4 * min(m, 4), m + (4 - min(m, 4)) * min(m, 4)
        for weights, targets in (((2, 1), (high, low)), ((1, 2), (low, high))):
            hits = enumerate_hidden_states(explicit_encoding(m, weights))
            assert hits == [4 * targets[0], 4 * targets[1], 4 * m]
    result = {
        "model": "independent uniform X1, X2, G in Z4; coherent received pilot P=G",
        "potential_hidden_states": 64,
        "distinct_nonempty_reproduction_sets": sum(comb(16, m) for m in range(1, 17)),
        "weighted_marginal_objectives_verified": WEIGHTS,
        "independent_pilot_control_hidden_states": 256,
        "absolute_accuracy_without_pilot_or_with_independent_pilot": 0.25,
        "budgets": [],
    }
    for bits in range(5):
        m = 2 ** bits
        counts = enumerate_hidden_states(explicit_encoding(m, (2, 1)))
        relative, absolute, joint = [n / 64 for n in counts]
        result["budgets"].append({
            "bits": bits, "states": m,
            "relative_priority_accuracy": [relative, absolute],
            "absolute_priority_accuracy": [absolute, relative],
            "best_both_correct_probability": joint,
            "best_equal_weight_mean_accuracy": (relative + absolute) / 2,
        })
    return result


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
