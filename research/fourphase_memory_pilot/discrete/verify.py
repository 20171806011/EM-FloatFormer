"""Recompute exact risk, and verify every deterministic frontier point by enumeration."""

from collections import defaultdict
import csv
from fractions import Fraction as F
from itertools import product
import sys


Q = range(4)
CELLS = list(product(Q, repeat=2))
PARAMETERS = {
    "ideal": (F(0), F(0), F(0), F(0)),
    "noisy": (F(1, 10), F(1, 10), F(1, 20), F(3, 50)),
    "asymmetric": (F(1, 20), F(1, 5), F(1, 10), F(1, 4)),
    "independent_pilot": (F(1, 10), F(1, 10), F(3, 4), F(0)),
    "independent_data": (F(3, 4), F(3, 4), F(0), F(0)),
}


def coeff(error):
    return 1 - 4 * error / 3


def symmetric_pmf(error):
    return [1 - error, error / 3, error / 3, error / 3]


def convolve(left, right):
    return [sum(left[j] * right[(i - j) % 4] for j in Q) for i in Q]


def integer_pmf(pmf):
    """All entries share an integer denominator."""
    from math import lcm
    denominator = lcm(*(p.denominator for p in pmf))
    return [int(p * denominator) for p in pmf], denominator


def enumerate_channel(e1, e2, ep, d, gap):
    """Direct latent-state joint counts of (U,V,P,R,X1)."""
    offset = symmetric_pmf(ep)
    for _ in range(gap):
        offset = convolve(offset, symmetric_pmf(d))
    law1, den1 = integer_pmf(symmetric_pmf(e1))
    law2, den2 = integer_pmf(symmetric_pmf(e2))
    lawp, denp = integer_pmf(offset)
    counts = defaultdict(int)
    for x1, x2, g, n1, n2, pilot_error in product(Q, repeat=6):
        weight = law1[n1] * law2[n2] * lawp[pilot_error]
        if weight:
            v = (x1 + g + n1) % 4
            u = (x2 + g + n2 - v) % 4
            counts[u, v, (g + pilot_error) % 4, (x2 - x1) % 4, x1] += weight
    total = 64 * den1 * den2 * denp
    assert sum(counts.values()) == total
    return counts, total


def centers_and_cache(size, extra_relative):
    diagonal = [(i, i) for i in Q]
    others = [cell for cell in CELLS if cell not in diagonal]
    centers = diagonal + others[:size - 4]
    cache = {cell: i for i, cell in enumerate(centers)}
    for j, cell in enumerate(c for c in CELLS if c not in centers):
        coord = 0 if j < extra_relative else 1
        cache[cell] = next(i for i, c in enumerate(centers) if c[coord] == cell[coord])
    return cache


def score(counts, total, cache):
    relative = defaultdict(lambda: [0] * 4)
    absolute = defaultdict(lambda: [0] * 4)
    for (u, v, pilot, r, x1), weight in counts.items():
        m = cache[u, v]
        relative[m][r] += weight
        absolute[m, pilot][x1] += weight
    return tuple(F(sum(max(row) for row in grouped.values()), total)
                 for grouped in (relative, absolute))


def success(k, h):
    return (1 - k) / 4 + k * h


def rows():
    for name, (e1, e2, ep, d) in PARAMETERS.items():
        gaps = [0, 1, 4, 16, 64] if name == "noisy" else [1]
        for gap in gaps:
            counts, total = enumerate_channel(e1, e2, ep, d, gap)
            k_r = coeff(e1) * coeff(e2)
            k_a = coeff(e1) * coeff(ep) * coeff(d) ** gap
            for bits in (2, 3, 4):
                size = 2 ** bits
                for j in range(17 - size):
                    cache = centers_and_cache(size, j)
                    h_u, h_v = F(size+j, 16), F(16-j, 16)
                    predicted = (success(k_r, h_u), success(k_a, h_v))
                    actual = score(counts, total, cache)
                    assert predicted == actual, (name, gap, bits, j, actual, predicted)
                    yield (name, str(e1), str(e2), str(ep), str(d), gap, bits, j,
                           str(h_u), str(h_v), f"{float(1-actual[0]):.9f}",
                           f"{float(1-actual[1]):.9f}")


def main():
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["scenario", "data1_error", "data2_error", "pilot_error",
                     "step_drift_error", "gap", "cache_bits", "j", "h_U", "h_V",
                     "relative_symbol_error", "absolute_symbol_error"])
    writer.writerows(rows())


if __name__ == "__main__":
    main()
