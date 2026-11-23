"""Exact rational, standard-library check. Run python moving_pair/verify.py."""
import csv
import itertools
import sys
from fractions import Fraction as F
from math import lcm


ALPHABET = range(4)


def channel(p):
    return [1 - p, p / 3, p / 3, p / 3]


def integer_weights(probabilities):
    denominator = lcm(*(p.denominator for p in probabilities))
    return [int(p * denominator) for p in probabilities]


def walk(d, steps):
    distribution = [F(1), F(0), F(0), F(0)]
    for _ in range(steps):
        distribution = [
            sum(distribution[(j - k) % 4] * channel(d)[k] for k in ALPHABET)
            for j in ALPHABET
        ]
    return integer_weights(distribution)


def enumerate_risks(h, gap, d, e, ep):
    drift01 = walk(d, h)
    drift12 = walk(d, gap)
    noise, pilot = map(integer_weights, (channel(e), channel(ep)))
    # tables are keyed by all observations available to each question
    rel = [[0] * 4 for _ in range(16)]
    absolute = [[0] * 4 for _ in range(64)]
    ablated = [[0] * 4 for _ in range(16)]
    for x1, x2, g, u, v, n1, n2, np in itertools.product(ALPHABET, repeat=8):
        mass = drift01[u] * drift12[v] * noise[n1] * noise[n2] * pilot[np]
        y1, y2 = (x1 + g + n1) % 4, (x2 + g + u + n2) % 4
        p = (g + u + v + np) % 4
        rel[4*y1 + y2][(x2 - x1) % 4] += mass
        absolute[16*y1 + 4*y2 + p][x1] += mass
        ablated[4*y1 + p][x1] += mass
    for y1, y2, p in itertools.product(ALPHABET, repeat=3):
        full_row = absolute[16*y1 + 4*y2 + p]
        assert [4 * val for val in full_row] == ablated[4*y1 + p]

    def risk(rows):
        return 1 - F(sum(map(max, rows)), sum(map(sum, rows)))

    return risk(rel), risk(absolute)


def bayes_error(c):
    return 1 - max((1 + 3*c) / 4, (1 - c) / 4)


def reliability(error_probability):
    return 1 - 4 * error_probability / 3


def main():
    ordinary = (F(12, 100), F(6, 100), F(3, 100))
    cases = [
        ("ideal", 0, 0, F(0), F(0), F(0)),
        ("coherent", 0, 0, *ordinary),
        ("gap_after_data", 0, 8, *ordinary),
        ("gap_between_data", 8, 0, *ordinary),
        ("split_gap", 4, 4, *ordinary),
        ("independent_pilot", 0, 1, F(3, 4), F(0), F(0)),
        ("independent_data", 1, 0, F(3, 4), F(0), F(0)),
        ("anti_correlated_step", 1, 0, F(1), F(0), F(0)),
    ]
    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(["case", "h", "L", "d", "e", "ep", "relative_error", "first_symbol_error"])
    for label, h, gap, d, e, ep in cases:
        a, b, t = map(reliability, (e, ep, d))
        observed = enumerate_risks(h, gap, d, e, ep)
        assert observed == (bayes_error(a*a*t**h), bayes_error(a*b*t**(h+gap)))
        writer.writerow([label, h, gap, str(d), str(e), str(ep), *(f"{float(p):.9f}" for p in observed)])


if __name__ == "__main__":
    main()
