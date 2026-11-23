"""Exact cross-check: python3 check_discrete.py. All labels are quarter-turns."""
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import product
import json

Q = range(4)
MODES = ("none", "valid", "independent", "duplicate")
# Integer weights: a valid pilot, in the shared receive frame, is right 5/8.
WEIGHTS = (5, 1, 1, 1)
ROTATIONS = {"original": (0, 0), "second_only": (0, 1), "both": (1, 1)}


def make_oracles():
    relative = defaultdict(Counter)
    absolute = {mode: defaultdict(Counter) for mode in MODES}
    for x1, x2, g, err, unrelated in product(Q, repeat=5):
        weight = WEIGHTS[err]
        y = ((x1 + g) % 4, (x2 + g) % 4)
        p = (g + err) % 4
        relative[y][(x2 - x1) % 4] += weight
        evidence = {"none": (), "valid": (p,), "independent": (unrelated,),
                    "duplicate": (p, p)}
        for mode, refs in evidence.items():
            absolute[mode][(y, refs)][x1] += weight
    return relative, absolute


def distribution(row):
    total = sum(row.values())
    return tuple(Fraction(row[k], total) for k in Q)


def advance(vec, steps):
    return tuple(vec[(k - steps) % 4] for k in Q)


def report():
    relative, absolute = make_oracles()
    checked = 0
    for y1, y2, displayed_p in product(Q, repeat=3):
        original = (y1, y2)
        rel0 = distribution(relative[original])
        for _, (a, b) in ROTATIONS.items():
            rotated = ((y1 + a) % 4, (y2 + b) % 4)
            assert distribution(relative[rotated]) == advance(rel0, b-a)
            for mode, refs in (("none", ()), ("valid", (displayed_p,)),
                               ("independent", (displayed_p,)),
                               ("duplicate", (displayed_p, displayed_p))):
                table = absolute[mode]
                expected = advance(distribution(table[original, refs]), a)
                assert distribution(table[rotated, refs]) == expected
                checked += 1
            for y in (original, rotated):
                good = distribution(absolute["valid"][y, (displayed_p,)])
                repeated = distribution(absolute["duplicate"][y, (displayed_p,)*2])
                null = distribution(absolute["none"][y, ()])
                indep = distribution(absolute["independent"][y, (displayed_p,)])
                assert good == repeated
                assert indep == null == (Fraction(1, 4),)*4
        new_y = ((y1 + 1) % 4, (y2 + 1) % 4)
        assert distribution(absolute["valid"][new_y, ((displayed_p+1) % 4,)]) == distribution(
            absolute["valid"][original, (displayed_p,)])
    y = (0, 1)
    p = 0
    valid = distribution(absolute["valid"][y, (p,)])
    naive = tuple(v*v/sum(w*w for w in valid) for v in valid)
    def strings(values):
        return [str(value) for value in values]
    return {
        "latent_tuples_enumerated": 4**5,
        "posterior_vector_rotation_checks_at_fixed_p": checked,
        "pilot_correct_probability": "5/8",
        "window_energies_all_rotations": [1, 1],
        "mother": {"data_quarter_turns": list(y), "displayed_pilot_quarter_turns": p,
                   "relative_label_original_second_only_both": [1, 2, 1],
                   "first_symbol_posterior_valid_original": strings(valid),
                   "first_symbol_posterior_valid_second_only": strings(distribution(absolute["valid"][(0, 2), (p,)])),
                   "first_symbol_posterior_valid_both_pilot_fixed": strings(distribution(absolute["valid"][(1, 2), (p,)])),
                   "first_symbol_posterior_valid_both_and_pilot_rotated": strings(distribution(absolute["valid"][(1, 2), (1,)])),
                   "first_symbol_posterior_independent_same_displayed_value": strings(distribution(absolute["independent"][y, (p,)])),
                   "first_symbol_posterior_no_reference": strings(distribution(absolute["none"][y, ()])),
                   "first_symbol_posterior_duplicate": strings(distribution(absolute["duplicate"][y, (p,p)])),
                   "naive_duplicate_as_independent": strings(naive)},
        "interpretation": "The independent pilot is drawn independently and has the same uniform marginal as the valid pilot. Comparing the resulting oracles at identical displayed p holds appearance fixed. Source relation and duplicate provenance are declared experimental conditions."
    }


if __name__ == "__main__":
    print(json.dumps(report(), ensure_ascii=False, indent=2))
