"""Paired QPSK questions, Python 3.9+, standard library. Run: python paired.py."""
import argparse
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
import json
import math
import random

Q1 = "第二个窗口的发射符号相对第一个窗口转了几个四分之一圈？"
Q2 = "第一个窗口原来发送的是哪个四相符号？"
ROOTS = (1, 1j, -1, -1j)
U = (0.25,) * 4
N = 3
SNR = 10 ** 0.2  # E_s/N_0 = 2 dB, signal energy per window is one
ROTATIONS = ((0, 0), (0, 1), (1, 1))
ARMS = ("none", "valid", "independent", "duplicate")
SPLITS = {"dev": (410061, 4), "test": (518339, 16)}  # blocks of 64 mothers
VIEWS = ("complex", "amplitude", "local_phase_removed")


@dataclass(frozen=True)
class Reference:
    # equality of source ids marks two renderings of one observation
    source: int
    shares_channel: bool
    samples: tuple


def rotate(window, quarter_turns):
    return tuple(y * ROOTS[quarter_turns % 4] for y in window)


def parents(split):
    seed, blocks = SPLITS[split]
    rng = random.Random(seed)
    noise_sd = math.sqrt(1 / (2 * SNR))

    def observe(label):
        mean = ROOTS[label % 4] / math.sqrt(N)
        return tuple(mean + complex(rng.gauss(0, noise_sd), rng.gauss(0, noise_sd))
                     for _ in range(N))

    for block in range(blocks):
        triples = list(product(range(4), repeat=3))
        rng.shuffle(triples)
        for j, (x1, x2, g) in enumerate(triples):
            y = (observe(x1 + g), observe(x2 + g))
            pilot = observe(g)
            g_other = rng.randrange(4)
            other = rotate(pilot, g_other - g)
            yield (split, block * 64 + j), (x1, x2), y, pilot, other


def reference_views(pilot, other):
    valid = Reference(0, True, pilot)
    return {"none": (), "valid": (valid,), "independent": (Reference(0, False, other),),
            "duplicate": (valid, valid)}


def features(win, view):
    if view == "complex":
        return win
    if view == "amplitude":
        return tuple(abs(z) for z in win)
    if view == "local_phase_removed":
        # One canonical orientation per window, with no absolute-angle side field.
        anchor = next((z for z in win if z != 0), None)
        if anchor is None:
            return win
        return tuple(z * anchor.conjugate() / abs(anchor) for z in win)
    raise ValueError(view)


def likelihood(win):
    total = sum(win)
    logw = [2 * SNR / math.sqrt(N) * (total * s.conjugate()).real for s in ROOTS]
    return normalize_exp(logw)


def normalize_exp(logw):
    a = max(logw)
    weights = [math.exp(v - a) for v in logw]
    return tuple(w / sum(weights) for w in weights)


def diff_prob(left, right):
    # P(right label - left label = k)
    return tuple(sum(left[i] * right[(i + k) % 4] for i in range(4)) for k in range(4))


def cache(data, view):
    # Preserve full observations in the selected representation. No bit cap here.
    return tuple(features(win, view) for win in data)


def read(cache_value, references, view):
    p1, p2 = (likelihood(w) if view == "complex" else U for w in cache_value)
    first_answer = diff_prob(p1, p2)
    shared = {ref.source: features(ref.samples, view) for ref in references if ref.shares_channel}
    if view != "complex" or not shared:
        return first_answer, U
    logw = [sum(2 * SNR / math.sqrt(N) * (sum(w) * s.conjugate()).real
                for w in shared.values()) for s in ROOTS]
    return first_answer, diff_prob(normalize_exp(logw), p1)


def shift(probs, steps):
    return tuple(probs[(k - steps) % 4] for k in range(4))


def maxdelta(a, b):
    return max(abs(x - y) for x, y in zip(a, b))


def argmax(p):
    return max(range(4), key=p.__getitem__)


def iq_pairs(window):
    return [[z.real, z.imag] for z in window]


def cases(split):
    """Yield evaluator metadata and the two timed inputs, kept in separate fields."""
    for mother, (x1, x2), y, pilot, other in parents(split):
        for a, b in ROTATIONS:
            windows = (rotate(y[0], a), rotate(y[1], b))
            for arm, refs in reference_views(pilot, other).items():
                yield {
                    "encode_once": {"windows_iq": [iq_pairs(w) for w in windows]},
                    "first_query": {"question": Q1},
                    "second_query": {"question": Q2, "references": [
                        {"observation_id": ref.source, "shares_channel": ref.shares_channel,
                         "iq": iq_pairs(ref.samples)} for ref in refs]},
                    "evaluator_only": {"mother": mother, "turns": [a, b], "arm": arm,
                                       "labels": [(x2+b-x1-a) % 4, (x1+a) % 4]},
                }


def run(split):
    scores = defaultdict(lambda: [0, 0, 0.0, 0, 0.0])
    max_change = defaultdict(float)
    naive_tv = []
    example = None
    n = 0
    corrections, regressions = 0, 0
    for parent_id, (x1, x2), y, pilot, other in parents(split):
        n += 1
        refs = reference_views(pilot, other)
        assert tuple(map(abs, pilot)) == tuple(map(abs, other))
        origin = {}
        for s1, s2 in ROTATIONS:
            data = (rotate(y[0], s1), rotate(y[1], s2))
            for k in (0, 1):
                assert tuple(map(abs, data[k])) == tuple(map(abs, y[k]))
            r, a = (x2 + s2 - x1 - s1) % 4, (x1 + s1) % 4
            for view in VIEWS:
                z = cache(data, view)
                for arm in ARMS:
                    pr, pa = read(z, refs[arm], view)
                    if s1 == s2 == 0:
                        origin[view, arm] = pr, pa
                    br, ba = origin[view, arm]
                    changes = {"relative_rotation": maxdelta(pr, shift(br, s2-s1)),
                               "absolute_rotation": maxdelta(pa, shift(ba, s1))}
                    if arm in ("none", "independent"):
                        changes["uninformative_pilot"] = maxdelta(U, pa)
                    if arm == "duplicate":
                        one_r, one_a = read(z, refs["valid"], view)
                        changes["duplicate"] = max(maxdelta(pa, one_a), maxdelta(pr, one_r))
                    for check, delta in changes.items():
                        max_change[check] = max(max_change[check], delta)
                    row = scores[view, arm]
                    row[0] += 1
                    row[1] += argmax(pr) == r
                    row[2] -= math.log(pr[r])
                    row[3] += argmax(pa) == a
                    row[4] -= math.log(pa[a])
            # Before the pilot the four labels tie; use the documented lowest-label tie break.
            z = cache(data, "complex")
            valid_r, valid_a = read(z, refs["valid"], "complex")
            before_r, before_a = read(z, (), "complex")
            assert maxdelta(valid_r, before_r) < 3e-13
            before_correct, after_correct = argmax(before_a) == a, argmax(valid_a) == a
            corrections += (not before_correct and after_correct)
            regressions += (before_correct and not after_correct)
            # Rotate the entire receiver coordinate system, including the pilot.
            shifted_z = cache(tuple(rotate(w, 1) for w in data), "complex")
            coordinate_r, coordinate_a = read(shifted_z, (Reference(0, True, rotate(pilot, 1)),), "complex")
            max_change["coordinate_change"] = max(max_change["coordinate_change"],
                maxdelta(valid_r, coordinate_r), maxdelta(valid_a, coordinate_a))
            if s1 == s2 == 0:
                # Deliberately misidentify one repeated acquisition as two independent ones.
                wrong = (refs["valid"][0], Reference(1, True, pilot))
                wrong_pa = read(cache(data, "complex"), wrong, "complex")[1]
                good_pa = origin["complex", "valid"][1]
                naive_tv.append(sum(abs(q-p) for q, p in zip(wrong_pa, good_pa)) / 2)
        if example is None:
            example = {"id_evaluator_only": parent_id, "x1": x1, "x2": x2,
                       "base_valid_relative_posterior": origin["complex", "valid"][0],
                       "base_valid_absolute_posterior": origin["complex", "valid"][1]}
    assert all(v < 3e-13 for v in max_change.values())
    assert sum(naive_tv) / n > 0.001
    return {"parent_count": n, "paired_rows": n * len(ROTATIONS) * len(ARMS),
            "rows_are_correlated_within_parent": True,
            "none_to_valid_first_symbol": {"paired_window_versions": n*len(ROTATIONS),
                                           "corrected": corrections, "new_errors": regressions},
            "rotation_and_reference_max_abs_residual": dict(max_change),
            "mean_duplicate_double_count_tv": sum(naive_tv) / n,
            "metrics": {view: {
                arm: dict(zip(("first_accuracy", "first_log_loss_nats", "second_accuracy",
                               "second_log_loss_nats"), [round(v / scores[view, arm][0], 8)
                                                          for v in scores[view, arm][1:]]))
                for arm in ARMS} for view in VIEWS}, "example": example}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=tuple(SPLITS), default="test")
    parser.add_argument("--cases", action="store_true", help="emit one paired case per line as JSON")
    args = parser.parse_args()
    split = args.split
    if args.cases:
        for case in cases(split):
            print(json.dumps(case, ensure_ascii=False))
        raise SystemExit(0)
    print(json.dumps({"split": split, "questions": [Q1, Q2], **run(split)},
                     ensure_ascii=False, indent=2))
