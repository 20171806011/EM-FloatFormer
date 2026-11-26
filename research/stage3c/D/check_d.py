"""A fresh i.i.d. top-1 regret certificate for the released C4 8-bit lookup."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import beta, binom


MOTHERS = 2 ** 22
SEED = 967193
CONFIRM_MOTHERS = 2 ** 24
CONFIRM_SEED = 967229
BATCH = 8192
TASKS = ("q1", "q3")
N_COMPARISONS = 2 * 16 * 3
RIVAL_TAIL = 0.035 / N_COMPARISONS
MASS_TAIL = 0.005 / N_COMPARISONS
ACC_TAIL = 0.005 / (2 * len(TASKS))
GLOBAL_ALPHA = 0.005
TABLE_SHA = "5d257d63f148b668b7a430f2fce37f2443bc386f275cd05c18d34f5f0355d75e"
WEIGHT_SHA = "e67d9fbafb90791ab19549de7789798d818286d7f52d7ac566cc39df71e78356"


def binom_upper(hit, size, tail):
    if hit == size or size == 0:
        return 1.0
    return float(beta.isf(tail, hit + 1, size - hit))


def certify(counts, decisions):
    total = int(counts.sum())
    correct = int(counts[np.arange(16), decisions].sum())
    certificate = []
    for cell, d in enumerate(decisions):
        count_decision = int(counts[cell, d])
        comparisons = []
        for rival in range(4):
            if rival == d:
                continue
            count_rival = int(counts[cell, rival])
            pair_count = count_decision + count_rival
            rival_share_upper = binom_upper(count_rival, pair_count, RIVAL_TAIL)
            pair_mass_upper = binom_upper(pair_count, total, MASS_TAIL)
            upper = pair_mass_upper * max(0.0, 2 * rival_share_upper - 1.0)
            comparisons.append({"rival": rival, "share_upper": rival_share_upper,
                                "pair_mass_upper": pair_mass_upper, "positive_gap_upper": upper})
        certificate.append({"cell": cell, "counts": counts[cell].tolist(), "chosen": int(d),
                            "comparisons": comparisons,
                            "regret_upper": max(c["positive_gap_upper"] for c in comparisons)})
    regret = sum(c["regret_upper"] for c in certificate)
    acc_upper = binom_upper(correct, total, ACC_TAIL)
    return {"accuracy": correct / total,
            "accuracy_interval": [1.0 - binom_upper(total-correct, total, ACC_TAIL), acc_upper],
            "bayes_accuracy_upper": min(1.0, acc_upper + regret),
            "regret_upper": regret, "certified_orbits": sum(c["regret_upper"] == 0 for c in certificate),
            "cells": certificate}


def log10_binomial_tail(hit, size, p_value):
    if p_value > 0:
        return math.log10(p_value)
    # The tail underflowed. Sum probabilities relative to the mass at hit.
    # Successive ratios decrease, so a geometric bound controls the omitted sum.
    term = total = 1.0
    for k in range(hit, size):
        ratio = (size - k) / (k + 1)
        term *= ratio
        total += term
        if ratio < 1 and term * ratio / (1-ratio) <= 1e-15 * total:
            break
    return (float(binom.logpmf(hit, size, 0.5)) + math.log(total)) / math.log(10)


def global_test(counts, decisions):
    comparisons = []
    for task in TASKS:
        for i, d in enumerate(decisions[task]):
            chosen = int(counts[task][i, d])
            for r in range(4):
                if r != d:
                    rival = int(counts[task][i, r])
                    p = float(binom.sf(chosen - 1, chosen + rival, 0.5))
                    comparisons.append({"task": task, "cell": i, "chosen": int(d),
                                        "rival": r, "chosen_count": chosen, "rival_count": rival,
                                        "p": p, "log10_p": log10_binomial_tail(chosen, chosen + rival, p)})
    worst = max(comparisons, key=lambda item: item["p"])
    return {"counts": {t: counts[t].tolist() for t in TASKS}, "weakest_comparison": worst,
            "comparisons": comparisons, "n_tests": len(comparisons),
            "alpha": GLOBAL_ALPHA, "global_p": worst["p"],
            "all_classes_certified": worst["p"] <= GLOBAL_ALPHA}


def simulate(encoder, total, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    count = {t: np.zeros((16, 4), dtype=np.int64) for t in TASKS}
    roots = np.array([1, 1j, -1, -1j])
    noise_sd = 1 / np.sqrt(2 * 10 ** 0.2)
    for start in range(0, total, BATCH):
        n = min(BATCH, total - start)
        u, v = rng.integers(0, 4, size=(2, n))
        a, b = rng.normal(0.7, 0.1 * np.sqrt(2), size=(2, n))

        def observe(signal):
            noise = rng.standard_normal((n, 3)) + 1j * rng.standard_normal((n, 3))
            return signal[:, None] / np.sqrt(3) + noise_sd * noise

        y1 = observe(roots[u])
        y2 = observe(roots[v] * np.exp(1j * a))
        y3 = observe(roots[u] * np.exp(1j * (a + b))).sum(axis=1)
        z1, z2 = encoder.encode(y1), encoder.encode(y2)
        k3 = np.floor((np.angle(y3) + np.pi/4) / (np.pi/2)).astype(np.int64) % 4
        cell = 4 * (z1 % 4) + z2 % 4
        offset1, offset3 = z2 // 4 - z1 // 4, z1 // 4
        relative1, relative3 = (v-u-offset1) % 4, (k3-offset3) % 4
        if start == 0:
            for turn in (1, 2, 3):
                for zz, yy in ((z1, y1), (z2, y2)):
                    assert np.array_equal(encoder.encode(yy[:96] * roots[turn]), (zz[:96] + 4*turn) % 16)
        for task, relative in zip(TASKS, (relative1, relative3)):
            count[task] += np.bincount(4*cell+relative, minlength=64).reshape(16, 4)
    for value in count.values():
        assert value.sum() == total
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    table_text = (repo / "research/stage3b_math/A/c4_s17_lookup.json").read_text(encoding="utf-8")
    assert hashlib.sha256(table_text.encode()).hexdigest() == TABLE_SHA
    table = json.loads(table_text)
    decision = {t: np.argmax([c[t] for c in table["cells"]], axis=1) for t in TASKS}
    matched = repo / "research/matched_cache"
    weights = matched / "weights/equivariant_b8_s17.npz"
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == WEIGHT_SHA
    sys.path.insert(0, str(matched))
    from learned import load
    model = load(weights)
    assert model.equivariant and len(model.centers) == 4
    counts = simulate(model, MOTHERS, SEED)
    confirmation = global_test(simulate(model, CONFIRM_MOTHERS, CONFIRM_SEED), decision)
    result = {"mothers": MOTHERS, "batch": BATCH, "PCG64_seed": SEED, "table_sha256": TABLE_SHA,
              "weight_sha256": WEIGHT_SHA, "family_confidence": 0.95,
              "family_tail_budgets": {"rival_share": 0.035, "pair_mass": 0.005, "accuracy": 0.005,
                                      "independent_global_confirmation": GLOBAL_ALPHA},
              "confirmation_mothers": CONFIRM_MOTHERS, "confirmation_seed": CONFIRM_SEED,
              "confirmation": confirmation, "tasks": {t: certify(counts[t], decision[t]) for t in TASKS}}
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
