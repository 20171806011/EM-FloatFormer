"""Evaluate the registered models and compute mother-paired confidence intervals."""
import argparse
import json
from pathlib import Path

import numpy as np

from evaluate import evaluate
from learned import load
from physical import Hand
from train import BUDGETS, ROOT, SEEDS, VARIANTS, name

METRICS = ("q1_accuracy", "q1_nll", "q2_valid_accuracy", "q2_valid_nll")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def differences(paired, draws):
    bootstrap = []
    for index in np.array_split(draws, 20):
        bootstrap.extend(paired[index].mean(axis=1))
    lo, hi = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    return {key: {"mean": float(v), "lo95": float(a), "hi95": float(b)}
            for key, v, a, b in zip(METRICS, paired.mean(axis=0), lo, hi)}


def estimate(folder, weights):
    summary = {}
    for split in ("dev", "test"):
        group = {}
        for bits in BUDGETS:
            baseline, hand = evaluate(Hand(bits), bits, split)
            write_json(folder / split / f"hand_b{bits}.json", baseline)
            n = len(hand)
            draws = np.random.default_rng(17429).integers(0, n, (2000, n))
            group[f"hand_b{bits}"] = {"means": dict(zip(METRICS, hand.mean(axis=0).tolist())),
                                      "passes": baseline["all_three"]["pass"]}
            for variant in VARIANTS:
                each = []
                contrasts = {}
                passes = []
                strict_failures = []
                for seed in SEEDS:
                    tag = name(variant, bits, seed)
                    model = load(weights / (tag + ".npz"))
                    detail, parent = evaluate(model, bits, split)
                    write_json(folder / split / (tag + ".json"), detail)
                    contrasts[str(seed)] = differences(parent - hand, draws)
                    each.append(parent)
                    passes.append(detail["all_three"]["pass"])
                    strict_failures.append(detail["all_three"]["mothers_failed"])
                across = np.mean(each, axis=0)
                group[f"{variant}_b{bits}"] = {
                    "means": dict(zip(METRICS, across.mean(axis=0).tolist())),
                    "mean_minus_hand": differences(across - hand, draws),
                    "seed_minus_hand": contrasts, "seeds": list(SEEDS),
                    "passes": passes, "strict_failed_mothers_by_seed": strict_failures}
        summary[split] = group
    write_json(folder / "summary.json", summary)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=ROOT / "results")
    cli.add_argument("--weights", type=Path, default=ROOT / "weights")
    args = cli.parse_args()
    estimate(args.output, args.weights)
