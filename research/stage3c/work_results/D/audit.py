"""Independent CPU audit of the Stage 3C s17 decision certificate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.special import betaincinv, gammaln, logsumexp


TASKS = ("q1", "q3")
BATCH = 8192
RUNS = ((2**22, 967193), (2**24, 967229))
ROOTS = np.array([1, 1j, -1, -1j])


def histogram(encode, size, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    hist = np.zeros((2, 16, 4), dtype=np.int64)
    scale = 1 / np.sqrt(2 * 10**0.2)
    for _ in range(size // BATCH):
        symbols = rng.integers(0, 4, (2, BATCH))
        increments = rng.normal(0.7, 0.1 * np.sqrt(2), (2, BATCH))
        carriers = ROOTS[symbols]
        directions = (carriers[0], carriers[1] * np.exp(1j * increments[0]),
                      carriers[0] * np.exp(1j * increments.sum(axis=0)))
        windows = []
        for direction in directions:
            real = rng.standard_normal((BATCH, 3))
            imag = rng.standard_normal((BATCH, 3))
            windows.append(direction[:, None] / np.sqrt(3) + scale * (real + 1j * imag))
        r1, j1 = np.divmod(encode(windows[0]), 4)
        r2, j2 = np.divmod(encode(windows[1]), 4)
        summed = windows[2].sum(axis=1)
        # Sector via signed distance to the two diagonal boundaries.
        diagonal = summed.real + summed.imag > 0
        antidiagonal = summed.imag - summed.real > 0
        sector = np.select((diagonal & antidiagonal, ~diagonal & antidiagonal,
                            ~diagonal & ~antidiagonal), (1, 2, 3))
        cells = 4 * j1 + j2
        labels = ((symbols[1] - symbols[0] - r2 + r1) % 4, (sector - r1) % 4)
        for table, label in zip(hist, labels):
            np.add.at(table, (cells, label), 1)
    assert np.all(hist.sum(axis=(1, 2)) == size)
    return hist


def upper(hits, total, alpha):
    return 1.0 if hits == total else 1 - float(betaincinv(total - hits, hits + 1, alpha))


def log_tail(chosen, rival):
    # Sum the entire right tail in log space, including underflowed probabilities.
    total = chosen + rival
    k = np.arange(chosen, total + 1)
    log_mass = gammaln(total + 1) - gammaln(k + 1) - gammaln(total - k + 1)
    return float((logsumexp(log_mass) - total * np.log(2)) / np.log(10))


def verify_orbits(table, decisions, read_pair):
    covered = set()
    for row in range(16):
        j1, j2 = divmod(row, 4)
        for a in range(4):
            for b in range(4):
                pair = (4*a + j1, 4*b + j2)
                assert pair not in covered
                covered.add(pair)
                decoded = read_pair(table, *pair)
                assert decoded["q1"]["prediction"] == (decisions[0, row] + b - a) % 4
                assert decoded["q3"]["prediction"] == (decisions[1, row] + a) % 4
    return len(covered)


def audit(first, second, decisions, baseline):
    rows = []
    summaries = {}
    confirmation = baseline["confirmation"]
    expected_tests = {(r["task"], r["cell"], r["rival"]): r
                      for r in confirmation["comparisons"]}
    assert len(expected_tests) == confirmation["n_tests"] == 96
    max_errors = {"first_stage_bound": 0.0, "log10_p": 0.0, "p": 0.0}

    def compare(field, actual, expected, tolerance):
        error = abs(actual - expected)
        max_errors[field] = max(max_errors[field], error)
        assert error <= tolerance, (field, actual, expected)

    for t, task in enumerate(TASKS):
        reference = baseline["tasks"][task]
        assert first[t].tolist() == [r["counts"] for r in reference["cells"]]
        assert second[t].tolist() == confirmation["counts"][task]
        size = int(first[t].sum())
        correct = 0
        row_bounds = []
        strict_rows = 0
        for row, d in enumerate(decisions[t].tolist()):
            chosen = int(first[t, row, d])
            correct += chosen
            ref_row = reference["cells"][row]
            assert ref_row["cell"] == row and ref_row["chosen"] == d
            rivals = {x["rival"]: x for x in ref_row["comparisons"]}
            gaps = []
            shares = []
            assert set(rivals) == set(range(4)) - {d}
            for k in sorted(set(range(4)) - {d}):
                saved = rivals[k]
                count = int(first[t, row, k])
                pair = chosen + count
                mass = upper(pair, size, 0.005 / 96)
                share = upper(count, pair, 0.035 / 96)
                gap = mass * max(0, 2*share - 1)
                for field, value in (("pair_mass_upper", mass), ("share_upper", share),
                                     ("positive_gap_upper", gap)):
                    compare("first_stage_bound", value, saved[field], 1e-12)
                gaps.append(gap)
                shares.append(share)
                c2, k2 = int(second[t, row, d]), int(second[t, row, k])
                log_p = log_tail(c2, k2)
                p = 10**log_p
                expected = expected_tests[task, row, k]
                assert (d, c2, k2) == (expected["chosen"], expected["chosen_count"],
                                        expected["rival_count"])
                compare("log10_p", log_p, expected["log10_p"], 1e-8)
                assert math.isclose(p, expected["p"], rel_tol=1e-8, abs_tol=1e-307)
                max_errors["p"] = max(max_errors["p"], abs(p - expected["p"]))
                rows.append(dict(task=task, row=row, chosen=d, rival=k, first_chosen=chosen,
                                 first_rival=count, share_upper=share, pair_mass_upper=mass,
                                 positive_gap_upper=gap, second_chosen=c2, second_rival=k2,
                                 p=p, log10_p=log_p))
            row_bounds.append(max(gaps))
            strict_rows += max(shares) < 0.5
            compare("first_stage_bound", row_bounds[-1], ref_row["regret_upper"], 1e-12)
        regret = sum(row_bounds)
        interval = [1 - upper(size-correct, size, 0.00125), upper(correct, size, 0.00125)]
        bayes_upper = min(1, interval[1] + regret)
        assert strict_rows == reference["certified_orbits"]
        for actual, expected in zip(
            [correct/size, *interval, regret, bayes_upper],
            [reference["accuracy"], *reference["accuracy_interval"],
             reference["regret_upper"], reference["bayes_accuracy_upper"]],
        ):
            compare("first_stage_bound", actual, expected, 1e-12)
        summaries[task] = dict(decisions=decisions[t].tolist(), correct=correct,
                               accuracy=correct/size, accuracy_interval=interval,
                               error_interval=[1-interval[1], 1-interval[0]],
                               bayes_accuracy_upper=bayes_upper, regret_upper=regret,
                               strict_rows=strict_rows)
    weakest = max(rows, key=lambda r: r["log10_p"])
    assert weakest["p"] <= 0.005
    assert math.isclose(weakest["p"], confirmation["global_p"], rel_tol=1e-8)
    return dict(summaries=summaries, weakest=weakest, global_p=weakest["p"],
                comparisons=rows, max_absolute_differences=max_errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    research = args.repo.resolve() / "research"
    lookup = (research / "stage3b_math/A/c4_s17_lookup.json").read_text(encoding="utf-8")
    weights = research / "matched_cache/weights/equivariant_b8_s17.npz"
    table_sha = hashlib.sha256(lookup.encode()).hexdigest()
    weight_sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    assert table_sha == "5d257d63f148b668b7a430f2fce37f2443bc386f275cd05c18d34f5f0355d75e"
    assert weight_sha == "e67d9fbafb90791ab19549de7789798d818286d7f52d7ac566cc39df71e78356"
    released = json.loads(lookup)
    table = released["cells"]
    decisions = np.array([np.argmax([r[t] for r in table], axis=1) for t in TASKS])
    assert all(np.all(np.sort([r[t] for r in table], axis=1)[:, -1] >
                      np.sort([r[t] for r in table], axis=1)[:, -2]) for t in TASKS)
    sys.path.insert(0, str(research / "stage3b_math/A"))
    from c4_lookup import read_pair

    coverage = verify_orbits(released, decisions, read_pair)
    sys.path.insert(0, str(research / "matched_cache"))
    from learned import load

    model = load(weights)
    assert model.equivariant and len(model.centers) == 4
    first, second = [histogram(model.encode, size, seed) for size, seed in RUNS]
    baseline = json.loads((research / "stage3c/D/results.json").read_text(encoding="utf-8"))
    result = audit(first, second, decisions, baseline)
    result.update(runs=RUNS, batch=BATCH, cache_pairs=coverage,
                  counts={"first": first.tolist(), "second": second.tolist()},
                  table_sha256=table_sha, weight_sha256=weight_sha,
                  versions=dict(python=platform.python_version(), numpy=np.__version__,
                                scipy=scipy.__version__))
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
