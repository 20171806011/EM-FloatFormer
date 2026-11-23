"""Score a frozen finite-state encoder on grouped, chronological paired questions."""
from functools import lru_cache
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fourphase_paired_questions"))
import paired

TOLERANCE = 3e-13
ROOTS = np.asarray(paired.ROOTS)


def pack(model, bits, first_windows, second_windows):
    """Only this one fixed-width integer per record crosses the cache boundary."""
    if bits not in (4, 8):
        raise ValueError("supported total cache widths are four and eight bits")
    ids = []
    for windows in (first_windows, second_windows):
        value = np.asarray(model.encode(windows))
        if value.shape != (len(windows),) or value.dtype.kind not in "iu":
            raise ValueError("encode must return one integer per window")
        if np.any(value < 0) or np.any(value >= 2 ** (bits // 2)):
            raise ValueError("window code exceeds assigned budget")
        ids.append(value.astype(np.uint8))
    if ids[0].shape != ids[1].shape:
        raise ValueError("data windows must be paired one to one")
    return (ids[0] << (bits // 2)) | ids[1]


def _indices(codes, table):
    states = len(table)
    if states not in (4, 16) or codes.dtype != np.uint8 or codes.ndim != 1:
        raise ValueError("invalid cache state format")
    if np.any(codes >= states * states):
        raise ValueError("packed code exceeds the selected budget")
    return codes // states, codes % states


def difference(left, right):
    """Posterior of right-minus-left (mod 4), summing in canonical order for ties."""
    sums = []
    for k in range(4):
        products = left * np.roll(right, -k, axis=-1)
        sums.append(np.sort(products, axis=-1).sum(axis=-1))
    return np.stack(sums, axis=-1)


def first_readout(codes, table):
    left, right = _indices(codes, table)
    return difference(table[left], table[right])


def second_readout(codes, table, references):
    first, _ = _indices(codes, table)
    shared = {}
    for ref in references:
        if ref.shares_channel:
            samples = np.asarray(ref.samples)
            if samples.shape != (len(first), paired.N):
                raise ValueError("reference batch must match cached records")
            if ref.source in shared and not np.array_equal(samples, shared[ref.source]):
                raise ValueError("one acquisition ID points to conflicting samples")
            shared[ref.source] = samples
    if not shared:
        return np.full((len(first), 4), 0.25)
    # Each distinct acquisition from the data channel contributes its likelihood once.
    amplitude = sum(window.sum(axis=1) for window in shared.values())
    logits = (2 * paired.SNR / np.sqrt(paired.N)) * (amplitude[:, None] * ROOTS.conj()).real
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    pilot = weights / np.sort(weights, axis=1).sum(axis=1, keepdims=True)
    return difference(pilot, table[first])


@lru_cache(maxsize=2)
def _parents(split):
    samples = list(paired.parents(split))
    ids, symbols, data, pilot, independent = zip(*samples)
    return ids, np.array(symbols), np.array(data), np.array(pilot), np.array(independent)


def _scores(prediction, labels):
    hits = (np.argmax(prediction, axis=1) == labels)
    probability = prediction[np.arange(len(labels)), labels]
    if np.any(probability <= 0):
        raise ValueError("a true label was assigned zero probability")
    return hits, -np.log(probability)


def evaluate(model, bits, split):
    """Return metrics and per-mother paired [Q1 accuracy,NLL, valid Q2 accuracy,NLL]."""
    ids, labels, data, pilot, unrelated = _parents(split)
    n = len(ids)
    table = np.asarray(model.table)
    if (table.shape != (2 ** (bits // 2), 4) or not np.isfinite(table).all()
            or np.any(table < 0) or not np.allclose(table.sum(axis=1), 1, atol=2e-12, rtol=0)):
        raise ValueError("decoding table must contain normalized distributions")
    refs = paired.reference_views(pilot, unrelated)
    q1, q2, codes, r_labels, a_labels = [], [], [], [], []
    for s1, s2 in paired.ROTATIONS:
        code = pack(model, bits, data[:, 0] * ROOTS[s1], data[:, 1] * ROOTS[s2])
        codes.append(code)
        # First readout completes before any reference is provided to the second.
        q1.append(first_readout(code, table))
        q2.append({arm: second_readout(code, table, refs[arm]) for arm in paired.ARMS})
        r_labels.append((labels[:, 1] - labels[:, 0] + s2 - s1) % 4)
        a_labels.append((labels[:, 0] + s1) % 4)
    checks = {}
    examples = {"first_failed_pair": None, "first_nonuniform_success": None}
    fails_by_property = {name: np.zeros(n, dtype=bool) for name in ("rotate", "hold", "duplicate")}
    worst_per_parent = np.zeros(n)
    max_conf_increase = 0.0

    def check(name, expected, actual, prop, arm=None, variant=None):
        error = np.abs(actual - expected)
        residual = error.max(axis=1)
        if not np.isfinite(error).all():
            raise ValueError(f"non-finite posterior residual for {name}")
        failed = residual > TOLERANCE
        fails_by_property[prop] |= failed
        np.maximum(worst_per_parent, residual, out=worst_per_parent)
        entry = checks.setdefault(name, dict(rows=0, failed_rows=0, max_abs=0.0, tv_sum=0.0))
        entry["max_abs"] = max(entry["max_abs"], float(residual.max()))
        entry["failed_rows"] += int(failed.sum())
        entry["rows"] += len(failed)
        entry["tv_sum"] += float(error.sum() / 2)
        if failed.any() and examples["first_failed_pair"] is None:
            i = int(np.flatnonzero(failed)[0])
            examples["first_failed_pair"] = dict(mother=ids[i], check=name, arm=arm,
                variant=variant, expected=expected[i].tolist(), actual=actual[i].tolist())

    for j, (s1, s2) in enumerate(paired.ROTATIONS):
        if j:
            expected_r = np.roll(q1[0], s2 - s1, axis=1)
            check(("q1_second_rotates" if s1 == 0 else "q1_both_invariant"),
                  expected_r, q1[j], "rotate" if s1 == 0 else "hold", variant=j)
            for arm in paired.ARMS:
                expected_a = np.roll(q2[0][arm], s1, axis=1)
                check(("q2_second_invariant" if s1 == 0 else "q2_both_rotates"),
                      expected_a, q2[j][arm], "hold" if s1 == 0 else "rotate", arm, j)
        check("independent_equals_none", q2[j]["none"], q2[j]["independent"],
              "hold", "independent", j)
        check("independent_uniform", np.full_like(q2[j]["none"], 0.25),
              q2[j]["independent"], "hold", "independent", j)
        check("duplicate_equals_valid", q2[j]["valid"], q2[j]["duplicate"],
              "duplicate", "duplicate", j)
        increase = q2[j]["duplicate"].max(axis=1) - q2[j]["valid"].max(axis=1)
        max_conf_increase = max(max_conf_increase, float(increase.max()))
        rotated_code = pack(model, bits, data[:, 0] * ROOTS[(s1 + 1) % 4],
                            data[:, 1] * ROOTS[(s2 + 1) % 4])
        rotated_pilot = (paired.Reference(0, True, pilot * 1j),)
        check("coordinate_q1_invariant", q1[j], first_readout(rotated_code, table), "hold", variant=j)
        check("coordinate_q2_invariant", q2[j]["valid"], second_readout(rotated_code, table, rotated_pilot),
              "hold", "valid", j)

    for entry in checks.values():
        entry["mean_tv"] = entry.pop("tv_sum") / entry["rows"]
    variants = len(q1)
    by_mother = np.zeros((n, 4))
    arms = {}
    corrected = harmed = 0
    all_correct = np.ones(n, dtype=bool)
    for arm in paired.ARMS:
        rows = []
        for j in range(variants):
            rel_hit, rel_loss = _scores(q1[j], r_labels[j])
            abs_hit, abs_loss = _scores(q2[j][arm], a_labels[j])
            measurements = np.stack([rel_hit, rel_loss, abs_hit, abs_loss], axis=1)
            rows.append(measurements)
            if arm == "valid":
                by_mother += measurements / variants
                all_correct &= rel_hit & abs_hit
                old_hit = np.argmax(q2[j]["none"], axis=1) == a_labels[j]
                corrected += int(np.sum(abs_hit & ~old_hit))
                harmed += int(np.sum(~abs_hit & old_hit))
        arms[arm] = dict(zip(("q1_accuracy", "q1_nll", "q2_accuracy", "q2_nll"),
                             np.mean(rows, axis=(0, 1)).tolist()))
    eligible = np.flatnonzero(all_correct & (worst_per_parent <= TOLERANCE)
                              & (q1[0].max(axis=1) > 0.26) & (q2[0]["valid"].max(axis=1) > 0.26))
    if len(eligible):
        i = int(eligible[0])
        examples["first_nonuniform_success"] = {
            "mother": ids[i], "base_q1": q1[0][i].tolist(), "second_only_q1": q1[1][i].tolist(),
            "both_q1": q1[2][i].tolist(), "base_valid_q2": q2[0]["valid"][i].tolist(),
            "both_valid_q2": q2[2]["valid"][i].tolist(),
            "base_duplicate_q2": q2[0]["duplicate"][i].tolist()}

    fake_independent_duplicate = (paired.Reference(0, True, pilot), paired.Reference(1, True, pilot))
    wrong = second_readout(codes[0], table, fake_independent_duplicate)
    wrong_tv = np.abs(wrong - q2[0]["valid"]).sum(axis=1) / 2
    flattened = np.concatenate(codes)
    surviving_ids = np.unique(np.concatenate(_indices(flattened, table)))
    failed_union = np.logical_or.reduce(list(fails_by_property.values()))
    valid = arms["valid"]
    nontrivial = (valid["q1_accuracy"] > 0.25 and valid["q2_accuracy"] > 0.25
                  and valid["q1_nll"] < np.log(4) and valid["q2_nll"] < np.log(4))
    return {
        "split": split, "mothers": n, "paired_rows": n * variants * len(paired.ARMS),
        "tolerance": TOLERANCE, "arms": arms, "checks": checks, "examples": examples,
        "three_properties": {name: {"mothers_failed": int(mask.sum()), "pass": not bool(mask.any())}
                             for name, mask in fails_by_property.items()},
        "all_three": {"mothers_failed": int(failed_union.sum()), "strict_vectors_pass": not bool(failed_union.any()),
                      "nontrivial_answers": bool(nontrivial), "pass": bool(nontrivial and not failed_union.any())},
        "valid_update": {"paired_windows": n * variants, "corrected": corrected, "new_errors": harmed},
        "negative_control": {"mean_tv_double_count_on_original": float(wrong_tv.mean()),
                             "max_tv_double_count_on_original": float(wrong_tv.max())},
        "duplicate_max_confidence_increase": max_conf_increase,
        "cache": {"bits": bits, "cell": "uint8", "packed_states_used": len(np.unique(flattened)),
                  "window_states_used": len(surviving_ids), "packed_states_allowed": 2**bits},
    }, by_mother
