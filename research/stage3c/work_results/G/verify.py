"""Independent CPU audit using the pinned encoder, C4 simulator and readers."""
import hashlib
import json
import platform
from itertools import product
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from research.matched_cache.learned import load
from research.stage3_ministral_cache.data import SPLIT_SEEDS, generate_mothers
from research.stage3_ministral_cache.schema import (
    CONDITION_SPECS, ROOTS, matched_sum, sector_of_complex,
)
from research.stage3c.verify import compare
from research.stage3c.A.q2_reader import read as q2

BASE = ROOT / "research/stage3c"
LOOKUP = ROOT / "research/stage3b_math/A/c4_s17_lookup.json"
TASKS = ("q1", "q2", "q3")
PAIRS = list(product(range(16), repeat=2))
GROUP = list(product(range(4), repeat=2))
Q2_KERNEL = json.loads((BASE / "A/q2_s17_kernel.json").read_text())


def shifts(a, b):
    return b - a, a, a


def move(code, turns):
    return (code + 4 * turns) % 16


def representative(reader, z, w, task, context):
    r, i = divmod(z, 4)
    s, j = divmod(w, 4)
    return np.roll(reader(i, j, task, context), shifts(r, s)[task])


def run():
    original = json.loads(subprocess.check_output([
        sys.executable, "-B", str(BASE / "G/check_g.py"), str(LOOKUP),
    ]))
    compare(json.loads((BASE / "G/results.json").read_text()), original, "original G", rel_tol=2e-13, abs_tol=2e-13)
    table = json.loads(LOOKUP.read_text())
    with tempfile.TemporaryDirectory() as tmp:
        rebuilt = Path(tmp) / "lookup.json"
        integration = json.loads(subprocess.check_output([
            sys.executable, "-B", str(LOOKUP.with_name("c4_lookup.py")), "build",
            str(ROOT / "research/matched_cache"), "--table", str(rebuilt),
        ], cwd=ROOT))
        new = json.loads(rebuilt.read_text())
        table_error = max(abs(a-b)
                          for x, y in zip(table["cells"], new["cells"])
                          for t in ("q1", "q3") for a, b in zip(x[t], y[t]))
        assert table_error < 2e-12
    weight = ROOT / "research/matched_cache/weights/equivariant_b8_s17.npz"
    digest = hashlib.sha256(weight.read_bytes()).hexdigest()
    assert digest == table["weights_sha256"]
    encoder = load(weight)
    condition = "C4_cfo_ramp"
    batch = generate_mothers("s3_test", condition)
    z, w = (encoder.encode(x) for x in (batch.y1, batch.y2))
    packed = (z << 4) | w
    assert packed.min() >= 0 and packed.max() <= 255
    pilot = [(0, True, complex(v)) for v in matched_sum(batch.p_valid1)]
    contexts = [(p,) for p in pilot]
    calls = 0

    def reader(i, j, task, context):
        nonlocal calls
        calls += 1
        if task == 1:
            return q2(Q2_KERNEL, i, j, context)
        return table["cells"][4*i+j][TASKS[task]]

    fixed = contexts[0]
    exact = 0
    max_direct_q2_error = 0.0
    for x, y in PAIRS:
        reference = [representative(reader, x, y, t, fixed) for t in range(3)]
        for a, b in GROUP:
            u, v = move(x, a), move(y, b)
            for t, shift in enumerate(shifts(a, b)):
                observed = representative(reader, u, v, t, fixed)
                np.testing.assert_array_equal(observed, np.roll(reference[t], shift))
                if t == 1:
                    error = np.max(abs(observed - np.array(q2(Q2_KERNEL, u, v, fixed))))
                    max_direct_q2_error = max(max_direct_q2_error, float(error))
                exact += 1
    assert calls == 3 * len(PAIRS) + exact
    assert max_direct_q2_error < 2e-12

    labels = [(batch.x2 - batch.x1) % 4, batch.x1, batch.k3_repeat]
    n = len(z)
    outcomes = np.zeros((3, 3))
    base_probs = [
        np.array([representative(reader, int(x), int(y), t, context)
                  for x, y, context in zip(z, w, contexts)])
        for t in range(3)
    ]
    for a, b in GROUP:
        u, v = (encoder.encode(x) for x in batch.rotated_view(a, b))
        np.testing.assert_array_equal(u, move(z, a))
        np.testing.assert_array_equal(v, move(w, b))
        target3 = sector_of_complex(matched_sum(batch.y3_repeat * ROOTS[a]))
        np.testing.assert_array_equal(target3, (labels[2] + a) % 4)
        for t, shift in enumerate(shifts(a, b)):
            target = (labels[t] + shift) % 4
            probs = np.array([representative(reader, int(x), int(y), t, context)
                              for x, y, context in zip(u, v, contexts)])
            np.testing.assert_array_equal(probs, np.roll(base_probs[t], shift, axis=1))
            p_true = probs[np.arange(n), target]
            outcomes[t] += (
                np.sum(probs.argmax(axis=1) == target),
                np.sum(-np.log(p_true)),
                np.sum(1 - 2*p_true + np.sum(probs**2, axis=1)),
            )

    good, bad = .501, .001
    raw = np.array([bad] + [good] * 15)
    predictions = np.stack([raw, np.full(16, raw.mean()), np.full(16, raw[0])])
    brier = np.mean(2 * (1 - predictions)**2, axis=1)
    nll = np.mean(-np.log(predictions), axis=1)
    acc = np.mean(predictions > .5, axis=1)
    assert brier[1] < brier[0] < brier[2]
    assert nll[1] < nll[0] < nll[2]
    assert acc.tolist() == [15/16, 0, 0]

    # Calibrate the sharp 16x bound with a deterministic canonical truth.
    # Only the representative makes an error; the other predictions are Bayes.
    sharp = np.array([bad] + [1.0] * 15)
    sharp_brier = 2 * (1 - sharp)**2
    sharp_nll = -np.log(sharp)
    ratios = [float(loss[0]/loss.mean()) for loss in (sharp_brier, sharp_nll)]
    assert ratios == [16.0, 16.0]

    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "encoder_sha256": digest,
        "original_json_within_tolerance": True, "original": original,
        "rebuilt_lookup": integration, "lookup_max_delta": table_error,
        "lookup_seed": table["integration_seed"],
        "lookup_independent_noise_windows": table["independent_noise_windows"],
        "lookup_hermite_order": table["hermite_order"],
        "independent_exact_cache_identities": exact,
        "fixed_pilot_direct_q2_max_delta": max_direct_q2_error,
        "simulation": {
            "condition": condition, "split": batch.split,
            "rng_seed": SPLIT_SEEDS[batch.split][0]*100 + sorted(CONDITION_SPECS).index(condition),
            "mothers": n, "observed_cache_pairs_before_rotation": int(len(set(packed))),
            "distinct_valid_pilots": len(set(pilot)),
            "waveform_code_checks": 2*n*len(GROUP),
            "q3_sector_checks": n*len(GROUP),
            "probability_vector_checks": 3*n*len(GROUP),
            "scores_per_task": {
                t: dict(zip(("accuracy", "nll", "brier"), row))
                for t, row in zip(TASKS, outcomes / (n*len(GROUP)))
            },
        },
        "sharp_representative_excess_ratios": dict(zip(("brier", "nll"), ratios)),
        "counterexample_raw_mean_representative": {
            "accuracy": acc.tolist(), "brier": brier.tolist(), "nll": nll.tolist(),
        },
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, allow_nan=False))
