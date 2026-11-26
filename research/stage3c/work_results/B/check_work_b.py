"""CPU reproduction of Stage 3C B with independent Fourier and s17 checks."""
from dataclasses import replace
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / "research/matched_cache")]
from learned import load
from research.stage3_ministral_cache import data, schema


def fourier_sectors(mu, variance, terms):
    """Integrate the wrapped normal density over the four sector arcs."""
    n = np.arange(1, terms + 1)
    center = (np.pi / 2) * np.arange(4)
    coefficient = np.exp(-variance * n**2 / 2) * np.sin(n * np.pi / 4) / n
    return 0.25 + (2 / np.pi) * (np.cos((center[:, None] - mu) * n) @ coefficient)


def encoded_collision(encoder, turns, h, tp):
    """Use actual s17 on the four noiseless cardinal windows."""
    roots = schema.ROOTS
    windows = np.repeat(roots[:, None], schema.N_SAMPLES, axis=1) / np.sqrt(schema.N_SAMPLES)
    codes = encoder.encode(windows)
    assert len(set(codes)) == 4
    matching = []
    for x1, x2, gauge in product(range(4), repeat=3):
        code1 = codes[(x1 + gauge) % 4]
        code2 = codes[(x2 + gauge + turns * h) % 4]
        pilot = roots[(gauge + turns * tp) % 4]
        if code1 == code2 == codes[0] and pilot == 1:
            matching.append([(x2 - x1) % 4, x1])
    assert len(matching) == 1
    return matching[0]


def simulate_counts(encoder, split, blocks):
    """Count (packed cache, sector) per sign; only cache is seen by readout."""
    seed, original_blocks = data.SPLIT_SEEDS[split]
    name = "C4_cfo_ramp"
    original = schema.CONDITION_SPECS[name]
    counts = []
    while_running = (seed, blocks)
    data.SPLIT_SEEDS[split] = while_running
    try:
        for sign in (1, -1):
            schema.CONDITION_SPECS[name] = replace(original, omega=sign * original.omega)
            batch = data.generate_mothers(split, name)
            cache = 16 * encoder.encode(batch.y1) + encoder.encode(batch.y2)
            joint = 4 * cache + batch.k3_repeat
            counts.append(np.bincount(joint, minlength=1024).reshape(256, 4))
    finally:
        schema.CONDITION_SPECS[name] = original
        data.SPLIT_SEEDS[split] = (seed, original_blocks)
    return np.stack(counts)


def held_out_scores(train, test):
    # Joint counts, with uniform prior over signs and one pseudocount per cell.
    # Summing before normalization weights the sign by its cache likelihood.
    joint = train + 1
    known = joint / joint.sum(axis=-1, keepdims=True)
    hidden_joint = joint.sum(axis=0)
    hidden = hidden_joint / hidden_joint.sum(axis=-1, keepdims=True)
    metrics = {}
    for name, probs in (("known", known), ("hidden", np.broadcast_to(hidden, known.shape))):
        actions = probs.argmax(axis=-1)[..., None]
        metrics[name] = {
            "accuracy": float(np.take_along_axis(test, actions, axis=-1).sum() / test.sum()),
            "logloss_nats": float(-(test * np.log(probs)).sum() / test.sum()),
        }
    return metrics


def main():
    frozen = ROOT / "research/stage3c/B"
    original = json.loads(subprocess.check_output([sys.executable, "-B", str(frozen / "check_b.py")]))
    assert original == json.loads((frozen / "results.json").read_text())
    weights = ROOT / "research/matched_cache/weights/equivariant_b8_s17.npz"
    weight_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    encoder = load(weights)
    collisions = {
        "unwrapped_total_collision_q1_q2": [
            encoded_collision(encoder, 1, 1, 1), encoded_collision(encoder, 1, 2, 2)
        ],
        "wrapped_total_alias_with_same_times_q1_q2": [
            encoded_collision(encoder, 0, 1, 2), encoded_collision(encoder, 1, 1, 2)
        ],
    }
    assert all(values == original[key] for key, values in collisions.items())
    fourier = np.stack([fourier_sectors(0.1, v, 64) for v in (0.08, 8)])
    error = float(np.max(np.abs(fourier - original["diffusion_collision"]["q3_sector_probabilities"])))
    assert error < 1e-13
    train = simulate_counts(encoder, "s3_train", 512)
    test = simulate_counts(encoder, "s3_test", 256)
    scores = held_out_scores(train, test)
    assert train.sum() == 65536 and test.sum() == 32768
    print(json.dumps({
        "numpy_version": np.__version__,
        "checkpoint_sha256": weight_hash,
        "original_json_equal": True,
        **collisions,
        "fourier_sectors": fourier.tolist(),
        "fourier_max_abs_error": error,
        "oracle_sign": original["unknown_sign_oracle_u1"],
        "cache_only_c4": {
            "factory_base_seeds": {"s3_train": 840101, "s3_test": 840303},
            "effective_rng_seeds": {
                split: data.SPLIT_SEEDS[split][0] * 100
                + sorted(schema.CONDITION_SPECS).index("C4_cfo_ramp")
                for split in ("s3_train", "s3_test")
            },
            "sign_order": [1, -1],
            "train_mothers_per_sign": train.sum(axis=(1, 2)).tolist(),
            "test_mothers_per_sign": test.sum(axis=(1, 2)).tolist(),
            "observed_test_cache_states_per_sign": (test.sum(axis=-1) > 0).sum(axis=1).tolist(),
            "scores": scores,
        },
    }, indent=2))


if __name__ == "__main__":
    main()
