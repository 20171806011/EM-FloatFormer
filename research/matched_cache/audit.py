"""Independent checks for the frozen QPSK benchmark and packed-cache decoder."""
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "fourphase_paired_questions"
sys.path.insert(0, str(SOURCE))
import paired

BASE = "6462a6ae04c2d96a72b70d69b69e1ba21f36896e"
BLOBS = {
    "paired.py": "0b6911fed4ba8405251e624f270384d3dd36826e",
    "dev.json": "cbfaec7a32aa4e7939c6053884cf66b8195bc419",
    "test.json": "0e8331f1fb6e7b6e78e78512e2e196ffb7ded200",
    "check_discrete.py": "41105cd96bd1b63f47a2056c874b8cfa706ce256",
    "exact.json": "f9f4619bd74d284c52fc7c558c8cc60f8584b9ea",
}
PARTITIONS = {"train": (730241, 64), "dev": (410061, 4), "test": (518339, 16)}


def check_parent():
    for name, expected in BLOBS.items():
        data = (SOURCE / name).read_bytes().replace(b"\r\n", b"\n")
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        assert blob == expected, (name, blob)
    assert paired.Q1 == "第二个窗口的发射符号相对第一个窗口转了几个四分之一圈？"
    assert paired.Q2 == "第一个窗口原来发送的是哪个四相符号？"
    assert tuple(paired.ROTATIONS) == ((0, 0), (0, 1), (1, 1))
    assert tuple(paired.ARMS) == ("none", "valid", "independent", "duplicate")


def window_bytes(window):
    return b"".join(struct.pack("!dd", complex(z).real + 0.0, complex(z).imag + 0.0)
                    for z in window)


def orbit(window):
    """A byte key invariant to any of the four quarter turns."""
    return min(window_bytes(np.asarray(window) * r) for r in (1, 1j, -1, -1j))


def check_splits():
    all_windows = set()
    all_parents = set()
    summaries = {}
    expected_training = []
    old_settings = dict(paired.SPLITS)
    try:
        assert all(paired.SPLITS[s] == PARTITIONS[s] for s in ("dev", "test"))
        paired.SPLITS.update(PARTITIONS)
        for split, (seed, blocks) in PARTITIONS.items():
            keys = set()
            mothers = set()
            raw_order = hashlib.sha256()
            for i, (identity, _, windows, pilot, _) in enumerate(paired.parents(split)):
                assert identity == (split, i)
                # Individual checks include the valid pilot, so accidental reuse
                # in a different role or as part of a different pair is caught.
                item = [orbit(w) for w in (*windows, pilot)]
                assert not keys.intersection(item) and len(set(item)) == 3
                keys.update(item)
                mother_key = b"".join(sorted(item[:2]))
                assert mother_key not in mothers
                mothers.add(mother_key)
                if split == "train":
                    expected_training.extend(windows)
                for w in windows:
                    raw_order.update(window_bytes(w))
            assert len(mothers) == 64 * blocks
            assert not keys.intersection(all_windows)
            assert not mothers.intersection(all_parents)
            all_windows.update(keys)
            all_parents.update(mothers)
            summaries[split] = {
                "seed": seed, "mothers": len(mothers), "distinct_data_or_pilot_window_orbits": len(keys),
                "ordered_data_iq_big_endian_f64_sha256": raw_order.hexdigest(),
            }
    finally:
        paired.SPLITS.clear()
        paired.SPLITS.update(old_settings)
    return summaries, np.asarray(expected_training)


def check_decoders(model, bits):
    """Exercise every possible wire byte in the B-bit range against a triple sum."""
    import evaluate
    K = 2 ** (bits // 2)
    table = np.asarray(model.table)
    np.testing.assert_allclose(table.sum(1), 1, atol=2e-14)
    assert table.shape == (K, 4) and np.all(table >= 0)
    codes = np.arange(2**bits, dtype=np.uint8)
    n = len(codes)
    # Independent noisy phase likelihood from squared distances to pilot means.
    means = np.asarray([1, 1j, -1, -1j]) / np.sqrt(3)
    rng = np.random.default_rng(251123 + bits)
    y = rng.normal(size=(n, 3)) + 1j*rng.normal(size=(n, 3))
    w = np.exp(-paired.SNR * np.sum(abs(y[:, :, None] - means)**2, axis=1))
    phase = w / w.sum(axis=1, keepdims=True)
    first = np.zeros((n, 4))
    second = np.zeros((n, 4))
    for c in range(n):
        a, b = divmod(c, K)
        for u in range(4):
            for v in range(4):
                first[c, (v-u) % 4] += table[a, u] * table[b, v]
            for g in range(4):
                second[c, (u-g) % 4] += table[a, u] * phase[c, g]
    predicted_r = evaluate.first_readout(codes, table)
    valid = paired.Reference(104, True, y)
    other = paired.Reference(105, False, 1j*y)
    predicted_x = evaluate.second_readout(codes, table, (valid,))
    np.testing.assert_allclose(first, predicted_r, rtol=0, atol=3e-13)
    np.testing.assert_allclose(second, predicted_x, rtol=0, atol=3e-13)
    for refs in ((), (other,)):
        np.testing.assert_allclose(evaluate.second_readout(codes, table, refs), 0.25,
                                   rtol=0, atol=3e-13)
    np.testing.assert_array_equal(evaluate.second_readout(codes, table, (valid, valid)), predicted_x)
    np.testing.assert_array_equal(evaluate.second_readout(codes, table, (other, valid)), predicted_x)
    fresh_id = paired.Reference(106, True, y)
    overweight = evaluate.second_readout(codes, table, (valid, fresh_id))
    double_phase = phase**2
    double_phase /= double_phase.sum(axis=1, keepdims=True)
    double_answer = np.zeros_like(second)
    for c in range(n):
        a = c // K
        for u in range(4):
            for g in range(4):
                double_answer[c, (u-g) % 4] += table[a,u] * double_phase[c,g]
    np.testing.assert_allclose(overweight, double_answer, rtol=0, atol=3e-13)
    delta = np.max(np.abs(overweight-predicted_x))
    assert delta > 0.001, "positive-control pilot likelihood failed to change the answer"
    z1 = rng.normal(size=(64, 3)) + 1j*rng.normal(size=(64, 3))
    z2 = rng.normal(size=(64, 3)) + 1j*rng.normal(size=(64, 3))
    packed = evaluate.pack(model, bits, z1, z2)
    assert packed.dtype == np.uint8 and packed.shape == (64,)
    assert np.all(packed.astype(np.int64) < 2**bits)
    np.testing.assert_array_equal(packed // K, model.encode(z1))
    np.testing.assert_array_equal(packed % K, model.encode(z2))
    return {
        "all_packed_states": n,
        "q1_max_abs_delta": float(np.max(np.abs(first-predicted_r))),
        "q2_valid_max_abs_delta": float(np.max(np.abs(second-predicted_x))),
        "same_id_replay_max_abs_delta": 0,
        "fresh_id_same_values_max_change": float(delta),
        "pack_roundtrip_samples": len(packed),
    }


def main():
    from learned import load
    from physical import Hand
    from train import training_data
    check_parent()
    splits, expected = check_splits()
    train, soft = training_data()
    np.testing.assert_array_equal(train, expected)
    analytic_targets = np.array([paired.likelihood(w) for w in expected])
    np.testing.assert_allclose(soft, analytic_targets, rtol=0, atol=3e-13)
    models = {}
    for bits in (4, 8):
        models[f"hand_b{bits}"] = check_decoders(Hand(bits), bits)
        for name in ("plain", "equivariant"):
            for seed in (17, 29, 43):
                key = f"{name}_b{bits}_s{seed}"
                models[key] = check_decoders(load(HERE / "weights" / f"{key}.npz"), bits)
    print(json.dumps({
        "benchmark_commit": BASE,
        "frozen_blob_ids_verified": BLOBS,
        "split_checks": splits, "cross_split_raw_window_orbit_collisions": 0,
        "cross_split_parent_orbit_collisions": 0,
        "train_windows_exactly_matched": len(train),
        "train_targets_max_abs_difference_vs_frozen_likelihood": float(np.max(abs(soft-analytic_targets))),
        "decoder_cross_checks": models,
    }, indent=2))


if __name__ == "__main__":
    main()
