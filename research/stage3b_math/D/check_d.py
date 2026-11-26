"""Compare signed CFO readouts on the Stage 3 frozen 8-bit encoder, on CPU."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


BASELINE = "69fbd29acdb4eba7961d56176693786b6181477a"
WEIGHT_SHA = "e67d9fbafb90791ab19549de7789798d818286d7f52d7ac566cc39df71e78356"
CONDITION = "C4_cfo_ramp"
OMEGAS = (-0.35, -0.12, 0.0, 0.12, 0.35)
CAL_SEEDS = tuple(range(961001, 961017))
TEST_SEEDS = (962011, 962023, 962039, 962063)
BLOCKS_PER_CAL_SEED = 128
BLOCKS_PER_TEST_SEED = 64


def metrics(prob, label):
    assert np.allclose(prob.sum(axis=1), 1.0) and np.all(prob > 0)
    return np.column_stack((prob.argmax(1) == label, -np.log(prob[np.arange(len(label)), label])))


def direct_table(readout, encoder, condition, omega):
    z1, z2 = np.indices((16, 16)).reshape(2, -1)
    result = readout(encoder.table[z1], encoder.table[z2], (), replace(condition, omega=omega))
    return result["q3"].reshape(16, 16, 4)


def make_codes(module, encoder, condition, omega, seed, blocks):
    from unittest.mock import patch

    tag = f"track_d_{seed}"
    with patch.dict(module.SPLIT_SEEDS, {tag: (seed, blocks)}), patch.dict(
        module.CONDITION_SPECS, {CONDITION: replace(condition, omega=omega)}
    ):
        batch = module.generate_mothers(tag, CONDITION)
    return encoder.encode(batch.y1), encoder.encode(batch.y2), batch.k3_repeat


def fit_symmetry_tables(data, encoder, condition, omega):
    counts = np.zeros((4, 4, 4), dtype=np.int64)
    # Rotating W1 and the repeated W3 together leaves the law invariant. W2
    # may independently rotate by a quarter turn because X2 is uniform.
    for seed in CAL_SEEDS:
        z1, z2, labels = make_codes(data, encoder, condition, omega, seed, BLOCKS_PER_CAL_SEED)
        relative_sector = (labels - z1 // 4) % 4
        np.add.at(counts, (z1 % 4, z2 % 4, relative_sector), 1)
    pair = counts + 0.5
    one = counts.sum(axis=1) + 0.5
    return pair / pair.sum(-1, keepdims=True), one / one.sum(-1, keepdims=True), int(counts.sum(-1).min())


def paired_report(block_runs, left, right, draws):
    # Independent simulation blocks, stratified by test generator seed.
    diff = np.stack([r[left] - r[right] for r in block_runs])
    samples = np.stack([d[take].mean(1) for d, take in zip(diff, draws)]).mean(0)
    interval = np.quantile(samples, [0.025, 0.975], axis=0)
    return {
        "accuracy": [float(x) for x in (diff.mean(axis=(0, 1))[0], *interval[:, 0])],
        "nll_nats": [float(x) for x in (diff.mean(axis=(0, 1))[1], *interval[:, 1])],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", required=True, type=Path, help="checkout containing pinned Stage 3 sources and weights")
    ap.add_argument("--out", type=Path, help="optional output JSON destination, otherwise stdout")
    args = ap.parse_args()
    repo = args.repo.resolve()
    cache_dir = repo / "research" / "matched_cache"
    sys.path[:0] = [str(repo), str(cache_dir)]
    from research.stage3_ministral_cache import data
    from research.stage3_ministral_cache.analytic import readout_from_table
    from learned import load

    path = cache_dir / "weights" / "equivariant_b8_s17.npz"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == WEIGHT_SHA
    encoder = load(path)
    assert encoder.equivariant and len(encoder.centers) == 4
    condition = data.CONDITION_SPECS[CONDITION]
    batch = data.generate_mothers("s3_test", CONDITION, max_blocks=2)
    original = direct_table(readout_from_table, encoder, condition, condition.omega)
    prob = original[encoder.encode(batch.y1), encoder.encode(batch.y2)]
    replay = metrics(prob, batch.k3_repeat).mean(0)
    # The paired host report uses these 128 mothers with quarter-turn rotations.
    assert replay[0] == 0.6015625 and abs(replay[1] - 0.972946919135067) < 1e-9
    legacy_controls = {}
    z1, z2 = encoder.encode(batch.y1), encoder.encode(batch.y2)
    for name, value in (("signed", condition.omega), ("ignored", 0.0), ("reversed", -condition.omega)):
        t = direct_table(readout_from_table, encoder, condition, value)
        legacy_controls[name] = metrics(t[z1, z2], batch.k3_repeat).mean(0).tolist()
    reports = {}
    rng = np.random.default_rng(963001)
    draws = rng.integers(BLOCKS_PER_TEST_SEED, size=(len(TEST_SEEDS), 2000, BLOCKS_PER_TEST_SEED))
    for omega in OMEGAS:
        tables = {
            name: direct_table(readout_from_table, encoder, condition, value)
            for name, value in (("signed", omega), ("ignored", 0.0), ("reversed", -omega))
        }
        pair, one, least = fit_symmetry_tables(data, encoder, condition, omega)
        block_runs = []
        for seed in TEST_SEEDS:
            z1, z2, labels = make_codes(data, encoder, condition, omega, seed, BLOCKS_PER_TEST_SEED)
            rel = (labels - z1 // 4) % 4
            runs = {name: metrics(t[z1, z2], labels) for name, t in tables.items()}
            runs.update(lookup_both=metrics(pair[z1 % 4, z2 % 4], rel), lookup_first=metrics(one[z1 % 4], rel))
            block_runs.append({name: arr.reshape(BLOCKS_PER_TEST_SEED, 64, 2).mean(1) for name, arr in runs.items()})
        means = {n: np.stack([r[n] for r in block_runs]).mean(axis=(0, 1)).tolist() for n in block_runs[0]}
        pairs = {
            left + "_minus_" + right: paired_report(block_runs, left, right, draws)
            for left, right in (("signed", "ignored"), ("signed", "reversed"),
                                ("lookup_both", "lookup_first"), ("lookup_both", "signed"))
        }
        if omega == 0.0:
            assert means["signed"] == means["ignored"] == means["reversed"]
        reports[str(omega)] = {"metrics": means, "paired": pairs, "min_calibration_cell_n": least}
    summary = {
        "upstream_commit": BASELINE, "weight_sha256": WEIGHT_SHA, "generator": CONDITION,
        "h": condition.h, "ell": condition.ell, "sigma": condition.sigma_phi, "snr_db": condition.snr_db,
        "cal_seeds": CAL_SEEDS, "test_seeds": TEST_SEEDS, "bootstrap_seed": 963001, "bootstrap_draws": 2000,
        "n_cal_per_omega": 64 * len(CAL_SEEDS) * BLOCKS_PER_CAL_SEED,
        "n_test_per_omega": 64 * len(TEST_SEEDS) * BLOCKS_PER_TEST_SEED,
        "published_128_mother_replay": replay.tolist(), "same_128_mother_controls": legacy_controls,
        "metric_order": ["accuracy", "nll_nats"],
        "paired_stat_order": ["estimate", "bootstrap_2.5pct", "bootstrap_97.5pct"], "angles": reports
    }
    output = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.write_text(output, encoding="utf-8")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
