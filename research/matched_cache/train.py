"""Fit the registered encoder/codebook pairs from fresh, training-only mothers."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "fourphase_paired_questions"))
import paired

from learned import fit, load

SEEDS = (17, 29, 43)
BUDGETS = (4, 8)
VARIANTS = ("plain", "equivariant")


def training_data():
    # Register a new stream without altering published dev/test parameters.
    paired.SPLITS["train"] = (730241, 64)
    windows = np.asarray([window for _, _, pair, _, _ in paired.parents("train") for window in pair])
    targets = np.asarray([paired.likelihood(window) for window in windows])
    assert windows.shape == (8192, 3) and targets.shape == (8192, 4)
    return windows, targets


def name(variant, bits, seed):
    return f"{variant}_b{bits}_s{seed}"


def train_one(setting):
    variant, bits, seed, folder = setting
    windows, targets = training_data()
    fitted = fit(windows, targets, bits, seed, equivariant=variant == "equivariant")
    output = folder / (name(variant, bits, seed) + ".npz")
    fitted.save(output)
    restored = load(output)
    np.testing.assert_array_equal(restored.encode(windows), fitted.encode(windows))
    np.testing.assert_array_equal(restored.table, fitted.table)
    return output.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "weights")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = [(v, b, s, args.output) for v in VARIANTS for b in BUDGETS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=args.jobs) as workers:
        for checkpoint in workers.map(train_one, settings):
            print(checkpoint, flush=True)


if __name__ == "__main__":
    main()
