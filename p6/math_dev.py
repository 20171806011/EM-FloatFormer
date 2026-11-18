"""Run objective controls and verify the integrated recipe on development pairs."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import torch

from p6 import experiment as ex, iter5_dev as old, study_dev
from p6.data import PAIRS, SNR, dev_frame, dev_seed
from p6.diagnostics import anchor_check
from p6.extended_audit import ARMS as GAUGED, gauge_rows, summarize
from p6.math_training import Recipe, receiver, train
from p6.run_dev_iter2 import digest, probe

OUT = ex.RESULTS / "dev_math"


def training_sources():
    files = (
        "axial_em_transformer.py",
        "axial_operator.py",
        "parity.py",
        "data.py",
        "math_training.py",
        "training_iter3.py",
        "experiment.py",
        "math_dev.py",
        "math_pa.py",
        "math_chirp.py",
        "math_chirp_fisher.py",
        "math_chirp_trust.py",
        "math_spatial.py",
    )
    return {n: digest(ex.ROOT / n) for n in files if (ex.ROOT / n).is_file()}


def recipe_for(label):
    if label == "loss_control":
        return Recipe(loss="brier")
    return Recipe(**ex.read(OUT / "final/recipe.json"))


def path_for(label, regime, pair, arm):
    dev_seed(regime, 5, pair)
    base = OUT if label == "loss_control" else OUT / "final"
    return base / label / f"r{regime}" / str(pair) / f"{arm}.json"


def run(key):
    action, regime, pair, arm = key
    label = "loss_control" if action == "loss_control" else "selected"
    recipe = recipe_for(label)
    source = training_sources()
    model = receiver(arm, dev_seed(regime, 5, pair, 900), recipe)
    path = path_for(label, regime, pair, arm)
    if action in ("gauge", "old_gauge"):
        ck = (path if action == "gauge" else old.location(regime, pair, arm, 48, 0.05, False)).with_suffix(
            ".pt"
        )
        sha = digest(ck)
        if sha != ex.read(ck.with_suffix(".json"))["checkpoint_sha256"]:
            raise ValueError(f"Wrong checkpoint {ck}")
        model.load_state_dict(torch.load(ck, weights_only=True)["state"])
        rows = gauge_rows(model.eval(), regime, pair, arm, action, 48, full_grid=True)
        ex.save(
            path_for(action, regime, pair, arm),
            dict(
                rows=rows,
                summary=summarize(rows),
                source=source,
                recipe=asdict(recipe),
                checkpoint_sha256=sha,
            ),
        )
        print(regime, pair, arm, action, summarize(rows)["max_grid_llr_diff"], flush=True)
        return
    validation = dev_frame(regime, 6, pair, 32)
    anchor = anchor_check(model, dev_frame(regime, 6, pair, 2).obs, arm, 3)
    initial = study_dev.initial_weights(model)
    initial_probe = (
        probe(model, dev_frame(3, 8, pair, 8), dev_seed(3, 9, pair))
        if regime == 3 and arm in ("parity", "pure")
        else None
    )
    gradients = {}

    def first_check(params):
        for (name, p), param in zip(model.head.named_parameters(), params, strict=True):
            if p is not param or p.grad is None:
                raise ValueError("Missing parameter gradient")
            v = torch.view_as_real(p.grad) if p.grad.is_complex() else p.grad
            gradients[name] = dict(
                active=int(v.count_nonzero()), size=v.numel(), finite=bool(v.isfinite().all())
            )

    train(model, lambda i: dev_frame(regime, 5, pair, 4, i), recipe, first_check)
    score, details = old.diagnose(model.eval(), regime, pair, validation, initial_probe)
    ex.save_weights(path.with_suffix(".pt"), model, repr(recipe))
    ex.save(
        path,
        dict(
            recipe=asdict(recipe),
            regime=regime,
            pair=pair,
            arm=arm,
            score=score,
            source=source,
            checkpoint_sha256=digest(path.with_suffix(".pt")),
            initial_sha=initial,
            anchor=anchor,
            gradients=gradients,
            parameters=ex.real_parameters(model),
            **details,
        ),
    )
    print(regime, pair, arm, action, score["block_errors"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("loss_control", "selected", "gauge", "old_gauge"))
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--regime", type=int, nargs="+", choices=SNR, default=SNR)
    args = parser.parse_args()
    arms = (
        ex.PARITY_ARMS
        if args.action == "loss_control"
        else (ex.ARMS if args.action == "selected" else GAUGED)
    )
    keys = [(args.action, r, p, a) for r in args.regime for p in PAIRS for a in arms]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=ex.setup) as pool:
        list(pool.map(run, keys, chunksize=1))
