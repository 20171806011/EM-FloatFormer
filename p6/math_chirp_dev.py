"""R2 prospective dev contrast. Three late-pass operators, the original 48-step objective."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from pathlib import Path

import torch

from p6 import axial_em_transformer as solver, experiment as ex, iter5_dev as old
from p6 import math_chirp
from p6.data import PAIRS, dev_frame, dev_seed
from p6.extended_audit import gauge_rows, summarize
from p6.inspect_iter3_paths import r2_observations, sign_summary
from p6.run_dev_iter2 import digest


OUT = ex.RESULTS / "dev_math" / "chirp"
MODES = ("tempered", "tangent", "hybrid", "conditional", "conditional_gn", "fisher")
HASH = hashlib.sha256((digest(Path(math_chirp.__file__)) + digest(ex.ROOT / "math_chirp_fisher.py")).encode()).hexdigest()
SOLVER_HASH = digest(Path(solver.__file__))


def folder(mode, source, pair):
    dev_seed(2, 6, pair)
    return OUT / mode / source / str(pair)


def checkpoint(model, path):
    row = ex.read(path)
    weights = path.with_suffix(".pt")
    if digest(weights) != row["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint hash disagrees with {path}")
    stored = torch.load(weights, weights_only=True, map_location="cpu")
    if stored["implementation"] != row["implementation"]:
        raise ValueError(f"Stored identity differs from the fit receipt at {path}")
    model.load_state_dict(stored["state"])
    return row


def one(key):
    action, mode, source, pair, arm = key
    dest = folder(mode, source, pair)
    score_file = dest / f"{arm}.json"
    model = old.model_for(2, pair, arm, .05, False)
    model.chirp_mode = mode
    previous_identity = None
    if source == "legacy":
        r = checkpoint(model, old.location(2, pair, arm, 48, .05, False))
        previous_identity = r["checkpoint_sha256"]
    elif action == "gauge":
        r = checkpoint(model, score_file)
        previous_identity = r["checkpoint_sha256"]
    if action == "gauge":
        rows = gauge_rows(model.eval(), 2, pair, arm, mode, 48, full_grid=True)
        summary = summarize(rows)
        ex.save(dest / f"{arm}_gauge.json", dict(mode=mode, pair=pair, arm=arm, rows=rows,
                summary=summary, weights=previous_identity, math_chirp_sha256=HASH,
                solver_source_import_sha256=SOLVER_HASH))
        return (key, summary["passed_checks"], summary["max_grid_llr_diff"])
    if source == "fit":
        old.optimize(model, 48, lambda update: dev_frame(2, 5, pair, ex.FIT_BATCH, update))
        ex.save_weights(score_file.with_suffix(".pt"), model, HASH)
        previous_identity = digest(score_file.with_suffix(".pt"))
    frame = dev_frame(2, 6, pair, 32)
    with torch.inference_mode():
        output, rays = r2_observations(model.eval(), frame)
        score = ex.score([frame], lambda _: output)
        signs = sign_summary([rays])["all"]
    ex.save(score_file, dict(mode=mode, pair=pair, arm=arm, score=score, signs=signs,
                            source=source, checkpoint_sha256=previous_identity,
                            implementation=HASH, solver_source_import_sha256=SOLVER_HASH,
                            init_seed=dev_seed(2, 5, pair, 900), score_seed=dev_seed(2, 6, pair)))
    return (key, score["block_errors"], signs["repaired"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("score", "gauge"))
    parser.add_argument("--source", choices=("legacy", "fit"), default="fit")
    parser.add_argument("--mode", nargs="+", choices=MODES, default=MODES)
    parser.add_argument("--pair", nargs="+", type=int, choices=PAIRS, default=PAIRS)
    parser.add_argument("--arm", nargs="+", choices=ex.ARMS, default=ex.PARITY_ARMS)
    parser.add_argument("--workers", type=int, choices=(1,2,3), default=3)
    a = parser.parse_args()
    jobs = [(a.action, m, a.source, p, arm) for m in a.mode for p in a.pair for arm in a.arm]
    with ProcessPoolExecutor(max_workers=a.workers, initializer=ex.setup) as pool:
        for row in pool.map(one, jobs):
            print(row, flush=True)


if __name__ == "__main__":
    main()
