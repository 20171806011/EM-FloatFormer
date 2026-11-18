"""R1 development factorial: bridge, pilot-referenced PA, and their combination.

Run all three pairs and both parity arms for each candidate at the original
48-update warm-cosine schedule. Prefer a candidate with a strict Axial word
improvement, no Local word regression and a positive fixed-weight intervention;
break ties by fewer changed mechanisms. Otherwise retain the Iteration 5 law.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch

from p6 import experiment as ex, iter5_dev, study_dev
from p6.data import PAIRS, dev_frame, dev_seed
from p6.diagnostics import anchor_check


OUT = ex.RESULTS / "dev_math/pa"
OPTIONS = {"bridge": (True, False), "pilot_pa": (False, True), "both": (True, True),
           "tangent": (True, False)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fit(key):
    pair, arm, mode = key
    bridge, differential = OPTIONS[mode]
    model = iter5_dev.model_for(1, pair, arm, .05, False)
    model.r1_bridge = bridge
    model.r1_differential = differential
    model.r1_tangent = mode == "tangent"
    valid = dev_frame(1, 6, pair, 32)
    anchor = anchor_check(model, valid.obs, arm, 3)
    if not anchor["p6_bit_exact"] or max(anchor["archived_max_llr_cfr"]) != 0.:
        raise AssertionError(f"Fixed anchor changed for {key}: {anchor}")
    initial = study_dev.initial_weights(model)
    sources = {p: sha(ex.ROOT / p) for p in ("math_pa.py", "math_pa_dev.py", "axial_em_transformer.py",
                                               "axial_operator.py", "iter5_dev.py", "training_iter3.py")}
    iter5_dev.optimize(model, 48, lambda i: dev_frame(1, 5, pair, 4, i))
    model.eval()
    def score():
        return ex.score([valid], lambda f: model(f.obs))
    full = score()
    off = {}
    if bridge:
        with patch.object(model, "r1_bridge", False):
            off["bridge"] = score()
    if differential:
        with patch.object(model, "r1_differential", False):
            off["pilot_pa"] = score()
    if model.r1_tangent:
        with patch.object(model, "r1_tangent", False):
            off["tangent"] = score()
    if bridge and differential:
        with patch.object(model, "r1_bridge", False), patch.object(model, "r1_differential", False):
            off["both"] = score()
    dest = OUT / mode / str(pair) / f"{arm}.json"
    dest.parent.mkdir(exist_ok=True, parents=True)
    ck = dest.with_suffix(".pt")
    ex.save_weights(ck, model, hashlib.sha256(repr(sources).encode()).hexdigest())
    record = dict(regime=1, pair=pair, arm=arm, mode=mode, bridge=bridge, differential=differential,
                  steps=48, depth=3, sources=sources, anchor=anchor, score=full, suppressed=off,
                  init_sha256=initial, checkpoint_sha256=sha(ck), parameters=ex.real_parameters(model),
                  init_seed=dev_seed(1,5,pair,900), fit_seeds=[dev_seed(1,5,pair,i) for i in range(48)],
                  score_seed=dev_seed(1,6,pair))
    ex.save(dest, record)
    print(key, full["block_errors"], {k:v["block_errors"] for k,v in off.items()}, flush=True)
    return mode, arm, record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tangent-only", action="store_true",
                        help="Blind waveform slope posterior follow-up to the declared factorial")
    args = parser.parse_args()
    modes = ("tangent",) if args.tangent_only else ("bridge", "pilot_pa", "both")
    keys = [(p,a,m) for m in modes for p in PAIRS for a in ex.PARITY_ARMS]
    groups = {}
    with ProcessPoolExecutor(max_workers=2, initializer=ex.setup) as pool:
        for mode, arm, record in pool.map(fit, keys, chunksize=1):
            groups.setdefault(mode, {}).setdefault(arm, []).append(record)
    summary = {}
    for mode, arms in groups.items():
        summary[mode] = {}
        for arm, rows in arms.items():
            errs = [r["score"]["block_errors"] for r in rows]
            summary[mode][arm] = dict(by_pair=errs, errors=sum(errs),
                suppressed={k:sum(r["suppressed"][k]["block_errors"] for r in rows)
                            for k in rows[0]["suppressed"]})
    ex.save(OUT / ("tangent_summary.json" if args.tangent_only else "summary.json"), summary)
    print(summary, flush=True)
