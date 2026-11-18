"""One separate R3 candidate: steer the branch contraction into each LS ray's ULA frame.

Fit from scratch at the same 48-update endpoint, with the IRC candidate disabled.
Estimate spatial frequency only from adjacent-antenna LS products. Gate by their
measured coherence and by strength relative to 1/sqrt(number of known pilots).
Choose only with a strict decoding gain against Iteration 5, positive full vs
suppressed gain, preserved joint Green >= .34, and all required invariants.
"""

from concurrent.futures import ProcessPoolExecutor
from functools import partial
import hashlib
from unittest.mock import patch

import torch

from p6 import axial_em_transformer as solver
from p6 import experiment as ex, iter5_dev, run_dev_iter2 as prior
from p6 import math_spatial_dev as irc_study, iter4_spatial
from p6.data import PAIRS, dev_frame, dev_seed
from p6.diagnostics import anchor_check
from p6.math_spatial import apply_in_ray_frame


OUT = irc_study.OUT / "ula"


def fit(key):
    pair, arm = key
    model = iter5_dev.model_for(3, pair, arm, 0.05, False)
    val = dev_frame(3, 6, pair, 32)
    transport = partial(apply_in_ray_frame, pilot_count=int(val.obs.pilot_mask.sum()))
    probe_frame = dev_frame(3, 8, pair, 8) if arm == "parity" else None
    baseline = ex.read(iter5_dev.location(3, pair, arm, 48, 0.05, False))
    with patch.object(solver, "_apply_spatial", transport):
        with torch.no_grad():
            anchor = anchor_check(model, val.obs, arm, 3)
        if not anchor["p6_bit_exact"] or max(anchor["archived_max_llr_cfr"]) != 0:
            raise ArithmeticError("ULA fit breaks anchor")
        initial = (
            prior.probe(model, probe_frame, dev_seed(3, 9, pair))
            if probe_frame is not None
            else None
        )
        iter5_dev.optimize(model, 48, lambda i: dev_frame(3, 5, pair, ex.FIT_BATCH, i))
        full, details = iter5_dev.diagnose(model.eval(), 3, pair, val, initial)
        gradients = irc_study.objective_sensitivities(model, pair)
    unsteered = ex.score([val], lambda f: model(f.obs))
    path = OUT / str(pair) / f"{arm}.json"
    sources = {
        **irc_study.SOURCE,
        "math_ula_dev.py": prior.digest(ex.ROOT / "math_ula_dev.py"),
    }
    signature = hashlib.sha256(repr(sorted(sources.items())).encode()).hexdigest()
    ex.save_weights(path.with_suffix(".pt"), model, signature)
    rec = dict(
        pair=pair,
        regime=3,
        arm=arm,
        steps=48,
        config="ULA lag-aligned M squared",
        anchor=anchor,
        source_sha256=sources,
        parameters=ex.real_parameters(model),
        train_batch0_gradient=gradients,
        seed=dev_seed(3, 6, pair),
        init=baseline["initial_weights_sha256"],
        score=full,
        prior=baseline["score"],
        unsteered=unsteered,
        checkpoint_sha256=prior.digest(path.with_suffix(".pt")),
        **details,
    )
    ex.save(path, rec)
    print(
        pair,
        arm,
        full["block_errors"],
        details["spatial_off_score"]["block_errors"],
        flush=True,
    )


def summarize():
    out = {}
    for arm in ex.PARITY_ARMS:
        rows = [ex.read(OUT / str(pair) / f"{arm}.json") for pair in PAIRS]
        rowsums = {
            k: [r[k]["block_errors"] for r in rows]
            for k in ("prior", "score", "unsteered", "spatial_off_score")
        }
        out[arm] = dict(
            counts=rowsums,
            pooled={k: sum(v) for k, v in rowsums.items()},
            steering=iter4_spatial.pool([r["steering"] for r in rows])[
                "waveform_solve_3"
            ]["steering_cosine_after"],
            steering_off=iter4_spatial.pool([r["steering_off"] for r in rows])[
                "waveform_solve_3"
            ]["steering_cosine_after"],
        )
        if arm == "parity":
            out[arm]["green"] = prior.probe_summary(rows)
    ex.save(OUT / "summary.json", out)
    print({k: g["pooled"] for k, g in out.items()}, flush=True)


if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=2, initializer=ex.setup) as pool:
        list(pool.map(fit, [(pair, arm) for pair in PAIRS for arm in ex.PARITY_ARMS]))
    summarize()
