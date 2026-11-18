"""Reproduce supplemental development gauge audits and direct handle interventions."""

import argparse
from dataclasses import replace
import hashlib
import math
from pathlib import Path
from unittest.mock import patch

import torch

from p5.study import bit_loss
from p5b.diagnostics import SCALES, erased, gain_check
from p6 import axial_em_transformer as solver
from p6 import experiment as ex
from p6.data import SNR, dev_frame, dev_seed
from p6.diagnostics import degenerate

ARMS = ("pure", "local", "parity", "local_parity")
EXTREMES = (1e-35, 1., 1e20)


def gauge_rows(model, regime, pair, arm, design, updates, *, full_grid=False):
    val = dev_frame(regime, 6, pair, 2)
    nv0 = replace(val.obs, noise_var=torch.zeros_like(val.obs.noise_var))
    noiseless = dev_frame(regime, 6, pair, 2, snr=math.inf).obs
    conditions = {k: erased(val.obs, k) for k in
                  ("intact", "first_pilot_row", "last_pilot_row", "all_pilots", "whole_field")}
    conditions.update(degenerate_pilots=degenerate(val.obs), zero_noise=nv0,
                      physical_noiseless=noiseless, all_pilots_zero_noise=erased(nv0, "all_pilots"),
                      whole_zero_noise=erased(nv0, "whole_field"))
    rows = []
    for i, (condition, obs) in enumerate(conditions.items()):
        audit = gain_check(model, obs, SCALES if full_grid or condition == "intact" else EXTREMES,
                           dev_seed(regime, 6, pair, 945 + i))
        rows.append(dict(pair=pair, regime=regime, arm=arm, design=design, updates=updates,
                         condition=condition, seed=dev_seed(regime, 6, pair), batch_size=2, **audit))
    return rows


@torch.no_grad()
def handle_interventions():
    """Intervene separately after the full-grid readout in both EM iterations."""
    rows = []
    physical_handles = solver._physical_handles
    for regime in SNR:
        f = dev_frame(regime, 6, 902, 2)
        model = solver.AxialEMTransformer("physics_parity").eval()
        base = model(f.obs)[0]
        for name in solver.Handles._fields:
            seeds = [dev_seed(regime, 6, 902, 551 + j) for j in range(model.steps)]
            step_seeds = iter(seeds)

            def intervene(*args):
                handles = physical_handles(*args)
                tensor = getattr(handles, name)
                gen = torch.Generator().manual_seed(next(step_seeds))
                perturbed = .02 * torch.randn(tensor.shape, dtype=tensor.dtype, generator=gen)
                if name == "delays":
                    perturbed[:, 0] = 0.
                return handles._replace(**{name: perturbed})

            with patch.object(solver, "_physical_handles", intervene):
                altered = model(f.obs)[0]
            rows.append(dict(regime=regime, pair=902, stream=6, seeds=seeds, handle=name,
                             max_llr_difference=float((altered - base).abs().max())))
    return rows


def summarize(rows):
    worst = max(rows, key=lambda x: x["max_grid_llr_diff"])
    return dict(checks=len(rows), passed_checks=sum(x["passed"] for x in rows),
                coded_flips=sum(x["coded_flips"] for x in rows),
                decisions=sum(x["decisions"] for x in rows),
                max_grid_llr_diff=worst["max_grid_llr_diff"],
                worst_llr_case={k: worst[k] for k in ("pair", "regime", "arm", "design", "condition")},
                max_relative_cfr_l2=max(x["max_relative_cfr_l2"] for x in rows),
                passed=all(x["passed"] for x in rows))


def once_updated():
    rows = []
    for pair in (902, 903):
        for regime in SNR:
            train = dev_frame(regime, 5, pair, 4)
            for arm in ARMS:
                model = ex.model_for(regime, pair, arm, 2)
                optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.003)
                bit_loss(train, model).backward()
                optimizer.step()
                rows.extend(gauge_rows(model.eval(), regime, pair, arm, "one_update", 1))
    return dict(rows=rows, summary=summarize(rows), handle_interventions=handle_interventions())


def trained():
    if ex.EVAL_BATCH != 32 or ex.VAL_BATCHES != 1:
        raise ValueError("Trained audit requires the final single-B32 validation design")
    pair = 901
    rows, checks = [], []
    for regime in SNR:
        for design, arms in (("main", ARMS), ("depth3", ex.PARITY_ARMS)):
            for arm in arms:
                folder = ex.job_dir(regime, pair, design)
                ck = folder / f"{arm}.pt"
                state = torch.load(ck, map_location="cpu", weights_only=True)
                result = ex.read(folder / f"{arm}.json")
                signature = ex.implementation_id(design)
                if signature != state["implementation"] or signature != result["implementation"]:
                    raise ValueError(f"Checkpoint/fit receipts do not match current source: {ck}")
                if "64" not in result["rows"] or result["score_seeds"] != [dev_seed(regime, 6, pair)]:
                    raise ValueError(f"Missing final 64-update B32 evaluation receipt: {ck}")
                checks.append(dict(file=ck.relative_to(ex.ROOT).as_posix(),
                                   implementation=signature, sha256=hashlib.sha256(ck.read_bytes()).hexdigest()))
                model = ex.model_for(regime, pair, arm, ex.DESIGNS[design].depth)
                model.load_state_dict(state["state"])
                rows.extend(gauge_rows(model.eval(), regime, pair, arm, design, 64))
    return dict(rows=rows, summary=summarize(rows), checkpoints=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("once_updated", "trained"))
    args = parser.parse_args()
    ex.setup()
    result = once_updated() if args.action == "once_updated" else trained()
    result["solver_sha256"] = hashlib.sha256(Path(solver.__file__).read_bytes()).hexdigest()
    ex.save(ex.RESULTS / f"extended_{args.action}.json", result)
    print(result["summary"], flush=True)
    if not result["summary"]["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
