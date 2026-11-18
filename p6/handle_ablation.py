"""Matched leave-one-handle-out inference interventions on development pair 901."""

import math
from unittest.mock import patch

import torch

from p6 import axial_em_transformer as solver
from p6 import experiment as ex
from p6.study import metrics


def interventions(regime):
    names = {"cpe": ("cpe",), "delay": ("delays",), "delta_h": ("delta_h",),
             "variance": ("log_variance",)}
    if regime != 2:
        names["pa_forward_inverse"] = ("radial",)
    if regime != 1:
        names["path_phase"] = ("phase",)
    if regime == 3:
        names["spatial_irc"] = ("irc", "spatial")
    names["all"] = solver.Handles._fields
    return names


def assert_replay(row, reference):
    if row.keys() != reference.keys():
        raise ValueError("Replay and original scores have different fields")
    for key, expected in reference.items():
        if isinstance(expected, int):
            matches = row[key] == expected
        else:
            matches = math.isclose(row[key], expected, rel_tol=1.e-12, abs_tol=1.e-12)
        if not matches:
            raise ValueError(f"Score replay mismatch for {key}: {row[key]} vs {expected}")


@torch.no_grad()
def measure(regime, pair, arm, frames, signature):
    path = ex.job_dir(regime, pair, "main")
    receipt = ex.read(path / f"{arm}.json")
    state = torch.load(path / f"{arm}.pt", map_location="cpu", weights_only=True)
    seeds = [ex.dev_seed(regime, 6, pair, i) for i in range(ex.VAL_BATCHES)]
    if (receipt["implementation"] != signature or state["implementation"] != signature
            or receipt["score_seeds"] != seeds or receipt["pair"] != pair
            or receipt["regime"] != regime or receipt["arm"] != arm
            or receipt["design"] != "main"):
        raise ValueError("Checkpoint, training receipt, seeds or implementation disagree")
    model = ex.model_for(regime, pair, arm, ex.MAIN.depth).eval()
    model.load_state_dict(state["state"])
    full = ex.score(frames, lambda frame: model(frame.obs))
    assert_replay(full, receipt["rows"]["64"])
    rows = {"full": metrics(full)}
    original = solver._physical_handles
    for name, fields in interventions(regime).items():
        def suppress(*args):
            active = original(*args)
            return active._replace(**{f: torch.zeros_like(getattr(active, f)) for f in fields})

        with patch.object(solver, "_physical_handles", suppress):
            row = ex.score(frames, lambda frame: model(frame.obs))
        rows[name] = metrics(row)

    fixed_path = ex.job_dir(regime, pair, "references") / "references.json"
    fixed = ex.read(fixed_path)
    if fixed["implementation"] != ex.implementation_id("references") or fixed["score_seeds"] != seeds:
        raise ValueError("Reference receipt belongs to another experiment")
    assert_replay({key: rows["all"][key] for key in full}, fixed["rows"]["physics_parity"])
    changes = {name: {key: row[key] - rows["full"][key] for key in
                      ("block_errors", "bit_errors", "bler", "pre_ber", "bce", "nmse_db")}
               for name, row in rows.items() if name != "full"}
    return dict(score_seeds=seeds, full_replay_passed=True, all_removed_fixed_check_passed=True,
                rows=rows, removed_minus_full=changes)


def main():
    ex.setup()
    signature = ex.implementation_id("main")
    pair = 901
    regimes = {}
    for regime in ex.SNR:
        frames = ex.valid_frames(regime, pair)
        arms = {arm: measure(regime, pair, arm, frames, signature) for arm in ex.PARITY_ARMS}
        regimes[regime] = arms
        for arm, record in arms.items():
            for name, row in record["rows"].items():
                print(f"R{regime} {arm} {name:18} {row['block_errors']}/{row['blocks']} "
                      f"BCE {row['bce']:.6f} NMSE {row['nmse_db']:.3f} dB", flush=True)
    report = dict(pair=pair, stream=6, frames_per_regime=ex.EVAL_BATCH*ex.VAL_BATCHES,
                  steps=64, design="main", implementation=signature,
                  interpretation=("Each omitted output is zeroed after physical projection at every "
                                  "EM step; subsequent states and predictions are recomputed with the "
                                  "same trained weights. Effects may interact and need not add. The "
                                  "PA intervention also removes the variance term tied to its radial "
                                  "coefficients; variance alone removes the free log-variance output."),
                  arms=regimes)
    ex.save(ex.RESULTS / "handle_ablation.json", report)


if __name__ == "__main__":
    main()
