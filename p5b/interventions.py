"""Hold final weights fixed and suppress one learned physical operator at inference."""

import time

import torch

from p5.study import EVAL_BATCH, PAIRS, TEST_BATCHES, add, counts, read, rng, save, setup
from p5.wavefield_sim import simulate
from p5b.operator_em_transformer import OperatorEMTransformer
from p5b.study import PHYSICS_STEPS, RESULTS, fold, model_for, restore, tensor_hash


# Components are indexed after complex centering and bilinear synthesis, before
# bounded physical coefficients. Component 2 supplies both phase and precision.
GROUPS = {
    "full": None,
    "without_radial": slice(0, 2),
    "without_phase_precision": slice(2, 3),
    "without_spectral_ici": slice(3, 5),
    "without_direct_channel": slice(5, 6),
    "all_off": slice(0, 6),
}
PRIMARY = {1: "pure", 2: "parity", 3: "parity"}


class IntervenedReceiver(OperatorEMTransformer):
    disabled = None

    def _operators(self, *args):
        grid, weights = super()._operators(*args)
        if self.disabled is not None:
            grid = grid.clone()
            grid[..., self.disabled] = 0
        return grid, weights


@torch.no_grad()
def evaluate_pair(regime, pair):
    dest = fold(RESULTS, regime, pair)
    saved = read(dest / "record.json")
    arm = PRIMARY[regime]
    model = restore(IntervenedReceiver(arm, steps=PHYSICS_STEPS), dest / f"{arm}.pt")
    anchor_name = "physics_parity" if arm == "parity" else "physics"
    anchor = model_for(anchor_name).eval()
    scores = {name: {} for name in GROUPS}
    for index in range(TEST_BATCHES):
        seed = rng(regime, 7, pair, index)
        frame = simulate(regime, EVAL_BATCH, saved["snr_db"], seed)
        expected = saved["frames"][index]
        if (seed != expected["seed"] or tensor_hash(frame.obs.y) != expected["field_sha256"]
                or tensor_hash(frame.bits) != expected["codewords_sha256"]):
            raise AssertionError(f"Formal frame mismatch R{regime} pair {pair} batch {index}")
        for name, disabled in GROUPS.items():
            model.disabled = disabled
            outputs = model(frame.obs)
            if name == "all_off" and not all(torch.equal(a, b) for a, b in zip(outputs, anchor(frame.obs))):
                raise AssertionError(f"All-off differs from the physical anchor R{regime} pair {pair} batch {index}")
            add(scores[name], counts(frame, *outputs))
    if scores["full"] != saved["rows"][arm] or scores["all_off"] != saved["rows"][anchor_name]:
        raise AssertionError(f"Official count/energy mismatch R{regime} pair {pair}")
    return dict(regime=regime, pair=pair, primary_tier=arm, physics_anchor=anchor_name,
                full_matches_formal=True, all_off_matches_formal_anchor=True,
                batches_matching_anchor_outputs_exactly=TEST_BATCHES, scores=scores)


def main():
    setup()
    torch.set_num_interop_threads(1)
    rows, pooled = [], []
    start = time.perf_counter()
    for regime, pairs in PAIRS.items():
        totals = {name: {} for name in GROUPS}
        for pair in range(1, pairs + 1):
            row = evaluate_pair(regime, pair)
            for name, counts_by_group in row["scores"].items():
                add(totals[name], counts_by_group)
            rows.append(row)
        blocks = totals["full"]["blocks"]
        for name, scores in totals.items():
            delta = scores["block_errors"] - totals["full"]["block_errors"]
            pooled.append(dict(regime=regime, arm=PRIMARY[regime], intervention=name,
                               block_error_change=delta, bler_change_pp=100 * delta / blocks, **scores))
        print(f"Completed R{regime} fixed-weight interventions on {pairs} seed pairs.", flush=True)
    save(RESULTS / "interventions.json",
         dict(primary_tiers=PRIMARY, meaning="Positive change means more errors after suppression.",
              interpretation="Fixed-weight intervention effects can interact and need not add up.",
              pairing="All official test batches, payload and field hashes verified against each pair record.",
              seconds=time.perf_counter()-start, per_pair=rows, pooled=pooled))


if __name__ == "__main__":
    main()
