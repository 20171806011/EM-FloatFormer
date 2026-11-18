"""R2 paired intervention on the learned path-time cue used for bank scoring."""
from unittest.mock import patch

import torch

from p6 import axial_em_transformer as solver
from p6 import experiment as ex
from p6 import run_dev_iter4 as study
from p6.data import PAIRS, dev_frame, dev_seed
from p6.handle_ablation import assert_replay
from p6.inspect_iter3_paths import r2_observations, sign_summary


@torch.inference_mode()
def run():
    result = dict(implementation=study.implementation(), regime=2, stream=6, pair_ids=PAIRS,
                  batch=0, batch_size=study.EVAL_BATCH, update=study.PRIMARY,
                  intervention="At every bank call, pass field_ray=None. All later effects are allowed "
                               "to propagate through the ordinary sharp bank, phase gate and receiver.",
                  arms={})
    compute = solver._soft_chirp_residual

    def without_cue(clean, x, basis, rot, field_ray):
        if field_ray is None:
            raise ValueError("Expected the learned cue at the intervention point")
        return compute(clean, x, basis, rot, field_ray=None)

    for arm in ex.PARITY_ARMS:
        by_pair = {}
        pooled_scores = {"full": {}, "no_cue": {}}
        rays = {"full": [], "no_cue": []}
        for pair in PAIRS:
            seed = dev_seed(2, 6, pair)
            frame = dev_frame(2, 6, pair, study.EVAL_BATCH)
            receipt = ex.read(study.location(2, pair, arm))
            if receipt["score_seed"] != seed:
                raise ValueError("R2 validation draw differs from training receipt")
            model = study.restored(2, pair, study.DEPTH, arm)
            by_pair[pair] = dict(seed=seed, checkpoint_sha256=receipt["checkpoint_sha256"])
            full = r2_observations(model, frame)
            with patch.object(solver, "_soft_chirp_residual", side_effect=without_cue) as hook:
                ablated = r2_observations(model, frame)
                if hook.call_count != model.steps - 1:
                    raise ValueError("Unexpected number of R2 soft bank calls")
            for name, (outputs, row) in (("full", full), ("no_cue", ablated)):
                score = ex.score([frame], lambda _: outputs)
                if name == "full":
                    assert_replay(score, receipt["score"])
                by_pair[pair][name] = dict(score=score, sign_summary=sign_summary([row]))
                ex.add(pooled_scores[name], score)
                rays[name].append(row)
        pooled = {name: dict(score=pooled_scores[name], sign_summary=sign_summary(rows))
                  for name, rows in rays.items()}
        result["arms"][arm] = dict(by_pair=by_pair, pooled=pooled)
    return result


if __name__ == "__main__":
    ex.setup()
    result = run()
    ex.save(ex.RESULTS / "iter4_checks" / "chirp_cue.json", result)
    for name, arm in result["arms"].items():
        for intervention, row in arm["pooled"].items():
            print(name, intervention, row["score"]["block_errors"],
                  row["sign_summary"]["initial_wrong_sign"]["repaired"])
