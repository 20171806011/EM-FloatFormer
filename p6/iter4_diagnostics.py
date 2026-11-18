"""Post-fit, paired interventions on the selected Iteration 4 parity receivers."""

from unittest.mock import patch

import torch

from p6 import axial_em_transformer as solver
from p6 import experiment as ex
from p6 import inspect_iter3_paths as signs
from p6 import run_dev_iter2 as prior
from p6 import run_dev_iter4 as study
from p6 import study_dev
from p6.data import dev_frame
from p6.handle_ablation import assert_replay


@torch.inference_mode()
def run_job(key):
    regime, pair, arm = key
    if arm not in ex.PARITY_ARMS:
        raise ValueError("Interventions require a parity arm")
    dest = study.location(*key)
    record = ex.read(dest)
    model = study.restored(regime, pair, study.DEPTH, arm)
    frame = dev_frame(regime, 6, pair, study.EVAL_BATCH)
    fixed = ex.read(study.location(regime, pair, "references"))["scores"]["physics_parity"]
    score = record["score"]
    suppressed, spans = prior.suppressions(regime, [frame], model, fixed, score)
    physical = {}
    if regime == 1:
        with patch.object(model, "pa_dewarp", False):
            suppressed["emit_only"] = ex.score([frame], lambda f: model(f.obs))
    if regime == 2:
        physical["phase_passes"] = study_dev.r2_path_diagnostic(model, frame)
        outputs, rows = signs.r2_observations(model, frame)
        assert_replay(ex.score([frame], lambda _: outputs), score)
        physical["sign_summary"] = signs.sign_summary([rows])
        physical["ray_values"] = {k: v.flatten().tolist() for k, v in rows.items()}
        compute = solver._soft_chirp_residual
        with patch.object(solver, "_soft_chirp_residual",
                          lambda *args, **kwargs: torch.zeros_like(compute(*args, **kwargs))):
            suppressed["soft_bank"] = ex.score([frame], lambda f: model(f.obs))
        with patch.object(model, "steps", 2):
            suppressed["third_pass"] = ex.score([frame], lambda f: model(f.obs))
    if regime == 3:
        from p6.iter4_spatial import measure
        outputs, physical["spatial"] = measure(model, frame)
        assert_replay(ex.score([frame], lambda _: outputs), score)
        baseline = solver.AxialEMTransformer("physics_parity", steps=study.DEPTH).eval()
        outputs, physical["fixed_spatial"] = measure(baseline, frame)
        assert_replay(ex.score([frame], lambda _: outputs), fixed)
        assert_replay(suppressed["phase"], score)
    if regime != 1:
        assert_replay(suppressed["cpe"], score)
    ex.save(dest.with_name(f"{arm}_interventions.json"), dict(
        regime=regime, pair=pair, arm=arm, score_seed=record["score_seed"],
        implementation=study.implementation(), checkpoint_sha256=record["checkpoint_sha256"],
        suppressed=suppressed, spans=spans, physical=physical))
    print(f"R{regime}/{pair} {arm} interventions", flush=True)
