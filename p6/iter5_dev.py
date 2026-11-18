"""Paired Iteration 5 fits. Each schedule gets its own four-update warmup and cosine endpoint."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import time
from unittest.mock import patch

import torch

from p6 import axial_em_transformer as solver
from p6 import experiment as ex
from p6 import iter4_spatial, run_dev_iter2 as prior, study_dev, training_iter3
from p6.data import PAIRS, SNR, dev_frame, dev_seed
from p6.inspect_iter3_paths import r2_observations, sign_summary


OUT = ex.RESULTS / "dev_iter5"
CONFIG = ex.Design(depth=3, lr=.0045, schedule="warmcos", clip=1., aux_nmse=.005)


@lru_cache(maxsize=1)
def signature():
    names = ("axial_em_transformer.py", "axial_operator.py", "data.py", "parity.py",
             "training_iter3.py", "experiment.py", "iter5_dev.py", "iter4_spatial.py",
             "inspect_iter3_paths.py", "study_dev.py", "run_dev_iter2.py")
    return hashlib.sha256(repr([(n, prior.digest(ex.ROOT / n)) for n in names]).encode()).hexdigest()


def model_for(regime, pair, arm, cue, defer):
    model = ex.model_for(regime, pair, arm, CONFIG.depth)
    model.cue_weight = cue
    model.defer_r1_pa = defer
    return model


def location(regime, pair, arm, steps, cue, defer):
    dev_seed(regime, 6, pair)
    if arm not in ex.ARMS or steps not in (36, 48) or cue not in (0., .05):
        raise ValueError("Unexpected dev condition")
    # Only one copy of a fit if the requested switch is inactive in its regime.
    setting = {1: f"defer{int(defer)}", 2: f"cue{cue:g}", 3: "shrinkage"}[regime]
    return OUT / "fits" / f"r{regime}" / f"{steps}_{setting}" / str(pair) / f"{arm}.json"


def optimize(model, steps, batch_at):
    parameters = list(model.head.parameters())
    optimizer = torch.optim.Adam(parameters, lr=CONFIG.lr)
    for index in range(steps):
        frame = batch_at(index)
        optimizer.param_groups[0]["lr"] = ex.learning_rate(CONFIG, index, steps)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = training_iter3.objective(frame, model, aux_nmse=CONFIG.aux_nmse)
        if not torch.isfinite(loss):
            raise ArithmeticError(f"Nonfinite training loss at update {index + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, CONFIG.clip, error_if_nonfinite=True)
        optimizer.step()


@torch.inference_mode()
def diagnose(model, regime, pair, frame, initial_probe):
    details = {}
    if regime == 2:
        outputs, rays = r2_observations(model, frame)
        details["signs"] = sign_summary([rays])["all"]
    elif regime == 3 and model.variant in ex.PARITY_ARMS:
        outputs, stages = iter4_spatial.measure(model, frame)
        read = solver._physical_handles
        def suppressed(*args, **kwargs):
            handles = read(*args, **kwargs)
            return handles._replace(spatial=torch.zeros_like(handles.spatial))
        with patch.object(solver, "_physical_handles", suppressed):
            off_outputs, off_stages = iter4_spatial.measure(model, frame)
            if initial_probe:
                details["probe_off"] = prior.probe(model, dev_frame(3, 8, pair, 8), dev_seed(3, 9, pair))
        details.update(steering=stages, steering_off=off_stages,
                       spatial_off_score=ex.score([frame], lambda _: off_outputs))
    else:
        outputs = model(frame.obs)
    if initial_probe:
        details["probe"] = dict(initial=initial_probe, trained=prior.probe(
            model, dev_frame(3, 8, pair, 8), dev_seed(3, 9, pair)))
    return ex.score([frame], lambda _: outputs), details


def run(key):
    regime, pair, arm, steps, cue, defer = key
    dest = location(*key)
    ck = dest.with_suffix(".pt")
    if dest.exists():
        record = ex.read(dest)
        if record["implementation"] == signature() and ck.exists() and prior.digest(ck) == record["checkpoint_sha256"]:
            return
    model = model_for(regime, pair, arm, cue, defer)
    initial = study_dev.initial_weights(model)
    initial_probe = None
    if regime == 3 and arm in ("pure", "parity"):
        initial_probe = prior.probe(model, dev_frame(3, 8, pair, 8), dev_seed(3, 9, pair))
    start = time.perf_counter()
    optimize(model, steps, lambda update: dev_frame(regime, 5, pair, ex.FIT_BATCH, update))
    elapsed = time.perf_counter() - start
    frame = dev_frame(regime, 6, pair, 32)
    score, details = diagnose(model.eval(), regime, pair, frame, initial_probe)
    ex.save_weights(ck, model, signature())
    ex.save(dest, dict(regime=regime, pair=pair, arm=arm, steps=steps,
                       cue=cue if regime == 2 else None, defer=defer if regime == 1 else None,
                       score_seed=dev_seed(regime, 6, pair), init_seed=dev_seed(regime, 5, pair, 900),
                       score=score, fit_seconds=elapsed, implementation=signature(),
                       initial_weights_sha256=initial, checkpoint_sha256=prior.digest(ck),
                       parameters=ex.real_parameters(model), **details))
    print(f"{key} {score['block_errors']} failed words", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", nargs="+", type=int, choices=SNR, default=list(SNR))
    parser.add_argument("--arm", nargs="+", choices=ex.ARMS, default=ex.PARITY_ARMS)
    parser.add_argument("--steps", type=int, nargs="+", choices=(36, 48), default=(36, 48))
    parser.add_argument("--cue", type=float, nargs="+", choices=(0., .05), default=(.05, 0.))
    parser.add_argument("--defer", type=int, nargs="+", choices=(0, 1), default=(0, 1))
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    jobs = [(r, p, a, s, c, bool(d)) for r in args.regime for s in args.steps
            for c in (args.cue if r == 2 else (.05,)) for d in (args.defer if r == 1 else (0,))
            for p in PAIRS for a in args.arm]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=ex.setup) as pool:
        list(pool.map(run, jobs, chunksize=1))


if __name__ == "__main__":
    main()
