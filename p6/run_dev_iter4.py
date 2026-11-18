"""Fourth development run. All simulations pass through the dev split guard."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import lru_cache
import hashlib
import time

import torch

from p6 import experiment as ex
from p6 import run_dev_iter2 as prior
from p6 import study_dev, training_iter3
from p6.data import PAIRS, SNR, dev_frame, dev_seed


DEPTH = 3
PRIMARY = 48
CHECKPOINTS = (0, 24, 32, 48, 64)
TRAIN_BATCH, EVAL_BATCH, PROBE_BATCH = 4, 32, 8
CONFIG = ex.Design(depth=DEPTH, lr=.0045, schedule="warmcos", clip=1., aux_nmse=.005)
JOBS = ex.RESULTS / "iter4_jobs"


def objective(frame, model):
    return training_iter3.objective(frame, model, aux_nmse=CONFIG.aux_nmse)


def location(regime, pair, arm):
    dev_seed(regime, 6, pair)
    if arm not in (*ex.ARMS, "references"):
        raise ValueError(f"Unknown arm {arm}")
    return JOBS / f"r{regime}" / str(pair) / f"{arm}.json"


@lru_cache(maxsize=1)
def implementation():
    files = ("run_dev_iter4.py", "axial_operator.py", "axial_em_transformer.py", "parity.py",
             "data.py", "training_iter3.py", "experiment.py", "run_dev_iter2.py", "study_dev.py")
    return hashlib.sha256(repr([(f, prior.digest(ex.ROOT / f)) for f in files]).encode()).hexdigest()


def fit(regime, pair, arm, dest):
    model = ex.model_for(regime, pair, arm, DEPTH)
    initial_hash = study_dev.initial_weights(model)
    params = list(model.head.parameters())
    optimizer = torch.optim.Adam(params, lr=CONFIG.lr)
    frame = dev_frame(regime, 6, pair, EVAL_BATCH)
    probe_frame = (dev_frame(3, 8, pair, PROBE_BATCH)
                   if regime == 3 and arm in ("pure", "parity") else None)
    trace, probes = {}, {}
    fit_seconds, lr = 0., None
    for update in range(max(CHECKPOINTS) + 1):
        if update in CHECKPOINTS:
            model.eval()
            score, confidence = study_dev.validation(model, frame)
            trace[update] = dict(score=score, confidence=confidence, lr=lr, fit_seconds=fit_seconds)
            if update == PRIMARY:
                ex.save_weights(dest.with_suffix(".pt"), model, implementation())
            if probe_frame is not None and update in (0, PRIMARY):
                stage = "initial" if update == 0 else "trained"
                probes[stage] = prior.probe(model, probe_frame, dev_seed(3, 9, pair))
            if update == max(CHECKPOINTS):
                break
            model.train()
        start = time.perf_counter()
        batch = dev_frame(regime, 5, pair, TRAIN_BATCH, update)
        lr = ex.learning_rate(CONFIG, min(update, PRIMARY - 1), PRIMARY)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        loss, _ = objective(batch, model)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at R{regime}/{pair}/{arm}/{update + 1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, CONFIG.clip, error_if_nonfinite=True)
        optimizer.step()
        fit_seconds += time.perf_counter() - start
    return dict(trace=trace, probe=probes, score=trace[PRIMARY]["score"],
                initial_weights_sha256=initial_hash, parameters=ex.real_parameters(model),
                checkpoint_sha256=prior.digest(dest.with_suffix(".pt")), checkpoint_update=PRIMARY,
                first_train_seed=dev_seed(regime, 5, pair),
                primary_last_train_seed=dev_seed(regime, 5, pair, PRIMARY - 1),
                continuation_last_train_seed=dev_seed(regime, 5, pair, max(CHECKPOINTS) - 1),
                init_seed=dev_seed(regime, 5, pair, 900), fit_seconds=trace[PRIMARY]["fit_seconds"])


def run_job(key):
    regime, pair, arm = key
    dest = location(*key)
    ck = dest.with_suffix(".pt")
    if dest.exists():
        old = ex.read(dest)
        if old["implementation"] == implementation() and (arm == "references" or
                (ck.is_file() and old["checkpoint_sha256"] == prior.digest(ck))):
            return
    start = time.perf_counter()
    row = (prior.references(regime, pair, DEPTH) if arm == "references"
           else fit(regime, pair, arm, dest))
    ex.save(dest, dict(regime=regime, pair=pair, arm=arm, implementation=implementation(),
                       score_seed=dev_seed(regime, 6, pair), wall_s=time.perf_counter()-start, **row))
    print(f"R{regime}/{pair} {arm}: {row.get('score', {}).get('block_errors', 'reference')}", flush=True)


def restored(regime, pair, depth, arm):
    if depth != DEPTH or arm not in ex.ARMS:
        raise ValueError("Only selected trained three-pass checkpoints are available")
    dest = location(regime, pair, arm)
    rec = ex.read(dest)
    ck = dest.with_suffix(".pt")
    if (rec["implementation"] != implementation() or rec["checkpoint_update"] != PRIMARY
            or rec["checkpoint_sha256"] != prior.digest(ck)
            or (rec["regime"], rec["pair"], rec["arm"]) != (regime, pair, arm)):
        raise ValueError(f"Checkpoint and receipt disagree: {ck}")
    saved = torch.load(ck, weights_only=True, map_location="cpu")
    if saved["implementation"] != implementation():
        raise ValueError(f"Stale state: {ck}")
    model = ex.model_for(regime, pair, arm, depth).eval()
    model.load_state_dict(saved["state"])
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fit", "diagnostics", "latency"))
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--pair", nargs="+", type=int, choices=PAIRS, default=PAIRS)
    parser.add_argument("--regime", nargs="+", type=int, choices=SNR, default=tuple(SNR))
    parser.add_argument("--arm", nargs="+", choices=("references", *ex.ARMS),
                        default=("references", *ex.ARMS))
    args = parser.parse_args()
    ex.setup()
    if args.action == "latency":
        from p6.latency_environment import benchmark_environment
        with benchmark_environment() as env:
            for regime in args.regime:
                study_dev.latency(regime, 5, env, loader=restored, jobs_dir=JOBS, source_id=implementation)
        return
    if args.action == "diagnostics":
        from p6.iter4_diagnostics import run_job as function
        arms = [a for a in args.arm if a in ex.PARITY_ARMS]
    else:
        function, arms = run_job, args.arm
    jobs = [(r, p, a) for r in args.regime for p in args.pair for a in arms]
    if args.workers == 1:
        for job in jobs:
            function(job)
    else:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=ex.setup) as pool:
            list(pool.map(function, jobs, chunksize=1))


if __name__ == "__main__":
    main()
