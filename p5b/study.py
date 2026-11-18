"""Paired P5-B fitting, archived-frame scoring, replay, and CPU timing."""

import argparse
import hashlib
import os
import random
import statistics
import time
from pathlib import Path

import torch

from p5.classical import METHODS
from p5.statistics import COUNT_FIELDS
from p5.study import (
    EVAL_BATCH, PAIRS, TEST_BATCHES, add, bit_loss, classical, counts,
    frozen, kernel_probe, read, rng, save, setup,
)
from p5.unrolled_em_transformer import UnrolledEMTransformer
from p5.wavefield_sim import simulate
from p5b.operator_em_transformer import OperatorEMTransformer


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ARCHIVE = ROOT.parent / "p5" / "results"
ARMS = ("pure", "parity", "local", "local_parity", "no_gauge")
ANCHORS = ("physics", "physics_parity")
PROBE_ARMS = ("pure", "parity")
TRAIN_STEPS = 64
FIT_BATCH = 4
LEARNING_RATE = .003
PHYSICS_STEPS = 2
VALID_EVERY = 16
PILOT_PAIR = 901


def fold(root, regime, pair):
    return root / f"r{regime}" / f"pair{pair:02d}"


def model_for(variant):
    return OperatorEMTransformer(variant=variant, steps=PHYSICS_STEPS)


def real_parameters(model):
    return sum(p.numel() * (2 if p.is_complex() else 1)
               for p in model.parameters() if p.requires_grad)


def new_models(regime, pair):
    models = {}
    for name in ARMS:
        torch.manual_seed(rng(regime, 4, pair))
        models[name] = model_for(name)
    counts_by_arm = {name: real_parameters(model) for name, model in models.items()}
    if len(set(counts_by_arm.values())) != 1:
        raise ValueError(f"Unmatched trainable counts: {counts_by_arm}")
    return models


def restore(model, checkpoint):
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    return model.eval()


def restored(regime, pair):
    dest = fold(RESULTS, regime, pair)
    return {name: restore(model_for(name), dest / f"{name}.pt") for name in ARMS}


def with_controls(models, regime, pair):
    archived = fold(ARCHIVE, regime, pair) / "full.pt"
    return {**models, **{name: model_for(name).eval() for name in ANCHORS},
            "p5_full": restore(UnrolledEMTransformer("full"), archived)}


def fit_models(regime, pair, snr, *, probe):
    models = new_models(regime, pair)
    optimizers = {name: torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                         lr=LEARNING_RATE)
                  for name, model in models.items()}
    validation = simulate(regime, 16, snr, rng(regime, 6, pair))
    trace = {name: [] for name in models}
    sums = dict.fromkeys(models, 0.)
    probes = {}
    if probe:
        probe_frame = simulate(regime, 8, snr, rng(regime, 8, pair))
        probe_seed = rng(regime, 9, pair)
        for name in PROBE_ARMS:
            probes[name] = {"initial": kernel_probe(models[name], probe_frame, probe_seed)}
    for step in range(TRAIN_STEPS + 1):
        if step % VALID_EVERY == 0:
            with torch.no_grad():
                for name, model in models.items():
                    bce = float(bit_loss(validation, model))
                    trace[name].append(dict(step=step, validation_bce=bce,
                                            train_mean_bce=sums[name] / VALID_EVERY if step else None))
                    sums[name] = 0.
        if step == TRAIN_STEPS:
            break
        # Shared Frame object means identical coded payload, paths, PA, oscillator and AWGN.
        frame = simulate(regime, FIT_BATCH, snr, rng(regime, 5, pair, step))
        for name, model in models.items():
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            loss = bit_loss(frame, model)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss, R{regime}/{pair}/{name}/{step}")
            loss.backward()
            optimizer.step()
            sums[name] += float(loss.detach())
    for model in models.values():
        model.eval()
    for name in probes:
        probes[name]["trained"] = kernel_probe(models[name], probe_frame, probe_seed)
    return models, trace, probes


def tensor_hash(tensor):
    array = tensor.detach().resolve_conj().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


@torch.no_grad()
def score(regime, pair, snr, models):
    models = with_controls(models, regime, pair)
    methods = [*models, *METHODS[regime]]
    rows = {name: {} for name in methods}
    frames = []
    for i in range(TEST_BATCHES):
        seed = rng(regime, 7, pair, i)
        frame = simulate(regime, EVAL_BATCH, snr, seed)
        frames.append(dict(seed=seed, batch=EVAL_BATCH, field_sha256=tensor_hash(frame.obs.y),
                           codewords_sha256=tensor_hash(frame.bits)))
        for name in methods:
            output = models[name](frame.obs) if name in models else classical(frame, name)
            add(rows[name], counts(frame, *output))
    params = {name: real_parameters(models[name]) if name in models else 0 for name in rows}
    return rows, frames, params


def compare_archive(regime, pair, rows):
    archived = read(fold(ARCHIVE, regime, pair) / "record.json")["rows"]
    matches = {}
    for name in ("p5_full", *METHODS[regime]):
        ref = archived["full" if name == "p5_full" else name]
        row = rows[name]
        matches[name] = dict(integer_counts_match=all(row[k] == ref[k] for k in COUNT_FIELDS),
                             all_fields_exact=row == ref,
                             channel_error_difference=row["channel_error"] - ref["channel_error"],
                             channel_energy_difference=row["channel_energy"] - ref["channel_energy"])
    if not all(entry["integer_counts_match"] for entry in matches.values()):
        raise ValueError(f"Regenerated P5 integer counts differ from the archive: {matches}")
    return matches


def pilot(regime):
    start = time.perf_counter()
    snr = frozen(regime)
    _, trace, _ = fit_models(regime, PILOT_PAIR, snr, probe=False)
    seconds = time.perf_counter() - start
    save(RESULTS / f"pilot_r{regime}.json",
         dict(regime=regime, pair=PILOT_PAIR, snr_db=snr, trace=trace, seconds=seconds,
              train_steps=TRAIN_STEPS, fit_batch=FIT_BATCH, lr=LEARNING_RATE,
              validation_seed=rng(regime, 6, PILOT_PAIR), formal_test_access=False))
    print(f"Pilot R{regime} pair {PILOT_PAIR}, {seconds:.1f}s", flush=True)


def fit(regime, pair):
    start = time.perf_counter()
    snr = frozen(regime)
    models, trace, probe = fit_models(regime, pair, snr, probe=regime == 3)
    dest = fold(RESULTS, regime, pair)
    dest.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        path = dest / f"{name}.pt"
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        torch.save(model.state_dict(), tmp)
        os.replace(tmp, path)
    train_seconds = time.perf_counter() - start
    rows, frames, params = score(regime, pair, snr, models)
    matches = compare_archive(regime, pair, rows)
    total_seconds = time.perf_counter() - start
    save(dest / "record.json",
         dict(regime=regime, pair=pair, snr_db=snr, train_steps=TRAIN_STEPS,
              fit_batch=FIT_BATCH, lr=LEARNING_RATE, physics_steps=PHYSICS_STEPS,
              initialization_seed=rng(regime, 4, pair), validation_seed=rng(regime, 6, pair),
              train_first_seed=rng(regime, 5, pair, 0),
              train_last_seed=rng(regime, 5, pair, TRAIN_STEPS-1),
              trace=trace, probe=probe, rows=rows, params=params, frames=frames,
              archive_comparisons=matches, train_seconds=train_seconds,
              total_seconds=total_seconds))
    print(f"R{regime} pair {pair} completed in {total_seconds:.1f}s", flush=True)


def replay(regime, pair):
    start = time.perf_counter()
    dest = fold(RESULTS, regime, pair)
    saved = read(dest / "record.json")
    models = restored(regime, pair)
    rows, frames, params = score(regime, pair, frozen(regime), models)
    matches = compare_archive(regime, pair, rows)
    probe_exact = None
    if regime == 3:
        frame = simulate(3, 8, frozen(3), rng(3, 8, pair))
        initial = new_models(3, pair)
        result = {name: {stage: kernel_probe(group[name], frame, rng(3, 9, pair))
                         for stage, group in (("initial", initial), ("trained", models))}
                  for name in PROBE_ARMS}
        probe_exact = saved["probe"] == result
    checks = dict(all_scores_exact=rows == saved["rows"], all_frames_exact=frames == saved["frames"],
                  all_params_exact=params == saved["params"],
                  archive_comparisons_exact=matches == saved["archive_comparisons"])
    result = dict(regime=regime, pair=pair, checkpoints=len(models), score_rows=len(rows),
                  test_batches=len(frames), probe_exact=probe_exact, checks=checks,
                  passed=all(checks.values()) and probe_exact is not False,
                  seconds=time.perf_counter() - start)
    if not result["passed"]:
        raise AssertionError(f"Checkpoint replay mismatch: {result}")
    print(f"Replayed R{regime} pair {pair}", flush=True)
    return result


@torch.inference_mode()
def latency(regime):
    models = with_controls(restored(regime, 1), regime, 1)
    methods = [*models, *METHODS[regime]]
    snr = frozen(regime)
    rows = []
    for batch in (1, 64):
        frame = simulate(regime, batch, snr, rng(regime, 10, batch=batch))
        rng_order = random.Random(rng(regime, 11, batch=batch))
        samples = {name: [] for name in methods}
        for repeat in range(14):
            order = methods.copy()
            rng_order.shuffle(order)
            for name in order:
                start = time.perf_counter_ns()
                if name in models:
                    models[name](frame.obs)
                else:
                    classical(frame, name)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                if repeat >= 3:
                    samples[name].append(elapsed)
        for name, times in samples.items():
            q1, _, q3 = statistics.quantiles(times, n=4, method="inclusive")
            rows.append(dict(method=name, batch=batch,
                             params=real_parameters(models[name]) if name in models else 0,
                             ms=statistics.median(times), q1_ms=q1, q3_ms=q3, samples_ms=times))
    save(RESULTS / f"latency_r{regime}.json", rows)
    print(f"R{regime} timings completed", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=("pilot", "fit", "replay", "latency"))
    parser.add_argument("--regime", required=True, type=int, choices=PAIRS)
    parser.add_argument("--pair", type=int, action="append")
    args = parser.parse_args()
    setup()
    torch.set_num_interop_threads(1)
    if args.action in ("pilot", "latency"):
        if args.pair:
            parser.error(f"{args.action} selects its own fixed pair")
        (pilot if args.action == "pilot" else latency)(args.regime)
        return
    for pair in args.pair or range(1, PAIRS[args.regime] + 1):
        if not 1 <= pair <= PAIRS[args.regime]:
            parser.error("pair outside the registered population")
        (fit if args.action == "fit" else replay)(args.regime, pair)


if __name__ == "__main__":
    main()
