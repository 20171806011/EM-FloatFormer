"""Iteration 3. Run fit, latency, and summarize from the repository root."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from functools import lru_cache
import hashlib
import inspect
import random
import statistics
import time
from unittest.mock import patch

import torch

from p6 import experiment as ex
from p6 import run_dev_iter2 as prior
from p6 import training_iter3 as training
from p6.data import PAIRS, SNR, dev_frame, dev_seed
from p6.green_support_control import run as support_control
from p6.handle_ablation import assert_replay
from p6.verify_iter2_checkpoints import path_diagnostic


JOBS = ex.RESULTS / "iter3_jobs"
DEPTH = 3
UPDATES = 80
CHECKPOINTS = (0, 24, 48, 64, 80)
TRAIN_BATCH = 4
EVAL_BATCH = 32
PROBE_BATCH = 8
CAP = 10.
CONFIG = ex.Design(depth=DEPTH, lr=.0045, schedule="warmcos", clip=1., aux_nmse=.04)
ARMS = {"capped": ("references", *ex.ARMS), "uncapped": ex.PARITY_ARMS}


def job_keys():
    for loss, arms in ARMS.items():
        for regime in SNR:
            for pair in PAIRS:
                for arm in arms:
                    yield loss, regime, pair, arm


def location(key):
    if key not in set(job_keys()):
        raise ValueError(f"Outside iteration-3 development design: {key}")
    loss, regime, pair, arm = key
    return JOBS / loss / f"r{regime}" / str(pair) / f"{arm}.json"


def initial_weights(model):
    digest = hashlib.sha256()
    for name, weight in model.head.named_parameters():
        digest.update(name.encode())
        digest.update(weight.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def validation(model, frame):
    """Preserve the raw logits for decoding, BCE, and confidence diagnostics."""
    llr, channel = model(frame.obs)
    score = ex.score([frame], lambda _: (llr, channel))
    coded = ex.payload(frame, llr)
    margin = coded * (2. * frame.bits.double() - 1.)
    confidence = dict(logit_abs_sum=coded.abs().sum().item(), bits=coded.numel(),
                      safe_margin_over_6=int((margin > 6.).sum()),
                      wrong_margin_below_minus_6=int((margin < -6.).sum()))
    return score, confidence


@torch.inference_mode()
def r2_path_diagnostic(model, frame):
    """Pair each inferred trajectory with simulation truth only after inference."""
    initial, increments = [], []
    initialize, read = prior.solver._initialize, prior.solver._physical_handles

    def capture_initial(*args, **kwargs):
        result = initialize(*args, **kwargs)
        initial.append(result[1])
        return result

    def capture_step(*args, **kwargs):
        handles = read(*args, **kwargs)
        increments.append(handles.phase)
        return handles

    with patch.object(prior.solver, "_initialize", capture_initial), patch.object(
            prior.solver, "_physical_handles", capture_step):
        model(frame.obs)
    truth = frame.truth.ray_evolution
    rotation = initial[0]
    # Unwrap each ray along time. Its constant phase is absorbed by path amplitude.
    product = rotation.mean(2) * truth.conj()
    inter_symbol = (product[:, 1:] * product[:, :-1].conj()).angle()
    unwrapped = torch.cat((torch.zeros_like(inter_symbol[:, :1]), inter_symbol), dim=1).cumsum(1)
    span = unwrapped.amax(1) - unwrapped.amin(1)
    quantiles = [0., .5, .95, 1.]

    def absolute_row_error(state):
        diff = state.mean(2) * truth.conj()
        unit = diff / diff.abs().clamp_min(torch.finfo(diff.real.dtype).tiny)
        return (unit * unit.mean(1, keepdim=True).conj()).angle().abs().mean(1)

    errors = [absolute_row_error(rotation)]
    for update in increments:
        rotation = rotation * torch.exp(1j * update)
        errors.append(absolute_row_error(rotation))
    groups = {}
    masks = {"over_0.5": span > .5, "over_0.5_through_3": (.5 < span) & (span <= 3.),
             "over_3": span > 3.}
    for label, mask in masks.items():
        count = int(mask.sum())
        groups[label] = dict(
            burst_rays=count,
            mean_absolute_wrapped_row_error_initial_then_steps=[
                e[mask].mean().item() if count else None for e in errors],
            improved_final_vs_initial=int((errors[-1][mask] < errors[0][mask]).sum()))
    return dict(units="radians", trajectories=span.numel(),
                phase_reference="One freely fitted constant phase per burst and ray.",
                mean_absolute_wrapped_row_error_initial_then_steps=[e.mean().item() for e in errors],
                initial_unwrapped_truth_minus_bank_span_quantiles=dict(
                    zip(quantiles, span.quantile(span.new_tensor(quantiles)).tolist(), strict=True)),
                groups_by_initial_unwrapped_truth_minus_bank_span=groups)


def fit(key, signature, dest):
    loss, regime, pair, arm = key
    model = ex.model_for(regime, pair, arm, DEPTH)
    init_digest = initial_weights(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(parameters, lr=CONFIG.lr)
    frame = dev_frame(regime, 6, pair, EVAL_BATCH)
    probe_frame = None
    probes = {}
    if loss == "capped" and regime == 3 and arm in ("pure", "parity"):
        probe_frame = dev_frame(regime, 8, pair, PROBE_BATCH)
        probes["initial"] = prior.probe(model, probe_frame, dev_seed(regime, 9, pair))

    trace = []
    objectives, raw_bces, norms = [], [], []
    lr = None
    seconds = 0.
    for update in range(UPDATES + 1):
        if update in CHECKPOINTS:
            model.eval()
            score, confidence = validation(model, frame)
            trace.append(dict(update=update, score=score, confidence=confidence, lr=lr,
                              mean_train_objective=statistics.mean(objectives) if objectives else None,
                              mean_raw_train_bce=statistics.mean(raw_bces) if raw_bces else None,
                              mean_preclip_norm=statistics.mean(norms) if norms else None,
                              fit_seconds=seconds))
            objectives, raw_bces, norms = [], [], []
            if update == UPDATES:
                break
            model.train()
        start = time.perf_counter()
        batch = dev_frame(regime, 5, pair, TRAIN_BATCH, update)
        lr = ex.learning_rate(CONFIG, update, UPDATES)
        opt.param_groups[0]["lr"] = lr
        opt.zero_grad(set_to_none=True)
        obj, raw_bce = training.objective(batch, model, cap=CAP if loss == "capped" else None)
        if not torch.isfinite(obj):
            raise ArithmeticError(f"Nonfinite objective for {key}, update {update + 1}")
        obj.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, CONFIG.clip, error_if_nonfinite=True)
        opt.step()
        seconds += time.perf_counter() - start
        objectives.append(obj.detach().item())
        raw_bces.append(raw_bce.detach().item())
        norms.append(norm.item())

    if probe_frame is not None:
        probes["trained"] = prior.probe(model, probe_frame, dev_seed(regime, 9, pair))
    interventions, spans, physical = {}, [], {}
    if loss == "capped" and arm in ex.PARITY_ARMS:
        fixed = prior.solver.AxialEMTransformer("physics_parity", steps=DEPTH).eval()
        fixed_row = ex.score([frame], lambda f: fixed(f.obs))
        interventions, spans = prior.suppressions(regime, [frame], model, fixed_row, score)
        if regime == 1:
            with patch.object(model, "pa_dewarp", False):
                interventions["emit_only"] = ex.score([frame], lambda f: model(f.obs))
        if regime > 1:
            for field in score:
                if interventions["cpe"][field] != score[field]:
                    raise ArithmeticError(f"Inactive CPE changed {field} for {key}")
            if regime == 2:
                physical = r2_path_diagnostic(model, frame)
                bank = prior.solver._soft_chirp_residual
                def without_bank(*args, **kwargs):
                    return torch.zeros_like(bank(*args, **kwargs))
                with patch.object(prior.solver, "_soft_chirp_residual", without_bank):
                    interventions["soft_bank"] = ex.score([frame], lambda f: model(f.obs))
                with patch.object(model, "steps", 2):
                    interventions["third_pass"] = ex.score([frame], lambda f: model(f.obs))
            else:
                physical = path_diagnostic(model, regime, pair)
    checkpoint = dest.with_suffix(".pt")
    ex.save_weights(checkpoint, model, signature)
    return dict(trace=trace, score=score, initial_score=trace[0]["score"], probe=probes,
                suppressions=interventions, trajectory_radians=spans, physical=physical,
                parameters=ex.real_parameters(model), initial_weights_sha256=init_digest,
                checkpoint_sha256=prior.digest(checkpoint), fit_seconds=seconds,
                init_seed=dev_seed(regime, 5, pair, 900), score_seed=dev_seed(regime, 6, pair),
                first_training_seed=dev_seed(regime, 5, pair, 0),
                last_training_seed=dev_seed(regime, 5, pair, UPDATES - 1))


@lru_cache(maxsize=1)
def implementation():
    files = ("axial_em_transformer.py", "axial_operator.py", "parity.py", "training_iter3.py",
             "experiment.py", "data.py", "run_dev_iter2.py", "verify_iter2_checkpoints.py",
             "handle_ablation.py")
    parts = [(fname, prior.digest(ex.ROOT / fname)) for fname in files]
    for function in (initial_weights, fit, validation, job_keys, r2_path_diagnostic):
        parts.append((function.__name__, inspect.getsource(function)))
    config = (asdict(CONFIG), CHECKPOINTS, CAP, list(ARMS.items()), TRAIN_BATCH, EVAL_BATCH,
              PROBE_BATCH, UPDATES, DEPTH)
    return hashlib.sha256(repr((parts, config)).encode()).hexdigest()


def run_job(key):
    dest = location(key)
    loss, regime, pair, arm = key
    signature = implementation()
    if dest.is_file():
        old = ex.read(dest)
        ck = dest.with_suffix(".pt")
        if old.get("implementation") == signature and (arm == "references" or
                (ck.is_file() and old["checkpoint_sha256"] == prior.digest(ck))):
            return
    start = time.perf_counter()
    if arm == "references":
        row = prior.references(regime, pair, DEPTH)
        row["score_seed"] = dev_seed(regime, 6, pair)
    else:
        row = fit(key, signature, dest)
    elapsed = time.perf_counter() - start
    ex.save(dest, dict(loss=loss, regime=regime, pair=pair, arm=arm, depth=DEPTH,
                       implementation=signature, wall_seconds=elapsed, **row))
    result = f", {row['score']['block_errors']} failures" if arm != "references" else ""
    print(f"{loss} R{regime}/{pair} {arm}: {elapsed:.1f}s{result}", flush=True)


def restored(regime, pair, depth, arm, loss="capped"):
    if depth != DEPTH or arm == "references":
        raise ValueError("Expected a trained three-step receiver")
    dest = location((loss, regime, pair, arm))
    ck = dest.with_suffix(".pt")
    record = ex.read(dest)
    if record["implementation"] != implementation() or record["checkpoint_sha256"] != prior.digest(ck):
        raise ValueError(f"Stale development checkpoint: {ck}")
    state = torch.load(ck, map_location="cpu", weights_only=True)
    if state["implementation"] != implementation():
        raise ValueError(f"Checkpoint implementation mismatch: {ck}")
    model = ex.model_for(regime, pair, arm, depth).eval()
    model.load_state_dict(state["state"])
    return model


@torch.inference_mode()
def latency(regime, repeats, environment, *, loader=restored, jobs_dir=JOBS, source_id=implementation):
    from p6.latency_environment import interval, project_workers, snapshot

    pair = PAIRS[0]
    methods = {}
    for arm in ex.ARMS:
        model = loader(regime, pair, DEPTH, arm)
        methods[arm] = lambda f, m=model: m(f.obs)
    for arm in ex.FIXED:
        model = ex.OperatorEMTransformer(arm, steps=DEPTH).eval()
        methods[arm] = lambda f, m=model: m(f.obs)
    for name in ex.METHODS[regime]:
        methods[name] = lambda f, a=name: ex.classical(f, a)
    rows, loads = {}, {}
    for batch_size in (1, 64):
        frame = dev_frame(regime, 6, pair, batch_size, 950 + batch_size // 64)
        order = list(methods)
        rng = random.Random(dev_seed(regime, 6, pair, 959))
        wall, cpu = ({arm: [] for arm in order} for _ in range(2))
        loads[batch_size] = []
        for rep in range(repeats + 1):
            project_workers()
            rng.shuffle(order)
            before = snapshot()
            for arm in order:
                cpu0 = time.process_time_ns()
                tick = time.perf_counter_ns()
                methods[arm](frame)
                elapsed = (time.perf_counter_ns() - tick) / 1e6
                cpu_ms = (time.process_time_ns() - cpu0) / 1e6
                if rep:
                    wall[arm].append(elapsed)
                    cpu[arm].append(cpu_ms)
            loads[batch_size].append(dict(warmup=rep == 0, **interval(before, snapshot())))
        rows[batch_size] = {a: dict(median_ms=statistics.median(wall[a]), wall_ms=wall[a],
                                    process_cpu_ms=cpu[a]) for a in order}
    ex.save(jobs_dir / f"latency_r{regime}.json", dict(
        implementation=source_id(), regime=regime, pair=pair, warmups=1, repeats=repeats,
        rows=rows, load_by_round=loads, environment=environment, torch_threads=torch.get_num_threads(),
        torch_interop_threads=torch.get_num_interop_threads(), quiet_wait_seconds=0,
        scope="Receiver forward only; simulation, final LDPC decode and scoring are excluded."))
    print(f"Timed R{regime} B1/B64, {repeats} repeats", flush=True)


def curve(records):
    points = {}
    for update, group in zip(CHECKPOINTS, zip(*(r["trace"] for r in records), strict=True), strict=True):
        if any(point["update"] != update for point in group):
            raise ValueError("Training budgets differ across pairs")
        row = prior.pooled([p["score"] for p in group])
        confidence = [p["confidence"] for p in group]
        nbits = sum(c["bits"] for c in confidence)
        row.update(mean_absolute_llr=sum(c["logit_abs_sum"] for c in confidence) / nbits,
                   safe_margin_over_6=sum(c["safe_margin_over_6"] for c in confidence),
                   wrong_margin_below_minus_6=sum(c["wrong_margin_below_minus_6"] for c in confidence))
        points[update] = row
    return points


def summarize():
    signature = implementation()
    records = {}
    for key in job_keys():
        dest = location(key)
        rec = ex.read(dest)
        loss, regime, pair, arm = key
        if (rec["implementation"] != signature or rec["score_seed"] != dev_seed(regime, 6, pair)
                or (rec["loss"], rec["regime"], rec["pair"], rec["arm"]) != key):
            raise ValueError(f"Stale development receipt: {dest}")
        if arm != "references" and prior.digest(dest.with_suffix(".pt")) != rec["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint and development receipt disagree: {dest}")
        records[key] = rec

    capacities = {r["parameters"] for r in records.values() if r["arm"] != "references"}
    if len(capacities) != 1:
        raise ValueError(f"Unequal trainable real counts: {capacities}")
    pooled, curves, suppression, kernels, paths, differences = {}, {}, {}, {}, {}, {}
    for regime in SNR:
        refs = [records["capped", regime, p, "references"]["scores"] for p in PAIRS]
        pooled[regime] = {name: prior.pooled([r[name] for r in refs]) for name in refs[0]}
        for pair in PAIRS:
            jobs = [r for r in records.values() if r["regime"] == regime and r["pair"] == pair
                    and r["arm"] != "references"]
            if len({r["initial_weights_sha256"] for r in jobs}) != 1:
                raise ValueError(f"Initial weight tensors differ for R{regime}/{pair}")
        for loss, arms in ARMS.items():
            for arm in arms:
                if arm == "references":
                    continue
                jobs = [records[loss, regime, p, arm] for p in PAIRS]
                fixed = "physics_parity" if arm in ex.PARITY_ARMS else "physics"
                for job, ref in zip(jobs, refs, strict=True):
                    assert_replay(job["initial_score"], ref[fixed])
                points = curve(jobs)
                curves.setdefault(loss, {}).setdefault(regime, {})[arm] = points
                if loss != "capped":
                    continue
                pooled[regime][arm] = points[UPDATES]
                if jobs[0]["suppressions"]:
                    labels = jobs[0]["suppressions"]
                    rows = {name: prior.pooled([j["suppressions"][name] for j in jobs]) for name in labels}
                    pair_deltas = {name: {j["pair"]: (j["suppressions"][name]["block_errors"] -
                                                     j["score"]["block_errors"]) for j in jobs}
                                   for name in labels}
                    suppression.setdefault(regime, {})[arm] = dict(
                        rows=rows, removed_minus_full={k: prior.contrasts(v, rows["full"]) for k, v in rows.items()},
                        block_delta_by_pair=pair_deltas)
                if jobs[0]["physical"] or jobs[0]["trajectory_radians"]:
                    paths.setdefault(regime, {})[arm] = {j["pair"]: dict(
                        fitted_truth=j["physical"], trajectory_radians=j["trajectory_radians"]) for j in jobs}
                if jobs[0]["probe"]:
                    kernels[arm] = prior.probe_summary(jobs)
        rows = pooled[regime]
        differences[regime] = {
            "pure_minus_local": prior.contrasts(rows["pure"], rows["local"]),
            "parity_minus_local_parity": prior.contrasts(rows["parity"], rows["local_parity"]),
            "parity_minus_fixed_parity": prior.contrasts(rows["parity"], rows["physics_parity"])}

    timing = {r: ex.read(JOBS / f"latency_r{r}.json") for r in SNR}
    if any(x["implementation"] != signature for x in timing.values()):
        raise ValueError("Stale latency receipt")
    report = dict(
        implementation=signature, config=dict(
            **asdict(CONFIG), updates=UPDATES, checkpoints=CHECKPOINTS, cap=CAP,
            anchor_coefficient_on_sum_sq=.5e-4, optimizer="Adam", train_batch=TRAIN_BATCH,
            validation_batch=EVAL_BATCH, probe_batch=PROBE_BATCH, selection="fixed update 80",
            dev_pairs=PAIRS, snr_db=SNR, streams=dict(train=5, score=6, r3_probe=8, r3_permutation=9),
            loss_control="Uncapped parity arms share the same 80-update schedule, seeds, auxiliary NMSE and anchor.",
            validation_logits="Unmodified raw receiver outputs used for every metric.",
            shuffle="Eight stream-9 key permutations per layer for overall probe correlations.",
            cpe="Common-phase correction structurally zero in R2/R3. Suppression verifies exact score identity.",
            all_handles="Every capped parity-arm all-handle suppression checked against fixed + parity.",
            eval_pair_draw="One B32 simulation at batch index zero in each regime and pair."),
        trainable_real_parameters=next(iter(capacities)), initial_weights_matched_within_pair=True,
        pooled=pooled, curves=curves, differences=differences, suppression=suppression,
        paths=paths, kernels=kernels, uniform_green_support=support_control(), latency=timing,
        by_pair=list(records.values()))
    checks = ex.RESULTS / "iter3_checks" / "summary.json"
    if checks.is_file():
        receipt = ex.read(checks)
        if receipt.get("implementation") not in (None, signature):
            raise ValueError("Invariant and benchmark implementations disagree")
        report["verification"] = dict(path=checks.relative_to(ex.ROOT).as_posix(),
                                      sha256=prior.digest(checks), summary=receipt)
    dest = ex.RESULTS / "dev_iter3.json"
    ex.save(dest, report)
    print(f"Wrote {dest}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    cmds = parser.add_subparsers(dest="action", required=True)
    fit_args = cmds.add_parser("fit")
    fit_args.add_argument("--workers", type=int, default=3, choices=range(1, 9))
    fit_args.add_argument("--loss", nargs="+", choices=ARMS, default=tuple(ARMS))
    fit_args.add_argument("--arm", nargs="+", choices=ARMS["capped"], default=ARMS["capped"])
    fit_args.add_argument("--regime", type=int, nargs="+", choices=SNR, default=tuple(SNR))
    fit_args.add_argument("--pair", type=int, nargs="+", choices=PAIRS, default=PAIRS)
    timing = cmds.add_parser("latency")
    timing.add_argument("--regime", type=int, nargs="+", choices=SNR, default=tuple(SNR))
    timing.add_argument("--repeats", type=int, default=5)
    cmds.add_parser("summarize")
    args = parser.parse_args()
    ex.setup()
    if args.action == "fit":
        jobs = [key for key in job_keys() if key[0] in args.loss and key[1] in args.regime
                and key[2] in args.pair and key[3] in args.arm]
        if not jobs:
            parser.error("No permitted iteration-3 jobs match the selection")
        if args.workers == 1:
            for key in jobs:
                run_job(key)
        else:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=ex.setup) as pool:
                for _ in pool.map(run_job, jobs, chunksize=1):
                    pass
    elif args.action == "latency":
        from p6.latency_environment import benchmark_environment
        if args.repeats < 1:
            parser.error("At least one timed repeat required")
        with benchmark_environment() as environment:
            for regime in args.regime:
                latency(regime, args.repeats, environment)
    else:
        summarize()


if __name__ == "__main__":
    main()
