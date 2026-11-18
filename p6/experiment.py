"""Single-thread development fits, identical validation batches, and R3 probes."""

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
from torch.nn import functional as F

from p5.classical import METHODS
from p5.study import add, classical, counts, payload
from p5.wavefield_sim import green_kernel
from p5b.operator_em_transformer import OperatorEMTransformer
from p6.data import PAIRS, SNR, dev_frame, dev_seed


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ARMS = ("pure", "local", "parity", "local_parity", "no_gauge")
PARITY_ARMS = ("parity", "local_parity")
FIXED = ("physics", "physics_parity")
FIT_BATCH = 4
EVAL_BATCH = 32
VAL_BATCHES = 1
CURVE_BATCH = 8


@dataclass(frozen=True)
class Design:
    depth: int = 2
    lr: float = .003
    schedule: str = "constant"
    clip: float | None = 1.
    aux_nmse: float = 0.


MAIN = Design()
DESIGNS = {
    "main": MAIN,
    "depth3": replace(MAIN, depth=3),
    "lr005": replace(MAIN, lr=.005),
    "warmcos": replace(MAIN, schedule="warmcos"),
    "aux": replace(MAIN, aux_nmse=.05),
    "unclipped": replace(MAIN, clip=None),
}


def setup():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def implementation_id(design):
    files = ["experiment.py", "data.py"]
    if design != "references":
        files.extend(("axial_operator.py", "axial_em_transformer.py", "parity.py"))
    digest = hashlib.sha256()
    for name in files:
        digest.update(name.encode())
        digest.update((ROOT / name).read_bytes())
    return digest.hexdigest()


def job_dir(regime, pair, design):
    return RESULTS / "jobs" / f"r{regime}" / str(pair) / design


def arms_for(regime, design):
    if design == "references":
        return ("references",)
    return ARMS if design == "main" else PARITY_ARMS


def checkpoints(design, arm):
    if design == "main" and arm in PARITY_ARMS:
        return (64, 96, 128)
    return (64,)


def model_for(regime, pair, arm, depth):
    from p6.axial_em_transformer import AxialEMTransformer
    torch.manual_seed(dev_seed(regime, 5, pair, 900))
    return AxialEMTransformer(variant=arm, steps=depth)


def real_parameters(model):
    return sum(p.numel() * (2 if p.is_complex() else 1)
               for p in model.parameters() if p.requires_grad)


def valid_frames(regime, pair):
    return [dev_frame(regime, 6, pair, EVAL_BATCH, i) for i in range(VAL_BATCHES)]


@torch.no_grad()
def score(frames, receiver):
    total = {}
    for frame in frames:
        llr, channel = receiver(frame)
        row = counts(frame, llr, channel)
        row["bce_sum"] = float(F.binary_cross_entropy_with_logits(
            payload(frame, llr), frame.bits.double(), reduction="sum"))
        add(total, row)
    return total


@torch.no_grad()
def reference_job(regime, pair):
    frames = valid_frames(regime, pair)
    fixed = {name: OperatorEMTransformer(name, steps=2).eval() for name in FIXED}
    rows = {name: score(frames, lambda frame: model(frame.obs)) for name, model in fixed.items()}
    for name in METHODS[regime]:
        rows[name] = score(frames, lambda frame: classical(frame, name))
    return dict(rows=rows, score_seeds=[dev_seed(regime, 6, pair, i) for i in range(VAL_BATCHES)])


def correlation(x, y):
    x = x - x.mean(-1, keepdim=True)
    y = y - y.mean(-1, keepdim=True)
    norm = x.norm(dim=-1) * y.norm(dim=-1)
    # Constant maps have undefined Pearson correlation.
    return torch.where(norm > 0, (x * y).sum(-1) / norm, float("nan"))


def nullable(values):
    return [float(v) if torch.isfinite(v) else None for v in values]


@torch.no_grad()
def probe_kernels(model, frame, shuffle_seed):
    _, _, layers = model(frame.obs, probe="axes")
    gen = torch.Generator().manual_seed(shuffle_seed)
    records = []
    for coords, joint, axes in layers:
        target = green_kernel(frame, coords)
        n = len(coords)
        off = ~torch.eye(n, dtype=torch.bool)
        perms = [torch.randperm(n, generator=gen) for _ in range(8)]
        maps = {}
        for axis, kernel in {"joint": joint, **axes}.items():
            if kernel.shape != target.shape:
                raise ValueError(f"{axis} probe and Green target disagree")
            raw = correlation(kernel[:, off], target[:, off])
            shuffled = torch.stack([correlation(kernel[:, :, p][:, off], target[:, off])
                                    for p in perms])
            entry = dict(overall=nullable(raw), shuffled=nullable(torch.nanmean(shuffled, dim=0)))
            if axis != "joint":
                dim = {"spatial": 0, "time": 1, "frequency": 2}[axis]
                others = [i for i in range(3) if i != dim]
                same = (coords[:, None, others] == coords[None, :, others]).all(-1) & off
                entry["axis_only"] = nullable(correlation(kernel[:, same], target[:, same]))
                entry["axis_only_entries"] = int(same.sum())
            maps[axis] = entry
        records.append(dict(coords=coords.tolist(), off_diagonal_entries=int(off.sum()), maps=maps))
    return records


def learning_rate(config, index, length):
    if config.schedule == "constant":
        return config.lr
    warmup = 4
    if index < warmup:
        return config.lr * (index + 1) / warmup
    phase = (index - warmup) / (length - 1 - warmup)
    return config.lr * (.1 + .9 * (1. + math.cos(math.pi * phase)) / 2.)


def objective(frame, model, aux_nmse):
    llr, channel = model(frame.obs)
    bce = F.binary_cross_entropy_with_logits(payload(frame, llr), frame.bits.double())
    if aux_nmse:
        nmse = (channel - frame.h_true).abs().square().sum() / frame.h_true.abs().square().sum()
        return bce + aux_nmse * nmse, bce
    return bce, bce


def save_weights(path, model, signature):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save({"implementation": signature, "state": model.state_dict()}, tmp)
    os.replace(tmp, path)


def train(regime, pair, design, arm, signature):
    config = DESIGNS[design]
    budgets = checkpoints(design, arm)
    model = model_for(regime, pair, arm, config.depth)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=config.lr)
    frames = valid_frames(regime, pair)
    # Same stream-6 seed, separate B8 simulator call. Drawing B32 changes
    # the RNG consumption for later physical variables within a burst.
    curve_frame = dev_frame(regime, 6, pair, CURVE_BATCH, 0)
    probe_frame = None
    probes = {}
    if regime == 3 and design == "main" and arm in ("pure", "parity", "local", "local_parity"):
        probe_frame = dev_frame(3, 8, pair, 8)
        probes["initial"] = probe_kernels(model, probe_frame, dev_seed(3, 9, pair))
    trace, rows, grad_norms = [], {}, []
    train_bce, train_obj, fit_seconds = 0., 0., 0.
    for step in range(max(budgets) + 1):
        if step % 16 == 0:
            with torch.no_grad():
                dev_bce = float(objective(curve_frame, model, 0.)[1])
            trace.append(dict(step=step, validation_bce=dev_bce,
                              train_mean_bce=train_bce / 16 if step else None,
                              train_mean_objective=train_obj / 16 if step else None,
                              mean_preclip_norm=statistics.mean(grad_norms) if step else None,
                              max_preclip_norm=max(grad_norms) if step else None,
                              lr=learning_rate(config, max(step-1, 0), max(budgets)),
                              cumulative_fit_seconds=fit_seconds))
            train_bce, train_obj, grad_norms = 0., 0., []
        if step in budgets:
            model.eval()
            rows[str(step)] = score(frames, lambda frame: model(frame.obs))
            if probe_frame is not None and step == 64:
                probes["trained"] = probe_kernels(model, probe_frame, dev_seed(3, 9, pair))
            if step == 64 and pair == PAIRS[0] and design in ("main", "depth3"):
                save_weights(job_dir(regime, pair, design) / f"{arm}.pt", model, signature)
            model.train()
        if step == max(budgets):
            break
        start = time.perf_counter()
        frame = dev_frame(regime, 5, pair, FIT_BATCH, step)
        optimizer.param_groups[0]["lr"] = learning_rate(config, step, max(budgets))
        optimizer.zero_grad(set_to_none=True)
        loss, bce = objective(frame, model, config.aux_nmse)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, config.clip or float("inf"),
                                              error_if_nonfinite=True)
        if not torch.isfinite(loss):
            raise ArithmeticError(f"Nonfinite loss at R{regime}/{pair}/{design}/{arm}/{step}")
        optimizer.step()
        grad_norms.append(float(norm))
        train_bce += float(bce.detach())
        train_obj += float(loss.detach())
        fit_seconds += time.perf_counter() - start
    return dict(config=asdict(config), parameters=real_parameters(model), trace=trace, rows=rows,
                probe=probes, fit_seconds=fit_seconds, train_batch=FIT_BATCH,
                scoring_batch_size=EVAL_BATCH, validation_curve_batch_size=CURVE_BATCH,
                validation_curve_seed=dev_seed(regime, 6, pair, 0),
                init_seed=dev_seed(regime, 5, pair, 900),
                train_seed_first=dev_seed(regime, 5, pair, 0),
                train_seed_last=dev_seed(regime, 5, pair, max(budgets)-1),
                score_seeds=[dev_seed(regime, 6, pair, i) for i in range(VAL_BATCHES)])


def jobs():
    # Each tuple trains a single arm. Controls always share seeds and schedules.
    for design in ("references", *DESIGNS):
        for regime in SNR:
            for pair in PAIRS if design in ("references", "main", "depth3") else PAIRS[:1]:
                for arm in arms_for(regime, design):
                    yield regime, pair, design, arm


def run_job(job):
    regime, pair, design, arm = job
    if job not in set(jobs()):
        raise ValueError(f"Job outside the Iteration-1 development design: {job}")
    start = time.perf_counter()
    signature = implementation_id(design)
    path = job_dir(regime, pair, design) / f"{arm}.json"
    if path.exists() and read(path)["implementation"] == signature:
        return False
    data = reference_job(regime, pair) if design == "references" else train(
        regime, pair, design, arm, signature)
    seconds = time.perf_counter() - start
    save(path, dict(regime=regime, pair=pair, design=design, arm=arm, snr_db=SNR[regime],
                    implementation=signature, wall_seconds=seconds, **data))
    print(f"R{regime} {pair} {design}/{arm} {seconds:.1f}s", flush=True)
    return True


@torch.inference_mode()
def latency(regime, repeats=5):
    from random import Random
    pair = PAIRS[0]
    signature = implementation_id("main")
    dest = RESULTS / "jobs" / f"latency_r{regime}.json"
    if dest.exists() and read(dest)["implementation"] == signature:
        return
    methods = {}
    for design, arms in (("main", ARMS), ("depth3", PARITY_ARMS)):
        for arm in arms:
            model = model_for(regime, pair, arm, DESIGNS[design].depth).eval()
            saved = torch.load(job_dir(regime, pair, design) / f"{arm}.pt", weights_only=True)
            if saved["implementation"] != signature:
                raise ValueError("Latency checkpoint signature mismatch")
            model.load_state_dict(saved["state"])
            methods[f"{design}/{arm}"] = lambda frame, m=model: m(frame.obs)
    for arm in FIXED:
        model = OperatorEMTransformer(arm, steps=2).eval()
        methods[arm] = lambda frame, m=model: m(frame.obs)
    for name in METHODS[regime]:
        methods[name] = lambda frame, method=name: classical(frame, method)
    rows = {}
    for batch_size in (1, 64):
        frame = dev_frame(regime, 6, pair, batch_size, 950 + batch_size // 64)
        samples = {name: [] for name in methods}
        randomizer = Random(dev_seed(regime, 6, pair, 959))
        for i in range(repeats + 1):
            order = list(methods)
            randomizer.shuffle(order)
            for name in order:
                start = time.perf_counter_ns()
                methods[name](frame)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                if i:
                    samples[name].append(elapsed)
        rows[str(batch_size)] = {k: dict(median_ms=statistics.median(v), samples_ms=v)
                                 for k, v in samples.items()}
    save(dest, dict(regime=regime, pair=pair, implementation=signature, rows=rows))
    print(f"R{regime} isolated CPU latency complete", flush=True)
