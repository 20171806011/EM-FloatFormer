"""Calibration, paired training and independent scoring. Run from the repository root."""
import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from p4.slot_2d import _get_default_ldpc, decode_min_sum
from p5.classical import METHODS, REFERENCES, receive
from p5.unrolled_em_transformer import UnrolledEMTransformer
from p5.wavefield_sim import green_kernel, simulate


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ARMS = ("full", "local", "no_unroll", "no_gauge")
PAIRS = {1: 10, 2: 5, 3: 5}
STEPS = 192
FIT_BATCH = 4
TEST_BATCHES = 10
EVAL_BATCH = 8


def setup():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def save(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rng(regime, stream, pair=0, batch=0):
    return 100_000_000 * regime + 1_000_000 * stream + 1_000 * pair + batch


def source_hashes():
    paths = [ROOT / "PROTOCOL.md", *(ROOT / name for name in
             ("study.py", "wavefield_sim.py", "unrolled_em_transformer.py", "classical.py"))]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def payload(frame, llrs):
    return llrs[:, ~frame.obs.pilot_mask]


def counts(frame, llrs, h_hat):
    data = payload(frame, llrs)
    if data.shape != frame.bits.shape or h_hat.shape != frame.h_true.shape:
        raise ValueError("receiver output dimensions disagree with frame")
    if not (torch.isfinite(data).all() and torch.isfinite(h_hat).all()):
        raise ValueError("nonfinite receiver output")
    blocks = data.reshape(*frame.info_bits.shape[:-1], 128)
    decoded = decode_min_sum(blocks, _get_default_ldpc().h_matrix, max_iter=12)
    wrong = decoded != frame.info_bits
    return dict(bit_errors=int(((data > 0) != frame.bits).sum()), bits=frame.bits.numel(),
                info_errors=int(wrong.sum()), info_bits=wrong.numel(),
                block_errors=int(wrong.any(-1).sum()), blocks=wrong.numel() // 64,
                channel_error=float((h_hat - frame.h_true).abs().square().sum()),
                channel_energy=float(frame.h_true.abs().square().sum()))


def add(total, row):
    for key, value in row.items():
        total[key] = total.get(key, 0) + value


def classical(frame, method):
    if method == "genie_pa":
        return receive(frame.obs, method, genie_pa=frame.truth.pa)
    return receive(frame.obs, method)


@torch.no_grad()
def calibration_cell(regime, stream, snr, bursts):
    total = {}
    for i in range(bursts // EVAL_BATCH):
        frame = simulate(regime, EVAL_BATCH, snr, rng(regime, stream, batch=i))
        add(total, counts(frame, *classical(frame, REFERENCES[regime])))
    return dict(snr_db=snr, bursts=bursts, **total,
                bler=total["block_errors"] / total["blocks"])


def calibrate(regime):
    start = time.perf_counter()
    def closest(rows):
        return min(rows, key=lambda row: (abs(row["bler"] - 0.2), row["snr_db"]))["snr_db"]
    coarse = [calibration_cell(regime, 1, snr, 24) for snr in range(-8, 33, 4)]
    center = closest(coarse)
    fine = [calibration_cell(regime, 2, snr, 64) for snr in range(center - 3, center + 4)]
    snr = closest(fine)
    valid = calibration_cell(regime, 3, snr, 96)
    passed = .05 <= valid["bler"] <= .35
    freeze = dict(regime=regime, reference=REFERENCES[regime], snr_db=snr, coarse=coarse,
                  fine=fine, validation=valid, passed=passed, sources=source_hashes(),
                  seconds=time.perf_counter() - start)
    save(RESULTS / f"calibration_r{regime}.json", freeze)
    print(f"R{regime}: {snr} dB, {valid['block_errors']}/{valid['blocks']}, passed={passed}", flush=True)
    if not passed:
        raise RuntimeError("waterfall validation failed; calibration record retained")


def frozen(regime):
    entry = read(RESULTS / f"calibration_r{regime}.json")
    if not entry["passed"] or entry["sources"] != source_hashes():
        raise ValueError("missing waterfall or source identity no longer matches calibration")
    return entry["snr_db"]


def bit_loss(frame, model):
    llrs, _ = model(frame.obs)
    return F.binary_cross_entropy_with_logits(payload(frame, llrs), frame.bits.double())


def correlation(a, b):
    x = a - a.mean(-1, keepdim=True)
    y = b - b.mean(-1, keepdim=True)
    return (x * y).sum(-1) / (x.norm(dim=-1) * y.norm(dim=-1))


@torch.no_grad()
def kernel_probe(model, frame, seed):
    _, _, layers = model(frame.obs, probe=True)
    generator = torch.Generator().manual_seed(seed)
    result = []
    for coords, attn in layers:
        target = green_kernel(frame, coords)
        n = coords.shape[0]
        off = ~torch.eye(n, dtype=torch.bool)
        corr = correlation(attn[:, off], target[:, off])
        shuffled = [correlation(attn[:, :, torch.randperm(n, generator=generator)][:, off],
                                target[:, off]) for _ in range(8)]
        result.append(dict(coords=coords.tolist(), corr=corr.tolist(),
                           shuffled_corr=torch.stack(shuffled).tolist()))
    return result


def fit(regime, pair):
    start = time.perf_counter()
    snr = frozen(regime)
    dest = RESULTS / f"r{regime}" / f"pair{pair:02d}"
    dest.mkdir(exist_ok=True, parents=True)
    models, optimizers = {}, {}
    for name in ARMS:
        torch.manual_seed(rng(regime, 4, pair))
        models[name] = UnrolledEMTransformer(variant=name)
        optimizers[name] = torch.optim.Adam(models[name].parameters(), lr=.006)
    trace = {name: [] for name in ARMS}
    train_sum = dict.fromkeys(ARMS, 0.)
    valid = simulate(regime, 16, snr, rng(regime, 6, pair))
    probe = {}
    if regime == 3:
        probe_frame = simulate(regime, 8, snr, rng(regime, 8, pair))
        probe["initial"] = kernel_probe(models["full"], probe_frame, rng(regime, 9, pair))
    for step in range(STEPS + 1):
        if step % 48 == 0:
            with torch.no_grad():
                for name, model in models.items():
                    trace[name].append(dict(step=step, validation_bce=float(bit_loss(valid, model)),
                                            train_mean_bce=train_sum[name] / 48 if step else None))
                    train_sum[name] = 0.
        if step == STEPS:
            break
        frame = simulate(regime, FIT_BATCH, snr, rng(regime, 5, pair, step))
        for name, model in models.items():
            opt = optimizers[name]
            opt.zero_grad(set_to_none=True)
            loss = bit_loss(frame, model)
            if not torch.isfinite(loss):
                raise ValueError(f"nonfinite training loss in {name}/{step}")
            loss.backward()
            opt.step()
            train_sum[name] += float(loss.detach())
    if regime == 3:
        probe["trained"] = kernel_probe(models["full"], probe_frame, rng(regime, 9, pair))
    for name, model in models.items():
        torch.save(model.state_dict(), dest / f"{name}.pt")
    train_s = time.perf_counter() - start
    rows = score(regime, pair, snr, models)
    save(dest / "record.json", dict(regime=regime, pair=pair, snr_db=snr, trace=trace,
                                    probe=probe, rows=rows, train_seconds=train_s,
                                    total_seconds=time.perf_counter() - start))
    print(f"R{regime} pair {pair}: {time.perf_counter() - start:.1f}s", flush=True)


@torch.no_grad()
def score(regime, pair, snr, models):
    models = {**models, "physics": UnrolledEMTransformer(variant="physics")}
    receivers = [*models, *METHODS[regime]]
    rows = {name: {} for name in receivers}
    for i in range(TEST_BATCHES):
        frame = simulate(regime, EVAL_BATCH, snr, rng(regime, 7, pair, i))
        for name in receivers:
            output = models[name](frame.obs) if name in models else classical(frame, name)
            add(rows[name], counts(frame, *output))
    return rows


def restored(regime, pair):
    models = {name: UnrolledEMTransformer(variant=name).eval() for name in ARMS}
    for name, model in models.items():
        path = RESULTS / f"r{regime}" / f"pair{pair:02d}" / f"{name}.pt"
        model.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
    return models


def replay(regime, pair):
    snr = frozen(regime)
    result = score(regime, pair, snr, restored(regime, pair))
    record = read(RESULTS / f"r{regime}" / f"pair{pair:02d}" / "record.json")
    if result != record["rows"]:
        raise AssertionError(f"checkpoint scoring mismatch R{regime} pair {pair}")
    print(f"Replayed R{regime} pair {pair}", flush=True)


@torch.inference_mode()
def latency(regime):
    snr = frozen(regime)
    models = {**restored(regime, 1), "physics": UnrolledEMTransformer(variant="physics")}
    rows = []
    for batch in (1, 64):
        frame = simulate(regime, batch, snr, rng(regime, 10, batch=batch))
        for name in [*models, *METHODS[regime]]:
            times = []
            for repeat in range(14):
                start = time.perf_counter_ns()
                if name in models:
                    models[name](frame.obs)
                else:
                    classical(frame, name)
                if repeat >= 3:
                    times.append((time.perf_counter_ns() - start) / 1e6)
            dofs = sum(p.numel() * (2 if p.is_complex() else 1)
                       for p in models[name].parameters() if p.requires_grad) if name in models else 0
            q1, _, q3 = statistics.quantiles(times, n=4, method="inclusive")
            rows.append(dict(method=name, batch=batch, params=dofs, ms=statistics.median(times),
                             q1_ms=q1, q3_ms=q3, samples_ms=times))
    save(RESULTS / f"latency_r{regime}.json", rows)


def main(regime=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=("calibrate", "fit", "replay", "latency"))
    if regime is None:
        parser.add_argument("--regime", required=True, type=int, choices=PAIRS)
    parser.add_argument("--pair", type=int, action="append")
    args = parser.parse_args()
    regime = regime or args.regime
    setup()
    if args.action in ("calibrate", "latency"):
        (calibrate if args.action == "calibrate" else latency)(regime)
    else:
        for pair in args.pair or range(1, PAIRS[regime] + 1):
            if not 1 <= pair <= PAIRS[regime]:
                parser.error("pair outside the prospective seed list")
            (fit if args.action == "fit" else replay)(regime, pair)


if __name__ == "__main__":
    main()
