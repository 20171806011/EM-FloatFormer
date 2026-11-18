"""Stage P4-D. All official draws require a frozen protocol and a passed P4-B gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
from filelock import FileLock

from .slot_2d import (EMFloatFormer2D, JointPhysicalLoss2D, SlotSimulator2D,
                      _evaluate_slot_bler, _synchronize_slot)

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parent
PROTOCOL = PACKAGE / "P4D_PROTOCOL.md"
DEST = PACKAGE / "results" / "p4d"
FREEZE = DEST / "P4D_FREEZE.json"
HISTORICAL_COMMIT = "f878a7363254c6e34bee5f6e278135d74e1d54e0"
OFFICIAL = tuple(range(74001, 74011))
SMOKE_SEEDS = tuple(range(68321, 68331))
SOURCES = ("p4/P4D_PROTOCOL.md", "p4/confirmatory_protocol.py", "p4/slot_2d.py",
           "p4/simulator.py", "p4/em_floatformer.py", "p4/classical_baselines.py",
           "PROJECT.md", "MANIFEST_SHA256.json", "p4/p4cd_factorial_protocol.json",
           "p4/factorial.py", "p4/audit_p4c.py", "p4/ablation_probing.py", "p4/stage3_factorial_confirm.py",
           "p4/stage3_confirm_audit.py", "p4/stage3_qualification.py", "p4/requirements.txt")
CONDITIONS = (
    ("id", "equal4", 0.01, 0.0),
    ("ood_doppler", "equal4", 0.05, 0.02),
    ("ood_delay", "long12", 0.01, 0.0),
)
EM = "emfloatformer2d"
COMPARATORS = ("lmmse_2d_fixed", "idd_pa_2d_fixed")
METHODS = (EM, *COMPARATORS, "lmmse_2d", "idd_pa_2d")
COUNTS = ("bit_errors", "total_bits", "block_errors", "total_blocks",
          "uncoded_bit_errors", "total_uncoded_bits")


@dataclass(frozen=True)
class Budget:
    steps: int
    batch: int
    microbatch: int
    eval_batch: int
    eval_batches: int


FULL = Budget(600, 64, 4, 4, 8)
SMOKE = Budget(2, 4, 2, 2, 1)


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(data):
    return (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_result(path, data, *, refresh_development_smoke=False):
    """Publish complete files. Formal evidence is immutable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        if refresh_development_smoke:
            if path != PACKAGE / "results" / "p4d_smoke.json":
                raise ValueError("Only the development smoke report can be refreshed.")
            os.replace(tmp, path)
            return
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise RuntimeError(f"Refusing to replace existing {path}")
    finally:
        tmp.unlink(missing_ok=True)


def _env():
    return {"python": platform.python_version(), "pytorch": str(torch.__version__),
            "device": "cpu", "torch_threads": 2}


def _historical_digest(root=ROOT):
    """Verify the complete pre-P4 Git tree, index and legacy manifest; fingerprint checkout bytes."""
    def git_records(*args):
        out = subprocess.check_output(["git", "-C", str(root), *args])
        return out.decode("utf-8").rstrip("\0").split("\0")

    index = {}
    for item in git_records("ls-files", "--stage", "--eol", "-z"):
        stage_info, eol, name = item.split("\t", 2)
        mode, oid, stage = stage_info.split()
        index[name] = (mode, oid, stage, eol.split()[0])

    legacy = {e["file"]: e["sha256"] for e in _read(root / "MANIFEST_SHA256.json")}
    raw = {}
    for item in git_records("ls-tree", "-rz", HISTORICAL_COMMIT):
        info, name = item.split("\t", 1)
        mode, kind, oid = info.split()
        if kind != "blob" or name not in index or index[name][:3] != (mode, oid, "0"):
            raise RuntimeError(f"Historical Git index mismatch: {name}")
        # The pinned tree contains only regular files. Git checks out LF text as CRLF on Windows.
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Missing or replaced historical file: {name}")
        data = path.read_bytes()
        canonical = data.replace(b"\r\n", b"\n") if index[name][3] == "i/lf" else data
        git_blob = b"blob " + str(len(canonical)).encode("ascii") + b"\0" + canonical
        if hashlib.sha1(git_blob).hexdigest() != oid:
            raise RuntimeError(f"Historical Git blob mismatch: {name}")
        raw[name] = _digest(data)
        # The README was revised between the older manifest and the pre-P4 commit.
        if name in legacy and name != "README.md" and legacy[name] not in (raw[name], _digest(canonical)):
            raise RuntimeError(f"Legacy manifest mismatch: {name}")
    if set(legacy) - raw.keys():
        raise RuntimeError("Legacy manifest includes paths outside the pre-P4 Git tree.")
    return {"commit": HISTORICAL_COMMIT, "file_count": len(raw),
            "checkout_sha256": _digest(_json_bytes(sorted(raw.items())))}


def freeze_after_gate(gate_report):
    """Seal live sources once the caller supplies the completed and accepted milestone record."""
    gate = Path(gate_report).resolve(strict=True)
    gate_name = gate.relative_to(ROOT).as_posix()
    gate_data = _read(gate)
    if gate_data.get("stage") != "P4-B" or gate_data.get("passed") is not True:
        raise ValueError("The Stage P4-B gate report must record stage=P4-B and passed=true.")
    if not all(key in gate_data for key in ("e2e", "observations", "evidence_files")):
        raise ValueError("P4-B qualification requires the original tests and measured scenario evidence.")
    from .stage3_qualification import validate_gate
    qualification = validate_gate(gate, ROOT)
    if FREEZE.exists():
        verify_protocol_freeze()
        return
    files = {**qualification, **{name: _digest((ROOT / name).read_bytes()) for name in (*SOURCES, gate_name)}}
    _write_result(FREEZE, _json_bytes({
        "stage": "P4-D", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "files": files, "environment": _env(), "p4b_gate": gate_name,
        "historical_checkout": _historical_digest(),
    }))


def verify_protocol_file(path):
    file = Path(path).resolve(strict=True)
    key = file.relative_to(ROOT).as_posix()
    expected = _read(FREEZE)["files"].get(key)
    if expected is None or _digest(file.read_bytes()) != expected:
        raise ValueError(f"File missing from or inconsistent with P4-D freeze: {file}")
    return True


def verify_protocol_freeze():
    freeze = _read(FREEZE)
    gate = _read(ROOT / freeze["p4b_gate"])
    required = {*SOURCES, freeze["p4b_gate"], *gate["evidence_files"]}
    if set(freeze["files"]) != required or freeze["environment"] != _env():
        raise ValueError("Protocol dependencies or runtime differ from the frozen version.")
    for name in sorted(required):
        verify_protocol_file(ROOT / name)
    from .stage3_qualification import validate_gate
    qualification = validate_gate(ROOT / freeze["p4b_gate"], ROOT)
    if any(freeze["files"].get(name) != digest for name, digest in qualification.items()):
        raise ValueError("The frozen evidence no longer validates the measured M3 qualification.")
    if _historical_digest() != freeze["historical_checkout"]:
        raise ValueError("Historical checkout bytes differ from the frozen version.")
    return True


def init_results_dir(dir_name):
    if Path(dir_name).name != dir_name or dir_name in ("", ".", ".."):
        raise ValueError("Specify a single directory name.")
    dest = PACKAGE / "results" / dir_name
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def format_error_record(seed, bit_errors, total_bits, block_errors=0, total_blocks=0):
    fields = dict(seed=seed, bit_errors=bit_errors, total_bits=total_bits,
                  block_errors=block_errors, total_blocks=total_blocks)
    if any(type(value) is not int for value in fields.values()):
        raise TypeError("Seed and error counts must be Python integers.")
    if not (0 <= bit_errors <= total_bits and 0 <= block_errors <= total_blocks):
        raise ValueError("Invalid denominators or error counts.")
    return fields


def exact_sign_flip(error_differences):
    """Use integer contrasts, which are proportional to per-seed rates at equal denominators."""
    if len(error_differences) != 10 or any(type(d) is not int for d in error_differences):
        raise ValueError("Exactly ten integer paired differences are required.")
    observed = abs(sum(error_differences))
    extremes = sum(abs(sum(s * d for s, d in zip(signs, error_differences))) >= observed
                   for signs in itertools.product((-1, 1), repeat=10))
    return extremes / 1024


def holm(raw_p):
    if any(not 0 <= p <= 1 for p in raw_p):
        raise ValueError("p-values must lie in [0, 1].")
    adjusted = [0.0] * len(raw_p)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(raw_p)), key=raw_p.__getitem__)):
        running = max(running, min(1.0, (len(raw_p) - rank) * raw_p[index]))
        adjusted[index] = running
    return adjusted


def _weights_hash(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(_json_bytes([name, str(tensor.dtype), list(tensor.shape)]))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _save_training_state(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _train(seed, budget, frozen_hash, checkpoint=None, *, stop_after=None):
    """Accumulate gradients across equally sized microbatches, then take one optimizer step."""
    if budget.batch % budget.microbatch:
        raise ValueError("Microbatch size must divide the effective batch size.")
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1000)
    sim = SlotSimulator2D(n_symb=14, n_sc=32, cp=16, profile="equal4")
    model = EMFloatFormer2D(d_model=16, n_heads=2, depth=2)
    loss_fn = JointPhysicalLoss2D(lambda_nmse=1, lambda_pa=0.25)
    optimizer = torch.optim.Adam([
        {"params": [p for name, p in model.named_parameters() if "gamma_eff" not in name], "lr": 0.008},
        {"params": model.gamma_eff.parameters(), "lr": 0.02},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=budget.steps, eta_min=0.0008)
    step0 = 0
    if checkpoint and checkpoint.exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (state["seed"], state["freeze"], state["budget"]) != (seed, frozen_hash, vars(budget)):
            raise RuntimeError("Checkpoint identity mismatch.")
        step0 = state["step"]
        if type(step0) is not int or not 0 <= step0 <= budget.steps:
            raise RuntimeError("Invalid checkpoint step.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        generator.set_state(state["data_rng"])
        torch.set_rng_state(state["global_rng"])

    stop = budget.steps if stop_after is None else stop_after
    if not step0 <= stop <= budget.steps:
        raise ValueError("Invalid training checkpoint boundary.")
    for step in range(step0, stop):
        optimizer.zero_grad()
        for _ in range(budget.batch // budget.microbatch):
            slot = sim.generate_slot(batch_size=budget.microbatch, snr_db=(10., 26.),
                                     pa_condition=("strong", "mild", "clean")[step % 3],
                                     fd=(0., 0.01, 0.02)[(step // 3) % 3], cfo=0., generator=generator)
            slot = _synchronize_slot(slot)
            output = model(slot.yp_sparse, slot.y_grid, slot.pilot_mask, slot.noise_var)
            loss = loss_fn(output, slot)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at seed {seed}, step {step + 1}")
            (loss * (budget.microbatch / budget.batch)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        if checkpoint and ((step + 1) % 20 == 0 or step + 1 == stop):
            _save_training_state(checkpoint, {
                "seed": seed, "budget": vars(budget), "step": step + 1, "freeze": frozen_hash,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "data_rng": generator.get_state(),
                "global_rng": torch.get_rng_state(),
            })
            if budget == FULL:
                print(f"Seed {seed} saved update {step + 1}/{budget.steps}", flush=True)
    return model.eval()


def _score(seed, model, budget):
    rows = []
    for index, (condition, profile, fd, cfo) in enumerate(CONDITIONS):
        draws = [10_000_000 + 1000 * seed + 100 * index + i for i in range(budget.eval_batches)]
        for method in METHODS:
            counts = dict.fromkeys(COUNTS, 0)
            for draw in draws:
                result = _evaluate_slot_bler(snr_db=10., num_slots=budget.eval_batch, n_symb=14, n_sc=32,
                                             fd=fd, cfo=cfo, profile=profile, pa_condition="strong",
                                             method=method, model=model if method == EM else None, seed=draw)
                if result["nonfinite_count"]:
                    raise FloatingPointError(f"Nonfinite output at {seed}/{condition}/{method}/{draw}")
                for key in COUNTS:
                    if type(result[key]) is not int:
                        raise TypeError(f"{key} must be an integer count.")
                    counts[key] += result[key]
            n_blocks = budget.eval_batch * budget.eval_batches * 13
            if counts["total_blocks"] != n_blocks or counts["total_bits"] != n_blocks * 64:
                raise ValueError("Unexpected number of decoded LDPC blocks or information bits.")
            format_error_record(seed, counts["bit_errors"], counts["total_bits"],
                                counts["block_errors"], counts["total_blocks"])
            rows.append({"condition": condition, "method": method, **counts})
    return rows


def _analyze(records, budget):
    """Check complete paired design and recompute six primary comparisons from raw counts."""
    expected = {(c[0], m) for c in CONDITIONS for m in METHODS}
    per_seed = []
    denomin = budget.eval_batch * budget.eval_batches * 13
    for record in records:
        rows = {(r["condition"], r["method"]): r for r in record["rows"]}
        if len(record["rows"]) != len(expected) or set(rows) != expected:
            raise ValueError("Incomplete receiver-by-condition design.")
        for row in rows.values():
            format_error_record(record["seed"], row["bit_errors"], row["total_bits"],
                                row["block_errors"], row["total_blocks"])
            if row["total_blocks"] != denomin or row["total_bits"] != 64 * denomin:
                raise ValueError("Unequal or incorrect denominator.")
            if not row["block_errors"] <= row["bit_errors"] <= 64 * row["block_errors"]:
                raise ValueError("Decoded bit errors are inconsistent with the number of failed LDPC blocks.")
            if (type(row["uncoded_bit_errors"]) is not int or type(row["total_uncoded_bits"]) is not int or
                    row["total_uncoded_bits"] != 128 * denomin or
                    not 0 <= row["uncoded_bit_errors"] <= row["total_uncoded_bits"]):
                raise ValueError("Invalid raw-bit count or denominator.")
        per_seed.append(rows)
    if len(records) != 10 or len({r["seed"] for r in records}) != 10:
        raise ValueError("Ten independent complete seed records are required.")
    tests = []
    for condition, *_ in CONDITIONS:
        for comparator in COMPARATORS:
            deltas = [rows[condition, comparator]["block_errors"] - rows[condition, EM]["block_errors"]
                      for rows in per_seed]
            tests.append({"condition": condition, "baseline": comparator,
                          "baseline_minus_em_block_errors": deltas, "denominator_per_seed": denomin,
                          "mean_bler_reduction": sum(deltas) / (10 * denomin),
                          "exact_p_value": exact_sign_flip(deltas)})
    for test, adjusted in zip(tests, holm([row["exact_p_value"] for row in tests])):
        test["holm_p_value"] = adjusted
        test["reject_at_0_05"] = adjusted < 0.05
    em_rows = [r for record in records for r in record["rows"] if r["method"] == EM]
    return {"records": records, "primary_tests": tests, "exact_p_value": tests[0]["exact_p_value"],
            "total_bits": sum(r["total_bits"] for r in em_rows),
            "total_bit_errors": sum(r["bit_errors"] for r in em_rows),
            "total_blocks": sum(r["total_blocks"] for r in em_rows),
            "total_block_errors": sum(r["block_errors"] for r in em_rows)}


def _record(seed, budget, frozen_hash, checkpoint=None):
    model = _train(seed, budget, frozen_hash, checkpoint)
    return {"seed": seed, "freeze": frozen_hash, "weights_sha256": _weights_hash(model),
            "rows": _score(seed, model, budget)}


def _result_csv(records):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["seed", "condition", "method", *COUNTS], lineterminator="\n")
    writer.writeheader()
    for record in records:
        for row in record["rows"]:
            writer.writerow({"seed": record["seed"], **row})
    return stream.getvalue().encode("utf-8")


def run_confirmatory_protocol(seeds=OFFICIAL, *, save=False, replay=False):
    """Full fresh run, or a checkpoint-backed count replay when completing/resuming a frozen run."""
    if tuple(seeds) != OFFICIAL:
        raise ValueError("The complete ordered official seed list is required.")
    verify_protocol_freeze()
    with FileLock(DEST / "run.lock", timeout=0):
        return _run_frozen(save, replay)


def _run_frozen(save, replay):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    frozen_hash = _digest(FREEZE.read_bytes())
    records = []
    for seed in OFFICIAL:
        seed_dir = DEST / str(seed)
        path = seed_dir / "counts.json"
        ckpt = seed_dir / "training.pt"
        if replay and (not path.is_file() or not ckpt.is_file()):
            raise FileNotFoundError(f"Missing evidence for independent count replay: {seed_dir}")
        # E2E runs with save=False. It can only read the existing state and recompute.
        if not save and not ckpt.is_file():
            raise RuntimeError("Execute the frozen run first to create checkpoints for read-only replay.")
        if ckpt.exists() and not save:
            state = torch.load(ckpt, map_location="cpu", weights_only=True)
            if state["step"] != FULL.steps:
                raise RuntimeError("The frozen training run is incomplete.")
        if save and path.is_file() and not replay:
            record = _read(path)
            if not ckpt.is_file():
                raise FileNotFoundError(ckpt)
            model = _train(seed, FULL, frozen_hash, ckpt)
            if record["seed"] != seed or record["freeze"] != frozen_hash or record["weights_sha256"] != _weights_hash(model):
                raise RuntimeError(f"Recorded identity mismatch for {seed}")
        else:
            record = _record(seed, FULL, frozen_hash, ckpt)
            if path.exists() and _read(path) != record:
                raise RuntimeError(f"Exact count/checkpoint replay mismatch at seed {seed}")
            if save:
                _write_result(path, _json_bytes(record))
        records.append(record)
        if replay:
            from .stage3_confirm_audit import count_again
            model = _train(seed, FULL, frozen_hash, ckpt)
            if count_again(model, seed, FULL) != record["rows"]:
                raise RuntimeError(f"Independent physical count replay disagrees at seed {seed}.")
    result = _analyze(records, FULL)
    result.update(stage="P4-D", freeze_sha256=frozen_hash)
    verify_protocol_freeze()
    if save:
        _write_result(DEST / "raw_by_seed.csv", _result_csv(records))
        _write_result(DEST / "summary.json", _json_bytes(result))
    if replay:
        from .stage3_confirm_audit import audit_csv
        if _read(DEST / "summary.json") != result:
            raise RuntimeError("Persisted coded summary disagrees with the physical replay.")
        audit = audit_csv(DEST / "raw_by_seed.csv", records, result["primary_tests"])
        _write_result(DEST / "coded_replay.json", _json_bytes({"freeze": frozen_hash, **audit}))
    return result


def run_smoke():
    """Exercise training, paired decoding, inference and count replay on disjoint development seeds."""
    if set(SMOKE_SEEDS).intersection(OFFICIAL):
        raise RuntimeError("Smoke seed overlap.")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    records = []
    for i, seed in enumerate(SMOKE_SEEDS):
        if i == 0:
            with tempfile.TemporaryDirectory() as temp:
                ckpt = Path(temp) / "training.pt"
                record = _record(seed, SMOKE, "smoke")
                _train(seed, SMOKE, "smoke", ckpt, stop_after=1)
                if _record(seed, SMOKE, "smoke", ckpt) != record:
                    raise RuntimeError("Smoke checkpoint/paired count replay failed.")
        else:
            record = _record(seed, SMOKE, "smoke")
        records.append(record)
    result = _analyze(records, SMOKE)
    return {"stage": "development smoke only", "budget": vars(SMOKE), "seeds": SMOKE_SEEDS,
            "checkpoint_replay_exact": True, "historical_checkout": _historical_digest(),
            "source_sha256": {s: _digest((ROOT / s).read_bytes()) for s in SOURCES},
            **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze-after-gate", metavar="GATE_JSON")
    action.add_argument("--execute", action="store_true")
    action.add_argument("--replay", action="store_true")
    action.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.freeze_after_gate:
        freeze_after_gate(args.freeze_after_gate)
        print(FREEZE)
    elif args.smoke:
        result = run_smoke()
        path = PACKAGE / "results" / "p4d_smoke.json"
        _write_result(path, _json_bytes(result), refresh_development_smoke=True)
        print(f"Development-only smoke: {len(result['records'])} seeds, exact count replay, {path}")
    else:
        result = run_confirmatory_protocol(save=args.execute, replay=args.replay)
        from .stage3_factorial_confirm import run as run_factorial
        factorial_rows = run_factorial(OFFICIAL, save=args.execute, replay=args.replay)
        if args.replay:
            verify_protocol_freeze()
            files = ("summary.json", "raw_by_seed.csv", "coded_replay.json", "factorial/summary.json")
            _write_result(DEST / "joint_replay.json", _json_bytes({
                "freeze": _digest(FREEZE.read_bytes()),
                "verified_artifacts": {name: _digest((DEST/name).read_bytes()) for name in files},
                "coded_seeds": len(result["records"]),
                "coded_cells": sum(len(r["rows"]) for r in result["records"]),
                "coded_contrasts": len(result["primary_tests"]),
                "independently_retrained_factorial_cells": len(factorial_rows),
                "factorial_gain_cells": sum(len(r["gains"]) for r in factorial_rows),
                "factorial_contrasts": len(_read(DEST / "factorial/summary.json")["tests"]),
            }))
        print(json.dumps(result["primary_tests"], indent=2))


if __name__ == "__main__":
    main()

