"""Frozen one-dimensional mechanism study, paired within seed across eight architectures."""
import json
import hashlib
from pathlib import Path

import torch
from filelock import FileLock

from . import audit_p4c, factorial
from . import confirmatory_protocol as formal
from .em_floatformer import _train_comparative_model


def _fit_and_score(seed, arm, independent=False):
    spec = factorial.ARMS[arm - 1]
    settings = factorial.PROTOCOL["train"]
    if independent:
        model, _ = audit_p4c.fit_again(seed, spec)
    else:
        model = audit_p4c.build(seed, spec)
        model, _ = _train_comparative_model(model, steps=settings["steps"], batch_size=settings["batch_size"],
                n_sc=32, profile=settings["profile"], seed=seed, backbone_lr=settings["backbone_lr"],
                gamma_lr=settings["gamma_lr"])
    evaluation = factorial.PROTOCOL["evaluation"]
    frame = factorial.observation(batch_size=evaluation["batch_size"],
                                  seed=seed + evaluation["data_seed_offset"],
                                  snr_db=evaluation["snr_db"], profile=evaluation["profile"])
    row = factorial.evaluate_arm(model, frame, seed, arm)
    row["weights_sha256"] = formal._weights_hash(model)
    if independent:
        with torch.inference_mode():
            for cell in row["gains"]:
                gain = cell["scale"]
                out = model(frame.yp * gain, frame.yd * gain, frame.noise_var * gain**2)
                count = audit_p4c.measure(out.llrs, frame.bits)
                if (count["bit_errors"], count["total_bits"]) != (cell["bit_errors"], cell["total_bits"]):
                    raise RuntimeError("Independent integer BER counter mismatch.")
    return row


def _check(rows, seeds):
    expected = [(seed, arm) for seed in seeds for arm in range(1, 9)]
    if [(r["seed"], r["arm_id"]) for r in rows] != expected:
        raise ValueError("Incomplete or out-of-order factorial cells.")
    bits = factorial.PROTOCOL["evaluation"]["batch_size"] * 32 * 4
    for row in rows:
        spec = factorial.ARMS[row["arm_id"]-1]
        if [row[k] for k in ("gauge", "hermitian", "joint", "width")] != spec:
            raise ValueError("Mismatched factorial architecture.")
        if [c["scale"] for c in row["gains"]] != factorial.GAINS:
            raise ValueError("Incomplete gain sweep.")
        for cell in row["gains"]:
            n = cell["bit_errors"]
            if (type(n) is not int or type(cell["total_bits"]) is not int or not 0 <= n <= bits
                    or cell["total_bits"] != bits or cell["ber"] != n / bits):
                raise ValueError("Invalid gain-conditioned integer BER record.")
        errs = sum(cell["bit_errors"] for cell in row["gains"])
        if (type(row["bit_errors"]) is not int or type(row["total_bits"]) is not int or
                (row["bit_errors"], row["total_bits"], row["ber"]) != (errs, 3*bits, errs/(3*bits))):
            raise ValueError("Pooled BER or denominator is inconsistent.")


def analyze(rows, seeds):
    _check(rows, seeds)
    stats = factorial.summary(rows)
    tests = []
    for factor in ("gauge", "hermitian", "joint"):
        deltas = [sum((-1 if r[factor] else 1) * r["bit_errors"] for r in rows if r["seed"] == seed)
                  for seed in seeds]
        denominator = 4 * rows[0]["total_bits"]
        tests.append({"factor": factor, "off_minus_on_integer_contrasts": deltas,
                      "per_seed_denominator": denominator, "mean_ber_reduction": sum(deltas)/(len(seeds)*denominator),
                      "raw_p": formal.exact_sign_flip(deltas)})
    for test, adjusted in zip(tests, formal.holm([t["raw_p"] for t in tests])):
        test["holm_p"] = adjusted
    return {"seed_count": len(seeds), "arm_count": len(rows), "tests": tests, **stats}


def run(seeds, *, save=False, replay=False):
    seeds = tuple(seeds)
    if set(seeds).isdisjoint(formal.OFFICIAL):
        return factorial.run_rows(seeds)
    if seeds != formal.OFFICIAL:
        raise ValueError("Formal factorial requires the ten prespecified seeds in order.")
    formal.verify_protocol_freeze()
    freeze = hashlib.sha256(formal.FREEZE.read_bytes()).hexdigest()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    root = formal.DEST / "factorial"
    root.mkdir(exist_ok=True)
    rows = []
    with FileLock(root / "run.lock", timeout=0):
        for seed in seeds:
            for arm in range(1, 9):
                path = root / str(seed) / f"arm_{arm}.json"
                if not path.is_file() and not save:
                    raise FileNotFoundError(f"Run the frozen factorial first: {path}")
                if path.is_file():
                    result = json.loads(path.read_bytes())
                    if result["freeze"] != freeze:
                        raise ValueError("Factorial result belongs to a different protocol.")
                    row = result["row"]
                    if replay and _fit_and_score(seed, arm, independent=True) != row:
                        raise RuntimeError(f"Independent factorial replay differs for {seed}/{arm}.")
                else:
                    row = _fit_and_score(seed, arm)
                    formal._write_result(path, formal._json_bytes({"freeze": freeze, "row": row}))
                rows.append(row)
        summary = {"freeze": freeze, **analyze(rows, seeds)}
        path = root / "summary.json"
        if save:
            formal._write_result(path, formal._json_bytes(summary))
        elif json.loads(path.read_bytes()) != summary:
            raise RuntimeError("Frozen factorial summary does not reproduce.")
        formal.verify_protocol_freeze()
    return rows
