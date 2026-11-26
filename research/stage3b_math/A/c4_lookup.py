"""Build and query the frozen s17, C4-ramp, no-pilot Q1 and Q3 probability table."""
import argparse
import hashlib
import json
from pathlib import Path

TABLE = Path(__file__).with_name("c4_s17_lookup.json")
WEIGHTS_SHA256 = "e67d9fbafb90791ab19549de7789798d818286d7f52d7ac566cc39df71e78356"


def read_pair(table, z1, z2):
    if not (0 <= z1 < 16 and 0 <= z2 < 16):
        raise ValueError("Expected two 4-bit cache indices.")
    turn1, i = divmod(z1, 4)
    turn2, j = divmod(z2, 4)
    row = table["cells"][4*i+j]
    result = {}
    for task, shift in (("q1", turn2-turn1), ("q3", turn1)):
        probs = [row[task][(k-shift) % 4] for k in range(4)]
        result[task] = {"probabilities": probs, "prediction": max(range(4), key=probs.__getitem__)}
    return result


def build(matched_dir, output):
    import numpy as np
    from check_a import learned_estimate
    weight = matched_dir / "weights" / "equivariant_b8_s17.npz"
    if hashlib.sha256(weight.read_bytes()).hexdigest() != WEIGHTS_SHA256:
        raise ValueError("Weights differ from the tested frozen checkpoint")
    rules = {}
    learned_estimate(matched_dir, 60000, 41117, 20, rules=rules)
    table = {
        "encoder": "equivariant_b8_s17", "weights_sha256": WEIGHTS_SHA256,
        "condition": "C4_cfo_ramp", "h": 2, "ell": 2, "snr_db": 2,
        "omega": 0.35, "sigma_phi": 0.10, "valid_pilots": 0,
        "integration_seed": 41117, "independent_noise_windows": 60000, "hermite_order": 20,
        "targets": {"q1": "(X2-X1) modulo 4", "q3": "received sector after retransmitting X1 with new noise"},
        "layout": "row=4*(z1%4)+(z2%4); right shift Q1 by z2//4-z1//4 and Q3 by z1//4",
        "cells": [{task: rules[task]["opt"][i,j].tolist() for task in ("q1","q3")}
                  for i in range(4) for j in range(4)]
    }
    wire = json.dumps(table, indent=2, allow_nan=False) + "\n"
    reloaded = json.loads(wire)
    error = 0.0
    for z1 in range(16):
        for z2 in range(16):
            actual = read_pair(reloaded, z1, z2)
            for task in ("q1", "q3"):
                expected = rules[task]["opt"][z1,z2]
                probs = np.asarray(actual[task]["probabilities"])
                np.testing.assert_allclose(probs, expected, rtol=0, atol=2e-12)
                np.testing.assert_allclose(probs.sum(), 1, rtol=0, atol=2e-12)
                if actual[task]["prediction"] != int(expected.argmax()):
                    raise AssertionError((task, z1, z2))
                error = max(error, float(np.max(np.abs(probs-expected))))
    output.write_text(wire, encoding="utf-8")
    print(json.dumps({"cache_pairs":256, "task_posteriors":512, "max_probability_error":error,
                      "argmax_mismatches":0}))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="cmd", required=True)
    export = commands.add_parser("build")
    export.add_argument("matched_dir", type=Path)
    export.add_argument("--table", type=Path, default=TABLE)
    read = commands.add_parser("read")
    read.add_argument("z1", type=int)
    read.add_argument("z2", type=int)
    read.add_argument("--table", type=Path, default=TABLE)
    args = cli.parse_args()
    if args.cmd == "build":
        build(args.matched_dir, args.table)
    else:
        print(json.dumps(read_pair(json.loads(args.table.read_text(encoding="utf-8")),
                                   args.z1, args.z2), indent=2))


if __name__ == "__main__":
    main()
