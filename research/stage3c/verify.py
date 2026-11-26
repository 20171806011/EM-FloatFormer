"""Replay all seven pre-host checks against their recorded outputs."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def compare(expected, got, path):
    if isinstance(expected, dict):
        assert isinstance(got, dict) and expected.keys() == got.keys(), path
        for key, value in expected.items():
            compare(value, got[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(got, list) and len(expected) == len(got), path
        for i, value in enumerate(expected):
            compare(value, got[i], f"{path}[{i}]")
    elif isinstance(expected, float):
        assert math.isclose(expected, got, rel_tol=1e-8, abs_tol=1e-10), (path, expected, got)
    else:
        assert expected == got, (path, expected, got)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path, help="checkout with the pinned Stage 3 inputs")
    args = parser.parse_args()
    repo = args.repo.resolve()
    studies = Path(__file__).resolve().parent
    table = repo / "research/stage3b_math/A/c4_s17_lookup.json"
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    with tempfile.TemporaryDirectory() as tmp:
        kernel = Path(tmp) / "kernel.json"
        runs = [
            ("A/check_q2.py", ["--repo", repo, "--export-kernel", kernel], "A/q2_results.json"),
            ("B/check_b.py", [], "B/results.json"),
            ("C/check.py", [], "C/results.json"),
            ("D/check_d.py", ["--repo", repo], "D/results.json"),
            ("E/design.py", [], "E/results.json"),
            ("F/check_f.py", ["--repo", repo], "F/results.json"),
            ("G/check_g.py", [table], "G/results.json"),
        ]
        for script, arguments, output in runs:
            actual = subprocess.check_output([sys.executable, "-B", studies/script, *arguments], env=env)
            compare(json.loads((studies/output).read_text()), json.loads(actual), output)
            print(f"Verified {output}", flush=True)
        compare(json.loads((studies/"A/q2_s17_kernel.json").read_text()), json.loads(kernel.read_text()), "A/kernel")
        print("Verified exported Q2 numerical kernel", flush=True)


if __name__ == "__main__":
    main()
