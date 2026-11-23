"""Replay the five work packages without changing published results."""
import csv
import json
import os
import math
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def main():
    for folder, script, reference in (
        ("ideal", "verify_ideal.py", "ideal_results.json"),
        ("discrete", "verify.py", "risks.csv"),
        ("moving_pair", "verify.py", "results.csv"),
        ("audit", "check.py", "results.json"),
    ):
        path = ROOT / folder
        output = subprocess.check_output([PYTHON, script], cwd=path, text=True, encoding="utf-8")
        if output != (path / reference).read_text(encoding="utf-8"):
            raise RuntimeError(f"{folder}/{reference} differs from replay")
        print(f"{folder}: exact replay", flush=True)
    witness = subprocess.check_output([PYTHON, "verify_continuous_converse.py"], cwd=ROOT / "ideal")
    actual = json.loads(witness)
    reference = json.loads((ROOT / "ideal/continuous_converse_results.json").read_text())
    if actual.keys() != reference.keys() or any(
        not math.isclose(actual[key], reference[key], rel_tol=1e-9, abs_tol=1e-14)
        for key in reference
    ):
        raise RuntimeError("continuous converse witness differs")
    print("continuous converse: numerical witness replay", flush=True)
    for directory, script, id_columns in (
        ("continuous", "run.py", 3), ("refinement", "run_refinement.py", 6)
    ):
        replay_csv(directory, script, id_columns)


def replay_csv(directory, script, id_columns):
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    with TemporaryDirectory() as tmp:
        out = Path(tmp) / "result.csv"
        subprocess.run([PYTHON, str(ROOT / directory / script), "--output", str(out)],
                       check=True, env=env)
        with out.open() as actual, (ROOT / directory / "results.csv").open() as expected:
            a, b = list(csv.reader(actual)), list(csv.reader(expected))
        if len(a) != len(b) or a[0] != b[0]:
            raise RuntimeError(f"{directory} result schema or row count differs")
        for x, y in zip(a[1:], b[1:]):
            if len(x) != len(y) or x[:id_columns] != y[:id_columns]:
                raise RuntimeError(f"{directory} settings differ")
            if any(not math.isclose(float(i), float(j), abs_tol=1e-9, rel_tol=0)
                   for i, j in zip(x[id_columns:], y[id_columns:])):
                raise RuntimeError(f"{directory} scores differ for {x[:id_columns]}")
        print(f"{directory}: numeric replay", flush=True)


if __name__ == "__main__":
    main()
