"""Freeze the prospective design and run independent regime workers."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def fit_regime(regime):
    cmd = [sys.executable, "-B", "-m", "p5b.study", "fit", "--regime", str(regime)]
    start = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT.parent, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    print(f"R{regime} worker exit={result.returncode}\n{result.stdout}{result.stderr}", flush=True)
    return dict(regime=regime, command=cmd, exit=result.returncode, seconds=time.perf_counter()-start,
                stdout=result.stdout, stderr=result.stderr)


def main():
    sources = ["PROTOCOL.md", "operator_em_transformer.py", "parity.py", "study.py",
               "diagnostics.py", "tests/test_p5b_invariants.py", "statistics.py", "sweep.py"]
    identity = {name: digest(ROOT / name) for name in sources}
    gates = ["preflight.json", "pilot_r1.json", "pilot_r2.json"]
    preflight = json.loads((OUT / gates[0]).read_text(encoding="utf-8"))
    if not preflight["passed"] or any(identity[k] != v for k,v in preflight["sources"].items()):
        raise ValueError("Preflight and implementation differ")
    for gate in gates[1:]:
        pilot = json.loads((OUT / gate).read_text(encoding="utf-8"))
        if (pilot["pair"], pilot["train_steps"], pilot["formal_test_access"]) != (901, 64, False):
            raise ValueError(f"Pilot identity mismatch in {gate}")
    freeze = dict(utc=datetime.now(timezone.utc).isoformat(), sources=identity,
                  inputs={name: digest(OUT / name) for name in gates},
                  p5_archive_inventory=digest(OUT / "p5_before.json"), python=sys.version,
                  host=platform.platform(), processor=platform.processor(), cpu_logical=os.cpu_count())
    write(OUT / "freeze.json", freeze)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=3) as workers:
        rows = list(workers.map(fit_regime, (1, 2, 3)))
    passed = (all(row["exit"] == 0 for row in rows) and
              all(digest(ROOT / name) == sha for name, sha in identity.items()))
    write(OUT / "sweep_run.json", dict(passed=passed, seconds=time.perf_counter()-start,
                                       freeze_sha256=digest(OUT / "freeze.json"), workers=rows))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
