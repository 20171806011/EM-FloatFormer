"""Reload every checkpoint, regenerate the formal test data, and record exact comparisons."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def verify_regime(regime):
    args = [sys.executable, "-B", "-m", "p5.study", "replay", "--regime", str(regime)]
    start = time.perf_counter()
    run = subprocess.run(args, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         encoding="utf-8", errors="replace")
    expected = 10 if regime == 1 else 5
    lines = run.stdout.strip().splitlines()
    passed = run.returncode == 0 and lines == [f"Replayed R{regime} pair {i}"
                                               for i in range(1, expected+1)]
    return dict(regime=regime, command=args, exit_code=run.returncode, output=lines,
                passed=passed, seconds=time.perf_counter()-start)


def main():
    with ThreadPoolExecutor(max_workers=3) as workers:
        rows = list(workers.map(verify_regime, (1, 2, 3)))
    receipt = dict(mode="Rescoring from saved weights with freshly regenerated test observations",
                   training_repeated=False, regimes=rows, passed=all(r["passed"] for r in rows))
    out = ROOT / "p5/results/checkpoint_replay.json"
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for row in rows:
        print(f"R{row['regime']} exit={row['exit_code']}, {len(row['output'])} completion lines")
    if not receipt["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
