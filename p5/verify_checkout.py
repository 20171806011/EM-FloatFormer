"""Run the final native checks and save their outputs inside Phase 5."""
import json
import subprocess
import sys
import time
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
BASELINE = "8380084c6dece64119ab72f9c3c1c212e154fb17"


def run(command):
    start = time.perf_counter()
    p = subprocess.run(command, cwd=BASE, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, encoding="utf-8", errors="replace")
    return dict(command=command, exit=p.returncode, output=p.stdout.strip(),
                seconds=time.perf_counter()-start)


def main():
    jobs = {
        "p5_tests": [sys.executable, "-B", "-m", "unittest", "discover", "-s", "p5/tests", "-v"],
        "p4_publication": [sys.executable, "-B", "-m", "publication.verify_p4"],
        "head": ["git", "rev-parse", "HEAD"],
        "diff": ["git", "diff", "--name-only"],
        "staged_diff": ["git", "diff", "--cached", "--name-only"],
        "status": ["git", "status", "--short"],
    }
    receipt = {k: run(cmd) for k, cmd in jobs.items()}
    ok = (all(v["exit"] == 0 for v in receipt.values())
          and receipt["head"]["output"] == BASELINE
          and not receipt["diff"]["output"] and not receipt["staged_diff"]["output"]
          and receipt["status"]["output"] == "?? p5/")
    receipt["passed"] = ok
    (BASE / "p5/results/checkout_verification.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, result in receipt.items():
        if isinstance(result, dict):
            print(name, result["exit"], result["output"])
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
