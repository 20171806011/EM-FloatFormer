"""Capture P5 identity once, then run the requested regressions and compare."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


BASE = Path(__file__).resolve().parents[1]
OUT = BASE / "p5b" / "results"
BASELINE = "8380084c6dece64119ab72f9c3c1c212e154fb17"
BEFORE = OUT / "p5_before.json"


def inventory():
    return {p.relative_to(BASE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((BASE / "p5").rglob("*")) if p.is_file()}


def call(args):
    start = time.perf_counter()
    result = subprocess.run(args, cwd=BASE, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return dict(command=args, exit=result.returncode, stdout=result.stdout.strip(),
                stderr=result.stderr.strip(), seconds=time.perf_counter() - start)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", action="store_true", help="Capture the read-only P5 baseline once")
    args = parser.parse_args()
    if args.before:
        if call(["git", "rev-parse", "HEAD"])["stdout"] != BASELINE:
            raise RuntimeError("Unexpected starting revision")
        OUT.mkdir(parents=True, exist_ok=True)
        with BEFORE.open("x", encoding="utf-8") as stream:
            json.dump(inventory(), stream, indent=2)
        print("Captured input archive", flush=True)
        return
    old = json.loads(BEFORE.read_text(encoding="utf-8"))
    jobs = {}
    suite_sizes = {}
    for suite in ("p5b", "p5"):
        jobs[suite] = call([sys.executable, "-B", "-m", "unittest", "discover", "-s",
                            f"{suite}/tests", "-v"])
        output = jobs[suite]["stderr"] + "\n" + jobs[suite]["stdout"]
        match = re.search(r"Ran (\d+) tests? in ", output)
        suite_sizes[suite] = int(match[1]) if match else 0
    jobs["publication"] = call([sys.executable, "-B", "-m", "publication.verify_p4"])
    for name, args in (("head", ["rev-parse", "HEAD"]), ("diff", ["diff", "--name-only"]),
                       ("staged", ["diff", "--cached", "--name-only"]),
                       ("status", ["status", "--porcelain", "-unormal"])):
        jobs[name] = call(["git", *args])
    current = inventory()
    different = sorted(k for k in old.keys() | current.keys() if old.get(k) != current.get(k))
    ok = (all(r["exit"] == 0 for r in jobs.values()) and not different and
          jobs["head"]["stdout"] == BASELINE and not jobs["diff"]["stdout"] and
          not jobs["staged"]["stdout"] and set(jobs["status"]["stdout"].splitlines()) ==
          {"?? p5/", "?? p5b/"} and suite_sizes["p5"] == 15 and suite_sizes["p5b"] > 0
          and all(jobs[k]["stderr"].endswith("OK") for k in suite_sizes))
    receipt = dict(passed=ok, p5_files_before=len(old), p5_files_after=len(current),
                   p5_changed=different, tests_run=suite_sizes, commands=jobs)
    (OUT / "checkout_verification.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print({k: (v["exit"], round(v["seconds"], 2)) for k, v in jobs.items()}, suite_sizes,
          f"P5 file differences={len(different)}, passed={ok}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
