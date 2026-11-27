"""Subprocess runner for parallel original Antigravity CLI (`agy-bin.exe`) tracks."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

AGY_BIN = Path(r"C:\Users\李则徐\AppData\Local\agy\bin\agy-bin.exe")
REPO_ROOT = Path(__file__).resolve().parents[3]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--track-dir", type=Path, required=True)
    ap.add_argument("--prompt-file", type=Path, required=True)
    ap.add_argument("--round", type=int, choices=(1, 2), required=True)
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    if not AGY_BIN.exists():
        raise FileNotFoundError(f"Original Antigravity CLI binary not found: {AGY_BIN}")

    track_dir = args.track_dir.resolve()
    track_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = args.prompt_file.read_text(encoding="utf-8")

    cmd = [
        str(AGY_BIN),
        "--output-format",
        "stream-json",
        "--model",
        "gemini-4-neon-xhigh",
        "--effort",
        "xhigh",
        "--dangerously-skip-permissions",
    ]
    if args.resume:
        cmd.extend(["--resume", args.resume])
    cmd.extend(["-p", prompt_text])

    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    out_jsonl = track_dir / f"round{args.round}_events.jsonl"
    err_log = track_dir / f"round{args.round}_stderr.log"

    t0 = time.perf_counter()
    with out_jsonl.open("w", encoding="utf-8", newline="\n") as f_out, err_log.open(
        "w", encoding="utf-8", newline="\n"
    ) as f_err:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=f_out,
            stderr=f_err,
            check=False,
        )
    wall = round(time.perf_counter() - t0, 2)
    out_bytes = out_jsonl.stat().st_size if out_jsonl.exists() else 0
    err_bytes = err_log.stat().st_size if err_log.exists() else 0
    print(
        f"[{track_dir.name} R{args.round}] exit={proc.returncode} wall_s={wall} "
        f"events_bytes={out_bytes} stderr_bytes={err_bytes}",
        flush=True,
    )
    return int(proc.returncode)


if __name__ == "__main__":
    sys.exit(main())
