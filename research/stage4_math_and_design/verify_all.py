#!/usr/bin/env python3
"""Sequential verification runner and repository hygiene gate for research/stage4_math_and_design."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent.parent
MAX_FILE_BYTES = 20 * 1024 * 1024
BASE_COMMIT = "387e7bc7"
PROTECTED_TREES = [
    "research/matched_cache",
    "research/stage3_ministral_cache",
    "research/stage3b_math",
    "research/stage3c",
    "research/stage3d_qwen_signed_phase",
]

SCRIPTS = [
    ("M1", "M1/verify_m1.py", "M1/results_m1.json", "M1/THEOREM_M1.md"),
    ("M2", "M2/verify_m2.py", "M2/results_m2.json", "M2/THEOREM_M2.md"),
    ("M3", "M3/verify_m3.py", "M3/results_m3.json", "M3/THEOREM_M3.md"),
    ("M4", "M4/verify_m4.py", "M4/results_m4.json", "M4/THEOREM_M4.md"),
    ("M5", "M5/verify_m5.py", "M5/results_m5.json", "M5/THEOREM_M5.md"),
    ("M6", "M6/verify_m6.py", "M6/results_m6.json", "M6/THEOREM_M6.md"),
    ("W1", "weight_mod/W1/verify_w1.py", "weight_mod/W1/results_w1.json", "weight_mod/W1/THEOREM_W1.md"),
    ("W2", "weight_mod/W2/verify_w2.py", "weight_mod/W2/results_w2.json", "weight_mod/W2/THEOREM_W2.md"),
    ("W3", "weight_mod/W3/verify_w3.py", "weight_mod/W3/results_w3.json", "weight_mod/W3/THEOREM_W3.md"),
    ("W4", "weight_mod/W4/verify_w4.py", "weight_mod/W4/results_w4.json", "weight_mod/W4/THEOREM_W4.md"),
]

REQUIRED_DISCUSSION_FILES = [
    ("prompt_r1.md", False),
    ("round1_events.jsonl", False),
    ("round1_stderr.log", True),
    ("reply_r1.md", False),
    ("prompt_r2.md", False),
    ("round2_events.jsonl", False),
    ("round2_stderr.log", True),
    ("reply_r2.md", False),
    ("FULL_CONVERSATION.md", False),
]


def run_pinned_script(rel_script: str) -> dict:
    """Run one standalone verifier sequentially under pinned uv dependencies."""
    script_path = ROOT / rel_script
    cmd = [
        "uv",
        "run",
        "--python",
        "3.12",
        "--with",
        "numpy==2.3.5",
        "--with",
        "scipy==1.17.0",
        "--with",
        "torch",
        "python",
        "-B",
        str(script_path),
    ]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed_s = round(time.perf_counter() - t0, 3)
    out_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    err_lines = [ln for ln in proc.stderr.splitlines() if ln.strip()]
    return {
        "script": rel_script,
        "exit_code": proc.returncode,
        "elapsed_s": elapsed_s,
        "stdout_tail": out_lines[-10:],
        "stderr_tail": err_lines[-5:],
    }


def check_utf8_file(path: pathlib.Path, allow_empty: bool) -> dict:
    """Verify file existence, non-emptiness unless allowed, size < 20 MB, and valid UTF-8."""
    if not path.is_file():
        return {"exists": False, "ok": False, "error": "missing"}
    raw = path.read_bytes()
    size = len(raw)
    if not allow_empty and size == 0:
        return {"exists": True, "size_bytes": size, "ok": False, "error": "empty"}
    if size >= MAX_FILE_BYTES:
        return {"exists": True, "size_bytes": size, "ok": False, "error": "exceeds_20mb"}
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        return {"exists": True, "size_bytes": size, "ok": False, "error": f"utf8_error: {exc}"}
    return {"exists": True, "size_bytes": size, "ok": True}


def main() -> int:
    t_all = time.perf_counter()
    script_results = []
    scripts_passed = True

    for idx, (track_id, rel_script, rel_json, rel_theorem) in enumerate(SCRIPTS, start=1):
        print(f"[verify_all] ({idx}/{len(SCRIPTS)}) Running {track_id}: {rel_script} ...", flush=True)
        res = run_pinned_script(rel_script)
        res["track_id"] = track_id

        json_path = ROOT / rel_json
        json_ok = False
        json_bytes = 0
        if json_path.is_file():
            try:
                raw_json = json_path.read_bytes()
                json_bytes = len(raw_json)
                json.loads(raw_json.decode("utf-8"))
                json_ok = True
            except Exception as exc:
                res["json_error"] = str(exc)
        else:
            res["json_error"] = "missing"

        thm_check = check_utf8_file(ROOT / rel_theorem, allow_empty=False)
        res["results_json"] = rel_json
        res["results_json_ok"] = json_ok
        res["results_json_bytes"] = json_bytes
        res["theorem_md"] = rel_theorem
        res["theorem_md_ok"] = thm_check["ok"]
        res["theorem_md_bytes"] = thm_check.get("size_bytes", 0)

        passed = (res["exit_code"] == 0) and json_ok and thm_check["ok"]
        res["passed"] = passed
        if not passed:
            scripts_passed = False

        print(
            f"[verify_all] -> {track_id} exit={res['exit_code']} elapsed={res['elapsed_s']}s "
            f"json_ok={json_ok} thm_ok={thm_check['ok']}",
            flush=True,
        )
        script_results.append(res)

    discussions_report = {}
    discussions_passed = True
    for track_id, _, _, _ in SCRIPTS:
        track_dir = ROOT / "discussions" / track_id
        file_checks = {}
        track_ok = True
        for fname, allow_empty in REQUIRED_DISCUSSION_FILES:
            chk = check_utf8_file(track_dir / fname, allow_empty=allow_empty)
            file_checks[fname] = chk
            if not chk["ok"]:
                track_ok = False
        if not track_ok:
            discussions_passed = False
        discussions_report[track_id] = {"ok": track_ok, "files": file_checks}

    bad_dirs = []
    oversized_files = []
    utf8_failures = []
    file_count = 0
    max_file_bytes = 0
    for dirpath, dirnames, filenames in os.walk(ROOT):
        for d in dirnames:
            if d == "__pycache__" or d.startswith(".uv-"):
                bad_dirs.append(os.path.relpath(os.path.join(dirpath, d), ROOT).replace("\\", "/"))
        for f in filenames:
            fp = pathlib.Path(dirpath) / f
            rel_fp = str(fp.relative_to(ROOT)).replace("\\", "/")
            file_count += 1
            sz = fp.stat().st_size
            if sz > max_file_bytes:
                max_file_bytes = sz
            if sz >= MAX_FILE_BYTES:
                oversized_files.append({"path": rel_fp, "bytes": sz})
            if fp.suffix.lower() in {".py", ".md", ".json", ".jsonl", ".log", ".sha256"}:
                try:
                    fp.read_bytes().decode("utf-8")
                except UnicodeDecodeError as exc:
                    utf8_failures.append({"path": rel_fp, "error": str(exc)})

    git_proc = subprocess.run(
        ["git", "diff", "--name-only", BASE_COMMIT, "--", *PROTECTED_TREES],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    protected_diff = [ln.strip() for ln in git_proc.stdout.splitlines() if ln.strip()]
    protected_trees_untouched = (git_proc.returncode == 0) and (len(protected_diff) == 0)

    hygiene_passed = (
        (len(bad_dirs) == 0)
        and (len(oversized_files) == 0)
        and (len(utf8_failures) == 0)
        and protected_trees_untouched
    )
    all_passed = scripts_passed and discussions_passed and hygiene_passed

    summary = {
        "all_passed": all_passed,
        "scripts_passed": scripts_passed,
        "discussions_passed": discussions_passed,
        "hygiene_passed": hygiene_passed,
        "total_elapsed_s": round(time.perf_counter() - t_all, 3),
        "scripts": script_results,
        "discussions": discussions_report,
        "hygiene": {
            "file_count_under_stage4_math_and_design": file_count,
            "max_file_bytes": max_file_bytes,
            "max_file_limit_bytes": MAX_FILE_BYTES,
            "temp_cache_dirs": bad_dirs,
            "oversized_files": oversized_files,
            "utf8_failures": utf8_failures,
            "base_commit": BASE_COMMIT,
            "protected_trees": PROTECTED_TREES,
            "protected_diff": protected_diff,
            "protected_trees_untouched": protected_trees_untouched,
        },
    }

    summary_path = ROOT / "verify_all_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"[verify_all] COMPLETE: all_passed={all_passed} scripts_passed={scripts_passed} "
        f"discussions_passed={discussions_passed} hygiene_passed={hygiene_passed} "
        f"total_elapsed={summary['total_elapsed_s']}s -> {summary_path}",
        flush=True,
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
