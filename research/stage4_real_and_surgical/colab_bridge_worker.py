"""Colab-side pull-worker daemon for the Google Drive bidirectional CLI bridge.

Runs inside a connected Google Colab runtime (notebook cell or Colab terminal),
polls `<work-root>/bridge/inbox/*.json`, dispatches strictly allowlisted actions
(never `shell=True`), writes periodic atomic `heartbeat.json` telemetry, and
streams stdout/stderr logs + atomic result JSONs back to `<work-root>/bridge/outbox/`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

_FILE_P = Path(__file__).resolve()
_REPO_ROOT = _FILE_P.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


ALLOWED_ACTIONS = (
    "status",
    "git_sync",
    "run_stage",
    "run_python",
    "tail_file",
    "stop_worker",
)

ALLOWED_STAGES = (
    "download_artifacts",
    "stage6_4b",
    "stage6_9b",
    "stage6_27b",
    "all",
)

ALLOWED_RUN_U1_MODES = ("auto", "true", "false")
ALLOWED_QUANT_MODES = ("bf16", "mixed_nf4")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sanitize_job_id(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(raw).strip())
    cleaned = cleaned.strip("._-")
    if not cleaned:
        raise ValueError(f"Invalid empty job_id from {raw!r}")
    return cleaned[:96]


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    """Atomically write JSON via sibling tempfile + `os.replace`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    data = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            fh.write(data)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def resolve_within(root: Path, rel_path: str) -> Path:
    """Resolve `rel_path` strictly inside `root`, rejecting traversal and absolute escapes."""
    raw = str(rel_path or "").strip()
    if not raw or "\x00" in raw:
        raise ValueError(f"Invalid relative path: {rel_path!r}")
    cand_in = Path(raw)
    if cand_in.is_absolute():
        raise ValueError(f"Absolute paths are forbidden: {rel_path!r}")
    root_res = root.resolve()
    target_res = (root_res / cand_in).resolve()
    if not target_res.is_relative_to(root_res):
        raise ValueError(f"Path traversal rejected: {rel_path!r} escapes {root_res}")
    return target_res


def query_gpu_stats() -> Dict[str, Any]:
    """Query GPU name, VRAM usage, and utilization with graceful CPU fallback."""
    if shutil.which("nvidia-smi") is None:
        return {
            "gpu_name": "CPU",
            "vram_used_mib": 0.0,
            "vram_total_mib": 0.0,
            "gpu_util_pct": 0.0,
        }
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            raise RuntimeError(proc.stderr.strip() or "empty nvidia-smi output")
        first_line = proc.stdout.strip().splitlines()[0]
        parts = [p.strip() for p in first_line.split(",")]
        return {
            "gpu_name": parts[0],
            "vram_used_mib": float(parts[1]),
            "vram_total_mib": float(parts[2]),
            "gpu_util_pct": float(parts[3]),
        }
    except Exception as exc:
        return {
            "gpu_name": f"unavailable ({type(exc).__name__})",
            "vram_used_mib": 0.0,
            "vram_total_mib": 0.0,
            "gpu_util_pct": 0.0,
        }


def safe_git_head(repo_dir: Path) -> str:
    """Return short git HEAD commit hash cleanly, or 'unknown'."""
    if not (repo_dir / ".git").exists():
        return "unknown"
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return proc.stdout.strip() if proc.returncode == 0 and proc.stdout.strip() else "unknown"
    except Exception:
        return "unknown"


def _list_files_meta(directory: Path, pattern: str = "*") -> List[Dict[str, Any]]:
    if not directory.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for item in sorted(directory.glob(pattern)):
        if item.is_file() and not item.name.startswith("."):
            st = item.stat()
            out.append(
                {
                    "name": item.name,
                    "bytes": int(st.st_size),
                    "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                }
            )
    return out


def _tail_text_file(path: Path, lines: int) -> List[str]:
    if not path.is_file():
        return []
    n = max(1, min(500, int(lines)))
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return content[-n:]


def _build_worker_env(hf_home: str) -> Dict[str, str]:
    env = dict(os.environ)
    env["HF_HOME"] = hf_home
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUTF8"] = "1"
    return env


def _build_validated_stage_cmd(job: Dict[str, Any], work_root: Path, repo_dir: Path) -> List[str]:
    stage = str(job.get("stage", "")).strip()
    if stage not in ALLOWED_STAGES:
        raise ValueError(f"Unsupported stage={stage!r}; expected one of {ALLOWED_STAGES}")

    runner = repo_dir / "research" / "stage4_real_and_surgical" / "colab_stage6_runner.py"
    if not runner.is_file():
        raise FileNotFoundError(f"Missing runner script: {runner}")

    cmd = [
        sys.executable,
        "-u",
        "-B",
        str(runner),
        "--stage",
        stage,
        "--work-dir",
        str(work_root / "work"),
        "--results-dir",
        str(work_root / "results"),
    ]
    if "seed" in job and job["seed"] is not None:
        cmd.extend(["--seed", str(int(job["seed"]))])
    if "steps_stage6" in job and job["steps_stage6"] is not None:
        steps = int(job["steps_stage6"])
        if steps < 1:
            raise ValueError(f"steps_stage6 must be >= 1, got {steps}")
        cmd.extend(["--steps-stage6", str(steps)])
    if "g1_blocks_scaled" in job and job["g1_blocks_scaled"] is not None:
        blocks = int(job["g1_blocks_scaled"])
        if blocks < 1:
            raise ValueError(f"g1_blocks_scaled must be >= 1, got {blocks}")
        cmd.extend(["--g1-blocks-scaled", str(blocks)])
    if "run_u1_scaled" in job and job["run_u1_scaled"] is not None:
        val = str(job["run_u1_scaled"]).strip().lower()
        if val not in ALLOWED_RUN_U1_MODES:
            raise ValueError(f"run_u1_scaled must be in {ALLOWED_RUN_U1_MODES}, got {val!r}")
        cmd.extend(["--run-u1-scaled", val])
    if "force_quant" in job and job["force_quant"] is not None:
        fq = str(job["force_quant"]).strip().lower()
        if fq not in ALLOWED_QUANT_MODES:
            raise ValueError(f"force_quant must be in {ALLOWED_QUANT_MODES}, got {fq!r}")
        cmd.extend(["--force-quant", fq])
    if "eval_bs" in job and job["eval_bs"] is not None:
        ebs = int(job["eval_bs"])
        if ebs < 1:
            raise ValueError(f"eval_bs must be >= 1, got {ebs}")
        cmd.extend(["--eval-bs", str(ebs)])
    if bool(job.get("dry_run", False)):
        cmd.append("--dry-run")
    return cmd


def _build_validated_python_cmd(job: Dict[str, Any], repo_dir: Path) -> List[str]:
    script_raw = str(job.get("script", "")).strip()
    extra_args_raw = job.get("args", [])
    if not isinstance(extra_args_raw, list) or len(extra_args_raw) > 32:
        raise ValueError("job['args'] must be a list of <= 32 strings")
    extra_args: List[str] = []
    for item in extra_args_raw:
        s_item = str(item)
        if "\x00" in s_item or "\n" in s_item or "\r" in s_item:
            raise ValueError("Invalid control character in run_python argument")
        extra_args.append(s_item)

    if script_raw == "-m pytest":
        test_file = repo_dir / "research" / "stage4_real_and_surgical" / "test_stage4_cpu.py"
        if not test_file.is_file():
            raise FileNotFoundError(f"Missing test suite: {test_file}")
        return [
            sys.executable,
            "-u",
            "-B",
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            str(test_file),
            *extra_args,
        ]

    allowed_dir = repo_dir / "research" / "stage4_real_and_surgical"
    target = resolve_within(allowed_dir, script_raw)
    if target.suffix != ".py":
        raise ValueError(f"run_python only permits .py scripts inside {allowed_dir}, got {script_raw!r}")
    if not target.is_file():
        raise FileNotFoundError(f"Python script not found: {target}")
    return [sys.executable, "-u", "-B", str(target), *extra_args]


def _stream_subprocess(
    cmd: Sequence[str],
    cwd: Path,
    env: Dict[str, str],
    log_path: Path,
    on_tick: Optional[Callable[[], None]] = None,
    flush_interval_s: float = 8.0,
) -> int:
    """Execute subprocess safely without `shell=True`, periodically flushing logs & heartbeats."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        list(cmd),
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert proc.stdout is not None
    q_lines: "queue.Queue[Optional[str]]" = queue.Queue()

    def _reader() -> None:
        try:
            for line in proc.stdout:
                q_lines.put(line)
        finally:
            q_lines.put(None)

    t_reader = threading.Thread(target=_reader, daemon=True)
    t_reader.start()

    last_flush = time.monotonic()
    last_tick = time.monotonic()
    eof_seen = False

    with log_path.open("a", encoding="utf-8", newline="\n") as log_fh:
        log_fh.write(f"[{_utc_now_iso()}] $ {' '.join(cmd)}\n")
        log_fh.flush()
        while not eof_seen or not q_lines.empty():
            try:
                item = q_lines.get(timeout=1.0)
                if item is None:
                    eof_seen = True
                else:
                    log_fh.write(item)
            except queue.Empty:
                pass
            now = time.monotonic()
            if (now - last_flush) >= flush_interval_s or eof_seen:
                log_fh.flush()
                try:
                    os.fsync(log_fh.fileno())
                except OSError:
                    pass
                last_flush = now
            if on_tick is not None and (now - last_tick) >= flush_interval_s:
                on_tick()
                last_tick = now
    rc = proc.wait()
    t_reader.join(timeout=2.0)
    with log_path.open("a", encoding="utf-8", newline="\n") as log_fh:
        log_fh.write(f"[{_utc_now_iso()}] [EXIT_CODE] {rc}\n")
        log_fh.flush()
    return int(rc)


class ColabBridgeWorker:
    """Pull-worker polling `<work-root>/bridge/inbox/` and writing results + heartbeats."""

    def __init__(
        self,
        work_root: Path,
        repo_dir: Path,
        branch: str = "research/stage4-math-and-realdata",
        poll_s: float = 5.0,
        heartbeat_s: float = 15.0,
        hf_home: str = "/content/hf_cache",
        verbose: bool = True,
    ):
        self.work_root = Path(work_root).resolve()
        self.repo_dir = Path(repo_dir).resolve()
        self.branch = str(branch)
        self.poll_s = float(poll_s)
        self.heartbeat_s = float(heartbeat_s)
        self.hf_home = str(hf_home)
        self.verbose = bool(verbose)

        self.bridge_dir = self.work_root / "bridge"
        self.inbox_dir = self.bridge_dir / "inbox"
        self.processed_dir = self.bridge_dir / "processed"
        self.outbox_dir = self.bridge_dir / "outbox"
        self.heartbeat_path = self.bridge_dir / "heartbeat.json"
        self.results_dir = self.work_root / "results"
        self.work_dir = self.work_root / "work"

        self.worker_id = f"{socket.gethostname()}-{os.getpid()}"
        self.completed_count = 0
        self.active_job_id: Optional[str] = None
        self.stop_requested = False

        for d in (
            self.work_root,
            self.results_dir,
            self.work_dir,
            self.inbox_dir,
            self.processed_dir,
            self.outbox_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[BridgeWorker {_utc_now_iso()}] {msg}", flush=True)

    def write_heartbeat(self) -> Dict[str, Any]:
        gpu = query_gpu_stats()
        payload = {
            "worker_id": self.worker_id,
            "pid": int(os.getpid()),
            "hostname": socket.gethostname(),
            "timestamp_utc": _utc_now_iso(),
            **gpu,
            "active_job_id": self.active_job_id,
            "git_head": safe_git_head(self.repo_dir),
            "results_files": _list_files_meta(self.results_dir, "*.json"),
            "completed_count": int(self.completed_count),
            "work_root": str(self.work_root),
            "repo_dir": str(self.repo_dir),
        }
        write_json_atomic(self.heartbeat_path, payload)
        return payload

    def _collect_status_details(self) -> Dict[str, Any]:
        gpu = query_gpu_stats()
        usage = shutil.disk_usage(self.work_root)
        git_log_lines: List[str] = []
        if (self.repo_dir / ".git").exists():
            proc = subprocess.run(
                ["git", "-C", str(self.repo_dir), "log", "-n", "3", "--oneline"],
                capture_output=True,
                text=True,
                timeout=8,
                check=False,
            )
            if proc.returncode == 0:
                git_log_lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        return {
            "timestamp_utc": _utc_now_iso(),
            "gpu": gpu,
            "disk_free_gib": round(usage.free / (1024.0 ** 3), 2),
            "disk_total_gib": round(usage.total / (1024.0 ** 3), 2),
            "git_head": safe_git_head(self.repo_dir),
            "git_recent": git_log_lines,
            "results_files": _list_files_meta(self.results_dir, "*.json"),
            "work_entries": sorted(p.name for p in self.work_dir.iterdir() if not p.name.startswith(".")),
        }

    def execute_job_file(self, job_path: Path) -> Dict[str, Any]:
        t0 = time.perf_counter()
        raw_text = job_path.read_text(encoding="utf-8")
        raw_obj = json.loads(raw_text)
        job_id = _sanitize_job_id(raw_obj.get("job_id") or job_path.stem)
        done_marker = self.processed_dir / f"{job_id}.done"
        log_path = self.outbox_dir / f"{job_id}.log"
        result_path = self.outbox_dir / f"{job_id}.result.json"

        action = str(raw_obj.get("action", "")).strip()
        self.active_job_id = job_id
        self.write_heartbeat()
        self._log(f"Picked up job={job_id} action={action!r}")

        started_at = _utc_now_iso()
        exit_code = 0
        status = "ok"
        error_msg: Optional[str] = None
        details: Dict[str, Any] = {}

        try:
            if action not in ALLOWED_ACTIONS:
                raise ValueError(f"Disallowed action={action!r}; allowed={ALLOWED_ACTIONS}")

            if action == "status":
                details = self._collect_status_details()
                log_path.write_text(
                    json.dumps(details, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
            elif action == "tail_file":
                rel_file = str(raw_obj.get("file") or "stage6_live.log")
                lines = int(raw_obj.get("lines", 60))
                target = resolve_within(self.work_root, rel_file)
                tail_lines = _tail_text_file(target, lines=lines)
                details = {
                    "file": str(target),
                    "exists": target.is_file(),
                    "line_count": len(tail_lines),
                    "lines": tail_lines,
                }
                log_path.write_text(("\n".join(tail_lines) + "\n") if tail_lines else "", encoding="utf-8")
            elif action == "stop_worker":
                self.stop_requested = True
                details = {"stop_requested": True}
                log_path.write_text(f"[{_utc_now_iso()}] Worker stop requested via {job_id}\n", encoding="utf-8")
            elif action == "git_sync":
                env = _build_worker_env(self.hf_home)
                if not (self.repo_dir / ".git").exists():
                    rc = _stream_subprocess(
                        [
                            "git",
                            "clone",
                            "--depth",
                            "1",
                            "-b",
                            self.branch,
                            "https://github.com/20171806011/EM-FloatFormer.git",
                            str(self.repo_dir),
                        ],
                        cwd=self.work_root,
                        env=env,
                        log_path=log_path,
                        on_tick=self.write_heartbeat,
                        flush_interval_s=self.heartbeat_s,
                    )
                else:
                    rc = _stream_subprocess(
                        ["git", "-C", str(self.repo_dir), "fetch", "origin", self.branch, "--depth", "1"],
                        cwd=self.repo_dir,
                        env=env,
                        log_path=log_path,
                        on_tick=self.write_heartbeat,
                        flush_interval_s=self.heartbeat_s,
                    )
                    if rc == 0:
                        rc = _stream_subprocess(
                            ["git", "-C", str(self.repo_dir), "reset", "--hard", f"origin/{self.branch}"],
                            cwd=self.repo_dir,
                            env=env,
                            log_path=log_path,
                            on_tick=self.write_heartbeat,
                            flush_interval_s=self.heartbeat_s,
                        )
                exit_code = rc
                if rc != 0:
                    status = "error"
                    error_msg = f"git_sync exited with {rc}"
                details = {"git_head": safe_git_head(self.repo_dir)}
            elif action == "run_stage":
                cmd = _build_validated_stage_cmd(raw_obj, self.work_root, self.repo_dir)
                env = _build_worker_env(self.hf_home)
                rc = _stream_subprocess(
                    cmd,
                    cwd=self.repo_dir,
                    env=env,
                    log_path=log_path,
                    on_tick=self.write_heartbeat,
                    flush_interval_s=self.heartbeat_s,
                )
                exit_code = rc
                if rc != 0:
                    status = "error"
                    error_msg = f"run_stage exited with {rc}"
                details = {"stage": raw_obj.get("stage"), "dry_run": bool(raw_obj.get("dry_run", False))}
            elif action == "run_python":
                cmd = _build_validated_python_cmd(raw_obj, self.repo_dir)
                env = _build_worker_env(self.hf_home)
                rc = _stream_subprocess(
                    cmd,
                    cwd=self.repo_dir,
                    env=env,
                    log_path=log_path,
                    on_tick=self.write_heartbeat,
                    flush_interval_s=self.heartbeat_s,
                )
                exit_code = rc
                if rc != 0:
                    status = "error"
                    error_msg = f"run_python exited with {rc}"
                details = {"script": raw_obj.get("script")}
        except Exception as exc:
            exit_code = 1
            status = "error"
            error_msg = f"{type(exc).__name__}: {exc}"
            with log_path.open("a", encoding="utf-8", newline="\n") as log_fh:
                log_fh.write(f"[{_utc_now_iso()}] [BRIDGE_ERROR] {error_msg}\n")

        result = {
            "job_id": job_id,
            "action": action,
            "status": status,
            "exit_code": int(exit_code),
            "error": error_msg,
            "started_at_utc": started_at,
            "finished_at_utc": _utc_now_iso(),
            "wall_s": round(time.perf_counter() - t0, 3),
            "worker_id": self.worker_id,
            "git_head": safe_git_head(self.repo_dir),
            "log_file": log_path.name,
            "details": details,
        }
        write_json_atomic(result_path, result)
        write_json_atomic(done_marker, {"job_id": job_id, "finished_at_utc": result["finished_at_utc"], "status": status})

        self.completed_count += 1
        self.active_job_id = None
        self.write_heartbeat()
        self._log(f"Completed job={job_id} status={status} exit_code={exit_code} wall={result['wall_s']}s")
        return result

    def process_pending(self) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        candidates = sorted(
            p for p in self.inbox_dir.glob("*.json") if p.is_file() and not p.name.startswith(".")
        )
        for job_path in candidates:
            try:
                raw = json.loads(job_path.read_text(encoding="utf-8"))
                jid = _sanitize_job_id(raw.get("job_id") or job_path.stem)
            except Exception:
                jid = _sanitize_job_id(job_path.stem)
            if (self.processed_dir / f"{jid}.done").exists():
                continue
            results.append(self.execute_job_file(job_path))
            if self.stop_requested:
                break
        return results

    def serve(self, once: bool = False) -> int:
        self.write_heartbeat()
        self._log(f"Started worker_id={self.worker_id} work_root={self.work_root} repo_dir={self.repo_dir}")
        last_hb = time.monotonic()
        while True:
            self.process_pending()
            if once or self.stop_requested:
                break
            now = time.monotonic()
            if (now - last_hb) >= self.heartbeat_s:
                self.write_heartbeat()
                last_hb = now
            time.sleep(max(0.5, self.poll_s))
        self.write_heartbeat()
        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Colab-side Google Drive bidirectional CLI bridge worker.")
    p.add_argument(
        "--work-root",
        type=Path,
        default=Path("/content/drive/MyDrive/EM-FloatFormer-Stage6"),
        help="Shared Google Drive (or local) working root.",
    )
    p.add_argument(
        "--repo-dir",
        type=Path,
        default=Path("/content/EM-FloatFormer"),
        help="Git clone root inside Colab runtime.",
    )
    p.add_argument("--branch", type=str, default="research/stage4-math-and-realdata")
    p.add_argument("--poll-s", type=float, default=5.0)
    p.add_argument("--heartbeat-s", type=float, default=15.0)
    p.add_argument("--hf-home", type=str, default="/content/hf_cache")
    p.add_argument("--once", action="store_true", help="Process all pending inbox jobs once and exit.")
    p.add_argument("--quiet", action="store_true")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    worker = ColabBridgeWorker(
        work_root=args.work_root,
        repo_dir=args.repo_dir,
        branch=args.branch,
        poll_s=args.poll_s,
        heartbeat_s=args.heartbeat_s,
        hf_home=args.hf_home,
        verbose=not args.quiet,
    )
    return worker.serve(once=bool(args.once))


if __name__ == "__main__":
    raise SystemExit(main())
