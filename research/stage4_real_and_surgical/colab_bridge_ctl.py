"""Local controller CLI for the Google Drive bidirectional Colab CLI bridge.

Talks either directly to Google Drive via the Antigravity `gdrive` CLI plugin
(`--transport gdrive`, read operations strictly via `gdrive readonly ...`) OR to a
local directory (`--transport local --local-root ...`) for zero-network CPU tests.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

_FILE_P = Path(__file__).resolve()
_REPO_ROOT = _FILE_P.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.stage4_real_and_surgical.colab_bridge_worker import (
    ALLOWED_ACTIONS,
    ALLOWED_QUANT_MODES,
    ALLOWED_RUN_U1_MODES,
    ALLOWED_STAGES,
    _sanitize_job_id,
    resolve_within,
    write_json_atomic,
)


FOLDER_MIME = "application/vnd.google-apps.folder"


def _utc_stamp_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")


class _BaseTransport:
    def ls_folder_path(self, rel_parts: List[str]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def read_text_path(self, rel_parts: List[str]) -> Optional[str]:
        raise NotImplementedError

    def download_path(self, rel_parts: List[str], dest_path: Path) -> bool:
        raise NotImplementedError

    def upload_bytes_path(self, rel_parts: List[str], data: bytes) -> Dict[str, Any]:
        raise NotImplementedError

    def root_identifier(self) -> str:
        raise NotImplementedError


class _LocalTransport(_BaseTransport):
    def __init__(self, local_root: Path):
        self.root = Path(local_root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, rel_parts: List[str]) -> Path:
        if not rel_parts:
            return self.root
        return resolve_within(self.root, "/".join(rel_parts))

    def root_identifier(self) -> str:
        return str(self.root)

    def ls_folder_path(self, rel_parts: List[str]) -> List[Dict[str, Any]]:
        target = self._resolve(rel_parts)
        if not target.is_dir():
            return []
        entries: List[Dict[str, Any]] = []
        for item in sorted(target.iterdir()):
            if item.name.startswith("."):
                continue
            st = item.stat()
            entries.append(
                {
                    "id": str(item),
                    "name": item.name,
                    "mimeType": FOLDER_MIME if item.is_dir() else "application/octet-stream",
                    "size": 0 if item.is_dir() else int(st.st_size),
                    "modifiedTime": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                }
            )
        return entries

    def read_text_path(self, rel_parts: List[str]) -> Optional[str]:
        target = self._resolve(rel_parts)
        if not target.is_file():
            return None
        return target.read_text(encoding="utf-8", errors="replace")

    def download_path(self, rel_parts: List[str], dest_path: Path) -> bool:
        src = self._resolve(rel_parts)
        if not src.is_file():
            return False
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_path)
        return True

    def upload_bytes_path(self, rel_parts: List[str], data: bytes) -> Dict[str, Any]:
        if not rel_parts:
            raise ValueError("Cannot upload directly onto root directory")
        dst = self._resolve(rel_parts)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.parent / f".{dst.name}.tmp"
        tmp.write_bytes(data)
        tmp.replace(dst)
        return {"id": str(dst), "name": dst.name, "size": len(data)}


class _GDriveTransport(_BaseTransport):
    def __init__(self, root_name: str = "EM-FloatFormer-Stage6", root_id: Optional[str] = None):
        self.root_name = str(root_name)
        self._root_id = root_id
        self._folder_cache: Dict[tuple, str] = {}

    def _run_gdrive(self, args: List[str]) -> str:
        # Resolve the PATH-registered `gdrive` wrapper cleanly across Windows/POSIX.
        exe = shutil.which("gdrive") or "gdrive"
        proc = subprocess.run(
            [exe, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"gdrive {' '.join(args)} failed (rc={proc.returncode}): {(proc.stderr or proc.stdout).strip()}"
            )
        return proc.stdout

    def _run_gdrive_json(self, args: List[str]) -> Any:
        out = self._run_gdrive([*args, "--json"]).strip()
        if not out:
            return []
        return json.loads(out)

    def root_identifier(self) -> str:
        if self._root_id:
            return self._root_id
        rows = self._run_gdrive_json(["readonly", "search", "--name-contains", self.root_name]) or []
        exact_folders = [
            r for r in rows if r.get("name") == self.root_name and r.get("mimeType") == FOLDER_MIME
        ]
        if not exact_folders:
            raise FileNotFoundError(
                f"Google Drive folder {self.root_name!r} not found. Run Colab Cell 1 to create/mount it."
            )
        exact_folders.sort(key=lambda r: str(r.get("modifiedTime", "")), reverse=True)
        self._root_id = str(exact_folders[0]["id"])
        self._folder_cache[()] = self._root_id
        return self._root_id

    def _resolve_folder_id(self, rel_parts: List[str], create_missing: bool = False) -> Optional[str]:
        cur_id = self.root_identifier()
        key = ()
        for part in rel_parts:
            key = (*key, part)
            if key in self._folder_cache:
                cur_id = self._folder_cache[key]
                continue
            items = self._run_gdrive_json(["readonly", "ls", cur_id, "--max", "0"]) or []
            matches = [
                it for it in items if it.get("name") == part and it.get("mimeType") == FOLDER_MIME
            ]
            if not matches:
                if not create_missing:
                    return None
                created = self._run_gdrive_json(["mutate", "mkdir", part, "--parent", cur_id])
                if isinstance(created, list):
                    created = created[0]
                cur_id = str(created["id"])
            else:
                matches.sort(key=lambda r: str(r.get("modifiedTime", "")), reverse=True)
                cur_id = str(matches[0]["id"])
            self._folder_cache[key] = cur_id
        return cur_id

    def _find_file_entry(self, rel_parts: List[str]) -> Optional[Dict[str, Any]]:
        if not rel_parts:
            return None
        folder_id = self._resolve_folder_id(rel_parts[:-1], create_missing=False)
        if not folder_id:
            return None
        target_name = rel_parts[-1]
        items = self._run_gdrive_json(["readonly", "ls", folder_id, "--max", "0"]) or []
        matches = [
            it for it in items if it.get("name") == target_name and it.get("mimeType") != FOLDER_MIME
        ]
        if not matches:
            return None
        matches.sort(key=lambda r: str(r.get("modifiedTime", "")), reverse=True)
        return matches[0]

    def ls_folder_path(self, rel_parts: List[str]) -> List[Dict[str, Any]]:
        folder_id = self._resolve_folder_id(rel_parts, create_missing=False)
        if not folder_id:
            return []
        items = self._run_gdrive_json(["readonly", "ls", folder_id, "--max", "0"]) or []
        return sorted(items, key=lambda it: str(it.get("name", "")))

    def download_path(self, rel_parts: List[str], dest_path: Path) -> bool:
        entry = self._find_file_entry(rel_parts)
        if not entry:
            return False
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        self._run_gdrive(["readonly", "download", str(entry["id"]), "--out", str(dest_path)])
        return dest_path.is_file()

    def read_text_path(self, rel_parts: List[str]) -> Optional[str]:
        with tempfile.TemporaryDirectory() as td:
            out_p = Path(td) / "item.txt"
            if not self.download_path(rel_parts, out_p):
                return None
            return out_p.read_text(encoding="utf-8", errors="replace")

    def upload_bytes_path(self, rel_parts: List[str], data: bytes) -> Dict[str, Any]:
        if not rel_parts:
            raise ValueError("Cannot upload directly onto root folder")
        parent_id = self._resolve_folder_id(rel_parts[:-1], create_missing=True)
        assert parent_id is not None
        with tempfile.TemporaryDirectory() as td:
            src_p = Path(td) / rel_parts[-1]
            src_p.write_bytes(data)
            res = self._run_gdrive_json(["mutate", "upload", str(src_p), "--parent", parent_id])
            return res[0] if isinstance(res, list) and res else (res or {"name": rel_parts[-1]})


def _build_transport(args: argparse.Namespace) -> _BaseTransport:
    if args.transport == "local":
        if not args.local_root:
            raise ValueError("--local-root is required when --transport local")
        return _LocalTransport(args.local_root)
    return _GDriveTransport(root_name=args.drive_root_name, root_id=args.drive_root_id)


def _format_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "name": str(it.get("name", "")),
            "id": str(it.get("id", "")),
            "is_dir": it.get("mimeType") == FOLDER_MIME,
            "bytes": int(it.get("size", 0) or 0),
            "modified_utc": it.get("modifiedTime"),
        }
        for it in entries
        if not str(it.get("name", "")).startswith(".")
    ]


def cmd_ls(tr: _BaseTransport) -> Dict[str, Any]:
    return {
        "root": tr.root_identifier(),
        "root_entries": _format_entries(tr.ls_folder_path([])),
        "results": _format_entries(tr.ls_folder_path(["results"])),
        "work": _format_entries(tr.ls_folder_path(["work"])),
        "bridge_inbox": _format_entries(tr.ls_folder_path(["bridge", "inbox"])),
        "bridge_outbox": _format_entries(tr.ls_folder_path(["bridge", "outbox"])),
        "bridge_processed": _format_entries(tr.ls_folder_path(["bridge", "processed"])),
    }


def cmd_status(tr: _BaseTransport, tail_lines: int = 25) -> Dict[str, Any]:
    hb_raw = tr.read_text_path(["bridge", "heartbeat.json"])
    heartbeat = json.loads(hb_raw) if hb_raw else None
    results_entries = [
        it for it in _format_entries(tr.ls_folder_path(["results"])) if it["name"].endswith(".json")
    ]
    work_entries = _format_entries(tr.ls_folder_path(["work"]))
    root_entries = _format_entries(tr.ls_folder_path([]))

    live_log_info: Optional[Dict[str, Any]] = None
    live_matches = [it for it in root_entries if it["name"] == "stage6_live.log"]
    log_rel: Optional[List[str]] = ["stage6_live.log"] if live_matches else None
    log_meta = live_matches[0] if live_matches else None

    if log_rel is None:
        outbox_logs = [
            it for it in _format_entries(tr.ls_folder_path(["bridge", "outbox"])) if it["name"].endswith(".log")
        ]
        if outbox_logs:
            outbox_logs.sort(key=lambda x: str(x.get("modified_utc") or x["name"]), reverse=True)
            log_meta = outbox_logs[0]
            log_rel = ["bridge", "outbox", log_meta["name"]]

    if log_rel is not None and log_meta is not None:
        txt = tr.read_text_path(log_rel)
        lines = txt.splitlines() if txt is not None else []
        live_log_info = {
            "name": log_meta["name"],
            "bytes": log_meta["bytes"],
            "modified_utc": log_meta["modified_utc"],
            "line_count": len(lines),
            "tail": lines[-max(1, int(tail_lines)) :],
        }

    return {
        "root": tr.root_identifier(),
        "heartbeat": heartbeat,
        "results_files": results_entries,
        "work_entries": work_entries,
        "live_log": live_log_info,
    }


def cmd_pull_results(tr: _BaseTransport, out_dir: Path, include_logs: bool = False) -> Dict[str, Any]:
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []

    res_items = [it for it in _format_entries(tr.ls_folder_path(["results"])) if it["name"].endswith(".json")]
    for it in res_items:
        dst = out_dir / it["name"]
        ok = tr.download_path(["results", it["name"]], dst)
        valid_json = False
        keys: List[str] = []
        if ok and dst.is_file():
            parsed = json.loads(dst.read_text(encoding="utf-8"))
            valid_json = isinstance(parsed, dict)
            if valid_json:
                keys = list(parsed.keys())[:12]
        manifest.append(
            {"file": it["name"], "path": str(dst), "bytes": dst.stat().st_size if ok else 0, "valid_json": valid_json, "top_keys": keys}
        )

    logs_pulled: List[str] = []
    if include_logs:
        logs_dir = out_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        if tr.download_path(["stage6_live.log"], logs_dir / "stage6_live.log"):
            logs_pulled.append("stage6_live.log")
        for it in _format_entries(tr.ls_folder_path(["bridge", "outbox"])):
            if it["name"].endswith(".log") or it["name"].endswith(".result.json"):
                if tr.download_path(["bridge", "outbox", it["name"]], logs_dir / it["name"]):
                    logs_pulled.append(it["name"])

    return {
        "root": tr.root_identifier(),
        "out_dir": str(out_dir),
        "downloaded_count": len(manifest),
        "manifest": manifest,
        "logs_pulled": logs_pulled,
    }


def cmd_tail(tr: _BaseTransport, file_rel: Optional[str], job_id: Optional[str], lines: int) -> Dict[str, Any]:
    if job_id:
        rel_parts = ["bridge", "outbox", f"{_sanitize_job_id(job_id)}.log"]
    else:
        clean_rel = (file_rel or "stage6_live.log").strip().replace("\\", "/").strip("/")
        rel_parts = [p for p in clean_rel.split("/") if p]
    txt = tr.read_text_path(rel_parts)
    if txt is None:
        return {"path": "/".join(rel_parts), "exists": False, "lines": []}
    split_lines = txt.splitlines()
    return {
        "path": "/".join(rel_parts),
        "exists": True,
        "total_lines": len(split_lines),
        "lines": split_lines[-max(1, int(lines)) :],
    }


def _build_job_payload(args: argparse.Namespace) -> Dict[str, Any]:
    action = str(args.action)
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Unsupported action={action!r}")
    job_id = _sanitize_job_id(args.job_id or f"job_{_utc_stamp_id()}_{action}")
    payload: Dict[str, Any] = {
        "job_id": job_id,
        "action": action,
        "submitted_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if action == "run_stage":
        if not args.stage or args.stage not in ALLOWED_STAGES:
            raise ValueError(f"--stage in {ALLOWED_STAGES} is required for action=run_stage")
        payload["stage"] = args.stage
        payload["seed"] = int(args.seed)
        payload["steps_stage6"] = int(args.steps_stage6)
        payload["g1_blocks_scaled"] = int(args.g1_blocks_scaled)
        payload["run_u1_scaled"] = args.run_u1_scaled
        if args.force_quant:
            payload["force_quant"] = args.force_quant
        if args.eval_bs is not None:
            payload["eval_bs"] = int(args.eval_bs)
        if args.dry_run:
            payload["dry_run"] = True
    elif action == "run_python":
        if not args.script:
            raise ValueError("--script is required for action=run_python")
        payload["script"] = args.script
        payload["args"] = json.loads(args.args_json) if args.args_json else []
    elif action == "tail_file":
        payload["file"] = args.file or "stage6_live.log"
        payload["lines"] = int(args.lines)
    return payload


def cmd_submit(tr: _BaseTransport, args: argparse.Namespace) -> Dict[str, Any]:
    payload = _build_job_payload(args)
    job_id = payload["job_id"]
    raw_bytes = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    uploaded = tr.upload_bytes_path(["bridge", "inbox", f"{job_id}.json"], raw_bytes)
    out: Dict[str, Any] = {"submitted": True, "job": payload, "upload": uploaded}

    if not args.wait:
        return out

    deadline = time.monotonic() + float(args.timeout_s)
    seen_chars = 0
    while True:
        log_txt = tr.read_text_path(["bridge", "outbox", f"{job_id}.log"])
        if log_txt is not None and len(log_txt) > seen_chars:
            sys.stdout.write(log_txt[seen_chars:])
            sys.stdout.flush()
            seen_chars = len(log_txt)

        res_txt = tr.read_text_path(["bridge", "outbox", f"{job_id}.result.json"])
        if res_txt is not None:
            res_obj = json.loads(res_txt)
            # Final log flush check
            log_txt2 = tr.read_text_path(["bridge", "outbox", f"{job_id}.log"])
            if log_txt2 is not None and len(log_txt2) > seen_chars:
                sys.stdout.write(log_txt2[seen_chars:])
                sys.stdout.flush()
            out["result"] = res_obj
            return out

        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out after {args.timeout_s}s waiting for job {job_id}")
        time.sleep(max(1.0, float(args.poll_s)))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Local CLI controller for the Google Drive Colab bridge.")
    p.add_argument("--transport", choices=("gdrive", "local"), default="gdrive")
    p.add_argument("--drive-root-name", type=str, default="EM-FloatFormer-Stage6")
    p.add_argument("--drive-root-id", type=str, default=None)
    p.add_argument("--local-root", type=Path, default=None)

    sub = p.add_subparsers(dest="subcmd", required=True)
    sub.add_parser("ls", help="List shared Drive/local workspace directories.")

    p_st = sub.add_parser("status", help="Inspect worker heartbeat, results/*.json, and live log tail.")
    p_st.add_argument("--tail-lines", type=int, default=25)

    p_pr = sub.add_parser("pull-results", help="Download and validate all results/*.json files.")
    p_pr.add_argument("--out", type=Path, required=True)
    p_pr.add_argument("--include-logs", action="store_true")

    p_tl = sub.add_parser("tail", help="Tail stage6_live.log or a specific outbox job log.")
    p_tl.add_argument("--file", type=str, default=None)
    p_tl.add_argument("--job", type=str, default=None)
    p_tl.add_argument("--lines", type=int, default=60)

    p_sb = sub.add_parser("submit", help="Enqueue an allowlisted job for colab_bridge_worker.py.")
    p_sb.add_argument("--action", required=True, choices=ALLOWED_ACTIONS)
    p_sb.add_argument("--job-id", type=str, default=None)
    p_sb.add_argument("--stage", choices=ALLOWED_STAGES, default=None)
    p_sb.add_argument("--seed", type=int, default=202611)
    p_sb.add_argument("--steps-stage6", type=int, default=240)
    p_sb.add_argument("--g1-blocks-scaled", type=int, default=32)
    p_sb.add_argument("--run-u1-scaled", choices=ALLOWED_RUN_U1_MODES, default="auto")
    p_sb.add_argument("--force-quant", choices=ALLOWED_QUANT_MODES, default=None)
    p_sb.add_argument("--eval-bs", type=int, default=None)
    p_sb.add_argument("--dry-run", action="store_true")
    p_sb.add_argument("--script", type=str, default=None)
    p_sb.add_argument("--args-json", type=str, default=None)
    p_sb.add_argument("--file", type=str, default=None)
    p_sb.add_argument("--lines", type=int, default=60)
    p_sb.add_argument("--wait", action="store_true")
    p_sb.add_argument("--poll-s", type=float, default=10.0)
    p_sb.add_argument("--timeout-s", type=float, default=7200.0)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    tr = _build_transport(args)

    if args.subcmd == "ls":
        res = cmd_ls(tr)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    if args.subcmd == "status":
        res = cmd_status(tr, tail_lines=args.tail_lines)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    if args.subcmd == "pull-results":
        res = cmd_pull_results(tr, out_dir=args.out, include_logs=bool(args.include_logs))
        print(json.dumps(res, ensure_ascii=False, indent=2))
        return 0
    if args.subcmd == "tail":
        res = cmd_tail(tr, file_rel=args.file, job_id=args.job, lines=args.lines)
        if not res["exists"]:
            print(f"[MISSING] {res['path']}", file=sys.stderr)
            return 0
        for ln in res["lines"]:
            print(ln)
        return 0
    if args.subcmd == "submit":
        res = cmd_submit(tr, args)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        if "result" in res:
            return int(res["result"].get("exit_code", 1))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
