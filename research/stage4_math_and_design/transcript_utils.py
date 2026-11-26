"""Utilities for parsing Codex CLI JSONL and Antigravity CLI stream-json logs into verbatim Markdown."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def read_text_auto(path: Path) -> str:
    """Read UTF-8 (with/without BOM) or PowerShell-redirected UTF-16 text files."""
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig", errors="replace")
    return raw.decode("utf-8", errors="replace")


def normalize_utf8_file(path: Path) -> Dict[str, Any]:
    """Normalize a log/transcript file in place to UTF-8 (no BOM) with LF line endings."""
    before = path.stat().st_size
    text = read_text_auto(path).replace("\r\n", "\n").replace("\r", "\n")
    path.write_text(text, encoding="utf-8", newline="\n")
    after = path.stat().st_size
    return {"path": str(path), "before_bytes": before, "after_bytes": after}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    if not path.exists():
        return events
    text = read_text_auto(path)
    for line_no, raw in enumerate(text.splitlines(), start=1):
        s = raw.strip()
        if not s:
            continue
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as exc:
            obj = {"type": "raw_parse_error", "line_no": line_no, "error": str(exc), "raw": s}
        events.append(obj)
    return events


def extract_thread_id(events_path: Path) -> str:
    """Extract persistent thread_id (Codex) or conversation_id (Antigravity CLI) from round1_events.jsonl."""
    for ev in load_jsonl(events_path):
        for key in ("thread_id", "conversation_id"):
            val = ev.get(key)
            if val:
                return str(val)
        for nested in ("item", "init", "result", "step_update"):
            obj = ev.get(nested)
            if isinstance(obj, dict):
                for key in ("thread_id", "conversation_id"):
                    val = obj.get(key)
                    if val:
                        return str(val)
    raise RuntimeError(f"Could not find thread_id or conversation_id in {events_path}")


def extract_antigravity_reply(events_path: Path) -> str:
    """Extract final response text from Antigravity CLI stream-json events."""
    events = load_jsonl(events_path)
    for ev in reversed(events):
        if ev.get("event") == "result" and isinstance(ev.get("result"), dict):
            resp = ev["result"].get("response")
            if resp is not None:
                return str(resp)
    # Fallback: concatenate text_delta from agent_response steps
    chunks: List[str] = []
    for ev in events:
        if ev.get("event") == "step_update" and isinstance(ev.get("step_update"), dict):
            su = ev["step_update"]
            if su.get("step_type") == "agent_response" and su.get("text_delta"):
                chunks.append(str(su["text_delta"]))
    return "".join(chunks)


def _fence(text: str, lang: str = "") -> str:
    """Wrap text in a Markdown fence guaranteed not to collide with inner backticks."""
    max_run = 3
    cur = 0
    for ch in text:
        if ch == "`":
            cur += 1
            if cur >= max_run:
                max_run = cur + 1
        else:
            cur = 0
    ticks = "`" * max_run
    body = text if text.endswith("\n") or text == "" else text + "\n"
    return f"{ticks}{lang}\n{body}{ticks}\n"


def _format_codex_item_event(idx: int, ev: Dict[str, Any]) -> str:
    ev_type = str(ev.get("type", "unknown"))
    lines: List[str] = []

    if ev_type == "thread.started":
        lines.append(f"### Event #{idx}: `thread.started` (thread_id=`{ev.get('thread_id', '')}`)\n")
        return "\n".join(lines)

    if ev_type == "turn.started":
        lines.append(f"### Event #{idx}: `turn.started`\n")
        return "\n".join(lines)

    if ev_type == "turn.completed":
        usage = ev.get("usage", {})
        lines.append(f"### Event #{idx}: `turn.completed`\n")
        lines.append(f"- Usage: `{json.dumps(usage, ensure_ascii=False)}`\n")
        return "\n".join(lines)

    if ev_type in ("turn.failed", "error"):
        lines.append(f"### Event #{idx}: `{ev_type}`\n")
        lines.append(_fence(json.dumps(ev, ensure_ascii=False, indent=2), "json"))
        return "\n".join(lines)

    item = ev.get("item")
    if not isinstance(item, dict):
        lines.append(f"### Event #{idx}: `{ev_type}`\n")
        lines.append(_fence(json.dumps(ev, ensure_ascii=False, indent=2), "json"))
        return "\n".join(lines)

    item_type = str(item.get("type", "unknown"))
    item_id = item.get("id", "")
    lines.append(f"### Event #{idx}: `{ev_type}` — `{item_type}` (`{item_id}`)\n")

    if item_type == "reasoning":
        txt = str(item.get("text", ""))
        lines.append((txt + "\n") if txt else _fence(json.dumps(item, ensure_ascii=False, indent=2), "json"))
    elif item_type == "agent_message":
        lines.append(str(item.get("text", "")) + "\n")
    elif item_type == "command_execution":
        cmd = str(item.get("command", ""))
        status = item.get("status", "")
        exit_code = item.get("exit_code", None)
        out = str(item.get("aggregated_output", ""))
        lines.append(f"- Status: `{status}`, Exit Code: `{exit_code}`\n")
        lines.append("**Command:**\n")
        lines.append(_fence(cmd, "powershell"))
        if out:
            lines.append("**Aggregated Output:**\n")
            lines.append(_fence(out, "text"))
    elif item_type == "file_change":
        lines.append(_fence(json.dumps(item.get("changes", item), ensure_ascii=False, indent=2), "json"))
    else:
        lines.append(_fence(json.dumps(item, ensure_ascii=False, indent=2), "json"))

    return "\n".join(lines)


def _select_codex_events(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    completed_ids = set()
    for ev in events:
        if ev.get("type") == "item.completed" and isinstance(ev.get("item"), dict) and ev["item"].get("id"):
            completed_ids.add(ev["item"]["id"])
    selected: List[Dict[str, Any]] = []
    for ev in events:
        ev_type = ev.get("type")
        item = ev.get("item")
        if ev_type in ("item.started", "item.updated") and isinstance(item, dict) and item.get("id") in completed_ids:
            continue
        selected.append(ev)
    return selected


def _format_antigravity_events(events: List[Dict[str, Any]]) -> List[str]:
    """Aggregate Antigravity stream-json delta events by step_index preserving all verbatim content."""
    parts: List[str] = []
    steps_order: List[Any] = []
    steps_data: Dict[Any, Dict[str, Any]] = {}
    idx = 0

    def _flush_step(s_key: Any) -> str:
        nonlocal idx
        idx += 1
        sd = steps_data[s_key]
        stype = sd.get("step_type", "unknown")
        state = sd.get("state", "")
        lines: List[str] = [f"### Step `{s_key}` (`{stype}`, state=`{state}`)\n"]
        if sd.get("duration_seconds") is not None:
            lines.append(f"- Duration: `{sd['duration_seconds']:.3f}s`")
        if sd.get("usage"):
            lines.append(f"- Usage: `{json.dumps(sd['usage'], ensure_ascii=False)}`")
        if sd.get("thinking"):
            lines.append("\n**Thinking:**\n")
            lines.append(_fence("".join(sd["thinking"]), "text"))
        if sd.get("text"):
            lines.append("\n**Response / Text:**\n")
            lines.append("".join(sd["text"]) + "\n")
        if sd.get("extras"):
            lines.append("\n**Structured Step Metadata / Tool Calls / Outputs:**\n")
            lines.append(_fence(json.dumps(sd["extras"], ensure_ascii=False, indent=2), "json"))
        return "\n".join(lines)

    for ev in events:
        ev_name = ev.get("event")
        if ev_name == "init":
            init_obj = ev.get("init", {})
            parts.append(
                f"### Session Init (conversation_id=`{ev.get('conversation_id', '')}`, model=`{init_obj.get('model', '')}`)\n"
            )
            parts.append(f"- CWD: `{init_obj.get('cwd', '')}`, Permission Mode: `{init_obj.get('permission_mode', '')}`\n")
        elif ev_name == "step_update":
            su = ev.get("step_update", {})
            s_idx = su.get("step_index")
            if s_idx not in steps_data:
                steps_order.append(s_idx)
                steps_data[s_idx] = {
                    "step_type": su.get("step_type", "unknown"),
                    "state": su.get("state", ""),
                    "thinking": [],
                    "text": [],
                    "extras": [],
                    "usage": None,
                    "duration_seconds": None,
                }
            sd = steps_data[s_idx]
            if su.get("step_type"):
                sd["step_type"] = su["step_type"]
            if su.get("state"):
                sd["state"] = su["state"]
            if su.get("thinking_delta"):
                sd["thinking"].append(str(su["thinking_delta"]))
            if su.get("text_delta"):
                sd["text"].append(str(su["text_delta"]))
            if su.get("usage"):
                sd["usage"] = su["usage"]
            if su.get("duration_seconds") is not None:
                sd["duration_seconds"] = su["duration_seconds"]
            # Capture any extra tool/payload fields verbatim
            extra_keys = {
                k: v
                for k, v in su.items()
                if k not in ("conversation_id", "step_index", "state", "step_type", "text_delta", "thinking_delta", "usage", "duration_seconds")
            }
            if extra_keys:
                sd["extras"].append(extra_keys)
        elif ev_name == "result":
            for s_key in steps_order:
                parts.append(_flush_step(s_key))
            steps_order.clear()
            res = ev.get("result", {})
            summary = {k: v for k, v in res.items() if k != "response"}
            parts.append("### Final `result` Telemetry\n")
            parts.append(_fence(json.dumps(summary, ensure_ascii=False, indent=2), "json"))
        else:
            parts.append("### Raw Event\n")
            parts.append(_fence(json.dumps(ev, ensure_ascii=False, indent=2), "json"))

    for s_key in steps_order:
        parts.append(_flush_step(s_key))
    return parts


def _is_antigravity_stream(events: List[Dict[str, Any]]) -> bool:
    return any("event" in ev and ev.get("event") in ("init", "step_update", "result") for ev in events[:10])


def render_full_conversation(track_dir: Path, track_id: str, out_path: Path) -> Dict[str, Any]:
    """Render complete multi-round Codex CLI or Antigravity CLI conversation verbatim."""
    r1_prompt = read_text_auto(track_dir / "prompt_r1.md")
    r1_events = load_jsonl(track_dir / "round1_events.jsonl")
    r1_reply_p = track_dir / "reply_r1.md"
    r1_reply = read_text_auto(r1_reply_p) if r1_reply_p.exists() else extract_antigravity_reply(track_dir / "round1_events.jsonl")

    r2_prompt_p = track_dir / "prompt_r2.md"
    r2_prompt = read_text_auto(r2_prompt_p) if r2_prompt_p.exists() else ""
    r2_events = load_jsonl(track_dir / "round2_events.jsonl")
    r2_reply_p = track_dir / "reply_r2.md"
    r2_reply = (
        read_text_auto(r2_reply_p)
        if r2_reply_p.exists()
        else (extract_antigravity_reply(track_dir / "round2_events.jsonl") if r2_events else "")
    )

    is_agy = _is_antigravity_stream(r1_events)
    cli_label = "Antigravity Original CLI (`agy-bin.exe`)" if is_agy else "Codex CLI (`codex exec`)"

    thread_id: Optional[str] = None
    try:
        thread_id = extract_thread_id(track_dir / "round1_events.jsonl")
    except RuntimeError:
        pass

    parts: List[str] = [
        f"# Full Verbatim Multi-Round Conversation — Track `{track_id}`\n",
        f"- **CLI Runtime**: {cli_label}",
        f"- **Session / Thread ID**: `{thread_id}`",
        f"- **Round 1 Events**: `{len(r1_events)}` raw JSONL events (`round1_events.jsonl`)",
        f"- **Round 2 Events**: `{len(r2_events)}` raw JSONL events (`round2_events.jsonl`)",
        "- **Completeness Guarantee**: All user prompts (`prompt_r1.md`, `prompt_r2.md`), all completed reasoning/tool/command events, and final responses (`reply_r1.md`, `reply_r2.md`) are included verbatim below without truncation or paraphrasing.\n",
        "---\n",
        "## Round 1 — Initial Mathematical Brief (`prompt_r1.md`)\n",
        _fence(r1_prompt, "markdown"),
        "\n## Round 1 — Execution & Reasoning Trace (`round1_events.jsonl`)\n",
    ]
    if is_agy:
        parts.extend(_format_antigravity_events(r1_events))
    else:
        for i, ev in enumerate(_select_codex_events(r1_events), start=1):
            parts.append(_format_codex_item_event(i, ev))

    parts.append("\n## Round 1 — Final Output (`reply_r1.md`)\n")
    parts.append(_fence(r1_reply, "markdown"))

    if r2_prompt or r2_events or r2_reply:
        parts.extend([
            "\n---\n",
            "## Round 2 — Cross-Review & Convergence Prompt (`prompt_r2.md`)\n",
            _fence(r2_prompt, "markdown"),
            "\n## Round 2 — Resumed Execution & Reasoning Trace (`round2_events.jsonl`)\n",
        ])
        if _is_antigravity_stream(r2_events) or is_agy:
            parts.extend(_format_antigravity_events(r2_events))
        else:
            for i, ev in enumerate(_select_codex_events(r2_events), start=1):
                parts.append(_format_codex_item_event(i, ev))
        parts.append("\n## Round 2 — Final Output (`reply_r2.md`)\n")
        parts.append(_fence(r2_reply, "markdown"))

    rendered = "\n".join(parts)
    out_path.write_text(rendered, encoding="utf-8", newline="\n")
    return {
        "track_id": track_id,
        "cli": cli_label,
        "thread_id": thread_id,
        "r1_raw_events": len(r1_events),
        "r2_raw_events": len(r2_events),
        "output_bytes": len(rendered.encode("utf-8")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_tid = sub.add_parser("thread-id")
    p_tid.add_argument("events_jsonl", type=Path)

    p_rep = sub.add_parser("extract-reply")
    p_rep.add_argument("events_jsonl", type=Path)
    p_rep.add_argument("out_md", type=Path)

    p_norm = sub.add_parser("normalize-utf8")
    p_norm.add_argument("paths", nargs="+", type=Path)

    p_ren = sub.add_parser("render")
    p_ren.add_argument("--track-dir", type=Path, required=True)
    p_ren.add_argument("--track-id", type=str, required=True)
    p_ren.add_argument("--out", type=Path, required=True)

    args = ap.parse_args()
    if args.cmd == "thread-id":
        print(extract_thread_id(args.events_jsonl))
    elif args.cmd == "extract-reply":
        txt = extract_antigravity_reply(args.events_jsonl)
        args.out_md.write_text(txt if txt.endswith("\n") else txt + "\n", encoding="utf-8", newline="\n")
        print(json.dumps({"out": str(args.out_md), "bytes": len(txt.encode("utf-8"))}, ensure_ascii=False))
    elif args.cmd == "normalize-utf8":
        res = [normalize_utf8_file(p) for p in args.paths]
        print(json.dumps(res, ensure_ascii=False))
    elif args.cmd == "render":
        info = render_full_conversation(args.track_dir, args.track_id, args.out)
        print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
