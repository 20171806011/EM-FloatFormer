"""Immutable prospective pre-experiment freeze manifest generator & verifier for Phase 11."""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Optional

import numpy as np
import scipy

from p11.contract import (
    OFFGRID_REGIMES_ID,
    OFFGRID_REGIMES_OOD,
    P10_BASELINE_COMMIT,
    P8C_BASELINE_COMMIT,
    P8_MANIFOLD_COMMIT,
    P9_BASELINE_COMMIT,
    PRIMARY_SNR_DB,
    REGIMES_ID,
    REGIMES_OOD,
    REGIMES_SANITY,
    SEED_P11_DEV,
    SEED_P11_E2E,
    SEED_P11_FORMAL_ID,
    SEED_P11_FORMAL_OOD,
    SEED_P11_TRAIN,
    SNR_GRID_DB,
)
from p11.independent_ref import check_p11_no_hardcoded_n0


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RESULTS_DIR = ROOT / "results"
DEFAULT_MANIFEST_PATH = RESULTS_DIR / "freeze_manifest.json"
CKPT_DIR = ROOT / "ckpts" / "p10b_dev_ckpts"

MAIN_BASELINE_COMMIT = "63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1"
OFFGRID_VARIANTS = (
    "em10_transport_attn",
    "em10_abs_no_transport",
    "em10_local",
    "em10_no_history",
)
OFFGRID_SEEDS = (1001, 1002, 1003)

FORMAL_CLUSTERS = tuple(range(1, 17))
LINKS_PER_CLUSTER_ID = 32
LINKS_PER_CLUSTER_OOD = 16
OFFGRID_LINKS_PER_CLUSTER_ID = 16
OFFGRID_LINKS_PER_CLUSTER_OOD = 16
REPLAY_CLUSTERS = (1, 16)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_git(args: list[str], *, strip: bool = True) -> str:
    out = subprocess.check_output(["git", *args], cwd=REPO, text=True, encoding="utf-8")
    return out.strip() if strip else out.rstrip("\r\n")


def _rev_parse_optional(ref: str) -> Optional[str]:
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", ref],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if proc.returncode != 0:
        return None
    out = proc.stdout.strip()
    return out or None


def _all_p11_py_files() -> list[Path]:
    excluded = {"__pycache__", ".pytest_cache"}
    return sorted(
        p for p in ROOT.rglob("*.py")
        if not any(part in excluded for part in p.parts)
    )


def check_ast_no_inference_mode(paths: list[Path]) -> list[dict[str, Any]]:
    """Return any AST occurrences of `inference_mode` across the provided Python files."""
    target = "inference_" + "mode"
    hits: list[dict[str, Any]] = []
    for p in paths:
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == target:
                hits.append({"file": p.relative_to(REPO).as_posix(), "line": int(node.lineno), "kind": "Name"})
            elif isinstance(node, ast.Attribute) and node.attr == target:
                hits.append({"file": p.relative_to(REPO).as_posix(), "line": int(node.lineno), "kind": "Attribute"})
    return hits


def environment() -> dict[str, Any]:
    import numba
    import torch

    return {
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "torch_threads": int(torch.get_num_threads()),
        "deterministic": bool(torch.are_deterministic_algorithms_enabled()),
        "precision": "float64/complex128",
    }


def design() -> dict[str, Any]:
    return {
        "p10_baseline_commit": P10_BASELINE_COMMIT,
        "p9_baseline_commit": P9_BASELINE_COMMIT,
        "p8c_baseline_commit": P8C_BASELINE_COMMIT,
        "p8_manifold_commit": P8_MANIFOLD_COMMIT,
        "main_baseline_commit": MAIN_BASELINE_COMMIT,
        "primary_snr_db": PRIMARY_SNR_DB,
        "snr_grid_db": list(SNR_GRID_DB),
        "regimes_id": list(REGIMES_ID),
        "regimes_ood": list(REGIMES_OOD),
        "regimes_sanity": list(REGIMES_SANITY),
        "offgrid_regimes_id": list(OFFGRID_REGIMES_ID),
        "offgrid_regimes_ood": list(OFFGRID_REGIMES_OOD),
        "formal_clusters": list(FORMAL_CLUSTERS),
        "links_per_cluster_id": LINKS_PER_CLUSTER_ID,
        "links_per_cluster_ood": LINKS_PER_CLUSTER_OOD,
        "offgrid_links_per_cluster_id": OFFGRID_LINKS_PER_CLUSTER_ID,
        "offgrid_links_per_cluster_ood": OFFGRID_LINKS_PER_CLUSTER_OOD,
        "replay_clusters": list(REPLAY_CLUSTERS),
        "seed_bases": {
            "dev": SEED_P11_DEV,
            "train": SEED_P11_TRAIN,
            "formal_id": SEED_P11_FORMAL_ID,
            "formal_ood": SEED_P11_FORMAL_OOD,
            "e2e": SEED_P11_E2E,
        },
        "decoder_config": {
            "list_size": "large",
            "bp_iters": 20,
            "turbo_pic": True,
            "pic_scale": 0.75,
        },
        "offgrid_config": {
            "variants": list(OFFGRID_VARIANTS),
            "seeds": list(OFFGRID_SEEDS),
            "d_model": 12,
            "d_scalar": 12,
            "n_heads": 4,
            "n_layers": 1,
            "d_ff": 12,
            "real_parameters": 3153,
            "ensemble_rule": "equal_weight_mean_logit",
        },
        "holm_families": {
            "primary": ["chained_vs_p9_jt", "chained_vs_payload_only"],
            "secondary": [
                "transport_attn_vs_abs_no_transport",
                "transport_attn_vs_local",
                "transport_attn_vs_no_history",
            ],
        },
        "noninferiority_margin": 0.01,
    }


def source_hashes() -> dict[str, str]:
    paths = list(_all_p11_py_files())
    math_doc = ROOT / "math_verification.md"
    if not math_doc.exists():
        raise FileNotFoundError(f"Missing required design doc: {math_doc}")
    paths.append(math_doc)
    gitattr = ROOT / ".gitattributes"
    if gitattr.exists():
        paths.append(gitattr)
    dev_json = RESULTS_DIR / "p11_dev.json"
    if dev_json.exists():
        paths.append(dev_json)
    return {p.relative_to(REPO).as_posix(): sha256_file(p) for p in sorted(set(paths))}


def checkpoint_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for var in OFFGRID_VARIANTS:
        for seed in OFFGRID_SEEDS:
            p = CKPT_DIR / f"{var}_seed{seed}.pt"
            if not p.exists():
                raise FileNotFoundError(f"Missing frozen off-grid checkpoint: {p}")
            hashes[p.relative_to(REPO).as_posix()] = sha256_file(p)
    return hashes


def _parse_porcelain_entries(status_output: str) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line in status_output.splitlines():
        if len(line) < 4 or line[2] != " ":
            continue
        xy = line[:2]
        raw_path = line[3:]
        if " -> " in raw_path:
            raw_path = raw_path.split(" -> ", 1)[1]
        if raw_path.startswith('"') and raw_path.endswith('"'):
            raw_path = raw_path[1:-1]
        entries.append((xy, raw_path.replace("\\", "/")))
    return entries


def _parse_porcelain_paths(status_output: str) -> list[str]:
    return [path for _, path in _parse_porcelain_entries(status_output)]


def assert_repo_guardrails(
    *,
    allow_dirty_agents: bool = False,
    manifest: Optional[Mapping[str, Any]] = None,
    manifest_path: Optional[Path] = None,
    allow_untracked_manifest: bool = False,
) -> dict[str, Any]:
    main_head = _rev_parse_optional("refs/heads/main")
    if main_head is None:
        main_head = _rev_parse_optional("refs/remotes/origin/main")
    if main_head is not None:
        if main_head != MAIN_BASELINE_COMMIT:
            raise ValueError(f"Branch `main` moved: expected {MAIN_BASELINE_COMMIT}, got {main_head}")
    else:
        main_head = MAIN_BASELINE_COMMIT

    anc_proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", MAIN_BASELINE_COMMIT, "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if anc_proc.returncode != 0:
        raise ValueError(f"HEAD is not a descendant of main baseline {MAIN_BASELINE_COMMIT}")

    committed_non_p11 = [
        line.strip()
        for line in _run_git(["diff", "--name-only", f"{P10_BASELINE_COMMIT}..HEAD", "--", ":!p11"]).splitlines()
        if line.strip()
    ]
    if committed_non_p11:
        raise ValueError(
            f"Non-p11 files modified in commits since {P10_BASELINE_COMMIT[:7]}: {committed_non_p11}"
        )

    current_branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    if current_branch == "main":
        raise ValueError("Refusing to freeze/verify on `main`; use the P11 research branch.")

    unstaged_non_p11 = [
        line.strip()
        for line in _run_git(["diff", "--name-only", P10_BASELINE_COMMIT, "--", ":!p11"]).splitlines()
        if line.strip()
    ]
    staged_non_p11 = [
        line.strip()
        for line in _run_git(["diff", "--cached", "--name-only", P10_BASELINE_COMMIT, "--", ":!p11"]).splitlines()
        if line.strip()
    ]
    if unstaged_non_p11 or staged_non_p11:
        raise ValueError(
            f"Non-p11 files modified vs {P10_BASELINE_COMMIT[:7]}: "
            f"unstaged={unstaged_non_p11}, staged={staged_non_p11}"
        )

    status_out = _run_git(
        ["-c", "core.quotepath=false", "status", "--porcelain", "--untracked-files=all"],
        strip=False,
    )
    dirty_entries = _parse_porcelain_entries(status_out)
    manifest_rel = (
        manifest_path.resolve().relative_to(REPO.resolve()).as_posix()
        if manifest_path is not None and manifest_path.resolve().is_relative_to(REPO.resolve())
        else None
    )

    allowed_post_freeze = {"p11/P11_REPORT.md", "p11/CLI_SYNC_SUMMARY.md"}
    for xy, rel in dirty_entries:
        if rel == ".agents" or rel.startswith(".agents/"):
            if not allow_dirty_agents:
                raise ValueError(
                    f"Dirty `.agents/` path {rel!r} detected without --allow-dirty-agents."
                )
            continue
        if not rel.startswith("p11/"):
            raise ValueError(f"Dirty path outside p11/: {rel!r}")
        if manifest is not None:
            if manifest_rel is not None and rel == manifest_rel:
                if allow_untracked_manifest and xy == "??":
                    continue
                raise ValueError(f"Freeze manifest modified or untracked on disk ({xy}): {rel!r}")
            if rel.endswith(".py") or rel.endswith(".pt") or rel in ("p11/math_verification.md", "p11/.gitattributes"):
                frozen_keys = set(manifest["sources"].keys()) | set(manifest["checkpoints"].keys())
                if rel not in frozen_keys:
                    raise ValueError(f"Unfrozen/new source or checkpoint inside p11/: {rel!r}")
            elif rel.startswith("p11/results/"):
                continue
            elif rel in allowed_post_freeze:
                continue
            else:
                raise ValueError(f"Unexpected modified/untracked file in p11/ after freeze: {rel!r}")

    py_files = _all_p11_py_files()
    n0_res = check_p11_no_hardcoded_n0(py_files, strict=True)
    if not n0_res["clean"]:
        raise ValueError(f"Hardcoded N0 violation in p11: {n0_res['violations']}")

    inf_hits = check_ast_no_inference_mode(py_files)
    if inf_hits:
        raise ValueError(f"Forbidden inference_mode AST usage in p11: {inf_hits}")

    return {
        "head": _run_git(["rev-parse", "HEAD"]),
        "branch": current_branch,
        "main_head": main_head,
        "p10_baseline_commit": P10_BASELINE_COMMIT,
        "p9_baseline_commit": P9_BASELINE_COMMIT,
        "p8c_baseline_commit": P8C_BASELINE_COMMIT,
        "p8_manifold_commit": P8_MANIFOLD_COMMIT,
        "non_p11_diff_vs_9e9d93b": [],
        "allow_dirty_agents": bool(allow_dirty_agents),
    }


def create_freeze(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    *,
    force: bool = False,
    allow_dirty_agents: bool = False,
) -> str:
    target = Path(manifest_path)
    if target.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing freeze manifest: {target}")

    git_meta = assert_repo_guardrails(allow_dirty_agents=allow_dirty_agents)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "registration": "within-session prospective freeze before formal P11 closed-book evaluation",
        "git": git_meta,
        "design": design(),
        "sources": source_hashes(),
        "checkpoints": checkpoint_hashes(),
        "environment": environment(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return sha256_file(target)


def verify_freeze(
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    *,
    allow_dirty_agents: bool = False,
    allow_untracked_manifest: bool = False,
) -> str:
    target = Path(manifest_path)
    if not target.exists():
        raise FileNotFoundError(f"Missing freeze manifest: {target}")

    manifest = json.loads(target.read_text(encoding="utf-8"))
    git_now = assert_repo_guardrails(
        allow_dirty_agents=allow_dirty_agents,
        manifest=manifest,
        manifest_path=target,
        allow_untracked_manifest=allow_untracked_manifest,
    )

    if manifest.get("git", {}).get("branch") != git_now["branch"]:
        raise ValueError(
            f"Frozen branch mismatch: manifest={manifest.get('git', {}).get('branch')!r} vs now={git_now['branch']!r}"
        )
    if manifest.get("design") != design():
        raise ValueError("Frozen P11 design does not match current code")

    cur_sources = source_hashes()
    manifest_sources = manifest.get("sources", {})
    selftest_key = "p11/results/p11_audit_selftest.json"
    if selftest_key in manifest_sources:
        st_path = REPO / selftest_key
        if st_path.exists():
            cur_sources[selftest_key] = sha256_file(st_path)
    if manifest_sources != cur_sources:
        changed = {
            k for k in set(manifest_sources) | set(cur_sources)
            if manifest_sources.get(k) != cur_sources.get(k)
        }
        raise ValueError(f"Frozen P11 source/doc hashes differ: {sorted(changed)}")

    cur_ckpts = checkpoint_hashes()
    if manifest.get("checkpoints") != cur_ckpts:
        changed = {
            k for k in set(manifest.get("checkpoints", {})) | set(cur_ckpts)
            if manifest.get("checkpoints", {}).get(k) != cur_ckpts.get(k)
        }
        raise ValueError(f"Frozen P11 checkpoint hashes differ: {sorted(changed)}")

    return sha256_file(target)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 11 pre-experiment freeze manifest tool.")
    mode_grp = parser.add_mutually_exclusive_group(required=True)
    mode_grp.add_argument("--write", action="store_true", help="Create freeze manifest JSON.")
    mode_grp.add_argument("--verify", action="store_true", help="Verify repository against freeze manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Path to freeze manifest JSON (default: p11/results/freeze_manifest.json).",
    )
    parser.add_argument("--force", action="store_true", help="Allow overwriting an existing manifest with --write.")
    parser.add_argument(
        "--allow-dirty-agents",
        action="store_true",
        help="Allow untracked/modified files under `.agents/`.",
    )
    args = parser.parse_args(argv)

    from p11.contract import enforce_determinism
    enforce_determinism(require_env=False)

    if args.write:
        digest = create_freeze(
            manifest_path=args.manifest,
            force=args.force,
            allow_dirty_agents=args.allow_dirty_agents,
        )
        verified = verify_freeze(
            manifest_path=args.manifest,
            allow_dirty_agents=args.allow_dirty_agents,
            allow_untracked_manifest=True,
        )
        assert digest == verified
        print(json.dumps({"action": "write_and_verify", "manifest": str(args.manifest), "sha256": digest}, indent=2))
    else:
        digest = verify_freeze(
            manifest_path=args.manifest,
            allow_dirty_agents=args.allow_dirty_agents,
        )
        print(json.dumps({"action": "verify", "manifest": str(args.manifest), "sha256": digest}, indent=2))


__all__ = [
    "CKPT_DIR",
    "DEFAULT_MANIFEST_PATH",
    "FORMAL_CLUSTERS",
    "LINKS_PER_CLUSTER_ID",
    "LINKS_PER_CLUSTER_OOD",
    "MAIN_BASELINE_COMMIT",
    "OFFGRID_LINKS_PER_CLUSTER_ID",
    "OFFGRID_LINKS_PER_CLUSTER_OOD",
    "OFFGRID_SEEDS",
    "OFFGRID_VARIANTS",
    "REPLAY_CLUSTERS",
    "RESULTS_DIR",
    "assert_repo_guardrails",
    "check_ast_no_inference_mode",
    "checkpoint_hashes",
    "create_freeze",
    "design",
    "environment",
    "sha256_file",
    "source_hashes",
    "verify_freeze",
]


if __name__ == "__main__":
    main()
