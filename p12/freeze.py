"""Prospective pre-experiment freeze & integrity verification for Phase 12 (`p12/`) G=16 closed-book study.

Locks exact SHA-256 hashes of all P12 source files, frozen P11 off-grid comparator checkpoints,
exact disjoint seed pools (3,584 unique formal seeds), and statistical protocol before any
`SEED_P12_FORMAL_ID` / `SEED_P12_FORMAL_OOD` seed is ever evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

from p12.contract import (
    OFFGRID_REGIME_TRUE_RESET,
    P11_BASELINE_COMMIT,
    P12_FOLD2_AUX,
    P12_FOLD2_TARGETS,
    P12_FOLD3_AUX,
    P12_FOLD3_TARGETS,
    PRIMARY_SNR_DB,
    SEED_P12_FORMAL_ID,
    SEED_P12_FORMAL_OOD,
    assert_p12_seed_disjoint,
)
from p12.primary_receiver import PRIMARY_ALL_P12_OOD_REGIMES

REPO_ROOT = Path(__file__).resolve().parent.parent
P12_DIR = REPO_ROOT / "p12"
RESULTS_DIR = P12_DIR / "results"
DEFAULT_MANIFEST_PATH = RESULTS_DIR / "freeze_manifest.json"
P11_CKPT_DIR = REPO_ROOT / "p11" / "ckpts" / "p10b_dev_ckpts"

# --- Frozen G=16 Formal Study Geometry ---
FORMAL_CLUSTERS: tuple[int, ...] = tuple(range(1, 17))
REPLAY_CLUSTERS: tuple[int, ...] = (1, 16)

PRIMARY_ID_REGIMES: tuple[str, ...] = ("continuous", "partial_shift", "full_reset")
PRIMARY_OOD_REGIMES: tuple[str, ...] = PRIMARY_ALL_P12_OOD_REGIMES  # 5 regimes (3 P11 + 2 true out-of-support)
PRIMARY_LINKS_PER_CLUSTER_ID: int = 32   # 3 * 16 * 32 = 1,536 links (24,576 core blocks)
PRIMARY_LINKS_PER_CLUSTER_OOD: int = 8   # 5 * 16 * 8  =   640 links (10,240 core blocks)

OFFGRID_ID_REGIMES: tuple[str, ...] = (
    "offgrid_continuous",
    "offgrid_partial_shift",
    "offgrid_midframe_shift",
    "offgrid_full_reset",
)
OFFGRID_OOD_REGIMES: tuple[str, ...] = (
    OFFGRID_REGIME_TRUE_RESET,
    "offgrid_ood_burst_and_shift",
    "offgrid_ood_five_ray_fast",
)
OFFGRID_LINKS_PER_CLUSTER_ID: int = 16   # 4 * 16 * 16 = 1,024 links (18,432 blocks, matching P11!)
OFFGRID_LINKS_PER_CLUSTER_OOD: int = 8   # 3 * 16 * 8  =   384 links (6,912 blocks)

PRIMARY_FORMAL_OFFSET = 100_000
OFFGRID_FORMAL_OFFSET = 500_000

FROZEN_CODE_FILES: tuple[str, ...] = (
    "p12/.gitattributes",
    "p12/__init__.py",
    "p12/contract.py",
    "p12/primary_folds.py",
    "p12/primary_receiver.py",
    "p12/offgrid_physics.py",
    "p12/offgrid_receiver.py",
    "p12/stage1_secondary_diag.py",
    "p12/freeze.py",
    "p12/audit.py",
    "p12/formal_study.py",
    "p12/report.py",
)

FROZEN_CKPT_FILES: tuple[str, ...] = (
    "p11/ckpts/p10b_dev_ckpts/em10_transport_attn_seed1001.pt",
    "p11/ckpts/p10b_dev_ckpts/em10_transport_attn_seed1002.pt",
    "p11/ckpts/p10b_dev_ckpts/em10_transport_attn_seed1003.pt",
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def planned_formal_seeds() -> dict[str, list[int]]:
    """Generate and strictly validate all 3,584 unique formal seeds across Primary & Secondary ID/OOD."""
    p_id = [
        int(SEED_P12_FORMAL_ID + PRIMARY_FORMAL_OFFSET + r_i * 50_000 + 1000 * c + l_i)
        for c in FORMAL_CLUSTERS
        for r_i in range(len(PRIMARY_ID_REGIMES))
        for l_i in range(PRIMARY_LINKS_PER_CLUSTER_ID)
    ]
    p_ood = [
        int(SEED_P12_FORMAL_OOD + PRIMARY_FORMAL_OFFSET + r_i * 50_000 + 1000 * c + l_i)
        for c in FORMAL_CLUSTERS
        for r_i in range(len(PRIMARY_OOD_REGIMES))
        for l_i in range(PRIMARY_LINKS_PER_CLUSTER_OOD)
    ]
    s_id = [
        int(SEED_P12_FORMAL_ID + OFFGRID_FORMAL_OFFSET + r_i * 50_000 + 1000 * c + l_i)
        for c in FORMAL_CLUSTERS
        for r_i in range(len(OFFGRID_ID_REGIMES))
        for l_i in range(OFFGRID_LINKS_PER_CLUSTER_ID)
    ]
    s_ood = [
        int(SEED_P12_FORMAL_OOD + OFFGRID_FORMAL_OFFSET + r_i * 50_000 + 1000 * c + l_i)
        for c in FORMAL_CLUSTERS
        for r_i in range(len(OFFGRID_OOD_REGIMES))
        for l_i in range(OFFGRID_LINKS_PER_CLUSTER_OOD)
    ]
    # Validate global disjointness across all historical/P11 ranges and uniqueness across all 4 pools
    assert_p12_seed_disjoint(p_id + s_id, expected_namespace="p12_formal_id")
    assert_p12_seed_disjoint(p_ood + s_ood, expected_namespace="p12_formal_ood")
    assert_p12_seed_disjoint(p_id + p_ood + s_id + s_ood)
    return {"primary_id": p_id, "primary_ood": p_ood, "secondary_id": s_id, "secondary_ood": s_ood}


def _git_output(args: Sequence[str]) -> str:
    proc = subprocess.run(["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True, check=True)
    return proc.stdout.strip()


def check_inherited_repo_clean() -> None:
    """Assert ZERO modified tracked files outside `p12/`."""
    out = _git_output(["status", "--porcelain"])
    bad: list[str] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        status = line[:2]
        path_str = line[3:].strip()
        if path_str.startswith(".agents/") or path_str.startswith("p12/"):
            continue
        if status != "??":
            bad.append(line)
    if bad:
        raise RuntimeError("Inherited tracked files modified outside p12/: " + "; ".join(bad))


def create_freeze_manifest(out_path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    check_inherited_repo_clean()
    seeds = planned_formal_seeds()
    file_hashes: dict[str, str] = {}
    for rel in FROZEN_CODE_FILES + FROZEN_CKPT_FILES:
        fp = REPO_ROOT / rel
        if not fp.is_file():
            raise FileNotFoundError(f"Missing frozen file: {rel}")
        file_hashes[rel] = sha256_file(fp)

    manifest: dict[str, Any] = {
        "phase": "P12",
        "baseline_commit": P11_BASELINE_COMMIT,
        "git_head_at_freeze": _git_output(["rev-parse", "HEAD"]),
        "snr_db": float(PRIMARY_SNR_DB),
        "clusters": list(FORMAL_CLUSTERS),
        "replay_clusters": list(REPLAY_CLUSTERS),
        "primary": {
            "id_regimes": list(PRIMARY_ID_REGIMES),
            "ood_regimes": list(PRIMARY_OOD_REGIMES),
            "links_per_cluster_id": PRIMARY_LINKS_PER_CLUSTER_ID,
            "links_per_cluster_ood": PRIMARY_LINKS_PER_CLUSTER_OOD,
            "fold2_targets": [list(x) for x in P12_FOLD2_TARGETS],
            "fold2_aux": [list(x) for x in P12_FOLD2_AUX],
            "fold3_targets": [list(x) for x in P12_FOLD3_TARGETS],
            "fold3_aux": [list(x) for x in P12_FOLD3_AUX],
        },
        "secondary": {
            "id_regimes": list(OFFGRID_ID_REGIMES),
            "ood_regimes": list(OFFGRID_OOD_REGIMES),
            "links_per_cluster_id": OFFGRID_LINKS_PER_CLUSTER_ID,
            "links_per_cluster_ood": OFFGRID_LINKS_PER_CLUSTER_OOD,
            "pilot_rows_a": [0, 9, 10, 19],
        },
        "seed_counts": {k: len(v) for k, v in seeds.items()},
        "total_unique_formal_seeds": sum(len(v) for v in seeds.values()),
        "file_sha256": file_hashes,
    }
    digest_payload = json.dumps(file_hashes, sort_keys=True).encode("utf-8")
    manifest["bundle_sha256"] = hashlib.sha256(digest_payload).hexdigest()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return manifest


def verify_freeze(manifest_path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Freeze manifest not found: {manifest_path}")
    check_inherited_repo_clean()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest["file_sha256"]
    mismatches: list[str] = []
    for rel, exp_hash in expected.items():
        fp = REPO_ROOT / rel
        if not fp.is_file():
            mismatches.append(f"MISSING: {rel}")
            continue
        actual = sha256_file(fp)
        if actual != exp_hash:
            mismatches.append(f"HASH_MISMATCH: {rel} (expected {exp_hash[:12]}, got {actual[:12]})")
    if mismatches:
        raise RuntimeError("Freeze verification FAILED:\n  " + "\n  ".join(mismatches))
    # Re-validate formal seed disjointness
    planned_formal_seeds()
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P12 Prospective Pre-Experiment Freeze Manager")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--verify", action="store_true")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_MANIFEST_PATH))
    args = parser.parse_args(argv)

    mpath = Path(args.manifest)
    if args.freeze:
        m = create_freeze_manifest(mpath)
        print(f"P12 FREEZE CREATED: {mpath} (bundle_sha256={m['bundle_sha256']}, seeds={m['total_unique_formal_seeds']})")
    else:
        m = verify_freeze(mpath)
        print(f"P12 FREEZE VERIFIED OK: {mpath} (bundle_sha256={m['bundle_sha256']}, files={len(m['file_sha256'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
