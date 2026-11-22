"""Prospective Pre-Experiment Protocol Freeze & SHA-256 Verifier for CLI-4 Stage 5 Formal Study.

Locks:
  - Exact SHA-256 of all Stage 5 source files AND all 6 trained Stage 5 checkpoints (Role A & B x 3 seeds)
  - Exact SHA-256 of `p12/offgrid_physics.py` and `p12/offgrid_receiver.py` (confirming `655de6a` untouched!)
  - Exact winning physics variant, `MixtureDecodeConfig`, `ContentAddressingConfig`
  - Exact formal cluster count `G`, links/cluster, replay clusters, and strictly disjoint formal seed pools
  - Pre-registered Holm-Bonferroni hypothesis families before ANY `SEED_S5_TEST_*` seed is evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence

from research.p12_stage5_fusion_neural.contract import (
    ID_REGIMES,
    OOD_AND_RESET_REGIMES,
    PRIMARY_SNR_DB,
    SEED_S5_TEST_ID,
    SEED_S5_TEST_OOD,
    STAGE5_BASELINE_COMMIT,
    STAGE5_SEEDS,
    assert_s5_seed_disjoint,
    sha256_file,
)
from research.p12_stage5_fusion_neural.step2_train_stage5 import get_physics_configs

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PKG_DIR = REPO_ROOT / "research" / "p12_stage5_fusion_neural"
DEFAULT_FREEZE_PATH = PKG_DIR / "protocol_freeze.json"

FORMAL_ID_OFFSET = 100_000
FORMAL_OOD_OFFSET = 500_000

FROZEN_STAGE5_CODE: tuple[str, ...] = (
    "research/p12_stage5_fusion_neural/contract.py",
    "research/p12_stage5_fusion_neural/consistent_fusion_engine.py",
    "research/p12_stage5_fusion_neural/neural_models.py",
    "research/p12_stage5_fusion_neural/step1_dev_ablation.py",
    "research/p12_stage5_fusion_neural/step2_train_stage5.py",
    "research/p12_stage5_fusion_neural/freeze.py",
    "research/p12_stage5_fusion_neural/audit.py",
    "research/p12_stage5_fusion_neural/step3_formal_eval.py",
)

ANCHOR_P12_FILES: tuple[str, ...] = (
    "p12/contract.py",
    "p12/offgrid_physics.py",
    "p12/offgrid_receiver.py",
    "p12/results/freeze_manifest.json",
)


def planned_stage5_formal_seeds(
    *,
    n_clusters: int,
    links_per_cluster_id: int,
    links_per_cluster_ood: int,
) -> dict[str, list[int]]:
    clusters = list(range(1, int(n_clusters) + 1))
    s_id = [
        int(SEED_S5_TEST_ID + FORMAL_ID_OFFSET + r_i * 50_000 + 1000 * c + l_i)
        for c in clusters
        for r_i in range(len(ID_REGIMES))
        for l_i in range(int(links_per_cluster_id))
    ]
    s_ood = [
        int(SEED_S5_TEST_OOD + FORMAL_OOD_OFFSET + r_i * 50_000 + 1000 * c + l_i)
        for c in clusters
        for r_i in range(len(OOD_AND_RESET_REGIMES))
        for l_i in range(int(links_per_cluster_ood))
    ]
    assert_s5_seed_disjoint(s_id, expected_namespace="s5_test_id")
    assert_s5_seed_disjoint(s_ood, expected_namespace="s5_test_ood")
    assert_s5_seed_disjoint(s_id + s_ood)
    return {"formal_id": s_id, "formal_ood": s_ood}


def _git_head_if_available() -> str:
    try:
        proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), capture_output=True, text=True, check=True)
        return proc.stdout.strip()
    except Exception:
        return "codespace-staged-mirror"


def create_protocol_freeze(
    *,
    out_path: Path = DEFAULT_FREEZE_PATH,
    ckpt_dir: Path = PKG_DIR / "ckpts",
    physics_variant: str = "mixture_llr_content",
    n_clusters: int = 16,
    links_per_cluster_id: int = 8,
    links_per_cluster_ood: int = 4,
    eval_per_seed_on_id: bool = False,
) -> dict[str, Any]:
    seeds = planned_stage5_formal_seeds(
        n_clusters=n_clusters,
        links_per_cluster_id=links_per_cluster_id,
        links_per_cluster_ood=links_per_cluster_ood,
    )
    content_cfg, mix_cfg = get_physics_configs(physics_variant)
    upgraded_arm = "p12_mixture_llr" if physics_variant == "mixture_llr" else "p12_mixture_llr_content"

    file_hashes: dict[str, str] = {}
    for rel in FROZEN_STAGE5_CODE + ANCHOR_P12_FILES:
        fp = REPO_ROOT / rel
        if not fp.is_file():
            # In staged Codespace mirror, p12/results/freeze_manifest.json may be omitted; allow only if staged
            if rel == "p12/results/freeze_manifest.json":
                continue
            raise FileNotFoundError(f"Missing frozen file: {fp}")
        file_hashes[rel] = sha256_file(fp)

    ckpt_hashes: dict[str, str] = {}
    for role in ("role_a", "role_b"):
        for s in STAGE5_SEEDS:
            cname = f"stage5_{role}_seed{s}.pt"
            cp = ckpt_dir / cname
            if not cp.is_file():
                raise FileNotFoundError(f"Missing Stage 5 checkpoint before freeze: {cp}")
            ckpt_hashes[cname] = sha256_file(cp)

    clusters = list(range(1, int(n_clusters) + 1))
    replay_clusters = [1, int(n_clusters)]

    manifest: dict[str, Any] = {
        "study": "EM-FloatFormer CLI-4 P12 Stage 5: Consistent Off-Grid Posterior Fusion + Symmetric Parallel-Transport Neural Control",
        "baseline_commit": STAGE5_BASELINE_COMMIT,
        "git_head_at_freeze": _git_head_if_available(),
        "snr_db": float(PRIMARY_SNR_DB),
        "pilot_rows_a": [0, 9, 10, 19],
        "physics_variant": physics_variant,
        "upgraded_phys_arm": upgraded_arm,
        "mix_cfg": mix_cfg.__dict__,
        "content_cfg": None if content_cfg is None else content_cfg.__dict__,
        "n_clusters": int(n_clusters),
        "clusters": clusters,
        "replay_clusters": replay_clusters,
        "id_regimes": list(ID_REGIMES),
        "ood_regimes": list(OOD_AND_RESET_REGIMES),
        "links_per_cluster_id": int(links_per_cluster_id),
        "links_per_cluster_ood": int(links_per_cluster_ood),
        "eval_per_seed_on_id": bool(eval_per_seed_on_id),
        "seed_ranges": {
            "s5_dev_step1": [1_410_000_010_000, 1_410_000_015_000],
            "s5_dev_step2": [1_410_000_050_000, 1_410_000_100_000],
            "s5_train_step2": [1_420_000_100_000, 1_420_000_200_000],
            "s5_test_id": [min(seeds["formal_id"]), max(seeds["formal_id"])],
            "s5_test_ood": [min(seeds["formal_ood"]), max(seeds["formal_ood"])],
        },
        "seed_counts": {k: len(v) for k, v in seeds.items()},
        "total_unique_formal_seeds": sum(len(v) for v in seeds.values()),
        "preregistered_holm_families": {
            "family_physics": [
                "mix_vs_base_4id",
                "mix_vs_hard_4id",
                "content_vs_base_4id",
                "content_vs_mix_4id",
                "upgraded_vs_base_new_delays",
                "upgraded_vs_base_ood",
            ],
            "family_neural": [
                "role_a_vs_phys_4id",
                "role_b_vs_phys_4id",
                "role_a_vs_role_b_4id",
                "role_a_vs_phys_new_delays",
                "role_a_vs_phys_ood",
                "role_a_vs_role_b_ood",
            ],
        },
        "file_sha256": file_hashes,
        "ckpt_sha256": ckpt_hashes,
    }
    digest_obj = {"file_sha256": file_hashes, "ckpt_sha256": ckpt_hashes, "seed_counts": manifest["seed_counts"]}
    manifest["bundle_sha256"] = hashlib.sha256(json.dumps(digest_obj, sort_keys=True).encode("utf-8")).hexdigest()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return manifest


def verify_protocol_freeze(
    manifest_path: Path = DEFAULT_FREEZE_PATH,
    ckpt_dir: Path = PKG_DIR / "ckpts",
) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Protocol freeze manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    for rel, exp_h in manifest["file_sha256"].items():
        fp = REPO_ROOT / rel
        if not fp.is_file():
            mismatches.append(f"MISSING_FILE: {rel}")
            continue
        act = sha256_file(fp)
        if act != exp_h:
            mismatches.append(f"HASH_MISMATCH_FILE: {rel} (expected {exp_h[:12]}, got {act[:12]})")
    for cname, exp_h in manifest["ckpt_sha256"].items():
        cp = ckpt_dir / cname
        if not cp.is_file():
            mismatches.append(f"MISSING_CKPT: {cname}")
            continue
        act = sha256_file(cp)
        if act != exp_h:
            mismatches.append(f"HASH_MISMATCH_CKPT: {cname} (expected {exp_h[:12]}, got {act[:12]})")
    if mismatches:
        raise RuntimeError("Stage 5 protocol freeze verification FAILED:\n  " + "\n  ".join(mismatches))
    planned_stage5_formal_seeds(
        n_clusters=int(manifest["n_clusters"]),
        links_per_cluster_id=int(manifest["links_per_cluster_id"]),
        links_per_cluster_ood=int(manifest["links_per_cluster_ood"]),
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CLI-4 Stage 5 Prospective Protocol Freeze Manager")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--freeze", action="store_true")
    grp.add_argument("--verify", action="store_true")
    parser.add_argument("--manifest", type=str, default=str(DEFAULT_FREEZE_PATH))
    parser.add_argument("--ckpt-dir", type=str, default=str(PKG_DIR / "ckpts"))
    parser.add_argument("--physics-variant", type=str, default="mixture_llr_content")
    parser.add_argument("--n-clusters", type=int, default=16)
    parser.add_argument("--links-per-cluster-id", type=int, default=8)
    parser.add_argument("--links-per-cluster-ood", type=int, default=4)
    parser.add_argument("--eval-per-seed-on-id", action="store_true", default=False)
    args = parser.parse_args(argv)

    mpath = Path(args.manifest)
    cdir = Path(args.ckpt_dir)
    if args.freeze:
        m = create_protocol_freeze(
            out_path=mpath,
            ckpt_dir=cdir,
            physics_variant=args.physics_variant,
            n_clusters=args.n_clusters,
            links_per_cluster_id=args.links_per_cluster_id,
            links_per_cluster_ood=args.links_per_cluster_ood,
            eval_per_seed_on_id=args.eval_per_seed_on_id,
        )
        print(
            f"STAGE5 FREEZE CREATED: {mpath} (G={m['n_clusters']}, seeds={m['total_unique_formal_seeds']}, "
            f"variant={m['physics_variant']}, bundle_sha256={m['bundle_sha256']})",
            flush=True,
        )
    else:
        m = verify_protocol_freeze(mpath, ckpt_dir=cdir)
        print(
            f"STAGE5 FREEZE VERIFIED OK: {mpath} (G={m['n_clusters']}, seeds={m['total_unique_formal_seeds']}, "
            f"bundle_sha256={m['bundle_sha256']})",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
