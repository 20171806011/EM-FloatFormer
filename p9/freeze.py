"""Immutable prospective freeze manifest creation and verification for Phase 9."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import scipy

from p9.contract import (
    P8C_BASELINE_COMMIT,
    P8_MANIFOLD_COMMIT,
    PRIMARY_SNR_DB,
    REGIMES_ID,
    REGIMES_OOD,
    REGIMES_SANITY,
    SEED_P9_DEV,
    SEED_P9_FORMAL_ID,
    SEED_P9_FORMAL_OOD,
    SEED_P9_TRAIN,
    SNR_GRID_DB,
)
from p9.independent_ref import check_no_hardcoded_n0


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RESULTS = ROOT / "results"
FREEZE_PATH = RESULTS / "freeze.json"

FORMAL_CLUSTERS = tuple(range(1, 17))   # G = 16 independent clusters (65,536 exact sign patterns)
LINKS_PER_CLUSTER_ID = 32               # 512 links = 8,192 common codewords per ID regime / SNR
LINKS_PER_CLUSTER_OOD = 16              # 256 links = 4,096 common codewords per OOD regime
REPLAY_CLUSTERS = (1, 16)
NEURAL_SEEDS = (901, 902, 903)
NEURAL_VARIANTS = ("transport_attn", "abs_no_transport", "local", "no_history")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment() -> dict:
    import numba
    import torch

    return dict(
        python=sys.version,
        numpy=np.__version__,
        scipy=scipy.__version__,
        numba=numba.__version__,
        torch=torch.__version__,
        platform=platform.platform(),
        machine=platform.machine(),
        cpu_count=os.cpu_count(),
        torch_threads=torch.get_num_threads(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        precision="float64/complex128",
    )


def design() -> dict:
    return dict(
        p8c_baseline_commit=P8C_BASELINE_COMMIT,
        p8_manifold_commit=P8_MANIFOLD_COMMIT,
        primary_snr_db=PRIMARY_SNR_DB,
        snr_grid_db=list(SNR_GRID_DB),
        regimes_id=list(REGIMES_ID),
        regimes_ood=list(REGIMES_OOD),
        regimes_sanity=list(REGIMES_SANITY),
        formal_clusters=list(FORMAL_CLUSTERS),
        links_per_cluster_id=LINKS_PER_CLUSTER_ID,
        links_per_cluster_ood=LINKS_PER_CLUSTER_OOD,
        replay_clusters=list(REPLAY_CLUSTERS),
        seed_bases=dict(
            dev=SEED_P9_DEV,
            train=SEED_P9_TRAIN,
            formal_id=SEED_P9_FORMAL_ID,
            formal_ood=SEED_P9_FORMAL_OOD,
        ),
        decoder_config=dict(list_size="large", bp_iters=20, turbo_pic=True, pic_scale=0.75),
        neural_config=dict(
            d_model=12,
            n_heads=4,
            n_layers=1,
            real_parameters=2426,
            seeds=list(NEURAL_SEEDS),
            variants=list(NEURAL_VARIANTS),
            ensemble_rule="equal_weight_mean_logit",
        ),
        holm_family=[
            "decoder: native_ldpc - strong_large (9 dB true_csi continuous)",
            "classical_gate: always_inherit - jeffreys_student_t (9 dB sparse 3-ID strong_large)",
            "attention_vs_classical: jeffreys_student_t - transport_attn (9 dB sparse 3-ID strong_large)",
            "attention_vs_local: local - transport_attn (9 dB sparse 3-ID strong_large)",
        ],
        noninferiority_margin=0.01,
    )


def source_hashes() -> dict:
    paths = sorted(p for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts)
    paths.append(ROOT / "PROTOCOL.md")
    for dev_json in ("baseline_check.json", "p9a_dev_attribution.json", "p9b_dev_gating.json", "p9c_dev_gate.json", "p9_audit_selftest.json"):
        p = RESULTS / dev_json
        if p.exists():
            paths.append(p)
    return {p.relative_to(REPO).as_posix(): sha256_file(p) for p in paths}


def checkpoint_hashes() -> dict:
    ckpt_dir = RESULTS / "p9c_dev_ckpts"
    hashes = {}
    for var in NEURAL_VARIANTS:
        for seed in NEURAL_SEEDS:
            p = ckpt_dir / f"{var}_seed{seed}.pt"
            if not p.exists():
                raise FileNotFoundError(f"Missing frozen neural gate checkpoint: {p}")
            hashes[p.relative_to(REPO).as_posix()] = sha256_file(p)
    return hashes


def assert_inherited_clean():
    changed = subprocess.check_output(["git", "diff", "--name-only", P8C_BASELINE_COMMIT], cwd=REPO, text=True)
    non_p9 = [line.strip() for line in changed.splitlines() if line.strip() and not line.strip().startswith("p9/")]
    if non_p9:
        raise ValueError(f"Inherited files modified since {P8C_BASELINE_COMMIT}: {non_p9}")
    prod_py = [p for p in sorted(ROOT.glob("*.py"))]
    n0_res = check_no_hardcoded_n0(prod_py)
    if not n0_res["clean"]:
        raise ValueError(f"Hardcoded N0 violation in production p9 modules: {n0_res['violations']}")


def create_freeze() -> str:
    if FREEZE_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing freeze manifest: {FREEZE_PATH}")
    assert_inherited_clean()
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        registration="within-session prospective freeze before formal P9 closed-book evaluation",
        design=design(),
        sources=source_hashes(),
        checkpoints=checkpoint_hashes(),
        environment=environment(),
    )
    RESULTS.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return sha256_file(FREEZE_PATH)


def verify_freeze() -> str:
    if not FREEZE_PATH.exists():
        raise FileNotFoundError(f"Missing freeze manifest: {FREEZE_PATH}")
    assert_inherited_clean()
    manifest = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if manifest["design"] != design():
        raise ValueError("Frozen P9 design does not match current code")
    if manifest["sources"] != source_hashes():
        changed = {
            k for k in set(manifest["sources"]) | set(source_hashes())
            if manifest["sources"].get(k) != source_hashes().get(k)
        }
        raise ValueError(f"Frozen P9 source/dev hashes differ: {sorted(changed)}")
    if manifest["checkpoints"] != checkpoint_hashes():
        raise ValueError("Frozen P9 neural checkpoint hashes differ")
    return sha256_file(FREEZE_PATH)


def main():
    import torch

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    digest = create_freeze()
    verified = verify_freeze()
    assert digest == verified
    print(f"Created and verified {FREEZE_PATH.relative_to(REPO).as_posix()} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
