"""Immutable prospective freeze manifest creation and verification for Phase 8 Manifold."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch

from p6 import experiment as ex
from p8_manifold.offgrid_sim import OFFGRID_CONDITIONS


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RESULTS = ROOT / "results"
FREEZE_PATH = RESULTS / "freeze.json"

BASE_COMMIT = "4dab1ac9ee15b4f1a7f78ae05486c55a2dc3bd72"
SEED_BASE = 802611190000
FORMAL_PAIRS = tuple(range(1, 11))
SNR_DB = 15.0
EVAL_BATCHES = 4
BATCH_SIZE = 8

LEARNED_ARMS = (
    ("EM8-Transport", "transport"),
    ("EM8-AbsNoTransport", "abs_no_transport"),
    ("EM8-Local", "local"),
    ("EM8-SingleHyp", "single_transport"),
)
FORMAL_ARMS = (
    "EM8-Transport",
    "EM8-AbsNoTransport",
    "EM8-Local",
    "EM8-SingleHyp",
    "EM8-Fixed",
    "OffGrid-2D-LMMSE",
)
AUDIT_CONDITIONS = ("O1_roving_island", "O2_blind_burst_erasure", "O3_holdout_sequential")


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def environment():
    import numpy
    import scipy
    import numba

    return dict(
        python=sys.version,
        torch=torch.__version__,
        numpy=numpy.__version__,
        scipy=scipy.__version__,
        numba=numba.__version__,
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        cpu_count=os.cpu_count(),
        threads=torch.get_num_threads(),
        interop_threads=torch.get_num_interop_threads(),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        precision="float64/complex128",
        device="cpu",
    )


def design():
    return dict(
        base_commit=BASE_COMMIT,
        snr_db=SNR_DB,
        seed_base=SEED_BASE,
        formal_pairs=list(FORMAL_PAIRS),
        model_config=dict(d_model=8, n_layers=1, n_heads=2, em_steps=2, n_hyp=4, gamma_init=10.0),
        train_recipe=dict(
            steps=60,
            lr=0.0035,
            min_lr_ratio=0.1,
            warmup=5,
            lambda_hold=0.08,
            lambda_nmse=0.01,
            lambda_ici=0.05,
            lambda_cal=0.02,
            anchor_l2=1e-4,
            probe_weight=0.10,
            batch_size=BATCH_SIZE,
            grad_clip=1.0,
            conditions=list(OFFGRID_CONDITIONS),
        ),
        eval_batches=EVAL_BATCHES,
        eval_batch_size=BATCH_SIZE,
        conditions=list(OFFGRID_CONDITIONS),
        audit_conditions=list(AUDIT_CONDITIONS),
        arms=list(FORMAL_ARMS),
        init_seeds={str(i): SEED_BASE + 900000 + i for i in FORMAL_PAIRS},
        train_seed_bases={str(i): SEED_BASE + 100000 + 10000 * i for i in FORMAL_PAIRS},
        eval_seeds={
            str(i): {
                cond: [SEED_BASE + 500000 + 10000 * i + 100 * ci + b for b in range(EVAL_BATCHES)]
                for ci, cond in enumerate(OFFGRID_CONDITIONS)
            }
            for i in FORMAL_PAIRS
        },
    )


def source_hashes():
    paths = sorted(p for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts)
    paths.append(ROOT / "PROTOCOL.md")
    gate = RESULTS / "dev_gate.json"
    if gate.exists():
        paths.append(gate)
    return {p.relative_to(REPO).as_posix(): sha256_file(p) for p in paths}


def assert_inherited_clean():
    changed = subprocess.check_output(["git", "diff", "--name-only", BASE_COMMIT], cwd=REPO, text=True)
    if changed.strip():
        raise ValueError(f"Inherited repository files modified since {BASE_COMMIT}: {changed}")


def create_freeze():
    if FREEZE_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing freeze manifest: {FREEZE_PATH}")
    gate_path = RESULTS / "dev_gate.json"
    if not gate_path.exists():
        raise FileNotFoundError("Run p8_manifold.dev_study before creating prospective freeze")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("gate_passed"):
        raise ValueError("Cannot freeze before dev_gate.json gate_passed=True")
    assert_inherited_clean()

    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        registration="within-session prospective freeze before formal scoring; not external public preregistration",
        design=design(),
        sources=source_hashes(),
        environment=environment(),
    )
    ex.save(FREEZE_PATH, manifest)
    return sha256_file(FREEZE_PATH)


def verify_freeze():
    if not FREEZE_PATH.exists():
        raise FileNotFoundError(f"Missing freeze manifest: {FREEZE_PATH}")
    assert_inherited_clean()
    manifest = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if manifest["design"] != design():
        raise ValueError("Frozen formal design does not match current code")
    if manifest["sources"] != source_hashes():
        changed = {
            k for k in set(manifest["sources"]) | set(source_hashes())
            if manifest["sources"].get(k) != source_hashes().get(k)
        }
        raise ValueError(f"Frozen source/dev-gate hashes differ: {sorted(changed)}")
    return sha256_file(FREEZE_PATH)


def main():
    from p5.study import setup

    setup()
    digest = create_freeze()
    verified = verify_freeze()
    assert digest == verified
    print(f"Created and verified {FREEZE_PATH.relative_to(REPO).as_posix()} sha256={digest}", flush=True)


if __name__ == "__main__":
    main()
