"""Immutable prospective freeze manifest creation and verification for Phase 7."""
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
from p6_followup.common import load_model
from p7.conditions import CONDITIONS, PILOT_AUDIT_CONDITIONS


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RESULTS = ROOT / "results"
FREEZE_PATH = RESULTS / "freeze.json"

BASE_COMMIT = "a4e6684830f68e6dfe715470dafb4ff95972e508"
P6_MAIN_COMMIT = "63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1"
SEED_BASE = 702611190000
FORMAL_PAIRS = tuple(range(1, 11))
FORMAL_ARMS = (
    "EM7-Axial",
    "EM7-Local",
    "EM7-Fixed",
    "EM7-SingleHyp-Axial",
    "P6-Aware-Consistent",
    "Waveform-DD",
)


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def environment():
    """Windows/Linux portable deterministic CPU environment descriptor."""
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
        p6_main_commit=P6_MAIN_COMMIT,
        regime=2,
        snr_db=9,
        seed_base=SEED_BASE,
        formal_pairs=list(FORMAL_PAIRS),
        train_recipe=dict(
            steps=40,
            lr=0.004,
            min_lr_ratio=0.1,
            warmup=4,
            lambda_hold=0.05,
            lambda_nmse=0.005,
            anchor_l2=5e-5,
            batch_size=8,
            grad_clip=1.0,
            conditions=["C0_intact", "C1_row9_erased", "C2_row0_erased", "C3_scattered_half", "C4a_roving_window"],
        ),
        eval_batches=4,
        eval_batch_size=8,
        conditions=list(CONDITIONS),
        pilot_audit_conditions=list(PILOT_AUDIT_CONDITIONS),
        arms=list(FORMAL_ARMS),
        p6_pair_offset=5,
        init_seeds={str(i): SEED_BASE + 900000 + i for i in FORMAL_PAIRS},
        mask_seed_bases={str(i): SEED_BASE + 200000 + 1000 * i for i in FORMAL_PAIRS},
        eval_seeds={str(i): [SEED_BASE + 500000 + 1000 * i + b for b in range(4)] for i in FORMAL_PAIRS},
        cond_seeds={str(i): [SEED_BASE + 700000 + 1000 * i + b for b in range(4)] for i in FORMAL_PAIRS},
    )


def source_hashes():
    paths = sorted(p for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts)
    paths.append(ROOT / "PROTOCOL.md")
    gate = RESULTS / "dev_gate.json"
    if gate.exists():
        paths.append(gate)
    return {p.relative_to(REPO).as_posix(): sha256_file(p) for p in paths}


def p6_comparator_hashes():
    receipts = {}
    for pair_idx in FORMAL_PAIRS:
        p6_pair = pair_idx + 5
        _, receipt = load_model(p6_pair, "parity", development=False)
        receipts[receipt["path"]] = receipt["sha256"]
    return receipts


def assert_inherited_clean():
    changed = subprocess.check_output(["git", "diff", "--name-only", BASE_COMMIT], cwd=REPO, text=True)
    if changed.strip():
        raise ValueError(f"Inherited repository files modified since {BASE_COMMIT}: {changed}")


def create_freeze():
    if FREEZE_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing freeze manifest: {FREEZE_PATH}")
    gate_path = RESULTS / "dev_gate.json"
    if not gate_path.exists():
        raise FileNotFoundError("Run dev_study before creating prospective freeze")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if not gate.get("gate_passed"):
        raise ValueError("Cannot freeze before dev_gate.json gate_passed=True")
    assert_inherited_clean()

    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        registration="within-session prospective freeze before formal scoring; not external public preregistration",
        design=design(),
        sources=source_hashes(),
        p6_checkpoints=p6_comparator_hashes(),
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
        missing_or_changed = {
            k for k in set(manifest["sources"]) | set(source_hashes())
            if manifest["sources"].get(k) != source_hashes().get(k)
        }
        raise ValueError(f"Frozen source/dev-gate hashes differ: {sorted(missing_or_changed)}")
    for rel_path, digest in manifest["p6_checkpoints"].items():
        if sha256_file(REPO / rel_path) != digest:
            raise ValueError(f"Archived P6 checkpoint modified: {rel_path}")
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
