"""Fixed design and provenance for the observation/soft-reliability study."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess

from p5.wavefield_sim import simulate
from p6_followup.common import (REPO, corrupt, environment, load_model, read,
                               save, sha, tensor_hash)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BASE = "bb9396f08132bb5f586ff728dcfa4e7872201c7b"
MAIN_BASE = "63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1"
SEED_BASE = 814611190000
CONDITIONS = ("intact", "first_attenuated", "first_erased", "interior_erased", "scattered_erased")
METHODS = ("legacy_mean", "legacy_variance", "aware_mean", "aware_variance",
           "aware_effective", "aware_cavity_mean", "aware_consistent")


def design():
    return dict(base_commit=BASE, published_main=MAIN_BASE, regime=2, snr_db=9,
        indices=list(range(1, 11)), checkpoint_pairs=list(range(6, 16)),
        batches=4, batch_size=8,
        seeds={str(i): [SEED_BASE+1000*i+j for j in range(4)] for i in range(1, 11)},
        methods=list(METHODS), conditions=list(CONDITIONS), no_retraining=True,
        primary_condition="scattered_erased", primary_endpoint="decoded_block_error_count",
        primary_tests=["observation_by_variance_interaction", "aware_consistency_addition"],
        family_size=2, correction="Holm", inferential_unit="checkpoint/fresh-waveform cluster",
        calibration_bins=[0., .01, .05, .1, .2, .4, .8, None],
        calibration_points="three actual amplitude-update posterior inputs; payload only",
        metadata="known receive gain for all aware arms; absent for legacy arms",
        candidate="aware_consistent", phase_rule="archived Trust, no profile Hessian",
        scope="mechanism diagnostic, not blind-receiver superiority or unknown-family generalization")


def source_hashes():
    paths = []
    for name in ("p4", "p5", "p5b", "p6", "p6_followup", "p6_softinfo"):
        paths += sorted((REPO/name).rglob("*.py"))
    names = ("__init__.py", "common.py", "observation.py", "weighted.py", "reliability.py",
             "receiver.py", "calibration.py", "study.py", "statistics.py", "test_weighted.py",
             "test_reliability.py", "test_receiver.py", "test_calibration.py", "PROTOCOL.md", "MATH.md")
    paths += [ROOT/name for name in names]
    return {p.relative_to(REPO).as_posix(): sha(p) for p in paths}


def inherited_changes():
    return subprocess.check_output(["git", "diff", BASE, "--name-only", "--",
        "p4", "p5", "p5b", "p6", "p6_followup", "p6_softinfo"], cwd=REPO, text=True).strip()


def freeze():
    destination = ROOT/"FREEZE.json"
    if destination.exists():
        raise FileExistsError("Never overwrite a frozen study")
    if inherited_changes():
        raise ValueError("Archived sources or evidence changed")
    checkpoints = {}
    for pair in range(6, 16):
        _, receipt = load_model(pair)
        checkpoints[receipt["path"]] = receipt["sha256"]
    save(destination, dict(created_utc=datetime.now(timezone.utc).isoformat(),
        registration="prospective within this session before fresh scoring; not external preregistration",
        design=design(), sources=source_hashes(), checkpoints=checkpoints, environment=environment()))
    return sha(destination)


def verify_freeze():
    record = read(ROOT/"FREEZE.json")
    if record["design"] != design() or record["sources"] != source_hashes():
        raise ValueError("Frozen design/source mismatch")
    for path, digest in record["checkpoints"].items():
        if sha(REPO/path) != digest:
            raise ValueError(f"Checkpoint mismatch: {path}")
    if inherited_changes():
        raise ValueError("Historical evidence changed")
    return sha(ROOT/"FREEZE.json")


def fresh_frame(index, batch=0):
    if index not in range(1, 11) or batch not in range(4):
        raise ValueError("Outside frozen fresh seed design")
    return simulate(2, 8, 9, SEED_BASE+1000*index+batch)
