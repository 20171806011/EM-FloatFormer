"""Frozen design, provenance and model loading for the soft-information study."""
from datetime import datetime, timezone
from pathlib import Path
import subprocess

from p5.wavefield_sim import simulate
from p6_followup.common import (REPO, RECIPE, corrupt, environment, load_model,
                               read, save, sha, tensor_hash)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BASE = "c699680dd84560747ddf40ecacd59821987bf545"
MAIN_BASE = "63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1"
SEED_BASE = 713611190000
CONDITIONS = ("intact", "first_attenuated", "first_erased", "interior_erased", "scattered_erased")
METHODS = ("trust", "smooth", "variance", "profile", "variance_profile",
           "local_parity", "fixed_parity", "waveform_dd", "lmmse_2d")
PRIMARY_CONDITIONS = ("intact", "first_erased", "interior_erased")
PRIMARY_BASELINES = ("trust", "smooth")


def design():
    return dict(base_commit=BASE, published_main=MAIN_BASE, regime=2, snr_db=9,
                indices=list(range(1, 11)), checkpoint_pairs=list(range(6, 16)),
                batches=4, batch_size=8,
                seeds={str(i): [SEED_BASE+1000*i+j for j in range(4)] for i in range(1, 11)},
                methods=list(METHODS), conditions=list(CONDITIONS), no_retraining=True,
                candidate="variance_profile", primary_conditions=list(PRIMARY_CONDITIONS),
                primary_baselines=list(PRIMARY_BASELINES), primary_endpoint="decoded_block_error_count",
                family_size=6, correction="Holm", inferential_unit="checkpoint/fresh-waveform cluster",
                profile_steps=2, phase_step_peak_cap_radians=.25,
                mode_mixture="algebraic unit tests only, not a BER receiver arm")


def source_hashes():
    paths = []
    for name in ("p4", "p5", "p5b", "p6", "p6_followup"):
        paths += sorted((REPO/name).rglob("*.py"))
    names = ("__init__.py", "common.py", "moments.py", "profile.py", "receiver.py",
             "study.py", "statistics.py", "test_moments.py", "test_profile.py",
             "test_receiver.py", "PROTOCOL.md", "MATH.md")
    paths += [ROOT/name for name in names]
    return {p.relative_to(REPO).as_posix(): sha(p) for p in paths}


def inherited_changes():
    return subprocess.check_output(["git", "diff", BASE, "--name-only", "--",
              "p4", "p5", "p5b", "p6", "p6_followup"], cwd=REPO, text=True).strip()


def freeze():
    destination = ROOT/"FREEZE.json"
    if destination.exists():
        raise FileExistsError("Never overwrite a frozen study")
    if inherited_changes():
        raise ValueError("Archived sources or evidence changed")
    checkpoints = {}
    for pair in range(6, 16):
        for arm in ("parity", "local_parity"):
            _, receipt = load_model(pair, arm)
            checkpoints[receipt["path"]] = receipt["sha256"]
    save(destination, dict(created_utc=datetime.now(timezone.utc).isoformat(),
         registration="prospective within this session, before fresh scoring; not external preregistration",
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
