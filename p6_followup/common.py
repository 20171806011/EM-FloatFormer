"""Shared provenance, observation interventions and scoped receiver adapters.

The adapters deliberately leave frozen p6 sources untouched. Their temporary
module hooks are process-global: use isolated processes, never concurrent threads.
"""
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import torch

from p5.study import rng
from p5.wavefield_sim import simulate
from p6 import experiment as ex, math_chirp_trust as trust
from p6.math_chirp_chart_stress import chart as hard_chart
from p6.math_training import Recipe, receiver

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
RESULTS = ROOT / "results"
BASE = "63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1"
SEED_BASE = 602611190000
CONDITIONS = ("intact", "first_attenuated", "first_erased", "interior_erased", "scattered_erased")
METHODS = ("trust", "smooth", "hard_chart", "local_parity", "fixed_parity", "waveform_dd", "lmmse_2d")
RECIPE = Recipe(chirp="fisher_trust", ula=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_hash(value):
    return hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    ex.save(Path(path), data)


def environment():
    import numpy, scipy, numba
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor())
    return dict(python=sys.version, torch=torch.__version__, numpy=numpy.__version__,
                scipy=scipy.__version__, numba=numba.__version__, platform=platform.platform(),
                cpu=cpu, threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                deterministic=torch.are_deterministic_algorithms_enabled(),
                precision="float64/complex128", device="cpu")


def design():
    return dict(base_commit=BASE, regime=2, snr_db=9, indices=list(range(1, 11)),
                checkpoint_pairs=list(range(6, 16)), batches=4, batch_size=8,
                seeds={str(i): [SEED_BASE + 1000*i + j for j in range(4)] for i in range(1, 11)},
                conditions=list(CONDITIONS), methods=list(METHODS), no_retraining=True,
                primary_conditions=["first_erased", "interior_erased"],
                primary_comparison=["trust", "smooth"], optional_exact_emit_in_primary=False)


def source_hashes():
    paths = []
    for name in ("p4", "p5", "p5b", "p6"):
        paths += sorted((REPO / name).rglob("*.py"))
    paths += [ROOT / name for name in ("__init__.py", "chirp.py", "exact_emit.py", "common.py",
                                      "study.py", "statistics.py", "PROTOCOL.md", "MATH.md")]
    paths += sorted((ROOT / "tests").glob("*.py"))
    return {p.relative_to(REPO).as_posix(): sha(p) for p in paths}


def checkpoint_path(pair, arm="parity", development=False):
    if development:
        return REPO / f"p6/results/dev_math/final/selected/r2/{pair}/{arm}.pt"
    return REPO / f"p6/results/formal/A/r2/pair{pair:02d}/{arm}.pt"


def load_model(pair, arm="parity", development=False):
    ck = checkpoint_path(pair, arm, development)
    record = read(ck.with_suffix(".json") if development else ck.parent / "fit.json")
    expected = record["checkpoint_sha256"] if development else record["checkpoints"][arm]
    if sha(ck) != expected:
        raise ValueError(f"Checkpoint digest mismatch: {ck}")
    state = torch.load(ck, map_location="cpu", weights_only=True)
    if not development and state["implementation"] != record["freeze_sha256"]:
        raise ValueError("Archived checkpoint source identity mismatch")
    model = receiver(arm, rng(2, 5, pair, 900), RECIPE)
    model.load_state_dict(state["state"])
    return model.eval(), dict(path=ck.relative_to(REPO).as_posix(), sha256=expected,
                             stored_implementation=state["implementation"])


def freeze():
    target = ROOT / "FREEZE.json"
    if target.exists():
        raise FileExistsError("Do not overwrite a study freeze")
    checkpoints = {}
    for pair in range(6, 16):
        for arm in ("parity", "local_parity"):
            _, receipt = load_model(pair, arm)
            checkpoints[receipt["path"]] = receipt["sha256"]
    inherited_changes = subprocess.check_output(
        ["git", "diff", BASE, "--name-only", "--", "p4", "p5", "p5b", "p6"], cwd=REPO, text=True)
    if inherited_changes.strip():
        raise ValueError("Frozen inherited code/results changed")
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    registration="prospective in this session; not external public preregistration",
                    design=design(), sources=source_hashes(), checkpoints=checkpoints,
                    environment=environment())
    save(target, manifest)
    return sha(target)


def verify_freeze():
    manifest = read(ROOT / "FREEZE.json")
    if manifest["design"] != design() or manifest["sources"] != source_hashes():
        raise ValueError("Frozen design or source differs")
    for name, digest in manifest["checkpoints"].items():
        if sha(REPO / name) != digest:
            raise ValueError(f"Frozen checkpoint differs: {name}")
    return sha(ROOT / "FREEZE.json")


def fresh_frame(index, batch=0, batch_size=8):
    if index not in range(1, 11) or batch not in range(4) or batch_size != 8:
        raise ValueError("Outside the fixed fresh evaluation design")
    return simulate(2, batch_size, 9, SEED_BASE + 1000*index + batch)


def corrupt(frame, condition):
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if condition == "intact":
        return frame
    mask = frame.obs.pilot_mask
    rows = torch.arange(mask.shape[0])[:, None]
    tones = torch.arange(mask.shape[1])[None] // 2
    if condition in ("first_attenuated", "first_erased"):
        selected = mask & (rows == 0)
    elif condition == "interior_erased":
        selected = mask & (rows == 9)
    else:
        selected = mask & ((rows + tones) % 2 == 0)
    y = frame.obs.y.clone()
    y[:, :, selected] *= .4 if condition == "first_attenuated" else 0.
    return replace(frame, obs=replace(frame.obs, y=y))


@contextmanager
def intervention(mode="trust", exact_emit=False):
    from p6_followup.chirp import residual
    from p6_followup.exact_emit import emit
    from p6 import axial_em_transformer as solver
    if mode not in ("trust", "smooth", "hard_chart"):
        raise ValueError(mode)
    with ExitStack() as stack:
        if mode != "trust":
            stack.enter_context(patch.object(trust, "residual", residual if mode == "smooth" else hard_chart))
        if exact_emit:
            stack.enter_context(patch.object(solver, "_emit", emit))
        yield


class AdaptedReceiver(torch.nn.Module):
    """Scoped opt-in intervention; shares the original model and parameters."""
    def __init__(self, model, mode="trust", exact_emit=False):
        super().__init__()
        self.model, self.mode, self.exact_emit = model, mode, exact_emit

    def forward(self, obs, **kwargs):
        with intervention(self.mode, self.exact_emit):
            return self.model(obs, **kwargs)
