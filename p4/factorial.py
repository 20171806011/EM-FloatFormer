"""Paired one-dimensional Stage P4 factorial study. Run as python -m p4.factorial."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from p4.em_floatformer import _train_comparative_model, create_floatformer
from p4.simulator import generate_frame


HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "p4cd_factorial_protocol.json"
PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
ARMS = PROTOCOL["arms"]
FACTORS = ("F1_gauge", "F2_hermitian", "F3_joint")
GAINS = PROTOCOL["evaluation"]["gains"]
RESERVED = frozenset(PROTOCOL["confirmatory_seeds"])


def require_development_seed(seed):
    if seed in RESERVED:
        raise RuntimeError(
            f"Seed {seed} is reserved for confirmation. The 1D development recipe "
            "cannot use reserved seeds. A separately frozen 1D protocol is required."
        )


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def instantiate_ablation_arm(arm_id=1, seed=71301, fixed_width=False):
    require_development_seed(seed)
    if not 1 <= arm_id <= len(ARMS):
        raise ValueError("arm_id must be between 1 and 8")
    gauge, hermitian, joint, width = ARMS[arm_id - 1]
    width = 8 if fixed_width else width

    def make(g, h, j):
        torch.manual_seed(seed)
        return create_floatformer(g, h, j, d_model=width, n_heads=2, depth=1, n_sc=32)

    # Corresponding stream, norm and readout parameters start identical when compatible.
    # Reference models are initialized normally, preserving correlated Hermitian Q/K.
    with torch.random.fork_rng(devices=[]):
        arm = make(gauge, hermitian, joint)
        reference = make(True, hermitian, True)
        gold = reference if hermitian else make(True, True, True)
        state = arm.state_dict()
        for model in (reference, gold):
            for name, value in model.state_dict().items():
                if name in state and value.shape == state[name].shape:
                    state[name] = value
        arm.load_state_dict(state)
    return arm


def get_yates_matrix():
    factors = np.asarray([arm[:3] for arm in ARMS], dtype=int) * 2 - 1
    g, h, j = factors.T
    return np.column_stack((np.ones(8, int), g, h, j, g*h, g*j, h*j, g*h*j))


def compute_main_effects(scores):
    values = np.asarray(scores, dtype=float).reshape(8)
    return dict(zip(FACTORS, (get_yates_matrix()[:, 1:4].T @ values / 4).tolist()))


def compute_interactions(scores):
    values = np.asarray(scores, dtype=float).reshape(8)
    return dict(zip(("G:H", "G:J", "H:J"), (get_yates_matrix()[:, 4:7].T @ values / 4).tolist()))


def training_run(arm_id, seed, fixed_width=False):
    require_development_seed(seed)
    settings = PROTOCOL["train"]
    model = instantiate_ablation_arm(arm_id, seed, fixed_width)
    model, _ = _train_comparative_model(
        model, steps=settings["steps"], batch_size=settings["batch_size"],
        n_sc=32, profile=settings["profile"], seed=seed,
        backbone_lr=settings["backbone_lr"], gamma_lr=settings["gamma_lr"],
    )
    return model


def observation(batch_size=96, seed=2017301, snr_db=10.0, profile="equal4"):
    return generate_frame(batch_size=batch_size, n=32, profile=profile, snr_db=snr_db,
                          pa_condition="strong", generator=torch.Generator().manual_seed(seed))


def scaled_inputs(frame, gain):
    return frame.yp * gain, frame.yd * gain, frame.noise_var * abs(gain)**2


@torch.no_grad()
def evaluate_arm(model, frame, seed, arm_id):
    per_gain = []
    for gain in GAINS:
        out = model(*scaled_inputs(frame, gain))
        if not all(torch.isfinite(x).all() for x in out):
            raise ArithmeticError(f"Non-finite receiver output in arm {arm_id} seed {seed} at gain {gain}")
        errors = int(torch.count_nonzero((out.llrs > 0) != frame.bits))
        total = frame.bits.numel()
        per_gain.append(dict(scale=gain, bit_errors=errors, total_bits=total, ber=errors/total))
    errors = sum(item["bit_errors"] for item in per_gain)
    bits = sum(item["total_bits"] for item in per_gain)
    return dict(seed=seed, arm_id=arm_id, gauge=model.use_gauge_norm,
                hermitian=model.use_hermitian_attention, joint=model.use_joint_pilot_data,
                width=model.d_model, parameters=sum(p.numel() for p in model.parameters()),
                bit_errors=errors, total_bits=bits, ber=errors/bits, gains=per_gain)


def run_rows(seeds, snr_db=10.0, fixed_width=False):
    seeds = tuple(seeds)
    for seed in seeds:
        require_development_seed(seed)
    rows = []
    for seed in seeds:
        frame = observation(seed=seed+PROTOCOL["evaluation"]["data_seed_offset"], snr_db=snr_db)
        for arm_id in range(1, 9):
            model = training_run(arm_id, seed, fixed_width)
            rows.append(evaluate_arm(model, frame, seed, arm_id))
    return rows


def summary(rows):
    all_seeds = sorted({r["seed"] for r in rows})
    summaries = []
    for seed in all_seeds:
        subset = sorted([r for r in rows if r["seed"] == seed], key=lambda r:r["arm_id"])
        if [r["arm_id"] for r in subset] != list(range(1, 9)):
            raise ValueError("Each seed must contain all eight arms once")
        # Score is negative BER, so positive contrasts indicate fewer errors.
        scores = -np.array([r["ber"] for r in subset])
        unit_scores = -np.array([r["gains"][1]["ber"] for r in subset])
        matrix = get_yates_matrix()
        coef = matrix.T @ scores / 8
        summaries.append(dict(seed=seed, main_effects=compute_main_effects(scores),
            unit_gain_main_effects=compute_main_effects(unit_scores), interactions=compute_interactions(scores),
            three_way=float(2*coef[-1]), decomposition_valid=bool(np.allclose(matrix @ coef, scores))))
    mean_effects = {f: float(np.mean([record["main_effects"][f] for record in summaries])) for f in FACTORS}
    return {"per_seed": summaries, "main_effects": mean_effects}


def artifact(rows):
    return {"protocol_sha256": digest(PROTOCOL_PATH), "source_sha256": digest(__file__),
            "pytorch": str(torch.__version__), "threads": torch.get_num_threads(),
            "rows": rows, **summary(rows)}


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f".{os.getpid()}.tmp")
    partial.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    partial.replace(path)


def run_full_factorial_evaluation(snr_db=10.0):
    seeds = PROTOCOL["dev_seeds"]
    archived = HERE / "results/p4c_factorial.json"
    if snr_db == PROTOCOL["evaluation"]["snr_db"] and archived.exists():
        result = json.loads(archived.read_text(encoding="utf-8"))
        if result["protocol_sha256"] != digest(PROTOCOL_PATH) or result["source_sha256"] != digest(__file__):
            raise ValueError("Factorial artifact belongs to a different experiment revision")
        rows = result["rows"]
    else:
        rows = run_rows(seeds, snr_db)
    stats = summary(rows)
    scores = [float(np.mean([-r["ber"] for r in rows if r["arm_id"] == a])) for a in range(1, 9)]
    return {"arm_scores": scores, "main_effects": stats["main_effects"],
            "decomposition_valid": all(r["decomposition_valid"] for r in stats["per_seed"])}


def run_ablation_confirmatory(seeds):
    """Allow disjoint development smoke runs. Formal 1D evaluation awaits a separate freeze."""
    return run_rows(seeds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", choices=("development", "fixed-width"))
    parser.add_argument("--threads", default=1, type=int)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    seeds = PROTOCOL["dev_seeds"]
    names = {"development":"p4c_factorial.json", "fixed-width":"p4c_fixed_width.json"}
    output = HERE / "results" / names[args.part]
    rows = []
    for seed in seeds:
        rows.extend(run_rows([seed], fixed_width=args.part == "fixed-width"))
        print(f"{args.part}: completed seed {seed}", flush=True)
    write_json(output, artifact(rows))

