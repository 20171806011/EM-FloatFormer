"""Evaluation-only R3 ULA steering measurements at the receiver's amplitude updates."""
import math
from unittest.mock import patch

import torch

from p5.wavefield_sim import steer
from p6 import axial_em_transformer as solver


REMOTE_EDGES = ((0, 2), (0, 3), (1, 3))
DEFINITION = (
    "On the stream-6 validation frames, record each per-ray complex antenna vector immediately "
    "before and after the spatial correction. After inference, compare against the simulator's "
    "unit-norm ULA direction for that ray in nominal delay order. Steering cosine is "
    "abs(u^H a)/norm(a). Remote phase cosine is the cosine of the relative-phase error on "
    "antenna pairs (0,2), (0,3), (1,3). Each nonzero burst-ray or edge contributes equally. "
    "Off-mode RMS ratio is sqrt(sum norm(a_after-u*u^H*a_after)^2 / sum norm(a_before-u*u^H*a_before)^2)."
)


def _state(amp, mode):
    overlap = (mode.conj() * amp).sum(dim=1, keepdim=True)
    off_mode = amp - mode * overlap
    magnitude = amp.norm(dim=1)
    valid = magnitude > 0
    stats = dict(steering_count=int(valid.sum()),
                 steering_sum=float((overlap[:, 0].abs()[valid] / magnitude[valid]).sum()),
                 off_mode_power=float(off_mode.abs().square().sum()))
    for i, j in REMOTE_EDGES:
        observed = amp[:, j] * amp[:, i].conj()
        expected = mode[:, j] * mode[:, i].conj()
        aligned = observed * expected.conj()
        valid = aligned.abs() > 0
        stats[f"edge_{i}{j}_count"] = int(valid.sum())
        stats[f"edge_{i}{j}_sum"] = float((aligned.real[valid] / aligned.abs()[valid]).sum())
    return stats


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _summarize(rows):
    sums = {key: sum(row["sums"][key] for row in rows) for key in rows[0]["sums"]}
    result = dict(burst_rays=sum(row["burst_rays"] for row in rows), sums=sums)
    edges = {}
    for state in ("before", "after"):
        result[f"steering_cosine_{state}"] = _ratio(sums[f"{state}_steering_sum"],
                                                    sums[f"{state}_steering_count"])
        edge_sum, edge_count = 0., 0
        for i, j in REMOTE_EDGES:
            name = f"{i}{j}"
            total, count = (sums[f"{state}_edge_{name}_{stat}"] for stat in ("sum", "count"))
            edges.setdefault(name, {})[state] = _ratio(total, count)
            edge_sum += total
            edge_count += count
        result[f"remote_phase_cosine_{state}"] = _ratio(edge_sum, edge_count)
    proportion = _ratio(sums["after_off_mode_power"], sums["before_off_mode_power"])
    result["true_off_mode_rms_ratio"] = math.sqrt(proportion) if proportion is not None else None
    result["remote_edges"] = edges
    return result


def pool(records):
    """Pool sufficient statistics across compatible stage records from any batch sizes."""
    if not records or any(record.keys() != records[0].keys() for record in records):
        raise ValueError("Expected nonempty R3 records with matching stage labels")
    stages = {name: _summarize([record[name] for record in records])
              for name in records[0] if name != "all_waveform_solves"}
    stages["all_waveform_solves"] = _summarize(
        [row for name, row in stages.items() if name.startswith("waveform_solve_")])
    return stages


@torch.inference_mode()
def measure(model, frame):
    """Run the ordinary receiver, then score the captured updates against true steering."""
    if frame.obs.regime != 3 or frame.obs.y.shape[1] != 4:
        raise ValueError("R3 spatial measurements require four ULA elements")
    snapshots = []
    apply = solver._apply_spatial

    def capture(amp, delta_amp):
        updated = apply(amp, delta_amp)
        snapshots.append((amp, updated))
        return updated

    with patch.object(solver, "_apply_spatial", capture):
        outputs = model(frame.obs)

    mode = steer(frame.truth.angles, 4).mT
    labels = ["pilot_redemap", *(f"waveform_solve_{i+1}" for i in range(model.steps))]
    stages = {}
    for label, (before, after) in zip(labels, snapshots, strict=True):
        sums = {f"{state}_{key}": value for state, amp in (("before", before), ("after", after))
                for key, value in _state(amp, mode).items()}
        stages[label] = dict(burst_rays=before.shape[0] * before.shape[2], sums=sums)
    return outputs, pool([stages])
