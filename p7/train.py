"""Training loop and checkpoint provenance helpers for Phase 7 models."""
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from p5.study import rng, setup
from p7.conditions import apply_observation_gain, mask_for_condition
from p7.metrics import holdout_nll_loss
from p7.model import EM7HypothesisReceiver


REAL = torch.float64
TRAIN_CONDITIONS = (
    "C0_intact",
    "C1_row9_erased",
    "C2_row0_erased",
    "C3_scattered_half",
    "C4a_roving_window",
)


def _step_mask_and_probe(pilot_mask, step_idx, batch_size, seed):
    cond = TRAIN_CONDITIONS[step_idx % len(TRAIN_CONDITIONS)]
    gain, _, probe = mask_for_condition(pilot_mask, cond, batch_size, seed=seed)
    if cond == "C0_intact":
        _, _, probe = mask_for_condition(pilot_mask, "C5_holdout_update", batch_size, seed=seed + 7)
        gain[probe] = 0.
        return gain, probe, cond
    probe = torch.zeros_like(gain, dtype=torch.bool)
    gen = torch.Generator().manual_seed((int(seed) * 997 + 13) % (2**63 - 1))
    for b in range(batch_size):
        erased = (pilot_mask & (gain[b] == 0)).nonzero(as_tuple=False)
        perm = torch.randperm(len(erased), generator=gen)[:4]
        coords = erased[perm]
        probe[b, coords[:, 0], coords[:, 1]] = True
    return gain, probe, cond


def lr_multiplier(step_idx, total_steps, warmup=4, min_ratio=0.1):
    if step_idx < warmup:
        return (step_idx + 1) / float(warmup)
    if total_steps <= warmup:
        return 1.0
    progress = (step_idx - warmup) / float(max(total_steps - warmup - 1, 1))
    import math
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def train_em7_model(
    variant,
    init_seed,
    frame_fn,
    *,
    steps=40,
    lr=0.004,
    lambda_hold=0.05,
    lambda_nmse=0.005,
    anchor_l2=5e-5,
    batch_size=8,
    grad_clip=1.0,
    mask_seed_base=0,
):
    """Train one EM7 variant with exact step-0 seed parity and warm-cosine Adam."""
    setup()
    torch.manual_seed(int(init_seed))
    model = EM7HypothesisReceiver(variant)
    if not model.learned:
        return model.eval(), []

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    traces = []
    t_start = time.perf_counter()

    for s in range(steps):
        mult = lr_multiplier(s, steps)
        for g in opt.param_groups:
            g["lr"] = lr * mult

        frame = frame_fn(s, batch_size)
        gain, probe, cond = _step_mask_and_probe(frame.obs.pilot_mask, s, batch_size, mask_seed_base + s * 37)
        obs_c = apply_observation_gain(frame, gain).obs

        opt.zero_grad(set_to_none=True)
        out = model(obs_c, gain)

        payload = out.bit_llr[:, ~frame.obs.pilot_mask]
        signed_bits = 2.0 * frame.bits.to(device=payload.device, dtype=REAL) - 1.0
        bce = F.softplus(-signed_bits * payload).mean()
        hold_nll = holdout_nll_loss(
            out.log_pi, out.branch_pred_y, out.branch_pred_var,
            frame.obs.y[:, 0], frame.obs.noise_var, probe,
        )
        nmse = (
            (out.h_hat[:, 0] - frame.h_true[:, 0]).abs().square().sum()
            / frame.h_true[:, 0].abs().square().sum().clamp_min(1e-300)
        )
        anchor = sum((p - q).abs().square().sum() for p, q in zip(model.head.parameters(), model.initial_map.parameters()))
        if model.multi_hyp:
            anchor = anchor + model.fusion_scale.square().sum()
        loss = bce + lambda_hold * hold_nll + lambda_nmse * nmse + anchor_l2 * anchor
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, grad_clip))
        opt.step()

        traces.append(dict(
            step=s, condition=cond, lr=lr * mult,
            loss=float(loss.detach()), bce=float(bce.detach()),
            hold_nll=float(hold_nll.detach()), nmse=float(nmse.detach()),
            grad_norm=gnorm,
        ))

    return model.eval(), dict(elapsed_seconds=time.perf_counter() - t_start, steps=steps, history=traces)


def save_checkpoint(path, model, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "variant": model.variant, "meta": meta}, path)
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = EM7HypothesisReceiver(payload["variant"])
    model.load_state_dict(payload["state"])
    return model.eval(), payload.get("meta", {})
