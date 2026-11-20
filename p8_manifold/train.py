"""Training loop, zero-leakage probe masking, and SHA-256 checkpoint helpers for Phase 8."""
from dataclasses import replace
import hashlib
import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from p4.simulator import q16
from p5.study import setup
from p8_manifold.eval_utils import offgrid_holdout_nll
from p8_manifold.manifold_charts import B_ICI
from p8_manifold.model import EM8ManifoldReceiver
from p8_manifold.offgrid_sim import OFFGRID_CONDITIONS, extract_payload_llrs, simulate_offgrid, true_offgrid_operator


REAL = torch.float64
COMPLEX = torch.complex128
TRAIN_CONDITIONS = OFFGRID_CONDITIONS  # ("O0_random_sparse", "O1_roving_island", "O2_blind_burst_erasure", "O3_holdout_sequential")


def lr_multiplier(step_idx, total_steps, warmup=5, min_ratio=0.1):
    if step_idx < warmup:
        return (step_idx + 1) / float(warmup)
    if total_steps <= warmup:
        return 1.0
    progress = (step_idx - warmup) / float(max(total_steps - warmup - 1, 1))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def _train_obs_and_probe(frame, step_seed):
    """Prepare zero-leakage training observation and held-out probe mask."""
    if frame.condition == "O3_holdout_sequential":
        return frame.obs, frame.probe_mask
    if frame.condition == "O2_blind_burst_erasure":
        # Keep blind condition untouched (all valid_mask True, blind=True)
        return frame.obs, torch.zeros_like(frame.probe_mask)

    # On O0/O1, hold out 8 random TX pilots per burst for direct zero-leakage predictive NLL supervision
    b = frame.obs.y.shape[0]
    probe = torch.zeros_like(frame.probe_mask)
    gen = torch.Generator().manual_seed((int(step_seed) * 1013 + 17) % (2**63 - 1))
    for i in range(b):
        coords = frame.obs.pilot_mask[i].nonzero(as_tuple=False)
        perm = torch.randperm(len(coords), generator=gen)[:8]
        sel = coords[perm]
        probe[i, sel[:, 0], sel[:, 1]] = True
    valid = frame.obs.valid_mask & (~probe)
    y_masked = torch.where(valid[:, None], frame.obs.y, torch.zeros_like(frame.obs.y))
    return replace(frame.obs, y=y_masked, valid_mask=valid), probe


def train_em8_model(
    variant,
    init_seed,
    train_seed_base,
    *,
    d_model=16,
    n_layers=2,
    n_heads=4,
    steps=40,
    lr=0.0035,
    snr_db=15.0,
    batch_size=8,
    lambda_hold=0.08,
    lambda_nmse=0.01,
    lambda_ici=0.05,
    lambda_cal=0.02,
    anchor_l2=1e-4,
    grad_clip=1.0,
    em_steps=2,
):
    """Train one EM8 variant with exact step-0 parity and warm-cosine Adam on Off-Grid O0..O3."""
    setup()
    torch.manual_seed(int(init_seed))
    model = EM8ManifoldReceiver(
        variant=variant, d_model=d_model, n_layers=n_layers, n_heads=n_heads, steps=em_steps
    )
    if not model.learned:
        return model.eval(), dict(elapsed_seconds=0.0, steps=0, history=[])

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    history = []
    t_start = time.perf_counter()

    for s in range(steps):
        mult = lr_multiplier(s, steps)
        for g in opt.param_groups:
            g["lr"] = lr * mult

        cond = TRAIN_CONDITIONS[s % len(TRAIN_CONDITIONS)]
        step_seed = int(train_seed_base) + s * 37
        frame = simulate_offgrid(batch_size, snr_db=snr_db, seed=step_seed, condition=cond)
        obs_in, probe = _train_obs_and_probe(frame, step_seed + 9)

        opt.zero_grad(set_to_none=True)
        out = model(obs_in)

        payload_llr = extract_payload_llrs(frame.obs.pilot_mask, out.bit_llr)
        signed_bits = 2.0 * frame.bits.to(device=payload_llr.device, dtype=REAL) - 1.0
        bce = F.softplus(-signed_bits * payload_llr).mean()

        if bool(probe.any()):
            hold_nll = offgrid_holdout_nll(
                out.log_pi, out.branch_pred_y, out.branch_pred_var,
                frame.y_clean_full[:, 0], frame.obs.noise_var, probe,
            )
        else:
            hold_nll = bce.new_zeros(())

        nmse = (
            (out.h_hat[:, 0] - frame.h_true[:, 0]).abs().square().sum()
            / frame.h_true[:, 0].abs().square().sum().clamp_min(1e-300)
        )
        # Exact physical banded-ICI supervision against true_offgrid_operator's 4 off-diagonal bands
        with torch.no_grad():
            G_true = true_offgrid_operator(frame)  # B, T, N, N
            f_idx = torch.arange(G_true.shape[-1], device=G_true.device)
            offs = [d for d in range(-B_ICI, B_ICI + 1) if d != 0]
            ici_true = torch.stack([G_true[:, :, f_idx, (f_idx + d) % G_true.shape[-1]] for d in offs], dim=-1)
        ici_hat = (out.pi[:, :, None, None, None] * out.state["ici_m"]).sum(1) * out.state["scale_c"][:, None, None, None]
        ici_loss = (
            (ici_hat - ici_true).abs().square().sum()
            / frame.h_true[:, 0].abs().square().sum().clamp_min(1e-300)
        )
        # Proper C*-invariant Gaussian symbol-moment calibration score on payload symbols
        b_cur = frame.bits.shape[0]
        pm = frame.obs.pilot_mask
        sym_m = torch.stack([out.symbol_mean[i, ~pm[i]] for i in range(b_cur)], dim=0)
        sym_v = torch.stack([out.symbol_var_total[i, ~pm[i]] for i in range(b_cur)], dim=0).clamp_min(1e-6)
        x_true = q16(frame.bits).to(COMPLEX)
        cal_score = (sym_v.log() + (x_true - sym_m).abs().square() / sym_v).mean()

        anchor = sum((p - q).abs().square().sum() for p, q in zip(model.backbone.parameters(), model.initial_map.parameters()))
        anchor = anchor + model.fusion_scale.square().sum()

        loss = (
            bce
            + lambda_hold * hold_nll
            + lambda_nmse * nmse
            + lambda_ici * ici_loss
            + lambda_cal * cal_score
            + anchor_l2 * anchor
        )
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, grad_clip))
        opt.step()

        history.append(dict(
            step=s,
            condition=cond,
            lr=lr * mult,
            loss=float(loss.detach()),
            bce=float(bce.detach()),
            hold_nll=float(hold_nll.detach()),
            nmse=float(nmse.detach()),
            grad_norm=gnorm,
        ))

    return model.eval(), dict(elapsed_seconds=time.perf_counter() - t_start, steps=steps, history=history)


def save_checkpoint(path, model, meta):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = dict(
        variant=model.variant,
        n_hyp=model.n_hyp,
        d_model=model.d_model,
        n_layers=model.n_layers,
        n_heads=model.n_heads,
        steps=model.steps,
        gamma_init=model.gamma_init,
    )
    torch.save({"state": model.state_dict(), "config": cfg, "meta": meta}, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = EM8ManifoldReceiver(**payload["config"])
    model.load_state_dict(payload["state"])
    return model.eval(), payload.get("meta", {})
