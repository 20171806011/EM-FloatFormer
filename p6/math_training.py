"""Shared training, parameter initialization and opt-in mathematical receiver variants."""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from p5.study import payload
from p6 import experiment as ex, training_iter3
from p6.axial_em_transformer import AxialEMTransformer
from p6.iter5_dev import CONFIG


@dataclass(frozen=True)
class Recipe:
    chirp: str = "baseline"
    bridge: bool = False
    differential: bool = False
    irc: bool = False
    ula: bool = False
    loss: str = "bce"
    steps: int = 48


def receiver(arm, seed, recipe):
    torch.manual_seed(seed)
    model = AxialEMTransformer(arm, steps=3)
    model.chirp_mode = recipe.chirp
    model.r1_bridge = recipe.bridge
    model.r1_differential = recipe.differential
    model.spatial_irc = recipe.irc
    model.spatial_ula = recipe.ula
    return model


def objective(frame, model, recipe):
    if recipe.loss == "bce":
        return training_iter3.objective(frame, model, aux_nmse=CONFIG.aux_nmse)[0]
    if recipe.loss != "brier":
        raise ValueError(f"Unknown loss {recipe.loss}")
    llr, channel = model(frame.obs)
    logits = payload(frame, llr)
    u = training_iter3.CAP * (logits / training_iter3.CAP).tanh()
    bits = frame.bits.double()
    loss = (F.binary_cross_entropy_with_logits(u, bits) + 2 * (u.sigmoid() - bits).square().mean()) / 2
    nmse = (channel - frame.h_true).abs().square().sum() / frame.h_true.abs().square().sum()
    anchor = sum(
        (p - p0).abs().square().sum()
        for p, p0 in zip(model.head.parameters(), model.initial_map.parameters(), strict=True)
    )
    return loss + CONFIG.aux_nmse * nmse + (training_iter3.ANCHOR / 2) * anchor


def train(model, batch_at, recipe, first_check=None):
    params = list(model.head.parameters())
    opt = torch.optim.Adam(params, lr=CONFIG.lr)
    for update in range(recipe.steps):
        opt.param_groups[0]["lr"] = ex.learning_rate(CONFIG, update, recipe.steps)
        opt.zero_grad(set_to_none=True)
        loss = objective(batch_at(update), model, recipe)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at update {update}")
        loss.backward()
        if not update and first_check is not None:
            first_check(params)
        torch.nn.utils.clip_grad_norm_(params, CONFIG.clip, error_if_nonfinite=True)
        opt.step()
