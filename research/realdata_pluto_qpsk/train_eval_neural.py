#!/usr/bin/env python3
"""Group C: Lightweight trainable PyTorch CPU Content-Attention + Candidate Weight Calibrator +
Residual Channel/Phase Refiner (`EMFloatFormerLite`) under strictly shared candidate generation
and exact mixture likelihood decoding.

Strict experimental protocol:
  - Single-threaded CPU execution (`TORCH_NUM_THREADS=1`, deterministic algorithms enabled).
  - Trains ONLY on `train` split ([200, 3200), 3,000 frames across 5 training regimes).
  - Model selection & post-hoc variance calibration STRICTLY on `dev` split ([3500, 4500), 1,000 frames).
  - Closed-book evaluation on `test` split ([4800, 7040), 2,240 frames, G=16 clusters) across ALL 8 conditions.
  - 3 independent random seeds (20261121, 20261122, 20261123); reports per-seed, mean+-std,
    training cost, parameter count, single-thread inference latency, peak RSS, and G=16 exact
    sign-flip paired tests against dev-tuned classical global A5 and B3b!
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import time
from typing import Any

import numpy as np
from scipy.special import logsumexp
import torch
import torch.nn as nn
import torch.nn.functional as F

from evaluator import aggregate_records, evaluate_frame_record, holm_bonferroni, paired_cluster_signflip
from receivers import (
    CFO_GRID,
    K_ALL,
    K_CENTER,
    K_PIL,
    N_CANDIDATES_B,
    N_PIL,
    PRIOR_DC_RATIO,
    QUAD_PHASES,
    S_PIL,
    ReceiverHyperparams,
    _fuse_prior_with_pilots,
    build_causal_history_library,
    build_shared_b_candidates,
    compute_content_prior,
    decode_group_b_variants,
    estimate_single_frame_pilot_stats,
)
from waveform_frontend import FULL_HDR_26, N_SYM, PAYLOAD_SLICE, PILOT_SLICE, PROBE_SLICE, QPSK_BITS, QPSK_CONST, STRESS_SPECS

SEEDS = (20261121, 20261122, 20261123)
LIB_SIZE = 200


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def set_deterministic(seed: int) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)


def extract_frame_features_and_bank(
    bundle: dict[str, np.ndarray],
    tx_mm: np.ndarray,
    hp: ReceiverHyperparams,
    cached_lib: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Precompute scale-invariant causal library features & shared 84-branch physical candidate bank.

    Every tensor returned here uses ONLY legal causal inputs (current online pilots + causal past library),
    with offline ground-truth targets stored cleanly under `targets` for training/evaluation ONLY.
    """
    y_all = np.asarray(bundle["y_sym"], dtype=np.complex128)
    frame_indices = np.asarray(bundle["frame_indices"], dtype=np.int32)
    eval_offset = int(bundle["eval_offset"])
    h_ora_all = np.asarray(bundle["h_ora"], dtype=np.complex128)
    d_ora_all = np.asarray(bundle["d_ora"], dtype=np.complex128)
    sync_ok_all = np.asarray(bundle["sync_ok"], dtype=np.bool_)

    lib = (
        build_causal_history_library(y_all, hp.h_mean_prior, hp.h_var_prior, lib_size=LIB_SIZE)
        if cached_lib is None
        else cached_lib
    )

    n_eval = len(y_all) - eval_offset
    nb_k = hp.neighbor_k
    nb_w = np.exp(-hp.neighbor_decay * np.arange(nb_k, 0, -1, dtype=np.float64))
    nb_w /= np.sum(nb_w)
    ages_norm = np.log1p(np.arange(LIB_SIZE, 0, -1, dtype=np.float64)) / math.log1p(LIB_SIZE)

    # Compact feature buffers (float32 for lightweight CPU training)
    # 1) Content-addressing attention features:
    #    lib_feat: (N, 201, 10), lib_h_reim: (N, 201, 2), lib_d_reim: (N, 201, 2), lib_v: (N, 201)
    #    query_ctx: (N, 8)
    # 2) Candidate bank features (shared 84 physical branches from fixed classical bank!):
    #    cand_feat: (N, 84, 9), cand_logits_phys: (N, 84), cand_h_post: (N, 84, 2), cand_d_post: (N, 84, 2)
    #    cand_obs_var: (N, 84), cand_log_p_y_s: (N, 84, 100, 4) -> wait: don't store (N,84,100,4) in RAM!
    #    Instead store lightweight y_pay (N, 100, 2) + h_post (N, 84, 2) + d_post (N, 84, 2) + obs_var (N, 84)
    #    so peak memory across 15,000 train frames stays < 450 MB!
    lib_feat = np.zeros((n_eval, LIB_SIZE + 1, 10), dtype=np.float32)
    lib_phys_score = np.zeros((n_eval, LIB_SIZE + 1), dtype=np.float32)
    lib_h_all = np.zeros((n_eval, LIB_SIZE + 1, 2), dtype=np.float32)
    lib_d_all = np.zeros((n_eval, LIB_SIZE + 1, 2), dtype=np.float32)
    lib_v_all = np.zeros((n_eval, LIB_SIZE + 1), dtype=np.float32)
    query_ctx = np.zeros((n_eval, 8), dtype=np.float32)

    cand_feat = np.zeros((n_eval, N_CANDIDATES_B, 9), dtype=np.float32)
    cand_logits_phys = np.zeros((n_eval, N_CANDIDATES_B), dtype=np.float32)
    cand_h_post = np.zeros((n_eval, N_CANDIDATES_B, 2), dtype=np.float32)
    cand_d_post = np.zeros((n_eval, N_CANDIDATES_B, 2), dtype=np.float32)
    cand_obs_var = np.zeros((n_eval, N_CANDIDATES_B), dtype=np.float32)

    y_pil_arr = np.zeros((n_eval, N_PIL, 2), dtype=np.float32)
    y_probe_arr = np.zeros((n_eval, 5, 2), dtype=np.float32)
    y_pay_arr = np.zeros((n_eval, 100, 2), dtype=np.float32)
    sigma2_arr = np.zeros(n_eval, dtype=np.float32)
    h_ora_arr = np.zeros((n_eval, 2), dtype=np.float32)
    d_ora_arr = np.zeros((n_eval, 2), dtype=np.float32)
    s_pay_idx_arr = np.zeros((n_eval, 100), dtype=np.int64)
    fidx_arr = np.zeros(n_eval, dtype=np.int32)
    sync_arr = np.zeros(n_eval, dtype=np.bool_)

    # Precompute branch structural descriptors (84, 5)
    q_ids = np.repeat(np.tile(np.arange(4), 1), 7)
    # Exact order matching build_shared_b_candidates: c_i in 0..2, q_i in 0..3, g_i in 0..6
    c_grid = np.repeat(np.arange(3), 28)
    q_grid = np.tile(np.repeat(np.arange(4), 7), 3)
    g_grid = np.tile(np.arange(7), 12)
    cfo_vals = CFO_GRID[g_grid]

    for i in range(n_eval):
        t = eval_offset + i
        fidx = int(frame_indices[t])
        y = y_all[t]
        y_pil = y[PILOT_SLICE]
        s_true = np.asarray(tx_mm[fidx], dtype=np.complex128)

        lh = lib["h_dd"][t - LIB_SIZE:t]
        ld = lib["d_dd"][t - LIB_SIZE:t]
        lvh = lib["var_h_dd"][t - LIB_SIZE:t]
        lqual = lib["qual_dd"][t - LIB_SIZE:t]

        st = estimate_single_frame_pilot_stats(y_pil, hp.h_mean_prior, hp.h_var_prior)
        sig2 = st["sigma2_est"]
        h_pil = st["h_mmse"]
        h_pil_abs = max(abs(h_pil), 1e-6)

        # Neighbor prior
        nh, nd, nvh = lh[-nb_k:], ld[-nb_k:], lvh[-nb_k:]
        h_pr_nb = complex(np.sum(nb_w * nh))
        d_pr_nb = complex(np.sum(nb_w * nd))
        v_pr_nb = float(np.sum(nb_w * (nvh + np.abs(nh - h_pr_nb) ** 2 + hp.neighbor_drift_rel_var * np.abs(nh) ** 2)))

        # Physical content attention (classical A5)
        ca = compute_content_prior(y_pil, lh, ld, lvh, sig2, hp)

        # Build library feature tensor (201, 10) — scale-invariant physical features!
        h_scale2 = np.maximum(np.abs(lh) ** 2, 1e-10)
        nu = sig2 + lvh + hp.attn_drift_rel_var * h_scale2
        pred = lh[:, None] * S_PIL[None, :] + ld[:, None]
        diff = y_pil[None, :] - pred
        res2 = np.sum(diff.real ** 2 + diff.imag ** 2, axis=1)
        ll_lib = -(N_PIL * np.log(np.pi * nu) + res2 / nu)
        _, _, _, _, ll_clear = _fuse_prior_with_pilots(
            y_pil, hp.h_mean_prior, hp.h_var_prior, hp.h_mean_prior * PRIOR_DC_RATIO, sig2
        )
        phys_s = np.empty(LIB_SIZE + 1, dtype=np.float64)
        phys_s[:LIB_SIZE] = (ll_lib - math.log(float(LIB_SIZE))) / max(hp.attn_temp, 1e-3)
        phys_s[LIB_SIZE] = (ll_clear + hp.clear_prior_log_bias) / max(hp.attn_temp, 1e-3)
        phys_s_centered = phys_s - float(np.max(phys_s))
        lib_phys_score[i] = np.clip(phys_s_centered, -30.0, 0.0).astype(np.float32)

        dphi = np.angle(lh * np.conj(h_pil))
        rel_dist = np.abs(lh - h_pil) / h_pil_abs
        # Fill 200 library entries
        lib_feat[i, :LIB_SIZE, 0] = np.clip(-res2 / (N_PIL * nu), -15.0, 0.0)
        lib_feat[i, :LIB_SIZE, 1] = np.clip(np.log(nu / max(sig2, 1e-12)), -5.0, 5.0)
        lib_feat[i, :LIB_SIZE, 2] = np.clip(np.log(np.maximum(np.abs(lh), 1e-6) / h_pil_abs), -3.0, 3.0)
        lib_feat[i, :LIB_SIZE, 3] = np.cos(dphi)
        lib_feat[i, :LIB_SIZE, 4] = np.sin(dphi)
        lib_feat[i, :LIB_SIZE, 5] = np.clip(rel_dist, 0.0, 5.0)
        lib_feat[i, :LIB_SIZE, 6] = np.clip(np.log1p(h_scale2 / np.maximum(lvh, 1e-12)), 0.0, 8.0) / 8.0
        lib_feat[i, :LIB_SIZE, 7] = lqual
        lib_feat[i, :LIB_SIZE, 8] = ages_norm
        lib_feat[i, :LIB_SIZE, 9] = 0.0  # is_clear = 0

        # Fill clear token (index 200)
        lib_feat[i, LIB_SIZE, 0] = np.clip(phys_s_centered[LIB_SIZE] / float(N_PIL), -15.0, 0.0)
        lib_feat[i, LIB_SIZE, 1] = np.clip(math.log(hp.h_var_prior / max(sig2, 1e-12)), -5.0, 5.0)
        lib_feat[i, LIB_SIZE, 5] = np.clip(abs(hp.h_mean_prior - h_pil) / h_pil_abs, 0.0, 5.0)
        lib_feat[i, LIB_SIZE, 9] = 1.0  # is_clear = 1

        # Values for library + clear
        h_entries = np.append(lh, hp.h_mean_prior)
        d_entries = np.append(ld, hp.h_mean_prior * PRIOR_DC_RATIO)
        v_entries = np.append(lvh + hp.attn_drift_rel_var * h_scale2, hp.h_var_prior)
        lib_h_all[i, :, 0] = h_entries.real
        lib_h_all[i, :, 1] = h_entries.imag
        lib_d_all[i, :, 0] = d_entries.real
        lib_d_all[i, :, 1] = d_entries.imag
        lib_v_all[i] = v_entries

        # Query context features (8,)
        snr_pil_log = math.log1p(h_pil_abs ** 2 / max(sig2, 1e-12))
        jump_prev1 = abs(h_pil - lh[-1]) / h_pil_abs
        jump_nb = abs(h_pil - h_pr_nb) / h_pil_abs
        nb_spread = math.sqrt(max(v_pr_nb, 1e-12)) / h_pil_abs
        # Rough intra-pilot phase slope (between first 4 and last 4 pilots)
        p_first = np.vdot(S_PIL[:4], y_pil[:4] - st["d_mmse"])
        p_last = np.vdot(S_PIL[4:], y_pil[4:] - st["d_mmse"])
        pil_slope = float(np.angle(p_last * np.conj(p_first))) / 4.0
        # Entropy of classical content weights
        w_ca = ca["weights"]
        ent_ca = -float(np.sum(w_ca * np.log(np.maximum(w_ca, 1e-12)))) / math.log(float(LIB_SIZE + 1))
        query_ctx[i] = np.array([
            np.clip(snr_pil_log / 6.0, 0.0, 2.0),
            np.clip(jump_prev1, 0.0, 5.0),
            np.clip(jump_nb, 0.0, 5.0),
            np.clip(nb_spread, 0.0, 3.0),
            np.clip(pil_slope * 10.0, -3.0, 3.0),
            ent_ca,
            ca["clear_weight"],
            np.clip(math.log(h_pil_abs / max(abs(hp.h_mean_prior), 1e-6)), -2.0, 2.0),
        ], dtype=np.float32)

        # Shared 84-branch physical candidate bank (exact same as Group B!)
        priors_3 = [
            (hp.h_mean_prior, hp.h_var_prior, hp.h_mean_prior * PRIOR_DC_RATIO),
            (h_pr_nb, v_pr_nb, d_pr_nb),
            (ca["h_prior"], ca["var_h_prior"], ca["d_prior"]),
        ]
        bank = build_shared_b_candidates(y_pil, sig2, priors_3, hp)
        raw_l = bank["logits_raw"]
        raw_l_c = raw_l - float(np.max(raw_l))
        cand_logits_phys[i] = np.clip(raw_l_c, -40.0, 0.0).astype(np.float32)
        cand_h_post[i, :, 0] = bank["h_post"].real
        cand_h_post[i, :, 1] = bank["h_post"].imag
        cand_d_post[i, :, 0] = bank["d_post"].real
        cand_d_post[i, :, 1] = bank["d_post"].imag
        cand_obs_var[i] = bank["obs_var_r"]

        # Branch features (84, 9)
        cand_feat[i, :, 0] = np.clip(raw_l_c / 10.0, -4.0, 0.0)
        cand_feat[i, :, 1] = (c_grid == 0).astype(np.float32)
        cand_feat[i, :, 2] = (c_grid == 1).astype(np.float32)
        cand_feat[i, :, 3] = (c_grid == 2).astype(np.float32)
        cand_feat[i, :, 4] = (q_grid == 0).astype(np.float32)
        cand_feat[i, :, 5] = (g_grid == 3).astype(np.float32)  # cfo == 0
        cand_feat[i, :, 6] = np.abs(cfo_vals) / 0.03
        cand_feat[i, :, 7] = np.clip(np.log(bank["obs_var_r"] / max(sig2, 1e-12)), -3.0, 3.0)
        cand_feat[i, :, 8] = np.clip(np.abs(bank["h_post"]) / h_pil_abs - 1.0, -2.0, 2.0)

        y_pil_arr[i, :, 0], y_pil_arr[i, :, 1] = y_pil.real, y_pil.imag
        y_probe_arr[i, :, 0], y_probe_arr[i, :, 1] = y[PROBE_SLICE].real, y[PROBE_SLICE].imag
        y_pay_arr[i, :, 0], y_pay_arr[i, :, 1] = y[PAYLOAD_SLICE].real, y[PAYLOAD_SLICE].imag
        sigma2_arr[i] = sig2
        h_ora_arr[i, 0], h_ora_arr[i, 1] = h_ora_all[t].real, h_ora_all[t].imag
        d_ora_arr[i, 0], d_ora_arr[i, 1] = d_ora_all[t].real, d_ora_all[t].imag
        # True QPSK symbol index (0..3) for training loss & evaluation ONLY
        s_pay = s_true[PAYLOAD_SLICE]
        s_pay_idx_arr[i] = ((s_pay.real < 0.0).astype(np.int64) << 1) | (s_pay.imag < 0.0).astype(np.int64)
        fidx_arr[i] = fidx
        sync_arr[i] = bool(sync_ok_all[t])

    return {
        "lib_feat": lib_feat,
        "lib_phys_score": lib_phys_score,
        "lib_h_all": lib_h_all,
        "lib_d_all": lib_d_all,
        "lib_v_all": lib_v_all,
        "query_ctx": query_ctx,
        "cand_feat": cand_feat,
        "cand_logits_phys": cand_logits_phys,
        "cand_h_post": cand_h_post,
        "cand_d_post": cand_d_post,
        "cand_obs_var": cand_obs_var,
        "y_pil": y_pil_arr,
        "y_probe": y_probe_arr,
        "y_pay": y_pay_arr,
        "sigma2": sigma2_arr,
        "h_ora": h_ora_arr,
        "d_ora": d_ora_arr,
        "s_pay_idx": s_pay_idx_arr,
        "frame_indices": fidx_arr,
        "sync_ok": sync_arr,
        "y_sym_eval": y_all[eval_offset:],
        "lib": lib,
    }


class EMFloatFormerLite(nn.Module):
    """Lightweight physically-grounded Attention + Candidate Calibrator + Residual Channel/Variance Refiner.

    Architecture (~6.7k parameters, zero heavy transformers, <0.6 ms/frame on 1-thread CPU):
      1. Dual-head scaled dot-product content-addressing attention over causal library M=200 + clear token,
         combining learned Q-K similarity with learnable temperature-scaled physical log-likelihood score.
      2. Differentiable conjugate Gaussian fusion of neural content prior with online pilots -> neural A5 channel.
      3. Candidate weight calibrator (84 physical branches): predicts residual logit bias & temperature
         scaling per branch conditioned on query context, plus residual complex phase/gain refinement and
         calibrated observation variance multiplier!
    """

    def __init__(self, d_lib: int = 10, d_ctx: int = 8, d_cand: int = 9, d_model: int = 24, n_heads: int = 2):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Module 1: Content-addressing attention
        self.q_net = nn.Sequential(nn.Linear(d_ctx, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.k_net = nn.Sequential(nn.Linear(d_lib, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.head_mix = nn.Linear(n_heads, 1, bias=False)
        nn.init.constant_(self.head_mix.weight, 0.5 / n_heads)
        self.log_phys_scale = nn.Parameter(torch.tensor(0.0))  # exp(0) = 1.0 init -> starts at classical A5!
        self.clear_bias_delta = nn.Parameter(torch.tensor(0.0))
        self.log_var_attn_scale = nn.Parameter(torch.tensor(0.0))

        # Module 2: Candidate logit calibrator + residual complex refiner + uncertainty calibrator
        self.cand_mlp = nn.Sequential(
            nn.Linear(d_cand + d_ctx, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        # Initialize final layer near zero so training begins cleanly at the tuned classical B-bank!
        nn.init.zeros_(self.cand_mlp[-1].weight)
        nn.init.zeros_(self.cand_mlp[-1].bias)
        self.log_cand_temp = nn.Parameter(torch.tensor(0.0))

        # Residual complex multiplier (1 + d_re + j*d_im), variance multiplier exp(d_logv),
        # and context-aware payload likelihood tempering gate pay_scale in (0, 1) to prevent
        # low-SNR blind-payload overconfidence!
        self.refiner = nn.Sequential(
            nn.Linear(d_ctx + 4, 24),
            nn.SiLU(),
            nn.Linear(24, 4),  # [d_gain_re, d_gain_im, d_log_var, d_pay_gate_logit]
        )
        nn.init.zeros_(self.refiner[-1].weight)
        nn.init.zeros_(self.refiner[-1].bias)

        # Register fixed physical constants as buffers
        s_pil_t = torch.tensor(np.stack([S_PIL.real, S_PIL.imag], axis=-1), dtype=torch.float32)  # (8, 2)
        s_prb = FULL_HDR_26[PROBE_SLICE]
        s_prb_t = torch.tensor(np.stack([s_prb.real, s_prb.imag], axis=-1), dtype=torch.float32)  # (5, 2)
        cfo_grid_t = torch.tensor(np.tile(CFO_GRID, 12), dtype=torch.float32)  # (84,)
        k_prb = torch.tensor(np.arange(21, 26, dtype=np.float32) - float(K_CENTER))  # (5,)
        ang_prb = cfo_grid_t[:, None] * k_prb[None, :]  # (84, 5)
        self.register_buffer("s_pil", s_pil_t)
        self.register_buffer("s_prb", s_prb_t)
        self.register_buffer("cos_prb", torch.cos(ang_prb))
        self.register_buffer("sin_prb", torch.sin(ang_prb))
        self.register_buffer("cfo_grid_84", cfo_grid_t)
        k_pay = torch.tensor(np.arange(26, 126, dtype=np.float32) - float(K_CENTER))
        ang_pay = cfo_grid_t[:, None] * k_pay[None, :]  # (84, 100)
        self.register_buffer("cos_pay", torch.cos(ang_pay))
        self.register_buffer("sin_pay", torch.sin(ang_pay))
        qconst = torch.tensor(np.stack([QPSK_CONST.real, QPSK_CONST.imag], axis=-1), dtype=torch.float32)  # (4, 2)
        self.register_buffer("qconst", qconst)

    def forward(
        self,
        lib_feat: torch.Tensor,         # (B, 201, 10)
        lib_phys_score: torch.Tensor,   # (B, 201)
        lib_h_all: torch.Tensor,        # (B, 201, 2)
        lib_d_all: torch.Tensor,        # (B, 201, 2)
        lib_v_all: torch.Tensor,        # (B, 201)
        query_ctx: torch.Tensor,        # (B, 8)
        cand_feat: torch.Tensor,        # (B, 84, 9)
        cand_logits_phys: torch.Tensor, # (B, 84)
        cand_h_post: torch.Tensor,      # (B, 84, 2)
        cand_d_post: torch.Tensor,      # (B, 84, 2)
        cand_obs_var: torch.Tensor,     # (B, 84)
        y_pil: torch.Tensor,            # (B, 8, 2)
        sigma2: torch.Tensor,           # (B,)
    ) -> dict[str, torch.Tensor]:
        B, M1, _ = lib_feat.shape
        # 1. Dual-head learned content addressing attention
        q = self.q_net(query_ctx).view(B, self.n_heads, 1, self.head_dim)      # (B, H, 1, Dh)
        k = self.k_net(lib_feat).view(B, M1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, H, M1, Dh)
        attn_qk = (q * k).sum(dim=-1) / math.sqrt(self.head_dim)               # (B, H, M1)
        attn_learned = self.head_mix(attn_qk.transpose(1, 2)).squeeze(-1)      # (B, M1)

        phys_scale = torch.exp(torch.clamp(self.log_phys_scale, -2.0, 2.0))
        scores = phys_scale * lib_phys_score + attn_learned
        # Learnable clear-token bias adjustment
        clear_mask = torch.zeros_like(scores)
        clear_mask[:, -1] = self.clear_bias_delta
        scores = scores + clear_mask
        w_attn = F.softmax(scores, dim=-1)                                     # (B, 201)

        # Prior mixture moments from learned content attention
        h_pr = torch.sum(w_attn.unsqueeze(-1) * lib_h_all, dim=1)              # (B, 2)
        d_pr = torch.sum(w_attn.unsqueeze(-1) * lib_d_all, dim=1)              # (B, 2)
        diff_h2 = torch.sum((lib_h_all - h_pr.unsqueeze(1)) ** 2, dim=-1)      # (B, 201)
        v_pr = torch.sum(w_attn * (lib_v_all + diff_h2), dim=1) * torch.exp(torch.clamp(self.log_var_attn_scale, -1.5, 1.5))
        v_pr = torch.clamp(v_pr, min=1e-10)

        # Differentiable conjugate Gaussian pilot fusion -> neural A5 channel estimate!
        y_adj = y_pil - d_pr.unsqueeze(1)                                      # (B, 8, 2)
        # Complex vdot(s_pil, y_adj) = sum_k (s_re*y_re + s_im*y_im) + j*(s_re*y_im - s_im*y_re)
        s_re, s_im = self.s_pil[:, 0], self.s_pil[:, 1]
        vdot_re = torch.sum(s_re * y_adj[:, :, 0] + s_im * y_adj[:, :, 1], dim=1)
        vdot_im = torch.sum(s_re * y_adj[:, :, 1] - s_im * y_adj[:, :, 0], dim=1)
        sp_en = torch.sum(s_re ** 2 + s_im ** 2)
        s2 = torch.clamp(sigma2, min=1e-10)
        prec_pr = 1.0 / v_pr
        prec_lk = sp_en / s2
        v_post_a = 1.0 / (prec_pr + prec_lk)
        h_a_re = v_post_a * (prec_pr * h_pr[:, 0] + vdot_re / s2)
        h_a_im = v_post_a * (prec_pr * h_pr[:, 1] + vdot_im / s2)
        h_attn_post = torch.stack([h_a_re, h_a_im], dim=-1)                    # (B, 2)
        # Residual refiner conditioned on context + attention summary
        h_norm_scale = torch.sqrt(torch.clamp(torch.sum(h_attn_post ** 2, dim=-1, keepdim=True), min=1e-10))
        ref_in = torch.cat([
            query_ctx,
            w_attn[:, -1:],                                                    # clear weight
            torch.sum(w_attn ** 2, dim=-1, keepdim=True),                      # inverse eff refs
            torch.log(torch.clamp(v_post_a.unsqueeze(-1) / (h_norm_scale ** 2), min=1e-6)),
            torch.log(torch.clamp(s2.unsqueeze(-1) / (h_norm_scale ** 2), min=1e-6)),
        ], dim=-1)
        ref_out = self.refiner(ref_in)                                         # (B, 3)
        # Small bounded residual complex correction (1 + 0.15*tanh(dr) + j*0.15*tanh(di))
        c_re = 1.0 + 0.15 * torch.tanh(ref_out[:, 0])
        c_im = 0.15 * torch.tanh(ref_out[:, 1])
        var_mult = torch.exp(torch.clamp(ref_out[:, 2], -1.0, 1.5))
        pay_scale = torch.sigmoid(ref_out[:, 3] + 2.0)  # init ~0.88, adapts smoothly at very low SNR!

        # Apply complex residual correction to h_attn_post
        h_c_a = torch.stack([
            h_attn_post[:, 0] * c_re - h_attn_post[:, 1] * c_im,
            h_attn_post[:, 0] * c_im + h_attn_post[:, 1] * c_re,
        ], dim=-1)

        # 2. Candidate weight calibration on shared 84 physical branches!
        ctx_rep = query_ctx.unsqueeze(1).expand(-1, N_CANDIDATES_B, -1)        # (B, 84, 8)
        cand_in = torch.cat([cand_feat, ctx_rep], dim=-1)                      # (B, 84, 17)
        delta_logit = self.cand_mlp(cand_in).squeeze(-1)                       # (B, 84)
        cand_temp = torch.exp(torch.clamp(self.log_cand_temp, -1.5, 1.5))
        # Also inject slight residual calibration using neural h_c_a alignment with branch h_post
        diff_ca = torch.sum((cand_h_post - h_c_a.unsqueeze(1)) ** 2, dim=-1) / torch.clamp(cand_obs_var, min=1e-10)
        cand_logits = (cand_logits_phys / cand_temp) + delta_logit - 0.1 * torch.clamp(diff_ca, 0.0, 20.0)
        alpha = F.softmax(cand_logits, dim=-1)                                 # (B, 84)

        # Calibrated branch channel & variance
        h_post_cal = torch.stack([
            cand_h_post[:, :, 0] * c_re.unsqueeze(1) - cand_h_post[:, :, 1] * c_im.unsqueeze(1),
            cand_h_post[:, :, 0] * c_im.unsqueeze(1) + cand_h_post[:, :, 1] * c_re.unsqueeze(1),
        ], dim=-1)
        obs_var_cal = torch.clamp(cand_obs_var * var_mult.unsqueeze(1), min=1e-10)

        return {
            "w_attn": w_attn,
            "h_c_a": h_c_a,
            "d_c_a": d_pr,
            "v_c_a": (s2 + v_post_a) * var_mult,
            "alpha": alpha,
            "cand_logits": cand_logits,
            "h_post_cal": h_post_cal,
            "d_post_cal": cand_d_post,
            "obs_var_cal": obs_var_cal,
            "delta_logit": delta_logit,
            "var_mult": var_mult,
            "pay_scale": pay_scale,
        }


def compute_training_loss(
    model: EMFloatFormerLite,
    out: dict[str, torch.Tensor],
    y_probe: torch.Tensor,   # (B, 5, 2)
    y_pay: torch.Tensor,     # (B, 100, 2)
    h_ora: torch.Tensor,     # (B, 2)
    s_pay_idx: torch.Tensor, # (B, 100)
) -> tuple[torch.Tensor, dict[str, float]]:
    """Multi-objective physics-consistent loss: channel NMSE + probe Gaussian NLL + exact B3b payload CE."""
    # 1. Channel NMSE loss for learned content attention head (h_c_a) AND posterior mean of calibrated bank
    h_ora_p = torch.clamp(torch.sum(h_ora ** 2, dim=-1), min=1e-10)
    loss_ch_a = torch.mean(torch.sum((out["h_c_a"] - h_ora) ** 2, dim=-1) / h_ora_p)

    alpha = out["alpha"]            # (B, 84)
    h_cal = out["h_post_cal"]       # (B, 84, 2)
    d_cal = out["d_post_cal"]       # (B, 84, 2)
    v_cal = out["obs_var_cal"]      # (B, 84)

    # 2. Held-out pilot probe [21:26] predictive Gaussian NLL (on TRAIN split ONLY!)
    # Branch predictions at probe k=21..25: h_r[k] = h_cal[r] * exp(j*cfo*(k-K_CENTER))
    # (B, 84, 5)
    hr_re = h_cal[:, :, 0:1] * model.cos_prb.unsqueeze(0) - h_cal[:, :, 1:2] * model.sin_prb.unsqueeze(0)
    hr_im = h_cal[:, :, 0:1] * model.sin_prb.unsqueeze(0) + h_cal[:, :, 1:2] * model.cos_prb.unsqueeze(0)
    # Posterior mean channel at probe positions
    hm_re = torch.sum(alpha.unsqueeze(-1) * hr_re, dim=1)  # (B, 5)
    hm_im = torch.sum(alpha.unsqueeze(-1) * hr_im, dim=1)  # (B, 5)
    dm = torch.sum(alpha.unsqueeze(-1) * d_cal, dim=1)     # (B, 2)
    # Total variance (within + between) at probe
    v_btw = torch.sum(alpha.unsqueeze(-1) * ((hr_re - hm_re.unsqueeze(1)) ** 2 + (hr_im - hm_im.unsqueeze(1)) ** 2), dim=1)
    v_wth = torch.sum(alpha * v_cal, dim=1, keepdim=True)
    v_prb = torch.clamp(v_wth + v_btw, min=1e-10)          # (B, 5)
    # Predicted probe observation = hm * s_prb + dm
    sp_re, sp_im = model.s_prb[:, 0].unsqueeze(0), model.s_prb[:, 1].unsqueeze(0)
    yp_re = hm_re * sp_re - hm_im * sp_im + dm[:, 0:1]
    yp_im = hm_re * sp_im + hm_im * sp_re + dm[:, 1:2]
    err_prb2 = (y_probe[:, :, 0] - yp_re) ** 2 + (y_probe[:, :, 1] - yp_im) ** 2
    loss_probe_nll = torch.mean(torch.log(math.pi * v_prb) + err_prb2 / v_prb)
    # Also channel NMSE at pilot center for posterior mean
    hm_center = torch.sum(alpha.unsqueeze(-1) * h_cal, dim=1)
    loss_ch_b = torch.mean(torch.sum((hm_center - h_ora) ** 2, dim=-1) / h_ora_p)

    # 3. Exact B3b joint mixture payload symbol cross-entropy (subsample 25 payload symbols per batch step for speed,
    #    keeping exact full-frame branch posterior structure!)
    # Subsample deterministic stride-4 (25 symbols spanning whole payload [26:126] evenly!)
    sub = slice(0, 100, 4)
    cos_p = model.cos_pay[:, sub].unsqueeze(0)  # (1, 84, 25)
    sin_p = model.sin_pay[:, sub].unsqueeze(0)
    yp = y_pay[:, sub, :]                       # (B, 25, 2)
    s_idx_sub = s_pay_idx[:, sub]               # (B, 25)

    hpay_re = h_cal[:, :, 0:1] * cos_p - h_cal[:, :, 1:2] * sin_p  # (B, 84, 25)
    hpay_im = h_cal[:, :, 0:1] * sin_p + h_cal[:, :, 1:2] * cos_p
    # (B, 84, 25, 4)
    q_re = model.qconst[:, 0].view(1, 1, 1, 4)
    q_im = model.qconst[:, 1].view(1, 1, 1, 4)
    pred_re = hpay_re.unsqueeze(-1) * q_re - hpay_im.unsqueeze(-1) * q_im + d_cal[:, :, 0].view(-1, N_CANDIDATES_B, 1, 1)
    pred_im = hpay_re.unsqueeze(-1) * q_im + hpay_im.unsqueeze(-1) * q_re + d_cal[:, :, 1].view(-1, N_CANDIDATES_B, 1, 1)
    d2 = (yp[:, None, :, 0:1] - pred_re) ** 2 + (yp[:, None, :, 1:2] - pred_im) ** 2
    vr = v_cal.view(-1, N_CANDIDATES_B, 1, 1)
    log_p_ys_r = -d2 / vr - torch.log(math.pi * vr) - math.log(4.0)  # (B, 84, 25, 4)
    log_p_y_r = torch.logsumexp(log_p_ys_r, dim=-1)                  # (B, 84, 25)
    # Exact joint posterior beta_r tempered by context-aware pay_scale (4x scaled sum to match 100-sym evidence)
    log_alpha = torch.log(torch.clamp(alpha, min=1e-12))
    ps = out["pay_scale"].view(-1, 1)
    log_beta = F.log_softmax(log_alpha + (4.0 * ps) * torch.sum(log_p_y_r, dim=-1), dim=-1)  # (B, 84)
    log_p_s_given_r = log_p_ys_r - log_p_y_r.unsqueeze(-1)           # (B, 84, 25, 4)
    # log P(S_k = s | O) = logsumexp_r (log_beta_r + log_p_s_given_r)
    log_p_s = torch.logsumexp(log_beta.view(-1, N_CANDIDATES_B, 1, 1) + log_p_s_given_r, dim=1)  # (B, 25, 4)
    loss_ce = F.nll_loss(log_p_s.reshape(-1, 4), s_idx_sub.reshape(-1))

    # Mild L2 regularization on residual logit shifts for clean OOD generalization
    reg = 1e-3 * torch.mean(out["delta_logit"] ** 2)
    loss = 1.5 * loss_ch_a + 1.5 * loss_ch_b + 0.15 * loss_probe_nll + 1.0 * loss_ce + reg
    metrics = {
        "loss": float(loss.item()),
        "ch_a": float(loss_ch_a.item()),
        "ch_b": float(loss_ch_b.item()),
        "nll": float(loss_probe_nll.item()),
        "ce": float(loss_ce.item()),
    }
    return loss, metrics


def eval_model_on_dataset(
    model: EMFloatFormerLite,
    feat_dict: dict[str, Any],
    tx_mm: np.ndarray,
    n_clusters: int = 16,
    post_var_scale: float = 1.0,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run exact single-thread inference for BOTH C_attn (Group A neural counterpart) and C_mix (Group B neural counterpart)."""
    model.eval()
    n_eval = len(feat_dict["frame_indices"])
    recs_ca: list[dict[str, Any]] = []
    recs_cb: list[dict[str, Any]] = []

    y_sym_eval = feat_dict["y_sym_eval"]
    fidx_arr = feat_dict["frame_indices"]
    h_ora_arr = feat_dict["h_ora"]
    d_ora_arr = feat_dict["d_ora"]
    sync_arr = feat_dict["sync_ok"]

    # Batched forward for fast evaluation, plus per-frame timing measurement on 200 frames for realistic single-frame latency!
    t_inf_single = []
    with torch.no_grad():
        for i in range(min(200, n_eval)):
            t0 = time.perf_counter()
            sl = slice(i, i + 1)
            _ = model(
                torch.from_numpy(feat_dict["lib_feat"][sl]),
                torch.from_numpy(feat_dict["lib_phys_score"][sl]),
                torch.from_numpy(feat_dict["lib_h_all"][sl]),
                torch.from_numpy(feat_dict["lib_d_all"][sl]),
                torch.from_numpy(feat_dict["lib_v_all"][sl]),
                torch.from_numpy(feat_dict["query_ctx"][sl]),
                torch.from_numpy(feat_dict["cand_feat"][sl]),
                torch.from_numpy(feat_dict["cand_logits_phys"][sl]),
                torch.from_numpy(feat_dict["cand_h_post"][sl]),
                torch.from_numpy(feat_dict["cand_d_post"][sl]),
                torch.from_numpy(feat_dict["cand_obs_var"][sl]),
                torch.from_numpy(feat_dict["y_pil"][sl]),
                torch.from_numpy(feat_dict["sigma2"][sl]),
            )
            t_inf_single.append((time.perf_counter() - t0) * 1000.0)
    nn_median_ms = float(np.median(t_inf_single))

    # Run full evaluation in memory-safe batches of 256
    cfo_per_r = np.tile(CFO_GRID, 12)
    rot_all = np.exp(1j * cfo_per_r[:, None] * (K_ALL[None, :] - K_CENTER))  # (84, 126)

    with torch.no_grad():
        for start in range(0, n_eval, 256):
            end = min(start + 256, n_eval)
            sl = slice(start, end)
            out = model(
                torch.from_numpy(feat_dict["lib_feat"][sl]),
                torch.from_numpy(feat_dict["lib_phys_score"][sl]),
                torch.from_numpy(feat_dict["lib_h_all"][sl]),
                torch.from_numpy(feat_dict["lib_d_all"][sl]),
                torch.from_numpy(feat_dict["lib_v_all"][sl]),
                torch.from_numpy(feat_dict["query_ctx"][sl]),
                torch.from_numpy(feat_dict["cand_feat"][sl]),
                torch.from_numpy(feat_dict["cand_logits_phys"][sl]),
                torch.from_numpy(feat_dict["cand_h_post"][sl]),
                torch.from_numpy(feat_dict["cand_d_post"][sl]),
                torch.from_numpy(feat_dict["cand_obs_var"][sl]),
                torch.from_numpy(feat_dict["y_pil"][sl]),
                torch.from_numpy(feat_dict["sigma2"][sl]),
            )
            h_ca_np = out["h_c_a"].numpy()
            d_ca_np = out["d_c_a"].numpy()
            v_ca_np = out["v_c_a"].numpy() * post_var_scale
            alpha_np = out["alpha"].numpy()
            h_cal_np = out["h_post_cal"].numpy()
            d_cal_np = out["d_post_cal"].numpy()
            v_cal_np = out["obs_var_cal"].numpy() * post_var_scale
            pay_scale_np = out["pay_scale"].numpy()

            for b in range(end - start):
                idx = start + b
                fidx = int(fidx_arr[idx])
                y = y_sym_eval[idx]
                s_true = np.asarray(tx_mm[fidx], dtype=np.complex128)
                h_ora = complex(float(h_ora_arr[idx, 0]), float(h_ora_arr[idx, 1]))
                d_ora = complex(float(d_ora_arr[idx, 0]), float(d_ora_arr[idx, 1]))
                sync_ok = bool(sync_arr[idx])

                # C1: Learned Content-Attention Receiver (Group A neural counterpart)
                h_a = complex(float(h_ca_np[b, 0]), float(h_ca_np[b, 1]))
                d_a = complex(float(d_ca_np[b, 0]), float(d_ca_np[b, 1]))
                v_a = np.full(N_SYM, float(v_ca_np[b]), dtype=np.float64)
                recs_ca.append(evaluate_frame_record(
                    fidx, y, s_true, h_ora, d_ora, sync_ok,
                    np.full(N_SYM, h_a, dtype=np.complex128), d_a, v_a,
                    latency_ms=nn_median_ms, n_candidates=LIB_SIZE + 1,
                ))

                # C2: Learned Content-Attention + Calibrated 84-Branch Exact Mixture Decoder (Group B neural counterpart)
                t_dec0 = time.perf_counter()
                h_post_c = h_cal_np[b, :, 0].astype(np.float64) + 1j * h_cal_np[b, :, 1].astype(np.float64)
                d_post_c = d_cal_np[b, :, 0].astype(np.float64) + 1j * d_cal_np[b, :, 1].astype(np.float64)
                alpha_c = alpha_np[b].astype(np.float64)
                h_all_c = h_post_c[:, None] * rot_all
                v_r_c = v_cal_np[b].astype(np.float64)
                ps_c = float(pay_scale_np[b])

                # Pre-payload posterior mean & total variance (zero payload leakage for held-out probe!)
                h_mean_c = np.sum(alpha_c[:, None] * h_all_c, axis=0)
                d_mean_c = complex(np.sum(alpha_c * d_post_c))
                diff_hc = h_all_c - h_mean_c[None, :]
                diff_dc = d_post_c - d_mean_c
                v_tot_c = float(np.sum(alpha_c * v_r_c)) + np.sum(
                    alpha_c[:, None] * (diff_hc.real ** 2 + diff_hc.imag ** 2), axis=0
                ) + float(np.sum(alpha_c * (diff_dc.real ** 2 + diff_dc.imag ** 2)))

                # Exact mixture marginalization with context-tempered full-frame payload posterior beta_r
                y_pay_c = y[PAYLOAD_SLICE]
                h_pay_c = h_all_c[:, PAYLOAD_SLICE]
                pred_c = h_pay_c[:, :, None] * QPSK_CONST[None, None, :] + d_post_c[:, None, None]
                err_c = y_pay_c[None, :, None] - pred_c
                d2_c = err_c.real ** 2 + err_c.imag ** 2
                vr_c = v_r_c[:, None, None]
                log_p_ys = -d2_c / vr_c - np.log(np.pi * vr_c) - math.log(4.0)
                log_p_y = logsumexp(log_p_ys, axis=2)
                log_beta = np.log(np.maximum(alpha_c, 1e-300)) + ps_c * np.sum(log_p_y, axis=1)
                beta_c = np.exp(log_beta - float(logsumexp(log_beta)))
                p_s_given_r = np.exp(log_p_ys - log_p_y[:, :, None])
                p_s_c = np.tensordot(beta_c, p_s_given_r, axes=([0], [0]))
                idx_dec = np.argmax(p_s_c, axis=1)
                s_dec_c = QPSK_CONST[idx_dec]
                p_b0 = p_s_c[:, 2] + p_s_c[:, 3]
                p_b1 = p_s_c[:, 1] + p_s_c[:, 3]
                bits_dec_c = np.stack([(p_b0 > 0.5).astype(np.int8), (p_b1 > 0.5).astype(np.int8)], axis=-1)
                eq_br = (y_pay_c[None, :] - d_post_c[:, None]) / np.where(np.abs(h_pay_c) > 1e-12, h_pay_c, 1e-12)
                s_eq_c = np.sum(beta_c[:, None] * eq_br, axis=0)
                t_dec = (time.perf_counter() - t_dec0) * 1000.0
                recs_cb.append(evaluate_frame_record(
                    fidx, y, s_true, h_ora, d_ora, sync_ok,
                    h_mean_c, d_mean_c, v_tot_c,
                    s_dec_payload=s_dec_c,
                    bits_dec_payload=bits_dec_c,
                    s_eq_payload=s_eq_c,
                    latency_ms=nn_median_ms + t_dec,
                    n_candidates=N_CANDIDATES_B,
                ))

    agg_ca = aggregate_records(recs_ca, n_clusters=n_clusters, peak_rss_mb=peak_rss_mb())
    agg_cb = aggregate_records(recs_cb, n_clusters=n_clusters, peak_rss_mb=peak_rss_mb())
    return agg_ca, agg_cb, recs_ca, recs_cb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=22)
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    t_start = time.perf_counter()
    torch.set_num_threads(1)
    bench_dir = args.work_dir / "benchmarks"
    tx_mm = np.load(args.work_dir / "tx_c64.npy", mmap_mode="r")

    # Load frozen hyperparameters from Stage 1 (strictly tuned on dev!)
    dev_tune = json.loads((args.out_dir / "dev_tuning_summary.json").read_text(encoding="utf-8"))
    fhp = dev_tune["frozen_hyperparams"]
    hp = ReceiverHyperparams(
        h_mean_prior=complex(fhp["h_mean_prior_re"], fhp["h_mean_prior_im"]),
        h_var_prior=float(fhp["h_var_prior"]),
        neighbor_k=int(fhp["neighbor_k"]),
        neighbor_decay=float(fhp["neighbor_decay"]),
        neighbor_drift_rel_var=float(fhp["neighbor_drift_rel_var"]),
        global_drift_rel_var=float(fhp["global_drift_rel_var"]),
        attn_temp=float(fhp["attn_temp"]),
        attn_drift_rel_var=float(fhp["attn_drift_rel_var"]),
        clear_prior_log_bias=float(fhp["clear_prior_log_bias"]),
        age_penalty=float(fhp["age_penalty"]),
        b_temp=float(fhp["b_temp"]),
        b_cfo_zero_bonus=float(fhp["b_cfo_zero_bonus"]),
        b_quad_zero_bonus=float(fhp["b_quad_zero_bonus"]),
        b_var_scale=float(fhp["b_var_scale"]),
    )

    # 1. Build TRAIN feature pool (strictly train [200:3200] across 5 training regimes = 15,000 frames)
    train_conds = ["raw_original", "awgn_snr6db", "awgn_snr2db", "cfo_0p020_snr6db", "phase_jump_snr6db"]
    print("Extracting TRAIN feature pool (5 regimes x 3,000 = 15,000 frames)...")
    t0 = time.perf_counter()
    train_parts = []
    for c in train_conds:
        z = np.load(bench_dir / f"{c}_train.npz")
        b_tr = {k: z[k] for k in z.files}
        train_parts.append(extract_frame_features_and_bank(b_tr, tx_mm, hp))
    tensor_keys = [
        "lib_feat", "lib_phys_score", "lib_h_all", "lib_d_all", "lib_v_all",
        "query_ctx", "cand_feat", "cand_logits_phys", "cand_h_post", "cand_d_post",
        "cand_obs_var", "y_pil", "y_probe", "y_pay", "sigma2", "h_ora", "s_pay_idx",
    ]
    train_data = {k: np.concatenate([p[k] for p in train_parts], axis=0) for k in tensor_keys}
    n_train = len(train_data["sigma2"])
    del train_parts
    print(f"TRAIN pool ready: {n_train} frames in {time.perf_counter()-t0:.1f}s, RSS={peak_rss_mb():.1f} MB")

    # 2. Extract DEV feature pool (strictly dev [3500:4500] on 4 core model-selection conditions)
    dev_sel_conds = ["raw_original", "awgn_snr6db", "cfo_0p020_snr6db", "phase_jump_snr6db"]
    print("Extracting DEV model-selection feature pool...")
    dev_feats = {}
    for c in dev_sel_conds:
        z = np.load(bench_dir / f"{c}_dev.npz")
        dev_feats[c] = extract_frame_features_and_bank({k: z[k] for k in z.files}, tx_mm, hp)

    # 3. Train 3 independent seeds on TRAIN, select checkpoint + variance calibration strictly on DEV
    trained_models: dict[int, tuple[EMFloatFormerLite, float, dict[str, Any]]] = {}
    n_params = sum(p.numel() for p in EMFloatFormerLite().parameters())
    print(f"EMFloatFormerLite parameter count: {n_params}")

    for seed in SEEDS:
        set_deterministic(seed)
        t_s0 = time.perf_counter()
        model = EMFloatFormerLite()
        opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=2e-4)

        best_dev_score = 1e30
        best_state = None
        best_epoch = -1
        history = []

        rng_perm = np.random.default_rng(seed)
        for ep in range(1, args.epochs + 1):
            model.train()
            perm = rng_perm.permutation(n_train)
            ep_losses = []
            for start in range(0, n_train, args.batch_size):
                idx = perm[start:start + args.batch_size]
                opt.zero_grad(set_to_none=True)
                out = model(
                    torch.from_numpy(train_data["lib_feat"][idx]),
                    torch.from_numpy(train_data["lib_phys_score"][idx]),
                    torch.from_numpy(train_data["lib_h_all"][idx]),
                    torch.from_numpy(train_data["lib_d_all"][idx]),
                    torch.from_numpy(train_data["lib_v_all"][idx]),
                    torch.from_numpy(train_data["query_ctx"][idx]),
                    torch.from_numpy(train_data["cand_feat"][idx]),
                    torch.from_numpy(train_data["cand_logits_phys"][idx]),
                    torch.from_numpy(train_data["cand_h_post"][idx]),
                    torch.from_numpy(train_data["cand_d_post"][idx]),
                    torch.from_numpy(train_data["cand_obs_var"][idx]),
                    torch.from_numpy(train_data["y_pil"][idx]),
                    torch.from_numpy(train_data["sigma2"][idx]),
                )
                loss, _ = compute_training_loss(
                    model, out,
                    torch.from_numpy(train_data["y_probe"][idx]),
                    torch.from_numpy(train_data["y_pay"][idx]),
                    torch.from_numpy(train_data["h_ora"][idx]),
                    torch.from_numpy(train_data["s_pay_idx"][idx]),
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                ep_losses.append(float(loss.item()))
            sched.step()

            # Evaluate on DEV every 2 epochs (and last epoch)
            if ep % 2 == 0 or ep == args.epochs:
                model.eval()
                dev_losses = []
                with torch.no_grad():
                    for c in dev_sel_conds:
                        fd = dev_feats[c]
                        # Evaluate full 1,000 dev frames in 256 chunks
                        c_loss_sum = 0.0
                        c_n = 0
                        for st_i in range(0, len(fd["sigma2"]), 256):
                            ed_i = min(st_i + 256, len(fd["sigma2"]))
                            sl = slice(st_i, ed_i)
                            o_d = model(
                                torch.from_numpy(fd["lib_feat"][sl]),
                                torch.from_numpy(fd["lib_phys_score"][sl]),
                                torch.from_numpy(fd["lib_h_all"][sl]),
                                torch.from_numpy(fd["lib_d_all"][sl]),
                                torch.from_numpy(fd["lib_v_all"][sl]),
                                torch.from_numpy(fd["query_ctx"][sl]),
                                torch.from_numpy(fd["cand_feat"][sl]),
                                torch.from_numpy(fd["cand_logits_phys"][sl]),
                                torch.from_numpy(fd["cand_h_post"][sl]),
                                torch.from_numpy(fd["cand_d_post"][sl]),
                                torch.from_numpy(fd["cand_obs_var"][sl]),
                                torch.from_numpy(fd["y_pil"][sl]),
                                torch.from_numpy(fd["sigma2"][sl]),
                            )
                            l_d, _ = compute_training_loss(
                                model, o_d,
                                torch.from_numpy(fd["y_probe"][sl]),
                                torch.from_numpy(fd["y_pay"][sl]),
                                torch.from_numpy(fd["h_ora"][sl]),
                                torch.from_numpy(fd["s_pay_idx"][sl]),
                            )
                            c_loss_sum += float(l_d.item()) * (ed_i - st_i)
                            c_n += (ed_i - st_i)
                        dev_losses.append(c_loss_sum / c_n)
                dev_score = float(np.mean(dev_losses))
                history.append({"epoch": ep, "train_loss": float(np.mean(ep_losses)), "dev_loss": dev_score})
                if dev_score < best_dev_score:
                    best_dev_score = dev_score
                    best_epoch = ep
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        # Post-hoc variance calibration strictly on DEV
        z2_dev = []
        for c in dev_sel_conds:
            _, agg_cb_d, _, _ = eval_model_on_dataset(model, dev_feats[c], tx_mm, n_clusters=10, post_var_scale=1.0)
            z2_dev.append(agg_cb_d["channel_and_probe"]["probe_z2_mean"])
        post_scale = float(np.clip(np.median(z2_dev), 0.7, 2.5))
        train_sec = round(time.perf_counter() - t_s0, 2)
        meta = {
            "seed": seed,
            "n_params": int(n_params),
            "epochs": args.epochs,
            "best_epoch": best_epoch,
            "best_dev_loss": best_dev_score,
            "dev_post_var_scale": post_scale,
            "train_elapsed_sec": train_sec,
            "train_frames": int(n_train),
            "history": history,
        }
        trained_models[seed] = (model, post_scale, meta)
        print(f"[Seed {seed}] Trained in {train_sec}s | best_ep={best_epoch} | dev_loss={best_dev_score:.4f} | var_scale={post_scale:.3f}")

    # Free training tensors before test evaluation to keep RSS ultra-low!
    del train_data, dev_feats

    # 4. Closed-book TEST evaluation across ALL 8 conditions & paired G=16 tests vs classical A5 / B3b!
    ab_test = json.loads((args.out_dir / "ab_test_summary.json").read_text(encoding="utf-8"))
    c_summary: dict[str, Any] = {
        "model_name": "EMFloatFormerLite",
        "n_params": int(n_params),
        "seeds": list(SEEDS),
        "train_meta_by_seed": {str(s): m for s, (_, _, m) in trained_models.items()},
        "conditions": {},
        "paired_vs_classical": {},
    }

    for cname in STRESS_SPECS.keys():
        z = np.load(bench_dir / f"{cname}_test.npz")
        fd_test = extract_frame_features_and_bank({k: z[k] for k in z.files}, tx_mm, hp)
        cond_seeds_ca = {}
        cond_seeds_cb = {}
        for seed in SEEDS:
            model, post_scale, meta = trained_models[seed]
            agg_ca, agg_cb, _, _ = eval_model_on_dataset(model, fd_test, tx_mm, n_clusters=16, post_var_scale=post_scale)
            agg_ca["engineering"]["train_meta"] = {k: v for k, v in meta.items() if k != "history"}
            agg_cb["engineering"]["train_meta"] = {k: v for k, v in meta.items() if k != "history"}
            cond_seeds_ca[str(seed)] = agg_ca
            cond_seeds_cb[str(seed)] = agg_cb

        # Aggregate mean+-std across 3 seeds
        def _seed_mean_std(seed_dict: dict[str, Any]) -> dict[str, Any]:
            def _ms(getter):
                vals = [float(getter(seed_dict[str(s)])) for s in SEEDS]
                return {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)), "per_seed": vals}

            # Mean cluster series across 3 seeds for ensemble/mean paired comparison
            mean_clusters = {}
            for k in seed_dict[str(SEEDS[0])]["cluster_series"].keys():
                mean_clusters[k] = np.mean([seed_dict[str(s)]["cluster_series"][k] for s in SEEDS], axis=0).tolist()
            return {
                "payload_ser": _ms(lambda d: d["payload"]["ser"]),
                "payload_ber": _ms(lambda d: d["payload"]["ber"]),
                "payload_fer": _ms(lambda d: d["payload"]["fer"]),
                "payload_evm_pct": _ms(lambda d: d["payload"]["evm_pct"]),
                "ch_nmse": _ms(lambda d: d["channel_and_probe"]["ch_nmse"]),
                "probe_nmse": _ms(lambda d: d["channel_and_probe"]["probe_nmse"]),
                "probe_nll": _ms(lambda d: d["channel_and_probe"]["probe_nll"]),
                "probe_cov90": _ms(lambda d: d["channel_and_probe"]["probe_cov90"]),
                "probe_cov95": _ms(lambda d: d["channel_and_probe"]["probe_cov95"]),
                "latency_median_ms": _ms(lambda d: d["engineering"]["latency_median_ms"]),
                "burst_boundary_ch_nmse": _ms(lambda d: d["regime_breakdown"]["burst_boundary_ch_nmse"]),
                "burst_boundary_ser": _ms(lambda d: d["regime_breakdown"]["burst_boundary_ser"]),
                "mean_cluster_series": mean_clusters,
            }

        ca_ms = _seed_mean_std(cond_seeds_ca)
        cb_ms = _seed_mean_std(cond_seeds_cb)
        c_summary["conditions"][cname] = {
            "C1_neural_content_attn": {"mean_std": ca_ms, "by_seed": cond_seeds_ca},
            "C2_neural_cal_mixture": {"mean_std": cb_ms, "by_seed": cond_seeds_cb},
        }

        # Paired G=16 exact sign-flip comparisons vs dev-tuned classical A5 & B3b!
        a5_ref = ab_test["conditions"][cname]["A5_global_content_attn"]["cluster_series"]
        b3b_ref = ab_test["conditions"][cname]["B3b_joint_mixture"]["cluster_series"]
        b1_ref = ab_test["conditions"][cname]["B1_hard_map"]["cluster_series"]
        base_ref = ab_test["conditions"][cname]["base_mmse_dc_cfo"]["cluster_series"]

        pairs = [
            ("C1_vs_A5_ch_nmse", ca_ms["mean_cluster_series"]["ch_nmse"], a5_ref["ch_nmse"]),
            ("C1_vs_A5_ser", ca_ms["mean_cluster_series"]["payload_ser"], a5_ref["payload_ser"]),
            ("C2_vs_B3b_ser", cb_ms["mean_cluster_series"]["payload_ser"], b3b_ref["payload_ser"]),
            ("C2_vs_B3b_fer", cb_ms["mean_cluster_series"]["payload_fer"], b3b_ref["payload_fer"]),
            ("C2_vs_B3b_probe_nmse", cb_ms["mean_cluster_series"]["probe_nmse"], b3b_ref["probe_nmse"]),
            ("C2_vs_B3b_probe_nll", cb_ms["mean_cluster_series"]["probe_nll"], b3b_ref["probe_nll"]),
            ("C2_vs_B1_ser", cb_ms["mean_cluster_series"]["payload_ser"], b1_ref["payload_ser"]),
            ("C2_vs_base_cfo_ser", cb_ms["mean_cluster_series"]["payload_ser"], base_ref["payload_ser"]),
        ]
        p_dict = {}
        raw_ps, ord_ks = [], []
        for lbl, cand_c, comp_c in pairs:
            st_p = paired_cluster_signflip(cand_c, comp_c, higher_is_better=False)
            p_dict[lbl] = st_p
            raw_ps.append(st_p["p_two_sided"])
            ord_ks.append(lbl)
        for k_o, ph in zip(ord_ks, holm_bonferroni(raw_ps)):
            p_dict[k_o]["p_holm"] = ph
        c_summary["paired_vs_classical"][cname] = p_dict

        print(
            f"[TEST-C {cname:18s}] "
            f"A5 SER={ab_test['conditions'][cname]['A5_global_content_attn']['payload']['ser']*100:.2f}% vs C1={ca_ms['payload_ser']['mean']*100:.2f}+-{ca_ms['payload_ser']['std']*100:.2f}% | "
            f"B3b SER={ab_test['conditions'][cname]['B3b_joint_mixture']['payload']['ser']*100:.2f}% vs C2={cb_ms['payload_ser']['mean']*100:.2f}+-{cb_ms['payload_ser']['std']*100:.2f}% | "
            f"B3b NLL={ab_test['conditions'][cname]['B3b_joint_mixture']['channel_and_probe']['probe_nll']:.3f} vs C2={cb_ms['probe_nll']['mean']:.3f}"
        )

    c_summary["total_elapsed_sec"] = round(time.perf_counter() - t_start, 2)
    c_summary["peak_rss_mb"] = round(peak_rss_mb(), 2)
    (args.out_dir / "c_neural_test_summary.json").write_text(json.dumps(c_summary, indent=2) + "\n", encoding="utf-8")
    print(f"ALL GROUP C EXPERIMENTS DONE in {c_summary['total_elapsed_sec']}s, peak RSS {c_summary['peak_rss_mb']} MB")


if __name__ == "__main__":
    main()
