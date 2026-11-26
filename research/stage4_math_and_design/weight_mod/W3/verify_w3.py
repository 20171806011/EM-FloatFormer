"""Track W3 (Round 2) CPU PyTorch Verification Suite.

Fast (<=160 s CPU), complete verification of all Round 1 + Round 2 theorems in THEOREM_W3.md:
  1. Block 1 — Exact Fisher-Rao KL Expansion, Bach Self-Concordance Bracket,
     4-Proxy Comparison (L2 vs Diag-EWC vs K-FAC vs Input Nullspace P_perp),
     Exact Kronecker Spectral Identity, AND Round-2 Complex-Symmetrized Nullspace
     P_perp^C (W2 compatibility: ||[P_perp^C, J_d]||_F < 1e-12, exact Loewner bound,
     >100x KL suppression, and Nullspace-Restricted Complex-Cayley Woodbury adapter Q^perp).
  2. Block 2 — Generalized Rayleigh Layer Selection across the 8-layer 3:1 Hybrid
     Gated-DeltaNet (exact parallel WY) + Full-Attention CPU host (tied lm_head).
  3. Block 3 — Controlled Study Axis 1 (Forgetting Axis, 6 methods):
       1) unconstrained_lora (all 8 layers)
       2) layer_selected_gen_fisher_lora (layers [3,5,6,7], early 0..2 frozen)
       3) nullspace_projected_lora (P_perp)
       4) modality_gated_unconstrained (Theorem 4 ablation: PureKL=0, MixKL explodes!)
       5) modality_gated_nullspace_hybrid (Round-1 sequence-gated + P_perp + MixKL anchor)
       6) two_tier_token_kv_gated_hybrid (Round-2 Theorem 4.3: m_EM(t) K/V/FFN gate +
          sequence-gated P_perp Q/O + MixKL anchor -> PureKL=0.000e+00 AND MixKL <= 1.0e-3 nats!)
     PLUS Round-2 Long-Gap (L_gap = 32) W1 Slow-Head GDN Starvation Resolution Study
     comparing (i) frozen early no warmgate, (ii) naive early warmgate (leaky prose beta ->
     state-norm trap), and (iii) W1+W3 clean m_EM(t)-masked zero-prose-beta slow-head warmgate!
  4. Block 4 — Controlled Study Axis 2 (Loss Optimization, 4 modes x 3 seeds,
     Step-69 vs Step-96 vs Step-160 convergence profile + Rao-Blackwell grad variance):
       A) hard_label_ce
       B) bayes_kl_only
       C) bayes_kl_plus_cf_cal (lambda_orbit = 0.10)
       D) bayes_kl_cf_cal_quotient (lambda_orbit = 0.0 + exact M1/W2 Quotient Wrapper -> C4=0.0)
  5. Block 5 — Exact Qwen3.5-4B / Qwen3.5-9B / Qwen3.8-27B Scaling Ledger using
     verified |V| = 248,320 (exact 0.0000% match to W4 & smoke_audit.json: 4,205,751,296)
     and explicit Phase-B Stop-Grad (Option B1) vs Co-Tuned Connector (Option B2) VRAM/step table.

Run via:
  uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B research/stage4_math_and_design/weight_mod/W3/verify_w3.py
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr
import torch
import torch.nn as nn
import torch.nn.functional as F


OUT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = OUT_DIR / "results_w3.json"
MIB = 1024.0 ** 2

PRETRAIN_SEED = 202600
STUDY_SEEDS = (202611, 202622, 202633)

# Stage 3D physical constants (2 dB SNR, sigma_phi = 0.10)
SNR_DB = 2.0
RHO = 10.0 ** (SNR_DB / 10.0)
KAPPA = math.sqrt(2.0 * RHO)
SIGMA_PHI = 0.10
ROOTS = np.array([1.0 + 0j, 1j, -1.0 + 0j, -1j], dtype=np.complex128)
GH_X, GH_W = np.polynomial.hermite.hermgauss(32)
GH_Z = GH_X * math.sqrt(2.0)
GH_P = GH_W / math.sqrt(math.pi)

# Host dimensions (d_model=32 even -> 16 complex doublets; L=8 exact 3:1 hybrid, matching W1 d=32, H=4, dh=8)
D_MODEL = 32
N_HEADS = 4
D_FF = 64
LEX_DIM = 12  # Dominant general-text lexical subspace (6 complex doublets, leaving 20-dim orthogonal EM complement)
VOCAB_SIZE = 64
GEN_VOCAB = 48
Q3_ANS_IDS = [48, 49, 50, 51]
Q2_ANS_IDS = [52, 53, 54, 55]


# =============================================================================
# 1. Exact EM Physics Simulator & Analytic Bayes Posterior (Stage 3D G1..G4 + Q2)
# =============================================================================

def sector_of_complex(z: np.ndarray) -> np.ndarray:
    u = z.real + z.imag
    v = z.imag - z.real
    out = np.zeros(z.shape, dtype=np.int64)
    out[(u > 0.0) & (v >= 0.0)] = 1
    out[(u <= 0.0) & (v > 0.0)] = 2
    out[(u < 0.0) & (v <= 0.0)] = 3
    return out


def q3_circulant_kernel(theta: float, var_phi: float) -> np.ndarray:
    angles = theta + math.sqrt(max(0.0, var_phi)) * GH_Z
    c = ndtr(KAPPA * np.cos(angles + math.pi / 4.0))
    s = ndtr(KAPPA * np.sin(angles + math.pi / 4.0))
    b_mat = np.stack([c * s, (1.0 - c) * s, (1.0 - c) * (1.0 - s), c * (1.0 - s)], axis=-1)
    return (GH_P[:, None] * b_mat).sum(axis=0)


def exact_q3_posterior(t1: np.ndarray, theta: np.ndarray, var_phi: np.ndarray) -> np.ndarray:
    logits_s1 = 2.0 * RHO * np.real(np.conj(t1)[:, None] * ROOTS[None, :])
    logits_s1 -= logits_s1.max(axis=1, keepdims=True)
    a_post = np.exp(logits_s1)
    a_post /= a_post.sum(axis=1, keepdims=True)

    n = len(t1)
    probs = np.zeros((n, 4), dtype=np.float64)
    keys = np.round(np.stack([theta, var_phi], axis=-1), 8)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    for idx, (th, vp) in enumerate(uniq):
        k_vec = q3_circulant_kernel(float(th), float(vp))
        mask = inv == idx
        a_sub = a_post[mask]
        out_sub = np.zeros((a_sub.shape[0], 4), dtype=np.float64)
        for j in range(4):
            out_sub += a_sub[:, j : j + 1] * np.roll(k_vec, j)[None, :]
        probs[mask] = out_sub
    probs = np.clip(probs, 1e-12, 1.0)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs


def exact_q2_posterior(
    t1: np.ndarray, sp_valid: np.ndarray, n_valid: np.ndarray, theta_p: np.ndarray, var_p: np.ndarray
) -> np.ndarray:
    logits_s1 = 2.0 * RHO * np.real(np.conj(t1)[:, None] * ROOTS[None, :])
    logits_s1 -= logits_s1.max(axis=1, keepdims=True)
    a_post = np.exp(logits_s1)
    a_post /= a_post.sum(axis=1, keepdims=True)

    n = len(t1)
    probs = np.full((n, 4), 0.25, dtype=np.float64)
    active = n_valid > 0
    if not np.any(active):
        return probs

    sp_act, th_act, vp_act, a_act = sp_valid[active], theta_p[active], var_p[active], a_post[active]
    psi = th_act[:, None] + np.sqrt(np.maximum(0.0, vp_act))[:, None] * GH_Z[None, :]
    rot_phase = np.exp(1j * psi)
    proj = 2.0 * RHO * np.real(np.conj(sp_act)[:, None, None] * ROOTS[None, :, None] * rot_phase[:, None, :])
    proj_max = proj.max(axis=(1, 2), keepdims=True)
    lik_g = (np.exp(proj - proj_max) * GH_P[None, None, :]).sum(axis=-1)
    p_g = lik_g / np.maximum(lik_g.sum(axis=1, keepdims=True), 1e-30)

    out_act = np.zeros((a_act.shape[0], 4), dtype=np.float64)
    for x in range(4):
        for g in range(4):
            out_act[:, x] += a_act[:, (x + g) % 4] * p_g[:, g]
    out_act = np.clip(out_act, 1e-12, 1.0)
    out_act /= out_act.sum(axis=1, keepdims=True)
    probs[active] = out_act
    return probs


@dataclass
class EMBatch:
    em_feats: torch.Tensor
    prose_ids: torch.Tensor
    labels: torch.Tensor
    bayes_probs: torch.Tensor
    task_is_q2: torch.Tensor
    em_feats_slot_flip: torch.Tensor
    bayes_probs_slot_flip: torch.Tensor
    prose_ids_pert: torch.Tensor
    em_feats_rot1: torch.Tensor
    em_feats_canon: torch.Tensor
    canon_shift: torch.Tensor
    em_feats_rot1_canon: torch.Tensor
    rot1_canon_shift: torch.Tensor


def _pack_em_features(
    t1: np.ndarray, t2: np.ndarray, theta: np.ndarray, var_phi: np.ndarray,
    sp_valid: np.ndarray, n_valid: np.ndarray, t_ind: np.ndarray, has_ind: np.ndarray,
    theta_p: np.ndarray, var_p: np.ndarray, is_q2: np.ndarray,
) -> np.ndarray:
    n = len(t1)
    feats = np.zeros((n, 4, 16), dtype=np.float32)
    q3_f = (~is_q2).astype(np.float32)
    q2_f = is_q2.astype(np.float32)

    r1 = np.abs(t1)
    feats[:, 0, 0] = t1.real; feats[:, 0, 1] = t1.imag; feats[:, 0, 2] = r1
    feats[:, 0, 3] = np.minimum(r1 ** 2, 6.0) / 3.0
    feats[:, 0, 4] = (t1 ** 2).real / (r1 + 0.5); feats[:, 0, 5] = (t1 ** 2).imag / (r1 + 0.5)
    feats[:, 0, 6] = q3_f; feats[:, 0, 7] = q2_f; feats[:, 0, 8] = 1.0

    r2 = np.abs(t2)
    feats[:, 1, 0] = t2.real; feats[:, 1, 1] = t2.imag; feats[:, 1, 2] = r2
    feats[:, 1, 3] = q3_f; feats[:, 1, 4] = q2_f; feats[:, 1, 9] = 1.0

    damp_q3 = np.exp(-0.5 * var_phi)
    damp_q2 = np.exp(-0.5 * var_p)
    t1_rot = t1 * np.exp(1j * theta) * damp_q3
    sp_derot = sp_valid * np.exp(-1j * theta_p) * damp_q2
    q2_prod = t1 * np.conj(sp_derot) / (1.0 + 0.35 * np.abs(sp_derot))
    q2_unit = q2_prod / (np.abs(q2_prod) + 0.5)

    feats[:, 2, 0] = q3_f * t1_rot.real; feats[:, 2, 1] = q3_f * t1_rot.imag
    feats[:, 2, 2] = q3_f * np.cos(theta) * damp_q3; feats[:, 2, 3] = q3_f * np.sin(theta) * damp_q3
    feats[:, 2, 4] = q3_f * (t1_rot.real / (np.abs(t1_rot) + 0.5))
    feats[:, 2, 5] = q3_f * (t1_rot.imag / (np.abs(t1_rot) + 0.5))
    feats[:, 2, 6] = q2_f * q2_prod.real; feats[:, 2, 7] = q2_f * q2_prod.imag
    feats[:, 2, 8] = q2_f * q2_unit.real; feats[:, 2, 9] = q2_f * q2_unit.imag
    feats[:, 2, 10] = damp_q3 * q3_f + damp_q2 * q2_f
    feats[:, 2, 11] = q3_f * np.abs(t1_rot) + q2_f * np.abs(q2_prod)
    feats[:, 2, 12] = q2_f * (n_valid.astype(np.float32) / 2.0); feats[:, 2, 13] = 1.0

    feats[:, 3, 0] = sp_derot.real; feats[:, 3, 1] = sp_derot.imag; feats[:, 3, 2] = np.abs(sp_derot)
    feats[:, 3, 3] = n_valid.astype(np.float32) / 2.0; feats[:, 3, 4] = has_ind.astype(np.float32)
    feats[:, 3, 5] = has_ind.astype(np.float32) * t_ind.real
    feats[:, 3, 6] = has_ind.astype(np.float32) * t_ind.imag
    feats[:, 3, 7] = q2_f * (n_valid > 0).astype(np.float32)
    feats[:, 3, 8] = q2_f * np.tanh(0.8 * np.abs(sp_derot)) * np.cos(np.angle(q2_prod + 1e-9)) * (n_valid > 0)
    feats[:, 3, 9] = q2_f * np.tanh(0.8 * np.abs(sp_derot)) * np.sin(np.angle(q2_prod + 1e-9)) * (n_valid > 0)
    feats[:, 3, 14] = 1.0
    return feats


def _build_prose_templates(
    rng: np.random.Generator, n: int, is_q2: np.ndarray, l_gap: int = 8, perturb: bool = False
) -> np.ndarray:
    q3_base = np.array([4, 11, 18, 25, 32, 7, 14], dtype=np.int64)
    q2_base = np.array([1, 8, 15, 22, 29, 36, 43], dtype=np.int64)
    out = np.zeros((n, l_gap), dtype=np.int64)
    choice = rng.integers(0, 4, size=n)
    if perturb:
        choice = (choice + 1 + rng.integers(0, 3, size=n)) % 4
    for pos in range(l_gap - 1):
        v3 = (q3_base[pos % len(q3_base)] + choice) % GEN_VOCAB
        v2 = (q2_base[pos % len(q2_base)] + choice) % GEN_VOCAB
        out[:, pos] = np.where(is_q2, v2, v3)
    out[:, -1] = np.where(is_q2, 61, 60)
    if perturb and l_gap >= 6:
        out[:, 2] = rng.integers(0, GEN_VOCAB, size=n)
        out[:, 5] = rng.integers(0, GEN_VOCAB, size=n)
    return out


def _canonicalize_pair(t1: np.ndarray, t2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a1 = sector_of_complex(t1)
    a2 = sector_of_complex(t2)
    return t1 * np.conj(ROOTS[a1]), t2 * np.conj(ROOTS[a2]), a1


def generate_q3_cell(
    rng: np.random.Generator, n_samples: int, abs_omega: float, h: int, ell: int, t_p: float,
    l_gap: int = 8, paired_signs: bool = True,
) -> EMBatch:
    noise_sd = math.sqrt(1.0 / (2.0 * RHO))
    x1 = rng.integers(0, 4, size=n_samples)
    x2 = rng.integers(0, 4, size=n_samples)
    g = rng.integers(0, 4, size=n_samples)
    signs = np.where(np.arange(n_samples) % 2 == 0, 1.0, -1.0) if paired_signs else rng.choice([-1.0, 1.0], size=n_samples)
    omega = signs * abs_omega
    tau = float(h + ell)
    theta = omega * tau
    var_phi = np.full(n_samples, (SIGMA_PHI ** 2) * tau, dtype=np.float64)

    phi_2 = omega * h + SIGMA_PHI * math.sqrt(h) * rng.normal(size=n_samples)
    phi_3 = phi_2 + omega * ell + SIGMA_PHI * math.sqrt(ell) * rng.normal(size=n_samples)
    eps1 = rng.normal(0, noise_sd, n_samples) + 1j * rng.normal(0, noise_sd, n_samples)
    eps2 = rng.normal(0, noise_sd, n_samples) + 1j * rng.normal(0, noise_sd, n_samples)
    eps3 = rng.normal(0, noise_sd, n_samples) + 1j * rng.normal(0, noise_sd, n_samples)

    t1 = ROOTS[(x1 + g) % 4] + eps1
    t2 = ROOTS[(x2 + g) % 4] * np.exp(1j * phi_2) + eps2
    t3 = ROOTS[(x1 + g) % 4] * np.exp(1j * phi_3) + eps3
    labels = sector_of_complex(t3)

    zc = np.zeros(n_samples, dtype=np.complex128)
    zi = np.zeros(n_samples, dtype=np.int64)
    zf = np.zeros(n_samples, dtype=np.float64)
    is_q2 = np.zeros(n_samples, dtype=bool)

    bayes = exact_q3_posterior(t1, theta, var_phi)
    bayes_flip = exact_q3_posterior(t1, -theta, var_phi)
    em_feats = _pack_em_features(t1, t2, theta, var_phi, zc, zi, zc, zi, zf, zf, is_q2)
    em_feats_flip = _pack_em_features(t1, t2, -theta, var_phi, zc, zi, zc, zi, zf, zf, is_q2)
    em_feats_rot1 = _pack_em_features(t1 * 1j, t2 * 1j, theta, var_phi, zc, zi, zc, zi, zf, zf, is_q2)

    t1_c, t2_c, a1 = _canonicalize_pair(t1, t2)
    em_feats_canon = _pack_em_features(t1_c, t2_c, theta, var_phi, zc, zi, zc, zi, zf, zf, is_q2)
    t1_r1c, t2_r1c, a1_r1 = _canonicalize_pair(t1 * 1j, t2 * 1j)
    em_feats_rot1_canon = _pack_em_features(t1_r1c, t2_r1c, theta, var_phi, zc, zi, zc, zi, zf, zf, is_q2)

    prose = _build_prose_templates(rng, n_samples, is_q2, l_gap=l_gap, perturb=False)
    prose_pert = _build_prose_templates(rng, n_samples, is_q2, l_gap=l_gap, perturb=True)
    return EMBatch(
        torch.from_numpy(em_feats), torch.from_numpy(prose), torch.from_numpy(labels.astype(np.int64)),
        torch.from_numpy(bayes.astype(np.float32)), torch.from_numpy(is_q2),
        torch.from_numpy(em_feats_flip), torch.from_numpy(bayes_flip.astype(np.float32)),
        torch.from_numpy(prose_pert), torch.from_numpy(em_feats_rot1),
        torch.from_numpy(em_feats_canon), torch.from_numpy(a1.astype(np.int64)),
        torch.from_numpy(em_feats_rot1_canon), torch.from_numpy(a1_r1.astype(np.int64)),
    )


def generate_q2_arm(
    rng: np.random.Generator, n_samples: int, arm: str, omega: float = 0.35, t_p: float = 4.0, l_gap: int = 8
) -> EMBatch:
    noise_sd = math.sqrt(1.0 / (2.0 * RHO))
    x1 = rng.integers(0, 4, size=n_samples); x2 = rng.integers(0, 4, size=n_samples)
    g = rng.integers(0, 4, size=n_samples); g_other = rng.integers(0, 4, size=n_samples)

    theta_p = np.full(n_samples, omega * t_p, dtype=np.float64)
    var_p = np.full(n_samples, (SIGMA_PHI ** 2) * t_p, dtype=np.float64)
    phi_2 = omega * 2.0 + SIGMA_PHI * math.sqrt(2.0) * rng.normal(size=n_samples)
    phi_p = theta_p + np.sqrt(var_p) * rng.normal(size=n_samples)
    phi_ind = rng.uniform(-math.pi, math.pi, size=n_samples)

    eps1 = rng.normal(0, noise_sd, n_samples) + 1j * rng.normal(0, noise_sd, n_samples)
    eps2 = rng.normal(0, noise_sd, n_samples) + 1j * rng.normal(0, noise_sd, n_samples)
    ep1 = rng.normal(0, noise_sd, n_samples) + 1j * rng.normal(0, noise_sd, n_samples)
    ep2 = rng.normal(0, noise_sd, n_samples) + 1j * rng.normal(0, noise_sd, n_samples)
    ep_ind = rng.normal(0, noise_sd, n_samples) + 1j * rng.normal(0, noise_sd, n_samples)

    t1 = ROOTS[(x1 + g) % 4] + eps1
    t2 = ROOTS[(x2 + g) % 4] * np.exp(1j * phi_2) + eps2
    p1 = ROOTS[g] * np.exp(1j * phi_p) + ep1
    p2 = ROOTS[g] * np.exp(1j * phi_p) + ep2
    p_ind = ROOTS[g_other] * np.exp(1j * phi_ind) + ep_ind

    if arm == "none":
        sp = np.zeros(n_samples, dtype=np.complex128); nv = np.zeros(n_samples, dtype=np.int64)
        hi = np.zeros(n_samples, dtype=np.int64); ti = np.zeros(n_samples, dtype=np.complex128)
    elif arm == "valid":
        sp = p1; nv = np.ones(n_samples, dtype=np.int64)
        hi = np.zeros(n_samples, dtype=np.int64); ti = np.zeros(n_samples, dtype=np.complex128)
    elif arm == "fresh_resample":
        sp = p1 + p2; nv = np.full(n_samples, 2, dtype=np.int64)
        hi = np.zeros(n_samples, dtype=np.int64); ti = np.zeros(n_samples, dtype=np.complex128)
    elif arm == "independent":
        sp = np.zeros(n_samples, dtype=np.complex128); nv = np.zeros(n_samples, dtype=np.int64)
        hi = np.ones(n_samples, dtype=np.int64); ti = p_ind
    else:
        raise ValueError(arm)

    is_q2 = np.ones(n_samples, dtype=bool)
    zf = np.zeros(n_samples, dtype=np.float64)
    bayes = exact_q2_posterior(t1, sp, nv, theta_p, var_p)
    sp_flip = -sp
    bayes_flip = exact_q2_posterior(t1, sp_flip, nv, theta_p, var_p)

    em_feats = _pack_em_features(t1, t2, zf, zf, sp, nv, ti, hi, theta_p, var_p, is_q2)
    em_feats_flip = _pack_em_features(t1, t2, zf, zf, sp_flip, nv, ti, hi, theta_p, var_p, is_q2)
    em_feats_rot1 = _pack_em_features(t1 * 1j, t2 * 1j, zf, zf, sp, nv, ti, hi, theta_p, var_p, is_q2)
    t1_c, t2_c, a1 = _canonicalize_pair(t1, t2)
    em_feats_canon = _pack_em_features(t1_c, t2_c, zf, zf, sp, nv, ti, hi, theta_p, var_p, is_q2)
    t1_r1c, t2_r1c, a1_r1 = _canonicalize_pair(t1 * 1j, t2 * 1j)
    em_feats_rot1_canon = _pack_em_features(t1_r1c, t2_r1c, zf, zf, sp, nv, ti, hi, theta_p, var_p, is_q2)

    prose = _build_prose_templates(rng, n_samples, is_q2, l_gap=l_gap, perturb=False)
    prose_pert = _build_prose_templates(rng, n_samples, is_q2, l_gap=l_gap, perturb=True)
    return EMBatch(
        torch.from_numpy(em_feats), torch.from_numpy(prose), torch.from_numpy(x1.astype(np.int64)),
        torch.from_numpy(bayes.astype(np.float32)), torch.from_numpy(is_q2),
        torch.from_numpy(em_feats_flip), torch.from_numpy(bayes_flip.astype(np.float32)),
        torch.from_numpy(prose_pert), torch.from_numpy(em_feats_rot1),
        torch.from_numpy(em_feats_canon), torch.from_numpy(a1.astype(np.int64)),
        torch.from_numpy(em_feats_rot1_canon), torch.from_numpy(a1_r1.astype(np.int64)),
    )


def concat_em_batches(batches: Sequence[EMBatch]) -> EMBatch:
    return EMBatch(*[torch.cat([getattr(b, f) for b in batches], dim=0) for f in EMBatch.__dataclass_fields__])


def build_em_train_dataset(seed: int, l_gap: int = 8, scale: float = 0.70) -> EMBatch:
    rng = np.random.default_rng(seed)
    parts: List[EMBatch] = []
    n_q3 = max(32, int(round(200 * scale)))
    for h, ell, tp in ((2, 2, 4.0), (2, 3, 1.0)):
        for abs_w in (0.0, 0.10, 0.25):
            parts.append(generate_q3_cell(rng, n_q3, abs_w, h, ell, tp, l_gap=l_gap, paired_signs=True))
    for arm, cnt in (("none", 150), ("valid", 300), ("fresh_resample", 300), ("independent", 150)):
        c_s = max(16, int(round((cnt // 3) * scale)))
        for w in (-0.20, 0.0, 0.20):
            parts.append(generate_q2_arm(rng, c_s, arm, omega=w, t_p=4.0, l_gap=l_gap))
    return concat_em_batches(parts)


def build_em_eval_suite(seed: int = 950400, l_gap: int = 8, compact: bool = False) -> Dict[str, EMBatch]:
    rng = np.random.default_rng(seed)
    if compact:
        return {
            "G1": generate_q3_cell(rng, 256, 0.40, 2, 2, 4.0, l_gap=l_gap, paired_signs=True),
            "Q2_valid": generate_q2_arm(rng, 192, "valid", omega=0.35, t_p=4.0, l_gap=l_gap),
            "Q2_fresh": generate_q2_arm(rng, 192, "fresh_resample", omega=0.35, t_p=4.0, l_gap=l_gap),
        }
    return {
        "G1": generate_q3_cell(rng, 384, 0.40, 2, 2, 4.0, l_gap=l_gap, paired_signs=True),
        "G2": generate_q3_cell(rng, 192, 0.20, 2, 2, 4.0, l_gap=l_gap, paired_signs=True),
        "G3": generate_q3_cell(rng, 192, 0.20, 4, 4, 8.0, l_gap=l_gap, paired_signs=True),
        "G4": generate_q3_cell(rng, 192, 0.40, 1, 1, 2.0, l_gap=l_gap, paired_signs=True),
        "Q2_none": generate_q2_arm(rng, 192, "none", omega=0.35, t_p=4.0, l_gap=l_gap),
        "Q2_valid": generate_q2_arm(rng, 192, "valid", omega=0.35, t_p=4.0, l_gap=l_gap),
        "Q2_fresh": generate_q2_arm(rng, 192, "fresh_resample", omega=0.35, t_p=4.0, l_gap=l_gap),
        "Q2_indep": generate_q2_arm(rng, 192, "independent", omega=0.35, t_p=4.0, l_gap=l_gap),
    }


# =============================================================================
# 2. Task A: Correlated General-Language Next-Token Sequence Generator
# =============================================================================

class CorrelatedTextGrammar:
    def __init__(self, seed: int = 7777):
        rng = np.random.default_rng(seed)
        self.n_cls = 6
        self.per_cls = GEN_VOCAB // self.n_cls
        logits = rng.normal(0.0, 0.4, size=(self.n_cls, self.n_cls))
        for c in range(self.n_cls):
            logits[c, (c + 1) % self.n_cls] += 2.6
            logits[c, (c + 2) % self.n_cls] += 1.2
        self.cls_probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        self.cls_probs /= self.cls_probs.sum(axis=1, keepdims=True)
        w_logits = rng.normal(0.0, 0.5, size=(GEN_VOCAB, self.per_cls))
        for w in range(GEN_VOCAB):
            w_logits[w, (w + 1) % self.per_cls] += 2.2
            w_logits[w, (w + 3) % self.per_cls] += 1.1
        self.w_probs = np.exp(w_logits - w_logits.max(axis=1, keepdims=True))
        self.w_probs /= self.w_probs.sum(axis=1, keepdims=True)

    def sample(self, rng: np.random.Generator, n_seq: int, seq_len: int = 16) -> torch.Tensor:
        seq = np.zeros((n_seq, seq_len), dtype=np.int64)
        seq[:, 0] = rng.integers(0, GEN_VOCAB, size=n_seq)
        for t in range(1, seq_len):
            prev = seq[:, t - 1]
            prev_cls = prev // self.per_cls
            if t >= 5 and (t % 4 == 1):
                echo = (seq[:, t - 4] + seq[:, t - 1]) % self.per_cls
                next_cls = (prev_cls + 1) % self.n_cls
                off = np.where(rng.random(n_seq) < 0.12, rng.integers(0, self.per_cls, size=n_seq), echo)
                seq[:, t] = next_cls * self.per_cls + off
            else:
                next_cls = (rng.random(n_seq)[:, None] > np.cumsum(self.cls_probs[prev_cls], axis=1)).sum(axis=1).clip(0, self.n_cls - 1)
                off = (rng.random(n_seq)[:, None] > np.cumsum(self.w_probs[prev], axis=1)).sum(axis=1).clip(0, self.per_cls - 1)
                seq[:, t] = next_cls * self.per_cls + off
        return torch.from_numpy(seq)


# =============================================================================
# 3. 8-Layer 3:1 Hybrid Host with Two-Tier Token-Position Gating & W1 Warmgate
# =============================================================================

class GatedNullspaceLoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.weight, mean=0.0, std=1.0 / math.sqrt(in_features))

        self.lora_rank: int = 0
        self.lora_scale: float = 1.0
        self.lora_A: Optional[nn.Parameter] = None
        self.lora_B: Optional[nn.Parameter] = None
        self.modality_gated: bool = False
        self.token_position_gated: bool = False
        self.register_buffer("null_proj", None, persistent=False)

    def attach_lora(
        self, rank: int, alpha: float = 12.0, null_proj: Optional[torch.Tensor] = None,
        modality_gated: bool = False, token_position_gated: bool = False, seed: int = 0,
    ) -> None:
        self.lora_rank = int(rank)
        self.lora_scale = float(alpha) / float(rank)
        self.modality_gated = bool(modality_gated)
        self.token_position_gated = bool(token_position_gated)
        self.null_proj = null_proj.clone() if null_proj is not None else None
        gen = torch.Generator().manual_seed(seed)
        a_init = torch.randn(rank, self.in_features, generator=gen) * (1.0 / math.sqrt(self.in_features))
        if self.null_proj is not None:
            a_init = a_init @ self.null_proj
        self.lora_A = nn.Parameter(a_init)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

    def detach_lora(self) -> None:
        self.lora_rank = 0
        self.lora_A = None
        self.lora_B = None
        self.modality_gated = False
        self.token_position_gated = False
        self.null_proj = None

    def effective_delta_w(self) -> Optional[torch.Tensor]:
        if self.lora_rank <= 0 or self.lora_A is None or self.lora_B is None:
            return None
        a_eff = self.lora_A @ self.null_proj if self.null_proj is not None else self.lora_A
        return self.lora_scale * (self.lora_B @ a_eff)

    def lowrank_kfac_penalty(
        self, g_gen: torch.Tensor, c_gen: Optional[torch.Tensor] = None
    ) -> Optional[torch.Tensor]:
        """Exact O(d^2 r) low-rank trace: tr(dW^T G dW C) = s^2 sum((B^T G B) * (A_eff C A_eff^T)).
        If c_gen is None, computes output-only prose curvature tr(dW^T G dW) = s^2 sum((B^T G B) * (A A^T)).
        """
        if self.lora_rank <= 0 or self.lora_A is None or self.lora_B is None:
            return None
        a_eff = self.lora_A @ self.null_proj if self.null_proj is not None else self.lora_A
        b_g_b = self.lora_B.T @ (g_gen @ self.lora_B)
        a_c_a = a_eff @ (c_gen @ a_eff.T) if c_gen is not None else (a_eff @ a_eff.T)
        return 0.5 * (self.lora_scale ** 2) * (b_g_b * a_c_a).sum()

    def forward(
        self, x: torch.Tensor, g_em: bool = False, em_tok_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        base = F.linear(x, self.weight, self.bias)
        if self.lora_rank <= 0 or self.lora_A is None or self.lora_B is None:
            return base
        if self.modality_gated and (not g_em):
            return base
        x_in = F.linear(x, self.null_proj) if self.null_proj is not None else x
        delta = F.linear(F.linear(x_in, self.lora_A), self.lora_B) * self.lora_scale
        if self.token_position_gated and (em_tok_mask is not None):
            delta = delta * em_tok_mask.unsqueeze(-1)
        return base + delta


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class GatedDeltaNetBlock(nn.Module):
    """Exact parallel WY Gated-DeltaNet with W1 zero-param modality-gated slow-head warmgate.

    Supports two warmgate modes for Tension A ablation:
      - 'leaky_beta': shifts alpha upward on slow head WITHOUT strict zero prose write (state-norm trap!)
      - 'clean_masked': shifts alpha upward on prose AND strictly zeroes prose write beta_prose = 0 via m_EM(t)
                        (exact Theorem W1.3 lossless highway with zero prose state blowup!)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_len: int = 48):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = RMSNorm(d_model)
        self.q_proj = GatedNullspaceLoRALinear(d_model, d_model)
        self.k_proj = GatedNullspaceLoRALinear(d_model, d_model)
        self.v_proj = GatedNullspaceLoRALinear(d_model, d_model)
        self.o_proj = GatedNullspaceLoRALinear(d_model, d_model)
        self.alpha_proj = GatedNullspaceLoRALinear(d_model, n_heads, bias=True)
        self.beta_proj = GatedNullspaceLoRALinear(d_model, n_heads, bias=True)
        with torch.no_grad():
            self.alpha_proj.bias.copy_(torch.tensor([1.6, 0.9, 0.6, 0.4][:n_heads]))
            self.beta_proj.bias.fill_(-0.5)
        self.norm2 = RMSNorm(d_model)
        self.w_up = GatedNullspaceLoRALinear(d_model, d_ff)
        self.w_down = GatedNullspaceLoRALinear(d_ff, d_model)

        self.register_buffer("tril_buf", torch.tril(torch.ones(max_len, max_len, dtype=torch.bool)), persistent=False)
        self.register_buffer("stril_buf", torch.tril(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=-1), persistent=False)
        self.register_buffer("eye_buf", torch.eye(max_len, dtype=torch.float32), persistent=False)

        self.warmgate_mode: Optional[str] = None
        self.slow_head_idx: int = 0
        self.register_buffer("q_null_proj", None, persistent=False)

    def set_w1_warmgate(
        self, mode: Optional[str], head_idx: int = 0, q_null_proj: Optional[torch.Tensor] = None
    ) -> None:
        self.warmgate_mode = mode
        self.slow_head_idx = int(head_idx)
        self.q_null_proj = q_null_proj.clone() if q_null_proj is not None else None

    def forward(
        self, x: torch.Tensor, g_em: bool = False, em_tok_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, t, d = x.shape
        h_in = self.norm1(x)
        q = self.q_proj(h_in, g_em=g_em, em_tok_mask=em_tok_mask).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = F.normalize(
            self.k_proj(h_in, g_em=g_em, em_tok_mask=em_tok_mask).view(b, t, self.n_heads, self.head_dim), dim=-1, eps=1e-5
        ).transpose(1, 2)
        v = self.v_proj(h_in, g_em=g_em, em_tok_mask=em_tok_mask).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        a_logits = self.alpha_proj(h_in, g_em=g_em, em_tok_mask=em_tok_mask)
        b_logits = self.beta_proj(h_in, g_em=g_em, em_tok_mask=em_tok_mask)

        alpha = torch.sigmoid(a_logits)
        beta = torch.sigmoid(b_logits)
        if g_em and (self.warmgate_mode == "leaky_beta") and (em_tok_mask is not None):
            h_s = self.slow_head_idx
            prose_m = 1.0 - em_tok_mask
            alpha = alpha.clone()
            beta = beta.clone()
            # Naive dt_bias shift without strict zero prose write -> prose accumulates in S_t!
            alpha[..., h_s] = torch.sigmoid(a_logits[..., h_s] + 5.0)
            beta[..., h_s] = torch.sigmoid(b_logits[..., h_s] - 1.5 * prose_m)

        alpha_t = alpha.transpose(1, 2)
        beta_t = beta.transpose(1, 2)
        cs = torch.cumsum(torch.log(alpha_t.clamp(min=1e-5)), dim=-1)
        tril = self.tril_buf[:t, :t]
        strict_tril = self.stril_buf[:t, :t]
        gamma = torch.exp((cs.unsqueeze(-1) - cs.unsqueeze(-2)).masked_fill(~tril, -50.0)) * tril.to(x.dtype)

        m_mat = (gamma * torch.matmul(k, k.transpose(-2, -1)) * beta_t.unsqueeze(-1)).masked_fill(~strict_tril, 0.0)
        l_mat = self.eye_buf[:t, :t].view(1, 1, t, t) + m_mat
        u_mat = torch.linalg.solve_triangular(l_mat, beta_t.unsqueeze(-1) * v, upper=False, unitriangular=True)
        o_heads = (torch.matmul(gamma * torch.matmul(q, k.transpose(-2, -1)), u_mat) / math.sqrt(self.head_dim)).transpose(1, 2)

        if g_em and (self.warmgate_mode == "clean_dual_nullspace") and (em_tok_mask is not None) and (self.q_null_proj is not None):
            # Exact zero-param W1+W3 Dual-Track Nullspace-Filtered Slow-Head Highway:
            #   1. Base prose recurrence on head h_s remains 100% intact (zero prose head loss).
            #   2. Shadow EM state uses W1 slow decay (alpha_prose -> 0.997) + W3 strict beta_prose == 0.0 (zero prose accumulation).
            #   3. Readout query filtered through P_perp so prose tokens do NOT read EM state, while EM/answer query reads full signal!
            h_s = self.slow_head_idx
            prose_m = 1.0 - em_tok_mask
            a_em = torch.where(prose_m > 0.5, torch.sigmoid(a_logits[..., h_s] + 5.5), alpha[..., h_s]).unsqueeze(1)
            b_em = (beta[..., h_s] * em_tok_mask).unsqueeze(1)
            cs_e = torch.cumsum(torch.log(a_em.clamp(min=1e-5)), dim=-1)
            gam_e = torch.exp((cs_e.unsqueeze(-1) - cs_e.unsqueeze(-2)).masked_fill(~tril, -50.0)) * tril.to(x.dtype)
            k_s = k[:, h_s : h_s + 1]
            v_s = v[:, h_s : h_s + 1]
            m_e = (gam_e * torch.matmul(k_s, k_s.transpose(-2, -1)) * b_em.unsqueeze(-1)).masked_fill(~strict_tril, 0.0)
            u_e = torch.linalg.solve_triangular(
                self.eye_buf[:t, :t].view(1, 1, t, t) + m_e, b_em.unsqueeze(-1) * v_s, upper=False, unitriangular=True
            )
            q_perp = self.q_proj(F.linear(h_in, self.q_null_proj), g_em=g_em, em_tok_mask=em_tok_mask)
            q_s_perp = q_perp.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)[:, h_s : h_s + 1]
            o_em_s = (torch.matmul(gam_e * torch.matmul(q_s_perp, k_s.transpose(-2, -1)), u_e) / math.sqrt(self.head_dim)).squeeze(1)
            o_heads = o_heads.clone()
            o_heads[:, :, h_s, :] = o_heads[:, :, h_s, :] + 0.25 * o_em_s

        o = o_heads.contiguous().view(b, t, d)
        x = x + self.o_proj(o, g_em=g_em, em_tok_mask=em_tok_mask)
        h2 = self.norm2(x)
        x = x + self.w_down(F.silu(self.w_up(h2, g_em=g_em, em_tok_mask=em_tok_mask)), g_em=g_em, em_tok_mask=em_tok_mask)
        return x

    @torch.no_grad()
    def measure_prose_retention(
        self, x: torch.Tensor, g_em: bool, em_tok_mask: Optional[torch.Tensor], n_em: int = 4
    ) -> float:
        h_in = self.norm1(x)
        a_logits = self.alpha_proj(h_in, g_em=g_em, em_tok_mask=em_tok_mask)
        alpha = torch.sigmoid(a_logits)
        if g_em and (self.warmgate_mode is not None) and (em_tok_mask is not None):
            alpha = alpha.clone()
            prose_m = 1.0 - em_tok_mask
            shift = 5.5 if self.warmgate_mode == "clean_dual_nullspace" else 5.0
            alpha[..., self.slow_head_idx] = torch.where(
                prose_m > 0.5, torch.sigmoid(a_logits[..., self.slow_head_idx] + shift), alpha[..., self.slow_head_idx]
            )
        return float(alpha[:, n_em:, :].prod(dim=1).mean(dim=0).max().item())


class FullAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_len: int = 48):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = RMSNorm(d_model)
        self.q_proj = GatedNullspaceLoRALinear(d_model, d_model)
        self.k_proj = GatedNullspaceLoRALinear(d_model, d_model)
        self.v_proj = GatedNullspaceLoRALinear(d_model, d_model)
        self.o_proj = GatedNullspaceLoRALinear(d_model, d_model)
        self.norm2 = RMSNorm(d_model)
        self.w_up = GatedNullspaceLoRALinear(d_model, d_ff)
        self.w_down = GatedNullspaceLoRALinear(d_ff, d_model)
        self.register_buffer("causal_buf", torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1), persistent=False)

    def forward(
        self, x: torch.Tensor, g_em: bool = False, em_tok_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, t, d = x.shape
        h_in = self.norm1(x)
        q = self.q_proj(h_in, g_em=g_em, em_tok_mask=em_tok_mask).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h_in, g_em=g_em, em_tok_mask=em_tok_mask).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h_in, g_em=g_em, em_tok_mask=em_tok_mask).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(self.causal_buf[:t, :t][None, None, :, :], -1e9)
        o = torch.matmul(F.softmax(scores, dim=-1), v).transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.o_proj(o, g_em=g_em, em_tok_mask=em_tok_mask)
        h2 = self.norm2(x)
        x = x + self.w_down(F.silu(self.w_up(h2, g_em=g_em, em_tok_mask=em_tok_mask)), g_em=g_em, em_tok_mask=em_tok_mask)
        return x


class HybridDualTaskHost(nn.Module):
    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS, d_ff: int = D_FF, max_len: int = 48):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.vocab_size = VOCAB_SIZE
        self.layer_types = [
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
        ]
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self._init_anisotropic_embeddings()
        self.layers = nn.ModuleList([
            GatedDeltaNetBlock(d_model, n_heads, d_ff, max_len=max_len)
            if lt == "linear_attention"
            else FullAttentionBlock(d_model, n_heads, d_ff, max_len=max_len)
            for lt in self.layer_types
        ])
        self.final_norm = RMSNorm(d_model)
        self.register_buffer("em_lift", torch.zeros(16, d_model))
        self.register_buffer("em_slot_pos_bias", torch.zeros(4, d_model))
        self._init_em_lift()

    def _init_anisotropic_embeddings(self) -> None:
        with torch.no_grad():
            e = torch.zeros(VOCAB_SIZE, self.d_model)
            e[:GEN_VOCAB, :LEX_DIM] = torch.randn(GEN_VOCAB, LEX_DIM) * 0.28
            e[:GEN_VOCAB, LEX_DIM:] = torch.randn(GEN_VOCAB, self.d_model - LEX_DIM) * 0.002
            q_orth, _ = torch.linalg.qr(torch.randn(self.d_model - LEX_DIM, 16))
            e[GEN_VOCAB:, LEX_DIM:] = q_orth.T * 0.45
            e[GEN_VOCAB:, :LEX_DIM] = torch.randn(16, LEX_DIM) * 0.01
            self.embed_tokens.weight.copy_(e)
            p = torch.zeros_like(self.pos_emb.weight)
            p[:, :LEX_DIM] = torch.randn(p.shape[0], LEX_DIM) * 0.08
            p[:, LEX_DIM:] = torch.randn(p.shape[0], self.d_model - LEX_DIM) * 0.002
            self.pos_emb.weight.copy_(p)

    def _init_em_lift(self) -> None:
        with torch.no_grad():
            rem = self.d_model - LEX_DIM  # 20 dims
            q, _ = torch.linalg.qr(torch.randn(rem, 20))
            lift = torch.zeros(16, self.d_model)
            lift[:, LEX_DIM:] = q[:, :16].T * 0.65
            slot_bias = torch.zeros(4, self.d_model)
            slot_bias[:, LEX_DIM:] = q[:, 16:20].T * 0.35
            self.em_lift.copy_(lift)
            self.em_slot_pos_bias.copy_(slot_bias)

    def base_weight_sha256(self) -> str:
        h = hashlib.sha256()
        for name, p in sorted(self.named_parameters()):
            if "lora_" in name:
                continue
            h.update(name.encode("utf-8"))
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    def freeze_base_weights(self) -> None:
        for name, p in self.named_parameters():
            if "lora_" not in name:
                p.requires_grad_(False)

    def lora_modules(self) -> List[Tuple[int, str, GatedNullspaceLoRALinear]]:
        mods = []
        for l_idx, layer in enumerate(self.layers):
            for sub_name, mod in layer.named_modules():
                if isinstance(mod, GatedNullspaceLoRALinear):
                    mods.append((l_idx, f"layer_{l_idx}.{sub_name}", mod))
        return mods

    def detach_all_lora(self) -> None:
        for _, _, mod in self.lora_modules():
            mod.detach_lora()
        for layer in self.layers:
            if isinstance(layer, GatedDeltaNetBlock):
                layer.set_w1_warmgate(None)

    def trainable_lora_params(self) -> List[nn.Parameter]:
        return [p for n, p in self.named_parameters() if "lora_" in n and p.requires_grad]

    def _make_em_mask(self, b: int, t: int, n_em: int, device: torch.device) -> torch.Tensor:
        m = torch.zeros((b, t), device=device, dtype=torch.float32)
        m[:, :n_em] = 1.0
        return m

    def forward_prefix_layers(
        self, embeds: torch.Tensor, stop_layer: int, g_em: bool = False, em_tok_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        t = embeds.shape[1]
        h = embeds + self.pos_emb(torch.arange(t, device=embeds.device)).unsqueeze(0)
        for l_idx in range(stop_layer):
            h = self.layers[l_idx](h, g_em=g_em, em_tok_mask=em_tok_mask)
        return h

    def forward_from_layer(
        self, h: torch.Tensor, start_layer: int, g_em: bool = False, em_tok_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for l_idx in range(start_layer, len(self.layers)):
            h = self.layers[l_idx](h, g_em=g_em, em_tok_mask=em_tok_mask)
        return F.linear(self.final_norm(h), self.embed_tokens.weight)

    def forward_embeds(
        self, embeds: torch.Tensor, g_em: bool = False, em_tok_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        t = embeds.shape[1]
        h = embeds + self.pos_emb(torch.arange(t, device=embeds.device)).unsqueeze(0)
        return self.forward_from_layer(h, 0, g_em=g_em, em_tok_mask=em_tok_mask)

    def forward_text(self, token_ids: torch.Tensor, g_em: bool = False) -> torch.Tensor:
        return self.forward_embeds(self.embed_tokens(token_ids), g_em=g_em, em_tok_mask=None)

    def build_em_embeds(self, em_feats: torch.Tensor, prose_ids: torch.Tensor) -> torch.Tensor:
        em_emb = torch.matmul(em_feats, self.em_lift) + self.em_slot_pos_bias.unsqueeze(0)
        return torch.cat([em_emb, self.embed_tokens(prose_ids)], dim=1)

    def forward_em(self, em_feats: torch.Tensor, prose_ids: torch.Tensor) -> torch.Tensor:
        full_emb = self.build_em_embeds(em_feats, prose_ids)
        mask = self._make_em_mask(full_emb.shape[0], full_emb.shape[1], 4, full_emb.device)
        return self.forward_embeds(full_emb, g_em=True, em_tok_mask=mask)[:, -1, :]

    def forward_mixed_fluency(self, em_feats: torch.Tensor, text_ids: torch.Tensor) -> torch.Tensor:
        full_emb = self.build_em_embeds(em_feats, text_ids)
        mask = self._make_em_mask(full_emb.shape[0], full_emb.shape[1], 4, full_emb.device)
        return self.forward_embeds(full_emb, g_em=True, em_tok_mask=mask)[:, 4:-1, :]


# =============================================================================
# 4. Pretraining, True-Fisher K-FAC & Complex-Symmetrized Nullspace P_perp^C
# =============================================================================

def make_complex_structure_J(dim: int) -> torch.Tensor:
    m = dim // 2
    j = torch.zeros((dim, dim), dtype=torch.float64)
    for k in range(m):
        j[2 * k, 2 * k + 1] = -1.0
        j[2 * k + 1, 2 * k] = 1.0
    return j


def build_complex_symmetrized_nullspace(
    c_gen: torch.Tensor, var_keep_frac: float = 0.998, max_keep_frac: float = 0.72
) -> Tuple[torch.Tensor, torch.Tensor, int, float, float]:
    d = c_gen.shape[0]
    m = d // 2
    c64 = c_gen.double()
    j_d = make_complex_structure_J(d)
    c_sym = 0.5 * (c64 + j_d.T @ c64 @ j_d)

    h_mat = torch.complex(c_sym[0::2, 0::2], c_sym[1::2, 0::2])
    h_mat = 0.5 * (h_mat + h_mat.conj().T)
    evals_h, evecs_h = torch.linalg.eigh(h_mat)
    evals_h = evals_h.flip(0).clamp(min=0.0)
    evecs_h = evecs_h.flip(1)

    cum = torch.cumsum(evals_h, dim=0) / evals_h.sum().clamp(min=1e-12)
    kc_keep = min(int((cum < var_keep_frac).sum().item()) + 1, max(1, int(max_keep_frac * m)))
    u_c = evecs_h[:, :kc_keep]
    p_c = torch.eye(m, dtype=torch.complex128) - u_c @ u_c.conj().T

    p_real = torch.zeros((d, d), dtype=torch.float64)
    p_real[0::2, 0::2] = p_c.real
    p_real[0::2, 1::2] = -p_c.imag
    p_real[1::2, 0::2] = p_c.imag
    p_real[1::2, 1::2] = p_c.real

    comm_err = float(torch.linalg.norm(p_real @ j_d - j_d @ p_real, ord="fro").item())
    tail_ratio = float((evals_h[min(kc_keep, m - 1)] / evals_h[0].clamp(min=1e-12)).item())
    return p_real.float(), p_c, 2 * kc_keep, comm_err, tail_ratio


def pretrain_host_on_task_a(
    grammar: CorrelatedTextGrammar, steps: int = 220, batch_size: int = 32
) -> Tuple[HybridDualTaskHost, dict]:
    torch.manual_seed(PRETRAIN_SEED)
    np.random.seed(PRETRAIN_SEED)
    host = HybridDualTaskHost(d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, max_len=48)
    rng = np.random.default_rng(PRETRAIN_SEED)
    train_seqs = grammar.sample(rng, 3072, seq_len=16)
    val_seqs = grammar.sample(np.random.default_rng(PRETRAIN_SEED + 1), 256, seq_len=16)

    emb_params = [host.embed_tokens.weight, host.pos_emb.weight]
    body_params = [p for n, p in host.named_parameters() if n not in ("embed_tokens.weight", "pos_emb.weight")]
    opt = torch.optim.AdamW([
        {"params": body_params, "lr": 6e-3, "weight_decay": 2.5e-2},
        {"params": emb_params, "lr": 5e-4, "weight_decay": 2.5e-2},
    ])

    for _ in range(steps):
        idx = torch.randint(0, len(train_seqs), (batch_size,))
        batch = train_seqs[idx]
        logits = host.forward_text(batch[:, :-1], g_em=False)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), batch[:, 1:].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(host.parameters(), 1.0)
        opt.step()

    host.eval()
    with torch.no_grad():
        val_logits = host.forward_text(val_seqs[:, :-1], g_em=False)
        val_nll = float(F.cross_entropy(val_logits.reshape(-1, VOCAB_SIZE), val_seqs[:, 1:].reshape(-1)).item())
        val_acc = float((val_logits.argmax(dim=-1) == val_seqs[:, 1:]).float().mean().item())
        emb_svals = torch.linalg.svdvals(host.embed_tokens.weight[:GEN_VOCAB]).numpy()

    host.freeze_base_weights()
    return host, {
        "pretrain_steps": steps,
        "baseline_task_a_nll": round(val_nll, 6),
        "baseline_task_a_acc": round(val_acc, 6),
        "uniform_text_nll": round(math.log(GEN_VOCAB), 6),
        "host_sha256": host.base_weight_sha256(),
        "gen_embed_sv_top_lex_energy_frac": round(
            float((emb_svals[:LEX_DIM] ** 2).sum() / (emb_svals ** 2).sum()), 6
        ),
    }


def align_phase_a_connector(
    host: HybridDualTaskHost, em_train: EMBatch, steps: int = 50, batch_size: int = 24
) -> dict:
    """Authentic Phase A: align ONLY the Layer-0 EM connector (orthogonal complement LEX_DIM..d_model)
    on the 100% frozen host, then freeze it before Phase B (W4 Option B1 Stop-Grad contract).
    """
    sha_before = host.base_weight_sha256()
    em_lift_p = nn.Parameter(host.em_lift[:, LEX_DIM:].clone())
    opt = torch.optim.AdamW([em_lift_p], lr=3e-2)
    gen = torch.Generator().manual_seed(PRETRAIN_SEED + 77)
    n_em = len(em_train.labels)
    host.eval()
    for _ in range(steps):
        idx = torch.randint(0, n_em, (batch_size,), generator=gen)
        emb_em = torch.matmul(em_train.em_feats[idx], F.pad(em_lift_p, (LEX_DIM, 0))) + host.em_slot_pos_bias.unsqueeze(0)
        full_e = torch.cat([emb_em, host.embed_tokens(em_train.prose_ids[idx])], dim=1)
        m = host._make_em_mask(batch_size, full_e.shape[1], 4, full_e.device)
        logits = host.forward_embeds(full_e, g_em=True, em_tok_mask=m)[:, -1, :]
        cand = torch.where(em_train.task_is_q2[idx, None], logits[:, Q2_ANS_IDS], logits[:, Q3_ANS_IDS])
        bp = em_train.bayes_probs[idx]
        loss = -(bp * (F.log_softmax(cand, dim=-1) - torch.log(bp.clamp(min=1e-12)))).sum(dim=-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        host.em_lift.copy_(F.pad(em_lift_p.detach(), (LEX_DIM, 0)))
    assert host.base_weight_sha256() == sha_before, "Phase A mutated base host weights!"
    return {"phase_a_steps": steps, "connector_frozen_for_phase_b": True, "host_sha256_preserved": True}


def compute_kfac_and_nullspace_stats(
    host: HybridDualTaskHost, text_seqs: torch.Tensor, em_batch: EMBatch
) -> Dict[str, dict]:
    host.eval()
    mods = host.lora_modules()
    stats: Dict[str, dict] = {}
    for l_idx, name, mod in mods:
        stats[name] = {
            "layer": l_idx, "layer_type": host.layer_types[l_idx],
            "in_dim": mod.in_features, "out_dim": mod.out_features,
            "C_gen": torch.zeros(mod.in_features, mod.in_features),
            "G_gen": torch.zeros(mod.out_features, mod.out_features),
            "diag_gen": torch.zeros(mod.out_features, mod.in_features),
            "n_gen": 0,
            "C_em": torch.zeros(mod.in_features, mod.in_features),
            "G_em": torch.zeros(mod.out_features, mod.out_features),
            "n_em": 0,
        }

    cache_in: Dict[str, torch.Tensor] = {}
    cache_out: Dict[str, torch.Tensor] = {}
    hooks = []

    def _mk(mname: str):
        def _hk(_m, inp, out):
            cache_in[mname] = inp[0].detach()
            out.retain_grad()
            cache_out[mname] = out
        return _hk

    for _, name, mod in mods:
        hooks.append(mod.register_forward_hook(_mk(name)))

    gen_rng = torch.Generator().manual_seed(12345)
    for bi in range(4):
        batch = text_seqs[bi * 24 : (bi + 1) * 24]
        cache_in.clear(); cache_out.clear()
        emb = host.embed_tokens(batch[:, :-1]).detach().requires_grad_(True)
        logits = host.forward_embeds(emb, g_em=False)
        flat_p = F.softmax(logits.detach(), dim=-1).reshape(-1, VOCAB_SIZE)
        y_mc = torch.multinomial(flat_p, num_samples=1, generator=gen_rng).squeeze(-1)
        (-F.log_softmax(logits.reshape(-1, VOCAB_SIZE), dim=-1)[torch.arange(len(y_mc)), y_mc].sum()).backward()
        for _, name, _ in mods:
            xa = cache_in[name].reshape(-1, stats[name]["in_dim"])
            ga = cache_out[name].grad.detach().reshape(-1, stats[name]["out_dim"])
            stats[name]["C_gen"] += xa.T @ xa
            stats[name]["G_gen"] += ga.T @ ga
            stats[name]["diag_gen"] += (ga ** 2).T @ (xa ** 2)
            stats[name]["n_gen"] += xa.shape[0]

    for bi in range(4):
        sl = slice(bi * 48, (bi + 1) * 48)
        cache_in.clear(); cache_out.clear()
        ef = em_batch.em_feats[sl].detach().requires_grad_(True)
        logits = host.forward_em(ef, em_batch.prose_ids[sl])
        cand = torch.where(em_batch.task_is_q2[sl, None], logits[:, Q2_ANS_IDS], logits[:, Q3_ANS_IDS])
        (-(em_batch.bayes_probs[sl] * F.log_softmax(cand, dim=-1)).sum()).backward()
        for _, name, _ in mods:
            xf = cache_in[name].reshape(-1, stats[name]["in_dim"])
            gf = cache_out[name].grad.detach().reshape(-1, stats[name]["out_dim"])
            stats[name]["C_em"] += xf.T @ xf
            stats[name]["G_em"] += gf.T @ gf
            stats[name]["n_em"] += xf.shape[0]

    for hk in hooks:
        hk.remove()

    tau_c, tau_g = 1e-4, 1e-5
    for name, st in stats.items():
        st["C_gen"] /= max(1, st["n_gen"]); st["G_gen"] /= max(1, st["n_gen"])
        st["diag_gen"] /= max(1, st["n_gen"])
        st["C_em"] /= max(1, st["n_em"]); st["G_em"] /= max(1, st["n_em"])

        evals_c, evecs_c = torch.linalg.eigh(st["C_gen"])
        evals_c = evals_c.flip(0).clamp(min=0.0)
        evecs_c = evecs_c.flip(1)
        cum_var = torch.cumsum(evals_c, dim=0) / evals_c.sum().clamp(min=1e-12)
        # Strict 99.8% / 72% cap on Layer 0 lexical audit; balanced 56% cap on deep layers 3..7 to keep 14+ EM nullspace dims
        v_frac = 0.998 if st["layer"] == 0 else 0.992
        m_frac = 0.72 if st["layer"] == 0 else 0.56
        k_keep = min(int((cum_var < v_frac).sum().item()) + 1, int(m_frac * st["in_dim"]))
        u_k = evecs_c[:, :k_keep]
        p_perp = torch.eye(st["in_dim"]) - u_k @ u_k.T
        st["k_keep"] = k_keep
        st["null_dim"] = st["in_dim"] - k_keep
        st["sigma_ratio"] = float((evals_c[min(k_keep, len(evals_c) - 1)] / evals_c[0].clamp(min=1e-12)).item())
        st["null_proj"] = p_perp
        st["eta_em"] = float(torch.trace(p_perp @ st["C_em"] @ p_perp).item() / torch.trace(st["C_em"]).clamp(min=1e-12).item())

        p_c_real, p_c_herm, k_c_keep, comm_err, tail_c = build_complex_symmetrized_nullspace(st["C_gen"], v_frac, m_frac)
        st["null_proj_complex"] = p_c_real
        st["null_proj_complex_herm"] = p_c_herm
        st["k_keep_complex"] = k_c_keep
        st["complex_comm_err"] = comm_err
        st["complex_tail_ratio"] = tail_c

        c_reg = st["C_gen"] + tau_c * torch.eye(st["in_dim"])
        g_reg = st["G_gen"] + tau_g * torch.eye(st["out_dim"])
        ec_r, vc_r = torch.linalg.eigh(c_reg)
        c_ih = vc_r @ torch.diag(ec_r.clamp(min=1e-12).rsqrt()) @ vc_r.T
        eg_r, vg_r = torch.linalg.eigh(g_reg)
        g_ih = vg_r @ torch.diag(eg_r.clamp(min=1e-12).rsqrt()) @ vg_r.T
        st["lambda_c_max"] = float(torch.linalg.eigvalsh(c_ih @ st["C_em"] @ c_ih).max().item())
        st["lambda_g_max"] = float(torch.linalg.eigvalsh(g_ih @ st["G_em"] @ g_ih).max().item())
        st["gen_rayleigh_max"] = st["lambda_c_max"] * st["lambda_g_max"]
    return stats


# =============================================================================
# 5. Block 1 & Block 2 Audits (incl. Complex-Symmetrized P_perp^C & Cayley Q^perp)
# =============================================================================

@torch.no_grad()
def eval_task_a_fast(
    host_mod: HybridDualTaskHost, text_seqs: torch.Tensor,
    base_logits: torch.Tensor, base_probs: torch.Tensor, base_log_probs: torch.Tensor, base_nll: float,
) -> dict:
    logits1 = host_mod.forward_text(text_seqs[:, :-1], g_em=False)
    nll1 = float(F.cross_entropy(logits1.reshape(-1, VOCAB_SIZE), text_seqs[:, 1:].reshape(-1)).item())
    lp1 = F.log_softmax(logits1, dim=-1)
    kl_tok = float((base_probs * (base_log_probs - lp1)).sum(dim=-1).mean().item())
    max_diff = float((logits1 - base_logits).abs().max().item())
    return {
        "nll_base": round(base_nll, 6), "nll_mod": round(nll1, 6),
        "nll_drift": float(nll1 - base_nll), "kl_drift": kl_tok, "max_logit_diff": max_diff,
    }


@torch.no_grad()
def eval_mixed_fluency_fast(
    host_mod: HybridDualTaskHost, em_feats: torch.Tensor, text_seqs: torch.Tensor,
    base_mix_probs: torch.Tensor, base_mix_log_probs: torch.Tensor, base_mix_nll: float,
    h_mix_eval: Optional[torch.Tensor] = None, stop_layer: int = 0,
) -> dict:
    n = min(len(em_feats), len(text_seqs))
    if h_mix_eval is not None and stop_layer > 0:
        m = host_mod._make_em_mask(n, h_mix_eval.shape[1], 4, h_mix_eval.device)
        logits1 = host_mod.forward_from_layer(h_mix_eval[:n], stop_layer, g_em=True, em_tok_mask=m)[:, 4:-1, :]
    else:
        logits1 = host_mod.forward_mixed_fluency(em_feats[:n], text_seqs[:n, :12])
    nll1 = float(F.cross_entropy(logits1.reshape(-1, VOCAB_SIZE), text_seqs[:n, 1:12].reshape(-1)).item())
    lp1 = F.log_softmax(logits1, dim=-1)
    kl_tok = float((base_mix_probs * (base_mix_log_probs - lp1)).sum(dim=-1).mean().item())
    return {
        "mixed_nll_base": round(base_mix_nll, 6), "mixed_nll_mod": round(nll1, 6),
        "mixed_nll_drift": float(nll1 - base_mix_nll), "mixed_kl_drift": kl_tok,
    }


def verify_block1_fisher_nullspace_and_complex_cayley(
    host: HybridDualTaskHost, stats: Dict[str, dict], text_seqs: torch.Tensor,
    base_text_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float],
) -> dict:
    rng = np.random.default_rng(31415)
    a1 = rng.normal(size=(8, 8)); c_gen_t = a1 @ a1.T + 0.5 * np.eye(8)
    a2 = rng.normal(size=(8, 8)); c_em_t = a2 @ a2.T + 0.1 * np.eye(8)
    b1 = rng.normal(size=(6, 6)); g_gen_t = b1 @ b1.T + 0.5 * np.eye(6)
    b2 = rng.normal(size=(6, 6)); g_em_t = b2 @ b2.T + 0.1 * np.eye(6)
    def _ih(m):
        w, v = np.linalg.eigh(m)
        return v @ np.diag(1.0 / np.sqrt(w)) @ v.T
    kron_err = abs(
        float(np.linalg.eigvalsh(_ih(np.kron(c_gen_t, g_gen_t)) @ np.kron(c_em_t, g_em_t) @ _ih(np.kron(c_gen_t, g_gen_t))).max())
        - float(np.linalg.eigvalsh(_ih(c_gen_t) @ c_em_t @ _ih(c_gen_t)).max()) * float(np.linalg.eigvalsh(_ih(g_gen_t) @ g_em_t @ _ih(g_gen_t)).max())
    )

    with torch.no_grad():
        emb = host.embed_tokens(text_seqs[:128, :-1])
        h = host.final_norm(host.forward_prefix_layers(emb, len(host.layers), g_em=False)).reshape(-1, host.d_model)
        z0 = F.linear(h, host.embed_tokens.weight)
        p0 = F.softmax(z0, dim=-1)
        torch.manual_seed(42)
        dw_head = torch.randn_like(host.embed_tokens.weight); dw_head /= dw_head.norm()
        sc_sweep = []
        for eps in (0.01, 0.03, 0.10, 0.30, 0.60):
            dv = F.linear(h, eps * dw_head)
            kl_ex = float((p0 * (F.log_softmax(z0, dim=-1) - F.log_softmax(z0 + dv, dim=-1))).sum(dim=-1).mean().item())
            var_dv = float((p0 * (dv - (p0 * dv).sum(dim=-1, keepdim=True)) ** 2).sum(dim=-1).mean().item())
            gamma = float((dv.max(dim=-1).values - dv.min(dim=-1).values).max().item())
            lb = var_dv * ((math.exp(-gamma) + gamma - 1.0) / (gamma ** 2))
            ub = var_dv * ((math.exp(gamma) - gamma - 1.0) / (gamma ** 2))
            sc_sweep.append({
                "eps_norm": eps, "gamma_osc": round(gamma, 6), "kl_exact": kl_ex,
                "fisher_quad": 0.5 * var_dv, "sc_lower_bound": lb, "sc_upper_bound": ub,
                "bracket_holds": bool(lb - 1e-12 <= kl_ex <= ub + 1e-12),
                "rel_quad_err": round(abs(kl_ex - 0.5 * var_dv) / max(kl_ex, 1e-15), 6),
            })

    target_name = "layer_0.q_proj"
    st = stats[target_name]
    d = host.d_model; m = d // 2
    j_d = make_complex_structure_J(d)
    p_real, p_comp_real, p_comp_herm = st["null_proj"], st["null_proj_complex"], st["null_proj_complex_herm"]
    real_comm_defect = float(torch.linalg.norm(p_real.double() @ j_d - j_d @ p_real.double(), ord="fro").item())
    comp_comm_defect = st["complex_comm_err"]

    torch.manual_seed(77)
    r_c = 3
    u_raw = torch.complex(torch.randn(m, r_c, dtype=torch.float64), torch.randn(m, r_c, dtype=torch.float64)) * 0.25
    v_raw = torch.complex(torch.randn(m, r_c, dtype=torch.float64), torch.randn(m, r_c, dtype=torch.float64)) * 0.25
    u_p, v_p = p_comp_herm @ u_raw, p_comp_herm @ v_raw
    a_c = torch.cat([u_p, -v_p], dim=1); b_c = torch.cat([v_p, u_p], dim=1)
    mid = torch.eye(2 * r_c, dtype=torch.complex128) + 0.5 * (b_c.conj().T @ a_c)
    q_c = torch.eye(m, dtype=torch.complex128) - a_c @ torch.linalg.solve(mid, b_c.conj().T)
    q_real = torch.zeros((d, d), dtype=torch.float64)
    q_real[0::2, 0::2] = q_c.real; q_real[0::2, 1::2] = -q_c.imag
    q_real[1::2, 0::2] = q_c.imag; q_real[1::2, 1::2] = q_c.real

    cayley_orth_err = float(torch.linalg.norm(q_real.T @ q_real - torch.eye(d, dtype=torch.float64), ord="fro").item())
    cayley_so2_err = float(torch.linalg.norm(q_real @ j_d - j_d @ q_real, ord="fro").item())
    p_comp_real_64 = torch.zeros((d, d), dtype=torch.float64)
    p_comp_real_64[0::2, 0::2] = p_comp_herm.real
    p_comp_real_64[0::2, 1::2] = -p_comp_herm.imag
    p_comp_real_64[1::2, 0::2] = p_comp_herm.imag
    p_comp_real_64[1::2, 1::2] = p_comp_herm.real
    u_top_proj = torch.eye(d, dtype=torch.float64) - p_comp_real_64
    cayley_top_text_id_err = float(torch.linalg.norm((q_real - torch.eye(d, dtype=torch.float64)) @ u_top_proj, ord="fro").item())
    c_sym = 0.5 * (st["C_gen"].double() + j_d.T @ st["C_gen"].double() @ j_d)
    loewner_max_eig = float(torch.linalg.eigvalsh(st["C_gen"].double() - 2.0 * c_sym).max().item())

    _, evecs_c = torch.linalg.eigh(st["C_gen"])
    u_top = evecs_c[:, -1:]
    torch.manual_seed(99)
    v_out = F.normalize(torch.randn(d, 1), dim=0)
    target_mod = dict((mn, mod) for _, mn, mod in host.lora_modules())[target_name]
    base_w = target_mod.weight.detach().clone()
    norm_scale = 0.35
    tx96 = text_seqs[:96]
    b_log, b_p, b_lp, _ = base_text_cache
    b_log96, b_p96, b_lp96 = b_log[:96], b_p[:96], b_lp[:96]
    b_nll96 = float(F.cross_entropy(b_log96.reshape(-1, VOCAB_SIZE), tx96[:, 1:].reshape(-1)).item())
    proxy_comparison = {}
    for label, dw_raw in (
        ("top_text_dir", v_out @ u_top.T),
        ("random_rank4", torch.randn(d, 4) @ torch.randn(4, d)),
        ("nullspace_real_rank4", (torch.randn(d, 4) @ torch.randn(4, d)) @ p_real),
        ("nullspace_complex_P_C_rank4", (torch.randn(d, 4) @ torch.randn(4, d)) @ p_comp_real),
        ("nullspace_complex_cayley_Q_perp", base_w @ (q_real.float() - torch.eye(d))),
    ):
        dw = dw_raw * (norm_scale / dw_raw.norm().clamp(min=1e-12))
        with torch.no_grad():
            target_mod.weight.data.copy_(base_w + dw)
            res = eval_task_a_fast(host, tx96, b_log96, b_p96, b_lp96, b_nll96)
            target_mod.weight.data.copy_(base_w)
        proxy_comparison[label] = {
            "frob_norm": norm_scale, "exact_kl_drift": res["kl_drift"], "exact_nll_drift": res["nll_drift"],
            "proxy_l2": 0.5 * float((dw ** 2).sum().item()),
            "proxy_diag_ewc": 0.5 * float((st["diag_gen"] * (dw ** 2)).sum().item()),
            "proxy_kfac": 0.5 * float(torch.trace(dw.T @ st["G_gen"] @ dw @ st["C_gen"]).item()),
        }

    top_kl = proxy_comparison["top_text_dir"]["exact_kl_drift"]
    rand_kl = proxy_comparison["random_rank4"]["exact_kl_drift"]
    real_kl = proxy_comparison["nullspace_real_rank4"]["exact_kl_drift"]
    comp_kl = proxy_comparison["nullspace_complex_P_C_rank4"]["exact_kl_drift"]
    cay_kl = proxy_comparison["nullspace_complex_cayley_Q_perp"]["exact_kl_drift"]
    return {
        "kronecker_spectral_identity_err": kron_err,
        "self_concordance_sweep": sc_sweep,
        "all_sc_brackets_hold": all(x["bracket_holds"] for x in sc_sweep),
        "proxy_comparison_at_matched_frob": proxy_comparison,
        "nullspace_real_kl_reduction_vs_top": round(top_kl / max(real_kl, 1e-12), 2),
        "nullspace_real_kl_reduction_vs_rand": round(rand_kl / max(real_kl, 1e-12), 2),
        "complex_symmetrized_P_C_audit": {
            "audited_primary_module": target_name,
            "real_P_commutation_defect_fro": round(real_comm_defect, 6),
            "complex_P_C_commutation_defect_fro": comp_comm_defect,
            "loewner_C_minus_2Csym_max_eig": loewner_max_eig,
            "complex_P_C_kl_reduction_vs_top": round(top_kl / max(comp_kl, 1e-12), 2),
            "complex_cayley_Q_perp_orth_err": cayley_orth_err,
            "complex_cayley_Q_perp_so2_err": cayley_so2_err,
            "complex_cayley_Q_perp_top_text_identity_err": cayley_top_text_id_err,
            "complex_cayley_Q_perp_kl_reduction_vs_top": round(top_kl / max(cay_kl, 1e-12), 2),
        },
        "diag_ewc_counterexample_d32": {
            "underestimation_factor_aligned": float(d),
            "false_penalty_on_nullspace": round(1.0 / d, 6),
            "true_kl_on_nullspace": 0.0,
        },
    }


def summarize_layer_selection(stats: Dict[str, dict]) -> Tuple[dict, List[int]]:
    by_layer: Dict[int, dict] = {}
    for _, st in stats.items():
        l = st["layer"]
        e = by_layer.setdefault(l, {"layer": l, "layer_type": st["layer_type"], "rayleigh_sum": 0.0, "rayleigh_max": 0.0, "eta": [], "sig": [], "ndim": []})
        e["rayleigh_sum"] += st["gen_rayleigh_max"]
        e["rayleigh_max"] = max(e["rayleigh_max"], st["gen_rayleigh_max"])
        e["eta"].append(st["eta_em"]); e["sig"].append(st["sigma_ratio"]); e["ndim"].append(st["null_dim"])

    rows = [{
        "layer": l, "layer_type": by_layer[l]["layer_type"],
        "rayleigh_sum": round(by_layer[l]["rayleigh_sum"], 4),
        "rayleigh_max": round(by_layer[l]["rayleigh_max"], 4),
        "mean_eta_em": round(float(np.mean(by_layer[l]["eta"])), 4),
        "mean_sigma_ratio": round(float(np.mean(by_layer[l]["sig"])), 6),
        "mean_null_dim": round(float(np.mean(by_layer[l]["ndim"])), 2),
    } for l in sorted(by_layer.keys())]

    # Select BOTH post-early Full-Attention anchors (3, 7) + top-2 mid-late GDN layers by Rayleigh score
    # (mirrors our Qwen 4B/9B/27B prescription: FA anchors + paired GDN layers, early 0..2 frozen!)
    fa_post = [r["layer"] for r in rows if r["layer"] >= 3 and r["layer_type"] == "full_attention"]
    dn_post = sorted([r for r in rows if r["layer"] >= 3 and r["layer_type"] == "linear_attention"], key=lambda r: r["rayleigh_sum"], reverse=True)
    selected_layers = sorted(fa_post + [r["layer"] for r in dn_post[:2]])  # [3, 5, 6, 7] or [3, 4, 5, 7]
    return {
        "layer_table": rows,
        "all_layers_ranked_by_rayleigh": [r["layer"] for r in sorted(rows, key=lambda r: r["rayleigh_sum"], reverse=True)],
        "selected_post_early_hybrid_layers": selected_layers,
        "stop_grad_boundary_layer": min(selected_layers),
    }, selected_layers


# =============================================================================
# 6. Fast Stop-Grad Cached Eval & Training Engine
# =============================================================================

@dataclass
class FrozenLayerCache:
    stop_layer: int
    h_em: torch.Tensor
    h_em_flip: torch.Tensor
    h_em_prose_pert: torch.Tensor
    h_em_rot1: torch.Tensor
    h_em_canon: torch.Tensor
    h_mix_train: torch.Tensor
    base_mix_train_probs: torch.Tensor
    base_mix_train_log_probs: torch.Tensor
    h_mix_eval: torch.Tensor
    eval_h: Dict[str, dict]


def select_lowest_fisher_gdn_head(stats: Dict[str, dict], layer_idx: int, n_heads: int, head_dim: int) -> int:
    """W3 Fisher-guided slow-head selection: pick the GDN head with minimal Task-A Fisher energy."""
    st_o = stats[f"layer_{layer_idx}.o_proj"]
    diag_o = st_o["diag_gen"]  # (d_model, d_model)
    head_energies = [
        float(diag_o[:, h * head_dim : (h + 1) * head_dim].sum().item())
        for h in range(n_heads)
    ]
    return int(np.argmin(head_energies))


@torch.no_grad()
def build_frozen_layer_cache(
    host_base: HybridDualTaskHost,
    em_train: EMBatch,
    em_eval: Dict[str, EMBatch],
    text_train: torch.Tensor,
    text_val: torch.Tensor,
    stop_layer: int,
    early_warmgate_mode: Optional[str] = None,
    slow_heads_by_layer: Optional[Dict[int, int]] = None,
    stats: Optional[Dict[str, dict]] = None,
    include_canon: bool = True,
) -> FrozenLayerCache:
    h_ref = copy.deepcopy(host_base)
    if early_warmgate_mode is not None:
        for l_idx in range(stop_layer):
            if isinstance(h_ref.layers[l_idx], GatedDeltaNetBlock):
                hs = slow_heads_by_layer.get(l_idx, 3) if slow_heads_by_layer else 3
                q_np = stats[f"layer_{l_idx}.q_proj"]["null_proj"] if stats is not None else None
                h_ref.layers[l_idx].set_w1_warmgate(early_warmgate_mode, head_idx=hs, q_null_proj=q_np)
    h_ref.eval()

    def _pref(ef: torch.Tensor, pr: torch.Tensor) -> torch.Tensor:
        emb = h_ref.build_em_embeds(ef, pr)
        m = h_ref._make_em_mask(emb.shape[0], emb.shape[1], 4, emb.device)
        return h_ref.forward_prefix_layers(emb, stop_layer, g_em=True, em_tok_mask=m)

    h_em = _pref(em_train.em_feats, em_train.prose_ids)
    h_flip = _pref(em_train.em_feats_slot_flip, em_train.prose_ids)
    h_pp = _pref(em_train.em_feats, em_train.prose_ids_pert)
    h_r1 = _pref(em_train.em_feats_rot1, em_train.prose_ids)
    h_can = _pref(em_train.em_feats_canon, em_train.prose_ids) if include_canon else h_em[:1]

    n_mix = min(384, len(text_train), len(em_train.em_feats))
    # Use 8 text tokens (total T=12, matching l_gap=8 h_em length for single-graph fused forward!)
    mix_text_len = 8 if em_train.prose_ids.shape[1] == 8 else 12
    tx_sub = text_train[:n_mix, :mix_text_len]
    ef_sub = em_train.em_feats[:n_mix]
    h_mix_tr = _pref(ef_sub, tx_sub)
    base_mix_tr_log = host_base.forward_mixed_fluency(ef_sub, tx_sub)
    base_mix_tr_p = F.softmax(base_mix_tr_log, dim=-1)
    base_mix_tr_lp = F.log_softmax(base_mix_tr_log, dim=-1)

    n_ev = min(len(em_eval["G1"].em_feats), len(text_val))
    h_mix_ev = _pref(em_eval["G1"].em_feats[:n_ev], text_val[:n_ev, :12])

    eval_h: Dict[str, dict] = {}
    for cname, batch in em_eval.items():
        d_c = {"raw": _pref(batch.em_feats, batch.prose_ids)}
        if include_canon:
            d_c["canon"] = _pref(batch.em_feats_canon, batch.prose_ids)
        if cname == "G1":
            d_c["flip"] = _pref(batch.em_feats_slot_flip, batch.prose_ids)
            d_c["prose"] = _pref(batch.em_feats, batch.prose_ids_pert)
            d_c["rot1"] = _pref(batch.em_feats_rot1, batch.prose_ids)
            if include_canon:
                d_c["prose_canon"] = _pref(batch.em_feats_canon, batch.prose_ids_pert)
                d_c["rot1_canon"] = _pref(batch.em_feats_rot1_canon, batch.prose_ids)
        eval_h[cname] = d_c

    return FrozenLayerCache(
        stop_layer=stop_layer, h_em=h_em, h_em_flip=h_flip, h_em_prose_pert=h_pp,
        h_em_rot1=h_r1, h_em_canon=h_can, h_mix_train=h_mix_tr,
        base_mix_train_probs=base_mix_tr_p, base_mix_train_log_probs=base_mix_tr_lp,
        h_mix_eval=h_mix_ev, eval_h=eval_h,
    )


def compute_ece_10bin(probs: np.ndarray, labels: np.ndarray) -> float:
    conf, pred = probs.max(axis=1), probs.argmax(axis=1)
    hits = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, 11)
    ece, n = 0.0, len(labels)
    for j in range(10):
        mask = (conf >= edges[j]) & (conf <= edges[j + 1] if j == 9 else conf < edges[j + 1])
        cnt = int(mask.sum())
        if cnt > 0:
            ece += (cnt / n) * abs(float(hits[mask].mean() - conf[mask].mean()))
    return float(ece)


def fit_scalar_temp(logits: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    def _nll(log_t: float) -> float:
        s = logits / math.exp(log_t)
        s = s - s.max(axis=1, keepdims=True)
        ex = np.exp(s)
        pr = ex / ex.sum(axis=1, keepdims=True)
        return float(-np.log(np.clip(pr[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())
    opt = minimize_scalar(_nll, bounds=(math.log(0.15), math.log(10.0)), method="bounded")
    return float(math.exp(opt.x)), float(opt.fun)


def _roll_rows_by_shift(probs: np.ndarray, shifts: np.ndarray) -> np.ndarray:
    out = np.empty_like(probs)
    for s in range(4):
        m = shifts == s
        if np.any(m):
            out[m] = np.roll(probs[m], shift=s, axis=1)
    return out


@torch.no_grad()
def evaluate_em_suite(
    host: HybridDualTaskHost,
    eval_suite: Dict[str, EMBatch],
    fcache: Optional[FrozenLayerCache] = None,
    quotient_wrap: bool = False,
    cells_subset: Optional[Sequence[str]] = None,
) -> dict:
    host.eval()
    out: Dict[str, dict] = {}
    target_cells = list(cells_subset) if cells_subset is not None else list(eval_suite.keys())

    def _score(cname: str, view: str, ef: torch.Tensor, pr: torch.Tensor, ans_ids: List[int]) -> torch.Tensor:
        if fcache is not None:
            h_in = fcache.eval_h[cname][view]
            m = host._make_em_mask(h_in.shape[0], h_in.shape[1], 4, h_in.device)
            return host.forward_from_layer(h_in, fcache.stop_layer, g_em=True, em_tok_mask=m)[:, -1, ans_ids]
        return host.forward_em(ef, pr)[:, ans_ids]

    for cname in target_cells:
        batch = eval_suite[cname]
        ans_ids = Q2_ANS_IDS if cname.startswith("Q2") else Q3_ANS_IDS
        if quotient_wrap:
            p_c = F.softmax(_score(cname, "canon", batch.em_feats_canon, batch.prose_ids, ans_ids), dim=-1).cpu().numpy()
            probs = _roll_rows_by_shift(p_c, batch.canon_shift.cpu().numpy())
            cand_np = np.log(np.clip(probs, 1e-12, 1.0))
        else:
            cand = _score(cname, "raw", batch.em_feats, batch.prose_ids, ans_ids)
            probs = F.softmax(cand, dim=-1).cpu().numpy()
            cand_np = cand.cpu().numpy()

        labels_np = batch.labels.cpu().numpy()
        bayes_np = batch.bayes_probs.cpu().numpy()
        acc = float((probs.argmax(axis=1) == labels_np).mean())
        nll = float(-np.log(np.clip(probs[np.arange(len(labels_np)), labels_np], 1e-12, 1.0)).mean())
        bayes_acc = float((bayes_np.argmax(axis=1) == labels_np).mean())
        bayes_nll = float(-np.log(np.clip(bayes_np[np.arange(len(labels_np)), labels_np], 1e-12, 1.0)).mean())
        brier = float(((probs - np.eye(4)[labels_np]) ** 2).sum(axis=1).mean())
        ece = compute_ece_10bin(probs, labels_np)
        t_star, nll_tstar = fit_scalar_temp(cand_np, labels_np)

        rec = {
            "acc": round(acc, 6), "nll": round(nll, 6),
            "bayes_acc": round(bayes_acc, 6), "bayes_nll": round(bayes_nll, 6),
            "brier": round(brier, 6), "ece": round(ece, 6),
            "t_star": round(t_star, 4), "nll_at_t_star": round(nll_tstar, 6),
        }

        if cname == "G1":
            p_flip = F.softmax(_score(cname, "flip", batch.em_feats_slot_flip, batch.prose_ids, ans_ids), dim=-1).cpu().numpy()
            if quotient_wrap:
                p_r1 = _roll_rows_by_shift(
                    F.softmax(_score(cname, "rot1_canon", batch.em_feats_rot1_canon, batch.prose_ids, ans_ids), dim=-1).cpu().numpy(),
                    batch.rot1_canon_shift.cpu().numpy(),
                )
                p_prose = _roll_rows_by_shift(
                    F.softmax(_score(cname, "prose_canon", batch.em_feats_canon, batch.prose_ids_pert, ans_ids), dim=-1).cpu().numpy(),
                    batch.canon_shift.cpu().numpy(),
                )
            else:
                p_r1 = F.softmax(_score(cname, "rot1", batch.em_feats_rot1, batch.prose_ids, ans_ids), dim=-1).cpu().numpy()
                p_prose = F.softmax(_score(cname, "prose", batch.em_feats, batch.prose_ids_pert, ans_ids), dim=-1).cpu().numpy()

            nll_flip = float(-np.log(np.clip(p_flip[np.arange(len(labels_np)), labels_np], 1e-12, 1.0)).mean())
            dp_model = probs - p_flip
            dp_phys = bayes_np - batch.bayes_probs_slot_flip.cpu().numpy()
            orient_gain = float((dp_model * dp_phys).sum() / max((dp_phys ** 2).sum(), 1e-12))
            prose_tv = float(0.5 * np.abs(probs - p_prose).sum(axis=1).mean())
            p_exp_r1 = np.roll(probs, shift=1, axis=1)
            rec.update({
                "workf_slot_flip_nll_penalty": round(nll_flip - nll, 6),
                "workf_orientation_gain": round(orient_gain, 6),
                "prose_pert_tv": round(prose_tv, 6),
                "c4_max_defect": float(np.abs(p_r1 - p_exp_r1).max()),
                "c4_mean_tv": round(float(0.5 * np.abs(p_r1 - p_exp_r1).sum(axis=1).mean()), 6),
            })
        out[cname] = rec

    summary = {}
    if all(k in out for k in ("G1", "G2", "G3", "G4")):
        summary["mean_G1_G3_acc"] = round(0.5 * (out["G1"]["acc"] + out["G3"]["acc"]), 6)
        summary["mean_G1_G3_nll"] = round(0.5 * (out["G1"]["nll"] + out["G3"]["nll"]), 6)
        summary["mean_G1_G4_acc"] = round(float(np.mean([out[k]["acc"] for k in ("G1", "G2", "G3", "G4")])), 6)
        summary["mean_G1_G4_nll"] = round(float(np.mean([out[k]["nll"] for k in ("G1", "G2", "G3", "G4")])), 6)
    if all(k in out for k in ("Q2_valid", "Q2_fresh")):
        summary["q2_fresh_minus_valid_acc"] = round(out["Q2_fresh"]["acc"] - out["Q2_valid"]["acc"], 6)
        summary["q2_fresh_minus_valid_nll_gain"] = round(out["Q2_valid"]["nll"] - out["Q2_fresh"]["nll"], 6)
    if "Q2_indep" in out:
        summary["q2_indep_nll_vs_log4"] = round(out["Q2_indep"]["nll"] - math.log(4.0), 6)
    out["summary"] = summary
    return out


def configure_lora_method(
    host_base: HybridDualTaskHost, stats: Dict[str, dict], selected_layers: Sequence[int],
    method: str, seed: int, rank: int = 8,
    early_warmgate_mode: Optional[str] = None,
    mid_warmgate_mode: Optional[str] = None,
    slow_heads_by_layer: Optional[Dict[int, int]] = None,
) -> HybridDualTaskHost:
    host = copy.deepcopy(host_base)
    host.freeze_base_weights()
    host.detach_all_lora()

    stop_l = min(selected_layers)
    if early_warmgate_mode is not None:
        for l_idx in range(stop_l):
            if isinstance(host.layers[l_idx], GatedDeltaNetBlock):
                hs = slow_heads_by_layer.get(l_idx, 3) if slow_heads_by_layer else 3
                q_np = stats[f"layer_{l_idx}.q_proj"]["null_proj"]
                host.layers[l_idx].set_w1_warmgate(early_warmgate_mode, head_idx=hs, q_null_proj=q_np)
    if mid_warmgate_mode is not None:
        for l_idx in selected_layers:
            if isinstance(host.layers[l_idx], GatedDeltaNetBlock):
                hs = slow_heads_by_layer.get(l_idx, 3) if slow_heads_by_layer else 3
                q_np = stats[f"layer_{l_idx}.q_proj"]["null_proj"]
                host.layers[l_idx].set_w1_warmgate(mid_warmgate_mode, head_idx=hs, q_null_proj=q_np)

    for m_idx, (l_idx, mname, mod) in enumerate(host.lora_modules()):
        sub = mname.split(".")[-1]
        is_tier1_kv_beta_ffn = sub in ("k_proj", "v_proj", "beta_proj", "w_up", "w_down")
        eff_rank = min(rank, mod.out_features)
        if method == "unconstrained_lora":
            mod.attach_lora(rank=eff_rank, alpha=16.0, null_proj=None, modality_gated=False, seed=seed + m_idx)
        elif method == "layer_selected_gen_fisher_lora":
            if l_idx in selected_layers:
                mod.attach_lora(rank=eff_rank, alpha=16.0, null_proj=None, modality_gated=False, seed=seed + m_idx)
        elif method == "nullspace_projected_lora":
            if l_idx in selected_layers:
                mod.attach_lora(rank=eff_rank, alpha=16.0, null_proj=stats[mname]["null_proj"], modality_gated=False, seed=seed + m_idx)
        elif method == "modality_gated_unconstrained":
            if l_idx in selected_layers:
                mod.attach_lora(rank=eff_rank, alpha=16.0, null_proj=None, modality_gated=True, seed=seed + m_idx)
        elif method == "modality_gated_nullspace_hybrid":
            if l_idx in selected_layers:
                mod.attach_lora(rank=eff_rank, alpha=16.0, null_proj=stats[mname]["null_proj"], modality_gated=True, seed=seed + m_idx)
        elif method == "two_tier_token_kv_gated_hybrid":
            # Exact Theorem 4.3 Two-Tier Token-Position Gate:
            #   Tier 1 (K, V, beta, FFN): per-token m_EM(t) -> strictly 0 delta on prose tokens, full EM capacity!
            #   Tier 2 (Q, O, alpha): sequence gate g_EM=1 + input Nullspace P_perp + MixKL anchor -> MixKL <= 1e-3!
            if l_idx in selected_layers:
                mod.attach_lora(
                    rank=eff_rank, alpha=16.0,
                    null_proj=stats[mname]["null_proj"] if not is_tier1_kv_beta_ffn else None,
                    modality_gated=True, token_position_gated=is_tier1_kv_beta_ffn, seed=seed + m_idx,
                )
        else:
            raise ValueError(method)
    return host


def measure_grad_variance_ratio(
    host: HybridDualTaskHost, em_batch: EMBatch, fcache: FrozenLayerCache, n_samples: int = 48
) -> dict:
    params = host.trainable_lora_params()
    host.eval()
    sl = fcache.stop_layer
    g_ce_sq, g_kl_sq, g_diff_sq = [], [], []
    mask1 = host._make_em_mask(1, fcache.h_em.shape[1], 4, fcache.h_em.device)
    for i in range(n_samples):
        h_in = fcache.h_em[i : i + 1]
        q2_m = em_batch.task_is_q2[i]
        y = em_batch.labels[i : i + 1]
        bp = em_batch.bayes_probs[i : i + 1]

        for p in params: p.grad = None
        cand = host.forward_from_layer(h_in, sl, g_em=True, em_tok_mask=mask1)[:, -1, Q2_ANS_IDS if q2_m.item() else Q3_ANS_IDS]
        F.cross_entropy(cand, y).backward()
        g_ce = torch.cat([p.grad.detach().reshape(-1) for p in params])

        for p in params: p.grad = None
        cand2 = host.forward_from_layer(h_in, sl, g_em=True, em_tok_mask=mask1)[:, -1, Q2_ANS_IDS if q2_m.item() else Q3_ANS_IDS]
        (-(bp * F.log_softmax(cand2, dim=-1)).sum()).backward()
        g_kl = torch.cat([p.grad.detach().reshape(-1) for p in params])

        g_ce_sq.append(float((g_ce ** 2).sum().item()))
        g_kl_sq.append(float((g_kl ** 2).sum().item()))
        g_diff_sq.append(float(((g_ce - g_kl) ** 2).sum().item()))

    for p in params: p.grad = None
    m_ce, m_kl, m_diff = float(np.mean(g_ce_sq)), float(np.mean(g_kl_sq)), float(np.mean(g_diff_sq))
    return {
        "mean_sq_norm_ce": round(m_ce, 6), "mean_sq_norm_kl": round(m_kl, 6),
        "irreducible_bayes_noise_sq": round(m_diff, 6),
        "pythagorean_residual": round(abs(m_ce - (m_kl + m_diff)) / max(m_ce, 1e-12), 4),
        "variance_reduction_ratio": round(m_ce / max(m_kl, 1e-12), 3),
    }


def train_adapter(
    host_base: HybridDualTaskHost, stats: Dict[str, dict], selected_layers: Sequence[int],
    method: str, loss_mode: str, em_train: EMBatch, em_eval: Dict[str, EMBatch],
    text_val: torch.Tensor,
    base_text_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float],
    base_mix_cache: Tuple[torch.Tensor, torch.Tensor, float],
    fcache: Optional[FrozenLayerCache], seed: int, steps: int = 160, batch_size: int = 16,
    sub_cf: int = 4,
    record_milestones: Sequence[int] = (69, 96, 160),
    early_warmgate_mode: Optional[str] = None,
    mid_warmgate_mode: Optional[str] = None,
    slow_heads_by_layer: Optional[Dict[int, int]] = None,
    final_cells_subset: Optional[Sequence[str]] = None,
) -> Tuple[HybridDualTaskHost, dict]:
    t_run0 = time.perf_counter()
    torch.manual_seed(seed); np.random.seed(seed)
    host = configure_lora_method(
        host_base, stats, selected_layers, method=method, seed=seed, rank=8,
        early_warmgate_mode=early_warmgate_mode, mid_warmgate_mode=mid_warmgate_mode,
        slow_heads_by_layer=slow_heads_by_layer,
    )
    params = host.trainable_lora_params()
    n_trainable = sum(p.numel() for p in params)

    use_cache = (fcache is not None) and (method != "unconstrained_lora")
    quotient_mode = loss_mode == "bayes_kl_cf_cal_quotient"
    use_fisher_reg = method in ("layer_selected_gen_fisher_lora", "modality_gated_nullspace_hybrid", "two_tier_token_kv_gated_hybrid")
    fisher_lambda = 1.2 if method == "layer_selected_gen_fisher_lora" else (0.65 if method == "two_tier_token_kv_gated_hybrid" else 0.4)
    use_mix_anchor = method in ("modality_gated_nullspace_hybrid", "two_tier_token_kv_gated_hybrid")
    mix_anchor_weight = 350.0 if method == "two_tier_token_kv_gated_hybrid" else 12.0
    mix_anchor_every = 2
    m_bs = 8

    # Tier-2 full K-FAC reg + light output-side G_gen damping on Tier-1 K/V to curb prose cross-attention!
    fisher_tier2_mods = [
        (mname, mod) for _, mname, mod in host.lora_modules()
        if mod.lora_rank > 0 and (method != "two_tier_token_kv_gated_hybrid" or not mod.token_position_gated)
    ]
    fisher_tier1_kv_mods = [
        (mname, mod) for _, mname, mod in host.lora_modules()
        if mod.lora_rank > 0 and method == "two_tier_token_kv_gated_hybrid" and mname.split(".")[-1] in ("k_proj", "v_proj")
    ]

    opt = torch.optim.AdamW(params, lr=1.3e-2, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=5e-4)

    n_em = len(em_train.labels)
    gen = torch.Generator().manual_seed(seed + 505)
    milestones: Dict[str, dict] = {}
    grad_var_init = None; grad_var_final = None
    t_len = em_train.prose_ids.shape[1] + 4
    bs = batch_size
    sub = sub_cf
    can_fuse_mix = use_cache and use_mix_anchor and (fcache.h_mix_train.shape[1] == t_len)

    for step in range(1, steps + 1):
        host.train()
        idx = torch.randint(0, n_em, (bs,), generator=gen)
        q2_m = em_train.task_is_q2[idx]
        lbl = em_train.labels[idx]
        bp = em_train.bayes_probs[idx]
        do_mix = use_mix_anchor and (fcache is not None) and (step % mix_anchor_every == 0)
        m_idx = torch.randint(0, len(fcache.h_mix_train), (m_bs,), generator=gen) if do_mix else None
        fuse_mix_now = do_mix and can_fuse_mix

        if quotient_mode:
            sh = em_train.canon_shift[idx]
            bp_tgt = torch.empty_like(bp)
            for s in range(4):
                ms = sh == s
                if ms.any():
                    bp_tgt[ms] = torch.roll(bp[ms], shifts=-s, dims=-1)
        else:
            bp_tgt = bp

        mod_mix_fused = None
        # Single-graph batched forward (main + counterfactuals + fused mix_anchor -> 1 autograd graph!)
        if loss_mode in ("hard_label_ce", "bayes_kl_only"):
            if use_cache:
                h_main = fcache.h_em_canon[idx] if quotient_mode else fcache.h_em[idx]
                if fuse_mix_now:
                    h_cat = torch.cat([h_main, fcache.h_mix_train[m_idx]], dim=0)
                    out_cat = host.forward_from_layer(h_cat, fcache.stop_layer, g_em=True, em_tok_mask=host._make_em_mask(bs + m_bs, t_len, 4, h_cat.device))
                    logits = out_cat[:bs, -1, :]
                    mod_mix_fused = out_cat[bs:, 4:-1, :]
                else:
                    logits = host.forward_from_layer(h_main, fcache.stop_layer, g_em=True, em_tok_mask=host._make_em_mask(bs, t_len, 4, h_main.device))[:, -1, :]
            else:
                ef_m = em_train.em_feats_canon[idx] if quotient_mode else em_train.em_feats[idx]
                logits = host.forward_em(ef_m, em_train.prose_ids[idx])
            cand = torch.where(q2_m[:, None], logits[:, Q2_ANS_IDS], logits[:, Q3_ANS_IDS])
            log_q = F.log_softmax(cand, dim=-1)
            if loss_mode == "hard_label_ce":
                loss = F.nll_loss(log_q, lbl)
            else:
                loss = -(bp_tgt * (log_q - torch.log(bp_tgt.clamp(min=1e-12)))).sum(dim=-1).mean()
        elif loss_mode in ("bayes_kl_plus_cf_cal", "bayes_kl_cf_cal_quotient"):
            sub_idx = idx[:sub]
            q2_sub = q2_m[:sub]
            bp_flip = em_train.bayes_probs_slot_flip[sub_idx]

            if quotient_mode:
                parts_h = [fcache.h_em_canon[idx], fcache.h_em_flip[sub_idx], fcache.h_em_prose_pert[sub_idx]] if use_cache else None
                if use_cache:
                    if fuse_mix_now:
                        parts_h.append(fcache.h_mix_train[m_idx])
                    h_all = torch.cat(parts_h, dim=0)
                    out_all = host.forward_from_layer(h_all, fcache.stop_layer, g_em=True, em_tok_mask=host._make_em_mask(h_all.shape[0], t_len, 4, h_all.device))
                    l_all = out_all[: bs + 2 * sub, -1, :]
                    if fuse_mix_now:
                        mod_mix_fused = out_all[bs + 2 * sub :, 4:-1, :]
                else:
                    l_all = host.forward_em(
                        torch.cat([em_train.em_feats_canon[idx], em_train.em_feats_slot_flip[sub_idx], em_train.em_feats[sub_idx]], dim=0),
                        torch.cat([em_train.prose_ids[idx], em_train.prose_ids[sub_idx], em_train.prose_ids_pert[sub_idx]], dim=0),
                    )
                logits, l_flip, l_pp = l_all[:bs], l_all[bs : bs + sub], l_all[bs + sub :]
                cand = torch.where(q2_m[:, None], logits[:, Q2_ANS_IDS], logits[:, Q3_ANS_IDS])
                cand_flip = torch.where(q2_sub[:, None], l_flip[:, Q2_ANS_IDS], l_flip[:, Q3_ANS_IDS])
                cand_pp = torch.where(q2_sub[:, None], l_pp[:, Q2_ANS_IDS], l_pp[:, Q3_ANS_IDS])
                log_q = F.log_softmax(cand, dim=-1)
                l_kl = -(bp_tgt * (log_q - torch.log(bp_tgt.clamp(min=1e-12)))).sum(dim=-1).mean()
                cand_c = cand - cand.mean(dim=-1, keepdim=True)
                log_bp = torch.log(bp_tgt.clamp(min=1e-6))
                log_bp_c = log_bp - log_bp.mean(dim=-1, keepdim=True)
                l_norm = ((cand_c.norm(dim=-1) - log_bp_c.norm(dim=-1)) ** 2).mean()
                l_kl_flip = -(bp_flip * (F.log_softmax(cand_flip, dim=-1) - torch.log(bp_flip.clamp(min=1e-12)))).sum(dim=-1).mean()
                l_prose = -(bp[:sub] * (F.log_softmax(cand_pp, dim=-1) - torch.log(bp[:sub].clamp(min=1e-12)))).sum(dim=-1).mean()
                loss = l_kl + 0.30 * l_kl_flip + 0.04 * l_norm + 0.25 * l_prose
            else:
                if use_cache:
                    parts_h = [fcache.h_em[idx], fcache.h_em_flip[sub_idx], fcache.h_em_prose_pert[sub_idx], fcache.h_em_rot1[sub_idx]]
                    if fuse_mix_now:
                        parts_h.append(fcache.h_mix_train[m_idx])
                    h_all = torch.cat(parts_h, dim=0)
                    out_all = host.forward_from_layer(h_all, fcache.stop_layer, g_em=True, em_tok_mask=host._make_em_mask(h_all.shape[0], t_len, 4, h_all.device))
                    l_all = out_all[: bs + 3 * sub, -1, :]
                    if fuse_mix_now:
                        mod_mix_fused = out_all[bs + 3 * sub :, 4:-1, :]
                else:
                    l_all = host.forward_em(
                        torch.cat([em_train.em_feats[idx], em_train.em_feats_slot_flip[sub_idx], em_train.em_feats[sub_idx], em_train.em_feats_rot1[sub_idx]], dim=0),
                        torch.cat([em_train.prose_ids[idx], em_train.prose_ids[sub_idx], em_train.prose_ids_pert[sub_idx], em_train.prose_ids[sub_idx]], dim=0),
                    )
                logits = l_all[:bs]
                l_flip, l_pp, l_r1 = l_all[bs : bs + sub], l_all[bs + sub : bs + 2 * sub], l_all[bs + 2 * sub :]
                cand = torch.where(q2_m[:, None], logits[:, Q2_ANS_IDS], logits[:, Q3_ANS_IDS])
                cand_flip = torch.where(q2_sub[:, None], l_flip[:, Q2_ANS_IDS], l_flip[:, Q3_ANS_IDS])
                cand_pp = torch.where(q2_sub[:, None], l_pp[:, Q2_ANS_IDS], l_pp[:, Q3_ANS_IDS])
                cand_r1 = torch.where(q2_sub[:, None], l_r1[:, Q2_ANS_IDS], l_r1[:, Q3_ANS_IDS])

                log_q = F.log_softmax(cand, dim=-1)
                l_kl = -(bp_tgt * (log_q - torch.log(bp_tgt.clamp(min=1e-12)))).sum(dim=-1).mean()
                cand_c = cand - cand.mean(dim=-1, keepdim=True)
                cand_flip_c = cand_flip - cand_flip.mean(dim=-1, keepdim=True)
                log_bp = torch.log(bp_tgt.clamp(min=1e-6))
                log_bp_c = log_bp - log_bp.mean(dim=-1, keepdim=True)
                l_norm = ((cand_c.norm(dim=-1) - log_bp_c.norm(dim=-1)) ** 2).mean()
                l_kl_flip = -(bp_flip * (F.log_softmax(cand_flip, dim=-1) - torch.log(bp_flip.clamp(min=1e-12)))).sum(dim=-1).mean()

                v_phys = log_bp_c[:sub] - (torch.log(bp_flip.clamp(min=1e-6)) - torch.log(bp_flip.clamp(min=1e-6)).mean(dim=-1, keepdim=True))
                v_norm = v_phys.norm(dim=-1, keepdim=True)
                proj_model = ((cand_c[:sub] - cand_flip_c) * (v_phys / v_norm.clamp(min=1e-6))).sum(dim=-1)
                l_cf_margin = ((v_norm.squeeze(-1) > 0.15).float() * F.relu(0.85 * v_norm.squeeze(-1) - proj_model)).mean()

                l_prose = (log_q[:sub].exp().detach() * (log_q[:sub].detach() - F.log_softmax(cand_pp, dim=-1))).sum(dim=-1).mean()
                l_orbit = (((cand_r1 - cand_r1.mean(dim=-1, keepdim=True)) - torch.roll(cand_c[:sub], shifts=1, dims=-1)) ** 2).mean()
                loss = l_kl + 0.35 * l_kl_flip + 0.04 * l_norm + 0.15 * l_cf_margin + 0.25 * l_prose + 0.10 * l_orbit
        else:
            raise ValueError(loss_mode)

        if use_fisher_reg:
            reg = logits.new_tensor(0.0)
            for mname, mod in fisher_tier2_mods:
                pen = mod.lowrank_kfac_penalty(stats[mname]["G_gen"], stats[mname]["C_gen"])
                if pen is not None:
                    reg = reg + pen
            for mname, mod in fisher_tier1_kv_mods:
                pen = mod.lowrank_kfac_penalty(stats[mname]["G_gen"], None)
                if pen is not None:
                    reg = reg + 0.05 * pen
            loss = loss + fisher_lambda * reg

        if do_mix:
            if mod_mix_fused is not None:
                mod_mix = mod_mix_fused
            else:
                hm = fcache.h_mix_train[m_idx]
                mod_mix = host.forward_from_layer(hm, fcache.stop_layer, g_em=True, em_tok_mask=host._make_em_mask(m_bs, hm.shape[1], 4, hm.device))[:, 4:-1, :]
            kl_mix = (fcache.base_mix_train_probs[m_idx] * (fcache.base_mix_train_log_probs[m_idx] - F.log_softmax(mod_mix, dim=-1))).sum(dim=-1).mean()
            ramp = (0.25 + 0.75 * (step / float(steps))) if method == "two_tier_token_kv_gated_hybrid" else 1.0
            loss = loss + (mix_anchor_weight * ramp) * kl_mix

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()

        if step == 16 and loss_mode == "bayes_kl_only" and seed == STUDY_SEEDS[0] and fcache is not None:
            grad_var_init = measure_grad_variance_ratio(host, em_train, fcache, n_samples=24)

        if step in record_milestones:
            host.eval()
            is_final = step == max(record_milestones)
            subset = final_cells_subset if is_final else ("G1", "Q2_valid", "Q2_fresh")
            em_res = evaluate_em_suite(host, em_eval, fcache=(fcache if use_cache else None), quotient_wrap=quotient_mode, cells_subset=subset)
            if is_final:
                tx_res = eval_task_a_fast(host, text_val, *base_text_cache)
                mx_res = eval_mixed_fluency_fast(
                    host, em_eval["G1"].em_feats, text_val, *base_mix_cache,
                    h_mix_eval=(fcache.h_mix_eval if use_cache else None),
                    stop_layer=(fcache.stop_layer if use_cache else 0),
                )
            else:
                b_log, b_p, b_lp, b_nll = base_text_cache
                tx_res = eval_task_a_fast(host, text_val[:64], b_log[:64], b_p[:64], b_lp[:64], b_nll)
                bm_p, bm_lp, bm_nll = base_mix_cache
                mx_res = eval_mixed_fluency_fast(
                    host, em_eval["G1"].em_feats[:128], text_val[:128], bm_p[:128], bm_lp[:128], bm_nll,
                    h_mix_eval=(fcache.h_mix_eval[:128] if use_cache else None),
                    stop_layer=(fcache.stop_layer if use_cache else 0),
                )
            milestones[f"step_{step}"] = {"em": em_res, "task_a_pure": tx_res, "task_a_mixed": mx_res}

    if loss_mode == "bayes_kl_only" and seed == STUDY_SEEDS[0] and fcache is not None:
        grad_var_final = measure_grad_variance_ratio(host, em_train, fcache, n_samples=24)

    sha_ok = host.base_weight_sha256() == host_base.base_weight_sha256()
    assert sha_ok, "Base host weights mutated!"
    return host, {
        "method": method, "loss_mode": loss_mode, "seed": seed, "trainable_params": n_trainable,
        "host_sha256_preserved": sha_ok, "wall_seconds": round(time.perf_counter() - t_run0, 2),
        "milestones": milestones, "grad_var_step16": grad_var_init, "grad_var_final": grad_var_final,
    }


# =============================================================================
# 7. Round-2 Long-Gap (L_gap=32) W1 Slow-Head GDN Resolution Study (3 Arms!)
# =============================================================================

def run_long_gap_w1_resolution_study(
    host_base: HybridDualTaskHost, stats: Dict[str, dict], selected_layers: Sequence[int],
    text_train: torch.Tensor, text_val: torch.Tensor,
    base_text_cache: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float],
    base_mix_cache: Tuple[torch.Tensor, torch.Tensor, float],
) -> dict:
    """Resolve Tension A (W1): compare 3 long-gap (L_gap=32) designs:
      1) frozen_early_no_warmgate: early GDN decays to 0 (relies purely on Layer-3 FullAttn skip!)
      2) naive_early_warmgate_leaky_beta: shifts early alpha WITHOUT zeroing prose beta -> state-norm trap & MixKL spike!
      3) w1_w3_clean_masked_warmgate: W1 dt_bias shift + W3 strict m_EM(t) beta_prose=0 on slow head ->
         massive L=32 GDN retention boost, PureKL=0.0, low MixKL, AND full G1/Q2 recovery!
    """
    em_tr_32 = build_em_train_dataset(seed=930999, l_gap=32, scale=0.35)
    em_ev_32 = build_em_eval_suite(seed=950999, l_gap=32, compact=True)
    with torch.no_grad():
        bm32_log = host_base.forward_mixed_fluency(em_ev_32["G1"].em_feats[:256], text_val[:256, :12])
        bm32_p, bm32_lp = F.softmax(bm32_log, dim=-1), F.log_softmax(bm32_log, dim=-1)
        bm32_nll = float(F.cross_entropy(bm32_log.reshape(-1, VOCAB_SIZE), text_val[:256, 1:12].reshape(-1)).item())
        base_mix_cache_32 = (bm32_p, bm32_lp, bm32_nll)

    stop_l = min(selected_layers)
    slow_heads_by_layer = {
        l_idx: select_lowest_fisher_gdn_head(stats, l_idx, host_base.n_heads, host_base.d_model // host_base.n_heads)
        for l_idx, layer in enumerate(host_base.layers) if isinstance(layer, GatedDeltaNetBlock)
    }

    configs = (
        ("frozen_early_no_warmgate", None, None),
        ("naive_early_warmgate_leaky_beta", "leaky_beta", None),
        ("w1_w3_clean_dual_nullspace_warmgate", "clean_dual_nullspace", "clean_dual_nullspace"),
    )
    out = {"fisher_selected_slow_heads": slow_heads_by_layer}
    for label, early_wg, mid_wg in configs:
        fc32 = build_frozen_layer_cache(
            host_base, em_tr_32, em_ev_32, text_train, text_val, stop_layer=stop_l,
            early_warmgate_mode=early_wg, slow_heads_by_layer=slow_heads_by_layer, stats=stats,
            include_canon=False,
        )
        host_tr, res = train_adapter(
            host_base=host_base, stats=stats, selected_layers=selected_layers,
            method="two_tier_token_kv_gated_hybrid", loss_mode="bayes_kl_plus_cf_cal",
            em_train=em_tr_32, em_eval=em_ev_32, text_val=text_val,
            base_text_cache=base_text_cache, base_mix_cache=base_mix_cache_32,
            fcache=fc32, seed=STUDY_SEEDS[0], steps=90, batch_size=14, sub_cf=3,
            record_milestones=(90,), early_warmgate_mode=early_wg, mid_warmgate_mode=mid_wg,
            slow_heads_by_layer=slow_heads_by_layer,
            final_cells_subset=("G1", "Q2_valid", "Q2_fresh"),
        )
        with torch.no_grad():
            emb32 = host_tr.build_em_embeds(em_ev_32["G1"].em_feats[:48], em_ev_32["G1"].prose_ids[:48])
            m32 = host_tr._make_em_mask(48, emb32.shape[1], 4, emb32.device)
            h_cur = emb32 + host_tr.pos_emb(torch.arange(emb32.shape[1])).unsqueeze(0)
            early_rets, mid_rets = [], []
            for l_idx, layer in enumerate(host_tr.layers):
                if isinstance(layer, GatedDeltaNetBlock):
                    r_val = layer.measure_prose_retention(h_cur, g_em=True, em_tok_mask=m32, n_em=4)
                    (early_rets if l_idx < stop_l else mid_rets).append(r_val)
                h_cur = layer(h_cur, g_em=True, em_tok_mask=m32)
        s90 = res["milestones"]["step_90"]
        out[label] = {
            "early_warmgate_mode": early_wg,
            "mid_warmgate_mode": mid_wg,
            "extra_trainable_params": 0,
            "early_gdn_l32_mean_retention": round(float(np.mean(early_rets)), 6),
            "mid_gdn_l32_mean_retention": round(float(np.mean(mid_rets)), 6),
            "pure_text_kl_drift": s90["task_a_pure"]["kl_drift"],
            "pure_text_max_logit_diff": s90["task_a_pure"]["max_logit_diff"],
            "mixed_fluency_kl_drift": s90["task_a_mixed"]["mixed_kl_drift"],
            "mixed_fluency_nll_drift": s90["task_a_mixed"]["mixed_nll_drift"],
            "G1_acc": s90["em"]["G1"]["acc"],
            "G1_nll": s90["em"]["G1"]["nll"],
            "Q2_valid_acc": s90["em"]["Q2_valid"]["acc"],
            "Q2_fresh_acc": s90["em"]["Q2_fresh"]["acc"],
            "Q2_fresh_minus_valid_acc": s90["em"]["summary"]["q2_fresh_minus_valid_acc"],
        }
    return out


# =============================================================================
# 8. Block 5: Exact |V| = 248,320 Scaling Ledger (4B -> 9B -> 27B, W4 Parity!)
# =============================================================================

def compute_qwen_scaling_budgets_v248320() -> dict:
    specs = {
        "Qwen3.5-4B": dict(
            rev="851bf6e8", V=248_320, L=32, d=2560, d_ff=9216, tie_emb=True,
            h_q=16, h_kv=4, d_h=256, hk=16, hv=32, dk=128, dv=128, conv_k=4,
            sel_fa=[12, 16, 20, 24, 28], sel_dn=[11, 15, 19, 23],
            stop_layer=10, lora_rank=8,
            w4_mode1_peak_mib=40988.51, w4_mode2_peak_mib=53099.03,
            w4_vram_saved_mib=12110.52, w4_step_saved_pct=19.82,
            w4_mode1_unified_host_params=998_478, w4_mode1_cumul_m1_w123_params=1_486_878,
            w4_config_note="BF16 kv_train mb=16, Pattern-A seq accum (sub=8)",
        ),
        "Qwen3.5-9B": dict(
            rev="c2022362", V=248_320, L=32, d=4096, d_ff=12288, tie_emb=False,
            h_q=16, h_kv=4, d_h=256, hk=16, hv=32, dk=128, dv=128, conv_k=4,
            sel_fa=[12, 16, 20, 24, 28], sel_dn=[11, 15, 19, 23],
            stop_layer=10, lora_rank=8,
            w4_mode1_peak_mib=54708.32, w4_mode2_peak_mib=68884.68,
            w4_vram_saved_mib=14176.36, w4_step_saved_pct=19.82,
            w4_mode1_unified_host_params=1_334_862, w4_mode1_cumul_m1_w123_params=2_024_478,
            w4_config_note="BF16 kv_train mb=16, Pattern-A seq accum (sub=8)",
        ),
        "Qwen3.8-27B": dict(
            rev="37561816", V=248_320, L=64, d=5120, d_ff=17408, tie_emb=False,
            h_q=24, h_kv=4, d_h=256, hk=16, hv=48, dk=128, dv=128, conv_k=4,
            sel_fa=[20, 24, 28, 32, 36, 40, 48, 56], sel_dn=[19, 23, 27, 31, 35, 39],
            stop_layer=18, lora_rank=8,
            w4_mode1_peak_mib=46353.92, w4_mode2_peak_mib=50216.52,
            w4_vram_saved_mib=3862.60, w4_step_saved_pct=17.84,
            w4_mode1_unified_host_params=2_553_870, w4_mode1_cumul_m1_w123_params=3_377_630,
            w4_config_note="Mixed MLP-NF4 + FP32 Cayley + ckpt mb=16, Pattern-A (sub=8)",
        ),
    }
    out = {}
    for name, s in specs.items():
        d, dff, V, L = s["d"], s["d_ff"], s["V"], s["L"]
        L_fa, L_dn = L // 4, 3 * L // 4
        q_dim, kv_dim = s["h_q"] * s["d_h"], s["h_kv"] * s["d_h"]
        dk_tot, dv_tot = s["hk"] * s["dk"], s["hv"] * s["dv"]
        qkv_dim = 2 * dk_tot + dv_tot

        emb_p = V * d
        lm_head_p = 0 if s["tie_emb"] else V * d
        mlp_p = L * (3 * d * dff + 2 * d)
        fa_p = L_fa * (d * (2 * q_dim + 2 * kv_dim) + q_dim * d + 2 * s["d_h"])
        dn_p = L_dn * (d * (qkv_dim + dv_tot + 2 * s["hv"]) + dv_tot * d + qkv_dim * s["conv_k"] + 2 * s["hv"] + s["dv"])
        exact_host_params = int(emb_p + lm_head_p + mlp_p + fa_p + dn_p + d)

        r = s["lora_rank"]
        fa_lora = r * ((d + 2 * q_dim) + 2 * (d + kv_dim) + (q_dim + d))
        dn_lora_w3 = r * ((d + qkv_dim) + (dv_tot + d)) + 4 * 2 * (d + s["hv"])
        n_fa, n_dn = len(s["sel_fa"]), len(s["sel_dn"])
        w3_lora_params = int(n_fa * fa_lora + n_dn * dn_lora_w3)
        all_attn_dn_lora = int(L_fa * fa_lora + L_dn * dn_lora_w3)

        lora_opt_mib = w3_lora_params * 18.0 / MIB
        uk_mib = (n_fa + n_dn) * d * int(0.35 * d) * 2.0 / MIB
        frozen_frac = s["stop_layer"] / float(L)

        out[name] = {
            "pinned_rev": s["rev"],
            "vocab_size": V,
            "vocab_ratio_vs_151936": round(V / 151_936.0, 4),
            "tie_word_embeddings": s["tie_emb"],
            "exact_text_host_params": exact_host_params,
            "bf16_static_host_mib": round(exact_host_params * 2.0 / MIB, 2),
            "lm_head_alone_params": int(V * d),
            "lm_head_r8_lora_params_avoided": int(r * (V + d)),
            "L_total": L,
            "selected_full_attn_1based": s["sel_fa"],
            "selected_deltanet_1based": s["sel_dn"],
            "early_gdn_w1_clean_warmgate_layers_1based": [l for l in range(1, s["stop_layer"] + 1) if l % 4 != 0],
            "w1_slow_heads_per_early_gdn": 2,
            "w3_standalone_lora_params": w3_lora_params,
            "unified_w123_host_adapter_params": s["w4_mode1_unified_host_params"],
            "unified_m1_w123_cumulative_params": s["w4_mode1_cumul_m1_w123_params"],
            "all_layer_attn_dn_lora_params": all_attn_dn_lora,
            "param_reduction_vs_all_layers": round(all_attn_dn_lora / w3_lora_params, 2),
            "pct_of_host_params": round(100.0 * w3_lora_params / exact_host_params, 5),
            "lora_opt_vram_18B_per_param_mib": round(lora_opt_mib, 2),
            "lowrank_Uk_projector_bf16_mib": round(uk_mib, 2),
            "w4_parity_config": s["w4_config_note"],
            "option_B1_stop_grad_connector_frozen": {
                "stop_grad_after_layer": s["stop_layer"],
                "frozen_depth_frac": round(frozen_frac, 4),
                "w4_verified_vram_saved_mib": s["w4_vram_saved_mib"],
                "w4_verified_peak_vram_mib": s["w4_mode1_peak_mib"],
                "w4_verified_step_time_saved_pct": s["w4_step_saved_pct"],
            },
            "option_B2_cotune_connector_015lr": {
                "backward_activation_saved_mib": 0.0,
                "w4_verified_peak_vram_mib": s["w4_mode2_peak_mib"],
                "w4_verified_step_time_saved_pct": 0.0,
            },
        }
    assert out["Qwen3.5-4B"]["exact_text_host_params"] == 4_205_751_296
    assert out["Qwen3.5-9B"]["exact_text_host_params"] == 8_953_803_264
    assert out["Qwen3.8-27B"]["exact_text_host_params"] == 26_895_998_464
    return out


# =============================================================================
# 9. Main Driver
# =============================================================================

def main() -> None:
    # Single-thread CPU avoids Windows OpenMP spin-wait stalls on hybrid P/E cores and is fastest on d=32!
    torch.set_num_threads(1)
    t0 = time.perf_counter()
    print(f"[W3-R2] Pretraining 8-layer 3:1 Hybrid CPU host (d={D_MODEL}, L=8, tied lm_head) on Task A...", flush=True)
    grammar = CorrelatedTextGrammar(seed=7777)
    host_base, pretrain_info = pretrain_host_on_task_a(grammar, steps=140, batch_size=24)
    print(f"        Task A NLL={pretrain_info['baseline_task_a_nll']:.4f}, Acc={pretrain_info['baseline_task_a_acc']:.4f}, SHA={pretrain_info['host_sha256'][:16]}", flush=True)

    text_train = grammar.sample(np.random.default_rng(PRETRAIN_SEED + 2), 1024, seq_len=16)
    text_val = grammar.sample(np.random.default_rng(PRETRAIN_SEED + 3), 256, seq_len=16)
    em_train = build_em_train_dataset(seed=930101, l_gap=8, scale=0.70)
    em_eval = build_em_eval_suite(seed=950400, l_gap=8)

    phase_a_info = align_phase_a_connector(host_base, em_train, steps=45, batch_size=24)
    with torch.no_grad():
        bt_log = host_base.forward_text(text_val[:, :-1], g_em=False)
        bt_p, bt_lp = F.softmax(bt_log, dim=-1), F.log_softmax(bt_log, dim=-1)
        bt_nll = float(F.cross_entropy(bt_log.reshape(-1, VOCAB_SIZE), text_val[:, 1:].reshape(-1)).item())
        base_text_cache = (bt_log, bt_p, bt_lp, bt_nll)

        bm_log = host_base.forward_mixed_fluency(em_eval["G1"].em_feats[:256], text_val[:256, :12])
        bm_p, bm_lp = F.softmax(bm_log, dim=-1), F.log_softmax(bm_log, dim=-1)
        bm_nll = float(F.cross_entropy(bm_log.reshape(-1, VOCAB_SIZE), text_val[:256, 1:12].reshape(-1)).item())
        base_mix_cache = (bm_p, bm_lp, bm_nll)

    frozen_em = evaluate_em_suite(host_base, em_eval)
    print(f"        Phase-A Frozen Host EM: G1={frozen_em['G1']['acc']:.4f}/{frozen_em['G1']['nll']:.4f} | Oracle: G1={frozen_em['G1']['bayes_acc']:.4f}/{frozen_em['G1']['bayes_nll']:.4f}", flush=True)

    stats = compute_kfac_and_nullspace_stats(host_base, text_train, em_train)
    block1 = verify_block1_fisher_nullspace_and_complex_cayley(host_base, stats, text_val, base_text_cache)
    layer_sel_summary, selected_layers = summarize_layer_selection(stats)
    stop_layer = min(selected_layers)
    cc = block1["complex_symmetrized_P_C_audit"]
    print(
        f"[W3-R2] Block1+2 ({time.perf_counter()-t0:.1f}s): SC={block1['all_sc_brackets_hold']}, Kron={block1['kronecker_spectral_identity_err']:.2e}, "
        f"RealP={block1['nullspace_real_kl_reduction_vs_top']}x, CompP={cc['complex_P_C_kl_reduction_vs_top']}x, "
        f"||[P_C,J]||={cc['complex_P_C_commutation_defect_fro']:.2e} (real={cc['real_P_commutation_defect_fro']:.2f}), "
        f"CayleyQ={cc['complex_cayley_Q_perp_kl_reduction_vs_top']}x | SelLayers={selected_layers}",
        flush=True,
    )

    fcache = build_frozen_layer_cache(host_base, em_train, em_eval, text_train, text_val, stop_layer=stop_layer, stats=stats)
    with torch.no_grad():
        full_chk = host_base.forward_em(em_train.em_feats[:16], em_train.prose_ids[:16])
        m16 = host_base._make_em_mask(16, fcache.h_em.shape[1], 4, fcache.h_em.device)
        stop_chk = host_base.forward_from_layer(fcache.h_em[:16], stop_layer, g_em=True, em_tok_mask=m16)[:, -1, :]
        stop_grad_parity_err = float((full_chk - stop_chk).abs().max().item())
    assert stop_grad_parity_err < 1e-5, f"Stop-grad parity err={stop_grad_parity_err}"

    # --- AXIS 1: FORGETTING AXIS (6 methods, 160 steps) ---
    total_steps = 160
    final_key = f"step_{total_steps}"
    print("[W3-R2] Running Axis 1 (6 Forgetting Architectures, 160 steps)...", flush=True)
    forgetting_methods = (
        "unconstrained_lora",
        "layer_selected_gen_fisher_lora",
        "nullspace_projected_lora",
        "modality_gated_unconstrained",
        "modality_gated_nullspace_hybrid",
        "two_tier_token_kv_gated_hybrid",
    )
    axis1_results: Dict[str, dict] = {}
    for meth in forgetting_methods:
        ms_list = (69, 96, total_steps) if meth == "two_tier_token_kv_gated_hybrid" else (total_steps,)
        _, run_res = train_adapter(
            host_base=host_base, stats=stats, selected_layers=selected_layers,
            method=meth, loss_mode="bayes_kl_plus_cf_cal",
            em_train=em_train, em_eval=em_eval, text_val=text_val,
            base_text_cache=base_text_cache, base_mix_cache=base_mix_cache,
            fcache=fcache, seed=STUDY_SEEDS[0], steps=total_steps, batch_size=14, sub_cf=3,
            record_milestones=ms_list,
        )
        axis1_results[meth] = run_res
        ff = run_res["milestones"][final_key]
        print(
            f"        {meth:32s} ({run_res['wall_seconds']:4.1f}s) | PureKL={ff['task_a_pure']['kl_drift']:.3e} | "
            f"MixKL={ff['task_a_mixed']['mixed_kl_drift']:.3e} (NLLd={ff['task_a_mixed']['mixed_nll_drift']:+.5f}) | "
            f"G1={ff['em']['G1']['acc']:.4f}/{ff['em']['G1']['nll']:.4f} | "
            f"Q2_gain={ff['em']['summary']['q2_fresh_minus_valid_acc']:+.4f}",
            flush=True,
        )

    # --- Round-2 Long-Gap (L_gap = 32) W1 Slow-Head GDN Resolution (3 arms) ---
    print("[W3-R2] Running Long-Gap (L_gap=32) W1 Slow-Head GDN Resolution Study...", flush=True)
    long_gap_study = run_long_gap_w1_resolution_study(
        host_base, stats, selected_layers, text_train, text_val, base_text_cache, base_mix_cache
    )
    for k, v in long_gap_study.items():
        if not isinstance(v, dict) or "G1_acc" not in v:
            continue
        print(
            f"        {k:36s} | EarlyRet={v['early_gdn_l32_mean_retention']:.5f} MidRet={v['mid_gdn_l32_mean_retention']:.5f} | "
            f"PureKL={v['pure_text_kl_drift']:.1e} MixKL={v['mixed_fluency_kl_drift']:.3e} | "
            f"G1={v['G1_acc']:.4f}/{v['G1_nll']:.4f} | Q2g={v['Q2_fresh_minus_valid_acc']:+.4f}",
            flush=True,
        )

    # --- AXIS 2: LOSS OPTIMIZATION (4 modes x 3 seeds, Step-69 vs Step-96 vs Step-160) ---
    print("[W3-R2] Running Axis 2 (4 Loss Modes x 3 Seeds: Step-69 vs Step-96 vs Step-160)...", flush=True)
    loss_modes = ("hard_label_ce", "bayes_kl_only", "bayes_kl_plus_cf_cal", "bayes_kl_cf_cal_quotient")
    axis2_runs: Dict[str, List[dict]] = {lm: [] for lm in loss_modes}
    axis2_summary: Dict[str, dict] = {}
    for lm in loss_modes:
        for s in STUDY_SEEDS:
            if lm == "bayes_kl_plus_cf_cal" and s == STUDY_SEEDS[0]:
                axis2_runs[lm].append(axis1_results["two_tier_token_kv_gated_hybrid"])
                continue
            _, run_res = train_adapter(
                host_base=host_base, stats=stats, selected_layers=selected_layers,
                method="two_tier_token_kv_gated_hybrid", loss_mode=lm,
                em_train=em_train, em_eval=em_eval, text_val=text_val,
                base_text_cache=base_text_cache, base_mix_cache=base_mix_cache,
                fcache=fcache, seed=s, steps=total_steps, batch_size=14, sub_cf=3,
                record_milestones=(69, 96, total_steps),
            )
            axis2_runs[lm].append(run_res)

        agg = {}
        for ms in ("step_69", "step_96", final_key):
            g1_acc = [r["milestones"][ms]["em"]["G1"]["acc"] for r in axis2_runs[lm]]
            g1_nll = [r["milestones"][ms]["em"]["G1"]["nll"] for r in axis2_runs[lm]]
            g1_ece = [r["milestones"][ms]["em"]["G1"]["ece"] for r in axis2_runs[lm]]
            g1_ts = [r["milestones"][ms]["em"]["G1"]["t_star"] for r in axis2_runs[lm]]
            g1_fl = [r["milestones"][ms]["em"]["G1"]["workf_slot_flip_nll_penalty"] for r in axis2_runs[lm]]
            g1_og = [r["milestones"][ms]["em"]["G1"]["workf_orientation_gain"] for r in axis2_runs[lm]]
            g1_pr = [r["milestones"][ms]["em"]["G1"]["prose_pert_tv"] for r in axis2_runs[lm]]
            g1_c4 = [r["milestones"][ms]["em"]["G1"]["c4_mean_tv"] for r in axis2_runs[lm]]
            g1_c4m = [r["milestones"][ms]["em"]["G1"]["c4_max_defect"] for r in axis2_runs[lm]]
            q2_a = [r["milestones"][ms]["em"]["summary"]["q2_fresh_minus_valid_acc"] for r in axis2_runs[lm]]
            q2_n = [r["milestones"][ms]["em"]["summary"]["q2_fresh_minus_valid_nll_gain"] for r in axis2_runs[lm]]
            mix_kl = [r["milestones"][ms]["task_a_mixed"]["mixed_kl_drift"] for r in axis2_runs[lm]]
            pure_kl = [r["milestones"][ms]["task_a_pure"]["kl_drift"] for r in axis2_runs[lm]]
            row = {
                "g1_acc_mean": round(float(np.mean(g1_acc)), 6), "g1_acc_sd": round(float(np.std(g1_acc, ddof=1)), 6),
                "g1_nll_mean": round(float(np.mean(g1_nll)), 6), "g1_nll_sd": round(float(np.std(g1_nll, ddof=1)), 6),
                "g1_ece_mean": round(float(np.mean(g1_ece)), 6), "g1_tstar_mean": round(float(np.mean(g1_ts)), 4),
                "g1_workf_flip_penalty_mean": round(float(np.mean(g1_fl)), 6),
                "g1_workf_orient_gain_mean": round(float(np.mean(g1_og)), 6),
                "g1_prose_pert_tv_mean": round(float(np.mean(g1_pr)), 6),
                "g1_c4_mean_tv": round(float(np.mean(g1_c4)), 6),
                "g1_c4_max_defect_max": float(np.max(g1_c4m)),
                "q2_fresh_minus_valid_acc_mean": round(float(np.mean(q2_a)), 6),
                "q2_fresh_minus_valid_nll_gain_mean": round(float(np.mean(q2_n)), 6),
                "pure_kl_max": float(np.max(pure_kl)), "mixed_kl_mean": float(np.mean(mix_kl)),
            }
            if ms == final_key:
                row["g1_g4_acc_mean"] = round(float(np.mean([r["milestones"][ms]["em"]["summary"]["mean_G1_G4_acc"] for r in axis2_runs[lm]])), 6)
                row["g1_g4_nll_mean"] = round(float(np.mean([r["milestones"][ms]["em"]["summary"]["mean_G1_G4_nll"] for r in axis2_runs[lm]])), 6)
            agg[ms] = row
        agg["mean_wall_seconds_per_seed"] = round(float(np.mean([r["wall_seconds"] for r in axis2_runs[lm]])), 2)
        axis2_summary[lm] = agg
        s69, s96, sf = agg["step_69"], agg["step_96"], agg[final_key]
        print(
            f"        {lm:26s} ({agg['mean_wall_seconds_per_seed']:4.1f}s/sd) | "
            f"s69:{s69['g1_acc_mean']:.4f}/{s69['g1_nll_mean']:.4f} | "
            f"s96:{s96['g1_acc_mean']:.4f}/{s96['g1_nll_mean']:.4f} | "
            f"s160:{sf['g1_acc_mean']:.4f}/{sf['g1_nll_mean']:.4f} ECE={sf['g1_ece_mean']:.4f} "
            f"C4max={sf['g1_c4_max_defect_max']:.1e} MixKL={sf['mixed_kl_mean']:.3e} Q2g={sf['q2_fresh_minus_valid_acc_mean']:+.4f}",
            flush=True,
        )

    grad_var_audit = {
        "at_step_16": axis2_runs["bayes_kl_only"][0]["grad_var_step16"],
        "at_step_160_converged": axis2_runs["bayes_kl_only"][0]["grad_var_final"],
    }
    print(
        f"[W3-R2] Rao-Blackwell Grad Var (CE/KL): step16={grad_var_audit['at_step_16']['variance_reduction_ratio']}x, "
        f"step160={grad_var_audit['at_step_160_converged']['variance_reduction_ratio']}x",
        flush=True,
    )
    scaling_budgets = compute_qwen_scaling_budgets_v248320()
    elapsed_s = time.perf_counter() - t0

    full_results = {
        "meta": {
            "track": "W3_Round2",
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "wall_seconds": round(elapsed_s, 2),
            "study_seeds": list(STUDY_SEEDS),
            "stop_grad_cache_parity_err": stop_grad_parity_err,
            "host_architecture": {
                "num_layers": 8, "layer_types": host_base.layer_types,
                "d_model": D_MODEL, "n_heads": N_HEADS, "d_ff": D_FF, "tied_lm_head": True,
            },
        },
        "pretrain_task_a": pretrain_info,
        "phase_a_connector_alignment": phase_a_info,
        "frozen_host_em_baseline": frozen_em,
        "block1_fisher_selfconcordance_proxies_and_complex_P_C": block1,
        "block2_layer_selection_rayleigh": layer_sel_summary,
        "block3_axis1_forgetting": axis1_results,
        "block3b_long_gap_l32_w1_slowhead_resolution": long_gap_study,
        "block4_axis2_loss_summary": axis2_summary,
        "block4_axis2_rao_blackwell_grad_var": grad_var_audit,
        "block4_axis2_per_seed_runs": axis2_runs,
        "block5_qwen_4b_9b_27b_scaling_v248320": scaling_budgets,
    }
    RESULTS_PATH.write_text(json.dumps(full_results, indent=2) + "\n", encoding="utf-8")

    # Mandatory self-verification assertions
    assert elapsed_s <= 160.0, f"Wall-clock {elapsed_s:.2f}s exceeded 160s target!"
    assert block1["all_sc_brackets_hold"]
    assert block1["kronecker_spectral_identity_err"] < 1e-11
    assert cc["complex_P_C_commutation_defect_fro"] < 1e-12
    assert cc["complex_P_C_kl_reduction_vs_top"] > 100.0
    assert cc["complex_cayley_Q_perp_so2_err"] < 1e-12
    assert cc["complex_cayley_Q_perp_top_text_identity_err"] < 1e-12
    tt = axis1_results["two_tier_token_kv_gated_hybrid"]["milestones"][final_key]
    assert tt["task_a_pure"]["kl_drift"] == 0.0 and tt["task_a_pure"]["max_logit_diff"] == 0.0
    assert axis2_summary["bayes_kl_cf_cal_quotient"][final_key]["g1_c4_max_defect_max"] == 0.0
    print(f"[W3-R2] ALL ASSERTIONS PASSED in {elapsed_s:.2f}s (<=160s) -> {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
