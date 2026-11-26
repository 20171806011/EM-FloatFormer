"""Track W1 (Round 2 Unified & Optimized <= 150s) CPU Verification:
1. Exact FP64 constructive algebraic & cross-track proofs:
   - [R1] Gated-DeltaNet <-> Online Complex Kalman / Sherman-Morrison-Woodbury RLS Isomorphism
   - [R1] Multi-clock orthogonal-key sufficient statistic & Wiener-bridge readout
   - [R1] Spectral dichotomy: standard beta in (0,1) vs Householder beta in (0,2), 8-neuron SwiGLU, 2-key GDN
   - [R1] Parallel WY triangular-solve exact equivalence to step recurrence
   - [R2] M3/ref_pool.py correlated-noise Schur-complement innovation & ledger deduplication parity inside GDN
   - [R2] W1 3-Key Schur-Locked GDN + W2 Complex-Cayley orthogonal readout exact global SO(2)/C4 identities
   - [R2] W4 surgical <= 2.20M parameter budget arithmetic (4B/9B/27B, |V|=248,320) & Delta_route_max(sigma_3) VoI bound
2. Pre-training W2 B_align = M_cand^+ F_real conditioning & cross-gap subspace stability audit across L_gap in {0, 16, 64, 128}
   comparing Frozen Host vs W1 warmgate vs W3 early-frozen Tier-1 Token-Position Gate.
3. Matched Hybrid [GDN x 3, FullAttn x 1] CPU Benchmark (exact Stage 3D G1..G4 + unseen G5(theta=pi) + Q2 C4 + OOD@128 + PureKL/MixKL):
   - L_gap = 16: prefix_only, attn_lora, deltanet_attn_lora_warmgate (2 seeds)
   - L_gap = 64: prefix_only, attn_lora, deltanet_attn_lora, deltanet_attn_lora_warmgate,
                 tier1_pos_gated_deltanet_early_frozen, unified_w1_w2_w3_warmgate_schur (2 seeds)

Run strictly via:
  uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B research/stage4_math_and_design/weight_mod/W1/verify_w1.py
"""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy
from scipy.special import ndtr
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.stage3_ministral_cache.schema import ROOTS, matched_sum
from research.stage3c.A import q2_reader
from research.stage3d_qwen_signed_phase.data_stage3d import (
    BLIND_CELLS,
    NS_BLIND_Q2_C4,
    NS_CFO_TRAIN,
    TRAIN_OMEGAS,
    TRAIN_TIMINGS,
    generate_paired_cfo_blocks,
    load_s17_encoder,
    public_phase_slots,
)

OUT_DIR = Path(__file__).resolve().parent
OUT_JSON = OUT_DIR / "results_w1.json"

# Deterministic Round-1-matched CPU host + parallel worker pool (<= 150s total wall-clock!)
HOST_SEED = 20260401
MODEL_SEEDS = (202611, 202622)
NS_BLIND_G5 = 950406  # disjoint namespace for unseen G5(|theta|=pi)

D_MODEL = 32
N_HEADS = 4
D_HEAD = 8
FFN_DIM = 64
PROSE_VOCAB_CLASSES = 16
NULLSPACE_TOP_K = 12  # retains >= 90% prose variance; 20 dims (10 complex doublets) intact for EM
LORA_RANK = 8
LORA_ALPHA = 16.0

TRAIN_STEPS = 112  # full joint Q3 + Q2 steps per arm (224 passes; 6 physical-core LJF -> <= 125s!)
BATCH_SIZE = 20
LR_CONNECTOR = 9.0e-3
LR_LORA = 5.0e-3
LR_GATE = 2.2e-2
WEIGHT_DECAY = 1e-4

# Cached triangular masks for parallel WY Gated-DeltaNet
_MASK_CACHE: Dict[Tuple[int, str, torch.dtype], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def _get_wy_masks(t_len: int, device: torch.device, dtype: torch.dtype):
    key = (t_len, str(device), dtype)
    if key not in _MASK_CACHE:
        ms = torch.tril(torch.ones((t_len, t_len), device=device, dtype=torch.bool), diagonal=-1)
        mc = torch.tril(torch.ones((t_len, t_len), device=device, dtype=torch.bool), diagonal=0)
        eye = torch.eye(t_len, device=device, dtype=dtype).view(1, 1, t_len, t_len)
        _MASK_CACHE[key] = (ms, mc, eye)
    return _MASK_CACHE[key]


# =============================================================================
# Part 1: Exact Constructive Mathematical Verifications (FP64)
# =============================================================================

def verify_kalman_rls_isomorphism() -> dict:
    """R1: Exact equivalence between Qwen3.5 Gated-DeltaNet and Complex Kalman / SMW-RLS."""
    rng = np.random.default_rng(7001)
    d_k, d_v = 8, 4
    k_em = np.zeros(d_k, dtype=np.float64)
    k_em[0] = 1.0
    Pi_C = np.zeros((d_v, 2), dtype=np.float64)
    Pi_C[0, 0] = 1.0
    Pi_C[1, 1] = 1.0

    A_log = 0.0
    dt_bias = 0.0

    def softplus_inv(y: float) -> float:
        return math.log(math.expm1(max(y, 1e-15)))

    def sigmoid(x: float) -> float:
        return 1.0 / (1.0 + math.exp(-x))

    sigma_phi = 0.15
    omega = 0.35
    events = [
        ("pilot", 0.0, 2.0),
        ("prose_block", 0.0, 0.0),
        ("duplicate", 0.0, 2.0),
        ("distractor", 0.0, 2.0),
        ("pilot", 0.0, 2.0),
        ("prose_block", 0.0, 0.0),
        ("pilot", 2.0, 4.0),
        ("pilot", 1.5, 0.0),
        ("pilot", 3.0, 6.0),
        ("prose_block", 0.0, 0.0),
    ]

    phi = 0.0
    g_true = np.exp(1j * 0.7)
    t_phys = 0.0
    z_kalman = np.zeros(2, dtype=np.float64)
    P_kalman = 1.0
    S_gdn = np.zeros((d_v, d_k), dtype=np.float64)

    max_err = max_prose_drift = max_dup_drift = max_distractor_drift = 0.0
    saved_first_y = None
    prose_ret_64 = 1.0

    for kind, dt, snr_db in events:
        if kind == "prose_block":
            z_before = Pi_C.T @ (S_gdn @ k_em)
            alpha_prod = 1.0
            for p_i in range(64):
                alpha_t = math.exp(-math.exp(A_log) * math.log1p(math.exp(-40.0 + dt_bias)))
                beta_t = sigmoid(-40.0)
                alpha_prod *= alpha_t
                k_prose = np.zeros(d_k, dtype=np.float64)
                k_prose[1 + (p_i % (d_k - 1))] = 1.0
                v_prose = rng.normal(size=d_v)
                S_gdn = alpha_t * (S_gdn - beta_t * np.outer(S_gdn @ k_prose, k_prose)) + beta_t * np.outer(v_prose, k_prose)
            max_prose_drift = max(max_prose_drift, float(np.max(np.abs(Pi_C.T @ (S_gdn @ k_em) - z_before))))
            prose_ret_64 = alpha_prod
            continue
        if kind in ("duplicate", "distractor"):
            z_before = Pi_C.T @ (S_gdn @ k_em)
            alpha_t = math.exp(-math.exp(A_log) * math.log1p(math.exp(-40.0 + dt_bias)))
            beta_t = sigmoid(-40.0)
            v_in = Pi_C @ (saved_first_y if kind == "duplicate" else rng.normal(size=2))
            S_gdn = alpha_t * (S_gdn - beta_t * np.outer(S_gdn @ k_em, k_em)) + beta_t * np.outer(v_in, k_em)
            drift = float(np.max(np.abs(Pi_C.T @ (S_gdn @ k_em) - z_before)))
            if kind == "duplicate":
                max_dup_drift = max(max_dup_drift, drift)
            else:
                max_distractor_drift = max(max_distractor_drift, drift)
            continue

        t_phys += dt
        phi += omega * dt + sigma_phi * math.sqrt(max(dt, 0.0)) * rng.normal()
        rho = 10.0 ** (snr_db / 10.0)
        noise_sd = math.sqrt(1.0 / (2.0 * rho))
        y_c = g_true * np.exp(1j * phi) + (rng.normal(scale=noise_sd) + 1j * rng.normal(scale=noise_sd))
        y_d = y_c * np.exp(-1j * omega * t_phys)
        y_vec = np.array([y_d.real, y_d.imag], dtype=np.float64)
        if saved_first_y is None:
            saved_first_y = y_vec.copy()

        alpha_exact = math.exp(-0.5 * (sigma_phi ** 2) * dt)
        z_pred = alpha_exact * z_kalman
        P_pred = (alpha_exact ** 2) * P_kalman + (1.0 - alpha_exact ** 2)
        K_exact = (P_pred * rho) / (1.0 + P_pred * rho)
        z_kalman = (1.0 - K_exact) * z_pred + K_exact * y_vec
        P_kalman = (1.0 - K_exact) * P_pred

        diff_arg = 0.5 * (sigma_phi ** 2) * dt
        a_raw = -40.0 if diff_arg == 0.0 else softplus_inv(diff_arg)
        b_raw = math.log(P_pred) + math.log(10.0) * (snr_db / 10.0)
        alpha_gdn = math.exp(-math.exp(A_log) * math.log1p(math.exp(a_raw + dt_bias)))
        beta_gdn = sigmoid(b_raw)
        S_gdn = alpha_gdn * (S_gdn - beta_gdn * np.outer(S_gdn @ k_em, k_em)) + beta_gdn * np.outer(Pi_C @ y_vec, k_em)
        max_err = max(max_err, float(np.max(np.abs(Pi_C.T @ (S_gdn @ k_em) - z_kalman))))

    # SMW matrix-RLS identity check
    p_t, rho_t, alpha_s = 0.73, 10.0 ** 0.2, 0.94
    k_vec = rng.normal(size=d_k); k_vec /= np.linalg.norm(k_vec)
    v_vec = rng.normal(size=d_v)
    Theta_prev = rng.normal(size=(d_v, d_k))
    Theta_bayes = (alpha_s * Theta_prev / p_t + rho_t * np.outer(v_vec, k_vec)) @ np.linalg.inv(
        np.eye(d_k) / p_t + rho_t * np.outer(k_vec, k_vec)
    )
    beta_smw = (p_t * rho_t) / (1.0 + p_t * rho_t)
    Theta_gdn = alpha_s * Theta_prev @ (np.eye(d_k) - beta_smw * np.outer(k_vec, k_vec)) + beta_smw * np.outer(v_vec, k_vec)
    smw_err = float(np.max(np.abs(Theta_bayes - Theta_gdn)))
    return {
        "kalman_max_abs_err": max_err,
        "smw_matrix_rls_max_abs_err": smw_err,
        "prose_64_max_drift": max_prose_drift,
        "prose_64_retention": prose_ret_64,
        "duplicate_copy_max_drift": max_dup_drift,
        "distractor_max_drift": max_distractor_drift,
    }


def verify_multiclock_and_m3_schur_gdn() -> dict:
    """R1+R2: Multi-clock orthogonal pool + M3/ref_pool.py correlated-noise Schur innovation & ledger deduplication."""
    rng = np.random.default_rng(7002)
    d_k, d_v = 16, 4
    K_clocks = 4
    D_clocks = rng.normal(size=(K_clocks, 2))
    P_clocks = np.array([1.5849, 2.5119, 1.0, 3.9811], dtype=np.float64)
    beta_const = 0.5
    S = np.zeros((d_v, d_k), dtype=np.float64)
    for j in range(K_clocks):
        k_j = np.zeros(d_k); k_j[j] = 1.0
        v_j = np.array([P_clocks[j] * D_clocks[j, 0], P_clocks[j] * D_clocks[j, 1], P_clocks[j], 0.0]) / beta_const
        S = S @ (np.eye(d_k) - beta_const * np.outer(k_j, k_j)) + beta_const * np.outer(v_j, k_j)
        for p in range(16):
            kp = np.zeros(d_k); kp[K_clocks + (p % (d_k - K_clocks))] = 1.0
            S = S @ (np.eye(d_k) - 0.3 * np.outer(kp, kp)) + 0.3 * np.outer(rng.normal(size=d_v), kp)

    q_sum = np.zeros(d_k); q_sum[:K_clocks] = 1.0
    err_sum = float(np.max(np.abs((S @ q_sum)[:3] - np.array([
        np.sum(P_clocks * D_clocks[:, 0]), np.sum(P_clocks * D_clocks[:, 1]), np.sum(P_clocks)
    ]))))
    times = np.array([1.0, 2.0, 4.0, 7.0])
    C = (0.10 ** 2) * np.minimum(times[:, None], times[None, :])
    c_h = (0.10 ** 2) * np.minimum(times, 3.0)
    w_bridge = np.linalg.solve(C, c_h)
    q_b = np.zeros(d_k); q_b[:K_clocks] = w_bridge
    err_bridge = float(np.max(np.abs((S @ q_b)[:3] - np.array([
        np.sum(w_bridge * P_clocks * D_clocks[:, 0]),
        np.sum(w_bridge * P_clocks * D_clocks[:, 1]),
        np.sum(w_bridge * P_clocks),
    ]))))

    # R2: Correlated complex Gaussian Schur-complement innovation inside Tier-1 Ledger-Gated GDN
    # Model: [Y_P; Y_N] | u ~ CN(M u, K), with non-zero correlation K_NP != 0 (M3 Eq 7.1-7.2)
    K_PP = np.array([[0.63 + 0j]])
    K_NN = np.array([[0.63 + 0j]])
    K_NP = np.array([[0.41 * np.exp(1j * 0.25)]])
    K_PN = K_NP.conj().T
    K_full = np.block([[K_PP, K_PN], [K_NP, K_NN]])
    M_P = np.array([[1.0 + 0j]])
    M_N = np.array([[1.0 * np.exp(1j * 0.35)]])
    M_full = np.vstack([M_P, M_N])

    u_true = np.exp(1j * 1.1)
    L_chol = np.linalg.cholesky(K_full)
    eps = L_chol @ ((rng.normal(size=(2, 1)) + 1j * rng.normal(size=(2, 1))) / math.sqrt(2.0))
    y_full = M_full * u_true + eps
    y_P, y_N = y_full[0:1], y_full[1:2]

    # Exact full-block log-likelihood over the 4 candidate carrier roots u in {1, i, -1, -i}
    ll_full = np.zeros(4, dtype=np.float64)
    K_inv = np.linalg.inv(K_full)
    for g in range(4):
        res = y_full - M_full * ROOTS[g]
        ll_full[g] = -float(np.real((res.conj().T @ K_inv @ res)[0, 0]))
    ll_full -= ll_full.max()

    # Exact sequential Tier-1 Ledger-Gated GDN storing (q_P, J_P) on key e_0 and Schur innovation (q_N|P, J_N|P) on key e_1
    seen_ids = set()
    S_ledger = np.zeros((4, 8), dtype=np.float64)  # rows: [Re(q), Im(q), J, 0]
    stream = [
        ("acq_0", True, "prior", y_P),
        ("acq_0", True, "duplicate", y_P),          # duplicate -> seen_ids masks beta=0
        ("acq_9", False, "distractor", y_N * 3.0),  # distractor -> shares_channel=False masks beta=0
        ("acq_1", True, "fresh_schur", y_N),        # fresh -> Schur complement innovation
    ]
    A_schur = K_NP @ np.linalg.inv(K_PP)
    S_schur = K_NN - A_schur @ K_PN
    S_inv = np.linalg.inv(S_schur)
    K_PP_inv = np.linalg.inv(K_PP)

    dup_drift = dist_drift = 0.0
    slot_idx = 0
    for acq_id, shares_ch, role, y_obs in stream:
        m_acc = 1.0 if (shares_ch and acq_id not in seen_ids) else 0.0
        S_before = S_ledger.copy()
        if m_acc == 1.0:
            seen_ids.add(acq_id)
            if role == "prior":
                q_c = complex((M_P.conj().T @ K_PP_inv @ y_obs)[0, 0])
                J_c = float(np.real((M_P.conj().T @ K_PP_inv @ M_P)[0, 0]))
            else:
                r_N = y_obs - A_schur @ y_P
                W_N = M_N - A_schur @ M_P
                q_c = complex((W_N.conj().T @ S_inv @ r_N)[0, 0])
                J_c = float(np.real((W_N.conj().T @ S_inv @ W_N)[0, 0]))
            k_slot = np.zeros(8); k_slot[slot_idx] = 1.0
            slot_idx += 1
            v_slot = np.array([q_c.real, q_c.imag, J_c, 0.0])
            # beta_t = 1 on fresh orthogonal slot (exact store without overwrite)
            S_ledger = S_ledger @ (np.eye(8) - np.outer(k_slot, k_slot)) + np.outer(v_slot, k_slot)
        else:
            # Tier-1 gate clamps alpha=1, beta=0 => exact identity
            if role == "duplicate":
                dup_drift = float(np.max(np.abs(S_ledger - S_before)))
            elif role == "distractor":
                dist_drift = float(np.max(np.abs(S_ledger - S_before)))

    q_read = np.zeros(8); q_read[:slot_idx] = 1.0
    agg = S_ledger @ q_read  # [Re(q_tot), Im(q_tot), J_tot, 0]
    q_tot = complex(agg[0], agg[1])
    J_tot = agg[2]
    ll_gdn = np.array([2.0 * np.real(np.conj(ROOTS[g]) * q_tot) - J_tot for g in range(4)], dtype=np.float64)
    ll_gdn -= ll_gdn.max()
    schur_ll_err = float(np.max(np.abs(ll_gdn - ll_full)))

    return {
        "same_clock_sum_max_abs_err": err_sum,
        "wiener_bridge_readout_max_abs_err": err_bridge,
        "m3_correlated_schur_gdn_ll_max_err": schur_ll_err,
        "m3_ledger_duplicate_drift": dup_drift,
        "m3_ledger_distractor_drift": dist_drift,
    }


def verify_spectral_cayley_schur_gdn() -> dict:
    """R1+R2: Spectral dichotomy + SwiGLU + 3-Key Schur-Locked GDN composed with W2 Complex-Cayley."""
    rng = np.random.default_rng(7003)

    # R1: 2-step beta in (0,1) real positive spectrum + 3-step beta=0.8 damping + Householder beta=2
    max_imag_2step = 0.0
    for _ in range(1000):
        b1, b2 = rng.uniform(0.01, 0.99, size=2)
        k1 = rng.normal(size=2); k1 /= np.linalg.norm(k1)
        k2 = rng.normal(size=2); k2 /= np.linalg.norm(k2)
        eigs = np.linalg.eigvals((np.eye(2) - b1 * np.outer(k1, k1)) @ (np.eye(2) - b2 * np.outer(k2, k2)))
        max_imag_2step = max(max_imag_2step, float(np.max(np.abs(eigs.imag))))

    M3 = np.eye(2)
    for ang in (0.0, math.pi / 3.0, 2.0 * math.pi / 3.0):
        u = np.array([math.cos(ang), math.sin(ang)])
        M3 = M3 @ (np.eye(2) - 0.8 * np.outer(u, u))
    eigs3 = np.linalg.eigvals(M3)

    hh_err = 0.0
    for theta in np.append(np.linspace(-2.0 * math.pi, 2.0 * math.pi, 129), [1.60, -1.60, math.pi]):
        psi = theta / 2.0
        k1 = np.array([1.0, 0.0])
        k2 = np.array([math.cos(psi), math.sin(psi)])
        H = (np.eye(2) - 2.0 * np.outer(k2, k2)) @ (np.eye(2) - 2.0 * np.outer(k1, k1))
        R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
        hh_err = max(hh_err, float(np.max(np.abs(H - R))))

    # R1: 8-neuron SwiGLU complex mult
    def silu(x):
        return x / (1.0 + np.exp(-x))

    u_s = rng.normal(size=(200, 2))
    th_s = np.append(rng.uniform(-4.0 * math.pi, 4.0 * math.pi, size=196), [1.60, -1.60, math.pi, -math.pi])
    a1, a2, b1, b2 = np.cos(th_s), np.sin(th_s), u_s[:, 0], u_s[:, 1]
    swiglu_re = (silu(a1) - silu(-a1)) * b1 - (silu(a2) - silu(-a2)) * b2
    swiglu_im = (silu(a2) - silu(-a2)) * b1 + (silu(a1) - silu(-a1)) * b2
    exact_re = b1 * a1 - b2 * a2
    exact_im = b1 * a2 + b2 * a1
    swiglu_err = float(max(np.max(np.abs(swiglu_re - exact_re)), np.max(np.abs(swiglu_im - exact_im))))

    # R2: W1 3-Key Schur-Locked GDN (V_1 n=1 + V_2 n=2, zero DC) composed with W2 Complex-Cayley Q_C
    # Real C4 DFT basis F_real in R^{4 x 3} on 1^perp = V_1 (+) V_2
    F_real = np.array([
        [1.0 / math.sqrt(2.0), 0.0, 0.5],
        [0.0, 1.0 / math.sqrt(2.0), -0.5],
        [-1.0 / math.sqrt(2.0), 0.0, 0.5],
        [0.0, -1.0 / math.sqrt(2.0), -0.5],
    ], dtype=np.float64)
    # Build Complex-Cayley unitary Q_C in U(m) via Woodbury (m=8 complex pairs => d=16)
    m_c, r_c = 8, 2
    U_c = rng.normal(size=(m_c, r_c)) + 1j * rng.normal(size=(m_c, r_c))
    V_c = rng.normal(size=(m_c, r_c)) + 1j * rng.normal(size=(m_c, r_c))
    A_c = np.hstack([U_c, -V_c])
    B_c = np.hstack([V_c, U_c])
    Q_c = np.eye(m_c, dtype=np.complex128) - A_c @ np.linalg.solve(
        np.eye(2 * r_c, dtype=np.complex128) + 0.5 * (B_c.conj().T @ A_c), B_c.conj().T
    )
    cayley_orth_err = float(np.linalg.norm(Q_c.conj().T @ Q_c - np.eye(m_c), ord="fro"))
    # Realify Q_c to Q_real in SO(16) and verify SO(2) commutation with J_16
    Q_real = np.zeros((2 * m_c, 2 * m_c), dtype=np.float64)
    J_16 = np.zeros((2 * m_c, 2 * m_c), dtype=np.float64)
    for r_i in range(m_c):
        J_16[2 * r_i : 2 * r_i + 2, 2 * r_i : 2 * r_i + 2] = [[0.0, -1.0], [1.0, 0.0]]
        for c_i in range(m_c):
            z = Q_c[r_i, c_i]
            Q_real[2 * r_i : 2 * r_i + 2, 2 * c_i : 2 * c_i + 2] = [[z.real, -z.imag], [z.imag, z.real]]
    cayley_so2_err = float(np.linalg.norm(Q_real @ J_16 - J_16 @ Q_real, ord="fro"))

    # 3-Key Schur-Locked GDN state S in R^{3 x 3} storing [V_1 block (2x2); V_2 scalar (1x1)]
    c1_re, c1_im, c2 = 0.72, -0.31, 0.45
    S_schur = np.array([
        [c1_re, -c1_im, 0.0],
        [c1_im,  c1_re, 0.0],
        [0.0,    0.0,   c2],
    ], dtype=np.float64)

    def eval_schur_gdn(theta: float, V: float) -> np.ndarray:
        q = np.array([
            math.exp(-0.5 * V) * math.cos(theta),
            math.exp(-0.5 * V) * math.sin(theta),
            math.exp(-2.0 * V) * math.cos(2.0 * theta),
        ], dtype=np.float64)
        return F_real @ (S_schur @ q)

    R1 = np.roll(np.eye(4), 1, axis=0)
    per2pi_err = qtr_cov_err = 0.0
    for th in (-1.60, 0.35, 1.60, math.pi):
        v0 = eval_schur_gdn(th, 0.04)
        per2pi_err = max(per2pi_err, float(np.max(np.abs(eval_schur_gdn(th + 2.0 * math.pi, 0.04) - v0))))
        qtr_cov_err = max(qtr_cov_err, float(np.max(np.abs(eval_schur_gdn(th + 0.5 * math.pi, 0.04) - R1 @ v0))))

    return {
        "beta_01_2step_max_imag_eig": max_imag_2step,
        "beta_01_3step_beta08_modulus": float(np.abs(eigs3[0])),
        "beta_01_3step_beta08_angle_rad": float(abs(np.angle(eigs3[0]))),
        "householder_beta2_so2_max_abs_err": hh_err,
        "swiglu_8neuron_complex_mult_max_abs_err": swiglu_err,
        "cayley_unitary_orth_err": cayley_orth_err,
        "cayley_so2_commutation_err": cayley_so2_err,
        "w1_w2_3key_schur_gdn_2pi_err": per2pi_err,
        "w1_w2_3key_schur_gdn_qtr_cov_err": qtr_cov_err,
    }


def verify_surgical_budget_and_voi_bound(prior_s17: np.ndarray) -> dict:
    """R2: Machine-verify exact <=2.20M parameter table (4B/9B/27B, |V|=248,320) & W4 Delta_route_max(sigma_3)."""
    def calc_budget(
        name: str, L_tot: int, d: int, d_ff: int,
        n_fa_sel: int, n_dn_sel: int, n_dn_early_exc: int,
        D_q: int, D_kv: int, D_qkv: int, D_v: int, h_em: int,
        r_fa: int, rc_o: int, r_dn: int, r_schur: int, m1_conn: int, r_ffn_opt: int, n_ffn_opt: int,
    ) -> dict:
        # Gated q_proj (2*D_q x d), k_proj (D_kv x d), v_proj (D_kv x d) LoRA + Complex-Cayley o_proj (2*d*rc_o)
        fa_qkv = n_fa_sel * r_fa * ((2 * D_q + d) + 2 * (D_kv + d))
        fa_o_cayley = n_fa_sel * (2 * d * rc_o)
        # Head-masked GDN gates (h_em rows of in_proj_a & in_proj_b + A_log + dt_bias)
        dn_gates = n_dn_sel * h_em * (2 * d + 2)
        dn_early_gates = n_dn_early_exc * h_em * (2 * d + 2)
        dn_qkv = n_dn_sel * r_dn * (D_qkv + d)
        dn_out = n_dn_sel * r_dn * (d + D_v)
        # W2 Schur-Irrep Dynamic Hyper-LoRA on final selected FA layer (rank r_schur)
        schur_readout = r_schur * d + 14  # exact W2 Table 3.1 formula (28,174 at d=2560, r=11)
        core_total = m1_conn + fa_qkv + fa_o_cayley + dn_gates + dn_qkv + dn_out + schur_readout
        with_early_exc = core_total + dn_early_gates
        opt_ffn = n_ffn_opt * r_ffn_opt * 3 * (d_ff + d)
        return {
            "model": name,
            "vocab_size": 248320,
            "layers_total": L_tot,
            "m1_connector": m1_conn,
            "fa_qkv_lora": fa_qkv,
            "fa_o_cayley": fa_o_cayley,
            "dn_warmgate_masked": dn_gates,
            "dn_qkv_out_lora": dn_qkv + dn_out,
            "w2_schur_irrep": schur_readout,
            "core_unified_total": core_total,
            "with_early_2layer_exception": with_early_exc,
            "optional_tier2_ffn_lora": opt_ffn,
            "total_with_opt_ffn": core_total + opt_ffn,
        }

    b4 = calc_budget("Qwen3.5-4B", 32, 2560, 9216, 5, 4, 2, 4096, 1024, 8192, 4096, 4, 8, 4, 8, 11, 488400, 2, 2)
    b9 = calc_budget("Qwen3.5-9B", 32, 4096, 12288, 5, 4, 2, 4096, 1024, 8192, 4096, 4, 8, 4, 8, 11, 689616, 2, 2)
    b27 = calc_budget("Qwen3.8-27B", 64, 5120, 17408, 8, 6, 2, 6144, 1024, 10240, 6144, 6, 8, 4, 8, 11, 823760, 2, 2)
    assert b4["core_unified_total"] <= 2_184_964
    assert b4["with_early_2layer_exception"] <= 2_184_964
    assert b4["total_with_opt_ffn"] <= 2_184_964

    # Exact Delta_route_max(gamma = sigma_3_frozen / sigma_crit) on Stage 3D 2dB G1 s17 posterior
    x_gh, w_gh = np.polynomial.hermite.hermgauss(64)
    w_gh = w_gh / math.sqrt(math.pi)
    kappa = math.sqrt(2.0 * (10.0 ** 0.2))
    ang = 1.60 + math.sqrt(2.0 * 0.04) * x_gh
    c = ndtr(kappa * np.cos(ang + math.pi / 4.0))
    s = ndtr(kappa * np.sin(ang + math.pi / 4.0))
    row0 = np.sum(w_gh[:, None] * np.stack([c * s, (1 - c) * s, (1 - c) * (1 - s), c * (1 - s)], axis=-1), axis=0)
    K = np.stack([np.roll(row0, j) for j in range(4)], axis=0)
    P_star = np.clip(prior_s17[:4] @ K, 1e-12, 1.0)  # (4, 4) canonical G1 posteriors
    L_star = np.log(P_star)
    L_star -= L_star.mean(axis=-1, keepdims=True)
    # Exact KL from uniform (gamma = 0)
    h_star = float(-np.mean(np.sum(P_star * np.log(P_star), axis=-1)))
    kl_from_unif = float(math.log(4.0) - h_star)
    voi_curve = {}
    for gamma in (0.0, 0.10, 0.149, 0.20, 0.25, 0.40, 0.60, 0.80, 1.0):
        # Logit-norm attenuation under frozen routing contraction gamma = sigma_3_frozen / sigma_crit
        L_att = gamma * L_star
        L_att -= L_att.max(axis=-1, keepdims=True)
        Q_att = np.exp(L_att); Q_att /= Q_att.sum(axis=-1, keepdims=True)
        kl = float(np.mean(np.sum(P_star * (np.log(P_star) - np.log(np.clip(Q_att, 1e-12, 1.0))), axis=-1)))
        voi_curve[f"gamma_{gamma:.3f}"] = round(kl, 6)

    return {
        "parameter_budgets": {"Qwen3.5-4B": b4, "Qwen3.5-9B": b9, "Qwen3.8-27B": b27},
        "g1_s17_uniform_kl_ceiling_nats": round(kl_from_unif, 6),
        "delta_route_max_vs_sigma3_ratio": voi_curve,
    }


# =============================================================================
# Part 2: Deterministic Hybrid Gated-DeltaNet + Softmax Full-Attention CPU Host
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt() * self.weight


class LoRALinear(nn.Module):
    """Frozen linear base layer with optional Modality-Gated + Nullspace-Projected LoRA or Complex-Cayley."""

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.lora_rank = 0
        self.lora_scale = 0.0
        self.lora_A: Optional[nn.Parameter] = None
        self.lora_B: Optional[nn.Parameter] = None
        self.register_buffer("P_null", None)
        # Optional W2 Theorem 7 Complex-Symmetrized Nullspace Cayley left orthogonal adapter Q_C on output space
        self.cayley_rc = 0
        self.cayley_U_re: Optional[nn.Parameter] = None
        self.cayley_U_im: Optional[nn.Parameter] = None
        self.cayley_V_re: Optional[nn.Parameter] = None
        self.cayley_V_im: Optional[nn.Parameter] = None
        self.register_buffer("cayley_Pi_re", None)
        self.register_buffer("cayley_Pi_im", None)
        self.modality_gated = False

    def enable_lora(self, rank: int, alpha: float = LORA_ALPHA, P_null: Optional[torch.Tensor] = None,
                    modality_gated: bool = False, init_scale: float = 0.05) -> None:
        self.lora_rank = int(rank)
        self.lora_scale = float(alpha) / float(rank)
        self.lora_A = nn.Parameter(torch.randn(rank, self.in_features) * init_scale)
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.modality_gated = bool(modality_gated)
        if P_null is not None:
            self.register_buffer("P_null", P_null.clone())

    def enable_cayley(self, r_c: int = 2, modality_gated: bool = True, init_scale: float = 0.04,
                      Pi_null_c: Optional[torch.Tensor] = None) -> None:
        assert self.out_features % 2 == 0
        m = self.out_features // 2
        self.cayley_rc = int(r_c)
        self.cayley_U_re = nn.Parameter(torch.randn(m, r_c) * init_scale)
        self.cayley_U_im = nn.Parameter(torch.randn(m, r_c) * init_scale)
        self.cayley_V_re = nn.Parameter(torch.zeros(m, r_c))
        self.cayley_V_im = nn.Parameter(torch.zeros(m, r_c))
        self.modality_gated = bool(modality_gated)
        if Pi_null_c is not None:
            self.register_buffer("cayley_Pi_re", Pi_null_c.real.float().clone())
            self.register_buffer("cayley_Pi_im", Pi_null_c.imag.float().clone())

    def enable_full_finetune(self, modality_gated: bool = False) -> None:
        self.modality_gated = bool(modality_gated)
        self.weight.requires_grad_(True)
        if self.bias is not None:
            self.bias.requires_grad_(True)

    def _apply_cayley_to(self, y: torch.Tensor) -> torch.Tensor:
        # Exact W2 Theorem 7 Complex-Symmetrized Nullspace Cayley Q_C^perp = I - A (I + 0.5 B^* A)^{-1} B^* in U(m)
        U = torch.complex(self.cayley_U_re, self.cayley_U_im)
        V = torch.complex(self.cayley_V_re, self.cayley_V_im)
        if self.cayley_Pi_re is not None and self.cayley_Pi_im is not None:
            Pi = torch.complex(self.cayley_Pi_re, self.cayley_Pi_im)
            U = Pi @ U
            V = Pi @ V
        A = torch.cat([U, -V], dim=-1)  # (m, 2rc)
        B = torch.cat([V, U], dim=-1)   # (m, 2rc)
        eye = torch.eye(2 * self.cayley_rc, device=y.device, dtype=torch.complex64)
        M_inv = torch.linalg.inv(eye + 0.5 * (B.conj().T @ A))
        y_c = torch.complex(y[..., 0::2], y[..., 1::2])  # (..., m)
        proj = torch.matmul(y_c, B.conj())               # (..., 2rc)
        out_c = y_c - torch.matmul(torch.matmul(proj, M_inv.T), A.T)
        return torch.stack([out_c.real, out_c.imag], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, g_em: float = 1.0, token_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self.modality_gated and g_em == 0.0:
            # W3 Theorem 3: bit-exact bypass when g_EM = 0 (zero FP addition!)
            return y
        if self.cayley_rc > 0:
            y_rot = self._apply_cayley_to(y)
            if token_mask is not None:
                y = torch.where(token_mask.unsqueeze(-1) > 0.5, y_rot, y)
            else:
                y = y_rot
        if self.lora_A is not None and self.lora_B is not None:
            x_in = F.linear(x, self.P_null) if self.P_null is not None else x
            delta = F.linear(F.linear(x_in, self.lora_A), self.lora_B) * self.lora_scale
            if token_mask is not None:
                delta = delta * token_mask.unsqueeze(-1)
            y = y + delta
        return y


def _orthogonal_init(param: torch.Tensor, gain: float = 1.0) -> None:
    with torch.no_grad():
        nn.init.orthogonal_(param, gain=gain)


def _apply_rope(x: torch.Tensor, base: float = 10000.0, rotary_frac: float = 0.5) -> torch.Tensor:
    _, _, t_len, d_head = x.shape
    d_rot = int(d_head * rotary_frac)
    d_rot = d_rot - (d_rot % 2)
    if d_rot == 0:
        return x
    half = d_rot // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=x.device, dtype=x.dtype) / half))
    pos = torch.arange(t_len, device=x.device, dtype=x.dtype)
    angles = pos[:, None] * inv_freq[None, :]
    cos, sin = angles.cos()[None, None, :, :], angles.sin()[None, None, :, :]
    x_rot, x_pass = x[..., :d_rot], x[..., d_rot:]
    x1, x2 = x_rot[..., 0::2], x_rot[..., 1::2]
    out_rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
    return torch.cat([out_rot, x_pass], dim=-1)


def _single_block_wy_gated_deltanet(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """Exact single-chunk WY triangular-solve formulation of Gated-DeltaNet."""
    b, h, t_len, d = q.shape
    ms, mc, eye = _get_wy_masks(t_len, q.device, q.dtype)
    G = g.cumsum(dim=-1)
    diff = (G.unsqueeze(-1) - G.unsqueeze(-2)).clamp(max=0.0)
    Gamma = torch.where(mc, diff.exp(), torch.zeros_like(diff))
    kk = torch.matmul(k, k.transpose(-1, -2))
    L = eye + torch.where(ms, beta.unsqueeze(-1) * Gamma * kk, torch.zeros_like(diff))
    rhs = (beta.unsqueeze(-1) * v).reshape(b * h, t_len, d)
    U = torch.linalg.solve_triangular(L.reshape(b * h, t_len, t_len), rhs, upper=False, unitriangular=True)
    U = U.reshape(b, h, t_len, d)
    return torch.matmul(Gamma * torch.matmul(q, k.transpose(-1, -2)), U)


def parallel_wy_gated_deltanet(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, g: torch.Tensor, beta: torch.Tensor
) -> torch.Tensor:
    """Exact chunked WY formulation of Gated-DeltaNet (FLA-style chunking; C=24/36, FP64 err ~2e-15)."""
    b, h, t_len, d = q.shape
    C = 24 if t_len <= 96 else 36
    if t_len <= C:
        return _single_block_wy_gated_deltanet(q, k, v, g, beta)
    pad = (C - (t_len % C)) % C
    if pad > 0:
        q = F.pad(q, (0, 0, 0, pad))
        k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        g = F.pad(g, (0, pad))
        beta = F.pad(beta, (0, pad))
    n_chunks = q.shape[2] // C
    qc = q.view(b, h, n_chunks, C, d)
    kc = k.view(b, h, n_chunks, C, d)
    vc = v.view(b, h, n_chunks, C, d)
    gc = g.view(b, h, n_chunks, C)
    bc = beta.view(b, h, n_chunks, C)

    ms, mc, eye = _get_wy_masks(C, q.device, q.dtype)
    G = gc.cumsum(dim=-1)
    diff = (G.unsqueeze(-1) - G.unsqueeze(-2)).clamp(max=0.0)
    Gamma = torch.where(mc, diff.exp(), torch.zeros_like(diff))
    kk = torch.matmul(kc, kc.transpose(-1, -2))
    L = eye.unsqueeze(2) + torch.where(ms, bc.unsqueeze(-1) * Gamma * kk, torch.zeros_like(diff))
    expG = G.exp().unsqueeze(-1)
    rhs = torch.cat([bc.unsqueeze(-1) * vc, expG * bc.unsqueeze(-1) * kc], dim=-1).reshape(b * h * n_chunks, C, 2 * d)
    sol = torch.linalg.solve_triangular(L.reshape(b * h * n_chunks, C, C), rhs, upper=False, unitriangular=True)
    sol = sol.reshape(b, h, n_chunks, C, 2 * d)
    U_c, W_c = sol[..., :d], sol[..., d:]

    S = torch.zeros(b, h, d, d, device=q.device, dtype=q.dtype)
    G_end = G[..., -1]
    decay_to_end = (G_end.unsqueeze(-1) - G).clamp(max=0.0).exp().unsqueeze(-1)
    qk_gamma = Gamma * torch.matmul(qc, kc.transpose(-1, -2))
    outs = []
    for c_i in range(n_chunks):
        U_i = U_c[:, :, c_i] - torch.matmul(W_c[:, :, c_i], S.transpose(-1, -2))
        o_intra = torch.matmul(qk_gamma[:, :, c_i], U_i)
        o_inter = torch.matmul(expG[:, :, c_i] * qc[:, :, c_i], S.transpose(-1, -2))
        outs.append(o_intra + o_inter)
        S = G_end[:, :, c_i].exp().unsqueeze(-1).unsqueeze(-1) * S + torch.matmul(
            U_i.transpose(-1, -2), decay_to_end[:, :, c_i] * kc[:, :, c_i]
        )
    return torch.cat(outs, dim=2)[:, :, :t_len, :]


def verify_parallel_wy_equivalence() -> dict:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7004)
        b, h, t_len, d = 3, 4, 80, 8  # crosses 4 WY chunks (C=24) to verify exact inter-chunk state carry
        k = F.normalize(torch.randn(b, h, t_len, d, dtype=torch.float64), dim=-1)
        q = F.normalize(torch.randn(b, h, t_len, d, dtype=torch.float64), dim=-1)
        v = torch.randn(b, h, t_len, d, dtype=torch.float64)
        g = -torch.rand(b, h, t_len, dtype=torch.float64) * 0.25
        beta = torch.sigmoid(torch.randn(b, h, t_len, dtype=torch.float64))
        S = torch.zeros(b, h, d, d, dtype=torch.float64)
        outs = []
        for t in range(t_len):
            at = g[:, :, t].exp().unsqueeze(-1).unsqueeze(-1)
            bt = beta[:, :, t].unsqueeze(-1).unsqueeze(-1)
            kt = k[:, :, t].unsqueeze(-1)
            vt = v[:, :, t].unsqueeze(-1)
            S = at * S + bt * (vt - at * torch.matmul(S, kt)) * kt.transpose(-1, -2)
            outs.append(torch.matmul(S, q[:, :, t].unsqueeze(-1)).squeeze(-1))
        O_ref = torch.stack(outs, dim=2)
        O_par = parallel_wy_gated_deltanet(q, k, v, g, beta)
        O_par32 = parallel_wy_gated_deltanet(q.float(), k.float(), v.float(), g.float(), beta.float())
    return {
        "fp64_max_abs_err": float((O_ref - O_par).abs().max().item()),
        "fp32_max_abs_err": float((O_ref.float() - O_par32).abs().max().item()),
    }


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = LoRALinear(d_model, hidden_dim, bias=False)
        self.up_proj = LoRALinear(d_model, hidden_dim, bias=False)
        self.down_proj = LoRALinear(hidden_dim, d_model, bias=False)
        _orthogonal_init(self.gate_proj.weight, gain=0.8)
        _orthogonal_init(self.up_proj.weight, gain=0.8)
        _orthogonal_init(self.down_proj.weight, gain=0.35)

    def forward(self, x: torch.Tensor, g_em: float = 1.0) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x, g_em=g_em)) * self.up_proj(x, g_em=g_em), g_em=g_em)


class GatedDeltaNetBlock(nn.Module):
    """Qwen3.5 Gated-DeltaNet block with optional W1 warmgate + W3 Tier-1 Token-Position Ledger Gate.

    Note: Query projection (in_proj_q) is separated from Key/Value write projections (in_proj_kv) so
    W3 Tier-1 Token-Position Gating (m_acc) strictly masks K/V/beta writes on non-EM tokens while
    allowing Q (via g_em + P_{perp,C}) to query the lossless Kalman-RLS state S_t at answer readout t_ans!
    """

    def __init__(self, d_model: int, n_heads: int, d_head: int, ffn_dim: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.norm1 = RMSNorm(d_model)
        self.in_proj_q = LoRALinear(d_model, n_heads * d_head, bias=False)
        self.in_proj_kv = LoRALinear(d_model, 2 * n_heads * d_head, bias=False)
        self.in_proj_a = LoRALinear(d_model, n_heads, bias=False)
        self.in_proj_b = LoRALinear(d_model, n_heads, bias=False)
        self.out_proj = LoRALinear(n_heads * d_head, d_model, bias=False)

        self.A_log = nn.Parameter(torch.linspace(-2.95, -1.20, n_heads))
        self.dt_bias = nn.Parameter(torch.full((n_heads,), 0.10))
        self.b_bias = nn.Parameter(torch.full((n_heads,), -0.50))

        self.tier1_pos_gated = False
        self.em_head_count = 1
        self.modality_gated_params = False
        self.dA_log: Optional[nn.Parameter] = None
        self.ddt_bias: Optional[nn.Parameter] = None
        self.db_bias: Optional[nn.Parameter] = None

        _orthogonal_init(self.in_proj_q.weight, gain=1.0)
        _orthogonal_init(self.in_proj_kv.weight, gain=1.0)
        _orthogonal_init(self.in_proj_a.weight, gain=0.45)
        _orthogonal_init(self.in_proj_b.weight, gain=0.45)
        _orthogonal_init(self.out_proj.weight, gain=0.45)

        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, ffn_dim)

    def enable_gate_deltas(self, warm_em_head: bool = False, tier1_pos_gated: bool = False,
                           modality_gated: bool = True, P_null: Optional[torch.Tensor] = None,
                           em_head_count: int = 1) -> None:
        self.tier1_pos_gated = bool(tier1_pos_gated)
        self.em_head_count = int(em_head_count)
        self.modality_gated_params = bool(modality_gated)
        self.in_proj_a.enable_lora(2, alpha=4.0, P_null=P_null, modality_gated=modality_gated)
        self.in_proj_b.enable_lora(2, alpha=4.0, P_null=P_null, modality_gated=modality_gated)
        self.dA_log = nn.Parameter(torch.zeros_like(self.A_log))
        self.ddt_bias = nn.Parameter(torch.zeros_like(self.dt_bias))
        self.db_bias = nn.Parameter(torch.zeros_like(self.b_bias))
        if warm_em_head:
            with torch.no_grad():
                self.ddt_bias[: self.em_head_count] = -3.5
                self.db_bias[: self.em_head_count] = -1.5

    def forward(
        self,
        x: torch.Tensor,
        g_em: float = 1.0,
        m_acc: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        b, t_len, _ = x.shape
        h = self.norm1(x)
        # Tier-2 (sequence-gated g_em + P_null): Q reads S_t at t_ans; Tier-1 (m_acc): K/V write ONLY on valid EM tokens!
        q = self.in_proj_q(h, g_em=g_em, token_mask=None)
        kv = self.in_proj_kv(h, g_em=g_em, token_mask=m_acc if self.tier1_pos_gated else None)
        k, v = kv.chunk(2, dim=-1)
        q = F.normalize(q.view(b, t_len, self.n_heads, self.d_head), p=2, dim=-1, eps=1e-6)
        k = F.normalize(k.view(b, t_len, self.n_heads, self.d_head), p=2, dim=-1, eps=1e-6)
        v = v.view(b, t_len, self.n_heads, self.d_head)

        A_eff = self.A_log
        dt_eff = self.dt_bias
        bb_eff = self.b_bias
        if self.dA_log is not None and not (self.modality_gated_params and g_em == 0.0):
            A_eff = A_eff + self.dA_log
            dt_eff = dt_eff + self.ddt_bias
            bb_eff = bb_eff + self.db_bias

        a_in = self.in_proj_a(h, g_em=g_em, token_mask=m_acc if self.tier1_pos_gated else None) + dt_eff
        g = (-torch.exp(A_eff).clamp(max=4.0) * F.softplus(a_in)).clamp(min=-20.0, max=0.0)
        beta = torch.sigmoid(self.in_proj_b(h, g_em=g_em, token_mask=m_acc if self.tier1_pos_gated else None) + bb_eff)

        # W3 Tier-1 Token-Position Ledger Gate on dedicated EM head 0 when g_em == 1:
        # Non-EM prose tokens, duplicate IDs, and independent distractors (m_acc == 0) clamp g_0 = 0 (alpha=1), beta_0 = 0!
        if self.tier1_pos_gated and g_em > 0.5 and m_acc is not None:
            em_mask = m_acc.unsqueeze(-1)  # (B, T, 1)
            g_em_h = g[:, :, : self.em_head_count] * em_mask
            b_em_h = beta[:, :, : self.em_head_count] * em_mask
            g = torch.cat([g_em_h, g[:, :, self.em_head_count :]], dim=-1)
            beta = torch.cat([b_em_h, beta[:, :, self.em_head_count :]], dim=-1)

        alpha = torch.exp(g)
        o_bhtd = parallel_wy_gated_deltanet(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            g.transpose(1, 2), beta.transpose(1, 2),
        )
        o = o_bhtd.transpose(1, 2).reshape(b, t_len, self.n_heads * self.d_head)
        x = x + self.out_proj(o, g_em=g_em)
        x = x + self.ffn(self.norm2(x), g_em=g_em)

        diag = {"alpha": alpha, "beta": beta, "k": k, "q": q, "v": v} if return_diagnostics else None
        return x, diag


class GatedFullAttentionBlock(nn.Module):
    """Qwen3.5 Full-Attention block with QK-Norm, partial RoPE, and optional W2 Complex-Cayley o_proj."""

    def __init__(self, d_model: int, n_heads: int, d_head: int, ffn_dim: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.norm1 = RMSNorm(d_model)
        self.q_proj = LoRALinear(d_model, n_heads * d_head, bias=False)
        self.k_proj = LoRALinear(d_model, n_heads * d_head, bias=False)
        self.v_proj = LoRALinear(d_model, n_heads * d_head, bias=False)
        self.o_proj = LoRALinear(n_heads * d_head, d_model, bias=False)
        self.q_norm = RMSNorm(d_head)
        self.k_norm = RMSNorm(d_head)
        self.tier1_kv_gated = False

        _orthogonal_init(self.q_proj.weight, gain=1.0)
        _orthogonal_init(self.k_proj.weight, gain=1.0)
        _orthogonal_init(self.v_proj.weight, gain=1.0)
        _orthogonal_init(self.o_proj.weight, gain=0.45)

        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        g_em: float = 1.0,
        m_acc: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        b, t_len, _ = x.shape
        h = self.norm1(x)
        kv_mask = m_acc if self.tier1_kv_gated else None
        q = self.q_proj(h, g_em=g_em).view(b, t_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(h, g_em=g_em, token_mask=kv_mask).view(b, t_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(h, g_em=g_em, token_mask=kv_mask).view(b, t_len, self.n_heads, self.d_head).transpose(1, 2)

        q = _apply_rope(self.q_norm(q), rotary_frac=0.5)
        k = _apply_rope(self.k_norm(k), rotary_frac=0.5)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones((t_len, t_len), device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask[None, None, :, :], -1e9)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(b, t_len, self.n_heads * self.d_head)
        x = x + self.o_proj(out, g_em=g_em)
        x = x + self.ffn(self.norm2(x), g_em=g_em)
        diag = {"attn": attn} if return_diagnostics else None
        return x, diag


class SchurIrrepReadoutAdapter(nn.Module):
    """W2 Theorem 5B Schur-Irrep Dynamic Hyper-LoRA anchored on V_cand = F_real (V_1 (+) V_2, zero DC).

    Operates directly on the final backbone answer hidden state h_ans (zero physical kernel / zero labels),
    using public (theta, V) to enforce exact 2*pi periodicity, exact quarter-turn phase covariance,
    and well-conditioned global extrapolation across all theta in [0, 2*pi) including G5(theta=pi).
    """

    def __init__(self, d_model: int, w_unembed: torch.Tensor, P_null: Optional[torch.Tensor] = None):
        super().__init__()
        F_real = torch.tensor([
            [1.0 / math.sqrt(2.0), 0.0, 0.5],
            [0.0, 1.0 / math.sqrt(2.0), -0.5],
            [-1.0 / math.sqrt(2.0), 0.0, 0.5],
            [0.0, -1.0 / math.sqrt(2.0), -0.5],
        ], dtype=torch.float32)
        # B_align = M_cand^+ F_real in residual hidden space: W_unembed @ B_align == F_real exactly!
        pinv_w = torch.linalg.pinv(w_unembed.float())  # (D_MODEL, 4)
        self.register_buffer("B_align", pinv_w @ F_real)  # (D_MODEL, 3)
        # W2 Theorem 5B: zero-parameter candidate-nullspace skip projector eliminating static n=0 DC leakage on 1^perp
        self.register_buffer("P_cand_null", torch.eye(d_model) - pinv_w @ w_unembed.float())
        if P_null is not None:
            self.register_buffer("P_null", P_null.clone())
        else:
            self.register_buffer("P_null", torch.eye(d_model))
        # Irrep-locked projections: A1 (2 x d) for V_1 fundamental, A2 (1 x d) for V_2 Nyquist
        self.A1 = nn.Parameter(torch.randn(2, d_model) * 0.08)
        self.A2 = nn.Parameter(torch.randn(1, d_model) * 0.08)
        # Complex-commuting scalar c_1 = (c1_re, c1_im) on V_1, and (c2_c, c2_s) on V_2
        self.c1_re = nn.Parameter(torch.tensor(0.5))
        self.c1_im = nn.Parameter(torch.tensor(0.0))
        self.c2_c = nn.Parameter(torch.tensor(0.4))
        self.c2_s = nn.Parameter(torch.tensor(0.0))

    def forward(self, h_ans: torch.Tensor, theta: torch.Tensor, var_phi: torch.Tensor) -> torch.Tensor:
        # Project input through P_null so pure prose activations receive minimal perturbation
        h_p = F.linear(h_ans, self.P_null)
        u1 = F.linear(h_p, self.A1)  # (B, 2)
        u2 = F.linear(h_p, self.A2)  # (B, 1)

        # Complex-commuting gain Phi(c1) u1 then R(theta) * exp(-V/2)
        v1_x = self.c1_re * u1[:, 0] - self.c1_im * u1[:, 1]
        v1_y = self.c1_im * u1[:, 0] + self.c1_re * u1[:, 1]
        a1 = torch.exp(-0.5 * var_phi)
        ct, st = torch.cos(theta), torch.sin(theta)
        out_v1_x = a1 * (ct * v1_x - st * v1_y)
        out_v1_y = a1 * (st * v1_x + ct * v1_y)

        # Nyquist V_2 harmonic * exp(-2V)
        a2 = torch.exp(-2.0 * var_phi)
        out_v2 = a2 * (self.c2_c * torch.cos(2.0 * theta) + self.c2_s * torch.sin(2.0 * theta)) * u2[:, 0]

        coords = torch.stack([out_v1_x, out_v1_y, out_v2], dim=-1)  # (B, 3)
        return F.linear(coords, self.B_align)  # (B, D_MODEL) delta added directly to residual before unembed!


class HybridDeltaNetAttnHost(nn.Module):
    """4-layer Qwen3.5 3:1 hybrid macro-block [GDN0, GDN1, GDN2, FullAttn3] + prose LM head + EM simplex readout."""

    def __init__(self, seed: int = HOST_SEED):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.gdn_layers = nn.ModuleList([
                GatedDeltaNetBlock(D_MODEL, N_HEADS, D_HEAD, FFN_DIM) for _ in range(3)
            ])
            self.full_attn = GatedFullAttentionBlock(D_MODEL, N_HEADS, D_HEAD, FFN_DIM)
            self.final_norm = RMSNorm(D_MODEL)

            # Fixed isotropic centered 4-candidate EM readout
            frame, r = torch.linalg.qr(torch.randn(D_MODEL, 3, dtype=torch.float64))
            frame = frame * torch.diagonal(r).sign()
            helmert = torch.tensor(
                [[1, 1, 1], [-1, 1, 1], [0, -2, 1], [0, 0, -3]], dtype=torch.float64
            ) / torch.tensor([2, 6, 12], dtype=torch.float64).sqrt()
            self.register_buffer("w_unembed", (helmert @ frame.T).float())

            # Fixed general-prose LM head for PureKL / MixKL anti-forgetting verification (W3!)
            self.register_buffer("w_prose_lm", torch.randn(PROSE_VOCAB_CLASSES, D_MODEL) * 0.65)

            # Structured low-rank general-prose vocabulary (concentrated in top-10 linguistic subspace, like real BPE!)
            u_ling, _ = torch.linalg.qr(torch.randn(D_MODEL, NULLSPACE_TOP_K))
            raw_prose = torch.randn(256, NULLSPACE_TOP_K) @ u_ling.T * 0.75 + torch.randn(256, D_MODEL) * 0.08
            self.register_buffer("prose_vocab", raw_prose)
            self.register_buffer("task_q3_ans", torch.randn(1, 1, D_MODEL) * 0.6)
            self.register_buffer("task_q2_ans", torch.randn(1, 1, D_MODEL) * 0.6)
            sink_dir = F.normalize(self.task_q3_ans.squeeze() + self.task_q2_ans.squeeze(), dim=-1)
            self.register_buffer("sink_token", (sink_dir * 2.2).view(1, 1, D_MODEL))

        self.schur_adapter: Optional[SchurIrrepReadoutAdapter] = None
        self.freeze_all()
        self.prose_null_projectors, self.prose_complex_null_projectors = self._compute_prose_nullspace_projectors()

    def freeze_all(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)

    @staticmethod
    def _real_and_complex_nullspace(hn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        C = (hn.T @ hn) / hn.shape[0]
        _, evecs = torch.linalg.eigh(C)
        U_k = evecs[:, -NULLSPACE_TOP_K:]
        P_real = torch.eye(D_MODEL) - U_k @ U_k.T
        # W2 Theorem 7: exact U(m)-invariant complexified covariance C_C = 0.5 * E[z z^*]
        z = torch.complex(hn[:, 0::2], hn[:, 1::2])  # (N, m)
        m = D_MODEL // 2
        C_c = 0.5 * (z.conj().T @ z) / z.shape[0]
        _, evecs_c = torch.linalg.eigh(C_c)
        kc = NULLSPACE_TOP_K // 2
        U_c = evecs_c[:, -kc:]
        Pi_c = torch.eye(m, dtype=torch.complex64) - U_c @ U_c.conj().T
        return P_real, Pi_c

    @torch.no_grad()
    def _compute_prose_nullspace_projectors(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Compute W3 real input nullspace P_{perp,C} & W2 Theorem 7 complex-symmetrized Pi_{perp,C}."""
        self.eval()
        seqs = []
        for s_i in range(16):
            idx = (torch.arange(48) + s_i * 13) % 256
            seqs.append(self.prose_vocab[idx])
        x = torch.stack(seqs, dim=0)  # (16, 48, D_MODEL)
        p_reals, p_comps = [], []
        h = x
        for gdn in self.gdn_layers:
            hn = gdn.norm1(h).reshape(-1, D_MODEL)
            pr, pc = self._real_and_complex_nullspace(hn)
            p_reals.append(pr); p_comps.append(pc)
            h, _ = gdn(h, g_em=0.0)
        hn = self.full_attn.norm1(h).reshape(-1, D_MODEL)
        pr, pc = self._real_and_complex_nullspace(hn)
        p_reals.append(pr); p_comps.append(pc)
        h, _ = self.full_attn(h, g_em=0.0)
        hn_f = self.final_norm(h).reshape(-1, D_MODEL)
        pr_f, pc_f = self._real_and_complex_nullspace(hn_f)
        p_reals.append(pr_f); p_comps.append(pc_f)
        return p_reals, p_comps  # [GDN0, GDN1, GDN2, FullAttn3, FinalNorm]

    def configure_arm(self, arm: str, seed: int) -> None:
        self.freeze_all()
        self.schur_adapter = None
        P_list = self.prose_null_projectors
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 1000)
            if arm == "prefix_only":
                return
            elif arm == "attn_lora":
                for proj in (self.full_attn.q_proj, self.full_attn.k_proj, self.full_attn.v_proj, self.full_attn.o_proj):
                    proj.enable_lora(LORA_RANK, LORA_ALPHA, modality_gated=False)
            elif arm in ("deltanet_attn_lora", "deltanet_attn_lora_warmgate"):
                for proj in (self.full_attn.q_proj, self.full_attn.k_proj, self.full_attn.v_proj, self.full_attn.o_proj):
                    proj.enable_lora(LORA_RANK, LORA_ALPHA, modality_gated=False)
                for gdn in self.gdn_layers:
                    gdn.in_proj_q.enable_lora(LORA_RANK, LORA_ALPHA, modality_gated=False)
                    gdn.in_proj_kv.enable_lora(LORA_RANK, LORA_ALPHA, modality_gated=False)
                    gdn.out_proj.enable_lora(LORA_RANK, LORA_ALPHA, modality_gated=False)
                    gdn.enable_gate_deltas(
                        warm_em_head=(arm == "deltanet_attn_lora_warmgate"),
                        tier1_pos_gated=False,
                        modality_gated=False,
                    )
            elif arm == "tier1_pos_gated_deltanet_early_frozen":
                # W3 early-freeze test: GDN0 & GDN1 100% FROZEN! Only late GDN2 + FullAttn3 adapted with
                # W3 Sequence Gate g_EM + Tier-1 Token-Position Gate m_acc(t) + Input Nullspace P_{perp, C} + W1 warmgate!
                gdn2 = self.gdn_layers[2]
                gdn2.in_proj_q.enable_lora(LORA_RANK, LORA_ALPHA, P_null=P_list[2], modality_gated=True)
                gdn2.in_proj_kv.enable_lora(LORA_RANK, LORA_ALPHA, P_null=P_list[2], modality_gated=True)
                gdn2.out_proj.enable_lora(LORA_RANK, LORA_ALPHA, modality_gated=True)
                gdn2.enable_gate_deltas(warm_em_head=True, tier1_pos_gated=True, modality_gated=True, P_null=P_list[2], em_head_count=2)
                self.full_attn.tier1_kv_gated = True
                for proj in (self.full_attn.q_proj, self.full_attn.k_proj, self.full_attn.v_proj):
                    proj.enable_lora(LORA_RANK, LORA_ALPHA, P_null=P_list[3], modality_gated=True)
                self.full_attn.o_proj.enable_lora(LORA_RANK, LORA_ALPHA, modality_gated=True)
            elif arm == "unified_w1_w2_w3_warmgate_schur":
                # Unified W1 x W2 x W3 x M3 surgical stack:
                # - Early GDN0/1 frozen (W3 savings) + Late GDN2 W1 warmgate (2 EM heads) + W3 Tier-1 Ledger Gate + P_{perp,C}
                # - FullAttn3 QKV with P_{perp,C} + Tier-1 KV gate + W2 Theorem 7 Complex-Symmetrized Nullspace Cayley o_proj
                # - W2 Schur-Irrep Dynamic Hyper-LoRA (V_1 (+) V_2) with zero-DC candidate-nullspace skip!
                Pi_list = self.prose_complex_null_projectors
                gdn2 = self.gdn_layers[2]
                gdn2.in_proj_q.enable_lora(LORA_RANK, LORA_ALPHA, P_null=P_list[2], modality_gated=True)
                gdn2.in_proj_kv.enable_lora(LORA_RANK, LORA_ALPHA, P_null=P_list[2], modality_gated=True)
                gdn2.out_proj.enable_cayley(r_c=2, modality_gated=True, Pi_null_c=Pi_list[3])
                gdn2.enable_gate_deltas(warm_em_head=True, tier1_pos_gated=True, modality_gated=True, P_null=P_list[2], em_head_count=2)
                self.full_attn.tier1_kv_gated = True
                for proj in (self.full_attn.q_proj, self.full_attn.k_proj, self.full_attn.v_proj):
                    proj.enable_lora(LORA_RANK, LORA_ALPHA, P_null=P_list[3], modality_gated=True)
                self.full_attn.o_proj.enable_cayley(r_c=2, modality_gated=True, Pi_null_c=Pi_list[4])
                self.schur_adapter = SchurIrrepReadoutAdapter(D_MODEL, self.w_unembed, P_null=P_list[4])
            else:
                raise ValueError(f"Unknown arm: {arm}")

    def forward_embeds(
        self,
        seq_embeds: torch.Tensor,
        g_em: float = 1.0,
        m_acc: Optional[torch.Tensor] = None,
        schur_theta: Optional[torch.Tensor] = None,
        schur_var: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False,
        return_prose_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        h = seq_embeds
        gdn_diags = []
        for gdn in self.gdn_layers:
            h, d = gdn(h, g_em=g_em, m_acc=m_acc, return_diagnostics=return_diagnostics)
            if return_diagnostics:
                gdn_diags.append(d)
        h, attn_diag = self.full_attn(h, g_em=g_em, m_acc=m_acc, return_diagnostics=return_diagnostics)
        normed_all = self.final_norm(h)
        ans_h = normed_all[:, -1, :]
        if self.schur_adapter is not None and g_em > 0.5 and schur_theta is not None and schur_var is not None:
            # W2 Theorem 5B: zero-DC candidate-nullspace skip eliminates n=0 static bias on 1^perp
            ans_h = F.linear(ans_h, self.schur_adapter.P_cand_null) + self.schur_adapter(ans_h, schur_theta, schur_var)
        logits = F.linear(ans_h, self.w_unembed)
        diag = None
        if return_diagnostics or return_prose_logits:
            diag = {}
            if return_diagnostics:
                diag.update({"gdn": gdn_diags, "attn": attn_diag, "ans_hidden": ans_h})
            if return_prose_logits:
                diag["prose_logits"] = F.linear(normed_all, self.w_prose_lm)
        return logits, diag


class SeparateModalityConnector(nn.Module):
    """Zero-leakage canonical connector emitting separate <W1>, <W2>, <PHASE>, <REF> soft tokens."""

    def __init__(self, prior_s17: np.ndarray, d_model: int = D_MODEL, P_embed_null: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("prior_s17", torch.as_tensor(prior_s17[:4], dtype=torch.float32))
        self.register_buffer("P_embed_null", P_embed_null.clone() if P_embed_null is not None else None)
        self.w1_emb = nn.Embedding(4, 12)
        self.w2_emb = nn.Embedding(4, 12)
        self.rel_emb = nn.Embedding(4, 12)
        self.w1_net = nn.Sequential(nn.Linear(12 + 4, 48), nn.GELU(), nn.Linear(48, 2 * d_model))
        self.w2_net = nn.Sequential(nn.Linear(12 + 12 + 4, 48), nn.GELU(), nn.Linear(48, d_model))
        self.phase_net = nn.Sequential(nn.Linear(8, 48), nn.GELU(), nn.Linear(48, 2 * d_model))
        self.ref_net = nn.Sequential(nn.Linear(8, 48), nn.GELU(), nn.Linear(48, d_model))
        self.out_norm = nn.LayerNorm(d_model)
        self.token_scale = nn.Parameter(torch.tensor(0.85))

    def _emit(self, x: torch.Tensor) -> torch.Tensor:
        y = self.out_norm(x) * self.token_scale.clamp(0.1, 2.5)
        if self.P_embed_null is not None:
            y = F.linear(y, self.P_embed_null)
        return y

    def encode_w1_w2_phase(self, z1: torch.Tensor, z2: torch.Tensor, phase_feats: torch.Tensor):
        u1, u2 = z1 % 4, z2 % 4
        rel = ((z2 // 4) - (z1 // 4)) % 4
        w1 = self.w1_net(torch.cat([self.w1_emb(u1), self.prior_s17[u1]], dim=-1)).view(-1, 2, D_MODEL)
        w2 = self.w2_net(torch.cat([self.w2_emb(u2), self.rel_emb(rel), self.prior_s17[u2]], dim=-1)).view(-1, 1, D_MODEL)
        ph = self.phase_net(phase_feats).view(-1, 2, D_MODEL)
        return self._emit(w1), self._emit(w2), self._emit(ph)

    def encode_ref(self, ref_feats: torch.Tensor) -> torch.Tensor:
        return self._emit(self.ref_net(ref_feats).view(-1, 1, D_MODEL))


class HybridEMSystem(nn.Module):
    """Assembles [SINK, <W1>(2), <W2>(1), <PHASE>(2), PROSE_GAP, <REF>..., ANS] + Causal M3 Ledger Mask m_acc(t)."""

    def __init__(self, host: HybridDeltaNetAttnHost, prior_s17: np.ndarray, seed: int,
                 use_embed_nullspace: bool = False):
        super().__init__()
        self.host = host
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            p_emb = host.prose_null_projectors[0] if use_embed_nullspace else None
            self.connector = SeparateModalityConnector(prior_s17, D_MODEL, P_embed_null=p_emb)

    def _prose_chunk(self, b: int, length: int, offset: int, device: torch.device) -> torch.Tensor:
        if length <= 0:
            return torch.zeros((b, 0, D_MODEL), device=device)
        idx = (torch.arange(length, device=device) + offset) % 256
        return self.host.prose_vocab[idx].unsqueeze(0).expand(b, -1, -1)

    def assemble_q3(self, z1: torch.Tensor, z2: torch.Tensor, phase_feats: torch.Tensor, l_gap: int):
        b = z1.shape[0]
        w1, w2, ph = self.connector.encode_w1_w2_phase(z1, z2, phase_feats)
        seq = torch.cat([
            self.host.sink_token.expand(b, -1, -1), w1, w2, ph,
            self._prose_chunk(b, l_gap, 0, z1.device),
            self.host.task_q3_ans.expand(b, -1, -1),
        ], dim=1)
        # m_acc(t) = 1 strictly on modality positions [1..5], 0 on sink, prose, and ans
        m_acc = torch.zeros((b, seq.shape[1]), device=z1.device)
        m_acc[:, 1:6] = 1.0
        return seq, m_acc

    def assemble_q2(
        self, z1: torch.Tensor, z2: torch.Tensor, phase_feats: torch.Tensor,
        ref_list: Sequence[torch.Tensor], l_gap: int,
        ref_ledger_valid: Optional[Sequence[float]] = None,
    ):
        b = z1.shape[0]
        w1, w2, ph = self.connector.encode_w1_w2_phase(z1, z2, phase_feats)
        sink = self.host.sink_token.expand(b, -1, -1)
        ans = self.host.task_q2_ans.expand(b, -1, -1)
        if ref_ledger_valid is None:
            ref_ledger_valid = [1.0] * len(ref_list)
        if len(ref_list) == 0:
            seq = torch.cat([sink, w1, w2, ph, self._prose_chunk(b, l_gap, 37, z1.device), ans], dim=1)
            m_acc = torch.zeros((b, seq.shape[1]), device=z1.device)
            m_acc[:, 1:6] = 1.0
        elif len(ref_list) == 1:
            g1, g2 = l_gap // 2, l_gap - (l_gap // 2)
            r1 = self.connector.encode_ref(ref_list[0])
            seq = torch.cat([sink, w1, w2, ph, self._prose_chunk(b, g1, 37, z1.device), r1,
                             self._prose_chunk(b, g2, 37 + g1, z1.device), ans], dim=1)
            m_acc = torch.zeros((b, seq.shape[1]), device=z1.device)
            m_acc[:, 1:6] = 1.0
            m_acc[:, 6 + g1] = float(ref_ledger_valid[0])
        else:
            g1, g2 = l_gap // 2, l_gap - (l_gap // 2)
            r1, r2 = self.connector.encode_ref(ref_list[0]), self.connector.encode_ref(ref_list[1])
            seq = torch.cat([sink, w1, w2, ph, self._prose_chunk(b, g1, 37, z1.device), r1,
                             self._prose_chunk(b, g2, 37 + g1, z1.device), r2,
                             self._prose_chunk(b, 4, 37 + l_gap, z1.device), ans], dim=1)
            m_acc = torch.zeros((b, seq.shape[1]), device=z1.device)
            m_acc[:, 1:6] = 1.0
            m_acc[:, 6 + g1] = float(ref_ledger_valid[0])
            m_acc[:, 6 + g1 + 1 + g2] = float(ref_ledger_valid[1])
        return seq, m_acc

    def predict_q3_logits(
        self, z1: torch.Tensor, z2: torch.Tensor, phase_feats: torch.Tensor,
        l_gap: int, theta_val: Optional[torch.Tensor] = None, var_val: Optional[torch.Tensor] = None,
        return_diagnostics: bool = False, return_prose_logits: bool = False,
    ):
        seq, m_acc = self.assemble_q3(z1, z2, phase_feats, l_gap)
        canon_logits, diag = self.host.forward_embeds(
            seq, g_em=1.0, m_acc=m_acc, schur_theta=theta_val, schur_var=var_val,
            return_diagnostics=return_diagnostics, return_prose_logits=return_prose_logits,
        )
        a1 = z1 // 4
        idx = (torch.arange(4, device=z1.device)[None, :] - a1[:, None]) % 4
        return canon_logits.gather(-1, idx), diag

    def predict_q2_logits(
        self, z1: torch.Tensor, z2: torch.Tensor, phase_feats: torch.Tensor,
        ref_list: Sequence[torch.Tensor], l_gap: int,
        ref_ledger_valid: Optional[Sequence[float]] = None,
        return_diagnostics: bool = False, return_prose_logits: bool = False,
    ):
        seq, m_acc = self.assemble_q2(z1, z2, phase_feats, ref_list, l_gap, ref_ledger_valid=ref_ledger_valid)
        # Q2 relies strictly on GDN+FullAttn pilot fusion (no Q3 theta Schur rotation!)
        return self.host.forward_embeds(
            seq, g_em=1.0, m_acc=m_acc, schur_theta=None, schur_var=None,
            return_diagnostics=return_diagnostics, return_prose_logits=return_prose_logits,
        )


# =============================================================================
# Part 3: Data, Exact Physical Oracles & Fast Canonical Table Evaluation
# =============================================================================

def make_phase_feats(omega: float, h: int, ell: int, t_p: float, sigma_phi: float) -> np.ndarray:
    slots6 = public_phase_slots(omega, h, ell, t_p, zero_arm=False)
    return np.concatenate([slots6, np.array([(sigma_phi ** 2) * (h + ell), (sigma_phi ** 2) * t_p], dtype=np.float32)])


def make_canonical_ref_feats(samples: np.ndarray, z1: np.ndarray, omega: float, t_p: float,
                             snr_db: float, shares_channel: bool) -> np.ndarray:
    t_raw = matched_sum(samples)
    a1 = (z1 // 4) % 4
    t_rot = t_raw * ROOTS[(-a1) % 4]
    t_derot = t_rot * np.exp(-1j * omega * t_p)
    m = len(z1)
    return np.stack([
        t_rot.real, t_rot.imag, np.abs(t_rot), t_derot.real, t_derot.imag,
        np.full(m, 1.0 if shares_channel else 0.0), np.full(m, float(snr_db) / 10.0), np.ones(m),
    ], axis=-1).astype(np.float32)


def exact_s17_q3_oracle_probs(prior_s17: np.ndarray, z1: np.ndarray, omega: float, h: int, ell: int,
                              sigma_phi: float, snr_db: float) -> np.ndarray:
    x_gh, w_gh = np.polynomial.hermite.hermgauss(64)
    w_gh = w_gh / math.sqrt(math.pi)
    theta, var = omega * (h + ell), (sigma_phi ** 2) * (h + ell)
    kappa = math.sqrt(2.0 * (10.0 ** (snr_db / 10.0)))
    ang = theta + math.sqrt(2.0 * var) * x_gh
    c, s = ndtr(kappa * np.cos(ang + math.pi / 4.0)), ndtr(kappa * np.sin(ang + math.pi / 4.0))
    row0 = np.sum(w_gh[:, None] * np.stack([c * s, (1 - c) * s, (1 - c) * (1 - s), c * (1 - s)], axis=-1), axis=0)
    K = np.stack([np.roll(row0, j) for j in range(4)], axis=0)
    u1, a1 = z1 % 4, z1 // 4
    canon = prior_s17[u1] @ K
    return np.take_along_axis(canon, (np.arange(4)[None, :] - a1[:, None]) % 4, axis=-1)


def exact_q2_same_clock_teacher(prior_s17: np.ndarray, z1: np.ndarray, ref_derot_sum: np.ndarray,
                                snr_db: float, var_tp: float) -> np.ndarray:
    """Fast vectorized causal Bayes teacher for Q2 (exact same-clock M3 Theorem 1 Gauss-Hermite posterior)."""
    x_gh, w_gh = np.polynomial.hermite.hermgauss(16)
    w_gh = w_gh / math.sqrt(math.pi)
    rho = 10.0 ** (snr_db / 10.0)
    phi_nodes = math.sqrt(2.0 * max(var_tp, 1e-8)) * x_gh  # (16,)
    # ref_derot_sum is already in window-1 frame i^{-a1} and derotated by e^{-i omega t_p}
    # Exponent: 2*rho * Re(S_derot * i^{-g} * e^{-i phi_node})
    rot_grid = np.exp(-1j * phi_nodes)[None, None, :] * (ROOTS[None, :, None].conj())  # (1, 4, 16)
    ll = 2.0 * rho * np.real(ref_derot_sum[:, None, None] * rot_grid)  # (N, 4, 16)
    ll -= ll.max(axis=(1, 2), keepdims=True)
    carrier = np.sum(w_gh[None, None, :] * np.exp(ll), axis=-1)  # (N, 4) over g' = G - a1
    carrier /= np.clip(carrier.sum(axis=-1, keepdims=True), 1e-12, None)
    # P(X1 = k) = sum_{g'} a_{u1}((k + g') % 4) * carrier[g']
    u1 = z1 % 4
    a_u1 = prior_s17[u1]  # (N, 4)
    out = np.zeros((len(z1), 4), dtype=np.float64)
    for k in range(4):
        idx = (k + np.arange(4)) % 4
        out[:, k] = np.sum(a_u1[:, idx] * carrier, axis=-1)
    out /= np.clip(out.sum(axis=-1, keepdims=True), 1e-12, None)
    return out.astype(np.float32)


def compute_acc_nll(logits_or_probs: np.ndarray, labels: np.ndarray, is_probs: bool = False) -> dict:
    if is_probs:
        probs = np.clip(logits_or_probs, 1e-12, 1.0)
        probs /= probs.sum(axis=-1, keepdims=True)
        logp = np.log(probs)
    else:
        shifted = logits_or_probs - logits_or_probs.max(axis=-1, keepdims=True)
        logp = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        probs = np.exp(logp)
    return {"acc": float(np.mean(probs.argmax(axis=-1) == labels)),
            "nll": float(-np.mean(logp[np.arange(len(labels)), labels]))}


def build_training_dataset(encoder, prior_s17: np.ndarray, blocks_per_setting: int = 8) -> dict:
    """Build Stage 3D NS_CFO_TRAIN (|omega|<=0.25, |theta|<=1.25) with exact causal physical teachers for W3 KL."""
    q3_z1, q3_z2, q3_ph, q3_th, q3_va, q3_y, q3_p, q3_ph_flip, q3_th_flip, q3_p_flip = ([] for _ in range(10))
    q2_buckets = {arm: {"z1": [], "z2": [], "ph": [], "r1": [], "r2": [], "y": [], "p": []}
                  for arm in ("none", "valid", "fresh_resample", "valid_plus_independent", "independent_only")}

    for t_idx, (h, ell, tp) in enumerate(TRAIN_TIMINGS):
        paired, _ = generate_paired_cfo_blocks(
            NS_CFO_TRAIN, t_idx, blocks_per_setting, h, ell, tp, TRAIN_OMEGAS, split_prefix="w1_train"
        )
        for omega in TRAIN_OMEGAS:
            m = paired[omega]
            z1, z2 = encoder.encode(m.y1), encoder.encode(m.y2)
            n = len(z1)
            ph = np.broadcast_to(make_phase_feats(omega, h, ell, tp, m.condition.sigma_phi), (n, 8)).copy()
            ph_flip = np.broadcast_to(make_phase_feats(-omega, h, ell, tp, m.condition.sigma_phi), (n, 8)).copy()
            p_q3 = exact_s17_q3_oracle_probs(prior_s17, z1, omega, h, ell, m.condition.sigma_phi, m.condition.snr_db).astype(np.float32)
            p_q3_flip = exact_s17_q3_oracle_probs(prior_s17, z1, -omega, h, ell, m.condition.sigma_phi, m.condition.snr_db).astype(np.float32)

            q3_z1.append(z1); q3_z2.append(z2); q3_ph.append(ph); q3_ph_flip.append(ph_flip)
            q3_th.append(np.full(n, omega * (h + ell), dtype=np.float32))
            q3_th_flip.append(np.full(n, -omega * (h + ell), dtype=np.float32))
            q3_va.append(np.full(n, (m.condition.sigma_phi ** 2) * (h + ell), dtype=np.float32))
            q3_y.append(m.k3_repeat); q3_p.append(p_q3); q3_p_flip.append(p_q3_flip)

            r_v1 = make_canonical_ref_feats(m.p_valid1, z1, omega, tp, m.condition.snr_db, True)
            r_v2 = make_canonical_ref_feats(m.p_valid2, z1, omega, tp, m.condition.snr_db, True)
            r_ind = make_canonical_ref_feats(m.p_indep, z1, omega, tp, m.condition.snr_db, False)

            s_v1 = r_v1[:, 3] + 1j * r_v1[:, 4]
            s_v2 = r_v2[:, 3] + 1j * r_v2[:, 4]
            var_tp = (m.condition.sigma_phi ** 2) * tp
            p_none = np.full((n, 4), 0.25, dtype=np.float32)
            p_valid = exact_q2_same_clock_teacher(prior_s17, z1, s_v1, m.condition.snr_db, var_tp)
            p_fresh = exact_q2_same_clock_teacher(prior_s17, z1, s_v1 + s_v2, m.condition.snr_db, var_tp)

            for arm_name, rf1, rf2, pt in (
                ("none", None, None, p_none),
                ("valid", r_v1, None, p_valid),
                ("fresh_resample", r_v1, r_v2, p_fresh),
                ("valid_plus_independent", r_v1, r_ind, p_valid),
                ("independent_only", r_ind, None, p_none),
            ):
                b = q2_buckets[arm_name]
                b["z1"].append(z1); b["z2"].append(z2); b["ph"].append(ph); b["y"].append(m.x1); b["p"].append(pt)
                if rf1 is not None: b["r1"].append(rf1)
                if rf2 is not None: b["r2"].append(rf2)

    q3_data = {
        "z1": torch.from_numpy(np.concatenate(q3_z1)).long(),
        "z2": torch.from_numpy(np.concatenate(q3_z2)).long(),
        "ph": torch.from_numpy(np.concatenate(q3_ph)).float(),
        "ph_flip": torch.from_numpy(np.concatenate(q3_ph_flip)).float(),
        "theta": torch.from_numpy(np.concatenate(q3_th)).float(),
        "theta_flip": torch.from_numpy(np.concatenate(q3_th_flip)).float(),
        "var": torch.from_numpy(np.concatenate(q3_va)).float(),
        "y": torch.from_numpy(np.concatenate(q3_y)).long(),
        "p_teacher": torch.from_numpy(np.concatenate(q3_p)).float(),
        "p_teacher_flip": torch.from_numpy(np.concatenate(q3_p_flip)).float(),
    }
    q2_data = {}
    for arm_name, b in q2_buckets.items():
        e = {
            "z1": torch.from_numpy(np.concatenate(b["z1"])).long(),
            "z2": torch.from_numpy(np.concatenate(b["z2"])).long(),
            "ph": torch.from_numpy(np.concatenate(b["ph"])).float(),
            "y": torch.from_numpy(np.concatenate(b["y"])).long(),
            "p_teacher": torch.from_numpy(np.concatenate(b["p"])).float(),
        }
        if b["r1"]: e["r1"] = torch.from_numpy(np.concatenate(b["r1"])).float()
        if b["r2"]: e["r2"] = torch.from_numpy(np.concatenate(b["r2"])).float()
        q2_data[arm_name] = e
    return {"q3": q3_data, "q2": q2_data}


def build_blind_benchmarks(encoder, prior_s17: np.ndarray, q2_kernel: dict) -> dict:
    """Build exact Stage 3D G1..G4 + unseen G5(theta=pi) + Q2 C4 with paired physical oracles."""
    specs = list(BLIND_CELLS.items())
    # Add unseen G5_half_turn (|theta| = pi, V = 0.04, 8 blocks = 512 mothers x 2 signs)
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class _G5:
        abs_omega: float = math.pi / 4.0
        h: int = 2
        ell: int = 2
        t_p: float = 4.0
        snr_db: float = 2.0
        sigma_phi: float = 0.10
        namespace: int = NS_BLIND_G5
        cum_angle_abs: float = math.pi

    specs.append(("G5_half_turn_pi", _G5()))

    q3_cells = {}
    for cid, spec in specs:
        omegas = (-spec.abs_omega, spec.abs_omega)
        paired, _ = generate_paired_cfo_blocks(
            spec.namespace, 0, 8, spec.h, spec.ell, spec.t_p, omegas,
            spec.snr_db, spec.sigma_phi, split_prefix=cid
        )
        signs_data = []
        y_all, ora_all = [], []
        for omega in omegas:
            m = paired[omega]
            z1, z2 = encoder.encode(m.y1), encoder.encode(m.y2)
            ph1 = make_phase_feats(omega, spec.h, spec.ell, spec.t_p, spec.sigma_phi)
            ora = exact_s17_q3_oracle_probs(prior_s17, z1, omega, spec.h, spec.ell, spec.sigma_phi, spec.snr_db)
            signs_data.append({
                "omega": omega,
                "theta": float(omega * (spec.h + spec.ell)),
                "var": float((spec.sigma_phi ** 2) * (spec.h + spec.ell)),
                "z1": z1, "z2": z2, "ph1": ph1, "y": m.k3_repeat,
            })
            y_all.append(m.k3_repeat); ora_all.append(ora)
        q3_cells[cid] = {
            "abs_omega": spec.abs_omega,
            "cum_angle": spec.cum_angle_abs,
            "signs": signs_data,
            "oracle": compute_acc_nll(np.concatenate(ora_all), np.concatenate(y_all), is_probs=True),
        }

    q2_map, _ = generate_paired_cfo_blocks(
        NS_BLIND_Q2_C4, 0, 8, 2, 2, 4.0, (0.35,), snr_db=2.0, sigma_phi=0.10, split_prefix="q2_c4_blind"
    )
    m = q2_map[0.35]
    z1, z2 = encoder.encode(m.y1), encoder.encode(m.y2)
    ph = np.broadcast_to(make_phase_feats(0.35, 2, 2, 4.0, 0.10), (len(z1), 8)).copy()
    r_v1 = make_canonical_ref_feats(m.p_valid1, z1, 0.35, 4.0, 2.0, True)
    r_v2 = make_canonical_ref_feats(m.p_valid2, z1, 0.35, 4.0, 2.0, True)
    r_ind = make_canonical_ref_feats(m.p_indep, z1, 0.35, 4.0, 2.0, False)

    t_v1, t_v2, t_ind = matched_sum(m.p_valid1), matched_sum(m.p_valid2), matched_sum(m.p_indep)
    oracle_q2 = {}
    for arm_name, rb in (
        ("none", lambda i: []),
        ("valid", lambda i: [(0, True, t_v1[i])]),
        ("fresh_resample", lambda i: [(0, True, t_v1[i]), (1, True, t_v2[i])]),
        ("valid_plus_independent", lambda i: [(0, True, t_v1[i]), (1, False, t_ind[i])]),
        ("duplicate_copy_raw", lambda i: [(0, True, t_v1[i]), (0, True, t_v1[i])]),
    ):
        probs = np.array([q2_reader.read(q2_kernel, int(z1[i]), int(z2[i]), rb(i)) for i in range(len(z1))])
        oracle_q2[arm_name] = compute_acc_nll(probs, m.x1, is_probs=True)

    return {
        "q3": q3_cells,
        "q2": {
            "z1": torch.from_numpy(z1).long(), "z2": torch.from_numpy(z2).long(),
            "ph": torch.from_numpy(ph).float(), "r_v1": torch.from_numpy(r_v1).float(),
            "r_v2": torch.from_numpy(r_v2).float(), "r_ind": torch.from_numpy(r_ind).float(),
            "y": m.x1, "oracle": oracle_q2,
        },
    }


# Bit-exact 64-entry canonical table evaluator for Q3 (8x speedup, zero approximation!)
_CANON_U1 = torch.arange(4).repeat_interleave(16)            # (64,)
_CANON_U2 = torch.arange(4).repeat_interleave(4).repeat(4)   # (64,)
_CANON_REL = torch.arange(4).repeat(16)                      # (64,)
_CANON_Z1 = _CANON_U1.clone()                                # a1 = 0
_CANON_Z2 = (_CANON_REL * 4 + _CANON_U2).clone()             # a2 = rel


@torch.no_grad()
def eval_q3_cell_exact_canonical(system: HybridEMSystem, cell: dict, l_gap: int) -> dict:
    logits_all, y_all = [], []
    for sd in cell["signs"]:
        ph64 = torch.from_numpy(sd["ph1"]).unsqueeze(0).expand(64, -1)
        th64 = torch.full((64,), sd["theta"], dtype=torch.float32)
        va64 = torch.full((64,), sd["var"], dtype=torch.float32)
        canon64, _ = system.predict_q3_logits(_CANON_Z1, _CANON_Z2, ph64, l_gap=l_gap, theta_val=th64, var_val=va64)
        canon_np = canon64.cpu().numpy()  # (64, 4)
        z1, z2 = sd["z1"], sd["z2"]
        u1, u2 = z1 % 4, z2 % 4
        a1 = z1 // 4
        rel = ((z2 // 4) - a1) % 4
        idx64 = u1 * 16 + u2 * 4 + rel
        raw = canon_np[idx64]  # (N, 4)
        lifted = np.take_along_axis(raw, (np.arange(4)[None, :] - a1[:, None]) % 4, axis=-1)
        logits_all.append(lifted)
        y_all.append(sd["y"])
    return compute_acc_nll(np.concatenate(logits_all, axis=0), np.concatenate(y_all, axis=0), is_probs=False)


@torch.no_grad()
def eval_q2_batched(system: HybridEMSystem, q2b: dict, ref_tensors: Sequence[torch.Tensor],
                    l_gap: int, ref_ledger_valid: Optional[Sequence[float]] = None,
                    chunk: int = 64, max_n: Optional[int] = None) -> dict:
    n = len(q2b["y"]) if max_n is None else min(len(q2b["y"]), int(max_n))
    outs = []
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        r_slice = [r[start:end] for r in ref_tensors]
        lg, _ = system.predict_q2_logits(
            q2b["z1"][start:end], q2b["z2"][start:end], q2b["ph"][start:end],
            r_slice, l_gap=l_gap, ref_ledger_valid=ref_ledger_valid,
        )
        outs.append(lg.cpu().numpy())
    return compute_acc_nll(np.concatenate(outs, axis=0), q2b["y"][:n], is_probs=False)


# =============================================================================
# Part 4: Routing Singular Values, W2 B_align Conditioning & W3 PureKL/MixKL
# =============================================================================

def compute_routing_jacobian_svd(system: HybridEMSystem, l_gap: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (singular_values_top3, V_right_3xKd) of D_answer projected onto 3D candidate subspace V_cand."""
    system.eval()
    z1_p = torch.tensor([0], dtype=torch.long)
    z2_p = torch.tensor([0], dtype=torch.long)
    ph_p = torch.from_numpy(make_phase_feats(0.40, 2, 2, 4.0, 0.10)).unsqueeze(0)
    seq0, m_acc = system.assemble_q3(z1_p, z2_p, ph_p, l_gap=l_gap)
    seq0 = seq0.detach()
    mod_slice = seq0[:, 1:6, :].clone().requires_grad_(True)
    full_seq = torch.cat([seq0[:, :1, :], mod_slice, seq0[:, 6:, :]], dim=1)
    _, diag_j = system.host.forward_embeds(full_seq, g_em=1.0, m_acc=m_acc, return_diagnostics=True)
    ans_h = diag_j["ans_hidden"][0]

    W_u = system.host.w_unembed.detach().cpu().numpy()
    _, _, Vt_u = np.linalg.svd(W_u, full_matrices=False)
    V_cand = torch.from_numpy(Vt_u[:3, :]).float()
    proj_ans = torch.matmul(V_cand, ans_h)
    rows = []
    for r in range(3):
        g_r = torch.autograd.grad(proj_ans[r], mod_slice, retain_graph=(r < 2))[0].reshape(-1)
        rows.append(g_r.detach().cpu().numpy())
    D_cand = np.stack(rows, axis=0)  # (3, 5*D_MODEL)
    _, sv, Vt = np.linalg.svd(D_cand, full_matrices=False)
    return sv, Vt[:3, :]


def audit_b_align_conditioning(prior_s17: np.ndarray) -> dict:
    """R2 Exp 1: Measure (sigma_1, sigma_2, sigma_3), kappa_3 = sigma_1/sigma_3, ||B_align||_2^2, and
    cross-gap right-subspace alignment sigma_min(V_3(64) V_3(128)^T) BEFORE training across 4 configs."""
    configs = (
        "frozen_base",
        "warmgate_all_gdn",
        "tier1_pos_gated_early_frozen",
        "tier1_pos_gated_all_gdn",
    )
    out = {}
    for cfg in configs:
        host = HybridDeltaNetAttnHost(seed=HOST_SEED)
        if cfg == "warmgate_all_gdn":
            for gdn in host.gdn_layers:
                gdn.enable_gate_deltas(warm_em_head=True, tier1_pos_gated=False, modality_gated=False)
        elif cfg == "tier1_pos_gated_early_frozen":
            host.gdn_layers[2].enable_gate_deltas(warm_em_head=True, tier1_pos_gated=True, modality_gated=True)
        elif cfg == "tier1_pos_gated_all_gdn":
            for gdn in host.gdn_layers:
                gdn.enable_gate_deltas(warm_em_head=True, tier1_pos_gated=True, modality_gated=True)
        sys_tmp = HybridEMSystem(host, prior_s17, seed=MODEL_SEEDS[0])
        per_gap = {}
        V_saved = {}
        for lg in (0, 16, 64, 128):
            sv, Vt = compute_routing_jacobian_svd(sys_tmp, l_gap=lg)
            V_saved[lg] = Vt
            s1, s2, s3 = float(sv[0]), float(sv[1]), float(sv[2])
            per_gap[str(lg)] = {
                "sigma_1": s1, "sigma_2": s2, "sigma_3": s3,
                "kappa_3_s1_over_s3": float(s1 / max(s3, 1e-12)),
                "b_align_norm": float(1.0 / max(s3, 1e-12)),
                "taylor_curv_amp_1_over_s3_sq": float(1.0 / max(s3 * s3, 1e-12)),
            }
        # Cross-gap frame drift: principal angle cosines between V_3(64) and V_3(128)
        cross_sv = np.linalg.svd(V_saved[64] @ V_saved[128].T, compute_uv=False)
        out[cfg] = {
            "per_gap": per_gap,
            "cross_gap_64_vs_128_min_cos": float(cross_sv[-1]),
            "cross_gap_64_vs_128_mean_cos": float(np.mean(cross_sv)),
        }
    return out


def measure_routing_and_forgetting(
    system: HybridEMSystem, base_host: HybridDeltaNetAttnHost, eval_gap: int
) -> dict:
    """Measure sigma_1..3(D_answer), GDN retention curve, FullAttn modality mass, PureKL, and MixKL."""
    system.eval()
    base_host.eval()
    z1_p = torch.tensor([0], dtype=torch.long)
    z2_p = torch.tensor([0], dtype=torch.long)
    ph_p = torch.from_numpy(make_phase_feats(0.40, 2, 2, 4.0, 0.10)).unsqueeze(0)

    ret_curve = {}
    attn_mass_eval = 0.0
    with torch.no_grad():
        for lg in (0, 16, 64, 128):
            seq, m_acc = system.assemble_q3(z1_p, z2_p, ph_p, l_gap=lg)
            _, diag = system.host.forward_embeds(seq, g_em=1.0, m_acc=m_acc, return_diagnostics=True)
            layer_rets = []
            for g_d in diag["gdn"]:
                alpha, beta, k = g_d["alpha"], g_d["beta"], g_d["k"]
                vec = k[:, 1].clone()
                if lg > 0:
                    for t in range(6, 6 + lg):
                        at = alpha[:, t, :, None]
                        bt = beta[:, t, :, None]
                        kt = k[:, t]
                        vec = at * (vec - bt * (vec * kt).sum(dim=-1, keepdim=True) * kt)
                h_ret = vec.norm(dim=-1)
                layer_rets.append((float(h_ret.mean().item()), float(h_ret.max(dim=-1).values.mean().item())))
            ret_curve[str(lg)] = {
                "mean_head_retention": float(np.mean([r[0] for r in layer_rets])),
                "best_head_retention": float(np.mean([r[1] for r in layer_rets])),
                "layer2_best_head_retention": layer_rets[2][1],
            }
            if lg == eval_gap:
                attn_mass_eval = float(diag["attn"]["attn"][:, :, -1, 1:6].sum(dim=-1).mean().item())

        # PureKL: pure prose sequences with g_em = 0.0
        pure_seqs = torch.stack([system.host.prose_vocab[(torch.arange(32) + i * 7) % 256] for i in range(4)], dim=0)
        _, base_pure = base_host.forward_embeds(pure_seqs, g_em=0.0, return_prose_logits=True)
        _, mod_pure = system.host.forward_embeds(pure_seqs, g_em=0.0, return_prose_logits=True)
        p0_pure = F.softmax(base_pure["prose_logits"], dim=-1)
        log_p0_pure = F.log_softmax(base_pure["prose_logits"], dim=-1)
        log_p1_pure = F.log_softmax(mod_pure["prose_logits"], dim=-1)
        pure_kl = float((p0_pure * (log_p0_pure - log_p1_pure)).sum(dim=-1).mean().item())
        pure_max_logit_diff = float((base_pure["prose_logits"] - mod_pure["prose_logits"]).abs().max().item())

        # MixKL: intervening prose tokens [6 .. 6+eval_gap) inside active EM sequence (g_em = 1.0)
        seq_mix, m_acc_mix = system.assemble_q3(z1_p, z2_p, ph_p, l_gap=eval_gap)
        _, base_mix = base_host.forward_embeds(seq_mix, g_em=0.0, return_prose_logits=True)
        _, mod_mix = system.host.forward_embeds(seq_mix, g_em=1.0, m_acc=m_acc_mix, return_prose_logits=True)
        sl = slice(6, 6 + eval_gap)
        p0_m = F.softmax(base_mix["prose_logits"][:, sl, :], dim=-1)
        lp0_m = F.log_softmax(base_mix["prose_logits"][:, sl, :], dim=-1)
        lp1_m = F.log_softmax(mod_mix["prose_logits"][:, sl, :], dim=-1)
        mix_kl = float((p0_m * (lp0_m - lp1_m)).sum(dim=-1).mean().item())

    sv, _ = compute_routing_jacobian_svd(system, l_gap=eval_gap)
    return {
        "eval_gap": eval_gap,
        "sigma_D_answer": [float(x) for x in sv],
        "sigma_min_D_answer": float(sv[2]),
        "kappa_3": float(sv[0] / max(sv[2], 1e-12)),
        "full_attn_mod_mass": attn_mass_eval,
        "deltanet_retention_vs_gap": ret_curve,
        "pure_kl": pure_kl,
        "pure_max_logit_diff": pure_max_logit_diff,
        "mix_kl": mix_kl,
    }


# =============================================================================
# Part 5: Matched Multi-Arm Training & Full Evaluation
# =============================================================================

def train_and_evaluate_arm(
    arm: str, l_gap: int, seed: int, prior_s17: np.ndarray,
    train_data: dict, blind_data: dict, base_host: HybridDeltaNetAttnHost,
) -> dict:
    host = HybridDeltaNetAttnHost(seed=HOST_SEED)
    host.configure_arm(arm, seed=seed)
    use_embed_null = arm in ("tier1_pos_gated_deltanet_early_frozen", "unified_w1_w2_w3_warmgate_schur")
    system = HybridEMSystem(host, prior_s17, seed=seed, use_embed_nullspace=use_embed_null)

    conn_params = list(system.connector.parameters())
    gate_params, lora_params = [], []
    for name, p in system.host.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ("in_proj_a", "in_proj_b", "dA_log", "ddt_bias", "db_bias", "A_log", "dt_bias", "b_bias")):
            gate_params.append(p)
        else:
            lora_params.append(p)
    host_params = gate_params + lora_params

    param_groups = [{"params": conn_params, "lr": LR_CONNECTOR}]
    if lora_params:
        param_groups.append({"params": lora_params, "lr": LR_LORA})
    if gate_params:
        param_groups.append({"params": gate_params, "lr": LR_GATE, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_STEPS, eta_min=2e-4)

    use_w3_distill = (arm == "unified_w1_w2_w3_warmgate_schur")
    use_mix_anchor = arm in ("tier1_pos_gated_deltanet_early_frozen", "unified_w1_w2_w3_warmgate_schur")
    use_ledger_mask = arm in ("tier1_pos_gated_deltanet_early_frozen", "unified_w1_w2_w3_warmgate_schur")

    rng = torch.Generator().manual_seed(seed + 777)
    q3_d, q2_d = train_data["q3"], train_data["q2"]
    n_q3, n_q2 = len(q3_d["y"]), len(q2_d["valid"]["y"])
    # Balanced Q2 arm cycle prioritizing valid/fresh Blackwell evidence + distractor rejection
    q2_arm_cycle = (
        "valid", "fresh_resample", "valid", "fresh_resample",
        "valid_plus_independent", "fresh_resample", "valid", "none",
        "fresh_resample", "independent_only",
    )

    # Anchor batch for fast W3 MixKL regularization
    z1_anc = torch.tensor([0, 1], dtype=torch.long)
    z2_anc = torch.tensor([0, 1], dtype=torch.long)
    ph_anc = torch.from_numpy(make_phase_feats(0.20, 2, 2, 4.0, 0.10)).unsqueeze(0).expand(2, -1)

    system.train()
    t0 = time.perf_counter()
    last_loss = 0.0
    for step in range(TRAIN_STEPS):
        # Full joint Q3 + Q2 step every iteration (eliminates alternating ping-pong; parallelized across CPU cores!)
        idx3 = torch.randint(0, n_q3, (BATCH_SIZE,), generator=rng)
        th3 = q3_d["theta"][idx3] if use_w3_distill else None
        va3 = q3_d["var"][idx3] if use_w3_distill else None
        logits_q3, _ = system.predict_q3_logits(
            q3_d["z1"][idx3], q3_d["z2"][idx3], q3_d["ph"][idx3], l_gap=l_gap, theta_val=th3, var_val=va3
        )
        if use_w3_distill:
            p_t3 = q3_d["p_teacher"][idx3].clamp_min(1e-7)
            loss_q3 = F.kl_div(F.log_softmax(logits_q3, dim=-1), p_t3, reduction="batchmean")
            # W3 Counterfactual opposite-sign margin every 4 steps
            if step % 4 == 0:
                logits_flip, _ = system.predict_q3_logits(
                    q3_d["z1"][idx3], q3_d["z2"][idx3], q3_d["ph_flip"][idx3],
                    l_gap=l_gap, theta_val=q3_d["theta_flip"][idx3], var_val=va3,
                )
                p_tf = q3_d["p_teacher_flip"][idx3].clamp_min(1e-7)
                loss_q3 = 0.5 * (loss_q3 + F.kl_div(F.log_softmax(logits_flip, dim=-1), p_tf, reduction="batchmean"))
                v_phys = p_t3.log() - p_tf.log()
                v_phys = v_phys - v_phys.mean(dim=-1, keepdim=True)
                v_pred = logits_q3 - logits_flip
                v_pred = v_pred - v_pred.mean(dim=-1, keepdim=True)
                loss_q3 = loss_q3 + 0.08 * F.mse_loss(v_pred, v_phys)
        else:
            loss_q3 = F.cross_entropy(logits_q3, q3_d["y"][idx3])

        q2_arm_name = q2_arm_cycle[step % len(q2_arm_cycle)]
        qd = q2_d[q2_arm_name]
        idx2 = torch.randint(0, n_q2, (BATCH_SIZE,), generator=rng)
        ref_list, r_mask = [], []
        if "r1" in qd:
            ref_list.append(qd["r1"][idx2])
            r_mask.append(0.0 if (use_ledger_mask and q2_arm_name == "independent_only") else 1.0)
        if "r2" in qd:
            ref_list.append(qd["r2"][idx2])
            r_mask.append(0.0 if (use_ledger_mask and q2_arm_name == "valid_plus_independent") else 1.0)

        logits_q2, _ = system.predict_q2_logits(
            qd["z1"][idx2], qd["z2"][idx2], qd["ph"][idx2], ref_list, l_gap=l_gap,
            ref_ledger_valid=r_mask if use_ledger_mask else None,
        )
        if use_w3_distill:
            loss_q2 = 1.2 * F.kl_div(F.log_softmax(logits_q2, dim=-1), qd["p_teacher"][idx2].clamp_min(1e-7), reduction="batchmean")
        else:
            loss_q2 = F.cross_entropy(logits_q2, qd["y"][idx2])

        loss = loss_q3 + loss_q2

        # Lightweight W3 mixed-turn prose KL anchor every 10 steps
        if use_mix_anchor and step % 10 == 0:
            seq_a, m_a = system.assemble_q3(z1_anc, z2_anc, ph_anc, l_gap=l_gap)
            with torch.no_grad():
                _, d0 = base_host.forward_embeds(seq_a.detach(), g_em=0.0, return_prose_logits=True)
                p0_a = F.softmax(d0["prose_logits"][:, 6 : 6 + l_gap, :], dim=-1)
            _, d1 = system.host.forward_embeds(seq_a, g_em=1.0, m_acc=m_a, return_prose_logits=True)
            lp1_a = F.log_softmax(d1["prose_logits"][:, 6 : 6 + l_gap, :], dim=-1)
            loss = loss + 0.35 * F.kl_div(lp1_a, p0_a, reduction="batchmean")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(conn_params + host_params, 2.0)
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.item())

    train_s = time.perf_counter() - t0
    assert math.isfinite(last_loss), f"Non-finite loss in {arm}"
    system.eval()

    # Fast bit-exact canonical evaluation for Q3 G1..G5 + OOD@128
    q3_metrics = {cid: eval_q3_cell_exact_canonical(system, cell, l_gap=l_gap) for cid, cell in blind_data["q3"].items()}
    q2b = blind_data["q2"]
    # Test genuine in-sequence Tier-1 Ledger Masking (tokens remain physically in sequence; m_acc=0 suppresses GDN/KV writes!)
    q2_specs = (
        ("none", [], None),
        ("valid", [q2b["r_v1"]], [1.0]),
        ("fresh_resample", [q2b["r_v1"], q2b["r_v2"]], [1.0, 1.0]),
        ("valid_plus_independent", [q2b["r_v1"], q2b["r_ind"]], [1.0, 0.0]),
        ("duplicate_copy_raw", [q2b["r_v1"], q2b["r_v1"]], [1.0, 0.0]),
    )
    q2_metrics = {
        name: eval_q2_batched(system, q2b, rts, l_gap=l_gap, ref_ledger_valid=lm if use_ledger_mask else None, chunk=64, max_n=192)
        for name, rts, lm in q2_specs
    }

    # Zero-shot OOD cross-gap test (16 -> 64, 64 -> 128)
    ood_gap = 64 if l_gap == 16 else 128
    ood_g1 = eval_q3_cell_exact_canonical(system, blind_data["q3"]["G1_main_extrap_both"], l_gap=ood_gap)
    ood_g5 = eval_q3_cell_exact_canonical(system, blind_data["q3"]["G5_half_turn_pi"], l_gap=ood_gap)
    ood_q2v = eval_q2_batched(system, q2b, [q2b["r_v1"]], l_gap=ood_gap, ref_ledger_valid=[1.0] if use_ledger_mask else None, chunk=64, max_n=64)
    ood_q2f = eval_q2_batched(system, q2b, [q2b["r_v1"], q2b["r_v2"]], l_gap=ood_gap, ref_ledger_valid=[1.0, 1.0] if use_ledger_mask else None, chunk=64, max_n=64)

    # Bitwise C4 x C4 quotient equivariance defect verification on G1
    with torch.no_grad():
        sd0 = blind_data["q3"]["G1_main_extrap_both"]["signs"][0]
        z1_eq = torch.from_numpy(sd0["z1"][:16]).long()
        z2_eq = torch.from_numpy(sd0["z2"][:16]).long()
        ph_eq = torch.from_numpy(sd0["ph1"]).unsqueeze(0).expand(16, -1)
        th_eq = torch.full((16,), sd0["theta"], dtype=torch.float32) if use_w3_distill else None
        va_eq = torch.full((16,), sd0["var"], dtype=torch.float32) if use_w3_distill else None
        base_eq, _ = system.predict_q3_logits(z1_eq, z2_eq, ph_eq, l_gap=l_gap, theta_val=th_eq, var_val=va_eq)
        c4_defect = 0.0
        for g_rot in (1, 2, 3):
            z1_r = (z1_eq % 4) + (((z1_eq // 4) + g_rot) % 4) * 4
            z2_r = (z2_eq % 4) + (((z2_eq // 4) + g_rot) % 4) * 4
            rot_eq, _ = system.predict_q3_logits(z1_r, z2_r, ph_eq, l_gap=l_gap, theta_val=th_eq, var_val=va_eq)
            expected = torch.roll(base_eq, shifts=g_rot, dims=-1)
            c4_defect = max(c4_defect, float((rot_eq - expected).abs().max().item()))

    routing_diag = measure_routing_and_forgetting(system, base_host, eval_gap=l_gap)
    routing_diag["c4_equivariance_defect"] = c4_defect

    return {
        "arm": arm,
        "l_gap": l_gap,
        "seed": seed,
        "trainable_connector_params": sum(p.numel() for p in conn_params),
        "trainable_host_params": sum(p.numel() for p in host_params),
        "trainable_total_params": sum(p.numel() for p in conn_params) + sum(p.numel() for p in host_params),
        "train_seconds": round(train_s, 2),
        "final_train_loss": round(last_loss, 5),
        "q3_blind": q3_metrics,
        "q2_blind": q2_metrics,
        "q2_blackwell_gain": {
            "fresh_minus_valid_acc": float(q2_metrics["fresh_resample"]["acc"] - q2_metrics["valid"]["acc"]),
            "valid_minus_fresh_nll": float(q2_metrics["valid"]["nll"] - q2_metrics["fresh_resample"]["nll"]),
            "valid_plus_indep_minus_valid_acc": float(q2_metrics["valid_plus_independent"]["acc"] - q2_metrics["valid"]["acc"]),
            "dup_raw_minus_valid_acc": float(q2_metrics["duplicate_copy_raw"]["acc"] - q2_metrics["valid"]["acc"]),
        },
        "zero_shot_ood_gap": {
            "ood_gap": ood_gap,
            "G1": ood_g1,
            "G5_pi": ood_g5,
            "Q2_valid": ood_q2v,
            "Q2_fresh_resample": ood_q2f,
        },
        "routing": routing_diag,
    }


def aggregate_runs(runs: List[dict]) -> dict:
    summary: Dict[str, dict] = {}
    for lg in sorted({r["l_gap"] for r in runs}):
        key = f"L_gap_{lg}"
        summary[key] = {}
        arms_here = list(dict.fromkeys(r["arm"] for r in runs if r["l_gap"] == lg))
        for arm in arms_here:
            sub = [r for r in runs if r["l_gap"] == lg and r["arm"] == arm]
            q3_agg = {
                cid: {
                    "acc": float(np.mean([r["q3_blind"][cid]["acc"] for r in sub])),
                    "nll": float(np.mean([r["q3_blind"][cid]["nll"] for r in sub])),
                }
                for cid in sub[0]["q3_blind"]
            }
            q2_agg = {
                k: {
                    "acc": float(np.mean([r["q2_blind"][k]["acc"] for r in sub])),
                    "nll": float(np.mean([r["q2_blind"][k]["nll"] for r in sub])),
                }
                for k in sub[0]["q2_blind"]
            }
            ret_agg = {
                gk: {
                    "mean_head_retention": float(np.mean([r["routing"]["deltanet_retention_vs_gap"][gk]["mean_head_retention"] for r in sub])),
                    "best_head_retention": float(np.mean([r["routing"]["deltanet_retention_vs_gap"][gk]["best_head_retention"] for r in sub])),
                    "layer2_best_head_retention": float(np.mean([r["routing"]["deltanet_retention_vs_gap"][gk]["layer2_best_head_retention"] for r in sub])),
                }
                for gk in ("0", "16", "64", "128")
            }
            summary[key][arm] = {
                "trainable_total_params": sub[0]["trainable_total_params"],
                "trainable_host_params": sub[0]["trainable_host_params"],
                "q3_blind": q3_agg,
                "q2_blind": q2_agg,
                "q2_blackwell_gain": {
                    k: float(np.mean([r["q2_blackwell_gain"][k] for r in sub]))
                    for k in sub[0]["q2_blackwell_gain"]
                },
                "zero_shot_ood_gap": {
                    "ood_gap": sub[0]["zero_shot_ood_gap"]["ood_gap"],
                    "G1_acc": float(np.mean([r["zero_shot_ood_gap"]["G1"]["acc"] for r in sub])),
                    "G1_nll": float(np.mean([r["zero_shot_ood_gap"]["G1"]["nll"] for r in sub])),
                    "G5_pi_acc": float(np.mean([r["zero_shot_ood_gap"]["G5_pi"]["acc"] for r in sub])),
                    "G5_pi_nll": float(np.mean([r["zero_shot_ood_gap"]["G5_pi"]["nll"] for r in sub])),
                    "Q2_valid_acc": float(np.mean([r["zero_shot_ood_gap"]["Q2_valid"]["acc"] for r in sub])),
                    "Q2_fresh_acc": float(np.mean([r["zero_shot_ood_gap"]["Q2_fresh_resample"]["acc"] for r in sub])),
                    "Q2_fresh_nll": float(np.mean([r["zero_shot_ood_gap"]["Q2_fresh_resample"]["nll"] for r in sub])),
                },
                "sigma_min_D_answer": float(np.mean([r["routing"]["sigma_min_D_answer"] for r in sub])),
                "kappa_3": float(np.mean([r["routing"]["kappa_3"] for r in sub])),
                "full_attn_mod_mass": float(np.mean([r["routing"]["full_attn_mod_mass"] for r in sub])),
                "pure_kl": float(np.mean([r["routing"]["pure_kl"] for r in sub])),
                "pure_max_logit_diff": float(np.mean([r["routing"]["pure_max_logit_diff"] for r in sub])),
                "mix_kl": float(np.mean([r["routing"]["mix_kl"] for r in sub])),
                "c4_equivariance_defect": float(max(r["routing"]["c4_equivariance_defect"] for r in sub)),
                "deltanet_retention_vs_gap": ret_agg,
            }
    return summary


_WORKER_STATE: Dict[str, object] = {}


def _init_worker() -> None:
    torch.set_num_threads(1)
    encoder, _ = load_s17_encoder(ROOT)
    q2_kernel = json.loads(q2_reader.KERNEL_PATH.read_text(encoding="utf-8"))
    prior_s17 = np.asarray(q2_kernel["u_given_z1"], dtype=np.float64)
    _WORKER_STATE["prior_s17"] = prior_s17
    _WORKER_STATE["train_data"] = build_training_dataset(encoder, prior_s17, blocks_per_setting=6)
    _WORKER_STATE["blind_data"] = build_blind_benchmarks(encoder, prior_s17, q2_kernel)
    _WORKER_STATE["base_host"] = HybridDeltaNetAttnHost(seed=HOST_SEED)


def _run_worker_job(job: Tuple[str, int, int]) -> dict:
    arm, lg, seed = job
    return train_and_evaluate_arm(
        arm, lg, seed,
        _WORKER_STATE["prior_s17"],
        _WORKER_STATE["train_data"],
        _WORKER_STATE["blind_data"],
        _WORKER_STATE["base_host"],
    )


def main() -> None:
    t_start = time.perf_counter()
    torch.set_num_threads(1)
    n_workers = min(6, max(1, (os.cpu_count() or 4) // 2))

    print("[W1-R2] 1/5 Running exact FP64 algebraic & cross-track proofs (W1 x W2 x W3 x W4 x M3)...", flush=True)
    encoder, s17_sha = load_s17_encoder(ROOT)
    q2_kernel = json.loads(q2_reader.KERNEL_PATH.read_text(encoding="utf-8"))
    prior_s17 = np.asarray(q2_kernel["u_given_z1"], dtype=np.float64)

    proof_kalman = verify_kalman_rls_isomorphism()
    proof_m3 = verify_multiclock_and_m3_schur_gdn()
    proof_spectral = verify_spectral_cayley_schur_gdn()
    proof_wy = verify_parallel_wy_equivalence()
    proof_budget_voi = verify_surgical_budget_and_voi_bound(prior_s17)

    assert proof_kalman["kalman_max_abs_err"] < 1e-12
    assert proof_kalman["smw_matrix_rls_max_abs_err"] < 1e-12
    assert proof_m3["m3_correlated_schur_gdn_ll_max_err"] < 1e-13
    assert proof_m3["m3_ledger_duplicate_drift"] == 0.0
    assert proof_m3["m3_ledger_distractor_drift"] == 0.0
    assert proof_spectral["beta_01_2step_max_imag_eig"] == 0.0
    assert proof_spectral["householder_beta2_so2_max_abs_err"] < 1e-14
    assert proof_spectral["cayley_unitary_orth_err"] < 1e-14
    assert proof_spectral["cayley_so2_commutation_err"] < 1e-14
    assert proof_spectral["w1_w2_3key_schur_gdn_qtr_cov_err"] < 1e-14
    assert proof_wy["fp64_max_abs_err"] < 1e-13
    print(f"  -> All 14 exact FP64 identities passed (<1e-12) in {time.perf_counter()-t_start:.2f}s!", flush=True)

    print("[W1-R2] 2/5 Auditing W2 B_align = M_cand^+ F_real conditioning & cross-gap stability...", flush=True)
    b_align_audit = audit_b_align_conditioning(prior_s17)
    for k, v in b_align_audit.items():
        g64 = v["per_gap"]["64"]
        print(
            f"  {k:30s} | L=64: s1={g64['sigma_1']:.4f} s3={g64['sigma_3']:.4f} "
            f"k3={g64['kappa_3_s1_over_s3']:.2f} cur_amp={g64['taylor_curv_amp_1_over_s3_sq']:.1f} | "
            f"cos(V64,V128)={v['cross_gap_64_vs_128_min_cos']:.3f}/{v['cross_gap_64_vs_128_mean_cos']:.3f}",
            flush=True,
        )

    print("[W1-R2] 3/5 Building Stage 3D blind physical oracles (G1..G5(pi) + Q2 C4)...", flush=True)
    blind_data = build_blind_benchmarks(encoder, prior_s17, q2_kernel)

    schedule = [
        (16, ("prefix_only", "attn_lora", "deltanet_attn_lora_warmgate"), (MODEL_SEEDS[0],)),
        (64, (
            "prefix_only",
            "attn_lora",
            "deltanet_attn_lora",
            "deltanet_attn_lora_warmgate",
            "tier1_pos_gated_deltanet_early_frozen",
            "unified_w1_w2_w3_warmgate_schur",
        ), MODEL_SEEDS),
    ]
    jobs = [(arm, lg, seed) for lg, arms, seeds_here in schedule for arm in arms for seed in seeds_here]
    # Submit Longest-Job-First (L=64 heavy arms first, L=16 last) to eliminate tail stragglers; restore canonical order after!
    indexed_jobs = list(enumerate(jobs))
    ljf_order = sorted(indexed_jobs, key=lambda x: (-x[1][1], -("unified" in x[1][0] or "warmgate" in x[1][0] or "tier1" in x[1][0]), x[0]))
    print(f"[W1-R2] 4/5 Running {len(jobs)} matched arms across L_gap in {{16, 64}} on {n_workers} parallel CPU workers (LJF)...", flush=True)

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx, initializer=_init_worker) as pool:
        ljf_results = list(pool.map(_run_worker_job, [j for _, j in ljf_order]))
    runs_by_idx = {orig_idx: res for (orig_idx, _), res in zip(ljf_order, ljf_results)}
    runs = [runs_by_idx[i] for i in range(len(jobs))]

    for res in runs:
        lg, arm, seed = res["l_gap"], res["arm"], res["seed"]
        ood = res["zero_shot_ood_gap"]
        rt = res["routing"]
        ret_l2 = rt["deltanet_retention_vs_gap"][str(lg)]["layer2_best_head_retention"]
        print(
            f"  gap={lg:2d} | {arm:37s} | s={seed} | "
            f"G1={res['q3_blind']['G1_main_extrap_both']['acc']*100:5.2f}%/{res['q3_blind']['G1_main_extrap_both']['nll']:.3f} "
            f"G5={res['q3_blind']['G5_half_turn_pi']['acc']*100:5.2f}%/{res['q3_blind']['G5_half_turn_pi']['nll']:.3f} | "
            f"Q2(v->f)={res['q2_blind']['valid']['acc']*100:4.1f}->{res['q2_blind']['fresh_resample']['acc']*100:4.1f}% "
            f"vi={res['q2_blind']['valid_plus_independent']['acc']*100:4.1f}% dup={res['q2_blind']['duplicate_copy_raw']['acc']*100:4.1f}% | "
            f"OOD@{ood['ood_gap']}:G1={ood['G1']['acc']*100:4.1f}%/G5={ood['G5_pi']['acc']*100:4.1f}% | "
            f"s3={rt['sigma_min_D_answer']:.3f} retL2={ret_l2:.3f} pKL={rt['pure_kl']:.4f} mKL={rt['mix_kl']:.4f} C4={rt['c4_equivariance_defect']:.1e} | "
            f"{res['train_seconds']:.1f}s",
            flush=True,
        )

    summary = aggregate_runs(runs)
    total_s = time.perf_counter() - t_start

    output = {
        "meta": {
            "track": "W1_round2_unified",
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "s17_sha256": s17_sha,
            "host_config": {
                "d_model": D_MODEL, "n_heads": N_HEADS, "d_head": D_HEAD,
                "ffn_dim": FFN_DIM, "gdn_layers": 3, "full_attn_layers": 1,
                "lora_rank": LORA_RANK, "train_steps": TRAIN_STEPS, "batch_size": BATCH_SIZE,
                "seeds": list(MODEL_SEEDS), "nullspace_top_k": NULLSPACE_TOP_K,
            },
            "total_runtime_seconds": round(total_s, 2),
        },
        "math_proof_checks": {
            "kalman_rls_isomorphism": proof_kalman,
            "multiclock_and_m3_schur_gdn": proof_m3,
            "spectral_cayley_schur_gdn": proof_spectral,
            "parallel_wy_equivalence": proof_wy,
            "surgical_budget_and_voi_bound": proof_budget_voi,
        },
        "b_align_conditioning_audit": b_align_audit,
        "physical_oracles": {
            "q3_s17": {cid: cell["oracle"] for cid, cell in blind_data["q3"].items()},
            "q2_c4_work_a": blind_data["q2"]["oracle"],
        },
        "arm_summary": summary,
        "per_seed_runs": runs,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(output, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\n[W1-R2] 5/5 Saved verified Round 2 results to {OUT_JSON} in {total_s:.2f}s (target <= 150s).", flush=True)


if __name__ == "__main__":
    main()
