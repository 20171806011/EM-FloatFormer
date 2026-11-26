"""Track W2 (Rounds 1 & 2): Parameterization, Complex-Nullspace Cayley & Canonical Merging Verification Suite.

Self-contained CPU PyTorch verification for:
  Round 1:
    1. Theorem 1: C4 / C4xC4 symmetry-breaking excess DoF, Reynolds variance identity,
       DoRA SO(2)-norm equivariance defect, and SVFT text-SVD basis obstruction.
    2. Theorem 2 & 5: Physical AWGN/Brownian sector kernel Fourier spectrum,
       Stage 3D training-arc harmonic design matrix conditioning vs Slepian/polynomial blowup,
       exact real cal_ref Work-F rank-3 spectrum, and Minimal Rank Theorem (r >= 3).
    3. Theorem 3: Short-horizon (step 69 pilot & step 384) optimization dynamics for
       Zero-Init LoRA, LoRA+ (eta_B >> eta_A), PiSSA (aligned vs text-orthogonal gradient),
       and OLoRA.
    4. Theorem 4: Complex-Paired Cayley/Givens orthogonal adapter (exact Woodbury O(r^3+mr),
       exact ||Q||_2 = 1, cond(Q) = 1, exact SO(2) commutation, 2x param savings, and
       32-layer gradient-norm preservation matching Qwen3.5-4B depth).
    5. Part B & C: Controlled CPU Multi-Layer Surrogate Benchmark (10 arms + 5-level minimal-rank
       ablation, train |omega| <= 0.25 -> test real Stage 3D G1..G4 + G5 |theta|=pi, 384 steps + step-69 pilot).
  Round 2 (Cross-Track Integration with W1, W3, W4):
    6. Theorem 7 Certificate: U(m)-invariant complexified covariance C_gen^C = 0.5(C_gen + J^T C_gen J),
       real-PCA vs complex-symmetrized nullspace Cayley Q_C^perp (exact [Q_perp, J_d]=0, exact unitarity,
       exact identity on 2*k_c text principal subspace, exact subspace survival bound).
    7. Part D (W3 Integration): 4c_bayes_kl & 4d_nullspace_cayley_schur_bayes_kl comparing
       Bayes-KL + CF-Margin vs 1-Hot CE at Step-69 pilot (seed SD collapse), lambda_orbit=0 parity
       (bitwise C4 = 0.0e+00), PureKL (0.000 gated), and MixKL suppression on post-EM prose tokens.
    8. Part E (W4 Prop 4.1 / Theorem 8 Integration): Multi-Regime Adapter Merging (Regime 1 low-SNR
       high-jitter vs Regime 2 higher-SNR large-CFO -> unseen mixed capture f708_snr10 analogue),
       comparing Unconstrained LoRA vs W2 Canonical Cayley+Schur on GL(r) gauge error, Lie-algebra
       unitarity, Fisher cross-interference Gamma_12 / Gamma_21, cos_sim, and zero-retraining NLL.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import scipy
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from research.stage3_ministral_cache.schema import ROOTS, ROTATIONS, matched_sum, sector_of_complex
from research.stage3d_qwen_signed_phase.data_stage3d import (
    BLIND_CELLS,
    DEV_OMEGAS,
    NS_CFO_DEV,
    NS_CFO_TRAIN,
    TRAIN_OMEGAS,
    TRAIN_TIMINGS,
    generate_paired_cfo_blocks,
    load_s17_encoder,
    public_phase_slots,
)
from research.stage3d_qwen_signed_phase.recompute_all import (
    paired_t_ci,
    score_vectors,
    softmax_temperature,
)

OUT_DIR = Path(__file__).resolve().parent
STAGE3D_RESULTS = ROOT / "research/stage3d_qwen_signed_phase/results"

MODEL_SEEDS = (202611, 202622, 202633)
HOST_SEED = 20260402
D_MODEL = 64
M_COMPLEX = D_MODEL // 2
N_LAYERS = 4
TOTAL_STEPS = 384
PILOT_STEP = 69
BATCH_SIZE = 64
BASE_LR = 0.012
TRAIN_BLOCKS = 32
DEV_BLOCKS = 16
NS_G5_STRESS = 950499

ALL_CODES = torch.cartesian_prod(torch.arange(16), torch.arange(16))
INNER_CODES = torch.cartesian_prod(torch.arange(4), torch.arange(4))


# ---------------------------------------------------------------------------
# Exact physical kernel & scoring helpers (aligned with M1 / Stage 3D)
# ---------------------------------------------------------------------------

class SectorTransport(nn.Module):
    """Exact 64-point Gauss-Hermite AWGN + Brownian sector kernel (M1 Theorem 4)."""

    def __init__(self) -> None:
        super().__init__()
        x, w = np.polynomial.hermite.hermgauss(64)
        self.register_buffer("nodes", torch.from_numpy(x))
        self.register_buffer("weights", torch.from_numpy(w / math.sqrt(math.pi)))

    def forward(self, theta: torch.Tensor, variance: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
        angle = theta[..., None] + (2.0 * variance[..., None]).sqrt() * self.nodes
        scale = (2.0 * (10.0 ** (snr_db[..., None] / 10.0))).sqrt()
        c = torch.special.ndtr(scale * (angle + math.pi / 4.0).cos())
        s = torch.special.ndtr(scale * (angle + math.pi / 4.0).sin())
        base = torch.stack((c * s, (1.0 - c) * s, (1.0 - c) * (1.0 - s), c * (1.0 - s)), dim=-1)
        sector = (self.weights[:, None] * base).sum(dim=-2)
        return torch.stack([sector.roll(j, dims=-1) for j in range(4)], dim=-2)


def physical_kernel(transport: SectorTransport, theta: float, variance: float, snr_db: float = 2.0) -> np.ndarray:
    with torch.no_grad():
        return transport(
            torch.tensor(theta, dtype=torch.float64),
            torch.tensor(variance, dtype=torch.float64),
            torch.tensor(snr_db, dtype=torch.float64),
        ).numpy()


def physical_oracle_table(prior: np.ndarray, kernels: np.ndarray) -> np.ndarray:
    """Exact direct-s17 pushforward table of shape (n_signs, 256, 4)."""
    z1 = ALL_CODES[:, 0].numpy()
    canonical = np.einsum("zi,sik->szk", prior[z1 % 4], kernels)
    return np.take_along_axis(canonical, (np.arange(4) - z1[:, None] // 4)[None] % 4, axis=-1)


def lift_scores(canonical: torch.Tensor, codes: torch.Tensor, task: str = "q3_repeat") -> torch.Tensor:
    a, b = (codes // 4).unbind(dim=-1)
    if task == "q1":
        shift = b - a
    elif task in ("q2", "q3_repeat", "q3_after_pilot"):
        shift = a
    elif task == "q3_indep":
        return torch.full_like(canonical, 0.25)
    else:
        raise ValueError(f"Unknown task {task}")
    idx = (torch.arange(4, device=canonical.device) - shift[:, None]) % 4
    return canonical.gather(-1, idx)


def orbit_defect(probs_table: np.ndarray, shift_rule=lambda r, s: r) -> float:
    """Maximum L_inf equivariance defect over all 256 codes and all 16 C4xC4 actions."""
    q = np.asarray(probs_table, dtype=np.float64).reshape(-1, 16, 16, 4)
    return float(
        max(
            np.abs(
                np.roll(q, (-r * 4, -s * 4), axis=(1, 2))
                - np.roll(q, shift_rule(r, s), axis=-1)
            ).max()
            for r in range(4)
            for s in range(4)
        )
    )


def build_cell_hist(indices: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Exact (n_om, 256, 4) integer count histogram for O(256)-fast bit-exact scoring."""
    n_om = indices.shape[0]
    hist = np.zeros((n_om, 256, 4), dtype=np.int64)
    for k in range(n_om):
        np.add.at(hist[k], (indices[k].ravel(), targets[k].ravel()), 1)
    return hist


def score_table_from_hist(tbl: np.ndarray, hist: np.ndarray) -> dict:
    """Bit-exact equivalent of score_vectors on tbl[om, indices] vs targets, 500x faster."""
    p = np.asarray(tbl, dtype=np.float64).reshape(-1, 4)
    h = np.asarray(hist, dtype=np.float64).reshape(-1, 4)
    row_n = h.sum(axis=1)
    n_total = float(row_n.sum())
    max_p = np.max(p, axis=1, keepdims=True)
    ties = (p == max_p).astype(np.float64)
    tie_counts = ties.sum(axis=1)
    row_hits = (h * ties).sum(axis=1) / tie_counts
    acc = float(row_hits.sum() / n_total)

    p_clip = np.clip(p, 1e-12, 1.0)
    nll = float(-(h * np.log(p_clip)).sum() / n_total)

    p_sq_sum = (p ** 2).sum(axis=1)
    brier = float(((row_n * (p_sq_sum + 1.0)) - 2.0 * (h * p).sum(axis=1)).sum() / n_total)

    confs = max_p[:, 0]
    edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for i in range(10):
        mask = (confs >= edges[i]) & (confs <= edges[i + 1]) if i == 0 else (confs > edges[i]) & (confs <= edges[i + 1])
        cnt = float(row_n[mask].sum())
        if cnt > 0.0:
            bin_acc = float(row_hits[mask].sum() / cnt)
            bin_conf = float((row_n[mask] * confs[mask]).sum() / cnt)
            ece += (cnt / n_total) * abs(bin_acc - bin_conf)
    return {"acc": acc, "nll": nll, "brier": brier, "ece": float(ece)}


def gram_spectrum(matrix: np.ndarray) -> dict:
    m = np.asarray(matrix, dtype=np.float64)
    s = np.linalg.svd(m, compute_uv=False)
    e = s ** 2
    total = float(e.sum())
    frac = e[e > 0] / total if total > 0 else np.empty(0)
    thresh = max(m.shape) * np.finfo(np.float64).eps * (float(s.max()) if s.size else 1.0)
    return {
        "gram_eigenvalues": e.tolist(),
        "singular_values": s.tolist(),
        "rank_svd": int((s > thresh).sum()),
        "stable_rank": float(total / e[0]) if total > 0 and e[0] > 0 else 0.0,
        "effective_rank": float(np.exp(-np.dot(frac, np.log(frac)))) if frac.size else 0.0,
        "top1_energy_frac": float(e[0] / total) if total > 0 else 0.0,
        "top2_energy_frac": float(e[:2].sum() / total) if total > 0 else 0.0,
        "frobenius_norm": math.sqrt(total),
        "best_rank1_rel_residual": math.sqrt(max(0.0, 1.0 - float(e[0] / total))) if total > 0 else 0.0,
    }


# ---------------------------------------------------------------------------
# Part A: Exact Mathematical Theorem Certificates
# ---------------------------------------------------------------------------

def make_complex_structure(d: int) -> np.ndarray:
    """Block-diagonal J_d = I_{d/2} kron [[0, -1], [1, 0]]."""
    assert d % 2 == 0
    j2 = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
    return np.kron(np.eye(d // 2, dtype=np.float64), j2)


def realify_complex_matrix(z: np.ndarray) -> np.ndarray:
    """Isomorphism Phi: C^{m x m} -> R^{2m x 2m} with 2x2 commuting blocks [[u, -v], [v, u]]."""
    m = z.shape[0]
    out = np.empty((2 * m, 2 * m), dtype=np.float64)
    out[0::2, 0::2] = z.real
    out[0::2, 1::2] = -z.imag
    out[1::2, 0::2] = z.imag
    out[1::2, 1::2] = z.real
    return out


def complex_cayley_woodbury(u_c: torch.Tensor, v_c: torch.Tensor) -> torch.Tensor:
    """Compute exact unitary Q_C = (I - 0.5 S)(I + 0.5 S)^{-1} for S = U V* - V U* in O(r^3 + mr)."""
    # u_c, v_c: (m, r_c) complex
    m, r_c = u_c.shape
    a_c = torch.cat([u_c, -v_c], dim=1)  # (m, 2 r_c)
    b_c = torch.cat([v_c, u_c], dim=1)   # (m, 2 r_c)
    eye_2r = torch.eye(2 * r_c, dtype=u_c.dtype, device=u_c.device)
    core = eye_2r + 0.5 * (b_c.conj().T @ a_c)  # (2 r_c, 2 r_c)
    # Solve core @ X = B* => X = core^{-1} B*
    sol = torch.linalg.solve(core, b_c.conj().T)  # (2 r_c, m)
    eye_m = torch.eye(m, dtype=u_c.dtype, device=u_c.device)
    return eye_m - a_c @ sol


def verify_theorem1_symmetry_breaking() -> dict:
    """Theorem 1: Exact Reynolds identity, 3/4 & 15/16 non-equivariant leakage, DoRA & SVFT defects."""
    rng = np.random.default_rng(202601)
    d = 64

    # Regular representation of C4 (16 copies of 4x4 cyclic shift)
    c4_gen = np.roll(np.eye(4, dtype=np.float64), 1, axis=0)
    c4_reps = [np.kron(np.eye(d // 4), np.linalg.matrix_power(c4_gen, k)) for k in range(4)]

    # Regular representation of C4 x C4 (4 copies of 16x16 shift)
    c4x4_reps = []
    for a in range(4):
        ra = np.linalg.matrix_power(c4_gen, a)
        for b in range(4):
            rb = np.linalg.matrix_power(c4_gen, b)
            c4x4_reps.append(np.kron(np.eye(d // 16), np.kron(ra, rb)))

    def reynolds(mat: np.ndarray, reps: List[np.ndarray]) -> Tuple[np.ndarray, float, float]:
        proj = sum(g.T @ mat @ g for g in reps) / len(reps)
        perp = mat - proj
        lhs = sum(np.sum((mat @ g - g @ mat) ** 2) for g in reps) / len(reps)
        rhs = 2.0 * float(np.sum(perp ** 2))
        return proj, abs(lhs - rhs), float(np.sum(perp ** 2) / np.sum(mat ** 2))

    c4_fracs, c4x4_fracs = {}, {}
    max_identity_err = 0.0
    for r in (1, 2, 4, 8, 16):
        f4, f16 = [], []
        for _ in range(100):
            dw = rng.normal(size=(d, r)) @ rng.normal(size=(r, d))
            _, e4, frac4 = reynolds(dw, c4_reps)
            _, e16, frac16 = reynolds(dw, c4x4_reps)
            max_identity_err = max(max_identity_err, e4, e16)
            f4.append(frac4)
            f16.append(frac16)
        c4_fracs[f"r={r}"] = {"mean_leakage_frac": float(np.mean(f4)), "theory": 0.75}
        c4x4_fracs[f"r={r}"] = {"mean_leakage_frac": float(np.mean(f16)), "theory": 15.0 / 16.0}

    # Exact DoRA SO(2)-equivariance violation vs doublet-norm preservation
    j_d = make_complex_structure(d)
    phi = 0.73
    rot = math.cos(phi) * np.eye(d) + math.sin(phi) * j_d
    z_base = rng.normal(size=(d // 2, d // 2)) + 1j * rng.normal(size=(d // 2, d // 2))
    w_equiv = realify_complex_matrix(z_base)
    base_so2_defect = float(np.max(np.abs(w_equiv @ rot - rot @ w_equiv)))

    # Standard row-wise DoRA with unit magnitude m=1 and with heterogeneous m
    row_norms = np.linalg.norm(w_equiv, axis=1, keepdims=True)
    w_dora_unit = w_equiv / row_norms
    # Note: for realified complex matrix, pairs (2k, 2k+1) have identical row norms!
    # Wait: test BOTH (a) heterogeneous DoRA magnitude m, and (b) non-equivariant column/row norm after LoRA!
    m_het = 1.0 + 0.3 * rng.normal(size=(d, 1))
    w_dora_het = m_het * (w_equiv / row_norms)
    dora_het_so2_defect = float(np.max(np.abs(w_dora_het @ rot - rot @ w_dora_het)))

    # Standard DoRA on column-norm with non-commuting W0 (real LLM) + equivariant DeltaW:
    # Even when DeltaW is SO(2)-equivariant, col-norm(W0 + DeltaW) breaks equivariance AND
    # heterogeneous m breaks equivariance even on equivariant W!
    m_paired = np.repeat(1.0 + 0.3 * rng.normal(size=(d // 2, 1)), 2, axis=0)
    w_dora_paired = m_paired * (w_equiv / row_norms)
    dora_paired_so2_defect = float(np.max(np.abs(w_dora_paired @ rot - rot @ w_dora_paired)))

    # Also check: standard DoRA per-row norm when W is real-orthogonal-invariant vs realified:
    # For a realified matrix, ||row_{2k}||_2 == ||row_{2k+1}||_2, so unit-m preserves SO(2) ONLY on
    # already-complex-realified W; if W0 is non-complex, per-row norm differs between 2k and 2k+1!
    dora_unit_on_equiv_defect = float(np.max(np.abs(w_dora_unit @ rot - rot @ w_dora_unit)))

    # SVFT obstruction: text-pretrained random W0 = U Sigma V^T, project an exact SO(2) rotation J_d
    # onto banded SVFT subspace {U M V^T : |i - j| <= b}
    u0, _, vt0 = np.linalg.svd(rng.normal(size=(d, d)))
    target_in_svd = u0.T @ j_d @ vt0.T  # exact M^* = U^T J_d V
    svft_proj_errors = {}
    for bw in (0, 1, 2, 4, 8, 16):
        mask = np.abs(np.arange(d)[:, None] - np.arange(d)[None, :]) <= bw
        m_proj = target_in_svd * mask
        rel_err = float(np.linalg.norm(target_in_svd - m_proj) / np.linalg.norm(target_in_svd))
        nnz = int(mask.sum())
        svft_proj_errors[f"bw={bw}"] = {"nnz": nnz, "rel_frobenius_error": rel_err}

    assert max_identity_err < 1e-10
    assert base_so2_defect < 1e-14 and dora_paired_so2_defect < 1e-14
    assert dora_het_so2_defect > 0.1
    return {
        "reynolds_identity_max_abs_err": max_identity_err,
        "c4_random_lora_non_equivariant_energy": c4_fracs,
        "c4xc4_random_lora_non_equivariant_energy": c4x4_fracs,
        "dora_so2_check": {
            "base_complex_realified_defect": base_so2_defect,
            "dora_unit_m_on_complex_realified_defect": dora_unit_on_equiv_defect,
            "dora_heterogeneous_m_so2_defect": dora_het_so2_defect,
            "dora_doublet_tied_m_so2_defect": dora_paired_so2_defect,
        },
        "svft_text_basis_projection_of_J_d": svft_proj_errors,
    }


def verify_theorem2_and_5_harmonics_and_rank(transport: SectorTransport, encoder, archive) -> dict:
    """Theorems 2 & 5: Exact Fourier spectrum of K(theta, V), training-arc conditioning, and rank-3 necessity."""
    # 1. Exact Fourier coefficients b_hat_n of B_0(phi) at 2 dB
    n_grid = 256
    phi = np.linspace(0.0, 2.0 * np.pi, n_grid, endpoint=False)
    rho = 10.0 ** 0.2
    kappa = math.sqrt(2.0 * rho)
    from scipy.special import ndtr
    b0 = ndtr(kappa * np.cos(phi + math.pi / 4.0)) * ndtr(kappa * np.sin(phi + math.pi / 4.0))
    b_hat = np.fft.fft(b0) / n_grid  # b_hat[n] = (1/2pi) int B_0(phi) e^{-i n phi} dphi
    b_hat_real = {str(n): float(b_hat[n].real) for n in range(9)}
    b_hat_imag_max = float(np.max(np.abs(b_hat[:9].imag)))
    assert abs(b_hat[0].real - 0.25) < 1e-14
    assert abs(b_hat[4]) < 1e-14 and abs(b_hat[8]) < 1e-14

    # 2. Exact C4 DFT eigenvalues & rank-1/2/3 truncation errors of centered pushforward Pi K(theta, V)
    def truncated_kernel(theta: float, var: float, n_max: int) -> np.ndarray:
        row = np.zeros(4, dtype=np.float64)
        for m in range(4):
            val = b_hat[0].real
            for n in range(1, n_max + 1):
                val += 2.0 * b_hat[n].real * math.exp(-0.5 * (n ** 2) * var) * math.cos(n * (theta - m * math.pi / 2.0))
            row[m] = val
        return np.stack([np.roll(row, j) for j in range(4)], axis=0)

    pi_mat = np.eye(4) - 0.25 * np.ones((4, 4))
    kernel_truncation = {}
    for label, theta, var in (
        ("G1_theta1p60_V0p04", 1.60, 0.04),
        ("G2_theta0p80_V0p04", 0.80, 0.04),
        ("G3_theta1p60_V0p08", 1.60, 0.08),
        ("G4_theta0p80_V0p02", 0.80, 0.02),
        ("G5_thetaPi_V0p04", math.pi, 0.04),
    ):
        k_exact = physical_kernel(transport, theta, var, 2.0)
        k_cent = pi_mat @ k_exact
        sv = np.linalg.svd(k_cent, compute_uv=False)
        e = sv ** 2
        entry = {
            "centered_singular_values": sv.tolist(),
            "rank1_rel_frobenius_err": float(math.sqrt(e[1:].sum() / e.sum())),
            "rank2_rel_frobenius_err": float(math.sqrt(e[2:].sum() / e.sum())),
            "nyquist_mode_energy_frac": float(e[2] / e.sum()),
            "harmonic_truncation_max_abs_err": {},
        }
        for n_max in (0, 1, 2, 3, 5):
            k_tr = truncated_kernel(theta, var, n_max)
            entry["harmonic_truncation_max_abs_err"][f"n<={n_max}"] = float(np.max(np.abs(k_exact - k_tr)))
        kernel_truncation[label] = entry

    # Exact NLL impact of harmonic truncation n<=1 (rank 2!) vs n<=2 (rank 3!) on real G1..G4 blind data
    prior = encoder.table
    trunc_nll_g1 = {}
    g1_codes = archive["G1_main_extrap_both__codes"]
    g1_targets = archive["G1_main_extrap_both__targets"]
    for n_max in (0, 1, 2, 3):
        k_pair = np.stack([truncated_kernel(-1.60, 0.04, n_max), truncated_kernel(+1.60, 0.04, n_max)], axis=0)
        tbl = np.clip(physical_oracle_table(prior, k_pair), 1e-6, 1.0)
        tbl /= tbl.sum(axis=-1, keepdims=True)
        probs = tbl[np.arange(2)[:, None, None], g1_codes]
        sc = score_vectors(probs.reshape(-1, 4), g1_targets.reshape(-1))
        trunc_nll_g1[f"n<={n_max}"] = {"acc": sc["acc"], "nll": sc["nll"], "brier": sc["brier"]}

    # 3. Real cal_ref Work-F Gram spectrum
    cal = archive["G1_main_extrap_both__cal_ref"]
    mass = archive["G1_main_extrap_both__cal_mass"]
    overlap = np.minimum(mass[0], mass[1])
    sqrt_w = np.sqrt(overlap / overlap.sum())[:, None]
    work_f_phys = gram_spectrum(sqrt_w * (cal[1] - cal[0]))

    # 4. Stage 3D Training-Arc Design Matrix Conditioning & Extrapolation Amplification
    train_pts = [(w * (h + ell), 0.01 * (h + ell)) for h, ell, _ in TRAIN_TIMINGS for w in TRAIN_OMEGAS]
    thetas_tr = np.array([t for t, _ in train_pts], dtype=np.float64)
    vars_tr = np.array([v for _, v in train_pts], dtype=np.float64)
    thetas_test = np.array([1.60, -1.60, math.pi], dtype=np.float64)
    vars_test = np.array([0.04, 0.04, 0.04], dtype=np.float64)

    def build_harmonic_design(th: np.ndarray, vr: np.ndarray, n_max: int) -> np.ndarray:
        cols = [np.ones_like(th)]
        for n in range(1, n_max + 1):
            damp = np.exp(-0.5 * (n ** 2) * vr)
            cols.append(damp * np.cos(n * th))
            cols.append(damp * np.sin(n * th))
        return np.column_stack(cols)

    design_conditioning = {}
    # True target function: K_0(theta, V) - 0.25 at 2 dB
    y_tr = np.array([physical_kernel(transport, t, v, 2.0)[0, 0] - 0.25 for t, v in train_pts])
    y_te = np.array([physical_kernel(transport, t, v, 2.0)[0, 0] - 0.25 for t, v in zip(thetas_test, vars_test)])

    for n_max in (1, 2, 3, 4):
        phi_tr = build_harmonic_design(thetas_tr, vars_tr, n_max)
        phi_te = build_harmonic_design(thetas_test, vars_test, n_max)
        sv_d = np.linalg.svd(phi_tr, compute_uv=False)
        coef, *_ = np.linalg.lstsq(phi_tr, y_tr, rcond=None)
        pred_te = phi_te @ coef
        design_conditioning[f"semigroup_harmonics_n<={n_max}"] = {
            "num_basis": int(phi_tr.shape[1]),
            "rank": int(np.sum(sv_d > 1e-10)),
            "cond": float(sv_d[0] / sv_d[-1]) if sv_d[-1] > 1e-14 else float("inf"),
            "min_sv": float(sv_d[-1]),
            "g1_1p60_abs_err": float(abs(pred_te[0] - y_te[0])),
            "g5_pi_abs_err": float(abs(pred_te[2] - y_te[2])),
        }

    # Schur-Irrep-Locked conditioning: V_1 (n=1 only) & V_2 (n=2 only) decoupled by C4 irreps!
    phi_v1_tr = np.column_stack([np.exp(-0.5 * vars_tr) * np.cos(thetas_tr), np.exp(-0.5 * vars_tr) * np.sin(thetas_tr)])
    phi_v1_te = np.column_stack([np.exp(-0.5 * vars_test) * np.cos(thetas_test), np.exp(-0.5 * vars_test) * np.sin(thetas_test)])
    phi_v2_tr = np.column_stack([np.exp(-2.0 * vars_tr) * np.cos(2.0 * thetas_tr), np.exp(-2.0 * vars_tr) * np.sin(2.0 * thetas_tr)])
    phi_v2_te = np.column_stack([np.exp(-2.0 * vars_test) * np.cos(2.0 * thetas_test), np.exp(-2.0 * vars_test) * np.sin(2.0 * thetas_test)])
    # Project K_0(theta, V) onto V_1 (c1 = (K_0 - K_2)/2) and V_2 (c2 = (K_0 - K_1 + K_2 - K_3)/4)
    k_tr_all = np.stack([physical_kernel(transport, t, v, 2.0)[0] for t, v in train_pts], axis=0)
    k_te_all = np.stack([physical_kernel(transport, t, v, 2.0)[0] for t, v in zip(thetas_test, vars_test)], axis=0)
    v1_tr = 0.5 * (k_tr_all[:, 0] - k_tr_all[:, 2])
    v1_te = 0.5 * (k_te_all[:, 0] - k_te_all[:, 2])
    v2_tr = 0.25 * (k_tr_all[:, 0] - k_tr_all[:, 1] + k_tr_all[:, 2] - k_tr_all[:, 3])
    v2_te = 0.25 * (k_te_all[:, 0] - k_te_all[:, 1] + k_te_all[:, 2] - k_te_all[:, 3])
    sv_v1 = np.linalg.svd(phi_v1_tr, compute_uv=False)
    sv_v2 = np.linalg.svd(phi_v2_tr, compute_uv=False)
    c_v1, *_ = np.linalg.lstsq(phi_v1_tr, v1_tr, rcond=None)
    c_v2, *_ = np.linalg.lstsq(phi_v2_tr, v2_tr, rcond=None)
    pred_k0_te = (phi_v1_te @ c_v1) + (phi_v2_te @ c_v2)
    design_conditioning["schur_irrep_locked_V1_n1_plus_V2_n2"] = {
        "V1_n1_cond": float(sv_v1[0] / sv_v1[-1]),
        "V2_n2_cond": float(sv_v2[0] / sv_v2[-1]),
        "V1_g5_pi_abs_err": float(abs((phi_v1_te @ c_v1)[2] - v1_te[2])),
        "V2_g5_pi_abs_err": float(abs((phi_v2_te @ c_v2)[2] - v2_te[2])),
        "g1_1p60_abs_err": float(abs(pred_k0_te[0] - y_te[0])),
        "g5_pi_abs_err": float(abs(pred_k0_te[2] - y_te[2])),
    }

    # Compare polynomial degrees 1..5 in theta on the exact same 10 Stage 3D training points
    for deg in (1, 2, 3, 4, 5):
        poly_tr = np.column_stack([thetas_tr ** p for p in range(deg + 1)])
        poly_te = np.column_stack([thetas_test ** p for p in range(deg + 1)])
        sv_p = np.linalg.svd(poly_tr, compute_uv=False)
        coef, *_ = np.linalg.lstsq(poly_tr, y_tr, rcond=None)
        pred_te = poly_te @ coef
        design_conditioning[f"polynomial_deg={deg}"] = {
            "num_basis": int(poly_tr.shape[1]),
            "cond": float(sv_p[0] / sv_p[-1]),
            "g1_1p60_abs_err": float(abs(pred_te[0] - y_te[0])),
            "g5_pi_abs_err": float(abs(pred_te[2] - y_te[2])),
        }

    return {
        "b0_fourier_coeffs_2db_real": b_hat_real,
        "b0_fourier_imag_max_abs": b_hat_imag_max,
        "real_cal_ref_work_f_spectrum": work_f_phys,
        "centered_kernel_rank_and_harmonic_truncation": kernel_truncation,
        "g1_harmonic_truncation_scores": trunc_nll_g1,
        "stage3d_train_arc_conditioning_and_ood_error": design_conditioning,
    }


def verify_theorem3_short_horizon_dynamics() -> dict:
    """Theorem 3: Step-69 & Step-384 convergence of Zero-LoRA, LoRA+, PiSSA (aligned vs orthogonal), OLoRA."""
    d, r = 64, 4
    rng = np.random.default_rng(202603)
    # Anisotropic pretrained text weight W0 = U Sigma V^T
    u0, _ = np.linalg.qr(rng.normal(size=(d, d)))
    v0, _ = np.linalg.qr(rng.normal(size=(d, d)))
    sigma = np.array([2.0 * ((k + 1) ** -0.6) for k in range(d)], dtype=np.float64)
    w0 = u0 @ np.diag(sigma) @ v0.T

    # Two target task deltas: (A) text-aligned top-4 subspace, (B) EM text-orthogonal subspace (cols 4..7)
    delta_aligned = u0[:, :r] @ np.diag([0.5, 0.45, 0.4, 0.35]) @ v0[:, :r].T
    delta_ortho = u0[:, r:2 * r] @ np.diag([0.5, 0.45, 0.4, 0.35]) @ v0[:, r:2 * r].T

    def run_gd(target_delta: np.ndarray, mode: str, lr: float = 0.08, lora_plus_ratio: float = 8.0) -> dict:
        g = np.random.default_rng(777)
        scale = 1.0 / math.sqrt(r)
        if mode in ("zero_lora", "lora_plus"):
            a = g.normal(0.0, 1.0 / math.sqrt(d), size=(r, d))
            b = np.zeros((d, r), dtype=np.float64)
            w_res = w0.copy()
        elif mode == "pissa":
            b = u0[:, :r] * np.sqrt(sigma[:r])[None, :]
            a = np.sqrt(sigma[:r])[:, None] * v0[:, :r].T
            w_res = w0 - b @ a
            scale = 1.0
        elif mode == "olora":
            q_mat, r_mat = np.linalg.qr(w0)
            b = q_mat[:, :r].copy()
            a = r_mat[:r, :].copy()
            w_res = w0 - b @ a
            scale = 1.0
        else:
            raise ValueError(mode)

        w_target = w0 + target_delta
        err_init = float(np.linalg.norm((w_res + scale * (b @ a)) - w_target))
        err_69, err_384 = None, None
        text_drift_384 = None
        for step in range(1, TOTAL_STEPS + 1):
            w_cur = w_res + scale * (b @ a)
            diff = w_cur - w_target  # gradient of 0.5 ||W - W_target||_F^2
            grad_b = scale * (diff @ a.T)
            grad_a = scale * (b.T @ diff)
            if mode == "lora_plus":
                lr_a = lr / math.sqrt(lora_plus_ratio)
                lr_b = lr * math.sqrt(lora_plus_ratio)
            else:
                lr_a = lr_b = lr
            b = b - lr_b * grad_b
            a = a - lr_a * grad_a
            err = float(np.linalg.norm((w_res + scale * (b @ a)) - w_target))
            if step == PILOT_STEP:
                err_69 = err
            if step == TOTAL_STEPS:
                err_384 = err
                # Projection of learned delta onto top-r pretrained text subspace (forgetting proxy!)
                dw_learned = (w_res + scale * (b @ a)) - w0
                proj_top = u0[:, :r] @ (u0[:, :r].T @ dw_learned @ v0[:, :r]) @ v0[:, :r].T
                text_drift_384 = float(np.linalg.norm(proj_top))
        return {
            "rel_err_step69": float(err_69 / err_init),
            "rel_err_step384": float(err_384 / err_init),
            "top_text_subspace_drift_step384": text_drift_384,
        }

    out = {"text_aligned_target": {}, "em_text_orthogonal_target": {}}
    for mode in ("zero_lora", "lora_plus", "pissa", "olora"):
        out["text_aligned_target"][mode] = run_gd(delta_aligned, mode)
        out["em_text_orthogonal_target"][mode] = run_gd(delta_ortho, mode)
    return out


def verify_theorem4_complex_cayley_depth32() -> dict:
    """Theorem 4: Exact orthogonality, SO(2) equivariance, 2x param savings, and L=32 gradient preservation."""
    torch.manual_seed(202604)
    d, m, r_c = 64, 32, 4
    j_d = torch.from_numpy(make_complex_structure(d)).float()
    phi = 1.17
    rot = math.cos(phi) * torch.eye(d) + math.sin(phi) * j_d

    max_ortho_err = 0.0
    max_so2_err = 0.0
    max_cond_err = 0.0
    qs_real = []
    for _ in range(32):  # Match Qwen3.5-4B 32-layer depth!
        u_c = torch.randn(m, r_c, dtype=torch.complex64) * 0.4
        v_c = torch.randn(m, r_c, dtype=torch.complex64) * 0.4
        q_c = complex_cayley_woodbury(u_c, v_c)
        q_r = torch.empty(d, d, dtype=torch.float32)
        q_r[0::2, 0::2] = q_c.real
        q_r[0::2, 1::2] = -q_c.imag
        q_r[1::2, 0::2] = q_c.imag
        q_r[1::2, 1::2] = q_c.real
        qs_real.append(q_r)
        ortho_err = float((q_r.T @ q_r - torch.eye(d)).abs().max().item())
        so2_err = float((q_r @ rot - rot @ q_r).abs().max().item())
        sv = torch.linalg.svdvals(q_r)
        cond_err = float(abs((sv[0] / sv[-1]).item() - 1.0))
        max_ortho_err = max(max_ortho_err, ortho_err)
        max_so2_err = max(max_so2_err, so2_err)
        max_cond_err = max(max_cond_err, cond_err)

    # Depth-32 gradient norm ratio: Cayley chain vs unconstrained LoRA chain (I + 0.25 B A)
    x0 = torch.randn(16, d, requires_grad=True)
    h = x0
    for q_r in qs_real:
        h = h @ q_r.T
    g_out = torch.randn_like(h)
    loss = (h * g_out).sum()
    loss.backward()
    cayley_grad_ratio = float(x0.grad.norm().item() / g_out.norm().item())

    x1 = torch.randn(16, d, requires_grad=True)
    h1 = x1
    for _ in range(32):
        dw = (torch.randn(d, 2 * r_c) @ torch.randn(2 * r_c, d)) * (0.25 / math.sqrt(d))
        h1 = h1 @ (torch.eye(d) + dw).T
    loss1 = (h1 * g_out).sum()
    loss1.backward()
    lora_grad_ratio = float(x1.grad.norm().item() / g_out.norm().item())

    assert max_ortho_err < 1e-5 and max_so2_err < 1e-5 and abs(cayley_grad_ratio - 1.0) < 1e-4
    return {
        "depth_L": 32,
        "d_model": d,
        "complex_rank_r_c": r_c,
        "real_rank_equivalent": 2 * r_c,
        "params_complex_cayley_per_layer": 4 * m * r_c,
        "params_real_cayley_same_real_rank_per_layer": 4 * d * r_c,
        "parameter_saving_factor": 2.0,
        "max_orthogonality_defect": max_ortho_err,
        "max_so2_commutation_defect": max_so2_err,
        "max_condition_number_minus_1": max_cond_err,
        "depth32_cayley_grad_norm_ratio": cayley_grad_ratio,
        "depth32_unconstrained_lora_grad_norm_ratio": lora_grad_ratio,
    }


def compute_complex_nullspace_projector(
    c_gen: np.ndarray, k_c: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Theorem 7: Build U(m)-invariant complexified covariance C_gen^C = Phi(C_C) and rank-(m-k_c) projector.

    Returns: (pi_perp_c [m,m complex128], p_perp_real [2m,2m float64], eigvals_desc [m], u_top_c [m,k_c complex128]).
    """
    d = c_gen.shape[0]
    m = d // 2
    j_d = make_complex_structure(d)
    c_sym = 0.5 * (c_gen + j_d.T @ c_gen @ j_d)
    # Extract unique Hermitian C_C in C^{m x m}: real part = c_sym[0::2, 0::2], imag part = c_sym[1::2, 0::2]
    c_c = c_sym[0::2, 0::2] + 1j * c_sym[1::2, 0::2]
    c_c = 0.5 * (c_c + c_c.conj().T)  # enforce exact Hermitian symmetry
    evals, evecs = np.linalg.eigh(c_c)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    evecs = evecs[:, order]
    u_top = evecs[:, :k_c]  # (m, k_c)
    pi_perp_c = np.eye(m, dtype=np.complex128) - u_top @ u_top.conj().T
    p_perp_real = realify_complex_matrix(pi_perp_c)
    return pi_perp_c, p_perp_real, evals, u_top


def verify_theorem7_complex_nullspace_cayley() -> dict:
    """Theorem 7 Certificate: Real-PCA vs U(m)-Complexified Nullspace Cayley Q_C^perp."""
    rng = np.random.default_rng(202607)
    d, m, r_c = 64, 32, 4
    j_d = make_complex_structure(d)
    phi = 0.89
    rot = math.cos(phi) * np.eye(d) + math.sin(phi) * j_d

    # Construct generic non-complex-aligned text covariance: rank-12 strong principal subspace + 1e-4 tail
    k_real_strong = 12
    u_rand, _ = np.linalg.qr(rng.normal(size=(d, d)))
    eig_text = np.concatenate([np.linspace(2.0, 0.8, k_real_strong), np.full(d - k_real_strong, 1e-4)])
    c_gen = u_rand @ np.diag(eig_text) @ u_rand.T
    c_gen_comm_err = float(np.max(np.abs(c_gen @ j_d - j_d @ c_gen)))

    # Real PCA projector blocking top k_real = 24 real eigenvectors
    k_real = 24
    u_pca_top = u_rand[:, :k_real]
    p_perp_real_pca = np.eye(d) - u_pca_top @ u_pca_top.T
    real_pca_j_comm_err = float(np.max(np.abs(p_perp_real_pca @ j_d - j_d @ p_perp_real_pca)))

    # Theorem 7 Complex-Symmetrized Nullspace blocking k_c = 12 complex doublets (= 24 real dims)
    k_c = 12
    pi_perp_c, p_perp_complex, evals_c, u_top_c = compute_complex_nullspace_projector(c_gen, k_c)
    c_sym = 0.5 * (c_gen + j_d.T @ c_gen @ j_d)
    c_c = c_sym[0::2, 0::2] + 1j * c_sym[1::2, 0::2]
    phi_iso_err = float(np.max(np.abs(c_sym - realify_complex_matrix(c_c))))
    complex_proj_j_comm_err = float(np.max(np.abs(p_perp_complex @ j_d - j_d @ p_perp_complex)))
    # Exact trace identity: tr(P_perp_C^C C_gen) == 2 * sum_{j > k_c} mu_j
    trace_lhs = float(np.trace(p_perp_complex @ c_gen))
    trace_rhs = float(2.0 * np.sum(evals_c[k_c:]))
    trace_identity_err = abs(trace_lhs - trace_rhs)

    # Random skew-Hermitian generator S_C = U V* - V U*
    u_c = (rng.normal(size=(m, r_c)) + 1j * rng.normal(size=(m, r_c))) * 0.4
    v_c = (rng.normal(size=(m, r_c)) + 1j * rng.normal(size=(m, r_c))) * 0.4
    s_c = u_c @ v_c.conj().T - v_c @ u_c.conj().T
    s_real = realify_complex_matrix(s_c)

    # 1. Unprojected Complex-Cayley Q
    q_unproj = realify_complex_matrix(
        complex_cayley_woodbury(torch.from_numpy(u_c), torch.from_numpy(v_c)).numpy()
    )
    # 2. Naive Post-Cayley Real PCA: Q_post = I + (Q - I) P_perp_real
    q_post_pca = np.eye(d) + (q_unproj - np.eye(d)) @ p_perp_real_pca
    # 3. Naive Pre-Cayley Real PCA: Q_pre = Cay(P_perp_real S P_perp_real)
    s_pre_pca = p_perp_real_pca @ s_real @ p_perp_real_pca
    q_pre_pca = (np.eye(d) - 0.5 * s_pre_pca) @ np.linalg.inv(np.eye(d) + 0.5 * s_pre_pca)
    # 4. Theorem 7 Complex-Symmetrized Nullspace Cayley via Woodbury on (Pi_perp U_C, Pi_perp V_C)
    u_tilde = pi_perp_c @ u_c
    v_tilde = pi_perp_c @ v_c
    q_c_perp = complex_cayley_woodbury(torch.from_numpy(u_tilde), torch.from_numpy(v_tilde)).numpy()
    q_complex_null = realify_complex_matrix(q_c_perp)

    # Exact text principal doublet basis (real 2*k_c columns corresponding to u_top_c)
    basis_top = np.zeros((d, 2 * k_c), dtype=np.float64)
    for j in range(k_c):
        col = u_top_c[:, j]
        basis_top[0::2, 2 * j] = col.real
        basis_top[1::2, 2 * j] = col.imag
        basis_top[0::2, 2 * j + 1] = -col.imag
        basis_top[1::2, 2 * j + 1] = col.real

    def inspect_q(q_mat: np.ndarray) -> dict:
        sv = np.linalg.svd(q_mat, compute_uv=False)
        dq = q_mat - np.eye(d)
        text_energy = float(np.trace(dq @ c_gen @ dq.T))
        top_doublet_drift = float(np.max(np.abs(dq @ basis_top)))
        return {
            "ortho_defect": float(np.max(np.abs(q_mat.T @ q_mat - np.eye(d)))),
            "so2_comm_defect": float(np.max(np.abs(q_mat @ rot - rot @ q_mat))),
            "j_comm_defect": float(np.max(np.abs(q_mat @ j_d - j_d @ q_mat))),
            "cond": float(sv[0] / max(sv[-1], 1e-15)),
            "text_perturbation_energy_tr_dQ_C_dQT": text_energy,
            "top_2kc_text_doublet_max_drift": top_doublet_drift,
        }

    m_unproj = inspect_q(q_unproj)
    m_post = inspect_q(q_post_pca)
    m_pre = inspect_q(q_pre_pca)
    m_th7 = inspect_q(q_complex_null)

    # Verify exact subspace survival bound on a strictly rank-k=10 totally-real text subspace
    k_exact = 10
    u_tr = u_rand[:, :k_exact]
    c_exact_rank = u_tr @ u_tr.T
    _, p_exact, evals_exact, _ = compute_complex_nullspace_projector(c_exact_rank, k_exact)
    surviving_doublets = int(np.sum(evals_exact < 1e-12))
    assert surviving_doublets >= m - k_exact
    assert complex_proj_j_comm_err < 1e-14 and trace_identity_err < 1e-12 and phi_iso_err < 1e-14
    assert m_th7["ortho_defect"] < 1e-12 and m_th7["so2_comm_defect"] < 1e-12
    assert m_th7["top_2kc_text_doublet_max_drift"] < 1e-12
    assert m_post["ortho_defect"] > 0.05 and m_post["so2_comm_defect"] > 0.05
    assert m_pre["so2_comm_defect"] > 0.05

    return {
        "c_gen_non_commutation_with_J": c_gen_comm_err,
        "real_pca_projector_J_comm_defect": real_pca_j_comm_err,
        "complex_projector_J_comm_defect": complex_proj_j_comm_err,
        "complex_realification_iso_err": phi_iso_err,
        "trace_identity_abs_err": trace_identity_err,
        "k_c_blocked_doublets": k_c,
        "intact_surviving_doublets": m - k_c,
        "exact_rank10_surviving_doublets": surviving_doublets,
        "exact_rank10_lower_bound_m_minus_k": m - k_exact,
        "text_energy_suppression_ratio_vs_unproj": float(
            m_unproj["text_perturbation_energy_tr_dQ_C_dQT"] / max(m_th7["text_perturbation_energy_tr_dQ_C_dQT"], 1e-30)
        ),
        "comparison": {
            "1_unprojected_complex_cayley": m_unproj,
            "2_naive_post_cayley_real_pca": m_post,
            "3_naive_pre_cayley_real_pca": m_pre,
            "4_theorem7_complex_nullspace_cayley": m_th7,
        },
    }


# ---------------------------------------------------------------------------
# Part B: Multi-Layer Frozen-Base Surrogate + Direct Weight Modification Arms
# ---------------------------------------------------------------------------

def make_frozen_backbone_weights(seed: int) -> dict:
    """Create shared frozen pretrained-like anisotropic 4-layer residual weights & input/output maps."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        # Input dimension: prior(z1):4 + prior(z2):4 + onehot(z1):16 + onehot(z2):16 + phase_slots:6 + meta:4 = 50
        in_dim = 50
        q_in, r_in = torch.linalg.qr(torch.randn(D_MODEL, in_dim, dtype=torch.float64))
        e_proj = (q_in * torch.diagonal(r_in).sign()).float()  # (64, 50), orthonormal columns

        # Anisotropic pretrained base weights W_{l,0} (64x64) with power-law singular values
        w_bases = []
        sigma = torch.tensor([1.15 * ((k + 1) ** -0.55) for k in range(D_MODEL)], dtype=torch.float64)
        for _ in range(N_LAYERS):
            u, _ = torch.linalg.qr(torch.randn(D_MODEL, D_MODEL, dtype=torch.float64))
            v, _ = torch.linalg.qr(torch.randn(D_MODEL, D_MODEL, dtype=torch.float64))
            w_bases.append((u @ torch.diag(sigma) @ v.T).float())

        # Exact M1 §6.2 isotropic centered readout W_out = H_4 Q^T (4 x 64)
        frame, rf = torch.linalg.qr(torch.randn(D_MODEL, 3, dtype=torch.float64))
        frame = frame * torch.diagonal(rf).sign()
        helmert = torch.tensor(
            [[1, 1, 1], [-1, 1, 1], [0, -2, 1], [0, 0, -3]], dtype=torch.float64
        ) / torch.tensor([2, 6, 12], dtype=torch.float64).sqrt()
        w_out = (helmert @ frame.T).float()
    return {"e_proj": e_proj, "w_bases": w_bases, "w_out": w_out}


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class StaticLoRALayer(nn.Module):
    """Standard LoRA / rsLoRA / PiSSA layer: W_eff = W_res + scale * B @ A."""

    def __init__(self, w0: torch.Tensor, rank: int, init_mode: str = "zero", rs_scale: bool = False) -> None:
        super().__init__()
        d_out, d_in = w0.shape
        self.rank = rank
        self.init_mode = init_mode
        if init_mode == "pissa":
            u, s, vh = torch.linalg.svd(w0.double(), full_matrices=False)
            b0 = (u[:, :rank] * s[:rank].sqrt()[None, :]).float()
            a0 = (s[:rank].sqrt()[:, None] * vh[:rank, :]).float()
            self.register_buffer("w_res", (w0.double() - b0.double() @ a0.double()).float())
            self.scale = 1.0
            self.A = nn.Parameter(a0)
            self.B = nn.Parameter(b0)
        else:
            self.register_buffer("w_res", w0.clone())
            self.scale = (2.0 / math.sqrt(rank)) if rs_scale else (2.0 / float(rank))
            self.A = nn.Parameter(torch.empty(rank, d_in))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            self.B = nn.Parameter(torch.zeros(d_out, rank))

    def effective_weight(self) -> torch.Tensor:
        return self.w_res + self.scale * (self.B @ self.A)

    def forward(self, x: torch.Tensor, theta: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.w_res) + self.scale * F.linear(F.linear(x, self.A), self.B)


class DoRALayer(nn.Module):
    """Weight-Decomposed LoRA: W_eff = m * (W0 + s B A) / ||W0 + s B A||_row."""

    def __init__(self, w0: torch.Tensor, rank: int) -> None:
        super().__init__()
        d_out, d_in = w0.shape
        self.register_buffer("w0", w0.clone())
        self.scale = 2.0 / math.sqrt(rank)
        self.A = nn.Parameter(torch.empty(rank, d_in))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.zeros(d_out, rank))
        self.m = nn.Parameter(w0.norm(dim=1, keepdim=True).clone())

    def effective_weight(self) -> torch.Tensor:
        v = self.w0 + self.scale * (self.B @ self.A)
        return self.m * (v / v.norm(dim=1, keepdim=True).clamp_min(1e-8))

    def forward(self, x: torch.Tensor, theta: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight())


class SVFTLayer(nn.Module):
    """Singular-Vector Fine-Tuning: W_eff = W0 + U M V^T with sparse banded M (exact nnz)."""

    def __init__(self, w0: torch.Tensor, target_nnz: int = 768) -> None:
        super().__init__()
        d = w0.shape[0]
        u, _, vh = torch.linalg.svd(w0.double(), full_matrices=False)
        self.register_buffer("w0", w0.clone())
        self.register_buffer("U", u.float())
        self.register_buffer("Vh", vh.float())
        # Build deterministic banded index list of exact length target_nnz
        coords = []
        for offset in range(d):
            if offset == 0:
                for i in range(d):
                    coords.append((i, i))
            else:
                for i in range(d - offset):
                    coords.append((i, i + offset))
                    coords.append((i + offset, i))
            if len(coords) >= target_nnz:
                break
        coords = coords[:target_nnz]
        self.register_buffer("rows", torch.tensor([c[0] for c in coords], dtype=torch.long))
        self.register_buffer("cols", torch.tensor([c[1] for c in coords], dtype=torch.long))
        self.values = nn.Parameter(torch.zeros(target_nnz))

    def effective_weight(self) -> torch.Tensor:
        m = torch.zeros_like(self.w0)
        m[self.rows, self.cols] = self.values
        return self.w0 + self.U @ m @ self.Vh

    def forward(self, x: torch.Tensor, theta: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight())


class ComplexCayleyLayer(nn.Module):
    """Complex-Paired Cayley orthogonal adapter Q(S_C) W0 (+ optional commuting complex LoRA).

    Uses direct O(B*m*r + r^3) Woodbury application in forward() (exact equivalence to effective_weight()).
    """

    def __init__(self, w0: torch.Tensor, r_cayley: int, r_comm: int = 0) -> None:
        super().__init__()
        d = w0.shape[0]
        assert d % 2 == 0
        m = d // 2
        self.m = m
        self.r_cayley = r_cayley
        self.r_comm = r_comm
        self.register_buffer("w0", w0.clone())
        self.register_buffer("eye_2r", torch.eye(2 * r_cayley, dtype=torch.complex64))
        # Parameterize U_C, V_C as real/imag tensors (exact 4 * m * r_cayley real params)
        self.u_re = nn.Parameter(torch.zeros(m, r_cayley))
        self.u_im = nn.Parameter(torch.zeros(m, r_cayley))
        self.v_re = nn.Parameter(torch.randn(m, r_cayley) / math.sqrt(m))
        self.v_im = nn.Parameter(torch.randn(m, r_cayley) / math.sqrt(m))
        if r_comm > 0:
            # Complex-commuting low-rank residual Phi(B_C A_C): 4 * m * r_comm real params
            self.bc_re = nn.Parameter(torch.zeros(m, r_comm))
            self.bc_im = nn.Parameter(torch.zeros(m, r_comm))
            self.ac_re = nn.Parameter(torch.randn(r_comm, m) / math.sqrt(m))
            self.ac_im = nn.Parameter(torch.randn(r_comm, m) / math.sqrt(m))
            self.comm_scale = 2.0 / math.sqrt(r_comm)

    def _realify_torch(self, mat_c: torch.Tensor) -> torch.Tensor:
        m = self.m
        out = mat_c.real.new_empty((2 * m, 2 * m))
        out[0::2, 0::2] = mat_c.real
        out[0::2, 1::2] = -mat_c.imag
        out[1::2, 0::2] = mat_c.imag
        out[1::2, 1::2] = mat_c.real
        return out

    def skew_generator_complex(self) -> torch.Tensor:
        u_c = torch.complex(self.u_re, self.u_im)
        v_c = torch.complex(self.v_re, self.v_im)
        return u_c @ v_c.conj().T - v_c @ u_c.conj().T

    def cayley_matrix(self) -> torch.Tensor:
        u_c = torch.complex(self.u_re, self.u_im)
        v_c = torch.complex(self.v_re, self.v_im)
        q_c = complex_cayley_woodbury(u_c, v_c)
        return self._realify_torch(q_c)

    def effective_weight(self) -> torch.Tensor:
        q_r = self.cayley_matrix()
        w = q_r @ self.w0
        if self.r_comm > 0:
            b_c = torch.complex(self.bc_re, self.bc_im)
            a_c = torch.complex(self.ac_re, self.ac_im)
            w = w + self.comm_scale * self._realify_torch(b_c @ a_c)
        return w

    def forward(self, x: torch.Tensor, theta: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.w0)
        u_c = torch.complex(self.u_re, self.u_im)
        v_c = torch.complex(self.v_re, self.v_im)
        a_c = torch.cat([u_c, -v_c], dim=1)
        b_c = torch.cat([v_c, u_c], dim=1)
        core = self.eye_2r + 0.5 * (b_c.conj().T @ a_c)
        y_c = torch.complex(y[:, 0::2], y[:, 1::2])  # (B, m)
        sol = torch.linalg.solve(core, b_c.conj().T @ y_c.T)  # (2 r_c, B)
        q_y = y_c - (a_c @ sol).T  # (B, m)
        if self.r_comm > 0:
            x_c = torch.complex(x[:, 0::2], x[:, 1::2])
            a_cm = torch.complex(self.ac_re, self.ac_im)
            b_cm = torch.complex(self.bc_re, self.bc_im)
            q_y = q_y + self.comm_scale * ((x_c @ a_cm.T) @ b_cm.T)
        out = y.new_empty(y.shape)
        out[:, 0::2] = q_y.real
        out[:, 1::2] = q_y.imag
        return out


class NullspaceComplexCayleyLayer(nn.Module):
    """Theorem 7: Input-side Complex-Symmetrized Nullspace Cayley W_eff = W0 Q_C^perp + Phi(B_C A_C Pi_perp).

    Guarantees exact unitarity ||Q||_2 = 1, exact SO(2) commutation [Q, J_d] = 0, and exact identity
    W_eff x = W0 x on the entire 2*k_c-dimensional general-text principal subspace!
    """

    def __init__(self, w0: torch.Tensor, pi_perp_c: np.ndarray, r_cayley: int, r_comm: int = 0) -> None:
        super().__init__()
        d = w0.shape[0]
        m = d // 2
        self.m = m
        self.r_cayley = r_cayley
        self.r_comm = r_comm
        self.register_buffer("w0", w0.clone())
        self.register_buffer("eye_2r", torch.eye(2 * r_cayley, dtype=torch.complex64))
        self.register_buffer("pi_perp_re", torch.from_numpy(pi_perp_c.real).float())
        self.register_buffer("pi_perp_im", torch.from_numpy(pi_perp_c.imag).float())
        self.register_buffer("pi_perp_c_buf", torch.from_numpy(pi_perp_c.astype(np.complex64)))
        self.u_re = nn.Parameter(torch.zeros(m, r_cayley))
        self.u_im = nn.Parameter(torch.zeros(m, r_cayley))
        self.v_re = nn.Parameter(torch.randn(m, r_cayley) / math.sqrt(m))
        self.v_im = nn.Parameter(torch.randn(m, r_cayley) / math.sqrt(m))
        if r_comm > 0:
            self.bc_re = nn.Parameter(torch.zeros(m, r_comm))
            self.bc_im = nn.Parameter(torch.zeros(m, r_comm))
            self.ac_re = nn.Parameter(torch.randn(r_comm, m) / math.sqrt(m))
            self.ac_im = nn.Parameter(torch.randn(r_comm, m) / math.sqrt(m))
            self.comm_scale = 2.0 / math.sqrt(r_comm)

    def _pi_perp_c(self) -> torch.Tensor:
        return self.pi_perp_c_buf

    def _realify_torch(self, mat_c: torch.Tensor) -> torch.Tensor:
        m = self.m
        out = mat_c.real.new_empty((2 * m, 2 * m))
        out[0::2, 0::2] = mat_c.real
        out[0::2, 1::2] = -mat_c.imag
        out[1::2, 0::2] = mat_c.imag
        out[1::2, 1::2] = mat_c.real
        return out

    def skew_generator_complex(self) -> torch.Tensor:
        pi_c = self._pi_perp_c()
        u_t = pi_c @ torch.complex(self.u_re, self.u_im)
        v_t = pi_c @ torch.complex(self.v_re, self.v_im)
        return u_t @ v_t.conj().T - v_t @ u_t.conj().T

    def cayley_matrix(self) -> torch.Tensor:
        pi_c = self._pi_perp_c()
        u_t = pi_c @ torch.complex(self.u_re, self.u_im)
        v_t = pi_c @ torch.complex(self.v_re, self.v_im)
        return self._realify_torch(complex_cayley_woodbury(u_t, v_t))

    def effective_weight(self) -> torch.Tensor:
        q_r = self.cayley_matrix()
        w = self.w0 @ q_r
        if self.r_comm > 0:
            pi_c = self._pi_perp_c()
            b_c = torch.complex(self.bc_re, self.bc_im)
            a_c = torch.complex(self.ac_re, self.ac_im) @ pi_c
            w = w + self.comm_scale * self._realify_torch(b_c @ a_c)
        return w

    def forward(self, x: torch.Tensor, theta: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        pi_c = self._pi_perp_c()
        u_t = pi_c @ torch.complex(self.u_re, self.u_im)
        v_t = pi_c @ torch.complex(self.v_re, self.v_im)
        a_c = torch.cat([u_t, -v_t], dim=1)
        b_c = torch.cat([v_t, u_t], dim=1)
        core = self.eye_2r + 0.5 * (b_c.conj().T @ a_c)
        x_c = torch.complex(x[:, 0::2], x[:, 1::2])  # (B, m)
        sol = torch.linalg.solve(core, b_c.conj().T @ x_c.T)
        qx_c = x_c - (a_c @ sol).T
        qx = x.new_empty(x.shape)
        qx[:, 0::2] = qx_c.real
        qx[:, 1::2] = qx_c.imag
        out = F.linear(qx, self.w0)
        if self.r_comm > 0:
            a_cm = torch.complex(self.ac_re, self.ac_im)
            b_cm = torch.complex(self.bc_re, self.bc_im)
            xp_c = x_c @ pi_c.T  # (Pi_perp x)^T
            comm_c = self.comm_scale * ((xp_c @ a_cm.T) @ b_cm.T)
            comm = x.new_empty(x.shape)
            comm[:, 0::2] = comm_c.real
            comm[:, 1::2] = comm_c.imag
            out = out + comm
        return out


class DynamicHarmonicHyperLoRALayer(nn.Module):
    """Phase-Conditioned Dynamic Hyper-LoRA (unconstrained harmonic cores):
    DeltaW(theta, V) = B [M_0 + e^{-V/2}(cos(theta) M_{c1} + sin(theta) M_{s1})
                              + e^{-2V}(cos(2theta) M_{c2} + sin(2theta) M_{s2})] A.
    """

    def __init__(self, w0: torch.Tensor, rank: int, include_base: bool = True, max_harmonic: int = 2) -> None:
        super().__init__()
        d_out, d_in = w0.shape
        self.rank = rank
        self.include_base = include_base
        self.max_harmonic = max_harmonic
        self.register_buffer("w0", w0.clone())
        self.scale = 2.0 / math.sqrt(rank)
        self.A = nn.Parameter(torch.empty(rank, d_in))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.B = nn.Parameter(torch.empty(d_out, rank))
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        # Harmonic cores: n=0 (1 core) + n=1..max_harmonic (2 cores each), initialized at 0 => zero shock!
        n_cores = 1 + 2 * max_harmonic
        self.cores = nn.Parameter(torch.zeros(n_cores, rank, rank))

    def harmonic_basis(self, theta: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        cols = [torch.ones_like(theta)]
        for n in range(1, self.max_harmonic + 1):
            damp = torch.exp(-0.5 * (n ** 2) * var)
            cols.append(damp * torch.cos(n * theta))
            cols.append(damp * torch.sin(n * theta))
        return torch.stack(cols, dim=-1)  # (B, n_cores)

    def forward(self, x: torch.Tensor, theta: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        # x: (B, d_in), theta/var: (B,)
        h_a = F.linear(x, self.A)  # (B, r)
        phi = self.harmonic_basis(theta, var)  # (B, n_cores)
        m_dyn = torch.einsum("bk,kij->bij", phi, self.cores)  # (B, r, r)
        h_mid = torch.bmm(m_dyn, h_a.unsqueeze(-1)).squeeze(-1)  # (B, r)
        delta_out = self.scale * F.linear(h_mid, self.B)
        if self.include_base:
            return F.linear(x, self.w0) + delta_out
        return delta_out


class SchurIrrepHyperLoRALayer(nn.Module):
    """Schur-Irrep-Locked Dynamic Harmonic Hyper-LoRA (Theorem 5B):
    Couples n=1 semigroup e^{-V/2} R(theta) strictly into the 2D fundamental irrep V_1
    and n=2 semigroup e^{-2V} (cos 2theta, sin 2theta) strictly into the 1D Nyquist irrep V_2,
    mapped into hidden space via frozen host candidate-aligned frame B_align = W_out^+ F_real.
    Zero n=0 bias leakage; guarantees exact global quarter-turn phase covariance across [0, 2pi).
    """

    def __init__(
        self,
        w_out: torch.Tensor,
        d_in: int,
        k1_doublets: int,
        k2_nyquist: int,
        input_proj: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.k1 = int(k1_doublets)
        self.k2 = int(k2_nyquist)
        self.total_rank = 2 * self.k1 + self.k2
        assert self.total_rank >= 1
        if input_proj is not None:
            self.register_buffer("input_proj", torch.from_numpy(input_proj).float())
        else:
            self.input_proj = None

        # Exact C4 real DFT orthonormal basis of centered R^4 (cols 0,1 = V_1; col 2 = V_2)
        s2 = 1.0 / math.sqrt(2.0)
        f_real = torch.tensor(
            [[s2, 0.0, 0.5], [0.0, s2, -0.5], [-s2, 0.0, 0.5], [0.0, -s2, -0.5]],
            dtype=torch.float64,
        )
        b_align = (torch.linalg.pinv(w_out.double()) @ f_real).float()  # (d_model, 3)
        self.register_buffer("b_align", b_align)

        if self.k1 > 0:
            self.A1 = nn.Parameter(torch.empty(2 * self.k1, d_in))
            nn.init.kaiming_uniform_(self.A1, a=math.sqrt(5))
            # Complex-commuting scalar weight per doublet: c_re + i c_im -> [[c_re, -c_im], [c_im, c_re]]
            self.c1_re = nn.Parameter(torch.zeros(self.k1))
            self.c1_im = nn.Parameter(torch.zeros(self.k1))
        if self.k2 > 0:
            self.A2 = nn.Parameter(torch.empty(self.k2, d_in))
            nn.init.kaiming_uniform_(self.A2, a=math.sqrt(5))
            self.c2_cos = nn.Parameter(torch.zeros(self.k2))
            self.c2_sin = nn.Parameter(torch.zeros(self.k2))

    def canonical_effective_rows(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Exact gauge-free canonical rows (A1_eff [2, d], a2_cos [d], a2_sin [d]) for Theorem 8."""
        d_in = self.A1.shape[1] if self.k1 > 0 else self.A2.shape[1]
        device = self.b_align.device
        a1_eff = torch.zeros(2, d_in, device=device)
        if self.k1 > 0:
            a1_v = self.A1.view(self.k1, 2, d_in)
            a1_eff[0] = (self.c1_re[:, None] * a1_v[:, 0] - self.c1_im[:, None] * a1_v[:, 1]).sum(dim=0)
            a1_eff[1] = (self.c1_im[:, None] * a1_v[:, 0] + self.c1_re[:, None] * a1_v[:, 1]).sum(dim=0)
        a2_cos = torch.zeros(d_in, device=device)
        a2_sin = torch.zeros(d_in, device=device)
        if self.k2 > 0:
            a2_cos = self.c2_cos @ self.A2
            a2_sin = self.c2_sin @ self.A2
        if self.input_proj is not None:
            a1_eff = a1_eff @ self.input_proj
            a2_cos = a2_cos @ self.input_proj
            a2_sin = a2_sin @ self.input_proj
        return a1_eff, a2_cos, a2_sin

    def forward(self, x: torch.Tensor, theta: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        if self.input_proj is not None:
            x = F.linear(x, self.input_proj)
        bsz = x.shape[0]
        coords = x.new_zeros((bsz, 3))
        if self.k1 > 0:
            h1 = F.linear(x, self.A1).view(bsz, self.k1, 2)  # (B, k1, 2)
            # Apply learned complex-commuting scaling per doublet, sum over k1, then rotate by e^{-V/2} R(theta)
            u_re = (h1[..., 0] * self.c1_re - h1[..., 1] * self.c1_im).sum(dim=1)
            u_im = (h1[..., 0] * self.c1_im + h1[..., 1] * self.c1_re).sum(dim=1)
            d1 = torch.exp(-0.5 * var)
            c1 = d1 * torch.cos(theta)
            s1 = d1 * torch.sin(theta)
            coords[:, 0] = c1 * u_re - s1 * u_im
            coords[:, 1] = s1 * u_re + c1 * u_im
        if self.k2 > 0:
            h2 = F.linear(x, self.A2)  # (B, k2)
            d2 = torch.exp(-2.0 * var)
            c2 = d2 * torch.cos(2.0 * theta)
            s2 = d2 * torch.sin(2.0 * theta)
            coords[:, 2] = c2 * (h2 @ self.c2_cos) + s2 * (h2 @ self.c2_sin)
        return F.linear(coords, self.b_align)  # (B, d_model)


class SurrogateAdapterNet(nn.Module):
    """4-layer frozen-base residual network testing direct weight modification parameterizations."""

    def __init__(
        self,
        backbone: dict,
        prior: np.ndarray,
        arm_type: str,
        *,
        quotient: bool,
        dyn_rank: int = 5,
        max_harmonic: int = 2,
        k1_doublets: int = 4,
        k2_nyquist: int = 3,
        nullspace_projs: dict | None = None,
    ) -> None:
        super().__init__()
        self.arm_type = arm_type
        self.quotient = quotient
        self.dyn_rank = dyn_rank
        self.register_buffer("prior", torch.as_tensor(prior, dtype=torch.float32))
        self.register_buffer("e_proj", backbone["e_proj"].clone())
        self.register_buffer("w_out", backbone["w_out"].clone())
        self.w_bases_frozen = nn.ParameterList([nn.Parameter(w.clone(), requires_grad=False) for w in backbone["w_bases"]])
        # Rank-3 nullspace projector P_null = I - W_out^+ W_out (zero-param frozen geometry)
        p_null = (torch.eye(D_MODEL, dtype=torch.float64) - torch.linalg.pinv(backbone["w_out"].double()) @ backbone["w_out"].double()).float()
        self.register_buffer("p_null", p_null)

        w_bases = backbone["w_bases"]
        layers: List[nn.Module] = []
        self.linear_final_dynamic = False
        self.schur_zero_dc_skip = False
        self.mask_input_phase = False

        if arm_type in ("lora_zero", "lora_plus"):
            for l_idx in range(N_LAYERS):
                layers.append(StaticLoRALayer(w_bases[l_idx], rank=6, init_mode="zero", rs_scale=(arm_type == "lora_plus")))
        elif arm_type == "pissa":
            for l_idx in range(N_LAYERS):
                layers.append(StaticLoRALayer(w_bases[l_idx], rank=6, init_mode="pissa"))
        elif arm_type == "dora":
            for l_idx, rk in enumerate((5, 6, 5, 6)):
                layers.append(DoRALayer(w_bases[l_idx], rank=rk))
        elif arm_type == "svft":
            for l_idx in range(N_LAYERS):
                layers.append(SVFTLayer(w_bases[l_idx], target_nnz=768))
        elif arm_type == "quotient_cayley_pure":
            for l_idx in range(N_LAYERS):
                layers.append(ComplexCayleyLayer(w_bases[l_idx], r_cayley=6, r_comm=0))
        elif arm_type == "quotient_cayley_hybrid":
            for l_idx in range(N_LAYERS):
                layers.append(ComplexCayleyLayer(w_bases[l_idx], r_cayley=3, r_comm=3))
        elif arm_type == "quotient_dynamic_all4":
            for l_idx in range(N_LAYERS):
                layers.append(DynamicHarmonicHyperLoRALayer(w_bases[l_idx], rank=dyn_rank, include_base=True, max_harmonic=max_harmonic))
        elif arm_type == "quotient_cayley_plus_harmonic":
            # Unconstrained harmonic final layer (tests n=0/2 -> V_1 collinearity leakage at theta=pi)
            self.linear_final_dynamic = True
            self.mask_input_phase = True
            for l_idx in range(N_LAYERS - 1):
                layers.append(ComplexCayleyLayer(w_bases[l_idx], r_cayley=3, r_comm=3))
            layers.append(
                DynamicHarmonicHyperLoRALayer(
                    w_bases[N_LAYERS - 1], rank=dyn_rank, include_base=False, max_harmonic=max_harmonic
                )
            )
        elif arm_type == "quotient_cayley_schur_hyper_lora":
            # Unified Synthesis (Theorem 4 + 5B + 6):
            # Layers 0..2: Complex-Cayley+Commuting on canonical evidence (no phase leak)
            # Layer 3: Schur-Irrep-Locked Dynamic Harmonic Hyper-LoRA with zero-DC readout skip
            self.linear_final_dynamic = True
            self.schur_zero_dc_skip = True
            self.mask_input_phase = True
            for l_idx in range(N_LAYERS - 1):
                layers.append(ComplexCayleyLayer(w_bases[l_idx], r_cayley=3, r_comm=3))
            layers.append(SchurIrrepHyperLoRALayer(backbone["w_out"], D_MODEL, k1_doublets, k2_nyquist))
        elif arm_type == "quotient_nullspace_cayley_schur":
            # Round 2 Theorem 7 Synthesis (4d): Complex-Symmetrized Nullspace Cayley + Nullspace Schur-Irrep
            assert nullspace_projs is not None
            self.linear_final_dynamic = True
            self.schur_zero_dc_skip = True
            self.mask_input_phase = True
            for l_idx in range(N_LAYERS - 1):
                layers.append(
                    NullspaceComplexCayleyLayer(
                        w_bases[l_idx], nullspace_projs["pi_perp_c"][l_idx], r_cayley=3, r_comm=3
                    )
                )
            layers.append(
                SchurIrrepHyperLoRALayer(
                    backbone["w_out"], D_MODEL, k1_doublets, k2_nyquist,
                    input_proj=nullspace_projs["p_perp_real"][N_LAYERS - 1],
                )
            )
        else:
            raise ValueError(f"Unknown arm_type={arm_type}")

        self.layers = nn.ModuleList(layers)
        # Precompute exact canonical 16-code zero-phase input embedding for fast exact quotient dedup
        with torch.no_grad():
            p1_16 = self.prior[INNER_CODES[:, 0]]
            p2_16 = self.prior[INNER_CODES[:, 1]]
            oh1_16 = F.one_hot(INNER_CODES[:, 0], num_classes=16).float()
            oh2_16 = F.one_hot(INNER_CODES[:, 1], num_classes=16).float()
            raw_16 = torch.cat((p1_16, p2_16, oh1_16, oh2_16, torch.zeros(16, 10)), dim=-1)
            self.register_buffer("inner_enc_16", F.linear(raw_16, self.e_proj))
            self.register_buffer("dummy_16", torch.zeros(16))

    def _layers012_canonical_16(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact 16-row Layers 0..2 pass when quotient & mask_input_phase (0.0 diff, 4x faster)."""
        h16 = self.inner_enc_16
        for l_idx in range(N_LAYERS - 1):
            h16 = h16 + 0.5 * F.gelu(self.layers[l_idx](rms_norm(h16), self.dummy_16, self.dummy_16))
        return h16, rms_norm(h16)

    def _apply_final_dynamic(self, h: torch.Tensor, hn: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        h_val, ell_val, _, omega_val, sigma_val, _ = meta[:, 6:12].unbind(dim=1)
        dt = h_val + ell_val
        theta = omega_val * dt
        var = (sigma_val ** 2) * dt
        dyn_vec = self.layers[N_LAYERS - 1](hn, theta, var)
        if self.schur_zero_dc_skip:
            h_out = F.linear(h, self.p_null) + dyn_vec
        else:
            h_out = h + dyn_vec
        return F.linear(h_out, self.w_out)

    def encode_features(self, codes: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        ids = (codes % 4) if self.quotient else codes
        p1 = self.prior[ids[:, 0]]
        p2 = self.prior[ids[:, 1]]
        oh1 = F.one_hot(ids[:, 0], num_classes=16).float()
        oh2 = F.one_hot(ids[:, 1], num_classes=16).float()
        if self.mask_input_phase:
            phase_part = torch.zeros(codes.shape[0], 10, device=codes.device, dtype=torch.float32)
        else:
            slots = meta[:, :6]
            h, ell, _, omega, sigma, _ = meta[:, 6:12].unbind(dim=1)
            dt = h + ell
            scalars = torch.stack((h / 4.0, ell / 4.0, omega * 2.0, sigma * dt.sqrt()), dim=-1)
            phase_part = torch.cat((slots, scalars), dim=-1)
        raw = torch.cat((p1, p2, oh1, oh2, phase_part), dim=-1)  # (B, 50)
        return F.linear(raw, self.e_proj)

    def canonical_logits(
        self, codes: torch.Tensor, meta: torch.Tensor, h_pre: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.linear_final_dynamic and self.quotient and self.mask_input_phase:
            h16, hn16 = self._layers012_canonical_16()
            inner_idx = (codes[:, 0] % 4) * 4 + (codes[:, 1] % 4)
            return self._apply_final_dynamic(h16[inner_idx], hn16[inner_idx], meta)

        h = h_pre if h_pre is not None else self.encode_features(codes, meta)
        h_val, ell_val, _, omega_val, sigma_val, _ = meta[:, 6:12].unbind(dim=1)
        dt = h_val + ell_val
        theta = omega_val * dt
        var = (sigma_val ** 2) * dt

        if self.linear_final_dynamic:
            for l_idx in range(N_LAYERS - 1):
                h = h + 0.5 * F.gelu(self.layers[l_idx](rms_norm(h), theta, var))
            dyn_vec = self.layers[N_LAYERS - 1](rms_norm(h), theta, var)
            if self.schur_zero_dc_skip:
                # Zero-parameter projection onto ker(W_out) eliminates static n=0 skip bias
                h = F.linear(h, self.p_null) + dyn_vec
            else:
                h = h + dyn_vec
            return F.linear(h, self.w_out)

        for layer in self.layers:
            h = h + 0.5 * F.gelu(layer(rms_norm(h), theta, var))
        return F.linear(h, self.w_out)

    def forward_logits(
        self, codes: torch.Tensor, meta: torch.Tensor, h_pre: torch.Tensor | None = None
    ) -> torch.Tensor:
        logits = self.canonical_logits(codes, meta, h_pre=h_pre)
        if not self.quotient:
            return logits
        return lift_scores(logits, codes, "q3_repeat")

    def forward_logits_pair(
        self, codes: torch.Tensor, meta: torch.Tensor, meta_cf: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact paired factual + counterfactual logits reusing shared Layers 0..2 when phase-masked."""
        if self.linear_final_dynamic and self.quotient and self.mask_input_phase:
            h16, hn16 = self._layers012_canonical_16()
            inner_idx = (codes[:, 0] % 4) * 4 + (codes[:, 1] % 4)
            h_b, hn_b = h16[inner_idx], hn16[inner_idx]
            l_f = self._apply_final_dynamic(h_b, hn_b, meta)
            l_cf = self._apply_final_dynamic(h_b, hn_b, meta_cf)
            return lift_scores(l_f, codes, "q3_repeat"), lift_scores(l_cf, codes, "q3_repeat")
        both_c = torch.cat((codes, codes), dim=0)
        both_m = torch.cat((meta, meta_cf), dim=0)
        both_l = self.forward_logits(both_c, both_m)
        n = codes.shape[0]
        return both_l[:n], both_l[n:]

    def forward_probs(
        self, codes: torch.Tensor, meta: torch.Tensor, h_pre: torch.Tensor | None = None
    ) -> torch.Tensor:
        probs = self.canonical_logits(codes, meta, h_pre=h_pre).softmax(dim=-1)
        if not self.quotient:
            return probs
        return lift_scores(probs, codes, "q3_repeat")

    def forward_text_logits(
        self, x_hid: torch.Tensor, w_text: torch.Tensor, g_em: bool, theta_val: float = 0.60, var_val: float = 0.04
    ) -> torch.Tensor:
        """Evaluate general-text readout with exact binary modality gate g_EM in {0, 1} (W3 Theorem 3/4)."""
        h = x_hid
        if not g_em:
            # Bit-exact frozen base bypass: zero adapter ops, PureKL = 0.000e+00 identically
            for l_idx in range(N_LAYERS):
                h = h + 0.5 * F.gelu(F.linear(rms_norm(h), self.w_bases_frozen[l_idx]))
            return F.linear(h, w_text)
        theta = x_hid.new_full((x_hid.shape[0],), theta_val)
        var = x_hid.new_full((x_hid.shape[0],), var_val)
        if self.linear_final_dynamic:
            for l_idx in range(N_LAYERS - 1):
                h = h + 0.5 * F.gelu(self.layers[l_idx](rms_norm(h), theta, var))
            # On post-EM prose tokens, base layer 3 still runs alongside active Schur readout perturbation
            h = h + 0.5 * F.gelu(F.linear(rms_norm(h), self.w_bases_frozen[N_LAYERS - 1])) + self.layers[N_LAYERS - 1](rms_norm(h), theta, var)
            return F.linear(h, w_text)
        for layer in self.layers:
            h = h + 0.5 * F.gelu(layer(rms_norm(h), theta, var))
        return F.linear(h, w_text)

    def orbit_table_calibrated(self, meta_1: torch.Tensor, temp: float = 1.0) -> np.ndarray:
        """Normalize ONCE in canonical order before integer gather -> bitwise 0.000e+00 C4 defect."""
        if not self.quotient:
            logits = self.canonical_logits(ALL_CODES, meta_1.expand(256, -1)).numpy().astype(np.float64)
            return softmax_temperature(logits, temp)
        canon_logits = self.canonical_logits(INNER_CODES, meta_1.expand(16, -1)).numpy().astype(np.float64)
        canon_probs = softmax_temperature(canon_logits, temp)  # (16, 4) normalized ONCE
        z1 = ALL_CODES[:, 0].numpy()
        z2 = ALL_CODES[:, 1].numpy()
        inner_idx = (z1 % 4) * 4 + (z2 % 4)
        shift = z1 // 4
        gathered = canon_probs[inner_idx]  # (256, 4)
        col_idx = (np.arange(4)[None, :] - shift[:, None]) % 4
        return np.take_along_axis(gathered, col_idx, axis=-1)

    def count_trainable(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def diagnostics(self) -> dict:
        conds = []
        cayley_ortho_max = 0.0
        cayley_so2_max = 0.0
        j_d = torch.from_numpy(make_complex_structure(D_MODEL)).float()
        rot = math.cos(0.9) * torch.eye(D_MODEL) + math.sin(0.9) * j_d
        with torch.no_grad():
            for layer in self.layers:
                if hasattr(layer, "effective_weight"):
                    sv = torch.linalg.svdvals(layer.effective_weight())
                    conds.append(float((sv[0] / sv[-1].clamp_min(1e-12)).item()))
                if isinstance(layer, (ComplexCayleyLayer, NullspaceComplexCayleyLayer)):
                    q_r = layer.cayley_matrix()
                    cayley_ortho_max = max(cayley_ortho_max, float((q_r.T @ q_r - torch.eye(D_MODEL)).abs().max().item()))
                    cayley_so2_max = max(cayley_so2_max, float((q_r @ rot - rot @ q_r).abs().max().item()))
        has_cayley = any(isinstance(l, (ComplexCayleyLayer, NullspaceComplexCayleyLayer)) for l in self.layers)
        return {
            "mean_effective_weight_cond": float(np.mean(conds)) if conds else None,
            "max_effective_weight_cond": float(np.max(conds)) if conds else None,
            "cayley_max_ortho_defect": cayley_ortho_max if has_cayley else None,
            "cayley_max_so2_defect": cayley_so2_max if has_cayley else None,
        }


# ---------------------------------------------------------------------------
# Data Loading, Training (Step 69 Pilot + Step 384 Final), and Evaluation
# ---------------------------------------------------------------------------

def metadata_row(cond) -> torch.Tensor:
    slots = public_phase_slots(cond.omega, cond.h, cond.ell, cond.t_p)
    scalars = (cond.h, cond.ell, cond.t_p, cond.omega, cond.sigma_phi, cond.snr_db)
    return torch.tensor([*slots, *scalars], dtype=torch.float32)


def encode_view(encoder, mother, r1: int, r2: int) -> np.ndarray:
    y1, y2 = mother.rotated_view(r1, r2)
    return np.column_stack((encoder.encode(y1), encoder.encode(y2)))


def draw_split(encoder, namespace: int, blocks: int, omegas: Tuple[float, ...], split: str):
    c_list, m_list, y_list = [], [], []
    seeds = []
    for i, (h, ell, tp) in enumerate(TRAIN_TIMINGS):
        paired, b_seeds = generate_paired_cfo_blocks(namespace, i, blocks, h, ell, tp, omegas, split_prefix=split)
        seeds.extend(b_seeds)
        for omega in omegas:
            mother = paired[omega]
            n = len(mother.x1)
            c_list.append(torch.from_numpy(encode_view(encoder, mother, 0, 0)))
            m_list.append(metadata_row(mother.condition).expand(n, -1))
            y_list.append(torch.from_numpy(mother.k3_repeat))
    return (torch.cat(c_list), torch.cat(m_list), torch.cat(y_list)), seeds


def build_blind_cells(encoder, transport: SectorTransport, archive, used_seeds: set) -> dict:
    cells = {}
    for cid, cell in BLIND_CELLS.items():
        omegas = (-cell.abs_omega, cell.abs_omega)
        paired, seeds = generate_paired_cfo_blocks(
            cell.namespace, 0, cell.default_blocks, cell.h, cell.ell, cell.t_p,
            omegas, cell.snr_db, cell.sigma_phi, split_prefix=cid,
        )
        assert set(seeds).isdisjoint(used_seeds)
        raw_codes, labels, metas, kernels = [], [], [], []
        for omega in omegas:
            mother = paired[omega]
            z = encode_view(encoder, mother, 0, 0)
            y_rep = np.stack([sector_of_complex(matched_sum(mother.y3_repeat * ROOTS[r])) for r, _ in ROTATIONS])
            raw_codes.append(np.stack([(z + [4 * r, 4 * s]) % 16 for r, s in ROTATIONS]))
            labels.append(y_rep)
            metas.append(metadata_row(mother.condition))
            kernels.append(physical_kernel(transport, omega * (cell.h + cell.ell), (cell.sigma_phi ** 2) * (cell.h + cell.ell), cell.snr_db))
        code_pairs = np.stack(raw_codes)
        indices = code_pairs @ [16, 1]
        targets = np.stack(labels)
        np.testing.assert_array_equal(indices, archive[f"{cid}__codes"])
        np.testing.assert_array_equal(targets, archive[f"{cid}__targets"])
        kernels_np = np.stack(kernels)
        oracle_tbl = physical_oracle_table(encoder.table, kernels_np)
        oracle_probs = oracle_tbl[np.arange(2)[:, None, None], indices]
        np.testing.assert_allclose(oracle_probs, archive[f"{cid}__direct_table_probs"], atol=1e-9, rtol=0.0)
        oracle_sc = score_vectors(oracle_probs.reshape(-1, 4), targets.reshape(-1))
        hist = build_cell_hist(indices, targets)
        hist_sc = score_table_from_hist(oracle_tbl, hist)
        for mk in ("acc", "nll", "brier", "ece"):
            assert abs(hist_sc[mk] - oracle_sc[mk]) < 1e-12, f"Hist mismatch {cid}.{mk}"
        cells[cid] = {
            "indices": indices,
            "targets": targets,
            "hist": hist,
            "metas": torch.stack(metas),
            "oracle_table": oracle_tbl,
            "oracle": {k: oracle_sc[k] for k in ("acc", "nll", "brier", "ece")},
        }

    # Extra Extreme-OOD Stress Cell G5: |theta| = pi (omega = +/- pi/4 = 0.785398, h=ell=2, V=0.04)
    g5_omega = math.pi / 4.0
    paired_g5, seeds_g5 = generate_paired_cfo_blocks(
        NS_G5_STRESS, 0, 8, 2, 2, 4.0, (-g5_omega, +g5_omega), 2.0, 0.10, split_prefix="G5_half_turn"
    )
    assert set(seeds_g5).isdisjoint(used_seeds)
    raw_codes, labels, metas, kernels = [], [], [], []
    for omega in (-g5_omega, +g5_omega):
        mother = paired_g5[omega]
        z = encode_view(encoder, mother, 0, 0)
        y_rep = np.stack([sector_of_complex(matched_sum(mother.y3_repeat * ROOTS[r])) for r, _ in ROTATIONS])
        raw_codes.append(np.stack([(z + [4 * r, 4 * s]) % 16 for r, s in ROTATIONS]))
        labels.append(y_rep)
        metas.append(metadata_row(mother.condition))
        kernels.append(physical_kernel(transport, omega * 4.0, 0.04, 2.0))
    indices_g5 = np.stack(raw_codes) @ [16, 1]
    targets_g5 = np.stack(labels)
    oracle_tbl_g5 = physical_oracle_table(encoder.table, np.stack(kernels))
    oracle_probs_g5 = oracle_tbl_g5[np.arange(2)[:, None, None], indices_g5]
    oracle_sc_g5 = score_vectors(oracle_probs_g5.reshape(-1, 4), targets_g5.reshape(-1))
    hist_g5 = build_cell_hist(indices_g5, targets_g5)
    cells["G5_extreme_half_turn_pi"] = {
        "indices": indices_g5,
        "targets": targets_g5,
        "hist": hist_g5,
        "metas": torch.stack(metas),
        "oracle_table": oracle_tbl_g5,
        "oracle": {k: oracle_sc_g5[k] for k in ("acc", "nll", "brier", "ece")},
    }
    return cells


def fit_dev_temperature(model: SurrogateAdapterNet, dev_data) -> float:
    codes, meta, y = dev_data
    model.eval()
    with torch.no_grad():
        probs = model.forward_probs(codes, meta).numpy()
    logits = np.log(np.clip(probs.astype(np.float64), 1e-12, 1.0))
    tgt = logits[np.arange(len(y)), y.numpy()]

    def obj(log_t: float) -> float:
        s = math.exp(-log_t)
        return float(np.mean(logsumexp(logits * s, axis=1) - tgt * s))

    res = minimize_scalar(obj, bounds=(math.log(0.1), math.log(10.0)), method="bounded")
    return float(math.exp(res.x))


def eval_snapshot(model: SurrogateAdapterNet, dev_data, cells: dict, cal_ref: np.ndarray, sqrt_w: np.ndarray) -> dict:
    temp = fit_dev_temperature(model, dev_data)
    model.eval()
    out_cells = {}
    d_ref = sqrt_w * (cal_ref[1] - cal_ref[0])
    d_ref_norm2 = float(np.sum(d_ref ** 2))
    g1_work_f = None

    with torch.no_grad():
        for cid, cdata in cells.items():
            # Normalize ONCE inside canonical slice before integer lift -> bitwise 0.000e+00 C4 defect!
            tbl_cal = np.stack([model.orbit_table_calibrated(m, temp) for m in cdata["metas"]], axis=0)  # (2, 256, 4)
            sc = score_table_from_hist(tbl_cal, cdata["hist"])
            defect = orbit_defect(tbl_cal)
            out_cells[cid] = {
                "acc": sc["acc"],
                "nll": sc["nll"],
                "brier": sc["brier"],
                "ece": sc["ece"],
                "equivariance_defect": defect,
            }
            if cid == "G1_main_extrap_both":
                d_mod = sqrt_w * (tbl_cal[1] - tbl_cal[0])
                orient_gain = float(np.sum(d_mod * d_ref) / d_ref_norm2)
                g1_work_f = {
                    "orientation_gain": orient_gain,
                    "rel_frob_err_to_cal_ref": float(np.linalg.norm(d_mod - d_ref) / math.sqrt(d_ref_norm2)),
                    **gram_spectrum(d_mod),
                }

        # Check exact 2*pi periodicity & quarter-turn phase covariance on G1 +omega
        m_base = cells["G1_main_extrap_both"]["metas"][1].clone()
        dt = float(m_base[6].item() + m_base[7].item())
        m_2pi = m_base.clone()
        m_2pi[9] = m_base[9] + (2.0 * math.pi / dt)
        m_qtr = m_base.clone()
        m_qtr[9] = m_base[9] + (0.5 * math.pi / dt)
        # Keep public phase slots consistent with shifted omega for fair check
        for m_sh in (m_2pi, m_qtr):
            w_sh = float(m_sh[9].item())
            m_sh[:6] = torch.from_numpy(public_phase_slots(w_sh, int(m_sh[6].item()), int(m_sh[7].item()), float(m_sh[8].item())))
        p_base = model.orbit_table_calibrated(m_base, temp)
        p_2pi = model.orbit_table_calibrated(m_2pi, temp)
        p_qtr = model.orbit_table_calibrated(m_qtr, temp)
        period_2pi_defect = float(np.max(np.abs(p_2pi - p_base)))
        phase_qtr_cov_defect = float(np.max(np.abs(p_qtr - np.roll(p_base, 1, axis=-1))))

    return {
        "dev_temp": temp,
        "cells": out_cells,
        "g1_work_f": g1_work_f,
        "period_2pi_defect": period_2pi_defect,
        "phase_quarter_turn_cov_defect": phase_qtr_cov_defect,
        "weight_diagnostics": model.diagnostics(),
    }


def precompute_bayes_kl_targets(
    transport: SectorTransport, prior: np.ndarray, codes: torch.Tensor, meta: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute exact physical Bayes posteriors p_phys(z1; theta, V) and counterfactual -omega pairs (W3 Thm 6/7)."""
    n = len(codes)
    idx_256 = (codes[:, 0] * 16 + codes[:, 1]).numpy()
    meta_np = meta.numpy()
    p_phys = np.empty((n, 4), dtype=np.float32)
    p_cf = np.empty((n, 4), dtype=np.float32)
    meta_cf = meta.clone()

    # Cache unique (h, ell, tp, omega, sigma, snr_db) conditions (only ~10 unique in Stage 3D train)
    cache: Dict[tuple, Tuple[np.ndarray, np.ndarray, torch.Tensor]] = {}
    keys = [
        (
            round(float(meta_np[i, 6]), 4),
            round(float(meta_np[i, 7]), 4),
            round(float(meta_np[i, 8]), 4),
            round(float(meta_np[i, 9]), 6),
            round(float(meta_np[i, 10]), 6),
            round(float(meta_np[i, 11]), 4),
        )
        for i in range(n)
    ]
    unique_keys = sorted(set(keys))
    for k in unique_keys:
        h, ell, tp, omega, sigma, snr_db = k
        dt = h + ell
        var = (sigma ** 2) * dt
        ker = physical_kernel(transport, omega * dt, var, snr_db)
        tbl = physical_oracle_table(prior, ker[None])[0].astype(np.float32)
        ker_cf = physical_kernel(transport, (-omega) * dt, var, snr_db)
        tbl_cf = physical_oracle_table(prior, ker_cf[None])[0].astype(np.float32)
        slots_cf = public_phase_slots(-omega, int(round(h)), int(round(ell)), tp)
        m_cf_row = torch.tensor([*slots_cf, h, ell, tp, -omega, sigma, snr_db], dtype=torch.float32)
        cache[k] = (tbl, tbl_cf, m_cf_row)

    key_to_indices: Dict[tuple, List[int]] = {k: [] for k in unique_keys}
    for i, k in enumerate(keys):
        key_to_indices[k].append(i)
    for k, pos_list in key_to_indices.items():
        pos = np.asarray(pos_list, dtype=np.int64)
        tbl, tbl_cf, m_cf_row = cache[k]
        p_phys[pos] = tbl[idx_256[pos]]
        p_cf[pos] = tbl_cf[idx_256[pos]]
        meta_cf[pos] = m_cf_row

    return torch.from_numpy(p_phys), meta_cf, torch.from_numpy(p_cf)


def train_and_evaluate_arm(
    arm_name: str,
    arm_type: str,
    quotient: bool,
    backbone: dict,
    prior: np.ndarray,
    train_data,
    dev_data,
    cells: dict,
    cal_ref: np.ndarray,
    sqrt_w: np.ndarray,
    *,
    dyn_rank: int = 5,
    max_harmonic: int = 2,
    k1_doublets: int = 4,
    k2_nyquist: int = 3,
    seeds: Tuple[int, ...] = MODEL_SEEDS,
    loss_mode: str = "ce",
    bayes_targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    nullspace_projs: dict | None = None,
    total_steps: int = TOTAL_STEPS,
    pilot_step: int = PILOT_STEP,
    return_models: bool = False,
):
    codes_tr, meta_tr, y_tr = train_data
    n_tr = len(y_tr)
    per_seed = {}
    trained_models: Dict[str, SurrogateAdapterNet] = {}
    trainable_params = None
    if loss_mode == "bayes_kl_cf":
        assert bayes_targets is not None
        p_phys_tr, meta_cf_tr, p_cf_tr = bayes_targets

    for seed in seeds:
        torch.manual_seed(seed)
        model = SurrogateAdapterNet(
            backbone, prior, arm_type, quotient=quotient,
            dyn_rank=dyn_rank, max_harmonic=max_harmonic,
            k1_doublets=k1_doublets, k2_nyquist=k2_nyquist,
            nullspace_projs=nullspace_projs,
        )
        trainable_params = model.count_trainable()

        trainable_list = [p for p in model.parameters() if p.requires_grad]
        if arm_type == "lora_plus":
            # Exact LoRA+ asymmetric LR: eta_B = 8 * eta_A with matched geometric mean
            ratio = 8.0
            a_params = [p for n, p in model.named_parameters() if p.requires_grad and n.endswith(".A")]
            b_params = [p for n, p in model.named_parameters() if p.requires_grad and n.endswith(".B")]
            opt = torch.optim.AdamW(
                [
                    {"params": a_params, "lr": BASE_LR / math.sqrt(ratio)},
                    {"params": b_params, "lr": BASE_LR * math.sqrt(ratio)},
                ],
                weight_decay=1e-4,
                foreach=True,
            )
        else:
            opt = torch.optim.AdamW(trainable_list, lr=BASE_LR, weight_decay=1e-4, foreach=True)

        rng = torch.Generator().manual_seed(seed + 999)
        pilot_snap = None
        h_pre_tr = None
        if not (model.linear_final_dynamic and model.quotient and model.mask_input_phase):
            with torch.no_grad():
                h_pre_tr = model.encode_features(codes_tr, meta_tr)
        model.train()
        for step in range(1, total_steps + 1):
            idx = torch.randint(0, n_tr, (BATCH_SIZE,), generator=rng)
            if loss_mode == "ce":
                probs = model.forward_probs(codes_tr[idx], meta_tr[idx], h_pre=(h_pre_tr[idx] if h_pre_tr is not None else None))
                loss = F.nll_loss(probs.clamp_min(1e-12).log(), y_tr[idx])
            elif loss_mode == "bayes_kl_cf":
                # W3 Theorem 6+7: Exact Physical Posterior KL + Temp-Decoupled Norm + CF Sign-Flip Margin (lambda_orbit = 0)
                c_b = codes_tr[idx]
                m_b = meta_tr[idx]
                p_b = p_phys_tr[idx]
                m_cf_b = meta_cf_tr[idx]
                p_cf_b = p_cf_tr[idx]
                logits, logits_cf = model.forward_logits_pair(c_b, m_b, m_cf_b)
                log_q = F.log_softmax(logits, dim=-1)
                kl_loss = (p_b * (p_b.clamp_min(1e-12).log() - log_q)).sum(dim=-1).mean()
                s_c = logits - logits.mean(dim=-1, keepdim=True)
                t_log = p_b.clamp_min(1e-12).log()
                t_c = t_log - t_log.mean(dim=-1, keepdim=True)
                norm_loss = ((s_c.norm(dim=-1) - t_c.norm(dim=-1)) ** 2).mean()

                log_q_cf = F.log_softmax(logits_cf, dim=-1)
                kl_cf = (p_cf_b * (p_cf_b.clamp_min(1e-12).log() - log_q_cf)).sum(dim=-1).mean()
                s_cf_c = logits_cf - logits_cf.mean(dim=-1, keepdim=True)
                t_cf_log = p_cf_b.clamp_min(1e-12).log()
                t_cf_c = t_cf_log - t_cf_log.mean(dim=-1, keepdim=True)
                v_phys = t_c - t_cf_c
                v_norm = v_phys.norm(dim=-1)
                mask = (v_norm > 0.05).float()
                v_hat = v_phys / v_norm.clamp_min(1e-8).unsqueeze(-1)
                proj = ((s_c - s_cf_c) * v_hat).sum(dim=-1)
                margin_loss = (mask * F.relu(0.95 * v_norm - proj)).mean()
                loss = kl_loss + 0.05 * norm_loss + 0.15 * margin_loss + 0.50 * kl_cf
            else:
                raise ValueError(f"Unknown loss_mode={loss_mode}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_list, 2.0, foreach=True)
            opt.step()
            if step == pilot_step:
                pilot_snap = eval_snapshot(model, dev_data, cells, cal_ref, sqrt_w)
                model.train()
        final_snap = eval_snapshot(model, dev_data, cells, cal_ref, sqrt_w)
        per_seed[str(seed)] = {"step_69_pilot": pilot_snap, "step_384_final": final_snap}
        if return_models:
            trained_models[str(seed)] = model

    # Aggregate seed means + cross-seed SD
    def aggregate(stage_key: str) -> dict:
        agg_cells = {}
        for cid in cells:
            agg_cells[cid] = {
                k: float(np.mean([per_seed[str(s)][stage_key]["cells"][cid][k] for s in seeds]))
                for k in ("acc", "nll", "brier", "ece")
            }
            agg_cells[cid]["acc_sd"] = float(np.std([per_seed[str(s)][stage_key]["cells"][cid]["acc"] for s in seeds], ddof=1)) if len(seeds) > 1 else 0.0
            agg_cells[cid]["nll_sd"] = float(np.std([per_seed[str(s)][stage_key]["cells"][cid]["nll"] for s in seeds], ddof=1)) if len(seeds) > 1 else 0.0
            agg_cells[cid]["max_equivariance_defect"] = float(
                max(per_seed[str(s)][stage_key]["cells"][cid]["equivariance_defect"] for s in seeds)
            )
        wf_keys = ("orientation_gain", "rel_frob_err_to_cal_ref", "effective_rank", "stable_rank", "frobenius_norm")
        agg_wf = {k: float(np.mean([per_seed[str(s)][stage_key]["g1_work_f"][k] for s in seeds])) for k in wf_keys}
        cond_vals = [
            per_seed[str(s)][stage_key]["weight_diagnostics"]["max_effective_weight_cond"]
            for s in seeds
            if per_seed[str(s)][stage_key]["weight_diagnostics"]["max_effective_weight_cond"] is not None
        ]
        ortho_vals = [
            per_seed[str(s)][stage_key]["weight_diagnostics"]["cayley_max_ortho_defect"]
            for s in seeds
            if per_seed[str(s)][stage_key]["weight_diagnostics"]["cayley_max_ortho_defect"] is not None
        ]
        so2_vals = [
            per_seed[str(s)][stage_key]["weight_diagnostics"]["cayley_max_so2_defect"]
            for s in seeds
            if per_seed[str(s)][stage_key]["weight_diagnostics"]["cayley_max_so2_defect"] is not None
        ]
        return {
            "cells": agg_cells,
            "g1_work_f": agg_wf,
            "mean_dev_temp": float(np.mean([per_seed[str(s)][stage_key]["dev_temp"] for s in seeds])),
            "max_period_2pi_defect": float(max(per_seed[str(s)][stage_key]["period_2pi_defect"] for s in seeds)),
            "max_phase_quarter_turn_cov_defect": float(
                max(per_seed[str(s)][stage_key]["phase_quarter_turn_cov_defect"] for s in seeds)
            ),
            "max_effective_weight_cond": float(max(cond_vals)) if cond_vals else None,
            "cayley_max_ortho_defect": float(max(ortho_vals)) if ortho_vals else None,
            "cayley_max_so2_defect": float(max(so2_vals)) if so2_vals else None,
        }

    out = {
        "arm_name": arm_name,
        "arm_type": arm_type,
        "quotient": quotient,
        "loss_mode": loss_mode,
        "lambda_orbit": 0.0,
        "trainable_params": trainable_params,
        "step_69_pilot_seed_mean": aggregate("step_69_pilot"),
        "step_384_final_seed_mean": aggregate("step_384_final"),
        "per_seed": per_seed,
    }
    if return_models:
        return out, trained_models
    return out


# ---------------------------------------------------------------------------
# Round 2 Part D: Theorem 7 Nullspace Cayley (4d) + W3 Bayes-KL & MixKL Audit
# ---------------------------------------------------------------------------

def measure_text_kl_metrics(
    model: SurrogateAdapterNet, x_gen: torch.Tensor, x_mix: torch.Tensor, w_text: torch.Tensor
) -> dict:
    model.eval()
    with torch.no_grad():
        z0_gen = model.forward_text_logits(x_gen, w_text, g_em=False)
        z0_mix = model.forward_text_logits(x_mix, w_text, g_em=False)
        zg_gen = model.forward_text_logits(x_gen, w_text, g_em=False)  # g_EM = 0 binary gate
        zu_gen = model.forward_text_logits(x_gen, w_text, g_em=True)   # g_EM = 1 ungated
        za_mix = model.forward_text_logits(x_mix, w_text, g_em=True)   # g_EM = 1 post-EM mixed prose

        def mean_kl(z_ref: torch.Tensor, z_mod: torch.Tensor) -> float:
            p = F.softmax(z_ref, dim=-1)
            return float((p * (F.log_softmax(z_ref, dim=-1) - F.log_softmax(z_mod, dim=-1))).sum(dim=-1).mean().item())

        return {
            "pure_kl_gated_gEM0": mean_kl(z0_gen, zg_gen),
            "pure_max_logit_diff_gated": float((z0_gen - zg_gen).abs().max().item()),
            "pure_kl_ungated_gEM1": mean_kl(z0_gen, zu_gen),
            "mix_kl_on_task_gEM1": mean_kl(z0_mix, za_mix),
        }


def measure_grad_variance_ratio(
    model: SurrogateAdapterNet,
    train_data,
    bayes_targets: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    n_batches: int = 64,
) -> dict:
    """Direct empirical Rao-Blackwell minibatch gradient variance comparison: 1-Hot CE vs Bayes-KL."""
    codes_tr, meta_tr, y_tr = train_data
    p_phys_tr, _, _ = bayes_targets
    n_tr = len(y_tr)
    params = [p for p in model.parameters() if p.requires_grad]
    rng = torch.Generator().manual_seed(202699)
    grads_ce, grads_kl = [], []
    model.eval()
    for _ in range(n_batches):
        idx = torch.randint(0, n_tr, (BATCH_SIZE,), generator=rng)
        c_b, m_b, y_b, p_b = codes_tr[idx], meta_tr[idx], y_tr[idx], p_phys_tr[idx]

        model.zero_grad(set_to_none=True)
        probs = model.forward_probs(c_b, m_b)
        F.nll_loss(probs.clamp_min(1e-12).log(), y_b).backward()
        grads_ce.append(torch.cat([p.grad.detach().reshape(-1) for p in params]))

        model.zero_grad(set_to_none=True)
        logits = model.forward_logits(c_b, m_b)
        log_q = F.log_softmax(logits, dim=-1)
        (p_b * (p_b.clamp_min(1e-12).log() - log_q)).sum(dim=-1).mean().backward()
        grads_kl.append(torch.cat([p.grad.detach().reshape(-1) for p in params]))

    g_ce = torch.stack(grads_ce, dim=0)
    g_kl = torch.stack(grads_kl, dim=0)
    var_ce = float(g_ce.var(dim=0, unbiased=True).sum().item())
    var_kl = float(g_kl.var(dim=0, unbiased=True).sum().item())
    return {
        "n_batches": n_batches,
        "trace_cov_1hot_ce": var_ce,
        "trace_cov_bayes_kl": var_kl,
        "variance_reduction_factor": float(var_ce / max(var_kl, 1e-30)),
    }


def run_round2_nullspace_and_bayes_kl(
    backbone: dict,
    prior: np.ndarray,
    transport: SectorTransport,
    train_data,
    dev_data,
    cells: dict,
    cal_ref: np.ndarray,
    sqrt_w: np.ndarray,
    arm_4c_ce_res: dict,
    arm_4c_ce_models: Dict[str, SurrogateAdapterNet],
) -> dict:
    """Round 2 Checks 1 & 2: Theorem 7 Nullspace Cayley (4d) + W3 Bayes-KL vs 1-Hot CE + MixKL."""
    # Verify exact direct Woodbury forward vs materialized effective_weight() (< 1e-5)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(202688)
        x_chk = torch.randn(16, D_MODEL)
        th_chk = torch.zeros(16)
        vr_chk = torch.full((16,), 0.04)
        l_cay = ComplexCayleyLayer(backbone["w_bases"][0], r_cayley=3, r_comm=3)
        with torch.no_grad():
            l_cay.u_re.normal_(0, 0.3)
            l_cay.u_im.normal_(0, 0.3)
            l_cay.bc_re.normal_(0, 0.2)
            l_cay.bc_im.normal_(0, 0.2)
            diff_cay = float((l_cay(x_chk, th_chk, vr_chk) - F.linear(x_chk, l_cay.effective_weight())).abs().max().item())
        assert diff_cay < 1e-5, f"ComplexCayley fast-forward mismatch: {diff_cay}"

    # Build generic non-complex-aligned general-text stream (10-dim principal cone + 0.02 tail)
    rng_np = np.random.default_rng(202677)
    n_gen = 512
    u_txt, _ = np.linalg.qr(rng_np.normal(size=(D_MODEL, D_MODEL)))
    sig_txt = np.concatenate([np.linspace(2.4, 1.0, 10), np.full(D_MODEL - 10, 0.02)])
    x_gen_np = (rng_np.normal(size=(n_gen, D_MODEL)) * sig_txt[None, :]) @ u_txt.T
    x_gen = torch.from_numpy(x_gen_np.astype(np.float32))
    # Frozen 16-class general-text readout head
    u_head, _ = np.linalg.qr(rng_np.normal(size=(D_MODEL, 16)))
    w_text = torch.from_numpy((1.4 * u_head.T).astype(np.float32))  # (16, 64)

    # Collect exact layerwise C_gen^(l) by propagating x_gen through frozen base backbone
    pi_perp_list, p_perp_list, layer_diag = [], [], []
    k_c = 10  # blocks 10 complex doublets (20 real dims), preserves 22 intact complex doublets (44 real dims = 68.75%)
    with torch.no_grad():
        h_cur = x_gen.clone()
        for l_idx in range(N_LAYERS):
            h_norm = rms_norm(h_cur)
            c_gen_l = (h_norm.double().T @ h_norm.double()).numpy() / float(n_gen)
            pi_c, p_real, evals_c, _ = compute_complex_nullspace_projector(c_gen_l, k_c=k_c)
            pi_perp_list.append(pi_c)
            p_perp_list.append(p_real)
            retained_txt_frac = float(np.sum(evals_c[:k_c]) / np.sum(evals_c))
            layer_diag.append({
                "layer": l_idx,
                "k_c_blocked": k_c,
                "intact_doublets": D_MODEL // 2 - k_c,
                "blocked_complex_energy_frac": retained_txt_frac,
            })
            h_cur = h_cur + 0.5 * F.gelu(F.linear(h_norm, backbone["w_bases"][l_idx]))

    nullspace_projs = {"pi_perp_c": pi_perp_list, "p_perp_real": p_perp_list}

    # Verify NullspaceComplexCayleyLayer fast-forward equivalence (< 1e-5)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(202689)
        l_ncay = NullspaceComplexCayleyLayer(backbone["w_bases"][0], pi_perp_list[0], r_cayley=3, r_comm=3)
        with torch.no_grad():
            l_ncay.u_re.normal_(0, 0.3)
            l_ncay.u_im.normal_(0, 0.3)
            l_ncay.bc_re.normal_(0, 0.2)
            l_ncay.bc_im.normal_(0, 0.2)
            diff_ncay = float((l_ncay(x_chk, th_chk, vr_chk) - F.linear(x_chk, l_ncay.effective_weight())).abs().max().item())
        assert diff_ncay < 1e-5, f"NullspaceComplexCayley fast-forward mismatch: {diff_ncay}"

    # Build post-EM mixed prose activations x_mix = x_gen + 0.15 * x_em (g_EM = 1)
    ref_model = arm_4c_ce_models[str(MODEL_SEEDS[0])]
    with torch.no_grad():
        dev_c, dev_m, _ = dev_data
        x_em = ref_model.encode_features(dev_c[:n_gen], dev_m[:n_gen])
        x_mix = x_gen + 0.15 * x_em

    # Precompute exact Bayes-KL + counterfactual targets for Stage 3D train_data
    bayes_targets = precompute_bayes_kl_targets(transport, prior, train_data[0], train_data[1])

    # Direct Rao-Blackwell variance reduction measurement on converged 4c
    grad_var_check = measure_grad_variance_ratio(ref_model, train_data, bayes_targets, n_batches=64)

    # Train 4c with W3 Bayes-KL + CF-Margin (lambda_orbit = 0)
    arm_4c_kl_res, arm_4c_kl_models = train_and_evaluate_arm(
        "4c_bayes_kl_cf", "quotient_cayley_schur_hyper_lora", True, backbone, prior,
        train_data, dev_data, cells, cal_ref, sqrt_w,
        k1_doublets=4, k2_nyquist=3, loss_mode="bayes_kl_cf", bayes_targets=bayes_targets,
        return_models=True,
    )
    # Train 4d: Theorem 7 Complex-Symmetrized Nullspace Cayley + Nullspace Schur + W3 Bayes-KL (lambda_orbit = 0)
    arm_4d_res, arm_4d_models = train_and_evaluate_arm(
        "4d_nullspace_cayley_schur_bayes_kl", "quotient_nullspace_cayley_schur", True, backbone, prior,
        train_data, dev_data, cells, cal_ref, sqrt_w,
        k1_doublets=4, k2_nyquist=3, loss_mode="bayes_kl_cf", bayes_targets=bayes_targets,
        nullspace_projs=nullspace_projs, return_models=True,
    )

    def mean_text_metrics(models_dict: Dict[str, SurrogateAdapterNet]) -> dict:
        ms = [measure_text_kl_metrics(m, x_gen, x_mix, w_text) for m in models_dict.values()]
        return {k: float(np.mean([d[k] for d in ms])) for k in ms[0]}

    txt_4c_ce = mean_text_metrics(arm_4c_ce_models)
    txt_4c_kl = mean_text_metrics(arm_4c_kl_models)
    txt_4d_kl = mean_text_metrics(arm_4d_models)

    # Strict Round 2 assertions for Checks 1 & 2
    assert txt_4c_ce["pure_kl_gated_gEM0"] == 0.0 and txt_4c_kl["pure_kl_gated_gEM0"] == 0.0 and txt_4d_kl["pure_kl_gated_gEM0"] == 0.0
    assert arm_4c_kl_res["step_384_final_seed_mean"]["cells"]["G1_main_extrap_both"]["max_equivariance_defect"] == 0.0
    assert arm_4d_res["step_384_final_seed_mean"]["cells"]["G1_main_extrap_both"]["max_equivariance_defect"] == 0.0
    assert arm_4d_res["step_384_final_seed_mean"]["cayley_max_ortho_defect"] < 1e-5
    assert arm_4d_res["step_384_final_seed_mean"]["cayley_max_so2_defect"] < 1e-5
    assert arm_4d_res["step_384_final_seed_mean"]["max_phase_quarter_turn_cov_defect"] < 1e-5
    assert txt_4d_kl["pure_kl_ungated_gEM1"] < txt_4c_kl["pure_kl_ungated_gEM1"] * 0.25
    assert txt_4d_kl["mix_kl_on_task_gEM1"] < txt_4c_kl["mix_kl_on_task_gEM1"] * 0.50

    return {
        "fast_woodbury_forward_max_err": {"complex_cayley": diff_cay, "nullspace_complex_cayley": diff_ncay},
        "nullspace_layer_diagnostics": layer_diag,
        "rao_blackwell_grad_variance": grad_var_check,
        "text_kl_audit": {
            "4c_1hot_ce": txt_4c_ce,
            "4c_bayes_kl_cf": txt_4c_kl,
            "4d_nullspace_cayley_schur_bayes_kl": txt_4d_kl,
            "pure_ungated_kl_reduction_4c_to_4d": float(txt_4c_kl["pure_kl_ungated_gEM1"] / max(txt_4d_kl["pure_kl_ungated_gEM1"], 1e-30)),
            "mix_kl_reduction_4c_to_4d": float(txt_4c_kl["mix_kl_on_task_gEM1"] / max(txt_4d_kl["mix_kl_on_task_gEM1"], 1e-30)),
        },
        "arms": {
            "4c_1hot_ce": {
                "trainable_params": arm_4c_ce_res["trainable_params"],
                "loss_mode": "ce",
                "lambda_orbit": 0.0,
                "step_69_pilot_seed_mean": arm_4c_ce_res["step_69_pilot_seed_mean"],
                "step_384_final_seed_mean": arm_4c_ce_res["step_384_final_seed_mean"],
            },
            "4c_bayes_kl_cf": arm_4c_kl_res,
            "4d_nullspace_cayley_schur_bayes_kl": arm_4d_res,
        },
    }


# ---------------------------------------------------------------------------
# Round 2 Part E: Theorem 8 Canonical Frame Alignment & Multi-Regime Merging
# ---------------------------------------------------------------------------

def _ties_merge_arrays(a1: np.ndarray, a2: np.ndarray, trim_frac: float = 0.40, scale: float = 1.0) -> np.ndarray:
    """Standard TIES-Merging (trim low-magnitude coords, elect mass sign, disjoint mean)."""
    def trim(x: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(x):
            mag = np.abs(x)
            thr = np.quantile(mag.ravel(), trim_frac)
            return np.where(mag >= thr, x, 0.0)
        mag = np.abs(x)
        thr = np.quantile(mag.ravel(), trim_frac)
        return np.where(mag >= thr, x, 0.0)

    t1, t2 = trim(a1), trim(a2)
    if np.iscomplexobj(a1):
        re = _ties_merge_arrays(a1.real, a2.real, trim_frac=trim_frac, scale=scale)
        im = _ties_merge_arrays(a1.imag, a2.imag, trim_frac=trim_frac, scale=scale)
        return re + 1j * im
    sign_sum = t1 + t2
    elected = np.sign(sign_sum)
    m1 = (t1 != 0.0) & (np.sign(t1) == elected)
    m2 = (t2 != 0.0) & (np.sign(t2) == elected)
    denom = m1.astype(np.float64) + m2.astype(np.float64)
    num = np.where(m1, t1, 0.0) + np.where(m2, t2, 0.0)
    return scale * np.where(denom > 0.0, num / np.maximum(denom, 1.0), 0.0)


def _draw_regime_split(
    encoder, transport: SectorTransport, namespace: int, blocks: int,
    omegas: Tuple[float, ...], snr_db: float, sigma_phi: float, prefix: str,
):
    paired, _ = generate_paired_cfo_blocks(
        namespace, 0, blocks, 2, 2, 4.0, omegas, snr_db, sigma_phi, split_prefix=prefix
    )
    c_list, m_list, y_list = [], [], []
    raw_codes, labels, metas, kernels = [], [], [], []
    for omega in omegas:
        mother = paired[omega]
        z = encode_view(encoder, mother, 0, 0)
        n = len(mother.x1)
        m_row = metadata_row(mother.condition)
        c_list.append(torch.from_numpy(z))
        m_list.append(m_row.expand(n, -1))
        y_list.append(torch.from_numpy(mother.k3_repeat))
        y_rep = np.stack([sector_of_complex(matched_sum(mother.y3_repeat * ROOTS[r])) for r, _ in ROTATIONS])
        raw_codes.append(np.stack([(z + [4 * r, 4 * s]) % 16 for r, s in ROTATIONS]))
        labels.append(y_rep)
        metas.append(m_row)
        kernels.append(physical_kernel(transport, omega * 4.0, (sigma_phi ** 2) * 4.0, snr_db))
    indices = np.stack(raw_codes) @ [16, 1]
    targets = np.stack(labels)
    oracle_tbl = physical_oracle_table(encoder.table, np.stack(kernels))
    oracle_probs = oracle_tbl[np.arange(len(omegas))[:, None, None], indices]
    oracle_sc = score_vectors(oracle_probs.reshape(-1, 4), targets.reshape(-1))
    train_tup = (torch.cat(c_list), torch.cat(m_list), torch.cat(y_list))
    eval_cell = {
        "indices": indices,
        "targets": targets,
        "hist": build_cell_hist(indices, targets),
        "metas": torch.stack(metas),
        "oracle": {k: oracle_sc[k] for k in ("acc", "nll")},
    }
    return train_tup, eval_cell


def _forward_lora_explicit_f64(
    backbone: dict, prior: np.ndarray, deltas: List[np.ndarray], codes: np.ndarray, meta: np.ndarray
) -> np.ndarray:
    """Exact float64 forward pass of 4-layer StaticLoRA network given explicit delta_W list."""
    p1 = prior[codes[:, 0]]
    p2 = prior[codes[:, 1]]
    oh1 = np.eye(16, dtype=np.float64)[codes[:, 0]]
    oh2 = np.eye(16, dtype=np.float64)[codes[:, 1]]
    slots = meta[:, :6]
    h_v, ell_v, _, om_v, sig_v, _ = [meta[:, j] for j in range(6, 12)]
    dt = h_v + ell_v
    scalars = np.stack((h_v / 4.0, ell_v / 4.0, om_v * 2.0, sig_v * np.sqrt(dt)), axis=-1)
    raw = np.concatenate((p1, p2, oh1, oh2, slots, scalars), axis=-1)
    h = raw @ backbone["e_proj"].double().numpy().T
    for l_idx in range(N_LAYERS):
        hn = h / np.sqrt(np.mean(h ** 2, axis=-1, keepdims=True) + 1e-6)
        w_eff = backbone["w_bases"][l_idx].double().numpy() + deltas[l_idx]
        z = hn @ w_eff.T
        # Exact GELU
        gelu_z = 0.5 * z * (1.0 + scipy.special.erf(z / math.sqrt(2.0)))
        h = h + 0.5 * gelu_z
    return h @ backbone["w_out"].double().numpy().T


def _forward_canonical_4c_f64(
    backbone: dict,
    prior: np.ndarray,
    s_list: List[np.ndarray],
    c_list: List[np.ndarray],
    schur_rows: Tuple[np.ndarray, np.ndarray, np.ndarray],
    codes: np.ndarray,
    meta: np.ndarray,
    *,
    q_override: List[np.ndarray] | None = None,
) -> np.ndarray:
    """Exact float64 forward pass of W2 Canonical 4c from (S_C, C_C, canonical Schur rows)."""
    ids = codes % 4
    p1 = prior[ids[:, 0]]
    p2 = prior[ids[:, 1]]
    oh1 = np.eye(16, dtype=np.float64)[ids[:, 0]]
    oh2 = np.eye(16, dtype=np.float64)[ids[:, 1]]
    phase_zero = np.zeros((len(codes), 10), dtype=np.float64)
    raw = np.concatenate((p1, p2, oh1, oh2, phase_zero), axis=-1)
    h = raw @ backbone["e_proj"].double().numpy().T

    dt = meta[:, 6] + meta[:, 7]
    theta = meta[:, 9] * dt
    var = (meta[:, 10] ** 2) * dt

    w_out = backbone["w_out"].double().numpy()
    p_null = np.eye(D_MODEL, dtype=np.float64) - np.linalg.pinv(w_out) @ w_out
    s2 = 1.0 / math.sqrt(2.0)
    f_real = np.array([[s2, 0.0, 0.5], [0.0, s2, -0.5], [-s2, 0.0, 0.5], [0.0, -s2, -0.5]], dtype=np.float64)
    b_align = np.linalg.pinv(w_out) @ f_real

    for l_idx in range(N_LAYERS - 1):
        hn = h / np.sqrt(np.mean(h ** 2, axis=-1, keepdims=True) + 1e-6)
        if q_override is not None:
            q_r = q_override[l_idx]
        else:
            s_c = s_list[l_idx]
            m = s_c.shape[0]
            q_c = (np.eye(m, dtype=np.complex128) - 0.5 * s_c) @ np.linalg.inv(np.eye(m, dtype=np.complex128) + 0.5 * s_c)
            q_r = realify_complex_matrix(q_c)
        w_eff = q_r @ backbone["w_bases"][l_idx].double().numpy() + realify_complex_matrix(c_list[l_idx])
        z = hn @ w_eff.T
        gelu_z = 0.5 * z * (1.0 + scipy.special.erf(z / math.sqrt(2.0)))
        h = h + 0.5 * gelu_z

    hn = h / np.sqrt(np.mean(h ** 2, axis=-1, keepdims=True) + 1e-6)
    a1_eff, a2_cos, a2_sin = schur_rows
    u = hn @ a1_eff.T  # (B, 2)
    d1 = np.exp(-0.5 * var)
    c1, s1 = d1 * np.cos(theta), d1 * np.sin(theta)
    d2 = np.exp(-2.0 * var)
    c2, s2_t = d2 * np.cos(2.0 * theta), d2 * np.sin(2.0 * theta)
    coords = np.stack(
        (
            c1 * u[:, 0] - s1 * u[:, 1],
            s1 * u[:, 0] + c1 * u[:, 1],
            c2 * (hn @ a2_cos) + s2_t * (hn @ a2_sin),
        ),
        axis=-1,
    )
    h_out = h @ p_null.T + coords @ b_align.T
    canon_logits = h_out @ w_out.T
    # Lift canonical logits by shift a1 = z1 // 4
    shifts = codes[:, 0] // 4
    col_idx = (np.arange(4)[None, :] - shifts[:, None]) % 4
    return np.take_along_axis(canon_logits, col_idx, axis=-1)


def _eval_logits_fn_on_cell(logits_fn, cell: dict, quotient: bool = False) -> dict:
    metas = cell["metas"].numpy().astype(np.float64)
    n_om = len(metas)
    all_codes_np = ALL_CODES.numpy()
    inner_codes_np = INNER_CODES.numpy()
    tables = []
    if quotient:
        z1 = all_codes_np[:, 0]
        z2 = all_codes_np[:, 1]
        inner_idx = (z1 % 4) * 4 + (z2 % 4)
        shift = z1 // 4
        col_idx = (np.arange(4)[None, :] - shift[:, None]) % 4
        for k in range(n_om):
            m_exp = np.broadcast_to(metas[k], (16, 12))
            canon_probs = softmax_temperature(logits_fn(inner_codes_np, m_exp), 1.0)
            tables.append(np.take_along_axis(canon_probs[inner_idx], col_idx, axis=-1))
    else:
        for k in range(n_om):
            m_exp = np.broadcast_to(metas[k], (256, 12))
            tables.append(softmax_temperature(logits_fn(all_codes_np, m_exp), 1.0))
    tbl = np.stack(tables, axis=0)
    sc = score_table_from_hist(tbl, cell["hist"])
    return {"acc": sc["acc"], "nll": sc["nll"], "equivariance_defect": orbit_defect(tbl)}


def _fisher_quad_form(logits_of_vec, base_vec: np.ndarray, direction: np.ndarray, eps: float = 1e-6) -> float:
    """Exact Gauss-Newton / Fisher quadratic form <v, F(base) v> = E[Var_{q(base)}(D_v logits)]."""
    lg0 = logits_of_vec(base_vec)
    q0 = softmax_temperature(lg0, 1.0)
    lg_p = logits_of_vec(base_vec + eps * direction)
    lg_m = logits_of_vec(base_vec - eps * direction)
    dv = (lg_p - lg_m) / (2.0 * eps)
    mean_dv = np.sum(q0 * dv, axis=-1, keepdims=True)
    var_dv = np.sum(q0 * ((dv - mean_dv) ** 2), axis=-1)
    return float(np.mean(var_dv))


def _train_regime_adapter(
    arm_type: str, quotient: bool, backbone: dict, prior: np.ndarray,
    train_tup, bayes_targets, seed: int, steps: int = 128,
) -> SurrogateAdapterNet:
    codes_tr, meta_tr, _ = train_tup
    p_phys_tr, meta_cf_tr, p_cf_tr = bayes_targets
    n_tr = len(codes_tr)
    torch.manual_seed(seed)
    model = SurrogateAdapterNet(backbone, prior, arm_type, quotient=quotient, k1_doublets=4, k2_nyquist=3)
    trainable_list = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_list, lr=BASE_LR, weight_decay=1e-4, foreach=True)
    rng = torch.Generator().manual_seed(seed + 555)
    model.train()
    for _ in range(steps):
        idx = torch.randint(0, n_tr, (BATCH_SIZE,), generator=rng)
        c_b, m_b, p_b = codes_tr[idx], meta_tr[idx], p_phys_tr[idx]
        m_cf_b, p_cf_b = meta_cf_tr[idx], p_cf_tr[idx]
        logits, logits_cf = model.forward_logits_pair(c_b, m_b, m_cf_b)
        log_q = F.log_softmax(logits, dim=-1)
        kl = (p_b * (p_b.clamp_min(1e-12).log() - log_q)).sum(dim=-1).mean()
        s_c = logits - logits.mean(dim=-1, keepdim=True)
        t_log = p_b.clamp_min(1e-12).log()
        t_c = t_log - t_log.mean(dim=-1, keepdim=True)
        norm_l = ((s_c.norm(dim=-1) - t_c.norm(dim=-1)) ** 2).mean()
        log_q_cf = F.log_softmax(logits_cf, dim=-1)
        kl_cf = (p_cf_b * (p_cf_b.clamp_min(1e-12).log() - log_q_cf)).sum(dim=-1).mean()
        s_cf_c = logits_cf - logits_cf.mean(dim=-1, keepdim=True)
        t_cf_log = p_cf_b.clamp_min(1e-12).log()
        t_cf_c = t_cf_log - t_cf_log.mean(dim=-1, keepdim=True)
        v_phys = t_c - t_cf_c
        v_norm = v_phys.norm(dim=-1)
        mask = (v_norm > 0.05).float()
        v_hat = v_phys / v_norm.clamp_min(1e-8).unsqueeze(-1)
        proj = ((s_c - s_cf_c) * v_hat).sum(dim=-1)
        margin_l = (mask * F.relu(0.95 * v_norm - proj)).mean()
        loss = kl + 0.05 * norm_l + 0.15 * margin_l + 0.50 * kl_cf
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_list, 2.0, foreach=True)
        opt.step()
    model.eval()
    return model


def run_round2_canonical_regime_merging(
    encoder, transport: SectorTransport, backbone: dict, g5_cell: dict
) -> dict:
    """Round 2 Check 3 (Theorem 8 / W4 Prop 4.1): Unconstrained LoRA vs W2 Canonical 4c Multi-Regime Merging."""
    prior = encoder.table
    # Regime 1 (Pluto-like: 1.5 dB, high jitter 0.24, small CFO |omega| <= 0.08)
    reg1_tr, reg1_eval = _draw_regime_split(encoder, transport, 9101, 16, (-0.08, 0.0, 0.08), 1.5, 0.24, "r2_reg1_pluto")
    # Regime 2 (CTTC-like: 5.0 dB, low jitter 0.05, large CFO |omega| in [0.32, 0.42])
    reg2_tr, reg2_eval = _draw_regime_split(encoder, transport, 9102, 12, (-0.42, -0.32, 0.32, 0.42), 5.0, 0.05, "r2_reg2_cttc")
    # Unseen Mixed Capture (f708_snr10 analogue: 3.5 dB, mid jitter 0.14, large CFO |omega| = 0.40 -> |theta| = 1.60)
    _, mix_eval = _draw_regime_split(encoder, transport, 9103, 8, (-0.40, 0.40), 3.5, 0.14, "r2_unseen_mix")

    bt1 = precompute_bayes_kl_targets(transport, prior, reg1_tr[0], reg1_tr[1])
    bt2 = precompute_bayes_kl_targets(transport, prior, reg2_tr[0], reg2_tr[1])

    # Independent regime seeds (realistic federated/independent adapter hub setting of W4 Prop 4.1)
    lora_1 = _train_regime_adapter("lora_zero", False, backbone, prior, reg1_tr, bt1, seed=202611, steps=128)
    lora_2 = _train_regime_adapter("lora_zero", False, backbone, prior, reg2_tr, bt2, seed=202622, steps=128)
    w2_1 = _train_regime_adapter("quotient_cayley_schur_hyper_lora", True, backbone, prior, reg1_tr, bt1, seed=202611, steps=128)
    w2_2 = _train_regime_adapter("quotient_cayley_schur_hyper_lora", True, backbone, prior, reg2_tr, bt2, seed=202622, steps=128)

    eval_suite = {
        "reg1_pluto_like": reg1_eval,
        "reg2_cttc_like": reg2_eval,
        "unseen_mixed_f708": mix_eval,
        "unseen_g5_half_turn": g5_cell,
    }
    oracles = {k: v["oracle"] for k, v in eval_suite.items()}

    # --- Family A: Unconstrained LoRA ---
    def extract_lora_deltas(m: SurrogateAdapterNet) -> List[np.ndarray]:
        return [
            (l.scale * (l.B.detach().double() @ l.A.detach().double())).numpy()
            for l in m.layers
        ]

    dw1 = extract_lora_deltas(lora_1)
    dw2 = extract_lora_deltas(lora_2)
    dw_soup = [0.5 * (a + b) for a, b in zip(dw1, dw2)]
    dw_ta = [a + b for a, b in zip(dw1, dw2)]
    dw_ties = [_ties_merge_arrays(a, b, trim_frac=0.40, scale=1.0) for a, b in zip(dw1, dw2)]
    # Naive GL(r) factor averaging: scale * (0.5*(B1+B2)) @ (0.5*(A1+A2))
    dw_naive_fac = [
        (
            l1.scale
            * (0.5 * (l1.B.detach().double() + l2.B.detach().double()))
            @ (0.5 * (l1.A.detach().double() + l2.A.detach().double()))
        ).numpy()
        for l1, l2 in zip(lora_1.layers, lora_2.layers)
    ]

    v_l1 = np.concatenate([x.ravel() for x in dw1])
    v_l2 = np.concatenate([x.ravel() for x in dw2])
    v_lsoup = np.concatenate([x.ravel() for x in dw_soup])
    v_lfac = np.concatenate([x.ravel() for x in dw_naive_fac])
    lora_gl_gauge_rel_err = float(np.linalg.norm(v_lfac - v_lsoup) / max(np.linalg.norm(v_lsoup), 1e-12))
    lora_cos_sim = float(np.dot(v_l1, v_l2) / (np.linalg.norm(v_l1) * np.linalg.norm(v_l2)))

    # Exact Fisher quadratic forms on 512 samples of Regime 1 and Regime 2
    r1_c, r1_m = reg1_tr[0][:512].numpy(), reg1_tr[1][:512].numpy().astype(np.float64)
    r2_c, r2_m = reg2_tr[0][:512].numpy(), reg2_tr[1][:512].numpy().astype(np.float64)

    def unpack_lora(vec: np.ndarray) -> List[np.ndarray]:
        sz = D_MODEL * D_MODEL
        return [vec[i * sz : (i + 1) * sz].reshape(D_MODEL, D_MODEL) for i in range(N_LAYERS)]

    f_l1 = lambda v: _forward_lora_explicit_f64(backbone, prior, unpack_lora(v), r1_c, r1_m)
    f_l2 = lambda v: _forward_lora_explicit_f64(backbone, prior, unpack_lora(v), r2_c, r2_m)
    l_f1_1 = _fisher_quad_form(f_l1, v_l1, v_l1)
    l_f1_2 = _fisher_quad_form(f_l1, v_l1, v_l2)
    l_f1_diff = _fisher_quad_form(f_l1, v_l1, v_l2 - v_l1)
    l_f2_2 = _fisher_quad_form(f_l2, v_l2, v_l2)
    l_f2_1 = _fisher_quad_form(f_l2, v_l2, v_l1)
    l_f2_diff = _fisher_quad_form(f_l2, v_l2, v_l1 - v_l2)

    lora_methods = {
        "adapter_1_only": dw1,
        "adapter_2_only": dw2,
        "naive_factor_avg_gl_pitfall": dw_naive_fac,
        "weight_soup_05_05": dw_soup,
        "ties_trim40_scale1": dw_ties,
        "task_arithmetic_10": dw_ta,
    }
    lora_eval = {}
    for m_name, dws in lora_methods.items():
        fn = lambda c, m, d=dws: _forward_lora_explicit_f64(backbone, prior, d, c, m)
        per_c = {cid: _eval_logits_fn_on_cell(fn, cell) for cid, cell in eval_suite.items()}
        per_c["macro_avg_nll_3reg"] = float(np.mean([per_c[k]["nll"] for k in ("reg1_pluto_like", "reg2_cttc_like", "unseen_mixed_f708")]))
        lora_eval[m_name] = per_c

    # --- Family B: W2 Canonical 4c (Complex-Cayley Lie Algebra + Gauge-Free Schur-Irrep) ---
    def extract_w2_canonical(m: SurrogateAdapterNet):
        s_list, c_list = [], []
        for l_idx in range(N_LAYERS - 1):
            lay = m.layers[l_idx]
            s_c = lay.skew_generator_complex().detach().to(torch.complex128).numpy()
            b_c = torch.complex(lay.bc_re.detach().double(), lay.bc_im.detach().double())
            a_c = torch.complex(lay.ac_re.detach().double(), lay.ac_im.detach().double())
            comm_c = (lay.comm_scale * (b_c @ a_c)).numpy()
            s_list.append(s_c)
            c_list.append(comm_c)
        a1, a2c, a2s = m.layers[N_LAYERS - 1].canonical_effective_rows()
        rows = (a1.detach().double().numpy(), a2c.detach().double().numpy(), a2s.detach().double().numpy())
        return s_list, c_list, rows

    s1, c1, r1_rows = extract_w2_canonical(w2_1)
    s2, c2, r2_rows = extract_w2_canonical(w2_2)

    # Verify Theorem 8 exact canonical Schur factor averaging == weight-space soup (< 1e-5)
    r_avg = tuple(0.5 * (a + b) for a, b in zip(r1_rows, r2_rows))
    schur_merged_layer = SchurIrrepHyperLoRALayer(backbone["w_out"], D_MODEL, k1_doublets=1, k2_nyquist=2)
    with torch.no_grad():
        schur_merged_layer.A1.copy_(torch.from_numpy(r_avg[0]).float())
        schur_merged_layer.c1_re.fill_(1.0)
        schur_merged_layer.c1_im.zero_()
        schur_merged_layer.A2.copy_(torch.from_numpy(np.stack([r_avg[1], r_avg[2]], axis=0)).float())
        schur_merged_layer.c2_cos.copy_(torch.tensor([1.0, 0.0]))
        schur_merged_layer.c2_sin.copy_(torch.tensor([0.0, 1.0]))
        rng_chk = torch.Generator().manual_seed(202691)
        x_t = torch.randn(32, D_MODEL, generator=rng_chk)
        th_t = torch.linspace(-math.pi, math.pi, 32)
        vr_t = torch.linspace(0.01, 0.30, 32)
        out_soup = 0.5 * (w2_1.layers[3](x_t, th_t, vr_t) + w2_2.layers[3](x_t, th_t, vr_t))
        out_canon = schur_merged_layer(x_t, th_t, vr_t)
        w2_schur_gauge_max_err = float((out_soup - out_canon).abs().max().item())
        assert w2_schur_gauge_max_err < 1e-5, f"Schur canonical merge err: {w2_schur_gauge_max_err}"

    # Verify Lie-Algebra Cayley merging unitarity & SO(2) vs Euclidean Q-Soup breakdown
    j_d = make_complex_structure(D_MODEL)
    s_soup = [0.5 * (a + b) for a, b in zip(s1, s2)]
    c_soup = [0.5 * (a + b) for a, b in zip(c1, c2)]
    s_ta = [a + b for a, b in zip(s1, s2)]
    c_ta = [a + b for a, b in zip(c1, c2)]
    r_ta = tuple(a + b for a, b in zip(r1_rows, r2_rows))
    s_ties = [0.5 * (t - t.conj().T) for t in (_ties_merge_arrays(a, b, 0.40, 1.0) for a, b in zip(s1, s2))]
    c_ties = [_ties_merge_arrays(a, b, 0.40, 1.0) for a, b in zip(c1, c2)]
    r_ties = tuple(_ties_merge_arrays(a, b, 0.40, 1.0) for a, b in zip(r1_rows, r2_rows))

    lie_ortho_max, lie_j_max = 0.0, 0.0
    euc_ortho_max, euc_sv_drop_max = 0.0, 0.0
    q_euc_list = []
    for l_idx in range(N_LAYERS - 1):
        m = D_MODEL // 2
        q1_c = (np.eye(m, dtype=np.complex128) - 0.5 * s1[l_idx]) @ np.linalg.inv(np.eye(m, dtype=np.complex128) + 0.5 * s1[l_idx])
        q2_c = (np.eye(m, dtype=np.complex128) - 0.5 * s2[l_idx]) @ np.linalg.inv(np.eye(m, dtype=np.complex128) + 0.5 * s2[l_idx])
        ql_c = (np.eye(m, dtype=np.complex128) - 0.5 * s_soup[l_idx]) @ np.linalg.inv(np.eye(m, dtype=np.complex128) + 0.5 * s_soup[l_idx])
        ql_r = realify_complex_matrix(ql_c)
        qe_r = realify_complex_matrix(0.5 * (q1_c + q2_c))
        q_euc_list.append(qe_r)
        lie_ortho_max = max(lie_ortho_max, float(np.max(np.abs(ql_r.T @ ql_r - np.eye(D_MODEL)))))
        lie_j_max = max(lie_j_max, float(np.max(np.abs(ql_r @ j_d - j_d @ ql_r))))
        euc_ortho_max = max(euc_ortho_max, float(np.max(np.abs(qe_r.T @ qe_r - np.eye(D_MODEL)))))
        sv_euc = np.linalg.svd(qe_r, compute_uv=False)
        euc_sv_drop_max = max(euc_sv_drop_max, float(1.0 - sv_euc[-1]))
    assert lie_ortho_max < 1e-12 and lie_j_max < 1e-12

    # Verify Theorem 7 + Theorem 8 closure: Lie-algebra merge of two nullspace skew generators stays in Pi_perp & U(m)
    rng_cl = np.random.default_rng(202692)
    u_cl, _ = np.linalg.qr(rng_cl.normal(size=(D_MODEL, D_MODEL)))
    c_cl = u_cl[:, :10] @ u_cl[:, :10].T
    pi_cl, _, _, u_top_cl = compute_complex_nullspace_projector(c_cl, k_c=10)
    s1_cl = pi_cl @ s1[0] @ pi_cl
    s2_cl = pi_cl @ s2[0] @ pi_cl
    s_m_cl = 0.5 * (s1_cl + s2_cl)
    q_m_cl = (np.eye(D_MODEL // 2, dtype=np.complex128) - 0.5 * s_m_cl) @ np.linalg.inv(np.eye(D_MODEL // 2, dtype=np.complex128) + 0.5 * s_m_cl)
    null_merge_text_drift = float(np.max(np.abs((q_m_cl - np.eye(D_MODEL // 2)) @ u_top_cl)))
    null_merge_ortho_err = float(np.max(np.abs(q_m_cl.conj().T @ q_m_cl - np.eye(D_MODEL // 2))))
    assert null_merge_text_drift < 1e-12 and null_merge_ortho_err < 1e-12

    # Pack/unpack canonical W2 operator vector (S_C_re, S_C_im, C_C_re, C_C_im for 3 layers + canonical Schur rows)
    m_dim = D_MODEL // 2
    m2 = m_dim * m_dim

    def pack_w2(sl, cl, rw) -> np.ndarray:
        parts = []
        for s_m, c_m in zip(sl, cl):
            parts.extend([s_m.real.ravel(), s_m.imag.ravel(), c_m.real.ravel(), c_m.imag.ravel()])
        parts.extend([rw[0].ravel(), rw[1].ravel(), rw[2].ravel()])
        return np.concatenate(parts)

    def unpack_w2(vec: np.ndarray):
        sl, cl = [], []
        off = 0
        for _ in range(N_LAYERS - 1):
            sr = vec[off : off + m2].reshape(m_dim, m_dim); off += m2
            si = vec[off : off + m2].reshape(m_dim, m_dim); off += m2
            cr = vec[off : off + m2].reshape(m_dim, m_dim); off += m2
            ci = vec[off : off + m2].reshape(m_dim, m_dim); off += m2
            s_c = sr + 1j * si
            s_c = 0.5 * (s_c - s_c.conj().T)  # stay strictly on Lie algebra u(m)
            sl.append(s_c)
            cl.append(cr + 1j * ci)
        a1 = vec[off : off + 2 * D_MODEL].reshape(2, D_MODEL); off += 2 * D_MODEL
        a2c = vec[off : off + D_MODEL]; off += D_MODEL
        a2s = vec[off : off + D_MODEL]
        return sl, cl, (a1, a2c, a2s)

    v_w1 = pack_w2(s1, c1, r1_rows)
    v_w2 = pack_w2(s2, c2, r2_rows)
    w2_cos_sim_full = float(np.dot(v_w1, v_w2) / (np.linalg.norm(v_w1) * np.linalg.norm(v_w2)))
    v_schur_1 = np.concatenate([x.ravel() for x in r1_rows])
    v_schur_2 = np.concatenate([x.ravel() for x in r2_rows])
    w2_cos_sim_schur = float(np.dot(v_schur_1, v_schur_2) / (np.linalg.norm(v_schur_1) * np.linalg.norm(v_schur_2)))

    f_w1 = lambda v: _forward_canonical_4c_f64(backbone, prior, *unpack_w2(v), r1_c, r1_m)
    f_w2 = lambda v: _forward_canonical_4c_f64(backbone, prior, *unpack_w2(v), r2_c, r2_m)
    w_f1_1 = _fisher_quad_form(f_w1, v_w1, v_w1)
    w_f1_2 = _fisher_quad_form(f_w1, v_w1, v_w2)
    w_f1_diff = _fisher_quad_form(f_w1, v_w1, v_w2 - v_w1)
    w_f2_2 = _fisher_quad_form(f_w2, v_w2, v_w2)
    w_f2_1 = _fisher_quad_form(f_w2, v_w2, v_w1)
    w_f2_diff = _fisher_quad_form(f_w2, v_w2, v_w1 - v_w2)

    w2_methods = {
        "adapter_1_only": (s1, c1, r1_rows, None),
        "adapter_2_only": (s2, c2, r2_rows, None),
        "canonical_lie_schur_soup_05_05": (s_soup, c_soup, r_avg, None),
        "euclidean_q_soup_breaks_unitarity": (s_soup, c_soup, r_avg, q_euc_list),
        "canonical_ties_trim40_scale1": (s_ties, c_ties, r_ties, None),
        "canonical_task_arithmetic_10": (s_ta, c_ta, r_ta, None),
    }
    w2_eval = {}
    for m_name, (sl, cl, rw, qo) in w2_methods.items():
        fn = lambda c, m, s_=sl, c_=cl, r_=rw, q_=qo: _forward_canonical_4c_f64(backbone, prior, s_, c_, r_, c, m, q_override=q_)
        per_c = {cid: _eval_logits_fn_on_cell(fn, cell, quotient=True) for cid, cell in eval_suite.items()}
        per_c["macro_avg_nll_3reg"] = float(np.mean([per_c[k]["nll"] for k in ("reg1_pluto_like", "reg2_cttc_like", "unseen_mixed_f708")]))
        w2_eval[m_name] = per_c

    return {
        "oracles": oracles,
        "lora_unconstrained": {
            "trainable_params": lora_1.count_trainable(),
            "gl_factor_avg_rel_err": lora_gl_gauge_rel_err,
            "cos_sim_deltaW": lora_cos_sim,
            "norm_reg1": float(np.linalg.norm(v_l1)),
            "norm_reg2": float(np.linalg.norm(v_l2)),
            "gamma_12_add": float(l_f1_2 / max(l_f1_1, 1e-30)),
            "gamma_21_add": float(l_f2_1 / max(l_f2_2, 1e-30)),
            "gamma_12_soup": float(l_f1_diff / max(l_f1_1, 1e-30)),
            "gamma_21_soup": float(l_f2_diff / max(l_f2_2, 1e-30)),
            "eval": lora_eval,
        },
        "w2_canonical_4c": {
            "trainable_params": w2_1.count_trainable(),
            "schur_canonical_factor_vs_soup_max_err": w2_schur_gauge_max_err,
            "lie_algebra_cayley_ortho_defect": lie_ortho_max,
            "lie_algebra_cayley_j_comm_defect": lie_j_max,
            "euclidean_q_soup_ortho_defect": euc_ortho_max,
            "euclidean_q_soup_max_sv_contraction": euc_sv_drop_max,
            "nullspace_lie_merge_text_drift": null_merge_text_drift,
            "nullspace_lie_merge_ortho_defect": null_merge_ortho_err,
            "cos_sim_full_canonical": w2_cos_sim_full,
            "cos_sim_schur_readout": w2_cos_sim_schur,
            "norm_reg1": float(np.linalg.norm(v_w1)),
            "norm_reg2": float(np.linalg.norm(v_w2)),
            "gamma_12_add": float(w_f1_2 / max(w_f1_1, 1e-30)),
            "gamma_21_add": float(w_f2_1 / max(w_f2_2, 1e-30)),
            "gamma_12_soup": float(w_f1_diff / max(w_f1_1, 1e-30)),
            "gamma_21_soup": float(w_f2_diff / max(w_f2_2, 1e-30)),
            "eval": w2_eval,
        },
    }


_WORKER_STATE: dict = {}


def _init_arm_worker(shared: dict) -> None:
    global _WORKER_STATE
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    _WORKER_STATE = shared


def _run_arm_worker(task: dict) -> dict:
    st = _WORKER_STATE
    need_models = bool(task.get("measure_text_kl") or task.get("measure_grad_var"))
    res_out = train_and_evaluate_arm(
        task["arm_name"],
        task["arm_type"],
        task["quotient"],
        st["backbone"],
        st["prior"],
        st["train_data"],
        st["dev_data"],
        st["cells"],
        st["cal_ref"],
        st["sqrt_w"],
        dyn_rank=task.get("dyn_rank", 5),
        max_harmonic=task.get("max_harmonic", 2),
        k1_doublets=task.get("k1_doublets", 4),
        k2_nyquist=task.get("k2_nyquist", 3),
        loss_mode=task.get("loss_mode", "ce"),
        bayes_targets=st["bayes_targets"] if task.get("loss_mode") == "bayes_kl_cf" else None,
        nullspace_projs=st["nullspace_projs"] if task.get("use_nullspace") else None,
        return_models=need_models,
    )
    if need_models:
        res, models = res_out
    else:
        res, models = res_out, None
    payload = {"group": task["group"], "arm_name": task["arm_name"], "res": res}
    if task.get("measure_text_kl") and models is not None:
        ms = [measure_text_kl_metrics(m, st["x_gen"], st["x_mix"], st["w_text"]) for m in models.values()]
        payload["text_kl"] = {k: float(np.mean([d[k] for d in ms])) for k in ms[0]}
    if task.get("measure_grad_var") and models is not None:
        ref_m = models[str(MODEL_SEEDS[0])]
        payload["grad_var"] = measure_grad_variance_ratio(ref_m, st["train_data"], st["bayes_targets"], n_batches=64)
    return payload


def _build_part_d_context(
    backbone: dict, prior: np.ndarray, transport: SectorTransport, train_data, dev_data
) -> dict:
    """Prepare Theorem 7 layerwise complex-nullspace projectors, x_gen/x_mix, w_text, and Bayes-KL targets."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(202688)
        x_chk = torch.randn(16, D_MODEL)
        th_chk = torch.zeros(16)
        vr_chk = torch.full((16,), 0.04)
        l_cay = ComplexCayleyLayer(backbone["w_bases"][0], r_cayley=3, r_comm=3)
        with torch.no_grad():
            l_cay.u_re.normal_(0, 0.3)
            l_cay.u_im.normal_(0, 0.3)
            l_cay.bc_re.normal_(0, 0.2)
            l_cay.bc_im.normal_(0, 0.2)
            diff_cay = float((l_cay(x_chk, th_chk, vr_chk) - F.linear(x_chk, l_cay.effective_weight())).abs().max().item())
        assert diff_cay < 1e-5, f"ComplexCayley fast-forward mismatch: {diff_cay}"

    rng_np = np.random.default_rng(202677)
    n_gen = 512
    u_txt, _ = np.linalg.qr(rng_np.normal(size=(D_MODEL, D_MODEL)))
    sig_txt = np.concatenate([np.linspace(2.4, 1.0, 10), np.full(D_MODEL - 10, 0.02)])
    x_gen_np = (rng_np.normal(size=(n_gen, D_MODEL)) * sig_txt[None, :]) @ u_txt.T
    x_gen = torch.from_numpy(x_gen_np.astype(np.float32))
    u_head, _ = np.linalg.qr(rng_np.normal(size=(D_MODEL, 16)))
    w_text = torch.from_numpy((1.4 * u_head.T).astype(np.float32))

    pi_perp_list, p_perp_list, layer_diag = [], [], []
    k_c = 10
    with torch.no_grad():
        h_cur = x_gen.clone()
        for l_idx in range(N_LAYERS):
            h_norm = rms_norm(h_cur)
            c_gen_l = (h_norm.double().T @ h_norm.double()).numpy() / float(n_gen)
            pi_c, p_real, evals_c, _ = compute_complex_nullspace_projector(c_gen_l, k_c=k_c)
            pi_perp_list.append(pi_c)
            p_perp_list.append(p_real)
            layer_diag.append({
                "layer": l_idx,
                "k_c_blocked": k_c,
                "intact_doublets": D_MODEL // 2 - k_c,
                "blocked_complex_energy_frac": float(np.sum(evals_c[:k_c]) / np.sum(evals_c)),
            })
            h_cur = h_cur + 0.5 * F.gelu(F.linear(h_norm, backbone["w_bases"][l_idx]))

    nullspace_projs = {"pi_perp_c": pi_perp_list, "p_perp_real": p_perp_list}

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(202689)
        l_ncay = NullspaceComplexCayleyLayer(backbone["w_bases"][0], pi_perp_list[0], r_cayley=3, r_comm=3)
        with torch.no_grad():
            l_ncay.u_re.normal_(0, 0.3)
            l_ncay.u_im.normal_(0, 0.3)
            l_ncay.bc_re.normal_(0, 0.2)
            l_ncay.bc_im.normal_(0, 0.2)
            diff_ncay = float((l_ncay(x_chk, th_chk, vr_chk) - F.linear(x_chk, l_ncay.effective_weight())).abs().max().item())
        assert diff_ncay < 1e-5, f"NullspaceComplexCayley fast-forward mismatch: {diff_ncay}"

    # Build post-EM mixed prose activations x_mix = x_gen + 0.15 * x_em (exact phase-masked canonical encoding)
    with torch.no_grad():
        dev_c, _, _ = dev_data
        ids = dev_c[:n_gen] % 4
        prior_t = torch.as_tensor(prior, dtype=torch.float32)
        raw_em = torch.cat(
            (
                prior_t[ids[:, 0]],
                prior_t[ids[:, 1]],
                F.one_hot(ids[:, 0], num_classes=16).float(),
                F.one_hot(ids[:, 1], num_classes=16).float(),
                torch.zeros(n_gen, 10, dtype=torch.float32),
            ),
            dim=-1,
        )
        x_em = F.linear(raw_em, backbone["e_proj"])
        x_mix = x_gen + 0.15 * x_em

    bayes_targets = precompute_bayes_kl_targets(transport, prior, train_data[0], train_data[1])
    return {
        "fast_woodbury_forward_max_err": {"complex_cayley": diff_cay, "nullspace_complex_cayley": diff_ncay},
        "nullspace_projs": nullspace_projs,
        "layer_diag": layer_diag,
        "x_gen": x_gen,
        "x_mix": x_mix,
        "w_text": w_text,
        "bayes_targets": bayes_targets,
    }


def main() -> None:
    t0 = time.perf_counter()
    t_cpu0 = time.process_time()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    encoder, s17_sha = load_s17_encoder(ROOT)
    transport = SectorTransport()
    archive_file = STAGE3D_RESULTS / "blind_q3_scores.npz"

    # Part A: Exact theorem certificates (Theorems 1, 2, 3, 4, 5, and Round 2 Theorem 7)
    print("[W2] Running Part A algebraic & spectral certificates (Thms 1-5 & R2 Thm 7)...", flush=True)
    cert_t1 = verify_theorem1_symmetry_breaking()
    cert_t3 = verify_theorem3_short_horizon_dynamics()
    cert_t4 = verify_theorem4_complex_cayley_depth32()
    cert_t7 = verify_theorem7_complex_nullspace_cayley()

    with np.load(archive_file, allow_pickle=False) as archive:
        cert_t25 = verify_theorem2_and_5_harmonics_and_rank(transport, encoder, archive)
        cal_ref = archive["G1_main_extrap_both__cal_ref"]
        cal_mass = archive["G1_main_extrap_both__cal_mass"]
        overlap = np.minimum(cal_mass[0], cal_mass[1])
        sqrt_w = np.sqrt(overlap / overlap.sum())[:, None]

        # Build train, dev, and real blind datasets
        train_data, train_seeds = draw_split(encoder, NS_CFO_TRAIN, TRAIN_BLOCKS, TRAIN_OMEGAS, "w2_train")
        dev_data, dev_seeds = draw_split(encoder, NS_CFO_DEV, DEV_BLOCKS, DEV_OMEGAS, "w2_dev")
        used_seeds = set(train_seeds) | set(dev_seeds)
        cells = build_blind_cells(encoder, transport, archive, used_seeds)

    backbone = make_frozen_backbone_weights(HOST_SEED)
    pd_ctx = _build_part_d_context(backbone, encoder.table, transport, train_data, dev_data)

    # Dispatch all 17 independent deterministic arms (Parts B + C + D) across worker pool (bit-identical!)
    arm_specs = [
        ("1_lora_zero_init", "lora_zero", False, 5, 2, 4, 3),
        ("1b_lora_plus_asym_lr", "lora_plus", False, 5, 2, 4, 3),
        ("2a_pissa_svd_init", "pissa", False, 5, 2, 4, 3),
        ("2b_dora_decomposed", "dora", False, 5, 2, 4, 3),
        ("2c_svft_banded", "svft", False, 5, 2, 4, 3),
        ("3a_quotient_cayley_pure", "quotient_cayley_pure", True, 5, 2, 4, 3),
        ("3b_quotient_cayley_hybrid", "quotient_cayley_hybrid", True, 5, 2, 4, 3),
        ("4a_quotient_dynamic_all4_unconstrained", "quotient_dynamic_all4", True, 5, 2, 4, 3),
        ("4b_quotient_harmonic_unconstrained", "quotient_cayley_plus_harmonic", True, 5, 2, 4, 3),
        ("4c_quotient_cayley_schur_hyper_lora", "quotient_cayley_schur_hyper_lora", True, 5, 2, 4, 3),
    ]
    rank_configs = [
        ("r=1_nyquist_only", 0, 1),
        ("r=2_fundamental_V1_only", 1, 0),
        ("r=3_minimal_complete_V1_plus_V2", 1, 1),
        ("r=4_V1_plus_2V2", 1, 2),
        ("r=6_2V1_plus_2V2", 2, 2),
    ]
    tasks: List[dict] = []
    for arm_name, arm_type, quot, dr, mh, k1, k2 in arm_specs:
        is_4c = arm_name == "4c_quotient_cayley_schur_hyper_lora"
        tasks.append({
            "group": "B", "arm_name": arm_name, "arm_type": arm_type, "quotient": quot,
            "dyn_rank": dr, "max_harmonic": mh, "k1_doublets": k1, "k2_nyquist": k2,
            "loss_mode": "ce", "measure_text_kl": is_4c, "measure_grad_var": is_4c,
        })
    for label, k1, k2 in rank_configs:
        tasks.append({
            "group": "C", "arm_name": label, "arm_type": "quotient_cayley_schur_hyper_lora",
            "quotient": True, "k1_doublets": k1, "k2_nyquist": k2, "loss_mode": "ce",
        })
    tasks.append({
        "group": "D", "arm_name": "4c_bayes_kl_cf", "arm_type": "quotient_cayley_schur_hyper_lora",
        "quotient": True, "k1_doublets": 4, "k2_nyquist": 3, "loss_mode": "bayes_kl_cf",
        "measure_text_kl": True,
    })
    tasks.append({
        "group": "D", "arm_name": "4d_nullspace_cayley_schur_bayes_kl", "arm_type": "quotient_nullspace_cayley_schur",
        "quotient": True, "k1_doublets": 4, "k2_nyquist": 3, "loss_mode": "bayes_kl_cf",
        "use_nullspace": True, "measure_text_kl": True,
    })

    # Strip large unused raw index/target arrays from worker cells payload (hist + metas are sufficient)
    worker_cells = {
        cid: {"hist": cdata["hist"], "metas": cdata["metas"]}
        for cid, cdata in cells.items()
    }
    shared_payload = {
        "backbone": backbone,
        "prior": encoder.table,
        "train_data": train_data,
        "dev_data": dev_data,
        "cells": worker_cells,
        "cal_ref": cal_ref,
        "sqrt_w": sqrt_w,
        "bayes_targets": pd_ctx["bayes_targets"],
        "nullspace_projs": pd_ctx["nullspace_projs"],
        "x_gen": pd_ctx["x_gen"],
        "x_mix": pd_ctx["x_mix"],
        "w_text": pd_ctx["w_text"],
    }
    n_workers = min(8, max(1, os.cpu_count() or 1))
    print(f"[W2] Running Parts B+C+D (17 deterministic arms x 3 seeds) across {n_workers} CPU workers + Part E concurrently...", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_arm_worker, initargs=(shared_payload,)) as pool:
        futs = [pool.submit(_run_arm_worker, t) for t in tasks]
        r2_part_e = run_round2_canonical_regime_merging(
            encoder, transport, backbone, cells["G5_extreme_half_turn_pi"]
        )
        completed = [f.result() for f in futs]

    benchmark_arms = {}
    rank_ablation = {}
    d_arms_raw = {}
    text_kl_map = {}
    grad_var_check = None
    for item in completed:
        if item["group"] == "B":
            benchmark_arms[item["arm_name"]] = item["res"]
            if "text_kl" in item:
                text_kl_map["4c_1hot_ce"] = item["text_kl"]
            if "grad_var" in item:
                grad_var_check = item["grad_var"]
        elif item["group"] == "C":
            rank_ablation[item["arm_name"]] = item["res"]
        elif item["group"] == "D":
            d_arms_raw[item["arm_name"]] = item["res"]
            text_kl_map[item["arm_name"]] = item["text_kl"]

    print("[W2] Part B matched-budget multi-layer surrogate benchmark (384 steps, step-69 pilot):", flush=True)
    for arm_name, *_ in arm_specs:
        res = benchmark_arms[arm_name]
        p69 = res["step_69_pilot_seed_mean"]["cells"]
        f384 = res["step_384_final_seed_mean"]["cells"]
        qcov = res["step_384_final_seed_mean"]["max_phase_quarter_turn_cov_defect"]
        print(
            f"  {arm_name:40s} | params={res['trainable_params']:4d} | "
            f"step69 G1={100*p69['G1_main_extrap_both']['acc']:5.2f}%/{p69['G1_main_extrap_both']['nll']:.4f} | "
            f"step384 G1={100*f384['G1_main_extrap_both']['acc']:5.2f}%/{f384['G1_main_extrap_both']['nll']:.4f} | "
            f"G3={100*f384['G3_extrap_angle']['acc']:5.2f}% | "
            f"G5(pi)={100*f384['G5_extreme_half_turn_pi']['acc']:5.2f}%/{f384['G5_extreme_half_turn_pi']['nll']:.4f} | "
            f"C4={f384['G1_main_extrap_both']['max_equivariance_defect']:.1e} | qtr={qcov:.1e}",
            flush=True,
        )

    print("[W2] Part C Minimal Rank Theorem (r = 1, 2, 3, 4, 6) Schur-Irrep ablation:", flush=True)
    for label, _, _ in rank_configs:
        res = rank_ablation[label]
        f384 = res["step_384_final_seed_mean"]["cells"]
        wf = res["step_384_final_seed_mean"]["g1_work_f"]
        print(
            f"  {label:32s} | params={res['trainable_params']:4d} | "
            f"G1={100*f384['G1_main_extrap_both']['acc']:5.2f}%/{f384['G1_main_extrap_both']['nll']:.4f} | "
            f"G2={100*f384['G2_interp_both']['acc']:5.2f}%/{f384['G2_interp_both']['nll']:.4f} | "
            f"G3={100*f384['G3_extrap_angle']['acc']:5.2f}%/{f384['G3_extrap_angle']['nll']:.4f} | "
            f"G5(pi)={100*f384['G5_extreme_half_turn_pi']['acc']:5.2f}%/{f384['G5_extreme_half_turn_pi']['nll']:.4f} | "
            f"WorkF_gain={wf['orientation_gain']:.4f}",
            flush=True,
        )

    # Assemble & assert Part D (Round 2)
    arm_4c_ce_res = benchmark_arms["4c_quotient_cayley_schur_hyper_lora"]
    arm_4c_kl_res = d_arms_raw["4c_bayes_kl_cf"]
    arm_4d_res = d_arms_raw["4d_nullspace_cayley_schur_bayes_kl"]
    txt_4c_ce = text_kl_map["4c_1hot_ce"]
    txt_4c_kl = text_kl_map["4c_bayes_kl_cf"]
    txt_4d_kl = text_kl_map["4d_nullspace_cayley_schur_bayes_kl"]
    assert txt_4c_ce["pure_kl_gated_gEM0"] == 0.0 and txt_4c_kl["pure_kl_gated_gEM0"] == 0.0 and txt_4d_kl["pure_kl_gated_gEM0"] == 0.0
    assert arm_4c_kl_res["step_384_final_seed_mean"]["cells"]["G1_main_extrap_both"]["max_equivariance_defect"] == 0.0
    assert arm_4d_res["step_384_final_seed_mean"]["cells"]["G1_main_extrap_both"]["max_equivariance_defect"] == 0.0
    assert arm_4d_res["step_384_final_seed_mean"]["cayley_max_ortho_defect"] < 1e-5
    assert arm_4d_res["step_384_final_seed_mean"]["cayley_max_so2_defect"] < 1e-5
    assert arm_4d_res["step_384_final_seed_mean"]["max_phase_quarter_turn_cov_defect"] < 1e-5
    assert txt_4d_kl["pure_kl_ungated_gEM1"] < txt_4c_kl["pure_kl_ungated_gEM1"] * 0.25
    assert txt_4d_kl["mix_kl_on_task_gEM1"] < txt_4c_kl["mix_kl_on_task_gEM1"] * 0.50

    r2_part_d = {
        "fast_woodbury_forward_max_err": pd_ctx["fast_woodbury_forward_max_err"],
        "nullspace_layer_diagnostics": pd_ctx["layer_diag"],
        "rao_blackwell_grad_variance": grad_var_check,
        "text_kl_audit": {
            "4c_1hot_ce": txt_4c_ce,
            "4c_bayes_kl_cf": txt_4c_kl,
            "4d_nullspace_cayley_schur_bayes_kl": txt_4d_kl,
            "pure_ungated_kl_reduction_4c_to_4d": float(txt_4c_kl["pure_kl_ungated_gEM1"] / max(txt_4d_kl["pure_kl_ungated_gEM1"], 1e-30)),
            "mix_kl_reduction_4c_to_4d": float(txt_4c_kl["mix_kl_on_task_gEM1"] / max(txt_4d_kl["mix_kl_on_task_gEM1"], 1e-30)),
        },
        "arms": {
            "4c_1hot_ce": {
                "trainable_params": arm_4c_ce_res["trainable_params"],
                "loss_mode": "ce",
                "lambda_orbit": 0.0,
                "step_69_pilot_seed_mean": arm_4c_ce_res["step_69_pilot_seed_mean"],
                "step_384_final_seed_mean": arm_4c_ce_res["step_384_final_seed_mean"],
            },
            "4c_bayes_kl_cf": arm_4c_kl_res,
            "4d_nullspace_cayley_schur_bayes_kl": arm_4d_res,
        },
    }
    print("[W2] Part D (Round 2): Thm 7 Nullspace Cayley (4d) + W3 Bayes-KL & MixKL audit:", flush=True)
    for k_arm in ("4c_1hot_ce", "4c_bayes_kl_cf", "4d_nullspace_cayley_schur_bayes_kl"):
        a_res = r2_part_d["arms"][k_arm]
        p69 = a_res["step_69_pilot_seed_mean"]["cells"]["G1_main_extrap_both"]
        f384 = a_res["step_384_final_seed_mean"]["cells"]
        tkl = r2_part_d["text_kl_audit"][k_arm]
        print(
            f"  {k_arm:36s} | step69 G1={100*p69['acc']:5.2f}%±{100*p69['acc_sd']:.2f}% / {p69['nll']:.4f}±{p69['nll_sd']:.4f} | "
            f"step384 G1={100*f384['G1_main_extrap_both']['acc']:5.2f}%/{f384['G1_main_extrap_both']['nll']:.4f} | "
            f"G5={100*f384['G5_extreme_half_turn_pi']['acc']:5.2f}%/{f384['G5_extreme_half_turn_pi']['nll']:.4f} | "
            f"PureKL(g0/g1)={tkl['pure_kl_gated_gEM0']:.1e}/{tkl['pure_kl_ungated_gEM1']:.4f} | MixKL={tkl['mix_kl_on_task_gEM1']:.4f}",
            flush=True,
        )

    # Part E (Round 2): Theorem 8 Canonical Frame Alignment & Multi-Regime Adapter Merging (computed concurrently above)
    print("[W2] Part E (Round 2): Thm 8 Multi-Regime Canonical Adapter Merging (W4 Prop 4.1):", flush=True)
    la, wb = r2_part_e["lora_unconstrained"], r2_part_e["w2_canonical_4c"]
    print(
        f"  LoRA Unconstrained | GL(r) err={la['gl_factor_avg_rel_err']:.4f} | cos={la['cos_sim_deltaW']:+.4f} | "
        f"Gamma_add(12/21)={la['gamma_12_add']:.3f}/{la['gamma_21_add']:.3f} | "
        f"Gamma_soup(12/21)={la['gamma_12_soup']:.3f}/{la['gamma_21_soup']:.3f} | "
        f"UnseenMix Soup={la['eval']['weight_soup_05_05']['unseen_mixed_f708']['nll']:.4f} | "
        f"G5 Soup={la['eval']['weight_soup_05_05']['unseen_g5_half_turn']['nll']:.4f}",
        flush=True,
    )
    print(
        f"  W2 Canonical 4c    | Schur err={wb['schur_canonical_factor_vs_soup_max_err']:.1e} | "
        f"Lie ortho={wb['lie_algebra_cayley_ortho_defect']:.1e} (Euc={wb['euclidean_q_soup_ortho_defect']:.3f}) | "
        f"cos(full/schur)={wb['cos_sim_full_canonical']:+.4f}/{wb['cos_sim_schur_readout']:+.4f} | "
        f"Gamma_add(12/21)={wb['gamma_12_add']:.3f}/{wb['gamma_21_add']:.3f} | "
        f"Gamma_soup(12/21)={wb['gamma_12_soup']:.3f}/{wb['gamma_21_soup']:.3f} | "
        f"UnseenMix LieSoup={wb['eval']['canonical_lie_schur_soup_05_05']['unseen_mixed_f708']['nll']:.4f} | "
        f"G5 LieSoup={wb['eval']['canonical_lie_schur_soup_05_05']['unseen_g5_half_turn']['nll']:.4f}",
        flush=True,
    )

    # Strict machine-verified assertions
    for q_arm in (
        "3a_quotient_cayley_pure",
        "3b_quotient_cayley_hybrid",
        "4a_quotient_dynamic_all4_unconstrained",
        "4b_quotient_harmonic_unconstrained",
        "4c_quotient_cayley_schur_hyper_lora",
    ):
        assert benchmark_arms[q_arm]["step_384_final_seed_mean"]["cells"]["G1_main_extrap_both"]["max_equivariance_defect"] == 0.0
    assert benchmark_arms["3a_quotient_cayley_pure"]["step_384_final_seed_mean"]["cayley_max_ortho_defect"] < 1e-5
    assert benchmark_arms["3a_quotient_cayley_pure"]["step_384_final_seed_mean"]["cayley_max_so2_defect"] < 1e-5
    assert benchmark_arms["4c_quotient_cayley_schur_hyper_lora"]["step_384_final_seed_mean"]["max_period_2pi_defect"] < 1e-6
    assert benchmark_arms["4c_quotient_cayley_schur_hyper_lora"]["step_384_final_seed_mean"]["max_phase_quarter_turn_cov_defect"] < 1e-6

    # Verify exact parameter parity (all main arms <= 3072)
    param_counts = {k: v["trainable_params"] for k, v in benchmark_arms.items()}
    assert max(param_counts.values()) == 3072

    wall_s = time.perf_counter() - t0
    cpu_s = time.process_time() - t_cpu0
    results = {
        "track": "W2_Round2",
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "device": "cpu",
            "wall_seconds": round(wall_s, 3),
            "cpu_process_seconds": round(cpu_s, 3),
        },
        "protocol": {
            "s17_sha256": s17_sha,
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "total_steps": TOTAL_STEPS,
            "pilot_step": PILOT_STEP,
            "batch_size": BATCH_SIZE,
            "base_lr": BASE_LR,
            "seeds": list(MODEL_SEEDS),
            "train_omegas": list(TRAIN_OMEGAS),
            "train_timings": [list(x) for x in TRAIN_TIMINGS],
            "dev_omegas": list(DEV_OMEGAS),
            "trainable_param_budget": param_counts,
        },
        "oracles": {cid: cdata["oracle"] for cid, cdata in cells.items()},
        "certificates": {
            "theorem1_symmetry_breaking": cert_t1,
            "theorem2_and_5_harmonics_and_rank": cert_t25,
            "theorem3_short_horizon_dynamics": cert_t3,
            "theorem4_complex_cayley_depth32": cert_t4,
            "theorem7_complex_nullspace_cayley": cert_t7,
        },
        "surrogate_benchmark": benchmark_arms,
        "minimal_rank_ablation": rank_ablation,
        "round2_nullspace_and_bayes_kl": r2_part_d,
        "round2_canonical_merging": r2_part_e,
    }

    out_path = OUT_DIR / "results_w2.json"
    out_path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"[W2] Completed cleanly in wall={wall_s:.2f}s (cpu={cpu_s:.2f}s) -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
