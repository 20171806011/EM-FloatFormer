"""Track N1 (Round 1) Machine-Verified Numerical Proof Suite.

Exact Algebraic Compilation into SwiGLU Connecting Neurons inside Qwen3.5-4B/9B/27B
Dimensions (d=2560, non-uniform RMSNorm weights gamma, nontrivial prose background
h_prose orthogonal to V_EM, final RMSNorm + lm_head B_align^(gamma_f)), verified
across FP64, FP32, and BF16 on adversarial G1 double-extrapolation shifts and
real locked Stage 3D equivariant_b8_s17 data.

Run strictly via:
  uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B \
      research/stage4_real_and_surgical/neuron_surgery/N1/verify_n1.py
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import nnls
from scipy.special import zeta
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.stage3_ministral_cache.schema import ROOTS, matched_sum, sector_of_complex
from research.stage3d_qwen_signed_phase.data_stage3d import (
    BLIND_CELLS,
    NS_BLIND_Q2_C4,
    build_q2_arm_references,
    condition_spec_from_params,
    generate_paired_cfo_blocks,
    load_s17_encoder,
)
from research.stage4_real_and_surgical.equi_connector import (
    PhysicalKernelS17,
    extract_canonical_ref_summary,
)

OUT_DIR = Path(__file__).resolve().parent
RESULTS_JSON = OUT_DIR / "results_n1.json"

# Qwen3.5-4B grounded dimensions
D_MODEL = 2560
D_FF_4B = 9216
N_LAYERS_4B = 32
RMS_EPS = 1e-6

# Real C4 DFT basis F_4 = [1/2 * 1, F_real] in O(4)
F_REAL_NP = np.array(
    [
        [1.0 / math.sqrt(2.0), 0.0, 0.5],
        [0.0, 1.0 / math.sqrt(2.0), -0.5],
        [-1.0 / math.sqrt(2.0), 0.0, 0.5],
        [0.0, -1.0 / math.sqrt(2.0), -0.5],
    ],
    dtype=np.float64,
)
F_4_NP = np.column_stack([np.full(4, 0.5, dtype=np.float64), F_REAL_NP])


# =============================================================================
# Core Architectural Primitives (Qwen3.5-Faithful RMSNorm + SwiGLU MLP)
# =============================================================================


class Qwen35RMSNorm(nn.Module):
    """Exact Qwen3.5 RMSNorm: (x / sqrt(mean(x^2) + eps)) * (1 + weight_delta) = (x / s(x)) * gamma."""

    def __init__(self, gamma: torch.Tensor, eps: float = RMS_EPS):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("gamma", gamma.clone())

    def rms(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        # Qwen3.5 / standard RMSNorm computes variance in float32 when running BF16,
        # or native dtype in FP64/FP32.
        compute_dtype = torch.float32 if orig_dtype == torch.bfloat16 else orig_dtype
        xf = x.to(compute_dtype)
        s = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        g = self.gamma.to(compute_dtype)
        out = (xf / s) * g
        return out.to(orig_dtype)


class CompiledSwiGLULayer(nn.Module):
    """Exact Pre-RMSNorm + K-neuron SwiGLU surgical sub-block (bias=False, matching Qwen3.5)."""

    def __init__(
        self,
        gamma: torch.Tensor,
        w_gate: torch.Tensor,  # (K, d)
        w_up: torch.Tensor,    # (K, d)
        w_down: torch.Tensor,  # (d, K)
        eps: float = RMS_EPS,
    ):
        super().__init__()
        self.norm = Qwen35RMSNorm(gamma, eps=eps)
        self.register_buffer("w_gate", w_gate.clone())
        self.register_buffer("w_up", w_up.clone())
        self.register_buffer("w_down", w_down.clone())

    @property
    def num_neurons(self) -> int:
        return int(self.w_gate.shape[0])

    def delta(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        wg = self.w_gate.to(dtype=z.dtype, device=z.device)
        wu = self.w_up.to(dtype=z.dtype, device=z.device)
        wd = self.w_down.to(dtype=z.dtype, device=z.device)
        g = F.silu(F.linear(z, wg))
        u = F.linear(z, wu)
        return F.linear(g * u, wd)

    def forward(self, x: torch.Tensor, g_em: int = 1) -> torch.Tensor:
        if g_em == 0:
            return x
        return x + self.delta(x)


def compile_antipodal_pair(
    a_res: torch.Tensor,
    b_res: torch.Tensor,
    w_out: torch.Tensor,
    gamma: torch.Tensor,
    compensate_gamma: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return 2 antipodal SwiGLU neurons computing w_out * (a_res^T x)(b_res^T x) / s(x)^2 exactly.

    With compensate_gamma=True, applies exact diagonal metric preconditioning D_gamma^{-1}.
    """
    inv_g = (1.0 / gamma) if compensate_gamma else torch.ones_like(gamma)
    a_tilde = a_res * inv_g
    b_tilde = b_res * inv_g
    w_gate = torch.stack([a_tilde, -a_tilde], dim=0)
    w_up = torch.stack([b_tilde, -b_tilde], dim=0)
    w_down = torch.stack([w_out, w_out], dim=1)
    return w_gate, w_up, w_down


def compile_complex_mul_neurons(
    e_ar: torch.Tensor,
    e_ai: torch.Tensor,
    e_br: torch.Tensor,
    e_bi: torch.Tensor,
    e_or: torch.Tensor,
    e_oi: torch.Tensor,
    gamma: torch.Tensor,
    conj_a: bool = False,
    conj_b: bool = False,
    scale: float = 1.0,
    method: str = "karatsuba6",
    compensate_gamma: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compile exact complex multiplication out = scale * A^{(*)} * B^{(*)}.

    method='karatsuba6' uses the proved global minimum of 6 SwiGLU neurons (3 antipodal pairs).
    method='direct8' uses the isotropic 8 SwiGLU neurons (4 antipodal pairs).
    """
    sa = -1.0 if conj_a else 1.0
    sb = -1.0 if conj_b else 1.0
    # Effective (ar + i*sa*ai) * (br + i*sb*bi):
    # Re = ar*br - (sa*sb)*ai*bi
    # Im = sa*ai*br + sb*ar*bi
    ai_eff = sa * e_ai
    bi_eff = sb * e_bi
    # Now exact product is (e_ar + i*ai_eff) * (e_br + i*bi_eff):
    # Re = ar*br - ai*bi
    # Im = ar*bi + ai*br
    pairs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    if method == "karatsuba6":
        # m1 = ar * br; m2 = ai * bi; m3 = (ar + ai) * (br + bi)
        # Re = m1 - m2; Im = m3 - m1 - m2
        pairs.append((e_br, e_ar, scale * (e_or - e_oi)))
        pairs.append((bi_eff, ai_eff, scale * (-e_or - e_oi)))
        pairs.append((e_br + bi_eff, e_ar + ai_eff, scale * e_oi))
    elif method == "direct8":
        pairs.append((e_br, e_ar, scale * e_or))
        pairs.append((bi_eff, ai_eff, -scale * e_or))
        pairs.append((bi_eff, e_ar, scale * e_oi))
        pairs.append((e_br, ai_eff, scale * e_oi))
    else:
        raise ValueError(f"Unknown method={method}")

    wg_list, wu_list, wd_list = [], [], []
    for a_v, b_v, w_v in pairs:
        wg, wu, wd = compile_antipodal_pair(a_v, b_v, w_v, gamma, compensate_gamma=compensate_gamma)
        wg_list.append(wg)
        wu_list.append(wu)
        wd_list.append(wd)
    return torch.cat(wg_list, dim=0), torch.cat(wu_list, dim=0), torch.cat(wd_list, dim=1)


def fermi_dirac_const(q: float) -> float:
    """Exact Mellin normalization C_q = int_0^inf u^q / (1 + e^u) du = (1 - 2^-q) Gamma(q+1) zeta(q+1)."""
    if q == 0.0:
        return math.log(2.0)
    return (1.0 - 2.0 ** (-q)) * math.gamma(q + 1.0) * float(zeta(q + 1.0))


def build_fermi_dirac_nodes(
    q: float,
    m_nodes: int,
    eta_min: float,
    eta_max: float,
    mode: str = "nnls_pos",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Construct strictly non-negative Fermi-Dirac SwiGLU nodes (t_m, w_m) approximating eta^{-q}.

    Satisfies: sum_m w_m * (-SiLU(-t_m * eta)) ~= eta^{-q} on [eta_min, eta_max] with w_m >= 0.
    mode='trapezoid': closed-form analytic Euler-Maclaurin trapezoidal rule on log-time tau = ln t.
    mode='nnls_pos': strictly non-negative least-squares weights on log-spaced t_m (zero sign cancellation).
    """
    etas = np.geomspace(eta_min, eta_max, 256)
    target = etas ** (-q)

    if mode == "trapezoid":
        # Center log-grid around peak u* ~ q + 1.2 for both eta_min and eta_max
        c_q = fermi_dirac_const(q)
        u_peak = max(0.8, q + 1.1)
        tau_lo_c = math.log(u_peak / eta_max) - (4.5 if q == 0 else 3.2)
        tau_hi_c = math.log(u_peak / eta_min) + 2.4
        best = (1e9, None, None)
        for d_lo in np.linspace(-1.5, 1.5, 13):
            for d_hi in np.linspace(-1.0, 1.5, 11):
                taus = np.linspace(tau_lo_c + d_lo, tau_hi_c + d_hi, m_nodes)
                h = taus[1] - taus[0]
                ts = np.exp(taus)
                ws = (h / c_q) * (ts ** q)
                u = etas[:, None] * ts[None, :]
                phi = u / (1.0 + np.exp(np.clip(u, -80.0, 80.0)))
                rel = float(np.max(np.abs(phi @ ws - target) / target))
                if rel < best[0]:
                    best = (rel, ts, ws)
        return best[1], best[2], best[0]

    if mode == "nnls_pos":
        u_lo = 0.015 if q == 0 else (0.08 if q == 1 else 0.18)
        u_hi = 18.0 if q <= 1 else (26.0 if q == 2 else 40.0)
        ts = np.geomspace(u_lo / eta_max, u_hi / eta_min, m_nodes)
        u = etas[:, None] * ts[None, :]
        phi = u / (1.0 + np.exp(np.clip(u, -80.0, 80.0)))
        ws, _ = nnls(phi / target[:, None], np.ones_like(etas))
        rel = float(np.max(np.abs(phi @ ws - target) / target))
        return ts, ws, rel

    raise ValueError(f"Unknown mode={mode}")


def compile_fermi_dirac_operator(
    e_eta: torch.Tensor,
    e_sig: torch.Tensor,
    w_out: torch.Tensor,
    gamma: torch.Tensor,
    ts: np.ndarray,
    ws: np.ndarray,
    scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compile M SwiGLU neurons computing F_q(eta, sig) = sum_m w_m * SiLU(-t_m * eta) * (-sig) * scale * w_out."""
    inv_g = 1.0 / gamma
    eta_tilde = e_eta * inv_g
    sig_tilde = e_sig * inv_g
    wg_list, wu_list, wd_list = [], [], []
    for t_val, w_val in zip(ts, ws):
        if float(w_val) == 0.0:
            continue
        wg_list.append(-float(t_val) * eta_tilde)
        wu_list.append(-sig_tilde)
        wd_list.append((scale * float(w_val)) * w_out)
    return torch.stack(wg_list, dim=0), torch.stack(wu_list, dim=0), torch.stack(wd_list, dim=1)


def stack_blocks(blocks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.cat([b[0] for b in blocks], dim=0),
        torch.cat([b[1] for b in blocks], dim=0),
        torch.cat([b[2] for b in blocks], dim=1),
    )


def make_orthogonal_host_geometry(
    d: int = D_MODEL,
    k_prose: int = 512,
    n_slots: int = 32,
    seed: int = 20261101,
) -> dict:
    """Build deterministic Qwen3.5-4B (d=2560) geometry with exact 3-way orthogonal decomposition:
        R^d = V_cand^(gamma_f) (3D) (+) V_prose (k_prose) (+) V_EM (n_slots) (+) V_free.
    """
    rng = np.random.default_rng(seed)

    # Non-uniform realistic RMSNorm weights (Qwen3.5 uses 1 + weight, spanning ~[0.45, 2.40])
    gammas = []
    for _ in range(4):
        g = np.exp(rng.normal(loc=0.05, scale=0.35, size=d))
        g = np.clip(g, 0.45, 2.50)
        gammas.append(torch.from_numpy(g).to(torch.float64))
    gamma_l1, gamma_l2, gamma_l3, gamma_final = gammas

    # Realistic frozen lm_head digit unembeddings W_digit (4, d)
    w_digit = torch.from_numpy(rng.normal(scale=0.05, size=(4, d))).to(torch.float64)
    m_cand = w_digit - w_digit.mean(dim=0, keepdim=True)  # (4, d), rank 3

    # Exact gamma_final-compensated canonical readout frame B_align^(gamma_f) = (M_cand D_{gamma_f})^+ F_real
    a_eff = m_cand * gamma_final[None, :]  # (4, d)
    f_real_t = torch.from_numpy(F_REAL_NP)
    b_align_gamma = torch.linalg.pinv(a_eff) @ f_real_t  # (d, 3)
    # Naive (uncompensated) B_align = M_cand^+ F_real (as in surgical_adapter.py)
    b_align_naive = torch.linalg.pinv(m_cand) @ f_real_t  # (d, 3)

    # Orthonormal 3D basis U_cand for row(M_cand D_{gamma_f}) = span(B_align^(gamma_f))
    u_cand_np, _ = np.linalg.qr(a_eff.T.numpy(), mode="reduced")
    u_cand = torch.from_numpy(u_cand_np[:, :3].copy()).to(torch.float64)  # (d, 3)

    # Build V_prose (k_prose) and V_EM (n_slots) strictly orthogonal to V_cand^(gamma_f) and to each other!
    raw = rng.standard_normal((d, k_prose + n_slots))
    # Project out U_cand first, then QR
    raw = raw - u_cand_np[:, :3] @ (u_cand_np[:, :3].T @ raw)
    q_mat, _ = np.linalg.qr(raw, mode="reduced")
    u_prose = torch.from_numpy(q_mat[:, :k_prose].copy()).to(torch.float64)   # (d, k_prose)
    u_em = torch.from_numpy(q_mat[:, k_prose:].copy()).to(torch.float64)      # (d, n_slots)

    # Verify strict 3-way orthogonality
    orth_prose_em = float((u_prose.T @ u_em).abs().max().item())
    orth_cand_em = float((u_cand.T @ u_em).abs().max().item())
    orth_cand_prose = float((u_cand.T @ u_prose).abs().max().item())
    assert orth_prose_em < 1e-13 and orth_cand_em < 1e-13 and orth_cand_prose < 1e-13

    # Verify exact readout identity: M_cand D_{gamma_f} B_align^(gamma_f) == F_real
    readout_id_err = float((a_eff @ b_align_gamma - f_real_t).abs().max().item())
    naive_readout_err = float((a_eff @ b_align_naive - f_real_t).abs().max().item())
    assert readout_id_err < 1e-13

    return {
        "u_cand": u_cand,
        "u_prose": u_prose,
        "u_em": u_em,
        "gamma_l1": gamma_l1,
        "gamma_l2": gamma_l2,
        "gamma_l3": gamma_l3,
        "gamma_final": gamma_final,
        "w_digit": w_digit,
        "m_cand": m_cand,
        "b_align_gamma": b_align_gamma,
        "b_align_naive": b_align_naive,
        "subspace_orth_err": max(orth_prose_em, orth_cand_em, orth_cand_prose),
        "b_align_gamma_err": readout_id_err,
        "b_align_naive_err": naive_readout_err,
    }


def sample_prose_batch(
    u_prose: torch.Tensor,
    target_rms: np.ndarray,
    rng: np.random.Generator,
) -> torch.Tensor:
    """Sample (B, d) prose vectors strictly inside span(u_prose) with exact specified per-sample RMS."""
    b, k = len(target_rms), u_prose.shape[1]
    coeffs = rng.standard_normal((b, k))
    coeffs /= np.linalg.norm(coeffs, axis=1, keepdims=True)
    # Target ||h_prose||_2 = sqrt(d) * target_rms
    scales = math.sqrt(float(u_prose.shape[0])) * target_rms[:, None]
    c_t = torch.from_numpy(coeffs * scales).to(torch.float64)
    return c_t @ u_prose.T


# =============================================================================
# Certificate 1: Antipodal Lower Bound, Karatsuba-6 vs Direct-8, Single-Layer Trilinear No-Go
# =============================================================================


def run_cert1_minimal_neurons(geom: dict) -> dict:
    """Verify exact pencil-rank lower bounds, Karatsuba-6 vs Direct-8, and single-layer trilinear no-go."""
    # 1A. Pencil rank lower bound for complex multiplication (K_min = 6) and centered C4 conv (K_min = 8)
    H1 = 0.5 * np.array(
        [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64
    )
    H2 = 0.5 * np.array(
        [[0, 0, 0, -1], [0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0]], dtype=np.float64
    )
    angles = np.linspace(0.0, 2.0 * math.pi, 720, endpoint=False)
    min_det_pencil = 1e9
    min_sv4_pencil = 1e9
    max_det_formula_err = 0.0
    for phi in angles:
        ca, sa = math.cos(phi), math.sin(phi)
        H = ca * H1 + sa * H2
        det_h = float(np.linalg.det(H))
        min_det_pencil = min(min_det_pencil, det_h)
        max_det_formula_err = max(max_det_formula_err, abs(det_h - 1.0 / 16.0))
        svs = np.linalg.svd(H, compute_uv=False)
        min_sv4_pencil = min(min_sv4_pencil, float(svs[-1]))
    assert min_det_pencil > 0.06 and min_sv4_pencil > 0.49

    # Centered C4 pencil on R^6 = R^4 (V_1) (+) R^2 (V_2):
    H_c4_1r = np.zeros((6, 6), dtype=np.float64); H_c4_1r[:4, :4] = H1
    H_c4_1i = np.zeros((6, 6), dtype=np.float64); H_c4_1i[:4, :4] = H2
    H_c4_2 = np.zeros((6, 6), dtype=np.float64); H_c4_2[4, 5] = H_c4_2[5, 4] = 0.5
    rng = np.random.default_rng(1001)
    rank_le2_only_v2 = True
    for _ in range(500):
        a, b, g = rng.standard_normal(3)
        M = a * H_c4_1r + b * H_c4_1i + g * H_c4_2
        r = int(np.linalg.matrix_rank(M, tol=1e-10))
        if math.hypot(a, b) > 1e-8 and r <= 2:
            rank_le2_only_v2 = False
    assert rank_le2_only_v2

    # 1B. Compile Karatsuba-6 vs Direct-8 at d=2560 with non-uniform gamma + nontrivial prose!
    u_em = geom["u_em"]
    gamma = geom["gamma_l1"]
    e_u2, e_v2, e_ch, e_sh, e_out_r, e_out_i = [u_em[:, i] for i in range(6)]

    # Test across extreme adversarial angles omega*h in [-20*pi, 20*pi] + Stage 3D G1/G5 angles
    n_test = 512
    u2_np = rng.normal(scale=1.2, size=n_test)
    v2_np = rng.normal(scale=1.2, size=n_test)
    wh_np = np.concatenate(
        [
            np.array([-20 * math.pi, -10 * math.pi, -math.pi, -1.60, 0.0, 1.60, math.pi, 10 * math.pi, 20 * math.pi]),
            rng.uniform(-20.0 * math.pi, 20.0 * math.pi, size=n_test - 9),
        ]
    )
    ch_np, sh_np = np.cos(wh_np), np.sin(wh_np)
    # Exact T_2' = T_2 * exp(-i*wh) = (u2*ch + v2*sh) + i*(v2*ch - u2*sh)
    exact_r = u2_np * ch_np + v2_np * sh_np
    exact_i = v2_np * ch_np - u2_np * sh_np

    prose_rms = rng.uniform(0.6, 2.5, size=n_test)
    h_prose = sample_prose_batch(geom["u_prose"], prose_rms, rng)
    x_em = (
        torch.from_numpy(u2_np)[:, None] * e_u2[None, :]
        + torch.from_numpy(v2_np)[:, None] * e_v2[None, :]
        + torch.from_numpy(ch_np)[:, None] * e_ch[None, :]
        + torch.from_numpy(sh_np)[:, None] * e_sh[None, :]
    )
    x_full = h_prose + x_em
    s_exact = torch.sqrt(x_full.pow(2).mean(dim=-1) + RMS_EPS).numpy()

    out_stats = {}
    for method in ("karatsuba6", "direct8"):
        wg, wu, wd = compile_complex_mul_neurons(
            e_u2, e_v2, e_ch, e_sh, e_out_r, e_out_i, gamma,
            conj_a=False, conj_b=True, scale=1.0, method=method, compensate_gamma=True,
        )
        layer = CompiledSwiGLULayer(gamma, wg, wu, wd)
        assert layer.num_neurons == (6 if method == "karatsuba6" else 8)
        m_res = {}
        for dt_name, dt in (("fp64", torch.float64), ("fp32", torch.float32), ("bf16", torch.bfloat16)):
            delta = layer.delta(x_full.to(dt)).to(torch.float64)
            # Extract output slots and multiply by s^2 to compare against exact T_2'
            got_r = (delta @ e_out_r).numpy() * (s_exact ** 2)
            got_i = (delta @ e_out_i).numpy() * (s_exact ** 2)
            err = float(max(np.max(np.abs(got_r - exact_r)), np.max(np.abs(got_i - exact_i))))
            rms_err = float(math.sqrt(0.5 * (np.mean((got_r - exact_r) ** 2) + np.mean((got_i - exact_i) ** 2))))
            # Pure prose check (h_EM = 0):
            prose_delta = layer.delta(h_prose.to(dt)).to(torch.float64)
            prose_leak_coord = float(prose_delta.abs().max().item())
            prose_leak_norm = float(prose_delta.norm(dim=-1).max().item())
            m_res[dt_name] = {
                "max_abs_err_unscaled": err,
                "rms_err_unscaled": rms_err,
                "pure_prose_max_delta": prose_leak_coord,
                "pure_prose_max_l2_norm": prose_leak_norm,
            }
        out_stats[method] = m_res

    # 1C. Gamma-compensation ablation (proving O(sqrt(d)) prose crosstalk without D_gamma^{-1}!)
    wg_bad, wu_bad, wd_bad = compile_complex_mul_neurons(
        e_u2, e_v2, e_ch, e_sh, e_out_r, e_out_i, gamma,
        conj_a=False, conj_b=True, scale=1.0, method="karatsuba6", compensate_gamma=False,
    )
    layer_bad = CompiledSwiGLULayer(gamma, wg_bad, wu_bad, wd_bad)
    delta_bad_prose_t = layer_bad.delta(h_prose)
    delta_bad_prose_l2 = float(delta_bad_prose_t.norm(dim=-1).max().item())
    delta_bad_prose_coord = float(delta_bad_prose_t.abs().max().item())
    delta_bad_full = layer_bad.delta(x_full)
    bad_r = (delta_bad_full @ e_out_r).numpy() * (s_exact ** 2)
    bad_err = float(np.max(np.abs(bad_r - exact_r)))

    # 1D. Single-layer trilinear impossibility check:
    # Even-symmetric pair SiLU(a)b + SiLU(-a)b = (a/2)tanh(a/2) * 2b = a^2 b / 2 - a^4 b / 24 + ...
    # Show error scales O(r^5) locally and diverges globally as amplitude grows!
    radii = [0.05, 0.2, 1.0, 3.0, 8.0]
    trilin_1l_rel_errs = {}
    for rad in radii:
        a = rad
        b = rad
        # Exact target: (1/2) * a^2 * b
        tgt = 0.5 * (a ** 2) * b
        got = (F.silu(torch.tensor(a, dtype=torch.float64)) + F.silu(torch.tensor(-a, dtype=torch.float64))).item() * b
        trilin_1l_rel_errs[f"r={rad}"] = float(abs(got - tgt) / abs(tgt))

    assert out_stats["karatsuba6"]["fp64"]["max_abs_err_unscaled"] < 1e-12
    assert out_stats["direct8"]["fp64"]["max_abs_err_unscaled"] < 1e-12
    assert out_stats["karatsuba6"]["fp64"]["pure_prose_max_l2_norm"] < 1e-13
    assert out_stats["direct8"]["fp64"]["pure_prose_max_l2_norm"] < 1e-13
    assert delta_bad_prose_l2 > 0.5  # confirms massive O(1)-O(sqrt(d)) uncompensated prose crosstalk!

    return {
        "pencil_min_det": min_det_pencil,
        "pencil_det_formula_err": max_det_formula_err,
        "pencil_min_sv4": min_sv4_pencil,
        "c4_rank_le2_only_v2": rank_le2_only_v2,
        "karatsuba6_vs_direct8": out_stats,
        "uncompensated_gamma_prose_leak_l2": delta_bad_prose_l2,
        "uncompensated_gamma_prose_leak_coord": delta_bad_prose_coord,
        "uncompensated_gamma_output_err": bad_err,
        "single_layer_trilinear_rel_err_by_radius": trilin_1l_rel_errs,
    }


# =============================================================================
# Certificate 2: 2-Layer Trilinear Cross-Correlation & Exact C4 Group Convolution (Q1, Q2, Q3)
# =============================================================================


def run_cert2_multilayer_algebra(geom: dict) -> dict:
    """Verify exact 2-layer Re/Im(T1^* T2 e^{-i*wh}) and exact C4 convolutions for Q1, Q2, Q3."""
    rng = np.random.default_rng(2002)
    u_em = geom["u_em"]
    g1, g2 = geom["gamma_l1"], geom["gamma_l2"]
    n_test = 512

    # Slot allocation inside V_EM (all mutually orthonormal & orthogonal to V_prose):
    # [0,1]: T1=(u1,v1); [2,3]: T2=(u2,v2); [4,5]: (cos wh, sin wh); [6,7]: scratch T2'=(u2',v2')
    # [8,9]: output Z = T1^* T2' = (Re, Im)
    e = [u_em[:, i] for i in range(20)]

    # --- 2A. Exact 2-layer Re & Im of T1^* T2 e^{-i*wh} (12 Karatsuba vs 16 Direct) ---
    u1 = rng.normal(scale=1.1, size=n_test); v1 = rng.normal(scale=1.1, size=n_test)
    u2 = rng.normal(scale=1.1, size=n_test); v2 = rng.normal(scale=1.1, size=n_test)
    wh = rng.uniform(-15.0 * math.pi, 15.0 * math.pi, size=n_test)
    ch, sh = np.cos(wh), np.sin(wh)
    z_exact = np.conj(u1 + 1j * v1) * (u2 + 1j * v2) * np.exp(-1j * wh)

    h_prose1 = sample_prose_batch(geom["u_prose"], rng.uniform(0.6, 2.4, size=n_test), rng)
    h_prose_delta = sample_prose_batch(geom["u_prose"], rng.uniform(0.3, 1.2, size=n_test), rng)
    x1 = (
        h_prose1
        + torch.from_numpy(u1)[:, None] * e[0][None, :]
        + torch.from_numpy(v1)[:, None] * e[1][None, :]
        + torch.from_numpy(u2)[:, None] * e[2][None, :]
        + torch.from_numpy(v2)[:, None] * e[3][None, :]
        + torch.from_numpy(ch)[:, None] * e[4][None, :]
        + torch.from_numpy(sh)[:, None] * e[5][None, :]
    )

    xcorr_res = {}
    for method in ("karatsuba6", "direct8"):
        # Layer 1: T2' = T2 * exp(-i*wh) -> slots (6, 7)
        wg1, wu1, wd1 = compile_complex_mul_neurons(
            e[2], e[3], e[4], e[5], e[6], e[7], g1, conj_a=False, conj_b=True, method=method
        )
        # Layer 2: Z = T1^* * T2' -> slots (8, 9)
        wg2, wu2, wd2 = compile_complex_mul_neurons(
            e[0], e[1], e[6], e[7], e[8], e[9], g2, conj_a=True, conj_b=False, method=method
        )
        l1 = CompiledSwiGLULayer(g1, wg1, wu1, wd1)
        l2 = CompiledSwiGLULayer(g2, wg2, wu2, wd2)
        m_out = {"total_neurons": l1.num_neurons + l2.num_neurons}
        for dt_name, dt in (("fp64", torch.float64), ("fp32", torch.float32), ("bf16", torch.bfloat16)):
            xt = x1.to(dt)
            s1 = torch.sqrt(xt.to(torch.float64).pow(2).mean(dim=-1) + RMS_EPS).numpy()
            x2_in = l1(xt) + h_prose_delta.to(dt)  # simulate intervening attention/prose residual drift!
            s2 = torch.sqrt(x2_in.to(torch.float64).pow(2).mean(dim=-1) + RMS_EPS).numpy()
            d2 = l2.delta(x2_in).to(torch.float64)
            got_re = (d2 @ e[8]).numpy() * (s1 ** 2) * (s2 ** 2)
            got_im = (d2 @ e[9]).numpy() * (s1 ** 2) * (s2 ** 2)
            err = float(max(np.max(np.abs(got_re - z_exact.real)), np.max(np.abs(got_im - z_exact.imag))))
            # Ratio / argmax invariance across C4 scores (Re, Im, -Re, -Im) WITHOUT unscaling s1, s2!
            raw_scores = np.stack(
                [(d2 @ e[8]).numpy(), (d2 @ e[9]).numpy(), -(d2 @ e[8]).numpy(), -(d2 @ e[9]).numpy()],
                axis=-1,
            )
            exact_scores = np.stack([z_exact.real, z_exact.imag, -z_exact.real, -z_exact.imag], axis=-1)
            argmax_agree = float(np.mean(raw_scores.argmax(-1) == exact_scores.argmax(-1)))
            m_out[dt_name] = {"max_abs_err_unscaled": err, "raw_argmax_agreement": argmax_agree}
        xcorr_res[method] = m_out

    assert xcorr_res["karatsuba6"]["fp64"]["max_abs_err_unscaled"] < 1e-11
    assert xcorr_res["direct8"]["fp64"]["max_abs_err_unscaled"] < 1e-11
    assert xcorr_res["karatsuba6"]["fp64"]["raw_argmax_agreement"] == 1.0

    # --- 2B. Exact C4 Group Convolution for Q2 (1 layer, 8 neurons) & Q1 (2 layers, 16 neurons) & Q3 (2 layers self-harmonic) ---
    # Generate random valid C4 probability distributions a1, a2, bp, w_quad
    def rand_probs(n: int) -> np.ndarray:
        raw = np.exp(rng.normal(scale=1.3, size=(n, 4)))
        return raw / raw.sum(axis=-1, keepdims=True)

    a1_p = rand_probs(n_test)
    a2_p = rand_probs(n_test)
    bp_p = rand_probs(n_test)
    wq_p = rand_probs(n_test)

    # Exact time-domain C4 cyclic correlations:
    # Q2: p_q2[c] = sum_u bp[u] * a1[(u+c)%4]
    p_q2_exact = np.stack([np.sum(bp_p * np.roll(a1_p, -c, axis=-1), axis=-1) for c in range(4)], axis=-1)
    # Q1: a2_rot[u] = sum_k wq[k] * a2[(u+k)%4]; p_q1[c] = sum_u a1[u] * a2_rot[(u+c)%4]
    a2_rot_exact = np.stack([sum(wq_p[:, k] * np.roll(a2_p, -k, axis=-1)[:, u] for k in range(4)) for u in range(4)], axis=-1)
    p_q1_exact = np.stack([np.sum(a1_p * np.roll(a2_rot_exact, -c, axis=-1), axis=-1) for c in range(4)], axis=-1)

    # Fourier coords: F_real^T p -> (v1r, v1i, v2)
    a1_f = a1_p @ F_REAL_NP
    a2_f = a2_p @ F_REAL_NP
    bp_f = bp_p @ F_REAL_NP
    wq_f = wq_p @ F_REAL_NP

    # Verify Q2 in 1 layer (8 neurons: 6 Karatsuba on V1 with scale sqrt(2) + 2 antipodal on V2 with scale 2.0)
    # Slots: [0,1,2]=bp_f, [3,4,5]=a1_f, [6,7,8]=out_q2_f
    b_v1 = compile_complex_mul_neurons(
        e[0], e[1], e[3], e[4], e[6], e[7], g1, conj_a=True, conj_b=False, scale=math.sqrt(2.0), method="karatsuba6"
    )
    b_v2 = compile_antipodal_pair(e[2], e[5], 2.0 * e[8], g1)
    wg_q2, wu_q2, wd_q2 = stack_blocks([b_v1, b_v2])
    l_q2 = CompiledSwiGLULayer(g1, wg_q2, wu_q2, wd_q2)
    assert l_q2.num_neurons == 8

    x_q2 = h_prose1.clone()
    for j in range(3):
        x_q2 = x_q2 + torch.from_numpy(bp_f[:, j])[:, None] * e[j][None, :]
        x_q2 = x_q2 + torch.from_numpy(a1_f[:, j])[:, None] * e[3 + j][None, :]
    s_q2 = torch.sqrt(x_q2.pow(2).mean(dim=-1) + RMS_EPS).numpy()
    q2_errs = {}
    for dt_name, dt in (("fp64", torch.float64), ("fp32", torch.float32), ("bf16", torch.bfloat16)):
        d_q2 = l_q2.delta(x_q2.to(dt)).to(torch.float64)
        hat_out = np.stack([(d_q2 @ e[6 + j]).numpy() * (s_q2 ** 2) for j in range(3)], axis=-1)
        p_rec = 0.25 + hat_out @ F_REAL_NP.T
        q2_errs[dt_name] = float(np.max(np.abs(p_rec - p_q2_exact)))
    assert q2_errs["fp64"] < 1e-13

    # Verify Q1 in 2 layers (8 + 8 = 16 neurons)
    # Slots: [0,1,2]=a1_f, [3,4,5]=a2_f, [6,7,8]=wq_f, [9,10,11]=a2_rot_f, [12,13,14]=out_q1_f
    # Note: a2_rot[u] = sum_k wq[k] a2[(u+k)%4] => DFT: hat(a2_rot)_1 = sqrt(2) * wq_1^* * a2_1, hat(a2_rot)_2 = 2 * wq_2 * a2_2
    q1_l1_v1 = compile_complex_mul_neurons(
        e[6], e[7], e[3], e[4], e[9], e[10], g1, conj_a=True, conj_b=False, scale=math.sqrt(2.0), method="karatsuba6"
    )
    q1_l1_v2 = compile_antipodal_pair(e[8], e[5], 2.0 * e[11], g1)
    l_q1_1 = CompiledSwiGLULayer(g1, *stack_blocks([q1_l1_v1, q1_l1_v2]))

    q1_l2_v1 = compile_complex_mul_neurons(
        e[0], e[1], e[9], e[10], e[12], e[13], g2, conj_a=True, conj_b=False, scale=math.sqrt(2.0), method="karatsuba6"
    )
    q1_l2_v2 = compile_antipodal_pair(e[2], e[11], 2.0 * e[14], g2)
    l_q1_2 = CompiledSwiGLULayer(g2, *stack_blocks([q1_l2_v1, q1_l2_v2]))
    assert l_q1_1.num_neurons == 8 and l_q1_2.num_neurons == 8

    x_q1 = h_prose1.clone()
    for j in range(3):
        x_q1 = x_q1 + torch.from_numpy(a1_f[:, j])[:, None] * e[j][None, :]
        x_q1 = x_q1 + torch.from_numpy(a2_f[:, j])[:, None] * e[3 + j][None, :]
        x_q1 = x_q1 + torch.from_numpy(wq_f[:, j])[:, None] * e[6 + j][None, :]
    s1_q1 = torch.sqrt(x_q1.pow(2).mean(dim=-1) + RMS_EPS).numpy()
    q1_errs = {}
    for dt_name, dt in (("fp64", torch.float64), ("fp32", torch.float32), ("bf16", torch.bfloat16)):
        x_mid = l_q1_1(x_q1.to(dt)) + h_prose_delta.to(dt)
        s2_q1 = torch.sqrt(x_mid.to(torch.float64).pow(2).mean(dim=-1) + RMS_EPS).numpy()
        d2 = l_q1_2.delta(x_mid).to(torch.float64)
        hat_out = np.stack([(d2 @ e[12 + j]).numpy() * (s1_q1 ** 2) * (s2_q1 ** 2) for j in range(3)], axis=-1)
        p_rec = 0.25 + hat_out @ F_REAL_NP.T
        q1_errs[dt_name] = float(np.max(np.abs(p_rec - p_q1_exact)))
    assert q1_errs["fp64"] < 1e-12

    # --- 2C. Exact 2-layer Q3_repeat_x1 generating cos(2*theta) from ONLY (cos theta, sin theta) + Constant-Anchor Homogenizer! ---
    # Layer 1:
    #   - V1 output: p_q3_v1 = (4*b1*e^{-V/2}*e^{i*theta}) * hat(a1)_1  (6 neurons)  -> scale s1^{-2}
    #   - cos(2*theta) = (cos th + sin th)(cos th - sin th)              (2 neurons)  -> scale s1^{-2}
    # Layer 2:
    #   - Homogenized V1 pass-through: p_q3_v1 * e_anc                   (4 neurons: 2+2) -> scale s1^{-2} s2^{-2}
    #   - V2 output: (8*b2*e^{-2V}*hat(a1)_2) * cos(2*theta)             (2 neurons)      -> scale s1^{-2} s2^{-2}
    # Total = 8 + 6 = 14 neurons, BOTH V1 and V2 share IDENTICAL s1^{-2} s2^{-2} scaling!
    th = rng.uniform(-12.0 * math.pi, 12.0 * math.pi, size=n_test)
    V_var = rng.uniform(0.01, 0.25, size=n_test)
    b1_c, b2_c = 0.18093, 0.07932  # exact 2 dB AWGN Fourier coeffs (W2 Thm 5A)
    c1_r = 4.0 * b1_c * np.exp(-0.5 * V_var) * np.cos(th)
    c1_i = 4.0 * b1_c * np.exp(-0.5 * V_var) * np.sin(th)
    c2_amp = 8.0 * b2_c * np.exp(-2.0 * V_var)
    # Exact n<=2 Q3 Fourier target:
    q3_v1_exact = (c1_r + 1j * c1_i) * (a1_f[:, 0] + 1j * a1_f[:, 1])
    q3_v2_exact = c2_amp * np.cos(2.0 * th) * a1_f[:, 2]
    q3_hat_exact = np.stack([q3_v1_exact.real, q3_v1_exact.imag, q3_v2_exact], axis=-1)

    # Slots: [0,1,2]=a1_f (with slot 2 carrying c2_amp*a1_f[:,2]), [3,4]=(c1_r, c1_i), [5,6]=(cos th, sin th),
    # [7]=e_anc (=1.0), [8,9]=scratch V1_L1, [10]=scratch cos2th_L1, [11,12,13]=out V1r,V1i,V2
    q3_l1_blocks = [
        compile_complex_mul_neurons(e[3], e[4], e[0], e[1], e[8], e[9], g1, method="karatsuba6"),
        compile_antipodal_pair(e[5] + e[6], e[5] - e[6], e[10], g1),  # cos(2th) = (c+s)(c-s)
    ]
    l_q3_1 = CompiledSwiGLULayer(g1, *stack_blocks(q3_l1_blocks))
    q3_l2_blocks = [
        compile_antipodal_pair(e[7], e[8], e[11], g2),   # homogenize V1_r by anchor c0=1
        compile_antipodal_pair(e[7], e[9], e[12], g2),   # homogenize V1_i by anchor c0=1
        compile_antipodal_pair(e[2], e[10], e[13], g2),  # V2 = (c2_amp * a1_2) * cos(2th)
    ]
    l_q3_2 = CompiledSwiGLULayer(g2, *stack_blocks(q3_l2_blocks))
    assert l_q3_1.num_neurons == 8 and l_q3_2.num_neurons == 6

    x_q3 = (
        h_prose1
        + torch.from_numpy(a1_f[:, 0])[:, None] * e[0][None, :]
        + torch.from_numpy(a1_f[:, 1])[:, None] * e[1][None, :]
        + torch.from_numpy(c2_amp * a1_f[:, 2])[:, None] * e[2][None, :]
        + torch.from_numpy(c1_r)[:, None] * e[3][None, :]
        + torch.from_numpy(c1_i)[:, None] * e[4][None, :]
        + torch.from_numpy(np.cos(th))[:, None] * e[5][None, :]
        + torch.from_numpy(np.sin(th))[:, None] * e[6][None, :]
        + 1.0 * e[7][None, :]
    )
    # Test ratio invariance across 100x prose RMS sweep s in [0.1, 10.0]!
    s_sweep = np.geomspace(0.1, 10.0, n_test)
    h_prose_sweep = sample_prose_batch(geom["u_prose"], s_sweep, rng)
    x_q3_sweep = x_q3 - h_prose1 + h_prose_sweep
    x_mid_q3 = l_q3_1(x_q3_sweep) + h_prose_delta
    d2_q3 = l_q3_2.delta(x_mid_q3)
    got_q3_raw = np.stack([(d2_q3 @ e[11 + j]).numpy() for j in range(3)], axis=-1)
    # Normalize each sample by its L2 norm to verify exact ratio / direction invariance across 100x s sweep!
    dir_got = got_q3_raw / np.linalg.norm(got_q3_raw, axis=-1, keepdims=True)
    dir_ex = q3_hat_exact / np.linalg.norm(q3_hat_exact, axis=-1, keepdims=True)
    hom_dir_err_100x = float(np.max(np.abs(dir_got - dir_ex)))
    # Compare against NON-homogenized (mixing V1 from L1 [s1^{-2}] with V2 from L2 [s1^{-2} s2^{-2}]):
    v1_unhom = np.stack([(x_mid_q3 @ e[8]).numpy(), (x_mid_q3 @ e[9]).numpy()], axis=-1)
    mixed_raw = np.column_stack([v1_unhom, (d2_q3 @ e[13]).numpy()])
    dir_mixed = mixed_raw / np.linalg.norm(mixed_raw, axis=-1, keepdims=True)
    unhom_dir_err_100x = float(np.max(np.abs(dir_mixed - dir_ex)))
    assert hom_dir_err_100x < 1e-9 and unhom_dir_err_100x > 0.25

    return {
        "xcorr_2layer": xcorr_res,
        "q2_1layer_8n_max_prob_err": q2_errs,
        "q1_2layer_16n_max_prob_err": q1_errs,
        "q3_self_harmonic_14n_100x_norm_sweep": {
            "homogenized_max_direction_err": hom_dir_err_100x,
            "unhomogenized_degree_mix_err": unhom_dir_err_100x,
        },
    }


# =============================================================================
# Certificate 3: Pre-MLP RMSNorm Obstruction & Self-Normalizing Fermi-Dirac SwiGLU Gadget
# =============================================================================


def run_cert3_rmsnorm_self_normalizing(geom: dict) -> dict:
    """Prove & verify exact Fermi-Dirac Mellin s(x)-compensation across independent layer norms & final RMSNorm."""
    rng = np.random.default_rng(3003)
    u_em = geom["u_em"]
    g1, g2, g3, gf = geom["gamma_l1"], geom["gamma_l2"], geom["gamma_l3"], geom["gamma_final"]
    m_cand = geom["m_cand"]
    b_align = geom["b_align_gamma"]
    final_norm = Qwen35RMSNorm(gf)
    e = [u_em[:, i] for i in range(24)]

    # --- 3A. Closed-form Trapezoidal vs Positive NNLS Fermi-Dirac quadrature convergence table ---
    fd_table = {}
    for q in (0.0, 1.0, 2.0):
        q_res = {}
        for mode in ("trapezoid", "nnls_pos"):
            m_dict = {}
            for m_n in (6, 8, 10, 12):
                ts, ws, rel64 = build_fermi_dirac_nodes(q, m_n, 0.2, 2.0, mode=mode)
                # BF16 verification
                etas = torch.from_numpy(np.geomspace(0.2, 2.0, 256))
                tgt = etas.pow(-q).numpy()
                eb = etas.to(torch.bfloat16)
                tb = torch.from_numpy(ts).to(torch.bfloat16)
                wb = torch.from_numpy(ws).to(torch.bfloat16)
                out_bf = (F.silu(-eb[:, None] * tb[None, :]) * (-wb[None, :])).sum(dim=-1).float().numpy()
                rel_bf = float(np.max(np.abs(out_bf - tgt) / tgt))
                m_dict[f"M={m_n}"] = {
                    "fp64_max_rel_err": rel64,
                    "bf16_max_rel_err": rel_bf,
                    "all_weights_nonneg": bool(np.all(ws >= 0.0)),
                    "nnz": int(np.sum(ws > 0.0)),
                }
            q_res[mode] = m_dict
        fd_table[f"q={int(q)}"] = q_res

    # --- 3B. 2-Layer Self-Normalizing Bilinear Circuit (Q3/Q2) with INDEPENDENT (s1, s2) + Final RMSNorm + lm_head! ---
    # Layer 1 (s1 in [0.5, 3.0]):
    #   - Bilinear C4 product: P_k = B_k / s1^2  (8 neurons: 6 Karatsuba + 2 Nyquist)
    #   - Anchor pass-through u1 = F_0(rho1, rho1) ~= c0 / s1  (M0=10 neurons on [c0/3.2, c0/0.45])
    # Layer 2 (s2 in [0.5, 3.0], INDEPENDENT of s1!):
    #   - Fused F_2(eta, P_k / s2) -> writes (s2 / c0^2) * B_align * B_k directly into lm_head subspace!
    #     where eta = u1 / s2 ~= c0 / (s1 * s2) in [c0/9.5, c0/0.22]
    #   - Optional: compare with & without Active Prose-Bias Nuller (q=1)!
    c0 = 1.0
    n_test = 256
    s1_vals = rng.uniform(0.55, 2.8, size=n_test)
    s2_vals = rng.uniform(0.55, 2.8, size=n_test)  # COMPLETELY INDEPENDENT from s1!

    # Synthetic C4 Fourier inputs: (a1_f, lam_f) -> exact centered logits = F_real @ y_exact
    a1_f = rng.normal(scale=0.35, size=(n_test, 3))
    th = rng.uniform(-10.0 * math.pi, 10.0 * math.pi, size=n_test)
    lam_r = 0.72 * np.cos(th); lam_i = 0.72 * np.sin(th); lam_2 = 0.58 * np.cos(2.0 * th)
    # Scale to realistic logit magnitude ~ 1.5 nats
    gain = 3.5
    y_ex_1 = gain * (lam_r + 1j * lam_i) * (a1_f[:, 0] + 1j * a1_f[:, 1])
    y_ex_2 = gain * lam_2 * a1_f[:, 2]
    y_exact = np.stack([y_ex_1.real, y_ex_1.imag, y_ex_2], axis=-1)  # (N, 3)
    logits_exact = y_exact @ F_REAL_NP.T  # (N, 4)

    # Build Layer 1 (K1 = 8 + M0) & Layer 2 (K2 = 3 * M2)
    # s1, s2 in [0.55, 2.8] => rho1 in [0.357, 1.818], eta = c0/(s1*s2) in [0.127, 3.306]
    ts0, ws0, _ = build_fermi_dirac_nodes(0.0, 10, c0 / 3.0, c0 / 0.50, mode="nnls_pos")
    ts2, ws2, _ = build_fermi_dirac_nodes(2.0, 12, c0 / 8.5, c0 / 0.28, mode="trapezoid")

    # Slots: [0,1,2]=a1_f, [3,4,5]=(lam_r, lam_i, lam_2), [6]=e_anc (c0), [7,8,9]=P_1..3, [10]=u1
    l1_blocks = [
        compile_complex_mul_neurons(e[3], e[4], e[0], e[1], e[7], e[8], g1, scale=gain, method="karatsuba6"),
        compile_antipodal_pair(e[5], e[2], gain * e[9], g1),
        compile_fermi_dirac_operator(e[6], e[6], e[10], g1, ts0, ws0, scale=1.0),
    ]
    layer1 = CompiledSwiGLULayer(g1, *stack_blocks(l1_blocks))

    l2_blocks = [
        compile_fermi_dirac_operator(e[10], e[7 + k], (c0 ** 2) * b_align[:, k], g2, ts2, ws2, scale=1.0)
        for k in range(3)
    ]
    layer2 = CompiledSwiGLULayer(g2, *stack_blocks(l2_blocks))

    # Also build Active Prose-Bias Nuller (q=1 on rho2 = c0/s2, cancelling M_cand D_{gamma_f} h_prose!)
    # Use closed-form trapezoid (max(t*w) ~ 668 vs NNLS 2.95e6) for tight conditioning on ungated rounding!
    ts1_null, ws1_null, _ = build_fermi_dirac_nodes(1.0, 8, c0 / 3.0, c0 / 0.50, mode="trapezoid")
    # Up weight reads (F_real^T M_cand D_{gamma_f})_k in residual space!
    a_eff = m_cand * gf[None, :]  # (4, d)
    proj_bias_dirs = (torch.from_numpy(F_REAL_NP).T @ a_eff)  # (3, d)
    nuller_blocks = [
        compile_fermi_dirac_operator(e[6], proj_bias_dirs[k], -c0 * b_align[:, k], g2, ts1_null, ws1_null, scale=1.0)
        for k in range(3)
    ]
    layer2_with_nuller = CompiledSwiGLULayer(g2, *stack_blocks(l2_blocks + nuller_blocks))

    # Construct prose vectors:
    # h_p2_clean lies strictly in V_prose (orthogonal to BOTH V_EM and V_cand^(gamma_f))!
    # h_p2_biased adds realistic O(1.5 nat) digit bias in V_cand^(gamma_f) (still orthogonal to V_EM!).
    h_p1_raw = sample_prose_batch(geom["u_prose"], s1_vals, rng)
    h_p2_clean = sample_prose_batch(geom["u_prose"], s2_vals, rng)
    bias_logits_3d = torch.from_numpy(rng.normal(scale=1.2, size=(n_test, 3))).to(torch.float64)
    # Scale bias by s2_vals so after final RMSNorm it produces O(1.2 nat) digit bias!
    h_p2_raw = h_p2_clean + (bias_logits_3d * torch.from_numpy(s2_vals)[:, None]) @ b_align.T

    x_em_in = (
        torch.from_numpy(a1_f[:, 0])[:, None] * e[0][None, :]
        + torch.from_numpy(a1_f[:, 1])[:, None] * e[1][None, :]
        + torch.from_numpy(a1_f[:, 2])[:, None] * e[2][None, :]
        + torch.from_numpy(lam_r)[:, None] * e[3][None, :]
        + torch.from_numpy(lam_i)[:, None] * e[4][None, :]
        + torch.from_numpy(lam_2)[:, None] * e[5][None, :]
        + c0 * e[6][None, :]
    )

    # Naive 1-layer uncompensated baseline (to quantify the s^{-3} collapse!)
    naive_l1 = CompiledSwiGLULayer(
        g1,
        *stack_blocks([
            compile_complex_mul_neurons(e[3], e[4], e[0], e[1], b_align[:, 0], b_align[:, 1], g1, scale=gain, method="karatsuba6"),
            compile_antipodal_pair(e[5], e[2], gain * b_align[:, 2], g1),
        ]),
    )
    x_naive_out = h_p2_clean + naive_l1.delta(h_p1_raw + x_em_in)
    # Wait: scale naive baseline at s=1 so we see the pure norm-sensitivity blowup/collapse across s in [0.55, 2.8]
    naive_logits = (final_norm(x_naive_out) @ m_cand.T).numpy()
    naive_scale_ratio = float(
        (np.linalg.norm(naive_logits, axis=-1) / np.maximum(np.linalg.norm(logits_exact, axis=-1), 1e-8)).max()
        / (np.linalg.norm(naive_logits, axis=-1) / np.maximum(np.linalg.norm(logits_exact, axis=-1), 1e-8)).min()
    )

    sn_2layer = {}
    for dt_name, dt in (("fp64", torch.float64), ("fp32", torch.float32), ("bf16", torch.bfloat16)):
        x1_in = (h_p1_raw + x_em_in).to(dt)
        d1 = layer1.delta(x1_in)
        # Move into Layer 2 with COMPLETELY INDEPENDENT prose vector h_p2_clean!
        x2_in = h_p2_clean.to(dt) + d1
        x_fin = layer2(x2_in)
        logits_got = (final_norm(x_fin).to(torch.float64) @ m_cand.T).numpy()
        max_logit_err = float(np.max(np.abs(logits_got - logits_exact)))
        mean_kl = float(
            np.mean(
                np.sum(
                    F.softmax(torch.from_numpy(logits_exact), dim=-1).numpy()
                    * (
                        F.log_softmax(torch.from_numpy(logits_exact), dim=-1).numpy()
                        - F.log_softmax(torch.from_numpy(logits_got), dim=-1).numpy()
                    ),
                    axis=-1,
                )
            )
        )
        # Also test WITH raw biased prose + Active Prose-Bias Nuller!
        x2_biased = h_p2_raw.to(dt) + d1 + c0 * e[6][None, :].to(dt)
        raw_prose_bias = float((final_norm(h_p2_raw.to(dt)).to(torch.float64) @ m_cand.T).abs().max().item())
        x_fin_nulled = layer2_with_nuller(x2_biased)
        logits_nulled = (final_norm(x_fin_nulled).to(torch.float64) @ m_cand.T).numpy()
        nulled_err = float(np.max(np.abs(logits_nulled - logits_exact)))
        nulled_kl = float(
            np.mean(
                np.sum(
                    F.softmax(torch.from_numpy(logits_exact), dim=-1).numpy()
                    * (
                        F.log_softmax(torch.from_numpy(logits_exact), dim=-1).numpy()
                        - F.log_softmax(torch.from_numpy(logits_nulled), dim=-1).numpy()
                    ),
                    axis=-1,
                )
            )
        )
        # Pure prose check:
        # (i) g_EM=0 -> bitwise 0.0
        # (ii) ungated pure EM-slot layer2 -> O(eps^2) ~ 1e-30!
        # (iii) ungated layer2_with_nuller -> O(t_max*w_max*eps) ~ 1e-13 in FP64 (first-order nuller tradeoff!)
        pure_prose_gated_diff = float((layer2_with_nuller(h_p2_raw.to(dt), g_em=0) - h_p2_raw.to(dt)).abs().max().item())
        pure_prose_ungated_em_diff = float((layer2(h_p2_raw.to(dt), g_em=1) - h_p2_raw.to(dt)).abs().max().item())
        pure_prose_ungated_nuller_diff = float((layer2_with_nuller(h_p2_raw.to(dt), g_em=1) - h_p2_raw.to(dt)).abs().max().item())
        sn_2layer[dt_name] = {
            "clean_prose_max_logit_err": max_logit_err,
            "clean_prose_mean_kl": mean_kl,
            "raw_prose_digit_bias_max": raw_prose_bias,
            "biased_prose_nulled_max_logit_err": nulled_err,
            "biased_prose_nulled_mean_kl": nulled_kl,
            "pure_prose_g_em_0_max_diff": pure_prose_gated_diff,
            "pure_prose_ungated_em_only_max_diff": pure_prose_ungated_em_diff,
            "pure_prose_ungated_with_nuller_max_diff": pure_prose_ungated_nuller_diff,
        }

    # --- 3C. 3-Layer Decoupled Self-Normalizing Trilinear Q1 Circuit with INDEPENDENT (s1, s2, s3) ---
    s3_vals = rng.uniform(0.55, 2.5, size=n_test)
    s1_q1 = rng.uniform(0.6, 2.2, size=n_test)
    s2_q1 = rng.uniform(0.6, 2.2, size=n_test)
    h_p1_q1 = sample_prose_batch(geom["u_prose"], s1_q1, rng)
    h_p2_q1 = sample_prose_batch(geom["u_prose"], s2_q1, rng)
    h_p3_q1 = sample_prose_batch(geom["u_prose"], s3_vals, rng)

    u1_q = rng.normal(scale=0.9, size=n_test); v1_q = rng.normal(scale=0.9, size=n_test)
    u2_q = rng.normal(scale=0.9, size=n_test); v2_q = rng.normal(scale=0.9, size=n_test)
    wh_q = rng.uniform(-12.0 * math.pi, 12.0 * math.pi, size=n_test)
    z_q1_ex = np.conj(u1_q + 1j * v1_q) * (u2_q + 1j * v2_q) * np.exp(-1j * wh_q)
    y_q1_ex = np.stack([z_q1_ex.real, z_q1_ex.imag, np.zeros(n_test)], axis=-1)
    logits_q1_ex = y_q1_ex @ F_REAL_NP.T

    ts0_1, ws0_1, _ = build_fermi_dirac_nodes(0.0, 8, c0 / 2.4, c0 / 0.55, mode="nnls_pos")
    ts0_2, ws0_2, _ = build_fermi_dirac_nodes(0.0, 8, c0 / 2.4, c0 / 0.55, mode="nnls_pos")
    ts2_3, ws2_3, _ = build_fermi_dirac_nodes(2.0, 14, c0 / 13.0, c0 / 0.19, mode="trapezoid")

    # Slots: [0,1]=T1, [2,3]=T2, [4,5]=(ch,sh), [6]=e_anc(c0), [7,8]=T2', [9]=u1, [10,11]=Z, [12]=u12
    q1_sn_l1 = CompiledSwiGLULayer(
        g1,
        *stack_blocks([
            compile_complex_mul_neurons(e[2], e[3], e[4], e[5], e[7], e[8], g1, conj_b=True, method="karatsuba6"),
            compile_fermi_dirac_operator(e[6], e[6], e[9], g1, ts0_1, ws0_1, scale=1.0),
        ]),
    )
    q1_sn_l2 = CompiledSwiGLULayer(
        g2,
        *stack_blocks([
            compile_complex_mul_neurons(e[0], e[1], e[7], e[8], e[10], e[11], g2, conj_a=True, method="karatsuba6"),
            compile_fermi_dirac_operator(e[6], e[9], e[12], g2, ts0_2, ws0_2, scale=1.0),
        ]),
    )
    q1_sn_l3 = CompiledSwiGLULayer(
        g3,
        *stack_blocks([
            compile_fermi_dirac_operator(e[12], e[10], (c0 ** 2) * b_align[:, 0], g3, ts2_3, ws2_3, scale=1.0),
            compile_fermi_dirac_operator(e[12], e[11], (c0 ** 2) * b_align[:, 1], g3, ts2_3, ws2_3, scale=1.0),
        ]),
    )
    q1_sn_total_n = q1_sn_l1.num_neurons + q1_sn_l2.num_neurons + q1_sn_l3.num_neurons
    assert q1_sn_total_n <= 64

    x_em_q1 = (
        torch.from_numpy(u1_q)[:, None] * e[0][None, :]
        + torch.from_numpy(v1_q)[:, None] * e[1][None, :]
        + torch.from_numpy(u2_q)[:, None] * e[2][None, :]
        + torch.from_numpy(v2_q)[:, None] * e[3][None, :]
        + torch.from_numpy(np.cos(wh_q))[:, None] * e[4][None, :]
        + torch.from_numpy(np.sin(wh_q))[:, None] * e[5][None, :]
        + c0 * e[6][None, :]
    )
    sn_3layer_q1 = {}
    for dt_name, dt in (("fp64", torch.float64), ("fp32", torch.float32), ("bf16", torch.bfloat16)):
        d1 = q1_sn_l1.delta((h_p1_q1 + x_em_q1).to(dt))
        # Layer 2 receives T1 + anchor + Layer 1 delta + independent prose h_p2_q1
        x2 = (h_p2_q1 + torch.from_numpy(u1_q)[:, None] * e[0][None, :] + torch.from_numpy(v1_q)[:, None] * e[1][None, :] + c0 * e[6][None, :]).to(dt) + d1
        d2 = q1_sn_l2.delta(x2)
        x3 = h_p3_q1.to(dt) + d2
        x_f = q1_sn_l3(x3)
        lg = (final_norm(x_f).to(torch.float64) @ m_cand.T).numpy()
        err = float(np.max(np.abs(lg - logits_q1_ex)))
        kl = float(
            np.mean(
                np.sum(
                    F.softmax(torch.from_numpy(logits_q1_ex), dim=-1).numpy()
                    * (
                        F.log_softmax(torch.from_numpy(logits_q1_ex), dim=-1).numpy()
                        - F.log_softmax(torch.from_numpy(lg), dim=-1).numpy()
                    ),
                    axis=-1,
                )
            )
        )
        sn_3layer_q1[dt_name] = {"max_logit_err": err, "mean_kl": kl}

    assert sn_2layer["fp64"]["clean_prose_mean_kl"] < 1e-4
    assert sn_2layer["fp64"]["biased_prose_nulled_mean_kl"] < 1e-4
    assert sn_2layer["fp64"]["pure_prose_g_em_0_max_diff"] == 0.0
    assert sn_2layer["bf16"]["pure_prose_g_em_0_max_diff"] == 0.0
    assert sn_2layer["fp64"]["pure_prose_ungated_em_only_max_diff"] < 1e-15
    assert sn_2layer["fp64"]["pure_prose_ungated_with_nuller_max_diff"] < 1e-10
    assert sn_3layer_q1["fp64"]["mean_kl"] < 5e-4
    assert naive_scale_ratio > 50.0  # confirms naive uncompensated collapses by >50x!

    return {
        "fermi_dirac_quadrature_table": fd_table,
        "naive_uncompensated_logit_scale_ratio_5x_s": naive_scale_ratio,
        "self_norm_2layer_neurons": layer1.num_neurons + layer2.num_neurons,
        "self_norm_2layer_with_nuller_neurons": layer1.num_neurons + layer2_with_nuller.num_neurons,
        "self_norm_2layer_independent_s1_s2": sn_2layer,
        "self_norm_3layer_q1_neurons": q1_sn_total_n,
        "self_norm_3layer_q1_independent_s1_s2_s3": sn_3layer_q1,
    }


# =============================================================================
# Certificate 4: Real Stage 3D G1 / G5 / Adversarial Extrapolation + PureKL=0 + Quadratic MixKL + Param Audit
# =============================================================================


def _fit_univariate_swiglu_log(p_lo: float = 0.015, p_hi: float = 0.88, k_knots: int = 4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit exact C4-equivariant univariate SwiGLU log-spline f(p) = sum_m w_m SiLU(alpha_m*(p - t_m)) ~= log(p) + c."""
    grid = np.geomspace(p_lo, p_hi, 600)
    knots = np.array([-0.10, 0.02, 0.10, 0.35][:k_knots], dtype=np.float64)
    alphas = np.array([15.0, 22.0, 10.0, 4.5][:k_knots], dtype=np.float64)
    phi = np.column_stack(
        [(alphas[m] * (grid - knots[m])) / (1.0 + np.exp(-alphas[m] * (grid - knots[m]))) for m in range(k_knots)]
    )
    # Include an intercept column in regression ONLY to absorb the additive constant, then drop intercept!
    design = np.column_stack([np.ones_like(grid), phi])
    # Weight by sqrt(p) so L2 error directly upper-bounds KL(p || q) ~ 0.5 * sum p_k * (err_k)^2!
    w_reg = np.sqrt(grid)
    coef, *_ = np.linalg.lstsq(design * w_reg[:, None], np.log(grid) * w_reg, rcond=None)
    return knots, alphas, coef[1:]


def run_cert4_real_g1_and_audits(geom: dict) -> dict:
    """End-to-end verification on locked real Stage 3D s17 data (G1, G5(pi), G_adv, Q2) + PureKL/MixKL + 4B/9B/27B budget."""
    rng = np.random.default_rng(4004)
    enc, s17_sha = load_s17_encoder(REPO_ROOT)
    kern = PhysicalKernelS17(enc.table)

    u_em = geom["u_em"]
    g1, g2, g3, gf = geom["gamma_l1"], geom["gamma_l2"], geom["gamma_l3"], geom["gamma_final"]
    m_cand = geom["m_cand"]
    b_align = geom["b_align_gamma"]
    final_norm = Qwen35RMSNorm(gf)
    e = [u_em[:, i] for i in range(28)]

    c0 = 1.0
    # Compile the 2-layer Self-Normalizing Linear-in-p Circuit (K = 39 <= 48 neurons)
    # AND the 3-layer Self-Normalizing C4-Equivariant SwiGLU-Log Circuit (K = 64 <= 64 neurons)!
    # Prose RMS sweeps s1, s2 in [0.6, 2.4] => rho in [0.417, 1.667], eta = c0/(s1*s2) in [0.174, 2.778]
    # Pure closed-form Euler-Maclaurin trapezoidal Fermi-Dirac quadrature (strictly positive w_m > 0!)
    ts0, ws0, _ = build_fermi_dirac_nodes(0.0, 7, c0 / 2.6, c0 / 0.55, mode="trapezoid")
    ts2, ws2, _ = build_fermi_dirac_nodes(2.0, 8, c0 / 6.2, c0 / 0.33, mode="trapezoid")
    ts2_b, ws2_b, _ = build_fermi_dirac_nodes(2.0, 7, c0 / 6.2, c0 / 0.33, mode="trapezoid")
    ts2_a1, ws2_a1, _ = build_fermi_dirac_nodes(2.0, 6, c0 / 2.6, c0 / 0.55, mode="trapezoid")
    ts3_a2, ws3_a2, _ = build_fermi_dirac_nodes(3.0, 6, c0 / 2.6, c0 / 0.55, mode="trapezoid")
    knots, alphas, w_log = _fit_univariate_swiglu_log(k_knots=4)  # 4 * 4 = 16 neurons

    # Circuit A (Linear-in-p, 2 layers, tau_lin=3.55):
    # Slots: [0,1,2]=a1_f, [3,4,5]=(lam1_r, lam1_i, lam2), [6]=e_anc(c0), [7,8,9]=P_1..3, [10]=u1
    # BF16 Scratch Pre-Gain Theorem: since P_k = hat_p_k / s1^2 shrinks to O(0.02) at large s1,
    # pre-scaling P_k by kappa_scratch=8.0 in Layer 1 (and dividing by 8.0 in Layer 2) lifts P_k
    # 8x above the BF16 residual-addition noise floor sigma_bf16 = eps_bf16 * s2 / sqrt(12)
    # while keeping ||d1||_2 << ||h_prose||_2 (zero RMS shift, zero extra neurons!).
    tau_lin = 3.55
    kappa_scratch = 8.0
    l1_a = CompiledSwiGLULayer(
        g1,
        *stack_blocks([
            compile_complex_mul_neurons(e[3], e[4], e[0], e[1], e[7], e[8], g1, scale=kappa_scratch, method="karatsuba6"),
            compile_antipodal_pair(e[5], e[2], kappa_scratch * e[9], g1),
            compile_fermi_dirac_operator(e[6], e[6], e[10], g1, ts0, ws0, scale=1.0),
        ]),
    )
    l2_a = CompiledSwiGLULayer(
        g2,
        *stack_blocks([
            compile_fermi_dirac_operator(e[10], e[7 + k], (tau_lin * (c0 ** 2) / kappa_scratch) * b_align[:, k], g2, ts2, ws2, scale=1.0)
            for k in range(3)
        ]),
    )
    n_neurons_lin = l1_a.num_neurons + l2_a.num_neurons

    # Circuit B (Exact C4-Equivariant SwiGLU-Log, 3 layers, 15 + 33 + 16 = 64 neurons!):
    # Apply second-stage BF16 pre-gain kappa_2=4.0 on slots [11..14] (divided back in l3_b alphas)
    # to lift Layer 2 -> Layer 3 probability slots 4x above BF16 residual-addition noise!
    kappa_2 = 4.0
    l2_b = CompiledSwiGLULayer(
        g2,
        *stack_blocks(
            [
                compile_fermi_dirac_operator(e[10], e[7 + k], (kappa_2 * (c0 ** 2) / kappa_scratch) * e[11 + k], g2, ts2_b, ws2_b, scale=1.0)
                for k in range(3)
            ]
            + [
                compile_fermi_dirac_operator(e[6], e[6], (kappa_2 * c0) * e[14], g2, ts2_a1, ws2_a1, scale=1.0),
                compile_fermi_dirac_operator(e[6], e[6], (c0 ** 2) * e[15], g2, ts3_a2, ws3_a2, scale=1.0),
            ]
        ),
    )
    inv_g3 = 1.0 / g3
    wg3_list, wu3_list, wd3_list = [], [], []
    f_real_t = torch.from_numpy(F_REAL_NP)
    for c_idx in range(4):
        p_c_vec = 0.25 * e[14] + sum(float(F_REAL_NP[c_idx, k]) * e[11 + k] for k in range(3))
        out_dir_c = b_align @ f_real_t[c_idx]  # (d,): maps into centered digit c via B_align F_real^T!
        for m_idx in range(len(knots)):
            gate_res = (float(alphas[m_idx]) / kappa_2) * (p_c_vec - float(knots[m_idx]) * e[14])
            wg3_list.append(gate_res * inv_g3)
            wu3_list.append(e[15] * inv_g3)
            wd3_list.append(float(w_log[m_idx]) * out_dir_c)
    l3_b = CompiledSwiGLULayer(g3, torch.stack(wg3_list, dim=0), torch.stack(wu3_list, dim=0), torch.stack(wd3_list, dim=1))
    n_neurons_log = l1_a.num_neurons + l2_b.num_neurons + l3_b.num_neurons
    assert n_neurons_lin <= 48 and n_neurons_log <= 64, f"Exceeded 64 neurons: lin={n_neurons_lin}, log={n_neurons_log}"

    # Evaluate on 3 real Stage 3D physical regimes:
    # 1. G1_main_extrap_both (|omega|=0.40, h=2, ell=2, |theta|=1.60 > pi/2, 16 blocks = 2048 rows)
    # 2. G5_half_turn        (|omega|=math.pi/4, h=2, ell=2, |theta|=pi, cos=-1.0, 8 blocks = 1024 rows)
    # 3. G_adv_extreme       (|omega|=3.70, h=3, ell=4, |theta|=25.9 rad ~ 8.24*pi, 8 blocks = 1024 rows)
    regimes = [
        ("G1_main_extrap_both", BLIND_CELLS["G1_main_extrap_both"].namespace, 0.40, 2, 2, 4.0, 16),
        ("G5_half_turn_pi", 950406, math.pi / 4.0, 2, 2, 4.0, 8),
        ("G_adv_omega3p7_th25p9", 950407, 3.70, 3, 4, 7.0, 8),
    ]
    eval_results = {}
    for name, ns, abs_om, h, ell, tp, n_blk in regimes:
        bmap, _ = generate_paired_cfo_blocks(ns, 0, n_blk, h, ell, tp, (-abs_om, abs_om), snr_db=2.0, sigma_phi=0.10)
        p_phys_all, lbl_all, a1_f_all, lam_all = [], [], [], []
        for om, mb in bmap.items():
            z1 = torch.from_numpy(np.asarray(enc.encode(mb.y1), dtype=np.int64))
            z2 = torch.from_numpy(np.asarray(enc.encode(mb.y2), dtype=np.int64))
            codes = torch.stack([z1, z2], dim=-1)
            n = len(z1)
            # Quotient-canonical inner1 = z1 % 4 (exact M1/W2 quotient wrap!)
            inner1 = z1 % 4
            a1_canon = kern.prior_canon[inner1].numpy().astype(np.float64)
            # Exact circulant kernel K(theta, V) eigenvalues in F_4 basis:
            theta_val = om * float(h + ell)
            var_val = (0.10 ** 2) * float(h + ell)
            k_mat = kern.kernel(
                torch.tensor([theta_val], dtype=torch.float32),
                torch.tensor([var_val], dtype=torch.float32),
                torch.tensor([2.0], dtype=torch.float32),
            )[0].numpy().astype(np.float64)
            # Eigenvalues of circulant K in F_real basis: F_real^T K^T F_real
            k_hat = F_REAL_NP.T @ k_mat.T @ F_REAL_NP
            # K^T acts on V1 as [[l1r, -l1i], [l1i, l1r]] and on V2 as l2
            l1r, l1i, l2 = float(k_hat[0, 0]), float(k_hat[1, 0]), float(k_hat[2, 2])
            # Exact canonical posterior and canonical label (unshifted by a1 = z1 // 4)
            p_canon = a1_canon @ k_mat
            y3_raw = sector_of_complex(matched_sum(mb.y3_repeat)).astype(np.int64)
            lbl_canon = (y3_raw - (z1.numpy() // 4)) % 4

            p_phys_all.append(p_canon)
            lbl_all.append(lbl_canon)
            a1_f_all.append(a1_canon @ F_REAL_NP)
            lam_all.append(np.repeat([[l1r, l1i, l2]], n, axis=0))

        P_ex = np.concatenate(p_phys_all, axis=0)
        Y_ex = np.concatenate(lbl_all, axis=0)
        A1_f = np.concatenate(a1_f_all, axis=0)
        Lam = np.concatenate(lam_all, axis=0)
        n_tot = len(Y_ex)

        oracle_acc = float(np.mean(P_ex.argmax(-1) == Y_ex))
        oracle_nll = float(-np.mean(np.log(np.clip(P_ex[np.arange(n_tot), Y_ex], 1e-12, 1.0))))

        # Randomize prose RMS across a 4x dynamic range s in [0.6, 2.4] per sample!
        s1_r = rng.uniform(0.6, 2.4, size=n_tot)
        s2_r = rng.uniform(0.6, 2.4, size=n_tot)
        h_p1 = sample_prose_batch(geom["u_prose"], s1_r, rng)
        h_p2 = sample_prose_batch(geom["u_prose"], s2_r, rng)
        # For Layer 3 of Circuit B, residual norm stays near s2_r (adjacent layer)
        h_p3 = h_p2

        x_em = (
            torch.from_numpy(A1_f[:, 0])[:, None] * e[0][None, :]
            + torch.from_numpy(A1_f[:, 1])[:, None] * e[1][None, :]
            + torch.from_numpy(A1_f[:, 2])[:, None] * e[2][None, :]
            + torch.from_numpy(Lam[:, 0])[:, None] * e[3][None, :]
            + torch.from_numpy(Lam[:, 1])[:, None] * e[4][None, :]
            + torch.from_numpy(Lam[:, 2])[:, None] * e[5][None, :]
            + c0 * e[6][None, :]
        )

        reg_out = {
            "n_samples": n_tot,
            "abs_omega": abs_om,
            "abs_theta": abs_om * float(h + ell),
            "oracle_acc": oracle_acc,
            "oracle_nll": oracle_nll,
        }
        for dt_name, dt in (("fp64", torch.float64), ("fp32", torch.float32), ("bf16", torch.bfloat16)):
            d1 = l1_a.delta((h_p1 + x_em).to(dt))
            # Circuit A (Linear-in-p, 2 layers)
            x_fin_a = l2_a(h_p2.to(dt) + d1)
            lg_a = (final_norm(x_fin_a).to(torch.float64) @ m_cand.T)
            q_a = F.softmax(lg_a, dim=-1).numpy()
            acc_a = float(np.mean(q_a.argmax(-1) == Y_ex))
            nll_a = float(-np.mean(np.log(np.clip(q_a[np.arange(n_tot), Y_ex], 1e-12, 1.0))))
            agree_a = float(np.mean(q_a.argmax(-1) == P_ex.argmax(-1)))

            # Circuit B (SwiGLU-Log calibrated, 3 layers, <= 64 neurons)
            x2_b = h_p2.to(dt) + d1 + c0 * e[6][None, :].to(dt)
            d2_b = l2_b.delta(x2_b)
            x_fin_b = l3_b(h_p3.to(dt) + d2_b)
            lg_b = (final_norm(x_fin_b).to(torch.float64) @ m_cand.T)
            q_b = F.softmax(lg_b, dim=-1).numpy()
            acc_b = float(np.mean(q_b.argmax(-1) == Y_ex))
            nll_b = float(-np.mean(np.log(np.clip(q_b[np.arange(n_tot), Y_ex], 1e-12, 1.0))))
            agree_b = float(np.mean(q_b.argmax(-1) == P_ex.argmax(-1)))
            kl_b = float(np.mean(np.sum(P_ex * (np.log(np.clip(P_ex, 1e-12, 1.0)) - np.log(np.clip(q_b, 1e-12, 1.0))), axis=-1)))

            reg_out[dt_name] = {
                "lin_2layer_acc": acc_a,
                "lin_2layer_nll": nll_a,
                "lin_2layer_oracle_argmax_agree": agree_a,
                "log_3layer_acc": acc_b,
                "log_3layer_nll": nll_b,
                "log_3layer_nll_gap": nll_b - oracle_nll,
                "log_3layer_mean_kl_vs_oracle": kl_b,
                "log_3layer_oracle_argmax_agree": agree_b,
            }
        eval_results[name] = reg_out

        # Strict assertions on FP64, FP32, and BF16!
        assert reg_out["fp64"]["lin_2layer_oracle_argmax_agree"] == 1.0
        assert reg_out["fp64"]["log_3layer_oracle_argmax_agree"] == 1.0
        assert reg_out["fp32"]["log_3layer_oracle_argmax_agree"] == 1.0
        assert reg_out["fp64"]["log_3layer_mean_kl_vs_oracle"] < 5e-4
        assert reg_out["bf16"]["lin_2layer_oracle_argmax_agree"] > 0.97
        assert reg_out["bf16"]["log_3layer_mean_kl_vs_oracle"] < 4e-3
        assert abs(reg_out["bf16"]["lin_2layer_acc"] - oracle_acc) < 0.01
        if name in ("G1_main_extrap_both", "G5_half_turn_pi"):
            assert reg_out["bf16"]["log_3layer_oracle_argmax_agree"] == 1.0
            assert reg_out["bf16"]["lin_2layer_oracle_argmax_agree"] == 1.0
            assert reg_out["bf16"]["log_3layer_acc"] == oracle_acc

    # --- 4A-bis. Real Stage 3D Q2_C4_blind (Blackwell + Deduplication + Distractor Immunity) via Compiled SwiGLU! ---
    # Compile exact C4 cross-correlation Q2 circuit (conj_a=True, scale sqrt(2)*kappa_scratch on V1 and 2.0*kappa_scratch on V2)
    l1_q2 = CompiledSwiGLULayer(
        g1,
        *stack_blocks([
            compile_complex_mul_neurons(e[3], e[4], e[0], e[1], e[7], e[8], g1, conj_a=True, scale=math.sqrt(2.0) * kappa_scratch, method="karatsuba6"),
            compile_antipodal_pair(e[5], e[2], (2.0 * kappa_scratch) * e[9], g1),
            compile_fermi_dirac_operator(e[6], e[6], e[10], g1, ts0, ws0, scale=1.0),
        ]),
    )
    q2_omega = 0.35
    q2_map, _ = generate_paired_cfo_blocks(NS_BLIND_Q2_C4, 0, 8, 2, 2, 4.0, (q2_omega,), snr_db=2.0, sigma_phi=0.10)
    mb_q2 = q2_map[q2_omega]
    z1_q2 = np.asarray(enc.encode(mb_q2.y1), dtype=np.int64)
    a1_q2 = kern.prior_canon[torch.from_numpy(z1_q2 % 4)].numpy().astype(np.float64)
    a1_q2_f = a1_q2 @ F_REAL_NP
    lbl_q2_canon = (mb_q2.x1 - (z1_q2 // 4)) % 4
    n_q2 = len(lbl_q2_canon)
    s1_q2 = rng.uniform(0.6, 2.4, size=n_q2)
    s2_q2 = rng.uniform(0.6, 2.4, size=n_q2)
    h_p1_q2 = sample_prose_batch(geom["u_prose"], s1_q2, rng)
    h_p2_q2 = sample_prose_batch(geom["u_prose"], s2_q2, rng)

    q2_arm_results = {}
    q2_probs_by_arm = {}
    for arm_name in ("none", "valid", "duplicate_copy_dedup", "independent_only", "valid_plus_independent", "fresh_resample"):
        refs, _ = build_q2_arm_references(mb_q2, arm_name, coord_rot=0)
        ref_sum, _ = extract_canonical_ref_summary(refs, n_q2, torch.device("cpu"), omega=q2_omega, sigma_phi=0.10, drift_type="cfo_continuous")
        bp_np = ref_sum[:, 0:4].numpy().astype(np.float64)
        bp_f = bp_np @ F_REAL_NP
        p_q2_ex = np.stack([np.sum(bp_np * np.roll(a1_q2, -c, axis=-1), axis=-1) for c in range(4)], axis=-1)
        p_q2_ex /= p_q2_ex.sum(axis=-1, keepdims=True)
        x_em_q2 = (
            torch.from_numpy(a1_q2_f[:, 0])[:, None] * e[0][None, :]
            + torch.from_numpy(a1_q2_f[:, 1])[:, None] * e[1][None, :]
            + torch.from_numpy(a1_q2_f[:, 2])[:, None] * e[2][None, :]
            + torch.from_numpy(bp_f[:, 0])[:, None] * e[3][None, :]
            + torch.from_numpy(bp_f[:, 1])[:, None] * e[4][None, :]
            + torch.from_numpy(bp_f[:, 2])[:, None] * e[5][None, :]
            + c0 * e[6][None, :]
        )
        d1_q2 = l1_q2.delta(h_p1_q2 + x_em_q2)
        d2_q2 = l2_b.delta(h_p2_q2 + d1_q2 + c0 * e[6][None, :])
        x_fin_q2 = l3_b(h_p2_q2 + d2_q2)
        lg_q2 = final_norm(x_fin_q2) @ m_cand.T
        q_q2 = F.softmax(lg_q2, dim=-1).numpy()
        q2_probs_by_arm[arm_name] = q_q2
        q2_arm_results[arm_name] = {
            "oracle_acc": float(np.mean(p_q2_ex.argmax(-1) == lbl_q2_canon)),
            "oracle_nll": float(-np.mean(np.log(np.clip(p_q2_ex[np.arange(n_q2), lbl_q2_canon], 1e-12, 1.0)))),
            "compiled_acc": float(np.mean(q_q2.argmax(-1) == lbl_q2_canon)),
            "compiled_nll": float(-np.mean(np.log(np.clip(q_q2[np.arange(n_q2), lbl_q2_canon], 1e-12, 1.0)))),
            "mean_kl_vs_oracle": float(np.mean(np.sum(p_q2_ex * (np.log(np.clip(p_q2_ex, 1e-12, 1.0)) - np.log(np.clip(q_q2, 1e-12, 1.0))), axis=-1))),
        }
    q2_dup_diff = float(np.max(np.abs(q2_probs_by_arm["duplicate_copy_dedup"] - q2_probs_by_arm["valid"])))
    q2_dist_diff = float(np.max(np.abs(q2_probs_by_arm["valid_plus_independent"] - q2_probs_by_arm["valid"])))
    q2_bw_gain = q2_arm_results["fresh_resample"]["compiled_acc"] - q2_arm_results["valid"]["compiled_acc"]
    assert q2_dup_diff == 0.0 and q2_dist_diff == 0.0 and q2_bw_gain > 0.05

    # --- 4A-ter. Real Stage 3D Q1 on G1 (|\omega h|=0.80) & G_adv (|\omega h|=11.10 rad) via 2-Layer Compiled SwiGLU! ---
    q1_real_results = {}
    q1_l1 = CompiledSwiGLULayer(
        g1,
        *stack_blocks([
            compile_complex_mul_neurons(e[6], e[7], e[3], e[4], e[9], e[10], g1, conj_a=True, scale=math.sqrt(2.0), method="karatsuba6"),
            compile_antipodal_pair(e[8], e[5], 2.0 * e[11], g1),
        ]),
    )
    q1_l2 = CompiledSwiGLULayer(
        g2,
        *stack_blocks([
            compile_complex_mul_neurons(e[0], e[1], e[9], e[10], b_align[:, 0], b_align[:, 1], g2, conj_a=True, scale=math.sqrt(2.0), method="karatsuba6"),
            compile_antipodal_pair(e[2], e[11], 2.0 * b_align[:, 2], g2),
        ]),
    )
    for q1_name, ns, abs_om, h, ell, tp, n_blk in (
        ("Q1_G1_wh0p80", BLIND_CELLS["G1_main_extrap_both"].namespace, 0.40, 2, 2, 4.0, 8),
        ("Q1_adv_wh11p10", 950407, 3.70, 3, 4, 7.0, 8),
    ):
        bmap_q1, _ = generate_paired_cfo_blocks(ns, 0, n_blk, h, ell, tp, (-abs_om, abs_om), snr_db=2.0, sigma_phi=0.10)
        p_ex_l, y_ex_l, a1f_l, a2f_l, wqf_l = [], [], [], [], []
        for om, mb in bmap_q1.items():
            z1 = np.asarray(enc.encode(mb.y1), dtype=np.int64)
            z2 = np.asarray(enc.encode(mb.y2), dtype=np.int64)
            a1_c = kern.prior_canon[torch.from_numpy(z1 % 4)].numpy().astype(np.float64)
            a2_c = kern.prior_canon[torch.from_numpy(z2 % 4)].numpy().astype(np.float64)
            wq = kern._wrapped_quad_probs_torch(
                torch.tensor([om * float(h)], dtype=torch.float32),
                torch.tensor([(0.10 ** 2) * float(h)], dtype=torch.float32),
            )[0].numpy().astype(np.float64)
            a2_rot = sum(wq[k] * np.roll(a2_c, -k, axis=-1) for k in range(4))
            p_c = np.stack([np.sum(a1_c * np.roll(a2_rot, -c, axis=-1), axis=-1) for c in range(4)], axis=-1)
            p_c /= p_c.sum(axis=-1, keepdims=True)
            lbl_c = ((mb.x2 - mb.x1) - ((z2 // 4) - (z1 // 4))) % 4
            p_ex_l.append(p_c); y_ex_l.append(lbl_c)
            a1f_l.append(a1_c @ F_REAL_NP); a2f_l.append(a2_c @ F_REAL_NP)
            wqf_l.append(np.repeat((wq @ F_REAL_NP)[None, :], len(z1), axis=0))
        P_q1 = np.concatenate(p_ex_l, axis=0); Y_q1 = np.concatenate(y_ex_l, axis=0)
        A1_q1 = np.concatenate(a1f_l, axis=0); A2_q1 = np.concatenate(a2f_l, axis=0); WQ_q1 = np.concatenate(wqf_l, axis=0)
        n_q1 = len(Y_q1)
        hp1 = sample_prose_batch(geom["u_prose"], rng.uniform(0.6, 2.4, size=n_q1), rng)
        hp2 = sample_prose_batch(geom["u_prose"], rng.uniform(0.6, 2.4, size=n_q1), rng)
        x_q1_in = hp1.clone()
        for j in range(3):
            x_q1_in = x_q1_in + torch.from_numpy(A1_q1[:, j])[:, None] * e[j] + torch.from_numpy(A2_q1[:, j])[:, None] * e[3 + j] + torch.from_numpy(WQ_q1[:, j])[:, None] * e[6 + j]
        d1_q1 = q1_l1.delta(x_q1_in)
        x2_q1_in = hp2 + sum(torch.from_numpy(A1_q1[:, j])[:, None] * e[j] for j in range(3)) + d1_q1
        x_q1_out = q1_l2(x2_q1_in)
        lg_q1 = (final_norm(x_q1_out) @ m_cand.T).numpy()
        agree_q1 = float(np.mean(lg_q1.argmax(-1) == P_q1.argmax(-1)))
        assert agree_q1 == 1.0
        q1_real_results[q1_name] = {
            "n_samples": n_q1,
            "abs_omega_h": abs_om * float(h),
            "neurons": q1_l1.num_neurons + q1_l2.num_neurons,
            "oracle_acc": float(np.mean(P_q1.argmax(-1) == Y_q1)),
            "compiled_acc": float(np.mean(lg_q1.argmax(-1) == Y_q1)),
            "oracle_argmax_agreement": agree_q1,
        }

    # --- 4B. PureKL = 0.0 & Quadratic O(eps^2) vs Linear O(eps) Prose-Leakage Audit ---
    wg_b, wu_b, wd_b = stack_blocks([
        compile_complex_mul_neurons(e[3], e[4], e[0], e[1], b_align[:, 0], b_align[:, 1], g1, method="karatsuba6"),
        compile_antipodal_pair(e[5], e[2], b_align[:, 2], g1),
    ])
    bilin_only = CompiledSwiGLULayer(g1, wg_b, wu_b, wd_b)

    h_p_audit = sample_prose_batch(geom["u_prose"], rng.uniform(0.8, 2.2, size=128), rng)
    pure_kl_audit = {}
    for dt_name, dt in (("fp64", torch.float64), ("fp32", torch.float32), ("bf16", torch.bfloat16)):
        hp = h_p_audit.to(dt)
        base_lg = final_norm(hp).to(torch.float64) @ m_cand.T
        # With g_EM = 0 (always bitwise 0.0 in FP64, FP32, and BF16!)
        gated_lg = final_norm(l2_a(l1_a(hp, g_em=0), g_em=0)).to(torch.float64) @ m_cand.T
        # With g_EM = 1 (NO external gate, pure bilinear 8-neuron block -> O(eps_dtype^2)!)
        ungated_bilin_lg = final_norm(bilin_only(hp, g_em=1)).to(torch.float64) @ m_cand.T
        # With g_EM = 1 (NO external gate, Fermi-Dirac self-normalizing block -> amplifies BF16 rounding floor!)
        ungated_fd_lg = final_norm(l2_a(l1_a(hp, g_em=1), g_em=1)).to(torch.float64) @ m_cand.T
        max_diff_gated = float((base_lg - gated_lg).abs().max().item())
        max_diff_bilin = float((base_lg - ungated_bilin_lg).abs().max().item())
        max_diff_fd = float((base_lg - ungated_fd_lg).abs().max().item())
        pure_kl_audit[dt_name] = {
            "g_em_0_max_logit_diff": max_diff_gated,
            "g_em_0_pure_kl": 0.0 if max_diff_gated == 0.0 else float("nan"),
            "no_gate_pure_bilinear_max_logit_diff": max_diff_bilin,
            "no_gate_fermi_dirac_max_logit_diff": max_diff_fd,
        }
    assert pure_kl_audit["fp64"]["g_em_0_max_logit_diff"] == 0.0
    assert pure_kl_audit["bf16"]["g_em_0_max_logit_diff"] == 0.0
    assert pure_kl_audit["fp64"]["no_gate_fermi_dirac_max_logit_diff"] < 1e-13
    assert pure_kl_audit["bf16"]["no_gate_pure_bilinear_max_logit_diff"] < 5e-3

    # Quadratic O(eps^2) leakage suppression under imperfect K-FAC nullspace (eps_proj > 0)
    # Build a pure bilinear 8-neuron C4 layer (unit-norm outputs in V_EM) vs a matched linear projection
    wg_bl, wu_bl, wd_bl = stack_blocks([
        compile_complex_mul_neurons(e[3], e[4], e[0], e[1], e[7], e[8], g1, method="karatsuba6"),
        compile_antipodal_pair(e[5], e[2], e[9], g1),
    ])
    bilin_leak_mod = CompiledSwiGLULayer(g1, wg_bl, wu_bl, wd_bl)
    leak_dir = (e[0] + e[3] + e[2] + e[5]) / 2.0  # unit vector inside V_EM
    leak_audit = {}
    for eps_p in (1e-3, 1e-2, 5e-2):
        # Perturb prose by eps_p * ||h_prose||_2 along leak_dir
        p_norms = h_p_audit.norm(dim=-1, keepdim=True)
        h_leaky = h_p_audit + (eps_p * p_norms) * leak_dir[None, :]
        # SwiGLU bilinear leakage (quadratic in eps_p!)
        sw_leak = float(bilin_leak_mod.delta(h_leaky).norm(dim=-1).mean().item())
        # Matched linear projection leakage along the same slot direction
        z_leaky = h_leaky / torch.sqrt(h_leaky.pow(2).mean(dim=-1, keepdim=True) + RMS_EPS)
        lin_leak = float((z_leaky @ leak_dir).abs().mean().item())
        leak_audit[f"eps={eps_p}"] = {
            "swiglu_bilin_mean_delta_norm": sw_leak,
            "linear_lora_mean_delta_norm": lin_leak,
            "suppression_factor_linear_over_swiglu": float(lin_leak / max(sw_leak, 1e-30)),
        }
    # Verify scaling exponent ~ 2.0 for SwiGLU bilinear vs ~ 1.0 for linear!
    sw_slope = math.log(leak_audit["eps=0.01"]["swiglu_bilin_mean_delta_norm"] / leak_audit["eps=0.001"]["swiglu_bilin_mean_delta_norm"]) / math.log(10.0)
    assert 1.95 < sw_slope < 2.05, f"Expected quadratic exponent ~2.0, got {sw_slope}"

    # --- 4C. Exact Parameter Count Ledger across Qwen3.5-4B / 9B / Qwen3.8-27B ---
    hosts = {
        "Qwen3.5-4B": {"d": 2560, "d_ff": 9216, "L": 32, "host_params": 4_205_751_296},
        "Qwen3.5-9B": {"d": 4096, "d_ff": 12288, "L": 32, "host_params": 8_953_803_264},
        "Qwen3.8-27B": {"d": 5120, "d_ff": 17408, "L": 64, "host_params": 26_895_998_464},
    }
    param_ledger = {}
    for hname, cfg in hosts.items():
        tot_neurons = cfg["d_ff"] * cfg["L"]
        configs_k = {
            "K=6_min_complex_rot": 6,
            "K=8_min_centered_C4_conv": 8,
            "K=16_2layer_Q1_full_C4": 16,
            f"K={n_neurons_lin}_2layer_selfnorm_lin": n_neurons_lin,
            f"K={n_neurons_log}_3layer_selfnorm_log": n_neurons_log,
            "K=64_budget_ceiling": 64,
        }
        h_entry = {"total_mlp_neurons": tot_neurons, "configs": {}}
        for cname, k_val in configs_k.items():
            raw_p = 3 * k_val * cfg["d"]
            h_entry["configs"][cname] = {
                "neurons": k_val,
                "neuron_fraction_pct": 100.0 * k_val / tot_neurons,
                "raw_dense_params": raw_p,
                "pct_of_tier_p1_2184964": 100.0 * raw_p / 2_184_964,
                "pct_of_host_params": 100.0 * raw_p / cfg["host_params"],
            }
        param_ledger[hname] = h_entry

    return {
        "s17_sha256": s17_sha,
        "circuit_a_lin_2layer_neurons": n_neurons_lin,
        "circuit_b_log_3layer_neurons": n_neurons_log,
        "real_s17_extrapolation_q3": eval_results,
        "real_s17_q2_c4_blind": {
            "arms": q2_arm_results,
            "blackwell_fresh_minus_valid_acc": q2_bw_gain,
            "duplicate_dedup_max_prob_diff": q2_dup_diff,
            "valid_plus_indep_max_prob_diff": q2_dist_diff,
        },
        "real_s17_q1_2layer": q1_real_results,
        "pure_kl_audit": pure_kl_audit,
        "quadratic_leakage_audit": {
            "by_eps": leak_audit,
            "empirical_swiglu_scaling_exponent": sw_slope,
        },
        "param_ledger_4b_9b_27b": param_ledger,
    }


# =============================================================================
# Certificate 5 (Round 2): Cross-Track Reconciliation (N1 x N2 x N3 x N4)
#   5A. Finite-M Fermi-Dirac Asymmetric Half-Line Amplification Law (Lemma N1.5)
#       & 6-Regime Safety Audit (N2 FP32 trigger leak, BF16 rounding explosion,
#       N4 FP32 Sparse Additive Overlay, N3 Student-t Zipf bursts & Dead-Zone Key,
#       hard g_EM=0).
#   5B. N2 Candidate-Orthogonal Bus D_{gamma_f} Leakage Audit + End-to-End
#       Converged Pipeline (N4 264-param carrier (+1 anchor) -> N2 r_hd=2 Shadow
#       GDN Ballast Bus (L_gap=64) -> N1 K=64 Circuit B FP32 Overlay -> B_align^(gamma_f))
#       on real locked Stage 3D G1 & G5 under BOTH BF16 host & Mixed-NF4 host.
#   5C. Unified 4B (BF16) & 27B (Mixed-NF4) Parameter & Live-Bug Accounting.
# =============================================================================


class FP32OverlaySwiGLULayer(nn.Module):
    """Exact N1+N4 FP32 Sparse Additive Overlay: casts host BF16/NF4 residual to FP32,
    applies Pre-RMSNorm + D_gamma^{-1}-preconditioned compiled SwiGLU in FP32, multiplies
    by optional scalar gate (stationary dead-zone key / g_EM), and casts delta back to host dtype.
    """

    def __init__(self, compiled_layer: CompiledSwiGLULayer):
        super().__init__()
        self.eps = compiled_layer.norm.eps
        self.register_buffer("gamma", compiled_layer.norm.gamma.detach().to(torch.float32))
        self.register_buffer("w_gate", compiled_layer.w_gate.detach().to(torch.float32))
        self.register_buffer("w_up", compiled_layer.w_up.detach().to(torch.float32))
        self.register_buffer("w_down", compiled_layer.w_down.detach().to(torch.float32))

    def delta_fp32(self, x: torch.Tensor, key_gate: torch.Tensor | float = 1.0) -> torch.Tensor:
        xf = x.to(torch.float32)
        s = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        z = (xf / s) * self.gamma
        g = F.silu(F.linear(z, self.w_gate))
        u = F.linear(z, self.w_up)
        d_out = F.linear(g * u, self.w_down)
        if isinstance(key_gate, torch.Tensor):
            d_out = d_out * key_gate.to(torch.float32).unsqueeze(-1)
        elif key_gate != 1.0:
            d_out = d_out * float(key_gate)
        return d_out

    def forward(self, x: torch.Tensor, g_em: int = 1, key_gate: torch.Tensor | float = 1.0) -> torch.Tensor:
        if g_em == 0:
            return x
        d_out = self.delta_fp32(x, key_gate=key_gate)
        return (x.to(torch.float32) + d_out).to(x.dtype)


def simulate_nf4_block64_quantize(w: torch.Tensor) -> torch.Tensor:
    """Exact Dettmers NF4 block-64 symmetric codebook quantization (for host MLP stress test)."""
    nf4_levels = torch.tensor(
        [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ],
        dtype=torch.float32,
        device=w.device,
    )
    orig_shape = w.shape
    flat = w.to(torch.float32).reshape(-1)
    pad = (-flat.numel()) % 64
    if pad:
        flat = F.pad(flat, (0, pad))
    blocks = flat.reshape(-1, 64)
    absmax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    normed = blocks / absmax
    idx = (normed.unsqueeze(-1) - nf4_levels[None, None, :]).abs().argmin(dim=-1)
    q_blocks = nf4_levels[idx] * absmax
    q_flat = q_blocks.reshape(-1)
    if pad:
        q_flat = q_flat[:-pad]
    return q_flat.reshape(orig_shape)


def run_shadow_gdn_ballast_bus(
    carrier_in: torch.Tensor,
    l_gap: int = 64,
    alpha_prose: float = 1.0 - 1e-7,
    beta_prose: float = 2.06e-9,
    ballast: float = 8.0,
    seed: int = 5052,
) -> Tuple[torch.Tensor, float]:
    """Simulate N2's r_hd=2 Shadow GDN Highway with RMSNormGated constant-norm ballast across L_gap prose tokens.

    Stores signed carrier_in (B, m_slots) into a 128-dim delta-rule head at t=0 using orthogonal
    EM write key k_em, steps through L_gap random prose tokens with compiled dual-barrier
    (alpha_prose, beta_prose), reads out via k_em + RMSNormGated with ballast b_0, and rescales
    by exact ballast recovery constant. Returns (carrier_out, max_rel_err).
    """
    b_size, m_slots = carrier_in.shape
    d_k = 128
    rng = np.random.default_rng(seed)
    # Write key k_em = e_0 in R^128; store [carrier_in, ballast] in value coordinates 0..m_slots
    # State S has shape (B, d_v_active) along key k_em, plus random orthogonal/leaky prose perturbation
    v_write = torch.cat(
        [carrier_in.to(torch.float64), torch.full((b_size, 1), ballast, dtype=torch.float64)],
        dim=-1,
    )  # (B, m_slots + 1)
    # Exact delta-rule state recurrence along k_em with random unit prose keys k_t (overlap ~ 1/sqrt(128))
    # S_0 = k_em^T v_write; at each prose token t=1..l_gap:
    # S_t = alpha_prose * S_{t-1} + beta_prose * k_t (v_t - S_{t-1}^T k_t)
    # Track full (B, d_k, m_slots+1) state in FP64 (faithful to N2 Theorem N2.1!)
    k_em = torch.zeros(d_k, dtype=torch.float64)
    k_em[0] = 1.0
    state = k_em[None, :, None] * v_write[:, None, :]  # (B, d_k, m+1)
    k_all_np = rng.standard_normal((l_gap, b_size, d_k))
    k_all_np /= np.linalg.norm(k_all_np, axis=-1, keepdims=True)
    v_all_np = rng.standard_normal((l_gap, b_size, m_slots + 1)) * 0.5
    k_all = torch.from_numpy(k_all_np)
    v_all = torch.from_numpy(v_all_np)
    for step_idx in range(l_gap):
        k_t = k_all[step_idx]  # (B, d_k)
        v_t = v_all[step_idx]  # (B, m+1)
        proj = torch.bmm(k_t.unsqueeze(1), state).squeeze(1)  # (B, m+1)
        state = alpha_prose * state + beta_prose * torch.bmm(k_t.unsqueeze(2), (v_t - proj).unsqueeze(1))

    # Readout at query token t = l_gap + 1 using q_em = k_em, followed by exact RMSNormGated!
    raw_read = state[:, 0, :]  # (B, m_slots + 1)
    # RMSNormGated with ballast: rms = sqrt(mean(raw_read^2) + eps)
    # Note: because ballast >> ||carrier||_2, we normalize by the ballast coordinate itself
    # or equivalently read ballast-scaled RMSNormGated with exact ratio raw_read[:, :-1] / raw_read[:, -1:] * ballast
    rms_g = torch.sqrt(raw_read.pow(2).mean(dim=-1, keepdim=True) + RMS_EPS)
    gate_em = F.silu(torch.full((b_size, 1), 4.0, dtype=torch.float64))
    y_normed = (raw_read / rms_g) * gate_em
    # Ballast coordinate y_normed[:, -1:] encodes exact (ballast / rms_g) * gate_em;
    # in N2's fixed-amplitude regime, rms_g = ballast / sqrt(m+1) * (1 + O(||c||^2 / b^2)),
    # and the exact recovered carrier from the ballast-locked amplitude is:
    # (for strict RMSNormGated constant ballast calibration, scale by rms_nom / gate_em):
    carrier_recovered = (y_normed[:, :-1] / y_normed[:, -1:]) * ballast
    max_rel = float((carrier_recovered - carrier_in.to(torch.float64)).abs().max().item() / max(float(carrier_in.abs().max().item()), 1e-12))
    return carrier_recovered, max_rel


def stationary_deadzone_key(s_norm: torch.Tensor, tau: float = 0.25, b_full: float = 0.65) -> torch.Tensor:
    """N3's Stationary SO(2)-Invariant Dead-Zone Key eta_{tau, B}(s_t) = clip((s_t - tau)/(B - tau), 0, 1)."""
    return torch.clamp((s_norm - tau) / (b_full - tau), min=0.0, max=1.0)


def run_cert5_round2_cross_track_reconciliation(geom: dict) -> dict:
    """Round 2 Cross-Track Numerical Proof Suite (N1 x N2 x N3 x N4)."""
    rng = np.random.default_rng(5005)
    u_em = geom["u_em"]
    u_prose = geom["u_prose"]
    g1, g2, g3, gf = geom["gamma_l1"], geom["gamma_l2"], geom["gamma_l3"], geom["gamma_final"]
    m_cand = geom["m_cand"]
    b_align = geom["b_align_gamma"]
    final_norm = Qwen35RMSNorm(gf)
    e = [u_em[:, i] for i in range(28)]
    c0 = 1.0

    # --- Build N1 Circuit A (39 neurons, 2L) and Circuit B (64 neurons, 3L) ---
    ts0, ws0, _ = build_fermi_dirac_nodes(0.0, 7, c0 / 2.6, c0 / 0.55, mode="trapezoid")
    ts2, ws2, _ = build_fermi_dirac_nodes(2.0, 8, c0 / 6.2, c0 / 0.33, mode="trapezoid")
    ts2_b, ws2_b, _ = build_fermi_dirac_nodes(2.0, 7, c0 / 6.2, c0 / 0.33, mode="trapezoid")
    ts2_a1, ws2_a1, _ = build_fermi_dirac_nodes(2.0, 6, c0 / 2.6, c0 / 0.55, mode="trapezoid")
    ts3_a2, ws3_a2, _ = build_fermi_dirac_nodes(3.0, 6, c0 / 2.6, c0 / 0.55, mode="trapezoid")
    knots, alphas, w_log = _fit_univariate_swiglu_log(k_knots=4)

    tau_lin = 3.55
    kappa_scratch = 8.0
    kappa_2 = 4.0

    l1_a = CompiledSwiGLULayer(
        g1,
        *stack_blocks([
            compile_complex_mul_neurons(e[3], e[4], e[0], e[1], e[7], e[8], g1, scale=kappa_scratch, method="karatsuba6"),
            compile_antipodal_pair(e[5], e[2], kappa_scratch * e[9], g1),
            compile_fermi_dirac_operator(e[6], e[6], e[10], g1, ts0, ws0, scale=1.0),
        ]),
    )
    l2_a = CompiledSwiGLULayer(
        g2,
        *stack_blocks([
            compile_fermi_dirac_operator(e[10], e[7 + k], (tau_lin * (c0 ** 2) / kappa_scratch) * b_align[:, k], g2, ts2, ws2, scale=1.0)
            for k in range(3)
        ]),
    )
    l2_b = CompiledSwiGLULayer(
        g2,
        *stack_blocks(
            [
                compile_fermi_dirac_operator(e[10], e[7 + k], (kappa_2 * (c0 ** 2) / kappa_scratch) * e[11 + k], g2, ts2_b, ws2_b, scale=1.0)
                for k in range(3)
            ]
            + [
                compile_fermi_dirac_operator(e[6], e[6], (kappa_2 * c0) * e[14], g2, ts2_a1, ws2_a1, scale=1.0),
                compile_fermi_dirac_operator(e[6], e[6], (c0 ** 2) * e[15], g2, ts3_a2, ws3_a2, scale=1.0),
            ]
        ),
    )
    inv_g3 = 1.0 / g3
    wg3_list, wu3_list, wd3_list = [], [], []
    f_real_t = torch.from_numpy(F_REAL_NP)
    for c_idx in range(4):
        p_c_vec = 0.25 * e[14] + sum(float(F_REAL_NP[c_idx, k]) * e[11 + k] for k in range(3))
        out_dir_c = b_align @ f_real_t[c_idx]
        for m_idx in range(len(knots)):
            gate_res = (float(alphas[m_idx]) / kappa_2) * (p_c_vec - float(knots[m_idx]) * e[14])
            wg3_list.append(gate_res * inv_g3)
            wu3_list.append(e[15] * inv_g3)
            wd3_list.append(float(w_log[m_idx]) * out_dir_c)
    l3_b = CompiledSwiGLULayer(g3, torch.stack(wg3_list, dim=0), torch.stack(wu3_list, dim=0), torch.stack(wd3_list, dim=1))

    # Wrap into FP32 Sparse Additive Overlays (N4 Theorem N4.2 + N1)
    l1_a_ov = FP32OverlaySwiGLULayer(l1_a)
    l2_a_ov = FP32OverlaySwiGLULayer(l2_a)
    l2_b_ov = FP32OverlaySwiGLULayer(l2_b)
    l3_b_ov = FP32OverlaySwiGLULayer(l3_b)

    # =========================================================================
    # 5A. Lemma N1.5: Exact Finite-M Fermi-Dirac Small-Noise Gain Law & 6-Regime Audit
    # =========================================================================
    g_gains = {
        "G0_M7": float(0.5 * np.sum(ws0 * ts0)),
        "G2_circuitA_M8": float(0.5 * np.sum(ws2 * ts2)),
        "G2_circuitB_M7": float(0.5 * np.sum(ws2_b * ts2_b)),
        "G2_anc_M6": float(0.5 * np.sum(ws2_a1 * ts2_a1)),
        "G3_anc_M6": float(0.5 * np.sum(ws3_a2 * ts3_a2)),
        "t_max_q2_circuitA": float(np.max(ts2)),
    }
    # Verify exact asymmetric negative half-line envelope:
    # For eta = -delta < 0, G_2 * delta * |v| <= |F_2(-delta, v)| <= 2 * G_2 * delta * |v|
    g2_val = g_gains["G2_circuitA_M8"]
    neg_halfline_checks = {}
    for delta in (1e-5, 1e-4, 1e-3, 5e-3, 2e-2):
        v_probe = 1e-3
        u_pos = ts2 * delta  # -t_m * (-delta) = +t_m * delta > 0
        f_neg = float(np.sum(ws2 * (u_pos / (1.0 + np.exp(-np.clip(u_pos, -80.0, 80.0)))) * (-v_probe)))
        lo_bound = g2_val * delta * abs(v_probe)
        hi_bound = 2.0 * g2_val * delta * abs(v_probe)
        assert lo_bound * (1.0 - 1e-12) <= abs(f_neg) <= hi_bound * (1.0 + 1e-12), (
            f"Negative half-line envelope failed at delta={delta}: {abs(f_neg)} not in [{lo_bound}, {hi_bound}]"
        )
        # Contrast with positive side eta = +delta > 0 at larger delta (e.g. 2e-2) vs small delta (1e-5)
        u_neg = -ts2 * delta
        f_pos = float(np.sum(ws2 * (u_neg / (1.0 + np.exp(-np.clip(u_neg, -80.0, 80.0)))) * (-v_probe)))
        neg_halfline_checks[f"delta={delta}"] = {
            "abs_F2_neg_eta": abs(f_neg),
            "abs_F2_pos_eta": abs(f_pos),
            "G2_delta_v_lower": lo_bound,
            "2G2_delta_v_upper": hi_bound,
            "neg_over_pos_ratio": float(abs(f_neg) / max(abs(f_pos), 1e-30)),
        }

    # 6-Regime Cross-Track Prose / Mixed-Turn Safety Audit (B = 512 tokens, d = 2560)
    b_audit = 512
    s_audit = rng.uniform(0.8, 2.2, size=b_audit)
    h_prose_5a = sample_prose_batch(u_prose, s_audit, rng)  # (512, 2560) FP64
    base_lg_fp64 = final_norm(h_prose_5a) @ m_cand.T
    base_lg_bf16 = final_norm(h_prose_5a.to(torch.bfloat16)).to(torch.float64) @ m_cand.T

    def _kl_digits(lg_p: torch.Tensor, lg_q: torch.Tensor) -> Tuple[float, float]:
        p = F.softmax(lg_p, dim=-1)
        log_p = F.log_softmax(lg_p, dim=-1)
        log_q = F.log_softmax(lg_q, dim=-1)
        kl = (p * (log_p - log_q)).sum(dim=-1)
        return float(kl.mean().item()), float(kl.max().item())

    # Regime 1: N2 FP32 intrinsic trigger leak (eta = 1.02e-8, v = 1.02e-8) entering N1 FP32 Overlay
    leak_n2 = 1.02e-8
    # Inject both signs (50% negative to stress the negative half-line!)
    signs_n2 = torch.from_numpy(rng.choice([-1.0, 1.0], size=(b_audit, 1))).to(torch.float64)
    h_r1 = (
        h_prose_5a
        + (signs_n2 * leak_n2) * e[6][None, :]
        + (signs_n2 * leak_n2) * (e[0] + e[3] + e[7])[None, :]
    ).to(torch.float32)
    x_r1 = l2_a_ov(l1_a_ov(h_r1, g_em=1), g_em=1)
    lg_r1 = final_norm(x_r1).to(torch.float64) @ m_cand.T
    r1_max_dz = float((lg_r1 - (final_norm(h_r1).to(torch.float64) @ m_cand.T)).abs().max().item())
    r1_kl_mean, r1_kl_max = _kl_digits(final_norm(h_r1).to(torch.float64) @ m_cand.T, lg_r1)

    # Regime 2: Zero EM input (h_EM = 0!), native BF16 Fermi-Dirac (ungated) -> EXPLODES!
    h_bf16 = h_prose_5a.to(torch.bfloat16)
    x_r2 = l2_a(l1_a(h_bf16, g_em=1), g_em=1)
    lg_r2 = final_norm(x_r2).to(torch.float64) @ m_cand.T
    r2_max_dz = float((lg_r2 - base_lg_bf16).abs().max().item())
    r2_kl_mean, r2_kl_max = _kl_digits(base_lg_bf16, lg_r2)

    # Regime 3a (CRITICAL DISCOVERY — Theorem N1.5B!): Zero EM input (h_EM = 0!), BF16 host residual
    # + UNGATED FP32 Overlay STILL EXPLODES (~1.06 logits!) because BF16 activation quantization
    # noise delta_act ~ 0.00225*s projects isotropically onto V_EM and triggers Lemma N1.5's
    # negative-half-line slope 2*G_2!
    x_r3a = l2_a_ov(l1_a_ov(h_bf16, g_em=1), g_em=1)
    lg_r3a = final_norm(x_r3a).to(torch.float64) @ m_cand.T
    r3a_max_dz = float((lg_r3a - base_lg_bf16).abs().max().item())
    r3a_kl_mean, r3a_kl_max = _kl_digits(base_lg_bf16, lg_r3a)
    u_in_7 = u_em[:, :7].to(torch.float32)
    bf16_act_leak_rms = float((h_bf16.to(torch.float32) @ u_in_7).pow(2).mean().sqrt().item())
    bf16_act_subspace_norm_max = float((h_bf16.to(torch.float32) @ u_in_7).norm(dim=-1).max().item())

    # Regime 3b: Same BF16 host residual + FP32 Overlay + N3 Stationary Dead-Zone Key (tau=0.25 >> 0.015)
    # -> EXACT BITWISE 0.0! (Crushes BF16 activation quantization floor completely!)
    s_norm_bf16_pure = (h_bf16.to(torch.float32) @ u_in_7).norm(dim=-1)
    key_bf16_pure = stationary_deadzone_key(s_norm_bf16_pure, tau=0.25, b_full=0.65)
    x_r3b = l2_a_ov(l1_a_ov(h_bf16, g_em=1, key_gate=key_bf16_pure), g_em=1, key_gate=key_bf16_pure)
    lg_r3b = final_norm(x_r3b).to(torch.float64) @ m_cand.T
    r3b_max_dz = float((lg_r3b - base_lg_bf16).abs().max().item())
    r3b_kl_mean, r3b_kl_max = _kl_digits(base_lg_bf16, lg_r3b)

    # Regime 3c: Compare FP32 host residual + UNGATED FP32 Overlay (proves activation precision is the root cause!)
    h_fp32 = h_prose_5a.to(torch.float32)
    base_lg_fp32 = final_norm(h_fp32).to(torch.float64) @ m_cand.T
    x_r3c = l2_a_ov(l1_a_ov(h_fp32, g_em=1), g_em=1)
    lg_r3c = final_norm(x_r3c).to(torch.float64) @ m_cand.T
    r3c_max_dz = float((lg_r3c - base_lg_fp32).abs().max().item())
    r3c_kl_mean, r3c_kl_max = _kl_digits(base_lg_fp32, lg_r3c)

    # Regime 4 & 5: Mixed-turn prose (g_EM = 1!) hit by N3 heavy-tailed Student-t (df=4.5) Zipf bursts
    # Model imperfect K-FAC nullspace leakage along V_in slots: eps_t ~ 0.008 * Student-t(df=4.5)
    # Note: 50% negative anchor sign! On extreme tail tokens, |eps_t| reaches ~0.04-0.08.
    t_draws = rng.standard_t(df=4.5, size=(b_audit, 7))
    eps_zipf = 0.008 * np.clip(t_draws, -10.0, 10.0)
    h_zipf_leak = sum(torch.from_numpy(eps_zipf[:, j])[:, None] * e[j][None, :] for j in range(7))
    h_mix_prose = (h_prose_5a + h_zipf_leak).to(torch.bfloat16)
    base_mix_lg = final_norm(h_mix_prose).to(torch.float64) @ m_cand.T

    # Regime 4: Mixed-turn Zipf burst into UNGATED FP32 Overlay Fermi-Dirac -> EXPLODES!
    x_r4 = l2_a_ov(l1_a_ov(h_mix_prose, g_em=1), g_em=1)
    lg_r4 = final_norm(x_r4).to(torch.float64) @ m_cand.T
    r4_max_dz = float((lg_r4 - base_mix_lg).abs().max().item())
    r4_kl_mean, r4_kl_max = _kl_digits(base_mix_lg, lg_r4)

    # Regime 5: Mixed-turn Zipf burst with COMBINED SAFEGUARD:
    #   FP32 Overlay + N3 Stationary SO(2)-Invariant Dead-Zone Key eta_{tau=0.25, B=0.65}(||U_in^T h||_2)
    s_norm_mix = (h_mix_prose.to(torch.float32) @ u_in_7).norm(dim=-1)
    key_mix = stationary_deadzone_key(s_norm_mix, tau=0.25, b_full=0.65)
    x_r5 = l2_a_ov(l1_a_ov(h_mix_prose, g_em=1, key_gate=key_mix), g_em=1, key_gate=key_mix)
    lg_r5 = final_norm(x_r5).to(torch.float64) @ m_cand.T
    r5_max_dz = float((lg_r5 - base_mix_lg).abs().max().item())
    r5_kl_mean, r5_kl_max = _kl_digits(base_mix_lg, lg_r5)
    r5_bulk_zero_frac = float((key_mix == 0.0).to(torch.float64).mean().item())

    # Regime 6: Hard structural g_EM = 0 on BF16 prose -> bitwise 0.0
    x_r6 = l2_a_ov(l1_a_ov(h_bf16, g_em=0), g_em=0)
    r6_max_dz = float((final_norm(x_r6).to(torch.float64) @ m_cand.T - base_lg_bf16).abs().max().item())

    assert r1_max_dz < 1e-6 and r1_kl_max < 1e-12, f"N2 FP32 trigger leak should be safe in FP32 host+overlay: {r1_max_dz}"
    assert r2_max_dz > 0.25, f"Expected native BF16 ungated Fermi-Dirac instability >0.25, got {r2_max_dz}"
    assert r3a_max_dz > 0.25, f"Expected BF16-host ungated FP32-overlay activation-floor instability >0.25, got {r3a_max_dz}"
    assert r3b_max_dz == 0.0, f"FP32 overlay + Dead-Zone Key must crush BF16 zero-leak drift to exact 0.0, got {r3b_max_dz}"
    assert r3c_max_dz < 1e-7, f"FP32 host + FP32 overlay must be <1e-7, got {r3c_max_dz}"
    assert r4_max_dz > 0.5 and r4_kl_mean > 1e-3, f"Expected ungated Zipf mixed-turn explosion, got dz={r4_max_dz}, kl={r4_kl_mean}"
    assert r5_kl_mean < 1e-6 and r5_kl_max < 1e-4, f"Combined safeguard should keep MixKL << 1e-3, got mean={r5_kl_mean}, max={r5_kl_max}"
    assert r6_max_dz == 0.0, "g_EM=0 must be bitwise exact 0.0"

    safety_6regimes = {
        "R1_n2_fp32_leak_into_fp32_host_and_overlay": {"max_dz": r1_max_dz, "kl_mean": r1_kl_mean, "kl_max": r1_kl_max, "verdict": "SAFE"},
        "R2_zero_leak_native_bf16_ungated": {"max_dz": r2_max_dz, "kl_mean": r2_kl_mean, "kl_max": r2_kl_max, "verdict": "EXPLODES"},
        "R3a_zero_leak_bf16_host_ungated_fp32_overlay": {
            "max_dz": r3a_max_dz,
            "kl_mean": r3a_kl_mean,
            "kl_max": r3a_kl_max,
            "bf16_act_per_slot_leak_rms": bf16_act_leak_rms,
            "bf16_act_7d_subspace_norm_max": bf16_act_subspace_norm_max,
            "verdict": "EXPLODES_VIA_BF16_ACT_FLOOR",
        },
        "R3b_zero_leak_bf16_host_fp32_overlay_plus_deadzone_key": {
            "max_dz": r3b_max_dz,
            "kl_mean": r3b_kl_mean,
            "kl_max": r3b_kl_max,
            "verdict": "BITWISE_ZERO",
        },
        "R3c_zero_leak_fp32_host_ungated_fp32_overlay": {
            "max_dz": r3c_max_dz,
            "kl_mean": r3c_kl_mean,
            "kl_max": r3c_kl_max,
            "verdict": "SAFE",
        },
        "R4_mixkl_zipf_t4p5_ungated_fp32_overlay": {"max_dz": r4_max_dz, "kl_mean": r4_kl_mean, "kl_max": r4_kl_max, "verdict": "EXPLODES"},
        "R5_mixkl_zipf_t4p5_combined_safeguard": {
            "max_dz": r5_max_dz,
            "kl_mean": r5_kl_mean,
            "kl_max": r5_kl_max,
            "bulk_exact_zero_frac": r5_bulk_zero_frac,
            "max_zipf_subspace_norm": float(s_norm_mix.max().item()),
            "verdict": "PASS_MIXKL",
        },
        "R6_pure_prose_hard_g_em_0": {"max_dz": r6_max_dz, "pure_kl": 0.0, "verdict": "BITWISE_EXACT"},
    }

    # =========================================================================
    # 5B. N2 Candidate-Orthogonal Bus D_{gamma_f} Audit & End-to-End 4-Track Pipeline
    # =========================================================================
    # Sub-check 5B.1: Uncompensated N2 bus (perp row(M_cand)) vs N1 upgraded bus (perp row(M_cand D_{gamma_f}))
    u_mcand_np, _ = np.linalg.qr(m_cand.T.numpy(), mode="reduced")
    u_mcand_3 = u_mcand_np[:, :3]  # row(M_cand), OMITTING D_{gamma_f}!
    # Build 16D naive bus orthogonal to u_prose and row(M_cand), NOT to row(M_cand D_{gamma_f})
    raw_bus = rng.standard_normal((D_MODEL, 16))
    u_p_np = u_prose.numpy()
    # Project out u_prose and row(M_cand)
    basis_naive = np.linalg.qr(np.hstack([u_p_np, u_mcand_3]), mode="reduced")[0]
    raw_bus_naive = raw_bus - basis_naive @ (basis_naive.T @ raw_bus)
    u_bus_naive = torch.from_numpy(np.linalg.qr(raw_bus_naive, mode="reduced")[0]).to(torch.float64)
    # Verify naive bus truly satisfies N2's Round-1 orthogonality conditions:
    assert float((m_cand @ u_bus_naive).abs().max().item()) < 1e-13
    assert float((u_prose.T @ u_bus_naive).abs().max().item()) < 1e-13

    # Upgraded bus u_bus_gamma is u_em[:, :16] (orthogonal to u_prose AND row(M_cand D_{gamma_f}))
    u_bus_gamma = u_em[:, :16]
    assert float(((m_cand * gf[None, :]) @ u_bus_gamma).abs().max().item()) < 1e-13

    # Inject unit-amplitude EM bus payload on 128 prose vectors and measure spurious digit logits
    c_bus = torch.from_numpy(rng.standard_normal((128, 16)))
    c_bus = c_bus / c_bus.norm(dim=-1, keepdim=True) * 4.0  # realistic O(4) carrier norm
    h_p_bus = sample_prose_batch(u_prose, rng.uniform(0.9, 1.8, size=128), rng)
    base_bus_lg = final_norm(h_p_bus) @ m_cand.T
    naive_bus_lg = final_norm(h_p_bus + c_bus @ u_bus_naive.T) @ m_cand.T
    gamma_bus_lg = final_norm(h_p_bus + c_bus @ u_bus_gamma.T) @ m_cand.T
    naive_bus_leak_dz = float((naive_bus_lg - base_bus_lg).abs().max().item())
    gamma_bus_leak_dz = float((gamma_bus_lg - base_bus_lg).abs().max().item())
    naive_bus_kl_mean, _ = _kl_digits(base_bus_lg, naive_bus_lg)
    gamma_bus_kl_mean, _ = _kl_digits(base_bus_lg, gamma_bus_lg)
    assert naive_bus_leak_dz > 0.01, f"Expected nonzero D_gamma_f leak from naive N2 bus, got {naive_bus_leak_dz}"
    assert gamma_bus_leak_dz < 1e-13, f"Expected exact zero leak from D_gamma_f-upgraded bus, got {gamma_bus_leak_dz}"

    # Sub-check 5B.2: End-to-End 4-Track Converged Pipeline on Real Locked Stage 3D G1 & G5
    # N4 264-param Orthogonal Linear Carrier (+1 anchor c0) -> N2 r_hd=2 Shadow GDN Ballast Bus (L_gap=64)
    # -> Host (BF16 OR Mixed-NF4 block-64 host MLP) -> N1 K=64 Circuit B FP32 Overlay + N3 Stationary Key -> B_align^(gamma_f)
    enc, s17_sha = load_s17_encoder(REPO_ROOT)
    kern = PhysicalKernelS17(enc.table)

    # Simulate frozen host MLP weights in Mixed-NF4 (block-64 quantized, d=2560, 128-neuron prose-only slice)
    # Strictly inside V_prose to model real frozen host MLP operating on prose while N1 runs in FP32 overlay!
    w_host_up = torch.from_numpy(rng.normal(scale=0.04, size=(128, D_MODEL))).to(torch.float32)
    w_host_down = torch.from_numpy(rng.normal(scale=0.04, size=(D_MODEL, 128))).to(torch.float32)
    # Project host frozen MLP onto V_prose so its ideal prose trajectory stays in V_prose,
    # while NF4 block-64 quantization introduces realistic cross-coordinate quantization noise (~1e-3)!
    u_p_f32 = u_prose.to(torch.float32)
    w_host_up_p = (w_host_up @ u_p_f32) @ u_p_f32.T
    w_host_down_p = u_p_f32 @ (u_p_f32.T @ w_host_down)
    w_host_up_nf4 = simulate_nf4_block64_quantize(w_host_up_p)
    w_host_down_nf4 = simulate_nf4_block64_quantize(w_host_down_p)

    def _frozen_nf4_host_mlp_delta(x_bf16: torch.Tensor) -> torch.Tensor:
        # Host MLP with block-64 NF4 quantized weights (dequantized to BF16 during matmul, per bitsandbytes)
        xf = x_bf16.to(torch.bfloat16)
        h = F.silu(F.linear(xf, w_host_up_nf4.to(torch.bfloat16)))
        return F.linear(h, w_host_down_nf4.to(torch.bfloat16)) * 0.05

    e2e_regimes = [
        ("G1_main_extrap_both", BLIND_CELLS["G1_main_extrap_both"].namespace, 0.40, 2, 2, 4.0, 8),
        ("G5_half_turn_pi", 950406, math.pi / 4.0, 2, 2, 4.0, 4),
    ]
    e2e_results = {}
    for rname, ns, abs_om, h, ell, tp, n_blk in e2e_regimes:
        bmap, _ = generate_paired_cfo_blocks(ns, 0, n_blk, h, ell, tp, (-abs_om, abs_om), snr_db=2.0, sigma_phi=0.10)
        p_all, y_all, a1_all, lam_all = [], [], [], []
        for om, mb in bmap.items():
            z1 = torch.from_numpy(np.asarray(enc.encode(mb.y1), dtype=np.int64))
            inner1 = z1 % 4
            a1_c = kern.prior_canon[inner1].numpy().astype(np.float64)
            theta_val = om * float(h + ell)
            var_val = (0.10 ** 2) * float(h + ell)
            k_mat = kern.kernel(
                torch.tensor([theta_val], dtype=torch.float32),
                torch.tensor([var_val], dtype=torch.float32),
                torch.tensor([2.0], dtype=torch.float32),
            )[0].numpy().astype(np.float64)
            k_hat = F_REAL_NP.T @ k_mat.T @ F_REAL_NP
            l1r, l1i, l2 = float(k_hat[0, 0]), float(k_hat[1, 0]), float(k_hat[2, 2])
            p_c = a1_c @ k_mat
            y3_raw = sector_of_complex(matched_sum(mb.y3_repeat)).astype(np.int64)
            lbl_c = (y3_raw - (z1.numpy() // 4)) % 4
            p_all.append(p_c)
            y_all.append(lbl_c)
            a1_all.append(a1_c @ F_REAL_NP)
            lam_all.append(np.repeat([[l1r, l1i, l2]], len(z1), axis=0))

        P_ex = np.concatenate(p_all, axis=0)
        Y_ex = np.concatenate(y_all, axis=0)
        A1_f = np.concatenate(a1_all, axis=0)
        Lam = np.concatenate(lam_all, axis=0)
        n_tot = len(Y_ex)
        oracle_acc = float(np.mean(P_ex.argmax(-1) == Y_ex))
        oracle_nll = float(-np.mean(np.log(np.clip(P_ex[np.arange(n_tot), Y_ex], 1e-12, 1.0))))

        # Step 1 (N4 Carrier): 7D physical sufficient stats + constant anchor c0=1.0
        carrier_in = torch.from_numpy(
            np.column_stack([A1_f, Lam, np.full(n_tot, c0, dtype=np.float64)])
        )  # (N, 7)

        # Step 2 (N2 Shadow GDN Ballast Bus across L_gap=64 prose tokens):
        carrier_bus, bus_rel_err = run_shadow_gdn_ballast_bus(carrier_in, l_gap=64, seed=5052 + n_blk)
        assert bus_rel_err < 1e-5, f"Shadow GDN bus rel err too large: {bus_rel_err}"

        # Inject delivered bus coordinates into upgraded subspace V_in = span(e[0..6])
        x_em_bus = sum(carrier_bus[:, j : j + 1] * e[j][None, :] for j in range(7))
        s1_r = rng.uniform(0.7, 2.2, size=n_tot)
        s2_r = rng.uniform(0.7, 2.2, size=n_tot)
        h_p1 = sample_prose_batch(u_prose, s1_r, rng)
        h_p2 = sample_prose_batch(u_prose, s2_r, rng)

        # Check N3 Stationary Key on ALL OOD EM queries (must be identically 1.0!)
        s_norm_em = ((h_p1 + x_em_bus).to(torch.float32) @ u_in_7).norm(dim=-1)
        key_em = stationary_deadzone_key(s_norm_em, tau=0.25, b_full=0.65)
        assert float(key_em.min().item()) == 1.0, f"Stationary key must be identically 1.0 on EM queries, min={key_em.min()}"

        host_modes = {}
        for hmode in ("bf16_host_fp32_overlay", "mixed_nf4_host_fp32_overlay"):
            x1_host = (h_p1 + x_em_bus).to(torch.bfloat16)
            if hmode == "mixed_nf4_host_fp32_overlay":
                # Add realistic NF4 block-64 frozen host MLP residual update (contains cross-subspace NF4 ripple!)
                x1_host = (x1_host.to(torch.float32) + _frozen_nf4_host_mlp_delta(x1_host).to(torch.float32)).to(torch.bfloat16)
            d1 = l1_a_ov.delta_fp32(x1_host, key_gate=key_em).to(torch.bfloat16)
            x2_host = (h_p2 + carrier_bus[:, 6:7] * e[6][None, :]).to(torch.bfloat16)
            if hmode == "mixed_nf4_host_fp32_overlay":
                x2_host = (x2_host.to(torch.float32) + _frozen_nf4_host_mlp_delta(x2_host).to(torch.float32)).to(torch.bfloat16)
            x2_in = (x2_host.to(torch.float32) + d1.to(torch.float32)).to(torch.bfloat16)
            d2 = l2_b_ov.delta_fp32(x2_in, key_gate=key_em).to(torch.bfloat16)
            x3_in = (x2_host.to(torch.float32) + d2.to(torch.float32)).to(torch.bfloat16)
            x_fin = l3_b_ov(x3_in, g_em=1, key_gate=key_em)
            lg_fin = (final_norm(x_fin).to(torch.float64) @ m_cand.T).numpy()
            probs = F.softmax(torch.from_numpy(lg_fin), dim=-1).numpy()
            acc = float(np.mean(probs.argmax(-1) == Y_ex))
            agree = float(np.mean(probs.argmax(-1) == P_ex.argmax(-1)))
            nll = float(-np.mean(np.log(np.clip(probs[np.arange(n_tot), Y_ex], 1e-12, 1.0))))
            kl = float(np.mean(np.sum(P_ex * (np.log(np.clip(P_ex, 1e-12, 1.0)) - np.log(np.clip(probs, 1e-12, 1.0))), axis=-1)))
            host_modes[hmode] = {
                "acc": acc,
                "oracle_agree": agree,
                "nll": nll,
                "kl_to_bayes_oracle": kl,
            }
            assert agree >= 0.995, f"{rname}/{hmode}: oracle agree {agree} < 0.995"
            assert kl < 5e-3, f"{rname}/{hmode}: KL {kl} >= 5e-3"

        e2e_results[rname] = {
            "n_rows": n_tot,
            "oracle_acc": oracle_acc,
            "oracle_nll": oracle_nll,
            "shadow_gdn_l64_max_rel_err": bus_rel_err,
            "stationary_key_min_on_em": float(key_em.min().item()),
            "host_modes": host_modes,
        }

    # =========================================================================
    # 5C. Unified 4-Track Parameter Ledger (Qwen3.5-4B BF16 & Qwen3.8-27B Mixed-NF4)
    # =========================================================================
    # Exact parameter budget combining:
    # - Phase-A Equivariant Connector (frozen or regime-2)
    # - N1 Compiled FP32 Overlay SwiGLU (Circuit A K=39 or Circuit B K=64; 3*K*d frozen buffers + K trainable gain/bias calibration scalars)
    # - N2 r_hd=2 Shadow GDN Ballast Bus + NoPE FA (or 18-dim subspace-sparse synapses)
    # - N3/N4 Bounded-Spectrum SpecCayley (r_c=4, 9L on 4B / 14L on 27B) + Folded P_perp FA/DN LoRA
    # - N3 Stationary SO(2)-Invariant Keyed Dead-Zone (d_key=16 subspace + 2 thresholds)
    # - N4 Regime-2 Orthogonal Linear Carrier (264 params on 4B, 268 on 27B) + Gated B_align^(gamma_f) readout
    unified_specs = {}
    for model_name, cfg in (
        (
            "Qwen3.5-4B_BF16_Overlay",
            {
                "d": 2560,
                "host_params": 4_205_751_296,
                "phase_a_connector": 488_400,
                "spec_cayley": 294_948,        # 9L, r_c=4 + 36 scale scalars
                "fa_bridge_lora_r6": 414_720,  # 5L folded P_perp
                "dn_bridge_lora": 291_840,     # 4L KV r=6 + unmasked in_proj_a/b r=4
                "regime2_carrier": 264,        # 16x16 + 8 slow-head bias (+1 anchor slot in 16D carrier)
                "stationary_key_and_readout": 30_756,  # gated B_align^(gamma_f) + Schur + 2 dead-zone scalars
                "n1_circuit_a_buffers": 3 * 39 * 2560,  # 299,520 FP32 compiled overlay buffers
                "n1_circuit_b_buffers": 3 * 64 * 2560,  # 491,520 FP32 compiled overlay buffers
                "n1_calibration_scalars": 64,           # trainable per-neuron gain/anchor scalars
                "n2_shadow_gdn_sparse": 10_604,         # 18-dim subspace-sparse GDN+NoPE bus
                "tier_p1_cap": 2_184_964,
            },
        ),
        (
            "Qwen3.8-27B_MixedNF4_Overlay",
            {
                "d": 5120,
                "host_params": 26_895_998_464,
                "phase_a_connector": 823_760,
                "spec_cayley": 688_184,        # 14L, r_c=4 + 56 scale scalars
                "fa_bridge_lora_r6": 1_130_496,  # 8L folded P_perp
                "dn_bridge_lora": 727_296,     # 6L KV r=6 + unmasked gates r=4
                "regime2_carrier": 268,        # 16x16 + 12 slow-head bias
                "stationary_key_and_readout": 61_476,
                "n1_circuit_a_buffers": 3 * 39 * 5120,  # 599,040 FP32 compiled overlay buffers
                "n1_circuit_b_buffers": 3 * 64 * 5120,  # 983,040 FP32 compiled overlay buffers
                "n1_calibration_scalars": 64,
                "n2_shadow_gdn_sparse": 18_284,
                "tier_p1_cap": None,
            },
        ),
    ):
        trainable_adapter = (
            cfg["spec_cayley"]
            + cfg["fa_bridge_lora_r6"]
            + cfg["dn_bridge_lora"]
            + cfg["regime2_carrier"]
            + cfg["stationary_key_and_readout"]
            + cfg["n1_calibration_scalars"]
        )
        cum_trainable = cfg["phase_a_connector"] + trainable_adapter
        cum_with_circ_a_buf = cum_trainable + cfg["n1_circuit_a_buffers"] + cfg["n2_shadow_gdn_sparse"]
        cum_with_circ_b_buf = cum_trainable + cfg["n1_circuit_b_buffers"] + cfg["n2_shadow_gdn_sparse"]
        entry = {
            "trainable_adapter_only": trainable_adapter,
            "cumulative_trainable_incl_connector": cum_trainable,
            "cumulative_incl_circuitA_and_n2_buffers": cum_with_circ_a_buf,
            "cumulative_incl_circuitB_and_n2_buffers": cum_with_circ_b_buf,
            "pct_of_host_trainable": 100.0 * cum_trainable / cfg["host_params"],
            "pct_of_host_total_incl_circuitB": 100.0 * cum_with_circ_b_buf / cfg["host_params"],
        }
        if cfg["tier_p1_cap"] is not None:
            entry["pct_of_tier_p1_trainable"] = 100.0 * cum_trainable / cfg["tier_p1_cap"]
            entry["pct_of_tier_p1_incl_circuitA_buffers"] = 100.0 * cum_with_circ_a_buf / cfg["tier_p1_cap"]
            entry["pct_of_tier_p1_incl_circuitB_buffers"] = 100.0 * cum_with_circ_b_buf / cfg["tier_p1_cap"]
            assert cum_trainable <= cfg["tier_p1_cap"]
            assert cum_with_circ_a_buf <= cfg["tier_p1_cap"]
            assert cum_with_circ_b_buf <= cfg["tier_p1_cap"]
        unified_specs[model_name] = entry

    return {
        "lemma_n1_5_fermi_dirac_small_noise_gains": g_gains,
        "lemma_n1_5_negative_halfline_checks": neg_halfline_checks,
        "safety_audit_6_regimes": safety_6regimes,
        "n2_bus_gamma_f_upgrade_audit": {
            "naive_n2_bus_spurious_logit_max_dz": naive_bus_leak_dz,
            "naive_n2_bus_spurious_kl_mean": naive_bus_kl_mean,
            "upgraded_gamma_f_bus_logit_max_dz": gamma_bus_leak_dz,
            "upgraded_gamma_f_bus_kl_mean": gamma_bus_kl_mean,
            "suppression_ratio": float(naive_bus_leak_dz / max(gamma_bus_leak_dz, 1e-30)),
        },
        "e2e_4track_converged_pipeline_s17": e2e_results,
        "unified_param_ledger_4b_27b": unified_specs,
    }


def main() -> int:
    t0 = time.perf_counter()
    torch.manual_seed(20261101)
    np.random.seed(20261101)

    geom = make_orthogonal_host_geometry(d=D_MODEL, k_prose=512, n_slots=32, seed=20261101)
    cert1 = run_cert1_minimal_neurons(geom)
    cert2 = run_cert2_multilayer_algebra(geom)
    cert3 = run_cert3_rmsnorm_self_normalizing(geom)
    cert4 = run_cert4_real_g1_and_audits(geom)
    cert5 = run_cert5_round2_cross_track_reconciliation(geom)

    elapsed = time.perf_counter() - t0
    report = {
        "track": "N1_round2_converged_swiglu_cross_track",
        "elapsed_s": round(elapsed, 3),
        "host_geometry": {
            "d_model": D_MODEL,
            "d_ff_4b": D_FF_4B,
            "n_layers_4b": N_LAYERS_4B,
            "total_mlp_neurons_4b": D_FF_4B * N_LAYERS_4B,
            "subspace_orth_err": geom["subspace_orth_err"],
            "b_align_gamma_err": geom["b_align_gamma_err"],
            "b_align_naive_err": geom["b_align_naive_err"],
        },
        "cert1_minimal_neurons": cert1,
        "cert2_multilayer_algebra": cert2,
        "cert3_rmsnorm_self_normalizing": cert3,
        "cert4_real_g1_and_audits": cert4,
        "cert5_round2_cross_track_reconciliation": cert5,
        "all_passed": True,
    }

    RESULTS_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\n[verify_n1.py] ALL 5 CERTIFICATES PASSED in {elapsed:.2f}s -> {RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
