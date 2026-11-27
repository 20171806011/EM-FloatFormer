"""Unified N1+N2+N3+N4 Connecting-Neuron Surgery (CN-SFT) adapter for Qwen3.5.

All surgical modifications operate strictly via non-mutating FP32 sparse additive
overlays and PyTorch forward hooks so that:
  1. Frozen host weights (`host_sha256`) stay bitwise immutable across BF16 and
     Mixed-NF4 hosts (`0` in-place neuron overwrites).
  2. Structural sequence gate `g_EM == 0` bypasses every EM branch prior to any
     floating-point addition (`PureKL == 0.0`, `pure_max_logit_diff == 0.0`).
  3. Within-turn mixed prompts (`g_EM == 1`) are isolated via 64-prompt
     Dual-Space K-FAC (`d` and head-output `d_o`), strict `192`-dim NoPE FA
     LoRA masking, head-space `SpecCayley`, non-mutating `ShadowGDNBallastBridge`,
     Hard-Plateau Stationary `SO(2)`-invariant Dead-Zone + Fisher Query gating,
     and Augmented-Lagrangian Hinge control (`MixKL <= 1e-3`).
"""

from __future__ import annotations

from contextlib import contextmanager
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import nnls
from scipy.special import erf
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.stage3_ministral_cache.schema import ROOTS
from research.stage4_real_and_surgical.equi_connector import decode_c16_tail
from research.stage4_real_and_surgical.surgical_adapter import (
    QWEN4B_DN_LAYERS,
    QWEN4B_FA_LAYERS,
    QWEN4B_STOP_LAYER,
    TIER_P1_MAX_PARAMS,
    _get_decoder_layers,
    compute_complex_nullspace_projector,
)

QWEN4B_MLP_TUNE_LAYERS: Tuple[int, ...] = (15, 19, 23)

F_REAL_4X3 = torch.tensor(
    [
        [1.0 / math.sqrt(2.0), 0.0, 0.5],
        [0.0, 1.0 / math.sqrt(2.0), -0.5],
        [-1.0 / math.sqrt(2.0), 0.0, 0.5],
        [0.0, -1.0 / math.sqrt(2.0), -0.5],
    ],
    dtype=torch.float32,
)

# Strictly disjoint 64-prompt calibration corpus (zero overlap with
# GENERAL_TEXT_TRAIN_PROMPTS or GENERAL_TEXT_EVAL_PROMPTS).
GENERAL_KFAC_64_PROMPTS: Tuple[str, ...] = (
    "In digital communications over linear time-varying channels, symbol synchronization precedes carrier phase recovery when Doppler offsets exceed several radians per block.",
    "请简要说明线性代数中厄米特矩阵谱分解的充要条件，并给出一维信号滤波中的直观几何解释。",
    '{"cluster": "ap-southeast-1", "replicas": 3, "quorum": true, "checkpoint_epoch": 48, "crc32": "7e9a2c11"}',
    "def householder_reflect(v, x):\n    scale = 2.0 * sum(vi * xi for vi, xi in zip(v, x)) / max(1e-12, sum(vi * vi for vi in v))\n    return [xi - scale * vi for vi, xi in zip(v, x)]",
    "SELECT region, COUNT(DISTINCT session_id) AS active_users FROM telemetry_events WHERE status_code = 200 GROUP BY region ORDER BY active_users DESC;",
    "对于有限维希尔伯特空间上的有界线性算子，若其与典范复结构对易，则实表示天然具有偶重奇异值配对。",
    '{"job_id": "sync-904", "worker_mem_mb": 16384, "nvme_cached": false, "tags": ["bf16", "kv-cache", "audit"]}',
    "When designing an orthonormal basis in high dimensions, modified Gram-Schmidt re-orthogonalization prevents catastrophic loss of orthogonality.",
    "class CircularQueue:\n    def __init__(self, n):\n        self.data = [0.0] * n\n        self.head = 0\n    def push(self, val):\n        self.data[self.head] = val\n        self.head = (self.head + 1) % len(self.data)",
    "统计物理中的配分函数将微观能量态的指数加权求和与宏观自由能极小化原理紧密联系在一起。",
    "INSERT INTO migration_log (version, applied_at, checksum) VALUES ('2026.11.27', CURRENT_TIMESTAMP, 'f84d190e');",
    '{"parser": "strict-utf8", "max_depth": 64, "allow_comments": false, "fallback_encoding": "latin-1"}',
    "A symplectic integrator preserves phase-space volume over exponentially long integration horizons by alternating drift and kick maps.",
    "在低信噪比短帧通信场景下，相干积分增益受限于未知剩余频偏与信道相位布朗扩散速度之比。",
    "def binary_search_left(arr, target):\n    lo, hi = 0, len(arr)\n    while lo < hi:\n        mid = (lo + hi) // 2\n        if arr[mid] < target:\n            lo = mid + 1\n        else:\n            hi = mid\n    return lo",
    '{"sensor": "imu-07", "sample_hz": 1000, "bias_xyz": [-0.0012, 0.0045, -0.0008], "calibrated": true}',
    "Cholesky factorization avoids full eigendecomposition when solving positive-definite linear systems of small matrix rank.",
    "现代大语言模型残差流呈现出显著的前层放大与尾层各向异性特征，少数超大权重通道主导激活范数。",
    "CREATE INDEX CONCURRENTLY idx_orders_created ON orders (customer_id, created_at DESC) WHERE archived = FALSE;",
    "Under bfloat16 rounding with a 7-bit mantissa, adding small perturbations onto large background vectors hits sub-ULP rounding boundaries.",
    "def cosine_anneal(step, total, min_ratio=0.05):\n    t = min(1.0, max(0.0, step / max(1, total)))\n    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * t))",
    '{"router": "core-sw-02", "bgp_peers": 14, "packet_drop_ppm": 0.02, "uptime_days": 412}',
    "群表示论中的舒尔引理表明，不可约复表示之间的等变线性映射要么为零，要么为可逆纯量倍。",
    "The Woodbury matrix identity reduces rank-k orthogonal updates in a high-dimensional ambient space to a tiny 2k-by-2k solve.",
    "WITH daily_totals AS (SELECT DATE(ts) AS d, SUM(bytes) AS b FROM netflow GROUP BY DATE(ts)) SELECT AVG(b) FROM daily_totals;",
    "在非平稳噪声环境中，基于高斯-埃尔米特求积的积分计算能够比蒙特卡洛抽样更稳定地逼近缠绕正态密度。",
    '{"compiler": "llvm-19", "opt_level": "-O3", "vectorize": true, "target_arch": "x86_64-v3"}',
    "def stable_logsumexp(xs):\n    m = max(xs)\n    return m + math.log(sum(math.exp(x - m) for x in xs))",
    "Barrett and Montgomery reductions allow fast modular multiplication in finite rings without expensive hardware integer division instructions.",
    "在软件工程评审中，不可变数据契约与纯函数式状态转换有助于消除并发读写带来的竞态缺陷。",
    '{"flight": "CA-982", "altitude_ft": 35000, "mach": 0.84, "crosswind_kt": 22, "icing": false}',
    "Principal component whitening removes pairwise linear correlations across features while amplifying low-variance tail directions unless regularized.",
    "import heapq\ndef top_k_indices(vals, k):\n    return [idx for _, idx in heapq.nlargest(k, ((v, i) for i, v in enumerate(vals)))]",
    "如果一个非线性激活函数在原点处存在一阶导数不连续或斜率突变，小幅正负微扰会产生非对称响应。",
    "UPDATE inventory SET reserved = reserved + 1 WHERE sku = 'EM-QPSK-16' AND stock - reserved >= 1 RETURNING id;",
    '{"experiment": "abl-run-12", "seed": 202611, "batch": 64, "notes": "disjoint calibration split verified"}',
    "Fourier duality turns circular convolution across finite cyclic groups into pointwise complex multiplication in frequency coordinates.",
    "分布式共识协议在存在拜占庭故障或网络分区时，必须严格区分安全性与活性两项系统属性。",
    "def prefix_sums(nums):\n    out, acc = [], 0\n    for x in nums:\n        acc += x\n        out.append(acc)\n    return out",
    '{"db": "postgres-17", "wal_level": "replica", "max_connections": 200, "tls": "required"}',
    "The Mellin transform expresses inverse powers over the positive half-line via strictly positive Fermi-Dirac quadrature nodes.",
    "设计硬件友好的定点量化方案时，非对称码本会破坏输入向量反号时的线性对消恒等式。",
    "SELECT endpoint, PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY duration_ms) FROM http_spans GROUP BY endpoint;",
    '{"mesh": "tetrahedral", "vertices": 18420, "boundary_faces": 1208, "watertight": true}',
    "Cauchy-Schwarz inequalities bound the worst-case inner product between unit signal vectors and arbitrary background residuals.",
    "def lcg_stream(seed, count):\n    state = seed & 0xFFFFFFFF\n    for _ in range(count):\n        state = (1664525 * state + 1013904223) & 0xFFFFFFFF\n        yield state / 4294967296.0",
    "在多跳注意力和状态空间混合网络中，旋转位置编码高频分量在相邻两三个词元内即可发生相位翻转。",
    '{"audio": "pcm-s24le", "channels": 2, "rate_hz": 48000, "dither": "triangular"}',
    "Augmented Lagrangian methods combine explicit dual multipliers with quadratic penalty terms to enforce equality or inequality constraints cleanly.",
    "梯度检查点技术通过在前向传播中丢弃中间激活并在反向传播时按块重算，显著降低显存峰值。",
    "DELETE FROM stale_tokens WHERE expires_at < NOW() - INTERVAL '7 days' AND revoked = TRUE;",
    '{"grid": [128, 128, 64], "courant_number": 0.42, "boundary": "periodic", "stable": true}',
    "Non-negative least squares guarantees weights of identical sign, eliminating floating-point catastrophic cancellation during summation.",
    "def matrix_trace_prod(a, b):\n    return sum(a[i][j] * b[j][i] for i in range(len(a)) for j in range(len(b)))",
    "在无线接收机前端，自动增益控制环路旨在将输入信号的均方根维持在模数转换器的最优动态范围内。",
    '{"finance": "ledger-v2", "currency": "CNY", "balanced": true, "entries": 16}',
    "Markov inequality applied to fourth-order moments provides robust polynomial tail bounds even for heavy-tailed distributions.",
    "当系统存在多个独立的归一化层级时，必须设计层间不变的特征比例传递机制以避免增益漂移。",
    "SELECT a.id, COALESCE(b.tag, 'unlabeled') FROM artifacts a LEFT JOIN metadata b ON a.id = b.artifact_id;",
    '{"kernel": "linux-6.12", "scheduler": "eevdf", "numa_nodes": 2, "hugepages": "2M"}',
    "def dot_complex_pairs(x, y):\n    return sum(x[2*k]*y[2*k] + x[2*k+1]*y[2*k+1] for k in range(len(x)//2))",
    "重尾稀疏概念叠加效应会导致基于有限样本筛选出来的所谓休眠神经元在测试阶段必然触发冲突。",
    '{"vision": "disabled", "text_only": true, "max_context": 32768, "precision": "bfloat16"}',
    "Every invariant check on weights and outputs must be backed by reproducible SHA-256 checksums before downstream deployment.",
)
assert len(GENERAL_KFAC_64_PROMPTS) == 64


# =============================================================================
# RMSNorm Gamma Extraction, B_align^(gamma_f), and 3-Way Orthogonal Bus
# =============================================================================


def _get_final_norm(lm: nn.Module) -> nn.Module:
    if hasattr(lm, "model") and hasattr(lm.model, "language_model") and hasattr(lm.model.language_model, "norm"):
        return lm.model.language_model.norm
    if hasattr(lm, "model") and hasattr(lm.model, "norm"):
        return lm.model.norm
    if hasattr(lm, "norm"):
        return lm.norm
    raise AttributeError(f"Could not locate final norm layer on {type(lm)}")


def is_zero_centered_rmsnorm(norm_mod: nn.Module) -> bool:
    """Detect Qwen3.5 / Gemma zero-centered RMSNorm ((1 + weight) * norm(x))."""
    cname = type(norm_mod).__name__
    if cname in ("Qwen3_5RMSNorm", "GemmaRMSNorm", "RMSNormZeroCentered"):
        return True
    if "ZeroCentered" in cname:
        return True
    return False


def extract_rmsnorm_effective_gamma(norm_mod: nn.Module) -> torch.Tensor:
    """Extract effective FP32 multiplier gamma matching forward(x) = norm(x) * gamma."""
    w = norm_mod.weight.detach().float()
    if is_zero_centered_rmsnorm(norm_mod):
        return 1.0 + w
    return w


def safe_gamma_inv(gamma: torch.Tensor, min_abs: float = 1e-4) -> torch.Tensor:
    """Invert non-uniform RMSNorm weights while guarding near-zero entries preserving sign."""
    g = gamma.detach().float()
    signs = torch.where(g >= 0.0, torch.ones_like(g), -torch.ones_like(g))
    mag = g.abs().clamp_min(min_abs)
    return (1.0 / (signs * mag)).float()


def verify_rmsnorm_gamma(norm_mod: nn.Module, n_probe: int = 8) -> float:
    """Verify extracted effective gamma reproduces module forward within BF16/FP32 tolerance."""
    gamma = extract_rmsnorm_effective_gamma(norm_mod)
    d = gamma.shape[0]
    eps = float(getattr(norm_mod, "eps", getattr(norm_mod, "variance_epsilon", 1e-6)))
    w_dev = norm_mod.weight.device
    w_dtype = norm_mod.weight.dtype
    rng = torch.Generator().manual_seed(202611)
    x = torch.randn((n_probe, d), generator=rng, dtype=torch.float32).to(device=w_dev, dtype=w_dtype)
    with torch.no_grad():
        actual = norm_mod(x).float()
        xf = x.float()
        rms = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
        expected = ((xf / rms) * gamma.to(device=w_dev)).float()
        err = float((actual - expected).abs().max().item())
    tol = 5e-2 if w_dtype in (torch.bfloat16, torch.float16) else 1e-4
    assert err <= tol, f"RMSNorm gamma verification failed ({type(norm_mod).__name__}): max_err={err} > {tol}"
    return err


def calibrate_b_align_gamma_f(host) -> Dict[str, Any]:
    """Compute N1 gamma_f-compensated readout frame B_align^(gamma_f) = (M_cand D_{gamma_f})^+ F_real."""
    out_emb = host.lm.get_output_embeddings()
    if out_emb is None:
        out_emb = host.lm.get_input_embeddings()
    w = out_emb.weight.detach()
    if getattr(host, "tokenizer", None) is None:
        cand_w = w[:4, :].to(dtype=torch.float64, device="cpu")
        digit_ids = [0, 1, 2, 3]
    else:
        digit_ids = [int(host.tokenizer.encode(d_str, add_special_tokens=False)[-1]) for d_str in ("0", "1", "2", "3")]
        cand_w = w[digit_ids, :].to(dtype=torch.float64, device="cpu")

    norm_f = _get_final_norm(host.lm)
    verify_rmsnorm_gamma(norm_f)
    gamma_f = extract_rmsnorm_effective_gamma(norm_f).to(dtype=torch.float64, device="cpu")
    m_cand = cand_w - cand_w.mean(dim=0, keepdim=True)  # (4, D)
    m_eff = m_cand * gamma_f.unsqueeze(0)                # (4, D)
    f_real64 = F_REAL_4X3.to(torch.float64)

    b_old = torch.linalg.pinv(m_cand) @ f_real64
    r_old = m_eff @ b_old
    old_scale = float(torch.sum(r_old * f_real64) / torch.sum(f_real64 * f_real64).clamp_min(1e-12))
    old_tilt_err = float((r_old / max(1e-8, old_scale) - f_real64).norm().item() / f_real64.norm().item())

    b_exact64 = torch.linalg.pinv(m_eff) @ f_real64
    b_exact = b_exact64.float().contiguous()
    fixed_err = float((m_eff.float() @ b_exact - F_REAL_4X3).norm().item() / F_REAL_4X3.norm().item())
    assert fixed_err < 1e-4, f"B_align^(gamma_f) calibration error {fixed_err} >= 1e-4"

    return {
        "b_align": b_exact,
        "m_eff": m_eff.float().contiguous(),
        "gamma_f": gamma_f.float().contiguous(),
        "digit_ids": digit_ids,
        "old_tilt_err": old_tilt_err,
        "fixed_err": fixed_err,
    }


def _realify_uc(u_c: Optional[torch.Tensor], d: int) -> torch.Tensor:
    """Convert complex U_c (m, k_c) -> real (d, 2*k_c) orthonormal basis of top-2k_c complex subspace."""
    if u_c is None or u_c.numel() == 0:
        return torch.zeros((d, 0), dtype=torch.float64)
    uc = u_c.detach().cpu().to(torch.complex128)
    k_c = uc.shape[1]
    u_real = torch.zeros((d, 2 * k_c), dtype=torch.float64)
    u_real[0::2, 0::2] = uc.real
    u_real[1::2, 0::2] = uc.imag
    u_real[0::2, 1::2] = -uc.imag
    u_real[1::2, 1::2] = uc.real
    return u_real


def build_three_way_orthogonal_bus(
    u_c_d: Optional[torch.Tensor],
    m_eff: torch.Tensor,
    bus_dim: int = 22,
    seed: int = 202611,
    extra_block: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, dict]:
    """Construct orthonormal e_bus (D, bus_dim) orthogonal to prose U_c_d, row(M_cand D_gamma_f), and extra_block."""
    d = int(m_eff.shape[-1])
    u_cand_raw = m_eff.detach().cpu().to(torch.float64).T  # (D, 4) rank <= 3
    uc_q, _ = torch.linalg.qr(u_cand_raw, mode="reduced")
    # Keep top 3 linearly independent candidate-contrast directions
    u_cand = uc_q[:, : min(3, uc_q.shape[1])]
    u_prose_real = _realify_uc(u_c_d, d)

    blk_parts = [u_cand]
    if u_prose_real.shape[1] > 0:
        blk_parts.append(u_prose_real)
    use_extra = extra_block is not None and d >= 512 and extra_block.numel() > 0
    extra_cpu64 = None
    if use_extra:
        extra_cpu64 = extra_block.detach().cpu().to(torch.float64)
        assert extra_cpu64.shape[0] == d, f"extra_block row dim {extra_cpu64.shape[0]} != {d}"
        blk_parts.append(extra_cpu64)

    blocked = torch.cat(blk_parts, dim=1)
    q_blk, r_blk = torch.linalg.qr(blocked, mode="reduced")
    # Keep numerically non-degenerate columns only
    diag_r = torch.abs(torch.diag(r_blk))
    keep_blk = diag_r > 1e-10
    if bool(keep_blk.any()):
        q_blk = q_blk[:, keep_blk]
    # Verify actual free subspace is large enough
    free_dim = d - q_blk.shape[1]
    assert free_dim >= bus_dim, f"Insufficient free dims ({free_dim}) for bus_dim={bus_dim}"

    rng = np.random.default_rng(seed)
    raw = torch.from_numpy(rng.standard_normal((d, bus_dim + 4))).to(torch.float64)
    # Double Gram-Schmidt against blocked subspace to push bleed to machine zero (~1e-16)
    for _ in range(2):
        raw = raw - q_blk @ (q_blk.T @ raw)
    q_bus, _ = torch.linalg.qr(raw, mode="reduced")
    q_bus = q_bus[:, :bus_dim].contiguous()
    for _ in range(2):
        q_bus = q_bus - q_blk @ (q_blk.T @ q_bus)
    q_bus, _ = torch.linalg.qr(q_bus, mode="reduced")

    e_bus = q_bus.float().contiguous()
    cand_bleed = float((m_eff.float().cpu() @ e_bus).abs().max().item())
    prose_bleed = (
        float((u_prose_real.float().T @ e_bus).abs().max().item()) if u_prose_real.shape[1] > 0 else 0.0
    )
    extra_bleed = (
        float((extra_cpu64.float().T @ e_bus).abs().max().item()) if extra_cpu64 is not None else 0.0
    )
    assert cand_bleed < 1e-4, f"3-way bus candidate bleed {cand_bleed} >= 1e-4"
    assert prose_bleed < 1e-4, f"3-way bus prose bleed {prose_bleed} >= 1e-4"
    assert extra_bleed < 1e-4, f"3-way bus extra_block bleed {extra_bleed} >= 1e-4"

    return e_bus, {
        "bus_dim": int(bus_dim),
        "blocked_dims": int(q_blk.shape[1]),
        "cand_bleed_max": cand_bleed,
        "prose_bleed_max": prose_bleed,
        "extra_bleed_max": extra_bleed,
    }


# =============================================================================
# Dual-Space 64-Prompt Complex K-FAC Collection (d AND head-output d_o)
# =============================================================================


def _adaptive_kc_from_cov(
    c_mat: torch.Tensor,
    n_tok: int,
    k_max: int,
    energy_thresh: float = 0.90,
) -> Dict[str, Any]:
    """Compute adaptive sample-capped complex nullspace projector per N3 Theorem N3.10."""
    d = int(c_mat.shape[0])
    m = d // 2
    c_reg = c_mat / max(1, n_tok) + 1e-6 * torch.eye(d, dtype=torch.float64, device=c_mat.device)
    k_sample_cap = max(1, n_tok // 8)
    probe_k = min(max(1, k_max), k_sample_cap, m - 1)
    res_full = compute_complex_nullspace_projector(c_reg, k_c=probe_k)
    evals = res_full["eigenvalues"].clamp_min(0.0)
    tot_e = float(evals.sum().item())
    if tot_e > 0.0:
        cum = torch.cumsum(evals, dim=0) / tot_e
        hit = torch.nonzero(cum >= energy_thresh, as_tuple=False)
        k_energy = int(hit[0].item()) + 1 if hit.numel() > 0 else probe_k
    else:
        k_energy = probe_k
    k_eff = max(1, min(int(k_max), k_sample_cap, k_energy, m - 1))
    u_c = res_full["U_c"][:, :k_eff].detach().cpu().contiguous()
    del c_reg, res_full, evals
    return {"U_c": u_c, "k_eff": int(k_eff), "k_energy": int(k_energy), "k_sample_cap": int(k_sample_cap)}


def collect_dual_space_kfac_projectors(
    host,
    fa_layers: Sequence[int],
    dn_layers: Sequence[int],
    mlp_layers: Sequence[int],
    prompts: Sequence[str] = GENERAL_KFAC_64_PROMPTS,
    k_max_d: int = 96,
    k_max_o: int = 160,
    k_max_mlp: int = 48,
    max_prose_samples: int = 1024,
    stop_layer: int = QWEN4B_STOP_LAYER,
) -> Dict[str, Any]:
    """Run one no-grad forward pass over diverse prompts to collect K-FAC at BOTH `d` and head `d_o`."""
    layers = _get_decoder_layers(host.lm)
    n_l = len(layers)
    d = int(host.hidden_size)
    cov_dev = host.device
    if d < 512:
        k_max_d = min(k_max_d, max(2, (d - 28) // 4))
        k_max_mlp = min(k_max_mlp, k_max_d)
        k_max_o = min(k_max_o, max(2, d // 4))

    attn_dn_layers = sorted({int(i) for i in (*fa_layers, *dn_layers) if 0 <= i < n_l})
    mlp_valid = sorted({int(i) for i in mlp_layers if 0 <= i < n_l})

    cov_d: Dict[int, torch.Tensor] = {
        i: torch.zeros((d, d), dtype=torch.float64, device=cov_dev) for i in attn_dn_layers
    }
    cov_o: Dict[int, torch.Tensor] = {}
    o_dims: Dict[int, int] = {}
    for i in attn_dn_layers:
        blk = layers[i]
        o_dim = None
        if hasattr(blk, "self_attn") and int(i) in fa_layers:
            o_dim = int(blk.self_attn.o_proj.in_features)
        elif hasattr(blk, "linear_attn") and int(i) in dn_layers:
            o_dim = int(blk.linear_attn.out_proj.in_features)
        if o_dim is not None:
            cov_o[i] = torch.zeros((o_dim, o_dim), dtype=torch.float64, device=cov_dev)
            o_dims[i] = o_dim

    cov_mlp: Dict[int, torch.Tensor] = {
        i: torch.zeros((d, d), dtype=torch.float64, device=cov_dev) for i in mlp_valid
    }
    cov_tail = torch.zeros((d, d), dtype=torch.float64, device=cov_dev)

    # Record median general-text residual RMS at last 3 layers (post_attention_layernorm input) and final layer out
    tail_indices = [i for i in range(max(0, n_l - 3), n_l)]
    rms_samples: Dict[int, List[torch.Tensor]] = {i: [] for i in tail_indices}
    final_rms_samples: List[torch.Tensor] = []
    prose_last_samples: List[torch.Tensor] = []
    prose_post_mlp_samples: List[torch.Tensor] = []
    stop_prose_samples: List[torch.Tensor] = []
    n_tok = 0
    handles: List[object] = []

    def _accumulate_cov(buf: torch.Tensor, x: torch.Tensor) -> None:
        flat = x.detach().reshape(-1, x.shape[-1]).to(device=cov_dev, dtype=torch.float64)
        buf += flat.T @ flat

    for i in attn_dn_layers:
        blk = layers[i]
        in_norm = getattr(blk, "input_layernorm", blk)

        def mk_d_hook(lid: int):
            def _h(mod, args, out):
                xs = out[0] if isinstance(out, tuple) else out
                _accumulate_cov(cov_d[lid], xs)
            return _h

        handles.append(in_norm.register_forward_hook(mk_d_hook(i)))

        if i in cov_o:
            if hasattr(blk, "self_attn") and i in fa_layers:
                proj_o = blk.self_attn.o_proj
            else:
                proj_o = blk.linear_attn.out_proj

            def mk_o_hook(lid: int):
                def _pre(mod, args):
                    _accumulate_cov(cov_o[lid], args[0])
                return _pre

            handles.append(proj_o.register_forward_pre_hook(mk_o_hook(i)))

    for i in mlp_valid:
        post_norm = getattr(layers[i], "post_attention_layernorm", layers[i])

        def mk_m_hook(lid: int):
            def _h(mod, args, out):
                xs = out[0] if isinstance(out, tuple) else out
                _accumulate_cov(cov_mlp[lid], xs)
            return _h

        handles.append(post_norm.register_forward_hook(mk_m_hook(i)))

    for i in tail_indices:
        post_norm = getattr(layers[i], "post_attention_layernorm", layers[i])

        def mk_rms_hook(lid: int):
            def _pre(mod, args):
                _accumulate_cov(cov_tail, args[0])
                x = args[0].detach().float()
                rms = torch.sqrt(x.pow(2).mean(dim=-1) + 1e-6).reshape(-1).cpu()
                rms_samples[lid].append(rms)
            return _pre

        handles.append(post_norm.register_forward_pre_hook(mk_rms_hook(i)))

    # Collect general-text last-layer states before final norm + stop-layer prose states for Fisher Query Gate
    def _last_layer_hook(mod, args, out):
        hs = (out[0] if isinstance(out, tuple) else out).detach().float().reshape(-1, d).cpu()
        prose_last_samples.append(hs)
        final_rms_samples.append(torch.sqrt(hs.pow(2).mean(dim=-1) + 1e-6))

    if n_l > 0:
        handles.append(layers[-1].register_forward_hook(_last_layer_hook))
        if hasattr(layers[-1], "post_attention_layernorm"):
            def _last_pre_mlp(mod, args):
                hs = args[0].detach().float().reshape(-1, d).cpu()
                prose_post_mlp_samples.append(hs)
            handles.append(layers[-1].post_attention_layernorm.register_forward_pre_hook(_last_pre_mlp))
        if n_l >= 4:
            stop_idx = max(0, min(int(stop_layer), n_l - 1))

            def _stop_collect_hook(mod, args, out):
                hs = (out[0] if isinstance(out, tuple) else out).detach().float().reshape(-1, d).cpu()
                stop_prose_samples.append(hs)

            handles.append(layers[stop_idx].register_forward_hook(_stop_collect_hook))
        else:
            def _l0_in_collect_hook(mod, args):
                hs = args[0].detach().float().reshape(-1, d).cpu()
                stop_prose_samples.append(hs)

            handles.append(layers[0].register_forward_pre_hook(_l0_in_collect_hook))

    try:
        embed_layer = host.lm.get_input_embeddings()
        with torch.no_grad():
            for idx, pr in enumerate(prompts):
                if getattr(host, "tokenizer", None) is None:
                    ids = torch.tensor([[1 + (idx % 5), 2 + (idx % 7), 3, 4, 5, 6]], device=host.device, dtype=torch.long)
                else:
                    toks = host.tokenizer.encode(pr, add_special_tokens=False)
                    if not toks:
                        toks = [int(host.tokenizer.eos_token_id or 1)]
                    ids = torch.tensor([toks], device=host.device, dtype=torch.long)
                n_tok += int(ids.shape[1])
                emb = embed_layer(ids)
                attn = torch.ones_like(ids)
                host.lm(inputs_embeds=emb, attention_mask=attn, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    u_c_d: Dict[int, torch.Tensor] = {}
    u_c_o: Dict[int, torch.Tensor] = {}
    u_c_mlp: Dict[int, torch.Tensor] = {}
    k_eff_stats: Dict[str, Any] = {"d": {}, "d_o": {}, "mlp": {}}

    for i, c_mat in cov_d.items():
        res = _adaptive_kc_from_cov(c_mat, n_tok=n_tok, k_max=k_max_d, energy_thresh=0.90)
        u_c_d[i] = res["U_c"]
        k_eff_stats["d"][str(i)] = res["k_eff"]
    for i, c_mat in cov_o.items():
        res = _adaptive_kc_from_cov(c_mat, n_tok=n_tok, k_max=k_max_o, energy_thresh=0.90)
        u_c_o[i] = res["U_c"]
        k_eff_stats["d_o"][str(i)] = res["k_eff"]
        assert u_c_o[i].shape[0] * 2 == o_dims[i], f"Head-space U_c dim mismatch on layer {i}"
    for i, c_mat in cov_mlp.items():
        res = _adaptive_kc_from_cov(c_mat, n_tok=n_tok, k_max=k_max_mlp, energy_thresh=0.80)
        u_c_mlp[i] = res["U_c"]
        k_eff_stats["mlp"][str(i)] = res["k_eff"]

    tail_rms_medians: Dict[Any, float] = {}
    for i in tail_indices:
        if rms_samples[i]:
            all_r = torch.cat(rms_samples[i], dim=0)
            tail_rms_medians[i] = float(torch.median(all_r).item())
        else:
            tail_rms_medians[i] = 1.0
    if final_rms_samples:
        tail_rms_medians["final"] = float(torch.median(torch.cat(final_rms_samples, dim=0)).item())
    else:
        tail_rms_medians["final"] = 1.0

    # Global U_c_d reference at stop/tail for bus & query projection
    ref_layer = attn_dn_layers[-1] if attn_dn_layers else (mlp_valid[-1] if mlp_valid else None)
    # Build union covariance across active layers AND tail layers so e_bus is clean across ALL tail/active layers
    cov_bus = torch.zeros((d, d), dtype=torch.float64, device=cov_dev)
    for c_mat in cov_d.values():
        cov_bus += c_mat
    for c_mat in cov_mlp.values():
        cov_bus += c_mat
    cov_bus += cov_tail
    if float(cov_bus.abs().max().item()) > 0.0:
        k_max_bus = min(128, max(k_max_d, 128)) if d >= 512 else k_max_d
        u_c_bus = _adaptive_kc_from_cov(cov_bus, n_tok=n_tok, k_max=k_max_bus, energy_thresh=0.90)["U_c"]
    elif ref_layer is not None:
        u_c_bus = u_c_d.get(ref_layer)
    else:
        u_c_bus = None

    if float(cov_tail.abs().max().item()) > 0.0:
        k_max_tail = min(64, max(2, (d - 28) // 6)) if d < 512 else 64
        u_c_tail = _adaptive_kc_from_cov(cov_tail, n_tok=n_tok, k_max=k_max_tail, energy_thresh=0.90)["U_c"]
    else:
        u_c_tail = None

    del cov_d, cov_o, cov_mlp, cov_tail, cov_bus
    if cov_dev.type == "cuda":
        torch.cuda.empty_cache()

    def _cap_samples(chunks: List[torch.Tensor]) -> torch.Tensor:
        if not chunks:
            return torch.zeros((0, d), dtype=torch.float32)
        cat = torch.cat(chunks, dim=0)
        if cat.shape[0] > max_prose_samples:
            step = max(1, cat.shape[0] // max_prose_samples)
            cat = cat[::step][:max_prose_samples]
        return cat.contiguous()

    return {
        "n_prompts": len(prompts),
        "n_tokens": int(n_tok),
        "d_model": d,
        "o_dims": o_dims,
        "u_c_d": u_c_d,
        "u_c_o": u_c_o,
        "u_c_mlp": u_c_mlp,
        "u_c_bus": u_c_bus,
        "u_c_tail": u_c_tail,
        "k_eff_stats": k_eff_stats,
        "tail_rms_medians": tail_rms_medians,
        "prose_last_samples": _cap_samples(prose_last_samples),
        "prose_pre_mlp_samples": _cap_samples(prose_post_mlp_samples),
        "stop_prose_samples": _cap_samples(stop_prose_samples),
    }


# =============================================================================
# Core Algebraic Building Blocks (Folded Nullspace, SpecCayley, NoPE LoRA, SwiGLU)
# =============================================================================


def folded_p_perp_rows(weight: torch.Tensor, u_c: Optional[torch.Tensor]) -> torch.Tensor:
    """Fold P_{perp, C}^C onto row vectors W (..., D) -> (W P_{perp}) strictly in FP32."""
    wf = weight.float()
    if u_c is None or u_c.numel() == 0:
        return wf
    z = torch.complex(wf[..., 0::2], wf[..., 1::2])
    uc = u_c.to(device=z.device, dtype=torch.complex64)
    z_perp = z - (z @ uc.conj()) @ uc.T
    out = torch.empty_like(wf)
    out[..., 0::2] = z_perp.real
    out[..., 1::2] = z_perp.imag
    return out


class SpecCayley(nn.Module):
    """Bounded-Spectrum Polar Relaxed Nullspace Cayley M_perp = Q_perp D_perp (N3/N4).

    Executes strictly in FP32 (`complex64`), projects BOTH skew-Hermitian Q_perp and
    radial D_perp into the head-output K-FAC nullspace `u_c_o`, commutes with `J_d`,
    bounds singular values in `[1/sigma_max, sigma_max]`, and initializes to `I` at step 0.
    """

    def __init__(
        self,
        dim: int,
        r_c: int = 3,
        r_s: int = 2,
        u_c: Optional[torch.Tensor] = None,
        sigma_max: float = 2.0,
        init_std: float = 0.02,
    ):
        super().__init__()
        assert dim % 2 == 0, f"Expected even dim, got {dim}"
        self.dim = int(dim)
        self.m = self.dim // 2
        self.r_c = int(r_c)
        self.r_s = int(r_s)
        self.log_sigma_max = math.log(float(sigma_max))

        # Skew-Hermitian Woodbury parameters (Q_perp = I at V = 0)
        self.u_re = nn.Parameter(torch.randn(self.m, self.r_c, dtype=torch.float32) * init_std)
        self.u_im = nn.Parameter(torch.randn(self.m, self.r_c, dtype=torch.float32) * init_std)
        self.v_re = nn.Parameter(torch.zeros(self.m, self.r_c, dtype=torch.float32))
        self.v_im = nn.Parameter(torch.zeros(self.m, self.r_c, dtype=torch.float32))

        # Nullspace polar radial parameters (D_perp = I at sigma_raw = 0)
        self.w_re = nn.Parameter(torch.randn(self.m, self.r_s, dtype=torch.float32) * init_std)
        self.w_im = nn.Parameter(torch.randn(self.m, self.r_s, dtype=torch.float32) * init_std)
        self.sigma_raw = nn.Parameter(torch.zeros(self.r_s, dtype=torch.float32))

        if u_c is not None and u_c.numel() > 0:
            assert u_c.shape[0] == self.m, f"SpecCayley dim mismatch: u_c has m={u_c.shape[0]}, expected m={self.m}"
            self.register_buffer("u_c", u_c.to(torch.complex64).clone())
        else:
            self.u_c = None

    def _project_cols(self, mat: torch.Tensor) -> torch.Tensor:
        if self.u_c is None:
            return mat
        uc = self.u_c.to(device=mat.device, dtype=torch.complex64)
        return mat - uc @ (uc.conj().T @ mat)

    def _orthonormal_w_perp(self) -> torch.Tensor:
        w = torch.complex(self.w_re.float(), self.w_im.float())
        w_p = self._project_cols(w)
        # Differentiable Cholesky-QR in complex64 (O(m r_s^2), r_s <= 4)
        gram = w_p.conj().T @ w_p
        eye_s = torch.eye(self.r_s, dtype=torch.complex64, device=w.device)
        gram_reg = 0.5 * (gram + gram.conj().T) + 1e-6 * eye_s
        chol = torch.linalg.cholesky(gram_reg)  # lower-triangular L: L L^* = gram_reg
        # W_orth^* = L^{-1} W_p^* -> W_orth = (solve_triangular(L, W_p^*))^H
        w_orth = torch.linalg.solve_triangular(chol, w_p.conj().T, upper=False).conj().T
        # One second projection step removes any ~1e-7 rounding inside U_c
        return self._project_cols(w_orth)

    def radial_scales(self) -> torch.Tensor:
        return torch.exp(self.log_sigma_max * torch.tanh(self.sigma_raw.float()))

    def _projected_uv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        u = self._project_cols(torch.complex(self.u_re.float(), self.u_im.float()))
        v = self._project_cols(torch.complex(self.v_re.float(), self.v_im.float()))
        return u, v

    def materialize_matrix(self) -> torch.Tensor:
        """Materialize full complex m x m matrix M_perp = Q_perp D_perp (for small dims/tests)."""
        u_p, v_p = self._projected_uv()
        eye_m = torch.eye(self.m, dtype=torch.complex64, device=u_p.device)
        a_c = torch.cat((u_p, -v_p), dim=1)
        b_c = torch.cat((v_p, u_p), dim=1)
        eye_2r = torch.eye(2 * self.r_c, dtype=torch.complex64, device=u_p.device)
        core = eye_2r + 0.5 * (b_c.conj().T @ a_c)
        q_perp = eye_m - a_c @ torch.linalg.solve(core, b_c.conj().T)

        w_perp = self._orthonormal_w_perp()
        s_m1 = (self.radial_scales() - 1.0).to(torch.complex64)
        d_perp = eye_m + (w_perp * s_m1.unsqueeze(0)) @ w_perp.conj().T
        return q_perp @ d_perp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply M_perp x = Q_perp (D_perp x) strictly via FP32 Woodbury + Cholesky-QR."""
        orig_dtype = x.dtype
        xf = x.to(torch.float32)
        z = torch.complex(xf[..., 0::2], xf[..., 1::2])
        lead = z.shape[:-1]
        z2d = z.reshape(-1, self.m)  # (N, m)

        # 1. Right-action D_perp on row vectors: z_d = z + (z @ conj(W)) * (s - 1) @ W^T
        w_perp = self._orthonormal_w_perp()
        s_m1 = (self.radial_scales() - 1.0).to(torch.complex64)
        coeffs = (z2d @ w_perp.conj()) * s_m1.unsqueeze(0)
        z_d = z2d + coeffs @ w_perp.T

        # 2. Right-action Q_perp via O(m r_c + r_c^3) Woodbury solve
        u_p, v_p = self._projected_uv()
        a_c = torch.cat((u_p, -v_p), dim=1)
        b_c = torch.cat((v_p, u_p), dim=1)
        eye_2r = torch.eye(2 * self.r_c, dtype=torch.complex64, device=x.device)
        core = eye_2r + 0.5 * (b_c.conj().T @ a_c)
        rhs = (z_d @ b_c.conj()).T
        w = torch.linalg.solve(core, rhs).T
        z_out = (z_d - w @ a_c.T).reshape(*lead, self.m)

        out = torch.empty_like(xf)
        out[..., 0::2] = z_out.real
        out[..., 1::2] = z_out.imag
        return out.to(orig_dtype)


class FoldedNoPELoRADelta(nn.Module):
    """Weight-side P_{perp, C}^C folded LoRA with optional strict Qwen3.5 NoPE subspace targeting."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 6,
        alpha: float = 16.0,
        u_c: Optional[torch.Tensor] = None,
        head_dim: Optional[int] = None,
        rot_dim: int = 0,
        interleaved_gate: bool = False,
        out_slice: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.r = int(r)
        self.scale = float(alpha) / float(max(1, r))
        self.head_dim = int(head_dim) if head_dim is not None else None
        self.rot_dim = int(rot_dim)
        self.interleaved_gate = bool(interleaved_gate)
        self.out_slice = out_slice

        if self.head_dim is not None:
            stride = 2 * self.head_dim if self.interleaved_gate else self.head_dim
            assert self.out_features % stride == 0, f"out_features={self.out_features} % stride={stride} != 0"
            self.n_heads = self.out_features // stride
            self.nope_dim = self.head_dim - self.rot_dim
            assert self.nope_dim > 0, f"NoPE dim must be > 0 (head_dim={self.head_dim}, rot_dim={self.rot_dim})"
            eff_out = self.n_heads * self.nope_dim
        elif out_slice is not None:
            self.n_heads = 0
            self.nope_dim = 0
            eff_out = int(out_slice[1] - out_slice[0])
        else:
            self.n_heads = 0
            self.nope_dim = 0
            eff_out = self.out_features

        self.lora_a = nn.Parameter(
            torch.randn(self.r, self.in_features, dtype=torch.float32) * (1.0 / math.sqrt(self.in_features))
        )
        self.lora_b = nn.Parameter(torch.zeros(eff_out, self.r, dtype=torch.float32))
        if u_c is not None and u_c.numel() > 0:
            assert u_c.shape[0] * 2 == self.in_features, (
                f"LoRA u_c mismatch: 2*m={u_c.shape[0]*2} != in_features={self.in_features}"
            )
            self.register_buffer("u_c", u_c.to(torch.complex64).clone())
        else:
            self.u_c = None

    def effective_a(self) -> torch.Tensor:
        return folded_p_perp_rows(self.lora_a, self.u_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.to(torch.float32)
        h = F.linear(xf, self.effective_a())
        return (F.linear(h, self.lora_b) * self.scale).to(x.dtype)

    def add_into_output(self, out: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        if self.head_dim is not None:
            stride = 2 * self.head_dim if self.interleaved_gate else self.head_dim
            res = out.clone().view(*out.shape[:-1], self.n_heads, stride)
            res[..., self.rot_dim : self.head_dim] = res[..., self.rot_dim : self.head_dim] + delta.view(
                *out.shape[:-1], self.n_heads, self.nope_dim
            )
            return res.reshape_as(out)
        if self.out_slice is not None:
            s0, s1 = self.out_slice
            res = out.clone()
            res[..., s0:s1] = res[..., s0:s1] + delta
            return res
        return out + delta


class SparseAntipodalSwiGLUOverlay(nn.Module):
    """Non-mutating FP32 antipodal virtual SwiGLU pairs with weight-side P_perp + D_gamma^-1."""

    def __init__(
        self,
        d_model: int,
        k_tune: int = 8,
        u_c: Optional[torch.Tensor] = None,
        gamma_inv: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.k_tune = int(k_tune)
        self.a_tune = nn.Parameter(
            torch.randn(self.k_tune, self.d_model, dtype=torch.float32) * (0.04 / math.sqrt(self.d_model))
        )
        self.b_tune = nn.Parameter(
            torch.randn(self.k_tune, self.d_model, dtype=torch.float32) * (0.04 / math.sqrt(self.d_model))
        )
        self.c_tune = nn.Parameter(torch.zeros(self.k_tune, self.d_model, dtype=torch.float32))
        if u_c is not None and u_c.numel() > 0:
            self.register_buffer("u_c", u_c.to(torch.complex64).clone())
        else:
            self.u_c = None
        if gamma_inv is not None:
            self.register_buffer("gamma_inv", gamma_inv.detach().to(device=self.a_tune.device, dtype=torch.float32).clone())
        else:
            self.register_buffer("gamma_inv", torch.ones(self.d_model, dtype=torch.float32, device=self.a_tune.device))

    def forward(self, x_normed: torch.Tensor, gate_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        xf = x_normed.float() * self.gamma_inv
        ap = folded_p_perp_rows(self.a_tune, self.u_c)
        bp = folded_p_perp_rows(self.b_tune, self.u_c)
        u = F.linear(xf, ap)
        v = F.linear(xf, bp)
        act = torch.clamp(F.silu(u) * v + F.silu(-u) * (-v), -16.0, 16.0)
        delta = F.linear(act, self.c_tune.T)
        if gate_mask is not None:
            delta = delta * gate_mask.float().unsqueeze(-1)
        return delta.to(x_normed.dtype)


# =============================================================================
# Non-Mutating Parallel 2-Head Shadow GDN Ballast Bridge & Compiled Circuit A
# =============================================================================


class ShadowGDNBallastBridge(nn.Module):
    """Non-mutating FP32 2-head Shadow GatedDeltaNet bridge with constant ballast & KV continuity.

    Bus layout (`e_bus`, width 22):
      At prefix EM tokens (read on input embeddings / carrier, slots 0..15):
        slots 0..2:   W1 `(m1_r, +/-m1_i, q1)`
        slots 3..5:   W2 `(m2_r, m2_i, q2)` (nonzero on Q1 only)
        slots 6..8:   PHASE Q1 rotation `(r_r, r_i, r_2)` (nonzero on Q1 only)
        slot  9:      anchor `c_0 = 1.0`
        slots 10..12: PHASE Q2/Q3 kernel/pilot `(k_r, k_i, k_2)` (nonzero on Q2/Q3)
        slots 13..15: mutually orthogonal indicators `(ind_w1, ind_w2, ind_ph)`
      At answer position (strictly disjoint across Circuit A layers):
        - ShadowGDN outputs: 0..2 (w1*s1), 3..5 (w2*s0), 6..8 (r_h*s0), 9 (c_0*s0),
          10..12 (k_23*s0), 13 (c_0*s2).
        - Circuit A l0_delta outputs: 14..16 (T2+K23)*s1, 17 (u_0*s1).
        - Circuit A l1_delta outputs: 18 (u_01*s2), 19..21 (P*s2).
    """

    BALLAST_CONST = 8.0
    A_ANCHOR = 2.0

    def __init__(
        self,
        e_bus: torch.Tensor,
        s_med_0: float,
        s_med_1: float,
        s_med_2: float,
    ):
        super().__init__()
        self.register_buffer("e_bus", e_bus.float().clone())  # (D, 22)
        self.s_med_0 = float(s_med_0)
        self.s_med_1 = float(s_med_1)
        self.s_med_2 = float(s_med_2)
        self.log_ret = nn.Parameter(torch.tensor(8.0, dtype=torch.float32))  # sigmoid ~ 0.9997+ -> clamped to 1
        self.h0_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.h1_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self._pref_states: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    def clear_state(self) -> None:
        self._pref_states = None

    def forward_pass(
        self,
        h_entry: torch.Tensor,
        m_acc_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Causal exact GDN scan across Pass 1 (prefix) OR Pass 2 (KV suffix continuation)."""
        b, seq_l, _ = h_entry.shape
        device = h_entry.device
        hf = h_entry.float()
        # Project ONLY onto 22-dim 3-way orthogonal bus (tiny (B, L, 22) tensor)
        bus_in = hf @ self.e_bus  # (B, L, 22)

        m_f = (
            m_acc_mask.to(device=device, dtype=torch.float32)
            if m_acc_mask is not None
            else torch.ones((b, seq_l), device=device, dtype=torch.float32)
        )
        m_len = m_f.shape[1]

        is_pass1 = seq_l >= m_len
        alpha = torch.sigmoid(self.log_ret + 8.0)  # ~ 0.999999 on Pass 1 & 2

        if is_pass1:
            if seq_l > m_len:
                pad_0 = torch.zeros((b if m_acc_mask is None else m_f.shape[0], seq_l - m_len), device=device)
                if m_f.shape[0] != b and b % m_f.shape[0] == 0:
                    m_f = m_f.repeat_interleave(b // m_f.shape[0], dim=0)
                    pad_0 = torch.zeros((b, seq_l - m_len), device=device)
                m_eff = torch.cat((m_f, pad_0), dim=1)
            else:
                m_eff = m_f if m_f.shape[0] == b else m_f.repeat_interleave(b // m_f.shape[0], dim=0)

            ind_w1 = bus_in[..., 13]
            ind_w2 = bus_in[..., 14]
            ind_ph = bus_in[..., 15]
            beta_w1 = m_eff * torch.sigmoid(24.0 * (ind_w1 - 0.5))
            beta_w2 = m_eff * torch.sigmoid(24.0 * (ind_w2 - 0.5))
            beta_ph = m_eff * torch.sigmoid(24.0 * (ind_ph - 0.5))

            s_w1 = torch.zeros((b, 3), device=device, dtype=torch.float32)
            s_w2 = torch.zeros((b, 3), device=device, dtype=torch.float32)
            # Head 1 payload: [r_h(3), k_23(3), c_0(1), ballast(1)] = 8 dims
            s_ph = torch.zeros((b, 8), device=device, dtype=torch.float32)

            v_ph_all = torch.cat(
                (
                    bus_in[..., 6:9],
                    bus_in[..., 10:13],
                    bus_in[..., 9:10],
                    torch.full((b, seq_l, 1), self.BALLAST_CONST, device=device, dtype=torch.float32),
                ),
                dim=-1,
            )  # (B, L, 8)

            o_w1, o_w2, o_ph = [], [], []
            for t in range(seq_l):
                bw1 = beta_w1[:, t : t + 1]
                bw2 = beta_w2[:, t : t + 1]
                bph = beta_ph[:, t : t + 1]
                s_w1 = alpha * (1.0 - bw1) * s_w1 + bw1 * bus_in[:, t, 0:3]
                s_w2 = alpha * (1.0 - bw2) * s_w2 + bw2 * bus_in[:, t, 3:6]
                s_ph = alpha * (1.0 - bph) * s_ph + bph * v_ph_all[:, t, :]
                o_w1.append(s_w1)
                o_w2.append(s_w2)
                o_ph.append(s_ph)

            t_w1 = torch.stack(o_w1, dim=1)  # (B, L, 3)
            t_w2 = torch.stack(o_w2, dim=1)  # (B, L, 3)
            t_ph = torch.stack(o_ph, dim=1)  # (B, L, 8)
            self._pref_states = (s_w1, s_w2, s_ph)
        else:
            # Pass 2 KV-cached suffix continuation: resume from exact Pass 1 final state with beta = 0
            assert self._pref_states is not None, "ShadowGDNBallastBridge called on Pass 2 before Pass 1!"
            s1_p, s2_p, sp_p = self._pref_states
            if b != s1_p.shape[0]:
                assert b % s1_p.shape[0] == 0
                rep = b // s1_p.shape[0]
                s1_p = s1_p.repeat_interleave(rep, dim=0)
                s2_p = s2_p.repeat_interleave(rep, dim=0)
                sp_p = sp_p.repeat_interleave(rep, dim=0)
            pows = (alpha ** torch.arange(1, seq_l + 1, device=device, dtype=torch.float32))[None, :, None]
            t_w1 = s1_p.unsqueeze(1) * pows
            t_w2 = s2_p.unsqueeze(1) * pows
            t_ph = sp_p.unsqueeze(1) * pows

        # Ballast-stabilized RMSNormGated (gamma_h === 1, ballast cancels scaling identically!)
        bal = t_ph[..., 7:8]
        rms_g = torch.sqrt(t_ph.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        gate_on = F.silu(torch.full_like(bal, 4.0))
        normed = (t_ph / rms_g) * gate_on
        ballast_out = normed[..., 7:8].clamp_min(1e-6)
        # Hard-plateau ballast gate: strictly 0.0 for bal <= 2.0, exact 1.0 for real <PHASE> (bal ~ 8.0)
        ph_active = torch.clamp((bal - 2.0) / 4.0, 0.0, 1.0)
        ph_rec = ph_active * (normed[..., :7] / ballast_out) * self.BALLAST_CONST

        r_h = ph_rec[..., 0:3] * self.h1_scale
        k_23 = ph_rec[..., 3:6] * self.h1_scale
        c_0_val = ph_rec[..., 6:7]  # 1.0 when EM active, 0.0 otherwise

        # Assemble disjoint bus payload matching Circuit A s_med scales
        out_bus = torch.zeros((b, seq_l, self.e_bus.shape[1]), device=device, dtype=torch.float32)
        out_bus[..., 0:3] = (t_w1 * self.h0_scale) * self.s_med_1
        out_bus[..., 3:6] = (t_w2 * self.h0_scale) * self.s_med_0
        out_bus[..., 6:9] = r_h * self.s_med_0
        out_bus[..., 9:10] = (self.A_ANCHOR * c_0_val) * self.s_med_0
        out_bus[..., 10:13] = k_23 * self.s_med_0
        out_bus[..., 13:14] = (self.A_ANCHOR * c_0_val) * self.s_med_2
        return out_bus @ self.e_bus.T  # (B, L, D) FP32


def fit_positive_fermi_dirac_nodes(
    q: int,
    eta_min: float,
    eta_max: float,
    m_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Fit strictly-positive Euler-Maclaurin/NNLS SwiGLU quadrature nodes (w_m >= 0, cond=1)."""
    u_lo = 0.015 if q == 0 else (0.04 if q == 1 else 0.08)
    u_hi = 22.0 if q <= 1 else 30.0
    ts = np.geomspace(u_lo / eta_max, u_hi / eta_min, m_nodes)
    etas = np.geomspace(eta_min, eta_max, 512)
    u = etas[:, None] * ts[None, :]
    phi = u / (1.0 + np.exp(np.clip(u, -80.0, 80.0)))
    target = etas ** (-q)
    ws, _ = nnls(phi / target[:, None], np.ones_like(etas))
    nz = ws > 1e-10
    if bool(nz.any()):
        ts = ts[nz]
        ws = ws[nz]
    phi_nz = (etas[:, None] * ts[None, :]) / (1.0 + np.exp(np.clip(etas[:, None] * ts[None, :], -80.0, 80.0)))
    rel_err = float(np.max(np.abs(phi_nz @ ws - target) / target))
    return torch.from_numpy(ts.astype(np.float32)), torch.from_numpy(ws.astype(np.float32)), rel_err


class CompiledN1CircuitAOverlay(nn.Module):
    """Exact 3-layer Pre-RMSNorm invariant N1 Circuit A (FP32 Sparse Additive Overlay).

    Computes canonical QPSK Fourier probabilities in B_align^(gamma_f) via strictly
    positive Fermi-Dirac SwiGLU operators (F_0 -> F_0 -> F_2) canceling independent layer
    RMSNorm scales (s_0^2, s_1^2, s_2), plus optional F_1 dynamic host digit debiaser,
    strictly guarded by Hard-Plateau Dead-Zone + Fisher Query gates + g_EM.
    """

    KAPPA_0 = 8.0
    KAPPA_1 = 8.0
    A_ANCHOR = 2.0

    def __init__(
        self,
        e_bus: torch.Tensor,
        b_align: torch.Tensor,
        m_eff: torch.Tensor,
        gamma_0: torch.Tensor,
        gamma_1: torch.Tensor,
        gamma_2: torch.Tensor,
        s_med_0: float,
        s_med_1: float,
        s_med_2: float,
        s_med_f: Optional[float] = None,
        tau_lin_init: float = 3.55,
    ):
        super().__init__()
        dev = e_bus.device
        eb_f = e_bus.to(device=dev, dtype=torch.float32).clone()
        ba_f = b_align.to(device=dev, dtype=torch.float32).clone()
        me_f = m_eff.to(device=dev, dtype=torch.float32).clone()
        fr_f = F_REAL_4X3.to(device=dev, dtype=torch.float32).clone()

        self.register_buffer("e_bus", eb_f)         # (D, 22)
        self.register_buffer("b_align", ba_f)       # (D, 3)
        self.register_buffer("m_eff", me_f)         # (4, D)
        self.register_buffer("f_real", fr_f)        # (4, 3)
        self.register_buffer("g0_inv", safe_gamma_inv(gamma_0).to(device=dev, dtype=torch.float32))
        self.register_buffer("g1_inv", safe_gamma_inv(gamma_1).to(device=dev, dtype=torch.float32))
        self.register_buffer("g2_inv", safe_gamma_inv(gamma_2).to(device=dev, dtype=torch.float32))

        self.s_med_0 = float(s_med_0)
        self.s_med_1 = float(s_med_1)
        self.s_med_2 = float(s_med_2)
        self.s_med_f = float(s_med_2 if s_med_f is None else s_med_f)

        # Precompute D_gamma^{-1} e_bus read projections (Factored, mathematically == dense neurons!)
        self.register_buffer("read_0", eb_f * self.g0_inv.unsqueeze(1))  # (D, 22)
        self.register_buffer("read_1", eb_f * self.g1_inv.unsqueeze(1))  # (D, 22)
        self.register_buffer("read_2", eb_f * self.g2_inv.unsqueeze(1))  # (D, 22)
        # Active prose nuller projection: (D, 3) = (M_eff.T @ F_real) * g2_inv[:, None] & unnormed mlp_out
        debias_mlp = (me_f.T @ fr_f).contiguous()
        self.register_buffer("read_debias_mlp", debias_mlp)
        self.register_buffer("read_debias_2", debias_mlp * self.g2_inv.unsqueeze(1))

        # Strictly positive NNLS Fermi-Dirac nodes (cond=1, sparse active nodes ~11-14 per layer, rel_err <= 0.06%)
        ts0, ws0, _ = fit_positive_fermi_dirac_nodes(0, 0.06, 7.0, 15)
        ts1, ws1, _ = fit_positive_fermi_dirac_nodes(1, 0.06, 7.0, 15)
        ts2, ws2, _ = fit_positive_fermi_dirac_nodes(2, 0.02, 9.0, 18)
        self.register_buffer("ts0", ts0.to(device=dev))
        self.register_buffer("ws0", ws0.to(device=dev))
        self.register_buffer("ts1", ts1.to(device=dev))
        self.register_buffer("ws1", ws1.to(device=dev))
        self.register_buffer("ts2", ts2.to(device=dev))
        self.register_buffer("ws2", ws2.to(device=dev))
        self._last_diag: Dict[str, float] = {}

        # Trainable per-circuit gain & debias scalars (< 10 params)
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau_lin_init), dtype=torch.float32, device=dev))
        self.comp_gain = nn.Parameter(torch.ones(3, dtype=torch.float32, device=dev))
        self.dyn_debias_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=dev))
        self.suffix_debias_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=dev))
        # Deterministic template suffix offset table per canonical task id (populated in Step 0, zero labels)
        self.register_buffer("template_suffix_canon_bias", torch.zeros((5, 3), dtype=torch.float32, device=dev))

    @staticmethod
    def _antipodal(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return F.silu(u) * v + F.silu(-u) * (-v)

    @staticmethod
    def _fermi_dirac(eta: torch.Tensor, v: torch.Tensor, ts: torch.Tensor, ws: torch.Tensor) -> torch.Tensor:
        # Computes sum_m w_m * SiLU(-t_m * eta) * (-v) = v * eta^{-q}
        t_v = ts.to(device=eta.device, dtype=torch.float32)
        w_v = ws.to(device=eta.device, dtype=torch.float32)
        terms = F.silu(-eta.unsqueeze(-1) * t_v) * (-v.unsqueeze(-1))
        return torch.sum(terms * w_v, dim=-1)

    def l0_delta(
        self,
        z_normed: torch.Tensor,
        gate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Layer n_l-3 MLP delta: T_2' (Q1), K_pre (Q2/Q3), and u_0 = c_0 / s_tilde_0."""
        gm = gate_mask.float().unsqueeze(-1)
        if not bool((gm > 0.0).any()):
            return torch.zeros_like(z_normed)
        c = z_normed.float() @ self.read_0  # (B, L, 22)
        u2_r, u2_i, u2_q = c[..., 3], c[..., 4], c[..., 5]
        r_r, r_i, r_2 = c[..., 6], c[..., 7], c[..., 8]
        rho_0 = c[..., 9] / self.A_ANCHOR   # 1 / s_tilde_0
        k_r, k_i, k_2 = c[..., 10], c[..., 11], c[..., 12]

        # Complex rotation T_2' (Karatsuba-6): (u2_r + i*u2_i)*(r_r + i*r_i)
        m1 = self._antipodal(u2_r, r_r)
        m2 = self._antipodal(u2_i, r_i)
        m3 = self._antipodal(u2_r + u2_i, r_r + r_i)
        t2_r = m1 - m2
        t2_i = m3 - m1 - m2
        t2_2 = self._antipodal(u2_q, r_2)

        # Copy K_23 with same s_tilde_0^{-2} scaling: (1/s_tilde_0) * (k / s_tilde_0)
        c0_kr = self._antipodal(rho_0, k_r)
        c0_ki = self._antipodal(rho_0, k_i)
        c0_k2 = self._antipodal(rho_0, k_2)

        # Self-normalizing anchor u_0 = F_0(rho_0, rho_0) = 1 / s_tilde_0
        u_0 = self._fermi_dirac(rho_0.clamp_min(1e-4), rho_0, self.ts0, self.ws0)

        out_bus = torch.zeros_like(c)
        # Disjoint slots 14..17 so Bridge slots 10..13 are never overwritten
        out_bus[..., 14] = (self.KAPPA_0 * (t2_r + c0_kr)) * self.s_med_1
        out_bus[..., 15] = (self.KAPPA_0 * (t2_i + c0_ki)) * self.s_med_1
        out_bus[..., 16] = (self.KAPPA_0 * (t2_2 + c0_k2)) * self.s_med_1
        out_bus[..., 17] = (self.A_ANCHOR * u_0) * self.s_med_1
        return ((out_bus * gm) @ self.e_bus.T).to(z_normed.dtype)

    def l1_delta(
        self,
        z_normed: torch.Tensor,
        gate_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Layer n_l-2 MLP delta: unified C_4 harmonic bilinear product P/(s_0^2 s_1^2) & u_01."""
        gm = gate_mask.float().unsqueeze(-1)
        if not bool((gm > 0.0).any()):
            return torch.zeros_like(z_normed)
        c = z_normed.float() @ self.read_1  # (B, L, 22)
        u1_r, u1_i, u1_q = c[..., 0], c[..., 1], c[..., 2]
        k_eff_r = c[..., 14] / self.KAPPA_0
        k_eff_i = c[..., 15] / self.KAPPA_0
        k_eff_2 = c[..., 16] / self.KAPPA_0
        u_0_in = c[..., 17] / self.A_ANCHOR  # 1 / (s_tilde_0 * s_tilde_1)

        m1 = self._antipodal(u1_r, k_eff_r)
        m2 = self._antipodal(u1_i, k_eff_i)
        m3 = self._antipodal(u1_r + u1_i, k_eff_r + k_eff_i)
        p_0 = (m1 - m2) / math.sqrt(2.0)
        p_1 = (m3 - m1 - m2) / math.sqrt(2.0)
        p_2 = 0.5 * self._antipodal(u1_q, k_eff_2)

        u_01 = self._fermi_dirac(u_0_in.clamp_min(1e-4), u_0_in, self.ts0, self.ws0)

        out_bus = torch.zeros_like(c)
        out_bus[..., 18] = (self.A_ANCHOR * u_01) * self.s_med_2
        out_bus[..., 19] = (self.KAPPA_1 * p_0) * self.s_med_2
        out_bus[..., 20] = (self.KAPPA_1 * p_1) * self.s_med_2
        out_bus[..., 21] = (self.KAPPA_1 * p_2) * self.s_med_2
        return ((out_bus * gm) @ self.e_bus.T).to(z_normed.dtype)

    def l2_delta(
        self,
        z_normed: torch.Tensor,
        gate_mask: torch.Tensor,
        mlp_out: Optional[torch.Tensor] = None,
        lift_shift_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        task_id_val: Optional[int] = None,
    ) -> torch.Tensor:
        """Layer n_l-1 MLP delta: F_2 self-normalized Fourier readout -> B_align^(gamma_f) + full debias."""
        gm = gate_mask.float().unsqueeze(-1)
        if not bool((gm > 0.0).any()):
            return torch.zeros_like(z_normed)
        zf = z_normed.float()
        c = zf @ self.read_2  # (B, L, 22)
        eta_2 = (c[..., 18] / self.A_ANCHOR).clamp_min(1e-4)  # 1 / (s_tilde_0 * s_tilde_1 * s_tilde_2)
        p_in = torch.stack((c[..., 19], c[..., 20], c[..., 21]), dim=-1) / self.KAPPA_1
        rho_2 = (c[..., 13] / self.A_ANCHOR).clamp_min(1e-4)  # 1 / s_tilde_2

        # Lift F_2(eta_2, p_in) -> s_tilde_2 * p_canon
        tau_lin = torch.exp(self.log_tau)
        lifted_canon = torch.stack(
            [self._fermi_dirac(eta_2, p_in[..., ch], self.ts2, self.ws2) * self.comp_gain[ch] for ch in range(3)],
            dim=-1,
        ) * tau_lin  # (B, L, 3)

        if lift_shift_fn is not None:
            lifted_coords = lift_shift_fn(lifted_canon)
        else:
            lifted_coords = lifted_canon
        raw_signal = lifted_coords * self.s_med_f

        # Full dynamic host bias cancellation: x_2 (via F_1 * s_med_2) + mlp_out
        d_host = zf @ self.read_debias_2  # (B, L, 3): F_real^T (M_eff (x_2 / s_2))
        bias_x2 = torch.stack(
            [self._fermi_dirac(rho_2, d_host[..., ch], self.ts1, self.ws1) for ch in range(3)],
            dim=-1,
        ) * self.s_med_2
        bias_mlp = (mlp_out.float() @ self.read_debias_mlp) if mlp_out is not None else 0.0
        bias_dyn = (bias_x2 + bias_mlp) * self.dyn_debias_scale

        # Deterministic template suffix debias in s_tilde_2 * s_med_f units via F_2(rho_2, rho_2) = s_tilde_2
        s_t2_factor = None
        bias_suf = 0.0
        if task_id_val is not None and 0 <= int(task_id_val) < self.template_suffix_canon_bias.shape[0]:
            s_t2_factor = self._fermi_dirac(rho_2, rho_2, self.ts2, self.ws2)
            suf_b = self.template_suffix_canon_bias[int(task_id_val)]
            bias_suf = s_t2_factor.unsqueeze(-1) * suf_b * self.s_med_f * self.suffix_debias_scale

        with torch.no_grad():
            gm_b = gate_mask > 0.5
            if bool(gm_b.any()):
                st2_val = s_t2_factor if s_t2_factor is not None else self._fermi_dirac(rho_2, rho_2, self.ts2, self.ws2)
                self._last_diag = {
                    "rho_2": float(rho_2[gm_b].mean().item()),
                    "eta_2": float(eta_2[gm_b].mean().item()),
                    "s_t2_factor": float(st2_val[gm_b].mean().item()),
                }

        net_coords = raw_signal - bias_dyn - bias_suf
        delta = (net_coords * gm) @ self.b_align.T
        return delta.to(z_normed.dtype)

    def n1_parameter_count_dense_equiv(self, d_model: int) -> int:
        # Factored weights equal 55 dense SwiGLU neurons + 15 F_1 debias neurons = 70 or 55 without F_1
        return 3 * 55 * d_model


class GatedSchurIrrepReadout(nn.Module):
    """D_gamma_f-compensated rank-11 Schur readout with stationary Dead-Zone + Fisher Query Gate."""

    def __init__(
        self,
        hidden_size: int,
        b_align: torch.Tensor,
        u_c_last: Optional[torch.Tensor] = None,
        r: int = 11,
        init_std: float = 0.01,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.r = int(r)
        self.proj_in = nn.Parameter(
            torch.randn(self.r, self.hidden_size, dtype=torch.float32) * (init_std / math.sqrt(self.hidden_size))
        )
        self.proj_coord = nn.Parameter(torch.zeros(3, self.r, dtype=torch.float32))
        self.qgate_w = nn.Parameter(torch.zeros(self.hidden_size, dtype=torch.float32))
        self.qgate_b = nn.Parameter(torch.tensor(-10.0, dtype=torch.float32))
        self.dz_tau = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
        self.dz_high = nn.Parameter(torch.tensor(0.65, dtype=torch.float32))
        self.register_buffer("b_align", b_align.detach().float().clone())
        if u_c_last is not None and u_c_last.numel() > 0:
            self.register_buffer("u_c", u_c_last.to(torch.complex64).clone())
        else:
            self.u_c = None

    def compute_qgate(self, h: torch.Tensor, hard_plateau: bool = True) -> torch.Tensor:
        score = F.linear(h.float(), self.qgate_w.unsqueeze(0), self.qgate_b.unsqueeze(0)).squeeze(-1)
        if hard_plateau:
            # Sigmoid plateau around calibrated Fisher margin: exact 0.0 for <= -4, exact 1.0 for >= +4
            return torch.clamp((torch.sigmoid(score) - 0.02) / 0.96, 0.0, 1.0)
        return torch.sigmoid(score)

    def forward_delta(self, h: torch.Tensor, gate_mask: torch.Tensor) -> torch.Tensor:
        gm = gate_mask.float().unsqueeze(-1)
        if not bool((gm > 0.0).any()):
            return torch.zeros_like(h)
        w_in = folded_p_perp_rows(self.proj_in, self.u_c)
        coords = F.linear(F.gelu(F.linear(h.float(), w_in)), self.proj_coord)
        delta = (coords @ self.b_align.T) * gm
        return delta.to(h.dtype)


# =============================================================================
# Separate-Modality Orthogonal Carrier Builder (Zero Data Leakage, Zero Pre-Mul)
# =============================================================================


def build_regime2_separate_carrier(
    prior_canon: np.ndarray | torch.Tensor,
    codes: torch.Tensor,
    task: str,
    phase_slots: torch.Tensor,
    meta_scalars: torch.Tensor,
    ref_summary: Optional[torch.Tensor] = None,
    discrete_d: Optional[torch.Tensor] = None,
    valid: Optional[torch.Tensor] = None,
    bits: int = 4,
) -> Dict[str, torch.Tensor]:
    """Compute strictly separated public `<W1>`, `<W2>`, `<PHASE>` 16-dim carrier tensors.

    Zero access to raw IQ, labels, PhysicalKernelS17 integral, or pre-multiplied R(theta)m1.
    """
    device = codes.device
    b = codes.shape[0]
    p_tbl = (
        torch.as_tensor(prior_canon, dtype=torch.float32, device=device)
        if not isinstance(prior_canon, torch.Tensor)
        else prior_canon.to(device=device, dtype=torch.float32)
    )
    roots_c = torch.as_tensor(ROOTS, dtype=torch.complex64, device=device)
    quad_sign = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=torch.float32, device=device)

    codes_4, _, eff_valid = decode_c16_tail(words=codes, bits=bits, valid=valid)
    inner = codes_4 % 4
    a1 = p_tbl[inner[:, 0]]
    a2 = p_tbl[inner[:, 1]]
    a1 = torch.where(eff_valid[:, 0:1], a1, torch.full_like(a1, 0.25))
    a2 = torch.where(eff_valid[:, 1:2], a2, torch.full_like(a2, 0.25))

    m1 = a1.to(torch.complex64) @ roots_c
    q1 = (a1 * quad_sign).sum(dim=-1)
    m2 = a2.to(torch.complex64) @ roots_c
    q2 = (a2 * quad_sign).sum(dim=-1)

    tname = task
    is_q1 = tname in ("Q1", "q1")
    is_q2 = tname in ("Q2", "q2")
    is_indep = tname in ("Q3_independent_x3", "q3_indep")

    # Separate token vectors in 16-dim raw carrier coordinates
    w1_vec = torch.zeros((b, 16), device=device, dtype=torch.float32)
    w1_vec[:, 0] = m1.real
    w1_vec[:, 1] = -m1.imag if is_q1 else m1.imag
    w1_vec[:, 2] = q1
    w1_vec[:, 13] = 1.0  # ind_w1

    w2_vec = torch.zeros((b, 16), device=device, dtype=torch.float32)
    if is_q1:
        w2_vec[:, 3] = m2.real
        w2_vec[:, 4] = m2.imag
        w2_vec[:, 5] = q2
    w2_vec[:, 14] = 1.0  # ind_w2

    ph_vec = torch.zeros((b, 16), device=device, dtype=torch.float32)
    ph_vec[:, 9] = 1.0   # anchor c_0 = 1.0
    ph_vec[:, 15] = 1.0  # ind_ph

    h, ell, _, omega, sigma, snr_db, rate_var = meta_scalars.float().unbind(dim=-1)
    dd = (
        discrete_d.to(device=device, dtype=torch.float32)
        if discrete_d is not None
        else torch.zeros_like(h)
    )
    is_disc = dd > 0.0

    if not is_indep:
        if is_q2:
            if ref_summary is not None:
                bp = ref_summary[:, 0:4].to(device=device, dtype=torch.float32)
            else:
                bp = torch.full((b, 4), 0.25, device=device, dtype=torch.float32)
            mbp = bp.to(torch.complex64) @ roots_c
            qbp = (bp * quad_sign).sum(dim=-1)
            ph_vec[:, 10] = mbp.real
            ph_vec[:, 11] = -mbp.imag
            ph_vec[:, 12] = qbp
        elif is_q1:
            # Public Q1 phase slots: slots[:, 0] = cos(w*h), slots[:, 1] = sin(w*h)
            var12 = sigma.square() * h + rate_var * h.square()
            c_h = torch.where(h <= 0.0, torch.ones_like(h), (1.0 - (4.0 / 3.0) * dd).clamp(0.0, 1.0).pow(h))
            amp1 = torch.where(is_disc, c_h, 0.900316 * torch.exp(-0.5 * var12))
            amp2 = torch.where(is_disc, c_h, 0.636620 * torch.exp(-2.0 * var12))
            c_wh = torch.where(is_disc, torch.ones_like(amp1), phase_slots[:, 0].to(device=device, dtype=torch.float32))
            s_wh = torch.where(is_disc, torch.zeros_like(amp1), phase_slots[:, 1].to(device=device, dtype=torch.float32))
            c_2wh = c_wh.square() - s_wh.square()
            ph_vec[:, 6] = amp1 * c_wh
            ph_vec[:, 7] = -amp1 * s_wh
            ph_vec[:, 8] = amp2 * c_2wh
        else:
            # Q3_repeat_x1 or Q3_after_pilot: slots[:, 2] = cos(w*(h+ell)), slots[:, 3] = sin(w*(h+ell))
            dt3 = h + ell
            var3 = sigma.square() * dt3 + rate_var * dt3.square()
            snr = 10.0 ** (snr_db / 10.0)
            erf_s = torch.erf(torch.sqrt(snr / 2.0))
            c_hl = torch.where(dt3 <= 0.0, torch.ones_like(dt3), (1.0 - (4.0 / 3.0) * dd).clamp(0.0, 1.0).pow(dt3))
            amp1 = torch.where(is_disc, erf_s * c_hl, erf_s * torch.exp(-0.5 * var3))
            amp2 = torch.where(is_disc, erf_s.square() * c_hl, erf_s.square() * torch.exp(-2.0 * var3))
            c_3 = torch.where(is_disc, torch.ones_like(amp1), phase_slots[:, 2].to(device=device, dtype=torch.float32))
            s_3 = torch.where(is_disc, torch.zeros_like(amp1), phase_slots[:, 3].to(device=device, dtype=torch.float32))
            c_23 = c_3.square() - s_3.square()
            ph_vec[:, 10] = amp1 * c_3
            ph_vec[:, 11] = amp1 * s_3
            ph_vec[:, 12] = amp2 * c_23

    # Canonical C4 shift delta_s for mapping canonical -> physical output coordinates
    a_shift, b_shift = (codes_4 // 4).unbind(dim=-1)
    if is_q1:
        shift = (b_shift - a_shift) % 4
    elif is_indep:
        shift = torch.zeros_like(a_shift)
    else:
        shift = a_shift % 4

    return {
        "w1_vec": w1_vec,
        "w2_vec": w2_vec,
        "ph_vec": ph_vec,
        "codes_4": codes_4,
        "shift": shift,
        "task": tname,
    }


def _apply_c4_coord_shift(coords_canon: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Apply cyclic C4 permutation R_{shift} in 3D zero-sum F_real coordinates."""
    # Convert 3D canon coords -> centered 4D probs -> roll -> project back to 3D F_real coords!
    p4_zero = coords_canon @ F_REAL_4X3.to(device=coords_canon.device, dtype=coords_canon.dtype).T  # (B, L, 4) or (B, 4)
    if p4_zero.ndim == 2:
        sh = shift.unsqueeze(-1)
        idx = (torch.arange(4, device=p4_zero.device).unsqueeze(0) - sh) % 4
        lifted = p4_zero.gather(-1, idx)
    else:
        sh = shift.view(-1, 1, 1)
        idx = (torch.arange(4, device=p4_zero.device).view(1, 1, 4) - sh) % 4
        idx = idx.expand(p4_zero.shape[0], p4_zero.shape[1], 4)
        lifted = p4_zero.gather(-1, idx)
    return lifted @ F_REAL_4X3.to(device=coords_canon.device, dtype=coords_canon.dtype)


# =============================================================================
# Unified CN-SFT Adapter Container (U0 / U1 / U2 / U3)
# =============================================================================


def _extract_rotary_dim(lm: nn.Module, attn: Optional[nn.Module] = None) -> int:
    """Resolve Qwen3.5 partial_rotary_factor (inside rope_parameters in transformers>=5.24) accurately."""
    text_cfg = getattr(lm.config, "text_config", lm.config)
    d_model = int(getattr(text_cfg, "hidden_size", 2560))
    n_heads = int(getattr(text_cfg, "num_attention_heads", 16))
    hd = int(getattr(attn, "head_dim", getattr(text_cfg, "head_dim", d_model // max(1, n_heads))))
    # 1. Inspect live rotary_emb inv_freq if attached
    rot_emb = None
    if hasattr(lm, "model") and hasattr(lm.model, "language_model") and hasattr(lm.model.language_model, "rotary_emb"):
        rot_emb = lm.model.language_model.rotary_emb
    elif hasattr(lm, "model") and hasattr(lm.model, "rotary_emb"):
        rot_emb = lm.model.rotary_emb
    # Only trust rotary_emb if partial_rotary_factor < 1.0 (full RoPE should give rot_dim=0 for NoPE fallback)
    rope_params = getattr(text_cfg, "rope_parameters", None) or getattr(text_cfg, "rope_scaling", None) or {}
    part_rot = rope_params.get("partial_rotary_factor", getattr(text_cfg, "partial_rotary_factor", 1.0))
    part_rot_f = float(1.0 if part_rot is None else part_rot)
    if part_rot_f >= 0.999:
        return 0
    if rot_emb is not None and hasattr(rot_emb, "inv_freq"):
        return int(rot_emb.inv_freq.shape[0] * 2)
    return int(round(hd * part_rot_f))


class CNSFTHostAdapter(nn.Module):
    """Converged N1+N2+N3+N4 non-mutating Connecting-Neuron Surgery adapter."""

    def __init__(
        self,
        lm: nn.Module,
        u_c_d: Dict[int, torch.Tensor],
        u_c_o: Dict[int, torch.Tensor],
        u_c_mlp: Dict[int, torch.Tensor],
        u_c_bus: Optional[torch.Tensor],
        b_align_info: Dict[str, Any],
        e_bus: torch.Tensor,
        tail_rms_medians: Dict[int, float],
        fa_layers: Sequence[int] = QWEN4B_FA_LAYERS,
        dn_layers: Sequence[int] = QWEN4B_DN_LAYERS,
        mlp_tune_layers: Sequence[int] = QWEN4B_MLP_TUNE_LAYERS,
        stop_layer: int = QWEN4B_STOP_LAYER,
        arm_mode: str = "u1_full_cnsft",
        regime: int = 2,
        r_c_cayley: int = 3,
        r_s_cayley: int = 2,
        r_lora: int = 6,
        k_tune: int = 8,
        r_schur: int = 11,
        rot_dim_override: Optional[int] = None,
        l10_ridge_proj: Optional[torch.Tensor] = None,
        l10_ridge_bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if arm_mode not in ("u0_zeroshot", "u1_full_cnsft", "u2_noswiglu_control"):
            raise ValueError(f"Unsupported arm_mode={arm_mode!r}")
        self.arm_mode = arm_mode
        self.regime = int(regime)
        self.stop_layer = int(stop_layer)
        self.gate_off_audit = False

        layers = _get_decoder_layers(lm)
        n_l = len(layers)
        self.n_layers = n_l
        text_cfg = getattr(lm.config, "text_config", lm.config)
        d_model = int(getattr(text_cfg, "hidden_size", 2560))
        self.d_model = d_model

        self.g_em: int = 0
        self.m_acc_mask: Optional[torch.Tensor] = None
        self.active_carrier: Optional[Dict[str, Any]] = None
        self.active_t_div: int = 0
        self.active_task_id: Optional[int] = None
        self.apply_c4_shift: bool = False
        self._in_mock_l0: bool = False
        self._mock_l0_delta: Optional[torch.Tensor] = None
        self._cached_inj_residual: Optional[torch.Tensor] = None
        self._cached_qgate_p1_last: Optional[torch.Tensor] = None
        self._cached_qgate_seq: Optional[torch.Tensor] = None
        self._cached_dz_mask: Optional[torch.Tensor] = None
        self._cached_bridge_out: Optional[torch.Tensor] = None
        self.handles: List[object] = []

        self.register_buffer("e_bus", e_bus.float().clone())  # (D, 22)
        if l10_ridge_proj is not None:
            self.register_buffer("l10_ridge_proj", l10_ridge_proj.float().clone())
        else:
            self.register_buffer("l10_ridge_proj", torch.zeros((d_model, 16), dtype=torch.float32))
        if l10_ridge_bias is not None:
            self.register_buffer("l10_ridge_bias", l10_ridge_bias.float().clone())
        else:
            self.register_buffer("l10_ridge_bias", torch.zeros((16,), dtype=torch.float32))

        b_align = b_align_info["b_align"]
        m_eff = b_align_info["m_eff"]

        fa_valid = [i for i in fa_layers if 0 <= i < n_l]
        if not fa_valid and n_l >= 1:
            fa_valid = [n_l - 1]
        dn_valid = [i for i in dn_layers if 0 <= i < n_l]
        mlp_valid = [i for i in mlp_tune_layers if 0 <= i < n_l]

        if n_l >= 3:
            self.comp_layers = (n_l - 3, n_l - 2, n_l - 1)
        elif n_l == 2:
            self.comp_layers = (0, 0, 1)
        else:
            self.comp_layers = (0, 0, 0)

        s0 = float(tail_rms_medians.get(self.comp_layers[0], 1.0))
        s1 = float(tail_rms_medians.get(self.comp_layers[1], 1.0))
        s2 = float(tail_rms_medians.get(self.comp_layers[2], 1.0))
        sf = float(tail_rms_medians.get("final", s2))
        self.s_med_0 = s0

        first_attn = getattr(layers[fa_valid[0]], "self_attn", None) if fa_valid else None
        rot_dim_cfg = int(rot_dim_override) if rot_dim_override is not None else _extract_rotary_dim(lm, first_attn)

        self.fa_cayleys = nn.ModuleDict()
        self.fa_loras = nn.ModuleDict()
        self.dn_cayleys = nn.ModuleDict()
        self.dn_loras = nn.ModuleDict()
        self.dn_gate_loras = nn.ModuleDict()
        self.mlp_overlays = nn.ModuleDict()

        # 1. Carrier Calibration (256 + 2*n_dn params)
        self.carrier_calib = nn.Linear(16, 16, bias=False)
        with torch.no_grad():
            self.carrier_calib.weight.copy_(torch.eye(16, dtype=torch.float32))
        self.slow_head_bias = nn.Parameter(torch.zeros(max(1, len(dn_valid)), 2, dtype=torch.float32))

        # 2. Shadow GDN Bridge + N1 Circuit A + Gated Schur Readout
        use_swiglu_compiled = arm_mode in ("u0_zeroshot", "u1_full_cnsft")
        self.shadow_gdn: Optional[ShadowGDNBallastBridge] = None
        self.circuit_a: Optional[CompiledN1CircuitAOverlay] = None
        if use_swiglu_compiled:
            self.shadow_gdn = ShadowGDNBallastBridge(self.e_bus, s_med_0=s0, s_med_1=s1, s_med_2=s2)
            g0 = extract_rmsnorm_effective_gamma(layers[self.comp_layers[0]].post_attention_layernorm).cpu()
            g1 = extract_rmsnorm_effective_gamma(layers[self.comp_layers[1]].post_attention_layernorm).cpu()
            g2 = extract_rmsnorm_effective_gamma(layers[self.comp_layers[2]].post_attention_layernorm).cpu()
            self.circuit_a = CompiledN1CircuitAOverlay(
                e_bus=self.e_bus,
                b_align=b_align,
                m_eff=m_eff,
                gamma_0=g0,
                gamma_1=g1,
                gamma_2=g2,
                s_med_0=s0,
                s_med_1=s1,
                s_med_2=s2,
                s_med_f=sf,
            )

        u_c_last = u_c_d.get(fa_valid[-1], u_c_bus)
        self.schur_readout = GatedSchurIrrepReadout(d_model, b_align=b_align, u_c_last=u_c_last, r=r_schur)

        # 3. Sparse Fine-Tuning Modules (enabled for u1_full_cnsft and u2_noswiglu_control)
        if arm_mode != "u0_zeroshot":
            for idx in fa_valid:
                attn = layers[idx].self_attn
                o_in = int(attn.o_proj.in_features)
                uc_o = u_c_o.get(idx)
                assert uc_o is not None and uc_o.shape[0] * 2 == o_in, f"Missing head K-FAC u_c_o for FA layer {idx}"
                self.fa_cayleys[str(idx)] = SpecCayley(o_in, r_c=r_c_cayley, r_s=r_s_cayley, u_c=uc_o)

                uc_d = u_c_d.get(idx)
                hd = int(getattr(attn, "head_dim", o_in // max(1, int(getattr(attn, "num_heads", 16)))))
                for pname in ("q_proj", "k_proj", "v_proj"):
                    proj = getattr(attn, pname)
                    if pname == "q_proj" and hasattr(attn, "num_key_value_groups"):
                        interleaved = (proj.out_features == 2 * o_in)
                        self.fa_loras[f"{idx}_{pname}"] = FoldedNoPELoRADelta(
                            proj.in_features,
                            proj.out_features,
                            r=r_lora,
                            u_c=uc_d,
                            head_dim=hd,
                            rot_dim=rot_dim_cfg,
                            interleaved_gate=interleaved,
                        )
                    elif pname == "k_proj" and rot_dim_cfg > 0:
                        self.fa_loras[f"{idx}_{pname}"] = FoldedNoPELoRADelta(
                            proj.in_features,
                            proj.out_features,
                            r=r_lora,
                            u_c=uc_d,
                            head_dim=hd,
                            rot_dim=rot_dim_cfg,
                            interleaved_gate=False,
                        )
                    else:
                        self.fa_loras[f"{idx}_{pname}"] = FoldedNoPELoRADelta(
                            proj.in_features, proj.out_features, r=r_lora, u_c=uc_d
                        )

            for idx in dn_valid:
                dn = layers[idx].linear_attn
                dn_o_in = int(dn.out_proj.in_features)
                uc_o = u_c_o.get(idx)
                assert uc_o is not None and uc_o.shape[0] * 2 == dn_o_in, f"Missing head K-FAC u_c_o for DN layer {idx}"
                self.dn_cayleys[str(idx)] = SpecCayley(dn_o_in, r_c=r_c_cayley, r_s=r_s_cayley, u_c=uc_o)

                uc_d = u_c_d.get(idx)
                kv_start, kv_end = int(dn.key_dim), int(2 * dn.key_dim + dn.value_dim)
                self.dn_loras[f"{idx}_in_proj_kv"] = FoldedNoPELoRADelta(
                    dn.in_proj_qkv.in_features, dn.in_proj_qkv.out_features, r=r_lora, u_c=uc_d, out_slice=(kv_start, kv_end)
                )
                # Tier-2 UNMASKED in_proj_a (fixes line 538 bug!), Tier-1 masked in_proj_b
                self.dn_gate_loras[f"{idx}_in_proj_a"] = FoldedNoPELoRADelta(
                    dn.in_proj_a.in_features, dn.in_proj_a.out_features, r=4, u_c=uc_d
                )
                self.dn_gate_loras[f"{idx}_in_proj_b"] = FoldedNoPELoRADelta(
                    dn.in_proj_b.in_features, dn.in_proj_b.out_features, r=4, u_c=uc_d
                )

            if arm_mode == "u1_full_cnsft":
                for idx in mlp_valid:
                    g_inv = safe_gamma_inv(extract_rmsnorm_effective_gamma(layers[idx].post_attention_layernorm)).cpu()
                    self.mlp_overlays[str(idx)] = SparseAntipodalSwiGLUOverlay(
                        d_model, k_tune=k_tune, u_c=u_c_mlp.get(idx, u_c_d.get(idx)), gamma_inv=g_inv
                    )
        else:
            # Zero-shot u0: freeze trainable fine-tuning scalars cleanly
            self.carrier_calib.requires_grad_(False)
            self.slow_head_bias.requires_grad_(False)
            self.schur_readout.requires_grad_(False)
            if self.shadow_gdn is not None:
                self.shadow_gdn.requires_grad_(False)
            if self.circuit_a is not None:
                self.circuit_a.requires_grad_(False)

        # In Regime 1 (if L10 preserves carrier), leave carrier_calib non-trainable unless Regime 2
        if self.regime == 1 and arm_mode != "u0_zeroshot":
            self.carrier_calib.requires_grad_(False)

        self._attach_hooks(lm)

    def _masked_write_tensor(self, x: torch.Tensor) -> torch.Tensor:
        m = self.m_acc_mask
        if m is None:
            return x
        if x.shape[0] != m.shape[0]:
            if x.shape[0] % m.shape[0] == 0:
                m = m.repeat_interleave(x.shape[0] // m.shape[0], dim=0)
            else:
                return torch.zeros_like(x)
        seq_l = x.shape[1]
        m_len = m.shape[1]
        if seq_l == m_len:
            return x * m.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
        if seq_l < m_len:
            return torch.zeros_like(x)
        zeros = torch.zeros((x.shape[0], seq_l - m_len), device=x.device, dtype=x.dtype)
        return x * torch.cat((m.to(device=x.device, dtype=x.dtype), zeros), dim=1).unsqueeze(-1)

    def _identify_prefix_marker_positions(self, m_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Locate the first <W1> (2 tok), <W2> (2 tok), <PHASE> (2 tok) indices inside m_acc_mask."""
        # m_mask: (B, m_len) bool -> deterministic across batch since prompt template is fixed
        row = m_mask[0].bool()
        true_pos = torch.nonzero(row, as_tuple=False).flatten()
        # First 6 True positions belong to <W1>(0,1), <W2>(2,3), <PHASE>(4,5)
        w1_idx = true_pos[0:2] if true_pos.numel() >= 2 else true_pos
        w2_idx = true_pos[2:4] if true_pos.numel() >= 4 else true_pos[0:0]
        ph_idx = true_pos[4:6] if true_pos.numel() >= 6 else true_pos[0:0]
        return w1_idx, w2_idx, ph_idx

    def _build_carrier_delta(self, h_entry: torch.Tensor) -> torch.Tensor:
        """Assemble Regime-2 orthogonal linear carrier at EM prefix markers inside e_bus (0 on prose)."""
        if self.active_carrier is None or self.m_acc_mask is None:
            return torch.zeros_like(h_entry)
        b, seq_l, _ = h_entry.shape
        m = self.m_acc_mask
        m_len = m.shape[1]
        if seq_l < m_len:
            return torch.zeros_like(h_entry)

        c_info = self.active_carrier
        w1_v = self.carrier_calib(c_info["w1_vec"].to(device=h_entry.device, dtype=torch.float32))
        w2_v = self.carrier_calib(c_info["w2_vec"].to(device=h_entry.device, dtype=torch.float32))
        ph_v = self.carrier_calib(c_info["ph_vec"].to(device=h_entry.device, dtype=torch.float32))
        if w1_v.shape[0] != b and b % w1_v.shape[0] == 0:
            rep = b // w1_v.shape[0]
            w1_v = w1_v.repeat_interleave(rep, dim=0)
            w2_v = w2_v.repeat_interleave(rep, dim=0)
            ph_v = ph_v.repeat_interleave(rep, dim=0)

        w1_idx, w2_idx, ph_idx = self._identify_prefix_marker_positions(m)
        bus_16 = self.e_bus[:, :16].to(device=h_entry.device, dtype=torch.float32)  # (D, 16)
        delta_bus = torch.zeros((b, seq_l, 16), device=h_entry.device, dtype=torch.float32)
        if w1_idx.numel() > 0:
            delta_bus[:, w1_idx, :] = w1_v.unsqueeze(1)
        if w2_idx.numel() > 0:
            delta_bus[:, w2_idx, :] = w2_v.unsqueeze(1)
        if ph_idx.numel() > 0:
            delta_bus[:, ph_idx, :] = ph_v.unsqueeze(1)
        return (delta_bus @ bus_16.T).to(h_entry.dtype)

    def _target_digit_mask(self, seq_l: int, b: int, device: torch.device) -> torch.Tensor:
        """Select ONLY the exact digit-predicting position across KV-cache Pass 1/Pass 2 or uncached."""
        mask = torch.zeros((b, seq_l), device=device, dtype=torch.float32)
        t_div = int(self.active_t_div)
        if self.m_acc_mask is None:
            # General pure/mix single-pass forward without score_candidates_kv
            mask[:, -1] = 1.0
            return mask
        m_len = int(self.m_acc_mask.shape[1])
        if seq_l == m_len:
            if t_div == 0:
                mask[:, seq_l - 1] = 1.0
        elif seq_l < m_len:
            # KV-cached Pass 2 continuation (step_len = max_len - 1)
            pos = t_div - 1
            if 0 <= pos < seq_l:
                mask[:, pos] = 1.0
        else:
            # Uncached batched forward: prefix + candidate suffix
            pos = m_len - 1 + t_div
            if 0 <= pos < seq_l:
                mask[:, pos] = 1.0
        return mask

    def _effective_gate_mask(self, h_cur: torch.Tensor, require_dz: bool = False) -> torch.Tensor:
        b, seq_l, _ = h_cur.shape
        pos_mask = self._target_digit_mask(seq_l, b, h_cur.device)
        if self.gate_off_audit:
            return pos_mask
        # Combine Pass 1 Fisher Query Gate + optional Stationary Dead-Zone Key
        qg = self._cached_qgate_seq
        if qg is not None and qg.shape[0] == b and qg.shape[1] == seq_l:
            pos_mask = pos_mask * qg
        elif self._cached_qgate_p1_last is not None:
            qp1 = self._cached_qgate_p1_last
            if qp1.shape[0] != b and b % qp1.shape[0] == 0:
                qp1 = qp1.repeat_interleave(b // qp1.shape[0], dim=0)
            pos_mask = pos_mask * qp1.unsqueeze(1)
        if require_dz and self._cached_dz_mask is not None:
            dz = self._cached_dz_mask
            if dz.shape[0] == b and dz.shape[1] == seq_l:
                pos_mask = pos_mask * dz
        return pos_mask

    def _shift_fn(self, canon_coords: torch.Tensor) -> torch.Tensor:
        if self.active_carrier is None or "shift" not in self.active_carrier:
            return canon_coords
        shift = self.active_carrier["shift"].to(device=canon_coords.device)
        b = canon_coords.shape[0]
        if shift.shape[0] != b and b % shift.shape[0] == 0:
            shift = shift.repeat_interleave(b // shift.shape[0], dim=0)
        return _apply_c4_coord_shift(canon_coords, shift)

    def _update_qgate_from_tensor(self, x: torch.Tensor) -> None:
        seq_l = x.shape[1]
        m_len = int(self.m_acc_mask.shape[1]) if self.m_acc_mask is not None else seq_l
        qg_full = self.schur_readout.compute_qgate(x, hard_plateau=(not self.gate_off_audit))
        if seq_l >= m_len:
            self._cached_qgate_p1_last = qg_full[:, m_len - 1]
            if seq_l == m_len or self.m_acc_mask is None:
                self._cached_qgate_seq = qg_full
            else:
                # Uncached continuation positions reuse Pass-1 prompt verdict for parity with KV-cached Pass 2
                suf = self._cached_qgate_p1_last.unsqueeze(1).expand(x.shape[0], seq_l - m_len)
                self._cached_qgate_seq = torch.cat((qg_full[:, :m_len], suf), dim=1)
        else:
            if self._cached_qgate_p1_last is not None:
                qp1 = self._cached_qgate_p1_last
                if qp1.shape[0] != x.shape[0] and x.shape[0] % qp1.shape[0] == 0:
                    qp1 = qp1.repeat_interleave(x.shape[0] // qp1.shape[0], dim=0)
                self._cached_qgate_seq = qp1.unsqueeze(1).expand(x.shape[0], seq_l)

    @staticmethod
    def _get_layer_mixer(blk: nn.Module) -> Optional[nn.Module]:
        if getattr(blk, "block_type", None) == "linear_attention" and hasattr(blk, "linear_attn"):
            return blk.linear_attn
        if hasattr(blk, "self_attn"):
            return blk.self_attn
        if hasattr(blk, "linear_attn"):
            return blk.linear_attn
        return None

    def _gate_delta_by_qgate(self, delta: torch.Tensor) -> torch.Tensor:
        if self.gate_off_audit or delta.ndim < 2:
            return delta
        b_d, l_d = delta.shape[:2]
        qg_seq = self._cached_qgate_seq
        if qg_seq is not None and qg_seq.shape[:2] == (b_d, l_d):
            return delta * qg_seq.to(delta.dtype).unsqueeze(-1)
        if self._cached_qgate_p1_last is not None:
            qp1 = self._cached_qgate_p1_last
            if qp1.shape[0] != b_d and b_d % qp1.shape[0] == 0:
                qp1 = qp1.repeat_interleave(b_d // qp1.shape[0], dim=0)
            if qp1.shape[0] == b_d:
                return delta * qp1.to(delta.dtype).view(b_d, 1, 1)
        return delta

    def _attach_hooks(self, lm: nn.Module) -> None:
        layers = _get_decoder_layers(lm)
        if not layers:
            return

        # Hook A: Layer 0 entry (zero host mutation; ShadowGDN Pass 1/Pass 2 scan on pure Regime-2 carrier + shallow qgate)
        def _l0_pre(mod, args):
            self._cached_qgate_seq = None
            self._cached_dz_mask = None
            self._cached_bridge_out = None
            self._mock_l0_delta = None
            self._cached_inj_residual = None
            h_in = args[0]
            m_len = int(self.m_acc_mask.shape[1]) if self.m_acc_mask is not None else h_in.shape[1]
            if self.g_em == 0 or self.m_acc_mask is None or h_in.shape[1] >= m_len:
                self._cached_qgate_p1_last = None
            if self.g_em == 0:
                if self.shadow_gdn is not None:
                    self.shadow_gdn.clear_state()
                return args
            if self.regime == 2 and self.shadow_gdn is not None:
                carrier_only = (
                    self._build_carrier_delta(h_in) if self.active_carrier is not None else torch.zeros_like(h_in)
                )
                self._cached_bridge_out = self.shadow_gdn.forward_pass(carrier_only, self.m_acc_mask)
            if len(layers) < 4:
                self._update_qgate_from_tensor(h_in.detach())
            return args

        self.handles.append(layers[0].register_forward_pre_hook(_l0_pre))

        # Hook B: Mode I Stop-Grad at stop_layer + qgate (on real deep hosts) + Regime-1 ShadowGDN bridge
        if 0 <= self.stop_layer < len(layers):
            def _stop_hook(mod, args, out):
                if self.g_em == 0:
                    return out
                h_raw = out[0] if isinstance(out, tuple) else out
                h_det = h_raw.detach()
                if len(layers) >= 4:
                    self._update_qgate_from_tensor(h_det)
                if self.regime == 1 and self.shadow_gdn is not None:
                    bus_proj = (h_det.float() @ self.l10_ridge_proj + self.l10_ridge_bias) @ self.e_bus[:, :16].T
                    self._cached_bridge_out = self.shadow_gdn.forward_pass(bus_proj, self.m_acc_mask)
                if isinstance(out, tuple):
                    return (h_det, *out[1:])
                return h_det

            self.handles.append(layers[self.stop_layer].register_forward_hook(_stop_hook))

        # Hook D0/D1/D2: Capture inject residual, zero host bus leak + inject ShadowGDN on mixer post-hook, record DZ
        inject_layer = self.comp_layers[0]
        blk_inj = layers[inject_layer]
        mixer_inj = self._get_layer_mixer(blk_inj)

        def _capture_inj_res_pre(mod, args):
            if self.g_em != 0 and self.circuit_a is not None:
                self._cached_inj_residual = args[0]
            else:
                self._cached_inj_residual = None
            return args

        self.handles.append(blk_inj.register_forward_pre_hook(_capture_inj_res_pre))

        def _inject_bridge_mixer_post(mod, args, out):
            if self.g_em == 0 or self._cached_bridge_out is None or self.circuit_a is None:
                self._cached_inj_residual = None
                return out
            bridge_active = bool((self._cached_bridge_out.abs().max() > 1e-6).item())
            if not bridge_active:
                self._cached_inj_residual = None
                return out
            h_out = out[0] if isinstance(out, tuple) else out
            qg_mask = self._effective_gate_mask(h_out, require_dz=False)
            res_0 = self._cached_inj_residual
            self._cached_inj_residual = None
            if res_0 is not None and res_0.shape == h_out.shape:
                x_0_host = res_0 + h_out
                host_bus_0 = (x_0_host.float() @ self.e_bus) @ self.e_bus.T
                bridge_delta = ((self._cached_bridge_out - host_bus_0) * qg_mask.unsqueeze(-1)).to(h_out.dtype)
            else:
                bridge_delta = (self._cached_bridge_out * qg_mask.unsqueeze(-1)).to(h_out.dtype)
            h_new = h_out + bridge_delta
            if isinstance(out, tuple):
                return (h_new, *out[1:])
            return h_new

        if mixer_inj is not None:
            self.handles.append(mixer_inj.register_forward_hook(_inject_bridge_mixer_post))

        def _record_dz_pre(mod, args):
            if self._in_mock_l0 or self.g_em == 0:
                return args
            x = args[0]
            bus_norm = ((x.detach().float() @ self.e_bus[:, :13]).norm(dim=-1)) / max(1e-4, self.s_med_0)
            if self.circuit_a is not None:
                tau_lo = self.schur_readout.dz_tau.clamp(0.15, 0.45)
                tau_hi = self.schur_readout.dz_high.clamp(0.55, 0.85)
                self._cached_dz_mask = torch.clamp((bus_norm - tau_lo) / (tau_hi - tau_lo).clamp_min(0.1), 0.0, 1.0)
                if self.comp_layers[0] == self.comp_layers[1]:
                    self._in_mock_l0 = True
                    try:
                        z0 = mod(x)
                    finally:
                        self._in_mock_l0 = False
                    gmask0 = self._effective_gate_mask(x, require_dz=True)
                    d0 = self.circuit_a.l0_delta(z0, gate_mask=gmask0)
                    self._mock_l0_delta = d0
                    return (x + d0, *args[1:])
            else:
                self._cached_dz_mask = torch.clamp((bus_norm - 0.25) / 0.40, 0.0, 1.0)
            return args

        self.handles.append(blk_inj.post_attention_layernorm.register_forward_pre_hook(_record_dz_pre))

        # Hook E: Sparse Fine-Tuned SpecCayley + NoPE FA LoRA + Unmasked Tier-2 DN Bridge LoRA
        for k_str, cay in self.fa_cayleys.items():
            idx = int(k_str)
            attn = layers[idx].self_attn

            def mk_cay_pre(mod):
                def _pre(m, args):
                    if self.g_em == 0:
                        return args
                    x = args[0]
                    delta = self._gate_delta_by_qgate(mod(x) - x)
                    return (x + delta, *args[1:])
                return _pre

            self.handles.append(attn.o_proj.register_forward_pre_hook(mk_cay_pre(cay)))

            for pname, masked_m in (("q_proj", False), ("k_proj", True), ("v_proj", True)):
                lmod = self.fa_loras[f"{idx}_{pname}"]
                proj = getattr(attn, pname)

                def mk_fa_post(mod, apply_m: bool):
                    def _post(m, args, out):
                        if self.g_em == 0:
                            return out
                        x = self._masked_write_tensor(args[0]) if apply_m else args[0]
                        delta = mod(x)
                        if not apply_m:
                            delta = self._gate_delta_by_qgate(delta)
                        return mod.add_into_output(out, delta)
                    return _post

                self.handles.append(proj.register_forward_hook(mk_fa_post(lmod, masked_m)))

        for dn_pos, (k_str, cay) in enumerate(self.dn_cayleys.items()):
            idx = int(k_str)
            dn = layers[idx].linear_attn

            def mk_dn_cay(mod):
                def _pre(m, args):
                    if self.g_em == 0:
                        return args
                    x = args[0]
                    delta = self._gate_delta_by_qgate(mod(x) - x)
                    return (x + delta, *args[1:])
                return _pre

            self.handles.append(dn.out_proj.register_forward_pre_hook(mk_dn_cay(cay)))

            kv_lora = self.dn_loras[f"{idx}_in_proj_kv"]

            def mk_dn_kv(mod):
                def _post(m, args, out):
                    if self.g_em == 0:
                        return out
                    delta = mod(self._masked_write_tensor(args[0]))
                    return mod.add_into_output(out, delta)
                return _post

            self.handles.append(dn.in_proj_qkv.register_forward_hook(mk_dn_kv(kv_lora)))

            # in_proj_a: UNMASKED Tier-2 + slow_head_bias (both qgate-gated)! in_proj_b: MASKED Tier-1!
            a_lora = self.dn_gate_loras[f"{idx}_in_proj_a"]
            b_lora = self.dn_gate_loras[f"{idx}_in_proj_b"]

            def mk_a_post(mod, b_idx: int):
                def _post(m, args, out):
                    if self.g_em == 0:
                        return out
                    delta_a = self._gate_delta_by_qgate(mod(args[0]))
                    res = out + delta_a
                    k_bias = min(2, res.shape[-1])
                    hb = self.slow_head_bias[b_idx, :k_bias].to(dtype=res.dtype).view(1, 1, k_bias).expand(
                        res.shape[0], res.shape[1], k_bias
                    )
                    hb = self._gate_delta_by_qgate(hb)
                    res = res.clone()
                    res[..., :k_bias] = res[..., :k_bias] + hb
                    return res
                return _post

            def mk_b_post(mod):
                def _post(m, args, out):
                    if self.g_em == 0:
                        return out
                    return out + mod(self._masked_write_tensor(args[0]))
                return _post

            self.handles.append(dn.in_proj_a.register_forward_hook(mk_a_post(a_lora, dn_pos)))
            self.handles.append(dn.in_proj_b.register_forward_hook(mk_b_post(b_lora)))

        # Hook F: Sparse Fine-Tuned FP32 Antipodal SwiGLU Overlays (residual layers 15, 19, 23)
        for k_str, mlp_ov in self.mlp_overlays.items():
            idx = int(k_str)
            mlp_mod = layers[idx].mlp

            def mk_mlp_ov(mod):
                def _post(m, args, out):
                    if self.g_em == 0:
                        return out
                    gmask = self._effective_gate_mask(args[0], require_dz=False)
                    return out + mod(args[0], gate_mask=gmask)
                return _post

            self.handles.append(mlp_mod.register_forward_hook(mk_mlp_ov(mlp_ov)))

        # Hook G: Compiled N1 Circuit A (FP32 3-layer Fermi-Dirac SwiGLU Overlay + gated P_{perp,bus} hygiene)
        if self.circuit_a is not None:
            l0_idx, l1_idx, l2_idx = self.comp_layers

            def _clear_bus_slice(t: torch.Tensor, gmask: torch.Tensor, cols: Tuple[int, ...]) -> torch.Tensor:
                if not bool((gmask > 0.0).any()):
                    return t
                e_sub = self.e_bus[:, list(cols)]
                proj = ((t.float() @ e_sub) * gmask.float().unsqueeze(-1)) @ e_sub.T
                return t - proj.to(t.dtype)

            if l0_idx != l1_idx:
                def _c_l0(m, args, out):
                    if self.g_em == 0:
                        return out
                    gmask = self._effective_gate_mask(args[0], require_dz=True)
                    out_clean = _clear_bus_slice(out, gmask, (0, 1, 2, 13, 14, 15, 16, 17, 18, 19, 20, 21))
                    return out_clean + self.circuit_a.l0_delta(args[0], gate_mask=gmask)

                self.handles.append(layers[l0_idx].mlp.register_forward_hook(_c_l0))

                mixer_l1 = self._get_layer_mixer(layers[l1_idx])
                if mixer_l1 is not None:
                    def _clear_l1_mixer_post(m, args, out):
                        if self.g_em == 0:
                            return out
                        h_out = out[0] if isinstance(out, tuple) else out
                        gmask = self._effective_gate_mask(h_out, require_dz=True)
                        h_clean = _clear_bus_slice(h_out, gmask, (0, 1, 2, 13, 14, 15, 16, 17, 18, 19, 20, 21))
                        return (h_clean, *out[1:]) if isinstance(out, tuple) else h_clean

                    self.handles.append(mixer_l1.register_forward_hook(_clear_l1_mixer_post))

            def _c_l1(m, args, out):
                if self.g_em == 0:
                    return out
                gmask = self._effective_gate_mask(args[0], require_dz=True)
                out_clean = _clear_bus_slice(out, gmask, (13, 18, 19, 20, 21))
                d1 = self.circuit_a.l1_delta(args[0], gate_mask=gmask)
                if self._mock_l0_delta is not None and self.comp_layers[0] == self.comp_layers[1]:
                    return out_clean + self._mock_l0_delta + d1
                return out_clean + d1

            if l2_idx != l1_idx:
                mixer_l2 = self._get_layer_mixer(layers[l2_idx])
                if mixer_l2 is not None:
                    def _clear_l2_mixer_post(m, args, out):
                        if self.g_em == 0:
                            return out
                        h_out = out[0] if isinstance(out, tuple) else out
                        gmask = self._effective_gate_mask(h_out, require_dz=True)
                        h_clean = _clear_bus_slice(h_out, gmask, (13, 18, 19, 20, 21))
                        return (h_clean, *out[1:]) if isinstance(out, tuple) else h_clean

                    self.handles.append(mixer_l2.register_forward_hook(_clear_l2_mixer_post))

            def _c_l2(m, args, out):
                if self.g_em == 0:
                    return out
                gmask = self._effective_gate_mask(args[0], require_dz=True)
                return out + self.circuit_a.l2_delta(
                    args[0],
                    gate_mask=gmask,
                    mlp_out=out,
                    lift_shift_fn=(self._shift_fn if self.apply_c4_shift else None),
                    task_id_val=self.active_task_id,
                )

            self.handles.append(layers[l1_idx].mlp.register_forward_hook(_c_l1))
            self.handles.append(layers[l2_idx].mlp.register_forward_hook(_c_l2))

        # Hook H: Gated D_{gamma_f} Schur Readout on last decoder layer output (require Stationary Dead-Zone!)
        def _schur_last(m, args, out):
            if self.g_em == 0 or self.arm_mode == "u0_zeroshot":
                return out
            h = out[0] if isinstance(out, tuple) else out
            gmask = self._effective_gate_mask(h, require_dz=(self.circuit_a is not None))
            delta = self.schur_readout.forward_delta(h, gate_mask=gmask)
            if isinstance(out, tuple):
                return (h + delta, *out[1:])
            return h + delta

        self.handles.append(layers[-1].register_forward_hook(_schur_last))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles.clear()

    @contextmanager
    def activate(
        self,
        g_em: int = 1,
        m_acc_mask: Optional[torch.Tensor] = None,
        carrier: Optional[Dict[str, Any]] = None,
        t_div: int = 0,
        task_id: Optional[int] = None,
        apply_c4_shift: bool = False,
    ):
        prev = (
            self.g_em,
            self.m_acc_mask,
            self.active_carrier,
            self.active_t_div,
            self.active_task_id,
            self.apply_c4_shift,
        )
        self.g_em = int(bool(g_em))
        self.m_acc_mask = m_acc_mask
        self.active_carrier = carrier
        self.active_t_div = int(t_div)
        self.active_task_id = task_id
        self.apply_c4_shift = bool(apply_c4_shift)
        try:
            yield
        finally:
            (
                self.g_em,
                self.m_acc_mask,
                self.active_carrier,
                self.active_t_div,
                self.active_task_id,
                self.apply_c4_shift,
            ) = prev

    def tier2_kfac_penalty(self) -> torch.Tensor:
        """Population K-FAC curvature regularizer on Tier-2 prose-active modules + mild prefix shrinkage."""
        reg = torch.zeros((), device=self.e_bus.device, dtype=torch.float32)
        for cay in (*self.fa_cayleys.values(), *self.dn_cayleys.values()):
            reg = reg + cay.sigma_raw.float().pow(2).sum() + cay.v_re.float().pow(2).sum() + cay.v_im.float().pow(2).sum()
        for k, mod in self.fa_loras.items():
            w_coef = 0.1 if k.endswith("_q_proj") else 1.5
            reg = reg + mod.lora_b.float().pow(2).sum() * w_coef
        for mod in self.dn_loras.values():
            reg = reg + mod.lora_b.float().pow(2).sum() * 1.5
        for k, mod in self.dn_gate_loras.items():
            w_coef = 0.25 if k.endswith("_in_proj_a") else 1.5
            reg = reg + mod.lora_b.float().pow(2).sum() * w_coef
        reg = reg + self.schur_readout.proj_coord.float().pow(2).sum() * 0.1
        return reg

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def buffer_count(self) -> int:
        return sum(b.numel() for b in self.buffers())


def first_divergent_token_index(host, choices: Tuple[str, ...]) -> int:
    """Compute exact 0-based index where candidate choice tokenizations first diverge."""
    if getattr(host, "tokenizer", None) is None:
        return 0
    id_lists = [host.tokenizer.encode(ch, add_special_tokens=False) for ch in choices]
    min_len = min(len(x) for x in id_lists)
    for t in range(min_len):
        toks_t = {x[t] for x in id_lists}
        if len(toks_t) > 1:
            return t
    return 0


def score_batch_with_cn_sft(
    connector,
    host,
    user_prompt: str,
    choices: Tuple[str, ...],
    task: str,
    words: torch.Tensor,
    bits: int,
    phase_slots: torch.Tensor,
    meta_scalars: torch.Tensor,
    adapter: Optional[CNSFTHostAdapter],
    ref_summary: Optional[torch.Tensor] = None,
    valid: Optional[torch.Tensor] = None,
    scorer_mode: str = "kv_cached",
    temperature: float = 1.0,
    discrete_d: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score batch through frozen Phase-A connector + CNSFTHostAdapter without mutating connector."""
    from research.stage4_real_and_surgical.equi_connector import TASK_TO_ID, lift_scores

    b = words.shape[0]
    task_id_val = int(TASK_TO_ID[task])
    task_id = torch.full((b,), task_id_val, device=words.device, dtype=torch.long)
    soft_tokens, codes, eff_valid, latent_h = connector.forward_tokens(
        words, bits, task_id, phase_slots, meta_scalars, ref_summary=ref_summary, valid=valid
    )
    prefix_embeds, m_acc_mask = connector.build_host_prefix_embeds(host, user_prompt, soft_tokens)
    t_div = first_divergent_token_index(host, choices)

    carrier_info = None
    if adapter is not None and adapter.regime == 2:
        carrier_info = build_regime2_separate_carrier(
            connector.prior_canon,
            words,
            task=task,
            phase_slots=phase_slots,
            meta_scalars=meta_scalars,
            ref_summary=ref_summary,
            discrete_d=discrete_d,
            valid=valid,
            bits=bits,
        )

    ctx = (
        adapter.activate(
            g_em=1,
            m_acc_mask=m_acc_mask,
            carrier=carrier_info,
            t_div=t_div,
            task_id=task_id_val,
            apply_c4_shift=(not connector.quotient),
        )
        if adapter is not None
        else None
    )
    if ctx is not None:
        ctx.__enter__()
    try:
        if getattr(host, "tokenizer", None) is None:
            seq_len = prefix_embeds.shape[1]
            attn = torch.ones((b, seq_len), device=host.device, dtype=torch.long)
            out = host.lm(inputs_embeds=prefix_embeds, attention_mask=attn, use_cache=False)
            raw_logits = out.logits[:, -1, :4].float()
        elif scorer_mode == "kv_cached":
            _, raw_logits = host.score_candidates_kv(prefix_embeds, choices, temperature=1.0)
        elif scorer_mode == "batched_uncached":
            _, raw_logits = host.score_candidates_batched_uncached(prefix_embeds, choices, temperature=1.0)
        else:
            _, raw_logits = host.score_candidates_naive_uncached(prefix_embeds, choices, temperature=1.0)
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)

    centered_logits = raw_logits - raw_logits.mean(dim=-1, keepdim=True)
    if not connector.quotient:
        scaled = centered_logits / float(temperature)
        lp = F.log_softmax(scaled, dim=-1)
        return lp, centered_logits, lp.exp(), latent_h

    canon_probs = F.softmax(centered_logits / float(temperature), dim=-1)
    probs = lift_scores(canon_probs, codes, task)
    return torch.log(probs.clamp_min(1e-12)), centered_logits, probs, latent_h
