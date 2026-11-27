"""Stage 4 Phase B (Mode-I Surgical-96 vs Matched LEMMO Tier-P1), PureKL/MixKL audit, and 7-model synthetic blind."""

from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.stage3_ministral_cache import analytic as s3_analytic
from research.stage3_ministral_cache.ministral_connector import _compute_host_fingerprint
from research.stage3_ministral_cache.schema import ROOTS, ROTATIONS, matched_sum, sector_of_complex
from research.stage3d_qwen_signed_phase.data_stage3d import (
    BLIND_CELLS,
    NS_BLIND_Q2_C4,
    Q2_BLIND_ARMS,
    build_q2_arm_references,
    build_stage3d_prompt,
    condition_spec_from_params,
    generate_paired_cfo_blocks,
    load_s17_encoder,
    public_phase_slots,
)
from research.stage3d_qwen_signed_phase.qwen_connector import (
    MODEL_ID_DEFAULT,
    PINNED_REVISION_DEFAULT,
    QwenCacheHost,
)
from research.stage4_real_and_surgical.equi_connector import (
    CANONICAL_TASK_NAME,
    PhysicalKernelS17,
    Stage4MatchedConnector,
    extract_canonical_ref_summary,
    lift_scores,
)
from research.stage4_real_and_surgical.surgical_adapter import (
    QWEN4B_DN_LAYERS,
    QWEN4B_FA_LAYERS,
    QWEN4B_STOP_LAYER,
    TIER_P1_MAX_PARAMS,
    SurgicalHostAdapter,
    UnconstrainedLoRAAdapter,
    _get_decoder_layers,
    calibrate_b_align_from_host,
    rao_blackwell_bayes_kl_cf_loss,
)
from research.stage4_real_and_surgical.train_eval_stage4 import (
    EXPECTED_4B_HOST_SHA256,
    EXPECTED_4B_VOCAB,
    _evaluate_dev_groups,
    build_stage4_pilot_groups,
    collect_kfac_complex_projectors,
    measure_candidate_jacobian,
)

GENERAL_TEXT_TRAIN_PROMPTS: Tuple[str, ...] = (
    "解释为什么在线性注意力与全注意力混合架构中，正交零空间约束能够避免灾难性遗忘：",
    '{"service": "telemetry-sync", "status": "healthy", "latency_ms": 12.4, "retries":',
    "def moving_variance(xs):\n    mean = sum(xs) / max(1, len(xs))\n    return sum((x - mean) ** 2 for x in",
    "Briefly summarize why complex baseband QPSK phase estimation uses matched filtering prior to hard slicing.",
    "在随机游走信道下，离散网格积分与一阶线性逼近之间的误差边界取决于",
    '{"experiment": "stage4", "gate": 2, "metrics": {"kl_gap": 0.0027, "passed":',
    "Write a concise SQL query that counts active sessions grouped by protocol version:",
    "When updating state via an online Kalman filter, the optimal gain balances innovation precision against prior",
)

GENERAL_TEXT_EVAL_PROMPTS: Tuple[str, ...] = (
    "请简述最大似然估计与贝叶斯后验均值在低信噪比下的差异：",
    '{"node_id": "eu-west-04", "checksum": "a94f1c", "packets_lost":',
    "def circular_distance(a, b):\n    d = (a - b) % 4\n    return min(d, 4 -",
    "Explain the purpose of guard bands between train, dev, and blind splits in radio waveform evaluation.",
    "复指数载波频偏在离散时间采样后体现为线性相位斜率，因此",
    '{"pipeline": "m5-c16-v1-le", "valid": true, "snr_db":',
    "class RingBuffer:\n    def __init__(self, cap):\n        self.buf = [None] *",
    "State the Cauchy-Schwarz inequality for inner products of complex Hilbert space vectors.",
    "多轮对话中保持未激活模态零干扰的最严格条件是门控开关满足",
    '{"audit": "pure_kl", "threshold": 0.001, "verdict":',
    "import math\ndef qpsk_angle(x, y):\n    return math.atan2(",
    "Summarize how Holm-Bonferroni correction controls family-wise error rate across paired statistical tests.",
)


def _encode_prompt_ids(host: QwenCacheHost, prompt: str, idx_fallback: int = 0) -> torch.Tensor:
    if host.tokenizer is None:
        toks = [1 + (idx_fallback % 7), 2 + (idx_fallback % 5), 3, 4, 5, 6]
        return torch.tensor([toks], device=host.device, dtype=torch.long)
    toks = host.tokenizer.encode(prompt, add_special_tokens=False)
    if not toks:
        toks = [int(host.tokenizer.eos_token_id or 1)]
    return torch.tensor([toks], device=host.device, dtype=torch.long)


def _forward_last_token_logits(
    host: QwenCacheHost,
    embeds: torch.Tensor,
) -> torch.Tensor:
    attn = torch.ones((embeds.shape[0], embeds.shape[1]), device=host.device, dtype=torch.long)
    kwargs: Dict[str, Any] = dict(inputs_embeds=embeds, attention_mask=attn, use_cache=False)
    if getattr(host, "_supports_logits_to_keep", False):
        kwargs["logits_to_keep"] = 1
    logits = host._lm_forward(**kwargs).logits
    return logits[:, -1, :].float()


def _build_fixed_mix_prefix(
    host: QwenCacheHost,
    connector: Stage4MatchedConnector,
    prompt: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mix_prompt = "先导信息：<W1><W2><PHASE>。\n请直接回答或续写下面内容：\n" + prompt
    words = torch.tensor([[2, 3]], device=host.device, dtype=torch.long)
    tid = torch.tensor([0], device=host.device, dtype=torch.long)
    ps = torch.tensor([[0.80, 0.60, 0.50, -0.50, 0.90, 0.10]], device=host.device, dtype=torch.float32)
    meta = torch.tensor([[2.0, 2.0, 4.0, 0.25, 0.10, 2.0, 0.0]], device=host.device, dtype=torch.float32)
    with torch.no_grad():
        st, _, _, _ = connector.forward_tokens(words, 4, tid, ps, meta, ref_summary=None)
    return connector.build_host_prefix_embeds(host, mix_prompt, st)


def compute_frozen_general_refs(
    host: QwenCacheHost,
    connector: Stage4MatchedConnector,
    prompts: Sequence[str],
) -> Dict[str, List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]]:
    """Compute baseline float64 log_probs + raw float logits on hookless frozen host."""
    pure_refs: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = []
    mix_refs: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]] = []
    embed_layer = host.lm.get_input_embeddings()
    with torch.no_grad():
        for idx, pr in enumerate(prompts):
            ids = _encode_prompt_ids(host, pr, idx_fallback=idx)
            pure_emb = embed_layer(ids)
            p_logits = _forward_last_token_logits(host, pure_emb)
            pure_refs.append((F.log_softmax(p_logits, dim=-1, dtype=torch.float64).cpu(), p_logits.cpu(), None))

            mix_emb, mix_mask = _build_fixed_mix_prefix(host, connector, pr)
            m_logits = _forward_last_token_logits(host, mix_emb)
            mix_refs.append((F.log_softmax(m_logits, dim=-1, dtype=torch.float64).cpu(), m_logits.cpu(), mix_mask.cpu()))
    return {"pure": pure_refs, "mix": mix_refs}


def audit_pure_and_mix_kl(
    host: QwenCacheHost,
    connector: Stage4MatchedConnector,
    adapter_or_none: Optional[nn.Module],
    eval_prompts: Sequence[str] = GENERAL_TEXT_EVAL_PROMPTS,
    is_gated_surgical: bool = True,
    frozen_refs: Optional[Dict[str, List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]]] = None,
) -> dict:
    """Measure PureKL (general prose without EM) and MixKL (general prose preceded by active EM prefix)."""
    detached_for_refs = False
    if frozen_refs is None:
        if adapter_or_none is not None and hasattr(adapter_or_none, "remove"):
            adapter_or_none.remove()
            detached_for_refs = True
        try:
            frozen_refs = compute_frozen_general_refs(host, connector, eval_prompts)
        finally:
            if detached_for_refs and hasattr(adapter_or_none, "_attach_hooks"):
                adapter_or_none._attach_hooks(host.lm)

    embed_layer = host.lm.get_input_embeddings()
    pure_kls: List[float] = []
    pure_diffs: List[float] = []
    mix_kls: List[float] = []

    with torch.no_grad():
        for idx, pr in enumerate(eval_prompts):
            ids = _encode_prompt_ids(host, pr, idx_fallback=idx)
            pure_emb = embed_layer(ids)
            ref_lp_pure, ref_raw_pure, _ = frozen_refs["pure"][idx]
            ref_lp_pure_dev = ref_lp_pure.to(host.device)
            ref_p_pure_dev = ref_lp_pure_dev.exp()

            if adapter_or_none is not None and is_gated_surgical:
                with adapter_or_none.activate(g_em=0):
                    test_raw_pure = _forward_last_token_logits(host, pure_emb)
            else:
                test_raw_pure = _forward_last_token_logits(host, pure_emb)
            pure_diffs.append(float((test_raw_pure.cpu() - ref_raw_pure).abs().max().item()))
            test_lp_pure = F.log_softmax(test_raw_pure, dim=-1, dtype=torch.float64)
            kl_p = float(torch.sum(ref_p_pure_dev * (ref_lp_pure_dev - test_lp_pure), dim=-1).clamp_min(0.0).mean().item())
            pure_kls.append(kl_p)

            mix_emb, mix_mask = _build_fixed_mix_prefix(host, connector, pr)
            ref_lp_mix, _, _ = frozen_refs["mix"][idx]
            ref_lp_mix_dev = ref_lp_mix.to(host.device)
            ref_p_mix_dev = ref_lp_mix_dev.exp()
            if adapter_or_none is not None:
                with adapter_or_none.activate(g_em=1, m_acc_mask=mix_mask):
                    test_raw_mix = _forward_last_token_logits(host, mix_emb)
            else:
                test_raw_mix = _forward_last_token_logits(host, mix_emb)
            test_lp_mix = F.log_softmax(test_raw_mix, dim=-1, dtype=torch.float64)
            kl_m = float(torch.sum(ref_p_mix_dev * (ref_lp_mix_dev - test_lp_mix), dim=-1).clamp_min(0.0).mean().item())
            mix_kls.append(kl_m)

    pure_mean = float(np.mean(pure_kls))
    pure_max = float(np.max(pure_kls))
    pure_diff_max = float(np.max(pure_diffs))
    mix_mean = float(np.mean(mix_kls))
    mix_max = float(np.max(mix_kls))
    return {
        "n_eval_prompts": len(eval_prompts),
        "pure_kl_mean": pure_mean,
        "pure_kl_max": pure_max,
        "pure_max_logit_diff": pure_diff_max,
        "mix_kl_mean": mix_mean,
        "mix_kl_max": mix_max,
        "passes_pure_kl_zero": bool(pure_max == 0.0 and pure_diff_max == 0.0),
        "passes_mix_kl_1e3": bool(mix_mean <= 1e-3),
    }


def _weighted_phaseB_minibatch_order(
    tr_groups: List[dict],
    batch_size: int,
    steps: int,
    seed: int,
) -> Tuple[List[Tuple[int, int, int]], List[int]]:
    tr_minibatches: List[Tuple[int, int, int]] = []
    weighted_pool: List[int] = []
    for gi, g in enumerate(tr_groups):
        n = len(g["words"])
        if g["qtype"] == "Q3_repeat_x1":
            w = 6
        elif g["qtype"] == "Q3_after_pilot":
            w = 4
        elif g["qtype"] == "Q2" and g["arm"] in ("valid", "fresh_resample"):
            w = 3
        elif g["qtype"] == "Q1":
            w = 2
        else:
            w = 1
        for st in range(0, n, batch_size):
            mb_id = len(tr_minibatches)
            tr_minibatches.append((gi, st, min(n, st + batch_size)))
            weighted_pool.extend([mb_id] * w)

    rng_order = random.Random(seed + 4099)
    order: List[int] = []
    while len(order) < steps:
        epoch_pool = list(weighted_pool)
        rng_order.shuffle(epoch_pool)
        order.extend(epoch_pool)
    return tr_minibatches, order[:steps]


def _load_or_init_phaseA_pure_connector(
    ckpt_dir: Path,
    encoder,
    hidden_size: int,
    seed: int,
    device: torch.device,
    mock_tiny_for_test: bool,
) -> Tuple[Stage4MatchedConnector, Optional[str]]:
    connector = Stage4MatchedConnector("equivariant_host_pure", encoder.table, hidden_size=hidden_size).to(device)
    ckpt_path = ckpt_dir / f"stage4_phaseA_equivariant_host_pure_s{seed}_step384.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        connector.load_state_dict(ck["state_dict"])
        loaded_from = ckpt_path.name
    elif mock_tiny_for_test:
        loaded_from = None
    else:
        raise FileNotFoundError(f"Missing Phase-A 384-step pure connector checkpoint: {ckpt_path}")
    connector.requires_grad_(False)
    connector.eval()
    return connector, loaded_from


def _load_or_build_kfac(
    host: QwenCacheHost,
    kfac_path: Path,
    fa_layers: Sequence[int],
    dn_layers: Sequence[int],
    mock_tiny_for_test: bool,
) -> Dict[int, torch.Tensor]:
    if kfac_path.exists():
        return torch.load(kfac_path, map_location="cpu", weights_only=False)
    if not mock_tiny_for_test:
        raise FileNotFoundError(f"Missing K-FAC projector file: {kfac_path}")
    k_c = 256 if host.hidden_size >= 1024 else max(4, host.hidden_size // 8)
    return collect_kfac_complex_projectors(host, sorted(set(list(fa_layers) + list(dn_layers))), k_c=k_c)


def _evaluate_adapter_dev(
    host: QwenCacheHost,
    connector: Stage4MatchedConnector,
    kernel_mod: PhysicalKernelS17,
    dv_groups: List[dict],
    adapter: nn.Module,
    scorer_mode: str,
) -> dict:
    """Evaluate a frozen Phase-A connector with a hook-attached Phase-B adapter across dev groups."""
    total_n = 0
    loss_sum = 0.0
    hits = 0
    inf_n = 0
    inf_loss_sum = 0.0
    inf_hits = 0
    phys_inf_loss = 0.0
    phys_inf_hits = 0
    q1_n = q1_hits = 0
    q1_loss = 0.0
    q2v_n = q2v_hits = 0
    q2v_loss = 0.0
    q2f_n = q2f_hits = 0
    q2f_loss = 0.0
    q3c_n = q3c_hits = 0
    q3c_loss = 0.0
    q3c_phys_loss = 0.0
    q3i_n = q3i_hits = 0
    q3a_n = q3a_hits = 0
    q3a_loss = 0.0

    del kernel_mod
    connector.eval()
    adapter.eval()
    with torch.no_grad():
        for g in dv_groups:
            n = min(16, len(g["words"]))
            is_inf = g["qtype"] != "Q3_independent_x3" and not (
                g["qtype"] == "Q2" and g["arm"] in ("none", "independent")
            )
            is_q3_cfo = bool(g["qtype"] == "Q3_repeat_x1" and g["source"] == "cfo")
            words = torch.from_numpy(g["words"][:n]).to(host.device)
            ps = torch.from_numpy(g["phase_slots"][:n]).to(host.device)
            meta = torch.from_numpy(g["meta_scalars"][:n]).to(host.device)
            rs = torch.from_numpy(g["ref_summary"][:n]).to(host.device)
            dd = torch.from_numpy(g["discrete_d"][:n]).to(host.device)
            lbl = torch.from_numpy(g["labels"][:n]).to(host.device)
            p_phys = torch.from_numpy(g["phys_probs"][:n]).to(host.device).clamp_min(1e-12)

            lp, _, _, _ = connector.score_batch(
                host,
                g["prompt"],
                g["choices"],
                g["qtype"],
                words,
                4,
                ps,
                meta,
                ref_summary=rs,
                scorer_mode=scorer_mode,
                surgical_adapter=adapter,
                discrete_d=dd,
            )
            ls = float(F.nll_loss(lp, lbl, reduction="sum").item())
            hs = int((lp.argmax(dim=-1) == lbl).sum().item())
            pls = float(F.nll_loss(torch.log(p_phys), lbl, reduction="sum").item())
            phs = int((p_phys.argmax(dim=-1) == lbl).sum().item())

            total_n += n
            loss_sum += ls
            hits += hs
            if is_inf:
                inf_n += n
                inf_loss_sum += ls
                inf_hits += hs
                phys_inf_loss += pls
                phys_inf_hits += phs
            if g["qtype"] == "Q1":
                q1_n += n
                q1_hits += hs
                q1_loss += ls
            elif g["qtype"] == "Q2" and g["arm"] == "valid":
                q2v_n += n
                q2v_hits += hs
                q2v_loss += ls
            elif g["qtype"] == "Q2" and g["arm"] == "fresh_resample":
                q2f_n += n
                q2f_hits += hs
                q2f_loss += ls
            elif is_q3_cfo:
                q3c_n += n
                q3c_hits += hs
                q3c_loss += ls
                q3c_phys_loss += pls
            elif g["qtype"] == "Q3_independent_x3":
                q3i_n += n
                q3i_hits += hs
            elif g["qtype"] == "Q3_after_pilot":
                q3a_n += n
                q3a_hits += hs
                q3a_loss += ls

    return {
        "n_all": total_n,
        "n_informative": inf_n,
        "dev_nll_all": loss_sum / max(1, total_n),
        "dev_acc_all": hits / max(1, total_n),
        "dev_nll_informative": inf_loss_sum / max(1, inf_n),
        "dev_acc_informative": inf_hits / max(1, inf_n),
        "direct_physical_dev_nll_informative": phys_inf_loss / max(1, inf_n),
        "direct_physical_dev_acc_informative": phys_inf_hits / max(1, inf_n),
        "q1_acc": q1_hits / max(1, q1_n),
        "q1_nll": q1_loss / max(1, q1_n),
        "q2_valid_acc": q2v_hits / max(1, q2v_n),
        "q2_valid_nll": q2v_loss / max(1, q2v_n),
        "q2_fresh_acc": q2f_hits / max(1, q2f_n),
        "q2_fresh_nll": q2f_loss / max(1, q2f_n),
        "q2_fresh_minus_valid_acc": (q2f_hits / max(1, q2f_n)) - (q2v_hits / max(1, q2v_n)),
        "q3_repeat_cfo_acc": q3c_hits / max(1, q3c_n),
        "q3_repeat_cfo_nll": q3c_loss / max(1, q3c_n),
        "q3_repeat_cfo_phys_nll": q3c_phys_loss / max(1, q3c_n),
        "q3_indep_acc": q3i_hits / max(1, q3i_n),
        "q3_after_pilot_acc": q3a_hits / max(1, q3a_n),
        "q3_after_pilot_nll": q3a_loss / max(1, q3a_n),
    }


def _make_lr_scheduler(opt: torch.optim.Optimizer, steps: int):
    warmup = max(1, min(10, steps // 4)) if steps > 2 else 1

    def _lr_lam(idx: int) -> float:
        if idx < warmup:
            return float(idx + 1) / float(warmup)
        prog = min(1.0, float(idx - warmup) / float(max(1, steps - warmup)))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lam)


def run_phaseB_surgical_and_lemmo(
    repo_root: Path,
    out_dir: Path,
    ckpt_dir: Path,
    kfac_path: Path,
    seed: int = 202611,
    steps: int = 96,
    batch_size: int = 32,
    micro_batch_size: int = 16,
    lr: float = 3e-4,
    train_blocks: int = 8,
    dev_blocks: int = 2,
    scorer_mode: str = "kv_cached",
    model_id: str = MODEL_ID_DEFAULT,
    revision: str = PINNED_REVISION_DEFAULT,
    mock_tiny_for_test: bool = False,
    host: Optional[QwenCacheHost] = None,
) -> dict:
    """Train Mode-I Surgical-96 and matched LEMMO Tier-P1 on identical frozen connector & minibatches."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if host is None:
        host = QwenCacheHost(model_id=model_id, revision=revision, mock_tiny_for_test=mock_tiny_for_test)
    if not mock_tiny_for_test:
        assert int(getattr(host.text_config, "vocab_size", 0)) == EXPECTED_4B_VOCAB
        assert host.host_sha256 == EXPECTED_4B_HOST_SHA256

    encoder, _ = load_s17_encoder(repo_root)
    kernel_mod = PhysicalKernelS17(encoder.table).to(host.device)
    connector, ckpt_used = _load_or_init_phaseA_pure_connector(
        ckpt_dir, encoder, host.hidden_size, seed, host.device, mock_tiny_for_test
    )
    connector_total_params = sum(p.numel() for p in connector.parameters())

    layers = _get_decoder_layers(host.lm)
    n_l = len(layers)
    fa_l = [i for i in QWEN4B_FA_LAYERS if i < n_l] or ([n_l - 1] if n_l >= 1 else [])
    dn_l = [i for i in QWEN4B_DN_LAYERS if i < n_l]
    stop_l = min(QWEN4B_STOP_LAYER, max(0, n_l - 2))

    proj_map = _load_or_build_kfac(host, kfac_path, fa_l, dn_l, mock_tiny_for_test)
    b_align = calibrate_b_align_from_host(host)
    tr_groups, dv_groups = build_stage4_pilot_groups(encoder, kernel_mod, train_blocks=train_blocks, dev_blocks=dev_blocks)
    tr_minibatches, order = _weighted_phaseB_minibatch_order(tr_groups, batch_size=batch_size, steps=steps, seed=seed)

    # Precompute hookless frozen general-text references BEFORE attaching either adapter
    eval_frozen_refs = compute_frozen_general_refs(host, connector, GENERAL_TEXT_EVAL_PROMPTS)
    train_mix_refs = compute_frozen_general_refs(host, connector, GENERAL_TEXT_TRAIN_PROMPTS)["mix"]
    frozen_dev_metrics = _evaluate_dev_groups(
        host, connector, kernel_mod, dv_groups, max_items_per_group=16, scorer_mode=scorer_mode, collect_latents=False
    )

    # --- Part 1: Mode-I Surgical-96 ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    surg = SurgicalHostAdapter(
        host.lm,
        fa_layers=fa_l,
        dn_layers=dn_l,
        stop_layer=stop_l,
        mode="mode1_stopgrad_l10",
        u_c_by_layer=proj_map,
        b_align=b_align,
    ).to(host.device)

    surg_summary: Dict[str, Any]
    try:
        surg_params = surg.param_count()
        cum_surg = connector_total_params + surg_params
        assert cum_surg <= TIER_P1_MAX_PARAMS, f"Surgical cumulative params {cum_surg} > {TIER_P1_MAX_PARAMS}"

        opt = torch.optim.AdamW(surg.parameters(), lr=lr, weight_decay=1e-4)
        sched = _make_lr_scheduler(opt, steps)
        t0 = time.perf_counter()
        if host.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(host.device)

        surg.train()
        last_losses: List[float] = []
        for step_i, mb_idx in enumerate(order):
            gi, st, ed = tr_minibatches[mb_idx]
            g = tr_groups[gi]
            mb_total = ed - st
            opt.zero_grad(set_to_none=True)
            step_loss_val = 0.0
            use_cf = (
                (step_i % 2 == 0)
                and g["qtype"] in ("Q1", "Q3_repeat_x1", "Q3_after_pilot")
                and abs(float(g["omega"])) > 0.0
            )
            for m_st in range(st, ed, micro_batch_size):
                m_ed = min(ed, m_st + micro_batch_size)
                words = torch.from_numpy(g["words"][m_st:m_ed]).to(host.device)
                ps = torch.from_numpy(g["phase_slots"][m_st:m_ed]).to(host.device)
                meta = torch.from_numpy(g["meta_scalars"][m_st:m_ed]).to(host.device)
                rs = torch.from_numpy(g["ref_summary"][m_st:m_ed]).to(host.device)
                dd = torch.from_numpy(g["discrete_d"][m_st:m_ed]).to(host.device)
                target_p = torch.from_numpy(g["phys_probs"][m_st:m_ed]).to(host.device)

                lp, _, _, _ = connector.score_batch(
                    host,
                    g["prompt"],
                    g["choices"],
                    g["qtype"],
                    words,
                    4,
                    ps,
                    meta,
                    ref_summary=rs,
                    scorer_mode=scorer_mode,
                    surgical_adapter=surg,
                    discrete_d=dd,
                )
                cf_lp = None
                if use_cf:
                    swap_ps = torch.from_numpy(g["swap_slots"][m_st:m_ed]).to(host.device)
                    cf_lp, _, _, _ = connector.score_batch(
                        host,
                        g["prompt"],
                        g["choices"],
                        g["qtype"],
                        words,
                        4,
                        swap_ps,
                        meta,
                        ref_summary=rs,
                        scorer_mode=scorer_mode,
                        surgical_adapter=surg,
                        discrete_d=dd,
                    )
                losses = rao_blackwell_bayes_kl_cf_loss(
                    lp, target_p, cf_swapped_logits=cf_lp, lambda_cf=0.15, lambda_norm=0.02, margin=0.25
                )
                scaled_loss = losses["total"] * (float(m_ed - m_st) / float(mb_total))
                scaled_loss.backward()
                step_loss_val += float(scaled_loss.item())
                del lp, cf_lp, losses, scaled_loss

            if step_i % 4 == 0:
                lambda_mix = 0.5 + 4.5 * float(step_i) / float(max(1, steps - 1))
                p_idx = (step_i // 4) % len(GENERAL_TEXT_TRAIN_PROMPTS)
                mix_emb, mix_mask = _build_fixed_mix_prefix(host, connector, GENERAL_TEXT_TRAIN_PROMPTS[p_idx])
                ref_lp_mix, _, _ = train_mix_refs[p_idx]
                ref_lp_mix_dev = ref_lp_mix.to(host.device)
                ref_p_mix_dev = ref_lp_mix_dev.exp()
                with surg.activate(g_em=1, m_acc_mask=mix_mask):
                    mix_logits = _forward_last_token_logits(host, mix_emb)
                q_lp_mix = F.log_softmax(mix_logits, dim=-1, dtype=torch.float64)
                mix_kl = torch.sum(ref_p_mix_dev * (ref_lp_mix_dev - q_lp_mix), dim=-1).mean()
                (lambda_mix * mix_kl.float()).backward()
                step_loss_val += float((lambda_mix * mix_kl).item())
                del mix_logits, q_lp_mix, mix_kl, mix_emb, mix_mask

            if step_i == 0:
                assert all(p.grad is None for p in connector.parameters()), "Mode-I must block connector grads!"
                assert all(p.grad is None for p in host.lm.parameters()), "Host weights must remain frozen!"

            torch.nn.utils.clip_grad_norm_(surg.parameters(), 1.0)
            opt.step()
            sched.step()
            last_losses.append(step_loss_val)
            if (step_i + 1) % 16 == 0 or (step_i + 1) == steps:
                print(
                    f"[Surgical-{steps} step {step_i + 1}/{steps}] loss={step_loss_val:.4f} elapsed={time.perf_counter() - t0:.1f}s",
                    flush=True,
                )

        surg_wall = round(time.perf_counter() - t0, 2)
        surg_peak_mb = round(torch.cuda.max_memory_allocated(host.device) / (1024.0 ** 2), 1) if host.device.type == "cuda" else 0.0
        surg_dev = _evaluate_adapter_dev(host, connector, kernel_mod, dv_groups, surg, scorer_mode=scorer_mode)
        surg_audit = audit_pure_and_mix_kl(
            host, connector, surg, GENERAL_TEXT_EVAL_PROMPTS, is_gated_surgical=True, frozen_refs=eval_frozen_refs
        )
        surg_sha_after = _compute_host_fingerprint(host.lm)
        assert surg_sha_after == host.host_sha256, "Host SHA mutated during Surgical-96!"
        surg_ckpt = ckpt_dir / f"stage4_phaseB_surgical96_s{seed}.pt"
        torch.save(
            {
                "arm": "mode1_surgical_96",
                "seed": seed,
                "steps": steps,
                "fa_layers": list(fa_l),
                "dn_layers": list(dn_l),
                "stop_layer": stop_l,
                "b_align": b_align.cpu(),
                "state_dict": {k: v.cpu() for k, v in surg.state_dict().items()},
            },
            surg_ckpt,
        )
        surg_summary = {
            "arm": "mode1_surgical_96",
            "steps": steps,
            "train_wall_s": surg_wall,
            "peak_vram_mb": surg_peak_mb,
            "final_train_loss": float(np.mean(last_losses[-min(8, len(last_losses)):])),
            "connector_params": connector_total_params,
            "adapter_params": surg_params,
            "cumulative_params": cum_surg,
            "tier_p1_cap": TIER_P1_MAX_PARAMS,
            "under_tier_p1_cap": bool(cum_surg <= TIER_P1_MAX_PARAMS),
            "dev_metrics": surg_dev,
            "audit": surg_audit,
            "host_sha256": surg_sha_after,
            "checkpoint": surg_ckpt.name,
            "checkpoint_sha256": hashlib.sha256(surg_ckpt.read_bytes()).hexdigest(),
        }
        del opt, sched
    finally:
        surg.remove()
        del surg
        gc.collect()
        if host.device.type == "cuda":
            torch.cuda.empty_cache()

    # --- Part 2: Matched LEMMO Tier-P1 (Unconstrained LoRA-96) ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    lemmo = UnconstrainedLoRAAdapter(host.lm, fa_layers=fa_l, dn_layers=dn_l, stop_layer=stop_l, r_lora=8).to(host.device)

    lemmo_summary: Dict[str, Any]
    try:
        lemmo_params = lemmo.param_count()
        cum_lemmo = connector_total_params + lemmo_params
        assert cum_lemmo <= TIER_P1_MAX_PARAMS, f"LEMMO cumulative params {cum_lemmo} > {TIER_P1_MAX_PARAMS}"

        opt_l = torch.optim.AdamW(lemmo.parameters(), lr=lr, weight_decay=1e-4)
        sched_l = _make_lr_scheduler(opt_l, steps)
        t0_l = time.perf_counter()
        if host.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(host.device)

        lemmo.train()
        last_losses_l: List[float] = []
        for step_i, mb_idx in enumerate(order):
            gi, st, ed = tr_minibatches[mb_idx]
            g = tr_groups[gi]
            mb_total = ed - st
            opt_l.zero_grad(set_to_none=True)
            step_loss_val = 0.0
            for m_st in range(st, ed, micro_batch_size):
                m_ed = min(ed, m_st + micro_batch_size)
                words = torch.from_numpy(g["words"][m_st:m_ed]).to(host.device)
                ps = torch.from_numpy(g["phase_slots"][m_st:m_ed]).to(host.device)
                meta = torch.from_numpy(g["meta_scalars"][m_st:m_ed]).to(host.device)
                rs = torch.from_numpy(g["ref_summary"][m_st:m_ed]).to(host.device)
                dd = torch.from_numpy(g["discrete_d"][m_st:m_ed]).to(host.device)
                target_p = torch.from_numpy(g["phys_probs"][m_st:m_ed]).to(host.device)

                lp, _, _, _ = connector.score_batch(
                    host,
                    g["prompt"],
                    g["choices"],
                    g["qtype"],
                    words,
                    4,
                    ps,
                    meta,
                    ref_summary=rs,
                    scorer_mode=scorer_mode,
                    surgical_adapter=lemmo,
                    discrete_d=dd,
                )
                losses = rao_blackwell_bayes_kl_cf_loss(
                    lp, target_p, cf_swapped_logits=None, lambda_cf=0.0, lambda_norm=0.0
                )
                scaled_loss = losses["total"] * (float(m_ed - m_st) / float(mb_total))
                scaled_loss.backward()
                step_loss_val += float(scaled_loss.item())
                del lp, losses, scaled_loss
            torch.nn.utils.clip_grad_norm_(lemmo.parameters(), 1.0)
            opt_l.step()
            sched_l.step()
            last_losses_l.append(step_loss_val)
            if (step_i + 1) % 16 == 0 or (step_i + 1) == steps:
                print(
                    f"[LEMMO-P1-{steps} step {step_i + 1}/{steps}] loss={step_loss_val:.4f} elapsed={time.perf_counter() - t0_l:.1f}s",
                    flush=True,
                )

        lemmo_wall = round(time.perf_counter() - t0_l, 2)
        lemmo_peak_mb = round(torch.cuda.max_memory_allocated(host.device) / (1024.0 ** 2), 1) if host.device.type == "cuda" else 0.0
        lemmo_dev = _evaluate_adapter_dev(host, connector, kernel_mod, dv_groups, lemmo, scorer_mode=scorer_mode)
        lemmo_audit = audit_pure_and_mix_kl(
            host, connector, lemmo, GENERAL_TEXT_EVAL_PROMPTS, is_gated_surgical=False, frozen_refs=eval_frozen_refs
        )
        lemmo_sha_after = _compute_host_fingerprint(host.lm)
        assert lemmo_sha_after == host.host_sha256, "Host SHA mutated during LEMMO Tier-P1!"
        lemmo_ckpt = ckpt_dir / f"stage4_phaseB_lemmo_tier_p1_s{seed}.pt"
        torch.save(
            {
                "arm": "lemmo_tier_p1",
                "seed": seed,
                "steps": steps,
                "fa_layers": list(fa_l),
                "dn_layers": list(dn_l),
                "stop_layer": stop_l,
                "state_dict": {k: v.cpu() for k, v in lemmo.state_dict().items()},
            },
            lemmo_ckpt,
        )
        lemmo_summary = {
            "arm": "lemmo_tier_p1",
            "steps": steps,
            "train_wall_s": lemmo_wall,
            "peak_vram_mb": lemmo_peak_mb,
            "final_train_loss": float(np.mean(last_losses_l[-min(8, len(last_losses_l)):])),
            "connector_params": connector_total_params,
            "adapter_params": lemmo_params,
            "cumulative_params": cum_lemmo,
            "tier_p1_cap": TIER_P1_MAX_PARAMS,
            "under_tier_p1_cap": bool(cum_lemmo <= TIER_P1_MAX_PARAMS),
            "dev_metrics": lemmo_dev,
            "audit": lemmo_audit,
            "host_sha256": lemmo_sha_after,
            "checkpoint": lemmo_ckpt.name,
            "checkpoint_sha256": hashlib.sha256(lemmo_ckpt.read_bytes()).hexdigest(),
        }
        del opt_l, sched_l
    finally:
        lemmo.remove()
        del lemmo
        gc.collect()
        if host.device.type == "cuda":
            torch.cuda.empty_cache()

    phaseB_rep = {
        "seed": seed,
        "steps": steps,
        "model_id": host.model_id,
        "host_sha256": host.host_sha256,
        "base_phaseA_checkpoint": ckpt_used,
        "frozen_pure_dev_metrics": {k: v for k, v in frozen_dev_metrics.items() if not k.startswith("_")},
        "surgical_96": surg_summary,
        "lemmo_tier_p1": lemmo_summary,
    }
    (out_dir / "phaseB_surgical_vs_lemmo.json").write_text(
        json.dumps(phaseB_rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return phaseB_rep


def _cluster_paired_stats(
    vals_a: np.ndarray,
    vals_b: np.ndarray,
    cluster_ids: np.ndarray,
    higher_is_better_a: bool = True,
    n_boot: int = 2000,
    seed: int = 202611,
) -> dict:
    """Compute paired block-cluster stats (diff = a - b, 95% cluster bootstrap CI, paired t-test p)."""
    uniq = np.unique(cluster_ids)
    cl_diffs = np.array([float(np.mean(vals_a[cluster_ids == c] - vals_b[cluster_ids == c])) for c in uniq], dtype=np.float64)
    n_c = len(cl_diffs)
    mean_diff = float(np.mean(cl_diffs))
    if n_c >= 2 and float(np.std(cl_diffs, ddof=1)) > 0.0:
        t_res = scipy_stats.ttest_rel(
            np.array([float(np.mean(vals_a[cluster_ids == c])) for c in uniq]),
            np.array([float(np.mean(vals_b[cluster_ids == c])) for c in uniq]),
        )
        p_two = float(t_res.pvalue)
    else:
        p_two = 0.0 if abs(mean_diff) > 1e-12 else 1.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_c, size=(n_boot, n_c))
    boot_means = cl_diffs[idx].mean(axis=1)
    ci_lo, ci_hi = float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))
    # One-sided improvement direction check
    improved = (mean_diff > 0.0) if higher_is_better_a else (mean_diff < 0.0)
    p_one = (0.5 * p_two) if improved else (1.0 - 0.5 * p_two)
    return {
        "n_clusters": int(n_c),
        "mean_diff": mean_diff,
        "ci95_low": ci_lo,
        "ci95_high": ci_hi,
        "p_two_sided": p_two,
        "p_one_sided_improved": float(p_one),
    }


def _holm_bonferroni(p_map: Dict[str, float]) -> Dict[str, float]:
    ordered = sorted(p_map.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adj: Dict[str, float] = {}
    running_max = 0.0
    for i, (k, p) in enumerate(ordered):
        val = min(1.0, (m - i) * float(p))
        running_max = max(running_max, val)
        adj[k] = running_max
    return adj


def _sym_kl_arrays(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    p_s = np.clip(p, 1e-12, 1.0)
    q_s = np.clip(q, 1e-12, 1.0)
    return 0.5 * np.sum((p_s - q_s) * (np.log(p_s) - np.log(q_s)), axis=-1)


def _build_blind_suites(
    encoder,
    g1_blocks: int,
    other_blocks: int,
) -> Dict[str, List[dict]]:
    suites: Dict[str, List[dict]] = {
        "G1_main_extrap_both": [],
        "G2_interp_both": [],
        "G3_extrap_angle": [],
        "G4_extrap_freq": [],
        "Q2_C4_blind": [],
        "G1_counterfactuals": [],
    }

    for cid, spec in BLIND_CELLS.items():
        n_blk = int(g1_blocks) if cid == "G1_main_extrap_both" else int(other_blocks)
        omegas = (-spec.abs_omega, spec.abs_omega)
        b_map, _ = generate_paired_cfo_blocks(
            spec.namespace, 0, n_blk, spec.h, spec.ell, spec.t_p, omegas, spec.snr_db, spec.sigma_phi, split_prefix=cid
        )
        rots = ROTATIONS if cid == "G1_main_extrap_both" else ((0, 0),)
        for oi, w in enumerate(omegas):
            batch = b_map[float(w)]
            cond = batch.condition
            n_m = len(batch.x1)
            blk_ids = np.arange(n_m, dtype=np.int64) // 64
            p_vec = public_phase_slots(cond.omega, cond.h, cond.ell, cond.t_p, zero_arm=False)
            p_swap = public_phase_slots(-cond.omega, cond.h, cond.ell, cond.t_p, zero_arm=False)
            meta_vec = np.array([cond.h, cond.ell, cond.t_p, cond.omega, cond.sigma_phi, cond.snr_db, 0.0], dtype=np.float32)
            phase_mat = np.repeat(p_vec[None, :], n_m, axis=0)
            swap_mat = np.repeat(p_swap[None, :], n_m, axis=0)
            meta_mat = np.repeat(meta_vec[None, :], n_m, axis=0)
            disc_vec = np.full(n_m, float(cond.d), dtype=np.float32)
            empty_ref = np.zeros((n_m, 29), dtype=np.float32)
            empty_ref[:, 0:8] = 0.25
            for rot in rots:
                s1, _ = rot
                v1, v2 = batch.rotated_view(*rot)
                z1_np = np.asarray(encoder.encode(v1), dtype=np.int64)
                z2_np = np.asarray(encoder.encode(v2), dtype=np.int64)
                t1, t2 = encoder.table[z1_np], encoder.table[z2_np]
                words = np.stack([z1_np, z2_np], axis=-1)
                prompt, choices = build_stage3d_prompt("Q3", cond, protocol="repeat_x1")
                res = s3_analytic.readout_from_table(t1, t2, (), cond, protocol="repeat_x1")
                labels = sector_of_complex(matched_sum(batch.y3_repeat * ROOTS[s1 % 4])).astype(np.int64)
                suites[cid].append(
                    {
                        "cell": cid,
                        "variant": "factual",
                        "arm": "pre_ref",
                        "omega": float(w),
                        "rot": rot,
                        "cluster_ids": blk_ids,
                        "qtype": "Q3_repeat_x1",
                        "prompt": prompt,
                        "choices": choices,
                        "words": words,
                        "phase_slots": phase_mat,
                        "meta_scalars": meta_mat,
                        "discrete_d": disc_vec,
                        "ref_summary": empty_ref,
                        "labels": labels,
                        "phys_probs": np.asarray(res["q3"], dtype=np.float32),
                    }
                )
                if cid == "G1_main_extrap_both" and rot == (0, 0):
                    cf_lim = min(n_m, int(other_blocks) * 64)
                    cond_flip = condition_spec_from_params(-cond.omega, cond.h, cond.ell, cond.t_p, cond.snr_db, cond.sigma_phi, tag=cond.name)
                    p_flip_text, _ = build_stage3d_prompt("Q3", cond_flip, protocol="repeat_x1")
                    for vname, pr_v, ps_v in (
                        ("factual", prompt, phase_mat[:cf_lim]),
                        ("slot_flipped", prompt, swap_mat[:cf_lim]),
                        ("text_flipped", p_flip_text, phase_mat[:cf_lim]),
                    ):
                        suites["G1_counterfactuals"].append(
                            {
                                "cell": "G1_cf",
                                "variant": vname,
                                "pair_group": oi,
                                "cluster_ids": blk_ids[:cf_lim],
                                "qtype": "Q3_repeat_x1",
                                "prompt": pr_v,
                                "choices": choices,
                                "words": words[:cf_lim],
                                "phase_slots": ps_v,
                                "meta_scalars": meta_mat[:cf_lim],
                                "discrete_d": disc_vec[:cf_lim],
                                "ref_summary": empty_ref[:cf_lim],
                                "labels": labels[:cf_lim],
                                "phys_probs": np.asarray(res["q3"][:cf_lim], dtype=np.float32),
                            }
                        )

    # Q2_C4_blind across all 7 arms
    q2_map, _ = generate_paired_cfo_blocks(NS_BLIND_Q2_C4, 0, int(other_blocks), 2, 2, 4.0, (0.35,), 2.0, 0.10, split_prefix="q2_c4")
    batch_q2 = q2_map[0.35]
    cond_q2 = batch_q2.condition
    n_q2 = len(batch_q2.x1)
    blk_q2 = np.arange(n_q2, dtype=np.int64) // 64
    v1_q2, v2_q2 = batch_q2.rotated_view(0, 0)
    z1_q2 = np.asarray(encoder.encode(v1_q2), dtype=np.int64)
    z2_q2 = np.asarray(encoder.encode(v2_q2), dtype=np.int64)
    t1_q2, t2_q2 = encoder.table[z1_q2], encoder.table[z2_q2]
    words_q2 = np.stack([z1_q2, z2_q2], axis=-1)
    p_vec_q2 = public_phase_slots(cond_q2.omega, cond_q2.h, cond_q2.ell, cond_q2.t_p, zero_arm=False)
    phase_q2 = np.repeat(p_vec_q2[None, :], n_q2, axis=0)
    meta_q2 = np.repeat(
        np.array([[cond_q2.h, cond_q2.ell, cond_q2.t_p, cond_q2.omega, cond_q2.sigma_phi, cond_q2.snr_db, 0.0]], dtype=np.float32),
        n_q2,
        axis=0,
    )
    disc_q2 = np.full(n_q2, float(cond_q2.d), dtype=np.float32)
    labels_q2 = batch_q2.x1.astype(np.int64)
    for arm in Q2_BLIND_ARMS:
        refs, dedup_flag = build_q2_arm_references(batch_q2, arm, coord_rot=0)
        p_q2, ch_q2 = build_stage3d_prompt("Q2", cond_q2, references=refs, deduplicate=dedup_flag)
        rs_q2, _ = extract_canonical_ref_summary(
            refs, n_q2, torch.device("cpu"), omega=cond_q2.omega, sigma_phi=cond_q2.sigma_phi, drift_type=cond_q2.drift_type, discrete_d=cond_q2.d
        )
        res_q2 = s3_analytic.readout_from_table(t1_q2, t2_q2, refs, cond_q2, protocol="repeat_x1")
        suites["Q2_C4_blind"].append(
            {
                "cell": "Q2_C4_blind",
                "arm": arm,
                "variant": arm,
                "cluster_ids": blk_q2,
                "qtype": "Q2",
                "prompt": p_q2,
                "choices": ch_q2,
                "words": words_q2,
                "phase_slots": phase_q2,
                "meta_scalars": meta_q2,
                "discrete_d": disc_q2,
                "ref_summary": rs_q2.numpy(),
                "labels": labels_q2,
                "phys_probs": np.asarray(res_q2["q2"], dtype=np.float32),
            }
        )
    return suites


def _reconstruct_probs_from_centered_logits(
    host: QwenCacheHost,
    connector: Stage4MatchedConnector,
    g: dict,
    all_logits_cpu: torch.Tensor,
    recon_bs: int = 4096,
) -> np.ndarray:
    """Reconstruct exact final probabilities from cached `centered_logits` `(N, 4)` CPU tensor."""
    n = len(g["words"])
    if n == 0:
        return np.empty((0, 4), dtype=np.float64)
    if not connector.quotient:
        return F.log_softmax(all_logits_cpu, dim=-1).exp().numpy().astype(np.float64)
    if connector.residual:
        chunks_out: List[np.ndarray] = []
        for st in range(0, n, recon_bs):
            ed = min(n, st + recon_bs)
            codes_c = torch.from_numpy(g["words"][st:ed]).to(device=host.device, dtype=torch.int64)
            meta_c = torch.from_numpy(g["meta_scalars"][st:ed]).to(host.device)
            rs_c = torch.from_numpy(g["ref_summary"][st:ed]).to(host.device)
            dd_c = torch.from_numpy(g["discrete_d"][st:ed]).to(host.device)
            c_log = all_logits_cpu[st:ed].to(host.device)
            p_c = connector.kernel_module.direct_physical_probs(
                codes_c,
                meta_c,
                g["qtype"],
                valid_w1=torch.ones(ed - st, dtype=torch.bool, device=host.device),
                m3_q2=rs_c[:, 0:4],
                m3_q3=rs_c[:, 4:8],
                residual_logits=c_log,
                gamma=connector.residual_gamma,
                discrete_d=dd_c,
            )
            chunks_out.append(p_c.detach().cpu().numpy().astype(np.float64))
        return np.concatenate(chunks_out, axis=0)
    canon_probs = F.softmax(all_logits_cpu, dim=-1)
    words_long = torch.from_numpy(g["words"]).to(dtype=torch.int64)
    return lift_scores(canon_probs, words_long, g["qtype"]).numpy().astype(np.float64)


def _predict_model_on_suites(
    host: QwenCacheHost,
    connector: Optional[Stage4MatchedConnector],
    suites: Dict[str, List[dict]],
    adapter: Optional[nn.Module],
    eval_bs: int,
    scorer_mode: str,
) -> Dict[str, List[np.ndarray]]:
    preds: Dict[str, List[np.ndarray]] = {}
    if connector is None:
        for sname, grps in suites.items():
            preds[sname] = [g["phys_probs"].astype(np.float64) for g in grps]
        return preds

    connector.eval()
    if adapter is not None:
        adapter.eval()
    with torch.no_grad():
        for sname, grps in suites.items():
            s_preds: List[np.ndarray] = []
            for g in grps:
                n = len(g["words"])
                if n == 0:
                    s_preds.append(np.empty((0, 4), dtype=np.float64))
                    continue
                code_key = (g["words"] % 4) if connector.quotient else g["words"]
                key_mat = np.ascontiguousarray(
                    np.concatenate(
                        [
                            code_key.astype(np.float32),
                            g["phase_slots"],
                            g["meta_scalars"],
                            g["ref_summary"],
                        ],
                        axis=-1,
                    )
                )
                _, first_idx, inv_idx = np.unique(
                    key_mat, axis=0, return_index=True, return_inverse=True
                )
                uniq_logits_chunks: List[torch.Tensor] = []
                n_u = len(first_idx)
                for st in range(0, n_u, eval_bs):
                    idx_u = first_idx[st : min(n_u, st + eval_bs)]
                    words = torch.from_numpy(g["words"][idx_u]).to(host.device)
                    ps = torch.from_numpy(g["phase_slots"][idx_u]).to(host.device)
                    meta = torch.from_numpy(g["meta_scalars"][idx_u]).to(host.device)
                    rs = torch.from_numpy(g["ref_summary"][idx_u]).to(host.device)
                    dd = torch.from_numpy(g["discrete_d"][idx_u]).to(host.device)
                    _, centered_logits, _, _ = connector.score_batch(
                        host,
                        g["prompt"],
                        g["choices"],
                        g["qtype"],
                        words,
                        4,
                        ps,
                        meta,
                        ref_summary=rs,
                        scorer_mode=scorer_mode,
                        surgical_adapter=adapter,
                        discrete_d=dd,
                    )
                    uniq_logits_chunks.append(centered_logits.detach().to(device="cpu", dtype=torch.float32))
                uniq_logits = torch.cat(uniq_logits_chunks, dim=0)
                inv_t = torch.from_numpy(inv_idx).to(dtype=torch.long)
                all_logits = uniq_logits[inv_t]
                s_preds.append(_reconstruct_probs_from_centered_logits(host, connector, g, all_logits))
            preds[sname] = s_preds
    return preds


def _summarize_model_blind(suites: Dict[str, List[dict]], model_preds: Dict[str, List[np.ndarray]]) -> dict:
    summary: Dict[str, Any] = {}
    for cid in ("G1_main_extrap_both", "G2_interp_both", "G3_extrap_angle", "G4_extrap_freq"):
        grps = suites[cid]
        ps = np.concatenate(model_preds[cid], axis=0)
        ys = np.concatenate([g["labels"] for g in grps], axis=0)
        nlls = -np.log(np.clip(ps[np.arange(len(ys)), ys], 1e-12, 1.0))
        hits = (ps.argmax(axis=-1) == ys).astype(np.float64)
        summary[cid] = {
            "n": int(len(ys)),
            "acc": float(np.mean(hits)),
            "nll": float(np.mean(nlls)),
            "_hits": hits,
            "_nlls": nlls,
            "_clusters": np.concatenate([g["cluster_ids"] for g in grps], axis=0),
        }

    q2_arms: Dict[str, Any] = {}
    q2_by_arm_probs: Dict[str, np.ndarray] = {}
    q2_by_arm_hits: Dict[str, np.ndarray] = {}
    q2_clusters: Optional[np.ndarray] = None
    for g, p_arm in zip(suites["Q2_C4_blind"], model_preds["Q2_C4_blind"]):
        arm = g["arm"]
        ys = g["labels"]
        nlls = -np.log(np.clip(p_arm[np.arange(len(ys)), ys], 1e-12, 1.0))
        hits = (p_arm.argmax(axis=-1) == ys).astype(np.float64)
        q2_by_arm_probs[arm] = p_arm
        q2_by_arm_hits[arm] = hits
        if q2_clusters is None:
            q2_clusters = g["cluster_ids"]
        q2_arms[arm] = {"n": int(len(ys)), "acc": float(np.mean(hits)), "nll": float(np.mean(nlls))}

    # Verify M3 anti-double-counting & distractor resistance relative to `valid`
    max_diff_dedup = float(np.max(np.abs(q2_by_arm_probs["duplicate_copy_dedup"] - q2_by_arm_probs["valid"])))
    max_diff_raw = float(np.max(np.abs(q2_by_arm_probs["duplicate_copy_raw"] - q2_by_arm_probs["valid"])))
    max_diff_indep = float(np.max(np.abs(q2_by_arm_probs["valid_plus_independent"] - q2_by_arm_probs["valid"])))
    q2_arms["fresh_minus_valid_acc"] = q2_arms["fresh_resample"]["acc"] - q2_arms["valid"]["acc"]
    q2_arms["max_prob_diff_duplicate_dedup_vs_valid"] = max_diff_dedup
    q2_arms["max_prob_diff_duplicate_raw_vs_valid"] = max_diff_raw
    q2_arms["max_prob_diff_valid_plus_indep_vs_valid"] = max_diff_indep
    q2_arms["_fresh_hits"] = q2_by_arm_hits["fresh_resample"]
    q2_arms["_valid_hits"] = q2_by_arm_hits["valid"]
    q2_arms["_clusters"] = q2_clusters
    summary["Q2_C4_blind"] = q2_arms

    # Counterfactual breakdown
    cf_grps = suites["G1_counterfactuals"]
    cf_preds = model_preds["G1_counterfactuals"]
    by_var: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {}
    for g, p_g in zip(cf_grps, cf_preds):
        by_var.setdefault(g["variant"], ([], []))
        by_var[g["variant"]][0].append(p_g)
        by_var[g["variant"]][1].append(g["labels"])
    p_fact = np.concatenate(by_var["factual"][0], axis=0)
    y_fact = np.concatenate(by_var["factual"][1], axis=0)
    p_slot = np.concatenate(by_var["slot_flipped"][0], axis=0)
    y_slot = np.concatenate(by_var["slot_flipped"][1], axis=0)
    p_text = np.concatenate(by_var["text_flipped"][0], axis=0)
    y_text = np.concatenate(by_var["text_flipped"][1], axis=0)
    summary["G1_counterfactuals"] = {
        "n": int(len(y_fact)),
        "factual_acc": float(np.mean(p_fact.argmax(axis=-1) == y_fact)),
        "slot_flipped_acc": float(np.mean(p_slot.argmax(axis=-1) == y_slot)),
        "text_flipped_acc": float(np.mean(p_text.argmax(axis=-1) == y_text)),
        "mean_sym_kl_factual_vs_slot_flipped": float(np.mean(_sym_kl_arrays(p_fact, p_slot))),
        "mean_tv_factual_vs_slot_flipped": float(np.mean(0.5 * np.sum(np.abs(p_fact - p_slot), axis=-1))),
        "mean_sym_kl_factual_vs_text_flipped": float(np.mean(_sym_kl_arrays(p_fact, p_text))),
    }
    return summary


def run_stage4_synthetic_blind(
    repo_root: Path,
    out_dir: Path,
    ckpt_dir: Path,
    kfac_path: Path,
    seed: int = 202611,
    g1_blocks: int = 131,
    eval_bs: int = 64,
    scorer_mode: str = "kv_cached",
    model_id: str = MODEL_ID_DEFAULT,
    revision: str = PINNED_REVISION_DEFAULT,
    mock_tiny_for_test: bool = False,
    host: Optional[QwenCacheHost] = None,
    cache_dir: Optional[Path] = None,
    on_model_done: Optional[Callable[[str], None]] = None,
) -> dict:
    """Evaluate all 7 Stage 4 models on disjoint G1..G4, Q2-C4 (7 arms), counterfactuals, and M2 paired stats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    if host is None:
        host = QwenCacheHost(model_id=model_id, revision=revision, mock_tiny_for_test=mock_tiny_for_test)
    if not mock_tiny_for_test:
        assert int(getattr(host.text_config, "vocab_size", 0)) == EXPECTED_4B_VOCAB
        assert host.host_sha256 == EXPECTED_4B_HOST_SHA256

    encoder, _ = load_s17_encoder(repo_root)
    other_blocks = 1 if mock_tiny_for_test else 8
    suites = _build_blind_suites(encoder, g1_blocks=g1_blocks, other_blocks=other_blocks)

    layers = _get_decoder_layers(host.lm)
    n_l = len(layers)
    fa_l = [i for i in QWEN4B_FA_LAYERS if i < n_l] or ([n_l - 1] if n_l >= 1 else [])
    dn_l = [i for i in QWEN4B_DN_LAYERS if i < n_l]
    stop_l = min(QWEN4B_STOP_LAYER, max(0, n_l - 2))
    proj_map = _load_or_build_kfac(host, kfac_path, fa_l, dn_l, mock_tiny_for_test)

    def _load_conn(arm_name: str) -> Stage4MatchedConnector:
        mod = Stage4MatchedConnector(arm_name, encoder.table, hidden_size=host.hidden_size).to(host.device)
        p = ckpt_dir / f"stage4_phaseA_{arm_name}_s{seed}_step384.pt"
        if p.exists():
            ck = torch.load(p, map_location=host.device, weights_only=False)
            mod.load_state_dict(ck["state_dict"])
        elif not mock_tiny_for_test:
            raise FileNotFoundError(f"Missing Phase-A checkpoint: {p}")
        mod.eval()
        return mod

    def _eval_cached(mname: str, eval_fn: Callable[[], Dict[str, List[np.ndarray]]]) -> dict:
        m_t0 = time.perf_counter()
        preds: Optional[Dict[str, List[np.ndarray]]] = None
        cache_p = (cache_dir / f"{mname}.pt") if cache_dir is not None else None
        if cache_p is not None and cache_p.exists():
            ck = torch.load(cache_p, map_location="cpu", weights_only=False)
            if (
                isinstance(ck, dict)
                and int(ck.get("g1_blocks", -1)) == int(g1_blocks)
                and int(ck.get("other_blocks", -1)) == int(other_blocks)
                and int(ck.get("seed", -1)) == int(seed)
                and str(ck.get("host_sha256", "")) == str(host.host_sha256)
                and "preds" in ck
            ):
                preds = ck["preds"]
        loaded_from_cache = preds is not None
        if preds is None:
            preds = eval_fn()
            if cache_p is not None:
                tmp_p = cache_dir / f".{mname}.pt.tmp"
                torch.save(
                    {
                        "g1_blocks": int(g1_blocks),
                        "other_blocks": int(other_blocks),
                        "seed": int(seed),
                        "host_sha256": str(host.host_sha256),
                        "preds": preds,
                    },
                    tmp_p,
                )
                tmp_p.replace(cache_p)
                if on_model_done is not None:
                    on_model_done(mname)
        summ = _summarize_model_blind(suites, preds)
        g1m = summ["G1_main_extrap_both"]
        q2m = summ["Q2_C4_blind"]
        tag = " (cached)" if loaded_from_cache else ""
        print(
            f"[BLIND {mname}] G1={g1m['acc']:.4f}/{g1m['nll']:.4f} "
            f"Q2_v={q2m['valid']['acc']:.4f} Q2_f={q2m['fresh_resample']['acc']:.4f} "
            f"elapsed={time.perf_counter() - m_t0:.1f}s{tag}",
            flush=True,
        )
        return summ

    raw_summaries: Dict[str, dict] = {}
    t0 = time.perf_counter()

    # 1. Direct physical oracle
    raw_summaries["direct_physical"] = _eval_cached(
        "direct_physical",
        lambda: _predict_model_on_suites(host, None, suites, adapter=None, eval_bs=eval_bs, scorer_mode=scorer_mode),
    )

    # 2-5. Frozen-host 384-step connectors
    for arm in ("zero", "naive_additive", "equivariant_host_pure", "equivariant_physical_residual"):
        def _eval_arm(arm_name: str = arm) -> Dict[str, List[np.ndarray]]:
            conn = _load_conn(arm_name)
            try:
                return _predict_model_on_suites(host, conn, suites, adapter=None, eval_bs=eval_bs, scorer_mode=scorer_mode)
            finally:
                del conn

        raw_summaries[arm] = _eval_cached(arm, _eval_arm)

    # 6. Mode-I Surgical-96
    def _eval_surg() -> Dict[str, List[np.ndarray]]:
        surg_ckpt_p = ckpt_dir / f"stage4_phaseB_surgical96_s{seed}.pt"
        if not surg_ckpt_p.exists():
            raise FileNotFoundError(f"Missing Phase-B Surgical-96 checkpoint: {surg_ckpt_p}")
        pure_conn = _load_conn("equivariant_host_pure")
        s_ck = torch.load(surg_ckpt_p, map_location="cpu", weights_only=False)
        surg = SurgicalHostAdapter(
            host.lm,
            fa_layers=fa_l,
            dn_layers=dn_l,
            stop_layer=stop_l,
            mode="mode1_stopgrad_l10",
            u_c_by_layer=proj_map,
            b_align=s_ck["b_align"],
        ).to(host.device)
        try:
            surg.load_state_dict(s_ck["state_dict"])
            return _predict_model_on_suites(host, pure_conn, suites, adapter=surg, eval_bs=eval_bs, scorer_mode=scorer_mode)
        finally:
            surg.remove()
            del surg, s_ck, pure_conn

    raw_summaries["mode1_surgical_96"] = _eval_cached("mode1_surgical_96", _eval_surg)

    # 7. Matched LEMMO Tier-P1
    def _eval_lemmo() -> Dict[str, List[np.ndarray]]:
        lemmo_ckpt_p = ckpt_dir / f"stage4_phaseB_lemmo_tier_p1_s{seed}.pt"
        if not lemmo_ckpt_p.exists():
            raise FileNotFoundError(f"Missing Phase-B LEMMO Tier-P1 checkpoint: {lemmo_ckpt_p}")
        pure_conn = _load_conn("equivariant_host_pure")
        l_ck = torch.load(lemmo_ckpt_p, map_location="cpu", weights_only=False)
        lemmo = UnconstrainedLoRAAdapter(host.lm, fa_layers=fa_l, dn_layers=dn_l, stop_layer=stop_l, r_lora=8).to(host.device)
        try:
            lemmo.load_state_dict(l_ck["state_dict"])
            return _predict_model_on_suites(host, pure_conn, suites, adapter=lemmo, eval_bs=eval_bs, scorer_mode=scorer_mode)
        finally:
            lemmo.remove()
            del lemmo, l_ck, pure_conn

    raw_summaries["lemmo_tier_p1"] = _eval_cached("lemmo_tier_p1", _eval_lemmo)

    assert _compute_host_fingerprint(host.lm) == host.host_sha256, "Host SHA mutated during blind eval!"

    # M2 Paired Cluster Statistics & IUT + Holm-Bonferroni
    def _g1_pair(m_a: str, m_b: str) -> dict:
        cl = raw_summaries[m_a]["G1_main_extrap_both"]["_clusters"]
        st_acc = _cluster_paired_stats(
            raw_summaries[m_a]["G1_main_extrap_both"]["_hits"],
            raw_summaries[m_b]["G1_main_extrap_both"]["_hits"],
            cl,
            higher_is_better_a=True,
            seed=seed,
        )
        st_nll = _cluster_paired_stats(
            raw_summaries[m_a]["G1_main_extrap_both"]["_nlls"],
            raw_summaries[m_b]["G1_main_extrap_both"]["_nlls"],
            cl,
            higher_is_better_a=False,
            seed=seed + 1,
        )
        iut_p = max(st_acc["p_one_sided_improved"], st_nll["p_one_sided_improved"])
        return {"acc_diff_a_minus_b": st_acc, "nll_diff_a_minus_b": st_nll, "iut_one_sided_p": float(iut_p)}

    g1_pure_vs_zero = _g1_pair("equivariant_host_pure", "zero")
    g1_pure_vs_naive = _g1_pair("equivariant_host_pure", "naive_additive")
    g1_res_vs_phys = _g1_pair("equivariant_physical_residual", "direct_physical")
    g1_surg_vs_pure = _g1_pair("mode1_surgical_96", "equivariant_host_pure")
    g1_surg_vs_lemmo = _g1_pair("mode1_surgical_96", "lemmo_tier_p1")

    # Pooled G1..G4 NLL comparison for Surgical-96 vs Equivariant-Host-Pure
    def _pooled_g14_nll(mname: str) -> Tuple[np.ndarray, np.ndarray]:
        nlls_list, cl_list = [], []
        for off, cid in enumerate(("G1_main_extrap_both", "G2_interp_both", "G3_extrap_angle", "G4_extrap_freq")):
            nlls_list.append(raw_summaries[mname][cid]["_nlls"])
            cl_list.append(raw_summaries[mname][cid]["_clusters"] + off * 1000)
        return np.concatenate(nlls_list), np.concatenate(cl_list)

    nll_s14, cl_14 = _pooled_g14_nll("mode1_surgical_96")
    nll_p14, _ = _pooled_g14_nll("equivariant_host_pure")
    g14_surg_vs_pure_nll = _cluster_paired_stats(nll_s14, nll_p14, cl_14, higher_is_better_a=False, seed=seed + 2)

    q2_gain_stats: Dict[str, Any] = {}
    for mname in ("equivariant_host_pure", "equivariant_physical_residual", "mode1_surgical_96", "lemmo_tier_p1"):
        q2d = raw_summaries[mname]["Q2_C4_blind"]
        q2_gain_stats[mname] = _cluster_paired_stats(
            q2d["_fresh_hits"], q2d["_valid_hits"], q2d["_clusters"], higher_is_better_a=True, seed=seed + 3
        )

    raw_family_p = {
        "H1_pure_beats_zero_G1_IUT": g1_pure_vs_zero["iut_one_sided_p"],
        "H2_pure_beats_naive_G1_IUT": g1_pure_vs_naive["iut_one_sided_p"],
        "H3_surg96_beats_pure_G14_NLL": g14_surg_vs_pure_nll["p_one_sided_improved"],
        "H4_surg96_Q2_Blackwell_gain": q2_gain_stats["mode1_surgical_96"]["p_one_sided_improved"],
        "H5_pure_Q2_Blackwell_gain": q2_gain_stats["equivariant_host_pure"]["p_one_sided_improved"],
    }
    holm_adj = _holm_bonferroni(raw_family_p)

    # Strip private numpy arrays before JSON serialization
    clean_models: Dict[str, dict] = {}
    for mname, mdict in raw_summaries.items():
        clean_models[mname] = {}
        for k, v in mdict.items():
            clean_models[mname][k] = {sk: sv for sk, sv in v.items() if not sk.startswith("_")}

    blind_report = {
        "seed": seed,
        "model_id": host.model_id,
        "host_sha256": host.host_sha256,
        "g1_blocks": int(g1_blocks),
        "other_blocks": int(other_blocks),
        "eval_wall_s": round(time.perf_counter() - t0, 2),
        "models": clean_models,
        "m2_paired_stats": {
            "G1_equivariant_host_pure_vs_zero": g1_pure_vs_zero,
            "G1_equivariant_host_pure_vs_naive_additive": g1_pure_vs_naive,
            "G1_equivariant_physical_residual_vs_direct_physical": g1_res_vs_phys,
            "G1_mode1_surgical_96_vs_equivariant_host_pure": g1_surg_vs_pure,
            "G1_mode1_surgical_96_vs_lemmo_tier_p1": g1_surg_vs_lemmo,
            "G1_G4_mode1_surgical_96_vs_equivariant_host_pure_NLL": g14_surg_vs_pure_nll,
            "Q2_C4_fresh_minus_valid_gain": q2_gain_stats,
            "holm_bonferroni_family": {"raw_p": raw_family_p, "adjusted_p": holm_adj},
        },
    }
    (out_dir / "stage4_synthetic_blind_results.json").write_text(
        json.dumps(blind_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return blind_report
