"""Unified N1+N2+N3+N4 Connecting-Neuron Surgery (CN-SFT) 4B probe & verification runner."""

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
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.stage3_ministral_cache.ministral_connector import _compute_host_fingerprint
from research.stage3d_qwen_signed_phase.data_stage3d import load_s17_encoder
from research.stage3d_qwen_signed_phase.qwen_connector import (
    MODEL_ID_DEFAULT,
    PINNED_REVISION_DEFAULT,
    QwenCacheHost,
)
from research.stage4_real_and_surgical.cn_sft_adapter import (
    CNSFTHostAdapter,
    F_REAL_4X3,
    GENERAL_KFAC_64_PROMPTS,
    QWEN4B_MLP_TUNE_LAYERS,
    _get_final_norm,
    _realify_uc,
    build_regime2_separate_carrier,
    build_three_way_orthogonal_bus,
    calibrate_b_align_gamma_f,
    collect_dual_space_kfac_projectors,
    first_divergent_token_index,
    score_batch_with_cn_sft,
    verify_rmsnorm_gamma,
)
from research.stage4_real_and_surgical.equi_connector import (
    PhysicalKernelS17,
    Stage4MatchedConnector,
    TASK_TO_ID,
)
from research.stage4_real_and_surgical.phaseB_surgical import (
    GENERAL_TEXT_EVAL_PROMPTS,
    GENERAL_TEXT_TRAIN_PROMPTS,
    _build_blind_suites,
    _build_fixed_mix_prefix,
    _cluster_paired_stats,
    _forward_last_token_logits,
    _holm_bonferroni,
    _load_or_init_phaseA_pure_connector,
    _make_lr_scheduler,
    _reconstruct_probs_from_centered_logits,
    _summarize_model_blind,
    _weighted_phaseB_minibatch_order,
    audit_pure_and_mix_kl,
    compute_frozen_general_refs,
)
from research.stage4_real_and_surgical.surgical_adapter import (
    QWEN4B_DN_LAYERS,
    QWEN4B_FA_LAYERS,
    QWEN4B_STOP_LAYER,
    TIER_P1_MAX_PARAMS,
    _get_decoder_layers,
    rao_blackwell_bayes_kl_cf_loss,
)
from research.stage4_real_and_surgical.train_eval_stage4 import (
    EXPECTED_4B_HOST_SHA256,
    EXPECTED_4B_VOCAB,
    _evaluate_dev_groups,
    build_stage4_pilot_groups,
)


def _encode_chat_or_raw_ids(host: QwenCacheHost, prompt: str) -> torch.Tensor:
    if host.tokenizer is None:
        return torch.tensor([[1, 2, 3, 4, 5, 6]], device=host.device, dtype=torch.long)
    parts = host.format_chat_parts(prompt)
    ids: List[int] = []
    for seg in parts:
        ids.extend(host.tokenizer.encode(seg, add_special_tokens=False))
    if not ids:
        ids = [int(host.tokenizer.eos_token_id or 1)]
    return torch.tensor([ids], device=host.device, dtype=torch.long)


def _collect_qgate_layer_states(
    host: QwenCacheHost,
    stop_layer: int,
    forwards: List[Tuple[str, Callable[[], None]]],
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Run callbacks while capturing states at exact Fisher Query Gate layer."""
    layers = _get_decoder_layers(host.lm)
    n_l = len(layers)
    captured_all: List[torch.Tensor] = []
    captured_last: List[torch.Tensor] = []
    latest: List[torch.Tensor] = []

    if n_l >= 4:
        tgt_idx = max(0, min(int(stop_layer), n_l - 1))

        def _cap_out(mod, args, out):
            hs = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()
            latest.append(hs)

        handle = layers[tgt_idx].register_forward_hook(_cap_out)
    else:
        def _cap_in(mod, args):
            hs = args[0].detach().float().cpu()
            latest.append(hs)

        handle = layers[0].register_forward_pre_hook(_cap_in)

    try:
        with torch.no_grad():
            for _tag, fn in forwards:
                latest.clear()
                fn()
                if latest:
                    hs = latest[-1]
                    captured_last.append(hs[:, -1, :].reshape(-1, host.hidden_size))
                    captured_all.append(hs.reshape(-1, host.hidden_size))
    finally:
        handle.remove()
    return captured_last, captured_all


def calibrate_fisher_qgate(
    host: QwenCacheHost,
    connector: Stage4MatchedConnector,
    kfac: Dict[str, Any],
    stop_layer: int,
    tr_groups: List[dict],
    seed: int = 202611,
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    """Calibrate Fisher LDA Query Gate separating EM Pass-1 last token vs prose & mixed-prefix prose (zero labels)."""
    d = int(host.hidden_size)
    rng = np.random.default_rng(seed + 5150)

    # 1. Positive states: Pass-1 last token across ALL distinct EM prompt templates (dummy random codes)
    seen_prompts: set[Tuple[str, str]] = set()
    em_templates: List[dict] = []
    for g in tr_groups:
        key = (g["qtype"], g["prompt"])
        if key not in seen_prompts:
            seen_prompts.add(key)
            em_templates.append(g)

    pos_calls: List[Tuple[str, Callable[[], None]]] = []
    for g in em_templates:
        b_dum = 2 if len(em_templates) > 4 else 4
        w_dum = torch.from_numpy(rng.integers(0, 16, size=(b_dum, 2), dtype=np.int64)).to(host.device)
        tid_val = int(TASK_TO_ID[g["qtype"]])
        tid_t = torch.full((b_dum,), tid_val, device=host.device, dtype=torch.long)
        ps_t = torch.from_numpy(g["phase_slots"][:b_dum]).to(host.device)
        meta_t = torch.from_numpy(g["meta_scalars"][:b_dum]).to(host.device)
        rs_dum = torch.zeros((b_dum, 29), device=host.device, dtype=torch.float32)
        rs_dum[:, 0:8] = 0.25

        def _make_em_fn(prompt_s=g["prompt"], w=w_dum, tid=tid_t, ps=ps_t, mt=meta_t, rs=rs_dum):
            def _run():
                st, _, _, _ = connector.forward_tokens(w, 4, tid, ps, mt, ref_summary=rs)
                pref, _ = connector.build_host_prefix_embeds(host, prompt_s, st)
                attn = torch.ones((pref.shape[0], pref.shape[1]), device=host.device, dtype=torch.long)
                host._lm_forward(inputs_embeds=pref, attention_mask=attn, use_cache=False)
            return _run

        pos_calls.append((g["qtype"], _make_em_fn()))

    pos_last, _ = _collect_qgate_layer_states(host, stop_layer, pos_calls)
    h_pos = torch.cat(pos_last, dim=0).to(torch.float64)

    # 2. Negative states: balanced mix of raw prose, mixed-prefix prose, & chat-wrapped prose
    neg_raw = kfac.get("stop_prose_samples", torch.zeros((0, d), dtype=torch.float32))
    if neg_raw.shape[0] > 128:
        step = max(1, neg_raw.shape[0] // 128)
        neg_raw = neg_raw[::step][:128]

    mix_prompts = list(GENERAL_TEXT_TRAIN_PROMPTS) + list(GENERAL_KFAC_64_PROMPTS[:16])
    mix_calls: List[Tuple[str, Callable[[], None]]] = []
    for pr in mix_prompts:
        def _make_mix_fn(prompt_s=pr):
            def _run():
                emb, _ = _build_fixed_mix_prefix(host, connector, prompt_s)
                attn = torch.ones((emb.shape[0], emb.shape[1]), device=host.device, dtype=torch.long)
                host._lm_forward(inputs_embeds=emb, attention_mask=attn, use_cache=False)
            return _run
        mix_calls.append(("mix", _make_mix_fn()))
    neg_mix_last, neg_mix_all = _collect_qgate_layer_states(host, stop_layer, mix_calls)
    h_neg_mix = torch.cat(neg_mix_last, dim=0).to(torch.float64)
    h_neg_mix_toks = (
        torch.cat(neg_mix_all, dim=0).to(torch.float64)
        if neg_mix_all
        else torch.zeros((0, d), dtype=torch.float64)
    )
    if h_neg_mix_toks.shape[0] > 256:
        step_mt = max(1, h_neg_mix_toks.shape[0] // 256)
        h_neg_mix_toks = h_neg_mix_toks[::step_mt][:256]

    chat_calls: List[Tuple[str, Callable[[], None]]] = []
    embed_layer = host.lm.get_input_embeddings()
    for pr in GENERAL_KFAC_64_PROMPTS[:16]:
        def _make_chat_fn(prompt_s=pr):
            def _run():
                ids = _encode_chat_or_raw_ids(host, prompt_s)
                emb = embed_layer(ids)
                attn = torch.ones_like(ids)
                host._lm_forward(inputs_embeds=emb, attention_mask=attn, use_cache=False)
            return _run
        chat_calls.append(("chat", _make_chat_fn()))
    neg_chat_last, neg_chat_all = _collect_qgate_layer_states(host, stop_layer, chat_calls)
    h_neg_chat = torch.cat(neg_chat_last, dim=0).to(torch.float64)
    h_neg_raw = neg_raw.to(torch.float64) if neg_raw.numel() > 0 else torch.zeros((0, d), dtype=torch.float64)

    # Repeat mix negatives 4x in covariance so mixed-prefix assistant tail matches raw weight without hiding tails
    neg_parts = [p for p in (h_neg_raw, h_neg_chat, h_neg_mix_toks, h_neg_mix) if p.numel() > 0]
    h_neg_all = torch.cat(neg_parts, dim=0)
    h_neg_cov_src = torch.cat(
        [p for p in (h_neg_raw, h_neg_chat, h_neg_mix_toks, h_neg_mix, h_neg_mix, h_neg_mix, h_neg_mix) if p.numel() > 0],
        dim=0,
    )

    mu_pos = h_pos.mean(dim=0)
    mu_neg = h_neg_cov_src.mean(dim=0)
    c_pos = ((h_pos - mu_pos).T @ (h_pos - mu_pos)) / max(1, h_pos.shape[0] - 1)
    c_neg = ((h_neg_cov_src - mu_neg).T @ (h_neg_cov_src - mu_neg)) / max(1, h_neg_cov_src.shape[0] - 1)
    sigma_w = 0.5 * (c_pos + c_neg)
    reg_lam = 0.1 * max(1e-6, float(torch.trace(sigma_w).item()) / float(d))
    sigma_reg = sigma_w + reg_lam * torch.eye(d, dtype=torch.float64)

    w_raw = torch.linalg.solve(sigma_reg, (mu_pos - mu_neg).unsqueeze(-1)).squeeze(-1)
    w_unit = w_raw / w_raw.norm().clamp_min(1e-12)

    s_pos = h_pos @ w_unit
    s_neg = h_neg_all @ w_unit
    s_mix = h_neg_mix @ w_unit
    s_pos_min = float(s_pos.min().item())
    s_pos_mean = float(s_pos.mean().item())
    s_neg_max = float(s_neg.max().item())
    s_neg_mean = float(s_neg.mean().item())
    s_mix_max = float(s_mix.max().item())

    if s_pos_min > s_neg_max:
        mid = 0.5 * (s_pos_min + s_neg_max)
        half_gap = 0.5 * (s_pos_min - s_neg_max)
        scale = min(40.0, 6.0 / max(1e-3, half_gap))
        separable = True
    else:
        mid = 0.5 * (s_pos_mean + s_neg_mean)
        half_gap = max(1e-2, 0.5 * abs(s_pos_mean - s_neg_mean))
        scale = min(40.0, 6.0 / half_gap)
        separable = False

    qgate_w = (w_unit * scale).to(torch.float32).contiguous()
    qgate_b = torch.tensor(-mid * scale, dtype=torch.float32)

    # Verify raw plateau values on calibration distributions
    with torch.no_grad():
        sc_pos = s_pos * scale - mid * scale
        sc_neg = s_neg * scale - mid * scale
        g_pos_min = float(torch.clamp((torch.sigmoid(sc_pos) - 0.02) / 0.96, 0.0, 1.0).min().item())
        g_neg_max = float(torch.clamp((torch.sigmoid(sc_neg) - 0.02) / 0.96, 0.0, 1.0).max().item())

    return qgate_w, qgate_b, {
        "n_pos": int(h_pos.shape[0]),
        "n_neg": int(h_neg_all.shape[0]),
        "linearly_separable": separable,
        "s_pos_min": round(s_pos_min, 5),
        "s_pos_mean": round(s_pos_mean, 5),
        "s_neg_max": round(s_neg_max, 5),
        "s_mix_max": round(s_mix_max, 5),
        "scale": round(scale, 4),
        "calib_pos_gate_min": round(g_pos_min, 6),
        "calib_neg_gate_max": round(g_neg_max, 6),
    }


def run_l10_ood_theta_probe(
    host: QwenCacheHost,
    connector: Stage4MatchedConnector,
    stop_layer: int,
    seed: int = 202611,
    mock_tiny_for_test: bool = False,
) -> Tuple[int, torch.Tensor, torch.Tensor, dict]:
    """Probe Layer-10 OOD-theta linear recoverability -> Regime 1 (R^2 >= 0.85) or Regime 2."""
    from research.stage3d_qwen_signed_phase.data_stage3d import (
        build_stage3d_prompt,
        condition_spec_from_params,
        public_phase_slots,
    )

    layers = _get_decoder_layers(host.lm)
    n_l = len(layers)
    probe_idx = max(0, min(int(stop_layer), n_l - 1)) if n_l >= 1 else 0
    d = int(host.hidden_size)
    thetas = (0.0, 0.5, 1.0, 1.60, math.pi)
    n_per = 8 if mock_tiny_for_test else 32
    rng = np.random.default_rng(seed + 91827)

    h_val, ell_val, tp_val, snr_val, sig_val = 2, 2, 4.0, 2.0, 0.10
    feats_list: List[torch.Tensor] = []
    tgts_list: List[torch.Tensor] = []
    theta_tags: List[float] = []
    kind_tags: List[int] = []  # 0=w1, 1=w2, 2=ph

    captured: List[torch.Tensor] = []

    def _cap_hook(mod, args, out):
        hs = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()
        captured.append(hs)

    h_reg = layers[probe_idx].register_forward_hook(_cap_hook)
    try:
        with torch.no_grad():
            for th in thetas:
                omega = float(th) / float(h_val + ell_val)
                cond = condition_spec_from_params(omega, h_val, ell_val, tp_val, snr_val, sig_val, tag="ood_probe")
                prompt, _ = build_stage3d_prompt("Q3", cond, protocol="repeat_x1")
                p_vec = public_phase_slots(omega, h_val, ell_val, tp_val, zero_arm=False)
                meta_vec = np.array([h_val, ell_val, tp_val, omega, sig_val, snr_val, 0.0], dtype=np.float32)

                words_np = rng.integers(0, 16, size=(n_per, 2), dtype=np.int64)
                ps_np = np.repeat(p_vec[None, :], n_per, axis=0)
                meta_np = np.repeat(meta_vec[None, :], n_per, axis=0)
                dd_np = np.zeros((n_per,), dtype=np.float32)

                words_t = torch.from_numpy(words_np).to(host.device)
                ps_t = torch.from_numpy(ps_np).to(host.device)
                meta_t = torch.from_numpy(meta_np).to(host.device)
                dd_t = torch.from_numpy(dd_np).to(host.device)
                tid_t = torch.full((n_per,), TASK_TO_ID["Q3_repeat_x1"], device=host.device, dtype=torch.long)

                carrier = build_regime2_separate_carrier(
                    connector.prior_canon,
                    words_t,
                    task="Q3_repeat_x1",
                    phase_slots=ps_t,
                    meta_scalars=meta_t,
                    discrete_d=dd_t,
                    bits=4,
                )
                st, _, _, _ = connector.forward_tokens(words_t, 4, tid_t, ps_t, meta_t)
                pref, m_mask = connector.build_host_prefix_embeds(host, prompt, st)
                true_pos = torch.nonzero(m_mask[0].bool(), as_tuple=False).flatten().cpu()
                attn = torch.ones((n_per, pref.shape[1]), device=host.device, dtype=torch.long)

                captured.clear()
                host._lm_forward(inputs_embeds=pref, attention_mask=attn, use_cache=False)
                hs_stop = captured[-1]  # (B, L, D)

                for kind_id, (idx_slice, key_name) in enumerate(
                    ((true_pos[0:2], "w1_vec"), (true_pos[2:4], "w2_vec"), (true_pos[4:6], "ph_vec"))
                ):
                    if idx_slice.numel() == 0:
                        continue
                    f_m = hs_stop[:, idx_slice, :].mean(dim=1)  # (B, D)
                    y_m = carrier[key_name].detach().float().cpu()  # (B, 16)
                    feats_list.append(f_m)
                    tgts_list.append(y_m)
                    theta_tags.extend([float(th)] * n_per)
                    kind_tags.extend([kind_id] * n_per)
    finally:
        h_reg.remove()

    x_all = torch.cat(feats_list, dim=0).to(torch.float64)
    y_all = torch.cat(tgts_list, dim=0).to(torch.float64)
    th_arr = np.asarray(theta_tags, dtype=np.float64)
    kd_arr = np.asarray(kind_tags, dtype=np.int64)

    tr_mask = torch.from_numpy(th_arr <= 1.05)
    ood_mask = torch.from_numpy(th_arr > 1.05)
    ood_ph_mask = torch.from_numpy((th_arr > 1.05) & (kd_arr == 2))

    x_tr = x_all[tr_mask]
    y_tr = y_all[tr_mask]
    x_mean = x_tr.mean(dim=0, keepdim=True)
    y_mean = y_tr.mean(dim=0, keepdim=True)
    xc = x_tr - x_mean
    yc = y_tr - y_mean

    # Fast exact ridge in dual space when N < D, primal otherwise
    n_tr = xc.shape[0]
    lam = 1e-2 * max(1.0, float((xc.pow(2).sum() / max(1, n_tr)).item()))
    if n_tr <= d:
        gram = xc @ xc.T + lam * torch.eye(n_tr, dtype=torch.float64)
        w_64 = xc.T @ torch.linalg.solve(gram, yc)
    else:
        gram = xc.T @ xc + lam * torch.eye(d, dtype=torch.float64)
        w_64 = torch.linalg.solve(gram, xc.T @ yc)
    b_64 = y_mean.squeeze(0) - x_mean.squeeze(0) @ w_64

    def _r2(y_t: torch.Tensor, y_p: torch.Tensor) -> float:
        ss_res = float((y_t - y_p).pow(2).sum().item())
        ss_tot = float((y_t - y_t.mean(dim=0, keepdim=True)).pow(2).sum().item())
        if ss_tot <= 1e-12:
            return 0.0
        return 1.0 - ss_res / ss_tot

    y_ood = y_all[ood_mask]
    p_ood = x_all[ood_mask] @ w_64 + b_64
    r2_all = _r2(y_ood[:, :13], p_ood[:, :13])

    y_ood_ph = y_all[ood_ph_mask][:, 10:13]
    p_ood_ph = (x_all[ood_ph_mask] @ w_64 + b_64)[:, 10:13]
    r2_ph = _r2(y_ood_ph, p_ood_ph)

    regime = 1 if (n_l >= 4 and r2_ph >= 0.85 and r2_all >= 0.85) else 2
    rep = {
        "probe_layer": int(probe_idx),
        "n_train": int(tr_mask.sum().item()),
        "n_ood": int(ood_mask.sum().item()),
        "r2_ood_phase_slots": round(r2_ph, 5),
        "r2_ood_all_slots": round(r2_all, 5),
        "selected_regime": int(regime),
    }
    return regime, w_64.float().contiguous(), b_64.float().contiguous(), rep


def calibrate_template_suffix_bias(
    host: QwenCacheHost,
    connector: Stage4MatchedConnector,
    adapter: CNSFTHostAdapter,
    tr_groups: List[dict],
) -> dict:
    """Calibrate zero-label 1-pass empty-carrier suffix logit bias per canonical task ID 0..4."""
    if adapter.circuit_a is None:
        return {"skipped": True}

    dev = host.device
    ca = adapter.circuit_a
    ca.dyn_debias_scale.data.fill_(1.0)
    ca.suffix_debias_scale.data.fill_(1.0)
    ca.template_suffix_canon_bias.zero_()

    # Pick one representative template group for each canonical task ID 0..4
    rep_by_tid: Dict[int, dict] = {}
    for g in tr_groups:
        tid = int(TASK_TO_ID[g["qtype"]])
        if tid not in rep_by_tid:
            rep_by_tid[tid] = g

    def _run_empty_carrier(g_row: dict, tid: int) -> Tuple[torch.Tensor, float]:
        w_dum = torch.zeros((1, 2), device=dev, dtype=torch.long)
        tid_t = torch.tensor([tid], device=dev, dtype=torch.long)
        ps_z = torch.zeros((1, 6), device=dev, dtype=torch.float32)
        mt = torch.from_numpy(g_row["meta_scalars"][0:1]).to(dev)
        rs = torch.zeros((1, 29), device=dev, dtype=torch.float32)
        rs[:, 0:8] = 0.25
        v_false = torch.zeros((1, 2), device=dev, dtype=torch.bool)
        with torch.no_grad():
            st, _, _, _ = connector.forward_tokens(w_dum, 4, tid_t, ps_z, mt, ref_summary=rs, valid=v_false)
            pref, m_mask = connector.build_host_prefix_embeds(host, g_row["prompt"], st)
        t_div = first_divergent_token_index(host, g_row["choices"])

        empty_carrier = None
        if adapter.regime == 2:
            w1_v = torch.zeros((1, 16), device=dev, dtype=torch.float32)
            w2_v = torch.zeros((1, 16), device=dev, dtype=torch.float32)
            ph_v = torch.zeros((1, 16), device=dev, dtype=torch.float32)
            w1_v[:, 13] = 1.0
            w2_v[:, 14] = 1.0
            ph_v[:, 9] = 1.0
            ph_v[:, 15] = 1.0
            empty_carrier = {"w1_vec": w1_v, "w2_vec": w2_v, "ph_vec": ph_v, "shift": torch.zeros((1,), device=dev, dtype=torch.long)}

        with torch.no_grad(), adapter.activate(
            g_em=1, m_acc_mask=m_mask, carrier=empty_carrier, t_div=t_div, task_id=tid, apply_c4_shift=False
        ):
            if getattr(host, "tokenizer", None) is None:
                attn = torch.ones((1, pref.shape[1]), device=dev, dtype=torch.long)
                raw = host.lm(inputs_embeds=pref, attention_mask=attn, use_cache=False).logits[:, -1, :4].float()
            else:
                _, raw = host.score_candidates_kv(pref, g_row["choices"], temperature=1.0)
            qg_val = float(adapter._cached_qgate_p1_last.mean().item()) if adapter._cached_qgate_p1_last is not None else -1.0
        centered = raw - raw.mean(dim=-1, keepdim=True)
        return centered, qg_val

    f_r = F_REAL_4X3.to(device=dev, dtype=torch.float32)
    pre_errs: Dict[str, float] = {}
    post_errs: Dict[str, float] = {}
    qgates: Dict[str, float] = {}
    g_effs: Dict[str, float] = {}
    l2_diags: Dict[str, Any] = {}

    for tid, g_row in sorted(rep_by_tid.items()):
        c0, qg = _run_empty_carrier(g_row, tid)
        err0 = float(c0.abs().max().item())
        pre_errs[str(tid)] = round(err0, 5)
        qgates[str(tid)] = round(qg, 4)
        l2_diags[str(tid)] = dict(getattr(ca, "_last_diag", {}))
        best_err = err0
        best_bias = torch.zeros(3, device=dev, dtype=torch.float32)
        if err0 > 0.10:
            delta_0 = (c0[0] @ f_r).detach()
            b1 = 0.5 * delta_0
            ca.template_suffix_canon_bias[tid].copy_(b1)
            c1, _ = _run_empty_carrier(g_row, tid)
            err1 = float(c1.abs().max().item())
            if err1 < best_err:
                best_err = err1
                best_bias = b1.clone()
            num = float(((c0[0] - c1[0]) @ c0[0]).item())
            den = max(1e-6, 0.5 * float((c0[0] @ c0[0]).item()))
            g_eff = max(0.65, min(2.5, num / den))
            g_effs[str(tid)] = round(g_eff, 4)
            b2 = delta_0 / g_eff
            ca.template_suffix_canon_bias[tid].copy_(b2)
            c2, _ = _run_empty_carrier(g_row, tid)
            err2 = float(c2.abs().max().item())
            if err2 < best_err:
                best_err = err2
                best_bias = b2.clone()
            b3 = b2 + 0.5 * (c2[0] @ f_r).detach() / g_eff
            ca.template_suffix_canon_bias[tid].copy_(b3)
            c3, _ = _run_empty_carrier(g_row, tid)
            err3 = float(c3.abs().max().item())
            if err3 < best_err:
                best_err = err3
                best_bias = b3.clone()
        else:
            g_effs[str(tid)] = 1.0
        ca.template_suffix_canon_bias[tid].copy_(best_bias)

    max_post = 0.0
    for tid, g_row in sorted(rep_by_tid.items()):
        c_post, _ = _run_empty_carrier(g_row, tid)
        err = float(c_post.abs().max().item())
        post_errs[str(tid)] = round(err, 5)
        l2_diags[str(tid)] = dict(getattr(ca, "_last_diag", {}))
        max_post = max(max_post, err)

    b_align_col_norms = [round(float(ca.b_align[:, j].norm().item()), 5) for j in range(ca.b_align.shape[1])]
    out_calib = {
        "pre_max_abs_by_tid": pre_errs,
        "post_max_abs_by_tid": post_errs,
        "qgate_by_tid": qgates,
        "g_eff_by_tid": g_effs,
        "b_align_col_norms": b_align_col_norms,
        "l2_diag": l2_diags,
        "max_post_err": round(max_post, 5),
        "suffix_calib_converged": bool(max_post <= 0.5),
    }
    print("[CN-SFT Suffix Calib] " + json.dumps(out_calib, ensure_ascii=False), flush=True)
    return out_calib


def audit_cn_sft_pure_and_mix_kl(
    host: QwenCacheHost,
    connector: Stage4MatchedConnector,
    adapter: CNSFTHostAdapter,
    eval_prompts: Sequence[str] = GENERAL_TEXT_EVAL_PROMPTS,
    frozen_refs: Optional[Dict[str, List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]]] = None,
) -> dict:
    """Measure PureKL, Carrier-Free MixKL, and Active-Carrier MixKL on disjoint 12 eval prompts."""
    base_audit = audit_pure_and_mix_kl(
        host,
        connector,
        adapter,
        eval_prompts=eval_prompts,
        is_gated_surgical=True,
        frozen_refs=frozen_refs,
    )
    if frozen_refs is None:
        adapter.remove()
        try:
            frozen_refs = compute_frozen_general_refs(host, connector, eval_prompts)
        finally:
            adapter._attach_hooks(host.lm)

    active_carrier = None
    if adapter.regime == 2:
        active_carrier = build_regime2_separate_carrier(
            connector.prior_canon,
            codes=torch.tensor([[2, 3]], device=host.device, dtype=torch.long),
            task="Q1",
            phase_slots=torch.tensor([[0.80, 0.60, 0.50, -0.50, 0.90, 0.10]], device=host.device, dtype=torch.float32),
            meta_scalars=torch.tensor([[2.0, 2.0, 4.0, 0.25, 0.10, 2.0, 0.0]], device=host.device, dtype=torch.float32),
            bits=4,
        )

    act_kls: List[float] = []
    with torch.no_grad():
        for idx, pr in enumerate(eval_prompts):
            mix_emb, mix_mask = _build_fixed_mix_prefix(host, connector, pr)
            ref_lp_mix, _, _ = frozen_refs["mix"][idx]
            ref_lp_mix_dev = ref_lp_mix.to(host.device)
            ref_p_mix_dev = ref_lp_mix_dev.exp()
            with adapter.activate(
                g_em=1, m_acc_mask=mix_mask, carrier=active_carrier, t_div=0, task_id=0, apply_c4_shift=False
            ):
                test_raw_act = _forward_last_token_logits(host, mix_emb)
            test_lp_act = F.log_softmax(test_raw_act, dim=-1, dtype=torch.float64)
            kl_a = float(torch.sum(ref_p_mix_dev * (ref_lp_mix_dev - test_lp_act), dim=-1).clamp_min(0.0).mean().item())
            act_kls.append(kl_a)

    act_mean = float(np.mean(act_kls))
    act_max = float(np.max(act_kls))
    base_audit["active_carrier_mix_kl_mean"] = act_mean
    base_audit["active_carrier_mix_kl_max"] = act_max
    base_audit["passes_active_carrier_mix_kl_1e3"] = bool(act_mean <= 1e-3)
    return base_audit


def _evaluate_cn_sft_dev(
    host: QwenCacheHost,
    connector: Stage4MatchedConnector,
    dv_groups: List[dict],
    adapter: CNSFTHostAdapter,
    scorer_mode: str,
) -> dict:
    total_n = hits = inf_n = inf_hits = 0
    loss_sum = inf_loss_sum = phys_inf_loss = 0.0
    phys_inf_hits = 0
    q1_n = q1_hits = q2v_n = q2v_hits = q2f_n = q2f_hits = 0
    q3c_n = q3c_hits = q3i_n = q3i_hits = q3a_n = q3a_hits = 0
    q1_loss = q2v_loss = q2f_loss = q3c_loss = q3a_loss = 0.0

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

            lp, _, _, _ = score_batch_with_cn_sft(
                connector,
                host,
                g["prompt"],
                g["choices"],
                g["qtype"],
                words,
                4,
                ps,
                meta,
                adapter=adapter,
                ref_summary=rs,
                scorer_mode=scorer_mode,
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
        "q3_indep_acc": q3i_hits / max(1, q3i_n),
        "q3_after_pilot_acc": q3a_hits / max(1, q3a_n),
        "q3_after_pilot_nll": q3a_loss / max(1, q3a_n),
    }


def _predict_cn_sft_on_suites(
    host: QwenCacheHost,
    connector: Stage4MatchedConnector,
    suites: Dict[str, List[dict]],
    adapter: CNSFTHostAdapter,
    eval_bs: int,
    scorer_mode: str,
) -> Dict[str, List[np.ndarray]]:
    preds: Dict[str, List[np.ndarray]] = {}
    connector.eval()
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
                            g["discrete_d"][:, None],
                        ],
                        axis=-1,
                    )
                )
                _, first_idx, inv_idx = np.unique(key_mat, axis=0, return_index=True, return_inverse=True)
                uniq_logits_chunks: List[torch.Tensor] = []
                n_u = len(first_idx)
                for st in range(0, n_u, eval_bs):
                    idx_u = first_idx[st : min(n_u, st + eval_bs)]
                    words = torch.from_numpy(g["words"][idx_u]).to(host.device)
                    ps = torch.from_numpy(g["phase_slots"][idx_u]).to(host.device)
                    meta = torch.from_numpy(g["meta_scalars"][idx_u]).to(host.device)
                    rs = torch.from_numpy(g["ref_summary"][idx_u]).to(host.device)
                    dd = torch.from_numpy(g["discrete_d"][idx_u]).to(host.device)
                    _, centered_logits, _, _ = score_batch_with_cn_sft(
                        connector,
                        host,
                        g["prompt"],
                        g["choices"],
                        g["qtype"],
                        words,
                        4,
                        ps,
                        meta,
                        adapter=adapter,
                        ref_summary=rs,
                        scorer_mode=scorer_mode,
                        discrete_d=dd,
                    )
                    uniq_logits_chunks.append(centered_logits.detach().to(device="cpu", dtype=torch.float32))
                uniq_logits = torch.cat(uniq_logits_chunks, dim=0)
                all_logits = uniq_logits[torch.from_numpy(inv_idx).to(dtype=torch.long)]
                s_preds.append(_reconstruct_probs_from_centered_logits(host, connector, g, all_logits))
            preds[sname] = s_preds
    return preds


def _strip_summary(raw_sum: dict) -> dict:
    return {k: {sk: sv for sk, sv in v.items() if not sk.startswith("_")} for k, v in raw_sum.items()}


def _train_cn_sft_arm(
    host: QwenCacheHost,
    connector: Stage4MatchedConnector,
    adapter: CNSFTHostAdapter,
    tr_groups: List[dict],
    tr_minibatches: List[Tuple[int, int, int]],
    order: List[int],
    train_mix_refs: List[Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
    micro_batch_size: int,
    lr: float,
    scorer_mode: str,
    seed: int,
    tag: str,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    active_carrier = None
    if adapter.regime == 2:
        active_carrier = build_regime2_separate_carrier(
            connector.prior_canon,
            codes=torch.tensor([[2, 3]], device=host.device, dtype=torch.long),
            task="Q1",
            phase_slots=torch.tensor([[0.80, 0.60, 0.50, -0.50, 0.90, 0.10]], device=host.device, dtype=torch.float32),
            meta_scalars=torch.tensor([[2.0, 2.0, 4.0, 0.25, 0.10, 2.0, 0.0]], device=host.device, dtype=torch.float32),
            bits=4,
        )

    prefix_kv_params: List[nn.Parameter] = []
    other_params: List[nn.Parameter] = []
    for name, p in adapter.named_parameters():
        if not p.requires_grad:
            continue
        if any(tag in name for tag in ("_k_proj.", "_v_proj.", "_in_proj_kv.", "_in_proj_b.")):
            prefix_kv_params.append(p)
        else:
            other_params.append(p)
    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": lr})
    if prefix_kv_params:
        param_groups.append({"params": prefix_kv_params, "lr": lr * 0.12})
    params = other_params + prefix_kv_params
    opt = torch.optim.AdamW(param_groups, lr=lr, weight_decay=1e-4)
    steps = len(order)
    sched = _make_lr_scheduler(opt, steps)

    epsilon = 5e-5
    rho = 3000.0
    eta_mu = 1000.0
    mu_mix = 15.0

    t0 = time.perf_counter()
    if host.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(host.device)

    adapter.train()
    last_losses: List[float] = []
    last_mix_kls: List[float] = []
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
            w_frac = float(m_ed - m_st) / float(mb_total)
            words = torch.from_numpy(g["words"][m_st:m_ed]).to(host.device)
            ps = torch.from_numpy(g["phase_slots"][m_st:m_ed]).to(host.device)
            meta = torch.from_numpy(g["meta_scalars"][m_st:m_ed]).to(host.device)
            rs = torch.from_numpy(g["ref_summary"][m_st:m_ed]).to(host.device)
            dd = torch.from_numpy(g["discrete_d"][m_st:m_ed]).to(host.device)
            target_p = torch.from_numpy(g["phys_probs"][m_st:m_ed]).to(host.device)

            lp, _, _, _ = score_batch_with_cn_sft(
                connector,
                host,
                g["prompt"],
                g["choices"],
                g["qtype"],
                words,
                4,
                ps,
                meta,
                adapter=adapter,
                ref_summary=rs,
                scorer_mode=scorer_mode,
                discrete_d=dd,
            )
            losses_f = rao_blackwell_bayes_kl_cf_loss(
                lp, target_p, cf_swapped_logits=None, lambda_cf=0.0, lambda_norm=0.02
            )
            factual_loss = losses_f["total"] * w_frac
            if use_cf:
                centered_f = lp - lp.mean(dim=-1, keepdim=True)
                fact_log_q_det = F.log_softmax(centered_f, dim=-1).detach()
                factual_loss.backward()
                step_loss_val += float(factual_loss.item())
                del lp, losses_f, centered_f, factual_loss

                swap_ps = torch.from_numpy(g["swap_slots"][m_st:m_ed]).to(host.device)
                cf_lp, _, _, _ = score_batch_with_cn_sft(
                    connector,
                    host,
                    g["prompt"],
                    g["choices"],
                    g["qtype"],
                    words,
                    4,
                    swap_ps,
                    meta,
                    adapter=adapter,
                    ref_summary=rs,
                    scorer_mode=scorer_mode,
                    discrete_d=dd,
                )
                swap_log_q = F.log_softmax(cf_lp - cf_lp.mean(dim=-1, keepdim=True), dim=-1)
                sym_kl = 0.5 * torch.sum((fact_log_q_det.exp() - swap_log_q.exp()) * (fact_log_q_det - swap_log_q), dim=-1)
                cf_loss = (0.15 * F.relu(0.25 - sym_kl).mean()) * w_frac
                cf_loss.backward()
                step_loss_val += float(cf_loss.item())
                del cf_lp, swap_log_q, sym_kl, cf_loss, fact_log_q_det
            else:
                factual_loss.backward()
                step_loss_val += float(factual_loss.item())
                del lp, losses_f, factual_loss

        if step_i % 2 == 0:
            p_idx = (step_i // 2) % len(GENERAL_TEXT_TRAIN_PROMPTS)
            use_active_c = ((step_i // 2) % 2 == 1) and (active_carrier is not None)
            mix_emb, mix_mask = _build_fixed_mix_prefix(host, connector, GENERAL_TEXT_TRAIN_PROMPTS[p_idx])
            ref_lp_mix, _, _ = train_mix_refs[p_idx]
            ref_lp_mix_dev = ref_lp_mix.to(host.device)
            ref_p_mix_dev = ref_lp_mix_dev.exp()
            with adapter.activate(
                g_em=1,
                m_acc_mask=mix_mask,
                carrier=(active_carrier if use_active_c else None),
                t_div=0,
                task_id=0,
                apply_c4_shift=False,
            ):
                mix_logits = _forward_last_token_logits(host, mix_emb)
            q_lp_mix = F.log_softmax(mix_logits, dim=-1, dtype=torch.float64)
            mix_kl = torch.sum(ref_p_mix_dev * (ref_lp_mix_dev - q_lp_mix), dim=-1).mean()
            viol = F.relu(mix_kl.float() - epsilon)
            aug_loss = mu_mix * viol + 0.5 * rho * viol.square() + 1e-3 * adapter.tier2_kfac_penalty()
            aug_loss.backward()
            mix_kl_val = float(mix_kl.item())
            step_loss_val += float(aug_loss.item())
            last_mix_kls.append(mix_kl_val)
            with torch.no_grad():
                mu_mix = min(100.0, max(0.0, mu_mix + eta_mu * (mix_kl_val - epsilon)))
            del mix_logits, q_lp_mix, mix_kl, viol, aug_loss, mix_emb, mix_mask

        if step_i == 0:
            assert all(p.grad is None for p in connector.parameters()), "Mode-I must block connector grads!"
            assert all(p.grad is None for p in host.lm.parameters()), "Host weights must remain frozen!"

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        last_losses.append(step_loss_val)
        if (step_i + 1) % 16 == 0 or (step_i + 1) == steps:
            print(
                f"[{tag} step {step_i + 1}/{steps}] loss={step_loss_val:.4f} "
                f"mu_mix={mu_mix:.2f} elapsed={time.perf_counter() - t0:.1f}s",
                flush=True,
            )

    train_wall = round(time.perf_counter() - t0, 2)
    peak_mb = round(torch.cuda.max_memory_allocated(host.device) / (1024.0 ** 2), 1) if host.device.type == "cuda" else 0.0
    del opt, sched
    return {
        "train_wall_s": train_wall,
        "peak_vram_mb": peak_mb,
        "final_train_loss": float(np.mean(last_losses[-min(8, len(last_losses)) :])),
        "final_train_mix_kl": float(last_mix_kls[-1]) if last_mix_kls else 0.0,
        "final_mu_mix": round(float(mu_mix), 4),
    }


def run_cn_sft_4b_probe(
    repo_root: Path,
    out_dir: Path,
    ckpt_dir: Path,
    seed: int = 202611,
    steps: int = 96,
    batch_size: int = 32,
    micro_batch_size: int = 16,
    lr: float = 1.2e-4,
    g1_blocks: int = 131,
    eval_bs: int = 64,
    scorer_mode: str = "kv_cached",
    model_id: str = MODEL_ID_DEFAULT,
    revision: str = PINNED_REVISION_DEFAULT,
    mock_tiny_for_test: bool = False,
    host: Optional[QwenCacheHost] = None,
    on_step0_done: Optional[Callable[[], None]] = None,
) -> dict:
    """Execute Step 0 + U0 Zero-Shot + U1 Unified CN-SFT + U3 Gate-Off + U2 No-SwiGLU Control."""
    t_total = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    blind_cache_dir = out_dir / "blind_cache"
    blind_cache_dir.mkdir(parents=True, exist_ok=True)

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
    mlp_l = [i for i in QWEN4B_MLP_TUNE_LAYERS if i < n_l] or ([0] if n_l >= 1 else [])
    stop_l = min(QWEN4B_STOP_LAYER, max(0, n_l - 2)) if n_l >= 2 else 0

    # --- Step 0: Pre-Flight Checks, Dual K-FAC, B_align^(gamma_f), 3-Way Bus, L10 Probe, Fisher Gate ---
    norm_errs = {"final_norm": verify_rmsnorm_gamma(_get_final_norm(host.lm))}
    for tidx in range(max(0, n_l - 3), n_l):
        norm_errs[f"layer_{tidx}_post_attn"] = verify_rmsnorm_gamma(layers[tidx].post_attention_layernorm)

    kfac_cache_p = out_dir / f"kfac_{'mock' if mock_tiny_for_test else '4b'}_dual_64p_v3.pt"
    if kfac_cache_p.exists() and not mock_tiny_for_test:
        kfac = torch.load(kfac_cache_p, map_location="cpu", weights_only=False)
        if int(kfac.get("d_model", -1)) != host.hidden_size:
            kfac = None
    else:
        kfac = None
    if kfac is None:
        prompts_kfac = GENERAL_KFAC_64_PROMPTS[:16] if mock_tiny_for_test else GENERAL_KFAC_64_PROMPTS
        kfac = collect_dual_space_kfac_projectors(
            host, fa_l, dn_l, mlp_l, prompts=prompts_kfac, stop_layer=stop_l
        )
        if not mock_tiny_for_test:
            torch.save(kfac, kfac_cache_p)
            if on_step0_done is not None:
                on_step0_done()

    # Assert head-space u_c_o covers all FA + DN layers
    for i in (*fa_l, *dn_l):
        assert i in kfac["u_c_o"], f"Missing u_c_o on layer {i}"

    b_align_info = calibrate_b_align_gamma_f(host)
    extra_parts: List[torch.Tensor] = []
    if host.hidden_size >= 512:
        extra_parts.append(connector.out_proj.weight.detach().float().cpu())
        u_tail = kfac.get("u_c_tail")
        if u_tail is not None and u_tail.numel() > 0:
            extra_parts.append(_realify_uc(u_tail, host.hidden_size).float())
    extra_blk = torch.cat(extra_parts, dim=1) if extra_parts else None
    e_bus, bus_stats = build_three_way_orthogonal_bus(
        kfac["u_c_bus"], b_align_info["m_eff"], bus_dim=22, seed=seed, extra_block=extra_blk
    )

    tr_blocks = 1 if mock_tiny_for_test else 8
    dv_blocks = 1 if mock_tiny_for_test else 2
    other_blocks = 1 if mock_tiny_for_test else 8
    tr_groups, dv_groups = build_stage4_pilot_groups(encoder, kernel_mod, train_blocks=tr_blocks, dev_blocks=dv_blocks)
    tr_minibatches, order = _weighted_phaseB_minibatch_order(tr_groups, batch_size=batch_size, steps=steps, seed=seed)

    # Precompute hookless frozen general-text references BEFORE attaching adapters
    eval_frozen_refs = compute_frozen_general_refs(host, connector, GENERAL_TEXT_EVAL_PROMPTS)
    train_mix_refs = compute_frozen_general_refs(host, connector, GENERAL_TEXT_TRAIN_PROMPTS)["mix"]

    regime, l10_proj, l10_bias, l10_rep = run_l10_ood_theta_probe(
        host, connector, stop_layer=stop_l, seed=seed, mock_tiny_for_test=mock_tiny_for_test
    )
    qgate_w, qgate_b, qgate_diag = calibrate_fisher_qgate(
        host, connector, kfac, stop_layer=stop_l, tr_groups=tr_groups, seed=seed
    )

    suites = _build_blind_suites(encoder, g1_blocks=g1_blocks, other_blocks=other_blocks)
    step0_info = {
        "norm_verify_max_err": norm_errs,
        "k_eff_stats": kfac["k_eff_stats"],
        "tail_rms_medians": {str(k): round(v, 5) for k, v in kfac["tail_rms_medians"].items()},
        "b_align_old_tilt_err": round(b_align_info["old_tilt_err"], 5),
        "b_align_fixed_err": round(b_align_info["fixed_err"], 6),
        "bus_stats": bus_stats,
        "l10_ood_probe": l10_rep,
        "fisher_qgate_calib": qgate_diag,
    }

    print(
        "[CN-SFT Step 0] "
        + json.dumps(
            {
                "regime": regime,
                "l10_ood_probe": l10_rep,
                "fisher_qgate_calib": qgate_diag,
                "b_align_old_tilt_err": step0_info["b_align_old_tilt_err"],
                "b_align_fixed_err": step0_info["b_align_fixed_err"],
                "tail_rms_medians": step0_info["tail_rms_medians"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    def _build_adapter(arm_mode: str) -> CNSFTHostAdapter:
        ad = CNSFTHostAdapter(
            host.lm,
            u_c_d=kfac["u_c_d"],
            u_c_o=kfac["u_c_o"],
            u_c_mlp=kfac["u_c_mlp"],
            u_c_bus=kfac["u_c_bus"],
            b_align_info=b_align_info,
            e_bus=e_bus,
            tail_rms_medians=kfac["tail_rms_medians"],
            fa_layers=fa_l,
            dn_layers=dn_l,
            mlp_tune_layers=mlp_l,
            stop_layer=stop_l,
            arm_mode=arm_mode,
            regime=regime,
            l10_ridge_proj=l10_proj,
            l10_ridge_bias=l10_bias,
        ).to(host.device)
        with torch.no_grad():
            ad.schur_readout.qgate_w.copy_(qgate_w.to(host.device))
            ad.schur_readout.qgate_b.copy_(qgate_b.to(host.device))
        assert len({t.device for t in (*ad.parameters(), *ad.buffers())}) == 1, "Adapter device mismatch!"
        return ad

    # --- Arm U0: Zero-Shot Compiled N1 Circuit A + ShadowGDN (0 gradient steps) ---
    u0 = _build_adapter("u0_zeroshot")
    try:
        u0_suf_calib = calibrate_template_suffix_bias(host, connector, u0, tr_groups)
        suffix_bias_tbl = u0.circuit_a.template_suffix_canon_bias.detach().cpu().clone()
        u0_dev = _evaluate_cn_sft_dev(host, connector, dv_groups, u0, scorer_mode=scorer_mode)
        u0_audit = audit_cn_sft_pure_and_mix_kl(
            host, connector, u0, GENERAL_TEXT_EVAL_PROMPTS, frozen_refs=eval_frozen_refs
        )
        u0_preds = _predict_cn_sft_on_suites(host, connector, suites, u0, eval_bs=eval_bs, scorer_mode=scorer_mode)
        u0_raw_blind = _summarize_model_blind(suites, u0_preds)
        u0_blind = _strip_summary(u0_raw_blind)
        u0_summary = {
            "arm": "u0_zeroshot",
            "suffix_calib": u0_suf_calib,
            "dev_metrics": u0_dev,
            "audit": u0_audit,
            "blind": u0_blind,
        }
        print(
            f"[CN-SFT U0 Zero-Shot] G1={u0_blind['G1_main_extrap_both']['acc']:.4f}/{u0_blind['G1_main_extrap_both']['nll']:.4f} "
            f"G3={u0_blind['G3_extrap_angle']['acc']:.4f} PureKL={u0_audit['pure_kl_max']:.2e} "
            f"MixKL={u0_audit['mix_kl_mean']:.2e} ActMixKL={u0_audit['active_carrier_mix_kl_mean']:.2e}",
            flush=True,
        )
    finally:
        u0.remove()
        del u0
        gc.collect()
        if host.device.type == "cuda":
            torch.cuda.empty_cache()

    # Check U0 Hard Early-Abort before spending compute on U1/U2
    early_abort_reasons: List[str] = []
    if not mock_tiny_for_test:
        if u0_blind["G1_main_extrap_both"]["acc"] < 0.45:
            early_abort_reasons.append(f"U0 G1_acc={u0_blind['G1_main_extrap_both']['acc']:.4f} < 0.45")
        if u0_audit["pure_max_logit_diff"] > 0.0:
            early_abort_reasons.append(f"U0 pure_max_logit_diff={u0_audit['pure_max_logit_diff']} > 0.0")
        if u0_audit["mix_kl_mean"] > 1e-3 or u0_audit["active_carrier_mix_kl_mean"] > 1e-3:
            early_abort_reasons.append(
                f"U0 mix_kl={u0_audit['mix_kl_mean']:.3e} / active={u0_audit['active_carrier_mix_kl_mean']:.3e} > 1e-3"
            )

    if early_abort_reasons:
        rep_abort = {
            "seed": seed,
            "model_id": host.model_id,
            "host_sha256": host.host_sha256,
            "base_phaseA_checkpoint": ckpt_used,
            "regime": regime,
            "step0": step0_info,
            "U0": u0_summary,
            "early_aborted": True,
            "early_abort_reasons": early_abort_reasons,
            "wall_s": round(time.perf_counter() - t_total, 2),
        }
        (out_dir / "cn_sft_4b_probe_results.json").write_text(
            json.dumps(rep_abort, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return rep_abort

    # --- Arm U1: Full Unified CN-SFT (+ U3 Gate-Off Audit before tearing down) ---
    u1 = _build_adapter("u1_full_cnsft")
    u1_ckpt_p = ckpt_dir / f"stage4_cn_sft_u1_s{seed}.pt"
    u1_summary: Dict[str, Any]
    u3_audit: Dict[str, Any]
    try:
        with torch.no_grad():
            if u1.circuit_a is not None:
                u1.circuit_a.template_suffix_canon_bias.copy_(suffix_bias_tbl.to(host.device))

        u1_params = u1.param_count()
        u1_bufs = u1.buffer_count()
        n1_dense = u1.circuit_a.n1_parameter_count_dense_equiv(host.hidden_size) if u1.circuit_a is not None else 0
        cum_trainable = connector_total_params + u1_params
        cum_with_n1_dense = cum_trainable + n1_dense
        if host.hidden_size >= 2048 and not mock_tiny_for_test:
            assert cum_trainable <= 1_850_000, f"4B cumulative trainable {cum_trainable} > 1,850,000"
            assert cum_with_n1_dense <= TIER_P1_MAX_PARAMS, f"4B dense-equiv {cum_with_n1_dense} > {TIER_P1_MAX_PARAMS}"

        u1_train_stats = _train_cn_sft_arm(
            host,
            connector,
            u1,
            tr_groups,
            tr_minibatches,
            order,
            train_mix_refs,
            micro_batch_size=micro_batch_size,
            lr=lr,
            scorer_mode=scorer_mode,
            seed=seed,
            tag=f"CN-SFT-U1-{steps}",
        )
        u1_dev = _evaluate_cn_sft_dev(host, connector, dv_groups, u1, scorer_mode=scorer_mode)
        u1_audit = audit_cn_sft_pure_and_mix_kl(
            host, connector, u1, GENERAL_TEXT_EVAL_PROMPTS, frozen_refs=eval_frozen_refs
        )
        sha_after_u1 = _compute_host_fingerprint(host.lm)
        assert sha_after_u1 == host.host_sha256, "Host SHA mutated during CN-SFT U1!"

        u1_preds = _predict_cn_sft_on_suites(host, connector, suites, u1, eval_bs=eval_bs, scorer_mode=scorer_mode)
        u1_raw_blind = _summarize_model_blind(suites, u1_preds)
        u1_blind = _strip_summary(u1_raw_blind)

        tmp_pred_p = blind_cache_dir / ".cn_sft_u1.pt.tmp"
        torch.save(
            {
                "g1_blocks": int(g1_blocks),
                "other_blocks": int(other_blocks),
                "seed": int(seed),
                "host_sha256": str(host.host_sha256),
                "preds": u1_preds,
            },
            tmp_pred_p,
        )
        tmp_pred_p.replace(blind_cache_dir / "cn_sft_u1.pt")

        torch.save(
            {
                "arm": "u1_full_cnsft",
                "seed": seed,
                "steps": steps,
                "regime": regime,
                "fa_layers": list(fa_l),
                "dn_layers": list(dn_l),
                "mlp_layers": list(mlp_l),
                "stop_layer": stop_l,
                "bus_stats": bus_stats,
                "e_bus": e_bus.cpu(),
                "b_align_info": {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in b_align_info.items()},
                "tail_rms_medians": kfac["tail_rms_medians"],
                "l10_ridge_proj": l10_proj.cpu(),
                "l10_ridge_bias": l10_bias.cpu(),
                "state_dict": {k: v.cpu() for k, v in u1.state_dict().items()},
            },
            u1_ckpt_p,
        )

        # U3 Gate-Off Audit on trained U1 weights
        u1.gate_off_audit = True
        try:
            u3_audit = audit_cn_sft_pure_and_mix_kl(
                host, connector, u1, GENERAL_TEXT_EVAL_PROMPTS, frozen_refs=eval_frozen_refs
            )
        finally:
            u1.gate_off_audit = False

        u1_summary = {
            "arm": "u1_full_cnsft",
            "steps": steps,
            **u1_train_stats,
            "connector_params": connector_total_params,
            "adapter_trainable_params": u1_params,
            "adapter_buffer_elems": u1_bufs,
            "n1_dense_equiv_params": n1_dense,
            "cumulative_trainable": cum_trainable,
            "cumulative_with_n1_dense": cum_with_n1_dense,
            "tier_p1_cap": TIER_P1_MAX_PARAMS,
            "under_1_85m_trainable": bool(cum_trainable <= 1_850_000) if host.hidden_size >= 2048 else True,
            "under_tier_p1_with_dense": bool(cum_with_n1_dense <= TIER_P1_MAX_PARAMS) if host.hidden_size >= 2048 else True,
            "dev_metrics": u1_dev,
            "audit": u1_audit,
            "blind": u1_blind,
            "host_sha256": sha_after_u1,
            "checkpoint": u1_ckpt_p.name,
            "checkpoint_sha256": hashlib.sha256(u1_ckpt_p.read_bytes()).hexdigest(),
        }
        print(
            f"[CN-SFT U1 Trained] G1={u1_blind['G1_main_extrap_both']['acc']:.4f}/{u1_blind['G1_main_extrap_both']['nll']:.4f} "
            f"G3={u1_blind['G3_extrap_angle']['acc']:.4f} Q2_gain={u1_blind['Q2_C4_blind']['fresh_minus_valid_acc']:+.4f} "
            f"PureKL={u1_audit['pure_kl_max']:.2e} MixKL={u1_audit['mix_kl_mean']:.2e}/{u1_audit['mix_kl_max']:.2e} "
            f"ActMixKL={u1_audit['active_carrier_mix_kl_mean']:.2e} U3_MixKL={u3_audit['mix_kl_mean']:.2e}",
            flush=True,
        )
    finally:
        u1.remove()
        del u1
        gc.collect()
        if host.device.type == "cuda":
            torch.cuda.empty_cache()

    # --- M2 Paired Cluster Statistics + IUT vs cached Blind Baselines & U0 ---
    comp_raw: Dict[str, dict] = {
        "u1_full_cnsft": u1_raw_blind,
        "u0_zeroshot": u0_raw_blind,
        "direct_physical": _summarize_model_blind(
            suites, {k: [g["phys_probs"].astype(np.float64) for g in grps] for k, grps in suites.items()}
        ),
    }
    for base_name in ("equivariant_host_pure", "mode1_surgical_96", "lemmo_tier_p1"):
        cp = blind_cache_dir / f"{base_name}.pt"
        if cp.exists():
            ck = torch.load(cp, map_location="cpu", weights_only=False)
            if (
                isinstance(ck, dict)
                and int(ck.get("g1_blocks", -1)) == int(g1_blocks)
                and int(ck.get("other_blocks", -1)) == int(other_blocks)
                and str(ck.get("host_sha256", "")) == str(host.host_sha256)
                and "preds" in ck
            ):
                comp_raw[base_name] = _summarize_model_blind(suites, ck["preds"])

    def _pair_g1(m_a: str, m_b: str) -> Optional[dict]:
        if m_a not in comp_raw or m_b not in comp_raw:
            return None
        cl = comp_raw[m_a]["G1_main_extrap_both"]["_clusters"]
        st_acc = _cluster_paired_stats(
            comp_raw[m_a]["G1_main_extrap_both"]["_hits"],
            comp_raw[m_b]["G1_main_extrap_both"]["_hits"],
            cl,
            higher_is_better_a=True,
            seed=seed,
        )
        st_nll = _cluster_paired_stats(
            comp_raw[m_a]["G1_main_extrap_both"]["_nlls"],
            comp_raw[m_b]["G1_main_extrap_both"]["_nlls"],
            cl,
            higher_is_better_a=False,
            seed=seed + 1,
        )
        return {
            "acc_diff_a_minus_b": st_acc,
            "nll_diff_a_minus_b": st_nll,
            "iut_one_sided_p": float(max(st_acc["p_one_sided_improved"], st_nll["p_one_sided_improved"])),
        }

    vs_lemmo = _pair_g1("u1_full_cnsft", "lemmo_tier_p1")
    vs_pure = _pair_g1("u1_full_cnsft", "equivariant_host_pure")
    vs_surg96 = _pair_g1("u1_full_cnsft", "mode1_surgical_96")
    vs_u0 = _pair_g1("u1_full_cnsft", "u0_zeroshot")
    vs_phys = _pair_g1("u1_full_cnsft", "direct_physical")

    u1_q2d = u1_raw_blind["Q2_C4_blind"]
    q2_bw_stat = _cluster_paired_stats(
        u1_q2d["_fresh_hits"], u1_q2d["_valid_hits"], u1_q2d["_clusters"], higher_is_better_a=True, seed=seed + 3
    )
    raw_fam_p = {
        k: v["iut_one_sided_p"]
        for k, v in (
            ("H1_u1_beats_lemmo_G1_IUT", vs_lemmo),
            ("H2_u1_beats_pure_G1_IUT", vs_pure),
            ("H3_u1_beats_surg96_G1_IUT", vs_surg96),
        )
        if v is not None
    }
    raw_fam_p["H4_u1_Q2_Blackwell_gain"] = q2_bw_stat["p_one_sided_improved"]

    m2_paired = {
        "G1_u1_vs_lemmo_tier_p1": vs_lemmo,
        "G1_u1_vs_equivariant_host_pure": vs_pure,
        "G1_u1_vs_mode1_surgical_96": vs_surg96,
        "G1_u1_vs_u0_zeroshot": vs_u0,
        "G1_u1_vs_direct_physical": vs_phys,
        "U1_Q2_C4_fresh_minus_valid": q2_bw_stat,
        "holm_bonferroni_family": {"raw_p": raw_fam_p, "adjusted_p": _holm_bonferroni(raw_fam_p)},
    }

    # --- Arm U2: No-SwiGLU Bug-Fixed Control (run when U1 clears G1 & MixKL, or in mock test) ---
    u2_summary: Dict[str, Any] = {"skipped": True}
    run_u2 = mock_tiny_for_test or (
        u1_blind["G1_main_extrap_both"]["acc"] >= 0.48 and u1_audit["mix_kl_mean"] <= 1e-3
    )
    if run_u2:
        u2_steps = min(steps, 64)
        _, u2_order = _weighted_phaseB_minibatch_order(tr_groups, batch_size=batch_size, steps=u2_steps, seed=seed + 7)
        u2 = _build_adapter("u2_noswiglu_control")
        try:
            u2_train_stats = _train_cn_sft_arm(
                host,
                connector,
                u2,
                tr_groups,
                tr_minibatches,
                u2_order,
                train_mix_refs,
                micro_batch_size=micro_batch_size,
                lr=lr,
                scorer_mode=scorer_mode,
                seed=seed,
                tag=f"CN-SFT-U2-{u2_steps}",
            )
            u2_dev = _evaluate_cn_sft_dev(host, connector, dv_groups, u2, scorer_mode=scorer_mode)
            u2_audit = audit_cn_sft_pure_and_mix_kl(
                host, connector, u2, GENERAL_TEXT_EVAL_PROMPTS, frozen_refs=eval_frozen_refs
            )
            u2_g1_b = min(g1_blocks, 32)
            u2_suites = _build_blind_suites(encoder, g1_blocks=u2_g1_b, other_blocks=other_blocks)
            u2_preds = _predict_cn_sft_on_suites(host, connector, u2_suites, u2, eval_bs=eval_bs, scorer_mode=scorer_mode)
            u2_blind = _strip_summary(_summarize_model_blind(u2_suites, u2_preds))
            u2_summary = {
                "skipped": False,
                "arm": "u2_noswiglu_control",
                "steps": u2_steps,
                "g1_blocks": int(u2_g1_b),
                **u2_train_stats,
                "adapter_trainable_params": u2.param_count(),
                "dev_metrics": u2_dev,
                "audit": u2_audit,
                "blind": u2_blind,
            }
            print(
                f"[CN-SFT U2 No-SwiGLU] G1={u2_blind['G1_main_extrap_both']['acc']:.4f}/{u2_blind['G1_main_extrap_both']['nll']:.4f} "
                f"PureKL={u2_audit['pure_kl_max']:.2e} MixKL={u2_audit['mix_kl_mean']:.2e}",
                flush=True,
            )
        finally:
            u2.remove()
            del u2
            gc.collect()
            if host.device.type == "cuda":
                torch.cuda.empty_cache()

    hard_pass_gates = {
        "pure_kl_zero": bool(u1_audit["passes_pure_kl_zero"]),
        "mix_kl_mean_le_1e3": bool(u1_audit["mix_kl_mean"] <= 1e-3),
        "mix_kl_max_le_3e3": bool(u1_audit["mix_kl_max"] <= 3e-3),
        "active_carrier_mix_kl_mean_le_1e3": bool(u1_audit["active_carrier_mix_kl_mean"] <= 1e-3),
        "active_carrier_mix_kl_max_le_3e3": bool(u1_audit["active_carrier_mix_kl_max"] <= 3e-3),
        "blind_g1_acc_ge_48": bool(u1_blind["G1_main_extrap_both"]["acc"] >= 0.48),
        "blind_g1_nll_le_1_15": bool(u1_blind["G1_main_extrap_both"]["nll"] <= 1.15),
        "blind_g3_acc_ge_45": bool(u1_blind["G3_extrap_angle"]["acc"] >= 0.45),
        "q2_blackwell_ge_5pp": bool(u1_blind["Q2_C4_blind"]["fresh_minus_valid_acc"] >= 0.05),
        "iut_vs_lemmo_p_lt_1e4": bool(vs_lemmo["iut_one_sided_p"] < 1e-4) if vs_lemmo is not None else None,
        "vram_le_65gib": bool(u1_summary["peak_vram_mb"] <= 65.0 * 1024.0),
        "params_under_1_85m_and_tier_p1": bool(
            u1_summary["under_1_85m_trainable"] and u1_summary["under_tier_p1_with_dense"]
        ),
    }
    if not mock_tiny_for_test:
        hard_pass_gates["all_passed"] = all(bool(v) for v in hard_pass_gates.values() if v is not None)

    report = {
        "seed": seed,
        "model_id": host.model_id,
        "host_sha256": host.host_sha256,
        "base_phaseA_checkpoint": ckpt_used,
        "regime": regime,
        "early_aborted": False,
        "wall_s": round(time.perf_counter() - t_total, 2),
        "step0": step0_info,
        "U0": u0_summary,
        "U1": u1_summary,
        "U3_gate_off_audit": u3_audit,
        "U2": u2_summary,
        "m2_paired_stats": m2_paired,
        "hard_pass_gates": hard_pass_gates,
    }
    (out_dir / "cn_sft_4b_probe_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
