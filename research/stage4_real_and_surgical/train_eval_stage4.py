"""Stage 4 GPU smoke/K-FAC/Jacobian probe, Gate-1 69-step 4-arm pilot, and Two-Probe VoI readout."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.stage3_ministral_cache import analytic as s3_analytic
from research.stage3_ministral_cache.data import MotherBatch, generate_mothers
from research.stage3_ministral_cache.ministral_connector import _compute_host_fingerprint
from research.stage3_ministral_cache.schema import ROOTS, ROTATIONS, matched_sum, sector_of_complex
from research.stage3d_qwen_signed_phase.data_stage3d import (
    DEV_OMEGAS,
    NS_CFO_DEV,
    NS_CFO_TRAIN,
    REPLAY_TRAIN_CONDITIONS,
    TRAIN_ARMS,
    TRAIN_OMEGAS,
    TRAIN_TIMINGS,
    build_stage3d_prompt,
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
    ARM_NAMES_STAGE4,
    CANONICAL_TASK_NAME,
    TASK_TO_ID,
    PhysicalKernelS17,
    Stage4MatchedConnector,
    extract_canonical_ref_summary,
    lift_scores,
    unlift_probs,
)
from research.stage4_real_and_surgical.surgical_adapter import (
    QWEN4B_DN_LAYERS,
    QWEN4B_FA_LAYERS,
    QWEN4B_STOP_LAYER,
    SurgicalHostAdapter,
    _get_decoder_layers,
    compute_complex_nullspace_projector,
    param_budget_audit,
    rao_blackwell_bayes_kl_cf_loss,
)

EXPECTED_4B_HOST_SHA256 = "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d"
EXPECTED_4B_VOCAB = 248_320

GENERAL_KFAC_PROMPTS = (
    "请用一句话解释量子相位估计中的高斯过程漂移。",
    '{"protocol": "m5-c16-v1-le", "bits": [2, 2, 6, 4, 2], "status": "ok"}',
    "在连续频偏信道下，保持通用语言能力不退化的充要条件是零空间正交投影与门控隔离。",
    "Summarize the difference between coherent integration and non-coherent accumulation in QPSK demodulation.",
)


def build_stage4_pilot_groups(
    encoder,
    kernel_mod: PhysicalKernelS17,
    train_blocks: int = 8,
    dev_blocks: int = 2,
) -> Tuple[List[dict], List[dict]]:
    """Build identical matched train/dev groups containing exact analytic Bayes posterior targets."""
    del kernel_mod  # Physical posterior targets now come directly from s3_analytic.readout_from_table
    tr_groups: List[dict] = []
    dv_groups: List[dict] = []

    def append_batch(groups: List[dict], batch: MotherBatch, source_tag: str) -> None:
        cond = batch.condition
        n_m = len(batch.x1)
        p_vec = public_phase_slots(cond.omega, cond.h, cond.ell, cond.t_p, zero_arm=False)
        p_swap = public_phase_slots(-cond.omega, cond.h, cond.ell, cond.t_p, zero_arm=False)
        phase_mat = np.repeat(p_vec[None, :], n_m, axis=0)
        swap_mat = np.repeat(p_swap[None, :], n_m, axis=0)
        meta_vec = np.array(
            [cond.h, cond.ell, cond.t_p, cond.omega, cond.sigma_phi, cond.snr_db, 0.0], dtype=np.float32
        )
        meta_mat = np.repeat(meta_vec[None, :], n_m, axis=0)
        disc_vec = np.full(n_m, float(cond.d), dtype=np.float32)
        empty_ref = np.zeros((n_m, 29), dtype=np.float32)
        empty_ref[:, 0:8] = 0.25

        for rot in ROTATIONS:
            s1, s2 = rot
            v1, v2 = batch.rotated_view(*rot)
            z1_np = np.asarray(encoder.encode(v1), dtype=np.int64)
            z2_np = np.asarray(encoder.encode(v2), dtype=np.int64)
            t1 = encoder.table[z1_np]
            t2 = encoder.table[z2_np]
            words = np.stack([z1_np, z2_np], axis=-1)

            res_pre = s3_analytic.readout_from_table(t1, t2, (), cond, protocol="repeat_x1")
            res_ind = s3_analytic.readout_from_table(t1, t2, (), cond, protocol="independent_x3")

            # Q1
            p_q1, ch_q1 = build_stage3d_prompt("Q1", cond)
            groups.append(
                {
                    "source": source_tag,
                    "cond": cond.name,
                    "omega": float(cond.omega),
                    "rot": rot,
                    "qtype": "Q1",
                    "arm": "pre_ref",
                    "prompt": p_q1,
                    "choices": ch_q1,
                    "words": words,
                    "phase_slots": phase_mat,
                    "swap_slots": swap_mat,
                    "meta_scalars": meta_mat,
                    "discrete_d": disc_vec,
                    "ref_summary": empty_ref,
                    "labels": ((batch.x2 - batch.x1 + s2 - s1) % 4).astype(np.int64),
                    "phys_probs": np.asarray(res_pre["q1"], dtype=np.float32),
                }
            )

            # Q3 repeat & indep
            for proto in ("repeat_x1", "independent_x3"):
                qt = f"Q3_{proto}"
                p_q3, ch_q3 = build_stage3d_prompt("Q3", cond, protocol=proto)
                y3_rot = (batch.y3_repeat if proto == "repeat_x1" else batch.y3_indep) * ROOTS[s1 % 4]
                phys_q3 = res_pre["q3"] if proto == "repeat_x1" else res_ind["q3"]
                groups.append(
                    {
                        "source": source_tag,
                        "cond": cond.name,
                        "omega": float(cond.omega),
                        "rot": rot,
                        "qtype": qt,
                        "arm": "pre_ref",
                        "prompt": p_q3,
                        "choices": ch_q3,
                        "words": words,
                        "phase_slots": phase_mat,
                        "swap_slots": swap_mat,
                        "meta_scalars": meta_mat,
                        "discrete_d": disc_vec,
                        "ref_summary": empty_ref,
                        "labels": sector_of_complex(matched_sum(y3_rot)).astype(np.int64),
                        "phys_probs": np.asarray(phys_q3, dtype=np.float32),
                    }
                )

            # Q2 across arms
            y_q2 = ((batch.x1 + s1) % 4).astype(np.int64)
            for q2_arm in TRAIN_ARMS:
                refs = batch.references_for_arm(q2_arm, coord_rot=0)
                p_q2, ch_q2 = build_stage3d_prompt("Q2", cond, references=refs, deduplicate=True)
                ref_sum_t, _ = extract_canonical_ref_summary(
                    refs,
                    n_m,
                    torch.device("cpu"),
                    omega=float(cond.omega),
                    sigma_phi=float(cond.sigma_phi),
                    drift_type=str(cond.drift_type),
                    discrete_d=float(cond.d),
                )
                ref_sum_np = ref_sum_t.numpy()
                res_q2 = s3_analytic.readout_from_table(t1, t2, refs, cond, protocol="repeat_x1")
                groups.append(
                    {
                        "source": source_tag,
                        "cond": cond.name,
                        "omega": float(cond.omega),
                        "rot": rot,
                        "qtype": "Q2",
                        "arm": q2_arm,
                        "prompt": p_q2,
                        "choices": ch_q2,
                        "words": words,
                        "phase_slots": phase_mat,
                        "swap_slots": swap_mat,
                        "meta_scalars": meta_mat,
                        "discrete_d": disc_vec,
                        "ref_summary": ref_sum_np,
                        "labels": y_q2,
                        "phys_probs": np.asarray(res_q2["q2"], dtype=np.float32),
                    }
                )

            # Q3 after valid pilot
            refs_v = batch.references_for_arm("valid", coord_rot=0)
            p_q3a, ch_q3a = build_stage3d_prompt("Q3_after_pilot", cond, protocol="repeat_x1", references=refs_v, deduplicate=True)
            ref_sum_v, _ = extract_canonical_ref_summary(
                refs_v,
                n_m,
                torch.device("cpu"),
                omega=float(cond.omega),
                sigma_phi=float(cond.sigma_phi),
                drift_type=str(cond.drift_type),
                discrete_d=float(cond.d),
            )
            ref_sum_v_np = ref_sum_v.numpy()
            res_q3a = s3_analytic.readout_from_table(t1, t2, refs_v, cond, protocol="repeat_x1")
            y_q3_rep = sector_of_complex(matched_sum(batch.y3_repeat * ROOTS[s1 % 4])).astype(np.int64)
            groups.append(
                {
                    "source": source_tag,
                    "cond": cond.name,
                    "omega": float(cond.omega),
                    "rot": rot,
                    "qtype": "Q3_after_pilot",
                    "arm": "valid",
                    "prompt": p_q3a,
                    "choices": ch_q3a,
                    "words": words,
                    "phase_slots": phase_mat,
                    "swap_slots": swap_mat,
                    "meta_scalars": meta_mat,
                    "discrete_d": disc_vec,
                    "ref_summary": ref_sum_v_np,
                    "labels": y_q3_rep,
                    "phys_probs": np.asarray(res_q3a["q3_after_pilot"], dtype=np.float32),
                }
            )

    for cname in REPLAY_TRAIN_CONDITIONS:
        append_batch(tr_groups, generate_mothers("s3_train", cname, max_blocks=train_blocks), "replay")
        append_batch(dv_groups, generate_mothers("s3_dev", cname, max_blocks=dev_blocks), "replay")

    for t_idx, (h, ell, tp) in enumerate(TRAIN_TIMINGS):
        tr_map, _ = generate_paired_cfo_blocks(NS_CFO_TRAIN, t_idx, train_blocks, h, ell, tp, TRAIN_OMEGAS, split_prefix="cfo_train")
        for w in TRAIN_OMEGAS:
            append_batch(tr_groups, tr_map[float(w)], "cfo")
        dv_map, _ = generate_paired_cfo_blocks(NS_CFO_DEV, t_idx, dev_blocks, h, ell, tp, DEV_OMEGAS, split_prefix="cfo_dev")
        for w in DEV_OMEGAS:
            append_batch(dv_groups, dv_map[float(w)], "cfo")

    return tr_groups, dv_groups


def collect_kfac_complex_projectors(
    host: QwenCacheHost,
    layer_indices: Sequence[int],
    k_c: int,
    prompts: Sequence[str] = GENERAL_KFAC_PROMPTS,
) -> Dict[int, torch.Tensor]:
    """Run a 1-pass forward over general prompts to extract C_gen^C top-k_c eigenvectors per layer."""
    layers = _get_decoder_layers(host.lm)
    d = host.hidden_size
    covs: Dict[int, torch.Tensor] = {int(i): torch.zeros((d, d), dtype=torch.float64) for i in layer_indices if 0 <= i < len(layers)}
    handles = []

    for idx in covs:
        def make_hook(layer_id: int):
            def _h(module, args, out):
                hs = out[0] if isinstance(out, tuple) else out
                flat = hs.detach().reshape(-1, hs.shape[-1]).to(dtype=torch.float64, device="cpu")
                covs[layer_id] += flat.T @ flat
            return _h

        target_mod = getattr(layers[idx], "input_layernorm", layers[idx])
        handles.append(target_mod.register_forward_hook(make_hook(idx)))

    try:
        embed_layer = host.lm.get_input_embeddings()
        with torch.no_grad():
            for pr in prompts:
                if host.tokenizer is None:
                    ids = torch.tensor([[1, 2, 3, 4]], device=host.device, dtype=torch.long)
                else:
                    toks = host.tokenizer.encode(pr, add_special_tokens=False)
                    ids = torch.tensor([toks], device=host.device, dtype=torch.long)
                emb = embed_layer(ids)
                attn = torch.ones_like(ids)
                host.lm(inputs_embeds=emb, attention_mask=attn, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    projectors: Dict[int, torch.Tensor] = {}
    for idx, c_mat in covs.items():
        # Add small ridge so PSD decomposition is always rock-solid
        c_mat = c_mat + 1e-6 * torch.eye(d, dtype=torch.float64)
        eff_kc = min(int(k_c), (d // 2) - 1)
        res = compute_complex_nullspace_projector(c_mat, k_c=eff_kc)
        projectors[idx] = res["U_c"]
    return projectors


def measure_candidate_jacobian(
    host: QwenCacheHost,
    connector: Stage4MatchedConnector,
    group: dict,
    max_items: int = 16,
    scorer_mode: str = "kv_cached",
) -> dict:
    """Measure live candidate-subspace Jacobian singular values (sigma_1, sigma_2, sigma_3) w.r.t. soft tokens."""
    n = min(int(max_items), len(group["words"]))
    words = torch.from_numpy(group["words"][:n]).to(host.device)
    ps = torch.from_numpy(group["phase_slots"][:n]).to(host.device)
    meta = torch.from_numpy(group["meta_scalars"][:n]).to(host.device)
    rs = torch.from_numpy(group["ref_summary"][:n]).to(host.device)
    tid = torch.full((n,), TASK_TO_ID[group["qtype"]], device=host.device, dtype=torch.long)

    # Build prefix once cleanly around soft_tokens leaf
    with torch.no_grad():
        st_base, _, _, _ = connector.forward_tokens(words, 4, tid, ps, meta, ref_summary=rs)
    # Single-sample Jacobian at median item (fast & deterministic)
    s0 = st_base[:1].detach().clone().requires_grad_(True)
    pref, _ = connector.build_host_prefix_embeds(host, group["prompt"], s0)
    if host.tokenizer is None:
        raw = host.lm(inputs_embeds=pref, attention_mask=torch.ones(1, pref.shape[1], device=host.device, dtype=torch.long), use_cache=False).logits[:, -1, :4].float()
    elif scorer_mode == "kv_cached":
        _, raw = host.score_candidates_kv(pref, group["choices"], temperature=1.0)
    else:
        _, raw = host.score_candidates_batched_uncached(pref, group["choices"], temperature=1.0)
    centered = raw[0] - raw[0].mean()  # (4,)
    j_rows = []
    for k in range(4):
        g = torch.autograd.grad(centered[k], s0, retain_graph=(k < 3), allow_unused=False)[0]
        j_rows.append(g.reshape(-1).detach().cpu().to(torch.float64))
    j_mat = torch.stack(j_rows, dim=0)  # (4, 8*D)
    svals = torch.linalg.svdvals(j_mat).numpy()
    return {
        "singular_values_4": [float(v) for v in svals],
        "sigma_1": float(svals[0]),
        "sigma_2": float(svals[1]),
        "sigma_3": float(svals[2]),
        "kappa_3": float(svals[0] / max(1e-12, svals[2])),
    }


def run_stage4_smoke_and_probe(
    repo_root: Path,
    out_dir: Path,
    model_id: str = MODEL_ID_DEFAULT,
    revision: str = PINNED_REVISION_DEFAULT,
    mock_tiny_for_test: bool = False,
) -> dict:
    """Execute Gate 0 / 4B smoke audit, host immutability, PureKL=0.0, K-FAC spectrum, and Jacobian probe."""
    out_dir.mkdir(parents=True, exist_ok=True)
    host = QwenCacheHost(model_id=model_id, revision=revision, mock_tiny_for_test=mock_tiny_for_test)
    encoder, _ = load_s17_encoder(repo_root)
    kernel_mod = PhysicalKernelS17(encoder.table).to(host.device)
    connector = Stage4MatchedConnector("equivariant_host_pure", encoder.table, hidden_size=host.hidden_size).to(host.device)

    vocab_size = int(getattr(host.text_config, "vocab_size", 0))
    if not mock_tiny_for_test:
        assert vocab_size == EXPECTED_4B_VOCAB, f"Unexpected vocab={vocab_size}"
        assert host.host_sha256 == EXPECTED_4B_HOST_SHA256, f"Unexpected host sha={host.host_sha256}"

    layers = _get_decoder_layers(host.lm)
    n_l = len(layers)
    fa_l = [i for i in QWEN4B_FA_LAYERS if i < n_l] or ([n_l - 1] if n_l >= 1 else [])
    dn_l = [i for i in QWEN4B_DN_LAYERS if i < n_l]
    stop_l = min(QWEN4B_STOP_LAYER, max(0, n_l - 2))
    k_c = 256 if host.hidden_size >= 1024 else max(4, host.hidden_size // 8)
    kfac_layers = sorted(set(fa_l + dn_l))
    proj_map = collect_kfac_complex_projectors(host, kfac_layers, k_c=k_c)

    adapter = SurgicalHostAdapter(
        host.lm,
        fa_layers=fa_l,
        dn_layers=dn_l,
        stop_layer=stop_l,
        mode="mode1_stopgrad_l10",
        u_c_by_layer=proj_map,
    ).to(host.device)

    mode1_surgical_live_ok = False
    try:
        # 1. PureKL == 0.0 audit when g_EM == 0
        with torch.no_grad():
            embed = host.lm.get_input_embeddings()
            ids = torch.tensor([[1, 2, 3, 4]], device=host.device, dtype=torch.long) if host.tokenizer is None else torch.tensor([host.tokenizer.encode(GENERAL_KFAC_PROMPTS[0], add_special_tokens=False)], device=host.device, dtype=torch.long)
            emb = embed(ids)
            attn = torch.ones_like(ids)
            base_logits = host.lm(inputs_embeds=emb, attention_mask=attn, use_cache=False).logits[:, -1, :].float()
            with adapter.activate(g_em=0):
                gated_logits = host.lm(inputs_embeds=emb, attention_mask=attn, use_cache=False).logits[:, -1, :].float()
            max_diff = float((base_logits - gated_logits).abs().max().item())
            log_p_base = F.log_softmax(base_logits, dim=-1, dtype=torch.float64)
            log_p_gate = F.log_softmax(gated_logits, dim=-1, dtype=torch.float64)
            p_base = log_p_base.exp()
            pure_kl = float(torch.sum(p_base * (log_p_base - log_p_gate)).clamp_min(0.0).item())

        # 2. Sample Jacobian & connector-only check across 4 microbatches
        _, dv_groups = build_stage4_pilot_groups(encoder, kernel_mod, train_blocks=1, dev_blocks=1)
        q3_grp = next(g for g in dv_groups if g["qtype"] == "Q3_repeat_x1" and abs(g["omega"]) > 0.0)
        jac = measure_candidate_jacobian(host, connector, q3_grp, max_items=4)

        alloc_mb = []
        if host.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(host.device)
        for _ in range(4):
            words = torch.from_numpy(q3_grp["words"][:4]).to(host.device)
            ps = torch.from_numpy(q3_grp["phase_slots"][:4]).to(host.device)
            meta = torch.from_numpy(q3_grp["meta_scalars"][:4]).to(host.device)
            rs = torch.from_numpy(q3_grp["ref_summary"][:4]).to(host.device)
            dd = torch.from_numpy(q3_grp["discrete_d"][:4]).to(host.device)
            lp, _, _, _ = connector.score_batch(
                host,
                q3_grp["prompt"],
                q3_grp["choices"],
                "Q3_repeat_x1",
                words,
                4,
                ps,
                meta,
                ref_summary=rs,
                discrete_d=dd,
            )
            lbl = torch.from_numpy(q3_grp["labels"][:4]).to(host.device)
            loss = F.nll_loss(lp, lbl)
            loss.backward()
            connector.zero_grad(set_to_none=True)
            del lp, loss, lbl
            if host.device.type == "cuda":
                torch.cuda.synchronize()
                alloc_mb.append(round(torch.cuda.memory_allocated(host.device) / (1024.0 ** 2), 1))

        # 3. Live Mode-I SurgicalHostAdapter pass (Pass 1 prefix + Pass 2 KV continuation)
        connector.zero_grad(set_to_none=True)
        adapter.zero_grad(set_to_none=True)
        words_s = torch.from_numpy(q3_grp["words"][:4]).to(host.device)
        ps_s = torch.from_numpy(q3_grp["phase_slots"][:4]).to(host.device)
        meta_s = torch.from_numpy(q3_grp["meta_scalars"][:4]).to(host.device)
        rs_s = torch.from_numpy(q3_grp["ref_summary"][:4]).to(host.device)
        dd_s = torch.from_numpy(q3_grp["discrete_d"][:4]).to(host.device)
        lp_s, _, _, _ = connector.score_batch(
            host,
            q3_grp["prompt"],
            q3_grp["choices"],
            "Q3_repeat_x1",
            words_s,
            4,
            ps_s,
            meta_s,
            ref_summary=rs_s,
            surgical_adapter=adapter,
            discrete_d=dd_s,
        )
        lbl_s = torch.from_numpy(q3_grp["labels"][:4]).to(host.device)
        loss_s = F.nll_loss(lp_s, lbl_s)
        loss_s.backward()
        assert all(p.grad is None for p in connector.parameters()), "Mode-I stop_grad failed to block connector grads!"
        assert all(p.grad is None for p in host.lm.parameters()), "Frozen host params must have no grad!"
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all() and float(p.grad.abs().sum().item()) > 0.0
            for p in adapter.parameters()
        ), "Expected finite non-zero gradients on SurgicalHostAdapter parameters!"
        adapter.zero_grad(set_to_none=True)
        del lp_s, loss_s, lbl_s, words_s, ps_s, meta_s, rs_s, dd_s
        mode1_surgical_live_ok = True

        peak_vram_mb = (
            round(torch.cuda.max_memory_allocated(host.device) / (1024.0 ** 2), 1)
            if host.device.type == "cuda"
            else 0.0
        )
        budget = param_budget_audit(connector, adapter, frozen_lm=host.lm)
        assert budget["host_sha256"] == host.host_sha256, "Host SHA mismatch after Mode-I surgical probe!"
        kfac_file = out_dir / "kfac_4b_top256.pt"
        torch.save({k: v.cpu() for k, v in proj_map.items()}, kfac_file)
    finally:
        adapter.remove()

    report = {
        "model_id": host.model_id,
        "hf_sha": host.hf_sha,
        "vocab_size": vocab_size,
        "frozen_param_count": host.frozen_param_count,
        "host_sha256_before": host.host_sha256,
        "pure_kl_g_em_0": pure_kl,
        "max_logit_diff_g_em_0": max_diff,
        "mode1_surgical_live_ok": mode1_surgical_live_ok,
        "jacobian_probe": jac,
        "four_microbatch_alloc_mb": alloc_mb,
        "peak_vram_mb": peak_vram_mb,
        "budget_audit": budget,
        "kfac_file": kfac_file.name,
    }
    (out_dir / "smoke_probe_4b.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _is_informative_group(g: dict) -> bool:
    if g["qtype"] == "Q3_independent_x3":
        return False
    if g["qtype"] == "Q2" and g["arm"] in ("none", "independent"):
        return False
    return True


def _probe_task_key(g: dict) -> str:
    if g["qtype"] == "Q2":
        return f"Q2_{g['arm']}"
    return str(g["qtype"])


def _evaluate_dev_groups(
    host: QwenCacheHost,
    connector: Stage4MatchedConnector,
    kernel_mod: PhysicalKernelS17,
    dv_groups: List[dict],
    eval_bs: int = 32,
    max_items_per_group: int = 16,
    scorer_mode: str = "kv_cached",
    collect_latents: bool = False,
) -> dict:
    """Evaluate a connector + direct_physical baseline on dev groups (with temp-cal & canonical Probe A)."""
    del kernel_mod
    total_n = 0
    loss_sum = 0.0
    hits = 0
    phys_loss_sum = 0.0
    phys_hits = 0

    inf_n = 0
    inf_loss_sum = 0.0
    inf_hits = 0
    inf_phys_loss_sum = 0.0
    inf_phys_hits = 0

    q1_n = 0
    q1_hits = 0
    q1_loss = 0.0
    q2v_n = 0
    q2v_hits = 0
    q2v_loss = 0.0
    q2f_n = 0
    q2f_hits = 0
    q2f_loss = 0.0
    q3c_n = 0
    q3c_hits = 0
    q3c_loss = 0.0
    q3c_phys_loss = 0.0
    q3i_n = 0
    q3i_hits = 0
    q3a_n = 0
    q3a_hits = 0
    q3a_loss = 0.0

    latent_chunks: List[np.ndarray] = []
    target_chunks: List[np.ndarray] = []
    task_key_chunks: List[str] = []
    cal_records: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, bool]] = []

    connector.eval()
    with torch.no_grad():
        for g in dv_groups:
            n = min(int(max_items_per_group), len(g["words"]))
            is_inf = _is_informative_group(g)
            is_q3_cfo = bool(g["qtype"] == "Q3_repeat_x1" and g["source"] == "cfo")
            tkey = _probe_task_key(g)
            for st in range(0, n, eval_bs):
                ed = min(n, st + eval_bs)
                words = torch.from_numpy(g["words"][st:ed]).to(host.device)
                ps = torch.from_numpy(g["phase_slots"][st:ed]).to(host.device)
                meta = torch.from_numpy(g["meta_scalars"][st:ed]).to(host.device)
                rs = torch.from_numpy(g["ref_summary"][st:ed]).to(host.device)
                dd = torch.from_numpy(g["discrete_d"][st:ed]).to(host.device)
                lbl = torch.from_numpy(g["labels"][st:ed]).to(host.device)

                lp, centered_logits, _, lat_h = connector.score_batch(
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
                    discrete_d=dd,
                )
                ls = float(F.nll_loss(lp, lbl, reduction="sum").item())
                hs = int((lp.argmax(dim=-1) == lbl).sum().item())

                p_phys = torch.from_numpy(g["phys_probs"][st:ed]).to(host.device).clamp_min(1e-12)
                p_ls = float(F.nll_loss(torch.log(p_phys), lbl, reduction="sum").item())
                p_hs = int((p_phys.argmax(dim=-1) == lbl).sum().item())

                cnt = ed - st
                total_n += cnt
                loss_sum += ls
                hits += hs
                phys_loss_sum += p_ls
                phys_hits += p_hs

                if is_inf:
                    inf_n += cnt
                    inf_loss_sum += ls
                    inf_hits += hs
                    inf_phys_loss_sum += p_ls
                    inf_phys_hits += p_hs
                    if not connector.residual:
                        cal_records.append(
                            (
                                centered_logits.detach().cpu().float(),
                                words.detach().cpu(),
                                lbl.detach().cpu(),
                                g["qtype"],
                                is_q3_cfo,
                            )
                        )

                if g["qtype"] == "Q1":
                    q1_n += cnt
                    q1_hits += hs
                    q1_loss += ls
                elif g["qtype"] == "Q2" and g["arm"] == "valid":
                    q2v_n += cnt
                    q2v_hits += hs
                    q2v_loss += ls
                elif g["qtype"] == "Q2" and g["arm"] == "fresh_resample":
                    q2f_n += cnt
                    q2f_hits += hs
                    q2f_loss += ls
                elif is_q3_cfo:
                    q3c_n += cnt
                    q3c_hits += hs
                    q3c_loss += ls
                    q3c_phys_loss += p_ls
                elif g["qtype"] == "Q3_independent_x3":
                    q3i_n += cnt
                    q3i_hits += hs
                elif g["qtype"] == "Q3_after_pilot":
                    q3a_n += cnt
                    q3a_hits += hs
                    q3a_loss += ls

                if collect_latents and is_inf:
                    t_canon = unlift_probs(p_phys, words, g["qtype"]) if connector.quotient else p_phys
                    latent_chunks.append(lat_h.detach().cpu().numpy())
                    target_chunks.append(t_canon.detach().cpu().numpy())
                    task_key_chunks.extend([tkey] * cnt)

    dev_nll_all = loss_sum / max(1, total_n)
    dev_acc_all = hits / max(1, total_n)
    phys_nll_all = phys_loss_sum / max(1, total_n)
    phys_acc_all = phys_hits / max(1, total_n)

    dev_nll_inf = inf_loss_sum / max(1, inf_n)
    dev_acc_inf = inf_hits / max(1, inf_n)
    phys_nll_inf = inf_phys_loss_sum / max(1, inf_n)
    phys_acc_inf = inf_phys_hits / max(1, inf_n)

    opt_temp_dev = 1.0
    temp_cal_inf = dev_nll_inf
    temp_cal_q3c = q3c_loss / max(1, q3c_n)
    if cal_records and not connector.residual:
        best_inf_sum = float("inf")
        best_q3c_sum = q3c_loss
        for cand_t in np.geomspace(0.25, 4.0, 61):
            t_val = float(cand_t)
            c_inf_sum = 0.0
            c_q3c_sum = 0.0
            for c_logits, c_words, c_lbl, c_qt, c_is_q3c in cal_records:
                scaled = c_logits / t_val
                if connector.quotient:
                    c_probs = lift_scores(F.softmax(scaled, dim=-1), c_words, c_qt)
                    c_lp = torch.log(c_probs.clamp_min(1e-12))
                else:
                    c_lp = F.log_softmax(scaled, dim=-1)
                nll_b = float(F.nll_loss(c_lp, c_lbl, reduction="sum").item())
                c_inf_sum += nll_b
                if c_is_q3c:
                    c_q3c_sum += nll_b
            if c_inf_sum < best_inf_sum:
                best_inf_sum = c_inf_sum
                best_q3c_sum = c_q3c_sum
                opt_temp_dev = t_val
        temp_cal_inf = best_inf_sum / max(1, inf_n)
        temp_cal_q3c = best_q3c_sum / max(1, q3c_n)

    out = {
        "n_all": total_n,
        "n_informative": inf_n,
        "dev_nll": dev_nll_all,
        "dev_acc": dev_acc_all,
        "direct_physical_dev_nll": phys_nll_all,
        "direct_physical_dev_acc": phys_acc_all,
        "dev_nll_all": dev_nll_all,
        "dev_acc_all": dev_acc_all,
        "direct_physical_dev_nll_all": phys_nll_all,
        "direct_physical_dev_acc_all": phys_acc_all,
        "dev_nll_informative": dev_nll_inf,
        "dev_acc_informative": dev_acc_inf,
        "direct_physical_dev_nll_informative": phys_nll_inf,
        "direct_physical_dev_acc_informative": phys_acc_inf,
        "opt_temp_dev": float(opt_temp_dev),
        "temp_cal_dev_nll_informative": float(temp_cal_inf),
        "temp_cal_q3_cfo_nll": float(temp_cal_q3c),
        "delta_cal_informative_nats": float(dev_nll_inf - temp_cal_inf),
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
    if collect_latents and latent_chunks:
        out["_latents"] = np.concatenate(latent_chunks, axis=0)
        out["_phys_targets"] = np.concatenate(target_chunks, axis=0)
        out["_task_keys"] = task_key_chunks
    return out


def _fit_single_probe_pair(
    latents: np.ndarray,
    phys_targets: np.ndarray,
    seed: int = 202611,
    steps: int = 300,
) -> dict:
    """Fit held-out CV linear and 1-hidden-layer MLP probes from E_thetaA latent_h to Bayes posterior."""
    n = len(latents)
    if n < 20:
        return {
            "n_train": 0,
            "n_test": 0,
            "oracle_bayes_entropy_nats": 0.0,
            "probe_a_linear_kl_gap_nats": 0.0,
            "probe_a_mlp_kl_gap_nats": 0.0,
            "probe_a_min_gap_nats": 0.0,
        }
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    split = int(0.6 * n)
    tr_idx, te_idx = perm[:split], perm[split:]

    x_tr = torch.from_numpy(latents[tr_idx]).to(torch.float32)
    y_tr = torch.from_numpy(phys_targets[tr_idx]).to(torch.float32)
    x_te = torch.from_numpy(latents[te_idx]).to(torch.float32)
    y_te = torch.from_numpy(phys_targets[te_idx]).to(torch.float32)
    y_te_nll = float(-torch.sum(y_te * torch.log(y_te.clamp_min(1e-12)), dim=-1).mean().item())

    def fit_probe(model: nn.Module) -> float:
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            log_q = F.log_softmax(model(x_tr), dim=-1)
            loss = torch.sum(y_tr * (torch.log(y_tr.clamp_min(1e-12)) - log_q), dim=-1).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            log_q_te = F.log_softmax(model(x_te), dim=-1)
            kl = torch.sum(y_te * (torch.log(y_te.clamp_min(1e-12)) - log_q_te), dim=-1).mean().item()
        return float(max(0.0, kl))

    torch.manual_seed(seed)
    lin_probe = nn.Linear(latents.shape[1], 4)
    mlp_probe = nn.Sequential(nn.Linear(latents.shape[1], 128), nn.GELU(), nn.Linear(128, 4))
    lin_gap = fit_probe(lin_probe)
    mlp_gap = fit_probe(mlp_probe)
    return {
        "n_train": int(len(tr_idx)),
        "n_test": int(len(te_idx)),
        "oracle_bayes_entropy_nats": y_te_nll,
        "probe_a_linear_kl_gap_nats": lin_gap,
        "probe_a_mlp_kl_gap_nats": mlp_gap,
        "probe_a_min_gap_nats": min(lin_gap, mlp_gap),
    }


def _fit_probe_a_on_latents(
    latents: np.ndarray,
    phys_targets: np.ndarray,
    seed: int = 202611,
    task_keys: Optional[Sequence[str]] = None,
) -> dict:
    """Fit pooled and per-task held-out CV probes on canonical E_thetaA latent_h."""
    res = _fit_single_probe_pair(latents, phys_targets, seed=seed, steps=300)
    if task_keys is not None and len(task_keys) == len(latents):
        tk_arr = np.asarray(task_keys)
        per_task: Dict[str, dict] = {}
        for idx_t, tk in enumerate(sorted(set(task_keys))):
            mask = tk_arr == tk
            if int(mask.sum()) >= 20:
                per_task[tk] = _fit_single_probe_pair(
                    latents[mask], phys_targets[mask], seed=seed + 17 + idx_t, steps=300
                )
        res["per_task"] = per_task
        if per_task:
            res["per_task_linear_mean_nats"] = float(
                np.mean([v["probe_a_linear_kl_gap_nats"] for v in per_task.values()])
            )
            res["per_task_mlp_mean_nats"] = float(
                np.mean([v["probe_a_mlp_kl_gap_nats"] for v in per_task.values()])
            )
    return res


def _gate2_voi_decision(probe_a_gap: float, delta_host: float, sigma_3: float) -> dict:
    """Apply Stage 4 Two-Probe Phase-A VoI decision rule (STAGE4_PROTOCOL.md §3)."""
    if probe_a_gap > 0.030:
        return {
            "decision": "ABORT_FIX_CPU_CONNECTOR",
            "apply_w1_w3_slowhead_highway": False,
            "reason": f"Probe A connector gap {probe_a_gap:.4f} > 0.030 nats.",
        }
    if delta_host <= 0.030:
        return {
            "decision": "STAY_FROZEN_BRANCH0",
            "apply_w1_w3_slowhead_highway": False,
            "reason": f"Frozen host already saturated (delta_host={delta_host:.4f} <= 0.030 nats).",
        }
    use_highway = bool(delta_host > 0.315 and sigma_3 < 0.10)
    return {
        "decision": "TRIGGER_MODE1_SURGICAL_96",
        "apply_w1_w3_slowhead_highway": use_highway,
        "reason": f"Probe A passed ({probe_a_gap:.4f} <= 0.030) and delta_host={delta_host:.4f} > 0.030 nats.",
    }


def run_gate1_pilot_69step(
    repo_root: Path,
    out_dir: Path,
    seed: int = 202611,
    steps: int = 69,
    batch_size: int = 32,
    micro_batch_size: int = 16,
    lr: float = 1.5e-3,
    train_blocks: int = 8,
    dev_blocks: int = 2,
    scorer_mode: str = "kv_cached",
    model_id: str = MODEL_ID_DEFAULT,
    revision: str = PINNED_REVISION_DEFAULT,
    mock_tiny_for_test: bool = False,
    out_prefix: str = "pilot_69step",
    on_arm_done: Optional[Callable[[str], None]] = None,
) -> dict:
    """Run matched 4-arm Phase-A pilot/continuation on single fixed seed + Two-Probe Phase-A VoI readout."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    host = QwenCacheHost(model_id=model_id, revision=revision, mock_tiny_for_test=mock_tiny_for_test)
    encoder, _ = load_s17_encoder(repo_root)
    kernel_mod = PhysicalKernelS17(encoder.table).to(host.device)
    tr_groups, dv_groups = build_stage4_pilot_groups(encoder, kernel_mod, train_blocks=train_blocks, dev_blocks=dev_blocks)

    # Identical minibatch pool & step sequence for all 4 arms
    tr_minibatches: List[Tuple[int, int, int]] = []
    weighted_pool: List[int] = []
    for gi, g in enumerate(tr_groups):
        n = len(g["words"])
        w = 4 if g["qtype"] == "Q3_repeat_x1" else (3 if (g["qtype"] == "Q2" and g["arm"] in ("valid", "fresh_resample")) else 2 if g["qtype"] in ("Q1", "Q3_after_pilot") else 1)
        for st in range(0, n, batch_size):
            mb_id = len(tr_minibatches)
            tr_minibatches.append((gi, st, min(n, st + batch_size)))
            weighted_pool.extend([mb_id] * w)

    rng_order = random.Random(seed + 1009)
    order: List[int] = []
    while len(order) < steps:
        epoch_pool = list(weighted_pool)
        rng_order.shuffle(epoch_pool)
        order.extend(epoch_pool)
    order = order[:steps]

    warmup_steps = max(1, int(round(0.10 * steps)))
    arm_reports: Dict[str, dict] = {}
    pure_eval_data: Optional[dict] = None
    pure_jac_q3: Optional[dict] = None
    pure_jac_q1: Optional[dict] = None

    for arm in ARM_NAMES_STAGE4:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        connector = Stage4MatchedConnector(arm=arm, prior_table=encoder.table, hidden_size=host.hidden_size).to(host.device)
        opt = torch.optim.AdamW(connector.parameters(), lr=lr, weight_decay=1e-4)

        def _lr_lam(idx: int) -> float:
            if idx < warmup_steps:
                return float(idx + 1) / float(warmup_steps)
            prog = min(1.0, float(idx - warmup_steps) / float(max(1, steps - warmup_steps)))
            return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lam)
        t0 = time.perf_counter()
        if host.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(host.device)

        connector.train()
        last_losses: List[float] = []
        for step_i, mb_idx in enumerate(order):
            gi, st, ed = tr_minibatches[mb_idx]
            g = tr_groups[gi]
            mb_total = ed - st
            opt.zero_grad(set_to_none=True)
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
                    discrete_d=dd,
                )
                losses = rao_blackwell_bayes_kl_cf_loss(
                    lp, target_p, cf_swapped_logits=None, lambda_cf=0.0, lambda_norm=0.0
                )
                scaled_loss = losses["total"] * (float(m_ed - m_st) / float(mb_total))
                scaled_loss.backward()
                step_loss_val += float(scaled_loss.item())
                del lp, losses, scaled_loss
            torch.nn.utils.clip_grad_norm_(connector.parameters(), 1.0)
            opt.step()
            sched.step()
            last_losses.append(step_loss_val)
            if (step_i + 1) % 32 == 0 or (step_i + 1) == steps:
                print(
                    f"[{arm} step {step_i + 1}/{steps}] loss={step_loss_val:.4f} elapsed={time.perf_counter() - t0:.1f}s",
                    flush=True,
                )

        train_s = round(time.perf_counter() - t0, 2)
        collect_lat = arm == "equivariant_host_pure"
        ev = _evaluate_dev_groups(
            host,
            connector,
            kernel_mod,
            dv_groups,
            max_items_per_group=16,
            scorer_mode=scorer_mode,
            collect_latents=collect_lat,
        )
        if collect_lat:
            pure_eval_data = ev
            q3_dev_g = next(g for g in dv_groups if g["qtype"] == "Q3_repeat_x1" and abs(g["omega"]) > 0.0)
            q1_dev_g = next(g for g in dv_groups if g["qtype"] == "Q1" and abs(g["omega"]) > 0.0)
            pure_jac_q3 = measure_candidate_jacobian(host, connector, q3_dev_g, max_items=4, scorer_mode=scorer_mode)
            pure_jac_q1 = measure_candidate_jacobian(host, connector, q1_dev_g, max_items=4, scorer_mode=scorer_mode)

        host_sha_after = _compute_host_fingerprint(host.lm)
        assert host_sha_after == host.host_sha256, "Frozen host hash mutated during Stage 4 Phase-A pilot!"

        ckpt_file = ckpt_dir / f"stage4_phaseA_{arm}_s{seed}_step{steps}.pt"
        torch.save({"arm": arm, "seed": seed, "steps": steps, "config": connector.connector_config(), "state_dict": connector.state_dict()}, ckpt_file)
        peak_mb = round(torch.cuda.max_memory_allocated(host.device) / (1024.0 ** 2), 1) if host.device.type == "cuda" else 0.0

        arm_reports[arm] = {
            "arm": arm,
            "trainable_params": connector.param_count(),
            "steps": steps,
            "train_wall_s": train_s,
            "peak_vram_mb": peak_mb,
            "final_train_loss": float(np.mean(last_losses[-min(8, len(last_losses)):])),
            "dev_metrics": {k: v for k, v in ev.items() if not k.startswith("_")},
            "checkpoint": ckpt_file.name,
            "checkpoint_sha256": hashlib.sha256(ckpt_file.read_bytes()).hexdigest(),
        }
        print(
            f"[DONE {arm}] wall={train_s}s dev_nll_inf={ev['dev_nll_informative']:.4f} "
            f"temp_cal_inf={ev['temp_cal_dev_nll_informative']:.4f} (T={ev['opt_temp_dev']:.2f}) "
            f"q1={ev['q1_acc']:.3f} q2v={ev['q2_valid_acc']:.3f} q2f={ev['q2_fresh_acc']:.3f} "
            f"q3c={ev['q3_repeat_cfo_acc']:.3f}",
            flush=True,
        )
        if on_arm_done is not None:
            on_arm_done(arm)
        del connector, opt, sched
        gc.collect()
        if host.device.type == "cuda":
            torch.cuda.empty_cache()

    assert pure_eval_data is not None and pure_jac_q3 is not None and pure_jac_q1 is not None
    probe_a = _fit_probe_a_on_latents(
        pure_eval_data["_latents"],
        pure_eval_data["_phys_targets"],
        seed=seed,
        task_keys=pure_eval_data.get("_task_keys"),
    )
    delta_host_all = float(pure_eval_data["dev_nll_all"] - pure_eval_data["direct_physical_dev_nll_all"])
    delta_host_informative_raw = float(
        pure_eval_data["dev_nll_informative"] - pure_eval_data["direct_physical_dev_nll_informative"]
    )
    delta_host_informative_tempcal = float(
        pure_eval_data["temp_cal_dev_nll_informative"] - pure_eval_data["direct_physical_dev_nll_informative"]
    )
    delta_host_q3_cfo_raw = float(
        pure_eval_data["q3_repeat_cfo_nll"] - pure_eval_data["q3_repeat_cfo_phys_nll"]
    )
    delta_host_q3_cfo_tempcal = float(
        pure_eval_data["temp_cal_q3_cfo_nll"] - pure_eval_data["q3_repeat_cfo_phys_nll"]
    )
    eff_probe_a = min(
        probe_a["probe_a_min_gap_nats"],
        probe_a.get("per_task_mlp_mean_nats", probe_a["probe_a_min_gap_nats"]),
    )
    decision = _gate2_voi_decision(eff_probe_a, delta_host_informative_tempcal, pure_jac_q3["sigma_3"])

    pilot_out = {
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "micro_batch_size": micro_batch_size,
        "model_id": host.model_id,
        "host_sha256": host.host_sha256,
        "arms": arm_reports,
    }
    voi_out = {
        "seed": seed,
        "steps": steps,
        "probe_a": probe_a,
        "frozen_host_pure_dev_nll_all": float(pure_eval_data["dev_nll_all"]),
        "direct_physical_dev_nll_all": float(pure_eval_data["direct_physical_dev_nll_all"]),
        "delta_host_all_nats": delta_host_all,
        "frozen_host_pure_dev_nll_informative": float(pure_eval_data["dev_nll_informative"]),
        "direct_physical_dev_nll_informative": float(pure_eval_data["direct_physical_dev_nll_informative"]),
        "delta_host_informative_nats": delta_host_informative_tempcal,
        "delta_host_informative_raw_nats": delta_host_informative_raw,
        "delta_host_informative_tempcal_nats": delta_host_informative_tempcal,
        "delta_cal_informative_nats": float(pure_eval_data["delta_cal_informative_nats"]),
        "opt_temp_dev": float(pure_eval_data["opt_temp_dev"]),
        "q3_repeat_cfo_nll": float(pure_eval_data["q3_repeat_cfo_nll"]),
        "q3_repeat_cfo_phys_nll": float(pure_eval_data["q3_repeat_cfo_phys_nll"]),
        "delta_host_q3_cfo_nats": delta_host_q3_cfo_raw,
        "delta_host_q3_cfo_raw_nats": delta_host_q3_cfo_raw,
        "delta_host_q3_cfo_tempcal_nats": delta_host_q3_cfo_tempcal,
        "post_phaseA_jacobian": pure_jac_q3,
        "post_phaseA_jacobian_q1": pure_jac_q1,
        "gate2_recommendation": decision,
    }
    pilot_file = "pilot_69step_4arms.json" if out_prefix == "pilot_69step" else f"{out_prefix}_4arms.json"
    voi_file = "two_probe_voi_readout.json" if out_prefix == "pilot_69step" else f"two_probe_voi_readout_{steps}.json"
    (out_dir / pilot_file).write_text(json.dumps(pilot_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_dir / voi_file).write_text(json.dumps(voi_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"pilot": pilot_out, "voi": voi_out}
