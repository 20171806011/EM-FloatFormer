"""Step C.1 Smoke Verification + Step C.2 Connector Training + Dev Locking for Ministral-3-3B."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F

from research.stage3_ministral_cache.data import MotherBatch, generate_mothers
from research.stage3_ministral_cache.eval_direct import load_frozen_models
from research.stage3_ministral_cache.ministral_connector import (
    MODEL_ID_DEFAULT,
    MinistralCacheHost,
    WaveformCacheConnector,
    _compute_host_fingerprint,
    build_ref_feature,
)
from research.stage3_ministral_cache.schema import (
    CONDITION_SPECS,
    ROOTS,
    ROTATIONS,
    Reference,
    build_query_prompt,
    dedup_references,
    matched_sum,
    sector_of_complex,
)


STAGE3_DIR = Path(__file__).resolve().parent
TRAIN_CONDITIONS = ("C1_coherent_2db", "C2_discrete_walk", "C3_mid_pilot_cont")
TRAIN_ARMS = ("none", "valid", "independent", "fresh_resample")


def _encode_split_cache(encoder_obj, batch: MotherBatch) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    """Pre-compute frozen integer codes `(z1, z2)` for each rotation once and discard raw Y1/Y2."""
    codes = {}
    for rot in ROTATIONS:
        v1, v2 = batch.rotated_view(*rot)
        z1 = np.asarray(encoder_obj.encode(v1), dtype=np.int64)
        z2 = np.asarray(encoder_obj.encode(v2), dtype=np.int64)
        codes[rot] = (z1, z2)
    return codes


def build_task_batches(
    split: str,
    encoder_obj,
    cond_names: Tuple[str, ...],
    max_blocks: Optional[int] = None,
) -> Tuple[List[dict], Dict[str, dict]]:
    """Build grouped zero-padding task units with ONLY integer cache z1/z2 and revealed deduped refs."""
    groups: List[dict] = []
    meta_batches: Dict[str, dict] = {}
    for cname in cond_names:
        batch = generate_mothers(split, cname, max_blocks=max_blocks)
        codes_map = _encode_split_cache(encoder_obj, batch)
        meta_batches[cname] = {"batch": batch, "codes": codes_map}
        cond = batch.condition
        n_m = len(batch.x1)

        # Note: all prompts share identical length within a (cond, rot, task_type, arm) group!
        for rot in ROTATIONS:
            s1, s2 = rot
            z1_np, z2_np = codes_map[rot]

            # 1. Q1
            p_q1, ch_q1 = build_query_prompt("Q1", cond)
            y_q1 = (batch.x2 - batch.x1 + s2 - s1) % 4
            groups.append(
                {
                    "cond": cname,
                    "rot": rot,
                    "qtype": "Q1",
                    "arm": "pre_ref",
                    "prompt": p_q1,
                    "choices": ch_q1,
                    "z1": z1_np,
                    "z2": z2_np,
                    "ref_feats_list": [],
                    "labels": y_q1,
                }
            )

            # 2. Q3 repeat_x1 & independent_x3
            for proto in ("repeat_x1", "independent_x3"):
                p_q3, ch_q3 = build_query_prompt("Q3", cond, protocol=proto)
                if proto == "repeat_x1":
                    # Notice: under (1, 1) both windows rotated -> true receiver k3 rotated by +1; under (0,1) holds!
                    y3_rot = batch.y3_repeat * ROOTS[s1 % 4]
                else:
                    y3_rot = batch.y3_indep * ROOTS[s1 % 4]
                y_q3 = sector_of_complex(matched_sum(y3_rot))
                groups.append(
                    {
                        "cond": cname,
                        "rot": rot,
                        "qtype": f"Q3_{proto}",
                        "arm": "pre_ref",
                        "prompt": p_q3,
                        "choices": ch_q3,
                        "z1": z1_np,
                        "z2": z2_np,
                        "ref_feats_list": [],
                        "labels": y_q3,
                    }
                )

            # 3. Q2 across reference arms (using shared dedup contract)
            y_q2 = (batch.x1 + s1) % 4
            for arm in TRAIN_ARMS:
                refs = batch.references_for_arm(arm, coord_rot=0)
                p_q2, ch_q2 = build_query_prompt("Q2", cond, ref_arm=arm, references=refs, deduplicate=True)
                # Extract revealed Reference features strictly matching unique ref tokens emitted in prompt
                seen_ids = set()
                r_feats = []
                for r in refs:
                    if r.acquisition_id in seen_ids:
                        continue
                    seen_ids.add(r.acquisition_id)
                    r_feats.append(build_ref_feature(r))
                groups.append(
                    {
                        "cond": cname,
                        "rot": rot,
                        "qtype": "Q2",
                        "arm": arm,
                        "prompt": p_q2,
                        "choices": ch_q2,
                        "z1": z1_np,
                        "z2": z2_np,
                        "ref_feats_list": r_feats,
                        "labels": y_q2,
                    }
                )

            # 4. Q3_after_pilot on valid arm (matching dev & test evaluation protocol)
            refs_v = batch.references_for_arm("valid", coord_rot=0)
            p_q3a, ch_q3a = build_query_prompt(
                "Q3_after_pilot", cond, protocol="repeat_x1", ref_arm="valid", references=refs_v, deduplicate=True
            )
            seen_ids_v = set()
            r_feats_v = []
            for r in refs_v:
                if r.acquisition_id in seen_ids_v:
                    continue
                seen_ids_v.add(r.acquisition_id)
                r_feats_v.append(build_ref_feature(r))
            y_q3_rep = sector_of_complex(matched_sum(batch.y3_repeat * ROOTS[s1 % 4]))
            groups.append(
                {
                    "cond": cname,
                    "rot": rot,
                    "qtype": "Q3_after_pilot",
                    "arm": "valid",
                    "prompt": p_q3a,
                    "choices": ch_q3a,
                    "z1": z1_np,
                    "z2": z2_np,
                    "ref_feats_list": r_feats_v,
                    "labels": y_q3_rep,
                }
            )
    return groups, meta_batches


def run_smoke_verification(
    model_id: str = MODEL_ID_DEFAULT,
    encoder_name: str = "equivariant_b8_s17",
    out_dir: Optional[Path] = None,
) -> dict:
    """Step C.1: Load 3B host on Colab GPU, verify frozen host, gradients, KV-cache parity, decode, & timing."""
    out_dir = out_dir or (STAGE3_DIR / "results" / "host")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    host = MinistralCacheHost(model_id=model_id)
    load_s = time.perf_counter() - t0
    actual_param_dtype = str(next(host.lm.parameters()).dtype)
    model_vram_mb_after_load = (
        float(torch.cuda.memory_allocated(host.device)) / (1024.0 ** 2)
        if host.device.type == "cuda"
        else 0.0
    )
    if host.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(host.device)

    models = load_frozen_models()
    enc = models[encoder_name]
    num_states = int(len(enc.table))
    connector = WaveformCacheConnector(num_states=num_states, hidden_size=host.hidden_size, target_embed_std=host.target_embed_std).to(host.device)

    dev_groups, _ = build_task_batches("s3_dev", enc, ("C1_coherent_2db", "C3_mid_pilot_cont"), max_blocks=1)
    # Grab valid Q2 examples from C3_mid_pilot_cont and Q3 examples
    q2_grp = [g for g in dev_groups if g["cond"] == "C3_mid_pilot_cont" and g["qtype"] == "Q2" and g["arm"] == "valid"][0]
    b = 4
    z1 = torch.tensor(q2_grp["z1"][:b], dtype=torch.long)
    z2 = torch.tensor(q2_grp["z2"][:b], dtype=torch.long)
    r_list = [torch.tensor(rf[:b], dtype=torch.float32) for rf in q2_grp["ref_feats_list"]]
    lbls = torch.tensor(q2_grp["labels"][:b], device=host.device, dtype=torch.long)

    # 1. KV-cache vs uncached full forward parity check
    connector.eval()
    with torch.no_grad():
        pref = host.build_prefix_embeds(connector, q2_grp["prompt"], z1, z2, r_list)
        lp_kv, ll_kv = host.score_candidates_kv(pref, q2_grp["choices"])
        lp_unc, ll_unc = host.score_candidates_uncached(pref, q2_grp["choices"])
        max_kv_ll_diff = float((ll_kv - ll_unc).abs().max().item())
        max_kv_prob_diff = float((lp_kv.exp() - lp_unc.exp()).abs().max().item())
    if host.device.type == "cuda":
        torch.cuda.empty_cache()

    # 2. Backward + Optimizer step gradient audit
    connector.train()
    opt = torch.optim.AdamW(connector.parameters(), lr=1e-3)
    opt.zero_grad(set_to_none=True)
    t_step0 = time.perf_counter()
    pref_tr = host.build_prefix_embeds(connector, q2_grp["prompt"], z1, z2, r_list)
    lp_tr, _ = host.score_candidates_kv(pref_tr, q2_grp["choices"])
    loss = F.nll_loss(lp_tr, lbls)
    assert torch.isfinite(loss), "Non-finite smoke training loss!"
    loss.backward()

    host_grads_nonzero = sum(1 for p in host.lm.parameters() if p.grad is not None)
    conn_grads_zero = [n for n, p in connector.named_parameters() if p.grad is None or float(p.grad.abs().sum().item()) == 0.0]
    assert host_grads_nonzero == 0, f"Frozen host received gradients: {host_grads_nonzero}"
    assert len(conn_grads_zero) == 0, f"Connector params missing gradients: {conn_grads_zero}"

    opt.step()
    step_ms = (time.perf_counter() - t_step0) * 1000.0
    sha_after = _compute_host_fingerprint(host.lm)
    assert sha_after == host.host_sha256, "Frozen host weights mutated!"

    # 3. Genuine Chinese greedy generation check on single sample
    connector.eval()
    pref_1 = host.build_prefix_embeds(connector, q2_grp["prompt"], z1[:1], z2[:1], [rf[:1] for rf in r_list])
    gen_txt = host.generate_text(pref_1, max_new_tokens=32)

    # 4. Pure inference timing sweep on b=32 (>=20 GiB GPU) or b=16
    inf_b = 32 if (host.device.type == "cuda" and torch.cuda.get_device_properties(host.device).total_memory >= 20 * (1024 ** 3)) else 16
    q3_grp = [g for g in dev_groups if g["cond"] == "C1_coherent_2db" and g["qtype"] == "Q3_repeat_x1"][0]
    t_inf0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(3):
            pref_i = host.build_prefix_embeds(
                connector, q3_grp["prompt"],
                torch.tensor(q3_grp["z1"][:inf_b], dtype=torch.long),
                torch.tensor(q3_grp["z2"][:inf_b], dtype=torch.long),
                [],
            )
            host.score_candidates_kv(pref_i, q3_grp["choices"])
    infer_ms_per_example = ((time.perf_counter() - t_inf0) * 1000.0) / (3 * inf_b)

    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated(host.device)) / (1024.0 ** 2)
        if host.device.type == "cuda"
        else 0.0
    )
    gpu_name = torch.cuda.get_device_name(host.device) if host.device.type == "cuda" else "CPU"

    # Decide blind test block budget cleanly before touching s3_test:
    # Each block = 64 mothers; across 3 encoders * ~31 groups * 7 conds = ~41,664 queries per block.
    est_s_1blk = (41664 * infer_ms_per_example) / 1000.0
    if 4 * est_s_1blk <= 900.0:
        rec_test_blocks = 4
    elif 2 * est_s_1blk <= 900.0:
        rec_test_blocks = 2
    else:
        rec_test_blocks = 1

    smoke_report = {
        "model_id": model_id,
        "hf_sha": host.hf_sha,
        "transformers_version": host.transformers_version,
        "gpu_name": gpu_name,
        "dtype": str(host.dtype),
        "actual_param_dtype": actual_param_dtype,
        "model_vram_mb_after_load": round(model_vram_mb_after_load, 1),
        "load_time_s": round(load_s, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "frozen_host_params": host.frozen_param_count,
        "trainable_connector_params": connector.param_count(),
        "host_sha256_before": host.host_sha256,
        "host_sha256_after": sha_after,
        "host_sha_identical": sha_after == host.host_sha256,
        "host_grads_nonzero": host_grads_nonzero,
        "conn_grads_all_nonzero": len(conn_grads_zero) == 0,
        "kv_cache_max_loglik_diff": max_kv_ll_diff,
        "kv_cache_max_prob_diff": max_kv_prob_diff,
        "smoke_train_loss": float(loss.item()),
        "prefix_len_tokens": int(pref.shape[1]),
        "smoke_batch_size": int(b),
        "smoke_step_ms_b4": round(step_ms, 1),
        "smoke_step_ms_b8": round(step_ms * 2.0, 1),
        "smoke_infer_ms_per_example": round(infer_ms_per_example, 2),
        "recommended_host_test_blocks": rec_test_blocks,
        "smoke_generated_sample": gen_txt,
        "candidate_token_audit": host.candidate_token_audit,
    }
    (out_dir / "smoke_audit.json").write_text(json.dumps(smoke_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(smoke_report, ensure_ascii=False, indent=2))
    return smoke_report


def train_and_lock_connectors(
    host: MinistralCacheHost,
    encoders: Sequence[str] = ("equivariant_b8_s17", "hand_b8", "plain_b8_s17"),
    epochs: int = 12,
    batch_size: int = 32,
    micro_batch_size: int = 32,
    lr: float = 1.5e-3,
    train_blocks: int = 8,     # 512 mothers * 3 conds * 24 groups -> high state coverage
    dev_blocks: int = 2,       # 128 mothers on s3_dev for early stop & tolerance lock
    host_test_blocks: Optional[int] = None,
    max_steps_per_epoch: Optional[int] = 250,
    seed: int = 20261123,
    out_dir: Optional[Path] = None,
    resume: bool = False,
) -> dict:
    """Train ONLY small connectors on s3_train, calibrate & lock tolerances STRICTLY on s3_dev."""
    out_dir = out_dir or STAGE3_DIR
    weights_dir = out_dir / "connector_weights"
    res_dir = out_dir / "results" / "host"
    weights_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    # Auto-cap micro_batch_size safely to GPU VRAM headroom so we never OOM across T4/L4/A100/G4
    eff_micro_bs = max(1, int(micro_batch_size))
    dev_eval_bs = int(batch_size)
    if host.device.type == "cuda":
        total_vram_gb = float(torch.cuda.get_device_properties(host.device).total_memory) / (1024.0 ** 3)
        if total_vram_gb < 18.0:
            eff_micro_bs = min(eff_micro_bs, 4)
        elif total_vram_gb < 30.0:
            eff_micro_bs = min(eff_micro_bs, 8)
        elif total_vram_gb < 65.0:
            eff_micro_bs = min(eff_micro_bs, 16)
        else:
            eff_micro_bs = min(eff_micro_bs, 32)
            dev_eval_bs = 64

    models = load_frozen_models()
    if host_test_blocks is None:
        smoke_p = res_dir / "smoke_audit.json"
        if smoke_p.exists():
            host_test_blocks = int(json.loads(smoke_p.read_text(encoding="utf-8")).get("recommended_host_test_blocks", 2))
        else:
            host_test_blocks = 2

    prior_summaries = (
        json.loads((res_dir / "training_summary.json").read_text(encoding="utf-8"))
        if (resume and (res_dir / "training_summary.json").exists())
        else {}
    )
    train_summaries = {}
    dev_lock_records = {
        "model_id": host.model_id,
        "hf_sha": host.hf_sha,
        "dtype": str(host.dtype),
        "host_sha256": host.host_sha256,
        "train_blocks": int(train_blocks),
        "dev_blocks": int(dev_blocks),
        "batch_size": int(batch_size),
        "micro_batch_size": int(eff_micro_bs),
        "max_steps_per_epoch": int(max_steps_per_epoch) if max_steps_per_epoch is not None else None,
        "host_test_blocks_locked": int(host_test_blocks),
        "seed": int(seed),
        "encoders": {},
    }

    for enc_name in encoders:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        enc = models[enc_name]
        connector = WaveformCacheConnector(
            num_states=int(len(enc.table)),
            hidden_size=host.hidden_size,
            target_embed_std=host.target_embed_std,
        ).to(host.device)

        ckpt_path = weights_dir / f"connector_{enc_name}.pt"
        if resume and ckpt_path.exists():
            prev_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            assert prev_ckpt.get("connector_config") == connector.connector_config(), (
                f"Connector config mismatch on resume for {enc_name}"
            )
            connector.load_state_dict(prev_ckpt["state_dict"])
            best_dev_nll = float(prev_ckpt.get("best_dev_nll", float("inf")))
            best_state = {k: v.detach().cpu().clone() for k, v in prev_ckpt["state_dict"].items()}
            best_epoch = int(prev_ckpt.get("best_epoch", 0))
            prior_rec = prior_summaries.get(enc_name, {})
            history = list(prior_rec.get("history", []))
            ep_offset = len(history)
            prior_time_s = float(prior_rec.get("train_time_s", 0.0))
        else:
            best_dev_nll = float("inf")
            best_state = None
            best_epoch = 0
            history = []
            ep_offset = 0
            prior_time_s = 0.0

        tr_groups, _ = build_task_batches("s3_train", enc, TRAIN_CONDITIONS, max_blocks=train_blocks)
        dv_groups, _ = build_task_batches("s3_dev", enc, TRAIN_CONDITIONS, max_blocks=dev_blocks)

        # Build minibatches preserving pure single-group homogeneous sequence lengths (zero padding!)
        tr_minibatches = []
        weighted_pool = []
        for gi, g in enumerate(tr_groups):
            n = len(g["labels"])
            if g["qtype"] == "Q1" or (g["qtype"] == "Q2" and g["arm"] in ("valid", "fresh_resample")):
                w = 3
            elif g["qtype"] in ("Q3_repeat_x1", "Q3_after_pilot"):
                w = 2
            else:
                w = 1
            for st in range(0, n, batch_size):
                mb_idx = len(tr_minibatches)
                tr_minibatches.append((gi, st, min(n, st + batch_size)))
                weighted_pool.extend([mb_idx] * w)

        steps_per_ep = (
            min(len(weighted_pool), int(max_steps_per_epoch))
            if (max_steps_per_epoch is not None and max_steps_per_epoch > 0)
            else len(weighted_pool)
        )
        total_steps = max(1, epochs * steps_per_ep)
        warmup_steps = max(1, int(round(0.05 * total_steps)))

        def _lr_lambda(step_idx: int) -> float:
            if step_idx < warmup_steps:
                return float(step_idx + 1) / float(warmup_steps)
            prog = min(1.0, float(step_idx - warmup_steps) / float(max(1, total_steps - warmup_steps)))
            return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

        opt = torch.optim.AdamW(connector.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)

        t_start = time.perf_counter()

        for ep in range(1, epochs + 1):
            ep_global = ep_offset + ep
            connector.train()
            rng_ep = random.Random(seed + ep_global)
            order = list(weighted_pool)
            rng_ep.shuffle(order)
            order = order[:steps_per_ep]

            tr_loss_sum = 0.0
            tr_cnt = 0
            mb_step = eff_micro_bs
            for mb_idx in order:
                gi, st, ed = tr_minibatches[mb_idx]
                g = tr_groups[gi]
                mb_total = ed - st

                opt.zero_grad(set_to_none=True)
                mb_loss_sum = 0.0
                for m_st in range(st, ed, mb_step):
                    m_ed = min(ed, m_st + mb_step)
                    z1 = torch.tensor(g["z1"][m_st:m_ed], dtype=torch.long)
                    z2 = torch.tensor(g["z2"][m_st:m_ed], dtype=torch.long)
                    r_list = [torch.tensor(rf[m_st:m_ed], dtype=torch.float32) for rf in g["ref_feats_list"]]
                    lbls = torch.tensor(g["labels"][m_st:m_ed], device=host.device, dtype=torch.long)

                    pref = host.build_prefix_embeds(connector, g["prompt"], z1, z2, r_list)
                    lp, _ = host.score_candidates_kv(pref, g["choices"])
                    loss = F.nll_loss(lp, lbls, reduction="sum") / float(mb_total)
                    assert torch.isfinite(loss), f"Non-finite training loss in {enc_name}!"
                    loss.backward()
                    mb_loss_sum += float(loss.item()) * float(mb_total)
                torch.nn.utils.clip_grad_norm_(connector.parameters(), 1.0)
                opt.step()
                sched.step()

                tr_loss_sum += mb_loss_sum
                tr_cnt += mb_total

            # Evaluate on unweighted s3_dev with detailed per-task breakdown
            connector.eval()
            dev_loss_sum = 0.0
            dev_cnt = 0
            bkt = {
                "q1": [0.0, 0, 0],
                "q3_rep": [0.0, 0, 0],
                "q2_val": [0.0, 0, 0],
                "q2_none": [0.0, 0, 0],
            }
            with torch.no_grad():
                for g in dv_groups:
                    n = len(g["labels"])
                    bkey = None
                    if g["qtype"] == "Q1":
                        bkey = "q1"
                    elif g["qtype"] == "Q3_repeat_x1":
                        bkey = "q3_rep"
                    elif g["qtype"] == "Q2" and g["arm"] == "valid":
                        bkey = "q2_val"
                    elif g["qtype"] == "Q2" and g["arm"] == "none":
                        bkey = "q2_none"

                    for st in range(0, n, dev_eval_bs):
                        ed = min(n, st + dev_eval_bs)
                        z1 = torch.tensor(g["z1"][st:ed], dtype=torch.long)
                        z2 = torch.tensor(g["z2"][st:ed], dtype=torch.long)
                        r_list = [torch.tensor(rf[st:ed], dtype=torch.float32) for rf in g["ref_feats_list"]]
                        lbls = torch.tensor(g["labels"][st:ed], device=host.device, dtype=torch.long)
                        pref = host.build_prefix_embeds(connector, g["prompt"], z1, z2, r_list)
                        lp, _ = host.score_candidates_kv(pref, g["choices"])
                        l_sum = float(F.nll_loss(lp, lbls, reduction="sum").item())
                        dev_loss_sum += l_sum
                        dev_cnt += (ed - st)
                        if bkey is not None:
                            bkt[bkey][0] += l_sum
                            bkt[bkey][1] += int((lp.argmax(dim=-1) == lbls).sum().item())
                            bkt[bkey][2] += (ed - st)

            dev_nll = dev_loss_sum / max(1, dev_cnt)
            tr_nll = tr_loss_sum / max(1, tr_cnt)
            ep_rec = {
                "epoch": ep_global,
                "train_nll": round(tr_nll, 5),
                "dev_nll": round(dev_nll, 5),
                "q1_acc": round(bkt["q1"][1] / max(1, bkt["q1"][2]), 4),
                "q1_nll": round(bkt["q1"][0] / max(1, bkt["q1"][2]), 5),
                "q3_rep_acc": round(bkt["q3_rep"][1] / max(1, bkt["q3_rep"][2]), 4),
                "q3_rep_nll": round(bkt["q3_rep"][0] / max(1, bkt["q3_rep"][2]), 5),
                "q2_val_acc": round(bkt["q2_val"][1] / max(1, bkt["q2_val"][2]), 4),
                "q2_val_nll": round(bkt["q2_val"][0] / max(1, bkt["q2_val"][2]), 5),
                "q2_none_nll": round(bkt["q2_none"][0] / max(1, bkt["q2_none"][2]), 5),
            }
            history.append(ep_rec)
            print(
                f"[{enc_name}] ep={ep_global}/{ep_offset + epochs} train_nll={tr_nll:.4f} dev_nll={dev_nll:.4f} | "
                f"q1={ep_rec['q1_acc']:.3f}({ep_rec['q1_nll']:.3f}) "
                f"q3r={ep_rec['q3_rep_acc']:.3f}({ep_rec['q3_rep_nll']:.3f}) "
                f"q2v={ep_rec['q2_val_acc']:.3f}({ep_rec['q2_val_nll']:.3f}) "
                f"q2n_nll={ep_rec['q2_none_nll']:.3f}",
                flush=True,
            )
            if dev_nll < best_dev_nll:
                best_dev_nll = dev_nll
                best_epoch = ep_global
                best_state = {k: v.detach().cpu().clone() for k, v in connector.state_dict().items()}

        train_time_s = prior_time_s + (time.perf_counter() - t_start)
        connector.load_state_dict(best_state)
        connector.eval()

        # Fit scalar temperature calibration + lock dev rotational / hold / duplicate tolerances!
        from research.stage3_ministral_cache.eval_host import evaluate_host_on_split_subset

        dev_res = evaluate_host_on_split_subset(
            host, connector, enc, "s3_dev", TRAIN_CONDITIONS, max_blocks=dev_blocks, batch_size=dev_eval_bs
        )
        # Measure max & p95 rotation/hold/duplicate residuals strictly on dev to lock tau_host before test!
        rot_max = max(
            v["max_abs"]
            for cname in TRAIN_CONDITIONS
            for k, v in dev_res["conditions"][cname]["checks"].items()
            if "rot" in k or "hold" in k or "indep" in k
        )
        dup_max = max(
            dev_res["conditions"][cname]["checks"][k]["max_abs"]
            for cname in TRAIN_CONDITIONS
            for k in dev_res["conditions"][cname]["checks"]
            if "dup" in k
        )
        # Optimal dev scalar temperature T in [0.50, 3.00]
        dev_raw_ll = np.concatenate([dev_res["conditions"][c]["raw_loglik_all"] for c in TRAIN_CONDITIONS], axis=0)
        dev_raw_lbl = np.concatenate([dev_res["conditions"][c]["labels_all"] for c in TRAIN_CONDITIONS], axis=0)
        best_temp = 1.0
        best_cal_nll = float("inf")
        for cand_t in (0.50, 0.65, 0.75, 0.85, 0.90, 0.95, 1.00, 1.10, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00):
            scaled = dev_raw_ll / cand_t
            scaled = scaled - scaled.max(axis=1, keepdims=True)
            ex = np.exp(scaled)
            pr = ex / ex.sum(axis=1, keepdims=True)
            nll_c = float(-np.log(np.clip(pr[np.arange(len(dev_raw_lbl)), dev_raw_lbl], 1e-12, 1.0)).mean())
            if nll_c < best_cal_nll:
                best_cal_nll = nll_c
                best_temp = cand_t

        torch.save(
            {
                "encoder_name": enc_name,
                "num_states": int(len(enc.table)),
                "hidden_size": host.hidden_size,
                "target_embed_std": host.target_embed_std,
                "connector_config": connector.connector_config(),
                "state_dict": best_state,
                "best_epoch": best_epoch,
                "best_dev_nll": best_dev_nll,
                "dev_cal_temp": best_temp,
                "model_id": host.model_id,
                "hf_sha": host.hf_sha,
            },
            ckpt_path,
        )
        ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()

        # Lock tau_host cleanly on dev: 2x observed dev max residual + BF16 margin 0.02 (never touches test!)
        tau_rot_hold = max(0.05, min(0.20, float(2.0 * rot_max + 0.02)))
        tau_dup = max(1e-4, float(2.0 * dup_max + 1e-5))

        train_summaries[enc_name] = {
            "trainable_params": connector.param_count(),
            "resumed_from_epoch": ep_offset,
            "best_epoch": best_epoch,
            "best_dev_nll": best_dev_nll,
            "dev_cal_temp": best_temp,
            "dev_cal_nll": best_cal_nll,
            "train_time_s": round(train_time_s, 2),
            "history": history,
            "checkpoint": str(ckpt_path.relative_to(STAGE3_DIR)),
            "checkpoint_sha256": ckpt_sha,
        }
        dev_lock_records["encoders"][enc_name] = {
            "dev_rot_hold_max_abs": float(rot_max),
            "dev_dup_max_abs": float(dup_max),
            "locked_tau_rot_hold": tau_rot_hold,
            "locked_tau_dup": tau_dup,
            "locked_temp": best_temp,
            "checkpoint_sha256": ckpt_sha,
        }
        del connector, best_state, tr_groups, dv_groups, dev_res
        gc.collect()
        if host.device.type == "cuda":
            torch.cuda.empty_cache()

    host_sha_after = _compute_host_fingerprint(host.lm)
    assert host_sha_after == host.host_sha256, "Frozen host weights changed during connector training!"
    peak_vram_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
    dev_lock_records["host_sha256_after"] = host_sha_after
    dev_lock_records["host_sha_identical"] = host_sha_after == host.host_sha256
    dev_lock_records["peak_vram_mb"] = round(peak_vram_mb, 1)

    (res_dir / "training_summary.json").write_text(json.dumps(train_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    (res_dir / "host_dev_lock.json").write_text(json.dumps(dev_lock_records, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"training_summary": train_summaries, "host_dev_lock": dev_lock_records}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--model-id", default=MODEL_ID_DEFAULT)
    ap.add_argument("--encoders", default="equivariant_b8_s17,hand_b8,plain_b8_s17")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--micro-batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--train-blocks", type=int, default=8)
    ap.add_argument("--dev-blocks", type=int, default=2)
    ap.add_argument("--test-blocks", type=int, default=None)
    ap.add_argument("--max-steps-per-epoch", type=int, default=250)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        run_smoke_verification(model_id=args.model_id)
    if args.train:
        host_obj = MinistralCacheHost(model_id=args.model_id)
        encs = tuple(x.strip() for x in args.encoders.split(",") if x.strip())
        train_and_lock_connectors(
            host_obj,
            encoders=encs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            micro_batch_size=args.micro_batch_size,
            lr=args.lr,
            train_blocks=args.train_blocks,
            dev_blocks=args.dev_blocks,
            host_test_blocks=args.test_blocks,
            max_steps_per_epoch=args.max_steps_per_epoch,
            resume=args.resume,
        )
