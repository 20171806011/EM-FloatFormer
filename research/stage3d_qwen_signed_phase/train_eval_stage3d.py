"""GPU training, dev calibration, smoke verification, and blind evaluation routines for Stage 3D."""
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize_scalar
import torch
import torch.nn.functional as F

from research.stage3_ministral_cache.data import MotherBatch, generate_mothers
from research.stage3_ministral_cache.ministral_connector import (
    _compute_host_fingerprint,
    build_ref_feature,
)
from research.stage3_ministral_cache.schema import (
    ROOTS,
    ROTATIONS,
    matched_sum,
    sector_of_complex,
)
from research.stage3d_qwen_signed_phase.data_stage3d import (
    ARM_NAMES,
    BLIND_CELLS,
    CONNECTOR_SEEDS,
    DEV_OMEGAS,
    NS_BLIND_Q2_C4,
    NS_CFO_DEV,
    NS_CFO_TRAIN,
    NS_DEV_PILOT,
    Q2_BLIND_ARMS,
    REPLAY_TRAIN_CONDITIONS,
    TRAIN_ARMS,
    TRAIN_OMEGAS,
    TRAIN_TIMINGS,
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
    SignedPhaseCacheConnector,
)


def _encode_batch_rotations(encoder, batch: MotherBatch) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    codes = {}
    for rot in ROTATIONS:
        v1, v2 = batch.rotated_view(*rot)
        z1 = np.asarray(encoder.encode(v1), dtype=np.int64)
        z2 = np.asarray(encoder.encode(v2), dtype=np.int64)
        codes[rot] = (z1, z2)
    return codes


def _append_batch_groups(
    groups: List[dict],
    batch: MotherBatch,
    encoder,
    zero_arm: bool,
    source_tag: str,
) -> None:
    cond = batch.condition
    codes_map = _encode_batch_groups(encoder, batch)
    n_m = len(batch.x1)
    p_vec = public_phase_slots(cond.omega, cond.h, cond.ell, cond.t_p, zero_arm=zero_arm)
    phase_mat = np.repeat(p_vec[None, :], n_m, axis=0)

    for rot in ROTATIONS:
        s1, s2 = rot
        z1_np, z2_np = codes_map[rot]

        # 1. Q1
        p_q1, ch_q1 = build_stage3d_prompt("Q1", cond)
        groups.append(
            {
                "source": source_tag,
                "cond": cond.name,
                "rot": rot,
                "qtype": "Q1",
                "arm": "pre_ref",
                "prompt": p_q1,
                "choices": ch_q1,
                "z1": z1_np,
                "z2": z2_np,
                "phase_slots": phase_mat,
                "ref_feats_list": [],
                "labels": (batch.x2 - batch.x1 + s2 - s1) % 4,
            }
        )

        # 2. Q3 repeat_x1 and independent_x3
        for proto in ("repeat_x1", "independent_x3"):
            p_q3, ch_q3 = build_stage3d_prompt("Q3", cond, protocol=proto)
            y3_rot = (batch.y3_repeat if proto == "repeat_x1" else batch.y3_indep) * ROOTS[s1 % 4]
            groups.append(
                {
                    "source": source_tag,
                    "cond": cond.name,
                    "rot": rot,
                    "qtype": f"Q3_{proto}",
                    "arm": "pre_ref",
                    "prompt": p_q3,
                    "choices": ch_q3,
                    "z1": z1_np,
                    "z2": z2_np,
                    "phase_slots": phase_mat,
                    "ref_feats_list": [],
                    "labels": sector_of_complex(matched_sum(y3_rot)),
                }
            )

        # 3. Q2 across reference arms
        y_q2 = (batch.x1 + s1) % 4
        for arm in TRAIN_ARMS:
            refs = batch.references_for_arm(arm, coord_rot=0)
            p_q2, ch_q2 = build_stage3d_prompt("Q2", cond, references=refs, deduplicate=True)
            seen_ids = set()
            r_feats = []
            for r in refs:
                if r.acquisition_id in seen_ids:
                    continue
                seen_ids.add(r.acquisition_id)
                r_feats.append(build_ref_feature(r))
            groups.append(
                {
                    "source": source_tag,
                    "cond": cond.name,
                    "rot": rot,
                    "qtype": "Q2",
                    "arm": arm,
                    "prompt": p_q2,
                    "choices": ch_q2,
                    "z1": z1_np,
                    "z2": z2_np,
                    "phase_slots": phase_mat,
                    "ref_feats_list": r_feats,
                    "labels": y_q2,
                }
            )

        # 4. Q3_after_pilot valid
        refs_v = batch.references_for_arm("valid", coord_rot=0)
        p_q3a, ch_q3a = build_stage3d_prompt("Q3_after_pilot", cond, protocol="repeat_x1", references=refs_v, deduplicate=True)
        r_feats_v = [build_ref_feature(refs_v[0])]
        y_q3_rep = sector_of_complex(matched_sum(batch.y3_repeat * ROOTS[s1 % 4]))
        groups.append(
            {
                "source": source_tag,
                "cond": cond.name,
                "rot": rot,
                "qtype": "Q3_after_pilot",
                "arm": "valid",
                "prompt": p_q3a,
                "choices": ch_q3a,
                "z1": z1_np,
                "z2": z2_np,
                "phase_slots": phase_mat,
                "ref_feats_list": r_feats_v,
                "labels": y_q3_rep,
            }
        )


def _encode_batch_groups(encoder, batch: MotherBatch) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
    return _encode_batch_rotations(encoder, batch)


def build_stage3d_train_dev_groups(
    encoder,
    zero_arm: bool,
    train_blocks: int = 8,
    dev_blocks: int = 2,
) -> Tuple[List[dict], List[dict]]:
    """Build identical grouped tasks for phase and zero arms on train and dev splits."""
    tr_groups: List[dict] = []
    dv_groups: List[dict] = []

    # 1. Stage 3 C1/C2/C3 replay
    for cname in REPLAY_TRAIN_CONDITIONS:
        _append_batch_groups(tr_groups, generate_mothers("s3_train", cname, max_blocks=train_blocks), encoder, zero_arm, "replay")
        _append_batch_groups(dv_groups, generate_mothers("s3_dev", cname, max_blocks=dev_blocks), encoder, zero_arm, "replay")

    # 2. Signed CFO supervision across both timing configs
    for t_idx, (h, ell, tp) in enumerate(TRAIN_TIMINGS):
        tr_map, _ = generate_paired_cfo_blocks(
            NS_CFO_TRAIN, t_idx, train_blocks, h, ell, tp, TRAIN_OMEGAS, split_prefix="cfo_train"
        )
        for w in TRAIN_OMEGAS:
            _append_batch_groups(tr_groups, tr_map[float(w)], encoder, zero_arm, "cfo")

        dv_map, _ = generate_paired_cfo_blocks(
            NS_CFO_DEV, t_idx, dev_blocks, h, ell, tp, DEV_OMEGAS, split_prefix="cfo_dev"
        )
        for w in DEV_OMEGAS:
            _append_batch_groups(dv_groups, dv_map[float(w)], encoder, zero_arm, "cfo")

    return tr_groups, dv_groups


def _score_group_with_host(
    host: QwenCacheHost,
    connector: SignedPhaseCacheConnector,
    g: dict,
    st: int,
    ed: int,
    scorer_mode: str = "batched_uncached",
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    z1 = torch.tensor(g["z1"][st:ed], dtype=torch.long)
    z2 = torch.tensor(g["z2"][st:ed], dtype=torch.long)
    ps = torch.tensor(g["phase_slots"][st:ed], dtype=torch.float32)
    r_list = [torch.tensor(rf[st:ed], dtype=torch.float32) for rf in g["ref_feats_list"]]
    pref = host.build_prefix_embeds(connector, g["prompt"], z1, z2, ps, r_list)
    if scorer_mode == "batched_uncached":
        return host.score_candidates_batched_uncached(pref, g["choices"], temperature=temperature)
    if scorer_mode == "kv_cached":
        return host.score_candidates_kv(pref, g["choices"], temperature=temperature)
    if scorer_mode == "naive_uncached":
        return host.score_candidates_naive_uncached(pref, g["choices"], temperature=temperature)
    raise ValueError(f"Unknown scorer_mode={scorer_mode}")


def run_smoke_audit(
    repo_root: Path,
    out_dir: Path,
    model_id: str = MODEL_ID_DEFAULT,
    revision: str = PINNED_REVISION_DEFAULT,
) -> dict:
    """Step 1 Smoke Audit: load Qwen3.5-4B, verify tokenizations, scorer parity, grads, reload, and timing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    host = QwenCacheHost(model_id=model_id, revision=revision)
    load_s = time.perf_counter() - t0

    if host.device.type == "cuda":
        model_vram_mb = float(torch.cuda.memory_allocated(host.device)) / (1024.0 ** 2)
        torch.cuda.reset_peak_memory_stats(host.device)
    else:
        model_vram_mb = 0.0

    encoder, enc_sha = load_s17_encoder(repo_root)
    torch.manual_seed(CONNECTOR_SEEDS[0])
    connector = SignedPhaseCacheConnector(
        num_states=int(len(encoder.table)),
        hidden_size=host.hidden_size,
        target_embed_std=host.target_embed_std,
    ).to(host.device)

    # Build small dev slice (Q3 CFO + Q2 C3)
    dv_cfo_map, _ = generate_paired_cfo_blocks(NS_CFO_DEV, 0, 1, 2, 2, 4.0, (0.18,), split_prefix="smoke_cfo")
    dv_c3 = generate_mothers("s3_dev", "C3_mid_pilot_cont", max_blocks=1)
    groups: List[dict] = []
    _append_batch_groups(groups, dv_cfo_map[0.18], encoder, zero_arm=False, source_tag="cfo")
    _append_batch_groups(groups, dv_c3, encoder, zero_arm=False, source_tag="replay")

    q3_grp = [g for g in groups if g["source"] == "cfo" and g["qtype"] == "Q3_repeat_x1"][0]
    q2_grp = [g for g in groups if g["source"] == "replay" and g["qtype"] == "Q2" and g["arm"] == "valid"][0]

    # 1. Compare naive_uncached vs batched_uncached vs kv_cached on b=4 for both Q3 and Q2
    connector.eval()
    parity = {}
    with torch.no_grad():
        for tag, grp in (("Q3_repeat", q3_grp), ("Q2_valid", q2_grp)):
            z1 = torch.tensor(grp["z1"][:4], dtype=torch.long)
            z2 = torch.tensor(grp["z2"][:4], dtype=torch.long)
            ps = torch.tensor(grp["phase_slots"][:4], dtype=torch.float32)
            r_list = [torch.tensor(rf[:4], dtype=torch.float32) for rf in grp["ref_feats_list"]]
            pref = host.build_prefix_embeds(connector, grp["prompt"], z1, z2, ps, r_list)
            lp_n, ll_n = host.score_candidates_naive_uncached(pref, grp["choices"])
            lp_b, ll_b = host.score_candidates_batched_uncached(pref, grp["choices"])
            lp_k, ll_k = host.score_candidates_kv(pref, grp["choices"])
            parity[tag] = {
                "prefix_len_tokens": int(pref.shape[1]),
                "batched_vs_naive_max_ll_diff": float((ll_b - ll_n).abs().max().item()),
                "batched_vs_naive_max_prob_diff": float((lp_b.exp() - lp_n.exp()).abs().max().item()),
                "kv_vs_naive_max_ll_diff": float((ll_k - ll_n).abs().max().item()),
                "kv_vs_naive_max_prob_diff": float((lp_k.exp() - lp_n.exp()).abs().max().item()),
            }

    # 2. Check gradient checkpointing forward parity and accumulate backward separately on Q3 and Q2
    host.set_grad_checkpointing(False)
    with torch.no_grad():
        lp_eval_ref, _ = _score_group_with_host(host, connector, q3_grp, 0, 4, scorer_mode="batched_uncached")
    host.set_grad_checkpointing(True)
    with torch.no_grad():
        lp_train_ref, _ = _score_group_with_host(host, connector, q3_grp, 0, 4, scorer_mode="batched_uncached")
    ckpt_fwd_prob_diff = float((lp_eval_ref.exp() - lp_train_ref.exp()).abs().max().item())
    assert ckpt_fwd_prob_diff < 1e-5, f"Grad-ckpt eval vs train forward mismatch: {ckpt_fwd_prob_diff}"

    connector.train()
    opt = torch.optim.AdamW(connector.parameters(), lr=1.5e-3)
    opt.zero_grad(set_to_none=True)
    t_step0 = time.perf_counter()
    lp_q3, _ = _score_group_with_host(host, connector, q3_grp, 0, 4, scorer_mode="batched_uncached")
    lbl_q3 = torch.tensor(q3_grp["labels"][:4], device=host.device, dtype=torch.long)
    loss_q3 = 0.5 * F.nll_loss(lp_q3, lbl_q3)
    loss_q3.backward()

    lp_q2, _ = _score_group_with_host(host, connector, q2_grp, 0, 4, scorer_mode="batched_uncached")
    lbl_q2 = torch.tensor(q2_grp["labels"][:4], device=host.device, dtype=torch.long)
    loss_q2 = 0.5 * F.nll_loss(lp_q2, lbl_q2)
    loss_q2.backward()
    loss = loss_q3.detach() + loss_q2.detach()
    assert torch.isfinite(loss), "Non-finite smoke loss!"

    host_grads_nonzero = sum(1 for p in host.lm.parameters() if p.grad is not None)
    conn_grad_norms = {
        n: float(p.grad.detach().float().norm().item()) if p.grad is not None else 0.0
        for n, p in connector.named_parameters()
    }
    conn_grads_zero = [n for n, v in conn_grad_norms.items() if not math.isfinite(v) or v == 0.0]
    assert host_grads_nonzero == 0, f"Frozen host received gradients: {host_grads_nonzero}"
    assert len(conn_grads_zero) == 0, f"Connector params missing gradients: {conn_grads_zero}"
    opt.step()
    smoke_step_ms = (time.perf_counter() - t_step0) * 1000.0
    host.set_grad_checkpointing(False)

    sha_after = _compute_host_fingerprint(host.lm)
    assert sha_after == host.host_sha256, "Frozen host weights mutated!"

    # 3. Checkpoint save and clean reload parity
    connector.eval()
    ckpt_path = out_dir / "smoke_ckpt.pt"
    torch.save({"connector_config": connector.connector_config(), "state_dict": connector.state_dict()}, ckpt_path)
    loaded = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    conn2 = SignedPhaseCacheConnector(**loaded["connector_config"]).to(host.device).eval()
    conn2.load_state_dict(loaded["state_dict"])
    with torch.no_grad():
        lp_before, _ = _score_group_with_host(host, connector, q3_grp, 0, 4, scorer_mode="batched_uncached")
        lp_after, _ = _score_group_with_host(host, conn2, q3_grp, 0, 4, scorer_mode="batched_uncached")
        reload_max_prob_diff = float((lp_before.exp() - lp_after.exp()).abs().max().item())
    assert reload_max_prob_diff == 0.0, f"Reload mismatch: {reload_max_prob_diff}"

    # 4. Greedy Chinese generation on 1 Q3 and 1 Q2 sample
    with torch.no_grad():
        pref_g3 = host.build_prefix_embeds(
            connector,
            q3_grp["prompt"],
            torch.tensor(q3_grp["z1"][:1], dtype=torch.long),
            torch.tensor(q3_grp["z2"][:1], dtype=torch.long),
            torch.tensor(q3_grp["phase_slots"][:1], dtype=torch.float32),
            [],
        )
        gen_q3 = host.generate_text(pref_g3, max_new_tokens=32)

    # 5. Microbatch throughput & VRAM sweep on training + inference
    bench_train = {}
    host.set_grad_checkpointing(True)
    for mb in (4, 8, 16, 32):
        if host.device.type != "cuda" and mb > 4:
            continue
        try:
            if host.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(host.device)
            connector.train()
            opt.zero_grad(set_to_none=True)
            t_b0 = time.perf_counter()
            lp_b, _ = _score_group_with_host(host, connector, q3_grp, 0, mb, scorer_mode="batched_uncached")
            l_b = F.nll_loss(lp_b, torch.tensor(q3_grp["labels"][:mb], device=host.device, dtype=torch.long))
            l_b.backward()
            opt.step()
            if host.device.type == "cuda":
                torch.cuda.synchronize()
                pk = float(torch.cuda.max_memory_allocated(host.device)) / (1024.0 ** 2)
            else:
                pk = 0.0
            bench_train[f"mb_{mb}"] = {
                "step_ms": round((time.perf_counter() - t_b0) * 1000.0, 1),
                "peak_vram_mb": round(pk, 1),
            }
        except torch.cuda.OutOfMemoryError:
            bench_train[f"mb_{mb}"] = {"oom": True}
            if host.device.type == "cuda":
                torch.cuda.empty_cache()
            break
    host.set_grad_checkpointing(False)

    bench_infer = {}
    connector.eval()
    for mode in ("batched_uncached", "kv_cached"):
        for ib in (32, 64):
            if host.device.type != "cuda" and ib > 32:
                continue
            try:
                if host.device.type == "cuda":
                    torch.cuda.empty_cache()
                with torch.no_grad():
                    # Warmup
                    _score_group_with_host(host, connector, q3_grp, 0, min(ib, len(q3_grp["z1"])), scorer_mode=mode)
                    if host.device.type == "cuda":
                        torch.cuda.synchronize()
                    t_i0 = time.perf_counter()
                    n_eff = min(ib, len(q3_grp["z1"]))
                    for _ in range(3):
                        _score_group_with_host(host, connector, q3_grp, 0, n_eff, scorer_mode=mode)
                    if host.device.type == "cuda":
                        torch.cuda.synchronize()
                    ms_ex = ((time.perf_counter() - t_i0) * 1000.0) / (3 * n_eff)
                bench_infer[f"{mode}_b{ib}"] = {"ms_per_example": round(ms_ex, 2)}
            except torch.cuda.OutOfMemoryError:
                bench_infer[f"{mode}_b{ib}"] = {"oom": True}
                if host.device.type == "cuda":
                    torch.cuda.empty_cache()

    gpu_name = torch.cuda.get_device_name(host.device) if host.device.type == "cuda" else "CPU"
    total_vram_mb = (
        float(torch.cuda.get_device_properties(host.device).total_memory) / (1024.0 ** 2)
        if host.device.type == "cuda"
        else 0.0
    )

    # Recommend largest OOM-free micro_batch_size staying comfortably under 80% VRAM
    rec_mb = 4
    for mb in (32, 16, 8, 4):
        rec = bench_train.get(f"mb_{mb}", {})
        if rec and not rec.get("oom", False) and (total_vram_mb == 0.0 or rec.get("peak_vram_mb", 0.0) < 0.80 * total_vram_mb):
            rec_mb = mb
            break

    report = {
        "model_id": model_id,
        "requested_revision": revision,
        "resolved_hf_sha": host.hf_sha,
        "transformers_version": host.transformers_version,
        "torch_version": torch.__version__,
        "encoder_name": "equivariant_b8_s17",
        "encoder_sha256": enc_sha,
        "gpu_name": gpu_name,
        "total_vram_mb": round(total_vram_mb, 1),
        "dtype": str(host.dtype),
        "load_time_s": round(load_s, 2),
        "model_vram_mb_after_load": round(model_vram_mb, 1),
        "thinking_mode": host.thinking_mode,
        "chat_template_sha256": host.chat_template_sha256,
        "frozen_host_params": host.frozen_param_count,
        "trainable_connector_params": connector.param_count(),
        "host_sha256_before": host.host_sha256,
        "host_sha256_after": sha_after,
        "host_sha_identical": sha_after == host.host_sha256,
        "host_grads_nonzero": host_grads_nonzero,
        "conn_grads_all_nonzero": len(conn_grads_zero) == 0,
        "conn_grad_norms": conn_grad_norms,
        "reload_max_prob_diff": reload_max_prob_diff,
        "scorer_parity": parity,
        "smoke_train_loss": float(loss.item()),
        "smoke_step_ms_b4_dual": round(smoke_step_ms, 1),
        "bench_train_microbatch": bench_train,
        "bench_infer": bench_infer,
        "recommended_micro_batch_size": rec_mb,
        "smoke_generated_sample_q3": gen_q3,
        "candidate_token_audit": host.candidate_token_audit,
    }
    (out_dir / "smoke_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _fit_dev_temperature(raw_ll: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    def _nll(log_t: float) -> float:
        temp = math.exp(log_t)
        s = raw_ll / temp
        s = s - s.max(axis=1, keepdims=True)
        ex = np.exp(s)
        pr = ex / ex.sum(axis=1, keepdims=True)
        p_t = np.clip(pr[np.arange(len(labels)), labels], 1e-12, 1.0)
        return float(-np.log(p_t).mean())

    opt = minimize_scalar(_nll, bounds=(math.log(0.25), math.log(4.0)), method="bounded")
    return float(math.exp(opt.x)), float(opt.fun)


def _score_noref_groups_dedup(
    host: QwenCacheHost,
    connector: SignedPhaseCacheConnector,
    indexed_groups: Sequence[Tuple[int, dict]],
    eval_bs: int,
    scorer_mode: str,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Score no-ref groups by deduplicating `(z1, z2)` across rotations sharing prompt and phase slots."""
    buckets: Dict[Tuple[str, Tuple[str, ...], bytes], List[Tuple[int, dict]]] = {}
    for idx, g in indexed_groups:
        assert len(g["ref_feats_list"]) == 0
        key = (
            str(g["prompt"]),
            tuple(g["choices"]),
            np.asarray(g["phase_slots"][0], dtype=np.float32).tobytes(),
        )
        buckets.setdefault(key, []).append((idx, g))

    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for items in buckets.values():
        rep = items[0][1]
        pair_arrays = [
            np.asarray(g["z1"], dtype=np.int64) * 16 + np.asarray(g["z2"], dtype=np.int64) for _, g in items
        ]
        uniq_codes = np.unique(np.concatenate(pair_arrays, axis=0))
        u_grp = {
            "prompt": rep["prompt"],
            "choices": rep["choices"],
            "z1": uniq_codes // 16,
            "z2": uniq_codes % 16,
            "phase_slots": np.repeat(np.asarray(rep["phase_slots"][:1], dtype=np.float32), len(uniq_codes), axis=0),
            "ref_feats_list": [],
        }
        lp_parts, raw_parts = [], []
        for st in range(0, len(uniq_codes), eval_bs):
            ed = min(len(uniq_codes), st + eval_bs)
            lp_u, raw_u = _score_group_with_host(host, connector, u_grp, st, ed, scorer_mode=scorer_mode)
            lp_parts.append(lp_u.detach().cpu().numpy().astype(np.float64))
            raw_parts.append(raw_u.detach().cpu().numpy().astype(np.float64))
        u_lp = np.concatenate(lp_parts, axis=0)
        u_raw = np.concatenate(raw_parts, axis=0)
        code_to_pos = np.full(256, -1, dtype=np.int64)
        code_to_pos[uniq_codes] = np.arange(len(uniq_codes), dtype=np.int64)
        for (idx, _), pairs in zip(items, pair_arrays):
            inv = code_to_pos[pairs]
            out[idx] = (u_lp[inv], u_raw[inv])
    return out


def train_single_arm_seed(
    host: QwenCacheHost,
    encoder,
    arm: str,
    seed: int,
    out_dir: Path,
    epochs: int = 10,
    batch_size: int = 32,
    micro_batch_size: int = 8,
    lr: float = 1.5e-3,
    max_steps_per_epoch: int = 200,
    train_blocks: int = 8,
    dev_blocks: int = 2,
    train_scorer: str = "kv_cached",
    eval_scorer: str = "kv_cached",
    eval_every_epochs: int = 2,
) -> dict:
    """Train one (arm, seed) connector identically across phase and zero arms and calibrate on dev."""
    assert arm in ARM_NAMES
    zero_arm = arm == "zero"
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Identical initialization for matched seed across both arms
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    connector = SignedPhaseCacheConnector(
        num_states=int(len(encoder.table)),
        hidden_size=host.hidden_size,
        target_embed_std=host.target_embed_std,
    ).to(host.device)

    tr_groups, dv_groups = build_stage3d_train_dev_groups(
        encoder=encoder,
        zero_arm=zero_arm,
        train_blocks=train_blocks,
        dev_blocks=dev_blocks,
    )

    # Identical task weighting & minibatch ordering for both arms
    tr_minibatches = []
    weighted_pool = []
    for gi, g in enumerate(tr_groups):
        n = len(g["labels"])
        if g["qtype"] == "Q3_repeat_x1":
            w = 4
        elif g["qtype"] == "Q2" and g["arm"] in ("valid", "fresh_resample"):
            w = 3
        elif g["qtype"] in ("Q1", "Q3_after_pilot"):
            w = 2
        else:
            w = 1
        for st in range(0, n, batch_size):
            mb_idx = len(tr_minibatches)
            tr_minibatches.append((gi, st, min(n, st + batch_size)))
            weighted_pool.extend([mb_idx] * w)

    steps_per_ep = min(len(weighted_pool), int(max_steps_per_epoch))
    total_steps = max(1, epochs * steps_per_ep)
    warmup_steps = max(1, int(round(0.05 * total_steps)))

    def _lr_lambda(step_idx: int) -> float:
        if step_idx < warmup_steps:
            return float(step_idx + 1) / float(warmup_steps)
        prog = min(1.0, float(step_idx - warmup_steps) / float(max(1, total_steps - warmup_steps)))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

    opt = torch.optim.AdamW(connector.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)

    best_score = float("inf")
    best_epoch = 0
    best_state = None
    history = []
    t_start = time.perf_counter()

    for ep in range(1, epochs + 1):
        host.set_grad_checkpointing(train_scorer == "batched_uncached")
        connector.train()
        rng_ep = random.Random(seed + ep * 1009)
        order = list(weighted_pool)
        rng_ep.shuffle(order)
        order = order[:steps_per_ep]

        tr_loss_sum = 0.0
        tr_cnt = 0
        for mb_idx in order:
            gi, st, ed = tr_minibatches[mb_idx]
            g = tr_groups[gi]
            mb_total = ed - st
            opt.zero_grad(set_to_none=True)
            mb_loss = 0.0
            for m_st in range(st, ed, micro_batch_size):
                m_ed = min(ed, m_st + micro_batch_size)
                lp, _ = _score_group_with_host(host, connector, g, m_st, m_ed, scorer_mode=train_scorer)
                lbls = torch.tensor(g["labels"][m_st:m_ed], device=host.device, dtype=torch.long)
                loss = F.nll_loss(lp, lbls, reduction="sum") / float(mb_total)
                assert torch.isfinite(loss), f"Non-finite training loss in {arm}/s{seed} ep={ep}"
                loss.backward()
                mb_loss += float(loss.item()) * float(mb_total)
                del lp, loss, lbls
            torch.nn.utils.clip_grad_norm_(connector.parameters(), 1.0)
            opt.step()
            sched.step()
            tr_loss_sum += mb_loss
            tr_cnt += mb_total

        eval_this_ep = (eval_every_epochs <= 1) or (ep % eval_every_epochs == 0) or (ep >= max(1, epochs - 1))
        if not eval_this_ep:
            rec = {
                "epoch": ep,
                "train_nll": round(tr_loss_sum / max(1, tr_cnt), 5),
                "dev_evaluated": False,
            }
            history.append(rec)
            print(f"[{arm} s={seed}] ep={ep}/{epochs} tr={rec['train_nll']:.4f} (eval skipped)", flush=True)
            continue

        # Evaluate on dev (full deduplicated no-ref + deterministic Block-0 rot=(0,0) subset for with-ref)
        host.set_grad_checkpointing(False)
        connector.eval()
        if host.device.type == "cuda":
            torch.cuda.empty_cache()
        dev_loss_sum = 0.0
        dev_cnt = 0
        cfo_q3_loss = 0.0
        cfo_q3_hits = 0
        cfo_q3_cnt = 0
        q2v_loss = 0.0
        q2v_hits = 0
        q2v_cnt = 0
        eval_bs = 64
        with torch.no_grad():
            noref_idx = [(gi, g) for gi, g in enumerate(dv_groups) if len(g["ref_feats_list"]) == 0]
            noref_scores = _score_noref_groups_dedup(
                host, connector, noref_idx, eval_bs=eval_bs, scorer_mode=eval_scorer
            )
            for gi, g in enumerate(dv_groups):
                n = len(g["labels"])
                if len(g["ref_feats_list"]) == 0:
                    lp_all, _ = noref_scores[gi]
                    lbls_np = np.asarray(g["labels"], dtype=np.int64)
                    ls = float(-np.sum(lp_all[np.arange(n), lbls_np]))
                    hs = int(np.sum(np.argmax(lp_all, axis=-1) == lbls_np))
                    dev_loss_sum += ls
                    dev_cnt += n
                    if g["source"] == "cfo" and g["qtype"] == "Q3_repeat_x1":
                        cfo_q3_loss += ls
                        cfo_q3_hits += hs
                        cfo_q3_cnt += n
                elif tuple(g["rot"]) == (0, 0):
                    n_eval = min(64, n)
                    for st in range(0, n_eval, eval_bs):
                        ed = min(n_eval, st + eval_bs)
                        lp, _ = _score_group_with_host(host, connector, g, st, ed, scorer_mode=eval_scorer)
                        lbls = torch.tensor(g["labels"][st:ed], device=host.device, dtype=torch.long)
                        ls = float(F.nll_loss(lp, lbls, reduction="sum").item())
                        hs = int((lp.argmax(dim=-1) == lbls).sum().item())
                        dev_loss_sum += ls
                        dev_cnt += ed - st
                        if g["qtype"] == "Q2" and g["arm"] == "valid":
                            q2v_loss += ls
                            q2v_hits += hs
                            q2v_cnt += ed - st
                        del lp, lbls
        if host.device.type == "cuda":
            torch.cuda.empty_cache()

        dev_overall_nll = dev_loss_sum / max(1, dev_cnt)
        dev_cfo_q3_nll = cfo_q3_loss / max(1, cfo_q3_cnt)
        dev_cfo_q3_acc = cfo_q3_hits / max(1, cfo_q3_cnt)
        dev_q2v_nll = q2v_loss / max(1, q2v_cnt)
        dev_q2v_acc = q2v_hits / max(1, q2v_cnt)
        sel_score = 0.5 * dev_cfo_q3_nll + 0.5 * dev_overall_nll

        rec = {
            "epoch": ep,
            "train_nll": round(tr_loss_sum / max(1, tr_cnt), 5),
            "dev_evaluated": True,
            "dev_overall_nll": round(dev_overall_nll, 5),
            "dev_cfo_q3_nll": round(dev_cfo_q3_nll, 5),
            "dev_cfo_q3_acc": round(dev_cfo_q3_acc, 4),
            "dev_q2v_nll": round(dev_q2v_nll, 5),
            "dev_q2v_acc": round(dev_q2v_acc, 4),
            "sel_score": round(sel_score, 5),
        }
        history.append(rec)
        print(
            f"[{arm} s={seed}] ep={ep}/{epochs} tr={rec['train_nll']:.4f} "
            f"dev_all={rec['dev_overall_nll']:.4f} cfo_q3={rec['dev_cfo_q3_acc']:.3f}({rec['dev_cfo_q3_nll']:.4f}) "
            f"q2v={rec['dev_q2v_acc']:.3f}({rec['dev_q2v_nll']:.4f})",
            flush=True,
        )
        if sel_score < best_score:
            best_score = sel_score
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in connector.state_dict().items()}

    train_wall_s = time.perf_counter() - t_start
    connector.load_state_dict(best_state)
    connector.eval()
    if host.device.type == "cuda":
        torch.cuda.empty_cache()

    # Fit dev scalar temperatures for Q3_repeat and Q2 across 100% of dv_groups on best checkpoint
    q3_ll_list, q3_lbl_list = [], []
    q2_ll_list, q2_lbl_list = [], []
    eval_bs = 64
    with torch.no_grad():
        calib_noref_idx = [
            (gi, g)
            for gi, g in enumerate(dv_groups)
            if len(g["ref_feats_list"]) == 0
            and (g["qtype"] == "Q3_repeat_x1" or (g["qtype"] == "Q2" and g["arm"] == "none"))
        ]
        calib_noref_scores = _score_noref_groups_dedup(
            host, connector, calib_noref_idx, eval_bs=eval_bs, scorer_mode=eval_scorer
        )
        for gi, g in enumerate(dv_groups):
            n = len(g["labels"])
            if len(g["ref_feats_list"]) == 0 and g["qtype"] == "Q3_repeat_x1":
                _, arr_all = calib_noref_scores[gi]
                q3_ll_list.append(arr_all)
                q3_lbl_list.append(g["labels"])
            elif g["qtype"] == "Q2" and g["arm"] in ("valid", "fresh_resample", "none", "independent"):
                if len(g["ref_feats_list"]) == 0:
                    _, arr_all = calib_noref_scores[gi]
                    q2_ll_list.append(arr_all)
                    q2_lbl_list.append(g["labels"])
                else:
                    for st in range(0, n, eval_bs):
                        ed = min(n, st + eval_bs)
                        _, raw_ll = _score_group_with_host(host, connector, g, st, ed, scorer_mode=eval_scorer)
                        arr_ll = raw_ll.detach().cpu().numpy().astype(np.float64)
                        arr_y = g["labels"][st:ed]
                        q2_ll_list.append(arr_ll)
                        q2_lbl_list.append(arr_y)
                        del raw_ll
    if host.device.type == "cuda":
        torch.cuda.empty_cache()

    temp_q3, dev_q3_cal_nll = _fit_dev_temperature(np.concatenate(q3_ll_list, axis=0), np.concatenate(q3_lbl_list))
    temp_q2, dev_q2_cal_nll = _fit_dev_temperature(np.concatenate(q2_ll_list, axis=0), np.concatenate(q2_lbl_list))

    host_sha_after = _compute_host_fingerprint(host.lm)
    assert host_sha_after == host.host_sha256, "Host mutated during training!"

    ckpt_path = ckpt_dir / f"connector_{arm}_s{seed}.pt"
    torch.save(
        {
            "arm": arm,
            "seed": int(seed),
            "model_id": host.model_id,
            "hf_sha": host.hf_sha,
            "encoder_name": "equivariant_b8_s17",
            "connector_config": connector.connector_config(),
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "best_sel_score": best_score,
            "temp_q3": temp_q3,
            "temp_q2": temp_q2,
            "dev_q3_cal_nll": dev_q3_cal_nll,
            "dev_q2_cal_nll": dev_q2_cal_nll,
        },
        ckpt_path,
    )
    ckpt_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated(host.device)) / (1024.0 ** 2)
        if host.device.type == "cuda"
        else 0.0
    )

    summary = {
        "arm": arm,
        "seed": int(seed),
        "trainable_params": connector.param_count(),
        "epochs": epochs,
        "batch_size": batch_size,
        "micro_batch_size": micro_batch_size,
        "lr": lr,
        "max_steps_per_epoch": max_steps_per_epoch,
        "train_scorer": train_scorer,
        "eval_scorer": eval_scorer,
        "eval_every_epochs": eval_every_epochs,
        "best_epoch": best_epoch,
        "best_sel_score": best_score,
        "temp_q3": temp_q3,
        "temp_q2": temp_q2,
        "dev_q3_cal_nll": dev_q3_cal_nll,
        "dev_q2_cal_nll": dev_q2_cal_nll,
        "train_wall_s": round(train_wall_s, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "host_sha256_after": host_sha_after,
        "checkpoint": str(ckpt_path.name),
        "checkpoint_sha256": ckpt_sha,
        "history": history,
    }
    (out_dir / f"train_summary_{arm}_s{seed}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def score_256_cache_table(
    host: QwenCacheHost,
    connector: SignedPhaseCacheConnector,
    omega: float,
    h: int,
    ell: int,
    t_p: float,
    zero_arm: bool,
    protocol: str = "repeat_x1",
    phase_override_omega: Optional[float] = None,
    batch_size: int = 64,
    scorer_mode: str = "kv_cached",
) -> np.ndarray:
    """Score all 256 discrete (z1, z2) cache pairs at fixed public metadata; returns raw_ll (256, 4)."""
    cond = condition_spec_from_params(omega, h, ell, t_p, snr_db=2.0, sigma_phi=0.10, tag="eval")
    prompt, choices = build_stage3d_prompt("Q3", cond, protocol=protocol)
    w_slot = omega if phase_override_omega is None else phase_override_omega
    p_vec = public_phase_slots(w_slot, h, ell, t_p, zero_arm=zero_arm)

    z1_all = np.arange(256, dtype=np.int64) // 16
    z2_all = np.arange(256, dtype=np.int64) % 16
    ps_all = np.repeat(p_vec[None, :], 256, axis=0)
    grp = {
        "prompt": prompt,
        "choices": choices,
        "z1": z1_all,
        "z2": z2_all,
        "phase_slots": ps_all,
        "ref_feats_list": [],
    }
    chunks = []
    with torch.no_grad():
        for st in range(0, 256, batch_size):
            ed = min(256, st + batch_size)
            _, raw_ll = _score_group_with_host(host, connector, grp, st, ed, scorer_mode=scorer_mode)
            chunks.append(raw_ll.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(chunks, axis=0)


def load_trained_connectors(host: QwenCacheHost, ckpt_dir: Path) -> Dict[Tuple[str, int], Tuple[SignedPhaseCacheConnector, float, float]]:
    loaded = {}
    for arm in ARM_NAMES:
        for seed in CONNECTOR_SEEDS:
            p = ckpt_dir / f"connector_{arm}_s{seed}.pt"
            d = torch.load(p, map_location="cpu", weights_only=True)
            conn = SignedPhaseCacheConnector(**d["connector_config"]).to(host.device).eval()
            conn.load_state_dict(d["state_dict"])
            loaded[(arm, seed)] = (conn, float(d["temp_q3"]), float(d["temp_q2"]))
    return loaded


def eval_dev_pilot_tables(
    host: QwenCacheHost,
    ckpt_dir: Path,
    out_dir: Path,
    batch_size: int = 64,
    scorer_mode: str = "kv_cached",
) -> Path:
    """Score 256x2 tables at dev pilot |omega|=0.18 across all 6 checkpoints for sample-size lock."""
    conns = load_trained_connectors(host, ckpt_dir)
    raw_ll = np.zeros((2, len(CONNECTOR_SEEDS), 2, 256, 4), dtype=np.float64)
    temps_q3 = np.zeros((2, len(CONNECTOR_SEEDS)), dtype=np.float64)
    for a_i, arm in enumerate(ARM_NAMES):
        for s_i, seed in enumerate(CONNECTOR_SEEDS):
            conn, t_q3, _ = conns[(arm, seed)]
            temps_q3[a_i, s_i] = t_q3
            for sign_i, w in enumerate((-0.18, 0.18)):
                raw_ll[a_i, s_i, sign_i] = score_256_cache_table(
                    host, conn, omega=w, h=2, ell=2, t_p=4.0, zero_arm=(arm == "zero"),
                    batch_size=batch_size, scorer_mode=scorer_mode,
                )
    out_path = out_dir / "dev_pilot_tables.npz"
    np.savez_compressed(out_path, pilot_raw_ll=raw_ll, temps_q3=temps_q3)
    return out_path


def eval_blind_gpu(
    repo_root: Path,
    host: QwenCacheHost,
    ckpt_dir: Path,
    out_dir: Path,
    batch_size: int = 64,
    scorer_mode: str = "kv_cached",
) -> dict:
    """Run all blind GPU evaluations: parity check, Q3 256-tables, single-modality contradictions, Q2 C4, and C1/C2/C3 replay."""
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder, _ = load_s17_encoder(repo_root)
    conns = load_trained_connectors(host, ckpt_dir)

    temps_q3 = np.zeros((2, len(CONNECTOR_SEEDS)), dtype=np.float64)
    temps_q2 = np.zeros((2, len(CONNECTOR_SEEDS)), dtype=np.float64)
    for a_i, arm in enumerate(ARM_NAMES):
        for s_i, seed in enumerate(CONNECTOR_SEEDS):
            temps_q3[a_i, s_i] = conns[(arm, seed)][1]
            temps_q2[a_i, s_i] = conns[(arm, seed)][2]

    q3_arrays: Dict[str, np.ndarray] = {"temps_q3": temps_q3}

    # 1. Q3 256-tables for G1..G4 and historical H_c4_035
    eval_cells = [
        ("G1_main_extrap_both", 0.40, 2, 2, 4.0),
        ("G2_interp_both", 0.20, 2, 2, 4.0),
        ("G3_extrap_angle", 0.20, 4, 4, 8.0),
        ("G4_extrap_freq", 0.40, 1, 1, 2.0),
        ("H_c4_035", 0.35, 2, 2, 4.0),
    ]
    for cid, w_abs, h, ell, tp in eval_cells:
        arr = np.zeros((2, len(CONNECTOR_SEEDS), 2, 256, 4), dtype=np.float64)
        for a_i, arm in enumerate(ARM_NAMES):
            for s_i, seed in enumerate(CONNECTOR_SEEDS):
                conn = conns[(arm, seed)][0]
                for sign_i, sign in enumerate((-1, 1)):
                    arr[a_i, s_i, sign_i] = score_256_cache_table(
                        host, conn, omega=sign * w_abs, h=h, ell=ell, t_p=tp, zero_arm=(arm == "zero"),
                        batch_size=batch_size, scorer_mode=scorer_mode,
                    )
        q3_arrays[f"{cid}__raw_ll"] = arr
        print(f"[BLIND Q3 TABLE] Finished {cid}", flush=True)

    # 2. G1 independent_x3 control + single-modality contradiction probes (on phase arm)
    indep_arr = np.zeros((2, len(CONNECTOR_SEEDS), 2, 256, 4), dtype=np.float64)
    contra_text_arr = np.zeros((len(CONNECTOR_SEEDS), 2, 256, 4), dtype=np.float64)
    contra_phase_arr = np.zeros((len(CONNECTOR_SEEDS), 2, 256, 4), dtype=np.float64)
    for a_i, arm in enumerate(ARM_NAMES):
        for s_i, seed in enumerate(CONNECTOR_SEEDS):
            conn = conns[(arm, seed)][0]
            for sign_i, sign in enumerate((-1, 1)):
                indep_arr[a_i, s_i, sign_i] = score_256_cache_table(
                    host, conn, omega=sign * 0.40, h=2, ell=2, t_p=4.0, zero_arm=(arm == "zero"),
                    protocol="independent_x3", batch_size=batch_size, scorer_mode=scorer_mode,
                )
                if arm == "phase":
                    # contra_text_flipped[s_i, sign_i]: phase slots match sign_i, text flipped to -sign
                    contra_text_arr[s_i, sign_i] = score_256_cache_table(
                        host, conn, omega=-sign * 0.40, h=2, ell=2, t_p=4.0, zero_arm=False,
                        phase_override_omega=sign * 0.40, batch_size=batch_size, scorer_mode=scorer_mode,
                    )
                    # contra_phase_flipped[s_i, sign_i]: text matches sign_i, phase slots flipped to -sign
                    contra_phase_arr[s_i, sign_i] = score_256_cache_table(
                        host, conn, omega=sign * 0.40, h=2, ell=2, t_p=4.0, zero_arm=False,
                        phase_override_omega=-sign * 0.40, batch_size=batch_size, scorer_mode=scorer_mode,
                    )
    q3_arrays["G1_main_extrap_both__indep_raw_ll"] = indep_arr
    q3_arrays["G1_main_extrap_both__contra_text_flipped_raw_ll"] = contra_text_arr
    q3_arrays["G1_main_extrap_both__contra_phase_flipped_raw_ll"] = contra_phase_arr

    # 3. Direct forward vs 256-table parity verification (exact matched batch_size + cross-batch b=16 diagnostic)
    g1_spec = BLIND_CELLS["G1_main_extrap_both"]
    g1_sample, _ = generate_paired_cfo_blocks(
        g1_spec.namespace, 0, 1, g1_spec.h, g1_spec.ell, g1_spec.t_p, (-0.40, 0.40), split_prefix="g1_parity"
    )
    n_par = min(64, int(batch_size))
    parity_max_diff = 0.0
    parity_max_prob_diff = 0.0
    parity_records = []
    cross_b16_records = []

    def _np_softmax(arr: np.ndarray) -> np.ndarray:
        s = arr - np.max(arr, axis=-1, keepdims=True)
        e = np.exp(s)
        return e / np.sum(e, axis=-1, keepdims=True)

    for a_i, arm in enumerate(ARM_NAMES):
        conn = conns[(arm, CONNECTOR_SEEDS[0])][0]
        for sign_i, w in enumerate((-0.40, 0.40)):
            batch = g1_sample[w]
            v1, v2 = batch.rotated_view(0, 1)
            z1 = np.asarray(encoder.encode(v1), dtype=np.int64)[:n_par]
            z2 = np.asarray(encoder.encode(v2), dtype=np.int64)[:n_par]
            codes = 16 * z1 + z2
            cond = condition_spec_from_params(w, 2, 2, 4.0, 2.0, 0.10, tag="parity")
            prompt, choices = build_stage3d_prompt("Q3", cond, protocol="repeat_x1")
            ps = np.repeat(public_phase_slots(w, 2, 2, 4.0, zero_arm=(arm == "zero"))[None, :], n_par, axis=0)
            grp = {"prompt": prompt, "choices": choices, "z1": z1, "z2": z2, "phase_slots": ps, "ref_feats_list": []}
            with torch.no_grad():
                _, direct_ll = _score_group_with_host(host, conn, grp, 0, n_par, scorer_mode=scorer_mode)
            direct_np = direct_ll.detach().cpu().numpy().astype(np.float64)
            tbl_np = q3_arrays["G1_main_extrap_both__raw_ll"][a_i, 0, sign_i][codes]
            diff = float(np.max(np.abs(direct_np - tbl_np)))
            pdiff = float(np.max(np.abs(_np_softmax(direct_np) - _np_softmax(tbl_np))))
            agree = float(np.mean(np.argmax(direct_np, axis=-1) == np.argmax(tbl_np, axis=-1)))
            parity_max_diff = max(parity_max_diff, diff)
            parity_max_prob_diff = max(parity_max_prob_diff, pdiff)
            parity_records.append(
                {
                    "arm": arm,
                    "omega": w,
                    "batch_size": n_par,
                    "n_samples": n_par,
                    "max_abs_ll_diff": diff,
                    "max_abs_prob_diff": pdiff,
                    "argmax_agreement": agree,
                }
            )
            if n_par > 16:
                with torch.no_grad():
                    _, dir_16 = _score_group_with_host(host, conn, grp, 0, 16, scorer_mode=scorer_mode)
                d16_np = dir_16.detach().cpu().numpy().astype(np.float64)
                t16_np = tbl_np[:16]
                cross_b16_records.append(
                    {
                        "arm": arm,
                        "omega": w,
                        "max_abs_ll_diff": float(np.max(np.abs(d16_np - t16_np))),
                        "max_abs_prob_diff": float(np.max(np.abs(_np_softmax(d16_np) - _np_softmax(t16_np)))),
                        "argmax_agreement": float(np.mean(np.argmax(d16_np, axis=-1) == np.argmax(t16_np, axis=-1))),
                    }
                )
    assert parity_max_diff < 1e-5, f"Matched-batch direct vs table parity failed: {parity_max_diff}"
    (out_dir / "direct_vs_table_parity.json").write_text(
        json.dumps(
            {
                "matched_batch_size": n_par,
                "total_checked": 4 * n_par,
                "max_abs_ll_diff": parity_max_diff,
                "max_abs_prob_diff": parity_max_prob_diff,
                "details": parity_records,
                "cross_batch_b16_vs_b64": cross_b16_records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(out_dir / "raw_q3_gpu.npz", **q3_arrays)

    # 4. Q2 C4 blind evaluation (512 mothers x 7 arms directly, zero compression!)
    q2_map, _ = generate_paired_cfo_blocks(
        NS_BLIND_Q2_C4, 0, 8, 2, 2, 4.0, (0.35,), snr_db=2.0, sigma_phi=0.10, split_prefix="q2_c4"
    )
    q2_batch = q2_map[0.35]
    z1_q2 = np.asarray(encoder.encode(q2_batch.y1), dtype=np.int64)
    z2_q2 = np.asarray(encoder.encode(q2_batch.y2), dtype=np.int64)
    n_q2 = len(q2_batch.x1)
    q2_raw = np.zeros((2, len(CONNECTOR_SEEDS), len(Q2_BLIND_ARMS), n_q2, 4), dtype=np.float64)
    for a_i, arm in enumerate(ARM_NAMES):
        ps = np.repeat(public_phase_slots(0.35, 2, 2, 4.0, zero_arm=(arm == "zero"))[None, :], n_q2, axis=0)
        for q_i, qarm in enumerate(Q2_BLIND_ARMS):
            refs, dedup_flag = build_q2_arm_references(q2_batch, qarm)
            prompt, choices = build_stage3d_prompt("Q2", q2_batch.condition, references=refs, deduplicate=dedup_flag)
            if dedup_flag:
                seen_ids = set()
                r_feats = []
                for r in refs:
                    if r.acquisition_id in seen_ids:
                        continue
                    seen_ids.add(r.acquisition_id)
                    r_feats.append(build_ref_feature(r))
            else:
                r_feats = [build_ref_feature(r) for r in refs]
            grp = {"prompt": prompt, "choices": choices, "z1": z1_q2, "z2": z2_q2, "phase_slots": ps, "ref_feats_list": r_feats}
            for s_i, seed in enumerate(CONNECTOR_SEEDS):
                conn = conns[(arm, seed)][0]
                chunks = []
                with torch.no_grad():
                    for st in range(0, n_q2, batch_size):
                        ed = min(n_q2, st + batch_size)
                        _, raw_ll = _score_group_with_host(host, conn, grp, st, ed, scorer_mode=scorer_mode)
                        chunks.append(raw_ll.detach().cpu().numpy().astype(np.float64))
                q2_raw[a_i, s_i, q_i] = np.concatenate(chunks, axis=0)
        print(f"[BLIND Q2 C4] Finished arm={arm}", flush=True)
    np.savez_compressed(out_dir / "raw_q2_gpu.npz", q2__raw_ll=q2_raw, temps_q2=temps_q2)

    # 5. Replay C1/C2/C3 forgetting check on s3_test (max_blocks=2)
    rep_arrays = {}
    for cname in REPLAY_TRAIN_CONDITIONS:
        batch = generate_mothers("s3_test", cname, max_blocks=2)
        cond = batch.condition
        n_m = len(batch.x1)
        z1_list, z2_list = [], []
        for rot in ROTATIONS:
            v1, v2 = batch.rotated_view(*rot)
            z1_list.append(np.asarray(encoder.encode(v1), dtype=np.int64))
            z2_list.append(np.asarray(encoder.encode(v2), dtype=np.int64))
        z1_cat = np.concatenate(z1_list)
        z2_cat = np.concatenate(z2_list)
        refs_v = batch.references_for_arm("valid")
        rf_v = [np.tile(build_ref_feature(refs_v[0]), (3, 1))]
        p_q1, ch_q1 = build_stage3d_prompt("Q1", cond)
        p_q3, ch_q3 = build_stage3d_prompt("Q3", cond, protocol="repeat_x1")
        p_q2, ch_q2 = build_stage3d_prompt("Q2", cond, references=refs_v, deduplicate=True)
        for qk in ("q1", "q3", "q2v"):
            rep_arrays[f"{cname}__{qk}_raw_ll"] = np.zeros((2, len(CONNECTOR_SEEDS), 3 * n_m, 4), dtype=np.float64)
        for a_i, arm in enumerate(ARM_NAMES):
            ps = np.repeat(public_phase_slots(cond.omega, cond.h, cond.ell, cond.t_p, zero_arm=(arm == "zero"))[None, :], 3 * n_m, axis=0)
            for s_i, seed in enumerate(CONNECTOR_SEEDS):
                conn = conns[(arm, seed)][0]
                for qk, pr, ch, rfl in (("q1", p_q1, ch_q1, []), ("q3", p_q3, ch_q3, []), ("q2v", p_q2, ch_q2, rf_v)):
                    grp = {"prompt": pr, "choices": ch, "z1": z1_cat, "z2": z2_cat, "phase_slots": ps, "ref_feats_list": rfl}
                    chunks = []
                    with torch.no_grad():
                        for st in range(0, 3 * n_m, batch_size):
                            ed = min(3 * n_m, st + batch_size)
                            _, raw_ll = _score_group_with_host(host, conn, grp, st, ed, scorer_mode=scorer_mode)
                            chunks.append(raw_ll.detach().cpu().numpy().astype(np.float64))
                    rep_arrays[f"{cname}__{qk}_raw_ll"][a_i, s_i] = np.concatenate(chunks, axis=0)
        print(f"[REPLAY TEST] Finished {cname}", flush=True)
    np.savez_compressed(out_dir / "raw_replay_gpu.npz", **rep_arrays)

    return {
        "parity_max_diff": parity_max_diff,
        "q3_file": "raw_q3_gpu.npz",
        "q2_file": "raw_q2_gpu.npz",
        "replay_file": "raw_replay_gpu.npz",
    }
