"""Real-SDR 4B GPU training/eval, W2 Canonical Lie+Schur Soup, and Stage 6 Active EIG."""

from __future__ import annotations

from contextlib import nullcontext
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import t as student_t
import torch
import torch.nn as nn
import torch.nn.functional as F

from research.stage3_ministral_cache.ministral_connector import _compute_host_fingerprint
from research.stage3_ministral_cache.schema import Q1_CHOICES, Q2_CHOICES, Q3_CHOICES, ROOTS
from research.stage3d_qwen_signed_phase.data_stage3d import (
    build_stage3d_prompt,
    condition_spec_from_params,
    load_s17_encoder,
)
from research.stage3d_qwen_signed_phase.qwen_connector import (
    MODEL_ID_DEFAULT,
    PINNED_REVISION_DEFAULT,
    QwenCacheHost,
)
from research.stage4_math_and_design.M5 import verify_m5
from research.stage4_real_and_surgical.equi_connector import (
    ARM_NAMES_STAGE4,
    PhysicalKernelS17,
    Stage4MatchedConnector,
    decode_c16_tail,
    lift_scores,
)
from research.stage4_real_and_surgical.phaseB_surgical import (
    GENERAL_TEXT_EVAL_PROMPTS,
    GENERAL_TEXT_TRAIN_PROMPTS,
    _build_fixed_mix_prefix,
    _cluster_paired_stats,
    _forward_last_token_logits,
    _holm_bonferroni,
    _load_or_build_kfac,
    _make_lr_scheduler,
    audit_pure_and_mix_kl,
    compute_frozen_general_refs,
)
from research.stage4_real_and_surgical.sdr_real_pipeline import (
    _c4_posterior_from_scalar,
    _decode_c16_to_complex_and_var,
    _eval_physical_probs_from_pair,
    load_bundle_npz,
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
)


# =============================================================================
# 1. W2 Theorem 8 Canonical Lie+Schur Convex Soup & Matched LEMMO Mergers
# =============================================================================

def _merge_cayley_lie_pair(
    cay_a,
    cay_b,
    cay_dst,
    w_a: float,
    w_b: float,
) -> float:
    """Merge two FP32ComplexNullspaceCayley modules in canonical skew-Hermitian Lie algebra U(m) ∩ Pi_perp."""
    with torch.no_grad():
        u1, v1 = cay_a._projected_uv()
        u2, v2 = cay_b._projected_uv()
        u1, v1 = u1.detach().cpu().to(torch.complex128), v1.detach().cpu().to(torch.complex128)
        u2, v2 = u2.detach().cpu().to(torch.complex128), v2.detach().cpu().to(torch.complex128)
        m, r_c = u1.shape

        cols = torch.cat((u1, v1, u2, v2), dim=1)  # (m, 4*r_c)
        if float(torch.linalg.norm(cols).item()) < 1e-12:
            cay_dst.u_re.zero_()
            cay_dst.u_im.zero_()
            cay_dst.v_re.zero_()
            cay_dst.v_im.zero_()
            return 0.0

        # Project basis cleanly onto complex nullspace Pi_perp before QR
        if cay_dst.u_c is not None:
            uc = cay_dst.u_c.detach().cpu().to(torch.complex128)
            cols = cols - uc @ (uc.conj().T @ cols)
        else:
            uc = None

        q_b, _ = torch.linalg.qr(cols, mode="reduced")  # (m, p), p <= 4*r_c
        # Re-project to eliminate any numerical leak inside U_c
        if uc is not None:
            q_b = q_b - uc @ (uc.conj().T @ q_b)
        # Keep only orthonormal non-degenerate columns via SVD of q_b
        uq, sq, _ = torch.linalg.svd(q_b, full_matrices=False)
        keep = sq > 1e-10
        if not bool(keep.any()):
            cay_dst.u_re.zero_()
            cay_dst.u_im.zero_()
            cay_dst.v_re.zero_()
            cay_dst.v_im.zero_()
            return 0.0
        q_b = uq[:, keep]  # (m, p_eff)

        u1q, v1q = q_b.conj().T @ u1, q_b.conj().T @ v1
        u2q, v2q = q_b.conj().T @ u2, q_b.conj().T @ v2
        h_small = 1j * (
            float(w_a) * (u1q @ v1q.conj().T - v1q @ u1q.conj().T)
            + float(w_b) * (u2q @ v2q.conj().T - v2q @ u2q.conj().T)
        )
        h_small = 0.5 * (h_small + h_small.conj().T)
        evals, evecs = torch.linalg.eigh(h_small)

        # Pair positive & negative eigenmodes or select top r_c eigenvalues by |lambda|
        order = torch.argsort(torch.abs(evals), descending=True)[:r_c]
        u_out = torch.zeros((m, r_c), dtype=torch.complex128)
        v_out = torch.zeros((m, r_c), dtype=torch.complex128)
        for j, idx_e in enumerate(order.tolist()):
            lam = float(evals[idx_e].item())
            if abs(lam) < 1e-12:
                continue
            eig_vec = q_b @ evecs[:, idx_e]
            amp = math.sqrt(0.5 * abs(lam))
            sgn = 1.0 if lam >= 0.0 else -1.0
            u_j = amp * eig_vec
            v_j = (1j * sgn) * u_j
            u_out[:, j] = u_j
            v_out[:, j] = v_j

        if uc is not None:
            u_out = u_out - uc @ (uc.conj().T @ u_out)
            v_out = v_out - uc @ (uc.conj().T @ v_out)

        dev = cay_dst.u_re.device
        cay_dst.u_re.copy_(u_out.real.to(torch.float32).to(dev))
        cay_dst.u_im.copy_(u_out.imag.to(torch.float32).to(dev))
        cay_dst.v_re.copy_(v_out.real.to(torch.float32).to(dev))
        cay_dst.v_im.copy_(v_out.imag.to(torch.float32).to(dev))

        # Measure exact Woodbury unitary preservation defect on random test batch
        probe = torch.randn((4, 2 * m), dtype=torch.float32, device=dev)
        q_probe = cay_dst(probe)
        norm_defect = float(
            torch.max(torch.abs(q_probe.norm(dim=-1) - probe.norm(dim=-1)) / probe.norm(dim=-1).clamp_min(1e-6)).item()
        )
        return norm_defect


def _merge_lora_weight_space(l_a, l_b, l_dst, w_a: float, w_b: float) -> None:
    """Exact GL(r)-gauge-invariant weight-space SVD convex merge of two NullspaceLoRADelta modules."""
    with torch.no_grad():
        ba = l_a.lora_b.detach().cpu().to(torch.float64) * math.sqrt(float(w_a) * l_a.scale / max(1e-12, l_dst.scale))
        aa = l_a.lora_a.detach().cpu().to(torch.float64) * math.sqrt(float(w_a) * l_a.scale / max(1e-12, l_dst.scale))
        bb = l_b.lora_b.detach().cpu().to(torch.float64) * math.sqrt(float(w_b) * l_b.scale / max(1e-12, l_dst.scale))
        ab = l_b.lora_a.detach().cpu().to(torch.float64) * math.sqrt(float(w_b) * l_b.scale / max(1e-12, l_dst.scale))

        # Compact QR factorization in output & input spaces (no full O(D^2) intermediate matrix)
        b_cat = torch.cat((ba, bb), dim=1)  # (out_eff, 2r)
        a_cat = torch.cat((aa, ab), dim=0)  # (2r, in_feat)
        qb, rb = torch.linalg.qr(b_cat, mode="reduced")
        qa, ra = torch.linalg.qr(a_cat.T, mode="reduced")
        core = rb @ ra.T  # (<=2r, <=2r)
        uc, sc, vhc = torch.linalg.svd(core, full_matrices=False)
        r = l_dst.r
        k_keep = min(r, sc.shape[0])
        s_sqrt = torch.sqrt(sc[:k_keep].clamp_min(0.0))

        new_b = torch.zeros_like(l_dst.lora_b, dtype=torch.float64, device="cpu")
        new_a = torch.zeros_like(l_dst.lora_a, dtype=torch.float64, device="cpu")
        new_b[:, :k_keep] = (qb @ uc[:, :k_keep]) * s_sqrt.unsqueeze(0)
        new_a[:k_keep, :] = s_sqrt.unsqueeze(1) * (vhc[:k_keep, :] @ qa.T)

        l_dst.lora_b.copy_(new_b.to(torch.float32).to(l_dst.lora_b.device))
        l_dst.lora_a.copy_(new_a.to(torch.float32).to(l_dst.lora_a.device))


def merge_surgical_lie_schur_soup(
    adapter_a: SurgicalHostAdapter,
    adapter_b: SurgicalHostAdapter,
    dst: SurgicalHostAdapter,
    w_a: float = 0.5,
) -> dict:
    """W2 Theorem 8 canonical Lie+Schur convex soup with shared `B_align` and `Pi_perp`."""
    w_a = float(w_a)
    w_b = 1.0 - w_a
    b_diff = float(torch.max(torch.abs(adapter_a.schur_readout.b_align.cpu() - adapter_b.schur_readout.b_align.cpu())).item())
    assert b_diff < 1e-6, f"Cannot soup mismatched B_align frames: diff={b_diff}"

    max_unitary_defect = 0.0
    for k in dst.fa_cayleys.keys():
        d_norm = _merge_cayley_lie_pair(adapter_a.fa_cayleys[k], adapter_b.fa_cayleys[k], dst.fa_cayleys[k], w_a, w_b)
        max_unitary_defect = max(max_unitary_defect, d_norm)
    for k in dst.dn_cayleys.keys():
        d_norm = _merge_cayley_lie_pair(adapter_a.dn_cayleys[k], adapter_b.dn_cayleys[k], dst.dn_cayleys[k], w_a, w_b)
        max_unitary_defect = max(max_unitary_defect, d_norm)

    for mdict_name in ("fa_loras", "dn_loras", "dn_gate_loras"):
        da, db, dd = getattr(adapter_a, mdict_name), getattr(adapter_b, mdict_name), getattr(dst, mdict_name)
        for k in dd.keys():
            _merge_lora_weight_space(da[k], db[k], dd[k], w_a, w_b)

    # Exact canonical Schur-Irrep readout linear weight average in fixed B_align gauge (W2 Theorem 8)
    with torch.no_grad():
        wa_mat = adapter_a.schur_readout.proj_coord.detach().cpu().to(torch.float64) @ adapter_a.schur_readout.proj_in.detach().cpu().to(torch.float64)
        wb_mat = adapter_b.schur_readout.proj_coord.detach().cpu().to(torch.float64) @ adapter_b.schur_readout.proj_in.detach().cpu().to(torch.float64)
        w_soup = w_a * wa_mat + w_b * wb_mat  # (3, hidden_size)
        u, s, vh = torch.linalg.svd(w_soup, full_matrices=False)  # (3, 3), (3,), (3, D)
        r = dst.schur_readout.r
        k_keep = min(3, r, s.shape[0])
        s_sqrt = torch.sqrt(s[:k_keep].clamp_min(0.0))
        pc = torch.zeros_like(dst.schur_readout.proj_coord, dtype=torch.float64, device="cpu")
        pi = torch.zeros_like(dst.schur_readout.proj_in, dtype=torch.float64, device="cpu")
        pc[:, :k_keep] = u[:, :k_keep] * s_sqrt.unsqueeze(0)
        pi[:k_keep, :] = s_sqrt.unsqueeze(1) * vh[:k_keep, :]
        dst.schur_readout.proj_coord.copy_(pc.to(torch.float32).to(dst.schur_readout.proj_coord.device))
        dst.schur_readout.proj_in.copy_(pi.to(torch.float32).to(dst.schur_readout.proj_in.device))
        rec_err = float(torch.max(torch.abs((pc @ pi) - w_soup)).item())

    assert max_unitary_defect < 1e-5, f"Unitary probe defect too large: {max_unitary_defect}"
    assert rec_err < 1e-5, f"Schur readout reconstruction error too large: {rec_err}"
    return {
        "w_a": w_a,
        "b_align_diff": b_diff,
        "max_cayley_unitary_norm_defect": max_unitary_defect,
        "schur_readout_reconstruction_err": rec_err,
    }


def merge_lemmo_adapters(
    adapter_a: UnconstrainedLoRAAdapter,
    adapter_b: UnconstrainedLoRAAdapter,
    dst: UnconstrainedLoRAAdapter,
    mode: str = "factor_avg",
    w_a: float = 0.5,
) -> dict:
    """Merge two UnconstrainedLoRAAdapter instances (`factor_avg` vs gauge-invariant `weight_avg`)."""
    w_a = float(w_a)
    w_b = 1.0 - w_a
    with torch.no_grad():
        for mdict_name in ("fa_loras", "dn_loras"):
            da, db, dd = getattr(adapter_a, mdict_name), getattr(adapter_b, mdict_name), getattr(dst, mdict_name)
            for k in dd.keys():
                if mode == "factor_avg":
                    dd[k].lora_a.copy_((w_a * da[k].lora_a.detach() + w_b * db[k].lora_a.detach()).to(dd[k].lora_a.device))
                    dd[k].lora_b.copy_((w_a * da[k].lora_b.detach() + w_b * db[k].lora_b.detach()).to(dd[k].lora_b.device))
                elif mode == "weight_avg":
                    _merge_lora_weight_space(da[k], db[k], dd[k], w_a, w_b)
                else:
                    raise ValueError(f"Unknown merge mode={mode!r}")
    return {"mode": mode, "w_a": w_a}


# =============================================================================
# 2. Real SDR GPU Runner (`Pluto + CTTC` -> Unseen `f708_snr10` Holdout)
# =============================================================================

def _make_real_prompt_set() -> Dict[str, Tuple[str, Tuple[str, ...]]]:
    cond_p = condition_spec_from_params(0.0, 2, 2, 4.0, snr_db=6.0, sigma_phi=0.05, tag="real_sdr")
    # Q1, Q2 (valid/none), Q3 prompts via Stage 3D zero-leakage formatter
    p_q1, ch_q1 = build_stage3d_prompt("Q1", cond_p)
    # Build generic Q2 prompt referencing valid pilots safely
    p_q2_none, ch_q2 = build_stage3d_prompt("Q2", cond_p, references=(), deduplicate=True)
    # Minimal placeholder for non-empty pilot reference notice without numeric leakage
    from research.stage3_ministral_cache.schema import Reference

    dummy_ref = Reference(0, True, np.zeros((1, 3), dtype=np.complex128), 4.0, 6.0)
    p_q2_one, _ = build_stage3d_prompt("Q2", cond_p, references=(dummy_ref,), deduplicate=True)
    p_q2_two, _ = build_stage3d_prompt("Q2", cond_p, references=(dummy_ref, Reference(1, True, np.zeros((1, 3), dtype=np.complex128), 4.0, 6.0)), deduplicate=True)
    p_q3, ch_q3 = build_stage3d_prompt("Q3", cond_p, protocol="repeat_x1")
    return {
        "Q1": (p_q1, ch_q1),
        "Q2_none": (p_q2_none, ch_q2),
        "Q2_one": (p_q2_one, ch_q2),
        "Q2_two": (p_q2_two, ch_q2),
        "Q3": (p_q3, ch_q3),
    }


def _compute_real_teacher_probs(shard: Dict[str, Any], encoder) -> Dict[str, np.ndarray]:
    t1_16, v1_16 = _decode_c16_to_complex_and_var(shard["words_c16"][:, 0], shard["v_norm"])
    t2_16, v2_16 = _decode_c16_to_complex_and_var(shard["words_c16"][:, 1], shard["v_norm"])
    av = shard["avail"]
    a1 = _c4_posterior_from_scalar(t1_16, v1_16, av[:, 0])
    a2 = _c4_posterior_from_scalar(t2_16, v2_16, av[:, 1])
    del encoder
    res = _eval_physical_probs_from_pair(a1, a2, shard)
    out = {"Q1": res["q1"].astype(np.float32)}
    if "ref_two" in res["q2"]:
        out["Q2_two"] = res["q2"]["ref_two"].astype(np.float32)
    if "ref_one" in res["q2"]:
        out["Q2_one"] = res["q2"]["ref_one"].astype(np.float32)
    if "ref_none" in res["q2"]:
        out["Q2_none"] = res["q2"]["ref_none"].astype(np.float32)
    return out


def _shard_to_eval_task(
    shard: Dict[str, Any],
    qtype_key: str,
    ref_key: str,
    prompt_info: Tuple[str, Tuple[str, ...]],
    bits: int,
    label_key: str,
) -> dict:
    words = shard["s17_codes"] if bits == 4 else shard["words_c16"]
    qtype = "Q1" if qtype_key == "Q1" else ("Q2" if qtype_key.startswith("Q2") else "Q3_repeat_x1")
    return {
        "qtype": qtype,
        "prompt": prompt_info[0],
        "choices": prompt_info[1],
        "words": words.astype(np.int64),
        "phase_slots": shard["phase_slots"].astype(np.float32),
        "meta_scalars": shard["meta_scalars"].astype(np.float32),
        "ref_summary": shard[ref_key].astype(np.float32),
        "discrete_d": shard["discrete_d"].astype(np.float32),
        "labels": shard[label_key].astype(np.int64),
        "cluster_ids": shard["cluster_ids"].astype(np.int64),
        "bits": int(bits),
    }


def _predict_real_task(
    host: QwenCacheHost,
    connector: Stage4MatchedConnector,
    task: dict,
    adapter: Optional[nn.Module],
    eval_bs: int,
    scorer_mode: str,
) -> np.ndarray:
    connector.eval()
    if adapter is not None:
        adapter.eval()
    n = len(task["words"])
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for st in range(0, n, eval_bs):
            ed = min(n, st + eval_bs)
            _, _, probs, _ = connector.score_batch(
                host,
                task["prompt"],
                task["choices"],
                task["qtype"],
                torch.from_numpy(task["words"][st:ed]).to(host.device),
                int(task["bits"]),
                torch.from_numpy(task["phase_slots"][st:ed]).to(host.device),
                torch.from_numpy(task["meta_scalars"][st:ed]).to(host.device),
                ref_summary=torch.from_numpy(task["ref_summary"][st:ed]).to(host.device),
                scorer_mode=scorer_mode,
                surgical_adapter=adapter,
                discrete_d=torch.from_numpy(task["discrete_d"][st:ed]).to(host.device),
            )
            chunks.append(probs.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(chunks, axis=0)


def _eval_real_domains(
    host: QwenCacheHost,
    connector: Stage4MatchedConnector,
    domains: Dict[str, Dict[str, dict]],
    adapter: Optional[nn.Module],
    bits: int,
    prompts: Dict[str, Tuple[str, Tuple[str, ...]]],
    eval_bs: int,
    scorer_mode: str,
) -> Tuple[dict, Dict[str, dict]]:
    summary: Dict[str, Any] = {}
    raw_arrays: Dict[str, dict] = {}
    for dname, tasks in domains.items():
        d_rep: Dict[str, Any] = {}
        raw_arrays[dname] = {}
        for tname, (qkey, rkey, lkey) in tasks.items():
            # Rebuild task tensor view using requested `bits` (4 vs 16)
            sh = qkey["shard"]
            task_d = _shard_to_eval_task(sh, qkey["qtype_key"], rkey, prompts[qkey["prompt_key"]], bits=bits, label_key=lkey)
            probs = _predict_real_task(host, connector, task_d, adapter=adapter, eval_bs=eval_bs, scorer_mode=scorer_mode)
            ys = task_d["labels"]
            nll = -np.log(np.clip(probs[np.arange(len(ys)), ys], 1e-12, 1.0))
            hits = (probs.argmax(axis=-1) == ys).astype(np.float64)
            d_rep[tname] = {"n": int(len(ys)), "acc": float(np.mean(hits)), "nll": float(np.mean(nll))}
            raw_arrays[dname][tname] = {"hits": hits, "nll": nll, "clusters": task_d["cluster_ids"], "probs": probs}
        if "q2_one" in d_rep and "q2_two" in d_rep:
            d_rep["q2_two_minus_one_acc"] = d_rep["q2_two"]["acc"] - d_rep["q2_one"]["acc"]
            d_rep["q2_two_minus_one_nll"] = d_rep["q2_two"]["nll"] - d_rep["q2_one"]["nll"]
        summary[dname] = d_rep
    return summary, raw_arrays


def run_stage4_real_sdr_gpu(
    repo_root: Path,
    cache_in_dir: Path,
    out_dir: Path,
    ckpt_dir: Path,
    kfac_path: Path,
    seed: int = 202611,
    steps_adapt: int = 96,
    steps_soup: int = 48,
    batch_size: int = 32,
    micro_batch_size: int = 16,
    eval_bs: int = 128,
    scorer_mode: str = "kv_cached",
    model_id: str = MODEL_ID_DEFAULT,
    revision: str = PINNED_REVISION_DEFAULT,
    mock_tiny_for_test: bool = False,
    host: Optional[QwenCacheHost] = None,
    cache_dir: Optional[Path] = None,
    on_substage_done: Optional[Callable[[str], None]] = None,
) -> dict:
    """Execute Real-SDR zero-shot 4b transfer, 16-bit vs 4-bit adaptation, and W2 Lie+Schur cross-domain soup."""
    t_all = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    if host is None:
        host = QwenCacheHost(model_id=model_id, revision=revision, mock_tiny_for_test=mock_tiny_for_test)
    if not mock_tiny_for_test:
        assert int(getattr(host.text_config, "vocab_size", 0)) == EXPECTED_4B_VOCAB
        assert host.host_sha256 == EXPECTED_4B_HOST_SHA256

    encoder, _ = load_s17_encoder(repo_root)
    pluto_bundle = load_bundle_npz(cache_in_dir / "pluto_dual_cache_c16.npz")
    cttc_bundle = load_bundle_npz(cache_in_dir / "cttc_dual_cache_c16.npz")

    # Separate CTTC strictly into in-domain vs held-out f708_snr10_QPSK!
    def _filter_shard(sh: Dict[str, np.ndarray], mask: np.ndarray) -> Dict[str, np.ndarray]:
        return {k: v[mask] for k, v in sh.items() if isinstance(v, np.ndarray)}

    cttc_tr_in = _filter_shard(cttc_bundle["splits"]["train"], ~cttc_bundle["splits"]["train"]["is_holdout"])
    cttc_te_in = _filter_shard(cttc_bundle["splits"]["test"], ~cttc_bundle["splits"]["test"]["is_holdout"])
    cttc_te_ho = _filter_shard(cttc_bundle["splits"]["test"], cttc_bundle["splits"]["test"]["is_holdout"])
    pluto_tr = pluto_bundle["splits"]["train"]
    pluto_te = pluto_bundle["splits"]["test"]
    pluto_stress = pluto_bundle["stress"]
    cttc_stress_in = _filter_shard(cttc_bundle["stress"], ~cttc_bundle["stress"]["is_holdout"])

    prompts = _make_real_prompt_set()

    # Primary full-coverage eval suite map across domains + Blackwell q2_none/q2_one/q2_two staircase
    domains = {
        "pluto_test": {
            "q1": ({"shard": pluto_te, "qtype_key": "Q1", "prompt_key": "Q1"}, "ref_none", "labels_q1"),
            "q2_two": ({"shard": pluto_te, "qtype_key": "Q2", "prompt_key": "Q2_two"}, "ref_two", "labels_q2"),
        },
        "cttc_in_domain_test": {
            "q1": ({"shard": cttc_te_in, "qtype_key": "Q1", "prompt_key": "Q1"}, "ref_none", "labels_q1"),
            "q2_two": ({"shard": cttc_te_in, "qtype_key": "Q2", "prompt_key": "Q2_two"}, "ref_two", "labels_q2"),
        },
        "cttc_unseen_holdout_f708_snr10": {
            "q1": ({"shard": cttc_te_ho, "qtype_key": "Q1", "prompt_key": "Q1"}, "ref_none", "labels_q1"),
            "q2_two": ({"shard": cttc_te_ho, "qtype_key": "Q2", "prompt_key": "Q2_two"}, "ref_two", "labels_q2"),
        },
        "pluto_stress": {
            "q2_two": ({"shard": pluto_stress, "qtype_key": "Q2", "prompt_key": "Q2_two"}, "ref_two", "labels_q2"),
        },
        "cttc_stress": {
            "q2_two": ({"shard": cttc_stress_in, "qtype_key": "Q2", "prompt_key": "Q2_two"}, "ref_two", "labels_q2"),
        },
    }

    blackwell_missing_domains = {
        dom_k: {
            "q2_none": ({"shard": dom_sh, "qtype_key": "Q2", "prompt_key": "Q2_none"}, "ref_none", "labels_q2"),
            "q2_one": ({"shard": dom_sh, "qtype_key": "Q2", "prompt_key": "Q2_one"}, "ref_one", "labels_q2"),
        }
        for dom_k, dom_sh in (
            ("pluto_test", pluto_te),
            ("cttc_in_domain_test", cttc_te_in),
            ("cttc_unseen_holdout_f708_snr10", cttc_te_ho),
        )
    }

    def _eval_blackwell_staircase(
        conn_mod: Stage4MatchedConnector,
        adapter_mod: Optional[nn.Module],
        bits_val: int,
        base_summary: dict,
    ) -> dict:
        bw_part, _ = _eval_real_domains(
            host, conn_mod, blackwell_missing_domains, adapter=adapter_mod, bits=bits_val, prompts=prompts, eval_bs=eval_bs, scorer_mode=scorer_mode
        )
        for dom_k, d_vals in bw_part.items():
            q2_two_entry = base_summary[dom_k]["q2_two"]
            d_vals["q2_two"] = q2_two_entry
            d_vals["q2_one_minus_none_acc"] = d_vals["q2_one"]["acc"] - d_vals["q2_none"]["acc"]
            d_vals["q2_one_minus_none_nll"] = d_vals["q2_one"]["nll"] - d_vals["q2_none"]["nll"]
            d_vals["q2_two_minus_one_acc"] = q2_two_entry["acc"] - d_vals["q2_one"]["acc"]
            d_vals["q2_two_minus_one_nll"] = q2_two_entry["nll"] - d_vals["q2_one"]["nll"]
            d_vals["q2_two_minus_none_acc"] = q2_two_entry["acc"] - d_vals["q2_none"]["acc"]
            d_vals["q2_two_minus_none_nll"] = q2_two_entry["nll"] - d_vals["q2_none"]["nll"]
        return bw_part

    layers = _get_decoder_layers(host.lm)
    n_l = len(layers)
    fa_l = [i for i in QWEN4B_FA_LAYERS if i < n_l] or ([n_l - 1] if n_l >= 1 else [])
    dn_l = [i for i in QWEN4B_DN_LAYERS if i < n_l]
    stop_l = min(QWEN4B_STOP_LAYER, max(0, n_l - 2))
    proj_map = _load_or_build_kfac(host, kfac_path, fa_l, dn_l, mock_tiny_for_test)
    b_align = calibrate_b_align_from_host(host)

    def _load_phaseA_conn(arm: str) -> Stage4MatchedConnector:
        mod = Stage4MatchedConnector(arm, encoder.table, hidden_size=host.hidden_size).to(host.device)
        p = ckpt_dir / f"stage4_phaseA_{arm}_s{seed}_step384.pt"
        if p.exists():
            mod.load_state_dict(torch.load(p, map_location=host.device, weights_only=False)["state_dict"])
        elif not mock_tiny_for_test:
            raise FileNotFoundError(f"Missing {p}")
        return mod

    def _cached_substage(key: str, fn: Callable[[], dict]) -> dict:
        cp = (cache_dir / f"{key}.pt") if cache_dir is not None else None
        if cp is not None and cp.exists():
            ck = torch.load(cp, map_location="cpu", weights_only=False)
            if isinstance(ck, dict) and ck.get("host_sha256") == host.host_sha256 and ck.get("seed") == seed:
                return ck["data"]
        val = fn()
        if cp is not None:
            tmp = cache_dir / f".{key}.tmp"
            torch.save({"host_sha256": host.host_sha256, "seed": seed, "data": val}, tmp)
            tmp.replace(cp)
            if on_substage_done is not None:
                on_substage_done(key)
        return val

    # --- Substage 1: Zero-Shot 4-bit Transfer of Existing Stage 4 Checkpoints ---
    def _run_sub1() -> dict:
        sub1: Dict[str, Any] = {}
        for arm in ARM_NAMES_STAGE4:
            conn = _load_phaseA_conn(arm)
            s_rep, _ = _eval_real_domains(host, conn, domains, adapter=None, bits=4, prompts=prompts, eval_bs=eval_bs, scorer_mode=scorer_mode)
            sub1[arm] = s_rep
            del conn
        pure_c = _load_phaseA_conn("equivariant_host_pure")
        surg_p = ckpt_dir / f"stage4_phaseB_surgical96_s{seed}.pt"
        lemmo_p = ckpt_dir / f"stage4_phaseB_lemmo_tier_p1_s{seed}.pt"
        if surg_p.exists() or mock_tiny_for_test:
            s_ck = torch.load(surg_p, map_location="cpu", weights_only=False) if surg_p.exists() else None
            ba = s_ck["b_align"] if s_ck is not None else b_align
            surg = SurgicalHostAdapter(host.lm, fa_layers=fa_l, dn_layers=dn_l, stop_layer=stop_l, mode="mode1_stopgrad_l10", u_c_by_layer=proj_map, b_align=ba).to(host.device)
            try:
                if s_ck is not None:
                    surg.load_state_dict(s_ck["state_dict"])
                sub1["mode1_surgical_96"], _ = _eval_real_domains(host, pure_c, domains, adapter=surg, bits=4, prompts=prompts, eval_bs=eval_bs, scorer_mode=scorer_mode)
            finally:
                surg.remove()
                del surg
        if lemmo_p.exists() or mock_tiny_for_test:
            l_ck = torch.load(lemmo_p, map_location="cpu", weights_only=False) if lemmo_p.exists() else None
            lem = UnconstrainedLoRAAdapter(host.lm, fa_layers=fa_l, dn_layers=dn_l, stop_layer=stop_l, r_lora=8).to(host.device)
            try:
                if l_ck is not None:
                    lem.load_state_dict(l_ck["state_dict"])
                sub1["lemmo_tier_p1"], _ = _eval_real_domains(host, pure_c, domains, adapter=lem, bits=4, prompts=prompts, eval_bs=eval_bs, scorer_mode=scorer_mode)
            finally:
                lem.remove()
                del lem
        del pure_c
        return sub1

    zero_shot_rep = _cached_substage("real_sub1_zeroshot", _run_sub1)
    print(f"[RealSDR Sub1] zero-shot 4b Pluto pure Q1={zero_shot_rep['equivariant_host_pure']['pluto_test']['q1']['acc']:.4f}", flush=True)

    # --- Build Real Train Batches (Pluto Train + CTTC In-Domain Train) ---
    pluto_teach = _compute_real_teacher_probs(pluto_tr, encoder)
    cttc_teach = _compute_real_teacher_probs(cttc_tr_in, encoder)

    def _build_train_tasks(sh: Dict[str, np.ndarray], teach: Dict[str, np.ndarray], bits: int, tag: str) -> List[dict]:
        t_list = []
        for qkey, rkey, lkey, pk in (("Q1", "ref_none", "labels_q1", "Q1"), ("Q2_two", "ref_two", "labels_q2", "Q2_two"), ("Q2_one", "ref_one", "labels_q2", "Q2_one")):
            td = _shard_to_eval_task(sh, qkey, rkey, prompts[pk], bits=bits, label_key=lkey)
            td["teacher_p"] = teach[qkey]
            td["domain"] = tag
            t_list.append(td)
        return t_list

    def _train_connector_real(arm: str, bits: int, steps: int) -> Tuple[Stage4MatchedConnector, dict]:
        torch.manual_seed(seed + (16 if bits == 16 else 4))
        rng = random.Random(seed + len(arm) + bits)
        conn = _load_phaseA_conn(arm)
        conn.train()
        opt = torch.optim.AdamW(conn.parameters(), lr=5e-4, weight_decay=1e-4)
        sched = _make_lr_scheduler(opt, steps)
        tr_tasks = _build_train_tasks(pluto_tr, pluto_teach, bits=bits, tag="pluto") + _build_train_tasks(cttc_tr_in, cttc_teach, bits=bits, tag="cttc")
        t0 = time.perf_counter()
        losses_log: List[float] = []
        for st_i in range(steps):
            tk = tr_tasks[st_i % len(tr_tasks)]
            n_tot = len(tk["words"])
            idx = np.array([rng.randrange(n_tot) for _ in range(min(batch_size, n_tot))], dtype=np.int64)
            opt.zero_grad(set_to_none=True)
            step_l = 0.0
            for m0 in range(0, len(idx), micro_batch_size):
                sub = idx[m0 : min(len(idx), m0 + micro_batch_size)]
                lp, _, _, _ = conn.score_batch(
                    host, tk["prompt"], tk["choices"], tk["qtype"],
                    torch.from_numpy(tk["words"][sub]).to(host.device),
                    bits,
                    torch.from_numpy(tk["phase_slots"][sub]).to(host.device),
                    torch.from_numpy(tk["meta_scalars"][sub]).to(host.device),
                    ref_summary=torch.from_numpy(tk["ref_summary"][sub]).to(host.device),
                    scorer_mode=scorer_mode,
                    discrete_d=torch.from_numpy(tk["discrete_d"][sub]).to(host.device),
                )
                y_sub = torch.from_numpy(tk["labels"][sub]).to(host.device)
                p_tch = torch.from_numpy(tk["teacher_p"][sub]).to(host.device).clamp_min(1e-12)
                p_tch = p_tch / p_tch.sum(dim=-1, keepdim=True)
                kl = torch.sum(p_tch * (torch.log(p_tch) - lp), dim=-1).mean()
                ce = F.nll_loss(lp, y_sub)
                loss = (0.5 * kl + 0.5 * ce) * (float(len(sub)) / float(len(idx)))
                loss.backward()
                step_l += float(loss.item())
                del lp, kl, ce, loss
            torch.nn.utils.clip_grad_norm_(conn.parameters(), 1.0)
            opt.step()
            sched.step()
            losses_log.append(step_l)
        conn.eval()
        s_rep, raw_a = _eval_real_domains(host, conn, domains, adapter=None, bits=bits, prompts=prompts, eval_bs=eval_bs, scorer_mode=scorer_mode)
        del opt, sched
        return conn, {"train_wall_s": round(time.perf_counter() - t0, 2), "final_loss": float(np.mean(losses_log[-4:])), "summary": s_rep, "raw": raw_a}

    # --- Substage 2: Matched 16-bit vs 4-bit Real-SDR Connector Adaptation ---
    pure_c16_state_path = (cache_dir if cache_dir is not None else out_dir) / f"stage4_real_pure_c16_s{seed}.pt"

    def _run_sub2() -> dict:
        sub2: Dict[str, Any] = {}
        raw_store: Dict[str, dict] = {}
        for arm in ARM_NAMES_STAGE4:
            conn_a, rep_a = _train_connector_real(arm, bits=16, steps=steps_adapt)
            entry_a = {k: v for k, v in rep_a.items() if k != "raw"}
            if arm in ("equivariant_host_pure", "equivariant_physical_residual"):
                entry_a["blackwell_staircase"] = _eval_blackwell_staircase(conn_a, None, 16, rep_a["summary"])
            sub2[f"{arm}_c16"] = entry_a
            raw_store[f"{arm}_c16"] = rep_a["raw"]
            if arm == "equivariant_host_pure":
                tmp_p = pure_c16_state_path.parent / f".{pure_c16_state_path.name}.tmp"
                torch.save({k: v.cpu() for k, v in conn_a.state_dict().items()}, tmp_p)
                tmp_p.replace(pure_c16_state_path)
            del conn_a
        conn_4, rep_4 = _train_connector_real("equivariant_host_pure", bits=4, steps=steps_adapt)
        entry_4 = {k: v for k, v in rep_4.items() if k != "raw"}
        entry_4["blackwell_staircase"] = _eval_blackwell_staircase(conn_4, None, 4, rep_4["summary"])
        sub2["equivariant_host_pure_s17_4bit"] = entry_4
        raw_store["equivariant_host_pure_s17_4bit"] = rep_4["raw"]
        del conn_4

        # M2 paired stats: 16-bit vs 4-bit & Pure vs Zero/Naive
        paired: Dict[str, Any] = {}
        for dom_k, tkey in (("pluto_test", "q2_two"), ("cttc_in_domain_test", "q2_two"), ("cttc_unseen_holdout_f708_snr10", "q2_two")):
            a16 = raw_store["equivariant_host_pure_c16"][dom_k][tkey]
            a4 = raw_store["equivariant_host_pure_s17_4bit"][dom_k][tkey]
            z16 = raw_store["zero_c16"][dom_k][tkey]
            paired[f"{dom_k}_{tkey}_pure_c16_vs_pure_4bit_nll"] = _cluster_paired_stats(a16["nll"], a4["nll"], a16["clusters"], higher_is_better_a=False, seed=seed)
            paired[f"{dom_k}_{tkey}_pure_c16_vs_zero_c16_nll"] = _cluster_paired_stats(a16["nll"], z16["nll"], a16["clusters"], higher_is_better_a=False, seed=seed + 1)
        sub2["paired_stats"] = paired
        return sub2

    sub2_rep = _cached_substage("real_sub2_c16_vs_4bit", _run_sub2)
    print(
        f"[RealSDR Sub2] Pure c16 vs 4b holdout f708 Q2_two: "
        f"{sub2_rep['equivariant_host_pure_c16']['summary']['cttc_unseen_holdout_f708_snr10']['q2_two']['acc']:.4f} vs "
        f"{sub2_rep['equivariant_host_pure_s17_4bit']['summary']['cttc_unseen_holdout_f708_snr10']['q2_two']['acc']:.4f}",
        flush=True,
    )

    # Ensure pure_c16 connector weights are loaded for Substage 3
    if not mock_tiny_for_test:
        assert pure_c16_state_path.exists(), f"Missing {pure_c16_state_path}"
    pure_conn_c16 = Stage4MatchedConnector("equivariant_host_pure", encoder.table, hidden_size=host.hidden_size).to(host.device)
    if pure_c16_state_path.exists():
        pure_conn_c16.load_state_dict(torch.load(pure_c16_state_path, map_location=host.device, weights_only=False))
    pure_conn_c16.requires_grad_(False)
    pure_conn_c16.eval()

    # --- Substage 3: Domain-Specific Adapters & Zero-Retrain W2 Lie+Schur Soup -> Unseen f708_snr10 ---
    def _run_sub3() -> dict:
        eval_frozen_refs = compute_frozen_general_refs(host, pure_conn_c16, GENERAL_TEXT_EVAL_PROMPTS)
        train_mix_refs = compute_frozen_general_refs(host, pure_conn_c16, GENERAL_TEXT_TRAIN_PROMPTS)["mix"]

        def _fit_domain_adapter(is_surgical: bool, domain_tasks: List[dict], tag: str) -> Tuple[nn.Module, dict]:
            torch.manual_seed(seed + len(tag))
            rng = random.Random(seed + len(tag) + 17)
            if is_surgical:
                ad: nn.Module = SurgicalHostAdapter(
                    host.lm, fa_layers=fa_l, dn_layers=dn_l, stop_layer=stop_l,
                    mode="mode1_stopgrad_l10", u_c_by_layer=proj_map, b_align=b_align,
                ).to(host.device)
            else:
                ad = UnconstrainedLoRAAdapter(host.lm, fa_layers=fa_l, dn_layers=dn_l, stop_layer=stop_l, r_lora=8).to(host.device)
            ad.train()
            opt = torch.optim.AdamW(ad.parameters(), lr=3e-4, weight_decay=1e-4)
            sched = _make_lr_scheduler(opt, steps_soup)
            for st_i in range(steps_soup):
                tk = domain_tasks[st_i % len(domain_tasks)]
                n_tot = len(tk["words"])
                idx = np.array([rng.randrange(n_tot) for _ in range(min(batch_size, n_tot))], dtype=np.int64)
                opt.zero_grad(set_to_none=True)
                for m0 in range(0, len(idx), micro_batch_size):
                    sub = idx[m0 : min(len(idx), m0 + micro_batch_size)]
                    lp, _, _, _ = pure_conn_c16.score_batch(
                        host, tk["prompt"], tk["choices"], tk["qtype"],
                        torch.from_numpy(tk["words"][sub]).to(host.device),
                        16,
                        torch.from_numpy(tk["phase_slots"][sub]).to(host.device),
                        torch.from_numpy(tk["meta_scalars"][sub]).to(host.device),
                        ref_summary=torch.from_numpy(tk["ref_summary"][sub]).to(host.device),
                        scorer_mode=scorer_mode,
                        surgical_adapter=ad,
                        discrete_d=torch.from_numpy(tk["discrete_d"][sub]).to(host.device),
                    )
                    p_tch = torch.from_numpy(tk["teacher_p"][sub]).to(host.device).clamp_min(1e-12)
                    p_tch = p_tch / p_tch.sum(dim=-1, keepdim=True)
                    losses = rao_blackwell_bayes_kl_cf_loss(lp, p_tch, cf_swapped_logits=None, lambda_cf=0.0, lambda_norm=0.02 if is_surgical else 0.0)
                    (losses["total"] * (float(len(sub)) / float(len(idx)))).backward()
                    del lp, losses
                if is_surgical and st_i % 4 == 0:
                    p_idx = (st_i // 4) % len(GENERAL_TEXT_TRAIN_PROMPTS)
                    m_emb, m_mask = _build_fixed_mix_prefix(host, pure_conn_c16, GENERAL_TEXT_TRAIN_PROMPTS[p_idx])
                    ref_lp = train_mix_refs[p_idx][0].to(host.device)
                    with ad.activate(g_em=1, m_acc_mask=m_mask):
                        m_log = _forward_last_token_logits(host, m_emb)
                    q_lp = F.log_softmax(m_log, dim=-1, dtype=torch.float64)
                    mix_kl = torch.sum(ref_lp.exp() * (ref_lp - q_lp), dim=-1).mean()
                    (2.5 * mix_kl.float()).backward()
                    del m_log, q_lp, mix_kl
                torch.nn.utils.clip_grad_norm_(ad.parameters(), 1.0)
                opt.step()
                sched.step()
            ad.eval()
            rep_d, _ = _eval_real_domains(host, pure_conn_c16, domains, adapter=ad, bits=16, prompts=prompts, eval_bs=eval_bs, scorer_mode=scorer_mode)
            aud_d = audit_pure_and_mix_kl(host, pure_conn_c16, ad, GENERAL_TEXT_EVAL_PROMPTS, is_gated_surgical=is_surgical, frozen_refs=eval_frozen_refs)
            sd = {k: v.cpu() for k, v in ad.state_dict().items()}
            ad.remove()
            del ad, opt, sched
            return sd, {"summary": rep_d, "audit": aud_d}

        pluto_tasks = _build_train_tasks(pluto_tr, pluto_teach, bits=16, tag="pluto")
        cttc_tasks = _build_train_tasks(cttc_tr_in, cttc_teach, bits=16, tag="cttc")

        sd_sp, rep_sp = _fit_domain_adapter(True, pluto_tasks, "surg_pluto")
        sd_sc, rep_sc = _fit_domain_adapter(True, cttc_tasks, "surg_cttc")
        sd_lp, rep_lp = _fit_domain_adapter(False, pluto_tasks, "lemmo_pluto")
        sd_lc, rep_lc = _fit_domain_adapter(False, cttc_tasks, "lemmo_cttc")

        # Perform zero-retraining W2 Canonical Lie+Schur Convex Soup (Pluto + CTTC -> unseen f708_snr10)
        s_a = SurgicalHostAdapter(host.lm, fa_layers=fa_l, dn_layers=dn_l, stop_layer=stop_l, mode="mode1_stopgrad_l10", u_c_by_layer=proj_map, b_align=b_align).to(host.device)
        s_a.remove()
        s_b = SurgicalHostAdapter(host.lm, fa_layers=fa_l, dn_layers=dn_l, stop_layer=stop_l, mode="mode1_stopgrad_l10", u_c_by_layer=proj_map, b_align=b_align).to(host.device)
        s_b.remove()
        s_soup = SurgicalHostAdapter(host.lm, fa_layers=fa_l, dn_layers=dn_l, stop_layer=stop_l, mode="mode1_stopgrad_l10", u_c_by_layer=proj_map, b_align=b_align).to(host.device)
        try:
            s_a.load_state_dict({k: v.to(host.device) for k, v in sd_sp.items()})
            s_b.load_state_dict({k: v.to(host.device) for k, v in sd_sc.items()})
            soup_diag = merge_surgical_lie_schur_soup(s_a, s_b, s_soup, w_a=0.5)
            rep_soup, raw_soup = _eval_real_domains(host, pure_conn_c16, domains, adapter=s_soup, bits=16, prompts=prompts, eval_bs=eval_bs, scorer_mode=scorer_mode)
            bw_soup = _eval_blackwell_staircase(pure_conn_c16, s_soup, 16, rep_soup)
            aud_soup = audit_pure_and_mix_kl(host, pure_conn_c16, s_soup, GENERAL_TEXT_EVAL_PROMPTS, is_gated_surgical=True, frozen_refs=eval_frozen_refs)
        finally:
            s_soup.remove()
            del s_a, s_b, s_soup

        # Merge LEMMO adapters via naive factor_avg and gauge-invariant weight_avg
        l_a = UnconstrainedLoRAAdapter(host.lm, fa_layers=fa_l, dn_layers=dn_l, stop_layer=stop_l, r_lora=8).to(host.device)
        l_a.remove()
        l_b = UnconstrainedLoRAAdapter(host.lm, fa_layers=fa_l, dn_layers=dn_l, stop_layer=stop_l, r_lora=8).to(host.device)
        l_b.remove()
        lem_merge_reps: Dict[str, Any] = {}
        raw_lem_fa: Dict[str, dict] = {}
        for mmode in ("factor_avg", "weight_avg"):
            l_dst = UnconstrainedLoRAAdapter(host.lm, fa_layers=fa_l, dn_layers=dn_l, stop_layer=stop_l, r_lora=8).to(host.device)
            try:
                l_a.load_state_dict({k: v.to(host.device) for k, v in sd_lp.items()})
                l_b.load_state_dict({k: v.to(host.device) for k, v in sd_lc.items()})
                merge_lemmo_adapters(l_a, l_b, l_dst, mode=mmode, w_a=0.5)
                rep_m, raw_m = _eval_real_domains(host, pure_conn_c16, domains, adapter=l_dst, bits=16, prompts=prompts, eval_bs=eval_bs, scorer_mode=scorer_mode)
                aud_m = audit_pure_and_mix_kl(host, pure_conn_c16, l_dst, GENERAL_TEXT_EVAL_PROMPTS, is_gated_surgical=False, frozen_refs=eval_frozen_refs)
                lem_merge_reps[mmode] = {"summary": rep_m, "audit": aud_m}
                if mmode == "factor_avg":
                    raw_lem_fa = raw_m
            finally:
                l_dst.remove()
                del l_dst
        del l_a, l_b

        soup_vs_lemmo_ho = _cluster_paired_stats(
            raw_soup["cttc_unseen_holdout_f708_snr10"]["q2_two"]["nll"],
            raw_lem_fa["cttc_unseen_holdout_f708_snr10"]["q2_two"]["nll"],
            raw_soup["cttc_unseen_holdout_f708_snr10"]["q2_two"]["clusters"],
            higher_is_better_a=False,
            seed=seed + 5,
        )
        return {
            "surgical_pluto": rep_sp,
            "surgical_cttc": rep_sc,
            "lemmo_pluto": rep_lp,
            "lemmo_cttc": rep_lc,
            "surgical_lie_schur_soup": {"soup_diagnostics": soup_diag, "summary": rep_soup, "blackwell_staircase": bw_soup, "audit": aud_soup},
            "lemmo_factor_avg": lem_merge_reps["factor_avg"],
            "lemmo_weight_avg": lem_merge_reps["weight_avg"],
            "holdout_f708_soup_vs_lemmo_factor_avg_q2_nll": soup_vs_lemmo_ho,
        }

    sub3_rep = _cached_substage("real_sub3_lie_schur_soup", _run_sub3)
    del pure_conn_c16
    assert _compute_host_fingerprint(host.lm) == host.host_sha256, "Host SHA mutated during Real SDR!"

    # Combine family p-values with Holm-Bonferroni
    raw_p_fam = {
        "H_real_pluto_c16_vs_4b_nll": sub2_rep["paired_stats"]["pluto_test_q2_two_pure_c16_vs_pure_4bit_nll"]["p_one_sided_improved"],
        "H_real_cttc_c16_vs_4b_nll": sub2_rep["paired_stats"]["cttc_in_domain_test_q2_two_pure_c16_vs_pure_4bit_nll"]["p_one_sided_improved"],
        "H_real_holdout_c16_vs_zero_nll": sub2_rep["paired_stats"]["cttc_unseen_holdout_f708_snr10_q2_two_pure_c16_vs_zero_c16_nll"]["p_one_sided_improved"],
        "H_real_holdout_soup_vs_factor_avg_nll": sub3_rep["holdout_f708_soup_vs_lemmo_factor_avg_q2_nll"]["p_one_sided_improved"],
    }
    holm_adj = _holm_bonferroni(raw_p_fam)

    report = {
        "seed": seed,
        "model_id": host.model_id,
        "host_sha256": host.host_sha256,
        "wall_s": round(time.perf_counter() - t_all, 2),
        "sub1_zeroshot_4bit": zero_shot_rep,
        "sub2_adapted_c16_vs_4bit": sub2_rep,
        "sub3_cross_domain_lie_schur_soup": sub3_rep,
        "holm_bonferroni_real_sdr": {"raw_p": raw_p_fam, "adjusted_p": holm_adj},
    }
    (out_dir / "stage4_real_sdr_4b_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


# =============================================================================
# 3. Stage 6 Active EIG (`m5-stage6-blind-v1`) on Qwen3.5-4B (C16 vs S17 4-bit)
# =============================================================================

class Stage6ActiveHostAdapter(nn.Module):
    """Compact <=620k parameter multi-window `m5-c16-v1-le` adapter & head for Stage 6 active EIG."""

    def __init__(self, hidden_size: int = 2560):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.win_emb = nn.Embedding(16, 16)
        self.fam_emb = nn.Embedding(3, 16)
        # 5 windows * (16 s17_emb + 6 tail + 2 turn_cos_sin) + 16 fam = 136
        self.feat_proj = nn.Linear(136, 128)
        self.tok_head = nn.Linear(128, 8 * 96)
        self.tok_up = nn.Linear(96, self.hidden_size)
        self.tok_norm = nn.LayerNorm(self.hidden_size)
        self.state_head = nn.Sequential(
            nn.Linear(self.hidden_size, 96),
            nn.GELU(),
            nn.Linear(96, 16),
        )
        nn.init.normal_(self.tok_up.weight, std=0.02)
        nn.init.zeros_(self.tok_up.bias)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def encode_soft_tokens(self, words_5: torch.Tensor, fam_ids: torch.Tensor, bits: int = 16) -> torch.Tensor:
        w_in = (words_5 >> 12) if int(bits) == 4 else words_5
        codes, tail, _ = decode_c16_tail(w_in, bits=int(bits))
        emb = self.win_emb(codes)  # (B, 5, 16)
        turns = (codes // 4).to(torch.float32) * (0.5 * math.pi)
        turn_cs = torch.stack((torch.cos(turns), torch.sin(turns)), dim=-1)  # (B, 5, 2)
        per_w = torch.cat((emb, tail, turn_cs), dim=-1).flatten(1)  # (B, 120)
        f_vec = torch.cat((per_w, self.fam_emb(fam_ids)), dim=-1)  # (B, 136)
        h = F.gelu(self.feat_proj(f_vec))
        toks = self.tok_head(h).view(-1, 8, 96)
        return self.tok_norm(self.tok_up(toks)) * 0.05

    def forward_states(
        self,
        host: QwenCacheHost,
        words_5: torch.Tensor,
        fam_ids: torch.Tensor,
        prose_ids: torch.Tensor,
        bits: int = 16,
        surgical_adapter: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        b = words_5.shape[0]
        soft_tokens = self.encode_soft_tokens(words_5, fam_ids, bits=bits).to(dtype=host.dtype)
        embed_layer = host.lm.get_input_embeddings()
        prose_emb = embed_layer(prose_ids.unsqueeze(0).expand(b, -1))
        full_emb = torch.cat((soft_tokens, prose_emb), dim=1)
        seq_len = full_emb.shape[1]
        attn = torch.ones((b, seq_len), device=host.device, dtype=torch.long)
        m_acc_mask = torch.cat(
            (
                torch.ones((b, 8), device=host.device, dtype=torch.bool),
                torch.zeros((b, prose_ids.shape[0]), device=host.device, dtype=torch.bool),
            ),
            dim=1,
        )
        ctx = surgical_adapter.activate(g_em=1, m_acc_mask=m_acc_mask) if surgical_adapter is not None else nullcontext()
        with ctx:
            fwd_kwargs: Dict[str, Any] = dict(
                inputs_embeds=full_emb,
                attention_mask=attn,
                output_hidden_states=True,
                use_cache=False,
            )
            if getattr(host, "_supports_logits_to_keep", False):
                fwd_kwargs["logits_to_keep"] = 1
            if host.tokenizer is not None and hasattr(host, "_make_position_ids"):
                fwd_kwargs["position_ids"] = host._make_position_ids(
                    torch.arange(seq_len, device=host.device, dtype=torch.long), b
                )
            out = host._lm_forward(**fwd_kwargs)
        h_last = out.hidden_states[-1][:, -1, :].to(dtype=torch.float32)
        return self.state_head(h_last)


def _compute_stage6_simultaneous_intervals(by_family: Sequence[Tuple[dict, dict]], controls: Sequence[str]) -> dict:
    """Exact M5 §9 Student-t Bonferroni simultaneous 95% CIs (non-asserting summary)."""
    family_count = len(by_family)
    sample_size = len(by_family[0][1]["max_EIG_window"])
    endpoints = ("nll_gain_nats", "accuracy_gain")
    fixed_actions = tuple(k for k in by_family[0][1] if k.startswith("constant_"))
    total_intervals = len(endpoints) * (len(controls) + family_count * len(fixed_actions))
    critical = float(student_t.ppf(1.0 - 0.05 / (2.0 * total_intervals), max(1, sample_size - 1)))

    def contrasts(action: str, datasets: Sequence[dict]) -> dict:
        paired = [scores["max_EIG_window"] - scores[action] for scores in datasets]
        mean = np.mean([d.mean(axis=0) for d in paired], axis=0)
        se = np.sqrt(np.sum([d.var(axis=0, ddof=1) / max(1, len(d)) for d in paired], axis=0)) / max(1, len(paired))
        return {
            key: {"gain": float(m), "simultaneous_95ci": [float(m - critical * s), float(m + critical * s)]}
            for key, m, s in zip(endpoints, mean, se)
        }

    fixed_by_family = {
        res["family"]: {act: contrasts(act, [scores]) for act in fixed_actions}
        for res, scores in by_family
    }
    comparisons = {pol: contrasts(pol, [scores for _, scores in by_family]) for pol in controls}
    # Recompute oracle gain per endpoint properly
    oracle_gain = {}
    for key in endpoints:
        pf = [[item[key] for item in fixed.values()] for fixed in fixed_by_family.values()]
        oracle_gain[key] = {
            "gain": float(np.mean([min(item["gain"] for item in g) for g in pf])),
            "simultaneous_95ci": [
                float(np.mean([min(item["simultaneous_95ci"][side] for item in g) for g in pf]))
                for side in (0, 1)
            ],
        }
    comparisons["fixed_best_oracle"] = oracle_gain
    passed = all(
        entry["simultaneous_95ci"][0] > 0.0
        for comp in comparisons.values()
        for entry in comp.values()
    )
    return {
        "bonferroni_elementary_intervals": int(total_intervals),
        "critical_value": critical,
        "pooled_adaptive_advantages": comparisons,
        "strict_nll_and_accuracy_gate_passed": bool(passed),
    }


def _evaluate_stage6_weights_on_catalogue(
    catalogue: Sequence[Any],
    encoder: Any,
    n_dev: int,
    n_test: int,
    weights_by_family_dev: Sequence[np.ndarray],
    weights_by_family_test: Sequence[np.ndarray],
    dev_seed: int = 0,
    test_seed: int = 1,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Run exact `m5-stage6-blind-v1` sealed protocol and M3 reveal verification on predicted state weights."""
    dev_bundles = [verify_m5.geometry_split(g, encoder, fid, n_dev, dev_seed) for fid, g in enumerate(catalogue)]
    test_bundles = [verify_m5.geometry_split(g, encoder, fid, n_test, test_seed) for fid, g in enumerate(catalogue)]
    by_family = []
    per_family_reports = []
    all_nlls: List[np.ndarray] = []
    all_hits: List[np.ndarray] = []

    for fid, geom in enumerate(catalogue):
        dev_w, dev_tr, _, dev_sealed, _ = dev_bundles[fid]
        w_dev_p = np.asarray(weights_by_family_dev[fid], dtype=np.float64)
        w_dev_p = np.maximum(w_dev_p, 1e-12)
        w_dev_p /= w_dev_p.sum(axis=-1, keepdims=True)
        n_dev = len(dev_w)
        dev_actions = {str(k): np.full(n_dev, k) for k in range(len(geom.candidate_time))}
        dev_receipts = verify_m5.commit_geometry(dev_sealed, dev_actions, dev_w, w_dev_p, geom)
        dev_nll = [
            float(verify_m5.reveal_and_score(geom, dev_sealed, act, dev_receipts[k], w_dev_p, dev_tr)[0].mean())
            for k, act in dev_actions.items()
        ]
        fixed_dev = int(np.argmin(dev_nll))

        words, truth, _, sealed, random_action = test_bundles[fid]
        w_test_p = np.asarray(weights_by_family_test[fid], dtype=np.float64)
        w_test_p = np.maximum(w_test_p, 1e-12)
        w_test_p /= w_test_p.sum(axis=-1, keepdims=True)
        assert np.all(np.isfinite(w_test_p)) and np.max(np.abs(w_test_p.sum(axis=-1) - 1.0)) <= 1e-6

        eig, _ = verify_m5.geometry_utilities(w_test_p, geom)
        n = len(words)
        rows = np.arange(n)
        nearest = int(np.argmin(abs(geom.candidate_time - geom.old_time[-1])))
        actions = {
            "max_EIG_window": eig.argmax(axis=1),
            "random_window": random_action,
            "nearest_window": np.full(n, nearest),
            "fixed_best_on_dev": np.full(n, fixed_dev),
            **{f"constant_{k}": np.full(n, k) for k in range(len(geom.candidate_time))},
        }
        receipts = verify_m5.commit_geometry(sealed, actions, words, w_test_p, geom)
        metrics: Dict[str, Any] = {}
        outcomes: Dict[str, np.ndarray] = {}
        m3_err = 0.0
        for pol, sel in actions.items():
            losses, hits, _, _, log_m, obs, post, log_c = verify_m5.reveal_and_score(
                geom, sealed, sel, receipts[pol], w_test_p, truth
            )
            if pol == "max_EIG_window":
                m3_err = verify_m5.check_m3_reveal(geom, w_test_p, sel, obs, post, log_c, log_m)
                all_nlls.append(np.asarray(losses, dtype=np.float64))
                all_hits.append(np.asarray(hits, dtype=np.float64))
            outcomes[pol] = np.column_stack((-losses, hits))
            metrics[pol] = {
                "nll": float(np.mean(losses)),
                "accuracy": float(np.mean(hits)),
                "commit_sha256": receipts[pol],
                "expected_eig": float(eig[rows, sel].mean()),
            }
        fam_res = {
            "family": geom.name,
            "fixed_dev_candidate_index": fixed_dev,
            "m3_reveal_max_abs_err": float(m3_err),
            "policy_metrics": metrics,
        }
        per_family_reports.append(fam_res)
        by_family.append((fam_res, outcomes))

    sim_rep = _compute_stage6_simultaneous_intervals(by_family, verify_m5.ADAPTIVE_CONTROLS)
    pooled_policies: Dict[str, Any] = {}
    for pol in ("max_EIG_window", *verify_m5.ADAPTIVE_CONTROLS):
        draws = np.concatenate([sc[pol] for _, sc in by_family], axis=0)
        pooled_policies[pol] = {"nll": float(-draws[:, 0].mean()), "accuracy": float(draws[:, 1].mean())}

    return {
        "pooled": pooled_policies,
        "per_family": per_family_reports,
        "simultaneous_intervals": sim_rep,
    }, np.concatenate(all_nlls), np.concatenate(all_hits)


def run_stage6_active_eig_gpu(
    repo_root: Path,
    out_dir: Path,
    ckpt_dir: Path,
    kfac_path: Path,
    seed: int = 202611,
    steps: int = 240,
    batch_size: int = 32,
    eval_bs: int = 128,
    model_id: str = MODEL_ID_DEFAULT,
    revision: str = PINNED_REVISION_DEFAULT,
    mock_tiny_for_test: bool = False,
    host: Optional[QwenCacheHost] = None,
    cache_dir: Optional[Path] = None,
    on_model_done: Optional[Callable[[str], None]] = None,
) -> dict:
    """Execute `m5-stage6-blind-v1` sealed active EIG evaluation on Qwen3.5-4B (`16-bit vs 4-bit` + Surgical vs LEMMO)."""
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    if host is None:
        host = QwenCacheHost(model_id=model_id, revision=revision, mock_tiny_for_test=mock_tiny_for_test)
    if not mock_tiny_for_test:
        assert int(getattr(host.text_config, "vocab_size", 0)) == EXPECTED_4B_VOCAB
        assert host.host_sha256 == EXPECTED_4B_HOST_SHA256

    encoder, _ = load_s17_encoder(repo_root)
    catalogue = verify_m5.geometry_catalogue()

    n_train = 64 if mock_tiny_for_test else 2048
    n_dev = 32 if mock_tiny_for_test else verify_m5.ADAPTIVE_DEV
    n_test = 64 if mock_tiny_for_test else verify_m5.ADAPTIVE_TEST

    train_bundles = [verify_m5.geometry_split(g, encoder, fid, n_train, 99) for fid, g in enumerate(catalogue)]
    dev_bundles = [verify_m5.geometry_split(g, encoder, fid, n_dev, 0) for fid, g in enumerate(catalogue)]
    test_bundles = [verify_m5.geometry_split(g, encoder, fid, n_test, 1) for fid, g in enumerate(catalogue)]

    # Exact physical posterior targets on disjoint train split + physical c16 reference on dev/test
    train_words = np.concatenate([b[0] for b in train_bundles], axis=0)
    train_fams = np.concatenate([np.full(len(b[0]), fid, dtype=np.int64) for fid, b in enumerate(train_bundles)], axis=0)
    train_targets = np.concatenate([verify_m5.geometry_posterior(b[0], catalogue[fid]) for fid, b in enumerate(train_bundles)], axis=0).astype(np.float32)

    dev_phys = [verify_m5.geometry_posterior(b[0], catalogue[fid]) for fid, b in enumerate(dev_bundles)]
    test_phys = [verify_m5.geometry_posterior(b[0], catalogue[fid]) for fid, b in enumerate(test_bundles)]

    # Pre-tokenize FROZEN_QUERY per family once
    prose_ids_by_fam: List[torch.Tensor] = []
    for g in catalogue:
        text = f"公有信息家族：{g.name}。\n{verify_m5.FROZEN_QUERY}"
        if host.tokenizer is None:
            ids = [1, 2, 3, 4, 5, 6]
        else:
            ids = host.tokenizer.encode(text, add_special_tokens=False)
        prose_ids_by_fam.append(torch.tensor(ids, device=host.device, dtype=torch.long))

    layers = _get_decoder_layers(host.lm)
    n_l = len(layers)
    fa_l = [i for i in QWEN4B_FA_LAYERS if i < n_l] or ([n_l - 1] if n_l >= 1 else [])
    dn_l = [i for i in QWEN4B_DN_LAYERS if i < n_l]
    stop_l = min(QWEN4B_STOP_LAYER, max(0, n_l - 2))
    proj_map = _load_or_build_kfac(host, kfac_path, fa_l, dn_l, mock_tiny_for_test)

    models_out: Dict[str, Any] = {}
    nll_arrays: Dict[str, np.ndarray] = {}
    hit_arrays: Dict[str, np.ndarray] = {}

    def _run_cached_model(mname: str, eval_fn: Callable[[], Tuple[dict, np.ndarray, np.ndarray]]) -> None:
        cp = (cache_dir / f"s6_{mname}.pt") if cache_dir is not None else None
        if cp is not None and cp.exists():
            ck = torch.load(cp, map_location="cpu", weights_only=False)
            if isinstance(ck, dict) and ck.get("host_sha256") == host.host_sha256 and ck.get("n_test") == n_test and ck.get("seed") == seed:
                models_out[mname], nll_arrays[mname], hit_arrays[mname] = ck["rep"], ck["nlls"], ck["hits"]
                return
        rep, nlls, hits = eval_fn()
        models_out[mname], nll_arrays[mname], hit_arrays[mname] = rep, nlls, hits
        if cp is not None:
            tmp = cache_dir / f".s6_{mname}.tmp"
            torch.save({"host_sha256": host.host_sha256, "n_test": n_test, "seed": seed, "rep": rep, "nlls": nlls, "hits": hits}, tmp)
            tmp.replace(cp)
            if on_model_done is not None:
                on_model_done(mname)

    # 1. Exact Physical c16 Reference
    _run_cached_model(
        "physical_c16_reference",
        lambda: _evaluate_stage6_weights_on_catalogue(catalogue, encoder, n_dev, n_test, dev_phys, test_phys, dev_seed=0, test_seed=1),
    )

    def _train_and_eval_host(mname: str, bits: int, adapter_kind: str) -> Tuple[dict, np.ndarray, np.ndarray]:
        t_m0 = time.perf_counter()
        torch.manual_seed(seed + bits + len(mname))
        rng = np.random.default_rng(seed + bits + len(mname))
        head_mod = Stage6ActiveHostAdapter(hidden_size=host.hidden_size).to(host.device)
        surg_or_lem: Optional[nn.Module] = None
        if adapter_kind == "surgical":
            surg_p = ckpt_dir / f"stage4_phaseB_surgical96_s{seed}.pt"
            s_ck = torch.load(surg_p, map_location="cpu", weights_only=False) if surg_p.exists() else None
            ba = s_ck["b_align"] if s_ck is not None else calibrate_b_align_from_host(host)
            surg_or_lem = SurgicalHostAdapter(host.lm, fa_layers=fa_l, dn_layers=dn_l, stop_layer=-1, mode="mode1_stopgrad_l10", u_c_by_layer=proj_map, b_align=ba).to(host.device)
            if s_ck is not None:
                surg_or_lem.load_state_dict(s_ck["state_dict"])
            surg_or_lem.requires_grad_(False)
            surg_or_lem.eval()
        elif adapter_kind == "lemmo":
            lem_p = ckpt_dir / f"stage4_phaseB_lemmo_tier_p1_s{seed}.pt"
            l_ck = torch.load(lem_p, map_location="cpu", weights_only=False) if lem_p.exists() else None
            surg_or_lem = UnconstrainedLoRAAdapter(host.lm, fa_layers=fa_l, dn_layers=dn_l, stop_layer=-1, r_lora=8).to(host.device)
            if l_ck is not None:
                surg_or_lem.load_state_dict(l_ck["state_dict"])
            surg_or_lem.requires_grad_(False)
            surg_or_lem.eval()

        try:
            a_params = surg_or_lem.param_count() if surg_or_lem is not None else 0
            cum_params = head_mod.param_count() + a_params
            assert cum_params <= TIER_P1_MAX_PARAMS, f"Stage 6 cumulative params {cum_params} > {TIER_P1_MAX_PARAMS}"

            opt = torch.optim.AdamW(head_mod.parameters(), lr=1e-3, weight_decay=1e-4)
            sched = _make_lr_scheduler(opt, steps)
            head_mod.train()
            for st_i in range(steps):
                fid = st_i % 3
                mask_f = np.where(train_fams == fid)[0]
                sel = rng.choice(mask_f, size=min(batch_size, len(mask_f)), replace=False)
                opt.zero_grad(set_to_none=True)
                w_b = torch.from_numpy(train_words[sel].astype(np.int64)).to(host.device)
                f_b = torch.full((len(sel),), fid, device=host.device, dtype=torch.long)
                tgt = torch.from_numpy(train_targets[sel]).to(host.device).clamp_min(1e-12)
                logits = head_mod.forward_states(host, w_b, f_b, prose_ids_by_fam[fid], bits=bits, surgical_adapter=surg_or_lem)
                lp = F.log_softmax(logits, dim=-1)
                loss = torch.sum(tgt * (torch.log(tgt) - lp), dim=-1).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head_mod.parameters(), 1.0)
                opt.step()
                sched.step()
                del logits, lp, loss

            head_mod.eval()
            def _predict_weights(bundles: Sequence[tuple]) -> List[np.ndarray]:
                out_w = []
                with torch.no_grad():
                    for fid_i, bnd in enumerate(bundles):
                        ws = bnd[0]
                        chunks = []
                        for s0 in range(0, len(ws), eval_bs):
                            s1 = min(len(ws), s0 + eval_bs)
                            wb = torch.from_numpy(ws[s0:s1].astype(np.int64)).to(host.device)
                            fb = torch.full((s1 - s0,), fid_i, device=host.device, dtype=torch.long)
                            lg = head_mod.forward_states(host, wb, fb, prose_ids_by_fam[fid_i], bits=bits, surgical_adapter=surg_or_lem)
                            chunks.append(F.softmax(lg, dim=-1).cpu().numpy().astype(np.float64))
                        out_w.append(np.concatenate(chunks, axis=0))
                return out_w

            w_dev_p = _predict_weights(dev_bundles)
            w_tst_p = _predict_weights(test_bundles)
            rep, nlls, hits = _evaluate_stage6_weights_on_catalogue(catalogue, encoder, n_dev, n_test, w_dev_p, w_tst_p, dev_seed=0, test_seed=1)
            if adapter_kind == "surgical":
                pure_aud_conn = Stage4MatchedConnector("equivariant_host_pure", encoder.table, hidden_size=host.hidden_size).to(host.device)
                aud_s6 = audit_pure_and_mix_kl(host, pure_aud_conn, surg_or_lem, GENERAL_TEXT_EVAL_PROMPTS, is_gated_surgical=True)
                assert aud_s6["passes_pure_kl_zero"], f"Stage 6 host_surgical_c16 PureKL != 0: {aud_s6}"
                rep["pure_kl_audit"] = aud_s6
                del pure_aud_conn
            rep["bits"] = int(bits)
            rep["adapter_kind"] = adapter_kind
            rep["stage6_adapter_params"] = head_mod.param_count()
            rep["cumulative_params"] = cum_params
            rep["under_tier_p1_cap"] = bool(cum_params <= TIER_P1_MAX_PARAMS)
            rep["wall_s"] = round(time.perf_counter() - t_m0, 2)
            del opt, sched
            print(
                f"[Stage6 {mname}] maxEIG={rep['pooled']['max_EIG_window']['accuracy']:.4f}/"
                f"{rep['pooled']['max_EIG_window']['nll']:.4f} "
                f"fixedDev={rep['pooled']['fixed_best_on_dev']['accuracy']:.4f} "
                f"gate={rep['simultaneous_intervals']['strict_nll_and_accuracy_gate_passed']} "
                f"elapsed={rep['wall_s']}s",
                flush=True,
            )
            return rep, nlls, hits
        finally:
            if surg_or_lem is not None:
                surg_or_lem.remove()
                del surg_or_lem
            del head_mod
            gc.collect()
            if host.device.type == "cuda":
                torch.cuda.empty_cache()

    for mname, bits, kind in (
        ("host_frozen_c16", 16, "none"),
        ("host_frozen_s17_4bit", 4, "none"),
        ("host_surgical_c16", 16, "surgical"),
        ("host_lemmo_c16", 16, "lemmo"),
    ):
        _run_cached_model(mname, lambda b=bits, k=kind, m=mname: _train_and_eval_host(m, b, k))

    assert _compute_host_fingerprint(host.lm) == host.host_sha256, "Host SHA mutated during Stage 6!"

    # Paired 16-bit vs 4-bit representation contrast on identical sealed recordings & frozen Qwen3.5-4B
    m_clusters = np.arange(len(nll_arrays["host_frozen_c16"]), dtype=np.int64)
    c16_vs_4bit_nll = _cluster_paired_stats(
        np.asarray(nll_arrays["host_frozen_c16"], dtype=np.float64),
        np.asarray(nll_arrays["host_frozen_s17_4bit"], dtype=np.float64),
        m_clusters,
        higher_is_better_a=False,
        seed=seed,
    )
    c16_vs_4bit_acc = _cluster_paired_stats(
        np.asarray(hit_arrays["host_frozen_c16"], dtype=np.float64),
        np.asarray(hit_arrays["host_frozen_s17_4bit"], dtype=np.float64),
        m_clusters,
        higher_is_better_a=True,
        seed=seed + 1,
    )

    report = {
        "seed": seed,
        "model_id": host.model_id,
        "host_sha256": host.host_sha256,
        "wire_id": "m5-c16-v1-le",
        "protocol_id": "m5-stage6-blind-v1",
        "n_test_per_family": n_test,
        "wall_s": round(time.perf_counter() - t0, 2),
        "models": models_out,
        "paired_16bit_vs_4bit_frozen_host": {
            "nll_diff_c16_minus_4bit": c16_vs_4bit_nll,
            "acc_diff_c16_minus_4bit": c16_vs_4bit_acc,
        },
    }
    (out_dir / "stage6_active_eig_4b_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
