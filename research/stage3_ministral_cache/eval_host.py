"""Step C.3 Frozen Ministral-3-3B host blind evaluation across C1..C6, controls, and bootstrap CIs."""
import argparse
import gc
import json
import math
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from research.stage3_ministral_cache.analytic import readout_from_table, solve_full_iq_known_param
from research.stage3_ministral_cache.data import MotherBatch, generate_mothers
from research.stage3_ministral_cache.eval_direct import load_frozen_models
from research.stage3_ministral_cache.ministral_connector import (
    MODEL_ID_DEFAULT,
    MinistralCacheHost,
    WaveformCacheConnector,
    build_ref_feature,
)
from research.stage3_ministral_cache.schema import (
    ARMS,
    CONDITION_SPECS,
    ROOTS,
    ROTATIONS,
    build_query_prompt,
    matched_sum,
    sector_of_complex,
)
from research.stage3_ministral_cache.scoring import (
    check_vector_relation,
    compute_classification_metrics,
    extract_worst_examples,
    paired_bootstrap_ci,
)
from research.stage3_ministral_cache.train_connector import _encode_split_cache


STAGE3_DIR = Path(__file__).resolve().parent


def _batched_score_group(
    host: MinistralCacheHost,
    connector: Optional[WaveformCacheConnector],
    prompt: str,
    choices: Tuple[str, ...],
    z1_np: np.ndarray,
    z2_np: np.ndarray,
    ref_feats_list: Sequence[np.ndarray],
    batch_size: int = 64,
    temperature: float = 1.0,
    text_only_mode: bool = False,
    extra_history: Optional[List[dict]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Score a homogeneous group with zero padding; returns `(probs (N,4), raw_loglik (N,4))`."""
    n = len(z1_np)
    probs_chunks = []
    ll_chunks = []
    with torch.no_grad():
        for st in range(0, n, batch_size):
            ed = min(n, st + batch_size)
            z1_t = torch.tensor(z1_np[st:ed], dtype=torch.long)
            z2_t = torch.tensor(z2_np[st:ed], dtype=torch.long)
            r_t = [torch.tensor(rf[st:ed], dtype=torch.float32) for rf in ref_feats_list]
            pref = host.build_prefix_embeds(
                connector,
                prompt,
                z1_t,
                z2_t,
                ref_feats_list=r_t,
                text_only_mode=text_only_mode,
                extra_history=extra_history,
            )
            lp, raw_ll = host.score_candidates_kv(pref, choices, temperature=temperature)
            if not torch.isfinite(raw_ll).all():
                raise FloatingPointError("Non-finite host logits encountered during candidate scoring")
            probs_chunks.append(lp.exp().detach().cpu().numpy().astype(np.float64))
            ll_chunks.append(raw_ll.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(probs_chunks, axis=0), np.concatenate(ll_chunks, axis=0)


def _unique_ref_feats(refs) -> List[np.ndarray]:
    seen = set()
    out = []
    for r in refs:
        if r.acquisition_id in seen:
            continue
        seen.add(r.acquisition_id)
        out.append(build_ref_feature(r))
    return out


def _summarize_rot_hold_dup(
    preds_q1: List[np.ndarray],
    preds_q3_rep: List[np.ndarray],
    preds_q3_ind: List[np.ndarray],
    preds_q2: Dict[str, List[np.ndarray]],
    coord_q1: np.ndarray,
    coord_q2: np.ndarray,
    coord_q3: np.ndarray,
    tol_rot_hold: float,
    tol_dup: float,
) -> Tuple[dict, dict, bool, int]:
    checks = {}
    n_m = len(preds_q1[0])
    fails_prop = {
        "rotate": np.zeros(n_m, dtype=bool),
        "hold": np.zeros(n_m, dtype=bool),
        "duplicate": np.zeros(n_m, dtype=bool),
    }

    def rec(cname: str, actual: np.ndarray, expected: np.ndarray, prop: str, tol: float):
        r = check_vector_relation(actual, expected, tol=tol)
        fails_prop[prop] |= r["failed_mask"]
        checks[cname] = {k: v for k, v in r.items() if k != "failed_mask"}

    rec("q1_rot01_rotates", preds_q1[1], np.roll(preds_q1[0], 1, axis=1), "rotate", tol_rot_hold)
    rec("q1_rot11_holds", preds_q1[2], preds_q1[0], "hold", tol_rot_hold)
    rec("q3_rep_rot01_holds", preds_q3_rep[1], preds_q3_rep[0], "hold", tol_rot_hold)
    rec("q3_rep_rot11_rotates", preds_q3_rep[2], np.roll(preds_q3_rep[0], 1, axis=1), "rotate", tol_rot_hold)
    rec("q3_ind_rot01_holds", preds_q3_ind[1], preds_q3_ind[0], "hold", tol_rot_hold)
    rec("q3_ind_rot11_holds", preds_q3_ind[2], preds_q3_ind[0], "hold", tol_rot_hold)
    rec("q3_ind_uniform", preds_q3_ind[0], np.full_like(preds_q3_ind[0], 0.25), "hold", tol_rot_hold)

    for j in range(len(ROTATIONS)):
        if j == 1:
            rec("q2_valid_rot01_holds", preds_q2["valid"][1], preds_q2["valid"][0], "hold", tol_rot_hold)
        elif j == 2:
            rec("q2_valid_rot11_rotates", preds_q2["valid"][2], np.roll(preds_q2["valid"][0], 1, axis=1), "rotate", tol_rot_hold)
        rec(f"q2_indep_equals_none_r{j}", preds_q2["independent"][j], preds_q2["none"][j], "hold", tol_rot_hold)
        rec(f"q2_dup_equals_valid_r{j}", preds_q2["duplicate_copy"][j], preds_q2["valid"][j], "duplicate", tol_dup)

    rec("coord_q1_holds", coord_q1, preds_q1[0], "hold", tol_rot_hold)
    rec("coord_q2_holds", coord_q2, preds_q2["valid"][0], "hold", tol_rot_hold)
    rec("coord_q3_rotates", coord_q3, np.roll(preds_q3_rep[0], 1, axis=1), "rotate", tol_rot_hold)

    three_props = {k: {"mothers_failed": int(v.sum()), "pass": not bool(v.any())} for k, v in fails_prop.items()}
    failed_union = fails_prop["rotate"] | fails_prop["hold"] | fails_prop["duplicate"]
    return checks, three_props, not bool(failed_union.any()), int(failed_union.sum())


def evaluate_host_on_split_subset(
    host: MinistralCacheHost,
    connector: Optional[WaveformCacheConnector],
    enc,
    split: str,
    cond_names: Sequence[str],
    max_blocks: Optional[int] = None,
    batch_size: int = 64,
    cal_temp: float = 1.0,
    tol_rot_hold: float = 0.08,
    tol_dup: float = 1e-4,
    text_only_mode: bool = False,
    shuffled_z: bool = False,
    shuffle_seed: int = 99123,
) -> dict:
    """Evaluate frozen host + connector across cond_names; returns metrics, checks, and arrays."""
    if connector is not None:
        connector.eval()
    cond_out: Dict[str, dict] = {}
    per_mother_map: Dict[str, np.ndarray] = {}

    for cname in cond_names:
        t0 = time.perf_counter()
        batch = generate_mothers(split, cname, max_blocks=max_blocks)
        cond = batch.condition
        n_m = len(batch.x1)
        codes_map = _encode_split_cache(enc, batch)
        if shuffled_z:
            new_map = {}
            for j, rot in enumerate(ROTATIONS):
                z1_orig, z2_orig = codes_map[rot]
                rng = np.random.default_rng(shuffle_seed + 100 * len(cname) + j + sum(ord(ch) for ch in cname))
                perm = rng.permutation(n_m)
                new_map[rot] = (z1_orig[perm], z2_orig[perm])
            codes_map = new_map

        preds_q1, ll_q1, labels_q1 = [], [], []
        preds_q3_rep, ll_q3_rep, labels_q3_rep = [], [], []
        preds_q3_ind, ll_q3_ind, labels_q3_ind = [], [], []
        preds_q2: Dict[str, List[np.ndarray]] = {a: [] for a in ARMS}
        ll_q2: Dict[str, List[np.ndarray]] = {a: [] for a in ARMS}
        labels_q2 = []
        preds_q3_after, ll_q3_after = [], []

        raw_ll_parts = []
        raw_lbl_parts = []

        for rot in ROTATIONS:
            s1, s2 = rot
            z1_np, z2_np = codes_map[rot]

            y_q1 = (batch.x2 - batch.x1 + s2 - s1) % 4
            y_q2 = (batch.x1 + s1) % 4
            y_q3_rep = sector_of_complex(matched_sum(batch.y3_repeat * ROOTS[s1 % 4]))
            y_q3_ind = sector_of_complex(matched_sum(batch.y3_indep * ROOTS[s1 % 4]))

            labels_q1.append(y_q1)
            labels_q2.append(y_q2)
            labels_q3_rep.append(y_q3_rep)
            labels_q3_ind.append(y_q3_ind)

            p1, ch1 = build_query_prompt("Q1", cond)
            pr1, raw1 = _batched_score_group(
                host, connector, p1, ch1, z1_np, z2_np, [], batch_size=batch_size, temperature=cal_temp, text_only_mode=text_only_mode
            )
            preds_q1.append(pr1)
            ll_q1.append(raw1)
            raw_ll_parts.append(raw1)
            raw_lbl_parts.append(y_q1)

            for proto, dst_p, dst_l, lbl_arr in (
                ("repeat_x1", preds_q3_rep, ll_q3_rep, y_q3_rep),
                ("independent_x3", preds_q3_ind, ll_q3_ind, y_q3_ind),
            ):
                p3, ch3 = build_query_prompt("Q3", cond, protocol=proto)
                pr3, raw3 = _batched_score_group(
                    host, connector, p3, ch3, z1_np, z2_np, [], batch_size=batch_size, temperature=cal_temp, text_only_mode=text_only_mode
                )
                dst_p.append(pr3)
                dst_l.append(raw3)
                raw_ll_parts.append(raw3)
                raw_lbl_parts.append(lbl_arr)

            for arm in ARMS:
                refs = batch.references_for_arm(arm, coord_rot=0)
                p2, ch2 = build_query_prompt("Q2", cond, ref_arm=arm, references=refs, deduplicate=True)
                r_feats = _unique_ref_feats(refs)
                pr2, raw2 = _batched_score_group(
                    host, connector, p2, ch2, z1_np, z2_np, r_feats, batch_size=batch_size, temperature=cal_temp, text_only_mode=text_only_mode
                )
                preds_q2[arm].append(pr2)
                ll_q2[arm].append(raw2)
                if arm != "duplicate_copy":
                    raw_ll_parts.append(raw2)
                    raw_lbl_parts.append(y_q2)

            refs_val = batch.references_for_arm("valid", coord_rot=0)
            p3a, ch3a = build_query_prompt("Q3_after_pilot", cond, protocol="repeat_x1", ref_arm="valid", references=refs_val, deduplicate=True)
            pr3a, raw3a = _batched_score_group(
                host, connector, p3a, ch3a, z1_np, z2_np, _unique_ref_feats(refs_val), batch_size=batch_size, temperature=cal_temp, text_only_mode=text_only_mode
            )
            preds_q3_after.append(pr3a)
            ll_q3_after.append(raw3a)
            raw_ll_parts.append(raw3a)
            raw_lbl_parts.append(y_q3_rep)

        # Coordinate rotation test: rotate both waveforms (+1, +1) AND reference (+1)
        z1_c, z2_c = codes_map[(1, 1)]
        refs_c = batch.references_for_arm("valid", coord_rot=1)
        rf_c = _unique_ref_feats(refs_c)
        pc1, chc1 = build_query_prompt("Q1", cond)
        coord_q1, _ = _batched_score_group(
            host, connector, pc1, chc1, z1_c, z2_c, [], batch_size=batch_size, temperature=cal_temp, text_only_mode=text_only_mode
        )
        pc2, chc2 = build_query_prompt("Q2", cond, ref_arm="valid", references=refs_c, deduplicate=True)
        coord_q2, _ = _batched_score_group(
            host, connector, pc2, chc2, z1_c, z2_c, rf_c, batch_size=batch_size, temperature=cal_temp, text_only_mode=text_only_mode
        )
        pc3, chc3 = build_query_prompt("Q3", cond, protocol="repeat_x1")
        coord_q3, _ = _batched_score_group(
            host, connector, pc3, chc3, z1_c, z2_c, [], batch_size=batch_size, temperature=cal_temp, text_only_mode=text_only_mode
        )

        # Raw-duplicate negative probe (prompt deduplicate=False, emitting BOTH duplicate <REF> tokens on rot=(0,0))
        z1_0, z2_0 = codes_map[(0, 0)]
        refs_dup = batch.references_for_arm("duplicate_copy", coord_rot=0)
        p_raw_dup, ch_raw_dup = build_query_prompt("Q2", cond, ref_arm="duplicate_copy", references=refs_dup, deduplicate=False)
        rf_raw_dup = [build_ref_feature(r) for r in refs_dup]
        pr_raw_dup, _ = _batched_score_group(
            host, connector, p_raw_dup, ch_raw_dup, z1_0, z2_0, rf_raw_dup, batch_size=batch_size, temperature=cal_temp, text_only_mode=text_only_mode
        )
        raw_dup_tv = 0.5 * np.abs(pr_raw_dup - preds_q2["valid"][0]).sum(axis=1)
        raw_dup_conf_inc = float(np.max(pr_raw_dup.max(axis=1) - preds_q2["valid"][0].max(axis=1)))

        checks, three_props, strict_pass, failed_any = _summarize_rot_hold_dup(
            preds_q1, preds_q3_rep, preds_q3_ind, preds_q2, coord_q1, coord_q2, coord_q3, tol_rot_hold, tol_dup
        )
        dedup_conf_inc = float(
            max(np.max(preds_q2["duplicate_copy"][j].max(axis=1) - preds_q2["valid"][j].max(axis=1)) for j in range(3))
        )

        # Metrics
        all_q1 = compute_classification_metrics(np.concatenate(preds_q1, axis=0), np.concatenate(labels_q1, axis=0))
        all_q3_rep = compute_classification_metrics(np.concatenate(preds_q3_rep, axis=0), np.concatenate(labels_q3_rep, axis=0))
        all_q3_ind = compute_classification_metrics(np.concatenate(preds_q3_ind, axis=0), np.concatenate(labels_q3_ind, axis=0))
        all_q3_after = compute_classification_metrics(np.concatenate(preds_q3_after, axis=0), np.concatenate(labels_q3_rep, axis=0))

        q2_by_arm = {}
        pm = np.zeros((n_m, 8), dtype=np.float64)
        for arm in ARMS:
            m_arm = compute_classification_metrics(np.concatenate(preds_q2[arm], axis=0), np.concatenate(labels_q2, axis=0))
            q2_by_arm[arm] = {k: m_arm[k] for k in ("n", "acc", "nll", "brier", "ece")}
            if arm == "valid":
                pm[:, 2] = m_arm["per_item_hit"].reshape(3, n_m).mean(axis=0)
                pm[:, 3] = m_arm["per_item_nll"].reshape(3, n_m).mean(axis=0)
            elif arm == "fresh_resample":
                pm[:, 6] = m_arm["per_item_hit"].reshape(3, n_m).mean(axis=0)
                pm[:, 7] = m_arm["per_item_nll"].reshape(3, n_m).mean(axis=0)
        pm[:, 0] = all_q1["per_item_hit"].reshape(3, n_m).mean(axis=0)
        pm[:, 1] = all_q1["per_item_nll"].reshape(3, n_m).mean(axis=0)
        pm[:, 4] = all_q3_rep["per_item_hit"].reshape(3, n_m).mean(axis=0)
        pm[:, 5] = all_q3_rep["per_item_nll"].reshape(3, n_m).mean(axis=0)
        per_mother_map[cname] = pm

        none_hits = compute_classification_metrics(np.concatenate(preds_q2["none"], axis=0), np.concatenate(labels_q2, axis=0))["per_item_hit"]
        valid_hits = compute_classification_metrics(np.concatenate(preds_q2["valid"], axis=0), np.concatenate(labels_q2, axis=0))["per_item_hit"]
        corrected = int(np.sum((valid_hits == 1) & (none_hits == 0)))
        harmed = int(np.sum((valid_hits == 0) & (none_hits == 1)))

        is_coherent_enough = (cond.drift_type != "continuous" or cond.sigma_phi < 1.0)
        nontrivial = (
            all_q1["acc"] > 0.25 and all_q1["nll"] < math.log(4)
            and q2_by_arm["valid"]["acc"] > 0.25 and q2_by_arm["valid"]["nll"] < math.log(4)
            and (all_q3_rep["nll"] < math.log(4) if is_coherent_enough else True)
        )

        worst_exs = {
            "q1_rot0": extract_worst_examples(batch.mother_ids, preds_q1[0], labels_q1[0], top_k=3),
            "q3_repeat_rot0": extract_worst_examples(batch.mother_ids, preds_q3_rep[0], labels_q3_rep[0], top_k=3),
            "q2_valid_rot0": extract_worst_examples(batch.mother_ids, preds_q2["valid"][0], labels_q2[0], top_k=3),
        }

        elapsed_s = time.perf_counter() - t0
        cond_out[cname] = {
            "condition": cname,
            "mothers": int(n_m),
            "elapsed_s": round(elapsed_s, 2),
            "cal_temp": float(cal_temp),
            "tol_rot_hold": float(tol_rot_hold),
            "tol_dup": float(tol_dup),
            "q1": {k: all_q1[k] for k in ("n", "acc", "nll", "brier", "ece")},
            "q3_repeat": {k: all_q3_rep[k] for k in ("n", "acc", "nll", "brier", "ece")},
            "q3_indep": {k: all_q3_ind[k] for k in ("n", "acc", "nll", "brier", "ece")},
            "q3_after_pilot_valid": {k: all_q3_after[k] for k in ("n", "acc", "nll", "brier", "ece")},
            "q2_by_arm": q2_by_arm,
            "valid_update": {"corrected": corrected, "new_errors": harmed},
            "fresh_resample_minus_valid_q2_nll": float(q2_by_arm["fresh_resample"]["nll"] - q2_by_arm["valid"]["nll"]),
            "duplicate_max_confidence_increase": dedup_conf_inc,
            "raw_duplicate_probe_mean_tv": float(raw_dup_tv.mean()),
            "raw_duplicate_probe_max_tv": float(raw_dup_tv.max()),
            "raw_duplicate_probe_max_conf_inc": raw_dup_conf_inc,
            "three_properties": three_props,
            "all_three_pass": bool(nontrivial and strict_pass),
            "strict_vectors_pass": strict_pass,
            "mothers_failed_any": failed_any,
            "checks": checks,
            "worst_examples": worst_exs,
            "raw_loglik_all": np.concatenate(raw_ll_parts, axis=0),
            "labels_all": np.concatenate(raw_lbl_parts, axis=0),
        }
    return {"conditions": cond_out, "per_mother_map": per_mother_map}


def _direct_per_mother_on_batch(comp_name: str, model_obj, batch: MotherBatch) -> Tuple[dict, np.ndarray]:
    """Compute paired direct readout summary & (M, 8) per-mother metrics on the exact same MotherBatch."""
    cond = batch.condition
    n_m = len(batch.x1)
    preds_q1, labels_q1 = [], []
    preds_q3_rep, labels_q3_rep = [], []
    preds_q2_val, preds_q2_fresh, labels_q2 = [], [], []

    for s1, s2 in ROTATIONS:
        v1, v2 = batch.rotated_view(s1, s2)
        labels_q1.append((batch.x2 - batch.x1 + s2 - s1) % 4)
        labels_q2.append((batch.x1 + s1) % 4)
        labels_q3_rep.append(sector_of_complex(matched_sum(batch.y3_repeat * ROOTS[s1 % 4])))

        if comp_name == "full_iq_known_param":
            out_n = solve_full_iq_known_param(v1, v2, (), cond, protocol="repeat_x1")
            out_v = solve_full_iq_known_param(v1, v2, batch.references_for_arm("valid"), cond, protocol="repeat_x1")
            out_f = solve_full_iq_known_param(v1, v2, batch.references_for_arm("fresh_resample"), cond, protocol="repeat_x1")
        else:
            z1 = np.asarray(model_obj.encode(v1), dtype=np.int64)
            z2 = np.asarray(model_obj.encode(v2), dtype=np.int64)
            a1, a2 = model_obj.table[z1], model_obj.table[z2]
            out_n = readout_from_table(a1, a2, (), cond, protocol="repeat_x1")
            out_v = readout_from_table(a1, a2, batch.references_for_arm("valid"), cond, protocol="repeat_x1")
            out_f = readout_from_table(a1, a2, batch.references_for_arm("fresh_resample"), cond, protocol="repeat_x1")
        preds_q1.append(out_n["q1"])
        preds_q3_rep.append(out_n["q3"])
        preds_q2_val.append(out_v["q2"])
        preds_q2_fresh.append(out_f["q2"])

    m_q1 = compute_classification_metrics(np.concatenate(preds_q1, axis=0), np.concatenate(labels_q1, axis=0))
    m_q3 = compute_classification_metrics(np.concatenate(preds_q3_rep, axis=0), np.concatenate(labels_q3_rep, axis=0))
    m_v = compute_classification_metrics(np.concatenate(preds_q2_val, axis=0), np.concatenate(labels_q2, axis=0))
    m_f = compute_classification_metrics(np.concatenate(preds_q2_fresh, axis=0), np.concatenate(labels_q2, axis=0))

    pm = np.stack(
        [
            m_q1["per_item_hit"].reshape(3, n_m).mean(axis=0),
            m_q1["per_item_nll"].reshape(3, n_m).mean(axis=0),
            m_v["per_item_hit"].reshape(3, n_m).mean(axis=0),
            m_v["per_item_nll"].reshape(3, n_m).mean(axis=0),
            m_q3["per_item_hit"].reshape(3, n_m).mean(axis=0),
            m_q3["per_item_nll"].reshape(3, n_m).mean(axis=0),
            m_f["per_item_hit"].reshape(3, n_m).mean(axis=0),
            m_f["per_item_nll"].reshape(3, n_m).mean(axis=0),
        ],
        axis=1,
    )
    summary = {
        "q1": {k: m_q1[k] for k in ("acc", "nll", "ece")},
        "q3_repeat": {k: m_q3[k] for k in ("acc", "nll", "ece")},
        "q2_valid": {k: m_v[k] for k in ("acc", "nll", "ece")},
        "q2_fresh": {k: m_f[k] for k in ("acc", "nll", "ece")},
    }
    return summary, pm


def _evaluate_question_order_probe(
    host: MinistralCacheHost,
    connector: WaveformCacheConnector,
    enc,
    split: str,
    cal_temp: float,
    batch_size: int = 32,
) -> dict:
    """Test unseen C6 multi-turn question order (Q3 pre-reveal asked BEFORE Q2 in chat history)."""
    cond = CONDITION_SPECS["C6_decoherent_order"]
    batch = generate_mothers(split, "C6_decoherent_order", max_blocks=1)
    z1_0 = np.asarray(enc.encode(batch.y1), dtype=np.int64)
    z2_0 = np.asarray(enc.encode(batch.y2), dtype=np.int64)
    refs_v = batch.references_for_arm("valid", coord_rot=0)
    p3, ch3 = build_query_prompt("Q3", cond, protocol="repeat_x1")
    p2, ch2 = build_query_prompt("Q2", cond, ref_arm="valid", references=refs_v, deduplicate=True)
    rf = _unique_ref_feats(refs_v)

    q2_standalone, _ = _batched_score_group(
        host, connector, p2, ch2, z1_0, z2_0, rf, batch_size=batch_size, temperature=cal_temp
    )
    extra_hist = [
        {"role": "user", "content": p3},
        {"role": "assistant", "content": ch3[0]},
    ]
    # Re-render turn 2 with only ref_text + question lines so <W1><W2> is supplied once in turn 1
    lines_p2 = p2.splitlines()
    p2_turn2 = "基于同一段已编码波形：\n" + "\n".join(lines_p2[2:])
    q2_after_q3, _ = _batched_score_group(
        host, connector, p2_turn2, ch2, z1_0, z2_0, rf, batch_size=batch_size, temperature=cal_temp, extra_history=extra_hist
    )
    tv = 0.5 * np.abs(q2_after_q3 - q2_standalone).sum(axis=1)
    m_st = compute_classification_metrics(q2_standalone, batch.x1)
    m_mt = compute_classification_metrics(q2_after_q3, batch.x1)
    return {
        "mothers": int(len(batch.x1)),
        "q2_standalone_acc": float(m_st["acc"]),
        "q2_standalone_nll": float(m_st["nll"]),
        "q2_after_q3_acc": float(m_mt["acc"]),
        "q2_after_q3_nll": float(m_mt["nll"]),
        "mean_tv": float(tv.mean()),
        "max_tv": float(tv.max()),
    }


def _strip_heavy_arrays(eval_res: dict) -> dict:
    clean = {}
    for cname, cdict in eval_res["conditions"].items():
        clean[cname] = {k: v for k, v in cdict.items() if k not in ("raw_loglik_all", "labels_all")}
    return clean


def run_blind_host_evaluation(
    out_dir: Optional[Path] = None,
    batch_size: int = 64,
    test_blocks_override: Optional[int] = None,
) -> dict:
    """Load dev-locked connector checkpoints + thresholds and run blind s3_test evaluation."""
    out_dir = out_dir or STAGE3_DIR
    res_dir = out_dir / "results" / "host"
    weights_dir = out_dir / "connector_weights"
    lock_path = res_dir / "host_dev_lock.json"
    if not lock_path.exists():
        raise FileNotFoundError(f"Missing dev lock file {lock_path}; run train_connector.py --train first.")

    dev_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    test_blocks = int(test_blocks_override or dev_lock["host_test_blocks_locked"])
    cond_names = list(CONDITION_SPECS.keys())

    t_total0 = time.perf_counter()
    host = MinistralCacheHost(model_id=dev_lock["model_id"])
    assert host.host_sha256 == dev_lock["host_sha256"], (
        f"Host fingerprint mismatch: {host.host_sha256} != {dev_lock['host_sha256']}"
    )
    if host.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(host.device)

    models = load_frozen_models()
    encoders = list(dev_lock["encoders"].keys())

    host_by_encoder: Dict[str, dict] = {}
    host_pm_by_encoder: Dict[str, Dict[str, np.ndarray]] = {}
    worst_dump: Dict[str, dict] = {}
    eq_connector = None

    for enc_name in encoders:
        enc_lock = dev_lock["encoders"][enc_name]
        ckpt_path = weights_dir / f"connector_{enc_name}.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        connector = WaveformCacheConnector(
            num_states=int(ckpt["num_states"]),
            hidden_size=int(ckpt["hidden_size"]),
            target_embed_std=float(ckpt["target_embed_std"]),
        ).to(host.device)
        connector.load_state_dict(ckpt["state_dict"])
        connector.eval()
        if enc_name == "equivariant_b8_s17":
            eq_connector = connector

        t_enc0 = time.perf_counter()
        raw_res = evaluate_host_on_split_subset(
            host=host,
            connector=connector,
            enc=models[enc_name],
            split="s3_test",
            cond_names=cond_names,
            max_blocks=test_blocks,
            batch_size=batch_size,
            cal_temp=float(enc_lock["locked_temp"]),
            tol_rot_hold=float(enc_lock["locked_tau_rot_hold"]),
            tol_dup=float(enc_lock["locked_tau_dup"]),
        )
        enc_wall = time.perf_counter() - t_enc0
        clean_conds = _strip_heavy_arrays(raw_res)
        host_by_encoder[enc_name] = {
            "wall_time_s": round(enc_wall, 2),
            "locked_temp": float(enc_lock["locked_temp"]),
            "locked_tau_rot_hold": float(enc_lock["locked_tau_rot_hold"]),
            "locked_tau_dup": float(enc_lock["locked_tau_dup"]),
            "conditions": clean_conds,
        }
        host_pm_by_encoder[enc_name] = raw_res["per_mother_map"]
        for cname in cond_names:
            worst_dump.setdefault(cname, {})[f"host_{enc_name}"] = clean_conds[cname]["worst_examples"]
            c_m = clean_conds[cname]
            print(
                f"[HOST {enc_name} | {cname}] Q1={c_m['q1']['acc']:.4f} "
                f"Q3_rep={c_m['q3_repeat']['acc']:.4f} (NLL={c_m['q3_repeat']['nll']:.4f}) "
                f"Q2_val={c_m['q2_by_arm']['valid']['acc']:.4f} "
                f"pass={c_m['all_three_pass']} raw_dup_tv={c_m['raw_duplicate_probe_mean_tv']:.4f}"
            )
        if enc_name != "equivariant_b8_s17":
            del connector
        gc.collect()
        if host.device.type == "cuda":
            torch.cuda.empty_cache()

    # Controls on primary equivariant_b8_s17
    prim_enc_name = "equivariant_b8_s17" if "equivariant_b8_s17" in encoders else encoders[0]
    prim_lock = dev_lock["encoders"][prim_enc_name]
    prim_conn = eq_connector
    if prim_conn is None:
        ckpt = torch.load(weights_dir / f"connector_{prim_enc_name}.pt", map_location="cpu", weights_only=True)
        prim_conn = WaveformCacheConnector(int(ckpt["num_states"]), int(ckpt["hidden_size"]), float(ckpt["target_embed_std"])).to(host.device)
        prim_conn.load_state_dict(ckpt["state_dict"])
        prim_conn.eval()

    ctrl_text = evaluate_host_on_split_subset(
        host=host,
        connector=prim_conn,
        enc=models[prim_enc_name],
        split="s3_test",
        cond_names=cond_names,
        max_blocks=test_blocks,
        batch_size=batch_size,
        cal_temp=float(prim_lock["locked_temp"]),
        tol_rot_hold=float(prim_lock["locked_tau_rot_hold"]),
        tol_dup=float(prim_lock["locked_tau_dup"]),
        text_only_mode=True,
    )
    ctrl_shuf = evaluate_host_on_split_subset(
        host=host,
        connector=prim_conn,
        enc=models[prim_enc_name],
        split="s3_test",
        cond_names=cond_names,
        max_blocks=test_blocks,
        batch_size=batch_size,
        cal_temp=float(prim_lock["locked_temp"]),
        tol_rot_hold=float(prim_lock["locked_tau_rot_hold"]),
        tol_dup=float(prim_lock["locked_tau_dup"]),
        shuffled_z=True,
    )
    c6_order_probe = _evaluate_question_order_probe(
        host, prim_conn, models[prim_enc_name], "s3_test", float(prim_lock["locked_temp"]), batch_size=batch_size
    )

    # Paired direct readout & full_iq on the EXACT same s3_test mothers + Paired Bootstrap CIs
    metric_names = ("q1_acc", "q1_nll", "q2_valid_acc", "q2_valid_nll", "q3_rep_acc", "q3_rep_nll", "q2_fresh_acc", "q2_fresh_nll")
    paired_direct_refs: Dict[str, dict] = {}
    paired_bootstrap_cis: Dict[str, dict] = {}

    for cname in cond_names:
        batch = generate_mothers("s3_test", cname, max_blocks=test_blocks)
        direct_pm = {}
        direct_summ = {}
        for dname in ["full_iq_known_param"] + encoders:
            ds, dpm = _direct_per_mother_on_batch(dname, models.get(dname), batch)
            direct_summ[dname] = ds
            direct_pm[dname] = dpm
        paired_direct_refs[cname] = direct_summ

        ci_map = {}
        for ename in encoders:
            diff_hd = host_pm_by_encoder[ename][cname] - direct_pm[ename]
            ci_map[f"host_{ename}_minus_direct_{ename}"] = {
                mname: paired_bootstrap_ci(diff_hd[:, k]) for k, mname in enumerate(metric_names)
            }
        if "equivariant_b8_s17" in encoders and "hand_b8" in encoders:
            diff_eh = host_pm_by_encoder["equivariant_b8_s17"][cname] - host_pm_by_encoder["hand_b8"][cname]
            ci_map["host_eq_b8_minus_host_hand_b8"] = {
                mname: paired_bootstrap_ci(diff_eh[:, k]) for k, mname in enumerate(metric_names)
            }
        if "equivariant_b8_s17" in encoders and "plain_b8_s17" in encoders:
            diff_ep = host_pm_by_encoder["equivariant_b8_s17"][cname] - host_pm_by_encoder["plain_b8_s17"][cname]
            ci_map["host_eq_b8_minus_host_plain_b8"] = {
                mname: paired_bootstrap_ci(diff_ep[:, k]) for k, mname in enumerate(metric_names)
            }
        if "equivariant_b8_s17" in encoders:
            diff_fi = host_pm_by_encoder["equivariant_b8_s17"][cname] - direct_pm["full_iq_known_param"]
            ci_map["host_eq_b8_minus_full_iq"] = {
                mname: paired_bootstrap_ci(diff_fi[:, k]) for k, mname in enumerate(metric_names)
            }
        paired_bootstrap_cis[cname] = ci_map

    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated(host.device) / (1024.0 ** 2))
        if host.device.type == "cuda"
        else 0.0
    )
    gpu_name = torch.cuda.get_device_name(host.device) if host.device.type == "cuda" else "CPU"
    host_summary = {
        "split": "s3_test",
        "test_blocks": int(test_blocks),
        "mothers_per_condition": int(test_blocks * 64),
        "model_id": dev_lock["model_id"],
        "hf_sha": host.hf_sha,
        "dtype": str(host.dtype),
        "gpu_name": gpu_name,
        "peak_vram_mb": round(peak_vram_mb, 1),
        "total_eval_wall_s": round(time.perf_counter() - t_total0, 2),
        "host_sha256": host.host_sha256,
        "encoders": host_by_encoder,
        "controls": {
            "text_only": _strip_heavy_arrays(ctrl_text),
            "shuffled_Z": _strip_heavy_arrays(ctrl_shuf),
            "c6_multi_turn_order_probe": c6_order_probe,
        },
        "paired_direct_same_mothers": paired_direct_refs,
        "paired_bootstrap_ci": paired_bootstrap_cis,
    }

    (res_dir / "host_summary.json").write_text(json.dumps(host_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (res_dir / "host_worst_examples.json").write_text(json.dumps(worst_dump, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] Saved host_summary.json & host_worst_examples.json in {res_dir} (total {host_summary['total_eval_wall_s']}s)")
    return host_summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--test-blocks", type=int, default=None)
    args = ap.parse_args()
    run_blind_host_evaluation(batch_size=args.batch_size, test_blocks_override=args.test_blocks)
