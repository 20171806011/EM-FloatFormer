"""Evaluate frozen direct readouts across R0 regression and Stage 3 C1..C6 conditions."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from research.stage3_ministral_cache.analytic import (
    readout_from_table,
    solve_full_iq_known_param,
)
from research.stage3_ministral_cache.data import PARENT_PARTITIONS, generate_mothers
from research.stage3_ministral_cache.schema import (
    ARMS,
    CONDITION_SPECS,
    ConditionSpec,
    PROTOCOLS,
    ROOTS,
    ROTATIONS,
    Reference,
    dedup_references,
    matched_sum,
    sector_of_complex,
)
from research.stage3_ministral_cache.scoring import (
    check_vector_relation,
    compute_classification_metrics,
    compute_complex_nmse,
    extract_worst_examples,
    paired_bootstrap_ci,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MATCHED_DIR = REPO_ROOT / "research" / "matched_cache"
PAIRED_DIR = REPO_ROOT / "research" / "fourphase_paired_questions"
STAGE3_DIR = REPO_ROOT / "research" / "stage3_ministral_cache"

TOLERANCE = 3e-13
SEEDS = (17, 29, 43)
BUDGETS = (4, 8)
VARIANTS = ("plain", "equivariant")


def load_frozen_models() -> Dict[str, object]:
    """Load exact physical Hand(4/8) and all 12 frozen .npz learned models (allow_pickle=False)."""
    if str(MATCHED_DIR) not in sys.path:
        sys.path.insert(0, str(MATCHED_DIR))
    from physical import Hand
    import learned as learned_mod

    models: Dict[str, object] = {"hand_b4": Hand(4), "hand_b8": Hand(8)}
    for variant in VARIANTS:
        for bits in BUDGETS:
            for seed in SEEDS:
                tag = f"{variant}_b{bits}_s{seed}"
                models[tag] = learned_mod.load(MATCHED_DIR / "weights" / f"{tag}.npz")
    return models


def run_regression_replay_check(models: Dict[str, object]) -> dict:
    """Verify exact <1e-12 replay parity with 504f3a44 matched_cache/results/summary.json."""
    if str(MATCHED_DIR) not in sys.path:
        sys.path.insert(0, str(MATCHED_DIR))
    import evaluate as parent_eval

    published = json.loads((MATCHED_DIR / "results" / "summary.json").read_text(encoding="utf-8"))
    max_diff = 0.0
    details = {}
    for split in ("dev", "test"):
        split_pub = published[split]
        details[split] = {}
        for bits in BUDGETS:
            for name, mobj in [(f"hand_b{bits}", models[f"hand_b{bits}"])] + [
                (f"{v}_b{bits}_s{s}", models[f"{v}_b{bits}_s{s}"]) for v in VARIANTS for s in SEEDS
            ]:
                res, _ = parent_eval.evaluate(mobj, bits, split)
                if name.startswith("hand_"):
                    expected = split_pub[name]["means"]
                    got = res["arms"]["valid"]
                    key_map = (
                        ("q1_accuracy", "q1_accuracy"),
                        ("q1_nll", "q1_nll"),
                        ("q2_accuracy", "q2_valid_accuracy"),
                        ("q2_nll", "q2_valid_nll"),
                    )
                    for got_k, exp_k in key_map:
                        d = abs(float(got[got_k]) - float(expected[exp_k]))
                        max_diff = max(max_diff, d)
                    assert bool(res["all_three"]["pass"]) == bool(split_pub[name]["passes"])
                    details[split][name] = {"pass": res["all_three"]["pass"], "valid": got}
                else:
                    variant = name.split("_")[0]
                    seed = int(name.split("_s")[1])
                    idx = SEEDS.index(seed)
                    assert bool(res["all_three"]["pass"]) == bool(split_pub[f"{variant}_b{bits}"]["passes"][idx])
                    details[split][name] = {"pass": res["all_three"]["pass"], "valid": res["arms"]["valid"]}
        # Also verify mean across seeds matches published means to < 1e-12
        for bits in BUDGETS:
            for variant in VARIANTS:
                agg = {
                    k: float(np.mean([details[split][f"{variant}_b{bits}_s{s}"]["valid"][k] for s in SEEDS]))
                    for k in ("q1_accuracy", "q1_nll", "q2_accuracy", "q2_nll")
                }
                pub_m = split_pub[f"{variant}_b{bits}"]["means"]
                for k_src, k_pub in zip(
                    ("q1_accuracy", "q1_nll", "q2_accuracy", "q2_nll"),
                    ("q1_accuracy", "q1_nll", "q2_valid_accuracy", "q2_valid_nll"),
                ):
                    max_diff = max(max_diff, abs(agg[k_src] - float(pub_m[k_pub])))
    assert max_diff < 1e-12, f"Regression mismatch vs 504f3a44: {max_diff}"
    return {"max_abs_diff_vs_504f3a44": max_diff, "regression_pass": True, "details": details}


def evaluate_comparator_on_condition(
    comp_name: str,
    model_obj: Optional[object],
    batch,
    write_jsonl_path: Optional[Path] = None,
) -> Tuple[dict, dict]:
    """Evaluate one comparator (`prior_no_waveform`, `full_iq_known_param`, or frozen cache model).

    Pre-reveal predictions (Q1 and Q3 repeat_x1 / independent_x3) are written to JSONL BEFORE
    scoring against realized noisy Window-3 samples. Returns `(summary_dict, per_mother_vectors)`.
    """
    cond = batch.condition
    n_m = len(batch.x1)
    snr = cond.snr

    # Containers per rotation j in ROTATIONS
    preds_q1 = []
    preds_q3_rep = []
    preds_q3_ind = []
    mu3_rep_list = []
    mu3_ind_list = []
    preds_q2: Dict[str, List[np.ndarray]] = {arm: [] for arm in ARMS}
    preds_q3_after_rep_valid: List[np.ndarray] = []

    y3_rep_rot_list = []
    y3_ind_rot_list = []
    labels_q1 = []
    labels_q2 = []
    labels_q3_rep = []
    labels_q3_ind = []

    t0 = time.perf_counter()
    for s1, s2 in ROTATIONS:
        v1, v2 = batch.rotated_view(s1, s2)
        # Note: Y3 is always in receiver frame (repeat_x1 has X3=X1, so if Y1,Y2 both rotated by s1=s2,
        # the physical transmitted symbol rotated by s1; for paired check, true noisy observation is batch.y3_repeat * ROOTS[s1] under (s1, s1))
        # WAIT: Careful! Let's follow PROTOCOL.md Section 5:
        # In paired data rotation tests:
        #   - (0, 0): windows=(Y1, Y2), X1'=X1, X2'=X2 -> Y3_rep'=Y3_rep (label K3(Y3_rep))
        #   - (0, 1): windows=(Y1, i*Y2), X1'=X1, X2'=X2+1 -> Y3_rep'=Y3_rep (because X3=X1 didn't change!)
        #   - (1, 1): windows=(i*Y1, i*Y2), X1'=X1+1, X2'=X2+1 -> Y3_rep' = i*Y3_rep (so true rotated Window 3 matches rotated X1+1!)
        y3_r = batch.y3_repeat * ROOTS[s1 % 4]
        y3_i = batch.y3_indep * ROOTS[s1 % 4]
        y3_rep_rot_list.append(y3_r)
        y3_ind_rot_list.append(y3_i)
        labels_q1.append((batch.x2 - batch.x1 + s2 - s1) % 4)
        labels_q2.append((batch.x1 + s1) % 4)
        labels_q3_rep.append(sector_of_complex(matched_sum(y3_r)))
        labels_q3_ind.append(sector_of_complex(matched_sum(y3_i)))

        if comp_name == "prior_no_waveform":
            u = np.full((n_m, 4), 0.25, dtype=np.float64)
            z0 = np.zeros(n_m, dtype=np.complex128)
            preds_q1.append(u)
            preds_q3_rep.append(u)
            preds_q3_ind.append(u)
            mu3_rep_list.append(z0)
            mu3_ind_list.append(z0)
            for arm in ARMS:
                preds_q2[arm].append(u)
            preds_q3_after_rep_valid.append(u)
        elif comp_name == "full_iq_known_param":
            out_rep_none = solve_full_iq_known_param(v1, v2, (), cond, protocol="repeat_x1")
            out_ind_none = solve_full_iq_known_param(v1, v2, (), cond, protocol="independent_x3")
            preds_q1.append(out_rep_none["q1"])
            preds_q3_rep.append(out_rep_none["q3"])
            preds_q3_ind.append(out_ind_none["q3"])
            mu3_rep_list.append(out_rep_none["mu3"])
            mu3_ind_list.append(out_ind_none["mu3"])
            for arm in ARMS:
                refs = batch.references_for_arm(arm, coord_rot=0)
                out_arm = solve_full_iq_known_param(v1, v2, refs, cond, protocol="repeat_x1")
                preds_q2[arm].append(out_arm["q2"])
                if arm == "valid":
                    preds_q3_after_rep_valid.append(out_arm["q3_after_pilot"])
        else:
            # Frozen encoder path: encode once per (s1, s2), discard raw v1, v2!
            z1 = np.asarray(model_obj.encode(v1), dtype=np.int64)
            z2 = np.asarray(model_obj.encode(v2), dtype=np.int64)
            a1 = model_obj.table[z1]
            a2 = model_obj.table[z2]
            out_rep_none = readout_from_table(a1, a2, (), cond, protocol="repeat_x1")
            out_ind_none = readout_from_table(a1, a2, (), cond, protocol="independent_x3")
            preds_q1.append(out_rep_none["q1"])
            preds_q3_rep.append(out_rep_none["q3"])
            preds_q3_ind.append(out_ind_none["q3"])
            mu3_rep_list.append(out_rep_none["mu3"])
            mu3_ind_list.append(out_ind_none["mu3"])
            for arm in ARMS:
                refs = batch.references_for_arm(arm, coord_rot=0)
                out_arm = readout_from_table(a1, a2, refs, cond, protocol="repeat_x1")
                preds_q2[arm].append(out_arm["q2"])
                if arm == "valid":
                    preds_q3_after_rep_valid.append(out_arm["q3_after_pilot"])

    # Pre-reveal log serialization BEFORE scoring
    if write_jsonl_path is not None:
        write_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with write_jsonl_path.open("w", encoding="utf-8") as fh:
            for j, (s1, s2) in enumerate(ROTATIONS):
                for i in range(n_m):
                    rec = {
                        "comparator": comp_name,
                        "condition": cond.name,
                        "mother_id": batch.mother_ids[i],
                        "rot": [s1, s2],
                        "q1_pred": [round(float(x), 6) for x in preds_q1[j][i]],
                        "q3_repeat_pre_reveal_pred": [round(float(x), 6) for x in preds_q3_rep[j][i]],
                        "q3_repeat_mu3": [round(float(mu3_rep_list[j][i].real), 6), round(float(mu3_rep_list[j][i].imag), 6)],
                        "q3_indep_pre_reveal_pred": [round(float(x), 6) for x in preds_q3_ind[j][i]],
                        "q2_valid_pred": [round(float(x), 6) for x in preds_q2["valid"][j][i]],
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Coordinate rotation test: rotate BOTH data windows AND references (+1 quarter turn) for variant (0,0)
    v1_c, v2_c = batch.rotated_view(1, 1)
    refs_c = batch.references_for_arm("valid", coord_rot=1)
    if comp_name == "prior_no_waveform":
        coord_q1 = preds_q1[0]
        coord_q2 = preds_q2["valid"][0]
        coord_q3 = preds_q3_rep[0]
    elif comp_name == "full_iq_known_param":
        out_c = solve_full_iq_known_param(v1_c, v2_c, refs_c, cond, protocol="repeat_x1")
        coord_q1, coord_q2, coord_q3 = out_c["q1"], out_c["q2"], out_c["q3"]
    else:
        z1_c = np.asarray(model_obj.encode(v1_c), dtype=np.int64)
        z2_c = np.asarray(model_obj.encode(v2_c), dtype=np.int64)
        out_c = readout_from_table(model_obj.table[z1_c], model_obj.table[z2_c], refs_c, cond, protocol="repeat_x1")
        coord_q1, coord_q2, coord_q3 = out_c["q1"], out_c["q2"], out_c["q3"]

    # Double-counting negative control on (0, 0) duplicate_copy
    double_refs = batch.references_for_arm("duplicate_copy")
    if comp_name == "prior_no_waveform":
        wrong_q2 = preds_q2["valid"][0]
    elif comp_name == "full_iq_known_param":
        wrong_q2 = solve_full_iq_known_param(
            batch.y1, batch.y2, double_refs, cond, allow_double_count_unsafe=True
        )["q2"]
    else:
        z1_0 = np.asarray(model_obj.encode(batch.y1), dtype=np.int64)
        z2_0 = np.asarray(model_obj.encode(batch.y2), dtype=np.int64)
        wrong_q2 = readout_from_table(
            model_obj.table[z1_0],
            model_obj.table[z2_0],
            double_refs,
            cond,
            allow_double_count_unsafe=True,
        )["q2"]

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    # --- Invariance checks (exact match to 3e-13 tolerance) ---
    checks = {}
    fails_prop = {"rotate": np.zeros(n_m, dtype=bool), "hold": np.zeros(n_m, dtype=bool), "duplicate": np.zeros(n_m, dtype=bool)}
    max_conf_inc = 0.0

    def rec_check(cname: str, actual: np.ndarray, expected: np.ndarray, prop: str):
        r = check_vector_relation(actual, expected, tol=TOLERANCE)
        fails_prop[prop] |= r["failed_mask"]
        checks[cname] = {k: v for k, v in r.items() if k != "failed_mask"}

    # Rotations (0,1) [idx=1] and (1,1) [idx=2] vs (0,0) [idx=0]
    rec_check("q1_rot01_rotates", preds_q1[1], np.roll(preds_q1[0], 1, axis=1), "rotate")
    rec_check("q1_rot11_holds", preds_q1[2], preds_q1[0], "hold")
    rec_check("q3_rep_rot01_holds", preds_q3_rep[1], preds_q3_rep[0], "hold")
    rec_check("q3_rep_rot11_rotates", preds_q3_rep[2], np.roll(preds_q3_rep[0], 1, axis=1), "rotate")
    rec_check("q3_ind_rot01_holds", preds_q3_ind[1], preds_q3_ind[0], "hold")
    rec_check("q3_ind_rot11_holds", preds_q3_ind[2], preds_q3_ind[0], "hold")
    rec_check("q3_ind_uniform", preds_q3_ind[0], np.full_like(preds_q3_ind[0], 0.25), "hold")

    for j in range(len(ROTATIONS)):
        if j == 1:
            rec_check("q2_valid_rot01_holds", preds_q2["valid"][1], preds_q2["valid"][0], "hold")
        elif j == 2:
            rec_check("q2_valid_rot11_rotates", preds_q2["valid"][2], np.roll(preds_q2["valid"][0], 1, axis=1), "rotate")
        rec_check(f"q2_indep_equals_none_r{j}", preds_q2["independent"][j], preds_q2["none"][j], "hold")
        rec_check(f"q2_dup_equals_valid_r{j}", preds_q2["duplicate_copy"][j], preds_q2["valid"][j], "duplicate")
        inc = float(np.max(preds_q2["duplicate_copy"][j].max(axis=1) - preds_q2["valid"][j].max(axis=1)))
        max_conf_inc = max(max_conf_inc, inc)

    rec_check("coord_q1_holds", coord_q1, preds_q1[0], "hold")
    rec_check("coord_q2_holds", coord_q2, preds_q2["valid"][0], "hold")
    rec_check("coord_q3_rotates", coord_q3, np.roll(preds_q3_rep[0], 1, axis=1), "rotate")

    wrong_tv = float(np.mean(0.5 * np.abs(wrong_q2 - preds_q2["valid"][0]).sum(axis=1)))

    # --- Accuracy, NLL, ECE, Brier, NMSE aggregated across the 3 rotation variants per mother ---
    # Per-mother [Q1_acc, Q1_nll, Q2_valid_acc, Q2_valid_nll, Q3_rep_acc, Q3_rep_nll, Q2_fresh_acc, Q2_fresh_nll]
    pm = np.zeros((n_m, 8), dtype=np.float64)
    all_q1 = compute_classification_metrics(np.concatenate(preds_q1, axis=0), np.concatenate(labels_q1, axis=0))
    all_q3_rep = compute_classification_metrics(np.concatenate(preds_q3_rep, axis=0), np.concatenate(labels_q3_rep, axis=0))
    all_q3_ind = compute_classification_metrics(np.concatenate(preds_q3_ind, axis=0), np.concatenate(labels_q3_ind, axis=0))
    all_q3_after = compute_classification_metrics(
        np.concatenate(preds_q3_after_rep_valid, axis=0), np.concatenate(labels_q3_rep, axis=0)
    )
    nmse_rep = compute_complex_nmse(np.concatenate(mu3_rep_list, axis=0), matched_sum(np.concatenate(y3_rep_rot_list, axis=0)))
    nmse_ind = compute_complex_nmse(np.concatenate(mu3_ind_list, axis=0), matched_sum(np.concatenate(y3_ind_rot_list, axis=0)))

    q2_by_arm = {}
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

    # Net helped vs harmed on valid pilot vs none
    none_hits = compute_classification_metrics(np.concatenate(preds_q2["none"], axis=0), np.concatenate(labels_q2, axis=0))["per_item_hit"]
    valid_hits = compute_classification_metrics(np.concatenate(preds_q2["valid"], axis=0), np.concatenate(labels_q2, axis=0))["per_item_hit"]
    corrected = int(np.sum((valid_hits == 1) & (none_hits == 0)))
    harmed = int(np.sum((valid_hits == 0) & (none_hits == 1)))

    fresh_minus_valid_nll = q2_by_arm["fresh_resample"]["nll"] - q2_by_arm["valid"]["nll"]

    three_props = {k: {"mothers_failed": int(v.sum()), "pass": not bool(v.any())} for k, v in fails_prop.items()}
    failed_union = fails_prop["rotate"] | fails_prop["hold"] | fails_prop["duplicate"]
    is_coherent_enough = (cond.drift_type != "continuous" or cond.sigma_phi < 1.0)
    nontrivial = (
        all_q1["acc"] > 0.25 and all_q1["nll"] < math.log(4)
        and q2_by_arm["valid"]["acc"] > 0.25 and q2_by_arm["valid"]["nll"] < math.log(4)
        and (all_q3_rep["nll"] < math.log(4) if is_coherent_enough else True)
    )

    worst_q3_rep = extract_worst_examples(batch.mother_ids, preds_q3_rep[0], labels_q3_rep[0], top_k=3)
    worst_q2_valid = extract_worst_examples(batch.mother_ids, preds_q2["valid"][0], labels_q2[0], top_k=3)

    summary = {
        "comparator": comp_name,
        "condition": cond.name,
        "mothers": int(n_m),
        "elapsed_ms": float(elapsed_ms),
        "q1": {k: all_q1[k] for k in ("n", "acc", "nll", "brier", "ece")},
        "q3_repeat": {**{k: all_q3_rep[k] for k in ("n", "acc", "nll", "brier", "ece")}, "nmse": nmse_rep},
        "q3_indep": {**{k: all_q3_ind[k] for k in ("n", "acc", "nll", "brier", "ece")}, "nmse": nmse_ind},
        "q3_after_pilot_valid": {k: all_q3_after[k] for k in ("n", "acc", "nll", "brier", "ece")},
        "q2_by_arm": q2_by_arm,
        "valid_update": {"corrected": corrected, "new_errors": harmed},
        "fresh_resample_minus_valid_q2_nll": float(fresh_minus_valid_nll),
        "duplicate_max_confidence_increase": float(max_conf_inc),
        "double_count_negative_control_mean_tv": float(wrong_tv),
        "three_properties": three_props,
        "all_three_pass": bool(nontrivial and not failed_union.any()),
        "strict_vectors_pass": not bool(failed_union.any()),
        "mothers_failed_any": int(failed_union.sum()),
        "checks": checks,
        "worst_examples": {"q3_repeat_rot0": worst_q3_rep, "q2_valid_rot0": worst_q2_valid},
    }
    return summary, {"per_mother": pm}


def run_all_direct(
    split: str = "s3_test",
    out_dir: Optional[Path] = None,
    max_blocks: Optional[int] = None,
    write_jsonl: bool = True,
) -> dict:
    out_dir = out_dir or (STAGE3_DIR / "results" / "direct")
    out_dir.mkdir(parents=True, exist_ok=True)
    models = load_frozen_models()

    reg = run_regression_replay_check(models)
    print(f"[R0 Regression] pass={reg['regression_pass']} max_abs_diff={reg['max_abs_diff_vs_504f3a44']:.3e}")

    comp_order = ["prior_no_waveform", "full_iq_known_param", "hand_b4", "hand_b8"] + [
        f"{v}_b{b}_s{s}" for v in VARIANTS for b in BUDGETS for s in SEEDS
    ]

    all_summaries: Dict[str, dict] = {"regression_504f3a44": reg, "split": split, "conditions": {}}
    worst_dump = {}

    for cname in CONDITION_SPECS:
        batch = generate_mothers(split, cname, max_blocks=max_blocks)
        cond_res = {}
        pm_map = {}
        for comp in comp_order:
            jsonl_p = (
                (out_dir / "pre_reveal_jsonl" / f"{cname}__{comp}.jsonl")
                if (write_jsonl and comp in ("full_iq_known_param", "hand_b4", "hand_b8", "equivariant_b8_s17", "plain_b8_s17"))
                else None
            )
            summ, vecs = evaluate_comparator_on_condition(comp, models.get(comp), batch, write_jsonl_path=jsonl_p)
            cond_res[comp] = summ
            pm_map[comp] = vecs["per_mother"]
            if comp in ("equivariant_b8_s17", "hand_b8", "full_iq_known_param"):
                worst_dump.setdefault(cname, {})[comp] = summ["worst_examples"]

        # Aggregate seed means & paired bootstrap CIs vs hand_b8 and equivariant_b8 vs hand_b8
        paired_ci = {}
        metric_names = ("q1_acc", "q1_nll", "q2_valid_acc", "q2_valid_nll", "q3_rep_acc", "q3_rep_nll", "q2_fresh_acc", "q2_fresh_nll")
        for b in BUDGETS:
            for v in VARIANTS:
                seed_pms = [pm_map[f"{v}_b{b}_s{s}"] for s in SEEDS]
                mean_pm = np.mean(seed_pms, axis=0)
                hand_pm = pm_map[f"hand_b{b}"]
                diff = mean_pm - hand_pm
                paired_ci[f"{v}_b{b}_minus_hand_b{b}"] = {
                    mname: paired_bootstrap_ci(diff[:, k]) for k, mname in enumerate(metric_names)
                }
            # Equivariant b8 minus full_iq_known_param
        eq8_mean = np.mean([pm_map[f"equivariant_b8_s{s}"] for s in SEEDS], axis=0)
        diff_eq_iq = eq8_mean - pm_map["full_iq_known_param"]
        paired_ci["equivariant_b8_minus_full_iq"] = {
            mname: paired_bootstrap_ci(diff_eq_iq[:, k]) for k, mname in enumerate(metric_names)
        }
        diff_h8_iq = pm_map["hand_b8"] - pm_map["full_iq_known_param"]
        paired_ci["hand_b8_minus_full_iq"] = {
            mname: paired_bootstrap_ci(diff_h8_iq[:, k]) for k, mname in enumerate(metric_names)
        }

        all_summaries["conditions"][cname] = {
            "by_comparator": cond_res,
            "paired_bootstrap_ci": paired_ci,
        }
        print(
            f"[{cname}] full_iq Q1={cond_res['full_iq_known_param']['q1']['acc']:.4f} "
            f"Q3_rep={cond_res['full_iq_known_param']['q3_repeat']['acc']:.4f} "
            f"(NLL={cond_res['full_iq_known_param']['q3_repeat']['nll']:.4f}) | "
            f"eq_b8_s17 Q3_rep={cond_res['equivariant_b8_s17']['q3_repeat']['acc']:.4f} "
            f"Q2_val={cond_res['equivariant_b8_s17']['q2_by_arm']['valid']['acc']:.4f} "
            f"pass={cond_res['equivariant_b8_s17']['all_three_pass']}"
        )

    (out_dir / "direct_summary.json").write_text(json.dumps(all_summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "worst_examples.json").write_text(json.dumps(worst_dump, ensure_ascii=False, indent=2), encoding="utf-8")
    return all_summaries


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="s3_test")
    ap.add_argument("--max-blocks", type=int, default=None)
    ap.add_argument("--no-jsonl", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    run_all_direct(split=args.split, out_dir=args.out_dir, max_blocks=args.max_blocks, write_jsonl=not args.no_jsonl)
