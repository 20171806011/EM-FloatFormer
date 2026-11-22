"""Independent Statistical Audit for CLI-4 Stage 5 Closed-Book Multi-Cluster Study.

Implements:
  - Exact 2^G paired cluster sign-flip permutation test (G=8 -> 256 states, G=16 -> 65,536 states)
  - Holm-Bonferroni family-wise error rate adjustment across pre-registered contrast families
  - Honest verbatim tri-state Chinese verdicts (`支持` / `反证` / `证据不足`)
  - Single-process exact replay verification (`max_float_diff == 0.0`)
  - Detailed regime, pooled 4-ID, true-delay-reset, OOD, ray-count, and anchor-row tables.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from research.p12_stage5_fusion_neural.contract import (
    ID_REGIMES,
    OOD_AND_RESET_REGIMES,
    OOD_ONLY_REGIMES,
    TRUE_RESET_REGIME,
)


def exact_sign_flip_p(cluster_diffs: Sequence[float]) -> dict[str, Any]:
    """Exact 2^G two-sided & one-sided paired cluster sign-flip permutation test."""
    d = np.asarray(cluster_diffs, dtype=np.float64).ravel()
    g = len(d)
    if g < 2 or g > 20:
        raise ValueError(f"Expected 2 <= G <= 20 clusters for exact enumeration, got {g}")
    n_states = 1 << g
    bits = ((np.arange(n_states, dtype=np.int64)[:, None] >> np.arange(g, dtype=np.int64)[None, :]) & 1)
    signs = np.where(bits == 1, 1.0, -1.0)
    means = (signs @ d) / float(g)
    obs = float(np.mean(d))
    tol = 1e-12
    p_two = float(np.mean(np.abs(means) >= abs(obs) - tol))
    p_pos = float(np.mean(means >= obs - tol))
    sd = float(np.std(d, ddof=1))
    se = sd / math.sqrt(float(g))
    t_975 = float(student_t.ppf(0.975, df=g - 1))
    return {
        "n_clusters": int(g),
        "n_perm_states": int(n_states),
        "mean_pp": obs,
        "sd_pp": sd,
        "se_pp": se,
        "ci95_lo_pp": obs - t_975 * se,
        "ci95_hi_pp": obs + t_975 * se,
        "wins": int(np.sum(d > tol)),
        "ties": int(np.sum(np.abs(d) <= tol)),
        "losses": int(np.sum(d < -tol)),
        "p_two_sided": p_two,
        "p_one_sided_pos": p_pos,
        "cluster_diffs_pp": [float(x) for x in d],
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: float(p_values[i]))
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, float(p_values[idx]) * float(m - rank))
        running = max(running, val)
        adj[idx] = running
    return adj


def verdict_superiority(stat: Mapping[str, Any], p_holm: float, alpha: float = 0.05) -> str:
    """Verbatim tri-state verdict: positive `mean_pp` = candidate outperforms comparator."""
    if float(p_holm) < float(alpha):
        return "支持" if float(stat["mean_pp"]) > 0.0 else "反证"
    return "证据不足"


def verify_replay_records_exact(
    main_records: Sequence[Mapping[str, Any]],
    replay_records: Sequence[Mapping[str, Any]],
    *,
    ignore_keys: frozenset[str] = frozenset({"elapsed_ms", "fold_ms", "decode_ms", "neural_ms", "wall_ms"}),
) -> dict[str, Any]:
    """Verify exact `0.0` float/int equality between main and single-process replayed records."""
    by_key = {(r["split"], int(r["cluster"]), r["regime"], int(r["link"]), int(r["seed"])): r for r in main_records}
    max_diff = 0.0
    mismatches = 0
    checked = 0

    def _walk(a: Any, b: Any) -> None:
        nonlocal max_diff, mismatches
        if type(a) is not type(b):
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool):
                diff = abs(float(a) - float(b))
                if diff > max_diff:
                    max_diff = diff
                if diff > 0.0:
                    mismatches += 1
                return
            mismatches += 1
            return
        if isinstance(a, dict):
            for k in a:
                if k in ignore_keys:
                    continue
                if k not in b:
                    mismatches += 1
                    continue
                _walk(a[k], b[k])
        elif isinstance(a, list):
            if len(a) != len(b):
                mismatches += 1
                return
            for x, y in zip(a, b):
                _walk(x, y)
        elif isinstance(a, float):
            diff = abs(a - b)
            if diff > max_diff:
                max_diff = diff
            if diff > 0.0:
                mismatches += 1
        else:
            if a != b:
                mismatches += 1

    for rep in replay_records:
        key = (rep["split"], int(rep["cluster"]), rep["regime"], int(rep["link"]), int(rep["seed"]))
        if key not in by_key:
            mismatches += 1
            continue
        _walk(by_key[key], rep)
        checked += 1

    return {
        "checked_links": int(checked),
        "mismatches": int(mismatches),
        "max_float_diff": float(max_diff),
        "exact_pass": bool(checked == len(replay_records) and checked > 0 and mismatches == 0 and max_diff == 0.0),
    }


def _cluster_bler_series(
    records: Sequence[Mapping[str, Any]],
    clusters: Sequence[int],
    regimes: Sequence[str],
    arm_key: str,
) -> list[float]:
    reg_set = set(regimes)
    out: list[float] = []
    for c in clusters:
        recs = [r for r in records if int(r["cluster"]) == int(c) and r["regime"] in reg_set]
        e = sum(int(r["arms"][arm_key]["block_errors"]) for r in recs)
        b = sum(int(r["arms"][arm_key]["blocks"]) for r in recs)
        if b <= 0:
            raise ValueError(f"Zero blocks for cluster={c}, regimes={regimes}, arm={arm_key}")
        out.append(100.0 * float(e) / float(b))
    return out


def _cluster_metric_series(
    records: Sequence[Mapping[str, Any]],
    clusters: Sequence[int],
    regimes: Sequence[str],
    arm_key: str,
    field: str,
) -> list[float]:
    reg_set = set(regimes)
    out: list[float] = []
    for c in clusters:
        recs = [r for r in records if int(r["cluster"]) == int(c) and r["regime"] in reg_set]
        if not recs:
            raise ValueError(f"No records for cluster={c}, regimes={regimes}, arm={arm_key}")
        out.append(float(np.mean([float(r["arms"][arm_key][field]) for r in recs])))
    return out


def _run_holm_family(
    specs: Sequence[tuple[str, Sequence[Mapping[str, Any]], Sequence[str], str, str]],
    clusters: Sequence[int],
) -> dict[str, Any]:
    raw: list[tuple[str, dict[str, Any]]] = []
    for name, dataset, regs, ref_arm, cand_arm in specs:
        s_ref = _cluster_bler_series(dataset, clusters, regs, ref_arm)
        s_cand = _cluster_bler_series(dataset, clusters, regs, cand_arm)
        diffs = [a - b for a, b in zip(s_ref, s_cand)]  # positive = cand better
        st = exact_sign_flip_p(diffs)
        st["ref_arm"] = ref_arm
        st["cand_arm"] = cand_arm
        st["regimes"] = list(regs)
        raw.append((name, st))
    holms = holm_adjust([st["p_two_sided"] for _, st in raw])
    out: dict[str, Any] = {}
    for (name, st), ph in zip(raw, holms):
        st["p_holm"] = float(ph)
        st["verdict"] = verdict_superiority(st, ph)
        out[name] = st
    return out


def _agg_arm_table(
    recs: Sequence[Mapping[str, Any]],
    regimes: Sequence[str],
    arm_order: Sequence[str],
) -> dict[str, Any]:
    tab: dict[str, Any] = {}
    for reg in regimes:
        sub = [r for r in recs if r["regime"] == reg]
        if not sub:
            continue
        rmap: dict[str, Any] = {"n_links": len(sub)}
        for arm in arm_order:
            if arm not in sub[0]["arms"]:
                continue
            be = sum(int(r["arms"][arm]["block_errors"]) for r in sub)
            bb = sum(int(r["arms"][arm]["blocks"]) for r in sub)
            entry: dict[str, Any] = {
                "bler_pct": 100.0 * be / max(bb, 1),
                "block_errors": be,
                "blocks": bb,
                "mean_rel_frob": float(np.mean([r["arms"][arm]["mean_rel_frob_err"] for r in sub])),
                "probe_clean_nmse": float(np.mean([r["arms"][arm]["probe_clean_nmse"] for r in sub])),
            }
            sample = sub[0]["arms"][arm]
            if "bit_errors" in sample:
                bie = sum(int(r["arms"][arm]["bit_errors"]) for r in sub)
                bit = sum(int(r["arms"][arm]["info_bits"]) for r in sub)
                entry["ber_pct"] = 100.0 * bie / max(bit, 1)
                entry["mean_rel_mse_ch"] = float(np.mean([r["arms"][arm]["mean_rel_mse_ch"] for r in sub]))
                entry["probe_cal_ratio"] = float(np.mean([r["arms"][arm]["probe_cal_ratio"] for r in sub]))
                entry["probe_nll"] = float(np.mean([r["arms"][arm]["probe_nll_per_sample"] for r in sub]))
                entry["delay_reset_pct"] = 100.0 * float(np.mean([float(r["arms"][arm]["delay_reset_triggered"]) for r in sub]))
                ha = sum(int(r["arms"][arm]["helpers_accepted"]) for r in sub)
                fa = sum(int(r["arms"][arm]["false_assimilations"]) for r in sub)
                entry["helpers_accepted"] = ha
                entry["false_assimilations"] = fa
                entry["false_assim_pct"] = 100.0 * fa / max(ha, 1)
                entry["mean_branches"] = float(np.mean([r["arms"][arm]["mean_branches"] for r in sub]))
                entry["median_ms"] = float(np.median([r["arms"][arm]["elapsed_ms"] for r in sub]))
                entry["mean_ms"] = float(np.mean([r["arms"][arm]["elapsed_ms"] for r in sub]))
            if "shared_union_hard_errors" in sample:
                she = sum(int(r["arms"][arm]["shared_union_hard_errors"]) for r in sub)
                cov = sum(int(r["arms"][arm]["true_cw_in_union_count"]) for r in sub)
                entry["shared_union_hard_bler_pct"] = 100.0 * she / max(bb, 1)
                entry["true_cw_list_coverage_pct"] = 100.0 * cov / max(bb, 1)
            rmap[arm] = entry
        tab[reg] = rmap
    return tab


def audit_stage5_formal(
    id_records: Sequence[Mapping[str, Any]],
    ood_records: Sequence[Mapping[str, Any]],
    replay_records: Sequence[Mapping[str, Any]],
    *,
    clusters: Sequence[int],
    upgraded_phys_arm: str,
) -> dict[str, Any]:
    """Run complete independent statistical audit across Physics Fusion & Stage-5 Neural families."""
    arm_order = [
        "p12_consistent_base",
        "p12_hard_choice",
        "p12_mixture_llr",
        "p12_mixture_llr_content",
        "stage5_role_a_ens",
        "stage5_role_b_ens",
        "stage5_role_a_s5001",
        "stage5_role_a_s5002",
        "stage5_role_a_s5003",
        "stage5_role_b_s5001",
        "stage5_role_b_s5002",
        "stage5_role_b_s5003",
        "oracle_true_channel",
    ]
    present_arms = [a for a in arm_order if id_records and a in id_records[0]["arms"]]

    # Family F1: Consistent Posterior Fusion Physics Upgrades (Pre-registered!)
    f1_specs = [
        ("mix_vs_base_4id", id_records, ID_REGIMES, "p12_consistent_base", "p12_mixture_llr"),
        ("mix_vs_hard_4id", id_records, ID_REGIMES, "p12_hard_choice", "p12_mixture_llr"),
        ("content_vs_base_4id", id_records, ID_REGIMES, "p12_consistent_base", "p12_mixture_llr_content"),
        ("content_vs_mix_4id", id_records, ID_REGIMES, "p12_mixture_llr", "p12_mixture_llr_content"),
        ("upgraded_vs_base_new_delays", ood_records, (TRUE_RESET_REGIME,), "p12_consistent_base", upgraded_phys_arm),
        ("upgraded_vs_base_ood", ood_records, OOD_ONLY_REGIMES, "p12_consistent_base", upgraded_phys_arm),
    ]
    family_physics = _run_holm_family(f1_specs, clusters)

    # Family F2: Stage 5 Neural Refiners on Top of Upgraded Physics (Pre-registered!)
    f2_specs = [
        ("role_a_vs_phys_4id", id_records, ID_REGIMES, upgraded_phys_arm, "stage5_role_a_ens"),
        ("role_b_vs_phys_4id", id_records, ID_REGIMES, upgraded_phys_arm, "stage5_role_b_ens"),
        ("role_a_vs_role_b_4id", id_records, ID_REGIMES, "stage5_role_b_ens", "stage5_role_a_ens"),
        ("role_a_vs_phys_new_delays", ood_records, (TRUE_RESET_REGIME,), upgraded_phys_arm, "stage5_role_a_ens"),
        ("role_a_vs_phys_ood", ood_records, OOD_ONLY_REGIMES, upgraded_phys_arm, "stage5_role_a_ens"),
        ("role_a_vs_role_b_ood", ood_records, OOD_ONLY_REGIMES, "stage5_role_b_ens", "stage5_role_a_ens"),
    ]
    family_neural = _run_holm_family(f2_specs, clusters)

    # Auxiliary continuous channel-estimation (rel-MSE & probe-NMSE) paired cluster tests (informative!)
    ch_mse_contrasts: dict[str, Any] = {}
    for label, ref_a, cand_a, field in (
        ("role_a_vs_phys_4id_ch_mse", upgraded_phys_arm, "stage5_role_a_ens", "mean_rel_mse_ch"),
        ("role_b_vs_phys_4id_ch_mse", upgraded_phys_arm, "stage5_role_b_ens", "mean_rel_mse_ch"),
        ("role_a_vs_role_b_4id_ch_mse", "stage5_role_b_ens", "stage5_role_a_ens", "mean_rel_mse_ch"),
        ("role_a_vs_phys_4id_probe_nmse", upgraded_phys_arm, "stage5_role_a_ens", "probe_clean_nmse"),
        ("role_a_vs_role_b_4id_probe_nmse", "stage5_role_b_ens", "stage5_role_a_ens", "probe_clean_nmse"),
    ):
        s_ref = _cluster_metric_series(id_records, clusters, ID_REGIMES, ref_a, field)
        s_cand = _cluster_metric_series(id_records, clusters, ID_REGIMES, cand_a, field)
        diffs = [100.0 * (a - b) for a, b in zip(s_ref, s_cand)]  # positive = cand lower MSE/NMSE (in 10^-2 units)
        ch_mse_contrasts[label] = exact_sign_flip_p(diffs)

    # Pooled summaries
    def _pooled(recs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for arm in present_arms:
            be = sum(int(r["arms"][arm]["block_errors"]) for r in recs)
            bb = sum(int(r["arms"][arm]["blocks"]) for r in recs)
            entry: dict[str, Any] = {
                "bler_pct": 100.0 * be / max(bb, 1),
                "block_errors": be,
                "blocks": bb,
                "mean_rel_frob": float(np.mean([r["arms"][arm]["mean_rel_frob_err"] for r in recs])),
                "probe_clean_nmse": float(np.mean([r["arms"][arm]["probe_clean_nmse"] for r in recs])),
            }
            if "mean_rel_mse_ch" in recs[0]["arms"][arm]:
                bie = sum(int(r["arms"][arm]["bit_errors"]) for r in recs)
                bit = sum(int(r["arms"][arm]["info_bits"]) for r in recs)
                entry["ber_pct"] = 100.0 * bie / max(bit, 1)
                entry["mean_rel_mse_ch"] = float(np.mean([r["arms"][arm]["mean_rel_mse_ch"] for r in recs]))
                entry["probe_cal_ratio"] = float(np.mean([r["arms"][arm]["probe_cal_ratio"] for r in recs]))
                entry["probe_nll"] = float(np.mean([r["arms"][arm]["probe_nll_per_sample"] for r in recs]))
                entry["mean_ms"] = float(np.mean([r["arms"][arm]["elapsed_ms"] for r in recs]))
                entry["median_ms"] = float(np.median([r["arms"][arm]["elapsed_ms"] for r in recs]))
            out[arm] = entry
        return out

    # Breakdowns by ray count P and anchor row (ID pooled)
    main_arms_bd = [a for a in ("p12_consistent_base", "p12_hard_choice", "p12_mixture_llr", "p12_mixture_llr_content", "stage5_role_a_ens", "stage5_role_b_ens", "oracle_true_channel") if a in present_arms]
    by_ray_count: dict[str, Any] = {}
    for p_cnt in (2, 3, 4):
        sub = [r for r in id_records if int(r["n_rays"]) == p_cnt]
        if sub:
            by_ray_count[str(p_cnt)] = {
                arm: 100.0 * sum(int(r["arms"][arm]["block_errors"]) for r in sub) / sum(int(r["arms"][arm]["blocks"]) for r in sub)
                for arm in main_arms_bd
            }
    by_anchor_row: dict[str, Any] = {}
    for anc in (3, 6, 10, 13, 16):
        sub = [r for r in id_records if int(r["anchor_row"]) == anc]
        if sub:
            by_anchor_row[str(anc)] = {
                arm: 100.0 * sum(int(r["arms"][arm]["block_errors"]) for r in sub) / sum(int(r["arms"][arm]["blocks"]) for r in sub)
                for arm in main_arms_bd
            }

    replay_audit = verify_replay_records_exact(list(id_records) + list(ood_records), replay_records)

    return {
        "n_clusters": len(clusters),
        "clusters": list(clusters),
        "upgraded_phys_arm": upgraded_phys_arm,
        "pooled_4id": _pooled(id_records),
        "pooled_ood_only": _pooled([r for r in ood_records if r["regime"] in OOD_ONLY_REGIMES]),
        "id_by_regime": _agg_arm_table(id_records, ID_REGIMES, present_arms),
        "ood_by_regime": _agg_arm_table(ood_records, OOD_AND_RESET_REGIMES, present_arms),
        "id_by_ray_count_bler_pct": by_ray_count,
        "id_by_anchor_row_bler_pct": by_anchor_row,
        "holm_family_physics": family_physics,
        "holm_family_neural": family_neural,
        "aux_channel_mse_contrasts": ch_mse_contrasts,
        "replay_audit": replay_audit,
    }


__all__ = [
    "audit_stage5_formal",
    "exact_sign_flip_p",
    "holm_adjust",
    "verdict_superiority",
    "verify_replay_records_exact",
]
