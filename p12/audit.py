"""Independent statistical audit & exact 65,536-state cluster inference for Phase 12 (`p12/`).

Implements:
  - Exact 2^16 = 65,536 paired cluster sign-flip permutation tests with Holm-Bonferroni correction.
  - One-sided 95% Student-t (df=15) upper bounds for:
      * Engineering non-inferiority (`continuous` sparse vs dense < +1.00 pp)
      * Absolute `full_reset` BLER sub-10% certification (`U95 < 10.00%`)
  - Fixed-sequence gatekeeping across families (Primary Physics -> Secondary Physics).
  - Exact single-worker replay verification (`max_float_diff == 0.0`) on BOTH ID and OOD clusters.
  - Unvarnished verbatim Chinese verdicts (`支持` / `反证` / `证据不足`).
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import t as student_t

from p12.freeze import (
    FORMAL_CLUSTERS,
    OFFGRID_ID_REGIMES,
    OFFGRID_OOD_REGIMES,
    PRIMARY_ID_REGIMES,
    PRIMARY_OOD_REGIMES,
)

T_CRIT_95_DF15: float = float(student_t.ppf(0.95, df=15))  # 1.7530503556925547
NI_MARGIN_PP: float = 1.00
SUB10_THRESHOLD_PCT: float = 10.00


def exact_sign_flip_p(cluster_diffs: Sequence[float]) -> dict[str, Any]:
    """Exact 2^G two-sided & one-sided paired sign-flip permutation test (G=16 -> 65,536 states)."""
    d = np.asarray(cluster_diffs, dtype=np.float64).ravel()
    g = len(d)
    if g != 16:
        raise ValueError(f"Expected G=16 clusters, got {g}")
    n_states = 1 << g
    # Vectorized exact enumeration of all 65,536 sign patterns
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


def audit_noninferiority_one_sided(cluster_excess_pp: Sequence[float], margin_pp: float = NI_MARGIN_PP) -> dict[str, Any]:
    """One-sided 95% Student-t upper bound for `candidate - comparator` (negative = candidate better)."""
    d = np.asarray(cluster_excess_pp, dtype=np.float64).ravel()
    g = len(d)
    mean_d = float(np.mean(d))
    sd_d = float(np.std(d, ddof=1))
    se_d = sd_d / math.sqrt(float(g))
    u95 = mean_d + T_CRIT_95_DF15 * se_d
    l95 = mean_d - T_CRIT_95_DF15 * se_d
    if u95 < float(margin_pp):
        verdict = "支持"
    elif l95 >= float(margin_pp):
        verdict = "反证"
    else:
        verdict = "证据不足"
    return {
        "mean_excess_pp": mean_d,
        "sd_pp": sd_d,
        "se_pp": se_d,
        "u95_pp": u95,
        "l95_pp": l95,
        "margin_pp": float(margin_pp),
        "verdict": verdict,
        "cluster_excess_pp": [float(x) for x in d],
    }


def audit_absolute_upper_bound(cluster_bler_pct: Sequence[float], threshold_pct: float = SUB10_THRESHOLD_PCT) -> dict[str, Any]:
    """One-sided 95% Student-t upper bound verifying absolute BLER < threshold_pct."""
    v = np.asarray(cluster_bler_pct, dtype=np.float64).ravel()
    g = len(v)
    mean_v = float(np.mean(v))
    sd_v = float(np.std(v, ddof=1))
    se_v = sd_v / math.sqrt(float(g))
    u95 = mean_v + T_CRIT_95_DF15 * se_v
    l95 = mean_v - T_CRIT_95_DF15 * se_v
    if u95 < float(threshold_pct):
        verdict = "支持"
    elif l95 >= float(threshold_pct):
        verdict = "反证"
    else:
        verdict = "证据不足"
    return {
        "mean_pct": mean_v,
        "sd_pct": sd_v,
        "se_pct": se_v,
        "u95_pct": u95,
        "l95_pct": l95,
        "threshold_pct": float(threshold_pct),
        "verdict": verdict,
        "cluster_bler_pct": [float(x) for x in v],
    }


def _cluster_bler_series(
    records: Sequence[Mapping[str, Any]],
    regimes: Sequence[str],
    arm_key: str,
    err_field: str = "core_errors",
    blk_field: str = "core_blocks",
) -> list[float]:
    reg_set = set(regimes)
    out: list[float] = []
    for c in FORMAL_CLUSTERS:
        recs = [r for r in records if int(r["cluster"]) == c and r["regime"] in reg_set]
        e = sum(int(r["arms"][arm_key][err_field]) for r in recs)
        b = sum(int(r["arms"][arm_key][blk_field]) for r in recs)
        if b <= 0:
            raise ValueError(f"Zero blocks for cluster={c}, regimes={regimes}, arm={arm_key}")
        out.append(100.0 * float(e) / float(b))
    return out


def verify_replay_records_exact(
    main_records: Sequence[Mapping[str, Any]],
    replay_records: Sequence[Mapping[str, Any]],
    *,
    ignore_keys: frozenset[str] = frozenset({"elapsed_ms", "wall_ms", "decode_ms", "gate_ms"}),
) -> dict[str, Any]:
    """Verify exact 0.0 float/int equality between main and single-worker replayed records."""
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


def audit_p12_formal(
    primary_id: Sequence[Mapping[str, Any]],
    primary_ood: Sequence[Mapping[str, Any]],
    primary_replay: Sequence[Mapping[str, Any]],
    secondary_id: Sequence[Mapping[str, Any]],
    secondary_ood: Sequence[Mapping[str, Any]],
    secondary_replay: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run full Phase 12 fixed-sequence gatekeeping audit across Primary & Secondary formal studies."""
    # --- Family P: Primary Axis (Gate 1) ---
    p_contrasts_spec = [
        ("k3_vs_p11_full_reset", ("full_reset",), "p11_legacy", "p12_symm_3fold"),
        ("k3_vs_k2_full_reset", ("full_reset",), "p12_symm_2fold", "p12_symm_3fold"),
        ("k2_vs_p11_full_reset", ("full_reset",), "p11_legacy", "p12_symm_2fold"),
        ("k3_vs_p11_3id", PRIMARY_ID_REGIMES, "p11_legacy", "p12_symm_3fold"),
    ]
    p_raw: list[tuple[str, dict[str, Any]]] = []
    for name, regs, ref_arm, cand_arm in p_contrasts_spec:
        s_ref = _cluster_bler_series(primary_id, regs, ref_arm)
        s_cand = _cluster_bler_series(primary_id, regs, cand_arm)
        diffs = [a - b for a, b in zip(s_ref, s_cand)]  # positive = cand better
        p_raw.append((name, exact_sign_flip_p(diffs)))

    p_holms = holm_adjust([st["p_two_sided"] for _, st in p_raw])
    primary_family: dict[str, Any] = {}
    for (name, st), ph in zip(p_raw, p_holms):
        st["p_holm"] = float(ph)
        st["verdict"] = verdict_superiority(st, ph)
        primary_family[name] = st

    # Continuous engineering NI (k3 & k2 vs dense/always_clear)
    cont_dense = _cluster_bler_series(primary_id, ("continuous",), "dense_always_clear")
    cont_k3 = _cluster_bler_series(primary_id, ("continuous",), "p12_symm_3fold")
    cont_k2 = _cluster_bler_series(primary_id, ("continuous",), "p12_symm_2fold")
    primary_ni_k3 = audit_noninferiority_one_sided([k - d for k, d in zip(cont_k3, cont_dense)], margin_pp=NI_MARGIN_PP)
    primary_ni_k2 = audit_noninferiority_one_sided([k - d for k, d in zip(cont_k2, cont_dense)], margin_pp=NI_MARGIN_PP)

    # Sub-10% absolute full_reset BLER certification
    fr_k3 = _cluster_bler_series(primary_id, ("full_reset",), "p12_symm_3fold")
    primary_sub10 = audit_absolute_upper_bound(fr_k3, threshold_pct=SUB10_THRESHOLD_PCT)

    # Gate 1 pass condition: primary reset superiority AND continuous NI both supported
    gate1_passed = bool(
        primary_family["k3_vs_p11_full_reset"]["verdict"] == "支持"
        and primary_ni_k3["verdict"] == "支持"
    )

    # --- Family S: Secondary Axis (Gate 2, fixed-sequence gated on Gate 1) ---
    s_contrasts_spec = [
        ("p12_phys_vs_p11_transport_4id", secondary_id, OFFGRID_ID_REGIMES, "p11_transport_attn_ens", "p12_offgrid_physics"),
        ("p12_phys_vs_brefit_off_4id", secondary_id, OFFGRID_ID_REGIMES, "p12_brefit_off_control", "p12_offgrid_physics"),
        ("p12_phys_vs_p11_transport_midframe", secondary_id, ("offgrid_midframe_shift",), "p11_transport_attn_ens", "p12_offgrid_physics"),
        ("p12_phys_vs_p11_transport_new_delays", secondary_ood, ("offgrid_full_reset_new_delays",), "p11_transport_attn_ens", "p12_offgrid_physics"),
        ("p12_phys_vs_p11_transport_ood", secondary_ood, ("offgrid_ood_burst_and_shift", "offgrid_ood_five_ray_fast"), "p11_transport_attn_ens", "p12_offgrid_physics"),
    ]
    s_raw: list[tuple[str, dict[str, Any]]] = []
    for name, dataset, regs, ref_arm, cand_arm in s_contrasts_spec:
        s_ref = _cluster_bler_series(dataset, regs, ref_arm, err_field="block_errors", blk_field="blocks")
        s_cand = _cluster_bler_series(dataset, regs, cand_arm, err_field="block_errors", blk_field="blocks")
        diffs = [a - b for a, b in zip(s_ref, s_cand)]
        s_raw.append((name, exact_sign_flip_p(diffs)))

    s_holms = holm_adjust([st["p_two_sided"] for _, st in s_raw])
    secondary_family: dict[str, Any] = {}
    for (name, st), ph in zip(s_raw, s_holms):
        st["p_holm"] = float(ph)
        st["raw_verdict"] = verdict_superiority(st, ph)
        st["verdict"] = st["raw_verdict"] if gate1_passed else "证据不足"
        secondary_family[name] = st

    # Replay audits (both ID and OOD clusters!)
    p_rep_audit = verify_replay_records_exact(list(primary_id) + list(primary_ood), primary_replay)
    s_rep_audit = verify_replay_records_exact(list(secondary_id) + list(secondary_ood), secondary_replay)

    # Aggregate descriptive tables
    def _agg_primary_table(recs: Sequence[Mapping[str, Any]], regs: Sequence[str]) -> dict[str, Any]:
        tab: dict[str, Any] = {}
        for reg in regs:
            sub = [r for r in recs if r["regime"] == reg]
            rmap: dict[str, Any] = {"n_links": len(sub)}
            for arm in ("dense_always_clear", "p11_legacy", "p12_symm_2fold", "p12_symm_3fold", "oracle_true_csi"):
                if not sub or arm not in sub[0]["arms"]:
                    continue
                ce = sum(int(r["arms"][arm]["core_errors"]) for r in sub)
                cb = sum(int(r["arms"][arm]["core_blocks"]) for r in sub)
                entry: dict[str, Any] = {"core_bler_pct": 100.0 * ce / max(cb, 1), "core_errors": ce, "core_blocks": cb}
                if "extra_errors" in sub[0]["arms"][arm]:
                    ee = sum(int(r["arms"][arm]["extra_errors"]) for r in sub)
                    eb = sum(int(r["arms"][arm]["extra_blocks"]) for r in sub)
                    ha = sum(int(r["arms"][arm]["helpers_accepted"]) for r in sub)
                    fa = sum(int(r["arms"][arm]["false_assimilations"]) for r in sub)
                    entry["extra_heldout_bler_pct"] = 100.0 * ee / max(eb, 1)
                    entry["helpers_accepted"] = ha
                    entry["false_assimilations"] = fa
                    entry["false_assim_pct"] = 100.0 * fa / max(ha, 1)
                    entry["mean_probe_nmse"] = float(np.mean([r["arms"][arm]["probe_clean_nmse"] for r in sub]))
                    entry["median_ms"] = float(np.median([r["arms"][arm]["elapsed_ms"] for r in sub]))
                    # Reclaimed-row goodput (bits/frame = 64 * (core_ok + extra_ok) / n_links)
                    entry["net_goodput_bits_per_frame"] = 64.0 * float((cb - ce) + (eb - ee)) / float(len(sub))
                elif arm == "dense_always_clear":
                    entry["net_goodput_bits_per_frame"] = 64.0 * float(cb - ce) / float(len(sub))
                rmap[arm] = entry
            tab[reg] = rmap
        return tab

    def _agg_secondary_table(recs: Sequence[Mapping[str, Any]], regs: Sequence[str]) -> dict[str, Any]:
        tab: dict[str, Any] = {}
        for reg in regs:
            sub = [r for r in recs if r["regime"] == reg]
            rmap: dict[str, Any] = {"n_links": len(sub)}
            for arm in ("p11_classical_per_ray", "p11_transport_attn_ens", "p12_brefit_off_control", "p12_offgrid_physics", "oracle_true_channel"):
                if not sub or arm not in sub[0]["arms"]:
                    continue
                be = sum(int(r["arms"][arm]["block_errors"]) for r in sub)
                bb = sum(int(r["arms"][arm]["blocks"]) for r in sub)
                entry: dict[str, Any] = {
                    "bler_pct": 100.0 * be / max(bb, 1),
                    "block_errors": be,
                    "blocks": bb,
                    "mean_rel_frob": float(np.mean([r["arms"][arm]["mean_rel_frob_err"] for r in sub])),
                    "mean_probe_nmse": float(np.mean([r["arms"][arm]["probe_clean_nmse"] for r in sub])),
                }
                if "delay_reset_triggered" in sub[0]["arms"][arm]:
                    entry["delay_reset_pct"] = 100.0 * float(np.mean([float(r["arms"][arm]["delay_reset_triggered"]) for r in sub]))
                    ha = sum(int(r["arms"][arm]["helpers_accepted"]) for r in sub)
                    fa = sum(int(r["arms"][arm]["false_assimilations"]) for r in sub)
                    entry["helpers_accepted"] = ha
                    entry["false_assimilations"] = fa
                    entry["false_assim_pct"] = 100.0 * fa / max(ha, 1)
                    entry["median_ms"] = float(np.median([r["arms"][arm]["elapsed_ms"] for r in sub]))
                rmap[arm] = entry
            tab[reg] = rmap
        return tab

    # Secondary breakdowns by ray count P and anchor row (ID pooled)
    by_ray_count: dict[str, Any] = {}
    for p_cnt in (2, 3, 4):
        sub = [r for r in secondary_id if int(r["n_rays"]) == p_cnt]
        if sub:
            by_ray_count[str(p_cnt)] = {
                arm: 100.0 * sum(int(r["arms"][arm]["block_errors"]) for r in sub) / sum(int(r["arms"][arm]["blocks"]) for r in sub)
                for arm in ("p11_transport_attn_ens", "p12_offgrid_physics", "oracle_true_channel")
            }
    by_anchor_row: dict[str, Any] = {}
    for anc in (3, 6, 10, 13, 16):
        sub = [r for r in secondary_id if int(r["anchor_row"]) == anc]
        if sub:
            by_anchor_row[str(anc)] = {
                arm: 100.0 * sum(int(r["arms"][arm]["block_errors"]) for r in sub) / sum(int(r["arms"][arm]["blocks"]) for r in sub)
                for arm in ("p11_transport_attn_ens", "p12_offgrid_physics", "oracle_true_channel")
            }

    # Pooled summaries
    p_3id_pooled = {
        arm: 100.0 * sum(int(r["arms"][arm]["core_errors"]) for r in primary_id) / sum(int(r["arms"][arm]["core_blocks"]) for r in primary_id)
        for arm in ("dense_always_clear", "p11_legacy", "p12_symm_2fold", "p12_symm_3fold", "oracle_true_csi")
    }
    s_4id_pooled = {
        arm: 100.0 * sum(int(r["arms"][arm]["block_errors"]) for r in secondary_id) / sum(int(r["arms"][arm]["blocks"]) for r in secondary_id)
        for arm in ("p11_classical_per_ray", "p11_transport_attn_ens", "p12_brefit_off_control", "p12_offgrid_physics", "oracle_true_channel")
    }

    return {
        "fixed_sequence_gate1_passed": gate1_passed,
        "primary": {
            "pooled_3id_bler_pct": p_3id_pooled,
            "id_by_regime": _agg_primary_table(primary_id, PRIMARY_ID_REGIMES),
            "ood_by_regime": _agg_primary_table(primary_ood, PRIMARY_OOD_REGIMES),
            "holm_family": primary_family,
            "ni_k3_continuous": primary_ni_k3,
            "ni_k2_continuous": primary_ni_k2,
            "sub10_k3_full_reset": primary_sub10,
            "replay_audit": p_rep_audit,
        },
        "secondary": {
            "pooled_4id_bler_pct": s_4id_pooled,
            "id_by_regime": _agg_secondary_table(secondary_id, OFFGRID_ID_REGIMES),
            "ood_by_regime": _agg_secondary_table(secondary_ood, OFFGRID_OOD_REGIMES),
            "id_by_ray_count_bler_pct": by_ray_count,
            "id_by_anchor_row_bler_pct": by_anchor_row,
            "holm_family": secondary_family,
            "replay_audit": s_rep_audit,
        },
    }


__all__ = [
    "NI_MARGIN_PP",
    "SUB10_THRESHOLD_PCT",
    "T_CRIT_95_DF15",
    "audit_absolute_upper_bound",
    "audit_noninferiority_one_sided",
    "audit_p12_formal",
    "exact_sign_flip_p",
    "holm_adjust",
    "verdict_superiority",
    "verify_replay_records_exact",
]
