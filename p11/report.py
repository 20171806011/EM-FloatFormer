"""Render Phase 11 formal research report (P11_REPORT.md) and CLI sync summary (CLI_SYNC_SUMMARY.md)."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from p11.audit import P10_9E9D93B_REFERENCE, VERDICT_ZH
from p11.contract import (
    OFFGRID_REGIMES_ID,
    OFFGRID_REGIMES_OOD,
    P10_BASELINE_COMMIT,
    REGIMES_ID,
)
from p11.freeze import (
    DEFAULT_MANIFEST_PATH,
    OFFGRID_SEEDS,
    OFFGRID_VARIANTS,
    RESULTS_DIR,
    ROOT,
    verify_freeze,
)


def _is_strict_int(val: Any) -> bool:
    return isinstance(val, int) and not isinstance(val, bool)


def _is_finite_real(val: Any) -> bool:
    return isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val))


def _reject_nonstandard_json_constant(const: str) -> Any:
    raise ValueError(f"Non-standard JSON constant {const!r} not allowed")


def _read_json_dict(path: Path) -> dict[str, Any]:
    obj = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_nonstandard_json_constant,
    )
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}, got {type(obj).__name__}")
    return obj


def _verdict_zh_from_row(row: Mapping[str, Any]) -> str:
    if "verdict_zh" in row:
        vz = row["verdict_zh"]
        if isinstance(vz, str) and vz in ("支持", "反证", "证据不足"):
            return vz
        raise ValueError(f"Invalid verdict_zh {vz!r} in row")
    if "verdict" in row:
        v = row["verdict"]
        if isinstance(v, str) and v in VERDICT_ZH:
            return VERDICT_ZH[v]
        raise ValueError(f"Invalid verdict {v!r} in row")
    raise ValueError("Missing required 'verdict_zh' / 'verdict' in row")


def _validate_audit_payload(audit_payload: Mapping[str, Any]) -> None:
    if not isinstance(audit_payload, Mapping):
        raise ValueError(f"audit_payload must be a mapping, got {type(audit_payload).__name__}")
    if "mode" in audit_payload:
        mode = audit_payload["mode"]
        if not isinstance(mode, str) or not mode.strip():
            raise ValueError(f"audit_payload['mode'] must be a non-empty string, got {mode!r}")
    if "seed_disjointness" in audit_payload and audit_payload["seed_disjointness"] is not None:
        if not isinstance(audit_payload["seed_disjointness"], Mapping):
            raise ValueError("audit_payload['seed_disjointness'] must be a mapping or None")
    for ax in ("primary", "secondary"):
        if ax in audit_payload:
            ax_obj = audit_payload[ax]
            if not isinstance(ax_obj, Mapping):
                raise ValueError(f"audit_payload[{ax!r}] must be a mapping")
            if "audit" in ax_obj:
                aud = ax_obj["audit"]
                if not isinstance(aud, Mapping):
                    raise ValueError(f"audit_payload[{ax!r}]['audit'] must be a mapping")
                if "clusters" in aud:
                    c = aud["clusters"]
                    if not _is_strict_int(c) or c < 1:
                        raise ValueError(f"audit_payload[{ax!r}]['audit']['clusters'] must be int >= 1, got {c!r}")
                for diff_k in ("max_aggregate_diff", "max_stat_diff"):
                    if diff_k in aud and aud[diff_k] is not None and not _is_finite_real(aud[diff_k]):
                        raise ValueError(
                            f"audit_payload[{ax!r}]['audit'][{diff_k!r}] must be a finite real number, got {aud[diff_k]!r}"
                        )


def _validate_tests_list(tests: Any, label: str) -> None:
    if not isinstance(tests, Sequence) or isinstance(tests, (str, bytes)) or len(tests) < 1:
        raise ValueError(f"{label} must be a non-empty list of contrast dicts")
    for idx, row in enumerate(tests):
        if not isinstance(row, Mapping):
            raise ValueError(f"{label}[{idx}] must be a mapping, got {type(row).__name__}")
        name = row.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label}[{idx}]['name'] must be a non-empty string")
        for num_k in ("mean_diff", "holm_p", "raw_p"):
            if not _is_finite_real(row.get(num_k)):
                raise ValueError(f"{label}[{idx}][{num_k!r}] must be a finite real number, got {row.get(num_k)!r}")
        for int_k in ("wins", "ties", "losses"):
            v = row.get(int_k)
            if not _is_strict_int(v) or v < 0:
                raise ValueError(f"{label}[{idx}][{int_k!r}] must be non-negative int, got {v!r}")
        for ex_k in ("holm_exact", "raw_exact"):
            ev = row.get(ex_k)
            if not isinstance(ev, str) or not ev:
                raise ValueError(f"{label}[{idx}][{ex_k!r}] must be a non-empty string, got {ev!r}")
        _verdict_zh_from_row(row)


def _validate_clusters_and_g(
    summary: Mapping[str, Any],
    axis: str,
    audit_payload: Optional[Mapping[str, Any]],
    *,
    require_clusters: bool,
) -> None:
    if "g" in summary:
        g_val = summary["g"]
        if not _is_strict_int(g_val) or g_val < 1:
            raise ValueError(f"{axis}.g must be integer >= 1, got {g_val!r}")
        if "clusters" not in summary:
            raise ValueError(f"{axis} specifies g={g_val} but is missing 'clusters'")
    if "clusters" in summary or require_clusters:
        if "clusters" not in summary:
            raise ValueError(f"{axis} summary missing required 'clusters'")
        clusters = summary["clusters"]
        if not isinstance(clusters, Sequence) or isinstance(clusters, (str, bytes)) or len(clusters) < 1:
            raise ValueError(f"{axis}.clusters must be a non-empty list")
        for idx, cl in enumerate(clusters):
            if not isinstance(cl, Mapping) or len(cl) == 0:
                raise ValueError(f"{axis}.clusters[{idx}] must be a non-empty mapping")
        if "g" in summary and len(clusters) != int(summary["g"]):
            raise ValueError(f"{axis} len(clusters)={len(clusters)} != g={summary['g']}")
        if audit_payload is not None:
            if audit_payload.get("mode") in ("primary", "secondary", "all") and len(clusters) != 16:
                raise ValueError(f"{axis} formal mode requires G=16 clusters, got {len(clusters)}")
            ax_obj = audit_payload.get(axis)
            ax_audit = ax_obj.get("audit") if isinstance(ax_obj, Mapping) else None
            if isinstance(ax_audit, Mapping) and "clusters" in ax_audit:
                if len(clusters) != int(ax_audit["clusters"]):
                    raise ValueError(
                        f"{axis} len(clusters)={len(clusters)} != audit clusters={ax_audit['clusters']}"
                    )


def _validate_arm_table(table: Any, label: str) -> None:
    if not isinstance(table, Mapping) or len(table) == 0:
        raise ValueError(f"{label} must be a non-empty mapping")
    for arm, stats in table.items():
        if not isinstance(stats, Mapping) or not _is_finite_real(stats.get("bler")):
            raise ValueError(f"{label}[{arm!r}] must be a mapping with finite real 'bler'")
        for opt_k in ("clean_nmse", "calibration", "net_delivered_info_bits_per_frame", "native_bler"):
            if opt_k in stats and stats[opt_k] is not None and not _is_finite_real(stats[opt_k]):
                raise ValueError(f"{label}[{arm!r}][{opt_k!r}] must be a finite real number, got {stats[opt_k]!r}")


def _validate_primary_summary(
    primary_summary: Mapping[str, Any],
    audit_payload: Optional[Mapping[str, Any]],
    *,
    require_disk_fields: bool,
) -> None:
    if not isinstance(primary_summary, Mapping):
        raise ValueError(f"primary_summary must be a mapping, got {type(primary_summary).__name__}")
    _validate_clusters_and_g(primary_summary, "primary", audit_payload, require_clusters=require_disk_fields)

    if "totals" not in primary_summary or not isinstance(primary_summary["totals"], Mapping):
        raise ValueError("primary_summary missing required 'totals' mapping")
    totals = primary_summary["totals"]
    if require_disk_fields:
        if len(totals) == 0 or "sparse" not in totals or totals["sparse"] is None:
            raise ValueError("primary_summary['totals'] must contain non-empty 'sparse' arm metrics")
        _validate_arm_table(totals["sparse"], "primary.totals.sparse")
    else:
        for b_key, b_tbl in totals.items():
            if b_tbl is not None:
                _validate_arm_table(b_tbl, f"primary.totals.{b_key}")

    if "by_regime" not in primary_summary or not isinstance(primary_summary["by_regime"], Mapping):
        raise ValueError("primary_summary missing required 'by_regime' mapping")
    if require_disk_fields and len(primary_summary["by_regime"]) == 0:
        raise ValueError("primary_summary['by_regime'] must be non-empty")
    for reg, reg_map in primary_summary["by_regime"].items():
        if reg_map is None:
            continue
        if not isinstance(reg_map, Mapping):
            raise ValueError(f"primary.by_regime[{reg!r}] must be a mapping")
        for b_key, b_tbl in reg_map.items():
            if b_tbl is not None:
                _validate_arm_table(b_tbl, f"primary.by_regime.{reg}.{b_key}")

    _validate_tests_list(primary_summary.get("tests"), "primary.tests")

    ni_full = primary_summary.get("noninferiority_full")
    if ni_full is not None:
        if not isinstance(ni_full, Mapping):
            raise ValueError("primary.noninferiority_full must be a mapping")
        _verdict_zh_from_row(ni_full)
        bler_ni = ni_full.get("bler_noninferiority")
        if not isinstance(bler_ni, Mapping):
            raise ValueError("primary.noninferiority_full missing 'bler_noninferiority' mapping")
        if not _is_finite_real(bler_ni.get("mean_diff")) or not _is_finite_real(bler_ni.get("upper_95")):
            raise ValueError("primary.noninferiority_full.bler_noninferiority missing finite mean_diff/upper_95")
        if ("lower_95" in bler_ni or require_disk_fields) and not _is_finite_real(bler_ni.get("lower_95")):
            raise ValueError("primary.noninferiority_full.bler_noninferiority missing finite lower_95")
    else:
        ni_row = primary_summary.get("noninferiority")
        if not isinstance(ni_row, Mapping) or len(ni_row) == 0:
            raise ValueError("primary_summary missing required 'noninferiority' mapping")
        _verdict_zh_from_row(ni_row)
        if not _is_finite_real(ni_row.get("mean_diff")) or not _is_finite_real(ni_row.get("upper_95")):
            raise ValueError("primary.noninferiority missing finite mean_diff/upper_95")
        if ("lower_95" in ni_row or require_disk_fields) and not _is_finite_real(ni_row.get("lower_95")):
            raise ValueError("primary.noninferiority missing finite lower_95")

    cpb = primary_summary.get("common_phase_blindspot")
    if cpb is not None:
        if not isinstance(cpb, Mapping):
            raise ValueError("primary.common_phase_blindspot must be a mapping")

        def _validate_cpb_subset(sub: Any, label: str) -> None:
            if sub is None:
                return
            if not isinstance(sub, Mapping):
                raise ValueError(f"{label} must be a mapping or None, got {type(sub).__name__}")
            for float_k in ("target_bler", "p10_9e9d93b_bler", "delta_vs_p10_9e9d93b_pp", "false_assimilation_rate"):
                if float_k in sub and sub[float_k] is not None and not _is_finite_real(sub[float_k]):
                    raise ValueError(f"{label}[{float_k!r}] must be a finite real number, got {sub[float_k]!r}")
            for int_k in ("false_assimilations", "total_assimilations"):
                if int_k in sub and sub[int_k] is not None:
                    iv = sub[int_k]
                    if not _is_strict_int(iv) or iv < 0:
                        raise ValueError(f"{label}[{int_k!r}] must be int >= 0, got {iv!r}")

        if "pooled_3id" in cpb:
            _validate_cpb_subset(cpb["pooled_3id"], "primary.common_phase_blindspot.pooled_3id")
        if "by_regime" in cpb and cpb["by_regime"] is not None:
            if not isinstance(cpb["by_regime"], Mapping):
                raise ValueError("primary.common_phase_blindspot.by_regime must be a mapping or None")
            for reg, reg_sub in cpb["by_regime"].items():
                _validate_cpb_subset(reg_sub, f"primary.common_phase_blindspot.by_regime.{reg}")


def _validate_secondary_summary(
    secondary_summary: Mapping[str, Any],
    audit_payload: Optional[Mapping[str, Any]],
    *,
    require_disk_fields: bool,
) -> None:
    if not isinstance(secondary_summary, Mapping):
        raise ValueError(f"secondary_summary must be a mapping, got {type(secondary_summary).__name__}")
    _validate_clusters_and_g(secondary_summary, "secondary", audit_payload, require_clusters=require_disk_fields)

    for eq_key in ("id_equal", "ood_equal"):
        eq_obj = secondary_summary.get(eq_key)
        if not isinstance(eq_obj, Mapping) or "sparse" not in eq_obj:
            raise ValueError(f"secondary_summary missing required '{eq_key}.sparse' mapping")
        _validate_arm_table(eq_obj["sparse"], f"secondary.{eq_key}.sparse")

    if require_disk_fields:
        if not isinstance(secondary_summary.get("totals"), Mapping) or len(secondary_summary["totals"]) == 0:
            raise ValueError("secondary_summary['totals'] must be a non-empty mapping")
        if not isinstance(secondary_summary.get("by_regime"), Mapping) or len(secondary_summary["by_regime"]) == 0:
            raise ValueError("secondary_summary['by_regime'] must be a non-empty mapping")

    _validate_tests_list(secondary_summary.get("tests"), "secondary.tests")
    if "ood_tests" in secondary_summary and secondary_summary["ood_tests"] is not None:
        _validate_tests_list(secondary_summary["ood_tests"], "secondary.ood_tests")


def _fmt_pct(val: Optional[float]) -> str:
    if val is None:
        return "N/A"
    if not _is_finite_real(val):
        raise ValueError(f"Percentage metric must be finite real, got {val!r}")
    return f"{100.0 * float(val):.2f}%"


def _fmt_real(val: Any, fmt: str, default: float = 0.0) -> str:
    v = default if val is None else val
    if not _is_finite_real(v):
        raise ValueError(f"Numeric metric must be finite real, got {val!r}")
    return format(float(v), fmt)


def _render_reports_validated(
    *,
    primary_summary: Optional[Mapping[str, Any]] = None,
    secondary_summary: Optional[Mapping[str, Any]] = None,
    audit_payload: Optional[Mapping[str, Any]] = None,
    freeze_sha: Optional[str] = None,
    allow_partial: bool = False,
    require_disk_fields: bool = False,
) -> tuple[str, str]:
    if primary_summary is None and secondary_summary is None:
        raise ValueError("At least one of primary_summary or secondary_summary must be provided.")

    if audit_payload is not None:
        _validate_audit_payload(audit_payload)
        if not allow_partial:
            mode = audit_payload.get("mode")
            axis = audit_payload.get("axis")
            need_pri = (
                ("primary" in audit_payload)
                or (mode in ("primary", "all"))
                or (mode == "dry_run" and axis in ("primary", "both"))
            )
            need_sec = (
                ("secondary" in audit_payload)
                or (mode in ("secondary", "all"))
                or (mode == "dry_run" and axis in ("secondary", "both"))
            )
            if need_pri and primary_summary is None:
                raise ValueError("Audit payload declares primary axis results, but primary_summary is missing.")
            if need_sec and secondary_summary is None:
                raise ValueError("Audit payload declares secondary axis results, but secondary_summary is missing.")

    if primary_summary is not None:
        _validate_primary_summary(primary_summary, audit_payload, require_disk_fields=require_disk_fields)
    if secondary_summary is not None:
        _validate_secondary_summary(secondary_summary, audit_payload, require_disk_fields=require_disk_fields)
    """Render `(P11_REPORT.md, CLI_SYNC_SUMMARY.md)` with verbatim Chinese verdicts."""
    freeze_label = f"`{freeze_sha}`" if freeze_sha else "`unfrozen/dry_run`"

    lines: list[str] = [
        "# EM-FloatFormer P11 研究报告：载荷协方差重构、波前同化与离格连续流形泛化",
        "",
        f"- **基线提交**：P10 [`{P10_BASELINE_COMMIT[:7]}`](https://github.com/20171806011/EM-FloatFormer/tree/{P10_BASELINE_COMMIT})",
        f"- **事前冻结哈希**：{freeze_label}",
    ]
    if audit_payload:
        lines.append(
            f"- **运行模式**：`{audit_payload.get('mode', 'unknown')}`（`axis={audit_payload.get('axis', 'both')}`）"
        )
    lines += [
        "",
        "## 核心判决速览（严格逐字输出 `支持` / `反证` / `证据不足`）",
        "",
        "| 轴向与预注册检验项 | 核心量化对比 | 精确 Holm p / 95% 上界 | 判定 |",
        "| :--- | :--- | :---: | :---: |",
    ]

    verdict_records: list[dict[str, str]] = []

    if primary_summary is not None:
        p_tot = (primary_summary.get("totals") or {}).get("sparse") or {}
        p_reg = primary_summary.get("by_regime") or {}
        for t_row in primary_summary.get("tests", []):
            vz = _verdict_zh_from_row(t_row)
            verdict_records.append({"axis": "Primary", "name": str(t_row["name"]), "verdict_zh": vz})
            lines.append(
                f"| **Primary / `{t_row['name']}`** | 均值差 `{100.0 * float(t_row['mean_diff']):+.2f} pp` (`{t_row['wins']}/{t_row['ties']}/{t_row['losses']}`) | `p_holm={float(t_row['holm_p']):.6f}` (`{t_row['holm_exact']}`) | **{vz}** |"
            )
        ni_full = primary_summary.get("noninferiority_full")
        ni_row = primary_summary.get("noninferiority") or {}
        if ni_full is not None:
            vz_ni = _verdict_zh_from_row(ni_full)
            u95 = float(ni_full["bler_noninferiority"]["upper_95"])
            mdiff = float(ni_full["bler_noninferiority"]["mean_diff"])
        else:
            vz_ni = _verdict_zh_from_row(ni_row)
            u95 = float(ni_row["upper_95"])
            mdiff = float(ni_row["mean_diff"])
        verdict_records.append({"axis": "Primary", "name": "engineering_noninferiority", "verdict_zh": vz_ni})
        cont_reg = p_reg.get("continuous") or {}
        c_cont_sp = (cont_reg.get("sparse") or {}).get("p11_chained_wavefront") or {}
        c_cont_dn = (cont_reg.get("dense") or {}).get("always_clear") or {}
        lines.append(
            f"| **Primary / 工程非劣性 (`< +1.00 pp`)** | `sparse/p11_chained` `{_fmt_pct(c_cont_sp.get('bler'))}` vs `dense/clear` `{_fmt_pct(c_cont_dn.get('bler'))}` (`{100.0 * mdiff:+.2f} pp`) | 95% 上界 `{100.0 * u95:+.2f} pp` | **{vz_ni}** |"
        )
        if "p11_chained_wavefront" in p_tot:
            lines.append("")

    if secondary_summary is not None:
        for t_row in secondary_summary.get("tests", []):
            vz = _verdict_zh_from_row(t_row)
            verdict_records.append({"axis": "Secondary", "name": str(t_row["name"]), "verdict_zh": vz})
            lines.append(
                f"| **Secondary / `{t_row['name']}`** | 均值差 `{100.0 * float(t_row['mean_diff']):+.2f} pp` (`{t_row['wins']}/{t_row['ties']}/{t_row['losses']}`) | `p_holm={float(t_row['holm_p']):.6f}` (`{t_row['holm_exact']}`) | **{vz}** |"
            )

    if not verdict_records:
        lines.append("| （暂无检验记录） | — | — | **证据不足** |")

    # Primary Axis Detailed Section
    if primary_summary is not None:
        lines += [
            "",
            "---",
            "",
            "## 一、Primary Axis：载荷协方差重构 + 奇偶认证波前同化",
            "",
            "| 方法（9 dB Sparse / Dense） | 连续链路 | 部分变化 | 完全重置 | **3-ID 等权 BLER** | 帧末无噪 NMSE | 校准比 | 净交付 (bit/帧) |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        p_tot = primary_summary.get("totals") or {}
        p_tot_sp = p_tot.get("sparse") or {}
        p_tot_dn = p_tot.get("dense") or {}
        p_reg = primary_summary.get("by_regime") or {}
        for arm in sorted(p_tot_sp.keys()):
            c_st = ((p_reg.get("continuous") or {}).get("sparse") or {}).get(arm) or {}
            p_st = ((p_reg.get("partial_shift") or {}).get("sparse") or {}).get(arm) or {}
            f_st = ((p_reg.get("full_reset") or {}).get("sparse") or {}).get(arm) or {}
            eq_st = p_tot_sp[arm]
            lines.append(
                f"| `sparse/{arm}` | {_fmt_pct(c_st.get('bler'))} | {_fmt_pct(p_st.get('bler'))} | {_fmt_pct(f_st.get('bler'))} | "
                f"**{_fmt_pct(eq_st.get('bler'))}** | {_fmt_real(eq_st.get('clean_nmse'), '.4f')} | {_fmt_real(eq_st.get('calibration'), '.2f')}x | {_fmt_real(eq_st.get('net_delivered_info_bits_per_frame'), '.1f')} |"
            )
        for arm in sorted(p_tot_dn.keys()):
            c_st = ((p_reg.get("continuous") or {}).get("dense") or {}).get(arm) or {}
            p_st = ((p_reg.get("partial_shift") or {}).get("dense") or {}).get(arm) or {}
            f_st = ((p_reg.get("full_reset") or {}).get("dense") or {}).get(arm) or {}
            eq_st = p_tot_dn[arm]
            lines.append(
                f"| `dense/{arm}` | {_fmt_pct(c_st.get('bler'))} | {_fmt_pct(p_st.get('bler'))} | {_fmt_pct(f_st.get('bler'))} | "
                f"**{_fmt_pct(eq_st.get('bler'))}** | {_fmt_real(eq_st.get('clean_nmse'), '.4f')} | {_fmt_real(eq_st.get('calibration'), '.2f')}x | {_fmt_real(eq_st.get('net_delivered_info_bits_per_frame'), '.1f')} |"
            )

        cpb = primary_summary.get("common_phase_blindspot")
        if cpb:
            pooled = cpb.get("pooled_3id") or {}
            by_reg_cpb = cpb.get("by_regime") or {}
            p_fa = 0 if pooled.get("false_assimilations") is None else pooled.get("false_assimilations")
            p_ta = 0 if pooled.get("total_assimilations") is None else pooled.get("total_assimilations")
            p_far = 0.0 if pooled.get("false_assimilation_rate") is None else pooled.get("false_assimilation_rate")
            lines += [
                "",
                "### 1.1 共模相位盲区突破与 Wave-1 条件 BLER / 注入词误同化（对照 `9e9d93b`）",
                "",
                f"- **3-ID 等权 BLER**：`{_fmt_pct(pooled.get('target_bler'))}` vs `9e9d93b` 参考 `{_fmt_pct(P10_9E9D93B_REFERENCE['3id_code_gate_bler'])}`（差值 `{_fmt_real(pooled.get('delta_vs_p10_9e9d93b_pp'), '+.2f', default=0.0)} pp`）",
                f"- **真实注入词误同化率（False Assimilation）**：`{p_fa}/{p_ta}`（`{_fmt_pct(p_far)}`）",
                "",
                "| 机制 | P11 BLER | `9e9d93b` 参考 | 相对改善 | 误同化率 (注入词) |",
                "| :--- | ---: | ---: | ---: | ---: |",
            ]
            for reg in REGIMES_ID:
                rs = by_reg_cpb.get(reg) or {}
                r_fa = 0 if rs.get("false_assimilations") is None else rs.get("false_assimilations")
                r_ta = 0 if rs.get("total_assimilations") is None else rs.get("total_assimilations")
                r_far = 0.0 if rs.get("false_assimilation_rate") is None else rs.get("false_assimilation_rate")
                lines.append(
                    f"| `{reg}` | {_fmt_pct(rs.get('target_bler'))} | {_fmt_pct(rs.get('p10_9e9d93b_bler'))} | "
                    f"{_fmt_real(rs.get('delta_vs_p10_9e9d93b_pp'), '+.2f', default=0.0)} pp | "
                    f"{r_fa}/{r_ta} ({_fmt_pct(r_far)}) |"
                )

    # Secondary Axis Detailed Section
    if secondary_summary is not None:
        lines += [
            "",
            "---",
            "",
            "## 二、Secondary Axis：离格连续时延流形泛化评估（4 ID + 2 OOD 机制，18 块/链路）",
            "",
            "| 方法（9 dB 离格） | 4-ID 等权 BLER (Strong-L / Native) | 2-OOD 等权 BLER (Strong-L / Native) | 帧末无噪 NMSE | 校准比 |",
            "| :--- | :---: | :---: | ---: | ---: |",
        ]
        s_id = (secondary_summary.get("id_equal") or {}).get("sparse") or {}
        s_ood = (secondary_summary.get("ood_equal") or {}).get("sparse") or {}
        ordered_arms = (
            "classical_always_inherit",
            "classical_always_clear",
            "classical_link_gate",
            "classical_per_ray_gate",
            *OFFGRID_VARIANTS,
            "oracle_true_channel",
        )
        for arm in ordered_arms:
            if arm not in s_id:
                continue
            i_st = s_id[arm]
            o_st = s_ood.get(arm, {})
            lines.append(
                f"| `{arm}` | {_fmt_pct(i_st.get('bler'))} / {_fmt_pct(i_st.get('native_bler'))} | "
                f"{_fmt_pct(o_st.get('bler'))} / {_fmt_pct(o_st.get('native_bler'))} | "
                f"{_fmt_real(i_st.get('clean_nmse'), '.4f')} | {_fmt_real(i_st.get('calibration'), '.2f')}x |"
            )

        lines += [
            "",
            "### 2.1 逐种子披露（绝不将种子与链路相乘虚增样本量）",
            "",
            "| 变体 / 种子 | 4-ID Strong-L BLER | 4-ID Native BLER | 2-OOD Strong-L BLER |",
            "| :--- | ---: | ---: | ---: |",
        ]
        for var in OFFGRID_VARIANTS:
            for sd in OFFGRID_SEEDS:
                k = f"{var}_s{sd}"
                if k in s_id:
                    lines.append(
                        f"| `{k}` | {_fmt_pct(s_id[k].get('bler'))} | {_fmt_pct(s_id[k].get('native_bler'))} | {_fmt_pct(s_ood.get(k, {}).get('bler'))} |"
                    )

        lines += [
            "",
            "### 2.2 离格 Holm-Bonferroni 检验明细",
            "",
            "| 对比 | 均值差 (`>0` 支持) | 胜/平/负 | 原始精确 p | Holm 精确 p | 判定 |",
            "| :--- | ---: | :---: | ---: | ---: | :---: |",
        ]
        for t_row in secondary_summary.get("tests", []):
            vz = _verdict_zh_from_row(t_row)
            lines.append(
                f"| `{t_row['name']}` | {100.0 * float(t_row['mean_diff']):+.2f} pp | {t_row['wins']}/{t_row['ties']}/{t_row['losses']} | "
                f"{t_row['raw_exact']} ({float(t_row['raw_p']):.6f}) | {t_row['holm_exact']} ({float(t_row['holm_p']):.6f}) | **{vz}** |"
            )

    # Audit & Replay Receipt
    lines += [
        "",
        "---",
        "",
        "## 三、独立审计与重放核验回执",
        "",
    ]
    if audit_payload:
        sd = audit_payload.get("seed_disjointness") or {}
        lines.append(f"- **种子池零交集（覆盖全部 24 个历史区间）**：`disjoint = {sd.get('disjoint')}`")
        for ax in ("primary", "secondary"):
            if ax in audit_payload:
                ax_info = audit_payload[ax] or {}
                rep_v = ax_info.get("replay_verify") or {}
                aud_v = ax_info.get("audit") or {}
                lines.append(
                    f"- **{ax.title()} Audit**：`links={aud_v.get('links')}`，`max_aggregate_diff={aud_v.get('max_aggregate_diff')}`，`max_stat_diff={aud_v.get('max_stat_diff')}`，"
                    f"`replay_passed={rep_v.get('passed')}`"
                )
    else:
        lines.append("- （未提供 `p11_audit.json` 审计载荷）")

    report_md = "\n".join(lines) + "\n"

    # Build concise CLI_SYNC_SUMMARY.md
    sync_lines = [
        "# P11 Cross-CLI Research Synchronization Summary",
        "",
        f"- **Baseline Commit**: `{P10_BASELINE_COMMIT}`",
        f"- **Freeze SHA-256**: {freeze_label}",
        f"- **Mode**: `{audit_payload.get('mode', 'unknown') if audit_payload else 'unknown'}`",
        "",
        "## Verbatim Verdicts (`支持` / `反证` / `证据不足`)",
        "",
        "| Axis | Claim / Contrast | Verdict |",
        "| :--- | :--- | :---: |",
    ]
    if verdict_records:
        for vr in verdict_records:
            sync_lines.append(f"| {vr['axis']} | `{vr['name']}` | **{vr['verdict_zh']}** |")
    else:
        sync_lines.append("| — | `none` | **证据不足** |")

    if primary_summary is not None:
        p_tot = (primary_summary.get("totals") or {}).get("sparse") or {}
        sync_lines += [
            "",
            "## Primary Axis Highlights",
            f"- `p11_chained_wavefront` 3-ID BLER: `{_fmt_pct((p_tot.get('p11_chained_wavefront') or {}).get('bler'))}`",
            f"- `p11_payload_only` 3-ID BLER: `{_fmt_pct((p_tot.get('p11_payload_only') or {}).get('bler'))}`",
            f"- `p9_jeffreys_student_t` 3-ID BLER: `{_fmt_pct((p_tot.get('p9_jeffreys_student_t') or {}).get('bler'))}`",
        ]
    if secondary_summary is not None:
        s_id = (secondary_summary.get("id_equal") or {}).get("sparse") or {}
        s_ood = (secondary_summary.get("ood_equal") or {}).get("sparse") or {}
        sync_lines += [
            "",
            "## Secondary Axis Highlights",
            f"- `em10_transport_attn` 4-ID BLER: `{_fmt_pct((s_id.get('em10_transport_attn') or {}).get('bler'))}` (OOD: `{_fmt_pct((s_ood.get('em10_transport_attn') or {}).get('bler'))}`)",
            f"- `em10_abs_no_transport` 4-ID BLER: `{_fmt_pct((s_id.get('em10_abs_no_transport') or {}).get('bler'))}`",
            f"- `em10_local` 4-ID BLER: `{_fmt_pct((s_id.get('em10_local') or {}).get('bler'))}`",
            f"- `em10_no_history` 4-ID BLER: `{_fmt_pct((s_id.get('em10_no_history') or {}).get('bler'))}`",
        ]

    sync_md = "\n".join(sync_lines) + "\n"
    return report_md, sync_md


def render_reports(
    *,
    primary_summary: Optional[Mapping[str, Any]] = None,
    secondary_summary: Optional[Mapping[str, Any]] = None,
    audit_payload: Optional[Mapping[str, Any]] = None,
    freeze_sha: Optional[str] = None,
    allow_partial: bool = False,
) -> tuple[str, str]:
    """Render `(P11_REPORT.md, CLI_SYNC_SUMMARY.md)` with verbatim Chinese verdicts."""
    return _render_reports_validated(
        primary_summary=primary_summary,
        secondary_summary=secondary_summary,
        audit_payload=audit_payload,
        freeze_sha=freeze_sha,
        allow_partial=allow_partial,
        require_disk_fields=False,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Render Phase 11 formal report and CLI sync summary.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR, help="Directory with summary & audit JSONs.")
    parser.add_argument("--out-report", type=Path, default=ROOT / "P11_REPORT.md", help="Output markdown report path.")
    parser.add_argument("--out-sync", type=Path, default=ROOT / "CLI_SYNC_SUMMARY.md", help="Output CLI sync summary path.")
    parser.add_argument("--audit-json", type=Path, default=None, help="Optional explicit p11_audit.json path.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH, help="Freeze manifest path.")
    parser.add_argument("--allow-dirty-agents", action="store_true", help="Allow dirty `.agents/` during freeze verify.")
    parser.add_argument("--no-verify-freeze", action="store_true", help="Skip freeze verification (for dry_run/tests).")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow rendering without p11_audit.json or when only one axis summary is present.",
    )
    args = parser.parse_args(argv)

    freeze_sha: Optional[str] = None
    if not args.no_verify_freeze:
        freeze_sha = verify_freeze(manifest_path=args.manifest, allow_dirty_agents=args.allow_dirty_agents)

    if not args.results_dir.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {args.results_dir}")
    if args.audit_json is not None and not args.audit_json.is_file():
        raise FileNotFoundError(f"Missing specified audit JSON: {args.audit_json}")

    audit_path = args.audit_json if args.audit_json is not None else (args.results_dir / "p11_audit.json")
    pri_path = args.results_dir / "p11_primary_summary.json"
    sec_path = args.results_dir / "p11_secondary_summary.json"

    pri_sum = _read_json_dict(pri_path) if pri_path.is_file() else None
    sec_sum = _read_json_dict(sec_path) if sec_path.is_file() else None
    audit_payload: Optional[dict[str, Any]] = _read_json_dict(audit_path) if audit_path.is_file() else None

    if not args.allow_partial:
        if audit_payload is None:
            raise FileNotFoundError(f"Missing required audit payload: {audit_path}")
        _validate_audit_payload(audit_payload)
        mode = audit_payload.get("mode")
        axis = audit_payload.get("axis")
        need_pri = (
            ("primary" in audit_payload)
            or (mode in ("primary", "all"))
            or (mode == "dry_run" and axis in ("primary", "both"))
        )
        need_sec = (
            ("secondary" in audit_payload)
            or (mode in ("secondary", "all"))
            or (mode == "dry_run" and axis in ("secondary", "both"))
        )
        if not need_pri and not need_sec:
            raise ValueError(f"Audit payload {audit_path} declares neither primary nor secondary axis results.")
        if need_pri and pri_sum is None:
            raise FileNotFoundError(f"Missing required primary summary: {pri_path}")
        if need_sec and sec_sum is None:
            raise FileNotFoundError(f"Missing required secondary summary: {sec_path}")
    else:
        if pri_sum is None and sec_sum is None:
            raise FileNotFoundError(
                f"No summary JSON (p11_primary_summary.json / p11_secondary_summary.json) found in {args.results_dir}"
            )
    if freeze_sha is None and audit_payload is not None:
        freeze_sha = audit_payload.get("freeze_sha256")

    report_md, sync_md = _render_reports_validated(
        primary_summary=pri_sum,
        secondary_summary=sec_sum,
        audit_payload=audit_payload,
        freeze_sha=freeze_sha,
        allow_partial=args.allow_partial,
        require_disk_fields=True,
    )
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_sync.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(report_md, encoding="utf-8", newline="\n")
    args.out_sync.write_text(sync_md, encoding="utf-8", newline="\n")
    print(json.dumps({
        "status": "OK",
        "out_report": str(args.out_report),
        "out_sync": str(args.out_sync),
        "freeze_sha256": freeze_sha,
    }, indent=2))


__all__ = ["main", "render_reports"]


if __name__ == "__main__":
    main()
