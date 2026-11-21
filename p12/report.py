"""Markdown report generator for Phase 12 (`p12/P12_REPORT.md`) strictly using audited verdicts."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def write_p12_report(audit: Mapping[str, Any], manifest: Mapping[str, Any], out_path: Path) -> None:
    p = audit["primary"]
    s = audit["secondary"]

    lines: list[str] = [
        "# EM-FloatFormer Phase 12（`p12/`）闭卷正式实验报告（`G = 16`, 9 dB）",
        "",
        f"> **事前冻结清单哈希**：`{manifest['bundle_sha256']}`  ",
        f"> **冻结基线提交**：`{manifest['git_head_at_freeze']}`（承接 P11 `307d62f`，零修改任何历史文件，`main` 未动）  ",
        f"> **严格互斥正式种子总数**：`{manifest['total_unique_formal_seeds']}`（主轴 `1,536` ID + `640` OOD；副轴 `1,024` ID + `384` OOD/真延迟重置）  ",
        f"> **单进程精确复现（ID + OOD, Clusters `1, 16`）**：主轴 `PASS={p['replay_audit']['exact_pass']}`（`{p['replay_audit']['checked_links']}` 链路，`max_float_diff={p['replay_audit']['max_float_diff']:.1e}`）；副轴 `PASS={s['replay_audit']['exact_pass']}`（`{s['replay_audit']['checked_links']}` 链路，`max_float_diff={s['replay_audit']['max_float_diff']:.1e}`）  ",
        f"> **固定顺序门控（Gate 1 Primary → Gate 2 Secondary）**：`gate1_passed={audit['fixed_sequence_gate1_passed']}`  ",
        "",
        "---",
        "",
        "## 一、主轴 Stage 3（格点 3-径，对称双侧 `K=3` 与等算力 `K=2` vs 原版 P11）",
        "",
        "### 1.1 ID 核心 BLER（`1,536` 链路 = `24,576` 核心块，9 dB）",
        "",
        "| 机制 | `dense/clear` | `p11_legacy` | `p12_symm_2fold` (`10` 助手) | **`p12_symm_3fold` (`23` 助手)** | `oracle_true_csi` | `K=3` 假同化率 | `K=3` 探针 NMSE | `K=3` 中位耗时 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for reg, row in p["id_by_regime"].items():
        k3 = row["p12_symm_3fold"]
        lines.append(
            f"| `{reg}` | `{row['dense_always_clear']['core_bler_pct']:.2f}%` | `{row['p11_legacy']['core_bler_pct']:.2f}%` | "
            f"`{row['p12_symm_2fold']['core_bler_pct']:.2f}%` | **`{k3['core_bler_pct']:.2f}%`** | "
            f"`{row['oracle_true_csi']['core_bler_pct']:.2f}%` | `{k3['false_assim_pct']:.2f}%` | `{k3['mean_probe_nmse']:.4f}` | `{k3['median_ms']:.1f} ms` |"
        )
    p3 = p["pooled_3id_bler_pct"]
    lines.append(
        f"| **3-ID 合并** | `{p3['dense_always_clear']:.2f}%` | `{p3['p11_legacy']:.2f}%` | `{p3['p12_symm_2fold']:.2f}%` | "
        f"**`{p3['p12_symm_3fold']:.2f}%`** | `{p3['oracle_true_csi']:.2f}%` | — | — | — |"
    )

    lines.extend([
        "",
        "### 1.2 主轴正式统计裁决（`G = 16`, 精确 `65,536` 态符号翻转 + Holm 族校正）",
        "",
        "| 预注册假设检验 | 均值差 / 指标 | 95% 区间 / 上界 | 胜/平/负 | $p_{\\text{raw}}$ / $p_{\\text{Holm}}$ | 逐字裁决 |",
        "| :--- | ---: | ---: | :---: | ---: | :---: |",
    ])
    for name, st in p["holm_family"].items():
        lines.append(
            f"| `{name}` | `{st['mean_pp']:+.2f} pp` | `[{st['ci95_lo_pp']:+.2f}, {st['ci95_hi_pp']:+.2f}] pp` | "
            f"`{st['wins']}/{st['ties']}/{st['losses']}` | `{st['p_two_sided']:.6f}` / `{st['p_holm']:.6f}` | **{st['verdict']}** |"
        )
    ni3 = p["ni_k3_continuous"]
    ni2 = p["ni_k2_continuous"]
    s10 = p["sub10_k3_full_reset"]
    lines.append(
        f"| `k3_continuous_ni` (`< +1.00 pp`) | `{ni3['mean_excess_pp']:+.4f} pp` | `U95 = {ni3['u95_pp']:+.4f} pp` | — | `df=15` 单侧 95% | **{ni3['verdict']}** |"
    )
    lines.append(
        f"| `k2_continuous_ni` (`< +1.00 pp`) | `{ni2['mean_excess_pp']:+.4f} pp` | `U95 = {ni2['u95_pp']:+.4f} pp` | — | `df=15` 单侧 95% | **{ni2['verdict']}** |"
    )
    lines.append(
        f"| `k3_full_reset_sub10` (`< 10.00%`) | `{s10['mean_pct']:.2f}%` | `U95 = {s10['u95_pct']:.2f}%` | — | `df=15` 单侧 95% | **{s10['verdict']}** |"
    )

    lines.extend([
        "",
        "### 1.3 主轴 OOD 与真正支撑外（Out-of-Support）评测（`640` 链路 = `10,240` 块）",
        "",
        "| OOD 机制 | `dense/clear` | `p11_legacy` | `p12_symm_2fold` | **`p12_symm_3fold`** | `oracle_true_csi` |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for reg, row in p["ood_by_regime"].items():
        lines.append(
            f"| `{reg}` | `{row['dense_always_clear']['core_bler_pct']:.2f}%` | `{row['p11_legacy']['core_bler_pct']:.2f}%` | "
            f"`{row['p12_symm_2fold']['core_bler_pct']:.2f}%` | **`{row['p12_symm_3fold']['core_bler_pct']:.2f}%`** | "
            f"`{row['oracle_true_csi']['core_bler_pct']:.2f}%` |"
        )

    # Extra held-out goodput line
    c_row = p["id_by_regime"]["continuous"]
    lines.extend([
        "",
        f"- **额外行 `(0, 9)` 零泄漏留出读出（`continuous`）**：`K=3` 留出 BLER = `{c_row['p12_symm_3fold']['extra_heldout_bler_pct']:.2f}%`，净有效吞吐 **`{c_row['p12_symm_3fold']['net_goodput_bits_per_frame']:.1f} bit/frame`** vs `dense/clear` `{c_row['dense_always_clear']['net_goodput_bits_per_frame']:.1f} bit/frame`（净增 `{c_row['p12_symm_3fold']['net_goodput_bits_per_frame'] - c_row['dense_always_clear']['net_goodput_bits_per_frame']:+.1f} bit/frame`）。",
        "",
        "---",
        "",
        "## 二、副轴 Stage 4（离格连续时延 2–5 径，锁定梳状 `(0, 9, 10, 19)`）",
        "",
        "### 2.1 ID 评测（`1,024` 链路 = `18,432` 块，与 P11 完全等规模）",
        "",
        "| 离格 ID 机制 | `p11_classical` | `p11_transport_attn` | `p12_brefit_off` | **`p12_offgrid_physics`** | `oracle_true_channel` | `P12` 延迟重置触发 | `P12` 探针 NMSE |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for reg, row in s["id_by_regime"].items():
        ph = row["p12_offgrid_physics"]
        lines.append(
            f"| `{reg}` | `{row['p11_classical_per_ray']['bler_pct']:.2f}%` | `{row['p11_transport_attn_ens']['bler_pct']:.2f}%` | "
            f"`{row['p12_brefit_off_control']['bler_pct']:.2f}%` | **`{ph['bler_pct']:.2f}%`** | "
            f"`{row['oracle_true_channel']['bler_pct']:.2f}%` | `{ph['delay_reset_pct']:.2f}%` | `{ph['mean_probe_nmse']:.4f}` |"
        )
    s4 = s["pooled_4id_bler_pct"]
    lines.append(
        f"| **4-ID 合并 (`18,432` 块)** | `{s4['p11_classical_per_ray']:.2f}%` | `{s4['p11_transport_attn_ens']:.2f}%` | "
        f"`{s4['p12_brefit_off_control']:.2f}%` | **`{s4['p12_offgrid_physics']:.2f}%`** | `{s4['oracle_true_channel']:.2f}%` | — | — |"
    )

    lines.extend([
        "",
        "### 2.2 副轴真延迟重置与 OOD 评测（`384` 链路 = `6,912` 块）",
        "",
        "| 离格 OOD / 真重置机制 | `p11_classical` | `p11_transport_attn` | `p12_brefit_off` | **`p12_offgrid_physics`** | `oracle_true_channel` | `P12` 延迟重置触发 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for reg, row in s["ood_by_regime"].items():
        ph = row["p12_offgrid_physics"]
        lines.append(
            f"| `{reg}` | `{row['p11_classical_per_ray']['bler_pct']:.2f}%` | `{row['p11_transport_attn_ens']['bler_pct']:.2f}%` | "
            f"`{row['p12_brefit_off_control']['bler_pct']:.2f}%` | **`{ph['bler_pct']:.2f}%`** | "
            f"`{row['oracle_true_channel']['bler_pct']:.2f}%` | `{ph['delay_reset_pct']:.2f}%` |"
        )

    lines.extend([
        "",
        "### 2.3 副轴正式统计裁决（固定顺序 Gate 2，`65,536` 态符号翻转 + Holm 族校正）",
        "",
        "| 预注册假设检验 | 均值改善 | 95% 区间 | 胜/平/负 | $p_{\\text{raw}}$ / $p_{\\text{Holm}}$ | 逐字裁决 |",
        "| :--- | ---: | ---: | :---: | ---: | :---: |",
    ])
    for name, st in s["holm_family"].items():
        lines.append(
            f"| `{name}` | `{st['mean_pp']:+.2f} pp` | `[{st['ci95_lo_pp']:+.2f}, {st['ci95_hi_pp']:+.2f}] pp` | "
            f"`{st['wins']}/{st['ties']}/{st['losses']}` | `{st['p_two_sided']:.6f}` / `{st['p_holm']:.6f}` | **{st['verdict']}** |"
        )

    lines.extend([
        "",
        "### 2.4 副轴分径数 $P$ 与分游走锚点位置拆解（ID 合并）",
        "",
        "| 维度 | 子集 | `p11_transport_attn` | **`p12_offgrid_physics`** | `oracle_true_channel` |",
        "| :--- | :--- | ---: | ---: | ---: |",
    ])
    for p_k, d in s["id_by_ray_count_bler_pct"].items():
        lines.append(
            f"| 径数 | `P={p_k}` | `{d['p11_transport_attn_ens']:.2f}%` | **`{d['p12_offgrid_physics']:.2f}%`** | `{d['oracle_true_channel']:.2f}%` |"
        )
    for anc_k, d in s["id_by_anchor_row_bler_pct"].items():
        lines.append(
            f"| 锚点行 | `r={anc_k}` | `{d['p11_transport_attn_ens']:.2f}%` | **`{d['p12_offgrid_physics']:.2f}%`** | `{d['oracle_true_channel']:.2f}%` |"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
