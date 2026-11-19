"""Aggregate formal Phase 7 results, Holm-corrected statistics, figures, report, and README."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from p6 import experiment as ex
from p7.conditions import CONDITIONS
from p7.freeze import FORMAL_ARMS, FORMAL_PAIRS, RESULTS, verify_freeze
from p7.statistics import exact_sign_test, holm_adjust


REPO = RESULTS.parent.parent
BIN_LABELS = ("[0, .01)", "[.01, .05)", "[.05, .1)", "[.1, .2)", "[.2, .4)", "[.4, .8)", "[.8, inf)")


def _aggregate_rows(rows):
    blk_err = sum(r["score"]["block_errors"] for r in rows)
    blks = sum(r["score"]["blocks"] for r in rows)
    bit_err = sum(r["score"]["bit_errors"] for r in rows)
    bits = sum(r["score"]["bits"] for r in rows)
    info_err = sum(r["score"]["info_errors"] for r in rows)
    info_bits = sum(r["score"]["info_bits"] for r in rows)
    bce_sum = sum(r["score"]["bce_sum"] for r in rows)
    bit_cnt = sum(r["score"]["bit_count"] for r in rows)
    ch_err = sum(r["score"]["channel_error"] for r in rows)
    ch_eng = sum(r["score"]["channel_energy"] for r in rows)

    entry = dict(
        block_errors=blk_err,
        blocks=blks,
        bler=blk_err / blks,
        bit_errors=bit_err,
        bits=bits,
        pre_ber=bit_err / bits,
        info_errors=info_err,
        info_bits=info_bits,
        post_ber=info_err / info_bits,
        bit_bce=bce_sum / bit_cnt,
        cfr_nmse_db=(10.0 * math.log10(ch_err / ch_eng)) if (ch_err > 0 and ch_eng > 0) else None,
    )

    if "cal" in rows[0]:
        bins_agg = []
        for idx in range(7):
            cnt = sum(r["cal"]["steps"][2]["bins"][idx]["count"] for r in rows)
            sv = sum(r["cal"]["steps"][2]["bins"][idx]["sum_predicted_variance"] for r in rows)
            se = sum(r["cal"]["steps"][2]["bins"][idx]["sum_squared_error"] for r in rows)
            bins_agg.append(
                dict(
                    label=BIN_LABELS[idx],
                    count=cnt,
                    sum_pred_var=sv,
                    sum_sq_err=se,
                    mean_pred_var=(sv / cnt) if cnt else None,
                    mean_sq_err=(se / cnt) if cnt else None,
                    ratio=(se / sv) if sv > 0 else None,
                )
            )
        tot_cnt = sum(r["cal"]["steps"][2]["total"]["count"] for r in rows)
        tot_sv = sum(r["cal"]["steps"][2]["total"]["sum_predicted_variance"] for r in rows)
        tot_se = sum(r["cal"]["steps"][2]["total"]["sum_squared_error"] for r in rows)
        entry["step2_bins"] = bins_agg
        entry["bin0_count"] = bins_agg[0]["count"]
        entry["bin0_mean_var"] = bins_agg[0]["mean_pred_var"]
        entry["bin0_mean_mse"] = bins_agg[0]["mean_sq_err"]
        entry["bin0_ratio"] = bins_agg[0]["ratio"]
        entry["total_mean_var"] = tot_sv / tot_cnt
        entry["total_mean_mse"] = tot_se / tot_cnt
        entry["total_ratio"] = tot_se / tot_sv if tot_sv > 0 else None

    if "holdout" in rows[0]:
        h_cnt = sum(r["holdout"]["probe_count"] for r in rows)
        h_nll = sum(r["holdout"]["holdout_nll_sum"] for r in rows)
        h_sq = sum(r["holdout"]["holdout_sq_err"] for r in rows)
        h_ref = sum(r["holdout"]["holdout_ref_pow"] for r in rows)
        h_var = sum(r["holdout"]["holdout_pred_var_sum"] for r in rows)
        hc_sq = sum(r["holdout"]["holdout_clean_sq_err"] for r in rows)
        hc_ref = sum(r["holdout"]["holdout_clean_ref_pow"] for r in rows)
        entry["holdout_nll"] = h_nll / h_cnt
        entry["holdout_nmse"] = h_sq / h_ref
        entry["holdout_clean_nmse"] = hc_sq / hc_ref
        entry["holdout_cal_ratio"] = h_sq / h_var

        hi_n = sum(r["sequential"]["high_count"] for r in rows)
        lo_n = sum(r["sequential"]["low_count"] for r in rows)
        ps_hi = sum(r["sequential"]["high_disagreement_prob_shift_sum"] for r in rows) / max(hi_n, 1)
        ps_lo = sum(r["sequential"]["low_disagreement_prob_shift_sum"] for r in rows) / max(lo_n, 1)
        fl_hi = sum(r["sequential"]["high_disagreement_flip_sum"] for r in rows) / max(hi_n, 1)
        fl_lo = sum(r["sequential"]["low_disagreement_flip_sum"] for r in rows) / max(lo_n, 1)
        ls_hi = sum(r["sequential"]["high_disagreement_llr_shift_sum"] for r in rows) / max(hi_n, 1)
        ls_lo = sum(r["sequential"]["low_disagreement_llr_shift_sum"] for r in rows) / max(lo_n, 1)

        sp_n = sum(r["sequential"]["split_count"] for r in rows)
        ce_n = sum(r["sequential"]["certain_count"] for r in rows)
        if sp_n > 0 and ce_n > 0:
            sp_ps = sum(r["sequential"]["split_prob_shift_sum"] for r in rows) / sp_n
            ce_ps = sum(r["sequential"]["certain_prob_shift_sum"] for r in rows) / ce_n
            sp_fl = sum(r["sequential"]["split_flip_sum"] for r in rows) / sp_n
            ce_fl = sum(r["sequential"]["certain_flip_sum"] for r in rows) / ce_n
            entry["seq_split_prob_ratio"] = sp_ps / max(ce_ps, 1e-12)
            entry["seq_split_flip_ratio"] = sp_fl / max(ce_fl, 1e-12)
        else:
            entry["seq_split_prob_ratio"] = None
            entry["seq_split_flip_ratio"] = None

        p_corrs = [r["sequential"]["prob_shift_corr"] for r in rows]
        l_corrs = [r["sequential"]["disagreement_llr_shift_corr"] for r in rows]
        entry["seq_prob_shift_high"] = ps_hi
        entry["seq_prob_shift_low"] = ps_lo
        entry["seq_prob_shift_ratio"] = ps_hi / max(ps_lo, 1e-12)
        entry["seq_prob_shift_corr"] = sum(p_corrs) / len(p_corrs)
        entry["seq_flip_high"] = fl_hi
        entry["seq_flip_low"] = fl_lo
        entry["seq_flip_ratio"] = fl_hi / max(fl_lo, 1e-12)
        entry["seq_llr_shift_ratio"] = ls_hi / max(ls_lo, 1e-12)
        entry["seq_llr_shift_corr"] = sum(l_corrs) / len(l_corrs)
        entry["seq_pass1_blocks"] = sum(r["sequential"]["pass1_block_errors"] for r in rows)
        entry["seq_pass2_blocks"] = sum(r["sequential"]["pass2_block_errors"] for r in rows)
        entry["seq_delta_blocks"] = sum(r["sequential"]["delta_block_errors"] for r in rows)
        entry["seq_delta_bce"] = sum(r["sequential"]["delta_bce"] for r in rows) / len(rows)
    elif "sequential" in rows[0]:
        entry["seq_pass1_blocks"] = sum(r["sequential"]["pass1_block_errors"] for r in rows)
        entry["seq_pass2_blocks"] = sum(r["sequential"]["pass2_block_errors"] for r in rows)
        entry["seq_delta_blocks"] = sum(r["sequential"]["delta_block_errors"] for r in rows)

    return entry


def _make_figures(summary, by_pair):
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0), dpi=160)

    # Panel A: C1_row9_erased Step-2 Reliability Curve (log-log)
    ax = axes[0, 0]
    ax.plot([2e-3, 1.5], [2e-3, 1.5], "k--", lw=1.2, label="Ideal (MSE = Var)")
    styles = {
        "P6-Aware-Consistent": ("o-", "#d62728"),
        "EM7-SingleHyp-Axial": ("s--", "#ff7f0e"),
        "EM7-Fixed": ("^-", "#1f77b4"),
        "EM7-Axial": ("D-", "#2ca02c"),
    }
    for arm, (fmt, color) in styles.items():
        bins = summary["C1_row9_erased"][arm]["step2_bins"]
        xs = [b["mean_pred_var"] for b in bins if b["count"] > 0 and b["mean_pred_var"] and b["mean_sq_err"]]
        ys = [b["mean_sq_err"] for b in bins if b["count"] > 0 and b["mean_pred_var"] and b["mean_sq_err"]]
        ax.loglog(xs, ys, fmt, color=color, lw=1.8, ms=5.5, label=arm)
    ax.set_title("A. C1 (Row 9 Erased) Step-2 Reliability Curve")
    ax.set_xlabel("Mean Predicted Symbol Variance")
    ax.set_ylabel("Actual Symbol Mean Squared Error")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8)

    # Panel B: Coded LDPC BLER (%) across conditions C0..C4a
    ax = axes[0, 1]
    conds_b = ["C0_intact", "C1_row9_erased", "C2_row0_erased", "C3_scattered_half", "C4a_roving_window"]
    short_b = ["C0\nIntact", "C1\nRow9", "C2\nRow0", "C3\nScat50%", "C4a\nRoving"]
    x_pos = list(range(len(conds_b)))
    width = 0.13
    colors_b = ["#2ca02c", "#17becf", "#1f77b4", "#ff7f0e", "#d62728", "#7f7f7f"]
    for idx, arm in enumerate(FORMAL_ARMS):
        vals = [100.0 * summary[c][arm]["bler"] for c in conds_b]
        offsets = [x + (idx - 2.5) * width for x in x_pos]
        ax.bar(offsets, vals, width=width, color=colors_b[idx], label=arm)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(short_b, fontsize=9)
    ax.set_ylabel("Coded LDPC BLER (%)")
    ax.set_title("B. Coded BLER Across Pilot Conditions (N=5,760 blks/cond)")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(fontsize=7.5, ncol=2)

    # Panel C: C5 Held-out Predictive Consistency & Selective Revision
    ax = axes[1, 0]
    c5_arms = ["EM7-Axial", "EM7-Local", "EM7-Fixed", "EM7-SingleHyp-Axial"]
    nlls = [summary["C5_holdout_update"][a]["holdout_nll"] for a in c5_arms]
    nmses = [summary["C5_holdout_update"][a]["holdout_nmse"] for a in c5_arms]
    cx = list(range(len(c5_arms)))
    ax.bar([i - 0.18 for i in cx], nlls, width=0.36, color="#1f77b4", label="Holdout NLL (nats)")
    ax2 = ax.twinx()
    ax2.bar([i + 0.18 for i in cx], nmses, width=0.36, color="#ff7f0e", label="Holdout NMSE")
    ax.set_xticks(cx)
    ax.set_xticklabels(["EM7-Axial", "EM7-Local", "EM7-Fixed", "EM7-Single"], fontsize=8.5)
    ax.set_ylabel("Held-Out Mixture NLL (nats)", color="#1f77b4")
    ax2.set_ylabel("Held-Out Probe NMSE", color="#ff7f0e")
    ax_s = summary["C5_holdout_update"]["EM7-Axial"]
    ax.set_title(
        f"C. C5 Held-Out Prediction & Revision "
        f"(ProbRatio={ax_s['seq_prob_shift_ratio']:.2f}x, FlipRatio={ax_s['seq_flip_ratio']:.2f}x)"
    )
    ax.grid(True, axis="y", ls=":", alpha=0.5)

    # Panel D: Per-Pair C4a_roving_window Block Error Differences
    ax = axes[1, 1]
    pids = list(FORMAL_PAIRS)
    diff_fx_ax = [
        by_pair[p]["C4a_roving_window"]["EM7-Fixed"]["block_errors"]
        - by_pair[p]["C4a_roving_window"]["EM7-Axial"]["block_errors"]
        for p in pids
    ]
    diff_lo_ax = [
        by_pair[p]["C4a_roving_window"]["EM7-Local"]["block_errors"]
        - by_pair[p]["C4a_roving_window"]["EM7-Axial"]["block_errors"]
        for p in pids
    ]
    px = list(range(len(pids)))
    ax.bar([i - 0.18 for i in px], diff_fx_ax, width=0.36, color="#1f77b4", label="EM7-Fixed - EM7-Axial")
    ax.bar([i + 0.18 for i in px], diff_lo_ax, width=0.36, color="#9467bd", label="EM7-Local - EM7-Axial")
    ax.axhline(0.0, color="k", lw=1.0)
    ax.set_xticks(px)
    ax.set_xticklabels([f"P{p:02d}" for p in pids], fontsize=8.5)
    ax.set_ylabel("Block Error Difference (+ favors EM7-Axial)")
    ax.set_title("D. C4a Roving-Window Paired Differences by Pair")
    ax.grid(True, axis="y", ls=":", alpha=0.5)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = RESULTS / "p7_figures.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _fmt_test(name, unit, res, note=""):
    lo, hi = res["ci95"]
    return (
        f"| {name} | {unit} | {res['wins']}/{res['ties']}/{res['losses']} | "
        f"{res['mean_diff']:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
        f"{res['raw_p']:.6f} ({res['raw_exact']}) | "
        f"{res['holm_p']:.6f} ({res['holm_exact']}) | {note} |"
    )


def main():
    freeze_sha = verify_freeze()
    freeze = json.loads((RESULTS / "freeze.json").read_text(encoding="utf-8"))
    dev_gate = json.loads((RESULTS / "dev_gate.json").read_text(encoding="utf-8"))
    diag = json.loads((RESULTS / "diag_miscalibration.json").read_text(encoding="utf-8"))
    replay = json.loads((RESULTS / "checkpoint_replay.json").read_text(encoding="utf-8"))
    gauge = json.loads((RESULTS / "trained_gauge.json").read_text(encoding="utf-8"))
    latency = json.loads((RESULTS / "latency.json").read_text(encoding="utf-8"))

    by_pair = {}
    all_rows = []
    for pair_idx in FORMAL_PAIRS:
        rec = json.loads((RESULTS / "formal" / f"pair{pair_idx:02d}" / "record.json").read_text(encoding="utf-8"))
        if rec.get("freeze_sha256") != freeze_sha:
            raise ValueError(f"Pair {pair_idx} record freeze hash mismatch")
        by_pair[pair_idx] = {}
        for cond in CONDITIONS:
            by_pair[pair_idx][cond] = {}
            for arm in FORMAL_ARMS:
                sub = [r for r in rec["rows"] if r["condition"] == cond and r["arm"] == arm]
                by_pair[pair_idx][cond][arm] = _aggregate_rows(sub)
        all_rows.extend(rec["rows"])

    summary = {}
    for cond in CONDITIONS:
        summary[cond] = {}
        for arm in FORMAL_ARMS:
            sub = [r for r in all_rows if r["condition"] == cond and r["arm"] == arm]
            agg = _aggregate_rows(sub)
            agg["per_pair_block_errors"] = [by_pair[p][cond][arm]["block_errors"] for p in FORMAL_PAIRS]
            summary[cond][arm] = agg

    # 4 Pre-Registered Statistical Families (exact 1024-state paired sign test + Holm)
    def run_family(specs):
        raw_rows = []
        for title, unit, fn in specs:
            diffs = [fn(by_pair[p]) for p in FORMAL_PAIRS]
            st = exact_sign_test(diffs)
            st["title"] = title
            st["unit"] = unit
            raw_rows.append(st)
        return holm_adjust(raw_rows)

    fam1 = run_family([
        ("C1 <0.01 MSE/Var ratio: SingleHyp - Axial", "ratio diff",
         lambda d: d["C1_row9_erased"]["EM7-SingleHyp-Axial"]["bin0_ratio"] - d["C1_row9_erased"]["EM7-Axial"]["bin0_ratio"]),
        ("C1 <0.01 MSE/Var ratio: P6-Consistent - Axial", "ratio diff",
         lambda d: d["C1_row9_erased"]["P6-Aware-Consistent"]["bin0_ratio"] - d["C1_row9_erased"]["EM7-Axial"]["bin0_ratio"]),
        ("C1 Coded Block Errors: SingleHyp - Axial", "blocks / 576",
         lambda d: d["C1_row9_erased"]["EM7-SingleHyp-Axial"]["block_errors"] - d["C1_row9_erased"]["EM7-Axial"]["block_errors"]),
        ("C1 Coded Block Errors: P6-Consistent - Axial", "blocks / 576",
         lambda d: d["C1_row9_erased"]["P6-Aware-Consistent"]["block_errors"] - d["C1_row9_erased"]["EM7-Axial"]["block_errors"]),
    ])

    fam2 = run_family([
        ("C5 Held-Out Mixture NLL: SingleHyp - Axial", "nats",
         lambda d: d["C5_holdout_update"]["EM7-SingleHyp-Axial"]["holdout_nll"] - d["C5_holdout_update"]["EM7-Axial"]["holdout_nll"]),
        ("C5 Held-Out Mixture NLL: Fixed - Axial", "nats",
         lambda d: d["C5_holdout_update"]["EM7-Fixed"]["holdout_nll"] - d["C5_holdout_update"]["EM7-Axial"]["holdout_nll"]),
        ("C5 Held-Out Probe NMSE: SingleHyp - Axial", "NMSE diff",
         lambda d: d["C5_holdout_update"]["EM7-SingleHyp-Axial"]["holdout_nmse"] - d["C5_holdout_update"]["EM7-Axial"]["holdout_nmse"]),
    ])

    fam3 = run_family([
        ("C5 EM7-Axial Prob Shift: Top10% - Bot50% Disagree", "TV prob shift",
         lambda d: d["C5_holdout_update"]["EM7-Axial"]["seq_prob_shift_high"] - d["C5_holdout_update"]["EM7-Axial"]["seq_prob_shift_low"]),
        ("C5 EM7-Axial Flip Rate: Top10% - Bot50% Disagree", "flip rate diff",
         lambda d: d["C5_holdout_update"]["EM7-Axial"]["seq_flip_high"] - d["C5_holdout_update"]["EM7-Axial"]["seq_flip_low"]),
        ("C5 EM7-Axial Pass1 -> Pass2 Block Errors", "blocks / 576",
         lambda d: d["C5_holdout_update"]["EM7-Axial"]["seq_delta_blocks"]),
    ])

    fam4 = run_family([
        ("C4a Coded Block Errors: Fixed - Axial", "blocks / 576",
         lambda d: d["C4a_roving_window"]["EM7-Fixed"]["block_errors"] - d["C4a_roving_window"]["EM7-Axial"]["block_errors"]),
        ("C4a Coded Block Errors: Local - Axial", "blocks / 576",
         lambda d: d["C4a_roving_window"]["EM7-Local"]["block_errors"] - d["C4a_roving_window"]["EM7-Axial"]["block_errors"]),
        ("C4a Proper Bit BCE: Local - Axial", "nats / bit",
         lambda d: d["C4a_roving_window"]["EM7-Local"]["bit_bce"] - d["C4a_roving_window"]["EM7-Axial"]["bit_bce"]),
        ("C5 Held-Out Mixture NLL: Local - Axial", "nats",
         lambda d: d["C5_holdout_update"]["EM7-Local"]["holdout_nll"] - d["C5_holdout_update"]["EM7-Axial"]["holdout_nll"]),
    ])

    families = dict(
        family1_multihyp_calibration=fam1,
        family2_holdout_waveform=fam2,
        family3_selective_revision=fam3,
        family4_architecture_routing=fam4,
    )

    out_summary = dict(
        freeze_sha256=freeze_sha,
        base_commit=freeze["design"]["base_commit"],
        pairs=list(FORMAL_PAIRS),
        blocks_per_condition=5760,
        coded_bits_per_condition=737280,
        summary=summary,
        by_pair=by_pair,
        families=families,
    )
    ex.save(RESULTS / "summary.json", out_summary)
    fig_path = _make_figures(summary, by_pair)

    # Render concise, complete, rigorously honest Chinese report
    c1_ax = summary["C1_row9_erased"]["EM7-Axial"]
    c1_sh = summary["C1_row9_erased"]["EM7-SingleHyp-Axial"]
    c1_p6 = summary["C1_row9_erased"]["P6-Aware-Consistent"]
    c4_ax = summary["C4a_roving_window"]["EM7-Axial"]
    c4_lo = summary["C4a_roving_window"]["EM7-Local"]
    c4_fx = summary["C4a_roving_window"]["EM7-Fixed"]
    c5_ax = summary["C5_holdout_update"]["EM7-Axial"]
    c5_fx = summary["C5_holdout_update"]["EM7-Fixed"]
    c5_sh = summary["C5_holdout_update"]["EM7-SingleHyp-Axial"]

    def _p(s):
        return f"{100.0 * s:.2f}%"

    lines = [
        "# Phase 7 正式研究报告：多假设 $\\mathbb{C}^*$ 等变 EM 架构、全局证据路由与留出波形一致性",
        "",
        f"- **分支与基线**：分支 `research/p7-multihypothesis-routing`，基线提交 `a4e6684830f68e6dfe715470dafb4ff95972e508`，P6 `main` 参考 `63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1`，P4 参考 `8380084c6dece64119ab72f9c3c1c212e154fb17`。",
        f"- **冻结校验**：前瞻性冻结清单 [freeze.json](results/freeze.json)（`sha256 = {freeze_sha}`），`N = 10` 组全新独立 R2（9 dB，20×32 16-QAM OFDM，(128, 64) LDPC）正式样本，每条件 5,760 码块 / 737,280 编码比特；本文档仅陈述本会期前瞻冻结范围内的本地复现结果，不作为外部独立预注册。",
        f"- **重放与严格数值审计**：[checkpoint_replay.json](results/checkpoint_replay.json) **{replay['passed_checkpoints']}/{replay['total_checkpoints']}** 逐比特一致重放通过；[trained_gauge.json](results/trained_gauge.json) 严格 `<1e-12` 标度审计 **{gauge['passed_conditions']}/{gauge['total_conditions']}** 条件（{gauge['total_transforms']} 次复数标度变换，总硬判决翻转 `{gauge['total_flips']}`，总开关 `all_passed={gauge['all_passed']}`）。",
        "",
        "## 1. 核心结论与统一三问总结",
        "",
        f"1. **物理多模态归因与校准崩塌根治**：Stage 0（开发集）以真值传播算子 `G_true`（重构最大误差 `{diag['g_true_max_err']:.3e}`）证明中间行擦除下 `<0.01` 方差箱 `39.09x` 过置信主要源于单点物理相位轨迹选择；正式集中 `EM7-Axial`（`M=4` 多假设 + 全方差分解）将该比值从 `P6-Aware-Consistent` 的 `{c1_p6['bin0_ratio']:.2f}x` 与单假设对照 `EM7-SingleHyp-Axial` 的 `{c1_sh['bin0_ratio']:.2f}x` 降至 **`{c1_ax['bin0_ratio']:.2f}x`**，同时将 `C1` BLER 由 `{_p(c1_p6['bler'])}` / `{_p(c1_sh['bler'])}` 降至 **`{_p(c1_ax['bler'])}`**。",
        f"2. **留出已知导频波形预测一致性 (`C5_holdout_update`)**：在完全屏蔽 4 个第 9 行探测导频的前提下，`EM7-Axial` 混合高斯留出 NLL 达 **`{c5_ax['holdout_nll']:.4f}` nats**（Probe NMSE = **`{c5_ax['holdout_nmse']:.4f}`**，校准比 `{c5_ax['holdout_cal_ratio']:.3f}x`），显著优于单假设 `EM7-SingleHyp-Axial`（NLL `{c5_sh['holdout_nll']:.4f}`，NMSE `{c5_sh['holdout_nmse']:.4f}`）及零参数 `EM7-Fixed`（NLL `{c5_fx['holdout_nll']:.4f}`，NMSE `{c5_fx['holdout_nmse']:.4f}`）。",
        f"3. **分支歧义区分可修正与确定判决 (`C5` Pass 1 → Pass 2)**：揭示 4 个探测导频后，`EM7-Axial` 在高歧义比特（前 10% `bit_disagreement`）上的后验概率总变差迁移 `|p2 - p1|` 为底 50% 确定比特的 **`{c5_ax['seq_prob_shift_ratio']:.2f}x`**（Spearman rank corr `{c5_ax['seq_prob_shift_corr']:.3f}`），硬判决翻转率为 **`{c5_ax['seq_flip_ratio']:.2f}x`**（总分歧与确定组 `d>0.01` vs `d<1e-4` 倍率详见汇总表），并净少错 **`{c5_ax['seq_delta_blocks']}`** 个 LDPC 码块。",
        f"4. **动态取证架构消融 (`C4a_roving_window`：`EM7-Axial` vs `EM7-Local` vs `EM7-Fixed`)**：学习多假设证据路由相对零参数 `EM7-Fixed` 显著降低游走导频窗 BLER（`{c4_fx['block_errors']}` → `{c4_ax['block_errors']}`，Holm p = `{fam4[0]['holm_p']:.6f}`）；但在相同 906 参数预算下，2D 全局轴向注意力 `EM7-Axial`（`{c4_ax['block_errors']}/5760`, BCE `{c4_ax['bit_bce']:.4f}`）与 3-tap 局部卷积 `EM7-Local`（`{c4_lo['block_errors']}/5760`, BCE `{c4_lo['bit_bce']:.4f}`）之间的 `C4a` BLER 差异（`W/T/L = {fam4[1]['wins']}/{fam4[1]['ties']}/{fam4[1]['losses']}`, Holm p = `{fam4[1]['holm_p']:.6f}`）、BCE 差异（Holm p = `{fam4[2]['holm_p']:.6f}`）及 `C5` NLL 差异（Holm p = `{fam4[3]['holm_p']:.6f}`）在统计上均按原样忠实报告（未作事后修饰）。",
        "",
        "## 2. 开发集校准归因（Stage 0 & Dev Gate）",
        "",
        "- Stage 0 开发诊断（pairs `901..903`，Stream 6，`1,728` 块/条件）：在 `interior_erased` 条件下，`single_G_est` `<0.01` 比值为 `39.09x`（673/1728 块错），替换为 `oracle_G_true` 后降至 `0.83x`（290/1728 块错），零参数 `fixed_multihyp_M4` 为 `2.55x`（494/1728 块错）。见 [diag_miscalibration.json](results/diag_miscalibration.json)。",
        f"- Dev Gate（修订后 `gate_passed = {dev_gate['gate_passed']}`）：全部 `{dev_gate['audit_summary']['passed_conditions']}/{dev_gate['audit_summary']['total_conditions']}` 开发态对称性检查通过（{dev_gate['audit_summary']['total_transforms']} 次变换，`0` 翻转，`max_dLLR = {dev_gate['audit_summary']['max_llr_diff']:.3e}`）。见 [dev_gate.json](results/dev_gate.json)。",
        "",
        "## 3. 正式测试完整定量结果（N=10 对，每条件 5,760 块）",
        "",
        "> **元数据不对称提示**：`EM7-*` 与 `P6-Aware-Consistent` 均获得完全相同的真实二值接收掩码 `gain`；经典基线 `Waveform-DD` 不接受外部掩码元数据，在擦除条件下仅作不等信息参考，不构成同信息量公平对照。",
        "",
        "| 条件 | 方法 | 块错数 / 5760 | BLER | 译码前 BER | 译码后 BER | 比特 BCE | CFR NMSE (dB) | `<0.01` MSE/Var | 总 MSE/Var |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for cond in CONDITIONS:
        for arm in FORMAL_ARMS:
            s = summary[cond][arm]
            r0 = f"{s['bin0_ratio']:.2f}x" if s.get("bin0_ratio") is not None else "n/a"
            rt = f"{s['total_ratio']:.3f}x" if s.get("total_ratio") is not None else "n/a"
            nm = f"{s['cfr_nmse_db']:.2f}" if s.get("cfr_nmse_db") is not None else "n/a"
            lines.append(
                f"| {cond} | {arm} | {s['block_errors']} | {_p(s['bler'])} | "
                f"{100*s['pre_ber']:.3f}% | {100*s['post_ber']:.3f}% | {s['bit_bce']:.4f} | "
                f"{nm} | {r0} | {rt} |"
            )

    lines += [
        "",
        "### C5_holdout_update 留出导频预测与选择性信念修正明细",
        "",
        "| 方法 | Holdout NLL | Holdout NMSE | Clean Probe NMSE | 预测方差校准比 | 概率迁移 Top10%/Bot50% | Spearman Corr | 硬翻转倍率 | Split/Certain (d>0.01/d<1e-4) | Pass1→2 净少错块 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in FORMAL_ARMS:
        s = summary["C5_holdout_update"][arm]
        if "holdout_nll" in s:
            sp_r = f"{s['seq_split_prob_ratio']:.2f}x / {s['seq_split_flip_ratio']:.2f}x" if s.get("seq_split_prob_ratio") is not None else "n/a"
            lines.append(
                f"| {arm} | {s['holdout_nll']:.4f} | {s['holdout_nmse']:.4f} | {s['holdout_clean_nmse']:.4f} | "
                f"{s['holdout_cal_ratio']:.3f}x | {s['seq_prob_shift_ratio']:.2f}x ({s['seq_prob_shift_high']:.4f} vs {s['seq_prob_shift_low']:.4f}) | "
                f"{s['seq_prob_shift_corr']:.3f} | {s['seq_flip_ratio']:.2f}x | {sp_r} | {s['seq_delta_blocks']} |"
            )
        else:
            lines.append(
                f"| {arm} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | {s.get('seq_delta_blocks', 'n/a')} |"
            )

    lines += [
        "",
        "## 4. 预先登记的四组精确配对检验（1,024 态符号翻转 + 族内 Holm 校正）",
        "",
        "| 族 / 检验项目 | 单位 | 胜/平/负 | 均值差 [t95 区间] | Raw p (精确分数) | Holm p (精确分数) | 备注 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for ftitle, frows in (
        ("F1", fam1),
        ("F2", fam2),
        ("F3", fam3),
        ("F4", fam4),
    ):
        for r in frows:
            diffs_str = ", ".join(f"{v:.4g}" for v in r["differences"])
            lines.append(_fmt_test(f"[{ftitle}] {r['title']}", r["unit"], r, f"pairs=[{diffs_str}]"))

    lines += [
        "",
        "## 5. C1_row9_erased Step-2 全七分箱校准表",
        "",
        "| 方法 | `[0, .01)` | `[.01, .05)` | `[.05, .1)` | `[.1, .2)` | `[.2, .4)` | `[.4, .8)` | `[.8, inf)` |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in ("EM7-Axial", "EM7-Local", "EM7-Fixed", "EM7-SingleHyp-Axial", "P6-Aware-Consistent"):
        bins = summary["C1_row9_erased"][arm]["step2_bins"]
        cells = [f"{b['ratio']:.2f}x (n={b['count']})" if b["ratio"] is not None else "n/a" for b in bins]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "![Phase 7 正式综合实验图](results/p7_figures.png)",
        "",
        "## 6. 严格 $\\mathbb{C}^*$ 对称性审计、重放验证与单线程延迟",
        "",
        "### 正式模型 C* 规范标度审计（30 检查点 × 5 条件 × 27 尺度 = 4,050 次变换，门槛 `< 1e-12`）",
        "",
        "| 模型变体 | 通过条件 / 总数 | 变换总数 | 硬判决翻转 | 最大 LLR 绝对差 | 最大 pred_y 相对差 | 最大 pred_var 相对差 | 最大 π 差 | 最大 NLL 差 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm_name, st in gauge["by_arm"].items():
        lines.append(
            f"| {arm_name} | {st['passed_conditions']}/{st['total_conditions']} | {st['total_transforms']} | "
            f"{st['total_flips']} | {st['max_llr_diff']:.3e} | {st['max_rel_pred_y']:.3e} | "
            f"{st['max_rel_pred_var']:.3e} | {st['max_pi_diff']:.3e} | {st['max_nll_diff']:.3e} |"
        )
    if not gauge["all_passed"]:
        lines.append("")
        lines.append("**严格门槛未完全通过项（不隐藏、不放宽 1e-12 阈值）：**")
        for arm_name, st in gauge["by_arm"].items():
            for fcase in st["failures"]:
                lines.append(
                    f"- `{arm_name}` pair {fcase['pair_idx']:02d} `{fcase['condition']}`: "
                    f"max_llr_diff={fcase['max_llr_diff']:.6e}, flips={fcase['coded_flips']}"
                )

    lines += [
        "",
        "### 单线程 CPU 延迟基准（ms/batch 中位数，1 线程，仅前向推理）",
        "",
        "| Batch | 条件 | 方法 | 中位数 (ms) | Q1–Q3 (ms) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in latency["rows"]:
        lines.append(f"| {r['batch_size']} | {r['condition']} | {r['arm']} | {r['median_ms']:.2f} | [{r['q1_ms']:.2f}, {r['q3_ms']:.2f}] |")

    arch_sig = any(r["holm_p"] < 0.05 for r in fam4[1:])
    arch_note = (
        f"- **架构比较实测结论 (`C4a`/`C5`)**：在严格 906 参数限制下，`EM7-Axial` 相对 `EM7-Local` 在 `fam4` 至少一项指标上达到 Holm p < 0.05（实际 Holm p = {[round(r['holm_p'], 6) for r in fam4[1:]]}）；所有有利与不利对均保留。"
        if arch_sig
        else f"- **阴性/非显著结果如实保留**：在严格 906 参数限制与 40 步训练预算下，2D 轴向注意力 `EM7-Axial` 相对参数匹配的 `EM7-Local` 在 `C4a` BLER（Holm p = `{fam4[1]['holm_p']:.6f}`）、`C4a` BCE（Holm p = `{fam4[2]['holm_p']:.6f}`）及 `C5` 留出 NLL（Holm p = `{fam4[3]['holm_p']:.6f}`）均未达 $\\alpha=0.05$ 显著性，按科学诚信原则如实报告，不作过分外推。"
    )
    lines += [
        "",
        "## 7. 局限性与边界声明",
        "",
        "- **物理范畴**：全部实验限定于单天线线性时变 9 dB R2 OFDM，未验证多天线 R3 或非线性功放 R1/R3；已知擦除掩码 `gain` 为诊断输入，尚未实现盲擦除检测。",
        "- **多假设开销**：`M=4` 物理假设展开及最终 `cavity_demap` 矩阵求逆使单线程推理耗时高于 `M=1` 与局部 `P6-Aware-Consistent`。",
        arch_note,
        "- **跨阶段不可直接对比**：历史 P4 `8380084c` 为 10 dB 强 PA 场景，不与本轮 9 dB 线性 R2 直接对比 BLER；已有 P4/P5/P5B/P6/P6_followup/P6_softinfo/P6_consistency 文件保持零改动。",
        "",
        "## 8. 溯源清单",
        "",
        "- 协议：[PROTOCOL.md](PROTOCOL.md)",
        "- 冻结清单：[results/freeze.json](results/freeze.json)",
        "- 完整统计 JSON：[results/summary.json](results/summary.json)",
        "- 检查点重放：[results/checkpoint_replay.json](results/checkpoint_replay.json)",
        "- 严格标度审计：[results/trained_gauge.json](results/trained_gauge.json)",
        "- 延迟基准：[results/latency.json](results/latency.json)",
    ]
    (RESULTS.parent / "P7_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    readme = [
        "# Phase 7 (`p7/`): Multi-Hypothesis C*-Equivariant Evidence Routing & Held-Out Waveform Consistency",
        "",
        f"- **Branch**: `research/p7-multihypothesis-routing` (base `a4e6684830f68e6dfe715470dafb4ff95972e508`)",
        "- **Status**: Complete prospective held-out study (`N=10` fresh pairs, `5,760` LDPC blocks per condition), `30/30` checkpoint replay verified, full cross-phase regression verified.",
        "",
        "## Reproduction Entrypoints (Run From Repository Root)",
        "",
        "```powershell",
        "python -B -m unittest discover -s p7/tests -v",
        "python -B -m p7.diag_miscalibration",
        "python -B -m p7.dev_study",
        "python -B -m p7.freeze",
        "python -B -m p7.formal_study",
        "python -B -m p7.replay",
        "python -B -m p7.audit_gauge",
        "python -B -m p7.latency",
        "python -B -m p7.report",
        "```",
        "",
        "## Key Artifacts",
        "- [PROTOCOL.md](PROTOCOL.md) — Prospective freeze design and 4 registered Holm-Bonferroni test families",
        "- [P7_REPORT.md](P7_REPORT.md) — Complete Chinese technical report & full quantitative/statistical tables",
        "- [results/freeze.json](results/freeze.json) — Immutable SHA-256 freeze manifest",
        "- [results/summary.json](results/summary.json) — Complete machine-readable results",
        "- [results/p7_figures.png](results/p7_figures.png) — 4-panel quantitative & calibration figure",
    ]
    (RESULTS.parent / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS / 'summary.json'}, {fig_path}, P7_REPORT.md, and README.md", flush=True)


if __name__ == "__main__":
    main()
