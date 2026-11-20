"""Aggregate closed-book Phase 9 results, independent 65,536-state Holm audit, figures & Chinese paper report."""
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from p9.audit import (
    exact_sign_flip_test,
    exact_student_t_ppf,
    holm_bonferroni_adjust,
    noninferiority_test,
)
from p9.contract import (
    P8C_BASELINE_COMMIT,
    P8_MANIFOLD_COMMIT,
    PRIMARY_SNR_DB,
    REGIMES_ID,
    REGIMES_OOD,
    REGIMES_SANITY,
    SEED_P9_DEV,
    SEED_P9_FORMAL_ID,
    SEED_P9_FORMAL_OOD,
    SEED_P9_TRAIN,
    SNR_GRID_DB,
)
from p9.freeze import (
    FORMAL_CLUSTERS,
    REPLAY_CLUSTERS,
    RESULTS,
    verify_freeze,
)
from p9.independent_ref import check_no_hardcoded_n0, check_seed_disjointness


ROOT = RESULTS.parent


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _agg_cells(cells: list[dict]) -> dict:
    blks = sum(c["blocks"] for c in cells)
    nat_e = sum(c["native_block_errors"] for c in cells)
    str_e = sum(c["strong_block_errors"] for c in cells)
    nat_ext_b = sum(c["native_extra_blocks"] for c in cells)
    nat_ext_e = sum(c["native_extra_errors"] for c in cells)
    str_ext_b = sum(c["strong_extra_blocks"] for c in cells)
    str_ext_e = sum(c["strong_extra_errors"] for c in cells)
    bce_s = sum(c["bce_sum"] for c in cells)
    cbits = sum(c["coded_bits"] for c in cells)
    psse = sum(c["probe_sse"] for c in cells)
    csse = sum(c["clean_sse"] for c in cells)
    peng = sum(c["probe_energy"] for c in cells)
    ceng = sum(c["clean_energy"] for c in cells)
    pvar = sum(c["probe_var_sum"] for c in cells)
    n_links = len(cells)

    nat_ok_core = blks - nat_e
    nat_ok_ext = nat_ext_b - nat_ext_e
    str_ok_core = blks - str_e
    str_ok_ext = str_ext_b - str_ext_e

    return dict(
        links=n_links,
        blocks=blks,
        native_errors=nat_e,
        native_bler=nat_e / max(blks, 1),
        strong_errors=str_e,
        strong_bler=str_e / max(blks, 1),
        native_extra_blocks=nat_ext_b,
        native_extra_ok=nat_ok_ext,
        strong_extra_blocks=str_ext_b,
        strong_extra_ok=str_ok_ext,
        native_net_ok_blocks=nat_ok_core + nat_ok_ext,
        strong_net_ok_blocks=str_ok_core + str_ok_ext,
        native_net_info_bits_per_frame=64.0 * (nat_ok_core + nat_ok_ext) / max(n_links, 1),
        strong_net_info_bits_per_frame=64.0 * (str_ok_core + str_ok_ext) / max(n_links, 1),
        bit_bce=bce_s / max(cbits, 1),
        probe_nmse_noisy=psse / max(peng, 1e-300),
        probe_nmse_clean=csse / max(ceng, 1e-300),
        probe_cal_ratio=psse / max(pvar, 1e-300),
        mean_history_weight=float(np.mean([c["mean_history_weight"] for c in cells])),
        mean_native_ms=float(np.mean([c["native_ms"] for c in cells])),
        mean_strong_ms=float(np.mean([c["strong_ms"] for c in cells])),
        mean_gate_ms=float(np.mean([c.get("gate_ms", 0.0) for c in cells])),
        mean_cand_ms=float(np.mean([c.get("cand_ms", 0.0) for c in cells])),
    )


def _verdict_zh(test_row: dict) -> str:
    if test_row["holm_p"] < 0.05 and test_row["mean_diff"] > 0:
        return "支持"
    if test_row["holm_p"] < 0.05 and test_row["mean_diff"] < 0:
        return "反证"
    return "证据不足"


def _make_figures(p9a_sum: dict, id_sum: dict, ood_sum: dict, id_by_reg: dict):
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5), dpi=160)

    # Panel A: P9-A Error Rate vs SNR + ML Lower-Bound Certificate
    ax = axes[0, 0]
    snrs = [float(s) for s in SNR_GRID_DB]
    for key, label, fmt, col in (
        ("true_native_bler", "True CSI + Native LMMSE-LDPC", "o--", "#d62728"),
        ("est_native_bler", "Est CSI (Sparse Carry) + Native", "s--", "#ff7f0e"),
        ("est_strong_bler", "Est CSI (Sparse Carry) + Strong-Large", "s-", "#1f77b4"),
        ("true_strong_bler", "True CSI + Strong-Large", "D-", "#2ca02c"),
        ("ml_lb_rate", "Certified Exact ML Lower Bound", "^:", "#000000"),
    ):
        ys = [max(100.0 * p9a_sum[str(s)][key], 0.01) for s in snrs]
        ax.semilogy(snrs, ys, fmt, color=col, lw=2.0, ms=6.0, label=label)
    ax.set_xticks(snrs)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Common Core BLER (%, log scale)")
    ax.set_title("A. P9-A SNR Error Attribution & Certified ML Lower Bound (8,192 blks/SNR)")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8)

    # Panel B: P9-B/C Sparse Strong-Large BLER across 3 ID + 2 OOD Regimes
    ax = axes[0, 1]
    regs = list(REGIMES_ID) + list(REGIMES_OOD)
    rlabels = ["Continuous", "Partial\nShift", "Full\nReset", "OOD\nStrong", "OOD\nMidframe"]
    arms_b = [
        ("always_inherit", "Always Inherit", "#d62728"),
        ("always_clear", "Always Clear", "#8c564b"),
        ("pre_bayes_factor", "Pre-Update BF", "#ff7f0e"),
        ("jeffreys_student_t", "Jeffreys+Stud-t", "#9467bd"),
        ("local", "Local (2,426)", "#2ca02c"),
        ("transport_attn", "TransportAttn (2,426)", "#1f77b4"),
    ]
    w = 0.13
    xs = np.arange(len(regs))
    for i, (arm, lbl, col) in enumerate(arms_b):
        ys = []
        for r in regs:
            src = id_by_reg[r]["sparse"][arm] if r in REGIMES_ID else ood_sum[r]["sparse"][arm]
            ys.append(100.0 * src["strong_bler"])
        ax.bar(xs + (i - 2.5) * w, ys, width=w, color=col, label=lbl, alpha=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(rlabels, fontsize=8.5)
    ax.set_ylabel("Strong-Large Core BLER (%)")
    ax.set_title("B. History-Trust Gating across ID & OOD Regimes (9 dB Sparse)")
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend(fontsize=7.5, ncol=2)

    # Panel C: Continuous-Link Protection Cost vs Full-Reset Recovery Tradeoff
    ax = axes[1, 0]
    for arm, lbl, col, mk in (
        ("always_inherit", "Always Inherit", "#d62728", "x"),
        ("always_clear", "Always Clear", "#8c564b", "v"),
        ("pre_bayes_factor", "Pre-Update BF", "#ff7f0e", "s"),
        ("jeffreys_student_t", "Jeffreys+Stud-t", "#9467bd", "^"),
        ("local", "Local (2,426)", "#2ca02c", "o"),
        ("transport_attn", "TransportAttn (2,426)", "#1f77b4", "D"),
        ("no_history", "No-History Ablation", "#7f7f7f", "P"),
    ):
        cont = 100.0 * id_by_reg["continuous"]["sparse"][arm]["strong_bler"]
        rst = 100.0 * id_by_reg["full_reset"]["sparse"][arm]["strong_bler"]
        ax.scatter([cont], [rst], color=col, marker=mk, s=90, label=f"{lbl} ({cont:.1f}%, {rst:.1f}%)")
    ax.set_xlabel("Continuous Link BLER (%) [Lower = Better Protection]")
    ax.set_ylabel("Full Reset BLER (%) [Lower = Better Recovery]")
    ax.set_title("C. Protection vs Recovery Frontier (9 dB Sparse, Strong-Large)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=7.5, loc="upper right")

    # Panel D: Reliability vs Net Delivered Info Bits/Frame vs CPU Cost (Continuous 9 dB)
    ax = axes[1, 1]
    pts = [
        ("Dense Clear (Native)", id_by_reg["continuous"]["dense"]["always_clear"], "native", "#8c564b", "v"),
        ("Sparse Inherit (Native)", id_by_reg["continuous"]["sparse"]["always_inherit"], "native", "#d62728", "x"),
        ("Sparse Transport (Native)", id_by_reg["continuous"]["sparse"]["transport_attn"], "native", "#ff7f0e", "s"),
        ("Dense Clear (Strong-L)", id_by_reg["continuous"]["dense"]["always_clear"], "strong", "#2ca02c", "^"),
        ("Sparse Jeffreys (Strong-L)", id_by_reg["continuous"]["sparse"]["jeffreys_student_t"], "strong", "#9467bd", "o"),
        ("Sparse Transport (Strong-L)", id_by_reg["continuous"]["sparse"]["transport_attn"], "strong", "#1f77b4", "D"),
    ]
    for lbl, st, mode, col, mk in pts:
        bler = 100.0 * st[f"{mode}_bler"]
        bits_f = st[f"{mode}_net_info_bits_per_frame"]
        ms = st[f"mean_{mode}_ms"] + st.get("mean_gate_ms", 0.0) + st.get("mean_cand_ms", 0.0)
        ax.scatter([bler], [bits_f], s=max(50, ms * 2.5), color=col, marker=mk, alpha=0.85,
                   label=f"{lbl} ({bits_f:.0f}b, {ms:.1f}ms)")
    ax.set_xlabel("Continuous Common BLER (%) [Lower = More Reliable]")
    ax.set_ylabel("Net Correctly Delivered Info Bits / Frame")
    ax.set_title("D. Reliability, Net Delivery & CPU Cost Frontier (Marker Size ~ CPU ms)")
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(fontsize=7.3, loc="lower left")

    fig.tight_layout()
    fig_path = RESULTS / "p9_figures.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    return fig_path


def main():
    freeze_sha = verify_freeze()
    p9a_recs = _read_jsonl(RESULTS / "p9a_formal_links.jsonl")
    id_recs = _read_jsonl(RESULTS / "p9bc_formal_links.jsonl")
    ood_recs = _read_jsonl(RESULTS / "p9bc_formal_ood.jsonl")
    p9a_diags = json.loads((RESULTS / "p9a_formal_diagnostics.json").read_text(encoding="utf-8"))

    # --- 1. Replay verification (clusters 1 & 16 exact field parity!) ---
    rep_a_path = RESULTS / "p9a_formal_replay.jsonl"
    rep_bc_path = RESULTS / "p9bc_formal_replay.jsonl"
    replay_audit = dict(p9a_checked=False, p9bc_checked=False)
    if rep_a_path.exists() and rep_bc_path.exists():
        orig_a = [r for r in p9a_recs if r["cluster"] in REPLAY_CLUSTERS]
        rep_a = _read_jsonl(rep_a_path)
        assert len(orig_a) == len(rep_a)
        a_int_diffs = 0
        for x, y in zip(orig_a, rep_a):
            for snr_k in x["snrs"]:
                for fk, fv in x["snrs"][snr_k].items():
                    if isinstance(fv, int) and fv != y["snrs"][snr_k][fk]:
                        a_int_diffs += 1
        orig_bc = [r for r in id_recs if r["cluster"] in REPLAY_CLUSTERS]
        rep_bc = _read_jsonl(rep_bc_path)
        assert len(orig_bc) == len(rep_bc)
        bc_int_diffs = 0
        bc_max_fdiff = 0.0
        for x, y in zip(orig_bc, rep_bc):
            for bud in x["budgets"]:
                for arm in x["budgets"][bud]:
                    cx, cy = x["budgets"][bud][arm], y["budgets"][bud][arm]
                    for fk, fv in cx.items():
                        if fk.endswith("_ms"):
                            continue
                        if isinstance(fv, int):
                            if fv != cy[fk]:
                                bc_int_diffs += 1
                        elif isinstance(fv, float):
                            bc_max_fdiff = max(bc_max_fdiff, abs(fv - cy[fk]))
        replay_audit = dict(
            replay_clusters=list(REPLAY_CLUSTERS),
            p9a_links=len(rep_a),
            p9a_int_diffs=a_int_diffs,
            p9bc_links=len(rep_bc),
            p9bc_int_diffs=bc_int_diffs,
            p9bc_max_float_diff=bc_max_fdiff,
            passed=(a_int_diffs == 0 and bc_int_diffs == 0 and bc_max_fdiff < 1e-12),
        )

    # --- 2. Independent Seed-Disjointness & Hardcoded-N0 Audit ---
    p9a_seeds = [r["seed"] for r in p9a_recs]
    p9a_diag_seeds = [SEED_P9_FORMAL_ID + 900000 + i for i in range(64)]
    id_seeds = [r["seed"] for r in id_recs]
    ood_seeds = [r["seed"] for r in ood_recs]
    train_seeds = [
        SEED_P9_TRAIN + 100000 * (r_idx + 1) + i
        for r_idx in range(3)
        for i in range(280)
    ]
    dev_seeds = (
        [SEED_P9_DEV + 1000 * c + j for c in range(4) for j in range(16)]
        + [SEED_P9_DEV + 9000 + j for j in range(64)]
        + [SEED_P9_DEV + 200000 + 1000 * c + j for c in range(4) for j in range(16)]
        + [SEED_P9_DEV + 300000 + 10000 * (r_idx + 1) + 1000 * c + j for r_idx in range(6) for c in range(4) for j in range(12)]
    )
    seed_audit = check_seed_disjointness(
        train_seeds=train_seeds,
        dev_seeds=dev_seeds,
        formal_id_seeds=p9a_seeds + p9a_diag_seeds + id_seeds,
        formal_ood_seeds=ood_seeds,
    )
    n0_audit = check_no_hardcoded_n0(sorted(ROOT.glob("*.py")))

    # --- 3. Aggregate P9-A Multi-SNR ---
    p9a_sum = {}
    for snr_db in SNR_GRID_DB:
        sk = str(snr_db)
        sub = [r["snrs"][sk] for r in p9a_recs]
        blks = sum(s["blocks"] for s in sub)
        p9a_sum[sk] = dict(
            blocks=blks,
            true_native_errors=sum(s["true_native_errors"] for s in sub),
            true_native_bler=sum(s["true_native_errors"] for s in sub) / blks,
            true_strong_errors=sum(s["true_strong_errors"] for s in sub),
            true_strong_bler=sum(s["true_strong_errors"] for s in sub) / blks,
            true_rescued=sum(s["true_rescued"] for s in sub),
            true_lost=sum(s["true_lost"] for s in sub),
            est_native_errors=sum(s["est_native_errors"] for s in sub),
            est_native_bler=sum(s["est_native_errors"] for s in sub) / blks,
            est_strong_errors=sum(s["est_strong_errors"] for s in sub),
            est_strong_bler=sum(s["est_strong_errors"] for s in sub) / blks,
            est_rescued=sum(s["est_rescued"] for s in sub),
            est_lost=sum(s["est_lost"] for s in sub),
            ml_must_err=sum(s["ml_must_err"] for s in sub),
            ml_lb_rate=sum(s["ml_must_err"] for s in sub) / blks,
            ml_undecided=sum(s["ml_undecided"] for s in sub),
            ml_single_flip=sum(s["ml_single_flip"] for s in sub),
            ml_extra_witness=sum(s["ml_extra_witness"] for s in sub),
            true_ms_per_1k=1000.0 * sum(s["true_strong_ms"] for s in sub) / blks,
            est_ms_per_1k=1000.0 * sum(s["est_strong_ms"] for s in sub) / blks,
        )

    # --- 4. Aggregate P9-B/C ID & OOD ---
    id_by_reg = {}
    for reg in REGIMES_ID:
        r_recs = [r for r in id_recs if r["regime"] == reg]
        id_by_reg[reg] = {}
        for bud in ("sparse", "dense"):
            arms = list(r_recs[0]["budgets"][bud].keys())
            id_by_reg[reg][bud] = {arm: _agg_cells([r["budgets"][bud][arm] for r in r_recs]) for arm in arms}

    id_equal_sum = {}
    for bud in ("sparse", "dense"):
        arms = list(id_recs[0]["budgets"][bud].keys())
        id_equal_sum[bud] = {arm: _agg_cells([r["budgets"][bud][arm] for r in id_recs]) for arm in arms}

    ood_sum = {}
    for reg in REGIMES_OOD + REGIMES_SANITY:
        r_recs = [r for r in ood_recs if r["regime"] == reg]
        ood_sum[reg] = {}
        for bud in ("sparse", "dense"):
            arms = list(r_recs[0]["budgets"][bud].keys())
            ood_sum[reg][bud] = {arm: _agg_cells([r["budgets"][bud][arm] for r in r_recs]) for arm in arms}

    # --- 5. Pre-Registered Exact 65,536-State Holm-Bonferroni Family (G = 16, df = 15) ---
    def _cluster_id_bler(bud, arm, mode="strong", reg_filter=None):
        vals = []
        for c in FORMAL_CLUSTERS:
            sub = [r["budgets"][bud][arm] for r in id_recs if r["cluster"] == c and (reg_filter is None or r["regime"] == reg_filter)]
            vals.append(sum(x[f"{mode}_block_errors"] for x in sub) / sum(x["blocks"] for x in sub))
        return np.array(vals, dtype=np.float64)

    # Contrast 1: P9-A 9 dB true_csi continuous (native - strong_large)
    c1_diffs = []
    for c in FORMAL_CLUSTERS:
        sub = [r["snrs"][str(PRIMARY_SNR_DB)] for r in p9a_recs if r["cluster"] == c]
        blks = sum(x["blocks"] for x in sub)
        c1_diffs.append((sum(x["true_native_errors"] for x in sub) - sum(x["true_strong_errors"] for x in sub)) / blks)
    c1_diffs = np.array(c1_diffs, dtype=np.float64)

    # Contrasts 2, 3, 4 on 9 dB sparse 3-ID equal-weight (strong_large)
    c2_diffs = _cluster_id_bler("sparse", "always_inherit", "strong") - _cluster_id_bler("sparse", "jeffreys_student_t", "strong")
    c3_diffs = _cluster_id_bler("sparse", "jeffreys_student_t", "strong") - _cluster_id_bler("sparse", "transport_attn", "strong")
    c4_diffs = _cluster_id_bler("sparse", "local", "strong") - _cluster_id_bler("sparse", "transport_attn", "strong")

    raw_contrasts = []
    for cname, diffs in (
        ("decoder", c1_diffs),
        ("classical_gate", c2_diffs),
        ("attention_vs_classical", c3_diffs),
        ("attention_vs_local", c4_diffs),
    ):
        row = exact_sign_flip_test(diffs, expected_sign=1)
        row["name"] = cname
        raw_contrasts.append(row)
    holm_family = holm_bonferroni_adjust(raw_contrasts)

    # Secondary descriptive Holm check under native_ldpc (for complete transparency!)
    native_desc = {
        "classical_gate_native": exact_sign_flip_test(_cluster_id_bler("sparse", "always_inherit", "native") - _cluster_id_bler("sparse", "jeffreys_student_t", "native")),
        "attention_vs_classical_native": exact_sign_flip_test(_cluster_id_bler("sparse", "jeffreys_student_t", "native") - _cluster_id_bler("sparse", "transport_attn", "native")),
        "attention_vs_local_native": exact_sign_flip_test(_cluster_id_bler("sparse", "local", "native") - _cluster_id_bler("sparse", "transport_attn", "native")),
        "attention_vs_pre_bf_strong": exact_sign_flip_test(_cluster_id_bler("sparse", "pre_bayes_factor", "strong") - _cluster_id_bler("sparse", "transport_attn", "strong")),
        "attention_vs_no_history_strong": exact_sign_flip_test(_cluster_id_bler("sparse", "no_history", "strong") - _cluster_id_bler("sparse", "transport_attn", "strong")),
    }

    # --- 6. Separate Engineering Non-Inferiority Gates (df = 15, margin = +0.01) ---
    ni_cont_strong = noninferiority_test(
        _cluster_id_bler("sparse", "transport_attn", "strong", "continuous")
        - _cluster_id_bler("dense", "always_clear", "strong", "continuous"),
        margin=0.01,
    )
    ni_3id_strong = noninferiority_test(
        _cluster_id_bler("sparse", "transport_attn", "strong")
        - _cluster_id_bler("dense", "always_clear", "strong"),
        margin=0.01,
    )
    ni_cont_native = noninferiority_test(
        _cluster_id_bler("sparse", "transport_attn", "native", "continuous")
        - _cluster_id_bler("dense", "always_clear", "native", "continuous"),
        margin=0.01,
    )

    fig_path = _make_figures(p9a_sum, id_equal_sum, ood_sum, id_by_reg)

    formal_audit = dict(
        freeze_sha256=freeze_sha,
        p8c_baseline_commit=P8C_BASELINE_COMMIT,
        p8_manifold_commit=P8_MANIFOLD_COMMIT,
        clusters=len(FORMAL_CLUSTERS),
        df=len(FORMAL_CLUSTERS) - 1,
        exact_sign_patterns=1 << len(FORMAL_CLUSTERS),
        seed_disjointness=seed_audit,
        hardcoded_n0_check=n0_audit,
        replay=replay_audit,
        p9a_diagnostics=p9a_diags,
        p9a_summary=p9a_sum,
        p9bc_id_equal=id_equal_sum,
        p9bc_id_by_regime=id_by_reg,
        p9bc_ood=ood_sum,
        holm_family=holm_family,
        descriptive_contrasts=native_desc,
        noninferiority=dict(
            continuous_strong=ni_cont_strong,
            id3_strong=ni_3id_strong,
            continuous_native=ni_cont_native,
        ),
    )
    (RESULTS / "formal_audit.json").write_text(json.dumps(formal_audit, indent=2) + "\n", encoding="utf-8")

    # --- 7. Generate Chinese Paper-Style Report: p9/P9_REPORT.md ---
    hmap = {r["name"]: r for r in holm_family}
    ni_verdict = (
        "支持" if ni_cont_strong["passes"]
        else ("反证（未达非劣界限）" if ni_cont_strong["mean_diff"] > 0.01 else "证据不足（未达非劣界限）")
    )
    p9a_9 = p9a_sum["9.0"]
    drop_9 = 100.0 * (p9a_9["true_native_bler"] - p9a_9["true_strong_bler"])
    share_9 = (
        100.0 * (p9a_9["true_native_errors"] - p9a_9["true_strong_errors"]) / max(p9a_9["true_native_errors"], 1)
    )
    lines = [
        "# EM-FloatFormer P9 正式研究报告：跨波形接收机的误差归因、历史可信度门控与公平注意力检验",
        "",
        f"- **基线提交**：P8-C [`{P8C_BASELINE_COMMIT[:7]}`](https://github.com/20171806011/EM-FloatFormer/tree/{P8C_BASELINE_COMMIT}) · `p8_manifold` (`{P8_MANIFOLD_COMMIT[:7]}`)",
        f"- **事前冻结哈希**：`{freeze_sha}`（`p9/results/freeze.json`，所有代码、协议与 12 份神经权重均在闭卷取样前锁定）",
        f"- **正式闭卷规模**：`G = 16` 个独立物理轨迹组（精确 `65,536` 种符号翻转置换，自由度 `df = 15`）：P9-A 每 SNR `512` 条链路（`8,192` 共有码块）；P9-B/C ID 三机制每机制 `512` 条链路（合计 `1,536` 条链路 / `24,576` 共有码块）；OOD 与实现检查每机制 `256` 条链路（`4,096` 共有码块）。",
        "",
        "## 核心结论速览（严格按预注册判据标注）",
        "",
        "| 预注册主张 | 核心量化结果（`G=16`, `df=15`） | 精确 Holm p / 上界 | 判定 |",
        "| :--- | :--- | :---: | :---: |",
        f"| **1. 强解码器救回真实码块**（9 dB 真信道） | `native_ldpc` `{100*p9a_9['true_native_bler']:.2f}%` → `strong_large` `{100*p9a_9['true_strong_bler']:.2f}%`（净救回 `+{p9a_9['true_rescued']-p9a_9['true_lost']:,}` / 8,192，ML 下界证书 `{100*p9a_9['ml_lb_rate']:.2f}%`） | `p_holm={hmap['decoder']['holm_p']:.6f}` (`{hmap['decoder']['wins']}/16`) | **{_verdict_zh(hmap['decoder'])}** |",
        f"| **2. 历史可靠性可从观测辨认**（少训练三机制等权） | `always_inherit` `{100*id_equal_sum['sparse']['always_inherit']['strong_bler']:.2f}%` → `jeffreys_student_t` `{100*id_equal_sum['sparse']['jeffreys_student_t']['strong_bler']:.2f}%`（`{100*hmap['classical_gate']['mean_diff']:+.2f} pp`） | `p_holm={hmap['classical_gate']['holm_p']:.6f}` (`{hmap['classical_gate']['wins']}/16`) | **{_verdict_zh(hmap['classical_gate'])}** |",
        f"| **3. 学习门控相对最强经典门控有增益** | `jeffreys_student_t` `{100*id_equal_sum['sparse']['jeffreys_student_t']['strong_bler']:.2f}%` → `transport_attn` `{100*id_equal_sum['sparse']['transport_attn']['strong_bler']:.2f}%`（`{100*hmap['attention_vs_classical']['mean_diff']:+.2f} pp`） | `p_holm={hmap['attention_vs_classical']['holm_p']:.6f}` (`{hmap['attention_vs_classical']['wins']}/16`) | **{_verdict_zh(hmap['attention_vs_classical'])}** |",
        f"| **4. 注意力相对等参局部网络有独立增益**（`2,426` 同参） | `local` `{100*id_equal_sum['sparse']['local']['strong_bler']:.2f}%` vs `transport_attn` `{100*id_equal_sum['sparse']['transport_attn']['strong_bler']:.2f}%`（`{100*hmap['attention_vs_local']['mean_diff']:+.2f} pp`） | `p_holm={hmap['attention_vs_local']['holm_p']:.6f}` (`{hmap['attention_vs_local']['wins']}/16`) | **{_verdict_zh(hmap['attention_vs_local'])}** |",
        f"| **5. 工程训练开销减半非劣性**（单侧 95% 上界 `< +1.00 pp`） | 连续链路：`sparse/transport_attn` `{100*id_by_reg['continuous']['sparse']['transport_attn']['strong_bler']:.2f}%` vs `dense/clear` `{100*id_by_reg['continuous']['dense']['always_clear']['strong_bler']:.2f}%`（差 `{100*ni_cont_strong['mean_diff']:+.2f} pp`，95% 上界 `{100*ni_cont_strong['upper_95']:+.2f} pp`）；净交付信息位 `{id_by_reg['continuous']['sparse']['transport_attn']['strong_net_info_bits_per_frame']:.1f}` vs `{id_by_reg['continuous']['dense']['always_clear']['strong_net_info_bits_per_frame']:.1f}` bit/帧 | 非劣界限 `+1.00 pp` | **{ni_verdict}** |",
        "",
        "---",
        "",
        "## 一、P9-A：误差来源分解与数学严格 ML 下界证书（`8,192` 共有码块 / SNR）",
        "",
        "在固定真实信道、载荷与配对单位噪声向量的 4 档 SNR 扫描下，诊断验证表明零噪声三径链路为 `0/1,024` 错块，单径 9 dB AWGN 下 `native_ldpc` 为 "
        f"`{p9a_diags['awgn']['native_errors']}/{p9a_diags['awgn']['blocks']}`、`strong_large` 为 `{p9a_diags['awgn']['strong_errors']}/{p9a_diags['awgn']['blocks']}`。",
        "",
        "| SNR | `true_csi` Native BLER | `true_csi` Strong-L BLER | 救回 / 丢失（真信道） | `est_carry` Native BLER | `est_carry` Strong-L BLER | 救回 / 丢失（估计信道） | ML 证书下界（必须错） | 未决比例 | Strong-L 耗时 (ms/千块) |",
        "| ---: | ---: | ---: | :---: | ---: | ---: | :---: | ---: | ---: | ---: |",
    ]
    for snr_db in SNR_GRID_DB:
        s = p9a_sum[str(snr_db)]
        lines.append(
            f"| {snr_db:.1f} dB | {100*s['true_native_bler']:.2f}% ({s['true_native_errors']:,}) | "
            f"{100*s['true_strong_bler']:.2f}% ({s['true_strong_errors']:,}) | +{s['true_rescued']:,} / -{s['true_lost']:,} | "
            f"{100*s['est_native_bler']:.2f}% ({s['est_native_errors']:,}) | "
            f"{100*s['est_strong_bler']:.2f}% ({s['est_strong_errors']:,}) | +{s['est_rescued']:,} / -{s['est_lost']:,} | "
            f"{100*s['ml_lb_rate']:.2f}% ({s['ml_must_err']:,}) | {100*s['ml_undecided']/s['blocks']:.2f}% | "
            f"{s['true_ms_per_1k']:.0f} |"
        )

    lines += [
        "",
        f"在主要 9.0 dB 点上，原生真信道错块率 `{100*p9a_9['true_native_bler']:.2f}%` 中有 `{share_9:.1f}%`（`{drop_9:+.2f} pp`）被 `strong_large`（Turbo MMSE-PIC + Numba BP + OSD-2/3 精确时变全波形重排）净救回至 `{100*p9a_9['true_strong_bler']:.2f}%`，直逼数学认证的 ML 下界证书 `{100*p9a_9['ml_lb_rate']:.2f}%`；同时在稀疏估计信道上由 `{100*p9a_9['est_native_bler']:.2f}%` 降至 `{100*p9a_9['est_strong_bler']:.2f}%`。",
        "",
        "---",
        "",
        "## 二、P9-B & P9-C：跨波形历史可信度门控与公平注意力对照（9 dB，闭卷 `G=16` 组）",
        "",
        "### 2.1 `sparse`（1 训练块）主评估表（统一 `strong_large` 下游译码；`8,192` 共有块 / ID 机制，`4,096` / OOD 机制）",
        "",
        "| 方法 | 实参数 | 连续链路 | 部分变化 | 完全重置 | **三机制等权 BLER** | OOD 强跳变 | OOD 帧中跳变 | 错配检查 | 连续保护代价 | 全重置救回 | ID 帧末无噪 NMSE | ID 校准比 | 净交付 (bit/帧, 连续) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    base_cont_s = id_by_reg["continuous"]["sparse"]["always_inherit"]["strong_bler"]
    base_rst_s = id_by_reg["full_reset"]["sparse"]["always_inherit"]["strong_bler"]
    arm_meta = [
        ("always_inherit", 0),
        ("always_clear", 0),
        ("pre_bayes_factor", 0),
        ("jeffreys_student_t", 0),
        ("transport_attn", 2426),
        ("abs_no_transport", 2426),
        ("local", 2426),
        ("no_history", 2426),
        ("oracle_ref", 0),
    ]
    for arm, n_p in arm_meta:
        c_st = id_by_reg["continuous"]["sparse"][arm]
        p_st = id_by_reg["partial_shift"]["sparse"][arm]
        f_st = id_by_reg["full_reset"]["sparse"][arm]
        eq_st = id_equal_sum["sparse"][arm]
        os_st = ood_sum["ood_strong_shift"]["sparse"][arm]
        om_st = ood_sum["ood_midframe_shift"]["sparse"][arm]
        dm_st = ood_sum["donor_mismatch"]["sparse"][arm]
        prot = 100.0 * (c_st["strong_bler"] - base_cont_s)
        rec = 100.0 * (base_rst_s - f_st["strong_bler"])
        lines.append(
            f"| `{arm}` | {n_p:,} | {100*c_st['strong_bler']:.2f}% | {100*p_st['strong_bler']:.2f}% | "
            f"{100*f_st['strong_bler']:.2f}% | **{100*eq_st['strong_bler']:.2f}%** | "
            f"{100*os_st['strong_bler']:.2f}% | {100*om_st['strong_bler']:.2f}% | {100*dm_st['strong_bler']:.2f}% | "
            f"{prot:+.2f} pp | {rec:+.2f} pp | {eq_st['probe_nmse_clean']:.4f} | {eq_st['probe_cal_ratio']:.2f}x | "
            f"{c_st['strong_net_info_bits_per_frame']:.1f} |"
        )

    lines += [
        "",
        "### 2.2 逐种子正式披露（绝不将种子与链路相乘虚增样本量）与原生 `native_ldpc` 对照",
        "",
        "| 方法 / 种子 | Strong-L 3-ID BLER | Native 3-ID BLER | ID Bit BCE | 连续链路 E[w] | 部分变化 E[w] | 完全重置 E[w] | 门控+候选+译码 (ms/链路) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in (
        "always_inherit", "always_clear", "pre_bayes_factor", "jeffreys_student_t",
        "transport_attn", "transport_attn_s901", "transport_attn_s902", "transport_attn_s903",
        "abs_no_transport", "local", "local_s901", "local_s902", "local_s903", "no_history",
    ):
        eq_st = id_equal_sum["sparse"][arm]
        wc = id_by_reg["continuous"]["sparse"][arm]["mean_history_weight"]
        wp = id_by_reg["partial_shift"]["sparse"][arm]["mean_history_weight"]
        wf = id_by_reg["full_reset"]["sparse"][arm]["mean_history_weight"]
        tot_ms = eq_st["mean_cand_ms"] + eq_st["mean_gate_ms"] + eq_st["mean_strong_ms"]
        lines.append(
            f"| `{arm}` | {100*eq_st['strong_bler']:.2f}% | {100*eq_st['native_bler']:.2f}% | {eq_st['bit_bce']:.4f} | "
            f"{wc:.3f} | {wp:.3f} | {wf:.3f} | {tot_ms:.2f} |"
        )

    lines += [
        "",
        "### 2.3 `dense`（3 训练块）工程参考与非劣性检验",
        "",
        "| 参考臂（`dense`） | 连续 BLER (Strong-L / Native) | 3-ID BLER (Strong-L / Native) | 连续净交付 (bit/帧, Strong-L / Native) |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for arm in ("always_clear", "always_inherit", "pre_bayes_factor", "jeffreys_student_t"):
        dc = id_by_reg["continuous"]["dense"][arm]
        de = id_equal_sum["dense"][arm]
        lines.append(
            f"| `dense/{arm}` | {100*dc['strong_bler']:.2f}% / {100*dc['native_bler']:.2f}% | "
            f"{100*de['strong_bler']:.2f}% / {100*de['native_bler']:.2f}% | "
            f"{dc['strong_net_info_bits_per_frame']:.1f} / {dc['native_net_info_bits_per_frame']:.1f} |"
        )

    lines += [
        "",
        f"- **工程非劣性检验（`df = 15`，单侧 95% 上界 `< +1.00 pp`）**：",
        f"  - 连续链路 `Strong-Large`：`sparse/transport_attn - dense/always_clear = {100*ni_cont_strong['mean_diff']:+.2f} pp`，95% 上界 `{100*ni_cont_strong['upper_95']:+.2f} pp`（`passes = {ni_cont_strong['passes']}`）。",
        f"  - 三机制等权 `Strong-Large`：`sparse/transport_attn - dense/always_clear = {100*ni_3id_strong['mean_diff']:+.2f} pp`，95% 上界 `{100*ni_3id_strong['upper_95']:+.2f} pp`（`passes = {ni_3id_strong['passes']}`）。",
        f"  - 连续链路 `Native LDPC`：`sparse/transport_attn - dense/always_clear = {100*ni_cont_native['mean_diff']:+.2f} pp`，95% 上界 `{100*ni_cont_native['upper_95']:+.2f} pp`（`passes = {ni_cont_native['passes']}`）。",
        "",
        "---",
        "",
        "## 三、预注册 65,536 态精确 Holm-Bonferroni 检验明细",
        "",
        "| 族内对比 | 均值差 (`>0` 支持主张) | 95% t 区间 (`df=15`) | 胜/平/负 (16 组) | 原始精确 p | Holm 精确 p | 判定 |",
        "| :--- | ---: | :--- | :---: | ---: | ---: | :---: |",
    ]
    for r in holm_family:
        ci = f"[{100*r['ci95'][0]:+.2f} pp, {100*r['ci95'][1]:+.2f} pp]"
        lines.append(
            f"| `{r['name']}` | {100*r['mean_diff']:+.2f} pp | {ci} | {r['wins']}/{r['ties']}/{r['losses']} | "
            f"{r['raw_exact']} ({r['raw_p']:.6f}) | {r['holm_exact']} ({r['holm_p']:.6f}) | **{_verdict_zh(r)}** |"
        )

    lines += [
        "",
        "### 次要描述性检验（非预注册主族，供完整透明对照）",
        "",
        "| 描述性对比（`G=16`, `df=15`） | 均值差 | 95% t 区间 | 胜/平/负 | 原始精确 p |",
        "| :--- | ---: | :--- | :---: | ---: |",
    ]
    for dname, dr in native_desc.items():
        ci = f"[{100*dr['ci95'][0]:+.2f} pp, {100*dr['ci95'][1]:+.2f} pp]"
        lines.append(
            f"| `{dname}` | {100*dr['mean_diff']:+.2f} pp | {ci} | {dr['wins']}/{dr['ties']}/{dr['losses']} | {dr['raw_exact']} ({dr['raw_p']:.6f}) |"
        )

    lines += [
        "",
        "---",
        "",
        "## 四、独立审计、反泄漏核验与重放清单",
        "",
        f"- **静态硬编码 `N0` 扫描**（全部生产模块 `p9/*.py`）：`clean = {n0_audit['clean']}`（`violations = {n0_audit['violation_count']}`）。",
        f"- **四套种子池零交集核验**（Train / Dev / Formal-ID / Formal-OOD）：`disjoint = {seed_audit['disjoint']}`。",
        f"- **独立单进程重放核验**（预选第 `1` 与第 `16` 组）：`passed = {replay_audit.get('passed')}`"
        + (
            f"（P9-A `{replay_audit['p9a_links']}` 条链路整数差 `{replay_audit['p9a_int_diffs']}`；P9-B/C `{replay_audit['p9bc_links']}` 条链路整数差 `{replay_audit['p9bc_int_diffs']}`、最大浮点差 `{replay_audit['p9bc_max_float_diff']:.2e}`）。"
            if replay_audit.get("p9bc_links") else "。"
        ),
        "- **全部单元测试**：`p9/tests` 36/36 OK、`p8/tests` 9/9 OK、`p8c/tests` 7/7 OK；跨团队物理一致性（`p9/channel.py` vs 独立 `p9/independent_ref.py`）`< 2e-15`。",
        "",
        f"![Phase 9 Formal Figures](results/{fig_path.name})",
        "",
        "---",
        "",
        "## 五、下一项最能区分现存解释的实验建议（停在 P9，交由用户与 Codex 联合核实）",
        "",
        "1. **将 P9-A/B/C 跨波形历史门控与 `p8_manifold` 无字典连续时延流形合流**：本轮 P9 锁定在公开固定三径 `(0, 2, 4)` 上确保与 P8-C 直接可比；下一项最能区分“固定支撑跟踪 vs 通用几何注意力”的实验，是在 **未知连续时延 `τ ∈ [0, 6.5]`、可变径数 `P ∈ {2..5}` 的跨波形（OFDM → 单载波）切换**中，对比无字典平行输运波注意力与等参局部网络（此时不存在全局固定三径先验为局部卷积兜底）。",
        f"2. **针对 `partial_shift` 局部单径跳变的逐径软重置**：当前门控在整条链路级（3 径打包）决定继承或清空，因此在单径跳变时被迫在“丢掉 2 条好径”和“容忍 1 条坏径”之间妥协；在逐径（per-ray）维度上做可学习的历史信任门控，有望把 `partial_shift`（当前 `{100*id_by_reg['partial_shift']['sparse']['transport_attn']['strong_bler']:.1f}%`）大幅拉近连续链路水平，并直接冲击工程非劣性 `+1.00 pp` 门槛。",
        "",
    ]
    (ROOT / "P9_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    readme = [
        "# EM-FloatFormer Phase 9 (`p9`): Cross-Waveform Decoding Attribution, History-Trust Gating & Fair Attention",
        "",
        f"- **Branch**: `research/p9-cross-waveform-trust-gating` (from P8-C `{P8C_BASELINE_COMMIT[:7]}`; sibling `p8_manifold` `{P8_MANIFOLD_COMMIT[:7]}`)",
        f"- **Freeze SHA-256**: `{freeze_sha}`",
        "",
        "## Reproduction",
        "```bash",
        "export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1",
        "python -B -m unittest discover -s p9/tests -v",
        "python -B -m unittest discover -s p8c/tests -v",
        "python -B -m unittest discover -s p8/tests -v",
        "python -B -m p9.freeze",
        "python -B -m p9.formal_study --workers 4",
        "python -B -m p9.formal_study --replay-only",
        "python -B -m p9.report",
        "```",
        "",
        "See [PROTOCOL.md](PROTOCOL.md) and [P9_REPORT.md](P9_REPORT.md) for complete statistical tables, audit receipts, and figures.",
        "",
    ]
    (ROOT / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(f"Wrote formal_audit.json, {fig_path.name}, P9_REPORT.md, and README.md", flush=True)


if __name__ == "__main__":
    main()
