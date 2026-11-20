"""Aggregate formal Phase 8 Manifold results, 4-family Holm-Bonferroni tests, scaling ladder, figures, report & README."""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from p6 import experiment as ex
from p7.statistics import exact_sign_test, holm_adjust
from p8_manifold.freeze import FORMAL_ARMS, FORMAL_PAIRS, RESULTS, verify_freeze
from p8_manifold.offgrid_sim import OFFGRID_CONDITIONS


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
        bler=blk_err / max(blks, 1),
        bit_errors=bit_err,
        bits=bits,
        pre_ber=bit_err / max(bits, 1),
        info_errors=info_err,
        info_bits=info_bits,
        post_ber=info_err / max(info_bits, 1),
        bit_bce=bce_sum / max(bit_cnt, 1),
        cfr_nmse=ch_err / max(ch_eng, 1e-300),
        cfr_nmse_db=(10.0 * math.log10(ch_err / ch_eng)) if (ch_err > 0 and ch_eng > 0) else None,
    )

    if "cal" in rows[0]:
        bins_agg = []
        for idx in range(7):
            cnt = sum(r["cal"]["bins"][idx]["count"] for r in rows)
            sv = sum(r["cal"]["bins"][idx]["sum_predicted_variance"] for r in rows)
            se = sum(r["cal"]["bins"][idx]["sum_squared_error"] for r in rows)
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
        tot_cnt = sum(r["cal"]["total"]["count"] for r in rows)
        tot_sv = sum(r["cal"]["total"]["sum_predicted_variance"] for r in rows)
        tot_se = sum(r["cal"]["total"]["sum_squared_error"] for r in rows)
        entry["bins"] = bins_agg
        entry["bin0_count"] = bins_agg[0]["count"]
        entry["bin0_ratio"] = bins_agg[0]["ratio"]
        entry["total_ratio"] = (tot_se / tot_sv) if tot_sv > 0 else None
    else:
        b0_v = sum(r["cal_stats"]["bin0_sum_var"] for r in rows)
        b0_e = sum(r["cal_stats"]["bin0_sum_mse"] for r in rows)
        t_v = sum(r["cal_stats"]["tot_sum_var"] for r in rows)
        t_e = sum(r["cal_stats"]["tot_sum_mse"] for r in rows)
        entry["bin0_count"] = sum(r["cal_stats"]["bin0_count"] for r in rows)
        entry["bin0_ratio"] = (b0_e / b0_v) if b0_v > 0 else None
        entry["total_ratio"] = (t_e / t_v) if t_v > 0 else None

    if any("blind_weights" in r for r in rows):
        cw = sum(r["blind_weights"]["clean_w_sum"] for r in rows)
        cc = sum(r["blind_weights"]["clean_count"] for r in rows)
        bw = sum(r["blind_weights"]["corrupt_w_sum"] for r in rows)
        bc = sum(r["blind_weights"]["corrupt_count"] for r in rows)
        c_m = cw / max(cc, 1)
        b_m = bw / max(bc, 1)
        entry["blind_clean_w"] = c_m
        entry["blind_corrupt_w"] = b_m
        entry["blind_rejection_ratio"] = c_m / max(b_m, 1e-12)

    if any("holdout" in r for r in rows):
        h_cnt = sum(r["holdout"]["count"] for r in rows)
        entry["holdout_nll"] = sum(r["holdout"]["nll"] * r["holdout"]["count"] for r in rows) / max(h_cnt, 1)
        h_mse = sum(r["holdout"]["mse_sum"] for r in rows)
        h_eng = sum(r["holdout"]["energy_sum"] for r in rows)
        h_var = sum(r["holdout"]["var_sum"] for r in rows)
        entry["holdout_nmse"] = h_mse / max(h_eng, 1e-300)
        entry["holdout_cal_ratio"] = h_mse / max(h_var, 1e-300)
        entry["holdout_var_aleatoric_frac"] = sum(r["holdout"]["var_aleatoric_sum"] for r in rows) / max(h_var, 1e-300)
        entry["holdout_var_epistemic_frac"] = sum(r["holdout"]["var_epistemic_sum"] for r in rows) / max(h_var, 1e-300)
        entry["holdout_var_between_frac"] = sum(r["holdout"]["var_between_sum"] for r in rows) / max(h_var, 1e-300)

    if any("sequential" in r for r in rows):
        p1 = sum(r["sequential"]["pass1_block_errors"] for r in rows)
        p2 = sum(r["sequential"]["pass2_block_errors"] for r in rows)
        blks_s = sum(r["sequential"]["blocks"] for r in rows)
        entry["seq_pass1_blocks"] = p1
        entry["seq_pass2_blocks"] = p2
        entry["seq_pass1_bler"] = p1 / max(blks_s, 1)
        entry["seq_pass2_bler"] = p2 / max(blks_s, 1)
        entry["seq_delta_blocks"] = p1 - p2
        entry["seq_delta_bler"] = (p1 - p2) / max(blks_s, 1)
        entry["seq_delta_bce"] = sum(r["sequential"]["delta_bce"] for r in rows) / len(rows)
        if "top25_prob_shift" in rows[0]["sequential"]:
            tp = sum(r["sequential"]["top25_prob_shift"] for r in rows) / len(rows)
            bp = sum(r["sequential"]["bot25_prob_shift"] for r in rows) / len(rows)
            tf = sum(r["sequential"]["top25_flip_rate"] for r in rows) / len(rows)
            bf = sum(r["sequential"]["bot25_flip_rate"] for r in rows) / len(rows)
            entry["seq_top25_prob_shift"] = tp
            entry["seq_bot25_prob_shift"] = bp
            entry["seq_prob_shift_ratio"] = tp / max(bp, 1e-12)
            entry["seq_top25_flip_rate"] = tf
            entry["seq_bot25_flip_rate"] = bf
            entry["seq_flip_ratio"] = tf / max(bf, 1e-12)
    return entry


def _run_family(tests):
    """Run exact 1,024-state paired sign-flip tests and apply exact rational Holm-Bonferroni within family."""
    rows = []
    for name, diffs in tests:
        st = exact_sign_test(diffs)
        st["name"] = name
        rows.append(st)
    adjusted = holm_adjust(rows)
    for r in adjusted:
        r["significant_05"] = (r["holm_p"] < 0.05)
    return adjusted


def _make_figures(summary, scaling_data):
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.2), dpi=160)

    # Panel A: Coded LDPC BLER (%) across O0..O3
    ax = axes[0, 0]
    conds = list(OFFGRID_CONDITIONS)
    labels = ["O0\nRandom Sparse", "O1\nRoving Island", "O2\nBlind Corrupt", "O3\nHoldout Seq"]
    arm_styles = [
        ("EM8-Transport", "#1f77b4"),
        ("EM8-AbsNoTransport", "#ff7f0e"),
        ("EM8-Local", "#2ca02c"),
        ("EM8-SingleHyp", "#9467bd"),
        ("EM8-Fixed", "#8c564b"),
        ("OffGrid-2D-LMMSE", "#d62728"),
    ]
    w = 0.13
    xs = list(range(len(conds)))
    for i, (arm, color) in enumerate(arm_styles):
        ys = [100.0 * summary[c][arm]["bler"] for c in conds]
        pos = [x + (i - 2.5) * w for x in xs]
        ax.bar(pos, ys, width=w, color=color, label=arm, alpha=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Coded LDPC BLER (%)")
    ax.set_title("A. Off-Grid Coded BLER (N=10 pairs, 5,760 blocks/cond)")
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend(fontsize=7.5, ncol=2)

    # Panel B: Bit BCE & CFR NMSE comparison on O0 & O1 (Transport vs AbsNoTransport vs Local)
    ax = axes[0, 1]
    show_arms = ["EM8-Transport", "EM8-AbsNoTransport", "EM8-Local", "EM8-SingleHyp", "EM8-Fixed"]
    colors_b = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]
    x_b = list(range(len(show_arms)))
    o0_bce = [summary["O0_random_sparse"][a]["bit_bce"] for a in show_arms]
    o1_bce = [summary["O1_roving_island"][a]["bit_bce"] for a in show_arms]
    ax.bar([x - 0.18 for x in x_b], o0_bce, width=0.34, color="#1f77b4", alpha=0.85, label="O0 Random Sparse BCE")
    ax.bar([x + 0.18 for x in x_b], o1_bce, width=0.34, color="#ff7f0e", alpha=0.85, label="O1 Roving Island BCE")
    ax.set_xticks(x_b)
    ax.set_xticklabels(["Transport", "AbsNoTr", "Local", "SingleHyp", "Fixed"], fontsize=8.5)
    ax.set_ylabel("Proper Bit BCE (nats)")
    ax.set_title("B. Construction A Ablation & Locality Gap (Exact 2,063 Params)")
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.legend(fontsize=8)

    # Panel C: O1 Reliability Curve (log-log predicted variance vs actual MSE)
    ax = axes[1, 0]
    ax.plot([2e-3, 1.5], [2e-3, 1.5], "k--", lw=1.2, label="Ideal (MSE = Var)")
    rel_arms = [
        ("EM8-Transport", "D-", "#1f77b4"),
        ("EM8-Local", "s--", "#2ca02c"),
        ("EM8-SingleHyp", "^:", "#9467bd"),
        ("OffGrid-2D-LMMSE", "o-", "#d62728"),
    ]
    for arm, fmt, color in rel_arms:
        bins = summary["O1_roving_island"][arm].get("bins", [])
        bx = [b["mean_pred_var"] for b in bins if b["count"] > 0 and b["mean_pred_var"] and b["mean_sq_err"]]
        by = [b["mean_sq_err"] for b in bins if b["count"] > 0 and b["mean_pred_var"] and b["mean_sq_err"]]
        if bx:
            ax.loglog(bx, by, fmt, color=color, lw=1.8, ms=5.5, label=arm)
    ax.set_xlabel("Mean Predicted Symbol Variance")
    ax.set_ylabel("Actual Symbol MSE")
    ax.set_title("C. O1 Roving Island Symbol Uncertainty Calibration")
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8)

    # Panel D: Scaling-Law Ladder (Small 2.1K -> Medium 12.8K -> Large 40.0K)
    ax = axes[1, 1]
    if scaling_data:
        rungs = ("Small", "Medium", "Large")
        ps = [scaling_data["rungs"][r]["real_parameters"] for r in rungs]
        for cond, ls, tag in (("O0_random_sparse", "-", "O0"), ("O1_roving_island", "--", "O1")):
            tr_y = [100.0 * scaling_data["rungs"][r]["conditions"][cond]["EM8-Transport"]["bler"] for r in rungs]
            lo_y = [100.0 * scaling_data["rungs"][r]["conditions"][cond]["EM8-Local"]["bler"] for r in rungs]
            ax.semilogx(ps, tr_y, "D" + ls, color="#1f77b4", lw=2.0, ms=6.0, label=f"EM8-Transport ({tag})")
            ax.semilogx(ps, lo_y, "s" + ls, color="#2ca02c", lw=2.0, ms=6.0, label=f"EM8-Local ({tag})")
        ax.set_xticks(ps)
        ax.set_xticklabels([f"{r}\n({p:,})" for r, p in zip(rungs, ps)])
        ax.set_ylabel("Coded LDPC BLER (%)")
        ax.set_title("D. Controlled Scaling-Law Ladder (EM8-Transport vs EM8-Local)")
        ax.grid(True, which="both", ls=":", alpha=0.5)
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Scaling ladder pending", ha="center", va="center")

    fig.tight_layout()
    fig_path = RESULTS / "p8_manifold_figures.png"
    fig.savefig(fig_path, dpi=160)
    plt.close(fig)
    return fig_path


def _fmt_ratio(v):
    return f"{v:.2f}x" if v is not None else "n/a"


def main():
    freeze_sha = verify_freeze()
    records = [
        json.loads((RESULTS / "formal" / f"pair{p:02d}" / "record.json").read_text(encoding="utf-8"))
        for p in FORMAL_PAIRS
    ]
    all_rows = [r for rec in records for r in rec["rows"]]

    summary = {}
    by_pair = {}
    for cond in OFFGRID_CONDITIONS:
        summary[cond] = {}
        by_pair[cond] = {}
        for arm in FORMAL_ARMS:
            c_rows = [r for r in all_rows if r["condition"] == cond and r["arm"] == arm]
            summary[cond][arm] = _aggregate_rows(c_rows)
            by_pair[cond][arm] = [
                _aggregate_rows([r for r in rec["rows"] if r["condition"] == cond and r["arm"] == arm])
                for rec in records
            ]

    # 4 Pre-Registered Exact 1,024-State Holm-Bonferroni Families
    fam1 = _run_family([
        ("O1 BLER: Local - Transport", [by_pair["O1_roving_island"]["EM8-Local"][i]["bler"] - by_pair["O1_roving_island"]["EM8-Transport"][i]["bler"] for i in range(10)]),
        ("O1 BCE: Local - Transport", [by_pair["O1_roving_island"]["EM8-Local"][i]["bit_bce"] - by_pair["O1_roving_island"]["EM8-Transport"][i]["bit_bce"] for i in range(10)]),
        ("O0 BLER: Local - Transport", [by_pair["O0_random_sparse"]["EM8-Local"][i]["bler"] - by_pair["O0_random_sparse"]["EM8-Transport"][i]["bler"] for i in range(10)]),
        ("O0 BCE: Local - Transport", [by_pair["O0_random_sparse"]["EM8-Local"][i]["bit_bce"] - by_pair["O0_random_sparse"]["EM8-Transport"][i]["bit_bce"] for i in range(10)]),
    ])
    fam2 = _run_family([
        ("O1 BLER: AbsNoTransport - Transport", [by_pair["O1_roving_island"]["EM8-AbsNoTransport"][i]["bler"] - by_pair["O1_roving_island"]["EM8-Transport"][i]["bler"] for i in range(10)]),
        ("O1 BCE: AbsNoTransport - Transport", [by_pair["O1_roving_island"]["EM8-AbsNoTransport"][i]["bit_bce"] - by_pair["O1_roving_island"]["EM8-Transport"][i]["bit_bce"] for i in range(10)]),
        ("O0 BLER: AbsNoTransport - Transport", [by_pair["O0_random_sparse"]["EM8-AbsNoTransport"][i]["bler"] - by_pair["O0_random_sparse"]["EM8-Transport"][i]["bler"] for i in range(10)]),
        ("O0 BCE: AbsNoTransport - Transport", [by_pair["O0_random_sparse"]["EM8-AbsNoTransport"][i]["bit_bce"] - by_pair["O0_random_sparse"]["EM8-Transport"][i]["bit_bce"] for i in range(10)]),
    ])
    fam3 = _run_family([
        ("O2 Blind BLER: LMMSE - Transport", [by_pair["O2_blind_burst_erasure"]["OffGrid-2D-LMMSE"][i]["bler"] - by_pair["O2_blind_burst_erasure"]["EM8-Transport"][i]["bler"] for i in range(10)]),
        ("O2 Blind BCE: SingleHyp - Transport", [by_pair["O2_blind_burst_erasure"]["EM8-SingleHyp"][i]["bit_bce"] - by_pair["O2_blind_burst_erasure"]["EM8-Transport"][i]["bit_bce"] for i in range(10)]),
        ("O1 BCE: SingleHyp - Transport", [by_pair["O1_roving_island"]["EM8-SingleHyp"][i]["bit_bce"] - by_pair["O1_roving_island"]["EM8-Transport"][i]["bit_bce"] for i in range(10)]),
    ])
    fam4 = _run_family([
        ("O3 Holdout NLL: SingleHyp - Transport", [by_pair["O3_holdout_sequential"]["EM8-SingleHyp"][i]["holdout_nll"] - by_pair["O3_holdout_sequential"]["EM8-Transport"][i]["holdout_nll"] for i in range(10)]),
        ("O3 Assimilate BLER Gain (P1 - P2): Transport", [by_pair["O3_holdout_sequential"]["EM8-Transport"][i]["seq_delta_bler"] for i in range(10)]),
        ("O3 Assimilate BCE Gain (P1 - P2): Transport", [by_pair["O3_holdout_sequential"]["EM8-Transport"][i]["seq_delta_bce"] for i in range(10)]),
        ("O3 Selective Revision (Top25 - Bot25): Transport", [by_pair["O3_holdout_sequential"]["EM8-Transport"][i]["seq_top25_prob_shift"] - by_pair["O3_holdout_sequential"]["EM8-Transport"][i]["seq_bot25_prob_shift"] for i in range(10)]),
    ])
    families = dict(
        family1_attention_vs_local=fam1,
        family2_construction_a_ablation=fam2,
        family3_blind_and_multichart=fam3,
        family4_holdout_and_assimilation=fam4,
    )

    scaling_path = RESULTS / "scaling_ladder.json"
    scaling_data = json.loads(scaling_path.read_text(encoding="utf-8")) if scaling_path.exists() else None
    stage80_path = RESULTS / "stage80_r2_fixes.json"
    stage80_data = json.loads(stage80_path.read_text(encoding="utf-8")) if stage80_path.exists() else None
    replay_path = RESULTS / "checkpoint_replay.json"
    replay_data = json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.exists() else None
    gauge_path = RESULTS / "trained_gauge.json"
    gauge_data = json.loads(gauge_path.read_text(encoding="utf-8")) if gauge_path.exists() else None
    latency_path = RESULTS / "latency.json"
    latency_data = json.loads(latency_path.read_text(encoding="utf-8")) if latency_path.exists() else None

    fig_path = _make_figures(summary, scaling_data)
    out_summary = dict(
        freeze_sha256=freeze_sha,
        conditions=list(OFFGRID_CONDITIONS),
        arms=list(FORMAL_ARMS),
        summary=summary,
        families=families,
        has_scaling_ladder=(scaling_data is not None),
        has_replay=(replay_data is not None),
        has_gauge=(gauge_data is not None),
        has_latency=(latency_data is not None),
    )
    ex.save(RESULTS / "summary.json", out_summary)

    # Generate concise, unvarnished P8_MANIFOLD_REPORT.md
    lines = [
        "# Phase 8 Manifold (`p8_manifold`) Formal Research Report",
        "",
        f"- **Prospective Freeze SHA-256**: `{freeze_sha}`",
        "- **Base Commit**: `4dab1ac9ee15b4f1a7f78ae05486c55a2dc3bd72` (`research/p7-multihypothesis-routing`, inherited files untouched)",
        "- **Sample Size**: `N = 10` fresh paired seeds (`5,760` LDPC blocks / condition; exact `2,063` real params parity across `EM8-Transport`, `EM8-AbsNoTransport`, `EM8-Local`, selected via pre-formal Scaling-Law Ladder)",
        "",
        "## 1. Main Off-Grid Benchmark Results (`O0..O3`, `5,760` blocks/cond)",
        "",
    ]
    for cond in OFFGRID_CONDITIONS:
        lines.append(f"### `{cond}`")
        lines.append("")
        if cond == "O3_holdout_sequential":
            lines.append("| Arm | Block Errors / 5,760 | BLER | Bit BCE | CFR NMSE | `<0.01` Cal | Total Cal | Holdout NLL | H-Cal | P1->P2 dBlk | Sel Prob | Sel Flip |")
            lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        elif cond == "O2_blind_burst_erasure":
            lines.append("| Arm | Block Errors / 5,760 | BLER | Bit BCE | CFR NMSE | `<0.01` Cal | Total Cal | Clean w | Corrupt w | Reject Ratio |")
            lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        else:
            lines.append("| Arm | Block Errors / 5,760 | BLER | Bit BCE | CFR NMSE | `<0.01` Cal | Total Cal |")
            lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for arm in FORMAL_ARMS:
            st = summary[cond][arm]
            base = (
                f"| `{arm}` | {st['block_errors']:,} | {100*st['bler']:.2f}% | {st['bit_bce']:.4f} | "
                f"{st['cfr_nmse']:.4f} | {_fmt_ratio(st['bin0_ratio'])} | {_fmt_ratio(st['total_ratio'])}"
            )
            if cond == "O3_holdout_sequential":
                hnll = f"{st['holdout_nll']:.3f}" if "holdout_nll" in st else "n/a"
                hcal = f"{st['holdout_cal_ratio']:.2f}x" if "holdout_cal_ratio" in st else "n/a"
                dblk = f"{st['seq_delta_blocks']:+d}" if "seq_delta_blocks" in st else "n/a"
                sp = f"{st['seq_prob_shift_ratio']:.2f}x" if "seq_prob_shift_ratio" in st else "n/a"
                sf = f"{st['seq_flip_ratio']:.2f}x" if "seq_flip_ratio" in st else "n/a"
                base += f" | {hnll} | {hcal} | {dblk} | {sp} | {sf} |"
            elif cond == "O2_blind_burst_erasure":
                wc = f"{st['blind_clean_w']:.3f}" if "blind_clean_w" in st else "n/a"
                wb = f"{st['blind_corrupt_w']:.4f}" if "blind_corrupt_w" in st else "n/a"
                rr = f"{st['blind_rejection_ratio']:.1f}x" if "blind_rejection_ratio" in st else "n/a"
                base += f" | {wc} | {wb} | {rr} |"
            else:
                base += " |"
            lines.append(base)
        lines.append("")

    lines.append("## 2. Pre-Registered Exact 1,024-State Holm-Bonferroni Inference")
    lines.append("")
    lines.append("| Family | Contrast | Mean Diff | 95% t-CI | Wins/10 | Exact p | Holm p | Sig (<0.05) |")
    lines.append("| :--- | :--- | ---: | :--- | ---: | ---: | ---: | :---: |")
    for fam_name, rows_f in families.items():
        for r in rows_f:
            ci = f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
            lines.append(
                f"| `{fam_name}` | {r['name']} | {r['mean_diff']:+.4f} | {ci} | {r['wins']}/{r['n']} | "
                f"{r['raw_exact']} ({r['raw_p']:.5f}) | {r['holm_exact']} ({r['holm_p']:.5f}) | {'YES' if r['significant_05'] else 'NO'} |"
            )
    lines.append("")

    if scaling_data:
        lines.append("## 3. Controlled 3-Rung Scaling-Law Ladder (`Small -> Medium -> Large`)")
        lines.append("")
        lines.append("| Condition | Rung | Real Params | `EM8-Transport` BLER / BCE | `EM8-Local` BLER / BCE | Gap (`Local - Transport`) |")
        lines.append("| :--- | :--- | ---: | :--- | :--- | :--- |")
        for cond in ("O0_random_sparse", "O1_roving_island", "O2_blind_burst_erasure", "O3_holdout_sequential"):
            for rn in ("Small", "Medium", "Large"):
                p_cnt = scaling_data["rungs"][rn]["real_parameters"]
                tr = scaling_data["rungs"][rn]["conditions"][cond]["EM8-Transport"]
                lo = scaling_data["rungs"][rn]["conditions"][cond]["EM8-Local"]
                lines.append(
                    f"| `{cond}` | {rn} | {p_cnt:,} | {100*tr['bler']:.2f}% / {tr['bce']:.4f} | "
                    f"{100*lo['bler']:.2f}% / {lo['bce']:.4f} | {100*(lo['bler']-tr['bler']):+.2f} pp / {lo['bce']-tr['bce']:+.4f} |"
                )
        lines.append("")

    lines.append("## 4. Independent Audits, Replay & Latency")
    lines.append("")
    if replay_data:
        lines.append(f"- **Checkpoint Replay**: `{replay_data['passed_checkpoints']}/{replay_data['total_checkpoints']}` exact match (`passed={replay_data['passed']}`).")
    if gauge_data:
        lines.append(
            f"- **Formal $\\mathbb{{C}}^*$ Gauge Audit**: rel `<1e-12` = `{gauge_data['passed_rel_conditions']}/{gauge_data['total_conditions']}`, "
            f"strict abs `<1e-12` = `{gauge_data['passed_abs_conditions']}/{gauge_data['total_conditions']}` "
            f"across `{gauge_data['total_transforms']:,}` transforms, coded bit flips = `{gauge_data['total_flips']}`."
        )
    if latency_data:
        lines.append("")
        lines.append("| Batch | Condition | Arm | Median (ms) | IQR [Q1, Q3] (ms) |")
        lines.append("| ---: | :--- | :--- | ---: | :--- |")
        for r in latency_data["rows"]:
            lines.append(f"| {r['batch_size']} | `{r['condition']}` | `{r['arm']}` | {r['median_ms']:.2f} | [{r['q1_ms']:.2f}, {r['q3_ms']:.2f}] |")
    lines.append("")
    lines.append(f"![Phase 8 Manifold Figures](results/{fig_path.name})")
    lines.append("")

    report_path = RESULTS.parent / "P8_MANIFOLD_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    readme_lines = [
        "# `p8_manifold` — Off-Grid Coherent Parallel-Transport Wave Attention & Multi-Chart Manifold Receiver",
        "",
        "Self-contained Phase 8 Manifold research track (`research/p8-coherent-transport-manifold`), cleanly isolated from prior phases (`p4..p7`) and separate classical multi-frame branches (`p8/`, `p8c/`).",
        "",
        "## Quick Reproduction",
        "```bash",
        "python -B -m unittest discover -s p8_manifold/tests -v",
        "python -B -m p8_manifold.dev_study --steps 40 --workers 3",
        "python -B -m p8_manifold.scaling_ladder --steps 40 --workers 4",
        "python -B -m p8_manifold.freeze",
        "python -B -m p8_manifold.formal_study --workers 4",
        "python -B -m p8_manifold.replay",
        "python -B -m p8_manifold.audit_gauge --workers 4",
        "python -B -m p8_manifold.latency",
        "python -B -m p8_manifold.report",
        "```",
        "",
        "See [PROTOCOL.md](PROTOCOL.md) and [P8_MANIFOLD_REPORT.md](P8_MANIFOLD_REPORT.md) for complete protocol, statistics, and audit receipts.",
        "",
    ]
    (RESULTS.parent / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    print(f"Wrote summary.json, {fig_path.name}, P8_MANIFOLD_REPORT.md, and README.md", flush=True)


if __name__ == "__main__":
    main()
