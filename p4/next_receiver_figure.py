"""Plot paired P4-B development measurements from the saved integer counts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


RESULTS = Path(__file__).resolve().parent / "results"


def main():
    chinese_font = next((name for name in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei")
                         if name in font_manager.get_font_names()), None)
    if chinese_font:
        plt.rcParams["font.family"] = chinese_font
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    neural = json.loads((RESULTS / "next_neural_summary.json").read_bytes())["summaries"]
    combined = json.loads((RESULTS / "next_multi_impairment.json").read_bytes())["rows"]
    seeds = (73161, 73162, 73163)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.2, 4.7), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.25, h_pad=0.15, wspace=0.22)
    families = ("fourier_full", "fourier_no_channel", "tap_phase_no_channel", "tap_phase_pilot_only")
    colors = ("#34719A", "#A55530", "#41755E")
    for seed, color, marker in zip(seeds, colors, ("o", "^", "s")):
        ys = []
        for family in families[:3]:
            row = neural[f"{family}_{seed}"]
            ys.append(100 * row["ood_block_errors"] / row["ood_blocks"])
        ax.plot(range(3), ys, color=color, marker=marker, label=str(seed), linewidth=1.6, markersize=6)
    pilot = neural[families[-1]]
    pilot_rate = 100 * pilot["ood_block_errors"] / pilot["ood_blocks"]
    ax.axhline(pilot_rate, linestyle="--", color="#435164", linewidth=1.5, label="逐径相位，仅用导频")
    ax.set_xticks(range(3), ["傅里叶\n完整模型", "傅里叶\n去信道修正", "逐径相位\n去信道修正"])
    ax.set_ylabel("误块率（%）")
    ax.set_ylim(0, 34)
    ax.set_xlim(-0.16, 2.15)
    ax.set_title("三个未见时延分布 · 每个模型 2,496 块", loc="left", fontsize=13, pad=12)
    ax.legend(frameon=False, ncol=2, loc="upper right", fontsize=8)
    methods = ("2D-IDD-PA", "2D-LMMSE", "pilot_only", "73161", "waveform_feedback")
    names = ("2D-IDD-PA", "2D-LMMSE", "逐径相位，仅用导频", "逐径相位，含学习修正", "波形重建与译码反馈")
    blocks = sum(r["slots"] * r["codewords_per_slot"] for r in combined)
    for i, (method, color) in enumerate(zip(methods, ("#9FAFBC", "#9FAFBC", "#637687", "#34719A", "#41755E"))):
        errors = sum(sum(r["methods"][method]["block_errors"]) for r in combined)
        rate = errors * 100 / blocks
        bx.barh(i, rate, height=0.55, color=color)
        bx.text(rate + 0.6, i, str(errors), va="center", fontsize=10)
    bx.set_yticks(range(len(names)), names)
    bx.invert_yaxis()
    bx.set_xlim(0, 48)
    bx.set_xlabel("误块率（%），条末数字为错误块数")
    bx.set_title("S06 多重损伤 · 每个接收机 1,248 块", loc="left", fontsize=13, pad=12)
    for axis, direction in ((ax, "y"), (bx, "x")):
        axis.grid(axis=direction, color="#e1e6e9")
        axis.set_axisbelow(True)
    fig.savefig(RESULTS / "next_receiver_comparison.png", dpi=160, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
