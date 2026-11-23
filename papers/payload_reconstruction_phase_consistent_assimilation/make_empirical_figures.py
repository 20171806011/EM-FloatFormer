"""Regenerate the three manuscript figures from recorded aggregate data."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output-dir", type=Path, default=ROOT / "figures",
                    help="Destination for PNG and PDF figures (default: bundled figures/).")
args = parser.parse_args()
OUT = args.output_dir.expanduser().resolve()
OUT.mkdir(parents=True, exist_ok=True)
E = json.loads((ROOT / "empirical_plot_data.json").read_text(encoding="utf-8"))
STUDY_A = {row["arm"]: row for row in E["p10_main_table"]}
STUDY_B = {(row["budget"], row["arm"]): row for row in E["p11_main_table"]}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10, "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5, "legend.fontsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.linewidth": 0.7, "savefig.facecolor": "white",
})
NAVY, TEAL, ORANGE, SLATE = "#17324D", "#087F8C", "#D17B27", "#6F7F90"
TEXT, GRID = "#263442", "#DCE2E7"
REGIMES = ["continuous", "partial_shift", "full_reset"]
REGIME_LABELS = ["Continuous", "One-path\nshift", "Full\nreset"]

def style(ax, ymax, step):
    ax.set_ylim(0, ymax)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MultipleLocator(step))
    ax.grid(axis="y", color=GRID, linewidth=0.55)
    ax.tick_params(axis="both", length=3, color="#81909D")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#81909D")
    ax.set_ylabel("Block error rate (%)", color=TEXT)

def label_bars(ax, bars, values, offset=1.2):
    for b, v in zip(bars, values):
        ax.text(b.get_x()+b.get_width()/2, v+offset, f"{v:.1f}",
                ha="center", va="bottom", fontsize=8, color=TEXT)

def save(fig, stem):
    fig.savefig(OUT / (stem + ".png"), dpi=220)
    fig.savefig(OUT / (stem + ".pdf"))
    plt.close(fig)

# Figure 1: Separate descriptive panels, all data from the P10 cohort.
fig = plt.figure(figsize=(6.5, 4.05))
gs = fig.add_gridspec(1, 2, width_ratios=[1.88, 1], wspace=0.38,
                      left=0.10, right=0.985, top=0.85, bottom=0.29)
ax, bx = fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1])
x = np.arange(3)
width = 0.245
specs = [
    ("default", "Frozen incumbent", NAVY),
    ("rebuild", "Standalone payload reconstruction", TEAL),
    ("code_gate", "Codeword-validated selection", ORANGE),
]
handles = []
for j, (key, label, color) in enumerate(specs):
    values = [STUDY_A[key][r]["bler_percent"] for r in REGIMES]
    bars = ax.bar(x+(j-1)*width, values, width*0.92, label=label,
                  color=color, edgecolor="white", linewidth=0.4)
    handles.append(bars)
    label_bars(ax, bars, values, 1.8)
    if key == "code_gate":
        # Stagger the nearly equal full-reset annotations without changing bar data.
        ax.texts[-1].set_y(values[-1] + 5.8)
style(ax, 100, 20)
ax.set_xticks(x, REGIME_LABELS)
ax.set_title("(a) Performance across mechanisms", loc="left", pad=9)
ax.legend(handles, [s[1] for s in specs], loc="upper left",
          bbox_to_anchor=(-0.02,-0.24), frameon=False, ncol=1,
          borderaxespad=0, handlelength=1.3, labelspacing=0.4)
counts = [E["p10_rescue"]["pilot_only_rescued"], E["p10_rescue"]["payload_rescued"]]
denom = E["p10_rescue"]["common_failure_blocks"]
vals = [100*c/denom for c in counts]
bars = bx.bar([0,1], vals, width=0.58, color=[SLATE,TEAL],
              edgecolor="white", linewidth=0.4)
style(bx, 72, 20)
bx.set_ylabel("Common failures recovered (%)")
bx.set_xticks([0,1], ["Pilot\nonly", "Payload\nassisted"])
bx.set_title("(b) Recovery\n2,666 common failures", loc="left", pad=9)
for bar,c,v in zip(bars,counts,vals):
    bx.text(bar.get_x()+bar.get_width()/2, v+1.9,
            f"{c:,}\n({v:.2f}%)", ha="center", va="bottom",
            fontsize=8, color=TEXT, linespacing=1.2)
fig.suptitle("Study A · paired two-batch evaluation at 9 dB",
             x=0.10, y=0.98, ha="left", fontsize=10.5, fontweight="semibold", color=TEXT)
save(fig, "figure1_study_a_reconstruction")

# Figure 2: Same-cohort P11 primary comparison.
fig, ax = plt.subplots(figsize=(6.5, 4.0))
fig.subplots_adjust(left=0.105, right=0.985, top=0.86, bottom=0.27)
x = np.arange(3)*1.30
width = 0.235
specs = [
    ("sparse", "p11_chained_wavefront", "Chained reconstruction + assimilation", TEAL),
    ("sparse", "p11_payload_only", "Standalone payload reconstruction", ORANGE),
    ("sparse", "p9_jeffreys_student_t", "Jeffreys / Student-t gate", NAVY),
    ("dense", "always_clear", "Dense-pilot clear reference", SLATE),
]
handles = []
for j,(budget,key,label,color) in enumerate(specs):
    vals = [100*STUDY_B[(budget,key)][r]["bler"] for r in REGIMES]
    bars = ax.bar(x+(j-1.5)*width, vals, width*0.92, label=label,
                  color=color, edgecolor="white", linewidth=0.4)
    handles.append(bars)
    label_bars(ax,bars,vals,1.5)
style(ax,100,20)
ax.set_xticks(x, REGIME_LABELS)
ax.set_xlim(-0.75,3.35)
ax.legend(handles,[s[2] for s in specs],loc="upper center",bbox_to_anchor=(0.5,-0.21),
          frameon=False,ncol=2,columnspacing=1.25,handlelength=1.4,
          borderaxespad=0,labelspacing=0.6)
fig.suptitle("Study B · matched evaluation on 24,576 core blocks at 9 dB",
             x=0.105,y=0.98,ha="left",fontsize=10.5,fontweight="semibold",color=TEXT)
fig.text(0.105,0.905,"8,192 blocks per mechanism; lower is better",fontsize=8.5,color=TEXT)
save(fig, "figure2_study_b_matched_bler")

# Figure 3: Whole-fold asymmetry, with actual target counts and helper locations.
fig, ax = plt.subplots(figsize=(6.5,3.9))
fig.subplots_adjust(left=0.105,right=0.985,top=0.855,bottom=0.30)
x = np.arange(3)
width=0.29
specs=[("fold0", "11-target fold",NAVY),("fold1","5-target fold",TEAL)]
handles=[]
for j,(fold,label,color) in enumerate(specs):
    vals=[]
    for r in REGIMES:
        m=STUDY_B[("sparse","p11_chained_wavefront")][r]
        vals.append(100*m[fold+"_errors"]/m[fold+"_blocks"])
    bars=ax.bar(x+(j-.5)*width,vals,width*0.93,color=color,
                edgecolor="white",linewidth=.4,label=label)
    handles.append(bars)
    label_bars(ax,bars,vals,.5)
style(ax,30,5)
ax.set_xticks(x,REGIME_LABELS)
ax.legend(handles,[s[1] for s in specs],loc="upper center",bbox_to_anchor=(.5,-.22),
          frameon=False,ncol=2,columnspacing=2.0,handlelength=1.5,
          borderaxespad=0)
fig.suptitle("Study B · residual error and held-out fold geometry",
             x=.105,y=.98,ha="left",fontsize=10.5,fontweight="semibold",color=TEXT)
fig.text(.105,.905,"Chained receiver; 5,632 and 2,560 target blocks per mechanism",fontsize=8.5,color=TEXT)
fig.text(.105,.058,"11-target fold: helpers at rows 0 and 9 (both before the pilot at row 10).",fontsize=8,color=TEXT)
fig.text(.105,.022,"5-target fold: helpers at rows 0, 3, 6, 9, 11, 14, 17 and 18 (both sides).",fontsize=8,color=TEXT)
save(fig,"figure3_study_b_fold_geometry")
print("Saved 3 PNG and 3 vector PDF figures.")
