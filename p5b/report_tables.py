"""Refresh tables in the P5-B report from the checked machine-readable evidence."""
from pathlib import Path
from statistics import fmean

from p5b.statistics import LABEL, OUT, load


REPORT = Path(__file__).with_name("P5B_REPORT.md")


def table(head, rows):
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in
                     [head, ["---"]*len(head), *rows])


def replace_table(text, name, value):
    start, end = f"<!-- {name} -->", f"<!-- end {name} -->"
    a, rest = text.split(start)
    _, b = rest.split(end)
    return a + start + "\n\n" + value + "\n\n" + end + b


def rates(row, field, denom):
    return f"{row[field]:,}/{row[denom]:,} ({100*row[field]/row[denom]:.2f}%)"


def main():
    summary = load(OUT / "summary.json")
    text = REPORT.read_text(encoding="utf-8")
    for regime in (1,2,3):
        rows = []
        for r in summary["aggregates"]:
            if r["regime"] == regime:
                rows.append([LABEL[r["method"]], rates(r, "bit_errors", "bits"),
                             rates(r, "info_errors", "info_bits"), rates(r, "block_errors", "blocks"),
                             f"{r['nmse_db']:.3f}", r["params"], f"{r['b1_ms']:.2f} / {r['b64_ms']:.2f}"])
        text = replace_table(text, f"r{regime}", table(
            ["Receiver", "Pre-decode errors / bits", "Decoded errors / info bits", "Block errors / blocks",
             "CFR NMSE (dB)", "Trainable real", "B1 / B64 (ms)"], rows))
    for family in ("primary", "ablation"):
        rows = [[f"R{r['regime']}", LABEL[r["arm"]], LABEL[r["comparison"]], f"{r['effect']:+.4f}",
                 f"{r['wins']}/{r['ties']}/{r['losses']}", f"{r['p_raw']:.6f}", f"{r['p_holm']:.6f}"]
                for r in summary["comparisons"][family]]
        text = replace_table(text, family, table(["Regime", "Arm", "Comparator", "Gain (pp)", "W/T/L",
                                                  "Exact p", "Holm p"], rows))
    probes = summary["kernel"]
    probe_table = []
    for pair in [1,2,3,4,5,None]:
        sets = [[r for r in probes if r["method"] == m and (pair is None or r["pair"] == pair)]
                for m in ("pure","parity")]
        stats = [fmean(r[k] for r in group) for group in sets for k in ("initial","trained","trained_shuffled")]
        stats.append(fmean(r["p5_trained"] for r in sets[0]))
        probe_table.append([pair or "Mean", *[f"{x:.5f}" for x in stats]])
    text = replace_table(text, "probe", table(["Pair", "Pure initial", "Pure trained", "Pure shuffled",
        "Parity initial", "Parity trained", "Parity shuffled", "P5 trained"], probe_table))
    blocks = load(OUT / "interventions.json")["pooled"]
    interventions = [("without_radial","Cubic and quintic PA"), ("without_phase_precision","Phase and precision"),
                     ("without_spectral_ici","Adjacent-carrier ICI"), ("without_direct_channel","Complex channel"),
                     ("all_off","All learned outputs")]
    changes = [[title, *[f"{r['block_error_change']:+d} ({r['bler_change_pp']:+.3f} pp)"
                         for r in blocks if r["intervention"] == key]] for key,title in interventions]
    text = replace_table(text, "interventions", table(["Suppressed at inference", "R1 pure",
                                                      "R2 parity", "R3 parity"], changes))
    curves = [[f"R{r['regime']}", LABEL[r["method"]], *[f"{v:.5f}" for v in r["validation_bce"]],
               f"{r['final_lower']}/{r['pairs']}"] for r in summary["validation"]]
    text = replace_table(text, "training", table(["Regime", "Arm", "0", "16", "32", "48", "64",
                                                  "Pairs improving"], curves))
    REPORT.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
