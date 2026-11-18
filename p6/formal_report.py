"""Build the frozen, pair-level formal comparisons and two-cohort report."""

from fractions import Fraction
import math
import statistics

from p6 import experiment as ex
from p6 import formal_study as study
from p6.math_verify import FREEZE, sha


ARMS = ("pure", "local", "parity", "local_parity", "no_gauge")
CLASSICAL = {1: "pilot_cpe_dd", 2: "waveform_dd", 3: "waveform_dd"}
DISPLAY = dict(
    pure="Axial-Pure",
    local="Local-Pure",
    parity="Axial-Parity",
    local_parity="Local-Parity",
    no_gauge="Axial-NoGauge",
    physics="Fixed",
    physics_parity="Fixed+Parity",
    pilot_cpe_dd="Pilot-CPE-DD",
    waveform_dd="Waveform-DD",
    nominal_idd_pa="Nominal IDD-PA",
    nominal_bussgang_lmmse="Nominal Bussgang LMMSE",
    genie_pa="Genie-PA (privileged)",
    lmmse_2d="2D-LMMSE",
    tap_phase_slope="Tap phase slope",
    p5_full="P5 Unrolled EM",
    joint="Historical product",
    parallel_joint="Direct joint",
    spatial="Spatial",
    frequency="Frequency",
    time="Time",
)


def name(key):
    if key.startswith("p5b_"):
        return "P5-B " + key.removeprefix("p5b_").replace("_", " ")
    return DISPLAY.get(key, key.replace("_", " "))


def sign_row(regime, arm, comparator, differences):
    values = list(differences.values())
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Expected a complete finite set of pair contrasts")
    wins = sum(v > 0 for v in values)
    losses = sum(v < 0 for v in values)
    n = wins + losses
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1))
    p = min(Fraction(1), Fraction(2 * tail, 2**n))
    return dict(
        regime=regime,
        arm=arm,
        comparator=comparator,
        wins=wins,
        ties=len(values) - n,
        losses=losses,
        non_tied=n,
        raw_p=float(p),
        raw_exact=str(p),
        by_pair=differences,
        mean_difference=statistics.mean(values),
    )


def holm(rows):
    running = Fraction(0)
    for rank, row in enumerate(sorted(rows, key=lambda x: Fraction(x["raw_exact"]))):
        adjusted = (len(rows) - rank) * Fraction(row["raw_exact"])
        running = max(running, min(Fraction(1), adjusted))
        row.update(
            holm_p=float(running),
            holm_exact=str(running),
            reject_05=running <= Fraction(1, 20),
        )
    return rows


def scores_by_cohort(records):
    out = {}
    for cohort, regimes in study.COHORTS.items():
        out[cohort] = {}
        for r, pairs in regimes.items():
            recs = [records[r, p] for p in pairs]
            names = recs[0]["scores"].keys()
            if any(rec["scores"].keys() != names for rec in recs):
                raise ValueError("Formal pairs scored different receiver sets")
            table = {}
            for method in names:
                source = {str(rec["pair"]): rec["scores"][method] for rec in recs}
                pooled = {
                    key: sum(row[key] for row in source.values())
                    for key in next(iter(source.values()))
                }
                expected = len(pairs) * (1440 if r == 2 else 320)
                if pooled["blocks"] != expected or any(
                    x["blocks"] != expected // len(pairs) for x in source.values()
                ):
                    raise ValueError(
                        f"Incomplete LDPC denominator in {cohort}, R{r}, {method}"
                    )
                pooled.update(
                    bler=pooled["block_errors"] / pooled["blocks"],
                    pre_ber=pooled["bit_errors"] / pooled["bits"],
                    post_ber=pooled["info_errors"] / pooled["info_bits"],
                    nmse_db=10
                    * math.log10(pooled["channel_error"] / pooled["channel_energy"]),
                )
                table[method] = dict(pooled=pooled, by_pair=source)
            out[cohort][r] = table
    return out


def green_summary(records):
    """Mean over bursts and EM passes in each pair, then equally over pairs."""
    summaries = {}
    for p in study.COHORTS["A"][3]:
        data = records[3, p]["green"]
        if len(data["uniform"]) != 8:
            raise ValueError("Unexpected uniform probe size")
        item = dict(uniform=statistics.mean(data["uniform"]))
        for arm in ("pure", "parity"):
            item[arm] = {}
            for stage in ("initial", "trained"):
                layers = data[arm][stage]
                if len(layers) != 3:
                    raise ValueError("Missing EM probe pass")
                maps = {}
                for key in ("parallel_joint", "joint", "spatial", "time", "frequency"):
                    filters = (
                        ("overall", "shuffled", "axis_only")
                        if key in ("spatial", "time", "frequency")
                        else ("overall", "shuffled")
                    )
                    maps[key] = {}
                    for kind in filters:
                        values = [
                            v for layer in layers for v in layer["maps"][key][kind]
                        ]
                        if len(values) != 24 or not all(
                            v is not None and math.isfinite(v) for v in values
                        ):
                            raise ValueError(
                                f"Undefined formal Green observation at {p}/{arm}/{stage}/{key}/{kind}"
                            )
                        maps[key][kind] = statistics.mean(values)
                item[arm][stage] = maps
        summaries[p] = item
    return summaries


def comparisons(scores, green):
    def blocks(r, arm, comparator):
        rows = scores["A"][r]
        delta = {
            p: comp["block_errors"] - rows[arm]["by_pair"][p]["block_errors"]
            for p, comp in rows[comparator]["by_pair"].items()
        }
        return sign_row(r, arm, comparator, delta)

    structural = [
        ("parity", "local_parity"),
        ("pure", "local"),
        ("parity", "pure"),
        ("pure", "no_gauge"),
    ]
    kernels = [
        (
            "Direct joint",
            "Shuffled keys",
            lambda row: row["parity"]["trained"]["parallel_joint"]["overall"]
            - row["parity"]["trained"]["parallel_joint"]["shuffled"],
        ),
        (
            "Direct joint",
            "Pair-specific uniform support",
            lambda row: row["parity"]["trained"]["parallel_joint"]["overall"]
            - row["uniform"],
        ),
        (
            "Historical product, trained",
            "Historical product, initial",
            lambda row: row["parity"]["trained"]["joint"]["overall"]
            - row["parity"]["initial"]["joint"]["overall"],
        ),
        (
            "Spatial axis-only",
            "Historical P5 reference 0.084",
            lambda row: row["parity"]["trained"]["spatial"]["axis_only"] - 0.084,
        ),
    ]
    return {
        "Classical comparators": holm(
            [blocks(r, "parity", CLASSICAL[r]) for r in CLASSICAL]
        ),
        "Fixed+Parity": holm(
            [blocks(r, "parity", "physics_parity") for r in CLASSICAL]
        ),
        "Structural comparisons": holm(
            [blocks(r, a, b) for a, b in structural for r in CLASSICAL]
        ),
        "R3 Green correlations": holm(
            [
                sign_row(3, a, b, {p: fn(g) for p, g in green.items()})
                for a, b, fn in kernels
            ]
        ),
    }


def collect():
    signature, _ = study.contract()
    records = {}
    for r, p in study.pairs():
        row = ex.read(study.folder(r, p) / "record.json")
        if (row["freeze_sha256"], row["regime"], row["pair"], row["cohort"]) != (
            signature,
            r,
            p,
            study.cohort(r, p),
        ):
            raise ValueError("Formal score provenance mismatch")
        frames = row["frames"]
        if len(frames) != 10 or any(f["batch"] != 8 for f in frames):
            raise ValueError("Formal draw count mismatch")
        records[r, p] = row
    scores = scores_by_cohort(records)
    green = green_summary(records)
    legacy = {
        str(r): {str(p): records[r, p]["legacy"] for p in pairs}
        for r, pairs in study.COHORTS["B"].items()
    }
    audits = {
        k: ex.read(ex.RESULTS / f"{k}.json")
        for k in (
            "checkpoint_replay",
            "trained_gauge",
            "interventions",
            "latency",
            "checkout_verification",
        )
    }
    for key, audit in audits.items():
        reference = audit.get("freeze_sha256", audit.get("frozen_source_verification"))
        if reference != signature:
            raise ValueError(f"Audit-freeze identity mismatch in {key}")
    freeze = ex.read(FREEZE)
    replay = audits["checkpoint_replay"]
    checks = audits["checkout_verification"]
    gauge = audits["trained_gauge"]
    integrity = dict(
        replay={k: replay[k] for k in ("pairs", "checkpoints", "passed")},
        score_or_digest_failures=[row for row in replay["rows"] if not row["passed"]],
        gauge=dict(summary=gauge["summary"], failed_cases=gauge["failed_cases"]),
        checkout={
            k: checks[k]
            for k in (
                "passed",
                "p6_tests",
                "p6_passed",
                "legacy_tests",
                "legacy_passed",
                "p4_exit",
                "tree",
            )
        },
    )
    return (
        dict(
            freeze_sha256=signature,
            recipe=freeze["recipe"],
            development_gate=freeze["dev_gate"],
            cohorts=scores,
            green=green,
            families=comparisons(scores, green),
            legacy=legacy,
            integrity=integrity,
            interventions=audits["interventions"]["pooled"],
            latency=audits["latency"]["regimes"],
            audit_files_sha256={key: sha(ex.RESULTS / f"{key}.json") for key in audits},
        ),
        audits,
    )


def table(lines, header, rows):
    lines.extend(
        [
            "",
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * len(header)) + " |",
        ]
    )
    for row in rows:
        lines.append("| " + " | ".join(map(str, row)) + " |")
    lines.append("")


def probabilities(row, field):
    fraction = row[field + "_exact"]
    dec = row[field + "_p"]
    return f"{dec} ({fraction})" if "/" in fraction else fraction


def score_tables(lines, result):
    for c, title in (("A", "Prospective cohort A"), ("B", "Legacy cohort B")):
        lines += ["## " + title, ""]
        for r, methods in result["cohorts"][c].items():
            pairs = list(study.COHORTS[c][r])
            lines += [
                f"### R{r} at {study.SNR[r]:g} dB",
                "",
                f"The per-pair column follows pair order {', '.join(map(str,pairs))}.",
            ]
            table(
                lines,
                [
                    "Receiver",
                    "Failed words by pair",
                    "Pooled errors / words",
                    "BLER",
                    "Pre-BER",
                    "Post-BER",
                    "CFR NMSE, dB",
                ],
                [
                    (
                        name(m),
                        ", ".join(
                            str(x["block_errors"]) for x in row["by_pair"].values()
                        ),
                        f"{s['block_errors']:,} / {s['blocks']:,}",
                        f"{s['bler']:.3%}",
                        f"{s['pre_ber']:.3%}",
                        f"{s['post_ber']:.3%}",
                        f"{s['nmse_db']:.3f}",
                    )
                    for m, row in methods.items()
                    for s in [row["pooled"]]
                ],
            )


def green_tables(lines, result):
    green = result["green"]
    rows = []
    for a in ("pure", "parity"):
        for kernel in ("parallel_joint", "joint", "spatial", "time", "frequency"):

            def mean(stage, field):
                return statistics.mean(
                    x[a][stage][kernel][field] for x in green.values()
                )

            axis = kernel in ("spatial", "time", "frequency")
            rows.append(
                [
                    name(a),
                    name(kernel),
                    f"{mean('initial','overall'):.6f}",
                    f"{mean('trained','overall'):.6f}",
                    f"{mean('trained','shuffled'):.6f}",
                    f"{mean('initial','axis_only'):.6f}" if axis else "—",
                    f"{mean('trained','axis_only'):.6f}" if axis else "—",
                ]
            )
    uniform = statistics.mean(x["uniform"] for x in green.values())
    lines += [
        "## R3 Green alignment",
        "",
        f"Each of the ten R3 probe pairs contributes eight stream-8 bursts. At each waveform pass the overall Pearson coefficient uses 2,256 off-diagonal token pairs. The direct map averages the three parallel axial transports on their supports. The historical product composes the three measured axis maps. The shuffled control averages eight stream-9 key permutations. The pair-matched uniform-support mean was {uniform:.6f} on cohort A and 0.268702 on development.",
        "",
        "Spatial axis-only correlation restricts endpoints to the same antenna line. The comparison against 0.084 uses a fixed historical P5 reference with different support. It describes distance from that reference and cannot establish a paired superiority claim against the P5 spatial operator.",
    ]
    table(
        lines,
        [
            "Receiver",
            "Map",
            "Initial",
            "Trained",
            "Trained shuffled",
            "Initial on-axis",
            "Trained on-axis",
        ],
        rows,
    )


def audit_tables(lines, result, audits):
    iv = audits["interventions"]
    lines += [
        "## Fixed-weight interventions",
        "",
        "Each row omits the named handle at inference on the ten prospective pairs. Positive changes in failed words indicate that the intervention worsened decoding. Multiple handles can affect the same block.",
    ]
    rows = []
    for regime, arms in iv["pooled"].items():
        for arm, handles in arms.items():
            baseline = handles["full"]["block_errors"]
            for handle, row in handles.items():
                rows.append(
                    [
                        f"R{regime}",
                        name(arm),
                        name(handle),
                        f"{row['block_errors']:,} / {row['blocks']:,}",
                        f"{row['block_errors']-baseline:+d}",
                    ]
                )
    table(
        lines, ["Regime", "Receiver", "Intervention", "Failed / total", "Change"], rows
    )
    latency = audits["latency"]
    lines += [
        "## Single-thread latency",
        "",
        "Each entry reports the median and interquartile interval in milliseconds for serial receiver forwards. Host background CPU includes unrelated work observed during the timed rounds.",
    ]
    rows = []
    for r, reg in latency["regimes"].items():
        for batch, timing in reg["batches"].items():
            for method, samples in timing["methods"].items():
                rows.append(
                    [
                        f"R{r}",
                        batch,
                        name(method),
                        f"{samples['median_ms']:.3f}",
                        f"{samples['q1_ms']:.3f}–{samples['q3_ms']:.3f}",
                        f"{timing['background_percent_median']:.1f}%",
                    ]
                )
    table(
        lines,
        ["Regime", "Batch", "Method", "Median ms", "Q1–Q3 ms", "Median background CPU"],
        rows,
    )
    check = audits["checkout_verification"]
    replay = audits["checkpoint_replay"]
    gauge = audits["trained_gauge"]["summary"]
    tested = list(replay["rows"])
    archival = [pair for group in result["legacy"].values() for pair in group.values()]
    lines += [
        "## Verification and benchmark provenance",
        "",
        f"Checkpoint replay covered {replay['checkpoints']} checkpoints across {replay['pairs']} pairs. Frame and codeword digests matched on {sum(x['frames_exact'] for x in tested)}/{len(tested)} pairs, integer scores on {sum(x['integer_exact'] for x in tested)}/{len(tested)} and complete score dictionaries on {sum(x['scores_exact'] for x in tested)}/{len(tested)}. The complete replay pass flag was {replay['passed']}. [Replay details](results/checkpoint_replay.json)",
        "",
        f"Trained Axial-Pure and Axial-Parity passed {gauge['passed_checks']}/{gauge['checks']} gain and pilot conditions, with {gauge['coded_flips']:,} coded-bit flips in {gauge['decisions']:,} comparisons, maximum absolute ΔLLR {gauge['max_grid_llr_diff']:.8g} and maximum relative CFR error {gauge['max_relative_cfr_l2']:.8g}. The strict ΔLLR bound was 1e-12. [Trained audit](results/trained_gauge.json)",
        "",
        f"Cohort B archive checks passed on {sum(x['passed'] for x in archival)}/{len(archival)} pairs. The six archived integer fields cover pre-decoder coded-bit errors and denominator, post-decoder information-bit errors and denominator, and block errors and denominator. Every archived P5-B frame and codeword digest is compared. The P5 Unrolled EM and P5-B entries in the cohort B tables use restored archived weights.",
        "",
        "P4 supplies the coding, waveform and baseline lineage. P4 used different experimental designs, including two-dimensional slots and sparse DMRS. P6 shares the P5/P5-B simulator draws and legacy seeds in cohort B. Paired numerical comparisons with P4's published blocks would lack a common realization. [P4 evidence](../p4/P4_COMPLETION_EVIDENCE.md)",
        "",
        f"Checkout passed = {check['passed']}. P6 ran {check['p6_tests']} tests (pass = {check['p6_passed']}), P5 ran {check['legacy_tests']['p5']} and P5-B ran {check['legacy_tests']['p5b']} (legacy pass = {check['legacy_passed']}). The publication P4 verifier exited {check['p4_exit']}. Protected checkout pass = {check['tree']['passed']}. [Checkout](results/checkout_verification.json)",
    ]


def report(result, audits):
    pairs = result["cohorts"]["A"]
    families = result["families"]
    gate = result["development_gate"]
    counts = [pairs[r]["parity"]["pooled"]["block_errors"] for r in (1, 2, 3)]
    lines = [
        "# Phase 6 held-out evaluation",
        "",
        f"On the ten prospective pairs per regime, Axial-Parity recorded {counts[0]:,}/3,200 R1, {counts[1]:,}/14,400 R2 and {counts[2]:,}/3,200 R3 failed LDPC words. ",
        "",
        f"Positive Holm-significant gains at familywise α = 0.05 occurred in {sum(x['reject_05'] and x['wins']>x['losses'] for x in families['Classical comparators'])}/3 equal-information classical comparisons and {sum(x['reject_05'] and x['wins']>x['losses'] for x in families['Fixed+Parity'])}/3 fixed-physics-plus-parity comparisons. The tables retain ties, unfavorable effects and their two-sided tests.",
        "",
        "## Design and attribution",
        "",
        "Each of the 50 regime/pair combinations received five receiver fits. Each fit's held-out measurement used ten independently seeded Stream 7 draws of eight bursts. A pair is the unit of inference. BLER and BER pool integer counts, while CFR NMSE divides pooled error energy by pooled true-channel energy before conversion to dB. The final decoder is shared by all arms. Genie-PA receives privileged PA parameters.",
        "",
        f"The development selection scored {gate['counts'][0]}, {gate['counts'][1]} and {gate['counts'][2]} failed words. It corrected {gate['repairs']}/10 initially wrong R2 signs. The formal model and protocol were frozen before scoring. [Mathematical derivations and ablations](DEV_ITER5_MATH.md) · [Frozen protocol](PROTOCOL.md) · [SHA-256 inventory](results/freeze.json)",
        "",
        "A positive table entry below is a pooled reduction in failed words for the second receiver relative to the first. The fixed-receiver parity comparison estimates the iterative parity contribution on shared waveform physics. Adding learned handles on top of Fixed+Parity measures their aggregate contribution, while matched learned arms isolate the change of mixer.",
    ]
    contrasts = [
        ("physics", "physics_parity"),
        ("physics_parity", "parity"),
        ("local_parity", "parity"),
        ("pure", "parity"),
    ]
    table(
        lines,
        ["Comparison", "R1", "R2", "R3"],
        [
            (
                f"{name(a)} → {name(b)}",
                *[
                    f"{pairs[r][a]['pooled']['block_errors']-pairs[r][b]['pooled']['block_errors']:+d}"
                    for r in (1, 2, 3)
                ],
            )
            for a, b in contrasts
        ],
    )
    lines += [
        "## Prospective inference",
        "",
        "For block errors, differences are comparator minus named arm. For Green, differences are named correlation minus comparator. Each exact two-sided sign test discards tied pairs and counts wins and losses with equal probability under the null. Holm adjustment operates separately within the four registered families. With 10 non-tied pairs the smallest raw p is 1/512. Exact fractions accompany decimals; all pair differences are in the machine-readable summary.",
    ]
    for family, rows in families.items():
        lines += ["### " + family]
        table(
            lines,
            [
                "Regime",
                "Arm or correlation",
                "Comparator",
                "W/T/L",
                "Mean paired difference",
                "Raw p",
                "Holm p",
            ],
            [
                (
                    f"R{r['regime']}",
                    name(r["arm"]),
                    name(r["comparator"]),
                    f"{r['wins']}/{r['ties']}/{r['losses']}",
                    f"{r['mean_difference']:+.6g}",
                    probabilities(r, "raw"),
                    probabilities(r, "holm"),
                )
                for r in rows
            ],
        )
    recipe = result["recipe"]
    lines += [
        "## Mathematical model",
        "",
        f"The fixed training endpoint was {recipe['steps']} warm-cosine Adam updates for both mixers. "
        f"The frozen configuration used R2 chirp mode `{recipe['chirp']}`, R1 bridge `{recipe['bridge']}`, R1 differential PA `{recipe['differential']}`, "
        f"R3 IRC `{recipe['irc']}`, R3 ULA-frame contraction `{recipe['ula']}`, and objective `{recipe['loss']}`. "
        "The learned operator width was eight. The three waveform passes share parameters, and the centered map gives exactly zero physical increments at initialization.",
        "",
        "For a bank with template phases θ and weights w = softmax(βc), the gradient of a predicted phase with respect to score cᵢ is βwᵢ(θᵢ − Σwⱼθⱼ). "
        "Large differences between unwrapped chirps can amplify a small coherence perturbation in a late waveform pass. Development compared final-pass bank conditioning and a continuous chirp correction while checking sign repair and decoded blocks. "
        "The R1 comparisons tested a sample-clock phase bridge and differential PA calibration. The R3 comparisons tested a pilot-innovation precision matrix and a ULA-frame contraction. "
        "For the latter, adjacent-sensor lag estimates a steering phasor; conjugating the branch contraction by the resulting diagonal phase matrix preserves that estimated steering mode. "
        "The objective comparison used the same candidate loss and optimizer for both mixers. [Derivations, ablations and invariant checks](DEV_ITER5_MATH.md)",
    ]
    score_tables(lines, result)
    green_tables(lines, result)
    audit_tables(lines, result, audits)
    return "\n".join(lines) + "\n"


def main():
    summary, audits = collect()
    body = report(summary, audits)
    ex.save(ex.RESULTS / "summary.json", summary)
    (ex.ROOT / "P6_REPORT.md").write_text(body, encoding="utf-8")
    (ex.ROOT / "README.md").write_text(
        "# Phase 6 waveform receiver study\n\n"
        "Five learned receivers share their physical solver, trainable capacity, initialization, minibatches, objective and schedule. "
        "The confirmatory cohort contains ten independent pairs per regime. The legacy cohort reproduces seeds and test batches from P5 and P5-B.\n\n"
        "[Formal evaluation and exact paired inference](P6_REPORT.md) · [Protocol](PROTOCOL.md) · [Development mathematics](DEV_ITER5_MATH.md)\n\n"
        "[Numerical summary](results/summary.json) · [Pair records and weights](results/formal/) · [Freeze manifest](results/freeze.json) · "
        "[Checkpoint replay](results/checkpoint_replay.json) · [Trained symmetry](results/trained_gauge.json) · "
        "[Handle interventions](results/interventions.json) · [Single-thread latency](results/latency.json) · [Checkout verification](results/checkout_verification.json)\n\n"
        "To regenerate the tables from completed receipts, run `python -B -m p6.formal_report` from the repository root. "
        "The report checks the frozen source identity before reading formal records.\n",
        encoding="utf-8",
    )
    print("Wrote Phase 6 formal report, README and summary.", flush=True)


if __name__ == "__main__":
    main()
