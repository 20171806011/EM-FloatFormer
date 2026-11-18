"""Paired accounting for the S07 fractional-delay development panel."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FOLDER = Path(__file__).resolve().parent / "results"


def interpolation_error(delays, pdp):
    """Expected squared interpolation error / expected channel energy for independent paths."""
    f = np.fft.fftfreq(32)
    components = np.exp(-2j * np.pi * np.asarray(delays)[:,None] * f)
    power = np.array(pdp) / np.sum(pdp)
    taps = np.fft.ifft(components[:, ::2], axis=1)
    out = {}
    for retained in (8, 16):
        pred = np.fft.fft(taps[:, :retained], n=32, axis=1)
        e = power @ np.abs(pred - components)**2
        out[f"dft{retained}"] = {"all_tones":float(e.mean()), "even_pilots":float(e[::2].mean()),
                                 "odd_tones":float(e[1::2].mean())}
    return out


def analyze():
    data = json.loads((FOLDER / "next_fractional_holdout.json").read_bytes())
    panels = defaultdict(list)
    for r in data["rows"]:
        panels[r["geometry"], r["method"], r["coded"]].append(r)
    results = []
    for (geometry, method, coded), rows in panels.items():
        if coded:
            continue
        ordered = sorted(rows, key=lambda r:r["payload_seed"])
        per_slot = lambda group, key: np.concatenate([r[key] for r in sorted(group, key=lambda r:r["payload_seed"])])
        errors = per_slot(rows, "raw_errors_by_slot")
        bits = sum(r["raw_bits"] for r in rows)
        coded_rows = panels[geometry, method, True]
        blocks = sum(r["blocks"] for r in coded_rows)
        info_bits = sum(r["info_bits"] for r in coded_rows)
        coded_err = int(per_slot(coded_rows, "wrong_info_by_slot").sum())
        failed = int(per_slot(coded_rows, "failed_by_slot").sum())
        if len(errors) != 64 or bits != 64*1664 or blocks != 64*13 or info_bits != blocks*64:
            raise AssertionError("Inconsistent holdout denominators")
        entry = {"geometry":geometry, "method":method, "iid_raw_errors":int(errors.sum()),
                 "iid_raw_bits":bits, "iid_ber":float(errors.sum()/bits),
                 "iid_channel_nmse":float(per_slot(rows, "nmse_by_slot").mean()),
                 "failed_ldpc_blocks":failed, "ldpc_blocks":blocks,
                 "wrong_decoded_info_bits":coded_err, "decoded_info_bits":info_bits,
                 "coded_panel_raw_errors":int(per_slot(coded_rows, "raw_errors_by_slot").sum())}
        for reference in ("dft8_linear", "dft8_tap_phase", "tap_phase_pilot_only"):
            ref = panels[geometry, reference, False]
            if [r["payload_seed"] for r in ordered] != sorted(r["payload_seed"] for r in ref):
                raise AssertionError("Unpaired payload streams")
            diff = (errors - per_slot(ref,"raw_errors_by_slot")) / (bits / len(errors))
            rng = np.random.default_rng(846900)
            q = np.quantile(rng.choice(diff, size=(5000,len(diff)), replace=True).mean(axis=1),[.025,.975])
            entry["minus_"+reference] = {"ber_delta":float(diff.mean()), "paired_slot_pct95":q.tolist()}
        results.append(entry)
    p = np.exp(-.28*np.arange(16))
    ideal = {"equal4_integer":interpolation_error(range(4),np.ones(4)),
             "exp16_integer":interpolation_error(range(16),p),
             "exp16_fractional":interpolation_error(data["fractional_delays_samples"],p)}
    (FOLDER/"next_fractional_summary.json").write_text(json.dumps({
        "note":"Differences are absolute BER. Percentile bootstrap resamples paired slots, conditional on each trained receiver and geometry. Replicates share evaluation slots; no adjustment for multiple comparisons.",
        "static_noiseless_ratio_of_expected_channel_error_energy_to_truth_energy":ideal,
        "rows":results}, indent=2, allow_nan=False), encoding="utf-8")
    for r in results:
        if r["geometry"]=="exp16_fractional":
            print(r["method"],r["iid_raw_errors"],round(r["iid_ber"],5),round(r["iid_channel_nmse"],4),r["failed_ldpc_blocks"],r["minus_dft8_tap_phase"])


if __name__ == "__main__":
    analyze()
