"""Replay the 23 archived P4-B conditions and both 20-slot S02 delay profiles."""
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from p4.em_floatformer import compute_ece
from p4.next_neural import TapPhaseNoChannel, digest, pilot_control
from p4.refit_pilot_prior import per_slot_errors
from p4.slot_2d import EMFloatFormer2D, SlotSimulator2D, _get_default_ldpc, _synchronize_slot, idd_pa_2d


RESULTS = Path(__file__).resolve().parent / "results"
ORIGINAL = "EMFloatFormer2D"
CONTROL = "tap_phase_pilot_only"
CLASSICAL = ("2D-LMMSE", "2D-IDD-PA", "2D-LMMSE-fixed", "2D-IDD-PA-fixed")
SEEDS = (73161, 73162, 73163)
COUNTS = ("block_errors", "total_blocks", "bit_errors", "total_bits", "uncoded_bit_errors",
          "total_uncoded_bits", "nonfinite_count")


def load_models():
    base = RESULTS / "p4b_seed2026_state.pt"
    original = EMFloatFormer2D().eval()
    original.load_state_dict(torch.load(base, weights_only=True, map_location="cpu"))
    models = {ORIGINAL: original, CONTROL: pilot_control(TapPhaseNoChannel)}
    hashes = {base.name: digest(base)}
    freeze = digest(RESULTS / "next_neural_freeze.json")
    for seed in SEEDS:
        name = f"tap_phase_no_channel_{seed}"
        path = RESULTS / f"next_neural_{name}.pt"
        ckpt = torch.load(path, weights_only=True, map_location="cpu")
        if ckpt["freeze"] != freeze:
            raise ValueError(f"Freeze mismatch for {name}")
        model = TapPhaseNoChannel().eval()
        model.load_state_dict(ckpt["weights"])
        models[name] = model
        hashes[path.name] = digest(path)
    return models, hashes


def receive(name, slot, models):
    if name in models:
        out = models[name](slot.yp_sparse, slot.y_grid, slot.pilot_mask, slot.noise_var)
        return out["llrs"][:, slot.data_mask, :], out["h_hat"]
    obs = replace(slot, profile="equal4", fd=.01, A=2.25, alpha=.5) if name.endswith("-fixed") else slot
    out = idd_pa_2d(slot_data=obs, iterations=0 if "LMMSE" in name else 2)
    return out["llrs"], out["h_hat"]


def score(llrs, h, slot, code):
    block_errors, bit_errors = per_slot_errors(llrs, slot, code)
    raw_errors = ((llrs > 0) != slot.bits).flatten(1).sum(-1).tolist()
    nmse = (h - slot.h_true).abs().square().sum((-2, -1)) / slot.h_true.abs().square().sum((-2, -1))
    b, words, k = slot.info_bits.shape
    return {"block_errors": sum(block_errors), "total_blocks": b*words,
            "bit_errors": sum(bit_errors), "total_bits": b*words*k,
            "uncoded_bit_errors": sum(raw_errors), "total_uncoded_bits": slot.bits.numel(),
            "nonfinite_count": int((~torch.isfinite(llrs)).sum() + (~torch.isfinite(h)).sum()),
            "nmse": float(nmse.mean()), "ece": compute_ece(llrs, slot.bits, n_bins=10),
            "per_slot": {"block_errors": block_errors, "bit_errors": bit_errors,
                         "uncoded_bit_errors": raw_errors, "nmse": nmse.tolist()}}


def evaluate(cfg, models, code, reference=None):
    sim = SlotSimulator2D(n_symb=cfg["n_symb"], n_sc=cfg["n_sc"], cp=cfg["cp"],
                          profile=cfg["profile"], fd=cfg["fd"], cfo=cfg["cfo"])
    slot = _synchronize_slot(sim.generate_slot(batch_size=cfg["slots"], snr_db=cfg["snr_db"],
        pa_condition=cfg["pa_condition"], coded=True, ldpc=code,
        generator=torch.Generator().manual_seed(cfg["eval_seed"])))
    rows = {}
    for name in [*models, *CLASSICAL]:
        result = score(*receive(name, slot, models), slot, code)
        if reference and name in reference:
            archived = reference[name]
            for key in COUNTS:
                if result[key] != archived[key]:
                    raise AssertionError((cfg, name, key, result[key], archived[key]))
            result["archive_float_abs_difference"] = {k: abs(result[k] - archived[k]) for k in ("nmse","ece")}
        if cfg["regime"] == "S02":
            result["meets_bler_and_nmse"] = (result["block_errors"]/result["total_blocks"] < .01
                                              and result["nmse"] < .05)
        rows[name] = result
    return {**cfg, "methods": rows}


def summarize(cells):
    totals = {}
    for name in cells[0]["methods"]:
        rows = [c["methods"][name] for c in cells]
        res = {k: sum(r[k] for r in rows) for k in COUNTS}
        res.update(bler=res["block_errors"]/res["total_blocks"], ber=res["bit_errors"]/res["total_bits"],
                   uncoded_ber=res["uncoded_bit_errors"]/res["total_uncoded_bits"],
                   nmse=float(np.mean([v for r in rows for v in r["per_slot"]["nmse"]])))
        totals[name] = res
    return totals


def paired_comparisons(cells):
    """Resample IID slots within each of the 23 fixed conditions, holding training seeds fixed."""
    rng = np.random.default_rng(838001)
    b = cells[0]["slots"]
    resamples = rng.integers(b, size=(len(cells), 3000, b))
    rows = {}
    for seed in SEEDS:
        name = f"tap_phase_no_channel_{seed}"
        rows[name] = {}
        for comparator in (ORIGINAL, CONTROL, *CLASSICAL):
            deltas = np.array([np.subtract(c["methods"][name]["per_slot"]["block_errors"],
                                           c["methods"][comparator]["per_slot"]["block_errors"]) for c in cells])
            n = sum(c["methods"][name]["total_blocks"] for c in cells)
            boot = deltas[np.arange(len(cells))[:, None, None], resamples].sum(axis=(0, 2)) / n
            by_cell = deltas.sum(1)
            rows[name][comparator] = {"block_count_difference": int(by_cell.sum()),
                "absolute_bler_difference": float(by_cell.sum()/n),
                "paired_slot_bootstrap_95": np.quantile(boot, [.025, .975]).tolist(),
                "cells_lower_tied_higher_block_errors": [
                    int((by_cell < 0).sum()), int((by_cell == 0).sum()), int((by_cell > 0).sum())]}
    return rows


@torch.inference_mode()
def run():
    torch.set_num_threads(2)
    torch.manual_seed(838000)
    archive_path = RESULTS / "p4b_coded_bler.json"
    archive = json.loads(archive_path.read_bytes())
    models, hashes = load_models()
    groups = {}
    for r in archive["rows"]:
        groups.setdefault((r["regime"], r["snr_db"]), {})[r["method"]] = r
    if len(groups) != 23 or any(set(rows) != {ORIGINAL, *CLASSICAL} for rows in groups.values()):
        raise ValueError("Unexpected P4-B archive layout")
    code = _get_default_ldpc(128)
    cells = []
    for ref in groups.values():
        cfg = {k: ref[ORIGINAL][k] for k in ("regime", "profile", "fd", "cfo", "snr_db", "eval_seed", "pa_condition")}
        cfg.update({k: archive[k] for k in ("n_symb", "n_sc", "cp")}, slots=archive["eval_slots"])
        cells.append(evaluate(cfg, models, code, ref))
        print(f'Replayed {len(cells)}/23 {cfg["regime"]} {cfg["snr_db"]} dB', flush=True)
    s02 = []
    for profile in ("long12", "sparse12"):
        cfg = dict(regime="S02", n_symb=14, n_sc=64, cp=16, slots=20, profile=profile,
                   fd=.005, cfo=0., snr_db=18., eval_seed=42, pa_condition="mild")
        s02.append(evaluate(cfg, models, code))
        print(f"Scored S02 {profile}", flush=True)
    regimes = dict.fromkeys(c["regime"] for c in cells)
    report = {"scope":"Development replay of existing archive conditions. All receivers share the same per-cell batch, observed CP frequency correction, pilot grid, midpoint channel reference and LDPC(128,64) decoder with 25 min-sum iterations. Reported raw errors on coded traffic count channel decisions before decoding. Each normalized MSE is the arithmetic mean of per-slot normalized energies. No checkpoint retraining or selection.",
              "controls":"Tap-phase pilot-only uses the same deterministic tap prior, unit complex gain, zero learned distortion and the same 16-QAM demapper. Matched classical methods use generating channel/PA metadata. Fixed classical methods use equal4, fd=.01, A=2.25, alpha=.5. The learned receivers take only pilot observations, received FFT, pilot mask and noise variance.",
              "paired_uncertainty":"Conditional on these 23 strata and fixed fits. Paired 3000-replicate stratified slot bootstrap (seed 838001). The training seeds share test waveforms and are never counted as independent test slots.",
              "archive_sha256":digest(archive_path), "weights_sha256":hashes, "source_sha256":digest(Path(__file__)),
              "torch_version":torch.__version__, "cpu_threads":torch.get_num_threads(),
              "verified_integer_rows":sum(len(rows) for rows in groups.values()), "verified_fields":list(COUNTS),
              "all_23":summarize(cells), "regime_totals":{r:summarize([c for c in cells if c["regime"]==r]) for r in regimes},
              "paired_comparisons":paired_comparisons(cells), "cells":cells, "S02":s02}
    target = RESULTS / "stage3_benchmark.json"
    tmp = target.with_suffix(".tmp")
    tmp.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf8")
    tmp.replace(target)


if __name__ == "__main__":
    run()
