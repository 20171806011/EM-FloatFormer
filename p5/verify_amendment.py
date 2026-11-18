"""Compare both complete frozen studies, including the deliberately erased pilot inputs."""

import csv
import hashlib
import io
import json
import os
import types
import zipfile
from dataclasses import replace
from pathlib import Path

import torch

from p5.unrolled_em_transformer import UnrolledEMTransformer
from p5.verify_results import (
    COUNT_KEYS, FITTED, SEEDS, reconcile_statistics, reconcile_table, require,
)
from p5.wavefield_sim import simulate


HERE = Path(__file__).resolve().parent
ARCHIVE = HERE / "revisions" / "first_run.zip"
RESULTS = HERE / "results"


def from_zip(z, name):
    return json.loads(z.read(name))


def current(name):
    return json.loads((RESULTS / name).read_bytes())


def historical_audit(z):
    receipt = from_zip(z, "results/study_verification.json")
    replay = from_zip(z, "results/checkpoint_replay.json")
    for name, digest in receipt["verified_source_sha256"].items():
        require(hashlib.sha256(z.read(name)).hexdigest() == digest, f"historical source hash: {name}")
    for name in ("classical.py", "wavefield_sim.py"):
        require((HERE / name).read_bytes() == z.read(name),
                f"historical dependency changed: {name}")
    require(receipt["verification"] == "passed" and replay["total_pairs"] == 20 and
            replay["fitted_states_loaded"] == 80 and replay["total_receiver_rows"] == 165 and
            all(v["exit_code"] == 0 and v["pairs_replayed"] == list(range(1,SEEDS[r]+1))
                for r,v in enumerate(replay["regimes"],1)), "original replay/verification receipt")
    records = {(r,p): from_zip(z,f"results/r{r}/pair{p:02d}/record.json")
               for r,n in SEEDS.items() for p in range(1,n+1)}
    summary = from_zip(z,"results/summary.json")
    raw = list(csv.DictReader(io.StringIO(z.read("results/raw_by_seed.csv").decode("utf-8"))))
    cal = {r: from_zip(z,f"results/calibration_r{r}.json") for r in SEEDS}
    aggregate, _ = reconcile_table(raw,summary["aggregates"],records,cal)
    tests = reconcile_statistics(summary,records)
    require(len(raw) == receipt["raw_csv_rows"] and len(aggregate) == receipt["aggregate_rows"] and
            tests == receipt["paired_tests_by_family"], "archive independent reconciliation")
    return records, {"archive_source_hashes":receipt["verified_source_sha256"], "replay_pairs":20,
                     "independently_reconciled_raw_rows":len(raw), "independently_reconciled_aggregates":len(aggregate),
                     "independently_reconciled_tests":tests}


def archived_model(z):
    # This project-local archived module uses the two verified unchanged P5
    # dependencies. All source bytes are checked against the historical receipt.
    name = "unrolled_em_transformer.py"
    module = types.ModuleType("phase5_archived_receiver")
    module.__file__ = f"{ARCHIVE}!{name}"
    exec(compile(z.read(name),module.__file__,"exec"), module.__dict__)
    model = module.UnrolledEMTransformer(variant="full").eval()
    state = torch.load(io.BytesIO(z.read("results/r1/pair01/full.pt")),map_location="cpu",weights_only=True)
    model.load_state_dict(state,strict=True)
    return model


@torch.no_grad()
def erasure_case(model, obs, name):
    y = obs.y.clone()
    rows = obs.pilot_mask.any(-1).nonzero().flatten()
    if name == "first_pilot_row":
        y[:,:,rows[0],obs.pilot_mask[rows[0]]] = 0
    elif name == "all_pilot_tones":
        y[:,:,obs.pilot_mask] = 0
    elif name == "whole_signal":
        y.zero_()
    else:
        raise ValueError(name)
    gain = 2j
    llr, h = model(replace(obs,y=y))
    other, scaled_h = model(replace(obs,y=y*gain,noise_var=obs.noise_var*abs(gain)**2))
    diff = (other-llr).abs()
    flips = (llr > 0) != (other > 0)
    return {"all_bit_flips":int(flips.sum()), "payload_bit_flips":int(flips[:,~obs.pilot_mask].sum()),
            "total_llrs":llr.numel(), "payload_llrs":llr[:,~obs.pilot_mask].numel(),
            "max_llr_abs_difference":float(diff.max()),
            "max_payload_llr_abs_difference":float(diff[:,~obs.pilot_mask].max()),
            "max_channel_backtransformed_abs_difference":float((scaled_h/gain-h).abs().max()),
            "nonzero_llrs_unscaled":int(torch.count_nonzero(llr)), "nonzero_llrs_scaled":int(torch.count_nonzero(other))}


def edge_measurements(z):
    old = archived_model(z)
    new = UnrolledEMTransformer(variant="full").eval()
    new.load_state_dict(torch.load(RESULTS / "r1/pair01/full.pt",map_location="cpu",weights_only=True),strict=True)
    frame = simulate(1,2,12,55139)
    rows = {}
    for case in ("first_pilot_row","all_pilot_tones","whole_signal"):
        rows[case] = {"original": erasure_case(old,frame.obs,case), "amended": erasure_case(new,frame.obs,case)}
    amended_pass = all(v["amended"]["all_bit_flips"] == 0 and v["amended"]["max_llr_abs_difference"] < 1e-5 and
                       v["amended"]["max_channel_backtransformed_abs_difference"] < 1e-10 for v in rows.values())
    amended_pass &= rows["whole_signal"]["amended"]["nonzero_llrs_unscaled"] == 0
    return {"regime":1, "pair":1, "variant":"full", "frame_seed":55139, "bursts":2, "snr_db":12,
            "complex_gain":[0,2], "variance_multiplier":4, "cases":rows, "amended_thresholds_met":amended_pass}


def compare_weights(old, new):
    if old.keys() != new.keys():
        return {"missing_keys":sorted(old.keys()-new.keys()), "added_keys":sorted(new.keys()-old.keys())}
    changes = {}
    for key, a in old.items():
        b = new[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            changes[key] = {"original_shape":list(a.shape), "amended_shape":list(b.shape),
                            "original_dtype":str(a.dtype), "amended_dtype":str(b.dtype)}
        elif not torch.equal(a,b):
            changes[key] = {"different_elements":int((a!=b).sum()), "max_abs_difference":float((a-b).abs().max())}
    return changes


def compare_records(z, historical):
    weights, traces, probes, scores = {}, {}, {}, {}
    count_rows_changed = 0
    score_rows = 0
    for (r,p), old in historical.items():
        folder = f"r{r}/pair{p:02d}"
        new = current(f"{folder}/record.json")
        for field in ("pair","regime","snr_db"):
            require(old[field] == new[field], f"{folder}: unmatched {field}")
        require(set(new["rows"]) == set(old["rows"]) and set(old["trace"]) == set(new["trace"]) == set(FITTED),
                f"{folder}: unmatched receiver")
        for arm in FITTED:
            path = f"{folder}/{arm}.pt"
            before = torch.load(io.BytesIO(z.read(f"results/{path}")),map_location="cpu",weights_only=True)
            after = torch.load(RESULTS / path,map_location="cpu",weights_only=True)
            delta = compare_weights(before,after)
            if delta:
                weights[path] = delta
            a,b = old["trace"][arm],new["trace"][arm]
            if a != b:
                traces[f"{folder}/{arm}"] = {"original":a,"amended":b}
        if old["probe"] != new["probe"]:
            # Full coordinates and correlations remain available in both parent records.
            probes[folder] = {"original":old["probe"],"amended":new["probe"]}
        for arm, a in old["rows"].items():
            b = new["rows"][arm]
            require(set(a)==set(b),f"{folder}/{arm}: unmatched score fields")
            changed = {k: {"original":a[k],"amended":b[k]} for k in a if a[k]!=b[k]}
            if changed:
                scores[f"{folder}/{arm}"] = changed
                count_rows_changed += bool(changed.keys() & set(COUNT_KEYS))
            score_rows += 1
    checkpoints = 4 * len(historical)
    return {"checkpoints_compared":checkpoints,"checkpoint_state_dicts_identical":checkpoints-len(weights),
            "checkpoint_differences":weights, "traces_compared":checkpoints,"traces_identical":checkpoints-len(traces),
            "trace_differences":traces, "r3_probe_pairs_compared":5,
            "r3_probe_pairs_identical":5-sum(k.startswith("r3/") for k in probes),"probe_differences":probes,
            "receiver_score_dicts_compared":score_rows, "receiver_score_dicts_identical":score_rows-len(scores),
            "integer_count_rows_identical":score_rows-count_rows_changed, "score_differences":scores}


def compare_calibrations(z):
    result = {}
    for r in SEEDS:
        name = f"calibration_r{r}.json"
        old, new = from_zip(z,"results/"+name), current(name)
        require(set(old) == set(new),f"R{r}: calibration schema mismatch")
        differences = {k:{"original":old[k],"amended":new[k]} for k in old
                       if k not in ("sources","seconds") and old[k] != new[k]}
        result[r] = {"original_snr_db":old["snr_db"],"amended_snr_db":new["snr_db"],
                     "matched_cells":len(old["coarse"])+len(old["fine"])+1,"differences":differences}
    return result


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    with zipfile.ZipFile(ARCHIVE) as z:
        records, historical = historical_audit(z)
        edge = edge_measurements(z)
        comparisons = compare_records(z,records)
        calibration = compare_calibrations(z)
    current_audit = current("study_verification.json")
    digest = hashlib.sha256((HERE / "unrolled_em_transformer.py").read_bytes()).hexdigest()
    require(current_audit["verification"] == "passed" and current_audit["verified_source_sha256"]["unrolled_em_transformer.py"] == digest,
            "amended study verification receipt")
    report = {"archive":"p5/revisions/first_run.zip", "archive_sha256":hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
              "amended_model_source_sha256":digest,"historical_verification":historical, "adversarial_erasure":edge,
              "formal_run_comparison":comparisons, "calibration_comparison":calibration}
    path = RESULTS / "amendment_verification.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(temp,path)
    print(json.dumps({"state_dicts_identical":comparisons["checkpoint_state_dicts_identical"],
                      "trace_series_identical":comparisons["traces_identical"],
                      "r3_probe_pairs_identical":comparisons["r3_probe_pairs_identical"],
                      "score_dicts_identical":comparisons["receiver_score_dicts_identical"],
                      "count_rows_identical":comparisons["integer_count_rows_identical"],
                      "amended_pass":edge["amended_thresholds_met"]},indent=2))
    require(edge["amended_thresholds_met"], "amended receiver failed erasure thresholds; receipt retained")


if __name__ == "__main__":
    main()
