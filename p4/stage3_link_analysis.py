"""Summarize sealed results by physical channel and codeword size; channel power is evaluation-only."""
from collections import defaultdict
import json
from pathlib import Path
import statistics

import torch

from p4.stage3_link_hst import group


ROOT=Path(__file__).resolve().parent/"results"


def average_ranks(values):
    ordered=sorted(values)
    return [(ordered.index(x)+1+len(ordered)-ordered[::-1].index(x))/2 for x in values]


def run():
    torch.set_num_threads(1)
    raw=json.loads((ROOT/"stage3_link_hst_results.json").read_bytes())["rows"]
    assert len(raw)==260 and all(r["nonfinite_count"]==0 for r in raw)
    grouped=defaultdict(list)
    energies={}
    for r in raw:
        group_key=r["doppler_hz"],r["actual_seed"]
        if group_key not in energies:
            _,_,taps,_=group(r["doppler_hz"],15.,1.25,r["actual_seed"])
            energies[group_key]=taps[0].abs().square().sum(-1).tolist()
        grouped[r["seed"]==42,r["doppler_hz"],r["propagation"],r["arm"]].append(r)
    rows=[]
    for (original,hz,mode,arm),rs in grouped.items():
        totals={k:sum(r[k] for r in rs) for k in
                ("blocks","block_errors","packets64","packet64_errors","total_bits","bit_errors","raw_bits","raw_bit_errors")}
        row={"historical":original,"doppler_hz":hz,"propagation":mode,"arm":arm,**totals,
             "misdecoded_word_bler":totals["block_errors"]/totals["blocks"],
             "ber":totals["bit_errors"]/totals["total_bits"],"group_count":len(rs),
             "four_slot_groups_with_an_error":sum(r["bit_errors"]>0 for r in rs)}
        if mode=="continuous" and not original:
            gain=[energies[(hz,r["actual_seed"])][0] for r in rs]
            err=[r["bit_errors"] for r in rs]
            row["spearman_true_channel_energy_vs_bit_errors"]=(
                statistics.correlation(average_ranks(gain),average_ranks(err)) if len(set(err))>1 else None)
            row["channel_energy_by_group"]=gain
            row["bit_errors_by_group"]=err
        rows.append(row)
    (ROOT/"stage3_link_hst_summary.json").write_text(json.dumps(rows,indent=2,allow_nan=False))
    validation=ROOT/"stage3_link_variability.json"
    if validation.exists():
        data=json.loads(validation.read_bytes())["rows"]
        if len(data)==12:
            out={k:sum(r[k] for r in data) for k in
                 ("block_errors","total_blocks","total_bit_errors","total_bits","raw_bit_errors","raw_bits","nonfinite")}
            out["burst_block_errors"]=[r["block_errors"] for r in data]
            out["bursts_below_1pct"]=sum(r["bler"]<.01 for r in data)
            out["burst_count"]=len(data)
            out["pooled_bler"]=out["block_errors"]/out["total_blocks"]
            (ROOT/"stage3_link_variability_summary.json").write_text(json.dumps(out,indent=2))
    return rows


if __name__=="__main__":
    run()
