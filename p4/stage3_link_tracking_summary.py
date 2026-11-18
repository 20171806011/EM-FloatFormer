"""Trajectory-level summaries of the fixed continuous HST tracking study."""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import statistics

import numpy as np


OUT=Path(__file__).resolve().parent/"results"


def summarize(bounded=False):
    name="stage3_link_bound" if bounded else "stage3_link_tracking"
    raw=json.loads((OUT/(name+".json")).read_bytes())["rows"]
    assert len(raw)==(296 if bounded else 200) and all(row["nonfinite_count"]==0 for row in raw)
    grouped=defaultdict(list)
    for row in raw:
        grouped[row["phase"],row["doppler_hz"],row["arm"]].append(row)
        if row["phase"]!="historical":
            grouped["all_development",row["doppler_hz"],row["arm"]].append(row)
    output=[]
    for (phase,hz,arm),rows in grouped.items():
        totals={k:sum(row[k] for row in rows) for k in
                ("blocks","block_errors","packets64","packet64_errors","bit_errors","total_bits","raw_bits","raw_bit_errors")}
        bler=totals["block_errors"]/totals["blocks"]
        data=dict(totals,phase=phase,doppler_hz=hz,arm=arm,bler=bler,
                  median_nmse=statistics.median(r["h_nmse"] for r in rows),trajectories=len(rows),
                  trajectories_below_target=sum(r["bler"]<.05 for r in rows),
                  maximum_trajectory_bler=max(r["bler"] for r in rows))
        if len(rows)>1:
            rng=np.random.default_rng(647321)
            # Every trajectory carries the same number of codewords within an arm.
            # Resample entire independent trajectories, keeping within-burst correlation.
            values=np.array([r["bler"] for r in rows])
            samples=values[rng.integers(0,len(rows),size=(10000,len(rows)))].mean(axis=1)
            data["cluster_percentile95_bler"]=np.quantile(samples,[.025,.975]).tolist()
        output.append(data)
    (OUT/(name+"_summary.json")).write_text(json.dumps(output,indent=2,allow_nan=False))
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bounded",action="store_true")
    summarize(parser.parse_args().bounded)
