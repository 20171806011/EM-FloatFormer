"""Recount frozen HST schedules by independent four-slot trajectory, with paired uncertainty."""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


P=Path(__file__).resolve().parent/"results"


def run():
    src=P/"stage3_spaced_results.json"
    seal=P/"stage3_spaced_protocol.json"
    spec=json.loads(seal.read_bytes()); raw=json.loads(src.read_bytes())
    if hashlib.sha256(seal.read_bytes()).hexdigest()!=raw["seal_sha256"]:
        raise ValueError("Protocol mismatch")
    rows={(r["seed"],r["hz"],r["arm"],r["spacing_slots"]):r for r in raw["rows"]}
    seeds=spec["development_seeds"]+[spec["historical_last"]]
    names=("qc1664","short128")
    keys=set(itertools.product(seeds,spec["doppler_hz"],names,(1,8)))
    if set(rows)!=keys or len(raw["rows"])!=len(keys):
        raise ValueError("Missing, duplicate or unexpected receiver/trajectory combination")
    out=[]; paired=[]
    rng=np.random.default_rng(839010)
    group_indices=rng.integers(len(seeds)-1,size=(10000,len(seeds)-1))
    for hz,arm in itertools.product(spec["doppler_hz"],names):
        for seed in seeds:
            a,b=(rows[seed,hz,arm,spacing] for spacing in (1,8))
            if not (a["transmit_active_power"]==b["transmit_active_power"] and
                    a["channel_mean_power_by_slot"][0]==b["channel_mean_power_by_slot"][0] and
                    all(r["total_bits"]==3328 and r["raw_bits"]==6656 and r["packets64"]==52 and
                        r["blocks"]==(4 if arm=="qc1664" else 52) and r["nonfinite_count"]==0 for r in (a,b))):
                raise ValueError("Paired energy or integer accounting mismatch")
        for spacing in (1,8):
            for panel,current in (("development",seeds[:-1]),("historical",seeds[-1:])):
                rs=[rows[seed,hz,arm,spacing] for seed in current]
                counts={k:sum(r[k] for r in rs) for k in ("blocks","block_errors","packets64","packet64_errors",
                                                          "total_bits","bit_errors","raw_bits","raw_bit_errors")}
                out.append({"panel":panel,"hz":hz,"arm":arm,"spacing":spacing,**counts,
                            "passing_bursts":sum(r["bler"]<.05 for r in rs),"bursts":len(rs),
                            "bler":counts["block_errors"]/counts["blocks"],
                            "packet64_loss":counts["packet64_errors"]/counts["packets64"]})
        for metric,den in (("block_errors","blocks"),("packet64_errors","packets64")):
            deltas=np.array([rows[s,hz,arm,8][metric]-rows[s,hz,arm,1][metric] for s in seeds[:-1]])
            denom=rows[seeds[0],hz,arm,1][den]
            boot=deltas[group_indices].mean(1)/denom
            reference=abs(deltas.sum())
            total=sum(abs(sum(sign*x for sign,x in zip(signs,deltas)))>=reference
                      for signs in itertools.product((-1,1),repeat=len(deltas)))
            paired.append({"panel":"development","hz":hz,"arm":arm,"endpoint":metric,
                           "gap_spaced_minus_adjacent":float(deltas.mean()/denom),
                           "paired_bootstrap_95":np.quantile(boot,[.025,.975]).tolist(),
                           "exact_two_sided_sign_flip_p":total/2**len(deltas),
                           "groups_improved_tied_worsened":[int((deltas<0).sum()),int((deltas==0).sum()),int((deltas>0).sum())]})
    result={"data_sha256":hashlib.sha256(src.read_bytes()).hexdigest(),"paired_unit":"An independently drawn continuous four-slot trajectory. Frequencies, schedules and codes are paired on underlying channel noise and bits. Bootstrap conditions on 12 development trajectories, descriptive and unadjusted.",
            "summary":out,"paired":paired}
    (P/"stage3_spaced_summary.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf8")


if __name__=="__main__":
    run()
