"""Dev-only training, legacy-checkpoint evaluation and audits of the information trust update."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from pathlib import Path
from unittest.mock import patch

import torch

from p6 import math_chirp as chirp, math_chirp_trust as trust, math_chirp_dev as run
from p6.data import PAIRS, dev_frame, dev_seed
from p6.extended_audit import gauge_rows, summarize


OUT = run.OUT / "trust"
HASH = hashlib.sha256((run.HASH + run.digest(Path(trust.__file__))).encode()).hexdigest()


def adapter(clean,x,basis,rot,field,cue,step,mode,pilot_reliability=None):
    if mode != "fisher":
        raise ValueError("Trust candidate requires pilot information")
    return trust.residual(clean,x,basis,rot,field,cue,step,pilot_reliability)


def work(key):
    action, source, pair, arm = key
    dest = OUT / source / str(pair) / f"{arm}.json"
    model = run.old.model_for(2,pair,arm,.05,False)
    model.chirp_mode = "fisher"
    with patch.object(chirp,"residual",adapter):
        if source == "legacy" or action == "gauge":
            path = run.old.location(2,pair,arm,48,.05,False) if source == "legacy" else dest
            source_sha = run.checkpoint(model,path)["checkpoint_sha256"]
        else:
            run.old.optimize(model,48,lambda i:dev_frame(2,5,pair,run.ex.FIT_BATCH,i))
            run.ex.save_weights(dest.with_suffix(".pt"),model,HASH)
            source_sha = run.digest(dest.with_suffix(".pt"))
        if action=="gauge":
            rows = gauge_rows(model.eval(),2,pair,arm,"trust",48,full_grid=True)
            result = summarize(rows)
            run.ex.save(dest.with_name(arm+"_gauge.json"),dict(rows=rows,summary=result,source=source,
                                       pair=pair,arm=arm,checkpoint_sha256=source_sha,implementation=HASH))
            return key,result["passed_checks"],result["max_grid_llr_diff"]
        frame=dev_frame(2,6,pair,32)
        with torch.inference_mode():
            outputs,rays=run.r2_observations(model.eval(),frame)
            score=run.ex.score([frame],lambda _:outputs)
            signs=run.sign_summary([rays])["all"]
        run.ex.save(dest,dict(score=score,signs=signs,mode="trust",pair=pair,arm=arm,
                              checkpoint_sha256=source_sha,source=source,implementation=HASH,
                              init_seed=dev_seed(2,5,pair,900),score_seed=dev_seed(2,6,pair)))
        return key,score["block_errors"],signs["repaired"]


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("score","gauge"))
    parser.add_argument("--source",nargs="+",choices=("fit","legacy"),default=("fit","legacy"))
    parser.add_argument("--arm",nargs="+",choices=run.ex.ARMS,default=run.ex.ARMS)
    args=parser.parse_args()
    jobs=[(args.action,s,p,a) for s in args.source for p in PAIRS for a in args.arm]
    with ProcessPoolExecutor(3,initializer=run.ex.setup) as pool:
        for result in pool.map(work,jobs):
            print(result,flush=True)
