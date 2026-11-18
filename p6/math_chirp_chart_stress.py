"""Fixed Iteration 5 weights, replacing only the weak-information tangent by a sign chart."""
from collections import defaultdict
from dataclasses import replace
from unittest.mock import patch

import torch

from p6 import math_chirp as chirp, math_chirp_dev as run
from p6 import math_chirp_trust as trust
from p6.data import PAIRS, dev_frame


def chart(clean,x,basis,rot,field,cue,step,support):
    if step==1:
        return chirp.coarse_bank(clean,x,basis,rot,field,cue)
    taps=chirp._soft_taps(clean,x,basis,field,cue)
    q=support[:,None,None]
    bank=chirp._bank(taps,rot,40+120*q)
    conditioned=chirp._conditional(taps,rot,False)
    return (q*bank+(1-q)*conditioned).mT


@torch.no_grad()
def main():
    run.ex.setup()
    rows=[]
    for pair in PAIRS:
        model=run.old.model_for(2,pair,"parity",.05,False).eval()
        model.chirp_mode="fisher_trust"
        ck=run.checkpoint(model,run.old.location(2,pair,"parity",48,.05,False))["checkpoint_sha256"]
        frame=dev_frame(2,6,pair,32)
        mask=frame.obs.pilot_mask
        time=torch.arange(mask.shape[0])[:,None]
        tone=torch.arange(mask.shape[1])[None]//2
        groups=[("intact",1.,mask),*((f"first_times_{s}",s,mask&(time==0)) for s in (.8,.6,.4,.2,0.)),
                ("inner_row_erased",0.,mask&(time==9)),
                ("interleaved_half_erased",0.,mask&((time+tone)%2==0)),
                ("four_staggered_pilots_per_row",0.,mask&((time+tone)%4==0))]
        for label,scale,selected in groups:
            field=frame.obs.y.clone()
            field[:,:,selected]*=scale
            probe=replace(frame,obs=replace(frame.obs,y=field))
            data={}
            for name,update in (("trust",trust.residual),("sign_chart",chart)):
                with patch.object(trust,"residual",update):
                    output,rays=run.r2_observations(model,probe)
                score=run.ex.score([probe],lambda _:output)
                data[name]=dict(score=score,signs=run.sign_summary([rays])["all"])
            rows.append(dict(condition=label,scale=scale,pair=pair,checkpoint=ck,
                             stream=6,batch_size=32,results=data))
    run.ex.save(run.OUT/"chart_stress.json",dict(rows=rows))


if __name__=="__main__":
    main()
