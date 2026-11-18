"""Decompose the trust update on the documented erasure and compare dev attenuation patterns."""
from dataclasses import replace
import math
from unittest.mock import patch

import torch

from p5b.diagnostics import SCALES, erased
from p6 import math_chirp as chirp, math_chirp_trust_dev as study
from p6 import math_chirp_aperture as aperture
from p6.data import dev_frame, dev_seed
from p6.math_chirp_fisher import reliability
from p6.axial_em_transformer import _gauge


@torch.no_grad()
def run():
    src = study.run.OUT / "fisher/fit/901/parity.json"
    model=study.run.old.model_for(2,901,"parity",.05,False).eval()
    ck=study.run.checkpoint(model,src)["checkpoint_sha256"]
    model.chirp_mode="fisher"
    obs=erased(dev_frame(2,6,901,2).obs,"first_pilot_row")
    seed=dev_seed(2,6,901,946)
    gen=torch.Generator().manual_seed(seed)
    for scale in SCALES:
        phase=2*math.pi*torch.rand(2,generator=gen,dtype=torch.float64)
        if scale==.5:
            break
    g=torch.polar(torch.full_like(phase,scale),phase)
    other=replace(obs,y=obs.y*g[:,None,None,None],noise_var=obs.noise_var*scale**2)
    records=[]
    for o in (obs,other):
        record={}
        def see(clean,x,basis,rot,field,cue,step,mode,info=None):
            result=study.adapter(clean,x,basis,rot,field,cue,step,mode,info)
            if step==2:
                taps=chirp._soft_taps(clean,x,basis,field,cue)
                idx=(1,2,19)
                q=info[1]
                c,angles=chirp._bank_geometry(taps,rot)
                beta=40+120*info[:,None,None]
                weights=(c*beta).softmax(-1)[1,2]
                target=weights@angles[:,19]
                jac=q*beta[1,0,0]*weights*(angles[:,19]-target)
                bank=chirp._bank(taps,rot,beta)[idx]
                tangent=chirp._tangent(taps,rot)[idx]
                record.update(q=float(q),bank=float(bank),tangent=float(tangent),
                              weighted_bank=float(q*bank),weighted_tangent=float((1-q)*tangent),
                              total=float(result[1,19,2]),jac_l1=float(jac.abs().sum()))
            return result
        with patch.object(chirp,"residual",see):
            llr=model(o)[0]
        records.append((record,llr))
    key=("bank","tangent","weighted_bank","weighted_tangent","total")
    result=dict(base=records[0][0],transformed=records[1][0],
                delta={k:records[1][0][k]-records[0][0][k] for k in key},
                traced_max_delta_llr=float((records[1][1]-records[0][1]).abs().max()),
                phase_seed=seed,gain=scale,phases=phase.tolist(),checkpoint_sha256=ck)
    study.run.ex.save(study.OUT/"conditioning.json",result)
    base=aperture.OUT
    with patch.object(chirp,"residual",study.adapter),patch.object(aperture,"OUT",study.OUT):
        aperture.evaluate()
    source=study.run.ex.read(study.OUT/"aperture.json")
    for row in source["rows"]:
        row["scores"]["trust"]=row["scores"].pop("fisher")
    study.run.ex.save(study.OUT/"aperture.json",source)
    print(result,flush=True)


if __name__=="__main__":
    study.run.ex.setup()
    run()
