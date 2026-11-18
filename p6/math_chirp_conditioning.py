"""Paired derivative measurement at the preidentified R2/901 erasure coordinate."""
from dataclasses import replace
import math
from unittest.mock import patch

import torch

from p5b.diagnostics import SCALES, erased
from p6 import axial_em_transformer as solver, experiment as ex, iter5_dev as old, math_chirp as chirp
from p6.data import dev_frame, dev_seed
from p6.math_chirp_dev import OUT, checkpoint
from p6.math_chirp_fisher import reliability


@torch.no_grad()
def trace(model, obs, mode):
    calls = []
    ordinary = solver._soft_chirp_residual
    bank = chirp._bank
    def capture(taps, rot, beta):
        coherence, angles = chirp._bank_geometry(taps, rot)
        weights = (coherence * beta).softmax(-1)
        target = weights @ angles
        jac = torch.as_tensor(beta).reshape(-1,1,1,1) * weights[...,None] * (
            angles - target[...,None,:])
        calls.append((coherence, target, jac))
    def legacy(clean,x,basis,rot,field,cue):
        capture(chirp._soft_taps(clean,x,basis,field,cue), rot, 160.)
        return ordinary(clean,x,basis,rot,field,cue)
    def adaptive(taps,rot,beta):
        capture(taps,rot,beta)
        return bank(taps,rot,beta)
    model.chirp_mode = mode
    with patch.object(solver, "_soft_chirp_residual", legacy), patch.object(chirp, "_bank", adaptive):
        llr = model(obs)[0]
    return calls[-1], llr


def main():
    ex.setup()
    obs = erased(dev_frame(2,6,901,2).obs, "first_pilot_row")
    gen = torch.Generator().manual_seed(dev_seed(2,6,901,946))
    for scale in SCALES:
        angle = 2*math.pi*torch.rand(2,dtype=torch.float64,generator=gen)
        if scale == .5:
            break
    gain = torch.polar(torch.full_like(angle,scale),angle)
    other = replace(obs,y=gain[:,None,None,None]*obs.y,noise_var=obs.noise_var*scale**2)
    y,nv,_=solver._gauge(obs,True)
    support=reliability(y,nv,obs.pilot_mask).tolist()
    rows=[]
    for source,path in (("legacy",old.location(2,901,"parity",48,.05,False)),
                        ("fit",OUT / "fisher/fit/901/parity.json")):
        model=old.model_for(2,901,"parity",.05,False).eval()
        rec=checkpoint(model,path)
        for mode in ("baseline","fisher"):
            (c0,t0,j0),l0=trace(model,obs,mode)
            (c1,t1,_),l1=trace(model,other,mode)
            i,p,t=1,2,19
            jac=j0[i,p,:,t]
            dc=(c1-c0)[i,p]
            rows.append(dict(source=source,mode=mode,lambda_min=support[i],
                beta=160 if mode=="baseline" else 40+120*support[i],
                jacobian_l1=float(jac.abs().sum()), all_coordinates_max_jacobian_l1=float(j0.abs().sum(-2).max()),
                measured_delta_target=float((t1-t0)[i,p,t]),
                linearized_delta_target=float(jac@dc), delta_coherence=float(dc.abs().max()),
                traced_max_delta_llr=float((l1-l0).abs().max()), weight=rec["checkpoint_sha256"]))
    ex.save(OUT/"conditioning.json",dict(rows=rows,dev_pair=901,regime=2,scale=scale,
        phases=angle.tolist(),coordinate=[1,2,19],pass_index=2,
        condition="first_pilot_row",phase_seed=dev_seed(2,6,901,946)))
    for row in rows:
        print(row["source"],row["mode"],row["beta"],row["jacobian_l1"],row["traced_max_delta_llr"])


if __name__=="__main__":
    main()
