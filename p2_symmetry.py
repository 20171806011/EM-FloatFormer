#!/usr/bin/env python3
"""Common global complex phase symmetry audit for passive OFDM receivers."""
import csv, math
from pathlib import Path
import torch
from ofdm_baseline import make_batch
from p2_models import ResidualBase
from p2_run import load_models, write_csv


def audit(checkpoints=Path('p2_checkpoints'),out=Path('p2c_results')):
    torch.set_num_threads(4)
    rows=[]
    for seed in (1201,1202,1203):
        models=load_models(seed,['res_equiv_local','res_unfold','p1_cnn_log'],checkpoints)
        torch.manual_seed(84130+seed)
        b=make_batch(512,32,10,10,'cpu')
        with torch.inference_mode():
            for name,m in models.items():
                orig=m(b.yp,b.yd,b.noise_var)
                for angle in (.3,1.2,2.7):
                    phase=torch.polar(torch.ones_like(b.yp.real),torch.full_like(b.yp.real,angle))
                    rotated=m(b.yp*phase,b.yd*phase,b.noise_var)
                    err=float((orig-rotated).abs().max())
                    changes=int(((orig>0)!=(rotated>0)).sum().item())
                    herror=''
                    if isinstance(m,ResidualBase):
                        h0=m.channel(b.yp,b.yd,b.noise_var)
                        hr=m.channel(b.yp*phase,b.yd*phase,b.noise_var)
                        herror=float((hr-h0*phase).abs().max())
                    rows.append(dict(seed=seed,method=name,phase_rad=angle,
                           max_logit_abs_change=err,hard_decisions_changed=changes,
                           max_channel_equiv_abs_error=herror))
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'phase_equivariance.csv',rows)
    for method in ('res_unfold','res_equiv_local'):
        e=[r for r in rows if r['method']==method]
        assert max(r['max_logit_abs_change'] for r in e)<1e-5,method
        assert max(r['max_channel_equiv_abs_error'] for r in e)<1e-5,method
        assert sum(r['hard_decisions_changed'] for r in e)==0,method
    print('PHASE AUDIT PASS',[(m,max(r['max_logit_abs_change'] for r in rows if r['method']==m))
                               for m in models])
    return rows

if __name__=='__main__':audit()
