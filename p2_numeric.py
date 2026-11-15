#!/usr/bin/env python3
"""Passive numerical representation checks; no wireless hardware."""
import csv, math
from pathlib import Path
import torch

def check(out='p2b_results/numeric_roundtrip.csv'):
    torch.manual_seed(9327)
    n=4096
    r=10.**torch.empty((n,32),dtype=torch.float64).uniform_(-5,5)
    phi=2*math.pi*torch.rand((n,32),dtype=torch.float64)
    z=torch.polar(r,phi)
    s=(z.abs().square().mean(dim=1,keepdim=True)+1e-12).sqrt()
    rectangular=torch.stack((z.real/s,z.imag/s,torch.log(s).expand_as(z.real)),dim=-1)
    zz=torch.complex(rectangular[...,0],rectangular[...,1])*torch.exp(rectangular[...,2])
    err=(zz-z).abs()
    med_relative=float(torch.median((err/(z.abs()+1e-12)).flatten()))
    max_relative=float((err/(z.abs()+1e-12)).max())
    # After quantizing all 3 channels to half precision, representation may
    # remain finite where raw IQ half overflows, but fidelity is not exact.
    orig=z.to(torch.complex64)
    qraw=torch.stack((orig.real,orig.imag),dim=-1).to(torch.float16)
    qrect=rectangular.to(torch.float16)
    recon=torch.complex(qrect[...,0].float(),qrect[...,1].float())*torch.exp(qrect[...,2].float())
    absrel=((recon.to(torch.complex128)-z).abs()/(z.abs()+1e-12)).flatten()
    assert max_relative<1e-10
    rows=[dict(setting='float64_roundtrip',median_rel_error=med_relative,max_rel_error=max_relative,
           finite_fraction=1.0),
          dict(setting='dual_storage_fp16',median_rel_error=float(absrel.median()),
               max_rel_error=float(absrel.max()),finite_fraction=float(torch.isfinite(recon).float().mean())),
          dict(setting='direct_iq_storage_fp16',median_rel_error='',max_rel_error='',
               finite_fraction=float(torch.isfinite(qraw).all(-1).float().mean()))]
    f=Path(out);f.parent.mkdir(parents=True,exist_ok=True)
    with f.open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('ROUNDTRIP',rows)
    return rows

if __name__=='__main__':check()
