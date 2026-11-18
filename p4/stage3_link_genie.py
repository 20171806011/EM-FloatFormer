"""Evaluation-only diagonal AWGN surrogate for diagnosing remaining continuous HST losses."""
import json
import math
from pathlib import Path

import torch

from p4.classical_baselines import bussgang_params
from p4.simulator import complex_randn,q16
from p4.slot_2d import _get_default_ldpc
from p4.stage3_link import _demap,_mapped
from p4.stage3_link_decoder import SoftListDecoder
from p4.stage3_link_bound_study import ARMS,FRESH
from p4.stage3_link_hst import PERM,group,render
from p4.stage3_link_long import lifted


@torch.inference_mode()
def evaluate():
    torch.set_num_threads(2)
    rows=[]
    for seed in FRESH+[42]:
        for hz in [833.9102379953802,500.]:
            sim,source,taps,state=group(hz,15.,1.25,seed)
            gen=torch.Generator().set_state(state)
            noise=complex_randn((1,source.rx_time.numel()),generator=gen,dtype=torch.complex128)
            noise=(noise*source.noise_var.flatten()[0].sqrt()).reshape(4,sim.n_symb,sim.n_sc+sim.cp)[...,sim.cp:]
            noise=torch.fft.fft(noise,norm="ortho")[:,source.data_mask]
            bits=((torch.arange(16)[:,None]>>torch.arange(4))&1)
            points=q16(bits).to(torch.complex128)
            for arm in ARMS:
                order=torch.randperm(source.data_tx.numel(),generator=torch.Generator().manual_seed(PERM)) if arm.endswith("spread") else None
                inverse=order.argsort() if order is not None else None
                short=arm.startswith("short")
                code=_get_default_ldpc(128) if short else lifted(13 if arm=="long_slot_spread" else 52)
                decoder=SoftListDecoder(code) if short else code
                logical=source.bits if short else code.encode(source.info_bits.reshape(-1,code.k))
                s=render(sim,source,taps,state,"continuous",logical,order)
                h=s.h_true[:,s.data_mask].to(torch.complex128)
                gamma,_,_=bussgang_params(A=s.A,alpha=s.alpha)
                y=gamma*h*s.data_tx+noise
                metric=-(y[...,None]-gamma*h[...,None]*points).abs().square()/s.noise_var.unsqueeze(-1)
                prior=torch.zeros_like(s.bits,dtype=torch.float64)
                channel0=_demap(metric,prior,bits.double())
                sign=2*s.bits-1
                bmd=4-torch.nn.functional.softplus(-sign*channel0).mean().item()*4/math.log(2)
                transmitted=(s.bits @ (2**torch.arange(4))).long()
                selected=metric.gather(-1,transmitted.unsqueeze(-1)).squeeze(-1)
                mi=4+(selected-metric.logsumexp(-1)).mean().item()/math.log(2)
                for _ in range(3):
                    demap=_demap(metric,prior,bits.double())
                    logical=_mapped(demap,inverse).reshape(-1,code.n)
                    post=decoder.posterior(logical)
                    extr=_mapped((post-logical).clamp(-20,20).reshape_as(demap),order)
                    prior=.5*prior+.5*extr
                if short:
                    est=decoder.list_decode(post,_mapped(metric,inverse).reshape(-1,32,16))
                else:
                    est=(post[:,:code.k]>0).long()
                err=est.flatten()!=source.info_bits.flatten()
                rows.append({"seed":seed,"hz":hz,"arm":arm,"block_errors":int(err.reshape(-1,code.k).any(-1).sum()),
                             "blocks":err.numel()//code.k,"bit_errors":int(err.sum()),"info_bits":err.numel(),
                             "payload_qam_symbols":source.data_tx.numel(),"symbol_empirical_information_bits_per_re":mi,
                             "bit_metric_empirical_information_bits_per_re":bmd,
                             "true_channel_tap_energy":float(taps[0][0].abs().square().sum()),
                             "gain_power_ratio":float(abs(gamma)**2)})
    out={"interpretation":"Privileged diagnostic only: replace full nonlinear waveform by gamma times exact physical midpoint H and exact transmitted X plus the original noise. Requires true H and X, perfect removal of distortion/ICI and no estimation. Information densities are finite realized samples relative to uniform symbol/bit priors, not rigorous bounds on the original nonlinear channel. Rate 1/2 is 2 payload information bits per 16-QAM RE.",
         "rows":rows}
    (Path(__file__).parent/"results"/"stage3_link_genie.json").write_text(json.dumps(out,indent=2,allow_nan=False))


if __name__=="__main__":
    evaluate()
