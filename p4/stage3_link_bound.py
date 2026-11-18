"""Physics-bounded joint DMRS and unknown-QAM likelihood for continuous HST tracking."""
import math

import torch

from p4.classical_baselines import bussgang_params, soft_slice_16qam
from p4.next_classical import _channel,_fit_waveform
from p4.simulator import q16,rapp_pa
from p4.slot_2d import (SLOT_PROFILES,_get_default_ldpc,compensate_cfo_time,
                        compute_gauge_anchor,estimate_cfo_cp_tensor)
from p4.stage3_link import _demap,_mapped,_mean_symbol
from p4.stage3_link_decoder import SoftListDecoder


from p4.stage3_link_tracking import KnownPriorEM
from itertools import product


def _bounded_pilots(y,pilot,mask,tap_count,cp,gamma,dist,noise_var,fd_max,cfo):
    batch,nsym,nsc=y.shape
    full=mask.repeat(batch,1)
    rows=full.any(-1).nonzero().flatten()
    p=y[:,mask]*pilot.conj()
    samples=p.reshape(len(rows),nsc//2)
    tap_obs=torch.fft.ifft(samples,dim=-1)[:,:tap_count].to(torch.complex128)/gamma
    n=nsym*batch
    t=(torch.arange(n)[:,None]*(cp+nsc)+cp+torch.arange(nsc)[None,:]).double()/(n*(cp+nsc))-.5
    mids=t[:,0]+(.5*nsc)/(n*(cp+nsc))
    scale=2*math.pi*n*(nsc+cp)/nsc
    limits=((-fd_max-float(cfo))*scale,(fd_max-float(cfo))*scale)
    grid=torch.linspace(*limits,1025,dtype=torch.float64)
    scores=torch.exp(-1j*grid[:,None]*mids[rows])@tap_obs
    peaks=scores.abs()
    best=peaks.argmax(0)
    # One alternate peak outside the first Rayleigh bin of each per-tap periodogram.
    allowed=(grid[:,None]-grid[best]).abs()>2*math.pi
    alt=peaks.masked_fill(~allowed,-1).argmax(0)
    alt=torch.where(allowed.any(0),alt,best)
    pairs=torch.stack((best,alt),1)
    choices=torch.tensor(list(product(range(2),repeat=tap_count)))
    taps=torch.arange(tap_count)
    index=pairs[taps,choices]
    omega=grid[index]
    amp=scores[index,taps]/len(rows)
    h=_channel(amp,omega,mids,nsc).reshape(len(choices),batch,nsym,nsc)
    gain=gamma*h
    # Rank bounded channel hypotheses by observed known-pilot likelihood and
    # uniform 16-point finite-alphabet marginal likelihood on unknown data.
    var=noise_var.flatten()[None,:,None,None]+dist*h.abs().square()
    bits=((torch.arange(16)[:,None]>>torch.arange(4))&1)
    points=q16(bits)
    data_score=(-(y[None,:,~mask,None]-gain[:,:,~mask,None]*points).abs().square()
                /var[:,:,~mask,None]).logsumexp(-1)-var[:,:,~mask].log()
    pilot_score=-(y[None,:,mask]-gain[:,:,mask]*pilot).abs().square()/var[:,:,mask]-var[:,:,mask].log()
    selected=(data_score.sum((-1,-2))+pilot_score.sum((-1,-2))).argmax()
    return t,mids,omega[selected:selected+1],amp[selected:selected+1],limits


@torch.inference_mode()
def bounded_receive(rx_time,pilot,mask,noise_var,*,cp,profile,A,alpha,neural,fd_max,code=None,order=None):
    batch=rx_time.size(0)
    nsym,nsc=mask.shape
    n=batch*nsym
    cfo=estimate_cfo_cp_tensor(rx_time.reshape(1,-1),n,nsc,cp)
    stream=compensate_cfo_time(rx_time.reshape(1,-1),cfo,nsc)
    y=torch.fft.fft(stream.reshape(batch,nsym,nsc+cp)[...,cp:],norm="ortho")
    gamma,dist,_=bussgang_params(A=A,alpha=alpha)
    pdp=torch.as_tensor(SLOT_PROFILES[profile],dtype=torch.float64)
    pdp=(pdp/pdp.sum()).clamp_min(1e-9)
    t,mids,omega,amplitude,limits=_bounded_pilots(y,pilot,mask,len(pdp),cp,gamma,dist,noise_var,fd_max,cfo)
    h=_channel(amplitude,omega,mids,nsc).reshape_as(y)
    p=y[:,mask]*pilot.conj()
    gauge=compute_gauge_anchor(p.to(torch.complex128),eps=1e-38)
    model=KnownPriorEM(neural.state_dict(),gamma*h/gauge.unsqueeze(-1))
    init=model(p,y,mask,noise_var)["llrs"][:,~mask].double()
    short=code is None
    decoder=SoftListDecoder(_get_default_ldpc(128)) if short else code
    width=128 if short else code.n
    inverse=order.argsort() if order is not None else None
    def exchange(channel,prior):
        logical=_mapped(channel,inverse).reshape(-1,width)
        post=decoder.posterior(logical)
        extr=_mapped((post-logical).clamp(-20,20).reshape_as(channel),order)
        return post,.5*prior+.5*extr
    post,prior=exchange(init,torch.zeros_like(init))
    estimate=soft_slice_16qam(init+prior)
    obs=y.to(torch.complex128)
    timeobs=torch.fft.ifft(obs,norm="ortho").reshape(1,-1,1)
    b=((torch.arange(16)[:,None]>>torch.arange(4))&1)
    points=q16(b).to(torch.complex128)
    bits=b.double()
    for i in range(3):
        tx=torch.empty_like(obs)
        tx[:,mask]=pilot.to(obs.dtype)
        tx[:,~mask]=estimate
        pa=rapp_pa(torch.fft.ifft(tx,norm="ortho"),A=A,alpha=alpha,beta=1.,knee=2.,oversample=4)
        basis=torch.stack([pa.roll(d,-1) for d in range(len(pdp))],-1).reshape(1,-1,len(pdp))
        amplitude,omega,pred=_fit_waveform(basis,timeobs,t.flatten(),omega,noise_var.double().mean().reshape(1,1),pdp)
        bounded=omega.clamp(*limits)
        if not torch.equal(bounded,omega):
            omega=bounded
            design=basis*torch.exp(1j*t.flatten()[:,None]*omega[:,None,:])
            ridge=torch.diag_embed(noise_var.double().mean().reshape(1,1)/pdp)
            fitted=torch.linalg.solve(design.mH@design+ridge,design.mH@timeobs)
            amplitude=fitted.squeeze(-1)
            pred=(design@fitted).squeeze(-1)
        h=_channel(amplitude,omega,mids,nsc).reshape_as(obs)
        gain=(gamma*h)[:,~mask]
        residual=obs-torch.fft.fft(pred.reshape_as(obs),norm="ortho")
        received=residual[:,~mask]+gain*estimate
        var=torch.maximum(residual.abs().square().mean((-1,-2)),noise_var.flatten())
        metric=-(received[:,:,None]-gain[:,:,None]*points).abs().square()/var[:,None,None]
        llrs=_demap(metric,prior,bits)
        post,prior=exchange(llrs,prior)
        if i<2:
            estimate=_mean_symbol(metric,prior,bits,points)
    if short:
        words=decoder.list_decode(post,_mapped(metric,inverse).reshape(-1,32,16))
    else:
        words=(post[:,:code.k]>0).long()
    return {"info_bits":words,"llrs":llrs,"cfo_estimate":cfo,"h_hat":h}
