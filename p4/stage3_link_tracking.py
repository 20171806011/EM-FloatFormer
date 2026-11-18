"""Four-slot pilot Doppler periodogram and joint waveform tracking for a continuous HST burst."""
import math

import torch

from p4.classical_baselines import bussgang_params, soft_slice_16qam
from p4.next_classical import _channel,_fit_waveform
from p4.next_neural import TapPhaseNoChannel
from p4.simulator import q16,rapp_pa
from p4.slot_2d import (SLOT_PROFILES,_get_default_ldpc,compensate_cfo_time,
                        compute_gauge_anchor,estimate_cfo_cp_tensor)
from p4.stage3_link import _demap,_mapped,_mean_symbol
from p4.stage3_link_decoder import SoftListDecoder


class KnownPriorEM(TapPhaseNoChannel):
    """Apply frozen neural weights normally after replacing only the pilot interpolator."""
    def __init__(self,weights,normalized_prior):
        super().__init__()
        self.load_state_dict(weights)
        self.normalized_prior=normalized_prior
        self.eval()

    def _compute_rbf_prior(self,yp_bar,t_p,f_p,t_g,f_g,n_symb,n_sc):
        return self.normalized_prior


def _track_pilots(y,pilot,mask,tap_count,cp,gamma):
    batch,nsym,nsc=y.shape
    full=mask.repeat(batch,1)
    rows=full.any(-1).nonzero().flatten()
    p=y[:,mask]*pilot.conj()
    samples=p.reshape(len(rows),nsc//2)
    tap_obs=torch.fft.ifft(samples,dim=-1)[:,:tap_count].to(torch.complex128)/gamma
    n=nsym*batch
    t=(torch.arange(n)[:,None]*(cp+nsc)+cp+torch.arange(nsc)[None,:]).double()/(n*(cp+nsc))-.5
    mids=t[:,0]+(.5*nsc)/(n*(cp+nsc))
    # Two DMRS rows per slot give alternating 9/5-symbol lags. Scan a fixed, wide
    # residual frequency interval up to 1/4 turn per symbol, using all eight rows.
    grid=torch.linspace(-math.pi*n/2,math.pi*n/2,1025,dtype=torch.float64)
    scores=torch.exp(-1j*grid[:,None]*mids[rows])@tap_obs
    ix=scores.abs().argmax(0)
    omega=grid[ix][None,:]
    amp=scores[ix,torch.arange(tap_count)][None,:]/len(rows)
    return t,mids,omega,amp


@torch.inference_mode()
def track_receive(rx_time,pilot,mask,noise_var,*,cp,profile,A,alpha,neural,code=None,order=None):
    batch=rx_time.size(0)
    nsym,nsc=mask.shape
    n=batch*nsym
    cfo=estimate_cfo_cp_tensor(rx_time.reshape(1,-1),n,nsc,cp)
    stream=compensate_cfo_time(rx_time.reshape(1,-1),cfo,nsc)
    y=torch.fft.fft(stream.reshape(batch,nsym,nsc+cp)[...,cp:],norm="ortho")
    gamma,_,_=bussgang_params(A=A,alpha=alpha)
    pdp=torch.as_tensor(SLOT_PROFILES[profile],dtype=torch.float64)
    pdp=(pdp/pdp.sum()).clamp_min(1e-9)
    t,mids,omega,amplitude=_track_pilots(y,pilot,mask,len(pdp),cp,gamma)
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
