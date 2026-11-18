"""Rate-half QC lifts of the native PEG/IRA base graph, with syndrome-tested soft decoding."""
from functools import lru_cache

import numpy as np
import torch

from p4.classical_baselines import bussgang_params, soft_slice_16qam
from p4.next_classical import _channel, _fit_waveform, _pilot_fit
from p4.simulator import q16, rapp_pa
from p4.slot_2d import SLOT_PROFILES, _get_default_ldpc
from p4.stage3_link import _demap, _mapped, _mean_symbol
from p4.stage3_link_decoder import _bp


class LiftedCode:
    """Keep the parity submatrix invertible by zero-shifting its circulants."""
    def __init__(self,lift):
        self.z=lift
        self.k=64*lift
        self.n=128*lift
        base=_get_default_ldpc(128).h_matrix.numpy().astype(np.uint8)
        rng=np.random.default_rng(893011)
        self.shifts=np.zeros_like(base,dtype=np.int64)
        self.shifts[:,:64]=rng.integers(0,lift,size=(64,64))
        self.info=[]
        edges=[]
        offsets=[0]
        for r,row in enumerate(base):
            cols=np.flatnonzero(row)
            self.info.append([(c,self.shifts[r,c]) for c in cols if c<64])
            for z in range(lift):
                edges.extend(c*lift+(z+self.shifts[r,c])%lift for c in cols)
                offsets.append(len(edges))
        self.edges=np.asarray(edges,dtype=np.int64)
        self.offsets=np.asarray(offsets,dtype=np.int64)
        aug=np.concatenate((base[:,64:],np.eye(64,dtype=np.uint8)),axis=1)
        for col in range(64):
            i=col+np.flatnonzero(aug[col:,col])[0]
            aug[[col,i]]=aug[[i,col]]
            use=aug[:,col].astype(bool)
            use[col]=False
            aug[use]^=aug[col]
        self.parity_inverse=aug[:,64:]

    def encode(self,info):
        u=info.reshape(-1,64,self.z).numpy().astype(np.uint8)
        rhs=np.zeros_like(u)
        for r,row in enumerate(self.info):
            for c,s in row:
                rhs[:,r]^=np.roll(u[:,c],-s,axis=-1)
        p=(self.parity_inverse @ rhs)%2
        code=np.concatenate((u,p),axis=1).reshape(-1,self.n)
        # Independently check each lifted parity equation on every actual word.
        if np.any(np.add.reduceat(code[:,self.edges],self.offsets[:-1],axis=1)%2):
            raise RuntimeError("Lifted code produced a nonzero syndrome")
        return torch.from_numpy(code.astype(np.int64))

    def posterior(self,channel):
        data=np.ascontiguousarray(channel.reshape(-1,self.n).double().numpy())
        return torch.from_numpy(_bp(data,self.edges,self.offsets,40)).reshape_as(channel)


@lru_cache(maxsize=2)
def lifted(lift):
    return LiftedCode(lift)


@torch.inference_mode()
def receive_long(y_grid,pilot_tx,mask,noise_var,*,code,cp,profile,A,alpha,neural,order):
    """Three waveform/extrinsic rounds with soft BP on the actual transmitted lifted code."""
    batch,nsym,nsc=y_grid.shape
    inv=None if order is None else order.argsort()
    observed=y_grid[:,mask]*pilot_tx.conj()
    init=torch.cat([neural(observed[i:i+2],y_grid[i:i+2],mask,noise_var[i:i+2])["llrs"][:,~mask]
                    for i in range(0,batch,2)]).double()
    def exchange(channel,old):
        logical=_mapped(channel,inv).reshape(-1,code.n)
        post=code.posterior(logical)
        message=_mapped((post-logical).clamp(-20,20).reshape_as(channel),order)
        return post,.5*old+.5*message
    post,prior=exchange(init,torch.zeros_like(init))
    estimate=soft_slice_16qam(init+prior)
    y=y_grid.to(torch.complex128)
    pdp=torch.as_tensor(SLOT_PROFILES[profile],dtype=torch.float64)
    pdp=(pdp/pdp.sum()).clamp_min(1e-9)
    noise=noise_var.double().reshape(batch,1)
    gamma,dist,_=bussgang_params(A=A,alpha=alpha)
    _,omega,t=_pilot_fit(y,pilot_tx,mask,noise,pdp,gamma,dist,cp)
    mid=t[:,0]+.5/nsym*nsc/(nsc+cp)
    rx=torch.fft.ifft(y,norm="ortho").reshape(batch,-1,1)
    b=((torch.arange(16)[:,None]>>torch.arange(4))&1)
    points=q16(b).to(torch.complex128)
    bits=b.double()
    for i in range(3):
        tx=torch.empty_like(y)
        tx[:,mask]=pilot_tx.to(y.dtype)
        tx[:,~mask]=estimate
        pa=rapp_pa(torch.fft.ifft(tx,norm="ortho"),A=A,alpha=alpha,beta=1.,knee=2.,oversample=4)
        basis=torch.stack([pa.roll(d,-1) for d in range(len(pdp))],-1)
        amp,omega,pred=_fit_waveform(basis.reshape(batch,-1,len(pdp)),rx,t.flatten(),omega,noise,pdp)
        gain=(gamma*_channel(amp,omega,mid,nsc))[:,~mask]
        residual=y-torch.fft.fft(pred.reshape_as(y),norm="ortho")
        received=residual[:,~mask]+gain*estimate
        var=torch.maximum(residual.abs().square().mean((-2,-1)),noise.flatten())
        metric=-(received[:,:,None]-gain[:,:,None]*points).abs().square()/var[:,None,None]
        llr=_demap(metric,prior,bits)
        post,prior=exchange(llr,prior)
        if i<2:
            estimate=_mean_symbol(metric,prior,bits,points)
    return {"info_bits":(post[:,:code.k]>0).long(),"llrs":llr.float()}
