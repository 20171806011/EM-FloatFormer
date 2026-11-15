#!/usr/bin/env python3
"""Synthetic civilian QPSK/OFDM time-domain *receive-only* simulator.

Everything is generated in memory; no hardware, real signals, emitters,
EW or active interference. Four independent stochastic/analog-imperfection axes.
P3B_PROTOCOL.md freezes the experiment before training.
"""
import math
from dataclasses import dataclass
import torch

N = 32
CP = 16
ANCHORS = (0, 8)
TRAIN = (.72, .20, .08)
PROFILES = {
    'train3':TRAIN, 'equal4':(.25,)*4,
    'long6':(.40,.22,.15,.10,.08,.05),
    'long12':(.35,.15,.10,.08,.07,.06,.05,.04,.03,.025,.025,.02),
    'sparse12':(.52,0,0,0,.30,0,0,0,0,0,0,.18),
}
AXES = ('pa','impulse','cpe','adc')
LEVELS = ('mild','strong')
ALL_CONDITIONS = ('clean',) + tuple(f'{axis}_{level}' for axis in AXES for level in LEVELS)
PAYLOAD_MASK = torch.tensor([i not in ANCHORS for i in range(N)])


def complex_rand(shape, g, device='cpu'):
    real = torch.randn(shape, generator=g, device=device)
    imag = torch.randn(shape, generator=g, device=device)
    return torch.complex(real,imag)/math.sqrt(2.)


def qpsk(bits):
    return torch.complex((2*bits[...,0]-1).float(),(2*bits[...,1]-1).float())/math.sqrt(2.)


@dataclass
class Frame:
    xpilot: torch.Tensor
    xdata: torch.Tensor
    bits: torch.Tensor
    h_time: torch.Tensor
    h: torch.Tensor
    n0: torch.Tensor
    noise_p: torch.Tensor
    noise_d: torch.Tensor

@dataclass
class Received:
    yp: torch.Tensor  # complex pilot-channel observation Ypilot / Xpilot
    yd: torch.Tensor  # complex received payload FFT
    bits: torch.Tensor # 32x2, first/9th tone fixed publicly known
    h: torch.Tensor    # diagnostic truth only, never passed into models
    noise_var: torch.Tensor # reported TOTAL mean noise variance in a frequency bin
    xpilot: torch.Tensor
    stats: dict


def draw_base(batch, profile, snr_db, generator):
    assert len(profile) <= CP+1, 'CP too short'
    p = torch.as_tensor(profile,dtype=torch.float32)
    p=p/p.sum()
    ht=complex_rand((batch,len(p)),generator)*p[None,:].sqrt()
    h=torch.fft.fft(ht,n=N,dim=-1)
    pb=torch.randint(0,2,(batch,N,2),generator=generator)
    db=torch.randint(0,2,(batch,N,2),generator=generator)
    db[:,list(ANCHORS),:] = 1
    xp=qpsk(pb);xd=qpsk(db)
    if isinstance(snr_db,tuple):
        snr = torch.empty((batch,1)).uniform_(*snr_db,generator=generator)
    else:
        snr=torch.full((batch,1),float(snr_db))
    n0=10.**(-snr/10.)
    np=complex_rand((batch,N),generator) * (n0/N).sqrt()
    nd=complex_rand((batch,N),generator) * (n0/N).sqrt()
    return Frame(xp,xd,db,ht,h,n0,np,nd)


def tx_cp(x):
    time=torch.fft.ifft(x,dim=-1)
    return torch.cat((time[:,-CP:],time),dim=-1)


def causal_linear_filter(x,h):
    # Real FIR convolution, not frequency-bin-wise multiplication.
    out=torch.zeros_like(x)
    for delay in range(h.shape[-1]):
        out[:,delay:] += h[:,delay,None]*x[:,:x.shape[-1]-delay]
    return out


def rapp(x, severity):
    assert severity in LEVELS
    A=(2.0 if severity=='mild' else 1.2)/math.sqrt(N)
    return x / (1+(x.abs()/A)**4).pow(.25)


def quantize_complex(x, severity):
    A=(2.3 if severity=='mild' else 1.3)/math.sqrt(N)
    nbits=8 if severity=='mild' else 5
    levels=2**nbits-1
    def q(v):
        clipped=v.clamp(-A,A)
        return ((clipped+A)/ (2*A)*levels).round()/levels*(2*A)-A
    saturated=((x.real.abs()>A)|(x.imag.abs()>A)).float().mean()
    return torch.complex(q(x.real),q(x.imag)),saturated


def link(frame, condition='clean', extra_seed=0, include_stats=True):
    if condition!='clean':
        axis,level=condition.split('_')
        if axis not in AXES or level not in LEVELS:raise ValueError(condition)
    else:axis=level=None
    # Independently randomized optional impairment; base frames, thermal noise
    # are kept identical when comparing the clean and each impaired branch.
    gen=torch.Generator(device='cpu').manual_seed(int(extra_seed))
    xp=tx_cp(frame.xpilot)
    xd=tx_cp(frame.xdata)
    inpower=(xp.abs().square().mean()+xd.abs().square().mean())/2
    if axis=='pa':
        xp=rapp(xp,level)
        xd=rapp(xd,level)
    outpower=(xp.abs().square().mean()+xd.abs().square().mean())/2
    # Simulate same known CP independently for pilot and payload: no ISI by design
    yp=causal_linear_filter(xp,frame.h_time)[:,CP:]+frame.noise_p
    yd=causal_linear_filter(xd,frame.h_time)[:,CP:]+frame.noise_d
    total_n0=frame.n0
    burst_rate=torch.tensor(0.)
    if axis=='impulse':
        p=0.03 if level=='mild' else 0.05
        excess=0.5 if level=='mild' else 3.0
        mask_p=(torch.rand(yp.shape,generator=gen)<p)
        mask_d=(torch.rand(yd.shape,generator=gen)<p)
        burst_rate=torch.stack((mask_p.float().mean(),mask_d.float().mean())).mean()
        amp=(frame.n0*excess/(N*p)).sqrt()
        yp=yp+complex_rand(yp.shape,gen)*amp*mask_p
        yd=yd+complex_rand(yd.shape,gen)*amp*mask_d
        total_n0=frame.n0*(1+excess)
    phi=torch.zeros((yp.shape[0],1))
    if axis=='cpe':
        half=.15 if level=='mild' else .60
        phi=(2*torch.rand((yp.shape[0],1),generator=gen)-1.)*half
        yd=yd*torch.polar(torch.ones_like(phi),phi)
    saturated=torch.tensor(0.)
    if axis=='adc':
        yp,sp=quantize_complex(yp,level)
        yd,sd=quantize_complex(yd,level)
        saturated=(sp+sd)/2
    Ypilot=torch.fft.fft(yp,dim=-1)
    Ydata=torch.fft.fft(yd,dim=-1)
    # Pilot tone known, |Xp|=1; no hidden-channel information.
    yp_channel=Ypilot*frame.xpilot.conj()
    stats={}
    if include_stats:
        papr=tx_cp(frame.xdata)[:,CP:].abs().square()
        stats=dict(tx_in_power=float(inpower),tx_out_power=float(outpower),
            tx_power_ratio=float(outpower/inpower),papr=float((papr.max(dim=-1).values/(papr.mean(dim=-1)+1e-12)).mean()),
            burst_rate=float(burst_rate),cpe_abs_mean=float(phi.abs().mean()),
            adc_saturation=float(saturated),total_noise_var=float(total_n0.mean()))
    return Received(yp_channel,Ydata,frame.bits,frame.h,total_n0,frame.xpilot,stats)


def build_batch(batch,profile,snr,base_seed,condition='clean',extra_seed=None,include_stats=False):
    gen=torch.Generator(device='cpu').manual_seed(int(base_seed))
    base=draw_base(batch,profile,snr,gen)
    observed=link(base,condition,extra_seed=int(extra_seed if extra_seed is not None else base_seed+7919),include_stats=include_stats)
    return observed
