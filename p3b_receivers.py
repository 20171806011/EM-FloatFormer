#!/usr/bin/env python3
"""Passive QPSK/OFDM neural & fixed-signal-processing receivers for P3-B."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from p3b_time_link import N, ANCHORS, PAYLOAD_MASK
from ofdm_baseline import estimate_channel, covariance_matrix
from p3_a import dft_threshold
from p3_a_mechanism import FactorialDelayConv

ENCODINGS=('raw','log_phase','rms_log')
REGIMES=('clean','mixed')


def phase_from_anchors(yd,h):
    # Known QPSK anchors x=(1+i)/sqrt2 at subcarrier 0 and 8.
    x_anchor=complex(1/math.sqrt(2),1/math.sqrt(2))
    c=(yd[:,list(ANCHORS)]*h[:,list(ANCHORS)].conj()/x_anchor).sum(dim=-1)
    # Stop gradient through decision-directed phase estimator: shared DSP step.
    phi=torch.angle(c.detach())[:,None]
    return torch.polar(torch.ones_like(phi),-phi),phi


def equalized(yd,h,nv):
    # Positive denominator preserves QPSK hard sign decisions.
    return yd*h.conj()/(h.abs().square()+nv+1e-2)


def encode(z,kind):
    r=z.abs()
    if kind=='raw':return torch.stack((z.real,z.imag,r),dim=-1)
    if kind=='log_phase':return torch.stack((torch.log1p(r),z.real/(r+1e-3),z.imag/(r+1e-3)),dim=-1)
    if kind=='rms_log':
        scale=(r.square().mean(dim=-1,keepdim=True)+1e-6).sqrt()
        return torch.stack((z.real/scale,z.imag/scale,scale.log().expand_as(r)),dim=-1)
    raise ValueError(kind)


def phase_aligned(yd,h):
    e,_=phase_from_anchors(yd,h)
    return yd*e


def logits_from_channel(yd,h,nv,align=True):
    if align:yd=phase_aligned(yd,h)
    z=equalized(yd,h,nv)
    return 3*torch.stack((z.real,z.imag),dim=-1)


def bits_from_logits(logits):return (logits>0).long()


class DataResidualHead(nn.Module):
    def __init__(self,encoding='raw',width=24):
        super().__init__()
        if encoding not in ENCODINGS:raise ValueError(encoding)
        self.encoding=encoding
        self.stem=nn.Conv1d(11,width,1)
        self.local=nn.Conv1d(width,width,3,padding=1,padding_mode='circular')
        self.out=nn.Conv1d(width,2,1)
        nn.init.normal_(self.out.weight,std=.005)
        nn.init.zeros_(self.out.bias)

    def forward(self,yd,h,nv):
        ya=phase_aligned(yd,h)
        z=equalized(ya,h,nv)
        n=yd.shape[-1]
        k=torch.arange(n,device=yd.device,dtype=z.real.dtype)
        loc=(2*math.pi*k/n)[None,:].expand_as(z.real)
        feat=torch.cat((encode(ya,self.encoding),encode(h,self.encoding),
            torch.stack((z.real,z.imag,nv.log10().expand_as(z.real),loc.cos(),loc.sin()),dim=-1)),dim=-1)
        x=F.silu(self.stem(feat.transpose(1,2)))
        x=x+0.5*F.silu(self.local(x))
        delta=self.out(x).transpose(1,2)
        return 3*torch.stack((z.real,z.imag),dim=-1)+.5*delta


def new_models(seed):
    import copy
    torch.manual_seed((int(seed)*11939+31337)%(2**31))
    ref=DataResidualHead('raw').state_dict()
    out={}
    for regime in REGIMES:
        for enc in ENCODINGS:
            key=f'{regime}__{enc}'
            model=DataResidualHead(enc)
            model.load_state_dict(copy.deepcopy(ref))
            out[key]=model
    return out


def load_frozen_d(checkpoint):
    obj=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if obj['name']!='D_no_pos_10' or obj['floor']!=.10 or obj['use_position']:
        raise ValueError('Expected frozen P3-A2 D model')
    model=FactorialDelayConv(False,.10)
    model.load_state_dict(obj['state_dict'],strict=True)
    model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


def prior_covariance():
    from p3b_time_link import PROFILES
    # Equal training-profile mixture, independent of held-out long12/sparse12.
    return sum(covariance_matrix(N,PROFILES[k],'cpu') for k in ('train3','equal4','long6'))/3


def lmmse_training_mix_fft(yp,nv):
    """LMMSE in Fourier delay coordinates with mixture training covariance.

    Population prior is the *equal training mixture* of three profiles;
    no test profile or instantaneous true channel is consumed.
    """
    from p3b_time_link import PROFILES
    p=torch.zeros(N,device=yp.device)
    for key in ('train3','equal4','long6'):
        a=torch.tensor(PROFILES[key],device=yp.device)
        p[:len(a)]+=a/a.sum()/3
    y=torch.fft.ifft(yp,dim=-1)
    gain=p[None,:]/(p[None,:]+nv/N+1e-12)
    return torch.fft.fft(y*gain,dim=-1)


def channels(observed,base):
    yp,nv=observed.yp,observed.noise_var
    cp=prior_covariance()
    # Exact same pilot and noise report as neural receivers.
    out=dict(ls=yp,
        dft8=estimate_channel(yp,nv,'dft',taps=8),
        dft12=estimate_channel(yp,nv,'dft',taps=12),
        threshold=dft_threshold(yp,nv,'hard'),
        train_lmmse=lmmse_training_mix_fft(yp,nv),
        frozen_d=base.channel(yp,observed.yd,nv))
    return out


def classical_logits(observed,base):
    hs=channels(observed,base)
    results={k:logits_from_channel(observed.yd,h,observed.noise_var,align=True) for k,h in hs.items()}
    results['threshold_no_align']=logits_from_channel(observed.yd,hs['threshold'],observed.noise_var,align=False)
    # Two-pass decision-directed estimator; unknown payload symbol hard estimates
    # are re-used as channel pseudo-pilots, known anchors enforced.
    h0=hs['threshold']
    yd_aligned=phase_aligned(observed.yd,h0)
    z=equalized(yd_aligned,h0,observed.noise_var)
    hard=torch.complex((2*(z.real>0).float()-1),(2*(z.imag>0).float()-1))/math.sqrt(2)
    hard[:,list(ANCHORS)]=complex(1/math.sqrt(2),1/math.sqrt(2))
    # Data-aligned equalized signal is used only after tentative decisions;
    # do not access actual payload bits in this classical receiver.
    pseudo=yd_aligned*hard.conj()
    refined=dft_threshold(.5*observed.yp+.5*pseudo,observed.noise_var,'hard')
    results['dd_threshold']=logits_from_channel(yd_aligned,refined,observed.noise_var,align=False)
    return results,hs


def bit_count(logits,bits):
    # Known anchors are excluded: 30 payload subcarriers x2 bits=60 /frame.
    return int(((logits>0).long()[:,PAYLOAD_MASK,:] != bits[:,PAYLOAD_MASK,:]).sum().item())


def evm(logits,bits):
    # Diagnostic unnormalized decision-proxy EVM only; not true physical EVM.
    sign=torch.complex(2*bits[:,PAYLOAD_MASK,0].float()-1,2*bits[:,PAYLOAD_MASK,1].float()-1)/math.sqrt(2)
    z=torch.complex(logits[:,PAYLOAD_MASK,0],logits[:,PAYLOAD_MASK,1])/3
    # Use optimal scalar fit only as a descriptive diagnostic.
    alpha=(z*sign.conj()).real.sum(dim=1,keepdim=True)/(sign.abs().square().sum(dim=1,keepdim=True)+1e-8)
    return float(((z-alpha*sign).abs().square()).mean().item())
