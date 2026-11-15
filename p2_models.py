#!/usr/bin/env python3
"""Synthetic receive-only civilian 32-subcarrier OFDM residual models.

No real RF data, no transmitter commands, no radar, no electronic warfare.
Only produces channel estimates and passive uncoded QPSK logits.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from ofdm_baseline import decisions_from_channel
from p1c_architecture import CircularCNN

N_SC = 32


def encode_complex(z, kind):
    if kind == 'raw':
        return torch.stack((z.real, z.imag, z.abs()), dim=-1)
    if kind == 'log':
        r = z.abs()
        return torch.stack((torch.log1p(r), z.real/(r+1e-3), z.imag/(r+1e-3)),dim=-1)
    if kind == 'dual':
        # Per-frame scalar RMS makes the rectilinear principal path scale-aware.
        scale = (z.abs().square().mean(dim=1,keepdim=True) + 1e-6).sqrt()
        return torch.stack((z.real/scale, z.imag/scale,
                            torch.log(scale).expand_as(z.real)),dim=-1)
    raise ValueError(kind)


def features(yp,yd,nv,kind):
    n=yp.shape[-1]
    az=yd*yp.conj()/(yp.abs().square()+nv+1e-2)
    angle=2*math.pi*torch.arange(n,device=yp.device)/n
    aux=torch.stack((az.real,az.imag,angle.cos()[None,:].expand_as(yp.real),
                     angle.sin()[None,:].expand_as(yp.real),
                     nv.log10().expand_as(yp.real)),dim=-1)
    return torch.cat((encode_complex(yp,kind),encode_complex(yd,kind),aux),dim=-1)


def bits_logits(yd,h,nv):
    z=yd*h.conj()/(h.abs().square()+nv+1e-2)
    return 3.0*torch.stack((z.real,z.imag),dim=-1)


class ResidualBase(nn.Module):
    def channel(self,yp,yd,nv):
        raise NotImplementedError
    def forward(self,yp,yd,nv):
        return bits_logits(yd,self.channel(yp,yd,nv),nv)


class PointResidual(ResidualBase):
    """Pointwise LS residual (negative-control: no cross-subcarrier estimator)."""
    def __init__(self,kind='raw',width=92):
        super().__init__(); self.kind=kind
        self.net=nn.Sequential(nn.Linear(11,width),nn.GELU(),nn.Linear(width,width),
                               nn.GELU(),nn.Linear(width,2))
        nn.init.normal_(self.net[-1].weight,std=0.002)
        nn.init.zeros_(self.net[-1].bias)
        self.logit_gate=nn.Parameter(torch.tensor(-2.0))
    def channel(self,yp,yd,nv):
        f=features(yp,yd,nv,self.kind)
        d=self.net(f);delta=torch.complex(d[...,0],d[...,1])
        scale=(yp.abs().square().mean(dim=1,keepdim=True)+1e-6).sqrt()
        return yp+torch.sigmoid(self.logit_gate)*scale*delta


class LocalResidual(ResidualBase):
    """Circular, unconstrained frequency-domain CNN with LS residual fallback."""
    def __init__(self,kind='raw',width=48):
        super().__init__();self.kind=kind
        self.stem=nn.Conv1d(11,width,kernel_size=1)
        self.layers=nn.ModuleList([nn.Conv1d(width,width,5,padding=2,
                                         padding_mode='circular') for _ in range(3)])
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.proj=nn.Conv1d(width,2,1)
        nn.init.normal_(self.proj.weight,std=0.001)
        nn.init.zeros_(self.proj.bias)
        self.logit_gate=nn.Parameter(torch.tensor(-2.0))
    def channel(self,yp,yd,nv):
        f=features(yp,yd,nv,self.kind).transpose(1,2)
        x=self.stem(f)
        for block,norm in zip(self.layers,self.norms):
            x=x+0.5*F.gelu(norm(block(x).transpose(1,2)).transpose(1,2))
        d=self.proj(x).transpose(1,2)
        delta=torch.complex(d[...,0],d[...,1]); scale=(yp.abs().square().mean(dim=1,keepdim=True)+1e-6).sqrt()
        return yp+torch.sigmoid(self.logit_gate)*scale*delta


class DelayUnfold(ResidualBase):
    """Trainable real Wiener-like shrinkage in delay domain; strict U(1) equivariance."""
    def __init__(self,nsc=N_SC):
        super().__init__();self.nsc=nsc
        self.mlp=nn.Sequential(nn.Linear(3,24),nn.SiLU(),nn.Linear(24,24),nn.SiLU(),nn.Linear(24,1))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.constant_(self.mlp[-1].bias,0.5)
        self.logit_gate=nn.Parameter(torch.tensor(-0.8))
    def channel(self,yp,yd,nv):
        t=torch.fft.ifft(yp,dim=-1)
        k=torch.arange(self.nsc,device=yp.device,dtype=nv.dtype)[None,:].expand_as(yp.real)
        noise_t=nv/self.nsc
        obs=t.abs().square()
        feats=torch.stack((torch.log1p(obs/(noise_t+1e-8)),
                 torch.log1p(k)/math.log1p(self.nsc),
                 nv.log10().expand_as(yp.real)),dim=-1)
        gate=torch.sigmoid(self.mlp(feats).squeeze(-1))
        filt=torch.fft.fft(t*gate,dim=-1)
        return yp+torch.sigmoid(self.logit_gate)*(filt-yp)


class EquivComplexLocal(ResidualBase):
    """Strict common-phase-equivariant circular complex FIR with invariant gain.

    If both yp/yd rotate by e^{j phi}, all real invariant gating features stay
    fixed, the complex-linear kernel rotates, Hhat rotates and decoded bits
    remain unchanged up to floating-point roundoff (not bitwise identity).
    """
    def __init__(self,kernel=9):
        super().__init__(); self.kernel=kernel
        wr=torch.zeros(1,1,kernel);wr[...,kernel//2]=1.
        self.wr=nn.Parameter(wr);self.wi=nn.Parameter(torch.zeros(1,1,kernel))
        self.gatenet=nn.Sequential(nn.Conv1d(2,24,5,padding=2,padding_mode='circular'),
             nn.SiLU(),nn.Conv1d(24,24,3,padding=1,padding_mode='circular'),
             nn.SiLU(),nn.Conv1d(24,1,1))
        self.logit_gate=nn.Parameter(torch.tensor(-0.8))
    def channel(self,yp,yd,nv):
        # F.conv1d has no padding_mode; circular pad explicitly.
        a=F.pad(yp.real.unsqueeze(1),(self.kernel//2,)*2,mode='circular')
        b=F.pad(yp.imag.unsqueeze(1),(self.kernel//2,)*2,mode='circular')
        ar=F.conv1d(a,self.wr)-F.conv1d(b,self.wi)
        ai=F.conv1d(a,self.wi)+F.conv1d(b,self.wr)
        k=yp.abs().square()
        feat=torch.stack((torch.log1p(k),nv.log10().expand_as(k)),dim=1)
        mix=torch.sigmoid(self.gatenet(feat)).squeeze(1)
        new=yp+torch.sigmoid(self.logit_gate)*mix*(torch.complex(ar.squeeze(1),ai.squeeze(1))-yp)
        return new


MODEL_NAMES=('res_point_raw','res_local_raw','res_unfold',
             'res_local_log','res_local_dual','res_equiv_local',
             'p1_cnn_log','p1_cnn_raw')


def model_factory(name):
    if name=='res_point_raw':return PointResidual('raw')
    if name=='res_local_raw':return LocalResidual('raw')
    if name=='res_unfold':return DelayUnfold()
    if name=='res_local_log':return LocalResidual('log')
    if name=='res_local_dual':return LocalResidual('dual')
    if name=='res_equiv_local':return EquivComplexLocal()
    if name=='p1_cnn_log':return CircularCNN(N_SC,'logamp_phase')
    if name=='p1_cnn_raw':return CircularCNN(N_SC,'raw_iq_magnitude')
    raise ValueError(name)


def verify_models():
    from ofdm_baseline import make_batch
    torch.manual_seed(44)
    b=make_batch(6,N_SC,10,10,'cpu')
    counts={}
    for name in MODEL_NAMES:
        m=model_factory(name)
        y=m(b.yp,b.yd,b.noise_var)
        assert y.shape==b.bits.shape and torch.isfinite(y).all(),name
        F.binary_cross_entropy_with_logits(y,b.bits.float()).backward()
        assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None),name
        counts[name]=sum(x.numel() for x in m.parameters())
        if isinstance(m,ResidualBase):
            assert torch.max(torch.abs(m.channel(b.yp,b.yd,b.noise_var)-b.yp))<2.0
    # Structural invariant tests independent of any trained checkpoint.
    phi=1.2345; e=torch.polar(torch.ones_like(b.yp.real),torch.full_like(b.yp.real,phi))
    for name in ('res_unfold','res_equiv_local'):
        m=model_factory(name).eval()
        with torch.inference_mode():
            h=m.channel(b.yp,b.yd,b.noise_var)
            hr=m.channel(b.yp*e,b.yd*e,b.noise_var)
            equiv=float((hr-h*e).abs().max())
            assert equiv<3e-6,(name,equiv)
            logits=m(b.yp,b.yd,b.noise_var)
            lr=m(b.yp*e,b.yd*e,b.noise_var)
            assert (lr-logits).abs().max()<3e-6,(name,(lr-logits).abs().max())
    return counts


if __name__=='__main__':print('PASS',verify_models())
