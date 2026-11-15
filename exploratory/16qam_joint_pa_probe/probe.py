#!/usr/bin/env python3
"""Exploratory synthetic civilian 16-QAM/OFDM read-only receiver comparison.

This model generates ordinary communication data in memory; no radio hardware,
intentional interference, radar, emitters or weapon-related systems.
"""
import argparse, csv, hashlib, json, math, os, statistics, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

N, CP=32,16
TRAIN=(.72,.20,.08)
PROFILES={'train3':TRAIN,'equal4':(.25,)*4,'long6':(.40,.22,.15,.10,.08,.05)}
SEEDS=(59101,59102,59103)
CONDITIONS=('clean','mild','strong')
SIGMA=math.sqrt(10)
THRESH=2/SIGMA

def crand(shape,g):
    return torch.complex(torch.randn(shape,generator=g),torch.randn(shape,generator=g))/math.sqrt(2)

def q16(bits):
    # Gray 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3.
    table=torch.tensor([-3.,-1.,3.,1.],dtype=torch.float32)
    indices=bits[...,0::2]*2+bits[...,1::2]
    return torch.complex(table[indices[...,0]],table[indices[...,1]])/SIGMA

def hard_bits(z,t=THRESH):
    return torch.stack((z.real>0,z.real.abs()<t,z.imag>0,z.imag.abs()<t),dim=-1).long()

def pa_magnitude(z,condition):
    if condition=='clean':return z
    # input RMS near 1 in orthonormal FFT. Non-adversarial Tx amplifier compression.
    A=2.25 if condition=='mild' else 1.25
    return z/(1+(z.abs()/A)**4).pow(.25)

def draw(batch,profile,snr,g):
    p=torch.as_tensor(profile,dtype=torch.float32);p=p/p.sum()
    h=crand((batch,len(p)),g)*p.sqrt()[None,:]
    H=torch.fft.fft(h,n=N)
    pbits=torch.randint(0,2,(batch,N,2),generator=g)
    pilot=torch.complex(2*pbits[...,0].float()-1,2*pbits[...,1].float()-1)/math.sqrt(2)
    bits=torch.randint(0,2,(batch,N,4),generator=g)
    data=q16(bits)
    var=torch.full((batch,1),10**(-snr/10.),dtype=torch.float32)
    noise_p=crand((batch,N),g)*var.sqrt()
    noise_d=crand((batch,N),g)*var.sqrt()
    return pilot,data,bits,H,var,noise_p,noise_d

def receive(base,cond):
    pilot,data,bits,H,var,np_,nd_=base
    # Time-domain memoryless PA -> Fourier symbol -> static CP-valid FIR channel.
    ptime=pa_magnitude(torch.fft.ifft(pilot,norm='ortho'),cond)
    dtime=pa_magnitude(torch.fft.ifft(data,norm='ortho'),cond)
    yp=H*torch.fft.fft(ptime,norm='ortho')+np_
    yd=H*torch.fft.fft(dtime,norm='ortho')+nd_
    return yp*pilot.conj(),yd,bits,var

def delay_est(yp,mode='dft8'):
    time=torch.fft.ifft(yp,dim=-1)
    if mode=='dft8':
        return torch.fft.fft(F.pad(time[:,:8],(0,N-8)),dim=-1)
    if mode=='thresh':
        raise ValueError('requires noise variance')
    raise ValueError(mode)

def threshold_est(yp,nv):
    t=torch.fft.ifft(yp,dim=-1)
    level=(nv/N*math.log(N/.05)).sqrt()
    return torch.fft.fft(t*(t.abs()>level),dim=-1)

def equalize(yd,hhat,nv):
    return yd*hhat.conj()/(hhat.abs().square()+nv+0.03)

def baseline_bits(observed,t=THRESH,method='dft8'):
    yp,yd,bits,nv=observed
    h=delay_est(yp) if method=='dft8' else threshold_est(yp,nv)
    return hard_bits(equalize(yd,h,nv),t)

class TinyJoint(nn.Module):
    def __init__(self,width=16):
        super().__init__()
        self.stem=nn.Conv1d(10,width,1)
        self.conv1=nn.Conv1d(width,width,3,padding=1,padding_mode='circular')
        self.conv2=nn.Conv1d(width,width,3,padding=1,padding_mode='circular')
        self.out=nn.Conv1d(width,4,1)
        nn.init.normal_(self.out.weight,std=.004)
        nn.init.zeros_(self.out.bias)
    def forward(self,yp,yd,nv):
        h=delay_est(yp)
        z=equalize(yd,h,nv)
        feat=torch.stack((z.real,z.imag,z.abs(),yp.real,yp.imag,yd.real,yd.imag,
                         h.real,h.imag,nv.log10().expand_as(z.real)),dim=1)
        y=F.silu(self.stem(feat))
        y=y+.4*F.silu(self.conv1(y))
        y=y+.4*F.silu(self.conv2(y))
        delta=self.out(y).transpose(1,2)
        base=torch.stack((z.real,THRESH-z.real.abs(),z.imag,THRESH-z.imag.abs()),dim=-1)*5.
        return base+0.5*delta

def validate():
    assert q16(torch.tensor([[[0,0,0,0],[0,1,1,1],[1,0,1,0]]])).shape==(1,3)
    b=torch.randint(0,2,(200,32,4),generator=torch.Generator().manual_seed(1))
    assert torch.equal(hard_bits(q16(b)),b),'16-QAM Gray map invert'
    x=torch.randn(64,dtype=torch.float32)
    # PA amplitude is monotonic and never increases input amplitude.
    for cond in ('mild','strong'):
        y=pa_magnitude(torch.complex(x,x),cond)
        assert torch.all(y.abs()<=torch.complex(x,x).abs()+1e-5)
        assert torch.isfinite(y).all()
    base=draw(4,TRAIN,200,torch.Generator().manual_seed(22))
    yp,yd,bits,nv=receive(base,'clean')
    # zero noise, no PA gives exact ideal Y=H*X.
    pp,xx,_,hh,_,_,_=base
    assert (yp-hh).abs().max()<1e-4
    assert (yd-hh*xx).abs().max()<1e-4
    m=TinyJoint()
    count=sum(p.numel() for p in m.parameters())
    assert count<=3000,count
    out=m(yp,yd,nv)
    assert out.shape==bits.shape and torch.isfinite(out).all()
    F.binary_cross_entropy_with_logits(out,bits.float()).backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    return count

def calibrated_threshold(seed):
    # Stronger traditional slicer gets training-label calibration, same distribution
    # family as the mixed neural arm. No test data used.
    g=torch.Generator().manual_seed(709100000+seed)
    zs=[];labels=[]
    with torch.no_grad():
        for i in range(12):
            cond=CONDITIONS[i%3]
            base=draw(256,TRAIN,10,g)
            yp,yd,bits,nv=receive(base,cond)
            zs.append(equalize(yd,delay_est(yp),nv))
            labels.append(bits)
        z=torch.cat(zs,dim=0);b=torch.cat(labels,dim=0)
        # Only fit magnitude threshold; sign bit keeps physical zero boundary.
        candidates=torch.linspace(.25,1.15,91)
        errors=[int(((z.real.abs()<float(t))!=b[...,1].bool()).sum()+((z.imag.abs()<float(t))!=b[...,3].bool()).sum()) for t in candidates]
        j=int(np.argmin(errors))
        return float(candidates[j]),int(errors[j]),int(b.numel())

def train(seed,steps,batch,out):
    torch.manual_seed(seed*947+7)
    state=TinyJoint().state_dict()
    models={x:TinyJoint() for x in ('clean_train','mixed_train')}
    for m in models.values():m.load_state_dict(state)
    opts={k:torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=1e-4) for k,m in models.items()}
    g=torch.Generator().manual_seed(seed*19873+41)
    start=time.monotonic()
    for step in range(steps):
        base=draw(batch,TRAIN,10,g)
        condition=CONDITIONS[step%3]
        observations={'clean_train':receive(base,'clean'), 'mixed_train':receive(base,condition)}
        for name,m in models.items():
            m.train();opts[name].zero_grad(set_to_none=True)
            yp,yd,bits,nv=observations[name]
            logits=m(yp,yd,nv)
            loss=F.binary_cross_entropy_with_logits(logits,bits.float())
            assert torch.isfinite(loss)
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
            nn.utils.clip_grad_norm_(m.parameters(),1.)
            opts[name].step()
        if (step+1)%100==0:print('train',seed,step+1,'/',steps,'secs',round(time.monotonic()-start,1),flush=True)
    for name,m in models.items():
        torch.save({'seed':seed,'name':name,'state_dict':m.state_dict(),'steps':steps,'params':sum(p.numel() for p in m.parameters())},out/f'{name}_{seed}.pt')
    return models

def eval_seed(seed,models,calib_t,frames):
    rows=[]
    for profile_name,profile in PROFILES.items():
        for cond in CONDITIONS:
            # Unique independent seed for every profile and condition;
            # within the scenario all methods see the same bits/channel/noise.
            scenario_seed=990000000+seed*1000+list(PROFILES).index(profile_name)*100+list(CONDITIONS).index(cond)
            g=torch.Generator().manual_seed(scenario_seed)
            names=['dft8_fixed','dft8_calibrated','threshold_fixed',*models]
            error={name:0 for name in names};symerr={name:0 for name in names}
            total_bits=0;total_sym=0
            for offset in range(0,frames,128):
                base=draw(min(128,frames-offset),profile,10,g)
                obs=receive(base,cond)
                yp,yd,bits,nv=obs
                with torch.inference_mode():
                    pred={'dft8_fixed':baseline_bits(obs),
                          'dft8_calibrated':baseline_bits(obs,calib_t),
                          'threshold_fixed':baseline_bits(obs,THRESH,method='thresh')}
                    for name,m in models.items():pred[name]=(m(yp,yd,nv)>0).long()
                for name,bhat in pred.items():
                    error[name]+=int((bhat!=bits).sum())
                    symerr[name]+=int((bhat!=bits).any(dim=-1).sum())
                total_bits+=bits.numel();total_sym+=bits.shape[0]*N
            for name in names:
                rows.append(dict(seed=seed,profile=profile_name,condition=cond,method=name,
                    errors=error[name],bits=total_bits,ber=error[name]/total_bits,
                    symbol_errors=symerr[name],symbols=total_sym,ser=symerr[name]/total_sym,
                    test_seed=scenario_seed))
    return rows

def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default='results')
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--frames',type=int,default=768)
    args=parser.parse_args()
    root=Path(args.out);root.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4)
    count=validate();print('SMOKE PASS, params=',count,flush=True)
    rows=[];thresholds=[];ck=[]
    for seed in SEEDS:
        threshold,errors,n=calibrated_threshold(seed)
        thresholds.append(dict(seed=seed,train_mix_threshold=threshold,cal_errors=errors,cal_bits=n))
        print('training calibrated traditional threshold',seed,threshold,flush=True)
        models=train(seed,args.steps,64,root)
        for m in models.values():m.eval()
        rows+=eval_seed(seed,models,threshold,args.frames)
        for name in models:
            path=root/f'{name}_{seed}.pt';ck.append(dict(name=path.name,sha256=checksum(path)))
        print('eval',seed,'completed',flush=True)
    with (root/'ber_by_seed.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summ=[]
    for profile in PROFILES:
        for cond in CONDITIONS:
            for method in ['dft8_fixed','dft8_calibrated','threshold_fixed','clean_train','mixed_train']:
                r=[row for row in rows if row['profile']==profile and row['condition']==cond and row['method']==method]
                e=sum(x['errors'] for x in r);n=sum(x['bits'] for x in r)
                se=sum(x['symbol_errors'] for x in r);ns=sum(x['symbols'] for x in r)
                summ.append(dict(profile=profile,condition=cond,method=method,
                    seeds=len(r),errors=e,bits=n,ber=e/n,symbol_errors=se,symbols=ns,ser=se/ns))
    with (root/'ber_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(summ[0]));writer.writeheader();writer.writerows(summ)
    audit=dict(protocol_sha256=checksum(Path(__file__).with_name('PROTOCOL.md')),
        code_sha256=checksum(__file__),params=count,python=torch.__version__,
        torch_threads=torch.get_num_threads(),steps=args.steps,eval_frames=args.frames,
        train_seeds=SEEDS,calibration=thresholds,checkpoints=ck,
        recorded_cells=len(rows),summary_cells=len(summ),
        tested_both_profiles=len(PROFILES),conditions=CONDITIONS)
    (root/'audit.json').write_text(json.dumps(audit,indent=2))
    print('done:',audit['recorded_cells'],'rows',audit['summary_cells'],'summaries',flush=True)
    for r in summ:
        if r['profile']=='equal4':print('RESULT',r['condition'],r['method'],'BER',round(100*r['ber'],4),'SER',round(100*r['ser'],4))

if __name__=='__main__':main()
