#!/usr/bin/env python3
"""P1-C: matched architecture comparison for receive-only simulated civilian OFDM.

No radar, emitting devices, adversarial emitters, jamming, or military data.
All models see identical synthetic pilot/data IQ, noise variance, coarse pilot
estimate and subcarrier coordinates. Classical baselines are included.

Usage:
 python p1c_architecture.py --smoke
 python p1c_architecture.py --steps 750 --eval-frames 2000 --seeds 101 202 303 --out p1c_results
"""
import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from ofdm_baseline import (make_batch, covariance_matrix, estimate_channel,
                           decisions_from_channel)
from p1b_representation import iq_features, RepresentationTransformer

TRAIN_PROFILE=(0.72,0.20,0.08)
TEST_PROFILES={'in_distribution':TRAIN_PROFILE,
               'unseen_four_tap':(0.25,0.25,0.25,0.25)}
ARCHES=('token_mlp','mlp_mixer','circular_cnn','transformer')
ENCODINGS=('raw_iq_magnitude','logamp_phase')
SNRS=(0,5,10,15,20)


def common_features(yp, yd, noise_var, encoding, omega):
    n=yp.shape[-1]
    z=yd*yp.conj()/(yp.abs().square()+noise_var+.01)
    a=omega[None,:].expand(yp.shape[0],-1)
    nv=noise_var.expand(-1,n)
    feat=torch.cat((iq_features(yp,encoding),iq_features(yd,encoding),
        torch.stack((z.real,z.imag,a.cos(),a.sin(),nv.log10()),dim=-1)),dim=-1)
    return feat, torch.stack((z.real,z.imag),dim=-1)


class InputReceiver(nn.Module):
    def __init__(self, nsc, encoding):
        super().__init__()
        self.nsc=nsc
        self.encoding=encoding
        self.register_buffer('omega',2*math.pi*torch.arange(nsc)/nsc)
    def make_input(self,yp,yd,nv):
        return common_features(yp,yd,nv,self.encoding,self.omega)


class TokenMLP(InputReceiver):
    """Pointwise MLP, independent per subcarrier except for shared equalizer."""
    def __init__(self,nsc,encoding,width=135):
        super().__init__(nsc,encoding)
        self.net=nn.Sequential(nn.Linear(11,width),nn.GELU(),
            nn.Linear(width,width),nn.GELU(),nn.Linear(width,width),
            nn.GELU(),nn.Linear(width,2))
    def forward(self,yp,yd,nv):
        f,z=self.make_input(yp,yd,nv)
        return self.net(f)+3*z


class MixerBlock(nn.Module):
    def __init__(self,nsc,width,expansion):
        super().__init__()
        self.nt=nn.LayerNorm(width)
        self.token=nn.Sequential(nn.Linear(nsc,nsc),nn.GELU(),nn.Linear(nsc,nsc))
        self.nc=nn.LayerNorm(width)
        self.channel=nn.Sequential(nn.Linear(width,expansion),nn.GELU(),
                                   nn.Linear(expansion,width))
    def forward(self,x):
        x=x+0.5*self.token(self.nt(x).transpose(1,2)).transpose(1,2)
        x=x+0.5*self.channel(self.nc(x))
        return x


class MLPMixer(InputReceiver):
    """All-subcarrier global context without attention; nsc fixed."""
    def __init__(self,nsc,encoding,width=48,expansion=171):
        super().__init__(nsc,encoding)
        self.in_proj=nn.Linear(11,width)
        self.blocks=nn.Sequential(MixerBlock(nsc,width,expansion),
                                  MixerBlock(nsc,width,expansion))
        self.out_proj=nn.Linear(width,2)
    def forward(self,yp,yd,nv):
        f,z=self.make_input(yp,yd,nv)
        return self.out_proj(self.blocks(self.in_proj(f)))+3*z


class CircularCNN(InputReceiver):
    """Local 1D convolution on subcarriers with circular padding."""
    def __init__(self,nsc,encoding,width=48):
        super().__init__(nsc,encoding)
        self.stem=nn.Conv1d(11,width,kernel_size=1)
        self.convs=nn.ModuleList([nn.Conv1d(width,width,kernel_size=5,
                                             padding=2,padding_mode='circular')
                                  for _ in range(3)])
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(3)])
        self.local=nn.Conv1d(width,width,1)
        self.out_proj=nn.Conv1d(width,2,1)
    def forward(self,yp,yd,nv):
        f,z=self.make_input(yp,yd,nv)
        x=self.stem(f.transpose(1,2))
        for c,n in zip(self.convs,self.norms):
            y=n(c(x).transpose(1,2)).transpose(1,2)
            x=x+0.5*F.gelu(y)
        x=x+0.5*F.gelu(self.local(x))
        return self.out_proj(x).transpose(1,2)+3*z


def model_factory(arch,encoding,nsc):
    if arch=='token_mlp':return TokenMLP(nsc,encoding)
    if arch=='mlp_mixer':return MLPMixer(nsc,encoding)
    if arch=='circular_cnn':return CircularCNN(nsc,encoding)
    if arch=='transformer':return RepresentationTransformer(nsc,encoding=encoding)
    raise ValueError(arch)


def write_csv(path,rows):
    if not rows:return
    os.makedirs(os.path.dirname(path) or '.',exist_ok=True)
    with open(path,'w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def smoke_check(nsc=32):
    torch.manual_seed(123)
    b=make_batch(4,nsc,10,10,'cpu',TRAIN_PROFILE)
    counts={}
    for arch in ARCHES:
        for enc in ENCODINGS:
            m=model_factory(arch,enc,nsc)
            y=m(b.yp,b.yd,b.noise_var)
            assert y.shape==b.bits.shape and torch.isfinite(y).all()
            loss=F.binary_cross_entropy_with_logits(y,b.bits.float())
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
            counts[(arch,enc)]=sum(p.numel() for p in m.parameters())
    for arch in ARCHES:assert counts[arch,ENCODINGS[0]]==counts[arch,ENCODINGS[1]]
    lo=min(counts.values());hi=max(counts.values())
    assert (hi-lo)/hi < 0.02,(lo,hi)
    print('SMOKE PASS: shape, finite gradients, encoder-paired equal params; budget',counts,flush=True)
    return counts


def train(seed,steps,batch_size,nsc,arches,encodings):
    torch.manual_seed(seed)
    models={}
    opt={}
    for arch in arches:
        base=model_factory(arch,encodings[0],nsc)
        for enc in encodings:
            key=f'{arch}__{enc}'
            m=model_factory(arch,enc,nsc)
            m.load_state_dict(base.state_dict()) # identical initial weights within architecture
            models[key]=m
            opt[key]=torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=1e-4)
    start=time.perf_counter()
    last={}
    for step in range(1,steps+1):
        b=make_batch(batch_size,nsc,0.,20.,'cpu',TRAIN_PROFILE)
        for key,m in models.items():
            m.train()
            opt[key].zero_grad(set_to_none=True)
            pred=m(b.yp,b.yd,b.noise_var)
            loss=F.binary_cross_entropy_with_logits(pred,b.bits.float())
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(),1.)
            opt[key].step()
            last[key]=float(loss.detach())
        if step==1 or step%250==0 or step==steps:
            print('train seed',seed,'step',step,'/',steps,'sec',round(time.perf_counter()-start,1),
                  'loss', {k:round(v,4) for k,v in last.items()},flush=True)
    return models,time.perf_counter()-start


def eval_seed(models,seed,nsc,eval_frames):
    rows=[]
    paired=[]
    baselines=('ls','dft_train_support','lmmse_train_prior','lmmse_matched_prior','oracle')
    priorcov=covariance_matrix(nsc,TRAIN_PROFILE,'cpu')
    for m in models.values():m.eval()
    with torch.inference_mode():
        for split,profile in TEST_PROFILES.items():
            matched_cov=covariance_matrix(nsc,profile,'cpu')
            for snr in SNRS:
                torch.manual_seed(seed*100000+snr*100+(0 if split=='in_distribution' else 1))
                names=list(models)+list(baselines)
                err={name:0 for name in names}
                pairs={key:[] for key in models if key.endswith('__logamp_phase')}
                nbits=0
                for off in range(0,eval_frames,128):
                    b=make_batch(min(128,eval_frames-off),nsc,snr,snr,'cpu',profile)
                    pred={k:(m(b.yp,b.yd,b.noise_var)>0).long() for k,m in models.items()}
                    ch={
                        'ls':estimate_channel(b.yp,b.noise_var,'ls'),
                        'dft_train_support':estimate_channel(b.yp,b.noise_var,'dft',taps=len(TRAIN_PROFILE)),
                        'lmmse_train_prior':estimate_channel(b.yp,b.noise_var,'lmmse',covariance=priorcov),
                        'lmmse_matched_prior':estimate_channel(b.yp,b.noise_var,'lmmse',covariance=matched_cov),
                        'oracle':b.h}
                    for name,h in ch.items():pred[name]=decisions_from_channel(b.yd,h,b.noise_var)
                    for key,p in pred.items():err[key]+=int((p!=b.bits).sum())
                    for logkey in pairs:
                        rawkey=logkey.replace('logamp_phase','raw_iq_magnitude')
                        dif=(pred[rawkey]!=b.bits).sum((1,2))-(pred[logkey]!=b.bits).sum((1,2))
                        pairs[logkey].extend((dif/(2*nsc)).tolist())
                    nbits+=b.bits.numel()
                for key in names:
                    rows.append(dict(seed=seed,split=split,snr_db=snr,
                                     method=key,errors=err[key],bits=nbits,ber=err[key]/nbits))
                for key,v in pairs.items():
                    d=torch.as_tensor(v,dtype=torch.float64)
                    mean=float(d.mean()); se=float(d.std(unbiased=True)/math.sqrt(len(d)))
                    paired.append(dict(seed=seed,split=split,snr_db=snr,arch=key.split('__')[0],
                                       raw_minus_log_ber=mean,frame_se=se,frames=len(d)))
            print('eval seed',seed,'split',split,'complete',flush=True)
    return rows,paired


def benchmark_latency(models,nsc,eval_batch=1,repeat=150):
    torch.manual_seed(8789)
    b=make_batch(eval_batch,nsc,10,10,'cpu',TRAIN_PROFILE)
    for m in models.values():m.eval()
    rows=[]
    with torch.inference_mode():
        for key,m in models.items():
            for i in range(20):_ = m(b.yp,b.yd,b.noise_var)
            samples=[]
            for i in range(repeat):
                t=time.perf_counter()
                _=m(b.yp,b.yd,b.noise_var)
                samples.append((time.perf_counter()-t)*1000.)
            samples.sort()
            rows.append(dict(method=key,batch_frames=eval_batch,repetitions=repeat,
                             median_ms=samples[len(samples)//2],p95_ms=samples[int(repeat*.95)],
                             mean_ms=sum(samples)/repeat))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=750)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--eval-frames',type=int,default=2000)
    parser.add_argument('--seeds',type=int,nargs='+',default=[101,202,303])
    parser.add_argument('--nsc',type=int,default=32)
    parser.add_argument('--out',default='p1c_results')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--arches',nargs='+',choices=ARCHES,default=list(ARCHES))
    parser.add_argument('--encodings',nargs='+',choices=ENCODINGS,default=list(ENCODINGS))
    args=parser.parse_args()
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    counts=smoke_check(args.nsc)
    if args.smoke:return
    os.makedirs(args.out,exist_ok=True)
    allrows=[];allpaired=[];timings=[];latencies=[]
    for seed in args.seeds:
        models,elapsed=train(seed,args.steps,args.batch_size,args.nsc,args.arches,args.encodings)
        for key,m in models.items():
            arch,enc=key.split('__')
            torch.save(dict(state_dict=m.state_dict(),arch=arch,encoding=enc,seed=seed,
                steps=args.steps,nsc=args.nsc),os.path.join(args.out,f'model_{key}_seed{seed}.pt'))
            timings.append(dict(seed=seed,method=key,group_train_seconds=elapsed,
                                parameters=sum(p.numel() for p in m.parameters())))
        if seed==args.seeds[0]:
            for sz in [1,64]:latencies.extend(benchmark_latency(models,args.nsc,sz,80 if sz==1 else 40))
            write_csv(os.path.join(args.out,'latency_cpu.csv'),latencies)
        rows,paired=eval_seed(models,seed,args.nsc,args.eval_frames)
        allrows.extend(rows);allpaired.extend(paired)
        write_csv(os.path.join(args.out,'ber_by_seed.csv'),allrows)
        write_csv(os.path.join(args.out,'paired_effects_by_seed.csv'),allpaired)
        write_csv(os.path.join(args.out,'training.csv'),timings)
    grouped=defaultdict(list)
    for r in allrows:grouped[(r['split'],r['snr_db'],r['method'])].append(r)
    summary=[]
    for (split,snr,method),items in sorted(grouped.items()):
        total_err=sum(a['errors'] for a in items)
        total_bits=sum(a['bits'] for a in items)
        values=torch.tensor([a['ber'] for a in items],dtype=torch.float64)
        summary.append(dict(split=split,snr_db=snr,method=method,
            pooled_ber=total_err/total_bits,errors=total_err,bits=total_bits,
            across_seed_mean=float(values.mean()),
            across_seed_sd=float(values.std(unbiased=True)) if len(items)>1 else float('nan')))
    write_csv(os.path.join(args.out,'ber_summary.csv'),summary)
    config=dict(**vars(args),torch_version=torch.__version__,platform='cpu',
                threads=torch.get_num_threads(),train_profile=TRAIN_PROFILE,
                test_profiles=TEST_PROFILES,design='fixed 32 subcarrier, all-pilot, uncoded QPSK',
                learning_rate=.002,optimizer='AdamW',weight_decay=1e-4,
                note='no transmitter, radar, EW, adversarial signals or hardware')
    with open(os.path.join(args.out,'run_config.json'),'w') as f:json.dump(config,f,indent=2)
    print('P1C DONE',args.out,flush=True)
    for split in TEST_PROFILES:
        print('SUMMARY',split,'10dB',flush=True)
        for row in summary:
            if row['split']==split and row['snr_db']==10:
                print(' ',row['method'],round(row['pooled_ber'],6),flush=True)

if __name__=='__main__':main()
