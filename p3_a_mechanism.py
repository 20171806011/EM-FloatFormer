#!/usr/bin/env python3
"""P3-A2: frozen 2x2 positional-input and residual-floor causal ablation.

Only generated, receive-only, civilian QPSK/OFDM. No emitters, radar,
active interference, transmit-control or real-world acquisition.
The frozen protocol was committed to GitHub before experiments:
P3_A_MECHANISM_PROTOCOL.md. Does not modify P3-A or P2-E datasets.
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from ofdm_baseline import (make_batch, covariance_matrix, estimate_channel,
                           decisions_from_channel)
from p3_models import EquivDelayConv
from p3_a import (TRAIN, LONG12, SPARSE12, dft_threshold)

N=32
SEEDS=(47101,47102,47103,47104,47105)
PROFILES={'train3': TRAIN, 'equal4':(.25,)*4,
          'long6':(.40,.22,.15,.10,.08,.05),
          'long12':LONG12,'sparse12':SPARSE12}
SNRS=(0,10,20)
ARMS={'A_pos_0': (True,0.00), 'B_no_pos_0':(False,0.00),
      'C_pos_10':(True,0.10),'D_no_pos_10':(False,0.10)}
BASELINES=('ls','dft8','dft12','dft_thresh_hard','dft_thresh_soft',
           'lmmse_train3','lmmse_matched_privileged','true_channel_inaccessible')
METHODS=(*ARMS,*BASELINES)
EXPECTED_PROTOCOL_SHA256='2955db97ee5203240cb76f30e068f0d12e4bf140909ecd8cdff9a4ae70aadc06'
GITHUB_FROZEN_PROTOCOL_SHA256='db3721cb5e3511f21d3ea4adafd92810becd2cd3ce0f375eb42057a2e94aab68'

def verify_frozen_protocol(path):
    """Accept the frozen GitHub upload (no terminal LF) OR exact original local LF.

    The pre-training GitHub connector dropped *only* the terminal newline.
    This byte-level transport difference cannot change the frozen experiment.
    Do NOT accept arbitrary edits to the scientific protocol.
    """
    raw=Path(path).read_bytes()
    h=hashlib.sha256(raw).hexdigest()
    if h==EXPECTED_PROTOCOL_SHA256:return h
    if h==GITHUB_FROZEN_PROTOCOL_SHA256 and hashlib.sha256(raw+b'\n').hexdigest()==EXPECTED_PROTOCOL_SHA256:
        return h
    raise RuntimeError('Frozen P3-A2 protocol content changed')

class FactorialDelayConv(EquivDelayConv):
    """Original 985-param P3 EquivDelayConv with only two controlled factors.

    use_position=False zeros the two coordinate *inputs*; parameters persist.
    floor=0.10 enforces effective gain >=0.1 with no extra parameters.
    """
    def __init__(self, use_position=True, floor=0.0, nsc=32, width=12):
        super().__init__(nsc=nsc,width=width)
        if floor not in (0.0,0.10):raise ValueError(floor)
        self.use_position=bool(use_position)
        self.floor=float(floor)

    def raw_gate(self, yp, nv):
        n=yp.shape[-1]
        if n!=self.nsc:raise ValueError(f'expected {self.nsc}, got {n}')
        t=torch.fft.ifft(yp,dim=-1)
        power=t.abs().square()
        mean=power.mean(dim=-1,keepdim=True)
        noise_t=nv/n
        k=torch.arange(n,device=yp.device,dtype=power.dtype)[None,:]
        phase=2*math.pi*k/n
        if self.use_position:
            cos=phase.cos().expand_as(power)
            sin=phase.sin().expand_as(power)
        else:
            cos=torch.zeros_like(power)
            sin=torch.zeros_like(power)
        f=torch.stack((
            torch.log1p(power/(noise_t+1e-7)),
            torch.log1p(power/(mean+1e-7)),
            torch.log1p(mean/(noise_t+1e-7)).expand_as(power),
            nv.log10().expand_as(power),
            cos,sin
        ),dim=1)
        x=F.silu(self.stem(f))
        x=x+0.5*F.silu(self.conv1(x))
        x=x+0.5*F.silu(self.conv2(x))
        return torch.sigmoid(self.out(x)).squeeze(1)

    def effective_gate(self,yp,nv):
        g=self.raw_gate(yp,nv)
        return g if self.floor==0 else self.floor+(1-self.floor)*g

    def channel(self,yp,yd,nv):
        del yd
        t=torch.fft.ifft(yp,dim=-1)
        return torch.fft.fft(t*self.effective_gate(yp,nv),dim=-1)


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for blob in iter(lambda: f.read(1024*1024),b''):h.update(blob)
    return h.hexdigest()


def init_shared(seed):
    torch.manual_seed((271828*seed+137)%(2**31))
    base=EquivDelayConv(nsc=N)
    state=copy.deepcopy(base.state_dict())
    out={}
    for name,(pos,beta) in ARMS.items():
        model=FactorialDelayConv(pos,beta,nsc=N)
        model.load_state_dict(copy.deepcopy(state),strict=True)
        out[name]=model
    return out


def save_csv(path,rows):
    if not rows:return
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def test_data_seed(seed,pname,snr):
    return 361000000+seed*1000+list(PROFILES).index(pname)*100+snr


def train_one(seed,steps,batch,ckpt,results):
    ckpt.mkdir(parents=True,exist_ok=True)
    results.mkdir(parents=True,exist_ok=True)
    models=init_shared(seed)
    optimizers={name:torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=1e-4)
                for name,m in models.items()}
    data_seed=84300000+seed
    torch.manual_seed(data_seed)
    begin=time.perf_counter()
    last={}
    for step in range(1,steps+1):
        b=make_batch(batch,N,0,20,'cpu',TRAIN)
        for name,m in models.items():
            m.train();optimizers[name].zero_grad(set_to_none=True)
            logits=m(b.yp,b.yd,b.noise_var)
            loss=F.binary_cross_entropy_with_logits(logits,b.bits.float())
            if not torch.isfinite(loss):raise FloatingPointError((seed,name,step,'loss'))
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):
                raise FloatingPointError((seed,name,step,'gradient'))
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            optimizers[name].step()
            last[name]=float(loss.detach())
        if step in (1,steps) or step%200==0:
            print('TRAIN',seed,step,'/',steps,'elapsed',round(time.perf_counter()-begin,2),'s',flush=True)
    records=[]
    for name,m in models.items():
        path=ckpt/f'model_{name}_seed{seed}.pt'
        torch.save(dict(name=name,seed=seed,steps=steps,batch=batch,
                 use_position=m.use_position,floor=m.floor,
                 data_seed=data_seed,profile=TRAIN,
                 protocol_sha256=EXPECTED_PROTOCOL_SHA256,
                 state_dict=m.state_dict()),path)
        records.append(dict(seed=seed,arm=name,params=sum(p.numel() for p in m.parameters()),
             active_params=sum(p.numel() for p in m.parameters())-(0 if m.use_position else 24),
             position=m.use_position,floor=m.floor,steps=steps,batch=batch,
             training_data_seed=data_seed,final_bce=last[name],
             total_seed_training_sec=round(time.perf_counter()-begin,3),
             checkpoint=path.name,sha256=sha256(path)))
    save_csv(results/f'training_seed{seed}.csv',records)
    return records


def load_models(seed,src,steps=600,batch=64):
    models={}
    for name,(pos,beta) in ARMS.items():
        path=src/f'model_{name}_seed{seed}.pt'
        if not path.exists():raise FileNotFoundError(path)
        o=torch.load(path,map_location='cpu',weights_only=True)
        for key,val in [('name',name),('seed',seed),('steps',steps),('batch',batch),
                        ('use_position',pos),('floor',beta),
                        ('protocol_sha256',EXPECTED_PROTOCOL_SHA256)]:
            if o[key]!=val:raise ValueError((path,key,o[key],val))
        m=FactorialDelayConv(pos,beta)
        m.load_state_dict(o['state_dict'],strict=True)
        models[name]=m.eval()
    return models


def baseline_channels(b,train_cov,test_cov):
    yp,nv=b.yp,b.noise_var
    return dict(ls=yp, dft8=estimate_channel(yp,nv,'dft',taps=8),
        dft12=estimate_channel(yp,nv,'dft',taps=12),
        dft_thresh_hard=dft_threshold(yp,nv,'hard'),
        dft_thresh_soft=dft_threshold(yp,nv,'soft'),
        lmmse_train3=estimate_channel(yp,nv,'lmmse',covariance=train_cov),
        lmmse_matched_privileged=estimate_channel(yp,nv,'lmmse',covariance=test_cov),
        true_channel_inaccessible=b.h)


def eval_condition(models,seed,pname,snr,frames):
    profile=PROFILES[pname]
    torch.manual_seed(test_data_seed(seed,pname,snr))
    cov_train=covariance_matrix(N,TRAIN,'cpu')
    cov_test=covariance_matrix(N,profile,'cpu')
    errors={name:0 for name in METHODS}
    sse={name:0.0 for name in METHODS}
    nmse_sum={name:0.0 for name in METHODS}
    aligned_sse={name:0.0 for name in METHODS}
    nbits=0
    with torch.inference_mode():
        for off in range(0,frames,128):
            b=make_batch(min(128,frames-off),N,snr,snr,'cpu',profile)
            chans=baseline_channels(b,cov_train,cov_test)
            for name,m in models.items():chans[name]=m.channel(b.yp,b.yd,b.noise_var)
            for name,h in chans.items():
                err=(decisions_from_channel(b.yd,h,b.noise_var)!=b.bits).sum()
                errors[name]+=int(err.item())
                per_frame=(h-b.h).abs().square().sum(dim=-1)
                power=b.h.abs().square().sum(dim=-1)
                sse[name]+=float(per_frame.sum())
                nmse_sum[name]+=float((per_frame/(power+1e-12)).sum())
                # Diagnostic only: optimal per-frame positive real amplitude correction.
                a=torch.clamp((h.conj()*b.h).real.sum(dim=1)/(h.abs().square().sum(dim=1)+1e-12),min=0)
                aligned_sse[name]+=float(((a[:,None]*h-b.h).abs().square().sum()))
            nbits+=b.bits.numel()
    assert nbits==frames*N*2
    return [dict(seed=seed,profile=pname,snr_db=snr,method=name,
                 errors=errors[name],bits=nbits,ber=errors[name]/nbits,
                 channel_mse=sse[name]/(frames*N),
                 nmse=nmse_sum[name]/frames,
                 aligned_channel_mse=aligned_sse[name]/(frames*N),
                 test_data_seed=test_data_seed(seed,pname,snr)) for name in METHODS]


def eval_all(seeds,ckpt,out,frames,steps,batch):
    out.mkdir(parents=True,exist_ok=True)
    for seed in seeds:
        models=load_models(seed,ckpt,steps,batch)
        rows=[]
        for profile in PROFILES:
            for snr in SNRS:
                rows.extend(eval_condition(models,seed,profile,snr,frames))
        save_csv(out/f'eval_seed{seed}.csv',rows)
        print('EVAL',seed,len(rows),'method × profile × SNR rows',flush=True)
    merge_eval(out,seeds,frames,steps,batch)


def merge_eval(out,seeds,frames,steps,batch):
    rows=[]
    for seed in seeds:
        with (out/f'eval_seed{seed}.csv').open() as f: rows+=list(csv.DictReader(f))
    save_csv(out/'ber_by_seed.csv',rows)
    summ=[]
    for p in PROFILES:
        for snr in SNRS:
            for name in METHODS:
                items=[r for r in rows if r['profile']==p and int(r['snr_db'])==snr and r['method']==name]
                if len(items)!=len(seeds):raise ValueError((p,snr,name,len(items)))
                errs=sum(int(r['errors']) for r in items)
                bits=sum(int(r['bits']) for r in items)
                summ.append(dict(profile=p,snr_db=snr,method=name,n_seeds=len(seeds),
                                 errors=errs,bits=bits,pooled_ber=errs/bits,
                                 mean_seed_ber=statistics.mean(float(r['ber']) for r in items),
                                 sd_seed_ber=statistics.stdev(float(r['ber']) for r in items),
                                 mean_channel_mse=statistics.mean(float(r['channel_mse']) for r in items),
                                 mean_nmse=statistics.mean(float(r['nmse']) for r in items),
                                 mean_aligned_mse=statistics.mean(float(r['aligned_channel_mse']) for r in items)))
    save_csv(out/'ber_summary.csv',summ)
    contrasts=[]
    for p in ('long12','sparse12'):
        for seed in seeds:
            d={r['method']:100*float(r['ber']) for r in rows if int(r['seed'])==seed and r['profile']==p and int(r['snr_db'])==10}
            a,b,c,e=[d[k] for k in ARMS]
            contrasts.append(dict(seed=seed,profile=p,snr_db=10,
                                  A=a,B=b,C=c,D=e,B_minus_A_pp=b-a,
                                  C_minus_A_pp=c-a,D_minus_A_pp=e-a,
                                  interaction_pp=(e-c)-(b-a)))
    save_csv(out/'factorial_by_seed.csv',contrasts)
    json_config=dict(protocol_sha256=EXPECTED_PROTOCOL_SHA256,seeds=list(seeds),steps=steps,
                     batch=batch,frames=frames,nsc=N,training_profile=TRAIN,profiles=PROFILES,
                     snrs=SNRS,arms=ARMS,methods=METHODS,
                     data_eval_seed_formula='361000000+seed*1000+profile_idx*100+snr',
                     train_data_seed_formula='84300000+seed',
                     torch_version=str(torch.__version__),python=platform.python_version(),
                     cpu_threads=torch.get_num_threads(),
                     notes='P3-A2 separately frozen exploratory ablation; not P2-E or P3-A retroactive result; simulated passive only')
    (out/'run_config.json').write_text(json.dumps(json_config,indent=2,ensure_ascii=False))
    return rows,summ,contrasts


def diagnostic_gates(models,seed,pname,snr,frames):
    torch.manual_seed(467000000+seed*1000+list(PROFILES).index(pname)*100+snr)
    per_arm={name:[] for name in ARMS}
    true_powers=[]
    with torch.inference_mode():
        for off in range(0,frames,128):
            b=make_batch(min(128,frames-off),N,snr,snr,'cpu',PROFILES[pname])
            true_powers.append(torch.fft.ifft(b.h,dim=-1).abs().square().cpu())
            for name,m in models.items():
                per_arm[name].append(m.effective_gate(b.yp,b.noise_var).detach().cpu())
    pt=torch.cat(true_powers,dim=0)
    power_by_tap=pt.mean(dim=0)
    out=[]
    for name,values in per_arm.items():
        x=torch.cat(values,dim=0)
        signal_wtd=(x.square()*pt).sum()/(pt.sum()+1e-12)
        late_mask=torch.arange(N)>=8
        late_pt=pt[:,late_mask]
        late_x=x[:,late_mask]
        late_ret=float((late_x.square()*late_pt).sum()/(late_pt.sum()+1e-12)) if late_pt.sum()>1e-10 else None
        for k in range(N):
            out.append(dict(seed=seed,profile=pname,snr_db=snr,arm=name,delay_index=k,
                            mean_gain=float(x[:,k].mean()),
                            p10_gain=float(torch.quantile(x[:,k],.10)),
                            median_gain=float(x[:,k].median()),
                            p90_gain=float(torch.quantile(x[:,k],.90)),
                            true_tap_power=float(power_by_tap[k]),
                            global_true_energy_retention=float(signal_wtd),
                            true_late_energy_retention=late_ret))
    return out


def latency(models,repeats=160):
    torch.manual_seed(66700000)
    rows=[]
    for batch in (1,64):
        b=make_batch(batch,N,10,10,'cpu',TRAIN)
        for name in ARMS:
            m=models[name]
            with torch.inference_mode():
                for _ in range(20):m(b.yp,b.yd,b.noise_var)
                ts=[]
                for _ in range(repeats):
                    start=time.perf_counter_ns();m(b.yp,b.yd,b.noise_var)
                    ts.append((time.perf_counter_ns()-start)/1e6)
            ts.sort()
            rows.append(dict(arm=name,params=sum(p.numel() for p in m.parameters()),
                             batch_frames=batch,repetitions=repeats,
                             median_ms=statistics.median(ts),p95_ms=ts[math.ceil(.95*repeats)-1],
                             note='PyTorch CPU model forward only; no SDR, acquisition, framer or real receiver'))
    return rows


def verify_one(seed,ckpt,out,frames,steps,batch):
    recorded=list(csv.DictReader((out/'ber_by_seed.csv').open()))
    lookup={(int(r['seed']),r['profile'],int(r['snr_db']),r['method']):(int(r['errors']),int(r['bits'])) for r in recorded}
    models=load_models(seed,ckpt,steps,batch)
    checked=0
    for p,snr in [('train3',10),('long6',10),('long12',10),('sparse12',10)]:
        for x in eval_condition(models,seed,p,snr,frames):
            assert lookup[(seed,p,snr,x['method'])]==(x['errors'],x['bits']),(seed,p,snr,x['method'])
            checked+=1
    return checked


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['train','eval','replay','diagnose','smoke'],required=True)
    ap.add_argument('--seed',type=int,default=SEEDS[0])
    ap.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    ap.add_argument('--steps',type=int,default=600)
    ap.add_argument('--batch',type=int,default=64)
    ap.add_argument('--frames',type=int,default=768)
    ap.add_argument('--ckpt',type=Path,default=Path('p3a2_checkpoints'))
    ap.add_argument('--out',type=Path,default=Path('p3a2_results'))
    args=ap.parse_args();torch.set_num_threads(4)
    verify_frozen_protocol(Path(__file__).with_name('P3_A_MECHANISM_PROTOCOL.md'))
    if args.mode=='smoke':
        from test_p3_a_mechanism import run_tests
        run_tests()
    elif args.mode=='train':
        for seed in args.seeds:train_one(seed,args.steps,args.batch,args.ckpt,args.out)
    elif args.mode=='eval':eval_all(args.seeds,args.ckpt,args.out,args.frames,args.steps,args.batch)
    elif args.mode=='replay':print('REPLAY VERIFIED',verify_one(args.seed,args.ckpt,args.out,args.frames,args.steps,args.batch))
    else:
        rows=[]
        for seed in args.seeds:
            models=load_models(seed,args.ckpt,args.steps,args.batch)
            for p in PROFILES:
                rows.extend(diagnostic_gates(models,seed,p,10,args.frames))
        save_csv(args.out/'gate_by_delay.csv',rows)
        save_csv(args.out/'latency_cpu.csv',latency(load_models(args.seeds[0],args.ckpt,args.steps,args.batch)))
        print('DIAGNOSTIC',len(rows),'delay rows',flush=True)

if __name__=='__main__':main()
