#!/usr/bin/env python3
"""Frozen P3-B physical receive-only OFDM benchmark (no real RF signals)."""
import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from p3b_time_link import (N,PAYLOAD_MASK,PROFILES,ALL_CONDITIONS,AXES,draw_base,link)
from p3b_receivers import (ENCODINGS,REGIMES,DataResidualHead,new_models,load_frozen_d,
                           classical_logits,bit_count,phase_aligned,logits_from_channel,
                           lmmse_training_mix_fft,equalized)

SEEDS=(48101,48102,48103,48104,48105)
SNRS=(0,10,20)
FRAMES=768
STEPS=600
BATCH=64
PROFILE_TRAIN=('train3','equal4','long6')
OLD_D_SEEDS=(47101,47102,47103,47104,47105)
CLASSICAL=('ls','dft8','dft12','threshold','train_lmmse','frozen_d',
           'threshold_no_align','dd_threshold')
NEURAL=tuple(f'{regime}__{enc}' for regime in REGIMES for enc in ENCODINGS)
METHODS=CLASSICAL+NEURAL


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(2**20),b''):h.update(part)
    return h.hexdigest()


def csvout(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    keys=list(rows[0])
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def load_d_for_seed(seed,old_dir):
    old=OLD_D_SEEDS[SEEDS.index(seed)]
    return load_frozen_d(old_dir/f'model_D_no_pos_10_seed{old}.pt')


def data_seed(seed,profile,snr,part):
    pi=list(PROFILES).index(profile)
    return 410000000+seed*100000+pi*10000+snr*100+part


def impairment_seed(seed,profile,snr,condition,part):
    return 1190000000+SEEDS.index(seed)*1000000+list(PROFILES).index(profile)*10000+snr*1000+ALL_CONDITIONS.index(condition)*100+part


def train_seed(seed,old_dir,checkpoints,results,steps=STEPS,batch=BATCH):
    checkpoints.mkdir(parents=True,exist_ok=True)
    results.mkdir(parents=True,exist_ok=True)
    frozen=load_d_for_seed(seed,old_dir)
    models=new_models(seed)
    opts={k:torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=1e-4) for k,m in models.items()}
    g=torch.Generator(device='cpu').manual_seed(603000000+seed)
    prng=random.Random(700400000+seed)
    start=time.perf_counter()
    last={}
    for step in range(steps):
        p=PROFILE_TRAIN[prng.randrange(3)]
        base=draw_base(batch,PROFILES[p],(0,20),g)
        roll=prng.random()
        if roll<.40:condition='clean'
        else:condition=f'{AXES[min(3,int((roll-.40)/.15))]}_mild'
        clean=link(base,'clean',include_stats=False)
        mixed=link(base,condition,extra_seed=970000000+seed*1000+step,include_stats=False)
        with torch.no_grad():
            h_clean=frozen.channel(clean.yp,clean.yd,clean.noise_var)
            h_mixed=frozen.channel(mixed.yp,mixed.yd,mixed.noise_var)
        for key,model in models.items():
            regime=key.split('__')[0]
            observed,h=(clean,h_clean) if regime=='clean' else (mixed,h_mixed)
            opt=opts[key]
            opt.zero_grad(set_to_none=True)
            logits=model(observed.yd,h,observed.noise_var)
            loss=F.binary_cross_entropy_with_logits(
                logits[:,PAYLOAD_MASK,:],observed.bits[:,PAYLOAD_MASK,:].float())
            if not torch.isfinite(loss):raise FloatingPointError((seed,key,step,'loss'))
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError((seed,key,step,'grad'))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            last[key]=float(loss.detach())
        if step in (0,steps-1) or (step+1)%200==0:
            print('TRAIN',seed,step+1,'/',steps,'elapsed',round(time.perf_counter()-start,1),flush=True)
    records=[]
    protocol=sha256(Path(__file__).with_name('P3B_PROTOCOL.md'))
    for key,model in models.items():
        file=checkpoints/f'model_{key}_seed{seed}.pt'
        torch.save({'key':key,'seed':seed,'steps':steps,'batch':batch,
            'training_data_seed':603000000+seed,'choice_seed':700400000+seed,
            'frozen_D_seed':OLD_D_SEEDS[SEEDS.index(seed)],'protocol_sha256':protocol,
            'state_dict':model.state_dict()},file)
        records.append(dict(seed=seed,model=key,params=sum(p.numel() for p in model.parameters()),
          steps=steps,batch=batch,final_loss=last[key],sec_total=round(time.perf_counter()-start,3),
          checkpoint=file.name,sha256=sha256(file),protocol_sha256=protocol))
    csvout(results/f'training_seed{seed}.csv',records)
    return records


def load_models(seed,checkpoint,steps=STEPS,batch=BATCH):
    out={}
    for name in NEURAL:
        f=checkpoint/f'model_{name}_seed{seed}.pt'
        obj=torch.load(f,map_location='cpu',weights_only=True)
        for k,v in [('key',name),('seed',seed),('steps',steps),('batch',batch)]:
            if obj[k]!=v:raise ValueError((f,k,obj[k],v))
        m=DataResidualHead(name.split('__',1)[1]);m.load_state_dict(obj['state_dict'],strict=True);m.eval()
        out[name]=m
    return out


def eval_condition(seed,profile,snr,condition,models,frozen,frames=FRAMES):
    errors={m:0 for m in METHODS}
    squares={m:0.0 for m in METHODS}
    ch_mse={m:0.0 for m in CLASSICAL if m not in ('threshold_no_align','dd_threshold')}
    metrics=defaultdict(float)
    total=0
    for part,start in enumerate(range(0,frames,128)):
        n=min(128,frames-start)
        g=torch.Generator().manual_seed(data_seed(seed,profile,snr,part))
        b=draw_base(n,PROFILES[profile],snr,g)
        observed=link(b,condition,extra_seed=impairment_seed(seed,profile,snr,condition,part),include_stats=True)
        with torch.inference_mode():
            classical,hs=classical_logits(observed,frozen)
            h=hs['frozen_d']
            preds=dict(classical)
            preds.update({k:m(observed.yd,h,observed.noise_var) for k,m in models.items()})
            for k,logit in preds.items():
                errors[k]+=bit_count(logit,observed.bits)
                # scale-aligned constellation proxy, not a calibrated EVM claim
                z=torch.complex(logit[:,PAYLOAD_MASK,0],logit[:,PAYLOAD_MASK,1])/3
                x=b.xdata[:,PAYLOAD_MASK]
                alpha=((z*x.conj()).real.sum(dim=-1,keepdim=True)/30).clamp(min=0.)
                squares[k]+=float(((z-alpha*x).abs().square().sum()).item())
            for k in ch_mse:
                ch_mse[k]+=float((hs[k]-b.h).abs().square().sum().item())
        for key,val in observed.stats.items():metrics[key]+=n*val
        total+=n
    return [dict(seed=seed,profile=profile,snr_db=snr,condition=condition,
          method=k,errors=errors[k],bits=total*60,frames=total,
          ber=errors[k]/(total*60),scale_aligned_symbol_mse=squares[k]/(total*30),
          channel_mse=(ch_mse[k]/(total*N) if k in ch_mse else ''),
          **{a:metrics[a]/total for a in ('tx_in_power','tx_out_power','tx_power_ratio','papr',
             'burst_rate','cpe_abs_mean','adc_saturation','total_noise_var')})
         for k in METHODS]


def evaluate(seed,ckpt,olddir,out,frames=FRAMES,steps=STEPS,batch=BATCH):
    models=load_models(seed,ckpt,steps,batch)
    frozen=load_d_for_seed(seed,olddir)
    rows=[]
    for p in PROFILES:
        for snr in SNRS:
            for condition in ALL_CONDITIONS:
                rows.extend(eval_condition(seed,p,snr,condition,models,frozen,frames))
        print('EVAL',seed,p,'complete',len(rows),'rows',flush=True)
    csvout(out/f'eval_seed{seed}.csv',rows)
    return rows


def summarize(rows):
    grouped=defaultdict(list)
    for r in rows:grouped[(r['profile'],int(r['snr_db']),r['condition'],r['method'])].append(r)
    output=[]
    for (p,snr,cond,method),group in sorted(grouped.items()):
        n=sum(int(x['bits']) for x in group)
        err=sum(int(x['errors']) for x in group)
        ber=[float(x['ber']) for x in group]
        out=dict(profile=p,snr_db=snr,condition=cond,method=method,
                 seeds=len(group),errors=err,bits=n,ber=err/n,
                 mean_seed_ber=statistics.mean(ber),sd_seed_ber=statistics.stdev(ber) if len(ber)>1 else 0,
                 scale_aligned_symbol_mse=statistics.mean(float(x['scale_aligned_symbol_mse']) for x in group),
                 channel_mse=(statistics.mean(float(x['channel_mse']) for x in group) if group[0]['channel_mse']!='' else ''),
                 tx_power_ratio=statistics.mean(float(x['tx_power_ratio']) for x in group),
                 adc_saturation=statistics.mean(float(x['adc_saturation']) for x in group),
                 burst_rate=statistics.mean(float(x['burst_rate']) for x in group),
                 cpe_abs_mean=statistics.mean(float(x['cpe_abs_mean']) for x in group))
        output.append(out)
    return output


def timing(seed,ckpt,old_dir,repeats=120):
    models=load_models(seed,ckpt)
    frozen=load_d_for_seed(seed,old_dir)
    g=torch.Generator().manual_seed(60606)
    base=draw_base(1,PROFILES['train3'],10,g)
    b=link(base)
    h=frozen.channel(b.yp,b.yd,b.noise_var)
    table=[]
    for name,m in list(models.items())+[('frozen_d',frozen)]:
        with torch.inference_mode():
            def once():
                if name=='frozen_d':return frozen.channel(b.yp,b.yd,b.noise_var)
                return m(b.yd,h,b.noise_var)
            for _ in range(20):once()
            x=[]
            for _ in range(repeats):
                st=time.perf_counter_ns();once();x.append((time.perf_counter_ns()-st)/1e6)
        x.sort()
        table.append(dict(method=name,params=sum(p.numel() for p in m.parameters()),
           median_ms=statistics.median(x),p95_ms=x[math.ceil(.95*repeats)-1],
           note='CPU inference, head-only for N1; frozen backbone extra; no SDR/device I/O'))
    with torch.inference_mode():
        x=[]
        for _ in range(repeats):
            st=time.perf_counter_ns();classical_logits(b,frozen);x.append((time.perf_counter_ns()-st)/1e6)
        table.append(dict(method='all_classical_baselines_combined',params=0,
                          median_ms=statistics.median(x),p95_ms=sorted(x)[math.ceil(.95*repeats)-1],
                          note='combined classical reference calculation, not one single baseline'))
    return table


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=['train','eval','verify','summary','timing','smoke'],required=True)
    ap.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    ap.add_argument('--frames',type=int,default=FRAMES)
    ap.add_argument('--steps',type=int,default=STEPS)
    ap.add_argument('--batch',type=int,default=BATCH)
    ap.add_argument('--old-dir',type=Path,default=Path('p3a2_checkpoints'))
    ap.add_argument('--ckpt',type=Path,default=Path('p3b_checkpoints'))
    ap.add_argument('--out',type=Path,default=Path('p3b_results'))
    args=ap.parse_args();torch.set_num_threads(4)
    if args.mode=='smoke':
        from test_p3b_link import TimeLinkTests
        import unittest
        out=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TimeLinkTests))
        if not out.wasSuccessful():raise SystemExit(1)
        return
    args.out.mkdir(parents=True,exist_ok=True)
    if args.mode=='train':
        for s in args.seeds:train_seed(s,args.old_dir,args.ckpt,args.out,args.steps,args.batch)
    if args.mode=='eval':
        for s in args.seeds:evaluate(s,args.ckpt,args.old_dir,args.out,args.frames,args.steps,args.batch)
    if args.mode in ('summary','verify'):
        rows=[]
        for s in args.seeds:
            with (args.out/f'eval_seed{s}.csv').open() as f:rows.extend(csv.DictReader(f))
        if args.mode=='summary':
            csvout(args.out/'ber_by_seed.csv',rows)
            csvout(args.out/'ber_summary.csv',summarize(rows))
            print('SUMMARY',len(rows),'raw cells',len(summarize(rows)),'aggregate cells',flush=True)
        else:
            for s in args.seeds:
                models=load_models(s,args.ckpt,args.steps,args.batch)
                frozen=load_d_for_seed(s,args.old_dir)
                lookup={(int(r['seed']),r['profile'],int(r['snr_db']),r['condition'],r['method']):
                        (int(r['errors']),int(r['bits'])) for r in rows if int(r['seed'])==s}
                count=0
                for p,snr,cond in [('train3',10,'clean'),('long6',10,'pa_mild'),('long12',10,'impulse_strong')]:
                    for r in eval_condition(s,p,snr,cond,models,frozen,args.frames):
                        key=(s,p,snr,cond,r['method'])
                        assert lookup[key]==(r['errors'],r['bits']),(key,r['errors'],lookup[key])
                        count+=1
                print('REPLAY',s,count,'integer error+bit records match',flush=True)
    if args.mode=='timing':csvout(args.out/'latency_cpu.csv',timing(args.seeds[0],args.ckpt,args.old_dir))

if __name__=='__main__':main()
