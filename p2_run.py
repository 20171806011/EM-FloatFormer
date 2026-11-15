#!/usr/bin/env python3
"""Preregistered P2 controlled, passive synthetic OFDM experiments.

No real-world RF, no transmit control, radar, warfare, or active signals.
"""
import argparse, csv, hashlib, json, math, os, platform, statistics, time
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from ofdm_baseline import make_batch, estimate_channel, covariance_matrix, decisions_from_channel
from p2_models import N_SC, MODEL_NAMES, model_factory, verify_models, ResidualBase
from p1c_architecture import TRAIN_PROFILE

PROFILES={'train3':TRAIN_PROFILE,'equal4':(.25,)*4,'long6':(.4,.22,.15,.10,.08,.05),
          'two_tap':(.7,.3),'late3':(.1,.2,.7)}
SNRS=(0,10,20)
BASELINES=('ls','dft8','lmmse_train','lmmse_matched_prior')


def model_init(name,seed):
    # Same random initial parameters for the three local residual encodings;
    # other model families get separate reproducible initializations.
    key = 'res_local' if name.startswith('res_local_') else name
    h=int(hashlib.sha256(key.encode()).hexdigest()[:7],16)
    torch.manual_seed((seed*971+h)%(2**31))
    return model_factory(name)


def train(seed,names,steps,bs,out):
    model={n:model_init(n,seed) for n in names}
    opts={n:torch.optim.AdamW(model[n].parameters(),lr=.002,weight_decay=1e-4) for n in names}
    torch.manual_seed(8100000+seed)
    start=time.perf_counter()
    for step in range(1,steps+1):
        batch=make_batch(bs,N_SC,0,20,'cpu',TRAIN_PROFILE)
        for n,m in model.items():
            m.train();opts[n].zero_grad(set_to_none=True)
            logits=m(batch.yp,batch.yd,batch.noise_var)
            loss=F.binary_cross_entropy_with_logits(logits,batch.bits.float())
            loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            opts[n].step()
        if step==1 or step%200==0 or step==steps:
            print('train',seed,step,steps,'seconds',round(time.perf_counter()-start,1),flush=True)
    for n,m in model.items():
        path=out/f'model_{n}_seed{seed}.pt'
        torch.save({'state_dict':m.state_dict(),'name':n,'seed':seed,'steps':steps,
                    'nsc':N_SC,'training_profile':TRAIN_PROFILE},path)
    return round(time.perf_counter()-start,4)


def load_models(seed,names,ckpt):
    models={}
    for n in names:
        obj=torch.load(ckpt/f'model_{n}_seed{seed}.pt',map_location='cpu',weights_only=True)
        assert obj['name']==n and obj['seed']==seed and obj['nsc']==N_SC
        m=model_factory(n);m.load_state_dict(obj['state_dict'],strict=True);m.eval()
        models[n]=m
    return models


def map_observation(b,condition,perturb_seed):
    yp,yd,nv=b.yp,b.yd,b.noise_var
    if condition=='bf16':
        yp=torch.complex(yp.real.to(torch.bfloat16).float(),yp.imag.to(torch.bfloat16).float())
        yd=torch.complex(yd.real.to(torch.bfloat16).float(),yd.imag.to(torch.bfloat16).float())
    elif condition=='nv_high':nv=nv*10**(.6)
    elif condition=='nv_low':nv=nv*10**(-.6)
    elif condition=='data_phase':
        g=torch.Generator(device='cpu');g.manual_seed(perturb_seed)
        theta=(torch.rand((yp.shape[0],1),generator=g)*2-1)*.35
        yd=yd*torch.polar(torch.ones_like(theta),theta)
    elif condition!='normal':raise ValueError(condition)
    return yp,yd,nv


def test_one(models,seed,profile,snr,frames,condition='normal',test_offset=0):
    # Separate test family and disjoint training PRNG range.
    ix=list(PROFILES).index(profile)
    perturb_ix=('normal','bf16','nv_high','nv_low','data_phase').index(condition)
    test_seed=98000000+10000*seed+100*ix+snr+test_offset+1000000*perturb_ix
    torch.manual_seed(test_seed)
    prior_train=covariance_matrix(N_SC,TRAIN_PROFILE,'cpu')
    prior_matched=covariance_matrix(N_SC,PROFILES[profile],'cpu')
    names=(*models,*BASELINES)
    err={n:0 for n in names};bits=0; mse={n:0. for n,m in models.items() if isinstance(m,ResidualBase)}
    with torch.inference_mode():
        for off in range(0,frames,128):
            b=make_batch(min(128,frames-off),N_SC,snr,snr,'cpu',PROFILES[profile])
            yp,yd,nv=map_observation(b,condition,test_seed+75000+off)
            pred={n:(m(yp,yd,nv)>0) for n,m in models.items()}
            for name,m in models.items():
                if isinstance(m,ResidualBase):
                    h=m.channel(yp,yd,nv)
                    mse[name]+=float((h-b.h).abs().square().sum().item())
            methods={
                'ls':estimate_channel(yp,nv,'ls'),
                'dft8':estimate_channel(yp,nv,'dft',taps=8),
                'lmmse_train':estimate_channel(yp,nv,'lmmse',covariance=prior_train),
                'lmmse_matched_prior':estimate_channel(yp,nv,'lmmse',covariance=prior_matched)}
            pred.update({n:decisions_from_channel(yd,h,nv).bool() for n,h in methods.items()})
            for n in names: err[n]+=int(torch.count_nonzero(pred[n]!=b.bits.bool()))
            bits+=b.bits.numel()
    return [dict(seed=seed,profile=profile,snr_db=snr,condition=condition,method=n,
                 errors=err[n],bits=bits,ber=err[n]/bits,
                 channel_mse=(mse[n]/(frames*N_SC) if n in mse else '')) for n in names]


def summarize(rows):
    group=defaultdict(list)
    for r in rows:group[r['profile'],r['snr_db'],r['condition'],r['method']].append(r)
    out=[]
    for k,vals in sorted(group.items()):
        errors=sum(v['errors'] for v in vals);bits=sum(v['bits'] for v in vals)
        means=[v['ber'] for v in vals]
        out.append(dict(profile=k[0],snr_db=k[1],condition=k[2],method=k[3],
                        n_seeds=len(means),errors=errors,bits=bits,ber=errors/bits,
                        seed_sd=statistics.stdev(means) if len(means)>1 else '',
                        mean_channel_mse=(statistics.mean(float(v['channel_mse']) for v in vals)
                             if all(v['channel_mse']!='' for v in vals) else '')))
    return out


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def timing(models,repetitions=100):
    torch.manual_seed(77999)
    out=[]
    with torch.inference_mode():
        for batchsize in (1,64):
            b=make_batch(batchsize,N_SC,10,10,'cpu',TRAIN_PROFILE)
            for n,m in models.items():
                for _ in range(12): m(b.yp,b.yd,b.noise_var)
                ts=[]
                for _ in range(repetitions):
                    t=time.perf_counter_ns();m(b.yp,b.yd,b.noise_var)
                    ts.append((time.perf_counter_ns()-t)*1e-6)
                ts.sort()
                out.append(dict(method=n,batch=batchsize,params=sum(p.numel() for p in m.parameters()),
                       repetitions=repetitions,median_ms=statistics.median(ts),p95_ms=ts[math.ceil(.95*len(ts))-1],
                       environment='CPU PyTorch 4 threads, forward only, no hardware or protocol latency'))
    return out


def eval_main(ckpt,names,seeds,frames,out,robust=False,test_offset=0):
    rows=[]
    for seed in seeds:
        models=load_models(seed,names,ckpt)
        conditions=[(profile,snr,'normal') for profile in PROFILES if robust or profile in ('train3','equal4','long6') for snr in SNRS]
        if robust:
            conditions += [(p,10,c) for p in ('train3','equal4','long6') for c in ('bf16','nv_high','nv_low','data_phase')]
        for p,s,c in conditions:
            rows.extend(test_one(models,seed,p,s,frames,c,test_offset))
        write_csv(out/'ber_by_seed.csv',rows)
        print('test',seed,'conditions',len(conditions),'complete',flush=True)
    write_csv(out/'ber_summary.csv',summarize(rows))
    write_csv(out/'latency_cpu.csv',timing(models))
    config={'names':names,'seeds':seeds,'frames':frames,'robust':robust,
      'test_offset':test_offset,'torch_version':torch.__version__,
      'python':platform.python_version(),'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
      'cpu_threads':torch.get_num_threads(),'profiles':PROFILES,
      'snrs':SNRS,'notes':'receive-only synthetic, matched LMMSE uses privileged per-profile prior'}
    (out/'run_config.json').write_text(json.dumps(config,indent=2))
    for r in summarize(rows):
        if r['snr_db']==10 and r['condition']=='normal' and r['profile'] in ('train3','equal4','long6'):
            print('RESULT',r['profile'],r['method'],round(100*r['ber'],3),'%')
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',choices=['smoke','train','eval'],required=True)
    ap.add_argument('--names',nargs='+',default=list(MODEL_NAMES),choices=MODEL_NAMES)
    ap.add_argument('--seeds',nargs='+',type=int,default=[1201,1202,1203])
    ap.add_argument('--steps',type=int,default=600)
    ap.add_argument('--batch',type=int,default=64)
    ap.add_argument('--frames',type=int,default=768)
    ap.add_argument('--out',default='p2_results')
    ap.add_argument('--ckpt',default='p2_checkpoints')
    ap.add_argument('--robust',action='store_true')
    ap.add_argument('--test-offset',type=int,default=0)
    args=ap.parse_args();torch.set_num_threads(4)
    if args.mode=='smoke':
        print('SMOKE PASS',verify_models());return
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cp=Path(args.ckpt);cp.mkdir(parents=True,exist_ok=True)
    if args.mode=='train':
        timings=[]
        for seed in args.seeds:
            elapsed=train(seed,args.names,args.steps,args.batch,cp)
            for n in args.names:
                timings.append(dict(seed=seed,model=n,seconds=elapsed,
                      params=sum(p.numel() for p in model_factory(n).parameters()),
                      steps=args.steps,batch=args.batch))
            write_csv(out/'training.csv',timings)
        (out/'training_config.json').write_text(json.dumps(dict(**vars(args),
                   torch_version=torch.__version__,threads=torch.get_num_threads()),indent=2))
    else:eval_main(cp,args.names,args.seeds,args.frames,out,args.robust,args.test_offset)

if __name__=='__main__':main()
