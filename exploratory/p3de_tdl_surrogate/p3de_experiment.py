#!/usr/bin/env python3
"""New-seed controlled, offline, civilian OFDM study on an independent
TDL-table *surrogate*. Not NVIDIA Sionna, not a 3GPP compliance test.

No real RF equipment, signal interception, radar, or electronic warfare.
"""
import argparse, copy, csv, hashlib, itertools, json, math, statistics, time
from pathlib import Path
from collections import defaultdict
import torch
from torch.nn import functional as F
from p3de_profile_bridge import N,SCENARIOS,TRAIN_SCENARIOS,TEST_SCENARIOS,SNRS,draw,covariance,baseline_channel,decisions,audit_math
from p3_a_mechanism import FactorialDelayConv

SEEDS=tuple(range(61001,61011))
MODELS=('A_pos_0','D_no_pos_10')
BASELINES=('LS','DFT8','DFT12','Threshold','Train-LMMSE','Test-LMMSE','TrueH')
METHODS=MODELS+BASELINES
PRIMARY=('test_C_1000ns','test_D_750ns')
STEPS=300
BATCH=64
FRAMES=2048

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def csvwrite(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def init_models(seed):
    torch.manual_seed(seed*3+101)
    base=FactorialDelayConv(True,0)
    common=copy.deepcopy(base.state_dict())
    models={}
    for name,pos,floor in [('A_pos_0',True,0.),('D_no_pos_10',False,.1)]:
        m=FactorialDelayConv(pos,floor);m.load_state_dict(copy.deepcopy(common));models[name]=m
    return models

def train_one(seed,out,steps=STEPS,batch=BATCH):
    out=Path(out);(out/'checkpoints').mkdir(parents=True,exist_ok=True)
    models=init_models(seed)
    opt={name:torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.0001) for name,m in models.items()}
    gen=torch.Generator().manual_seed(900000+seed)
    t0=time.perf_counter(); last={}
    for step in range(steps):
        profile=TRAIN_SCENARIOS[step%len(TRAIN_SCENARIOS)]
        b=draw(batch,profile,(0,20),gen)
        for name,m in models.items():
            m.train();opt[name].zero_grad(set_to_none=True)
            logits=m(b.yp,b.yd,b.noise_var)
            loss=F.binary_cross_entropy_with_logits(logits,b.bits.float())
            if not torch.isfinite(loss):raise FloatingPointError((seed,step,name,'loss'))
            loss.backward()
            if not all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):
                raise FloatingPointError((seed,step,name,'gradient'))
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
            opt[name].step();last[name]=float(loss.detach())
    records=[]
    for name,m in models.items():
        fp=out/'checkpoints'/f'{name}_seed{seed}.pt'
        torch.save({'name':name,'seed':seed,'steps':steps,'batch':batch,
                    'training_scenarios':list(TRAIN_SCENARIOS),
                    'state_dict':m.cpu().state_dict()},fp)
        records.append({'seed':seed,'model':name,'params':sum(p.numel() for p in m.parameters()),
                        'steps':steps,'batch':batch,'last_bce':last[name],
                        'train_pair_seconds':time.perf_counter()-t0,'checkpoint_sha256':sha(fp)})
    return records

def load_models(seed,out,steps=STEPS,batch=BATCH):
    models=init_models(seed)
    for name,m in models.items():
        fp=Path(out)/'checkpoints'/f'{name}_seed{seed}.pt'
        obj=torch.load(fp,map_location='cpu',weights_only=True)
        assert obj['name']==name and obj['seed']==seed and obj['steps']==steps and obj['batch']==batch
        m.load_state_dict(obj['state_dict'],strict=True);m.eval()
    return models

def eval_seed_scenario(seed,scenario,snr,models,frames=FRAMES):
    gen=torch.Generator().manual_seed(157000000+seed*1000+list(TEST_SCENARIOS).index(scenario)*100+snr)
    cov_train=.5*(covariance(TRAIN_SCENARIOS[0])+covariance(TRAIN_SCENARIOS[1]))
    cov_test=covariance(scenario)
    n0=10.**(-snr/10.)
    eye=torch.eye(N,dtype=torch.complex64)
    W_train=cov_train @ torch.linalg.inv(cov_train+n0*eye)
    W_test=cov_test @ torch.linalg.inv(cov_test+n0*eye)
    errors={name:0 for name in METHODS};mse={name:0. for name in METHODS}
    n_bits=0;n_chan=0
    with torch.inference_mode():
        for offset in range(0,frames,128):
            b=draw(min(128,frames-offset),scenario,snr,gen)
            estimates={name:m.channel(b.yp,b.yd,b.noise_var) for name,m in models.items()}
            estimates['LS']=b.yp
            estimates['DFT8']=baseline_channel(b.yp,b.noise_var,'DFT8')
            estimates['DFT12']=baseline_channel(b.yp,b.noise_var,'DFT12')
            estimates['Threshold']=baseline_channel(b.yp,b.noise_var,'Threshold')
            estimates['Train-LMMSE']=b.yp @ W_train.T
            estimates['Test-LMMSE']=b.yp @ W_test.T
            estimates['TrueH']=b.h
            for name in METHODS:
                pred=decisions(b.yd,estimates[name],b.noise_var)
                errors[name]+=int((pred!=b.bits.bool()).sum())
                mse[name]+=float((estimates[name]-b.h).abs().square().sum())
            n_bits+=b.bits.numel();n_chan+=b.h.numel()
    assert n_bits==frames*N*2
    return [dict(seed=seed,scenario=scenario,snr_db=snr,method=name,errors=errors[name],bits=n_bits,
                 ber=errors[name]/n_bits,channel_mse=mse[name]/n_chan,
                 privilege='privileged' if name in ('Test-LMMSE','TrueH') else 'nonprivileged') for name in METHODS]

def eval_all(out,steps=STEPS,batch=BATCH,frames=FRAMES,seeds=SEEDS):
    rows=[]
    for seed in seeds:
        models=load_models(seed,out,steps,batch)
        for scenario in TEST_SCENARIOS:
            for snr in SNRS:
                rows.extend(eval_seed_scenario(seed,scenario,snr,models,frames))
        print('EVAL',seed,'rows',len(rows),flush=True)
    csvwrite(Path(out)/'ber_by_seed.csv',rows)
    aggregate=[]
    groups=defaultdict(list)
    for row in rows:groups[(row['scenario'],row['snr_db'],row['method'])].append(row)
    for (scenario,snr,method),group in sorted(groups.items()):
        err=sum(x['errors'] for x in group);bits=sum(x['bits'] for x in group)
        aggregate.append(dict(scenario=scenario,snr_db=snr,method=method,
                              pooled_ber=err/bits,errors=err,bits=bits,
                              seeds=len(group),mean_channel_mse=statistics.mean(x['channel_mse'] for x in group),
                              privilege=group[0]['privilege']))
    csvwrite(Path(out)/'ber_summary.csv',aggregate)
    return rows

def primary(rows):
    by={(r['seed'],r['scenario'],r['snr_db'],r['method']):r for r in rows}
    out=[]
    for scenario in PRIMARY:
        differences=[100*(by[s,scenario,10,'Threshold']['ber']-by[s,scenario,10,'D_no_pos_10']['ber']) for s in SEEDS]
        ob=abs(sum(differences));count=0
        for sign in itertools.product((-1,1),repeat=len(SEEDS)):
            if abs(sum(x*v for x,v in zip(differences,sign)))>=ob-1e-12:count+=1
        p=count/2**len(SEEDS)
        mean=statistics.mean(differences)
        sd=statistics.stdev(differences)
        from scipy.stats import t
        q=float(t.ppf(.975,len(SEEDS)-1))
        half=q*sd/math.sqrt(len(SEEDS))
        out.append(dict(scenario=scenario,contrast='Threshold BER − D BER (pp)',
                        mean_delta_pp=mean,seed_delta_pp=differences,
                        ci95_t_pp=[mean-half,mean+half],exact_twosided_p=p))
    order=sorted(range(len(out)),key=lambda i:out[i]['exact_twosided_p']);prev=0.
    for rank,i in enumerate(order):
        p=min(1.,(len(out)-rank)*out[i]['exact_twosided_p']);prev=max(prev,p)
        out[i]['holm_p']=prev
    return out

def benchmark(out,seed=SEEDS[0]):
    models=load_models(seed,out)
    gen=torch.Generator().manual_seed(1255595)
    results=[]
    with torch.inference_mode():
        for frames in (1,64):
            b=draw(frames,TEST_SCENARIOS[0],10,gen)
            for name in ('A_pos_0','D_no_pos_10','Threshold','DFT8'):
                if name in models:
                    fn=lambda:models[name].channel(b.yp,b.yd,b.noise_var)
                else:
                    fn=lambda:baseline_channel(b.yp,b.noise_var,name)
                for _ in range(10):fn()
                durations=[]
                for _ in range(120):
                    t0=time.perf_counter_ns();fn();durations.append((time.perf_counter_ns()-t0)/1e6)
                durations.sort()
                results.append(dict(method=name,batch_frames=frames,median_ms=statistics.median(durations),
                                    p95_ms=durations[math.ceil(.95*len(durations))-1],trials=120))
    csvwrite(Path(out)/'latency_cpu.csv',results)

def audit(out,steps=STEPS,batch=BATCH,frames=FRAMES):
    audit_math()
    out=Path(out)
    with (out/'ber_by_seed.csv').open() as f:rows=list(csv.DictReader(f))
    with (out/'ber_summary.csv').open() as f:group=list(csv.DictReader(f))
    assert len(rows)==len(SEEDS)*len(TEST_SCENARIOS)*len(SNRS)*len(METHODS),(len(rows),len(group))
    assert len(group)==len(TEST_SCENARIOS)*len(SNRS)*len(METHODS)
    lookup={(int(x['seed']),x['scenario'],int(x['snr_db']),x['method']):(int(x['errors']),int(x['bits'])) for x in rows}
    checked=0
    for seed in (SEEDS[0],SEEDS[-1]):
        models=load_models(seed,out,steps,batch)
        for scenario in PRIMARY:
            for x in eval_seed_scenario(seed,scenario,10,models,frames):
                key=(seed,scenario,10,x['method'])
                assert lookup[key]==(x['errors'],x['bits']),key
                checked+=1
    for x in group:
        s,c,m=x['scenario'],int(x['snr_db']),x['method']
        corresponding=[r for r in rows if r['scenario']==s and int(r['snr_db'])==c and r['method']==m]
        assert sum(int(r['errors']) for r in corresponding)==int(x['errors'])
        assert sum(int(r['bits']) for r in corresponding)==int(x['bits'])
    checkpoints=list((out/'checkpoints').glob('*.pt'))
    assert len(checkpoints)==2*len(SEEDS)
    registry=[]
    for fp in sorted(checkpoints):registry.append(dict(filename=fp.name,sha256=sha(fp)))
    (out/'checkpoint_sha256.json').write_text(json.dumps(registry,indent=2))
    return {'pass':True,'direct_integer_replays':checked,'raw_rows':len(rows),
            'aggregate_rows':len(group),'checkpoint_count':len(checkpoints),
            'profile_bridge':'independent non-Sionna TDL 2.2 JSON transcription',
            'sionna_runtime_available':False}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('train','eval','finalize','smoke'),required=True)
    parser.add_argument('--out',default='p3de_results')
    parser.add_argument('--steps',type=int,default=STEPS)
    parser.add_argument('--batch',type=int,default=BATCH)
    parser.add_argument('--frames',type=int,default=FRAMES)
    args=parser.parse_args();torch.set_num_threads(4)
    if args.mode=='smoke':
        audit_math();g=torch.Generator().manual_seed(12)
        b=draw(8,'train_A_500ns',10,g)
        models=init_models(61001)
        for name,model in models.items():
            pred=model(b.yp,b.yd,b.noise_var)
            assert pred.shape==b.bits.shape and torch.isfinite(pred).all()
            F.binary_cross_entropy_with_logits(pred,b.bits.float()).backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        print('SMOKE PASS')
    elif args.mode=='train':
        training=[]
        for seed in SEEDS:
            training.extend(train_one(seed,args.out,args.steps,args.batch))
            print('TRAIN',seed,'complete',flush=True)
        csvwrite(Path(args.out)/'training_registry.csv',training)
    elif args.mode=='eval':
        rows=eval_all(args.out,args.steps,args.batch,args.frames)
        (Path(args.out)/'primary_endpoints.json').write_text(json.dumps(primary(rows),indent=2))
        benchmark(args.out)
    else:
        report=audit(args.out,args.steps,args.batch,args.frames)
        (Path(args.out)/'independent_audit.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))

if __name__=='__main__':main()
