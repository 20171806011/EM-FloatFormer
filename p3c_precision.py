#!/usr/bin/env python3
"""P3-C: frozen-checkpoint CPU mixed-native real-Conv precision and IQ-storage audit.

No real RF, emitting hardware, military uses, or active intentional interference.
Reuses P3-B simulated passive civilian QPSK/OFDM and frozen weights, no training.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from contextlib import nullcontext

import torch
from torch import nn
from p3b_time_link import N, PROFILES, PAYLOAD_MASK, draw_base, link, ALL_CONDITIONS
from p3b_train_eval import (SEEDS, OLD_D_SEEDS, load_models, load_d_for_seed,
                            STEPS, BATCH)
from p3b_receivers import (classical_logits, logits_from_channel,
                           lmmse_training_mix_fft, DataResidualHead)

PROJECT=Path(__file__).parent
P3C_PROFILES=('train3','equal4','long6','sparse12')
SNRS=(0,10,20)
CONDITIONS=('clean','pa_strong','impulse_strong','adc_strong')
N_FRAMES=256
EVAL_MODES=('fp32','fp16_realconv','bf16_realconv')
STORAGE_PROFILES=P3C_PROFILES
STORAGE_CONDITIONS=('clean','adc_strong')
STORAGE_MODES=('iq_fp16_fp32','iq_bf16_fp32','iq_fp16_fp16_realconv','iq_bf16_bf16_realconv')
GAIN_EXPONENTS=(-12,0,12)
REF_CLASSICAL=('threshold_no_align','dft8','train_lmmse')
EXPECTED_PROTOCOL_SHA='e1c2f1e0cc8bb0b4a50b515bae40d6daeb62fee8e5ac2974afa3178c9d66371a'


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()


def ensure_protocol():
    p=PROJECT/'P3C_PROTOCOL.md'
    found=sha256(p)
    if found!=EXPECTED_PROTOCOL_SHA:raise AssertionError(f'P3-C protocol bytes changed! {found}')
    return found


def inspect_capabilities():
    a=torch.randn(1,4)
    caps={'torch':torch.__version__,'python':platform.python_version(),'platform':platform.platform(),
          'cuda_available':bool(torch.cuda.is_available()),'mps_available':bool(torch.backends.mps.is_available()),
          'threads':torch.get_num_threads()}
    for dtype in (torch.float16,torch.bfloat16):
        key=str(dtype).split('.')[-1]
        with torch.inference_mode(), torch.autocast(device_type='cpu',dtype=dtype):
            conv=nn.Conv1d(2,3,1);x=conv(torch.randn(1,2,8))
        caps[key+'_cpu_autocast_conv_dtype']=str(x.dtype)
        try:
            f=torch.fft.fft(torch.randn(8,dtype=dtype))
            caps[key+'_cpu_fft']={'supported':True,'dtype':str(f.dtype)}
        except Exception as e:
            caps[key+'_cpu_fft']={'supported':False,'error':type(e).__name__+': '+str(e)[:200]}
    return caps


def models_for(seed):
    frozen=load_d_for_seed(seed,PROJECT/'p3a2_checkpoints')
    heads=load_models(seed,PROJECT/'p3b_checkpoints',STEPS,BATCH)
    return frozen,heads


def test_seed(seed,profile,snr,part):
    return 731000000 + SEEDS.index(seed)*10000000 + P3C_PROFILES.index(profile)*200000 + snr*5000 + part


def impairment_seed(seed,profile,snr,cond,part):
    return 911000000 + SEEDS.index(seed)*10000000 + P3C_PROFILES.index(profile)*200000 + snr*5000 + CONDITIONS.index(cond)*100 + part


def get_observed(seed,profile,snr,condition,part,frames=N_FRAMES):
    n=min(128,frames-128*part)
    g=torch.Generator(device='cpu').manual_seed(test_seed(seed,profile,snr,part))
    base=draw_base(n,PROFILES[profile],snr,g)
    return link(base,condition,extra_seed=impairment_seed(seed,profile,snr,condition,part),include_stats=False)


def quantize_received(received,dtype):
    def q_complex(z):
        return torch.complex(z.real.to(dtype).to(torch.float32),z.imag.to(dtype).to(torch.float32))
    return replace(received,yp=q_complex(received.yp),yd=q_complex(received.yd))


def autocast_for(mode):
    if 'fp16_realconv' in mode:return torch.autocast(device_type='cpu',dtype=torch.float16)
    if 'bf16_realconv' in mode:return torch.autocast(device_type='cpu',dtype=torch.bfloat16)
    return nullcontext()


def transform_observed(received,mode):
    if mode.startswith('iq_fp16_'):return quantize_received(received,torch.float16)
    if mode.startswith('iq_bf16_'):return quantize_received(received,torch.bfloat16)
    return received


def safe_predictions(logits,truth):
    target=truth[:,PAYLOAD_MASK,:]
    q=logits[:,PAYLOAD_MASK,:]
    finite=torch.isfinite(q)
    unknown=int(torch.count_nonzero(~finite).item())
    if unknown:return (None,unknown,None)
    pred=(q>0)
    err=int(torch.count_nonzero(pred!=target.bool()).item())
    return (err,unknown,pred)


def forward_methods(obs,frozen,heads,mode,include_classical=False):
    with torch.inference_mode(),autocast_for(mode):
        h=frozen.channel(obs.yp,obs.yd,obs.noise_var)
        logits={k:m(obs.yd,h,obs.noise_var) for k,m in heads.items()}
    if include_classical:
        # Classical comparisons use the same quantized pilot/data (where requested),
        # but remain numerically FP32. Do not autocast them.
        with torch.inference_mode():
            c,_=classical_logits(obs,frozen)
            logits.update({k:c[k] for k in REF_CLASSICAL})
    return logits,h


def build_rows_for_case(seed,profile,snr,cond,frozen,heads,frames=N_FRAMES,
                        modes=EVAL_MODES,kind='operator',gain_exponent=None):
    acc={}
    for mode in modes:
        for model in list(heads)+(list(REF_CLASSICAL) if (mode.startswith('iq_') or kind in ('input_storage','coherent_gain')) else []):
            acc[mode,model]={'errors':0,'nonfinite':0,'flip':0,'logit_mean_acc':0.,
                              'logit_elements':0,'logit_max_abs':0.,'max_h_abs':0.,'h_samples':0,
                              'input_nonfinite':0,'total':0}
    for part in range(math.ceil(frames/128)):
        orig=get_observed(seed,profile,snr,cond,part,frames)
        if gain_exponent is not None:
            scalar=float(2.**gain_exponent)
            orig=replace(orig,yp=orig.yp*scalar,yd=orig.yd*scalar,noise_var=orig.noise_var*(scalar**2))
        reference,href=forward_methods(orig,frozen,heads,'fp32',include_classical=(kind in ('input_storage','coherent_gain') or any(m.startswith('iq_') for m in modes)))
        for mode in modes:
            obs=transform_observed(orig,mode)
            preds,h=forward_methods(obs,frozen,heads,mode,include_classical=(mode.startswith('iq_') or kind in ('input_storage','coherent_gain')))
            orig_nonfinite=int((~torch.isfinite(obs.yp)).sum().item()+(~torch.isfinite(obs.yd)).sum().item())
            for name,logits in preds.items():
                state=acc[mode,name]
                nb=orig.bits[:,PAYLOAD_MASK,:].numel()
                state['total']+=nb
                state['input_nonfinite']+=orig_nonfinite
                err,nonfinite,hard=safe_predictions(logits,orig.bits)
                if err is not None and state['errors'] is not None:state['errors']+=err
                else:state['errors']=None
                state['nonfinite']+=nonfinite
                if name in reference:
                    ref=reference[name][:,PAYLOAD_MASK,:]
                    pred=logits[:,PAYLOAD_MASK,:]
                    both=torch.isfinite(ref)&torch.isfinite(pred)
                    if int(both.sum()) and state['errors'] is not None:
                        dif=(pred[both].float()-ref[both].float()).abs()
                        state['logit_mean_acc']+=float(dif.sum())
                        state['logit_elements']+=int(both.sum())
                        state['logit_max_abs']=max(state['logit_max_abs'],float(dif.max()))
                        state['flip']+=int(torch.count_nonzero((pred[both]>0)!=(ref[both]>0)).item())
                if name in heads:
                    hf=(h-href).abs()
                    good=hf[torch.isfinite(hf)]
                    if good.numel():state['max_h_abs']=max(state['max_h_abs'],float(good.max()))
                    state['h_samples']+=int(hf.numel())
    rows=[]
    for (mode,name),stat in sorted(acc.items()):
        record=dict(kind=kind,seed=seed,profile=profile,snr_db=snr,condition=cond,
                    gain_exponent='' if gain_exponent is None else gain_exponent,
                    mode=mode,method=name,encoding=name.rsplit('__',1)[-1] if '__' in name else '',
                    training_regime=name.split('__')[0] if '__' in name else '',
                    errors='' if stat['errors'] is None else stat['errors'],bits=stat['total'],
                    ber='' if stat['errors'] is None else stat['errors']/stat['total'],
                    nonfinite_logits=stat['nonfinite'],nonfinite_input=stat['input_nonfinite'],
                    hard_flips_vs_fp32=stat['flip'],
                    mean_logit_abs_delta=stat['logit_mean_acc']/max(1,stat['logit_elements']),
                    max_logit_abs_delta=stat['logit_max_abs'],
                    max_channel_abs_delta=stat['max_h_abs'],
                    reference='same new heldout frame; fp32 original input')
        rows.append(record)
    return rows


def write_csv(path,rows):
    if not rows:return
    path=Path(path);path.parent.mkdir(exist_ok=True,parents=True)
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)


def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def summarize(rows):
    agg=defaultdict(list)
    for r in rows:agg[(r['kind'],r['profile'],r['snr_db'],r['condition'],r['gain_exponent'],r['mode'],r['method'])].append(r)
    out=[]
    for key,group in sorted(agg.items()):
        errors=None if any(r['errors']=='' for r in group) else sum(int(r['errors']) for r in group)
        bits=sum(int(r['bits']) for r in group)
        out.append(dict(kind=key[0],profile=key[1],snr_db=key[2],condition=key[3],gain_exponent=key[4],
                 mode=key[5],method=key[6],seeds=len(group),
                 errors='' if errors is None else errors,bits=bits,
                 ber='' if errors is None else errors/bits,
                 total_nonfinite_logits=sum(int(r['nonfinite_logits']) for r in group),
                 total_nonfinite_input=sum(int(r['nonfinite_input']) for r in group),
                 total_hard_flips=sum(int(r['hard_flips_vs_fp32']) for r in group),
                 max_logit_abs_delta=max(float(r['max_logit_abs_delta']) for r in group),
                 max_channel_abs_delta=max(float(r['max_channel_abs_delta']) for r in group),
                 mean_of_seed_mean_logit_abs_delta=statistics.mean(float(r['mean_logit_abs_delta']) for r in group)))
    return out


def representation_stress():
    # Mathematical reference uses FP64 and exact inverse of the stated encodings.
    # Values are not actual radio observations and not fed into a learned model.
    angles=torch.linspace(0.,2*math.pi,steps=32,dtype=torch.float64)
    mags=torch.logspace(-8,8,steps=32,dtype=torch.float64)
    rs=[]
    for i in range(33):
        # high intra-frame dynamic range and a frame with a single exact zero.
        re=mags.roll(i)*angles.cos().roll(3*i%32)
        im=mags.roll(i)*angles.sin().roll(i%32)
        if i==32:re[0]=0.;im[0]=0.
        rs.append(torch.complex(re,im))
    z=torch.stack(rs)
    r=z.abs(); eps=1e-12; sigma=1e-3
    s=(r.square().mean(dim=-1,keepdim=True)+eps*eps).sqrt()
    encs={
        'raw_cartesian': (torch.stack([z.real,z.imag],-1),lambda q:torch.complex(q[...,0],q[...,1])),
        'log_phase': (torch.stack([torch.log1p(r/sigma),z.real/(r+eps),z.imag/(r+eps)],-1),
                      lambda q: torch.complex(q[...,1],q[...,2])*(sigma*torch.expm1(q[...,0])+eps)),
        'rms_log': (torch.stack([z.real/s,z.imag/s,s.log().expand_as(r)],-1),
                    lambda q:torch.complex(q[...,0],q[...,1])*torch.exp(q[...,2]))}
    rows=[]
    for enc,(e,inv) in encs.items():
        for dtype in (torch.float16,torch.bfloat16):
            q=e.to(dtype).to(torch.float64)
            z2=inv(q)
            valid=torch.isfinite(z2)
            abs_err=(z2-z).abs()
            nonzero=(z.abs()>0)&valid
            rel=(abs_err[nonzero]/z.abs()[nonzero]).numpy()
            rows.append(dict(encoding=enc,storage=str(dtype).split('.')[-1],samples=z.numel(),
                 finite=int(valid.sum()),nonfinite=int((~valid).sum()),
                 finite_ratio=float(valid.float().mean()),
                 rel_error_median=float(__import__('numpy').median(rel)) if rel.size else '',
                 rel_error_p95=float(__import__('numpy').quantile(rel,.95)) if rel.size else '',
                 rel_error_max=float(rel.max()) if rel.size else '',
                 zero_samples=int((z.abs()==0).sum()),
                 source_min_magnitude=float(z.abs()[z.abs()>0].min()),
                 source_max_magnitude=float(z.abs().max())))
    return rows


def profile_operators():
    seed=SEEDS[0]
    frozen,heads=models_for(seed)
    obs=get_observed(seed,'train3',10,'clean',0,1)
    report=[]
    for mode in EVAL_MODES:
        output_dtypes=defaultdict(list)
        hooks=[]
        def hook_cb(name):
            def inner(module,args,result):output_dtypes[name].append(str(result.dtype))
            return inner
        for prefix,model in [('frozen_d',frozen),('mixed__raw',heads['mixed__raw'])]:
            for name,module in model.named_modules():
                if isinstance(module,nn.Conv1d):hooks.append(module.register_forward_hook(hook_cb(prefix+'.'+name)))
        with torch.inference_mode(),autocast_for(mode):
            h=frozen.channel(obs.yp,obs.yd,obs.noise_var)
            y=heads['mixed__raw'](obs.yd,h,obs.noise_var)
        for hh in hooks:hh.remove()
        recorded={k:sorted(set(v)) for k,v in output_dtypes.items()}
        expected='torch.float32' if mode=='fp32' else ('torch.float16' if mode=='fp16_realconv' else 'torch.bfloat16')
        assert recorded and all(v==[expected] for v in recorded.values()), (mode,recorded)
        assert h.dtype==torch.complex64 and y.dtype==torch.float32
        report.append(dict(mode=mode,conv_dtypes=recorded,fft_output_dtype=str(h.dtype),logit_dtype=str(y.dtype),
                           note='native CPU autocast real Conv1d only; FFT remains complex64'))
    return report


def timing(seed,repeat=100):
    frozen,heads=models_for(seed)
    recorded=[]
    for batch in (1,64):
        g=torch.Generator().manual_seed(73470000+batch)
        base=draw_base(batch,PROFILES['equal4'],10,g)
        obs=link(base,'clean',include_stats=False)
        for mode in EVAL_MODES:
            for method in ('mixed__raw','mixed__log_phase','mixed__rms_log'):
                model=heads[method]
                def work():
                    with torch.inference_mode(),autocast_for(mode):
                        h=frozen.channel(obs.yp,obs.yd,obs.noise_var)
                        return model(obs.yd,h,obs.noise_var)
                for _ in range(20):work()
                durations=[]
                for _ in range(repeat):
                    a=time.perf_counter_ns();work();durations.append((time.perf_counter_ns()-a)/1e6)
                durations.sort()
                recorded.append(dict(method=method,mode=mode,batch_frames=batch,repeats=repeat,
                                     median_ms=statistics.median(durations),
                                     p95_ms=durations[math.ceil(.95*repeat)-1],
                                     note='CPU frozen backbone + real Conv residual head, excludes signal acquisition/simulator'))
    return recorded


def validate_checkpoints():
    names=[]
    for seed in SEEDS:
        frozen,heads=models_for(seed)
        for name in heads:
            path=PROJECT/'p3b_checkpoints'/f'model_{name}_seed{seed}.pt'
            obj=torch.load(path,map_location='cpu',weights_only=True)
            assert obj['seed']==seed and obj['key']==name and obj['steps']==600
            names.append(dict(seed=seed,method=name,sha256=sha256(path),bytes=path.stat().st_size))
        old=OLD_D_SEEDS[SEEDS.index(seed)]
        path=PROJECT/'p3a2_checkpoints'/f'model_D_no_pos_10_seed{old}.pt'
        obj=torch.load(path,map_location='cpu',weights_only=True)
        assert obj['seed']==old and obj['name']=='D_no_pos_10' and obj['floor']==.10
        names.append(dict(seed=old,method='frozen_d',sha256=sha256(path),bytes=path.stat().st_size))
    return names


def evaluate(out,frames=N_FRAMES,seeds=SEEDS):
    ensure_protocol()
    torch.set_num_threads(4)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    inventory=validate_checkpoints();write_csv(out/'checkpoint_sha256.csv',inventory)
    capabilities=inspect_capabilities()
    (out/'hardware_capabilities.json').write_text(json.dumps(capabilities,indent=2,ensure_ascii=False)+'\n')
    (out/'operator_dtypes.json').write_text(json.dumps(profile_operators(),indent=2)+'\n')
    write_csv(out/'representation_stress.csv',representation_stress())
    for seed in seeds:
        frozen,heads=models_for(seed)
        rows=[]
        for profile in P3C_PROFILES:
            for snr in SNRS:
                for cond in CONDITIONS:
                    rows.extend(build_rows_for_case(seed,profile,snr,cond,frozen,heads,frames,EVAL_MODES,'operator'))
            print('OPERATORS',seed,profile,'done',flush=True)
        write_csv(out/f'operator_seed{seed}.csv',rows)
        store=[]
        for profile in STORAGE_PROFILES:
            for cond in STORAGE_CONDITIONS:
                store.extend(build_rows_for_case(seed,profile,10,cond,frozen,heads,frames,
                         ('fp32',)+STORAGE_MODES,'input_storage'))
        write_csv(out/f'storage_seed{seed}.csv',store)
        gain=[]
        for e in GAIN_EXPONENTS:
            gain.extend(build_rows_for_case(seed,'equal4',10,'clean',frozen,heads,frames,
                     ('fp32','iq_fp16_fp16_realconv','iq_bf16_bf16_realconv'),'coherent_gain',e))
        write_csv(out/f'gain_seed{seed}.csv',gain)
        print('SEED',seed,'ALL COMPLETE',len(rows),len(store),len(gain),flush=True)
    all_rows=[]
    for kind in ('operator','storage','gain'):
        for seed in seeds:
            all_rows.extend(read_csv(out/f'{kind}_seed{seed}.csv'))
    write_csv(out/'raw_by_seed.csv',all_rows)
    write_csv(out/'summary.csv',summarize(all_rows))
    write_csv(out/'latency_cpu.csv',timing(seeds[0]))
    print('SUMMARY',len(all_rows),'raw cells',len(summarize(all_rows)),'aggregate cells',flush=True)
    return all_rows


def replay(out,seeds=SEEDS,frames=N_FRAMES):
    # Three prespecified test cells per seed using exact new heldout RNG;
    # actual numerical error counts and different-precision decisions must match.
    out=Path(out)
    check=[('equal4',10,'clean',('fp32','fp16_realconv','bf16_realconv')),
           ('sparse12',10,'adc_strong',('fp32','fp16_realconv','bf16_realconv')),
           ('train3',10,'impulse_strong',('fp32','fp16_realconv','bf16_realconv'))]
    all_records={}
    for seed in seeds:
        for line in read_csv(out/f'operator_seed{seed}.csv'):
            all_records[(seed,line['profile'],int(line['snr_db']),line['condition'],line['mode'],line['method'])]=line
    counts=0
    for seed in seeds:
        frozen,heads=models_for(seed)
        for p,snr,c,modes in check:
            res=build_rows_for_case(seed,p,snr,c,frozen,heads,frames,modes,'operator')
            for row in res:
                key=(seed,p,snr,c,row['mode'],row['method'])
                prior=all_records[key]
                assert str(row['errors'])==prior['errors'] and int(row['bits'])==int(prior['bits'])
                assert int(row['nonfinite_logits'])==int(prior['nonfinite_logits'])
                assert int(row['hard_flips_vs_fp32'])==int(prior['hard_flips_vs_fp32'])
                counts+=1
        print('REPLAY',seed,'done',flush=True)
    return counts


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mode',choices=['evaluate','replay','audit','smoke'],required=True)
    p.add_argument('--out',type=Path,default=PROJECT/'p3c_results')
    p.add_argument('--frames',type=int,default=N_FRAMES)
    p.add_argument('--seeds',nargs='+',type=int,default=list(SEEDS))
    args=p.parse_args();torch.set_num_threads(4)
    if args.mode=='smoke':
        ensure_protocol()
        caps=inspect_capabilities()
        ops=profile_operators()
        cks=validate_checkpoints()
        print('SMOKE PASS',len(ops),'precision modes',len(cks),'verified checkpoints',caps['cuda_available'])
    elif args.mode=='evaluate':evaluate(args.out,args.frames,args.seeds)
    elif args.mode=='replay':print('REPLAY_MATCHED',replay(args.out,args.seeds,args.frames))
    else:
        from p3c_independent_audit import audit
        audit(args.out)

if __name__=='__main__':main()
