#!/usr/bin/env python3
"""P1-E new-training-seed replication for a benign, synthetic, receive-only OFDM benchmark.

Uses only ordinary Gaussian thermal noise and static multipath. No radio
transmitters, radar, active RF countermeasures or real-world threat models.

Train frozen model checkpoints first with:
python p1c_architecture.py --steps 750 --batch-size 64 --eval-frames 2000 \
    --seeds 404 505 606 707 808 --arches circular_cnn transformer \
    --encodings raw_iq_magnitude logamp_phase --out p1e_train
Then run:
python p1e_replication.py --checkpoints p1e_train --out p1e_results
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import statistics
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from ofdm_baseline import make_batch, covariance_matrix, estimate_channel, decisions_from_channel
from p1c_architecture import TRAIN_PROFILE, model_factory, write_csv

SEEDS=(404,505,606,707,808)
ARCHES=('circular_cnn','transformer')
ENCODINGS=('raw_iq_magnitude','logamp_phase')
NSC=32
PROFILES={'train3':TRAIN_PROFILE,'equal4':(0.25,)*4,
          'long6':(0.40,0.22,0.15,0.10,0.08,0.05)}
SNRS=(0,10,20)
BASELINES=('ls','dft_train3','lmmse_train3','lmmse_matched_profile_prior','true_channel_reference')
PRIMARY=('equal4','long6')
CNN='circular_cnn__logamp_phase'
TRANS='transformer__logamp_phase'


def load_models(checkpoints,seed):
    models={}
    for arch in ARCHES:
        for enc in ENCODINGS:
            name=f'{arch}__{enc}'
            path=Path(checkpoints)/f'model_{name}_seed{seed}.pt'
            obj=torch.load(path, map_location='cpu',weights_only=True)
            for field,expected in [('arch',arch),('encoding',enc),('seed',seed),('nsc',NSC),('steps',750)]:
                if obj[field]!=expected:
                    raise ValueError(f'Checkpoint {path}: {field} mismatch')
            model=model_factory(arch,enc,NSC)
            model.load_state_dict(obj['state_dict'], strict=True)
            model.eval()
            models[name]=model
    return models


def eval_one(models, seed, profile_name, snr_db, frames, output_frame_differences=False):
    profile=PROFILES[profile_name]
    pi=list(PROFILES).index(profile_name)
    # P1-E test RNG is deliberately disjoint from P1-C/P1-D test generators.
    test_seed=90000000+seed*1000+pi*100+snr_db
    torch.manual_seed(test_seed)
    prior_train=covariance_matrix(NSC,TRAIN_PROFILE,'cpu')
    prior_test=covariance_matrix(NSC,profile,'cpu')
    names=[*models,*BASELINES]
    errors={name:0 for name in names}
    counts={}
    frame_differences=[]
    totalbits=0
    with torch.inference_mode():
        for off in range(0,frames,128):
            b=make_batch(min(128,frames-off),NSC,snr_db,snr_db,'cpu',profile)
            decisions={name:(m(b.yp,b.yd,b.noise_var)>0) for name,m in models.items()}
            hs={
                'ls':estimate_channel(b.yp,b.noise_var,'ls'),
                'dft_train3':estimate_channel(b.yp,b.noise_var,'dft',taps=len(TRAIN_PROFILE)),
                'lmmse_train3':estimate_channel(b.yp,b.noise_var,'lmmse',covariance=prior_train),
                'lmmse_matched_profile_prior':estimate_channel(b.yp,b.noise_var,'lmmse',covariance=prior_test),
                'true_channel_reference':b.h,
            }
            decisions.update({name:decisions_from_channel(b.yd,h,b.noise_var).bool() for name,h in hs.items()})
            truth=b.bits.bool()
            for name,pred in decisions.items():
                if pred.shape!=truth.shape:raise AssertionError((name,pred.shape,truth.shape))
                errors[name]+=int((pred!=truth).sum().item())
            if output_frame_differences:
                e_c=(decisions[CNN]!=truth).sum((1,2))
                e_t=(decisions[TRANS]!=truth).sum((1,2))
                frame_differences.extend(int(v) for v in (e_t-e_c).tolist())
            totalbits+=truth.numel()
    assert totalbits==frames*NSC*2
    rows=[dict(seed=seed,profile=profile_name,snr_db=snr_db,method=method,
               errors=errors[method],bits=totalbits,ber=errors[method]/totalbits,test_rng=test_seed)
          for method in names]
    return rows,frame_differences


def summary(rows):
    by=defaultdict(list)
    for row in rows:by[(row['profile'],row['snr_db'],row['method'])].append(row)
    out=[]
    for (profile,snr,method), records in sorted(by.items()):
        err=sum(r['errors'] for r in records)
        n=sum(r['bits'] for r in records)
        x=[r['ber'] for r in records]
        out.append(dict(profile=profile,snr_db=snr,method=method,seeds=len(x),
                        errors=err,bits=n,pooled_ber=err/n,mean_seed_ber=statistics.mean(x),
                        sd_seed_ber=statistics.stdev(x) if len(x)>1 else 0.0))
    return out


def exact_two_sided_signflip(xs):
    observed=abs(sum(xs))
    count=0
    for signs in itertools.product((-1,1),repeat=len(xs)):
        if abs(sum(x*s for x,s in zip(xs,signs)))>=observed-1e-10:
            count+=1
    return count/(2**len(xs))


def primary_results(rows):
    idx={(r['seed'],r['profile'],r['snr_db'],r['method']):r for r in rows}
    out=[]
    try:
        from scipy.stats import t
        t975=float(t.ppf(.975,len(SEEDS)-1))
    except ImportError:
        t975=2.7764451051977987 # t(4), two sided 95%
    for p in PRIMARY:
        xs=[100*(idx[s,p,10,TRANS]['ber']-idx[s,p,10,CNN]['ber']) for s in SEEDS]
        mean=statistics.mean(xs)
        sd=statistics.stdev(xs)
        half=t975*sd/math.sqrt(len(xs))
        out.append(dict(profile=p,snr_db=10,n_seeds=len(xs),
                        seed_delta_ber_pp=[round(v,6) for v in xs],
                        mean_delta_ber_pp=mean,
                        interval_95_pp=[mean-half,mean+half],
                        signflip_p_two_sided=exact_two_sided_signflip(xs),
                        expectation='positive delta favors CNN'))
    ordered=sorted(range(len(out)),key=lambda i:out[i]['signflip_p_two_sided'])
    cum=0.0
    for rank,i in enumerate(ordered):
        corrected=min(1.0,(len(out)-rank)*out[i]['signflip_p_two_sided'])
        cum=max(cum,corrected)
        out[i]['holm_adjusted_p']=cum
    return out


def timed_models(models,repeats=300):
    rows=[]
    with torch.inference_mode():
        for batch_frames in (1,128):
            torch.manual_seed(73004+batch_frames)
            b=make_batch(batch_frames,NSC,10,10,'cpu',TRAIN_PROFILE)
            for key,model in models.items():
                for _ in range(20):_=model(b.yp,b.yd,b.noise_var)
                samples=[]
                for _ in range(repeats):
                    start=time.perf_counter_ns()
                    _=model(b.yp,b.yd,b.noise_var)
                    samples.append((time.perf_counter_ns()-start)/1e6)
                samples.sort()
                rows.append(dict(method=key,batch_frames=batch_frames,
                                 trials=repeats,median_ms=statistics.median(samples),
                                 p95_ms=samples[math.ceil(.95*repeats)-1],
                                 mean_ms=statistics.mean(samples),
                                 parameters=sum(p.numel() for p in model.parameters()),
                                 note='cpu forward including input feature extraction; no simulator or device I/O'))
    return rows


def validate_simulator():
    torch.manual_seed(777)
    b=make_batch(1024,NSC,10,10,'cpu',TRAIN_PROFILE)
    observed_pilot_noise=b.yp-b.h
    empirical_power=float(observed_pilot_noise.abs().square().mean())
    if abs(empirical_power-.1)>.012:raise AssertionError(f'noise calibration {empirical_power}')
    if not torch.isfinite(b.yp).all():raise AssertionError('non-finite data')
    cv=covariance_matrix(NSC,TRAIN_PROFILE,'cpu')
    if (cv-cv.conj().T).abs().max()>1e-5:raise AssertionError('covariance not Hermitian')
    # Cross-check LMMSE matrix solve using independent Fourier-domain tap shrinkage.
    for profile in PROFILES.values():
        p=torch.tensor(profile,dtype=torch.float32)
        p/=p.sum()
        cov=covariance_matrix(NSC,profile,'cpu')
        ref=estimate_channel(b.yp[:8],b.noise_var[:8],'lmmse',covariance=cov)
        gain=p[None,:]/(p[None,:]+b.noise_var[:8]/NSC)
        taps=torch.fft.ifft(b.yp[:8],dim=-1)[:,:len(profile)]*gain
        independent=torch.fft.fft(F.pad(taps,(0,NSC-len(profile))),dim=-1)
        diff=float((ref-independent).abs().max())
        if diff>5e-5:raise AssertionError(f'LMMSE cross-check {diff}')
    return dict(pilot_noise_variance=empirical_power,lmmse_independent_formula='PASS',
                expected_noise_variance=.1,baseband_input_finite=True)


def csv_load(path):
    with open(path,newline='') as f:return list(csv.DictReader(f))


def verify_reloaded_seed(path,checkpoints,frames):
    reference=csv_load(path)
    expected={(r['profile'],int(r['snr_db']),r['method']): (int(r['errors']),int(r['bits']))
              for r in reference if int(r['seed'])==SEEDS[0]}
    models=load_models(checkpoints,SEEDS[0])
    actual={}
    for profile in PROFILES:
        for snr in SNRS:
            rows,_=eval_one(models,SEEDS[0],profile,snr,frames)
            actual.update({(r['profile'],r['snr_db'],r['method']):(r['errors'],r['bits']) for r in rows})
    if expected!=actual:
        diff=[k for k in actual if actual.get(k)!=expected.get(k)]
        raise AssertionError(f'Reloaded checkpoint produced mismatches: {diff[:10]}')
    return dict(seed=SEEDS[0],tested_scenarios=len(PROFILES)*len(SNRS),
                exact_matching_cells=len(expected),total_cells=len(actual),status='PASS')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoints',default='p1e_train')
    ap.add_argument('--out',default='p1e_results')
    ap.add_argument('--frames',type=int,default=2048)
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--verify',action='store_true')
    args=ap.parse_args()
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    sanity=validate_simulator()
    print('SIMULATOR AND INDEPENDENT LMMSE CHECK PASS',sanity,flush=True)
    if args.smoke:return
    if args.verify:
        result=verify_reloaded_seed(out/'ber_by_seed.csv',args.checkpoints,args.frames)
        (out/'replay_verification.json').write_text(json.dumps(result,indent=2)+'\n')
        print('REPLAY PASS',result,flush=True)
        return
    model_files=sorted(Path(args.checkpoints).glob('model_*_seed*.pt'))
    if len(model_files)!=len(SEEDS)*len(ARCHES)*len(ENCODINGS):
        raise AssertionError(f'Expected 20 checkpoints, got {len(model_files)}')
    checksums={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in model_files}
    all_rows=[]
    frame_rows=[]
    for seed in SEEDS:
        models=load_models(args.checkpoints,seed)
        print('EVALUATE NEW SEED',seed,flush=True)
        for profile in PROFILES:
            for snr in SNRS:
                rows,diffs=eval_one(models,seed,profile,snr,args.frames,
                                    output_frame_differences=(profile in PRIMARY and snr==10))
                all_rows.extend(rows)
                for index,diff in enumerate(diffs):
                    frame_rows.append(dict(seed=seed,profile=profile,snr_db=snr,
                                           frame=index,transformer_minus_cnn_bit_errors=diff))
            print('  profile',profile,'complete',flush=True)
    write_csv(out/'ber_by_seed.csv',all_rows)
    write_csv(out/'ber_summary.csv',summary(all_rows))
    write_csv(out/'paired_frame_deltas.csv',frame_rows)
    primary=primary_results(all_rows)
    (out/'primary_endpoints.json').write_text(json.dumps(primary,indent=2)+'\n')
    (out/'checkpoint_sha256.json').write_text(json.dumps(checksums,indent=2)+'\n')
    (out/'simulator_crosschecks.json').write_text(json.dumps(sanity,indent=2)+'\n')
    timings=timed_models(load_models(args.checkpoints,SEEDS[0]),repeats=300)
    write_csv(out/'latency_cpu.csv',timings)
    config=dict(seeds=SEEDS,steps=750,eval_frames=args.frames,profiles=PROFILES,snrs=SNRS,
                cpu_threads=torch.get_num_threads(),torch_version=torch.__version__,
                python_version=platform.python_version(),platform=platform.platform(),
                models=[f'{a}__{e}' for a in ARCHES for e in ENCODINGS],
                source='P1-C frozen training settings, novel training seeds and test streams',
                method='receive-only synthetic civilian QPSK / simple Gaussian static multipath')
    (out/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
    print('PRIMARY RESULTS',json.dumps(primary,indent=2),flush=True)
    print('COMPLETED',len(all_rows),'result rows',len(frame_rows),'paired frames',flush=True)

if __name__=='__main__':main()
