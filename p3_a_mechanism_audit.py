#!/usr/bin/env python3
"""Independent record reconciliation and deterministic checkpoint replay.

Runs after fixed P3-A2 evaluation. Does not retrain or change the protocol.
"""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import torch
from p3_models import EquivDelayConv
from ofdm_baseline import make_batch
from p3_a_mechanism import (ARMS, METHODS, PROFILES, SEEDS, SNRS, N,
    EXPECTED_PROTOCOL_SHA256, sha256, verify_frozen_protocol, load_models, eval_condition)


def rows(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def verify(root,ckpt,frames=768,steps=600,batch=64):
    torch.set_num_threads(4)
    verify_frozen_protocol(Path(__file__).with_name('P3_A_MECHANISM_PROTOCOL.md'))
    registry=[r for seed in SEEDS for r in rows(root/f'training_seed{seed}.csv')]
    assert len(registry)==20 and len({(x['seed'],x['arm']) for x in registry})==20
    hashes={}
    for r in registry:
        p=ckpt/r['checkpoint']
        actual=sha256(p)
        assert actual==r['sha256'],p
        hashes[p.name]=actual
        assert int(r['steps'])==steps and int(r['batch'])==batch
        assert int(r['params'])==985 and int(r['active_params']) in (985,961)
    raw=rows(root/'ber_by_seed.csv')
    assert len(raw)==len(SEEDS)*len(PROFILES)*len(SNRS)*len(METHODS)==900
    by={}
    for r in raw:
        key=(int(r['seed']),r['profile'],int(r['snr_db']),r['method'])
        assert key not in by,key
        assert int(r['bits'])==frames*N*2
        assert 0<=int(r['errors'])<=int(r['bits'])
        assert abs(float(r['ber'])-int(r['errors'])/int(r['bits']))<1e-13
        for field in ['channel_mse','nmse','aligned_channel_mse']:
            assert math.isfinite(float(r[field])) and float(r[field])>=0,(key,field)
        by[key]=r
    summary=rows(root/'ber_summary.csv')
    assert len(summary)==len(PROFILES)*len(SNRS)*len(METHODS)
    groups=defaultdict(list)
    for r in raw:groups[(r['profile'],int(r['snr_db']),r['method'])].append(r)
    for r in summary:
        g=groups[(r['profile'],int(r['snr_db']),r['method'])]
        assert len(g)==len(SEEDS)
        err=sum(int(z['errors']) for z in g);bits=sum(int(z['bits']) for z in g)
        assert err==int(r['errors']) and bits==int(r['bits'])
        assert abs(float(r['pooled_ber'])-err/bits)<1e-13
    factorial=rows(root/'factorial_by_seed.csv')
    assert len(factorial)==len(SEEDS)*2
    for r in factorial:
        seed=int(r['seed']);p=r['profile'];s=10
        x=[100*float(by[seed,p,s,a]['ber']) for a in ARMS]
        a,b,c,d=x
        expect=dict(A=a,B=b,C=c,D=d,B_minus_A_pp=b-a,C_minus_A_pp=c-a,
                    D_minus_A_pp=d-a,interaction_pp=(d-c)-(b-a))
        for k,v in expect.items():assert abs(float(r[k])-v)<1e-10,(k,r,v)
    # Full checkpoint reload and (integer errors,bits) replay for 4 selected cells per seed.
    replay=0;phase_checks=0;shift_checks=0;exact_a=0
    per_seed=[]
    for seed in SEEDS:
        m=load_models(seed,ckpt,steps,batch)
        for profile,snr in [('train3',10),('long6',10),('long12',10),('sparse12',10)]:
            for r in eval_condition(m,seed,profile,snr,frames):
                old=by[seed,profile,snr,r['method']]
                assert (r['errors'],r['bits'])==(int(old['errors']),int(old['bits'])),(seed,profile,snr,r['method'])
                replay+=1
        torch.manual_seed(8900000+seed)
        b=make_batch(24,N,10,10,'cpu')
        old=EquivDelayConv();old.load_state_dict(m['A_pos_0'].state_dict())
        old.eval()
        with torch.inference_mode():
            assert torch.equal(old.channel(b.yp,b.yd,b.noise_var),m['A_pos_0'].channel(b.yp,b.yd,b.noise_var))
            exact_a+=1
            for name,model in m.items():
                h=model.channel(b.yp,b.yd,b.noise_var)
                logits=model(b.yp,b.yd,b.noise_var)
                for phi in (.41,1.21,2.43):
                    u=torch.polar(torch.ones_like(b.yp.real),torch.full_like(b.yp.real,phi))
                    hr=model.channel(b.yp*u,b.yd*u,b.noise_var)
                    lr=model(b.yp*u,b.yd*u,b.noise_var)
                    assert float((hr-h*u).abs().max())<1e-5
                    assert float((lr-logits).abs().max())<1e-5
                    assert torch.equal(lr>0,logits>0)
                    phase_checks+=1
                if not model.use_position:
                    ht=torch.fft.ifft(h,dim=-1)
                    yt=torch.fft.ifft(b.yp,dim=-1)
                    for offset in (3,11):
                        shifted=torch.fft.fft(torch.roll(yt,offset,dims=-1),dim=-1)
                        hr=model.channel(shifted,b.yd,b.noise_var)
                        ref=torch.fft.fft(torch.roll(ht,offset,dims=-1),dim=-1)
                        assert float((hr-ref).abs().max())<1e-5
                        shift_checks+=1
        per_seed.append(seed)
    gates=rows(root/'gate_by_delay.csv')
    assert len(gates)==len(SEEDS)*len(PROFILES)*len(ARMS)*N
    for g in gates:
        arm=g['arm'];floor=ARMS[arm][1]
        assert floor-1e-6<=float(g['mean_gain'])<=1+1e-6
        assert float(g['true_tap_power'])>=0
    result=dict(status='PASS',seeds=list(SEEDS),trained_models=len(hashes),
        checkpoint_sha256=hashes,protocol_sha256=EXPECTED_PROTOCOL_SHA256,
        record_rows=len(raw),summary_rows=len(summary),factorial_rows=len(factorial),
        integer_error_bit_replay_cells=replay,
        trained_common_phase_cases=phase_checks,
        trained_delay_shift_cases=shift_checks,
        trained_original_a_exact_matches=exact_a,
        gate_diagnostic_rows=len(gates),tested_profiles=list(PROFILES),snrs=list(SNRS),
        note='Record reconciliation and deterministic same-code replays; not independent codebase or SDR replication.')
    (root/'independent_audit.json').write_text(json.dumps(result,indent=2,sort_keys=True))
    print('AUDIT PASS:',{k:v for k,v in result.items() if k!='checkpoint_sha256'})
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('p3a2_results'))
    p.add_argument('--ckpt',type=Path,default=Path('p3a2_checkpoints'))
    args=p.parse_args();verify(args.out,args.ckpt)
