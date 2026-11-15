#!/usr/bin/env python3
"""Post-training *diagnostic-only* refs for P3-B (not a trained receiver or frozen primary comparison).
Recreates the exact saved test RNG streams; no neural retraining, no test-side tuning.
Only simulated civilian passive QPSK/OFDM. Additional information is explicitly privileged.
"""
import csv
from pathlib import Path
import torch
from p3b_time_link import PROFILES, ALL_CONDITIONS, draw_base, link, PAYLOAD_MASK, N
from p3b_train_eval import SEEDS, SNRS, FRAMES, data_seed, impairment_seed
from p3b_receivers import logits_from_channel, bit_count

def write_csv(path, rows):
    if not rows: return
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def matched_prior_linear(yp,noise,profile):
    # Independent spectral representation of Gaussian-prior LMMSE.
    p=torch.zeros(N,dtype=yp.real.dtype,device=yp.device)
    x=torch.tensor(profile,dtype=yp.real.dtype,device=yp.device)
    p[:len(x)]=x/x.sum()
    d=torch.fft.ifft(yp,dim=-1)
    return torch.fft.fft(d*(p[None,:]/(p[None,:]+noise/N+1e-12)),dim=-1)

def evaluate(frames=FRAMES):
    torch.set_num_threads(4)
    results=[]
    for seed in SEEDS:
        for pname, profile in PROFILES.items():
            for snr in SNRS:
                for condition in ALL_CONDITIONS:
                    e={'matched_test_prior_privileged':0,'true_channel_privileged':0}
                    bits=0
                    for part,off in enumerate(range(0,frames,128)):
                        n=min(128,frames-off)
                        gen=torch.Generator().manual_seed(data_seed(seed,pname,snr,part))
                        base=draw_base(n,profile,snr,gen)
                        observed=link(base,condition,extra_seed=impairment_seed(seed,pname,snr,condition,part),include_stats=False)
                        with torch.inference_mode():
                            hh=matched_prior_linear(observed.yp,observed.noise_var,profile)
                            preds={
                                'matched_test_prior_privileged':logits_from_channel(observed.yd,hh,observed.noise_var,align=True),
                                'true_channel_privileged':logits_from_channel(observed.yd,base.h,observed.noise_var,align=True)
                            }
                            for key,val in preds.items(): e[key]+=bit_count(val,base.bits)
                        bits+=n*60
                    for method,err in e.items():results.append(dict(seed=seed,profile=pname,snr_db=snr,condition=condition,method=method,errors=err,bits=bits,ber=err/bits,note='privileged, not a matched-information comparator; no PA/CPE/ADC oracle'))
        print('PRIVILEGED',seed,'complete',flush=True)
    return results

if __name__=='__main__':
    path=Path('p3b_results/privileged_reference_by_seed.csv');path.parent.mkdir(exist_ok=True)
    rows=evaluate();write_csv(path,rows)
    print('SAVED',path,len(rows),'records')
