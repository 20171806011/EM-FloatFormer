#!/usr/bin/env python3
"""Conditional upper reference: classical receiver given the TEST delay prior.
This reference has extra information at OOD evaluation. It is NOT a fair
competitor for a model that receives only the TRAIN distribution prior.
No transmitter or active interference is implemented.
"""
import sys, os, math, csv, torch
sys.path.insert(0,os.path.dirname(os.path.dirname(__file__)))
from ofdm_baseline import make_batch,estimate_channel,covariance_matrix,decisions_from_channel
from p1b_representation import TEST_PROFILES

torch.set_num_threads(4)
rows=[]
for seed in [101,202,303]:
 for profile_name,profile in TEST_PROFILES.items():
  cov=covariance_matrix(32,profile,'cpu')
  for snr in [0,5,10,15,20]:
   torch.manual_seed(seed*100000+snr*100+(0 if profile_name=='in_distribution' else 1))
   errs={'dft_matched_test_support':0,'lmmse_matched_test_prior':0}
   bits=0
   for start in range(0,2000,128):
    b=make_batch(min(128,2000-start),32,snr,snr,'cpu',profile)
    pred={
      'dft_matched_test_support':estimate_channel(b.yp,b.noise_var,'dft',taps=len(profile)),
      'lmmse_matched_test_prior':estimate_channel(b.yp,b.noise_var,'lmmse',covariance=cov),
    }
    for method,hhat in pred.items():
     errs[method]+=int((decisions_from_channel(b.yd,hhat,b.noise_var)!=b.bits).sum())
    bits+=b.bits.numel()
   for method,e in errs.items():
    rows.append(dict(seed=seed,split=profile_name,snr_db=snr,method=method,
                     bit_errors=e,total_bits=bits,ber=e/bits))
with open(os.path.join(os.path.dirname(__file__),'matched_prior_oracle.csv'),'w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for profile in TEST_PROFILES:
 for snr in [10,20]:
  print(profile,snr,[(method,round(sum(r['bit_errors'] for r in rows if r['split']==profile and r['snr_db']==snr and r['method']==method)/384000,6)) for method in ['dft_matched_test_support','lmmse_matched_test_prior']])
