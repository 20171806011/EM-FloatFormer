"""Pinned, read-only verification of one Dev frame. Optional B240 historical regression.
Requires Python 3.10+, CPU PyTorch, NumPy and SciPy. No data downloads or training.
Usage: python real_smoke.py --source ../realdata_ofdm_stress_and_theory --audit-dir ../B --pth /approved/local/dataset.pth
Pass --b240 only when separately released after the Dev check. Prints aggregate JSON only.
"""
import argparse
import hashlib
import importlib
import json
import time
from pathlib import Path
import sys

import numpy as np

PIN = '5f88a5be984493bcb6aa164d84253b6f29876eef'
PTH_SIZE = 27684430
PTH_SHA = 'e760079a888f37d8c33b21f327646930b4f2cb4511e8a309bcde22c6b1332058'
TEMP = 6.0
STAY = 0.8
TCAL = 2.3154744812056767
TARGETS = ((2,39), (7,45), (13,100))
CONDITIONS = ('C0_raw_ota','C1_awgn_stress','C2_cfo_doppler','C3_phase_jump_jitter','C4_nbi_sparse7')
SRC_BLOBS = {
    'hypothesis_frontend.py': '735233d5ad34f77788f7ea2536048eb489be34ae',
    'group_b_posterior_fusion.py': 'dd32790b893bb432376622f2bf7a85561955df0a',
    'ofdm_data.py': 'e2ed1a2e7a71415ef205498bb3f104827a1468ae',
    'stress_channel.py': '898bccc0c4d4bcd1ced2100272c5ea4a4977ea06',
}

def validate_files(source, pth):
    for name, expected in SRC_BLOBS.items():
        raw=(source/name).read_bytes()
        actual=hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
        if actual!=expected:
            raise ValueError('wrong receiver source: '+name)
    if pth.stat().st_size!=PTH_SIZE:
        raise ValueError('wrong input size')
    h=hashlib.sha256()
    with pth.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):
            h.update(chunk)
    if h.hexdigest()!=PTH_SHA:
        raise ValueError('wrong known input digest; stop before deserialization')

def single_dev(ds, refs, d, f, stress, audit):
    t=360
    out={}
    for cond in (CONDITIONS[0],CONDITIONS[2],CONDITIONS[3]):
        p=stress.apply_stress_condition(ds.iq[t],cond,seed=stress.condition_seed(t,cond))
        records=[]
        before=time.perf_counter()
        for s,k in TARGETS:
            idx=np.flatnonzero((d.DATA_COORDS_101==[s,k]).all(axis=1)).item()
            rec,inputs=audit.physical(p.iq,p.pilot_scs,refs,(s,k),TEMP,STAY,d,f,check_reference=(cond==CONDITIONS[0]))
            audit.check_invariance(p.iq,p.pilot_scs,refs,(s,k),TEMP,STAY,d,f,inputs)
            truth=ds.labels[t,idx].astype(int)
            old=np.asarray(rec['original']['hard'])
            modes={}
            for m in ('original','explicit','rebuild'):
                pred=np.asarray(rec[m]['hard'])
                modes[m]={'errors':int((pred!=truth).sum()),'rescued_vs_original':int(((old!=truth)&(pred==truth)).sum()),
                          'added_vs_original':int(((old==truth)&(pred!=truth)).sum())}
            records.append({'coordinate':[int(s),int(k)],'modes':modes})
        out[cond]={'bits':4*len(TARGETS),'records':records,'all_modes_checks_wall_s':time.perf_counter()-before}
    return out

def historical(ds,refs,d,b):
    # Read archived CSV for keyed regression, never for parameter selection.
    import csv
    with (Path(b.__file__).parent/'results/group_b_per_tti.csv').open(newline='') as handle:
        archived={(r['condition'],int(r['tti'])):float(r['ber']) for r in csv.DictReader(handle) if r['method']=='B3_mixture_likelihood'}
    oracle=d.build_oracle_channel(ds)  # upstream reporting diagnostics only
    out={}
    for cond in CONDITIONS:
        start=time.perf_counter()
        errs,raw,cal,max_abs=[],[],[],0.
        for t in range(502,742):
            r=b.evaluate_group_b_on_tti(ds.iq[t],ds.labels[t],oracle[t],t,cond,refs,
                                        ll_temp=TEMP,p_stay=STAY,use_hmm=True)['B3_mixture_likelihood']
            ll=r['llrs']; truth=ds.labels[t]
            errs.append(int(((ll>0)!=truth).sum()))
            raw.append(d.compute_bit_metrics(ll,truth)['nll'])
            cal.append(d.compute_bit_metrics(ll/TCAL,truth)['nll'])
            max_abs=max(max_abs,abs(errs[-1]/5600-archived[cond,t]))
        out[cond]={'bit_count':1344000,'bit_errors':sum(errs),'ber':sum(errs)/1344000,
                   'raw_nll':float(np.mean(raw)),'cal_nll':float(np.mean(cal)),
                   'reference_max_abs_tti_ber_difference':max_abs,'all_six_plus_checks_wall_s':time.perf_counter()-start}
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--pth',type=Path,required=True)
    p.add_argument('--audit-dir',type=Path,required=True,help='directory containing the B audit.py in this research package')
    p.add_argument('--b240',action='store_true',help='optional archived regression; requires results/group_b_per_tti.csv in source')
    a=p.parse_args()
    validate_files(a.source,a.pth)
    sys.path[:0]=[str(a.source.resolve()),str(a.audit_dir.resolve())]
    d,f,b,stress,audit=[importlib.import_module(m) for m in ('ofdm_data','hypothesis_frontend','group_b_posterior_fusion','stress_channel','audit')]
    ds=d.load_real_ofdm_dataset(a.pth)
    assert ds.iq.shape==(742,14,128) and ds.labels.shape==(742,1400,4) and ds.sha256==PTH_SHA
    refs=f.build_train_reference_bank(ds.iq[:340],ds.sinr_db[:340],m_refs=16)
    out={'pin':PIN,'data_sha256':PTH_SHA,'dev_tti':360,'tau':TEMP,'stay':STAY,'dev_smoke':single_dev(ds,refs,d,f,stress,audit)}
    if a.b240:
        out['historical_b240']=historical(ds,refs,d,b)
    print(json.dumps(out,indent=2,allow_nan=False))
if __name__=='__main__':
    main()
