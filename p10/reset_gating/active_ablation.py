"""Post hoc matched-calibration diagnostic: include vs exclude inactive calibration folds.

Uses identical fitted models, evaluation observations and 128 calibration links.
Only the construction of the calibration maximum changes.
"""
import json
from pathlib import Path

import numpy as np

from active_rank import REGIMES, SCORES, calibrated_scores, contrast, extract, load_records, validate_inputs


def analyze(cal, ev, fitted):
    validate_inputs(cal,ev,fitted)
    cu,cs=extract(cal,fitted)
    ev=sorted(ev,key=lambda r:(r['seed'],r['regime']))
    used,z=extract(ev,fitted)
    n=len(set(r['seed'] for r in ev))
    b=np.random.default_rng(1000501).integers(n,size=(10000,n))
    dtr=[np.array([np.array(f['p9_errors'])-f['v1_errors'] for r in ev for f in [r['folds'][j]]]) for j in (0,1)]
    ebase=np.array([sum((f['p9_errors'] for f in r['folds']),[]) for r in ev],float)
    regimes=np.array([r['regime'] for r in ev])
    seeds=np.array([r['seed'] for r in ev])
    out={}
    for name in SCORES:
        old,po,_=calibrated_scores(np.ones_like(cu),cs[name],used,z[name])
        new,pn,_=calibrated_scores(cu,cs[name],used,z[name])
        assert np.all(new>=old) and np.all(pn<=po)
        diff=np.concatenate([dtr[j]*(new[:,j].astype(int)-old[:,j].astype(int))[:,None] for j in (0,1)],axis=1)
        oldloss=ebase-np.concatenate([dtr[j]*old[:,j,None] for j in (0,1)],axis=1)
        res={}
        for reg in (*REGIMES,'equal_weight'):
            ix=np.arange(len(ev)) if reg=='equal_weight' else np.flatnonzero(regimes==reg)
            seed_groups=np.unique(seeds[ix])
            mx=np.array([diff[ix][seeds[ix]==s] for s in seed_groups],int)
            res[reg]=dict(old_active_links=int(old[ix].any(1).sum()),new_active_links=int(new[ix].any(1).sum()),
                          old_errors=int(oldloss[ix].sum()),new_errors=int((oldloss-diff)[ix].sum()),
                          old_minus_new=contrast(mx,b))
        out[name]=res
    return out


if __name__=='__main__':
    data=analyze(load_records('cal_active.json'),load_records('strong_fresh.json'),json.loads(Path('learn_weights.json').read_text()))
    Path('active_ablation.json').write_text(json.dumps(data,indent=2)+'\n')
    for name,groups in data.items():
        print(name,[(g,r['old_errors'],r['new_errors'],round(r['old_minus_new']['gain_pp'],3),[round(x,3) for x in r['old_minus_new']['ci_pp']]) for g,r in groups.items()])
