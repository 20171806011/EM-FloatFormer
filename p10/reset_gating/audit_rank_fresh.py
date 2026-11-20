"""Independent second-stage reconstruction from frozen weights and public evidence.

The standalone check does not choose thresholds. It recomputes the predetermined
rank at each fold using the maximum eligible fold score of each reset link.
"""
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.special import expit

MODELS=('physics','linear','mlp','attention')


def record_scores(f, fit):
    t=np.asarray(f['tokens'],float)
    p=np.asarray(f['pilot'],float)
    x=np.column_stack((t[:,0]-t[:,1],t[:,0]-t[:,2],t[:,0]))
    u=np.array([p[0]-p[1],p[1]])
    for pts,key in [(x,'token'),(u,'pilot')]:
        median,scale=np.asarray(fit['normalization'][key])
        pts[:]=np.arcsinh((pts-median)/scale)
    design=np.r_[1.,x.ravel(),u]
    keys=np.column_stack((x,np.eye(7)[:,:6]))
    query=np.r_[1.,u,x[:5,:2].mean(axis=0),x[5:,:2].mean(axis=0)]
    scores={'physics':float(f['score'])}
    for model in MODELS[1:]:
        params=np.array(fit['models'][model]['parameters'])
        z=design@params[:24]
        for head in params[24:].reshape(-1,25):
            if model=='mlp':
                z+=head[24]*np.tanh(design@head[:24])
            else:
                logits=np.tanh(query@head[:7])*(keys@head[7:16])
                w=np.exp(logits-logits.max());w/=w.sum()
                z+=w@(keys@head[16:25])
        scores[model]=float(expit(z))
    return scores


def calculate():
    read=lambda f:json.loads(Path(f).read_text())
    fit=read('learn_weights.json')
    cal=read('cal_active.json')['records']
    fresh=read('strong_fresh.json')['records']
    cal_scores={a:[] for a in MODELS}
    for record in cal:
        eligible=[record_scores(f,fit) for f in record['folds'] if f['used']]
        for a in MODELS:
            cal_scores[a].append(max((s[a] for s in eligible),default=float('-inf')))
    wrong=defaultdict(Counter)
    accept_links=defaultdict(Counter)
    accepted={}
    for r in fresh:
        arms={a:[] for a in MODELS}
        for f in r['folds']:
            scores=record_scores(f,fit)
            for a in MODELS:
                numer=1+sum(c>=scores[a] for c in cal_scores[a])
                use=f['used'] and 100*numer<=5*(1+len(cal))
                arms[a].append(bool(use))
                wrong[r['regime']][a]+=sum(f['v1_errors' if use else 'p9_errors'])
        for a,flags in arms.items():
            accept_links[r['regime']][a]+=any(flags)
        accepted[r['regime'],r['seed']]=arms
    return wrong,accept_links,accepted

if __name__=='__main__':
    wrong,links,_=calculate()
    reference=json.loads(Path('active_rank_results.json').read_text())
    for row in reference['fold_decisions']:
        k=(row['regime'],row['seed'])
        for m in MODELS:
            arm='rank_'+('physical' if m=='physics' else m)
            assert list(map(int,_[k][m]))==row['active'][arm]
    for regime,counts in wrong.items():
        for m,count in counts.items():
            arm='rank_'+('physical' if m=='physics' else m)
            reported=reference['by_regime'][regime]['arms'][arm]
            assert reported['wrong']==count and reported['active_links']==links[regime][m]
    print(json.dumps(dict(errors=wrong,accepted_links=links,fold_decisions_checked=4*2*len(_))))
