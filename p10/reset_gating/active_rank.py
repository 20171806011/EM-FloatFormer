"""Risk-calibrate active, two-fold repair proposals with a held-out reset sample.

Four pre-specified scores use the same published candidate branch. We calibrate
the maximum score over *eligible* folds of an entire reset link. An empty set
has score -infinity. No simulated truth/target enters either folding decision.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import beta

from learn_gate import REGIMES, decisions, observations

ALPHA = .05
SCORES = ('physical','linear','mlp','attention')
TRAIN_START = 971120010000
CAL_START = 971120040000
EVAL_START = 971120050000
N_SEEDS = 128


def load_records(path):
    d=json.loads(Path(path).read_text())
    return d['records'] if isinstance(d,dict) else d


def extract(rows,fitted):
    folds=[f for r in rows for f in r['folds']]
    if not all(len(r['folds'])==2 for r in rows):
        raise ValueError('Expected two cross-fitted folds per physical link')
    used=np.array([f['used'] for f in folds],bool).reshape(-1,2)
    probs=decisions(fitted,*observations(folds))
    z={'physical':np.array([f['score'] for f in folds])}
    z.update(probs)
    z={k:v.reshape(-1,2) for k,v in z.items()}
    if not all(np.isfinite(v).all() for v in z.values()):
        raise ValueError('Only finite pre-calibration scores permitted for proposed actions')
    return used,z


def calibrated_scores(cal_used,cal_score,used,score):
    s=np.where(cal_used,cal_score,-np.inf).max(axis=1)
    count=np.sum(s[None,None,:]>=score[:,:,None],axis=-1)
    pv=(1+count)/(len(s)+1)
    accepted=used & (pv<=ALPHA)
    return accepted,pv,s


def exact_cp(k,n):
    lo=0. if k==0 else beta.ppf(.025,k,n-k+1)
    hi=1. if k==n else beta.ppf(.975,k+1,n-k)
    return [float(lo),float(hi)]


def contrast(d,sampled):
    link=(d.mean(-1) if d.ndim==2 else d.mean(axis=(1,2)))
    return dict(gain_pp=float(100*link.mean()),ci_pp=(100*np.quantile(link[sampled].mean(axis=1),[.025,.975])).tolist(),
                rescued=int((d>0).sum()),newly_wrong=int((d<0).sum()))


def validate_inputs(cal,ev,fitted):
    if fitted['train_seeds']!=list(range(TRAIN_START,TRAIN_START+N_SEEDS)):
        raise ValueError('Unexpected training namespace or training sample size')
    if len(cal)!=N_SEEDS or sorted(r['seed'] for r in cal)!=list(range(CAL_START,CAL_START+N_SEEDS)):
        raise ValueError('Unexpected calibration namespace, sample size or duplicate')
    if any(r['regime']!='full_reset' for r in cal):
        raise ValueError('Full-reset calibration required')
    if len(ev)!=N_SEEDS*len(REGIMES):
        raise ValueError('Unexpected independent evaluation size')
    for reg in REGIMES:
        if sorted(r['seed'] for r in ev if r['regime']==reg)!=list(range(EVAL_START,EVAL_START+N_SEEDS)):
            raise ValueError('Unexpected evaluation namespace or duplicate')
    for r in ev:
        fs=r['folds']
        if len(fs)!=2 or len(fs[0]['targets'])!=11 or len(fs[1]['targets'])!=5 or set(fs[0]['targets']) & set(fs[1]['targets']):
            raise ValueError('Invalid held-out target row sets')
        for f in fs:
            p,v = np.array(f['p9_errors']),np.array(f['v1_errors'])
            if len(p)!=len(f['targets']) or p.shape!=v.shape or not (np.isin(v,[0,1]).all() and np.isin(p,[0,1]).all()):
                raise ValueError('Need paired per-target 0/1 losses')
            if not f['used'] and not np.array_equal(p,v):
                raise ValueError('Ineligible P9/v1 loss mismatch')


def test(cal,ev,fitted):
    validate_inputs(cal,ev,fitted)
    ev=sorted(ev,key=lambda r:(r['seed'],r['regime']))
    cal=sorted(cal,key=lambda r:r['seed'])
    cv,cs=extract(cal,fitted)
    used,scores=extract(ev,fitted)
    # Always decode with P9 or the original v1 repair. Vetoes restrict the action space.
    choices={'p9':np.zeros_like(used), 'v1':used}
    pvals={}; cal_info={}
    for name in SCORES:
        choices['rank_'+name],pvals[name],m=calibrated_scores(cv,cs[name],used,scores[name])
        cutoff=np.sort(m)[-int(np.floor(ALPHA*(len(cal)+1)))]
        cal_info[name]=dict(active_links=int(cv.any(1).sum()),active_folds=int(cv.sum()),
                            strict_score_cutoff=float(cutoff) if np.isfinite(cutoff) else '-infinity')
    for name in ('linear','mlp','attention'):
        choices['fixed_'+name]=used & (scores[name]>=.5)
    reg=np.array([r['regime'] for r in ev])
    seeds=np.array([r['seed'] for r in ev])
    choices['mechanism_oracle']=used & (reg[:,None]=='partial_shift')
    delta=np.array([[sum(f['p9_errors']) -sum(f['v1_errors']) for f in r['folds']] for r in ev])
    choices['utility_oracle']=used & (delta>0)
    rows=[dict(seed=r['seed'],regime=r['regime'],
               fold_rank_p={k:v[i].tolist() for k,v in pvals.items()},
               active={k:v[i].astype(int).tolist() for k,v in choices.items()}) for i,r in enumerate(ev)]
    err={a:np.array([sum((f['v1_errors'] if sw else f['p9_errors'] for f,sw in zip(r['folds'],act[i])),[])
                    for i,r in enumerate(ev)],int) for a,act in choices.items()}
    tests=[('p9','rank_attention'),('v1','rank_attention'),('rank_mlp','rank_attention'),
           ('rank_linear','rank_attention'),('rank_physical','rank_attention'),('rank_mlp','rank_linear'),
           ('rank_physical','rank_mlp'),('fixed_mlp','fixed_attention'),('fixed_attention','rank_attention'),
           ('fixed_mlp','rank_mlp'),('v1','rank_physical')]
    gids=np.unique(seeds)
    sampled=np.random.default_rng(1000501).integers(0,len(gids),size=(10000,len(gids)))
    results={}
    for group in (*REGIMES,'equal_weight'):
        ix=np.arange(len(reg)) if group=='equal_weight' else np.flatnonzero(reg==group)
        arms={}; cis={}
        for a,e in err.items():
            data=e[ix]; active=choices[a][ix]; gain=err['p9'][ix]-data
            t=dict(wrong=int(data.sum()),blocks=int(data.size),bler=float(data.mean()),
                   active_folds=int(active.sum()),active_links=int(active.any(1).sum()),
                   paired_vs_p9_rescued=int((gain>0).sum()),paired_vs_p9_newly_wrong=int((gain<0).sum()))
            if group!='equal_weight':
                t['activation_ci_cp_95']=exact_cp(t['active_links'],len(ix))
            arms[a]=t
        for a,c in tests:
            diff=err[a][ix]-err[c][ix]
            by_seed=np.array([diff[seeds[ix]==gid] for gid in gids])
            cis[a+'_minus_'+c]=contrast(by_seed,sampled)
        ceiling={a:float(100*(err[a][ix]-err['utility_oracle'][ix]).mean()) for a in err}
        results[group]=dict(links=len(ix),arms=arms,comparisons=cis,excess_over_empirical_oracle_pp=ceiling)
    return dict(alpha=ALPHA,reset_marginal_bound=float(np.floor(ALPHA*(len(cal)+1))/(len(cal)+1)),calibration=cal_info,
                by_regime=results,fold_decisions=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cal',type=Path,default=Path('cal_active.json'))
    p.add_argument('--eval',type=Path,default=Path('strong_fresh.json'))
    p.add_argument('--fitted',type=Path,default=Path('learn_weights.json'))
    p.add_argument('--out',type=Path,default=Path('active_rank_results.json'))
    args=p.parse_args()
    metrics=test(load_records(args.cal),load_records(args.eval),json.loads(args.fitted.read_text()))
    args.out.write_text(json.dumps(metrics,separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps(metrics['by_regime']['equal_weight'],indent=2))
