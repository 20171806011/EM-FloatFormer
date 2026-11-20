"""Use frozen row-score networks on both strong-decoder folds. No fitting."""
import argparse
import json
from pathlib import Path

import numpy as np

from learn_gate import REGIMES, decisions, observations


def run(study, fitted):
    records=study['records']
    ids=[(r['seed'],r['regime']) for r in records]
    if len(ids)!=len(set(ids)) or set(r['seed'] for r in records)&set(fitted['train_seeds']):
        raise ValueError('Duplicate trajectory or training overlap')
    ordered=sorted(records,key=lambda t:(t['seed'],t['regime']))
    flat=[f for r in ordered for f in r['folds']]
    prob=decisions(fitted,*observations(flat))
    stats_by_link=[]
    for i,r in enumerate(ordered):
        first, second = r['folds']
        if len(first['targets'])!=11 or len(second['targets'])!=5 or set(first['targets'])&set(second['targets']):
            raise ValueError('Cross-fold targets overlap or have wrong length')
        kept=[]
        for j,f in enumerate(r['folds']):
            i_flat=2*i+j
            chosen={'p9':False,'v1':bool(f['used']),'v2':bool(f['v2_used'])}
            p={a:float(v[i_flat]) for a,v in prob.items()}
            chosen.update({a:bool(f['used']) and val>=.5 for a,val in p.items()})
            for a in ('mlp','attention'):
                chosen['v2_and_'+a]=chosen['v2'] and chosen[a]
            chosen['mechanism_oracle']=bool(f['used']) and r['regime']=='partial_shift'
            chosen['utility_oracle']=bool(f['used']) and sum(f['v1_errors'])<sum(f['p9_errors'])
            losses={a:(f['v1_errors'] if active else f['p9_errors']) for a,active in chosen.items()}
            assert losses['v2']==f['v2_errors']
            kept.append(dict(fold=j,probabilities=p,accepted=chosen,losses=losses))
        stats_by_link.append(dict(seed=r['seed'],regime=r['regime'],folds=kept))

    groups=np.unique([r['seed'] for r in ordered])
    reg=np.array([r['regime'] for r in ordered]); seeds=np.array([r['seed'] for r in ordered])
    if len(ordered)!=3*len(groups):
        raise ValueError('Seed groups must have three mechanisms')
    rng=np.random.default_rng(49402)
    boots=rng.integers(0,len(groups),size=(10000,len(groups)))
    res={}
    all_names=tuple(stats_by_link[0]['folds'][0]['accepted'])
    for fold_id, selection in [('fold_11',[0]),('fold_5',[1]),('all_16',[0,1])]:
        losses={name:np.array([sum((r['folds'][f]['losses'][name] for f in selection),[])
                              for r in stats_by_link],int) for name in all_names}
        acts={name:np.array([sum(bool(r['folds'][f]['accepted'][name]) for f in selection)
                             for r in stats_by_link]) for name in all_names}
        res[fold_id]={}
        for subset in (*REGIMES,'equal_weight'):
            index=np.arange(len(reg)) if subset=='equal_weight' else np.flatnonzero(reg==subset)
            arms={a:dict(wrong=int(v[index].sum()),blocks=int(v[index].size),
                         bler=float(v[index].mean()),accepted_folds=int(acts[a][index].sum())) for a,v in losses.items()}
            contrasts={}
            for a,c in [('mlp','attention'),('linear','attention'),('v2','attention'),('v1','attention'),
                        ('v2','v2_and_attention'),('v2_and_mlp','v2_and_attention')]:
                d=losses[a][index]-losses[c][index]
                agg=np.array([d[seeds[index]==s].mean() for s in groups])
                contrasts[a+'_minus_'+c]=dict(gain_pp=100.*float(agg.mean()),rescued=int((d>0).sum()),
                    newly_wrong=int((d<0).sum()),ci_pp=(100.*np.quantile(agg[boots].mean(1),[.025,.975])).tolist())
            res[fold_id][subset]=dict(arms=arms,contrasts=contrasts)
    return dict(by_fold=res,links=stats_by_link)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,default=Path('strong16.json'))
    p.add_argument('--fitted',type=Path,default=Path('learn_weights.json'))
    p.add_argument('--out',type=Path,default=Path('learn_strong_results.json'))
    a=p.parse_args()
    metrics=run(json.loads(a.study.read_text()),json.loads(a.fitted.read_text()))
    a.out.write_text(json.dumps(metrics,separators=(',',':'))+'\n')
    print(json.dumps(metrics['by_fold']['all_16'],indent=2))
