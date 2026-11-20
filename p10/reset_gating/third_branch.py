"""Post-hoc attainable strong BLER for a proposed third, history-free action.

The scorer sees the labels when selecting its hindsight best action. These results
are ceilings on the value of a selector, not a deployed receiver.
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from p9.channel import simulate_link
from p9.gating import build_candidates
from strong_eval import strong_errors

REGIMES=('partial_shift','continuous','full_reset')


def run_regime(args):
    regime,rows=args
    records=[]
    for r in rows:
        obs,truth=simulate_link(r['seed'],9.,regime,'sparse')
        clear=build_candidates(obs)[1]
        assert clear.name=='clear'
        targets=[k for f in r['folds'] for k in f['targets']]
        assert len(targets)==16 and len(set(targets))==16
        err=strong_errors(obs,truth,clear.mean_ray,clear.var_ray,clear.llrs,targets)
        offset=0
        folds=[]
        for f in r['folds']:
            n=len(f['targets'])
            clear_err=err[offset:offset+n]
            offset+=n
            ca=np.array([f['p9_errors'],f['v1_errors'],clear_err]); assert np.isin(ca,[0,1]).all()
            choices=ca.sum(axis=1)
            o2=int(choices[:2].min());o3=int(choices.min())
            word2=int(ca[:2].min(axis=0).sum());word3=int(ca.min(axis=0).sum())
            folds.append(dict(fold=f['fold'],targets=f['targets'],clear_errors=clear_err,
                p9_errors=f['p9_errors'],v1_errors=f['v1_errors'],clear=int(sum(clear_err)),
                oracle2=o2,oracle3=o3,clear_fold_strict_gain=o2-o3,
                block_oracle2=word2,block_oracle3=word3))
        records.append(dict(seed=r['seed'],regime=regime,folds=folds))
    return records


def summary(records):
    folds=[f for r in records for f in r['folds']]
    totals={k:sum(f[k] for f in folds) for k in ('clear','oracle2','oracle3','clear_fold_strict_gain','block_oracle2','block_oracle3')}
    totals.update(links=len(records),blocks=sum(len(f['targets']) for f in folds),
        folds_with_strict_gain=sum(f['oracle2']>f['oracle3'] for f in folds),
        links_with_strict_gain=sum(any(f['oracle2']>f['oracle3'] for f in r['folds']) for r in records),
        extra_per_block_gain=totals['block_oracle2']-totals['block_oracle3'])
    return totals


if __name__=='__main__':
    links=json.loads(Path('strong_fresh.json').read_text())['records']
    old=json.loads(Path('active_rank_results.json').read_text())['by_regime']
    jobs=[(r,[row for row in links if row['regime']==r]) for r in REGIMES]
    with ProcessPoolExecutor(max_workers=len(REGIMES)) as pool:
        all_records=sum(list(pool.map(run_regime,jobs)),[])
    assert len({(r['regime'],r['seed']) for r in all_records})==384
    report={}
    for reg in (*REGIMES,'equal_weight'):
        subset=all_records if reg=='equal_weight' else [r for r in all_records if r['regime']==reg]
        report[reg]=summary(subset)
        assert report[reg]['oracle2']==old[reg]['arms']['utility_oracle']['wrong']
    dest=Path('third_branch.json')
    tmp=dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(summary=report,records=all_records),allow_nan=False,separators=(',',':'))+'\n')
    tmp.replace(dest)
    print(json.dumps(report,indent=2))
