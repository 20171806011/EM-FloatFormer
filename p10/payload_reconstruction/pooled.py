"""Paired descriptive pooling across the original and unchanged replication batches."""
import json
from pathlib import Path
import numpy as np

FILES=('rebuild_raw.json','rebuild_replication.json')
REGIMES=('full_reset','partial_shift','continuous')
ARMS=('p9','v1','clear','pilot_only','rebuild','default','gate','code_gate')


def collect(raw):
    out={}
    for r in raw['records']:
        errs={a:np.array(sum([f['errors'][a] for f in r['folds']],[]),bool) for a in ARMS}
        common=errs['p9'] & errs['v1']
        triple=common & errs['clear']
        row={'blocks':len(common),'common':common.sum(),'triple':triple.sum()}
        for a,err in errs.items():
            row[a+'_wrong']=err.sum()
            row[a+'_recover']=(common & ~err).sum()
        row['rebuild_lost_if_either_correct']=(~common & errs['rebuild']).sum()
        row['rebuild_lost_if_both_correct']=(~errs['p9'] & ~errs['v1'] & errs['rebuild']).sum()
        row['both_correct']=(~errs['p9'] & ~errs['v1']).sum()
        row['rebuild_triple_recover']=(triple & ~errs['rebuild']).sum()
        row['pilot_to_payload_gain_subset']=(common & errs['pilot_only'] & ~errs['rebuild']).sum()
        row['pilot_to_payload_loss_subset']=(common & ~errs['pilot_only'] & errs['rebuild']).sum()
        row['code_recover_over_incumbent']=(errs['default'] & ~errs['code_gate']).sum()
        row['code_lose_over_incumbent']=(~errs['default'] & errs['code_gate']).sum()
        row['rebuild_recover_over_p9']=(errs['p9'] & ~errs['rebuild']).sum()
        row['rebuild_loss_over_p9']=(~errs['p9'] & errs['rebuild']).sum()
        out[r['regime'],r['seed']]=row
    return out


def run():
    batches=[collect(json.loads(Path(f).read_text())) for f in FILES]
    assert not {s for _,s in batches[0]} & {s for _,s in batches[1]}
    keys=list(next(iter(batches[0].values())))
    random=np.random.default_rng(98112006)
    idx=[random.integers(128,size=(10000,128)) for _ in FILES]
    summary={}
    for regime in (*REGIMES,'equal_weight'):
        group=REGIMES if regime=='equal_weight' else (regime,)
        point={};boot={}
        for key in keys:
            arrays=[]
            for batch in batches:
                seeds=sorted({s for _,s in batch})
                arrays.append(np.array([sum(batch[reg,s][key] for reg in group) for s in seeds]))
            point[key]=int(sum(a.sum() for a in arrays))
            boot[key]=sum(a[i].sum(1) for a,i in zip(arrays,idx))
        stats={}
        for a in ARMS:
            for num,den in [(a+'_wrong','blocks'),(a+'_recover','common')]:
                stats[num]=dict(rate=100*point[num]/point[den],ci95_percent=np.quantile(100*boot[num]/boot[den],[.025,.975]).tolist())
        for a,b in [('default','code_gate'),('pilot_only','rebuild'),('default','gate')]:
            numerator=boot[a+'_wrong']-boot[b+'_wrong']
            stats[a+'_to_'+b]=dict(pp=100*(point[a+'_wrong']-point[b+'_wrong'])/point['blocks'],
                                    ci95_pp=np.quantile(100*numerator/boot['blocks'],[.025,.975]).tolist())
        summary[regime]=dict(counts=point,metrics=stats)
    return summary

if __name__=='__main__':
    summary=run()
    Path('pooled.json').write_text(json.dumps(summary,indent=2)+'\n')
    for reg,r in summary.items():
        print(reg,'D',r['counts']['common'],'rescued',r['counts']['rebuild_recover'],r['metrics']['rebuild_recover'],
              'gate gain',r['metrics']['default_to_code_gate'])
