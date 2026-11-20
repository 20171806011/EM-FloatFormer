"""Compare the pilot-only and payload-assisted estimate on matched trajectories."""
import json
from pathlib import Path
import numpy as np


def summarize(records):
    out = {}
    for reg in sorted({r['regime'] for r in records}):
        rows = sorted([r for r in records if r['regime']==reg],key=lambda r:r['seed'])
        if len(set(r['seed'] for r in rows))!=len(rows):
            raise ValueError('Duplicate link')
        boot = np.random.default_rng(54213).integers(len(rows),size=(4000,len(rows)))
        out[reg] = {}
        for metric in ('rel_phase','rel_freq','abs_phase','abs_freq'):
            sse = np.array([[sum(f['phase'][name][metric+'_sse'] for f in r['folds'])
                            for name in ('pilot_only','rebuild')] for r in rows])
            weight = np.array([sum(f['phase']['rebuild'][metric+'_weight'] for f in r['folds']) for r in rows])
            est = np.sqrt(sse.sum(0)/weight.sum())
            draws = np.sqrt(sse[boot].sum(1)/weight[boot].sum(1)[:,None])
            out[reg][metric] = dict(pilot_only_rmse=float(est[0]),rebuild_rmse=float(est[1]),
                                    links=len(rows),improved_links=int(np.sum(sse[:,1]<sse[:,0])),
                                    rebuild_minus_pilot_ci95=np.quantile(draws[:,1]-draws[:,0],[.025,.975]).tolist())
    return out

if __name__=='__main__':
    records=json.loads(Path('rebuild_raw.json').read_text())['records']
    print(json.dumps(summarize(records),indent=2))
