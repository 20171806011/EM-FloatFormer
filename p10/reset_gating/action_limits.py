"""Exact lower bounds on this finite test set for choosing between two hard outputs."""
import json
from pathlib import Path
import numpy as np

def bounds(file):
    records = json.loads(Path(file).read_text())['records']
    out = {}
    for regime in sorted({r['regime'] for r in records}):
        values = np.zeros(5,dtype=int)
        for r in records:
            if r['regime']!=regime:
                continue
            for fold in r['folds']:
                p = np.asarray(fold['p9_errors'])
                v = np.asarray(fold['v1_errors'])
                assert len(v)==len(p)
                values += [len(p),p.sum(),v.sum(),min(p.sum(),v.sum()),np.count_nonzero(p*v)]
        out[regime] = dict(zip(['blocks','p9','v1','fold_oracle','block_oracle'],map(int,values)))
    out['equal_weight'] = {k:sum(r[k] for r in out.values()) for k in next(iter(out.values()))}
    return out

if __name__=='__main__':
    result={f:bounds(f) for f in ('strong16.json','strong_fresh.json')}
    print(json.dumps(result,indent=2))
    Path('action_limits.json').write_text(json.dumps(result,indent=2)+'\n')
