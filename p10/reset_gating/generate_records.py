import argparse
import json
from pathlib import Path

import numpy as np
from p8.coding import decode
from p9.channel import simulate_link, track_frame_a
from p10_covariance import AUX,TARGET,TARGET_INFO
from receiver_next import candidates,pilot_evidence

OFFSETS = dict(train=971120010000,cal=971120020000,eval=971120030000)
REGIMES = ('continuous','partial_shift','full_reset')


def evaluate(split,count):
    dest = Path(split+'.jsonl')
    tmp = dest.with_suffix('.tmp')
    with tmp.open('w') as stream:
        for i in range(count):
            for regime in (('full_reset',) if split=='cal' else REGIMES):
                seed = OFFSETS[split]+i
                obs,truth = simulate_link(seed,9,regime,'sparse')
                if split=='cal':
                    pilot = pilot_evidence(obs,track_frame_a(obs.y_a,obs.n0))
                    rec = dict(score=float(pilot[0]-pilot[1]))
                else:
                    result = candidates(obs,AUX)
                    rec = {k:v for k,v in result.items() if k not in ('p9','v1')}
                    rec['energy'] = float(np.sum(abs(truth.y_b_probe_clean)**2))
                    for key in ('p9','v1'):
                        _,_,llrs,probe = result[key]
                        bits = decode(llrs[TARGET].reshape(len(TARGET),128))
                        rec[key+'_errors'] = (bits!=truth.b_info[TARGET_INFO]).any(1).astype(int).tolist()
                        rec[key+'_sse'] = float(np.sum(abs(probe-truth.y_b_probe_clean)**2))
                stream.write(json.dumps(dict(rec,seed=seed,regime=regime),allow_nan=False)+'\n')
    tmp.replace(dest)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('split',choices=list(OFFSETS))
    parser.add_argument('--count',type=int,default=128)
    args=parser.parse_args()
    evaluate(args.split,args.count)
