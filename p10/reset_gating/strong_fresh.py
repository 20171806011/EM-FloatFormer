"""Independent replication on previously unused seeds plus separate reset-only calibration."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import strong16
from p9.channel import simulate_link
from receiver_next import candidates

SEED_CAL = 971120040000
SEED_EVAL = 971120050000
LINKS = 128


def run_test_regime(regime):
    strong16.SEED_START = SEED_EVAL
    staging = Path('fresh_stage')
    staging.mkdir(exist_ok=True)
    return strong16.run_regime((regime, LINKS, str(staging)))


def calibration_record(i):
    seed = SEED_CAL+i
    obs, _ = simulate_link(seed, 9., 'full_reset', 'sparse')
    folds = []
    for aux in (strong16.AUX1, strong16.AUX2):
        c = candidates(obs,aux)
        folds.append(dict(tokens=np.asarray(c['tokens']).tolist(),pilot=np.asarray(c['pilot']).tolist(),
                          score=float(c['score']),used=bool(c['used'])))
    assert np.isclose(folds[0]['score'],folds[1]['score'],atol=1e-11,rtol=1e-11)
    return dict(seed=seed,regime='full_reset',folds=folds)


def dump_atomic(path, data):
    path=Path(path)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,allow_nan=False,separators=(',',':'))+'\n')
    tmp.replace(path)


if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=5) as workers:
        pending=[workers.submit(run_test_regime,r) for r in strong16.REGIMES]
        cal=list(workers.map(calibration_record,range(LINKS),chunksize=8))
        dump_atomic('cal_active.json',dict(records=cal))
        print('reset calibration done',len(cal),flush=True)
        records=sum((f.result() for f in pending), [])
    dump_atomic('strong_fresh.json',dict(records=records))
    print('strong replication done',len(records),flush=True)
