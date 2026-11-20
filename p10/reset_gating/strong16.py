"""Paired 9 dB experiment, frozen strong_large decoder, all 16 common payload blocks.

Run from the directory containing the experiment's cal.jsonl and receiver_next.py.
Each output payload block is selected on other rows of the same frame.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np

from p9.channel import simulate_link
from p9.contract import CORE_ROWS
from strong_eval import strong_errors

SEED_START = 971120030000
REGIMES = ('partial_shift','continuous','full_reset')
AUX1 = (2,5,8,13,16)
AUX2 = (1,4,7,12,15)
VALID = (0,9)
TARGETS = (tuple(r for r in CORE_ROWS if r not in AUX1), AUX1)
ALPHA = .05
BLOCKS = len(CORE_ROWS)
assert set(TARGETS[0]).isdisjoint(TARGETS[1]) and sum(map(len,TARGETS)) == BLOCKS
assert set(TARGETS[0] + TARGETS[1]) == set(CORE_ROWS)


def read_cal(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(set(r['seed'] for r in records)) != len(records):
        raise ValueError('Duplicate reset calibration seeds')
    if any(r.get('regime','full_reset') != 'full_reset' for r in records):
        raise ValueError('Expected full_reset calibration')
    return records


def pvalue(s, scores):
    return (1. + np.count_nonzero(scores >= s)) / (len(scores)+1.)


def check_gate_isolation(receiver, obs, aux, targets, ref):
    # The target-row LLRs themselves depend on the target observations. Channel states,
    # input scores and gating must remain invariant when those observations change.
    y = obs.y_b.copy()
    y[list(targets)] = 2.125 - .75j
    out = receiver(replace(obs, y_b=y), aux)
    assert bool(out['used']) == bool(ref['used'])
    for k in ('score','tokens','pilot'):
        np.testing.assert_allclose(out[k],ref[k],atol=1e-11,rtol=1e-11)
    for k in ('p9','v1'):
        for idx in (0,1,3):
            np.testing.assert_allclose(out[k][idx],ref[k][idx],atol=1e-10,rtol=1e-10)


def run_regime(job):
    from receiver_next import candidates
    regime, links, dest = job
    raw = []
    for i in range(links):
        seed = SEED_START + i
        obs, truth = simulate_link(seed, 9., regime, 'sparse')
        folds = []
        for j, (aux, targets) in enumerate(zip((AUX1,AUX2),TARGETS)):
            assert set(targets).isdisjoint((*aux,*VALID,*obs.training_rows,obs.probe_row))
            est = candidates(obs, aux)
            if i == 0:
                check_gate_isolation(candidates,obs,aux,targets,est)
            p9 = strong_errors(obs,truth,*est['p9'][:3],targets)
            if est['used']:
                v1 = strong_errors(obs,truth,*est['v1'][:3],targets)
            else:
                v1 = p9.copy()
            folds.append(dict(fold=j, targets=list(targets), score=float(est['score']), used=bool(est['used']),
                              tokens=np.asarray(est['tokens']).tolist(), pilot=np.asarray(est['pilot']).tolist(),
                              p9_errors=p9, v1_errors=v1))
        if not np.isclose(folds[0]['score'],folds[1]['score'],rtol=1e-11,atol=1e-11):
            raise ValueError('Reset score depends on auxiliary fold')
        raw.append(dict(seed=seed,regime=regime,folds=folds))
    Path(dest, f'.strong16_{regime}.json').write_text(json.dumps(raw, allow_nan=False))
    return raw


def summarize(rows):
    # All three mechanisms share seed-level trajectories. Independently sample seeds
    # once per bootstrap replicate and re-use the index across the mechanisms.
    rng = np.random.default_rng(971120)
    n = sum(r['regime'] == REGIMES[0] for r in rows)
    sampled = rng.integers(n,size=(10000,n))
    summary = {}
    for reg in REGIMES:
        traces = sorted([r for r in rows if r['regime']==reg], key=lambda r:r['seed'])
        arms = ('p9','v1','v2')
        e = {a:np.array([sum((f[a+'_errors'] for f in r['folds']),[]) for r in traces]) for a in arms}
        stats = dict(links=len(traces), blocks=len(traces)*BLOCKS, initial_seed=traces[0]['seed'],
                     methods={a:int(e[a].sum()) for a in arms}, paired={})
        for a,b in [('p9','v1'),('p9','v2'),('v1','v2')]:
            d = e[a]-e[b]
            avg = d.mean(axis=1)
            bounds = np.quantile(avg[sampled].mean(axis=1),[.025,.975])
            stats['paired'][a+'_minus_'+b] = dict(gain_pp=float(100*avg.mean()),ci_pp=(100*bounds).tolist(),
                                                  rescue=int(np.count_nonzero(d>0)),loss=int(np.count_nonzero(d<0)))
        for arm in ('v1','v2'):
            key = 'used' if arm == 'v1' else 'v2_used'
            stats[arm+'_switch_folds'] = sum(sum(int(f[key]) for f in r['folds']) for r in traces)
            stats[arm+'_switch_links'] = sum(any(f[key] for f in r['folds']) for r in traces)
        summary[reg] = stats
    return summary


def finalize(raw,cal,out):
    if {r['seed'] for r in cal} & {r['seed'] for r in raw}:
        raise ValueError('Reset calibration and scored links require disjoint seeds')
    scores = np.array([r['score'] for r in cal])
    if not np.isfinite(scores).all():
        raise ValueError('Invalid pilot evidence')
    for r in raw:
        for f in r['folds']:
            pv = pvalue(f['score'],scores)
            f.update(reset_p=pv,v2_used=bool(f['used'] and pv<=ALPHA),
                     v2_errors=f['v1_errors'] if pv<=ALPHA else f['p9_errors'])
    summary=summarize(raw)
    out.write_text(json.dumps(dict(summary=summary, alpha=ALPHA, calibration_n=len(scores), records=raw),
                              allow_nan=False,separators=(',',':'))+'\n')
    print(json.dumps(summary,indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--links',type=int,default=128)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--out',type=Path,default=Path('strong16.json'))
    parser.add_argument('--cal',type=Path,default=Path('cal.jsonl'))
    parser.add_argument('--finalize-only',action='store_true',help='Use completed per-regime raw files')
    parser.add_argument('--raw-only',action='store_true',help='Freeze calibration rule and decode independently')
    args=parser.parse_args()
    jobs=[(regime,args.links,str(args.out.parent)) for regime in REGIMES]
    if args.finalize_only:
        raw=[json.loads((args.out.parent/f'.strong16_{regime}.json').read_text()) for regime in REGIMES]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            raw=list(executor.map(run_regime,jobs))
    if not args.raw_only:
        finalize(sum(raw,[]),read_cal(args.cal),args.out)


if __name__=='__main__':
    main()
