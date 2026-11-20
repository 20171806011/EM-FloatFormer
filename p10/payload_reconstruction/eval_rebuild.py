"""Frozen 9 dB rebuild trial, with two out-of-fold strong_large evaluations per link.

The estimator sees B's known pilot and five auxiliary unknown rows. No target (even
unlabeled) or B probe is passed as a training/evidence row. All hidden comparisons,
including the deliberately privileged phase oracle, run after the public choices.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import lru_cache
import json
import os
from pathlib import Path
import time

import numpy as np

from active_rank import extract
from learn_gate import decisions, observations
from p9.channel import readout_frame_b, simulate_link
from p9.contract import CORE_ROWS
from p9.gating import build_candidates
from receiver_next import candidates, covariance_scores
from strong_eval import strong_errors
from strong16 import AUX1, AUX2, TARGETS, VALID


REGIMES = ('full_reset','partial_shift','continuous')
SEED_START = 981120030000
COUNT = 128
ORACLE_LINKS = 32
SNR_DB = 9.


@lru_cache(maxsize=1)
def prior_gate(prior_dir):
    with open(Path(prior_dir)/'learn_weights.json') as stream:
        fitted = json.load(stream)
    with open(Path(prior_dir)/'cal_active.json') as stream:
        records = json.load(stream)['records']
    assert len(records) == 128 and all(r['regime']=='full_reset' for r in records)
    assert len({r['seed'] for r in records}) == 128
    assert not set(r['seed'] for r in records).intersection(range(SEED_START,SEED_START+COUNT))
    flags, scores = extract(records,fitted)
    maxima = np.where(flags, scores['attention'], -np.inf).max(axis=1)
    return fitted, maxima


def choose_incumbent(old, prior_dir):
    weights, ranks = prior_gate(prior_dir)
    tokens,pilot = observations([old])
    key = float(decisions(weights,tokens,pilot)['attention'][0])
    pvalue = (1+int(np.count_nonzero(ranks>=key)))/(1+len(ranks))
    return ('v1' if old['used'] and pvalue<=.05 else 'p9'), pvalue, key


def as_readout(obs, mu, var):
    mu = np.asarray(mu)
    var = np.asarray(var)
    if mu.shape!=var.shape or mu.shape!=(3,*obs.y_b.shape):
        raise ValueError('Both ray-level arrays must be (3,T,N)')
    if not np.isfinite(mu).all() or not np.isfinite(var).all() or np.any(var<0):
        raise ValueError('Invalid estimator output')
    llrs, probe, _ = readout_frame_b(mu,var,obs.y_b,obs.training_rows,obs.n0)
    return mu,var,llrs,probe


def assert_isolation(obs, aux, enabled, ref):
    from estimator import estimate
    contam = obs.y_b.copy()
    keep = set(obs.training_rows) | (set(aux) if enabled else set())
    for r in range(len(contam)):
        if r not in keep:
            contam[r] = (100.+r)*np.exp(1j*np.arange(contam.shape[1]))
    changed = replace(obs, y_a=np.full_like(obs.y_a, 50-150j), y_b=contam)
    alt = estimate(changed,aux,use_payload=enabled)
    np.testing.assert_allclose(alt[0],ref[0],rtol=1e-10,atol=1e-11)
    np.testing.assert_allclose(alt[1],ref[1],rtol=1e-10,atol=1e-11)


def oracle_common(obs, truth, candidate, targets):
    mu,var = candidate[:2]
    rotated = mu.copy()
    for row in targets:
        inner = np.vdot(mu[:,row],truth.h_b_true[:,row])
        phase = inner/abs(inner) if abs(inner)>1e-14 else 1.
        rotated[:,row] *= phase
    model = as_readout(obs,rotated,var)
    return strong_errors(obs,truth,*model[:3],targets)


def public_models(obs, aux, clear, prior_dir):
    from estimator import estimate
    from code_check import list_scores
    old = candidates(obs,aux)
    incumbent,pval,att = choose_incumbent(old,prior_dir)
    pilot = as_readout(obs,*estimate(obs,aux,use_payload=False))
    new = as_readout(obs,*estimate(obs,aux,use_payload=True))
    models = dict(p9=old['p9'],v1=old['v1'],clear=clear,pilot_only=pilot,rebuild=new)
    default = models[incumbent]
    val_scores = {name:covariance_scores(obs,*m[:2],VALID).tolist()
                  for name,m in [('default',default),('rebuild',new)]}
    totals = {name:sum(s) for name,s in val_scores.items()}
    action = 'rebuild' if totals['rebuild']>totals['default'] else incumbent
    code_scores = [float(s) for s in list_scores(obs,default,new,rows=VALID)]
    assert len(code_scores)==2 and np.isfinite(code_scores).all()
    code_used = code_scores[1] > code_scores[0]
    return models,dict(incumbent=incumbent, action=action, scores=totals, val_per_row=val_scores,
                       v1_proposed=old['proposed'],v1_used=old['used'],attention_pvalue=pval,
                       attention_score=att,code_scores=code_scores,code_used=code_used)


def process(job):
    from phase_check import metrics
    regime,seed,prior_dir = job
    obs,truth = simulate_link(seed,SNR_DB,regime,'sparse')
    c = build_candidates(obs)[1]
    clear = c.mean_ray,c.var_ray,c.llrs,c.probe_pred
    folds=[]
    for index,(aux,targets) in enumerate(zip((AUX1,AUX2),TARGETS)):
        assert not set(targets).intersection((*aux,*VALID,*obs.training_rows,obs.probe_row))
        models, f = public_models(obs,aux,clear,prior_dir)
        if seed==SEED_START:
            for name, enabled in [('pilot_only',False),('rebuild',True)]:
                assert_isolation(obs,aux,enabled,models[name])
        # Truth is used for the following scoring only, after the public actions.
        errors = {}
        for name,model in models.items():
            if name=='v1' and not f['v1_used']:
                errors[name]=errors['p9'].copy()
            else:
                errors[name]=strong_errors(obs,truth,*model[:3],targets)
        sse={name:float(np.sum(abs(m[3]-truth.y_b_probe_clean)**2)) for name,m in models.items()}
        for a, choice in [('default',f['incumbent']),('gate',f['action']),
                          ('code_gate','rebuild' if f['code_used'] else f['incumbent'])]:
            errors[a] = errors[choice].copy()
            sse[a] = sse[choice]
        phases={name:metrics(models[name][0],truth.h_b_true,rows=targets)
                for name in ['clear','pilot_only','rebuild']}
        f.update(fold=index,aux=list(aux),targets=list(targets),errors=errors,probe_sse=sse,phase=phases)
        if regime=='full_reset' and seed<SEED_START+ORACLE_LINKS:
            f['oracle_common_phase'] = {
                name:oracle_common(obs,truth,models[name],targets) for name in ('pilot_only','rebuild')}
        folds.append(f)
    return dict(seed=seed,regime=regime,probe_energy=float(np.sum(abs(truth.y_b_probe_clean)**2)),folds=folds)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-dir',required=True,type=Path)
    parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--out',type=Path,default=Path('rebuild_raw.json'))
    parser.add_argument('--resume',action='store_true',help='reuse identical frozen inputs and partial records')
    args=parser.parse_args()
    if os.environ.get('OPENBLAS_NUM_THREADS')!='1':
        raise ValueError('Set OPENBLAS_NUM_THREADS=1 before launching the trial')
    start=time.monotonic()
    prior_dir=str(args.prior_dir.resolve())
    prior_gate(prior_dir)
    jobs = [(reg,SEED_START+i,prior_dir) for i in range(COUNT) for reg in REGIMES]
    interim = args.out.with_suffix('.progress.jsonl')
    rows=[]
    if args.resume and interim.exists():
        rows=[json.loads(line) for line in interim.read_text().splitlines() if line.strip()]
    already={(r['regime'],r['seed']) for r in rows}
    assert len(rows)==len(already) and already.issubset({(r,s) for r,s,_ in jobs})
    todo=[j for j in jobs if j[:2] not in already]
    with interim.open('a' if args.resume else 'w') as stream:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for r in pool.map(process,todo,chunksize=1):
                stream.write(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n')
                stream.flush()
                rows.append(r)
                if len(rows)%32==0:
                    print(f'{len(rows)}/{len(jobs)} links, elapsed {time.monotonic()-start:.1f}s',flush=True)
    assert len(rows)==len(jobs)
    elapsed=time.monotonic()-start
    data=dict(links_per_regime=COUNT,seed_start=SEED_START,regimes=REGIMES,snr_db=SNR_DB,
              workers=args.workers,elapsed_seconds=elapsed,records=rows)
    tmp=args.out.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,separators=(',',':'),allow_nan=False)+'\n')
    tmp.replace(args.out)
    print(f'Wrote {args.out} in {elapsed:.1f}s',flush=True)


if __name__=='__main__':
    main()
