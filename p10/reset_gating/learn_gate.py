"""A small query-key gate and a parameter-matched global MLP for auditing P10 vetoes.

Fit uses regime labels solely from the training trajectories. The inference entry
point takes an (N,7,3) array of public, row-level scores and (N,2) pilot scores.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp, softmax

REGIMES = ("partial_shift", "continuous", "full_reset")
N_HEAD = 2
REG = 1e-3
STARTS = (1869, 8122, 7623)
ARCHS = ("linear", "mlp", "attention")


def read_records(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    identities = [(r['seed'], r['regime']) for r in rows]
    if not rows or len(identities) != len(set(identities)):
        raise ValueError('Empty or duplicated physical seed/regime records')
    return rows


def observations(rows):
    data = np.asarray([r['tokens'] for r in rows], dtype=float)
    pilot = np.asarray([r['pilot'] for r in rows], dtype=float)
    if data.shape != (len(rows),7,3) or pilot.shape != (len(rows),2):
        raise ValueError('Expected seven rows of three public scores and two pilot scores')
    if not (np.isfinite(data).all() and np.isfinite(pilot).all()):
        raise ValueError('Incomplete public scores')
    return data, pilot


def raw_features(tokens, pilot):
    # Invertible changes of basis; no mechanism/correctness information.
    x = np.stack([tokens[:,:,0]-tokens[:,:,1], tokens[:,:,0]-tokens[:,:,2], tokens[:,:,0]], axis=2)
    u = np.column_stack([pilot[:,0]-pilot[:,1],pilot[:,1]])
    return x, u


def fit_norm(tokens, pilot):
    x, u = raw_features(tokens, pilot)
    def moments(a, axis):
        median = np.median(a, axis=axis)
        width = np.quantile(a, .75, axis=axis) - np.quantile(a, .25, axis=axis)
        return [median.tolist(), np.maximum(width, 1e-4).tolist()]
    return dict(token=moments(x,(0,1)), pilot=moments(u,0))


def design(tokens, pilot, norm):
    x,u = raw_features(tokens, pilot)
    def transform(a, pars):
        center, width = (np.asarray(v) for v in pars)
        return np.arcsinh((a-center)/width)
    x,u = transform(x,norm['token']), transform(u,norm['pilot'])
    g = np.column_stack([np.ones(len(x)), x.reshape(len(x),21), u])  # shared linear readout, dim 24
    # Explicit row coordinates; MLP sees all seven labeled row slots in g.
    row_id = np.broadcast_to(np.eye(7)[:,:6], (len(x),7,6))
    keys = np.concatenate([x,row_id],axis=2)  # dim 9
    query = np.column_stack([np.ones(len(x)),u,x[:,:5,:2].mean(1),x[:,5:,:2].mean(1)])  # dim 7
    return g, keys, query


def nparams(arch):
    return 24 if arch == 'linear' else 24 + 25*N_HEAD


def forward(arch, theta, data, target=None):
    g, x, q = data
    logits = g @ theta[:24]
    cache = []
    if arch != 'linear':
        for head in theta[24:].reshape(N_HEAD,25):
            if arch == 'mlp':
                hidden = np.tanh(g @ head[:24])
                logits += head[24] * hidden
                cache.append(hidden)
            else:
                query = np.tanh(q @ head[:7])
                key = x @ head[7:16]
                value = x @ head[16:]
                weight = softmax(query[:,None] * key,axis=1)
                out = np.sum(weight*value,axis=1)
                logits += out
                cache.append((query,key,value,weight,out))
    if target is None:
        return logits
    dz = (expit(logits) - target)/len(target)
    loss = np.mean(np.logaddexp(0.,logits) - target*logits)
    grad = np.zeros_like(theta)
    grad[:24] = g.T @ dz
    for j, info in enumerate(cache):
        lo = 24 + 25*j
        head = theta[lo:lo+25]
        if arch == 'mlp':
            grad[lo:lo+24] = g.T @ (dz * head[24] * (1-info**2))
            grad[lo+24] = dz @ info
        else:
            query,key,value,weight,out = info
            dm = dz[:,None]*weight*(value-out[:,None])
            grad[lo:lo+7] = q.T @ ((dm*key).sum(1)*(1-query**2))
            grad[lo+7:lo+16] = np.einsum('nr,nrd->d',dm*query[:,None],x)
            grad[lo+16:lo+25] = np.einsum('nr,nrd->d',dz[:,None]*weight,x)
    # Same L2 penalty and exclusion of the shared classifier intercept.
    return loss + REG * (theta[1:] @ theta[1:])/2, grad + REG * np.r_[0.,theta[1:]]


def fit(rows):
    tokens, pilot = observations(rows)
    norm = fit_norm(tokens, pilot)
    data = design(tokens,pilot,norm)
    labels = np.array([r['regime']=='partial_shift' for r in rows],float)
    result = {}
    for arch in ARCHS:
        attempts = []
        for start in STARTS:
            init = np.random.default_rng(start).normal(0.,.2,size=nparams(arch))
            opt = minimize(lambda w: forward(arch,w,data,labels),init,jac=True,
                           method='L-BFGS-B',options=dict(maxiter=320,ftol=1e-10,gtol=1e-7))
            if not (np.isfinite(opt.fun) and np.isfinite(opt.x).all()):
                raise ValueError(f'Optimization failed: {arch} {start}: {opt.message}')
            attempts.append((opt.fun,opt.x,opt.nit,opt.success))
        best = min(attempts,key=lambda a:a[0])
        result[arch] = dict(parameters=best[1].tolist(), train_objective=float(best[0]),
                            start_diagnostics=[dict(start=s,iterations=int(v[2]), objective=float(v[0]),
                                                    converged=bool(v[3])) for s,v in zip(STARTS,attempts)])
    return dict(train_seeds=sorted(set(r['seed'] for r in rows)),
                train_regime_counts={g:sum(r['regime']==g for r in rows) for g in REGIMES},
                normalization=norm,models=result)


def decisions(fitted, tokens, pilot):
    data=design(tokens,pilot,fitted['normalization'])
    return {arch:expit(forward(arch,np.array(fitted['models'][arch]['parameters']),data)) for arch in ARCHS}


def evaluate(fitted, rows):
    seeds=np.array([r['seed'] for r in rows])
    regimes=np.array([r['regime'] for r in rows])
    groups=np.unique(seeds)
    if np.intersect1d(groups,fitted['train_seeds']).size:
        raise ValueError('Training/test physical seed overlap')
    for reg in REGIMES:
        if not np.array_equal(np.unique(seeds[regimes==reg]),groups):
            raise ValueError('Expected matched seed sets across regimes')
    probs=decisions(fitted,*observations(rows))
    used=np.array([r['used'] for r in rows],bool)
    incumbent=np.array([r['p9_errors'] for r in rows],int)
    repair=np.array([r['v1_errors'] for r in rows],int)
    if (incumbent.ndim!=2 or incumbent.shape!=repair.shape or
        not np.all(np.isin(incumbent,[0,1])) or not np.all(np.isin(repair,[0,1])) or
        not np.array_equal(incumbent[~used],repair[~used])):
        raise ValueError('Expected paired per-word error flags, identical when repair is unused')
    target=regimes=='partial_shift'
    choices={'p9':np.zeros(len(rows),bool), 'v1':used}
    choices.update({arch:used & (p>=.5) for arch,p in probs.items()})
    # These two are evaluator-only and cannot be used as receiver outputs.
    choices['mechanism_oracle']=used & target
    choices['utility_oracle']=used & (repair.sum(1)<incumbent.sum(1))
    err={k:np.where(c[:,None],repair,incumbent) for k,c in choices.items()}
    results={}; intervals={}
    rng=np.random.default_rng(49402)
    b=rng.integers(0,len(groups),size=(4000,len(groups)))
    for reg in (*REGIMES,'equal_weight'):
        idx=np.flatnonzero(regimes==reg) if reg!='equal_weight' else np.arange(len(rows))
        stats={}
        for name,e in err.items():
            values=dict(block_errors=int(e[idx].sum()),bler=float(e[idx].mean()), accepted=int(choices[name][idx].sum()))
            if name in probs:
                vals=np.clip(probs[name][idx],1e-15,1-1e-15); t=target[idx]
                values.update(brier=float(np.mean((vals-t)**2)),logloss=float(-np.mean(np.where(t,np.log(vals),np.log1p(-vals)))))
            stats[name]=values
        results[reg]=dict(links=len(idx), blocks=int(incumbent[idx].size), arms=stats)
        cis={}
        for ref in ('v1','linear','mlp'):
            diffs=(err[ref][idx] - err['attention'][idx]).mean(1)
            per_seed=np.array([diffs[seeds[idx]==s].mean() for s in groups])
            cis[ref+'_minus_attention']=dict(gain=float(per_seed.mean()),
                percentile_95=np.quantile(per_seed[b].mean(1),[.025,.975]).tolist())
        intervals[reg]=cis
    records=[dict(seed=int(seeds[i]),regime=r['regime'],used=bool(used[i]),
                  prob={k:float(p[i]) for k,p in probs.items()}) for i,r in enumerate(rows)]
    return dict(result=results,paired_seed_intervals=intervals,probabilities=records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train');p.add_argument('--eval');p.add_argument('--fitted',default='learn_weights.json')
    p.add_argument('--out',default='learn_results.json')
    args=p.parse_args()
    if not (args.train or args.eval):
        p.error('Provide --train and/or --eval.')
    if args.train:
        fitted=fit(read_records(args.train))
        fitted['train_sha256']=hashlib.sha256(Path(args.train).read_bytes()).hexdigest()
        Path(args.fitted).write_text(json.dumps(fitted,indent=2)+'\n')
    if args.eval:
        fitted=json.loads(Path(args.fitted).read_text())
        result=evaluate(fitted,read_records(args.eval))
        result['eval_sha256']=hashlib.sha256(Path(args.eval).read_bytes()).hexdigest()
        Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result['result'],indent=2))


if __name__=='__main__':
    main()
