"""Calibrate and test the no-I/Q-replay Q2 reader at the pinned Stage 3 C4 condition."""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from q2_reader import read


RHO = 10**.2
SD = (2*RHO)**-.5
ROOTS = np.array([1, 1j, -1, -1j])
LABEL = np.arange(4)
SEED = 920617
CHECKPOINT_SHA = "e67d9fbafb90791ab19549de7789798d818286d7f52d7ac566cc39df71e78356"


def gaussian(order, mean, sd):
    x, w = np.polynomial.hermite.hermgauss(order)
    return np.asarray(mean)[..., None] + math.sqrt(2)*sd*x, w/math.sqrt(math.pi)


def normalize(v):
    return v / v.sum(axis=-1, keepdims=True)


def fit_likelihood(model, count, seed):
    # Importance proposal is the equally weighted four on-grid Gaussian components.
    # Quarter-turn equivariance means an invariant expectation needs one component.
    rng = np.random.default_rng(seed)
    y = 1/math.sqrt(3) + SD*(rng.normal(size=(count, 3)) + 1j*rng.normal(size=(count, 3)))
    t = y.sum(axis=1)/math.sqrt(3)
    code = model.encode(y)
    basis = 2*RHO*t
    denom = np.cosh(basis.real) + np.cosh(basis.imag)
    em0 = np.zeros(16)
    for r in LABEL:
        # Estimate p(z|U=0) with all rotated proposal draws.
        weight = np.exp((basis*ROOTS[r]).real)*2/denom
        em0 += np.bincount((code+4*r)%16, weights=weight, minlength=16)
    em0 = em0/em0.sum()
    table = normalize(np.column_stack([np.roll(em0, 4*r) for r in LABEL]))
    def marginal(angle):
        out = []
        for theta in np.asarray(angle).ravel():
            shifted = basis*np.exp(-1j*theta)
            weight = (np.cosh(shifted.real) + np.cosh(shifted.imag))/denom
            bins = np.bincount(code%4, weights=weight, minlength=4)
            out.append(np.tile(bins/bins.sum()/4, 4))
        return np.array(out).reshape((*np.shape(angle), 16))
    return table, marginal


def make_bridge(marginal, order, h=2., pilot_time=4., omega=.35, sigma=.10):
    # Unwrapped Gaussian bridge. Over a fixed pilot phase, marginalize second code.
    phase, w = gaussian(order, omega*pilot_time, sigma*math.sqrt(pilot_time))
    cov_t = min(h, pilot_time)
    if pilot_time == 0:
        mean = np.full_like(phase, omega*h)
        cond_var = sigma**2*h
    else:
        mean = omega*h + cov_t/pilot_time*(phase-omega*pilot_time)
        cond_var = sigma**2*(h-cov_t**2/pilot_time)
    if cond_var == 0:
        by_code = marginal(mean)
    else:
        inside, wi = gaussian(order, mean, math.sqrt(cond_var))
        by_code = np.einsum('jlk,l->jk', marginal(inside), wi)
    return phase, normalize((w[:, None]*by_code).T), w


def posterior(a1, z2, pilot_sum, phase, conditional, w, condition_on_z2=True):
    # All measurements passed here must be unique acquisitions at one pilot time.
    model_means = np.exp(1j*phase[:, None])*ROOTS
    log_l = 2*RHO*(pilot_sum[:, None, None]*model_means.conj()).real
    likelihood = np.exp(log_l - log_l.max(axis=(1,2), keepdims=True))
    drift = conditional[z2] if condition_on_z2 else np.broadcast_to(w, (len(z2),len(w)))
    g = normalize(np.einsum('nj,njk->nk', drift, likelihood))
    return np.column_stack([np.sum(g * a1[:, (LABEL+k)%4], axis=1) for k in LABEL])


def factor_check(hand_emission):
    """Directly enumerate x1,x2,g and TWO shared path atoms. Pilots are noisy."""
    g = np.exp(1j*np.pi/2*LABEL)
    paths = [(.31, -.53, -.36), (.69, .72, 1.18)]
    split = .68-.41j
    E1 = hand_emission(np.pi/2*LABEL)
    z1, z2 = 3,12
    def brute(obs):
        out = np.zeros(4)
        for mass, ph2, php in paths:
            E2 = hand_emission(np.pi/2*LABEL+ph2)
            L = np.exp(2*RHO*(np.sum(obs)*np.conj(g*np.exp(1j*php))).real)
            for x1 in LABEL:
                for x2 in LABEL:
                    for gi in LABEL:
                        out[x1] += mass/64*E1[(x1+gi)%4,z1]*E2[(x2+gi)%4,z2]*L[gi]
        return out/out.sum()
    maxima = []
    for observations in ([split], [split, -.27+.86j]):
        w = np.array([p[0] for p in paths])
        phase = np.array([p[2] for p in paths])
        E2 = np.stack([hand_emission(np.pi/2*LABEL+ph2).mean(0) for _,ph2,_ in paths])
        cond = normalize((w[:,None]*E2).T)
        q = posterior(normalize(E1[:, z1])[None,:], np.array([z2]), np.array([sum(observations)]),phase,cond,w)[0]
        maxima.append(np.max(np.abs(q-brute(observations))))
    assert max(maxima) < 2e-14
    return max(maxima)


def bench(repo, count, fit_count, export_path):
    sys.path[:0] = [str(repo), str(repo/'research/matched_cache'), str(repo/'research/stage3b_math/A')]
    import learned
    import check_a
    from research.stage3_ministral_cache.analytic import readout_from_table
    from research.stage3_ministral_cache.schema import CONDITION_SPECS, Reference
    weights = repo/'research/matched_cache/weights/equivariant_b8_s17.npz'
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == CHECKPOINT_SHA
    enc = learned.load(weights)
    table, marginal = fit_likelihood(enc, fit_count, 41117)
    bridge = make_bridge(marginal, 20)
    bridge2 = make_bridge(marginal, 28)
    hand = lambda t: check_a.hand_emission(np.asarray(t)[...,None]+np.pi/2*LABEL).mean(axis=-2)
    finite_err = factor_check(check_a.hand_emission)
    nodes, weights, prior = bridge
    kernel = dict(encoder='equivariant_b8_s17', checkpoint_sha256=CHECKPOINT_SHA,
                  condition='C4_cfo_ramp', h=2, pilot_time=4, omega=.35, sigma=.10,
                  rho=RHO, calibration_seed=41117, calibration_count=fit_count,
                  pilot_phase_nodes=nodes.tolist(), u_given_z1=table.tolist(),
                  weights_given_z2=weights.tolist())
    # Compare public reader and its duplicate/independent-reference semantics
    # against the vectorized estimator for all 256 code pairs and both pilot arms.
    serialized=json.dumps(kernel, indent=2)+'\n'
    decoded=json.loads(serialized)
    idx1, idx2 = np.indices((16,16)).reshape(2,-1)
    example1, example2 = .38-.76j, 1.21+.44j
    read_err=0.0
    for obs in [[example1],[example1,example2]]:
        expected=posterior(table[idx1],idx2,np.full(256,sum(obs)),*bridge)
        refs=[(a,True,t) for a,t in enumerate(obs)]
        for n,(i,j) in enumerate(zip(idx1.tolist(),idx2.tolist())):
            q=read(decoded,i,j,refs)
            read_err=max(read_err,float(np.max(np.abs(q-expected[n]))))
            assert q == read(decoded,i,j,refs+[refs[0],('other',False,-4.9j)])
            assert read(decoded,i,j,[('other',False,-4.9j)]) == [.25]*4
    assert read_err < 2e-14
    if export_path:
        export_path.write_text(serialized,encoding='utf-8')
    # Coincident and zero-time bridge limits: weights are exactly a product at coincident time,
    # constant p when the pilot has zero prior time.
    phase, cond, w = make_bridge(hand, 16, h=4., pilot_time=4.)
    np.testing.assert_allclose(cond, normalize((w[:,None]*hand(phase)).T), atol=1e-14)
    _, zero, w0 = make_bridge(hand, 16, pilot_time=0.)
    np.testing.assert_allclose(zero, np.broadcast_to(w0, zero.shape), atol=1e-14)
    rng = np.random.default_rng(SEED)
    agg={}
    gain = np.zeros(4)
    ref_err = grid_err = tv = 0.
    for start in range(0,count,2048):
        n = min(count-start,2048)
        u,v,g = rng.integers(4,size=(3,n))
        truth=(u-g)%4
        ph2, extra=rng.normal(.7,.1*math.sqrt(2),size=(2,n))
        php=ph2+extra
        def window(sym, phase):
            noise = rng.normal(size=(n,3)) + 1j*rng.normal(size=(n,3))
            return ROOTS[sym,None]*np.exp(1j*np.asarray(phase))[...,None]/math.sqrt(3)+SD*noise
        z1=enc.encode(window(u,0))
        z2=enc.encode(window(v,ph2))
        p1=window(g,php)
        p2=window(g,php)
        for arm,raw in [('valid',[p1]),('fresh',[p1,p2])]:
            p=sum(wave.sum(axis=1)/math.sqrt(3) for wave in raw)
            legacy=posterior(enc.table[z1],z2,p,*bridge,condition_on_z2=False)
            first=posterior(table[z1],z2,p,*bridge,condition_on_z2=False)
            fixed=posterior(enc.table[z1],z2,p,*bridge)
            joint=posterior(table[z1],z2,p,*bridge)
            if arm=='valid':
                valid_hits=joint.argmax(-1)==truth
            else:
                diff=(joint.argmax(-1)==truth).astype(float)-valid_hits
                gain += [diff.sum(), diff@diff, (diff>0).sum(), (diff<0).sum()]
            if start==0:
                refs=[Reference(j,True,wave[:128],4.,2.) for j,wave in enumerate(raw)]
                check=readout_from_table(enc.table[z1[:128]],enc.table[z2[:128]],refs,CONDITION_SPECS['C4_cfo_ramp'])['q2']
                ref_err=max(ref_err,float(np.max(np.abs(check-legacy[:128]))))
                check=posterior(table[z1[:128]],z2[:128],p[:128],*bridge2)
                grid_err=max(grid_err,float(np.max(np.abs(check-joint[:128]))))
            tv=max(tv,float(np.max(np.abs(joint-first))))
            ix=np.arange(n)
            for tag,q,base in [('legacy',legacy,legacy),('calibrate_z1',first,legacy),
                              ('condition_z2',fixed,legacy),('joint',joint,legacy),
                              ('joint_vs_calibrated',joint,first)]:
                correct=(q.argmax(-1)==truth).astype(float)
                diff=correct-(base.argmax(-1)==truth)
                nll=-np.log(q[ix,truth])
                dnll=nll+np.log(base[ix,truth])
                arr=agg.setdefault((arm,tag),np.zeros(8))
                arr += [correct.sum(),nll.sum(),diff.sum(),diff@diff,dnll.sum(),dnll@dnll,
                        np.sum(diff>0),np.sum(diff<0)]
    def summarize(a):
        def se(s,ss):
            return float(np.sqrt(max(0,ss-s*s/count)/(count*(count-1))))
        return dict(acc=a[0]/count,nll=a[1]/count,delta_acc=a[2]/count,
                    paired_acc_se=se(a[2],a[3]),delta_nll=a[4]/count,paired_nll_se=se(a[4],a[5]),
                    corrected=int(a[6]),new_errors=int(a[7]))
    assert ref_err<1e-8 and grid_err<1e-8
    return dict(condition='C4_cfo_ramp',encoder='equivariant_b8_s17',seed=SEED,
                mothers=count,calibration_seed=41117,calibration_windows=fit_count,
                quadrature_order=20,finite_enumeration_max_error=float(finite_err),
                release_reader_max_error=read_err, release_posteriors_checked=512,
                fresh_minus_valid=dict(delta_acc=gain[0]/count, paired_se=math.sqrt(
                    (gain[1]-gain[0]**2/count)/(count*(count-1))),
                    corrected=int(gain[2]),new_errors=int(gain[3])),
                pilot_grid20_vs_grid28_max_first128=grid_err,
                legacy_vs_pinned_max_first128=ref_err,max_p_change_condition_on_z2=tv,
                results={arm:{tag:summarize(a) for (arm2,tag),a in agg.items() if arm2==arm}
                         for arm in ('valid','fresh')})


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--repo',type=Path,default=Path.cwd())
    cli.add_argument('--mothers',type=int,default=262144)
    cli.add_argument('--calibration',type=int,default=60000)
    cli.add_argument('--export-kernel',type=Path)
    args=cli.parse_args()
    print(json.dumps(bench(args.repo.resolve(),args.mothers,args.calibration,args.export_kernel),indent=2))
