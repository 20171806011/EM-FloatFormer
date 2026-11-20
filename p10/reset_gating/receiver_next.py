"""Compare a pre-B single-path likelihood with the independent reset model."""
from functools import lru_cache
import numpy as np
from scipy.special import logsumexp, softmax
from p8c.link import PILOTS
from p9.channel import (_training_dictionary, build_sc_channel_matrices, readout_frame_b,
                        sample_moments_b, track_frame_a)
from p9.contract import DELAYS, N
from p9.gating import build_candidates, gate_jeffreys_student_t
from p10_covariance import BANK, CLASS, LOG_PRIOR, TIMES
from p10_repair import CONFIDENCE, MODES, VALIDATION, refit
from reset_density import reset_logpdf


def pilot_evidence(obs, state):
    """Per-ray mean and centered covariance, then rotate before moment matching."""
    rows = tuple(obs.training_rows)
    assert rows == (10,)
    design, _, _ = _training_dictionary(state.epoch, rows)
    w = np.exp(state.log_weight)
    mean = np.einsum('rh,rhn->rn', w*state.mean, design)
    power = w*(abs(state.mean)**2 + state.variance)
    gram = (design.swapaxes(1,2) * power[:,None,:]) @ design.conj()
    covariance = gram - mean[:,:,None]*mean[:,None,:].conj()
    baseline = covariance.sum(0) + obs.n0*np.eye(N)
    row = rows[0]
    ll = []
    for ray, delta in BANK[1:]:
        rotation = np.exp(2j*np.pi*delta*TIMES[row])
        m = mean.sum(0) + (rotation - 1)*mean[ray]
        cov = baseline + (rotation[:,None]*rotation[None,:].conj()-1)*covariance[ray]
        c = np.linalg.cholesky((cov + cov.conj().T)/2)
        u = np.linalg.solve(c, obs.y_b[row] - m)
        ll.append(-np.vdot(u,u).real - 2*np.log(c.diagonal().real).sum() - N*np.log(np.pi))
    return np.array([logsumexp(ll)-np.log(len(ll)),reset_logpdf(obs)])


def covariance_scores(obs, mu, var, rows):
    h = build_sc_channel_matrices(mu, rows=rows)
    return matrix_scores(obs, h, var, rows)


def matrix_scores(obs, h, var, rows):
    cov = h @ h.conj().swapaxes(-1,-2)
    n = np.arange(N)
    cov[...,n,n] += obs.n0 + var[:,list(rows)].sum(0)
    y = obs.y_b[list(rows)]
    z = np.linalg.solve(cov, np.broadcast_to(y,cov.shape[:-1])[...,None])[...,0]
    return -(y.conj()*z).sum(-1).real - np.linalg.slogdet(cov)[1]


def candidates(obs, aux):
    aux = tuple(aux)
    evidence_rows = (*aux,*VALIDATION)
    assert len(set(evidence_rows + tuple(obs.training_rows) + (obs.probe_row,))) == len(evidence_rows)+2
    state = track_frame_a(obs.y_a, obs.n0)
    mean, var = sample_moments_b(state)
    pilot = pilot_evidence(obs, state)
    rows = (*obs.training_rows,*aux)
    transforms = np.broadcast_to(mean[:,list(rows)],(len(BANK),3,len(rows),N)).copy()
    for j,(ray,delta) in enumerate(BANK[1:],1):
        transforms[j,ray] *= np.exp(2j*np.pi*delta*TIMES[list(rows)])
    n = np.arange(N)
    matrices = np.zeros((len(BANK),len(rows),N,N),complex)
    for r,d in enumerate(DELAYS):
        matrices[:,:,n,(n-d)%N] = transforms[:,r]
    row = rows[0]
    b = matrices[:,0] @ PILOTS[row]
    s_p = LOG_PRIOR - (abs(obs.y_b[row]-b)**2/(obs.n0+var[:,row].sum(0))).sum(-1)
    w = softmax(s_p + matrix_scores(obs,matrices[:,1:],var,aux).sum(-1))
    masses = [w[CLASS==r].sum() for r in range(3)]
    ray = int(np.argmax(masses))
    ids = np.flatnonzero(CLASS==ray)
    top = ids[np.argsort(w[ids])[-MODES:]]
    mode_weights = softmax(np.log(np.maximum(w[top], 1e-300)))
    branches = [refit(obs,state,*BANK[j]) for j in top]
    mu = sum(q*m for q,(m,v) in zip(mode_weights,branches))
    v = sum(q*(s+abs(m-mu)**2) for q,(m,s) in zip(mode_weights,branches))
    inh,clear = build_candidates(obs)
    old = gate_jeffreys_student_t(obs,inh,clear)
    base = (old.mix_mean_ray,old.mix_var_ray,old.mix_llrs,old.mix_probe_pred)
    repair_score = covariance_scores(obs,mu,v,evidence_rows)
    tokens = np.stack([repair_score,covariance_scores(obs,inh.mean_ray,inh.var_ray,evidence_rows),
                       covariance_scores(obs,clear.mean_ray,clear.var_ray,evidence_rows)],axis=1)
    proposed = masses[ray]>=CONFIDENCE
    used = proposed and repair_score[-2:].sum()>covariance_scores(obs,*base[:2],VALIDATION).sum()
    repair = base
    if used:
        llrs, probe, _ = readout_frame_b(mu,v,obs.y_b,obs.training_rows,obs.n0)
        repair = (mu,v,llrs,probe)
    return dict(p9=base,v1=repair,used=bool(used),proposed=bool(proposed),score=float(pilot[0]-pilot[1]),
                pilot=pilot.tolist(),tokens=tokens.tolist())
