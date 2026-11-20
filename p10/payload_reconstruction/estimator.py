"""B-only three-tap reconstruction from one known pilot and unknown payload second moments.

Frequency quadrature uses the public P9 reset prior.  The pilot gives an analytic
posterior for gains at its center.  Payload uses a Gaussian covariance score on
independent codeword rows; sharing uncertain gains across rows makes this a
composite likelihood.  Per-row Fourier diagonals screen the candidates, followed
by full time-varying waveform covariance scoring on the selected nodes.
"""
from functools import lru_cache
import numpy as np
from scipy.special import i0e, i1e, softmax
from p8.link import DELAYS, LINK_DURATION, N, POWERS, T, clock
from p8c.link import PILOTS

ANCHOR = 10
ORDER = 3
FINAL_NODES = 160
ABS_TIME = clock(T, start_row=T)
ORIGIN = ABS_TIME[ANCHOR].mean()
U = ABS_TIME - ORIGIN
IND = np.arange(N)
PATHS = np.arange(3)
# Input symbol m contributes to y at (m + delay) modulo N.
OUT = (IND[None,:] + DELAYS[:,None]) % N


@lru_cache(maxsize=1)
def frequency_bank():
    g, w = np.polynomial.legendre.leggauss(ORDER)
    values = .0625 + .0175*g
    v, e, sign = np.meshgrid(values, values, (-1.,1.), indexing='ij')
    start, end = (v*sign).ravel(), (e*sign).ravel()
    wa, wb, _ = np.meshgrid(w/2,w/2,(-1.,1.),indexing='ij')
    prior = (wa*wb*.5).ravel()
    slope = (end-start)/LINK_DURATION
    local = start+slope*ORIGIN
    rot = np.exp(2j*np.pi*(local[:,None,None]*U + .5*slope[:,None,None]*U**2))
    nodes = np.stack(np.meshgrid(*(np.arange(len(local)),)*3,indexing='ij'),-1).reshape(-1,3)
    return nodes, rot, np.log(prior[nodes]).sum(axis=1)


@lru_cache(maxsize=4)
def pilot_model(n0):
    nodes,rot,_ = frequency_bank()
    d = np.array([np.roll(PILOTS[ANCHOR],int(tau)) for tau in DELAYS])
    dictionary = rot[:,ANCHOR,None,:] * d[None,:,:]
    gram = np.empty((len(nodes),3,3),complex)
    for a in PATHS:
        for b in PATHS:
            cross = dictionary[:,a].conj() @ dictionary[:,b].T
            gram[:,a,b] = cross[nodes[:,a],nodes[:,b]]
    weak = gram[:,1:,1:] + np.diag(n0/POWERS[1:])
    inv = np.linalg.inv(weak)
    b = gram[:,1:,0]
    c = (inv @ b[:,:,None])[...,0]
    dom_energy = (gram[:,0,0] - (b.conj()*c).sum(1)).real/n0
    logdet = np.linalg.slogdet(weak)[1] + np.log(POWERS[1:]).sum() + (N-2)*np.log(n0)
    return dictionary, inv, c, dom_energy, logdet


def anchor_posterior(y,n0):
    """Integrate unknown dominant phase and two complex Gaussian weak gains."""
    nodes,_,logprior = frequency_bank()
    dictionary,inv,c,dom_energy,logdet = pilot_model(n0)
    per_node = dictionary.conj() @ y
    rhs = per_node[nodes,PATHS]
    w = (inv @ rhs[:,1:,None])[...,0]
    matched = (rhs[:,0]-(c.conj()*rhs[:,1:]).sum(1))/n0
    radius = 2*np.sqrt(POWERS[0])*abs(matched)
    phase = np.exp(1j*np.angle(matched))
    m0 = np.sqrt(POWERS[0])*phase*i1e(radius)/i0e(radius)
    var0 = np.maximum(0.,POWERS[0]-abs(m0)**2)
    mean = np.column_stack([m0,w-c*m0[:,None]])
    # Conditional on a0, weak mean is w-c*a0 and weak covariance is n0*inv.
    direction = np.column_stack([np.ones(len(nodes)),-c])
    covariance = var0[:,None,None]*direction[:,:,None]*direction[:,None,:].conj()
    covariance[:,1:,1:] += n0*inv
    second = mean[:,:,None]*mean[:,None,:].conj() + covariance
    # Discard the common -N log pi - ||y||^2/n0.
    pilot = (logprior - logdet + (rhs[:,1:].conj()*w).sum(1).real/n0
             - POWERS[0]*dom_energy + radius + np.log(i0e(radius)))
    return mean,second,pilot


@lru_cache(maxsize=8)
def frequency_geometry(rows):
    _,rot,_ = frequency_bank()
    pair_gram = {}
    for a in PATHS:
        for b in range(a):
            ra = rot[:,list(rows)][:,:,OUT[a]]
            rb = rot[:,list(rows)][:,:,OUT[b]]
            # Exact FFT-diagonal projection of the full time-varying channel covariance.
            pair_gram[a,b] = np.einsum('hrm,krm->hkr',ra,rb.conj())/N
    return pair_gram


def coarse_payload(y,rows,second,n0):
    nodes,_,_ = frequency_bank()
    geom = frequency_geometry(rows)
    # All path delays are even, giving identical model powers at k and k+N/2.
    energy = abs(np.fft.fft(y,axis=-1,norm='ortho'))**2
    multiplicity = 2
    half = N//multiplicity
    energy = energy[:,:half]+energy[:,half:]
    powers = np.empty((len(nodes),len(rows),half))
    powers[:] = (n0+np.trace(second,axis1=1,axis2=2).real)[:,None,None]
    for (a,b),g in geom.items():
        coef = second[:,a,b,None]*g[nodes[:,a],nodes[:,b]]
        ramp = np.exp(-2j*np.pi*(DELAYS[a]-DELAYS[b])*np.arange(half)/N)
        powers += 2*(coef[:,:,None]*ramp).real
    powers = np.maximum(powers,n0)
    return -(multiplicity*np.log(powers)+energy/powers).sum(axis=(1,2))


def exact_payload(y,rows,idx,second,n0):
    """Use all covariance entries, including cyclic lag-2 and lag-4 cross terms."""
    nodes,rot,_=frequency_bank()
    selected = nodes[idx]
    gains_rot = [rot[selected[:,a]][:,list(rows)][:,:,OUT[a]] for a in PATHS]
    cov = np.zeros((len(idx),len(rows),2,N//2,N//2),complex)
    for a in PATHS:
        for b in PATHS:
            cross = second[idx,a,b,None,None]*gains_rot[a]*gains_rot[b].conj()
            for parity in (0,1):
                cov[:,:,parity,OUT[a,parity::2]//2,OUT[b,parity::2]//2] += cross[:,:,parity::2]
    n=np.arange(N//2)
    cov[...,n,n] += n0
    z=np.stack([y[:,::2],y[:,1::2]],axis=1)
    z=np.broadcast_to(z,cov.shape[:-1])
    q=np.linalg.solve(cov,z[...,None])[...,0]
    score=-(q*z.conj()).sum(axis=-1).real-np.linalg.slogdet(cov)[1]
    return score.sum(axis=(1,2))


def estimate(obs,aux,use_payload=True):
    """Return posterior means/variances shaped [3,T,N]; no A or target data are read."""
    if tuple(obs.training_rows)!=(ANCHOR,) or not (obs.n0>0):
        raise ValueError('Requires sparse P9 with positive noise variance')
    aux=tuple(int(r) for r in aux)
    if len(set(aux))!=5 or len(aux)!=5 or any(r<0 or r>=T or r in (ANCHOR,obs.probe_row) for r in aux):
        raise ValueError('Expected five distinct available non-pilot payload rows')
    nodes, rot, _ = frequency_bank()
    mean,second,logpilot = anchor_posterior(obs.y_b[ANCHOR],float(obs.n0))
    if use_payload:
        y=obs.y_b[list(aux)]
        coarse=coarse_payload(y,aux,second,obs.n0)
        proposal=softmax(logpilot + .75*coarse)
        # Deterministic, stratified compression of the tempered quadrature distribution.
        cdf=np.cumsum(proposal)
        cdf[-1]=1.
        idx,counts=np.unique(np.searchsorted(cdf,(np.arange(FINAL_NODES)+.5)/FINAL_NODES),
                             return_counts=True)
        exact=exact_payload(y,aux,idx,second,obs.n0)
        weights=softmax(np.log(counts) + exact - .75*coarse[idx])
    else:
        idx=np.arange(len(nodes))
        weights=softmax(logpilot)
    out=np.empty((3,T,N),complex)
    var=np.empty((3,T,N),float)
    for a in PATHS:
        coeff=np.zeros(len(rot),complex)
        np.add.at(coeff,nodes[idx,a],weights*mean[idx,a])
        out[a]=np.einsum('j,jtn->tn',coeff,rot)
        var[a]=np.maximum(0.,weights @ second[idx,a,a].real - abs(out[a])**2)
    return out,var
