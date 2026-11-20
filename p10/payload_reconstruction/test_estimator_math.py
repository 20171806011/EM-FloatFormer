"""Independent numerical crosschecks of pilot integration, covariance score and observation isolation."""
from dataclasses import replace
import json
import numpy as np
from scipy.special import logsumexp, softmax
from p8.link import DELAYS, N, POWERS, T
from p8c.link import PILOTS
from p9.channel import simulate_link
from estimator import (ANCHOR, anchor_posterior, coarse_payload, estimate, exact_payload,
                       frequency_bank)

def test():
    obs, _ = simulate_link(981120010019, 9., 'full_reset', 'sparse')
    rows = (2,5,8,13,16)
    nodes,rot,lp=frequency_bank()
    y=obs.y_b[ANCHOR]
    n0=obs.n0
    m,second,logjoint=anchor_posterior(y,n0)
    coarse=coarse_payload(obs.y_b[list(rows)],rows,second,n0)
    idx=np.array([0,317,2500,len(nodes)-1])
    exact=exact_payload(obs.y_b[list(rows)],rows,idx,second,n0)
    err={'mean':0.,'second':0.,'pilot':0.,'coarse':0.,'exact':0.}
    F=np.fft.fft(np.eye(N),norm='ortho',axis=0)
    test_phases=np.sqrt(POWERS[0])*np.exp(2j*np.pi*np.arange(4096)/4096)
    for ix,k in enumerate(idx):
        params=nodes[k]
        d=np.stack([rot[p,ANCHOR]*np.roll(PILOTS[ANCHOR],int(lag))
                    for p,lag in zip(params,DELAYS)],axis=1)
        w=d[:,1:]
        K=n0*np.eye(N) + (w*POWERS[None,1:])@w.conj().T
        resid=y[:,None]-d[:,:1]*test_phases
        kr=np.linalg.solve(K,resid)
        ll=-(resid.conj()*kr).sum(0).real-np.linalg.slogdet(K)[1]-N*np.log(np.pi)
        weights=softmax(ll)
        weak=(w.conj().T@kr)*POWERS[1:,None]
        weakcov=np.linalg.inv(w.conj().T@w/n0+np.diag(1/POWERS[1:]))
        latent=np.vstack([test_phases,weak])
        independent_mean=latent@weights
        independent_second=(latent*weights)@latent.conj().T
        independent_second[1:,1:]+=weakcov
        full_lp=logjoint[k]-np.vdot(y,y).real/n0-N*np.log(np.pi)
        err['pilot']=max(err['pilot'],abs(full_lp-(logsumexp(ll)-np.log(len(ll))+lp[k])))
        err['mean']=max(err['mean'],np.max(abs(independent_mean-m[k])))
        err['second']=max(err['second'],np.max(abs(independent_second-second[k])))
        coarse_ref=0.; exact_ref=0.
        for row in rows:
            h=[rot[p,row,:,None]*np.roll(np.eye(N),int(lag),axis=0) for p,lag in zip(params,DELAYS)]
            cov=sum(independent_second[a,b]*(A@B.conj().T) for a,A in enumerate(h) for b,B in enumerate(h))
            cov+=n0*np.eye(N)
            z=obs.y_b[row]
            exact_ref-=np.linalg.slogdet(cov)[1]+np.vdot(z,np.linalg.solve(cov,z)).real
            diagonal=np.diag(F@cov@F.conj().T).real
            coarse_ref-=np.sum(np.log(diagonal)+abs(F@z)**2/diagonal)
        err['coarse']=max(err['coarse'],abs(coarse_ref-coarse[k]))
        err['exact']=max(err['exact'],abs(exact_ref-exact[ix]))
    assert max(err.values())<1e-9,err
    probe=obs.y_b.copy()
    forbidden=[i for i in range(T) if i not in (*rows,ANCHOR)]
    probe[forbidden]=np.nan
    def check(mode):
        base=estimate(obs,rows,mode)
        modified=estimate(replace(obs,y_a=np.full_like(obs.y_a,np.nan),y_b=probe),rows,mode)
        for a,b in zip(base,modified):
            np.testing.assert_array_equal(a,b)
        phase=np.exp(.42j)
        rotated=estimate(replace(obs,y_b=phase*probe),rows,mode)
        np.testing.assert_allclose(rotated[0],phase*base[0],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(rotated[1],base[1],rtol=1e-10,atol=1e-10)
    check(True)
    probe[list(rows)]=np.nan
    check(False)
    return dict(max_absolute_error=err,matrices_checked=20,anchor_nodes=len(idx),isolation=True,
                common_phase_equivariance=True)

if __name__=='__main__':
    print(json.dumps(test(),indent=2))
