"""Independent physics diagnostic: local curvature of the complex-Gaussian covariance score.

Evaluation only. The operating channel, amplitudes, slopes and anchor phases
are supplied by the simulator. Perturb each ray's frequency at the anchor by
u_l * delta, keeping its anchor phase fixed. Directions have unit Euclidean norm.
"""
import json
import numpy as np
from p8.link import clock
from p9.channel import simulate_link, build_sc_channel_matrices
from p9.contract import T, N

ROWS=(2,5,8,13,16)
SAMPLES=48
SEED_START=981120000100


def local_curvature(obs, gain):
    time = clock(T,start_row=T)
    tau = time-time[obs.training_rows[0]].mean()
    h = build_sc_channel_matrices(gain,ROWS)
    cov = h @ h.conj().swapaxes(-1,-2) + obs.n0*np.eye(N)
    directions = np.array([[1.,1.,1.],[1.,-1.,0.],[1.,1.,-2.]])
    directions /= np.linalg.norm(directions,axis=1)[:,None]
    curvature = []
    for u in directions:
        d_gain = 2j*np.pi*tau*gain*u[:,None,None]
        dh = build_sc_channel_matrices(d_gain,ROWS)
        half = dh @ h.conj().swapaxes(-1,-2)
        dcov = half+half.conj().swapaxes(-1,-2)
        if len(curvature)==0:
            clock_diff = (np.arange(N)[:,None]-np.arange(N)[None,:])/N
            closed = (2j*np.pi/np.sqrt(3))*clock_diff*cov
            np.testing.assert_allclose(dcov,closed,rtol=1e-10,atol=1e-12)
        z = np.linalg.solve(cov,dcov)
        curvature.append(float(np.einsum('bij,bji->',z,z).real))
    return curvature


def main():
    curv = []
    for i in range(SAMPLES):
        obs,truth = simulate_link(SEED_START+i,9.,'full_reset','sparse')
        curv.append(local_curvature(obs,truth.h_b_true))
    curv = np.array(curv)
    ratios=curv[:,1:].mean(1)/curv[:,0]
    print(json.dumps(dict(n=SAMPLES,seed_start=SEED_START,rows=ROWS,
                         median_direction_curvatures=np.median(curv,axis=0).tolist(),
                         relative_to_common_curvature_quartiles=np.quantile(ratios,[.25,.5,.75]).tolist()),
                     indent=2))


if __name__=='__main__':
    main()
