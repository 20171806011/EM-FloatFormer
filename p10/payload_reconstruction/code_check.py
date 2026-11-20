"""Phase-sensitive, list-truncated predictive score on unknown extra validation words."""
import numpy as np
from scipy.special import logsumexp

from p8.coding import qam
from p9.channel import build_sc_channel_matrix, effective_row_noise
from p9.contract import DELAYS, N
from p9.decoder import decode_strong_sc


def list_scores(obs, default, rebuilt, rows=(0,9)):
    """Use only the two validation words. Both channel models score the same list."""
    if tuple(rows)!=(0,9) or tuple(obs.training_rows)!=(10,):
        raise ValueError('This validation rule is defined for the sparse layout.')
    models = (default, rebuilt)
    total = np.zeros(2)
    indices = (np.arange(N)[None,:] - np.asarray(DELAYS)[:,None]) % N
    for row in rows:
        y = obs.y_b[row]
        channel = [build_sc_channel_matrix(m[:,row]) for m,_,_,_ in models]
        words = []
        for h,(_,v,llrs,_) in zip(channel,models):
            noise = effective_row_noise(v,obs.n0,rows=(row,))[0]
            decoded = decode_strong_sc(y,h,noise,list_size='large',bp_iters=20,turbo_pic=True,
                                       pic_scale=.75,initial_llrs=llrs[row].reshape(128))
            words.append(decoded.codeword)
        x = qam(np.unique(words,axis=0)).reshape(-1,N)
        # Marginalize independent path gain errors at each sample; off-diagonal
        # uncertainty and the unlisted codewords are omitted in both models.
        for k,(h,(_,v,_,_)) in enumerate(zip(channel,models)):
            noise = obs.n0 + (v[:,row]*abs(x[:,indices])**2).sum(axis=1)
            residual = y[None,:] - x @ h.T
            logp = -(abs(residual)**2/noise+np.log(noise)).sum(-1)
            total[k] += logsumexp(logp)
    return tuple(map(float,total))


if __name__=='__main__':
    from p9.channel import simulate_link
    from p9.gating import build_candidates
    obs,_ = simulate_link(981120000003,9.,'full_reset','sparse')
    left,right = [(c.mean_ray,c.var_ray,c.llrs,c.probe_pred) for c in build_candidates(obs)]
    a,b = list_scores(obs,left,right)
    b2,a2 = list_scores(obs,right,left)
    np.testing.assert_allclose([a,b],[a2,b2])
    a,b = list_scores(obs,left,left)
    np.testing.assert_allclose(a,b)
    print('Identical arms and simultaneous candidate permutations passed.')
