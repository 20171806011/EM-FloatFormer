"""Compare the frozen 160-node restriction to exhaustive scoring on unrelated development links."""
import json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from p9.channel import simulate_link
from estimator import ANCHOR,anchor_posterior,estimate,exact_payload,frequency_bank

def run(n=8):
    nodes,rot,_=frequency_bank()
    rows=(2,5,8,13,16)
    result=[]
    for i in range(n):
        obs,_=simulate_link(981120010100+i,9,'full_reset','sparse')
        m,s,lp=anchor_posterior(obs.y_b[ANCHOR],obs.n0)
        pieces=np.array_split(np.arange(len(nodes)),(len(nodes)+95)//96)
        score=np.concatenate([exact_payload(obs.y_b[list(rows)],rows,ids,s,obs.n0) for ids in pieces])
        w=softmax(lp+score)
        exact=np.zeros((3,*rot.shape[1:]),complex)
        var=np.zeros_like(exact.real)
        for a in range(3):
            coeff=np.zeros(len(rot),complex)
            np.add.at(coeff,nodes[:,a],w*m[:,a])
            exact[a]=np.einsum('j,jtn->tn',coeff,rot)
            var[a]=np.maximum(w@s[:,a,a].real-abs(exact[a])**2,0.)
        approx,av=estimate(obs,rows)
        result.append(dict(seed=981120010100+i,
                      field_nmse=float(np.sum(abs(approx-exact)**2)/np.sum(abs(exact)**2)),
                      variance_mean_abs_error=float(np.mean(abs(var-av))),
                      exact_effective_sample_size=float(1/np.sum(w*w))))
    return result

if __name__=='__main__':
    records=run()
    print(json.dumps(records,indent=2))
    Path('quadrature_audit.json').write_text(json.dumps(records,indent=2)+'\n')
