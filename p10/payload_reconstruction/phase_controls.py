"""Independent reset-only mechanistic controls; test seeds disjoint from main study."""
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from p9.channel import simulate_link,readout_frame_b
from strong16 import AUX1,AUX2,TARGETS
from strong_eval import strong_errors
from phase_check import metrics
from estimator import estimate


def score_variant(obs,truth,gains,variances,targets):
    llrs,_,_=readout_frame_b(gains,variances,obs.y_b,obs.training_rows,obs.n0)
    wrong=strong_errors(obs,truth,gains,variances,llrs,targets)
    phase=metrics(gains,truth.h_b_true,rows=targets)
    return {'wrong':sum(wrong),'n':len(wrong),'metrics':phase}


def run():
    records=[]
    for i in range(32):
        seed=981120020000+i
        obs,truth=simulate_link(seed,9.,'full_reset','sparse')
        for fold,(aux,targets) in enumerate(zip((AUX1,AUX2),TARGETS)):
            rows=list(aux)
            # A fixed nonidentity rotation keeps the complete five waveforms intact,
            # and changes only their association with actual sample times.
            y=obs.y_b.copy()
            y[rows]=np.roll(obs.y_b[rows],2,axis=0)
            scrambled=replace(obs,y_b=y)
            m,v=estimate(obs,aux)
            if i < 2:
                y=obs.y_b.copy()
                angles=np.array([.23,2.,-1.1,.89,2.58])
                y[rows]*=np.exp(1j*angles[:,None])
                m2,v2=estimate(replace(obs,y_b=y),aux)
                np.testing.assert_allclose(m,m2,atol=2e-10,rtol=2e-10)
                np.testing.assert_allclose(v,v2,atol=2e-10,rtol=2e-10)
            shuffled=estimate(scrambled,aux)
            anchor=estimate(obs,aux,use_payload=False)
            record=dict(seed=seed,fold=fold)
            for name,(gm,gv) in dict(actual=(m,v),time_shuffle=shuffled,anchor=anchor).items():
                record[name]=score_variant(obs,truth,gm,gv,targets)
            records.append(record)
    stats={}
    for arm in ('actual','time_shuffle','anchor'):
        models=[r[arm] for r in records]
        stats[arm]=dict(wrong=sum(m['wrong'] for m in models),blocks=sum(m['n'] for m in models))
        for metric in ('rel_phase','rel_freq','abs_phase','abs_freq'):
            sse=sum(m['metrics'][metric+'_sse'] for m in models)
            den=sum(m['metrics'][metric+'_weight'] for m in models)
            stats[arm][metric+'_rmse']=(sse/den)**.5
    return dict(summary=stats,records=records,row_phase_invariance_folds_checked=4)

if __name__=='__main__':
    data=run()
    Path('phase_controls.json').write_text(json.dumps(data,indent=1,allow_nan=False)+'\n')
    print(json.dumps(data['summary'],indent=1))
