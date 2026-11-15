"""Independent audit for the non-Sionna TDL-table surrogate experiment.
Does not train or select models. Counts as same-runtime repetition, not hardware validation.
"""
import csv, hashlib, json, math, statistics
from pathlib import Path
import numpy as np
import torch
from p3de_profile_bridge import N,TEST_SCENARIOS,SCENARIOS,TDL,profile_data,covariance,draw,audit_math
from p3de_experiment import SEEDS,PRIMARY,load_models,eval_seed_scenario,STEPS,BATCH,FRAMES,MODELS


def verify(out='p3de_results'):
    out=Path(out)
    assert audit_math()
    with (out/'ber_by_seed.csv').open() as f:
        rows=list(csv.DictReader(f))
    lookup={(int(r['seed']),r['scenario'],int(r['snr_db']),r['method']):(int(r['errors']),int(r['bits'])) for r in rows}
    checked=0
    # every fresh seed on both preselected endpoints, 9 estimators per evaluation
    for seed in SEEDS:
        model=load_models(seed,out,STEPS,BATCH)
        for scene in PRIMARY:
            replay=eval_seed_scenario(seed,scene,10,model,FRAMES)
            for r in replay:
                key=(seed,scene,10,r['method'])
                assert lookup[key]==(r['errors'],r['bits']),key
                checked+=1
        # Exact designed common-phase equivariance (predictive logits) in both model families
        generator=torch.Generator().manual_seed(438100000+seed)
        frame=draw(20,TEST_SCENARIOS[0],10,generator)
        ph=torch.tensor([.12,.76,1.43,2.87])
        for name,m in model.items():
            m.eval()
            with torch.inference_mode():
                ref=m.channel(frame.yp,frame.yd,frame.noise_var)
                raw=m(frame.yp,frame.yd,frame.noise_var)
                for phi in ph:
                    angle=torch.polar(torch.ones_like(frame.yp.real),torch.full_like(frame.yp.real,float(phi)))
                    rotated=m.channel(frame.yp*angle,frame.yd*angle,frame.noise_var)
                    logits=m(frame.yp*angle,frame.yd*angle,frame.noise_var)
                    assert float((rotated-ref*angle).abs().max())<5e-6
                    assert float((logits-raw).abs().max())<1e-5
    # Independent double-precision NumPy closed-form channel covariance implementation
    max_cov_error=0.0
    for name in SCENARIOS:
        model,delay,powers=profile_data(name)
        f=(np.arange(N)-N//2)*30000.0
        rows_f=np.exp(-2j*np.pi*f[:,None]*delay.numpy()[None,:]);pp=powers.numpy()
        independent=(rows_f*pp[None,:]) @ rows_f.conj().T
        compare=covariance(name).numpy().astype(np.complex128)
        max_cov_error=max(max_cov_error,float(np.max(np.abs(independent-compare))))
    assert max_cov_error<3e-6,max_cov_error
    # Independent observed complex Gaussian noise calibration, not using training/test RNGs
    g=torch.Generator().manual_seed(776544)
    frame=draw(8192,'train_A_500ns',10,g)
    n_obs=float((frame.yp-frame.h).abs().square().mean())
    assert abs(n_obs-.1)<.007,n_obs
    out_rows=[]
    for fp in sorted((out/'checkpoints').glob('*.pt')):
        out_rows.append({'file':fp.name,'sha256':hashlib.sha256(fp.read_bytes()).hexdigest()})
    assert len(out_rows)==20
    saved=json.loads((out/'checkpoint_sha256.json').read_text())
    assert sorted((x['filename'],x['sha256']) for x in saved)==sorted((x['file'],x['sha256']) for x in out_rows)
    result={'status':'PASS','source':'independent NumPy covariance + same-code seeded weight replay',
            'per_seed_endpoint_integer_replay_cells':checked,
            'trained_model_phase_cases':len(SEEDS)*len(MODELS)*len(ph),
            'max_numpy_covariance_abs_error':max_cov_error,
            'measured_pilot_noise_var_at_10db':n_obs,
            'training_checkpoints_sha_verified':len(out_rows),
            'sionna_2_2_executed':False}
    (out/'standalone_audit.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    torch.set_num_threads(4)
    print(json.dumps(verify(),indent=2))
