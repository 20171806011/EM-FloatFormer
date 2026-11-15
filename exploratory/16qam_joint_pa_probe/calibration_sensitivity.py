"""Post-hoc sensitivity check only: match neural and classic calibration frame budgets.
No model weights are trained or changed. Uses training RNG, no heldout labels.
"""
from pathlib import Path
import sys,json,csv
import torch
sys.path.insert(0,str(Path(__file__).parent))
import probe
root=Path(__file__).parent
r=root/'formal_results'
torch.set_num_threads(4)
thresholds=[]
rows=[]
for seed in probe.SEEDS:
    g=torch.Generator().manual_seed(seed*19873+41)
    vals=[];labels=[]
    with torch.no_grad():
        for step in range(300):
            base=probe.draw(64,probe.TRAIN,10,g)
            yp,yd,bits,nv=probe.receive(base,probe.CONDITIONS[step%3])
            z=probe.equalize(yd,probe.delay_est(yp),nv)
            vals.append(torch.cat((z.real.abs().flatten(),z.imag.abs().flatten())))
            labels.append(torch.cat((bits[...,1].flatten().bool(),bits[...,3].flatten().bool())))
    v=torch.cat(vals);l=torch.cat(labels)
    grid=torch.linspace(.25,1.15,91)
    errors=[int(((v<t)!=l).sum()) for t in grid]
    best=int(torch.argmin(torch.tensor(errors)))
    t=float(grid[best]); thresholds.append(dict(seed=seed,threshold=t,train_frames=300*64,train_magnitude_bits=l.numel(),calibration_errors=errors[best]))
    for pname,p in probe.PROFILES.items():
        for cond in probe.CONDITIONS:
            s=990000000+seed*1000+list(probe.PROFILES).index(pname)*100+list(probe.CONDITIONS).index(cond)
            gtest=torch.Generator().manual_seed(s)
            er=0;n=0
            for off in range(0,768,128):
                base=probe.draw(min(128,768-off),p,10,gtest)
                obs=probe.receive(base,cond)
                bpred=probe.baseline_bits(obs,t,'dft8')
                er+=int((bpred!=obs[2]).sum());n+=obs[2].numel()
            rows.append(dict(seed=seed,profile=pname,condition=cond,method='dft8_equal_budget_calibrated',errors=er,bits=n,ber=er/n,threshold=t))
with (r/'equal_budget_calibration.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(r/'equal_budget_calibration_config.json').write_text(json.dumps({'posthoc':True,'description':'Training-sample-count-matched classical calibration sensitivity analysis, 19200 frames per seed. Not a protocol-confirmed primary endpoint.','thresholds':thresholds},indent=2))
for p in probe.PROFILES:
    for cond in probe.CONDITIONS:
        x=[a for a in rows if a['profile']==p and a['condition']==cond]
        print(p,cond,'threshold BER',100*sum(y['errors'] for y in x)/sum(y['bits'] for y in x))
print('calibration thresholds',thresholds)
