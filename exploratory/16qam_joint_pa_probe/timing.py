from pathlib import Path
import sys,json,statistics,time,hashlib
import torch
sys.path.insert(0,str(Path(__file__).parent))
import probe
root=Path(__file__).parent
torch.set_num_threads(4)
obj=torch.load(root/'formal_results/mixed_train_59101.pt',weights_only=True,map_location='cpu')
model=probe.TinyJoint().eval();model.load_state_dict(obj['state_dict'])
results=[]
for batch in (1,64):
    base=probe.draw(batch,probe.PROFILES['equal4'],10,torch.Generator().manual_seed(10000+batch))
    obs=probe.receive(base,'strong')
    yp,yd,bits,nv=obs
    cases={'neural_joint':lambda: (model(yp,yd,nv)>0).long(),
           'dft8_fixed':lambda: probe.baseline_bits(obs),
           'dft8_calibrated':lambda: probe.baseline_bits(obs,.53),
           'threshold_fixed':lambda: probe.baseline_bits(obs,method='thresh')}
    with torch.inference_mode():
        for name,fn in cases.items():
            for _ in range(25):fn()
            a=[]
            for _ in range(350):
                t=time.perf_counter_ns();fn();a.append((time.perf_counter_ns()-t)/1e6)
            a.sort()
            results.append(dict(method=name,batch=batch,median_ms=statistics.median(a),p95_ms=a[int(.95*len(a))]))
(root/'formal_results/timing.json').write_text(json.dumps(results,indent=2))
for r in results:print(r)
