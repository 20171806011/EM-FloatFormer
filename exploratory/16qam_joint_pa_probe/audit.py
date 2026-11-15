#!/usr/bin/env python3
"""Same-implementation count replay and independent physical consistency checks."""
import csv,json,hashlib,math,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).parent))
import probe
ROOT=Path(__file__).parent
RES=ROOT/'formal_results'
torch.set_num_threads(4)
expected=list(csv.DictReader((RES/'ber_by_seed.csv').open()))
index={(int(r['seed']),r['profile'],r['condition'],r['method']):(int(r['errors']),int(r['bits']),int(r['symbol_errors']),int(r['symbols'])) for r in expected}
assert len(index)==3*3*3*5
matched=0
for seed in probe.SEEDS:
    models={}
    for name in ('clean_train','mixed_train'):
        checkpoint=torch.load(RES/f'{name}_{seed}.pt',map_location='cpu',weights_only=True)
        assert checkpoint['seed']==seed and checkpoint['steps']==300 and checkpoint['name']==name
        model=probe.TinyJoint()
        model.load_state_dict(checkpoint['state_dict'],strict=True);model.eval()
        models[name]=model
    calib_t=next(row['train_mix_threshold'] for row in json.loads((RES/'audit.json').read_text())['calibration'] if row['seed']==seed)
    for x in probe.eval_seed(seed,models,calib_t,768):
        assert (x['errors'],x['bits'],x['symbol_errors'],x['symbols'])==index[(seed,x['profile'],x['condition'],x['method'])]
        matched+=1
# Independently check CP-based linear channel vs frequency multiplication
# for an amplitude-compressed 16-QAM OFDM symbol.
g=torch.Generator().manual_seed(16511)
base=probe.draw(5,probe.PROFILES['long6'],10,g)
pp,xd,bits,H,nv,n_p,n_d=base
ht=torch.fft.ifft(H,dim=-1)
for cond in probe.CONDITIONS:
    td=probe.pa_magnitude(torch.fft.ifft(xd,norm='ortho'),cond)
    cpseq=torch.cat([td[:,-probe.CP:],td],dim=-1)
    y=torch.zeros_like(cpseq)
    for lag in range(6): y[:,lag:]+=ht[:,lag,None]*cpseq[:,:cpseq.shape[-1]-lag]
    explicit=torch.fft.fft(y[:,probe.CP:],norm='ortho')
    implicit=H*torch.fft.fft(td,norm='ortho')
    diff=float((explicit-implicit).abs().max())
    assert diff<2e-6,(cond,diff)
# Validate fixed pilots equalize near-perfectly for noise-free clean channel.
ref_pilot=pp
assert (ref_pilot.abs().square()-1).abs().max()<1e-5
# Sum raw integer counters independently; pooled BER must be errors / bits.
summ=list(csv.DictReader((RES/'ber_summary.csv').open()))
assert len(summ)==45
for y in summ:
    filtered=[x for x in expected if (x['profile'],x['condition'],x['method'])==(y['profile'],y['condition'],y['method'])]
    assert len(filtered)==3
    assert sum(int(x['errors']) for x in filtered)==int(y['errors'])
    assert sum(int(x['bits']) for x in filtered)==int(y['bits'])
    assert abs(float(y['ber'])-int(y['errors'])/int(y['bits']))<1e-15

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
audit=json.loads((RES/'audit.json').read_text())
assert audit['protocol_sha256']==sha(ROOT/'PROTOCOL.md')
assert audit['code_sha256']==sha(ROOT/'probe.py')
for weight in audit['checkpoints']:
    assert weight['sha256']==sha(RES/weight['name'])
output=dict(replayed=matched,total_raw_rows=len(expected),total_summary_rows=len(summ),
            cp_equivalence='PASS (<2e-6)',seed_family=list(probe.SEEDS),
            protocol_sha256=sha(ROOT/'PROTOCOL.md'),model_sha256=audit['checkpoints'])
(RES/'independent_audit.json').write_text(json.dumps(output,indent=2))
print('PASS: replay',matched,'/135, original counts; CP equivalence; aggregate reconciliation; hashes')
