#!/usr/bin/env python3
"""Independent second evaluation using saved checkpoints, no retraining."""
import argparse, csv, os, json
import torch
from p1c_architecture import (ARCHES, ENCODINGS, model_factory, eval_seed)

p=argparse.ArgumentParser()
p.add_argument('--dir',default='p1c_results')
p.add_argument('--seed',type=int,default=101)
p.add_argument('--eval-frames',type=int,default=2000)
args=p.parse_args()
torch.set_num_threads(4)
models={}
for arch in ARCHES:
    for enc in ENCODINGS:
        key=f'{arch}__{enc}'
        path=os.path.join(args.dir,f'model_{key}_seed{args.seed}.pt')
        obj=torch.load(path,map_location='cpu',weights_only=True)
        m=model_factory(arch,enc,obj['nsc'])
        m.load_state_dict(obj['state_dict'],strict=True)
        models[key]=m
rows,_=eval_seed(models,args.seed,32,args.eval_frames)
expect={(r['seed'],r['split'],int(r['snr_db']),r['method']):r for r in csv.DictReader(open(os.path.join(args.dir,'ber_by_seed.csv')))}
wrong=[]
for r in rows:
    k=(str(r['seed']),r['split'],int(r['snr_db']),r['method'])
    if int(expect[k]['errors'])!=r['errors'] or int(expect[k]['bits'])!=r['bits']:
        wrong.append((k,int(expect[k]['errors']),r['errors']))
status=dict(seed=args.seed,eval_frames=args.eval_frames,compared_cells=len(rows),mismatches=wrong,
            checkpoint_loads=len(models),verified=(len(wrong)==0 and len(rows)==130))
with open(os.path.join(args.dir,'checkpoint_verification.json'),'w') as f:json.dump(status,f,indent=2)
print(json.dumps(status,indent=2))
assert not wrong
