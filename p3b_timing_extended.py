#!/usr/bin/env python3
"""CPU-only timing for complete synthesized physical-receiver compute paths.
These measure neither radio over-the-air latency nor ADC/SDR I/O.
"""
import time, statistics, math, csv
from pathlib import Path
import torch
from p3b_time_link import draw_base, link, PROFILES
from p3b_train_eval import load_d_for_seed, load_models
from p3b_receivers import logits_from_channel
from p3_a import dft_threshold

def measure(fn,reps=180,warmup=30):
    with torch.inference_mode():
        for _ in range(warmup):fn()
        vals=[]
        for _ in range(reps):
            s=time.perf_counter_ns(); fn(); vals.append((time.perf_counter_ns()-s)*1e-6)
    vals.sort();return statistics.median(vals),vals[math.ceil(.95*reps)-1]

def main():
    torch.set_num_threads(4)
    base_path=Path(__file__).parent
    seed=48101
    models=load_models(seed,base_path/'p3b_checkpoints')
    frozen=load_d_for_seed(seed,base_path/'p3a2_checkpoints')
    r=[]
    for n in (1,64):
        g=torch.Generator().manual_seed(731100+n)
        base=draw_base(n,PROFILES['equal4'],10,g)
        received=link(base,'clean')
        h=frozen.channel(received.yp,received.yd,received.noise_var)
        def pure_link():return link(base,'clean')
        def frozen_channel():return frozen.channel(received.yp,received.yd,received.noise_var)
        def full_both():
            obs=link(base,'clean')
            hh=frozen.channel(obs.yp,obs.yd,obs.noise_var)
            return models['mixed__raw'](obs.yd,hh,obs.noise_var)
        def full_postlink():
            hh=frozen.channel(received.yp,received.yd,received.noise_var)
            return models['mixed__raw'](received.yd,hh,received.noise_var)
        def threshold_path():
            h=dft_threshold(received.yp,received.noise_var,'hard')
            return logits_from_channel(received.yd,h,received.noise_var,align=False)
        def threshold_total():
            obs=link(base,'clean')
            hh=dft_threshold(obs.yp,obs.noise_var,'hard')
            return logits_from_channel(obs.yd,hh,obs.noise_var,align=False)
        task={'simulated_link_only':pure_link,'frozen_backbone_only':frozen_channel,
              'mixed_raw_head_only':lambda:models['mixed__raw'](received.yd,h,received.noise_var),
              'mixed_raw_backbone_plus_head':full_postlink,
              'neural_full_simulated_pipeline':full_both,
              'threshold_no_anchor_postlink':threshold_path,
              'threshold_no_anchor_full_simulated_pipeline':threshold_total}
        for k,f in task.items():
            m,p95=measure(f)
            r.append(dict(batch_frames=n,task=k,median_ms=m,p95_ms=p95,repeat=180,
                notes='PyTorch CPU 4 threads; no real radio, ADC I/O, device transport, FEC, protocol'))
        print('TIMED batch',n,flush=True)
    with (base_path/'p3b_results'/'latency_fullpath_cpu.csv').open('w',newline='') as fh:
        writer=csv.DictWriter(fh,fieldnames=list(r[0]));writer.writeheader();writer.writerows(r)

if __name__=='__main__':main()
