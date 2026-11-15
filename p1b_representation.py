#!/usr/bin/env python3
"""P1-B: mathematical checks and *paired* representation ablation.

Synthetic, passive civilian QPSK/OFDM channel estimation only. No emissions,
radar, intentional interference, spectrum attacks, or hardware control.

Run:
 python p1b_representation.py --steps 750 --eval-frames 2000 --seeds 101 202 303 --out p1b_results

Fairness: Both receivers receive EXACTLY the same pilot and data observations,
noise variance, hand-built LS equalizer, positional features, initial weights,
training batches, optimization schedule, and test frames. Both have the SAME
number of trainable parameters. They differ only in two complex sample encoders:
  raw_iq_magnitude  : [real, imag, magnitude]
  logamp_phase      : [log1p(magnitude/sigma), real/(magnitude+eps), imag/(magnitude+eps)]
The third raw feature intentionally matches input dimensionality and supplies
redundant information without changing what is physically observed. Neither
encoding sees the true instantaneous channel. Classical baselines are fixed
to the TRAINING channel prior, including out-of-distribution evaluation.
"""
import argparse
import csv
import json
import math
import os
import time
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from ofdm_baseline import (make_batch, estimate_channel, covariance_matrix,
                           decisions_from_channel)

TRAIN_PROFILE = (0.72, 0.20, 0.08)
TEST_PROFILES = {
    'in_distribution': TRAIN_PROFILE,
    'unseen_four_tap_profile': (0.25, 0.25, 0.25, 0.25),
}


def iq_features(z, encoding, sigma=1.0, eps=1e-5):
    magnitude = z.abs()
    if encoding == 'raw_iq_magnitude':
        return torch.stack((z.real, z.imag, magnitude), dim=-1)
    if encoding == 'logamp_phase':
        return torch.stack((torch.log1p(magnitude / sigma),
                            z.real / (magnitude + eps),
                            z.imag / (magnitude + eps)), dim=-1)
    raise ValueError(encoding)


class RepresentationTransformer(nn.Module):
    def __init__(self, nsc=32, width=48, depth=2, heads=4, encoding='raw_iq_magnitude'):
        super().__init__()
        self.nsc = nsc
        self.encoding = encoding
        self.in_proj = nn.Linear(11, width)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=2*width,
            batch_first=True, norm_first=True, dropout=0., activation='gelu')
        self.net = nn.TransformerEncoder(encoder_layer, num_layers=depth,
                                         enable_nested_tensor=False)
        self.out_proj = nn.Linear(width, 2)
        self.register_buffer('omega', 2 * math.pi * torch.arange(nsc) / nsc)

    def forward(self, yp, yd, noise_var):
        n = yp.shape[-1]
        if n != self.nsc:
            raise ValueError('wrong OFDM subcarrier count')
        z = yd * yp.conj() / (yp.abs().square() + noise_var + .01)
        ang = self.omega[None, :].expand(yp.shape[0], -1)
        nv = noise_var.expand(-1, n)
        feature = torch.cat((
            iq_features(yp, self.encoding), iq_features(yd, self.encoding),
            torch.stack((z.real, z.imag, ang.cos(), ang.sin(), nv.log10()), dim=-1)
        ), dim=-1)
        return self.out_proj(self.net(self.in_proj(feature))) + \
               3.0*torch.stack((z.real,z.imag),dim=-1)


def float_roundtrip(out_file):
    """Round both encodings to a chosen storage dtype, then decode in float64.

    Values are computed in float64 before rounding, isolating *storage*
    quantization (not full low-precision layer computation or ADC quality).
    """
    rows=[]
    phase = torch.linspace(0, 2*math.pi, steps=37, dtype=torch.float64)[:-1]
    amplitudes = [0.0] + [10.0**p for p in range(-8,9)]
    for dtype_name, dtype in [('float32',torch.float32),('float16',torch.float16),
                               ('bfloat16',torch.bfloat16)]:
        for a in amplitudes:
            zre = a * phase.cos()
            zim = a * phase.sin()
            raw_re = zre.to(dtype).double()
            raw_im = zim.to(dtype).double()
            # stable functions log1p/expm1 avoid log(1+x) cancellation.
            loga = torch.log1p(torch.hypot(zre,zim)).to(dtype).double()
            ui = (zre/(a+1e-5)).to(dtype).double()
            uq = (zim/(a+1e-5)).to(dtype).double()
            recovered_r = torch.expm1(loga)
            rec_re=(recovered_r+1e-5)*ui
            rec_im=(recovered_r+1e-5)*uq
            for label, rr, ri in [('raw_iq',raw_re,raw_im),
                                  ('logamp_phase',rec_re,rec_im)]:
                error=torch.hypot(rr-zre,ri-zim)
                if a>0:
                    rel=error/a
                    valid=(torch.isfinite(rr)&torch.isfinite(ri)&torch.isfinite(rel))
                    frac_nonfinite = 1-valid.double().mean().item()
                    finite = rel[valid]
                    err_median = finite.median().item() if len(finite) else None
                    err_max = finite.max().item() if len(finite) else None
                    # near-zero round-to-zero is a separate failure mode
                    frac_zero = ((rr==0)&(ri==0)).double().mean().item()
                else:
                    frac_nonfinite=0.0
                    err_median=error.median().item()
                    err_max=error.max().item()
                    frac_zero=((rr==0)&(ri==0)).double().mean().item()
                rows.append(dict(dtype=dtype_name,amplitude=a,encoding=label,
                                 median_relative_error=err_median,
                                 max_relative_error=err_max,
                                 fraction_nonfinite=frac_nonfinite,
                                 fraction_zero=frac_zero))
    write_csv(out_file,rows)
    return rows


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or '.',exist_ok=True)
    with open(path,'w',newline='') as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        out.writeheader();out.writerows(rows)


def check_encoding():
    torch.manual_seed(87)
    z=torch.complex(torch.randn(500,dtype=torch.float64),
                    torch.randn(500,dtype=torch.float64))
    z=torch.cat([z,torch.zeros(1,dtype=torch.complex128)])
    mag=z.abs()
    f=iq_features(z,'logamp_phase')
    r=torch.expm1(f[...,0])
    reconstruction=torch.complex((r+1e-5)*f[...,1],(r+1e-5)*f[...,2])
    max_abs=(reconstruction-z).abs().max().item()
    assert max_abs < 1e-12, max_abs
    assert torch.isfinite(f).all()
    a=RepresentationTransformer(32,encoding='raw_iq_magnitude')
    b=RepresentationTransformer(32,encoding='logamp_phase')
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    return max_abs, sum(p.numel() for p in a.parameters())


def train_two_models(seed,steps,batch_size,nsc,device):
    torch.manual_seed(seed)
    raw=RepresentationTransformer(nsc, encoding='raw_iq_magnitude').to(device)
    logp=RepresentationTransformer(nsc, encoding='logamp_phase').to(device)
    # Identical starting weights, not merely same distribution.
    logp.load_state_dict(raw.state_dict())
    models={'raw_iq_magnitude':raw,'logamp_phase':logp}
    opt={name:torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=1e-4)
         for name,m in models.items()}
    for m in models.values(): m.train()
    start=time.time()
    last={}
    for step in range(1,steps+1):
        batch=make_batch(batch_size,nsc,0.,20.,device,TRAIN_PROFILE)
        for name,m in models.items():
            opt[name].zero_grad(set_to_none=True)
            logits=m(batch.yp,batch.yd,batch.noise_var)
            loss=F.binary_cross_entropy_with_logits(logits,batch.bits.float())
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(),1.)
            opt[name].step()
            last[name]=round(float(loss.detach().item()),5)
        if step==1 or step%250==0 or step==steps:
            print('seed',seed,'step',step,'/',steps,'loss',last,
                  'elapsed_s',round(time.time()-start,1), flush=True)
    return models, time.time()-start


def eval_models(models, nsc, device, eval_frames, seed):
    for m in models.values():m.eval()
    cov=covariance_matrix(nsc,TRAIN_PROFILE,device)
    all_rows=[]
    paired=[]
    with torch.inference_mode():
        for profile_name,profile in TEST_PROFILES.items():
            for snr in [0,5,10,15,20]:
                # Independent and reproducible test streams; same frames for
                # all receivers. Training RNG state is not relied upon.
                torch.manual_seed(seed*100000 + snr*100 + (0 if profile_name=='in_distribution' else 1))
                methods=list(models.keys())+['ls','dft_fixed_train_support','lmmse_fixed_train_prior','oracle']
                err={k:0 for k in methods}
                frame_differences=[]
                bit_total=0
                for start in range(0,eval_frames,128):
                    batch=make_batch(min(128,eval_frames-start),nsc,snr,snr,device,profile)
                    pred={}
                    for name,m in models.items():
                        pred[name]=(m(batch.yp,batch.yd,batch.noise_var)>0).long()
                    channel={
                        'ls':estimate_channel(batch.yp,batch.noise_var,'ls'),
                        'dft_fixed_train_support':estimate_channel(batch.yp,batch.noise_var,'dft',taps=len(TRAIN_PROFILE)),
                        'lmmse_fixed_train_prior':estimate_channel(batch.yp,batch.noise_var,'lmmse',covariance=cov),
                        'oracle':batch.h
                    }
                    for name,hhat in channel.items():
                        pred[name]=decisions_from_channel(batch.yd,hhat,batch.noise_var)
                    for name,p in pred.items():
                        err[name]+=int((p!=batch.bits).sum().item())
                    raw_err=(pred['raw_iq_magnitude']!=batch.bits).sum(dim=(1,2))
                    log_err=(pred['logamp_phase']!=batch.bits).sum(dim=(1,2))
                    frame_differences.extend(((raw_err-log_err)/float(2*nsc)).tolist())
                    bit_total+=batch.bits.numel()
                for name in methods:
                    all_rows.append(dict(seed=seed,split=profile_name,snr_db=snr,
                                         encoding_or_baseline=name,
                                         bit_errors=err[name],total_bits=bit_total,
                                         ber=err[name]/bit_total))
                d=torch.tensor(frame_differences,dtype=torch.float64)
                mean=d.mean().item()
                se=d.std(unbiased=True).item()/math.sqrt(d.numel())
                paired.append(dict(seed=seed,split=profile_name,snr_db=snr,
                    raw_minus_log_ber=mean, frame_cluster_se=se,
                    ci95_low=mean-1.96*se,ci95_high=mean+1.96*se,
                    frames=int(d.numel()),bits=bit_total))
            print('eval seed',seed,'split',profile_name,'done',flush=True)
    return all_rows,paired


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps',type=int,default=750)
    parser.add_argument('--batch-size',type=int,default=64)
    parser.add_argument('--eval-frames',type=int,default=2000)
    parser.add_argument('--seeds',type=int,nargs='+',default=[101,202,303])
    parser.add_argument('--nsc',type=int,default=32)
    parser.add_argument('--out',default='p1b_results')
    parser.add_argument('--math-only',action='store_true')
    args=parser.parse_args()
    if args.steps<=0 or args.eval_frames<=1 or args.nsc<8:
        parser.error('steps>0, eval-frames>1, nsc>=8 required')
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    os.makedirs(args.out,exist_ok=True)
    max_error,n_params=check_encoding()
    roundtrip=float_roundtrip(os.path.join(args.out,'float_roundtrip.csv'))
    print(f'MATH CHECK PASSED: max float64 inverse residual={max_error:.3e}; '
          f'equal model parameter counts={n_params}',flush=True)
    print('float16 at amplitude 100000000 (raw vs logamp):',
          [(r['encoding'],r['fraction_nonfinite'],r['median_relative_error'])
            for r in roundtrip if r['dtype']=='float16' and r['amplitude']==1e8], flush=True)
    if args.math_only: return
    allrows=[]; allpaired=[]; timing=[]
    for seed in args.seeds:
        models,t=train_two_models(seed,args.steps,args.batch_size,args.nsc,'cpu')
        for name, m in models.items():
            torch.save({'state_dict': m.cpu().state_dict(), 'encoding': name,
                        'seed': seed, 'nsc': args.nsc, 'steps': args.steps,
                        'train_profile': TRAIN_PROFILE},
                       os.path.join(args.out, f'model_seed{seed}_{name}.pt'))
        rows,paired=eval_models(models,args.nsc,'cpu',args.eval_frames,seed)
        allrows.extend(rows); allpaired.extend(paired)
        timing.append(dict(seed=seed,train_seconds=t,params=n_params))
        write_csv(os.path.join(args.out,'ber_by_seed.csv'),allrows)
        write_csv(os.path.join(args.out,'paired_effects_by_seed.csv'),allpaired)
    write_csv(os.path.join(args.out,'timing.csv'),timing)
    summary=[]
    grouped=defaultdict(list)
    for row in allrows:
        grouped[(row['split'],row['snr_db'],row['encoding_or_baseline'])].append(row)
    for (split,snr,method),items in sorted(grouped.items()):
        errors=sum(r['bit_errors'] for r in items)
        bits=sum(r['total_bits'] for r in items)
        vals=torch.tensor([r['ber'] for r in items],dtype=torch.float64)
        summary.append(dict(split=split,snr_db=snr,method=method,
                            pooled_ber=errors/bits, pooled_errors=errors,
                            total_bits=bits,per_seed_mean_ber=vals.mean().item(),
                            per_seed_sd=vals.std(unbiased=True).item()
                              if len(vals)>1 else 0.))
    write_csv(os.path.join(args.out,'ber_summary.csv'),summary)
    with open(os.path.join(args.out,'run_config.json'),'w') as f:
        json.dump(dict(**vars(args),parameters=n_params,device='cpu',
                       torch_version=torch.__version__,
                       train_channel_profile=TRAIN_PROFILE,
                       test_channel_profiles=TEST_PROFILES,
                       sigma=1.,epsilon=1e-5,
                       matched_initial_weights=True,
                       shared_train_and_test_samples=True,
                       identical_architecture=True,
                       fixed_classical_training_prior=True),f,indent=2)
    for split in TEST_PROFILES:
        print('==',split,'10dB pooled==',flush=True)
        for r in summary:
            if r['split']==split and r['snr_db']==10:
                print('   ',r['method'],round(r['pooled_ber'],7),flush=True)
        print('  difference(raw-log) per seed',
              [round(x['raw_minus_log_ber'],6) for x in allpaired
               if x['split']==split and x['snr_db']==10],flush=True)
    print('EXPERIMENT COMPLETE:',args.out,flush=True)

if __name__=='__main__': main()
