#!/usr/bin/env python3
"""P1-D: preregistered receive-only civilian synthetic OFDM robustness evaluation.

Reuses the frozen P1-C checkpoints: NO retraining or test-aware tuning. The
32-subcarrier frequency-domain OFDM test model is deliberately limited to
ordinary Gaussian noise, multipath channel and non-adversarial receiver errors.
There is no radar, transmitter, EW, jammer or waveform generation/control.

Examples:
  python p1d_robustness.py --smoke
  python p1d_robustness.py --frames 1024 --seeds 101 202 303 --out p1d_results
  python p1d_robustness.py --verify --frames 1024 --seeds 101 --out p1d_results
"""
import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from ofdm_baseline import make_batch, covariance_matrix, estimate_channel, decisions_from_channel
from p1c_architecture import ARCHES, ENCODINGS, TRAIN_PROFILE, model_factory, write_csv

N_SC = 32
PROFILES = {
    'train3': TRAIN_PROFILE,
    'equal4': (0.25, 0.25, 0.25, 0.25),
    'two_tap': (0.7, 0.3),
    'late_energy3': (0.1, 0.2, 0.7),
    'long6': (0.40, 0.22, 0.15, 0.10, 0.08, 0.05),
}
SNRS = (0, 10, 20)
BASELINES = ('ls', 'dft_train3', 'lmmse_train3', 'lmmse_test_profile_prior', 'true_channel_reference')


@dataclass(frozen=True)
class Scenario:
    name: str
    profile: str
    snr_db: int
    iq_storage: str = 'fp32'
    supplied_noise_db_error: int = 0
    max_phase_drift_rad: float = 0.0


def scenarios():
    # Fixed prior to reading test outcomes: distinct physical perturbation axes.
    tests = [Scenario(f'profile_{p}_{s}dB', p, s) for p in PROFILES for s in SNRS]
    for p in ('train3', 'equal4'):
        for precision in ('fp16_storage', 'bf16_storage'):
            tests.append(Scenario(f'iq_{precision}_{p}_10dB', p, 10, iq_storage=precision))
        for delta in (-6, 6):
            tests.append(Scenario(f'noise_report_{delta:+d}dB_{p}_10dB', p, 10,
                                  supplied_noise_db_error=delta))
        tests.append(Scenario(f'phase_drift_0p35rad_{p}_10dB', p, 10,
                              max_phase_drift_rad=0.35))
    assert len({s.name for s in tests}) == len(tests)
    return tests


def to_complex_storage(x, storage):
    if storage == 'fp32':
        return x
    dtype = {'fp16_storage': torch.float16, 'bf16_storage': torch.bfloat16}[storage]
    return torch.complex(x.real.to(dtype).float(), x.imag.to(dtype).float())


def observed_batch(batch, spec, generator):
    # The same unperturbed simulated frames are used for all methods in a scenario.
    yp = to_complex_storage(batch.yp, spec.iq_storage)
    yd = to_complex_storage(batch.yd, spec.iq_storage)
    if spec.max_phase_drift_rad:
        delta = ((2.0 * torch.rand((yp.shape[0], 1), generator=generator)) - 1.0) * spec.max_phase_drift_rad
        yd = yd * torch.polar(torch.ones_like(delta), delta)
    # Here +6dB means the reported noise power is 3.981x the true variance.
    nv = batch.noise_var * (10.0 ** (spec.supplied_noise_db_error / 10.0))
    return yp, yd, nv


def load_models(checkpoints, seed, archs=ARCHES, encodings=ENCODINGS):
    models = {}
    for arch in archs:
        for encoding in encodings:
            key = f'{arch}__{encoding}'
            path = checkpoints / f'model_{key}_seed{seed}.pt'
            if not path.exists():
                raise FileNotFoundError(f'Missing P1-C checkpoint: {path}')
            obj = torch.load(path, map_location='cpu', weights_only=True)
            for field, expected in [('arch', arch), ('encoding', encoding), ('seed', seed), ('nsc', N_SC)]:
                if obj[field] != expected:
                    raise ValueError(f'{path}: {field} does not match P1-C run')
            model = model_factory(arch, encoding, N_SC)
            model.load_state_dict(obj['state_dict'], strict=True)
            model.eval()
            models[key] = model
    return models


def evaluate_scenario(models, seed, spec, n_frames):
    profile = PROFILES[spec.profile]
    # Same random data across all variants of the same profile/SNR/seed,
    # plus an independent stream for optional phase drift.
    profile_index = list(PROFILES).index(spec.profile)
    data_seed = int(1000000 + seed*10000 + profile_index*100 + spec.snr_db)
    torch.manual_seed(data_seed)
    drift_gen = torch.Generator(device='cpu')
    drift_gen.manual_seed(data_seed + 75000000)
    prior_train = covariance_matrix(N_SC, TRAIN_PROFILE, 'cpu')
    prior_profile = covariance_matrix(N_SC, profile, 'cpu')
    keys = [*models, *BASELINES]
    errors = {key: 0 for key in keys}
    bits = 0
    frames = 0
    with torch.inference_mode():
        for off in range(0, n_frames, 128):
            n = min(128, n_frames-off)
            batch = make_batch(n, N_SC, spec.snr_db, spec.snr_db, 'cpu', profile)
            yp, yd, nv = observed_batch(batch, spec, drift_gen)
            predictions = {key: (m(yp, yd, nv)>0) for key,m in models.items()}
            hs = {
                'ls': estimate_channel(yp, nv, 'ls'),
                'dft_train3': estimate_channel(yp, nv, 'dft', taps=len(TRAIN_PROFILE)),
                'lmmse_train3': estimate_channel(yp, nv, 'lmmse', covariance=prior_train),
                'lmmse_test_profile_prior': estimate_channel(yp, nv, 'lmmse', covariance=prior_profile),
                'true_channel_reference': batch.h,
            }
            predictions.update({key: decisions_from_channel(yd, h, nv).bool() for key,h in hs.items()})
            truth = batch.bits.bool()
            for key,pred in predictions.items():
                assert pred.shape == truth.shape, (key, pred.shape, truth.shape)
                errors[key] += int(torch.count_nonzero(pred != truth).item())
            bits += truth.numel()
            frames += n
    assert frames == n_frames and bits == n_frames * N_SC * 2
    return [dict(seed=seed,scenario=spec.name,profile=spec.profile,snr_db=spec.snr_db,
                 iq_storage=spec.iq_storage,supplied_noise_db_error=spec.supplied_noise_db_error,
                 max_phase_drift_rad=spec.max_phase_drift_rad,method=key,
                 errors=errors[key],bits=bits,ber=errors[key]/bits)
            for key in keys]


def summarize(rows):
    by_key=defaultdict(list)
    for r in rows:
        by_key[(r['scenario'],r['method'])].append(r)
    summaries=[]
    for (scenario,method), items in sorted(by_key.items()):
        err=sum(int(r['errors']) for r in items)
        bits=sum(int(r['bits']) for r in items)
        mean=sum(float(r['ber']) for r in items)/len(items)
        sd=math.sqrt(sum((float(r['ber'])-mean)**2 for r in items)/(len(items)-1)) if len(items)>1 else None
        summaries.append(dict(scenario=scenario,profile=items[0]['profile'],snr_db=items[0]['snr_db'],
                              iq_storage=items[0]['iq_storage'],
                              supplied_noise_db_error=items[0]['supplied_noise_db_error'],
                              max_phase_drift_rad=items[0]['max_phase_drift_rad'],
                              method=method,seeds=len(items),errors=err,bits=bits,
                              pooled_ber=err/bits,mean_seed_ber=mean,sd_seed_ber=sd))
    return summaries


def verify_against(rows, existing):
    observed={(int(r['seed']),r['scenario'],r['method']): (int(r['errors']),int(r['bits'])) for r in rows}
    expected={(int(r['seed']),r['scenario'],r['method']): (int(r['errors']),int(r['bits']))
              for r in existing if (int(r['seed']),r['scenario'],r['method']) in observed}
    if len(observed) != len(expected):
        raise AssertionError(f'verification coverage mismatch: {len(observed)} vs {len(expected)}')
    mismatches=[(k,v,expected[k]) for k,v in observed.items() if v!=expected[k]]
    if mismatches:
        raise AssertionError(f'{len(mismatches)} mismatches, e.g. {mismatches[:3]}')
    return len(observed)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--frames',type=int,default=1024)
    ap.add_argument('--seeds',type=int,nargs='+',default=[101,202,303])
    ap.add_argument('--checkpoints',default='p1c_results')
    ap.add_argument('--out',default='p1d_results')
    ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--verify',action='store_true')
    args=ap.parse_args()
    if args.frames<=0:ap.error('--frames must be positive')
    torch.set_num_threads(min(4,os.cpu_count() or 1))
    cp=Path(args.checkpoints)
    out=Path(args.out)
    tests=scenarios()
    if args.smoke:
        models=load_models(cp,args.seeds[0],archs=('circular_cnn','transformer'))
        rows=evaluate_scenario(models,args.seeds[0],tests[0],8)
        assert len(rows)==len(models)+len(BASELINES)
        for r in rows:
            assert 0<=r['errors']<=r['bits'] and math.isfinite(r['ber'])
        print('P1-D SMOKE PASS: checkpoint loading, multi-method same-frame BER, finite outcomes')
        return
    if args.verify:
        path=out/'ber_by_seed.csv'
        if not path.exists():ap.error(f'verification requires {path}')
        with path.open(newline='') as f:old=list(csv.DictReader(f))
    rows=[]
    for seed in args.seeds:
        models=load_models(cp,seed)
        for i,spec in enumerate(tests):
            rows.extend(evaluate_scenario(models,seed,spec,args.frames))
            if (i+1)%5==0:
                print(f'P1-D seed={seed} scenarios={i+1}/{len(tests)} evaluated',flush=True)
        print(f'P1-D seed={seed} complete',flush=True)
    if args.verify:
        count=verify_against(rows,old)
        print(f'P1-D VERIFY PASS: {count}/{count} exact error-count cells reproduced',flush=True)
        return
    out.mkdir(parents=True,exist_ok=True)
    write_csv(str(out/'ber_by_seed.csv'),rows)
    summaries=summarize(rows)
    write_csv(str(out/'ber_summary.csv'),summaries)
    with (out/'scenarios.json').open('w') as f:
        json.dump([asdict(s) for s in tests],f,indent=2)
    with (out/'run_config.json').open('w') as f:
        json.dump(dict(frames=args.frames,seeds=args.seeds,checkpoints=str(cp),
                       torch_version=torch.__version__,cpu_threads=torch.get_num_threads(),
                       nsc=N_SC,profiles=PROFILES,snrs=SNRS,
                       note='no checkpoint training/tuning; simulated passive civilian receiver only'),f,indent=2)
    print('P1-D DONE',str(out),flush=True)
    for name in ('profile_train3_10dB','profile_equal4_10dB','profile_long6_10dB'):
        print('RESULT',name,flush=True)
        for r in summaries:
            if r['scenario']==name and r['method'] in ('circular_cnn__logamp_phase','transformer__logamp_phase','lmmse_test_profile_prior'):
                print(' ',r['method'],f"{100*r['pooled_ber']:.3f}%",flush=True)


if __name__=='__main__':main()
