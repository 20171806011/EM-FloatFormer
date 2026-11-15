#!/usr/bin/env python3
"""P2-E independent-new-seed statistical audit and immutable final report.

All primary endpoint choices were written to P2E_FREEZE.json before final seed
training. This script reads actual stored per-seed error counts, does not refit.
"""
import csv, hashlib, itertools, json, math, statistics, time
from collections import Counter
from pathlib import Path
import torch
from torch.nn import functional as F
from ofdm_baseline import make_batch,covariance_matrix,estimate_channel
from p2_models import ResidualBase
from p2_run import (N_SC, TRAIN_PROFILE, PROFILES, BASELINES, SNRS, load_models,
                    test_one, summarize, write_csv, timing)

ROOT=Path(__file__).resolve().parent
SEEDS=list(range(2201,2211))
FREEZE_SHA='935ac73e1cf56a2e9c95f51613a883d4ebf05d15eb672a3ec14dd5b189241342'
CAND='res_equiv_local';CONTROL='p1_cnn_log'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def exact_p(values):
    total=2**len(values);threshold=abs(sum(values))
    accepted=sum(abs(sum(a*x for a,x in zip(signs,values)))>=threshold-1e-12
                 for signs in itertools.product((-1,1),repeat=len(values)))
    return accepted/total


def stats(rows):
    key={(int(r['seed']),r['profile'],int(r['snr_db']),r['method']):r for r in rows}
    main=[]
    for profile in ('equal4','long6'):
        delta=[100*(float(key[s,profile,10,CONTROL]['ber'])-float(key[s,profile,10,CAND]['ber'])) for s in SEEDS]
        mean=statistics.mean(delta)
        # Student t (df 9) 97.5% quantile, from numerical reference table.
        h=2.262157163*statistics.stdev(delta)/math.sqrt(len(delta))
        main.append(dict(profile=profile,snr_db=10,unit='BER percentage points',
                         paired_seed_effects_pp=delta,mean_effect_pp=mean,
                         descriptive_t95_interval_pp=[mean-h,mean+h],
                         exact_two_sided_signflip_p=exact_p(delta)))
    order=sorted(range(len(main)),key=lambda i:main[i]['exact_two_sided_signflip_p'])
    maxadjust=0
    for j,i in enumerate(order):
        maxadjust=max(maxadjust,min(1.0,main[i]['exact_two_sided_signflip_p']*(len(main)-j)))
        main[i]['holm_adjusted_p']=maxadjust
    return main


def audit_replay(rows,cp):
    # Replay one seed against 3x 10dB profile cells, including all baselines.
    idx={(int(r['seed']),r['profile'],int(r['snr_db']),r['method']):r for r in rows}
    models=load_models(2201,[CAND,CONTROL],cp)
    replay=[]
    for p in ('train3','equal4','long6'):
        r=test_one(models,2201,p,10,2048,'normal')
        for x in r:
            original=idx[2201,p,10,x['method']]
            assert int(original['errors'])==int(x['errors']), (p,x['method'])
            assert int(original['bits'])==int(x['bits'])
            replay.append((p,x['method'],x['errors']))
    return len(replay)


def audit_equivariance(cp):
    models=load_models(2201,[CAND],cp)
    torch.manual_seed(73763)
    b=make_batch(256,N_SC,10,10,'cpu',TRAIN_PROFILE)
    m=models[CAND]
    with torch.inference_mode():
        old=m.channel(b.yp,b.yd,b.noise_var)
        oldbits=m(b.yp,b.yd,b.noise_var)
        maxh=0;maxlog=0;changed=0
        for a in (.1,.8,1.9,2.9):
            fac=torch.polar(torch.ones_like(b.yp.real),torch.full_like(b.yp.real,a))
            now=m.channel(b.yp*fac,b.yd*fac,b.noise_var)
            log=m(b.yp*fac,b.yd*fac,b.noise_var)
            maxh=max(maxh,float((now-old*fac).abs().max()))
            maxlog=max(maxlog,float((log-oldbits).abs().max()))
            changed+=int(((log>0)!=(oldbits>0)).sum())
    assert maxh<1e-5 and maxlog<1e-5 and changed==0
    return dict(max_channel_equivariance_abs_error=maxh,max_logit_abs_error=maxlog,
                hard_decision_changes=changed)


def audit_lmmse():
    torch.manual_seed(66442)
    b=make_batch(32,32,10,10,'cpu',TRAIN_PROFILE)
    maxdiff=0
    for prof in (TRAIN_PROFILE,PROFILES['equal4'],PROFILES['long6']):
        p=torch.tensor(prof,dtype=torch.float32)
        p=p/p.sum()
        cv=covariance_matrix(32,prof,'cpu')
        h1=estimate_channel(b.yp,b.noise_var,'lmmse',covariance=cv)
        tf=torch.fft.ifft(b.yp,dim=-1)
        factor=p[None,:]/(p[None,:]+b.noise_var/32)
        h2=torch.fft.fft(F.pad(tf[:,:len(prof)]*factor,(0,32-len(prof))),dim=-1)
        maxdiff=max(maxdiff,float((h1-h2).abs().max()))
    assert maxdiff<5e-5
    return maxdiff


def do_plot(summ,out):
    try:
        import matplotlib.pyplot as plt
        profs=('train3','equal4','long6')
        fig,ax=plt.subplots(figsize=(8,4.8))
        for method in (CAND,CONTROL,'ls','dft8','lmmse_matched_prior'):
            y=[100*next(row['ber'] for row in summ if row['profile']==p and row['snr_db']==10
               and row['condition']=='normal' and row['method']==method) for p in profs]
            ax.plot(profs,y,marker='o',label=method)
        ax.set_ylabel('Bit error rate (%)  •  lower is better')
        ax.set_title('P2-E: fresh-seed 10 dB comparison, synthetic QPSK/OFDM')
        ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(out/'ber_10dB.png',dpi=145);plt.close(fig)
        return True
    except ImportError:return False


def report(main,summ,audit,latency):
    q=lambda prof,name:100*next(r['ber'] for r in summ if r['profile']==prof and r['snr_db']==10
                      and r['condition']=='normal' and r['method']==name)
    s=['# EM-FloatFormer P2-E — Final fresh-seed replication and audit', '',
       '**Completed 2026-11-15. Simulated, receive-only civilian QPSK/OFDM ONLY.**', '',
       'The model and both primary endpoints were frozen before final training: [`P2E_FREEZE.json`](P2E_FREEZE.json), SHA256 '+FREEZE_SHA+'.', '',
       '## Design and sample size','',
       '- 10 **fresh** seeds 2201–2210; both P2 equivariant FIR and P1 circular CNN/log retrained from scratch, identical per-seed training examples, **600 steps** each, batch64, AdamW 0.002, train3 only.',
       '- Per seed × condition: 2048 held-out frames × 32 QPSK carriers × 2 bits = 131,072 bits; per receiver / condition total **1,310,720 bits** across 10 seeds.',
       '- Testing profiles train3, equal4, long6 × 0/10/20 dB. Sample unit for confirmatory test: **training seed**, not individual correlated bits.',
       '- Primary (predeclared) effect: BER(P1 CNN/log) - BER(P2 equivariant FIR), in **percentage points**; positive favors P2. Two-sided exact paired sign-flip test and Holm adjustment across equal4@10dB and long6@10dB.', '',
       '## 10 dB held-out pooled BER (%)', '',
       '| Method | Train3 | Unseen equal4 | Unseen long6 |', '|---|---:|---:|---:|']
    for m,label in [(CAND,'P2 strict common-phase-equivariant complex FIR (2,060 params)'),
                    (CONTROL,'P1 CNN/log (38,018 params)'),('ls','Pilot-only LS'),
                    ('dft8','Fixed 8-tap DFT smoothing (known max-delay support)'),
                    ('lmmse_train','LMMSE train3 prior'),
                    ('lmmse_matched_prior','LMMSE matched test prior (privileged)')]:
        s.append(f'| {label} | {q("train3",m):.4f} | {q("equal4",m):.4f} | {q("long6",m):.4f} |')
    s+=['','## Predeclared statistical endpoints','',
        '| Profile | Mean P1 minus P2 (BER pp) | 95% descriptive t interval (pp) | Raw exact p | Holm p |',
        '|---|---:|---|---:|---:|']
    for x in main:
        a,b=x['descriptive_t95_interval_pp']
        s.append(f"| {x['profile']} | {x['mean_effect_pp']:+.4f} | [{a:+.4f}, {b:+.4f}] | {x['exact_two_sided_signflip_p']:.6f} | {x['holm_adjusted_p']:.6f} |")
    s+=['', 'Per-seed primary effects (BER percentage points):','']
    for x in main:s.append(f"- {x['profile']}: "+', '.join(f'{v:+.4f}' for v in x['paired_seed_effects_pp']))
    s+=['', '**Interpretation:** P2 is consistently better than retrained P1 CNN on long6; equal4 includes one seed with a tiny negative difference. Both predeclared two-sided tests meet the 0.05 cutoff after Holm correction in this synthetic benchmark. However, **DFT-8 is still better on both unseen profiles**, and matched-prior LMMSE (privileged) is better still. No universal superiority, no physical radio validation, and no novelty claim for the architectural components. Common phase-equivariance is an exact architecture property to finite precision, not by itself causal proof of BER improvement.', '',
        '## Audits', '',
        f"- Recomputed and matched **{audit['replay_cells']}** stored error-count and bit-count entries from a saved seed checkpoint, with no training.",
        f"- Complex-common-phase test max channel equivariance error {audit['phase']['max_channel_equivariance_abs_error']:.3g}, max equalized-logit difference {audit['phase']['max_logit_abs_error']:.3g}, hard bit flips {audit['phase']['hard_decision_changes']}.",
        f"- Independent Fourier-domain Wiener shrinkage LMMSE vs matrix posterior solve: max difference {audit['lmmse_maxdiff']:.3g}.",
        '- All original per-seed integer errors, bit counts, profiles, held-out PRNGs and checkpoint SHA256 are saved; all 10 seed weights are in the separate local full archive, omitted from public GitHub tree.', '',
        '## CPU-only inference overhead (single measured final-seed run, batch sizes 1/64; excludes RF hardware, FFT chain, transmission and protocol timing)', '',
        '| Method | Params | Batch 1 median ms | Batch 64 median ms |','|---|---:|---:|---:|']
    for method in (CAND,CONTROL):
        z={int(r['batch']):r for r in latency if r['method']==method}
        s.append(f"| {method} | {z[1]['params']} | {float(z[1]['median_ms']):.4f} | {float(z[64]['median_ms']):.4f} |")
    s+=['', '## Reproduce', '', '```bash',
         'python -m pip install -r requirements.txt',
         'python -m unittest test_p2_integrity -v',
         'python p2_run.py --mode smoke',
         'python p2_run.py --mode train --names res_equiv_local p1_cnn_log --seeds 2201 2202 2203 2204 2205 2206 2207 2208 2209 2210 --steps 600 --batch 64 --ckpt p2e_checkpoints --out p2e_training',
         'python p2_run.py --mode eval --names res_equiv_local p1_cnn_log --seeds 2201 2202 2203 2204 2205 2206 2207 2208 2209 2210 --frames 2048 --ckpt p2e_checkpoints --out p2e_recomputed',
         'python p2_finalize.py', '```', '',
         '## Limits and next steps', '',
         'Pilot fully occupies every subcarrier; only one pilot and one payload symbol; static linear channel within pair; idealized perfect synchronization and Gaussian noise. No channel coding, realistic scheduling, dynamic physical impairments, moving vehicles, sample-rate changes, actual antennas, power consumption, or real data. Neural models use the known simulator training distribution and are judged on a handful of synthetic test profiles. Confirmation here means fresh training seeds **within the same implementation and simulator**; not independent laboratory replication. Protocol frozen before testing, but previous exploratory P2 results informed candidate selection. Future P3, if ever approved, requires a stronger multi-symbol, sparse-pilot, receiver-realism benchmark and comparison to classical estimators with matched information. **P2 ends here.**', '']
    return '\n'.join(s)


def main():
    assert sha(ROOT/'P2E_FREEZE.json')==FREEZE_SHA,'P2E candidate protocol was modified'
    config=json.loads((ROOT/'P2E_FREEZE.json').read_text())
    assert config['candidate']==CAND and config['control']==CONTROL
    torch.set_num_threads(4)
    rows=[]
    for group in ('2201_2203','2204_2206','2207_2210'):
        with (ROOT/f'p2e_eval_{group}/ber_by_seed.csv').open(newline='') as f:
            rows.extend(dict(r) for r in csv.DictReader(f))
    methods=[CAND,CONTROL,*BASELINES]
    expect={(s,p,snr,m) for s in SEEDS for p in ('train3','equal4','long6') for snr in SNRS for m in methods}
    actual=[(int(r['seed']),r['profile'],int(r['snr_db']),r['method']) for r in rows]
    assert len(actual)==len(expect)==540 and set(actual)==expect, (len(actual),len(set(actual)))
    for r in rows:
        assert int(r['bits'])==2048*N_SC*2
        assert 0<=int(r['errors'])<=int(r['bits'])
        assert abs(float(r['ber'])-int(r['errors'])/int(r['bits']))<1e-12
    summary=summarize([{**r,'seed':int(r['seed']),'snr_db':int(r['snr_db']),
                      'errors':int(r['errors']),'bits':int(r['bits']),'ber':float(r['ber'])} for r in rows])
    # Convert final aggregated numbers to numeric for report plot/read.
    final=ROOT/'p2e_results';final.mkdir(exist_ok=True)
    write_csv(final/'ber_by_seed.csv',rows)
    write_csv(final/'ber_summary.csv',summary)
    primary=stats(rows)
    (final/'primary_endpoints.json').write_text(json.dumps(primary,indent=2))
    cp=ROOT/'p2e_checkpoints'
    replay_cells=audit_replay(rows,cp)
    audit={'replay_cells':replay_cells,'phase':audit_equivariance(cp),'lmmse_maxdiff':audit_lmmse(),
           'preregistered_freeze_sha256':FREEZE_SHA,'training_seeds':SEEDS,
           'per_seed_test_frames':2048,'unique_cells':len(actual),
           'scope':'synthetic passive civilian QPSK/OFDM only'}
    (final/'final_audit.json').write_text(json.dumps(audit,indent=2))
    models=load_models(2210,[CAND,CONTROL],cp)
    times=timing(models,150)
    write_csv(final/'latency_cpu.csv',times)
    hashes={p.name:sha(p) for p in sorted(cp.glob('*.pt'))}
    assert len(hashes)==20
    (final/'checkpoint_sha256.json').write_text(json.dumps(hashes,indent=2))
    config['status']='completed; no parameters changed after freeze'
    (final/'run_config.json').write_text(json.dumps(config,indent=2))
    do_plot(summary,final)
    (ROOT/'P2E_REPORT.md').write_text(report(primary,summary,audit,times))
    for r in summary:
        if r['snr_db']==10 and r['condition']=='normal':
            print('SUMMARY',r['profile'],r['method'],round(100*r['ber'],4))
    print('PRIMARY',json.dumps(primary,indent=2))
    print('FINAL_AUDIT',json.dumps(audit,indent=2))

if __name__=='__main__':main()
