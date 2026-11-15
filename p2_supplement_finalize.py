#!/usr/bin/env python3
"""Evidence-first aggregation for P2 supplementary scope closure (no tuning)."""
from __future__ import annotations
import csv, hashlib, json, math, statistics, time, zipfile
from pathlib import Path
from collections import defaultdict
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ofdm_baseline import make_batch, covariance_matrix, estimate_channel
import p2_supplement as s

ROOT=Path(__file__).parent
OUT=ROOT/'p2_supplement'
THREADS=4

def rows(path):
    with path.open() as f:return list(csv.DictReader(f))

def summary(tag):return rows(OUT/f'{tag}_results/ber_summary.csv')
def lookup(tag,**terms):
    x=[r for r in summary(tag) if all(str(r[k])==str(v) for k,v in terms.items())]
    assert len(x)==1,(tag,terms,len(x))
    return 100*float(x[0]['ber'])
def pp(x):return f'{x:.3f}%'

def delta_seeds(tag,keyfield,a,b,profile='equal4',snr=10,**others):
    rs=rows(OUT/f'{tag}_results/ber_by_seed.csv')
    grouped=defaultdict(dict)
    for r in rs:
        if r['profile']==profile and int(r['snr_db'])==snr and all(str(r[k])==str(v) for k,v in others.items()):
            grouped[int(r['seed'])][r[keyfield]]=float(r['ber'])
    vals=[100*(grouped[seed][a]-grouped[seed][b]) for seed in sorted(grouped)]
    assert len(vals)==3
    return vals

def generate_training_registry():
    for tag in ('s1','s2','s3'):
        checkpoint=OUT/f'{tag}_checkpoints'
        registry=[]
        for seed in s.SEEDS[tag.upper()]:
            p=checkpoint/f'training_seed{seed}.csv'
            if p.exists(): registry+=rows(p)
            elif tag=='s1' and seed==34001:
                # The checkpoint was saved before an early process-level timeout;
                # elapsed wall clock for this particular training pass is unavailable.
                for model in s.ENCODINGS:
                    registry.append(dict(seed=seed,model=model,steps=s.STEPS,params=35667,
                                         seconds='',note='checkpoint complete; timing unavailable after interruption'))
            else:raise AssertionError(f'missing training registry {p}')
        if not registry:raise AssertionError(tag)
        keys=list(dict.fromkeys(k for r in registry for k in r))
        with (OUT/f'{tag}_results/training_registry.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(registry)


def median_p95_ms(fn,reps=100):
    with torch.inference_mode():
        for _ in range(12):fn()
        ts=[]
        for _ in range(reps):
            t=time.perf_counter_ns();fn();ts.append((time.perf_counter_ns()-t)/1e6)
    ts.sort()
    return round(statistics.median(ts),6),round(ts[math.ceil(.95*reps)-1],6)


def latency():
    torch.set_num_threads(THREADS);out=[]
    for bs in (1,64):
        torch.manual_seed(805015)
        b=make_batch(bs,32,10,10,'cpu',s.TRAIN)
        cov=covariance_matrix(32,s.TRAIN,'cpu')
        for tag in ('s1','s2','s3'):
            if tag=='s1':
                for name in ('rms_global_only','dual_global','rms_local_only','dual_local'):
                    obj=torch.load(OUT/f's1_checkpoints/{name}_seed34001.pt',weights_only=True)
                    m=s.LocalResidualNew(name);m.load_state_dict(obj['state_dict']);m.eval()
                    fn=lambda m=m: m(b.yp,b.yd,b.noise_var)
                    med,p95=median_p95_ms(fn)
                    out.append(dict(branch=tag,method=name,batch=bs,reps=100,median_ms=med,p95_ms=p95,
                                    notes='neural forward only'))
            if tag=='s2':
                count=8;obs,idx=s.sparse_obs(b.yp,count)
                obj=torch.load(OUT/f's2_checkpoints/density8_seed35001.pt',weights_only=True)
                m=s.EquivComplexLocal();m.load_state_dict(obj['state_dict']);m.eval()
                funcs={'sparse_lmmse_train3':lambda:s.sparse_lmmse(obs,idx,b.noise_var,cov),
                       'sparse_dft8_ridge':lambda:s.ridge_dft8(obs,idx,b.noise_var),
                       'lmmse_plus_equiv_neural':lambda:m(s.sparse_lmmse(obs,idx,b.noise_var,cov),b.yd,b.noise_var)}
                for name,fn in funcs.items():
                    med,p95=median_p95_ms(fn)
                    out.append(dict(branch=tag,method=name,batch=bs,reps=100,median_ms=med,p95_ms=p95,
                                    notes='8 known tones, CPU method end-to-end from observed pilot (not RF I/O)'))
            if tag=='s3':
                for name in s.S3_MODELS:
                    obj=torch.load(OUT/f's3_checkpoints/{name}_mixture_seed36001.pt',weights_only=True)
                    m=s.init_s3(name,36001);m.load_state_dict(obj['state_dict']);m.eval()
                    med,p95=median_p95_ms(lambda m=m:m(b.yp,b.yd,b.noise_var))
                    out.append(dict(branch=tag,method=name,batch=bs,reps=100,median_ms=med,p95_ms=p95,
                                    notes='neural forward only'))
                c_prior=torch.load(OUT/'s3_checkpoints/prior_mixture_seed36001.pt',weights_only=True)['covariance']
                med,p95=median_p95_ms(lambda:estimate_channel(b.yp,b.noise_var,'lmmse',covariance=c_prior))
                out.append(dict(branch=tag,method='mixture_empirical_lmmse',batch=bs,reps=100,median_ms=med,p95_ms=p95,
                                notes='full-pilot covariance estimator only'))
    s.write_csv(OUT/'latency_cpu.csv',out)
    return out


def checksum_evidence():
    obj={'protocol_sha256':s.PROTO_SHA,'original_p2e_frozen_hashes':{
       f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
       ('P2E_REPORT.md','P2E_FREEZE.json','P2_PROTOCOL.md','p2e_results/primary_endpoints.json')},
       'new_training_model_hashes':{},'new_prior_hashes':{}}
    for tag in ('s1','s2','s3'):
        for p in sorted((OUT/f'{tag}_checkpoints').glob('*.pt')):
            obj['new_training_model_hashes' if not p.name.startswith('prior_') else 'new_prior_hashes'][f'{tag}_checkpoints/{p.name}']=hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'supplement_checkpoint_sha256.json').write_text(json.dumps(obj,indent=2))


def plot_data():
    profiles=('train3','equal4','long6')
    fig,ax=plt.subplots(figsize=(9,4.5))
    choices=('raw','rms_global_only','dual_global','rms_local_only','dual_local','log')
    x=list(range(len(profiles)))
    for name in choices:
        ax.plot(x,[lookup('s1',profile=p,snr_db=10,method=name) for p in profiles],marker='o',label=name)
    ax.set_xticks(x,profiles);ax.set_ylabel('BER at 10 dB (%) ↓');ax.set_title('S1: RMS-only and log scale branches');ax.legend(ncol=3,fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'s1_results/s1_ablation_10dB.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    counts=(32,16,8,4)
    for name in ('sparse_ls_interp','dft8_ridge','lmmse_train3','res_equiv_lmmse_residual','lmmse_matched_privileged'):
        ax.plot(counts,[lookup('s2',profile='long6',snr_db=10,pilots=k,method=name) for k in counts],marker='o',label=name)
    ax.set_xticks(counts);ax.set_xlabel('Known pilot tones out of 32');ax.set_ylabel('BER (%) ↓');ax.set_title('S2: long6, sparse-pilot information budget');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'s2_results/s2_long6_pilot_density.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    px=('train3','equal4','long6','two_tap','late3','eight_tap');xx=list(range(len(px)))
    for name in ('train3__p1_cnn_log','mixture__p1_cnn_log','train3_empirical_lmmse','mixture_empirical_lmmse','dft8'):
        ax.plot(xx,[lookup('s3',profile=p,snr_db=10,method=name) for p in px],marker='o',label=name)
    ax.set_xticks(xx,px,rotation=20);ax.set_ylabel('BER (%) ↓');ax.set_title('S3: training-information route comparison');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(OUT/'s3_results/s3_mixture_generalization.png',dpi=160);plt.close(fig)


def compile_report():
    s1_rows=[]
    for prof in ('train3','equal4','long6'):
        x=[prof]
        for n in ('raw','log','rms_global_only','dual_global','rms_local_only','dual_local'):
            x.append(pp(lookup('s1',profile=prof,snr_db=10,method=n)))
        s1_rows.append('| '+' | '.join(x)+' |')
    s2_rows=[]
    for prof in ('equal4','long6'):
        for count in (32,16,8,4):
            x=[prof,str(count)]
            for n in ('sparse_ls_interp','dft8_ridge','lmmse_train3','res_equiv_lmmse_residual','lmmse_matched_privileged'):
                x.append(pp(lookup('s2',profile=prof,snr_db=10,pilots=count,method=n)))
            s2_rows.append('| '+' | '.join(x)+' |')
    s3_rows=[]
    for prof in ('train3','equal4','long6','two_tap','late3','eight_tap'):
        x=[prof]
        for n in ('train3__p1_cnn_log','mixture__p1_cnn_log','train3__res_equiv_local','mixture__res_equiv_local',
                  'train3_empirical_lmmse','mixture_empirical_lmmse','dft8'):
            x.append(pp(lookup('s3',profile=prof,snr_db=10,method=n)))
        s3_rows.append('| '+' | '.join(x)+' |')
    delta=[]
    for a,b,prof in [
      ('rms_global_only','dual_global','equal4'),('rms_global_only','dual_global','long6'),
      ('rms_local_only','dual_local','equal4'),('rms_local_only','dual_local','long6')]:
        vals=delta_seeds('s1','method',a,b,profile=prof)
        delta.append(f'| {prof} | `{a}` − `{b}` | {statistics.mean(vals):+.3f} | '+', '.join(f'{v:+.3f}' for v in vals)+' |')
    mixed_d=[]
    for arch in s.S3_MODELS:
        for prof in ('late3','eight_tap'):
            vals=delta_seeds('s3','method','train3__'+arch,'mixture__'+arch,profile=prof)
            mixed_d.append(f'| `{arch}` | {prof} | {statistics.mean(vals):+.3f} | '+', '.join(f'{v:+.3f}' for v in vals)+' |')
    lat=rows(OUT/'latency_cpu.csv')
    summary10=[r for r in lat if r['batch']=='1']
    tlines=['| '+ ' | '.join((r['branch'],r['method'],f"{float(r['median_ms']):.3f}",f"{float(r['p95_ms']):.3f}"))+' |' for r in summary10]
    n1=sum(1 for _ in rows(OUT/'s1_results/ber_by_seed.csv'))
    n2=sum(1 for _ in rows(OUT/'s2_results/ber_by_seed.csv'))
    n3=sum(1 for _ in rows(OUT/'s3_results/ber_by_seed.csv'))
    ckpt=sum(1 for tag in ('s1','s2','s3') for _ in (OUT/f'{tag}_checkpoints').glob('*.pt'))
    s1pair=statistics.mean(delta_seeds('s1','method','rms_local_only','dual_local',profile='long6'))
    p2report=f'''# EM-FloatFormer — P2 plan-alignment supplementary audit results

**Completed: 2026-11-15.** **New exploratory experiments** under a separate protocol with SHA256 `{s.PROTO_SHA}`; local freeze git commit `682f868f7a8bdca70a83943b8a611723c3c9850c` recorded *before any S1/S2/S3 training*. The GitHub publication is later than execution, so this is **not** an externally timestamped preregistration. The original P2-E frozen protocol, report, endpoints, seeds and conclusion are **unchanged**. No active transmission, radar, military systems or hardware control.

## Plan → gap → evidence closure

| Original P2 scope | Confirmed prior omission | New evidence | Conclusion |
|---|---|---|---|
| P2-B separate RMS-only/local normalized IQ | P2-B had only `raw`, `log`, frame-wide `dual`, not independent RMS-only or local 5-tone window | `p2_supplement.py` S1; `s1_results/ber_by_seed.csv`, `ber_summary.csv` | **Performed**, six paired representations, including global and local RMS-only controls. |
| P2-D sparse pilot densities | Existing OFDM observation had 32/32 known pilot tones | S2, masked pilot 32/16/8/4, 4 classical controls; masks/linear algebra independently checked | **Performed**, limited to evenly spaced pilots, fixed per-tone pilot power. |
| Independent mixed-channel training route | Original P2 A–E train3 only; matched-test prior not information-matched | S3 separate 400-step 3-seed train3 vs mixture arms and finite-training covariance estimates | **Performed**, 2 architectures × 2 regimes; held-out late3/eight_tap. |

## Frozen supplementary design and provenance

- New seeds: S1 `34001/34002/34003`, S2 `35001/35002/35003`, S3 `36001/36002/36003`, **none overlap** P2-E `2201..2210`. Each arm is trained from scratch for **400** steps with batch64 (shorter than the old 600-step P2-E runs and **not comparable as a confirmatory replication**). AdamW lr0.002, weight decay 1e-4, gradient clip 1.0, SNR 0–20 dB.
- New evaluation: **512 independent frames per seed/scenario**, 32 uncoded data QPSK carriers × 2 = 64 bits/frame, so **98,304 pooled bits** per arm/scenario across 3 seeds. Evaluation SNR 0/10/20 dB; tables below highlight 10 dB. Paired receiver predictions use identical sample frames per scenario.
- Logged raw integer error counts: **S1 {n1} rows, S2 {n2} rows, S3 {n3} rows**. Saved **{ckpt} checkpoint and training-prior files** with SHA256; files excluded from lightweight GitHub publication but included in full offline archive. No confirmatory p-values, no post-test tuning or added training steps.
- Earlier P2-E used ten fresh seeds and Holm-corrected exact tests: **its original two endpoints and p=0.00390625 remain intact**, without these supplements changing their interpretation. The narrower frozen finding remains that P2-E phase-equivariant model beat the earlier P1 CNN on those synthetic test profiles; DFT8 still won against the new neural model.

## S1 — RMS-only vs explicit log-scale side channel

BER at 10 dB, %, lower is better:

| Profile | Raw | Log/phase | Global RMS-only | Global RMS+log | Local 5-tone RMS-only | Local RMS+log |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(s1_rows)}

**Frozen paired contrasts** (mean of 3 seed-wise differences, in BER **percentage points**; positive `RMS-only − dual` means the log side path helped):

| Test profile | Contrast | Mean pp | Per-seed pp |
|---|---|---:|---|
{chr(10).join(delta)}

**Interpretation:** Adding log RMS to frame-wide RMS normalized IQ helped slightly in the two specified OOD profiles, but with the local five-tone normalizer the log side branch **hurt** on both. At long6 the local-RMS-only branch beat local dual by about {s1pair:+.3f} pp under the defined sign convention. This is **not** a universal lossless representation result: RMS-only sets one channel to zero, auxiliary features and residual scaling contain other magnitude information, and changing neighborhoods changes inductive biases. All six models have the same 35,667 parameter topology/initialization per seed, but the supplemental training used new 400-step budgets (not the original P2-B 600). `s1_results/s1_ablation_10dB.png` plots all six representations.

## S2 — pilot density and fair sparse-pilot baselines

BER at 10 dB, %, lower is better:

| Profile | Known pilots | Circular sparse-LS interpolation | DFT8 + fixed ridge | Train3-prior LMMSE | Learned residual on train3-prior LMMSE | Test-profile LMMSE **privileged** |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(s2_rows)}

**Interpretation:** Classical estimates and learned residual use only the selected pilot observations. The neural model uses the train3-prior LMMSE estimate as its base; it does not see missing pilot observations. This helps on the training profile but remains highly biased on unknown four- and six-tap laws: its performance is close to the **mismatched** train3-prior LMMSE, often much worse than a generic DFT8 with 8–32 pilots. With just 4 pilots, an 8-tap channel is underdetermined and DFT8-ridge loses substantial accuracy. The matched-test-prior LMMSE in the last column knows the test profile and is not equal-information for OOD. Dropping pilot tones also drops total pilot energy at fixed per-tone power: this is **not** a constant-total-training-energy comparison. The baseline uses complex periodic interpolation, a fixed-ridge delay-domain estimator, a train-prior posterior and a privileged matched-profile posterior. 0/20 dB results and **train3** results are preserved in CSV. No trained model is claimed to be optimal.

## S3 — mixed channel information on both sides

BER at 10 dB, %, lower better. `mixture` training population = 1/4 each train3, equal4, long6, two_tap. `late3` and `eight_tap` were held out:

| Test profile | CNN train3 | CNN mixture | Equivariant train3 | Equivariant mixture | Empirical LMMSE train3 prior | Empirical LMMSE mixture prior | Fixed DFT8 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(s3_rows)}

**Predeclared held-out architecture contrasts:** `train3 BER − mixture BER` (positive = mixed training helps), percentage points; 3 new seeds, descriptive only:

| Architecture | Unseen profile | Mean pp | Per-seed pp |
|---|---|---:|---|
{chr(10).join(mixed_d)}

- Mixing training profiles improved CNN on long6 and entirely unseen eight_tap, but not on every profile: for example two_tap CNN could worsen. This should not be confused with extrapolation from a train3-only model, because `equal4` and `long6` **are included in mixture training**.
- The 2,060-parameter equivariant model did **not** gain comparable performance in the mixed-information route under the frozen **400-step** budget. It remained near 8.5–9.1% on several profiles, including train3. Possible optimization underfitting is a hypothesis, not experimentally proven; we did not increase training steps after viewing outcomes.
- Traditional empirical covariance LMMSE was estimated from the **same 25,600 training pilot frames per seed/regime**, with average noise variance subtracted and a fixed PSD projection. Thus it uses the same declared training distribution and no test-specific covariance. It improved markedly on mixture-member equal4/long6; it remained mismatched on held-out eight_tap. Its parametric family (linear covariance estimator) and supervision differ from CNN, so equal exposure does not imply identical statistical capacity. `matched_test_privileged` is separately reported only in raw evidence.
- A fixed DFT8 delay-support prior (known support 0–7, no per-profile power prior) remains a strong classical benchmark. On held-out eight_tap, it outperformed all the newly trained neural models. There is no generalized neural-algorithm superiority claim.

## Independent technical checks, CPU cost and scope boundaries

- `python -m unittest test_p1d_invariants test_p1e_integrity test_p2_integrity test_p2_supplement_independent -v` returned **35/35 PASS**. Includes 13 new independent supplement tests: masked-pilot nonleakage even after corrupting all hidden pilot samples by large values; exact sparse-pilot interpolation at observed tones; independent Fourier/tap-space sparse posterior vs full covariance computation; DFT ridge normal-equation residual; 32-pilot posterior vs original code; RMS channel isolation; matched bits and noise in mixed/train3 streams; reconstructed BER error-count bookkeeping and three separate checkpoint replays. Mathematical checks are numerical within stated tolerances.
- CPU latency (PyTorch 2.10 CPU; batch1, 100 repetitions after warmup). These times exclude hardware, FFT/ADC, protocol, signal acquisition, host transfers and end-to-end RF latency; batch64 and p95 in `p2_supplement/latency_cpu.csv`:

| Branch | Method | Median ms | p95 ms |
|---|---|---:|---:|
{chr(10).join(tlines)}

- Deterministic replay verifies 6 S1 methods at equal4/10dB, a masked-pilot learned S2 method at long6/10dB, and 4 S3 methods at eight_tap/10dB for one new seed per branch (11 integer-count cells). It is **not** an independent codebase or physical replication. No tests on live RF, sample-rate variation, coding, real oscillators, 5G specifications, asynchronous multi-frame fading or active emission.
- **Local-only freeze provenance:** the signed chronological implication cannot be independently established from the later GitHub upload; it is an internal freeze commit before execution, with printed content hash, not a third-party preregistration timestamp.
- S1 seed34001 checkpoints completed before a process-level time limit; separate training runtime for that seed was not retained, so per-seed timing is blank in `training_registry.csv`. Its checkpoint protocol SHA and saved-count replay passed. 34002/34003 and all other training timings are preserved.

## Scope conclusion

The three omissions are now **substantively tested**, not assumed successful. They produce mixed or negative results, reinforcing that representation, pilot availability, channel-prior information and optimization budget can dominate architectural comparisons. The original P2-E freeze and conclusions stand separate. The final scope has progressed from original P2-A–E (full-pilot/train3) to **P2-S1/S2/S3 independently logged exploratory supplements**; neither set establishes superior general RF intelligence or an operational receiver.

Evidence: `P2_SCOPE_ALIGNMENT_AUDIT.md`, `P2_SUPPLEMENT_PROTOCOL.md`, `p2_supplement.py`, `test_p2_supplement_independent.py`, `p2_supplement/*_results/*.csv`, new training-prior SHA256 manifest, and original P2-E files unchanged.
'''
    (ROOT/'P2_SUPPLEMENT_REPORT.md').write_text(p2report)
    return len(p2report)


def finish():
    generate_training_registry();latency();checksum_evidence();plot_data()
    size=compile_report()
    print('Written supplement report',size,'chars, original P2-E untouched')

if __name__=='__main__':finish()
