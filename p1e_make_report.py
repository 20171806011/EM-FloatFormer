#!/usr/bin/env python3
"""Render a reproducible P1-E research report from measured CSV/JSON only."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).parent
R=ROOT/'p1e_results'

def csv_rows(filename):
    with (R/filename).open(newline='') as f:return list(csv.DictReader(f))

s=csv_rows('ber_summary.csv')
t=csv_rows('latency_cpu.csv')
p=json.loads((R/'primary_endpoints.json').read_text())
v=json.loads((R/'replay_verification.json').read_text())
train=json.loads((R/'run_config.json').read_text())
assert v['status']=='PASS' and v['exact_matching_cells']==81
assert len(set(train['seeds']))==5
assert len(s)==81

def ber(profile,method,snr=10):
    a=[row for row in s if row['profile']==profile and row['method']==method and int(row['snr_db'])==snr]
    assert len(a)==1
    return 100*float(a[0]['pooled_ber'])

def f(num):return f'{num:.3f}'

lines=[]
def add(txt=''):lines.append(txt)
add('# EM-FloatFormer P1-E — New-seed replication and computational cost audit')
add('')
add('**2026-11-15 · Completed · CPU PyTorch · Synthetic receive-only civilian QPSK/OFDM**')
add('')
add('## Summary')
add('')
add('The two P1-D out-of-distribution architectural comparisons were tested with **five entirely new training seeds** (404, 505, 606, 707, 808), **20 new checkpoints** (CNN/Transformer × two encodings), and **2048 newly generated test frames per seed/scenario**. At 10 dB, CNN/log-amplitude/phase had lower BER than Transformer/log-amplitude/phase on four-tap and six-tap tests for **all five seeds**. The corrected exact two-sided sign-flip test, however, returned **Holm-adjusted p=0.125** for both endpoints, which does not pass the 0.05 decision threshold. It is not justified to claim confirmatory statistical significance. Well-calibrated matched-prior LMMSE remained better in all three reported 10-dB channels but uses extra distributional prior knowledge outside the equal-information model comparison.')
add('')
add('## Preregistered protocol and statistical design')
add('')
add('- Prespecified protocol: [`P1E_PROTOCOL.md`](P1E_PROTOCOL.md), written before test evaluation. Two primary comparisons: CNN/log vs Transformer/log on equal-four-tap and six-tap channel profiles, both at 10 dB.')
add('- **P1-C training code and settings unchanged**: 32 subcarriers, known all-subcarrier pilot, static multipath within pilot/data, 750 optimization steps, 64 frames/batch, learning rate 0.002, train3 profile `[0.72,0.20,0.08]`, random training SNR 0–20 dB. 20 new checkpoints; no test-aware tuning.')
add('- Each seed and each of three profiles (`train3`, `equal4`, `long6`) × three SNRs (0/10/20 dB): 2048 new frames, 64 bits/frame = 131072 bits per seed/scenario = **655360 pooled bits** per receiver/scenario across five seeds. All models and conventional references get identical within-scenario data. Test seeds are different from P1-C/P1-D.')
add('- Statistical sampling unit is **training seed**, not individual correlated bit. For each primary endpoint, delta = BER(Transformer/log) − BER(CNN/log), positive favors CNN. Report two-sided exact enumeration of all `2^5=32` seed sign assignments, Holm-adjusted p values across two comparisons, and descriptive two-sided t(4) confidence intervals. Secondary comparisons are exploratory.')
add('')
add('## Held-out BER results (10 dB, percentage; lower better)')
add('')
add('| Receiver | Train3 | Equal4 (OOD) | Long6 (OOD) |')
add('|---|---:|---:|---:|')
methods=[('CNN + raw IQ','circular_cnn__raw_iq_magnitude'),
         ('CNN + log amplitude/phase','circular_cnn__logamp_phase'),
         ('Transformer + raw IQ','transformer__raw_iq_magnitude'),
         ('Transformer + log amplitude/phase','transformer__logamp_phase'),
         ('Pilot LS','ls'),
         ('DFT, fixed 3-tap support','dft_train3'),
         ('LMMSE, train3 prior','lmmse_train3'),
         ('LMMSE, **matched test-profile prior**','lmmse_matched_profile_prior'),
         ('True-channel reference (unavailable)','true_channel_reference')]
for label,method in methods:
    add(f'| {label} | {f(ber("train3",method))}% | {f(ber("equal4",method))}% | {f(ber("long6",method))}% |')
add('')
add('The matched-test-prior LMMSE is not information-matched to models (it receives the actual test profile as a statistical prior); it is an informative classical reference, **not** proof of intrinsic superiority when information differs. In the six-tap profile, even the best tested CNN remains worse than pilot-only LS. As expected, the true-channel reference is unavailable to practical systems.')
add('')
add('## Primary test results')
add('')
add('| Prespecified endpoint | Mean delta, BER percentage points | Per-seed deltas, pp | Two-sided 95% t interval, pp | Exact two-sided sign-flip p | Holm-adjusted p |')
add('|---|---:|---|---|---:|---:|')
for x in p:
    lo,hi=x['interval_95_pp']
    add(f'| {x["profile"]} @ 10dB | +{f(x["mean_delta_ber_pp"])} | '+', '.join(f'{z:+.3f}' for z in x['seed_delta_ber_pp'])+f' | [{lo:+.3f}, {hi:+.3f}] | {x["signflip_p_two_sided"]:.4f} | {x["holm_adjusted_p"]:.4f} |')
add('')
add('With only five training seeds, a two-sided exact sign-flip test cannot have a p value below `2/32=0.0625`, even when all five deltas share a sign. The 95% **parametric** t intervals exclude zero, while the conservative exact nonparametric tests do not pass 0.05 after Holm correction. These tests use different assumptions; do not report the parametric interval alone as confirmatory proof. The endpoints were motivated by earlier P1-D results, which were not independent preregistered discoveries. This P1-E evaluation is a fresh-seed replication, not an independently implemented or physical-radio replication.')
add('')
add('## Exploratory encoding comparisons')
add('')
add('| Profile, 10dB | CNN raw IQ BER | CNN log/phase BER | Difference (raw − log) |')
add('|---|---:|---:|---:|')
for profile in ('train3','equal4','long6'):
    raw=ber(profile,'circular_cnn__raw_iq_magnitude');log=ber(profile,'circular_cnn__logamp_phase')
    add(f'| {profile} | {raw:.3f}% | {log:.3f}% | {raw-log:+.3f} pp |')
add('')
add('Log/phase helps the CNN in the two tested out-of-distribution profiles, but hurts it on the in-distribution profile. It is not universally better, and the network only accepts normalized/simulated baseband I/Q, not RF measurements across extreme ADC dynamic ranges. Other SNR (0/20 dB) and model cells appear in [`p1e_results/ber_summary.csv`](p1e_results/ber_summary.csv).')
add('')
add('## Measured CPU inference cost')
add('')
add('**300 timing repetitions, 20 warm-ups, torch.set_num_threads(4); includes neural feature-extraction forward path but not simulator, SDR hardware, transport, FFT or decode protocol.**')
add('')
add('| Model | Parameters | Batch 1 median / p95 (ms) | Batch 128 median / p95 (ms) |')
add('|---|---:|---:|---:|')
for arch in ('circular_cnn','transformer'):
    for enc in ('raw_iq_magnitude','logamp_phase'):
        name=f'{arch}__{enc}'
        a=[x for x in t if x['method']==name and int(x['batch_frames'])==1][0]
        b=[x for x in t if x['method']==name and int(x['batch_frames'])==128][0]
        add(f'| {name} | {int(a["parameters"]):,} | {float(a["median_ms"]):.3f} / {float(a["p95_ms"]):.3f} | {float(b["median_ms"]):.3f} / {float(b["p95_ms"]):.3f} |')
add('')
add('Performance on one CPU is hardware-, thread- and batch-dependent. These numbers are not evidence of an operational sub-millisecond over-the-air receiver.')
add('')
add('## Integrity and reproducibility')
add('')
add(f'- **Exact replay**: reloaded the five-seed training checkpoint for seed {v["seed"]} and recomputed {v["tested_scenarios"]} scenario cells; **{v["exact_matching_cells"]}/{v["total_cells"]} recorded error-count cells matched exactly**. This is repeatability on the same CPU/PyTorch environment, not cross-platform independence.')
add('- Channel/noise simulator check: generated pilot thermal-noise variance ≈0.10043 for target 0.1 at 10 dB; theoretical covariance is Hermitian; **independent Fourier-domain Wiener shrinkage expression** numerically matches the matrix-solve LMMSE implementation (max absolute deviation below 5e-5 in tested cases).')
add('- Source and model metadata: [`p1e_results/run_config.json`](p1e_results/run_config.json), [`p1e_results/checkpoint_sha256.json`](p1e_results/checkpoint_sha256.json). Saved raw counts: [`p1e_results/ber_by_seed.csv`](p1e_results/ber_by_seed.csv). Prespecified differences: [`p1e_results/primary_endpoints.json`](p1e_results/primary_endpoints.json).')
add('- Model weights are packaged in the *full archive* but omitted from the lightweight public GitHub tree to keep the public repository small. P1-C models are also not modified.')
add('')
add('## Reproduction')
add('')
add('```bash')
add('python p1c_architecture.py --steps 750 --batch-size 64 --eval-frames 2000 \\')
add('  --seeds 404 505 606 707 808 --arches circular_cnn transformer \\')
add('  --encodings raw_iq_magnitude logamp_phase --out p1e_train')
add('python p1e_replication.py --smoke')
add('python p1e_replication.py --checkpoints p1e_train --out p1e_results --frames 2048')
add('python p1e_replication.py --verify --checkpoints p1e_train --out p1e_results --frames 2048')
add('python p1e_make_report.py')
add('```')
add('')
add('**Limits:** Idealized, frequency-domain, perfect synchronization, 32-subcarrier full-pilot uncoded OFDM, synthetic Gaussian thermal noise and static channel, exactly known noise variance in unperturbed scenarios. No fading within each pilot-data pair, no real SDR data, no blind channel estimation, no sample-rate changes, no physical hardware/energy efficiency, no million-sample input, no active electronic warfare or transmitter-control algorithms. P1-E reports a narrow repeatable benchmarking result, not a newly discovered RF principle, a deployed radio receiver, or evidence of a universal superior architecture.')
add('')
add('**Phase-1 status: P1-A through P1-E complete.** Future research may extend the synthetic model to richer benign communications settings, but this stage ends here.')

path=ROOT/'P1E_REPORT.md'
path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(f'Wrote {path}, {len(lines)} lines, {path.stat().st_size} bytes')
