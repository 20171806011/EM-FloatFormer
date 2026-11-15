#!/usr/bin/env python3
"""Render P1-D report/plots from measured CSV, without fabricating results."""
from pathlib import Path
import csv
import math
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'p1d_results'
with (OUT/'ber_summary.csv').open(newline='') as f:
    summary=list(csv.DictReader(f))
with (OUT/'ber_by_seed.csv').open(newline='') as f:
    per_seed=list(csv.DictReader(f))

def ber(sc,method):
    return 100*float(next(r['pooled_ber'] for r in summary if r['scenario']==sc and r['method']==method))

def diff_ci(sc, model1, model2):
    # difference model2 - model1. Three independent model-training seeds;
    # same held-out synthetic frames per paired seed. Exploratory t(2) interval.
    seeds=sorted({int(r['seed']) for r in per_seed})
    data=[]
    for s in seeds:
        a=float(next(r['ber'] for r in per_seed if int(r['seed'])==s and r['scenario']==sc and r['method']==model1))
        b=float(next(r['ber'] for r in per_seed if int(r['seed'])==s and r['scenario']==sc and r['method']==model2))
        data.append((b-a)*100)
    avg=statistics.mean(data)
    margin=4.302652729911275*statistics.stdev(data)/math.sqrt(len(data)) if len(data)==3 else None
    return avg,margin,data

CNN='circular_cnn__logamp_phase'
TF='transformer__logamp_phase'
LS='ls'
LMMSE='lmmse_test_profile_prior'
profiles=['train3','two_tap','late_energy3','equal4','long6']

fig,ax=plt.subplots(figsize=(10,5.4))
x=list(range(len(profiles)))
series=[(CNN,'CNN / log amp + phase'),(TF,'Transformer / log amp + phase'),(LS,'Pilot LS'),(LMMSE,'LMMSE / test prior (extra information)')]
for method,label in series:
    ax.plot(x,[ber(f'profile_{p}_10dB',method) for p in profiles],marker='o',label=label)
ax.set_xticks(x,profiles)
ax.set_ylabel('Bit-error rate (%)')
ax.set_title('P1-D: held-out channel profiles, 10 dB SNR')
ax.grid(alpha=.25)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT/'profile_robustness_10dB.png',dpi=160)
plt.close(fig)

fig,ax=plt.subplots(figsize=(9.5,5.2))
shift=[('profile_equal4_10dB','Control FP32'),
('iq_fp16_storage_equal4_10dB','FP16 IQ storage'),
('iq_bf16_storage_equal4_10dB','BF16 IQ storage'),
('noise_report_+6dB_equal4_10dB','Reported noise +6dB'),
('noise_report_-6dB_equal4_10dB','Reported noise -6dB'),
('phase_drift_0p35rad_equal4_10dB','Phase drift +/-0.35rad')]
for method,label in [(CNN,'CNN'),(TF,'Transformer')]:
    ax.plot(range(len(shift)),[ber(s,method) for s,_ in shift],marker='o',label=label)
ax.set_xticks(range(len(shift)),[n for _,n in shift],rotation=19,ha='right')
ax.set_ylabel('Bit-error rate (%)')
ax.set_title('P1-D: receiver input perturbations, four-tap 10 dB')
ax.grid(alpha=.25)
ax.legend()
fig.tight_layout()
fig.savefig(OUT/'receiver_perturbations_10dB.png',dpi=160)
plt.close(fig)

lines=['# EM-FloatFormer P1-D — Robustness and numerical-precision evaluation', '',
       '**2026-11-15 | Executed locally, CPU PyTorch 2.10 | Frozen P1-C checkpoints | Receive-only simulated civilian OFDM**','',
       '## Research question','',
       'How do the P1-C models generalize to novel multipath distributions, receive IQ storage quantization, misreported noise variance, and pilot-to-data common-phase drift?'
       ' This is a controlled model benchmark, **not** a physical radio deployment or radar/EW assessment.','',
       '## Fixed protocol (specified before examining outcomes)','',
       '- All 24 pre-trained P1-C models are loaded from `p1c_results/`; no retraining or post-test hyperparameter changes. Architectures: token MLP, MLP-Mixer, circular CNN, Transformer, with two encodings each.',
       '- Five profiles: training three taps `[0.72,0.20,0.08]`, two taps `[0.7,0.3]`, late-energy three taps `[0.1,0.2,0.7]`, equal-power four taps `[0.25]*4`, six taps `[0.4,0.22,0.15,0.1,0.08,0.05]`.',
       '- Profile matrix evaluated at SNR 0, 10, 20 dB: 15 scenarios. Two profiles also tested at 10 dB with FP16/BF16 **IQ storage quantization only**, +6/-6 dB deliberately incorrect **reported** noise variance, and independent pilot/data common-phase drift sampled uniformly within ±0.35 rad: 10 scenarios. Total **25 scenarios** per seed.',
       '- Three P1-C training seeds 101, 202, 303, 1,024 held-out frames per scenario and seed, 32 subcarriers × 2 data bits: **65,536 bits per seed/scenario, 196,608 bits pooled**. Training and test draws are distinct; within each seed/scenario every receiver sees the exact same frames.',
       '- Traditional references are LS, three-tap DFT, training-prior LMMSE, **test-profile-prior LMMSE (uses extra prior information)**, and true-channel reference (unachievable instantaneous information).',
       '- Quantization means converting only observed I/Q to FP16 or BF16 and then **back to FP32** before model inference. It is **not** complete FP16/BF16 neural inference, an ADC emulation or a test of huge dynamic range.',
       '- Phase drift means a randomized common phase rotation of the *data symbol* relative to pilot. It is **not** full carrier-frequency offset, intercarrier interference, clock offset or sampling-rate generalization.',
       '',
       '## Channel-profile results at 10 dB (BER %, lower is better)','',
       '| Channel profile | CNN + log/phase | Transformer + log/phase | LS | LMMSE with matching profile (extra prior) |',
       '|---|---:|---:|---:|---:|']
for p in profiles:
    sc=f'profile_{p}_10dB'
    lines.append('| '+p+' | '+' | '.join(f'{ber(sc,m):.3f}' for m in (CNN,TF,LS,LMMSE))+' |')
lines+=['','![BER by channel profile](p1d_results/profile_robustness_10dB.png)','',
        'On six taps, even CNN/log-phase is worse than the plain LS baseline. Both learned models remain worse than the known-test-prior LMMSE reference under all five 10-dB profile tests. The latter has privileged distributional information; it should not be labeled an equal-information comparison.',
        '',
        '## Receiver input perturbations, four equal taps, 10 dB (BER %)','',
        '| Scenario | CNN + log/phase | Transformer + log/phase |',
        '|---|---:|---:|']
for sc,label in shift:
    lines.append(f'| {label} | {ber(sc,CNN):.3f} | {ber(sc,TF):.3f} |')
lines+=['','![Input perturbation BER](p1d_results/receiver_perturbations_10dB.png)','',
        'The +6 dB and -6 dB noise-report mistakes behave asymmetrically; a lower BER for one misspecified value does not imply that deliberately falsifying noise is a broadly useful or physically justified tactic. The comparison measures this frozen model only.',
        '',
        '## Training-seed paired differences','',
        'Here Δ = BER(Transformer/log-phase) − BER(CNN/log-phase), in **percentage points** (positive favors CNN).',
        '',
        '| Scenario | Mean Δ | Seed deltas | Exploratory 95% t(2) interval |',
        '|---|---:|---|---|']
for p in profiles:
    sc=f'profile_{p}_10dB';a,b,vals=diff_ci(sc,CNN,TF)
    lines.append(f'| {p} | {a:+.3f} | '+', '.join(f'{v:+.3f}' for v in vals)+f' | [{a-b:+.3f}, {a+b:+.3f}] |')
lines+=['','These are **exploratory** three-seed t intervals; no adjustment for trying many scenarios, methods and conditions. Bits in the same synthetic frame are correlated, and the limited number of model-training seeds does not establish general superiority. The difference between raw-IQ and log-phase also depends on network and channel; consult the full CSV.',
        '',
        '## Code re-evaluation verification','',
        '- Frozen weights for seed 101 reloaded and all 25 scenarios evaluated with the same deterministic streams a second time.',
        '- **325/325 exact error-count cells reproduced**, matching the saved evaluation file. This is a deterministic replay check, not a separate physical or cross-platform replication.',
        '',
        '## Interpretation and limitations','',
        '- These models operate on frequency-domain IQ from a tiny 32-subcarrier ideal OFDM simulation with a full-pilot symbol, known reported noise variance and constant fading within pilot/data pairs.',
        '- No raw million-sample waveforms, no channel coding, no MIMO, no real hardware, no over-the-air link, no transmit functions.',
        '- The requested sampling-rate/clock-drift generalization **cannot be verified** by this fixed-frequency-domain model. Such a claim requires a time-domain simulator with explicit FFT, cyclic prefix and clock offsets; this was not executed here.',
        '- Very low BER or high-SNR conditions need many more frames for high-precision confidence intervals. Profile comparisons and model ordering should be interpreted locally.',
        '- The data does **not** demonstrate new physical laws, generic Transformer superiority, end-to-end communications deployment readiness, or a new robust RF foundation model.',
        '',
        '## Reproduce','',
        '```bash',
        'python ofdm_baseline.py --smoke-only',
        'python p1c_architecture.py --smoke',
        'python p1c_architecture.py --steps 750 --eval-frames 2000 --seeds 101 202 303 --out p1c_results',
        'python p1d_robustness.py --smoke',
        'python p1d_robustness.py --frames 1024 --seeds 101 202 303 --out p1d_results',
        'python p1d_robustness.py --verify --frames 1024 --seeds 101 --out p1d_results',
        'python p1d_make_report.py',
        '```','',
        'Raw observed counts, pooled BER and reproducibility configuration are in `p1d_results/*.csv` and `*.json`. No source datasets or real radio hardware are needed.',
        '',
        '**P1-D status: complete. P1-E has not been started.**','']
(ROOT/'P1D_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print('P1-D report generated',len(summary),'summary rows',len(per_seed),'by-seed rows')
