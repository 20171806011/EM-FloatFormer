#!/usr/bin/env python3
"""Generate evidence-based P3-B report from frozen result CSVs, without training or tuning."""
import csv,json,hashlib,statistics,collections
from pathlib import Path
from p3b_time_link import PROFILES,ALL_CONDITIONS
R=Path(__file__).resolve().parent
OUT=R/'p3b_results'
def read(path):
    with open(path,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def mdtable(cols,rows):
    return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+['| '+' | '.join(map(str, r))+' |' for r in rows])
def fmt(v):return f'{100*float(v):.3f}%'
def main():
    data=read(OUT/'ber_summary.csv'); idx={(x['profile'],int(x['snr_db']),x['condition'],x['method']):x for x in data}
    neural=['clean__raw','clean__log_phase','clean__rms_log','mixed__raw','mixed__log_phase','mixed__rms_log']
    trad=['ls','dft8','dft12','threshold','threshold_no_align','train_lmmse','dd_threshold']
    minor=[]; major=[]
    for p in ['equal4','long6']:
        for a in ['clean','pa_mild','impulse_mild','cpe_mild','adc_mild','pa_strong','impulse_strong','cpe_strong','adc_strong']:
            get=lambda m:float(idx[p,10,a,m]['ber'])
            cb=min((get(m),m) for m in trad)
            nb=min((get(m),m) for m in neural)
            (minor if a=='clean' or a.endswith('_mild') else major).append([p,a,fmt(get('frozen_d')),fmt(get('clean__raw')),fmt(get('mixed__raw')),
                      f'{fmt(nb[0])} ({nb[1]})',f'{fmt(cb[0])} ({cb[1]})'])
    axis=[]
    for a in ALL_CONDITIONS:
        row=idx['equal4',10,a,'ls']
        axis.append([a,f'{float(row["tx_power_ratio"]):.3f}',f'{float(row["burst_rate"]):.4f}',
                     f'{float(row["cpe_abs_mean"]):.4f}',f'{float(row["adc_saturation"]):.4f}'])
    d8=[]
    for p in ['train3','equal4','long6','long12','sparse12']:
        get=lambda m:idx[p,10,'clean',m]['ber']
        d8.append([p,fmt(get('frozen_d')),fmt(get('mixed__raw')),fmt(get('mixed__log_phase')),fmt(get('mixed__rms_log')),fmt(get('threshold_no_align')),fmt(get('dft12'))])
    priv=read(OUT/'privileged_reference_by_seed.csv')
    priv_agg=collections.defaultdict(list)
    for x in priv:priv_agg[(x['profile'],int(x['snr_db']),x['condition'],x['method'])].append(x)
    full=[]
    for k,rr in sorted(priv_agg.items()):
        n=sum(int(x['bits']) for x in rr); err=sum(int(x['errors']) for x in rr)
        full.append(dict(profile=k[0],snr_db=k[1],condition=k[2],method=k[3],errors=err,bits=n,ber=err/n))
    with (OUT/'privileged_reference_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(full[0]));w.writeheader();w.writerows(full)
    privtable=[]
    for p in ['equal4','long6','long12','sparse12']:
        r={(x['profile'],x['condition'],x['method']):x for x in full if x['snr_db']==10}
        privtable.append([p,fmt(r[p,'clean','matched_test_prior_privileged']['ber']),fmt(r[p,'clean','true_channel_privileged']['ber']),fmt(idx[p,10,'clean','threshold_no_align']['ber'])])
    timings=read(OUT/'latency_fullpath_cpu.csv')
    def tm(batch,task):return next((f'{float(x["median_ms"]):.3f} / {float(x["p95_ms"]):.3f}' for x in timings if int(x['batch_frames'])==batch and x['task']==task),'not measured')
    timed=[]
    for task in ['simulated_link_only','mixed_raw_head_only','frozen_backbone_only','mixed_raw_backbone_plus_head','threshold_no_anchor_postlink','neural_full_simulated_pipeline','threshold_no_anchor_full_simulated_pipeline']:
        timed.append([task,tm(1,task),tm(64,task)])
    audit=json.loads((OUT/'independent_audit.json').read_text())
    # Emit full neutral plots (predefined fixed model selection, not per-cell winner chasing).
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for p in ['equal4','long6']:
        cases=['clean','pa_mild','pa_strong','impulse_mild','impulse_strong','cpe_mild','cpe_strong','adc_mild','adc_strong']
        fig,ax=plt.subplots(figsize=(11,4.8))
        for method in ['threshold_no_align','frozen_d','clean__raw','mixed__raw','mixed__rms_log']:
            ax.plot(range(len(cases)),[100*float(idx[p,10,c,method]['ber']) for c in cases],marker='o',label=method)
        ax.set_xticks(list(range(len(cases))),cases,rotation=35,ha='right')
        ax.set_ylabel('BER (%) · lower is better')
        ax.set_title(f'P3-B / {p} / 10 dB · all nine conditions · exploratory')
        ax.grid(alpha=.22)
        ax.legend(fontsize=8,ncol=3)
        fig.tight_layout();fig.savefig(OUT/f'{p}_10db_robustness.png',dpi=150)
        plt.close(fig)
    lines=[]
    lines.append('''# EM-FloatFormer P3-B — single-factor nonidealities in a synthetic civilian OFDM receiver

**2026-11-15 · completed exploratory benchmark · passive receive-only simulation**

## Snapshot and provenance

P3-B protocol was committed to the original repository **before training** as `P3B_PROTOCOL.md` (commit `440c4aba5881306e29165727dee03cb7f0dc1550`). This run does not amend previously frozen P2-E, P3-A or P3-A2 findings. This report is generated after model evaluation using fixed outputs, not a revised preregistration. No real radio hardware, radar, military data, adversarial emissions or transmitter control are used.

**Principal finding: No trained neural receiver surpasses the per-condition envelope of the seven predeclared nonprivileged classical methods in any of the 135 scenario/SNR/impairment cells.** The 30 trained heads also add very little to the old frozen D backbone. This is a *descriptive* comparison: choosing the minimum classical error retrospectively within a fixed predeclared method list forms a classical-favoring **oracle selection envelope**, not a single deployable adaptive chooser. We also report individual fixed classical methods so no inference depends on that envelope.

**Do not call neural logit residual scale-aligned symbol MSE 'measured EVM'.** It is a diagnostic computed from arbitrary classification logits, not calibrated physical RF error-vector magnitude. The channel MSE applies to pilot-derived channel estimates (including frozen D); the neural heads do not emit a separately learned channel estimate.
''')
    lines.append('''## B0: time-domain link checks

- New random QPSK pilots, 32 subcarriers, 16-point CP, two publicly known QPSK anchors in the data symbol. The **30 unknown QPSK tones** yield 60 scored bits per frame; anchors do not enter BER.
- Explicit IFFT -> CP -> optional memoryless PA magnitude compression -> causal multipath FIR -> CP removal -> time-domain receiver noise/ADC/CPE -> FFT. Noise power is scaled to yield `N0` per frequency tone. The zero-impairment, zero-noise limit agrees with the old `H×X` frequency model to float tolerance, while having different random streams.
- `test_p3b_link.py`: **12/12 tests pass**, including calibration and no leakage. The additional LMMSE Fourier-delay formula was independently cross-checked against a direct matrix solve before training (maximum tested discrepancy around 2.9e-6).
- This chain does not include a standardized 3GPP channel, changing sample clocks, realistic analog front ends or multiple moving OFDM symbols. It does not replace P3-D.
''')
    lines.append('''## B1: impairment calibration (same shared underlying frames)

Single additional factor per test frame: PA compression (`pa`), Bernoulli-Gaussian random receiver impulse (`impulse`), data-only common-phase error (`cpe`), or time-domain scalar ADC clipping/quantization (`adc`). `clean` is the reference. Mild and strong severities are frozen in the protocol; random impulsive noise is non-adversarial. `noise_var` for impulse includes *mean* burst power, not labels or realizations.

Representative calibration aggregates, train3 at 10 dB; numbers are synthetic diagnostic means:

'''+mdtable(['Condition','PA out/in power','Burst rate','|CPE| rad','ADC saturation'],axis)+'\n')
    lines.append('''
## B2: model fitting and matched observations

- Five **new** seeds `48101–48105`; **30 new head checkpoints** (`clean`/`mixed` × `raw`/`log_phase`/`rms_log` × five seeds); fixed 600 training steps, batch 64, AdamW learning rate .002 and three-profile train mixture (train3/equal4/long6). Clean vs mixed differ only in availability of one mild physical factor in 60% of mixed training batches.
- Each head has **2,090 learned parameters**, on top of a seed-matched **985-parameter frozen P3-A2 D** pilot channel model (from seeds 47101–47105). The frozen model is unchanged and its copied weights agree SHA256 with P3-A2 source checkpoints.
- All six models per seed share initialization and underlying input stream. Classical estimators also receive the same known random pilot, two public data anchors and declared reported mean noise. Real true channel and unknown data bits are never inference inputs.
- Important exception: no matched-calibration nonlinear-PA inverse/Bussgang receiver was implemented. Hence this benchmark cannot establish how the neural model compares to all strong non-linear-aware signal-processing receivers.
''')
    lines.append('''## B3: held-out results

Each method has five seeds × 768 independent frames = **230,400 scored bits** per profile, SNR and condition. Complete grid: 5 channel profiles × 3 SNR values (0/10/20 dB) × nine conditions × 14 methods = **1,890 aggregate cells**, from **9,450 original per-seed rows**.

### Clean 10-dB baseline (BER; lower is better)

'''+mdtable(['Profile','Frozen P3-A2 D','Mixed/raw head','Mixed/log head','Mixed/RMS head','Fixed threshold without anchor correction','DFT-12'],d8)+'\n')
    lines.append('''
### Eight predeclared *mild* endpoint conditions (equal4/long6 @ 10 dB)

`Best neural` and `best classical` are each per-cell minima selected from **predeclared fixed lists**, included only as an intentionally optimistic neural / conservative classical **descriptive envelope**. These are not deployable policy selectors and are not independent performance claims. `threshold_no_align` is a **fixed** classical method, not itself a tuned oracle.

'''+mdtable(['Profile','Condition','Frozen D','Clean/raw','Mixed/raw','Best neural among six','Best classical among seven'],[x for x in minor if x[1]!='clean'])+'\n')
    lines.append('''
### Strong-impairment descriptive stress cells (equal4/long6 @ 10 dB)

'''+mdtable(['Profile','Condition','Frozen D','Clean/raw','Mixed/raw','Best neural among six','Best classical among seven'],major)+'\n')
    lines.append('''
For both profiles, anchor-based phase correction can make all *aligned* receivers' BER counts unchanged under a pure data-symbol common rotation. However, **in clean conditions noisy-anchor correction often hurts more than it helps**; an unaligned hard-threshold baseline is often better. Strong `cpe` does change the unaligned receiver's BER. This is common-phase rotation, **not** within-symbol CFO or ICI, so it must not be extrapolated to general oscillator errors.

`mixed`-trained neural heads show small, condition-dependent changes relative to the clean heads, usually measured in hundredths of BER percentage points. They cannot be credited with overcoming an impairment when a substantially better fixed DSP estimator exists. Detailed paired training-seed differences are in `p3b_results/paired_neural_vs_frozen.csv` and `primary_exploratory_10db.csv`.

We report **absolute BER** and per-cell impairment penalty against the **same method's clean BER**; differences are directly computable from the paired `ber_summary.csv`. Profile/SNR/seed are not independent RF experiments. No confirmatory p values are claimed and no additional training occurred after viewing the grid.
''')
    lines.append('''### Privileged diagnostic references (added AFTER fixed primary evaluation)

The frozen protocol called for two additional diagnostic references that were absent from the original 14-method table. We computed them separately on **exactly regenerated input frames** without any neural training or re-selection:

- `matched_test_prior_privileged`: a linear-Gaussian posterior using the *actual test profile covariance*, with extra knowledge unavailable to ordinary receivers; no awareness of sample-wise PA, ADC or CPE perturbations.
- `true_channel_privileged`: exact linear multipath transfer function, but **not** an oracle for nonlinear symbol distortion, common-phase drift or saturation.

These are post-run supplement calculations, **not** an amendment to the frozen test list or equal-information win. Their counts are separate in `privileged_reference_by_seed.csv`.

'''+mdtable(['Profile','Matched-test prior LMMSE (clean 10 dB)','True linear H (clean 10 dB)','Fixed threshold unaligned'],privtable)+'\n')
    lines.append('''
## CPU inference costs (median/p95 milliseconds)

Single four-thread PyTorch CPU runtime; 180 post-warmup repetitions in one environment. `full simulated pipeline` includes synthetic in-memory waveform generation/channel application, **not hardware ADC/SDR data transfer, protocol frame handling or a real air interface**. Neither FLOPs nor actual device power consumption are measured; only parameter memory is known.

'''+mdtable(['Path','batch1 median/p95','batch64 median/p95'],timed)+'\n')
    lines.append('''
The `mixed__raw` head has 2,090 trained parameters + 985 frozen backbone parameters = 3,075 total model parameters, storing about 12.3 KB of FP32 weights alone, excluding activations, optimizers and interpreter overhead. The unaligned classical threshold channel estimator has no trainable parameters. Times should not be read as throughput or deployment latency certificates.

## B4: independent integrity audit and exclusions

- `p3b_independent_audit.py` recomputed all **1,890** aggregate BER integer counts from **9,450** original per-seed counts and independently matched all original per-seed CSV files. Every row stores 768 frames, 46,080 scored bits, the number of actual errors, and its exact error ratio.
- **30/30 saved head checkpoint SHA256 values match** their training registry; the model weights reload with correct architecture, seed and frozen protocol SHA. **210/210 re-executed error/bit records exactly match** the originals for the three preselected replay cells across all five seeds. Frozen P3-A2 D weights were reused unchanged.
- All physical B0 tests (12) pass and 1,350 extra privileged-reference diagnostics were calculated on the existing seeded grid. Independent calibration checks confirm the intended relative PA power, impulse event rate, CPE magnitude and ADC saturation severity; no hidden test tuning was permitted.
- The main results come from the **same simulator codebase and CPU platform**, not an independent lab or physical RF receiver. The 60 scored bits within a frame and many frames within a training seed are correlated; **the training seed is the unit of model-replication uncertainty**. This exploratory study does not justify a cross-domain neural-superiority claim.
- B3 prespecified an EVM measure; in fact the neural outputs are arbitrary BCE logits and cannot be interpreted as calibrated physical constellation values. The saved field `scale_aligned_symbol_mse` is therefore an *uncalibrated decision-proxy* and must **not** be called conventional RF EVM. Model-channel MSE is the frozen pilot estimator's performance, not a separately learned channel regressor. True RF EVM, activation memory, FLOPs and hardware power were **not** determined; report as unfulfilled instrumentation rather than fabricated values.
- The nonlinear PA classical baseline family is intentionally incomplete; no direct transceiver measurement has been made. Do not claim neural models outperform non-oracle signal processing just because a single chosen traditional method fails in some cells. Here the classical fixed estimator performs better anyway.

## Conclusion and stage gate

**P3-B completed with a negative result for the tested learned residual heads.** Four isolated ordinary receive-link deviations were reproducibly simulated; mild-vs-strong severities and receiver behavior were independently audited. No neural configuration demonstrated universal gains over the fixed classical comparators; the P3-A2 frozen model did not gain materially from training a lightweight 2,090-parameter data-aware head. Clean and mixed training differences were small and inconsistent. The extra anchors can be a liability in otherwise unrotated data, while they are important for correcting the simulated data-only common-phase error.

This result is conditional on a 32-subcarrier uncoded ideal QPSK experiment with random pilots, 2 public anchors, finite training and simulation biases. It is **not** a theorem that learned nonlinear receivers never help, and does not invalidate the earlier P2-E results. Do not reuse or revise P2-E or P3-A/P3-A2 frozen hypotheses; do not open P3-C without a separately stated protocol.

## Reproduction

```bash
python -m unittest test_p3b_link -v
python p3b_train_eval.py --mode train --seeds 48101 48102 48103 48104 48105 --steps 600 --batch 64 \\
  --old-dir p3a2_checkpoints --ckpt p3b_checkpoints --out p3b_results
python p3b_train_eval.py --mode eval --seeds 48101 48102 48103 48104 48105 --frames 768 \\
  --old-dir p3a2_checkpoints --ckpt p3b_checkpoints --out p3b_results
python p3b_train_eval.py --mode summary --out p3b_results
python p3b_train_eval.py --mode verify --seeds 48101 48102 48103 48104 48105 \\
  --old-dir p3a2_checkpoints --ckpt p3b_checkpoints --out p3b_results
python p3b_privileged_reference.py
python p3b_independent_audit.py
python p3b_timing_extended.py
python p3b_make_report.py
```

`p3a2_checkpoints` are included in the separate complete offline archive; they can be regenerated from the earlier P3-A2 protocol. Newly trained `p3b_checkpoints` are also archived offline, not committed to the lightweight public GitHub repository.
''')
    text='\n'.join(lines)
    (R/'P3B_REPORT.md').write_text(text,encoding='utf-8')
    (R/'P3B_SCOPE_ADDENDUM.md').write_text('''# EM-FloatFormer P3-B scope addendum

**Status:** P3-B0/B1/B2/B3/B4 completed as **exploratory, receive-only synthetic civilian QPSK/OFDM**. P3-C/D/E not started. P2-E/P3-A/P3-A2 frozen experiments remain unchanged.

- Pre-run protocol: `P3B_PROTOCOL.md`, committed before model fitting (`440c4aba5881306e29165727dee03cb7f0dc1550`).
- All five model seeds, 30 newly trained heads, nine clean/single-factor conditions, 0/10/20 dB, five channel profiles and all fixed baseline methods were evaluated. 9,450 raw measurements and 1,890 aggregate table rows; exact checkpoint replay 210/210.
- Main negative finding: learned residual heads do not beat the fixed classical ensemble envelope; no claim about absolute optimality of any estimator or future nonlinear receivers. Finite-seed empirical data do not establish broad scientific laws.
- Added missing privileged oracle-context references afterward, in separate CSV; original frozen primary data were not rewritten or retuned. Inference code uses only observed pilot, payload, public anchors, prior learned during training, and reported total mean noise.
- Actual RF-EVM, FLOP counts, activation memory, real radio latency, ADC electronics, full CFO/ICI, continuous multipath fading, coding and Sionna/3GPP benchmark remain untested. The saved scale-aligned symbol MSE is *not* EVM. PA-nonlinearity-aware classical baselines were not implemented. No new parameter sweeps or extra training are supported by these results.
- Full model weights are in the separate full local research archive; GitHub release may include source, CSV, checksum and plots. Reproduction requires the previously frozen P3-A2 D checkpoint weights or retraining with its public code/protocol.
''')
    print('REPORT',len(text),'chars; 2 plots created; privilege rows',len(full))

if __name__=='__main__':main()
