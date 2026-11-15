#!/usr/bin/env python3
"""Generate P3-D/E surrogate evidence report from recorded CSV/JSON only."""
import csv,json,math,hashlib,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'p3de_results'
with (DATA/'ber_summary.csv').open() as f: rows=list(csv.DictReader(f))
primary=json.loads((DATA/'primary_endpoints.json').read_text())
audit=json.loads((DATA/'standalone_audit.json').read_text())
lookup={(r['scenario'],int(r['snr_db']),r['method']):r for r in rows}
scenes=['train_A_500ns','test_C_1000ns','test_D_750ns']
names=['A_pos_0','D_no_pos_10','LS','DFT8','DFT12','Threshold','Train-LMMSE','Test-LMMSE','TrueH']

# Measured figure, no custom colors.
fig,ax=plt.subplots(figsize=(10,4.2))
labels=['TDL-A (500ns)','TDL-C (1000ns)','TDL-D (750ns)']
for n in ['A_pos_0','D_no_pos_10','Threshold','Train-LMMSE','Test-LMMSE']:
    vals=[100*float(lookup[sc,10,n]['pooled_ber']) for sc in scenes]
    ax.plot(labels,vals,marker='o',label=n)
ax.set_ylabel('BER (%) — lower is better')
ax.set_title('Frozen new-seed comparison on a static TDL-table surrogate — NOT Sionna')
ax.legend(fontsize=7,ncol=2)
fig.tight_layout();fig.savefig(DATA/'ber_10db.png',dpi=160);plt.close(fig)

report=[]
def add(s=''):report.append(s)
add('# EM-FloatFormer — P3-D / P3-E standards-bridge **surrogate** results')
add('')
add('**2026-11-15 · Completed for the standalone surrogate only · Formal NVIDIA Sionna 2.2 P3-D/P3-E NOT completed.**')
add('')
add('## Evidence scope and provenance')
add('')
add('The independent CPU Python simulation uses the **TDL-A/C/D power-delay parameter values** from NVIDIA Sionna 2.2 (TR 38.901 V19.2) as **input tables**, but **does not import, execute or validate against the Sionna 2.2 package**. Installation of `sionna-no-rt==2.2.0` was attempted but unavailable because PyPI hostname resolution failed in the isolated container. This study cannot be called a 3GPP-certified, Sionna-validated or real RF receiver experiment.')
add('')
add('The [separate frozen protocol](P3DE_SURROGATE_PROTOCOL.md) was committed to the original repository **before training**. Existing P2-E and P3-A/A2/B/C frozen results were not overwritten. New code maintains passive, memory-only, civilian QPSK/OFDM scope; there is no intentional interference, radar, emission control or real equipment.')
add('')
add('### Modelled link')
add('')
add('32 subcarriers spaced by 30kHz, one all-subcarrier known pilot, one static uncoded QPSK data symbol, complex Gaussian thermal noise, analytical fractional-delay frequency response with TDL powers and delays. The 16-sample CP is an **assumption checked for maximum path delay**, not an executed time-domain CP/FFT pipeline. TDL-D has an approximate independent specular LoS component; no Doppler, motion, shadowing, full TDL/CDL polarisation, NR scheduling or Sionna receiver grid.')
add('')
add('10 new training seeds **61001–61010** trained two P3-A2 architecture families (A with location and no gain floor; D with location removed and 10% gain floor), 300 AdamW steps × batch64 per model, alternating TDL-A/TDL-C at 500ns and SNR uniform 0–20dB; **20 independent checkpoint files**. They share initialization and exact training frames within each seed. Testing uses disjoint seeded draws, 2048 frames per seed × 64 bits = **1,310,720 bits per method/scenario/SNR**. Independent training seeds are the replication units, not individual bits.')
add('')
add('## Full held-out 10dB results')
add('')
add('| Receiver | TDL-A 500ns (train family) | TDL-C 1000ns (unseen spread) | TDL-D 750ns (unseen LoS) |')
add('|---|---:|---:|---:|')
for n in names:
    values=[float(lookup[(sc,10,n)]['pooled_ber'])*100 for sc in scenes]
    suffix=' **(privileged)**' if n=='Test-LMMSE' else ' **(true H, inaccessible)**' if n=='TrueH' else ''
    add('| `'+n+'`'+suffix+' | '+' | '.join(f'{v:.3f}%' for v in values)+' |')
add('')
add('`Train-LMMSE` uses an analytic covariance computed solely from the **two known training distributions**, which gives it an explicit distributional model; `Test-LMMSE` knows the held-out test covariance and is privileged. DFT8 and DFT12 assume integer-delay support, which is not matched exactly by the fractional-delay profiles. No method accesses instantaneous H except the labelled unreachable diagnostic.')
add('')
add('![Recorded surrogate BER](p3de_results/ber_10db.png)')
add('')
add('## Frozen surrogate primary comparisons, 10dB')
add('')
add('Prespecified delta = `BER(Threshold) - BER(D)` (percentage points). Positive favors D; two-sided exhaustive 10-seed sign-flip tests, Holm-adjusted for two endpoints.')
add('')
add('| Scenario | Mean difference (pp) | Two-sided descriptive t(9) 95% interval (pp) | Exact two-sided p | Holm p |')
add('|---|---:|---|---:|---:|')
for r in primary:
    add(f"| {r['scenario']} | {r['mean_delta_pp']:+.4f} | [{r['ci95_t_pp'][0]:+.4f}, {r['ci95_t_pp'][1]:+.4f}] | {r['exact_twosided_p']:.6f} | {r['holm_p']:.6f} |")
add('')
for r in primary:
    add(f"Seed differences for `{r['scenario']}` (pp): "+', '.join(f'{x:+.4f}' for x in r['seed_delta_pp'])+'.')
add('')
add('**Interpretation:** D improves relative to the *fixed hard-threshold* estimator on these specific static table-driven scenarios. This is **not** a general neural victory: on both held-out 10dB profiles the **nonprivileged training-mixture LMMSE matches or improves on D**, and model A has lower pooled BER than D. The static surrogate cannot establish that any of these differences persist in Sionna 2.2 dynamic fading, realistic pilot patterns, or hardware.')
add('')
add('## Other SNRs and interpretation')
add('')
add('| Profile | SNR | A | D | Threshold | Train-LMMSE | Test-LMMSE (privileged) |')
add('|---|---:|---:|---:|---:|---:|---:|')
for sc in scenes:
  for snr in (0,20):
    vs=[float(lookup[(sc,snr,n)]['pooled_ber'])*100 for n in ('A_pos_0','D_no_pos_10','Threshold','Train-LMMSE','Test-LMMSE')]
    add('| '+sc+f' | {snr}dB | '+' | '.join(f'{v:.3f}%' for v in vs)+' |')
add('')
add('Reported complex channel MSE must not be interpreted as an accurate physical channel estimate if the neural model is trained only with bit BCE: at 10dB, D has high channel MSE despite competitive QPSK BER (channel magnitude affects MSE but not positive-real-scale QPSK sign decisions). Full channel-MSE columns are in raw CSV.')
add('')
add('## CPU model-forward latency, exact tested process')
add('')
add('| Method | Single frame median | Batch64 median |')
add('|---|---:|---:|')
for r in csv.DictReader((DATA/'latency_cpu.csv').open()):
    if int(r['batch_frames'])==1:
        other=next(z for z in csv.DictReader((DATA/'latency_cpu.csv').open()) if z['method']==r['method'] and int(z['batch_frames'])==64)
        add(f"| {r['method']} | {float(r['median_ms']):.4f}ms | {float(other['median_ms']):.4f}ms |")
add('')
add('These timings include only PyTorch CPU model/channel-estimate forward calls, not simulator generation, complete OFDM receive pipeline, transport, host/device transfer, energy or GPU inference. Model and baseline pipelines perform different arithmetic. A lower BER alone is not evidence of a better engineering choice.')
add('')
add('## Independent audit')
add('')
add(f"- Surrogate profile mathematical checks passed: normalized powers, positive-semidefinite Hermitian covariance, CP-path-envelope, cyclic integer-delay convolution identity. Independent NumPy analytic covariance cross-check max absolute error **{audit['max_numpy_covariance_abs_error']:.3g}**; 10dB observed pilot-noise variance **{audit['measured_pilot_noise_var_at_10db']:.5f}** (target 0.1).")
add(f"- **{audit['training_checkpoints_sha_verified']}/20** checkpoint SHA256 files matched, **{audit['per_seed_endpoint_integer_replay_cells']}** exact endpoint integer error+bit counts reproduced for all 10 seeds, **{audit['trained_model_phase_cases']}** complex-common-phase numerical checks passed; 810 raw per-seed BER records reconciled to 81 summaries. This is a same-runtime replay, not independent hardware or Sionna replication.")
add('- Training and held-out data RNGs are disjoint; within-scenario received observations are paired among neural and classical methods. The tables and p-values were calculated from saved raw integer error counts, with no post-test hyperparameter updates.')
add('')
add('## Formal P3-D / P3-E status and explicit blockers')
add('')
add('**Unfinished (do not mark complete):** true Sionna 2.2 execution; validated 3GPP TDL-A/C/D and CDL time evolution with Doppler; Sionna `LMMSEChannelEstimator`; standard multi-symbol resource-grid pilot patterns and coding; complete receiver-link timing/GPU benchmark. The failed `pip install sionna-no-rt==2.2.0` cannot be resolved in this internet-isolated runtime, so a Sionna-backed P3-E ten-seed confirmation does not exist. The above 10-seed tests confirm only the **narrow standalone surrogate endpoints**; they are separate from the proposed official P3-E.')
add('')
add('If a compatible Sionna environment is available, first run a new Sionna-specific pilot/data channel and estimator-equivalence audit, freeze separate endpoints/compute plan before inspecting held-out outcomes, and then undertake formal P3-D and P3-E with new seeds. Do not recycle these surrogate test seeds as a formal confirmation set.')
add('')
add('## Reproduction')
add('')
add('```bash')
add('python -m pip install torch numpy scipy matplotlib')
add('python p3de_profile_bridge.py')
add('python p3de_experiment.py --mode smoke')
add('python p3de_experiment.py --mode train --out p3de_results')
add('python p3de_experiment.py --mode eval --out p3de_results')
add('python p3de_experiment.py --mode finalize --out p3de_results')
add('python p3de_independent_audit.py')
add('python p3de_make_report.py')
add('```')
add('')
add('No arbitrary changes to models, steps, profiles or fixed classical thresholds were performed after frozen surrogate protocol publication. Original P1/P2/P3-A/B/C GitHub history remains unchanged.')
(ROOT/'P3DE_SURROGATE_REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
print('Report generated from recorded data:',len(report),'lines')
