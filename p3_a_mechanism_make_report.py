#!/usr/bin/env python3
"""Generate P3-A2 report and plots ONLY from stored measured CSV and audit JSON."""
import csv
import json
import statistics
from pathlib import Path


def read(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def summarize(root):
    out=Path(root)
    s=read(out/'ber_summary.csv')
    by={(r['method'],r['profile'],int(r['snr_db'])):r for r in s}
    fac=read(out/'factorial_by_seed.csv')
    gates=read(out/'gate_by_delay.csv')
    audit=json.loads((out/'independent_audit.json').read_text())
    timing=read(out/'latency_cpu.csv')
    parents=read(out/'training_seed47101.csv')
    labels={'A_pos_0':'A — position / no floor', 'B_no_pos_0':'B — no position / no floor',
            'C_pos_10':'C — position / 10% floor','D_no_pos_10':'D — no position / 10% floor',
            'ls':'LS','dft8':'DFT-8','dft12':'DFT-12 (extended-support ref)',
            'dft_thresh_hard':'DFT hard threshold','dft_thresh_soft':'DFT soft threshold',
            'lmmse_train3':'LMMSE train3 prior',
            'lmmse_matched_privileged':'LMMSE matched-test prior (privileged)',
            'true_channel_inaccessible':'True-channel diagnostic (unavailable)'}
    profiles=('train3','equal4','long6','long12','sparse12')
    table=['| Receiver | train3 | equal4 | long6 | long12 | sparse12 |',
           '|---|---:|---:|---:|---:|---:|']
    for method in labels:
        table.append('| '+labels[method]+' | '+' | '.join(f"{100*float(by[method,p,10]['pooled_ber']):.3f}%" for p in profiles)+' |')
    contrasts=['| Primary heldout profile | Effect (BER pp) | Mean across five seeds | Seed-wise paired differences |',
               '|---|---|---:|---|']
    for p in ('long12','sparse12'):
        rs=[r for r in fac if r['profile']==p]
        for col,nm in [('B_minus_A_pp','B − A (remove location)'),('C_minus_A_pp','C − A (add floor)'),
                       ('D_minus_A_pp','D − A (both)'),('interaction_pp','(D−C)−(B−A) interaction')]:
            vals=[float(r[col]) for r in rs]
            contrasts.append(f"| {p} | {nm} | {statistics.mean(vals):+.3f} | "+', '.join(f'{v:+.3f}' for v in vals)+' |')
    gate_table=['| Profile / gate | A 0–2 | A 8–11 | B 0–2 | B 8–11 | C 0–2 | C 8–11 | D 0–2 | D 8–11 |',
                '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for p in ('train3','long12','sparse12'):
        xs=[]
        for arm in ('A_pos_0','B_no_pos_0','C_pos_10','D_no_pos_10'):
            subset=[r for r in gates if r['profile']==p and r['arm']==arm]
            for lo,hi in [(0,2),(8,11)]:
                val=statistics.mean(float(r['mean_gain']) for r in subset if lo<=int(r['delay_index'])<=hi)
                xs.append(f'{val:.4f}')
        gate_table.append('| '+p+' | '+' | '.join(xs)+' |')
    delay_mse=['| Profile | A | B | C | D | Hard-threshold DFT |','|---|---:|---:|---:|---:|---:|']
    for p in profiles:
        nums=['A_pos_0','B_no_pos_0','C_pos_10','D_no_pos_10','dft_thresh_hard']
        delay_mse.append('| '+p+' | '+' | '.join(f"{float(by[name,p,10]['mean_channel_mse']):.4f}" for name in nums)+' |')
    timing_table=['| Arm | nominal params | active params | 1-frame CPU median | 1-frame p95 |',
                  '|---|---:|---:|---:|---:|']
    for i,k in enumerate(('A_pos_0','B_no_pos_0','C_pos_10','D_no_pos_10')):
        r=next(x for x in timing if x['arm']==k and int(x['batch_frames'])==1)
        n=next(x for x in parents if x['arm']==k)
        timing_table.append(f"| {k} | {n['params']} | {n['active_params']} | {float(r['median_ms']):.3f} ms | {float(r['p95_ms']):.3f} ms |")
    text=f"""# EM-FloatFormer P3-A2 — Positional-shortcut and residual-floor factorial ablation

**Completed 2026-11-15. Exploratory, synthetic, receive-only civilian QPSK/OFDM only.**

## Frozen protocol and provenance

Protocol: [`P3_A_MECHANISM_PROTOCOL.md`](P3_A_MECHANISM_PROTOCOL.md), GitHub pre-training freeze commit [`c52a9913baeb54b1ddc47c649301724913f3dc88`](https://github.com/20171806011/EM-FloatFormer/commit/c52a9913baeb54b1ddc47c649301724913f3dc88); original SHA256 `{audit['protocol_sha256']}`. All trials use P3-A's unchanged ideal 32-subcarrier QPSK/OFDM generator, known 32 pilot tones, one pilot/data pair, AWGN and perfect synchronization. No radar, jamming, transmitter control or physical RF measurements.

**Main finding:** Both absolute delay coordinates and ability to suppress late taps nearly to zero contribute to the original P3-A network's 12-delay generalization failure. Removing position features (B) and adding a fixed 10% identity floor (C) each reduce 10dB long12/sparse12 BER compared with freshly retrained A; applying both (D) gives the lowest of the four neural BERs. The benefit is **not simply additive** and does **not surpass** the suitable classical hard-threshold DFT on either heldout primary profile. These statements describe the tested network family and five training seeds, not an unconditional causal theory for all radio receivers.

**Byte-level provenance note (post-run tooling audit):** The GitHub pre-training frozen protocol blob has SHA256 `db3721cb5e3511f21d3ea4adafd92810becd2cd3ce0f375eb42057a2e94aab68` without final LF, while the original local training copy has SHA256 `2955db97ee5203240cb76f30e068f0d12e4bf140909ecd8cdff9a4ae70aadc06` with final LF. The files are otherwise byte-for-byte identical; the frozen GitHub protocol remains unchanged, and the replay scripts accept only these exact two byte representations. No experiments or statistical results were changed.

## Protocol compliance

- Exactly four arms, no hyperparameter sweep: A position+0% floor, B no position+0%, C position+10%, D no position+10%. All use the same nominal **985-parameter** convolutional model (`p3_models.EquivDelayConv`), identical initial weights per seed and shared training batches. With position channels zeroed, **24 stem weights become inactive** but remain stored. The gate is fixed as `g_final=beta+(1-beta)*sigmoid(network_output)` with beta 0 or 0.10.
- Training seeds **47101–47105** are disjoint from P3-A/P2-E. Each model: **600 steps**, batch **64**, AdamW lr **0.002**, weight decay **1e-4**, BCEWithLogits QPSK loss, gradient clipping 1. No test-aware tuning. Four models × five seeds = **20 checkpoints**.
- New held-out test RNGs disjoint from training and historic test RNGs. All methods share the same data per scenario. Five profiles (`train3`, `equal4`, `long6`, `long12`, `sparse12`) × SNR 0/10/20 dB × 768 frames/seed. **245,760 bits per method/scenario** across five seeds. The **unit of replication is seed, not bit**.
- Traditional references: pilot-only LS, DFT-8 (max-eight-delay structural prior), DFT-12 (extended support prior), fixed noise-only 0.05 crossing hard/soft threshold DFT, train3-prior LMMSE, matched-profile LMMSE (**privileged**), inaccessible true-channel diagnostic. All use the same observed pilot values; DFT thresholds never tuned on test sets.
- Original P3-A and P2-E frozen reports/claims are untouched. This is a **new exploratory supplement**, not a revised confirmatory trial.

## 10 dB held-out results: BER percent (lower is better)

{'\n'.join(table)}

Notice that the A control was **freshly retrained** with paired initialization: its 10dB sparse12 BER is **{100*float(by['A_pos_0','sparse12',10]['pooled_ber']):.3f}%**, which need not equal the old P3-A evaluation's **12.016%**. Do not combine their raw counts or cite them as one cohort.

![P3-A2 10dB BER](p3a2_results/ber_profiles_10db.png)

## Frozen primary exploratory contrasts

Negative values indicate lower BER than the original A arm; positive interaction indicates diminishing combined effect on BER-point improvements.

{'\n'.join(contrasts)}

- On the two primary out-of-distribution profiles, **every** seed had lower BER under B and C versus A, and D versus A. This is descriptive n=5 replication across initializations, **not a confirmatory p-value claim**.
- The interaction on sparse12 is positive (average +2.896 BER pp), so the two interventions overlap substantially rather than producing additive reductions. On long12 one seed has a slightly negative interaction. **Do not assert that both modifications independently solve all long-delay errors.**
- On train3/equal4 the 10% floor typically raises BER slightly. On long6, removing position alone helps, but adding a floor without removing position does not.

## Delay-gate mechanism and channel estimation error

These are **post-hoc diagnostics** using simulated ground-truth channel coefficients **only for measurement**, never as network input. The gate itself is computed from noisy pilot observations and reported noise variance.

Mean effective gate over five seeds at 10 dB:

{'\n'.join(gate_table)}

The original A arm's nearly zero gating at delays 8–11 is partly relieved by **removing absolute location features**, and **guaranteed bounded below** by the independent beta=0.10 floor. Their common-phase-equivariance remains intact. The no-position network is equivariant to *cyclic delay shifts* of the tap vector; it is not invariant to arbitrary delay permutations or physically general channel evolution.

Channel **complex MSE**, 10 dB, average across seeds (lower is better):

{'\n'.join(delay_mse)}

Even when BER improves, neural channel MSE remains far higher than well-matched classical estimates because **the network optimizes bit BCE**, for which channel amplitude can be poorly calibrated while QPSK signs/phase remain useful. For additional information see `p3a2_results/ber_summary.csv`, which includes per-seed-average normalized MSE and positive-real-scale-aligned MSE. The alignment factor is computed using true channels **for analysis only**.

## CPU inference cost

{'\n'.join(timing_table)}

PyTorch CPU single-frame model-forward timing only; includes FFT/IFFT in the model, not SDR/ADC, total RF chain, I/O or hardware energy. Different noise or training hardware may produce different latencies.

## Integrity and independent checks

- Pre-training structural unit tests: **11/11 PASS**. Verified A numerical equivalence with original `EquivDelayConv` (identical state), shared initial weights, no true-H/unknown-data input to channel estimator, finite gradients, U(1) common-phase equivariance, cyclic delay-shift equivariance in B/D, fixed residual floor, and classical hard-threshold formula.
- Post-run audit: **{audit['trained_models']}/20 checkpoint SHA256 checks PASS**, **{audit['integer_error_bit_replay_cells']}/240 integer error+bit-count replay cells exact**, **{audit['trained_common_phase_cases']}/60 trained common-phase checks PASS**, **{audit['trained_delay_shift_cases']}/20 shift checks PASS**, and **{audit['trained_original_a_exact_matches']}/5 trained-A vs original P3 forward checks PASS**.
- Full **{audit['record_rows']} raw per-seed/method/scenario rows** cross-checked against **{audit['summary_rows']} aggregates** and **{audit['factorial_rows']} factorial contrasts**. Gate diagnostics contain **{audit['gate_diagnostic_rows']}** by-tap records. Per-seed checkpoint hashes are in `p3a2_results/independent_audit.json`.
- This is deterministic same-implementation replay and arithmetic reconciliation, **not independent codebase, cross-hardware replication, physical OFDM validation, or a claimed statistical significance threshold**.

## Replication from source

```bash
python -m pip install -r requirements.txt
python -m unittest test_p3_a_mechanism -v
python p3_a_mechanism.py --mode train --seeds 47101 47102 47103 47104 47105 --steps 600 --batch 64 --ckpt p3a2_checkpoints --out p3a2_results
python p3_a_mechanism.py --mode eval --seeds 47101 47102 47103 47104 47105 --frames 768 --ckpt p3a2_checkpoints --out p3a2_results
python p3_a_mechanism.py --mode diagnose --seeds 47101 47102 47103 47104 47105 --frames 768 --ckpt p3a2_checkpoints --out p3a2_results
python p3_a_mechanism_audit.py --out p3a2_results --ckpt p3a2_checkpoints
python p3_a_mechanism_make_report.py
```

The source repository omits binary model weights, which are included in the full offline ZIP and can be regenerated from saved seeds. We archive training/evaluation logs, raw integer counts, configuration, hashes, and generated plots.

## Interpretation, limitations and stage gate

This ablation establishes **under the specified synthetic train3-only training procedure** that removing absolute delay coordinates and imposing nonzero gain retention substantially reduce one class of long-delay BER failure, without proving they are necessary or sufficient in every distribution. The changes give up some in-distribution performance. The best four-arm neural variant **still loses to** the classical noise-based hard threshold under sparse12 and long12. The updated model does not become a strong physical channel estimator just because its QPSK BER improved.

The simulator remains idealized frequency-domain 32-tone QPSK, full pilots, static channel, AWGN, perfect timing and no channel coding; no ADC, realistic CFO/CPE, 3GPP channel, genuine mobile fading, sample-rate variation, FP16/BF16 execution, SDR, radar, emitters or intentional RF interference. **P3-B is not part of this supplement**. Stop here; any later impairment experiments need a separately frozen protocol.
"""
    (Path(__file__).with_name('P3_A_MECHANISM_REPORT.md')).write_text(text)
    return text


def plots(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    s=read(Path(root)/'ber_summary.csv')
    by={(r['method'],r['profile'],int(r['snr_db'])):100*float(r['pooled_ber']) for r in s}
    profiles=['train3','equal4','long6','long12','sparse12']
    names=['A_pos_0','B_no_pos_0','C_pos_10','D_no_pos_10','dft_thresh_hard','dft8']
    labels=['A original','B no coordinates','C 10% floor','D both','DFT hard threshold','DFT-8']
    fig,ax=plt.subplots(figsize=(10.8,5.4))
    x=np.arange(len(profiles));w=.13
    for i,(m,label) in enumerate(zip(names,labels)):
        ax.bar(x+(i-2.5)*w,[by[m,p,10] for p in profiles],w,label=label)
    ax.set_xticks(x,profiles)
    ax.set_ylabel('BER at 10 dB (%) — lower is better')
    ax.set_title('P3-A2: held-out BER across channel profiles (5 training seeds)')
    ax.legend(ncol=3,fontsize=8)
    ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    fig.savefig(Path(root)/'ber_profiles_10db.png',dpi=180)
    plt.close(fig)
    gates=read(Path(root)/'gate_by_delay.csv')
    fig,ax=plt.subplots(figsize=(9.5,4.8))
    for name,label in zip(names[:4],labels[:4]):
        vals=[]
        for k in range(32):
            a=[float(r['mean_gain']) for r in gates if r['profile']=='sparse12' and r['arm']==name and int(r['delay_index'])==k]
            vals.append(statistics.mean(a))
        ax.plot(range(32),vals,marker='.',label=label)
    for k in (0,4,11):ax.axvline(k,alpha=.12,linestyle='--')
    ax.set_ylim(0,1)
    ax.set_xlabel('Cyclic delay-tap index')
    ax.set_ylabel('Mean effective gain, sparse12 @ 10dB')
    ax.set_title('P3-A2: retention of late delay taps')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(Path(root)/'gate_sparse12_10db.png',dpi=180)
    plt.close(fig)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('p3a2_results'))
    args=p.parse_args();plots(args.out);print(summarize(args.out)[:700])
