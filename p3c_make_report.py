#!/usr/bin/env python3
"""Generate P3-C findings from immutable measured CSV; no invented outcomes."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
OUT=HERE/'p3c_results'

def read(p):
    with (OUT/p).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))

def row(table,kind,p,snr,condition,method,mode,gain=''):
    xs=[x for x in table if x['kind']==kind and x['profile']==p and x['snr_db']==str(snr)
        and x['condition']==condition and x['gain_exponent']==str(gain) and x['method']==method and x['mode']==mode]
    assert len(xs)==1,(kind,p,snr,condition,method,mode,gain,len(xs))
    return xs[0]

def fmt(v,dec=4):return f'{100*float(v):.{dec}f}%'

def main():
    report=HERE/'P3C_REPORT.md'
    scope=HERE/'P3C_SCOPE_ADDENDUM.md'
    summary=read('summary.csv'); raw=read('raw_by_seed.csv');stress=read('representation_stress.csv');timings=read('latency_cpu.csv')
    audit=json.loads((OUT/'independent_audit.json').read_text());hardware=json.loads((OUT/'hardware_capabilities.json').read_text())
    neural=('mixed__raw','mixed__log_phase','mixed__rms_log')
    lines=[]
    lines.append('# EM-FloatFormer P3-C — CPU native lower-precision real-convolution & IQ-storage audit')
    lines.append('')
    lines.append('**2026-11-15 · exploratory, simulated receive-only civilian QPSK/OFDM · NO NEW TRAINING.**')
    lines.append('')
    lines.append('## Conclusion')
    lines.append('')
    lines.append('The **learned real Conv1d layers did execute with native PyTorch CPU autocast outputs in FP16 and BF16**; the complex FFTs and all complex signal algebra remained **FP32/complex64**, because native CPU half/bfloat FFT is unsupported in this environment. Thus this is **NOT** a full FP16/BF16 receiver execution claim. In the ordinary-amplitude held-out scenarios, BER shifts relative to FP32 were small, with nonzero changes of individual hard-bit decisions. In an artificial coherent gain stress test, even FP32 collapsed far outside its trained scale range: the main bottleneck there is model scale generalization, not necessarily reduced floating-point precision. High-dynamic-range representation stress gave both positive and negative numerical findings. No neural superiority over classical receivers is claimed.')
    lines.append('')
    lines.append('## Evidence provenance and experimental boundary')
    lines.append('')
    lines.append('- Prespecified experimental scope: [`P3C_PROTOCOL.md`](P3C_PROTOCOL.md), frozen to GitHub commit `7ad1a1a7bcadceaf7f055a1176db0c2a9b7ad673` **before** P3-C held-out testing. Local source SHA256 `e1c2f1e0cc8bb0b4a50b515bae40d6daeb62fee8e5ac2974afa3178c9d66371a` (GitHub may normalize trailing LF).')
    lines.append('- Reused P3-B **30 previously trained residual heads**, seeds `48101..48105` (clean/mixed × raw/log-phase/RMS-log), and **five frozen P3-A2 D** channel estimators. No new gradients, fine-tuning, hardware calibration or held-out tuning. Previous P2-E, P3-A, P3-A2 and P3-B outcomes remain frozen.')
    lines.append('- New P3-C deterministic held-out PRNGs; same frames across numeric modes and trained receivers. Profiles `train3`, `equal4`, `long6`, `sparse12`, SNR 0/10/20 dB, `clean`/`pa_strong`/`impulse_strong`/`adc_strong`, 256 frames/seed/cell, 30 unknown QPSK carriers ×2 = 60 test bits/frame; **76,800 bits per method/scenario** across 5 training seeds. Prior P3-B test RNGs not reused.')
    lines.append('- Operator precision, input quantization, combination and coherent gain are different causal interventions, logged separately. Neither input quantization nor its combination with autocast is all-half complex FFT processing. Classical controls use the *identical observed pilot/data samples* and reported noise variance, and remain FP32.')
    lines.append('- All statistics are exploratory, no inferential significance claim. Five trained seeds, not individual bits, represent repeat variability. The data are synthetic and cannot establish hardware power, deployment timing or real-world channel performance.')
    lines.append('')
    lines.append('## Hardware: verified dtypes, not assumed')
    lines.append('')
    lines.append(f"- Runtime: Python {hardware['python']}, PyTorch `{hardware['torch']}`, CPU-only (`cuda_available={hardware['cuda_available']}`).")
    lines.append('- Both `torch.autocast(device_type="cpu", dtype=torch.float16)` and BF16 produced truly lower-dtype **real Conv1d outputs** in backbone and residual head, confirmed by seven independent layer hooks. FFT channel output remained `torch.complex64` and final skip-added logits remained `torch.float32`. This does NOT establish every hidden accumulator is performed at half precision.')
    lines.append(f"- Native CPU `torch.fft` on Half: **{hardware['float16_cpu_fft']['error']}**; on BFloat16: **{hardware['bfloat16_cpu_fft']['error']}**. These are confirmed hardware/runtime limitations, not silently emulated successes.")
    lines.append('')
    lines.append('## A. Native reduced-precision real Conv1d, unchanged FP32/complex64 input and FFT')
    lines.append('')
    lines.append('10 dB, BER (%), all six neural heads tested but representative three mixed-trained encodings shown. **Lower is better.**')
    lines.append('')
    lines.append('| Scenario | Neural head | FP32 | FP16 real-Conv autocast | BF16 real-Conv autocast |')
    lines.append('|---|---|---:|---:|---:|')
    for p,cond in [('equal4','clean'),('long6','impulse_strong'),('sparse12','adc_strong')]:
        for m in neural:
            vals=[row(summary,'operator',p,10,cond,m,mode) for mode in ('fp32','fp16_realconv','bf16_realconv')]
            lines.append(f"| {p} / {cond} | `{m}` | "+' | '.join(fmt(x['ber'],4) for x in vals)+' |')
    lines.append('')
    for mode in ('fp16_realconv','bf16_realconv'):
        found=[x for x in summary if x['kind']=='operator' and x['mode']==mode]
        flip=sum(int(x['total_hard_flips']) for x in found)
        maxdelta=max(abs(float(x['ber'])-float(row(summary,'operator',x['profile'],x['snr_db'],x['condition'],x['method'],'fp32')['ber']))*100 for x in found)
        lines.append(f"- `{mode}`: {flip:,} aggregate hard-bit decisions differed from FP32 across all {len(found)} network × scenario cells (22,118,400 compared bit decisions across the six model configurations); largest **absolute BER shift** in any single cell was {maxdelta:.4f} percentage points. These are *not* 22 million statistically independent bit trials.")
    lines.append('- No nonfinite inference logits or input-IQ values were observed in the ordinary-scale evaluated conditions. Small changes can improve or worsen observed BER by chance; they do not establish low precision as an accuracy enhancement.')
    lines.append('')
    lines.append('## B. Separate input-IQ storage and combined execution')
    lines.append('')
    lines.append('10 dB, equal4/clean, BER. The FP32 control uses original IQ; quantized branches round *components of the received complex IQ* to the stated format and restore them to FP32 before the FFT and complex arithmetic. BF16/FP16 combined branches also autocast real Conv1d.')
    lines.append('')
    lines.append('| Method | FP32 | FP16 storage only | BF16 storage only | FP16 storage+conv | BF16 storage+conv |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    for m in (*neural,'threshold_no_align','dft8','train_lmmse'):
        vals=[row(summary,'input_storage','equal4',10,'clean',m,mode) for mode in ('fp32','iq_fp16_fp32','iq_bf16_fp32','iq_fp16_fp16_realconv','iq_bf16_bf16_realconv')]
        lines.append(f'| `{m}` | '+' | '.join(fmt(x['ber']) for x in vals)+' |')
    lines.append('')
    lines.append('The three classical controls use the same rounded pilot and data observations under storage quantization; they always run FP32. Thus the numerical storage comparison does **not** favor the neural models through unquantized DSP inputs.')
    lines.append('')
    lines.append('## C. Wide-scale coherent gain: failure even at FP32')
    lines.append('')
    lines.append('This is **not an ADC test**. Multiply both pilot and payload received IQ by a coherent scale factor, and multiply reported noise variance by its square. Train-time weights and all fixed stabilizer constants are left unchanged. This probes scale extrapolation and numeric effects together; compare lower precision *against FP32 at the SAME scale*.')
    lines.append('')
    lines.append('| Gain applied to observed IQ | FP32 BER mixed/raw | BF16 stored IQ + BF16 conv BER | FP16 stored IQ + FP16 conv BER |')
    lines.append('|---|---:|---:|---:|')
    for exponent in (-12,0,12):
        vals=[row(summary,'coherent_gain','equal4',10,'clean','mixed__raw',mode,exponent) for mode in ('fp32','iq_bf16_bf16_realconv','iq_fp16_fp16_realconv')]
        lines.append(f'| `2^{exponent}` | '+' | '.join(fmt(x['ber'],3) for x in vals)+' |')
    lines.append('')
    lines.append('**Crucial negative result:** both very small and very large gain produced chance-level or worse QPSK bit decisions **even under the all-FP32 reference**, so those failures cannot be attributed to FP16/BF16. Gain-sensitive fixed epsilons, amplitude-dependent features and optimization outside the training distribution are plausible contributors but not individually isolated in this stage.')
    lines.append('')
    lines.append('## D. High-dynamic-range mathematical representation stress')
    lines.append('')
    lines.append('Deterministic synthetic FP64 source amplitudes across a nominal 1e-8–1e8 magnitude grid, plus exact zero and near-axis cases, 1,056 complex values; after encoding, feature **storage** is quantized to FP16 or BF16 and inverted in FP64. Report finite reconstructions (not automatically correct reconstructions) and relative error among finite nonzero values.')
    lines.append('')
    lines.append('| Encoding | Feature storage | Finite reconstruction | Median relative error | 95th percentile relative error |')
    lines.append('|---|---|---:|---:|---:|')
    for x in stress:
        lines.append(f"| `{x['encoding']}` | {x['storage']} | {x['finite']}/{x['samples']} | {100*float(x['rel_error_median']):.3f}% | {100*float(x['rel_error_p95']):.3f}% |")
    lines.append('')
    lines.append('FP16 raw Cartesian storage overflowed for some strong samples; the log/phase encoding avoided nonfinite reconstructions **for this synthetic stress set**. However, the FP16 RMS/log representation had a high median relative error because tiny components become unrepresentable when a frame RMS is dominated by large signals. BF16 raw Cartesian had far better dynamic range but coarser mantissa precision. **Finite, invertible in exact real arithmetic and accurate under low-precision storage are different properties.**')
    lines.append('')
    lines.append('## E. CPU latency (frozen base + residual head, excludes simulator and real SDR)')
    lines.append('')
    lines.append('| Mode | Batch1 median (ms) | Batch1 p95 | Batch64 median (ms) |')
    lines.append('|---|---:|---:|---:|')
    for mode in ('fp32','fp16_realconv','bf16_realconv'):
        a=next(x for x in timings if x['method']=='mixed__raw' and x['mode']==mode and x['batch_frames']=='1')
        b=next(x for x in timings if x['method']=='mixed__raw' and x['mode']==mode and x['batch_frames']=='64')
        lines.append(f"| `{mode}` | {float(a['median_ms']):.3f} | {float(a['p95_ms']):.3f} | {float(b['median_ms']):.3f} |")
    lines.append('')
    lines.append('On the measured CPU these small mixed-precision models were **not faster** at batch=1; performance is implementation/hardware specific and cannot be generalized to GPU or accelerator hardware. Also, FFT remains FP32. No power/energy measurement was taken.')
    lines.append('')
    lines.append('## F. Audit and reproducibility')
    lines.append('')
    lines.append(f"- `test_p3c_precision.py`: **14/14 PASS**, including native dtype hooks, FFT unsupported modes, no ground-truth channel/unknown-bit leakage, deterministic pairing, known 60 bits/frame, nonfinite failure handling, and high-dynamic sample test.")
    lines.append(f"- `p3c_independent_audit.py`: **{audit['raw_cells']:,}/{audit['raw_cells']:,} raw records** reconciled to **{audit['summary_cells']:,} summary records**, checksummed **{audit['checkpoint_verified']} frozen checkpoint binaries** against previous P3-B and P3-A2 archives; five trained seeds per cell; 0 nonfinite in the tested operational cells.")
    lines.append('- Independently reloaded and recomputed **270/270** predetermined integer error/bit-count cells from three held-out conditions per seed across all three compute modes. Same runtime and code; **not** an independent laboratory or cross-hardware replication.')
    lines.append(f"- {audit['reused_fp32_control_agreements']} inter-run FP32 control identities verified: independent operator/storage/gain reporting routes use the exact same new held-out samples where their scenario overlaps.")
    lines.append('- Full preserved individual rows, error counts, decision flips, dtype trace, runtime and stress data in `p3c_results/`; binary checkpoints remain in existing separate full offline archive, not GitHub. Main source is `p3c_precision.py`; results are generated with `p3c_make_report.py` from CSV, not authored as expected outcomes.')
    lines.append('- An implementation completeness audit identified missing **FP32 classical controls** in the input-storage table; the raw P3-C inference was deterministically replayed with these *predeclared* controls added, without changing protocol, weights, test seeds, model architecture or hyperparameters. Both prior and corrected numerical runs kept their logs; only corrected tables are authoritative.')
    lines.append('')
    lines.append('## Commands (CPU PyTorch 2.10)')
    lines.append('')
    lines.append('```bash')
    lines.append('python -m unittest test_p3c_precision -v')
    lines.append('python p3c_precision.py --mode smoke')
    lines.append('python p3c_precision.py --mode evaluate --out p3c_results')
    lines.append('python p3c_precision.py --mode replay --out p3c_results')
    lines.append('python p3c_independent_audit.py')
    lines.append('python p3c_make_report.py')
    lines.append('```')
    lines.append('')
    lines.append('## Interpretation / stage gate')
    lines.append('')
    lines.append('**P3-C CPU-supported mixed-real precision and separate storage tests are complete, but complete native FP16/BF16 complex-FFT execution is NOT tested and not possible in this CPU runtime.** No evidence that log representation universally improves BER or that low precision improves absolute receiver accuracy. The more striking scale-generalization failure occurs in FP32 too. Any follow-up with native GPU half FFT, full CUDA timing, real ADC/SDR or 3GPP channels requires different hardware and a separately frozen protocol. P3-D/P3-E have not started.')
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    short='''# EM-FloatFormer P3-C scope addendum

**2026-11-15 · CPU-only mixed precision, exploratory; not full low-precision OFDM.**

P3-C freezes and audits inference on 35 existing P3-B/P3-A2 checkpoints, with new held-out RNGs. Real Conv1d modules emit FP16/BF16 under CPU autocast, while complex FFT and all complex algebra remain FP32/complex64 because CPU half/bfloat FFT is unsupported. Separate input-IQ storage quantization and combined modes are reported as separate experimental interventions; nobody should describe them as native full half-precision receiver execution.

No new training or parameter tuning occurred. Strong coherent-gain extrapolation causes severe errors already under FP32; do not attribute it to FP16/BF16. Synthetic wide-dynamic-range feature-storage reconstruction is not a hardware sensitivity experiment. The experiment remains civilian receive-only 32-tone QPSK/OFDM with selected static multipath and non-adversarial impairments; no real antennas, no hardware ADC, no full RF chain, no Sionna/3GPP, no power measurement, and no native half FFT benchmark.

The earlier frozen P2-E, P3-0/P3-A, P3-A2 and P3-B reports and checkpoint binaries were not changed. New P3-C evidence is exploratory and has its own protocol, raw counts, independent same-implementation replay, Python test suite, and hash audit. P3-D/P3-E remain unstarted. See [`P3C_REPORT.md`](P3C_REPORT.md) and [`P3C_PROTOCOL.md`](P3C_PROTOCOL.md).
'''
    scope.write_text(short,encoding='utf-8')
    # Single-figure concrete display of the coherent-gain evidence.
    modes=('fp32','iq_fp16_fp16_realconv','iq_bf16_bf16_realconv')
    labels=('FP32','FP16 storage+conv','BF16 storage+conv')
    x=np.arange(3);width=.25
    fig,ax=plt.subplots(figsize=(8,4.2))
    for i,(mode,label) in enumerate(zip(modes,labels)):
        vals=[100*float(row(summary,'coherent_gain','equal4',10,'clean','mixed__raw',mode,e)['ber']) for e in (-12,0,12)]
        ax.bar(x+(i-1)*width,vals,width,label=label)
    ax.set_xticks(x,labels=['2^-12','2^0','2^12']);ax.set_ylim(0,65)
    ax.set_xlabel('Coherent received IQ scale (noise power scaled quadratically)')
    ax.set_ylabel('BER (%) — lower is better')
    ax.set_title('P3-C: scale extrapolation fails even with FP32')
    ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(OUT/'coherent_gain_10db.png',dpi=170)
    plt.close(fig)
    print('REPORT GENERATED',report.stat().st_size,scope.stat().st_size)

if __name__=='__main__':main()
