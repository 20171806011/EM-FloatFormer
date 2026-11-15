# EM-FloatFormer P3-C — CPU native lower-precision real-convolution & IQ-storage audit

**2026-11-15 · exploratory, simulated receive-only civilian QPSK/OFDM · NO NEW TRAINING.**

## Conclusion

The **learned real Conv1d layers did execute with native PyTorch CPU autocast outputs in FP16 and BF16**; the complex FFTs and all complex signal algebra remained **FP32/complex64**, because native CPU half/bfloat FFT is unsupported in this environment. Thus this is **NOT** a full FP16/BF16 receiver execution claim. In the ordinary-amplitude held-out scenarios, BER shifts relative to FP32 were small, with nonzero changes of individual hard-bit decisions. In an artificial coherent gain stress test, even FP32 collapsed far outside its trained scale range: the main bottleneck there is model scale generalization, not necessarily reduced floating-point precision. High-dynamic-range representation stress gave both positive and negative numerical findings. No neural superiority over classical receivers is claimed.

## Evidence provenance and experimental boundary

- Prespecified experimental scope: [`P3C_PROTOCOL.md`](P3C_PROTOCOL.md), frozen to GitHub commit `7ad1a1a7bcadceaf7f055a1176db0c2a9b7ad673` **before** P3-C held-out testing. Local source SHA256 `e1c2f1e0cc8bb0b4a50b515bae40d6daeb62fee8e5ac2974afa3178c9d66371a` (GitHub may normalize trailing LF).
- Reused P3-B **30 previously trained residual heads**, seeds `48101..48105` (clean/mixed × raw/log-phase/RMS-log), and **five frozen P3-A2 D** channel estimators. No new gradients, fine-tuning, hardware calibration or held-out tuning. Previous P2-E, P3-A, P3-A2 and P3-B outcomes remain frozen.
- New P3-C deterministic held-out PRNGs; same frames across numeric modes and trained receivers. Profiles `train3`, `equal4`, `long6`, `sparse12`, SNR 0/10/20 dB, `clean`/`pa_strong`/`impulse_strong`/`adc_strong`, 256 frames/seed/cell, 30 unknown QPSK carriers ×2 = 60 test bits/frame; **76,800 bits per method/scenario** across 5 training seeds. Prior P3-B test RNGs not reused.
- Operator precision, input quantization, combination and coherent gain are different causal interventions, logged separately. Neither input quantization nor its combination with autocast is all-half complex FFT processing. Classical controls use the *identical observed pilot/data samples* and reported noise variance, and remain FP32.
- All statistics are exploratory, no inferential significance claim. Five trained seeds, not individual bits, represent repeat variability. The data are synthetic and cannot establish hardware power, deployment timing or real-world channel performance.

## Hardware: verified dtypes, not assumed

- Runtime: Python 3.13.5, PyTorch `2.10.0+cpu`, CPU-only (`cuda_available=False`).
- Both `torch.autocast(device_type="cpu", dtype=torch.float16)` and BF16 produced truly lower-dtype **real Conv1d outputs** in backbone and residual head, confirmed by seven independent layer hooks. FFT channel output remained `torch.complex64` and final skip-added logits remained `torch.float32`. This does NOT establish every hidden accumulator is performed at half precision.
- Native CPU `torch.fft` on Half: **RuntimeError: Unsupported dtype Half**; on BFloat16: **RuntimeError: Unsupported dtype BFloat16**. These are confirmed hardware/runtime limitations, not silently emulated successes.

## A. Native reduced-precision real Conv1d, unchanged FP32/complex64 input and FFT

10 dB, BER (%), all six neural heads tested but representative three mixed-trained encodings shown. **Lower is better.**

| Scenario | Neural head | FP32 | FP16 real-Conv autocast | BF16 real-Conv autocast |
|---|---|---:|---:|---:|
| equal4 / clean | `mixed__raw` | 7.0182% | 7.0143% | 7.0169% |
| equal4 / clean | `mixed__log_phase` | 7.0612% | 7.0612% | 7.0508% |
| equal4 / clean | `mixed__rms_log` | 7.0417% | 7.0469% | 7.0456% |
| long6 / impulse_strong | `mixed__raw` | 20.4479% | 20.4414% | 20.4479% |
| long6 / impulse_strong | `mixed__log_phase` | 20.4648% | 20.4622% | 20.4622% |
| long6 / impulse_strong | `mixed__rms_log` | 20.4531% | 20.4570% | 20.4883% |
| sparse12 / adc_strong | `mixed__raw` | 11.2135% | 11.2148% | 11.1927% |
| sparse12 / adc_strong | `mixed__log_phase` | 11.2057% | 11.2109% | 11.2057% |
| sparse12 / adc_strong | `mixed__rms_log` | 11.2266% | 11.2227% | 11.1836% |

- `fp16_realconv`: 2,368 aggregate hard-bit decisions differed from FP32 across all 288 network × scenario cells (22,118,400 compared bit decisions across the six model configurations); largest **absolute BER shift** in any single cell was 0.0130 percentage points. These are *not* 22 million statistically independent bit trials.
- `bf16_realconv`: 19,127 aggregate hard-bit decisions differed from FP32 across all 288 network × scenario cells (22,118,400 compared bit decisions across the six model configurations); largest **absolute BER shift** in any single cell was 0.0612 percentage points. These are *not* 22 million statistically independent bit trials.
- No nonfinite inference logits or input-IQ values were observed in the ordinary-scale evaluated conditions. Small changes can improve or worsen observed BER by chance; they do not establish low precision as an accuracy enhancement.

## B. Separate input-IQ storage and combined execution

10 dB, equal4/clean, BER. The FP32 control uses original IQ; quantized branches round *components of the received complex IQ* to the stated format and restore them to FP32 before the FFT and complex arithmetic. BF16/FP16 combined branches also autocast real Conv1d.

| Method | FP32 | FP16 storage only | BF16 storage only | FP16 storage+conv | BF16 storage+conv |
|---|---:|---:|---:|---:|---:|
| `mixed__raw` | 7.0182% | 7.0195% | 7.0156% | 7.0078% | 7.0143% |
| `mixed__log_phase` | 7.0612% | 7.0612% | 7.0534% | 7.0560% | 7.0365% |
| `mixed__rms_log` | 7.0417% | 7.0404% | 7.0326% | 7.0430% | 7.0352% |
| `threshold_no_align` | 4.9870% | 4.9857% | 4.9870% | 4.9857% | 4.9870% |
| `dft8` | 6.5247% | 6.5260% | 6.5208% | 6.5260% | 6.5208% |
| `train_lmmse` | 6.1549% | 6.1549% | 6.1576% | 6.1549% | 6.1576% |

The three classical controls use the same rounded pilot and data observations under storage quantization; they always run FP32. Thus the numerical storage comparison does **not** favor the neural models through unquantized DSP inputs.

## C. Wide-scale coherent gain: failure even at FP32

This is **not an ADC test**. Multiply both pilot and payload received IQ by a coherent scale factor, and multiply reported noise variance by its square. Train-time weights and all fixed stabilizer constants are left unchanged. This probes scale extrapolation and numeric effects together; compare lower precision *against FP32 at the SAME scale*.

| Gain applied to observed IQ | FP32 BER mixed/raw | BF16 stored IQ + BF16 conv BER | FP16 stored IQ + FP16 conv BER |
|---|---:|---:|---:|
| `2^-12` | 49.961% | 49.986% | 50.009% |
| `2^0` | 7.018% | 7.014% | 7.008% |
| `2^12` | 54.887% | 54.875% | 54.891% |

**Crucial negative result:** both very small and very large gain produced chance-level or worse QPSK bit decisions **even under the all-FP32 reference**, so those failures cannot be attributed to FP16/BF16. Gain-sensitive fixed epsilons, amplitude-dependent features and optimization outside the training distribution are plausible contributors but not individually isolated in this stage.

## D. High-dynamic-range mathematical representation stress

Deterministic synthetic FP64 source amplitudes across a nominal 1e-8–1e8 magnitude grid, plus exact zero and near-axis cases, 1,056 complex values; after encoding, feature **storage** is quantized to FP16 or BF16 and inverted in FP64. Report finite reconstructions (not automatically correct reconstructions) and relative error among finite nonzero values.

| Encoding | Feature storage | Finite reconstruction | Median relative error | 95th percentile relative error |
|---|---|---:|---:|---:|
| `raw_cartesian` | float16 | 825/1056 | 0.024% | 100.000% |
| `raw_cartesian` | bfloat16 | 1056/1056 | 0.142% | 0.284% |
| `log_phase` | float16 | 1056/1056 | 0.082% | 0.602% |
| `log_phase` | bfloat16 | 1056/1056 | 0.635% | 5.005% |
| `rms_log` | float16 | 1056/1056 | 49.090% | 100.000% |
| `rms_log` | bfloat16 | 1056/1056 | 2.343% | 5.484% |

FP16 raw Cartesian storage overflowed for some strong samples; the log/phase encoding avoided nonfinite reconstructions **for this synthetic stress set**. However, the FP16 RMS/log representation had a high median relative error because tiny components become unrepresentable when a frame RMS is dominated by large signals. BF16 raw Cartesian had far better dynamic range but coarser mantissa precision. **Finite, invertible in exact real arithmetic and accurate under low-precision storage are different properties.**

## E. CPU latency (frozen base + residual head, excludes simulator and real SDR)

| Mode | Batch1 median (ms) | Batch1 p95 | Batch64 median (ms) |
|---|---:|---:|---:|
| `fp32` | 0.455 | 0.664 | 1.253 |
| `fp16_realconv` | 0.507 | 1.530 | 1.730 |
| `bf16_realconv` | 0.855 | 1.103 | 1.330 |

On the measured CPU these small mixed-precision models were **not faster** at batch=1; performance is implementation/hardware specific and cannot be generalized to GPU or accelerator hardware. Also, FFT remains FP32. No power/energy measurement was taken.

## F. Audit and reproducibility

- `test_p3c_precision.py`: **14/14 PASS**, including native dtype hooks, FFT unsupported modes, no ground-truth channel/unknown-bit leakage, deterministic pairing, known 60 bits/frame, nonfinite failure handling, and high-dynamic sample test.
- `p3c_independent_audit.py`: **6,525/6,525 raw records** reconciled to **1,305 summary records**, checksummed **35 frozen checkpoint binaries** against previous P3-B and P3-A2 archives; five trained seeds per cell; 0 nonfinite in the tested operational cells.
- Independently reloaded and recomputed **270/270** predetermined integer error/bit-count cells from three held-out conditions per seed across all three compute modes. Same runtime and code; **not** an independent laboratory or cross-hardware replication.
- 15 inter-run FP32 control identities verified: independent operator/storage/gain reporting routes use the exact same new held-out samples where their scenario overlaps.
- Full preserved individual rows, error counts, decision flips, dtype trace, runtime and stress data in `p3c_results/`; binary checkpoints remain in existing separate full offline archive, not GitHub. Main source is `p3c_precision.py`; results are generated with `p3c_make_report.py` from CSV, not authored as expected outcomes.
- An implementation completeness audit identified missing **FP32 classical controls** in the input-storage table; the raw P3-C inference was deterministically replayed with these *predeclared* controls added, without changing protocol, weights, test seeds, model architecture or hyperparameters. Both prior and corrected numerical runs kept their logs; only corrected tables are authoritative.

## Commands (CPU PyTorch 2.10)

```bash
python -m unittest test_p3c_precision -v
python p3c_precision.py --mode smoke
python p3c_precision.py --mode evaluate --out p3c_results
python p3c_precision.py --mode replay --out p3c_results
python p3c_independent_audit.py
python p3c_make_report.py
```

## Interpretation / stage gate

**P3-C CPU-supported mixed-real precision and separate storage tests are complete, but complete native FP16/BF16 complex-FFT execution is NOT tested and not possible in this CPU runtime.** No evidence that log representation universally improves BER or that low precision improves absolute receiver accuracy. The more striking scale-generalization failure occurs in FP32 too. Any follow-up with native GPU half FFT, full CUDA timing, real ADC/SDR or 3GPP channels requires different hardware and a separately frozen protocol. P3-D/P3-E have not started.
