# EM-FloatFormer P3-C scope addendum

**2026-11-15 · CPU-only mixed precision, exploratory; not full low-precision OFDM.**

P3-C freezes and audits inference on 35 existing P3-B/P3-A2 checkpoints, with new held-out RNGs. Real Conv1d modules emit FP16/BF16 under CPU autocast, while complex FFT and all complex algebra remain FP32/complex64 because CPU half/bfloat FFT is unsupported. Separate input-IQ storage quantization and combined modes are reported as separate experimental interventions; nobody should describe them as native full half-precision receiver execution.

No new training or parameter tuning occurred. Strong coherent-gain extrapolation causes severe errors already under FP32; do not attribute it to FP16/BF16. Synthetic wide-dynamic-range feature-storage reconstruction is not a hardware sensitivity experiment. The experiment remains civilian receive-only 32-tone QPSK/OFDM with selected static multipath and non-adversarial impairments; no real antennas, no hardware ADC, no full RF chain, no Sionna/3GPP, no power measurement, and no native half FFT benchmark.

The earlier frozen P2-E, P3-0/P3-A, P3-A2 and P3-B reports and checkpoint binaries were not changed. New P3-C evidence is exploratory and has its own protocol, raw counts, independent same-implementation replay, Python test suite, and hash audit. P3-D/P3-E remain unstarted. See [`P3C_REPORT.md`](P3C_REPORT.md) and [`P3C_PROTOCOL.md`](P3C_PROTOCOL.md).
