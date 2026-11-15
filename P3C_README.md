# P3-C CPU real-operator mixed-precision audit

This is an exploratory extension to the [EM-FloatFormer](https://github.com/20171806011/EM-FloatFormer) *receive-only civilian QPSK/OFDM* benchmark. It does not perform radio transmission, radar, interference generation, or any actuation.

## What was actually tested

- **Frozen checkpoints**, not newly trained weights: P3-B residual heads (5 seeds × 6 heads) and P3-A2 D backbones (5 seeds); original SHA256 verified from their prior archives.
- **Actual CPU mixed autocast Conv1d**: FP16 and BF16 learned real convolutions, but complex FFT/complex arithmetic remain FP32/complex64 due to missing native CPU FP16/BF16 FFT support.
- Separately quantized received IQ components to FP16/BF16 and reconstructed FP32 before FFT; also evaluated simultaneous IQ quantization + CPU Conv1d autocast.
- Tested coherent numerical signal gain changes and FP64 synthetic 1e-8..1e8 representation/storage round trips.
- All five seeds evaluated on disjoint new held-out frames with full raw integer error counts, replay, independent aggregate checks and CPU timing.

The detailed methods/results are in [`P3C_PROTOCOL.md`](P3C_PROTOCOL.md), [`P3C_REPORT.md`](P3C_REPORT.md), and [`P3C_SCOPE_ADDENDUM.md`](P3C_SCOPE_ADDENDUM.md). Negative results and limitations are mandatory; no claim of fully native FP16/BF16 FFT or real receiver throughput.

## Reproduction

Python 3.11+ with CPU PyTorch 2.10.0 and NumPy/Matplotlib. Start from a directory containing the repository's P3-A/P3-B Python dependencies. P3-C needs the **original frozen weight files**, which are intentionally omitted from the public source-only ZIP but present in the separately provided P3-C full archive.

```
python -m pip install -r requirements_p3c.txt
python -m unittest test_p3c_precision -v
python p3c_precision.py --mode smoke
python p3c_precision.py --mode evaluate --out p3c_results
python p3c_precision.py --mode replay --out p3c_results
python p3c_independent_audit.py
python p3c_make_report.py
```

`p3b_checkpoints/` must contain `model_{clean,mixed}__{raw,log_phase,rms_log}_seed48101..48105.pt`; `p3a2_checkpoints/` must contain `model_D_no_pos_10_seed47101..47105.pt`. These are unmodified copied historical weights; the P3-C stage does not train additional models.

**Limitations:** The CPU autocast kernel can accumulate internally at higher precision even when Conv1d outputs are FP16/BF16; we verify output dtypes, not hidden microarchitecture. No CUDA hardware was available. CPU timing is not energy/throughput of SDR or hardware. Extreme coherent-gain stress changes more than floating-point quantization, so it is not a pure numerical accuracy test. All claims remain limited to the specified simulator and checkpoint family.
