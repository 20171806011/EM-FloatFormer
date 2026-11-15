# EM-FloatFormer P1-E — New-seed replication and computational cost audit

**2026-11-15 · Completed · CPU PyTorch · Synthetic receive-only civilian QPSK/OFDM**

## Summary

The two P1-D out-of-distribution architectural comparisons were tested with **five entirely new training seeds** (404, 505, 606, 707, 808), **20 new checkpoints** (CNN/Transformer × two encodings), and **2048 newly generated test frames per seed/scenario**. At 10 dB, CNN/log-amplitude/phase had lower BER than Transformer/log-amplitude/phase on four-tap and six-tap tests for **all five seeds**. The corrected exact two-sided sign-flip test, however, returned **Holm-adjusted p=0.125** for both endpoints, which does not pass the 0.05 decision threshold. It is not justified to claim confirmatory statistical significance. Well-calibrated matched-prior LMMSE remained better in all three reported 10-dB channels but uses extra distributional prior knowledge outside the equal-information model comparison.

## Preregistered protocol and statistical design

- Prespecified protocol: [`P1E_PROTOCOL.md`](P1E_PROTOCOL.md), written before test evaluation. Two primary comparisons: CNN/log vs Transformer/log on equal-four-tap and six-tap channel profiles, both at 10 dB.
- **P1-C training code and settings unchanged**: 32 subcarriers, known all-subcarrier pilot, static multipath within pilot/data, 750 optimization steps, 64 frames/batch, learning rate 0.002, train3 profile `[0.72,0.20,0.08]`, random training SNR 0–20 dB. 20 new checkpoints; no test-aware tuning.
- Each seed and each of three profiles (`train3`, `equal4`, `long6`) × three SNRs (0/10/20 dB): 2048 new frames, 64 bits/frame = 131072 bits per seed/scenario = **655360 pooled bits** per receiver/scenario across five seeds. All models and conventional references get identical within-scenario data. Test seeds are different from P1-C/P1-D.
- Statistical sampling unit is **training seed**, not individual correlated bit. For each primary endpoint, delta = BER(Transformer/log) − BER(CNN/log), positive favors CNN. Report two-sided exact enumeration of all `2^5=32` seed sign assignments, Holm-adjusted p values across two comparisons, and descriptive two-sided t(4) confidence intervals. Secondary comparisons are exploratory.

## Held-out BER results (10 dB, percentage; lower better)

| Receiver | Train3 | Equal4 (OOD) | Long6 (OOD) |
|---|---:|---:|---:|
| CNN + raw IQ | 5.212% | 6.793% | 9.059% |
| CNN + log amplitude/phase | 5.321% | 6.597% | 8.564% |
| Transformer + raw IQ | 5.308% | 7.922% | 10.021% |
| Transformer + log amplitude/phase | 5.251% | 7.410% | 9.557% |
| Pilot LS | 8.075% | 8.061% | 8.071% |
| DFT, fixed 3-tap support | 4.778% | 14.358% | 13.637% |
| LMMSE, train3 prior | 4.776% | 14.362% | 13.637% |
| LMMSE, **matched test-profile prior** | 4.776% | 4.909% | 5.131% |
| True-channel reference (unavailable) | 4.356% | 4.380% | 4.345% |

The matched-test-prior LMMSE is not information-matched to models (it receives the actual test profile as a statistical prior); it is an informative classical reference, **not** proof of intrinsic superiority when information differs. In the six-tap profile, even the best tested CNN remains worse than pilot-only LS. As expected, the true-channel reference is unavailable to practical systems.

## Primary test results

| Prespecified endpoint | Mean delta, BER percentage points | Per-seed deltas, pp | Two-sided 95% t interval, pp | Exact two-sided sign-flip p | Holm-adjusted p |
|---|---:|---|---|---:|---:|
| equal4 @ 10dB | +0.813 | +0.659, +0.794, +1.070, +0.747, +0.796 | [+0.622, +1.004] | 0.0625 | 0.1250 |
| long6 @ 10dB | +0.993 | +0.893, +1.049, +1.162, +0.753, +1.108 | [+0.785, +1.201] | 0.0625 | 0.1250 |

With only five training seeds, a two-sided exact sign-flip test cannot have a p value below `2/32=0.0625`, even when all five deltas share a sign. The 95% **parametric** t intervals exclude zero, while the conservative exact nonparametric tests do not pass 0.05 after Holm correction. These tests use different assumptions; do not report the parametric interval alone as confirmatory proof. The endpoints were motivated by earlier P1-D results, which were not independent preregistered discoveries. This P1-E evaluation is a fresh-seed replication, not an independently implemented or physical-radio replication.

## Exploratory encoding comparisons

| Profile, 10dB | CNN raw IQ BER | CNN log/phase BER | Difference (raw − log) |
|---|---:|---:|---:|
| train3 | 5.212% | 5.321% | -0.109 pp |
| equal4 | 6.793% | 6.597% | +0.196 pp |
| long6 | 9.059% | 8.564% | +0.495 pp |

Log/phase helps the CNN in the two tested out-of-distribution profiles, but hurts it on the in-distribution profile. It is not universally better, and the network only accepts normalized/simulated baseband I/Q, not RF measurements across extreme ADC dynamic ranges. Other SNR (0/20 dB) and model cells appear in [`p1e_results/ber_summary.csv`](p1e_results/ber_summary.csv).

## Measured CPU inference cost

**300 timing repetitions, 20 warm-ups, torch.set_num_threads(4); includes neural feature-extraction forward path but not simulator, SDR hardware, transport, FFT or decode protocol.**

| Model | Parameters | Batch 1 median / p95 (ms) | Batch 128 median / p95 (ms) |
|---|---:|---:|---:|
| circular_cnn__raw_iq_magnitude | 38,018 | 0.317 / 0.507 | 3.720 / 6.588 |
| circular_cnn__logamp_phase | 38,018 | 0.341 / 0.586 | 4.063 / 5.293 |
| transformer__raw_iq_magnitude | 38,594 | 0.411 / 0.725 | 5.164 / 8.673 |
| transformer__logamp_phase | 38,594 | 0.408 / 0.533 | 5.104 / 6.998 |

Performance on one CPU is hardware-, thread- and batch-dependent. These numbers are not evidence of an operational sub-millisecond over-the-air receiver.

## Integrity and reproducibility

- **Exact replay**: reloaded the five-seed training checkpoint for seed 404 and recomputed 9 scenario cells; **81/81 recorded error-count cells matched exactly**. This is repeatability on the same CPU/PyTorch environment, not cross-platform independence.
- Channel/noise simulator check: generated pilot thermal-noise variance ≈0.10043 for target 0.1 at 10 dB; theoretical covariance is Hermitian; **independent Fourier-domain Wiener shrinkage expression** numerically matches the matrix-solve LMMSE implementation (max absolute deviation below 5e-5 in tested cases).
- Source and model metadata: [`p1e_results/run_config.json`](p1e_results/run_config.json), [`p1e_results/checkpoint_sha256.json`](p1e_results/checkpoint_sha256.json). Saved raw counts: [`p1e_results/ber_by_seed.csv`](p1e_results/ber_by_seed.csv). Prespecified differences: [`p1e_results/primary_endpoints.json`](p1e_results/primary_endpoints.json).
- Model weights are packaged in the *full archive* but omitted from the lightweight public GitHub tree to keep the public repository small. P1-C models are also not modified.

## Reproduction

```bash
python p1c_architecture.py --steps 750 --batch-size 64 --eval-frames 2000 \
  --seeds 404 505 606 707 808 --arches circular_cnn transformer \
  --encodings raw_iq_magnitude logamp_phase --out p1e_train
python p1e_replication.py --smoke
python p1e_replication.py --checkpoints p1e_train --out p1e_results --frames 2048
python p1e_replication.py --verify --checkpoints p1e_train --out p1e_results --frames 2048
python p1e_make_report.py
```

**Limits:** Idealized, frequency-domain, perfect synchronization, 32-subcarrier full-pilot uncoded OFDM, synthetic Gaussian thermal noise and static channel, exactly known noise variance in unperturbed scenarios. No fading within each pilot-data pair, no real SDR data, no blind channel estimation, no sample-rate changes, no physical hardware/energy efficiency, no million-sample input, no active electronic warfare or transmitter-control algorithms. P1-E reports a narrow repeatable benchmarking result, not a newly discovered RF principle, a deployed radio receiver, or evidence of a universal superior architecture.

**Phase-1 status: P1-A through P1-E complete.** Future research may extend the synthetic model to richer benign communications settings, but this stage ends here.
