# P8-A/B • Continuous propagation state across two OFDM frames

Parent study: P7 at `4dab1ac9ee15b4f1a7f78ae05486c55a2dc3bd72`. This is a classical mechanism experiment with zero fitted parameters. A and B each have 20 symbols, 32 subcarriers, a 16-sample cyclic prefix, Gray 16-QAM and the P4 systematic (128,64) LDPC. The pilot schedule, powers and Doppler ranges follow P5 R2. P8 uses independent NumPy draws and a continuous 40-symbol channel realization. Units of time are 32 samples. SNR is 9 dB per unit transmit symbol.

## Simulator and public information

Three rays at public delays (0, 2, 4) have powers (.75, .17, .08). Each signed Doppler starts and ends at independently drawn magnitudes uniform on [.045, .080], with one random sign per path throughout both frames. Doppler varies linearly over the full pair. Complex gain has constant magnitude and evolves in phase sample by sample. The dominant magnitude is fixed, with uniform initial phase. The other initial gains are circular complex Gaussian. The boundary advances the same ray trajectory through the CP interval. A separate independent channel, payload and noise generate the donor frame.

Every A has 64 transmitted QPSK pilots on even subcarriers of rows 0, 9, 10 and 19. Every B physically transmits 16 diagnostic pilots on row 19. Their received values stay hidden through estimation, prediction and decoding. All methods know pilot symbols, masks, AWGN power, the sample clock and the common statistical path model. Only the evaluator sees true payloads, clean target samples and held-out noisy observations.

| B budget | Acquisition rows | Acquisition pilots | Held-out pilots | Total transmitted pilots | Payload |
| --- | --- | ---: | ---: | ---: | --- |
| Dense | 0, 9, 10 | 48 | 16 | 64 | 18 LDPC words |
| Sparse | 0 | 16 | 16 | 32 | Same 18 words in the same cells, plus one independent LDPC word |

Within a budget, the three methods use exactly the same B array. Across budgets, the propagation path, noise and common payload are paired. The transmit waveform changes because 32 released pilot positions carry the additional word. Their receive samples can change other bins through ICI. The last-row clean signal and noise are identical across budgets under this CP-guarded model. No B acquisition pilot follows the prediction target. Block decoding uses the available frame pilots and can smooth earlier B payload.

## Classical state and controls

The bank uses 128 Doppler/chirp nodes per delay. These are the Cartesian product of eight equally spaced positive start/end magnitudes (including endpoints) and both common signs. Each node carries a complex Gaussian gain mean, variance and log probability. A comb LS projection gives one noisy tap estimate per ray and pilot row. For signature u, observation z and Gaussian gain (m,P), the scalar update uses innovation z−um, variance S=R+|u|²P, posterior mean m+Pu*(z−um)/S, posterior variance PR/S and log-weight increment −log S−|z−um|²/S. The signature averages the sample-level rotation over the useful symbol. R=[N0+1−sinc(.0625)²]/16, with normalized sinc. The Gaussian amplitude prior and the independent-tap projected-noise approximation apply identically to each arm.

Transport advances each node's gain phase by 2π(fΔ+½cΔ²) and frequency by cΔ at the boundary. Relative evidence and conditional gain variance travel with it. All methods run the same B update and data decoder.

| Arm | State at the beginning of B |
| --- | --- |
| carry | Posterior estimated from this link's A, transported by 30 useful-symbol durations |
| reset | Public ensemble prior advanced to the identical B epoch |
| mismatch | Posterior estimated from an independent donor A and transported by the same duration |

The receiver builds the full frequency operator from the posterior mean sample-domain rotations. It predicts row-19 pilot means using known transmitted pilots and zero-mean data. Predictive variance includes within- and between-node uncertainty, unknown data ICI and noise, assuming independent rays and unit-variance uncorrelated unknown symbols. We also record moment-matched complex Gaussian NLL normalized by N0. Data LLRs use row-wise LMMSE on data receive bins, known-pilot cancellation and average posterior channel-error power. The native parity graph uses normalized min-sum (12 rounds, α=.75, per-word stopping). Its uint8 row-major SHA-256 matches the archived P4 parity graph.

## Before opening the formal cohort

Freeze this protocol, source, tests and parity graph in a dedicated Git commit before evaluating the following seeds. Development and deterministic test examples use other seeds. No parameter fitting occurs. The study contains ten independent clusters of 32 links each. For cluster i=1..10 and link j=0..31, the seed is 802611190000+1000i+j. RNG substreams separate the channel, A payload/noise, B payload/noise and donor channel/payload/noise. Per condition this gives 5,760 common codewords, 320 additional sparse words and 5,120 end-of-frame probes. No optional stopping.

The four primary comparisons use the sparse B budget. For each control (reset and mismatch), test the control-minus-carry differences in (i) common-word BLER and (ii) noisy held-out NMSE. For each cluster NMSE is the ratio of summed error energy to summed target energy. Use the P7 two-sided exhaustive paired sign-flip routine (1,024 sign patterns) and Holm correction across all four tests. Positive differences favor carry. Report counts and cluster differences.

The separate cross-budget question compares sparse/carry against dense/reset for the 18 common codewords. A pilot-saving claim requires all four primary contrasts to favor carry at Holm p<.05 AND the one-sided 95% paired t upper bound on BLER(sparse/carry)−BLER(dense/reset) to be below +1.00 percentage point. This margin is an experimental choice. The ten independent clusters are the uncertainty units. Report the interval bound even if the criterion fails. Dense comparisons, calibration, noisy and clean prediction, coded and information BER, first/last nine words and the extra word are descriptive. This design tests two budgets and one channel family.

## Reproduction

From the repository root (Python 3.12, NumPy, SciPy):

```sh
OPENBLAS_NUM_THREADS=1 python -m unittest discover -s p8/tests -v
OPENBLAS_NUM_THREADS=1 python -m p8.study --workers 4
python -m p8.summary
```

The raw link records are deterministic and ordered by cluster then link. Preselected clusters 1 and 10 will be rerun in a fresh process, and their full records compared. Physics will be checked against an independently implemented literal CP stream, probe isolation by poisoning the hidden receives, and predictive moments by sampling the declared posterior directly in the time domain.
