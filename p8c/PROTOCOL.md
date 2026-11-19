# P8-C · Does a noisy OFDM channel estimate transfer to single-carrier data?

The parent study is P8-A/B at `4e7ccdaaacc14cd4101a6f5b824d9974a9255b58`. Freeze this design, receiver, simulator, statistical analysis and tests on the research branch before drawing the formal cohort. No training or test-set parameter selection takes place.

## Link and information boundary

Frame A is the P8 OFDM link verbatim. B carries cyclic-prefix single-carrier blocks with 32 successive time-domain symbols each, including native Gray 16-QAM payload. A and B share one synchronized clock, sample rate, bandwidth, single antenna, guard interval, public delay support (0,2,4) and a single draw of the three-path time-varying R2 trajectory at 9 dB. The CP has 16 samples. Path phase evolves at every received sample, including across the boundary. A and B payload and noise draws are independent. Each donor A comes from an independently sampled path, payload and noise on the same schedule.

Every A sends 64 known pilots. A three-path bank retains 128 possible signed Doppler/chirp nodes per path and conditional complex gain moments at each node. Advance these physical parameters by 30 useful-block durations to B. The receive grid, payload and bit likelihoods from A do not enter B. Acquisition in B uses full, known root-3 Zadoff–Chu training blocks. All row numbers are zero-based.

| B layout | Acquisition rows | Acquisition symbols | Reserved row 19 | Total known symbols sent in B | Payload LDPC words |
| --- | --- | ---: | ---: | ---: | ---: |
| Sparse | 10 | 32 | 32 | 64 | 16 common + 2 extra |
| Dense | 0,9,10 | 96 | 32 | 128 | 16 common |

The 16 shared words occupy identical positions at both budgets. The sparse layout places two further words in rows 0 and 9. Both layouts share the same last acquisition time, channel, common payload, noise realization, and clean and noisy held-out targets. Prediction and decoding receive NaNs for all 32 withheld samples. Payload decoding may use midframe training to smooth earlier data. Frame-end prediction uses earlier observations only.

## Receiver and controls

Carry, reset and mismatch use the same acquisition and readout. Carry starts from the transported posterior obtained from this link's A. Reset starts from the public ensemble prior at the same epoch. Mismatch starts from the transported donor posterior. Each update fits the single-carrier time-domain training samples with delayed known symbols multiplied by the sample-level ray rotations. Four deterministic cyclic coordinate sweeps update the factorized posterior from the original B prior. The scheduled sweeps reuse the original prior and the same likelihood. Posterior mean and per-sample variance produce the frame-end probe prediction. A time-domain linear MMSE equalizer and the native (128,64) LDPC decoder (12 rounds) handle payload. A separate diagnostic supplies exact continuous path parameters to the same kind of linear readout; it accesses simulator truth and has no role in confirmatory contrasts.

## Frozen comparisons

Use ten independent clusters of 32 newly drawn links each. Seed per link is 822611190000+1000i+j, for i=1..10 and j=0..31. Each cell scores 5,120 common words, 10,240 withheld observations and, on the sparse layout, 640 additional words. No formal waveform is reused from P8-A/B.

The four primary tests are sparse control minus sparse carry in common-word BLER and in end-of-frame noisy probe NMSE, for reset and mismatch separately. A cluster NMSE is a ratio of summed energies. Use all 1,024 paired cluster sign flips, two-sided, and Holm correction across the four tests at 0.05. Support for this cross-waveform history effect requires all four contrasts to favor carry with adjusted p<.05. The separate engineering comparison is sparse/carry BLER minus dense/reset BLER. Apply the P8 one-sided 95% paired t upper bound over the ten clusters with a +1.00 percentage-point margin. Support for this pilot-saving claim requires the four primary gates and an upper bound below the margin.

Report every arm and disclose the extra-word outcomes, information BER, coded BER, probe calibration, noiseless-target NMSE and moment-matched Gaussian NLL as descriptive endpoints. Count total correctly decoded words including extra words to examine payload yield at identical airtime. Verify clusters 1 and 10 by fresh-process replay and recompute contrasts from raw records. This study tests one directed waveform change under one continuous channel law.
