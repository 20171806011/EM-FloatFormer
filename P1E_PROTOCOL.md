# P1-E Protocol — new-seed replication and computational-cost audit

**Prespecified before P1-E model evaluation, 2026-11-15.** Synthetic, receive-only civilian QPSK/OFDM only. No electromagnetic emitters, adversarial countermeasures, radar or weapons.

## Scientific question

Does the P1-D 10-dB out-of-distribution advantage of **circular CNN + log-amplitude/phase** over **Transformer + log-amplitude/phase** reproduce with **new, disjoint training seeds**? Do the two representations affect that comparison? How does performance compare with LS and with matched-test-prior LMMSE (which has additional prior information)?

## Two prespecified comparisons

At **10 dB**, separately in **equal-four-tap** and **six-tap** channels, define

`delta_seed = BER(transformer__logamp_phase) - BER(circular_cnn__logamp_phase)`

in percentage points, measured on identical held-out frames per seed. Prediction: positive delta. Five independent new training seeds (404, 505, 606, 707, 808), interval over 5 training seeds, and two-sided exact sign-flip p-values (32 sign assignments). Holm correction across two endpoints. New-seed replication only, **not** independent implementation or real-world hardware testing.

## Training plan

Reuse unchanged P1-C train function and hyperparameters (32 carriers, 750 steps, 64 frames/batch, AdamW lr 0.002, 0–20 dB on train3 channel). Train two architectures (circular CNN and Transformer) × two representations (raw IQ + magnitude and log amp + regularized phase), 4 configs per seed, 20 checkpoints in total. No P1-D tuning.

## Held-out evaluation

For each seed, evaluate 2048 new simulated frames per scenario, same samples for every model and classical baseline, 64 uncoded QPSK bits/frame, 655360 bits per method/scenario pooled across seeds. Three channel profiles (train3, equal4, long6) × 3 SNRs (0,10,20 dB) = 9 scenarios. Baselines LS, DFT train-support, LMMSE train-prior, LMMSE matched-test-prior (privileged prior), true-channel reference (unavailable in operation). Separate test RNG stream. P1-C/P1-D evaluation samples not reused by construction.

## Additional audits

Paired random-seed effect estimates, exact error counts, CPU batch=1/128 latency median and p95 (300 repeated runs after warmup, fixed thread count), independent reloading/replay of one seed, checksum of checkpoints, consistency and adversarial-free unit tests. Secondary comparisons exploratory; no unadjusted multiple-testing claims. CPU timings are not full SDR receive-system latency.
