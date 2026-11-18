# P4-D pre-outcome protocol

This document defines the intended confirmatory run. Sealing occurs only when the P4-B gate has passed and the write-once P4D_FREEZE.json exists. The completion criterion is ten fully trained seeds with all paired receiver counts, a six-contrast analysis and an independent replay. A development smoke run cannot meet this criterion.

## Training and held-out data

Use independent training seeds 74001 through 74010, in numerical order. Initialize the corrected EMFloatFormer2D with width 16, two heads and two blocks per seed. Train each from scratch on 600 Adam updates, 64 independent uncoded OFDM slots per update. Accumulate sixteen equal microbatch means of four slots before gradient clipping to norm 2 and one optimizer step. Model initialization uses the training seed. A dedicated training-data generator uses seed + 1000. An exhausted run resumes from the last saved update with model, optimizer, cosine schedule, training-data generator and global PyTorch generator restored. Save after each twentieth update and at completion. Record the SHA-256 of the final named parameter and buffer values for each seed.

The primary model is the two-dimensional `p4.slot_2d.EMFloatFormer2D` with its Gaussian pilot initializer and learned channel and distortion heads. The M3 workload qualification may use task-specific receivers, including the tap-phase initializer, learned soft-symbol channel updates and waveform feedback. Their measurements appear in the M3 development panel. The six primary P4-D contrasts quantify the specified Gaussian-initialized architecture. All ten official seeds remain reserved until qualification.

Use the P4-B 14-symbol, 32-tone slot, CP 16, oversampling 4, 16-QAM and two comb-2 DMRS symbols. Training draws the equal4 delay profile and frame SNR uniformly on [10, 26] dB. For zero-based update j, take PA at index j mod 3 in (strong, mild, clean) and normalized Doppler at index floor(j/3) mod 3 in (0, 0.01, 0.02). Training CFO is zero. The blind CP synchronizer processes every training and evaluation slot. Use JointPhysicalLoss2D with unit BCE and channel NMSE weights and PA-consistency weight 0.25. The P4-B forward-consistency term uses the diagonal channel approximation. Use Adam with default betas and epsilon, backbone learning rate 0.008 and gamma learning rate 0.02. Both cosine schedules have eta_min 0.0008. Fixed CPU, two PyTorch threads, deterministic algorithms and the runtime versions in the freeze apply to every seed.

At precisely 10 dB and strong Rapp PA (A=1.25, alpha=0.5), test the three prespecified conditions below. The Doppler condition also includes a small CFO. Its comparison addresses that joint impairment.

| Index | Condition | Profile | Normalized Doppler | Normalized CFO |
|---|---|---|---|---|
| 0 | In distribution | equal4 | 0.01 | 0 |
| 1 | OOD Doppler with CFO | equal4 | 0.05 | 0.02 |
| 2 | OOD delay | long12 | 0.01 | 0 |

For every seed s and condition c, use eight batches indexed b = 0,...,7, each with four independently generated slots. Use evaluation RNG seed 10000000 + 1000s + 100c + b. Regenerate the identical coded bits, channel, noise and pilots from this seed for every method. The stored LDPC specification uses a deterministic (128,64) code and at most 25 soft min-sum iterations. Each 32-slot condition contains 416 complete codewords and 26,624 information bits per training seed. For each method record exact integer pre-decoder transmitted-bit errors, post-decoder information-bit errors, failed codewords and their denominators. Terminate on nonfinite outputs, missing records or inconsistent denominators. All ten seeds remain required following any interruption.

## Primary statistical comparisons

Compare EM against two prespecified classical receivers in each of the three conditions. Both classical receivers use the same fixed equal4 prior with normalized Doppler 0.01 and mild PA coefficient A=2.25 for all scenarios. The first uses the 2D LMMSE and Bussgang front end alone. The second adds two PA-cancellation iterations. All receivers have access to the same received samples, known pilots and noise variance. Also record descriptive results for the two analogous classical receivers given the actual delay, Doppler and PA settings.

For each of six comparisons calculate paired differences in decoded block-error rate (classical minus EM) separately for each independent training seed. Positive values favor EM. The two-sided sign-flip test conditions on the magnitudes and assumes independent, sign-exchangeable seed differences under its null. Enumerate all 1024 sign assignments, counting ties in the tail. Apply Holm step-down jointly to the six p-values, with strict adjusted p < 0.05. Compute the statistic from integer error differences with their common denominator. The seed is the unit of inference. Conditions, codewords sharing a slot, matched-parameter comparators, BER, and P4-C factorial estimates supply no additional replicates. Report direction, magnitude, per-seed counts and adjusted p, including ties and adverse results.

## One-dimensional factorial replication

The joint freeze also covers the eight matched-parameter arms in `p4cd_factorial_protocol.json`. Use seeds 74001–74010 for both studies. The coded and factorial endpoints share identifiers and answer separate questions. For each of the 80 one-dimensional fits use the specified paired initializations, 32 Adam steps of 12 frames, equal4, and the archived joint loss. At 10 dB and strong PA, generate 96 held-out frames at seed + 2000000. Score all three gains on those same bits. Save the final weight fingerprint and integer errors for every seed, arm and gain. Each gain contributes 12,288 bit decisions per arm. A seed remains the inferential replicate.

For each factor sum the four off-arm error counts and subtract the four on-arm counts at each seed. Divide by four times the pooled per-arm denominator to obtain its BER reduction. Enumerate the 1,024 sign assignments and apply Holm jointly to these three factorial contrasts in a separate prespecified family. Two-way and three-way interactions and unit-gain contrasts are descriptive. The six coded BLER comparisons retain their own Holm family. The primary coded claim and mechanism claims must be identified by family when reported.

The main execution command publishes per-arm records as they complete and can resume a partially completed factorial. Replay retrains all 80 arms using the independent training loop and checks each fitted weight fingerprint, all gain-specific counts and the summary. Development rehearsals use disjoint identifiers.

## Freeze, execution and replay

The hash manifest covers this protocol, the experiment code and physical receiver dependencies, the project plan, the legacy manifest, the accepted P4-B gate record and the runtime identity. The accepted P4-B evidence must be recorded in a repository-local JSON file with stage "P4-B" and passed true. Its author is responsible for the underlying qualification. At freeze and replay, verify every one of the 230 tracked paths in pre-P4 commit `f878a7363254c6e34bee5f6e278135d74e1d54e0` against its Git blob and index entry. This includes later P3 and exploratory work. Git text normalization accommodates Windows checkouts. The legacy SHA-256 manifest supplies a second check on its 115 entries, with the documented README revision verified against Git. A SHA-256 fingerprint of all 230 files' raw checkout bytes must match again on replay. Later additions under P4 remain outside this historical tree. The sealed sources and completed counts are write-once. Any subsequent source changes require a separately identified study.

Run from the project root with the recorded Python environment.

```text
python -m p4.confirmatory_protocol --smoke
python -m p4.confirmatory_protocol --freeze-after-gate path/to/accepted_P4B_gate.json
python -m p4.confirmatory_protocol --execute
python -m p4.confirmatory_protocol --replay
```

The smoke uses development seeds 68321 through 68330 with two updates of four slots. It tests the checkpoint interruption/resume and count replay on the first seed. Hold all official draws while the P4-B gate is pending. Official execution writes one checkpoint and count record per seed, a raw integer CSV and a summary JSON. Replay recalculates decisions and decoder counts for all five receivers from each completed checkpoint. Independent audit additionally recomputes the six statistics from the CSV.

