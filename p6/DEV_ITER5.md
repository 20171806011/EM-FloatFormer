# Phase 6 development study, iteration 5

**The development gate remains closed.** The uniform 48-update recipe with original R1 PA timing, R2 cue 0.05 and the first spatial shrinkage operator is the development selection. Its pooled Axial-Parity counts are 25/384, 337/1,728 and 49/384. It clears the three explicit count thresholds, the sign-repair threshold and the joint Green threshold. Its trained gain audit exceeds the absolute LLR limit under one erasure condition. The 48-update cue-off alternative also exceeds that limit. Both 36-update spatial candidates miss the R3 count threshold. There is no qualifying configuration among these fits. Stream 7 stays sealed and the formal protocol remains unfrozen.

## Development design

The comparison uses the three existing development pairs 901–903. Each fit uses the pair's stream-5 initialization and B4 minibatches. Validation uses one stream-6 B32 draw at 12, 9 or 7 dB for R1, R2 or R3. R3 kernel measurements use eight stream-8 bursts and eight stream-9 permutations. Training and validation keep the Iteration 4 loss, width, physical bounds, optimizer and depth.

The fixed candidate grid crosses separate 36- and 48-update warm-cosine schedules with two R1 choices (original or deferred learned PA) and two R2 cue coefficients (0.05 or zero). Both parity arms train from scratch. Each schedule reaches the same terminal learning rate at its own final update. The R1 deferral retains the nominal waveform physics throughout and starts both the learned forward PA distortion and receive-side dewarp at EM pass index 1. Within its first pass, equalization uses observed pilots and parity-free soft symbols.

Selection uses one schedule for every regime and arm. The stated Axial-Parity count gates are 25, 340 and 49, with at least eight of ten R2 sign repairs and a direct joint R3 Green correlation of at least 0.34. The spatial check additionally compares the full model with fixed-weight suppression on the same bursts, including the requested 48-word objective, steering cosine, and joint and spatial Green coefficients. Among schedules that clear the checks, prefer 36 updates. Keep 0.05 unless removing the cue weakly improves both parity arms and improves at least one. Adopt the R1 delay if it improves Axial-Parity without breaking initialization or PA invariants. Preserve all tested candidates and failures.

## PA timing and chirp cues

Each row pools three independently initialized pairs. The two schedules start from identical tensors for each arm and pair. The cap is 10, auxiliary NMSE coefficient 0.005, anchor coefficient on squared displacement 0.00005 (ANCHOR/2) and global gradient-norm cap 1. The four warm-up updates reach LR 0.0045, followed by cosine decay to 0.00045. All decoding uses raw logits.

| Regime and change | Updates | Axial-Parity failures | Local-Parity failures | Repaired signs, Axial / Local |
|---|---:|---:|---:|---:|
| R1 original PA timing | 36 | 25 | 27 | — |
| R1 defer learned PA | 36 | 25 | 28 | — |
| R1 original PA timing | 48 | 25 | 24 | — |
| R1 defer learned PA | 48 | 25 | 24 | — |
| R2 cue 0.05 | 36 | 341 | 342 | 8 / 9 |
| R2 cue off | 36 | 340 | 344 | 8 / 9 |
| R2 cue 0.05 | 48 | 337 | 339 | 9 / 9 |
| R2 cue off | 48 | 338 | 343 | 8 / 8 |

R1 scored 384 words per cell. Delaying learned PA tied Attention on every pair and did not reach 24 pooled failures. Keep the original PA timing. R2 scored 1,728 words per cell. At 48 updates, 0.05 gave fewer failures in both arms and repaired an additional initially wrong chirp direction in each arm. All eight R2 cells introduced zero wrong signs. Fixed+Parity reproduced 29/429/65 failures across R1/R2/R3. The designated classical receivers reproduced 40/446/65.

The short schedule uses 25% fewer optimizer updates. For Axial-Parity with the original R1 timing, 0.05 R2 cue and initial R3 shrinkage, mean seconds per development fit were 7.57/7.96 for R1, 9.66/12.26 for R2 and 10.73/15.71 for R3 (36/48). These are concurrent CPU fit times and exclude post-fit scoring.

## Antenna transport and fixed-weight suppression

The global branch averages its actual complex steered transport over time and frequency and takes the Hermitian part. A spectral bound, projection onto the subspace orthogonal to the constant vector, and restoration of the constant mode produce a positive semidefinite contraction with unit row sums. The local branch derives a symmetric tridiagonal matrix from the learned three-tap stencil with reflecting endpoints and the same spectral bounds. The projection of the centered space field onto nominal delays and estimated rays gives one nonnegative tanh-bounded coefficient per burst and ray. A zero coefficient preserves the current LS solution exactly. Each arm uses 856 trainable real scalars.

| Initial shrinkage candidate | Full errors by pair | Full total | Spatial off by pair | Spatial off total |
|---|---|---:|---|---:|
| Axial-Parity, 36 | 11, 16, 24 | 51 | 11, 16, 24 | 51 |
| Local-Parity, 36 | 11, 16, 24 | 51 | 11, 16, 24 | 51 |
| Axial-Parity, 48 | 9, 16, 24 | 49 | 9, 16, 24 | 49 |
| Local-Parity, 48 | 9, 16, 24 | 49 | 9, 16, 24 | 49 |

| Axial-Parity, 48 | Initial | Trained, full | Trained, spatial off |
|---|---:|---:|---:|
| Direct joint Green | 0.351957 | 0.352660 | 0.352768 |
| Spatial Green, full probe grid | 0.164458 | 0.164072 | 0.164042 |
| Final steering cosine on validation bursts | — | 0.980314 | 0.980248 |

The trained direct joint exceeds the matched uniform-support control (0.268702) and its shuffled null (0.001817). The historical product reaches 0.512210, starting at 0.509588. The spatial-line-only trained coefficient is 0.858708. Enabling shrinkage slightly raises final steering cosine and full-grid spatial Green correlation. It slightly lowers direct joint correlation against fixed-weight suppression. The direct joint also stays below Iteration 4's 0.35311. The handle costs no decoded words on any pair, but this intervention shows no decoding gain. In the initial candidate the median 48-step Axial shrinkage strength is 0.01261, changing the LS ray vector norm relatively by a median of 0.00155. The 49-error result meets the explicit gate and misses the separate 48-word spatial objective.

A single follow-up replaces the ray-power denominator with regularized cross-power of the projected field and the ray. The existing 0.04 denominator floor and 0.18 tanh cap are retained. The formula and twelve refits were specified before evaluating the follow-up.

| Cross-power-normalized alternative | Full errors by pair | Full total | Suppressed total |
|---|---|---:|---:|
| Axial-Parity, 36 | 11, 16, 24 | 51 | 51 |
| Local-Parity, 36 | 10, 16, 24 | 50 | 50 |
| Axial-Parity, 48 | 9, 16, 24 | 49 | 49 |
| Local-Parity, 48 | 9, 17, 24 | 50 | 49 |

The alternative raised 48-step final steering cosine from 0.980234 with suppression to 0.980413 with the handle enabled. Its across-pass direct Green coefficient fell from 0.352861 to 0.352437 under the same intervention. It did not improve Attention decoding and added one Local error. The receiver retains the first shrinkage formula.

[Development scores, references, Green controls and pair results](results/dev_iter5/dev_scores.json) · [Checkpoint and training receipts](results/dev_iter5/fits/) · [Follow-up specification](iter5_spatial_refine.py)

## Verification

The P6 suite passed 30/30 with zero skips. The 75 tensor-direction finite-difference checks had a maximum absolute error of 9.676e-10. Two historical tests that assumed an additive spatial handle now check bounded shrinkage and operator structure. Two additional tests exercise the R1 timing switch and the R2 zero-cue path. All 45 actual first minibatches of the 48-update, 0.05-cue configuration had 856/856 nonzero finite real scalar gradients. All five arms in each pair and regime had identical initial state tensors and exactly matched their corresponding fixed receiver's LLR and CFR at initialization.

On trained checkpoints the 0.05-cue configuration passed 359/360 gain-and-pilot condition checks. R2/901 Axial-Parity with the first pilot row erased reached a maximum absolute LLR change of 1.0675904604795505e-12 against the strict upper bound of 1e-12. Its relative CFR error was 2.984e-14. All 360 cases had zero changed payload coded-bit decisions. The four gauge-enabled arms were tested on every development pair, with ten conditions and 27 magnitudes each, spanning 1e-35 to 1e20.

An independent replay of the failing case again exceeded the limit (1.1670664434859646e-12 on a payload logit at magnitude 0.5 and phase 3.5961264 rad). The exact maximum changed between execution contexts. The gate decision for this candidate is the same on both executions. [Detailed replay](results/dev_iter5/audit_r2_901_failure_detail.json)

In a separate instrumented run at that gain, the initializer selected the same chirps and rotations for both inputs. On the third EM pass, the measured change in one bank target was −1.794e-13 radians. The softmax derivative applied to the observed coherence perturbation predicted −1.833e-13 radians. The final maximum ΔLLR in this instrumented run was 5.01e-13 with the trained model, 3.91e-14 with its bank suppressed, and 3.55e-14 with Fixed+Parity. This localizes amplification to the sharp bank in this case. The instrumented comparison does not replace the full audit. [Per-pass trace](results/dev_iter5/audit_r2_901_failure_trace.json) · [Conditioning calculation](results/dev_iter5/audit_r2_901_conditioning.json)

The 48-step cue-off candidate gave 25/338/49 Axial-Parity failures and repaired 8/10 R2 signs. Its 45/45 anchor and first-update checks passed. Its trained symmetry suite passed 359/360 cases with zero flips in 21,565,440 coded-bit comparisons. R2/901 Axial-Parity with the first pilot row erased reached 1.8403056856186595e-12 absolute ΔLLR and 4.394e-14 relative CFR error. The R2 cases were run on the cue-off fits. R1 and R3 used the same verified checkpoints as the first candidate because the cue only runs in R2. [Cue-off gain audit](results/dev_iter5/cue_off_audits/audit_gauge.json) · [Cue-off first update](results/dev_iter5/cue_off_audits/audit_first.json)

| 48-update development selection | R1 failures / 384 | R2 failures / 1,728 | R3 failures / 384 |
|---|---:|---:|---:|
| Fixed physics | 40 | 591 | 71 |
| Fixed physics + parity | 29 | 429 | 65 |
| Axial-Pure | 43 | 458 | 55 |
| Local-Pure | 42 | 461 | 54 |
| Axial-Parity | 25 | 337 | 49 |
| Local-Parity | 24 | 339 | 49 |
| Axial-NoGauge | 36 | 448 | 58 |
| Equal-information classical comparator | 40 | 446 | 65 |

The three-pass fixed receiver gains 11, 162 and 6 words from parity alone. With parity already present, the learned Axial receiver gains a further 4, 92 and 16 words. The matched Local-Parity receiver wins one word in R1, loses two in R2 and ties R3. These three reused development pairs establish no prospective Attention advantage. For R1, privileged Genie-PA scored 38/384 and nominal IDD-PA scored 81/384.

P5 passed 15/15 and P5-B passed 13/13, with all original test bodies executed. The existing fixture adapter used 30 distinct development draws to keep the Stage 1 firewall intact. The default publication P4 verifier exited 0 on its archived verification receipts. These are separate from a Stream 7 replay.

[P6 tests](results/dev_iter5/audit_suite.json) · [First-update and anchor evidence](results/dev_iter5/audit_first.json) · [Full gain audit](results/dev_iter5/audit_gauge.json) · [P5, P5-B and P4 verification](results/dev_iter5/legacy_checks.json)

At closeout the tracked worktree and index matched commit `8380084` on all 617 paths. The 138 files under `p5/` and 165 under `p5b/` are external to that commit. Their paths, byte sizes and nanosecond modification times match the pre-development inventory. The SHA-256 of the development firewall remained `85ebeb49bdc56a7dcc4961eb16d414c175478ba5640ea742d0d32a95b63aa6b1`. Closeout checked the 108 development fits and their allowed seeds. The prospective archives stayed sealed, so the scheduled formal SHA-256 inventory and checkout replay have not taken place. [Integrity and isolation receipt](results/dev_iter5/integrity_after.json) · [Machine-readable development gate](results/dev_iter5/gate.json)
