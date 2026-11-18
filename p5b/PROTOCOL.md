# Phase 5-B prospective protocol

## Design and sampling

This extension uses the P5 simulator, code, scoring, and operating points at revision `8380084` plus its supplied P5 archive. The archived P5 outcomes were known before this design. P5-B is a paired comparison on that benchmark. Replication on unseen benchmark seeds would require another experiment.

| Regime | Impairment | Es/N₀ | Pairs | Bursts per pair | Blocks |
|---|---|---:|---:|---:|---:|
| R1 | Blind PA drift and oscillator noise | 12 dB | 1–10 | 80 | 3,200 |
| R2 | Super-aliased Doppler | 9 dB | 1–5 | 80 | 7,200 |
| R3 | Four receive antennas and fractional delay | 7 dB | 1–5 | 80 | 1,600 |

Use `p5.study.rng` with the original regime, stream, pair and batch indices. In each formal pair, run all receivers on the same observations from stream 7, ten independent batches of eight. Keep the batching, since changing it changes the simulator draws. Confirm the reproduced P5 results against each pair's archived record. Only the named Genie-PA reference may see the realized PA. Every other receiver accepts the observed field, pilot geometry and symbols, public regime information, noise variance, and its declared code access. Physical truth and transmitted bits are available only for simulation, supervised training and scoring.

## Training and diagnostics

Train five independent arms in every regime. The two attention tiers use either pure wavefield updates or one public-H check-node extrinsic update between two tied physical steps. Give each tier an identical-capacity axis-local counterpart. Remove the pilot scale and feature RMS in a fifth, pure-wavefield attention ablation. Also evaluate the corresponding fixed physical anchor in each of the two code-access tiers. Treat all classical receivers supported by the P5 regime and archived P5 Unrolled EM as references.

Reset the allocation seed before each arm. Train with Adam at 0.003 for 64 updates of four bursts, on the first 64 batches of P5 stream 5. Use mean pre-decode coded-bit BCE with the P5 sign convention. Validate on the same 16 stream-6 bursts at steps 0, 16, 32, 48 and 64. All five arms receive the same waveforms and objective. The two operator tiers have exactly two tied EM steps. Use the final iterate. P5's historical arm had three physical steps and 192 Adam steps; a cross-generation difference measures the complete receiver and training recipe.

Before formal fitting, run the invariant suite and numerical preflight. Check initialization against the declared physics-only solver for each tier, backpropagation and central finite differences, gain and phase transformations in complex128, erasures, matched capacity and parity exclusion of a bit's own input. Run development fits on pair 901 in R1 and R2 using only training and validation streams. Record finite losses, validation trajectories and wall time. No early stopping or arm-specific tuning. Preserve failed diagnostics or substantive amendments; rerun affected checks before launching formal fits. Poor validation improvement is a reportable outcome. A nonfinite loss invalidates that run. Test results never select schedules or checkpoints.

Use one PyTorch CPU thread per worker. Separate independent fitting workers may overlap. Measure latency serially on the same host, with fitting and replay stopped. Both batch 1 and 64 use independent stream-10 observations, three warmups and eleven timed inference calls. Report the median and inclusive quartiles, including all internal code and physical updates. Exclude simulation, checkpoint loading, and the shared terminal decoder. Measure the archived P5 neural and classical implementations in the same timing session.

## Outcomes and inference

Record integer pre-decode errors over all coded bits, decoded errors over all information bits, and erroneous blocks over all complete LDPC words. Use P5's terminal (128,64) normalized min-sum decoder with at most 12 iterations. Compute channel NMSE as ten times the base-10 logarithm of the pooled sum of squared errors divided by pooled ideal CFR energy. Preserve unrounded counts, errors, and energies. No thresholds or floors on observations, error rates, or reported effects to manufacture a pass.

Each independent training/test seed pair supplies one paired block-error difference. Positive effects mean the first named arm saves errors. Use all sign assignments, two-sided, including ties, and monotone Holm adjustment within each family at 0.05. Report means of paired differences in percentage points, pooled counts, and paired wins, ties and losses. Five pairs have minimum two-sided p of 0.0625. A nonsignificant result does not establish equality, equivalence or noninferiority. No equivalence margin is prespecified.

* **Primary family, three tests.** R1 pure attention versus Pilot-CPE-DD. R2 and R3 parity attention versus Waveform-DD.
* **Ablation family, eighteen tests.** In each regime, compare pure attention with local pure, parity attention with local parity, pure attention with no-gauge pure, parity attention with pure attention, pure attention with its fixed physics anchor, and parity attention with its fixed parity-physics anchor.
* **Kernel family, three tests.** On the P5 R3 stream-8 probe of eight bursts per seed, compare pure attention's trained alignment with its initialization, its trained shuffled-key null, and the paired archived P5 trained alignment. Use the P5 off-diagonal Pearson scorer on actual attention weights, matched coordinates, the same physical target and eight stream-9 column permutations per layer. Average bursts, layers and permutations inside each seed before inference. Also report whether the mean passes 0.084 descriptively. The parity tier's kernel scores are descriptive. The kernel scorer has access to path truth only after predictions. It measures association.

The remaining receiver contrasts and both latency ratios are descriptive. Attribute separate effects to operator learning, internal code feedback and global attention using their corresponding controls. To examine use of the individual operators, intervene on the frozen primary-tier checkpoints over the same test frames. Suppress each group in turn at every unroll step: cubic/quintic radial terms, shared phase/precision output, spectral residual coupling, or direct complex CFR output. Compare each intervention with the intact fit. Suppressing all six complex outputs must reproduce the arm's fixed anchor. These intervention deltas measure dependence of this trained receiver on each group with its remaining weights fixed. Interactions prevent an additive decomposition; the test cannot establish recovery of the true PA curve or physical interference coefficients.

## Reproducibility

Save per-pair final state dictionaries, validation traces, integer counts, seed identities, and probe values. Replay all final fits by reloading the checkpoints and regenerating the scored frames. Preserve an exact per-file digest inventory of the input P5 directory. The final checkout receipt reruns both test suites and the P4 publication verifier, checks tracked and staged diffs, and compares every P5 file against that inventory. All writes for the extension belong under `p5b/`.

## Pre-fit numerical amendment

The first diagnostics passed the simulated gain and erasure cases. A constructed R1 observation with opposite pilot-row tap estimates exposed a branch ambiguity. Roundoff could move the integrated cross phase between +π and −π after multiplying the received field by a complex gain. These endpoints give different intermediate symbol rotations. Use the archived P5 positive-half-turn convention at a numerically antipodal cross product. Two additional fixtures exposed ties between mirrored Doppler hypotheses in R2 and R3. Resolve numerically equal candidates in fixed dictionary order. Repeat the preflight and pair-901 development fits on the revised solver before the first formal fit. The architecture, loss, training budget, seed list and registered hypotheses above apply throughout.
