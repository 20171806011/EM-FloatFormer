# EM-FloatFormer P3-B — single-factor nonidealities in a synthetic civilian OFDM receiver

**2026-11-15 · completed exploratory benchmark · passive receive-only simulation**

## Snapshot and provenance

P3-B protocol was committed to the original repository **before training** as `P3B_PROTOCOL.md` (commit `440c4aba5881306e29165727dee03cb7f0dc1550`). This run does not amend previously frozen P2-E, P3-A or P3-A2 findings. This report is generated after model evaluation using fixed outputs, not a revised preregistration. No real radio hardware, radar, military data, adversarial emissions or transmitter control are used.

**Principal finding: No trained neural receiver surpasses the per-condition envelope of the seven predeclared nonprivileged classical methods in any of the 135 scenario/SNR/impairment cells.** The 30 trained heads also add very little to the old frozen D backbone. This is a *descriptive* comparison: choosing the minimum classical error retrospectively within a fixed predeclared method list forms a classical-favoring **oracle selection envelope**, not a single deployable adaptive chooser. We also report individual fixed classical methods so no inference depends on that envelope.

**Do not call neural logit residual scale-aligned symbol MSE 'measured EVM'.** It is a diagnostic computed from arbitrary classification logits, not calibrated physical RF error-vector magnitude. The channel MSE applies to pilot-derived channel estimates (including frozen D); the neural heads do not emit a separately learned channel estimate.

## B0: time-domain link checks

- New random QPSK pilots, 32 subcarriers, 16-point CP, two publicly known QPSK anchors in the data symbol. The **30 unknown QPSK tones** yield 60 scored bits per frame; anchors do not enter BER.
- Explicit IFFT -> CP -> optional memoryless PA magnitude compression -> causal multipath FIR -> CP removal -> time-domain receiver noise/ADC/CPE -> FFT. Noise power is scaled to yield `N0` per frequency tone. The zero-impairment, zero-noise limit agrees with the old `H×X` frequency model to float tolerance, while having different random streams.
- `test_p3b_link.py`: **12/12 tests pass**, including calibration and no leakage. The additional LMMSE Fourier-delay formula was independently cross-checked against a direct matrix solve before training (maximum tested discrepancy around 2.9e-6).
- This chain does not include a standardized 3GPP channel, changing sample clocks, realistic analog front ends or multiple moving OFDM symbols. It does not replace P3-D.

## B1: impairment calibration (same shared underlying frames)

Single additional factor per test frame: PA compression (`pa`), Bernoulli-Gaussian random receiver impulse (`impulse`), data-only common-phase error (`cpe`), or time-domain scalar ADC clipping/quantization (`adc`). `clean` is the reference. Mild and strong severities are frozen in the protocol; random impulsive noise is non-adversarial. `noise_var` for impulse includes *mean* burst power, not labels or realizations.

Representative calibration aggregates, train3 at 10 dB; numbers are synthetic diagnostic means:

| Condition | PA out/in power | Burst rate | |CPE| rad | ADC saturation |
| --- | --- | --- | --- | --- |
| clean | 1.000 | 0.0000 | 0.0000 | 0.0000 |
| pa_mild | 0.887 | 0.0000 | 0.0000 | 0.0000 |
| pa_strong | 0.649 | 0.0000 | 0.0000 | 0.0000 |
| impulse_mild | 1.000 | 0.0300 | 0.0000 | 0.0000 |
| impulse_strong | 1.000 | 0.0497 | 0.0000 | 0.0000 |
| cpe_mild | 1.000 | 0.0000 | 0.0753 | 0.0000 |
| cpe_strong | 1.000 | 0.0000 | 0.2972 | 0.0000 |
| adc_mild | 1.000 | 0.0000 | 0.0000 | 0.0097 |
| adc_strong | 1.000 | 0.0000 | 0.0000 | 0.1527 |


## B2: model fitting and matched observations

- Five **new** seeds `48101–48105`; **30 new head checkpoints** (`clean`/`mixed` × `raw`/`log_phase`/`rms_log` × five seeds); fixed 600 training steps, batch 64, AdamW learning rate .002 and three-profile train mixture (train3/equal4/long6). Clean vs mixed differ only in availability of one mild physical factor in 60% of mixed training batches.
- Each head has **2,090 learned parameters**, on top of a seed-matched **985-parameter frozen P3-A2 D** pilot channel model (from seeds 47101–47105). The frozen model is unchanged and its copied weights agree SHA256 with P3-A2 source checkpoints.
- All six models per seed share initialization and underlying input stream. Classical estimators also receive the same known random pilot, two public data anchors and declared reported mean noise. Real true channel and unknown data bits are never inference inputs.
- Important exception: no matched-calibration nonlinear-PA inverse/Bussgang receiver was implemented. Hence this benchmark cannot establish how the neural model compares to all strong non-linear-aware signal-processing receivers.

## B3: held-out results

Each method has five seeds × 768 independent frames = **230,400 scored bits** per profile, SNR and condition. Complete grid: 5 channel profiles × 3 SNR values (0/10/20 dB) × nine conditions × 14 methods = **1,890 aggregate cells**, from **9,450 original per-seed rows**.

### Clean 10-dB baseline (BER; lower is better)

| Profile | Frozen P3-A2 D | Mixed/raw head | Mixed/log head | Mixed/RMS head | Fixed threshold without anchor correction | DFT-12 |
| --- | --- | --- | --- | --- | --- | --- |
| train3 | 6.711% | 6.745% | 6.746% | 6.743% | 5.072% | 7.933% |
| equal4 | 7.082% | 7.088% | 7.107% | 7.097% | 5.061% | 7.538% |
| long6 | 8.180% | 8.199% | 8.200% | 8.210% | 5.523% | 7.614% |
| long12 | 10.660% | 10.721% | 10.697% | 10.720% | 7.064% | 7.555% |
| sparse12 | 9.695% | 9.726% | 9.721% | 9.740% | 4.978% | 8.428% |


### Eight predeclared *mild* endpoint conditions (equal4/long6 @ 10 dB)

`Best neural` and `best classical` are each per-cell minima selected from **predeclared fixed lists**, included only as an intentionally optimistic neural / conservative classical **descriptive envelope**. These are not deployable policy selectors and are not independent performance claims. `threshold_no_align` is a **fixed** classical method, not itself a tuned oracle.

| Profile | Condition | Frozen D | Clean/raw | Mixed/raw | Best neural among six | Best classical among seven |
| --- | --- | --- | --- | --- | --- | --- |
| equal4 | pa_mild | 7.941% | 7.939% | 7.937% | 7.934% (clean__rms_log) | 5.680% (threshold_no_align) |
| equal4 | impulse_mild | 9.764% | 9.784% | 9.770% | 9.759% (mixed__log_phase) | 7.243% (threshold_no_align) |
| equal4 | cpe_mild | 7.082% | 7.095% | 7.088% | 7.088% (mixed__raw) | 5.189% (threshold_no_align) |
| equal4 | adc_mild | 7.131% | 7.136% | 7.128% | 7.128% (clean__rms_log) | 5.092% (threshold_no_align) |
| long6 | pa_mild | 9.134% | 9.140% | 9.128% | 9.128% (mixed__raw) | 6.300% (threshold_no_align) |
| long6 | impulse_mild | 11.104% | 11.109% | 11.112% | 11.098% (clean__log_phase) | 7.992% (threshold_no_align) |
| long6 | cpe_mild | 8.180% | 8.194% | 8.199% | 8.194% (clean__raw) | 5.645% (threshold_no_align) |
| long6 | adc_mild | 8.236% | 8.258% | 8.250% | 8.227% (mixed__log_phase) | 5.555% (threshold_no_align) |


### Strong-impairment descriptive stress cells (equal4/long6 @ 10 dB)

| Profile | Condition | Frozen D | Clean/raw | Mixed/raw | Best neural among six | Best classical among seven |
| --- | --- | --- | --- | --- | --- | --- |
| equal4 | pa_strong | 11.137% | 11.136% | 11.138% | 11.136% (clean__raw) | 8.091% (threshold_no_align) |
| equal4 | impulse_strong | 18.707% | 18.688% | 18.684% | 18.684% (mixed__raw) | 15.293% (threshold_no_align) |
| equal4 | cpe_strong | 7.082% | 7.095% | 7.088% | 7.088% (mixed__raw) | 6.025% (dd_threshold) |
| equal4 | adc_strong | 8.316% | 8.341% | 8.321% | 8.309% (mixed__log_phase) | 5.924% (threshold_no_align) |
| long6 | pa_strong | 12.591% | 12.622% | 12.610% | 12.603% (mixed__log_phase) | 8.988% (threshold_no_align) |
| long6 | impulse_strong | 20.160% | 20.187% | 20.185% | 20.183% (mixed__log_phase) | 16.901% (threshold_no_align) |
| long6 | cpe_strong | 8.180% | 8.194% | 8.199% | 8.194% (clean__raw) | 6.535% (train_lmmse) |
| long6 | adc_strong | 9.405% | 9.403% | 9.408% | 9.403% (clean__raw) | 6.461% (threshold_no_align) |


For both profiles, anchor-based phase correction can make all *aligned* receivers' BER counts unchanged under a pure data-symbol common rotation. However, **in clean conditions noisy-anchor correction often hurts more than it helps**; an unaligned hard-threshold baseline is often better. Strong `cpe` does change the unaligned receiver's BER. This is common-phase rotation, **not** within-symbol CFO or ICI, so it must not be extrapolated to general oscillator errors.

`mixed`-trained neural heads show small, condition-dependent changes relative to the clean heads, usually measured in hundredths of BER percentage points. They cannot be credited with overcoming an impairment when a substantially better fixed DSP estimator exists. Detailed paired training-seed differences are in `p3b_results/paired_neural_vs_frozen.csv` and `primary_exploratory_10db.csv`.

We report **absolute BER** and per-cell impairment penalty against the **same method's clean BER**; differences are directly computable from the paired `ber_summary.csv`. Profile/SNR/seed are not independent RF experiments. No confirmatory p values are claimed and no additional training occurred after viewing the grid.

### Privileged diagnostic references (added AFTER fixed primary evaluation)

The frozen protocol called for two additional diagnostic references that were absent from the original 14-method table. We computed them separately on **exactly regenerated input frames** without any neural training or re-selection:

- `matched_test_prior_privileged`: a linear-Gaussian posterior using the *actual test profile covariance*, with extra knowledge unavailable to ordinary receivers; no awareness of sample-wise PA, ADC or CPE perturbations.
- `true_channel_privileged`: exact linear multipath transfer function, but **not** an oracle for nonlinear symbol distortion, common-phase drift or saturation.

These are post-run supplement calculations, **not** an amendment to the frozen test list or equal-information win. Their counts are separate in `privileged_reference_by_seed.csv`.

| Profile | Matched-test prior LMMSE (clean 10 dB) | True linear H (clean 10 dB) | Fixed threshold unaligned |
| --- | --- | --- | --- |
| equal4 | 6.141% | 5.531% | 5.061% |
| long6 | 6.487% | 5.506% | 5.523% |
| long12 | 7.498% | 5.497% | 7.064% |
| sparse12 | 6.579% | 6.063% | 4.978% |


## CPU inference costs (median/p95 milliseconds)

Single four-thread PyTorch CPU runtime; 180 post-warmup repetitions in one environment. `full simulated pipeline` includes synthetic in-memory waveform generation/channel application, **not hardware ADC/SDR data transfer, protocol frame handling or a real air interface**. Neither FLOPs nor actual device power consumption are measured; only parameter memory is known.

| Path | batch1 median/p95 | batch64 median/p95 |
| --- | --- | --- |
| simulated_link_only | 0.196 / 0.273 | 0.318 / 0.479 |
| mixed_raw_head_only | 0.259 / 0.346 | 0.667 / 0.973 |
| frozen_backbone_only | 0.216 / 0.281 | 0.627 / 1.003 |
| mixed_raw_backbone_plus_head | 0.517 / 0.599 | 1.350 / 1.793 |
| threshold_no_anchor_postlink | 0.045 / 0.097 | 0.080 / 0.175 |
| neural_full_simulated_pipeline | 0.735 / 1.052 | 2.005 / 2.306 |
| threshold_no_anchor_full_simulated_pipeline | 0.267 / 0.407 | 0.447 / 0.671 |


The `mixed__raw` head has 2,090 trained parameters + 985 frozen backbone parameters = 3,075 total model parameters, storing about 12.3 KB of FP32 weights alone, excluding activations, optimizers and interpreter overhead. The unaligned classical threshold channel estimator has no trainable parameters. Times should not be read as throughput or deployment latency certificates.

## B4: independent integrity audit and exclusions

- `p3b_independent_audit.py` recomputed all **1,890** aggregate BER integer counts from **9,450** original per-seed counts and independently matched all original per-seed CSV files. Every row stores 768 frames, 46,080 scored bits, the number of actual errors, and its exact error ratio.
- **30/30 saved head checkpoint SHA256 values match** their training registry; the model weights reload with correct architecture, seed and frozen protocol SHA. **210/210 re-executed error/bit records exactly match** the originals for the three preselected replay cells across all five seeds. Frozen P3-A2 D weights were reused unchanged.
- All physical B0 tests (12) pass and 1,350 extra privileged-reference diagnostics were calculated on the existing seeded grid. Independent calibration checks confirm the intended relative PA power, impulse event rate, CPE magnitude and ADC saturation severity; no hidden test tuning was permitted.
- The main results come from the **same simulator codebase and CPU platform**, not an independent lab or physical RF receiver. The 60 scored bits within a frame and many frames within a training seed are correlated; **the training seed is the unit of model-replication uncertainty**. This exploratory study does not justify a cross-domain neural-superiority claim.
- B3 prespecified an EVM measure; in fact the neural outputs are arbitrary BCE logits and cannot be interpreted as calibrated physical constellation values. The saved field `scale_aligned_symbol_mse` is therefore an *uncalibrated decision-proxy* and must **not** be called conventional RF EVM. Model-channel MSE is the frozen pilot estimator's performance, not a separately learned channel regressor. True RF EVM, activation memory, FLOPs and hardware power were **not** determined; report as unfulfilled instrumentation rather than fabricated values.
- The nonlinear PA classical baseline family is intentionally incomplete; no direct transceiver measurement has been made. Do not claim neural models outperform non-oracle signal processing just because a single chosen traditional method fails in some cells. Here the classical fixed estimator performs better anyway.

## Conclusion and stage gate

**P3-B completed with a negative result for the tested learned residual heads.** Four isolated ordinary receive-link deviations were reproducibly simulated; mild-vs-strong severities and receiver behavior were independently audited. No neural configuration demonstrated universal gains over the fixed classical comparators; the P3-A2 frozen model did not gain materially from training a lightweight 2,090-parameter data-aware head. Clean and mixed training differences were small and inconsistent. The extra anchors can be a liability in otherwise unrotated data, while they are important for correcting the simulated data-only common-phase error.

This result is conditional on a 32-subcarrier uncoded ideal QPSK experiment with random pilots, 2 public anchors, finite training and simulation biases. It is **not** a theorem that learned nonlinear receivers never help, and does not invalidate the earlier P2-E results. Do not reuse or revise P2-E or P3-A/P3-A2 frozen hypotheses; do not open P3-C without a separately stated protocol.

## Reproduction

```bash
python -m unittest test_p3b_link -v
python p3b_train_eval.py --mode train --seeds 48101 48102 48103 48104 48105 --steps 600 --batch 64 \
  --old-dir p3a2_checkpoints --ckpt p3b_checkpoints --out p3b_results
python p3b_train_eval.py --mode eval --seeds 48101 48102 48103 48104 48105 --frames 768 \
  --old-dir p3a2_checkpoints --ckpt p3b_checkpoints --out p3b_results
python p3b_train_eval.py --mode summary --out p3b_results
python p3b_train_eval.py --mode verify --seeds 48101 48102 48103 48104 48105 \
  --old-dir p3a2_checkpoints --ckpt p3b_checkpoints --out p3b_results
python p3b_privileged_reference.py
python p3b_independent_audit.py
python p3b_timing_extended.py
python p3b_make_report.py
```

`p3a2_checkpoints` are included in the separate complete offline archive; they can be regenerated from the earlier P3-A2 protocol. Newly trained `p3b_checkpoints` are also archived offline, not committed to the lightweight public GitHub repository.
