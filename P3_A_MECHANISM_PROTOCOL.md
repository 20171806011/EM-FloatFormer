# EM-FloatFormer · P3-A Mechanism Addendum — frozen protocol v1.0

**Status: frozen exploratory protocol — committed before any P3-A2 training or held-out evaluation.**  Prepared 2026-11-15. Applies only to simulated, passive civilian 32-tone QPSK/OFDM reception, with ideal synchronization, Gaussian thermal noise and one static pilot/data pair. This is a *separate exploratory follow-up*, not a modification of frozen P3-0/P3-A or P2-E results.

## 0. Question and evidence

In P3-A the new 985-parameter `EquivDelayConv` has 10 dB BER 4.863% (train3), 6.442% (long6), 11.411% (long12), 12.016% (sparse12). One seed's delay shrinkage gates averaged 0.410 at delays 0–2, 0.002 at 8–11 and 0.005 beyond 12. These are **descriptive**, not a causal identification.

Code inspection of `p3_models.py` confirms **shared convolution weights**, not 32 independent gate parameters. The gate input nonetheless includes `cos(2πk/32)` and `sin(2πk/32)` as absolute-position features. The older `res_unfold` also uses delay position but has a nonzero identity residual floor. Two potentially interacting causes must be separated: positional shortcuts and the ability to nearly erase unforeseen delayed paths.

## 1. Preselected 2×2 experiment — only two changed factors

All four arms share the *same* 985-parameter `EquivDelayConv` architecture, the same mathematical nonlinearities, channel decoder, convolution widths, kernel sizes, and all training settings. The coordinate-channel weights exist in each arm, even when their input is zero; report both nominal and active parameter counts.

| Arm | Absolute delay coordinates | Identity-retention floor `β` | Purpose |
|---|---|---:|---|
| **A: pos/no-floor** | original cosine + sine | 0.00 | Freshly retrained P3-A control; exact original forward function |
| **B: no-pos/no-floor** | both position channels forced to 0 | 0.00 | Isolate absolute-position removal |
| **C: pos/floor** | original cosine + sine | 0.10 | Isolate residual retention |
| **D: no-pos/floor** | both position channels forced to 0 | 0.10 | Joint intervention / interaction |

For delay observation `t = IFFT(Ypilot)`, current gate `g_theta(t,v,k) = sigmoid(net(features))` and estimated delay taps `hhat_tau = s_tau * t_tau`, define **fixed** `s_tau = β + (1-β)g_theta`, with `β ∈ {0,0.10}`. **β is not trainable** and cannot be selected after inspecting held-out BER. This retains U(1) common-phase equivariance in all arms. Replacing position-channel inputs with zeros, rather than deleting dimensions, preserves tensor/weight shapes and common initialization. It makes the position-free networks circular-delay-translation equivariant in the mathematical model; this does **not** imply invariance to arbitrary permutations of taps.

The `β=.10` floor is deliberately a modest, preselected intervention rather than a claimed optimum. No other floor values or new loss terms enter this addendum.

## 2. Data, training and held-out scenarios

- **Existing simulator unchanged** (`ofdm_baseline.make_batch`): `N=32` full known pilots, 32 QPSK data carriers, one pilot/data pair per frame, complex Gaussian noise; training profile `train3=(.72,.20,.08)` only. The receiver never uses true channel `H` as an input.
- **Five fresh seeds:** `47101,47102,47103,47104,47105`. Never use P3-A seeds `46001–46005`, P2-E seeds `2201–2210`, or supplement seeds `34001..36003` for model selection. For each seed, initialize one base state dict and deep-copy it to all four arms; feed precisely the same minibatches to all arms. This explicitly removes the original training script's model-name-dependent initialization difference as a possible confound.
- **Budget per model:** 600 AdamW steps, batch64, `lr=.002`, weight decay `1e-4`, BCEWithLogits QPSK loss, gradient clip norm 1, training SNR uniformly 0–20 dB. No architecture or optimization changes after a pilot smoke test; the smoke test must not consume the held-out test sets.
- **Held-out:** `train3`, `equal4`, `long6`, `long12`, `sparse12`, with exactly the same P3-A profile definitions, at SNR `0,10,20` dB. Evaluate **768 independent frames per seed/scenario** on new disjoint evaluation RNG streams, identical across all arms/baselines. With 64 QPSK bits/frame this yields **245,760 evaluated bits per receiver/scenario** pooled across five training seeds. Never use test errors or profile labels for choosing weights.
- Retain same split/cyclic-prefix/ideal synchrony assumptions as P3-A; do not silently enlarge the dataset or add nonideal RF impairment factors here.

## 3. Fixed classical references and main measurements

For every scenario and identical input frames report LS, DFT-8, hard and soft threshold DFT with P3-A's **unchanged threshold**, train3-prior LMMSE; label DFT-12 (known extended support), matched-test-profile LMMSE and true-channel output **privileged / non-deployable diagnostic** where appropriate. No threshold fitted on test data. Include both BER and channel-MSE; do not compare the learned models only against an intentionally misspecified DFT-8.

**Primary exploratory cells (frozen in advance):** `long12@10dB` and `sparse12@10dB`. Record six paired contrasts: `B−A`, `C−A`, `D−A` in BER percentage points for each cell. Descriptive factorial interaction: `I=(BER_D−BER_C)−(BER_B−BER_A)`; this distinguishes whether removing position has a different effect under floor retention. Also report `train3@10dB`, `long6@10dB` and *every* 0/10/20dB cell, whether favorable or unfavorable.

**Mechanism diagnostics** (true channel only for post-hoc **measurement**, never fed into receiver): mean/quantile shrinkage gates for real versus noise-only taps, grouped by delay index and relative input energy; energy retained on true late taps; channel MSE, normalized MSE and optional positive-real-scale-aligned MSE (the scale factor is computed for analysis only); CPU latency/p95 and inferred FLOPs where feasible. Record each seed separately; **the statistical unit is the independent training seed**, not the correlated bit count. With only five seeds, treat all conclusions as exploratory; do not hunt for p<0.05 or claim a strict causal explanation outside this controlled model family.

## 4. Independent implementation/integrity checks

Before training: (i) A with `β=0` exactly matches the original `EquivDelayConv.forward/channel` for identical weights/inputs; (ii) all arms preserve common-phase U(1) channel equivariance and invariant decoded logits to floating tolerance; (iii) B and D pass a circular delay-shift equivariance test when the entire tap vector is shifted; (iv) test no true-H leakage, finite gradients, paired initialization and identical batch streams; (v) verify classical estimator outputs and the unchanged hard-threshold formula.

After training: save all 20 checkpoint SHA256 hashes and training logs; verify at least one previously recorded test scenario per seed by loading the saved checkpoints anew and recomputing **integer error counts**, not just decimal BER; recompute aggregates independently from raw CSV; inspect the full 2×2 factorial rather than selectively showing the best arm. Preserve failures and report any CPU/precision limitations.

## 5. Interpretation and stop rule

- If B and D improve long-delay BER, this is evidence that absolute-position inputs contribute to failure **in the tested family**, not proof of the sole cause.
- If C and D help despite retention of coordinates, near-zero gain suppression is another plausible causal mechanism.
- If only D helps, interaction between the two modifications matters; if none helps, record the negative result and investigate training-data support and model mismatch instead.
- If the model improves OOD BER but worsens training channels, MSE, phase reconstruction or latency, report tradeoffs. Do not state that the result must reach the hard-threshold DFT's ≈4.857% sparse12 baseline.

**Deliverables for this execution:** `P3_A_MECHANISM_PROTOCOL.md` (frozen before training), `p3_a_mechanism.py`, `test_p3_a_mechanism.py`, per-seed raw counts/weights/hashes, analysis graphics, `P3_A_MECHANISM_REPORT.md`, and a separate scope addendum. Append a new commit to the *existing* `20171806011/EM-FloatFormer` repository without rewriting P3-A/P2-E frozen evidence. **The original P3-A and P2-E reports are immutable; no new P3-B work belongs to this protocol.**