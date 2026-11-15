# EM-FloatFormer P3-0 / P3-A — frozen exploratory protocol

**Scope:** 2026-11-15, simulated, receive-only civilian QPSK/OFDM. No radar, active interference, emitting devices, or transmitter/weapon control. This protocol is deposited before P3-A training and held-out evaluation. P2-E and P2 supplements remain unchanged.

## P3-0: mathematical audit and hypothesis

- Baseline `res_equiv_local` in `p2_models.py` uses a nine-coefficient complex circular *frequency-domain* convolution, plus learned input-dependent invariant gates. Its *fixed linear convolution component* maps under IFFT to a delay-wise multiplication whose frequency kernel has support nine. It cannot exactly implement the 32-subcarrier rectangular first-eight-delay mask (the exact equivalent frequency convolution kernel has 25 nonzero elements). The full nonlinear network is not bounded by this fixed-kernel approximation theorem.
- Baseline `res_unfold` already performs IFFT -> learned real delay-wise shrinkage -> FFT, with 722 parameters; P3-A must compare against it directly, not claim this operation is new.
- Testable hypothesis: allowing a small local real-valued gating convolution **in the delay domain**, while maintaining common-phase equivariance, may improve held-out channel BER or channel MSE relative to P2 receivers, but may still lose to classical delay-domain estimators. Do not assume a neural win.

## Models and training — fixed before evaluating

- Use the unchanged `ofdm_baseline.make_batch` ideal-frequency-domain generator: 32 known pilot subcarriers (Xp=1), 32 QPSK data carriers, a pilot-data-static channel, ideal synchronization, complex Gaussian thermal noise, unencoded QPSK.
- Retrain **three** models from scratch using the *same frames per seed*: `p3_equiv_delay` (new, <3000 parameters, IFFT/invariant real circular 1D gate/FFT); `res_unfold` (existing 722-parameter P2 delay shrinker); `res_equiv_local` (existing 2060-parameter P2 complex circular FIR). No P2-E checkpoints reused.
- Training distribution **train3** `(.72,.20,.08)`; 5 distinct new training seeds **46001–46005**, 600 steps/seed, batch64, AdamW lr0.002 weight_decay1e-4, BCEWithLogits loss, gradient clipping norm 1, SNR sampled uniformly 0..20 dB. Freeze architecture, loss and hyperparameters before test. The new model may depend on `Ypilot` and reported variance only for channel estimation; any `Ydata` is used only downstream for QPSK equalization.
- Testing: five channel profiles `train3`, `equal4`, `long6`, `long12`, `sparse12`, at **0, 10, 20 dB**, **768 independent held-out frames/seed/condition**, same frames across receivers, 64 bits/frame, **245760 bits per receiver per scenario**. New test PRNG family disjoint from training and previous P1/P2 seeds. `long12` is a contiguous 12-delay power profile; `sparse12` has active delays separated by zeros extending to index 11. These test profiles are not in train3.
- All reported primary comparisons are **exploratory**, not an additional P2-E confirmatory endpoint. Preselect descriptive contrasts on long6@10dB and long12@10dB: BER(new) minus BER(res_unfold) and minus BER(best of nonprivileged DFT-8 and predetermined DFT-threshold). Report exact pooled bit counts, per-seed effects, channel estimation MSE, and results on all scenarios; no significance-threshold winner-picking.

## Classical controls (all pilot-only)

1. Pilot LS `Hhat=Ypilot`.
2. **DFT-8**: IFFT, retain delays 0..7, zero remaining taps, FFT. Assumes support max 8; this information is correct for train3/equal4/long6 but **misspecified** for long12/sparse12. Describe advantage.
3. **DFT-12**: same but 12 taps, labeled *known-extended-support reference*, not an equal-support claim against DFT-8.
4. **DFT threshold hard / soft**: for each of 32 delay taps, use reported noise variance `v=N0/32` and fixed threshold `T=sqrt(v*ln(32/0.05))` (expected 0.05 noise-only threshold crossings per frame under independent circular-Gaussian samples). Hard: `t * 1[abs(t)>T]`; soft: `t * max(0,1-T/max(abs(t),eps))`. No knowledge of channel path support, test profile, or test bits; fixed before any test.
5. LMMSE with train3 covariance (nonprivileged training-distribution prior); matched-profile LMMSE clearly labeled *privileged* reference; true-channel decisions only as inaccessible diagnostic.
- Use the SAME observed samples and noise variance per scenario for all methods; do not tune the threshold on held-out data.

## Verification / publishing

- Before using trained model data: verify convolution theorem with independent FFT construction; nine-tap fixed-kernel projection error; common `U(1)` phase-equivariance in arbitrary and trained models; differentiability and finite gradients; classical estimator implementation against independent direct/delay formulation; no ground-truth-channel leakage; independent test seeds.
- Save all model weights locally (not GitHub), original per-seed integer errors/bits, channel MSE, timing, SHA256 per checkpoint, training/eval configs, test reports. Count training seeds as sampling units; frame/bit counts do not represent independent training replications. No opportunistic training-extension or post-hoc threshold changes.
- Only after validation, publish files into existing `20171806011/EM-FloatFormer` main branch without changing frozen P2 reports. If some tests fail or cannot run, label incomplete and disclose why.
