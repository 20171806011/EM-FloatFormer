# EM-FloatFormer P2 Protocol — physics-guided receive-only QPSK/OFDM

**Written before any P2 training or held-out P2 evaluations. Date: 2026-11-15.**

Scope: synthetic, passive receive-only, civilian QPSK/OFDM, 32 subcarriers, one known all-subcarrier pilot + one unknown QPSK payload symbol, cyclic prefix assumed adequate, static multipath pair, complex AWGN. No RF emitters, radar, adversarial electronic operations, weapons, or transmitter-control instructions. This is deliberately NOT an operational receiver or field-tested model.

## Freeze points and hypotheses

- P2-A: Three small residual channel estimators (pointwise gate, local complex residual, delay-domain unfolded shrinkage), all initialized with an LS-compatible fallback. Training only on three-tap power profile `(0.72,0.20,0.08)`, 0–20 dB uniform SNR. Compare with pilot LS, fixed 8-tap DFT smoothing (physical generic support assumption), LMMSE with training-profile prior, matched-test-profile LMMSE (privileged and explicitly marked), and P1 circular CNN (retrained control). Primary exploratory question: do any models reduce BER at 10 dB in held-out equal-4 and long-6 profiles versus P1 CNN and pilot LS? Failure triggers diagnosis, not automatic architecture escalation.
- P2-B: Hold architecture fixed and test raw rectangular IQ, log-amplitude/phase and dual-path RMS/log-scale representations. Compare same total source information, same train/eval samples, and identical step/seed budgets. Check encode/decode precision and whether normalized-IQ-side-input improves BER; no assumptions of dominance.
- P2-C: Prove and numerically test *global common-phase equivariance* of complex-linear and delay-shrinkage estimators, plus bit invariance when pilot AND data both rotate. Compare constrained/relaxed models at same budget; assess symmetry separately from BER. If no gain from attention, omit attention and document the stopping rule.
- P2-D: Freeze weights and evaluate distinct multipath profiles, SNR 0/10/20 dB, noisy variance reporting ±6 dB, BF16 IQ storage quantization, and data-only phase error 0.35 radians. No sample-rate claims (the simulator has no explicit physical time sampling model). Include invariant pre-checks and no test-aware tuning.
- P2-E: After A–D report, freeze a *single candidate* (chosen on exploratory A–D only) and comparison to P1 CNN log-phase with **10 fresh training seeds** plus traditional baselines, 3 conditions x 3 SNRs; two primary endpoints: BER difference on unseen equal4 and long6 at 10 dB. Exact two-sided paired sign-flip test over training seeds and Holm adjustment. Test seeds disjoint from training. Report effect size, per-seed variation, compute budget, latency batch1/64, and SHA256 of output. No optional tuning after endpoints are revealed.

## Fixed implementation choices

- Seed sets for exploratory P2-A/B/C: `1201,1202,1203`. New final P2-E seeds: `2201..2210`. Training 600 steps, batch64, AdamW learning rate 0.002 with 1e-4 weight decay, gradient clipping 1.0, CPU 4 threads. Train3 only; testing profiles `train3`, `equal4`, `long6`, with 0,10,20 dB.
- Test frames per seed/condition: exploratory 768; final 2048. A common held-out RNG stream per seed/scenario for all comparison receivers. Training RNG separate from evaluation. Sampling unit for confirmatory statistics is training seed.
- Main loss: BCE on QPSK hard-bit target through differentiable channel equalization. Optional channel-estimation squared-error diagnostic, not a training supervision loss. No access to true H in the neural forward path.
- All hyperparameter choices and architectural differences are logged. Diagnostic failures are included; no guarantees of BER improvement or p<0.05.
- Performance gate for advancing one model as strongest candidate: lower paired mean BER than P1 CNN at 10 dB on either held-out profile without catastrophic 10 dB train3 regression (>2 percentage points), but if no candidate passes, still complete B/C/D on the safest physics baseline and record no architectural win. P2-E tests only ONE frozen candidate and predeclares how it was chosen in P2-D report.

## Interpretation guardrails

- Gaussian MMSE channel posterior mean is optimal under MSE for the matching linear-Gaussian model; it is NOT a universal BER optimum.
- LS residual `Hhat=Hls+gate*delta` gives an LS fallback when the gate is zero but DOES NOT guarantee BER >= (or <=) LS in other settings.
- Common rotation equivariance does not imply robustness to independent data-only phase error, nor arbitrary frequency or sampling-rate changes.
- P1 numbers are historical measurements, not fresh P2 results. Final statistical significance is unguaranteed; show exact p values even for negative results.
