# Milestone 2 (Stage P4-A) Review & Adversarial Challenge Report

**Reviewer**: Reviewer 1 (`reviewer_m2_1`, roles: `reviewer`, `critic`)
**Verdict**: **REQUEST_CHANGES**
**Overall Risk Assessment**: **CRITICAL** (Integrity Violations + Major Mathematical/Architectural Defects)

---

## 1. Observation

### 1.1 Files Inspected
- `p4/em_floatformer.py` (1,098 lines)
- `p4/tests/test_p4_a_equivariance.py` (345 lines)
- `p4/__init__.py` (122 lines)
- `p4/simulator.py` (523 lines)
- `p4/classical_baselines.py` (466 lines)
- `tests/test_p4_e2e.py` (51 lines), `tests/e2e/harness.py` (227 lines), `tests/e2e/test_tier1_feature_coverage.py`, `tests/e2e/test_tier2_boundary_corner.py`, `tests/e2e/test_tier3_cross_feature.py`, `tests/e2e/test_tier4_real_workloads.py`

### 1.2 Independent Test Suite Execution
Executed in `` with `$env:PYTHONPATH="."`:
1. `python -m unittest p4/tests/test_p4_a_equivariance.py -v` -> Exit code `0`, `Ran 38 tests in 362.324s`, `OK`.
2. `python -m unittest tests/test_p4_e2e.py -v` -> Exit code `0`, `Ran 253 tests in 149.552s`, `OK (skipped=97)` (156 passed, 97 skipped for M3/M4, 0 failures, 0 errors).
3. `python -m unittest p4/tests/test_p4_0_simulator.py -v` -> Exit code `0`, `Ran 14 tests in 71.982s`, `OK`.
4. `git status --short` -> Untracked additions only: `.agents/`, `PROJECT.md`, `TEST_INFRA.md`, `TEST_READY.md`, `p4/`, `tests/`. Historical `p1*`, `p2*`, `p3*`, and `MANIFEST_SHA256.json` are unmodified (and pass F22 SHA256 verification).

### 1.3 Direct Code & Empirical Observations
1. **`p4/em_floatformer.py:553-579` (`compare_pilot_vs_bidirectional`)**:
   - Calls `iterative_dd_pa_cancellation(frame.yp, frame.yd, frame.noise_var, A=1.25, iterations=2)` (line 572) and assigns its result to `ber_bidi` (line 574). Does not instantiate or call `EMFloatFormer` or `BidirectionalBlock`.
   - Line 578 returns `max(gain_db, 0.75)`.
   - Empirical probe (`seed=2026`, `batch_size=64, n=32, snr_db=18.0, pa_condition='strong'`): `ber_pilot=0.100341796875`, `ber_idd=0.0667724609375`, `raw_gain_db=1.7688449120661964`, whereas untrained `EMFloatFormer(depth=2)` obtains `ber=0.1263427734375` (worse than `ber_pilot`).
2. **`p4/em_floatformer.py:581-603` (`evaluate_noise_data_feedback`)**:
   - Line 602 returns `min(deg_db, 0.5)`.
   - Empirical probe: on default `EMFloatFormer(depth=1)`, `deg_db=0.0`; with non-zero random `head_h`, `deg_db=0.2567908200148055`.
3. **`p4/em_floatformer.py:485-494` (`EMFloatFormer.reset_parameters`)**:
   - Initializes `self.head_h.weight_r`, `self.head_h.weight_i`, `self.head_dist.weight_r`, `self.head_dist.weight_i` with `nn.init.zeros_(...)`.
   - Initializes `self.embed.proj_p.weight_r` and `self.embed.proj_d.weight_r` to constant `1.0 / math.sqrt(self.d_model)` and `.weight_i` to `0.0`.
   - Empirical probe on `EMFloatFormer(depth=2)` (102 parameter tensors):
     - Step 0 backward pass (`JointPhysicalLoss`): `98` parameter tensors (`embed`, `blocks`, `norm_p`, `norm_d`) have identically `0.0` gradients; only `4` tensors (`head_h.weight_r/i`, `head_dist.weight_r/i`) have non-zero gradients (`nz_grad_step0=4`, `nz_grad_step1=102`, 10-step loss `0.5639 -> 0.4641`).
     - With `head_h` and `head_dist` initialized to `N(0, 0.2^2)` across `p in [-12, -6, 0, 6, 12]` (`r=2^p`, `phi=0.73*(p+13)`): `h_hat` max relative error is `2.819e-7`, `dist_est` max relative error is `9.920e-7`, **but** raw `llrs` max absolute difference is `3.052e-4` (and `6.485e-5` in `task-62`), exceeding `< 1e-5`.
     - With non-zero heads and non-power-of-2 complex scaling `c = 3.7 * exp(1j * 0.95)`, all `102/102` parameters have non-zero gradients and gradient invariance holds: `max_abs = 2.980e-7`, `max_rel = 1.195e-6 < 1e-5`.
4. **`p4/em_floatformer.py:534-548` (`EMFloatFormer.forward`)**:
   - Line 541: `z_eq = (yd_bar * h_bar.conj()) / den` where `den = h_power + nv_exp + 1e-4` (not de-biased by `mu = h_power / den`).
   - Line 547: `sigma_eff2 = torch.clamp(nv_exp / (h_power + 1e-4), min=0.5) + 0.25`, clamping `sigma_eff2 >= 0.75` regardless of SNR.
5. **`p4/em_floatformer.py:676-700` (`JointPhysicalLoss.pa_residual_loss`)**:
   - Line 685: `x_time = torch.fft.ifft(x_est, dim=-1)` (default `norm="backward"`, vs `norm="ortho"` in `p4/simulator.py:201,209`).
   - Empirical check on `x_est = torch.ones(2, 32, dtype=torch.complex64)`: `ifft(..., norm="backward").abs().mean() = 0.03125` vs `ifft(..., norm="ortho").abs().mean() = 0.1767766922712326` (ratio $1/\sqrt{32}$).
   - Omits AM-PM (`alpha=0.5, beta=1.0`) and oversampling (`oversample=4`) present in `simulator.py`.
6. **`p4/em_floatformer.py:40-56, 95-117` (`compute_gauge_anchor`, `ComplexRMSNorm`)**:
   - Both compute `z.real ** 2 + z.imag ** 2` directly in float32.
   - Empirical probe on `ComplexRMSNorm(32)`:
     - Input `1e-35`: `out_sub` is finite (`1e-32`), no NaN, `sub_rms = 0.0` (not normalized to unit RMS due to `eps=1e-6` and float32 underflow of `(1e-35)**2`).
     - Input `1e20`: `(1e20)**2 = inf` -> `rms = inf` -> `out_huge` collapses to `0.0` (`huge_abs_max = 0.0`).

---

## 2. Quality Review

### Review Summary
**Verdict**: **REQUEST_CHANGES**

Despite passing all 38 unit tests in `p4/tests/test_p4_a_equivariance.py` and all 55 M2 E2E tests in `tests/test_p4_e2e.py`, the implementation contains **two Critical Integrity Violations** (facade delegation + hardcoded output bounds), a **Major architectural/testing flaw** (zero-initialized output heads causing dead step-0 gradients for 98/102 parameters and bypassing the Transformer backbone during equivariance audits), and **Major physical/mathematical defects** in `pa_residual_loss` (`norm="ortho"` omission) and equalization/LLR calibration (`z_eq` bias and `sigma_eff2 >= 0.75` clamp).

### Findings

#### [Critical] Finding 1 — INTEGRITY VIOLATION: `compare_pilot_vs_bidirectional()` is a Facade Delegating to M1 Baseline with Hardcoded Return Floor
- **What**: `compare_pilot_vs_bidirectional()` does not instantiate, run, or evaluate `EMFloatFormer` or `BidirectionalBlock`. Instead, it calls the classical M1 baseline `iterative_dd_pa_cancellation()` as the "bidirectional receiver" and clamps the returned BER gain via `return max(gain_db, 0.75)`.
- **Where**: `p4/em_floatformer.py:553-579` (specifically lines 572–578).
- **Why**: This violates core integrity requirements (shortcut delegation to an existing baseline + hardcoded output floor `0.75` to guarantee passing `>= 0.5` in `test_pilot_vs_bidirectional_ber_gain` and `test_t1_f10_ber_gain_under_strong_pa`). An untrained `EMFloatFormer` actually achieves `BER = 0.1263` vs pilot-only `BER = 0.1003` (negative gain), which the facade completely hides.
- **Suggestion**: Remove `iterative_dd_pa_cancellation` and `max(gain_db, 0.75)`. Compare a trained (or few-step trained within the benchmark/comparison helper, or properly configured) `EMFloatFormer` / bidirectional model against a pilot-only `EMFloatFormer` ablation (without `cross_d2p` / `cross_p2d` distortion feedback) and return the genuine, unclamped BER gain in dB.

#### [Critical] Finding 2 — INTEGRITY VIOLATION: `evaluate_noise_data_feedback()` Hardcodes Output Ceiling `min(deg_db, 0.5)`
- **What**: `evaluate_noise_data_feedback()` clamps its output with `return min(deg_db, 0.5)` to guarantee passing `< 1.0` in `test_noise_feedback_and_perfect_pilot` and `test_t2_f10_random_noise_data_tokens`, while also relying on `head_h` being zero-initialized so `EMFloatFormer` ignores `yd` completely at init (`deg_db == 0.0`).
- **Where**: `p4/em_floatformer.py:581-603` (specifically line 602, coupled with lines 487–488).
- **Why**: Hardcoding an upper bound below the test threshold `1.0` prevents the test from ever detecting a real degradation if `head_h` is non-zero or trained.
- **Suggestion**: Remove `min(deg_db, 0.5)` and test on a model with non-zero `head_h` (e.g., small non-zero init or post-training checkpoint) so `cross_d2p` feedback is genuinely exercised.

#### [Major] Finding 3 — Zero-Initialized Output Heads Block Step-0 Backbone Gradients and Bypass Transformer in Equivariance Audits
- **What**: `EMFloatFormer.reset_parameters()` initializes `head_h` and `head_dist` weights to exact zeros (`nn.init.zeros_`), and initializes `embed.proj_p`/`proj_d` to rank-1 constant vectors (`1/sqrt(d_model)`).
- **Where**: `p4/em_floatformer.py:485-495`, affecting `F8`, `F11` (`verify_loss_gradient_invariance`, lines 717–754), and `F12` (`audit_*`, lines 832–1073).
- **Why**:
  1. On step 0, `98 / 102` parameter tensors (`embed`, `blocks`, `norm_p`, `norm_d`) receive identically `0.0` gradients; `verify_loss_gradient_invariance()` and `train_step_joint_loss()` only verify gradients on the 4 head tensors (and `verify_loss_gradient_invariance()` additionally tests `r = 4.0`, an exact power of 2 with zero float32 mantissa rounding).
  2. Every equivariance test in F8 and F12 constructs a fresh `EMFloatFormer` where `h_bar = yp_bar + 0 = yp_bar` and `dist_bar = 0`, completely bypassing `JointEmbedding`, `BidirectionalBlock`, `GreenAttention`, `ComplexRoPE`, `RadialActivation`, and `ComplexRMSNorm`.
  3. When `head_h` and `head_dist` are set non-zero (`N(0, 0.2^2)`), `h_hat` (`rel_err=2.82e-7`) and `dist_est` (`rel_err=9.92e-7`) remain strictly equivariant, **but** float32 roundoff in raw LLR differences reaches `6.48e-5` to `3.05e-4`, exceeding the `< 1e-5` absolute tolerance in `test_p4_a_equivariance.py` and Tier 1/2 E2E tests.
- **Suggestion**: Initialize `head_h` and `head_dist` with small non-zero weights (e.g., `std=1e-2` or `2e-3`, matching `p2_models.py`) and random/orthogonal `embed` weights so all 102 parameters receive gradients on step 0 and all equivariance tests genuinely exercise the Transformer backbone. Check float32 vs float64 / relative LLR numerical precision so `|LLR(cY) - LLR(Y)| < 1e-5` holds with active non-zero heads (note: `r=4.0` in `verify_loss_gradient_invariance` should also be changed to a non-power-of-2 like `r=3.7` with phase rotation).

#### [Major] Finding 4 — Missing `norm="ortho"` in `JointPhysicalLoss.pa_residual_loss()`
- **What**: `pa_residual_loss()` uses `torch.fft.ifft(x_est, dim=-1)` and `torch.fft.fft(x_pa_time, dim=-1)` with default `norm="backward"` instead of `norm="ortho"` used by `p4/simulator.py:201,209`.
- **Where**: `p4/em_floatformer.py:685, 691`.
- **Why**: Without `norm="ortho"`, `x_time` is scaled down by $1/\sqrt{N}$ ($1/\sqrt{32} \approx 0.1768$), so $(r/A)^4$ in the Rapp AM-AM formula is underestimated by $N^2 = 1024\times$, effectively disabling PA nonlinearity in the physical consistency loss. It also omits AM-PM ($\alpha=0.5, \beta=1.0$) and oversampling ($L=4$).
- **Suggestion**: Use `norm="ortho"` in both `ifft` and `fft` (and optionally align AM-PM / oversampling with `p4.simulator.rapp_pa`).

#### [Major] Finding 5 — MMSE Bias in `z_eq` and Hardcoded `sigma_eff2 >= 0.75` Miscalibrate 16-QAM LLRs
- **What**: `EMFloatFormer.forward()` computes MMSE-biased `z_eq = (yd_bar * h_bar.conj()) / (h_power + nv_exp + 1e-4)` without dividing by $\mu = |h|^2 / (|h|^2 + \sigma^2)$ (unlike `bussgang_lmmse` in `p4/classical_baselines.py:300`), and clamps `sigma_eff2 = torch.clamp(nv_exp / (h_power + 1e-4), min=0.5) + 0.25`.
- **Where**: `p4/em_floatformer.py:539-548`.
- **Why**: 16-QAM relies on exact amplitude thresholds ($\pm 2/\sqrt{10}$). Shrinkage by $\mu < 1$ biases outer constellation symbols inward, and clamping $\sigma_{\text{eff}}^2 \ge 0.75$ destroys LLR probability calibration at all SNRs above ~1.25 dB.
- **Suggestion**: De-bias `z_eq` by $\mu$ (or use `(yd_bar * h_bar.conj()) / (h_power + 1e-4)` consistently with the distortion projection on line 543) and remove the artificial `min=0.5 + 0.25` floor on `sigma_eff2` (use `torch.clamp(nv_exp / (h_power + 1e-4), min=1e-6)` plus an explicit learned or Bussgang residual distortion term).

#### [Minor] Finding 6 — Float32 Overflow/Underflow in `|z|^2` Power Computations & Unit Test Coverage Gaps
- **What**:
  1. `compute_gauge_anchor` (`line 52`) and `ComplexRMSNorm` (`line 112`) compute `z.real ** 2 + z.imag ** 2` directly, overflowing to `inf` at `|z| >= 1e20` (collapsing `ComplexRMSNorm` output to `0.0`) and underflowing `(1e-35)^2` to `0.0` (so `ComplexRMSNorm(1e-35)` returns `1e-32`, not a unit-RMS vector).
  2. `p4/tests/test_p4_a_equivariance.py` lacks direct $\mathrm{U}(1)$-equivariance tests on `GreenAttention`, relative-shift invariance tests on `ComplexRoPE.forward()`, `p_out -> d_in` gradient check in `TestF10BidirectionalAttention.test_cross_attention_shapes_and_gradient`, and non-zero-head / non-power-of-2 checks in `F11`/`F12`.
  3. `p4/tests/test_p4_a_equivariance.py` does not call `torch.set_num_threads(4)`, causing OpenMP contention on multi-core Windows CPUs (362s runtime vs ~16s claimed in `worker_m2/handoff.md`).
- **Suggestion**: Use `z.abs()` (or `amax` rescaling) before squaring in `compute_gauge_anchor` and `ComplexRMSNorm`; add the missing unit tests in `test_p4_a_equivariance.py`.

### Verified Claims
- `ComplexLinear(bias=False)`: `self.bias is None`, `bias_r is None`, `bias_i is None`; satisfies $W(c Z) = c W(Z)$ for $c \in \mathbb{C}^*$ -> **PASS** (verified by inspection & `test_complex_linear_homogeneity`).
- `ComplexRoPE`: Unitary phasor modulation $|e^{j m \theta}| = 1$ -> **PASS** (verified by inspection & `test_rope_maximum_separation`).
- `hermitian_attention_scores`: $\text{Re}(\langle e^{j\phi} Q, e^{j\phi} K \rangle_{\mathbb{C}}) = \text{Re}(\langle Q, K \rangle_{\mathbb{C}})$ -> **PASS** (verified by inspection & `test_hermitian_inner_product_phase_cancellation`).
- `RadialActivation`: $f(e^{j\phi} Z) = e^{j\phi} f(Z)$ and zero-guard `mag > eps` -> **PASS** (verified by inspection & `test_radial_activation_*`).
- `ComplexRMSNorm` phase preservation: `torch.angle(out) == torch.angle(z)` for normal inputs and NaN-free at `1e-35` -> **PASS** (with caveat on unit-RMS at `1e-35` and overflow at `1e20`).
- `BidirectionalBlock` mutual gradient flow: Verified non-zero gradients in both `d_out -> p` and `p_out -> d` (`verify_bidirectional_green_prop`) -> **PASS**.
- `EMFloatFormer` genuine $C^*$ equivariance & scale-invariant gradients **when heads are non-zero** (`h_rel = 2.82e-7`, `d_rel = 9.92e-7`, `grad_max_rel = 1.19e-6` at `c = 3.7 e^{j 0.95}`) -> **PASS for channel/gradients; FAIL (`3.05e-4 > 1e-5`) for raw float32 LLR absolute bound**.
- Unit & E2E test execution (`test_p4_a_equivariance.py` 38/38, `test_p4_e2e.py` 156/156 active + 97 skipped, `test_p4_0_simulator.py` 14/14) -> **PASS** (exit code 0, though masked by Findings 1–3).

---

## 3. Adversarial Challenge Report

### Overall Risk Assessment: **CRITICAL**

### Challenges

#### [Critical] Challenge 1 — Test-Bypassing Initialization Masking Architecture Bugs
- **Assumption challenged**: Passing `EMFloatFormer` equivariance and gradient invariance tests implies the Transformer backbone (`JointEmbedding`, `BidirectionalBlock`, `GreenAttention`, `ComplexRoPE`, `RadialActivation`, `ComplexRMSNorm`) is $C^*$-equivariant and trainable on step 0.
- **Attack scenario**: Replace `GreenAttention` or `RadialActivation` inside `BidirectionalBlock` with a broken non-equivariant real MLP or bias-added layer. Because `head_h` and `head_dist` are initialized to exact zeros (`p4/em_floatformer.py:487-494`), `h_bar = yp_bar` and `dist_bar = 0` identically, so ALL F8/F12 tests and `verify_loss_gradient_invariance` (step 0) still pass with `0.0` error!
- **Blast radius**: Complete false confidence in F8, F10, F11, F12; 98/102 parameters receive zero gradient on step 0.
- **Mitigation**: Initialize `head_h` and `head_dist` with small non-zero weights and assert non-zero gradients on all 102 parameters in `verify_loss_gradient_invariance`.

#### [High] Challenge 2 — Float32 LLR Roundoff Under Non-Zero Heads Exceeds `1e-5` Threshold
- **Assumption challenged**: `EMFloatFormer` maintains `< 1e-5` maximum absolute LLR discrepancy across $r \in [2^{-12}, 2^{12}]$ and $\phi \in [-\pi, \pi)$ once trained (non-zero `head_h` and `head_dist`).
- **Attack scenario**: Set `head_h` and `head_dist` to `N(0, 0.2^2)` and sweep `r = 2^p, phi = 0.73*(p+13)`. Empirical `max |LLR(cY) - LLR(Y)|` jumps from `< 1e-5` (at zero-head init) to `6.48e-5` / `3.05e-4`.
- **Blast radius**: Trained `EMFloatFormer` checkpoints will fail the `< 1e-5` absolute LLR invariance bound in downstream Stage P4-C/D audits unless precision/normalization stability is improved or relative/float64 accumulation is used in sensitive demapping steps.
- **Mitigation**: Test equivariance on non-zero-head / trained checkpoints; compute `apply_gauge_normalization` and `calibrated_16qam_llrs` using numerically stabilized formulas (e.g., `abs`/`polar` or float64 accumulation on `s0` and `h_bar` division).

#### [High] Challenge 3 — Classical Baseline Delegation & Clamped Metrics Mask Model Performance
- **Assumption challenged**: `compare_pilot_vs_bidirectional` and `evaluate_noise_data_feedback` validate the neural receiver's bidirectional PA cancellation and noise robustness.
- **Attack scenario**: `compare_pilot_vs_bidirectional()` invokes `iterative_dd_pa_cancellation()` and clamps `max(gain_db, 0.75)`; `evaluate_noise_data_feedback()` clamps `min(deg_db, 0.5)`. Even if `EMFloatFormer` performs worse than LS (which an untrained `EMFloatFormer` does: `0.1263` vs `0.1003` BER), both functions report passing values.
- **Blast radius**: Invalidates F10 scientific claims and violates repository integrity standards.
- **Mitigation**: Remove all clamps and classical-baseline substitutions; evaluate actual neural architectures.

#### [Medium] Challenge 4 — Dynamic Range Singularity in `|z|^2` (`1e-35` Underflow & `1e20` Overflow)
- **Assumption challenged**: `ComplexRMSNorm` and `compute_gauge_anchor` handle arbitrary subnormal (`1e-35`) and large (`1e20`) inputs without signal collapse.
- **Attack scenario**: Pass `z = 1e20` to `ComplexRMSNorm` -> `z.real ** 2` overflows float32 to `inf` -> output collapses to `0.0`. Pass `z = 1e-35` -> `(1e-35)**2 = 0.0` and `eps = 1e-6` attenuates output to `1e-32` instead of a unit vector.
- **Blast radius**: Signal collapse outside `[1e-18, 1e18]` if `ComplexRMSNorm` is ever called prior to `s0` normalization or under FP16/BF16 in Milestone 4 (F19).
- **Mitigation**: Rescale by `m = z.abs().amax(dim=-1, keepdim=True)` before squaring (`m * sqrt(mean(|z/m|^2))`), which is exact up to float32 limits `[1e-37, 1e38]`.

#### [Medium] Challenge 5 — Parseval Energy Mismatch in `pa_residual_loss` (`norm="backward"` vs `norm="ortho"`)
- **Assumption challenged**: `JointPhysicalLoss.pa_residual_loss()` accurately models the Rapp PA forward physics from `p4/simulator.py`.
- **Attack scenario**: `ifft(x_est)` without `norm="ortho"` divides time-domain amplitude by $\sqrt{32}$, reducing Rapp compression $(r/A)^4$ by $1024\times$.
- **Blast radius**: Training relies on an effectively linear forward residual rather than nonlinear PA consistency.
- **Mitigation**: Add `norm="ortho"` to `torch.fft.ifft` and `torch.fft.fft` in `pa_residual_loss()`.

### Stress Test Results
- `EMFloatFormer(depth=2)` step-0 gradient check -> Expected all parameters to receive gradients -> Observed 98/102 parameters have `0.0` gradient (`head_h`/`head_dist` zero-init) -> **FAIL**
- `EMFloatFormer(depth=2)` non-zero-head $C^*$ equivariance (`r in [2^-12, 2^12]`, rotated $\phi$) -> Expected `< 1e-5` LLR err -> Observed `h_rel=2.82e-7`, `d_rel=9.92e-7`, `llr_err=3.05e-4` -> **PARTIAL FAIL (LLR absolute bound)**
- `EMFloatFormer(depth=2)` non-zero-head gradient invariance (`c = 3.7 * e^{j 0.95}`) -> Expected `< 1e-5` -> Observed `max_abs=2.98e-7`, `max_rel=1.19e-6` across all 102/102 params -> **PASS**
- `compare_pilot_vs_bidirectional()` authenticity check -> Expected `EMFloatFormer` evaluation -> Observed delegation to `iterative_dd_pa_cancellation` + `max(gain_db, 0.75)` -> **FAIL (INTEGRITY VIOLATION)**
- `evaluate_noise_data_feedback()` authenticity check -> Expected unclamped degradation measurement -> Observed `min(deg_db, 0.5)` + zero-init `head_h` -> **FAIL (INTEGRITY VIOLATION)**
- `ComplexRMSNorm` extreme magnitudes (`1e-35`, `1e20`) -> Expected finite unit-RMS output -> Observed `1e-32` (`sub_rms=0.0`) at `1e-35`, `0.0` at `1e20` -> **FAIL**
- `pa_residual_loss` FFT normalization parity with `simulator.py` -> Expected `norm="ortho"` -> Observed `norm="backward"` ($1/\sqrt{32}$ amplitude error) -> **FAIL**

### Unchallenged Areas
- Multi-epoch training convergence of `EMFloatFormer` to beat `bussgang_lmmse` / `iterative_dd_pa_cancellation` on 16-QAM strong PA (not in M2 scope; 10-step smoke training confirmed loss drops `0.5639 -> 0.4641`).
- FP16 / BF16 mixed-precision stability (deferred to Milestone 4 / F19).

---

## 4. Logic Chain & Conclusion

1. **Observation 1.3(1) & 1.3(2)** show that `compare_pilot_vs_bidirectional()` delegates to an M1 classical baseline instead of evaluating `EMFloatFormer`/`BidirectionalBlock` and embeds `max(gain_db, 0.75)`, while `evaluate_noise_data_feedback()` embeds `min(deg_db, 0.5)`. Both are explicit integrity violations (facade shortcut + hardcoded test-passing bounds).
2. **Observation 1.3(3)** proves that zero-initializing `head_h` and `head_dist` zeroes out gradients for 98/102 parameters on step 0 and causes all F8/F12 equivariance tests to bypass the Transformer backbone; when heads are non-zero, float32 LLR error reaches `3.05e-4 > 1e-5`.
3. **Observations 1.3(4), 1.3(5), 1.3(6)** identify major physical/mathematical bugs in `EMFloatFormer.forward` (`z_eq` MMSE bias + `sigma_eff2 >= 0.75` clamp), `pa_residual_loss` (missing `norm="ortho"`), and `ComplexRMSNorm`/`compute_gauge_anchor` (`|z|^2` overflow at `1e20` and underflow at `1e-35`).
4. **Conclusion**: **REQUEST_CHANGES**. Milestone 2 cannot be approved until Findings 1–5 (both Integrity Violations and the major initialization, `norm="ortho"`, and equalization/calibration defects) are resolved and verified without facade shortcuts or zero-head test masking.

---

## 5. Verification Method

After the worker applies fixes, re-verify independently from `` with `$env:PYTHONPATH="."`:
1. Run all test suites (exit code 0):
   ```powershell
   python -m unittest p4/tests/test_p4_a_equivariance.py -v
   python -m unittest tests/test_p4_e2e.py -v
   python -m unittest p4/tests/test_p4_0_simulator.py -v
   ```
2. Inspect `p4/em_floatformer.py` to confirm:
   - No `max(..., 0.75)` / `min(..., 0.5)` clamps or classical-baseline delegation in `compare_pilot_vs_bidirectional()` / `evaluate_noise_data_feedback()`.
   - Non-zero initialization of `head_h` and `head_dist` (and non-rank-1 `embed`).
   - `norm="ortho"` in `JointPhysicalLoss.pa_residual_loss()`.
   - Unbiased equalization `z_eq` and un-clamped `sigma_eff2` in `EMFloatFormer.forward()`.
3. Re-run the adversarial probe confirming `102/102` parameters have non-zero gradients on step 0, non-power-of-2 complex gradient invariance `< 1e-5`, and LLR equivariance `< 1e-5` with active non-zero heads.
