# Milestone 2 (Stage P4-A) Mathematical & Theoretical Equivariance Review Handoff (`reviewer_m2_3`)

**Reviewer**: Reviewer 3 (`reviewer_m2_3`, roles: `reviewer`, `critic`)
**Verdict**: **APPROVE**
**Overall Risk Assessment**: **LOW**

---

## 1. Observation

### 1.1 Files Inspected
- `p4/em_floatformer.py` (1,225 lines)
- `p4/__init__.py` (124 lines)
- `p4/tests/test_p4_a_equivariance.py` (462 lines)

### 1.2 Independent Test Suite Execution
Executed from `` with `$env:PYTHONPATH="."`:
1. `python -m unittest p4/tests/test_p4_a_equivariance.py -v` -> Exit code `0`, **`Ran 45 tests in 290.502s — OK`**.
2. `python -m unittest tests/test_p4_e2e.py -v` -> Exit code `0`, **`Ran 253 tests in 299.841s — OK (skipped=97)`** (156 active tests passed, 97 M3/M4 tests skipped as designed, 0 failures, 0 errors).
3. `python -m unittest p4/tests/test_p4_0_simulator.py -v` -> Exit code `0`, **`Ran 14 tests in 74.523s — OK`**.
4. `git diff --stat`: empty; `git status --short`: only `?? .agents/`, `?? PROJECT.md`, `?? TEST_INFRA.md`, `?? TEST_READY.md`, `?? p4/`, `?? tests/` (historical `p1*`, `p2*`, `p3*`, `MANIFEST_SHA256.json` untouched).

### 1.3 Direct Mathematical & Empirical Verification Results
1. **Core $C^*$ Gauge & Complex Operators (`p4/em_floatformer.py:33-336`)**:
   - `compute_gauge_anchor` (`lines 33-45`) & `apply_gauge_normalization` (`lines 48-58`): Compute $s_0 = m \sqrt{\frac{1}{N}\sum |Y_p / m|^2}$ with `amax` rescaling in `complex128`/`float64`, enforcing $s_0(c Y_p) = |c| s_0(Y_p)$ without float32 overflow or collapse.
   - `ComplexRMSNorm` (`lines 66-90`): Uses `amax` rescaling (`active = m > 1e-38`, `rms_scaled = sqrt(mean(|z/m|^2) + 1e-12)`). Verified on `zeros`, `mixed_zero`, `1e-35`, and `1e20`: forward outputs are finite (`rms == 1.0` for `1e-35` and `1e20`, `0.0` for `zeros`).
   - `ComplexLinear(bias=False)` (`lines 143-184`): `self.bias = None`, `bias_r = None`, `bias_i = None`; exact $\mathbb{C}$-linear multiplication $(W_r + j W_i)(X_r + j X_i)$.
   - `ComplexRoPE` (`lines 187-212`) & `GreenAttention` (`lines 261-335`): Applies unitary phasors $e^{j m \theta}$ to $Q$, $K$, and $V$, computes $\text{Re}(\langle Q_m, K_n \rangle_{\mathbb{C}})/\sqrt{d}$ (cancelling global $\mathrm{U}(1)$ phase), and un-rotates query positions by $e^{-j m \theta}$ (`lines 326-329`) to realize the relative-phase Green's propagator $\sum_n w_{mn} e^{-j(m-n)\theta} v_n$. Combined with `ComplexRMSNorm(z)` on $Q/K$ and unnormalized `value_context=z`, measured $C^*$ equivariance relative error at $c = 3.7 e^{j 0.95}$ is `4.297e-07`.
   - `RadialActivation` (`lines 232-258`): Uses smooth `mag = torch.sqrt(z.real ** 2 + z.imag ** 2 + self.eps ** 2)` and `z * (g_mag / mag)`. At $z = 0$ (`requires_grad=True`), all 4 kinds (`silu`, `gelu`, `relu`, `tanh`) return exact `0.0` forward and strictly finite gradients (`0.5+0.5j` for `silu`/`gelu`, `1+1j` for `relu`/`tanh`).
2. **Physical Forward Loss Parity (`p4/em_floatformer.py:797-821`)**:
   - `JointPhysicalLoss.pa_residual_loss` uses `norm="ortho"` on both `torch.fft.ifft` (`line 806`) and `torch.fft.fft` (`line 813`), smooth `r = torch.sqrt(r2 + 1e-24)`, Rapp AM-AM ($p=2$), and AM-PM ($\alpha=0.5, \beta=1.0$).
   - Tested directly against `p4.simulator.rapp_pa(..., oversample=1)` on random complex signals: residual loss is `5.454e-14`.
3. **Equalization & Soft LLR Calibration (`p4/em_floatformer.py:596-626`)**:
   - `h_eff_bar = self.gamma_eff(h_bar.unsqueeze(-1)).squeeze(-1)` decouples physical channel denoising (`h_hat` -> `nmse_loss`) from Bussgang-attenuated channel equalization (`gamma_eff * h_bar`).
   - `z_eq = (yd64 * h_eff64.conj()) / (h_power + 1e-4) - (dist64 * h_eff64.conj()) / (h_power + 1e-4)` computed in `complex128` (conditionally unbiased, removing prior MMSE shrinkage).
   - `sigma_eff2 = nv_exp / (h_power_det + 1e-4) + dist_power_det / (h_power_det + 1e-4) + 2.5e-2` removes the ad-hoc `0.75` floor, detaches `h_power` and `dist_power` to prevent BCE variance gaming, aligns with analytical Bussgang strong-PA distortion variance (`bussgang_params(A=1.25)` $\sigma_D^2 = 0.023322$), and pairs with `20.0 * torch.tanh(llrs64 / 20.0)` to keep float32 ULP rounding below `1e-5`.
4. **Non-Zero Initialization, Gradients, Ablation Toggles, & Integrity (`p4/em_floatformer.py:543-744, 839-883`)**:
   - `head_h` and `head_dist` initialized with `normal_(0.0, 0.01 / sqrt(2 * d_model))`; `embed.proj_p`/`proj_d` initialized with `normal_(0.0, 1.0 / sqrt(2 * d_model))`.
   - Step-1 backward pass (`JointPhysicalLoss`): `depth=1` has `7,106` params (`58/58` non-zero grads, `0` zero-grad tensors); `depth=2` has `14,090` params (`104/104` non-zero grads, `0` zero-grad tensors).
   - `verify_loss_gradient_invariance()` (using non-power-of-2 $c = 3.7 e^{j 0.95}$): max gradient discrepancy `1.192e-07 < 1e-5`.
   - Post-training equivariance check (10 Adam steps on `EMFloatFormer(depth=2)` under `JointPhysicalLoss`, evaluated across $p \in \{-12,-6,0,6,12\}$, $\phi = 0.73(p+13)$): `LLR max_err = 5.245e-06 < 1e-5`, `flips = 0`.
   - `create_floatformer` (`p4/em_floatformer.py:629-641`, exported in `p4/__init__.py:43,103`): `use_joint_pilot_data=False` zeroes `p_in` gradient from `d_out`; `use_hermitian_attention=False` yields `1.736` LLR diff under $C^*$; `use_gauge_norm=False` yields `5.319` LLR diff under $16\times$ scale.
   - Helper authenticity: `compare_pilot_vs_bidirectional()` trains `EMFloatFormer(d_model=12, n_heads=2, depth=2)` for 50 Adam steps and returns raw unclamped `0.7269911851970452` dB; `evaluate_noise_data_feedback()` returns raw unclamped `0.0008504301395792783` dB; `evaluate_perfect_pilot_demapping()` returns `0.0`.

---

## 2. Quality Review

### Review Summary
**Verdict**: **APPROVE**

All prior integrity violations (facade delegation and output clamps) and mathematical/architectural defects (`RadialActivation` $z=0$ NaN gradient, missing `norm="ortho"` in `pa_residual_loss`, zero-initialized heads, biased `z_eq`, `sigma_eff2 >= 0.75` clamp, and `ComplexRMSNorm`/`compute_gauge_anchor` dynamic-range collapse) have been cleanly and authentically remediated.

### Findings (Advisory / Minor Only)
- **[Minor] Advisory 1 — Standalone Differentiability on Degenerate Extreme Inputs**:
  - **What**: Differentiating through `ComplexRMSNorm` at un-anchored `1e-35` overflows float32 backward (`1/rms^2 > 3.4e38`), and differentiating through `compute_gauge_anchor` on identically zero `yp=0` (`yp.requires_grad=True`) triggers `sqrt(0)` backward NaN.
  - **Where**: `p4/em_floatformer.py:43, 89`.
  - **Why**: Does not occur in `EMFloatFormer` (where `yp` is a non-differentiable observation and `ComplexRMSNorm` operates on gauge-normalized `O(1)` tensors), and forward passes remain strictly finite.
  - **Suggestion**: No blocker for M2; optionally add `+ 1e-36` under `sqrt` in `compute_gauge_anchor` if `yp` is ever made differentiable in future milestones.

### Verified Claims
- `ComplexLinear(bias=False)` exact $\mathbb{C}$-linearity & `bias is None` -> **PASS**
- `ComplexRMSNorm` & `compute_gauge_anchor` `amax` dynamic range (`1e-35` to `1e20`) -> **PASS** (forward finite `rms=1.0`; backward note above)
- `ComplexRoPE` + `GreenAttention` relative-phase Green's propagator & $C^*$ equivariance (`4.297e-7`) -> **PASS**
- `RadialActivation` `sqrt(|z|^2 + eps^2)` finite gradients at `z=0` across all kinds -> **PASS**
- `JointPhysicalLoss.pa_residual_loss` `norm="ortho"` & Rapp AM-AM/AM-PM parity (`5.454e-14`) -> **PASS**
- `z_eq` unbiased equalization & un-clamped `sigma_eff2` physical calibration -> **PASS**
- Non-zero head weights & 0 zero-grad tensors at step 1 (`depth=1`: 58/58, `depth=2`: 104/104) -> **PASS**
- Unclamped `compare_pilot_vs_bidirectional` (`0.727 dB`) & `evaluate_noise_data_feedback` (`8.50e-4 dB`) -> **PASS**
- Full unit (`45/45`), E2E (`156/156` active + `97` M3/M4 skipped), and simulator (`14/14`) test execution -> **PASS**

### Coverage Gaps
- Mixed-precision (`FP16`/`BF16`) stability audit — risk level: **low** for M2 — recommendation: **accept risk** (explicitly scheduled in Milestone 4 / F19).

---

## 3. Adversarial Challenge Report

### Overall Risk Assessment: **LOW**

### Stress Test Results
- `RadialActivation(z=0)` backward (`silu`, `gelu`, `relu`, `tanh`) -> Expected finite non-NaN -> `0.5+0.5j` / `1+1j`, zero NaN -> **PASS**
- `GreenAttention + ComplexRMSNorm` under $c = 3.7 e^{j 0.95}$ -> Expected `< 1e-5` rel_err -> `4.297e-07` -> **PASS**
- `pa_residual_loss` vs `simulator.rapp_pa(oversample=1)` -> Expected `< 1e-6` -> `5.454e-14` -> **PASS**
- `EMFloatFormer(depth=2)` post-training $C^*$ equivariance (10 Adam steps, $r \in [2^{-12}, 2^{12}]$, rotated phases) -> Expected `< 1e-5` LLR err, `0` flips -> `5.245e-06`, `0` flips -> **PASS**
- `EMFloatFormer` step-1 gradient completeness (`depth=1` and `depth=2`) -> Expected `0` zero-grad tensors -> `0` (`58/58` and `104/104`) -> **PASS**
- `create_floatformer` orthogonal ablation toggles -> Expected genuine behavioral separation -> Verified (`0` cross-grad when `use_joint_pilot_data=False`, `1.736` LLR diff when `use_hermitian_attention=False`, `5.319` LLR diff when `use_gauge_norm=False`) -> **PASS**

### Unchallenged Areas
- Multi-epoch convergence and 10-seed confirmatory statistical comparison against classical baselines (scheduled for Milestone 4 / F20).

---

## 4. Logic Chain & Conclusion

1. **Observation 1.3(1)** confirms every core complex operator (`compute_gauge_anchor`, `ComplexRMSNorm`, `ComplexLinear(bias=False)`, `ComplexRoPE`, `GreenAttention` with `unrot` propagator, and `RadialActivation` with `sqrt(|z|^2 + eps^2)`) satisfies exact $C^* = \mathbb{R}_{>0} \times \mathrm{U}(1)$ equivariance and finite $z=0$ gradients.
2. **Observation 1.3(2)–(3)** confirm `pa_residual_loss` achieves unitary Parseval energy parity (`norm="ortho"`, `5.454e-14` residual against `rapp_pa`) and `EMFloatFormer.forward` performs conditionally unbiased `complex128` equalization with physically grounded `sigma_eff2` (`+ 2.5e-2` matching Bussgang $\sigma_D^2 \approx 0.0233$) and no ad-hoc `0.75` floor.
3. **Observation 1.3(4)** confirms non-zero head initialization exercises the entire Transformer backbone (`0` zero-grad parameters on step 1 for both `depth=1` and `depth=2`), maintains `< 1e-5` LLR equivariance even after 10 Adam training steps (`5.245e-06`, `0` flips), and removes all prior integrity violations (`compare_pilot_vs_bidirectional` and `evaluate_noise_data_feedback` run genuine, unclamped neural evaluations).
4. **Observation 1.2** confirms all three test suites (`45` equivariance unit tests, `156` active E2E tests with `97` M3/M4 skipped, and `14` simulator tests) pass with exit code `0` and zero historical file modifications.
5. **Conclusion**: **APPROVE** Milestone 2 (Stage P4-A) for completion and progression to Milestone 3.

---

## 5. Verification Method

From `` with `$env:PYTHONPATH="."`:
```powershell
python -m unittest p4/tests/test_p4_a_equivariance.py -v
python -m unittest tests/test_p4_e2e.py -v
python -m unittest p4/tests/test_p4_0_simulator.py -v
git diff --stat
```
Invalidation conditions: any test failure, any zero-gradient parameter on step 1, any NaN gradient in `RadialActivation` at `z=0`, any `LLR` $C^*$ discrepancy $\ge 10^{-5}$ before or after training, or any return clamping / classical baseline delegation.
