# Milestone 2 Explorer Audit Handoff (`explorer_m2_r1_1`)

**Summary:** On-disk inspection and empirical verification confirm that **all 4 blocking defects** in `challenger_m2_3/handoff.md` (plus the ablation `norm_p2`/`norm_d2` hygiene item) are **already fixed in `p4/em_floatformer.py`** (`p4/__init__.py` exports are consistent, and all 3 existing test suites pass). However, `p4/tests/test_p4_a_equivariance.py` has **not yet added explicit regression tests** for Bugs 1–4, and Milestone 2 sub-items **M2.2** (calibrating the `5.0e-2` floor / `8.0 * tanh` bound at `p4/em_floatformer.py:640-642`) and **M2.3** (controlled comparative benchmarks & `p4/results/` artifacts) remain open.

---

## 1. Observation

### 1.1 On-Disk State of `p4/em_floatformer.py` and `p4/__init__.py`
1. **Bug 1 Fix Present (`p4/em_floatformer.py:575-645`)**:
   - `EMFloatFormer.forward` casts inputs to `complex128`/`float64` (`yp64`, `yd64`, `nv64`), runs `apply_gauge_normalization`, `JointEmbedding`, `BidirectionalBlock` stack, `norm_p`/`norm_d`, `head_h`/`head_dist`, `gamma_eff`, and `calibrated_16qam_llrs` in `complex128`/`float64`, and casts `h_hat`, `dist_est`, `llrs` back to caller dtypes at the boundary.
   - Empirical results (all executed with `torch.set_num_threads(4)` across 10 `(r, phi)` pairs covering `r in [2^-12, 4096.0]`, non-power-of-2 `0.0625`, `0.13`, `3.7`, `1023.5`, pure scale, pure phase):
     - Fresh `seed=11` `EMFloatFormer(depth=2)`: `max_llr_err = 2.623e-06`, `max_h_rel = 1.469e-07`, `flips = 0`.
     - Trained `EMFloatFormer(depth=2)` (50 Adam steps on `JointPhysicalLoss(1.0, 0.25)`, `head_h=0.0542`, `head_dist=0.1195`, `gamma_eff=(0.8891, 0.0511)`):
       - `frame20` (`8x32`, strong PA): `max_llr_err = 1.431e-06`, `max_h_rel = 1.435e-07`, `flips = 0`.
       - `frame30_clean` (`8x32`, clean PA): `max_llr_err = 3.815e-06`, `max_h_rel = 1.377e-07`, `flips = 0`.
       - `frame30_strong_256` (`256x32`, strong PA): `max_llr_err = 3.815e-06`, `max_h_rel = 1.183e-07`, `flips = 0`.
       - Gradient equivariance at `(0.13, -1.7)`, `(3.7, 0.95)`, `(1023.5, 2.5)`: `max_diff = 3.353e-08`.
     - Boosted O(1)-head `EMFloatFormer(depth=2)` (`head_h=1.397`, `head_dist=1.560`, `gamma_eff=0.85+0.15j`) on `frame25` (`1024x32`, strong PA): `max_llr_err = 3.457e-06`, `max_h_rel = 1.170e-07`, `flips = 0`.
2. **Bug 2 Fix Present (`p4/em_floatformer.py:33-58`)**:
   - `compute_gauge_anchor(yp, eps=1e-38)` uses `complex128` `amax` rescaling with `eps=1e-38`.
   - Empirical result on `yp = torch.full((2, 32), 1e-35, dtype=torch.complex64)`:
     - `s0 = [[1.0000000180025095e-35], [1.0000000180025095e-35]]`, `rms_bar = [1.0, 1.0]`, `finite = True`.
3. **Bug 3 Fix Present (`p4/em_floatformer.py:66-93`)**:
   - `ComplexRMSNorm.forward` uses `complex128` `amax` rescaling (`active = m > 1e-38`) and factored `out64 = z_scaled * (gamma64 / rms_safe)`.
   - Empirical result on `1e-35 + 0j`, `1e-35 - 1e-35j`, and `1e20 + 1e20j` (`complex64`, `requires_grad=True`):
     - Forward `out_finite = True`, `rms = [1.0, 1.0]`; backward `z_grad_finite = True`, `gamma_grad_finite = True`.
4. **Bug 4 Fix Present (`p4/em_floatformer.py:239-270`)**:
   - `RadialActivation.forward` executes in `complex128` with `m_safe` rescaling when `m > 1e150`.
   - Empirical result for all 4 kinds (`silu`, `gelu`, `relu`, `tanh`):
     - At `1e20 + 1e20j`: `out_finite = True`, `grad_finite = True` for all 4 kinds.
     - At `z = 0`: `out_finite = True`, `grad_finite = True` (`0.5+0.5j` for `silu`/`gelu`, `1+1j` for `relu`/`tanh`).
5. **Ablation Gradient Usage (`p4/em_floatformer.py:453-464`)**:
   - `BidirectionalBlock` sets `norm_p2`, `cross_d2p`, `norm_d2`, `cross_p2d` to `None` when `use_joint_pilot_data=False`.
   - Empirical check across all 8 `(use_gauge_norm, use_hermitian_attention, use_joint_pilot_data)` configurations in `create_floatformer`: `none = []`, `zero = []`, `nonfinite = []`.
6. **Package Exports (`p4/__init__.py:27-64, 88-123`)**:
   - All 37 Stage P4-A classes/functions are cleanly imported and listed in `__all__`.

### 1.2 On-Disk State of `p4/tests/test_p4_a_equivariance.py` & Downstream M2 Items
- **Missing Regression Tests in `p4/tests/test_p4_a_equivariance.py`**:
  - `test_non_power_of_two_scale_phase_active_backbone` (`lines 405-419`) tests only untrained `EMFloatFormer(depth=2)` at `(r=3.7, phi=0.95)`; no test covers trained/O(1)-head `depth=2` or `seed=11` across `r in [2^-12, 2^12]` + non-power-of-2 scales.
  - No test exercises `compute_gauge_anchor` / `apply_gauge_normalization` on `1e-35` inputs.
  - `test_complex_rms_norm_subnormal_guard` (`lines 83-94`) checks forward only, not backward gradients on `1e-35` (or `1e20`).
  - `test_radial_activation_zero_and_extreme` (`lines 177-187`) checks `1e6` forward for default `silu` only, not `1e20` forward+backward across `silu`, `gelu`, `relu`, `tanh`.
  - `test_create_floatformer_factory` (`lines 295-309`) checks forward shapes/finiteness only, not backward parameter gradient usage across all 8 toggles.
- **M2.2 LLR Floor / Saturation Observation (`p4/em_floatformer.py:640-642`)**:
  - Currently `sigma_eff2 = nv_exp / (h_power_det + 1e-4) + dist_power_det / (h_power_det + 1e-4) + 5.0e-2` and `llrs = (8.0 * torch.tanh(llrs64 / 8.0)).to(out_real_dtype)`.
  - On `SNR=30dB` clean PA (`batch_size=64`), `min(llrs)=-7.999`, `max(llrs)=7.999`, and `24.0%` (`frac(|LLR|>7.9)=0.240`) of bits hit the `8.0` saturation ceiling.
- **M2.3 Comparative Benchmarks (`p4/results/`)**:
  - `p4/results/` currently contains only Stage P4-0 `latency_cpu.csv` and `timing.json`. No M2 comparative benchmark script or results artifact (vs local equivariant conv, `TinyJoint`, `Bussgang-LMMSE`, `DD-PA-Cancel`) exists yet.

### 1.3 Exact Current Test Suite Counts
| Suite | Command | Total | Passed | Skipped | Failed / Errors | Exit Code | Duration |
|---|---|---|---|---|---|---|---|
| Stage P4-0 Unit | `python -m unittest p4.tests.test_p4_0_simulator -v` | 14 | 14 | 0 | 0 | 0 | 3.734s |
| Stage P4-A Unit | `python -m unittest p4.tests.test_p4_a_equivariance -v` | 45 | 45 | 0 | 0 | 0 | 18.107s |
| Master E2E (Tiers 1–4) | `python -m unittest tests.test_p4_e2e -v` | 253 | 156 | 97 (M3/M4) | 0 | 0 | 28.374s |

---

## 2. Logic Chain

1. **`p4/em_floatformer.py` already received the Iteration 3 hardening edits** before `worker_m2_3` paused: executing the gauge-normalized `EMFloatFormer.forward` in `complex128`/`float64` eliminates the `~5e-7` `complex64` rounding drift that `calibrated_16qam_llrs` previously amplified to `8.011e-05`, bringing all tested fresh (`2.623e-06`), trained (`<= 3.815e-06`), and O(1)-head (`3.457e-06`) `depth=2` models well below `< 1e-5` with `0` bit flips.
2. **Numerical singularity hardening is complete in `p4/em_floatformer.py`**: `eps=1e-38` in `compute_gauge_anchor` preserves unit RMS at `1e-35`; factored `amax`-rescaled `ComplexRMSNorm` avoids `1/rms^2` overflow in `float32` backward; and `complex128` + `m > 1e150` `amax` rescaling in `RadialActivation` eliminates `1e20` overflow across all 4 activations while keeping `z=0` gradients finite.
3. **Why M2 is not yet ready for Gate sign-off**:
   - `p4/tests/test_p4_a_equivariance.py` still reflects pre-Iteration-3 coverage and does not lock down Bugs 1–4 with automated regression tests.

---

## 3. Caveats

- All checks were executed on CPU with `torch.set_num_threads(4)`.
- When running `tests.test_p4_e2e`, Tier 1 F7 writes `p4/results/latency_cpu.csv` and `p4/results/timing.json` in-place (the files retain Stage P4-0 benchmark schemas).
- `p4/tests/test_gauge_stress_adversarial.py` exists on disk (314 lines, targeting untrained `EMFloatFormer(d_model=64, depth=2)`); it was inspected by `read_file` but not separately executed since it is outside the 3 requested suites and tests only untrained models.

---

## 4. Recommended Strategy (Conclusion)

**Do NOT re-implement the core fixes in `p4/em_floatformer.py`** — the Iteration 3 `complex128`/`float64` backbone and `amax`-rescaled normalization/activation fixes are already in place and verified. Proceed directly with the remaining M2.1–M2.4 items while strictly avoiding all `DEAD_ENDS.md` anti-patterns:

1. **Close M2.1 Regression Tests (`p4/tests/test_p4_a_equivariance.py`)**:
   - Add tests for: (a) trained `EMFloatFormer(depth=2)` and fresh `seed=11` `depth=2` C* equivariance (`max_llr_err < 1e-5`, `flips == 0`, `max_h_rel < 1e-5`) across `r in [2^-12, 2^12]` and non-power-of-2 scales; (b) `compute_gauge_anchor` / `apply_gauge_normalization` on `1e-35` inputs (`rms_bar ≈ 1.0`); (c) `ComplexRMSNorm` backward finite gradients at `1e-35` and `1e20`; (d) `RadialActivation` (`silu`, `gelu`, `relu`, `tanh`) forward & backward finiteness at `1e20` and `z=0`; (e) all 8 `create_floatformer` configurations having zero `None` or `0.0` gradients in backward.
   - **Avoid `DEAD_ENDS.md`**: Keep `complex128`/`float64` internal execution, `amax` rescaling, `eps=1e-38`, factored `ComplexRMSNorm`, non-zero head init, and `norm="ortho"` in `pa_residual_loss`.
2. **Execute M2.2 LLR Calibration Audit (`p4/em_floatformer.py:640-642`)**:
   - Benchmark LLR dynamic range, ECE/BCE calibration, and C* equivariance when relaxing `sigma_eff2` floor (e.g. from `5.0e-2` toward `1e-2`–`2.5e-2` or SNR-adaptive floor) and `tanh` bound (e.g. `20.0 * torch.tanh(llrs64 / 20.0)`), ensuring `max_llr_err < 1e-5` and `flips == 0` remain satisfied under `complex128`/`float64`.
   - **Avoid `DEAD_ENDS.md`**: Do not re-introduce an artificial `sigma_eff2 >= 0.75` floor or biased `z_eq`.
3. **Execute M2.3 Controlled Comparative Benchmarks (`p4/results/`)**:
   - Implement genuine, trained comparative evaluation of `EMFloatFormer` vs parameter-matched local equivariant convolution, `TinyJoint`, `Bussgang-LMMSE`, and `DD-PA-Cancel` (no clamping or delegating to classical baselines per `DEAD_ENDS.md`), saving metrics under `p4/results/` without overwriting `latency_cpu.csv` / `timing.json` schemas.
4. **Re-run M2.4 Gate Suites**:
   - Verify `p4.tests.test_p4_0_simulator`, `p4.tests.test_p4_a_equivariance`, and `tests.test_p4_e2e` all pass with exit code 0, and historical P1–P3 / `MANIFEST_SHA256.json` remain untouched.

---

## 5. Verification Method

From ``:
1. Re-verify Bugs 1–4 & ablation gradients:
   - Run the `python -c "..."` checks with `torch.set_num_threads(4)` testing trained `EMFloatFormer(depth=2)`, `apply_gauge_normalization` at `1e-35`, `ComplexRMSNorm` backward at `1e-35`/`1e20`, `RadialActivation` (`silu`/`gelu`/`relu`/`tanh`) forward+backward at `1e20`/`0`, and all 8 `create_floatformer` configurations.
2. Run the 3 regression/E2E suites:
```powershell
python -m unittest p4.tests.test_p4_0_simulator -v
python -m unittest p4.tests.test_p4_a_equivariance -v
python -m unittest tests.test_p4_e2e -v
```
3. **Invalidation conditions**: M2.1 is invalidated if any regression test fails, any trained/fresh `depth=2` model shows `max_llr_err >= 1e-5` or `flips > 0`, any `1e-35`/`1e20`/`0` forward or backward produces `NaN`/`Inf` or anchor collapse, or any `create_floatformer` parameter has `grad is None`.
