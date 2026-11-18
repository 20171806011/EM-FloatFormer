Verdict: APPROVE

# Milestone 2 Iteration 2 Challenger Handoff (`challenger_m2_r2_1`)

**Overall risk assessment**: LOW (All `challenger_m2_3` blocking defects are empirically resolved; all singularities, ablations, benchmark reproducibility checks, and 4 test suites pass).

---

## 1. Observation

All empirical attacks and verification suites were executed on CPU with `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `torch.set_num_threads(4)`.

### 1.1 `challenger_m2_3` Regression Attack & Equivariance Stress Results

1. **Exact `challenger_m2_3` 50-step `seed=11` `256`-frame `30 dB` `strong` PA diagnostic + Boosted / Deeper / Conv Baselines** across `r in [2^-12, 2^12]` and non-power-of-2 scales `(0.13, 3.7, 1023.5)` with random/non-zero phases:
   - `trained_d2_seed11_30db_256`: `max_llr_err = 7.629e-06`, `flips = 0`, `max_h_rel = 1.223e-07` (worst at `r=2^-10, phi=-0.6076`). *(Previously `8.011e-05` in `challenger_m2_3`.)*
   - `fresh_d2_seed11_30db_256`: `max_llr_err = 5.722e-06`, `flips = 0`, `max_h_rel = 1.212e-07`. *(Previously `1.049e-05`.)*
   - `boosted_d2_30db_256` (`50x` boosted `head_h`/`head_dist` norms `0.3846` / `0.3688`, `gamma_eff = 0.85+0.15j`): `max_llr_err = 5.245e-06`, `flips = 0`, `max_h_rel = 1.649e-07`. *(Previously `6.771e-05`.)*
   - `boosted_d4_30db_256` (`depth=4`, `50x` boosted heads): `max_llr_err = 6.676e-06`, `flips = 0`, `max_h_rel = 1.472e-07`.
   - `trained_local_conv_30db_256` (`LocalEquivConv16QAM`, 50 Adam steps): `max_llr_err = 3.815e-06`, `flips = 0`, `max_h_rel = 1.035e-07`.
   - **Benchmark-trained `EMFloatFormer` (`240` steps, `seed=2026`, `_train_comparative_model`) on `256` frames at `30 dB`**:
     - `strong`: `c64_max = 4.292e-06`, `flips = 0` | `c128_max = 3.553e-14`, `flips = 0`.
     - `mild`: `c64_max = 4.053e-06`, `flips = 0` | `c128_max = 4.263e-14`, `flips = 0`.
     - `clean`: `c64_max = 3.815e-06`, `flips = 0` | `c128_max = 3.553e-14`, `flips = 0`.
   - **Gradient equivariance** (`20 dB`, `strong` PA, `8` frames, across `(2^-12, 2.7183)`, `(0.13, -1.7)`, `(3.7, 0.95)`, `(1023.5, 2.5356)`, `(4096.0, -0.6076)`):
     - `trained_d2`: `grad_equiv_max_diff = 5.960e-08`
     - `trained_d4`: `grad_equiv_max_diff = 1.192e-07`
     - `trained_local_conv`: `grad_equiv_max_diff = 4.470e-08`
   - **Ablation `(use_gauge_norm=True, use_hermitian_attention=True, use_joint_pilot_data=False)`** (`c = 1023.5 * exp(j*2.5356)`): `max_llr_err = 2.146e-06`, `flips = 0`.

2. **Additional Multi-Seed / `depth=4` `complex64` vs `complex128` Boundary Probe** (50-step `lr=1e-2` unconverged models on `30 dB` `256`-frame `strong` PA):
   - `trained_d4` (`seed=11`): `c64_max = 1.049042e-05` (`-9.654384` vs `-9.654394` at `r=2^-10` and `r=4096, phi=-0.6076`), `flips = 0` | `c128_max = 1.990e-13`, `flips = 0`.
   - `trained_d2` (`d=12, depth=2`):
     - `seed=7`: `c64_max = 1.049e-05`, `flips = 0` | `c128_max = 1.599e-13`, `flips = 0`.
     - `seed=42`: `c64_max = 1.097e-05`, `flips = 0` | `c128_max = 1.172e-13`, `flips = 0`.
     - `seed=123`: `c64_max = 1.287e-05`, `flips = 0` | `c128_max = 1.563e-13`, `flips = 0`.
     - `seed=2026`: `c64_max = 6.676e-06`, `flips = 0`.
   - `trained_d2` (`d=64, depth=2`):
     - `seed=7`: `c64_max = 3.912e-07`, `flips = 0`.
     - `seed=11`: `c64_max = 2.187e-06`, `flips = 0`.
     - `seed=42`: `c64_max = 9.537e-07`, `flips = 0`.
     - `seed=123`: `c64_max = 4.247e-07`, `flips = 1` (single near-zero bit at `r=0.13, phi=-1.7`: `-3.75620479e-10` vs `+3.84055565e-11`) | `c128_max = 5.551e-15`, `flips = 0`.
     - `seed=2026`: `c64_max = 4.403e-06`, `flips = 0`.

### 1.2 Singularity, Boundary & Ablation Attack Results

1. **`compute_gauge_anchor` / `apply_gauge_normalization`**:
   - `1e-35` (real): `s0 = [1.000e-35, 1.000e-35]`, `rms = [1.0, 1.0]`, `finite = True`.
   - `1e-35 - 1e-35j`: `s0 = [1.414e-35, 1.414e-35]`, `rms = [1.0, 1.0]`, `finite = True`.
   - `1e-38` (real): `s0 = [1.000e-38, 1.000e-38]`, `rms = [1.0, 1.0]`, `finite = True`.
   - `0.0`: `s0 = [1.000e-38, 1.000e-38]`, `rms = [0.0, 0.0]`, `finite = True`.
   - `1e20` (real): `s0 = [1.000e+20, 1.000e+20]`, `rms = [1.0, 1.0]`, `finite = True`.
   - `1e20 + 1e20j`: `s0 = [1.414e+20, 1.414e+20]`, `rms = [1.0, 1.0]`, `finite = True`.
2. **`ComplexRMSNorm` forward + backward**:
   - `1e-35`, `1e-35 - 1e-35j`, `1.2e-38`, `1e20`, `1e20 + 1e20j`, `1e38`: `rms_out = [1.0, 1.0]`, `out_fin = True`, `z_grad_fin = True`, `gamma_grad_fin = True`.
   - `1e-38`, `0.0`: `rms_out = [0.0, 0.0]` (below `1e-38` active threshold), `out_fin = True`, `z_grad_fin = True`, `gamma_grad_fin = True`.
3. **`RadialActivation` (`silu`, `gelu`, `relu`, `tanh`) forward + backward**:
   - At `0.0 + 0.0j`, `1e-35 + 1e-35j`, `1e20 + 1e20j`, and mixed `[1e38+0j, -1e38+0j, 0+1e38j, 1e-38-1e-38j]`: **ALL** `out_fin = True`, `grad_fin = True` (`silu`/`gelu` at `0.0` yield `out=0j, grad=0.5+0.5j`; `relu`/`tanh` yield `out=0j, grad=1+1j`).
4. **Full `EMFloatFormer(depth=2)` forward + backward at extremes**:
   - `all_zero` (`yp=0, yd=0`), `r=1e-18`, `r=1e18`: **ALL** `out_fin = True`, `loss_fin = True`, `grad_fin = True`.
5. **All 8 `create_floatformer` ablation configs (`(g, h, j) in {False, True}^3`)** with `JointPhysicalLoss`:
   - `(False, False, False)`: 58 params, `none=0`, `zero=0`, `nonfin=0`
   - `(False, False, True)`: 90 params, `none=0`, `zero=0`, `nonfin=0`
   - `(False, True, False)`: 58 params, `none=0`, `zero=0`, `nonfin=0`
   - `(False, True, True)`: 90 params, `none=0`, `zero=0`, `nonfin=0`
   - `(True, False, False)`: 68 params, `none=0`, `zero=0`, `nonfin=0`
   - `(True, False, True)`: 104 params, `none=0`, `zero=0`, `nonfin=0`
   - `(True, True, False)`: 68 params, `none=0`, `zero=0`, `nonfin=0`
   - `(True, True, True)`: 104 params, `none=0`, `zero=0`, `nonfin=0`

### 1.3 Test Suites, Benchmark Replay & Repository Governance

1. `python -m unittest p4.tests.test_p4_0_simulator -v`: **Ran 14 tests in 3.203s — OK** (exit code 0).
2. `python -m unittest p4.tests.test_p4_a_equivariance -v`: **Ran 53 tests in 110.742s — OK** (exit code 0).
3. `python -m unittest tests.test_p4_e2e -v`: **Ran 253 tests in 40.732s — OK (skipped=97)** (156 active M1/M2/F22 passed, 97 M3/M4 cleanly skipped, exit code 0).
4. `python -m unittest p4.tests.test_gauge_stress_adversarial -v`: **Ran 5 tests in 38.391s — OK** (exit code 0).
5. **RNG-perturbed benchmark replay** (`torch.randn(19)` prior to `run_p4a_comparative_benchmarks(output_dir=None)`): **`EXACT_REPRO_MATCH_100PCT`** across all `75` rows (`param_counts = {'EMFloatFormer': 14090, 'LocalEquivConv16QAM': 14198, 'TinyJoint16QAM': 1880, 'Bussgang-LMMSE': 0, 'DD-PA-Cancel': 0}`).
6. `git diff --name-only`: empty (exit code 0; 0 tracked historical P1–P3 / `exploratory/*` / `MANIFEST_SHA256.json` files modified).

---

## 2. Logic Chain

1. Executing the `EMFloatFormer` and `LocalEquivConv16QAM` internal backbones, `apply_gauge_normalization`, `ComplexRMSNorm`, `RadialActivation`, and `calibrated_16qam_llrs` in `complex128`/`float64` (while preserving caller dtypes at the external boundary) completely eliminated the `~5e-7` stacked `complex64` backbone drift identified in `challenger_m2_3`. As observed in Section 1.1, `challenger_m2_3`'s exact 50-step `seed=11` diagnostic dropped from `8.011e-05` to `7.629e-06 < 1e-5` (`0` flips), fresh `depth=2` dropped from `1.049e-05` to `5.722e-06`, boosted O(1)-head `depth=2` dropped from `6.771e-05` to `5.245e-06`, and the 240-step benchmark `EMFloatFormer` achieves `<= 4.292e-06` (`complex64`) and `<= 4.263e-14` (`complex128`).
2. `compute_gauge_anchor` (`eps=1e-38` with `amax` rescaling in `complex128`/`float64`) preserves unit-RMS normalization down to `1e-35` and `1e-38` and up to `1e20` without anchor collapse.
3. `ComplexRMSNorm` (factored `z_scaled * (gamma64 / rms_safe)` in `complex128`/`float64`) eliminates the `1/rms^2` autograd overflow at `1e-35` and `1e-38`, producing finite outputs and finite gradients across `[0.0, 1e38]`.
4. `RadialActivation` (`m_safe` rescaling in `complex128` for `m > 1e150`) eliminates `z.real**2 + z.imag**2` overflow across all 4 activation kinds (`silu`, `gelu`, `relu`, `tanh`) up to `1e38` in both forward and backward passes.
5. `BidirectionalBlock` gating `norm_p2`/`norm_d2` on `use_joint_pilot_data` ensures zero `None`, zero `0.0`, and zero non-finite gradients across all 8 `create_floatformer` ablation configurations.

---

## 3. Caveats

- **IEEE-754 `complex64` caller-side input quantization floor (non-blocking)**: When a caller computes `yp * c64` in `complex64` (24-bit mantissa, `eps_f32 ≈ 1.192e-7`), the input tensor itself carries `~1e-7` relative rounding error before entering `EMFloatFormer`. At `30 dB` SNR (`sigma_eff2 ≈ 0.036`) with `20.0 * tanh(llrs64 / 20.0)`, `calibrated_16qam_llrs` has an analytical Lipschitz gain of `O(100)` with respect to `z_eq`, so some arbitrary 50-step `lr=1e-2` unconverged seeds (`d=12, depth=2` seeds `7, 42, 123`; `d=12, depth=4` seed `11`) observe `1.049e-05` to `1.287e-05` in `complex64`, and `d=64, depth=2` seed `123` had `1` sign flip on an exact-zero-boundary LLR (`-3.756e-10` vs `+3.841e-11`) despite `c64_max = 4.247e-07`. Passing `complex128` inputs (`yp128 * c128`) proves the model's exact mathematical equivariance (`<= 1.990e-13`, `0` flips across every tested seed/depth/width), and the converged 240-step comparative `EMFloatFormer` stays `<= 4.292e-06 < 1e-5` with `0` flips even in `complex64`.
- **Execution scope**: All tests ran on CPU (`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `torch.set_num_threads(4)`). Mixed-precision (`FP16`/`BF16`) and 2D/FEC (`slot_2d`) are scoped to Milestones M3–M4.
- Note that running `tests.test_p4_e2e` executes `test_tier2_boundary_corner.py`, which regenerates `p4/results/latency_cpu.csv` and `p4/results/timing.json` at runtime (these are untracked Stage P4-0 outputs, not modified by `challenger_m2_r2_1`).

---

## 4. Conclusion

**Verdict: `APPROVE`.**

`p4/em_floatformer.py` passes all empirical adversarial challenges for Milestone 2 (Stage P4-A: F8–F12) Iteration 2:
- All 4 blocking bugs from `challenger_m2_3` (`depth=2` `seed=11` `30 dB` `strong` PA, `1e-35` anchor collapse, `1e-35` `ComplexRMSNorm` backward `NaN`, and `1e20` `RadialActivation` overflow) are verified fixed.
- All 8 `create_floatformer` ablation configs have clean gradients (`0` `None`, `0` zero, `0` non-finite).
- Comparative benchmarks in `p4/results/p4a_comparative_benchmark.{json,csv}` are 100% bit-exact reproducible under global RNG perturbation.
- All 4 test suites (`test_p4_0_simulator`, `test_p4_a_equivariance`, `test_p4_e2e`, `test_gauge_stress_adversarial`) pass with exit code 0 and `git diff --name-only` is clean.

---

## 5. Verification Method

From ``:
```powershell
$env:OMP_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'
python -m unittest p4.tests.test_p4_0_simulator -v
python -m unittest p4.tests.test_p4_a_equivariance -v
python -m unittest tests.test_p4_e2e -v
python -m unittest p4.tests.test_gauge_stress_adversarial -v
git diff --name-only
```
Invalidation conditions: any test failure/error in the 4 suites, any non-zero bit flip or `>= 1e-5` error on `test_challenger_m2_3_50step_seed11_30db_strong_256frames` or benchmark-trained `EMFloatFormer`, any `NaN`/`Inf` in `1e-35`/`1e20` forward/backward checks, or any modification to tracked historical files.
