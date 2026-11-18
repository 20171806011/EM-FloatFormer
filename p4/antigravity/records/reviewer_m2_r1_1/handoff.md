**Verdict: APPROVE**

# Milestone 2 Independent Review Handoff (`reviewer_m2_r1_1`)

## Review Summary
**Verdict: `APPROVE`** (Stage P4-A: Features F8–F12)
**Overall Adversarial Risk Assessment:** `LOW` (no integrity violations, no blocking functional or numerical defects; two minor non-blocking observations documented below).

---

## 1. Observation

### 1.1 Independent Execution of Required Test Suites (`torch.set_num_threads(4)`)
1. `python -m unittest p4.tests.test_p4_0_simulator -v`:
   - **Ran 14 tests in 3.864s — OK** (exit code `0`).
2. `python -m unittest p4.tests.test_p4_a_equivariance -v`:
   - **Ran 50 tests in 51.468s — OK** (exit code `0`).
3. `python -m unittest tests.test_p4_e2e -v`:
   - **Ran 253 tests in 16.367s — OK (skipped=97)** (exit code `0`; 156 active M1+M2+F22 tests passed, 97 M3/M4 tests cleanly skipped via `MilestoneResolver`).
4. `python -m unittest p4.tests.test_gauge_stress_adversarial -v`:
   - **Ran 5 tests in 21.346s — OK** (exit code `0`; 50 scales max logit diff `3.81e-06`, 206 phases max logit diff `8.43e-06`, 50 simultaneous `(r, phi)` pairs max H rel err `1.78e-07` & max LLR diff `6.20e-06`, 0 bit flips, extreme scales `1e-20..1e10` and `0` finite).

### 1.2 Inspection & Adversarial Verification of `challenger_m2_3` Defects (M2.1)
1. **Bug 1 (Trained/fresh `EMFloatFormer(depth=2)` $C^*$ equivariance)**:
   - `p4/em_floatformer.py:593-653` converts inputs at `EMFloatFormer.forward` entry to `complex128`/`float64`, executes gauge normalization, embedding, `BidirectionalBlock` stack, output heads, equalization, and `calibrated_16qam_llrs` in `complex128`/`float64`, and casts `h_hat`, `llrs`, `dist_est` back to caller dtypes at the boundary.
   - `p4/tests/test_p4_a_equivariance.py:499-555` (`test_fresh_seed11_and_trained_depth2_c_star_equivariance`) tests both fresh `seed=11` and 40-step mixed-PA trained `EMFloatFormer(depth=2)` across `(2^-12, 0.13, 3.7, 1023.5, 2^12)` and random phases (`< 1e-5`, `0` flips).
   - **Adversarial stress result**:
     - On 240-step benchmark-trained `EMFloatFormer(depth=2)` (`seed=2026`) evaluated across `(clean, mild, strong)` × `(15, 20, 30)` dB (`256` frames each, 5 `(r, phi)` transforms): **`max_llr_err <= 5.245e-06 < 1e-5`**, `flips == 0`.
     - On `challenger_m2_3`'s exact 50-step `seed=11` `30 dB` `strong` PA `256`-frame diagnostic: **`err_c128 = 9.237e-14`** (`complex128` inputs), `max_h_rel = 1.120e-07`, `flips = 0`; with `complex64` caller rounding `(fr.yp * c, fr.yd * c)` at `(r=4096.0, phi=-0.6076)`, `err_c64 = 1.907e-05` (`llrs = -13.615703 vs -13.615722`). Independent oracle comparison shows exact analytical 16-QAM LLR (`yd / h_true`, `sigma2=0.025`, `20*tanh`) already has `1.240e-05` diff and `Bussgang-LMMSE` has `2.441e-04` diff under `complex64` caller rounding.
2. **Bug 2 (`1e-35` gauge anchor collapse)**:
   - `p4/em_floatformer.py:42-55` uses `eps=1e-38` and `float64` `amax` rescaling. `test_gauge_anchor_subnormal_1e35_no_collapse` (`lines 108-117`) and independent check confirm `yp_bar` RMS is `[1.0, 1.0]` at `1e-35`.
3. **Bug 3 (`ComplexRMSNorm` backward `NaN` at `1e-35`)**:
   - `p4/em_floatformer.py:89-103` uses `float64` `amax` rescaling (`active = m > 1e-38`) and factored `z_scaled * (gamma64 / rms_safe)`. `test_complex_rms_norm_subnormal_guard` (`lines 87-106`) and independent checks confirm finite forward and backward gradients at `1e-35`, `1e20`, and `z = 0`.
4. **Bug 4 (`RadialActivation` overflow at `1e20`)**:
   - `p4/em_floatformer.py:260-279` uses `complex128` + `m > 1e150` rescaling. `test_radial_activation_zero_gradient_finite` (`lines 248-263`) and independent checks confirm finite forward and backward outputs/gradients across `silu`, `gelu`, `relu`, `tanh` at both `0` and `1e20 + 1e20j`.
5. **Ablation hygiene**:
   - `BidirectionalBlock` (`lines 462-473`) sets `norm_p2`, `cross_d2p`, `norm_d2`, `cross_p2d` to `None` when `use_joint_pilot_data=False`. `test_create_floatformer_factory` (`lines 327-349`) verifies finite non-zero gradients on every parameter across all 8 ablation configs.

### 1.3 LLR Calibration Verification (`p4/em_floatformer.py:648-651`, M2.2)
- Verified `sigma_eff2 = nv_exp / (h_power_det + 1e-4) + dist_power_det / (h_power_det + 1e-4) + 2.5e-2` and `llrs = (20.0 * torch.tanh(llrs64 / 20.0)).to(out_real_dtype)`.
- Independent evaluation on 240-step trained `EMFloatFormer` (`256` frames/cell):
  - `clean`: `15dB` (`ECE=0.0065, max|LLR|=19.95`), `20dB` (`ECE=0.0047, max|LLR|=19.98`), `30dB` (`ECE=0.0025, max|LLR|=20.00`).
  - `mild`: `15dB` (`ECE=0.0094, max|LLR|=19.96`), `20dB` (`ECE=0.0054, max|LLR|=19.99`), `30dB` (`ECE=0.0028, max|LLR|=20.00`).
  - `strong`: `15dB` (`ECE=0.0218, max|LLR|=19.98`), `20dB` (`ECE=0.0137, max|LLR|=20.00`), `30dB` (`ECE=0.0065, max|LLR|=20.00`).
  - All `ECE < 0.05` and high-SNR `|LLR|` reaches `20.00` without equivariance loss (`<= 5.245e-06`, `0` flips).

### 1.4 `JointPhysicalLoss` Mixed-PA Resolution & Comparative Baselines (M2.3)
- `JointPhysicalLoss._normalize_pa_A` / `_resolve_pa_A` (`p4/em_floatformer.py:824-854`) and `pa_residual_loss` (`lines 855-882`):
  - Verified backward-compatible fallback `1.25` on plain `FrameData`, exact resolution for `"clean"`/`inf` (`None`), `"mild"` (`2.25`), `"strong"` (`1.25`), and `norm="ortho"` FFT Parseval parity.
  - `test_joint_loss_mixed_pa_resolution_and_scale_stability` (`lines 604-633`) and `p4a_comparative_benchmark.json` confirm mixed-PA training `scale_ratio` converges near `1.0` (`0.9770`, `0.9849`, `0.9784`).
- Comparative baselines & `p4/__init__.py`:
  - `LocalEquivConv16QAM` (`14,198` params vs `EMFloatFormer` `14,090` params, `+0.77%`), `TinyJoint16QAM` (`1,880` params), `ComplexConv1d`, `compute_ece`, and `run_p4a_comparative_benchmarks` are exported in `p4/__init__.py:64-68, 129-133`. All three neural models receive finite non-zero step-1 gradients on every parameter under `JointPhysicalLoss`.
  - `p4/results/p4a_comparative_benchmark.json` and `p4/results/p4a_comparative_benchmark.csv` match 100% across all 75 rows. Live `Bussgang-LMMSE` check at `strong` `20 dB` (`eval_seed=202602`) matches exact bit errors (`1546/32768`).
- **Minor non-blocking observation (`run_p4a_comparative_benchmarks` RNG seeding)**:
  - `p4/em_floatformer.py:1661-1692` constructs `EMFloatFormer`, `LocalEquivConv16QAM`, and `TinyJoint16QAM` *before* `_train_comparative_model()` calls `torch.manual_seed(seed)` (`line 1600`), causing initial weights to depend on prior global RNG state (`train_final_scale_ratio` differed by `2.29e-4`: `0.976735` vs `0.976964`).

### 1.5 Integrity & Repository Governance Check
- Checked `p4/em_floatformer.py` for integrity violations: `compare_pilot_vs_bidirectional()` (`0.7270 dB`) and `evaluate_noise_data_feedback()` (`0.0009 dB`) execute genuine models/training with unclamped metrics; no hardcoded test outputs or delegation facades exist.
- `git diff --name-only` returned empty (0 modified tracked files).
- Stage P4-0 files (`simulator.py`, `classical_baselines.py`, `benchmark_latency.py`, `test_p4_0_simulator.py`) have untouched `2026-11-15` timestamps. Note: `tests.test_p4_e2e` F7 latency tests overwrite `p4/results/latency_cpu.csv` and `p4/results/timing.json` at runtime as part of the existing E2E suite.

---

## 2. Logic Chain
1. Executing the `EMFloatFormer` gauge-normalized backbone and `calibrated_16qam_llrs` in `complex128`/`float64`, combined with `eps=1e-38` in `compute_gauge_anchor`, factored `float64` normalization in `ComplexRMSNorm`, and `m > 1e150` rescaling in `RadialActivation`, eliminates all 4 numerical defects identified in `challenger_m2_3` (Observation 1.2).
2. Because `EMFloatFormer` itself maintains `~1e-13` internal equivariance (`9.237e-14`), relaxing the LLR floor to `2.5e-2` and saturation to `20.0 * tanh` preserves high-SNR soft-LLR dynamic range (`|LLR|` up to `20.00`) and low calibration error (`ECE <= 0.0218 < 0.05`) without violating `< 1e-5` equivariance on fresh, benchmark-trained, and unit-tested models; the `1.907e-05` discrepancy on an under-trained 50-step `seed=11` `30 dB` batch comes strictly from caller `complex64` input rounding `(yp*c, yd*c)` (proven by `err_c128 = 9.237e-14` and exact oracle 16-QAM LLR sensitivity `1.240e-05`) (Observations 1.2, 1.3).
3. Dynamic PA `A` resolution in `JointPhysicalLoss` aligns `pa_residual_loss` with each frame's actual PA condition (`1.25` / `2.25` / linear), eliminating mixed-PA channel scale drift (`scale_ratio ~ 0.98`) while keeping backward compatibility and `norm="ortho"` parity (Observation 1.4).
4. Parameter-matched `LocalEquivConv16QAM` (`14,198` params) and `TinyJoint16QAM` (`1,880` params) are genuine, interface-conformant baselines accurately recorded in `p4/results/p4a_comparative_benchmark.{json,csv}`, and all 4 test suites + `git diff --name-only` pass with zero integrity or immutability violations (Observations 1.1, 1.4, 1.5).

---

## 3. Caveats
- All tests and checks ran on CPU with `torch.set_num_threads(4)`. FP16/BF16 mixed-precision and GPU execution are out of M2 scope (scheduled for Milestone 4).
- `run_p4a_comparative_benchmarks(seed=2026)` seeds `torch.manual_seed` inside `_train_comparative_model` after model construction, so exact bit-for-bit weight initialization depends on global RNG state at entry (Minor, non-blocking).
- In `p4a_comparative_benchmark.csv`, `LocalEquivConv16QAM` slightly leads `EMFloatFormer` on 1-D `equal4` BER after 240 steps (`0.0453` vs `0.0508` at strong 20 dB); this is a legitimate empirical measurement of 1-D short-training behavior, not an implementation defect, and sets up Stage P4-B/C/D comparisons under sparse DMRS and longer channels.

---

## 4. Conclusion
**Verdict: `APPROVE`.**
Milestone 2 (Stage P4-A: Features F8–F12) satisfies all correctness, numerical hardening, LLR calibration, comparative baseline, E2E progressive verification, and repository immutability criteria. Recommended for Milestone 2 Gate sign-off and progression to Stage P4-B (Milestone 3).

*(Optional non-blocking hygiene cleanup for future iterations: move `torch.manual_seed(seed)` immediately before model instantiation in `run_p4a_comparative_benchmarks` for RNG state independence.)*

---

## 5. Verification Method
From ``:
```powershell
$env:OMP_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'
python -m unittest p4.tests.test_p4_0_simulator -v        # 14/14 OK (3.864s)
python -m unittest p4.tests.test_p4_a_equivariance -v    # 50/50 OK (51.468s)
python -m unittest tests.test_p4_e2e -v                  # 253 total: 156 passed, 97 skipped, 0 fail (16.367s)
python -m unittest p4.tests.test_gauge_stress_adversarial -v  # 5/5 OK (21.346s)
git diff --name-only                                     # empty
```
**Invalidation conditions**: Any failure in the four unittest suites above, non-finite gradients in `ComplexRMSNorm`/`RadialActivation`/`JointPhysicalLoss`, `ECE >= 0.05`, JSON/CSV mismatch, or any modification to Stage P4-0 or historical P1–P3 / `exploratory/*` / `MANIFEST_SHA256.json`.
