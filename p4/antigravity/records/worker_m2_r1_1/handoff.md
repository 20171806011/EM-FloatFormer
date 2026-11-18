# Milestone 2 Implementation Handoff (`worker_m2_r1_1`)

**Status:** COMPLETE — All M2.1, M2.2, M2.3, and M2.4 tasks implemented, benchmarked, and verified with zero test failures and zero modifications to historical P1–P3, `exploratory/*`, or Stage P4-0 files.

---

## 1. Observation

### 1.1 Code Changes Applied
1. **`p4/em_floatformer.py`** (preserved all Iteration 3 `complex128`/`float64` hardening, `eps=1e-38`, factored `ComplexRMSNorm`, `m > 1e150` `RadialActivation`, `norm_p2`/`norm_d2 = None` ablation guard, non-zero head initialization, and `norm="ortho"` FFT parity):
   - **M2.2 LLR Calibration (`p4/em_floatformer.py:648-650`)**: Updated from `+ 5.0e-2` and `8.0 * torch.tanh(llrs64 / 8.0)` to:
     ```python
     sigma_eff2 = nv_exp / (h_power_det + 1e-4) + dist_power_det / (h_power_det + 1e-4) + 2.5e-2
     llrs64 = calibrated_16qam_llrs(z_eq, sigma_eff2, method="exact")
     llrs = (20.0 * torch.tanh(llrs64 / 20.0)).to(out_real_dtype)
     ```
   - **M2.3 `JointPhysicalLoss` Mixed-PA Resolution (`p4/em_floatformer.py:824-902`)**:
     - Added `_normalize_pa_A` and `_resolve_pa_A(self, frame, A=None)` supporting explicit `A` (float, `inf`, or `"clean"`/`"mild"`/`"strong"`), `frame.A`, or `frame.pa_condition` (`"strong"` -> `1.25`, `"mild"` -> `2.25`, `"clean"` -> `None`), with backward-compatible fallback `1.25` when no PA metadata is provided.
     - Updated `pa_residual_loss(self, yd, h_est, x_est, A=1.25)` so `A is None` or `math.isinf(A)` applies linear PA (`x_pa_freq = x_est`), and updated `forward(self, out, frame, A=None)` to resolve `A` dynamically.
   - **M2.3 Comparative Baselines & Benchmark Runner (`p4/em_floatformer.py:1291-1778`)**:
     - `ComplexConv1d`: bias-free circular complex 1D convolution.
     - `LocalEquivConv16QAM` (`14,198` params, within `0.77%` of `EMFloatFormer` `14,090` params): genuine $C^*$-equivariant local circular Conv1d 16-QAM receiver using `apply_gauge_normalization`, `ComplexRMSNorm`, `ComplexConv1d`, `RadialActivation`, and calibrated `2.5e-2` / `20.0 * tanh` LLRs (no global Hermitian attention).
     - `TinyJoint16QAM` (`1,880` params): genuine non-equivariant Conv1d 16-QAM baseline inspired by `exploratory/16qam_joint_pa_probe/probe.py` (with `head_h` and `head_dist` feeding into `h_hat` and `z_corr`, returning `FloatFormerOutput`).
     - `compute_ece(llrs, bits, n_bins=10)`: 10-bin reliability ECE over `[0.5, 1.0]`.
     - `run_p4a_comparative_benchmarks(...)`: genuinely trains `EMFloatFormer`, `LocalEquivConv16QAM`, and `TinyJoint16QAM` (`240` Adam + CosineAnnealing steps, mixed `strong`/`mild`/`clean` PA, `snr_db=(10.0, 30.0)`) under `A`-resolved `JointPhysicalLoss(1.0, 0.25)` and evaluates all 5 methods (`EMFloatFormer`, `LocalEquivConv16QAM`, `TinyJoint16QAM`, `Bussgang-LMMSE`, `DD-PA-Cancel` iterations=2) across `pa_condition in ("strong", "mild", "clean")` and `snr_db in (10, 15, 20, 25, 30)` (`256` frames = `32,768` bits per cell), saving `p4/results/p4a_comparative_benchmark.json` and `p4/results/p4a_comparative_benchmark.csv`.

2. **`p4/__init__.py` (`lines 64-70, 129-135`)**:
   - Kept all existing M1 and 37 M2 exports intact; added `ComplexConv1d`, `LocalEquivConv16QAM`, `TinyJoint16QAM`, `compute_ece`, and `run_p4a_comparative_benchmarks`.

3. **`p4/tests/test_p4_a_equivariance.py`**:
   - Extended `test_complex_rms_norm_subnormal_guard` (`lines 89-111`) with backward finite-gradient checks at `1e-35` and `1e20`.
   - Added `test_gauge_anchor_subnormal_1e35_no_collapse` (`lines 113-122`).
   - Extended `test_radial_activation_zero_gradient_finite` (`lines 252-269`) across `silu`, `gelu`, `relu`, `tanh` at both `z=0` and `1e20 + 1e20j` forward + backward.
   - Extended `test_create_floatformer_factory` (`lines 331-351`) to assert zero `None`, non-finite, or `0.0` parameter gradients across all 8 `(use_gauge_norm, use_hermitian_attention, use_joint_pilot_data)` configurations.
   - Added `test_fresh_seed11_and_trained_depth2_c_star_equivariance` (`lines 498-548`) verifying `max_llr_err < 1e-5`, `flips == 0`, and `h_rel < 1e-5` across `(2^-12, 0.13, 3.7, 1023.5, 2^12)` and random phases.
   - Added `TestM2CalibrationAndBaselines` (`lines 551-656`):
     - `test_calibrated_llr_dynamic_range_ece_and_equivariance_high_snr`: verifies `ECE < 0.05`, high-SNR `max|LLR| > 15.0` (`30 dB clean`), `max_llr_err < 1e-5`, `flips == 0`, and `h_rel < 1e-5` across `(clean, mild, strong)` × `(15, 20, 30)` dB on trained `EMFloatFormer(depth=2)`.
     - `test_joint_loss_mixed_pa_resolution_and_scale_stability`: verifies `_resolve_pa_A` on `pa_condition="clean"` / `A=2.25` and mixed-PA training `scale_ratio` converging near `1.0` (`|sr - 1.0| < 0.08`).
     - `test_comparative_baseline_models_equivariance_and_params`: verifies `LocalEquivConv16QAM` parameter matching (`< 5%` diff vs `EMFloatFormer`), `C*` equivariance (`< 1e-5`, `0` flips), and `TinyJoint16QAM` forward/backward finiteness on all parameters.

### 1.2 Comparative Benchmark Summary (`p4/results/p4a_comparative_benchmark.csv`)
Executed with `torch.set_num_threads(4)`, `seed=2026`, `train_steps=240`, `eval_frames=256` (`32,768` bits/cell).
- **Parameter Counts**: `EMFloatFormer = 14,090`, `LocalEquivConv16QAM = 14,198`, `TinyJoint16QAM = 1,880`, `Bussgang-LMMSE = 0`, `DD-PA-Cancel = 0`.
- **Mixed-PA Final Training `scale_ratio`**: `EMFloatFormer = 0.9770`, `LocalEquivConv16QAM = 0.9849`, `TinyJoint16QAM = 0.9784` (upward `1.1495` scale drift eliminated).

| PA Condition (`A`) | SNR (dB) | Method | BER | NMSE | BCE | ECE | `scale_ratio` |
|---|---|---|---|---|---|---|---|
| `strong` (`1.25`) | `10` | `EMFloatFormer` | `0.1901` | `0.0887` | `0.3898` | `0.0340` | `0.8935` |
| `strong` (`1.25`) | `10` | `LocalEquivConv16QAM` | `0.1765` | `0.0567` | `0.3642` | `0.0248` | `0.8898` |
| `strong` (`1.25`) | `10` | `TinyJoint16QAM` | `0.1802` | `0.0726` | `0.4765` | `0.0813` | `0.8684` |
| `strong` (`1.25`) | `10` | `Bussgang-LMMSE` | `0.1806` | `0.0564` | `0.3668` | `0.0123` | `1.0266` |
| `strong` (`1.25`) | `10` | `DD-PA-Cancel` | `0.1870` | `0.0564` | `0.3946` | `0.0338` | `1.0266` |
| `strong` (`1.25`) | `20` | `EMFloatFormer` | `0.0508` | `0.0400` | `0.1228` | `0.0134` | `0.8561` |
| `strong` (`1.25`) | `20` | `LocalEquivConv16QAM` | `0.0453` | `0.0310` | `0.1058` | `0.0058` | `0.8685` |
| `strong` (`1.25`) | `20` | `TinyJoint16QAM` | `0.0526` | `0.0361` | `0.1563` | `0.0179` | `0.8625` |
| `strong` (`1.25`) | `20` | `Bussgang-LMMSE` | `0.0472` | `0.0104` | `0.1083` | `0.0033` | `1.0103` |
| `strong` (`1.25`) | `20` | `DD-PA-Cancel` | `0.0399` | `0.0104` | `0.0946` | `0.0105` | `1.0103` |
| `strong` (`1.25`) | `30` | `EMFloatFormer` | `0.0141` | `0.0387` | `0.0430` | `0.0058` | `0.8462` |
| `strong` (`1.25`) | `30` | `LocalEquivConv16QAM` | `0.0120` | `0.0305` | `0.0304` | `0.0013` | `0.8603` |
| `strong` (`1.25`) | `30` | `TinyJoint16QAM` | `0.0229` | `0.0314` | `0.0963` | `0.0291` | `0.8624` |
| `strong` (`1.25`) | `30` | `Bussgang-LMMSE` | `0.0121` | `0.0058` | `0.0295` | `0.0013` | `1.0051` |
| `strong` (`1.25`) | `30` | `DD-PA-Cancel` | `0.0067` | `0.0058` | `0.0191` | `0.0029` | `1.0051` |
| `mild` (`2.25`) | `10` | `EMFloatFormer` | `0.1528` | `0.0680` | `0.3168` | `0.0218` | `1.0429` |
| `mild` (`2.25`) | `10` | `LocalEquivConv16QAM` | `0.1421` | `0.0438` | `0.2948` | `0.0149` | `1.0472` |
| `mild` (`2.25`) | `10` | `TinyJoint16QAM` | `0.1475` | `0.0522` | `0.3924` | `0.0552` | `1.0241` |
| `mild` (`2.25`) | `10` | `Bussgang-LMMSE` | `0.1448` | `0.0380` | `0.3003` | `0.0145` | `1.0207` |
| `mild` (`2.25`) | `10` | `DD-PA-Cancel` | `0.1451` | `0.0380` | `0.3042` | `0.0202` | `1.0207` |
| `mild` (`2.25`) | `20` | `EMFloatFormer` | `0.0323` | `0.0206` | `0.0756` | `0.0062` | `1.0022` |
| `mild` (`2.25`) | `20` | `LocalEquivConv16QAM` | `0.0280` | `0.0151` | `0.0646` | `0.0026` | `1.0177` |
| `mild` (`2.25`) | `20` | `TinyJoint16QAM` | `0.0325` | `0.0182` | `0.1152` | `0.0248` | `1.0141` |
| `mild` (`2.25`) | `20` | `Bussgang-LMMSE` | `0.0284` | `0.0054` | `0.0638` | `0.0045` | `1.0029` |
| `mild` (`2.25`) | `20` | `DD-PA-Cancel` | `0.0264` | `0.0054` | `0.0605` | `0.0050` | `1.0029` |
| `mild` (`2.25`) | `30` | `EMFloatFormer` | `0.0072` | `0.0168` | `0.0220` | `0.0035` | `0.9962` |
| `mild` (`2.25`) | `30` | `LocalEquivConv16QAM` | `0.0043` | `0.0123` | `0.0118` | `0.0013` | `1.0150` |
| `mild` (`2.25`) | `30` | `TinyJoint16QAM` | `0.0125` | `0.0135` | `0.0647` | `0.0248` | `1.0203` |
| `mild` (`2.25`) | `30` | `Bussgang-LMMSE` | `0.0042` | `0.0017` | `0.0096` | `0.0015` | `1.0026` |
| `mild` (`2.25`) | `30` | `DD-PA-Cancel` | `0.0035` | `0.0017` | `0.0085` | `0.0012` | `1.0026` |

*(Full 75-row grid covering `strong`, `mild`, `clean` × `{10, 15, 20, 25, 30}` dB is stored in `p4/results/p4a_comparative_benchmark.json` and `p4/results/p4a_comparative_benchmark.csv`.)*

### 1.3 Verification & Governance Results
| Check / Suite | Command | Total | Passed | Skipped | Failed | Exit Code | Duration |
|---|---|---|---|---|---|---|---|
| Stage P4-0 Unit | `python -m unittest p4.tests.test_p4_0_simulator -v` | 14 | 14 | 0 | 0 | 0 | 2.010s |
| Stage P4-A Unit | `python -m unittest p4.tests.test_p4_a_equivariance -v` | 50 | 50 | 0 | 0 | 0 | 47.312s |
| Master E2E (Tiers 1–4) | `python -m unittest tests.test_p4_e2e -v` | 253 | 156 | 97 (M3/M4) | 0 | 0 | 14.358s |
| Syntax check | `python -m py_compile p4/em_floatformer.py p4/__init__.py p4/tests/test_p4_a_equivariance.py` | — | — | — | — | 0 | — |
| Governance check | `git diff --name-only` | 0 tracked modified | — | — | — | 0 | — |

---

## 2. Logic Chain
1. Preserving the `complex128`/`float64` internal backbone execution in `EMFloatFormer.forward` (`p4/em_floatformer.py:595-652`) allows relaxing the LLR variance floor from `5.0e-2` to `2.5e-2` and raising the `tanh` ceiling from `8.0` to `20.0`, restoring 2.5× high-SNR soft-LLR dynamic range (`max|LLR| > 15.0` at 30 dB) and low calibration error (`ECE < 0.05` across all PA/SNR cells) while staying strictly within `< 1e-5` LLR equivariance error and `0` bit flips across 24 gain octaves and non-power-of-2 scales.
2. Hardcoding `A=1.25` in `JointPhysicalLoss.forward()` penalized `mild` (`A=2.25`) and `clean` frames for insufficient PA compression, forcing `h_hat` upward (`scale_ratio = 1.1495`). Resolving `A` per frame/batch (`1.25` / `2.25` / linear PA for `clean`) keeps mixed-PA training `scale_ratio` tightly centered at `0.977–0.985 ≈ 1.0`.
3. Connecting `TinyJoint16QAM.head_dist` into `z_corr` (mirroring `EMFloatFormer` and `LocalEquivConv16QAM`'s distortion subtraction) ensures every trainable tensor receives finite non-zero gradients under `JointPhysicalLoss` and avoids stale/disconnected head parameters.
4. Executing all three verification suites (`test_p4_0_simulator`, `test_p4_a_equivariance`, `tests.test_p4_e2e`) and `git diff --name-only` confirms Stage P4-A (F8–F12) meets all numerical, calibration, comparative, and governance criteria without touching any historical or Stage P4-0 files.

---

## 3. Caveats
- All benchmarks and tests ran on CPU with `torch.set_num_threads(4)` / `OMP_NUM_THREADS=4`.
- Non-privileged neural receivers (`EMFloatFormer`, `LocalEquivConv16QAM`, `TinyJoint16QAM`) do not receive `A` at inference time; because `yp = H * gamma * X_p + W_p` already embeds Bussgang attenuation `|gamma| ≈ 0.8035` (`strong`) vs `1.0` (`clean`), per-condition evaluation `scale_ratio` centers near `~0.85–0.89` (`strong`), `~1.00–1.02` (`mild`), and `~1.04–1.08` (`clean`), while the mixed-PA training `scale_ratio` centers at `0.977–0.985`.
- Running `tests.test_p4_e2e` executes Tier 1/Tier 2 F7 latency tests, which write `p4/results/latency_cpu.csv` and `p4/results/timing.json` during test execution (we did not manually edit those files).

---

## 4. Conclusion
Milestone 2 (Stage P4-A: Features F8–F12) is ready for Gate review and sign-off:
- **M2.1**: Iteration 3 numerical hardening is preserved and locked down by 5 new automated regression tests in `p4/tests/test_p4_a_equivariance.py`.
- **M2.2**: LLR variance floor (`2.5e-2`) and saturation bound (`20.0 * tanh`) preserve high-SNR dynamic range (`|LLR|` up to 20, `ECE < 0.05`) and strict $C^*$ equivariance (`< 1e-5`, `0` flips).
- **M2.3**: `JointPhysicalLoss` mixed-PA scale drift is fixed (`scale_ratio ≈ 0.98`), `LocalEquivConv16QAM` (`14,198` params) and `TinyJoint16QAM` (`1,880` params) are implemented in `p4/em_floatformer.py`, and genuine 5-method comparative benchmarks are saved to `p4/results/p4a_comparative_benchmark.{json,csv}`.
- **M2.4**: `test_p4_0_simulator` (14/14), `test_p4_a_equivariance` (50/50), `tests.test_p4_e2e` (253 total / 156 active passed / 97 M3-M4 skipped), and `git diff --name-only` (empty) all pass.

---

## 5. Verification Method
From ``:
```powershell
$env:OMP_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'
python -m unittest p4.tests.test_p4_0_simulator -v
python -m unittest p4.tests.test_p4_a_equivariance -v
python -m unittest tests.test_p4_e2e -v
git diff --name-only
```
Files to inspect:
- `p4/em_floatformer.py`
- `p4/__init__.py`
- `p4/tests/test_p4_a_equivariance.py`
- `p4/results/p4a_comparative_benchmark.json`
- `p4/results/p4a_comparative_benchmark.csv`

**Invalidation conditions**: Any failure in the three unittest suites above, `max_llr_err >= 1e-5` or `flips > 0` on fresh/trained `depth=2` models, `ECE >= 0.05`, mixed-PA training `scale_ratio` drifting away from `1.0`, or any tracked historical P1–P3 / `exploratory/*` / Stage P4-0 file appearing in `git diff --name-only`.
