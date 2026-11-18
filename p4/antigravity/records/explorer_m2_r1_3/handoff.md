# Explorer Handoff Report — Milestone 2 (Stage P4-A): Joint Training, Comparative Benchmarks & Governance (`explorer_m2_r1_3`)

**Core Summary:** Stage P4-A (`p4/em_floatformer.py`) implements `EMFloatFormer` (`14,090` params default), `create_floatformer`, `JointPhysicalLoss`, and F8–F12 unit/equivariance APIs, but **`p4/results/` contains zero Stage P4-A comparative benchmark data** (only Stage P4-0 latency files), **no parameter-matched 16-QAM local equivariant convolution, `TinyJoint` adapter, or comparative benchmark runner exists inside `p4/`**, and `JointPhysicalLoss.forward()` hardcodes `A=1.25` (causing `1.1495` channel scale drift on mixed `A=1.25/2.25` training) while `EMFloatFormer.forward()` caps LLRs at `±8.0` with a `5.0e-2` variance floor. All historical P1–P3 / `exploratory/*` files and Stage P4-0 source modules remain 100% unmodified and passing.

---

## 1. Observation

### 1.1 Existing Model, Loss, and Benchmark APIs

| Component | Exact Location | Signature / Details | Status in `p4/` |
|---|---|---|---|
| `EMFloatFormer` | `p4/em_floatformer.py:509-644` | `EMFloatFormer(d_model=12, n_heads=2, depth=2, n_sc=32, ...)` -> `FloatFormerOutput(h_hat, llrs, dist_est)`. Params: `14,090` (`depth=2, d=12`), `7,106` (`depth=1, d=12`), `24,930` (`d=16, depth=2`), `394,626` (`d=64, depth=2`). | Implemented |
| `create_floatformer` | `p4/em_floatformer.py:647-659` | `create_floatformer(use_gauge_norm=True, use_hermitian_attention=True, use_joint_pilot_data=True, **kwargs)` | Implemented |
| `JointPhysicalLoss` | `p4/em_floatformer.py:787-854` | `JointPhysicalLoss(lambda_nmse=1.0, lambda_pa=0.25)`; `bce_loss`, `nmse_loss`, `pa_residual_loss(..., A=1.25)`, `forward(out, frame)` | Implemented (hardcodes `A=1.25` in `forward`) |
| `compare_pilot_vs_bidirectional` | `p4/em_floatformer.py:662-715` | Trains `EMFloatFormer(d_model=12, n_heads=2, depth=2)` for 50 Adam steps (`snr=18.0`, `pa="strong"`, `lambda_nmse=2.0, lambda_pa=0.0`) vs pilot-only LS. Measured return: `0.7205619540909347` dB. | Implemented (single-point smoke check only) |
| `evaluate_noise_data_feedback` | `p4/em_floatformer.py:718-741` | Untrained `EMFloatFormer(depth=1)` NMSE change with noisy `yd`. Measured return: `0.0008504301395792783` dB. | Implemented |
| `evaluate_perfect_pilot_demapping` | `p4/em_floatformer.py:744-761` | Untrained `EMFloatFormer(depth=1)` BER at `snr=50 dB`, `clean`, `fd=0.0`. Measured return: `0.0`. | Implemented |
| `bussgang_lmmse` | `p4/classical_baselines.py:240-318` | `bussgang_lmmse(yp, yd, nv, A=2.25, est_method="dft8", ...) -> EqualizerResult` (`h_hat, h_eff, llrs, hard_bits, z_hat, mu, sigma2_eff, w_filter`) | Implemented (M1) |
| `iterative_dd_pa_cancellation` | `p4/classical_baselines.py:336-466` | `iterative_dd_pa_cancellation(yp, yd, nv, A=1.25, iterations=2, ...) -> (h_hat, llrs, hard_bits, history)` | Implemented (M1) |
| `TinyJoint` | `exploratory/16qam_joint_pa_probe/probe.py:83-102` | `TinyJoint(width=16)`: `1,812` params, non-equivariant `Conv1d`, hardcoded `N=32`, uses `delay_est(yp)` + `equalize`, outputs uncalibrated 16-QAM logits `(B, 32, 4)` (no `h_hat`/`dist_est`). | **NOT in `p4/`** (only in read-only `exploratory/`) |
| `EquivComplexLocal` / `EquivDelayConv` | `p2_models.py:111-136`, `p3_models.py:13-54` | `EquivComplexLocal(kernel=9)` (`2,060` params) & `EquivDelayConv` are QPSK-only (`bits_logits` returning `(B, N, 2)`). | **NOT in `p4/`** (no 16-QAM parameter-matched local equiv conv exists) |
| Stage P4-A Benchmark / Training Runner | — | Controlled benchmark across `EMFloatFormer`, param-matched local equiv conv, `TinyJoint`, `Bussgang-LMMSE`, `DD-PA-Cancel` across SNRs and `A in {1.25, 2.25}`. | **MISSING** |

### 1.2 E2E Coverage for F8–F12 (`tests/e2e/`)
- `tests/e2e/test_tier1_feature_coverage.py:486-731` (25 tests: 5 each for F8–F12): tests `compute_gauge_anchor`, `apply_gauge_normalization`, `ComplexRMSNorm`, `rescale_channel_estimate`, `EMFloatFormer` scale invariance (`4096.0`), `ComplexLinear`, `ComplexRoPE`, `hermitian_attention_scores`, `RadialActivation`, `JointEmbedding`, `BidirectionalBlock`, `compare_pilot_vs_bidirectional` (`>= 0.5 dB`), `JointPhysicalLoss`, `verify_loss_gradient_invariance`, and F12 audit functions.
- `tests/e2e/test_tier2_boundary_corner.py:415-626` (25 tests: 5 each for F8–F12): tests `2^-12`, `2^12`, zero power, `pi` phase, `1e-35` ComplexRMSNorm, `N=1`, `evaluate_noise_data_feedback` (`< 1.0 dB`), `evaluate_perfect_pilot_demapping` (`< 1e-3`), `audit_loss_scale_invariance`, dense 100-phase sweep, irrational phases, quadratic noise scaling.
- `tests/e2e/test_tier3_cross_feature.py:156-232`: covers P08 (`F8+F9`), P09 (`F8+F12`), P11 (`F9+F10`), P13 (`F10+F11`), P14 (`F11+F12`).
- **Gap**: None of the E2E F8–F12 tests or `p4/tests/test_p4_a_equivariance.py` check multi-SNR/multi-PA comparative benchmarks against `TinyJoint`, parameter-matched local equivariant convolution, `Bussgang-LMMSE`, or `DD-PA-Cancel`, nor LLR ECE/BCE calibration at high SNR (`25–30 dB`).

### 1.3 `p4/results/` Directory Contents
Inspected `p4\results\`:
- Contains **only two files**:
  1. `p4/results/latency_cpu.csv` (`917` bytes, `LastWriteTime: 2026/11/17 18:44:22`)
  2. `p4/results/timing.json` (`2552` bytes, `LastWriteTime: 2026/11/17 18:44:22`)
- **Missing**: Zero Stage P4-A benchmark artifacts (no BER, NMSE, LLR BCE/ECE, or parameter-count comparison CSV/JSON across SNRs or `A=1.25`, `A=2.25`).
- **Runtime Side-Effect Observed**: Both `tests/e2e/test_tier1_feature_coverage.py:436-458` (`test_t1_f07_benchmark_script_invocation`) and `tests/e2e/test_tier2_boundary_corner.py:396` (`test_t2_f07_single_thread_concurrency`) call `run_benchmarks(..., output_dir="p4/results")`, overwriting `latency_cpu.csv` and `timing.json` on every E2E run (currently recording `num_threads=1, batch_size=1, warmup_runs=1, timed_runs=3` from `18:44:22`).

### 1.4 Empirical `JointPhysicalLoss` Convergence & LLR Calibration Diagnostics (Read-Only Runs)
1. **Single-PA `strong` (`A=1.25`, `snr=18.0 dB`, `EMFloatFormer(d_model=12, n_heads=2, depth=2)`, `JointPhysicalLoss(lambda_nmse=1.0, lambda_pa=0.25)`, 100 Adam steps, batch=32)**:
   - `step=  1`: `loss=0.4205, bce=0.2479, nmse=0.1406, pa=0.1281, ber=0.1123, scale_ratio=0.8223`
   - `step= 25`: `loss=0.2761, bce=0.2125, nmse=0.0514, pa=0.0488, ber=0.0955, scale_ratio=0.9404`
   - `step= 50`: `loss=0.2557, bce=0.2133, nmse=0.0325, pa=0.0397, ber=0.0913, scale_ratio=0.9837`
   - `step=100`: `loss=0.1839, bce=0.1558, nmse=0.0197, pa=0.0337, ber=0.0632, scale_ratio=1.0086`
   - `scale_ratio = ||h_hat|| / ||h_true||` converges cleanly to `1.0086` without scale drift.
2. **Mixed-PA (`strong` `A=1.25` + `mild` `A=2.25`, `snr in [10, 30] dB`, 150 Adam steps)**:
   - `step=  1 (strong)`: `nmse=0.1632, scale_ratio=0.8273`
   - `step= 50 (mild)`: `nmse=0.1391, scale_ratio=1.1286`
   - `step=150 (mild)`: `nmse=0.0789, scale_ratio=1.1495` (upward scale drift!)
   - Cross-SNR evaluation (`batch_size=128` per cell) of that 150-step model vs `Bussgang-LMMSE` and `DD-PA-Iter2`:
     - `A=1.25, 10 dB`: EMFF `BER=0.2031, NMSE=0.0936, BCE=0.4006, ECE=0.0303` | Buss `BER=0.1881, NMSE=0.0596, BCE=0.3762, ECE=0.0166` | DD2 `BER=0.1948, BCE=0.4052, ECE=0.0372`
     - `A=1.25, 20 dB`: EMFF `BER=0.0576, NMSE=0.0253, BCE=0.1338, ECE=0.0058` | Buss `BER=0.0469, NMSE=0.0107, BCE=0.1130, ECE=0.0065` | DD2 `BER=0.0403, BCE=0.1015, ECE=0.0114`
     - `A=1.25, 30 dB`: EMFF `BER=0.0185, NMSE=0.0200, BCE=0.0498, ECE=0.0044, maxLLR=8.00` | Buss `BER=0.0104, NMSE=0.0055, BCE=0.0265, ECE=0.0018` | DD2 `BER=0.0061, BCE=0.0145, ECE=0.0021`
     - `A=2.25, 10 dB`: EMFF `BER=0.1634, NMSE=0.0826, BCE=0.3343, ECE=0.0153` | Buss `BER=0.1533, NMSE=0.0401, BCE=0.3145, ECE=0.0175` | DD2 `BER=0.1549, BCE=0.3195, ECE=0.0221`
     - `A=2.25, 30 dB`: EMFF `BER=0.0088, NMSE=0.0081, BCE=0.0267, ECE=0.0030, maxLLR=8.00` | Buss `BER=0.0038, NMSE=0.0017, BCE=0.0079, ECE=0.0011` | DD2 `BER=0.0036, BCE=0.0074, ECE=0.0010`
3. **Inspection of `p4/em_floatformer.py:640-642`**:
   - Line 640: `sigma_eff2 = nv_exp / (h_power_det + 1e-4) + dist_power_det / (h_power_det + 1e-4) + 5.0e-2`
   - Line 642: `llrs = (8.0 * torch.tanh(llrs64 / 8.0)).to(out_real_dtype)`

### 1.5 Governance & Stage P4-0 Preservation Check
- `git log -n 5 --oneline`: HEAD is `f878a73 Publish independent TDL-table surrogate and 10-seed audit; retain Sionna P3-D/E blocker`.
- `git diff --name-only` and `git ls-files -m`: **empty** (0 modified tracked files).
- `git status --short`: only untracked Stage P4 / teamwork paths (`?? .agents/`, `?? PROJECT.md`, `?? TEST_INFRA.md`, `?? TEST_READY.md`, `?? p4/`, `?? tests/`).
- `MANIFEST_SHA256.json`: 115 entries checked — 0 missing, 114 exact matches (the single mismatch `README.md` is from pre-existing commit `f878a73`, explicitly documented in `tests/e2e/test_tier1_feature_coverage.py:1128-1130`).
- `P2_SUPPLEMENT_MANIFEST_SHA256.json`: 26 entries checked — 0 missing, 25 exact matches (`README.md` only mismatch, same pre-existing commit).
- `exploratory/p3de_tdl_surrogate/P3DE_SOURCE_MANIFEST_SHA256.json`: 0 mismatches among paths relative to its subdirectory (`34` entries; paths are relative to `exploratory/p3de_tdl_surrogate/`).
- Stage P4-0 source/test files (`p4/simulator.py`, `p4/classical_baselines.py`, `p4/benchmark_latency.py`, `p4/tests/test_p4_0_simulator.py`) retain `2026/11/15` timestamps, and `python -m unittest p4.tests.test_p4_0_simulator -v` passes **14/14 tests in 3.550s** (exit code 0).

---

## 2. Logic Chain

1. **Why Stage P4-A Comparative Benchmarks Are Incomplete**: Observations 1.1–1.3 show that `p4/em_floatformer.py` only contains `compare_pilot_vs_bidirectional()` (a 50-step `strong`-PA check against raw LS) and unit/equivariance verifiers. Neither `TinyJoint` (`exploratory/16qam_joint_pa_probe/probe.py`) nor a 16-QAM parameter-matched local equivariant convolution (`~14k` params to match `EMFloatFormer(d=12, depth=2)`) exists inside `p4/`, and `p4/results/` holds zero P4-A metric files.
2. **Why Mixed-PA Training Drifts Channel Scale (`1.1495`)**: Observation 1.4 shows `JointPhysicalLoss.forward()` (`p4/em_floatformer.py:852`) invokes `self.pa_residual_loss(frame.yd, h_est, x_est)` with hardcoded `A=1.25` (and `oversample=1`). On `A=1.25` alone, `scale_ratio` converges to `1.0086`, whereas on `A=2.25` (`mild`) frames, `pa_residual_loss` expects stronger `A=1.25` compression and penalizes `yd - h_est * x_pa(A=1.25)`, forcing `h_est` upward (`scale_ratio = 1.1495`) and inflating `NMSE` (`0.0789–0.0826`).
3. **Why High-SNR LLR Calibration & Performance Lag Baselines (`30 dB`)**: Observation 1.4 shows `sigma_eff2` adds a constant `+ 5.0e-2` floor (corresponding to ~13 dB effective SNR ceiling) and `8.0 * torch.tanh(llrs64 / 8.0)` clips LLRs at `±8.0`. Together with only 150 mixed-PA steps (vs `19,200` frames / 300+ steps in `probe.py`) and `pa_residual_loss` `A=1.25` mismatch, `EMFloatFormer` yields `BCE=0.0498 / BER=0.0185` at `30 dB, A=1.25` vs `DD-PA-Iter2` `BCE=0.0145 / BER=0.0061` (and `ECE=0.0044` vs `0.0021`). Note also from `auditor_m2_2/handoff.md` and `challenger_m2_3/handoff.md` that tightening `sigma_eff2` or `tanh` interacts with the `< 1e-5` LLR equivariance bound when inputs are scaled in `complex64`.
4. **Governance & Stage P4-0 Integrity**: Observation 1.5 proves zero modifications to tracked P1–P3 / `exploratory/*` files and intact Stage P4-0 code/tests (`14/14` passing), with only runtime overwrites of `p4/results/{latency_cpu.csv,timing.json}` caused by E2E F7 tests.

---

## 3. Caveats

- Only a 100-step (`A=1.25`) and 150-step (mixed `A=1.25/2.25`) CPU training probe of `EMFloatFormer(d_model=12, n_heads=2, depth=2)` was run; larger widths (`d_model=16/32/64`), longer training schedules, `lambda_pa=0` / `A`-conditioned `pa_residual_loss`, and `sigma_eff2` / `tanh` bound sweeps were not exhaustively tuned because this is a read-only exploration.
- `P3_A2_PUBLIC_SHA256_MANIFEST.json` reports missing untracked log/result files (`p3a2_results/*`, `p3_a.py`) that were excluded from the git repo commit `f878a73`; `P3_A_MECHANISM_PROTOCOL.md` hash in `P3_A2_PUBLIC_SHA256_MANIFEST.json` also predates its committed state (same in clean HEAD).

---

## 4. Conclusion

1. **Existing APIs Ready**: `EMFloatFormer`, `create_floatformer`, `JointPhysicalLoss`, F8–F12 audit functions (`p4/em_floatformer.py`), `bussgang_lmmse`, and `iterative_dd_pa_cancellation` (`p4/classical_baselines.py`) are implemented and callable.
2. **Required Stage P4-A Additions / Calibration**:
   - **Fix `JointPhysicalLoss` PA parameter & oversampling**: Pass `A` (e.g., `getattr(frame, "A", A)` or explicit `A` arg in `JointPhysicalLoss.forward`) instead of hardcoding `A=1.25` when training across both `A=1.25` and `A=2.25` so channel `scale_ratio` stays `~1.0` without drift.
   - **Calibrate `p4/em_floatformer.py:640-642`**: Audit `sigma_eff2` floor (`5.0e-2`) and `tanh` cap (`8.0`) jointly against (a) `< 1e-5` C* equivariance on trained `depth=2` in `complex64` caller inputs, and (b) high-SNR LLR dynamic range / BCE / ECE needed for Stage P4-B soft FEC.
   - **Add Missing Comparative Baselines & Benchmark Artifact Generation**: Implement a 16-QAM parameter-matched local equivariant convolution (`~14k` params) and 16-QAM `TinyJoint` baseline in `p4/` (without altering `exploratory/` or `p2_models.py`), run controlled comparative benchmarks vs `EMFloatFormer`, `Bussgang-LMMSE`, and `DD-PA-Cancel` (`A=1.25` and `A=2.25`, multi-SNR), and save BER, NMSE, LLR BCE/ECE, and parameter counts under `p4/results/` (distinct from `latency_cpu.csv` and `timing.json`).
3. **Governance**: Historical P1–P3, `exploratory/*`, and Stage P4-0 source/tests are 100% intact. Restore/re-run `python -m p4.benchmark_latency --threads 4 --batch-sizes 1 64` after E2E runs if canonical 4-thread `p4/results/latency_cpu.csv` / `timing.json` artifacts are desired.

---

## 5. Verification Method

From ``:
1. **Stage P4-0 Regression & Governance**:
   ```powershell
   git diff --name-only
   python -m unittest p4.tests.test_p4_0_simulator -v
   python -m unittest -k "f22" tests/test_p4_e2e.py -v
   ```
2. **Inspect `p4/results/` & Training Convergence**:
   ```powershell
   Get-ChildItem p4\results
   python -c "import p4.em_floatformer as em; print('pilot_vs_bidi:', em.compare_pilot_vs_bidirectional())"
   ```
3. **Invalidation Conditions**: Any non-empty `git diff --name-only` on historical files, any failure in `test_p4_0_simulator` or `f22` manifest tests, or `JointPhysicalLoss` channel `scale_ratio` diverging from `1.0` under `A=2.25` after `A`-conditioning fix.
