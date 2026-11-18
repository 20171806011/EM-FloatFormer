# Milestone 2 (Stage P4-A) Independent Review & Adversarial Challenge Report (`reviewer_m2_4`)

**Verdict**: **APPROVE**
**Overall Risk Assessment**: **LOW** (All integrity violations remediated; all 45 unit tests, 156 active E2E tests, and 14 simulator regression tests pass independently; historical repository state is immutable)

---

## 1. Observation

### 1.1 Files & Contracts Inspected
- `p4/em_floatformer.py` (1,225 lines): Remediated Stage P4-A implementation (`F8`–`F12`).
- `p4/__init__.py` (124 lines): Package exports (20 M1 symbols + 36 M2 symbols, including `create_floatformer`).
- `p4/tests/test_p4_a_equivariance.py` (462 lines): 45 unit tests across `TestF8GaugeNormalization` (8), `TestF9HermitianGreenAttention` (12), `TestF10BidirectionalAttention` (7), `TestF11JointPhysicalLoss` (9), `TestF12NumericalEquivariance` (9).

### 1.2 Independent Test Suite Execution Results
All commands executed from ``:
1. `python -m unittest p4/tests/test_p4_a_equivariance.py -v`
   - **Result**: `Ran 45 tests in 275.051s` — **`OK`** (exit code `0`).
2. `python -m unittest tests/test_p4_e2e.py -v`
   - **Result**: `Ran 253 tests in 278.030s` — **`OK (skipped=97)`** (156 active tests passed including all 55 M2 tests across Tier 1, Tier 2, and Tier 3, plus M1 and F22; 97 M3/M4 tests progressively skipped; exit code `0`).
3. `python -m unittest p4/tests/test_p4_0_simulator.py -v`
   - **Result**: `Ran 14 tests in 59.290s` — **`OK`** (exit code `0`).

### 1.3 Interface Conformance & Parameter Budget Observations
1. **Exports in `p4/__init__.py` (lines 27–64, 87–123)**:
   - Exports all 36 M2 symbols (the 35 symbols previously audited plus `create_floatformer` at lines 43 and 103).
   - Every callable/class queried by `MilestoneResolver` in Tier 1 (lines 486–731), Tier 2 (lines 415–626), and Tier 3 (P08, P09, P11, P13, P14) resolves cleanly with matching signatures.
   - `create_floatformer(use_gauge_norm: bool = True, use_hermitian_attention: bool = True, use_joint_pilot_data: bool = True, **kwargs: Any) -> EMFloatFormer` (`p4/em_floatformer.py:629-641`) matches `PROJECT.md:99`.
   - `FloatFormerOutput` (`p4/em_floatformer.py:482-492`) inherits from `dict` (`out["h_hat"]`, `out["llrs"]`, `out["dist_est"]`) and defines `__iter__` returning `(self.h_hat, self.llrs, self.dist_est)`, satisfying both dict-style E2E calls and tuple-unpacking `PROJECT.md:90`.
2. **Parameter Budget**:
   - Default `EMFloatFormer(d_model=12, n_heads=2, depth=2, n_sc=32)`: **14,090 parameters**, **104 parameter tensors** (`zero_grads=0`, `nan_grads=0`).
   - `EMFloatFormer(depth=1)`: **7,106 parameters**, **58 parameter tensors**.
   - All 8 `create_floatformer` ablation variants:
     - `(g=True, h=True, j=True)`: `14,090` params, `zero_grads=0, nan_grads=0`
     - `(g=True, h=True, j=False)`: `9,482` params, `zero_grads=4, nan_grads=0` (unused `norm_p2`/`norm_d2` when cross-attention is disabled)
     - `(g=True, h=False, j=True)`: `24,074` params, `zero_grads=0, nan_grads=0`
     - `(g=True, h=False, j=False)`: `14,474` params, `zero_grads=4, nan_grads=0`
     - `(g=False, h=True, j=True)`: `13,922` params, `zero_grads=0, nan_grads=0`
     - `(g=False, h=True, j=False)`: `9,314` params, `zero_grads=0, nan_grads=0`
     - `(g=False, h=False, j=True)`: `23,906` params, `zero_grads=0, nan_grads=0`
     - `(g=False, h=False, j=False)`: `14,306` params, `zero_grads=0, nan_grads=0`

### 1.4 Integrity, Shortcut & Remediation Verification
1. **Source Scan (`p4/em_floatformer.py`)**:
   - Substring scan for `'0.75'`, `'min(deg_db'`, `'max(gain_db'`, `'iterative_dd_pa_cancellation'`, and `'bussgang_lmmse'`: all **`False`** (0 occurrences).
2. **`compare_pilot_vs_bidirectional()` (`p4/em_floatformer.py:644-697`)**:
   - Trains `EMFloatFormer(d_model=12, n_heads=2, depth=2)` for 50 Adam steps on `seed=101` strong-PA frames (`batch_size=128`, mini-batches of 32, `snr_db=18.0`, `pa_condition="strong"`) and evaluates against pilot-only LS on unseen `seed=2026` strong-PA frames (`batch_size=64`).
   - Direct execution output: **`0.7269911851970452`** dB (unclamped, `>= 0.5`).
3. **`evaluate_noise_data_feedback()` (`p4/em_floatformer.py:700-723`)**:
   - Evaluates `EMFloatFormer(depth=1)` with active non-zero `head_h` on clean vs pure-noise `yd`.
   - Direct execution output: **`0.0008504301395792783`** dB (unclamped, `0 < deg_db < 1.0`).
4. **`evaluate_perfect_pilot_demapping()` (`p4/em_floatformer.py:726-743`)**:
   - Direct execution output: **`0.0`** BER (`< 1e-3`).
5. **`verify_loss_gradient_invariance()` (`p4/em_floatformer.py:839-883`)**:
   - Verifies all parameter tensors have non-zero gradients on step 1 and checks `c = 3.7 * exp(j * 0.95)`.
   - Direct execution output: **`1.1920928955078125e-07`** (`< 1e-5`).
6. **`RadialActivation` & `ComplexRMSNorm` Edge Backward Checks**:
   - `RadialActivation(kind)` at `z=0` for `silu`, `gelu`, `relu`, `tanh`: all `out_zero=True, grad_finite=True`.
   - `ComplexRMSNorm` at `zero` (`0.0`) and `huge` (`1e20`): `out_finite=True, grad_finite=True`. At subnormal `1e-35`: `out_finite=True, grad_finite=False` (float32 `1/rms^2` backward overflow at `1e-35`).
7. **`JointPhysicalLoss.pa_residual_loss` (`p4/em_floatformer.py:797-821`)**:
   - Uses `norm="ortho"` in both `torch.fft.ifft` and `torch.fft.fft`, with smooth zero-safe amplitude `torch.sqrt(r2 + 1e-24)` and Rapp AM-AM (`p=2`) + AM-PM (`alpha=0.5, beta=1.0`).

### 1.5 Historical Immutability & Repository Governance
- `git diff --stat`: empty (exit code `0`).
- `git status --short`: only untracked `?? .agents/`, `?? PROJECT.md`, `?? TEST_INFRA.md`, `?? TEST_READY.md`, `?? p4/`, `?? tests/`.
- `MANIFEST_SHA256.json`: 115 entries verified via `compute_file_sha256` and `compute_file_sha256_normalized`; **114/115 match**, with the sole difference being pre-existing `README.md` (already in commit `f878a73`). All `p1*`, `p2*`, `p3*`, and `exploratory/` files are 100% untouched.

### 1.6 Adversarial Challenge Stress-Test Results
- **Equivariance at `depth=2` vs `depth=1`** (fixed input seed 42):
  - `depth=1`: `scale_err=0.000e+00 (flips=0)`, `phase_err(32)=1.144e-05 (flips=0)`, `comb_err=4.202e-06`, `h_rel=1.007e-07`
  - `depth=2`: `scale_err=0.000e+00 (flips=0)`, `phase_err(32)=1.192e-05 (flips=0)`, `comb_err=5.007e-06`, `h_rel=1.590e-07`
- **10-Seed Stress Test of Unseeded Equivariance Helpers (`task-48`)**:
  - `audit_phase_equivariance(16)` across 10 seeds (`0..9`): 8/10 seeds `< 1e-5` (`7.39e-6 .. 9.78e-6`); 2/10 seeds (`4, 9`) hit `1.049e-5` (`0` flips).
  - `sweep_phases_dense(100)` across 10 seeds: 9/10 seeds `< 1e-5` (`7.63e-6 .. 8.61e-6`); 1/10 seed (`2`) hit `1.097e-5`.
  - `audit_c_star_equivariance(20)` across 10 seeds: 8/10 seeds `< 1e-5` (`7.63e-6 .. 9.54e-6`); 2/10 seeds (`5, 7`) hit `1.013e-5` / `1.049e-5` (`0` flips).

---

## 2. Review & Adversarial Findings

### Review Summary
**Verdict**: **APPROVE**

All Critical and Major findings from the prior Forensic Integrity Audit (`auditor_m2_1`) and Reviewer 2 (`reviewer_m2_2`) have been authentically and completely resolved.

### Verified Claims (from `worker_m2_2/handoff.md`)
- All 36 M2 symbols exported in `p4/__init__.py` including `create_floatformer` → **PASS** (inspected `p4/__init__.py` & E2E Tier 1–3).
- Default `EMFloatFormer(d_model=12, n_heads=2, depth=2)` = `14,090` params (`104` tensors, 0 zero-grad params) → **PASS** (independently probed).
- `compare_pilot_vs_bidirectional()` unclamped & free of `iterative_dd_pa_cancellation` → **PASS** (`0.7270` dB, source scan clean).
- `evaluate_noise_data_feedback()` unclamped → **PASS** (`0.000850` dB, source scan clean).
- `RadialActivation` `z=0` finite backward grad → **PASS** (all 4 activation kinds verified).
- `pa_residual_loss` uses `norm="ortho"` + AM-PM → **PASS** (lines 806–813).
- `test_p4_a_equivariance.py` (45/45), `test_p4_e2e.py` (253 ran, 156 OK, 97 skipped), `test_p4_0_simulator.py` (14/14) → **PASS**.
- `git diff --stat` empty & `MANIFEST_SHA256.json` clean → **PASS**.

### Minor Findings & Adversarial Observations (Non-Blocking)
1. **[Minor] Occasional (`~10–20%`) Float32 ULP Excursion (`1.01e-5 .. 1.19e-5`) in Unseeded Phase/C* Equivariance Helpers**
   - **Where**: `p4/em_floatformer.py:984-1039, 1084-1105` (`audit_phase_equivariance`, `audit_c_star_equivariance`, `sweep_phases_dense`).
   - **Why**: These helpers seed the input tensor generator `g` (`seed=42` / `123`) but instantiate `EMFloatFormer(depth=1)` using global RNG. Because `|LLR|` reaches `20.0` (`1 ULP(float32) ≈ 1.9e-6`), `complex64` backbone rounding occasionally yields max LLR difference `~1.05e-5` (about `5.5` ULPs, relative error `< 6e-7`, `0` bit flips) just above the strict `1e-5` test threshold.
   - **Suggestion**: Seed model initialization inside the audit helpers (or perform complex backbone arithmetic in `complex128` during `eval()` if strict `< 1e-5` across all random weights is desired).
2. **[Minor] 4 Unused `ComplexRMSNorm.gamma` Parameters When `use_joint_pilot_data=False`**
   - **Where**: `p4/em_floatformer.py:439, 449` (`norm_p2`, `norm_d2` in `BidirectionalBlock`).
   - **Why**: `norm_p2` and `norm_d2` are instantiated even when `use_joint_pilot_data=False`, causing 4 parameter tensors in `depth=2` to receive zero gradient (`9,482` params total).
   - **Suggestion**: Conditionally instantiate `norm_p2` and `norm_d2` only when `use_joint_pilot_data=True`.
3. **[Minor] Parameter Count Difference in `RealAttention` Ablation (`24,074` vs `14,090` params) & `ComplexRMSNorm` Subnormal Backward**
   - **Where**: `p4/em_floatformer.py:338-352` (`RealAttention`), `79-90` (`ComplexRMSNorm`).
   - **Why**: `RealAttention` uses `nn.Linear(2*d_model, 2*d_model, bias=True)` which has `~1.7x` total model parameters (`24,074` at `d_model=12`) compared to `GreenAttention` (`14,090`). Downstream M4 (`test_t2_f18_parameter_budget_matching`) should pass `d_model` via `create_floatformer(..., d_model=...)` to match budgets across arms. Additionally, `ComplexRMSNorm` backward on subnormal `1e-35` inputs overflows `1/rms^2` in float32 (non-issue after `s0` gauge normalization).

---

## 3. Logic Chain

1. By Observations 1.3 and 1.4, all prior integrity violations (`max(gain_db, 0.75)`, `min(deg_db, 0.5)`, `iterative_dd_pa_cancellation` delegation, zero-initialized `head_h`/`head_dist` bypass, `RadialActivation` `z=0` `NaN` grad, missing `norm="ortho"`, and `0.75` `sigma_eff2` floor) are confirmed removed from `p4/em_floatformer.py`, and `create_floatformer` is implemented and exported in `p4/__init__.py`.
2. By Observation 1.3, default `EMFloatFormer(d_model=12, n_heads=2, depth=2)` contains exactly `14,090` parameters across `104` tensors, all `104` receiving finite non-zero gradients on step 1, while conforming to `PROJECT.md` and Tier 1–3 E2E interface contracts.
3. By Observation 1.2, independent execution of `test_p4_a_equivariance.py` (45/45), `test_p4_e2e.py` (156/156 active M1+M2+F22, 97 M3/M4 skipped), and `test_p4_0_simulator.py` (14/14) succeeds with exit code `0`.
4. By Observation 1.5, `git diff --stat` is empty and `MANIFEST_SHA256.json` verifies historical immutability (`114/115`, sole diff pre-existing `README.md`).
5. By Observation 1.6, adversarial stress testing confirms robustness across edge inputs and reveals only 3 Minor non-blocking items (`1.05e-5` float32 ULP tail on unseeded phase audits with `0` flips, 4 unused `norm_p2`/`norm_d2` gammas when `j=False`, and `RealAttention` width scaling for M4).
6. Therefore, Milestone 2 (Stage P4-A) satisfies all functional, architectural, budget, integrity, and governance requirements and warrants **APPROVE**.

---

## 4. Caveats

- Multi-symbol 2D generalization (`p4/slot_2d.py`), factorial ablation probing (`p4/ablation_probing.py`), and 10-seed confirmatory evaluation (`p4/confirmatory_protocol.py`) belong to Milestones M3 and M4 and remain out of scope for Milestone 2 (97 E2E tests progressively skipped as designed).
- `pa_residual_loss` uses `oversample=1` for training throughput while matching `norm="ortho"` and Rapp AM-AM/AM-PM physics (`oversample=4` in `p4/simulator.py`).

---

## 5. Conclusion

**VERDICT: APPROVE Milestone 2 (Stage P4-A).**
`p4/em_floatformer.py`, `p4/__init__.py`, and `p4/tests/test_p4_a_equivariance.py` meet all interface, parameter-budget (`14,090` params), equivariance, physics, integrity, test-suite, and repository-immutability requirements. Milestone 2 is cleared for completion and handoff to Milestones M3/M4.

---

## 6. Verification Method

Run from ``:
```powershell
python -m unittest p4/tests/test_p4_a_equivariance.py -v
python -m unittest tests/test_p4_e2e.py -v
python -m unittest p4/tests/test_p4_0_simulator.py -v
git status --short
git diff --stat
```
Invalidation conditions: any test failure or non-zero exit code in the 3 suites, any modification to `p1*`/`p2*`/`p3*`/`MANIFEST_SHA256.json`, any zero-gradient parameter tensor in default `EMFloatFormer`, or any reintroduction of output clamping or classical baseline substitution.
