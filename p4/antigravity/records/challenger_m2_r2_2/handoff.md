**Verdict: APPROVE**

# Milestone 2 Iteration 2 Empirical Adversarial Verification Handoff (`challenger_m2_r2_2`)

## 1. Observation

All empirical checks were executed directly under `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `torch.set_num_threads(4)` in `` without modifying any source, test, or result file.

### 1.1 Existing Verification Suites (Direct Execution)
1. `python -m unittest p4.tests.test_p4_0_simulator -v`: **Ran 14 tests in 2.284s — OK** (exit code `0`).
2. `python -m unittest p4.tests.test_gauge_stress_adversarial -v`: **Ran 5 tests in 23.118s — OK** (exit code `0`).
   - 206 phase angles: max logit diff `6.869435e-06`, `0` flips.
   - 50 scales: max logit diff `3.814697e-06`, `0` flips.
   - 50 `(r, phi)` pairs: max H rel err `1.781575e-07`, max LLR diff `5.245209e-06`, `0` flips.
   - Zero-input and extreme-scale `[1e-20, 1e10]` checks passed without NaN/Inf.
3. `python -m unittest p4.tests.test_p4_a_equivariance -v`: **Ran 53 tests in 94.211s — OK** (exit code `0`).
4. `python -m unittest tests.test_p4_e2e -v`: **Ran 253 tests in 30.911s — OK (skipped=97)** (all `156` active M1, M2, and F22 governance tests passed; `97` pending M3/M4 tests cleanly skipped; exit code `0`).

### 1.2 LLR Calibration, Soft-FEC Dynamic Range & `3.5e-2` / `20.0 * tanh` Verification
Independently trained `EMFloatFormer(depth=2)` for 80 steps under mixed-PA `JointPhysicalLoss(lambda_nmse=1.0, lambda_pa=0.25)` across `(clean, mild, strong)` for two seeds (`train_seed=11` and `train_seed=2026`) and evaluated `256` frames per cell across `(clean, mild, strong) x (15, 20, 30)` dB with 24-octave $C^*$ transforms `[(2^-12, 2.7183), (0.13, -1.7), (3.7, 0.95), (1023.5, 2.5356), (4096.0, -0.6076)]`:

- **`train_seed=11`**:
  - `clean  15dB`: `BER=0.06421`, `ECE=0.00384`, `max|LLR|=19.748`, `p50=5.84`, `p90=16.27`, `p99=19.18`, `sat(|LLR|>=19.99)=0.0000`, `eq_llr=3.34e-06`, `eq_h=1.43e-07`, `flips=0`
  - `clean  20dB`: `BER=0.02658`, `ECE=0.00305`, `max|LLR|=19.856`, `p50=8.53`, `p90=18.83`, `p99=19.61`, `sat=0.0000`, `eq_llr=3.81e-06`, `eq_h=1.06e-07`, `flips=0`
  - `clean  30dB`: `BER=0.00760`, `ECE=0.00397`, `max|LLR|=19.946`, `p50=10.42`, `p90=19.57`, `p99=19.76`, `sat=0.0000`, `eq_llr=4.29e-06`, `eq_h=9.67e-08`, `flips=0`
  - `mild   15dB`: `BER=0.07410`, `ECE=0.00522`, `max|LLR|=19.826`, `p50=5.54`, `p90=15.85`, `p99=19.17`, `sat=0.0000`, `eq_llr=3.81e-06`, `eq_h=1.57e-07`, `flips=0`
  - `mild   20dB`: `BER=0.03131`, `ECE=0.00332`, `max|LLR|=19.925`, `p50=8.32`, `p90=18.69`, `p99=19.65`, `sat=0.0000`, `eq_llr=3.81e-06`, `eq_h=1.62e-07`, `flips=0`
  - `mild   30dB`: `BER=0.00885`, `ECE=0.00349`, `max|LLR|=19.972`, `p50=10.59`, `p90=19.55`, `p99=19.83`, `sat=0.0000`, `eq_llr=4.29e-06`, `eq_h=9.90e-08`, `flips=0`
  - `strong 15dB`: `BER=0.10703`, `ECE=0.01779`, `max|LLR|=19.850`, `p50=4.52`, `p90=14.12`, `p99=18.94`, `sat=0.0000`, `eq_llr=2.86e-06`, `eq_h=1.10e-07`, `flips=0`
  - `strong 20dB`: `BER=0.05328`, `ECE=0.00993`, `max|LLR|=19.955`, `p50=7.56`, `p90=18.10`, `p99=19.67`, `sat=0.0000`, `eq_llr=3.34e-06`, `eq_h=1.40e-07`, `flips=0`
  - `strong 30dB`: `BER=0.01877`, `ECE=0.00670`, `max|LLR|=19.989`, `p50=10.75`, `p90=19.51`, `p99=19.90`, `sat=0.0000`, `eq_llr=4.77e-06`, `eq_h=1.03e-07`, `flips=0`
- **`train_seed=2026`**:
  - Across all 9 cells: `ECE` ranges `0.00414` (`clean 20dB`) to `0.02387` (`strong 15dB`), all `< 0.05`; `max|LLR|` at 30 dB is `19.923` (`clean`), `19.966` (`mild`), `19.990` (`strong`), all `> 15.0` and `<= 20.0`; `sat(|LLR|>=19.99)=0.0000` across all 9 cells; `eq_llr <= 5.01e-06 < 1e-5`, `eq_h <= 1.67e-07 < 1e-5`, `flips=0`.
- **Exact `challenger_m2_3` 50-step `seed=11` `30 dB` `strong` PA `256`-frame diagnostic**:
  - `complex64`: `max_llr_err = 7.629e-06 < 1e-5`, `max_h_rel = 9.853e-08 < 1e-5`, `flips = 0`.
  - `complex128`: `max_llr_err = 1.315e-13 < 1e-11`, `max_h_rel = 4.573e-16 < 1e-11`, `flips = 0`.
- **Extreme numerical stress (`1e-35` and `1e+20`) on `EMFloatFormer(depth=2)`**:
  - Both forward outputs (`llrs`, `h_hat`) and backward gradients (both `yp.grad`/`yd.grad` and all model parameters) are strictly finite with zero NaN/Inf.

### 1.3 Joint Physical Loss, Bidirectional Gradients & Scale Stability
- **Mutual bidirectional feedback**:
  - `BidirectionalBlock`: `|d(d_out)/d(p_in)| = 7.8624e+02`, `|d(p_out)/d(d_in)| = 3.6027e+03`.
  - Full `EMFloatFormer(depth=2)`: perturbing `yd` shifts `h_hat` (`max|dh_hat/dyd| = 5.2066e-03`); perturbing `yp` shifts `dist_est` (`max|ddist/dyp| = 6.6526e-03`) and `llrs` (`max|dllr/dyp| = 3.0907e+01`).
- **Zero `gamma_eff`/head gradient deadlocks (Step 1 under `JointPhysicalLoss`)**:
  - `depth=1` (`58` param tensors): `missing/zero = []`, `min_grad = 4.837e-04`, `gamma_eff = 4.892e-02 / 2.206e-01`, `head_h = 1.195e+00`, `head_dist = 1.248e+00`.
  - `depth=2` (`104` param tensors): `missing/zero = []`, `min_grad = 2.286e-04`, `gamma_eff = 2.514e-02 / 5.016e-02`, `head_h = 4.557e+00`, `head_dist = 8.754e-01`.
  - `depth=4` (`196` param tensors): `missing/zero = []`, `min_grad = 7.016e-04`, `gamma_eff = 1.138e-03 / 1.096e-01`, `head_h = 4.795e+00`, `head_dist = 5.219e+00`.
- **Mixed-PA `A`-resolution**: Verified `_resolve_pa_A` on `"clean"` (`None`), `"mild"` (`2.25`), `"strong"` (`1.25`), `" Mild "` (`2.25`), `float("inf")` (`None`), plain `FrameData` fallback (`1.25`), `frame.A = float("inf")` (`None`), `frame.pa_condition = "mild"` (`2.25`), and `ValueError` on invalid strings (`"invalid_pa"`, `"linear"` via `A=` or `frame.pa_condition`).
- **Parseval `norm="ortho"` parity**: `pa_residual_loss` frequency-vs-time Parseval energy max diff = `3.815e-06`; exact forward-model `yd_exact` residual = `0.000e+00`; `A=float("inf")` vs `A=None` diff = `0.000e+00`.
- **Mixed-PA `scale_ratio` stability**: 60-step mixed-PA training tail mean `0.971604` (min `0.829587`, max `1.077226`); 240-step benchmark `train_final_scale_ratio` = `0.9793996016184489` (`EMFloatFormer`), `0.9988971432050069` (`LocalEquivConv16QAM`), `0.9782894730567933` (`TinyJoint16QAM`).

### 1.4 Comparative Benchmarks, Exact Reproducibility & Diagnostic Helpers
- **RNG-perturbed exact reproducibility**: After calling `torch.randn(19)`, `run_p4a_comparative_benchmarks(output_dir=None)` reproduced `p4/results/p4a_comparative_benchmark.json` and `p4/results/p4a_comparative_benchmark.csv` with **100% exact equality** across `param_counts`, `train_final_scale_ratio`, and all `75` rows (`3 PA x 5 SNR x 5 methods`).
- **Baseline models**:
  - `LocalEquivConv16QAM`: `14,198` params (`0.7665%` difference vs `EMFloatFormer` `14,090` params, `< 1%`); 1D `(32,)` input supported; `step1_bad_grads = []`; 24-octave $C^*$ equivariance `max_llr_err = 3.338e-06 < 1e-5`, `max_h_rel = 1.351e-07 < 1e-5`, `flips = 0`.
  - `TinyJoint16QAM`: `1,880` params; 1D `(32,)` input supported; finite `h_hat`/`dist_est`; `step1_bad_grads = []`; confirmed non-equivariant (`max_llr_diff = 32.286`, `177` bit flips under `c = 4096 * exp(j*1.23)`).
- **All 20 diagnostic/audit helpers determinism**: Executed every one of the 20 helpers twice under distinct RNG perturbations (`torch.randn(13)` vs `torch.randn(97)`), obtaining bit-identical outputs on all 20/20:
  1. `audit_single_scale(2**-12)`: `(0.0, 0)`
  2. `audit_single_phase(pi)`: `0`
  3. `compare_pilot_vs_bidirectional()`: `0.5939365174016732`
  4. `evaluate_noise_data_feedback()`: `0.0014482116725634663`
  5. `evaluate_perfect_pilot_demapping()`: `0.000244140625`
  6. `verify_loss_gradient_invariance()`: `5.960464477539063e-08`
  7. `audit_loss_scale_invariance()`: `0.0`
  8. `train_step_joint_loss()`: `True`
  9. `check_loss_gradient_scale()`: `5.960464477539063e-08`
  10. `audit_gain_equivariance()`: `(0.0, 0)`
  11. `audit_phase_equivariance()`: `(5.245208740234375e-06, 0)`
  12. `audit_c_star_equivariance()`: `(7.62939453125e-06, 0)`
  13. `audit_channel_equivariance()`: `1.0825020259021115e-07`
  14. `run_standalone_equivariance_check()`: `True`
  15. `sweep_phases_dense()`: `5.7220458984375e-06`
  16. `check_irrational_phases()`: `0`
  17. `check_noise_quad_scaling()`: `1.9073486328125e-06`
  18. `verify_gauge_attention_joint()`: `True`
  19. `audit_full_equivariance()`: `(3.814697265625e-06, 0)`
  20. `verify_bidirectional_green_prop()`: `True`

### 1.5 Preservation & Governance
- `git diff --name-only`: **empty** (`0` tracked files modified relative to `HEAD` `f878a73`).
- Stage P4-0 source/test files (`p4/simulator.py`, `p4/classical_baselines.py`, `p4/benchmark_latency.py`) retain their original `2026-11-15` timestamps.
- `MANIFEST_SHA256.json` (`115` P1/P2 historical files): `58` raw-byte diffs are purely Windows CRLF (`\r\n` vs `\n`), and `1` file (`README.md`) was already updated at commit `f878a73` prior to Stage P4; `114/115` files match SHA256 after LF normalization, and E2E F22 governance checks pass.

---

## 2. Logic Chain
1. Independent execution of all 4 unittest suites (`test_p4_0_simulator` 14/14, `test_gauge_stress_adversarial` 5/5, `test_p4_a_equivariance` 53/53, `test_p4_e2e` 156 active / 97 skipped) establishes baseline test integrity with zero failures or errors.
2. Independent multi-seed (`11`, `2026`) mixed-PA training and evaluation across `(clean, mild, strong) x (15, 20, 30)` dB confirms that the `3.5e-2` variance floor + `20.0 * tanh` bound keeps `ECE < 0.05` (max `0.02387`), preserves high-SNR LLR dynamic range (`max|LLR| in [19.923, 19.990] > 15.0` at 30 dB) with `0.0000` saturation collapse, and satisfies `< 1e-5` $C^*$ equivariance (`0` flips) in `complex64` as well as `< 1e-11` in `complex128` on `challenger_m2_3`'s exact 50-step `seed=11` diagnostic.
3. Bidirectional perturbation/gradient checks across depths 1/2/4, mixed-PA `_resolve_pa_A` resolution tests, Parseval `norm="ortho"` parity checks, and training `scale_ratio` measurements demonstrate zero gradient deadlocks and consistent physical loss scaling near `1.0`.
4. Full RNG-perturbed replay (`torch.randn(19)`) of `run_p4a_comparative_benchmarks(output_dir=None)` matches the saved JSON and CSV artifacts across all 75 rows and `train_final_scale_ratio`, all 20 diagnostic helpers are bit-exact deterministic under RNG perturbation, `LocalEquivConv16QAM` (`14,198` params, `0.7665%` from `EMFloatFormer`'s `14,090`) and `TinyJoint16QAM` (`1,880` params) satisfy their architectural and gradient contracts, and `git diff --name-only` confirms zero modifications to tracked historical files.

---

## 3. Caveats
- **Runtime mutation of Stage P4-0 latency files by `tests.test_p4_e2e`**: Executing `tests.test_p4_e2e` updates `p4/results/latency_cpu.csv` and `p4/results/timing.json` on disk (SHA256 changed from `b37c817b...`/`47f6dfaf...` before E2E to `822246ef...`/`265891bc...` after E2E) because the E2E F7 tests run the latency benchmark writing into `p4/results/`. `p4a_comparative_benchmark.json` (`8c16c95e...`) and `p4a_comparative_benchmark.csv` (`64b100f6...`) remained byte-identical.
- **Comparative BER ranking on 1D dense-pilot 32-subcarrier symbols (`equal4`, 240 steps)**: In `p4a_comparative_benchmark.json`, `EMFloatFormer` strictly outperforms `TinyJoint16QAM` (`0.01447` vs `0.02380` at 30 dB `strong`; `ECE <= 0.02921` vs `0.07889`) and achieves exact 24-octave $C^*$ equivariance, while `LocalEquivConv16QAM` (`0.01181` at 30 dB `strong`) and classical `DD-PA-Cancel` (`0.00674`) achieve lower BER on this 1D dense-pilot setup. This is an honest, un-hacked empirical comparison under frozen comparative hyperparameters; the Stage P4-B 2D sparse-DMRS Doppler/ICI regime is deferred to Milestone 3.
- **CPU-only verification**: All checks ran on CPU with `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, `torch.set_num_threads(4)`.

---

## 4. Conclusion
**Verdict: APPROVE.** All Milestone 2 (Stage P4-A: Features F8–F12) Iteration 2 empirical verification criteria pass cleanly: calibrated LLR dynamic range (`ECE < 0.025`, `max|LLR| > 19.9` at 30 dB, zero saturation collapse), strict `< 1e-5` (`complex64`) and `< 1e-11` (`complex128`) $C^*$ equivariance, zero gradient deadlocks and bidirectional pilot-data flow, mixed-PA `A`-resolution and Parseval parity, 100% exact RNG-perturbed reproducibility of all 75 comparative benchmark rows and all 20 diagnostic helpers, and clean repository governance (`git diff --name-only` empty).

---

## 5. Verification Method
Run from ``:
```powershell
$env:OMP_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'
python -m unittest p4.tests.test_p4_0_simulator -v
python -m unittest p4.tests.test_gauge_stress_adversarial -v
python -m unittest p4.tests.test_p4_a_equivariance -v
python -m unittest tests.test_p4_e2e -v
git diff --name-only
```
And run the RNG-perturbed reproducibility check:
```powershell
python -c "import json, torch; from pathlib import Path; from p4.em_floatformer import run_p4a_comparative_benchmarks; torch.set_num_threads(4); torch.randn(19); s=json.loads(Path('p4/results/p4a_comparative_benchmark.json').read_text(encoding='utf-8')); r=run_p4a_comparative_benchmarks(output_dir=None); assert s['train_final_scale_ratio']==r['train_final_scale_ratio'] and s['rows']==r['rows']; print('EXACT_REPRO_OK')"
```
**Invalidation conditions**: Any failure/error in the 4 unittest suites, any `ECE >= 0.05`, `max|LLR| <= 15.0` at 30 dB, `max|LLR_err| >= 1e-5` (`complex64`) / `>= 1e-11` (`complex128`), any non-deterministic helper among the 20 helpers, any mismatch in the 75 comparative benchmark rows under RNG perturbation, or any tracked historical file appearing in `git diff --name-only`.
