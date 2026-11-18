**Verdict: REQUEST_CHANGES**

# Milestone 2 Independent Review & Adversarial Challenge Report (`reviewer_m2_r1_2`)

## Review Summary

**Verdict**: `REQUEST_CHANGES` (1 Major blocking reproducibility defect; zero integrity violations; all numerical hardening, $C^*$ equivariance, LLR calibration, and unit/E2E suites pass).

### Findings

#### [Major] Finding 1: `run_p4a_comparative_benchmarks` Instantiates Models Before `torch.manual_seed(seed)` — Saved Benchmark Results Are Not Reproducible for `EMFloatFormer`
- **What**: In `run_p4a_comparative_benchmarks` (`p4/em_floatformer.py:1661-1692`), `EMFloatFormer(...)`, `LocalEquivConv16QAM(...)`, and `TinyJoint16QAM(...)` are instantiated as arguments to `_train_comparative_model(...)` **before** `_train_comparative_model` calls `torch.manual_seed(seed)` (`p4/em_floatformer.py:1600`).
- **Where**: `p4\em_floatformer.py:1661-1692`, `p4/results/p4a_comparative_benchmark.json`, `p4/results/p4a_comparative_benchmark.csv`.
- **Why**:
  1. `EMFloatFormer`'s initial weights depend on whatever unseeded global PyTorch RNG state exists before `run_p4a_comparative_benchmarks(seed=2026)` is called (verified: two consecutive `EMFloatFormer(d_model=12, n_heads=2, depth=2, n_sc=32)` calls without `manual_seed` produce different weights).
  2. Re-running `run_p4a_comparative_benchmarks(output_dir=None)` (which defaults to `seed=2026`) in a fresh Python process with `OMP_NUM_THREADS=4` / `MKL_NUM_THREADS=4` reproduces `LocalEquivConv16QAM`, `TinyJoint16QAM`, `Bussgang-LMMSE`, and `DD-PA-Cancel` bit-for-bit across all rows, **but fails to reproduce `EMFloatFormer`**:
     - `train_final_scale_ratio['EMFloatFormer']`: saved `0.9769642790158589` vs repro `0.9784150123596191`.
     - `strong, A=1.25, snr_db=10.0, EMFloatFormer`: saved `bit_errors=6228, ber=0.1900634765625, nmse=0.08866406977176666, ece=0.034031724821034004` vs repro `bit_errors=5936, ber=0.18115234375, nmse=0.07188382744789124, ece=0.023113259798265062` (`max_ber_diff = 0.0089111328125`).
  3. Additionally, because `_train_comparative_model` consumes no global RNG during training (`generate_frame` uses `gen = torch.Generator().manual_seed(seed + 1000)`), `LocalEquivConv16QAM` accidentally inherits `seed` (`2026`) instead of `seed + 1` (`2027`), and `TinyJoint16QAM` inherits `seed + 1` (`2027`) instead of `seed + 2` (`2028`).
- **Suggestion**:
  1. In `run_p4a_comparative_benchmarks` (`p4/em_floatformer.py:1661-1692`), call `torch.manual_seed(seed)`, `torch.manual_seed(seed + 1)`, and `torch.manual_seed(seed + 2)` **before** instantiating each model (or pass a model factory callable into `_train_comparative_model` and invoke it right after `torch.manual_seed(seed)`).
  2. Re-run `run_p4a_comparative_benchmarks()` to regenerate `p4/results/p4a_comparative_benchmark.json` and `p4/results/p4a_comparative_benchmark.csv` so they match `run_p4a_comparative_benchmarks(seed=2026)` bit-for-bit.
  3. Add a fast reproducibility assertion in `p4/tests/test_p4_a_equivariance.py` (e.g., perturb global RNG between two small `run_p4a_comparative_benchmarks(..., train_steps=2, eval_frames=4, output_dir=None)` calls with `seed=2026` and assert identical `rows` and `train_final_scale_ratio`).

#### [Minor] Finding 2: Unseeded `EMFloatFormer(depth=1)` in `evaluate_noise_data_feedback` & `evaluate_perfect_pilot_demapping`
- **What**: Unlike `compare_pilot_vs_bidirectional` (`p4/em_floatformer.py:677`), `evaluate_noise_data_feedback` (`line 733`) and `evaluate_perfect_pilot_demapping` (`line 759`) instantiate `EMFloatFormer(depth=1)` before any `torch.manual_seed(...)`, making their diagnostics depend on ambient global RNG state.
- **Where**: `p4/em_floatformer.py:733`, `p4/em_floatformer.py:759`.
- **Suggestion**: Seed deterministically (`torch.manual_seed(42)` / `torch.manual_seed(99)`) or use an isolated generator before constructing the model.

---

## Challenge Summary (Adversarial Assessment)

**Overall risk assessment**: **MEDIUM** (for M2 comparative reproducibility; **LOW** for $C^*$ equivariance, numerical hardening, and calibration).

### Confirmed Strengths & Verified Claims
- **Zero `DEAD_ENDS.md` regressions** → verified via source inspection (`p4/em_floatformer.py`) → **PASS**:
  - `ComplexLinear(..., bias=False)` strictly enforces `self.bias = None` (`lines 174-176`).
  - `JointEmbedding` uses full-rank complex linear embeddings (`lines 419-420`, `570-573`).
  - `head_h` and `head_dist` use non-zero `0.01 / sqrt(2*d_model)` initialization (`lines 568, 575-578`).
  - `JointPhysicalLoss.pa_residual_loss` uses `norm="ortho"` on both `ifft` and `fft` (`lines 867, 874`).
  - `compare_pilot_vs_bidirectional` (`lines 671-724`) and `evaluate_noise_data_feedback` (`lines 727-750`) are unclamped and make zero delegations to `iterative_dd_pa_cancellation`.
  - No artificial `sigma_eff2 >= 0.75` floor (`line 649` uses `+ 2.5e-2`; `line 651` uses `20.0 * torch.tanh(llrs64 / 20.0)`).
- **Iteration 3 numerical hardening & LLR calibration (`M2.1` + `M2.2`)** → verified via `p4.tests.test_p4_a_equivariance` + standalone stress checks → **PASS**:
  - `compute_gauge_anchor` (`eps=1e-38`, `complex128` `amax` rescaling), `ComplexRMSNorm` (factored `z_scaled * (gamma64 / rms_safe)` in `complex128`), and `RadialActivation` (`complex128`, `m > 1e150` rescaling) are finite in forward & backward on `1e-35`, `0`, and `1e20`.
  - `EMFloatFormer` (`depth=2` fresh/trained and `depth=4`) satisfies `max_llr_err < 1e-5`, `flips == 0`, `h_rel < 1e-5` across $r \in [2^{-12}, 2^{12}]$, non-power-of-2 scales `(0.13, 3.7, 1023.5)`, and `r in {1e-10, 1e10}`; `ECE < 0.05` across `(clean, mild, strong) x (15, 20, 30)` dB and `max|LLR| > 15.0` at 30 dB `clean`.
- **`LocalEquivConv16QAM` & `TinyJoint16QAM` & `JointPhysicalLoss._resolve_pa_A`** → verified via code inspection & independent stress script → **PASS**:
  - `LocalEquivConv16QAM` (`14,198` params, `+0.77%` vs `EMFloatFormer` `14,090` params) is genuinely $C^*$-equivariant across 24 octaves (both untrained and trained), with non-zero finite step-1 gradients on all parameter tensors.
  - `TinyJoint16QAM` (`1,880` params) wires `head_dist` into `z_corr`, has non-zero finite step-1 gradients on all parameter tensors, and handles 1D `(32,)` inputs.
  - `JointPhysicalLoss._resolve_pa_A` handles `A` (`float`, `inf`, `"clean"`/`"mild"`/`"strong"`), `frame.A`, `frame.pa_condition`, and raises `ValueError` on unknown strings; mixed-PA training `scale_ratio` stabilizes near `1.0` (`0.977-0.985`).
- **Benchmark JSON/CSV internal consistency & physical plausibility** → verified via cross-check script → **PASS**:
  - `p4/results/p4a_comparative_benchmark.json` and `.csv` match row-for-row across all 75 rows (`3 PA x 5 SNR x 5 methods`), `total_bits = 32768`, BER decreases monotonically with SNR for every method/PA, `ECE < 0.05` for `EMFloatFormer`, `LocalEquivConv16QAM`, `Bussgang-LMMSE`, and `DD-PA-Cancel`.
  - Note: No integrity violation / data fabrication detected — 4 of 5 methods matched bit-exact on re-execution; only `EMFloatFormer` differed due to Finding 1's unseeded initialization bug.

---

## 1. Observation

1. **Test Suites Executed Independently (`OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`)**:
   - `python -m unittest p4.tests.test_p4_0_simulator -v`: **Ran 14 tests in 2.347s — OK** (exit code 0).
   - `python -m unittest p4.tests.test_p4_a_equivariance -v`: **Ran 50 tests in 58.241s — OK** (exit code 0).
   - `python -m unittest tests.test_p4_e2e -v`: **Ran 253 tests in 22.485s — OK (skipped=97)** (156 active M1/M2/F22 tests passed, 97 M3/M4 skipped, exit code 0).
2. **Governance & Manifest Verification**:
   - `git diff --name-only`: empty (exit code 0; 0 tracked historical P1–P3 or `exploratory/*` files modified).
   - `git status --short`: `?? .agents/`, `?? PROJECT.md`, `?? TEST_INFRA.md`, `?? TEST_READY.md`, `?? p4/`, `?? tests/`.
   - `MANIFEST_SHA256.json`: 115 entries, 0 missing; 114/115 match byte-exact under LF (`\r\n` -> `\n`) normalization; the sole difference (`README.md`) was committed prior to Stage P4 in commit `f878a73` and is explicitly noted in `tests/e2e/test_tier1_feature_coverage.py:1128-1130`.
3. **Adversarial Stress & Benchmark Reproducibility Commands**:
   - Stress script checking JSON/CSV parity (75 rows), BER monotonicity, `ECE < 0.05`, `LocalEquivConv16QAM` step-1 gradients & trained 24-octave equivariance, `TinyJoint16QAM` step-1 gradients & 1D input, `EMFloatFormer(depth=4)` tuple unpacking / 1D / zeros / `r in {1e-10, 1e10}`, and `JointPhysicalLoss._resolve_pa_A` edge cases: all 6 checks `[PASS]` (exit code 0).
   - Reproducibility script comparing `run_p4a_comparative_benchmarks(output_dir=None)` against `p4/results/p4a_comparative_benchmark.json`:
     - `LocalEquivConv16QAM`, `TinyJoint16QAM`, `Bussgang-LMMSE`, and `DD-PA-Cancel` matched bit-exact across rows (`5783`, `5905`, `5918`, `6127` bit errors at `strong 10dB`).
     - `EMFloatFormer` differed: `train_final_scale_ratio` `0.976964` (saved) vs `0.978415` (repro); `strong 10dB` `bit_errors=6228, ber=0.190063` (saved) vs `bit_errors=5936, ber=0.181152` (repro); `max_ber_diff = 0.0089111328125`.
   - Direct RNG check on `EMFloatFormer(d_model=12, n_heads=2, depth=2, n_sc=32)`: two consecutive instantiations without `torch.manual_seed` returned `Same weight without manual_seed? False`.

---

## 2. Logic Chain

1. Observation 1 & 2 confirm all existing Stage P4-0 (14/14), Stage P4-A (50/50), active E2E (156/156), and governance checks (`git diff`, `MANIFEST_SHA256.json`) pass cleanly without `DEAD_ENDS.md` regressions.
2. Observation 3 demonstrates that `EMFloatFormer`, `LocalEquivConv16QAM`, `TinyJoint16QAM`, and `JointPhysicalLoss` satisfy all $C^*$ equivariance, numerical hardening, calibration, and gradient-flow contracts on both unit and adversarial stress inputs.
3. However, Observation 3 also proves `run_p4a_comparative_benchmarks(seed=2026)` constructs `EMFloatFormer(...)` at `p4/em_floatformer.py:1662` before `_train_comparative_model` calls `torch.manual_seed(seed)` at `p4/em_floatformer.py:1600`. Because `EMFloatFormer.__init__` consumes global RNG (`Same weight without manual_seed? False`), `EMFloatFormer` benchmark results in `p4/results/p4a_comparative_benchmark.{json,csv}` depend on unseeded ambient process RNG and cannot be reproduced by re-running `run_p4a_comparative_benchmarks(seed=2026)` (`max_ber_diff = 0.00891`).
4. Therefore, Milestone 2 cannot be signed off until `run_p4a_comparative_benchmarks` seeds deterministically before constructing each model, regenerates `p4/results/p4a_comparative_benchmark.{json,csv}`, and covers benchmark reproducibility with a unit test.

---

## 3. Caveats

- All tests and benchmarks were executed on CPU with `torch.set_num_threads(4)` / `OMP_NUM_THREADS=4` / `MKL_NUM_THREADS=4`. GPU and mixed-precision (`FP16`/`BF16`) audits are deferred to Milestone 4 per `PROJECT.md`.
- Running `tests.test_p4_e2e` invokes `TestTier2BoundaryCorner.test_t2_f07_single_thread_concurrency`, which writes `p4/results/latency_cpu.csv` and `p4/results/timing.json` with `threads=1, batch_size=1, timed_runs=3` (a pre-existing behavior in the read-only E2E suite).

---

## 4. Conclusion

**Verdict: `REQUEST_CHANGES`.**

While Milestone 2's architecture, Iteration 3 numerical hardening, LLR calibration (`2.5e-2` floor / `20.0 * tanh`), comparative model implementations, and test suites are technically sound and free of integrity violations, **Finding 1 (Major)** blocks sign-off:

1. Seed `torch.manual_seed(seed)` / `seed + 1` / `seed + 2` **before** instantiating `EMFloatFormer`, `LocalEquivConv16QAM`, and `TinyJoint16QAM` in `run_p4a_comparative_benchmarks` (`p4/em_floatformer.py:1661-1692`).
2. Regenerate `p4/results/p4a_comparative_benchmark.json` and `p4/results/p4a_comparative_benchmark.csv` via `run_p4a_comparative_benchmarks()`.
3. Add a quick determinism regression test for `run_p4a_comparative_benchmarks` in `p4/tests/test_p4_a_equivariance.py`.

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
Reproducibility check (must exit 0 with `max_ber_diff == 0.0` and `max_nmse_diff < 1e-6` even after perturbing global RNG beforehand):
```powershell
python -c "
import json, torch
from pathlib import Path
from p4.em_floatformer import run_p4a_comparative_benchmarks
torch.randn(17)  # perturb global RNG before calling
saved = json.loads(Path('p4/results/p4a_comparative_benchmark.json').read_text(encoding='utf-8'))
repro = run_p4a_comparative_benchmarks(output_dir=None)
assert saved['train_final_scale_ratio'] == repro['train_final_scale_ratio']
assert max(abs(a['ber'] - b['ber']) for a, b in zip(saved['rows'], repro['rows'])) == 0.0
assert max(abs(a['nmse'] - b['nmse']) for a, b in zip(saved['rows'], repro['rows'])) < 1e-6
"
```
**Invalidation conditions**: Any failure in the three unittest suites above, any mismatch between `run_p4a_comparative_benchmarks(output_dir=None)` and `p4/results/p4a_comparative_benchmark.json` after RNG perturbation, or any tracked historical file appearing in `git diff --name-only`.
