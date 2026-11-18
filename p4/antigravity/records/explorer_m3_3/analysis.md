# M3 Explorer 3 Analysis: Historical Conventions, Repository Governance, and Baseline Verification

## 1. Historical Inventory & Repository Governance (`MANIFEST_SHA256.json` & F22)

### 1.1 Historical Source & Exploratory Files (Read-Only / Immutable)
- **`p1*.py` (8 files):** `p1b_representation.py`, `p1c_architecture.py`, `p1c_make_report.py`, `p1c_verify_checkpoints.py`, `p1d_make_report.py`, `p1d_robustness.py`, `p1e_make_report.py`, `p1e_replication.py`.
- **`p2*.py` (8 files):** `p2_finalize.py`, `p2_models.py`, `p2_numeric.py`, `p2_package.py`, `p2_run.py`, `p2_supplement.py`, `p2_supplement_finalize.py`, `p2_symmetry.py`.
- **`p3*.py` (15 files):** `p3_a_mechanism.py`, `p3_a_mechanism_audit.py`, `p3_a_mechanism_make_report.py`, `p3_models.py`, `p3b_independent_audit.py`, `p3b_make_report.py`, `p3b_package.py`, `p3b_privileged_reference.py`, `p3b_receivers.py`, `p3b_time_link.py`, `p3b_timing_extended.py`, `p3b_train_eval.py`, `p3c_independent_audit.py`, `p3c_make_report.py`, `p3c_precision.py`.
- **Root baseline & historical test files (8 files):** `ofdm_baseline.py`, `test_p1d_invariants.py`, `test_p1e_integrity.py`, `test_p2_integrity.py`, `test_p2_supplement_independent.py`, `test_p3_a_mechanism.py`, `test_p3b_link.py`, `test_p3c_precision.py`.
- **`exploratory/` (33 files across 2 subdirectories):**
  - `exploratory/16qam_joint_pa_probe/` (14 files): `PROTOCOL.md`, `REPORT.md`, `probe.py`, `audit.py`, `calibration_sensitivity.py`, `timing.py`, `__pycache__/probe.cpython-312.pyc`, and `formal_results/{audit.json, ber_by_seed.csv, ber_summary.csv, equal_budget_calibration.csv, equal_budget_calibration_config.json, independent_audit.json, timing.json}`.
  - `exploratory/p3de_tdl_surrogate/` (19 files): `P3DE_SCOPE_ADDENDUM.md`, `P3DE_SOURCE_MANIFEST_SHA256.json`, `P3DE_SURROGATE_PROTOCOL.md`, `P3DE_SURROGATE_REPORT.md`, `P3DE_source_and_results_2026-11-15.zip`, `README_P3DE.md`, `p3de_experiment.py`, `p3de_independent_audit.py`, `p3de_make_report.py`, `p3de_profile_bridge.py`, `requirements_p3de_surrogate.txt`, and `p3de_results/{ber_by_seed.csv, ber_summary.csv, checkpoint_sha256.json, independent_audit.json, latency_cpu.csv, primary_endpoints.json, standalone_audit.json, training_registry.csv}`.

### 1.2 `MANIFEST_SHA256.json` Structure
- **Format:** JSON list of **115** entries, each with schema `{"file": str, "bytes": int, "sha256": str}`.
- **Distribution by top-level path:** `<root>` (34), `p1b_results` (12), `p1c_results` (10), `p1d_results` (7), `p1e_results` (9), `p1e_train` (6), `p2a_results` (6), `p2b_results` (7), `p2c_results` (7), `p2d_results` (4), `p2e_results` (10), `results` (3).

### 1.3 F22 Repository Cleanliness & Isolation Rules
Verified in `tests/e2e/test_tier1_feature_coverage.py:1114-1174` and `tests/e2e/test_tier2_boundary_corner.py:991-1031`:
1. **SHA256 Manifest Verification (`test_t1_f22_manifest_sha256_verification`):** Every file in `MANIFEST_SHA256.json` must exist and match either raw SHA256 or `\r\n`->`\n` normalized SHA256 (with `README.md` explicitly exempted).
2. **Root `.py` Ban (`test_t1_f22_p4_isolation_boundary`):** Every `.py` file directly in `PROJECT_ROOT` must match `startswith("p1" | "p2" | "p3" | "test_p")` or `== "ofdm_baseline.py"`. Any new `.py` file in root fails F22.
3. **Root Directory Allowlist (`test_t2_f22_new_file_outside_p4_detection`):** Every directory directly in `PROJECT_ROOT` must be in `{"p4", "tests", ".agents", "exploratory", "figures", ".git", "results", "__pycache__", "p2_supplement", "p1e_train", ".pytest_cache"}` or end with `"_results"`. Creating any other top-level directory (e.g., `checkpoints/`, `tmp/`, `logs/`) fails F22.
4. **Results Isolation (`test_t2_f22_results_directory_isolation`):** `p4/results/` must exist; all M3 deliverables (`p4/slot_2d.py`, `p4/tests/test_p4_b_2d_fec.py`, `p4/results/p4b_coded_bler.{json,csv}`) must stay strictly inside `p4/` and `tests/`.

---

## 2. Historical Domain Conventions Built Upon by Stage P4

| Domain Axis | Historical Convention (P1–P3 & Exploratory) | Stage P4 Inheritance / M3 Relevance |
|---|---|---|
| **Subcarriers & CP** | `N = 32`, `CP = 16` (`p3b_time_link.py:12-13`, `exploratory/16qam_joint_pa_probe/probe.py:14`) | `p4/simulator.py:20-22` uses `DEFAULT_N = 32`, `DEFAULT_CP = 16`, `DEFAULT_OVERSAMPLE = 4`; M3 extends to `N_symb=14, N_sc in {32, 64}`. |
| **Channel Profiles** | `train3: (0.72, 0.20, 0.08)`, `equal4: (0.25,)*4`, `long6: (0.40, 0.22, 0.15, 0.10, 0.08, 0.05)`, `long12: (0.35, 0.15, 0.10, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.025, 0.025, 0.02)`, `sparse12: (0.52, 0,0,0, 0.30, 0,0,0,0,0,0, 0.18)` (`p3b_time_link.py:15-21`). `p3de_profile_bridge.py` adds TDL-A/C/D at `DELTA_F=30000.0`. | `p4/simulator.py:30-36` preserves exact same 5 `PROFILES` (`train3`, `equal4`, `long6`, `long12`, `sparse12`). M3 evaluates OOD delay spread including `sparse12` (`test_t1_f17_ood_delay_spread_bler_resilience`, `test_t3_p19`). |
| **SNR Grids** | Training uniformly draws `snr_db in [0, 20]` (`ofdm_baseline.py:193`, `p2_run.py:36`, `p3b_train_eval.py:75`). Evaluation grids: `[0, 5, 10, 15, 20, 25]` (`ofdm_baseline.py:211`), `(0, 5, 10, 15, 20)` (`p1c_architecture.py:33`), `(0, 10, 20)` (`p1e_replication.py:38`, `p2_run.py:17`, `p3b_train_eval.py:22`, `p3c_precision.py:31`). `noise_var = 10 ** (-snr_db / 10)`. | Stage P4-A uses `[10, 15, 20, 25, 30]` (`p4a_comparative_benchmark.csv`); E2E M3 tests use `[0, 5, 10, 15, 20]` (`test_t3_p18`) and `[5.0, 10.0, 15.0]` (`test_t1_f17`). |
| **16-QAM Gray Map & Rapp PA** | `SIGMA = sqrt(10)`, `THRESH = 2 / sqrt(10)`, Gray table `[-3, -1, 3, 1] / sqrt(10)` (`b0`=sign I, `b1`=inner/outer I, `b2`=sign Q, `b3`=inner/outer Q), `hard_bits`: `(z.real>0, z.real.abs()<THRESH, z.imag>0, z.imag.abs()<THRESH)`. Rapp PA `A=2.25` (mild), `A=1.25` (strong), `knee=2` (`probe.py:19-38`). | Identical Gray mapping & PA `A in {1.25, 2.25}` in `p4/simulator.py:25-66, 89-129` and `tests/e2e/harness.py:75-115`. |
| **Pilot Patterns** | P1/P2/P3A/16qam-probe: full-tone `N=32` QPSK pilot symbol (`Xp` known). `p3b_time_link.py:14, 68` adds `ANCHORS = (0, 8)` in payload for CPE phase tracking (`PAYLOAD_MASK` = 30 tones). | M3 transitions to 2D sparse DMRS: `N_symb=14`, comb-2 at symbols 2 & 11 (`7.14%` overhead <= `10%`), pseudo-random unit-energy QPSK (`test_t1_f13_*`). |
| **RNG Seeding** | Explicit `torch.Generator(device='cpu').manual_seed(...)` with disjoint numeric namespaces across train vs test vs impairment (`p1e_replication.py:66`, `p2_run.py:33,81`, `p3b_train_eval.py:54-60,69-70`, `p3c_precision.py:81-86`). | `tests/e2e/harness.py:122-130` locks `HISTORICAL_SEEDS = [42, 100, 2024, 777, 999, 12345, 54321, 99999]` and `FROZEN_SEEDS_P4D = [74001..74010]`. Use dedicated disjoint `torch.Generator` seeds for M3 training/evaluation. |
| **Metrics & Accounting** | Exact integer `errors` and `bits` (or `bit_errors`, `total_bits`, `ber = errors/bits`), `symbol_errors`, `channel_mse` / `nmse`, `bce`, `ece`, `scale_ratio` (`p1e_replication.py:97`, `p2_run.py:104`, `p3b_train_eval.py:155`, `probe.py:200`, `p4a_comparative_benchmark.csv`). | M3 adds exact Python `int` `block_errors` / `total_blocks` (`test_t1_f17_integer_block_counters`, `test_t2_f17_massive_blocks_accounting_overflow_check`) alongside `ber`, `bler`, `nmse`. |
| **LLR / Logit Polarity** | Positive logit = bit `1`, negative = bit `0`; hard decision `(logits > 0).long()`, loss `F.binary_cross_entropy_with_logits(logits, bits.float())` (`ofdm_baseline.py:148,196`, `p3b_receivers.py:51`, `p4/em_floatformer.py:652-653`). | Preserved in Stage P4 (`3.5e-2` variance floor, `20.0 * tanh(llrs64 / 20.0)`). Outer FEC Min-Sum decoder must match this polarity (`llrs > 0 => bit 1`, verified by `test_t1_f15_min_sum_soft_ldpc_decoding`: `llrs = (2*c - 1) * 10`). |
| **CPU Thread Controls** | Every main/audit/benchmark script explicitly sets `torch.set_num_threads(4)` (or `min(4, os.cpu_count() or 1)`) (`ofdm_baseline.py:180`, `p1e_replication.py:230`, `p2_run.py:186`, `p3b_train_eval.py:239`, `p3de_experiment.py:215`). | `TEST_INFRA.md:109` & `tests/e2e/test_tier1_feature_coverage.py:50` enforce `torch.set_num_threads(4)`. |

---

## 3. Workspace Status, `p4/results/`, and Environment

### 3.1 `p4/results/` Contents
- `latency_cpu.csv` (6 rows for methods `LS`, `DFT-8`, `Bussgang-LMMSE`, `DD-PA-Iter1`, `DD-PA-Iter2`, `DD-PA-Iter3`)
- `timing.json` (structured latency metadata; **note:** running `tests.test_p4_e2e` invokes `run_benchmarks(..., output_dir="p4/results")` in `test_t1_f07_benchmark_script_invocation` and `test_t1_f07_latency_hierarchy_ordering`, updating these two files in place)
- `p4a_comparative_benchmark.csv` (75 data rows: 3 PA conditions `strong/mild/clean` x 5 SNRs `10,15,20,25,30` x 5 methods `EMFloatFormer, LocalEquivConv16QAM, TinyJoint16QAM, Bussgang-LMMSE, DD-PA-Cancel`)
- `p4a_comparative_benchmark.json` (full Stage P4-A benchmark metadata & rows, `profile="equal4"`, `n_sc=32`, `train_steps=240`, `batch_size=32`, `eval_frames=256`, `seed=2026`, `num_threads=4`)

### 3.2 `git status` & Historical Immutability
- `git status -s` shows **zero modified tracked files** (`git diff --stat` is empty). Only untracked Stage P4 / workflow paths appear: `?? .agents/`, `?? PROJECT.md`, `?? TEST_INFRA.md`, `?? TEST_READY.md`, `?? p4/`, `?? tests/`.
- All `p1*.py`, `p2*.py`, `p3*.py`, `exploratory/*`, and `MANIFEST_SHA256.json` are untouched (confirmed both by `git status`/`git diff` and by `test_t1_f22_manifest_sha256_verification` passing).

### 3.3 Python, PyTorch, and CPU Runtime Environment
- **Python:** `3.12.7` (Anaconda, MSC v.1929 64-bit AMD64, Windows)
- **PyTorch:** `2.12.1+cpu` (`torch.cuda.is_available() == False`)
- **CPU Cores & Threads:** `os.cpu_count() == 12`, default `torch.get_num_threads() == 10`, `torch.get_num_interop_threads() == 10`; `torch.set_num_threads(4)` cleanly sets intra-op threads to `4`.

---

## 4. Baseline Test Suite Verification (Pre-M3)

| Suite | Command | Result | Details |
|---|---|---|---|
| **M1 Unit Tests** | `python -m unittest p4.tests.test_p4_0_simulator -v` | **PASS** (exit 0) | `Ran 14 tests in 2.174s`, `OK` |
| **M2 Unit Tests** | `python -m unittest p4.tests.test_p4_a_equivariance -v` | **PASS** (exit 0) | `Ran 53 tests in 66.516s`, `OK` |
| **Stage P4 E2E Suite** | `python -m unittest tests.test_p4_e2e -v` | **PASS** (exit 0) | `Ran 253 tests in 23.030s`, `OK (skipped=97)` -> **156 passed**, **97 skipped** (pending M3 `p4.slot_2d` & M4 `p4.ablation_probing` / `p4.confirmatory_protocol`), 0 failures, 0 errors |
