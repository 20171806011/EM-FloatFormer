# Handoff Report — `explorer_m3_3` (Milestone 3 / Stage P4-B Pre-Flight)

**Handoff Type:** Hard
**Scope:** Historical reference catalog, `MANIFEST_SHA256.json` & F22 governance rules, historical domain conventions, `p4/results/` & `git status`, baseline M1/M2/E2E pass counts, and Python/PyTorch/CPU runtime check.
**Detailed Analysis:** `explorer_m3_3\analysis.md`

---

## 1. Observation

1. **Historical & Exploratory Files Verified on Disk (Read-Only):**
   - `p1*.py` (8): `p1b_representation.py`, `p1c_architecture.py`, `p1c_make_report.py`, `p1c_verify_checkpoints.py`, `p1d_make_report.py`, `p1d_robustness.py`, `p1e_make_report.py`, `p1e_replication.py`.
   - `p2*.py` (8): `p2_finalize.py`, `p2_models.py`, `p2_numeric.py`, `p2_package.py`, `p2_run.py`, `p2_supplement.py`, `p2_supplement_finalize.py`, `p2_symmetry.py`.
   - `p3*.py` (15): `p3_a_mechanism.py`, `p3_a_mechanism_audit.py`, `p3_a_mechanism_make_report.py`, `p3_models.py`, `p3b_independent_audit.py`, `p3b_make_report.py`, `p3b_package.py`, `p3b_privileged_reference.py`, `p3b_receivers.py`, `p3b_time_link.py`, `p3b_timing_extended.py`, `p3b_train_eval.py`, `p3c_independent_audit.py`, `p3c_make_report.py`, `p3c_precision.py`.
   - Related root files (8): `ofdm_baseline.py`, `test_p1d_invariants.py`, `test_p1e_integrity.py`, `test_p2_integrity.py`, `test_p2_supplement_independent.py`, `test_p3_a_mechanism.py`, `test_p3b_link.py`, `test_p3c_precision.py`.
   - `exploratory/` (33 files): `exploratory/16qam_joint_pa_probe/` (14 files) and `exploratory/p3de_tdl_surrogate/` (19 files).
2. **`MANIFEST_SHA256.json` & F22 Rules:**
   - `MANIFEST_SHA256.json` contains **115** JSON objects with keys `['file', 'bytes', 'sha256']` spanning `<root>` (34), `p1b_results` (12), `p1c_results` (10), `p1d_results` (7), `p1e_results` (9), `p1e_train` (6), `p2a_results` (6), `p2b_results` (7), `p2c_results` (7), `p2d_results` (4), `p2e_results` (10), `results` (3).
   - `tests/e2e/test_tier1_feature_coverage.py:1114-1174` checks SHA256 (raw or CRLF->LF normalized; `README.md` skipped at line 1129) and enforces (`test_t1_f22_p4_isolation_boundary`, lines 1151-1164) that every `.py` file in `PROJECT_ROOT` starts with `p1`, `p2`, `p3`, `test_p`, or equals `ofdm_baseline.py`.
   - `tests/e2e/test_tier2_boundary_corner.py:1004-1031` enforces (`test_t2_f22_new_file_outside_p4_detection`, lines 1010-1014) that every directory in `PROJECT_ROOT` is in `{"p4", "tests", ".agents", "exploratory", "figures", ".git", "results", "__pycache__", "p2_supplement", "p1e_train", ".pytest_cache"}` or ends with `"_results"`, and (`test_t2_f22_results_directory_isolation`, lines 1027-1030) that `p4/results` exists.
3. **Historical Domain Conventions:**
   - `N=32, CP=16` (`p3b_time_link.py:12-13`, `exploratory/16qam_joint_pa_probe/probe.py:14`, `p4/simulator.py:20-22`).
   - Standard 5 delay profiles (`p3b_time_link.py:15-21`, `p4/simulator.py:30-36`): `train3`, `equal4`, `long6`, `long12`, `sparse12`.
   - Training SNR uniform `[0, 20]` dB (`ofdm_baseline.py:193`, `p2_run.py:36`, `p3b_train_eval.py:75`); eval grids `[0, 5, 10, 15, 20, 25]` / `(0, 5, 10, 15, 20)` / `(0, 10, 20)`; Stage P4-A uses `[10, 15, 20, 25, 30]` (`p4/results/p4a_comparative_benchmark.csv`); E2E M3 uses `[0, 5, 10, 15, 20]` (`test_t3_p18`) and `[5.0, 10.0, 15.0]` (`test_t1_f17`).
   - 16-QAM Gray map (`SIGMA = sqrt(10)`, `THRESH = 2/sqrt(10)`, `[-3, -1, 3, 1] / sqrt(10)`) & Rapp PA (`A=2.25` mild, `A=1.25` strong, `knee=2`) (`probe.py:19-38`, `p4/simulator.py:25-129`).
   - LLR polarity: positive logit = bit `1`, `hard_bits = (llrs > 0).long()`, `F.binary_cross_entropy_with_logits` (`ofdm_baseline.py:148,196`, `p4/em_floatformer.py:652-653`, `test_t1_f15_min_sum_soft_ldpc_decoding`).
   - Disjoint explicit `torch.Generator` seeding (`p1e_replication.py:66`, `p2_run.py:33,81`, `p3b_train_eval.py:54-60`); `HISTORICAL_SEEDS` & `FROZEN_SEEDS_P4D` locked in `tests/e2e/harness.py:122-130`.
   - Integer error accounting (`errors`, `bits`, `ber = errors/bits`) and `torch.set_num_threads(4)` in all eval/benchmark runners (`p1e_replication.py:230`, `p2_run.py:186`, `p3b_train_eval.py:239`, `TEST_INFRA.md:109`).
4. **`p4/results/` & `git status`:**
   - `p4/results/` contains `latency_cpu.csv`, `timing.json`, `p4a_comparative_benchmark.csv` (75 data rows), and `p4a_comparative_benchmark.json`. (Note: running `tests.test_p4_e2e` invokes `run_benchmarks(..., output_dir="p4/results")` in `test_t1_f07_benchmark_script_invocation` & `test_t1_f07_latency_hierarchy_ordering`, refreshing `latency_cpu.csv` and `timing.json` in place.)
   - `git status -s` shows only untracked paths (`?? .agents/`, `?? PROJECT.md`, `?? TEST_INFRA.md`, `?? TEST_READY.md`, `?? p4/`, `?? tests/`); `git diff --stat` is empty.
5. **Baseline Test Results (Pre-M3):**
   - `python -m unittest p4.tests.test_p4_0_simulator -v` -> **Ran 14 tests in 2.174s — OK** (exit code 0)
   - `python -m unittest p4.tests.test_p4_a_equivariance -v` -> **Ran 53 tests in 66.516s — OK** (exit code 0)
   - `python -m unittest tests.test_p4_e2e -v` -> **Ran 253 tests in 23.030s — OK (skipped=97)** -> **156 passed, 97 skipped**, 0 failures, 0 errors (exit code 0)
6. **Runtime Environment:**
   - Python `3.12.7` (Anaconda, AMD64 Windows), PyTorch `2.12.1+cpu` (`cuda=False`), `os.cpu_count()=12`, default `torch.get_num_threads()=10`, `torch.get_num_interop_threads()=10`, verified `torch.set_num_threads(4)` -> `4`.

---

## 2. Logic Chain

1. From Observation 1 & 4 (`git status -s`, `git diff --stat`, and `test_t1_f22_manifest_sha256_verification` passing), all historical `p1*`, `p2*`, `p3*`, `exploratory/*`, and `MANIFEST_SHA256.json` files are confirmed 100% unmodified prior to M3.
2. From Observation 2 (`test_t1_f22_p4_isolation_boundary` and `test_t2_f22_new_file_outside_p4_detection`), any M3 deliverable or scratch script placed in `PROJECT_ROOT` outside `p4/`, `tests/`, or `.agents/` (or any new root `.py` file or non-allowlisted root directory) will immediately fail F22 governance checks. All M3 implementation, unit tests, and `p4b_coded_bler.{json,csv}` outputs must stay inside `p4/` (`p4/slot_2d.py`, `p4/tests/test_p4_b_2d_fec.py`, `p4/results/`).
3. From Observation 3, M3 extends M1/M2 conventions directly: it reuses `PROFILES` (`train3`, `equal4`, `long6`, `long12`, `sparse12`), 16-QAM Gray mapping (`SIGMA=sqrt(10)`), Rapp PA (`A=1.25/2.25`), `llrs > 0 => bit 1` polarity, integer bit/block accounting, disjoint `torch.Generator` seeding, and `torch.set_num_threads(4)`.
4. From Observation 5 & 6, the M1 (14/14), M2 (53/53), and E2E (156/253 active pass, 97 pending M3/M4 skips) baselines are green on CPU (`PyTorch 2.12.1+cpu`, Python `3.12.7`), giving an exact pre-M3 baseline.

---

## 3. Caveats

- Only representative historical files across P1, P2, P3, and `exploratory/*` were read line-by-line (full coverage of `ofdm_baseline.py`, `p1b_representation.py`, `p1c_architecture.py`, `p1e_replication.py`, `p2_models.py`, `p2_run.py`, `p2_symmetry.py`, `test_p2_integrity.py`, `p3_a_mechanism.py`, `p3b_time_link.py`, `p3b_receivers.py`, `p3b_train_eval.py`, `p3c_precision.py`, `exploratory/16qam_joint_pa_probe/probe.py`, `exploratory/p3de_tdl_surrogate/{p3de_profile_bridge.py, p3de_experiment.py}`), rather than every report/packaging script (`*_make_report.py`, `*_package.py`).
- Running `tests.test_p4_e2e` overwrote `p4/results/latency_cpu.csv` and `p4/results/timing.json` because `test_t1_f07_benchmark_script_invocation` and `test_t1_f07_latency_hierarchy_ordering` write to `"p4/results"` by design. Neither file is in `MANIFEST_SHA256.json` or tracked by git.

---

## 4. Conclusion

The repository is clean and ready for Milestone 3 (`p4/slot_2d.py`, `p4/tests/test_p4_b_2d_fec.py`, `p4/results/p4b_coded_bler.{json,csv}`) with all historical files untouched (`git diff --stat` empty, F22 passing), a verified baseline of **14/14 M1**, **53/53 M2**, and **156/253 active E2E** tests passing (97 skipped pending M3/M4), and strict F22 isolation rules confining all new M3 code and artifacts to `p4/` and `tests/`.

---

## 5. Verification Method

Run the following read-only checks from ``:
1. `git status -s && git diff --stat` — verify only untracked `.agents/`, `PROJECT.md`, `TEST_INFRA.md`, `TEST_READY.md`, `p4/`, `tests/` exist and zero tracked files are modified.
2. `python -m unittest p4.tests.test_p4_0_simulator -v` — expect `Ran 14 tests`, `OK`, exit code `0`.
3. `python -m unittest p4.tests.test_p4_a_equivariance -v` — expect `Ran 53 tests`, `OK`, exit code `0`.
4. `python -m unittest tests.test_p4_e2e -v` — expect `Ran 253 tests`, `OK (skipped=97)`, exit code `0`.
5. Any new root `.py` file not matching `p1*/p2*/p3*/test_p*/ofdm_baseline.py` or any root directory not in the F22 allowlist invalidates F22 compliance.
