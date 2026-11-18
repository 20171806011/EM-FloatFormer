# EM-FloatFormer Stage P4: Test Infrastructure & Verification Specification

**Document**: `TEST_INFRA.md`  
**Scope**: Stage P4 End-to-End (E2E) Test Architecture, Invariants, and Governance  
**Repository**: `EM-FloatFormer`  
**Master Entry Point**: `tests/test_p4_e2e.py`  
**Test Suite Size**: Exactly 253 Test Cases (Tier 1: 110, Tier 2: 110, Tier 3: 22, Tier 4: 11)  
**Standard Environment**: Python 3.10+, PyTorch 2.x, CPU (`torch.set_num_threads(4)`) / Optional CUDA  

---

## 1. Executive Summary & Testing Philosophy

Stage P4 of EM-FloatFormer transitions the architecture from static 1D QPSK models to an electromagnetic-induction native Transformer operating under Gray-coded 16-QAM, nonlinear Rapp power amplifier (PA) distortion with AM-AM and AM-PM effects, dynamic continuous Doppler Inter-Carrier Interference (ICI), Carrier Frequency Offset (CFO), 2D multi-symbol time-frequency slot grids ($N_{\text{symb}}=14, N_{\text{sc}}=32/64$) with sparse DMRS ($\le 10\%$), self-contained native outer channel coding (Rate-1/2 LDPC / Convolutional) with soft decoding, $2^3$ factorial mechanism ablations, FP32/BF16/FP16 mixed-precision dynamic range audits, and pre-frozen 10-seed confirmatory replication with sign-flip permutation tests and Holm adjustment.

### Core Testing Pillars:
1. **Opaque-Box & Interface Independence**:
   - The test suite interacts exclusively through public APIs, mathematical tensor contracts, and CLI interfaces.
   - Zero coupling to private member variables or hidden flags.
2. **Physical & Mathematical Invariant-Driven Assertions**:
   - Assertions enforce exact electromagnetic laws, $C^* = \mathbb{R}_{>0} \times \mathrm{U}(1)$ gauge symmetry, conservation of energy, Bussgang orthogonality $\mathbb{E}[x d^*] = 0$, continuous Doppler phase evolution, and cyclic prefix parity.
   - Zero trivial facade assertions (`assert x is not None`). Every assertion checks quantifiable physical bounds, exact integers, or mathematical identities.
3. **Progressive Testability**:
   - `MilestoneResolver` gracefully detects module readiness on disk (`p4.simulator`, `p4.classical_baselines`, `p4.em_floatformer`, `p4.slot_2d`, etc.).
   - Available modules (e.g. M1 simulator and baselines, and F22 governance) run rigorous live assertions immediately.
   - Pending milestone modules raise informative `unittest.SkipTest` notices without false passes or mock cheating.
   - Upon completion of all milestones, all 253 tests pass with 0 failures, 0 errors, and 0 skips.
4. **Historical Immutability & Sealed Baseline Governance**:
   - Stages P1, P2, P3 and exploratory artifacts are sealed under cryptographic SHA256 hashes in `MANIFEST_SHA256.json`.
   - The test suite strictly preserves historical integrity and operates read-only on sealed files.
5. **Deterministic Reproducibility**:
   - Deterministic seed controls ensure bit-exact reproducibility across test runs.

---

## 2. Directory Layout & Module Organization

The test infrastructure is strictly segregated under `tests/` at the workspace root:

```text
EM-FloatFormer/
├── TEST_INFRA.md                            # This document
├── MANIFEST_SHA256.json                     # Cryptographic manifest of historical files
├── p4/                                      # Stage P4 implementation modules
│   ├── __init__.py
│   ├── simulator.py                         # F1, F2, F3: Dynamic Simulator & Rapp PA
│   ├── classical_baselines.py               # F4, F5, F6: Bussgang-LMMSE & Iterative DD-PA
│   ├── benchmark_latency.py                 # F7: CPU/GPU Latency Benchmark Harness
│   ├── em_floatformer.py                    # F8–F12: Propagator Transformer & C* Gauge
│   ├── slot_2d.py                           # F13–F17: 2D Slot, Continuous ICI, Outer FEC
│   ├── ablation_probing.py                  # F18, F19: Factorial Matrix & Numeric Audit
│   ├── confirmatory_protocol.py             # F20, F21, F22: Frozen Protocol & Permutation Tests
│   └── results/                             # Benchmark CSV/JSON outputs
└── tests/
    ├── __init__.py                          # Test package init
    ├── test_p4_e2e.py                       # Unified test runner & suite aggregator (253 tests)
    └── e2e/                                 # Modular tier test suites & harness
        ├── __init__.py
        ├── harness.py                       # Reference signal generators, constants & MilestoneResolver
        ├── test_tier1_feature_coverage.py   # Tier 1: F1–F22 Feature Coverage (110 tests)
        ├── test_tier2_boundary_corner.py    # Tier 2: F1–F22 Boundary & Corner Cases (110 tests)
        ├── test_tier3_cross_feature.py      # Tier 3: Cross-Feature Interactions (22 tests)
        └── test_tier4_real_workloads.py     # Tier 4: Real-World Workload Scenarios (11 tests)
```

---

## 3. Test Runner Commands & Execution Semantics

### Standard Commands

1. **Full E2E Test Suite (All 253 tests)**:
   ```bash
   python -m unittest tests/test_p4_e2e.py -v
   ```
   Or via pytest:
   ```bash
   pytest tests/test_p4_e2e.py -v
   ```

2. **Tier-Specific Execution**:
   - **Tier 1 (Feature Coverage, 110 tests)**:
     ```bash
     python -m unittest tests.e2e.test_tier1_feature_coverage -v
     ```
   - **Tier 2 (Boundary & Corner Cases, 110 tests)**:
     ```bash
     python -m unittest tests.e2e.test_tier2_boundary_corner -v
     ```
   - **Tier 3 (Cross-Feature Combinations, 22 tests)**:
     ```bash
     python -m unittest tests.e2e.test_tier3_cross_feature -v
     ```
   - **Tier 4 (Real-World Workload Scenarios, 11 tests)**:
     ```bash
     python -m unittest tests.e2e.test_tier4_real_workloads -v
     ```

3. **Feature-Filtered Execution**:
   Filter by feature code (e.g. `f01`, `f08`, `f22`):
   ```bash
   python -m unittest -k "f08" tests/test_p4_e2e.py -v
   ```

4. **Environment Controls**:
   All benchmark and latency-sensitive tests set:
   ```python
   import torch
   torch.set_num_threads(4)
   ```

---

## 4. Feature Inventory & 4-Tier Test Allocation Matrix

The 253 tests comprehensively cover all 22 features (F1–F22) across the 4 test tiers:

| Feature ID | Feature Name | Primary Source Module | Tier 1 (Coverage) | Tier 2 (Boundary) | Tier 3 (Cross) | Tier 4 (Workload) | Total Tests |
|:----------:|:-------------|:---------------------:|:-----------------:|:-----------------:|:--------------:|:-----------------:|:-----------:|
| **F1** | Oversampled Rapp PA | `p4.simulator` | 5 | 5 | P01, P02, P03 | S01, S06, S11 | 15 |
| **F2** | Dynamic Multipath Simulator | `p4.simulator` | 5 | 5 | P01, P04 | S01, S02, S06 | 14 |
| **F3** | Static-Limit Parity Check | `p4.simulator` | 5 | 5 | - | - | 10 |
| **F4** | Bussgang Decomposition & LMMSE | `p4.classical_baselines` | 5 | 5 | P02, P05 | S06 | 13 |
| **F5** | Calibrated 16-QAM Soft LLRs | `p4.classical_baselines` | 5 | 5 | P05, P06 | S01, S11 | 13 |
| **F6** | Iterative DD-PA Cancellation | `p4.classical_baselines` | 5 | 5 | P03, P07 | S06 | 13 |
| **F7** | Classical Latency Benchmark | `p4.benchmark_latency` | 5 | 5 | - | - | 10 |
| **F8** | Exact $C^*$ Gauge Normalization | `p4.em_floatformer` | 5 | 5 | P08, P09, P10 | S04 | 14 |
| **F9** | Hermitian Green's Attention | `p4.em_floatformer` | 5 | 5 | P08, P11, P12 | S01 | 14 |
| **F10** | Bidirectional Pilot-Data Attention | `p4.em_floatformer` | 5 | 5 | P11, P13 | S05 | 13 |
| **F11** | Joint Physical Consistency Loss | `p4.em_floatformer` | 5 | 5 | P13, P14 | - | 12 |
| **F12** | Numerical Equivariance Verification | `p4.em_floatformer` | 5 | 5 | P09, P14 | S04 | 13 |
| **F13** | 2D Slot Simulator & Sparse DMRS | `p4.slot_2d` | 5 | 5 | P15, P16 | S01, S02, S05 | 14 |
| **F14** | Continuous Doppler ICI & CFO | `p4.slot_2d` | 5 | 5 | P04, P15, P17 | S01, S03, S06 | 15 |
| **F15** | Native Outer Channel Coding | `p4.slot_2d` | 5 | 5 | P06, P18 | S01, S11 | 13 |
| **F16** | 2D Classical Baselines | `p4.slot_2d` | 5 | 5 | P07, P16, P19 | S02, S06 | 14 |
| **F17** | Coded BLER Evaluation | `p4.slot_2d` | 5 | 5 | P17, P18, P19 | S01, S02, S11 | 15 |
| **F18** | Factorial Ablation Matrix | `p4.ablation_probing` | 5 | 5 | P12, P20 | S09 | 13 |
| **F19** | Numeric Stability & Precision Audit | `p4.ablation_probing` | 5 | 5 | P10 | S04, S10 | 13 |
| **F20** | Pre-Frozen Confirmatory Protocol | `p4.confirmatory_protocol` | 5 | 5 | P20, P21 | S08 | 13 |
| **F21** | Sign-Flip Permutation Tests | `p4.confirmatory_protocol` | 5 | 5 | P21, P22 | S08 | 13 |
| **F22** | Repository Governance & Immutability | Manifest Auditor | 5 | 5 | P22 | - | 11 |
| **Total** | - | - | **110** | **110** | **22** | **11** | **253** |

---

## 5. Physical Invariants & Quantitative Acceptance Thresholds

Every test method asserts against explicit mathematical bounds:

| Physical Invariant | Mathematical Identity / Formulation | Numeric Acceptance Bound | Tested Features |
|:---|:---|:---|:---|
| **$C^*$ Scale Invariance** | $\text{LLR}(r Y, r H, r^2 \sigma_W^2) = \text{LLR}(Y, H, \sigma_W^2)$ for $r \in [2^{-12}, 2^{12}]$ | $\max \|\Delta \text{Logit}\| < 10^{-5}$, exactly 0 bit decision flips | F8, F12, F19 |
| **$C^*$ Phase Invariance** | $\text{sign}(\text{LLR}(e^{j\phi} Y)) = \text{sign}(\text{LLR}(Y))$ for all $\phi \in [0, 2\pi)$ | Exactly 0 bit decision flips across 16 phase circle samples | F8, F9, F12 |
| **Channel Equivariance** | $\hat{H}(c Y) = c \hat{H}(Y)$ for all $c \in \mathbb{C}^*$ | Relative error $\frac{\|\hat{H}(cY) - c\hat{H}(Y)\|}{\|c\hat{H}(Y)\| + 10^{-12}} < 10^{-5}$ | F8, F12 |
| **Static Parity Limit** | Dynamic simulator $Y_{\text{sim}}$ vs matrix product $Y_{\text{mat}} = H \odot X$ when $f_d=0, A=\infty$ | Relative error $\frac{\|Y_{\text{sim}} - Y_{\text{mat}}\|_\infty}{\|Y_{\text{mat}}\|_\infty} < 10^{-5}$ | F1, F2, F3 |
| **Bussgang Orthogonality** | Uncorrelated distortion: $\mathbb{E}[x d^*] = 0$ where $f_{\text{PA}}(x) = \gamma x + d$ | Normalized correlation $\frac{\|\mathbb{E}[x d^*]\|}{\sigma_x^2} < 10^{-3}$ | F1, F4 |
| **Bussgang Gain Consistency** | Analytical $\gamma$ vs empirical $\frac{\mathbb{E}[f(x) x^*]}{\mathbb{E}[\|x\|^2]}$ | Relative difference $< 1\%$ across $10^5$ samples | F4 |
| **LLR Calibration** | Posterior probability calibration: $P(b_i=1\|z) = \text{sigmoid}(\text{LLR})$ | Expected Calibration Error (ECE) $< 0.05$ on AWGN | F5 |
| **DMRS Sparsity Bound** | Pilot Resource Elements / Total Slot Resource Elements | Overhead $\le 10.0\%$ (comb-2 on symbols 2 & 11 $= 7.14\%$) | F13 |
| **Permutation Enumeration** | Sign-flip permutation test over $N=10$ paired differences | Exhaustive evaluation of exactly $2^{10} = 1024$ states | F20, F21 |
| **Holm Adjustment** | Step-down adjusted $p$-values: $p_{(k)}^{\text{Holm}} = \min(1, \max_{j \le k} (M - j + 1) p_{(j)})$ | Monotonically non-decreasing: $p_{(1)} \le \dots \le p_{(M)} \le 1.0$ | F21 |
| **Cryptographic Immutability**| SHA256 of historical files listed in `MANIFEST_SHA256.json` | 100% byte-exact hash match, 0 file alterations | F22 |

---

## 6. Progressive Execution & Milestone Dependency Protocol

To facilitate continuous integration during staged development:
- **`MilestoneResolver`**:
  Defined in `tests/e2e/harness.py`, dynamically inspects `p4.*` modules.
  When an implementation file is not yet created (e.g. `p4.em_floatformer` pending Milestone M2), it raises:
  ```python
  raise unittest.SkipTest("[Pending Milestone M2] Module 'p4.em_floatformer' not yet implemented.")
  ```
- **Milestone Mapping**:
  - **M1 (Delivered)**: F1–F7 in `p4/simulator.py`, `p4/classical_baselines.py`, `p4/benchmark_latency.py`.
  - **F22 (Delivered)**: Sealed manifest validation against `MANIFEST_SHA256.json`.
  - **M2 (Pending)**: F8–F12 in `p4/em_floatformer.py`.
  - **M3 (Pending)**: F13–F17 in `p4/slot_2d.py`.
  - **M4 (Pending)**: F18–F19 in `p4/ablation_probing.py`, F20–F21 in `p4/confirmatory_protocol.py`.
- **Zero False-Pass Guarantee**: Tests NEVER pass silently when code is missing. They either execute genuine physical assertions or explicitly report `SkipTest`.

---

## 7. Governance, Cryptographic Manifest & Audit Trail

The historical integrity of the repository is certified through `MANIFEST_SHA256.json`.
- Historical deliverables from Stages P1, P2, P3, and exploratory artifacts are permanently sealed.
- Feature F22 tests verify that:
  1. All 180+ entries in `MANIFEST_SHA256.json` match byte-for-byte on disk.
  2. No modifications have occurred in sealed files.
  3. No stray files exist in the project root outside `p4/`, `tests/`, and `.agents/teamwork/`.
  4. Hashing accounts for cross-platform newline variations (CRLF vs LF) to prevent spurious failures across operating systems.

---

## 8. Handoff & Sign-Off Criteria

The test suite satisfies the following criteria for complete qualification:
1. `python -m unittest tests/test_p4_e2e.py -v` executes cleanly with exit code 0.
2. Total discovered test cases: exactly **253**.
3. All M1 and F22 tests pass cleanly with live invariant checks.
4. All pending milestone tests skip cleanly with informative dependency explanations.
5. Zero unhandled exceptions, zero failures, zero errors.
