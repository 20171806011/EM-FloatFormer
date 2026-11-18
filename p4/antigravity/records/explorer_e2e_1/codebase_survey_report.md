# Codebase Survey & Execution Environment Report: Stage P4 E2E Testing Track

- **Author**: Explorer 1 (`explorer_e2e_1`), EM-FloatFormer Stage P4 E2E Testing Track
- **Date**: 2026-11-15
- **Working Directory**: ``
- **State Directory**: `explorer_e2e_1`

---

## 1. Executive Summary

This report delivers a rigorous survey of the repository layout, historical milestone artifacts, Python execution environment, interface contracts, and immutability governance for the **EM-FloatFormer Stage P4 E2E Testing Track**.

### Core Discoveries & Direct Directives
1. **Repository Layout**:
   - `p4/` and `tests/` directories do not yet exist on disk; `p4/` is currently under active construction by Milestone 1 (`worker_m1_1` under `sub_orch_m1`), and `tests/` is the assigned deliverable for this E2E track (`tests/test_p4_e2e.py`).
   - `exploratory/` exists, containing two historical probe directories: `16qam_joint_pa_probe` (preliminary 16-QAM and memoryless Rapp PA tests) and `p3de_tdl_surrogate` (standalone 3GPP TDL channel surrogate).
   - Historical files (`p1*.py`, `p2*.py`, `p3*.py`, `ofdm_baseline.py`) and legacy test suites (`test_p1*.py`, `test_p2*.py`, `test_p3*.py`) occupy the repository root.
2. **Execution Environment**:
   - Python: **3.12.7** (packaged by Anaconda, 64-bit AMD64) at `python`.
   - PyTorch: **2.12.1+cpu** (CPU only; `torch.cuda.is_available() == False`).
   - Key Libraries: **NumPy 1.26.4**, **SciPy 1.13.1**, **Pytest 9.1.1**, **Matplotlib 3.9.2**, and Python standard library `unittest`.
   - Hardware: Host CPU with 10 logical threads. CPU execution requires standardizing on `torch.set_num_threads(4)` for reproducible latency benchmarking.
3. **Cryptographic Manifest & Windows CRLF Invariance**:
   - `MANIFEST_SHA256.json` tracks 115 historical files. A live byte-level audit verified that **114 of 115 files match perfectly** once Windows CRLF line-ending translation is accounted for (56 exact binary matches + 58 matches under CRLF $\to$ LF normalization).
   - The single non-matching file is `README.md`, which was updated in commit `f878a73` after the manifest was frozen at commit `0c7088b`. `git status` confirms that the working tree is completely clean on all tracked files.
4. **Historical Immutability & Test Execution Guardrail**:
   - Running blind test discovery from project root (`python -m unittest discover .`) is **strictly forbidden and fatal**: it executes historical tests that fail due to omitted large weight checkpoints (`p2_supplement/*.pt`), a legacy broken import (`p3b_receivers.py` attempting `from p3_a import dft_threshold`), and raw CRLF hash discrepancies in `test_p2_integrity.py`.
   - The E2E test runner must **always** be executed via targeted invocation: `python -m unittest tests/test_p4_e2e.py -v` (or `pytest tests/test_p4_e2e.py -v`).
5. **Progressive Test Runner Design**:
   - Because M1 through M4 are being delivered in parallel or sequential milestones, `tests/test_p4_e2e.py` must employ dynamic import detection (`importlib.import_module` with conditional `unittest.skipIf`).
   - When M1 components (`p4.simulator`, `p4.classical_baselines`) are present, their Tier 1–4 tests execute immediately and pass with exit code 0.
   - For pending milestones (M2 `em_floatformer`, M3 `slot_2d`, M4 `confirmatory_protocol`), tests either verify formal Protocol/interface duck-typing mocks or report cleanly as skipped, ensuring **zero crashes and exit code 0** at every stage of development.

---

## 2. Repository & Filesystem Survey

### 2.1 Working Directory Structure
A comprehensive inspection of `` reveals the following layout:

```text
EM-FloatFormer/
├── .agents/                                 # Multi-agent coordination metadata
│   └── teamwork/
│       ├── explorer_e2e_1/                  # [CURRENT AGENT] Codebase survey state & reports
│       ├── explorer_e2e_2/                  # Test architecture & TEST_INFRA blueprint
│       ├── spec_miner_e2e_1/                # F1–F22 specification mining
│       └── worker_m1_1/                     # Milestone 1 implementation worker
├── exploratory/                             # Historical exploratory probes (READ-ONLY, IMMUTABLE)
│   ├── 16qam_joint_pa_probe/                # Preliminary 16-QAM and Rapp AM-AM probe
│   └── p3de_tdl_surrogate/                  # 3GPP TDL profile table surrogate
├── p1*_results/, p2*_results/, p3*_results/ # Historical experimental data & CSVs (IMMUTABLE)
├── p2_supplement/                           # Supplemental P2 results and checkpoint hashes (IMMUTABLE)
├── p1*.py, p2*.py, p3*.py                   # Historical pipeline implementations (IMMUTABLE)
├── test_p1*.py, test_p2*.py, test_p3*.py    # Historical test suites (IMMUTABLE)
├── MANIFEST_SHA256.json                     # Root cryptographic manifest (IMMUTABLE)
├── PROJECT.md                               # Living master blueprint & feature tracker
├── README.md                                # Repository overview & scientific documentation
├── requirements.txt                         # Base dependencies: torch>=2.1,<3, matplotlib>=3.8,<4
├── requirements_p3c.txt                     # P3-C dependencies: torch>=2.10,<2.11, numpy>=2.3
├── p4/                                      # [UNDER CONSTRUCTION by worker_m1_1] Stage P4 modules
└── tests/                                   # [TARGET of E2E track] tests/test_p4_e2e.py
```

### 2.2 Status of `p4/` and `tests/`
- `p4/`: Verified non-existent prior to this survey (`Test-Path` returned `False`). It is exclusively owned by parallel milestone workers (starting with `worker_m1_1` for M1).
- `tests/`: Verified non-existent. It must be created by the E2E track test author to house `tests/test_p4_e2e.py`.

### 2.3 Status of `exploratory/`
- `exploratory/16qam_joint_pa_probe`:
  - Contains `probe.py`, `audit.py`, `calibration_sensitivity.py`, `PROTOCOL.md`, `REPORT.md`, `timing.py`.
  - Implements synthetic 16-QAM mapping (`q16`), threshold-based hard slicing (`hard_bits`), and memoryless Rapp PA amplitude compression (`pa_magnitude`).
  - Limitations observed: Lacks oversampling ($L \ge 4$), lacks AM-PM phase distortion ($\Phi(r)$), lacks dynamic Doppler ICI, and lacks calibrated soft LLR computation.
- `exploratory/p3de_tdl_surrogate`:
  - Standalone 3GPP TDL profile table approximation (`p3de_experiment.py`, `p3de_profile_bridge.py`).

### 2.4 Cryptographic Manifest Audit (`MANIFEST_SHA256.json`)
The manifest tracks 115 files. An automated SHA256 audit against disk revealed:
- **Missing files**: 0
- **Exact binary hash matches**: 56 files
- **Matches under CRLF $\to$ LF normalization**: 58 files
- **Total matched files**: 114 / 115 (99.13%)
- **Single unmatching file**: `README.md`
  - Expected SHA256 in manifest: `39e8f1eac095b821f4eefe4b042e7376eb0bf9f36e93ed98b4bdaf0aef20f054`
  - Actual raw SHA256: `d4604ef059e8d084b0e0327182905d59b0d5dbf25b13aa2ddd7733be9a4d2ef6`
  - Cause: Commit history shows `MANIFEST_SHA256.json` was committed at `0c7088b` (end of P2), while `README.md` was subsequently updated in commit `f878a73` ("Publish independent TDL-table surrogate...").
  - Git working tree verification: `git status` reports `nothing added to commit but untracked files present (.agents/, PROJECT.md)`. All tracked repository files are completely pristine and unmodified.

---

## 3. Python Runtime & Execution Environment

### 3.1 Interpreter & System Details
- **Python Version**: `3.12.7 | packaged by Anaconda, Inc. | (main, Oct 4 2024, 13:17:27) [MSC v.1929 64 bit (AMD64)]`
- **Python Executable**: `python`
- **Platform**: Windows 11 AMD64

### 3.2 Installed Packages & Versions
| Package | Installed Version | Requirement Spec | Status / Notes |
|---------|-------------------|------------------|----------------|
| `torch` | `2.12.1+cpu` | `>=2.1,<3` | Installed; CPU build only |
| `numpy` | `1.26.4` | `>=2.3` in p3c | Fully functional; compatible with PyTorch 2.12 |
| `scipy` | `1.13.1` | standard | Fully functional |
| `pytest` | `9.1.1` | standard | Fully functional; available for test runner |
| `matplotlib` | `3.9.2` | `>=3.8,<4` | Fully functional |
| `unittest` | built-in (3.12.7) | standard library | Standard test harness runner |

### 3.3 Hardware & Execution Constraints
1. **CUDA Availability**: `torch.cuda.is_available() == False`.
   - **Implication**: All PyTorch tensor operations must default to device `'cpu'`. Code should use `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`. Any unconditional `.cuda()` call will crash.
2. **CPU Threading**:
   - Host machine has 10 CPU threads.
   - For reproducible latency benchmarks (Feature F7), standard practice established in historical milestones (`p1e`, `p2_run`, `p3b_timing_extended`) is to explicitly set:
     ```python
     torch.set_num_threads(4)
     ```
3. **External Dependencies**:
   - Proprietary/heavy external libraries (such as NVIDIA Sionna) are not available offline. All physical models (Rapp PA, Doppler ICI FIR, LDPC/convolutional FEC, Bussgang LMMSE) must remain self-contained native PyTorch/NumPy implementations.

---

## 4. Interface Contracts & Milestone Construction Status

### 4.1 In-Flight Construction: Milestone 1 (Stage P4-0)

1. `p4/simulator.py`:
   - **Feature F1 (Oversampled Rapp PA)**:
     - Signature: `rapp_pa(x_time, knee=2.0, A=2.25, alpha=0.5, beta=1.0, oversample=4) -> y_time`
     - AM-AM: $g(r) = \frac{r}{(1 + (r/A)^{2p})^{1/(2p)}}$ with $p=2, A \in \{1.25, 2.25\}$.
     - AM-PM: $\Phi(r) = \frac{\alpha r^2}{1 + (r/\beta)^2}$ with $\alpha=0.5, \beta=1.0$ ($\Phi \equiv 0$ when linear).
     - Oversampling $L \ge 4$ via frequency-domain zero-padding and decimation.
   - **Feature F2 (Dynamic Multipath Simulator)**:
     - Numerology: $N=32$ (or 64), $N_{CP}=16$.
     - Continuous Doppler: $h_l[n] = h_{l,0} e^{j 2\pi f_{d, l} n T_s / N}$.
     - Time-domain FIR convolution, CP insertion and discard.
   - **Feature F3 (Static-Limit Parity Check)**:
     - When $f_d = 0$ and $A = \infty$, dynamic simulation matches static matrix model $Y = H X + W$ with relative error $< 10^{-5}$.
2. `p4/classical_baselines.py`:
   - **Feature F4 (Bussgang Decomposition & LMMSE)**:
     - Attenuation $\gamma = \mathbb{E}[f(x) x^*] / \mathbb{E}[|x|^2]$, distortion variance $\sigma_D^2 = P_{out} - |\gamma|^2 \sigma_x^2$.
     - Equalizer: `bussgang_lmmse(yp, yd, nv, A) -> (h_hat, llrs, hard_bits)`.
   - **Feature F5 (Calibrated 16-QAM Soft LLRs)**:
     - Exact or piecewise max-log posterior log-odds for Gray-coded 16-QAM bits.
     - Calibrated by effective noise variance $\sigma_k^2 = \sigma_{v,k}^2 / \mu_k^2$.
   - **Feature F6 (Iterative Decision-Directed PA Cancellation)**:
     - Signature: `iterative_dd_pa_cancellation(yp, yd, nv, A, iterations=2) -> (h_hat, llrs, hard_bits)`.
     - 2–3 iterations of time-domain distortion reconstruction, channel filtering, subtraction, and re-equalization.
3. `p4/benchmark_latency.py`:
   - **Feature F7 (Classical Latency Benchmark)**:
     - Benchmarking CPU (4 threads) / GPU execution for LS, DFT-8, Bussgang-LMMSE, and IDD-PA (1, 2, 3 iterations) at batch sizes 1 and 64.
     - Outputs: `p4/results/latency_cpu.csv` and `p4/results/timing.json`.
4. `p4/tests/test_p4_0_simulator.py`:
   - Unit tests covering F1–F7.

### 4.2 Downstream Milestone Contracts (M2, M3, M4)
Based on `PROJECT.md` and survey reports:
- **M2 (`p4/em_floatformer.py`)**:
  - `create_floatformer(use_gauge_norm=True, use_hermitian_attention=True, use_joint_pilot_data=True)`
  - Forward call: `model(yp, yd, nv) -> (h_hat, llrs, pa_dist_est)`
  - Input shapes: `yp`: `(B, N_p)`, `yd`: `(B, N_d)`, `nv`: `(B,)`
  - Output shapes: `h_hat`: `(B, N)`, `llrs`: `(B, N_d, 4)`, `pa_dist_est`: `(B, N_d)`
  - Invariance: Strict invariance under gain scale $r \in [2^{-12}, 2^{12}]$ and phase rotation $e^{j\phi}$.
- **M3 (`p4/slot_2d.py`)**:
  - 2D grid: $(N_{\text{symb}} \times N_{\text{sc}}) = (14 \times 32)$ or $(14 \times 64)$.
  - Sparse DMRS: boolean mask `dmrs_mask` with $\le 10\%$ pilot density.
  - Rate-1/2 FEC encoder and soft decoder (LDPC / Convolutional).
  - 2D-LMMSE and 2D-IDD-PA baselines.
- **M4 (`p4/ablation_probing.py`, `p4/confirmatory_protocol.py`)**:
  - $2^3$ orthogonal ablation configurations.
  - Mixed-precision dynamic range audits across FP32, BF16, FP16.
  - Confirmatory protocol: 10 seeds ($74001–74010$), integer bit/block error accounting, two-sided 1024-state sign-flip permutation tests with Holm adjustment.

---

## 5. Historical Immutability & Test Execution Guardrails

### 5.1 Why Blind Test Discovery Fails
Executing `python -m unittest discover .` from the repository root scans all `test_*.py` files and fails with 3 failures and 7 errors:
1. `test_p3b_link.py`, `test_p3c_precision.py`, `test_p3_a_mechanism.py`: Attempt `from p3_a import dft_threshold`, which fails with `ModuleNotFoundError: No module named 'p3_a'` because the file is named `p3_a_mechanism.py`.
2. `test_p2_supplement_independent.py`: Attempts to load `p2_supplement/s1_checkpoints/*.pt`, which were omitted from the git tree due to file size.
3. `test_p2_integrity.py`: Compares raw byte SHA256 of `P2E_FREEZE.json`, failing due to Windows CRLF line endings (the hash matches identically under LF normalization).

### 5.2 Immutable Boundary Rules for E2E Tests
1. **Never use blind discovery**: Never run `python -m unittest discover .` or `pytest .`.
2. **Always use targeted test invocation**:
   ```bash
   python -m unittest tests/test_p4_e2e.py -v
   # or
   pytest tests/test_p4_e2e.py -v
   ```
3. **No file writes outside designated output folders**: E2E tests must never write, touch, or delete files in the repository root or historical directories (`p1*`, `p2*`, `p3*`, `exploratory/`). All persistent test artifacts must be routed to `p4/results/` or handled purely in memory.
4. **CRLF vs LF handling in Feature F22 tests**: Any test verifying `MANIFEST_SHA256.json` or protocol freezes must perform CRLF $\to$ LF normalization on text files before hashing, or inspect git tree blobs directly, to avoid false positive failures on Windows.

---

## 6. Progressive Architecture Recommendations for `tests/test_p4_e2e.py`

### 6.1 The Progressive Execution Problem
Parallel tracks mean that at any given moment:
- M1 (`p4.simulator`, `p4.classical_baselines`) may be complete or nearing completion.
- M2 (`p4.em_floatformer`), M3 (`p4.slot_2d`), and M4 (`p4.confirmatory_protocol`) are planned and pending.

If `test_p4_e2e.py` executes a top-level unconditional import of an unbuilt module:
```python
# WRONG: Will immediately crash test collection with ModuleNotFoundError!
from p4.em_floatformer import create_floatformer
from p4.slot_2d import SlotSimulator2D
```
The entire test suite will crash before a single test runs, producing exit code 1.

### 6.2 The Progressive Architecture Pattern
To ensure `test_p4_e2e.py` executes cleanly and returns exit code 0 at every stage of development, the following architectural pattern is recommended:

#### Pattern 1: Dynamic Import Detection Helper
```python
import importlib
from typing import Any, Optional

def try_import(module_path: str) -> Optional[Any]:
    try:
        return importlib.import_module(module_path)
    except (ImportError, ModuleNotFoundError):
        return None

# Milestone module probes
p4_sim = try_import("p4.simulator")
p4_base = try_import("p4.classical_baselines")
p4_bench = try_import("p4.benchmark_latency")
p4_trans = try_import("p4.em_floatformer")
p4_slot = try_import("p4.slot_2d")
p4_ablation = try_import("p4.ablation_probing")
p4_confirm = try_import("p4.confirmatory_protocol")
```

#### Pattern 2: Conditional Skip Decorator
```python
import unittest

# Example for Feature F1 (M1)
@unittest.skipIf(p4_sim is None, "M1: p4.simulator is under construction")
class TestF1_OversampledRappPA(unittest.TestCase):
    def test_rapp_am_am_compression(self):
        # Executes against real p4.simulator once available
        ...

# Example for Feature F8 (M2)
@unittest.skipIf(p4_trans is None, "M2: p4.em_floatformer is under construction")
class TestF8_GaugeNormalization(unittest.TestCase):
    def test_pilot_rms_gauge_anchor(self):
        ...
```

#### Pattern 3: Protocol/Mock Interface Compliance for Pending Milestones
For Tier 1 interface validation where the test verifies protocol compliance:
- For existing modules, test against the real module.
- For pending modules, the test suite can inspect duck-typing conformance against the contract specification in `PROJECT.md`:
```python
class TestInterfaceContracts(unittest.TestCase):
    def test_m1_to_m2_tensor_contract(self):
        """Verify M1 signal generator outputs conform to M2 expected input shapes and dtypes."""
        if p4_sim is None:
            self.skipTest("p4.simulator not yet available")
        # Generate synthetic M1 signal
        y, yp, bits, h, nv = p4_sim.generate_synthetic_frame(...)
        self.assertEqual(yp.dtype, torch.complex64)
        self.assertEqual(yp.dim(), 2)  # (B, N_p)
        self.assertEqual(nv.dtype, torch.float32)
```

### 6.3 Feature Mapping & Availability Matrix
This matrix guides the test author on module sources and current readiness:

| Feature | Name | Milestone | Implementation Module | Test Readiness |
|---------|------|-----------|----------------------|:--------------:|
| **F1** | Oversampled Rapp PA | M1 | `p4.simulator` | In Progress (worker_m1_1) |
| **F2** | Dynamic Multipath Simulator | M1 | `p4.simulator` | In Progress (worker_m1_1) |
| **F3** | Static-Limit Parity Check | M1 | `p4.simulator` | In Progress (worker_m1_1) |
| **F4** | Bussgang Decomposition & LMMSE | M1 | `p4.classical_baselines` | In Progress (worker_m1_1) |
| **F5** | Calibrated 16-QAM Soft LLRs | M1 | `p4.classical_baselines` | In Progress (worker_m1_1) |
| **F6** | Iterative DD-PA Cancellation | M1 | `p4.classical_baselines` | In Progress (worker_m1_1) |
| **F7** | Classical Latency Benchmark | M1 | `p4.benchmark_latency` | In Progress (worker_m1_1) |
| **F8** | Exact $C^*$ Gauge Normalization | M2 | `p4.em_floatformer` | Planned (M2) |
| **F9** | Hermitian Green's Attention | M2 | `p4.em_floatformer` | Planned (M2) |
| **F10** | Bidirectional Pilot-Data Attention | M2 | `p4.em_floatformer` | Planned (M2) |
| **F11** | Joint Physical Consistency Loss | M2 | `p4.em_floatformer` | Planned (M2) |
| **F12** | Numerical Equivariance Verification | M2 | `p4.em_floatformer` | Planned (M2) |
| **F13** | 2D Slot Simulator & Sparse DMRS | M3 | `p4.slot_2d` | Planned (M3) |
| **F14** | Continuous Doppler ICI & CFO | M3 | `p4.slot_2d` | Planned (M3) |
| **F15** | Native Outer Channel Coding | M3 | `p4.slot_2d` | Planned (M3) |
| **F16** | 2D Classical Baselines | M3 | `p4.slot_2d` | Planned (M3) |
| **F17** | Coded BLER Evaluation | M3 | `p4.slot_2d` | Planned (M3) |
| **F18** | Factorial Ablation Matrix | M4 | `p4.ablation_probing` | Planned (M4) |
| **F19** | Numeric Stability & Precision Audit | M4 | `p4.ablation_probing` | Planned (M4) |
| **F20** | Pre-Frozen Confirmatory Protocol | M4 | `p4.confirmatory_protocol` | Planned (M4) |
| **F21** | Sign-Flip Permutation Tests | M4 | `p4.confirmatory_protocol` | Planned (M4) |
| **F22** | Repository Governance & Immutability | M4 | Self-contained / `MANIFEST_SHA256.json` | **Ready Immediately** |

> **Note on F22**: Feature F22 tests repository governance and historical immutability. Because `MANIFEST_SHA256.json` and historical files already exist, **all Tier 1–4 tests for F22 can be written and executed immediately** without waiting for any `p4/` modules!

---

## 7. Concrete Next Steps for the E2E Team

1. **For Explorer 2 (`test_architecture_blueprint.md`)**:
   - Incorporate the dynamic import helper and conditional skip structure into the overall test hierarchy.
   - Establish the test case ID convention (e.g. `test_tier1_f01_rapp_knee`, `test_tier2_f08_gain_extremes`, `test_tier3_f01_f02_rapp_doppler`, `test_tier4_scenario_01_severe_fading`).
   - Finalize `TEST_INFRA.md` structure.
2. **For Spec Miner (`spec_mining_report.md`)**:
   - Extract numeric thresholds and tolerance bounds ($\epsilon < 10^{-5}$, exact integer error counters, specific Doppler frequency bounds $f_d T_s \in [0.001, 0.05]$) directly aligned with CPU FP32 precision.
3. **For Test Implementer (`tests/test_p4_e2e.py`)**:
   - Implement the test suite with self-contained mathematical helpers where appropriate (e.g. ideal static model verification, CRLF-safe SHA256 checking, Gray 16-QAM symbol tables).
   - Ensure the test suite exits with code 0 on both `python -m unittest tests/test_p4_e2e.py -v` and `pytest tests/test_p4_e2e.py -v`.
