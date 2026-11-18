# Stage P4 E2E Testing Track: 4-Tier Test Architecture Blueprint
**EM-FloatFormer Stage P4: Electromagnetic-Induction Native Transformer under 16-QAM and Nonlinear PA Distortion**

- **Document Version**: 1.0.0
- **Author**: Teamwork Explorer (`explorer_e2e_2`)
- **Date**: 2026-11-15
- **Working Directory**: ``
- **Target Suite Entry Point**: `tests/test_p4_e2e.py`
- **Infrastructure Blueprint**: `TEST_INFRA.md` (to be created at project root)
- **Total Test Cases Architected**: **253 test cases** (Tier 1: 110, Tier 2: 110, Tier 3: 22, Tier 4: 11)

---

## 1. Executive Summary & Core Testing Philosophy

### 1.1 Scope and Objective
This blueprint defines the architecture, test harness, execution semantics, and implementation outline for the comprehensive 4-Tier End-to-End (E2E) Test Suite for Stage P4 of the `EM-FloatFormer` project.

Stage P4 advances EM-FloatFormer from static 1D QPSK models to an electromagnetic-induction native Transformer operating under Gray-coded 16-QAM, nonlinear Rapp power amplifier (PA) distortion with AM-AM and AM-PM effects, dynamic continuous Doppler Inter-Carrier Interference (ICI), Carrier Frequency Offset (CFO), 2D multi-symbol time-frequency slot grids ($N_{\text{symb}}=14, N_{\text{sc}}=32/64$) with sparse DMRS ($\le 10\%$), self-contained native outer channel coding (Rate-1/2 LDPC / Convolutional) with soft decoding, $2^3$ factorial mechanism ablations, FP32/BF16/FP16 mixed-precision dynamic range audits, and pre-frozen 10-seed confirmatory replication with sign-flip permutation tests and Holm adjustment.

### 1.2 Five Pillars of Test Architecture Philosophy
1. **Opaque-Box & Public Interface Governance**:
   - The test suite acts as an external certifier. It interacts exclusively with public functions, classes, CLI interfaces, and physical signal tensors.
   - Zero coupling to private member variables, hidden hooks, or ad-hoc internals. If an internal implementation changes, the tests remain valid as long as physical input/output contracts hold.
2. **Physical & Mathematical Invariant-Driven Assertions**:
   - Tests assert physical laws: electromagnetic reciprocity, $C^* = \mathbb{R}_{>0} \times \mathrm{U}(1)$ gauge symmetry, conservation of energy, Bussgang orthogonality $\mathbb{E}[x d^*] = 0$, continuous Doppler phase evolution, cyclic convolution parity in static limits, and probability calibration of soft log-likelihood ratios (LLRs).
   - Trivial assertions (e.g. `assert x is not None`) are strictly prohibited. Every test asserts quantifiable numeric bounds, exact integer counts, or closed-form theoretical values.
3. **Graceful Progressive Testability**:
   - Milestone 1 (M1: F1–F7) is currently under active implementation, with M2 (F8–F12), M3 (F13–F17), and M4 (F18–F22) planned in sequence.
   - The test suite must be executable immediately. If a module is not yet delivered, tests for that module report structured `unittest.SkipTest` notices with clear milestone dependency labels, while all delivered modules run full assertions.
   - When all milestones are complete, the entire 253-test suite executes with **0 failures, 0 errors, and 0 skips**.
4. **Historical Immutability & Zero Contamination**:
   - Historical records (Stages P1, P2, P3, exploratory artifacts, and `MANIFEST_SHA256.json`) are cryptographically sealed.
   - Tests run in development integrity mode without modifying or polluting historical files.
5. **Deterministic Execution & Reproducibility**:
   - All tests use deterministic seed management.
   - Benchmarking standardizes on 4 CPU threads (`torch.set_num_threads(4)`) with optional GPU execution if CUDA is present.

---

## 2. Directory Layout & Infrastructure Specification

### 2.1 File Organization
The E2E test infrastructure is organized under `tests/` at the repository root, modularized into clean sub-suites while exposed through a unified master runner:

```text
EM-FloatFormer/
├── TEST_INFRA.md                            # Comprehensive test infrastructure documentation
├── tests/
│   ├── __init__.py                          # Package initialization
│   ├── test_p4_e2e.py                       # Unified test runner / test suite aggregator
│   └── e2e/                                 # Modular test tiers and test harness
│       ├── __init__.py
│       ├── harness.py                       # Reference signal generators, tolerance constants & loaders
│       ├── test_tier1_feature_coverage.py   # Tier 1: F1–F22 Happy paths (110 tests)
│       ├── test_tier2_boundary_corner.py    # Tier 2: F1–F22 Extremes & singularities (110 tests)
│       ├── test_tier3_cross_feature.py      # Tier 3: Pairwise feature interactions (22 tests)
│       └── test_tier4_real_workloads.py     # Tier 4: End-to-end communication workloads (11 tests)
├── p4/                                      # Stage P4 implementation package
│   ├── simulator.py                         # F1, F2, F3
│   ├── classical_baselines.py               # F4, F5, F6
│   ├── benchmark_latency.py                 # F7
│   ├── em_floatformer.py                    # F8, F9, F10, F11, F12
│   ├── slot_2d.py                           # F13, F14, F15, F16, F17
│   ├── ablation_probing.py                  # F18, F19
│   ├── confirmatory_protocol.py             # F20, F21, F22
│   └── tests/                               # Unit-level test suites per milestone
└── MANIFEST_SHA256.json                     # Cryptographic manifest of historical files
```

### 2.2 Test Runner Command & Pass/Fail Semantics
- **Primary Test Runner Command**:
  ```bash
  python -m unittest tests/test_p4_e2e.py -v
  ```
- **Alternative Pytest Runner Command**:
  ```bash
  pytest tests/test_p4_e2e.py -v
  ```
- **Exit Code Semantics**:
  - `0`: All executed tests passed successfully.
  - Non-zero (`1` or `2`): One or more assertions failed or encountered an uncaught runtime exception.
- **Selective Execution Commands**:
  - Run Tier 1 only: `python -m unittest tests.e2e.test_tier1_feature_coverage -v`
  - Run Tier 2 only: `python -m unittest tests.e2e.test_tier2_boundary_corner -v`
  - Run Tier 3 only: `python -m unittest tests.e2e.test_tier3_cross_feature -v`
  - Run Tier 4 only: `python -m unittest tests.e2e.test_tier4_real_workloads -v`
  - Run single feature (e.g. F8): `python -m unittest -k "f08" tests/test_p4_e2e.py -v`

### 2.3 Test Naming Conventions
To guarantee programmatic tracking, report generation, and traceability back to requirements:
- **Tier 1 (Feature Coverage)**:
  `test_t1_f{XX}_{mnemonic}_{aspect}`
  - Example: `test_t1_f01_rapp_am_am_compression`, `test_t1_f08_gauge_scale_invariance`
- **Tier 2 (Boundary & Corner Cases)**:
  `test_t2_f{XX}_{mnemonic}_{corner_type}`
  - Example: `test_t2_f08_gauge_extreme_gain_2p12`, `test_t2_f14_max_doppler_ici_boundary`
- **Tier 3 (Cross-Feature Combinations)**:
  `test_t3_p{YY}_{featureA}_{featureB}_{interaction}`
  - Example: `test_t3_p01_f01_f02_rapp_pa_dynamic_doppler`, `test_t3_p08_f08_f09_gauge_green_attention`
- **Tier 4 (Real-World Workloads)**:
  `test_t4_s{ZZ}_{scenario_mnemonic}`
  - Example: `test_t4_s01_high_speed_train_doppler_strong_pa_ldpc`, `test_t4_s04_dynamic_range_shadowing_swing`

---

## 3. Progressive Test Execution & Graceful Milestone Adaptation

### 3.1 Milestone Dependencies and Progressive Matrix
The implementation progresses through four milestones:
- **M1 (Stage P4-0)**: F1–F7 (Simulator & Classical Baselines)
- **M2 (Stage P4-A)**: F8–F12 (EM-Native Propagator Transformer & $C^*$ Gauge Equivariance)
- **M3 (Stage P4-B)**: F13–F17 (2D Multi-Symbol Slot, Continuous Doppler/CFO, Outer FEC & BLER)
- **M4 (Stage P4-C/D)**: F18–F22 (Factorial Probing, Numeric Audits, Confirmatory Protocol & Governance)
- **M5 (Stage P4 Final)**: Full E2E Integration (Tiers 1–4, 253 tests passing with 0 skips)

### 3.2 The Test Harness Resolution Protocol (`tests/e2e/harness.py`)
To prevent test execution from breaking with `ModuleNotFoundError` during intermediate development while maintaining strict requirement checking:

```python
import unittest
import importlib
from typing import Any, Optional, Tuple

class MilestoneResolver:
    """
    Safely resolves Stage P4 modules or raises descriptive SkipTest
    when a milestone is still under construction.
    """
    @staticmethod
    def get_module(module_name: str, required_milestone: str) -> Any:
        try:
            return importlib.import_module(f"p4.{module_name}")
        except ImportError as e:
            raise unittest.SkipTest(
                f"[PROGRESSIVE EXECUTION] Module 'p4.{module_name}' not available. "
                f"Requires {required_milestone}. Dependency error: {e}"
            )

    @staticmethod
    def get_callable(module_name: str, func_name: str, required_milestone: str) -> Any:
        mod = MilestoneResolver.get_module(module_name, required_milestone)
        if not hasattr(mod, func_name):
            raise unittest.SkipTest(
                f"[PROGRESSIVE EXECUTION] Callable '{func_name}' not found in 'p4.{module_name}'. "
                f"Requires {required_milestone} completion."
            )
        return getattr(mod, func_name)
```

### 3.3 Behavioral Interface Assertion vs Dummy Mocking
- The test harness implements **zero dummy stub mocks**. Tests either execute against genuine implementations or safely skip pending milestone readiness.
- The harness provides self-contained reference signal generators (e.g. ideal Gray 16-QAM constellation tables, synthetic multipath impulse response generators, reference Gold sequences) used strictly as independent ground truth to verify the system under test.

---

## 4. Feature Inventory & Precise Mathematical Acceptance Criteria (F1 to F22)

Every feature is cataloged with its exact physical equation, tensor contract, acceptance criteria, and failure invariants:

| # | Feature | Exact Mathematical Formulation | Tensor Signatures & Shapes | Acceptance Bounds |
|---|---|---|---|---|
| **F1** | Oversampled Rapp PA | $g(r) = \frac{r}{(1 + (r/A)^{2p})^{1/(2p)}}$, $\Phi(r) = \frac{\alpha r^2}{1 + (r/\beta)^2}$ at oversample factor $L \ge 4$ | In: $x_{\text{time}} \in \mathbb{C}^{B \times N}$<br>Out: $y_{\text{time}} \in \mathbb{C}^{B \times N}$ | Parity with linear limit ($A \to \infty \implies \Delta < 10^{-6}$); spectral regrowth at $L \ge 4$ contains out-of-band energy |
| **F2** | Dynamic Multipath Simulator | $h_l[n] = h_{l,0} e^{j 2\pi f_{d,l} \frac{n T_s}{N}}$, $y[n] = \sum_l h_l[n] x[n - d_l] + w[n]$ with CP=16 | In: $X_{\text{freq}} \in \mathbb{C}^{B \times N}$<br>Out: $Y_{\text{freq}} \in \mathbb{C}^{B \times N}$ | When $f_d > 0$, off-diagonal frequency transfer matrix entries $> 10^{-4}$ (Doppler ICI verified) |
| **F3** | Static-Limit Parity Check | $Y[k] = H[k] X[k] + W[k]$ when $f_d = 0$ and $A = \infty$ | In: $X \in \mathbb{C}^{B \times N}, H \in \mathbb{C}^{B \times N}$<br>Out: $Y \in \mathbb{C}^{B \times N}$ | Relative error $\|Y_{\text{sim}} - Y_{\text{mat}}\|_2 / \|Y_{\text{mat}}\|_2 < 10^{-5}$ |
| **F4** | Bussgang Decomposition & LMMSE | $\gamma = \frac{\mathbb{E}[f(x) x^*]}{\mathbb{E}[\|x\|^2]}$, $\sigma_D^2 = P_{\text{out}} - \|\gamma\|^2 \sigma_x^2$, $W = \frac{\gamma^* H^*}{\|\gamma H\|^2 + \|H\|^2 \sigma_D^2 + \sigma_W^2}$ | In: $Y_p, Y_d \in \mathbb{C}^{B \times N}, \sigma_W^2, A$<br>Out: $\hat{H} \in \mathbb{C}^{B \times N}, \hat{X}_d \in \mathbb{C}^{B \times N}$ | Bussgang orthogonality $\|\mathbb{E}[d x^*]\| < 10^{-3}$; LMMSE MSE lower than standard linear LS under $A=1.25$ |
| **F5** | Calibrated 16-QAM Soft LLRs | Posterior log-odds: $L(b_i) = \ln \frac{P(b_i=1\|Y)}{P(b_i=0\|Y)}$, scaled by $\sigma_{\text{eff}}^2 = \sigma_D^2 + \sigma_W^2$ | In: $\hat{X}_d \in \mathbb{C}^{B \times N}, \sigma_{\text{eff}}^2$<br>Out: $\text{LLR} \in \mathbb{R}^{B \times N \times 4}$ | Calibration ECE $< 0.05$; $\text{sign}(\text{LLR})$ matches Gray quadrant; monotonic scaling with SNR |
| **F6** | Iterative DD-PA Cancellation | $\hat{d}^{(i)} = f_{\text{PA}}(\hat{x}^{(i)}) - \gamma \hat{x}^{(i)}$, $Y_{\text{clean}}^{(i+1)} = Y_d - \hat{H} \hat{D}^{(i)}$ (2–3 iterations) | In: $Y_p, Y_d, \sigma_W^2, A$<br>Out: $\hat{X}_{\text{final}}, \text{LLRs}_{\text{final}}, \text{hist}$ | Monotonic residual error reduction across iterations: $\text{MSE}^{(i+1)} < \text{MSE}^{(i)}$ under $A=1.25$ |
| **F7** | Classical Latency Benchmark | Wall-clock latency measurement on CPU (4 threads) / GPU for batch 1 & 64 | In: Execution script<br>Out: CSV & JSON timing artifacts | Warmup completed, median and P95 recorded, valid positive floats, execution time $\le 50$ ms/symbol |
| **F8** | Exact $C^*$ Gauge Normalization | Anchor $s_0 = \sqrt{\frac{1}{N}\sum \|Y_p\|^2}$, $\bar{Y} = Y / s_0$, $\text{cRMSNorm}(Z) = \frac{Z}{\text{RMS}(Z)} \odot \gamma_c$ | In: $Y \in \mathbb{C}^{B \times N}, s_0 \in \mathbb{R}_{>0}$<br>Out: $\tilde{Y} \in \mathbb{C}^{B \times N}$ (RMS $\equiv 1$) | Invariance under $r \in [2^{-12}, 2^{12}]$: $\Delta \text{Logits} < 10^{-5}$, channel estimate scales as $r \hat{H}$ |
| **F9** | Hermitian Green's Attention | $Q=X W_Q^T, K=X W_K^T (b=0)$, RoPE $\tilde{Q} = Q e^{j m \theta}$, $S_{m,n} = \frac{1}{\sqrt{d}}\text{Re}(\langle \tilde{Q}_m, \tilde{K}_n \rangle_\mathbb{C})$, $f(z) = g(\|z\|)\frac{z}{\|z\|}$ | In: $Z \in \mathbb{C}^{B \times N \times D}$<br>Out: $Z_{\text{out}} \in \mathbb{C}^{B \times N \times D}$ | Exact phase rotation invariance: $S_{m,n}(e^{j\phi} Z) = S_{m,n}(Z)$; radial activation preserves phase $\arg(f(z))=\arg(z)$ |
| **F10** | Bidirectional Pilot-Data Attention | Dual-stream cross-attention: $\text{Pilot} \leftrightarrow \text{Data}$ for joint channel propagation & distortion feedback | In: $Z_{\text{pilot}}, Z_{\text{data}}$<br>Out: $Z_{\text{pilot}}', Z_{\text{data}}'$ | BER reduction over unidirectional pilot-only model under strong PA ($A=1.25$) by $\ge 0.5$ dB equivalent |
| **F11** | Joint Physical Consistency Loss | $\mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{BCE}} + \lambda_1 \mathcal{L}_{\text{NMSE}} + \lambda_2 \mathcal{L}_{\text{PA-res}}$ | In: $\text{LLR}, H_{\text{est}}, Y_d, b, H_{\text{true}}$<br>Out: Scalar loss $\mathcal{L}_{\text{joint}} \in \mathbb{R}_{\ge 0}$ | Scale-invariant loss: $\mathcal{L}(c Y, c H, \|c\|^2 \sigma^2) = \mathcal{L}(Y, H, \sigma^2)$ for all $c \in \mathbb{C}^*$ |
| **F12** | Numerical Equivariance Verification | Machine-level test of $C^*$ equivariance across 24 gain octaves and full $[0, 2\pi)$ phase circle | In: Scale $r \in [2^{-12}, 2^{12}]$, phase $\phi \in [0, 2\pi)$<br>Out: Equivariance error | Max logit absolute error $< 10^{-5}$, exactly 0 bit decision flips across all scale-phase pairs |
| **F13** | 2D Slot Simulator & Sparse DMRS | Grid: $N_{\text{symb}}=14, N_{\text{sc}}=32/64$, DMRS at symbols 2 & 11, comb-2 ($\le 10\%$ overhead) | In: Config parameters<br>Out: Slot tensor $\mathbb{C}^{B \times 14 \times N_{\text{sc}}}$, DMRS mask | DMRS RE count $\le 10\%$ of total REs; payload positions disjoint from pilot positions |
| **F14** | Continuous Doppler ICI & CFO | High-Doppler time-varying fading ($f_d T_{\text{sym}} \le 0.05$) and continuous sample CFO drift $e^{j 2\pi \epsilon t / N}$ | In: Slot grid, $f_d$, $\epsilon$<br>Out: Received grid $\mathbb{C}^{B \times 14 \times N}$ | Continuous phase roll verified across consecutive symbols; off-diagonal leakage energy $> 0$ |
| **F15** | Native Outer Channel Coding | Rate-1/2 FEC (LDPC Min-Sum / Convolutional BCJR) with soft LLR decoding | In: Information bits $u \in \{0, 1\}^K$, LLRs $\in \mathbb{R}^{N}$<br>Out: Decoded bits $\hat{u} \in \{0, 1\}^K$ | Parity check $H \hat{c}^T = 0 \pmod 2$; soft decoding provides coding gain $> 3$ dB over uncoded 16-QAM |
| **F16** | 2D Classical Baselines | Separable 2D-LMMSE ($R_{\text{time}} \otimes R_{\text{freq}}$) and 2D Iterative Decision-Directed PA Cancellation | In: Received slot, DMRS pilots, $\sigma_W^2$<br>Out: Grid $\hat{H}_{s,k}$, soft LLRs, hard bits | 2D interpolation error lower than nearest-neighbor or 1D LMMSE on dynamic channels |
| **F17** | Coded BLER Evaluation | Block error rate evaluation: $\text{BLER} = \frac{1}{N_{\text{blk}}}\sum \mathbb{I}(\hat{u} \ne u)$ across OOD channels | In: Monte Carlo simulation<br>Out: BLER curve across SNR | Monotonic decrease with SNR; exact integer accounting of block errors |
| **F18** | Factorial Ablation Matrix | Orthogonal $2^3$ ablation matrix: $C^*$ gauge ($F_1$), Green Attn ($F_2$), Joint Prop ($F_3$) | In: 8 model configurations<br>Out: Main effects & interaction contrasts | All 8 arms instantiate cleanly; contrast calculations follow Yates' matrix orthogonality |
| **F19** | Numeric Stability & Precision Audit | Stress testing in FP32, BF16, and FP16 across 24 octaves of dynamic range ($r \in [2^{-12}, 2^{12}]$) | In: Model forward passes under autocast<br>Out: Stability metrics | 0 nonfinite (`NaN`/`Inf`) outputs; max logit deviation between half and FP32 $< 0.15$; flip rate $< 0.05\%$ |
| **F20** | Pre-Frozen Confirmatory Protocol | Locked protocol over 10 fresh seeds ($74001–74010$) with integer bit/block error accounting | In: Seed list, run configuration<br>Out: Integer CSV error records | Exact integer counts ($E_{\text{bits}}, N_{\text{bits}} \in \mathbb{Z}_{\ge 0}$); seeds match declared list strictly |
| **F21** | Sign-Flip Permutation Tests | Two-sided 1024-state exact permutation test with Holm-Bonferroni step-down adjustment | In: 10 paired metric deltas $\delta_s$<br>Out: Exact $p$-values and Holm-adjusted $p$ | Exhaustive sum over $2^{10}=1024$ states; Holm monotonic adjustment $p_{(k)}^{\text{Holm}} \ge p_{(k-1)}^{\text{Holm}}$ |
| **F22** | Repository Governance & Immutability | SHA256 integrity check against historical records in `MANIFEST_SHA256.json` | In: Workspace root<br>Out: Hash comparison table | 100% byte-exact hash match for all historical files; zero file additions/deletions in sealed paths |

---

## 5. Comprehensive 4-Tier Test Case Specification (253 Tests)

### 5.1 Tier 1: Feature Coverage (110 Tests, >=5 per Feature F1 to F22)
Tier 1 verifies nominal "happy path" functionality and isolated execution for each feature:

#### Feature F1: Oversampled Rapp PA (5 tests)
1. `test_t1_f01_rapp_am_am_compression`: Verifies magnitude compression obeys $g(r) = r / (1 + (r/A)^4)^{0.25}$ for $A=2.25$ and $A=1.25$.
2. `test_t1_f01_rapp_am_pm_phase_rotation`: Verifies phase distortion obeys $\Phi(r) = \frac{\alpha r^2}{1 + (r/\beta)^2}$ with $\alpha=0.5, \beta=1.0$.
3. `test_t1_f01_oversampling_spectral_regrowth`: Confirms oversampling $L=4$ captures out-of-band spectral regrowth without Nyquist aliasing.
4. `test_t1_f01_linear_limit_infinite_saturation`: Verifies that as $A \to \infty$ and $\alpha=0$, output equals input with relative error $< 10^{-6}$.
5. `test_t1_f01_power_attenuation_bussgang_scaling`: Confirms average output power $P_{\text{out}} < P_{\text{in}}$ under strong saturation ($A=1.25$).

#### Feature F2: Dynamic Multipath Simulator (5 tests)
6. `test_t1_f02_multipath_delay_profile_fir`: Verifies causal FIR convolution with 4-tap power delay profile matches expected delay structure.
7. `test_t1_f02_cyclic_prefix_insertion_removal`: Confirms CP length 16 prepending and discard preserves payload integrity.
8. `test_t1_f02_doppler_time_varying_evolution`: Verifies tap gain evolves according to $e^{j 2\pi f_d n T_s / N}$ across samples.
9. `test_t1_f02_ici_off_diagonal_energy`: Confirms non-zero Doppler generates measurable off-diagonal subcarrier interference in frequency domain.
10. `test_t1_f02_awgn_noise_variance_scaling`: Verifies empirical noise variance matches configured $\sigma_W^2$ within $5\%$ tolerance over $10^5$ samples.

#### Feature F3: Static-Limit Parity Check (5 tests)
11. `test_t1_f03_zero_doppler_linear_pa_matrix_parity`: Verifies dynamic simulator output matches static matrix product $Y = H X$ to $\|Y_{\text{sim}} - Y_{\text{mat}}\|_\infty < 10^{-5}$.
12. `test_t1_f03_single_subcarrier_impulse_response`: Injects a single non-zero subcarrier and verifies circular convolution property.
13. `test_t1_f03_snr_sweep_parity`: Verifies static parity across SNR values from 0 dB to 30 dB.
14. `test_t1_f03_different_fft_sizes_parity`: Confirms static parity holds for both $N=32$ and $N=64$.
15. `test_t1_f03_multi_channel_realizations_parity`: Verifies parity across 20 distinct random channel seeds.

#### Feature F4: Bussgang Decomposition & LMMSE (5 tests)
16. `test_t1_f04_bussgang_gain_analytical_vs_empirical`: Verifies computed $\gamma$ matches Monte Carlo expectation $\mathbb{E}[f(x) x^*] / \mathbb{E}[|x|^2]$ within $1\%$.
17. `test_t1_f04_distortion_variance_calculation`: Confirms $\sigma_D^2 = P_{\text{out}} - |\gamma|^2 \sigma_x^2$ is strictly positive for nonlinear PA.
18. `test_t1_f04_bussgang_orthogonality`: Verifies uncorrelatedness $\mathbb{E}[x d^*] < 10^{-3}$.
19. `test_t1_f04_lmmse_equalizer_filter_weights`: Verifies MMSE filter $W[k] = \frac{\gamma^* H^*[k]}{|\gamma H[k]|^2 + |H[k]|^2 \sigma_D^2 + \sigma_W^2}$ matches analytical Wiener solution.
20. `test_t1_f04_lmmse_superiority_over_ls`: Confirms Bussgang-LMMSE achieves lower symbol MSE than standard LS under $A=1.25$.

#### Feature F5: Calibrated 16-QAM Soft LLRs (5 tests)
21. `test_t1_f05_gray_mapping_bit_definitions`: Verifies 16-QAM symbol mapping correctly encodes 4 bits with Gray distance 1.
22. `test_t1_f05_max_log_llr_sign_quadrant_concordance`: Confirms signs of $LLR(b_0)$ and $LLR(b_2)$ match real and imaginary quadrants.
23. `test_t1_f05_magnitude_boundary_bits`: Confirms signs of $LLR(b_1)$ and $LLR(b_3)$ correctly identify inner vs outer constellation rings.
24. `test_t1_f05_llr_variance_scaling`: Verifies LLR magnitudes scale inversely with effective noise variance $\sigma_{\text{eff}}^2$.
25. `test_t1_f05_awgn_probability_calibration_ece`: Confirms expected calibration error (ECE) of sigmoid probabilities $< 0.05$ on AWGN.

#### Feature F6: Iterative DD-PA Cancellation (5 tests)
26. `test_t1_f06_iteration_loop_execution`: Verifies 2-iteration and 3-iteration DD-PA loops run and return expected history shapes.
27. `test_t1_f06_distortion_reconstruction_subtraction`: Confirms estimated distortion $\hat{D} = \text{FFT}(f(\hat{x}) - \gamma \hat{x})$ is subtracted from received grid.
28. `test_t1_f06_monotonic_mse_reduction`: Verifies $\text{MSE}^{(2)} < \text{MSE}^{(1)} < \text{MSE}^{(0)}$ under strong PA ($A=1.25$) at 15 dB SNR.
29. `test_t1_f06_ber_improvement_across_iterations`: Confirms bit error rate decreases with iteration count.
30. `test_t1_f06_linear_channel_early_convergence`: Confirms that under linear PA ($A=\infty$), iteration 1 already matches optimal LMMSE.

#### Feature F7: Classical Latency Benchmark (5 tests)
31. `test_t1_f07_benchmark_script_invocation`: Executes `p4.benchmark_latency` and verifies exit code 0.
32. `test_t1_f07_csv_timing_artifact_generation`: Confirms `p4/results/latency_cpu.csv` is created with valid header and numeric columns.
33. `test_t1_f07_json_timing_artifact_generation`: Confirms `p4/results/timing.json` contains structured timing records.
34. `test_t1_f07_cpu_thread_concurrency_control`: Confirms benchmark respects `torch.set_num_threads(4)`.
35. `test_t1_f07_latency_hierarchy_ordering`: Verifies LS latency $<$ Bussgang-LMMSE $<$ DD-PA (3 iterations).

#### Feature F8: Exact $C^*$ Gauge Normalization (5 tests)
36. `test_t1_f08_pilot_energy_gauge_anchor`: Verifies $s_0 = \sqrt{\frac{1}{N}\sum |Y_p|^2}$ extracts correct root power.
37. `test_t1_f08_unit_rms_normalization`: Confirms normalized grid $\bar{Y} = Y / s_0$ has exactly unit RMS power ($\pm 10^{-6}$).
38. `test_t1_f08_c_rms_norm_phase_preservation`: Confirms Complex RMSNorm preserves complex phase angles $\arg(\text{cRMSNorm}(Z)) = \arg(Z)$.
39. `test_t1_f08_channel_rescaling_exactness`: Confirms output physical channel estimate recovers true physical scale via $s_0 \cdot \bar{H}$.
40. `test_t1_f08_scale_invariance_under_arbitrary_gain`: Multiplies input by $r = 3.7 \times 10^3$ and confirms bit logits are identical to $< 10^{-5}$.

#### Feature F9: Hermitian Green's Propagator Attention (5 tests)
41. `test_t1_f09_complex_linear_zero_bias`: Verifies complex linear projection has zero additive bias ($b \equiv 0$).
42. `test_t1_f09_complex_linear_homogeneity`: Confirms $W(c Z) = c W(Z)$ for complex scalar $c$.
43. `test_t1_f09_relative_phase_rope`: Verifies rotary embeddings modulate relative subcarrier distances $m - n$ via complex phasor multiplication.
44. `test_t1_f09_hermitian_inner_product_phase_cancellation`: Confirms $\text{Re}(\langle e^{j\phi} Q, e^{j\phi} K \rangle) = \text{Re}(\langle Q, K \rangle)$.
45. `test_t1_f09_radial_activation_u1_equivariance`: Verifies $f(e^{j\phi} Z) = e^{j\phi} f(Z)$ for radial activation $f(z) = g(|z|) \frac{z}{|z|}$.

#### Feature F10: Bidirectional Pilot-Data Attention (5 tests)
46. `test_t1_f10_dual_stream_embedding_shapes`: Verifies joint token grid properly embeds pilot and data tokens with coordinate indicators.
47. `test_t1_f10_cross_attention_pilot_to_data`: Confirms data tokens query pilot channel representations.
48. `test_t1_f10_cross_attention_data_to_pilot`: Confirms pilot tokens query data distortion estimates.
49. `test_t1_f10_gradient_flow_bidirectional`: Confirms non-zero gradients flow from data loss back into pilot representations and vice versa.
50. `test_t1_f10_ber_gain_under_strong_pa`: Verifies bidirectional attention achieves lower BER than decoupled pilot-only backbone.

#### Feature F11: Joint Physical Consistency Loss (5 tests)
51. `test_t1_f11_calibrated_llr_bce_loss`: Verifies BCE loss computes correct binary cross-entropy on 16-QAM bit projections.
52. `test_t1_f11_normalized_channel_nmse_loss`: Confirms $\mathcal{L}_{\text{NMSE}} = \|H - \hat{H}\|^2 / (\|H\|^2 + \epsilon)$ is non-negative and scale-invariant.
53. `test_t1_f11_pa_residual_consistency_loss`: Verifies forward PA residual $\|Y_d - \hat{H} \odot f_{\text{PA}}(\hat{X})\|^2$ is computed without ground truth channel.
54. `test_t1_f11_joint_loss_weighting_schedule`: Confirms $\mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{BCE}} + \lambda_1 \mathcal{L}_{\text{NMSE}} + \lambda_2 \mathcal{L}_{\text{PA-res}}$ weights sum properly.
55. `test_t1_f11_loss_gradient_scale_invariance`: Verifies parameter gradients $\nabla_\theta \mathcal{L}$ are invariant under input scaling $Y \mapsto r Y$.

#### Feature F12: Numerical Equivariance Verification (5 tests)
56. `test_t1_f12_gain_invariance_grid_24_octaves`: Evaluates model across 24 gain octaves $r \in [2^{-12}, 2^{12}]$ and asserts max logit delta $< 10^{-5}$.
57. `test_t1_f12_phase_invariance_circle`: Evaluates model across 16 phase angles $\phi \in [0, 2\pi)$ and asserts zero bit decision flips.
58. `test_t1_f12_simultaneous_scale_and_phase`: Tests random combinations of $(r, \phi) \in \mathbb{C}^*$ and confirms zero bit flips.
59. `test_t1_f12_channel_estimate_c_equivariance`: Confirms $\hat{H}(c Y) = c \hat{H}(Y)$ with relative error $< 10^{-5}$.
60. `test_t1_f12_standalone_unit_runner`: Executes `p4/tests/test_p4_a_equivariance.py` and confirms exit code 0.

#### Feature F13: 2D Slot Simulator & Sparse DMRS (5 tests)
61. `test_t1_f13_grid_dimensions_14_symb`: Verifies 2D time-frequency grid generates $N_{\text{symb}}=14$ OFDM symbols and $N_{\text{sc}}=32/64$ subcarriers.
62. `test_t1_f13_sparse_dmrs_pattern_allocation`: Confirms DMRS symbols are inserted at symbol indices 2 and 11 in comb-2 pattern.
63. `test_t1_f13_pilot_overhead_bound`: Asserts total DMRS pilot RE overhead is strictly $\le 10\%$ of total slot REs ($7.14\%$).
64. `test_t1_f13_disjoint_pilot_payload_masks`: Verifies pilot mask and data payload mask have zero intersection and span the entire grid.
65. `test_t1_f13_gold_sequence_pseudo_random_pilots`: Verifies DMRS pilots are generated with unit energy QPSK pseudo-random sequences.

#### Feature F14: Continuous Doppler ICI & CFO (5 tests)
66. `test_t1_f14_continuous_time_sample_convolution`: Verifies time-varying channel evolves per time sample rather than per OFDM symbol.
67. `test_t1_f14_high_doppler_subcarrier_leakage`: Confirms Doppler spread $f_d T_{\text{sym}} = 0.05$ produces inter-carrier interference across adjacent bins.
68. `test_t1_f14_continuous_cfo_phase_drift`: Verifies sample-level phase accumulation $e^{j 2\pi \epsilon t / N}$ creates steady phase rotation across slot symbols.
69. `test_t1_f14_cfo_fractional_estimation`: Confirms cyclic prefix correlation reliably estimates normalized CFO $\epsilon$.
70. `test_t1_f14_combined_doppler_and_cfo_grid`: Verifies received 2D grid incorporates both Doppler ICI and CFO phase roll simultaneously.

#### Feature F15: Native Outer Channel Coding (5 tests)
71. `test_t1_f15_rate_half_parity_check_construction`: Verifies native PyTorch Rate-1/2 parity matrix $H$ has valid dimensions $M \times N$ with $M = N/2$.
72. `test_t1_f15_codeword_validity`: Encodes information bits $u$ and verifies syndrome $H c^T \equiv 0 \pmod 2$.
73. `test_t1_f15_min_sum_soft_ldpc_decoding`: Tests Min-Sum soft decoder convergence on synthetic LLRs within 10 iterations.
74. `test_t1_f15_early_stopping_parity_check`: Confirms decoder terminates early when syndrome check passes.
75. `test_t1_f15_coding_gain_waterfall`: Verifies coded BER achieves $> 3$ dB waterfall coding gain over uncoded transmission at $10^{-3}$ BER.

#### Feature F16: 2D Classical Baselines (5 tests)
76. `test_t1_f16_2d_separable_correlation_matrix`: Computes $R_{tf} = R_{\text{time}} \otimes R_{\text{freq}}$ with Jakes time correlation and multipath frequency correlation.
77. `test_t1_f16_2d_lmmse_channel_interpolation`: Interpolates full $14 \times 32$ grid channel from sparse DMRS positions via 2D Wiener filter.
78. `test_t1_f16_2d_lmmse_superior_to_1d`: Confirms 2D-LMMSE achieves lower NMSE than 1D frequency-only LMMSE under Doppler fading.
79. `test_t1_f16_2d_idd_pa_loop_execution`: Verifies 2D Iterative Decision-Directed PA cancellation runs for 2 iterations on 2D slot.
80. `test_t1_f16_2d_idd_pa_residual_cleanup`: Confirms 2D IDD-PA improves LLR accuracy on payload REs affected by PA distortion.

#### Feature F17: Coded BLER Evaluation (5 tests)
81. `test_t1_f17_block_error_counting_exactness`: Verifies a block is declared in error if and only if $\hat{u} \ne u$ for any information bit.
82. `test_t1_f17_bler_monotonic_snr_response`: Confirms BLER decreases monotonically across SNR grid $[0, 5, 10, 15, 20]$ dB.
83. `test_t1_f17_integer_block_counters`: Confirms total blocks and errored blocks are tracked as exact Python integers.
84. `test_t1_f17_ood_doppler_bler_degradation`: Asserts BLER increases predictably when Doppler frequency increases from $50$ Hz to $600$ Hz.
85. `test_t1_f17_ood_delay_spread_bler_resilience`: Benchmarks BLER on held-out `sparse12` channel profile.

#### Feature F18: Factorial Ablation Matrix (5 tests)
86. `test_t1_f18_full_arm_instantiation`: Instantiates full EM-Native Propagator Transformer ($+F_1, +F_2, +F_3$).
87. `test_t1_f18_ablated_arms_instantiation`: Instantiates all 7 ablated variants (Arm 2 through Arm 8) with parameter matching ($\approx 25\text{k}-35\text{k}$ params).
88. `test_t1_f18_yates_contrast_orthogonality`: Verifies the $8 \times 8$ Hadamard/Yates contrast design matrix has orthogonal columns.
89. `test_t1_f18_main_effects_computation`: Confirms main effect functions for $F_1, F_2, F_3$ compute unbiased linear contrasts.
90. `test_t1_f18_two_way_interaction_computation`: Confirms calculation of interaction contrasts $\text{INT}(F_1, F_2)$, $\text{INT}(F_1, F_3)$, $\text{INT}(F_2, F_3)$.

#### Feature F19: Numeric Stability & Precision Audit (5 tests)
91. `test_t1_f19_fp32_baseline_forward`: Verifies model forward pass executes cleanly in standard FP32 with 0 NaNs.
92. `test_t1_f19_bf16_autocast_forward`: Verifies forward pass executes under `torch.bfloat16` CPU autocast without runtime errors.
93. `test_t1_f19_fp16_autocast_forward`: Verifies forward pass executes under `torch.float16` CPU autocast without overflow.
94. `test_t1_f19_dynamic_range_sweep_finite_check`: Sweeps input scale across 24 octaves $r \in [2^{-12}, 2^{12}]$ in FP32/BF16/FP16; asserts all outputs finite.
95. `test_t1_f19_half_precision_logit_discrepancy_bound`: Confirms $\max |L_{\text{half}} - L_{\text{fp32}}| < 0.15$ across test inputs.

#### Feature F20: Pre-Frozen Confirmatory Protocol (5 tests)
96. `test_t1_f20_seed_list_exactness`: Confirms seed set strictly equals $\mathcal{S}_{\text{P4D}} = \{74001, 74002, \dots, 74010\}$.
97. `test_t1_f20_disjointness_from_historical_seeds`: Confirms no seeds in $\mathcal{S}_{\text{P4D}}$ overlap with Stages P1, P2, P3, or exploratory probes.
98. `test_t1_f20_seed_reproducibility`: Confirms running seed 74001 twice yields identical integer bit error counts.
99. `test_t1_f20_integer_bit_and_block_error_records`: Verifies all confirmatory results are recorded as exact integer counts.
100. `test_t1_f20_protocol_hash_freeze_verification`: Verifies SHA256 of `P4D_PROTOCOL.md` matches frozen manifest hash.

#### Feature F21: Sign-Flip Permutation Tests (5 tests)
101. `test_t1_f21_exact_1024_state_enumeration`: Verifies permutation test enumerates all $2^{10} = 1024$ sign configurations without sampling approximation.
102. `test_t1_f21_two_sided_p_value_calculation`: Tests permutation test on known delta vector and verifies analytical $p$-value.
103. `test_t1_f21_zero_delta_null_behavior`: Confirms $p=1.0$ when paired differences are identically zero.
104. `test_t1_f21_identical_signs_extreme_p_value`: Confirms $p = 2 / 1024 \approx 0.00195$ when all 10 seeds show strictly positive improvement.
105. `test_t1_f21_holm_step_down_adjustment_monotonicity`: Verifies Holm adjustment enforces $p_{(1)}^{\text{Holm}} \le p_{(2)}^{\text{Holm}} \le \dots \le p_{(M)}^{\text{Holm}} \le 1.0$.

#### Feature F22: Repository Governance & Immutability (5 tests)
106. `test_t1_f22_manifest_sha256_verification`: Computes SHA256 of all files listed in `MANIFEST_SHA256.json` and verifies 100% match.
107. `test_t1_f22_historical_p1_p2_p3_untouched`: Confirms zero modifications to any `p1*.py`, `p2*.py`, `p3*.py` files.
108. `test_t1_f22_exploratory_directory_read_only`: Confirms files under `exploratory/` remain byte-identical.
109. `test_t1_f22_p4_isolation_boundary`: Asserts all new source code files reside exclusively within the `p4/` package.
110. `test_t1_f22_line_ending_robust_hashing`: Verifies SHA256 checker handles Windows CRLF vs LF line endings without spurious failures.

---

### 5.2 Tier 2: Boundary & Corner Cases (110 Tests, >=5 per Feature F1 to F22)
Tier 2 exercises extreme operating conditions, boundary limits, and mathematical edge cases:

#### Feature F1: Oversampled Rapp PA (5 boundary tests)
111. `test_t2_f01_extreme_clipping_knee_p1`: Tests Rapp PA with $p=1.0$ (soft knee) and verifies asymptotic saturation.
112. `test_t2_f01_hard_clipping_limit_p10`: Tests Rapp PA with $p=10.0$ approximating ideal hard limiter; verifies sharp thresholding at $A$.
113. `test_t2_f01_near_zero_input_amplitude`: Inputs signal with peak amplitude $10^{-15}$ and asserts linear response without NaN or zero division.
114. `test_t2_f01_extreme_overdrive_input_100x_A`: Inputs signal with amplitude $100 \times A$ and asserts output amplitude remains bounded by $A$.
115. `test_t2_f01_nyquist_vs_oversampled_aliasing_boundary`: Demonstrates that $L=1$ Nyquist clipping produces unphysical in-band aliased intermodulation while $L=8$ eliminates it.

#### Feature F2: Dynamic Multipath Simulator (5 boundary tests)
116. `test_t2_f02_max_doppler_nyquist_boundary`: Tests extreme Doppler frequency approaching subcarrier spacing ($f_d T_s \to 0.5$); asserts numeric stability.
117. `test_t2_f02_zero_multipath_flat_fading`: Sets delay spread to 0 (single tap at delay 0); asserts channel frequency response is flat across all subcarriers.
118. `test_t2_f02_extreme_delay_spread_exceeding_cp`: Sets channel delay spread $> N_{\text{cp}}$ (e.g. tap at delay 24 with $N_{\text{cp}}=16$); asserts inter-symbol interference (ISI) occurs.
119. `test_t2_f02_zero_snr_pure_noise_limit`: Sets SNR to $-20$ dB; asserts simulator runs stably and outputs valid Gaussian noise.
120. `test_t2_f02_high_snr_infinite_snr_limit`: Sets $\sigma_W^2 = 0$; asserts received signal contains pure noiseless channel convolution.

#### Feature F3: Static-Limit Parity Check (5 boundary tests)
121. `test_t2_f03_single_path_delay_zero_parity`: Tests static parity for single path at delay 0 with uniform frequency response.
122. `test_t2_f03_max_taps_density_parity`: Tests static parity with channel having non-zero taps at every delay index from 0 to 15.
123. `test_t2_f03_extreme_phase_rotation_parity`: Applies arbitrary constant phase rotation $\theta = \pi \sqrt{2}$ and tests static parity.
124. `test_t2_f03_all_zeros_input_parity`: Inputs all-zero frequency vector; asserts zero output vector to $< 10^{-15}$.
125. `test_t2_f03_single_symbol_vs_batch_parity`: Confirms static parity error remains $< 10^{-5}$ across varying batch sizes ($B \in \{1, 8, 64\}$).

#### Feature F4: Bussgang Decomposition & LMMSE (5 boundary tests)
126. `test_t2_f04_extreme_saturation_A_0p5`: Evaluates Bussgang decomposition under extreme saturation $A=0.5$; asserts valid $\gamma \in (0, 1)$ and $\sigma_D^2 > 0$.
127. `test_t2_f04_linear_limit_A_1000`: Evaluates Bussgang decomposition with $A=1000$; asserts $\gamma \to 1.0$ and $\sigma_D^2 \to 0$ ($\le 10^{-6}$).
128. `test_t2_f04_deep_channel_fade_zero_channel`: Tests LMMSE filter when $H[k] \to 0$; asserts regularized weight $W[k] \to 0$ without division-by-zero NaN.
129. `test_t2_f04_extreme_high_noise_variance`: Tests LMMSE when $\sigma_W^2 = 100$; asserts weight scales smoothly to 0.
130. `test_t2_f04_extreme_low_noise_variance`: Tests LMMSE when $\sigma_W^2 = 10^{-8}$; asserts weight converges to Zero-Forcing inverse $\gamma^{-1} H^{-1}$ (regularized by $\sigma_D^2$).

#### Feature F5: Calibrated 16-QAM Soft LLRs (5 boundary tests)
131. `test_t2_f05_exact_constellation_points_zero_noise`: Feeds noiseless constellation points; asserts LLR signs are strictly correct with high confidence ($|LLR| > 20$).
132. `test_t2_f05_exact_decision_boundaries`: Feeds inputs exactly on quadrant boundaries ($\text{Re}(z) = 0$ or $|\text{Re}(z)| = 2/\sqrt{10}$); asserts $LLR \approx 0$ ($\pm 10^{-4}$).
133. `test_t2_f05_extreme_noise_variance_llr_collapse`: Feeds $\sigma^2 \to \infty$; asserts all LLRs smoothly collapse to 0 (uninformative prior).
134. `test_t2_f05_extreme_low_noise_clipping`: Feeds $\sigma^2 = 10^{-8}$; asserts LLRs do not overflow to `+inf` (clamped or numerically stable).
135. `test_t2_f05_huge_amplitude_outlier_llr`: Injects an extreme outlier symbol ($|z| = 100$); asserts LLR outputs remain finite floats.

#### Feature F6: Iterative DD-PA Cancellation (5 boundary tests)
136. `test_t2_f06_zero_iteration_identity`: Configures iterations=0; asserts output equals initial Bussgang-LMMSE estimate.
137. `test_t2_f06_high_iterations_stability_10_iters`: Runs 10 iterations; asserts algorithm does not diverge or explode in energy.
138. `test_t2_f06_perfect_decision_feedback_oracle`: Provides ground truth symbols as decisions; asserts distortion is completely eliminated up to channel noise.
139. `test_t2_f06_zero_snr_decision_feedback_robustness`: Runs IDD-PA at $-10$ dB SNR; asserts decisions do not induce catastrophic error propagation.
140. `test_t2_f06_residual_variance_underflow_prevention`: Asserts that updated residual variance $\sigma_{D, i}^2$ has a strictly positive lower bound ($\ge 10^{-10}$).

#### Feature F7: Classical Latency Benchmark (5 boundary tests)
141. `test_t2_f07_batch_size_1_latency`: Benchmarks single-symbol batch ($B=1$); asserts latency $< 20$ ms on 4 CPU threads.
142. `test_t2_f07_batch_size_256_throughput`: Benchmarks large batch ($B=256$); asserts memory usage remains bounded and latency scales sub-linearly.
143. `test_t2_f07_single_thread_concurrency`: Runs benchmark with `torch.set_num_threads(1)`; asserts stable execution without thread lock.
144. `test_t2_f07_empty_warmup_recovery`: Benchmarks without prior warmup; asserts timing logic cleanly discards first-call compilation overhead.
145. `test_t2_f07_zero_division_timing_guard`: Asserts that timer calculations guard against 0-microsecond elapsed times.

#### Feature F8: Exact $C^*$ Gauge Normalization (5 boundary tests)
146. `test_t2_f08_extreme_attenuation_scale_2m12`: Scales input by $r = 2^{-12} \approx 2.44 \times 10^{-4}$; asserts output LLRs and bit decisions match unscaled input exactly ($\Delta < 10^{-5}$, 0 flips).
147. `test_t2_f08_extreme_amplification_scale_2p12`: Scales input by $r = 2^{+12} = 4096$; asserts output LLRs and bit decisions match unscaled input exactly ($\Delta < 10^{-5}$, 0 flips).
148. `test_t2_f08_zero_signal_power_eps_guard`: Tests input with pilot power $= 0$; asserts gauge normalization utilizes dimensionless relative stabilizer without crashing.
149. `test_t2_f08_pure_phase_rotation_pi`: Multiplies by $e^{j\pi} = -1$; asserts channel estimate satisfies $\hat{H}(-Y) = -\hat{H}(Y)$ and bit decisions are 100% identical.
150. `test_t2_f08_complex_rms_norm_subnormal_guard`: Tests Complex RMSNorm with subnormal complex inputs ($10^{-35}$); asserts finite unit vector output.

#### Feature F9: Hermitian Green's Propagator Attention (5 boundary tests)
151. `test_t2_f09_single_subcarrier_self_attention`: Tests attention on $N=1$ subcarrier; asserts attention weight is identically 1.0.
152. `test_t2_f09_maximum_subcarrier_separation`: Evaluates relative RoPE at maximum distance $|m - n| = N - 1$; asserts unitary phasor $|e^{j(m-n)\theta}| = 1.0$.
153. `test_t2_f09_zero_magnitude_token_radial_activation`: Injects a token with exact zero magnitude ($z=0$); asserts radial activation outputs $0$ without NaN.
154. `test_t2_f09_extreme_magnitude_token_radial_activation`: Injects token with magnitude $10^6$; asserts radial activation does not overflow.
155. `test_t2_f09_identical_keys_uniform_attention`: Sets all keys identical; asserts softmax attention distribution is perfectly uniform ($1/N$).

#### Feature F10: Bidirectional Pilot-Data Attention (5 boundary tests)
156. `test_t2_f10_all_pilot_no_data_slot`: Evaluates block when data tokens are zero; asserts pilot channel representation remains uncorrupted.
157. `test_t2_f10_all_data_minimal_pilot_single_tone`: Evaluates block with only 1 pilot subcarrier; asserts cross-attention propagates channel across all data subcarriers.
158. `test_t2_f10_random_noise_data_tokens`: Feeds pure noise data tokens; asserts bidirectional feedback does not degrade channel MSE by more than 1 dB over pilot-only estimator.
159. `test_t2_f10_perfect_channel_pilot_tokens`: Feeds noiseless pilots; asserts data demapping achieves AWGN lower bound.
160. `test_t2_f10_multi_layer_bidirectional_depth`: Evaluates 1, 2, and 4 stacked bidirectional blocks; asserts numerical stability and finite outputs across all layers.

#### Feature F11: Joint Physical Consistency Loss (5 boundary tests)
161. `test_t2_f11_perfect_predictions_zero_loss`: Feeds ground truth channel and perfect LLRs; asserts $\mathcal{L}_{\text{joint}} \to 0$ ($\le 10^{-4}$).
162. `test_t2_f11_completely_erroneous_predictions_bounded_loss`: Feeds inverted bit signs; asserts BCE loss remains finite (bounded by BCE max).
163. `test_t2_f11_loss_scale_invariance_24_octaves`: Multiplies input by $r \in [2^{-12}, 2^{12}]$; asserts $\mathcal{L}_{\text{joint}}$ is identical within relative tolerance $10^{-5}$.
164. `test_t2_f11_zero_channel_denominator_protection`: Sets $H_{\text{true}} = 0$; asserts NMSE loss utilizes $\epsilon_H$ without division by zero.
165. `test_t2_f11_zero_payload_power_pa_loss_protection`: Sets $Y_d = 0$; asserts PA residual loss remains finite.

#### Feature F12: Numerical Equivariance Verification (5 boundary tests)
166. `test_t2_f12_extreme_gain_boundary_2m12_verification`: Direct verification of $r=2^{-12}$ in pure FP32; asserts 0 bit flips.
167. `test_t2_f12_extreme_gain_boundary_2p12_verification`: Direct verification of $r=2^{+12}$ in pure FP32; asserts 0 bit flips.
168. `test_t2_f12_dense_phase_sweep_100_points`: Tests 100 uniformly spaced phases on $[0, 2\pi)$; asserts max logit deviation $< 10^{-5}$.
169. `test_t2_f12_irrational_phase_angles`: Tests phases $\phi \in \{\pi/7, e, \sqrt{2}, \pi/\sqrt{3}\}$; asserts 0 bit flips.
170. `test_t2_f12_noise_variance_quad_scaling`: Verifies that scaling noise variance by $r^2$ alongside signal scaling by $r$ maintains exact bit LLRs.

#### Feature F13: 2D Slot Simulator & Sparse DMRS (5 boundary tests)
171. `test_t2_f13_single_symbol_slot_boundary`: Tests mini-slot of length $N_{\text{symb}}=1$; asserts simulator handles degenerate slot length gracefully.
172. `test_t2_f13_large_slot_boundary_28_symb`: Tests double-length slot $N_{\text{symb}}=28$; asserts grid construction scales correctly.
173. `test_t2_f13_dense_subcarriers_boundary_128`: Tests subcarrier count $N_{\text{sc}}=128$; asserts pilot overhead and grid mapping remain valid.
174. `test_t2_f13_ultra_sparse_dmrs_single_pilot_re`: Configures minimal possible pilot pattern (1 pilot RE); asserts mask validity.
175. `test_t2_f13_full_dmrs_slot_100_percent`: Configures full pilot slot (100% pilots); asserts payload mask is empty and simulator runs without crash.

#### Feature F14: Continuous Doppler ICI & CFO (5 boundary tests)
176. `test_t2_f14_zero_doppler_zero_cfo_limit`: Configures $f_d=0, \epsilon=0$; asserts channel impulse response is strictly time-invariant across all slot symbols.
177. `test_t2_f14_extreme_cfo_half_subcarrier`: Sets $\epsilon = 0.5$ (maximum ambiguous CFO); asserts demodulation experiences expected severe ICI and attenuation.
178. `test_t2_f14_negative_cfo_phase_roll`: Sets $\epsilon = -0.15$; asserts phase accumulation rotates in reverse direction.
179. `test_t2_f14_extreme_doppler_jakes_spectrum`: Tests Jake's Doppler model with maximum user speed ($v=500$ km/h at $fc=3.5$ GHz); asserts spectrum bounds at $\pm f_d$.
180. `test_t2_f14_phase_continuity_across_cp`: Asserts continuous phase accumulation does not experience artificial discontinuous phase jumps at CP boundaries.

#### Feature F15: Native Outer Channel Coding (5 boundary tests)
181. `test_t2_f15_all_zeros_codeword`: Encodes all-zero message; asserts all-zero codeword and successful decoding at $-10$ dB.
182. `test_t2_f15_all_ones_message`: Encodes all-one message; asserts valid codeword satisfying parity check.
183. `test_t2_f15_single_bit_error_recovery`: Injects exactly 1 flipped bit in codeword; asserts Min-Sum decoder corrects it in 1 iteration.
184. `test_t2_f15_max_iterations_unconverged_syndrome`: Feeds pure random noise LLRs; asserts decoder terminates at $I_{\text{max}}=15$ with non-zero syndrome rather than hanging.
185. `test_t2_f15_empty_payload_guard`: Tests 0-length input; asserts proper exception or graceful boundary handling.

#### Feature F16: 2D Classical Baselines (5 boundary tests)
186. `test_t2_f16_2d_lmmse_zero_doppler_flat_channel`: Tests 2D-LMMSE under static AWGN; asserts interpolated channel is constant across all 14 symbols.
187. `test_t2_f16_2d_lmmse_singular_covariance_regularization`: Tests 2D-LMMSE when pilot SNR is high ($\sigma_W^2 \to 0$); asserts matrix inversion regularizer prevents numerical singularity.
188. `test_t2_f16_2d_idd_pa_linear_pa_boundary`: Tests 2D IDD-PA with $A=\infty$; asserts distortion subtraction term is identically zero.
189. `test_t2_f16_2d_idd_pa_extreme_clipping_convergence`: Tests 2D IDD-PA with $A=1.0$; asserts iteration does not diverge.
190. `test_t2_f16_2d_lmmse_sparse_dmrs_extrapolation`: Verifies channel interpolation accuracy at edge symbols (symbols 0 and 13) where DMRS is absent.

#### Feature F17: Coded BLER Evaluation (5 boundary tests)
191. `test_t2_f17_zero_noise_zero_bler`: Evaluates BLER at 40 dB SNR; asserts $\text{BLER} \equiv 0$ over 500 blocks.
192. `test_t2_f17_extreme_noise_100_percent_bler`: Evaluates BLER at $-15$ dB SNR; asserts $\text{BLER} \equiv 1.0$.
193. `test_t2_f17_single_block_simulation`: Runs evaluation for exactly 1 block; asserts valid integer output ($0$ or $1$).
194. `test_t2_f17_massive_blocks_accounting_overflow_check`: Tests integer accumulator over $10^6$ simulated blocks; asserts Python 64-bit integer integrity without overflow.
195. `test_t2_f17_bler_confidence_interval_computation`: Confirms Clopper-Pearson or Wilson binomial confidence intervals compute valid bounds $[p_{\text{low}}, p_{\text{high}}] \subseteq [0, 1]$.

#### Feature F18: Factorial Ablation Matrix (5 boundary tests)
196. `test_t2_f18_ablation_model_single_sample_batch`: Evaluates forward pass for all 8 arms with batch size $B=1$; asserts correct tensor shapes.
197. `test_t2_f18_ablation_model_large_batch_128`: Evaluates all 8 arms with batch size $B=128$; asserts memory stability.
198. `test_t2_f18_parameter_budget_matching`: Asserts parameter counts of all 8 arms are within $15\%$ of each other ($\approx 25\text{k}-35\text{k}$ parameters).
199. `test_t2_f18_decoupled_arm_zero_feedback`: Verifies that in decoupled arms (Arm 4, 6, 7, 8), data gradients never propagate into the pilot channel estimator.
200. `test_t2_f18_real_attention_arm_loss_of_u1_equivariance`: Proves that Arm 3 (real attention) suffers logit shifts $> 1.0$ under phase rotation $e^{j\pi/4}$, confirming necessity of Green's attention.

#### Feature F19: Numeric Stability & Precision Audit (5 boundary tests)
201. `test_t2_f19_fp16_subnormal_scale_stability`: Evaluates FP16 at $r=2^{-12}$; asserts no subnormal underflow crashes or NaNs.
202. `test_t2_f19_fp16_max_representable_scale`: Evaluates FP16 at $r=2^{+12}$; asserts no overflow to `+inf`.
203. `test_t2_f19_bf16_wide_exponent_stability`: Evaluates BF16 across $r \in [2^{-20}, 2^{+20}]$; asserts uncorrupted forward execution.
204. `test_t2_f19_zero_input_autocast_finite`: Injects all-zero tensor under FP16/BF16 autocast; asserts all outputs are finite numbers.
205. `test_t2_f19_loss_backward_autocast_stability`: Runs backward pass under autocast; asserts parameter gradients contain 0 NaNs or Infs.

#### Feature F20: Pre-Frozen Confirmatory Protocol (5 boundary tests)
206. `test_t2_f20_frozen_manifest_missing_file_detection`: Simulates missing or modified protocol file; asserts validation function raises immediate verification error.
207. `test_t2_f20_seed_list_length_exactly_10`: Asserts seed list has length exactly 10 (not 9, not 11).
208. `test_t2_f20_seed_values_in_declared_range`: Asserts all seeds satisfy $74001 \le s \le 74010$.
209. `test_t2_f20_integer_type_enforcement`: Asserts that passing float error counts to the recorder raises a `TypeError`.
210. `test_t2_f20_empty_results_directory_initialization`: Verifies that confirmatory runner creates clean `p4d_results/` directory without altering existing folders.

#### Feature F21: Sign-Flip Permutation Tests (5 boundary tests)
211. `test_t2_f21_single_observation_degeneracy`: Tests permutation logic with $N=1$ seed; asserts valid handling ($p=1.0$).
212. `test_t2_f21_all_positive_deltas_exact_p_value`: Tests vector with 10 strictly positive values; asserts $p = 2 / 2^{10} = 2 / 1024 \approx 0.001953125$.
213. `test_t2_f21_balanced_alternating_signs_p_value`: Tests vector with 5 positive and 5 negative equal values; asserts $T_{\text{obs}}=0$ and $p=1.0$.
214. `test_t2_f21_holm_adjustment_single_hypothesis`: Tests Holm adjustment with $M=1$; asserts $p^{\text{Holm}} = p$.
215. `test_t2_f21_holm_adjustment_large_m_ceiling`: Tests Holm adjustment with $M=20$ and high $p$-values; asserts adjusted $p$-values never exceed 1.0.

#### Feature F22: Repository Governance & Immutability (5 boundary tests)
216. `test_t2_f22_tampered_byte_detection`: Injects a 1-byte simulated alteration into memory cache of a historical file; asserts hash mismatch triggers immediate failure.
217. `test_t2_f22_new_file_outside_p4_detection`: Asserts that creating files outside `p4/`, `tests/`, or `` is flagged as an immutability violation.
218. `test_t2_f22_historical_manifest_read_only_permissions`: Confirms `MANIFEST_SHA256.json` cannot be opened in write mode by unauthorized test routines.
219. `test_t2_f22_crlf_lf_cross_platform_hash_stability`: Confirms hash computation yields identical cryptographic digest regardless of whether file has Windows `\r\n` or Unix `\n`.
220. `test_t2_f22_results_directory_isolation`: Verifies all test outputs are routed strictly to designated output folders (`p4/results/`, `p4b_results/`, etc.).

---

### 5.3 Tier 3: Cross-Feature Combinations (22 Pairwise Interaction Tests)
Tier 3 tests exercise complex multi-feature interactions where physical impairments and architectural modules interact directly:

221. **Pair 1 (`F1` + `F2`) — Rapp PA + Dynamic Multipath Doppler ICI**:
    - `test_t3_p01_f01_f02_rapp_pa_dynamic_doppler`: Verifies that time-domain PA distortion combined with continuous Doppler phase evolution produces both spectral regrowth and inter-carrier interference without simulator instability.
222. **Pair 2 (`F1` + `F4`) — Rapp PA + Bussgang LMMSE Equalization**:
    - `test_t3_p02_f01_f04_rapp_pa_bussgang_lmmse`: Tests Bussgang-LMMSE equalization on dynamic multipath signal distorted by Rapp PA ($A=1.25$); confirms Bussgang-LMMSE achieves lower symbol MSE than linear LMMSE.
223. **Pair 3 (`F1` + `F6`) — Rapp PA + Iterative DD-PA Cancellation**:
    - `test_t3_p03_f01_f06_rapp_pa_idd_cancellation`: Verifies 2-iteration DD-PA cancellation on Rapp-distorted transmission; confirms successive reduction of nonlinear residual energy.
224. **Pair 4 (`F2` + `F14`) — Dynamic Multipath + Continuous CFO Phase Drift**:
    - `test_t3_p04_f02_f14_multipath_continuous_cfo`: Verifies joint simulation of continuous multipath fading and carrier frequency offset phase accumulation $e^{j 2\pi \epsilon t / N}$.
225. **Pair 5 (`F4` + `F5`) — Bussgang LMMSE + Calibrated 16-QAM Soft LLRs**:
    - `test_t3_p05_f04_f05_bussgang_calibrated_llrs`: Confirms soft LLRs derived from Bussgang conditional mean $\mu$ and residual error variance $\sigma_v^2$ produce calibrated bit probabilities.
226. **Pair 6 (`F5` + `F15`) — Calibrated Soft LLRs + Native Outer LDPC Decoder**:
    - `test_t3_p06_f05_f15_calibrated_llrs_outer_ldpc`: Feeds calibrated soft LLRs into native PyTorch Rate-1/2 LDPC Min-Sum decoder; verifies successful decoding and waterfall gain under nonlinear distortion.
227. **Pair 7 (`F6` + `F16`) — Iterative DD-PA + 2D Classical Baseline**:
    - `test_t3_p07_f06_f16_2d_idd_pa_cancellation`: Verifies 2D time-frequency iterative decision-directed PA cancellation utilizing tentative payload decisions across the 2D slot grid.
228. **Pair 8 (`F8` + `F9`) — Exact $C^*$ Gauge Normalization + Hermitian Green's Attention**:
    - `test_t3_p08_f08_f09_gauge_norm_hermitian_attention`: Verifies that combining pilot RMS gauge anchoring with Hermitian relative-phase attention achieves strict scale-and-phase equivariance across the network.
229. **Pair 9 (`F8` + `F12`) — Exact $C^*$ Gauge + Numerical Equivariance Verification**:
    - `test_t3_p09_f08_f12_gauge_norm_numerical_equivariance`: Evaluates the integrated EM-FloatFormer model across $r \in [2^{-12}, 2^{12}]$ and $\phi \in [0, 2\pi)$; verifies zero bit flips and max logit error $< 10^{-5}$.
230. **Pair 10 (`F8` + `F19`) — Exact $C^*$ Gauge + FP16/BF16 Mixed Precision**:
    - `test_t3_p10_f08_f19_gauge_norm_mixed_precision_audit`: Tests $C^*$ gauge normalization under FP16 and BF16 autocast across 24 octaves; asserts no overflow or subnormal underflow crashes.
231. **Pair 11 (`F9` + `F10`) — Hermitian Green's Attention + Bidirectional Pilot-Data Attention**:
    - `test_t3_p11_f09_f10_green_attention_bidirectional_prop`: Verifies bidirectional cross-attention using Hermitian inner products and relative RoPE; confirms mutual pilot-data distortion exchange.
232. **Pair 12 (`F9` + `F18`) — Hermitian Green's Attention + Factorial Ablation Architecture**:
    - `test_t3_p12_f09_f18_green_attention_factorial_ablation`: Verifies ablation toggling of Green's attention against standard real attention in parameter-matched configurations.
233. **Pair 13 (`F10` + `F11`) — Bidirectional Attention + Joint Physical Consistency Loss**:
    - `test_t3_p13_f10_f11_bidirectional_joint_loss_convergence`: Trains 1 step of bidirectional model with joint loss ($\mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{NMSE}} + \mathcal{L}_{\text{PA-res}}$); verifies non-zero gradients on all parameters.
234. **Pair 14 (`F11` + `F12`) — Joint Physical Consistency Loss + Scale Equivariance of Gradients**:
    - `test_t3_p14_f11_f12_joint_loss_gradient_equivariance`: Verifies that parameter gradients $\nabla_\theta \mathcal{L}$ remain identical when inputs are scaled by $r \in [2^{-12}, 2^{12}]$.
235. **Pair 15 (`F13` + `F14`) — 2D Slot Simulator + Continuous High-Doppler & CFO**:
    - `test_t3_p15_f13_f14_2d_slot_doppler_cfo`: Simulates 14-symbol slot with sparse DMRS under high Doppler ($f_d T_{\text{sym}} = 0.05$) and continuous CFO ($\epsilon = 0.05$).
236. **Pair 16 (`F13` + `F16`) — 2D Slot Simulator + 2D Classical LMMSE Baseline**:
    - `test_t3_p16_f13_f16_2d_slot_2d_lmmse_estimation`: Performs 2D-LMMSE channel estimation from sparse DMRS REs (comb-2 on symbols 2 & 11) over the full 14-symbol slot grid.
237. **Pair 17 (`F14` + `F17`) — Continuous High-Doppler + Coded BLER Evaluation**:
    - `test_t3_p17_f14_f17_high_doppler_coded_bler`: Evaluates outer coded BLER under high Doppler ($f_d = 600$ Hz) and verifies graceful degradation over low Doppler.
238. **Pair 18 (`F15` + `F17`) — Native Outer Channel Coding + Coded BLER Waterfall**:
    - `test_t3_p18_f15_f17_outer_coding_bler_waterfall`: Traces full coded BLER curve across SNR grid $[0, 5, 10, 15, 20]$ dB, demonstrating steep waterfall characteristic.
239. **Pair 19 (`F16` + `F17`) — 2D Baselines + Out-of-Distribution Multipath BLER**:
    - `test_t3_p19_f16_f17_2d_baselines_ood_multipath_bler`: Benchmarks 2D-LMMSE and 2D IDD-PA BLER on held-out `sparse12` channel profile.
240. **Pair 20 (`F18` + `F20`) — Factorial Ablation Matrix + Pre-Frozen Confirmatory Seeds**:
    - `test_t3_p20_f18_f20_factorial_ablation_confirmatory_seeds`: Runs all 8 ablation arms across the 10 pre-frozen seeds $\mathcal{S}_{\text{P4D}}$ with paired initializations.
241. **Pair 21 (`F20` + `F21`) — Integer Accounting + Sign-Flip Permutation Tests**:
    - `test_t3_p21_f20_f21_integer_accounting_permutation_tests`: Feeds exact integer error records into the 1024-state exact permutation test runner; computes exact two-sided $p$-values.
242. **Pair 22 (`F21` + `F22`) — Holm Step-Down Adjustment + Repository Governance**:
    - `test_t3_p22_f21_f22_holm_adjustment_governance_audit`: Verifies Holm-adjusted $p$-value table is logged into dedicated report without modifying historical manifests.

---

### 5.4 Tier 4: Real-World Workload Scenarios (11 Full End-to-End Scenarios)
Tier 4 exercises complete, mission-profile communication workloads combining simulation, channel distortion, transceiver processing, decoding, and statistical auditing:

243. **Scenario 1 — High-Speed Train Channel (HST-Doppler with Severe PA Distortion)**:
    - `test_t4_s01_high_speed_train_workload`:
      - Configuration: $N_{\text{symb}}=14, N_{\text{sc}}=32$, 16-QAM, vehicle speed $v = 300$ km/h ($f_d = 500$ Hz at $f_c=3.0$ GHz), Rapp PA with $A=1.25$ and AM-PM phase rotation ($\alpha=0.5, \beta=1.0$), DMRS comb-2 on symbols 2 & 11, Rate-1/2 LDPC soft decoding.
      - Execution: Full TX generation $\to$ dynamic multipath $\to$ RX front-end $\to$ EM-FloatFormer forward pass $\to$ soft LLR demapping $\to$ LDPC decoding.
      - Invariants: Verified BLER $< 5 \times 10^{-2}$ at 15 dB SNR; zero non-finite outputs; exact integer error accounting.
244. **Scenario 2 — Urban Macrocell with Severe Multipath Delay Spread**:
    - `test_t4_s02_urban_macro_dense_multipath_workload`:
      - Configuration: $N_{\text{sc}}=64, N_{\text{cp}}=16$, 12-tap exponential power delay profile (`long12` and `sparse12`), mild PA ($A=2.25$), $14$-symbol slot, DMRS overhead $\le 7.5\%$.
      - Execution: Transmit 20 slots; evaluate channel estimation NMSE and coded BLER.
      - Invariants: NMSE $< 0.05$ ($-13$ dB); coded BLER $< 10^{-2}$ at 18 dB SNR.
245. **Scenario 3 — Extreme Local Oscillator Drift & Continuous Carrier Frequency Offset (CFO)**:
    - `test_t4_s03_extreme_cfo_drift_workload`:
      - Configuration: Normalized CFO $\epsilon = \Delta f / \Delta f_{\text{sub}} = 0.06$, continuous sample-level phase accumulation across 14 OFDM symbols ($N_{\text{total}} = 14 \times (32 + 16) = 672$ samples), Doppler $f_d = 150$ Hz.
      - Execution: Test CP-correlation CFO tracking, baseband de-rotation, and EM-FloatFormer symbol recovery.
      - Invariants: Residual constellation phase error $< 3^\circ$; zero quadrant flips at 20 dB SNR.
246. **Scenario 4 — Deep Shadowing & 24-Octave Dynamic Range Swing**:
    - `test_t4_s04_dynamic_range_shadowing_swing_workload`:
      - Configuration: Slot-to-slot received signal power varies dynamically across 24 octaves: $r \in \{2^{-12}, 2^{-8}, 2^{-4}, 2^0, 2^4, 2^8, 2^{12}\}$, combined with additive noise variance scaling $r^2 \sigma_W^2$.
      - Execution: Stream slots through model under BF16 mixed precision.
      - Invariants: Bit decision flip rate $\equiv 0\%$; max logit absolute deviation $< 10^{-5}$ relative to normalized reference; zero NaNs/Infs.
247. **Scenario 5 — Ultra-Sparse Pilot Deployment ($\le 5\%$ Overhead)**:
    - `test_t4_s05_ultra_sparse_dmrs_deployment_workload`:
      - Configuration: Single DMRS symbol at symbol 2 with comb-4 pattern (only 8 pilot subcarriers out of $14 \times 32 = 448$ REs, overhead $\approx 1.78\%$).
      - Execution: Test joint pilot-data bidirectional propagation to interpolate and extrapolate channel across all 14 symbols.
      - Invariants: Model maintains stable channel estimates without divergence; outperforms 1D LS baseline by $\ge 4$ dB NMSE.
248. **Scenario 6 — Multi-Impairment Simultaneous Stress Workload**:
    - `test_t4_s06_multi_impairment_stress_workload`:
      - Configuration: Concurrent activation of (i) Rapp AM-AM ($A=1.25$), (ii) Rapp AM-PM ($\alpha=0.5$), (iii) Doppler ICI ($f_d T_{\text{sym}} = 0.03$), (iv) CFO drift ($\epsilon = 0.04$), and (v) multipath delay spread (`long6`).
      - Execution: End-to-end evaluation comparing EM-FloatFormer against 2D-LMMSE and 2D IDD-PA.
      - Invariants: EM-FloatFormer achieves statistically significant BLER reduction ($p < 0.05$); all outputs finite.
249. **Scenario 7 — Out-of-Distribution Channel Delay Profile Generalization**:
    - `test_t4_s07_ood_channel_generalization_workload`:
      - Configuration: Model trained strictly on 4-tap equal-power channels (`equal4`); evaluated on unseen 16-tap exponentially decaying channel with fractional delays.
      - Execution: Evaluate uncoded BER and channel NMSE without fine-tuning.
      - Invariants: Absence of coordinate shortcut collapse (BER does not exceed classical DFT-8 baseline by more than $0.5\%$).
250. **Scenario 8 — Confirmatory 10-Seed Pre-Frozen Replication Workload**:
    - `test_t4_s08_confirmatory_ten_seed_replication_workload`:
      - Configuration: Full replication protocol executed across all 10 pre-frozen seeds $\mathcal{S}_{\text{P4D}} = \{74001, \dots, 74010\}$.
      - Execution: Run seed loop, record exact integer bit errors and block errors, compute paired contrasts against 2D-LMMSE baseline.
      - Invariants: Total tested bits recorded as integer; exact 1024-state permutation test runs and outputs two-sided $p$-value with Holm adjustment.
251. **Scenario 9 — Full $2^3$ Factorial Ablation Matrix Evaluation Workload**:
    - `test_t4_s09_factorial_ablation_matrix_workload`:
      - Configuration: Evaluate all 8 parameter-matched architecture variants (Arms 1 through 8) across identical channel realizations at 10 dB SNR.
      - Execution: Decompose performance via Yates' contrast matrix into main effects of $C^*$ gauge, Green's attention, and joint propagation, alongside 2-way interactions.
      - Invariants: Contrast sum verifies linear decomposition; $C^*$ gauge main effect demonstrates statistically significant gain.
252. **Scenario 10 — Full-Precision vs Mixed-Precision Numerical Audit Workload**:
    - `test_t4_s10_mixed_precision_audit_workload`:
      - Configuration: Run identical batch of 100 slots through FP32, BF16, and FP16 execution engines across SNR grid $[0, 5, 10, 15, 20]$ dB.
      - Execution: Measure logit max absolute error, hard decision flip count, and throughput (slots/sec).
      - Invariants: Zero NaNs/Infs; decision flip rate $< 0.05\%$; BF16 execution speedup $\ge 1.2\times$ on supported hardware.
253. **Scenario 11 — End-to-End Mission-Critical Coded Transmission Workload**:
    - `test_t4_s11_mission_critical_coded_transmission_workload`:
      - Configuration: Complete operational communication link transmitting $10^5$ information bits across 50 consecutive 14-symbol slots under 16-QAM, Rapp PA ($A=1.25$), high Doppler ($f_d = 300$ Hz), sparse DMRS, and outer LDPC decoding.
      - Execution: Full end-to-end pipeline execution from information bit generation to verified decoded payload.
      - Invariants: Zero unhandled exceptions; $\text{BLER} < 10^{-2}$ at 15 dB; exact bit error count matches integer sum; exit code 0.

---

## 6. Verification Mechanisms & Non-Trivial Assertions

To guarantee that tests provide genuine verification rather than shallow execution checks, all assertions are grounded in physical, mathematical, and algorithmic invariants:

### 6.1 Mathematical Invariant Formulations
1. **$C^*$ Gauge Equivariance Invariant**:
   For any complex scalar $c = r e^{j\phi} \in \mathbb{C}^*$ with $r \in [2^{-12}, 2^{12}]$ and $\phi \in [0, 2\pi)$:
   $$\max_{b, k, i} \left| \text{LLR}_{b, k, i}(c Y_p, c Y_d, |c|^2 \sigma_W^2) - \text{LLR}_{b, k, i}(Y_p, Y_d, \sigma_W^2) \right| < 10^{-5}$$
   $$\sum_{b, k, i} \mathbb{I}\left( \text{sign}\left(\text{LLR}(c Y)\right) \ne \text{sign}\left(\text{LLR}(Y)\right) \right) \equiv 0$$
2. **Channel Equivariance Invariant**:
   $$\frac{\left\| \hat{H}(c Y_p, c Y_d, |c|^2 \sigma_W^2) - c \hat{H}(Y_p, Y_d, \sigma_W^2) \right\|_2}{\|c \hat{H}(Y_p, Y_d, \sigma_W^2)\|_2 + 10^{-12}} < 10^{-5}$$
3. **Static Limit Parity Invariant**:
   When $f_d = 0$ and $A \to \infty$:
   $$\frac{\left\| Y_{\text{sim}} - (H \odot X) \right\|_\infty}{\|H \odot X\|_\infty} < 10^{-5}$$
4. **Bussgang Orthogonality Invariant**:
   For PA output $y = f(x) = \gamma x + d$:
   $$\left| \frac{1}{M} \sum_{m=1}^M d[m] x^*[m] \right| < 10^{-3} \cdot \sigma_x^2$$
5. **Permutation Test Invariant**:
   For paired seed differences $\delta_s$ ($s=1,\dots,10$):
   $$p = \frac{1}{1024} \sum_{k=0}^{1023} \mathbb{I}\left( \left| \sum_{s=1}^{10} (-1)^{b_k(s)} \delta_s \right| \ge \left| \sum_{s=1}^{10} \delta_s \right| - 10^{-10} \right)$$
   Exact rational value with denominator 1024.
6. **Cryptographic Immutability Invariant**:
   For every historical file $f \in \text{Manifest}$:
   $$\text{SHA256}(\text{bytes}(f)) \equiv \text{Manifest}[f]$$

---

## 7. Concrete Implementation Blueprint for the Test Writer

### 7.1 Architecture of `tests/test_p4_e2e.py`
The master suite entry point aggregates all 4 tiers and provides a clean CLI entry point:

```python
"""
Master E2E Test Suite Aggregator for EM-FloatFormer Stage P4.
Covers 253 test cases across Tiers 1-4 and Features F1-F22.
"""
import unittest
import sys

def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    # Discover all tier submodules under tests/e2e/
    from tests.e2e import (
        test_tier1_feature_coverage,
        test_tier2_boundary_corner,
        test_tier3_cross_feature,
        test_tier4_real_workloads,
    )
    for mod in [
        test_tier1_feature_coverage,
        test_tier2_boundary_corner,
        test_tier3_cross_feature,
        test_tier4_real_workloads,
    ]:
        suite.addTests(loader.loadTestsFromModule(mod))
    return suite

if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(load_tests(unittest.defaultTestLoader, None, None))
    sys.exit(0 if result.wasSuccessful() else 1)
```

### 7.2 Structure of `tests/e2e/harness.py`
The test harness centralizes shared constants, independent signal synthesizers, and safe milestone resolution:

```python
"""
Independent E2E Test Harness: Reference Signal Generators, Physical Constants & Resolvers.
"""
import torch
import numpy as np
from typing import Tuple, Dict, Any, Optional

# Reference 16-QAM Gray constellation mapping normalized to unit energy
QAM16_POINTS = torch.tensor([
    -3 - 3j, -3 - 1j, -3 + 3j, -3 + 1j,
    -1 - 3j, -1 - 1j, -1 + 3j, -1 + 1j,
    +3 - 3j, +3 - 1j, +3 + 3j, +3 + 1j,
    +1 - 3j, +1 - 1j, +1 + 3j, +1 + 1j,
], dtype=torch.complex64) / np.sqrt(10.0)

# Bit mappings [b0, b1, b2, b3] for each of the 16 constellation symbols
QAM16_BITS = torch.tensor([
    [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1],
    [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1],
    [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1],
    [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1],
], dtype=torch.int64)

# 10 Pre-Frozen Confirmatory Seeds
FROZEN_SEEDS_P4D = [74001, 74002, 74003, 74004, 74005, 74006, 74007, 74008, 74009, 74010]

def generate_random_bits(batch_size: int, num_bits: int, seed: Optional[int] = None) -> torch.Tensor:
    if seed is not None:
        g = torch.Generator().manual_seed(seed)
        return torch.randint(0, 2, (batch_size, num_bits), dtype=torch.int64, generator=g)
    return torch.randint(0, 2, (batch_size, num_bits), dtype=torch.int64)

def map_bits_to_16qam(bits: torch.Tensor) -> torch.Tensor:
    """Independent reference Gray 16-QAM modulator."""
    B, total_bits = bits.shape
    assert total_bits % 4 == 0
    symbols_count = total_bits // 4
    bits_reshaped = bits.view(B, symbols_count, 4)
    # Integer value from bits
    idx = bits_reshaped[..., 0] * 8 + bits_reshaped[..., 1] * 4 + bits_reshaped[..., 2] * 2 + bits_reshaped[..., 3]
    return QAM16_POINTS[idx]
```

### 7.3 Guidelines for the Test Writer
1. **Module Segregation**:
   - `test_tier1_feature_coverage.py`: Implements exactly 110 test methods (`test_t1_f01_01` to `test_t1_f22_05`).
   - `test_tier2_boundary_corner.py`: Implements exactly 110 test methods (`test_t2_f01_01` to `test_t2_f22_05`).
   - `test_tier3_cross_feature.py`: Implements exactly 22 test methods (`test_t3_p01_...` to `test_t3_p22_...`).
   - `test_tier4_real_workloads.py`: Implements exactly 11 test methods (`test_t4_s01_...` to `test_t4_s11_...`).
2. **Total Test Count Guarantee**:
   - The test writer must verify that running `python -m unittest tests/test_p4_e2e.py` registers exactly **253 tests**.
3. **Progressive Execution Compliance**:
   - During Milestone 1 development, running the suite will execute M1 tests (F1–F7) and gracefully skip M2–M4 tests with informative messages.
   - Zero test failures or exceptions should be observed on supported modules.

---

## 8. Requirements for `TEST_INFRA.md` (Project Root)

To document this test framework comprehensively at the root of the project repository, `TEST_INFRA.md` must be created containing:
1. **Test Infrastructure Overview**:
   - Explanation of the 4-tier testing hierarchy (Feature Coverage, Boundaries & Corners, Cross-Feature Combinations, Real-World Workloads).
   - Traceability table mapping each requirement (R1, R2, R3, R4) and feature (F1 to F22) to test methods.
2. **Execution Guide & CLI Commands**:
   - Commands for full test execution, tier-by-tier execution, and single-feature filtering.
   - Environment configurations (`torch.set_num_threads(4)`, seed controls).
3. **Progressive Testability Protocol**:
   - How test skips work during milestone progression.
   - Criteria for final sign-off (0 failures, 0 skips, 253 passing tests).
4. **Physical Invariants & Assertion Catalog**:
   - Reference table of mathematical laws and formulas enforced by the assertions.
5. **Governance & Audit Trail**:
   - Verification procedures for `MANIFEST_SHA256.json`.
   - Protocol freeze integrity verification.

---

## 9. Conclusion & Handoff Readiness
This blueprint provides the complete architectural foundation, mathematical specifications, invariant formulations, and test case inventory required to implement the 253-test E2E suite for EM-FloatFormer Stage P4. The test framework satisfies all requirements of R1–R4, enforces strict opaque-box independence, guarantees historical immutability, and provides progressive milestone adaptability for the Test Writer.
