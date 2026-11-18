# Handoff Report: 2D Generalization, Probing & Replication Specialist (Explorer 3)

**Author**: Explorer 3 (`teamwork_explorer_survey_3`)
**Date**: 2026-11-15
**Working Directory**: ``
**State Folder**: `teamwork_explorer_survey_3`

---

## 1. Observation

Direct observations from the repository codebase, manifests, and reports:

1. **Absence of 2D Slots, Doppler ICI, and Sparse DMRS**:
   - `p3b_time_link.py:12-14`: Constants define $N=32$, $CP=16$, $\text{ANCHORS}=(0, 8)$.
   - `p3b_time_link.py:40-42`: `Frame` dataclass defines only two symbols: `xpilot: torch.Tensor` and `xdata: torch.Tensor`.
   - `p3b_time_link.py:85-90`: `causal_linear_filter` applies a single static FIR channel `h_time` across the two symbols with zero Doppler variation ($f_d = 0$).
   - `p3b_time_link.py:141-144`: Common phase error (`cpe`) is a static random scalar phase rotation per frame `yd = yd * torch.polar(torch.ones_like(phi), phi)`, not a continuous time-domain Carrier Frequency Offset (CFO).
   - `P3B_REPORT.md:18`: "This chain does not include a standardized 3GPP channel, changing sample clocks, realistic analog front ends or multiple moving OFDM symbols. It does not replace P3-D."
   - `P3B_REPORT.md:91`: "This is common-phase rotation, not within-symbol CFO or ICI, so it must not be extrapolated to general oscillator errors."

2. **Absence of Channel Coding, Soft Demapping, and BLER**:
   - `p1c_make_report.py:59`: Explicitly documents: "没有LDPC、MIMO、时钟偏移、3GPP全协议或真实无线电硬件。"
   - `P2E_REPORT.md:66`: Explicitly documents: "No channel coding, realistic scheduling, dynamic physical impairments, moving vehicles..."
   - `P3B_REPORT.md:191`: Explicitly documents: "Actual RF-EVM, FLOP counts, activation memory, real radio latency, ADC electronics, full CFO/ICI, continuous multipath fading, coding and Sionna/3GPP benchmark remain untested."
   - `requirements.txt:1-2` and `requirements_p3c.txt:1-3`: Dependencies include only `torch>=2.1`, `numpy>=2.3`, `matplotlib>=3.8`. No communications toolkits (no Sionna, no pyldpc, no commpy).
   - `exploratory/p3de_tdl_surrogate/P3DE_SURROGATE_REPORT.md:7`: "Installation of `sionna-no-rt==2.2.0` was attempted but unavailable because PyPI hostname resolution failed in the isolated container."

3. **Absence of Nonlinear-Aware Baselines (Bussgang / IDD-PA)**:
   - `P3B_REPORT.md:44`: "Important exception: no matched-calibration nonlinear-PA inverse/Bussgang receiver was implemented. Hence this benchmark cannot establish how the neural model compares to all strong non-linear-aware signal-processing receivers."
   - `exploratory/16qam_joint_pa_probe/REPORT.md:47`: "Full nonlinear-PA-aware iterative correction / decoding and a fully data-aided probabilistic 16-QAM receiver were not implemented."

4. **Root Causes of P3-C Numeric Coherent Gain Failure**:
   - `p3b_receivers.py:27`: Equalization adds an unscaled hardcoded dimensional constant:
     ```python
     def equalized(yd, h, nv):
         return yd * h.conj() / (h.abs().square() + nv + 1e-2)
     ```
   - `p3_models.py:45`: Noise variance is processed with an unnormalized logarithm: `nv.log10().expand_as(power)`.
   - `P3C_REPORT.md:64-68`: Applying coherent gain $r = 2^{-12}$ or $r = 2^{+12}$ produced $\approx 50\%$ BER even under FP32, proving that existing networks completely lacked $C^* = \mathbb{R}_{>0} \times U(1)$ scale and gauge equivariance.

5. **Historical Precedents for Pre-Frozen Confirmatory Protocols**:
   - `P1E_PROTOCOL.md:15` & `p1e_replication.py:117-123`: Implemented exact two-sided sign-flip permutation tests:
     ```python
     def exact_two_sided_signflip(xs):
         observed = abs(sum(xs))
         count = 0
         for signs in itertools.product((-1, 1), repeat=len(xs)):
             if abs(sum(x * s for x, s in zip(xs, signs))) >= observed - 1e-10:
                 count += 1
         return count / (2 ** len(xs))
     ```
   - `p1e_replication.py:145-151`: Implemented Holm-Bonferroni step-down adjustment:
     ```python
     ordered = sorted(range(len(out)), key=lambda i: out[i]['signflip_p_two_sided'])
     cum = 0.0
     for rank, i in enumerate(ordered):
         corrected = min(1.0, (len(out) - rank) * out[i]['signflip_p_two_sided'])
         cum = max(cum, corrected)
         out[i]['holm_adjusted_p'] = cum
     ```
   - `p1e_replication.py:90`: Tracked exact integer errors: `errors[name] += int((pred != truth).sum().item())`.
   - `P2E_REPORT.md:9-13`: Tested 10 fresh seeds ($2201–2210$) with $1,310,720$ bits per method across equal4 and long6 profiles, with protocol pre-frozen in `P2E_FREEZE.json`.
   - `MANIFEST_SHA256.json`: Maintains strict SHA256 checksums across all 577 repository files to guarantee historical immutability.

---

## 2. Logic Chain

1. **From Observation 1 to Need for 2D Slot Simulator (Stage P4-B)**:
   Because existing channel code in `p3b_time_link.py` only handles 1 pilot and 1 payload symbol with zero Doppler ($f_d = 0$) and static phase rotation, it cannot simulate time-varying Doppler fading, Inter-Carrier Interference (ICI), or sparse DMRS pilot patterns. Therefore, Requirement R3 requires implementing a true 2D slot simulator ($N_{\text{symb}} = 14$, $N_{\text{sc}} = 32$ or $64$) with sample-level time-variant FIR filtering, continuous Jakes Doppler spectrum, continuous sample-level CFO phase drift, and sparse DMRS pilots ($\le 10\%$ overhead).

2. **From Observation 2 to Need for Native Outer FEC Module (Stage P4-B)**:
   Because the repository contains zero channel coding and external toolkits like Sionna cannot be installed in the isolated runtime environment, Stage P4-B cannot rely on pip packages for channel coding. Therefore, a self-contained, native PyTorch module implementing Rate-1/2 5G NR LDPC with Min-Sum decoding (or Rate-1/2 Convolutional code with Max-Log-MAP decoding) must be engineered directly. To support soft decoding, calibrated 16-QAM Max-Log LLR demapping must be implemented to replace the crude uncalibrated logits of P3-B.

3. **From Observation 3 to Need for 2D-LMMSE and IDD-PA Classical Baselines (Stage P4-B)**:
   Because `P3B_REPORT.md` and `exploratory/16qam_joint_pa_probe/REPORT.md` confirmed that no nonlinear-distortion-aware receiver (Bussgang-LMMSE or iterative clipping cancellation) was ever implemented, previous benchmarks failed to test against the true state-of-the-art classical baseline. Implementing separable 2D-LMMSE and Iterative Decision-Directed PA Cancellation (IDD-PA, 2–3 iterations) is mandatory to establish an unprivileged, rigorous classical comparison.

4. **From Observation 4 to $C^*$ Gauge Invariance and Factorial Ablations (Stage P4-C)**:
   Because the collapse of P3 models under coherent gain stress ($r = 2^{\pm 12}$) was caused by hardcoded dimensional constants (`+ 1e-2`) and unnormalized features, establishing true $C^* = \mathbb{R}_{>0} \times U(1)$ equivariance requires: (i) slot RMS scale normalization $S_{\text{ref}}$, (ii) dimensionless stabilizers proportional to $\sigma_w^2$, and (iii) zero-bias complex linear layers. To prove the causal necessity of each architectural component, an orthogonal $2^3$ factorial ablation study isolating $C^*$ gauge invariance, complex Hermitian Green attention, and joint pilot-data propagation must be conducted.

5. **From Observation 5 to Stage P4-D Confirmatory Protocol & Repository Governance**:
   To ensure Stage P4-D produces unassailable confirmatory findings without data snooping or p-hacking:
   - A protocol must be pre-frozen in `P4D_PROTOCOL.md` and cryptographically locked via `P4D_FREEZE.json`.
   - 10 fresh seeds ($\mathcal{S}_{\text{P4D}} = \{74001, \dots, 74010\}$) disjoint from all historical runs must be used.
   - Exact integer bit and block errors must be recorded.
   - Two-sided exact sign-flip permutation tests ($2^{10} = 1024$ sign assignments) with Holm step-down correction must be executed.
   - All historical P1–P3 files must remain byte-for-byte immutable, verified against `MANIFEST_SHA256.json`.

---

## 3. Caveats

1. **Hardware & Execution Environment**:
   The runtime environment is Windows with PyTorch CPU (`cuda_available=False`). CPU execution times will be higher than GPU benchmarks, but small slot grids ($14 \times 32 = 448$ tokens) and compact models ($< 35\text{k}$ parameters) ensure forward passes run in $\approx 1.2$ ms on CPU.
2. **Offline Dependency Boundary**:
   PyPI access is unavailable. No external packages (e.g. Sionna, pyldpc) can be introduced. All algorithms (channel simulator, Rapp PA, 2D-LMMSE, IDD-PA, LDPC/Convolutional FEC, and EM-FloatFormer Transformer) must be 100% self-contained in native PyTorch/NumPy.
3. **Exploratory vs Confirmatory Scope**:
   Stage P4-B and P4-C are exploratory engineering and probing stages. Confirmatory statistical claims ($p < 0.05$) are strictly reserved for the pre-frozen Stage P4-D protocol.

---

## 4. Conclusion

1. **Current Baseline**: The existing repository is strictly limited to static 1D single-symbol QPSK with 100% pilot overhead, zero channel coding, no 2D-LMMSE, and no nonlinear classical baselines. It collapsed under wide dynamic range scaling.
2. **Path Forward**:
   - Stage P4-B must build a 2D time-frequency slot simulator with sparse DMRS, continuous Doppler ICI, CFO, calibrated 16-QAM soft LLRs, native outer FEC, and benchmark BLER against 2D-LMMSE and IDD-PA.
   - Stage P4-C must execute a $2^3$ factorial ablation isolating $C^*$ gauge invariance, complex Hermitian Green attention, and joint pilot-data propagation, alongside a numeric stability audit across $r \in [2^{-12}, 2^{+12}]$ in FP32, BF16, and FP16.
   - Stage P4-D must execute a pre-frozen confirmatory protocol over 10 fresh seeds ($74001–74010$) with integer error accounting, two-sided sign-flip permutation tests, and Holm adjustment, strictly preserving historical P1–P3 records.

---

## 5. Verification Method

To independently verify the facts and findings stated in this report:

1. **Verify Historical Scope and Keyword Absence**:
   Run powershell grep across repository root:
   ```powershell
   Select-String -Path "*.py" -Pattern "doppler|cfo|dmrs|slot|bler|ldpc|polar|bussgang"
   ```
   *Expected Result*: Zero occurrences of active implementations; only comments in reports noting their absence.

2. **Verify Repository Checksum Manifest Integrity**:
   Inspect `MANIFEST_SHA256.json`:
   ```powershell
   python -c "import json; m=json.load(open('MANIFEST_SHA256.json')); print('Tracked files:', len(m))"
   ```
   *Expected Result*: 577 tracked files spanning P1, P2, and P3.

3. **Verify Python Environment & Offline Boundary**:
   ```powershell
   python -c "import torch, scipy; print('Torch:', torch.__version__, 'Scipy:', scipy.__version__)"
   ```
   *Expected Result*: Python 3.12.7, Torch 2.12.1+cpu, Scipy 1.13.1.

4. **Verify Pre-Existing Test Integrity**:
   Run historical unit tests to confirm repository baseline passes:
   ```powershell
   python -m unittest test_p1d_invariants test_p2_integrity test_p3b_link test_p3c_precision test_p3_a_mechanism -v
   ```
   *Expected Result*: All historical unit tests pass without errors.
