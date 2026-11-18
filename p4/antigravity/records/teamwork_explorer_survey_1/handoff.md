# Handoff Report: Survey of Historical Baselines, Channel Simulators, and Technical Gaps for Stage P4-0

**Agent:** Explorer 1 (Historical Baseline & Simulator Specialist, `teamwork_explorer_survey_1`)
**Timestamp:** 2026-11-15T23:02:00Z
**Type:** Hard Handoff (Survey Phase Complete)
**Location:** `teamwork_explorer_survey_1\handoff.md`

---

## 1. Observation

1. **Host Execution Environment**:
   - Command: `python -c "import sys, torch; print('Python:', sys.version); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device count:', torch.cuda.device_count() if torch.cuda.is_available() else 0); print('Num threads:', torch.get_num_threads())"`
   - Output:
     ```text
     Python: 3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 13:17:27) [MSC v.1929 64 bit (AMD64)]
     PyTorch: 2.12.1+cpu
     CUDA available: False
     Device count: 0
     Num threads: 10
     ```
   - Standard CPU thread limit for benchmarks in historical code: `torch.set_num_threads(4)` (observed in `p3b_timing_extended.py:22` and `exploratory/16qam_joint_pa_probe/timing.py:7`).

2. **Historical Governance and File Immutability**:
   - `MANIFEST_SHA256.json` (lines 1–50) tracks 115 files spanning P1-A through P2-E.
   - P2 Supplement added `P2_SUPPLEMENT_MANIFEST_SHA256.json` and `test_p2_supplement_independent.py`.
   - P3-A2 added `P3_A2_PUBLIC_SHA256_MANIFEST.json`.
   - `test_p2_integrity.py` line 20 checks byte SHA256 of `P2_PROTOCOL.md` against `'935ac73e1cf56a2e9c95f51613a883d4ebf05d15eb672a3ec14dd5b189241342'`.
   - Line endings on Windows: `(ROOT / 'P2_PROTOCOL.md').read_bytes()` contains CRLF `\r\n`, which altered raw byte SHA256 when checked out with default git `core.autocrlf`.

3. **Existing Channel Simulator & PA Implementation**:
   - `p3b_time_link.py` line 80–83:
     ```python
     def tx_cp(x):
         time=torch.fft.ifft(x,dim=-1)
         return torch.cat((time[:,-CP:],time),dim=-1)
     ```
   - `p3b_time_link.py` line 85–90:
     ```python
     def causal_linear_filter(x,h):
         # Real FIR convolution, not frequency-bin-wise multiplication.
         out=torch.zeros_like(x)
         for delay in range(h.shape[-1]):
             out[:,delay:] += h[:,delay,None]*x[:,:x.shape[-1]-delay]
         return out
     ```
   - `p3b_time_link.py` line 93–96:
     ```python
     def rapp(x, severity):
         assert severity in LEVELS
         A=(2.0 if severity=='mild' else 1.2)/math.sqrt(N)
         return x / (1+(x.abs()/A)**4).pow(.25)
     ```
   - `exploratory/16qam_joint_pa_probe/probe.py` line 34–38:
     ```python
     def pa_magnitude(z,condition):
         if condition=='clean':return z
         # input RMS near 1 in orthonormal FFT. Non-adversarial Tx amplifier compression.
         A=2.25 if condition=='mild' else 1.25
         return z/(1+(z.abs()/A)**4).pow(.25)
     ```
   - In both files, PA distortion is memoryless AM-AM magnitude compression only; AM-PM phase distortion is absent. Both evaluate PA on critically sampled Nyquist rate ($N=32$), with no oversampling.
   - Channel taps $h$ in `p3b_time_link.py` are constant across the entire OFDM symbol duration; no time-varying Doppler evolution or Inter-Carrier Interference (ICI) exists.

4. **Existing Classical Baselines**:
   - In `ofdm_baseline.py` line 64–79, estimators are: `ls` (returns `yp`), `dft` (truncates delay taps to `taps`), and `lmmse` (linear Gaussian prior covariance matrix inversion).
   - In `p3_a.py` (inside `P3-0_A_source_and_results.zip`) and `p3_a_mechanism.py` line 28, `dft_threshold` implements hard/soft thresholding in the delay domain based on familywise false-crossings.
   - In `exploratory/16qam_joint_pa_probe/probe.py` line 78–81:
     ```python
     def baseline_bits(observed,t=THRESH,method='dft8'):
         yp,yd,bits,nv=observed
         h=delay_est(yp) if method=='dft8' else threshold_est(yp,nv)
         return hard_bits(equalize(yd,h,nv),t)
     ```
   - In `exploratory/16qam_joint_pa_probe/REPORT.md` lines 7–8 and 47–48:
     > *"The 1,812-parameter CNN consumes received pilot observations, received data, a DFT-8 pilot-channel estimate and reported noise variance, and predicts four un-calibrated binary logits per 16-QAM symbol. These are not independently verified probability-calibrated LLRs."*
     > *"Full nonlinear-PA-aware iterative correction / decoding and a fully data-aided probabilistic 16-QAM receiver were not implemented. Therefore the suggested definitive 'beat DFT-Thresh plus classical clipping repair, otherwise end the project' test has NOT been completed."*
   - Neither Bussgang-LMMSE, mathematically calibrated 16-QAM soft LLRs, nor Iterative Decision-Directed PA Cancellation exists anywhere in the repository.

5. **Existing Latency Benchmarking**:
   - `p3b_timing_extended.py` line 13–19: `measure(fn, reps=180, warmup=30)` measures `time.perf_counter_ns()` with `torch.set_num_threads(4)`.
   - `exploratory/16qam_joint_pa_probe/timing.py` line 21–26: measures warmup 25 runs, 350 timed runs, computes `median_ms` and `p95_ms` for batches 1 and 64.

---

## 2. Logic Chain

1. From Observation 1, the execution host is strictly CPU-based (PyTorch 2.12.1 CPU, no CUDA). Therefore, all simulator operations, oversampled convolutions, and matrix solvers for Stage P4-0 must run efficiently on CPU without CUDA dependencies. Latency benchmarks must run on CPU (4 threads) by default while providing GPU support if a CUDA device is detected.
2. From Observation 2, repository governance enforces strict immutability of historical stages (P1–P3, exploratory) backed by SHA256 manifests. Any in-place modification of historical files would violate acceptance criteria and potentially fail frozen integrity tests. Therefore, all Stage P4-0 code, baselines, and test harnesses must be isolated in a dedicated directory: `p4/`.
3. From Observation 3, existing channel models are limited to static multipath with Nyquist-rate AM-AM-only PA compression. To fulfill Requirement R1 ("time-domain dynamic channel simulator supporting cyclic prefix (CP) insertion, multipath, Doppler inter-carrier interference (ICI), and oversampled Rapp power amplifier with AM-AM and AM-PM distortion"), we must implement:
   - Dynamic time-varying channel path gains $h_l[n] = h_{l, 0} e^{j 2\pi f_{d, l} n T_s / N}$ across the OFDM symbol, generating ICI;
   - Oversampled ($L \ge 4$) Rapp PA with AM-AM knee $p=2$, saturation levels $A \in \{1.25, 2.25\}$, and non-zero AM-PM phase distortion $\Phi(r) = \frac{\alpha r^q}{1 + (r/\beta)^q}$;
   - CP insertion at transmitter and removal at receiver;
   - Static limit parity verification: zero Doppler ($f_d=0$) and linear PA ($A \to \infty$) must match the static matrix model $Y = H X + W$ with complex error $< 10^{-5}$.
4. From Observation 4, the repository has an explicit documented deficit: neither Bussgang-LMMSE, calibrated soft LLR computation, nor Iterative Decision-Directed PA Cancellation has ever been built. To satisfy Requirement R1, we must:
   - Calculate the Bussgang attenuation factor $\gamma$ and distortion variance $\sigma_D^2$ for the oversampled Rapp PA;
   - Implement Bussgang-LMMSE equalization: $\hat{X} = \frac{\gamma^* H^*}{|\gamma H|^2 + |H|^2 \sigma_D^2 + \sigma_W^2} Y$;
   - Implement calibrated 16-QAM soft LLRs mapping effective AWGN noise variance into true bit posterior log-odds;
   - Implement Iterative Decision-Directed PA Cancellation over 2–3 iterations (time-domain reconstruction of transmit symbol $\hat{x}$, PA distortion modeling $\hat{d} = f_{PA}(\hat{x}) - \gamma \hat{x}$, channel filtering $\hat{Y}_{dist} = \hat{H} \hat{D}$, subtraction $Y_{clean} = Y - \hat{Y}_{dist}$, and re-equalization).
5. From Observation 5, standardizing the latency benchmarking harness requires adapting the established methodology (4 threads, warmup iterations, 180+ repetitions, reporting median and P95 latency in ms for batch sizes 1 and 64) for all classical baselines.

---

## 3. Caveats

1. **Host Hardware**: The current environment has no CUDA GPU (`CUDA available: False`). All benchmarking will reflect CPU performance (Intel/AMD x86_64, 4 threads). Code should include `torch.cuda.is_available()` branches to support GPU benchmarking seamlessly if run in a GPU environment.
2. **External Libraries**: NVIDIA Sionna 2.2 cannot be used due to network isolation. All channel dynamics, Doppler evolution, and 16-QAM numerology must be self-contained in native PyTorch.
3. **Line Ending Sensitivity in Existing Tests**: Running historical tests on Windows may produce hash mismatches on `.md` files due to git CRLF conversion. P4 test scripts should avoid raw byte hashing of text files without line-ending normalization.

---

## 4. Conclusion

The repository is thoroughly mapped. Historical P1, P2, and P3 implementations provide foundational concepts (frequency-domain OFDM, U(1) equivariance, and static time-domain filtering), but **all four core requirements of Stage P4-0 (dynamic Doppler ICI simulator, oversampled Rapp AM-AM/AM-PM PA, Bussgang-LMMSE with calibrated soft LLRs, and Iterative DD-PA Cancellation)** represent genuine unbuilt capabilities.

To execute Stage P4-0 (Requirement R1) without impacting historical integrity:
1. Establish a dedicated package `p4/` containing `simulator.py`, `classical_baselines.py`, `benchmark_latency.py`, and `test_p4_0_integrity.py`.
2. Implement the oversampled time-domain dynamic channel simulator with exact static-limit parity verification.
3. Implement Bussgang-LMMSE, exact 16-QAM soft LLR calculation, and 2–3 iteration DD-PA cancellation.
4. Execute latency benchmarks on CPU and produce reproducible evidence (`latency_cpu.csv`, `timing.json`, `ber_summary.csv`).

---

## 5. Verification Method

To independently reproduce and verify the findings of this survey:
1. **Environment Verification**:
   ```powershell
   python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available(), 'Threads:', torch.get_num_threads())"
   ```
2. **Audit Check for Missing Components**:
   ```powershell
   cd ''
   # Verify Bussgang is absent:
   Select-String -Path "*.py", "exploratory\**\*.py" -Pattern "Bussgang"
   # Verify AM-PM is absent:
   Select-String -Path "*.py", "exploratory\**\*.py" -Pattern "AM-PM"
   # Verify Doppler is absent:
   Select-String -Path "*.py", "exploratory\**\*.py" -Pattern "Doppler"
   ```
   All three commands return empty results in source files, proving that none of these components currently exist.
3. **Detailed Survey Report Inspection**:
   Inspect `teamwork_explorer_survey_1\report.md` for mathematical formulations, existing parameter audits, and exact P4-0 feature specifications.
