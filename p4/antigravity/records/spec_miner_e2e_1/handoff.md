# Handoff Report: Specification Mining for Stage P4 E2E Testing (Features F1 to F22)

**Agent:** `teamwork_spec_miner`
**Milestone:** Stage P4 E2E Testing Track
**Working Directory:** ``
**State Directory:** `spec_miner_e2e_1`
**Date:** 2026-11-15

---

## 1. Observation
1. **Mandatory Input Documents Inspected**:
   - `PROJECT.md`: Defines feature catalog F1–F22 across milestones M1–M4, plus F23 (E2E Test Suite) and F24 (Adversarial Hardening) (Lines 31–57). Line 101 predeclares confirmatory seed set $\mathcal{S}_{\text{P4D}} = [74001, \dots, 74010]$.
   - `sub_orch_e2e\SCOPE.md`: Prescribes the 4-tier testing hierarchy:
     - Tier 1: $\ge 5$ tests per feature isolated ($\ge 110$ tests).
     - Tier 2: $\ge 5$ boundary/corner tests per feature ($\ge 110$ tests).
     - Tier 3: $\ge 22$ pairwise cross-feature interaction tests.
     - Tier 4: $\ge 11$ realistic end-to-end communication scenarios.
     - Total $\ge 253$ test cases.
2. **Authoritative Survey Reports Audited**:
   - `teamwork_explorer_survey_1\report.md`: Complete mathematical derivations for F1–F7 (Rapp AM-AM/AM-PM formulas, Bussgang attenuation $\gamma$ and distortion variance $\sigma_D^2$, Bussgang-LMMSE matrix filter, calibrated 16-QAM soft LLR equations, IDD-PA cancellation loop).
   - `teamwork_explorer_survey_2\report.md`: Formalization of $C^* = \mathbb{R}_{>0} \times \mathrm{U}(1)$ gauge normalization (pilot RMS anchor $s_0$, complex RMSNorm), Hermitian Green's propagator attention (complex linear $b=0$, complex relative RoPE, radial activation), bidirectional pilot-data cross attention, and joint physical consistency loss.
   - `teamwork_explorer_survey_3\report.md`: Detailed 2D slot grid ($14 \times 32/64$), sparse DMRS pattern (symbols 2, 11; comb-2), continuous Doppler ICI matrix, sample-level CFO phase drift, native outer Rate-1/2 FEC soft decoding, $2^3$ orthogonal factorial ablation matrix, and exact 1024-state sign-flip permutation test with Holm adjustment.
3. **Historical Codebase Integrity**:
   - `MANIFEST_SHA256.json` verified to exist in the repository root (17,811 bytes). All historical stages P1–P3 are frozen. No file outside the state directory and designated test target was modified during this mining turn.

---

## 2. Logic Chain
1. *From Requirement R1 and Survey 1*: F1–F7 require exact closed-form expressions for the nonlinear Rapp PA ($p=2, A \in \{1.25, 2.25\}$ with oversampling factor $L \ge 4$), dynamic multipath with continuous Doppler phase evolution, static limit parity check ($\epsilon < 10^{-5}$), Bussgang parameter evaluation ($\gamma, \sigma_D^2$), calibrated 16-QAM soft LLR demapping, 2-3 iteration IDD-PA cancellation, and standardized 4-thread CPU latency benchmarking.
2. *From Requirement R2 and Survey 2*: F8–F12 necessitate strict mathematical formulation of scale-and-phase gauge transformations $(c Y_p, c Y_d, |c|^2 \sigma^2)$, zero-bias complex linear transformations, relative subcarrier distance phase modulation via RoPE, radial activations preserving $\mathrm{U}(1)$ angle, bidirectional pilot-data iterative coupling, multi-task physical consistency loss ($\mathcal{L}_{\text{BCE}} + \lambda_1 \mathcal{L}_{\text{NMSE}} + \lambda_2 \mathcal{L}_{\text{PA-res}} + \lambda_3 \mathcal{L}_{\text{delay}}$), and deterministic unit test assertions ($\epsilon_H < 10^{-5}, N_{\text{flip}} \equiv 0$).
3. *From Requirement R3 and Survey 3*: F13–F17 require extending to 2D time-frequency grids ($14 \times 32/64$) with sparse DMRS ($\le 10\%$), sample-level continuous Doppler ICI and CFO phase accumulation, native Rate-1/2 FEC (LDPC / Conv) with soft message passing, 2D Wiener interpolation and iterative cancellation baselines, and coded BLER evaluation across OOD Doppler and delay spreads.
4. *From Requirement R4 and Survey 3*: F18–F22 dictate an orthogonal $2^3$ factorial ablation study isolating $C^*$ gauge norm, Green's attention, and joint propagation; a 24-octave numeric stability audit across FP32, BF16, and FP16; a pre-frozen confirmatory protocol over 10 fresh seeds ($74001–74010$) with exact integer bit error tallies; exact 1024-state two-sided sign-flip permutation tests with Holm step-down adjustment; and cryptographic SHA256 governance.
5. *Synthesis into Test Hierarchy*: Mapping these 22 features into Tier 1 (110 tests), Tier 2 (110 tests), Tier 3 (22 tests), and Tier 4 (11 tests) produces exactly 253 verifiable test specifications.

---

## 3. Caveats
1. No source code in `p4/` or `tests/` was modified or executed, preserving strict read-only compliance for this agent.
2. The specific outer FEC implementation choice between 5G NR LDPC Base Graph 2 and Rate-1/2 Convolutional code ($K=7$) has been fully specified for both options in the report so test engineers can support either or both.
3. Tests for M2–M4 components must support progressive execution (e.g. running mockable interface contract verifications before M2/M3 weights are trained).

---

## 4. Conclusion
A comprehensive, authoritative specification mining report covering all 22 features (F1 through F22) has been compiled and saved to:
`spec_miner_e2e_1\spec_mining_report.md`

The report provides:
- The canonical **Features Discovered Table** and **Edge Cases Table**.
- Complete mathematical formulations and equations for every feature.
- Explicit input/output tensor shapes, dtypes, and expected parameter ranges.
- Quantifiable numeric tolerances and acceptance thresholds (e.g., $\epsilon_{\text{parity}} < 10^{-5}, N_{\text{flip}} \equiv 0, \text{ECE} < 0.02$).
- Complete breakdown of 110 Tier 1 tests, 110 Tier 2 boundary tests, 22 Tier 3 cross-feature interaction pairs, and 11 Tier 4 real-world communication scenarios (total 253 tests).

---

## 5. Verification Method
1. **Report Verification**:
   Inspect the mining report:
   ```powershell
   Get-Content 'spec_miner_e2e_1\spec_mining_report.md' -Head 40
   ```
2. **Completeness Verification**:
   Check that all 22 features F1 through F22 are present in the report:
   ```powershell
   Select-String -Path 'spec_miner_e2e_1\spec_mining_report.md' -Pattern 'Feature F[0-9]+:'
   ```
3. **Test Count Verification**:
   Verify that Tier 1 (110 tests), Tier 2 (110 tests), Tier 3 (22 tests), and Tier 4 (11 scenarios) are fully enumerated.
