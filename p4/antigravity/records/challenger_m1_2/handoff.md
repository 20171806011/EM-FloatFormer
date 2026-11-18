# Adversarial Challenge Report: Stage P4-0 Classical Receiver Baselines

**Author:** Challenger 2 (`challenger_m1_2`)
**Role:** Critic / Specialist
**Working Directory:** ``
**State Directory:** `challenger_m1_2`
**Milestone:** Milestone 1 (Stage P4-0: Dynamic Channel Simulation & Nonlinear Classical Baselines)
**Date:** 2026-11-15
**Verdict:** **APPROVE**

---

## 1. Observation

### 1.1 Code Under Review
The classical baselines implementation in `p4/classical_baselines.py` provides:
1. Analytical 1D quadrature calculation of Bussgang parameters $\gamma$, $\sigma_D^2$, and $P_{out}$ (`bussgang_params`, lines 42–93).
2. Truncated DFT-8 channel delay-domain estimation (`delay_est_dft`, lines 104–116).
3. Bussgang-LMMSE equalization accounting for PA distortion variance (`bussgang_lmmse`, lines 239–318).
4. Calibrated 16-QAM soft LLR calculation via exact LogSumExp and piecewise MaxLog (`calibrated_16qam_llrs`, lines 122–202).
5. Iterative Decision-Directed PA Cancellation for 2–3 iterations (`iterative_dd_pa_cancellation`, lines 335–465).

### 1.2 Independent Adversarial Probe Execution
An independent adversarial harness was developed and executed from `challenger_m1_2/adversarial_probe_classical.py`:
- Command: `python challenger_m1_2/adversarial_probe_classical.py`
- Result: Exit code 0 (`ALL ADVERSARIAL CHALLENGES COMPLETED WITH ZERO ERRORS / FAILURES`).

### 1.3 Measured Verification Results

#### Experiment 1: Bussgang Decomposition & LMMSE Equalizer Numerical Robustness
1. **Extreme SNR Regimes**:
   - $\text{SNR} = -10.0 \text{ dB}$: `NaN: False`, `Inf: False`, `mu: [5.84e-06, 0.7128]`, `sigma2_eff: [4.03e-01, 1.71e+05]`, `BER: 0.4819`.
   - $\text{SNR} = -5.0 \text{ dB}$: `NaN: False`, `Inf: False`, `mu: [1.50e-04, 0.7663]`, `sigma2_eff: [3.05e-01, 6.66e+03]`, `BER: 0.4523`.
   - $\text{SNR} = 0.0 \text{ dB}$: `NaN: False`, `Inf: False`, `mu: [2.92e-05, 0.8517]`, `sigma2_eff: [1.74e-01, 3.43e+04]`, `BER: 0.3862`.
   - $\text{SNR} = 10.0 \text{ dB}$: `NaN: False`, `Inf: False`, `mu: [1.75e-03, 0.9474]`, `sigma2_eff: [5.55e-02, 5.71e+02]`, `BER: 0.1877`.
   - $\text{SNR} = 20.0 \text{ dB}$: `NaN: False`, `Inf: False`, `mu: [9.44e-03, 0.9633]`, `sigma2_eff: [3.81e-02, 1.05e+02]`, `BER: 0.0481`.
   - $\text{SNR} = 30.0 \text{ dB}$: `NaN: False`, `Inf: False`, `mu: [1.53e-01, 0.9649]`, `sigma2_eff: [3.63e-02, 5.54e+00]`, `BER: 0.0106`.
   - $\text{SNR} = 50.0 \text{ dB}$: `NaN: False`, `Inf: False`, `mu: [9.45e-01, 0.9651]`, `sigma2_eff: [3.61e-02, 5.82e-02]`, `BER: 0.0040`.
   - Infinite SNR limit ($\sigma_W^2 = 0$): `NaN: False`, `Inf: False`, `mu: [0.9901, 0.9901]`, `sigma2_eff: [1.00e-02, 1.00e-02]`.
2. **Singular & Deep Fading Channels**:
   - Exact zero channel ($H[k] = 0, \forall k$): `NaN: False`, `Inf: False`, `z_hat max: 0.00e+00`, `mu: 1.00e-08`, `sigma2_eff: 1.00e+08`, `llrs max: 0.00e+00`. LLR collapses to 0.0 (maximum entropy, $P(b_i=1)=0.5$) with zero division errors.
   - Deep notch ($H[15] = 10^{-12}$): `NaN: False`, `Inf: False`, subcarrier 15 `mu: 1.00e-08`, `sigma2_eff: 1.00e+08`, `LLR max: 0.00e+00`.
   - Massive gain ($H[k] = 10^6$): `NaN: False`, `Inf: False`, `mu mean: 0.9651`, `sigma2 mean: 3.61e-02`.
3. **Bussgang Parameter Quadrature Stability**:
   - Tested clipping thresholds $A \in \{0.2, 0.5, 0.8, 1.0, 1.25, 1.5, 2.0, 2.25, 3.0, 5.0, 10.0, \infty\}$.
   - Across all $A$, energy balance $P_{out} = |\gamma|^2 + \sigma_D^2$ holds with discrepancy $< 10^{-15}$.
   - $A=1.25$: $\gamma = 0.7711 + 0.2257j$ ($|\gamma| = 0.8035$), $\sigma_D^2 = 0.02332$, $P_{out} = 0.6689$.
   - $A=2.25$: $\gamma = 0.9121 + 0.2770j$ ($|\gamma| = 0.9532$), $\sigma_D^2 = 0.00912$, $P_{out} = 0.9177$.
   - $A=\infty$: $\gamma = 1.0 + 0.0j$, $\sigma_D^2 = 0.0$, $P_{out} = 1.0$.

#### Experiment 2: 16-QAM Soft LLR Calibration & Extreme Value Numerical Auditing
1. **Calibration Accuracy ($\text{BER}$ vs $\mathbb{E}[\sigma(-|LLR|)]$):**
   - Under perfect CSI (Known channel):
     * $\text{SNR} = -5.0 \text{ dB}$: $\text{BER} = 0.41770$, $\mathbb{E}[\sigma(-|LLR|)] = 0.41643$, Gap = $0.00127$ ($0.12\%$).
     * $\text{SNR} = 0.0 \text{ dB}$: $\text{BER} = 0.34891$, $\mathbb{E}[\sigma(-|LLR|)] = 0.35284$, Gap = $0.00392$ ($0.39\%$).
     * $\text{SNR} = 10.0 \text{ dB}$: $\text{BER} = 0.16206$, $\mathbb{E}[\sigma(-|LLR|)] = 0.16800$, Gap = $0.00593$ ($0.59\%$).
     * $\text{SNR} = 20.0 \text{ dB}$: $\text{BER} = 0.03871$, $\mathbb{E}[\sigma(-|LLR|)] = 0.04398$, Gap = $0.00527$ ($0.53\%$).
     * $\text{SNR} = 30.0 \text{ dB}$: $\text{BER} = 0.00702$, $\mathbb{E}[\sigma(-|LLR|)] = 0.00899$, Gap = $0.00197$ ($0.20\%$).
   - Under DFT-8 estimated CSI:
     * Calibration gap remains $< 0.0072$ for $\text{SNR} \ge 15 \text{ dB}$. At $-5 \text{ dB}$, gap rises to $0.0593$ because the baseline LMMSE variance model does not incorporate pilot noise estimation variance $\sigma_{\tilde{H}}^2$.
2. **Sign Agreement & Bit-MAP Behavior**:
   - Piecewise MaxLog sign matches nearest-neighbor hard slicing $100.0\%$ under all SNR regimes and constellation inputs.
   - Exact LogSumExp sign computes true marginal posterior bit-MAP decisions. In low SNR / deep fade regimes ($\sigma^2_{eff} \gg 1$), Exact LogSumExp sign deviates from nearest-neighbor slicing (agreement is $84.2\%$ at $-5 \text{ dB}$, $94.0\%$ at $5 \text{ dB}$, $99.0\%$ at $15 \text{ dB}$, $99.7\%$ at $20 \text{ dB}$, and $100.0\%$ at $30 \text{ dB}$).
   - In low-SNR regimes, Exact LogSumExp bit-MAP decisions achieve strictly lower BER than nearest-neighbor hard slicing ($0.4420 < 0.4454$ at $-5 \text{ dB}$).
3. **Exact LogSumExp vs MaxLog Approximation**:
   - Exact achieves strictly lower log-loss (BCE) than MaxLog: at $0 \text{ dB}$, $\text{BCE}_{exact} = 0.5384 < \text{BCE}_{maxlog} = 0.5447$.
   - The absolute difference $|LLR_{exact} - LLR_{maxlog}|$ is strictly bounded by the theoretical maximum $\ln(2) \approx 0.6931$ (empirical maximum $0.509$ at $0 \text{ dB}$, $0.693$ at $10 \text{ dB}$).
   - Sign mismatch between Exact and MaxLog is $6.22\%$ at $0 \text{ dB}$, $0.67\%$ at $5 \text{ dB}$, $0.0012\%$ at $10 \text{ dB}$, and $0.0000\%$ at $20 \text{ dB}$.
4. **Numerical Stability on Extreme Inputs**:
   - Evaluated inputs up to $Z = 10^7$ and $Z = 10^{-8}$ with variances $\sigma^2 \in [10^{-8}, 10^8]$ and zero amplitude $Z=0$.
   - Zero NaNs, zero Infs for both Exact LogSumExp and MaxLog methods.

#### Experiment 3: Iterative DD-PA Cancellation Stress Testing
1. **Distortion-Limited Regime ($\text{SNR} \ge 18 \text{ dB}, A=1.25$):**
   - $\text{SNR} = 18.0 \text{ dB}$: $\text{BER} = [0.0646 \to 0.0591 \to 0.0591 \to 0.0590]$, $\text{MSE} = [0.3652 \to 0.3559 \to 0.3589 \to 0.3599]$.
   - $\text{SNR} = 20.0 \text{ dB}$: $\text{BER} = [0.0459 \to 0.0393 \to 0.0385 \to 0.0386]$, $\text{MSE} = [0.2285 \to 0.2149 \to 0.2159 \to 0.2164]$ ($16.1\%$ BER reduction).
   - $\text{SNR} = 25.0 \text{ dB}$: $\text{BER} = [0.0222 \to 0.0154 \to 0.0150 \to 0.0150]$, $\text{MSE} = [0.1347 \to 0.1166 \to 0.1158 \to 0.1158]$ ($32.4\%$ BER reduction).
   - $\text{SNR} = 30.0 \text{ dB}$: $\text{BER} = [0.0118 \to 0.0064 \to 0.0061 \to 0.0061]$, $\text{MSE} = [0.0553 \to 0.0351 \to 0.0341 \to 0.0341]$ ($48.3\%$ BER reduction).
   - Iteration 1 provides $>90\%$ of total cancellation gain; Iteration 2 provides an additional small gain; Iteration 3 converges stably with no divergence.
2. **Low-SNR Stability & Error Propagation**:
   - Swept $\text{SNR} \in \{-10.0, -5.0, -2.0, 0.0, 2.0, 5.0\} \text{ dB}$.
   - Despite high decision error rates ($\text{BER} \in [0.28, 0.48]$), DD-PA never diverges or explodes:
     * $\text{SNR} = -10.0 \text{ dB}$: $\text{MSE} = [50.57 \to 51.04 \to 51.19 \to 51.22]$, $\Delta \text{MSE} \le 1.3\%$.
     * $\text{SNR} = 0.0 \text{ dB}$: $\text{MSE} = [17.03 \to 17.19 \to 17.29 \to 17.32]$, $\Delta \text{MSE} \le 1.7\%$.
     * Zero NaNs, zero Infs across all iterations.
3. **Channel Estimation Error Sensitivity**:
   - Injected synthetic channel estimation error variance $\sigma_{\tilde{H}}^2 \in [0.0, 1.0]$.
   - Under $100\%$ channel estimation error ($\sigma_{\tilde{H}}^2 = 1.0$), DD-PA remains numerically stable ($\text{MSE} \approx 3.71$ across all iterations, with zero NaNs).
4. **Soft Decisions vs Hard Decisions**:
   - At high SNR ($25 \text{ dB}$): Soft-decision DD-PA achieves $0.0145$ BER vs $0.0148$ hard-decision DD-PA.
   - At low SNR ($5 \text{ dB}$): Soft-decision DD-PA achieves $0.2920$ BER vs $0.2946$ hard-decision DD-PA, demonstrating that soft-amplitude weighting attenuates unreliable decisions and reduces error propagation.

### 1.4 Test Suite & Repository Governance
- Official unit test suite (`python -m unittest p4/tests/test_p4_0_simulator.py -v`) executed with 14/14 tests passing in 2.25s.
- `git status --short` confirmed zero modified tracked files. Historical milestones P1–P3 and manifests remain 100% untouched.

---

## 2. Logic Chain

1. **Numerical Stability of Bussgang-LMMSE**:
   - Observations 1.3.1 and 1.3.2 demonstrate that `bussgang_lmmse` incorporates two protective mechanisms: (a) adding $10^{-12}$ to the MMSE denominator, and (b) clamping $\mu = \text{clamp}(|H_{eff}|^2 / \text{denom}, \min=10^{-8}, \max=1.0)$.
   - When channel $H \to 0$ or deep notches occur, $W \to 0$, $X \to 0$, and $\mu \to 10^{-8}$. Consequently, $Z = X / \mu \to 0$, and effective variance $\sigma^2_{eff} = (1-\mu)/\mu \to 10^8$.
   - In `calibrated_16qam_llrs`, a variance of $10^8$ scales all distance exponentials to near zero, yielding $LLR \to 0.00$ ($P(b=1) = 0.5$, maximum entropy).
   - This mathematically proves that divide-by-zero, floating-point overflow, and NaN corruption are precluded across all channel gains and noise levels.

2. **16-QAM Soft LLR Calibration and Boundary Mechanics**:
   - Observation 1.3.3 confirms that under known CSI, empirical BER matches predicted bit error probability $\mathbb{E}[\sigma(-|LLR|)]$ to within $< 0.006$ ($0.6\%$) across all SNRs from $-5 \text{ dB}$ to $+30 \text{ dB}$.
   - Observation 1.3.3 also reveals that Exact LogSumExp and Piecewise MaxLog represent two distinct, well-defined mathematical objects:
     * Piecewise MaxLog evaluates the distance to the single nearest constellation point, whose sign strictly coincides with nearest-neighbor hard slicing ($100.0\%$ agreement).
     * Exact LogSumExp evaluates the marginal posterior probability sum $\sum_{s \in S_1} e^{-|z-s|^2/2\sigma^2} / \sum_{s \in S_0} e^{-|z-s|^2/2\sigma^2}$. In low SNR or deep fades, secondary constellation points have non-negligible probability mass. Marginal bit-MAP detection legitimately deviates from nearest-neighbor slicing and achieves lower bit error rate ($0.4420 < 0.4454$ at $-5 \text{ dB}$).
   - The difference $|LLR_{exact} - LLR_{maxlog}|$ is strictly bounded by $\ln(2) \approx 0.6931$. Thus, both implementations are structurally and mathematically correct.

3. **Convergence and Resilience of Iterative DD-PA Cancellation**:
   - Observation 1.3.4 proves that in distortion-limited scenarios ($A=1.25, \text{SNR} \ge 18 \text{ dB}$), DD-PA reduces BER by $16.1\%$ at $20 \text{ dB}$, $32.4\%$ at $25 \text{ dB}$, and $48.3\%$ at $30 \text{ dB}$.
   - Iteration 1 captures over $90\%$ of the cancellable distortion; Iterations 2 and 3 converge without oscillation or overshoot.
   - In noise-limited regimes ($\text{SNR} \le 10 \text{ dB}$) or under high channel estimation error ($\sigma_{\tilde{H}}^2 = 1.0$), decision feedback introduces residual errors, but the iteration does NOT diverge: MSE is bounded to within $2\%$ of Iteration 0, and no numerical explosions occur.

---

## 3. Caveats

1. **Unmodeled Channel Estimation Variance at Low SNR**: In `bussgang_lmmse`, the disturbance variance denominator is modeled as $|H_{eff}|^2 + |H_{phys}|^2 \sigma_D^2 + \sigma_W^2$. It assumes $\hat{H}$ is the true channel and does not include the pilot estimation error variance $\sigma_{\tilde{H}}^2 = \sigma_W^2 / (N / taps)$. At low SNR ($-5 \text{ dB}$), this causes a minor calibration gap ($0.059$), which disappears under perfect CSI. This is standard for non-iterative classical LMMSE and is fully acceptable.
2. **Decision Feedback Error Propagation at Low SNR**: In severe noise regimes ($\text{SNR} < 10 \text{ dB}$), DD-PA provides no BER gain over Iteration 0 because clipping distortion ($\sigma_D^2 \approx 0.023$) is small compared to noise ($\sigma_W^2 \ge 0.1$). In practical systems, DD-PA would be gated by an SNR threshold ($\text{SNR} \ge 15 \text{ dB}$).
3. **Constellation Scope**: Evaluations were conducted under 16-QAM and QPSK as specified in Milestone 1 requirements. Higher constellations (e.g. 64-QAM) are out of scope for Stage P4-0.

---

## 4. Conclusion

The classical algorithms in `p4/classical_baselines.py` have been rigorously challenged and stress-tested against:
1. Extreme SNR regimes ($-10 \text{ dB}$ to $+50 \text{ dB}$ and zero noise).
2. Singular channel states ($H=0$, near-zero notches $10^{-12}$, massive gain $10^6$).
3. Soft LLR calibration across exact LogSumExp and piecewise MaxLog under both estimated and perfect CSI.
4. Iterative DD-PA cancellation under low SNR, high channel estimation error, and strong PA distortion ($A=1.25$).

All algorithms demonstrated complete numerical stability (zero NaNs, zero Infs, bounded MSE), valid calibration, and consistent error reduction in distortion-limited regimes.

**Explicit Verdict: APPROVE**.

---

## 5. Verification Method

To independently reproduce and verify this challenger assessment from ``:

1. **Run Independent Adversarial Probe Harness**:
   ```powershell
   python challenger_m1_2\adversarial_probe_classical.py
   ```
   *Expected result*: Executes Experiments 1, 2, and 3, prints all numerical values, and exits with code 0 (`### ALL ADVERSARIAL CHALLENGES COMPLETED WITH ZERO ERRORS / FAILURES ###`).

2. **Run Official Test Suite**:
   ```powershell
   python -m unittest p4/tests/test_p4_0_simulator.py -v
   ```
   *Expected result*: 14 tests run, 14 pass, exit code 0 (`OK`).

3. **Verify Git Immutability**:
   ```powershell
   git status --short
   ```
   *Expected result*: Zero modified tracked files; historical milestones and manifests remain 100% untouched.
