# Track M5 (Round 1): First-Principles Unified Multi-Query Cache, Active Reference Selection (Ultimate Blind Test) & Stage 4→6 Bridge

You are **Track M5**, added to our parallel mathematical design team specifically to examine the **long-term ultimate goal** in GitHub (`docs/LONG_TERM_RESEARCH_PLAN.md` + `docs/MULTIMODAL_INTEGRATION_PLAN.md`) from **first principles and information theory**, ensuring that the representations and architectures we build in Stage 4 generalize cleanly through **Stage 5 (event/trajectory model selection & jump localization)** and **Stage 6 (Ultimate Blind Test active window selection + multi-host 4B/9B/27B transfer + LEMMO comparison)** without dead-end rework or trial-and-error!

## 0. Strict Workspace, Environment, and Immutability Rules
- **Repository Root**: `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`
- **Your Writable Directory**: `research/stage4_math_and_design/M5/` ONLY.
- **FORBIDDEN Edits**: Do NOT modify or delete any files outside `research/stage4_math_and_design/M5/`.
- **Python Environment Guardrail**: System/Anaconda `python` crashes with MKL `libiomp5md.dll` conflicts on this machine. **ALWAYS** execute Python scripts via:
  ```powershell
  uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B <script.py>
  ```
  (Omit `--with torch` if only numpy/scipy are needed.)

## 1. Grounded Context to Inspect First (Read-Only)
Inspect relevant sections of:
1. `docs/LONG_TERM_RESEARCH_PLAN.md` (especially "总目标与双重验收", "二、我们要争取的贡献", "三、阶段安排" Stages 4–6, and "五、实验怎样表明模型读到了波形")
2. `docs/MULTIMODAL_INTEGRATION_PLAN.md` (3B–27B hosts: Ministral-3-3B, Qwen3.5-4B/9B, Qwen3.8-27B, LEMMO comparison protocol)
3. `research/matched_cache/learned.py` (`equivariant_b8_s17`, `input_features`, 4-bit-per-window discrete quantization `z = 4*turns + local`)
4. `research/stage3c/G/quotient.py` (`canonical_input`, `rotate`, `one_call`)
5. `research/stage3c/A/q2_reader.py` (`read`)
6. `research/stage3d_qwen_signed_phase/data_stage3d.py` & `REPORT.md`

### Core Ultimate Goal Stated in `docs/LONG_TERM_RESEARCH_PLAN.md`
- **Pre-Question Fixed Cache**: Model reads a civilian IQ recording $Y$ ONCE before any question is asked, forming a compact size-bounded, time-window-indexed cache $Z = T(Y)$ under bit budget $B$. Without re-reading raw $Y$, the model must support arbitrary follow-up questions:
  1. Relative phase / relationship (`Q1`, `Q3`);
  2. Belief revision when an additional revealed reference $A$ arrives (`Q2`, `Q3_after_pilot`);
  3. **Stage 5 Event & Generative-Model Discrimination**: Given two windows, can one smooth trajectory ($\omega, \sigma_\phi$) explain them, or did a discrete jump happen in a specific interval $\tau$, or are they two independent processes, and how much probability mass remains on the rival hypothesis?
  4. **Stage 6 Ultimate Blind Test ("先编码，后揭题" + Active Evidence Selection)**: When two physical explanations remain compatible with $Z$, the model may request ONLY ONE additional recorded candidate window $a^* \in \mathcal{C}$ before knowing its value, FIRST commit to $a^*$ AND write down what observation $Y_{a^*}$ each rival hypothesis predicts, and THEN reveal $Y_{a^*}$ and update its answer — compared against random-window, nearest-window, and physical Bayes EIG baselines!

## 2. First-Principles Mathematical Questions You Must Answer Rigorously
1. **Universal Pre-Question Minimal Sufficiency & Rate-Distortion Frontier (Across Q1, Q2, Stage-5 Jump Localization, and Stage-6 Active EIG)**:
   - Model a multi-window QPSK sequence $Y = (Y_1, \dots, Y_L)$ where each window $k$ has complex matched-sum $T_k = \alpha_k i^{X_k + G_k} e^{i \phi_k} + \eta_k$.
   - Derive the **exact task-union minimal sufficient statistic** (and finite-bit rate-distortion risk frontier $R_q(B)$ as a function of cache bit-budget $B$ bits/window, reference SNR, and phase lag) needed to simultaneously minimize excess Bayes risk across ALL four query families (`Q1`, `Q2`, `Stage 5 hypothesis test`, `Stage 6 EIG selection`)!
   - Explicitly prove the exact gap between single-task quantization and universal sufficiency (generalizing the 2-bit vs 4-bit noiseless separation in `LONG_TERM_RESEARCH_PLAN.md` to noisy partial-coherence, amplitude fading $|\alpha_k|$, and multi-hypothesis regimes!).
2. **Exact Information-Theoretic Ceiling Audit of `equivariant_b8_s17` vs Future-Proof Canonical Complex-Evidence Cache**:
   - Audit `research/matched_cache/learned.py`'s `equivariant_b8_s17` (`4 bits/window` = `2 bits turn + 2 bits canonical sector`, fit ONLY on noiseless/unitary-amplitude 2 dB coherent `C1`):
     - Quantify mathematically WHERE `s17` is near-sufficient (`12.2 ms` vs `0.504 nat`, only `~0.012 nat` below `full_iq` on `G1` synthetic 2 dB `Q3`!) vs WHERE keeping ONLY 4-bit discrete `s17` **permanently destroys essential information** required for Stage 4 real Pluto/CTTC (`SNR` varies `-3..+24 dB`, amplitude fading `>10 dB`, residual DC/spur quality), Stage 5 (distinguishing smooth CFO drift vs discrete jump vs low-SNR noise burst), and Stage 6 (pre-reveal amplitude/phase prediction under rival hypotheses)!
     - Derive a **compact, host-agnostic canonical cache schema** $\tilde{Z}_k = (\text{turn}_k, \text{canon\_phasor}_k, \text{log\_precision}_k, \text{resid\_quality}_k)$ quantized cleanly to finite bits (e.g. compare `4 bits` s17 vs `8, 12, 16, 24 bits/window` canonical factored float/fixed-point cache) such that:
       (a) It strictly generalizes and remains **100% backward-compatible** with `s17` (a discrete projection recovers `s17` cosets `(turn, inner)`!),
       (b) Every Stage 4→5→6 connector (`EquivariantEvidenceConnector` + `RefSufficientPool` + Stage 5 event head + Stage 6 EIG selector) works identically across `Qwen3.5-4B`, `Qwen3.5-9B`, and `Qwen3.8-27B` with ZERO architectural rewrite!
3. **Exact Expected Information Gain (EIG) Theorem for the Ultimate Blind Test (Stage 6)**:
   - Suppose after encoding $Y$, two rival hypotheses $H_0$ (e.g., smooth CFO trajectory with unknown $\omega$ or small jump) vs $H_1$ (discrete jump / change-point $\tau \in \mathcal{T}$ / rival sector) have posterior weights $(\pi_0, 1-\pi_0)$ consistent with $Z$.
   - Derive the closed-form Bayes **Expected Information Gain** $\text{EIG}(a \mid Z) = I(H; Y_a \mid Z)$ for revealing candidate reference window $a \in \mathcal{C}$ BEFORE seeing $Y_a$, show how predicted observations $p(Y_a \mid Z, H_h)$ are scored properly via log-score prior to reveal, and prove exact structural dominance theorems showing when `active EIG` strictly beats BOTH `random window` and `nearest window` (e.g., midpoint split / anti-aliasing multiple-spacing intervals $t_a$ resolving $\omega$ aliasing & localizing change-point $\tau$!).
4. **Mandatory CPU Empirical Verification (`verify_m5.py`)**:
   - Write `research/stage4_math_and_design/M5/verify_m5.py` (using `uv run --python 3.12 ...`).
   - Load the REAL `equivariant_b8_s17.npz` encoder (`research/matched_cache/weights/equivariant_b8_s17.npz`) and simulate multi-window QPSK recordings covering Q1, Q2, Stage-5 jump-vs-CFO model discrimination, and Stage-6 Ultimate Blind Test active reference selection across fading/SNR variations:
     1. Compute empirical **Bit-Budget vs Multi-Task Risk Pareto table** across `B = 2, 4 (s17), 8, 12, 16 bits/window` vs Full IQ (`infty`);
     2. Quantify exact discriminability loss of `s17` vs canonical continuous/16-bit cache on varying amplitude/SNR + smooth-CFO-vs-jump separation;
     3. Run the end-to-end **Ultimate Blind Test CPU simulation**: compare `random_window`, `nearest_window`, and `max_EIG_window` on pre-reveal hypothesis prediction consistency and post-reveal Bayes accuracy & NLL!
   - Write results to `research/stage4_math_and_design/M5/results_m5.json`.

## 3. Deliverables Required from Track M5
1. `research/stage4_math_and_design/M5/THEOREM_M5.md`
2. `research/stage4_math_and_design/M5/verify_m5.py` (executed cleanly via `uv run`)
3. `research/stage4_math_and_design/M5/results_m5.json`
4. Crisp executive summary in your final message (`reply_r1.md`) reporting exact rate-distortion/EIG theorems, `s17` vs future-proof canonical cache quantitative comparison, and concrete design recommendations for M1/M3/M4/M6 Round 2 so Stage 4 builds directly toward Stage 6 with zero dead ends.
