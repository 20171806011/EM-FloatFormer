# Track M6 (Round 1): First-Principles Zero-Trial-and-Error Decision Theory, Frozen-Host Linearization & Optimal Compute Allocation

You are **Track M6**, added to our parallel mathematical design team specifically to answer a critical meta-scientific & mathematical engineering question from the user:
> **How do we formulate, diagnose, gate, and execute Stages 4 → 6 of GitHub's ultimate EM-FloatFormer roadmap with MINIMUM trial-and-error (ideally ZERO wasted GPU runs), using exact mathematical laws rather than empirical guesswork?**

## 0. Strict Workspace, Environment, and Immutability Rules
- **Repository Root**: `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`
- **Your Writable Directory**: `research/stage4_math_and_design/M6/` ONLY.
- **FORBIDDEN Edits**: Do NOT modify or delete any files outside `research/stage4_math_and_design/M6/`.
- **Python Environment Guardrail**: System/Anaconda `python` crashes with MKL `libiomp5md.dll` conflicts on this machine. **ALWAYS** execute Python scripts via:
  ```powershell
  uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B <script.py>
  ```

## 1. Grounded Context to Inspect First (Read-Only)
Inspect relevant sections of:
1. `docs/LONG_TERM_RESEARCH_PLAN.md` (historical P1–P12 outcomes + "防止研究兜圈子的约定" + Stage 4–6 goals)
2. `docs/MULTIMODAL_INTEGRATION_PLAN.md` (3B–27B multi-host scaling + LEMMO comparison)
3. `research/stage3d_qwen_signed_phase/PROTOCOL_LOCK.json` (`work_e_scenario`, `checkpoints`, `versions`)
4. `research/stage3d_qwen_signed_phase/REPORT.md` (execution timeline & billing: `$17.12` spend out of `$30.00` Modal free tier, **`~$12.88` remaining free compute credits**, `Billed Cost: $0.00`!)
5. `research/stage3d_qwen_signed_phase/qwen_connector.py` (lines 33–139, 432–640)
6. `research/stage3d_qwen_signed_phase/results/summary_master.json`, `blind_q3_scores.npz`, `blind_q2_scores.npz`

### Crucial Historical Diagnosis Motivating This Track
In Stage 3D, `$17.12` of A100-80GB compute was spent training 6 checkpoints (`phase` & `zero`, 3 seeds each) and evaluating 150k+ Q3 predictions + Q2 arms — ONLY to discover AFTER GPU completion that:
1. Even on synthetic 2 dB `G1`, `Full IQ` achieved `64.64% / 0.9290` and frozen `s17` achieved `64.62% / 0.9411` (`s17` lost only `0.012 nats`!), YET Qwen+Connector lost **`0.504 nats`**, collapsing to `23.80% / 1.4452` (`G1`) and `28.78% -> 26.43%` (`Q2`)!
2. Notice: That entire bottleneck (`s17 -> frozen LLM host` additive extrapolation failure, broken $\mathbb{Z}_4$ symmetry, and `<REF><REF>` variable-length attention breakdown) was **mathematically and computationally predictable BEFORE launching a 1-hour A100 training run**!
Meanwhile, we now have **`~$12.88` of free Modal compute remaining** to execute **Stage 4** (synthetic extrapolation + real Pluto 850 MHz + real CTTC 9 sessions) and lay the mathematical foundation for Stage 5–6!

## 2. First-Principles Mathematical Questions You Must Answer Rigorously
1. **Exact Data Processing Inequality (DPI) & Information-Loss Ladder Across Every Pipeline Stage**:
   - Formalize the Markov chain $Y \to Z = T(Y) \to E_\theta(Z, C, \mathcal{R}) \to H_{\text{LLM}}(E_\theta, \text{prompt}) \to \hat{P}_\tau$:
     - Prove exact KL/entropy identities decomposition along the pipeline:
       $$\text{NLL}(\hat{P}_\tau) - \text{NLL}_{\text{Bayes}}(Y) = \underbrace{\Delta_{\text{enc}}(Y \to Z)}_{\text{Representation Loss}} + \underbrace{\Delta_{\text{conn}}(Z \to E_\theta)}_{\text{Connector Sufficient-Stat Loss}} + \underbrace{\Delta_{\text{host}}(E_\theta \to \text{LLM})}_{\text{Frozen Host Alignment Loss}} + \underbrace{\Delta_{\text{cal}}(\tau)}_{\text{Calibration Loss}}$$
     - Evaluate the observable rungs of this EXACT ladder on REAL Stage 3D data (`blind_q3_scores.npz` across `G1..G4`, `H_c4_035` AND `blind_q2_scores.npz` across all Q2 arms), proving quantitatively why Stage 3D failed at rung 2→3 (`\Delta_{conn} + \Delta_{host}`) rather than rung 0→1 (`\Delta_{enc} \approx 0.012 nats` on synthetic 2 dB)!
2. **Necessary-Condition CPU Pre-Screening & Frozen-Host Linearization Theory (Detecting Failures at \$0 Before GPU Training)**:
   - Derive mathematically verifiable **Tier-0 (\$0 Symbolic)** and **Tier-1 (\$0 CPU Surrogate / Linear Readout / Geometry Checks)** **NECESSARY CONDITIONS** whose violation mathematically proves a connector/host interface CANNOT succeed on GPU:
     - *Criterion 1 (Analytical Channel/Cache Ceiling Gap)*: Before training ANY neural module, compute Bayes risk upper/lower bounds from `(Z, C, R)` alone (`direct_table` / causal real-data frontend MI). If Bayes NLL on real Pluto/CTTC `Z` fails a threshold, STOP & fix frontend (`s17` vs canonical cache) before GPU!
     - *Criterion 2 (Group-Orbit & Idempotence Algebraic Invariants)*: Exact `0` defect on $\mathbb{Z}_4$-rotation, duplicate-ID invariance, distractor masking, and token-count invariance.
     - *Criterion 3 (Connector Out-of-Distribution Extrapolation & Frozen-Host Jacobian Separability Condition)*: Let $E_\theta(z, \theta_{\text{cfo}}) \in \mathbb{R}^{K \times d}$ be the connector tokens injected into frozen LLM embeddings. Analyze why linear probing / fixed-tangent NTK or frozen random-transformer surrogate training on CPU (`train: |\omega| \le 0.25, eval: |\omega|=0.40, |\theta| > \pi/2`) exposes rank/additive extrapolation bottlenecks in `< 30s` at `$0` cost, AND prove sufficient rank/conditioning bounds on the host projection $W_{\text{out}} \in \mathbb{R}^{d_{\text{hid}} \times d_{\text{LM}}}$ (orthogonal isotropic frame preservation with host vocabulary unembedding `E_{\text{vocab}}` so injected soft evidence survives layer-norm + residual stream without logit swamping or collapse)!
3. **Sequential Value-of-Information (VoI) Experimental Design & Optimal Budget Allocation (`< $12.88` Free-Credit Calculus)**:
   - Model an R&D sequence of candidate architectures/datasets as sequential Bayesian decisions with GPU cost $c_i$ per epoch/eval block and expected info value $V_i$:
     - Derive a **2-Stage Group-Sequential Stop-Go / Alpha-Spending Design** (Pocock / O'Brien-Fleming boundary + Bayes posterior predictive futility stopping after short pilot `15%–20%` training steps / `1` seed before committing full `3` seeds × full blind eval)!
     - Contrast total expected cost $\mathbb{E}[C_{\text{seq}}]$ and false-promotion probability vs fixed-horizon `3 seeds × 8 epochs × full eval` (`$17.12` in Stage 3D)!
     - Construct an **explicit exact Modal GPU budget plan (`<= $9.50`–`$10.00`, leaving `$2.88+` buffer inside `$12.88` free credits)** covering:
       - Stage 4 Synthetic Extrapolation (`G1..G4` + `Q2_C4` with Equivariant+RefPool connector),
       - Stage 4 Real ADALM-PLUTO (850 MHz, `2,240` blind test frames, `G=16`),
       - Stage 4 Real CTTC (9 sessions, `1,440` blind test packets, `G=72`),
       - Plus exact CPU vs GPU split (downloading & frontending Pluto/CTTC on CPU/Volume first vs A100/L4/A10G forward-only KV-cached scoring!).
4. **Anti-Trial-and-Error Critical-Path DAG Across Stages 4 → 5 → 6**:
   - Systematically diagnose the root recurring traps across P1–P12 and Stage 3D and produce a **strict acyclic dependency DAG** from Stage 4 → Stage 5 → Stage 6 (including `Qwen3.5-4B -> Qwen3.5-9B -> Qwen3.8-27B` connector portability & fair `LEMMO` comparison) where EVERY step has a mathematically falsifiable CPU gate BEFORE GPU spend!
5. **Mandatory CPU Empirical Verification (`verify_m6.py`)**:
   - Write `research/stage4_math_and_design/M6/verify_m6.py` using `uv run --python 3.12 ...`:
     - Compute the exact **Stage 3D DPI Loss Ladder** directly from REAL `blind_q3_scores.npz` (`G1..G4`, `H_c4_035`) and `blind_q2_scores.npz` (assert parity with `summary_master.json`);
     - Run a **10,000-trial Monte Carlo benchmark** comparing fixed brute-force `3-seed full-horizon` (`Stage 3D style`) vs `Tier-1 CPU Pre-Screen + 2-Stage Futility-Gated Sequential GPU` under realistic mixture of broken architectures (like additive `<PHASE>`) and working architectures (like equivariant coupled): verify exact Type-I error (`<= 0.05`), power (`>= 0.90`), and `% compute cost reduction`;
     - Verify exact Modal budget table arithmetic (`<= $10.00 < $12.88`) and save everything to `research/stage4_math_and_design/M6/results_m6.json`.

## 3. Deliverables Required from Track M6
1. `research/stage4_math_and_design/M6/THEOREM_M6.md`
2. `research/stage4_math_and_design/M6/verify_m6.py` (executed cleanly via `uv run`)
3. `research/stage4_math_and_design/M6/results_m6.json`
4. Crisp executive summary in your final message (`reply_r1.md`) giving the exact Stage 3D DPI numbers, Tier-0/1/2 zero-waste checklist, exact compute-cost reduction stats, and pointed meta-architectural questions for M1–M5 Round 2.
