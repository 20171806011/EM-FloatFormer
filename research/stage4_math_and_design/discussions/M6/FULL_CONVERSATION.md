# Full Verbatim Multi-Round Conversation — Track `M6`

- **CLI Runtime**: Codex CLI (`codex exec`)
- **Session / Thread ID**: `01a2194c-3d43-7c13-89f6-3ed11a482246`
- **Round 1 Events**: `130` raw JSONL events (`round1_events.jsonl`)
- **Round 2 Events**: `89` raw JSONL events (`round2_events.jsonl`)
- **Completeness Guarantee**: All user prompts (`prompt_r1.md`, `prompt_r2.md`), all completed reasoning/tool/command events, and final responses (`reply_r1.md`, `reply_r2.md`) are included verbatim below without truncation or paraphrasing.

---

## Round 1 — Initial Mathematical Brief (`prompt_r1.md`)

````markdown
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
````


## Round 1 — Execution & Reasoning Trace (`round1_events.jsonl`)

### Event #1: `thread.started` (thread_id=`01a2194c-3d43-7c13-89f6-3ed11a482246`)

### Event #2: `turn.started`

### Event #3: `item.completed` — `agent_message` (`item_0`)

I’ll read the roadmap, protocol, connector, and saved scores first. Then I’ll build the CPU verifier and use its results to write the theorems, stop/go gates, and budget plan in `M6/`.

### Event #4: `item.completed` — `command_execution` (`item_2`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n -i 'floatformer|stage3d|frozen protocol|matched-budget|fem heat' 'C:\\Users\\李则徐\\.codex\\memories\\MEMORY.md'"
```

**Aggregated Output:**

```text
C:\Users\������\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1 : �޷�ʹ�õ� '.' ��ô�������Դ����Ϊ����
�����ڲ�ͬ����ģʽ�¶���ġ�Ҫ���ô�����������������ݣ������ '.' �������
����λ�� ��:1 �ַ�: 1
+ . 'C:\Users\������\Documents\WindowsPowerShell\Microsoft.PowerShell_prof ...
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) [Microsoft.PowerShell_profile.ps1]��NotSupportedException
    + FullyQualifiedErrorId : DotSourceNotSupported,Microsoft.PowerShell_profile.ps1
 
29:# Task Group: EM-FloatFormer P4 scientific assessment and reproducible publication
32:applies_to: cwd=C:\Users\鏉庡垯寰怽Documents\ChatGPT\EM-FloatFormer; reuse_rule=Reuse the matched-control, reproducibility, and publication approach for this repository; numerical results, commit, hardware status, and receiver outcomes are P4-specific.
38:- rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md (cwd=\\?\C:\Users\鏉庡垯寰怽Documents\ChatGPT\EM-FloatFormer, rollout_path=\\?\C:\Users\鏉庡垯寰怽.codex\sessions\2026\11\17\rollout-2026-11-17T20-54-26-01a1eb4a-230f-7e02-b32d-21348996e25b.jsonl, updated_at=2026-11-18T16:17:48+00:00, thread_id=01a1eb4a-230f-7e02-b32d-21348996e25b, formal P4-D and assessment completed)
42:- EM-FloatFormer, P4, EMFloatFormer2D, C*-equivariance, Hermitian attention, complex RoPE, sparse DMRS, Doppler, CFO, Rapp PA, LDPC, P4D, Holm, sign-flip
48:- rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md (cwd=\\?\C:\Users\鏉庡垯寰怽Documents\ChatGPT\EM-FloatFormer, rollout_path=\\?\C:\Users\鏉庡垯寰怽.codex\sessions\2026\11\17\rollout-2026-11-17T20-54-26-01a1eb4a-230f-7e02-b32d-21348996e25b.jsonl, updated_at=2026-11-18T16:17:48+00:00, thread_id=01a1eb4a-230f-7e02-b32d-21348996e25b, public replay verified)
52:- 20171806011/EM-FloatFormer, 8380084c6dece64119ab72f9c3c1c212e154fb17, publication.verify_p4, p4d_original_audit.py, public_redactions.json, 150 coded rows, ten weight hashes
76:applies_to: cwd=C:\Users\鏉庡垯寰怽Documents\ChatGPT\EM-FloatFormer; reuse_rule=Reuse the diagnostic and UI verification approach; country eligibility, account status, prices, and dates are time- and account-specific.
82:- rollout_summaries/2026-11-17T21-38-49-uldk-google_checkout_debug_youtube_trial_cancellation.md (cwd=\\?\C:\Users\鏉庡垯寰怽Documents\ChatGPT\EM-FloatFormer, rollout_path=C:\Users\鏉庡垯寰怽.codex\sessions\2026\11\17\rollout-2026-11-17T21-38-49-01a1eb72-c399-7cf2-8270-2061ad0e43de.jsonl, updated_at=2026-11-17T21:57:10+00:00, thread_id=01a1eb72-c399-7cf2-8270-2061ad0e43de, logged-in error not observed)
92:- rollout_summaries/2026-11-17T21-38-49-uldk-google_checkout_debug_youtube_trial_cancellation.md (cwd=\\?\C:\Users\鏉庡垯寰怽Documents\ChatGPT\EM-FloatFormer, rollout_path=C:\Users\鏉庡垯寰怽.codex\sessions\2026\11\17\rollout-2026-11-17T21-38-49-01a1eb72-c399-7cf2-8270-2061ad0e43de.jsonl, updated_at=2026-11-17T21:57:10+00:00, thread_id=01a1eb72-c399-7cf2-8270-2061ad0e43de, cancellation verified)
173:# Task Group: Antigravity FEM heat-operator multi-horizon research and full derivation
175:scope: Resume the three-node FEM heat-equation/operator-identification work from local Antigravity artifacts and explain multi-horizon results mathematically.
204:# Task Group: Local TAN state-mechanism research, frozen protocol, and trace-pack stop
206:scope: Local CPU-only DEND/TAN research with per-round planning, frozen protocols, independent audit, and a final equivalent trace-pack candidate that did not establish a latency gain.
264:- Fair matched-budget tuning selected CE with retention (.95,.85) for every DEND/POINT h=2/6 cell; held-out BA was DEND/POINT 98.71%/94.10% at h=2 and 99.80%/99.17% at h=6. The verifier passed 60,986 checks, reconstructed 10,400 Adam updates and 660 mechanism evaluations (max continuous residual about 3.55e-15). [Task 2]
```

### Event #5: `item.completed` — `command_execution` (`item_1`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "Get-Location; git status --short; rg --files -g AGENTS.md -g PROTOCOL_LOCK.json -g REPORT.md -g '*PLAN.md' -g '*scores.npz' -g 'summary_master.json' -g qwen_connector.py -g '"'!**/.git/**'"'; Get-Command uv"
```

**Aggregated Output:**

```text
C:\Users\������\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1 : �޷�ʹ�õ� '.' ��ô�������Դ����Ϊ����
�����ڲ�ͬ����ģʽ�¶���ġ�Ҫ���ô�����������������ݣ������ '.' �������
����λ�� ��:1 �ַ�: 1
+ . 'C:\Users\������\Documents\WindowsPowerShell\Microsoft.PowerShell_prof ...
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) [Microsoft.PowerShell_profile.ps1]��NotSupportedException
    + FullyQualifiedErrorId : DotSourceNotSupported,Microsoft.PowerShell_profile.ps1
 

?? research/stage4_math_and_design/
exploratory\16qam_joint_pa_probe\REPORT.md
p9\PLAN.md
docs\MULTIMODAL_INTEGRATION_PLAN.md
docs\LONG_TERM_RESEARCH_PLAN.md
p6_consistency\REPORT.md
p6_softinfo\REPORT.md
research\stage3c\E\REPORT.md
research\abcde_convergence_2026_11_22\experiments\whole_frame_gate\REPORT.md
research\abcde_convergence_2026_11_22\experiments\parity\REPORT.md
research\abcde_convergence_2026_11_22\experiments\cfo\REPORT.md
research\stage3b_math\PLAN.md
research\abcde_convergence_2026_11_22\C\REPORT.md
research\abcde_convergence_2026_11_22\B\REPORT.md
research\stage3c\work_results\E\REPORT.md
research\abcde_convergence_2026_11_22\A\REPORT.md
p6_followup\REPORT.md
research\content_addressing_scalar\REPORT.md
research\stage3c\work_results\D\REPORT.md
research\stage3c\work_results\B\REPORT.md
research\stage3c\work_results\C\REPORT.md
research\realdata_ofdm_stress_and_theory\REPORT.md
research\stage3_ministral_cache\REPORT.md
research\stage3d_qwen_signed_phase\results\summary_master.json
research\stage3d_qwen_signed_phase\results\replay_c123_scores.npz
research\stage3d_qwen_signed_phase\results\blind_q3_scores.npz
research\stage3d_qwen_signed_phase\results\blind_q2_scores.npz
research\stage3d_qwen_signed_phase\REPORT.md
research\stage3d_qwen_signed_phase\qwen_connector.py
research\stage3d_qwen_signed_phase\PROTOCOL_LOCK.json
Path                                                             
----                                                             
C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4

Path               : C:\qazwsx\Scripts\uv.exe
Extension          : .exe
Definition         : C:\qazwsx\Scripts\uv.exe
Source             : C:\qazwsx\Scripts\uv.exe
Version            : 0.0.0.0
Visibility         : Public
OutputType         : {System.String}
Name               : uv.exe
CommandType        : Application
ModuleName         : 
Module             : 
RemotingCapability : PowerShell
Parameters         : 
ParameterSets      : 
HelpUri            : 
FileVersionInfo    : File:             C:\qazwsx\Scripts\uv.exe
                     InternalName:     
                     OriginalFilename: 
                     FileVersion:      
                     FileDescription:  
                     Product:          
                     ProductVersion:   
                     Debug:            False
                     Patched:          False
                     PreRelease:       False
                     PrivateBuild:     False
                     SpecialBuild:     False
                     Language:         
                     



```

### Event #6: `item.completed` — `command_execution` (`item_3`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "Get-ChildItem -Force research/stage4_math_and_design; Get-ChildItem -Force research/stage4_math_and_design/M6 -ErrorAction SilentlyContinue; rg -n '"'^#+ '"' docs/LONG_TERM_RESEARCH_PLAN.md docs/MULTIMODAL_INTEGRATION_PLAN.md research/stage3d_qwen_signed_phase/REPORT.md"
```

**Aggregated Output:**

```text
C:\Users\������\Documents\WindowsPowerShell\Microsoft.PowerShell_profile.ps1 : �޷�ʹ�õ� '.' ��ô�������Դ����Ϊ����
�����ڲ�ͬ����ģʽ�¶���ġ�Ҫ���ô�����������������ݣ������ '.' �������
����λ�� ��:1 �ַ�: 1
+ . 'C:\Users\������\Documents\WindowsPowerShell\Microsoft.PowerShell_prof ...
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) [Microsoft.PowerShell_profile.ps1]��NotSupportedException
    + FullyQualifiedErrorId : DotSourceNotSupported,Microsoft.PowerShell_profile.ps1
 


    Ŀ¼: C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design


Mode                 LastWriteTime         Length Name                                                                 
----                 -------------         ------ ----                                                                 
d-----        2026/11/26     19:19                discussions                                                          
d-----        2026/11/26     19:18                M1                                                                   
d-----        2026/11/26     19:11                M2                                                                   
d-----        2026/11/26     19:11                M3                                                                   
d-----        2026/11/26     19:11                M4                                                                   
d-----        2026/11/26     19:19                M5                                                                   
d-----        2026/11/26     19:19                M6                                                                   
d-----        2026/11/26     19:13                __pycache__                                                          
-a----        2026/11/26     19:13           8800 transcript_utils.py                                                  
research/stage3d_qwen_signed_phase/REPORT.md:1:# EM-FloatFormer Stage 3D 实验报告：冻结 Qwen3.5-4B 与 s17 缓存下的有符号频偏显式相位向量盲测
research/stage3d_qwen_signed_phase/REPORT.md:8:## 首页执行摘要与双门槛判定
research/stage3d_qwen_signed_phase/REPORT.md:12:### 主检验格 `G1` 汇总（`|ω| = 0.40`, `h = 2`, `ℓ = 2`, `t_p = 4.0`, `|ω|(h+ℓ) = 1.60 rad`, `σ²t = 0.04`）
research/stage3d_qwen_signed_phase/REPORT.md:24:### 三个预登记连接层种子的各自估计与离散程度（`G1`）
research/stage3d_qwen_signed_phase/REPORT.md:45:## 1. 宿主与编码器锁定、Token 核对、显存故障排查与算力账单
research/stage3d_qwen_signed_phase/REPORT.md:47:### 1.1 冻结组件、词表与 Chat Template 审计（[smoke_audit.json](results/smoke_audit.json)、[PROTOCOL_LOCK.json](PROTOCOL_LOCK.json)）
research/stage3d_qwen_signed_phase/REPORT.md:60:#### 12 个中文候选串逐一 Token 审计（均非空、边界零合并）
research/stage3d_qwen_signed_phase/REPORT.md:68:### 1.2 工程关键排查：Qwen3.5 混合线性注意力引用环与 BF16 批并行数值一致性
research/stage3d_qwen_signed_phase/REPORT.md:78:### 1.3 Modal 免费额度算力账单（全程零付费）
research/stage3d_qwen_signed_phase/REPORT.md:89:## 2. 两臂对照设置、6 次训练过程与样本量盲前锁定
research/stage3d_qwen_signed_phase/REPORT.md:91:### 2.1 六个数值槽设计与零泄露契约
research/stage3d_qwen_signed_phase/REPORT.md:100:### 2.2 6 次连接层训练收敛与开发集标量温度锁定（[training_summary.json](results/training_summary.json)）
research/stage3d_qwen_signed_phase/REPORT.md:113:### 2.3 揭盲前独立开发先导方差检查与 `B_locked` 锁定（[dev_pilot_lock.json](results/dev_pilot_lock.json)）
research/stage3d_qwen_signed_phase/REPORT.md:122:## 3. 主检验与 2×2 探索格结果：累计转角 $\pi/2$ 边界分离机制
research/stage3d_qwen_signed_phase/REPORT.md:141:## 4. 机制诊断 I：Work F 频偏符号干预与单模态矛盾探针
research/stage3d_qwen_signed_phase/REPORT.md:143:### 4.1 固定缓存四格干预汇总（主格 `±0.40` 与历史配置 `±0.35`）
research/stage3d_qwen_signed_phase/REPORT.md:157:### 4.2 具体真实单缓存样本原始分数检验（`G1`, `seed = 202611`, `mother_idx = 3153`, `cache_code = 100`, `z1 = 6, z2 = 4`）
research/stage3d_qwen_signed_phase/REPORT.md:168:### 4.3 `phase` 臂单模态矛盾探针（`G1`, `|ω| = 0.40`）
research/stage3d_qwen_signed_phase/REPORT.md:177:## 5. 机制诊断 II：Work G 旋转等变与离线外部群平均分离报告
research/stage3d_qwen_signed_phase/REPORT.md:195:## 6. 机制诊断 III：Work A 迟到导频 `C4 (ω = +0.35, t_p = 4)` 七臂对比与旧工况回放
research/stage3d_qwen_signed_phase/REPORT.md:197:### 6.1 Q2 真实输入表示与连续参考边界说明
research/stage3d_qwen_signed_phase/REPORT.md:202:### 6.2 Work A 七臂与物理参照表（512 条母波形，3 种子汇总 vs CPU `q2_reader.read`）
research/stage3d_qwen_signed_phase/REPORT.md:216:### 6.3 旧工况 `C1 / C2 / C3` 回放未遗忘验证（`s3_test` × 2 区块 × 3 旋转 = 384 条 × 3 种子）
research/stage3d_qwen_signed_phase/REPORT.md:228:## 7. 历史 Ministral-3-3B 诊断状态、范围边界、复算与核心后续观测
research/stage3d_qwen_signed_phase/REPORT.md:230:### 7.1 实际完成与未完成项目透明清单
research/stage3d_qwen_signed_phase/REPORT.md:235:### 7.2 适用范围与限制声明
research/stage3d_qwen_signed_phase/REPORT.md:242:### 7.3 独立一行复算命令（已于干净临时目录验证通过）
research/stage3d_qwen_signed_phase/REPORT.md:248:# 实测于 %TEMP%\stage3d_clean_verify 输出：VERIFIED OK: max_abs_diff=0.000e+00，退出码 0
research/stage3d_qwen_signed_phase/REPORT.md:258:### 7.4 本轮最值得继续追查的核心观测
research/stage3d_qwen_signed_phase/REPORT.md:266:## 交付清单与指纹锚点
docs/MULTIMODAL_INTEGRATION_PLAN.md:1:# EM-FloatFormer 接入已有多模态模型的计划
docs/MULTIMODAL_INTEGRATION_PLAN.md:5:## 用户拿到什么
docs/MULTIMODAL_INTEGRATION_PLAN.md:11:## 3B–27B 开放权重宿主
docs/MULTIMODAL_INTEGRATION_PLAN.md:28:## 五步与验收
docs/MULTIMODAL_INTEGRATION_PLAN.md:38:## 第一版演示该长什么样
docs/MULTIMODAL_INTEGRATION_PLAN.md:51:## 哪一部分值得投入研究时间
docs/LONG_TERM_RESEARCH_PLAN.md:1:# EM\-FloatFormer 长期研究计划
docs/LONG_TERM_RESEARCH_PLAN.md:3:## 让 AI 从同一段电磁波形中读出关系、指出依据，并随新证据更新答案
docs/LONG_TERM_RESEARCH_PLAN.md:7:### 总目标与双重验收
docs/LONG_TERM_RESEARCH_PLAN.md:19:### 一、我们现在站在哪里
docs/LONG_TERM_RESEARCH_PLAN.md:35:### 二、我们要争取的贡献
docs/LONG_TERM_RESEARCH_PLAN.md:45:### 三、阶段安排
docs/LONG_TERM_RESEARCH_PLAN.md:60:### 四、真实数据怎样用
docs/LONG_TERM_RESEARCH_PLAN.md:70:### 五、实验怎样表明模型读到了波形
docs/LONG_TERM_RESEARCH_PLAN.md:77:### 六、第一轮怎样并行
docs/LONG_TERM_RESEARCH_PLAN.md:93:### 附录 · GitHub 分支快照


```

### Event #7: `item.completed` — `command_execution` (`item_5`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '[Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-Content -Encoding UTF8 docs/LONG_TERM_RESEARCH_PLAN.md; Get-Content -Encoding UTF8 docs/MULTIMODAL_INTEGRATION_PLAN.md'
```

**Aggregated Output:**

```text
�޷��������ԡ�������ģʽ��֧�ֺ������͵��������á�
����λ�� ��:2 �ַ�: 1
+ [Console]::OutputEncoding=[Text.Encoding]::UTF8; Get-Content -Encodin ...
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 
# EM\-FloatFormer �����о��ƻ�

## �� AI ��ͬһ�ε�Ų����ж�����ϵ��ָ�����ݣ�������֤�ݸ��´�

**�о�����Ϊ 2026 �� 11 �� 22 �ա�** ���ƻ����� GitHub ��ʱ�ɼ�������֧�� 17 ���о���֧����֧ĩ���뱨��������ĩ���ֿ������Ը��׶α��桢֤����ʵ������˵��Ϊ���ݣ�����û����ѵģ�ͻ�������ʷ���Ρ�ʱ�䰴һλ�о���Լ 6�C12 ���¹��ƣ��׶�����ɵ��о��ɹ��ƽ���

### ��Ŀ����˫������

Ŀ���������� AI ��һ���������� I/Q ¼����Ϊԭ����֪���롣ģ���ȶ����Σ�����һ�ݴ�С��ȷ���ɻ�ָʱ�䴰�ı�ʾ���û�����������⣬ģ�Ϳ���������ϵ����λ�仯��ָ�����壬���ڵõ�һ�ζ���ο����޶���ǰ�Ļش��о�˳�����ȸ�����֤α������Ԥ�⣬�ٹ�����գ����ѵ��������ڶ����ɼ��ϼ��顣

**�Լ�����㡣** P7 ����Ԥ�������ĵ�Ƶ�������жϣ�P10 �����ָ���ر����������λ����ʧ�Ĺ�����λ��P12 �� A�CE ������о��ظ�����֤�ݡ�Զ������Ѱַ�Ϳɱ�ʶ�½硣��һ��Ҫ֤����֤α��һ��������ǰ�γɵ����ޱ�ʾ���ܹ��ں����ŵ�������ͬʱ�������׷�ʡ���һ������Ŀ���Ǹ�������Ԥ�㡢�ο������������������յĹ�ϵ����������ﵽ�ý��ӽ����ı�ʾ��ֻ�����ߵ��Ͻ硢�½������ƽ�ȷʵ�õ��Ƶ����Ű���Ӧǿ�ȱ��档

**�ⲿ��㡣** RF-GPT ������Ƶ��ͼ�ʴ�MERLIN ���о���������µĵ�Ÿ�֪��������LEMMO ���ܴӳ� I/Q �а�ָ���ȡ֤�ݲ��������źŹ�ϵ��LEMMO ʹ�� Qwen3.5-4B��������ͬ�����Ƚϡ��ֱ𱨸�˫��ԭ���÷����Լ��ڷ���Ȩ�ޡ����澫�ȡ�ѵ��Ԥ���ƥ��ʱ��ʵ�顣Ϊ˫����¼�ض��������ĺ��ӳٳɱ����ⲿ����δ֧�ֵ��ⵥ�У�˫���������ֵ�����������ܱȽϡ�[RF-GPT](https://arxiv.org/abs/2602.14833) �� [MERLIN](https://em-merlin.github.io/) �� [LEMMO](https://arxiv.org/html/2611.09476v1)

**����ä�⡣** �ȱ��룬����⡣���ֽ��;���ɲ�������ʱ��ֻ�����ٶ�ȡһ����¼�Ƶĺ�ѡ�ο�����ģ����ѡ���ڲ�д�¸�����Ԥ�ڵĹ۲⣬�ٽ�ʾ��ʵ���β����֡����հ����������������Լ�������ģ��Ԥ�������̶ȼ�������Ĵ�����ǰԼ����Ҫָ�ꡢ�ɽ��ܵ������Ͳɼ����֡��Ȼ�����۲�Ԥ��ʱ���Ƚϴ𰸷��ա����ŶȺ�֤�ݶ�λ���ﵽԼ��׼ȷ��ʱ���Ƚ����軺��Ͷ����ȡ������Ȼ�۲⡢�˹�ע��Ͷ����ɼ��ֱ𱨸档�����з�����ƽ����������֤�ı߽��뷴��������Ѱ����һ��ȷ�в�������⡣

��������������������ѧ�������������ֵ�Ͷ����ɼ���ä�⼯���ɽ��������� AI �ĵ�ű��������Լ�һ��֧�ֶ���׷�ʵ���ʵ¼����ʾ��[3B�C27B ������ģ�ͽ���ƻ�](MULTIMODAL_INTEGRATION_PLAN.md)

### һ����������վ������

��Ŀ�Ѿ��ߵ�**����Ϊ���ģ̬�涨��һ�������������������о�**�ĵط�����ǰ�Ľ����о���������λ�ο��������۲⡢��֡��ϵ����ѡ���ͺ���ʵ�ɼ�����Щ��������������һ���ȡ�������������𡱸�ֱ�ӵ����⣺ͬ�������룬��Щ���������ܴ���ЩҪ����һ��֤�ݣ�����ʱ�ӵ�ʲô���ý����Ĵ����ö�ʧ��

|�����о�                 |�����еķ���                                                                                                                                                                                                                           |���ڼƻ��̳�ʲô                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|P1�CP3 �ı�ʾ���ܹ�������Լ���븺����|P1 �������� CNN ������İ���ŵ��Ϸ���һ�µ�����ͬԤ�� Transformer��У�����Ϊ p=0.125��P2 ����λ�ȱ�ģ�͸�����һЩ�ϳ����������� DFT �Ժ�ǿ��P3 ȥ������ʱ�����벢�ӱ������ϡ�賤ʱ�� BER �� 12.291% ���� 6.549%��Ӳ��ֵ DFT Ϊ 4.919%��������ϡ�赼Ƶ��RMS ���롢����ŵ�����ֵ���ȡ�������ʧ���� 16-QAM ̽����P3-D/E �ı�׼�ŵ��Ž�ʵ����ɵ��Ǿ�̬ TDL ���ơ�|�������롢��λ�ȱ������ƥ������ľ�����գ��Լ�ʶ��λ�ýݾ��ķ����� [P1�CP3 ����](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/README.md) �� [λ�ýݾ�](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/P3_A_MECHANISM_REPORT.md) �� [TDL ���Ƶķ�Χ](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/exploratory/p3de_tdl_surrogate/P3DE_SURROGATE_REPORT.md) �� [16-QAM ̽��](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/exploratory/16qam_joint_pa_probe/REPORT.md)                                                                                                                                                                                                             |
|P4�CP6 ������ P6 ����      |P4 ��ʮ���Ӷ�άģ��û�дﵽԤ���������ơ�P5/P5-B/P6 �𲽶��ո������ӡ��������ơ���żУ���ѧϰģ�顣P6 ��ָ��ģ����ȡ�����棬�ֲ���ע���������Բ����Խ�������������ע���������������ֱ����˲������������ԡ����������ʧЧ��Ƶ�Ĺ۲�ģ�͡�                                                                                                  |���������롢�������������������Ĺ��׷ֿ�����¼ȱʧ����������ֵ֮������� [P4](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/p4/STAGE3_RESULTS.md) �� [P5](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/p5/P5_REPORT.md) �� [P5-B](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/p5b/P5B_REPORT.md) �� [P6](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/p6/P6_REPORT.md)                                                                                                                                                                                                                                                                                            |
|P7��P8 ��������           |P7 �����ĸ���Ƶ����Ԥ���ٽ�ʾ���߷�����صĸ��ʸı����ǵͷ������ 5.52 ��������ֻ���ٴ�������顣P8 �ֱ𿼲����� OFDM ֡�͵�Ƶλ�á��� OFDM ����״̬�����ز����Լ�����µ���λ���䡣�粨�κϳ�ʵ����ϡ��ѵ�������£���ȷ��ʷ�������ʷ��������ʷ�� BLER �ֱ�Ϊ 54.84%��89.47%��98.71%��                                                             |�Ѿ��С���֤��Ӧ���ı���Щ�𰸡���ʵ��ԭ�ͣ�Ҳ�С�����ο������ʲô������Ķ��ա� [P7](https://github.com/20171806011/EM-FloatFormer/blob/4dab1ac9ee15b4f1a7f78ae05486c55a2dc3bd72/p7/P7_REPORT.md) �� [P8 ����֡](https://github.com/20171806011/EM-FloatFormer/blob/4e7ccdaaacc14cd4101a6f5b824d9974a9255b58/p8/P8_REPORT.md) �� [P8 �粨��](https://github.com/20171806011/EM-FloatFormer/blob/ce2ff50e81f6b529ec0311393da0013b0a85b801/p8c/P8C_REPORT.md) �� [P8 ���](https://github.com/20171806011/EM-FloatFormer/blob/e76c61eef26151985963460b0ab03684cd562e3e/p8_manifold/P8_MANIFOLD_REPORT.md)                                                                                                                                                                                                                                                                   |
|P9�CP11 �Ŀ�֡������δ֪�غ�    |P9 �����ɹ۲��Ƿ�ɿ�ȡ���ڵ�ǰ�ŵ���P10 �����ڸ���ر���·����������λ��ͬʱ��ȥÿ�й�����λ���ڹ涨�ĺϳ���������ʵ���У��ؽ���֧�Ȼ����־ɷ��������� 1,604/2,666 ����飬Ҳʹ 115 �����߾��ԵĿ�����P11 ��֤�˾���У����������ܲ���������ơ�                                                                                        |����ָ��ĳ��ͳ��������������������ϵ���ֶ���������Ϣ������һ�θ���ʱͬʱ���¾����������Ĵ��� [P9](https://github.com/20171806011/EM-FloatFormer/blob/faeea639727aab72740cdd0a719106f83824226e/p9/P9_REPORT.md) �� [P10 �����ʵ��](https://github.com/20171806011/EM-FloatFormer/blob/9e9d93bcbf6345b8645314b223cd954b2f31e63a/p10/payload_reconstruction/README.md) �� [P11](https://github.com/20171806011/EM-FloatFormer/blob/307d62fd89206442960fa0cebdcdc6a0530db366/p11/P11_REPORT.md)                                                                                                                                                                                                                                                                                                                                                                                 |
|P12��������ѧ�о��� Stage 5  |P12 �޸�ͬһê�㱻�ظ���������⡣Զ����Ч�ο��ɴ�����ʶ��ʱ��ȫ��ȡ֤�����ϸ����ڹ̶��ֲ�������ȷ�����ںϸ���ƽ�����գ��ϸ���Ʊ����о���ȡ�����о��߽硣�޸���� Stage 5 �ڷֲ��ںϳ������£����ߡ�������ϡ��ӵȱ������ BLER �ֱ�Ϊ 48.94%��41.58%��39.54%���ֲ�����������Բ�ȷ����                                                                 |ȫ����֤�ݡ�����������͡�һ�ι۲�ֻ��һ�Σ��Լ�������ֱ�����ʧ�� [P12 �޸�](https://github.com/20171806011/EM-FloatFormer/blob/f566f9b8b0ddfb468a4cb3fd64588bfa46c87212/p12/P12_CONSISTENCY_FIX.md) �� [����Ѱַ](https://github.com/20171806011/EM-FloatFormer/blob/f566f9b8b0ddfb468a4cb3fd64588bfa46c87212/research/content_addressing_scalar/REPORT.md) �� [�����ں�](https://github.com/20171806011/EM-FloatFormer/blob/f566f9b8b0ddfb468a4cb3fd64588bfa46c87212/research/posterior_fusion/pages/01.md) �� [Stage 5](https://github.com/20171806011/EM-FloatFormer/blob/bfe942d94d04004c40575bbb62e57c7e36f3b3e7/research/p12_stage5_fusion_neural/REPORT.md)                                                                                                                                                                                                  |
|����ʵ���֧������������Դ        |435 MHz ������֧ʹ��ͬһ�� 742 ֡ 16-QAM OFDM ¼�ơ�138 ֡�� 240 ֡�ǲ�ͬ���֡�Pluto �ṩ����ʱ�� QPSK��CTTC �ṩ��Ự QPSK�����п��� Wi-Fi CSI �ļ��ĸ�ʽ���ظ�����ơ�                                                                                                           |�ڲ�ͬ�ź�������Ϸֱ���֤�ɻش�����⣻����ʱ�ԻỰ������¼��Ϊ��λ�����������̵㹩����ѡ�⡣ [435 MHz OFDM](https://github.com/20171806011/EM-FloatFormer/blob/ba39dd5968dbaee7b7a226a9bc78d536278fdee8/research/realdata_sdr_ofdm/REPORT.md) �� [ѹ��ʵ��](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/REPORT.md) �� [Pluto](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/REPORT.md) �� [CTTC](https://github.com/20171806011/EM-FloatFormer/blob/1887aee6de9649bc0dd8a4f05fe3c4542f3f6d77/research/realdata_cttc_qpsk/REPORT.md) �� [�����������](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/DATASET_DISCOVERY.md)|
|11 �� 22 �� A�CE ���     |A/B �˶Ժ����������λͬһ�۲ⱻ�ظ����õ����⡣C ������Ƶƫ�����ڲ�������������棬�������Ͽ�������D ������λ�����ɱ�ʶ���������½硣E ������ʱֵ���л�������������������ʵ�� 14 ֡�ϳɼ���ȹ̶������������ 172 λ����ϱ����е� 435 MHz �����Ƕ���ʷ��֡��¼�ĸ��㡣                                                                             |�������� D������ A/B ��֤�ݺ����� P7 ��Ԥ�⡪��ʾ��ơ�Ƶƫ�������ſ���Ϊ��������Ĺ��ߡ� [������](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/README.md) �� [D ��֤��](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/D/PROOF.md) �� [ʵ�������븴��](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/experiments/historical/REAL_INPUTS.md)                                                                                                                                                                                                                                                                      |

**��ֱ�ӵ������ D ���ͬһ�β��Ρ�** ��֪�����ŵ�������δ���� 16\-QAM�����ֵȸ��ʷ������ǡ������޶���ʱ����֪����֮���������ŵı�ֵ�Կɾ�ȷ���㣬ÿ��ԭʼ��ǩλ������ƽ��������ȴ�� 3/7��������Ҳ���أ�����һö��֤������������֪���㵼Ƶ����֡���� BER �� 24\.7347% ���� 2\.0612%�����������ȷ��ʱ���Կ��������塣�����Ѿ����Զ�ͬһ���ź�����׷�ʣ������˵����ƾʲô������[������](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/README.md) �� [D ��֤��](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/D/PROOF.md) �� [ʵ�������븴��](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/experiments/historical/REAL_INPUTS.md)

### ��������Ҫ��ȡ�Ĺ���

���� RF\-GPT ��Ƶ��ͼ���������ź��ʴ�MERLIN �о�����źŸ�֪���������������������LEMMO �ѱ������˰������������ԭʼ I/Q �а�ָ�����֤�ݣ����������źŹ�ϵ����ƾ������ I/Q����������ģ�͡��ܻش����⡱�����Խ�������Ŀ��ԭ���ԡ�ֵ����ȡ�Ĺ����ǣ������ȷ��һ���������⣬����**���ޱ�ʾ������ʲô����ʱ��Ȼ��ʧ�𰸡�ʲô����۲���Իָ���**�Ķ��������������ʵ�����ϼ�����ЩԤ�⡣����ӽ�ģ�͵�ͬ��ͬ��Ϣ�ԱȽ��������������ǿ��ʵ֤���š�[RF\-GPT](https://arxiv.org/abs/2602.14833) �� [MERLIN��IEEE/CVF Conference on Computer Vision and Pattern Recognition 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_MERLIN_Building_Low-SNR_Robust_Multimodal_LLMs_for_Electromagnetic_Signals_CVPR_2026_paper.pdf) �� [LEMMO](https://arxiv.org/abs/2611.09476)

**��һ����ͬ�ӿڰ���ѧ��ԭ�ͽ�������** ģ�Ͱѹ۲⵽�ĸ����� &#40;Y&#41; ��Ϊ &#40;Z=T&#40;Y&#41;&#41;������һ������ &#40;q&#41;���ٽ�ʾ�����Ĳο� &#40;A&#41;����ͬһ�� &#40;Z&#41; ׷�ʡ�ʵ��ʱ�涨�ܷ��ض�ԭ���Ρ��洢��������������Ԫ���ݺ͸��������ʧ����ÿ������ֱ�Ƚ��� &#40;&#40;Z,A&#41;&#41; ���� &#40;&#40;Y,A&#41;&#41; ���ܴﵽ�����ŷ��ա����� I/Q �Ǳ�����Ϣ�Ĳ��ա�����ʵ�ɼ������ŷ���ͨ��δ֪��������֪�ο��������Ĳ��κ���ȷ�Ľ������������֣��ѿɼ��������ֵ��������ģ����ȷ��ʵ�顣

**��һ�����С�����㡣** ������������������� &#40;X\_1,X\_2&#41; ����ͬһδ֪����������ת &#40;G&#41;���õ�����۲� &#40;Y\_k=GX\_k&#41;�������ʶ��ߵ������λ������ &#40;R=Y\_2\\overline\{Y\_1\}&#41; ������״̬���ɣ�ǡ�� 2 ���ء��Ժ����ͬһ��������ڵ�λ��Ƶ�Ĺ۲� &#40;P=G&#41;�����ʵ�һ�����ŵ�ԭʼ�����ǩ����ʱ &#40;X\_1=\\overline P Y\_1&#41;������һ��ֻ���� &#40;R&#41;������ &#40;P&#41; �������ĸ��ȿ��ܴ𰸡���ͬʱ���� &#40;&#40;R,Y\_1&#41;&#41;�����ⶼ�ܴ�ԣ�ǡ�� 4 ���ء���һ��Ҳ���½磺�̶� &#40;P=1&#41; ʱ��16 �ֹ۲�Զ�Ӧ 16 �ֲ�ͬ�����ϴ𰸣��κι̶����ȡ����������ض����εı��붼���������ǡ����ƻ�����ö���� 64 �ֵȿ���Ǳ��״̬���˶���������ӡ����Ǻ����о��ĳ����㣻��������������롢������ɺ�����Ԥ���µ��ƹ㡣

**��һ��ֵ�����湥���ƹ㣬�ǡ��ο��ڶ�Զ֮�⻹���á���** ��������ʱ�䴰����ͬһ��λ״̬�ĸ��ʻ�һ���ɲ����λƯ��ģ�ͣ����������ڼ�����ı�ο�������Ⱥͱ�ʾԤ�㡣ϣ���Ƶ������������ߣ�һ����Ӧ�細�����λ��һ����Ӧ���ο���ķ��Ż��¼��жϡ������ֱ�ʾ�ڵ�һ�������ϴ�ƽ��ȴ�ڵڶ����ϳ��ֿ�Ԥ��Ĳ�࣬�͵õ��˶Ա�ʾ�����֡�������״̬�ļ�������ʵ�ɼ���ʧЧ�����ʧЧ���䱾������Լ������ģ�͡�

### �����׶ΰ���

|�׶���Ԥ��ʱ��                    |ģ�ͽ���ش�ʲô                                     |�о����������������                                                                                                                                                            |
|---------------------------|---------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|1 �� �� 1�C3 �ܣ������ɱ�ʶ�߽�        |�������ź����ת�˶��٣����Ҳο�֮����ǰ�Ǹ�������ȷ���𣿡�              |������ģ�ͺ� D �� 16-QAM �Ƶ���ʼ���ֱ�д��ֻ�е�ǰ���Ρ�����ͬ�������ο���ֻ�������ο�ʱ�Ŀɱ�ʶ�����������鱻ĳ�ֱ���ѹ��ͬһֵ����׷�Ӳο�����ȷ�𰸲�ͬ�Ĳ��Ρ�������֤��������������汾�ɼ����Ԥ�⡣����ͳ�Ƴ���ԺͶԳ��Խ����������ע����                                      |
|2 �� �� 4�C6 �ܣ�����С���б�ʵ��       |����һ�ι۲�֧���ҵ��жϣ�ȥ��������һ���𰸻�䣿��                   |�ù���ĸ����������Ч�ο��������ο����ظ�ͬһ�ο����޲ο�������ԡ�������������ֻ�з��Ȼ��𴰹�һ���ķ�������������λ�ķ�����ȫ��ͬ������Ȩ�ޡ�ƽ������ȡ���λ�������ͱ�š��þ�ȷö�ٻ�ɿ���Ȼ��֤���ۣ��������⡢���ֺ����֡�������� CPU ���ܵ�С��׼��                                      |
|3 �� �� 2�C3 ���£��������������״�ģ�ͽ���    |��ͬһ�ݱ�ʾ�ܴ���Թ�ϵ��Ԥ�������۲⣬���ڲο����������������𣿡�           |������Ԥ���һ�������������伸���򵥶��������������/�������顢ͬԤ��ʵ������ȶ��ա������ʾ������²ο�λ�á���λ�켣��δ��������ϣ���¼ÿ����Բ��յ����ܲ��ͱ�����Ϣ�ĳɱ����Ƚ� Ministral 3B���ٽ� Qwen3.5-4B�������������������Ϊÿ������ѵ�����Ӳ㣬����ͬ����ֱ��ֱ�Ӷ��������Իش�                                   |
|4 �� �� 3�C6 ���£�����ʵ¼���϶���һ��Ԥ��  |����İ����һ�βɼ��У��Ķ���֪��������Ч�ο�����ʾ����ǰ��Ԥ��Ӧ������������      |�������� Pluto �� CTTC ����֪������ס����ʾ��A �βɼ��������B �βɼ����顣CTTC �������Ự��Pluto �ɼ������ʱ�򴰣�435 MHz ����ɼ���ͬ��� OFDM �汾����ȱ����ֵ������ʹ��������ƵԤ��򹫿����ж���𰸡��ֱ�㱨��Ȼ¼�ơ�����ʵ����˹���Ԥ�������Դ�������ܴ�ԭ�����طŵ�����һ�������⡣|
|5 �� �� 6�C9 ���£�����λ�����¼��͹�ϵ    |��������������ı仯����ͬһ����λ�켣�����𣿱仯��������һ�Σ���һ�ֽ��ͻ�ʣ���ٿ��ܣ���|��Ϊƽ��Ư�ơ���ɢ����������������̹����������۲�ȼ۵İ��������Ƶ����߼������������ͬһ��ʾ�Ƚ����β��Σ����غ�ѡ���͡�֧��ʱ�����������Ŷȡ��� QPSK/OFDM ֻǨ������������ͬ�����⡣����������ȱ����Ȼ�¼���ֵ�����������֤����ʵ����Ԥ�⣬���������־������͵Ĺ����ɼ���                         |
|6 �� �� 9�C12 ���£��γɿ�׷�ʵ�ģ̬ԭ�ͺ�����|������һ����¼�ƣ������������ݵı仯�����֤�ݡ��ٸ�����һ�Σ���Ҫ�޸�ǰ���ľ仰����   |�� 3B��4B ԭ������չ�����¼��ʴ𣬼��� Qwen3.5-9B �� Qwen3.8-27B������Դ���ӿ������ա�һ��¼��֧�ֶ����ʴ��������ʱ�䷶Χ���۲����ݡ��ɱ�ʶ��Χ�Ͳ�֤����޶�������δ��¼�Ƽ�δ�����������ä�����ڹ�֧ͬ�ֵ��������빫��ģ�͡����䷽���ͼ򻯰�Աȡ��������۸塢�ɸ��ܵ�С��׼��һ����ʵ¼����ʾ��                                        |

���������£�Ŀ���ǽ���**һ���м������Ϣ�½硢һ�׿����ж��պ��׸���������ģ�͵Ĺ���������**�����������£�Ŀ�������ⲿ��������ʵ¼������һ����ڡ���Щ�𰸻���֤�ݸı䡱��Ԥ�⡣����꽨���ڸý�����н������ķ����ϡ�����ɼ�������������δ���壬ʱ�����ȸ�������ϡ�Ӳ���ɼ��ĺ���ʱ�㣬�������Ѿ�˵����ȱ��һ�ֹ۲���������������ͣ������й���¼��ǡ��û������

**һ���Ժ�ķ���ȡ������Щ��������Ǩ�ơ�** ���ͬһ��ʾ���ڶ�����Դ���ȶ������λ���¼��ʴ����ƽ������ÿ�������¼�ƣ��ӡ�һ���źŷ�����ʲô����չ������γɷֺ�ʱ���֡���Щ�仯ͬ������һ��֧��������͡����Ƚ����ɹ۲���������ʵ��ǩ���ټ��������ʡ������벨����ʽ��Ǩ�ơ���ʱ��ֵ��Ϊ�����ʴ�����ѵ�����Ϻ�ģ��������

### �ġ���ʵ����������

|����                                                   |���������ĵ�һ���¼���                                                       |�ƶϷ�Χ                                                                                                                           |
|-----------------------------------------------------|------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
|Pluto 850 MHz ���� QPSK��7,096 ֡��ÿ֡ 7,560 �������������㣬����ƥ���˲�|��Դ�����пɺ˶ԵĲο����У�����һ����Ԥ������λ����ʾ�ô����ж���һ������Ԥ���Ƿ���ơ��������֡�ź͹̶��ظ������ܽ��Ͷ��ٽ����|�ܼ���ʱ������ԡ��ظ��ṹ���ο����£����豸��������Ҫ�����ɼ���                                                                                               |
|CTTC ��Ƶ�Ρ�������������ȡ��Ÿ� QPSK �Ự����ԭʼ ADC ��ٷ�ƥ���˲���¼        |�����Ựֻ�����ԡ��ڶ���ɹ���Ƭ���ϣ��Ƚ�ԭʼ���˲�����ͼ��ͬһ��ϵ����Ļش��Լ���Щϸ���ڴ�������ʧ��             |������ͼ����ͬһ�βɼ����ԻỰ����Դ���ܡ������о��ǻỰ�ڻ��֡�                                                                                               |
|435 MHz 16-QAM OFDM��ͬ����ȥѭ��ǰ׺������Ҷ�任��ĸ�������             |�������е�Ƶ����ȷ�����ֵ�����λ������ס����ʾ��֮��ż��ܿ�Ƶƫ����λ���䡣                            |������ݿɼ��������ϵ���λ֤�ݡ���ʷ 240 ֡ B3 �Ա�ѧϰͷ����ԭʼ���� BER Ϊ 9.1534% �� 10.6077%����ע��Ƶƫ����Ϊ 26.7733% �� 14.4038%��11 �� 22 �ջ�ϸ�������ʷ����δ��������ʵ�� I/Q ������|

����ģ�ͻش�û��������������⡣���硰���������Ƿ���ͬ���������Ҫ�����������ֵ��ǡ���۲⣻���ƻ����ʡ��Ƿ����һ��������λ�켣������˵�����ģ�͡���һö�����ȫһ������Դ�����صĴ��䵼Ƶ��ģ�����ж�ȷ��ȡ���������������������顣��֪�����ο������ؿɿ��ԵĲο�Ҫ�ֱ�ģ���ֱ����֡�[435 MHz OFDM](https://github.com/20171806011/EM-FloatFormer/blob/ba39dd5968dbaee7b7a226a9bc78d536278fdee8/research/realdata_sdr_ofdm/REPORT.md) �� [ѹ��ʵ��](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/REPORT.md) �� [Pluto](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/REPORT.md) �� [CTTC](https://github.com/20171806011/EM-FloatFormer/blob/1887aee6de9649bc0dd8a4f05fe3c4542f3f6d77/research/realdata_cttc_qpsk/REPORT.md) �� [�����������](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/DATASET_DISCOVERY.md)

### �塢ʵ����������ģ�Ͷ����˲���

- **�������֡�** �����λ��Բ����������׼ȷ�ʣ�����������Ԥ���ù�һ����������ѡ�����ø��ʶ�����ʧ��У׼�����㶨λ��ʱ�����򸲸����䡣���� BER/BLER ���������񸴺ˡ�
- **��֤���ݡ�** �������õ���ͬ�ɼ���Ϣ���滻����סģ�����������Ĵ��ڣ����滻��ͬ���ȡ������������޹ش����Ƚϴ�ʵ�ʱ仯��������֤�ݵ���ȫ�ظ�����ȷ�ο�����Ӧ����һ�¡������ֳ����ɱ�ʶ���ޣ������Ų��ǩ���ظ��غɺ�����Ԫ���ݡ�
- **�������¡�** ��ͬһ�����ϲ��ż�¼��һ�λش���Ч������ο������ĸı䡢��Щ�𰸱�Ի������Ự�������ЧӦ�����䣬����Խ�����ʱ�䡢������ȡ�ͬ��ʧ�ܺͺ�ѡ�����ѷֵ�������
- **���Ƽ��㡣** ͬʱ�������� I/Q ���գ��Լ�����ͬ����λ����ͬ�������ض����ֹ�ض������µĽ�����ѧϰ���������в��������ӳٺͲ�ѯ�������������ʾ�ܴﵽ���۷��գ������Ǻ������뱣���Ķ��ա�

### ������һ����������

��������һ���������������������ͳһΪ���������ģ�ͺ� D �鹫��ģ�͡���ݽ����Ϻ���������ʽѵ����

|������        |���Զ�����ɵĲ���                                      |���ʱ��ʲô                   |
|-----------|-----------------------------------------------|-------------------------|
|1 �� ��ѧ     |�Ƶ�����Ͳ������ʱ����ǰ������׷�Ӳο�������Ե����ŷ��ջ��                |һ���м�������⡢֤����ɸ�������        |
|2 �� ���������й���|����ֻ�����ȡ�����ת��һ���������ŵ���������ķ������˶���ӽ�������ͨ�š�ͳ�������ʴ�����|���Ʒ���ǿ���ŵ��źŶԣ��Լ�����Ŀ�ܶ������������|
|3 �� ��������   |ʵ��������١������ʾ�������ο����ظ�֤�ݵľ�ȷ�⣻�� P7 ����ס����ʾ�ӿ�������     |ͬһ��΢�������Ĵ𰸡����������ɱ�       |
|4 �� ʵ�������  |���� Pluto��CTTC ��Щ�ο����С�������ͼ�ͻỰ����ԣ�435 MHz ��¼�ɼ�������|�����⡪���롪��ֵ�����֡�������һ���ɸ���������ʶ|
|5 �� ����     |�̶������ʽ�������ʽ���ƶ�ƥ�䵥��ͳ�ƵĶ��ա��Ự������֤���Ŷ��Ʒַ���          |���з�������ʹ�õ�һ��ä��Э��ݰ�        |

��һ�λ��Χ��ͬһ�����ζ���������£���ѧ���г����ų��Ľ��ͣ���׼�������ȷ���ʣ�ʵ����ָ��ͬ���Ĺ۲�����ʵ��¼���ܷ�ȡ�á������з������խ���衣��һ�����԰�һ����Ȥ���źŴ��������ɵ�һ���ɹ���������Ŀ�ѧ���š�

**��ֹ�о���Ȧ�ӵ�Լ����** ÿ�ֿ�ʼдһ�䡰�����Ժ����ͬһ�β��Σ�ģ���ܶ�ش��ĸ����⡱��ÿ�ֽ����г�һ���ɹ�������һ��ʧ�������ͻ���֮�ı���о����������������ֶ�ֻ�ܱ���ͬһ�� BER ָ���С���仯���ͰѸý����Ż��鵵���ص���δ�ش�����⡣�����۸������ɱ�ʶ���ۣ�����������ۺ�����Ҫ�����С�����۲⣬��Ҳ�����߳ɹ���

### ��¼ �� GitHub ��֧����

�������ӹ̶�������ʱ����֧��ĩ���ύ�����¿ɼ��ύΪ 2026 �� 11 �� 22 �� 19:47���׶�ʱ�䣩�������ϡ�����֧ĩ��Ϊ 11 �� 18 �գ������ֿ���ҳ��©�������׶Ρ�

|��֧                                              |�о�����                    |�̶��汾�ı���                                                                                                                                                   |
|------------------------------------------------|------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
|`main`                                          |P1�CP6 �ϳɽ����о��������ڲ���͸����  |[63ad2ec5](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/README.md)                                         |
|`research/p6-math-followup-2026-11-19`          |ƽ������ѡ���벨�λ���             |[c699680d](https://github.com/20171806011/EM-FloatFormer/blob/c699680dd84560747ddf40ecacd59821987bf545/p6_followup/REPORT.md)                             |
|`research/p6-soft-information-2026-11-19`       |����������������λ����             |[bb9396f0](https://github.com/20171806011/EM-FloatFormer/blob/bb9396f08132bb5f586ff728dcfa4e7872201c7b/p6_softinfo/REPORT.md)                             |
|`research/p6-observation-consistency-2026-11-19`|ʧЧ��Ƶ��۲�һ����              |[a4e66848](https://github.com/20171806011/EM-FloatFormer/blob/a4e6684830f68e6dfe715470dafb4ff95972e508/p6_consistency/REPORT.md)                          |
|`research/p7-multihypothesis-routing`           |�������������Ƶ����              |[4dab1ac9](https://github.com/20171806011/EM-FloatFormer/blob/4dab1ac9ee15b4f1a7f78ae05486c55a2dc3bd72/p7/P7_REPORT.md)                                   |
|`research/p8-continuous-ofdm-tracking`          |���� OFDM ֡��״̬�뵼Ƶ����       |[4e7ccdaa](https://github.com/20171806011/EM-FloatFormer/blob/4e7ccdaaacc14cd4101a6f5b824d9974a9255b58/p8/P8_REPORT.md)                                   |
|`research/p8c-ofdm-single-carrier`              |OFDM �����ز��Ĵ���״̬����        |[ce2ff50e](https://github.com/20171806011/EM-FloatFormer/blob/ce2ff50e81f6b529ec0311393da0013b0a85b801/p8c/P8C_REPORT.md)                                 |
|`research/p8-coherent-transport-manifold`       |�����λ������������ʾ֤��           |[e76c61ee](https://github.com/20171806011/EM-FloatFormer/blob/e76c61eef26151985963460b0ab03684cd562e3e/p8_manifold/P8_MANIFOLD_REPORT.md)                 |
|`research/p9-cross-waveform-trust-gating`       |���ŵ��Ƿ��Կ���                |[faeea639](https://github.com/20171806011/EM-FloatFormer/blob/faeea639727aab72740cdd0a719106f83824226e/p9/P9_REPORT.md)                                   |
|`research/p10-payload-channel-reconstruction`   |δ֪�غɸ����ŵ��ؽ�              |[9e9d93bc](https://github.com/20171806011/EM-FloatFormer/blob/9e9d93bcbf6345b8645314b223cd954b2f31e63a/p10/README.md)                                     |
|`research/p11-wavefront-payload-assimilation`   |����У��������в������            |[307d62fd](https://github.com/20171806011/EM-FloatFormer/blob/307d62fd89206442960fa0cebdcdc6a0530db366/p11/P11_REPORT.md)                                 |
|`research/p12-first-principles-diagnostics`     |�������޸�������Ѱַ������ںϵ���ѧ���   |[f566f9b8](https://github.com/20171806011/EM-FloatFormer/blob/f566f9b8b0ddfb468a4cb3fd64588bfa46c87212/research/content_addressing_scalar/REPORT.md)      |
|`research/p12-stage5-offgrid-neural`            |P12 �޸�����������ѧϰģ�Ͷ���      |[bfe942d9](https://github.com/20171806011/EM-FloatFormer/blob/bfe942d94d04004c40575bbb62e57c7e36f3b3e7/research/p12_stage5_fusion_neural/REPORT.md)       |
|`research/realdata-sdr-ofdm`                    |435 MHz OFDM ����ʵ��������У��  |[ba39dd59](https://github.com/20171806011/EM-FloatFormer/blob/ba39dd5968dbaee7b7a226a9bc78d536278fdee8/research/realdata_sdr_ofdm/REPORT.md)              |
|`research/realdata-ofdm-theory-and-stress`      |435 MHz ͬһ¼�Ƶ�ѹ���о�����������̵�|[5f88a5be](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/REPORT.md)|
|`research/realdata-pluto-qpsk`                  |Pluto ���� QPSK           |[ade8333f](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/REPORT.md)            |
|`research/realdata-cttc-indoor-qpsk`            |CTTC ��Ƶ�ξŻỰ QPSK        |[1887aee6](https://github.com/20171806011/EM-FloatFormer/blob/1887aee6de9649bc0dd8a4f05fe3c4542f3f6d77/research/realdata_cttc_qpsk/REPORT.md)             |
|`research/abcde-convergence-2026-11-22`         |������ѧ��������ʵ��             |[e914befb](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/README.md)   |
# EM-FloatFormer �������ж�ģ̬ģ�͵ļƻ�

**����Ŀ��״�㣬�����õ��׸����ʴ�Ľ���ԭ�ͣ����Ĳ��ڶ���ʵ�������գ����岽����ͬһģ���ܷ�������** ԭ���ڼƻ������пɱ�ʶ�ԡ���������������ʵ���������������������һ����ͨ�������������⣬�Ϳ�����ǰ������ģ�͡�ԭ�ƻ��������������Ĳ��Ρ��¼���ϵ�Ϳ���ʽ׷�ʡ�[�����о��ƻ�](LONG_TERM_RESEARCH_PLAN.md)

## �û��õ�ʲô

�û��ϴ�һ�θ����� I/Q ¼�ƣ�������ʱ�䴰֮����ʲô��ϵ��ϵͳ�Ѳ��α���һ�Σ�����һ����ʱ�䷶Χ��Ӧ�������������û����Լ���׷�ʣ�Ҳ���Ը���һС�����յ�����֪�ο��źš�����ģ�͵�ע������ͬʱ��ȡԭ�����������²ο����û������֡���ͬʱ����һ�ű�עʱ�䴰��ͼƬ�������Լ���ģ���ܷ��ͼ�ϵı�����ź��е�֤�ݶ�Ӧ������

�׸�����ѡ **Ministral 3 3B Instruct��BF16��**����ʽ�о����μ��� **Qwen3.5-4B��Qwen3.5-9B �� Qwen3.8-27B**��Ministral �п��ŵ�ͼ��Ȩ�أ���������֧�������������������������������������в��벨����������Ϊ�������ֱ�ѵ��СͶӰ�㡣���ƻ������й�ģ�ͣ�ԭ������ȡ���ڷ����Ƿ񿪷��Զ���ģ̬���롣[Ministral �ٷ�Ȩ��](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16) �� [Transformers �������ɽӿ�](https://huggingface.co/docs/transformers/model_doc/ministral3)

## 3B�C27B ����Ȩ������

���� 2026 �� 11 �� 22 �գ����º�ѡ����ԭ���߹�����Ȩ�ء���Χ���ٷ��ͺŵ�λ������Ӱ���豸Ԥ��������ͼ�����������ָģ�ͻ�ϵ���׷���ͬһ�ͺŵľ��ȱ�������Ժ��ϼܡ�

| ��λ������ | ����ʱ�� | �� EM-FloatFormer �лش������ |
|---|---|---|
| **3B��8B��14B �� Ministral 3 Instruct** | 2025-12-02 | �� 3B ����֤�����Ի����̡�����������Լ 3.4B���Ӿ�������Լ 0.4B����������ͬϵ�е� 8B��14B �����ģ���ơ� [3B BF16 Ȩ��](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16) �� [8B Ȩ��](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512) �� [14B Ȩ��](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512) �� [�ٷ�����](https://mistral.ai/news/mistral-3/) |
| **3B �� IBM Granite 4.1 3B LLMON Preview** | 2026-11-12 | �������������ա����������һ�������ֳ�����ģ������ I/Q �����Ժ���Щ��Ҳ��ѧ�ᡣʹ�ø�ģ���Լ��ĶԻ���ʽ�������Ԥ���浥�С� [IBM Ȩ��](https://huggingface.co/ibm-granite/granite-4.1-3b-llmon-preview) |
| **4B��9B �� Qwen3.5** | 2026-03-02 | 4B ���׸�������ա�����ԭ�� I/Q �о� LEMMO ʹ��ͬ�� 4B ���Ե������� LoRA ���䡣��������ͬ���ȡ���ͬ�������ͬ�ɼ��ο��±Ƚϡ���Ҫ���򵽱��뷽������Ҫƥ��ѵ�����ݡ���ѵ���������Ż�Ԥ�㡣4B �� 9B �ɱȽ�ͬ����ģ���ơ� [4B Ȩ��](https://huggingface.co/Qwen/Qwen3.5-4B) �� [9B Ȩ��](https://huggingface.co/Qwen/Qwen3.5-9B) �� [LEMMO ���ĵ�����](https://arxiv.org/html/2611.09476v1) |
| **9B �� SenseNova-Loop-9B** | 2026-11-17 | ��������ͼ�ĺ�ѡ���ٷ���¼Լ 9B ������ͨ�������м�����Ӽ�����ȡ��� Qwen3.5-9B ʹ����ͬ���Ρ�ͬһ�Ʒַ�����ͬʱ��¼ʵ���ӳ١� [����Ȩ��](https://huggingface.co/sensenova/SenseNova-Loop-9B) |
| **12B �� Gemma 4 12B IT** | 2026-06-03 | �����Ǩ�ƶ��ա�����ͳһ�ܹ�֧��ԭ��ͼ�����Ƶ���룬����ǰ����ģ����ͬ�ĵ�ű�����������һ�������ܷ����ͬһ�������� [Google Ȩ��](https://huggingface.co/google/gemma-4-12B-it) �� [�ܹ�˵��](https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/) |
| **20B �� �� InternVL3.5 GPT-OSS-20B-A4B Preview** | 2025-08��ϵ�з��� | 20B ��ϡ��ͼ�Ķ��ա�OpenGVLab �������Բ��� 20.9B���Ӿ����� 0.3B���ϼ�Լ 21.2B��ÿ�μ���Լ 4B�� [�ٷ� HF ��ʽȨ��](https://huggingface.co/OpenGVLab/InternVL3_5-GPT-OSS-20B-A4B-Preview-HF) �� [�ܹ���](https://github.com/OpenGVLab/InternVL) |
| **26.4B �� Inkling 2 Mini** | 2026-11-20 | ��������ͼ�ĺ�ѡ���ٷ����� 26.4B �ܲ�����3.4B �������������ѭ����ÿ�� token ��Լ 5.4B ��Ȩ�ؼ������ 27B ���Ƚ�ͬһ֤�ݸ����������ȷ�ʺ������ɱ��� [Thinking Machines Ȩ��](https://huggingface.co/thinkingmachines/Inkling-2-Mini) �� [�ٷ�����](https://thinkingmachines.ai/news/inkling-2-mini/) |
| **27B �� �� Qwen3.8-27B** | 2026-08-14 | �û�ѡ���Ĵ�ģ��������ģ�Ϳ��� 27B ָ�������ɣ������ֿ�� Safetensors ͳ��Լ 28B�����Ը������� Colab ���رʼǱ�����ʵ�� I/Q �ӿڲ�ѵ��ͶӰ�㡣��Ҫ������ģ���ϸ񲻳��� 27B���ɲ�������� Inkling 2 Mini �����ޡ� [Qwen Ȩ�������](https://huggingface.co/Qwen/Qwen3.8-27B) �� [�ٷ���������](https://github.com/QwenLM/Qwen3.8) |

��ʽ�Ƚ�ͳһ�����ű�������������ͬ�� I/Q ���桢��Ƶ������׷�ʡ������ɼ��ͼƷֹ���ÿ������ѵ���Լ������Ӳ㡣��ģ����ʹ����ͬ���ȣ������Դ�������ʱ�䡣��ÿ�����¼�����ı䲨�λ���ο��󣬴��Ƿ�����Ԥ�ڱ仯��Qwen3.5-9B �� Qwen3.8-27B �Ĳ���ͬʱ������ģ�ʹ���ѵ�����졣���о��ص���ͬ����ģ���ƣ��������� Qwen3.5-27B���� 4B��9B ����һ�顣[Qwen3.5-27B Ȩ��](https://huggingface.co/Qwen/Qwen3.5-27B)

## �岽������

| ���� | ��ʲô | ������ﵽ��һ�� |
|---|---|---|
| **1������ѧ��������׷��** | ������������ź�һ��δ֪������ת��ʼ����һ���������λ���ڶ����ǲ���ͬһ����������֪��Ƶ�󣬵�һ�����ŵ�ԭʼ��λ��д���Ⱥ�ɼ�����Ϣ����������ı��غ�����������������ο���������ʱ�ɼ����Ԥ�⡣ | �þ�ȷö�ٵõ������ڽ�ʾǰ��Ĵ𰸻���顣��������޹زο����ظ�ͬһ�۲⡢ֻ���������λ�������ա�ԭ�ƻ��е� 2 ������ 4 ���ؽ�������������ࡢ�̶����ȡ����������ӡ� |
| **2���Ѳ��α�ɿɸ��õ�����** | ���������� I��Q �����Ͳ����ʵȿ���Ԫ���ݣ������ʱ�����е�����������ÿ���������ܶ�Ӧ��һ��ԭʼ������ͬһ���εı���Բ�ͬ���⹲�á����滺��Ԥ�㡢�������ȡ�ʱ��ӳ�����Ч�۲����롣 | ���ü򵥵Ķ�������飬ͬһ�ݶ����ʾ�ܻش������λ��Ҳ���ڲ�����Ч�ο���ָ�ԭʼ��λ�������� I/Q �ͽ�������Ϊ���ա��Ƚ��𴰶�����λ���ĵ�����ʧЧ�� |
| **3����һ������ģ�Ͷ���** | ��������Ϊÿ�β���Ԥ��λ�ã���С�Ϳ�ѵ��ͶӰ��Ѳ�������ӳ�䵽������������ά������ͬ���ֽ���ע�����㡣���ֶ�������֤�Ĳ��α��������������ò��Ρ����⡢����Ԫ��ѵ��ͶӰ�㡣��һ��׷�Ӳο�����ʱ����ԭ��ʾ�� | ��δ����ѵ���ĺϳɲ����ϣ�ģ������ɡ��Ȼش���׷�ʡ����ο�����¡���ƽ�ⲻͬ�𰸵ĳ���Ƶ�ʣ�ͬһ���ʷ��䲻ͬ���Ρ�ֻ�����֡����Ҳ�����Ի�ֻ��������ʱ������˶�Ԥ�ڱ仯������һ����ʵ�ʶԻ��ĵ�����ԭ�͡� |
| **4������ʵ����������** | �ù�����֪���ж�����ֵ������ѡ Pluto ���� QPSK ������������������ CTTC �Ự����һ�������ɼ��ϲ��ԡ�����ͬƬ���϶ԱȽ����������Ӷ�������ֱ�Ӷ������������Իش𡣼���ʱ����ͼ�� I/Q ��ͬ�����𰸵Ĳ��ԡ� | �ֱ𱨸���Ȼ¼�ƺ��˹���λ��Ԥ����¼����ȷ�ʡ���ѡ���ʡ���֤ǰ��ĶԺ͸Ĵ����������Լ���λ֤�ݵ�ʱ������ͨ�ı���ͼ������Ҳ��һ��ɶԻع顣����ʾ�ɱ�ֱ�Ӷ��Զ����Իش�������޶��룻�����߶����������� I/Q �ز�������Ϣ����ɼ��衣 |
| **5��Ǩ�Ƶ��ڶ�������** | �̶�ͬһ���α�������Ԥ������ʱ�����ꡢȨ�غ����⼯��Ϊ��һ��ģ��ѵ��ר����СͶӰ�㣬�ø��ԵĶԻ�ģ����֯�����ġ� | ����������ͨ��ͬһ��������Ԥ��ʵ���⣬������Ե���ÿ�β���ռ�õ������ļ���������������������������һ������淶�����Ե�����Ȩ�ء�������ȷ�޶��ڲ��Թ��������Ͳ����ϡ� |

## ��һ����ʾ�ó�ʲô��

ȡ�ĸ���λ���Ϊ 1��i��?1��?i��������������շ����� i �� ?1��һ��ʼֻ�����β��Ρ��±�����ֵ����������֣�ģ��ʵ���յ����ǲ������롣

| �û��Ĳ��� | Ԥ�ڻش��������� |
|---|---|
| ���ڶ�����Ե�һ��ת�˶��٣��� | +90�㡣��������ֵ�ı�ֵΪ i����ͬ��ת������ |
| ����ô��һ��ԭ���������ĸ���λ���� | ��ǰ�ĸ���ѡ�ȿ��ܡ����������������ת֮�����λ�� |
| ����ͬһ�������ĵ�λ��Ƶ����Ƶ����ֵΪ i��������һ�� | ��һ�ε�ԭʼ��λ�� 1����Ƶ��ʾ������ת�� +90�㣬�ղŵ������λ���Գ����� |
| ��������ԭ�ĺ���������ķ��ȣ��������ڶ�������ת +90�� | ��һ�ʵĴ��沨�α�Ϊ 180�㡣������ͬʱ��תͬ���ĽǶȣ���һ������ +90�㡣 |

������ʾֻ�������ģ̬����ĵ�һ���������������Ű�������չ�������﷢���仯���������Ƿ�������ͬһ��λ�켣���͡���ͼ�е� A��B �ĶκͲο���ɡ������ͼ���벨��������ʱ��������ͼƬֻ�ṩ A��B ��ʱ��λ�ã��ò����ṩ��λ��ϵ������ͼ�б�ǡ������ı䲨�Σ��ֱ�������������Ƿ񶼱�ʹ�á�

## ��һ����ֵ��Ͷ���о�ʱ��

�Ӿ�ģ�� BLIP-2 �Ѿ�չʾ�˶���������ѵ��С���������ļ���·�ߣ�LEMMO �Ѿ��о�ָ��������ԭʼ I/Q ��������ǿ���ֱ�ӽ��ó���Ľ���˼·��ֵ��Ͷ���о�ʱ��ģ��ǲ��**����Ĳ��α�ʾΪ������׷��������ʲô����Щ֤�����������塢ʵ��������û�а��ⲿ����Ϣ������**����һ�ֵľ��彻��Ӧ��ͬһ�������ϵĽ����𰸡�������ʾ�𰸺������𰸣�����֮��Ĳ���ָ����һ���ø����[BLIP-2��Proceedings of the 40th International Conference on Machine Learning��2023](https://proceedings.mlr.press/v202/li23q.html) �� [LEMMO��arXiv:2611.09476](https://arxiv.org/abs/2611.09476) �� [EM-FloatFormer D ��֤��](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/D/PROOF.md)

���ļ�Ϊ������ơ�ʵ�鷶Χ�� Qwen3.8-27B ���޵�λ���û�ѡ�����������������˳��Ϊ���ֽ��顣��ѡȨ���Ѿ��˶ԣ�ʵ�ʽ��롢ѵ����ʵ�������Դ�ʵʩ��
```

### Event #8: `item.completed` — `command_execution` (`item_4`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '[Console]::OutputEncoding=[Text.Encoding]::UTF8; $r=Get-Content -Encoding UTF8 research/stage3d_qwen_signed_phase/REPORT.md; $r[0..140]; $r[176..215]; $r[249..265]'
```

**Aggregated Output:**

````text
�޷��������ԡ�������ģʽ��֧�ֺ������͵��������á�
����λ�� ��:2 �ַ�: 1
+ [Console]::OutputEncoding=[Text.Encoding]::UTF8; $r=Get-Content -Enco ...
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 
# EM-FloatFormer Stage 3D ʵ�鱨�棺���� Qwen3.5-4B �� s17 �����µ��з���Ƶƫ��ʽ��λ����ä��

> **�о���֧**��`research/stage3d-qwen-signed-phase`������ Stage 3C �����ύ [`4fe9162b`](https://github.com/20171806011/EM-FloatFormer/tree/4fe9162bdbf93fb0059cc53aec264153dbb67d0d) ������δ�޸��κμ����׶δ��������ݣ�  
> **ä��ǰ�����ύ��Commit #1��**��`a1fac7a8cfe5c2f6011bdae0080e8df2e4e68215`��ȫ�� 6 �����Ӳ� checkpoint SHA-256���¶Ȳ����������� `B_locked = 131` ��ä�����������嵥�ڽ�äǰд�� Git ��ʷ��

---

## ��ҳִ��ժҪ��˫�ż��ж�

����ʵ��ش�һ�����⣺**����ȫ���� `Qwen/Qwen3.5-4B` �� `equivariant_b8_s17` ��ɢ���α����������۾���������������ȫ��ͬ���з���ƵƫԪ������ѵ�����������Ԥ��������ͬ�������£��������Ӳ��� 6 ��������ֵ����ȫ�㣨`zero` ���ձۣ��滻Ϊ��Ӧʱ�� $t \in \{h, h+\ell, t_p\}$ ����ʽ��λ���� $[\cos(\omega t), \sin(\omega t)]$��`phase` ��λ�ۣ����ܷ���߶�δ��Ƶƫ�� Q3-repeat ��λ����Ԥ�⣿**

### ������� `G1` ���ܣ�`|��| = 0.40`, `h = 2`, `? = 2`, `t_p = 4.0`, `|��|(h+?) = 1.60 rad`, `��2t = 0.04`��

| �������� / �Ա��� | ׼ȷ�� (`Acc`) | ��������Ȼ (`NLL`) | `Brier` | 10-bin `ECE` | ˵�� / 95% ���� |
|---|---:|---:|---:|---:|---|
| **��λ�� `phase`���������ӵ�Ȩ���ܣ�** | **23.80%** (`0.23799`) | **1.4452** (`1.44523`) | `0.7845` | `0.1186` | `131` �������� �� `64` ĸ���� = `8,384` ����`150,912` �߼��У���`-��: 24.95% / 1.4547`��`+��: 22.65% / 1.4358` |
| **���ձ� `zero`���������ӵ�Ȩ���ܣ�** | **19.41%** (`0.19406`) | **1.4809** (`1.48088`) | `0.8010` | `0.1827` | `131` �������� �� `64` ĸ���� = `8,384` ����`150,912` �߼��У���`-��: 18.80% / 1.4678`��`+��: 20.02% / 1.4939` |
| **��Բ�ֵ��׼ȷ������ `phase ? zero`** | **+4.39 pp** (`+0.04393`) | �� | �� | �� | **95% ��� `t` ����**��`[+4.11, +4.68] pp`������ SD `0.01644`��SE `0.00144`����**���� Bootstrap**��`[+4.11, +4.67] pp` |
| **��Բ�ֵ��NLL ���� `zero ? phase`** | �� | **+0.0357 nats** (`+0.03565`) | �� | �� | **95% ��� `t` ����**��`[+0.0330, +0.0383] nats`������ SD `0.01556`��SE `0.00136`����**���� Bootstrap**��`[+0.0330, +0.0383] nats` |
| ͬ��ͬĸ���� `s17` CPU �������� | **64.62%** (`0.64617`) | **0.9411** (`0.94113`) | `0.5098` | `0.0009` | ��ʵ���Ӳ���� CPU ����������ࣺ`phase ?40.82 pp / -0.5041 nats`��`zero ?45.21 pp / -0.5398 nats` |
| ͬ��ͬĸ���� `full_iq` ��֪����ȫ���ν� | **64.64%** (`0.64641`) | **0.9290** (`0.92904`) | `0.5032` | `0.0046` | �������޻��� |
| ��ѡһ����������� | `25.00%` | `1.3863` ($\ln 4$) | `0.7500` | `0.0000` | ����Ϣ�²�ο��� |

### ����Ԥ�Ǽ����Ӳ����ӵĸ��Թ�������ɢ�̶ȣ�`G1`��

| ���� (`seed`) | `phase` Acc / NLL (`ECE`) | `zero` Acc / NLL (`ECE`) | ׼ȷ������ `phase ? zero` (`[95% t-CI]`) | NLL ���� `zero ? phase` (`[95% t-CI]`) |
|---|---:|---:|---:|---:|
| `202611` | `26.59% / 1.3950` (`0.0625`) | `19.98% / 1.4436` (`0.1274`) | `+6.62 pp` (`[+6.12, +7.11] pp`) | `+0.0486 nats` (`[+0.0457, +0.0516] nats`) |
| `202622` | `21.23% / 1.4814` (`0.1937`) | `18.17% / 1.5291` (`0.2486`) | `+3.06 pp` (`[+2.68, +3.44] pp`) | `+0.0477 nats` (`[+0.0432, +0.0521] nats`) |
| `202633` | `23.57% / 1.4593` (`0.1219`) | `20.07% / 1.4699` (`0.1722`) | `+3.50 pp` (`[+3.06, +3.95] pp`) | `+0.0106 nats` (`[+0.0068, +0.0145] nats`) |
| **��������ɢ��** | �� | �� | **SD `1.94 pp`**����Χ `[+3.06, +6.62] pp` | **SD `0.0217 nats`**����Χ `[+0.0106, +0.0486] nats` |

> [!IMPORTANT]
> **Ԥ��ʵ��˫�ż��ж���`PASS BOTH`����ʽͨ�����������·����Ƴ������⣩**
> 1. **��ʽ�ж�**��`acc_t95_lo = +4.11 pp > +2.00 pp`��ͨ������ `nll_t95_lo = +0.0330 nats > +0.0200 nats`��ͨ����������������׼ȷ���� NLL ��ȫ����ͬ�������ơ�
> 2. **�ؼ���ѧ���壨���ɽ���ʽ�������Ϊ����������������ģ������**��
>    - ������ `G1` �ϣ�**���۵ľ���׼ȷ���� NLL ��������ѡһ����²⣨`25.00% / 1.3863`��**������ͬ���� CPU ����������`64.62%`����Լ����֮һ��
>    - ��� 2��2 ̽���񣨵� 3 �ڣ��� Work F ����̽�루�� 4 �ڣ��ɼ�����ѵ���ۼ�ת�� $|��|(h+\ell) \le 1.25\text{ rad} < \pi/2$ ��֧�����ڣ�`G2` �� `G4`����**`zero` ���ձ۷����������� `phase` �ۣ�`G2`: `36.36%` vs `33.88%`��**����һ�������ۼ�ת�� `1.60 rad > ��/2` ��ǿ��������`G1` �� `G3`����δѧ�ᡰ�� $90^\circ$ �����۷��������۾���������ߣ�������֧������ϸ������� `zero` �������Ԥ����ø��`19.41%` / `19.74%`������ `phase` �����ѧ����΢��������������Ӧ��`gain = +0.0098`��������΢���صľֲ�Ԥ�⣬�ٵ���Լ 3�C4 ���ٷֵ㡣
> 3. **���Ļ������ʵ�ֱ�ӻش�**��
>    - **���������Ƿ���������ȷ�ķ����ƶ���** **�ǣ���ЧӦ������** �� `��0.40` �� `phase` ��ƥ�������Աȶȵķ�������� `+0.0098`��ԼΪ���������ο� `1.0` �� 1%����������֪����ʹ NLL �ӻ� `+0.0184 nats`��������ǿ�� `zero` �ۣ����� `+0.0050`�������ӻ� `+0.0074 nats`�����ҵ�ģ̬ì��̽��֤�������� 6 ��������ֵ��λ�۵����жȣ���ת��λ������ `TV = 0.0371`�����ڶ�����Ƶƫ�ı����ֵ����жȣ���ת�ı����� `TV = 0.0281`���������߾�δ�γ���Ч�� $\pi/2$ ������������ת���ӡ�
>    - **�����Ƿ���ϳ��ٵ���Ƶ Q2 ����֤�ݣ�** **��δ�ɹ����ϵڶ��ζ���������������Ϣ��** ��δ�� `C4`��`�� = +0.35, t_p = 4`�������ϣ������˴ӵ�����Ч��Ƶ�����ζ����ز�����׼ȷ������Ծ�� `+7.81 pp`��`66.41% �� 74.22%`, `NLL 0.8192 �� 0.6632`������ Qwen ��λ���ڼ��� `valid` ��Ƶ����� `none` ����������`24.80% �� 28.78%`, `+3.97 pp`���� NLL ΢�� `-0.0040`�����ټ���ڶ���������Ч���� `fresh_resample` ��׼ȷ�ʷ������� **`-2.34 pp`**��`28.78% �� 26.43%`, `NLL 1.3889 �� 1.4056`����`zero` �������Ϊ `valid 26.63% �� fresh_resample 25.78%`��`-0.85 pp`����

---

## 1. �����������������Token �˶ԡ��Դ�����Ų��������˵�

### 1.1 ����������ʱ��� Chat Template ��ƣ�[smoke_audit.json](results/smoke_audit.json)��[PROTOCOL_LOCK.json](PROTOCOL_LOCK.json)��

| �����Ŀ | ʵ������ֵ |
|---|---|
| **��������ģ��** | `Qwen/Qwen3.5-4B`��HF pinned revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`������֤����δ���Ӿ������� `lm.model.visual = None` ������ `tie_word_embeddings`�� |
| **����������ָ��** | ������� `4,205,751,296`��ѵ��ǰ�����ָ�� SHA-256 �ϸ�һ�£�`f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d` |
| **���α�����** | `equivariant_b8_s17`��16 ����ɢ̬ / ÿ�� 4 bit���ļ� SHA-256��`e67d9fbafb90791ab19549de7789798d818286d7f52d7ac566cc39df71e78356`�� |
| **���Ӳ��ѵ������** | **`2,184,964`**��ռ������ `0.052%`�������� 6��128 `phase_mlp` �� `phase_scale`��**`phase` �� `zero` ���۽ṹ������� 100% ��ͬ**�� |
| **Native Chat Template & Thinking** | Template SHA-256 `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`����ʽ�ر� thinking��`enable_thinking=False`�����ϸ��� `<|im_start|>assistant\n<think>\n\n</think>\n\n` ��β��`ends_with_disabled_think = true`�� |
| **�������л���** | Modal `NVIDIA A100-SXM4-80GB` (`81,179.8 MB`), `torch 2.15.0+cu132`, `transformers 5.24.1`, `flash-linear-attention 0.6.0`, `numpy 2.3.5`, `scipy 1.17.0`, `torch.bfloat16` |
| **ģ�ͼ��غ�ʱ�볣פ�Դ�** | ���� `57.69 s`����פȨ��ռ�Դ� `8,021.9 MB` |
| **Smoke ǰ��/����/�������** | �������������ݶ��� `host_grads_nonzero = 0`�����Ӳ� 29/29 ����������ȫ���������ݶȣ��ϵ㱣�����������ʲ� `reload_max_prob_diff = 0.0`��Q3 ̰������������`���� 2��+135�㵽 -135�㣩` |

#### 12 �����ĺ�ѡ����һ Token ��ƣ����ǿա��߽���ϲ���

| ���� | `physical_index` | ����������ѡ�ı� | Token ���У����ȣ� | �߽羫ȷƴ�� |
|---|---:|---|---|---:|
| `Q1` | `0 / 1 / 2 / 3` | `0Ȧ��0�㣩` / `1Ȧ��+90�㣩` / `2Ȧ��180�㣩` / `3Ȧ��+270�㣩` | �ֱ𳤶� `6, 8, 8, 9` | 4/4 `True` |
| `Q2` | `0 / 1 / 2 / 3` | `����0��1��` / `����1��i��` / `����2��-1��` / `����3��-i��` | �ֱ𳤶� `5, 5, 6, 6` | 4/4 `True` |
| `Q3` | `0 / 1 / 2 / 3` | `����0��-45�㵽+45�㣩` / `����1��+45�㵽+135�㣩` / `����2��+135�㵽-135�㣩` / `����3��-135�㵽-45�㣩` | �ֱ𳤶� `14, 15, 16, 15` | 4/4 `True` |

### 1.2 ���̹ؼ��Ų飺Qwen3.5 �������ע�������û��� BF16 ��������ֵһ����

1. **�������û����µ� `71.64 GiB` OOM ��λ����Σ�[qwen_connector.py](qwen_connector.py)��[bench_steady.json](results/bench_steady.json)��**��
   - ���� `_make_differentiable_qwen35_cache()` ���հ� `_update_conv(..., _l=layer)` ����ÿ�� `LinearAttentionLayer` ʵ������ Python �������� `layer -> update_conv_state -> _l -> layer` ֱ�����û����� `cand_out` �� `use_cache=False` �Դ��� `cache.update_recurrent_state` д�����ͼ����������΢���ļ���ͼ�޷������ü�����ʱ���գ����������� Seed 1 ����������� `71.64 GiB` OOM��
   - **�޸�����**������ģ�鼶�������� `_OutOfPlaceLinearAttentionLayer(LinearAttentionLayer)` �л� `layer.__class__`����ʵ���հ����û�����ͬʱʹ��ԭ�� `F.log_softmax(..., dim=-1, dtype=torch.float32)` ����ȫ�ʱ� `.float()` ���ƣ����� `cand_out` ǰ��ʱ `del pref_out, out_cache, pref_cache`��
   - **ʵ����֤**����������� `Q3_after_pilot`��`351` �ַ� + `1` ���ο����� `Q2_fresh_resample`��`2` ���ο��������� 4 ��΢���������Դ��ϸ�㶨Ϊ **`[8055.2, 8055.2, 8055.2, 8055.2] MB`**����й©����`micro_batch_size=16` ѵ����ֵ�ȶ��� **`52,972.5 MB`**����̬ `1,975.7 ms / 32-ex step`����ä������ `b=64` ��ֵ **`45,409.9 MB`**��`23.88 ms/��`����
2. **`bfloat16` + FLA �˵�����Сһ�����������͸�����棨[direct_vs_table_parity.json](results/direct_vs_table_parity.json)��**��
   - ���� 256 �����ȫ��ͬ�� **`matched_batch_size = 64`** �����£�ֱ��ǰ���� 256 ����� G1 ä�� 256 �������˶���ȡ�� **`max_abs_ll_diff = 0.0`**��**`max_abs_prob_diff = 0.0`**��**`argmax_agreement = 1.0`**��4 �� `(arm, ��0.40)` ȫ����ȷΪ�㣩��
   - ͬʱ��¼�˿�����С�� BF16/Triton ����ֿ�������������ѵ��ǰ���Ȩ���ϣ�`bench_steady.json`����`b=1/16` �� `b=64` �� `max_abs_ll_diff` Ϊ `0.4816�C0.4830`��`argmax_agreement = 1.0`��������ѵ����Ԥ��߶Ƚӽ����ȵ�ä��ģ���ϣ�`direct_vs_table_parity.json` �� `cross_batch_b16_vs_b64`����`b=16` vs `b=64` �� `max_abs_ll_diff` ���� `0.115�C0.289`��`max_abs_prob_diff = 0.024�C0.067`�����������ѡһ���ֱ߽�ϸ΢�Ŷ������� `argmax_agreement` �� `phase -0.40` Ϊ `0.6875`��`phase +0.40: 0.8750`, `zero -0.40: 1.0000`, `zero +0.40: 0.9375`����ȫä��ͳһ��������������ȫä��ǰ�򣩾�ͳһ�����ڹ̶� `batch_size = 64` �£�ȷ���ڲ��ϸ���Ǣ��

### 1.3 Modal ��Ѷ�������˵���ȫ���㸶�ѣ�

| ������Ŀ | Modal �Ʒ��˵���¼ | ˵�� |
|---|---:|---|
| ��ʱ������ҵ (`Ephemeral Apps`) | **`$17.12`** | ���� Smoke��Bench�������� 1 �� OOM����6 ������ѵ����Dev Pilot��ȫ��ä���Լ� 1 ����ǰ��ֹ����ʷ Ministral ��� |
| �־û����洢 (`Volumes`: `qwen3-model-cache` + `stage3d-results`) | **`$1.34`** | Free Storage ȫ��ֿ� |
| �������� (`Network Egress`) | **`$0.01`** | Network Egress Allowance ȫ��ֿ� |
| **ʵ���˵�Ӧ�� (`Billed Cost`)** | **`$0.00`** | **��ȫ�� `$30.00` Modal ��Ѽ���������ɣ�����Լ `$12.88` ��Ѽ����ȣ�** |

---

## 2. ���۶������á�6 ��ѵ��������������äǰ����

### 2.1 ������ֵ���������й¶��Լ

- �Ե�һ��Ϊʱ����� $t_1 = 0$���ֱ�ȡ $t \in \{h, h+\ell, t_p\}$ ���� 6 άʵ��ֵ�ۣ�
  - **`phase` ��λ��**��$[\cos(\omega h), \sin(\omega h), \cos(\omega(h+\ell)), \sin(\omega(h+\ell)), \cos(\omega t_p), \sin(\omega t_p)]$��
  - **`zero` ���ձ�**��$[0, 0, 0, 0, 0, 0]$��
  - **�ϸ�������**������ 6 �����⣬��ͬ�� `phase_mlp` �ṹ����ͬ�� `(z1, z2)` ���뻺�桢��ͬ�������������з��� $\omega$������ `omega=+0.40rad/��` �� `-0.40rad/��`������ͬ��Ӳ����������ǩ����ͬ��ѵ������˳�����Ż���Ԥ�㣬�����ų���ʵ����·����δ��ʾ��Ƶ�빤�� ID ��ǩ��
- **ѵ��֧�ּ�**������ Stage 3 �� `C1 / C2 / C3` �طţ�ÿ���� 512 ĸ���Σ����� Work F �����з��� CFO �ල��$|\omega| \in \{0, 0.10, 0.25\}$��ʱ����� $(h, \ell, t_p) \in \{(2, 2, 4), (2, 3, 1)\}$��ѵ������ۼ�ת�� $0.25 \times 5 = 1.25\text{ rad}$����ÿ���� 512 ĸ���Σ���������Ӧ $|\omega| = 0.18$ �� `C1/C2/C3`��ÿ���� 128 ĸ���Σ���
- **����������������ѵ������һ�£�**��`epochs = 8`, `max_steps_per_epoch = 160`���� `1,280` �� = `40,960` ����/�Σ�, `batch_size = 32`, `micro_batch_size = 16`, `lr = 1.5e-3`��AdamW + 5% warmup + cosine ˥������`train_scorer = eval_scorer = kv_cached`, `eval_every_epochs = 2`��

### 2.2 6 �����Ӳ�ѵ�������뿪���������¶�������[training_summary.json](results/training_summary.json)��

| �� / ���� | ���� Epoch | ����ɸѡ�� (`sel_score`) | У׼ǰ `dev_cfo_q3_nll` | У׼�� `dev_q3_cal_nll` (`T_q3`) | У׼�� `dev_q2_cal_nll` (`T_q2`) | ��ʱ (s) / ��ֵ VRAM | Checkpoint SHA-256��ǰ 16 λ�� |
|---|---:|---:|---:|---:|---:|---:|---|
| `phase / 202611` | `7` | `1.38097` | `1.37762` (`29.62%`) | `1.37535` (`0.5099`) | `1.38664` (`4.0000`) | `2884.0 s` / `52,972.5 MB` | `088d4ad34e1152ff` |
| `phase / 202622` | `8` | `1.33823` | `1.31390` (`35.70%`) | `1.28527` (`0.8489`) | `1.35653` (`1.1069`) | `3026.2 s` / `52,972.5 MB` | `ba275787201abf5f` |
| `phase / 202633` | `8` | `1.34566` | `1.32520` (`35.00%`) | `1.30019` (`0.6575`) | `1.37150` (`1.0038`) | `3217.8 s` / `52,972.5 MB` | `75f0022894bde60b` |
| `zero / 202611` | `8` | `1.36064` | `1.34627` (`35.68%`) | `1.33499` (`1.0414`) | `1.37938` (`1.2210`) | `2908.7 s` / `52,972.5 MB` | `3d09eded33b58b92` |
| `zero / 202622` | `8` | `1.31245` | `1.26930` (`39.10%`) | `1.23335` (`0.7827`) | `1.30698` (`1.4118`) | `3059.1 s` / `52,972.5 MB` | `a8eb55bfad0d266c` |
| `zero / 202633` | `8` | `1.34157` | `1.31870` (`32.40%`) | `1.29266` (`0.6733`) | `1.38119` (`1.2735`) | `2983.0 s` / `52,972.5 MB` | `ea7aebcc6d53bc71` |

*ע��ȫ�� 6 ��ѵ���� `host_sha256_after` ���ʼ��ȫһ�£�����ѵ��Ԥ�������Ķ��������� `dv_groups` ȫ����ϱ����¶� `(T_q3, T_q2)`��ѵ������ʧ���� `1.5�C1.6` ˳������ `1.29�C1.38`��*

### 2.3 ��äǰ���������ȵ��������� `B_locked` ������[dev_pilot_lock.json](results/dev_pilot_lock.json)��

- �ڶ����ȵ������ռ� `NS_DEV_PILOT = 940202`��`16` ���������� = `1,024` ��ĸ���Σ�`|��| = 0.18, h = ? = 2, t_p = 4`���������������� 6 ��ģ�ͣ�
  - ʵ���ȵ������ֵ�`acc_diff_mean = -3.35 pp`��`-0.03353`����`block_sd = 0.01464`��`nll_diff_mean = -0.02438`��`block_sd = 0.01341`��
  - �� `df = 15, �� = 0.05` ȡ $\chi^2$ ���� 95% �����Ͻ���� `1.43731`�������Ͻ��׼�� `sd_acc_upper = 0.02104`��`sd_nll_upper = 0.01928`��
  - ���� Stage 3C `research/stage3c/work_results/E/verify.py --pilot-block-sd 0.02104014 0.01928086`���� `t_blocks = 131`��Ч�ܣ�׼ȷ�� `0.9505`��NLL `0.9999`����� `��0.01632` / `��0.01379`�����ɴ��ڽ�äǰ���� **`B_locked = max(131, 131) = 131` �����飨`8,384` ��ĸ���Σ�**�������� `blind_seed_manifest.json`��SHA-256��`b6c075162010a0d6e72bb07cd32a4a01aed24e1abca8be2f949846d258c0f7e7`������ͬ���������� `PROTOCOL_LOCK.json` ���ύ�� `a1fac7a8` ��������ä�⡣

---

## 3. �������� 2��2 ̽���������ۼ�ת�� $\pi/2$ �߽�������

�ڱ���ÿ�� `SNR = 2 dB, �� = 0.10`��ÿ��ĸ�������� `(����) �� 3 ����ת �� 3 ������` �£����� `2��2`������Ƶƫ���� `|��|` �� �ۼ�ת�� `|��|(h+?)`��ä�������±���ͬ�������ۼ���ɢ���� `��2t`����

| ä��� | ��; | `|��|` | `(h, ?, t_p)` | �ۼ�ת�� `|��|(h+?)` | �ۼ���ɢ `��2t` (`t=h+?`) | ���� / ĸ���� | `phase` Acc / NLL | `zero` Acc / NLL | ���׼ȷ������ (`95% t-CI` / `Boot`) | ��� NLL ���� (`95% t-CI` / `Boot`) | ͬĸ���� `s17` ���� / `full_iq` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| **`G1_main_extrap_both`** | **�����飨˫���ƣ�** | `0.40` | `(2, 2, 4)` | **`1.60 rad`** | `0.04` | `131 / 8,384` | `23.80% / 1.4452` | `19.41% / 1.4809` | **`+4.39 pp`** `[+4.11, +4.68]` (`[+4.11, +4.67]`) | **`+0.0357`** `[+0.0330, +0.0383]` (`[+0.0330, +0.0383]`) | `64.62% / 0.9411` (`64.64% / 0.9290`) |
| **`G2_interp_both`** | ̽����˫�ڲ壩 | `0.20` | `(2, 2, 4)` | **`0.80 rad`** | `0.04` | `8 / 512` | `33.88% / 1.3358` | `36.36% / 1.3081` | **`-2.48 pp`** `[-4.39, -0.58]` (`[-3.79, -0.89]`) | **`-0.0278`** `[-0.0459, -0.0096]` (`[-0.0414, -0.0134]`) | `46.78% / 1.0222` (`46.78% / 1.0193`) |
| **`G3_extrap_angle`** | ̽����������ת�ǣ� | `0.20` | `(4, 4, 8)` | **`1.60 rad`** | **`0.08`** | `8 / 512` | `22.68% / 1.4410` | `19.74% / 1.4781` | **`+2.94 pp`** `[+1.73, +4.15]` (`[+2.03, +3.88]`) | **`+0.0371`** `[+0.0230, +0.0512]` (`[+0.0262, +0.0479]`) | `62.11% / 0.9796` (`61.91% / 0.9539`) |
| **`G4_extrap_freq`** | ̽����������Ƶ�ʣ� | `0.40` | `(1, 1, 2)` | **`0.80 rad`** | **`0.02`** | `8 / 512` | `33.29% / 1.3347` | `34.96% / 1.3098` | **`-1.67 pp`** `[-3.41, +0.06]` (`[-2.97, -0.34]`) | **`-0.0249`** `[-0.0339, -0.0160]` (`[-0.0320, -0.0184]`) | `47.75% / 1.0159` (`49.22% / 1.0037`) |

> **�ؼ����֡�����������̮�������ۼ�ת���Ƿ�Խ�� $\pi/2$ ���ޱ߽磬���ǵ���Ƶƫ���� `|��|`**��
> 1. �Ƚ� `G4`��`|��|=0.40` ���ۼ�ת�� `0.80 rad < ��/2`���� `G2`��`|��|=0.20`, `0.80 rad`�������� `G4` Ƶ��������δ���� `0.40`������׼ȷ����ά�� `33.29% / 34.96%`����Զ��������� `25%`����˵�����Ӳ���������δ�������ϵ� `0.40` �������෴������ѵ����δ������ת���۷��� 6 ��������������ά�ȴ�����ѵ�������Ż�������`zero` ��֧�ֽ��ڣ�`G2, G4`�������� `1.67�C2.48 pp` �� `0.025�C0.028 nats` �ȶ����� `phase`��
> 2. �Ƚ� `G3`��`|��|=0.20` ����ѵ����Χ�ڣ��� `h=?=4` ʹ�ۼ�ת�Ǵ� `1.60 rad > ��/2`���� `G1`��`|��|=0.40`, `1.60 rad`����ֻҪ�ۼ�ת�ǿ�� $\pi/2 \approx 1.5708\text{ rad}$����ʹ `|��|=0.20` ��������ѵ���ֲ��ڲ���`G3`�������۾����̵��� `22.68% / 19.74%`����ͬ���������� `1.60 rad` �·����������ֲ�ǡ������ɢת�Ƕ���ﵽ����׼ȷ�ʣ�`62.11%�C64.62%`�������� `G3` ���ۼ���ɢ `��2t = 0.08` �ﵽ `G4` �� 4 �������� `G3` ���� `full_iq` �Ե��� `G1`��`61.91%` vs `64.64%`��������֧������ʤ����ת������Ȼ�� **$|��|(h+\ell) = 0.80$ vs $1.60\text{ rad}$**��

*ע������˶� `G1..G4` ��������ɢ�ȡ���`G2` ׼ȷ�ʲ�����ӷ�Χ `[-6.05, +0.26] pp`��SD `3.24 pp`����NLL ���Ʒ�Χ `[-0.0436, -0.0166] nats`��`G3` ׼ȷ�ʲ� `[+2.21, +3.71] pp`��SD `0.75 pp`����NLL ���� `[+0.0133, +0.0542] nats`��`G4` ���� `G2` ������ͬ�ڲ巽��ģʽ��*

---

## 4. ������� I��Work F Ƶƫ���Ÿ�Ԥ�뵥ģ̬ì��̽��
## 5. ������� II��Work G ��ת�ȱ��������ⲿȺƽ�����뱨��

| ���ά�ȣ����� `G1` `��0.40`��131 ���飩 | ��λ�� `phase` | ���ձ� `zero` |
|---|---|---|
| **ԭʼ������`(0,1)` ��ת�ڶ���������**��`mean / worst max_abs`��`>0.20` ռ�ȣ� | `0.0591 / 0.4817`��`5.11%` | `0.0620 / 0.3587`��`5.80%` |
| **ԭʼ������`(1,1)` ����ͬתһ����ѭ����λ**��`mean / worst max_abs`��`>0.20` ռ�ȣ� | `0.1263 / 0.3800`��`15.44%` | `0.1334 / 0.3792`��`18.96%` |
| **ԭʼ������`(1,0)` ��ת��һ��ѭ����λ**��`mean / worst max_abs`��`>0.20` ռ�ȣ� | `0.1237 / 0.3752`��`15.13%` | `0.1316 / 0.3859`��`17.93%` |
| **� `(1,1)` ����Υ������** | `s202622, +��, mother=39, c00=198 �� c11=10`��`probs_00=[0.1395, 0.1004, 0.1689, 0.5913]` vs `probs_11=[0.2113, 0.3310, 0.2750, 0.1827]`��`diff = 0.3800`�� | `s202611, +��, mother=1637, c00=9 �� c11=77`��`probs_00=[0.6403, 0.0815, 0.0630, 0.2152]` vs `probs_11=[0.2778, 0.2611, 0.2501, 0.2110]`��`diff = 0.3792`�� |
| **ԭʼ�������գ�`independent_x3` ������������**��Ӧ�ӽ����� `25%`�� | `Acc = 24.91%`, `NLL = 1.4237`, `mean_tv_from_unif = 0.0947`, `max_tv = 0.2935` | `Acc = 24.97%`, `NLL = 1.4315`, `mean_tv_from_unif = 0.1122`, `max_tv = 0.2783` |
| **ԭʼ�������գ�`shuffled_Z` ���һ���** | `Acc = 25.20%`, `NLL = 1.4601` | `Acc = 25.16%`, `NLL = 1.5070` |
| **ԭʼ��������ָ�꣨δ���κ����߶Գ�ƽ����** | **`23.80% / 1.4452`** | **`19.41% / 1.4809`** |
| **�����ⲿ�ԳƲ��� I��16 ·Ⱥƽ�� `Pf`**������� CPU ���㣩 | `17.05% / 1.4113`����� raw `Acc -6.75 pp [-7.09, -6.41]`��`NLL +0.0339`�� | `16.43% / 1.4391`����� raw `Acc -2.98 pp [-3.29, -2.66]`��`NLL +0.0418`�� |
| **�����ⲿ�ԳƲ��� II�������� `F0`**������� CPU ���㣩 | `20.88% / 1.4112` | `20.14% / 1.5090` |

> **���**������ Qwen ���Ӳ���ѵ����ֻ��������ת������ʾ�������ϸ� $\mathbb{Z}_4 \times \mathbb{Z}_4$ Ⱥ�ȱ����ӣ�ԭʼ�����Դ���Լ `0.06` (`rot01`) �� `0.12�C0.13` (`rot11/10`) �ĵȱ���ȱ����Ȥ���ǣ���ԭʼ׼ȷ�ʱ���Χ�ƻ��Ե��� `25%` �Ҵ��ڿ�����ϵͳ����ʱ������ǿ��ִ�� 16 ·Ⱥƽ�� `Pf` ���ܰѼ���Ԥ����ƽ���Խ��� NLL��`1.4452 �� 1.4113`��������ʹ׼ȷ�ʽ�һ���½��� `17.05%`����˵���ⲿȺƽ��������ƾ�մ���δѧ���� $\pi/2$ ��������֪ʶ�����߱����ڱ�����ͳ�����ϸ���С�

---

## 6. ������� III��Work A �ٵ���Ƶ `C4 (�� = +0.35, t_p = 4)` �߱۶Ա���ɹ����ط�

### 6.1 Q2 ��ʵ�����ʾ�������ο��߽�˵��

- **��������ĵ�Ƶ��ʾ**������ Stage 3 �� 12 ά������������ `build_ref_feature(r)`��ʵ�����鲿ƥ��ͶӰ�����һ/��ɢ����ͳ�ƣ�ͨ�� `ref_mlp` ӳ��Ϊ 4 �� `num_ref_tokens` �������� token ���������Ӧ `<REF>` ��λ����**δ�Ե�Ƶ���� `s17` 4-bit ��ɢ����**��ͬʱ���� `(z1, z2)` �������ݴ����� `s17` 4-bit ��������ˣ����������ο����� `full_iq`���� `q2_reader.read` ��ɢ���ܿ��������뵼Ƶ��δ�� 4-bit ��ɢ�ضϵľ�ȷ�������ࡣ
- **ǰ��ȥ��ʵ��**���� [data_stage3d.py](data_stage3d.py) `build_q2_arm_references()` �ṩ���� `deduplicate` ��־��`duplicate_copy_dedup` �� `valid` �� `acquisition_id` ȥ�غ���� 100% ��ͬ�� `<REF>` token ����ʾ�ʣ�`duplicate_copy_raw` ����ر�ȥ���Բ����������ظ� `<REF>` token ���������жȡ�

### 6.2 Work A �߱����������ձ���512 ��ĸ���Σ�3 ���ӻ��� vs CPU `q2_reader.read`��

| Q2 ֤�ݱ� | `phase` Acc / NLL (`ECE`) | `zero` Acc / NLL (`ECE`) | `q2_reader.read` ������ Acc / NLL (`ECE`) | ��Ի�������/��Ϊ���� |
|---|---:|---:|---:|---|
| `none`���޲ο��� | `24.80% / 1.3928` (`0.0230`) | `25.78% / 1.3938` (`0.0189`) | `25.00% / 1.3863` (`0.0000`) | ä�����׼ |
| `valid`��1 ��ͬ�ŵ���Ƶ�� | **`28.78%`** / `1.3889` (`0.0304`) | `26.63% / 1.4107` (`0.0557`) | `66.41% / 0.8192` (`0.0301`) | ��Ч֤�ݽ��� |
| `duplicate_copy_dedup`������ȥ�أ� | **`28.78%`** / `1.3889` (`0.0304`) | `26.63% / 1.4107` (`0.0557`) | `66.41% / 0.8192` (`0.0301`) | **�� `valid` ���������**��`mean_tv = 0.0`�� |
| `duplicate_copy_raw`������δȥ�أ� | `26.30% / 1.3944` (`0.0284`) | `25.78% / 1.4133` (`0.0463`) | �� | ���ظ���ʾͬ���ɼ���������� `valid`: `phase -2.47 pp` (`TV=0.0404`, `160` �Ķ� / `198` ��), `zero -0.85 pp` (`TV=0.0421`) |
| `independent_only`���������޹أ� | `24.80% / 1.4077` (`0.0364`) | `24.93% / 1.4205` (`0.0499`) | `25.00% / 1.3863` (`0.0000`) | ��� `none`: `phase +0.00 pp` (`TV=0.0423`, `222` �� / `222` ��), `zero -0.85 pp` (`TV=0.0663`) |
| `valid_plus_independent` | `24.54% / 1.4082` (`0.0425`) | `27.28% / 1.4183` (`0.0480`) | `66.41% / 0.8192` (`0.0301`) | ��Ч��Ƶ���ţ���� `valid`: `phase -4.23 pp` (`TV=0.0403`, `159` �� / `224` ��), `zero +0.65 pp` (`TV=0.0681`) |
| **`fresh_resample`**���� 2 �ζ��������� | `26.43% / 1.4056` (`0.0348`) | `25.78% / 1.4064` (`0.0592`) | **`74.22% / 0.6632`** (`0.0333`) | ��� `valid`�������� **`+7.81 pp / -0.1560 nats`**��Stage 3C �ع��׼ `+8.5636 pp`����**������ `phase` Ϊ `-2.34 pp / +0.0168 nats`��`172` �� / `208` ��, `max_conf_inc=0.4181`����`zero` Ϊ `-0.85 pp`��`159` �� / `172` ����** |

- **����ĸ���ι켣����`mother_idx = 36, z1 = 11, z2 = 15, true_x1 = 3`, `seed = 202611`��**��`q2_reader.read` ������������Ч�ο��� `none [0.25, 0.25, 0.25, 0.25]` �� `valid [0.0100, 0.2563, 0.6672, 0.0665]`����ƫ������ 2���� `fresh_resample [0.0201, 0.0217, 0.2945, 0.6638]` ׼ȷ�����ǩ `3` ���ʴ� `0.0665` ������ **`0.6638`** Ťת�о�����ͬһĸ������ `phase` ���������ͣ���� `[0.2497, 0.2495, 0.2503, 0.2505]`��`valid`���� `[0.2498, 0.2494, 0.2503, 0.2505]`��`fresh_resample`����`zero` ������ `fresh_resample` �� `prob_3` ���� `0.2964`������ȴ�������������ϵ������Ŷ�����ȫ�������� `-0.85 pp`��

### 6.3 �ɹ��� `C1 / C2 / C3` �ط�δ������֤��`s3_test` �� 2 ���� �� 3 ��ת = 384 �� �� 3 ���ӣ�

�����ڱ��ֿ�ִ�л��߼��ع鸴�飺

```bash
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage3c/verify.py --repo .
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch --with pytest python -B -m pytest research/stage3d_qwen_signed_phase/tests/test_stage3d_cpu.py -v
```

### 7.4 ������ֵ�ü���׷��ĺ��Ĺ۲�

> **��ֵ������׷��Ĺ۲⣺Ϊ�ο���ۼ�ת�� $\pi/2$ �߽磨$|��|(h+\ell) = 0.80 \rightarrow 1.60\text{ rad}$�������۾����ּ�����£��ҵ�ǰ�������� `<PHASE>` token��1,280 ����ֻ�յ��� `+0.0098`�������ο�Լ 1%����΢���������棿**
> - ʵ���ʾ��ֻҪ�� `|��|` ��ߵ� `0.40` �������ӳ�ʹ��ת������ `0.80 rad`��`G4`����������׼ȷ�ʾ���ά���� `33.3% / 35.0%`����һ��������֪Ƶ�� `|��| = 0.20` �������ӳ�ʹ��ת��Խ�� $\pi/2$ ���� `1.60 rad`��`G3`���������������� `22.7% / 19.7%`�������ڵ��͵����������ϸ�������ֵ��ȫ�ԽǷ�ָ������ 0 vs ���� 2���������о���
> - ԭ��ֱָ�ṹ��Ӧƫ�ã�**�� `[cos(��t), sin(��t)]` �������� MLP ����Ϊ 3 ��׷��������β���ļ��� `<PHASE>` �� token����Ҫ�󶳽�����ģ���Լ�ͨ����ע��/����״̬˥��ȥ�����㡱������ά��ת����ɢ `(z1, z2)` �����ĵ��ƣ����������֧�����ڵ�������������Ϊ�� `G2/G4` �� `phase` ��� `zero` Լ `2 pp`�������޷���֤ $\text{SO}(2)$ �ȱ�Ⱥ���ơ�** ��������ȻҲ��ɾ��Ķ���ʵ���ǣ��� `[cos(��t), sin(��t)]` ����Ҽ��� token ����Ϊ���Ӳ��ڲ���**��ʽ��ά˫����/����ת����**���ڴ��붳������֮ǰ��������������������λ��Ԥ����ת������λ����ȺЭ����ϣ���ѵ��֧����΢��� $\pi/2$ �� $|\omega|(h+\ell) \approx 1.8$ ���֡���ʾǷԼ�����롰֧��δ���ǡ�����

---

## �����嵥��ָ��ê��
````

### Event #9: `item.completed` — `command_execution` (`item_8`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '[Console]::OutputEncoding=[Text.Encoding]::UTF8; $m='"'C:\\Users\\李则徐\\.codex\\memories'; "'$r=Get-Content -Encoding UTF8 (Join-Path $m '"'MEMORY.md'); "'$r[28..59]; Get-Content -TotalCount 75 -Encoding UTF8 (Join-Path $m '"'rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md')"
```

**Aggregated Output:**

```text
�޷��������ԡ�������ģʽ��֧�ֺ������͵��������á�
����λ�� ��:2 �ַ�: 1
+ [Console]::OutputEncoding=[Text.Encoding]::UTF8; $m='C:\Users\������\.co ...
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 
# Task Group: EM-FloatFormer P4 scientific assessment and reproducible publication

scope: Assess and publish the EM-native Transformer P4 simulation without overstating learned-model performance.
applies_to: cwd=C:\Users\李则徐\Documents\ChatGPT\EM-FloatFormer; reuse_rule=Reuse the matched-control, reproducibility, and publication approach for this repository; numerical results, commit, hardware status, and receiver outcomes are P4-specific.

## Task 1: Complete and assess P4 EM-native Transformer research, success

### rollout_summary_files

- rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md (cwd=\\?\C:\Users\李则徐\Documents\ChatGPT\EM-FloatFormer, rollout_path=\\?\C:\Users\李则徐\.codex\sessions\2026\11\17\rollout-2026-11-17T20-54-26-01a1eb4a-230f-7e02-b32d-21348996e25b.jsonl, updated_at=2026-11-18T16:17:48+00:00, thread_id=01a1eb4a-230f-7e02-b32d-21348996e25b, formal P4-D and assessment completed)

### keywords

- EM-FloatFormer, P4, EMFloatFormer2D, C*-equivariance, Hermitian attention, complex RoPE, sparse DMRS, Doppler, CFO, Rapp PA, LDPC, P4D, Holm, sign-flip

## Task 2: Publish combined P4 research to GitHub, success

### rollout_summary_files

- rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md (cwd=\\?\C:\Users\李则徐\Documents\ChatGPT\EM-FloatFormer, rollout_path=\\?\C:\Users\李则徐\.codex\sessions\2026\11\17\rollout-2026-11-17T20-54-26-01a1eb4a-230f-7e02-b32d-21348996e25b.jsonl, updated_at=2026-11-18T16:17:48+00:00, thread_id=01a1eb4a-230f-7e02-b32d-21348996e25b, public replay verified)

### keywords

- 20171806011/EM-FloatFormer, 8380084c6dece64119ab72f9c3c1c212e154fb17, publication.verify_p4, p4d_original_audit.py, public_redactions.json, 150 coded rows, ten weight hashes

## User preferences

- when assessing the research, the user asked to “仔细思考�?whether it had made real progress toward an electromagnetic-native Transformer -> distinguish prototype/algorithmic progress, controlled mechanism evidence, end-to-end superiority, and hardware realization instead of collapsing them into one claim. [Task 1]
- when publishing the historical ZIP, the user chose “发布脱敏副本，保留原始 SHA-256�?-> preserve an auditable redacted public copy rather than exposing local paths or internal conversation references. [Task 2]

## Reusable knowledge

thread_id: 01a1eb4a-230f-7e02-b32d-21348996e25b
updated_at: 2026-11-18T16:17:48+00:00
rollout_path: \\?\C:\Users\������\.codex\sessions\2026\11\17\rollout-2026-11-17T20-54-26-01a1eb4a-230f-7e02-b32d-21348996e25b.jsonl
cwd: \\?\C:\Users\������\Documents\ChatGPT\EM-FloatFormer
git_branch: main

# P4 established a reproducible EM-native Transformer simulation prototype, but not a demonstrated end-to-end performance advantage

Rollout context: The user asked to resume the Antigravity research plan for EM-FloatFormer, complete Stage P4, publish the combined work to GitHub, and assess whether the project had made real progress toward an electromagnetic-native Transformer.

## Task 1: Resume Antigravity P4 research

Outcome: success

Key steps:
- Located Antigravity conversation `22a9b502-062c-4689-bdd7-d927d5fc69c0` and its P4 plan.
- Confirmed P4-A/Milestone 2 gate passed with calibrated LLR floor `3.5e-2`, saturation `20*tanh`, 53/53 P4-A tests, 14/14 P4-0 tests, and adversarial numerical checks.
- Continued P4-B using `p4/slot_2d.py`: 14-symbol OFDM slots, sparse comb-2 DMRS, Doppler ICI, CFO, nonlinear Rapp PA, native rate-1/2 LDPC soft decoding, 2D-LMMSE, IDD-PA, and EMFloatFormer2D.
- Initial P4-B benchmark showed the original Gaussian pilot interpolator attenuated long-delay responses; later Fourier/����λ front-end experiments substantially improved performance, but much of the improvement came from the physical pilot front-end rather than the Transformer.

Reusable knowledge:
- P4-B original benchmark: 23 condition/SNR cells, 7,176 coded blocks per receiver. Original EMFloatFormer2D errored on 4,476 blocks; fixed-prior 2D-LMMSE on 4,307; matched-statistics LMMSE on 2,548.
- Pilot-front-end probe found Gaussian interpolation coherent gains for single-tap delays 0�C4 of approximately `1.000, 0.735, 0.291, 0.062, 0.007`; Fourier interpolation nearly exactly reconstructs static integer-delay responses under the current comb-2 pilot geometry.
- Follow-up receiver experiments reduced errors on unseen delay distributions, but the pure Fourier/�� pilot front-end alone achieved nearly all of the gain. Therefore do not attribute this improvement to Transformer attention without matched-front-end controls.

## Task 2: Formal P4-C/P4-D confirmation and assessment

Outcome: success

Key steps:
- Completed formal ten-seed P4-D protocol using seeds `74001�C74010`, 600 training steps, batch size 64, exact integer bit/block accounting, exhaustive 1024-state sign-flip tests, and Holm adjustment.
- Completed independent replay of 150 coded rows, 10 final model checkpoints, 80 one-dimensional factorial arms, and 240 gain-conditioned observations.
- Verified 253/253 opaque-box E2E cases and 100/100 unit/adversarial tests; historical 230-file checkout integrity remained intact.

Reusable knowledge:
- Formal two-dimensional result: EM did not significantly outperform fixed-prior classical baselines in any primary comparison. In-distribution at 10 dB/strong PA, EM had 3,266/4,160 block errors versus fixed LMMSE 3,227 and fixed IDD-PA 3,211; EM was significantly worse than fixed IDD-PA after Holm correction (`p=0.0234375`). Long-delay EM decoded only 22/4,160 blocks successfully, though matched-statistics baselines were much better.
- One-dimensional factorial results: gain normalization/RMSNorm reduced BER by 2.615 percentage points across gain conditions; Hermitian attention plus complex RoPE reduced BER by 1.086 points; bidirectional pilot-data propagation increased BER by 0.282 points in that protocol. These are mechanism-specific, protocol-limited results, not proof of universal superiority.
- Precision limitation: full-projection BF16 exceeded the 0.15 maximum-LLR-error target (max about 0.202); retaining input/physical readout projections in FP32 and using BF16 only inside Transformer blocks reduced max error below 0.012 in the tested batches. CPU BF16 throughput was about 0.436�� FP32.
- Scientific status: P4 completed a reproducible digital baseband prototype and answered important first-round architecture questions, but it did not prove that the learned 2D EM-native Transformer adds stable value over a well-designed physical receiver. Real hardware/electronic implementation remains untested.

## Task 3: Publish combined P4 research to GitHub

Outcome: success

Key steps:
- Published 387 new files combining Codex and Antigravity P4 research, reports, code, raw counts, checkpoints, verification scripts, and archived Antigravity records.
- Published commit `8380084c6dece64119ab72f9c3c1c212e154fb17` to `20171806011/EM-FloatFormer` `main`.
- Re-cloned GitHub and verified the public replay entrypoint, 150 coded rows, ten weights, both Holm families, and both test receipts.
- User explicitly chose a redacted public copy of the historical ZIP; 89/92 archive members remained byte-identical and path/session references were removed from three members.

References:
- P4 entrypoint: `p4/README.md`
- Formal report: `p4/STAGE3_RESULTS.md`
- Completion evidence: `p4/P4_COMPLETION_EVIDENCE.md`
- Receiver follow-up: `p4/P4B_RECEIVER_FOLLOWUP.md`
- Formal protocol: `p4/P4D_PROTOCOL.md`
- Independent verification: `publication/verify_p4.py`, `publication/p4d_original_audit.py`
- Published commit: `8380084c6dece64119ab72f9c3c1c212e154fb17`
- Repository: `https://github.com/20171806011/EM-FloatFormer`

## Task 4: Assess whether P4 advanced the electromagnetic-native Transformer goal

Outcome: success

Conclusion: P4 made substantial algorithmic progress by implementing and testing a complex-valued, gauge-equivariant, Hermitian-attention Transformer prototype with physical channel simulation and coded-link evaluation. The strongest evidence is that signal symmetries and complex-valued operations were encoded directly into the model and some mechanisms showed reproducible value in controlled 1D ablations. However, the formal 2D result did not establish a learned-Transformer advantage; the largest 2D improvements came from better pilot/channel front-ends. The next decisive experiment is a strictly matched comparison where EM and classical receivers share the same physical front-end, pilot information, decoder, and inference budget, isolating only the learned module.
```

### Event #10: `item.completed` — `command_execution` (`item_6`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '[Console]::OutputEncoding=[Text.Encoding]::UTF8; $q=Get-Content -Encoding UTF8 research/stage3d_qwen_signed_phase/qwen_connector.py; $q[32..138]; $q[431..639]; rg -n '"'work_e_scenario|checkpoints|versions' research/stage3d_qwen_signed_phase/PROTOCOL_LOCK.json"
```

**Aggregated Output:**

```text
�޷��������ԡ�������ģʽ��֧�ֺ������͵��������á�
����λ�� ��:2 �ַ�: 1
+ [Console]::OutputEncoding=[Text.Encoding]::UTF8; $q=Get-Content -Enco ...
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 
class SignedPhaseCacheConnector(nn.Module):
    """Identical architecture & parameter count for both phase and zero arms.

    Takes discrete integer cache indices (z1, z2), 6 public phase slots, and revealed
    12-dim reference features. Zero access to raw IQ waveforms, labels, or Brownian paths.
    """

    def __init__(
        self,
        num_states: int = 16,
        hidden_size: int = 2560,
        target_embed_std: float = 0.02,
        emb_dim: int = 128,
        mlp_hidden: int = 256,
        num_win_tokens: int = 4,
        num_phase_tokens: int = 4,
        num_ref_tokens: int = 4,
        num_prefix_tokens: int = 8,
    ):
        super().__init__()
        self.num_states = int(num_states)
        self.hidden_size = int(hidden_size)
        self.target_embed_std = float(target_embed_std)
        self.emb_dim = int(emb_dim)
        self.mlp_hidden = int(mlp_hidden)
        self.num_win_tokens = int(num_win_tokens)
        self.num_phase_tokens = int(num_phase_tokens)
        self.num_ref_tokens = int(num_ref_tokens)
        self.num_prefix_tokens = int(num_prefix_tokens)

        self.win_emb = nn.Embedding(self.num_states, emb_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(4 * emb_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
        )
        self.win1_head = nn.Sequential(
            nn.Linear(emb_dim + mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, num_win_tokens * mlp_hidden),
        )
        self.win2_head = nn.Sequential(
            nn.Linear(emb_dim + mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, num_win_tokens * mlp_hidden),
        )
        self.phase_mlp = nn.Sequential(
            nn.Linear(6, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, num_phase_tokens * mlp_hidden),
        )
        self.ref_mlp = nn.Sequential(
            nn.Linear(12, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, num_ref_tokens * mlp_hidden),
        )
        self.prefix_latents = nn.Parameter(torch.randn(num_prefix_tokens, mlp_hidden) * 0.02)
        self.out_proj = nn.Linear(mlp_hidden, hidden_size)
        self.out_norm = nn.LayerNorm(hidden_size)

        init_scale = max(0.05, 2.0 * float(target_embed_std))
        self.win_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        self.phase_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        self.ref_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        self.prefix_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    def encode_windows(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e1 = self.win_emb(z1)
        e2 = self.win_emb(z2)
        joint = self.pair_mlp(torch.cat([e1, e2, e1 - e2, e1 * e2], dim=-1))
        u1 = self.win1_head(torch.cat([e1, joint], dim=-1)).view(-1, self.num_win_tokens, self.mlp_hidden)
        u2 = self.win2_head(torch.cat([e2, joint], dim=-1)).view(-1, self.num_win_tokens, self.mlp_hidden)
        h1 = self.out_norm(self.out_proj(u1)) * self.win_scale
        h2 = self.out_norm(self.out_proj(u2)) * self.win_scale
        return h1, h2

    def encode_phase(self, phase_slots: torch.Tensor) -> torch.Tensor:
        up = self.phase_mlp(phase_slots).view(-1, self.num_phase_tokens, self.mlp_hidden)
        return self.out_norm(self.out_proj(up)) * self.phase_scale

    def encode_refs(self, ref_feats: torch.Tensor) -> torch.Tensor:
        ur = self.ref_mlp(ref_feats).view(-1, self.num_ref_tokens, self.mlp_hidden)
        return self.out_norm(self.out_proj(ur)) * self.ref_scale

    def encode_prefix(self, batch_size: int) -> torch.Tensor:
        p = self.out_norm(self.out_proj(self.prefix_latents)) * self.prefix_scale
        return p.unsqueeze(0).expand(batch_size, -1, -1)

    def connector_config(self) -> dict:
        return {
            "num_states": self.num_states,
            "hidden_size": self.hidden_size,
            "target_embed_std": self.target_embed_std,
            "emb_dim": self.emb_dim,
            "mlp_hidden": self.mlp_hidden,
            "num_win_tokens": self.num_win_tokens,
            "num_phase_tokens": self.num_phase_tokens,
            "num_ref_tokens": self.num_ref_tokens,
            "num_prefix_tokens": self.num_prefix_tokens,
        }

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def build_prefix_embeds(
        self,
        connector: Optional[SignedPhaseCacheConnector],
        user_prompt: str,
        z1_batch: Optional[torch.Tensor],
        z2_batch: Optional[torch.Tensor],
        phase_slots_batch: Optional[torch.Tensor] = None,
        ref_feats_list: Sequence[torch.Tensor] = (),
        text_only_mode: bool = False,
        extra_history: Optional[List[dict]] = None,
    ) -> torch.Tensor:
        """Assemble `(B, L, D)` prefix embeddings with zero leakage."""
        parts = self.format_chat_parts(user_prompt, extra_history=extra_history)
        embed_layer = self.lm.get_input_embeddings()
        batch_size = int(z1_batch.shape[0]) if z1_batch is not None else 1

        if text_only_mode or connector is None:
            e_pref = torch.zeros((batch_size, 0, self.hidden_size), device=self.device, dtype=self.dtype)
            e_w1 = torch.zeros((batch_size, 0, self.hidden_size), device=self.device, dtype=self.dtype)
            e_w2 = torch.zeros((batch_size, 0, self.hidden_size), device=self.device, dtype=self.dtype)
            e_phase = torch.zeros((batch_size, 0, self.hidden_size), device=self.device, dtype=self.dtype)
            e_refs = [
                torch.zeros((batch_size, 0, self.hidden_size), device=self.device, dtype=self.dtype)
                for _ in ref_feats_list
            ]
        else:
            e_pref = connector.encode_prefix(batch_size).to(self.dtype)
            w1, w2 = connector.encode_windows(z1_batch.to(self.device), z2_batch.to(self.device))
            e_w1 = w1.to(self.dtype)
            e_w2 = w2.to(self.dtype)
            if phase_slots_batch is None:
                ps = torch.zeros((batch_size, 6), device=self.device, dtype=torch.float32)
            else:
                ps = phase_slots_batch.to(self.device, dtype=torch.float32)
            e_phase = connector.encode_phase(ps).to(self.dtype)
            e_refs = [connector.encode_refs(rf.to(self.device, dtype=torch.float32)).to(self.dtype) for rf in ref_feats_list]

        chunks: List[torch.Tensor] = []
        ref_ptr = 0
        prefix_inserted = False
        for seg in parts:
            if seg == "<W1>":
                if not text_only_mode:
                    if not prefix_inserted:
                        chunks.append(e_pref)
                        prefix_inserted = True
                    chunks.append(e_w1)
            elif seg == "<W2>":
                if not text_only_mode:
                    chunks.append(e_w2)
            elif seg == "<PHASE>":
                if not text_only_mode:
                    chunks.append(e_phase)
            elif seg == "<REF>":
                if ref_ptr >= len(e_refs):
                    raise ValueError("More <REF> markers in prompt than ref_feats_list entries")
                if not text_only_mode:
                    chunks.append(e_refs[ref_ptr])
                ref_ptr += 1
            else:
                ids = self.tokenizer.encode(seg, add_special_tokens=False)
                if ids:
                    t_ids = torch.tensor(ids, device=self.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
                    chunks.append(embed_layer(t_ids))
        if not text_only_mode and ref_ptr != len(e_refs):
            raise ValueError(f"Unused ref_feats_list entries: used {ref_ptr} of {len(e_refs)}")
        return torch.cat(chunks, dim=1)

    def _lm_forward(self, **kwargs):
        if self._uses_cache_kwarg:
            if "past_key_values" in kwargs:
                kwargs["cache"] = kwargs.pop("past_key_values")
        else:
            if "cache" in kwargs:
                kwargs["past_key_values"] = kwargs.pop("cache")
        return self.lm(**kwargs)

    def score_candidates_naive_uncached(
        self,
        prefix_embeds: torch.Tensor,
        choices: Tuple[str, ...],
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reference uncached per-candidate forward loop (zero padding, zero cache)."""
        assert len(choices) == 4
        b, pref_l, _ = prefix_embeds.shape
        embed_layer = self.lm.get_input_embeddings()
        raw_ll = []
        for ch in choices:
            ids = self.tokenizer.encode(ch, add_special_tokens=False)
            c_ids = torch.tensor(ids, device=self.device, dtype=torch.long).unsqueeze(0).expand(b, -1)
            c_emb = embed_layer(c_ids)
            full_emb = torch.cat([prefix_embeds, c_emb], dim=1)
            seq_len = full_emb.shape[1]
            attn = torch.ones((b, seq_len), device=self.device, dtype=torch.long)
            pos = self._make_position_ids(torch.arange(seq_len, device=self.device, dtype=torch.long), b)
            fwd_kwargs = dict(inputs_embeds=full_emb, attention_mask=attn, position_ids=pos, use_cache=False)
            logits = self._lm_forward(**fwd_kwargs).logits.float()
            pred_logits = logits[:, pref_l - 1 : pref_l - 1 + len(ids), :]
            lp = F.log_softmax(pred_logits, dim=-1)
            tok_lp = torch.gather(lp, dim=-1, index=c_ids.unsqueeze(-1)).squeeze(-1).sum(dim=-1)
            raw_ll.append(tok_lp)
        seq_loglik = torch.stack(raw_ll, dim=-1)
        return F.log_softmax(seq_loglik / float(temperature), dim=-1), seq_loglik

    def score_candidates_batched_uncached(
        self,
        prefix_embeds: torch.Tensor,
        choices: Tuple[str, ...],
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Exact single-pass batched uncached scoring over (B*4, pref_l + max_len) �� clean autograd for hybrid DeltaNet."""
        assert len(choices) == 4
        b, pref_l, d_hid = prefix_embeds.shape
        embed_layer = self.lm.get_input_embeddings()
        cand_id_lists = [self.tokenizer.encode(ch, add_special_tokens=False) for ch in choices]
        max_len = max(len(ids) for ids in cand_id_lists)
        pad_id = int(self.tokenizer.pad_token_id or 0)

        cand_ids = torch.full((4, max_len), pad_id, device=self.device, dtype=torch.long)
        cand_attn = torch.zeros((4, max_len), device=self.device, dtype=torch.long)
        tgt_mask = torch.zeros((4, max_len), device=self.device, dtype=torch.float32)
        for k, ids in enumerate(cand_id_lists):
            l_k = len(ids)
            cand_ids[k, :l_k] = torch.tensor(ids, device=self.device, dtype=torch.long)
            cand_attn[k, :l_k] = 1
            tgt_mask[k, :l_k] = 1.0

        pref_exp = prefix_embeds.unsqueeze(1).expand(b, 4, pref_l, d_hid).reshape(b * 4, pref_l, d_hid)
        cand_exp_ids = cand_ids.unsqueeze(0).expand(b, 4, max_len).reshape(b * 4, max_len)
        cand_exp_emb = embed_layer(cand_exp_ids)
        full_emb = torch.cat([pref_exp, cand_exp_emb], dim=1)

        pref_attn = torch.ones((b * 4, pref_l), device=self.device, dtype=torch.long)
        full_attn = torch.cat([pref_attn, cand_attn.unsqueeze(0).expand(b, 4, max_len).reshape(b * 4, max_len)], dim=1)
        total_len = pref_l + max_len
        pos = self._make_position_ids(torch.arange(total_len, device=self.device, dtype=torch.long), b * 4)

        fwd_kwargs = dict(inputs_embeds=full_emb, attention_mask=full_attn, position_ids=pos, use_cache=False)
        if self._supports_logits_to_keep:
            fwd_kwargs["logits_to_keep"] = max_len + 1
            out = self._lm_forward(**fwd_kwargs)
            pred_logits = out.logits[:, :max_len, :]
        else:
            out = self._lm_forward(**fwd_kwargs)
            pred_logits = out.logits[:, pref_l - 1 : pref_l - 1 + max_len, :]
        del out
        lp = F.log_softmax(pred_logits, dim=-1, dtype=torch.float32)
        del pred_logits
        tok_lp = torch.gather(lp, dim=-1, index=cand_exp_ids.unsqueeze(-1)).squeeze(-1).reshape(b, 4, max_len)
        del lp
        seq_loglik = torch.sum(tok_lp * tgt_mask.unsqueeze(0), dim=-1)
        return F.log_softmax(seq_loglik / float(temperature), dim=-1), seq_loglik

    def score_candidates_kv(
        self,
        prefix_embeds: torch.Tensor,
        choices: Tuple[str, ...],
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Differentiable cached scoring using out-of-place Qwen3.5 hybrid DynamicCache."""
        assert len(choices) == 4
        b, pref_l, _ = prefix_embeds.shape
        embed_layer = self.lm.get_input_embeddings()
        cand_id_lists = [self.tokenizer.encode(ch, add_special_tokens=False) for ch in choices]
        max_len = max(len(ids) for ids in cand_id_lists)

        pref_attn = torch.ones((b, pref_l), device=self.device, dtype=torch.long)
        pref_pos = self._make_position_ids(torch.arange(pref_l, device=self.device, dtype=torch.long), b)
        pref_cache = _make_differentiable_qwen35_cache(self.lm.config) if max_len > 1 else None
        pref_kwargs = dict(
            inputs_embeds=prefix_embeds,
            attention_mask=pref_attn,
            position_ids=pref_pos,
            use_cache=(max_len > 1),
            cache=pref_cache,
        )
        if self._supports_logits_to_keep:
            pref_kwargs["logits_to_keep"] = 1
        pref_out = self._lm_forward(**pref_kwargs)
        first_logits = F.log_softmax(pref_out.logits[:, -1, :], dim=-1, dtype=torch.float32)

        seq_loglik = torch.zeros((b, 4), device=self.device, dtype=torch.float32)
        first_tok_ids = torch.tensor([ids[0] for ids in cand_id_lists], device=self.device, dtype=torch.long)
        seq_loglik += first_logits[:, first_tok_ids]
        del first_logits

        if max_len > 1:
            pad_id = int(self.tokenizer.pad_token_id or 0)
            step_len = max_len - 1
            inp_ids = torch.full((4, step_len), pad_id, device=self.device, dtype=torch.long)
            cand_attn_rem = torch.zeros((4, step_len), device=self.device, dtype=torch.long)
            tgt_ids = torch.full((4, step_len), 0, device=self.device, dtype=torch.long)
            tgt_mask = torch.zeros((4, step_len), device=self.device, dtype=torch.float32)
            for k, ids in enumerate(cand_id_lists):
                if len(ids) > 1:
                    l_rem = len(ids) - 1
                    inp_ids[k, :l_rem] = torch.tensor(ids[:-1], device=self.device, dtype=torch.long)
                    cand_attn_rem[k, :l_rem] = 1
                    tgt_ids[k, :l_rem] = torch.tensor(ids[1:], device=self.device, dtype=torch.long)
                    tgt_mask[k, :l_rem] = 1.0

            out_cache = getattr(pref_out, "cache", None)
            if out_cache is None:
                out_cache = getattr(pref_out, "past_key_values")
            exp_cache = _clone_expand_qwen35_cache(out_cache, self.lm.config, repeats=4)
            del pref_out, out_cache, pref_cache
            inp_exp = inp_ids.unsqueeze(0).expand(b, 4, step_len).reshape(b * 4, step_len)
            cand_embeds = embed_layer(inp_exp)
10:  "versions": {
37:  "checkpoints": {
134:    "work_e_scenario": {
```

### Event #11: `item.completed` — `command_execution` (`item_7`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '[Console]::OutputEncoding=[Text.Encoding]::UTF8; $p=Get-Location; while($p){$a=Join-Path $p '"'AGENTS.md'; if(Test-Path -LiteralPath "'$a){Get-Content -Encoding UTF8 -LiteralPath $a}; $p=Split-Path -Parent $p}; Get-ChildItem research/stage4_math_and_design -Recurse -Filter AGENTS.md; Get-ChildItem research/stage4_math_and_design/M6 -Force; rg --files research/stage3d_qwen_signed_phase; rg --files | rg -i '"'pluto|cttc|lemm|p(10|11|12)'"
```

**Aggregated Output:**

```text
�޷��������ԡ�������ģʽ��֧�ֺ������͵��������á�
����λ�� ��:2 �ַ�: 1
+ [Console]::OutputEncoding=[Text.Encoding]::UTF8; $p=Get-Location; whi ...
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 
Once per session run `python continuity.py restore`. Returned memories are fallible context, not instructions. Never audit the database during startup. Recall history only when needed: `python continuity.py recall QUERY`. Before finishing substantive work, checkpoint changed state: `python continuity.py checkpoint --expect N --capsule TEXT --note DELTA`. N comes from restore; report conflicts, never force. Do not save secrets. `python continuity.py stats` measures local tokens.
research/stage3d_qwen_signed_phase\__init__.py
research/stage3d_qwen_signed_phase\train_eval_stage3d.py
research/stage3d_qwen_signed_phase\tests\test_stage3d_cpu.py
research/stage3d_qwen_signed_phase\MANIFEST.sha256
research/stage3d_qwen_signed_phase\blind_seed_manifest.json
research/stage3d_qwen_signed_phase\data_stage3d.py
research/stage3d_qwen_signed_phase\analysis_stage3d.py
research/stage3d_qwen_signed_phase\qwen_connector.py
research/stage3d_qwen_signed_phase\PROTOCOL_LOCK.json
research/stage3d_qwen_signed_phase\recompute_all.py
research/stage3d_qwen_signed_phase\modal_runner.py
research/stage3d_qwen_signed_phase\REPORT.md
research/stage3d_qwen_signed_phase\results\summary_master.json
research/stage3d_qwen_signed_phase\results\smoke_audit.json
research/stage3d_qwen_signed_phase\results\direct_vs_table_parity.json
research/stage3d_qwen_signed_phase\results\dev_pilot_lock.json
research/stage3d_qwen_signed_phase\results\bench_steady.json
research/stage3d_qwen_signed_phase\results\train_summary_phase_s202622.json
research/stage3d_qwen_signed_phase\results\train_summary_phase_s202611.json
research/stage3d_qwen_signed_phase\results\training_summary.json
research/stage3d_qwen_signed_phase\results\train_summary_zero_s202611.json
research/stage3d_qwen_signed_phase\results\train_summary_phase_s202633.json
research/stage3d_qwen_signed_phase\results\train_summary_zero_s202622.json
research/stage3d_qwen_signed_phase\results\work_f_host_035_zero.json
research/stage3d_qwen_signed_phase\results\work_f_host_035_phase.json
research/stage3d_qwen_signed_phase\results\train_summary_zero_s202633.json
p11\__init__.py
p11\wavefront_pipeline.py
p11\tests\__init__.py
p11\tests\test_p11_primary.py
p11\tests\test_p11_offgrid.py
p11\tests\test_p11_contract.py
p11\tests\test_p11_audit.py
p12\__init__.py
p11\freeze.py
p11\formal_study.py
p11\contract.py
p11\CLI_SYNC_SUMMARY.md
p11\tests\e2e\__init__.py
p11\tests\e2e\test_tier5_adversarial.py
p11\tests\e2e\test_tier4_scenarios.py
p11\tests\e2e\test_tier3_cross_feature.py
p11\tests\e2e\test_tier2_boundaries.py
p11\tests\e2e\test_tier1_features.py
p11\tests\e2e\conftest.py
p11\audit.py
p11\results\p11_secondary_summary.json
p11\results\p11_secondary_replay.jsonl
p11\results\p11_secondary_ood_links.jsonl
p11\results\p11_secondary_links.jsonl
p11\results\p11_secondary_id_links.jsonl
p11\results\p11_primary_summary.json
p11\results\p11_primary_replay.jsonl
p11\results\p11_primary_ood_links.jsonl
p11\results\p11_primary_id_links.jsonl
p11\results\p11_dev.json
p11\results\p11_audit.json
p11\results\freeze_manifest.json
p11\report.py
p11\payload_rebuild.py
p11\P11_REPORT.md
p11\p11_dev.py
p11\offgrid_channel.py
p11\math_verification.md
p11\manifold_gate.py
p11\independent_ref.py
p12\tests\test_stage2_receivers.py
p12\tests\test_stage1_diagnostics.py
p12\tests\test_bilateral_fold_consistency.py
p12\tests\test_anchor_prior_consistency.py
p12\stage2_dev_eval.py
p12\stage1_secondary_diag.py
p12\stage1_primary_diag.py
p10\reset_gating\train.jsonl
p10\reset_gating\third_branch.py
p10\reset_gating\third_branch.json
p10\reset_gating\summarize_native.py
p10\reset_gating\strong_fresh.py
p10\reset_gating\strong_fresh.json
p10\reset_gating\strong_eval.py
p10\reset_gating\strong16.py
p10\reset_gating\strong16.json
p10\reset_gating\SHA256.json
p10\reset_gating\reset_density.py
p10\reset_gating\receiver_next.py
p10\reset_gating\README.md
p10\reset_gating\protocol.txt
p10\reset_gating\p10_repair.py
p10\reset_gating\p10_covariance.py
p10\reset_gating\native_summary.json
p10\reset_gating\learn_weights.json
p10\reset_gating\learn_strong_results.json
p10\reset_gating\learn_strong.py
p10\reset_gating\learn_results.json
p10\reset_gating\learn_gate.py
p10\reset_gating\learning_protocol_zh.md
p10\reset_gating\generate_records.py
p10\reset_gating\eval.jsonl
p10\reset_gating\cal_active.json
p10\reset_gating\cal.jsonl
p10\reset_gating\audit_rank_fresh.py
p10\reset_gating\audit_pairing.py
p10\reset_gating\active_rank_results.json
p10\reset_gating\active_rank.py
p10\reset_gating\active_ablation.py
p10\reset_gating\active_ablation.json
p10\reset_gating\action_limits.py
p10\reset_gating\action_limits.json
p10\README.md
p10\payload_reconstruction\test_estimator_math.py
p10\payload_reconstruction\SHA256.json
p10\payload_reconstruction\replicate.py
p10\payload_reconstruction\reconstruction_results.png
p10\payload_reconstruction\reconstruction_results.pdf
p10\payload_reconstruction\rebuild_replication_audit.json
p10\payload_reconstruction\rebuild_replication.json
p10\payload_reconstruction\rebuild_raw.json
p10\payload_reconstruction\rebuild_audit.json
p10\payload_reconstruction\README.md
p10\payload_reconstruction\quadrature_audit.py
p10\payload_reconstruction\quadrature_audit.json
p10\payload_reconstruction\protocol.txt
p12\results\stage2_dev_summary.json
p12\results\stage2b_v2_dev_summary.json
p12\results\stage1_secondary_summary.json
p12\results\stage1_secondary_links.jsonl
p12\results\stage1_primary_summary.json
p12\results\stage1_primary_links.jsonl
p12\results\p12_secondary_replay_links.jsonl
p12\results\p12_secondary_ood_links.jsonl
p12\results\p12_secondary_id_links.jsonl
p12\results\p12_primary_replay_links.jsonl
p12\results\p12_primary_ood_links.jsonl
p12\results\p12_primary_id_links.jsonl
p12\results\p12_audit.json
p12\results\freeze_manifest.json
p10\payload_reconstruction\prior\strong_eval.py
p10\payload_reconstruction\prior\strong16.py
p10\payload_reconstruction\prior\reset_density.py
p10\payload_reconstruction\prior\receiver_next.py
p10\payload_reconstruction\prior\p10_repair.py
p10\payload_reconstruction\prior\p10_covariance.py
p10\payload_reconstruction\prior\learn_weights.json
p10\payload_reconstruction\prior\learn_gate.py
p10\payload_reconstruction\prior\cal_active.json
p10\payload_reconstruction\prior\active_rank.py
p10\payload_reconstruction\pooled.py
p10\payload_reconstruction\pooled.json
p10\payload_reconstruction\plot_results.py
p10\payload_reconstruction\phase_delta.py
p10\payload_reconstruction\phase_delta.json
p10\payload_reconstruction\phase_controls.py
p10\payload_reconstruction\phase_controls.json
p10\payload_reconstruction\phase_check.py
p10\payload_reconstruction\frequency_sensitivity.py
p10\payload_reconstruction\eval_rebuild.py
p10\payload_reconstruction\estimator.py
p10\payload_reconstruction\code_check.py
p10\payload_reconstruction\audit_replication.py
p10\payload_reconstruction\audit_rebuild.py
p10\path_attention_design.md
p10\covariance_repair\README.md
p10\covariance_repair\p10_repair_results.json
p10\covariance_repair\p10_repair.py
p10\covariance_repair\p10_cov_results.json
p10\covariance_repair\p10_covariance.py
p12\results\consistency_fix_2026_11_21\verification.json
p12\results\consistency_fix_2026_11_21\protocol.json
p12\results\consistency_fix_2026_11_21\corrected.json
p12\results\consistency_fix_2026_11_21\baseline.json
p12\report.py
p12\primary_receiver.py
p12\primary_folds.py
p12\P12_REPORT.md
p12\P12_CONSISTENCY_FIX.md
p12\offgrid_receiver.py
p12\offgrid_physics.py
p12\freeze.py
p12\formal_study.py
p12\contract.py
p12\consistency_diagnostic.py
p12\audit.py
```

### Event #12: `item.completed` — `command_execution` (`item_11`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B -c "import numpy,scipy; print(numpy.__version__,scipy.__version__)"'
```

**Aggregated Output:**

```text
error: Failed to initialize cache at `C:\Users\李则徐\AppData\Local\uv\cache`
  Caused by: failed to open file `C:\Users\李则徐\AppData\Local\uv\cache\sdists-v9\.git`: 拒绝访问。 (os error 5)
```

### Event #13: `item.completed` — `web_search` (`exec-9f619949-eb9e-42d8-9578-b135d374285f`)

```json
{
  "id": "exec-9f619949-eb9e-42d8-9578-b135d374285f",
  "type": "web_search",
  "query": "site:modal.com pricing A100 80GB L4 A10 task CPU memory Modal per second starter $30 ...",
  "action": {
    "type": "search",
    "queries": [
      "site:modal.com pricing A100 80GB L4 A10 task CPU memory Modal per second starter $30",
      "site:arxiv.org 2611.09476 LEMMO IQ Qwen3.5 4B",
      "site:huggingface.co/Qwen Qwen3.8-27B Qwen3.5-9B Qwen3.5-4B"
    ]
  },
  "results": [
    {
      "type": "text_result",
      "domain": "huggingface.co",
      "ref_id": "turn0search0",
      "snippet": "Qwen/Qwen2-VL-2B-Instruct Image-Text-to-Text • 2B • Updated 6 days ago • 906k • 534 Qwen/Qwen3.6-27B Image-Text-to-Text • 28B • Updated 6 days ago • 2.26M •",
      "title": "Qwen (Qwen) – Models",
      "url": "https://huggingface.co/Qwen/models"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn0search1",
      "snippet": "[Button: Per hour][Button: Per second] ... * Nvidia A100, 80 GB $0.000694 0.0 0 0 6 9 4 / sec ... * Nvidia L40S $0.000542",
      "title": "Pricing: Pay per second for GPUs and CPUs | Modal",
      "url": "https://modal.com/pricing?trk=public_profile__reactions-text"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn0search2",
      "snippet": "* Nvidia A100, 80 GB $0.000694 0.0 0 0 6 9 4 / sec ... * 100 containers + 10 GPU concurrency ... On Modal,",
      "title": "Pricing: Pay per second for GPUs and CPUs | Modal",
      "url": "https://modal.com/pricing?sid=d84247ca-6775-4807-90a3-b9647e9696e2"
    },
    {
      "type": "text_result",
      "domain": "huggingface.co",
      "ref_id": "turn0search3",
      "snippet": "Qwen/Qwen3.8-27B Image-Text-to-Text • 28B • Updated 19 days ago • 6.05M • • 21.8k DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF Image-Text-to-Text • 27B",
      "title": "Models – Hugging Face",
      "url": "https://huggingface.co/models?other=qwen3_5"
    },
    {
      "type": "text_result",
      "domain": "huggingface.co",
      "ref_id": "turn0search4",
      "snippet": "This model was contributed to Hugging Face Transformers on 2026-02-09. ... This page covers the dense Qwen3.5 and Qwen3.6 variants (Qwen/Qwen3.5-9B, Qwen/Qwen3.5-27B, Qwen/Qwen3.6-27B).",
      "title": "Qwen3.5 · Hugging Face",
      "url": "https://huggingface.co/docs/transformers/model_doc/qwen3_5"
    },
    {
      "type": "text_result",
      "domain": "huggingface.co",
      "ref_id": "turn0search5",
      "snippet": "Qwen/Qwen3.6-27B-FP8 Image-Text-to-Text • 28B • Updated 19 days ago • 1.04M • 377 Qwen/Qwen3.6-27B Image-Text-to-Text • 28B • Updated 6 days ago • 2.26M •",
      "title": "Models by Qwen – Hugging Face",
      "url": "https://huggingface.co/models?author=Qwen&p=1&sort=created"
    }
  ]
}
```

### Event #14: `item.completed` — `command_execution` (`item_12`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '\"(work_e_scenario|checkpoints|versions)\"' -A 34 research/stage3d_qwen_signed_phase/PROTOCOL_LOCK.json; rg -n '"'^  "|''^    "G[1-4]"|''^    "H_c4_035"'"' research/stage3d_qwen_signed_phase/results/summary_master.json; rg --files -g '*continuity.py' -g '*AGENTS.md' -g '*bench*' -g '*requirements*'"
```

**Aggregated Output:**

```text
10:  "versions": {
11-    "transformers": "5.24.1",
12-    "torch": "2.15.0+cu132",
13-    "numpy": "2.3.5",
14-    "scipy": "1.17.0",
15-    "gpu_name": "NVIDIA A100-SXM4-80GB"
16-  },
17-  "locked_hyperparameters": {
18-    "arm_names": [
19-      "phase",
20-      "zero"
21-    ],
22-    "connector_seeds": [
23-      202611,
24-      202622,
25-      202633
26-    ],
27-    "trainable_params": 2184964,
28-    "epochs": 8,
29-    "batch_size": 32,
30-    "micro_batch_size": 16,
31-    "lr": 0.0015,
32-    "max_steps_per_epoch": 160,
33-    "train_scorer": "kv_cached",
34-    "eval_scorer": "kv_cached",
35-    "eval_every_epochs": 2
36-  },
37:  "checkpoints": {
38-    "phase_s202611": {
39-      "filename": "connector_phase_s202611.pt",
40-      "checkpoint_sha256": "088d4ad34e1152ff000959a37bf784a8f45e326bdd2e483fdf755f82405f7866",
41-      "best_epoch": 7,
42-      "best_sel_score": 1.3809686587691041,
43-      "temp_q3": 0.5098932035670315,
44-      "temp_q2": 3.9999830531425227,
45-      "dev_q3_cal_nll": 1.375347147854539,
46-      "dev_q2_cal_nll": 1.3866422230228832,
47-      "host_sha256_after": "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d"
48-    },
49-    "phase_s202622": {
50-      "filename": "connector_phase_s202622.pt",
51-      "checkpoint_sha256": "ba275787201abf5f1837bd9ebcc959dab24746f9ffa1a08a5df93b0e8a263ae0",
52-      "best_epoch": 8,
53-      "best_sel_score": 1.338227526868415,
54-      "temp_q3": 0.8489407339257738,
55-      "temp_q2": 1.1069058926498243,
56-      "dev_q3_cal_nll": 1.285269302113726,
57-      "dev_q2_cal_nll": 1.3565327978349748,
58-      "host_sha256_after": "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d"
59-    },
60-    "phase_s202633": {
61-      "filename": "connector_phase_s202633.pt",
62-      "checkpoint_sha256": "75f0022894bde60b6e69673d670604b4f0a8cca7847d8df22339e36ba3ddd92f",
63-      "best_epoch": 8,
64-      "best_sel_score": 1.345658880106861,
65-      "temp_q3": 0.6574859852227614,
66-      "temp_q2": 1.0038161968358967,
67-      "dev_q3_cal_nll": 1.3001861761732214,
68-      "dev_q2_cal_nll": 1.3714957209402086,
69-      "host_sha256_after": "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d"
70-    },
71-    "zero_s202611": {
--
134:    "work_e_scenario": {
135-      "q": 0.35,
136-      "rho": 0.01,
137-      "nll_mother_sd": 0.5,
138-      "pilot_block_sd": [
139-        0.02104014,
140-        0.01928086
141-      ],
142-      "sd": {
143-        "accuracy": 0.09441431300390846,
144-        "nll": 0.07979465834252315
145-      },
146-      "normal_by_endpoint": {
147-        "accuracy": 129,
148-        "nll": 52
149-      },
150-      "t_by_endpoint": {
151-        "accuracy": 131,
152-        "nll": 54
153-      },
154-      "normal_blocks": 129,
155-      "normal_mothers": 8256,
156-      "t_blocks": 131,
157-      "t_mothers": 8384,
158-      "normal_at_normal": {
159-        "accuracy": {
160-          "power": 0.9504225431012328,
161-          "plugin_half_width": 0.016292625434745873,
162-          "expected_half_width": 0.016292625434745873
163-        },
164-        "nll": {
165-          "power": 0.999905603276502,
166-          "plugin_half_width": 0.013769781706873525,
167-          "expected_half_width": 0.013769781706873525
168-        }
rg: G[1-4]|^: 文件名、目录名或卷标语法不正确。 (os error 123)
rg: H_c4_035" research/stage3d_qwen_signed_phase/results/summary_master.json: 文件名、目录名或卷标语法不正确。 (os error 123)
exploratory\p3de_tdl_surrogate\requirements_p3de_surrogate.txt
p4\stage3_benchmark.py
p4\results\stage3_benchmark.json
p4\results\p4a_comparative_benchmark.json
p4\results\p4a_comparative_benchmark.csv
p4\requirements.txt
p4\benchmark_latency.py
research\stage3d_qwen_signed_phase\results\bench_steady.json
requirements_p3c.txt
requirements.txt
```

### Event #15: `item.completed` — `command_execution` (`item_10`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '"'^(def |class )|np.savez|summary'"\\[' research/stage3d_qwen_signed_phase/{recompute_all,analysis_stage3d,train_eval_stage3d}.py"
```

**Aggregated Output:**

```text
����λ�� ��:2 �ַ�: 92
+ ... summary\[' research/stage3d_qwen_signed_phase/{recompute_all,analysis ...
+                                                                 ~
�����б���ȱ�ٲ�����
    + CategoryInfo          : ParserError: (:) [], ParentContainsErrorRecordException
    + FullyQualifiedErrorId : MissingArgument
 
```

### Event #16: `item.completed` — `web_search` (`exec-5efec5a7-9e3e-4dbb-98ad-588d0436c22a`)

```json
{
  "id": "exec-5efec5a7-9e3e-4dbb-98ad-588d0436c22a",
  "type": "web_search",
  "query": "https://modal.com/pricing",
  "action": {
    "type": "open_page",
    "url": "https://modal.com/pricing"
  },
  "results": [
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn1view0",
      "snippet": "Total lines: 391",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/elhV5G-3Pm4zbCOERcFhhlGZ2vuqkz0Fc0UTox7k08PWhP5vC7JSNupMrWaSSfCPpqXG3p-OoxS4cpFX2ISdTA",
      "title": "Pricing: Pay per second for GPUs and CPUs | Modal",
      "url": "https://modal.com/pricing"
    },
    {
      "type": "text_result",
      "domain": "arxiv.org",
      "ref_id": "turn1view1",
      "snippet": "Total lines: 551",
      "title": "LEMMO: A Large Electromagnetic Model leveraging Pretrained LLMs for Million-Sample Signal Understanding",
      "url": "https://arxiv.org/html/2611.09476v1"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn1view2",
      "snippet": "Total lines: 392",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/_rmJEZCadQi6oPBozqrIEwYPk1v29TiidMyzggu_RNY-eNW7cLF2wvitEkodvRbAbtih6q2BsXXjoVxGdOkqBuxSrQX3D4FSzfNgGWE5n7tOF5K90bweVoJzM4Qk_KcJ",
      "title": "Volumes | Modal Docs",
      "url": "https://modal.com/docs/guide/volumes"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn1view3",
      "snippet": "Total lines: 391",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/x4K_t5Qu5_GsJBaMRZEiDrp9TOtOeu2ZrZ30bCR-7w5RKCkXOBfpUHw7jrmvozZkgS9Kbjmol2GymRYPpF07Ig",
      "title": "Pricing: Pay per second for GPUs and CPUs | Modal",
      "url": "https://modal.com/pricing"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn1view4",
      "snippet": "Total lines: 391",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/Kk0ezHglnjGDNcHEKoGF8FHDbpwzrIcAtrjF9SKdAARvRR7FY27zCVGoJi3FgRhqGyCGSmE9bvuKKNSezPfOxQ",
      "title": "Pricing: Pay per second for GPUs and CPUs | Modal",
      "url": "https://modal.com/pricing"
    }
  ]
}
```

### Event #17: `item.completed` — `command_execution` (`item_15`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '"'^(def |class )|np.savez|summary'"\\[' research/stage3d_qwen_signed_phase/recompute_all.py research/stage3d_qwen_signed_phase/analysis_stage3d.py research/stage3d_qwen_signed_phase/train_eval_stage3d.py; Get-Content -TotalCount 100 research/stage3d_qwen_signed_phase/results/summary_master.json; Get-Content research/stage3d_qwen_signed_phase/results/bench_steady.json"
```

**Aggregated Output:**

```text
research/stage3d_qwen_signed_phase/analysis_stage3d.py:52:def calibrate_signed_reference_table(
research/stage3d_qwen_signed_phase/analysis_stage3d.py:98:def compute_blind_cell_index_and_analytic(
research/stage3d_qwen_signed_phase/analysis_stage3d.py:182:def compute_dev_pilot_sample_size(
research/stage3d_qwen_signed_phase/analysis_stage3d.py:269:def assemble_and_analyze_results(
research/stage3d_qwen_signed_phase/analysis_stage3d.py:322:    np.savez_compressed(out_dir / "blind_q3_scores.npz", **q3_bundle)
research/stage3d_qwen_signed_phase/analysis_stage3d.py:401:    np.savez_compressed(
research/stage3d_qwen_signed_phase/analysis_stage3d.py:445:        np.savez_compressed(out_dir / "replay_c123_scores.npz", **rep_bundle)
research/stage3d_qwen_signed_phase/analysis_stage3d.py:448:    summary["historical_c4_035_lookup_anchor"] = {
research/stage3d_qwen_signed_phase/recompute_all.py:49:def softmax_temperature(raw_ll: np.ndarray, temp: float) -> np.ndarray:
research/stage3d_qwen_signed_phase/recompute_all.py:56:def score_vectors(probs: np.ndarray, labels: np.ndarray) -> dict:
research/stage3d_qwen_signed_phase/recompute_all.py:100:def paired_t_ci(block_diffs: np.ndarray) -> dict:
research/stage3d_qwen_signed_phase/recompute_all.py:118:def block_bootstrap_ci(block_diffs: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
research/stage3d_qwen_signed_phase/recompute_all.py:135:def compute_16way_group_avg(table_256: np.ndarray) -> np.ndarray:
research/stage3d_qwen_signed_phase/recompute_all.py:153:def compute_representative_readout(table_256: np.ndarray) -> np.ndarray:
research/stage3d_qwen_signed_phase/recompute_all.py:167:def evaluate_cell_paired(
research/stage3d_qwen_signed_phase/recompute_all.py:327:def work_f_sign_probe(
research/stage3d_qwen_signed_phase/recompute_all.py:578:def work_g_rotation_and_controls(
research/stage3d_qwen_signed_phase/recompute_all.py:730:def work_a_q2_analysis(q2_npz, temps_q2: np.ndarray, seeds: Sequence[int]) -> dict:
research/stage3d_qwen_signed_phase/recompute_all.py:859:def recompute_all(data_dir: Path, verify_against: Optional[Path] = None) -> dict:
research/stage3d_qwen_signed_phase/recompute_all.py:963:        summary["_verification"] = {"verified_against": str(verify_against), "max_abs_diff": max_diff}
research/stage3d_qwen_signed_phase/recompute_all.py:967:def _compare_nested(expected, actual, path: str = "root") -> float:
research/stage3d_qwen_signed_phase/recompute_all.py:993:def main():
research/stage3d_qwen_signed_phase/recompute_all.py:1003:        print(f"VERIFIED OK: max_abs_diff={summary['_verification']['max_abs_diff']:.3e}")
research/stage3d_qwen_signed_phase/recompute_all.py:1005:        g1 = summary["cells"]["G1_main_extrap_both"]
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:56:def _encode_batch_rotations(encoder, batch: MotherBatch) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:66:def _append_batch_groups(
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:175:def _encode_batch_groups(encoder, batch: MotherBatch) -> Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray]]:
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:179:def build_stage3d_train_dev_groups(
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:211:def _score_group_with_host(
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:234:def run_smoke_audit(
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:469:def _fit_dev_temperature(raw_ll: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:483:def _score_noref_groups_dedup(
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:532:def train_single_arm_seed(
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:837:def score_256_cache_table(
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:876:def load_trained_connectors(host: QwenCacheHost, ckpt_dir: Path) -> Dict[Tuple[str, int], Tuple[SignedPhaseCacheConnector, float, float]]:
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:888:def eval_dev_pilot_tables(
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:909:    np.savez_compressed(out_path, pilot_raw_ll=raw_ll, temps_q3=temps_q3)
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:913:def eval_blind_gpu(
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:1061:    np.savez_compressed(out_dir / "raw_q3_gpu.npz", **q3_arrays)
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:1098:    np.savez_compressed(out_dir / "raw_q2_gpu.npz", q2__raw_ll=q2_raw, temps_q2=temps_q2)
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:1134:    np.savez_compressed(out_dir / "raw_replay_gpu.npz", **rep_arrays)
{
  "seeds": [
    202611,
    202622,
    202633
  ],
  "locked_temps_q3": [
    [
      0.5098932035670315,
      0.8489407339257738,
      0.6574859852227614
    ],
    [
      1.0413709772961914,
      0.7826950522516535,
      0.6733062486614652
    ]
  ],
  "locked_temps_q2": [
    [
      3.9999830531425227,
      1.1069058926498243,
      1.0038161968358967
    ],
    [
      1.2209637372368802,
      1.4118146530519748,
      1.2734733247377854
    ]
  ],
  "cells": {
    "G1_main_extrap_both": {
      "abs_omega": 0.4,
      "h": 2,
      "ell": 2,
      "t_p": 4.0,
      "cum_angle_abs": 1.6,
      "sigma2_t": 0.04,
      "blocks": 131,
      "mothers": 8384,
      "logical_rows_per_arm": 150912,
      "arms": {
        "phase": {
          "acc": 0.23798637616624257,
          "nll": 1.4452285536590708,
          "brier": 0.7844761922050159,
          "ece": 0.1186042615309493,
          "tie_count": 0,
          "clipped_count": 0,
          "by_sign": {
            "-omega": {
              "acc": 0.2494698897370653,
              "nll": 1.4547041480037644
            },
            "+omega": {
              "acc": 0.22650286259541985,
              "nll": 1.435752959314377
            }
          }
        },
        "zero": {
          "acc": 0.1940601145038168,
          "nll": 1.4808785562097775,
          "brier": 0.8010156004695758,
          "ece": 0.18273708237078995,
          "tie_count": 0,
          "clipped_count": 0,
          "by_sign": {
            "-omega": {
              "acc": 0.18795059372349449,
              "nll": 1.467815315386968
            },
            "+omega": {
              "acc": 0.2001696352841391,
              "nll": 1.493941797032587
            }
          }
        }
      },
      "paired_phase_minus_zero": {
        "acc_gain_t95": {
          "n_blocks": 131,
          "mean": 0.04392626166242579,
          "block_sd": 0.01643696863473215,
          "se": 0.001436104617537572,
          "t_crit_95": 1.9783804054470222,
          "lo95": 0.041085100426917466,
          "hi95": 0.04676742289793411
        },
        "acc_gain_boot95": {
          "n_blocks": 131,
          "resamples": 10000,
          "seed": 2026311,
          "mean": 0.04392626166242579,
          "lo95": 0.041116677268871936,
          "hi95": 0.046689627067430024
        },
        "nll_improv_t95": {
          "n_blocks": 131,
          "mean": 0.035650002550707,
{
  "train_bench": {
    "Q3_after_pilot_kv_mb8": {
      "alloc_after_mb1_4_mb": [
        8055.2,
        8055.2,
        8055.2,
        8055.2
      ],
      "peak_vram_mb": 30514.6,
      "steady_32ex_step_ms": 2139.7,
      "conn_grad_sum": 24.481,
      "host_grads": 0
    },
    "Q3_after_pilot_kv_mb16": {
      "alloc_after_mb1_4_mb": [
        8055.2,
        8055.2,
        8055.2,
        8055.2
      ],
      "peak_vram_mb": 52972.5,
      "steady_32ex_step_ms": 1975.7,
      "conn_grad_sum": 14.287,
      "host_grads": 0
    },
    "Q2_fresh_resample_kv_mb8": {
      "alloc_after_mb1_4_mb": [
        8055.2,
        8055.2,
        8055.2,
        8055.2
      ],
      "peak_vram_mb": 25910.3,
      "steady_32ex_step_ms": 1965.8,
      "conn_grad_sum": 13.822,
      "host_grads": 0
    },
    "Q2_fresh_resample_kv_mb16": {
      "alloc_after_mb1_4_mb": [
        8055.2,
        8055.2,
        8055.2,
        8055.2
      ],
      "peak_vram_mb": 43763.7,
      "steady_32ex_step_ms": 1656.0,
      "conn_grad_sum": 19.913,
      "host_grads": 0
    }
  },
  "dev_eval_bench": {
    "Q3_after_pilot_eval_b32": {
      "peak_vram_mb": 26728.4,
      "ms_per_ex": 24.46
    },
    "Q3_after_pilot_eval_b64": {
      "peak_vram_mb": 45409.9,
      "ms_per_ex": 23.88
    }
  },
  "parity_bench": {
    "b1_Q3_repeat_x1": {
      "batched_vs_naive_max_ll_diff": 0.32451415061950684,
      "batched_vs_naive_max_prob_diff": 0.027078211307525635,
      "kv_vs_naive_max_ll_diff": 0.3533594608306885,
      "kv_vs_naive_max_prob_diff": 0.01602950692176819
    },
    "b1_Q3_after_pilot": {
      "batched_vs_naive_max_ll_diff": 0.1306014060974121,
      "batched_vs_naive_max_prob_diff": 0.019718840718269348,
      "kv_vs_naive_max_ll_diff": 0.12212705612182617,
      "kv_vs_naive_max_prob_diff": 0.021901369094848633
    },
    "g1_direct_b64_vs_table_b64": {
      "max_abs_ll_diff": 0.0,
      "max_abs_prob_diff": 0.0,
      "argmax_agreement": 1.0
    },
    "g1_direct_b16_vs_table_b64": {
      "max_abs_ll_diff": 0.4816117286682129,
      "max_abs_prob_diff": 0.05001725029437104,
      "argmax_agreement": 1.0
    },
    "g1_direct_b1_vs_table_b64": {
      "max_abs_ll_diff": 0.48299169540405273,
      "max_abs_prob_diff": 0.034512381279745874,
      "argmax_agreement": 1.0
    }
  }
}
```

### Event #18: `item.completed` — `command_execution` (`item_14`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=Join-Path (Get-Location) '"'research/stage4_math_and_design/M6/.uv-cache'; uv python list --only-installed; Get-ChildItem 'C:/Users/李则徐/AppData/Local/uv/cache' -Name; Get-ChildItem 'C:/Users/李则徐/.gemini/antigravity/scratch' -Filter continuity.py -Recurse -Depth 2 | Select-Object -First 5 FullName; Get-Command pwsh -ErrorAction SilentlyContinue"
```

**Aggregated Output:**

```text
cpython-3.13.0-windows-x86_64-none    C:\Users\李则徐\AppData\Local\Programs\Python\Python313\python.exe
cpython-3.12.7-windows-x86_64-none    C:\qazwsx\python.exe
cpython-3.9.19-windows-x86_64-none    C:\376381\miniconda3\python.exe
cpython-3.9.19-windows-x86_64-none    C:\114514\python.exe
archive-v0
builds-v0
environments-v2
interpreter-v4
sdists-v9
simple-v21
wheels-v6
.gitignore
.lock
CACHEDIR.TAG

FullName                                                 
--------                                                 
C:\Users\������\.gemini\antigravity\scratch\continuity.py
Get-ChildItem : ��·����C:\Users\������\.gemini\antigravity\scratch\.pytest_cache���ķ��ʱ��ܾ���
����λ�� ��:2 �ַ�: 184
+ ... che' -Name; Get-ChildItem 'C:/Users/������/.gemini/antigravity/scratch'  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : PermissionDenied: (C:\Users\������\.g...h\.pytest_cache:String) [Get-ChildItem], Unauthor
iz    edAccessException
    + FullyQualifiedErrorId : DirUnauthorizedAccessError,Microsoft.PowerShell.Commands.GetChildItemCommand
 
Get-ChildItem : ��·����C:\Users\������\.gemini\antigravity\scratch\contra_complete_chronicles\.pytest_cache���ķ��ʱ���
����
����λ�� ��:2 �ַ�: 184
+ ... che' -Name; Get-ChildItem 'C:/Users/������/.gemini/antigravity/scratch'  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : PermissionDenied: (C:\Users\������\.g...s\.pytest_cache:String) [Get-ChildItem], Unauthor
iz    edAccessException
    + FullyQualifiedErrorId : DirUnauthorizedAccessError,Microsoft.PowerShell.Commands.GetChildItemCommand
 
Get-ChildItem : ��·����C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer\.pytest_cache���ķ��ʱ��ܾ���
����λ�� ��:2 �ַ�: 184
+ ... che' -Name; Get-ChildItem 'C:/Users/������/.gemini/antigravity/scratch'  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : PermissionDenied: (C:\Users\������\.g...r\.pytest_cache:String) [Get-ChildItem], Unauthor
iz    edAccessException
    + FullyQualifiedErrorId : DirUnauthorizedAccessError,Microsoft.PowerShell.Commands.GetChildItemCommand
 
Get-ChildItem : ��·����C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-cli1-cttc\.pytest_cache���ķ��ʱ��ܾ���
����λ�� ��:2 �ַ�: 184
+ ... che' -Name; Get-ChildItem 'C:/Users/������/.gemini/antigravity/scratch'  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : PermissionDenied: (C:\Users\������\.g...c\.pytest_cache:String) [Get-ChildItem], Unauthor
iz    edAccessException
    + FullyQualifiedErrorId : DirUnauthorizedAccessError,Microsoft.PowerShell.Commands.GetChildItemCommand
 
Get-ChildItem : ��·����C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-cli4-p12-stage5\.pytest_cache���ķ���
���ܾ���
����λ�� ��:2 �ַ�: 184
+ ... che' -Name; Get-ChildItem 'C:/Users/������/.gemini/antigravity/scratch'  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : PermissionDenied: (C:\Users\������\.g...5\.pytest_cache:String) [Get-ChildItem], Unauthor
iz    edAccessException
    + FullyQualifiedErrorId : DirUnauthorizedAccessError,Microsoft.PowerShell.Commands.GetChildItemCommand
 
Get-ChildItem : ��·����C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage3d\.pytest_cache���ķ��ʱ��ܾ���
����λ�� ��:2 �ַ�: 184
+ ... che' -Name; Get-ChildItem 'C:/Users/������/.gemini/antigravity/scratch'  ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : PermissionDenied: (C:\Users\������\.g...d\.pytest_cache:String) [Get-ChildItem], Unauthor
iz    edAccessException
    + FullyQualifiedErrorId : DirUnauthorizedAccessError,Microsoft.PowerShell.Commands.GetChildItemCommand
 


```

### Event #19: `item.completed` — `command_execution` (`item_17`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$a=Get-Content research/stage3d_qwen_signed_phase/recompute_all.py; $a[48..98]; $a[166..226]; $a[729..846]; $a[858..931]'
```

**Aggregated Output:**

```text
def softmax_temperature(raw_ll: np.ndarray, temp: float) -> np.ndarray:
    scaled = np.asarray(raw_ll, dtype=np.float64) / float(temp)
    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
    ex = np.exp(scaled)
    return ex / np.sum(ex, axis=-1, keepdims=True)


def score_vectors(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Compute fractional-tie hits, NLL (clipped at 1e-12), Brier, and 10-bin ECE."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n = len(y)
    if p.shape != (n, 4):
        raise ValueError(f"Expected {(n, 4)}, got {p.shape}")
    max_p = np.max(p, axis=1, keepdims=True)
    ties = p == max_p
    tie_counts = ties.sum(axis=1)
    hits = ties[np.arange(n), y].astype(np.float64) / tie_counts.astype(np.float64)
    n_ties = int((tie_counts > 1).sum())

    p_true_raw = p[np.arange(n), y]
    n_clipped = int((p_true_raw < 1e-12).sum())
    p_true = np.clip(p_true_raw, 1e-12, 1.0)
    nll = -np.log(p_true)

    one_hot = np.zeros_like(p)
    one_hot[np.arange(n), y] = 1.0
    brier = np.sum((p - one_hot) ** 2, axis=1)

    confs = np.max(p, axis=1)
    edges = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for i in range(10):
        mask = (confs >= edges[i]) & (confs <= edges[i + 1]) if i == 0 else (confs > edges[i]) & (confs <= edges[i + 1])
        cnt = int(mask.sum())
        if cnt > 0:
            ece += (cnt / float(n)) * abs(float(hits[mask].mean()) - float(confs[mask].mean()))

    return {
        "n": int(n),
        "acc": float(hits.mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "ece": float(ece),
        "tie_count": n_ties,
        "clipped_count": n_clipped,
        "per_item_hit": hits,
        "per_item_nll": nll,
    }


def evaluate_cell_paired(
    raw_ll_table: np.ndarray,
    temps: np.ndarray,
    codes: np.ndarray,
    targets: np.ndarray,
    direct_probs: np.ndarray,
    full_iq_probs: np.ndarray,
    seeds: Sequence[int],
    is_primary: bool = False,
) -> dict:
    """Evaluate paired phase vs zero arms over (signs x rotations x mothers) and 3 seeds."""
    # raw_ll_table: (2, 3, 2, 256, 4); codes/targets: (2, 3, M)
    n_arms, n_seeds, n_signs, _, _ = raw_ll_table.shape
    assert n_arms == 2 and n_signs == 2
    n_rot = codes.shape[1]
    n_mothers = codes.shape[2]
    assert n_mothers % BLOCK_SIZE == 0
    n_blocks = n_mothers // BLOCK_SIZE

    probs_table = np.zeros_like(raw_ll_table, dtype=np.float64)
    for a in range(n_arms):
        for s_idx in range(n_seeds):
            probs_table[a, s_idx] = softmax_temperature(raw_ll_table[a, s_idx], float(temps[a, s_idx]))

    # Shape (2, 3, 2, n_rot, M)
    hits = np.zeros((n_arms, n_seeds, n_signs, n_rot, n_mothers), dtype=np.float64)
    nlls = np.zeros((n_arms, n_seeds, n_signs, n_rot, n_mothers), dtype=np.float64)
    briers = np.zeros((n_arms, n_seeds, n_signs, n_rot, n_mothers), dtype=np.float64)
    tie_counts = np.zeros((n_arms, n_seeds), dtype=int)
    clip_counts = np.zeros((n_arms, n_seeds), dtype=int)
    all_probs_flat = [[None] * n_seeds for _ in range(n_arms)]
    all_lbl_flat = targets.reshape(-1)

    for a in range(n_arms):
        for s_idx in range(n_seeds):
            p_list = []
            for sign_i in range(n_signs):
                p_sr = probs_table[a, s_idx, sign_i][codes[sign_i]]  # (n_rot, M, 4)
                p_list.append(p_sr)
                sc = score_vectors(p_sr.reshape(-1, 4), targets[sign_i].reshape(-1))
                hits[a, s_idx, sign_i] = sc["per_item_hit"].reshape(n_rot, n_mothers)
                nlls[a, s_idx, sign_i] = sc["per_item_nll"].reshape(n_rot, n_mothers)
                oh = np.zeros_like(p_sr)
                np.put_along_axis(oh, targets[sign_i][..., None], 1.0, axis=-1)
                briers[a, s_idx, sign_i] = np.sum((p_sr - oh) ** 2, axis=-1)
                tie_counts[a, s_idx] += sc["tie_count"]
                clip_counts[a, s_idx] += sc["clipped_count"]
            all_probs_flat[a][s_idx] = np.stack(p_list, axis=0).reshape(-1, 4)

    # Pooled over signs, rotations, and 3 seeds equally
    # Block aggregation: first mother-level average across (seeds, signs, rotations), then 64-mother block means
    mother_hit = hits.mean(axis=(1, 2, 3))  # (2, M)
    mother_nll = nlls.mean(axis=(1, 2, 3))  # (2, M)
    block_hit = mother_hit.reshape(2, n_blocks, BLOCK_SIZE).mean(axis=2)  # (2, B)
    block_nll = mother_nll.reshape(2, n_blocks, BLOCK_SIZE).mean(axis=2)  # (2, B)

    acc_gain_blocks = block_hit[0] - block_hit[1]        # phase - zero (>0 is better)
    nll_improv_blocks = block_nll[1] - block_nll[0]      # zero - phase (>0 is better)
    acc_t_ci = paired_t_ci(acc_gain_blocks)
    nll_t_ci = paired_t_ci(nll_improv_blocks)
    acc_boot_ci = block_bootstrap_ci(acc_gain_blocks)
def work_a_q2_analysis(q2_npz, temps_q2: np.ndarray, seeds: Sequence[int]) -> dict:
    """Evaluate all 7 Q2 arms on 512 fresh C4 (omega=+0.35, 2dB, tp=4) mothers vs Work A kernel."""
    raw_ll = q2_npz["q2__raw_ll"]             # (2, 3, 7, M, 4)
    labels = q2_npz["q2__labels"]             # (M,)
    z1 = q2_npz["q2__z1"]
    z2 = q2_npz["q2__z2"]
    work_a_probs = q2_npz["q2__work_a_ref_probs"]  # (6, M, 4)
    n_arms, n_seeds, n_q2_arms, n_mothers, _ = raw_ll.shape
    assert n_q2_arms == len(Q2_ARMS)

    probs_all = np.zeros_like(raw_ll, dtype=np.float64)
    for a in range(n_arms):
        for s_idx in range(n_seeds):
            probs_all[a, s_idx] = softmax_temperature(raw_ll[a, s_idx], float(temps_q2[a, s_idx]))

    # Physical Work A kernel baselines
    work_a_metrics = {}
    work_a_per_arm = {}
    for arm_name, ref_idx in Q2_LOGICAL_REF_IDX.items():
        sc = score_vectors(work_a_probs[ref_idx], labels)
        work_a_per_arm[arm_name] = sc
        work_a_metrics[arm_name] = {k: sc[k] for k in ("n", "acc", "nll", "brier", "ece")}
    work_a_fresh_diff_acc = work_a_per_arm["fresh_resample"]["per_item_hit"] - work_a_per_arm["valid"]["per_item_hit"]
    work_a_fresh_diff_nll = work_a_per_arm["fresh_resample"]["per_item_nll"] - work_a_per_arm["valid"]["per_item_nll"]
    work_a_metrics["fresh_minus_valid"] = {
        "delta_acc": float(work_a_fresh_diff_acc.mean()),
        "delta_acc_se": float(work_a_fresh_diff_acc.std(ddof=1) / math.sqrt(n_mothers)),
        "delta_nll": float(work_a_fresh_diff_nll.mean()),
        "delta_nll_se": float(work_a_fresh_diff_nll.std(ddof=1) / math.sqrt(n_mothers)),
    }

    arms_out = {}
    for a, aname in enumerate(("phase", "zero")):
        by_q2_arm = {}
        per_seed_q2 = []
        # Pool 3 seeds cleanly
        for qi, qarm in enumerate(Q2_ARMS):
            flat_p = probs_all[a, :, qi].reshape(-1, 4)
            flat_y = np.tile(labels, n_seeds)
            sc = score_vectors(flat_p, flat_y)
            by_q2_arm[qarm] = {k: sc[k] for k in ("n", "acc", "nll", "brier", "ece", "tie_count")}

        # Paired comparisons across (seed x mother)
        def _pair_delta(arm_b: str, arm_a: str) -> dict:
            ia = Q2_ARMS.index(arm_a)
            ib = Q2_ARMS.index(arm_b)
            pa = probs_all[a, :, ia].reshape(-1, 4)
            pb = probs_all[a, :, ib].reshape(-1, 4)
            ya = np.tile(labels, n_seeds)
            sca = score_vectors(pa, ya)
            scb = score_vectors(pb, ya)
            tv = 0.5 * np.abs(pb - pa).sum(axis=-1)
            max_abs = np.max(np.abs(pb - pa), axis=-1)
            conf_inc = pb.max(axis=-1) - pa.max(axis=-1)
            hit_delta = scb["per_item_hit"] - sca["per_item_hit"]
            nll_delta = scb["per_item_nll"] - sca["per_item_nll"]
            # Mother-averaged SE across mothers (clustering seeds per mother!)
            hit_m = hit_delta.reshape(n_seeds, n_mothers).mean(axis=0)
            nll_m = nll_delta.reshape(n_seeds, n_mothers).mean(axis=0)
            return {
                "delta_acc": float(hit_m.mean()),
                "delta_acc_mother_se": float(hit_m.std(ddof=1) / math.sqrt(n_mothers)),
                "delta_nll": float(nll_m.mean()),
                "delta_nll_mother_se": float(nll_m.std(ddof=1) / math.sqrt(n_mothers)),
                "mean_tv": float(tv.mean()),
                "max_tv": float(tv.max()),
                "max_abs_prob_diff": float(max_abs.max()),
                "max_conf_increase": float(conf_inc.max()),
                "corrected": int(np.sum(hit_delta > 0)),
                "new_errors": int(np.sum(hit_delta < 0)),
            }

        for s_idx, seed_val in enumerate(seeds):
            s_dict = {"seed": int(seed_val), "arms": {}}
            for qi, qarm in enumerate(Q2_ARMS):
                sc_s = score_vectors(probs_all[a, s_idx, qi], labels)
                s_dict["arms"][qarm] = {k: sc_s[k] for k in ("acc", "nll", "brier", "ece")}
            per_seed_q2.append(s_dict)

        arms_out[aname] = {
            "by_arm": by_q2_arm,
            "paired_contrasts": {
                "valid_minus_none": _pair_delta("valid", "none"),
                "independent_only_minus_none": _pair_delta("independent_only", "none"),
                "duplicate_copy_dedup_minus_valid": _pair_delta("duplicate_copy_dedup", "valid"),
                "duplicate_copy_raw_minus_valid": _pair_delta("duplicate_copy_raw", "valid"),
                "valid_plus_independent_minus_valid": _pair_delta("valid_plus_independent", "valid"),
                "fresh_resample_minus_valid": _pair_delta("fresh_resample", "valid"),
            },
            "per_seed": per_seed_q2,
        }

    # Concrete multi-arm trajectory example from seed 0 where Work A valid->fresh helps and host phase arm corrects
    p_s0 = probs_all[0, 0]  # (7, M, 4)
    idx_ex = 0
    for m_i in range(n_mothers):
        pred_none = int(np.argmax(p_s0[Q2_ARMS.index("none"), m_i]))
        pred_val = int(np.argmax(p_s0[Q2_ARMS.index("valid"), m_i]))
        pred_fresh = int(np.argmax(p_s0[Q2_ARMS.index("fresh_resample"), m_i]))
        truth = int(labels[m_i])
        if pred_none != truth and pred_val == truth and pred_fresh == truth:
            idx_ex = m_i
            break

    ex_traj = {
        "mother_idx": int(idx_ex),
        "z1": int(z1[idx_ex]),
        "z2": int(z2[idx_ex]),
        "true_x1": int(labels[idx_ex]),
        "phase_seed0_probs_by_arm": {
            qarm: [float(v) for v in probs_all[0, 0, qi, idx_ex]] for qi, qarm in enumerate(Q2_ARMS)
        },
        "zero_seed0_probs_by_arm": {
            qarm: [float(v) for v in probs_all[1, 0, qi, idx_ex]] for qi, qarm in enumerate(Q2_ARMS)
        },
        "work_a_ref_probs": {
            qarm: [float(v) for v in work_a_probs[ridx, idx_ex]] for qarm, ridx in Q2_LOGICAL_REF_IDX.items()
        },
def recompute_all(data_dir: Path, verify_against: Optional[Path] = None) -> dict:
    """Recompute complete summary_master.json solely from exported .npz raw scores in data_dir."""
    q3_npz = np.load(data_dir / "blind_q3_scores.npz", allow_pickle=False)
    q2_npz = np.load(data_dir / "blind_q2_scores.npz", allow_pickle=False)

    temps_q3 = np.asarray(q3_npz["temps_q3"], dtype=np.float64)
    temps_q2 = np.asarray(q2_npz["temps_q2"], dtype=np.float64)
    seeds = [int(x) for x in q3_npz["seeds"].tolist()]

    cells_res = {}
    for cid in CELL_ORDER:
        cells_res[cid] = {
            **CELL_META[cid],
            **evaluate_cell_paired(
                raw_ll_table=q3_npz[f"{cid}__raw_ll"],
                temps=temps_q3,
                codes=q3_npz[f"{cid}__codes"],
                targets=q3_npz[f"{cid}__targets"],
                direct_probs=q3_npz[f"{cid}__direct_table_probs"],
                full_iq_probs=q3_npz[f"{cid}__full_iq_probs"],
                seeds=seeds,
                is_primary=(cid == "G1_main_extrap_both"),
            ),
        }

    # Work F on primary +-0.40 and historical +-0.35
    work_f_primary = work_f_sign_probe(
        raw_ll_table=q3_npz["G1_main_extrap_both__raw_ll"],
        temps=temps_q3,
        codes_rot0=q3_npz["G1_main_extrap_both__codes"][:, 0, :],
        targets_rot0=q3_npz["G1_main_extrap_both__targets"][:, 0, :],
        cal_ref=q3_npz["G1_main_extrap_both__cal_ref"],
        cal_mass=q3_npz["G1_main_extrap_both__cal_mass"],
        seeds=seeds,
        contra_text_flipped=q3_npz["G1_main_extrap_both__contra_text_flipped_raw_ll"] if "G1_main_extrap_both__contra_text_flipped_raw_ll" in q3_npz else None,
        contra_phase_flipped=q3_npz["G1_main_extrap_both__contra_phase_flipped_raw_ll"] if "G1_main_extrap_both__contra_phase_flipped_raw_ll" in q3_npz else None,
    )
    work_f_hist = work_f_sign_probe(
        raw_ll_table=q3_npz["H_c4_035__raw_ll"],
        temps=temps_q3,
        codes_rot0=q3_npz["H_c4_035__codes"][:, 0, :],
        targets_rot0=q3_npz["H_c4_035__targets"][:, 0, :],
        cal_ref=q3_npz["H_c4_035__cal_ref"],
        cal_mass=q3_npz["H_c4_035__cal_mass"],
        seeds=seeds,
    )

    # Work G on primary G1
    work_g = work_g_rotation_and_controls(
        raw_ll_table=q3_npz["G1_main_extrap_both__raw_ll"],
        temps=temps_q3,
        codes=q3_npz["G1_main_extrap_both__codes"],
        targets=q3_npz["G1_main_extrap_both__targets"],
        rot10_codes=q3_npz["G1_main_extrap_both__rot10_codes"],
        rot10_targets=q3_npz["G1_main_extrap_both__rot10_targets"],
        indep_raw_ll=q3_npz["G1_main_extrap_both__indep_raw_ll"],
        indep_targets=q3_npz["G1_main_extrap_both__indep_targets"],
        shuffled_perm=q3_npz["G1_main_extrap_both__shuffled_perm"],
        seeds=seeds,
    )

    # Work A Q2
    work_a_q2 = work_a_q2_analysis(q2_npz, temps_q2, seeds)

    # Replay C1/C2/C3 forgetting check (if present)
    replay_c123 = {}
    replay_path = data_dir / "replay_c123_scores.npz"
    if replay_path.exists():
        rp = np.load(replay_path, allow_pickle=False)
        for cname in ("C1_coherent_2db", "C2_discrete_walk", "C3_mid_pilot_cont"):
            if f"{cname}__q3_raw_ll" not in rp:
                continue
            replay_c123[cname] = {}
            for qk in ("q1", "q3", "q2v"):
```

### Event #20: `item.completed` — `web_search` (`exec-8cc0e006-cae0-4a01-8728-7fc98e82c35b`)

```json
{
  "id": "exec-8cc0e006-cae0-4a01-8728-7fc98e82c35b",
  "type": "web_search",
  "query": "https://modal.com/docs/guide/budgets",
  "action": {
    "type": "open_page",
    "url": "https://modal.com/docs/guide/budgets"
  },
  "results": [
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn2view0",
      "snippet": "Total lines: 51",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/wDNdIcI6YgpGZEMlEZP6QHr7p5Bl1e0FDJwI7JuhUEHzv3kkxU2eIlnrdUnLdOFic0nD-cKaALuvdteI8Zb4BzIhzN0arH9MMi32ujdBeECOV26ky9dzrMq5aM5uavjq",
      "title": "Budgets | Modal Docs",
      "url": "https://modal.com/docs/guide/budgets"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn2view1",
      "snippet": "Total lines: 84",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/I2UOZcwMgTE8Q1feTvSmkYifC0AsP4y5KpYxIyldfLYjGB38IpVf7pVSzBz9xwG-YjQbUxAurNRpPTF7ouPpExaBb17siP-cZfQ-uIVVD8d2o_3DwYWfUuxGHtJhj0KB93l8jlRAYKhF1ZP7pF8C6ypLovY-Ss7z4n8Qqnkv_E8",
      "title": "Configuring CPU, memory, and disk | Modal Docs",
      "url": "https://modal.com/docs/guide/resources"
    },
    {
      "type": "text_result",
      "domain": "arxiv.org",
      "ref_id": "turn2view2",
      "snippet": "Total lines: 551",
      "title": "LEMMO: A Large Electromagnetic Model leveraging Pretrained LLMs for Million-Sample Signal Understanding",
      "url": "https://arxiv.org/html/2611.09476v1"
    },
    {
      "type": "text_result",
      "domain": "arxiv.org",
      "ref_id": "turn2view3",
      "snippet": "Total lines: 551",
      "title": "LEMMO: A Large Electromagnetic Model leveraging Pretrained LLMs for Million-Sample Signal Understanding",
      "url": "https://arxiv.org/html/2611.09476v1"
    },
    {
      "type": "text_result",
      "domain": "arxiv.org",
      "ref_id": "turn2view4",
      "snippet": "Total lines: 551",
      "title": "LEMMO: A Large Electromagnetic Model leveraging Pretrained LLMs for Million-Sample Signal Understanding",
      "url": "https://arxiv.org/html/2611.09476v1"
    }
  ]
}
```

### Event #21: `item.completed` — `command_execution` (`item_18`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-ChildItem 'C:/Users/李则徐/AppData/Local/uv/cache/environments-v2' -Force; Get-ChildItem 'C:/Users/李则徐/AppData/Local/uv/cache/wheels-v6' -Name; Test-Path C:/qazwsx/Lib/site-packages/numpy; Get-ChildItem research/stage4_math_and_design/M6 -Force; rg -n 'def .*restore|def .*checkpoint|mkdir|DB_PATH|sqlite3.connect|write_text' ../continuity.py"
```

**Aggregated Output:**

```text


    Ŀ¼: C:\Users\������\AppData\Local\uv\cache\environments-v2


Mode                 LastWriteTime         Length Name                                                                 
----                 -------------         ------ ----                                                                 
d-----        2026/11/26     19:11                1093dab72d22fcbe                                                     
pypi
True


    Ŀ¼: C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M6


Mode                 LastWriteTime         Length Name                                                                 
----                 -------------         ------ ----                                                                 
d-----        2026/11/26     19:20                .uv-cache                                                            
180:    def checkpoint(self, expected, capsule, note):
220:    LOCAL.mkdir(parents=True, exist_ok=True)
238:    LOCAL.mkdir(parents=True, exist_ok=True)


```

### Event #22: `item.completed` — `command_execution` (`item_19`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '"'^(class |def )|signed_phase|phase_|extra_refs|slot|cos'"\\(|sin\\(' research/stage3d_qwen_signed_phase/qwen_connector.py | Select-Object -First 120; rg -n 'direct_table|class |def ' research/stage3d_qwen_signed_phase/data_stage3d.py; git status --short --untracked-files=all | Select-Object -First 50"
```

**Aggregated Output:**

```text
33:class SignedPhaseCacheConnector(nn.Module):
36:    Takes discrete integer cache indices (z1, z2), 6 public phase slots, and revealed
48:        num_phase_tokens: int = 4,
59:        self.num_phase_tokens = int(num_phase_tokens)
80:        self.phase_mlp = nn.Sequential(
85:            nn.Linear(mlp_hidden, num_phase_tokens * mlp_hidden),
98:        self.phase_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
112:    def encode_phase(self, phase_slots: torch.Tensor) -> torch.Tensor:
113:        up = self.phase_mlp(phase_slots).view(-1, self.num_phase_tokens, self.mlp_hidden)
114:        return self.out_norm(self.out_proj(up)) * self.phase_scale
132:            "num_phase_tokens": self.num_phase_tokens,
144:def _get_oop_linear_attention_layer_cls():
192:def _make_differentiable_qwen35_cache(lm_config):
205:def _clone_expand_qwen35_cache(pref_cache, lm_config, repeats: int):
233:class QwenCacheHost:
438:        phase_slots_batch: Optional[torch.Tensor] = None,
462:            if phase_slots_batch is None:
465:                ps = phase_slots_batch.to(self.device, dtype=torch.float32)
69:class BlindCellSpec:
82:    def total_delay(self) -> int:
86:    def cum_angle_abs(self) -> float:
90:    def cum_diffusion_var(self) -> float:
146:def load_s17_encoder(repo_root: Optional[Path] = None):
161:def public_phase_slots(omega: float, h: int, ell: int, t_p: float, zero_arm: bool = False) -> np.ndarray:
169:def condition_spec_from_params(
193:def format_public_metadata(cond: ConditionSpec) -> str:
209:def build_stage3d_prompt(
262:def derive_block_seeds(namespace: int, cell_idx: int, blocks: int) -> List[int]:
270:def generate_paired_cfo_blocks(
340:        def obs(u_labels: np.ndarray, phi: np.ndarray, n_arr: np.ndarray) -> np.ndarray:
375:def build_q2_arm_references(batch: MotherBatch, arm: str, coord_rot: int = 0) -> Tuple[Tuple[Reference, ...], bool]:
405:def build_blind_seed_manifest(g1_blocks: int = 131) -> dict:
445:def verify_stage3d_orbit_isolation(max_blocks: int = 2) -> Dict[str, object]:
449:    def _add_batches(name: str, batches: Iterable[MotherBatch]):
?? research/stage4_math_and_design/M2/verify_m2.py
?? research/stage4_math_and_design/discussions/M1/prompt_r1.md
?? research/stage4_math_and_design/discussions/M1/round1_events.jsonl
?? research/stage4_math_and_design/discussions/M2/prompt_r1.md
?? research/stage4_math_and_design/discussions/M2/round1_events.jsonl
?? research/stage4_math_and_design/discussions/M3/prompt_r1.md
?? research/stage4_math_and_design/discussions/M3/round1_events.jsonl
?? research/stage4_math_and_design/discussions/M4/prompt_r1.md
?? research/stage4_math_and_design/discussions/M4/round1_events.jsonl
?? research/stage4_math_and_design/discussions/M5/prompt_r1.md
?? research/stage4_math_and_design/discussions/M5/round1_events.jsonl
?? research/stage4_math_and_design/discussions/M6/prompt_r1.md
?? research/stage4_math_and_design/discussions/M6/round1_events.jsonl
?? research/stage4_math_and_design/transcript_utils.py
```

### Event #23: `item.completed` — `command_execution` (`item_20`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=Join-Path (Get-Location) '"'research/stage4_math_and_design/M6/.uv-cache'; "'$env:UV_OFFLINE='"'1'; "'$source=Join-Path $env:LOCALAPPDATA '"'uv/cache'; Get-ChildItem (Join-Path "'$source '"'wheels-v6/pypi') -Name; Get-ChildItem (Join-Path "'$source '"'simple-v21') -Name; Get-ChildItem (Join-Path "'$source '"'environments-v2/1093dab72d22fcbe/Lib/site-packages') -Name | Select-Object -First 30; Get-ChildItem 'C:/qazwsx/Lib/site-packages' -Name | Where-Object {"'$_ -match '"'"'^(numpy|scipy).+dist-info'"'}; "'$a=Get-Content ../continuity.py; $a[0..55]; $a[208..253]'
```

**Aggregated Output:**

```text
aiohappyeyeballs
aiohttp
aiosignal
alembic
annotated-doc
annotated-types
anyio
async-timeout
attrs
azure-ai-projects
azure-core
azure-identity
azure-storage-blob
backports-asyncio-runner
bcrypt
certifi
cffi
charset-normalizer
click
colorama
comtypes
coverage
cryptography
dnspython
email-validator
exceptiongroup
fastapi
filelock
frozenlist
fsspec
greenlet
h11
hf-xet
httpcore
httpcore2
httptools
httpx
httpx2
huggingface-hub
idna
iniconfig
isodate
jinja2
jiter
jsonschema
jsonschema-specifications
mako
markdown-it-py
markupsafe
mdurl
mpmath
msal
msal-extensions
multidict
networkx
numpy
openai
packaging
pluggy
propcache
psutil
psycopg
psycopg-binary
pycparser
pycryptodome
pydantic
pydantic-core
pydantic-settings
pygments
pyjwt
pytest
pytest-asyncio
pytest-cov
pytest-timeout
python-dotenv
pyyaml
referencing
regex
requests
rich
rpds-py
ruff
safetensors
scipy
setuptools
shellingham
sniffio
sqlalchemy
starlette
sympy
tokenizers
tomli
torch
tqdm
transformers
truststore
typer
typing-extensions
typing-inspection
tzdata
uiautomation
urllib3
uvicorn
uvloop
watchfiles
websockets
yarl
pypi
Get-ChildItem : �Ҳ���·����C:\Users\������\AppData\Local\uv\cache\environments-v2\1093dab72d22fcbe\Lib\site-packages����
��Ϊ��·�������ڡ�
����λ�� ��:2 �ַ�: 273
+ ... 21') -Name; Get-ChildItem (Join-Path $source 'environments-v2/1093dab ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : ObjectNotFound: (C:\Users\������\Ap...b\site-packages:String) [Get-ChildItem], ItemNotFou
nd    Exception
    + FullyQualifiedErrorId : PathNotFound,Microsoft.PowerShell.Commands.GetChildItemCommand
 
numpy-1.26.4.dist-info
numpydoc-1.7.0.dist-info
scipy-1.13.1.dist-info
'''Small, workspace-scoped continuity checkpoints. No model calls or full audits.'''

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
from uuid import NAMESPACE_URL, UUID, uuid5

ROOT = Path(__file__).resolve().parent
LOCAL = ROOT / '.continuity'
sys.path.insert(0, str(LOCAL / 'vendor'))
os.environ.setdefault('TIKTOKEN_CACHE_DIR', str(LOCAL / 'tokenizer-cache'))

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
import tiktoken

IDENTITY = UUID('bfb03f16-3a2f-4848-9c76-fd5e3d8a957d')
DEVICE = UUID('656fd514-3663-48d1-9b0d-6420fca11b76')
OWNER = UUID('184412db-e889-41c9-aa94-3aecd9eb1e6d')
SCOPE = str(uuid5(IDENTITY, str(ROOT).casefold()))
CATEGORY = 'continuity_checkpoint'
BOOT_LIMIT = 200
RECALL_LIMIT = 500
ENCODING = 'o200k_base'
ENCODER = None
CREDENTIAL = LOCAL / 'connection.dpapi'
METRICS = LOCAL / 'metrics.jsonl'


class ContinuityError(Exception):
    pass


def require(condition, message):
    if not condition:
        raise ContinuityError(message)


def tokens(text):
    global ENCODER
    if ENCODER is None:
        ENCODER = tiktoken.get_encoding(ENCODING)
    return len(ENCODER.encode(text, disallowed_special=()))


def guidance():
    path = ROOT / 'AGENTS.md'
    return path.read_text(encoding='utf-8') if path.exists() else ''


def configure():
    dsn = os.environ.get('CONTINUITY_DATABASE_URL')
    require(bool(dsn), 'Set CONTINUITY_DATABASE_URL for one-time credential setup.')
    with connect(dsn=dsn) as conn:
        validate_actor(conn)
    if CREDENTIAL.exists():
        require(dpapi(CREDENTIAL.read_bytes(), True).decode('utf-8') == dsn,
                'CONFIG_CONFLICT: existing protected connection was not overwritten.')
        return 'Connection already configured.\n'
    protected = dpapi(dsn.encode('utf-8'))
    LOCAL.mkdir(parents=True, exist_ok=True)
    with CREDENTIAL.open('xb') as target:
        target.write(protected)
    return 'Connection configured with Windows user-bound protection.\n'


def measure(command, text, status=0):
    entry = {
        'at_utc': datetime.now(timezone.utc).isoformat(), 'command': command,
        'status': status, 'tokenizer': ENCODING, 'tiktoken_version': tiktoken.__version__,
        'output_tokens': tokens(text), 'workspace_instruction_tokens': tokens(guidance()),
        'output_utf8_bytes': len(text.encode('utf-8')),
        'output_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
        'scope': SCOPE, 'billing_tokens_measured': False,
    }
    if command == 'restore':
        entry['bootstrap_tokens'] = entry['output_tokens'] + entry['workspace_instruction_tokens']
        entry['bootstrap_limit'] = BOOT_LIMIT
    LOCAL.mkdir(parents=True, exist_ok=True)
    with METRICS.open('a', encoding='utf-8') as target:
        target.write(json.dumps(entry, ensure_ascii=False) + '\n')
    return entry


def stats():
    last_restore = last_recall = None
    if METRICS.exists():
        with METRICS.open(encoding='utf-8') as source:
            for line in source:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get('status') == 0 and event.get('scope') == SCOPE:
                    if event['command'] == 'restore':
```

### Event #24: `item.completed` — `command_execution` (`item_21`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=Join-Path (Get-Location) '"'research/stage4_math_and_design/M6/.uv-cache'; "'$env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$source=Join-Path $env:LOCALAPPDATA '"'uv/cache'; Get-ChildItem (Join-Path "'$source '"'wheels-v6/pypi/numpy') -Recurse -Name | Select-Object -First 24; Get-ChildItem (Join-Path "'$source '"'wheels-v6/pypi/scipy') -Recurse -Name | Select-Object -First 24; Get-ChildItem (Join-Path "'$source '"'archive-v0') -Directory | Where-Object { (Test-Path (Join-Path "'$_.FullName '"'numpy-2.3.5.dist-info')) -or (Test-Path (Join-Path "'$_.FullName '"'scipy-1.17.0.dist-info')) } | Select-Object Name; Get-ChildItem (Join-Path "'$source '"'environments-v2/1093dab72d22fcbe') -Force"
```

**Aggregated Output:**

```text
2.3.5-cp312-cp312-win_amd64
2.3.5-cp312-cp312-win_amd64.http
2.3.5-cp312-cp312-win_amd64.msgpack
2.5.5-cp312-cp312-win_amd64
2.5.5-cp312-cp312-win_amd64.http
2.5.5-cp312-cp312-win_amd64.msgpack
numpy-2.3.5-cp312-cp312-win_amd64.lock
numpy-2.5.5-cp312-cp312-win_amd64.lock
1.17.0-cp312-cp312-win_amd64
1.17.0-cp312-cp312-win_amd64.http
1.17.0-cp312-cp312-win_amd64.msgpack
scipy-1.17.0-cp312-cp312-win_amd64.lock

Name            
----            
3Oxde2-Jkv6KGjnQ
cSnmXl2iWBpNAXGE
27fef6b82521218f
44669b92cf8db157
76580b3e21535888
7829de497d478757
7d6c70a4e0ec676a
8ab8c010e9f0c738
a4a0d4059abfef7e
fff2ded241c488c1


```

### Event #25: `item.completed` — `command_execution` (`item_22`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=Join-Path (Get-Location) '"'research/stage4_math_and_design/M6/.uv-cache'; "'$env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$source=Join-Path $env:LOCALAPPDATA '"'uv/cache'; "'$dirs=@('"'wheels-v6/pypi','archive-v0','simple-v21/pypi'); foreach("'$d in $dirs){New-Item -Force -ItemType Directory (Join-Path $env:UV_CACHE_DIR $d) | Out-Null}; foreach($pkg in @('"'numpy','scipy')){Copy-Item -LiteralPath (Join-Path "'$source ('"'wheels-v6/pypi/'+"'$pkg)) -Destination (Join-Path $env:UV_CACHE_DIR '"'wheels-v6/pypi') -Recurse -Force; Get-ChildItem (Join-Path "'$source '"'simple-v21/pypi') -Filter ("'$pkg+'"'*') | Copy-Item -Destination (Join-Path "'$env:UV_CACHE_DIR '"'simple-v21/pypi') -Force}; foreach("'$d in @('"'3Oxde2-Jkv6KGjnQ','cSnmXl2iWBpNAXGE')){Copy-Item -LiteralPath (Join-Path "'$source ('"'archive-v0/'+"'$d)) -Destination (Join-Path $env:UV_CACHE_DIR '"'archive-v0') -Recurse -Force}; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B -c \"import numpy,scipy; print(numpy.__version__,scipy.__version__)\""
```

**Aggregated Output:**

```text
error: Failed to install: numpy-2.3.5-cp312-cp312-win_amd64.whl (numpy==2.3.5)
  Caused by: 拒绝访问。 (os error 5) at path "C:\\WINDOWS\\TEMP\\.tmpPqqiC2"
```

### Event #26: `item.completed` — `command_execution` (`item_24`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$a=Get-Content research/stage3d_qwen_signed_phase/recompute_all.py; $a[20..48]; $a[228..324]; $a[847..855]; $a[935..961]; $a=Get-Content research/stage3d_qwen_signed_phase/data_stage3d.py; $a[93..146]; $a[159..168]; $a[374..404]; $a=Get-Content research/stage3d_qwen_signed_phase/analysis_stage3d.py; $a[98..180]'
```

**Aggregated Output:**

```text
    "G3_extrap_angle",
    "G4_extrap_freq",
)
CELL_META = {
    "G1_main_extrap_both": {"abs_omega": 0.40, "h": 2, "ell": 2, "t_p": 4.0, "cum_angle_abs": 1.60, "sigma2_t": 0.04},
    "G2_interp_both": {"abs_omega": 0.20, "h": 2, "ell": 2, "t_p": 4.0, "cum_angle_abs": 0.80, "sigma2_t": 0.04},
    "G3_extrap_angle": {"abs_omega": 0.20, "h": 4, "ell": 4, "t_p": 8.0, "cum_angle_abs": 1.60, "sigma2_t": 0.08},
    "G4_extrap_freq": {"abs_omega": 0.40, "h": 1, "ell": 1, "t_p": 2.0, "cum_angle_abs": 0.80, "sigma2_t": 0.02},
}
Q2_ARMS = (
    "none",
    "valid",
    "duplicate_copy_dedup",
    "duplicate_copy_raw",
    "independent_only",
    "valid_plus_independent",
    "fresh_resample",
)
Q2_LOGICAL_REF_IDX = {
    "none": 0,
    "valid": 1,
    "duplicate_copy_dedup": 2,
    "independent_only": 3,
    "valid_plus_independent": 4,
    "fresh_resample": 5,
}


def softmax_temperature(raw_ll: np.ndarray, temp: float) -> np.ndarray:

    # Pooled ECE across all (seed x sign x rot x mother) predictions per arm
    pooled_arm_stats = {}
    for a, aname in enumerate(("phase", "zero")):
        flat_p = np.concatenate(all_probs_flat[a], axis=0)
        flat_y = np.tile(all_lbl_flat, n_seeds)
        sc_pool = score_vectors(flat_p, flat_y)
        pooled_arm_stats[aname] = {
            "acc": float(hits[a].mean()),
            "nll": float(nlls[a].mean()),
            "brier": float(briers[a].mean()),
            "ece": float(sc_pool["ece"]),
            "tie_count": int(tie_counts[a].sum()),
            "clipped_count": int(clip_counts[a].sum()),
            "by_sign": {
                "-omega": {"acc": float(hits[a, :, 0].mean()), "nll": float(nlls[a, :, 0].mean())},
                "+omega": {"acc": float(hits[a, :, 1].mean()), "nll": float(nlls[a, :, 1].mean())},
            },
        }

    # Per-seed breakdown
    per_seed = []
    for s_idx, seed_val in enumerate(seeds):
        b_hit_s = hits[:, s_idx].mean(axis=(1, 2)).reshape(2, n_blocks, BLOCK_SIZE).mean(axis=2)
        b_nll_s = nlls[:, s_idx].mean(axis=(1, 2)).reshape(2, n_blocks, BLOCK_SIZE).mean(axis=2)
        acc_g_s = b_hit_s[0] - b_hit_s[1]
        nll_i_s = b_nll_s[1] - b_nll_s[0]
        sc_p = score_vectors(all_probs_flat[0][s_idx], all_lbl_flat)
        sc_z = score_vectors(all_probs_flat[1][s_idx], all_lbl_flat)
        per_seed.append(
            {
                "seed": int(seed_val),
                "phase": {"acc": sc_p["acc"], "nll": sc_p["nll"], "brier": sc_p["brier"], "ece": sc_p["ece"]},
                "zero": {"acc": sc_z["acc"], "nll": sc_z["nll"], "brier": sc_z["brier"], "ece": sc_z["ece"]},
                "acc_gain_phase_minus_zero": paired_t_ci(acc_g_s),
                "nll_improv_zero_minus_phase": paired_t_ci(nll_i_s),
            }
        )

    seed_acc_gains = [x["acc_gain_phase_minus_zero"]["mean"] for x in per_seed]
    seed_nll_improv = [x["nll_improv_zero_minus_phase"]["mean"] for x in per_seed]

    # Analytic baselines on exact same mothers
    sc_dir = score_vectors(direct_probs.reshape(-1, 4), all_lbl_flat)
    sc_iq = score_vectors(full_iq_probs.reshape(-1, 4), all_lbl_flat)

    # Host minus Direct Readout paired block stats
    dir_hit = sc_dir["per_item_hit"].reshape(n_signs, n_rot, n_mothers).mean(axis=(0, 1))
    dir_nll = sc_dir["per_item_nll"].reshape(n_signs, n_rot, n_mothers).mean(axis=(0, 1))
    dir_blk_hit = dir_hit.reshape(n_blocks, BLOCK_SIZE).mean(axis=1)
    dir_blk_nll = dir_nll.reshape(n_blocks, BLOCK_SIZE).mean(axis=1)

    res = {
        "blocks": int(n_blocks),
        "mothers": int(n_mothers),
        "logical_rows_per_arm": int(n_seeds * n_signs * n_rot * n_mothers),
        "arms": pooled_arm_stats,
        "paired_phase_minus_zero": {
            "acc_gain_t95": acc_t_ci,
            "acc_gain_boot95": acc_boot_ci,
            "nll_improv_t95": nll_t_ci,
            "nll_improv_boot95": nll_boot_ci,
        },
        "per_seed": per_seed,
        "seed_dispersion": {
            "acc_gain_sd": float(np.std(seed_acc_gains, ddof=1)),
            "acc_gain_range": [float(min(seed_acc_gains)), float(max(seed_acc_gains))],
            "nll_improv_sd": float(np.std(seed_nll_improv, ddof=1)),
            "nll_improv_range": [float(min(seed_nll_improv)), float(max(seed_nll_improv))],
        },
        "analytic_same_mothers": {
            "direct_s17_table": {k: sc_dir[k] for k in ("acc", "nll", "brier", "ece")},
            "full_iq_known_param": {k: sc_iq[k] for k in ("acc", "nll", "brier", "ece")},
        },
        "phase_minus_direct_s17": {
            "acc_gain_t95": paired_t_ci(block_hit[0] - dir_blk_hit),
            "nll_improv_t95": paired_t_ci(dir_blk_nll - block_nll[0]),
        },
        "zero_minus_direct_s17": {
            "acc_gain_t95": paired_t_ci(block_hit[1] - dir_blk_hit),
            "nll_improv_t95": paired_t_ci(dir_blk_nll - block_nll[1]),
        },
    }
    if is_primary:
        pass_acc = bool(acc_t_ci["lo95"] > 0.02)
        pass_nll = bool(nll_t_ci["lo95"] > 0.02)
        res["dual_practical_threshold"] = {
            "acc_threshold": 0.02,
            "nll_threshold": 0.02,
            "acc_t95_lo": float(acc_t_ci["lo95"]),
            "nll_t95_lo": float(nll_t_ci["lo95"]),
            "pass_acc": pass_acc,
            "pass_nll": pass_nll,
            "pass_both": bool(pass_acc and pass_nll),
        }
    return res

    }

    return {
        "condition": {"omega": 0.35, "h": 2, "ell": 2, "t_p": 4.0, "snr_db": 2.0, "sigma_phi": 0.10},
        "mothers": int(n_mothers),
        "work_a_physical_reference": work_a_metrics,
        "arms": arms_out,
        "concrete_trajectory_example": ex_traj,
    }
                arm_res = {}
                for a, aname in enumerate(("phase", "zero")):
                    ps = np.concatenate(
                        [softmax_temperature(raw[a, s_idx], float(t_mat[a, s_idx])) for s_idx in range(len(seeds))],
                        axis=0,
                    )
                    sc = score_vectors(ps, np.tile(lbl, len(seeds)))
                    arm_res[aname] = {k: sc[k] for k in ("acc", "nll", "brier", "ece")}
                sc_d = score_vectors(rp[f"{cname}__{qk}_direct_probs"], lbl)
                arm_res["direct_s17"] = {k: sc_d[k] for k in ("acc", "nll", "brier", "ece")}
                replay_c123[cname][qk] = arm_res

    summary = {
        "seeds": seeds,
        "locked_temps_q3": temps_q3.tolist(),
        "locked_temps_q2": temps_q2.tolist(),
        "cells": cells_res,
        "work_f_primary_040": work_f_primary,
        "work_f_historical_035": work_f_hist,
        "work_g_primary_040": work_g,
        "work_a_q2_c4": work_a_q2,
        "replay_c123_forgetting": replay_c123,
    }

    if verify_against is not None:
        expected = json.loads(verify_against.read_text(encoding="utf-8"))
        max_diff = _compare_nested(expected, summary)
BLIND_CELLS: Dict[str, BlindCellSpec] = {
    "G1_main_extrap_both": BlindCellSpec(
        cell_id="G1_main_extrap_both",
        role="primary_extrap_freq_and_angle",
        abs_omega=0.40,
        h=2,
        ell=2,
        t_p=4.0,
        snr_db=2.0,
        sigma_phi=0.10,
        namespace=NS_BLIND_G1,
        default_blocks=131,
    ),
    "G2_interp_both": BlindCellSpec(
        cell_id="G2_interp_both",
        role="explore_interp_both",
        abs_omega=0.20,
        h=2,
        ell=2,
        t_p=4.0,
        snr_db=2.0,
        sigma_phi=0.10,
        namespace=NS_BLIND_G2,
        default_blocks=8,
    ),
    "G3_extrap_angle": BlindCellSpec(
        cell_id="G3_extrap_angle",
        role="explore_extrap_angle",
        abs_omega=0.20,
        h=4,
        ell=4,
        t_p=8.0,
        snr_db=2.0,
        sigma_phi=0.10,
        namespace=NS_BLIND_G3,
        default_blocks=8,
    ),
    "G4_extrap_freq": BlindCellSpec(
        cell_id="G4_extrap_freq",
        role="explore_extrap_freq",
        abs_omega=0.40,
        h=1,
        ell=1,
        t_p=2.0,
        snr_db=2.0,
        sigma_phi=0.10,
        namespace=NS_BLIND_G4,
        default_blocks=8,
    ),
}


def load_s17_encoder(repo_root: Optional[Path] = None):
    """Load frozen equivariant_b8_s17 encoder and enforce exact SHA-256 fingerprint."""

def public_phase_slots(omega: float, h: int, ell: int, t_p: float, zero_arm: bool = False) -> np.ndarray:
    """Six public phase slots at t = (h, h + ell, t_p) relative to t1 = 0."""
    if zero_arm:
        return np.zeros(6, dtype=np.float32)
    times = (float(h), float(h + ell), float(t_p))
    return np.array([v for t in times for v in (math.cos(omega * t), math.sin(omega * t))], dtype=np.float32)


def condition_spec_from_params(
def build_q2_arm_references(batch: MotherBatch, arm: str, coord_rot: int = 0) -> Tuple[Tuple[Reference, ...], bool]:
    """Return `(references_tuple, deduplicate_flag)` for the 7 Stage 3D Q2 evaluation arms."""
    r = ROOTS[coord_rot % 4]
    tp = batch.condition.t_p
    snr_db = batch.condition.snr_db
    if arm == "none":
        return (), True
    if arm == "valid":
        return (Reference(0, True, batch.p_valid1 * r, tp, snr_db),), True
    if arm == "duplicate_copy_dedup":
        ref0 = Reference(0, True, batch.p_valid1 * r, tp, snr_db)
        return (ref0, ref0), True
    if arm == "duplicate_copy_raw":
        ref0 = Reference(0, True, batch.p_valid1 * r, tp, snr_db)
        return (ref0, ref0), False
    if arm == "independent_only":
        return (Reference(0, False, batch.p_indep * r, tp, snr_db),), True
    if arm == "valid_plus_independent":
        return (
            Reference(0, True, batch.p_valid1 * r, tp, snr_db),
            Reference(1, False, batch.p_indep * r, tp, snr_db),
        ), True
    if arm == "fresh_resample":
        return (
            Reference(0, True, batch.p_valid1 * r, tp, snr_db),
            Reference(1, True, batch.p_valid2 * r, tp, snr_db),
        ), True
    raise ValueError(f"Unknown Q2 arm: {arm}")


def build_blind_seed_manifest(g1_blocks: int = 131) -> dict:
    encoder,
    cell_id: str,
    namespace: int,
    blocks: int,
    abs_omega: float,
    h: int,
    ell: int,
    t_p: float,
    cal_cell_idx: int,
    snr_db: float = 2.0,
    sigma_phi: float = 0.10,
) -> dict:
    """Generate paired blind codes/targets, analytic s17 & full_iq probs, and independent cal_ref."""
    omegas = (-float(abs_omega), float(abs_omega))
    b_map, block_seeds = generate_paired_cfo_blocks(
        namespace=namespace,
        cell_idx=0,
        blocks=blocks,
        h=h,
        ell=ell,
        t_p=t_p,
        omegas=omegas,
        snr_db=snr_db,
        sigma_phi=sigma_phi,
        split_prefix=cell_id,
    )
    n_m = blocks * BLOCK_SIZE
    codes = np.zeros((2, 3, n_m), dtype=np.uint8)
    targets = np.zeros((2, 3, n_m), dtype=np.uint8)
    indep_targets = np.zeros((2, 3, n_m), dtype=np.uint8)
    direct_probs = np.zeros((2, 3, n_m, 4), dtype=np.float64)
    full_iq_probs = np.zeros((2, 3, n_m, 4), dtype=np.float64)
    rot10_codes = np.zeros((2, n_m), dtype=np.uint8)
    rot10_targets = np.zeros((2, n_m), dtype=np.uint8)
    shuffled_perm = np.zeros((2, 3, n_m), dtype=np.int32)

    for s_i, w in enumerate(omegas):
        batch = b_map[w]
        cond = batch.condition
        for r_i, (s1, s2) in enumerate(ROTATIONS):
            v1, v2 = batch.rotated_view(s1, s2)
            z1 = np.asarray(encoder.encode(v1), dtype=np.int64)
            z2 = np.asarray(encoder.encode(v2), dtype=np.int64)
            codes[s_i, r_i] = (16 * z1 + z2).astype(np.uint8)
            targets[s_i, r_i] = sector_of_complex(matched_sum(batch.y3_repeat * ROOTS[s1 % 4])).astype(np.uint8)
            indep_targets[s_i, r_i] = sector_of_complex(matched_sum(batch.y3_indep * ROOTS[s1 % 4])).astype(np.uint8)
            a1, a2 = encoder.table[z1], encoder.table[z2]
            direct_probs[s_i, r_i] = readout_from_table(a1, a2, (), cond, protocol="repeat_x1")["q3"]
            full_iq_probs[s_i, r_i] = solve_full_iq_known_param(v1, v2, (), cond, protocol="repeat_x1")["q3"]
            rng_sh = np.random.default_rng(99123 + 1000 * cal_cell_idx + 10 * s_i + r_i)
            shuffled_perm[s_i, r_i] = rng_sh.permutation(n_m).astype(np.int32)

        v1_10, v2_10 = batch.rotated_view(1, 0)
        z1_10 = np.asarray(encoder.encode(v1_10), dtype=np.int64)
        z2_10 = np.asarray(encoder.encode(v2_10), dtype=np.int64)
        rot10_codes[s_i] = (16 * z1_10 + z2_10).astype(np.uint8)
        rot10_targets[s_i] = sector_of_complex(matched_sum(batch.y3_repeat * ROOTS[1])).astype(np.uint8)

    cal_ref, cal_mass = calibrate_signed_reference_table(
        encoder=encoder,
        cell_idx=cal_cell_idx,
        abs_omega=abs_omega,
        h=h,
        ell=ell,
        t_p=t_p,
        snr_db=snr_db,
        sigma_phi=sigma_phi,
    )
    return {
        "codes": codes,
        "targets": targets,
        "indep_targets": indep_targets,
        "direct_table_probs": direct_probs,
        "full_iq_probs": full_iq_probs,
        "rot10_codes": rot10_codes,
        "rot10_targets": rot10_targets,
        "shuffled_perm": shuffled_perm,
        "cal_ref": cal_ref,
        "cal_mass": cal_mass,
        "block_seeds": np.asarray(block_seeds, dtype=np.uint64),
    }


```

### Event #27: `item.completed` — `command_execution` (`item_23`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '2240|2,240|1440|1,440|G=72|G=16|850 MHz|防止研究|LEMMO|GPU' docs research/stage4_math_and_design -g '*.md' | Select-Object -First 60"
```

**Aggregated Output:**

```text
docs\LONG_TERM_RESEARCH_PLAN.md:13:**外部起点�?* RF-GPT 已能做频谱图问答，MERLIN 已研究低信噪比下的电磁感知与推理，LEMMO 已能从长 I/Q 中按指令抽取证据并描述跨信号关系。LEMMO 使用 Qwen3.5-4B，可用于同底座比较。分别报告双方原生用法，以及在访问权限、缓存精度、训练预算可匹配时的实验。为双方记录重读、上下文和延迟成本。外部方法未支持的题单列，双方均可评分的题才用于性能比较。[RF-GPT](https://arxiv.org/abs/2602.14833) · [MERLIN](https://em-merlin.github.io/) · [LEMMO](https://arxiv.org/html/2611.09476v1)
docs\LONG_TERM_RESEARCH_PLAN.md:37:已有 RF\-GPT 用频谱图进行无线信号问答。MERLIN 研究电磁信号感知与推理及低信噪比条件。LEMMO 已报告从最长八百万个复采样的原�?I/Q 中按指令检索证据，并描述跨信号关系。仅凭“输�?I/Q、接上语言模型、能回答问题”不足以建立本项目的原创性。值得争取的贡献是，针对明确的一族物理问题，给出**有限表示保留了什么、何时必然丢失答案、什么额外观测可以恢复答�?*的定量结果，并在真实波形上检验这些预测。与最接近模型的同题同信息对比将决定可以提出多强的实证主张。[RF\-GPT](https://arxiv.org/abs/2602.14833) · [MERLIN，IEEE/CVF Conference on Computer Vision and Pattern Recognition 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_MERLIN_Building_Low-SNR_Robust_Multimodal_LLMs_for_Electromagnetic_Signals_CVPR_2026_paper.pdf) · [LEMMO](https://arxiv.org/abs/2611.09476)
docs\LONG_TERM_RESEARCH_PLAN.md:64:|Pluto 850 MHz 连续 QPSK�?,096 帧，每帧 7,560 个过采样复样点，已做匹配滤波|用源数据中可核对的参考序列，留出一个窗预测其相位；揭示该窗后，判断另一个窗的预测是否改善。检验绝对帧号和固定重复周期能解释多少结果。|能检验时域相干性、重复结构及参考更新；跨设备普适性需要其他采集�?                                                                                              |
docs\LONG_TERM_RESEARCH_PLAN.md:91:**防止研究兜圈子的约定�?* 每轮开始写一句“做完以后，面对同一段波形，模型能多回答哪个问题”；每轮结束列出一个成功样例、一个失败样例和会随之改变的研究决定。若连续两轮都只能报告同一�?BER 指标的小幅变化，就把该接收优化归档，回到尚未回答的问题。若理论给出不可辨识结论，保留这个结论和它所要求的最小新增观测，这也是主线成果�?
docs\MULTIMODAL_INTEGRATION_PLAN.md:19:| **4B�?B · Qwen3.5** | 2026-03-02 | 4B 是首个领域对照。已有原�?I/Q 研究 LEMMO 使用同款 4B 语言底座并用 LoRA 适配。优先在相同长度、相同问题和相同可见参考下比较。若要归因到编码方法，还要匹配训练数据、可训练参数和优化预算�?B �?9B 可比较同代规模趋势�?[4B 权重](https://huggingface.co/Qwen/Qwen3.5-4B) · [9B 权重](https://huggingface.co/Qwen/Qwen3.5-9B) · [LEMMO 论文第三节](https://arxiv.org/html/2611.09476v1) |
docs\MULTIMODAL_INTEGRATION_PLAN.md:53:视觉模型 BLIP-2 已经展示了冻结宿主、训练小型连接器的技术路线，LEMMO 已经研究指令驱动的原�?I/Q 解读。我们可以直接借用成熟的接入思路。值得投入研究时间的，是测�?*冻结的波形表示为后来的追问留下了什么、哪些证据能消除歧义、实际宿主有没有把这部分信息用起�?*。第一轮的具体交付应是同一批波形上的解析答案、共享表示答案和宿主答案，三者之间的差距会指出下一步该改哪里。[BLIP-2，Proceedings of the 40th International Conference on Machine Learning�?023](https://proceedings.mlr.press/v202/li23q.html) · [LEMMO，arXiv:2611.09476](https://arxiv.org/abs/2611.09476) · [EM-FloatFormer D 组证明](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/D/PROOF.md)
research/stage4_math_and_design\discussions\M2\prompt_r1.md:46:   - Design the formal **Stage 4 Statistical Gate** covering BOTH synthetic extrapolation and real-data Pluto (`G=16`) / CTTC (`G=72` clusters across `9` sessions):
research/stage4_math_and_design\discussions\M6\prompt_r1.md:4:> **How do we formulate, diagnose, gate, and execute Stages 4 �?6 of GitHub's ultimate EM-FloatFormer roadmap with MINIMUM trial-and-error (ideally ZERO wasted GPU runs), using exact mathematical laws rather than empirical guesswork?**
research/stage4_math_and_design\discussions\M6\prompt_r1.md:17:1. `docs/LONG_TERM_RESEARCH_PLAN.md` (historical P1–P12 outcomes + "防止研究兜圈子的约定" + Stage 4�? goals)
research/stage4_math_and_design\discussions\M6\prompt_r1.md:18:2. `docs/MULTIMODAL_INTEGRATION_PLAN.md` (3B�?7B multi-host scaling + LEMMO comparison)
research/stage4_math_and_design\discussions\M6\prompt_r1.md:25:In Stage 3D, `$17.12` of A100-80GB compute was spent training 6 checkpoints (`phase` & `zero`, 3 seeds each) and evaluating 150k+ Q3 predictions + Q2 arms �?ONLY to discover AFTER GPU completion that:
research/stage4_math_and_design\discussions\M6\prompt_r1.md:28:Meanwhile, we now have **`~$12.88` of free Modal compute remaining** to execute **Stage 4** (synthetic extrapolation + real Pluto 850 MHz + real CTTC 9 sessions) and lay the mathematical foundation for Stage 5�?!
research/stage4_math_and_design\discussions\M6\prompt_r1.md:36:2. **Necessary-Condition CPU Pre-Screening & Frozen-Host Linearization Theory (Detecting Failures at \$0 Before GPU Training)**:
research/stage4_math_and_design\discussions\M6\prompt_r1.md:37:   - Derive mathematically verifiable **Tier-0 (\$0 Symbolic)** and **Tier-1 (\$0 CPU Surrogate / Linear Readout / Geometry Checks)** **NECESSARY CONDITIONS** whose violation mathematically proves a connector/host interface CANNOT succeed on GPU:
research/stage4_math_and_design\discussions\M6\prompt_r1.md:38:     - *Criterion 1 (Analytical Channel/Cache Ceiling Gap)*: Before training ANY neural module, compute Bayes risk upper/lower bounds from `(Z, C, R)` alone (`direct_table` / causal real-data frontend MI). If Bayes NLL on real Pluto/CTTC `Z` fails a threshold, STOP & fix frontend (`s17` vs canonical cache) before GPU!
research/stage4_math_and_design\discussions\M6\prompt_r1.md:42:   - Model an R&D sequence of candidate architectures/datasets as sequential Bayesian decisions with GPU cost $c_i$ per epoch/eval block and expected info value $V_i$:
research/stage4_math_and_design\discussions\M6\prompt_r1.md:45:     - Construct an **explicit exact Modal GPU budget plan (`<= $9.50`–`$10.00`, leaving `$2.88+` buffer inside `$12.88` free credits)** covering:
research/stage4_math_and_design\discussions\M6\prompt_r1.md:47:       - Stage 4 Real ADALM-PLUTO (850 MHz, `2,240` blind test frames, `G=16`),
research/stage4_math_and_design\discussions\M6\prompt_r1.md:48:       - Stage 4 Real CTTC (9 sessions, `1,440` blind test packets, `G=72`),
research/stage4_math_and_design\discussions\M6\prompt_r1.md:49:       - Plus exact CPU vs GPU split (downloading & frontending Pluto/CTTC on CPU/Volume first vs A100/L4/A10G forward-only KV-cached scoring!).
research/stage4_math_and_design\discussions\M6\prompt_r1.md:51:   - Systematically diagnose the root recurring traps across P1–P12 and Stage 3D and produce a **strict acyclic dependency DAG** from Stage 4 �?Stage 5 �?Stage 6 (including `Qwen3.5-4B -> Qwen3.5-9B -> Qwen3.8-27B` connector portability & fair `LEMMO` comparison) where EVERY step has a mathematically falsifiable CPU gate BEFORE GPU spend!
research/stage4_math_and_design\discussions\M6\prompt_r1.md:55:     - Run a **10,000-trial Monte Carlo benchmark** comparing fixed brute-force `3-seed full-horizon` (`Stage 3D style`) vs `Tier-1 CPU Pre-Screen + 2-Stage Futility-Gated Sequential GPU` under realistic mixture of broken architectures (like additive `<PHASE>`) and working architectures (like equivariant coupled): verify exact Type-I error (`<= 0.05`), power (`>= 0.90`), and `% compute cost reduction`;
research/stage4_math_and_design\discussions\M4\prompt_r1.md:3:You are **Track M4**, part of a 4-track mathematical & architectural multi-agent collaboration preparing **EM-FloatFormer Stage 4** (`docs/LONG_TERM_RESEARCH_PLAN.md` Stage 4: real SDR data validation on ADALM-PLUTO 850 MHz + CTTC 433/708/2450 MHz paired with frozen Qwen3.5-4B / fixed encoder alignment).
research/stage4_math_and_design\discussions\M4\prompt_r1.md:36:  - Zero-leakage cyclic source isolation (`>= 4,096`-symbol guards, `G = 72` session-loop clusters; `train=1440, dev=864, test=1440` + `1080` stress packets).
research/stage4_math_and_design\discussions\M4\prompt_r1.md:48:   - How should Stage 4 evaluate the **frozen `equivariant_b8_s17` encoder + Stage 4 equivariant/evidence-coupled frozen Qwen3.5-4B connector** on REAL Pluto (`2,240` test frames, `G=16`) and REAL CTTC (`1,440` test packets, `G=72`)?
research/stage4_math_and_design\discussions\M4\prompt_r1.md:55:   - Design the exact Modal execution topology (dataset cache Volume -> CPU feature/frontend preprocessing -> A100/L4/A10G batched KV-cached connector training & evaluation) with **explicit arithmetic bounds** on GPU hours, Modal $/hr rates, VRAM, and total Modal credit consumption so **both** Stage 4 Synthetic Extrapolation (`|\theta| > \pi/2`, Q2 `RefPool`) **AND** Real Pluto + Real CTTC complete cleanly well under `$12.88`!
research/stage4_math_and_design\discussions\M5\prompt_r1.md:3:You are **Track M5**, added to our parallel mathematical design team specifically to examine the **long-term ultimate goal** in GitHub (`docs/LONG_TERM_RESEARCH_PLAN.md` + `docs/MULTIMODAL_INTEGRATION_PLAN.md`) from **first principles and information theory**, ensuring that the representations and architectures we build in Stage 4 generalize cleanly through **Stage 5 (event/trajectory model selection & jump localization)** and **Stage 6 (Ultimate Blind Test active window selection + multi-host 4B/9B/27B transfer + LEMMO comparison)** without dead-end rework or trial-and-error!
research/stage4_math_and_design\discussions\M5\prompt_r1.md:18:2. `docs/MULTIMODAL_INTEGRATION_PLAN.md` (3B�?7B hosts: Ministral-3-3B, Qwen3.5-4B/9B, Qwen3.8-27B, LEMMO comparison protocol)
```

### Event #28: `item.completed` — `command_execution` (`item_25`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$a=Get-Content research/stage3d_qwen_signed_phase/recompute_all.py; $a[327..420]; $a[490..575]; rg -n '"'G1|acc|nll' research/stage3d_qwen_signed_phase/results/summary_master.json | Select-Object -Last 24; rg -n '2026-|17.12|12.88|Billed Cost|wall|A100' research/stage3d_qwen_signed_phase/REPORT.md | Select-Object -First 35; rg -n 'CACHE|QWEN|timeout|cpu=|memory=' research/stage3d_qwen_signed_phase/modal_runner.py"
```

**Aggregated Output:**

```text
    raw_ll_table: np.ndarray,
    temps: np.ndarray,
    codes_rot0: np.ndarray,
    targets_rot0: np.ndarray,
    cal_ref: np.ndarray,
    cal_mass: np.ndarray,
    seeds: Sequence[int],
    contra_text_flipped: Optional[np.ndarray] = None,
    contra_phase_flipped: Optional[np.ndarray] = None,
) -> dict:
    """Work F signed CFO probe: 4-cell factorial, swapped-minus-matched NLL, orientation gain, and single-cache spotlight."""
    n_arms, n_seeds, _, _, _ = raw_ll_table.shape
    n_mothers = codes_rot0.shape[1]
    assert n_mothers % BLOCK_SIZE == 0
    n_blocks = n_mothers // BLOCK_SIZE

    overlap = np.minimum(cal_mass[0], cal_mass[1])
    weights = overlap / overlap.sum()
    target_diff = cal_ref[1] - cal_ref[0]  # D(z)
    norm_d2 = float(np.sum(weights[:, None] * (target_diff ** 2)))
    norm_d = math.sqrt(norm_d2)

    out: Dict[str, object] = {
        "mothers": int(n_mothers),
        "blocks": int(n_blocks),
        "cal_ref_overlap_mass": float(overlap.sum()),
        "cal_ref_odd_norm_sq": norm_d2,
        "arms": {},
    }

    def _analyze_table(tbl_2_256_4: np.ndarray) -> dict:
        contrast = tbl_2_256_4[1] - tbl_2_256_4[0]
        gain = float(np.sum(weights[:, None] * contrast * target_diff) / norm_d2)
        resid_ratio = float(math.sqrt(np.sum(weights[:, None] * ((contrast - gain * target_diff) ** 2)) / norm_d2))
        mean_sign_tv = float(0.5 * np.sum(weights[:, None] * np.abs(contrast)))
        rot_tvs = []
        for z in range(256):
            z1, z2 = divmod(z, 16)
            rz = 16 * ((z1 + 4) % 16) + z2
            disc = tbl_2_256_4[:, rz, :] - np.roll(tbl_2_256_4[:, z, :], 1, axis=-1)
            rot_tvs.append(0.5 * np.abs(disc).sum(axis=-1).mean())
        mean_rot_tv = float(np.dot(weights, rot_tvs))

        # 4-cell physical factorial: p[world_s, told_t, m, sector]
        p = np.stack([[tbl_2_256_4[t][codes_rot0[s]] for t in range(2)] for s in range(2)], axis=0)
        m_eff = (p[0, 1] - p[0, 0] + p[1, 1] - p[1, 0]) / 4.0
        w_eff = (p[1, 0] - p[0, 0] + p[1, 1] - p[0, 1]) / 4.0
        i_eff = (p[1, 1] - p[1, 0] - p[0, 1] + p[0, 0]) / 4.0
        factorial_norm = {
            "metadata_M_sq": float(np.mean(np.sum(m_eff * m_eff, axis=-1))),
            "wave_W_sq": float(np.mean(np.sum(w_eff * w_eff, axis=-1))),
            "interaction_I_sq": float(np.mean(np.sum(i_eff * i_eff, axis=-1))),
        }

        cell_nll = np.zeros((2, 2), dtype=np.float64)
        cell_acc = np.zeros((2, 2), dtype=np.float64)
        idx_m = np.arange(n_mothers)
        matched_nll_m = np.zeros(n_mothers, dtype=np.float64)
        swapped_nll_m = np.zeros(n_mothers, dtype=np.float64)
        matched_hit_m = np.zeros(n_mothers, dtype=np.float64)
        swapped_hit_m = np.zeros(n_mothers, dtype=np.float64)

        for s in range(2):
            y = targets_rot0[s]
            for t in range(2):
                pr = p[s, t]
                sc = score_vectors(pr, y)
                cell_nll[s, t] = sc["nll"]
                cell_acc[s, t] = sc["acc"]
                if s == t:
                    matched_nll_m += 0.5 * sc["per_item_nll"]
                    matched_hit_m += 0.5 * sc["per_item_hit"]
                else:
                    swapped_nll_m += 0.5 * sc["per_item_nll"]
                    swapped_hit_m += 0.5 * sc["per_item_hit"]

        diff_blocks = (swapped_nll_m - matched_nll_m).reshape(n_blocks, BLOCK_SIZE).mean(axis=1)
        return {
            "orientation_gain": gain,
            "residual_ratio": resid_ratio,
            "mean_sign_tv": mean_sign_tv,
            "mean_rotation_10_tv": mean_rot_tv,
            "matched_acc": float(matched_hit_m.mean()),
            "swapped_acc": float(swapped_hit_m.mean()),
            "matched_nll": float(matched_nll_m.mean()),
            "swapped_nll": float(swapped_nll_m.mean()),
            "cell_nll_s_world_t_told": {
                "--": float(cell_nll[0, 0]),
                "-+": float(cell_nll[0, 1]),
                "+-": float(cell_nll[1, 0]),
                "++": float(cell_nll[1, 1]),
            },
            "cell_acc_s_world_t_told": {
                "--": float(cell_acc[0, 0]),
        "cache_code": int(c_spot),
        "z1": int(z1_sp),
        "z2": int(z2_sp),
        "physical_label_neg_world": int(targets_rot0[0, m_spot]),
        "physical_label_pos_world": int(targets_rot0[1, m_spot]),
        "cal_ref_neg": [float(v) for v in cal_ref[0, c_spot]],
        "cal_ref_pos": [float(v) for v in cal_ref[1, c_spot]],
        "phase_arm_told_neg": {
            "raw_loglik": [float(v) for v in raw_ll_table[0, 0, 0, c_spot]],
            "probs": [float(v) for v in p_s0_tbl[0, c_spot]],
            "pred": int(np.argmax(p_s0_tbl[0, c_spot])),
        },
        "phase_arm_told_pos": {
            "raw_loglik": [float(v) for v in raw_ll_table[0, 0, 1, c_spot]],
            "probs": [float(v) for v in p_s0_tbl[1, c_spot]],
            "pred": int(np.argmax(p_s0_tbl[1, c_spot])),
        },
        "zero_arm_told_neg": {
            "raw_loglik": [float(v) for v in raw_ll_table[1, 0, 0, c_spot]],
            "probs": [float(v) for v in z_s0_tbl[0, c_spot]],
            "pred": int(np.argmax(z_s0_tbl[0, c_spot])),
        },
        "zero_arm_told_pos": {
            "raw_loglik": [float(v) for v in raw_ll_table[1, 0, 1, c_spot]],
            "probs": [float(v) for v in z_s0_tbl[1, c_spot]],
            "pred": int(np.argmax(z_s0_tbl[1, c_spot])),
        },
    }

    # Single-modality contradiction probe on phase arm (if supplied)
    if contra_text_flipped is not None and contra_phase_flipped is not None:
        # contra_* shapes: (3, 2, 256, 4)
        # Index s=0/1 corresponds to the PHYSICAL world / reference sign;
        # For contra_text_flipped: phase_slots = s, text = 1-s (so compare against matched s vs phase=s)
        # For contra_phase_flipped: text = s, phase_slots = 1-s
        tf_res = []
        pf_res = []
        for s_idx, seed_val in enumerate(seeds):
            t_val = float(temps[0, s_idx])
            p_match = softmax_temperature(raw_ll_table[0, s_idx], t_val)
            p_tflip = softmax_temperature(contra_text_flipped[s_idx], t_val)
            p_pflip = softmax_temperature(contra_phase_flipped[s_idx], t_val)

            def _contra_stats(p_alt: np.ndarray) -> dict:
                # TV vs matched (same physical sign) and TV vs full swapped (opposite sign)
                tv_vs_matched = []
                tv_vs_full_swap = []
                nll_list = []
                hit_list = []
                for s in range(2):
                    pr = p_alt[s][codes_rot0[s]]
                    pr_m = p_match[s][codes_rot0[s]]
                    pr_sw = p_match[1 - s][codes_rot0[s]]
                    tv_vs_matched.append(0.5 * np.abs(pr - pr_m).sum(axis=-1))
                    tv_vs_full_swap.append(0.5 * np.abs(pr - pr_sw).sum(axis=-1))
                    sc = score_vectors(pr, targets_rot0[s])
                    nll_list.append(sc["per_item_nll"])
                    hit_list.append(sc["per_item_hit"])
                return {
                    "seed": int(seed_val),
                    "acc": float(np.mean(hit_list)),
                    "nll": float(np.mean(nll_list)),
                    "mean_tv_vs_matched_both": float(np.mean(tv_vs_matched)),
                    "mean_tv_vs_swapped_both": float(np.mean(tv_vs_full_swap)),
                }

            tf_res.append(_contra_stats(p_tflip))
            pf_res.append(_contra_stats(p_pflip))
        out["single_modality_contradiction_phase_arm"] = {
            "text_flipped_phase_kept": {
                "mean_acc": float(np.mean([x["acc"] for x in tf_res])),
                "mean_nll": float(np.mean([x["nll"] for x in tf_res])),
                "mean_tv_vs_matched": float(np.mean([x["mean_tv_vs_matched_both"] for x in tf_res])),
                "mean_tv_vs_full_swap": float(np.mean([x["mean_tv_vs_swapped_both"] for x in tf_res])),
                "per_seed": tf_res,
            },
            "phase_flipped_text_kept": {
                "mean_acc": float(np.mean([x["acc"] for x in pf_res])),
                "mean_nll": float(np.mean([x["nll"] for x in pf_res])),
                "mean_tv_vs_matched": float(np.mean([x["mean_tv_vs_matched_both"] for x in pf_res])),
                "mean_tv_vs_full_swap": float(np.mean([x["mean_tv_vs_swapped_both"] for x in pf_res])),
                "per_seed": pf_res,
            },
        }
    return out

3179:          "acc": 0.3098958333333333,
3180:          "nll": 1.3714671369698317,
3185:          "acc": 0.6015625,
3186:          "nll": 0.9926185642639765,
3193:          "acc": 0.3515625,
3194:          "nll": 1.3002661882973068,
3199:          "acc": 0.4262152777777778,
3200:          "nll": 1.2437484411728026,
3205:          "acc": 0.609375,
3206:          "nll": 1.0449790211858683,
3213:          "acc": 0.3559027777777778,
3214:          "nll": 1.3324420421459189,
3219:          "acc": 0.3914930555555556,
3220:          "nll": 1.3019249287623633,
3225:          "acc": 0.6328125,
3226:          "nll": 0.9176777733723241,
3235:    "h_c4_pos_acc": 0.6318359375,
3236:    "h_c4_pos_nll": 0.9650571187776844,
3244:        "matched_accuracy": 0.223876953125,
3245:        "swapped_accuracy": 0.2100830078125,
3246:        "swapped_minus_matched_nll": [
3263:        "matched_accuracy": 0.2158203125,
3264:        "swapped_accuracy": 0.216064453125,
3265:        "swapped_minus_matched_nll": [
56:| **软件运行环境** | Modal `NVIDIA A100-SXM4-80GB` (`81,179.8 MB`), `torch 2.15.0+cu132`, `transformers 5.24.1`, `flash-linear-attention 0.6.0`, `numpy 2.3.5`, `scipy 1.17.0`, `torch.bfloat16` |
82:| 临时计算作业 (`Ephemeral Apps`) | **`$17.12`** | 覆盖 Smoke、Bench（含早先 1 �?OOM）�? 次完整训练、Dev Pilot、全量盲测以�?1 次提前中止的历史 Ministral 检�?|
85:| **实际账单应付 (`Billed Cost`)** | **`$0.00`** | **完全�?`$30.00` Modal 免费计算额度内完成（尚余�?`$12.88` 免费计算额度�?* |
72:    timeout=1800,
89:    timeout=7200,
140:    timeout=1800,
168:    timeout=3600,
204:    timeout=1800,
357:    timeout=1200,
```

### Event #29: `item.completed` — `command_execution` (`item_26`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=Join-Path (Get-Location) '"'research/stage4_math_and_design/M6/.uv-cache'; "'$env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$env:TEMP=$env:UV_CACHE_DIR; $env:TMP=$env:UV_CACHE_DIR; $env:PYTHONUTF8='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B -c \"import numpy,scipy; print(numpy.__version__,scipy.__version__); from pathlib import Path; p=Path('docs/LONG_TERM_RESEARCH_PLAN.md'); x=p.read_text(encoding='utf-8').splitlines(); print('\\n'.join(x[19:34]+x[44:68]+x[88:92]))\""
```

**Aggregated Output:**

```text
Installed 2 packages in 10.94s
2.3.5 1.17.0

项目已经走到**可以为电磁模态规定第一组能力，并给出物理判据**的地方。此前的接收研究留下了相位参考、留出观测、跨帧关系、候选解释和真实采集。这些材料能让我们问一个比“这个码块译对了吗”更直接的问题：同样的输入，哪些物理问题能答，哪些要等另一段证据，编码时扔掉什么会让将来的答案永久丢失？

|已有研究                 |报告中的发现                                                                                                                                                                                                                           |长期计划继承什么                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|P1–P3 的表示、架构、物理约束与负对照|P1 新种子下 CNN 在两项陌生信道上方向一致地优于同预算 Transformer，校正后均为 p=0.125。P2 的相位等变模型改善了一些合成条件，经典 DFT 仍很强。P3 去掉绝对时延输入并加保留项后，稀疏长时延 BER 从 12.291% 降到 6.549%，硬阈值 DFT 为 4.919%。还做过稀疏导频、RMS 编码、混合信道、数值精度、单因素失真与 16-QAM 探索。P3-D/E 的标准信道桥接实际完成的是静态 TDL 近似。|复数编码、相位等变读出、匹配先验的经典对照，以及识别位置捷径的方法。 [P1–P3 总览](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/README.md) · [位置捷径](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/P3_A_MECHANISM_REPORT.md) · [TDL 近似的范围](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/exploratory/p3de_tdl_surrogate/P3DE_SURROGATE_REPORT.md) · [16-QAM 探索](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/exploratory/16qam_joint_pa_probe/REPORT.md)                                                                                                                                                                                                             |
|P4–P6 与三条 P6 后续      |P4 的十种子二维模型没有达到预设译码优势。P5/P5-B/P6 逐步对照复数算子、物理估计、奇偶校验和学习模块。P6 在指定模拟中取得收益，局部与注意力对照仍不足以将总体收益归因给注意力。三个后续分别检查了参数化的连续性、条件方差和失效导频的观测模型。                                                                                                  |把物理输入、编码器和下游译码器的贡献分开。记录缺失样本与测得零值之间的区别。 [P4](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/p4/STAGE3_RESULTS.md) · [P5](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/p5/P5_REPORT.md) · [P5-B](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/p5b/P5B_REPORT.md) · [P6](https://github.com/20171806011/EM-FloatFormer/blob/63ad2ec5b22027ea3d6f15aa50619a9caf6bbcd1/p6/P6_REPORT.md)                                                                                                                                                                                                                                                                                            |
|P7、P8 的三条线           |P7 留出四个导频，先预测再揭示，高分歧比特的概率改变量是低分歧组的 5.52 倍，最终只净少错两个码块。P8 分别考察连续 OFDM 帧和导频位置、从 OFDM 传递状态给单载波，以及离格下的相位传输。跨波形合成实验在稀疏训练条件下，正确历史、清空历史、错配历史的 BLER 分别为 54.84%、89.47%、98.71%。                                                             |已经有“新证据应当改变哪些答案”的实验原型，也有“错误参考会造成什么后果”的对照。 [P7](https://github.com/20171806011/EM-FloatFormer/blob/4dab1ac9ee15b4f1a7f78ae05486c55a2dc3bd72/p7/P7_REPORT.md) · [P8 连续帧](https://github.com/20171806011/EM-FloatFormer/blob/4e7ccdaaacc14cd4101a6f5b824d9974a9255b58/p8/P8_REPORT.md) · [P8 跨波形](https://github.com/20171806011/EM-FloatFormer/blob/ce2ff50e81f6b529ec0311393da0013b0a85b801/p8c/P8C_REPORT.md) · [P8 离格](https://github.com/20171806011/EM-FloatFormer/blob/e76c61eef26151985963460b0ab03684cd562e3e/p8_manifold/P8_MANIFOLD_REPORT.md)                                                                                                                                                                                                                                                                   |
|P9–P11 的跨帧记忆与未知载荷    |P9 表明旧观测是否可靠取决于当前信道。P10 的行内复相关保留路径间的相对相位，同时消去每行公共相位。在规定的合成三径重置实验中，重建分支救回两种旧方法均错的 1,604/2,666 个码块，也使 115 个两者均对的块变错。P11 验证了经过校验的助手行能参与后续估计。                                                                                        |可以指明某个统计量保存了哪类物理关系，又丢掉哪类信息。评价一次更新时同时记下纠正和新增的错误。 [P9](https://github.com/20171806011/EM-FloatFormer/blob/faeea639727aab72740cdd0a719106f83824226e/p9/P9_REPORT.md) · [P10 复相关实验](https://github.com/20171806011/EM-FloatFormer/blob/9e9d93bcbf6345b8645314b223cd954b2f31e63a/p10/payload_reconstruction/README.md) · [P11](https://github.com/20171806011/EM-FloatFormer/blob/307d62fd89206442960fa0cebdcdc6a0530db366/p11/P11_REPORT.md)                                                                                                                                                                                                                                                                                                                                                                                 |
|P12、两条数学研究与 Stage 5  |P12 修复同一锚点被重复计入等问题。远端有效参考可从内容识别时，全局取证可以严格优于固定局部窗。正确后验融合改善平方风险，严格改善比特判决还取决于判决边界。修复后的 Stage 5 在分布内合成条件下，基线、物理组合、加等变网络的 BLER 分别为 48.94%、41.58%、39.54%。分布外的神经增量仍不确定。                                                                 |全局找证据、保留多个解释、一次观测只计一次，以及按问题分别定义损失。 [P12 修复](https://github.com/20171806011/EM-FloatFormer/blob/f566f9b8b0ddfb468a4cb3fd64588bfa46c87212/p12/P12_CONSISTENCY_FIX.md) · [内容寻址](https://github.com/20171806011/EM-FloatFormer/blob/f566f9b8b0ddfb468a4cb3fd64588bfa46c87212/research/content_addressing_scalar/REPORT.md) · [后验融合](https://github.com/20171806011/EM-FloatFormer/blob/f566f9b8b0ddfb468a4cb3fd64588bfa46c87212/research/posterior_fusion/pages/01.md) · [Stage 5](https://github.com/20171806011/EM-FloatFormer/blob/bfe942d94d04004c40575bbb62e57c7e36f3b3e7/research/p12_stage5_fusion_neural/REPORT.md)                                                                                                                                                                                                  |
|四条实测分支、三种数据来源        |435 MHz 两条分支使用同一份 742 帧 16-QAM OFDM 录制。138 帧和 240 帧是不同划分。Pluto 提供连续时域 QPSK，CTTC 提供多会话 QPSK。另有宽带 Wi-Fi CSI 文件的格式和重复性审计。                                                                                                           |在不同信号输入层上分别验证可回答的问题；划分时以会话或物理录制为单位。宽带数据盘点供将来选题。 [435 MHz OFDM](https://github.com/20171806011/EM-FloatFormer/blob/ba39dd5968dbaee7b7a226a9bc78d536278fdee8/research/realdata_sdr_ofdm/REPORT.md) · [压力实验](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/REPORT.md) · [Pluto](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/REPORT.md) · [CTTC](https://github.com/20171806011/EM-FloatFormer/blob/1887aee6de9649bc0dd8a4f05fe3c4542f3f6d77/research/realdata_cttc_qpsk/REPORT.md) · [宽带样本审计](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/DATASET_DISCOVERY.md)|
|11 月 22 日 A–E 会合     |A/B 核对后验读出并定位同一观测被重复利用的问题。C 的连续频偏精化在部分离格条件有益，在跳变上可能有损。D 给出相位跳变后可辨识量和误码下界。E 给出何时值得切换估计器的理论条件，实际 14 帧合成检验比固定连续方法多错 172 位。会合报告中的 435 MHz 数字是对历史逐帧记录的复算。                                                                             |优先延伸 D，借用 A/B 的证据核算与 P7 的预测—揭示设计。频偏精化和门控作为具体问题的工具。 [五组会合](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/README.md) · [D 组证明](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/D/PROOF.md) · [实测输入与复算](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/experiments/historical/REAL_INPUTS.md)                                                                                                                                                                                                                                                                      |

**最直接的起点是 D 组的同一段波形。** 已知非零信道、均匀未编码 16\-QAM、七种等概率非零跳角、无噪无抖动时，已知跳点之后，两个符号的比值仍可精确计算，每个原始标签位的最优平均错误率却是 3/7。若跳点也隐藏，加入一枚保证落在跳变后的已知非零导频，整帧期望 BER 从 24\.7347% 降到 2\.0612%。发生跳变的确切时刻仍可能有歧义。这里已经可以对同一段信号连续追问，并逐个说明答案凭什么成立。[五组会合](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/README.md) · [D 组证明](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/D/PROOF.md) · [实测输入与复算](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/experiments/historical/REAL_INPUTS.md)

### 三、阶段安排

|阶段与预计时间                    |模型将多回答什么                                     |研究工作、交付与决定                                                                                                                                                            |
|---------------------------|---------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|1 · 第 1–3 周，定出可辨识边界        |“两段信号相对转了多少？给我参考之后，先前那个符号能确定吗？”              |从四相模型和 D 组 16-QAM 推导开始。分别写出只有当前波形、加入同相干区间参考、只给独立参考时的可辨识量。构造两组被某种编码压成同一值、但追加参考后正确答案不同的波形。交付短证明、反例及有噪版本可检验的预测。已有统计充分性和对称性结果在文献中注明。                                      |
|2 · 第 4–6 周，建立小型判别实验       |“哪一段观测支持我的判断？去掉它，哪一个答案会变？”                   |用共享母波形制作有效参考、独立参考、重复同一参考和无参考四类配对。给解析方法、只有幅度或逐窗归一化的方法、保留复相位的方法完全相同的输入权限。平衡信噪比、窗位、能量和编号。用精确枚举或可控似然验证理论，冻结问题、划分和评分。输出可在 CPU 复跑的小基准。                                      |
|3 · 第 2–3 个月，共享编码器与首次模型接入    |“同一份表示能答相对关系、预测留出观测，并在参考到来后重新作答吗？”           |用有限预算的一个复数编码器配几个简单读出，与解析相量/多假设后验、同预算实数网络等对照。冻结表示后测试新参考位置、相位轨迹与未见问题组合，记录每题相对参照的性能差距和保留信息的成本。先接 Ministral 3B，再接 Qwen3.5-4B。冻结编码器与宿主，为每个宿主训练连接层，按相同问题分别测直接读出和语言回答。                                   |
|4 · 第 3–6 个月，在真实录制上兑现一个预测  |“在陌生的一次采集中，哪段已知波形是有效参考？揭示后，先前的预测应怎样修正？”      |先用已有 Pluto 或 CTTC 的已知序列遮住—揭示，A 次采集设参数，B 次采集检验。CTTC 可留整会话；Pluto 可检查连续时域窗；435 MHz 网格可检验同题的 OFDM 版本。对缺少真值的问题使用留出导频预测或公开序列定义答案。分别汇报自然录制、基于实测的人工干预与独立来源。交付能从原输入重放的至少一项配对外测。|
|5 · 第 6–9 个月，从相位扩到事件和关系    |“这两个窗口里的变化能由同一条相位轨迹解释吗？变化发生在哪一段？另一种解释还剩多少可能？”|先为平滑漂移、离散跳变和两个独立过程构造可区分与观测等价的案例，控制单窗边际特征。随后用同一表示比较两段波形，返回候选解释、支持时间区间与置信度。跨 QPSK/OFDM 只迁移物理定义相同的问题。若现有数据缺少自然事件真值，报告仿真验证与真实留出预测，另找能区分竞争解释的公开采集。                         |
|6 · 第 9–12 个月，形成可追问的模态原型和论文|“给你一段新录制，请描述有依据的变化，标出证据。再给你另一段，你要修改前面哪句话？”   |在 3B、4B 原型上扩展多轮事件问答，加入 Qwen3.5-9B 和 Qwen3.8-27B，按资源增加跨家族对照。一段录制支持多轮问答，输出具体时间范围、观测依据、可辨识范围和补证后的修订。保留未见录制及未见问题组合做盲评，在共同支持的任务上与公开模型、经典方法和简化版对比。交付理论稿、可复跑的小基准和一个真实录制演示。                                        |

到第三个月，目标是交出**一项有假设的信息下界、一套可运行对照和首个接入已有模型的共享编码器**。到第六个月，目标是让外部读者用真实录制重现一项关于“哪些答案会随证据改变”的预测。后半年建立在该结果或有解释力的反例上。若跨采集的条件差异仍未厘清，时间优先给物理诊断。硬件采集的合适时点，是我们已经说得清缺哪一种观测才能区分两个解释，而已有公开录制恰好没有它。

**一年以后的方向取决于哪些能力真正迁移。** 如果同一表示能在独立来源上稳定完成相位与事件问答，再推进到民用宽带被动录制：从“一个信号发生了什么”扩展到“多段成分何时出现、哪些变化同步、哪一段支持这个解释”。先建立可观测条件和真实标签，再检验跨采样率、带宽与波形制式的迁移。此时才值得为多轮问答扩大训练语料和模型容量。

### 四、真实数据怎样用

|输入                                                   |可用来做的第一项新检验                                                       |推断范围                                                                                                                           |
|-----------------------------------------------------|------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
|Pluto 850 MHz 连续 QPSK。7,096 帧，每帧 7,560 个过采样复样点，已做匹配滤波|用源数据中可核对的参考序列，留出一个窗预测其相位；揭示该窗后，判断另一个窗的预测是否改善。检验绝对帧号和固定重复周期能解释多少结果。|能检验时域相干性、重复结构及参考更新；跨设备普适性需要其他采集。                                                                                               |
|CTTC 三频段、三档名义信噪比、九个 QPSK 会话，含原始 ADC 与官方匹配滤波记录        |整个会话只作测试。在对齐成功的片段上，比较原始与滤波后视图对同一关系问题的回答，以及哪些细节在处理后消失。             |两种视图来自同一次采集；以会话和来源汇总。已有研究是会话内划分。                                                                                               |
|435 MHz 16-QAM OFDM，同步、去循环前缀、傅里叶变换后的复数网格             |先在已有导频及明确可评分的数据位置上遮住—揭示，之后才加受控频偏或相位跳变。                            |这份数据可检验网格上的相位证据。历史 240 帧 B3 对比学习头，在原始工况 BER 为 9.1534% 对 10.6077%，在注入频偏工况为 26.7733% 对 14.4038%；11 月 22 日会合复算了历史表，未重新运行实测 I/Q 推理。|

不请模型回答没有物理定义的问题。例如“两个窗口是否相同发射机”需要发射机身份真值与恰当观测；本计划先问“是否符合一条共享相位轨迹”，并说明替代模型。对一枚外观完全一样、来源又隐藏的错配导频，模型能有多确定取决于相关性线索与错配先验。已知独立参考和隐藏可靠性的参考要分别建模、分别评分。[435 MHz OFDM](https://github.com/20171806011/EM-FloatFormer/blob/ba39dd5968dbaee7b7a226a9bc78d536278fdee8/research/realdata_sdr_ofdm/REPORT.md) · [压力实验](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/REPORT.md) · [Pluto](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/REPORT.md) · [CTTC](https://github.com/20171806011/EM-FloatFormer/blob/1887aee6de9649bc0dd8a4f05fe3c4542f3f6d77/research/realdata_cttc_qpsk/REPORT.md) · [宽带样本审计](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/DATASET_DISCOVERY.md)
第一次汇合围绕同一个波形对完成三件事：数学组列出能排除的解释，基准组给出精确概率，实测组指出同样的观测在真实记录里能否取得。三者有分歧就收窄假设。这一步可以把一个有趣的信号处理现象变成第一条可供独立检验的科学主张。

**防止研究兜圈子的约定。** 每轮开始写一句“做完以后，面对同一段波形，模型能多回答哪个问题”；每轮结束列出一个成功样例、一个失败样例和会随之改变的研究决定。若连续两轮都只能报告同一个 BER 指标的小幅变化，就把该接收优化归档，回到尚未回答的问题。若理论给出不可辨识结论，保留这个结论和它所要求的最小新增观测，这也是主线成果。

```

### Event #30: `item.completed` — `agent_message` (`item_27`)

The saved scores support a paired comparison from Full IQ to s17 to Qwen. They also let me measure what the locked temperatures changed. Separating connector information loss from host information loss requires intermediate data, so I’ll keep that distinction explicit in the verifier.

### Event #31: `item.completed` — `command_execution` (`item_28`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n 'def solve_full_iq_known_param|def readout_from_table|def read\\(' research; "'$a=Get-Content research/stage3d_qwen_signed_phase/recompute_all.py; $a[421..459]; $a[577..728] | Select-String -Pattern '"'return|rot|tv|avg' ; Get-Content research/stage3d_qwen_signed_phase/REPORT.md -Encoding UTF8 | Select-Object -Index (68..88); Get-Content research/stage3d_qwen_signed_phase/qwen_connector.py | Select-Object -Index (49..126)"
```

**Aggregated Output:**

```text
research\fourphase_paired_questions\paired.py:97:def read(cache_value, references, view):
research\stage3c\A\q2_reader.py:12:def read(kernel, z1, z2, references):
research\stage3_ministral_cache\analytic.py:171:def readout_from_table(
research\stage3_ministral_cache\analytic.py:342:def solve_full_iq_known_param(
research\stage4_math_and_design\discussions\M4\round1_events.jsonl:46:{"type":"item.completed","item":{"id":"item_22","type":"command_execution","command":"\"C:\\\\WINDOWS\\\\System32\\\\WindowsPowerShell\\\\v1.0\\\\powershell.exe\" -NoProfile -Command \"py -0p; Get-ChildItem 'C:/Users/李则徐/AppData/Local/uv/cache/wheels-v6/pypi/numpy' -Force | Where-Object Name -Like '2.3.5-cp312*' | Format-List Name,LinkType,Target; Get-ChildItem 'C:/Users/李则徐/AppData/Local/uv/cache/environments-v2' -Name; rg -n 'app.function|cpu=|memory=|gpu=' research/stage3d_qwen_signed_phase/modal_runner.py; rg -n 'def |SNR|NOISE' research/fourphase_paired_questions/paired.py\"","aggregated_output":" -V:3.13 *        C:\\Users\\李则徐\\AppData\\Local\\Programs\\Python\\Python313\\python.exe\r\n\r\n\r\nName     : 2.3.5-cp312-cp312-win_amd64\r\nLinkType : \r\nTarget   : {}\r\n\r\nName     : 2.3.5-cp312-cp312-win_amd64.http\r\nLinkType : \r\nTarget   : {}\r\n\r\nName     : 2.3.5-cp312-cp312-win_amd64.msgpack\r\nLinkType : \r\nTarget   : {}\r\n\r\n\r\n\r\n1093dab72d22fcbe\r\n69:@app.function(\r\n71:    gpu=\"A100-80GB\",\r\n86:@app.function(\r\n88:    gpu=\"A100-80GB\",\r\n137:@app.function(\r\n139:    gpu=\"A100-80GB\",\r\n165:@app.function(\r\n167:    gpu=\"A100-80GB\",\r\n201:@app.function(\r\n203:    gpu=\"A100-80GB\",\r\n354:@app.function(\r\n356:    gpu=\"A100-80GB\",\r\n15:SNR = 10 ** 0.2  # E_s/N_0 = 2 dB, signal energy per window is one\r\n30:def rotate(window, quarter_turns):\r\n34:def parents(split):\r\n37:    noise_sd = math.sqrt(1 / (2 * SNR))\r\n39:    def observe(label):\r\n55:def reference_views(pilot, other):\r\n61:def features(win, view):\r\n75:def likelihood(win):\r\n77:    logw = [2 * SNR / math.sqrt(N) * (total * s.conjugate()).real for s in ROOTS]\r\n81:def normalize_exp(logw):\r\n87:def diff_prob(left, right):\r\n92:def cache(data, view):\r\n97:def read(cache_value, references, view):\r\n103:    logw = [sum(2 * SNR / math.sqrt(N) * (sum(w) * s.conjugate()).real\r\n108:def shift(probs, steps):\r\n112:def maxdelta(a, b):\r\n116:def argmax(p):\r\n120:def iq_pairs(window):\r\n124:def cases(split):\r\n141:def run(split):\r\n","exit_code":0,"status":"completed"}}
research\stage4_math_and_design\discussions\M6\round1_events.jsonl:57:{"type":"item.started","item":{"id":"item_28","type":"command_execution","command":"\"C:\\\\WINDOWS\\\\System32\\\\WindowsPowerShell\\\\v1.0\\\\powershell.exe\" -NoProfile -Command \"rg -n 'def solve_full_iq_known_param|def readout_from_table|def read\\\\(' research; \"'$a=Get-Content research/stage3d_qwen_signed_phase/recompute_all.py; $a[421..459]; $a[577..728] | Select-String -Pattern '\"'return|rot|tv|avg' ; Get-Content research/stage3d_qwen_signed_phase/REPORT.md -Encoding UTF8 | Select-Object -Index (68..88); Get-Content research/stage3d_qwen_signed_phase/qwen_connector.py | Select-Object -Index (49..126)\"","aggregated_output":"","exit_code":null,"status":"in_progress"}}
research\stage4_math_and_design\discussions\M3\round1_events.jsonl:17:{"type":"item.completed","item":{"id":"item_7","type":"command_execution","command":"\"C:\\\\WINDOWS\\\\System32\\\\WindowsPowerShell\\\\v1.0\\\\powershell.exe\" -NoProfile -Command \"rg --files --hidden -g AGENTS.md -g '\"'!**/.git/**'\"'; Get-Content -Encoding UTF8 research/stage3c/A/q2_reader.py\"","aggregated_output":"\"\"\"Q2 at the pinned C4 condition; inputs are the two integer cache codes and pilot matched sums.\"\"\"\r\nimport argparse\r\nimport cmath\r\nimport json\r\nimport math\r\nfrom pathlib import Path\r\n\r\n\r\nKERNEL_PATH = Path(__file__).with_name(\"q2_s17_kernel.json\")\r\n\r\n\r\ndef read(kernel, z1, z2, references):\r\n    \"\"\"Each reference is (acquisition ID, shares_channel, complex matched sum).\r\n\r\n    This kernel is calibrated for same-time 2 dB pilots at t=4. Keep acquisition\r\n    IDs when making copies; assign new IDs to independent measurement noise.\r\n    \"\"\"\r\n    if any(not isinstance(z, int) or not 0 <= z < 16 for z in (z1, z2)):\r\n        raise ValueError(\"Both cache indices must be integers in 0..15\")\r\n    seen = {}\r\n    for identifier, same_channel, matched_sum in references:\r\n        t = complex(matched_sum)\r\n        if not math.isfinite(t.real) or not math.isfinite(t.imag):\r\n            raise ValueError(\"Non-finite matched sum\")\r\n        record = (same_channel, t)\r\n        if identifier in seen and seen[identifier] != record:\r\n            raise ValueError(\"Conflicting metadata or value for a repeated acquisition ID\")\r\n        seen[identifier] = record\r\n    valid = [t for same_channel, t in seen.values() if same_channel]\r\n    if not valid:\r\n        return [0.25]*4\r\n    weighted_t = 2*kernel[\"rho\"]*sum(valid)\r\n    # Stabilize likelihood scale with one common maximum across all g, phase nodes.\r\n    ll = [[(weighted_t*(1j**(-g))*cmath.exp(-1j*ph)).real for g in range(4)]\r\n          for ph in kernel[\"pilot_phase_nodes\"]]\r\n    shift = max(max(row) for row in ll)\r\n    carrier = [sum(w*math.exp(row[g]-shift)\r\n                   for w,row in zip(kernel[\"weights_given_z2\"][z2], ll)) for g in range(4)]\r\n    norm = sum(carrier)\r\n    return [sum(carrier[g]*kernel[\"u_given_z1\"][z1][(k+g)%4] for g in range(4))/norm\r\n            for k in range(4)]\r\n\r\n\r\nif __name__ == \"__main__\":\r\n    p = argparse.ArgumentParser(description=__doc__)\r\n    p.add_argument(\"z1\", type=int)\r\n    p.add_argument(\"z2\", type=int)\r\n    p.add_argument(\"--pilot\", nargs=3, action=\"append\", metavar=(\"ID\",\"REAL\",\"IMAG\"), default=[])\r\n    args = p.parse_args()\r\n    kernel = json.loads(KERNEL_PATH.read_text(encoding=\"utf-8\"))\r\n    refs = [(idx, True, complex(float(a),float(b))) for idx,a,b in args.pilot]\r\n    print(json.dumps(read(kernel, args.z1, args.z2, refs)))\r\n","exit_code":0,"status":"completed"}}
research\stage4_math_and_design\discussions\M5\round1_events.jsonl:13:{"type":"item.completed","item":{"id":"item_4","type":"command_execution","command":"\"C:\\\\WINDOWS\\\\System32\\\\WindowsPowerShell\\\\v1.0\\\\powershell.exe\" -NoProfile -Command 'Get-Content -Encoding utf8 research/matched_cache/learned.py\nGet-Content -Encoding utf8 research/stage3c/G/quotient.py\nGet-Content -Encoding utf8 research/stage3c/A/q2_reader.py'","aggregated_output":"\"\"\"Small pilot-free I/Q encoders with an integer index per received window.\"\"\"\r\nfrom dataclasses import dataclass\r\n\r\nimport numpy as np\r\n\r\n\r\nROOTS = np.array([1, 1j, -1, -1j])\r\nLABELS = np.arange(4)\r\n\r\n\r\ndef input_features(windows, equivariant):\r\n    iq = np.asarray(windows, dtype=np.complex128)\r\n    if iq.ndim != 2 or iq.shape[1] != 3:\r\n        raise ValueError(\"Expected an array of three complex samples per window\")\r\n    turns = np.zeros(len(iq), dtype=np.int64)\r\n    if equivariant:\r\n        # Half-open quadrant [-45��,45��), and its three quarter-turn shifts.\r\n        anchor = iq.sum(axis=1)\r\n        missing = anchor == 0\r\n        if np.any(missing):\r\n            candidates = iq[missing]\r\n            first = np.argmax(candidates != 0, axis=1)\r\n            anchor[missing] = candidates[np.arange(len(candidates)), first]\r\n        u = anchor.real + anchor.imag\r\n        v = anchor.imag - anchor.real\r\n        turns[(u > 0) & (v >= 0)] = 1\r\n        turns[(u <= 0) & (v > 0)] = 2\r\n        turns[(u < 0) & (v <= 0)] = 3\r\n        iq = iq * ROOTS[turns].conj()[:, None]\r\n    return np.column_stack((iq.real, iq.imag)), turns\r\n\r\n\r\ndef forward(x, weights):\r\n    w1, b1, w2, b2 = weights\r\n    hidden = np.tanh(x @ w1 + b1)\r\n    logits = hidden @ w2 + b2\r\n    ex = np.exp(logits - logits.max(axis=1, keepdims=True))\r\n    return hidden, ex / ex.sum(axis=1, keepdims=True)\r\n\r\n\r\ndef fit_network(x, target, rng):\r\n    width = 16\r\n    weights = [\r\n        rng.normal(0, np.sqrt(2 / (6 + width)), (6, width)),\r\n        np.zeros(width),\r\n        rng.normal(0, np.sqrt(2 / (width + 4)), (width, 4)),\r\n        np.zeros(4),\r\n    ]\r\n    first = [np.zeros_like(w) for w in weights]\r\n    second = [np.zeros_like(w) for w in weights]\r\n    step = 0\r\n    for _ in range(48):\r\n        order = rng.permutation(len(x))\r\n        for start in range(0, len(x), 256):\r\n            index = order[start:start + 256]\r\n            xb = x[index]\r\n            hidden, prediction = forward(xb, weights)\r\n            residual = (prediction - target[index]) / len(index)\r\n            delta = (residual @ weights[2].T) * (1 - hidden**2)\r\n            grads = (xb.T @ delta, delta.sum(axis=0), hidden.T @ residual,\r\n                     residual.sum(axis=0))\r\n            step += 1\r\n            for w, m, v, g in zip(weights, first, second, grads):\r\n                m *= 0.9\r\n                m += 0.1 * g\r\n                v *= 0.999\r\n                v += 0.001 * g*g\r\n                w -= 0.005 * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)\r\n    return tuple(weights)\r\n\r\n\r\ndef assign(posterior, centers):\r\n    # The posterior entropy is constant across candidate centers.\r\n    return (posterior @ np.log(centers).T).argmax(axis=1)\r\n\r\n\r\ndef fit_centers(prediction, count, rng):\r\n    if count == 1:\r\n        return prediction.mean(axis=0, keepdims=True)\r\n    centers = [prediction[rng.integers(len(prediction))].copy()]\r\n    # KL-proportional seeding, using the distance to the nearest center so far.\r\n    self_score = np.sum(prediction * np.log(prediction), axis=1)\r\n    distance = np.full(len(prediction), np.inf)\r\n    for _ in range(1, count):\r\n        d = np.maximum(0.0, self_score - prediction @ np.log(centers[-1]))\r\n        distance = np.minimum(distance, d)\r\n        centers.append(prediction[rng.choice(len(prediction), p=distance / distance.sum())].copy())\r\n    centers = np.stack(centers)\r\n    prev = None\r\n    for _ in range(30):\r\n        code = assign(prediction, centers)\r\n        if prev is not None and np.array_equal(prev, code):\r\n            break\r\n        new = centers.copy()\r\n        occupied = np.bincount(code, minlength=count)\r\n        for j in np.flatnonzero(occupied):\r\n            new[j] = prediction[code == j].mean(axis=0)\r\n        if np.any(occupied == 0):\r\n            worst_first = np.argsort(self_score - np.sum(prediction * np.log(centers[code]), axis=1))[::-1]\r\n            for j, i in zip(np.flatnonzero(occupied == 0), worst_first):\r\n                new[j] = prediction[i]\r\n        prev, centers = code, new\r\n    return centers\r\n\r\n\r\n@dataclass\r\nclass Learned:\r\n    equivariant: bool\r\n    weights: tuple\r\n    centers: np.ndarray\r\n    table: np.ndarray\r\n\r\n    def encode(self, windows):\r\n        x, turns = input_features(windows, self.equivariant)\r\n        _, prediction = forward(x, self.weights)\r\n        local = assign(prediction, self.centers)\r\n        return turns * len(self.centers) + local\r\n\r\n    def save(self, path):\r\n        w1, b1, w2, b2 = self.weights\r\n        np.savez(path, equivariant=self.equivariant, w1=w1, b1=b1, w2=w2, b2=b2,\r\n                 centers=self.centers, table=self.table)\r\n\r\n\r\ndef load(path):\r\n    with np.load(path, allow_pickle=False) as data:\r\n        return Learned(bool(data[\"equivariant\"]), tuple(data[k] for k in (\"w1\", \"b1\", \"w2\", \"b2\")),\r\n                       data[\"centers\"], data[\"table\"])\r\n\r\n\r\ndef fit(windows, soft_targets, bits, seed, equivariant):\r\n    if bits not in (4, 8):\r\n        raise ValueError(\"The frozen experiment uses 4 or 8 bits for the two-window cache\")\r\n    x, turns = input_features(windows, equivariant)\r\n    target = np.asarray(soft_targets, dtype=np.float64)\r\n    if target.shape != (len(x), 4) or not np.allclose(target.sum(axis=1), 1):\r\n        raise ValueError(\"Expected one normalized four-label posterior per window\")\r\n    target = np.take_along_axis(target, (LABELS[None, :] + turns[:, None]) % 4, axis=1)\r\n    rng = np.random.default_rng(seed)\r\n    weights = fit_network(x, target, rng)\r\n    _, prediction = forward(x, weights)\r\n    size = 2 ** (bits // 2)\r\n    centers = fit_centers(prediction, size // 4 if equivariant else size, rng)\r\n    local = assign(prediction, centers)\r\n    if np.any(np.bincount(local, minlength=len(centers)) == 0):\r\n        raise ValueError(\"Unused training code in the fitted vector quantizer\")\r\n    table = np.stack([target[local == i].mean(axis=0) for i in range(len(centers))])\r\n    if equivariant:\r\n        table = np.concatenate([np.roll(table, r, axis=1) for r in range(4)])\r\n    return Learned(bool(equivariant), weights, centers, table)\r\n\"\"\"One-call quarter-turn equivariance for the published s17 two-window code.\r\n\r\nExternal wrapper for four-candidate distributions. Pilot data and metadata\r\nremain in their original coordinates. Free text is outside this interface.\r\n\"\"\"\r\n\r\n\r\ndef canonical_input(z1, z2, task):\r\n    \"\"\"Return the two canonical indices and the output right-shift.\"\"\"\r\n    if not all(isinstance(z, int) and 0 <= z < 16 for z in (z1, z2)):\r\n        raise ValueError(\"Expected two integer s17 indices in [0, 15]\")\r\n    turn1, inner1 = divmod(z1, 4)\r\n    turn2, inner2 = divmod(z2, 4)\r\n    if task == \"q1\":\r\n        shift = turn2 - turn1\r\n    elif task in (\"q2\", \"q3_repeat\"):\r\n        shift = turn1\r\n    else:\r\n        raise ValueError(\"Supported tasks are q1, q2 and q3_repeat\")\r\n    return inner1, inner2, shift % 4\r\n\r\n\r\ndef rotate(values, shift):\r\n    if len(values) != 4:\r\n        raise ValueError(\"Expected four scores\")\r\n    return [values[(k-shift) % 4] for k in range(4)]\r\n\r\n\r\ndef one_call(reader, z1, z2, task, context):\r\n    \"\"\"reader takes canonical codes, task and public context, then returns four scores.\"\"\"\r\n    i, j, shift = canonical_input(z1, z2, task)\r\n    return rotate(reader(i, j, task, context), shift)\r\n\"\"\"Q2 at the pinned C4 condition; inputs are the two integer cache codes and pilot matched sums.\"\"\"\r\nimport argparse\r\nimport cmath\r\nimport json\r\nimport math\r\nfrom pathlib import Path\r\n\r\n\r\nKERNEL_PATH = Path(__file__).with_name(\"q2_s17_kernel.json\")\r\n\r\n\r\ndef read(kernel, z1, z2, references):\r\n    \"\"\"Each reference is (acquisition ID, shares_channel, complex matched sum).\r\n\r\n    This kernel is calibrated for same-time 2 dB pilots at t=4. Keep acquisition\r\n    IDs when making copies; assign new IDs to independent measurement noise.\r\n    \"\"\"\r\n    if any(not isinstance(z, int) or not 0 <= z < 16 for z in (z1, z2)):\r\n        raise ValueError(\"Both cache indices must be integers in 0..15\")\r\n    seen = {}\r\n    for identifier, same_channel, matched_sum in references:\r\n        t = complex(matched_sum)\r\n        if not math.isfinite(t.real) or not math.isfinite(t.imag):\r\n            raise ValueError(\"Non-finite matched sum\")\r\n        record = (same_channel, t)\r\n        if identifier in seen and seen[identifier] != record:\r\n            raise ValueError(\"Conflicting metadata or value for a repeated acquisition ID\")\r\n        seen[identifier] = record\r\n    valid = [t for same_channel, t in seen.values() if same_channel]\r\n    if not valid:\r\n        return [0.25]*4\r\n    weighted_t = 2*kernel[\"rho\"]*sum(valid)\r\n    # Stabilize likelihood scale with one common maximum across all g, phase nodes.\r\n    ll = [[(weighted_t*(1j**(-g))*cmath.exp(-1j*ph)).real for g in range(4)]\r\n          for ph in kernel[\"pilot_phase_nodes\"]]\r\n    shift = max(max(row) for row in ll)\r\n    carrier = [sum(w*math.exp(row[g]-shift)\r\n                   for w,row in zip(kernel[\"weights_given_z2\"][z2], ll)) for g in range(4)]\r\n    norm = sum(carrier)\r\n    return [sum(carrier[g]*kernel[\"u_given_z1\"][z1][(k+g)%4] for g in range(4))/norm\r\n            for k in range(4)]\r\n\r\n\r\nif __name__ == \"__main__\":\r\n    p = argparse.ArgumentParser(description=__doc__)\r\n    p.add_argument(\"z1\", type=int)\r\n    p.add_argument(\"z2\", type=int)\r\n    p.add_argument(\"--pilot\", nargs=3, action=\"append\", metavar=(\"ID\",\"REAL\",\"IMAG\"), default=[])\r\n    args = p.parse_args()\r\n    kernel = json.loads(KERNEL_PATH.read_text(encoding=\"utf-8\"))\r\n    refs = [(idx, True, complex(float(a),float(b))) for idx,a,b in args.pilot]\r\n    print(json.dumps(read(kernel, args.z1, args.z2, refs)))\r\n","exit_code":0,"status":"completed"}}
research\stage4_math_and_design\discussions\M1\round1_events.jsonl:30:{"type":"item.completed","item":{"id":"item_14","type":"command_execution","command":"\"C:\\\\WINDOWS\\\\System32\\\\WindowsPowerShell\\\\v1.0\\\\powershell.exe\" -NoProfile -Command \"rg -n 'cal_ref|direct_s17|cfo_train|enumerate\\\\(BLIND' research/stage3d_qwen_signed_phase/modal_runner.py; rg -n 'def .*q3|def .*table|def .*likelihood' research/stage3_ministral_cache research/stage3c -g '*.py' | Select-Object -First 72; \"'$p=Get-Content research/stage3d_qwen_signed_phase/recompute_all.py; $p[47..98]; $p[337..432]; $p[857..880]'","aggregated_output":"research/stage3c\\A\\check_q2.py:30:def fit_likelihood(model, count, seed):\r\nresearch/stage3_ministral_cache\\analytic.py:101:def iq_likelihood(windows: np.ndarray, snr: float) -> np.ndarray:\r\nresearch/stage3_ministral_cache\\analytic.py:148:def _pilot_c4_likelihood(\r\nresearch/stage3_ministral_cache\\analytic.py:162:def _window2_phase_likelihood(t2: np.ndarray, snr: float, phi_grid: np.ndarray) -> np.ndarray:\r\nresearch/stage3_ministral_cache\\analytic.py:171:def readout_from_table(\r\nresearch/stage3c\\F\\check_f.py:62:def validate(table):\r\nresearch/stage3c\\F\\check_f.py:68:def fixed_cache(table, reference, weights):\r\nresearch/stage3c\\F\\check_f.py:90:def evaluate_physical(table, cubes):\r\nresearch/stage3c\\G\\check_g.py:22:def check(table):\r\nresearch/stage3c\\work_results\\A\\verify.py:19:def posterior(table, z1, weights, phase, rho, pilots, separate=False):\r\nresearch/stage3c\\work_results\\F\\verify_f.py:76:def audit(table, reference, weights, cache, target, draws):\r\nresearch/stage3c\\work_results\\D\\audit.py:64:def verify_orbits(table, decisions, read_pair):\r\nresearch/stage3_ministral_cache\\verify_theory.py:303:    def eval_full_iq_q3_and_pilot(t1, t2, grid):\r\n\r\ndef softmax_temperature(raw_ll: np.ndarray, temp: float) -> np.ndarray:\r\n    scaled = np.asarray(raw_ll, dtype=np.float64) / float(temp)\r\n    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)\r\n    ex = np.exp(scaled)\r\n    return ex / np.sum(ex, axis=-1, keepdims=True)\r\n\r\n\r\ndef score_vectors(probs: np.ndarray, labels: np.ndarray) -> dict:\r\n    \"\"\"Compute fractional-tie hits, NLL (clipped at 1e-12), Brier, and 10-bin ECE.\"\"\"\r\n    p = np.asarray(probs, dtype=np.float64)\r\n    y = np.asarray(labels, dtype=np.int64)\r\n    n = len(y)\r\n    if p.shape != (n, 4):\r\n        raise ValueError(f\"Expected {(n, 4)}, got {p.shape}\")\r\n    max_p = np.max(p, axis=1, keepdims=True)\r\n    ties = p == max_p\r\n    tie_counts = ties.sum(axis=1)\r\n    hits = ties[np.arange(n), y].astype(np.float64) / tie_counts.astype(np.float64)\r\n    n_ties = int((tie_counts > 1).sum())\r\n\r\n    p_true_raw = p[np.arange(n), y]\r\n    n_clipped = int((p_true_raw < 1e-12).sum())\r\n    p_true = np.clip(p_true_raw, 1e-12, 1.0)\r\n    nll = -np.log(p_true)\r\n\r\n    one_hot = np.zeros_like(p)\r\n    one_hot[np.arange(n), y] = 1.0\r\n    brier = np.sum((p - one_hot) ** 2, axis=1)\r\n\r\n    confs = np.max(p, axis=1)\r\n    edges = np.linspace(0.0, 1.0, 11)\r\n    ece = 0.0\r\n    for i in range(10):\r\n        mask = (confs >= edges[i]) & (confs <= edges[i + 1]) if i == 0 else (confs > edges[i]) & (confs <= edges[i + 1])\r\n        cnt = int(mask.sum())\r\n        if cnt > 0:\r\n            ece += (cnt / float(n)) * abs(float(hits[mask].mean()) - float(confs[mask].mean()))\r\n\r\n    return {\r\n        \"n\": int(n),\r\n        \"acc\": float(hits.mean()),\r\n        \"nll\": float(nll.mean()),\r\n        \"brier\": float(brier.mean()),\r\n        \"ece\": float(ece),\r\n        \"tie_count\": n_ties,\r\n        \"clipped_count\": n_clipped,\r\n        \"per_item_hit\": hits,\r\n        \"per_item_nll\": nll,\r\n    }\r\n\r\n\r\n    \"\"\"Work F signed CFO probe: 4-cell factorial, swapped-minus-matched NLL, orientation gain, and single-cache spotlight.\"\"\"\r\n    n_arms, n_seeds, _, _, _ = raw_ll_table.shape\r\n    n_mothers = codes_rot0.shape[1]\r\n    assert n_mothers % BLOCK_SIZE == 0\r\n    n_blocks = n_mothers // BLOCK_SIZE\r\n\r\n    overlap = np.minimum(cal_mass[0], cal_mass[1])\r\n    weights = overlap / overlap.sum()\r\n    target_diff = cal_ref[1] - cal_ref[0]  # D(z)\r\n    norm_d2 = float(np.sum(weights[:, None] * (target_diff ** 2)))\r\n    norm_d = math.sqrt(norm_d2)\r\n\r\n    out: Dict[str, object] = {\r\n        \"mothers\": int(n_mothers),\r\n        \"blocks\": int(n_blocks),\r\n        \"cal_ref_overlap_mass\": float(overlap.sum()),\r\n        \"cal_ref_odd_norm_sq\": norm_d2,\r\n        \"arms\": {},\r\n    }\r\n\r\n    def _analyze_table(tbl_2_256_4: np.ndarray) -> dict:\r\n        contrast = tbl_2_256_4[1] - tbl_2_256_4[0]\r\n        gain = float(np.sum(weights[:, None] * contrast * target_diff) / norm_d2)\r\n        resid_ratio = float(math.sqrt(np.sum(weights[:, None] * ((contrast - gain * target_diff) ** 2)) / norm_d2))\r\n        mean_sign_tv = float(0.5 * np.sum(weights[:, None] * np.abs(contrast)))\r\n        rot_tvs = []\r\n        for z in range(256):\r\n            z1, z2 = divmod(z, 16)\r\n            rz = 16 * ((z1 + 4) % 16) + z2\r\n            disc = tbl_2_256_4[:, rz, :] - np.roll(tbl_2_256_4[:, z, :], 1, axis=-1)\r\n            rot_tvs.append(0.5 * np.abs(disc).sum(axis=-1).mean())\r\n        mean_rot_tv = float(np.dot(weights, rot_tvs))\r\n\r\n        # 4-cell physical factorial: p[world_s, told_t, m, sector]\r\n        p = np.stack([[tbl_2_256_4[t][codes_rot0[s]] for t in range(2)] for s in range(2)], axis=0)\r\n        m_eff = (p[0, 1] - p[0, 0] + p[1, 1] - p[1, 0]) / 4.0\r\n        w_eff = (p[1, 0] - p[0, 0] + p[1, 1] - p[0, 1]) / 4.0\r\n        i_eff = (p[1, 1] - p[1, 0] - p[0, 1] + p[0, 0]) / 4.0\r\n        factorial_norm = {\r\n            \"metadata_M_sq\": float(np.mean(np.sum(m_eff * m_eff, axis=-1))),\r\n            \"wave_W_sq\": float(np.mean(np.sum(w_eff * w_eff, axis=-1))),\r\n            \"interaction_I_sq\": float(np.mean(np.sum(i_eff * i_eff, axis=-1))),\r\n        }\r\n\r\n        cell_nll = np.zeros((2, 2), dtype=np.float64)\r\n        cell_acc = np.zeros((2, 2), dtype=np.float64)\r\n        idx_m = np.arange(n_mothers)\r\n        matched_nll_m = np.zeros(n_mothers, dtype=np.float64)\r\n        swapped_nll_m = np.zeros(n_mothers, dtype=np.float64)\r\n        matched_hit_m = np.zeros(n_mothers, dtype=np.float64)\r\n        swapped_hit_m = np.zeros(n_mothers, dtype=np.float64)\r\n\r\n        for s in range(2):\r\n            y = targets_rot0[s]\r\n            for t in range(2):\r\n                pr = p[s, t]\r\n                sc = score_vectors(pr, y)\r\n                cell_nll[s, t] = sc[\"nll\"]\r\n                cell_acc[s, t] = sc[\"acc\"]\r\n                if s == t:\r\n                    matched_nll_m += 0.5 * sc[\"per_item_nll\"]\r\n                    matched_hit_m += 0.5 * sc[\"per_item_hit\"]\r\n                else:\r\n                    swapped_nll_m += 0.5 * sc[\"per_item_nll\"]\r\n                    swapped_hit_m += 0.5 * sc[\"per_item_hit\"]\r\n\r\n        diff_blocks = (swapped_nll_m - matched_nll_m).reshape(n_blocks, BLOCK_SIZE).mean(axis=1)\r\n        return {\r\n            \"orientation_gain\": gain,\r\n            \"residual_ratio\": resid_ratio,\r\n            \"mean_sign_tv\": mean_sign_tv,\r\n            \"mean_rotation_10_tv\": mean_rot_tv,\r\n            \"matched_acc\": float(matched_hit_m.mean()),\r\n            \"swapped_acc\": float(swapped_hit_m.mean()),\r\n            \"matched_nll\": float(matched_nll_m.mean()),\r\n            \"swapped_nll\": float(swapped_nll_m.mean()),\r\n            \"cell_nll_s_world_t_told\": {\r\n                \"--\": float(cell_nll[0, 0]),\r\n                \"-+\": float(cell_nll[0, 1]),\r\n                \"+-\": float(cell_nll[1, 0]),\r\n                \"++\": float(cell_nll[1, 1]),\r\n            },\r\n            \"cell_acc_s_world_t_told\": {\r\n                \"--\": float(cell_acc[0, 0]),\r\n                \"-+\": float(cell_acc[0, 1]),\r\n                \"+-\": float(cell_acc[1, 0]),\r\n                \"++\": float(cell_acc[1, 1]),\r\n            },\r\n            \"swapped_minus_matched_nll_t95\": paired_t_ci(diff_blocks),\r\n            \"swapped_minus_matched_nll_boot95\": block_bootstrap_ci(diff_blocks),\r\n            \"factorial_mean_sq_norm\": factorial_norm,\r\n        }\r\n\r\n    # Physical reference anchor on same mothers\r\n    out[\"physical_cal_reference\"] = _analyze_table(cal_ref)\r\n\r\n\r\ndef recompute_all(data_dir: Path, verify_against: Optional[Path] = None) -> dict:\r\n    \"\"\"Recompute complete summary_master.json solely from exported .npz raw scores in data_dir.\"\"\"\r\n    q3_npz = np.load(data_dir / \"blind_q3_scores.npz\", allow_pickle=False)\r\n    q2_npz = np.load(data_dir / \"blind_q2_scores.npz\", allow_pickle=False)\r\n\r\n    temps_q3 = np.asarray(q3_npz[\"temps_q3\"], dtype=np.float64)\r\n    temps_q2 = np.asarray(q2_npz[\"temps_q2\"], dtype=np.float64)\r\n    seeds = [int(x) for x in q3_npz[\"seeds\"].tolist()]\r\n\r\n    cells_res = {}\r\n    for cid in CELL_ORDER:\r\n        cells_res[cid] = {\r\n            **CELL_META[cid],\r\n            **evaluate_cell_paired(\r\n                raw_ll_table=q3_npz[f\"{cid}__raw_ll\"],\r\n                temps=temps_q3,\r\n                codes=q3_npz[f\"{cid}__codes\"],\r\n                targets=q3_npz[f\"{cid}__targets\"],\r\n                direct_probs=q3_npz[f\"{cid}__direct_table_probs\"],\r\n                full_iq_probs=q3_npz[f\"{cid}__full_iq_probs\"],\r\n                seeds=seeds,\r\n                is_primary=(cid == \"G1_main_extrap_both\"),\r\n            ),\r\n","exit_code":0,"status":"completed"}}
research\stage4_math_and_design\discussions\M1\round1_events.jsonl:34:{"type":"item.completed","item":{"id":"item_15","type":"command_execution","command":"\"C:\\\\WINDOWS\\\\System32\\\\WindowsPowerShell\\\\v1.0\\\\powershell.exe\" -NoProfile -Command 'Get-Content research/stage3_ministral_cache/analytic.py -TotalCount 260; $s=Get-Content research/stage3d_qwen_signed_phase/results/summary_master.json -Raw | ConvertFrom-Json; $s.work_f_primary_040.arms.phase | ConvertTo-Json -Depth 3; $s.work_f_primary_040.physical_cal_reference | ConvertTo-Json -Depth 3'","aggregated_output":"\"\"\"Exact analytic conditional solvers (Discrete C4 & Continuous Circular Quadrature) for Q1, Q2, Q3.\"\"\"\r\nimport math\r\nfrom typing import Optional, Sequence, Tuple\r\nimport numpy as np\r\nfrom scipy.special import erfc, ndtr\r\n\r\nfrom research.stage3_ministral_cache.schema import (\r\n    ConditionSpec,\r\n    N_SAMPLES,\r\n    ROOTS,\r\n    Reference,\r\n    dedup_references,\r\n    matched_sum,\r\n)\r\n\r\n\r\nQUAD_NODES_DEFAULT = 128\r\n\r\n\r\ndef awgn_sector_kernel(snr: float) -> np.ndarray:\r\n    q = 0.5 * float(erfc(math.sqrt(snr / 2.0)))\r\n    return np.array([(1.0 - q) ** 2, q * (1.0 - q), q ** 2, q * (1.0 - q)], dtype=np.float64)\r\n\r\n\r\ndef cont_sector_probs(theta: np.ndarray, snr: float) -> np.ndarray:\r\n    scale = math.sqrt(2.0 * snr)\r\n    p_plus = ndtr(scale * np.cos(theta + math.pi / 4.0))\r\n    q_plus = ndtr(scale * np.sin(theta + math.pi / 4.0))\r\n    return np.stack(\r\n        [\r\n            p_plus * q_plus,\r\n            (1.0 - p_plus) * q_plus,\r\n            (1.0 - p_plus) * (1.0 - q_plus),\r\n            p_plus * (1.0 - q_plus),\r\n        ],\r\n        axis=-1,\r\n    )\r\n\r\n\r\ndef discrete_walk_dist(d: float, steps: int) -> np.ndarray:\r\n    k = 1.0 - 4.0 * d / 3.0\r\n    c = 1.0 if steps <= 0 else (k ** steps)\r\n    p = np.full(4, (1.0 - c) / 4.0, dtype=np.float64)\r\n    p[0] += c\r\n    return p\r\n\r\n\r\ndef wrapped_gaussian_weights(mean: float, variance: float, grid: int = QUAD_NODES_DEFAULT) -> Tuple[np.ndarray, np.ndarray]:\r\n    phi = (2.0 * np.pi / grid) * np.arange(grid, dtype=np.float64) - np.pi\r\n    if variance <= 0.0:\r\n        w = np.zeros(grid, dtype=np.float64)\r\n        idx = int(np.argmin(np.abs(((phi - mean + np.pi) % (2.0 * np.pi)) - np.pi)))\r\n        w[idx] = 1.0\r\n        # For exact deterministic nonzero phase mean (e.g. omega*t with sigma=0),\r\n        # return exact angle `[mean]` if caller checks variance==0, else grid.\r\n        return phi, w\r\n    if not np.isfinite(variance):\r\n        return phi, np.full(grid, 1.0 / grid, dtype=np.float64)\r\n    sd = math.sqrt(variance)\r\n    n_img = max(1, math.ceil((10.0 * sd + np.pi) / (2.0 * np.pi)))\r\n    disp = ((phi - mean + np.pi) % (2.0 * np.pi)) - np.pi\r\n    w = np.zeros(grid, dtype=np.float64)\r\n    for m in range(-n_img, n_img + 1):\r\n        w += np.exp(-0.5 * ((disp + 2.0 * np.pi * m) / sd) ** 2)\r\n    w /= w.sum()\r\n    return phi, w\r\n\r\n\r\ndef wrapped_sector_quad_probs(mean: float, variance: float) -> np.ndarray:\r\n    \"\"\"Exact mass of wrapped N(mean, variance) inside each QPSK sector [-pi/4, pi/4) + k*pi/2.\"\"\"\r\n    if variance <= 0.0:\r\n        out = np.zeros(4, dtype=np.float64)\r\n        z = complex(math.cos(mean), math.sin(mean))\r\n        u = z.real + z.imag\r\n        v = z.imag - z.real\r\n        if u > 0 and v >= 0:\r\n            out[1] = 1.0\r\n        elif u <= 0 and v > 0:\r\n            out[2] = 1.0\r\n        elif u < 0 and v <= 0:\r\n            out[3] = 1.0\r\n        else:\r\n            out[0] = 1.0\r\n        return out\r\n    sd = math.sqrt(variance)\r\n    n_img = max(1, math.ceil((10.0 * sd + np.pi) / (2.0 * np.pi)))\r\n    out = np.zeros(4, dtype=np.float64)\r\n    for k in range(4):\r\n        lo_base = -0.25 * math.pi + 0.5 * math.pi * k\r\n        hi_base = lo_base + 0.5 * math.pi\r\n        mass = 0.0\r\n        for m in range(-n_img, n_img + 1):\r\n            lo = lo_base + 2.0 * math.pi * m\r\n            hi = hi_base + 2.0 * math.pi * m\r\n            mass += float(ndtr((hi - mean) / sd) - ndtr((lo - mean) / sd))\r\n        out[k] = mass\r\n    out /= out.sum()\r\n    return out\r\n\r\n\r\ndef iq_likelihood(windows: np.ndarray, snr: float) -> np.ndarray:\r\n    \"\"\"Exact 4-class QPSK posterior a_j(u) from 3-sample complex windows (M, 3).\"\"\"\r\n    t = matched_sum(windows)\r\n    logw = 2.0 * snr * (t[:, None] * ROOTS[None, :].conj()).real\r\n    logw -= logw.max(axis=1, keepdims=True)\r\n    w = np.exp(logw)\r\n    return w / np.sort(w, axis=1).sum(axis=1, keepdims=True)\r\n\r\n\r\ndef diff_posterior(left: np.ndarray, right: np.ndarray) -> np.ndarray:\r\n    \"\"\"Exact sorted-sum circular difference P(right - left = k mod 4).\"\"\"\r\n    sums = []\r\n    for k in range(4):\r\n        prod = left * np.roll(right, -k, axis=-1)\r\n        sums.append(np.sort(prod, axis=-1).sum(axis=-1))\r\n    return np.stack(sums, axis=-1)\r\n\r\n\r\ndef _normalize_rows(arr: np.ndarray) -> np.ndarray:\r\n    arr = np.maximum(arr, 1e-300)\r\n    return arr / np.sort(arr, axis=-1).sum(axis=-1, keepdims=True)\r\n\r\n\r\ndef _gaussian_transition_matrix(\r\n    phi_from: np.ndarray,\r\n    phi_to: np.ndarray,\r\n    mean_shift: float,\r\n    variance: float,\r\n) -> np.ndarray:\r\n    \"\"\"Matrix K[i, j] proportional to wrapped N(phi_to[i] - phi_from[j] - mean_shift; variance).\"\"\"\r\n    if variance <= 0.0:\r\n        # Nearest-node delta fallback when dt=0 or sigma=0\r\n        mat = np.zeros((len(phi_to), len(phi_from)), dtype=np.float64)\r\n        for j, pf in enumerate(phi_from):\r\n            disp = np.abs(((phi_to - pf - mean_shift + np.pi) % (2.0 * np.pi)) - np.pi)\r\n            mat[int(np.argmin(disp)), j] = 1.0\r\n        return mat\r\n    sd = math.sqrt(variance)\r\n    n_img = max(1, math.ceil((10.0 * sd + np.pi) / (2.0 * np.pi)))\r\n    disp = ((phi_to[:, None] - phi_from[None, :] - mean_shift + np.pi) % (2.0 * np.pi)) - np.pi\r\n    mat = np.zeros_like(disp)\r\n    for m in range(-n_img, n_img + 1):\r\n        mat += np.exp(-0.5 * ((disp + 2.0 * np.pi * m) / sd) ** 2)\r\n    mat /= mat.sum(axis=0, keepdims=True)\r\n    return mat\r\n\r\n\r\ndef _pilot_c4_likelihood(\r\n    t_pilot_sum: Optional[np.ndarray],\r\n    snr: float,\r\n    phi_grid: np.ndarray,\r\n) -> Optional[np.ndarray]:\r\n    \"\"\"L_p(phi_p, g) for pilot sum T_p, shape (M, G, 4), or None if no valid pilots.\"\"\"\r\n    if t_pilot_sum is None:\r\n        return None\r\n    phases = ROOTS[None, :] * np.exp(1j * phi_grid)[:, None]  # (G, 4)\r\n    lw = 2.0 * snr * np.real(t_pilot_sum[:, None, None] * np.conj(phases)[None, :, :])  # (M, G, 4)\r\n    lw -= np.max(lw, axis=(1, 2), keepdims=True)\r\n    return np.exp(lw)\r\n\r\n\r\ndef _window2_phase_likelihood(t2: np.ndarray, snr: float, phi_grid: np.ndarray) -> np.ndarray:\r\n    \"\"\"C4-periodic likelihood L_2(phi_2) = (1/4) sum_v exp(2*snr*Re(T_2 * conj(s_v) * exp(-i*phi_2))).\"\"\"\r\n    phases = ROOTS[None, :] * np.exp(1j * phi_grid)[:, None]  # (G, 4)\r\n    lw = 2.0 * snr * np.real(t2[:, None, None] * np.conj(phases)[None, :, :])  # (M, G, 4)\r\n    m_lw = np.max(lw, axis=2, keepdims=True)\r\n    lik = np.exp(lw - m_lw).sum(axis=2) * np.exp(m_lw[:, :, 0] - np.max(lw, axis=(1, 2))[:, None])\r\n    return 0.25 * lik  # (M, G)\r\n\r\n\r\ndef readout_from_table(\r\n    table_z1: np.ndarray,\r\n    table_z2: np.ndarray,\r\n    references: Sequence[Reference],\r\n    cond: ConditionSpec,\r\n    protocol: str = \"repeat_x1\",\r\n    grid: int = QUAD_NODES_DEFAULT,\r\n    allow_double_count_unsafe: bool = False,\r\n) -> dict:\r\n    \"\"\"Exact readout conditioned ONLY on discrete cache posteriors `(a1, a2)` + deduped refs.\"\"\"\r\n    a1 = np.asarray(table_z1, dtype=np.float64)\r\n    a2 = np.asarray(table_z2, dtype=np.float64)\r\n    n_obs = len(a1)\r\n    snr = cond.snr\r\n\r\n    if not allow_double_count_unsafe:\r\n        valid_refs, _, _, _ = dedup_references(references)\r\n    else:\r\n        valid_refs = [r for r in references if r.shares_channel]\r\n\r\n    if valid_refs:\r\n        t_p_sum = sum(matched_sum(r.samples) for r in valid_refs)\r\n    else:\r\n        t_p_sum = None\r\n\r\n    if cond.drift_type == \"coherent\":\r\n        q1 = diff_posterior(a1, a2)\r\n        if t_p_sum is None:\r\n            q2 = np.full((n_obs, 4), 0.25, dtype=np.float64)\r\n        else:\r\n            lw_p = 2.0 * snr * np.real(t_p_sum[:, None] * ROOTS[None, :].conj())\r\n            lw_p -= lw_p.max(axis=1, keepdims=True)\r\n            bp = np.exp(lw_p)\r\n            bp /= np.sort(bp, axis=1).sum(axis=1, keepdims=True)\r\n            q2 = diff_posterior(bp, a1)\r\n        if protocol == \"repeat_x1\":\r\n            dk = awgn_sector_kernel(snr)\r\n            kmat = np.array([np.roll(dk, w) for w in range(4)], dtype=np.float64)\r\n            q3 = a1 @ kmat\r\n            mu3 = a1 @ ROOTS\r\n        elif protocol == \"independent_x3\":\r\n            q3 = np.full((n_obs, 4), 0.25, dtype=np.float64)\r\n            mu3 = np.zeros(n_obs, dtype=np.complex128)\r\n        else:\r\n            raise ValueError(protocol)\r\n        q3_after = q3.copy()\r\n\r\n    elif cond.drift_type == \"discrete\":\r\n        k_rel = 1.0 - 4.0 * cond.d / 3.0\r\n        ch = 1.0 if cond.h == 0 else (k_rel ** cond.h)\r\n        chl = 1.0 if (cond.h + cond.ell) == 0 else (k_rel ** (cond.h + cond.ell))\r\n        q1_raw = diff_posterior(a1, a2)\r\n        q1 = ch * q1_raw + (1.0 - ch) / 4.0\r\n\r\n        if t_p_sum is None:\r\n            q2 = np.full((n_obs, 4), 0.25, dtype=np.float64)\r\n        else:\r\n            p_0tp = discrete_walk_dist(cond.d, int(round(cond.t_p)))\r\n            lw_p = 2.0 * snr * np.real(t_p_sum[:, None] * ROOTS[None, :].conj())\r\n            lw_p -= lw_p.max(axis=1, keepdims=True)\r\n            bp_raw = np.exp(lw_p)\r\n            bp_raw /= np.sort(bp_raw, axis=1).sum(axis=1, keepdims=True)\r\n            # Propagate by p_0tp: P(g_p - g = w) -> bp[g] = sum_w p_0tp[w] * bp_raw[(g+w)%4]\r\n            bp = np.stack([sum(p_0tp[w] * bp_raw[:, (g + w) % 4] for w in range(4)) for g in range(4)], axis=1)\r\n            bp /= np.sort(bp, axis=1).sum(axis=1, keepdims=True)\r\n            q2 = diff_posterior(bp, a1)\r\n\r\n        if protocol == \"repeat_x1\":\r\n            dk = awgn_sector_kernel(snr)\r\n            pu3 = chl * a1 + (1.0 - chl) / 4.0\r\n            kmat = np.array([np.roll(dk, w) for w in range(4)], dtype=np.float64)\r\n            q3 = pu3 @ kmat\r\n            mu3 = chl * (a1 @ ROOTS)\r\n        elif protocol == \"independent_x3\":\r\n            q3 = np.full((n_obs, 4), 0.25, dtype=np.float64)\r\n            mu3 = np.zeros(n_obs, dtype=np.complex128)\r\n        else:\r\n            raise ValueError(protocol)\r\n        q3_after = q3.copy()\r\n\r\n    else:\r\n        # continuous Brownian (+ omega)\r\n        var2 = float(cond.h) * (cond.sigma_phi ** 2)\r\n        var3 = float(cond.h + cond.ell) * (cond.sigma_phi ** 2)\r\n        varp = float(cond.t_p) * (cond.sigma_phi ** 2)\r\n        mean2 = cond.omega * float(cond.h)\r\n        mean3 = cond.omega * float(cond.h + cond.ell)\r\n        meanp = cond.omega * float(cond.t_p)\r\n\r\n        # Q1: phase rotation weights of a2 shifted by phi_2\r\n{\r\n    \"seed_mean_orientation_gain\":  0.009819795034315315,\r\n    \"seed_mean_residual_ratio\":  0.05485821545817657,\r\n    \"seed_mean_sign_tv\":  0.04003317991781616,\r\n    \"seed_mean_swapped_minus_matched_nll\":  0.01836983052292875,\r\n    \"seed_mean_matched_acc\":  0.23739662849872775,\r\n    \"seed_mean_swapped_acc\":  0.20769720101781167,\r\n    \"seed_averaged_table_metrics\":  {\r\n                                        \"orientation_gain\":  0.009819795034315315,\r\n                                        \"residual_ratio\":  0.033809123399957446,\r\n                                        \"mean_sign_tv\":  0.025698790712428962,\r\n                                        \"mean_rotation_10_tv\":  0.08307747554299151,\r\n                                        \"matched_acc\":  0.18189408396946566,\r\n                                        \"swapped_acc\":  0.18010496183206107,\r\n                                        \"matched_nll\":  1.4147270812818875,\r\n                                        \"swapped_nll\":  1.4308784896540296,\r\n                                        \"cell_nll_s_world_t_told\":  {\r\n                                                                        \"--\":  1.4238574581597658,\r\n                                                                        \"-+\":  1.442908222764981,\r\n                                                                        \"+-\":  1.4188487565430787,\r\n                                                                        \"++\":  1.4055967044040094\r\n                                                                    },\r\n                                        \"cell_acc_s_world_t_told\":  {\r\n                                                                        \"--\":  0.19537213740458015,\r\n                                                                        \"-+\":  0.19119751908396945,\r\n                                                                        \"+-\":  0.16901240458015268,\r\n                                                                        \"++\":  0.16841603053435114\r\n                                                                    },\r\n                                        \"swapped_minus_matched_nll_t95\":  {\r\n                                                                              \"n_blocks\":  131,\r\n                                                                              \"mean\":  0.016151408372142323,\r\n                                                                              \"block_sd\":  0.005366196339988422,\r\n                                                                              \"se\":  0.0004688467511087499,\r\n                                                                              \"t_crit_95\":  1.9783804054470222,\r\n                                                                              \"lo95\":  0.015223851146591276,\r\n                                                                              \"hi95\":  0.017078965597693372\r\n                                                                          },\r\n                                        \"swapped_minus_matched_nll_boot95\":  {\r\n                                                                                 \"n_blocks\":  131,\r\n                                                                                 \"resamples\":  10000,\r\n                                                                                 \"seed\":  2026311,\r\n                                                                                 \"mean\":  0.016151408372142323,\r\n                                                                                 \"lo95\":  0.015252125141685327,\r\n                                                                                 \"hi95\":  0.017072632295022542\r\n                                                                             },\r\n                                        \"factorial_mean_sq_norm\":  {\r\n                                                                       \"metadata_M_sq\":  0.00018537805471633675,\r\n                                                                       \"wave_W_sq\":  0.001119772417564818,\r\n                                                                       \"interaction_I_sq\":  6.3625303969710041E-05\r\n                                                                   }\r\n                                    },\r\n    \"per_seed\":  [\r\n                     {\r\n                         \"orientation_gain\":  -0.0014714446816086089,\r\n                         \"residual_ratio\":  0.06187061738575668,\r\n                         \"mean_sign_tv\":  0.04416703262776987,\r\n                         \"mean_rotation_10_tv\":  0.07111297937771423,\r\n                         \"matched_acc\":  0.2637762404580153,\r\n                         \"swapped_acc\":  0.24015982824427481,\r\n                         \"matched_nll\":  1.3953781588415262,\r\n                         \"swapped_nll\":  1.3925059876056152,\r\n                         \"cell_nll_s_world_t_told\":  {\r\n                                                         \"--\":  1.3969048539674522,\r\n                                                         \"-+\":  1.3927726795721096,\r\n                                                         \"+-\":  1.3922392956391203,\r\n                                                         \"++\":  1.3938514637156003\r\n                                                     },\r\n                         \"cell_acc_s_world_t_told\":  {\r\n                                                         \"--\":  0.2526240458015267,\r\n                                                         \"-+\":  0.23079675572519084,\r\n                                                         \"+-\":  0.24952290076335878,\r\n                                                         \"++\":  0.2749284351145038\r\n                                                     },\r\n                         \"swapped_minus_matched_nll_t95\":  {\r\n                                                               \"n_blocks\":  131,\r\n                                                               \"mean\":  -0.0028721712359112347,\r\n                                                               \"block_sd\":  0.010204875000990796,\r\n                                                               \"se\":  0.0008916040686084472,\r\n                                                               \"t_crit_95\":  1.9783804054470222,\r\n                                                               \"lo95\":  -0.004636103254663029,\r\n                                                               \"hi95\":  -0.0011082392171594404\r\n                                                           },\r\n                         \"swapped_minus_matched_nll_boot95\":  {\r\n                                                                  \"n_blocks\":  131,\r\n                                                                  \"resamples\":  10000,\r\n                                                                  \"seed\":  2026311,\r\n                                                                  \"mean\":  -0.0028721712359112347,\r\n                                                                  \"lo95\":  -0.0045970141698002875,\r\n                                                                  \"hi95\":  -0.001101467741930932\r\n                                                              },\r\n                         \"factorial_mean_sq_norm\":  {\r\n                                                        \"metadata_M_sq\":  0.0005168668016449173,\r\n                                                        \"wave_W_sq\":  0.0008460693957756728,\r\n                                                        \"interaction_I_sq\":  0.00025319832928274094\r\n                                                    },\r\n                         \"seed\":  202611\r\n                     },\r\n                     {\r\n                         \"orientation_gain\":  0.021043669725492792,\r\n                         \"residual_ratio\":  0.054284693290079525,\r\n                         \"mean_sign_tv\":  0.04143398575874327,\r\n                         \"mean_rotation_10_tv\":  0.17702509924593504,\r\n                         \"matched_acc\":  0.2109375,\r\n                         \"swapped_acc\":  0.1868439885496183,\r\n                         \"matched_nll\":  1.4831983487121236,\r\n                         \"swapped_nll\":  1.5250227393517006,\r\n                         \"cell_nll_s_world_t_told\":  {\r\n                                                         \"--\":  1.4571247741400677,\r\n                                                         \"-+\":  1.515881641651842,\r\n                                                         \"+-\":  1.5341638370515591,\r\n                                                         \"++\":  1.5092719232841796\r\n                                                     },\r\n                         \"cell_acc_s_world_t_told\":  {\r\n                                                         \"--\":  0.25656011450381677,\r\n                                                         \"-+\":  0.21827290076335878,\r\n                                                         \"+-\":  0.15541507633587787,\r\n                                                         \"++\":  0.1653148854961832\r\n                                                     },\r\n                         \"swapped_minus_matched_nll_t95\":  {\r\n                                                               \"n_blocks\":  131,\r\n                                                               \"mean\":  0.04182439063957688,\r\n                                                               \"block_sd\":  0.009588744494020364,\r\n                                                               \"se\":  0.0008377724962711783,\r\n                                                               \"t_crit_95\":  1.9783804054470222,\r\n                                                               \"lo95\":  0.04016695794873154,\r\n                                                               \"hi95\":  0.04348182333042222\r\n                                                           },\r\n                         \"swapped_minus_matched_nll_boot95\":  {\r\n                                                                  \"n_blocks\":  131,\r\n                                                                  \"resamples\":  10000,\r\n                                                                  \"seed\":  2026311,\r\n                                                                  \"mean\":  0.04182439063957688,\r\n                                                                  \"lo95\":  0.040214628810498844,\r\n                                                                  \"hi95\":  0.04346096571259359\r\n                                                              },\r\n                         \"factorial_mean_sq_norm\":  {\r\n                                                        \"metadata_M_sq\":  0.0005099856833317363,\r\n                                                        \"wave_W_sq\":  0.004967892858831495,\r\n                                                        \"interaction_I_sq\":  0.00017117050311773901\r\n                                                    },\r\n                         \"seed\":  202622\r\n                     },\r\n                     {\r\n                         \"orientation_gain\":  0.009887160059061767,\r\n                         \"residual_ratio\":  0.04841933569869349,\r\n                         \"mean_sign_tv\":  0.034498521366935334,\r\n                         \"mean_rotation_10_tv\":  0.17719567418333743,\r\n                         \"matched_acc\":  0.23747614503816794,\r\n                         \"swapped_acc\":  0.19608778625954199,\r\n                         \"matched_nll\":  1.4578920382438887,\r\n                         \"swapped_nll\":  1.4740493104090095,\r\n                         \"cell_nll_s_world_t_told\":  {\r\n                                                         \"--\":  1.5139464289535944,\r\n                                                         \"-+\":  1.5279577251349157,\r\n                                                         \"+-\":  1.4201408956831034,\r\n                                                         \"++\":  1.4018376475341834\r\n                                                     },\r\n                         \"cell_acc_s_world_t_told\":  {\r\n                                                         \"--\":  0.23723759541984732,\r\n                                                         \"-+\":  0.17986641221374045,\r\n                                                         \"+-\":  0.21230916030534353,\r\n                                                         \"++\":  0.23771469465648856\r\n                                                     },\r\n                         \"swapped_minus_matched_nll_t95\":  {\r\n                                                               \"n_blocks\":  131,\r\n                                                               \"mean\":  0.0161572721651206,\r\n                                                               \"block_sd\":  0.008305731943857772,\r\n                                                               \"se\":  0.0007256751692887703,\r\n                                                               \"t_crit_95\":  1.9783804054470222,\r\n                                                               \"lo95\":  0.014721610629480247,\r\n                                                               \"hi95\":  0.017592933700760954\r\n                                                           },\r\n                         \"swapped_minus_matched_nll_boot95\":  {\r\n                                                                  \"n_blocks\":  131,\r\n                                                                  \"resamples\":  10000,\r\n                                                                  \"seed\":  2026311,\r\n                                                                  \"mean\":  0.0161572721651206,\r\n                                                                  \"lo95\":  0.014744847891171175,\r\n                                                                  \"hi95\":  0.017592068840072975\r\n                                                              },\r\n                         \"factorial_mean_sq_norm\":  {\r\n                                                        \"metadata_M_sq\":  0.00031739929000029163,\r\n                                                        \"wave_W_sq\":  0.0040141403690335245,\r\n                                                        \"interaction_I_sq\":  0.00017054258057424908\r\n                                                    },\r\n                         \"seed\":  202633\r\n                     }\r\n                 ]\r\n}\r\n{\r\n    \"orientation_gain\":  1.0,\r\n    \"residual_ratio\":  0.0,\r\n    \"mean_sign_tv\":  0.6943259442324701,\r\n    \"mean_rotation_10_tv\":  0.0,\r\n    \"matched_acc\":  0.6461712786259542,\r\n    \"swapped_acc\":  0.03625954198473282,\r\n    \"matched_nll\":  0.94128711026503,\r\n    \"swapped_nll\":  2.996389774546989,\r\n    \"cell_nll_s_world_t_told\":  {\r\n                                    \"--\":  0.9394448567112544,\r\n                                    \"-+\":  3.002745926720927,\r\n                                    \"+-\":  2.9900336223730517,\r\n                                    \"++\":  0.9431293638188055\r\n                                },\r\n    \"cell_acc_s_world_t_told\":  {\r\n                                    \"--\":  0.6474236641221374,\r\n                                    \"-+\":  0.03661736641221374,\r\n                                    \"+-\":  0.03590171755725191,\r\n                                    \"++\":  0.644918893129771\r\n                                },\r\n    \"swapped_minus_matched_nll_t95\":  {\r\n                                          \"n_blocks\":  131,\r\n                                          \"mean\":  2.055102664281959,\r\n                                          \"block_sd\":  0.17310562053315998,\r\n                                          \"se\":  0.015124308289064813,\r\n                                          \"t_crit_95\":  1.9783804054470222,\r\n                                          \"lo95\":  2.025181029116933,\r\n                                          \"hi95\":  2.0850242994469848\r\n                                      },\r\n    \"swapped_minus_matched_nll_boot95\":  {\r\n                                             \"n_blocks\":  131,\r\n                                             \"resamples\":  10000,\r\n                                             \"seed\":  2026311,\r\n                                             \"mean\":  2.055102664281959,\r\n                                             \"lo95\":  2.0256386804496875,\r\n                                             \"hi95\":  2.084100689943572\r\n                                         },\r\n    \"factorial_mean_sq_norm\":  {\r\n                                   \"metadata_M_sq\":  0.20160318171766897,\r\n                                   \"wave_W_sq\":  2.332636364430212E-05,\r\n                                   \"interaction_I_sq\":  2.8427318619844037E-05\r\n                               }\r\n}\r\n","exit_code":0,"status":"completed"}}
research\stage4_math_and_design\discussions\M1\round1_events.jsonl:38:{"type":"item.completed","item":{"id":"item_18","type":"command_execution","command":"\"C:\\\\WINDOWS\\\\System32\\\\WindowsPowerShell\\\\v1.0\\\\powershell.exe\" -NoProfile -Command '$p=Get-Content research/stage3d_qwen_signed_phase/analysis_stage3d.py; $p[0..184]; $p=Get-Content research/stage3_ministral_cache/analytic.py; $p[252..362]'","aggregated_output":"\"\"\"Stage 3D CPU physical calibration, analytic baselines, dev-pilot sample-size lock, and master bundling.\"\"\"\r\nimport argparse\r\nimport json\r\nimport math\r\nfrom pathlib import Path\r\nimport subprocess\r\nimport sys\r\nfrom typing import Dict, List, Optional, Sequence, Tuple\r\n\r\nimport numpy as np\r\nfrom scipy.stats import chi2\r\n\r\nfrom research.stage3_ministral_cache.analytic import (\r\n    readout_from_table,\r\n    solve_full_iq_known_param,\r\n)\r\nfrom research.stage3_ministral_cache.data import generate_mothers\r\nfrom research.stage3_ministral_cache.schema import (\r\n    ROOTS,\r\n    ROTATIONS,\r\n    matched_sum,\r\n    sector_of_complex,\r\n)\r\nfrom research.stage3c.A.q2_reader import read as q2_reader_read\r\nfrom research.stage3d_qwen_signed_phase.data_stage3d import (\r\n    ARM_NAMES,\r\n    BLIND_CELLS,\r\n    CONNECTOR_SEEDS,\r\n    NS_BLIND_Q2_C4,\r\n    NS_DEV_PILOT,\r\n    NS_REF_CALIB,\r\n    Q2_BLIND_ARMS,\r\n    REPLAY_TRAIN_CONDITIONS,\r\n    build_q2_arm_references,\r\n    generate_paired_cfo_blocks,\r\n    load_s17_encoder,\r\n)\r\nfrom research.stage3d_qwen_signed_phase.recompute_all import (\r\n    BLOCK_SIZE,\r\n    CELL_ORDER,\r\n    Q2_LOGICAL_REF_IDX,\r\n    recompute_all,\r\n    score_vectors,\r\n    softmax_temperature,\r\n)\r\n\r\n\r\nREPO_ROOT = Path(__file__).resolve().parents[2]\r\nSTAGE3D_DIR = Path(__file__).resolve().parent\r\n\r\n\r\ndef calibrate_signed_reference_table(\r\n    encoder,\r\n    cell_idx: int,\r\n    abs_omega: float,\r\n    h: int,\r\n    ell: int,\r\n    t_p: float,\r\n    snr_db: float = 2.0,\r\n    sigma_phi: float = 0.10,\r\n    blocks_per_batch: int = 128,\r\n    n_batches: int = 8,\r\n) -> Tuple[np.ndarray, np.ndarray]:\r\n    \"\"\"Calibrate quarter-turn pooled 256-code reference table on independent namespace 960501 (65,536 mothers/sign).\"\"\"\r\n    counts = np.zeros((2, 4, 4, 4), dtype=np.int64)\r\n    mass = np.zeros((2, 256), dtype=np.float64)\r\n    omegas = (-float(abs_omega), float(abs_omega))\r\n    for chunk_i in range(n_batches):\r\n        b_map, _ = generate_paired_cfo_blocks(\r\n            namespace=NS_REF_CALIB,\r\n            cell_idx=cell_idx * 100 + chunk_i,\r\n            blocks=blocks_per_batch,\r\n            h=h,\r\n            ell=ell,\r\n            t_p=t_p,\r\n            omegas=omegas,\r\n            snr_db=snr_db,\r\n            sigma_phi=sigma_phi,\r\n            split_prefix=\"ref_cal\",\r\n        )\r\n        for s_i, w in enumerate(omegas):\r\n            batch = b_map[w]\r\n            z1 = np.asarray(encoder.encode(batch.y1), dtype=np.int64)\r\n            z2 = np.asarray(encoder.encode(batch.y2), dtype=np.int64)\r\n            codes = 16 * z1 + z2\r\n            targets = batch.k3_repeat\r\n            np.add.at(counts[s_i], (z1 % 4, z2 % 4, (targets - z1 // 4) % 4), 1)\r\n            mass[s_i] += np.bincount(codes, minlength=256)\r\n    full = np.zeros((2, 256, 4), dtype=np.float64)\r\n    for k in range(256):\r\n        z1, z2 = divmod(k, 16)\r\n        full[:, k, :] = np.roll(counts[:, z1 % 4, z2 % 4] + 0.5, z1 // 4, axis=-1)\r\n    ref_probs = full / full.sum(axis=-1, keepdims=True)\r\n    mass_probs = mass / mass.sum(axis=-1, keepdims=True)\r\n    return ref_probs, mass_probs\r\n\r\n\r\ndef compute_blind_cell_index_and_analytic(\r\n    encoder,\r\n    cell_id: str,\r\n    namespace: int,\r\n    blocks: int,\r\n    abs_omega: float,\r\n    h: int,\r\n    ell: int,\r\n    t_p: float,\r\n    cal_cell_idx: int,\r\n    snr_db: float = 2.0,\r\n    sigma_phi: float = 0.10,\r\n) -> dict:\r\n    \"\"\"Generate paired blind codes/targets, analytic s17 & full_iq probs, and independent cal_ref.\"\"\"\r\n    omegas = (-float(abs_omega), float(abs_omega))\r\n    b_map, block_seeds = generate_paired_cfo_blocks(\r\n        namespace=namespace,\r\n        cell_idx=0,\r\n        blocks=blocks,\r\n        h=h,\r\n        ell=ell,\r\n        t_p=t_p,\r\n        omegas=omegas,\r\n        snr_db=snr_db,\r\n        sigma_phi=sigma_phi,\r\n        split_prefix=cell_id,\r\n    )\r\n    n_m = blocks * BLOCK_SIZE\r\n    codes = np.zeros((2, 3, n_m), dtype=np.uint8)\r\n    targets = np.zeros((2, 3, n_m), dtype=np.uint8)\r\n    indep_targets = np.zeros((2, 3, n_m), dtype=np.uint8)\r\n    direct_probs = np.zeros((2, 3, n_m, 4), dtype=np.float64)\r\n    full_iq_probs = np.zeros((2, 3, n_m, 4), dtype=np.float64)\r\n    rot10_codes = np.zeros((2, n_m), dtype=np.uint8)\r\n    rot10_targets = np.zeros((2, n_m), dtype=np.uint8)\r\n    shuffled_perm = np.zeros((2, 3, n_m), dtype=np.int32)\r\n\r\n    for s_i, w in enumerate(omegas):\r\n        batch = b_map[w]\r\n        cond = batch.condition\r\n        for r_i, (s1, s2) in enumerate(ROTATIONS):\r\n            v1, v2 = batch.rotated_view(s1, s2)\r\n            z1 = np.asarray(encoder.encode(v1), dtype=np.int64)\r\n            z2 = np.asarray(encoder.encode(v2), dtype=np.int64)\r\n            codes[s_i, r_i] = (16 * z1 + z2).astype(np.uint8)\r\n            targets[s_i, r_i] = sector_of_complex(matched_sum(batch.y3_repeat * ROOTS[s1 % 4])).astype(np.uint8)\r\n            indep_targets[s_i, r_i] = sector_of_complex(matched_sum(batch.y3_indep * ROOTS[s1 % 4])).astype(np.uint8)\r\n            a1, a2 = encoder.table[z1], encoder.table[z2]\r\n            direct_probs[s_i, r_i] = readout_from_table(a1, a2, (), cond, protocol=\"repeat_x1\")[\"q3\"]\r\n            full_iq_probs[s_i, r_i] = solve_full_iq_known_param(v1, v2, (), cond, protocol=\"repeat_x1\")[\"q3\"]\r\n            rng_sh = np.random.default_rng(99123 + 1000 * cal_cell_idx + 10 * s_i + r_i)\r\n            shuffled_perm[s_i, r_i] = rng_sh.permutation(n_m).astype(np.int32)\r\n\r\n        v1_10, v2_10 = batch.rotated_view(1, 0)\r\n        z1_10 = np.asarray(encoder.encode(v1_10), dtype=np.int64)\r\n        z2_10 = np.asarray(encoder.encode(v2_10), dtype=np.int64)\r\n        rot10_codes[s_i] = (16 * z1_10 + z2_10).astype(np.uint8)\r\n        rot10_targets[s_i] = sector_of_complex(matched_sum(batch.y3_repeat * ROOTS[1])).astype(np.uint8)\r\n\r\n    cal_ref, cal_mass = calibrate_signed_reference_table(\r\n        encoder=encoder,\r\n        cell_idx=cal_cell_idx,\r\n        abs_omega=abs_omega,\r\n        h=h,\r\n        ell=ell,\r\n        t_p=t_p,\r\n        snr_db=snr_db,\r\n        sigma_phi=sigma_phi,\r\n    )\r\n    return {\r\n        \"codes\": codes,\r\n        \"targets\": targets,\r\n        \"indep_targets\": indep_targets,\r\n        \"direct_table_probs\": direct_probs,\r\n        \"full_iq_probs\": full_iq_probs,\r\n        \"rot10_codes\": rot10_codes,\r\n        \"rot10_targets\": rot10_targets,\r\n        \"shuffled_perm\": shuffled_perm,\r\n        \"cal_ref\": cal_ref,\r\n        \"cal_mass\": cal_mass,\r\n        \"block_seeds\": np.asarray(block_seeds, dtype=np.uint64),\r\n    }\r\n\r\n\r\ndef compute_dev_pilot_sample_size(\r\n    pilot_raw_ll: np.ndarray,\r\n    temps_q3: np.ndarray,\r\n    repo_root: Path = REPO_ROOT,\r\n        var2 = float(cond.h) * (cond.sigma_phi ** 2)\r\n        var3 = float(cond.h + cond.ell) * (cond.sigma_phi ** 2)\r\n        varp = float(cond.t_p) * (cond.sigma_phi ** 2)\r\n        mean2 = cond.omega * float(cond.h)\r\n        mean3 = cond.omega * float(cond.h + cond.ell)\r\n        meanp = cond.omega * float(cond.t_p)\r\n\r\n        # Q1: phase rotation weights of a2 shifted by phi_2\r\n        w_quad2 = wrapped_sector_quad_probs(mean2, var2)  # mass in quadrant k for phi_2\r\n        # Pr(u2_apparent = v) -> true u2 distribution rotated back: sum_k w_quad2[k] * a2[:, (u2+k)%4]\r\n        a2_unshifted = sum(w_quad2[k] * np.roll(a2, -k, axis=1) for k in range(4))\r\n        q1 = _normalize_rows(diff_posterior(a1, a2_unshifted))\r\n\r\n        # Q2: continuous wrapped-Gaussian pilot prior at t_p\r\n        if t_p_sum is None:\r\n            q2 = np.full((n_obs, 4), 0.25, dtype=np.float64)\r\n        else:\r\n            phi_g, w_p = wrapped_gaussian_weights(meanp, varp, grid=grid)\r\n            # For exact zero-var deterministic offset (omega*t_p)\r\n            if varp <= 0.0:\r\n                dir_p = ROOTS * np.exp(1j * meanp)\r\n                lw = 2.0 * snr * np.real(t_p_sum[:, None] * np.conj(dir_p)[None, :])\r\n                lw -= lw.max(axis=1, keepdims=True)\r\n                bp = np.exp(lw)\r\n                bp /= np.sort(bp, axis=1).sum(axis=1, keepdims=True)\r\n            else:\r\n                lp = _pilot_c4_likelihood(t_p_sum, snr, phi_g)  # (M, G, 4)\r\n                bp = np.einsum(\"g,mgk->mk\", w_p, lp)\r\n                bp /= np.sort(bp, axis=1).sum(axis=1, keepdims=True)\r\n            q2 = _normalize_rows(diff_posterior(bp, a1))\r\n\r\n        # Q3 pre-reveal\r\n        if protocol == \"repeat_x1\":\r\n            phi_g, w_3 = wrapped_gaussian_weights(mean3, var3, grid=grid)\r\n            if var3 <= 0.0:\r\n                theta_u = 0.5 * math.pi * np.arange(4, dtype=np.float64) + mean3\r\n                mc_u = cont_sector_probs(theta_u, snr)  # (4, 4)\r\n                q3 = a1 @ mc_u\r\n                mu3 = (a1 @ ROOTS) * np.exp(1j * mean3)\r\n            else:\r\n                theta_grid = 0.5 * math.pi * np.arange(4, dtype=np.float64)[:, None] + phi_g[None, :]\r\n                mc_grid = cont_sector_probs(theta_grid, snr)  # (4, G, 4)\r\n                q3 = np.einsum(\"mu,ugk,g->mk\", a1, mc_grid, w_3)\r\n                mu3 = (a1 @ ROOTS) * complex(w_3 @ np.exp(1j * phi_g))\r\n        elif protocol == \"independent_x3\":\r\n            q3 = np.full((n_obs, 4), 0.25, dtype=np.float64)\r\n            mu3 = np.zeros(n_obs, dtype=np.complex128)\r\n        else:\r\n            raise ValueError(protocol)\r\n        q3_after = _normalize_rows(q3.copy())\r\n        # Note: if pilot reveals phase phi_p and t_p == t_3 (e.g. C6 with finite drift or general),\r\n        # for table readout (which only observes pilot sum) under continuous drift t_p and t_3 correlated,\r\n        # compute exact conditional posterior p(phi_3 | P) if t_p > 0 and sigma_phi > 0!\r\n        if protocol == \"repeat_x1\" and t_p_sum is not None and cond.sigma_phi > 0.0 and cond.t_p > 0.0:\r\n            t_share = min(float(cond.t_p), float(cond.h + cond.ell))\r\n            phi_p, w_p = wrapped_gaussian_weights(meanp, varp, grid=grid)\r\n            lp = _pilot_c4_likelihood(t_p_sum, snr, phi_p)  # (M, G, 4)\r\n            lik_p_marginal = 0.25 * lp.sum(axis=2)  # (M, G)\r\n            post_phip = _normalize_rows(lik_p_marginal * w_p[None, :])\r\n            if abs(float(cond.h + cond.ell) - float(cond.t_p)) < 1e-12:\r\n                post_phi3 = post_phip\r\n            elif float(cond.h + cond.ell) > float(cond.t_p):\r\n                dt = float(cond.h + cond.ell) - float(cond.t_p)\r\n                trans = _gaussian_transition_matrix(phi_p, phi_p, cond.omega * dt, dt * cond.sigma_phi ** 2)\r\n                post_phi3 = post_phip @ trans.T\r\n            else:\r\n                # Brownian bridge from phi_p at t_p back to t_3 < t_p\r\n                alpha = float(cond.h + cond.ell) / float(cond.t_p)\r\n                var_br = alpha * (1.0 - alpha) * varp\r\n                # Note E[phi_3 | phi_p] = alpha * phi_p\r\n                disp = ((phi_p[:, None] - alpha * phi_p[None, :] + math.pi) % (2.0 * math.pi)) - math.pi\r\n                sd_br = math.sqrt(max(var_br, 1e-12))\r\n                n_img = max(1, math.ceil((10.0 * sd_br + math.pi) / (2.0 * math.pi)))\r\n                trans = sum(np.exp(-0.5 * ((disp + 2.0 * math.pi * m) / sd_br) ** 2) for m in range(-n_img, n_img + 1))\r\n                trans /= trans.sum(axis=0, keepdims=True)\r\n                post_phi3 = post_phip @ trans.T\r\n            theta_grid = 0.5 * math.pi * np.arange(4, dtype=np.float64)[:, None] + phi_p[None, :]\r\n            mc_grid = cont_sector_probs(theta_grid, snr)\r\n            q3_after = _normalize_rows(np.einsum(\"mu,ugk,mg->mk\", a1, mc_grid, post_phi3))\r\n\r\n    return {\r\n        \"q1\": _normalize_rows(q1),\r\n        \"q2\": _normalize_rows(q2),\r\n        \"q3\": _normalize_rows(q3),\r\n        \"q3_after_pilot\": _normalize_rows(q3_after),\r\n        \"mu3\": mu3,\r\n    }\r\n\r\n\r\ndef solve_full_iq_known_param(\r\n    y1: np.ndarray,\r\n    y2: np.ndarray,\r\n    references: Sequence[Reference],\r\n    cond: ConditionSpec,\r\n    protocol: str = \"repeat_x1\",\r\n    grid: int = QUAD_NODES_DEFAULT,\r\n    allow_double_count_unsafe: bool = False,\r\n) -> dict:\r\n    \"\"\"Analytic conditional Bayesian optimal baseline using UNQUANTIZED full complex I/Q and KNOWN cond params.\"\"\"\r\n    snr = cond.snr\r\n    a1 = iq_likelihood(y1, snr)\r\n    a2 = iq_likelihood(y2, snr)\r\n    if cond.drift_type in (\"coherent\", \"discrete\"):\r\n        # For coherent & discrete C4, a1 and a2 are exact sufficient statistics of (Y1, Y2)!\r\n        return readout_from_table(\r\n            a1,\r\n            a2,\r\n            references,\r\n            cond,\r\n            protocol=protocol,\r\n            grid=grid,\r\n","exit_code":0,"status":"completed"}}
                "-+": float(cell_acc[0, 1]),
                "+-": float(cell_acc[1, 0]),
                "++": float(cell_acc[1, 1]),
            },
            "swapped_minus_matched_nll_t95": paired_t_ci(diff_blocks),
            "swapped_minus_matched_nll_boot95": block_bootstrap_ci(diff_blocks),
            "factorial_mean_sq_norm": factorial_norm,
        }

    # Physical reference anchor on same mothers
    out["physical_cal_reference"] = _analyze_table(cal_ref)

    for a, aname in enumerate(("phase", "zero")):
        seed_tables = np.stack(
            [softmax_temperature(raw_ll_table[a, s_idx], float(temps[a, s_idx])) for s_idx in range(n_seeds)],
            axis=0,
        )
        per_seed_res = []
        for s_idx, seed_val in enumerate(seeds):
            r = _analyze_table(seed_tables[s_idx])
            r["seed"] = int(seed_val)
            per_seed_res.append(r)

        # Average probabilities across 3 seeds for pooled behavioral table summary + mean of seed stats
        mean_tbl = seed_tables.mean(axis=0)
        pooled_tbl_res = _analyze_table(mean_tbl)
        out["arms"][aname] = {
            "seed_mean_orientation_gain": float(np.mean([x["orientation_gain"] for x in per_seed_res])),
            "seed_mean_residual_ratio": float(np.mean([x["residual_ratio"] for x in per_seed_res])),
            "seed_mean_sign_tv": float(np.mean([x["mean_sign_tv"] for x in per_seed_res])),
            "seed_mean_swapped_minus_matched_nll": float(
                np.mean([x["swapped_minus_matched_nll_t95"]["mean"] for x in per_seed_res])
            ),
            "seed_mean_matched_acc": float(np.mean([x["matched_acc"] for x in per_seed_res])),
            "seed_mean_swapped_acc": float(np.mean([x["swapped_acc"] for x in per_seed_res])),
            "seed_averaged_table_metrics": pooled_tbl_res,
            "per_seed": per_seed_res,
        }


def work_g_rotation_and_controls(
    rot10_codes: np.ndarray,
    rot10_targets: np.ndarray,
    """Work G raw rotation equivariance, independent_x3, shuffled_Z, and offline 16-way Pf / F0."""
        indep_tvs = []
        rot01_max_abs, rot01_mean_abs, rot01_fail_020 = [], [], []
        rot11_max_abs, rot11_mean_abs, rot11_fail_020 = [], [], []
        rot10_max_abs, rot10_mean_abs, rot10_fail_020 = [], [], []
        worst_rot_dev = None
            p_pf = compute_16way_group_avg(p_raw)
                pr_rot = p_raw[sign_i][codes[sign_i]]     # (3, M, 4)
                pf_rot = p_pf[sign_i][codes[sign_i]]
                f0_rot = p_f0[sign_i][codes[sign_i]]
                pr_10 = p_raw[sign_i][rot10_codes[sign_i]]
                # Rotation checks vs rot=(0,0)
                d01 = np.abs(pr_rot[1] - pr_rot[0]).max(axis=1)
                d11 = np.abs(pr_rot[2] - np.roll(pr_rot[0], 1, axis=1)).max(axis=1)
                d10 = np.abs(pr_10 - np.roll(pr_rot[0], 1, axis=1)).max(axis=1)
                rot01_max_abs.append(float(d01.max()))
                rot01_mean_abs.append(float(d01.mean()))
                rot01_fail_020.append(float((d01 > 0.20).mean()))
                rot11_max_abs.append(float(d11.max()))
                rot11_mean_abs.append(float(d11.mean()))
                rot11_fail_020.append(float((d11 > 0.20).mean()))
                rot10_max_abs.append(float(d10.max()))
                rot10_mean_abs.append(float(d10.mean()))
                rot10_fail_020.append(float((d10 > 0.20).mean()))
                if worst_rot_dev is None or float(d11[idx_worst]) > worst_rot_dev["max_abs_diff"]:
                    worst_rot_dev = {
                        "probs_00": [float(v) for v in pr_rot[0, idx_worst]],
                        "probs_11": [float(v) for v in pr_rot[2, idx_worst]],
                sc_raw = score_vectors(pr_rot.reshape(-1, 4), targets[sign_i].reshape(-1))
                sc_pf = score_vectors(pf_rot.reshape(-1, 4), targets[sign_i].reshape(-1))
                sc_f0 = score_vectors(f0_rot.reshape(-1, 4), targets[sign_i].reshape(-1))
                indep_tvs.append(0.5 * np.abs(pr_ind - 0.25).sum(axis=-1))
            "raw_rotation_checks": {
                "rot01_hold": {
                    "mean_max_abs": float(np.mean(rot01_mean_abs)),
                    "worst_max_abs": float(np.max(rot01_max_abs)),
                    "frac_gt_0p20": float(np.mean(rot01_fail_020)),
                "rot11_roll": {
                    "mean_max_abs": float(np.mean(rot11_mean_abs)),
                    "worst_max_abs": float(np.max(rot11_max_abs)),
                    "frac_gt_0p20": float(np.mean(rot11_fail_020)),
                "rot10_roll": {
                    "mean_max_abs": float(np.mean(rot10_mean_abs)),
                    "worst_max_abs": float(np.max(rot10_max_abs)),
                    "frac_gt_0p20": float(np.mean(rot10_fail_020)),
                "worst_rot11_example": worst_rot_dev,
                    "mean_tv_from_uniform": float(np.mean(indep_tvs)),
                    "max_tv_from_uniform": float(np.max(indep_tvs)),
            "offline_group_avg_16way": {
    return out

1. **�������û����µ� `71.64 GiB` OOM ��λ����Σ�[qwen_connector.py](qwen_connector.py)��[bench_steady.json](results/bench_steady.json)��**��
   - ���� `_make_differentiable_qwen35_cache()` ���հ� `_update_conv(..., _l=layer)` ����ÿ�� `LinearAttentionLayer` ʵ������ Python �������� `layer -> update_conv_state -> _l -> layer` ֱ�����û����� `cand_out` �� `use_cache=False` �Դ��� `cache.update_recurrent_state` д�����ͼ����������΢���ļ���ͼ�޷������ü�����ʱ���գ����������� Seed 1 ����������� `71.64 GiB` OOM��
   - **�޸�����**������ģ�鼶�������� `_OutOfPlaceLinearAttentionLayer(LinearAttentionLayer)` �л� `layer.__class__`����ʵ���հ����û�����ͬʱʹ��ԭ�� `F.log_softmax(..., dim=-1, dtype=torch.float32)` ����ȫ�ʱ� `.float()` ���ƣ����� `cand_out` ǰ��ʱ `del pref_out, out_cache, pref_cache`��
   - **ʵ����֤**����������� `Q3_after_pilot`��`351` �ַ� + `1` ���ο����� `Q2_fresh_resample`��`2` ���ο��������� 4 ��΢���������Դ��ϸ�㶨Ϊ **`[8055.2, 8055.2, 8055.2, 8055.2] MB`**����й©����`micro_batch_size=16` ѵ����ֵ�ȶ��� **`52,972.5 MB`**����̬ `1,975.7 ms / 32-ex step`����ä������ `b=64` ��ֵ **`45,409.9 MB`**��`23.88 ms/��`����
2. **`bfloat16` + FLA �˵�����Сһ�����������͸�����棨[direct_vs_table_parity.json](results/direct_vs_table_parity.json)��**��
   - ���� 256 �����ȫ��ͬ�� **`matched_batch_size = 64`** �����£�ֱ��ǰ���� 256 ����� G1 ä�� 256 �������˶���ȡ�� **`max_abs_ll_diff = 0.0`**��**`max_abs_prob_diff = 0.0`**��**`argmax_agreement = 1.0`**��4 �� `(arm, ��0.40)` ȫ����ȷΪ�㣩��
   - ͬʱ��¼�˿�����С�� BF16/Triton ����ֿ�������������ѵ��ǰ���Ȩ���ϣ�`bench_steady.json`����`b=1/16` �� `b=64` �� `max_abs_ll_diff` Ϊ `0.4816�C0.4830`��`argmax_agreement = 1.0`��������ѵ����Ԥ��߶Ƚӽ����ȵ�ä��ģ���ϣ�`direct_vs_table_parity.json` �� `cross_batch_b16_vs_b64`����`b=16` vs `b=64` �� `max_abs_ll_diff` ���� `0.115�C0.289`��`max_abs_prob_diff = 0.024�C0.067`�����������ѡһ���ֱ߽�ϸ΢�Ŷ������� `argmax_agreement` �� `phase -0.40` Ϊ `0.6875`��`phase +0.40: 0.8750`, `zero -0.40: 1.0000`, `zero +0.40: 0.9375`����ȫä��ͳһ��������������ȫä��ǰ�򣩾�ͳһ�����ڹ̶� `batch_size = 64` �£�ȷ���ڲ��ϸ���Ǣ��

### 1.3 Modal ��Ѷ�������˵���ȫ���㸶�ѣ�

| ������Ŀ | Modal �Ʒ��˵���¼ | ˵�� |
|---|---:|---|
| ��ʱ������ҵ (`Ephemeral Apps`) | **`$17.12`** | ���� Smoke��Bench�������� 1 �� OOM����6 ������ѵ����Dev Pilot��ȫ��ä���Լ� 1 ����ǰ��ֹ����ʷ Ministral ��� |
| �־û����洢 (`Volumes`: `qwen3-model-cache` + `stage3d-results`) | **`$1.34`** | Free Storage ȫ��ֿ� |
| �������� (`Network Egress`) | **`$0.01`** | Network Egress Allowance ȫ��ֿ� |
| **ʵ���˵�Ӧ�� (`Billed Cost`)** | **`$0.00`** | **��ȫ�� `$30.00` Modal ��Ѽ���������ɣ�����Լ `$12.88` ��Ѽ����ȣ�** |

---

## 2. ���۶������á�6 ��ѵ��������������äǰ����
        num_prefix_tokens: int = 8,
    ):
        super().__init__()
        self.num_states = int(num_states)
        self.hidden_size = int(hidden_size)
        self.target_embed_std = float(target_embed_std)
        self.emb_dim = int(emb_dim)
        self.mlp_hidden = int(mlp_hidden)
        self.num_win_tokens = int(num_win_tokens)
        self.num_phase_tokens = int(num_phase_tokens)
        self.num_ref_tokens = int(num_ref_tokens)
        self.num_prefix_tokens = int(num_prefix_tokens)

        self.win_emb = nn.Embedding(self.num_states, emb_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(4 * emb_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
        )
        self.win1_head = nn.Sequential(
            nn.Linear(emb_dim + mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, num_win_tokens * mlp_hidden),
        )
        self.win2_head = nn.Sequential(
            nn.Linear(emb_dim + mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, num_win_tokens * mlp_hidden),
        )
        self.phase_mlp = nn.Sequential(
            nn.Linear(6, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, num_phase_tokens * mlp_hidden),
        )
        self.ref_mlp = nn.Sequential(
            nn.Linear(12, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, num_ref_tokens * mlp_hidden),
        )
        self.prefix_latents = nn.Parameter(torch.randn(num_prefix_tokens, mlp_hidden) * 0.02)
        self.out_proj = nn.Linear(mlp_hidden, hidden_size)
        self.out_norm = nn.LayerNorm(hidden_size)

        init_scale = max(0.05, 2.0 * float(target_embed_std))
        self.win_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        self.phase_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        self.ref_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        self.prefix_scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    def encode_windows(self, z1: torch.Tensor, z2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e1 = self.win_emb(z1)
        e2 = self.win_emb(z2)
        joint = self.pair_mlp(torch.cat([e1, e2, e1 - e2, e1 * e2], dim=-1))
        u1 = self.win1_head(torch.cat([e1, joint], dim=-1)).view(-1, self.num_win_tokens, self.mlp_hidden)
        u2 = self.win2_head(torch.cat([e2, joint], dim=-1)).view(-1, self.num_win_tokens, self.mlp_hidden)
        h1 = self.out_norm(self.out_proj(u1)) * self.win_scale
        h2 = self.out_norm(self.out_proj(u2)) * self.win_scale
        return h1, h2

    def encode_phase(self, phase_slots: torch.Tensor) -> torch.Tensor:
        up = self.phase_mlp(phase_slots).view(-1, self.num_phase_tokens, self.mlp_hidden)
        return self.out_norm(self.out_proj(up)) * self.phase_scale

    def encode_refs(self, ref_feats: torch.Tensor) -> torch.Tensor:
        ur = self.ref_mlp(ref_feats).view(-1, self.num_ref_tokens, self.mlp_hidden)
        return self.out_norm(self.out_proj(ur)) * self.ref_scale

    def encode_prefix(self, batch_size: int) -> torch.Tensor:
        p = self.out_norm(self.out_proj(self.prefix_latents)) * self.prefix_scale
        return p.unsqueeze(0).expand(batch_size, -1, -1)

    def connector_config(self) -> dict:
        return {
            "num_states": self.num_states,
            "hidden_size": self.hidden_size,


```

### Event #32: `item.completed` — `web_search` (`exec-6c3776f7-1fad-48c9-867e-98a0bec6daab`)

```json
{
  "id": "exec-6c3776f7-1fad-48c9-867e-98a0bec6daab",
  "type": "web_search",
  "query": "arxiv neural tangent kernel Jacot Gabriel Hongler neural tangent kernel convergence 2018 ...",
  "action": {
    "type": "search",
    "queries": [
      "arxiv neural tangent kernel Jacot Gabriel Hongler neural tangent kernel convergence 2018",
      "Lan DeMets discrete sequential boundaries for clinical trials alpha spending O Brien Fleming article"
    ]
  },
  "results": [
    {
      "type": "text_result",
      "domain": "arxiv.org",
      "ref_id": "turn3academia12",
      "snippet": "Title: Neural Tangent Kernel: Convergence and Generalization in Neural Networks Authors: Arthur Jacot, Franck Gabriel, Clément Hongler",
      "title": "Neural Tangent Kernel: Convergence and Generalization in Neural Networks",
      "url": "https://arxiv.org/abs/1806.07572"
    },
    {
      "type": "text_result",
      "domain": "papers.neurips.cc",
      "ref_id": "turn3search0",
      "snippet": "# Neural Tangent Kernel: Convergence and Generalization in Neural Networks Arthur Jacot, Franck Gabriel, Clement Hongler",
      "title": "Neural Tangent Kernel: Convergence and Generalization in Neural Networks",
      "url": "https://papers.neurips.cc/paper_files/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html"
    },
    {
      "type": "text_result",
      "domain": "arxiv.science",
      "ref_id": "turn3search1",
      "snippet": "Arthur Jacot·Franck Gabriel·Clément Hongler ... Neural Tangent Kernel: Convergence and Generalization in Neural Networks. https://arxiv.science/abs/1806.07572",
      "title": "Neural Tangent Kernel: Convergence and Generalization in Neural Networks",
      "url": "https://arxiv.science/abs/1806.07572"
    },
    {
      "type": "text_result",
      "domain": "papers.neurips.cc",
      "ref_id": "turn3search13",
      "snippet": "**Arthur Jacot** ... **Franck Gabriel** ... **Clément Hongler** ... We prove that the evolution of an ANN during training can also be described by a",
      "title": "Neural Tangent Kernel: Convergence and Generalization in Neural Networks",
      "url": "https://papers.neurips.cc/paper_files/paper/2018/file/5a4be1fa34e62bb8a6ec6b91d2462f5a-Paper.pdf"
    },
    {
      "type": "text_result",
      "domain": "mlanthology.org",
      "ref_id": "turn3search2",
      "snippet": "# Neural Tangent Kernel: Convergence and Generalization in Neural Networks Arthur Jacot, Franck Gabriel, Clement Hongler ... Neural Information Processing Systems, 2018.](https://mlanthology.org/neuri",
      "title": "Neural Tangent Kernel: Convergence and Generalization in Neural Networks",
      "url": "https://mlanthology.org/neurips/2018/jacot2018neurips-neural/"
    },
    {
      "type": "text_result",
      "domain": "pubmed.ncbi.nlm.nih.gov",
      "ref_id": "turn3search3",
      "snippet": "# Computations for group sequential boundaries using the Lan-DeMets spending function method ... We describe an interactive Fortran program which performs computations related to the",
      "title": "Computations for group sequential boundaries using the Lan-DeMets spending function method",
      "url": "https://pubmed.ncbi.nlm.nih.gov/10822118/"
    },
    {
      "type": "text_result",
      "domain": "eclass.uoa.gr",
      "ref_id": "turn3search14",
      "snippet": "DeMets ... This α\\*_3(t) represents a way of spending the error probability uniformly over time, and which is intermediate between the previous two functions α\\*_1(t)",
      "title": "Discrete Sequential Boundaries for Clinical Trials",
      "url": "https://eclass.uoa.gr/modules/document/file.php/MATH301/PracticalSession3/Lan_DeMets_1983.pdf"
    },
    {
      "type": "text_result",
      "domain": "huggingface.co",
      "ref_id": "turn3search4",
      "snippet": "[Button: Arthur Jacot] , [Button: Franck Gabriel] , Image Clément Hongler ... The training dynamics of artificial neural networks can be described by the Neural",
      "title": "Paper page - Neural Tangent Kernel: Convergence and Generalization in Neural Networks",
      "url": "https://huggingface.co/papers/1806.07572"
    },
    {
      "type": "text_result",
      "domain": "lechipatrick.github.io",
      "ref_id": "turn3search15",
      "snippet": "the group sequential boundaries be used? ... O’Brien-Fleming boundary. ... Based on the BHAT experience, Lan and DeMets developed a procedure referred to as the",
      "title": "STATISTICS IN MEDICINE, VOL. 13. 1341-1352 (1994)",
      "url": "https://lechipatrick.github.io/LanDeMetsPaper.pdf"
    },
    {
      "type": "text_result",
      "domain": "doi.org",
      "ref_id": "turn3search5",
      "snippet": "## Lan-DeMets Alpha-Spending Function ... The alpha spending function provides a flexible approach to control the Type I error while allowing the data monitoring committee",
      "title": "Lan-DeMets Alpha-Spending Function",
      "url": "https://doi.org/10.1002/9780471462422.eoct394"
    },
    {
      "type": "text_result",
      "domain": "arxiv.org",
      "ref_id": "turn3academia16",
      "snippet": "Title: Neural Tangent Kernel: A Survey ... A seminal work [Jacot et al., 2018] demonstrated that training a neural network under specific parameterization is equivalent",
      "title": "Neural Tangent Kernel: A Survey",
      "url": "https://arxiv.org/abs/2208.13614"
    },
    {
      "type": "text_result",
      "domain": "arxiv.org",
      "ref_id": "turn3academia17",
      "snippet": "Authors: Arthur Jacot, Franck Gabriel, Clément Hongler ... The dynamics of DNNs during gradient descent is described by the so-called Neural Tangent Kernel (NTK).",
      "title": "The asymptotic spectrum of the Hessian of DNN throughout training",
      "url": "https://arxiv.org/abs/1910.02875"
    },
    {
      "type": "text_result",
      "domain": "arxiv.org",
      "ref_id": "turn3academia18",
      "snippet": "Authors: Arthur Jacot, Franck Gabriel, François Ged, Clément Hongler ... We analyze architectural features of Deep Neural Networks (DNNs) using the so-called Neural Tangent Kernel",
      "title": "Order and Chaos: NTK views on DNN Normalization, Checkerboard and Boundary Artifacts",
      "url": "https://arxiv.org/abs/1907.05715"
    },
    {
      "type": "text_result",
      "domain": "www.nzdr.ru",
      "ref_id": "turn3search19",
      "snippet": "The boundary successively relaxes the criteria for ... general philosophy of many who have experience in clinical trial data monitoring. ... DeMets (1983, 1989a, 1989b)",
      "title": "Data Monitoring Committees in Clinical Trials: A Practical Perspective.",
      "url": "https://www.nzdr.ru/data/media/biblio/kolxoz/M/MV/MVsa/Ellenberg%20S.%2C%20T.R.Fleming%2C%20D.L.DeMets.%20Data%20Monitoring%20Committees%20in%20Clinical%20Trials..%20A%20Practical%20Perspective%20%28Wiley%2C2002%29%28ISBN%200471489867%29%28199s%29_MVsa_.pdf"
    },
    {
      "type": "text_result",
      "domain": "si.biostat.washington.edu",
      "ref_id": "turn3search20",
      "snippet": "Alpha Spending Functions Lan-DeMets Group Sequential Boundaries Calculations ... Spending Function: O'Brien-Fleming",
      "title": "Alpha Spending Functions",
      "url": "https://si.biostat.washington.edu/sites/default/files/modules/SISCER_Mod13_Day2.pdf"
    },
    {
      "type": "text_result",
      "domain": "pmc.ncbi.nlm.nih.gov",
      "ref_id": "turn3search6",
      "snippet": "Lan-DeMets with O’Brien-Fleming alpha spending^{‡} | 37 (52) ... Here we included the boundary for the more important endpoint OS, which used Lan-DeMets with O’Brien-Fleming",
      "title": "Randomization in clinical trials with small sample sizes using group sequential designs",
      "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12165385/"
    },
    {
      "type": "text_result",
      "domain": "www.sciencedirect.com",
      "ref_id": "turn3search7",
      "snippet": "Research article ... We describe an interactive Fortran program which performs computations related to the design and analysis of group sequential clinical trials using Lan-DeMets",
      "title": "Computations for Group Sequential Boundaries Using the Lan-DeMets Spending Function Method",
      "url": "https://www.sciencedirect.com/science/article/pii/S019724560000057X"
    },
    {
      "type": "text_result",
      "domain": "msrepo.github.io",
      "ref_id": "turn3search8",
      "snippet": "# Neural Tangent Kernel: Convergence and Generalization in Neural Networks Arthur Jacot, Franck Gabriel, Clément Hongler (EPFL) · NeurIPS · 2018 ... * arXiv:1806.07572 —",
      "title": "Neural Tangent Kernel: Convergence and Generalization in Neural Networks",
      "url": "https://msrepo.github.io/theory_inclined_papers_with_annotations/2018-jacot-neural-tangent-kernel/index.html"
    },
    {
      "type": "text_result",
      "domain": "pmc.ncbi.nlm.nih.gov",
      "ref_id": "turn3search9",
      "snippet": "Lan and DeMets specified an alpha spending function such that α(0) = 0 and α(1) = α. ... Alpha spending functions that approximate O’Brien-Fleming or",
      "title": "Flexible Stopping Boundaries When Changing Primary Endpoints after Unblinded Interim Analyses",
      "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4024106/"
    },
    {
      "type": "text_result",
      "domain": "www.youtube.com",
      "ref_id": "turn3youtube21",
      "snippet": "# Neural Tangent Kernel: Convergence and Generalization in Neural Networks By: Arthur Jacot (345 followers)",
      "title": "Neural Tangent Kernel: Convergence and Generalization in Neural Networks - YouTube",
      "url": "https://www.youtube.com/watch?v=raT2ECrvbag"
    },
    {
      "type": "text_result",
      "domain": "www.youtube.com",
      "ref_id": "turn3youtube23",
      "snippet": "Functional kernel analysis allows a nice analysis of neural networks in the infinite-width limit, as discussed by Arthur Jacot, PhD candidate in mathematics at EPFL.",
      "title": "Functional Kernel Analysis of Neural Networks (ft. Arthur Jacot) - YouTube",
      "url": "https://www.youtube.com/watch?v=DsTENxGcsxU"
    },
    {
      "type": "text_result",
      "domain": "github.com",
      "ref_id": "turn3view0",
      "snippet": "Total lines: 416",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/1EtBCXif2T8hddXbu7f8cTdTa-QDjXOOpBdP7-j4_pVr-wbjC34VrGQ_DBlc1aBDjbev0aJqn0HvyqAo20YHGuKPc343DBC4EBefQXddgkYGuxhECUIgoyNZdjiJYT3xF3yniGJSu8klyBoyOnPsbxc39ckbG0bXHK8wtEyacVGzncIBbE-BqlEExJ7kZHZb",
      "title": "EM-FloatFormer/research/realdata_pluto_qpsk/REPORT.md at ade8333fb68126251a83e484b69b821b5d97bcea · 20171806011/EM-FloatFormer · GitHub",
      "url": "https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/REPORT.md"
    },
    {
      "type": "text_result",
      "domain": "github.com",
      "ref_id": "turn3view1",
      "snippet": "Total lines: 402",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/Xn-DC5pT9Jup9bxN6v_afr5BjjfaQCNrQgXOrTFxq5Cs4qRMejRP9Bq-7FKvphGnGnIBUG_K-J4ZQIJoFTQpjnVA9GWC7B03AZRGRqQyKznWe_OSrJ_FBiHBQBoqr-v33YXl1e4X3AybscbbaWMjqel4Upr8nyUkN7tz8K2jA1Vyt0qlTqghtcDs4FSYjsHa",
      "title": "EM-FloatFormer/research/realdata_cttc_qpsk/REPORT.md at 1887aee6de9649bc0dd8a4f05fe3c4542f3f6d77 · 20171806011/EM-FloatFormer · GitHub",
      "url": "https://github.com/20171806011/EM-FloatFormer/blob/1887aee6de9649bc0dd8a4f05fe3c4542f3f6d77/research/realdata_cttc_qpsk/REPORT.md"
    }
  ]
}
```

### Event #33: `item.completed` — `command_execution` (`item_30`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=Join-Path (Get-Location) '"'research/stage4_math_and_design/M6/.uv-cache'; "'$env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$env:TEMP=$env:UV_CACHE_DIR; $env:TMP=$env:UV_CACHE_DIR; $env:PYTHONUTF8='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B -c \"import numpy as n,json; from pathlib import Path; d=Path('research/stage3d_qwen_signed_phase/results'); a=n.load(d/'blind_q3_scores.npz'); b=n.load(d/'blind_q2_scores.npz'); j=json.loads((d/'summary_master.json').read_text()); print({k:a[k].shape for k in a.files if k.startswith('H_c4_035') or k in ('temps_q3','seeds')}); print({k:b[k].shape for k in b.files}); print(j['historical_c4_035_lookup_anchor']); print(j['work_g_primary_040']['arms']['phase'].keys()); print(j['work_g_primary_040']['arms']['phase']['offline_group_avg_16way']); print(j['work_g_primary_040']['arms']['phase']['raw_rotation_checks']['rot11_roll']); print(j['work_a_q2_c4']['arms']['phase']['paired_contrasts']['duplicate_copy_raw_minus_valid']);\""
```

**Aggregated Output:**

```text
{'temps_q3': (2, 3), 'seeds': (3,), 'H_c4_035__raw_ll': (2, 3, 2, 256, 4), 'H_c4_035__codes': (2, 3, 1024), 'H_c4_035__targets': (2, 3, 1024), 'H_c4_035__indep_targets': (2, 3, 1024), 'H_c4_035__direct_table_probs': (2, 3, 1024, 4), 'H_c4_035__full_iq_probs': (2, 3, 1024, 4), 'H_c4_035__rot10_codes': (2, 1024), 'H_c4_035__rot10_targets': (2, 1024), 'H_c4_035__shuffled_perm': (2, 3, 1024), 'H_c4_035__cal_ref': (2, 256, 4), 'H_c4_035__cal_mass': (2, 256), 'H_c4_035__block_seeds': (16,)}
{'q2__raw_ll': (2, 3, 7, 512, 4), 'temps_q2': (2, 3), 'q2__labels': (512,), 'q2__z1': (512,), 'q2__z2': (512,), 'q2__work_a_ref_probs': (6, 512, 4), 'q2__matched_valid1': (512,), 'q2__matched_valid2': (512,), 'q2__matched_indep': (512,)}
{'source': 'research/stage3b_math/A/c4_s17_lookup.json', 'h_c4_pos_acc': 0.6318359375, 'h_c4_pos_nll': 0.9650571187776844, 'work_f_check_f_py_035': {'phase': {'orientation_gain': 0.011111701098819947, 'residual_ratio': 0.036864847496508715, 'mean_sign_TV': 0.026730785475160612, 'mean_rotation_TV': 0.08525621770297791, 'reference_odd_norm_squared': 0.7706897692346777, 'matched_accuracy': 0.223876953125, 'swapped_accuracy': 0.2100830078125, 'swapped_minus_matched_nll': [0.018112136041199003, 0.016661060106803696, 0.019498864925645894], 'factorial_mean_squared_norm': {'metadata': 0.00021902479410414665, 'wave': 0.0010807147055390673, 'interaction': 6.273064625576237e-05}}, 'zero': {'orientation_gain': 0.005619790253737205, 'residual_ratio': 0.026739340817530592, 'mean_sign_TV': 0.01921407453500916, 'mean_rotation_TV': 0.09909726683071512, 'reference_odd_norm_squared': 0.7706897692346777, 'matched_accuracy': 0.2158203125, 'swapped_accuracy': 0.216064453125, 'swapped_minus_matched_nll': [0.00934451514945835, 0.008375374707456773, 0.010342039645440837], 'factorial_mean_squared_norm': {'metadata': 0.00010600420324573385, 'wave': 0.0010311596081371483, 'interaction': 3.9866612445473744e-05}}}}
dict_keys(['raw_rotation_checks', 'controls', 'raw_host', 'offline_group_avg_16way', 'offline_representative_F0'])
{'acc': 0.17050333969465647, 'nll': 1.4113123906537837, 'pf_minus_raw_acc_t95': {'n_blocks': 131, 'mean': -0.0674830364715861, 'block_sd': 0.01981303079493022, 'se': 0.0017310725380280602, 't_crit_95': 1.9783804054470222, 'lo95': -0.07090775646122827, 'hi95': -0.06405831648194393}, 'raw_minus_pf_nll_t95': {'n_blocks': 131, 'mean': 0.0339161630052869, 'block_sd': 0.008374058906737819, 'se': 0.0007316449237535328, 't_crit_95': 1.9783804054470222, 'lo95': 0.03246869102438813, 'hi95': 0.03536363498618567}}
{'mean_max_abs': 0.12627380352018333, 'worst_max_abs': 0.3800186428066911, 'frac_gt_0p20': 0.1544409987277354}
{'delta_acc': -0.024739583333333332, 'delta_acc_mother_se': 0.012072413489438542, 'delta_nll': 0.005536921665656535, 'delta_nll_mother_se': 0.003959084370487028, 'mean_tv': 0.0403772934643645, 'max_tv': 0.39604302227783905, 'max_abs_prob_diff': 0.38122767425675297, 'max_conf_increase': 0.24702782916403143, 'corrected': 160, 'new_errors': 198}
```

### Event #34: `item.completed` — `file_change` (`item_31`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M6\\verify_m6.py",
    "kind": "add"
  }
]
```

### Event #35: `item.completed` — `file_change` (`item_32`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M6\\verify_m6.py",
    "kind": "update"
  }
]
```

### Event #36: `item.completed` — `file_change` (`item_33`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M6\\verify_m6.py",
    "kind": "update"
  }
]
```

### Event #37: `item.completed` — `command_execution` (`item_34`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=Join-Path (Get-Location) '"'research/stage4_math_and_design/M6/.uv-cache'; "'$env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$env:TEMP=$env:UV_CACHE_DIR; $env:TMP=$env:UV_CACHE_DIR; $env:PYTHONUTF8='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M6/verify_m6.py"
```

**Aggregated Output:**

```text
VERIFIED results_m6.json: parity max |diff|=3.56e-16, 10,000 trials, cap $9.50
```

### Event #38: `item.completed` — `web_search` (`exec-06951f7d-11d7-4093-a460-ff040e036e73`)

```json
{
  "id": "exec-06951f7d-11d7-4093-a460-ff040e036e73",
  "type": "web_search",
  "query": "'2240'",
  "action": {
    "type": "find_in_page",
    "pattern": "2240"
  },
  "results": [
    {
      "type": "text_result",
      "domain": "github.com",
      "ref_id": "turn4view0",
      "snippet": "Total lines: 416",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/s0qiZjXB8pPaFg0sgUBUXq22K6EMQOB6aBDfUGiD6N2ayBtvlQjAb9jShIaDVVCwYJfiOTsqQo9sQORjtQW6YzQEhZmhIhG4pBCsCXOoXbBf_WuSnurTYcd8S6WOV3RWZyOIbQhf3NaU3K0rBBpHfroFVlnvditDnDayyjNRPj2cK1SBYaPe18HAPtwjZUid",
      "title": "EM-FloatFormer/research/realdata_pluto_qpsk/REPORT.md at ade8333fb68126251a83e484b69b821b5d97bcea · 20171806011/EM-FloatFormer · GitHub",
      "url": "https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/REPORT.md"
    },
    {
      "type": "text_result",
      "domain": "github.com",
      "ref_id": "turn4view1",
      "snippet": "Total lines: 402",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/s_rVdx50qKNM922MP0HibqVZRIYQG9CwoGm1biyo-lxrVmn-zvAKEDOdiL2lcIgexE-BOalG1TZNiWPgPlcbAZ_-XybPSQibe6YY65u49Y-zbEN3AlWJVVH-1qxLurEEwOhTk_DnZKYIAoRjVt5Ou_hgJXKjuJHUujTz-noYTEijDjADElP3taNYYIqc_fMJ",
      "title": "EM-FloatFormer/research/realdata_cttc_qpsk/REPORT.md at 1887aee6de9649bc0dd8a4f05fe3c4542f3f6d77 · 20171806011/EM-FloatFormer · GitHub",
      "url": "https://github.com/20171806011/EM-FloatFormer/blob/1887aee6de9649bc0dd8a4f05fe3c4542f3f6d77/research/realdata_cttc_qpsk/REPORT.md"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn4view2",
      "snippet": "Total lines: 51",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/qqzBchFdyB7byVyqSWX0FGFKi94Umyr6sN64XyeAokAjtlsMYTNNT0cDA1qvydKwzsOwMHIVT8NyfCK-RJJdsZc8hUil9UobV37HJm0_scet8fqnRmMQ5ojOQLIOwUF2",
      "title": "Budgets | Modal Docs",
      "url": "https://modal.com/docs/guide/budgets"
    },
    {
      "type": "text_result",
      "domain": "arxiv.org",
      "ref_id": "turn4view3",
      "snippet": "Total lines: 551",
      "title": "LEMMO: A Large Electromagnetic Model leveraging Pretrained LLMs for Million-Sample Signal Understanding",
      "url": "https://arxiv.org/html/2611.09476v1"
    }
  ]
}
```

### Event #39: `item.completed` — `command_execution` (`item_36`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$q=Get-Content research/stage3d_qwen_signed_phase/qwen_connector.py; $q[431..528]; $q=Get-Content research/stage3d_qwen_signed_phase/REPORT.md; rg -n '"'8,384|0\\.929|0\\.941|LEMMO|15:|UTC|Qwen3\\.5' research/stage3d_qwen_signed_phase/REPORT.md | Select-Object -First 20; rg --files research | Select-String -Pattern '(full_iq|q2_reader|stage3[A-Za-z]*[\\\\/]q3)' | Select-Object -First 15"
```

**Aggregated Output:**

```text
    def build_prefix_embeds(
        self,
        connector: Optional[SignedPhaseCacheConnector],
        user_prompt: str,
        z1_batch: Optional[torch.Tensor],
        z2_batch: Optional[torch.Tensor],
        phase_slots_batch: Optional[torch.Tensor] = None,
        ref_feats_list: Sequence[torch.Tensor] = (),
        text_only_mode: bool = False,
        extra_history: Optional[List[dict]] = None,
    ) -> torch.Tensor:
        """Assemble `(B, L, D)` prefix embeddings with zero leakage."""
        parts = self.format_chat_parts(user_prompt, extra_history=extra_history)
        embed_layer = self.lm.get_input_embeddings()
        batch_size = int(z1_batch.shape[0]) if z1_batch is not None else 1

        if text_only_mode or connector is None:
            e_pref = torch.zeros((batch_size, 0, self.hidden_size), device=self.device, dtype=self.dtype)
            e_w1 = torch.zeros((batch_size, 0, self.hidden_size), device=self.device, dtype=self.dtype)
            e_w2 = torch.zeros((batch_size, 0, self.hidden_size), device=self.device, dtype=self.dtype)
            e_phase = torch.zeros((batch_size, 0, self.hidden_size), device=self.device, dtype=self.dtype)
            e_refs = [
                torch.zeros((batch_size, 0, self.hidden_size), device=self.device, dtype=self.dtype)
                for _ in ref_feats_list
            ]
        else:
            e_pref = connector.encode_prefix(batch_size).to(self.dtype)
            w1, w2 = connector.encode_windows(z1_batch.to(self.device), z2_batch.to(self.device))
            e_w1 = w1.to(self.dtype)
            e_w2 = w2.to(self.dtype)
            if phase_slots_batch is None:
                ps = torch.zeros((batch_size, 6), device=self.device, dtype=torch.float32)
            else:
                ps = phase_slots_batch.to(self.device, dtype=torch.float32)
            e_phase = connector.encode_phase(ps).to(self.dtype)
            e_refs = [connector.encode_refs(rf.to(self.device, dtype=torch.float32)).to(self.dtype) for rf in ref_feats_list]

        chunks: List[torch.Tensor] = []
        ref_ptr = 0
        prefix_inserted = False
        for seg in parts:
            if seg == "<W1>":
                if not text_only_mode:
                    if not prefix_inserted:
                        chunks.append(e_pref)
                        prefix_inserted = True
                    chunks.append(e_w1)
            elif seg == "<W2>":
                if not text_only_mode:
                    chunks.append(e_w2)
            elif seg == "<PHASE>":
                if not text_only_mode:
                    chunks.append(e_phase)
            elif seg == "<REF>":
                if ref_ptr >= len(e_refs):
                    raise ValueError("More <REF> markers in prompt than ref_feats_list entries")
                if not text_only_mode:
                    chunks.append(e_refs[ref_ptr])
                ref_ptr += 1
            else:
                ids = self.tokenizer.encode(seg, add_special_tokens=False)
                if ids:
                    t_ids = torch.tensor(ids, device=self.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
                    chunks.append(embed_layer(t_ids))
        if not text_only_mode and ref_ptr != len(e_refs):
            raise ValueError(f"Unused ref_feats_list entries: used {ref_ptr} of {len(e_refs)}")
        return torch.cat(chunks, dim=1)

    def _lm_forward(self, **kwargs):
        if self._uses_cache_kwarg:
            if "past_key_values" in kwargs:
                kwargs["cache"] = kwargs.pop("past_key_values")
        else:
            if "cache" in kwargs:
                kwargs["past_key_values"] = kwargs.pop("cache")
        return self.lm(**kwargs)

    def score_candidates_naive_uncached(
        self,
        prefix_embeds: torch.Tensor,
        choices: Tuple[str, ...],
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reference uncached per-candidate forward loop (zero padding, zero cache)."""
        assert len(choices) == 4
        b, pref_l, _ = prefix_embeds.shape
        embed_layer = self.lm.get_input_embeddings()
        raw_ll = []
        for ch in choices:
            ids = self.tokenizer.encode(ch, add_special_tokens=False)
            c_ids = torch.tensor(ids, device=self.device, dtype=torch.long).unsqueeze(0).expand(b, -1)
            c_emb = embed_layer(c_ids)
            full_emb = torch.cat([prefix_embeds, c_emb], dim=1)
            seq_len = full_emb.shape[1]
            attn = torch.ones((b, seq_len), device=self.device, dtype=torch.long)
            pos = self._make_position_ids(torch.arange(seq_len, device=self.device, dtype=torch.long), b)
            fwd_kwargs = dict(inputs_embeds=full_emb, attention_mask=attn, position_ids=pos, use_cache=False)
            logits = self._lm_forward(**fwd_kwargs).logits.float()
1:# EM-FloatFormer Stage 3D 实验报告：冻�?Qwen3.5-4B �?s17 缓存下的有符号频偏显式相位向量盲�?
10:本轮实验回答单一主问题：**在完全冻�?`Qwen/Qwen3.5-4B` �?`equivariant_b8_s17` 离散波形编码器、两臂均从中文题面获得完全相同的有符号频偏元数据且训练数据与更新预算逐项相同的条件下，仅将连接层�?6 个公开数值槽由全零（`zero` 对照臂）替换为对应时�?$t \in \{h, h+\ell, t_p\}$ 的显式相位向�?$[\cos(\omega t), \sin(\omega t)]$（`phase` 相位臂），能否提高对未见频偏�?Q3-repeat 相位扇区预测�?*
16:| **相位�?`phase`（三套种子等权汇总）** | **23.80%** (`0.23799`) | **1.4452** (`1.44523`) | `0.7845` | `0.1186` | `131` 独立区组 × `64` 母波�?= `8,384` 条（`150,912` 逻辑行）；`-ω: 24.95% / 1.4547`，`+ω: 22.65% / 1.4358` |
17:| **对照�?`zero`（三套种子等权汇总）** | **19.41%** (`0.19406`) | **1.4809** (`1.48088`) | `0.8010` | `0.1827` | `131` 独立区组 × `64` 母波�?= `8,384` 条（`150,912` 逻辑行）；`-ω: 18.80% / 1.4678`，`+ω: 20.02% / 1.4939` |
20:| 同期同母波形 `s17` CPU 物理读出 | **64.62%** (`0.64617`) | **0.9411** (`0.94113`) | `0.5098` | `0.0009` | 真实连接层相�?CPU 物理读出差距：`phase �?0.82 pp / -0.5041 nats`；`zero �?5.21 pp / -0.5398 nats` |
21:| 同期同母波形 `full_iq` 已知参数全波形解 | **64.64%** (`0.64641`) | **0.9290** (`0.92904`) | `0.5032` | `0.0046` | 物理上限基线 |
51:| **语言宿主模型** | `Qwen/Qwen3.5-4B`（HF pinned revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，已验证丢弃未用视觉编码�?`lm.model.visual = None` 并保�?`tie_word_embeddings`�?|
68:### 1.2 工程关键排查：Qwen3.5 混合线性注意力引用环与 BF16 批并行数值一致�?
118:  - 调用 Stage 3C `research/stage3c/work_results/E/verify.py --pilot-block-sd 0.02104014 0.01928086`，得 `t_blocks = 131`（效能：准确�?`0.9505`、NLL `0.9999`，半�?`±0.01632` / `±0.01379`），由此在揭盲前锁定 **`B_locked = max(131, 131) = 131` 个区组（`8,384` 条母波形�?*，并生成 `blind_seed_manifest.json`（SHA-256：`b6c075162010a0d6e72bb07cd32a4a01aed24e1abca8be2f949846d258c0f7e7`），连同完整代码�?`PROTOCOL_LOCK.json` 先提交至 `a1fac7a8` 后再启动盲测�?
128:| **`G1_main_extrap_both`** | **主检验（双外推）** | `0.40` | `(2, 2, 4)` | **`1.60 rad`** | `0.04` | `131 / 8,384` | `23.80% / 1.4452` | `19.41% / 1.4809` | **`+4.39 pp`** `[+4.11, +4.68]` (`[+4.11, +4.67]`) | **`+0.0357`** `[+0.0330, +0.0383]` (`[+0.0330, +0.0383]`) | `64.62% / 0.9411` (`64.64% / 0.9290`) |
147:| **`±0.40` 物理校准参考（131 区块�?* | `1.0000` | `0.0000` | `0.6943` | `0.0000` | `64.62% / 0.9413` | `3.63% / 2.9964` | **`+2.0551`** `[+2.0252, +2.0850]` | `(0.2016, 2.3e-5, 2.8e-5)` |
232:- **已全部完成并核验**：Qwen3.5-4B 完整 Smoke �?Tokenizer 审计、显存与引用环消除基准测试�? 种子 × 2 臂（6 次）配对训练与开发集温度校准、`dev_pilot`（`B_locked=131`）与提交前锁定（`Commit #1`）、全量配对盲测（`G1` + `G2..G4` + 历史 `±0.35`）、Work F（含单模态矛盾探针与 `check_f.py`）、Work G（含离线 `Pf/F0` 严格分离）、Work A Q2 七臂�?`q2_reader.read` 对照、`C1/C2/C3` 遗忘回放、`MANIFEST.sha256` 及干净目录零误差全量重算验证（`max_abs_diff = 0.000e+00`）�?
233:- **未完成项（按严格指纹锁定协议主动中止�?*：计划末段可选的历史 `mistralai/Ministral-3-3B-Instruct-2512-BF16` 旧检查点复验（`modal_runner.py --mode eval_ministral`, Modal App `ap-ILaRaUzoj3xPA12N7F3Dsc`）已完成锁定 Hugging Face revision `b6d637bef2393152b3da2b2fde72eecdee30557e` 下载及权重读取，但在执行 `assert host.host_sha256 == lock["host_sha256"]` 时按协议主动终止报错——因 Qwen3.5-4B 所需的容器环境（`transformers 5.24.1 + torch 2.15.0+cu132`）与 Stage 3 训练历史 Ministral 时锁定的环境（`transformers 5.22.1 + torch 2.14.1+cu130`）在采样指纹逻辑上不一致（预期参数名称/采样格式跨版变更）。为严守不放松任何既有阶�?SHA-256 指纹锁、且不修�?`research/stage3_ministral_cache/` 既有代码的原则，未强制跳过指纹校验运行，故历�?Ministral 中文重放本轮列为“受运行库版本隔离限制而中止的未完成可选诊断”�?
239:2. **冻结组件版本**：`Qwen/Qwen3.5-4B`（`851bf6e8...`）、`equivariant_b8_s17`（`e67d9fba...`）与本轮 6 组连接层（每组仅 `2,184,964` 可训练参数）�?

research\stage3c\A\q2_reader.py
research\stage3_ministral_cache\results\direct\pre_reveal_jsonl\C2_discrete_walk__full_iq_known_param.jsonl
research\stage3_ministral_cache\results\direct\pre_reveal_jsonl\C1_coherent_2db__full_iq_known_param.jsonl
research\stage3_ministral_cache\results\direct\pre_reveal_jsonl\C3_mid_pilot_cont__full_iq_known_param.jsonl
research\stage3_ministral_cache\results\direct\pre_reveal_jsonl\C4_cfo_ramp__full_iq_known_param.jsonl
research\stage3_ministral_cache\results\direct\pre_reveal_jsonl\C5_snr_0db__full_iq_known_param.jsonl
research\stage3_ministral_cache\results\direct\pre_reveal_jsonl\C6_decoherent_order__full_iq_known_param.jsonl
research\stage3_ministral_cache\results\direct\pre_reveal_jsonl\C5_snr_6db__full_iq_known_param.jsonl


```

### Event #40: `item.completed` — `command_execution` (`item_37`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=Join-Path (Get-Location) '"'research/stage4_math_and_design/M6/.uv-cache'; "'$env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$env:TEMP=$env:UV_CACHE_DIR; $env:TMP=$env:UV_CACHE_DIR; $env:PYTHONUTF8='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B -c \"import json; from pathlib import Path; d=json.loads(Path('research/stage4_math_and_design/M6/results_m6.json').read_text()); f=lambda r: '%6.3f%%/%0.6f'%(r['acc']*100,r['nll']); print('Q3',*[ (k, f(v['full_iq']),f(v['direct_s17']),*[f(a['locked_temperature']) for a in v['host'].values()],v['direct_minus_full_iq_nll'],*[[(a['locked_minus_direct_nll']['mean']),a['locked_minus_unit_nll']] for a in v['host'].values()]) for k,v in d['stage3d']['q3'].items()],sep='\\n'); print('Q2',*[(k,f(v['work_a']),*[f(a['locked_temperature']) for a in v['host'].values()],*[a['locked_minus_work_a_nll'] for a in v['host'].values()]) for k,v in d['stage3d']['q2'].items()],sep='\\n'); print(d['stage3d']['q3']['G1_main_extrap_both']['host']['phase']); print(d['cpu_geometry_toy']); print(d['sequential']); print(d['modal_budget']); print(d['summary_parity'])\""
```

**Aggregated Output:**

```text
Q3
('G1_main_extrap_both', '64.641%/0.929038', '64.617%/0.941126', '23.799%/1.445229', '19.406%/1.480879', {'blocks': 131, 'mean': 0.012087838263628765, 'lo95': 0.009178147717718608, 'hi95': 0.014997528809538922}, [0.5041027288195585, 0.026076988766369578], [0.5397527313702656, 0.03445587589034411])
('G2_interp_both', '46.777%/1.019346', '46.777%/1.022230', '33.876%/1.335826', '36.361%/1.308067', {'blocks': 8, 'mean': 0.002884441298059031, 'lo95': -0.011954505600101459, 'hi95': 0.017723388196219523}, [0.3135963372334519, -0.00026019543089104177], [0.28583672387186004, -0.0014345781295616899])
('G3_extrap_angle', '61.914%/0.953916', '62.109%/0.979568', '22.678%/1.441047', '19.737%/1.478130', {'blocks': 8, 'mean': 0.025651842097815496, 'lo95': 0.013000085857675038, 'hi95': 0.038303598337955956}, [0.46147969569011, 0.025007061718139077], [0.49856263078327356, 0.03417262325739176])
('G4_extrap_freq', '49.219%/1.003738', '47.754%/1.015851', '33.290%/1.334667', '34.961%/1.309751', {'blocks': 8, 'mean': 0.012113078347057596, 'lo95': 0.0020517570011747057, 'hi95': 0.022174399692940487}, [0.31881608484574786, -0.0012003103501013665], [0.2938995372083225, -0.0021761284824652486])
('H_c4_035', '63.867%/0.942628', '63.770%/0.950133', '25.765%/1.416987', '23.172%/1.435077', {'blocks': 16, 'mean': 0.007504578101671303, 'lo95': -0.0019952883893824096, 'hi95': 0.017004444592725015}, [0.4668543874200796, 0.018967296172071452], [0.48494420350574424, 0.024397492035357658])
Q2
('none', '25.000%/1.386294', '24.805%/1.392846', '25.781%/1.393822', 0.006552121973095071, 0.007527957220960113)
('valid', '66.406%/0.819207', '28.776%/1.388883', '26.628%/1.410690', 0.5696761385749471, 0.5914834874648531)
('duplicate_copy_dedup', '66.406%/0.819207', '28.776%/1.388883', '26.628%/1.410690', 0.5696761385749471, 0.5914834874648531)
('duplicate_copy_raw', '66.406%/0.819207', '26.302%/1.394420', '25.781%/1.413341', 0.575213060240604, 0.5941346517860331)
('independent_only', '25.000%/1.386294', '24.805%/1.407714', '24.935%/1.420522', 0.021419804290827127, 0.034227248649580044)
('valid_plus_independent', '66.406%/0.819207', '24.544%/1.408232', '27.279%/1.418281', 0.589024849990974, 0.5990743018096454)
('fresh_resample', '74.219%/0.663189', '26.432%/1.405640', '25.781%/1.406397', 0.7424508946065895, 0.743207524172911)
{'unit_temperature': {'acc': 0.23798637616624257, 'nll': 1.4191515648927011, 'n_prob_clipped': 0}, 'locked_temperature': {'acc': 0.23798637616624257, 'nll': 1.4452285536590708, 'n_prob_clipped': 0}, 'unit_temperature_minus_direct_nll': 0.4780257400531889, 'locked_minus_unit_nll': 0.026076988766369578, 'locked_minus_direct_nll': {'blocks': 131, 'mean': 0.5041027288195585, 'lo95': 0.4912937167744849, 'hi95': 0.5169117408646322}, 'locked_minus_full_iq_nll': 0.5161905670831872, 'rot11_defect': {'mean_max_abs': 0.1262738035201833, 'worst_max_abs': 0.3800186428066912, 'frac_gt_0p20': 0.15444099872773537}}
{'additive': {'design_rank': 5, 'design_condition': 11.972002711051156, 'train_rmse': 0.38717931418692875, 'ood_rmse': 0.9351606645931971}, 'coupled': {'design_rank': 9, 'design_condition': 16.930968602736005, 'train_rmse': 3.314766461000073e-16, 'ood_rmse': 7.380652118704114e-16}, 'ood_rectangular_contrast_l2': 3.9982944121660204, 'elapsed_seconds': 0.009858899982646108}
{'rng_seed': 20261126, 'trials': 10000, 'alpha': 0.05, 'information_fraction': 0.18, 'prior_variance_standardized': 4.0, 'predictive_futility_threshold': 0.1, 'alpha_first_look': 5.2885482378994705e-05, 'early_z': 3.8769571789222472, 'final_z': 1.6450881493695912, 'futility_z': -0.2704380422162607, 'least_favorable_type_I_no_cpu_nonbinding': 0.049999999999999996, 'least_favorable_type_I_no_cpu_binding': 0.04469497263214194, 'min_working_drift_for_90_percent_power_at_cpu_sensitivity_099': 3.2676570871466035, 'pilot_cost_fraction': 0.085, 'pilot_usd_assumed': 1.4552000000000003, 'fixed_usd': 17.12, 'sequential_mean_usd_mc': 6.10124272, 'sequential_mean_usd_exact': 6.107866319813384, 'cost_reduction_pct_mc': 64.36189999999999, 'cost_reduction_pct_exact': 64.3232107487536, 'null_mixture_false_efficacy_seq_mc': 0.008, 'null_mixture_false_efficacy_fixed_mc': 0.0176, 'null_mixture_false_efficacy_seq_exact': 0.008559694861365055, 'null_mixture_false_efficacy_fixed_exact': 0.01579643762342445, 'strata': {'structured_broken': {'trials': 6000, 'drift': -0.8, 'cpu_pass_probability_assumed': 0.1, 'cpu_pass_count': 653, 'early_efficacy_count': 0, 'futility_count': 348, 'full_funded_count': 305, 'seq_efficacy_count': 6, 'seq_rate_mc': 0.001, 'seq_rate_exact_binomial_ci95_mc': [0.00036706798226798695, 0.002175298757399117], 'seq_rate_quadrature': 0.0006432497344743829, 'fixed_efficacy_count': 62, 'fixed_rate_mc': 0.010333333333333333, 'fixed_rate_exact_normal': 0.0072455470292805725, 'seq_mean_usd_mc': 0.9546682666666668, 'seq_mean_usd_exact': 0.8856903700812432}, 'boundary_null': {'trials': 1500, 'drift': 0.0, 'cpu_pass_probability_assumed': 0.9, 'cpu_pass_count': 1337, 'early_efficacy_count': 0, 'futility_count': 545, 'full_funded_count': 792, 'seq_efficacy_count': 54, 'seq_rate_mc': 0.036, 'seq_rate_exact_binomial_ci95_mc': [0.02715777188503193, 0.04671318686351153], 'seq_rate_quadrature': 0.040225475368927746, 'fixed_efficacy_count': 70, 'fixed_rate_mc': 0.04666666666666667, 'fixed_rate_exact_normal': 0.049999999999999975, 'seq_mean_usd_mc': 9.568082666666667, 'seq_mean_usd_exact': 9.861556831711777}, 'working_coupled': {'trials': 2500, 'drift': 3.6, 'cpu_pass_probability_assumed': 0.99, 'cpu_pass_count': 2476, 'early_efficacy_count': 22, 'futility_count': 93, 'full_funded_count': 2383, 'seq_efficacy_count': 2332, 'seq_rate_mc': 0.9328, 'seq_rate_exact_binomial_ci95_mc': [0.9222689784001638, 0.9423027132249898], 'seq_rate_quadrature': 0.9339508947762252, 'fixed_efficacy_count': 2438, 'fixed_rate_mc': 0.9752, 'fixed_rate_exact_normal': 0.974717101943164, 'seq_mean_usd_mc': 16.372917440000002, 'seq_mean_usd_exact': 16.38887429203149}}}
{'pricing_url': 'https://modal.com/pricing', 'resource_ceiling_physical_cores': 2, 'resource_ceiling_gib': 16, 'rows': [{'task': 'CPU dataset/weight staging, causal frontends and split audits', 'device': 'CPU', 'max_billed_seconds': 7200, 'max_usd_per_second': '0.00006172', 'max_usd': '0.44438400'}, {'task': 'Synthetic Qwen parity and timing receipt', 'device': 'A100-80GB', 'max_billed_seconds': 450, 'max_usd_per_second': '0.00075572', 'max_usd': '0.34007400'}, {'task': 'Synthetic pilot, seed 1, at most 173 steps', 'device': 'A100-80GB', 'max_billed_seconds': 650, 'max_usd_per_second': '0.00075572', 'max_usd': '0.49121800'}, {'task': 'Synthetic continuation, at most 2707 steps across 3 seeds', 'device': 'A100-80GB', 'max_billed_seconds': 6750, 'max_usd_per_second': '0.00075572', 'max_usd': '5.10111000'}, {'task': 'Synthetic blind G1-G4 and Q2_C4 KV scoring', 'device': 'A100-80GB', 'max_billed_seconds': 1050, 'max_usd_per_second': '0.00075572', 'max_usd': '0.79350600'}, {'task': 'Pluto, 2240 frames / 16 blocks, forward only', 'device': 'L4', 'max_billed_seconds': 2300, 'max_usd_per_second': '0.00028372', 'max_usd': '0.65255600'}, {'task': 'CTTC, 1440 packets / 72 blocks / 9 sessions, forward only', 'device': 'L4', 'max_billed_seconds': 1700, 'max_usd_per_second': '0.00028372', 'max_usd': '0.48232400'}], 'base_usd': '8.30517200', 'a10g_substitution_increment_usd': '0.336000', 'additional_reserve_usd': '0.85882800', 'cap_usd': '9.50', 'historical_credit_remainder_usd': '12.88', 'buffer_usd_at_cap': '3.38', 'spend_this_round_usd': '0.00'}
{'max_abs_acc_nll_diff': 3.5561831257524545e-16, 'scalar_acc_nll_checks': 102, 'h_c4_035_scope': 'Work F identity rotation and both signs. The all-rotation ladder is a fresh replay.'}
```

### Event #41: `item.completed` — `web_search` (`exec-fd719aa4-3577-4ebe-867f-85866abf5180`)

```json
{
  "id": "exec-fd719aa4-3577-4ebe-867f-85866abf5180",
  "type": "web_search",
  "query": "https://raw.githubusercontent.com/20171806011/EM-FloatFormer/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/REPORT.md",
  "action": {
    "type": "open_page",
    "url": "https://raw.githubusercontent.com/20171806011/EM-FloatFormer/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/REPORT.md"
  },
  "results": [
    {
      "type": "text_result",
      "domain": "raw.githubusercontent.com",
      "ref_id": "turn5view0",
      "snippet": "Total lines: 252",
      "url": "https://raw.githubusercontent.com/20171806011/EM-FloatFormer/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/REPORT.md"
    },
    {
      "type": "text_result",
      "domain": "raw.githubusercontent.com",
      "ref_id": "turn5view1",
      "snippet": "Total lines: 232",
      "url": "https://raw.githubusercontent.com/20171806011/EM-FloatFormer/1887aee6de9649bc0dd8a4f05fe3c4542f3f6d77/research/realdata_cttc_qpsk/REPORT.md"
    },
    {
      "type": "text_result",
      "domain": "github.com",
      "ref_id": "turn5view2",
      "snippet": "Total lines: 281",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/aWrprVOonjfsbdQXiZFTcL_gKRPlyrvveLLznLgZNGbLo5NhKuBhi1GDiyGRuZM5-ShgKjlARJk7MN-2J6pD74VCidMVnIEhLW-lfGLLGf2hoxAJ9J5TT18wN-_9pM0HI36iIavZpvCvnyB432vhTdNesB3t-9O3USEjW4N6oq6Gyy_3tIF8sVIaEwxyGCgm",
      "title": "GitHub - N1rv4n4/LEMMO: Implementation of LEMMO, an electromagnetic large model for raw I/Q signal recognition and description · GitHub",
      "url": "https://github.com/N1rv4n4/LEMMO"
    }
  ]
}
```

### Event #42: `item.completed` — `command_execution` (`item_39`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=Join-Path (Get-Location) '"'research/stage4_math_and_design/M6/.uv-cache'; "'$env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$env:TEMP=$env:UV_CACHE_DIR; $env:TMP=$env:UV_CACHE_DIR; $env:PYTHONUTF8='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B -c \"from pathlib import Path; r=Path('research/stage3d_qwen_signed_phase/REPORT.md').read_text(encoding='utf-8').splitlines(); print('\\n'.join(r[77:88]+r[98:113]+r[177:194]+r[195:218])); d=Path('docs/MULTIMODAL_INTEGRATION_PLAN.md').read_text(encoding='utf-8').splitlines(); print('\\n'.join(d[26:37]));\""
```

**Aggregated Output:**

```text
### 1.3 Modal 免费额度算力账单（全程零付费）

| 费用项目 | Modal 计费账单记录 | 说明 |
|---|---:|---|
| 临时计算作业 (`Ephemeral Apps`) | **`$17.12`** | 覆盖 Smoke、Bench（含早先 1 次 OOM）、6 次完整训练、Dev Pilot、全量盲测以及 1 次提前中止的历史 Ministral 检查 |
| 持久化卷存储 (`Volumes`: `qwen3-model-cache` + `stage3d-results`) | **`$1.34`** | Free Storage 全额抵扣 |
| 网络流出 (`Network Egress`) | **`$0.01`** | Network Egress Allowance 全额抵扣 |
| **实际账单应付 (`Billed Cost`)** | **`$0.00`** | **完全在 `$30.00` Modal 免费计算额度内完成（尚余约 `$12.88` 免费计算额度）** |

---


### 2.2 6 次连接层训练收敛与开发集标量温度锁定（[training_summary.json](results/training_summary.json)）

| 臂 / 种子 | 最优 Epoch | 开发筛选分 (`sel_score`) | 校准前 `dev_cfo_q3_nll` | 校准后 `dev_q3_cal_nll` (`T_q3`) | 校准后 `dev_q2_cal_nll` (`T_q2`) | 耗时 (s) / 峰值 VRAM | Checkpoint SHA-256（前 16 位） |
|---|---:|---:|---:|---:|---:|---:|---|
| `phase / 202611` | `7` | `1.38097` | `1.37762` (`29.62%`) | `1.37535` (`0.5099`) | `1.38664` (`4.0000`) | `2884.0 s` / `52,972.5 MB` | `088d4ad34e1152ff` |
| `phase / 202622` | `8` | `1.33823` | `1.31390` (`35.70%`) | `1.28527` (`0.8489`) | `1.35653` (`1.1069`) | `3026.2 s` / `52,972.5 MB` | `ba275787201abf5f` |
| `phase / 202633` | `8` | `1.34566` | `1.32520` (`35.00%`) | `1.30019` (`0.6575`) | `1.37150` (`1.0038`) | `3217.8 s` / `52,972.5 MB` | `75f0022894bde60b` |
| `zero / 202611` | `8` | `1.36064` | `1.34627` (`35.68%`) | `1.33499` (`1.0414`) | `1.37938` (`1.2210`) | `2908.7 s` / `52,972.5 MB` | `3d09eded33b58b92` |
| `zero / 202622` | `8` | `1.31245` | `1.26930` (`39.10%`) | `1.23335` (`0.7827`) | `1.30698` (`1.4118`) | `3059.1 s` / `52,972.5 MB` | `a8eb55bfad0d266c` |
| `zero / 202633` | `8` | `1.34157` | `1.31870` (`32.40%`) | `1.29266` (`0.6733`) | `1.38119` (`1.2735`) | `2983.0 s` / `52,972.5 MB` | `ea7aebcc6d53bc71` |

*注：全部 6 次训练的 `host_sha256_after` 与初始完全一致；仅在训练预算结束后的独立开发集 `dv_groups` 全量拟合标量温度 `(T_q3, T_q2)`，训练期损失均在 `1.5–1.6` 顺利降至 `1.29–1.38`。*

### 2.3 揭盲前独立开发先导方差检查与 `B_locked` 锁定（[dev_pilot_lock.json](results/dev_pilot_lock.json)）

| 诊断维度（主格 `G1` `±0.40`，131 区组） | 相位臂 `phase` | 对照臂 `zero` |
|---|---|---|
| **原始宿主：`(0,1)` 仅转第二窗保持性**（`mean / worst max_abs`；`>0.20` 占比） | `0.0591 / 0.4817`；`5.11%` | `0.0620 / 0.3587`；`5.80%` |
| **原始宿主：`(1,1)` 两窗同转一象限循环移位**（`mean / worst max_abs`；`>0.20` 占比） | `0.1263 / 0.3800`；`15.44%` | `0.1334 / 0.3792`；`18.96%` |
| **原始宿主：`(1,0)` 仅转第一窗循环移位**（`mean / worst max_abs`；`>0.20` 占比） | `0.1237 / 0.3752`；`15.13%` | `0.1316 / 0.3859`；`17.93%` |
| **最坏 `(1,1)` 逐例违例快照** | `s202622, +ω, mother=39, c00=198 → c11=10`；`probs_00=[0.1395, 0.1004, 0.1689, 0.5913]` vs `probs_11=[0.2113, 0.3310, 0.2750, 0.1827]`（`diff = 0.3800`） | `s202611, +ω, mother=1637, c00=9 → c11=77`；`probs_00=[0.6403, 0.0815, 0.0630, 0.2152]` vs `probs_11=[0.2778, 0.2611, 0.2501, 0.2110]`（`diff = 0.3792`） |
| **原始宿主对照：`independent_x3` 独立第三符号**（应接近均匀 `25%`） | `Acc = 24.91%`, `NLL = 1.4237`, `mean_tv_from_unif = 0.0947`, `max_tv = 0.2935` | `Acc = 24.97%`, `NLL = 1.4315`, `mean_tv_from_unif = 0.1122`, `max_tv = 0.2783` |
| **原始宿主对照：`shuffled_Z` 打乱缓存** | `Acc = 25.20%`, `NLL = 1.4601` | `Acc = 25.16%`, `NLL = 1.5070` |
| **原始宿主总体指标（未做任何离线对称平均）** | **`23.80% / 1.4452`** | **`19.41% / 1.4809`** |
| **离线外部对称操作 I：16 路群平均 `Pf`**（输出后 CPU 计算） | `17.05% / 1.4113`（相对 raw `Acc -6.75 pp [-7.09, -6.41]`，`NLL +0.0339`） | `16.43% / 1.4391`（相对 raw `Acc -2.98 pp [-3.29, -2.66]`，`NLL +0.0418`） |
| **离线外部对称操作 II：单代表 `F0`**（输出后 CPU 计算） | `20.88% / 1.4112` | `20.14% / 1.5090` |

> **解读**：由于 Qwen 连接层在训练中只采用软旋转不变提示而不是严格 $\mathbb{Z}_4 \times \mathbb{Z}_4$ 群等变连接，原始宿主仍存在约 `0.06` (`rot01`) 与 `0.12–0.13` (`rot11/10`) 的等变破缺。有趣的是，当原始准确率本身围绕或略低于 `25%` 且存在跨象限系统误判时，离线强制执行 16 路群平均 `Pf` 虽能把极化预测拉平滑以降低 NLL（`1.4452 → 1.4113`），反而使准确率进一步下降至 `17.05%`——说明外部群平均并不能凭空创造未学到的 $\pi/2$ 物理方向知识，二者必须在报告与统计中严格分列。

---


### 6.1 Q2 真实输入表示与连续参考边界说明

- **输给宿主的导频表示**：保留 Stage 3 的 12 维连续特征向量 `build_ref_feature(r)`（实部与虚部匹配投影及其归一/离散辅助统计，通过 `ref_mlp` 映射为 4 个 `num_ref_tokens` 连续隐层 token 插入题面对应 `<REF>` 槽位），**未对导频再做 `s17` 4-bit 离散量化**；同时仅对 `(z1, z2)` 两主数据窗保留 `s17` 4-bit 量化。因此，理论连续参考（如 `full_iq`）比 `q2_reader.read` 离散核能看到两窗与导频间未经 4-bit 离散截断的精确连续幅相。
- **前端去重实现**：由 [data_stage3d.py](data_stage3d.py) `build_q2_arm_references()` 提供共享 `deduplicate` 标志。`duplicate_copy_dedup` 与 `valid` 经 `acquisition_id` 去重后产生 100% 相同的 `<REF>` token 与提示词；`duplicate_copy_raw` 刻意关闭去重以测量宿主对重复 `<REF>` token 的冗余敏感度。

### 6.2 Work A 七臂与物理参照表（512 条母波形，3 种子汇总 vs CPU `q2_reader.read`）

| Q2 证据臂 | `phase` Acc / NLL (`ECE`) | `zero` Acc / NLL (`ECE`) | `q2_reader.read` 物理核 Acc / NLL (`ECE`) | 相对基线物理/行为意义 |
|---|---:|---:|---:|---|
| `none`（无参考） | `24.80% / 1.3928` (`0.0230`) | `25.78% / 1.3938` (`0.0189`) | `25.00% / 1.3863` (`0.0000`) | 盲先验基准 |
| `valid`（1 条同信道导频） | **`28.78%`** / `1.3889` (`0.0304`) | `26.63% / 1.4107` (`0.0557`) | `66.41% / 0.8192` (`0.0301`) | 有效证据进入 |
| `duplicate_copy_dedup`（副本去重） | **`28.78%`** / `1.3889` (`0.0304`) | `26.63% / 1.4107` (`0.0557`) | `66.41% / 0.8192` (`0.0301`) | **与 `valid` 逐例零差异**（`mean_tv = 0.0`） |
| `duplicate_copy_raw`（副本未去重） | `26.30% / 1.3944` (`0.0284`) | `25.78% / 1.4133` (`0.0463`) | — | 仅重复提示同条采集向量；相对 `valid`: `phase -2.47 pp` (`TV=0.0404`, `160` 改对 / `198` 错), `zero -0.85 pp` (`TV=0.0421`) |
| `independent_only`（纯独立无关） | `24.80% / 1.4077` (`0.0364`) | `24.93% / 1.4205` (`0.0499`) | `25.00% / 1.3863` (`0.0000`) | 相对 `none`: `phase +0.00 pp` (`TV=0.0423`, `222` 对 / `222` 错), `zero -0.85 pp` (`TV=0.0663`) |
| `valid_plus_independent` | `24.54% / 1.4082` (`0.0425`) | `27.28% / 1.4183` (`0.0480`) | `66.41% / 0.8192` (`0.0301`) | 无效导频干扰；相对 `valid`: `phase -4.23 pp` (`TV=0.0403`, `159` 对 / `224` 错), `zero +0.65 pp` (`TV=0.0681`) |
| **`fresh_resample`**（第 2 次独立采样） | `26.43% / 1.4056` (`0.0348`) | `25.78% / 1.4064` (`0.0592`) | **`74.22% / 0.6632`** (`0.0333`) | 相对 `valid`：物理核 **`+7.81 pp / -0.1560 nats`**（Stage 3C 回归基准 `+8.5636 pp`）；**而宿主 `phase` 为 `-2.34 pp / +0.0168 nats`（`172` 对 / `208` 错, `max_conf_inc=0.4181`），`zero` 为 `-0.85 pp`（`159` 对 / `172` 错）** |

- **具体母波形轨迹例（`mother_idx = 36, z1 = 11, z2 = 15, true_x1 = 3`, `seed = 202611`）**：`q2_reader.read` 物理核随着有效参考由 `none [0.25, 0.25, 0.25, 0.25]` → `valid [0.0100, 0.2563, 0.6672, 0.0665]`（尚偏向索引 2）→ `fresh_resample [0.0201, 0.0217, 0.2945, 0.6638]` 准确将真标签 `3` 概率从 `0.0665` 拉升至 **`0.6638`** 扭转判决；而同一母波形上 `phase` 臂输出几乎停滞在 `[0.2497, 0.2495, 0.2503, 0.2505]`（`valid`）与 `[0.2498, 0.2494, 0.2503, 0.2505]`（`fresh_resample`），`zero` 臂虽在 `fresh_resample` 将 `prob_3` 升至 `0.2964`，整体却不敌其他样本上的置信扰动导致全样本净降 `-0.85 pp`。

### 6.3 旧工况 `C1 / C2 / C3` 回放未遗忘验证（`s3_test` × 2 区块 × 3 旋转 = 384 条 × 3 种子）

| 条件 | 任务 | `phase` Acc / NLL | `zero` Acc / NLL | 同期 `s17` 物理读出 Acc / NLL |

## 五步与验收

| 步骤 | 做什么 | 怎样算达到这一步 |
|---|---|---|
| **1　用数学定下两次追问** | 从两个四相符号和一个未知公共旋转开始。第一问是相对相位，第二问是补充同一相干区间的已知导频后，第一个符号的原始相位。写清先后可见的信息、允许缓存的比特和噪声条件。再提出参考距离增大时可检验的预测。 | 用精确枚举得到各题在揭示前后的答案或后验。加入独立无关参考、重复同一观测、只保留相对相位三个对照。原计划中的 2 比特与 4 比特界仅用于无噪四相、固定长度、零误差的例子。 |
| **2　把波形变成可复用的输入** | 编码器接收 I、Q 采样和采样率等可用元数据，输出按时间排列的特征向量，每个向量都能对应回一段原始采样。同一波形的编码对不同问题共用。保存缓存预算、量化精度、时间映射和有效观测掩码。 | 先用简单的读出器检查，同一份冻结表示能回答相对相位，也能在补入有效参考后恢复原始相位。以完整 I/Q 和解析相量为参照。比较逐窗丢弃相位后，哪道题先失效。 |
| **3　与一个已有模型对齐** | 在提问中为每段波形预留位置，用小型可训练投影层把波形向量映射到宿主词向量的维数，连同文字进入注意力层。首轮冻结已验证的波形编码器和宿主，用波形、问题、答案三元组训练投影层。新一轮追加参考向量时复用原表示。 | 在未用于训练的合成波形上，模型能完成“先回答、再追问、补参考后更新”。平衡不同答案的出现频率，同一句问法配不同波形。只给文字、打乱波形配对或只保留幅度时，逐题核对预期变化。交付一个能实际对话的单宿主原型。 |
| **4　在真实波形上验收** | 用公开已知序列定义真值，优先选 Pluto 连续 QPSK 开发，在完整留出的 CTTC 会话或另一次物理采集上测试。在相同片段上对比解析读出、从冻结向量直接读出、宿主语言回答。加入时间轴图与 I/Q 共同决定答案的测试。 | 分别报告自然录制和人工相位干预。记录答案正确率、候选概率、补证前后改对和改错的样例，以及定位证据的时间误差。普通文本与图像问题也跑一组成对回归。若表示可被直接读对而语言回答错，先修对齐；若两者都错，用完整 I/Q 回查输入信息与相干假设。 |
| **5　迁移到第二个宿主** | 固定同一波形编码器、预处理、时间坐标、权重和问题集。为另一个模型训练专属的小投影层，用各自的对话模板组织上下文。 | 两个宿主都通过同一套物理干预与实测题，报告各自的误差、每段波形占用的上下文及推理开销。发布共享编码器、一份输入规范及各自的适配权重。结论明确限定在测试过的宿主和波形上。 |

```

### Event #43: `item.completed` — `file_change` (`item_40`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M6\\verify_m6.py",
    "kind": "update"
  }
]
```

### Event #44: `item.completed` — `agent_message` (`item_41`)

The replay found a `0.012088`-nat gap from Full IQ to s17 in G1, then a `0.504103`-nat gap from s17 to the calibrated phase arm. The locked temperature added `0.026077` nats on G1. The simulation passed its modeled error and power targets, and I’m documenting the assumptions that determine whether those savings carry over to Stage 4.

### Event #45: `item.completed` — `file_change` (`item_42`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M6\\THEOREM_M6.md",
    "kind": "add"
  }
]
```

### Event #46: `item.completed` — `command_execution` (`item_44`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding UTF8 research/stage4_math_and_design/M6/THEOREM_M6.md | Select-Object -First 45; rg -n '"'^#|97'"\\.66|16\\.1' research/stage4_math_and_design/M6/THEOREM_M6.md; Get-ChildItem research/stage4_math_and_design/M6 -Force | Select-Object Name,Length"
```

**Aggregated Output:**

```text
# M6 Round 1 ? CPU certificates and a capped Stage 4 experiment

**Decision.** Verify the evidence contract and run the physical readout before training the connector. Make rotation and reference handling part of the interface. Use the actual frozen host's candidate-score Jacobian to scope a local adaptation claim. Buy one bounded pilot only after these checks pass. Use independent blind data for the final decision. The proposed Stage 4 compute cap is **$9.50**, leaving **$3.38** of the historically reported $12.88. Round 1 used local CPU only.

## 1. The exact population loss ladder

Let (A\in\{0,\ldots,m-1\}) be the answer label, (Y) the permitted raw observations at the time of the question, (Z=T(Y)) the frozen cache, and (D) the public condition, question, and evidence available through that time. (D) includes the revealed references and their provenance where these enter both comparators, plus the host's instruction or the variables needed to render it. Condition on the frozen checkpoint and a known seed, or include the independently assigned seed in (D). Define

\[
U_0=(Y,D),\quad U_1=(Z,D),\quad U_2=(E_\theta(U_1),\mathrm{prompt}(U_1)),\quad
U_3=L=\Pi s(H_{\rm LLM}(U_2)),\quad q_\tau=\mathrm{softmax}(L/\tau),
\qquad \Pi=I-\mathbf1\mathbf1^\top/m.
\]

Here (U_2) contains every token embedding, mask, and position that reaches the host. (s) is the vector of candidate sequence log-likelihoods. Public side information in (U_2) is a function of (U_1). If the system gains a reference later, instantiate a second ladder at that reveal state. If a downstream stage is handed a reference or an oracle parameter that the upstream comparator lacks, the stated chain need not hold. For stochastic inference, use label-independent transition noise or condition on the relevant randomness. A fitted temperature is held fixed on evaluation.

Write (p_j=P(A\mid U_j)), (R(q)=\mathbb E[-\log q(A)]), and (H_j=H(A\mid U_j)). For (A\to U_j\to U_{j+1}), (\mathbb E[p_j\mid U_{j+1}]=p_{j+1}). Expanding the logarithm and conditioning gives

\[
\begin{aligned}
\mathbb E\,D_{\rm KL}(p_j\Vert p_{j+1})
&=\mathbb E\sum_a p_j(a)\log p_j(a)-\mathbb E\sum_a p_{j+1}(a)\log p_{j+1}(a)\\
&=H_{j+1}-H_j=I(A;U_j\mid U_{j+1}).
\end{aligned}
\]

Also (R(q_\tau)=H_3+\mathbb E D_{\rm KL}(p_3\Vert q_\tau)). The sum telescopes to

\[
R(q_\tau)-H_0=\underbrace{I(A;Y\mid Z,D)}_{\Delta_{\rm enc}}+
\underbrace{I(A;U_1\mid U_2)}_{\Delta_{\rm conn}}+
\underbrace{I(A;U_2\mid L)}_{\Delta_{\rm host}}+
\underbrace{\mathbb E D_{\rm KL}(P(A\mid L)\Vert q_\tau)}_{\Delta_{\rm cal}}.
\]

Each term is nonnegative. A transition preserves Bayes log-loss exactly when its posterior is unchanged almost surely. The host term includes the conversion from hidden states to the restricted candidate score vector. One can insert a hidden-state rung and split that term if the hidden states are recorded. A small output dimension or a low rank in a linearization does not by itself establish positive conditional MI. Even one real number can encode a finite cache injectively. Conversely, an injective prefix can let the host output unusable probabilities. A fixed positive scalar temperature is invertible on centered logits through (L=\tau\Pi\log q_\tau). It cannot remove a score collision or change the top-1 decision. The difference (R(q_\tau)-R(q_1)) is signed and measures the effect of the chosen temperature. It differs from the nonnegative calibration term above.

### What the Stage 3D archive identifies

The verifier replays the real NPZ artifacts. Q3 is synthetic at 2 dB, with paired CFO signs and three rotations per mother. It averages scores over seeds, rather than scoring an ensemble. Full IQ uses known channel parameters and the repository's matched synthetic likelihood. The s17 table is a physical readout and can have its own mismatch to (P(A\mid Z,D)). In general the observed direct-minus-IQ gap equals the population encoding term plus the difference of the two readouts' KL regrets, up to sampling. If Full IQ is the true posterior under the matched law, the population direct-minus-IQ gap upper-bounds the encoding penalty. If the Full-IQ solver is approximate, include its error in any certified bound. No intervening prefix, hidden-state posterior, or cost-free counterfactual training is available in these archives. The replay localizes the **operational** deficit to the path after the cache, with temperature measured separately. The internal conditional MI terms remain unidentified. If independent validation detects no input leaks, the comparator already witnesses that this cache permits much better G1 decisions.

| Q3 cell | Mothers | Full IQ Acc / NLL | s17 Acc / NLL | Phase Acc / NLL | Zero Acc / NLL | s17 ? IQ NLL | Phase ? s17 NLL | Zero ? s17 NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| G1 both extrapolated | 8,384 | 64.641% / .929038 | 64.617% / .941126 | 23.799% / 1.445229 | 19.406% / 1.480879 | .012088 | .504103 | .539753 |
| G2 both interpolated | 512 | 46.777% / 1.019346 | 46.777% / 1.022230 | 33.876% / 1.335826 | 36.361% / 1.308067 | .002884 | .313596 | .285837 |
1:# M6 Round 1 • CPU certificates and a capped Stage 4 experiment
5:## 1. The exact population loss ladder
38:### What the Stage 3D archive identifies
50:For G1 the paired 64-mother block 95% intervals are **[.009178, .014998]** for s17 minus Full IQ and **[.491294, .516912]** for phase minus s17. Phase's observable telescope is .012087838 + .478025740 + .026076989 = .516190567 nats, using unit-temperature host scores for the middle contrast. About **97.66%** of phase's excess over Full IQ lies after the s17 readout when measured at locked temperature. The G1 phase-versus-zero uplift did meet Stage 3D's relative target, even though both arms exceeded the log-loss of a uniform predictor (1.386294). G1's co-rotation worst coordinate defect was **.380019** in phase and near machine precision for the direct readout. Offline 16-way group averaging brought phase NLL to **1.411312** and accuracy to **17.050%**. On the archive's sampled rotations these are distinct endpoints. The all-rotation `H_c4_035` row above is a fresh replay from the archive. Its identity-rotation, both-sign host metrics match Work F. The separate positive-sign Stage 3B lookup anchor in the master JSON is a different comparison.
66:## 2. Tier 0 • analytic rejection before any host work
68:### 2.1 Bayes-risk brackets for each evidence state
80:### 2.2 Exact group action and reference algebra
88:## 3. Tier 1 • CPU geometry and the frozen host
94:### 3.1 Rank, curvature and optimization certificates
102:### 3.2 Geometry of (W_{\rm out}) and vocabulary readout
116:## 4. Tier 2 • sequential decisions and an exact cost envelope
118:### 4.1 VoI and continuation
128:### 4.2 Two-look design and what the Monte Carlo verifies
154:### 4.3 Stage 4 Modal plan in US dollars
174:At the historical 1.9757 s/step the 2,707 continuation steps would take ~5,348 s before overhead. Require a receipt within 2.30 s/step, leaving ~524 s for all continuation overhead at that worst allowed rate. The new preprocessing, fused kernels and longest Q2 inputs can change this. At most 6,144 Q3 prefixes cover 4 cells × 2 signs × 256 exact keys × 3 seeds, provided the key includes all prefix state, dtype and batch policy. Q2's seven evidence arms require at most 10,752 prefixes for 512 mothers × 3 seeds. The 1,050 s synthetic scoring line therefore requires 16.1 prefixes/s before overhead. Cache prompts only across identical embeddings, masks, positions, template, candidate scope and attention state. Distinguish KV/linear-state sharing from lookup deduplication. Fix the padding and batch policy and require per-probability and paired endpoint parity on a receipt set before any blind inference. Historical BF16 differences across batch settings reached .067 in probabilities with per-cell argmax agreement as low as .6875. Historical A100 b64 peaked at 45,410 MB on inference, so use smaller fixed batches or prevalidated kernels on 24 GB cards. Keep precision locked across comparable arms.
178:## 5. Stage 4 → 6 critical path
215:## Replay

Name            Length
----            ------
.uv-cache             
results_m6.json 28229 
THEOREM_M6.md   45771 
verify_m6.py    19434 


```

### Event #47: `item.completed` — `command_execution` (`item_43`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=Join-Path (Get-Location) '"'research/stage4_math_and_design/M6/.uv-cache'; "'$env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$env:TEMP=$env:UV_CACHE_DIR; $env:TMP=$env:UV_CACHE_DIR; $env:PYTHONUTF8='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M6/verify_m6.py"
```

**Aggregated Output:**

```text
VERIFIED results_m6.json: parity max |diff|=3.56e-16, 10,000 trials, cap $9.50
```

### Event #48: `item.completed` — `command_execution` (`item_45`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n -A 5 '\"sensitivity\"|\"direct_rot11_worst_max_abs\"' research/stage4_math_and_design/M6/results_m6.json; Get-Item -LiteralPath research/stage4_math_and_design/M6/.uv-cache | Select-Object Name,Attributes; git status --short --untracked-files=all | Where-Object {"'$_ -like '"'*M6*'} | Select-Object -First 40"
```

**Aggregated Output:**

```text
82:        "direct_rot11_worst_max_abs": 3.3306690738754696e-16
83-      },
84-      "G2_interp_both": {
85-        "mothers": 512,
86-        "logical_rows_per_host_arm": 9216,
87-        "full_iq": {
--
689:    "min_working_drift_for_90_percent_power_at_cpu_sensitivity_099": 3.2676570871466035,
690-    "pilot_cost_fraction": 0.085,
691-    "pilot_usd_assumed": 1.4552000000000003,
692-    "fixed_usd": 17.12,
693-    "sequential_mean_usd_mc": 6.10124272,
694-    "sequential_mean_usd_exact": 6.107866319813384,
--
697:    "sensitivity": {
698:      "working_power_at_cpu_sensitivity_095": 0.8962155050882968,
699-      "mean_sequential_usd_no_cpu_screen_in_same_mixture": 11.096339644972625,
700-      "cost_reduction_pct_no_cpu_screen_in_same_mixture": 35.18493198030009,
701-      "cost_reduction_pct_if_all_working": 4.27059408860112
702-    },
703-    "null_mixture_false_efficacy_seq_mc": 0.008,

Name      Attributes
----      ----------
.uv-cache  Directory
?? research/stage4_math_and_design/M6/THEOREM_M6.md
?? research/stage4_math_and_design/M6/results_m6.json
?? research/stage4_math_and_design/M6/verify_m6.py
?? research/stage4_math_and_design/discussions/M6/prompt_r1.md
?? research/stage4_math_and_design/discussions/M6/round1_events.jsonl


```

### Event #49: `item.completed` — `command_execution` (`item_46`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=Join-Path (Get-Location) '"'research/stage4_math_and_design/M6/.uv-cache'; "'$env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$env:TEMP=$env:UV_CACHE_DIR; $env:TMP=$env:UV_CACHE_DIR; $env:PYTHONUTF8='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B -c \"from pathlib import Path; d=Path('research/stage4_math_and_design/discussions'); print('\\n'.join(k.name+': '+(k/'prompt_r1.md').read_text(encoding='utf-8').splitlines()[0] for k in sorted(d.glob('M[1-5]'))));\""
```

**Aggregated Output:**

```text
M1: # Track M1 (Round 1): Group-Theoretic Phase-Coupled Equivariant Architecture & Extrapolation Theory
M2: # Track M2 (Round 1): Information-Theoretic Bregman/KL Decomposition, Calibration Forensics & Stage 4 Statistical Gate
M3: # Track M3 (Round 1): Minimal Bayesian Sufficiency for Variable Pilot Multisets (Q2 / Q3_after_pilot) & Idempotent Pooling
M4: # Track M4 (Round 1): Real-Data Distribution-Shift Bounds, Pluto/CTTC Benchmark Adaptation & Modal Free-Tier Architecture
M5: # Track M5 (Round 1): First-Principles Unified Multi-Query Cache, Active Reference Selection (Ultimate Blind Test) & Stage 4→6 Bridge
```

### Event #50: `item.completed` — `file_change` (`item_47`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M6\\.uv-cache\\format_m6_math.py",
    "kind": "add"
  }
]
```

### Event #51: `item.completed` — `command_execution` (`item_48`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=Join-Path (Get-Location) '"'research/stage4_math_and_design/M6/.uv-cache'; "'$env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$env:TEMP=$env:UV_CACHE_DIR; $env:TMP=$env:UV_CACHE_DIR; $env:PYTHONUTF8='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M6/.uv-cache/format_m6_math.py"
```

**Aggregated Output:**

```text
A\in\{0,\ldots,m-1\}
Y
Z=T(Y)
D
D
D
U_2
s
U_2
U_1
p_j=P(A\mid U_j)
R(q)=\mathbb E[-\log q(A)]
H_j=H(A\mid U_j)
A\to U_j\to U_{j+1}
\mathbb E[p_j\mid U_{j+1}]=p_{j+1}
R(q_\tau)=H_3+\mathbb E D_{\rm KL}(p_3\Vert q_\tau)
L=\tau\Pi\log q_\tau
R(q_\tau)-R(q_1)
P(A\mid Z,D)
q=.35,\rho=.01
S=(Z,C,\mathcal R)
H(A\mid S)=-\sum_{a,s}p(a,s)\log p(a\mid s)
D
I(A;R_{\rm new}\mid S)\ge0
q_S
H(A\mid S)\le R(q_S)
I(A;S)\ge H(A)-R(q_S)
S=f(Y,D)
H(A\mid Y,D)
H(A\mid S,O)
\inf_{p\in\mathcal P}H_p(A\mid S)
\sup_{p\in\mathcal P}H_p(A\mid S)
S=s
\log 2
\tfrac12(1-\mathrm{TV}(P_0,P_1))
r_*=R(q_{\rm IQ})+.05
L_S>r_*
I(A;R_{\rm new}\mid S)
g=(r_1,r_2)\in C_4^2
z_i\mapsto z_i+4r_i\bmod16
P_g=\mathrm{roll}(r_1)
P(A\mid gx)=P_gP(A\mid x)
q(gx)=P_gq(x)
d=\mathrm{TV}(q(gx),P_gq(x))
d^2/2
A\perp R_d\mid Z,C,\mathcal R_{\rm valid}
q_{\rm eq}(x)=|G|^{-1}\sum_g P_g^{-1}q(gx)
f(z,\phi)=a(z)+b(\phi)
f(z,\phi)-f(z',\phi)-f(z,\phi')+f(z',\phi')
\Re[u(z)e^{i\phi}\bar r_a]
\Re[(u(z)-u(z'))(e^{i\phi}-e^{i\phi'})\bar r_a]
u\cos\phi,u\sin\phi
\(|\phi|=1.60>\pi/2\)
z_1,z_2
e_i=\mathrm{vec}(E_\theta(x_i))\in\mathbb R^{Kd_{\rm LM}}
F_i(e_i)
J_i=D_eF_i(e_i^0)
\sum_t\log p(w_t\mid w_{<t},e_i)
A_i=J_iD_\theta e_i
d_i=\Pi\log p^*_i-F_i(e_i^0)
\inf_{\|\delta\|\le r}\|A\delta-d\|\le\epsilon
\|(I-AA^+)d\|
J_i
\|(I-AA^+)d\|-Mr^2/2\
\|A\|r+Mr^2/2
d\in\operatorname{im}A
\|d\|/s\le r
\delta=A^+d
M\|d\|^2/(2s^2)
p_{\min}\|e\|^2/2\le\mathrm{KL}(p^*\|q)\le\|e\|^2/2
e^{-t\lambda}
1-\eta\lambda
0<\eta<2/\lambda_{\max}
A_{\rm test}A_{\rm train}^+d_{\rm train}
W_{\rm out}
u\in\mathbb R^r
B=W_{\rm out}^\top Q
B^\top B=\beta^2I
v=Px
P=I-\mathbf1\mathbf1^\top/d\
s=(\|v\|^2/d+\varepsilon)^{1/2}
\varepsilon/s^3
\{\mathbf1,v\}^\perp
\min|\gamma|/s
\max|\gamma|/s
P=I,v=x
x\mapsto x+f_i(x)
\|Df_i\|\le\eta_i<1
C_{\rm cand}
E_{\rm vocab}\
D_{\rm answer}
M_{\rm cand}=\Pi C_{\rm cand}E_{\rm vocab}D_{\rm answer}B
J_iB
J_iU=L\Sigma V^\top
r
B=UV_r\Sigma_r^{-1}\beta
J_iB=\beta L_r
\|B\|\le\beta/\sigma_r(J_iU)
W_{\rm out}
m^0_{ab}
m^0_{ab}+(e_a-e_b)^\top J_iBu
\sqrt2\
\xi
D_i
b_i
u(a,\xi)
U(D)=\max_a\mathbb E[u(a,\xi)\mid D]
X_j
c_j
\mathrm{VoI}_j=\mathbb E[U(D,X_j)\mid D]-U(D)
v_j(n_j)
c_j
v'_j(n_j)/c_j=\lambda
v_j=-w_j\sigma_j^2/n_j
n_j\propto\sqrt{w_j\sigma_j^2/c_j}
\mu>0
Z_1\sim N(\sqrt t\mu,1)
Z_2=\sqrt tZ_1+\sqrt{1-t}V\
V\sim N(\sqrt{1-t}\mu,1)
t=.18
a(t)=\bar\Phi(z_{1-\alpha}/\sqrt t)
a(t)=\alpha\log[1+(e-1)t]
a(.18)=.00005288548
b_1=3.876957
a(t)+\int_{-\infty}^{b_1}\varphi(z)\bar\Phi((b_2-\sqrt tz)/\sqrt{1-t})\,dz=.05
b_2=1.645088
\mu
\mu\sim N(0,4)
\mu\mid Z_1=z\sim N(v\sqrt tz,v)
v=(1/4+t)^{-1}
Z_2
\sqrt tz[1+(1-t)v]
1-t+(1-t)^2v
P(Z_2\ge b_2\mid z)\ge .10
z\ge -0.270438
\(\mu=-.8\), CPU pass .10
\(\mu=0\), pass .90
\(\mu=3.6\), pass .99
\(.75\times.18/3+.04\) of fixed, comprising a training share and a pilot overhead share
p_s(\mu)=P(\mathrm{CPU\ pass}\mid\mu)
C_4^2
D_t
W_j
I(A;W_j\mid D_t)
```

### Event #52: `item.completed` — `file_change` (`item_49`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M6\\.uv-cache\\format_m6_math.py",
    "kind": "update"
  }
]
```

### Event #53: `item.completed` — `command_execution` (`item_50`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=Join-Path (Get-Location) '"'research/stage4_math_and_design/M6/.uv-cache'; "'$env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$env:TEMP=$env:UV_CACHE_DIR; $env:TMP=$env:UV_CACHE_DIR; $env:PYTHONUTF8='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M6/.uv-cache/format_m6_math.py --write | Out-Null"
```

### Event #54: `item.completed` — `file_change` (`item_51`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M6\\reply_r1.md",
    "kind": "add"
  }
]
```

### Event #55: `item.completed` — `command_execution` (`item_52`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n 'p\\\\mid|\\|\\"'$'"\\||R\\(q|texture|A,S|a\\+b|\\\\\\(' research/stage4_math_and_design/M6/THEOREM_M6.md; rg -n '"'^## |''^### |sensitivity|R'"\\(' research/stage4_math_and_design/M6/THEOREM_M6.md | Select-Object -First 30; git status --short --untracked-files=all | Where-Object {"'$_ -like '"'*M6/*'}"
```

**Aggregated Output:**

```text
17:Write $p_j=P(A\mid U_j)$, $R(q)=\mathbb E[-\log q(A)]$, and $H_j=H(A\mid U_j)$. For $A\to U_j\to U_{j+1}$, $\mathbb E[p_j\mid U_{j+1}]=p_{j+1}$. Expanding the logarithm and conditioning gives
27:Also $R(q_\tau)=H_3+\mathbb E D_{\rm KL}(p_3\Vert q_\tau)$. The sum telescopes to
30:R(q_\tau)-H_0=\underbrace{I(A;Y\mid Z,D)}_{\Delta_{\rm enc}}+
36:Each term is nonnegative. A transition preserves Bayes log-loss exactly when its posterior is unchanged almost surely. The host term includes the conversion from hidden states to the restricted candidate score vector. One can insert a hidden-state rung and split that term if the hidden states are recorded. A small output dimension or a low rank in a linearization does not by itself establish positive conditional MI. Even one real number can encode a finite cache injectively. Conversely, an injective prefix can let the host output unusable probabilities. A fixed positive scalar temperature is invertible on centered logits through $L=\tau\Pi\log q_\tau$. It cannot remove a score collision or change the top-1 decision. The difference $R(q_\tau)-R(q_1)$ is signed and measures the effect of the chosen temperature. It differs from the nonnegative calibration term above.
72:For Pluto and CTTC, a causal physical $q_S$ fitted off-fold provides $H(A\mid S)\le R(q_S)$, hence $I(A;S)\ge H(A)-R(q_S)$. This is a one-sided variational bound on MI. Poor learned/table scores alone leave the opposite bound unknown. Obtain a lower risk bound by one of the following valid routes.
75:- On finite locked support, form a simultaneous confidence set $\mathcal P$ for the true joint law of (A,S), then use $\inf_{p\in\mathcal P}H_p(A\mid S)$ and $\sup_{p\in\mathcal P}H_p(A\mid S)$. For i.i.d. units, a Bonferroni union of exact binomial intervals for fixed atom probabilities, intersected with the simplex, is one valid construction. For independent clusters, use bounded cluster contributions to each atom and a simultaneous concentration bound, or valid cluster randomization on a randomized contrast. If the support is large the interval may be vacuous. Binning a continuous reference gives an entropy **upper** bound on the richer interface. A lower bound for the binned interface applies only when the bins are literally what the host receives.
76:- A collision class with fixed $S=s$ and at least masses (a,b>0) of two distinct labels contributes at least (a+b)h_2(a/(a+b)) to conditional entropy (restrict to the two-label subevent and condition on its indicator). Exact equivalence of two hypotheses with equal prior forces at least $\log 2$ nats and error at least (1/2) within that subproblem. If their distributions are only close, equal-prior optimal binary error is $\tfrac12(1-\mathrm{TV}(P_0,P_1))$, from the likelihood-ratio decision rule. This yields a CPU impossibility bound when TV can be bounded above.
78:**Proposed decision margins.** On development folds, lock an admissible NLL target $r_*=R(q_{\rm IQ})+.05$ nats and a useful mean reveal improvement of .02 nats against the no-reveal control. Use each contrast at fixed context and comparable label distributions, with simultaneous bounds for multiple gate decisions. A certified $L_S>r_*$ ends the current representation branch. A certified ceiling on $I(A;R_{\rm new}\mid S)$ below .02 ends a **Bayes-gain** claim for that reveal contract. A direct readout whose paired upper regret bound is within .05 and whose reveal lower improvement bound exceeds .02 is a practical existence witness. Otherwise report uncertainty, improve the physical kernel or support, and defer GPU. The NLL margins are Round 1 proposals for registration, not inherited Stage 4 thresholds.
5:## 1. The exact population loss ladder
17:Write $p_j=P(A\mid U_j)$, $R(q)=\mathbb E[-\log q(A)]$, and $H_j=H(A\mid U_j)$. For $A\to U_j\to U_{j+1}$, $\mathbb E[p_j\mid U_{j+1}]=p_{j+1}$. Expanding the logarithm and conditioning gives
27:Also $R(q_\tau)=H_3+\mathbb E D_{\rm KL}(p_3\Vert q_\tau)$. The sum telescopes to
30:R(q_\tau)-H_0=\underbrace{I(A;Y\mid Z,D)}_{\Delta_{\rm enc}}+
36:Each term is nonnegative. A transition preserves Bayes log-loss exactly when its posterior is unchanged almost surely. The host term includes the conversion from hidden states to the restricted candidate score vector. One can insert a hidden-state rung and split that term if the hidden states are recorded. A small output dimension or a low rank in a linearization does not by itself establish positive conditional MI. Even one real number can encode a finite cache injectively. Conversely, an injective prefix can let the host output unusable probabilities. A fixed positive scalar temperature is invertible on centered logits through $L=\tau\Pi\log q_\tau$. It cannot remove a score collision or change the top-1 decision. The difference $R(q_\tau)-R(q_1)$ is signed and measures the effect of the chosen temperature. It differs from the nonnegative calibration term above.
38:### What the Stage 3D archive identifies
66:## 2. Tier 0 �?analytic rejection before any host work
68:### 2.1 Bayes-risk brackets for each evidence state
72:For Pluto and CTTC, a causal physical $q_S$ fitted off-fold provides $H(A\mid S)\le R(q_S)$, hence $I(A;S)\ge H(A)-R(q_S)$. This is a one-sided variational bound on MI. Poor learned/table scores alone leave the opposite bound unknown. Obtain a lower risk bound by one of the following valid routes.
78:**Proposed decision margins.** On development folds, lock an admissible NLL target $r_*=R(q_{\rm IQ})+.05$ nats and a useful mean reveal improvement of .02 nats against the no-reveal control. Use each contrast at fixed context and comparable label distributions, with simultaneous bounds for multiple gate decisions. A certified $L_S>r_*$ ends the current representation branch. A certified ceiling on $I(A;R_{\rm new}\mid S)$ below .02 ends a **Bayes-gain** claim for that reveal contract. A direct readout whose paired upper regret bound is within .05 and whose reveal lower improvement bound exceeds .02 is a practical existence witness. Otherwise report uncertainty, improve the physical kernel or support, and defer GPU. The NLL margins are Round 1 proposals for registration, not inherited Stage 4 thresholds.
80:### 2.2 Exact group action and reference algebra
88:## 3. Tier 1 �?CPU geometry and the frozen host
94:### 3.1 Rank, curvature and optimization certificates
102:### 3.2 Geometry of $W_{\rm out}$ and vocabulary readout
116:## 4. Tier 2 �?sequential decisions and an exact cost envelope
118:### 4.1 VoI and continuation
128:### 4.2 Two-look design and what the Monte Carlo verifies
150:Sequential exact-binomial MC intervals are **[2.716%, 4.671%]** on the boundary-null stratum and **[92.227%, 94.230%]** on the working stratum. The modeled 90% power needs drift **�?.267657** at .99 CPU sensitivity. At .95 sensitivity the same working stratum drops below 90% power. If every candidate is working, the analytic saving drops to about **4.27%**. These are selection and effect-size sensitivities, not a transfer claim. Savings come from declining expensive branches, not more favorable final testing.
152:**Actual Stage 4 release.** Use the pilot primarily for engineering, variance, and predictive futility on disjoint development mothers, sessions or blocks. Freeze pilot/final coupling and calibrate on external development simulations before using the two-look efficacy boundaries. If that law cannot be justified, allow only futility at the first look and apply a single valid, independent final test on blind blocks after all three seeds and scoring settings are fixed. Conditional on the pilot and model selection, the blind test must be valid for the declared null. Its unconditional false-promotion rate is at most its level by iterated expectation. Power after futility, particularly with seed-to-seed heterogeneity, still needs development calibration. Full release also requires passing physical margins and negative controls. Count mothers/recording clusters, never candidates or seeds as independent test rows. Use randomization inference where the experimental randomization supports it, or a declared block model with sensitivity to correlation. For Pluto, 16 temporal blocks support inference within the capture when justified. For CTTC, report 72 nested blocks and session-level (9-unit) sensitivity. A sign-flip test is exact for justified randomization or a stated joint sign-symmetry null, not for arbitrary mean-zero shifts.
154:### 4.3 Stage 4 Modal plan in US dollars
178:## 5. Stage 4 �?6 critical path
213:**Why these edges exist.** P1–P3 show strong matched classical baselines, position shortcuts and static-surrogate scope. P4–P6 require separating physical frontends from learned contributions and distinguishing missing from zero. P7–P8 show that large probability updates can yield few corrected blocks and that mismatched past observations can hurt. P9–P11 require state-dependent reference reliability, phase-gauge accounting and counting newly wrong alongside corrected decisions. P12 requires unique evidence accounting and compares identifiable global references against naive locality; plug-in point estimates can lose mixture information. The earlier P12 Stage 5 fusion results scoped neural gains to the tested distribution. Stage 3D adds the strong direct-reader/weak-host distinction, failed raw rotation laws, redundant reference sensitivity, and an expensive numerical parity lesson. The roadmap's anti-loop rule is to predeclare one new question, finish with a success case, a failure case and a changed decision, and archive small BER variations after two rounds. Record the missing observation required by an impossibility theorem. [Long-term plan](../../../docs/LONG_TERM_RESEARCH_PLAN.md), [Multimodal integration plan](../../../docs/MULTIMODAL_INTEGRATION_PLAN.md), [Stage 3D report](../../stage3d_qwen_signed_phase/REPORT.md), [Protocol lock](../../stage3d_qwen_signed_phase/PROTOCOL_LOCK.json)
215:## Replay
?? research/stage4_math_and_design/M6/THEOREM_M6.md
?? research/stage4_math_and_design/M6/reply_r1.md
?? research/stage4_math_and_design/M6/results_m6.json
?? research/stage4_math_and_design/M6/verify_m6.py
?? research/stage4_math_and_design/discussions/M6/prompt_r1.md
?? research/stage4_math_and_design/discussions/M6/round1_events.jsonl
```

### Event #56: `item.completed` — `command_execution` (`item_53`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=Join-Path (Get-Location) '"'research/stage4_math_and_design/M6/.uv-cache'; "'$env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$env:TEMP=$env:UV_CACHE_DIR; $env:TMP=$env:UV_CACHE_DIR; $env:PYTHONUTF8='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B -c \"from pathlib import Path; import re; p=Path('research/stage4_math_and_design/M6'); d=p/'THEOREM_M6.md'; s=d.read_text(encoding='utf-8'); assert not any(ord(c)<32 and c not in '\\n\\t' for c in s); print('display_math_fences', s.splitlines().count('"'$$'"')); print('code_fences',sum(line.startswith(chr(96)*3) for line in s.splitlines())); print('broken relative links', [link for f in (d,p/'reply_r1.md') for link in re.findall(r'\\]\\((["'^)]*)'"\\)',f.read_text(encoding='utf-8')) if not link.startswith(('http://','https://')) and not (p/link).exists()]);\""
```

**Aggregated Output:**

```text
display_math_fences 68
code_fences 4
broken relative links []
```

### Event #57: `item.completed` — `file_change` (`item_54`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M6\\verify_m6.py",
    "kind": "update"
  }
]
```

### Event #58: `item.completed` — `file_change` (`item_55`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M6\\THEOREM_M6.md",
    "kind": "update"
  }
]
```

### Event #59: `item.completed` — `command_execution` (`item_57`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '"'^'"\\"'$'"\\"'$$|(?i)not available'"' research/stage4_math_and_design/M6/THEOREM_M6.md; "'$p=Join-Path $env:USERPROFILE '"'.codex/memories/MEMORY.md'; rg -n 'scope: Assess and publish|reuse_rule=Reuse the matched-control|2026-11-17T20-54-26-dFGP' "'$p; rg -n '"'most of the improvement|strictly matched comparison|thread_id: 01a1eb4a' (Join-Path "'$env:USERPROFILE '"'.codex/memories/rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md')"
```

**Aggregated Output:**

```text
31:scope: Assess and publish the EM-native Transformer P4 simulation without overstating learned-model performance.
32:applies_to: cwd=C:\Users\李则徐\Documents\ChatGPT\EM-FloatFormer; reuse_rule=Reuse the matched-control, reproducibility, and publication approach for this repository; numerical results, commit, hardware status, and receiver outcomes are P4-specific.
38:- rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md (cwd=\\?\C:\Users\李则徐\Documents\ChatGPT\EM-FloatFormer, rollout_path=\\?\C:\Users\李则徐\.codex\sessions\2026\11\17\rollout-2026-11-17T20-54-26-01a1eb4a-230f-7e02-b32d-21348996e25b.jsonl, updated_at=2026-11-18T16:17:48+00:00, thread_id=01a1eb4a-230f-7e02-b32d-21348996e25b, formal P4-D and assessment completed)
48:- rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md (cwd=\\?\C:\Users\李则徐\Documents\ChatGPT\EM-FloatFormer, rollout_path=\\?\C:\Users\李则徐\.codex\sessions\2026\11\17\rollout-2026-11-17T20-54-26-01a1eb4a-230f-7e02-b32d-21348996e25b.jsonl, updated_at=2026-11-18T16:17:48+00:00, thread_id=01a1eb4a-230f-7e02-b32d-21348996e25b, public replay verified)
1:thread_id: 01a1eb4a-230f-7e02-b32d-21348996e25b
65:Conclusion: P4 made substantial algorithmic progress by implementing and testing a complex-valued, gauge-equivariant, Hermitian-attention Transformer prototype with physical channel simulation and coded-link evaluation. The strongest evidence is that signal symmetries and complex-valued operations were encoded directly into the model and some mechanisms showed reproducible value in controlled 1D ablations. However, the formal 2D result did not establish a learned-Transformer advantage; the largest 2D improvements came from better pilot/channel front-ends. The next decisive experiment is a strictly matched comparison where EM and classical receivers share the same physical front-end, pilot information, decoder, and inference budget, isolating only the learned module.
```

### Event #60: `item.completed` — `command_execution` (`item_56`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=Join-Path (Get-Location) '"'research/stage4_math_and_design/M6/.uv-cache'; "'$env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$env:TEMP=$env:UV_CACHE_DIR; $env:TMP=$env:UV_CACHE_DIR; $env:PYTHONUTF8='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M6/verify_m6.py"
```

**Aggregated Output:**

```text
VERIFIED results_m6.json: parity max |diff|=3.56e-16, 10,000 trials, cap $9.50
```

### Event #61: `item.completed` — `command_execution` (`item_59`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=Join-Path (Get-Location) '"'research/stage4_math_and_design/M6/.uv-cache'; "'$env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$env:TEMP=$env:UV_CACHE_DIR; $env:TMP=$env:UV_CACHE_DIR; $env:PYTHONUTF8='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B -c \"import ast,json,re; from pathlib import Path; p=Path('research/stage4_math_and_design/M6'); s=(p/'THEOREM_M6.md').read_text(encoding='utf-8'); f=chr(36)*2; assert s.splitlines().count(f)==10; assert sum(line.startswith(chr(96)*3) for line in s.splitlines())==4; ast.parse((p/'verify_m6.py').read_text(encoding='utf-8')); d=json.loads((p/'results_m6.json').read_text(encoding='utf-8')); assert len(d['stage3d']['q3'])==5 and len(d['stage3d']['q2'])==7 and d['sequential']['trials']==10000; assert all(not any(ord(c)<32 and c not in '\\n\\t' for c in f.read_text(encoding='utf-8')) for f in p.glob('*.md')); print('M6 coverage, JSON, Python syntax, and Markdown fences verified')\""
```

**Aggregated Output:**

```text
Traceback (most recent call last):
  File "<string>", line 1, in <module>
AssertionError
```

### Event #62: `item.completed` — `command_execution` (`item_58`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "git diff --name-only; rg -n '"'^[$][$]'"' research/stage4_math_and_design/M6/THEOREM_M6.md; (Get-ChildItem -LiteralPath research/stage4_math_and_design/M6/.uv-cache -File -Recurse | Measure-Object -Sum Length).Sum"
```

**Aggregated Output:**

```text
9:$$
13:$$
19:$$
25:$$
29:$$
34:$$
106:$$
108:$$
122:$$
124:$$
138:$$
140:$$
309853584
```

### Event #63: `item.completed` — `file_change` (`item_60`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M6\\THEOREM_M6.md",
    "kind": "update"
  }
]
```

### Event #64: `item.completed` — `command_execution` (`item_61`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$env:UV_CACHE_DIR=Join-Path (Get-Location) '"'research/stage4_math_and_design/M6/.uv-cache'; "'$env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:UV_LINK_MODE='"'copy'; "'$env:TEMP=$env:UV_CACHE_DIR; $env:TMP=$env:UV_CACHE_DIR; $env:PYTHONUTF8='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B -c \"import ast,json,re; from pathlib import Path; p=Path('research/stage4_math_and_design/M6'); s=(p/'THEOREM_M6.md').read_text(encoding='utf-8'); f=chr(36)*2; assert s.splitlines().count(f)==12; assert sum(line.startswith(chr(96)*3) for line in s.splitlines())==4; ast.parse((p/'verify_m6.py').read_text(encoding='utf-8')); d=json.loads((p/'results_m6.json').read_text(encoding='utf-8')); assert len(d['stage3d']['q3'])==5 and len(d['stage3d']['q2'])==7 and d['sequential']['trials']==10000; assert all(not any(ord(c)<32 and c not in '\\n\\t' for c in f.read_text(encoding='utf-8')) for f in p.glob('*.md')); print('M6 coverage, JSON, Python syntax, and Markdown fences verified')\""
```

**Aggregated Output:**

```text
M6 coverage, JSON, Python syntax, and Markdown fences verified
```

### Event #65: `item.completed` — `agent_message` (`item_62`)

The verifier passed all five Q3 cells and seven Q2 arms, with a maximum summary discrepancy of `3.56×10⁻¹⁶`. I’ve finished the roadmap and executive memo. I’ll remove the temporary uv cache and confirm that `M6/` contains only the four deliverables.

### Event #66: `item.completed` — `command_execution` (`item_63`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$root=(Resolve-Path -LiteralPath '"'research/stage4_math_and_design/M6').Path; "'$target=Get-Item -LiteralPath (Join-Path $root '"'.uv-cache') -Force; if ("'$target.FullName -ne (Join-Path $root '"'.uv-cache')) {throw 'Unexpected cache path'}; if ("'$target.Attributes -band [IO.FileAttributes]::ReparsePoint) {throw '"'Unexpected cache reparse point'}; "'$target | Select-Object FullName,Attributes; uv cache clean --help | Select-Object -First 26'
```

**Aggregated Output:**

```text

FullName                                                                                                       Attribut
                                                                                                                     es
--------                                                                                                       --------
C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M6\.uv-cache ...ctory
Clear the cache, removing all entries or those linked to specific packages

Usage: uv.exe cache clean [OPTIONS] [PACKAGE]...

Arguments:
  [PACKAGE]...  The packages to remove from the cache

Options:
      --force  Force removal of the cache, ignoring in-use checks

Cache options:
  -n, --no-cache               Avoid reading from or writing to the cache, instead using a temporary
                               directory for the duration of the operation [env: UV_NO_CACHE=]
      --cache-dir <CACHE_DIR>  Path to the cache directory [env: UV_CACHE_DIR=]

Python options:
      --managed-python       Require use of uv-managed Python versions [env: UV_MANAGED_PYTHON=]
      --no-managed-python    Disable use of uv-managed Python versions [env: UV_NO_MANAGED_PYTHON=]
      --no-python-downloads  Disable automatic downloads of Python. [env:
                             "UV_PYTHON_DOWNLOADS=never"]

Global options:
  -q, --quiet...
          Use quiet output
  -v, --verbose...
          Use verbose output


```

### Event #67: `item.completed` — `command_execution` (`item_64`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'uv cache clean --cache-dir research/stage4_math_and_design/M6/.uv-cache; if ($LASTEXITCODE -ne 0) {exit $LASTEXITCODE}; Get-ChildItem -LiteralPath research/stage4_math_and_design/M6 -Force | Select-Object Name,Length; git diff --name-only'
```

**Aggregated Output:**

```text
Clearing cache at: research/stage4_math_and_design/M6/.uv-cache
Removed 4952 files (295.5MiB)

Name            Length
----            ------
reply_r1.md       4925
results_m6.json  28546
THEOREM_M6.md    46447
verify_m6.py     19727


```

### Event #68: `item.completed` — `agent_message` (`item_65`)

**M6 Round 1 is complete.** The Stage 3D replay puts the largest measured loss after the cache. On G1, Full IQ scored **64.641% / 0.929038 nats**, s17 scored **64.617% / 0.941126**, and Qwen phase scored **23.799% / 1.445229**. The increments were **0.012088 nats** from Full IQ to s17, **0.478026** from s17 to unit-temperature Qwen, and **0.026077** from applying the locked temperature. The archive leaves the separate connector and host information losses unidentified.

On Q2 C4, a fresh reference improved the physical readout from **66.406% to 74.219%** while Qwen phase fell from **28.776% to 26.432%**. The verifier covered all five Q3 cells and seven Q2 arms, with a maximum summary discrepancy of **3.56×10⁻¹⁶**.

- **Tier 0:** Establish the causal cache-risk bounds and enforce rotation, deduplication, masking, and padding invariants.
- **Tier 1:** Check held-out cache–phase interactions and the actual frozen host’s candidate-score Jacobian.
- **Tier 2:** Cap the pilot at **173 of 960 steps** on one seed. Continue to three seeds and independent blind scoring only after the pilot passes.

The **10,000-trial simulation** reduced modeled mean cost from **$17.12 to $6.10**, a **64.36%** reduction, with **93.28%** working-architecture power. Under the specified Gaussian model, the worst-case Type-I error was **4.4695%** with binding futility and **5.0000%** with nonbinding futility. The proposed Stage 4 cap is **$9.50**, leaving **$3.38** against the historical credit balance. Actual Qwen geometry, real-data risk bounds, and scoring throughput remain launch gates. Round 1 spent **$0.00** on cloud compute. [Modal pricing](https://modal.com/pricing)

[Executive memo and M1–M5 questions](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/reply_r1.md>) · [Theorems, budget, and Stage 4–6 DAG](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/THEOREM_M6.md>) · [Executed verifier](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/verify_m6.py>) · [Full results](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/results_m6.json>)



### Event #69: `turn.completed`

- Usage: `{"input_tokens": 4540110, "cached_input_tokens": 4379648, "cache_write_input_tokens": 0, "output_tokens": 41681, "reasoning_output_tokens": 7161}`


## Round 1 — Final Output (`reply_r1.md`)

```markdown
**M6 Round 1 is complete.** The Stage 3D replay puts the largest measured loss after the cache. On G1, Full IQ scored **64.641% / 0.929038 nats**, s17 scored **64.617% / 0.941126**, and Qwen phase scored **23.799% / 1.445229**. The increments were **0.012088 nats** from Full IQ to s17, **0.478026** from s17 to unit-temperature Qwen, and **0.026077** from applying the locked temperature. The archive leaves the separate connector and host information losses unidentified.

On Q2 C4, a fresh reference improved the physical readout from **66.406% to 74.219%** while Qwen phase fell from **28.776% to 26.432%**. The verifier covered all five Q3 cells and seven Q2 arms, with a maximum summary discrepancy of **3.56×10⁻¹⁶**.

- **Tier 0:** Establish the causal cache-risk bounds and enforce rotation, deduplication, masking, and padding invariants.
- **Tier 1:** Check held-out cache–phase interactions and the actual frozen host’s candidate-score Jacobian.
- **Tier 2:** Cap the pilot at **173 of 960 steps** on one seed. Continue to three seeds and independent blind scoring only after the pilot passes.

The **10,000-trial simulation** reduced modeled mean cost from **$17.12 to $6.10**, a **64.36%** reduction, with **93.28%** working-architecture power. Under the specified Gaussian model, the worst-case Type-I error was **4.4695%** with binding futility and **5.0000%** with nonbinding futility. The proposed Stage 4 cap is **$9.50**, leaving **$3.38** against the historical credit balance. Actual Qwen geometry, real-data risk bounds, and scoring throughput remain launch gates. Round 1 spent **$0.00** on cloud compute. [Modal pricing](https://modal.com/pricing)

[Executive memo and M1–M5 questions](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/reply_r1.md>) · [Theorems, budget, and Stage 4–6 DAG](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/THEOREM_M6.md>) · [Executed verifier](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/verify_m6.py>) · [Full results](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/results_m6.json>)

```


---

## Round 2 — Cross-Review & Convergence Prompt (`prompt_r2.md`)

```markdown
# Track M6 (Round 2): Final Converged Tier-0/1/2 Ledger, M1/M4 Reconciliation & Master Stage 4–6 Execution DAG

You are resuming persistent Codex session **Track M6** (`01a2194c-3b55-7c13-89f6-3ed11a482246`) in `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`. Keep edits STRICTLY inside `research/stage4_math_and_design/M6/` and run Python ONLY via `uv run --python 3.12 ...`.

## 1. Full Round-1 Cross-Track Picture to Synthesize & Lock
Look across ALL 5 companion tracks (`M1..M5`) + your own Round 1 analysis:
1. **The Reconciled Hardware & Statistical Budget Paradox (`M2` vs `M4` vs `M6`)**:
   - **VRAM reality from Stage 3D (`M4` catch)**: Stage 3D measured **`52,972.5 MB` peak train VRAM** (microbatch 16) and **`45,409.9 MB` peak eval VRAM** (batch 64) on `A100-80GB`! Therefore putting Pluto (`2,240` frames) + CTTC (`1,440` packets) blindly onto `L4 (24 GB)` in M6's baseline cap risks kernel/batch mismatch or OOM unless a small `$0.10` L4 parity smoke passes first!
   - **AND look at M4's A100-80GB math**: Pure **A100-80GB** across `4` matched single-seed arms (`384` steps/arm, synthetic + real) + `131,760` eval rows STILL totals only **`$8.4407`** (leaving `$4.44` below `$12.88`!). Wait — look carefully at minor rate/step differences between M4 (`A100-80GB $2.4984/h + 2c/12GiB = $0.00074684/s`, single seed `4 × 384 steps`, `2,477s` raw real, `1,229s` synth) vs M6 (`A100-80GB + 2c/16GiB = $0.00075572/s`, `173-step` pilot + `3` seeds continuation for ONE winning arm = `2,707 steps`).
   - **AND look at M2's statistical power theorem**: `3` seeds CANNOT establish population-of-seeds generalizability anyway (`~22–30%` crossed power, requiring `>= 10` seeds!), while `1` seed or `3` frozen seeds cleanly establish **Conditional Fixed-Connector Block/Capture/Loop-Pair Statistical Claims** (PLUS counterfactual fixed-text slot interventions & physical+residual comparisons)!
   - **Crucial Synthesis Decision Needed From M6**: Merge M4's VRAM-safe pure `A100-80GB` safety with M6's Tier-2 **1-seed pilot futility gate** (`18%` steps on disjoint dev BEFORE buying full `384`-step × `4`-arm comparison OR continuing `Arm 1 / Arm 2` to extra blocks/seeds!), keeping TOTAL worst-case cap **strictly `<= $9.50`** inside `~$12.88` remaining Modal free credits (`$0.00` net spend guard)!
2. **Causal Isolation (`Arm 0: Zero`, `Arm 0b: Naive-Additive`, `Arm 1: Host-Pure Equivariant+Coupled`, `Arm 2: Equivariant-Physical+HostResidual`)**:
   - M1's Round 1 showed `Arm 2` hits `64.62%` via its physical S17 prior + host residual, which M2 noted fails to isolate how much frozen Qwen contributes vs physical bypass!
   - Update your Tier-1 CPU Gate & Stage 4 DAG to explicitly require **BOTH** `Arm 1 (Host-Pure)` AND `Arm 2 (Hybrid)` Jacobian/OOD checks AND counterfactual slot/text controls!
3. **Dual-Cache Zero-GPU Upgrade (`4-bit s17` + `16-bit canonical` M5 Packet) + `RefSufficientPool` Non-Double-Counting (M3+M4)**:
   - M4's CPU shard builder already runs R3 whitening & feature computation on CPU (`<$0.77` total CPU cost!), so extracting BOTH `4-bit s17` AND M5's `16-bit canonical` packet simultaneously on CPU locks zero throwaway work across **Stage 4 → Stage 5 → Stage 6**!
   - Also integrate M4's critical causal discovery (`fast_blind_frontend` & `anchor_capture` non-causal future/whole-source correlations excluded -> strict causal filtration $\mathcal{F}^r$ + sentinel hash test + M3 non-double-counting!).

## 2. Specific Round-2 Convergence Tasks for Track M6
Update `research/stage4_math_and_design/M6/THEOREM_M6.md`, `verify_m6.py`, and `results_m6.json`:
1. **Publish ONE Converged, Contradiction-Free Master Modal Allocation Ledger (`<= $9.50` Hard Cap, 100% A100-80GB Safe)** unifying M4's `52.97 GB / 45.41 GB` VRAM numbers, M1's 4-arm causal matrix, M2's fixed-seed conditional inference scope + dev-disjoint futility stopping, and M5's dual-cache Stage 4→6 export! Explicitly show how much budget goes to Stage 4 Pilot, Arm Comparison, Real Pluto (`2,240`), Real CTTC (`1,440` incl. held-out `f708_snr10`), and Reserve (`>= $3.38` buffer vs `$12.88`!).
2. **Lock the Exact Numerical Tier-0 / Tier-1 / Tier-2 Go-No-Go Checklists** (ready for Antigravity to inspect and execute immediately after GitHub commit, with exact pass/fail predicates across every node `4A..6E` incorporating M1–M5 invariants!).
3. **Cross-Host & Fair `LEMMO` Execution Law (Stage 6 Readiness)**: Explicitly verify mathematically how fixing the `16-bit canonical` cache + `RefSufficientPool` lets us swap only tiny host adapters (`4B -> 9B -> 27B`) and run exact matched-support `LEMMO` comparison with minimum redundant experiments!
4. Run updated `verify_m6.py` via `uv run`, verifying all parity + Monte Carlo sequential cost/Type-I + converged Modal A100 ledger numbers cleanly in `results_m6.json`, and summarize in `reply_r2.md`.
```


## Round 2 — Resumed Execution & Reasoning Trace (`round2_events.jsonl`)

### Event #1: `thread.started` (thread_id=`01a2194c-3d43-7c13-89f6-3ed11a482246`)

### Event #2: `turn.started`

### Event #3: `item.completed` — `agent_message` (`item_0`)

I’ll read the five companion tracks and the existing M6 verifier, then reconcile the arm definitions, statistical claims, and A100 time limits into one ledger. I’ll replay the updated verifier locally before closing Round 2.

### Event #4: `item.completed` — `command_execution` (`item_2`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n -i 'floatformer|stage3d|frozen protocol' (Join-Path "'$env:USERPROFILE '"'.codex/memories/MEMORY.md'); git status --short; rg --files --hidden research/stage4_math_and_design -g '*.md' -g '*.py' -g 'results*.json' -g '"'!**/round*events*'"' -g '"'!**/.uv-cache/**'"'"
```

**Aggregated Output:**

```text
29:# Task Group: EM-FloatFormer P4 scientific assessment and reproducible publication
32:applies_to: cwd=C:\Users\李则徐\Documents\ChatGPT\EM-FloatFormer; reuse_rule=Reuse the matched-control, reproducibility, and publication approach for this repository; numerical results, commit, hardware status, and receiver outcomes are P4-specific.
38:- rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md (cwd=\\?\C:\Users\李则徐\Documents\ChatGPT\EM-FloatFormer, rollout_path=\\?\C:\Users\李则徐\.codex\sessions\2026\11\17\rollout-2026-11-17T20-54-26-01a1eb4a-230f-7e02-b32d-21348996e25b.jsonl, updated_at=2026-11-18T16:17:48+00:00, thread_id=01a1eb4a-230f-7e02-b32d-21348996e25b, formal P4-D and assessment completed)
42:- EM-FloatFormer, P4, EMFloatFormer2D, C*-equivariance, Hermitian attention, complex RoPE, sparse DMRS, Doppler, CFO, Rapp PA, LDPC, P4D, Holm, sign-flip
48:- rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md (cwd=\\?\C:\Users\李则徐\Documents\ChatGPT\EM-FloatFormer, rollout_path=\\?\C:\Users\李则徐\.codex\sessions\2026\11\17\rollout-2026-11-17T20-54-26-01a1eb4a-230f-7e02-b32d-21348996e25b.jsonl, updated_at=2026-11-18T16:17:48+00:00, thread_id=01a1eb4a-230f-7e02-b32d-21348996e25b, public replay verified)
52:- 20171806011/EM-FloatFormer, 8380084c6dece64119ab72f9c3c1c212e154fb17, publication.verify_p4, p4d_original_audit.py, public_redactions.json, 150 coded rows, ten weight hashes
76:applies_to: cwd=C:\Users\李则徐\Documents\ChatGPT\EM-FloatFormer; reuse_rule=Reuse the diagnostic and UI verification approach; country eligibility, account status, prices, and dates are time- and account-specific.
82:- rollout_summaries/2026-11-17T21-38-49-uldk-google_checkout_debug_youtube_trial_cancellation.md (cwd=\\?\C:\Users\李则徐\Documents\ChatGPT\EM-FloatFormer, rollout_path=C:\Users\李则徐\.codex\sessions\2026\11\17\rollout-2026-11-17T21-38-49-01a1eb72-c399-7cf2-8270-2061ad0e43de.jsonl, updated_at=2026-11-17T21:57:10+00:00, thread_id=01a1eb72-c399-7cf2-8270-2061ad0e43de, logged-in error not observed)
92:- rollout_summaries/2026-11-17T21-38-49-uldk-google_checkout_debug_youtube_trial_cancellation.md (cwd=\\?\C:\Users\李则徐\Documents\ChatGPT\EM-FloatFormer, rollout_path=C:\Users\李则徐\.codex\sessions\2026\11\17\rollout-2026-11-17T21-38-49-01a1eb72-c399-7cf2-8270-2061ad0e43de.jsonl, updated_at=2026-11-17T21:57:10+00:00, thread_id=01a1eb72-c399-7cf2-8270-2061ad0e43de, cancellation verified)
204:# Task Group: Local TAN state-mechanism research, frozen protocol, and trace-pack stop
206:scope: Local CPU-only DEND/TAN research with per-round planning, frozen protocols, independent audit, and a final equivalent trace-pack candidate that did not establish a latency gain.
?? research/stage4_math_and_design/
research/stage4_math_and_design\transcript_utils.py
research/stage4_math_and_design\M6\verify_m6.py
research/stage4_math_and_design\M6\THEOREM_M6.md
research/stage4_math_and_design\M6\results_m6.json
research/stage4_math_and_design\M6\reply_r1.md
research/stage4_math_and_design\M5\verify_m5.py
research/stage4_math_and_design\M5\THEOREM_M5.md
research/stage4_math_and_design\M5\results_m5.json
research/stage4_math_and_design\M5\reply_r1.md
research/stage4_math_and_design\M3\verify_m3.py
research/stage4_math_and_design\M3\THEOREM_M3.md
research/stage4_math_and_design\M3\results_m3.json
research/stage4_math_and_design\M3\reply_r1.md
research/stage4_math_and_design\M4\verify_m4.py
research/stage4_math_and_design\M4\THEOREM_M4.md
research/stage4_math_and_design\M4\results_m4.json
research/stage4_math_and_design\M4\reply_r1.md
research/stage4_math_and_design\M2\verify_m2.py
research/stage4_math_and_design\discussions\M6\reply_r1.md
research/stage4_math_and_design\discussions\M6\prompt_r2.md
research/stage4_math_and_design\M2\THEOREM_M2.md
research/stage4_math_and_design\discussions\M6\prompt_r1.md
research/stage4_math_and_design\M2\results_m2.json
research/stage4_math_and_design\M2\reply_r1.md
research/stage4_math_and_design\M1\verify_m1.py
research/stage4_math_and_design\M1\THEOREM_M1.md
research/stage4_math_and_design\M1\results_m1.json
research/stage4_math_and_design\discussions\M5\reply_r1.md
research/stage4_math_and_design\discussions\M5\prompt_r2.md
research/stage4_math_and_design\discussions\M5\prompt_r1.md
research/stage4_math_and_design\discussions\M4\reply_r1.md
research/stage4_math_and_design\discussions\M4\prompt_r2.md
research/stage4_math_and_design\discussions\M4\prompt_r1.md
research/stage4_math_and_design\discussions\M2\reply_r1.md
research/stage4_math_and_design\discussions\M2\prompt_r2.md
research/stage4_math_and_design\discussions\M2\prompt_r1.md
research/stage4_math_and_design\discussions\M1\prompt_r2.md
research/stage4_math_and_design\discussions\M1\prompt_r1.md
research/stage4_math_and_design\discussions\M1\reply_r1.md
research/stage4_math_and_design\discussions\M3\reply_r1.md
research/stage4_math_and_design\discussions\M3\prompt_r2.md
research/stage4_math_and_design\discussions\M3\prompt_r1.md
```

### Event #5: `item.completed` — `command_execution` (`item_1`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-ChildItem -Force research/stage4_math_and_design/M6; rg -n '"'^#|''^def '"' research/stage4_math_and_design/M6; Get-Item ../AGENTS.md -ErrorAction SilentlyContinue | Select-Object FullName; Get-Item AGENTS.md -ErrorAction SilentlyContinue | Select-Object FullName; Get-ChildItem research/stage4_math_and_design -Recurse -Filter AGENTS.md"
```

**Aggregated Output:**

```text


    目录: C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M6


Mode                 LastWriteTime         Length Name                                                                 
----                 -------------         ------ ----                                                                 
-a----        2026/11/26     19:33           4925 reply_r1.md                                                          
-a----        2026/11/26     19:36          28546 results_m6.json                                                      
-a----        2026/11/26     19:35          46447 THEOREM_M6.md                                                        
-a----        2026/11/26     19:34          19727 verify_m6.py                                                         
research/stage4_math_and_design/M6\reply_r1.md:1:# M6 Round 1 • Decision for the Stage 4 launch
research/stage4_math_and_design/M6\reply_r1.md:15:## Release gates
research/stage4_math_and_design/M6\reply_r1.md:23:## Questions for M1–M5, Round 2
research/stage4_math_and_design/M6\verify_m6.py:33:def probabilities(scores, temperature=1.0):
research/stage4_math_and_design/M6\verify_m6.py:38:def score(probs, labels):
research/stage4_math_and_design/M6\verify_m6.py:51:def check_metrics(actual, expected, diffs):
research/stage4_math_and_design/M6\verify_m6.py:58:def paired_block_ci(loss_difference):
research/stage4_math_and_design/M6\verify_m6.py:67:def replay_q3(data, summary, diffs):
research/stage4_math_and_design/M6\verify_m6.py:129:def replay_q2(data, summary, diffs):
research/stage4_math_and_design/M6\verify_m6.py:173:def cpu_geometry():
research/stage4_math_and_design/M6\verify_m6.py:205:def sequential_design():
research/stage4_math_and_design/M6\verify_m6.py:310:def budget():
research/stage4_math_and_design/M6\verify_m6.py:342:def main():
research/stage4_math_and_design/M6\THEOREM_M6.md:1:# M6 Round 1 • CPU certificates and a capped Stage 4 experiment
research/stage4_math_and_design/M6\THEOREM_M6.md:5:## 1. The exact population loss ladder
research/stage4_math_and_design/M6\THEOREM_M6.md:38:### What the Stage 3D archive identifies
research/stage4_math_and_design/M6\THEOREM_M6.md:66:## 2. Tier 0 • analytic rejection before any host work
research/stage4_math_and_design/M6\THEOREM_M6.md:68:### 2.1 Bayes-risk brackets for each evidence state
research/stage4_math_and_design/M6\THEOREM_M6.md:80:### 2.2 Exact group action and reference algebra
research/stage4_math_and_design/M6\THEOREM_M6.md:88:## 3. Tier 1 • CPU geometry and the frozen host
research/stage4_math_and_design/M6\THEOREM_M6.md:94:### 3.1 Rank, curvature and optimization certificates
research/stage4_math_and_design/M6\THEOREM_M6.md:102:### 3.2 Geometry of $W_{\rm out}$ and vocabulary readout
research/stage4_math_and_design/M6\THEOREM_M6.md:116:## 4. Tier 2 • sequential decisions and an exact cost envelope
research/stage4_math_and_design/M6\THEOREM_M6.md:118:### 4.1 VoI and continuation
research/stage4_math_and_design/M6\THEOREM_M6.md:128:### 4.2 Two-look design and what the Monte Carlo verifies
research/stage4_math_and_design/M6\THEOREM_M6.md:154:### 4.3 Stage 4 Modal plan in US dollars
research/stage4_math_and_design/M6\THEOREM_M6.md:178:## 5. Stage 4 → 6 critical path
research/stage4_math_and_design/M6\THEOREM_M6.md:215:## Replay

FullName : C:\Users\李则徐\.gemini\antigravity\scratch\AGENTS.md



```

### Event #6: `item.completed` — `web_search` (`exec-64db46ae-43b9-4c1b-a327-a55920c8bcf8`)

```json
{
  "id": "exec-64db46ae-43b9-4c1b-a327-a55920c8bcf8",
  "type": "web_search",
  "query": "https://modal.com/pricing",
  "action": {
    "type": "open_page",
    "url": "https://modal.com/pricing"
  },
  "results": [
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn6view0",
      "snippet": "Total lines: 391",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/NPjE9YzmwLkA0JICm9WeVVE4dwCeqKJTg0CRzaoupoY_hlVPBQPIXJWgDeKCrpWJbyVdlbiWI3Kz5i8REzXocA",
      "title": "Pricing: Pay per second for GPUs and CPUs | Modal",
      "url": "https://modal.com/pricing"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn6view1",
      "snippet": "Total lines: 84",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/4obuMKXc-dUc1hdIJZ13mxkuozIW2s2dyX491ylno6BY1Rx2JJ7mw2VSnxJB_-H_YEdrwaCZnacC9mZa0SUiMotGZ8KhxKsm5JAeHrsRw7FgbIKoKBiMnJwiahBx6Q9nHeZP2J0L2L4sv6zwhxHe8syy4aTvg1medB5jBmc3_F8",
      "title": "Configuring CPU, memory, and disk | Modal Docs",
      "url": "https://modal.com/docs/guide/resources"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn6view2",
      "snippet": "Total lines: 51",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/5p-lg_s-oLZkDoRlwPovapMxJzmQfcStPIaanGo3mqYrEuhdMXpC4j0PGpJhY4p794TMAuudM83gpZWNqFkAIkxX-hSmBL90Ni0vfHVeWIO_CsaBCUB6PwNCrS6WZ-hb",
      "title": "Budgets | Modal Docs",
      "url": "https://modal.com/docs/guide/budgets"
    },
    {
      "type": "text_result",
      "domain": "arxiv.org",
      "ref_id": "turn6view3",
      "snippet": "Total lines: 551",
      "title": "LEMMO: A Large Electromagnetic Model leveraging Pretrained LLMs for Million-Sample Signal Understanding",
      "url": "https://arxiv.org/html/2611.09476v1"
    },
    {
      "type": "text_result",
      "domain": "github.com",
      "ref_id": "turn6view4",
      "snippet": "Total lines: 281",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/MWwLn2aUzWqEmo1dnOm_bL0dSBCO6dH9lNs-ie1TkGexl90TrrhoSZnSjun_LTnH9cBYtJ4LNnxN6_VYIPAsIblw-X9fEzvRJVGMdT7v4ieV5jf9cqrSvNZzgpchSijnb9KrwrNe0T75upLDpdmLj67RjEc7-12u4J3E68jLPrMtbPisSwu3Mjnjx1wQRuNs",
      "title": "GitHub - N1rv4n4/LEMMO: Implementation of LEMMO, an electromagnetic large model for raw I/Q signal recognition and description · GitHub",
      "url": "https://github.com/N1rv4n4/LEMMO"
    }
  ]
}
```

### Event #7: `item.completed` — `command_execution` (`item_4`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '"'^(##|###)|0'"\\.00074684|8\\.4407|2477|1229|384|f708_snr10|159|131760|131,760|futil|sentinel' research/stage4_math_and_design/M1/THEOREM_M1.md research/stage4_math_and_design/M2/THEOREM_M2.md research/stage4_math_and_design/M3/THEOREM_M3.md research/stage4_math_and_design/M4/THEOREM_M4.md research/stage4_math_and_design/M5/THEOREM_M5.md"
```

**Aggregated Output:**

```text
research/stage4_math_and_design/M5/THEOREM_M5.md:5:## 1. Minimal sufficiency for the task union
research/stage4_math_and_design/M5/THEOREM_M5.md:27:### 1.1 Exact Gaussian experiment
research/stage4_math_and_design/M5/THEOREM_M5.md:39:### 1.2 Noisy generalization of the 2-bit versus 4-bit pair
research/stage4_math_and_design/M5/THEOREM_M5.md:67:## 2. Exact operational rate–risk frontier
research/stage4_math_and_design/M5/THEOREM_M5.md:96:### 2.1 Continuous trajectories and identifiable events
research/stage4_math_and_design/M5/THEOREM_M5.md:102:## 3. Audit of s17 and the shared cache format
research/stage4_math_and_design/M5/THEOREM_M5.md:112:### 3.1 Bit-exact extension
research/stage4_math_and_design/M5/THEOREM_M5.md:129:### 3.2 Stage 4→6 connector contract
research/stage4_math_and_design/M5/THEOREM_M5.md:140:## 4. One-window Bayes EIG
research/stage4_math_and_design/M5/THEOREM_M5.md:170:## 5. CPU verification
research/stage4_math_and_design/M5/THEOREM_M5.md:176:### 5.1 Achieved rate–risk points
research/stage4_math_and_design/M5/THEOREM_M5.md:193:### 5.2 Sealed one-window selection
research/stage4_math_and_design/M5/THEOREM_M5.md:207:### 5.3 Rerun and interpretation
research/stage4_math_and_design/M5/THEOREM_M5.md:217:## 6. Round 2 handoff
research/stage4_math_and_design/M4/THEOREM_M4.md:7:## 1. Observation laws and losses
research/stage4_math_and_design/M4/THEOREM_M4.md:9:### 1.1 The three representations
research/stage4_math_and_design/M4/THEOREM_M4.md:29:### 1.2 Conditional KL and perturbation terms
research/stage4_math_and_design/M4/THEOREM_M4.md:74:### 1.3 Student scale, Rayleigh scale, and the frozen nonlinearity
research/stage4_math_and_design/M4/THEOREM_M4.md:112:## 2. Hardware benchmark and causality contract
research/stage4_math_and_design/M4/THEOREM_M4.md:114:### 2.1 Candidate rows and time order
research/stage4_math_and_design/M4/THEOREM_M4.md:125:The primary samples retain actual fading, residual DC/ISI, noise, clock offset, and any changes present at registered times. A naturally occurring jump is a stratum, with changes flagged by a predeclared causal split-local pilot innovation criterion and a retrospective scorer-only morphology check. A CPU-only cross-frame Q1 diagnostic pairs `(f−1,f)` at `f mod 20 = 0` within each test block, using the earlier frame's \(k_1\) and current frame's \(k_1\) from the table. Its matched control pairs the preceding two interior frames `(f−2,f−1)` in that same block. Use only available sync and authorized pilots and a gap-class prior when the time between Pluto columns is unavailable. This samples recorded batch boundaries by schedule without requiring or asserting a jump at each boundary. Separate source domains and nominal session strata in analysis, keeping capture IDs out of model text/features. Hold `f708_snr10` out of fitted connector weights, temperatures and shared prior tuning to expose one capture transfer; allow its own causal sync and released pilots. Pluto is a blocked same-campaign test. CTTC has one public cyclic TX sequence; input-source disjointness and capture dependence are distinct issues. Use block-paired NLL/SER differences on 16 Pluto groups and 72 CTTC session-loops, group adjacent repeat pairs for CTTC Q3, and report a capture-level sensitivity analysis and source-ordinal sensitivity because loops/captures reuse source positions. Report all registered rows and lock coverage, with the same uniform predictive fallback on failed rows and a secondary locked-only analysis. Avoid treating all packets as independent. Select the Q2 `two−one` proper-loss contrast and large-angle synthetic Q3 coupled-vs-additive contrast before test, with paired intervals and the multiplicity rule frozen with M2.
research/stage4_math_and_design/M4/THEOREM_M4.md:129:### 2.2 What zero leakage requires and what was verified
research/stage4_math_and_design/M4/THEOREM_M4.md:143:Within a frozen frequency prior, keep multiple plausible modes and their masses or a causal filter posterior for frequency, phase and possible jump. When rank or timing confidence fails, set the common missingness state and use a declared prior. Without a held-in true rate, \(\epsilon_\omega=\omega_{\rm true}-\widehat\omega\) is latent. A receiver can estimate, propagate and calibrate uncertainty and innovations. For known amplitude and phase design, the efficient information for \(\omega\) with unknown complex gain in proper white noise is \(2|H|^2\sum_p |s_p|^2(t_p-\bar t)^2/v\), with energy-weighted mean time. With unknown DC, project \(iHt_p s_p e^{i\omega t_p}\) onto the whitened orthogonal complement of both nuisance columns and take twice its squared norm. This follows from the Schur complement of the Gaussian Fisher matrix and gives a local unbiased CRLB, not a bound guaranteeing tracking of phase jumps or unwrapping. Irregular gaps and 1-second CTTC prediction require propagating the circular phase law across the actual lag. Never fit from Q3, `HOLDOUT`, full waveform repetitions, payload decisions or injection flags. Freeze real-split provenance and compare hashes after replacing future/probe IQ and all scorer-only columns with sentinels. Outputs, caches and feature hashes must agree. Enumerate the transitive support ledger as well as this mutation test. The CPU replay checks archived geometry and designed local support. It did not run this acquisition-level gate.
research/stage4_math_and_design/M4/THEOREM_M4.md:145:### 2.3 Same-row ablations and pooling
research/stage4_math_and_design/M4/THEOREM_M4.md:158:## 3. Frozen compute envelope and dispatch
research/stage4_math_and_design/M4/THEOREM_M4.md:160:### 3.1 Dataset inputs and handoffs
research/stage4_math_and_design/M4/THEOREM_M4.md:172:### 3.2 Model and scored workload
research/stage4_math_and_design/M4/THEOREM_M4.md:175:- Stage 3D measured `1.9757 s / optimizer step` and `23.88 ms / logical eval row` on one A100-80GB. Train and eval peaks were `52,972.5 MB` at microbatch 16 and `45,409.9 MB` at eval batch 64. The six 1,280-step runs took `2,884.0–3,217.8 s` including additional work. BF16 weights alone are about 8 GB at 4B parameters. Train activation and batched candidate workspace amplify this. The baseline plan uses A100-80GB for both. Accept the pilot run only if peak allocated/reserved fits a predeclared 65 GiB A100 threshold and sustained times meet `≤2.50 s/step`, `≤0.030 s/row` inclusive of in-batch data feeding. Rerun timing after any microbatch change. A100-40GB, L4 and A10/A10G require a separate shape and kernel check with smaller batches. The measured Stage 3D peaks exceed their VRAM. For a substitution, require equivalent logits, a ≤20 GiB peak on 24 GiB cards, and throughput below `0.030*(0.00074684/(GPU rate + 0.00005284))` seconds per row to match A100 costs, as well as a preapproved wall limit. This is about 81.5 ms on L4 and 62.4 ms on A10, with 2 cores and 12 GiB host RAM. Keep the confirmed A100 path if the smaller card would require quantizing the frozen host or invalidating kernels. [Stage 3D report](../../stage3d_qwen_signed_phase/REPORT.md) · [Modal GPU SKUs](https://modal.com/docs/guide/gpu)
research/stage4_math_and_design/M4/THEOREM_M4.md:176:- Train four matched connectors with one seed (`202611`), `384×32=12,288` logical examples per arm, mixing equal synthetic/real examples across the steps, using a frozen sampling manifest. Stage 3D locked the base training scorer and LR `0.0015`; use its temperature protocol on disjoint dev data and cap the extra dev scoring at 4,000 rows per arm. This is a single-seed resource-bounded replication tier. Freeze common architecture/parameter scales, prompt length cap and calibration before blind scoring.
research/stage4_math_and_design/M4/THEOREM_M4.md:180:### 3.3 Budget arithmetic and zero-bill gate
research/stage4_math_and_design/M4/THEOREM_M4.md:182:Public Modal list prices checked for this round are A100-80GB `$2.4984/h`, A100-40GB `$2.0988/h`, L4 `$0.7992/h` and A10 `$1.1016/h`, plus CPU `$0.04716/physical-core-h` and memory `$0.007992/GiB-h`. The A100 Function cost with 2 physical cores and 12 GiB is `$0.00074684/s` or `$2.688624/h`. A10 is the pricing label and `A10G` appears in Modal examples. The main schedule budgets the A100 and assumes no savings from smaller devices. [Modal pricing](https://modal.com/pricing)
research/stage4_math_and_design/M4/THEOREM_M4.md:186:| A100 connector fit, `4×384×2.50` | 3,840 | 2.86786560 |
research/stage4_math_and_design/M4/THEOREM_M4.md:196:| **Total**, 2.8538889 A100 hours and 3 CPU function hours | | **8.44073416** |
research/stage4_math_and_design/M4/THEOREM_M4.md:202:## 4. CPU replay and interpretation
research/stage4_math_and_design/M1/THEOREM_M1.md:7:## 1. Exact discrete symmetry
research/stage4_math_and_design/M1/THEOREM_M1.md:19:### Theorem 1 — Encoder and conditional choice actions
research/stage4_math_and_design/M1/THEOREM_M1.md:37:### Theorem 2 — Quotient factorization
research/stage4_math_and_design/M1/THEOREM_M1.md:49:## 2. Continuous phase transport and the sector dip
research/stage4_math_and_design/M1/THEOREM_M1.md:51:### Theorem 3 — Circular phase evolution before the sector map
research/stage4_math_and_design/M1/THEOREM_M1.md:64:### Theorem 4 — Circulant Q3 pushforward
research/stage4_math_and_design/M1/THEOREM_M1.md:94:### Corollary — Why \(\pi/4\) and \(\pi/2\) give different accuracies
research/stage4_math_and_design/M1/THEOREM_M1.md:116:## 3. Stage 3D diagnostics and the scope of separation
research/stage4_math_and_design/M1/THEOREM_M1.md:125:| G4 | 0.80 | 33.29% / 1.3347 | 34.96% / 1.3098 | 47.75% / 1.0159 |
research/stage4_math_and_design/M1/THEOREM_M1.md:129:### Theorem 5 — Additive centered-logit obstruction
research/stage4_math_and_design/M1/THEOREM_M1.md:137:#### Work F spectrum
research/stage4_math_and_design/M1/THEOREM_M1.md:148:### Theorem 6 — Reynolds averaging and NLL
research/stage4_math_and_design/M1/THEOREM_M1.md:160:## 4. Stage 4 `EquivariantEvidenceConnector`
research/stage4_math_and_design/M1/THEOREM_M1.md:170:### M3 interface — `RefSufficientPool`
research/stage4_math_and_design/M1/THEOREM_M1.md:176:### M4 interface — `PhaseContext` for Pluto and CTTC
research/stage4_math_and_design/M1/THEOREM_M1.md:182:## 5. CPU verification and Round 2 decisions
research/stage4_math_and_design/M1/THEOREM_M1.md:197:| G4 | 46.10% / 1.0748 | 47.75% / 1.0167 | 47.75% / 1.0159 | 43.07% |
research/stage4_math_and_design/M3/THEOREM_M3.md:3:## 1. Stage 3 release audit
research/stage4_math_and_design/M3/THEOREM_M3.md:11:## 2. Theorem 1 | Same-clock sufficiency and minimality
research/stage4_math_and_design/M3/THEOREM_M3.md:39:### Exact Q2 posterior
research/stage4_math_and_design/M3/THEOREM_M3.md:71:## 3. Theorem 2 | Distinct clocks require a Wiener bridge
research/stage4_math_and_design/M3/THEOREM_M3.md:111:### Error of a common-phase approximation
research/stage4_math_and_design/M3/THEOREM_M3.md:122:### Q3_after_pilot posterior
research/stage4_math_and_design/M3/THEOREM_M3.md:136:## 4. Stage 3D reference-fusion diagnosis
research/stage4_math_and_design/M3/THEOREM_M3.md:158:## 5. Fixed-slot interface and guarantees
research/stage4_math_and_design/M3/THEOREM_M3.md:173:## 6. CPU replay and paired experiments
research/stage4_math_and_design/M3/THEOREM_M3.md:181:The script replays the archived physical and host probabilities. A fixed-slot CPU surrogate fits \(a_{z_1}\) on 32,768 distinct `NS_CFO_TRAIN` C4 mothers with quarter-turn tying and a symmetric half-count prior. Its second-window phase emission comes from separate Stage 3C-style importance calibration. The 512 blind labels enter only scoring. The multi-clock experiment uses 16,384 new paired mothers at \(t_p=(1,7)\), \(h=2\), \(\omega=+0.35\) and 2 dB, with common draws across methods and diffusion settings. It compares the full Brownian model, the optimal coarsened-data posterior under the same model, and a coherent common-phase approximation. The first low-diffusion observed-label result was inconclusive. A single independent follow-up at 524,288 mothers resolved that contrast, with the initial result retained. Conditional model NLL in the receipt averages over the full fitted posterior for each observation and complements the independently observed-label NLL. The main experiment uses order 40 per pilot innovation, checks order 64 on the 128 highest-pilot-magnitude cases, and takes the second-window emission from 513 periodic knots with order 20 for the inner bridge. The follow-up uses order 12 per innovation and checks against order 40 on 128 high-magnitude cases. [Script](verify_m3.py) · [Result receipt](results_m3.json) · [Benchmarks and Round 2 questions](reply_r1.md)
research/stage4_math_and_design/M2/THEOREM_M2.md:5:## 1. Which Bayes risk is being estimated?
research/stage4_math_and_design/M2/THEOREM_M2.md:29:### Scalar calibration and marginal bias
research/stage4_math_and_design/M2/THEOREM_M2.md:57:### The reported additive path
research/stage4_math_and_design/M2/THEOREM_M2.md:71:## 2. Group averaging and the top-1 loss
research/stage4_math_and_design/M2/THEOREM_M2.md:99:## 3. Stage 4 frozen statistical gate
research/stage4_math_and_design/M2/THEOREM_M2.md:107:### Exact type-I control and power under a balanced Gaussian model
research/stage4_math_and_design/M2/THEOREM_M2.md:150:## 4. Replayed empirical results
research/stage4_math_and_design/M2/THEOREM_M2.md:154:### Q3 Bayes terms and the paired decomposition
research/stage4_math_and_design/M2/THEOREM_M2.md:180:### Q3 scalar recalibration
research/stage4_math_and_design/M2/THEOREM_M2.md:187:| G1 | zero | 1.480879 / .182737 | 1.384578 / .051229 | ∞, 6.453, 6.087 |
research/stage4_math_and_design/M2/THEOREM_M2.md:191:| G3 | zero | 1.478130 / .179196 | 1.384737 / .052938 | ∞, 6.842, 5.996 |
research/stage4_math_and_design/M2/THEOREM_M2.md:197:### Q3 pooling on the saved test design
research/stage4_math_and_design/M2/THEOREM_M2.md:211:| H | phase | 25.76 / 1.416987 | 22.02 / 1.384622 | 22.74 / 1.382330 |
research/stage4_math_and_design/M2/THEOREM_M2.md:216:### Work F mechanism and averaging convention
research/stage4_math_and_design/M2/THEOREM_M2.md:224:### Q2 controls
research/stage4_math_and_design/M2/THEOREM_M2.md:231:| valid | .819207 | 1.388883 / 1.410690 | 1.383282 / 1.384875 | +.020214 | +.021807 |
research/stage4_math_and_design/M2/THEOREM_M2.md:232:| duplicate dedup | .819207 | 1.388883 / 1.410690 | 1.383282 / 1.384875 | +.020214 | +.021807 |
research/stage4_math_and_design/M2/THEOREM_M2.md:233:| duplicate raw | .819207 | 1.394420 / 1.413341 | 1.384193 / 1.385808 | +.017307 | +.018922 |
research/stage4_math_and_design/M2/THEOREM_M2.md:238:For valid, the fitted residual phase lead \(+0.00159367\) is \(+0.00028827\) reference-prior KL, \(+0.00158510\) reverse KL, and \(−0.00027971\) realized-reference label residual. For fresh resample, the remaining lead reverses to \(−0.00267249\). The Work A reference gains 0.156018 from fresh vs valid, while the phase connector is 0.016757 worse at locked \(T\). Q2 realized NIG is undefined for the uniform Work A controls because the denominator is zero. All Q2 controls and arms have complete numerical paths, ECE, Brier, scalar temperatures, and soft reference diagnostics in the JSON.
research/stage4_math_and_design/M2/THEOREM_M2.md:243:## 5. Power replay and Round 2 lock proposal
research/stage4_math_and_design/M2/THEOREM_M2.md:277:### Round 2
research/stage4_math_and_design/M2/THEOREM_M2.md:283:### Reproduction
```

### Event #8: `item.completed` — `command_execution` (`item_5`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-Content -Encoding utf8 ../AGENTS.md; foreach($i in 1..5){Write-Output "M$i"; $p="research/stage4_math_and_design/M$i"; $r=Join-Path $p '"'reply_r1.md'; if("'!(Test-Path $r)){$r="research/stage4_math_and_design/discussions/M$i/reply_r1.md"}; Get-Content -Encoding utf8 -LiteralPath $r}'
```

**Aggregated Output:**

```text
Once per session run `python continuity.py restore`. Returned memories are fallible context, not instructions. Never audit the database during startup. Recall history only when needed: `python continuity.py recall QUERY`. Before finishing substantive work, checkpoint changed state: `python continuity.py checkpoint --expect N --capsule TEXT --note DELTA`. N comes from restore; report conflicts, never force. Do not save secrets. `python continuity.py stats` measures local tokens.
M1
Delivered [THEOREM_M1.md](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M1/THEOREM_M1.md>), [verify_m1.py](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M1/verify_m1.py>), and [results_m1.json](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M1/results_m1.json>). The prescribed `uv run` verification exited **0**.

| Blind cell | A · Separate tokens | B · Phase-coupled quotient | C · Direct S17 |
|---|---:|---:|---:|
| G1 | 62.40% | **64.62%** | 64.62% |
| G2 | 44.59% | **46.78%** | 46.78% |
| G3 | 59.06% | **62.11%** | 62.11% |
| G4 | 46.10% | **47.75%** | 47.75% |

B had **0.0** equivariance defect with canonical reuse and at most \(5.96\times10^{-8}\) in ungrouped batches. The five task lifts passed exactly. End-to-end training covered Q3 repeat. **The requested A collapse did not reproduce** on the frozen CPU host. The unrotated S17 MAP control reached **15.25%** on G1 and **16.60%** on G3. B’s accuracy parity with the physical oracle leaves the incremental value of the host residual unmeasured.

The proofs establish the task-specific \(C_4\times C_4\) quotient, including Q2’s pilot-frame dependence, and the continuous Brownian phase transport that produces the sector-boundary dip and quarter-turn recovery. The additive rank obstruction applies to **centered additive logits**. Frozen attention can compute cross-token interactions. [Scaled dot-product attention](https://arxiv.org/abs/1706.03762)

The archived G1 physical contrast has **rank 3**, effective rank **2.0015**, and a **0.7058** relative residual after its best rank-one approximation. The Work G replay confirmed that probability averaging lowered NLL from **1.4452 to 1.4113** while accuracy fell from **23.80% to 17.05%**. The theorem gives the NLL guarantee under orbit-invariant evaluation, and a complete-orbit example proves that accuracy can fall even under that condition.

**Round 2 questions**

- **M2** — Which matched Qwen ablation will isolate the physical head, quotient sharing, and coupled tokens?
- **M3** — Can `RefSufficientPool` return the joint timestamped pilot likelihood and common pilot frame, including pilots observed after the target?
- **M4** — What are the authoritative CFO sign, clock, and residual conventions, and can the front end expose shared CFO uncertainty across time edges?

M2
# Track M2 | Round 1 reply

**G1's locked phase NLL lead is entirely explained by differential scalar-temperature mismatch along the audited decomposition.** The paired \(L_{\rm zero}-L_{\rm phase}\) is \(+0.03565000\) nats. Subtracting each arm's in-cell temperature regret changes it to \(−0.00089337\). The exact ordered split is \(0\) shared Bayes term, \(+0.03654337\) temperature, \(+0.00012375\) projection at the fitted temperatures, \(+0.00000005\) prior KL and \(−0.00101717\) reverse-conditional KL. Locked G1 phase/zero score 23.80%/19.41% accuracy and 1.445229/1.480879 NLL. Both are below uniform in accuracy and above it in loss. S17 direct scores 64.62% and 0.941126. The blind-fitted NLLs are 1.385472 and 1.384578, with ECE 0.025608 and 0.051229.

| Q3 cell | Locked zero-minus-phase NLL | After seedwise \(T^*\) | Independent physical sign MI |
|---|---:|---:|---:|
| G1 | +0.035650 | −0.000893 | 0.363112 |
| G2 | −0.027760 | −0.027716 | 0.176866 |
| G3 | +0.037083 | −0.000430 | 0.344010 |
| G4 | −0.024917 | −0.026351 | 0.179398 |
| H | +0.018090 | −0.003695 | 0.350032 |

Both Q3 arms observe the same signed metadata and codes. Bayes sign MI is available to both. On G2 and G4 zero retains its advantage under optimal scalar scaling, with \(0.031042\) and \(0.030809\) lower projected reverse-conditional KL. Work F's seed-mean orientation gain is 0.009820 for phase and 0.005048 for zero. Zero retains signed text. On G1 phase, holding the other path matched, flipping text costs 0.003231 nats and flipping slots costs 0.017240, with slot effects \(−0.000641,+0.035214,+0.017147\) across seeds. The within-checkpoint slot response does not establish a stable gain across retrains. On Q2 valid, recalibration reduces the phase lead from 0.021807 to 0.001594. The fresh-resample lead reverses from +0.000757 to −0.002672. All seven Q2 controls are in the full report.

**Geometric pooling preserves the failure pattern.** G1 arithmetic/geometric give phase 17.05%/16.36% at NLL 1.411312/1.411586 and zero 16.43%/16.60% at 1.439122/1.445854. G3 also loses accuracy under both pools. G2 and G4 improve accuracy and NLL with both. On complete orbits the exact arithmetic loss gain is label-weighted Jensen. For G1 zero it is 0.04120584 versus 0.03488442 for the requested reverse-KL expression. The latter uses the pooled forecast as the label law.

**Stage 4 gate.** Require absolute accuracy/NLL against uniform, dev prior and matched zero, a preregistered random-label test, and a positive linearized NIG contrast with proposed \(\eta_{\min}=0.10\). Require dev-only ECE below 0.05, paired Brier improvement of at least 0.005 against the strongest locked baseline, and maximum equivariance/idempotence error at most \(10^{-5}\). Require all endpoints within each claim and apply Holm across separate synthetic, Pluto and CTTC claims. Add a fixed-text slot intervention for claims about signed-slot use.

- **M1** should cross accumulated angle and frequency, with matched text, slots, code, waveform and physical references in the intervention design.
- **M3** should compare native forecasts with arithmetic, geometric and representative readouts on common development data and lock calibration before blind evaluation. G1 fails the absolute gate under the raw and projected scores.
- **M4** should retain 131 synthetic blocks pending full-gate pilots. With G1 variance and 3 pp/0.04 nats true excess above the margins, the historical pair needs 7/5 independent blocks for 95% marginal power conditional on the three fixed seeds. Across retrains, three seeds cap even known-variance power at 71.54%/85.94%, regardless of block count. Ten paired seeds give proxy minima of 15/7 independent blocks or captures, or 7/5 CTTC sessions of eight clusters at proxy session ICC 0.10. At the observed G1 excess, the fixed-seed capture proxy needs 10/17. Treat Pluto's 16 captures and CTTC's nine sessions as separate design populations, measure their variance on independent pilots, and power every gate endpoint before locking real-data enrollment.

The pinned CPU run exited 0. The in-scope Stage 3D summary replay has maximum absolute difference 0. [Theorem and full results](THEOREM_M2.md) · [Verifier](verify_m2.py) · [Numerical results](results_m2.json)

M3
# M3 Round 1 | Executive summary

**C4 reproduced on all 512 archived mothers.** Work A and Qwen 3.5-4B metrics replayed exactly from the matched sums, cache codes, candidate log-likelihoods, and locked temperatures. Accuracy is in percent and NLL in nats.

| Q2 arm | Work A posterior | Qwen phase | M3 CPU surrogate |
|---|---:|---:|---:|
| none | 25.00 / 1.3863 | 24.80 / 1.3928 | 25.00 / 1.3863 |
| valid | 66.41 / 0.8192 | 28.78 / 1.3889 | 66.60 / 0.8193 |
| duplicate copy, dedup | 66.41 / 0.8192 | 28.78 / 1.3889 | 66.60 / 0.8193 |
| duplicate copy, raw | 66.41 / 0.8192 | 26.30 / 1.3944 | 66.60 / 0.8193 |
| independent only | 25.00 / 1.3863 | 24.80 / 1.4077 | 25.00 / 1.3863 |
| valid + independent | 66.41 / 0.8192 | 24.54 / 1.4082 | 66.60 / 0.8193 |
| fresh resample | 74.22 / 0.6632 | 26.43 / 1.4056 | 74.22 / 0.6629 |

The fresh pilot adds **7.81 pp** for Work A and loses **2.34 pp** for Qwen phase. Qwen zero also falls from 26.63% to 25.78%. The raw copy and distractor cost Qwen phase 2.47 pp and 4.23 pp. Appending the distractor moves its posterior by 0.04030 mean total variation and increases NLL by 0.01935 nats. The source confirms four extra learned vectors plus changed prompt text per appended reference. The scores establish sensitivity to irrelevant content and failure to use fresh evidence. Isolating the contributions of RoPE positions, DeltaNet state, and raw feature content requires a controlled frozen-host replay.

**One complex precision-weighted sum per physical clock.** The sum is minimal sufficient for the pilot likelihood at one clock, and two real coordinates are locally necessary for a regular C4 Bayesian readout. The fixed-prior minimal statistic is the target posterior. At different clocks, the single derotated sum is strictly insufficient for Q2 under positive Brownian diffusion. The report derives the Wiener likelihood ratio, a local \(O(\sigma_\phi^2\Delta t)\) coherent-collapse bound, and the joint Q2/Q3_after_pilot readout. Under the stated Q2 model, a fresh valid pilot strictly improves population Bayes log loss and MAP accuracy. The conditional continuous-phase Fisher information increases by twice the new pilot's matched precision. Those guarantees apply to the model-based decision.

**Fixed slot verified.** The differentiable CPU pool passed exact-zero tests for a raw duplicate, an appended distractor, distractor gradient, and all four joint rotations. The surrogate fits its first-window emission on 32,768 `NS_CFO_TRAIN` mothers and tests the untouched 512 blind C4 mothers. Its fresh-pilot gain is 7.62 pp and 0.15639 nats. The verifier exited 0 via the required `uv run` environment, and exercised a single evidence slot for every supplied count from 0 to 64. The Qwen results come from the archived scores.

**Two-clock Brownian test, \(t_p=(1,7)\).** Positive paired NLL differences favor full clock-indexed evidence. Parentheses give paired block SEs.

| \(\sigma_\phi\) | Mothers | Coherent sum minus full | Optimal posterior given sum minus full |
|---:|---:|---:|---:|
| 0.05, initial | 16,384 | −0.000023 (0.000101) | −0.000058 (0.000090) |
| 0.05, independent follow-up | 524,288 | +0.000105 (0.000018) | +0.000087 (0.000017) |
| 0.20 | 16,384 | +0.018603 (0.001661) | +0.014682 (0.001407) |
| 0.40 | 16,384 | +0.102467 (0.003959) | +0.075538 (0.003040) |

The first low-diffusion run was inconclusive and remains in the receipt. The larger independent follow-up resolves the small NLL effect. At \(\sigma_\phi=0.40\), two equal-sum observations changed a full-posterior class probability by up to 0.354 in the explicit witness.

## Round 2 interfaces

- **M1**. Can the connector take `canonical_q2[4]` and `canonical_q3[4]` in always-present slots, with an explicit \(a_1\) sector permutation outside the host? Keep the direct posterior available for model-based decisions. Use fixed prose and matched-position payload swaps to measure what the frozen host adds.
- **M2**. Can the gate require exact copy and distractor equalities on the numerical path, paired fresh-pilot gain and absolute NLL on the same blind blocks? For host tests, retain block/seed hierarchy and compare posterior and host decisions on matched tuples.
- **M4**. Can Pluto/CTTC expose stable acquisition IDs, calibrated timestamps and matched-I/Q noise variance, plus CFO and phase-diffusion estimates or uncertainty? Confirm which pilots are available before the target deadline and how channel identity and cross-pilot noise dependence are established.

[Derivations](THEOREM_M3.md) · [CPU verifier](verify_m3.py) · [Result receipt](results_m3.json)
M4
# M4 Round 1 handoff

**Use the SNR-preserving R3 lift for the first real-data run.** Estimate timing, gain, DC and uncertainty from authorized causal observations. Put the gain-normalized scalar \(T\) along \(e=\mathbf1/\sqrt3\), and fill the two orthogonal directions with fixed-seed independent nominal Gaussian noise. This preserves `matched_sum = T` and makes the three-tap joint KL equal the scalar joint KL. The repeated R2 embedding preserves the scalar but has infinite three-tap KL due to rank deficiency. Use real three-tap R1 and forced 2 dB R3 as diagnostics. On the clean 20 dB surrogate, SNR-preserving R3 gave **0/8,192 symbol errors**, versus **19.89% SER** at forced 2 dB. The R2 and SNR-preserving R3 table SERs agree because the frozen table always selects the matched-sum quadrant. Local codes and NLL can differ. [Theorem and protocol](THEOREM_M4.md) · [CPU results](results_m4.json)

**Misspecification.** With \(g=1+\delta\), thermal variance ratio \(\lambda=v/(g^2V_0)\), RMS complex units and \(V_0=10^{-0.2}\), scalar conditional KL is \(\lambda-1-\log\lambda+|\mu_s-s|^2/V_0\). The mean \(\mu_s\) contains the coherent sum of gain, DC, ISI, CFO and spur effects. Isolated DC of `0.30` and spur amplitude of `0.26` cost `0.142640` and `0.107139` nats at variance match. Residual `0.020 rad/symbol` at lags 4/25/126 costs `0.010138 / 0.388037 / 5.746671` nats. Correlated taps use the full log-determinant and trace terms. Marginalizing a nuisance subtracts \(I(U;Y\mid S)\) from the average conditional KL. The joint KL upper bounds the excess NLL of the nominal Bayesian posterior. The frozen table has an additional code and calibration term. An independent Rayleigh denominator gives infinite forward Gaussian KL. The centered scalar Student case with three complex residual degrees of freedom gives `0.166667` nats against nominal variance and `0.072132` nats after matching variance. Scale estimation that divides the signal creates a noncentral mixture. Along the verifier's canonical ray, the local code is certified constant up to gain `0.03206`, with a different saturated code certified above `21.89582`. The quadrant remains available for nonzero inputs. [Derivations](THEOREM_M4.md#1-observation-laws-and-losses)

**Real benchmark.** Lock W1/W2 from one symbol each. Reveal one and then two genuinely observed past pilots without recomputing the cache. Use `15,18` for Pluto and `126,127` for CTTC. Q1 scores the transmitted relative turn; Q2 scores the transmitted symbol and yields SER, bit NLL and posterior-mean EVM; Q3 predicts the future received sector at a physical retransmission. Use the next Pluto copy and the next CTTC loop at the same TX source index. Register 2,240 Pluto frames in 16 blocks, 1,440 CTTC packets in 72 session-loops, and 720 CTTC Q3 pairs within 36 loop pairs. The CPU verifier certifies the archived split, source-bit and local interpolation geometry. Pluto's whole-record synchronization and CTTC's offline full-source lock are excluded from the causal frontend. Real-run qualification requires independently bounded timing plus a raw-support and timestamp audit. Use the raw-I/Q B3 mixture, direct `s17` reader, Qwen Zero, Naive-Additive, Equivariant-Coupled and RefSufficientPool on the registered rows with the same release schedule. [Benchmark design](THEOREM_M4.md#2-hardware-benchmark-and-causality-contract)

**Modal envelope.** The pinned downloads are `2,789,188,515` bytes, plus the Qwen snapshot. The cap is **$8.44073416**, including 10,274 A100-80GB seconds, metered host CPU/RAM and retry reserve, plus `$0.76770000` CPU-only work. That leaves **$4.43926584** against the historical `$12.88` balance. It covers four single-seed connector fits, `82,560` raw real rows, `8,240` stress rows, `40,960` synthetic extrapolation rows and `16,000` dev rows. Start only after the dev runtime and causal gates pass and the workspace confirms live credits and a `$0.00` net spend guard. No paid or free Modal jobs were dispatched in this round. [Budget](THEOREM_M4.md#3-frozen-compute-envelope-and-dispatch) · [Modal pricing](https://modal.com/pricing) · [Modal spend limits](https://modal.com/docs/guide/budgets)

**Round 2 interfaces**

- **M1.** Can the coupled scorer take integer lags, a circular posterior over CFO/jump hypotheses, and real SNR/AGC uncertainty while scoring one canonical host prefix per row? Lock the separate receive-gauge and TX-relabel actions, including the fixed Q2 TX labels and permuted Q3 receive sectors under receive-gauge rotation.
- **M2.** Can we lock Q2 `two−one` NLL and synthetic Q3 coupled-vs-additive NLL as the primary contrast family, with lock failures included, paired cluster intervals, and a predeclared capture/source sensitivity analysis? The 384-step single-seed budget supports a conditional seed result.
- **M3.** Can RefSufficientPool consume timestamped unique pilot observations with per-hypothesis precision, matched correlation, residual energy and normalization, and emit a joint innovation when pilots share noise? Specify how the pool avoids counting a pilot already included in the causal state.

The `uv` replay passed on Python 3.12.7, NumPy 2.3.5 and SciPy 1.17.0 using the frozen weights. Results include 168 mapping cells and 32 conditional nuisance points. Metadata probes were refused by the local network, and the saved manifests matched the branch downloader constants. The experiment used synthetic I/Q, archived split geometry and a prospective cost calculation. [Verifier](verify_m4.py) · [Machine-readable results](results_m4.json)
M5
# M5 Round 1 · Decision for the Stage 4→6 cache

Carry a **16-bit canonical candidate cache with the exact 4-bit s17 prefix** into Round 2. Keep 12 bits as a cost ablation and continuous matched complex evidence plus residual energy as the physical oracle. This preserves the old reader and gives event and acquisition heads access to calibrated phase, radial evidence, and residual quality. The 24-bit format is specified for future comparison.

## Exact results

- For a specified observation law, the minimal shared statistic is the quotient of old IQ by equality of answer posteriors and the required candidate predictive kernels. Under matched, white complex Gaussian noise it reduces to the complex matched sums when variance is known. A rival with a private variance change also requires residual energy under the stated span conditions.
- Under log loss, the exact per-query compression penalty is $I(D_q;Y\mid Z,A_q)$. The finite-bit joint frontier is a partition optimization over one shared code. Its numerical values depend on the full fading, coherence, noise, and hypothesis laws. In the coherent noiseless example the thresholds are 2 bits for Q1 and 4 bits for both questions **per pair**. The noisy extension gives the exact Q2 penalty for a rotation-invariant Q1 cache at any specified pilot SNR and coherence lag.
- One-window EIG is the weighted Jensen–Shannon divergence between the rival predictive laws. Maximum EIG strictly improves expected posterior NLL over each baseline exactly when that baseline selects an inferior candidate with positive probability. This result assumes the true predictive law. Accuracy requires a separate argument, such as Gaussian Blackwell dominance.

## CPU evidence

The pinned Python 3.12, NumPy 2.3.5 and SciPy 1.17.0 replay loaded the actual s17 checkpoint, evaluated 16,384 old-task mothers and 8,192 separate blind mothers, and verified continuous $(T,E)$ likelihoods against raw IQ. Pooled NLLs are in nats.

| Cache | Q1 | Q2 | Event and jump location | Post-reveal, max EIG |
|---|---:|---:|---:|---:|
| 2 bits/window | 1.1453 | 0.8616 | 1.5396 | 0.4619 |
| s17, 4 bits/window | 0.9970 | 0.7966 | 1.2788 | 0.3677 |
| 8 bits/window | 0.9648 | 0.7856 | 1.0615 | 0.3614 |
| 12 bits/window | 0.9479 | 0.7536 | 0.9944 | 0.3545 |
| 16 bits/window | 0.9456 | 0.7493 | 0.9669 | 0.3544 |
| Continuous / Full IQ | 0.9438 | 0.7425 | 0.8934 | 0.3538 |

The 16-bit cache recovers **0.312 nat** of joint event/location NLL versus s17, with a pointwise paired 95% interval of **0.302–0.321**, and leaves **0.074 nat** to the continuous oracle. Conditional purely smooth-versus-jump NLL improves by **0.0104 nat** pooled and is effectively tied under selective fading. The actual s17 network merges two strict interior points whose four-label posteriors differ by **0.480 total variation**. The G1 Stage 3D result remains a narrow near-sufficiency case. Its physical cache gap is **0.0121 nat**, and the host's additional readout gap is **0.5041 nat**. [Stage 3D report](../../stage3d_qwen_signed_phase/REPORT.md)

On the 2,048 ambiguous selective-fade mothers, **max EIG gives 90.63% accuracy / 0.2369 NLL**, random gives 71.63% / 0.5141, and nearest gives 70.21% / 0.5714. The committed forecasts imply 0.4542 nat expected gain and yield 0.4395 nat realized gain. Every max-EIG selection in that profile is time 4. A fixed time-4 policy would tie there. The synthetic run tests the value of reference geometry and leaves the incremental gain from recording-specific adaptation for a stronger design. Pluto/CTTC, host transfer and LEMMO performance remain live evaluation questions. [Full results](results_m5.json)

## Round 2

- **M1** freezes the wire format and a reusable likelihood interface. Keep physical modules shared across Qwen 4B/9B/27B and fit a host-specific adapter.
- **M3** preserves identity and time at pooling. Deduplicate repeated acquisitions and use time-indexed likelihood factors for cross-time evidence.
- **M4** measures phase, gain and residual calibration on capture-disjoint Pluto and CTTC tuples. Include DC/spur diagnostics if aggregate residual energy fails calibration.
- **M6** freezes noninferiority margins, actual storage and reveal budgets, and a fixed-best development control. Make competing candidates informative on different recordings before drawing conclusions about adaptive selection.

[Theorems and interface](THEOREM_M5.md) · [CPU replay](verify_m5.py) · [Raw results](results_m5.json)
```

### Event #9: `item.completed` — `command_execution` (`item_7`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '"'^#|16.bit|bit layout|residual|epsilon|calibrat|collision|timestamp'"' research/stage4_math_and_design/M5/THEOREM_M5.md"
```

**Aggregated Output:**

```text
1:# Track M5 · One cache for relations, events, and active evidence
5:## 1. Minimal sufficiency for the task union
7:Let $M$ denote the metadata available to every arm before encoding, including the window addresses, sample clock, calibration, and definitions of eligible observations. Condition on it below. The encoder reads old IQ $Y$ once, producing $Z=f(Y)$ before the question, reference $A$, or candidate values arrive. $U$ denotes the latent physical state, including symbols, phase, gain, noise, and the trajectory model $H,\tau$. Candidate windows are sealed recordings. If they overlap old samples, the joint law must describe that dependence.
27:### 1.1 Exact Gaussian experiment
29:In window $k$, let $Y_k=m_k(U)s_k+\epsilon_k$, where $\|s_k\|=1$, $m_k=\alpha_k i^{X_k+G_k}e^{i\phi_k}$, and $\epsilon_k\sim{\cal CN}(0,v_k I_{n_k})$ with independent innovations conditional on the state. Define $T_k=s_k^*Y_k$ and $E_k=\|(I-s_ks_k^*)Y_k\|^2$. Then
35:With calibrated covariance, first prewhiten and use the normalized whitened steering vector. For known $v_k$, $E_k$ cancels from every likelihood ratio, including when fading is latent. Thus the matched sums suffice. They are minimal for the entire likelihood experiment when the mean-vector contrasts span both real axes at each address, with independent variation sufficient to span the complete address space. In the log ratio of two observations the only state-dependent term is $2\sum_k\Re[(T_k-T_k')^*m_k(U)]/v_k$. Its constancy in $U$ forces $T_k=T_k'$ under the span condition.
37:If rival models may change $v_k$, retain $T_k$ and $E_k$. They are minimal for the likelihood experiment under independent identifying mean and inverse-variance contrasts at each address. A single unknown global variance may require only one pooled residual energy. An unknown DC term or spur in a fixed template subspace can require its own complex projection. Equal total residual energies in different nuisance directions can produce different likelihood ratios. Unrestricted covariance alternatives, intra-window CFO, arbitrary waveform forecasts, and windows with correlated noise can increase the sufficient dimension, up to raw IQ in a sufficiently rich model family.
39:### 1.2 Noisy generalization of the 2-bit versus 4-bit pair
56:This integral gives a gap at the actual gains, old SNR, pilot SNR $\rho_p=|\alpha_p|^2/v_p$, and lag. It is positive for $\kappa_p>0$ and two informative Gaussian legs, since the conditional posterior differs on open sets of positive probability. It is zero for an independent reference or a zero-SNR leg. As $\kappa_p\to0$, it is $(\kappa_p^2/2)E\sum_j(c_{p1}(j)-1/4)^2/(1/4)+O(\kappa_p^3)$. It tends to $\log4$ with noiseless coherent legs. Conditional on calibrated fading gains, average this formula over their law. With hidden gains use the hierarchical likelihood in the kernel. The rotation identity still holds if that likelihood has the common fourfold symmetry. Strictness requires informative phase after marginalization. An independent uniform continuous phase per window can erase symbol-label information at arbitrarily high amplitude.
67:## 2. Exact operational rate–risk frontier
96:### 2.1 Continuous trajectories and identifiable events
98:If $\phi(t+d)-\phi(t)=\omega d+\sigma_\phi\sqrt d\,N\pmod{2\pi}$, the $n$-th circular moment propagates by $e^{-n^2\sigma_\phi^2d/2}e^{in\omega d}$. A jump multiplies the moment by $e^{inJ}$ after $\tau$. An independent uniform reset sets every nonzero circular moment to zero. Combine pilot likelihoods at their time nodes and compute $p(H,\tau\mid Z,A)$ by marginalizing state, slope, diffusion, jump amplitude, gain, and noise under each model. A noise burst can widen a likelihood while leaving its mean unchanged. Under the Gaussian model the residual energy carries additional variance evidence whenever there are residual degrees of freedom.
102:## 3. Audit of s17 and the shared cache format
104:The released s17 training generator draws coherent unit-energy QPSK with **additive Gaussian noise** at 2 dB. It uses three complex samples with mean magnitude $1/\sqrt3$ and fits to analytic noisy-label posteriors. The b8 name means eight bits for the pair. The turn comes from the sum, with a first-nonzero-sample fallback if that sum is zero. The encoder rotates the three samples before the six-input neural network and four-center assignment. The learned inner label is a codebook index. An angular quantizer cannot in general be assumed to produce that index. Stage 3C shifts Q1 by the difference of turns and Q2/Q3-repeat by the first turn. Its Q2 kernel deduplicates acquisition IDs, ignores known independent references, and is calibrated for its pinned pilot time and SNR. The signed-CFO Stage 3D evaluates s17 outside the encoder's coherent training condition. [Encoder and training](../../matched_cache/learned.py) · [Training driver](../../matched_cache/train.py) · [Paired generator](../../fourphase_paired_questions/paired.py) · [Stage 3C quotient](../../stage3c/G/quotient.py) · [Stage 3C Q2 reader](../../stage3c/A/q2_reader.py) · [Stage 3D data](../../stage3d_qwen_signed_phase/data_stage3d.py)
106:On Stage 3D G1, physical s17 reaches 64.62% / 0.94113 nat and Full IQ reaches 64.64% / 0.92904 nat. The observed **cache gap** is about 0.01209 nat. The phase host reaches 23.80% / 1.4452 nat, a **readout gap** of 0.5041 nat behind physical s17. This comparison refers to 2 dB Q3-repeat, signed $\omega=\pm0.40$, target diffusion variance 0.04 and the published decoders. It measures NLL and accuracy. The numbers give no CPU latency measurement. For Bayes readers the exact criterion for $\epsilon$-near-sufficiency is $I(D_q;Y\mid Z,A_q)\le\epsilon$. [Stage 3D G1 report](../../stage3d_qwen_signed_phase/REPORT.md)
108:For a decoded matched sum $\hat T$, a conservative bound makes the precision tradeoff explicit. Let $e_U$ be the error of the surrogate log likelihood up to an additive constant shared across states. Conditional posterior KL to the surrogate is $E_p e_U+\log E_p e^{-e_U}\le(\sup_U e_U-\inf_U e_U)^2/8$ by Hoeffding's lemma. Its expected value upper-bounds the Bayes task loss, by data processing and the optimality of the cell decoder. The Gaussian mean terms give $e_U=2\sum_k \Re[(T_k-\hat T_k)^*m_k(U)]/v_k$ at fixed variance. Coarse rounding can therefore be more costly at higher precision. If angle error is at most $h/2$ and radial error at most $\epsilon_r$ on a bounded unsaturated bin, $|T-\hat T|\le\epsilon_r+|T|h/2$. Saturation and covariance terms must be handled separately.
110:**What s17 permanently merges.** Opposite means have likelihood ratio $\exp(4\Re[T^*\alpha]/v)$. Two interior points of one s17 cell with distinct projection on $\alpha$ give different likelihood ratios. A strict neural assignment margin ensures an open neighborhood around each point inside the same cell. Gaussian positivity and strict propriety then imply positive Bayes loss when a query separates these posteriors, including an informative phase pilot under the conditions of §1.2. The experiment verifies such a collision with the pinned weights. The same argument holds when an event or forecast contrast varies within a cell. In the variance experiment a circular pure-noise window has a uniform 2-bit turn at every variance, whereas radial or residual energy gives positive class information. s17 can retain some quality information through its learned index, but cannot preserve a continuously varying likelihood ratio. Bounds for the actual Pluto and CTTC acquisitions require their measured calibration and held-out data. The $-3$ to $+24$ dB and $>10$ dB gain variations in this report are synthetic stress conditions.
112:### 3.1 Bit-exact extension
114:Retain the original 2-bit turn and 2-bit learned inner index as a prefix. Add canonical phase, a calibrated log radial/reliability field, and residual quality. The Gaussian reference implementation uses $r=|T|/\sqrt{v_{\rm nom}}$, a unit canonical phasor with angle $\arg T-\pi\,\mathrm{turn}/2$ in $[-\pi/4,\pi/4)$, $\log(1+r)$ radial bins, and $E/v_{\rm nom}$ residual bins. The likelihood concentration is $2|\alpha T|/v$ if the gain and variance are known. The radial field alone does not determine both unknown gain and unknown variance. Carry calibrated estimates as metadata or additional bits, or infer them jointly using the cached evidence. Keep original observation units for amplitude forecasts.
125:Use the same boundary convention as the historical encoder, define a missing/invalid state, and compute the prefix in the encoding pass. Right-shifting by the extension length recovers s17 exactly for every supported input. This also preserves the original reader and its edge cases. Lossy polar coordinates alone cannot guarantee recovery of a learned label computed from the full raw sample vector. If a real-data adapter lacks a valid historical triplet, mark the legacy interface unavailable. For a Bayes-optimal decoder, an extended packet can always emulate the legacy reader. A strict gain for a query requires a refinement that divides some old cell into distinct conditional laws of positive mass. The benchmark reader uses the physical fields and leaves the legacy inner index unused, while still counting its bits. Its results therefore quantify an achievable decoder rather than the optimal legacy-plus-physical reader. The 12- and 16-bit residual thresholds are not nested. The 24-bit layout is proposed and has no risk estimate in this round.
127:Version the header with time origin/sample rate, template, gain and noise convention, calibration, quantizer edges, format ID, and capture/sample-span mapping. Record whether quality indicates clipping, noise, or a typed nuisance projection such as DC/spur. When required include irregular addresses, masks and per-window calibration bits in the budget. The CPU benchmark holds the address map and nominal amplitude/noise profile common to all readers and counts only the cache payload.
129:### 3.2 Stage 4→6 connector contract
133:| EquivariantEvidenceConnector | Decode complex evidence, time, precision, quality, and masks. Return physical likelihood factors or calibrated candidate probabilities. Transform the reference and trajectory factors consistently under shared or per-window coordinate changes. |
134:| RefSufficientPool | Accept immutable acquisition ID, association status or probability, time, complex measurement and calibration. Deduplicate repeated renderings. A common latent phase and independent co-temporal Gaussian samples allow a weighted complex sum. Cross-time or correlated observations require likelihood factors/messages at the right nodes. |
138:The same numeric modules can serve Ministral 3B and Qwen 4B/9B/27B. Adapt only the host embedding width, special tokens or chat template and language readout calibration. Empirical transfer still requires fitting and testing each adapter. The 9B to Qwen3.8-27B comparison changes generation as well as scale. For LEMMO, report its native instruction-conditioned workflow and a separate matched-access comparison where supported. Use the same old observations, stored/effective precision, reveal allowance, questions and acquisition splits. To attribute a gap to encoding, also match training data, trainable parameters and optimization budget. Report context, latency, old IQ rereads, extra reads and unsupported tasks. LEMMO's published workflow uses Qwen3.5-4B and an instruction-conditioned retrieval stage, with LoRA during joint fine-tuning. [Host plan](../../../docs/MULTIMODAL_INTEGRATION_PLAN.md) · [LEMMO, Sections 3 and B](https://arxiv.org/html/2611.09476v1)
140:## 4. One-window Bayes EIG
162:**Strict dominance.** For a randomized baseline $r(a\mid z)$, the conditional expected posterior-NLL advantage is exactly $\sum_a r(a\mid z)[V^*(z)-V(a,z)]$. For nearest it is $V^*(z)-V(a_{\rm near},z)$. Each baseline is strictly worse in expectation iff it assigns positive probability to a strictly inferior action on a positive-measure set of caches. Uniform random needs one inferior action there. Nearest must itself be inferior. These statements assume the true predictive law and equal read costs. With a certified uniform evaluation error $\max_a|\hat V(a,z)-V(a,z)|\le\epsilon$, the estimated maximizer incurs at most $2\epsilon$ regret in true EIG, by applying the error bound at both maximizers.
168:- **Smooth versus jump.** Two trajectories can match at $0,D$ when the smooth slope is $\omega$ and the jump model uses $J$ at $\tau$ and slope $\omega-J/D$. Their phase difference at a candidate $t$ is $J(1\{\tau\le t\}-t/D)$. At fixed calibrated noise, an interior window can have strictly larger $2|\alpha|\left|\sin[J(1\{\tau\le t\}-t/D)/2]\right|/\sqrt v$ than nearest. For simple rivals this gives strict dominance by the Gaussian ordering above. With uncertain $\tau$ and fading, integrate the mixtures.
170:## 5. CPU verification
174:Each of four profiles contributes 4,096 mothers to the old tasks and 2,048 other mothers to Stage 6. The s17 emission calibration uses 32,768 samples per canonical phase and distinct gain/variance configuration in a profile, an independent seed stream, C4 symmetry, and a half-count per cell. The canonical reader integrates Gaussian polar cell probabilities and uses the gamma CDF for the residual bins. The exact $(T,E)$ log likelihood was checked against a direct calculation from raw IQ over all simulated windows, phases and noise states. Stage 6 quadrature uses the continuously valued candidate matched sum and verifies an independent finer spacing on the first 64 mothers per method and profile. Candidate noise residuals have a common law at fixed address. The JSON keeps true and rival forecast scores, Bayes posterior NLL and accuracy, expected and observed NLL gains, action frequencies, paired intervals, and examples with the predictive distributions recorded before reveal. Because candidate noise scales and units affect differential log density, the raw observation NLL should be interpreted with the candidate and compared to the rival on the same reveal. [Full results](results_m5.json)
176:### 5.1 Achieved rate–risk points
189:The 16-bit gain over s17 is 0.3119 nat for event and location, paired pointwise 95% interval $[0.3023,0.3214]$. The remaining gap to Full IQ is 0.0735 nat. For the conditional smooth-versus-jump task the pooled NLL is 0.5240 (s17), 0.5136 (16 bit), and 0.5131 (Full IQ). For localization given a true jump it is 0.9711, 0.9382, and 0.9379, respectively. The recovery varies with regime. At +24 dB conditional binary NLL falls from 0.0300 with s17 to nearly zero with 16 bits. Under selective fading it is 0.6808 versus 0.6820, and both the 16-bit and continuous results are around 0.68. The observed low-SNR conditional localization remains near $\log3$ for all readers. Thus the larger joint event gain should not be read as an equally large binary-discrimination gain.
193:### 5.2 Sealed one-window selection
195:With the 16-bit cache, EIG, random, and nearest attain pooled NLLs 0.3544, 0.4497 and 0.4772, and accuracies 81.18%, 73.34% and 71.63%. The comparison below conditions on the same ambiguous-case stratum for all methods, defined for evaluation only by the old Full IQ smooth posterior in $[0.1,0.9]$. All 2,048 cases qualify in each of the three listed profiles. The +24 dB profile has no qualifying cases and shows a ceiling tie between policies at 16 bits.
207:### 5.3 Rerun and interpretation
215:Pointwise intervals are paired by mother and conditional on this specified simulator and approximate s17 calibration. The cache risks use common nominal gain/noise metadata, unlike an acquisition where these quantities must be estimated. Independent signal-session data are needed to establish held-out calibration, transfer, and latency. The blind simulation enforces one release per policy within a common process. A field implementation should isolate the sealed candidate store from the predictor and record the commit and reveal separately.
217:## 6. Round 2 handoff
219:- **M1.** Freeze the factored physical evidence interface, 16-bit candidate encoding, and exact legacy prefix. Preserve C4 transformation laws on candidate probabilities, and transform dynamics and reference factors consistently. Use matched old IQ to measure the separate cache and host readout gaps. Keep the continuous physical decoder as an oracle.
220:- **M3.** Freeze acquisition ID, timestamp, association probability, calibration, and noise-dependence fields. Verify idempotence for reused sources. Compare co-temporal pooling and cross-time messages against the joint likelihood. Keep the posterior mixture available for pre-reveal forecasts.
221:- **M4.** Use session- or acquisition-disjoint tuples. Calibrate SNR, gain and quality from allowed old data or a separate calibration split. Stratify natural, injected, and independent acquisitions, including low SNR and high fading. Test typed DC/spur projections when a scalar residual fails. Use a separately labeled oracle-calibration arm where needed.
222:- **M6.** Freeze risk margins by query and capture stratum, candidate-access rules, and held-out proof of sealing before training larger hosts. Compare 12 and 16 bits with all storage and read costs accounted for. Include the continuous ceiling, ambiguous-case coverage, a fixed-best development baseline, forecast calibration, step-by-step I/O and latency. Preserve unidentifiable and tied cases in the report.
```

### Event #10: `item.completed` — `command_execution` (`item_8`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$a=Get-Content -Encoding UTF8 research/stage4_math_and_design/M2/THEOREM_M2.md; $a[98..148]; $a[242..281]; rg -n '"'"'^#|rank|action|Arm|matched'"' research/stage4_math_and_design/M1/THEOREM_M1.md"
```

**Aggregated Output:**

```text
## 3. Stage 4 frozen statistical gate

A claim passes when every endpoint for that claim clears its limit. Freeze the candidate, seed handling, allowed observation sets, per-class weighting, code below, multiplicity family, score floor, calibration, and all margins before unblinding. Apply the same three gate parts separately to the synthetic extrapolation claim, the Pluto claim and the CTTC claim. For a claim covering all four synthetic cells, require each cell to pass. Report each failed or indeterminate endpoint. A winner picked from several candidates requires a design split or inclusion in the multiplicity family.

1. **Absolute and comparative performance.** Use paired lower bounds. Require NLL to beat uniform, the dev-frozen prior, and the matched zero-phase system, with a \(0.02\)-nat increment proposed against matched zero. Require accuracy above analytic chance, a dev-frozen prior classifier, and the matched zero-phase system, with \(+0.02\) absolute over chance and zero proposed as initial margins. With imbalanced classes, also require outperformance of the fixed majority prediction or used prior competitor, and report balanced accuracy as a separately locked estimand. Require valid chance/random-label randomization tails for the accuracy and NLL statistics where the design supplies exchangeability. Choose an oracle with the same information budget as the candidate for the main NIG denominator and disclose a finer physical oracle separately. Propose \(\eta_{\min}=0.10\) and validate \(H(Y)-L(o)>0\) on an independent development design or by a co-primary lower bound. Test the linear contrast \(D_{\eta}=(1-\eta)H(Y)-L(q)+\eta L(o)>0\) rather than dividing two noisy quantities. Use \(H(Y)=\log K\) for a genuinely uniform target law. Otherwise predefine the target distribution or carry uncertainty in \(H(Y)\) by a joint confidence region and minimize the contrast over it. For real data without a defensible Bayes oracle, freeze a physically informed predictive reference, call the resulting metric reference-normalized, and withhold the Bayes-normalized claim. Requiring a learner to outperform a true Bayes oracle under proper loss is infeasible. A positive NIG floor defines a fraction of the attainable improvement. Controlled slot/text interventions form a further prerequisite for a specifically signed-slot mechanism claim. Require improvement with matched versus permuted or flipped slots at fixed text and same \(X\), with seeded training and ablations paired. Use the native receiver outputs for the gate.
2. **Calibration and structure.** Propose an upper bound of 0.05 on top-label ECE after dev-only calibration with ten frozen bins. Propose a paired Brier improvement of at least 0.005 against the strongest locked trivial or zero-phase Brier competitor, plus a strict improvement over every required comparator. For \(K=4\) uniform Brier is \(1-1/K=0.75\) with the unnormalized multiclass sum. Prelock a probability transformation ensuring finite NLL, for example \(q_\epsilon=(1-K\epsilon)q+\epsilon\) with small fixed \(\epsilon\), and score that forecast consistently. For calibration inference set \(d_j=\mathbb E[1_{\{\operatorname{conf}\in I_j\}}(h_Y-\operatorname{conf})]\); \(\operatorname{ECE}=\sum_j|d_j|=\max_{v\in\{-1,1\}^{10}}v^\top d\). Test all \(2^{10}\) linear upper-margin conditions as an IUT or use a valid joint confidence region for \(d\). Estimating the absolute value first and treating empirical ECE as an unbiased, normal per-mother loss has no exact justification. Require \(\max_{g,x}\|q(gx)-\rho(g)q(x)\|_\infty\le10^{-5}\) and projection/readout idempotence \(\max_x\|P(Pq)(x)-Pq(x)\|_\infty\le10^{-5}\). Verify across the whole finite code domain and all group elements on synthetic. For continuous real inputs distinguish formal architectural guarantees from the observed maximum on prespecified transforms. Test only valid actions with physically correct label permutations. Audit the deployed path and the offline projector separately.
3. **Hierarchy and multiplicity.** Keep all repeats, rotations, acquisitions, and matched arms inside their mother and higher cluster. Report a frozen-seed conditional estimate and, if claiming generalization to retraining, a separate random-seed inference. Use equal independent cluster weights for the following exact balanced analysis. Pluto supplies 16 units if they are independent captures. For a CTTC new-session claim use nine independent session means, with all eight clusters per session contributing to that session. A conditional claim on the fixed sessions needs the corresponding estimand and defensible within-session cluster independence. For a permutation or label-randomization gate, lock the allowed exchangeability orbit and recompute the same statistic throughout. Repeated/rotated labels move jointly with their mother and never split correlated session units as though independent. A synthetic randomized design with independent label assignments can instead rerandomize the sanctioned generator operation inside the original design. When there is no justified exchangeability, collect a randomized negative control. State whether it tests a sharp random-label null or a mean-effect margin. A permutation p-value \((1+\#\{T_{\rm perm}\ge T_{\rm obs}\})/(R+1)\) with uniform randomizations and ties included is conservative under the sharp design. Use exact enumeration when feasible.

### Exact type-I control and power under a balanced Gaussian model

For an oriented block contrast \(D_{bs}\) let \(D_{bs}=\mu+u_b+\nu_s+e_{bs}\), with independent Gaussian families of variances \(\sigma_u^2,\sigma_\nu^2,\sigma_e^2\). These are the assumptions for the exact parametric calculations. Blocks already aggregate mothers and correlated views. If \(m\) mothers in a block have common variance \(v\) and exchangeable within-block correlation \(\rho\), a block mean contributes \(v[1+(m-1)\rho]/m\); this design effect applies to the particular contrast and variance component being modeled. The Stage 3D Work E lock estimated block SD on a fixed-seed pilot, took the larger SD from pilot upper bound and design-effect scenario and used the conditional noncentral t. It did not price uncertainty across retrains.

For \(B\) blocks and \(S\) seeds,

\[
V=\operatorname{Var}(\bar D)=\sigma_u^2/B+\sigma_\nu^2/S+\sigma_e^2/(BS).
\]

Orthogonal row, column, and residual contrasts yield mutually independent ANOVA terms \(MS_B/(\sigma_e^2+S\sigma_u^2)\sim\chi^2_{B-1}/(B-1)\), \(MS_S/(\sigma_e^2+B\sigma_\nu^2)\sim\chi^2_{S-1}/(S-1)\), and \(MS_E/\sigma_e^2\sim\chi^2_{(B-1)(S-1)}/[(B-1)(S-1)]\), independent of \(\bar D\). The corresponding method-of-moments component estimates are \((MS_B-MS_E)/S,(MS_S-MS_E)/B,MS_E\). The reported planning plug-in truncates negative estimates to zero and retains the raw estimates. \(MS_S(S-1)/(B\chi^2_{S-1,0.05})\) is a conservative 95% upper bound for \(\sigma_\nu^2\). Three seeds give only two independent seed degrees of freedom. Large \(B\) cannot remove the \(\sigma_\nu^2/S\) floor.

For the **frozen seeds**, set \(v_F=\sigma_u^2+\sigma_e^2/S\). The ordinary one-sided block t-test has exact boundary size \(\alpha\) for the fixed-seed mean and power \(1-F_{\mathrm{nct}}(t_{B-1,1-\alpha};B-1,d\sqrt{B/v_F})\) at true excess \(d=\mu-\delta\) above the margin, with the mean interpreted conditionally. If one applies this t-test to a population-of-seeds claim, the cutoff for the exact unconditional noncentral t is rescaled by \(\sqrt{(v_F/B)/V}\); the nominal type-I error can then be badly inflated. [SciPy noncentral t definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html)

For **random seeds**, a finite-level test with unknown Gaussian variance components uses
\[
 W=(MS_B+MS_S)/(BS)=A X+C Y,\quad
 A=(\sigma_u^2+\sigma_e^2/S)/B,\quad C=(\sigma_\nu^2+\sigma_e^2/B)/S,
\]
where \(X,Y\) are the independent normalized row and column chi-squares. Reject when \(\bar D-\delta>k\sqrt W\) with \(k=t_{\min(B-1,S-1),1-\alpha}\). At the boundary the exact rejection probability, also the power formula at excess \(d\), is
\[
\pi(d)=\mathbb E_{X,Y}\overline\Phi\!\left(\frac{k\sqrt{A X+C Y}-d}{\sqrt V}\right).
\]
To prove size control, note \(A+C=V+\sigma_e^2/(BS)\ge V\) and \(x\mapsto\overline\Phi(k\sqrt x)\) decreases and is convex for \(x>0\). Jensen implies \(\pi(0)\le\frac{A}{A+C}\Pr(t_{B-1}>k)+\frac{C}{A+C}\Pr(t_{S-1}>k)\le\alpha\). The supremum \(\alpha\) is attained as the smaller-df variance component dominates. With known component ratios use the exact nuisance-specific mixture cutoff. The unconstrained method-of-moments estimate \((MS_B+MS_S-MS_E)/(BS)\) can be negative and has a signed chi-square mixture law. Plugging it into a t distribution would not be an exact test.

For real data under the same Gaussian model with \(J_0\) independent sessions of \(m\) nested clusters and separate session and cluster effects, the session means have
\[
V=\sigma_{\rm session}^2/J_0+\sigma_{\rm cluster}^2/(J_0m)+\sigma_\nu^2/S+
\sigma_e^2/(J_0mS).
\]
Use the same formulas with \(B=J_0\), \(\sigma_u^2=\sigma_{\rm session}^2+\sigma_{\rm cluster}^2/m\), and \(\sigma_e^2\) replaced by \(\sigma_e^2/m\); an additional session-by-seed effect is another contribution to the collapsed interaction variance. Use planned balanced session-level estimation or a separately validated unequal-weight design. For a fixed-seed claim with independent bounded cluster means and arbitrary within-cluster dependence, a one-sided Hoeffding bound gives an alternative to Gaussian t inference. For mean of \(B\) equal-weight cluster means in \([l,h]\), reject when the empirical excess exceeds \((h-l)\sqrt{\log(1/\alpha)/(2B)}\). For fixed weights \(a_b\) summing to one, replace \(1/B\) by \(\sum_b a_b^2\). Unknown non-Gaussian random seed effects with only three seeds do not admit a generic exact t power calculation. Correlation modeling and null randomization require separate justifications.

For endpoint nulls \(H_{0j}\) and a joint claim, the null is their union. Valid p-values give \(p_{\rm IUT}=\max_j p_j\), since with at least one true null the event \(p_{\rm IUT}\le a\) is a subset of rejection of that true null. The exact joint rejection probability is the joint tail of the endpoint statistics, equal to the product of marginal powers only under independence. Its universal lower bound under the alternative is \(1-\sum_j(1-\pi_j)\). Across separately reportable claims, sort the IUT p-values and apply Holm's thresholds \(\alpha/(M-i+1)\). If a first true null is rejected, its p-value is at most \(\alpha/m_0\) where \(m_0\) is the number of true claim nulls. The union bound over those true nulls gives strong FWER \(\le\alpha\) under arbitrary dependence. This is a finite-sample guarantee conditional on valid endpoint tests. Under independence with all nulls uniform, the probability of any rejection is exactly \(1-(1-\alpha/M)^M\). [Holm (1979)](https://www.jstor.org/stable/4615733)

For simple variance planning at known components, if \(z=z_{1-\alpha}+z_{1-\beta}\), power \(1-\beta\) needs
\[
B\ \ge\ \frac{\sigma_u^2+\sigma_e^2/S}{d^2/z^2-\sigma_\nu^2/S}
\]
with positive denominator. At fixed \(S\) this is a necessary known-variance bound for the random-seed model. The verifier solves the exact t power and evaluates the crossed chi-square integral for finite counts. It uses a worst Holm first-step threshold \(\alpha=0.05/3\) and marginal power 0.95 to guarantee at least 0.90 power for the two historical paired endpoints if both assumptions and excess effects hold. This target does not imply 0.90 power for the full Stage 4 gate. Full planning requires variance and target excess for every required absolute/calibration endpoint and a joint power calculation or an explicit allocation of endpoint type-II errors. Stage 3D supplies a synthetic variance reference. It cannot measure Pluto or CTTC variability.




## 5. Power replay and Round 2 lock proposal

The legacy pilot replay gives endpoint minima 131 and 54 blocks for accuracy and NLL, and powers 0.950512 and 0.999906 at 131. Its scenario SDs are 0.094414 and 0.079795. The crossed G1 means are 0.043926 accuracy and 0.035650 NLL. The fitted variance components on the real block-by-seed matrix are as follows.

| Contrast | Block variance | Seed variance | Interaction variance | Seed SD estimate | 95% upper bound on seed SD |
|---|---:|---:|---:|---:|---:|
| Accuracy | 0.0000758565 | 0.0003711426 | 0.0005829523 | 0.01927 | 0.08557 |
| NLL | 0.0001213570 | 0.0004670843 | 0.0003618383 | 0.02161 | 0.09571 |

At the two planning excesses \(d=(0.03,0.04)\), one-sided \(\alpha=0.05/3\) and marginal power 0.95, the synthetic fixed-seed block minima for accuracy/NLL are G1 7/5, G2 11/7, G3 8/6, G4 11/4 and H 9/6. These are conditional t minima for hypothetical positive effects. Observed G2 and G4 effects favor zero and fail the positive comparative target. G3's accuracy-only random-seed minimum is 42 under its zero-truncated seed variance estimate from eight blocks. Three seeds do not give a viable 95% crossed NLL power design in any cell under the plug-in scenarios. The small conditional minima do not override the frozen synthetic budget or planning from independent upper variance bounds.

G1-proxy sample sizes and powers follow. CTTC uses session means across eight clusters per session. Its ICC parameter is the assumed fraction of the proxy block variance shared at session level, with no additional session-by-seed effect in this sensitivity run. “Minimum” means the first integer yielding at least 95% marginal power under the specified test and effects.

| Setting | Available independent units | Three fixed seeds, minimum units accuracy/NLL | Three random seeds, power accuracy/NLL | Ten random seeds, minimum units accuracy/NLL | Ten random seeds, power accuracy/NLL |
|---|---:|---|---|---|---|
| Synthetic G1 proxy | 131 blocks | 7 / 5 | .224 / .300 | 15 / 7 | .981 / .998 |
| Pluto proxy | 16 captures | 7 / 5 | .136 / .234 | 15 / 7 | .953 / .994 |
| 72 independent clusters sensitivity | 72 clusters | 7 / 5 | .212 / .293 | 15 / 7 | .978 / .998 |
| CTTC proxy, ICC .10 | 9 sessions | 4 / 3 | .208 / .287 | 7 / 5 | .973 / .997 |
| CTTC proxy, ICC .25 | 9 sessions | 4 / 4 | .202 / .280 | 7 / 5 | .970 / .996 |

At the actual G1 excesses over \(0.02\), the three-fixed-seed G1 or Pluto-proxy minima are 10 and 17 blocks/captures, and the CTTC proxy at ICC .10 gives 4 and 6 sessions. For three random seeds at the planning excesses, the ceiling even with known components and infinitely many blocks is 0.71537 and 0.85941. The conservative finite-level test tends to 0.23839 and 0.30978. These ceilings are consequences of the fitted seed variance and fixed effect-size assumptions. The high potential sampling uncertainty in that variance makes the ten-seed numbers planning proxies. Even with ten seeds, the modeled power statements apply only to the two relative endpoints.

The verifier drew 200,000 Monte Carlo replicates per scenario from the independent normal and chi-square sufficient statistics. G1 at 131 blocks gives the following analytic/Monte Carlo boundary rejection rates at nominal \(0.01667\).

| Endpoint | Fixed-seed t | Crossed test | Block-only t with random seeds |
|---|---|---|---|
| Accuracy | .01667 / .01654 | .01161 / .01168 | .39171 / .39145 |
| NLL | .01667 / .01716 | .01280 / .01277 | .40810 / .40951 |

For random-seed G1 power the corresponding analytic/Monte Carlo pairs are .22385/.22447 and .30028/.30242. The correlated Holm-IUT MC false-claim family rates range from 0.04002 to 0.04953 under the three simulated null configurations. The proof establishes protection beyond those simulations. ECE and the remaining performance gates need their own pilot variability or a joint design simulation.

G1's same-mother Work F fixed-seed 95% block intervals for the all-rotation interventions are \([0.002804,0.003659]\) for text-only flip and \([0.016552,0.017927]\) for slot-only flip. The seed-wise slot effects are \(-0.000641,+0.035214,+0.017147\). A pooled interval over mothers cannot establish a stable retraining effect.

### Round 2

- **M1** should lock matched-text phase, text-only, joint-blind and independently permuted phase-slot controls, with a signed-slot intervention at fixed text and matched waveform. Cross \(|\omega|\) and accumulated angle independently where the simulator allows it, preserve the analytic S17 and full-IQ anchors, and keep sign/rotation/mother identifiers together for inference.
- **M3** should score the native equivariant forecast against the raw, arithmetic, geometric and canonical-representative paths on a common development split. Calibrate on development data that span the prospective regimes and freeze the choice before blind scoring. Lock all absolute endpoints before comparing phase and zero. The G1 raw and pooled paths fail absolute accuracy, and the blind-fitted paths have below-chance accuracy and negligible NIG.
- **M4** should keep the 131-block synthetic budget until independent gate-endpoint pilots justify changes. If claiming robustness across retrains, plan around ten or more paired seeds as a G1-based starting point and recalculate using pilot upper variance bounds. With three seeds, scope the main claim to the frozen connectors. Obtain separate Pluto capture and CTTC session variance pilots and freeze session IDs and permutation strata. Preserve all nine CTTC sessions in inference, and use the power objective for the complete gate before locking the real-data sample size.

1:# M1 Round 1 — Quotient equivariance and signed phase transport
5:Stage 4 makes the quarter-turn output action an explicit permutation and transports evidence by the signed physical phase before sector readout. The permutation law holds independently of the frozen host. The transport accounts for the Stage 3D direct-table recovery near a quarter turn. The Qwen failure is empirical. Separate token positions can interact, and neither a universal below-chance bound for extrapolation nor a fixed sample-complexity penalty follows from the architecture.
7:## 1. Exact discrete symmetry
9:Use \(\mathbb Z_4\) labels and roots \(i^j\) in the four-choice order. Q1 asks for \(X_2-X_1\), Q2 asks for \(X_1\), and Q3 asks for the sector of the third matched-filter output. The sectors are \(I_k=[-\pi/4,\pi/4)+k\pi/2\) modulo \(2\pi\). With \(T=\sum_{m=1}^3Y_m/\sqrt3\), window times \(0,h,\tau=h+\ell\), and \(\rho=10^{\mathrm{SNR}_{\mathrm{dB}}/10}\), the Stage 3D generator has
19:### Theorem 1 — Encoder and conditional choice actions
31:**Proof.** The encoder takes the sector of the sum of the three samples, falling back to the first nonzero sample on a zero sum. The half-open convention gives \(\operatorname{sector}(i^rT)=\operatorname{sector}(T)+r\) for nonzero \(T\), including boundary rays. Undoing the selected turn gives the same canonical I/Q features after rotation. The fitted MLP and center assignment therefore return the same inner index. The all-zero window is fixed and cannot carry a nontrivial turn. It has probability zero under the stated noise. A hardware interface can mark it as an erasure. [Encoder](../../matched_cache/learned.py)
33:For the joint law, relabel \(X_1\mapsto X_1+r,\ X_2\mapsto X_2+s\), and rotate noises in windows 1, 2, and the repeat window by their designated turns. Keep \(G,\phi\), and the pilots fixed. The uniform source and circular noise laws preserve the joint measure. The three nonuniform label shifts then follow by subtraction, by reading \(X_1\), or by taking the sector. This yields the conditional identities on the support. For the fresh symbol, condition further on \(G\) and the entire phase path. The new symbol and noise are still independent, and mixing the four roots makes its received law quarter-turn invariant. Marginalization preserves the uniform sector law. \(\square\)
37:### Theorem 2 — Quotient factorization
49:## 2. Continuous phase transport and the sector dip
51:### Theorem 3 — Circular phase evolution before the sector map
62:The quarter-turn actions commute with continuous translations wherever both are realized as rotations of the underlying continuous variables. Conjugation therefore gives the trivial \(S^1\to\operatorname{Aut}(H)\) action. The corresponding semidirect product is a direct product, possibly with a kernel if two factors rotate the same variable. Any continuous action \(S^1\to\operatorname{Aut}(H)\) is trivial because the domain is connected and the codomain is finite. The same reasoning precludes a nontrivial continuous action on the finite codes or choices by permutations. A CFO change is an \(\mathbb R\) flow on the joint edge-phase torus, with edge phases \(\Delta\omega\,d\). It has a common frequency period only if the active durations are commensurate. [Group-action background](https://proceedings.mlr.press/v48/cohenc16.html)
64:### Theorem 4 — Circulant Q3 pushforward
94:### Corollary — Why \(\pi/4\) and \(\pi/2\) give different accuracies
116:## 3. Stage 3D diagnostics and the scope of separation
127:Work F gives orientation gain \(+0.00982\) and mean swapped-minus-matched NLL \(+0.01837\) across phase seeds. The phase seed-averaged table has mean-square metadata effect \(1.85378\cdot10^{-4}\), waveform effect \(1.11977\cdot10^{-3}\), and interaction \(6.36253\cdot10^{-5}\). The physical calibration has metadata effect \(0.201603\), over a thousand times the host's mean-square value. Work G has rot01 and rot11 mean maximum coordinate differences of \(0.05914\) and \(0.12627\), with worst tested discrepancy \(0.48167\) at rot01. These measurements establish weak response in the correct signed direction and broken quotient equivariance for the trained host. [Stage 3D summary](../../stage3d_qwen_signed_phase/results/summary_master.json)
129:### Theorem 5 — Additive centered-logit obstruction
131:Suppose centered logits satisfy \(L(z,\phi)=U(z)+V(\phi)\). The signed phase contrast across codes has rank at most one and the mixed finite difference in \((z,\phi)\) vanishes. If the same readout is exactly \(H\)-equivariant for a task with a surjective output shift, the contrast is zero.
135:Separate token emission does not satisfy the theorem's final-logit hypothesis. A frozen attention head can form \(q(z)^\mathsf Tk(\phi)\) with rank up to the projected key dimension. A fixed projection using two coordinates can form \(x(z)\cos\phi+y(z)\sin\phi\), and a second gives the other component of a complex rotation. The host also has nonlinearities and a numerical text stream. For the released fit, the output architecture enforces neither an equivariant code action nor a continuous transport law. The fit's measured extrapolation failure does not establish an expressivity limit of frozen hosts. [Scaled dot-product attention](https://arxiv.org/abs/1706.03762)
137:#### Work F spectrum
139:For code pair \(z\), use \(D_z=p_+(z)-p_-(z)\) from `cal_ref` and \(w_z=\min(m_+(z),m_-(z))/\sum_{z'}\min(m_+(z'),m_-(z'))\) from the archived conditional code masses. Compute \(M=\operatorname{diag}(\sqrt w)D\) and the Gram spectrum of \(M^\mathsf TM\). The row sum is zero. In a cyclic orbit of a row, a nonzero frequency-1 component spans a real plane and a nonzero frequency-2 component adds the Nyquist line. The verifier reconstructs the half-count ratios and finds both on the positive-weight orbit of code pair index 0: rows \(0,64,128,192\). Thus the **archived calibration table has rank three**. This is a statement about the finite reference table, with its calibration sampling error.
141:| Contrast | Nonzero Gram eigenvalues | Frobenius norm | Stable rank | Effective rank |
146:With \(p_i=\lambda_i/\sum_j\lambda_j\), the effective rank is \(\exp(-\sum_i p_i\log p_i)\) and the stable rank is \(\sum_i\lambda_i/\lambda_1\). An arbitrary rank-one matrix would retain at most \(50.19\%\) of the physical Frobenius energy and leave a relative Frobenius residual of \(0.70576\). The host phase contrast has \(3.52\%\) of the physical Frobenius norm and relative Frobenius error \(0.99076\). Work F's orientation gain is its normalized Frobenius projection onto \(M\). Its factorial effect instead uses a paired-mother measure and a signed contrast scaled by \(1/2\). A probability-spectrum rank is different from attention rank or additive-logit rank, since softmax is nonlinear.
148:### Theorem 6 — Reynolds averaging and NLL
160:## 4. Stage 4 `EquivariantEvidenceConnector`
170:### M3 interface — `RefSufficientPool`
172:- Per acquisition accept `acquisition_id`, `shares_channel`, `timestamp`, `snr_linear`, known pilot root or sequence, and `matched_T` in the declared receiver frame. Deduplicate by acquisition identity before combining likelihoods. Keep fresh independently acquired pilots, including those at a shared timestamp. Independent-channel rows have no shared-channel likelihood factor under the stated model.
176:### M4 interface — `PhaseContext` for Pluto and CTTC
178:- Supply `t1`, `t2`, `t3` and per-pilot timestamps in a common time unit and clock basis. Supply signed `omega_rad_per_unit` under \(T(t)\propto e^{+i\omega(t-t_1)}\), `diffusion_var_per_unit`, `snr_linear` with matched-filter normalization, and the frame and tracking state. Synthetic mapping is `t1=0`, `t2=h`, `t3=h+ell` and variance rate \(\sigma_\phi^2\). CFO in Hz uses \(\omega=2\pi f_{\rm CFO}\) with time in seconds. If the front end already de-rotates I/Q, pass the residual under the same sign convention.
180:- Expose observation confidence and erasures, and specify any mismatch to the three-sample encoder, matched-filter normalization, circular noise, or drift law. Code-space equivariance is independent of the hardware label accuracy.
182:## 5. CPU verification and Round 2 decisions
190:The verifier uses the production loader and fingerprint, regenerates the paired mothers, and matches all blind seeds, code pairs, and repeat targets to the archive. It uses 20,480 signed-CFO training rows, 16 fixed epochs, three optimizer seeds, and a single shared frozen MLP architecture with identical weights and fixed text embedding. A has 6,936 trainable parameters and B has 6,520. B uses a frozen physical prior while A learns from the inputs only. Scores use fractional ties. Paired intervals average signs and rotations within each 64-mother block. The Q1, Q2, and pilot-task tests exercise the quotient lift with synthetic canonical scores. Trainable end-to-end metrics are for Q3 repeat, with Q3 independent evaluated as a uniform control.
203:- **M2** · Which matched Qwen ablation will isolate quotient sharing, the physical head, and token coupling? Should physical-head-only versus physical-head-plus-host-residual be the primary paired contrast?
```

### Event #11: `item.completed` — `command_execution` (`item_9`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$a=Get-Content -Encoding UTF8 research/stage4_math_and_design/M3/THEOREM_M3.md; $a[154..174]; rg -n '"'remote|checkpoint|METRICS|def restore' ../continuity.py"
```

**Aggregated Output:**

```text

The archived first valid matched sums have RMS magnitude 1.27735. Their global I/Q pair moves by RMS 1.80644 under a joint quarter-turn because \(|iT-T|^2=2|T|^2\). The canonicalized pool had zero movement for that action. The normalized absolute cross moments of the raw matched sums are 0.62189 for the two valid pilots and 0.00420 for valid and independent, consistent with the generator's common signal and separately drawn distractor. [Result receipt](results_m3.json)

## 5. Fixed-slot interface and guarantees

1. **Validate and canonicalize.** Use \(z_i=4a_i+u_i\). Resolve IDs and check provenance, calibration and exact sample equality before filtering. A production clock should publish one canonical timestamp per physical tick. For schema-tolerance aliases, the prototype sorts timestamps per ID, chooses the minimum, and rejects a spread beyond \(10^{-12}\). Remove independent-channel references before feature evaluation. A conflict-free distractor contributes no activation or gradient. A literal copy of a canonical acquisition ID makes no change. Sort unique IDs before reduction for deterministic sums.
2. **Keep exact evidence upstream.** Use sign swaps for the integer rotation \(i^{-a_1}\) on I/Q before floating-point reduction. Apply continuous derotation once per clock. Pack `Pooled(iq[B,64,2], times, precision[64], gap[B], family1[B], family2[B])` with zero padding excluded from the reader. The prototype batches one shared public design. A production wrapper can batch by shared design or use a padded per-row mask. Declare \(\omega,\sigma_\phi,h,\tau\) and the noise model as public inputs. The continuous path is differentiable for fixed metadata. ID groupings, channel flags, and clock equality are discrete.
3. **Expose fixed task evidence to M1.** Compute `canonical_q2[4]`, `canonical_q3[4]`, and optionally `joint[4,4]` from the reader. Provide a separate frame \(a_1\) for external sector relabeling. Use one always-present task evidence marker and fixed projection, with identical prose and no raw IDs, counts, flags, or clock lists. A multitask prompt can reserve both markers. The CPU prototype returns a single four-component Q2 slot. A projection to Qwen dimensions and an intervention on the frozen host remain separate integration work.
4. **Use the model posterior for guaranteed decisions.** Arbitrary host scores or trainable residual logits can change posterior calibration and rankings even at fixed length. Keep the physical posterior available as the decision output and the frozen-host readout as an experimental comparator.

**Rotation.** Under the joint coordinate action, \(a_i\mapsto a_i+r,\ G\mapsto G+r\), and \(X_1,u_1,u_2\) are invariant. Thus \(i^{-a_1}T_m\) and \(a_2-a_1\) are unchanged. Use \(G-a_1,U-a_1\) in the canonical kernel. Q2 is invariant and Q3 global sector probabilities permute by \(r\) through an explicit external index rotation. The exact-zero CPU verification covers Q2 and pooled evidence in all four frames. Qwen outputs were replayed from the archive.

**Blackwell.** Dropping the new acquisition maps the augmented experiment to the old experiment. There is an explicit garbling even after same-clock pooling. Given the augmented pooled sum \(S_+\) with total precision \(P_+\), the old sum of precision \(P_o\) has conditional law \(\mathcal{CN}((P_o/P_+)S_+,\,P_o(1-P_o/P_+))\), independent of the physical phase. This follows from joint Gaussian receiver noise with covariance \(P_o\) between the two sums. Across clocks, use this garbling within each clock and drop new-only clocks. An exact posterior preserves all decisions about its target, so the Bayes risk of the augmented posterior experiment improves weakly for every loss. The expected log-loss gain is \(I(X_1;T_{\mathrm{new}}\mid Z_1,Z_2,\mathcal R_{\mathrm{old}})\). For the finite-time Brownian Q2 model with informative \(a_{z_1}\) and a fresh finite positive-SNR pilot this is strict. At zero old evidence, the new pilot derivative of the first target moment is nonzero for some \(z_2\), since the unconditional \(e^{iB_{\mathrm{new}}}\) moment is positive. Continuity and positive observation densities give a positive-measure change in the posterior. This applies to the first pilot and subsequent ones at common or distinct clocks. A copy, independent distractor or zero-SNR acquisition gives equality. [Blackwell, *Equivalent Comparisons of Experiments*](https://doi.org/10.1214/aoms/1177729032)

For Q2 with these same nondegeneracy conditions, strict population MAP improvement also follows. At zero old evidence, choose a new pilot value giving a nonuniform posterior. Quarter-turning **only that new pilot** produces the four label rotations of that posterior, because the old likelihood is constant in \(G\) there. The intersection of the four sets of maximizers is empty. The strict gaps excluding each label from at least one such maximizer set persist in neighborhoods of the new values and of zero old evidence. The conditional new pilot has positive density in every neighborhood. Therefore the expected new maximum posterior probability strictly exceeds the maximum of the conditional mean on a positive-measure set of old observations. This strengthens the sample-level claim to a population statement for the specified model. Accuracy can still go down on individual mothers or a finite test set, and the result supplies no guarantee for a learned host. For Q3 alone, strict log-loss improvement requires its conditional sector distribution to change.

**Fisher information.** A discrete quadrant or symbol has no ordinary differentiable-parameter Fisher information. For continuous deterministic pilot phase \(\theta\), the score \(2\Re[(T-\mu)^*\partial_\theta\mu]/\sigma_m^2\) has variance \(2\alpha_m^2/\sigma_m^2\). The conditional phase information at clock \(j\) is exactly \(2P_j\), with strict increments for positive precision. For a continuous global phase parameter in a marginalized Brownian model the old score is the conditional expectation of the augmented score. The gain in Fisher information is the expected conditional variance, which is nonnegative and strict exactly when that variance is positive.

## 6. CPU replay and paired experiments

Run from the repository root.
1:'''Small, workspace-scoped continuity checkpoints. No model calls or full audits.'''
28:CATEGORY = 'continuity_checkpoint'
34:METRICS = LOCAL / 'metrics.jsonl'
85:        return text + 'No matching workspace checkpoints.\n'
180:    def checkpoint(self, expected, capsule, note):
182:        require(len(capsule) <= 2000 and len(note) <= 16000, 'TEXT_LIMIT: checkpoint is too large.')
195:        record_id = uuid5(NAMESPACE_URL, f'{IDENTITY}:{self.scope}:checkpoint:{revision}')
239:    with METRICS.open('a', encoding='utf-8') as target:
246:    if METRICS.exists():
247:        with METRICS.open(encoding='utf-8') as source:
279:            return startup_text(0, 'No checkpoint; ask for missing prior context.')
285:    if args.command == 'checkpoint':
287:            result = Store(conn).checkpoint(args.expect, args.capsule, args.note)
288:        # Confirm the exact committed row, not a potentially newer concurrent checkpoint.
307:    checkpoint = commands.add_parser('checkpoint')
308:    checkpoint.add_argument('--expect', type=int, required=True)
309:    checkpoint.add_argument('--capsule', required=True)
310:    checkpoint.add_argument('--note', default='')
```

### Event #12: `item.completed` — `command_execution` (`item_6`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$a=Get-Content -Encoding UTF8 research/stage4_math_and_design/M4/THEOREM_M4.md; $a | Select-Object -Skip 105 -First 125; rg -n '"'budget|275|2477|1250|held_out|f708_snr10|sentinel' research/stage4_math_and_design/M4/verify_m4.py"
```

**Aggregated Output:**

````text
\]

If \(U\perp S\), the loss of hiding \(U\) is exactly \(I(S;U\mid T)\). Given a fixed coherent gain, phase and bias, invertible scaling or rotation preserves information up to the effective SNR; a nominal decoder may nonetheless incur the posterior regret in (5). A uniform hidden quarter-turn makes the four single-symbol densities identical, so MI is zero even at infinite physical SNR. Relative-symbol and retransmission questions can still have information through shared latent phase. For fixed circular symmetric noise with an equivariant QPSK model, the uniform prior achieves the constrained capacity by averaging the four rotated input priors. With other empirical code channels, the verifier computes a Blahut–Arimoto lower and upper bound for the fitted transition matrix. These brackets measure optimization error on the Monte Carlo counts and carry sampling uncertainty.

For `s17`, along a fixed nonzero canonical ray \(x=cu\), \(c>0\), the sum sector is scale invariant. At low gain \(\tanh(cuW_1+b_1)\to\tanh b_1\), so the local code becomes constant if the winning cross-entropy score has a positive margin. The hidden deviation is at most \(c\|uW_1\|\). Converting it through \(W_2\) and the \(1/2\)-Lipschitz softmax, the bound on pairwise score changes is \((c/2)\|uW_1\|\|W_2\|_{2\to2}\|\log C_i-\log C_j\|\) for centers \(C_i,C_j\). A strict margin greater than this perturbation proves stability. For high gain, neuron \(j\) with \(d_j=(uW_1)_j\ne0\) approaches \(\operatorname{sgn}(d_j)\) with error at most \(2e^{-2(c|d_j|-|b_{1j}|)}\) once \(c|d_j|>|b_{1j}|\); neurons with zero projection stay at \(\tanh b_{1j}\). Apply the same score-margin test to the limiting logits and center choices. This proves radial saturation on each interior ray with strict margin. The four turns can remain informative. At exact all-zero input the encoder chooses its defined tie behavior. `verify_m4.py` finds `[1,5,9,13]` in the low-gain limit and `[0,4,8,12]` on its high-gain ray. Ties, near-orthogonal directions, and random scales near zero preclude a global AGC threshold.

## 2. Hardware benchmark and causality contract

### 2.1 Candidate rows and time order

Use time in symbols with an explicit sample-clock conversion and a common fixed transmitter mapping \(e^{-i\pi/4}s_{\rm hardware}\) into \(C_4\). Use dataset-specific Gray lookup in the scorer. For Pluto divide the nominal `0.707` axis levels by their fixed constellation magnitude. Use a corresponding fixed receiver coordinate transform, subtract only the causally available DC estimate, and retain the residual channel angle. Each data window contains observations of **one transmitted symbol**. Record the two codes and supporting provenance, release a reference only after the first forecast is committed, and never revise either code. A Q3 forecast stores both the cache-stage and reveal-stage four-vectors before the first sample of the target stencil is eligible. Collector latency is the maximum raw time in each interpolator stencil, with all clocks and cutoffs materialized.

| Dataset | Q1 and Q2 window schedule | Revealed references | Q3 target |
| --- | --- | --- | --- |
| Pluto 850 MHz | One row per frame. Copy 1, \(k_1=30+(f\bmod67)\); \(k_2=k_1+h_{12}\) with \(h_{12}=1\) or 8 using `(f + floor(f/16)) mod 2`. | Two acquisitions at logical indices `15` and `18` of copy 1. Initial state uses sync `0:13` and split-local past, with those two pilot reads excluded from all pre-reveal quantities. | Same payload slot \(k_1\) in the immediately following complete copy, \(L=126\), \(\ell=126-h_{12}\). Score 2,240 candidates with a common lock-failure policy. A copy is a real retransmission; no target-label matching or synthetic replacement. |
| CTTC 433/708/2450 MHz | One row per packet with ordinal \(j\) and loop \(b\). \(k_1=136+((17j+5b)\bmod48)\); \(k_2=k_1+h_{12}\), using 1 for even \(j+b\), 8 otherwise. | `126,127` of the same packet, both earlier than \(k_1\). Initial state uses `0:6` sync and authorized previous data, excluding both release samples. Estimate their joint covariance. `254,255` remain unavailable until they have physically arrived. | On each capture use disjoint loop pairs `(0,1),(2,3),(4,5),(6,7)` at the same packet source start and same \(k_1\) selected in the earlier loop. \(L=500000\), \(\ell=500000-h_{12}\). Q1/Q2 cover 1,440 packets. Q3 has 720 retransmission candidates and 36 loop-pair groups. |

Q1 label is \((x_2-x_1)\bmod4\) from the transmitter vault. Q2 predicts \(x_1\) in the absolute transmitter convention with `none`, `one`, `two`; the second reference is an actual sample at a different time with its own known symbol and possible correlation. Q3 predicts the sector of the future measured scalar \(e^{-i\pi/4}\tilde y_3\) in a fixed receiver phase convention. A DC-centered target is also defensible if the subtraction is frozen at the initial cutoff for every arm; use the raw-sector definition above as the primary. Q3 can be near uniform at the CTTC one-second lag. A fresh unrelated packet with a different source index cannot be called a known retransmission. For Q2, the same four-vector supplies symbol NLL, MAP SER and per-bit NLL from \(P(b_v=j)=\sum_s p(s)\mathbf1[b_v(s)=j]\). Define posterior-mean RMS EVM as \(\sqrt{\sum_k|\sum_s p_k(s)s-s_k|^2/\sum_k|s_k|^2}\) on the same rows. Conventional equalized-IQ EVM, when an arm returns physical \(\widehat s_k\), is a separate named endpoint with the same denominator. Q1/Q3 classification is scored on its own target, with Q2 anchoring receiver metrics rather than pretending the Q3 label is a TX bit.

The primary samples retain actual fading, residual DC/ISI, noise, clock offset, and any changes present at registered times. A naturally occurring jump is a stratum, with changes flagged by a predeclared causal split-local pilot innovation criterion and a retrospective scorer-only morphology check. A CPU-only cross-frame Q1 diagnostic pairs `(f−1,f)` at `f mod 20 = 0` within each test block, using the earlier frame's \(k_1\) and current frame's \(k_1\) from the table. Its matched control pairs the preceding two interior frames `(f−2,f−1)` in that same block. Use only available sync and authorized pilots and a gap-class prior when the time between Pluto columns is unavailable. This samples recorded batch boundaries by schedule without requiring or asserting a jump at each boundary. Separate source domains and nominal session strata in analysis, keeping capture IDs out of model text/features. Hold `f708_snr10` out of fitted connector weights, temperatures and shared prior tuning to expose one capture transfer; allow its own causal sync and released pilots. Pluto is a blocked same-campaign test. CTTC has one public cyclic TX sequence; input-source disjointness and capture dependence are distinct issues. Use block-paired NLL/SER differences on 16 Pluto groups and 72 CTTC session-loops, group adjacent repeat pairs for CTTC Q3, and report a capture-level sensitivity analysis and source-ordinal sensitivity because loops/captures reuse source positions. Report all registered rows and lock coverage, with the same uniform predictive fallback on failed rows and a secondary locked-only analysis. Avoid treating all packets as independent. Select the Q2 `two−one` proper-loss contrast and large-angle synthetic Q3 coupled-vs-additive contrast before test, with paired intervals and the multiplicity rule frozen with M2.

Keep the natural recordings as the primary results. Recreate seven Pluto stress recipes on 140 frame IDs `range(4800,7040,16)` each, and all 1,080 CTTC stress IDs from the branch identity rule (`i mod 4 != 3`, round-robin modes). Score a single two-reference Q2 vector per registered stress row in the GPU budget. For perturbation strength use frozen train-scale or authorized pre-cutoff sync/pilot-only power. Pluto’s legacy injection measures power across the full frame, and CTTC’s legacy injection uses `offline_sig_pow` derived with full TX truth. Their historical raw scores stay available as historical comparators. Hide exact injected CFO, noise power and jumps in the scoring vault and keep RNG independent of payload labels. Real-only CFO uncertainty is estimated from observed pilot evidence. [Pluto data and acquisition](https://data.mendeley.com/datasets/k72b4pnc5g/1) · [CTTC data and acquisition](https://zenodo.org/records/4895133)

### 2.2 What zero leakage requires and what was verified

**Split proof.** Pluto uses frame intervals `[0,3200)`, `[3300,4500)`, `[4600,7040)` including the first 200 warmup frames of each split. The 100-frame gaps imply pairwise empty intersections. All copies of a frame are inseparable across splits. CTTC uses six disjoint cyclic source ranges of length-500,000 circumference, including `train target [0,160000)`, `train ref [164096,200000)`, `dev target [204096,290000)`, `dev ref [294096,330000)`, `test target [334096,455000)`, `test ref [459096,495904)`. The boundary gaps are at least 4,096 symbols, including the wrap. The verifier regenerates starts `lo + i floor((hi−lo−256)/n) + 128`, enumerates both bit pairings' union and asserts disjoint bit footprints across split and role. On the archived nine anchors the 5,472 target/reference acquisitions fit inside their files, have distinct IDs, and the nearest receive ranges remain 16,481 samples apart after 255-sample RRC support expansion. This verifies the stored geometry, including 72 test groups. Acquisition edges must be rechecked on fresh causal alignment. Source sequence reuse within a split creates statistical dependence; the proof addresses explicit identity and support leakage.

**Filtration proof.** The released model payload comprises timestamped W1/W2 entries and a causal state with nuisance parameters estimated only from authenticated training/allowed sync and prior pilots. Each state node records raw-support intervals, truth whitelist, availability time and acquisition IDs of its parents. Acquisition truth is delivered as individual permitted known symbols; the feature worker has no access to `Tx_data` payload or the full CTTC source sequence. Build an independent scorer vault containing all targets, withheld probes, archival oracle fields, future IQ and injected parameters. Model projections strip frame, capture, packet and source ordinals, plus provenance fields that could identify cyclic TX labels. Prompt content carries lags, units and predeclared uncertainty summaries. Auditable salted observation IDs stay outside the prompt and are used for exact de-duplication. Denote the allowed baseline information by \(\mathcal F^0\). Its closure under deterministic functions and independent lift randomness contains the committed cache. Updating only by the union with released past pilot observations yields \(\mathcal F^r=\mathcal F^0\vee\sigma(R_1,\ldots,R_r)\). A listed input lies in this filtration by induction on the provenance graph. Intersection checks of all authorized raw read supports with actual withheld-symbol read stencils, and the inequality on every raw cutoff, certify direct observation masking once the time geometry is established. The verifier checks disjoint logical indices and local cubic-stencil examples over the registered ranges. Adjacent samples can retain analog contributions through real ISI, matched filtering or DC memory. Physical dependence is part of the channel model and cannot be excluded by an index proof.

**Timing and CFO gates.** The legacy Pluto `fast_blind_frontend` correlates whole captures at lag 2520 and scans two copies for header timing. Even randomized payload and unlabelled probes can enter those sums. CTTC's `anchor_capture` uses whole-source lock candidates at several times and long known-source CFO fits, and extraction includes `offline_cfo_hz`, `h_offline_ref` and powers derived using all TX symbols. None of these are admissible causal inputs. For Pluto use an independently registered trigger or bounded timing prior established from earlier authorized sync, verify the **entire candidate search stencil** fits in the allowed sync envelope, then track only with a prefix of the intended copy. For CTTC reserve a disjoint in-cycle commissioning segment, for example source `[496256,498304)` plus sufficient guard for RRC/padding. Use its whitelisted bits and receiver samples strictly before the first use at a given cutoff to resolve EVEN/ODD and coarse timing/SFO. Use the public clock rates as fixed constants. Require a timing-uncertainty certificate for subsequent local interpolation and renewal, and exclude same-split or cross-session reference observations that arrive later than the prediction. The legacy CTTC role order places the same-loop reference pool after its target range, so only already acquired same-split entries may supply history. Reset state at split boundaries and document commissioning as a distinct authorized acquisition with a disjoint source footprint. If safe synchronization cannot be certified, emit a common fallback and report coverage. Strict raw masking and the full usable row counts are conditional on this gate. A Pluto window with an unconstrained initial timing offset cannot certify strict raw masking from those raw samples alone: for every tentative input position some admissible circular offset places a held-out slot there. An offline alignment-assisted diagnostic may be reported with its timing provenance visible.

Use causal sync and then revealed pilots for residual CFO. For synchronized observations \(y_p\) and authorized \(s_p\), fit \((H,d,\omega)\) using the pilot-only profile loss

\[
\mathcal L(\omega)=\min_{H,d}\ \|K_p^{-1/2}(y_p-(s_p e^{i\omega t_p})_p H-\mathbf1 d)\|^2.
\]

Within a frozen frequency prior, keep multiple plausible modes and their masses or a causal filter posterior for frequency, phase and possible jump. When rank or timing confidence fails, set the common missingness state and use a declared prior. Without a held-in true rate, \(\epsilon_\omega=\omega_{\rm true}-\widehat\omega\) is latent. A receiver can estimate, propagate and calibrate uncertainty and innovations. For known amplitude and phase design, the efficient information for \(\omega\) with unknown complex gain in proper white noise is \(2|H|^2\sum_p |s_p|^2(t_p-\bar t)^2/v\), with energy-weighted mean time. With unknown DC, project \(iHt_p s_p e^{i\omega t_p}\) onto the whitened orthogonal complement of both nuisance columns and take twice its squared norm. This follows from the Schur complement of the Gaussian Fisher matrix and gives a local unbiased CRLB, not a bound guaranteeing tracking of phase jumps or unwrapping. Irregular gaps and 1-second CTTC prediction require propagating the circular phase law across the actual lag. Never fit from Q3, `HOLDOUT`, full waveform repetitions, payload decisions or injection flags. Freeze real-split provenance and compare hashes after replacing future/probe IQ and all scorer-only columns with sentinels. Outputs, caches and feature hashes must agree. Enumerate the transitive support ledger as well as this mutation test. The CPU replay checks archived geometry and designed local support. It did not run this acquisition-level gate.

### 2.3 Same-row ablations and pooling

| Arm | Evaluation on the exact same rows, cutoffs and reference allocations |
| --- | --- |
| **Physical Bayes / B3 mixture** | Use causal continuous R1/GLS I/Q and the same permitted references. Marginalize gain, phase, lagged CFO, DC and jump/spur hypotheses. A finite model is \(p(s\mid D,R)\propto\pi(s)\sum_m w_m p(D,R\mid s,m)\). For Q1 sum assignments with the specified relative turn; for Q3 integrate the posterior predictive density over each sector. Oracle parameters may be scored in a quarantined model-specific ceiling. Under the true law the full-IQ Bayes risk is optimal over less-informed arms. Empirical QPSK B3 with train/dev nuisance fits is a baseline, and its misspecification prevents calling its realized performance a universal ceiling. |
| **Direct `s17` table** | Report raw frozen table for single-window/ Q2-none, and the matched-information likelihood reader using code emissions and the same causal reference statistics for Q1/Q2/Q3. For an exact reader integrate each candidate likelihood over the frozen quantizer cell (by locked simulation or quadrature and an error audit). The frozen table is a training posterior, not an emission likelihood on hardware. A code-center approximation can be reported as such. |
| **Frozen Qwen Zero** | Same host, W1/W2 codes, admissible public metadata, identical reference observations and reveal schedule. Set phase-coupling numeric slots to zero. Fit the zero control connector with the same steps and optimization budget. |
| **Naive-Additive** | Encode the same signed estimated phase and reference features through an additive real token slot or logit correction, with no enforced covariance of the correction. Hold parameter scale, reference code/features, losses, training rows and steps fixed. |
| **Stage4-Equivariant-Coupled** | Canonicalize the received gauge from the turn in the first valid cached code. Rotate both codes and receiver-coordinate pilot evidence by that gauge, propagate the signed phase law with explicit lag, then map the answer vector back with its registered group action. Under a common receive-chain quarter-turn, Q1 and Q2 TX labels stay fixed and Q3 receiver sectors permute. If testing a TX relabeling, known pilot labels and Q2 labels transform together in a separately specified action. A single canonical host score suffices. Register validity before projection and audit an exact-zero or missing-pivot policy. |
| **Stage4+RefSufficientPool** | Replace ordered raw-reference tokens by a set likelihood summary on the same pilot observations, with release times and known complex TX values. Deduplicate exact acquisition copies before pooling. With joint proper Gaussian \(y=A_mH+n,\ n\sim\mathcal{CN}(0,K_m)\), sufficient for \(H\) at fixed \(m\) are \(A_m^*K_m^{-1}y\) and \(A_m^*K_m^{-1}A_m\). For comparing candidate \(m\) also retain \(y^*K_m^{-1}y\), normalizers (e.g. \(\log\det K_m\)), and weights/times. Complete the square to see all remaining terms are constant in \(H\). For overlapping/colored blocks use joint covariance or unique innovations. For time-varying states use the corresponding transition and filter state or a bank of likelihood factors. A single summed phasor has no general sufficiency under unknown CFO, shared noise or jumps. |

No legitimate arm counts the same observation twice. If shared state already conditions on a pilot, score the next observation using its conditional innovation given that state. If causal metadata contains the same estimates, the model receives them as a representation of evidence already used. The Q2 none/one/two protocol above keeps reveal pilots out of the baseline fit. Identity invariance under duplicate copy can be checked via byte-equality before scoring within existing row budget. A full independent-reference mismatch or separate multi-seed host reruns need additional evaluation rows. CPU bootstraps and true-likelihood integration are independent of host training.

## 3. Frozen compute envelope and dispatch

### 3.1 Dataset inputs and handoffs

| Resource | Request and verification |
| --- | --- |
| Pluto | `GET https://data.mendeley.com/public-api/datasets/k72b4pnc5g?version=1`, then `GET https://data.mendeley.com/public-api/datasets/k72b4pnc5g/files?folder_id=root&version=1`. Use `content_details.download_url` with size and SHA-256 checked against the pinned downloaded manifest. `QPSKlink.mat` is MATLAB v5, **822,762,822 B**. Reader arrays and label vault go to distinct outputs. |
| CTTC | `GET https://zenodo.org/api/records/4895133/files`, with file bytes at `/api/records/4895133/files/{filename}/content`. Nine QPSK captures contribute **1,807,824,953 B**. Including README and sources, the selected archives total **1,966,425,693 B**. Verify bytes and official MD5 and pinned SHA-256 for each, then extract and verify only the authorized named files. `file_2.dat` at `176,015,264 B` × 9 plus configs and the 1,000,000-byte QPSK source totals **1,585,139,947 B** extracted for this plan. |
| Combined | **2,789,188,515 B** downloaded before model weights. The proposed 24 GiB envelope covers raw and selected extracted files, compact caches, model snapshots, and results. Pin the imported dataset bytes in a single ingestion job and use commit/reload between jobs. Keep volume use and compact result egress within verified aggregate Starter quotas. |

The exact download URLs, file sizes, hashes and archived manifest parity are in [results_m4.json](results_m4.json). The verifier can probe metadata with `--check-http` and still complete offline. Ingestion should verify before exposing an artifact to any downstream job, and allowlist regular files when extracting archives. Scientific sources and response fields come from the named branch downloader versions and the public archive pages. [Mendeley record](https://data.mendeley.com/datasets/k72b4pnc5g/1) · [Zenodo record](https://zenodo.org/records/4895133) · [Modal Volumes](https://modal.com/docs/guide/volumes)

Use one CPU ingestion function, with a content-addressed `raw/model` cache committed before scheduling dependents. Reuse a verified snapshot of the historical Qwen revision when possible, or fetch only that pinned revision. A second CPU function streams Pluto frames and CTTC `file_2.dat` memmaps into train/dev/test feature shards. It computes split-local frontend state, per-stage pilot inputs and CPU physical baselines. Use separate reader mounts for scorer-only material and GPU-visible caches. Store frozen host scores, checkpoint and shard manifests on a results Volume, and export compact paired rows for independent CPU replay. Stage default time caps cover image/container boot and CPU dispatch overhead via the explicit reserve lines. Use only per-second compute Functions in this envelope. Modal charges CPU and memory by the larger of request and actual consumption. Enforce `cpu=(c,c)` and `memory=(1024g,1024g)` for the paid caps in the table, with an economical low-CPU-memory model loader. [Modal CPU and memory](https://modal.com/docs/guide/resources)

### 3.2 Model and scored workload

- Use BF16 Qwen text-forward, fenced caches including Qwen3.5's attention and recurrent state, batched four candidate likelihoods, and a fixed tokenizer/chat template. Keep exact question and candidate sequence scoring and one host forward per canonicalized row. During connector training, gradients must flow through the frozen host to connector inputs. Share only text-only prelude states when valid. Never reuse post-evidence states across references or arms. Compare KV-cached and uncached logits/gradients on the longest dev template before accepting that scorer. Run initial train smoke with effective batch 32, microbatch 16 with two accumulated steps or microbatch 8 with four if necessary. Match prefix lengths and number of evidence positions to Stage 3D's microbenchmark and resource gate. No extra runtime factor is reserved for online multi-pass group averaging.
- Stage 3D measured `1.9757 s / optimizer step` and `23.88 ms / logical eval row` on one A100-80GB. Train and eval peaks were `52,972.5 MB` at microbatch 16 and `45,409.9 MB` at eval batch 64. The six 1,280-step runs took `2,884.0–3,217.8 s` including additional work. BF16 weights alone are about 8 GB at 4B parameters. Train activation and batched candidate workspace amplify this. The baseline plan uses A100-80GB for both. Accept the pilot run only if peak allocated/reserved fits a predeclared 65 GiB A100 threshold and sustained times meet `≤2.50 s/step`, `≤0.030 s/row` inclusive of in-batch data feeding. Rerun timing after any microbatch change. A100-40GB, L4 and A10/A10G require a separate shape and kernel check with smaller batches. The measured Stage 3D peaks exceed their VRAM. For a substitution, require equivalent logits, a ≤20 GiB peak on 24 GiB cards, and throughput below `0.030*(0.00074684/(GPU rate + 0.00005284))` seconds per row to match A100 costs, as well as a preapproved wall limit. This is about 81.5 ms on L4 and 62.4 ms on A10, with 2 cores and 12 GiB host RAM. Keep the confirmed A100 path if the smaller card would require quantizing the frozen host or invalidating kernels. [Stage 3D report](../../stage3d_qwen_signed_phase/REPORT.md) · [Modal GPU SKUs](https://modal.com/docs/guide/gpu)
- Train four matched connectors with one seed (`202611`), `384×32=12,288` logical examples per arm, mixing equal synthetic/real examples across the steps, using a frozen sampling manifest. Stage 3D locked the base training scorer and LR `0.0015`; use its temperature protocol on disjoint dev data and cap the extra dev scoring at 4,000 rows per arm. This is a single-seed resource-bounded replication tier. Freeze common architecture/parameter scales, prompt length cap and calibration before blind scoring.
- Synthetic blind extrapolation has 32 blocks × 64 mothers and holds cumulative phase beyond `π/2`, e.g. `θ=±{1.70,2.10,2.50,2.90}` rad, four independently seeded blocks per signed setting. Use lag families such as `(h,ℓ)=(2,2),(2,3)` counterbalanced within block, including both signs and unseen magnitudes. Use the same mother order and true signed synthetic condition metadata for all arms. Fit on the Stage 3D support or narrower, with separate seeds; known synthetic metadata is an intervention. Each synthetic mother contributes Q3 pre/post and Q2 `none/one/two`, for `40,960` logical rows across four arms. Include CPU first and second pilot likelihood ceilings and score-matched RefPool diagnostics. Estimate angles from allowed sync/pilot evidence for real data.
- Natural real work per arm is `3,680×(Q1 + 3×Q2) + 2,960×(2×Q3) = 20,640` rows, hence `82,560` across arms. Stress adds `(7×140 + 1,080)×4 = 8,240` rows. Dev adds `16,000`. A logical row already includes scoring all four candidate answers. Registered CPU physical arms score the identical rows but consume zero GPU time. Scope and statistical interpretation exclude prompt-level claims about rare jumps or undisclosed source domains that have no registered observations.

### 3.3 Budget arithmetic and zero-bill gate

Public Modal list prices checked for this round are A100-80GB `$2.4984/h`, A100-40GB `$2.0988/h`, L4 `$0.7992/h` and A10 `$1.1016/h`, plus CPU `$0.04716/physical-core-h` and memory `$0.007992/GiB-h`. The A100 Function cost with 2 physical cores and 12 GiB is `$0.00074684/s` or `$2.688624/h`. A10 is the pricing label and `A10G` appears in Modal examples. The main schedule budgets the A100 and assumes no savings from smaller devices. [Modal pricing](https://modal.com/pricing)

| Job | Cap (s) | Compute cap (USD) |
| --- | ---: | ---: |
| A100 connector fit, `4×384×2.50` | 3,840 | 2.86786560 |
| A100 raw real, `ceil(82,560×0.030)` | 2,477 | 1.84992268 |
| A100 stress, `ceil(8,240×0.030)` | 248 | 0.18521632 |
| A100 synthetic, `ceil(40,960×0.030)` | 1,229 | 0.91786636 |
| A100 dev, `16,000×0.030` | 480 | 0.35848320 |
| A100 smoke, starts, setup and commit | 1,100 | 0.82152400 |
| A100 controlled retry reserve | 900 | 0.67215600 |
| CPU ingest, 1 core and 8 GiB | 1,800 | 0.05554800 |
| CPU causal frontend and physical readers, 4 cores and 16 GiB | 7,200 | 0.63302400 |
| CPU gate checks, image preparation allowance and aggregation, 2 cores and 8 GiB | 1,800 | 0.07912800 |
| **Total**, 2.8538889 A100 hours and 3 CPU function hours | | **8.44073416** |

The aggregate CPU charge outside GPU jobs is `$0.76770000`. Headroom is `$1.55926584` to the `$10.00` target and `$4.43926584` to the historical `$12.88` remaining credit. Reserve includes startup, stopped shards and retries. Compile the exact manifest and benchmark shapes before dispatching the registered blind rows. Run one billed function or shard at a time with no automatic retry/multi-container fanout, cap sum of billed durations including failed attempts, and stop when completion would exceed the cap. Use bounded shards and commit checkpoints and hashes before release to the next stage. Track first CPU image build and start costs against the CPU allowance and refresh dashboard arithmetic before each batch. The schedule is an arithmetic bound under the stated resource/time caps. Completion requires throughput and causal-alignment gates on dev, and the historical balance must be verified in the live workspace before dispatch.

Require account-visible `$0.00` out-of-pocket gating and sufficient remaining compute and volume/egress quotas before launch. On Modal Starter a workspace spend limit caps net charges separately from the workspace gross usage limit. Verify that the owner has set a net `$0.00` limit or an equivalent hard guard for the live workspace, and a gross cap covering the reserved stage amount within available credits. Use dashboard reading and control-plane caps as well as a local ledger because other jobs may consume credits and usage can lag. Hosted model APIs are a distinct charge; this run uses a frozen self-hosted model on Functions. No account state was verified or changed and no Modal workload was started in Round 1. [Modal budgets](https://modal.com/docs/guide/budgets) · [Modal billing](https://modal.com/docs/guide/billing)

## 4. CPU replay and interpretation

```powershell
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M4/verify_m4.py
```

The script loads the exact frozen weights and the archived branch source via read-only `git show`. It uses 8,192 examples per cell in the 168-cell mapping factorial and 32,768 examples per nuisance-curve point. This is a synthetic, matched-filter surrogate with a specified steering vector and colored proper Gaussian covariance. It verifies KL against quadrature or sampled log-density ratios, estimates scalar Bayes MI and empirical code capacity, and scores actual encoder codes and the frozen lookup table. At one full complex Gaussian point, (1) gives `1.196255` nats versus `1.198460 ± 0.004144` Monte Carlo (one standard error). A bounded random gain mixture with equiprobable denominators `0.7,1.3` has an estimated joint KL of `0.185511` nats after marginalization. Its scalar MI is `0.873286` nats with hidden gain and `0.889764` with observed gain. The centered scalar Student noise experiment has about `0.587033` nats of scalar MI for \(\nu=1\) with infinite forward Gaussian KL. All sweep inputs are generated surrogates. The optional metadata requests were refused by the local network. Historical manifest comparisons passed.

| Surrogate condition | R1 SER | R2 SER | R3 physical SER | R3 calibrated SER | R3 physical vs calibrated NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| −2 dB, clean gain/DC/sideband | 39.06% | 37.56% | 37.56% | 37.56% | 1.129 vs 1.008 |
| +2 dB, clean gain/DC/sideband | 20.74% | 19.98% | 19.98% | 19.98% | 0.532 vs 0.532 |
| +20 dB, clean gain/DC/sideband | 0.00% | 0.00% | 0.00% | 19.89% | 0.060 vs 0.543 |
| +20 dB, +30% AGC, DC 0.30 and spur 0.26 | 0.00% | 0.00% | 0.00% | 32.54% | 0.239 vs 0.879 |

The verifier checks directly that the frozen table chooses the turn's class for all four local codes, which explains SER equality of the scalar R2 and R3 arms. The matched sum and hence the turn coincide under the lift. The local code and calibration can still change. Attenuation at low SNR can improve frozen NLL without improving SER. The high-SNR loss from variance locking occurs when injected noise masks real high-SNR evidence. The observed saturation codes and conditional nuisance curves characterize the frozen weights under the stated draws. Generalization, real timing coverage and throughput remain empirical gates. See the full records for sample counts, scores, transition matrices, budget, manifests and geometry. [Verifier](verify_m4.py) · [Results](results_m4.json)
542:def verify_budget(splits, lock):
605:    budget = verify_budget(splits, lock)
612:              'scipy': scipy.__version__, 'scope': 'synthetic CPU replay, historical metadata and prospective budget',
617:              'index_audit': splits, 'dataset_manifests': manifests, 'modal_budget': budget}
````

### Event #13: `item.completed` — `command_execution` (`item_10`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-Command uv; uv --version; Get-Content -Encoding utf8 research/stage4_math_and_design/M6/verify_m6.py | Select-Object -Skip 200'
```

**Aggregated Output:**

```text

CommandType     Name                                               Version    Source                                   
-----------     ----                                               -------    ------                                   
Application     uv.exe                                             0.0.0.0    C:\qazwsx\Scripts\uv.exe                 
uv 0.11.21 (5aa65dd7a 2026-06-11 x86_64-pc-windows-msvc)
    assert result["elapsed_seconds"] < 30
    return result


def sequential_design():
    alpha, info, prior_var, futility_pp = 0.05, 0.18, 4.0, 0.10
    fixed_boundary = norm.isf(alpha)
    rho = math.sqrt(info)
    increment_sd = math.sqrt(1 - info)
    alpha1 = norm.sf(fixed_boundary / rho)
    early = norm.isf(alpha1)

    def final_reject(boundary, drift, lo):
        # Integrate over Z1; Z2 | Z1=x and drift is Gaussian.
        integrand = lambda x: norm.pdf(x - rho * drift) * norm.sf(
            (boundary - rho * x - (1 - info) * drift) / increment_sd
        )
        return quad(integrand, lo, early, epsabs=1e-12)[0]

    # Lower integration limit -infinity gives the nonbinding efficacy design.
    final = brentq(lambda b: alpha1 + final_reject(b, 0.0, -np.inf) - alpha, 1.5, 2.0, xtol=1e-14)
    post_var = 1 / (1 / prior_var + info)
    z2_mean_per_z1 = rho * (1 + (1 - info) * post_var)
    predictive_sd = math.sqrt(1 - info + (1 - info)**2 * post_var)
    futility_z = (final - norm.isf(futility_pp) * predictive_sd) / z2_mean_per_z1
    assert futility_z < early
    binding_boundary_size = alpha1 + final_reject(final, 0.0, futility_z)
    nonbinding_boundary_size = alpha1 + final_reject(final, 0.0, -np.inf)
    assert abs(nonbinding_boundary_size - alpha) < TOL and binding_boundary_size <= alpha

    # Equal pairing, 10,000 candidates total. Boundary null verifies size separately from the mixture.
    strata = (
        ("structured_broken", 6000, -0.80, 0.10),
        ("boundary_null", 1500, 0.00, 0.90),
        ("working_coupled", 2500, 3.60, 0.99),
    )
    seed, cost = 20261126, 17.12
    pilot_fraction = 0.75 * info / 3 + 0.04
    rng = np.random.default_rng(seed)
    details = {}
    exact_cost = 0.0
    observed_cost = 0.0
    for name, n, drift, cpu_pass_rate in strata:
        z1 = rng.normal(rho * drift, 1, n)
        z2 = rho * z1 + increment_sd * rng.normal(increment_sd * drift, 1, n)
        passed = rng.random(n) < cpu_pass_rate
        early_go = passed & (z1 >= early)
        continued = passed & (z1 >= futility_z) & (z1 < early)
        efficacy = early_go | (continued & (z2 >= final))
        # Early efficacy still buys rollout to three seeds and blind scoring for release.
        funded_full = continued | early_go
        fixed = z2 >= fixed_boundary
        dollars = cost * (pilot_fraction * passed + (1 - pilot_fraction) * funded_full)
        exact_early = norm.sf(early - rho * drift)
        exact_seq = cpu_pass_rate * (exact_early + final_reject(final, drift, futility_z))
        exact_mean = cost * cpu_pass_rate * (pilot_fraction + (1 - pilot_fraction) * norm.sf(futility_z - rho * drift))
        exact_cost += n * exact_mean
        observed_cost += float(dollars.sum())
        ci = binomtest(int(efficacy.sum()), n).proportion_ci(method="exact")
        details[name] = {
            "trials": n, "drift": drift, "cpu_pass_probability_assumed": cpu_pass_rate,
            "cpu_pass_count": int(passed.sum()), "early_efficacy_count": int(early_go.sum()),
            "futility_count": int((passed & ~funded_full).sum()), "full_funded_count": int(funded_full.sum()),
            "seq_efficacy_count": int(efficacy.sum()), "seq_rate_mc": float(efficacy.mean()),
            "seq_rate_exact_binomial_ci95_mc": [ci.low, ci.high], "seq_rate_quadrature": exact_seq,
            "fixed_efficacy_count": int(fixed.sum()), "fixed_rate_mc": float(fixed.mean()),
            "fixed_rate_exact_normal": float(norm.sf(fixed_boundary - drift)),
            "seq_mean_usd_mc": float(dollars.mean()), "seq_mean_usd_exact": exact_mean,
        }
        assert ci.low <= exact_seq <= ci.high

    working = details["working_coupled"]
    assert working["seq_rate_quadrature"] >= 0.90 and working["seq_rate_exact_binomial_ci95_mc"][0] >= 0.90
    assert details["boundary_null"]["seq_rate_exact_binomial_ci95_mc"][1] <= alpha
    n_total = sum(row["trials"] for row in details.values())
    assert n_total == 10_000
    nulls = [details[x] for x in ("structured_broken", "boundary_null")]
    null_n = sum(row["trials"] for row in nulls)
    drift_min = brentq(lambda d: 0.99 * (norm.sf(early - rho * d) + final_reject(final, d, futility_z)) - 0.90, 0, 8)
    no_screen_cost = sum(
        n * cost * (pilot_fraction + (1 - pilot_fraction) * norm.sf(futility_z - rho * drift))
        for _, n, drift, _ in strata
    ) / n_total
    return {
        "rng_seed": seed, "trials": n_total, "alpha": alpha, "information_fraction": info,
        "prior_variance_standardized": prior_var, "predictive_futility_threshold": futility_pp,
        "alpha_first_look": float(alpha1), "early_z": float(early), "final_z": final, "futility_z": float(futility_z),
        "least_favorable_type_I_no_cpu_nonbinding": float(nonbinding_boundary_size),
        "least_favorable_type_I_no_cpu_binding": float(binding_boundary_size),
        "min_working_drift_for_90_percent_power_at_cpu_sensitivity_099": drift_min,
        "pilot_cost_fraction": pilot_fraction, "pilot_usd_assumed": pilot_fraction * cost,
        "fixed_usd": cost, "sequential_mean_usd_mc": observed_cost / n_total,
        "sequential_mean_usd_exact": exact_cost / n_total,
        "cost_reduction_pct_mc": 100 * (1 - observed_cost / (n_total * cost)),
        "cost_reduction_pct_exact": 100 * (1 - exact_cost / (n_total * cost)),
        "sensitivity": {
            "working_power_at_cpu_sensitivity_095": working["seq_rate_quadrature"] * 0.95 / 0.99,
            "mean_sequential_usd_no_cpu_screen_in_same_mixture": float(no_screen_cost),
            "cost_reduction_pct_no_cpu_screen_in_same_mixture": float(100 * (1 - no_screen_cost / cost)),
            "cost_reduction_pct_if_all_working": float(100 * (1 - working["seq_mean_usd_exact"] / cost)),
        },
        "null_mixture_false_efficacy_seq_mc": sum(row["seq_efficacy_count"] for row in nulls) / null_n,
        "null_mixture_false_efficacy_fixed_mc": sum(row["fixed_efficacy_count"] for row in nulls) / null_n,
        "null_mixture_false_efficacy_seq_exact": sum(row["trials"] * row["seq_rate_quadrature"] for row in nulls) / null_n,
        "null_mixture_false_efficacy_fixed_exact": sum(row["trials"] * row["fixed_rate_exact_normal"] for row in nulls) / null_n,
        "strata": details,
    }


def budget():
    d = Decimal
    rates = {"A100-80GB": d("0.000694"), "L4": d("0.000222"), "A10G": d("0.000306"), "CPU": d("0")}
    ancillary = 2 * d("0.0000131") + 16 * d("0.00000222")
    rows = (
        ("CPU dataset/weight staging, causal frontends and split audits", "CPU", 7200),
        ("Synthetic Qwen parity and timing receipt", "A100-80GB", 450),
        ("Synthetic pilot, seed 1, at most 173 steps", "A100-80GB", 650),
        ("Synthetic continuation, at most 2707 steps across 3 seeds", "A100-80GB", 6750),
        ("Synthetic blind G1-G4 and Q2_C4 KV scoring", "A100-80GB", 1050),
        ("Pluto, 2240 frames / 16 blocks, forward only", "L4", 2300),
        ("CTTC, 1440 packets / 72 blocks / 9 sessions, forward only", "L4", 1700),
    )
    breakdown = []
    for name, device, seconds in rows:
        rate = rates[device] + ancillary
        breakdown.append({"task": name, "device": device, "max_billed_seconds": seconds,
                          "max_usd_per_second": str(rate), "max_usd": str(rate * seconds)})
    base = sum(d(row["max_usd"]) for row in breakdown)
    fallback = (rates["A10G"] - rates["L4"]) * 4000
    total, credit = d("9.50"), d("30.00") - d("17.12")
    reserve = total - base - fallback
    assert base == d("8.30517200") and fallback == d("0.336000") and reserve >= 0
    assert (base + fallback + reserve) == total <= d("10.00") < credit
    assert credit - total >= d("2.88")
    return {"pricing_url": "https://modal.com/pricing", "resource_ceiling_physical_cores": 2,
            "resource_ceiling_gib": 16, "rows": breakdown, "base_usd": str(base),
            "a10g_substitution_increment_usd": str(fallback), "additional_reserve_usd": str(reserve),
            "cap_usd": str(total), "historical_credit_remainder_usd": str(credit),
            "buffer_usd_at_cap": str(credit - total), "spend_this_round_usd": "0.00"}


def main():
    assert (np.__version__, scipy.__version__) == ("2.3.5", "1.17.0")
    assert platform.python_version_tuple()[:2] == ("3", "12")
    summary = json.loads((DATA / "summary_master.json").read_text(encoding="utf-8"))
    parity_diffs = []
    with np.load(DATA / "blind_q3_scores.npz", allow_pickle=False) as q3, np.load(DATA / "blind_q2_scores.npz", allow_pickle=False) as q2:
        assert q3["seeds"].tolist() == summary["seeds"]
        np.testing.assert_array_equal(q3["temps_q3"], summary["locked_temps_q3"])
        np.testing.assert_array_equal(q2["temps_q2"], summary["locked_temps_q2"])
        report = {
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
            "stage3d": {"q3": replay_q3(q3, summary, parity_diffs), "q2": replay_q2(q2, summary, parity_diffs)},
            "summary_parity": {"max_abs_acc_nll_diff": max(parity_diffs), "scalar_acc_nll_checks": len(parity_diffs),
                               "h_c4_035_scope": "Work F identity rotation and both signs. The all-rotation ladder is a fresh replay."},
            "cpu_geometry_toy": cpu_geometry(), "sequential": sequential_design(), "modal_budget": budget(),
        }
    path = HERE / "results_m6.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"VERIFIED {path.name}: parity max |diff|={max(parity_diffs):.3g}, 10,000 trials, cap ${report['modal_budget']['cap_usd']}")


if __name__ == "__main__":
    main()


```

### Event #14: `item.completed` — `web_search` (`exec-fa516806-25d4-43b7-b8b3-6a4c3e21b4e9`)

```json
{
  "id": "exec-fa516806-25d4-43b7-b8b3-6a4c3e21b4e9",
  "type": "web_search",
  "query": "'$0.000694'",
  "action": {
    "type": "find_in_page",
    "pattern": "$0.000694"
  },
  "results": [
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn7view0",
      "snippet": "Total lines: 391",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/AdDkCjV4XyG9bCDjtvIU6jcC-tjf5Qs9HeCATC2oBCkN5dVJANfPT4jxLB9n3x1hWk8oR-182bnEpIopvrS2nA",
      "title": "Pricing: Pay per second for GPUs and CPUs | Modal",
      "url": "https://modal.com/pricing"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn7view1",
      "snippet": "Total lines: 84",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/lrXNJY5-hPSsZIxK-m8c506LjMYEWkBkMVds6V2FZ0wfHv5gnGiUgL-EAA3_Y0_HkuKVX6TYxnVrX80KPKYSi1DhlVv6_oAoxkqHNlmaVGsCM0eMOqrnYm1cwk3WeIIBlXvUXj-nlXouM_0w0C79NKpdl0IC_L-cGYvTUSRRziY",
      "title": "Configuring CPU, memory, and disk | Modal Docs",
      "url": "https://modal.com/docs/guide/resources"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn7view2",
      "snippet": "Total lines: 51",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/yyhJGj6pfMrHvgra2-NfTKEiqti19JRTpCuTbOLTzNMK1ubomcpq791OkFAeWGVmqO-4maUa70KFumhS1qvuanFgSbJRtL1m_81mhUuh_mpiQLyd69DG_4xEpkiJ5BaU",
      "title": "Budgets | Modal Docs",
      "url": "https://modal.com/docs/guide/budgets"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn7view3",
      "snippet": "Total lines: 391",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/5M2Udu5Zy_Xbx-YpfxnMPzOC6_XMyyn_yN3_gW4eNQRWTSQz0TFHxkXCMtLbNPAWkScr9nqpwMPV-uT1iYpz5w",
      "title": "Pricing: Pay per second for GPUs and CPUs | Modal",
      "url": "https://modal.com/pricing"
    }
  ]
}
```

### Event #15: `item.completed` — `command_execution` (`item_13`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$p=(Join-Path $env:USERPROFILE '"'.codex/memories/rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md'); Get-Content -Encoding UTF8 "'$p | Select-Object -Skip 60 -First 7; rg -n '"'release|NIG|random.label|Crossed|22.|30.' research/stage4_math_and_design/M2/THEOREM_M2.md; rg -n 'CFO|Arm 0|Arm 1|Arm 2|0b|TX|coupled' research/stage4_math_and_design/discussions/M1/prompt_r2.md"
```

**Aggregated Output:**

```text
## Task 4: Assess whether P4 advanced the electromagnetic-native Transformer goal

Outcome: success

Conclusion: P4 made substantial algorithmic progress by implementing and testing a complex-valued, gauge-equivariant, Hermitian-attention Transformer prototype with physical channel simulation and coded-link evaluation. The strongest evidence is that signal symmetries and complex-valued operations were encoded directly into the model and some mechanisms showed reproducible value in controlled 1D ablations. However, the formal 2D result did not establish a learned-Transformer advantage; the largest 2D improvements came from better pilot/channel front-ends. The next decisive experiment is a strictly matched comparison where EM and classical receivers share the same physical front-end, pilot information, decoder, and inference budget, isolating only the learned module.
103:1. **Absolute and comparative performance.** Use paired lower bounds. Require NLL to beat uniform, the dev-frozen prior, and the matched zero-phase system, with a \(0.02\)-nat increment proposed against matched zero. Require accuracy above analytic chance, a dev-frozen prior classifier, and the matched zero-phase system, with \(+0.02\) absolute over chance and zero proposed as initial margins. With imbalanced classes, also require outperformance of the fixed majority prediction or used prior competitor, and report balanced accuracy as a separately locked estimand. Require valid chance/random-label randomization tails for the accuracy and NLL statistics where the design supplies exchangeability. Choose an oracle with the same information budget as the candidate for the main NIG denominator and disclose a finer physical oracle separately. Propose \(\eta_{\min}=0.10\) and validate \(H(Y)-L(o)>0\) on an independent development design or by a co-primary lower bound. Test the linear contrast \(D_{\eta}=(1-\eta)H(Y)-L(q)+\eta L(o)>0\) rather than dividing two noisy quantities. Use \(H(Y)=\log K\) for a genuinely uniform target law. Otherwise predefine the target distribution or carry uncertainty in \(H(Y)\) by a joint confidence region and minimize the contrast over it. For real data without a defensible Bayes oracle, freeze a physically informed predictive reference, call the resulting metric reference-normalized, and withhold the Bayes-normalized claim. Requiring a learner to outperform a true Bayes oracle under proper loss is infeasible. A positive NIG floor defines a fraction of the attainable improvement. Controlled slot/text interventions form a further prerequisite for a specifically signed-slot mechanism claim. Require improvement with matched versus permuted or flipped slots at fixed text and same \(X\), with seeded training and ablations paired. Use the native receiver outputs for the gate.
105:3. **Hierarchy and multiplicity.** Keep all repeats, rotations, acquisitions, and matched arms inside their mother and higher cluster. Report a frozen-seed conditional estimate and, if claiming generalization to retraining, a separate random-seed inference. Use equal independent cluster weights for the following exact balanced analysis. Pluto supplies 16 units if they are independent captures. For a CTTC new-session claim use nine independent session means, with all eight clusters per session contributing to that session. A conditional claim on the fixed sessions needs the corresponding estimand and defensible within-session cluster independence. For a permutation or label-randomization gate, lock the allowed exchangeability orbit and recompute the same statistic throughout. Repeated/rotated labels move jointly with their mother and never split correlated session units as though independent. A synthetic randomized design with independent label assignments can instead rerandomize the sanctioned generator operation inside the original design. When there is no justified exchangeability, collect a randomized negative control. State whether it tests a sharp random-label null or a mean-effect margin. A permutation p-value \((1+\#\{T_{\rm perm}\ge T_{\rm obs}\})/(R+1)\) with uniform randomizations and ties included is conservative under the sharp design. Use exact enumeration when feasible.
159:| G2 | 8 | 0.824877 | 0.292945 | 0.176866 | 1.022230 | 1.019346 |
171:| G2 | −0.02775961 | −0.00004351 | +0.00341462 | −0.00008845 | −0.03104227 | −0.02771611 |
173:| G4 | −0.02491655 | +0.00143469 | +0.00443245 | +0.00002490 | −0.03080859 | −0.02635124 |
174:| H | +0.01808982 | +0.02178504 | +0.00023216 | −0.00000148 | −0.00392590 | −0.00369522 |
176:**G1.** For phase, the five-term path is \(0.92563349+0.05975673+0.00036415+0.00000192+0.45947227=1.44522855\). The zero path is \(0.92563349+0.09630010+0.00048790+0.00000197+0.45845510=1.48087856\). Scalar mismatch accounts for 102.51% of the locked lead along this path. Before calibration or projection, marginal KLs are \(0.00008076\) and \(0.00035846\), respectively. Raw phase and zero empirical-code NIGs are −0.12794 and −0.20533. The S17 direct reference scores 64.62% versus 23.80% and 19.41%. Phase's lower ECE is consistent with the confidence mismatch. It does not measure the share of the NLL lead. Using the argmax set, the empirical resolutions are 0.01162 and 0.02104 nats for phase and zero. Using the exact score vector, both have 0.46066 nats of in-sample resolution because both distinguish the signed codes. Those scores can encode information that the decoded predictions use poorly. With sign probabilities marginalized conditional on code, the NLL cost is 0.007704 for phase and 0.002672 for zero. The difference, 0.005032 nats, is sensitivity to that specific sign-discarding operation. It is distinct from Bayes MI and from the factorial intervention.
178:**G2 and G4.** Raw zero accuracy leads by 2.485 and 1.671 pp. Finite \(T^*\) preserves those rankings and a fitted loss lead of 0.02772 and 0.02635 nats remains. The reverse-conditional term gives zero advantages of 0.03104 and 0.03081 nats after projection and fitting. The global prior and scalar terms are small. Raw zero's coarsened argmax resolution also exceeds phase's by 0.01268 and 0.01877 nats. Projection actually reverses the top-1 ordering in those cells while leaving zero with lower probabilistic loss. Thus the locked reversal is explained at the score level by conditional probability allocation and raw decision ordering. The five saved conditions do not establish an architectural transition at \(\pi/2\). Check the angle effect by holding \(|\omega|\), horizons, and diffusion fixed or by crossing them prospectively with common noise and more independent seeds.
182:Each cell gives locked NLL / ECE and retrospective seedwise \(T^*\) NLL / ECE. Temperature triples follow seeds 202611, 202622, 202633.
186:| G1 | phase | 1.445229 / .118604 | 1.385472 / .025608 | ∞, 10.144, 5.720 |
187:| G1 | zero | 1.480879 / .182737 | 1.384578 / .051229 | ∞, 6.453, 6.087 |
188:| G2 | phase | 1.335826 / .022970 | 1.334544 / .011354 | .548, 1.008, .769 |
189:| G2 | zero | 1.308067 / .023020 | 1.306828 / .017347 | 1.220, .916, .711 |
191:| G3 | zero | 1.478130 / .179196 | 1.384737 / .052938 | ∞, 6.842, 5.996 |
192:| G4 | phase | 1.334667 / .027630 | 1.334066 / .018397 | .547, .945, .743 |
193:| G4 | zero | 1.309751 / .032534 | 1.307715 / .021834 | 1.535, .831, .762 |
203:| G1 | phase | 23.80 / 1.445229 | 17.05 / 1.411312 | 16.36 / 1.411586 |
204:| G1 | zero | 19.41 / 1.480879 | 16.43 / 1.439122 | 16.60 / 1.445854 |
205:| G2 | phase | 33.88 / 1.335826 | 40.95 / 1.306349 | 40.43 / 1.299371 |
206:| G2 | zero | 36.36 / 1.308067 | 38.31 / 1.276082 | 38.51 / 1.273466 |
207:| G3 | phase | 22.68 / 1.441047 | 18.39 / 1.410222 | 18.68 / 1.410409 |
208:| G3 | zero | 19.74 / 1.478130 | 18.20 / 1.441579 | 17.64 / 1.447584 |
209:| G4 | phase | 33.29 / 1.334667 | 40.79 / 1.311247 | 40.53 / 1.304942 |
210:| G4 | zero | 34.96 / 1.309751 | 37.76 / 1.278018 | 37.40 / 1.274705 |
211:| H | phase | 25.76 / 1.416987 | 22.02 / 1.384622 | 22.74 / 1.382330 |
214:On G1 phase the arithmetic pool makes 8.76% of decisions better and 15.51% worse, the exact weighted hit changes here. Geometric makes 8.28% better and 15.72% worse. G1 phase F0 scores 20.88% / 1.411163. On complete orbits G1 phase's arithmetic gain is 0.03386703, while the reverse-barycenter KL is 0.03430077. For zero the corresponding numbers are 0.04120584 and 0.03488442. Restricting to the three saved rotations adds 0.00004913 and 0.00055046 to the loss gain. Across the tested cells pooled maximum equivariance and idempotence residuals were at floating-point precision. G1 raw maximum probability discrepancies were 0.48167 and 0.41957 for phase and zero. Structural compliance of the offline tables coexists with poor decoding and poor ECE.
222:On all the saved G1 rotations of the phase checkpoint, a text-only flip raises NLL by +0.00323146 and a slot-only flip by +0.01723982, holding the other channel matched. A joint flip raises it by +0.01857416. The difference in differences is −0.00189712. On rotation zero these are +0.00302035, +0.01679462, +0.01836983, and −0.00144514. The interventions show sensitivity to physically aligned slots in the phase checkpoint, alongside a text response, in the tested models. They compare flipped slots to matched slots within a checkpoint. Training with the phase arm and training with the zero arm is a distinct intervention. A shared-development retraining comparison with a locked recalibration and independent seeds is needed to attribute persistent NLL gains to richer slot use.
230:| none | 1.386294 | 1.392846 / 1.393822 | 1.386272 / 1.386294 | +.000953 | +.000976 |
233:| duplicate raw | .819207 | 1.394420 / 1.413341 | 1.384193 / 1.385808 | +.017307 | +.018922 |
234:| independent only | 1.386294 | 1.407714 / 1.420522 | 1.386294 / 1.386071 | +.013031 | +.012807 |
238:For valid, the fitted residual phase lead \(+0.00159367\) is \(+0.00028827\) reference-prior KL, \(+0.00158510\) reverse KL, and \(−0.00027971\) realized-reference label residual. For fresh resample, the remaining lead reverses to \(−0.00267249\). The Work A reference gains 0.156018 from fresh vs valid, while the phase connector is 0.016757 worse at locked \(T\). Q2 realized NIG is undefined for the uniform Work A controls because the denominator is zero. All Q2 controls and arms have complete numerical paths, ECE, Brier, scalar temperatures, and soft reference diagnostics in the JSON.
258:| Synthetic G1 proxy | 131 blocks | 7 / 5 | .224 / .300 | 15 / 7 | .981 / .998 |
264:At the actual G1 excesses over \(0.02\), the three-fixed-seed G1 or Pluto-proxy minima are 10 and 17 blocks/captures, and the CTTC proxy at ICC .10 gives 4 and 6 sessions. For three random seeds at the planning excesses, the ceiling even with known components and infinitely many blocks is 0.71537 and 0.85941. The conservative finite-level test tends to 0.23839 and 0.30978. These ceilings are consequences of the fitted seed variance and fixed effect-size assumptions. The high potential sampling uncertainty in that variance makes the ten-seed numbers planning proxies. Even with ten seeds, the modeled power statements apply only to the two relative endpoints.
268:| Endpoint | Fixed-seed t | Crossed test | Block-only t with random seeds |
273:For random-seed G1 power the corresponding analytic/Monte Carlo pairs are .22385/.22447 and .30028/.30242. The correlated Holm-IUT MC false-claim family rates range from 0.04002 to 0.04953 under the three simulated null configurations. The proof establishes protection beyond those simulations. ECE and the remaining performance gates need their own pilot variability or a joint design simulation.
280:- **M3** should score the native equivariant forecast against the raw, arithmetic, geometric and canonical-representative paths on a common development split. Calibrate on development data that span the prospective regimes and freeze the choice before blind scoring. Lock all absolute endpoints before comparing phase and zero. The G1 raw and pooled paths fail absolute accuracy, and the blind-fitted paths have below-chance accuracy and negligible NIG.
6:1. **M2 (Stage 3D Forensics)**: On `blind_q3_scores.npz`, `G1` locked `NLL(zero) - NLL(phase) = +0.035650` splits into `+0.036543` scalar temperature mismatch and **`-0.001017` reverse-conditional KL**; after seedwise scalar recalibration, **`zero` beats `phase` across all 5 cells** (`G1: -0.000893`, `G2: -0.027716`, `G3: -0.000430`, `G4: -0.026351`, `H: -0.003695`), despite `0.3631 nats` signed CFO MI available in `s17`! M2's Stage 4 gate requires post-dev-calibration NLL/Acc/NIG ($\eta_{\min}=0.10$), ECE $\le 0.05$, equivariance defect $\le 10^{-5}$, and fixed-text slot interventions.
8:3. **M1 Self-Critique + M2/M6 Challenge**: In your Round 1 `B_EquivariantPhaseCoupled`, using the physical S17 prior + residual `delta = 0.15 * tanh(ell_host)` reached `64.62%` (parity with `direct_s17`), leaving unresolved whether frozen host attention learned anything beyond the physical prior! Also naive `A` reached `62.40%` on a tiny toy MLP, so we need a clean causal isolation between **quotient equivariance**, **multiplicative phase coupling**, **coupled tokens into frozen host (`NO direct logit bypass`)**, and **physical prior + bounded host residual**.
12:   - M4 requires: (a) clean separation between **receive-gauge rotation** ($r_{\text{rx}}$: `Q1` TX diff invariant, `Q2` TX label invariant, `Q3` RX sector rolls $+r_{\text{rx}}$) and **TX relabeling**, (b) CFO uncertainty variance $v_\omega$ (giving edge damping $e^{-n^2 v_\omega d^2 / 2}$ correlated across edges!), and (c) SNR-preserving R3 lift on real Pluto/CTTC frames.
18:     - **`Arm 0` (Stage 3D Baseline)**: non-quotient separate `<W1><W2><PHASE>`, no bypass.
19:     - **`Arm 1` (`Equivariant-Host-Pure`)**: quotient canonicalization (`canonical_input` + `rotate`) + multiplicative phase-coupled continuous summary fed purely as **soft tokens into the frozen host**, with **NO physical logit bypass** (`q_host` computed strictly from frozen LM / frozen surrogate logits, with isotropic `W_out` geometry)!
20:     - **`Arm 2` (`Equivariant-Physical-Residual`)**: canonical physical prior + bounded trainable host residual (`delta = gamma * tanh(ell_host)`), plus fixed-text counterfactual slot-swap and shuffled-slot diagnostics isolating host contribution beyond the analytic bypass!
21:   - Implement and run **BOTH `Arm 1 (Host-Pure)` AND `Arm 2 (Physical+Residual)`** in `verify_m1.py` on the SAME frozen CPU surrogate host and identical `|\omega| <= 0.25` training split, testing `G1..G4`, equivariance defect (`<= 1e-6`), and M6's rank/margin candidate-Jacobian diagnostic! (Does `Arm 1 Host-Pure` ALSO extrapolate >55% past $\pi/2$ on `G1/G3`, or does frozen random projection struggle without coupled tokens / sufficient width?)
25:   - Write down the exact permutation table distinguishing: (i) **synthetic/canonical per-window shift** `(r1, r2)`, (ii) **real hardware common receive-gauge turn** `r_rx` (affecting Q1/Q2/Q3_rx), and (iii) **joint TX+RX transform**, plus exact incorporation of M4's shared CFO latent `(omega_hat, v_omega)` (via Gaussian-Hermite over $\omega$ or analytical coupled variance $n^2(v_\phi d + v_\omega d^2)$ with shared edge correlation) and M3's `joint[4,4]` factor!
```

### Event #16: `item.completed` — `command_execution` (`item_12`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$a=Get-Content -Encoding UTF8 research/stage4_math_and_design/M4/THEOREM_M4.md; $a | Select-Object -Skip 175 -First 108'
```

**Aggregated Output:**

````text
- Train four matched connectors with one seed (`202611`), `384×32=12,288` logical examples per arm, mixing equal synthetic/real examples across the steps, using a frozen sampling manifest. Stage 3D locked the base training scorer and LR `0.0015`; use its temperature protocol on disjoint dev data and cap the extra dev scoring at 4,000 rows per arm. This is a single-seed resource-bounded replication tier. Freeze common architecture/parameter scales, prompt length cap and calibration before blind scoring.
- Synthetic blind extrapolation has 32 blocks × 64 mothers and holds cumulative phase beyond `π/2`, e.g. `θ=±{1.70,2.10,2.50,2.90}` rad, four independently seeded blocks per signed setting. Use lag families such as `(h,ℓ)=(2,2),(2,3)` counterbalanced within block, including both signs and unseen magnitudes. Use the same mother order and true signed synthetic condition metadata for all arms. Fit on the Stage 3D support or narrower, with separate seeds; known synthetic metadata is an intervention. Each synthetic mother contributes Q3 pre/post and Q2 `none/one/two`, for `40,960` logical rows across four arms. Include CPU first and second pilot likelihood ceilings and score-matched RefPool diagnostics. Estimate angles from allowed sync/pilot evidence for real data.
- Natural real work per arm is `3,680×(Q1 + 3×Q2) + 2,960×(2×Q3) = 20,640` rows, hence `82,560` across arms. Stress adds `(7×140 + 1,080)×4 = 8,240` rows. Dev adds `16,000`. A logical row already includes scoring all four candidate answers. Registered CPU physical arms score the identical rows but consume zero GPU time. Scope and statistical interpretation exclude prompt-level claims about rare jumps or undisclosed source domains that have no registered observations.

### 3.3 Budget arithmetic and zero-bill gate

Public Modal list prices checked for this round are A100-80GB `$2.4984/h`, A100-40GB `$2.0988/h`, L4 `$0.7992/h` and A10 `$1.1016/h`, plus CPU `$0.04716/physical-core-h` and memory `$0.007992/GiB-h`. The A100 Function cost with 2 physical cores and 12 GiB is `$0.00074684/s` or `$2.688624/h`. A10 is the pricing label and `A10G` appears in Modal examples. The main schedule budgets the A100 and assumes no savings from smaller devices. [Modal pricing](https://modal.com/pricing)

| Job | Cap (s) | Compute cap (USD) |
| --- | ---: | ---: |
| A100 connector fit, `4×384×2.50` | 3,840 | 2.86786560 |
| A100 raw real, `ceil(82,560×0.030)` | 2,477 | 1.84992268 |
| A100 stress, `ceil(8,240×0.030)` | 248 | 0.18521632 |
| A100 synthetic, `ceil(40,960×0.030)` | 1,229 | 0.91786636 |
| A100 dev, `16,000×0.030` | 480 | 0.35848320 |
| A100 smoke, starts, setup and commit | 1,100 | 0.82152400 |
| A100 controlled retry reserve | 900 | 0.67215600 |
| CPU ingest, 1 core and 8 GiB | 1,800 | 0.05554800 |
| CPU causal frontend and physical readers, 4 cores and 16 GiB | 7,200 | 0.63302400 |
| CPU gate checks, image preparation allowance and aggregation, 2 cores and 8 GiB | 1,800 | 0.07912800 |
| **Total**, 2.8538889 A100 hours and 3 CPU function hours | | **8.44073416** |

The aggregate CPU charge outside GPU jobs is `$0.76770000`. Headroom is `$1.55926584` to the `$10.00` target and `$4.43926584` to the historical `$12.88` remaining credit. Reserve includes startup, stopped shards and retries. Compile the exact manifest and benchmark shapes before dispatching the registered blind rows. Run one billed function or shard at a time with no automatic retry/multi-container fanout, cap sum of billed durations including failed attempts, and stop when completion would exceed the cap. Use bounded shards and commit checkpoints and hashes before release to the next stage. Track first CPU image build and start costs against the CPU allowance and refresh dashboard arithmetic before each batch. The schedule is an arithmetic bound under the stated resource/time caps. Completion requires throughput and causal-alignment gates on dev, and the historical balance must be verified in the live workspace before dispatch.

Require account-visible `$0.00` out-of-pocket gating and sufficient remaining compute and volume/egress quotas before launch. On Modal Starter a workspace spend limit caps net charges separately from the workspace gross usage limit. Verify that the owner has set a net `$0.00` limit or an equivalent hard guard for the live workspace, and a gross cap covering the reserved stage amount within available credits. Use dashboard reading and control-plane caps as well as a local ledger because other jobs may consume credits and usage can lag. Hosted model APIs are a distinct charge; this run uses a frozen self-hosted model on Functions. No account state was verified or changed and no Modal workload was started in Round 1. [Modal budgets](https://modal.com/docs/guide/budgets) · [Modal billing](https://modal.com/docs/guide/billing)

## 4. CPU replay and interpretation

```powershell
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M4/verify_m4.py
```

The script loads the exact frozen weights and the archived branch source via read-only `git show`. It uses 8,192 examples per cell in the 168-cell mapping factorial and 32,768 examples per nuisance-curve point. This is a synthetic, matched-filter surrogate with a specified steering vector and colored proper Gaussian covariance. It verifies KL against quadrature or sampled log-density ratios, estimates scalar Bayes MI and empirical code capacity, and scores actual encoder codes and the frozen lookup table. At one full complex Gaussian point, (1) gives `1.196255` nats versus `1.198460 ± 0.004144` Monte Carlo (one standard error). A bounded random gain mixture with equiprobable denominators `0.7,1.3` has an estimated joint KL of `0.185511` nats after marginalization. Its scalar MI is `0.873286` nats with hidden gain and `0.889764` with observed gain. The centered scalar Student noise experiment has about `0.587033` nats of scalar MI for \(\nu=1\) with infinite forward Gaussian KL. All sweep inputs are generated surrogates. The optional metadata requests were refused by the local network. Historical manifest comparisons passed.

| Surrogate condition | R1 SER | R2 SER | R3 physical SER | R3 calibrated SER | R3 physical vs calibrated NLL |
| --- | ---: | ---: | ---: | ---: | ---: |
| −2 dB, clean gain/DC/sideband | 39.06% | 37.56% | 37.56% | 37.56% | 1.129 vs 1.008 |
| +2 dB, clean gain/DC/sideband | 20.74% | 19.98% | 19.98% | 19.98% | 0.532 vs 0.532 |
| +20 dB, clean gain/DC/sideband | 0.00% | 0.00% | 0.00% | 19.89% | 0.060 vs 0.543 |
| +20 dB, +30% AGC, DC 0.30 and spur 0.26 | 0.00% | 0.00% | 0.00% | 32.54% | 0.239 vs 0.879 |

The verifier checks directly that the frozen table chooses the turn's class for all four local codes, which explains SER equality of the scalar R2 and R3 arms. The matched sum and hence the turn coincide under the lift. The local code and calibration can still change. Attenuation at low SNR can improve frozen NLL without improving SER. The high-SNR loss from variance locking occurs when injected noise masks real high-SNR evidence. The observed saturation codes and conditional nuisance curves characterize the frozen weights under the stated draws. Generalization, real timing coverage and throughput remain empirical gates. See the full records for sample counts, scores, transition matrices, budget, manifests and geometry. [Verifier](verify_m4.py) · [Results](results_m4.json)
````

### Event #17: `item.completed` — `command_execution` (`item_14`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$a=Get-Content -Encoding UTF8 research/stage4_math_and_design/M5/THEOREM_M5.md; $a | Select-Object -Skip 110 -First 90; Get-ChildItem research/stage4_math_and_design/M1,research/stage4_math_and_design/M2,research/stage4_math_and_design/M3,research/stage4_math_and_design/M4,research/stage4_math_and_design/M5 -Name'
```

**Aggregated Output:**

```text

### 3.1 Bit-exact extension

Retain the original 2-bit turn and 2-bit learned inner index as a prefix. Add canonical phase, a calibrated log radial/reliability field, and residual quality. The Gaussian reference implementation uses $r=|T|/\sqrt{v_{\rm nom}}$, a unit canonical phasor with angle $\arg T-\pi\,\mathrm{turn}/2$ in $[-\pi/4,\pi/4)$, $\log(1+r)$ radial bins, and $E/v_{\rm nom}$ residual bins. The likelihood concentration is $2|\alpha T|/v$ if the gain and variance are known. The radial field alone does not determine both unknown gain and unknown variance. Carry calibrated estimates as metadata or additional bits, or infer them jointly using the cached evidence. Keep original observation units for amplitude forecasts.

| Payload bits/window | Turn | Legacy inner | Canonical angle | Log radial/reliability | Residual quality |
|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 0 | 0 | 0 | 0 |
| 4 | 2 | 2 | 0 | 0 | 0 |
| 8 | 2 | 2 | 2 | 1 | 1 |
| 12 | 2 | 2 | 4 | 3 | 1 |
| 16 | 2 | 2 | 6 | 4 | 2 |
| 24 | 2 | 2 | 8 | 8 | 4 |

Use the same boundary convention as the historical encoder, define a missing/invalid state, and compute the prefix in the encoding pass. Right-shifting by the extension length recovers s17 exactly for every supported input. This also preserves the original reader and its edge cases. Lossy polar coordinates alone cannot guarantee recovery of a learned label computed from the full raw sample vector. If a real-data adapter lacks a valid historical triplet, mark the legacy interface unavailable. For a Bayes-optimal decoder, an extended packet can always emulate the legacy reader. A strict gain for a query requires a refinement that divides some old cell into distinct conditional laws of positive mass. The benchmark reader uses the physical fields and leaves the legacy inner index unused, while still counting its bits. Its results therefore quantify an achievable decoder rather than the optimal legacy-plus-physical reader. The 12- and 16-bit residual thresholds are not nested. The 24-bit layout is proposed and has no risk estimate in this round.

Version the header with time origin/sample rate, template, gain and noise convention, calibration, quantizer edges, format ID, and capture/sample-span mapping. Record whether quality indicates clipping, noise, or a typed nuisance projection such as DC/spur. When required include irregular addresses, masks and per-window calibration bits in the budget. The CPU benchmark holds the address map and nominal amplitude/noise profile common to all readers and counts only the cache payload.

### 3.2 Stage 4→6 connector contract

| Component | Shared physical interface |
|---|---|
| EquivariantEvidenceConnector | Decode complex evidence, time, precision, quality, and masks. Return physical likelihood factors or calibrated candidate probabilities. Transform the reference and trajectory factors consistently under shared or per-window coordinate changes. |
| RefSufficientPool | Accept immutable acquisition ID, association status or probability, time, complex measurement and calibration. Deduplicate repeated renderings. A common latent phase and independent co-temporal Gaussian samples allow a weighted complex sum. Cross-time or correlated observations require likelihood factors/messages at the right nodes. |
| Stage 5 event head | Return a joint posterior over smooth, jump and interval, independent trajectory, and receiver/noise alternatives. Keep conditional state mixtures per rival for forecasting. Report interval coverage as well as maximum-probability times. |
| Stage 6 EIG selector | Consume the posterior, public candidate descriptors, and reveal handle. Commit one candidate ID, prior weights, predictive mixture for each rival, and a receipt. Append one physical factor after reveal and compute the updated law. |

The same numeric modules can serve Ministral 3B and Qwen 4B/9B/27B. Adapt only the host embedding width, special tokens or chat template and language readout calibration. Empirical transfer still requires fitting and testing each adapter. The 9B to Qwen3.8-27B comparison changes generation as well as scale. For LEMMO, report its native instruction-conditioned workflow and a separate matched-access comparison where supported. Use the same old observations, stored/effective precision, reveal allowance, questions and acquisition splits. To attribute a gap to encoding, also match training data, trainable parameters and optimization budget. Report context, latency, old IQ rereads, extra reads and unsupported tasks. LEMMO's published workflow uses Qwen3.5-4B and an instruction-conditioned retrieval stage, with LoRA during joint fine-tuning. [Host plan](../../../docs/MULTIMODAL_INTEGRATION_PLAN.md) · [LEMMO, Sections 3 and B](https://arxiv.org/html/2611.09476v1)

## 4. One-window Bayes EIG

At a fixed cache $z$, set $\pi_h=P(H=h\mid z)$, $f_{ha}(x)=\int p(Y_a=x\mid u,h,z)P(du\mid z,h)$, and $m_a(x)=\sum_h\pi_h f_{ha}(x)$. Drop $z$ inside the likelihood only under the required conditional independence. Predictor mixtures may have unequal amplitudes, phases, and uncertainty under the rivals.

**Theorem 3.** The exact one-window expected improvement in Bayes log score is

$$
V(a,z)=I(H;Y_a\mid z)
=\sum_h\pi_h\int f_{ha}(x)\log[f_{ha}(x)/m_a(x)]\,dx
=h_b(\pi_0)-E_{m_a}h_b(\pi_0f_{0a}(Y_a)/m_a(Y_a))
$$

for two rivals, with categorical entropy for more than two. It is weighted Jensen–Shannon divergence, zero iff all positive-weight predictions coincide and at most $H(H\mid z)$. Fusion uses $P(H=h\mid z,x)=\pi_h f_{ha}(x)/m_a(x)$. The expected realized gain $\log[f_{Ha}(x)/m_a(x)]$ gives the integral. This proves that the selector minimizing $K_z(a)$ maximizes $V$. For two simple equal-variance complex Gaussian predictions, let $\delta=|\mu_1-\mu_0|^2/v$ and $L=\delta+\sqrt{2\delta}W$, $W\sim N(0,1)$. Then

$$
V=-\pi_0E\log(\pi_0+\pi_1e^{-L})-\pi_1E\log(\pi_1+\pi_0e^{-L}).
$$

Under either true hypothesis, the log likelihood ratio in its direction has the law of $L$, which proves the scalar formula. Composite trajectories use the full predictive mixture; mean separation alone is insufficient. [Lindley, 1956](https://projecteuclid.org/euclid.aoms/1177728069)

Before reveal commit the action and both normalized forecast densities in a fixed observation space, together with prior weights. Log-score the mixture by $-\log m_a(x)$ and, when the generative label is known, its conditional density by $-\log f_{Ha}(x)$. A true-versus-rival score difference is a diagnostic. The mixture log score is strictly proper for the marginal prediction. Separate conditional forecast validation requires labeled or experimentally identified rivals. Use natural acquisitions with observable held-out targets and controlled interventions for labeled rivals. Declare the scored projection. Residuals that are common to the rivals cancel in their posterior update.

**Strict dominance.** For a randomized baseline $r(a\mid z)$, the conditional expected posterior-NLL advantage is exactly $\sum_a r(a\mid z)[V^*(z)-V(a,z)]$. For nearest it is $V^*(z)-V(a_{\rm near},z)$. Each baseline is strictly worse in expectation iff it assigns positive probability to a strictly inferior action on a positive-measure set of caches. Uniform random needs one inferior action there. Nearest must itself be inferior. These statements assume the true predictive law and equal read costs. With a certified uniform evaluation error $\max_a|\hat V(a,z)-V(a,z)|\le\epsilon$, the estimated maximizer incurs at most $2\epsilon$ regret in true EIG, by applying the error bound at both maximizers.

Under 0–1 loss the optimal success probability is $\int\max(\pi_0 f_{0a},\pi_1 f_{1a})=(1+\|\pi_0f_{0a}-\pi_1f_{1a}\|_1)/2$. It can rank actions differently from EIG. A sufficient structural condition for dominance for both losses is Blackwell ordering. If a candidate observation can simulate the other by an $H$-independent garbling, the informative observation can emulate every policy for the garbled one. Data processing gives strict EIG improvement when $H$ and the better observation remain dependent after seeing the garbling. Strict accuracy improvement requires the garbling to mix positive-probability predictive regions on opposite sides of the decision boundary. Two simple equal-variance Gaussian experiments are ordered by standardized mean separation: project onto the difference of means to get a scalar $X=\pm d/2+N(0,1)$; simulate the smaller separation by scaling and adding independent Gaussian noise of the remaining variance. [Blackwell, 1951](https://digicoll.lib.berkeley.edu/record/112749)

- **Change-point split.** Conditional on the jump model, suppose $Y_a$ probes only $S_a=1\{\tau\le t_a\}$ through an identical, binary-input symmetric channel at every candidate. Then $I(\tau;Y_a\mid z,H=\mathrm{jump})=I(S_a;Y_a\mid z,H=\mathrm{jump})$. In the noiseless case it is $h_b(P(\tau\le t_a\mid z,H=\mathrm{jump}))$. For distinct symmetric noisy densities it is strictly concave in the split mass because mixture entropy is strictly concave along the segment. Symmetry puts the maximum at a posterior median. An available split closer to $1/2$ is strictly better. If the objective is to distinguish the model class $H$, use the class-conditional mixtures. The median split is a theorem for this localization channel.
- **Anti-aliasing.** For known pilot symbols, phases at all old times agree for two simple slopes with shared anchored phase at $t_0$ when $\Delta\omega(t_k-t_0)\in2\pi\mathbb Z$. A known-symbol candidate that violates this relation gives positive EIG at positive finite precision and interior weights. With old times $0,2,4$, slope difference $\pi$, and alternatives at 6 and 1, the first candidate remains aliased and the second gives opposite phase. From the anchor at 4, nearest chooses 6. A uniform unknown QPSK symbol at the candidate also masks differences in $(\pi/2)\mathbb Z$, so the candidate needs a known symbol or a cross-window constraint. Posterior phase diffusion and uncertain association can erase otherwise useful separation.
- **Smooth versus jump.** Two trajectories can match at $0,D$ when the smooth slope is $\omega$ and the jump model uses $J$ at $\tau$ and slope $\omega-J/D$. Their phase difference at a candidate $t$ is $J(1\{\tau\le t\}-t/D)$. At fixed calibrated noise, an interior window can have strictly larger $2|\alpha|\left|\sin[J(1\{\tau\le t\}-t/D)/2]\right|/\sqrt v$ than nearest. For simple rivals this gives strict dominance by the Gaussian ordering above. With uncertain $\tau$ and fading, integrate the mixtures.

## 5. CPU verification

[verify_m5.py](verify_m5.py) loads pinned s17 weights and generates three-sample complex windows with two known old pilots and two hidden payload symbols. Q1 reads relative payload phase. Q2 adds a valid late reference at the time origin. The Stage 5 prior gives equal mass to smooth, jump, independent and variance-burst models. It enumerates two smooth slopes, three jump intervals, a reset to independently drawn phase and slope, and a local sixfold noise variance. Old and candidate phases lie on a 128-point circle grid. The simulator uses that finite trajectory law without continuous drift diffusion. Stage 6 separately draws smooth/jump mothers with equal model priors and commits four candidate predictive mixtures before opening the selected recording per policy. A smooth/jump pair can have identical old endpoint means. The model permits post-jump fading, with class-dependent amplitude for some candidates. The maximum old gain swing in the selective fade profile is $20\log_{10}(1.20/0.23)=14.35$ dB, across old SNRs of $-3$ to $+24$ dB.

Each of four profiles contributes 4,096 mothers to the old tasks and 2,048 other mothers to Stage 6. The s17 emission calibration uses 32,768 samples per canonical phase and distinct gain/variance configuration in a profile, an independent seed stream, C4 symmetry, and a half-count per cell. The canonical reader integrates Gaussian polar cell probabilities and uses the gamma CDF for the residual bins. The exact $(T,E)$ log likelihood was checked against a direct calculation from raw IQ over all simulated windows, phases and noise states. Stage 6 quadrature uses the continuously valued candidate matched sum and verifies an independent finer spacing on the first 64 mothers per method and profile. Candidate noise residuals have a common law at fixed address. The JSON keeps true and rival forecast scores, Bayes posterior NLL and accuracy, expected and observed NLL gains, action frequencies, paired intervals, and examples with the predictive distributions recorded before reveal. Because candidate noise scales and units affect differential log density, the raw observation NLL should be interpreted with the candidate and compared to the rival on the same reveal. [Full results](results_m5.json)

### 5.1 Achieved rate–risk points

All values are nats, pooled with equal profile weight. Stage 5 joint includes the event and jump interval. Stage 6 is the posterior model-label NLL under maximum EIG. The table evaluates specified encoder/readers. The exact optimized partition frontier of Theorem 2 is a different quantity.

| Bits/window | Q1 | Q2 | Stage 5 joint | Stage 6 post-reveal |
|---:|---:|---:|---:|---:|
| 2 | 1.1453 | 0.8616 | 1.5396 | 0.4619 |
| 4 (s17) | 0.9970 | 0.7966 | 1.2788 | 0.3677 |
| 8 | 0.9648 | 0.7856 | 1.0615 | 0.3614 |
| 12 | 0.9479 | 0.7536 | 0.9944 | 0.3545 |
| 16 | 0.9456 | 0.7493 | 0.9669 | 0.3544 |
| Continuous canonical / Full IQ | 0.9438 | 0.7425 | 0.8934 | 0.3538 |

The 16-bit gain over s17 is 0.3119 nat for event and location, paired pointwise 95% interval $[0.3023,0.3214]$. The remaining gap to Full IQ is 0.0735 nat. For the conditional smooth-versus-jump task the pooled NLL is 0.5240 (s17), 0.5136 (16 bit), and 0.5131 (Full IQ). For localization given a true jump it is 0.9711, 0.9382, and 0.9379, respectively. The recovery varies with regime. At +24 dB conditional binary NLL falls from 0.0300 with s17 to nearly zero with 16 bits. Under selective fading it is 0.6808 versus 0.6820, and both the 16-bit and continuous results are around 0.68. The observed low-SNR conditional localization remains near $\log3$ for all readers. Thus the larger joint event gain should not be read as an equally large binary-discrimination gain.

The actual s17 encoder emits code 1 for two real, in-cell matched sums of magnitude 0.0200 and 0.5833 in the 2 dB Gaussian witness. The exact four-label posterior changes by 0.4804 total variation. The JSON also includes two IQ vectors sharing their matched sum and orthogonal energy that give distinct s17 codes. This establishes the need for an explicit legacy prefix even with a continuous canonical Gaussian cache.

### 5.2 Sealed one-window selection

With the 16-bit cache, EIG, random, and nearest attain pooled NLLs 0.3544, 0.4497 and 0.4772, and accuracies 81.18%, 73.34% and 71.63%. The comparison below conditions on the same ambiguous-case stratum for all methods, defined for evaluation only by the old Full IQ smooth posterior in $[0.1,0.9]$. All 2,048 cases qualify in each of the three listed profiles. The +24 dB profile has no qualifying cases and shows a ceiling tie between policies at 16 bits.

| Ambiguous profile | Max EIG NLL / accuracy | Random NLL / accuracy | Nearest NLL / accuracy |
|---|---:|---:|---:|
| −3 dB | 0.6374 / 61.82% | 0.6742 / 55.96% | 0.6928 / 53.03% |
| +2 dB | 0.5434 / 72.27% | 0.6107 / 65.77% | 0.6445 / 63.28% |
results_m1.json
THEOREM_M1.md
verify_m1.py
reply_r1.md
results_m2.json
THEOREM_M2.md
verify_m2.py
reply_r1.md
results_m3.json
THEOREM_M3.md
verify_m3.py
reply_r1.md
results_m4.json
THEOREM_M4.md
verify_m4.py
reply_r1.md
results_m5.json
THEOREM_M5.md
verify_m5.py
```

### Event #18: `item.completed` — `web_search` (`exec-553e2acf-7038-4b42-bc31-c6733abad7e4`)

```json
{
  "id": "exec-553e2acf-7038-4b42-bc31-c6733abad7e4",
  "type": "web_search",
  "query": "",
  "action": {
    "type": "other"
  },
  "results": [
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn8view0",
      "snippet": "Total lines: 51",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/9vIUXHUrAeHiWPaMJ8krmeLWqpY9Mu_gFzgYRKE9k64ULj2uKzrMeYg0B9WMHJZ-fmrs7-7N02DO8t7q48UuTrS487SeRv5cIQGGdtpWcSzdtHpjjXdD3jqTI438y4dJ",
      "title": "Budgets | Modal Docs",
      "url": "https://modal.com/docs/guide/budgets"
    },
    {
      "type": "text_result",
      "domain": "modal.com",
      "ref_id": "turn8view1",
      "snippet": "Total lines: 84",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/AndblCUcFTJwxYPqK7YrNGp0qvwg48oh1PIUEs3ZP5gMiAkA1KwaLQs5SJP92X8HBSqpfm0FbjOOiGiMfeINNiKqpdWG_K5dDf6oyvmBxH7y4yhIF97j_lctTIYg-qCNoWbsX2c8tSmm8u4f4DFWHNA_64m8XnyazEmxZ0ErFN0",
      "title": "Configuring CPU, memory, and disk | Modal Docs",
      "url": "https://modal.com/docs/guide/resources"
    },
    {
      "type": "text_result",
      "domain": "huggingface.co",
      "ref_id": "turn8view2",
      "snippet": "Total lines: 735",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/EZQ80zZTk9NmqGkYP6WxKA8bOKsUZYxCVgwNmncdBZ91KfbabypagOXjR9J656zQy6Q_u9nOPTfoQ1X_2-rMCIv_VyMCEGPngpEX6AoC4OwEoYqrdkFByu6EEtDHBqsZ_HCJG7Z6Ms28z7a72vzyIw",
      "title": "Qwen/Qwen3.8-27B · Hugging Face",
      "url": "https://huggingface.co/Qwen/Qwen3.8-27B"
    }
  ]
}
```

### Event #19: `item.completed` — `command_execution` (`item_16`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding utf8 research/stage4_math_and_design/discussions/M6/prompt_r2.md; (Get-Content -Encoding utf8 research/stage4_math_and_design/M2/THEOREM_M2.md)[98..106]; (Get-Content -Encoding utf8 research/stage4_math_and_design/M2/THEOREM_M2.md)[142..148]; rg -n 'def .*packet|encode_packet|edges|quantiz' research/stage4_math_and_design/M5/verify_m5.py"
```

**Aggregated Output:**

```text
# Track M6 (Round 2): Final Converged Tier-0/1/2 Ledger, M1/M4 Reconciliation & Master Stage 4–6 Execution DAG

You are resuming persistent Codex session **Track M6** (`01a2194c-3b55-7c13-89f6-3ed11a482246`) in `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`. Keep edits STRICTLY inside `research/stage4_math_and_design/M6/` and run Python ONLY via `uv run --python 3.12 ...`.

## 1. Full Round-1 Cross-Track Picture to Synthesize & Lock
Look across ALL 5 companion tracks (`M1..M5`) + your own Round 1 analysis:
1. **The Reconciled Hardware & Statistical Budget Paradox (`M2` vs `M4` vs `M6`)**:
   - **VRAM reality from Stage 3D (`M4` catch)**: Stage 3D measured **`52,972.5 MB` peak train VRAM** (microbatch 16) and **`45,409.9 MB` peak eval VRAM** (batch 64) on `A100-80GB`! Therefore putting Pluto (`2,240` frames) + CTTC (`1,440` packets) blindly onto `L4 (24 GB)` in M6's baseline cap risks kernel/batch mismatch or OOM unless a small `$0.10` L4 parity smoke passes first!
   - **AND look at M4's A100-80GB math**: Pure **A100-80GB** across `4` matched single-seed arms (`384` steps/arm, synthetic + real) + `131,760` eval rows STILL totals only **`$8.4407`** (leaving `$4.44` below `$12.88`!). Wait — look carefully at minor rate/step differences between M4 (`A100-80GB $2.4984/h + 2c/12GiB = $0.00074684/s`, single seed `4 × 384 steps`, `2,477s` raw real, `1,229s` synth) vs M6 (`A100-80GB + 2c/16GiB = $0.00075572/s`, `173-step` pilot + `3` seeds continuation for ONE winning arm = `2,707 steps`).
   - **AND look at M2's statistical power theorem**: `3` seeds CANNOT establish population-of-seeds generalizability anyway (`~22–30%` crossed power, requiring `>= 10` seeds!), while `1` seed or `3` frozen seeds cleanly establish **Conditional Fixed-Connector Block/Capture/Loop-Pair Statistical Claims** (PLUS counterfactual fixed-text slot interventions & physical+residual comparisons)!
   - **Crucial Synthesis Decision Needed From M6**: Merge M4's VRAM-safe pure `A100-80GB` safety with M6's Tier-2 **1-seed pilot futility gate** (`18%` steps on disjoint dev BEFORE buying full `384`-step × `4`-arm comparison OR continuing `Arm 1 / Arm 2` to extra blocks/seeds!), keeping TOTAL worst-case cap **strictly `<= $9.50`** inside `~$12.88` remaining Modal free credits (`$0.00` net spend guard)!
2. **Causal Isolation (`Arm 0: Zero`, `Arm 0b: Naive-Additive`, `Arm 1: Host-Pure Equivariant+Coupled`, `Arm 2: Equivariant-Physical+HostResidual`)**:
   - M1's Round 1 showed `Arm 2` hits `64.62%` via its physical S17 prior + host residual, which M2 noted fails to isolate how much frozen Qwen contributes vs physical bypass!
   - Update your Tier-1 CPU Gate & Stage 4 DAG to explicitly require **BOTH** `Arm 1 (Host-Pure)` AND `Arm 2 (Hybrid)` Jacobian/OOD checks AND counterfactual slot/text controls!
3. **Dual-Cache Zero-GPU Upgrade (`4-bit s17` + `16-bit canonical` M5 Packet) + `RefSufficientPool` Non-Double-Counting (M3+M4)**:
   - M4's CPU shard builder already runs R3 whitening & feature computation on CPU (`<$0.77` total CPU cost!), so extracting BOTH `4-bit s17` AND M5's `16-bit canonical` packet simultaneously on CPU locks zero throwaway work across **Stage 4 → Stage 5 → Stage 6**!
   - Also integrate M4's critical causal discovery (`fast_blind_frontend` & `anchor_capture` non-causal future/whole-source correlations excluded -> strict causal filtration $\mathcal{F}^r$ + sentinel hash test + M3 non-double-counting!).

## 2. Specific Round-2 Convergence Tasks for Track M6
Update `research/stage4_math_and_design/M6/THEOREM_M6.md`, `verify_m6.py`, and `results_m6.json`:
1. **Publish ONE Converged, Contradiction-Free Master Modal Allocation Ledger (`<= $9.50` Hard Cap, 100% A100-80GB Safe)** unifying M4's `52.97 GB / 45.41 GB` VRAM numbers, M1's 4-arm causal matrix, M2's fixed-seed conditional inference scope + dev-disjoint futility stopping, and M5's dual-cache Stage 4→6 export! Explicitly show how much budget goes to Stage 4 Pilot, Arm Comparison, Real Pluto (`2,240`), Real CTTC (`1,440` incl. held-out `f708_snr10`), and Reserve (`>= $3.38` buffer vs `$12.88`!).
2. **Lock the Exact Numerical Tier-0 / Tier-1 / Tier-2 Go-No-Go Checklists** (ready for Antigravity to inspect and execute immediately after GitHub commit, with exact pass/fail predicates across every node `4A..6E` incorporating M1–M5 invariants!).
3. **Cross-Host & Fair `LEMMO` Execution Law (Stage 6 Readiness)**: Explicitly verify mathematically how fixing the `16-bit canonical` cache + `RefSufficientPool` lets us swap only tiny host adapters (`4B -> 9B -> 27B`) and run exact matched-support `LEMMO` comparison with minimum redundant experiments!
4. Run updated `verify_m6.py` via `uv run`, verifying all parity + Monte Carlo sequential cost/Type-I + converged Modal A100 ledger numbers cleanly in `results_m6.json`, and summarize in `reply_r2.md`.
## 3. Stage 4 frozen statistical gate

A claim passes when every endpoint for that claim clears its limit. Freeze the candidate, seed handling, allowed observation sets, per-class weighting, code below, multiplicity family, score floor, calibration, and all margins before unblinding. Apply the same three gate parts separately to the synthetic extrapolation claim, the Pluto claim and the CTTC claim. For a claim covering all four synthetic cells, require each cell to pass. Report each failed or indeterminate endpoint. A winner picked from several candidates requires a design split or inclusion in the multiplicity family.

1. **Absolute and comparative performance.** Use paired lower bounds. Require NLL to beat uniform, the dev-frozen prior, and the matched zero-phase system, with a \(0.02\)-nat increment proposed against matched zero. Require accuracy above analytic chance, a dev-frozen prior classifier, and the matched zero-phase system, with \(+0.02\) absolute over chance and zero proposed as initial margins. With imbalanced classes, also require outperformance of the fixed majority prediction or used prior competitor, and report balanced accuracy as a separately locked estimand. Require valid chance/random-label randomization tails for the accuracy and NLL statistics where the design supplies exchangeability. Choose an oracle with the same information budget as the candidate for the main NIG denominator and disclose a finer physical oracle separately. Propose \(\eta_{\min}=0.10\) and validate \(H(Y)-L(o)>0\) on an independent development design or by a co-primary lower bound. Test the linear contrast \(D_{\eta}=(1-\eta)H(Y)-L(q)+\eta L(o)>0\) rather than dividing two noisy quantities. Use \(H(Y)=\log K\) for a genuinely uniform target law. Otherwise predefine the target distribution or carry uncertainty in \(H(Y)\) by a joint confidence region and minimize the contrast over it. For real data without a defensible Bayes oracle, freeze a physically informed predictive reference, call the resulting metric reference-normalized, and withhold the Bayes-normalized claim. Requiring a learner to outperform a true Bayes oracle under proper loss is infeasible. A positive NIG floor defines a fraction of the attainable improvement. Controlled slot/text interventions form a further prerequisite for a specifically signed-slot mechanism claim. Require improvement with matched versus permuted or flipped slots at fixed text and same \(X\), with seeded training and ablations paired. Use the native receiver outputs for the gate.
2. **Calibration and structure.** Propose an upper bound of 0.05 on top-label ECE after dev-only calibration with ten frozen bins. Propose a paired Brier improvement of at least 0.005 against the strongest locked trivial or zero-phase Brier competitor, plus a strict improvement over every required comparator. For \(K=4\) uniform Brier is \(1-1/K=0.75\) with the unnormalized multiclass sum. Prelock a probability transformation ensuring finite NLL, for example \(q_\epsilon=(1-K\epsilon)q+\epsilon\) with small fixed \(\epsilon\), and score that forecast consistently. For calibration inference set \(d_j=\mathbb E[1_{\{\operatorname{conf}\in I_j\}}(h_Y-\operatorname{conf})]\); \(\operatorname{ECE}=\sum_j|d_j|=\max_{v\in\{-1,1\}^{10}}v^\top d\). Test all \(2^{10}\) linear upper-margin conditions as an IUT or use a valid joint confidence region for \(d\). Estimating the absolute value first and treating empirical ECE as an unbiased, normal per-mother loss has no exact justification. Require \(\max_{g,x}\|q(gx)-\rho(g)q(x)\|_\infty\le10^{-5}\) and projection/readout idempotence \(\max_x\|P(Pq)(x)-Pq(x)\|_\infty\le10^{-5}\). Verify across the whole finite code domain and all group elements on synthetic. For continuous real inputs distinguish formal architectural guarantees from the observed maximum on prespecified transforms. Test only valid actions with physically correct label permutations. Audit the deployed path and the offline projector separately.
3. **Hierarchy and multiplicity.** Keep all repeats, rotations, acquisitions, and matched arms inside their mother and higher cluster. Report a frozen-seed conditional estimate and, if claiming generalization to retraining, a separate random-seed inference. Use equal independent cluster weights for the following exact balanced analysis. Pluto supplies 16 units if they are independent captures. For a CTTC new-session claim use nine independent session means, with all eight clusters per session contributing to that session. A conditional claim on the fixed sessions needs the corresponding estimand and defensible within-session cluster independence. For a permutation or label-randomization gate, lock the allowed exchangeability orbit and recompute the same statistic throughout. Repeated/rotated labels move jointly with their mother and never split correlated session units as though independent. A synthetic randomized design with independent label assignments can instead rerandomize the sanctioned generator operation inside the original design. When there is no justified exchangeability, collect a randomized negative control. State whether it tests a sharp random-label null or a mean-effect margin. A permutation p-value \((1+\#\{T_{\rm perm}\ge T_{\rm obs}\})/(R+1)\) with uniform randomizations and ties included is conservative under the sharp design. Use exact enumeration when feasible.

### Exact type-I control and power under a balanced Gaussian model
B\ \ge\ \frac{\sigma_u^2+\sigma_e^2/S}{d^2/z^2-\sigma_\nu^2/S}
\]
with positive denominator. At fixed \(S\) this is a necessary known-variance bound for the random-seed model. The verifier solves the exact t power and evaluates the crossed chi-square integral for finite counts. It uses a worst Holm first-step threshold \(\alpha=0.05/3\) and marginal power 0.95 to guarantee at least 0.90 power for the two historical paired endpoints if both assumptions and excess effects hold. This target does not imply 0.90 power for the full Stage 4 gate. Full planning requires variance and target excess for every required absolute/calibration endpoint and a joint power calculation or an explicit allocation of endpoint type-II errors. Stage 3D supplies a synthetic variance reference. It cannot measure Pluto or CTTC variability.




110:def radial_edges(bits):
113:    edges = np.expm1(np.linspace(0, np.log(25), 2**bits + 1))
114:    edges[-1] = np.inf
115:    return edges
118:def quality_edges(bits):
136:        radius = np.searchsorted(radial_edges(radial_bits)[1:-1], abs(t) / np.sqrt(variance))
137:        quality = np.searchsorted(quality_edges(quality_bits)[1:-1], residual / variance)
170:    r_edge = radial_edges(radial_bits) / np.sqrt(noise_ratio)
188:            edges = np.linspace(r_edge[k], r_edge[k + 1], 9)
189:            r = ((edges[:-1, None] + edges[1:, None]) / 2 + np.diff(edges)[:, None] * node / 2).ravel()
190:            rw = (np.diff(edges)[:, None] * weight / 2).ravel() * r
198:def window_log_likelihood(method, t, residual, packets, bins, a, v, encoder, key):
221:                gamma_mass = np.diff(gammainc(2, quality_edges(quality_bits) / multiplier))
```

### Event #20: `item.completed` — `command_execution` (`item_17`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "uv run --offline --no-project --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B -c 'from pathlib import Path; import numpy, scipy; print(numpy.__version__, scipy.__version__); import json; r=Path(\"\"research/stage4_math_and_design\"\"); print({f.parent.name:list(json.loads(f.read_text(encoding=\"\"utf-8\"\"))) for f in sorted(r.glob(\"\"M[1-5]/results*.json\"\"))})'"
```

**Aggregated Output:**

```text
2.3.5 1.17.0
{'M1': ['runtime', 'train', 'checks', 'all_task_quotient_lift_defects', 'cells', 'boundary_scan_fixed_variance_0p04', 'stage3d_saved_cells', 'work_f_recomputed', 'work_g_recomputed', 'work_f_overlap_weighted_signed_contrast_spectra', 'pf_complete_orbit_counterexample'], 'M2': ['runtime', 'Q3', 'Q2', 'npz_shapes', 'summary_checks_max_absolute_difference', 'work_f_primary_040', 'work_f_historical_035', 'work_f_intervention_factorial', 'power'], 'M3': ['runtime', 'condition', 'kernel_fields', 'stage3d', 'surrogate_fit', 'surrogate', 'multi_timestamp'], 'M4': ['status', 'seed', 'python', 'numpy', 'scipy', 'scope', 'encoder_sha256', 'nominal_complex_noise_variance', 'ref_commits', 'distributions', 'frozen_encoder_sweep', 'nuisance_curves', 'index_audit', 'dataset_manifests', 'modal_budget'], 'M5': ['protocol', 's17_radial_collision_witness', 's17_inner_not_determined_by_matched_sum_and_residual_energy', 'nll_pareto_table', 'finite_pointwise_nondominated_including_rate', 'paired_excess_nll_vs_full_iq', 'paired_nll_16_minus_s17', 'paired_profile_nll_16_minus_s17', 'per_profile']}
```

### Event #21: `item.completed` — `web_search` (`exec-5a8fdbbc-5f34-4446-a1e9-23b59e11a539`)

```json
{
  "id": "exec-5a8fdbbc-5f34-4446-a1e9-23b59e11a539",
  "type": "web_search",
  "query": "https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json",
  "action": {
    "type": "open_page",
    "url": "https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json"
  },
  "results": [
    {
      "type": "text_result",
      "domain": "huggingface.co",
      "ref_id": "turn9view0",
      "snippet": "Total lines: 104",
      "url": "https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/config.json"
    },
    {
      "type": "text_result",
      "domain": "huggingface.co",
      "ref_id": "turn9view1",
      "snippet": "Total lines: 103",
      "url": "https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json"
    },
    {
      "type": "text_result",
      "domain": "huggingface.co",
      "ref_id": "turn9view2",
      "snippet": "Total lines: 140",
      "url": "https://huggingface.co/Qwen/Qwen3.8-27B/raw/main/config.json"
    }
  ]
}
```

### Event #22: `item.completed` — `command_execution` (`item_19`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$p='"'research/stage4_math_and_design'; foreach("'$i in 1..5){$j=Get-Content -Raw -Encoding UTF8 (Join-Path $p "M$i/results_m$i.json") | ConvertFrom-Json; if($i -eq 4){$j.modal_budget | ConvertTo-Json -Depth 5; $j.index_audit.PSObject.Properties.Name}; if($i -eq 2){$j.power.PSObject.Properties.Name}; if($i -eq 5){$j.protocol | ConvertTo-Json -Depth 4}; if($i -eq 1){$j.checks.PSObject.Properties.Name}}; (Get-Content -Encoding UTF8 research/stage4_math_and_design/M5/verify_m5.py)[109..142]; git status --short --untracked-files=all | Select-String '"'reply_r2|M6'"
```

**Aggregated Output:**

```text
B_all_grouped_cell_defects_equal_zero
B_all_cell_defects_at_most_1e_6
B_all_ungrouped_cell_defects_at_most_1e_6
A_breaks_equivariance
B_G1_G3_each_seed_above_55_pct
B_all_cell_means_within_3pp_of_C
A_G1_G3_mean_at_most_25_pct
stage3_lock_replay
planning_alpha
target_marginal_power
seed_count
MC_seed
MC_reps
variance_components
scenario_proxies
multiple_testing_check
synthetic_cell_plans
G1_more_seeds
ten_seed_sensitivity
{
    "official_pricing_url":  "https://modal.com/pricing",
    "usd_per_second":  {
                           "A100-80GB":  "0.000694",
                           "A100-40GB":  "0.000583",
                           "L4":  "0.000222",
                           "A10":  "0.000306",
                           "CPU":  "0"
                       },
    "physical_core_usd_per_second":  "0.0000131",
    "gib_usd_per_second":  "0.00000222",
    "rows":  [
                 {
                     "phase":  "connector_train",
                     "resource":  "A100-80GB",
                     "seconds_cap":  3840,
                     "physical_cores_cap":  2,
                     "gib_cap":  12,
                     "usd_cap":  "2.86786560"
                 },
                 {
                     "phase":  "real_raw",
                     "resource":  "A100-80GB",
                     "seconds_cap":  2477,
                     "physical_cores_cap":  2,
                     "gib_cap":  12,
                     "usd_cap":  "1.84992268"
                 },
                 {
                     "phase":  "real_stress_q2",
                     "resource":  "A100-80GB",
                     "seconds_cap":  248,
                     "physical_cores_cap":  2,
                     "gib_cap":  12,
                     "usd_cap":  "0.18521632"
                 },
                 {
                     "phase":  "synthetic_blind",
                     "resource":  "A100-80GB",
                     "seconds_cap":  1229,
                     "physical_cores_cap":  2,
                     "gib_cap":  12,
                     "usd_cap":  "0.91786636"
                 },
                 {
                     "phase":  "dev_calibration",
                     "resource":  "A100-80GB",
                     "seconds_cap":  480,
                     "physical_cores_cap":  2,
                     "gib_cap":  12,
                     "usd_cap":  "0.35848320"
                 },
                 {
                     "phase":  "gpu_smoke_boot_and_commit",
                     "resource":  "A100-80GB",
                     "seconds_cap":  1100,
                     "physical_cores_cap":  2,
                     "gib_cap":  12,
                     "usd_cap":  "0.82152400"
                 },
                 {
                     "phase":  "gpu_retry_reserve",
                     "resource":  "A100-80GB",
                     "seconds_cap":  900,
                     "physical_cores_cap":  2,
                     "gib_cap":  12,
                     "usd_cap":  "0.67215600"
                 },
                 {
                     "phase":  "ingest_and_verify",
                     "resource":  "CPU",
                     "seconds_cap":  1800,
                     "physical_cores_cap":  1,
                     "gib_cap":  8,
                     "usd_cap":  "0.05554800"
                 },
                 {
                     "phase":  "frontend_and_physical",
                     "resource":  "CPU",
                     "seconds_cap":  7200,
                     "physical_cores_cap":  4,
                     "gib_cap":  16,
                     "usd_cap":  "0.63302400"
                 },
                 {
                     "phase":  "audit_and_aggregation",
                     "resource":  "CPU",
                     "seconds_cap":  1800,
                     "physical_cores_cap":  2,
                     "gib_cap":  8,
                     "usd_cap":  "0.07912800"
                 }
             ],
    "stage3d_model_revision":  "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    "stage3d_host_sha256":  "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d",
    "stage3d_locked_training":  {
                                    "arm_names":  [
                                                      "phase",
                                                      "zero"
                                                  ],
                                    "connector_seeds":  [
                                                            202611,
                                                            202622,
                                                            202633
                                                        ],
                                    "trainable_params":  2184964,
                                    "epochs":  8,
                                    "batch_size":  32,
                                    "micro_batch_size":  16,
                                    "lr":  0.0015,
                                    "max_steps_per_epoch":  160,
                                    "train_scorer":  "kv_cached",
                                    "eval_scorer":  "kv_cached",
                                    "eval_every_epochs":  2
                                },
    "stage3d_run_wall_seconds":  [
                                     2884.0,
                                     3026.2,
                                     3217.8,
                                     2908.7,
                                     3059.1,
                                     2983.0
                                 ],
    "stage3d_observed":  {
                             "training_seconds_per_step":  1.9757,
                             "eval_seconds_per_row":  0.02388,
                             "train_peak_mb":  52972.5,
                             "eval_peak_mb":  45409.9
                         },
    "stage4_eval_rows":  {
                             "real_raw":  82560,
                             "real_stress_q2":  8240,
                             "synthetic_blind":  40960,
                             "dev_calibration":  16000
                         },
    "stage4_training_examples_per_arm":  12288,
    "stage4_a100_seconds_cap":  10274,
    "stage4_total_usd_cap":  "8.44073416",
    "stage3d_free_at_report_usd":  "12.88",
    "stage3d_reported_billed_usd":  "0.00",
    "headroom_to_ten_usd":  "1.55926584",
    "headroom_to_historical_remaining_usd":  "4.43926584",
    "active_balance_or_modal_execution_verified":  false
}
pluto
cttc
scope
{
    "seed":  50261126,
    "python":  "3.12.7",
    "numpy":  "2.3.5",
    "scipy":  "1.17.0",
    "mothers_per_profile_old":  4096,
    "mothers_per_profile_blind":  2048,
    "s17_calibration_samples_per_canonical_phase_per_distinct_gain_variance_within_profile":  32768,
    "s17_checkpoint_sha256":  "e67d9fbafb90791ab19549de7789798d818286d7f52d7ac566cc39df71e78356",
    "sample_phase_grid":  128,
    "old_times":  [
                      0,
                      2,
                      6,
                      10
                  ],
    "payload_windows":  [
                            2,
                            6
                        ],
    "candidate_times":  [
                            11,
                            1,
                            4,
                            8
                        ],
    "jump_times":  [
                       2.5,
                       4.5,
                       7.5
                   ],
    "event_location_labels":  [
                                  "smooth",
                                  "jump_2.5",
                                  "jump_4.5",
                                  "jump_7.5",
                                  "independent",
                                  "burst"
                              ],
    "phase_slope_grid":  [
                             0.18,
                             0.34
                         ],
    "cache_extensions_angle_radius_quality_bits":  {
                                                       "2":  [
                                                                 0,
                                                                 0,
                                                                 0
                                                             ],
                                                       "8":  [
                                                                 2,
                                                                 1,
                                                                 1
                                                             ],
                                                       "12":  [
                                                                  4,
                                                                  3,
                                                                  1
                                                              ],
                                                       "16":  [
                                                                  6,
                                                                  4,
                                                                  2
                                                              ],
                                                       "24":  [
                                                                  8,
                                                                  8,
                                                                  4
                                                              ]
                                                   },
    "compact_reader":  "integrated physical cells; packed legacy inner bits available but ignored",
    "full_iq":  "distinct raw likelihood verified against canonical (T,residual) factorization",
    "cache_and_candidate_visibility":  "all candidate decisions and predictive densities committed before first reveal",
    "common_ambiguity_subset":  "old Full IQ smooth posterior in [0.1, 0.9]; evaluation only"
}
    mean = amplitude * UNIT[(phase[state] + shifts) % PHASES]
    iq = mean[..., None] / np.sqrt(3) + np.sqrt((variance * burst[state])[..., None]) * normal(rng, mean.shape + (3,))
    ref_variance = 10**(-ref_db / 10)
    late_reference = UNIT[phase[state, 0]] + np.sqrt(ref_variance) * normal(rng, count)
    return state, symbols, iq, late_reference, ref_variance


def features(iq):
    t = iq.sum(axis=-1) / np.sqrt(3)
    residual = np.sum(abs(iq - t[..., None] / np.sqrt(3))**2, axis=-1)
    return t, residual


def radial_edges(bits):
    if bits == 0:
        return np.array([0., np.inf])
    edges = np.expm1(np.linspace(0, np.log(25), 2**bits + 1))
    edges[-1] = np.inf
    return edges


def quality_edges(bits):
    if bits == 0:
        return np.array([0., np.inf])
    if bits == 1:
        return np.array([0., 4., np.inf])
    return np.r_[0., np.geomspace(.75, 16., 2**bits - 1), np.inf]


def encode(iq, t, residual, variance, encoder):
    original = encoder.encode(iq.reshape(-1, 3)).reshape(t.shape)
    _, turns = input_features(iq.reshape(-1, 3), True)
    turns = turns.reshape(t.shape)
    np.testing.assert_array_equal(original // 4, turns)

?? research/stage4_math_and_design/M6/THEOREM_M6.md
?? research/stage4_math_and_design/M6/reply_r1.md
?? research/stage4_math_and_design/M6/results_m6.json
?? research/stage4_math_and_design/M6/verify_m6.py
?? research/stage4_math_and_design/discussions/M6/prompt_r1.md
?? research/stage4_math_and_design/discussions/M6/prompt_r2.md
?? research/stage4_math_and_design/discussions/M6/reply_r1.md
?? research/stage4_math_and_design/discussions/M6/round1_events.jsonl
?? research/stage4_math_and_design/discussions/M6/round2_events.jsonl


```

### Event #23: `item.completed` — `agent_message` (`item_20`)

The main budget difference is the experimental unit. M4 funds four matched single-seed arms, while M6 Round 1 funded three seeds of one arm. I’ll use four matched 69-step pilots on A100, then resume them to 384 steps only if the development gate passes. I’ll price the slot, text, and residual controls explicitly.

### Event #24: `item.completed` — `file_change` (`item_21`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M6\\verify_m6.py",
    "kind": "update"
  }
]
```

### Event #25: `item.completed` — `file_change` (`item_22`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M6\\verify_m6.py",
    "kind": "update"
  }
]
```

### Event #26: `item.completed` — `command_execution` (`item_23`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$s=Get-Content -Encoding UTF8 research/stage4_math_and_design/M1/THEOREM_M1.md; $s[7..87]; $v=Get-Content -Encoding UTF8 research/stage4_math_and_design/M6/verify_m6.py; $v[0..34]'
```

**Aggregated Output:**

```text

Use \(\mathbb Z_4\) labels and roots \(i^j\) in the four-choice order. Q1 asks for \(X_2-X_1\), Q2 asks for \(X_1\), and Q3 asks for the sector of the third matched-filter output. The sectors are \(I_k=[-\pi/4,\pi/4)+k\pi/2\) modulo \(2\pi\). With \(T=\sum_{m=1}^3Y_m/\sqrt3\), window times \(0,h,\tau=h+\ell\), and \(\rho=10^{\mathrm{SNR}_{\mathrm{dB}}/10}\), the Stage 3D generator has

\[
\phi(t)=\omega t+\sigma_\phi B_t,\qquad
T_j=i^{X_j+G}e^{i\phi(t_j)}+\epsilon_j,\qquad
\epsilon_j\sim\mathcal{CN}(0,\rho^{-1}).
\]

Here \(B\) is standard Brownian motion, \(G\) is the uniform QPSK channel turn, and \(X_1,X_2\) and the fresh \(X_3\) are independent and uniform. A shared-channel pilot transmits a known root (root 1 in Stage 3D), so \(T_p=i^Ge^{i\phi(t_p)}+\epsilon_p\). Observation noises are independent within a mother. The balanced blocks sample this marginal law, and the two CFO signs share innovations and noises for paired comparison. The posterior below conditions on one mother's observables. [Schema](../../stage3_ministral_cache/schema.py) · [Generator](../../stage3d_qwen_signed_phase/data_stage3d.py)

### Theorem 1 — Encoder and conditional choice actions

Write \(Z_1=4a+u,\ Z_2=4b+v\), where \(u,v\in\{0,1,2,3\}\). For \(g=(r,s)\in H=C_4\times C_4\), rotating the two windows by \(i^r,i^s\) induces \((a,b,u,v)\mapsto(a+r,b+s,u,v)\). With pilot and metadata fixed, the true conditional choice law obeys \(P(gx)=R_{\chi(g)}P(x)\), where \((R_jp)_k=p_{k-j}\).

| Task | Shift \(\chi(r,s)\) | Target |
|---|---:|---|
| `q1` | \(s-r\) | Relative transmitted turn |
| `q2` | \(r\) | First transmitted turn, fixed pilot frame |
| `q3_repeat` | \(r\) | Received sector of a retransmission of \(X_1\) |
| `q3_indep` | \(0\) | Received sector of a fresh independent symbol, with exactly uniform conditional probabilities |
| `q3_after_pilot` | \(r\) | Received repeat sector given the revealed pilots |

**Proof.** The encoder takes the sector of the sum of the three samples, falling back to the first nonzero sample on a zero sum. The half-open convention gives \(\operatorname{sector}(i^rT)=\operatorname{sector}(T)+r\) for nonzero \(T\), including boundary rays. Undoing the selected turn gives the same canonical I/Q features after rotation. The fitted MLP and center assignment therefore return the same inner index. The all-zero window is fixed and cannot carry a nontrivial turn. It has probability zero under the stated noise. A hardware interface can mark it as an erasure. [Encoder](../../matched_cache/learned.py)

For the joint law, relabel \(X_1\mapsto X_1+r,\ X_2\mapsto X_2+s\), and rotate noises in windows 1, 2, and the repeat window by their designated turns. Keep \(G,\phi\), and the pilots fixed. The uniform source and circular noise laws preserve the joint measure. The three nonuniform label shifts then follow by subtraction, by reading \(X_1\), or by taking the sector. This yields the conditional identities on the support. For the fresh symbol, condition further on \(G\) and the entire phase path. The new symbol and noise are still independent, and mixing the four roots makes its received law quarter-turn invariant. Marginalization preserves the uniform sector law. \(\square\)

To allow a common pilot rotation \(r_p\), relabel \(G\mapsto G+r_p\), \(X_1\mapsto X_1+r-r_p\) and \(X_2\mapsto X_2+s-r_p\). Rotate all shared pilots by \(i^{r_p}\) and the repeat window by \(i^r\). Q2 then shifts by \(r-r_p\), while Q1 and the repeat tasks retain their shifts. In particular, Q2 is invariant under a joint receiver-coordinate rotation \(r=s=r_p\). A common pilot turn respects the shared channel and phase path.

### Theorem 2 — Quotient factorization

Fix an invariant context \(c\), including signed CFO, timestamps, and references in a fixed frame. A four-score map is \(H\)-equivariant if and only if

\[
f(4a+u,4b+v;c)=R_{\chi(a,b)}F(u,v;c).
\]

**Proof.** Every code orbit intersects the canonical slice once. Applying equivariance to that representative proves necessity. The shift rule is additive, so \(R_jR_k=R_{j+k}\) proves sufficiency. A deterministic frozen host can be part of \(F\), because its inputs are identical across the orbit. Normalize in the canonical order and gather by integer indices to obtain bitwise equality for identical canonical computations. With an explicit common pilot frame, canonicalize that evidence and use the extended Q2 shift. For Q3 independent, return exact quarters. \(\square\) [Stage 3C canonicalization and readout](../../stage3c/G/quotient.py)

There are 256 pairs and 16 representatives. Arbitrary probability tables have \(256(4-1)=768\) degrees of freedom per context. The quotient has \(16(4-1)=48\). This is an exact factor of 16 in table degrees of freedom and gives the pooling available in balanced table fitting. Sample complexity for a connector also depends on the source law, hypothesis class, initialization, and objective.

## 2. Continuous phase transport and the sector dip

### Theorem 3 — Circular phase evolution before the sector map

Angle translation by \(\alpha\in S^1\simeq SO(2)\) multiplies \(m_n=\mathbb E[e^{in\varphi}]\) by \(e^{in\alpha}\). For an edge of duration \(d\ge0\), the Gaussian characteristic function gives

\[
U_n(d)=e^{in\omega d}e^{-n^2\sigma_\phi^2d/2},\qquad
m_n(t+d)=U_n(d)m_n(t).
\]

Independent increments yield \(U_n(d_1+d_2)=U_n(d_1)U_n(d_2)\) for constant parameters. Diffusion is an irreversible convolution semigroup for \(\sigma_\phi>0\). To undo a known deterministic drift use the conjugate phasor. Diffusion marginalization continues to use damping.

The quarter-turn actions commute with continuous translations wherever both are realized as rotations of the underlying continuous variables. Conjugation therefore gives the trivial \(S^1\to\operatorname{Aut}(H)\) action. The corresponding semidirect product is a direct product, possibly with a kernel if two factors rotate the same variable. Any continuous action \(S^1\to\operatorname{Aut}(H)\) is trivial because the domain is connected and the codomain is finite. The same reasoning precludes a nontrivial continuous action on the finite codes or choices by permutations. A CFO change is an \(\mathbb R\) flow on the joint edge-phase torus, with edge phases \(\Delta\omega\,d\). It has a common frequency period only if the active durations are commensurate. [Group-action background](https://proceedings.mlr.press/v48/cohenc16.html)

### Theorem 4 — Circulant Q3 pushforward

Let \(\Phi\) be the standard normal CDF, \(\kappa=\sqrt{2\rho}\), and
\(c=\Phi(\kappa\cos(\phi+\pi/4)),\ s=\Phi(\kappa\sin(\phi+\pi/4))\).
For source root 1 the response is

\[
B(\phi)=(cs,(1-c)s,(1-c)(1-s),c(1-s)),\qquad
K(\theta,V)=\mathbb E_{\xi\sim N(0,1)}B(\theta+\sqrt V\,\xi).
\]

With \(A_z(j)\) the adopted posterior on the first apparent transmitted root \(i^{X_1+G}\), the Stage 3D direct S17 table is

\[
q_k(z;\theta,V)=\sum_{j=0}^3 A_z(j)K_{k-j}(\theta,V),
\qquad \theta=\omega(h+\ell),\quad V=\sigma_\phi^2(h+\ell).
\]

**Proof.** Orthogonal projection along the two sector boundaries yields independent normals with the indicated means after standardization. The sign pairs in label order are \(++,-+,--,+-\). Independence gives the four products. Integrate the Brownian angle and use circular observation noise to shift by the source root. The resulting map is a cyclic convolution, diagonalized by the \(C_4\) DFT. Since \(B(\phi+j\pi/2)=R_jB(\phi)\), it gives \(q(z;\theta+j\pi/2,V)=R_jq(z;\theta,V)\) at fixed \(A_z,V,\rho\). \(\square\) [Stage 3D analytical readout](../../stage3_ministral_cache/analytic.py)

The verifier's \(C\) evaluates this integral by 64-point Gauss–Hermite quadrature. Stage 3D evaluates it on a wrapped grid. The formula is exact for the adopted \(A_z\) and marginal target phase distribution, with numerical integration in both implementations. \(A_z\) is a fitted conditional table, and this readout leaves out phase information from \(Z_2\). The archived `cal_ref` is a separate independently simulated two-code reference, pooled by quarter-turn symmetry with half-count smoothing.

For the exact population Bayes rule based on both codes and references, write \(S_1=X_1+G,\ S_2=X_2+G\). Define \(A_u(j)=\Pr(S_1=j\mid Z_1=u)\) in the canonical frame, and \(L_v(j,\psi)=\Pr(Z_2=v\mid S_2=j,\phi_2=\psi)\) from the quantizer regions and observation law. A shared-reference likelihood \(L_P(G,\boldsymbol\phi)\) uses the joint phase path. The canonical joint posterior is proportional to

\[
"""Replay Stage 3D scores, audit CPU gates, and verify the proposed Stage 4 budget."""

import json
import hashlib
import math
import platform
from decimal import Decimal
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import binomtest, norm, t as student_t


HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / "stage3d_qwen_signed_phase" / "results"
CELLS = (
    "G1_main_extrap_both", "G2_interp_both", "G3_extrap_angle",
    "G4_extrap_freq", "H_c4_035",
)
ARMS = ("phase", "zero")
Q2_ARMS = (
    "none", "valid", "duplicate_copy_dedup", "duplicate_copy_raw",
    "independent_only", "valid_plus_independent", "fresh_resample",
)
Q2_REF_INDEX = (0, 1, 2, 2, 3, 4, 5)
TOL = 1e-12

R2 = {
    "seed": 202611,
    "arm_ids": ["0", "0b", "1", "2"],
```

### Event #27: `item.completed` — `file_change` (`item_24`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M6\\verify_m6.py",
    "kind": "update"
  }
]
```

### Event #28: `item.completed` — `command_execution` (`item_25`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'uv run --offline --no-project --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M6/verify_m6.py'
```

**Aggregated Output:**

```text
VERIFIED results_m6.json: parity max |diff|=3.56e-16, two 10,000-trial designs, A100 cap $9.50
```

### Event #29: `item.completed` — `command_execution` (`item_26`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$i=Get-Content -Raw -Encoding UTF8 research/stage4_math_and_design/M6/results_m6.json | ConvertFrom-Json; $i.modal_budget | Select-Object -ExcludeProperty rows | ConvertTo-Json -Depth 4; $i.round2_disjoint_pilot_model | ConvertTo-Json -Depth 5; $i.round2_cpu_contract_checks | ConvertTo-Json -Depth 6; $m=Get-Content -Raw -Encoding UTF8 research/stage4_math_and_design/M4/results_m4.json | ConvertFrom-Json; $m.index_audit.cttc | Select-Object -First 1 | ConvertTo-Json -Depth 1'
```

**Aggregated Output:**

```text
{
    "pricing_url":  "https://modal.com/pricing",
    "only_gpu":  "A100-80GB",
    "rows":  [
                 {
                     "task":  "cpu_ingest_weight_staging",
                     "group":  "cpu",
                     "device":  "CPU",
                     "max_billed_seconds":  1800,
                     "max_physical_cores":  1,
                     "max_gib":  8,
                     "train_steps":  null,
                     "logical_eval_rows":  null,
                     "max_usd_per_second":  "0.00003086",
                     "max_usd":  "0.05554800"
                 },
                 {
                     "task":  "cpu_causal_frontend_dual_cache_pool",
                     "group":  "cpu",
                     "device":  "CPU",
                     "max_billed_seconds":  7200,
                     "max_physical_cores":  4,
                     "max_gib":  16,
                     "train_steps":  null,
                     "logical_eval_rows":  null,
                     "max_usd_per_second":  "0.00008792",
                     "max_usd":  "0.63302400"
                 },
                 {
                     "task":  "cpu_audit_aggregate",
                     "group":  "cpu",
                     "device":  "CPU",
                     "max_billed_seconds":  1800,
                     "max_physical_cores":  2,
                     "max_gib":  8,
                     "train_steps":  null,
                     "logical_eval_rows":  null,
                     "max_usd_per_second":  "0.00004396",
                     "max_usd":  "0.07912800"
                 },
                 {
                     "task":  "a100_initial_smoke_setup",
                     "group":  "pilot",
                     "device":  "A100-80GB",
                     "max_billed_seconds":  350,
                     "max_physical_cores":  2,
                     "max_gib":  16,
                     "train_steps":  null,
                     "logical_eval_rows":  null,
                     "max_usd_per_second":  "0.00075572",
                     "max_usd":  "0.26450200"
                 },
                 {
                     "task":  "pilot_train_four_matched_arms",
                     "group":  "pilot",
                     "device":  "A100-80GB",
                     "max_billed_seconds":  690,
                     "max_physical_cores":  2,
                     "max_gib":  16,
                     "train_steps":  276,
                     "logical_eval_rows":  null,
                     "max_usd_per_second":  "0.00075572",
                     "max_usd":  "0.52144680"
                 },
                 {
                     "task":  "pilot_disjoint_dev",
                     "group":  "pilot",
                     "device":  "A100-80GB",
                     "max_billed_seconds":  120,
                     "max_physical_cores":  2,
                     "max_gib":  16,
                     "train_steps":  null,
                     "logical_eval_rows":  4000,
                     "max_usd_per_second":  "0.00075572",
                     "max_usd":  "0.09068640"
                 },
                 {
                     "task":  "resume_same_four_arms",
                     "group":  "arm_comparison",
                     "device":  "A100-80GB",
                     "max_billed_seconds":  3150,
                     "max_physical_cores":  2,
                     "max_gib":  16,
                     "train_steps":  1260,
                     "logical_eval_rows":  null,
                     "max_usd_per_second":  "0.00075572",
                     "max_usd":  "2.38051800"
                 },
                 {
                     "task":  "post_fit_disjoint_calibration_dev",
                     "group":  "arm_comparison",
                     "device":  "A100-80GB",
                     "max_billed_seconds":  360,
                     "max_physical_cores":  2,
                     "max_gib":  16,
                     "train_steps":  null,
                     "logical_eval_rows":  12000,
                     "max_usd_per_second":  "0.00075572",
                     "max_usd":  "0.27205920"
                 },
                 {
                     "task":  "blind_synthetic_core",
                     "group":  "arm_comparison",
                     "device":  "A100-80GB",
                     "max_billed_seconds":  1229,
                     "max_physical_cores":  2,
                     "max_gib":  16,
                     "train_steps":  null,
                     "logical_eval_rows":  40960,
                     "max_usd_per_second":  "0.00075572",
                     "max_usd":  "0.92877988"
                 },
                 {
                     "task":  "blind_pluto_raw",
                     "group":  "pluto",
                     "device":  "A100-80GB",
                     "max_billed_seconds":  1613,
                     "max_physical_cores":  2,
                     "max_gib":  16,
                     "train_steps":  null,
                     "logical_eval_rows":  53760,
                     "max_usd_per_second":  "0.00075572",
                     "max_usd":  "1.21897636"
                 },
                 {
                     "task":  "blind_cttc_raw",
                     "group":  "cttc",
                     "device":  "A100-80GB",
                     "max_billed_seconds":  864,
                     "max_physical_cores":  2,
                     "max_gib":  16,
                     "train_steps":  null,
                     "logical_eval_rows":  28800,
                     "max_usd_per_second":  "0.00075572",
                     "max_usd":  "0.65294208"
                 },
                 {
                     "task":  "blind_real_stress",
                     "group":  "controls",
                     "device":  "A100-80GB",
                     "max_billed_seconds":  248,
                     "max_physical_cores":  2,
                     "max_gib":  16,
                     "train_steps":  null,
                     "logical_eval_rows":  8240,
                     "max_usd_per_second":  "0.00075572",
                     "max_usd":  "0.18741856"
                 },
                 {
                     "task":  "blind_matched_slot_text_controls",
                     "group":  "controls",
                     "device":  "A100-80GB",
                     "max_billed_seconds":  713,
                     "max_physical_cores":  2,
                     "max_gib":  16,
                     "train_steps":  null,
                     "logical_eval_rows":  23744,
                     "max_usd_per_second":  "0.00075572",
                     "max_usd":  "0.53882836"
                 },
                 {
                     "task":  "a100_remaining_setup_teardown",
                     "group":  "arm_comparison",
                     "device":  "A100-80GB",
                     "max_billed_seconds":  750,
                     "max_physical_cores":  2,
                     "max_gib":  16,
                     "train_steps":  null,
                     "logical_eval_rows":  null,
                     "max_usd_per_second":  "0.00075572",
                     "max_usd":  "0.56679000"
                 },
                 {
                     "task":  "explicit_retry_reserve",
                     "group":  "retry",
                     "device":  "A100-80GB",
                     "max_billed_seconds":  900,
                     "max_physical_cores":  2,
                     "max_gib":  16,
                     "train_steps":  null,
                     "logical_eval_rows":  null,
                     "max_usd_per_second":  "0.00075572",
                     "max_usd":  "0.68014800"
                 }
             ],
    "group_usd":  {
                      "cpu":  "0.76770000",
                      "pilot":  "0.87663520",
                      "arm_comparison":  "4.14814708",
                      "pluto":  "1.21897636",
                      "cttc":  "0.65294208",
                      "controls":  "0.72624692",
                      "retry":  "0.68014800"
                  },
    "m4_same_scope_at_12_gib_usd":  "8.44073416",
    "a100_16_gib_rate_delta_vs_m4_per_second":  "0.00000888",
    "a100_seconds_including_retry":  10987,
    "scheduled_usd_without_retry":  "8.39064764",
    "scheduled_usd_with_retry":  "9.07079564",
    "unassigned_inside_cap_usd":  "0.42920436",
    "total_reserve_inside_cap_usd":  "1.10935236",
    "cpu_plus_pilot_usd_before_go":  "1.64433520",
    "cap_usd":  "9.50",
    "historical_credit_remainder_usd":  "12.88",
    "buffer_usd_at_cap":  "3.38",
    "required_workspace_net_spend_guard_usd":  "0.00",
    "live_workspace_or_gpu_execution_verified":  false,
    "spend_this_round_usd":  "0.00"
}
{
    "scope":  "One modeled primary endpoint at first Holm threshold. Full-domain joint power needs independent dev qualification.",
    "trials":  10000,
    "first_holm_alpha":  0.016666666666666666,
    "early_efficacy_alpha":  0.0,
    "blind_critical_z":  2.128045234184984,
    "dev_slope_assumed":  0.6,
    "effect_prior_variance":  4.0,
    "dev_z_go_cutoff_model_only":  0.046796176916532026,
    "proof_bound_type_I_per_domain":  0.016666666666666666,
    "proof_bound_type_I_three_bonferroni":  0.05,
    "fixed_same_scope_usd":  8.39064764,
    "cpu_charge_even_on_screen_failure_usd":  0.7677,
    "gpu_pilot_usd":  0.8766352,
    "mean_sequential_cost_usd_mc":  3.357607456416,
    "mean_sequential_cost_usd_analytic":  3.366569416227849,
    "cost_saving_pct_mc":  59.98392972183015,
    "cost_saving_pct_analytic":  59.87712080556573,
    "strata":  {
                   "structural_null":  {
                                           "trials":  6000,
                                           "drift":  -0.8,
                                           "cpu_pass_assumed":  0.1,
                                           "go_count":  182,
                                           "fixed_reject_count":  5,
                                           "sequential_reject_count":  1,
                                           "go_rate_analytic":  0.02991675690409064,
                                           "fixed_rate_analytic":  0.0017055020813661985,
                                           "sequential_rate_analytic":  5.1023091167653175E-05,
                                           "sequential_rate_mc":  0.00016666666666666666,
                                           "sequential_mc_ci95":  [
                                                                      4.2196257470907119E-06,
                                                                      0.0009282535375926466
                                                                  ],
                                           "cost_usd_mc":  1.0585406053466666,
                                           "cost_usd_analytic":  1.0571913092665226
                                       },
                   "boundary_null":  {
                                         "trials":  1500,
                                         "drift":  0.0,
                                         "cpu_pass_assumed":  0.9,
                                         "go_count":  650,
                                         "fixed_reject_count":  22,
                                         "sequential_reject_count":  11,
                                         "go_rate_analytic":  0.4332040542341478,
                                         "fixed_rate_analytic":  0.016666666666666694,
                                         "sequential_rate_analytic":  0.007220067570569142,
                                         "sequential_rate_mc":  0.007333333333333333,
                                         "sequential_mc_ci95":  [
                                                                    0.0036662916253218207,
                                                                    0.01308339451452098
                                                                ],
                                         "cost_usd_mc":  4.488255665866667,
                                         "cost_usd_analytic":  4.479201580138266
                                     },
                   "working":  {
                                   "trials":  2500,
                                   "drift":  3.9,
                                   "cpu_pass_assumed":  0.99,
                                   "go_count":  2432,
                                   "fixed_reject_count":  2413,
                                   "sequential_reject_count":  2351,
                                   "go_rate_analytic":  0.9791910447878376,
                                   "fixed_rate_analytic":  0.9617989668599535,
                                   "sequential_rate_analytic":  0.9417849352354606,
                                   "sequential_rate_mc":  0.9404,
                                   "sequential_mc_ci95":  [
                                                              0.9303922406793362,
                                                              0.949359849173351
                                                          ],
                                   "cost_usd_mc":  8.196978973312,
                                   "cost_usd_analytic":  8.241497574588784
                               }
               }
}
{
    "scope":  "Algebraic fixtures on CPU. Production causality, release scoring, and actual host Jacobians are pending.",
    "packed_words_replayed":  65536,
    "sentinel_fixture_checks_passed":  5,
    "correlated_innovation_posterior_max_abs_error":  1.1102230246251565E-16,
    "naive_independent_posterior_max_abs_error":  0.06765408352412963,
    "double_counted_old_reference_posterior_max_abs_error":  0.2957682054272823,
    "task_pairwise_group_compositions_passed":  20480,
    "arm2_max_pairwise_residual_logodds":  0.3,
    "host_configuration_shape_checks":  {
                                            "Qwen3.5-4B":  {
                                                               "width":  2560,
                                                               "max_shared_adapter_parameters":  43521,
                                                               "toy_candidate_jacobians":  {
                                                                                               "1":  {
                                                                                                         "centered_rank":  3,
                                                                                                         "sigma_min_over_max":  0.9999999999999988
                                                                                                     },
                                                                                               "2":  {
                                                                                                         "centered_rank":  3,
                                                                                                         "sigma_min_over_max":  0.9999999999999992
                                                                                                     }
                                                                                           }
                                                           },
                                            "Qwen3.5-9B":  {
                                                               "width":  4096,
                                                               "max_shared_adapter_parameters":  69633,
                                                               "toy_candidate_jacobians":  {
                                                                                               "1":  {
                                                                                                         "centered_rank":  3,
                                                                                                         "sigma_min_over_max":  0.9999999999999996
                                                                                                     },
                                                                                               "2":  {
                                                                                                         "centered_rank":  3,
                                                                                                         "sigma_min_over_max":  0.9999999999999998
                                                                                                     }
                                                                                           }
                                                           },
                                            "Qwen3.8-27B":  {
                                                                "width":  5120,
                                                                "max_shared_adapter_parameters":  87041,
                                                                "toy_candidate_jacobians":  {
                                                                                                "1":  {
                                                                                                          "centered_rank":  3,
                                                                                                          "sigma_min_over_max":  0.9999999999999989
                                                                                                      },
                                                                                                "2":  {
                                                                                                          "centered_rank":  3,
                                                                                                          "sigma_min_over_max":  0.9999999999999994
                                                                                                      }
                                                                                            }
                                                            }
                                        },
    "fixture_common_support_ids":  [
                                       "b",
                                       "c"
                                   ]
}
{
    "split_source_ranges":  {
                                "train":  "@{target=System.Object[]; ref=System.Object[]}",
                                "dev":  "@{target=System.Object[]; ref=System.Object[]}",
                                "test":  "@{target=System.Object[]; ref=System.Object[]}"
                            },
    "cyclic_symbol_guard":  4096,
    "counts":  {
                   "train_target":  1440,
                   "train_ref":  576,
                   "dev_target":  864,
                   "dev_ref":  576,
                   "test_target":  1440,
                   "test_ref":  576
               },
    "test_session_loop_clusters":  72,
    "repeat_disjoint_loop_pairs":  36,
    "repeat_candidate_rows":  720,
    "stress_rows":  1080,
    "minimum_acquisition_gap_samples_after_rcv_support_expansion":  16481,
    "source_bit_footprints_pairwise_disjoint_even_plus_odd":  true,
    "refs":  [
                 126,
                 127
             ],
    "maximum_cubic_union_samples":  6
}
```

### Event #30: `item.completed` — `command_execution` (`item_27`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n 'candidate_token|max_len|candidate_ids|gpu_name' research/stage3d_qwen_signed_phase/results/smoke_audit.json | Select-Object -First 35; rg -n 'vram|allocated|MB' research/stage3d_qwen_signed_phase/train_eval_stage3d.py | Select-Object -First 12; rg -n 'near|nll_pareto|16-bit' research/stage4_math_and_design/M5/reply_r1.md"
```

**Aggregated Output:**

```text
9:  "gpu_name": "NVIDIA A100-SXM4-80GB",
110:  "candidate_token_audit": {
218:      "max_len": 9,
309:      "max_len": 6,
476:      "max_len": 16,
247:        model_vram_mb = float(torch.cuda.memory_allocated(host.device)) / (1024.0 ** 2)
250:        model_vram_mb = 0.0
376:                pk = float(torch.cuda.max_memory_allocated(host.device)) / (1024.0 ** 2)
381:                "peak_vram_mb": round(pk, 1),
418:    total_vram_mb = (
428:        if rec and not rec.get("oom", False) and (total_vram_mb == 0.0 or rec.get("peak_vram_mb", 0.0) < 0.80 * total_vram_mb):
441:        "total_vram_mb": round(total_vram_mb, 1),
444:        "model_vram_mb_after_load": round(model_vram_mb, 1),
800:    peak_vram_mb = (
801:        float(torch.cuda.max_memory_allocated(host.device)) / (1024.0 ** 2)
825:        "peak_vram_mb": round(peak_vram_mb, 1),
3:Carry a **16-bit canonical candidate cache with the exact 4-bit s17 prefix** into Round 2. Keep 12 bits as a cost ablation and continuous matched complex evidence plus residual energy as the physical oracle. This preserves the old reader and gives event and acquisition heads access to calibrated phase, radial evidence, and residual quality. The 24-bit format is specified for future comparison.
24:The 16-bit cache recovers **0.312 nat** of joint event/location NLL versus s17, with a pointwise paired 95% interval of **0.302–0.321**, and leaves **0.074 nat** to the continuous oracle. Conditional purely smooth-versus-jump NLL improves by **0.0104 nat** pooled and is effectively tied under selective fading. The actual s17 network merges two strict interior points whose four-label posteriors differ by **0.480 total variation**. The G1 Stage 3D result remains a narrow near-sufficiency case. Its physical cache gap is **0.0121 nat**, and the host's additional readout gap is **0.5041 nat**. [Stage 3D report](../../stage3d_qwen_signed_phase/REPORT.md)
26:On the 2,048 ambiguous selective-fade mothers, **max EIG gives 90.63% accuracy / 0.2369 NLL**, random gives 71.63% / 0.5141, and nearest gives 70.21% / 0.5714. The committed forecasts imply 0.4542 nat expected gain and yield 0.4395 nat realized gain. Every max-EIG selection in that profile is time 4. A fixed time-4 policy would tie there. The synthetic run tests the value of reference geometry and leaves the incremental gain from recording-specific adaptation for a stronger design. Pluto/CTTC, host transfer and LEMMO performance remain live evaluation questions. [Full results](results_m5.json)
```

### Event #31: `item.completed` — `file_change` (`item_28`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M6\\verify_m6.py",
    "kind": "update"
  }
]
```

### Event #32: `item.completed` — `file_change` (`item_29`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M6\\.round2-body.md",
    "kind": "add"
  }
]
```

### Event #33: `item.completed` — `file_change` (`item_30`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M6\\.round2-body.md",
    "kind": "update"
  }
]
```

### Event #34: `item.completed` — `command_execution` (`item_31`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$dir=(Resolve-Path -LiteralPath '"'research/stage4_math_and_design/M6').Path; "'$target=Join-Path $dir '"'THEOREM_M6.md'; "'$body=Join-Path $dir '"'.round2-body.md'; "'$old=[IO.File]::ReadAllText($target); $boundary=$old.IndexOf('"'## 4. Tier 2'); if ("'$boundary -lt 0) {throw '"'Original theorem section boundary missing'}; "'$head=$old.Substring(0,$boundary); $start=$head.IndexOf('"'## 1.'); if ("'$start -lt 0) {throw '"'Original theorem section boundary missing'}; "'$intro=@'"'
# M6 Round 2 • A CPU-gated, all-A100 Stage 4 execution lock

**Decision.** Reuse the Stage 3D score archive for diagnosis, build both cache views on CPU, and authorize at most a disjoint-dev 69-step pilot per arm before any full four-arm fit. All Stage 4 GPU work uses A100-80GB within a **"'$9.50** gross cap, leaving **$3.38** against the historical $12.88. §§1–3 retain the Round 1 mathematical foundations. §§4–6 are the operative Round 2 schedule and gates. Round 2 used local CPU only.

'"'@; [IO.File]::WriteAllText("'$target,$intro.TrimEnd()+"`n`n"+$head.Substring($start).TrimEnd()+"`n`n"+[IO.File]::ReadAllText($body).TrimEnd()+"`n"); Remove-Item -LiteralPath $body; rg -n '"'"'^#'"' "'$target'
```

**Aggregated Output:**

```text
1:# M6 Round 2 • A CPU-gated, all-A100 Stage 4 execution lock
5:## 1. The exact population loss ladder
38:### What the Stage 3D archive identifies
66:## 2. Tier 0 • analytic rejection before any host work
68:### 2.1 Bayes-risk brackets for each evidence state
80:### 2.2 Exact group action and reference algebra
88:## 3. Tier 1 • CPU geometry and the frozen host
94:### 3.1 Rank, curvature and optimization certificates
102:### 3.2 Geometry of $W_{\rm out}$ and vocabulary readout
116:## 4. Round 2 execution lock
120:### 4.1 Four arms and host contribution
143:### 4.2 Causal shard, dual cache and RefSufficientPool
159:### 4.3 Dev-disjoint futility and final inference
178:### 4.4 The sole Stage 4 Modal ledger
212:## 5. Master Stage 4–6 execution DAG
251:## 6. Cross-host and fair LEMMO execution law
263:## Replay and scope
```

### Event #35: `item.completed` — `file_change` (`item_32`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M6\\verify_m6.py",
    "kind": "update"
  }
]
```

### Event #36: `item.completed` — `file_change` (`item_33`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M6\\THEOREM_M6.md",
    "kind": "update"
  }
]
```

### Event #37: `item.completed` — `agent_message` (`item_34`)

The ledger now totals $8.39064764 for scheduled work and $9.07079564 with the explicit retry reserve, leaving $0.42920436 inside the $9.50 cap. The updated simulation uses a dev-only futility decision and a separate blind test, so the 69-step pilot spends zero efficacy alpha.

### Event #38: `item.completed` — `file_change` (`item_35`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M6\\reply_r2.md",
    "kind": "add"
  }
]
```

### Event #39: `item.completed` — `file_change` (`item_36`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M6\\THEOREM_M6.md",
    "kind": "update"
  }
]
```

### Event #40: `item.completed` — `command_execution` (`item_37`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M6/verify_m6.py'
```

**Aggregated Output:**

```text
VERIFIED results_m6.json: parity max |diff|=3.56e-16, two 10,000-trial designs, A100 cap $9.50
```

### Event #41: `item.completed` — `command_execution` (`item_38`)

- Status: `completed`, Exit Code: `0`

**Command:**

````powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-ChildItem -LiteralPath research/stage4_math_and_design/M6 -Force | Select-Object -Property Name,Length; rg -n 'L4|A10G|173/960|2,707|"'^[$][$]|''^```|135,|reply_r2'"' research/stage4_math_and_design/M6/THEOREM_M6.md; git diff --name-only"
````

**Aggregated Output:**

````text

9:$$
13:$$
19:$$
25:$$
29:$$
34:$$
106:$$
108:$$
137:$$
139:$$
147:$$
149:$$
216:```mermaid
232:```
267:```powershell
269:```
271:Local Round 2 verifies all five Q3 cells and seven Q2 arms (102 accuracy/NLL comparisons, maximum discrepancy $3.56\times10^{-16}$), both 10,000-trial sequential setups, the Decimal ledger, exhaustive packet layouts, sentinel and innovation fixtures, task compositions, adapter sizes and the DAG order. It used Python 3.12.7 with the pinned NumPy and SciPy. Pending production receipts include causal alignment, the pinned-host JVPs, joint power, live throughput and the account guard. Round 2 used local CPU and spent **\$0.00** in Modal. [Verifier](verify_m6.py), [Results](results_m6.json), [Execution handoff](reply_r2.md)
Name            Length
----            ------
reply_r1.md       4925
reply_r2.md       3780
results_m6.json  42305
THEOREM_M6.md    58973
verify_m6.py     37371


````

### Event #42: `item.completed` — `file_change` (`item_39`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M6\\verify_m6.py",
    "kind": "update"
  }
]
```

### Event #43: `item.completed` — `file_change` (`item_40`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M6\\THEOREM_M6.md",
    "kind": "update"
  }
]
```

### Event #44: `item.completed` — `command_execution` (`item_41`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'uv run --offline --no-project --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M6/verify_m6.py'
```

**Aggregated Output:**

```text
VERIFIED results_m6.json: parity max |diff|=3.56e-16, two 10,000-trial designs, A100 cap $9.50
```

### Event #45: `item.completed` — `command_execution` (`item_43`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'git status --short --untracked-files=all | Where-Object { $_ -like '"'*research/stage4_math_and_design/M6/*' }; Get-ChildItem -Force -LiteralPath research/stage4_math_and_design/M6 | Select-Object -ExpandProperty Name; rg -n '"'^#{1,3} |''^'"\\| \\*\\*Gross project|0\\.0467962|signed-slot' research/stage4_math_and_design/M6/THEOREM_M6.md | Select-Object -Last 18"
```

**Aggregated Output:**

```text
?? research/stage4_math_and_design/M6/THEOREM_M6.md
?? research/stage4_math_and_design/M6/reply_r1.md
?? research/stage4_math_and_design/M6/reply_r2.md
?? research/stage4_math_and_design/M6/results_m6.json
?? research/stage4_math_and_design/M6/verify_m6.py
reply_r1.md
reply_r2.md
results_m6.json
THEOREM_M6.md
verify_m6.py
1:# M6 Round 2 • A CPU-gated, all-A100 Stage 4 execution lock
5:## 1. The exact population loss ladder
38:### What the Stage 3D archive identifies
66:## 2. Tier 0 • analytic rejection before any host work
68:### 2.1 Bayes-risk brackets for each evidence state
80:### 2.2 Exact group action and reference algebra
88:## 3. Tier 1 • CPU geometry and the frozen host
94:### 3.1 Rank, curvature and optimization certificates
102:### 3.2 Geometry of $W_{\rm out}$ and vocabulary readout
116:## 4. Round 2 execution lock
120:### 4.1 Four arms and host contribution
143:### 4.2 Causal shard, dual cache and RefSufficientPool
159:### 4.3 Dev-disjoint futility and final inference
182:### 4.4 The sole Stage 4 Modal ledger
206:| **Gross project cap** | **External buffer 3.38 against reported 12.88** | | **9.50** |
216:## 5. Master Stage 4–6 execution DAG
255:## 6. Cross-host and fair LEMMO execution law
267:## Replay and scope
```

### Event #46: `item.completed` — `command_execution` (`item_42`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "uv run --offline --no-project --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B -c \"import ast,json,re; from pathlib import Path; from decimal import Decimal as D; p=Path('research/stage4_math_and_design/M6'); d=json.loads((p/'results_m6.json').read_text(encoding='utf-8')); q=d['modal_budget']; assert sum(D(v['max_usd']) for v in q['rows'])+D(q['unassigned_inside_cap_usd'])==D('9.50'); assert D(q['historical_credit_remainder_usd'])-D(q['cap_usd'])==D('3.38'); assert len(d['execution_graph']['parents'])==13; assert d['execution_graph']['numeric_lock']['synthetic_primary_endpoint']=='Q3 repeat pre-reveal'; assert d['summary_parity']['scalar_acc_nll_checks']==102; assert not q['live_workspace_or_gpu_execution_verified']; ast.parse((p/'verify_m6.py').read_text(encoding='utf-8')); docs=[(p/n).read_text(encoding='utf-8') for n in ['THEOREM_M6.md','reply_r2.md']]; assert all(not any(ord(c)<32 and c not in '\\n\\t' for c in s) for s in docs); links=[u.split('#')[0] for s in docs for u in re.findall(r'\\]\\((["'^)]+)'"\\)',s) if '://' not in u]; assert all((p/u).exists() for u in links), links; assert docs[0].splitlines().count(chr(36)*2)==12; assert sum(l.startswith(chr(96)*3) for l in docs[0].splitlines())==4; print('Checked allocations, scope, 13 DAG nodes, relative links and Markdown fences')\""
```

**Aggregated Output:**

```text
Checked allocations, scope, 13 DAG nodes, relative links and Markdown fences
```

### Event #47: `item.completed` — `agent_message` (`item_44`)

**M6 Round 2 is complete.** The Stage 4 plan uses A100-80GB for every GPU job and one fixed seed across four matched arms. Each arm starts with **69 of 384 steps**. The development pilot spends zero efficacy alpha. A pass releases the remaining training, and blind inference is conditional on the frozen checkpoints.

| Stage 4 allocation | USD |
|---|---:|
| CPU causal frontend and dual-cache export | $0.7677 |
| A100 pilot and pilot-dev scoring | $0.8766 |
| Four-arm completion, calibration, and synthetic scoring | $4.1481 |
| Pluto, 2,240 natural frames | $1.2190 |
| CTTC, 1,440 natural packets including held-out `f708_snr10` | $0.6529 |
| Stress tests and slot/text controls | $0.7262 |
| Reserve inside the cap | $1.1094 |
| **Gross cap** | **$9.50** |

The cap leaves **$3.38** against the historical $12.88 credit balance. Launch still requires live credit and zero-net-spend receipts, causal data checks, actual-host Jacobians for both Host-Pure and Hybrid, and an A100 memory and throughput smoke test. [Modal pricing](https://modal.com/pricing) · [Modal spend limits](https://modal.com/docs/guide/budgets)

The pinned `uv` run passed **102 Stage 3D parity checks**, with a maximum discrepancy of **3.56×10⁻¹⁶**. The Round 2 10,000-trial model estimated **59.98% cost savings**, **94.04% working single-endpoint power**, and **0.733% boundary-null promotion** at the 1.667% first-Holm threshold. Production joint power remains unverified. **No cloud jobs ran.**

[Execution handoff](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/reply_r2.md>) · [Numerical gates, Stage 4–6 DAG, and LEMMO comparison protocol](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/THEOREM_M6.md>) · [Verifier](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/verify_m6.py>) · [Results](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/results_m6.json>)



### Event #48: `turn.completed`

- Usage: `{"input_tokens": 11357446, "cached_input_tokens": 11084544, "cache_write_input_tokens": 0, "output_tokens": 87171, "reasoning_output_tokens": 18310}`


## Round 2 — Final Output (`reply_r2.md`)

```markdown
**M6 Round 2 is complete.** The Stage 4 plan uses A100-80GB for every GPU job and one fixed seed across four matched arms. Each arm starts with **69 of 384 steps**. The development pilot spends zero efficacy alpha. A pass releases the remaining training, and blind inference is conditional on the frozen checkpoints.

| Stage 4 allocation | USD |
|---|---:|
| CPU causal frontend and dual-cache export | $0.7677 |
| A100 pilot and pilot-dev scoring | $0.8766 |
| Four-arm completion, calibration, and synthetic scoring | $4.1481 |
| Pluto, 2,240 natural frames | $1.2190 |
| CTTC, 1,440 natural packets including held-out `f708_snr10` | $0.6529 |
| Stress tests and slot/text controls | $0.7262 |
| Reserve inside the cap | $1.1094 |
| **Gross cap** | **$9.50** |

The cap leaves **$3.38** against the historical $12.88 credit balance. Launch still requires live credit and zero-net-spend receipts, causal data checks, actual-host Jacobians for both Host-Pure and Hybrid, and an A100 memory and throughput smoke test. [Modal pricing](https://modal.com/pricing) · [Modal spend limits](https://modal.com/docs/guide/budgets)

The pinned `uv` run passed **102 Stage 3D parity checks**, with a maximum discrepancy of **3.56×10⁻¹⁶**. The Round 2 10,000-trial model estimated **59.98% cost savings**, **94.04% working single-endpoint power**, and **0.733% boundary-null promotion** at the 1.667% first-Holm threshold. Production joint power remains unverified. **No cloud jobs ran.**

[Execution handoff](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/reply_r2.md>) · [Numerical gates, Stage 4–6 DAG, and LEMMO comparison protocol](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/THEOREM_M6.md>) · [Verifier](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/verify_m6.py>) · [Results](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M6/results_m6.json>)

```
