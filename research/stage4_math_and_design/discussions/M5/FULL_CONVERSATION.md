# Full Verbatim Multi-Round Conversation — Track `M5`

- **CLI Runtime**: Codex CLI (`codex exec`)
- **Session / Thread ID**: `01a2194c-3b55-7153-bc9f-570b6f8c3c3f`
- **Round 1 Events**: `99` raw JSONL events (`round1_events.jsonl`)
- **Round 2 Events**: `120` raw JSONL events (`round2_events.jsonl`)
- **Completeness Guarantee**: All user prompts (`prompt_r1.md`, `prompt_r2.md`), all completed reasoning/tool/command events, and final responses (`reply_r1.md`, `reply_r2.md`) are included verbatim below without truncation or paraphrasing.

---

## Round 1 — Initial Mathematical Brief (`prompt_r1.md`)

````markdown
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
````


## Round 1 — Execution & Reasoning Trace (`round1_events.jsonl`)

### Event #1: `thread.started` (thread_id=`01a2194c-3b55-7153-bc9f-570b6f8c3c3f`)

### Event #2: `turn.started`

### Event #3: `item.completed` — `agent_message` (`item_0`)

I’ll inspect the research plans, encoder, and prior results first. Then I’ll derive the conditions for exact sufficiency and EIG, and run a seeded CPU verification inside the M5 directory.

### Event #4: `item.completed` — `command_execution` (`item_1`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n -i 'EM-FloatFormer|stage4_math|matched_cache|s17|LEMMO' 'C:/Users/李则徐/.codex/memories/MEMORY.md'"
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
```

### Event #5: `item.completed` — `command_execution` (`item_2`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "Get-Location
Get-ChildItem -Force -Name
rg --files -g AGENTS.md -g '*M5*' -g '*LONG_TERM*' -g '*MULTIMODAL*'
git status --short
Get-Command uv | Select-Object -ExpandProperty Source"
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
 

docs\MULTIMODAL_INTEGRATION_PLAN.md
docs\LONG_TERM_RESEARCH_PLAN.md
?? research/stage4_math_and_design/
Path                                                             
----                                                             
C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4
.github
docs
exploratory
p10
p11
p12
p1b_results
p1c_results
p1d_results
p1e_results
p1e_train
p2a_results
p2b_results
p2c_results
p2d_results
p2e_results
p2_supplement
p3a2_results
p3b_results
p3c_results
p4
p5
p5b
p6
p6_consistency
p6_followup
p6_softinfo
p7
p8
p8c
p9
publication
research
results
tests
.git
.gitattributes
.gitignore
LICENSE
MANIFEST_SHA256.json
ofdm_baseline.py
P1B_REPORT.md
p1b_representation.py
p1c_architecture.py
p1c_make_report.py
P1C_REPORT.md
p1c_verify_checkpoints.py
p1d_make_report.py
P1D_REPORT.md
p1d_robustness.py
p1e_make_report.py
P1E_PROTOCOL.md
p1e_replication.py
P1E_REPORT.md
P2A_REPORT.md
P2B_REPORT.md
P2C_REPORT.md
P2D_REPORT.md
P2E_FREEZE.json
P2E_REPORT.md
p2_finalize.py
P2_FINAL_AUDIT.md
P2_FINAL_SCOPE_ADDENDUM.md
p2_models.py
p2_numeric.py
p2_package.py
P2_PROTOCOL.md
p2_run.py
P2_SCOPE_ALIGNMENT_AUDIT.md
p2_supplement.py
p2_supplement_finalize.py
P2_SUPPLEMENT_MANIFEST_SHA256.json
P2_SUPPLEMENT_PROTOCOL.md
P2_SUPPLEMENT_REPORT.md
p2_symmetry.py
P3-0_A_source_and_results.zip
P3-A2_source_and_results.zip
P3B_evidence_2026-11-15.zip
p3b_independent_audit.py
p3b_make_report.py
p3b_package.py
p3b_privileged_reference.py
P3B_PROTOCOL.md
p3b_receivers.py
P3B_REPORT.md
P3B_SCOPE_ADDENDUM.md
p3b_time_link.py
p3b_timing_extended.py
p3b_train_eval.py
P3C_evidence_2026-11-15.zip
p3c_independent_audit.py
p3c_make_report.py
p3c_precision.py
P3C_PROTOCOL.md
P3C_README.md
P3C_REPORT.md
P3C_SCOPE_ADDENDUM.md
P3_0_A_PROTOCOL.md
P3_0_A_REPORT.md
P3_A2_PUBLIC_SHA256_MANIFEST.json
P3_A2_README.md
p3_a_mechanism.py
p3_a_mechanism_audit.py
p3_a_mechanism_make_report.py
P3_A_MECHANISM_PROTOCOL.md
P3_A_MECHANISM_REPORT.md
P3_A_MECHANISM_SCOPE_ADDENDUM.md
p3_models.py
PROJECT.md
README.md
requirements.txt
requirements_p3c.txt
TEST_INFRA.md
test_p1d_invariants.py
test_p1e_integrity.py
test_p2_integrity.py
test_p2_supplement_independent.py
test_p3b_link.py
test_p3c_precision.py
test_p3_a_mechanism.py
TEST_READY.md
C:\qazwsx\Scripts\uv.exe


```

### Event #6: `item.completed` — `command_execution` (`item_3`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '"'^#|s17|0'"\\.504|12\\.2|Pluto|CTTC' research/stage3d_qwen_signed_phase/REPORT.md research/stage3d_qwen_signed_phase/data_stage3d.py research/matched_cache research/stage3c -g '*.md' -g '*.py' -g '"'!*.ipynb'"'"
```

**Aggregated Output:**

```text
research/stage3d_qwen_signed_phase/data_stage3d.py:42:# Disjoint seed namespaces (strictly isolated from Stage 3 s3_train/s3_dev/s3_test and Stage 3C work_e 316027)
research/stage3d_qwen_signed_phase/data_stage3d.py:146:def load_s17_encoder(repo_root: Optional[Path] = None):
research/stage3d_qwen_signed_phase/data_stage3d.py:147:    """Load frozen equivariant_b8_s17 encoder and enforce exact SHA-256 fingerprint."""
research/stage3d_qwen_signed_phase/data_stage3d.py:150:    weight_path = matched_dir / "weights" / "equivariant_b8_s17.npz"
research/stage3d_qwen_signed_phase/data_stage3d.py:153:        raise RuntimeError(f"s17 SHA-256 mismatch: {digest} != {EQUIVARIANT_B8_S17_SHA256}")
research/stage3d_qwen_signed_phase/REPORT.md:1:# EM-FloatFormer Stage 3D 实验报告：冻结 Qwen3.5-4B 与 s17 缓存下的有符号频偏显式相位向量盲测
research/stage3d_qwen_signed_phase/REPORT.md:8:## 首页执行摘要与双门槛判定
research/stage3d_qwen_signed_phase/REPORT.md:10:本轮实验回答单一主问题：**在完全冻结 `Qwen/Qwen3.5-4B` 与 `equivariant_b8_s17` 离散波形编码器、两臂均从中文题面获得完全相同的有符号频偏元数据且训练数据与更新预算逐项相同的条件下，仅将连接层中 6 个公开数值槽由全零（`zero` 对照臂）替换为对应时刻 $t \in \{h, h+\ell, t_p\}$ 的显式相位向量 $[\cos(\omega t), \sin(\omega t)]$（`phase` 相位臂），能否提高对未见频偏的 Q3-repeat 相位扇区预测？**
research/stage3d_qwen_signed_phase/REPORT.md:12:### 主检验格 `G1` 汇总（`|ω| = 0.40`, `h = 2`, `ℓ = 2`, `t_p = 4.0`, `|ω|(h+ℓ) = 1.60 rad`, `σ²t = 0.04`）
research/stage3d_qwen_signed_phase/REPORT.md:20:| 同期同母波形 `s17` CPU 物理读出 | **64.62%** (`0.64617`) | **0.9411** (`0.94113`) | `0.5098` | `0.0009` | 真实连接层相对 CPU 物理读出差距：`phase −40.82 pp / -0.5041 nats`；`zero −45.21 pp / -0.5398 nats` |
research/stage3d_qwen_signed_phase/REPORT.md:24:### 三个预登记连接层种子的各自估计与离散程度（`G1`）
research/stage3d_qwen_signed_phase/REPORT.md:45:## 1. 宿主与编码器锁定、Token 核对、显存故障排查与算力账单
research/stage3d_qwen_signed_phase/REPORT.md:47:### 1.1 冻结组件、词表与 Chat Template 审计（[smoke_audit.json](results/smoke_audit.json)、[PROTOCOL_LOCK.json](PROTOCOL_LOCK.json)）
research/stage3d_qwen_signed_phase/REPORT.md:53:| **波形编码器** | `equivariant_b8_s17`（16 个离散态 / 每窗 4 bit，文件 SHA-256：`e67d9fbafb90791ab19549de7789798d818286d7f52d7ac566cc39df71e78356`） |
research/stage3d_qwen_signed_phase/REPORT.md:60:#### 12 个中文候选串逐一 Token 审计（均非空、边界零合并）
research/stage3d_qwen_signed_phase/REPORT.md:68:### 1.2 工程关键排查：Qwen3.5 混合线性注意力引用环与 BF16 批并行数值一致性
research/stage3d_qwen_signed_phase/REPORT.md:78:### 1.3 Modal 免费额度算力账单（全程零付费）
research/stage3d_qwen_signed_phase/REPORT.md:89:## 2. 两臂对照设置、6 次训练过程与样本量盲前锁定
research/stage3d_qwen_signed_phase/REPORT.md:91:### 2.1 六个数值槽设计与零泄露契约
research/stage3d_qwen_signed_phase/REPORT.md:100:### 2.2 6 次连接层训练收敛与开发集标量温度锁定（[training_summary.json](results/training_summary.json)）
research/stage3d_qwen_signed_phase/REPORT.md:113:### 2.3 揭盲前独立开发先导方差检查与 `B_locked` 锁定（[dev_pilot_lock.json](results/dev_pilot_lock.json)）
research/stage3d_qwen_signed_phase/REPORT.md:122:## 3. 主检验与 2×2 探索格结果：累计转角 $\pi/2$ 边界分离机制
research/stage3d_qwen_signed_phase/REPORT.md:126:| 盲测格 | 用途 | `|ω|` | `(h, ℓ, t_p)` | 累计转角 `|ω|(h+ℓ)` | 累计扩散 `σ²t` (`t=h+ℓ`) | 区组 / 母波形 | `phase` Acc / NLL | `zero` Acc / NLL | 配对准确率增益 (`95% t-CI` / `Boot`) | 配对 NLL 改善 (`95% t-CI` / `Boot`) | 同母波形 `s17` 物理 / `full_iq` |
research/stage3d_qwen_signed_phase/REPORT.md:141:## 4. 机制诊断 I：Work F 频偏符号干预与单模态矛盾探针
research/stage3d_qwen_signed_phase/REPORT.md:143:### 4.1 固定缓存四格干预汇总（主格 `±0.40` 与历史配置 `±0.35`）
research/stage3d_qwen_signed_phase/REPORT.md:154:- **历史 `±0.35` 正世界 lookup 锚点验证**：Stage 3B [c4_s17_lookup.json](../stage3b_math/A/c4_s17_lookup.json) 在 `H_c4_035` `+0.35` 正世界（16 区块 = `3,072` 条）实测 **`63.18% / 0.9651`**，与本轮独立重新校准物理参考在 `+0.35` 正世界下逐位相等（`63.18359375% / 0.9662`）。
research/stage3d_qwen_signed_phase/REPORT.md:157:### 4.2 具体真实单缓存样本原始分数检验（`G1`, `seed = 202611`, `mother_idx = 3153`, `cache_code = 100`, `z1 = 6, z2 = 4`）
research/stage3d_qwen_signed_phase/REPORT.md:168:### 4.3 `phase` 臂单模态矛盾探针（`G1`, `|ω| = 0.40`）
research/stage3d_qwen_signed_phase/REPORT.md:177:## 5. 机制诊断 II：Work G 旋转等变与离线外部群平均分离报告
research/stage3d_qwen_signed_phase/REPORT.md:195:## 6. 机制诊断 III：Work A 迟到导频 `C4 (ω = +0.35, t_p = 4)` 七臂对比与旧工况回放
research/stage3d_qwen_signed_phase/REPORT.md:197:### 6.1 Q2 真实输入表示与连续参考边界说明
research/stage3d_qwen_signed_phase/REPORT.md:199:- **输给宿主的导频表示**：保留 Stage 3 的 12 维连续特征向量 `build_ref_feature(r)`（实部与虚部匹配投影及其归一/离散辅助统计，通过 `ref_mlp` 映射为 4 个 `num_ref_tokens` 连续隐层 token 插入题面对应 `<REF>` 槽位），**未对导频再做 `s17` 4-bit 离散量化**；同时仅对 `(z1, z2)` 两主数据窗保留 `s17` 4-bit 量化。因此，理论连续参考（如 `full_iq`）比 `q2_reader.read` 离散核能看到两窗与导频间未经 4-bit 离散截断的精确连续幅相。
research/stage3d_qwen_signed_phase/REPORT.md:202:### 6.2 Work A 七臂与物理参照表（512 条母波形，3 种子汇总 vs CPU `q2_reader.read`）
research/stage3d_qwen_signed_phase/REPORT.md:216:### 6.3 旧工况 `C1 / C2 / C3` 回放未遗忘验证（`s3_test` × 2 区块 × 3 旋转 = 384 条 × 3 种子）
research/stage3d_qwen_signed_phase/REPORT.md:218:| 条件 | 任务 | `phase` Acc / NLL | `zero` Acc / NLL | 同期 `s17` 物理读出 Acc / NLL |
research/stage3d_qwen_signed_phase/REPORT.md:228:## 7. 历史 Ministral-3-3B 诊断状态、范围边界、复算与核心后续观测
research/stage3d_qwen_signed_phase/REPORT.md:230:### 7.1 实际完成与未完成项目透明清单
research/stage3d_qwen_signed_phase/REPORT.md:235:### 7.2 适用范围与限制声明
research/stage3d_qwen_signed_phase/REPORT.md:239:2. **冻结组件版本**：`Qwen/Qwen3.5-4B`（`851bf6e8...`）、`equivariant_b8_s17`（`e67d9fba...`）与本轮 6 组连接层（每组仅 `2,184,964` 可训练参数）。
research/stage3d_qwen_signed_phase/REPORT.md:242:### 7.3 独立一行复算命令（已于干净临时目录验证通过）
research/stage3d_qwen_signed_phase/REPORT.md:248:# 实测于 %TEMP%\stage3d_clean_verify 输出：VERIFIED OK: max_abs_diff=0.000e+00，退出码 0
research/stage3d_qwen_signed_phase/REPORT.md:258:### 7.4 本轮最值得继续追查的核心观测
research/stage3d_qwen_signed_phase/REPORT.md:266:## 交付清单与指纹锚点
research/stage3c\A\q2_reader.py:9:KERNEL_PATH = Path(__file__).with_name("q2_s17_kernel.json")
research/stage3c\crossreviews\A.md:1:# A 组独立复核：迟到导频下的 Q2
research/stage3c\crossreviews\A.md:5:## 复跑
research/stage3c\crossreviews\A.md:15:cmp "$out/kernel.json" "$A_DIR/q2_s17_kernel.json"
research/stage3c\crossreviews\A.md:21:## 参考的概率假设
research/stage3c\B\B_features.md:1:# Signed carrier-offset features for variable window times
research/stage3c\B\B_features.md:7:## Exact metadata theorem
research/stage3c\B\B_features.md:20:### Task-dependent feature selection
research/stage3c\B\B_features.md:33:## Three exact scope checks
research/stage3c\B\B_features.md:39:## Uncertain or absent offset
research/stage3c\A\Q2.md:1:# Recovering the first transmitted QPSK symbol after a late pilot
research/stage3c\A\Q2.md:3:The receiver retains two 4-bit indices from the frozen equivariant_b8_s17 encoder. Q2 asks for the first transmitted label $X_1$. At the C4 frequency offset, the second data window provides a mathematically identifiable source of information about the later reference. Incorporating it makes little difference to average Q2 accuracy in this condition. Receiving a genuinely new pilot sample at the same time makes a large difference.
research/stage3c\A\Q2.md:5:## The conditional reader
research/stage3c\A\Q2.md:37:## Actual frozen-checkpoint computation
research/stage3c\A\Q2.md:52:### Reproduction and use
research/stage3c\A\Q2.md:58:OPENBLAS_NUM_THREADS=1 python check_q2.py --repo "$(git rev-parse --show-toplevel)" --export-kernel /tmp/q2_s17_kernel.json
research/stage3c\crossreviews\B.md:1:# B 组独立复核：有符号频偏与扩散
research/stage3c\E\REPORT.md:1:# Prospective sample size for the next host campaign
research/stage3c\E\REPORT.md:7:## One primary comparison
research/stage3c\E\REPORT.md:15:## What a sample buys
research/stage3c\E\REPORT.md:29:## Generalization and allocation
research/stage3c\E\REPORT.md:44:### Optional predeclared invariance reliability claim
research/stage3c\E\REPORT.md:48:## Lock and reporting
research/stage3c\crossreviews\C.md:1:# C 组独立复核：后验教师何时改善训练目标
research/stage3c\crossreviews\D.md:1:# D 组交叉复核：固定 C4 分类表的总体判决证书
research/stage3c\crossreviews\D.md:3:**结论。** 在发布时冻结的 s17 编码器、已知 C4 参数和 8 比特观测权限下，确认试验支持 Q1 与无导频重发 Q3 的已发布首选标签在每个正概率缓存对上都具有最大条件概率。这是针对该模拟分布的联合频率学检验。后验概率本身、对数损失、Q2 和宿主模型不在证书的覆盖范围内。
research/stage3c\crossreviews\D.md:5:## 全量重跑
research/stage3c\crossreviews\D.md:24:## 核对交并检验
research/stage3c\crossreviews\D.md:32:## 从 16 轨道到 256 缓存对
research/stage3c\F\F_signed_host_probe.md:1:# Can a frozen host use the sign of a carrier offset?
research/stage3c\F\F_signed_host_probe.md:5:## One physical intervention and one input intervention
research/stage3c\F\F_signed_host_probe.md:13:## Direction, magnitude and symmetry on common cache support
research/stage3c\F\F_signed_host_probe.md:15:Independently estimate the two reference tables R_t(z) on 65,536 calibration mothers per sign using the frozen s17 encoder. Pool certified quarter-turn orbits and add 0.5 pseudocount per bin. No reflection symmetry is assumed. A different item in the cache can represent the conjugate state: quarter-turn equivariance by itself does not imply a sign-reflection symmetry. Both signs are fitted from their own actual observations. This reference estimates a conditional distribution and has Monte Carlo error.
research/stage3c\F\F_signed_host_probe.md:25:## Executed controls
research/stage3c\F\F_signed_host_probe.md:41:## Running an actual host
research/stage3c\C\C_distillation.md:1:# C. When should a host connector learn from a numerical posterior?
research/stage3c\C\C_distillation.md:3:Baseline: [Stage 3B, cbf3d2c0](https://github.com/20171806011/EM-FloatFormer/tree/cbf3d2c03f773731fdf1b0d695a63ec57cb62ca0/research/stage3b_math). The [inherited candidate loss](https://github.com/20171806011/EM-FloatFormer/blob/cbf3d2c03f773731fdf1b0d695a63ec57cb62ca0/research/stage3_ministral_cache/train_connector.py#L465-L482) uses the true class attached to each simulated mother. This study changes the training target only. The exported 3B teacher covers the fixed C4, s17, pre-pilot Q1 and repeat-Q3 tasks. Other offsets, SNRs and post-pilot questions require their own matched teacher or hard targets.
research/stage3c\C\C_distillation.md:5:## Exact teacher
research/stage3c\C\C_distillation.md:15:## Approximate teacher and an optimal finite-cell tradeoff
research/stage3c\C\C_distillation.md:29:## Calibration, classification and a failure case
research/stage3c\C\C_distillation.md:46:## Paired connector trial and estimands
research/matched_cache\physical.py:13:# Rotating positive-real, positive-imaginary, negative-real, negative-imaginary
research/matched_cache\physical.py:14:# QPSK by pi/4 gives coordinate signs (++), (-+), (--), (+-).
research/stage3c\crossreviews\F.md:1:# F 组交叉复核：固定缓存后的频偏符号干预
research/stage3c\crossreviews\F.md:5:复跑使用提交 cbf3d2c0 继承的生成器和冻结 s17 权重。下列等效命令从冻结提交的仓库根目录运行。
research/matched_cache\README.md:1:# 同预算下，四相信号的旋转关系能否保留到下一次提问？
research/matched_cache\README.md:5:## 怎样公平地问两次
research/matched_cache\README.md:15:## 是否同时满足三项要求
research/matched_cache\README.md:32:## 看一条母波形的四项概率
research/matched_cache\README.md:44:在同一母本上，普通编码器原本给相对角 90° 的概率为 0.903；两窗一起转后降至 0.732。它两次都选 90°。这给出了单看正确率会漏掉的失败样例。[普通模型逐项记录](results/test/plain_b8_s17.json) · [带约束模型逐项记录](results/test/equivariant_b8_s17.json)
research/matched_cache\README.md:46:## 哪些性质来自哪一层
research/matched_cache\README.md:52:## 复现
research/stage3c\A\check_q2.py:120:    weights = repo/'research/matched_cache/weights/equivariant_b8_s17.npz'
research/stage3c\A\check_q2.py:129:    kernel = dict(encoder='equivariant_b8_s17', checkpoint_sha256=CHECKPOINT_SHA,
research/stage3c\A\check_q2.py:211:    return dict(condition='C4_cfo_ramp',encoder='equivariant_b8_s17',seed=SEED,
research/stage3c\G\README.md:1:# One model call per representative, with the quarter-turn law enforced
research/stage3c\G\README.md:3:Use the Stage 3 frozen equivariant encoder `equivariant_b8_s17`. Its index has the form `4r+i`, with a quadrant r and an inner index i. These results assume independent symbols in the first and second windows, each uniform over four labels, and circular receiver noise as specified by the simulator. For the repeat protocol the first symbol is sent again in window three. Public oscillator parameters can be arbitrary within that model.
research/stage3c\G\README.md:5:## Exact reduction
research/stage3c\G\README.md:21:## Which loss improvement is guaranteed?
research/stage3c\G\README.md:27:## Evaluation cost with many independent mothers
research/stage3c\G\README.md:33:## CPU check
research/stage3c\G\README.md:38:python research/stage3c/G/check_g.py research/stage3b_math/A/c4_s17_lookup.json
research/stage3c\G\quotient.py:1:"""One-call quarter-turn equivariance for the published s17 two-window code.
research/stage3c\G\quotient.py:11:        raise ValueError("Expected two integer s17 indices in [0, 15]")
research/stage3c\F\check_f.py:1:"""Behavioral sign diagnostics, Q3-repeat, Stage 3 s17 cache. CPU only."""
research/stage3c\F\check_f.py:125:    encoder_file = repo/"research/matched_cache/weights/equivariant_b8_s17.npz"
research/matched_cache\PROTOCOL.md:1:# Frozen matched-cache experiment
research/matched_cache\PROTOCOL.md:3:## Snapshot before fitting and evaluation
research/matched_cache\PROTOCOL.md:21:## Contract
research/matched_cache\PROTOCOL.md:29:## Scoring
research/matched_cache\PHYSICAL.md:1:# Fixed-budget hand encoder
research/stage3c\PROMPTS.md:1:# Stage 3C — questions to resolve before the host campaign
research/stage3c\PROMPTS.md:3:Parent: [Stage 3B, cbf3d2c0](https://github.com/20171806011/EM-FloatFormer/tree/cbf3d2c03f773731fdf1b0d695a63ec57cb62ca0/research/stage3b_math). Finish each question with a derivation, a counterexample or scope boundary, reproducible CPU evidence and an actionable experiment decision. Use the existing frozen s17 cache. Record known channel metadata, exposure to any training-time teacher and the observation seen at test time. A single campaign is the design goal; no particular host score is assumed.
research/stage3c\PROMPTS.md:5:## Parallel research prompts
research/matched_cache\LEARNED.md:1:# Learned per-window cache
research/matched_cache\LEARNED.md:5:## Training
research/matched_cache\LEARNED.md:11:## Rotations and what can be attributed to training
research/stage3c\README.md:1:# Stage 3C：上 GPU 前，哪些数学问题已经解决？
research/stage3c\README.md:3:**结论。** 在原始 C4 工况、已知频偏、固定 s17 的 8 比特缓存及无导频条件下，独立统计检验支持已发布的两张表在全部缓存状态上都选中了贝叶斯最优类别。把这两张表作为 Q1/Q3 的判决基准。下一笔算力应优先用于检验冻结宿主能否使用有符号相位信息和新证据。这项结论针对指定模拟器的 top-1 判决；任意工况的概率校准、Q2 的最优类别、GPU 宿主收益和真实录制仍需分别评估。
research/stage3c\README.md:17:## 一轮正式实验的执行决定
research/stage3c\README.md:19:1. **确定一个主问题。** 冻结 Qwen3.5-4B 和 s17，比较两个等预算连接层。两臂读取相同缓存、文字、训练波形和非零正负频偏；相位臂额外取由同一公开参数算出的各时刻正余弦，对照臂同尺寸槽填零。时刻、扩散和 SNR 等其他信息对等，不额外泄露漂移实现或标签。保持候选打分器、温度校准程序、监督目标、初始化配对和训练步数一致。建议主对比使用硬物理标签，暂把新老师用作诊断；训练时加入老师概率会改变所检验的干预。
research/stage3c\README.md:26:## 复现
research/stage3c\verify.py:67:    table = repo / "research/stage3b_math/A/c4_s17_lookup.json"
research/stage3c\verify.py:87:        compare(json.loads((studies/"A/q2_s17_kernel.json").read_text()), json.loads(kernel.read_text()), "A/kernel")
research/stage3c\work_results\C\REPORT.md:1:# Stage 3C · Work C：后验软标签的梯度风险
research/stage3c\work_results\C\REPORT.md:5:## 梯度与混合系数
research/stage3c\work_results\C\REPORT.md:21:## 新的最小反例
research/stage3c\work_results\C\REPORT.md:30:## 冻结模拟器和 8 比特缓存的 CPU 核验
research/stage3c\work_results\C\REPORT.md:32:调用冻结版 D 的 C4 波形生成流程和 s17 权重。每个窗口编码成 4 bit，两窗共同给出 8 bit 缓存。按冻结版四分之一转对称关系折叠到 16 个相对标签单元。两次各生成 262,144 个独立母样本，每批 8,192 个。复用 D 的 PCG64 种子 967193、967229 的序列前缀。固定独立发布教师的积分种子为 41117，权重和教师表均通过原 SHA-256 检查。两个样本批次均与教师积分独立，但与原 D 审计的样本前缀重合。
research/stage3c\work_results\C\REPORT.md:43:## 宿主配对实验
research/stage3c\work_results\C\REPORT.md:45:固定 s17 缓存、公开输入权限、两个候选任务的样本和权重、prompt、候选顺序、冻结宿主、每对训练的初始化、批次顺序、学习率及计算预算。H 用真实硬标签，S 用独立积分的全概率向量，M 用预设混合网格并通过独立开发母样本的硬标签 NLL 选权重。可加 W，用教师 argmax 的 one-hot 估计舍弃概率信息的影响。教师向量只进训练损失。覆盖范围外的工况各组都用硬标签，并单列结果。按母样本隔离积分、训练、开发、测评，旋转视图随母样本划分。对各训练种子在同一批新母样本上以真实硬标签配对报告 NLL、Brier、top-1 和按母样本聚类的区间。Q1 和重发 Q3 各自报告，所有组共用校准流程。剩余问题是宿主能否把方差减少转为同预算的 NLL 或准确率收益，以及这个收益对训练种子和工况的敏感性。
research/stage3c\work_results\C\REPORT.md:47:## 复现
research/stage3c\work_results\A\verify.py:82:    kernel = json.loads((frozen / "q2_s17_kernel.json").read_text())
research/stage3c\work_results\A\verify.py:84:    checkpoint = repo / "research/matched_cache/weights/equivariant_b8_s17.npz"
research/stage3c\work_results\D\REPORT.md:1:# Stage 3C Work D · s17 首选类别复现
research/stage3c\work_results\D\REPORT.md:3:冻结基线 `bbc76e3813a24840b9821c59cb8047cb85652575`。审计固定发布的 s17 权重与查找表，校验两份 SHA-256，直接取发布表里各行的唯一最大类别。没有重新拟合查找表。下文行号和类别均从 0 开始。
research/stage3c\work_results\D\REPORT.md:10:## 独立单位与 256 对缓存
research/stage3c\work_results\D\REPORT.md:14:对窗一和窗三同时乘 i^a、对窗二乘 i^b，等价于令 (U,V) 加上 (a,b)，并旋转对应窗的圆对称噪声，联合律保持。s17 编码先以复和扇区 r 把输入旋回，再计算局部索引 j，故 z=4r+j 在四分之一圈转动下只有 r 随之变化。对角边界处的集合在连续噪声下概率为零。Q1 增加 b−a，Q3 增加 a。于是从两题标签分别减去 r₂−r₁、r₁ 并取模 4 后，标签与行 c=4j₁+j₂ 的联合概率在同一轨道的 16 对缓存中等分。每个给定 (j₁,j₂) 的轨道能唯一枚举全部 16 个 (r₁,r₂)，共 16×16=256 对。实际调用发布读表器核对了全部 256 对的两题首选。故在行内最优即在完整 8 比特缓存上逐对最优。
research/stage3c\work_results\D\REPORT.md:16:## 两阶段统计
research/stage3c\work_results\D\REPORT.md:38:## 适用条件与宿主实验
research/stage3c\work_results\D\REPORT.md:40:结论适用于已知 C4、漂移 +0.35 rad/步、扩散 0.10 rad/√步、2 dB、两段间隔均为 2、无导频、冻结 s17 双窗 8 比特缓存。两题的发布首选在该模拟总体中获得逐缓存对的零一损失 Bayes 决策证书。完整概率校准、对数损失、导频后的问答、未知或变化参数下的表现需要单独评估。宿主实验需检验语言模型能否仅凭同样的冻结缓存和公开条件，保持两题的首选映射并在改变参考条件后正确更新答案。
research/stage3c\work_results\D\REPORT.md:42:## 复现
research/stage3c\work_results\B\REPORT.md:1:# Stage 3C Work B · 有符号频偏与布朗扩散
research/stage3c\work_results\B\REPORT.md:5:## 联合律
research/stage3c\work_results\B\REPORT.md:17:把同一符号与初始朝向先验、AWGN、采集协议和固定 s17 编码施加于同一联合律，可得到相同的观测及目标联合分布，因而给出相同条件预测。仅 Q1 看两窗时只涉及 $h$。Q3 重发还需预测时刻 $T=h+\ell$，授权的同信道导频进入后需保留 $t_p$。新采集可共享潜相位并具有独立测量噪声。同编号复制只有一份证据。若不给有效导频，平衡独立均匀标签与随机公共四相朝向使 Q2 保持均匀。若 Q3 发独立均匀新符号，圆对称噪声下未来四个扇区条件概率各为 $1/4$。
research/stage3c\work_results\B\REPORT.md:19:## 遗漏信息的反例
research/stage3c\work_results\B\REPORT.md:24:第一组用 s17 实际编码四个无噪声窗口，确认四个朝向各有不同码。每工况枚举 64 个均匀 $(x_1,x_2,G)$ 三元组，给定相同观测只留下一个。第二组独立积分圆上高斯密度的傅里叶展开。扇区中心为 $c$ 时，常数项为 $1/4$，第 $n$ 项为 $\frac{2}{\pi n} e^{-vn^2/2} \sin(n\pi/4)\cos(n(c-\mu))$。取前 64 项，最小方差 0.08 下尾项绝对和小于 $10^{-74}$。与冻结版逐周期正态 CDF 算法比较，八个概率的最大绝对差为 $1.11\times10^{-16}$。
research/stage3c\work_results\B\REPORT.md:26:## 已知与未知符号
research/stage3c\work_results\B\REPORT.md:31:2. 无初始朝向提示的 CPU 补充检查复用冻结模拟器、s17 权重和每窗 4 比特的缓存对。原 C4 的 $+0.35$ 配上镜像 $-0.35$，$\sigma=0.10,h=\ell=2,t_p=4$，2 dB，Q3 重发且无导频。每个符号 32,768 条拟合、16,384 条留出；同一划分内两符号复用随机数，拟合与留出使用不同种子。沿用基种子 840101 与 840303，实际 RNG 种子为 84010103 与 84030303，只扩充区组数。观察量为两窗缓存，两个符号的留出各覆盖全部 256 种码对。每个符号的联合直方图加一平滑，隐藏符号时先合并联合计数再作条件归一化。留出准确率为 63.2355% 与 33.7128%，平均对数损失为 0.966528 与 1.309907 nats。这是拟合查表读出的留出分数。它与受控例子的时序、频偏、噪声和可见信息均有差异，因此分差不能单独归因于撤掉朝向提示。
research/stage3c\work_results\B\REPORT.md:33:## 正式实验接口
research/stage3c\work_results\B\REPORT.md:37:| 缓存、中文题面与候选、协议及参考 | 相同 s17 码对、题序、Q3 重发或新发规则、参考揭示时点、可见导频、共享信道标记及采集编号与去重规则 |
research/stage3c\work_results\B\REPORT.md:43:## 复现
research/stage3c\work_results\E\verify.py:107:    weight = ROOT / "research/matched_cache/weights/equivariant_b8_s17.npz"
research/stage3c\work_results\G\verify.py:24:LOOKUP = ROOT / "research/stage3b_math/A/c4_s17_lookup.json"
research/stage3c\work_results\G\verify.py:28:Q2_KERNEL = json.loads((BASE / "A/q2_s17_kernel.json").read_text())
research/stage3c\work_results\G\verify.py:62:    weight = ROOT / "research/matched_cache/weights/equivariant_b8_s17.npz"
research/stage3c\work_results\A\README.md:1:# Work A · 迟到导频的 Q2 贝叶斯更新
research/stage3c\work_results\A\README.md:3:基线提交为 `bbc76e3813a24840b9821c59cb8047cb85652575`。条件固定为 C4，SNR 2 dB，数据窗在 0 和 2，导频在 4。使用冻结 `equivariant_b8_s17` 权重和两枚 4 比特缓存索引。标签顺序对应 `[1,i,-1,-i]`。
research/stage3c\work_results\A\README.md:5:## 从缓存索引到后验
research/stage3c\work_results\A\README.md:33:## 实跑和评测接口
research/stage3c\work_results\A\README.md:39:cmp research/stage3c/A/q2_s17_kernel.json /tmp/stage3c-A-kernel.json
research/stage3c\work_results\A\README.md:47:宿主评分可直接调用 `research.stage3c.work_results.A.verify.read_reference(kernel, z1, z2, records)`。核来自冻结的 `research/stage3c/A/q2_s17_kernel.json`。记录为 `(acquisition_id, same_channel, time, complex_matched_sum)`，同信道记录固定在时刻 4，SNR 为 2 dB。保留副本的编号，给独立采样分配新编号，无关参考传 `False`。以 argmax 计准确率、以真实标签的负自然对数计 NLL。此参考使用复匹配和。宿主若只收到量化后的导频特征，需要为量化观测单独建立似然；正交的宿主实验要回答模型能否在严格配对的输入上利用新测量，并对副本和无关参考保持同一后验。
research/stage3c\work_results\A\README.md:49:## 核对结果
research/stage3c\work_results\D\audit.py:1:"""Independent CPU audit of the Stage 3C s17 decision certificate."""
research/stage3c\work_results\D\audit.py:166:    lookup = (research / "stage3b_math/A/c4_s17_lookup.json").read_text(encoding="utf-8")
research/stage3c\work_results\D\audit.py:167:    weights = research / "matched_cache/weights/equivariant_b8_s17.npz"
research/stage3c\work_results\B\check_work_b.py:1:"""CPU reproduction of Stage 3C B with independent Fourier and s17 checks."""
research/stage3c\work_results\B\check_work_b.py:29:    """Use actual s17 on the four noiseless cardinal windows."""
research/stage3c\work_results\B\check_work_b.py:87:    weights = ROOT / "research/matched_cache/weights/equivariant_b8_s17.npz"
research/stage3c\work_results\C\check.py:118:    weight_path = matched / "weights/equivariant_b8_s17.npz"
research/stage3c\work_results\C\check.py:119:    table_path = REPO / "research/stage3b_math/A/c4_s17_lookup.json"
research/stage3c\work_results\E\REPORT.md:1:# Stage 3C · Work E · 正式宿主盲测需要多少独立区组
research/stage3c\work_results\E\REPORT.md:5:## 主比较与推导
research/stage3c\work_results\E\REPORT.md:7:冻结 Qwen3.5-4B 和 s17 8 比特编码器，比较同尺寸数值槽中的真实相量与全零。对齐文字、缓存、监督、训练预算和配对初始化，测揭晓导频前的 Q3 重发题。主工况为 ±0.40 rad/步、h=ℓ=2、累计转角 ±1.60 rad、2 dB、布朗 σ=0.10。每条母波形的双臂结果先配对，再等权平均两个频偏符号、三种旋转及三组预登记的固定配对连接层种子。随后每 64 条生成一个平衡区组均值。准确率取相量臂减零槽臂，NLL 取零槽臂减相量臂。两项双侧 95% 配对 t 区间的下端都须超过实际收益门槛。
research/stage3c\work_results\E\REPORT.md:20:## 数值结果
research/stage3c\work_results\E\REPORT.md:35:另用冻结模拟器及现有 s17（编码器训练种子 17）权重实际生成 131 个 64 条平衡区组。CPU 专用命名空间 316027，每块用 SeedSequence 派生种子，两符号共享符号、漂移创新与噪声。核对 50,304 条符号×旋转观测的 Q3 标签变换及 8 比特索引，每个符号均观察到 256 种缓存码。两臂各三固定种子时，同批母波形对应 301,824 条逻辑计分行；固定提示、确定性打分且仅有码和符号变化时，每种子每臂最多 512 个不同输入。CPU 核验未运行连接层或宿主，未使用历史盲测种子或观测值来估计方差。
research/stage3c\work_results\E\REPORT.md:37:## 开发、封存与盲测
research/stage3c\work_results\E\REPORT.md:43:## CPU 复现
research/stage3c\work_results\G\README.md:1:# Stage 3C Work G · 一次代表元读出与群平均
research/stage3c\work_results\G\README.md:5:## 标签作用与一次读出
research/stage3c\work_results\G\README.md:19:## 群平均的风险与单次上界
research/stage3c\work_results\G\README.md:36:## CPU 核对
research/stage3c\work_results\G\README.md:41:python -B research/stage3c/G/check_g.py research/stage3b_math/A/c4_s17_lookup.json
research/stage3c\work_results\G\README.md:46:- 用冻结 s17 权重在临时目录重建 Q1/Q3 表。积分种子 41117、60,000 个独立噪声窗、20 阶 Gauss–Hermite。重建代表元表与发布值逐元素一致。重建程序比较 512 个全缓存后验与直接数值积分，最大差 1.44e-15，argmax 零分歧。
research/stage3c\work_results\G\README.md:48:- 冻结 C4 模拟器生成 s3_test 的 1,024 个母样本（基种子 840303，实际 RNG 种子 84030303，16 个区块）。用冻结 s17 重新编码两窗的所有旋转，检查 32,768 个窗码、16,384 个重发扇区和 49,152 个概率向量，全部通过。原始样本覆盖 218 个缓存对，1,024 个有效导频值互异。旋转复本属于各自母样本。Q1 / Q2 / Q3 的 top-1 为 58.79% / 65.23% / 64.84%，细分 NLL 与 Brier 在结果文件。原 G 的人工偏置风险和这里的模拟打分使用不同的读出与样本律。
research/stage3c\work_results\G\README.md:50:## 读出调用数及宿主实验
research/stage3c\work_results\F\README.md:1:# Stage 3C · Work F · 固定缓存的频偏符号干预
research/stage3c\work_results\F\README.md:3:起点为 `bbc76e3813a24840b9821c59cb8047cb85652575`。实跑使用冻结 C4 模拟器和 s17 权重（SHA-256 见结果），Q3-repeat、无导频、2 dB、两个四比特索引、`h=ell=2`，布朗扩散系数 0.10 rad/√step，频偏为 ±0.35 rad/step。结论对应这个条件和这对频偏。
research/stage3c\work_results\F\README.md:5:## 物理配对和读出干预
research/stage3c\work_results\F\README.md:15:## CPU 结果
research/stage3c\work_results\F\README.md:33:## 冻结宿主需要导出什么
research/stage3c\work_results\F\README.md:39:## 复现
research/stage3c\work_results\F\verify_f.py:20:WEIGHT = Path("research/matched_cache/weights/equivariant_b8_s17.npz")
research/stage3c\work_prompts\A.md:1:# Work 任务 A · 迟到的导频能提供多少新信息
research/stage3c\work_prompts\A.md:5:**起点**　在 [EM-FloatFormer 的冻结提交](https://github.com/20171806011/EM-FloatFormer/tree/bbc76e3813a24840b9821c59cb8047cb85652575) 中阅读 [本题报告](https://github.com/20171806011/EM-FloatFormer/blob/bbc76e3813a24840b9821c59cb8047cb85652575/research/stage3c/A/Q2.md) 和同目录代码、JSON。使用该提交内的模拟器、s17 权重及 8 比特缓存。以下命令从仓库根目录运行。
research/stage3c\work_prompts\A.md:13:**验收**　对照 q2_results.json 与 q2_s17_kernel.json。分别报告增加真实观测和只精化旧读出的配对准确率、对数损失和不确定度，交付一个可直接用于模型评测的参考读出规则。
research/stage3c\work_prompts\B.md:1:# Work 任务 B · 哪些相位与时间信息足以确定预测
research/stage3c\work_prompts\B.md:5:**起点**　在 [EM-FloatFormer 的冻结提交](https://github.com/20171806011/EM-FloatFormer/tree/bbc76e3813a24840b9821c59cb8047cb85652575) 中阅读 [本题报告](https://github.com/20171806011/EM-FloatFormer/blob/bbc76e3813a24840b9821c59cb8047cb85652575/research/stage3c/B/B_features.md) 和同目录代码、JSON。使用该提交内的模拟器、s17 权重及 8 比特缓存。以下命令从仓库根目录运行。
research/stage3c\work_prompts\C.md:1:# Work 任务 C · 硬标签与后验软标签应怎样混合
research/stage3c\work_prompts\C.md:5:**起点**　在 [EM-FloatFormer 的冻结提交](https://github.com/20171806011/EM-FloatFormer/tree/bbc76e3813a24840b9821c59cb8047cb85652575) 中阅读 [本题报告](https://github.com/20171806011/EM-FloatFormer/blob/bbc76e3813a24840b9821c59cb8047cb85652575/research/stage3c/C/C_distillation.md) 和同目录代码、JSON。使用该提交内的模拟器、s17 权重及 8 比特缓存。以下命令从仓库根目录运行。
research/stage3c\work_prompts\D.md:1:# Work 任务 D · 固定的 Q1/Q3 查找表是否选对每一类
research/stage3c\work_prompts\D.md:5:**起点**　在 [EM-FloatFormer 的冻结提交](https://github.com/20171806011/EM-FloatFormer/tree/bbc76e3813a24840b9821c59cb8047cb85652575) 中阅读 [本题报告](https://github.com/20171806011/EM-FloatFormer/blob/bbc76e3813a24840b9821c59cb8047cb85652575/research/stage3c/D/D_certificate.md) 和同目录代码、JSON。使用该提交内的模拟器、s17 权重及 8 比特缓存。以下命令从仓库根目录运行。
research/stage3c\work_prompts\D.md:7:**研究问题**　审查已发布 s17 查找表在已知 C4 和无导频条件下的首选判决。推导为什么 16 个旋转轨道覆盖 256 对缓存；复算两阶段的独立母波形计数、最接近翻转的一行、96 项竞争类别的检验及全局 p 值。解释第一阶段同时置信界和第二阶段交并检验的错误预算。
research/stage3c\work_prompts\E.md:1:# Work 任务 E · 一次正式宿主实验该测多少条独立波形
research/stage3c\work_prompts\E.md:5:**起点**　在 [EM-FloatFormer 的冻结提交](https://github.com/20171806011/EM-FloatFormer/tree/bbc76e3813a24840b9821c59cb8047cb85652575) 中阅读 [本题报告](https://github.com/20171806011/EM-FloatFormer/blob/bbc76e3813a24840b9821c59cb8047cb85652575/research/stage3c/E/REPORT.md) 和同目录代码、JSON。使用该提交内的模拟器、s17 权重及 8 比特缓存。以下命令从仓库根目录运行。
research/stage3c\work_prompts\F.md:1:# Work 任务 F · 怎样判断宿主真的使用了频偏符号
research/stage3c\work_prompts\F.md:5:**起点**　在 [EM-FloatFormer 的冻结提交](https://github.com/20171806011/EM-FloatFormer/tree/bbc76e3813a24840b9821c59cb8047cb85652575) 中阅读 [本题报告](https://github.com/20171806011/EM-FloatFormer/blob/bbc76e3813a24840b9821c59cb8047cb85652575/research/stage3c/F/F_signed_host_probe.md) 和同目录代码、JSON。使用该提交内的模拟器、s17 权重及 8 比特缓存。以下命令从仓库根目录运行。
research/stage3c\work_prompts\G.md:1:# Work 任务 G · 一次模型读出能保证哪一种旋转对称性
research/stage3c\work_prompts\G.md:5:**起点**　在 [EM-FloatFormer 的冻结提交](https://github.com/20171806011/EM-FloatFormer/tree/bbc76e3813a24840b9821c59cb8047cb85652575) 中阅读 [本题报告](https://github.com/20171806011/EM-FloatFormer/blob/bbc76e3813a24840b9821c59cb8047cb85652575/research/stage3c/G/README.md) 和同目录代码、JSON。使用该提交内的模拟器、s17 权重及 8 比特缓存。以下命令从仓库根目录运行。
research/stage3c\work_prompts\G.md:10:python -B research/stage3c/G/check_g.py research/stage3b_math/A/c4_s17_lookup.json
research/stage3c\work_prompts\README.md:1:# 分别交给七个 Work 对话的提示词
research/stage3c\work_prompts\README.md:15:## 已执行到哪一步
research/stage3c\work_prompts\README.md:21:## 七个独立 Work 对话的交付
research/stage3c\crossreviews\E.md:1:# E 组独立复核：宿主样本量与功效
research/stage3c\crossreviews\G.md:1:# G 组独立复核 · 两个数据窗各自旋转后的等变读出
research/stage3c\crossreviews\G.md:6:python research/stage3c/G/check_g.py research/stage3b_math/A/c4_s17_lookup.json
research/stage3c\D\check_d.py:136:    table_text = (repo / "research/stage3b_math/A/c4_s17_lookup.json").read_text(encoding="utf-8")
research/stage3c\D\check_d.py:141:    weights = matched / "weights/equivariant_b8_s17.npz"
research/stage3c\D\CONFIRMATION.md:1:# Additional independent test of the published decisions
research/stage3c\D\CONFIRMATION.md:3:The first C4 cache audit, using 4,194,304 independent mothers, suggested one difficult Q1 orbit. This motivated a separate confirmatory run. The fixed classifier is the already published s17 lookup at parent cbf3d2c0. For the follow-up, use exactly 2^24 new i.i.d. mothers, seed 967229, and no interim looks or refitting. Test whether *all* 96 chosen-versus-rival comparisons across two tasks and 16 orbits have positive population margins. Reject the union null only when every one-sided exact conditional binomial test rejects at .005; use the maximum component p-value. A single true component null suffices to bound the global false positive rate. No per-cell significance is claimed from selecting individual p-values. For a joint 95% claim that also includes stage-one intervals, assign at most .045 to stage one and .005 to this global test. The follow-up design is informed by the first audit; no portion of the follow-up simulation was used to fit the published classifier.
research/stage3c\D\D_certificate.md:1:# A finite-sample top-1 certificate for the released C4 readout
research/stage3c\D\D_certificate.md:3:The object under audit is the actual [Stage 3B s17 lookup](https://github.com/20171806011/EM-FloatFormer/blob/cbf3d2c03f773731fdf1b0d695a63ec57cb62ca0/research/stage3b_math/A/c4_s17_lookup.json). Its published probability table was computed with 60,000 independent integration windows, seed 41117 and Hermite order 20. We take its decisions as fixed before collecting each forward sample. The two tasks are Q1 (relative symbol) and pre-pilot Q3 (received sector for a repeat of the first symbol) in C4 with known rate +0.35 rad/step, phase diffusion 0.10 rad per square-root step, 2 dB, and two consecutive time gaps of 2. The information is the pair of frozen 4-bit window codes from the actual s17 weights, and the public condition. Bayes optimality throughout means population top-1 accuracy at this information and under this simulator.
research/stage3c\D\D_certificate.md:5:## Direct i.i.d. experiment
research/stage3c\D\D_certificate.md:9:## Why 16 rows cover 256 cache pairs
research/stage3c\D\D_certificate.md:13:## Simultaneous bounds from the first sample
research/stage3c\D\D_certificate.md:21:## Independent global confirmation
research/stage3c\D\D_certificate.md:27:## Interpretation and reproduction
```

### Event #7: `item.completed` — `command_execution` (`item_4`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-Content -Encoding utf8 research/matched_cache/learned.py
Get-Content -Encoding utf8 research/stage3c/G/quotient.py
Get-Content -Encoding utf8 research/stage3c/A/q2_reader.py'
```

**Aggregated Output:**

```text
"""Small pilot-free I/Q encoders with an integer index per received window."""
from dataclasses import dataclass

import numpy as np


ROOTS = np.array([1, 1j, -1, -1j])
LABELS = np.arange(4)


def input_features(windows, equivariant):
    iq = np.asarray(windows, dtype=np.complex128)
    if iq.ndim != 2 or iq.shape[1] != 3:
        raise ValueError("Expected an array of three complex samples per window")
    turns = np.zeros(len(iq), dtype=np.int64)
    if equivariant:
        # Half-open quadrant [-45��,45��), and its three quarter-turn shifts.
        anchor = iq.sum(axis=1)
        missing = anchor == 0
        if np.any(missing):
            candidates = iq[missing]
            first = np.argmax(candidates != 0, axis=1)
            anchor[missing] = candidates[np.arange(len(candidates)), first]
        u = anchor.real + anchor.imag
        v = anchor.imag - anchor.real
        turns[(u > 0) & (v >= 0)] = 1
        turns[(u <= 0) & (v > 0)] = 2
        turns[(u < 0) & (v <= 0)] = 3
        iq = iq * ROOTS[turns].conj()[:, None]
    return np.column_stack((iq.real, iq.imag)), turns


def forward(x, weights):
    w1, b1, w2, b2 = weights
    hidden = np.tanh(x @ w1 + b1)
    logits = hidden @ w2 + b2
    ex = np.exp(logits - logits.max(axis=1, keepdims=True))
    return hidden, ex / ex.sum(axis=1, keepdims=True)


def fit_network(x, target, rng):
    width = 16
    weights = [
        rng.normal(0, np.sqrt(2 / (6 + width)), (6, width)),
        np.zeros(width),
        rng.normal(0, np.sqrt(2 / (width + 4)), (width, 4)),
        np.zeros(4),
    ]
    first = [np.zeros_like(w) for w in weights]
    second = [np.zeros_like(w) for w in weights]
    step = 0
    for _ in range(48):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 256):
            index = order[start:start + 256]
            xb = x[index]
            hidden, prediction = forward(xb, weights)
            residual = (prediction - target[index]) / len(index)
            delta = (residual @ weights[2].T) * (1 - hidden**2)
            grads = (xb.T @ delta, delta.sum(axis=0), hidden.T @ residual,
                     residual.sum(axis=0))
            step += 1
            for w, m, v, g in zip(weights, first, second, grads):
                m *= 0.9
                m += 0.1 * g
                v *= 0.999
                v += 0.001 * g*g
                w -= 0.005 * (m / (1 - 0.9**step)) / (np.sqrt(v / (1 - 0.999**step)) + 1e-8)
    return tuple(weights)


def assign(posterior, centers):
    # The posterior entropy is constant across candidate centers.
    return (posterior @ np.log(centers).T).argmax(axis=1)


def fit_centers(prediction, count, rng):
    if count == 1:
        return prediction.mean(axis=0, keepdims=True)
    centers = [prediction[rng.integers(len(prediction))].copy()]
    # KL-proportional seeding, using the distance to the nearest center so far.
    self_score = np.sum(prediction * np.log(prediction), axis=1)
    distance = np.full(len(prediction), np.inf)
    for _ in range(1, count):
        d = np.maximum(0.0, self_score - prediction @ np.log(centers[-1]))
        distance = np.minimum(distance, d)
        centers.append(prediction[rng.choice(len(prediction), p=distance / distance.sum())].copy())
    centers = np.stack(centers)
    prev = None
    for _ in range(30):
        code = assign(prediction, centers)
        if prev is not None and np.array_equal(prev, code):
            break
        new = centers.copy()
        occupied = np.bincount(code, minlength=count)
        for j in np.flatnonzero(occupied):
            new[j] = prediction[code == j].mean(axis=0)
        if np.any(occupied == 0):
            worst_first = np.argsort(self_score - np.sum(prediction * np.log(centers[code]), axis=1))[::-1]
            for j, i in zip(np.flatnonzero(occupied == 0), worst_first):
                new[j] = prediction[i]
        prev, centers = code, new
    return centers


@dataclass
class Learned:
    equivariant: bool
    weights: tuple
    centers: np.ndarray
    table: np.ndarray

    def encode(self, windows):
        x, turns = input_features(windows, self.equivariant)
        _, prediction = forward(x, self.weights)
        local = assign(prediction, self.centers)
        return turns * len(self.centers) + local

    def save(self, path):
        w1, b1, w2, b2 = self.weights
        np.savez(path, equivariant=self.equivariant, w1=w1, b1=b1, w2=w2, b2=b2,
                 centers=self.centers, table=self.table)


def load(path):
    with np.load(path, allow_pickle=False) as data:
        return Learned(bool(data["equivariant"]), tuple(data[k] for k in ("w1", "b1", "w2", "b2")),
                       data["centers"], data["table"])


def fit(windows, soft_targets, bits, seed, equivariant):
    if bits not in (4, 8):
        raise ValueError("The frozen experiment uses 4 or 8 bits for the two-window cache")
    x, turns = input_features(windows, equivariant)
    target = np.asarray(soft_targets, dtype=np.float64)
    if target.shape != (len(x), 4) or not np.allclose(target.sum(axis=1), 1):
        raise ValueError("Expected one normalized four-label posterior per window")
    target = np.take_along_axis(target, (LABELS[None, :] + turns[:, None]) % 4, axis=1)
    rng = np.random.default_rng(seed)
    weights = fit_network(x, target, rng)
    _, prediction = forward(x, weights)
    size = 2 ** (bits // 2)
    centers = fit_centers(prediction, size // 4 if equivariant else size, rng)
    local = assign(prediction, centers)
    if np.any(np.bincount(local, minlength=len(centers)) == 0):
        raise ValueError("Unused training code in the fitted vector quantizer")
    table = np.stack([target[local == i].mean(axis=0) for i in range(len(centers))])
    if equivariant:
        table = np.concatenate([np.roll(table, r, axis=1) for r in range(4)])
    return Learned(bool(equivariant), weights, centers, table)
"""One-call quarter-turn equivariance for the published s17 two-window code.

External wrapper for four-candidate distributions. Pilot data and metadata
remain in their original coordinates. Free text is outside this interface.
"""


def canonical_input(z1, z2, task):
    """Return the two canonical indices and the output right-shift."""
    if not all(isinstance(z, int) and 0 <= z < 16 for z in (z1, z2)):
        raise ValueError("Expected two integer s17 indices in [0, 15]")
    turn1, inner1 = divmod(z1, 4)
    turn2, inner2 = divmod(z2, 4)
    if task == "q1":
        shift = turn2 - turn1
    elif task in ("q2", "q3_repeat"):
        shift = turn1
    else:
        raise ValueError("Supported tasks are q1, q2 and q3_repeat")
    return inner1, inner2, shift % 4


def rotate(values, shift):
    if len(values) != 4:
        raise ValueError("Expected four scores")
    return [values[(k-shift) % 4] for k in range(4)]


def one_call(reader, z1, z2, task, context):
    """reader takes canonical codes, task and public context, then returns four scores."""
    i, j, shift = canonical_input(z1, z2, task)
    return rotate(reader(i, j, task, context), shift)
"""Q2 at the pinned C4 condition; inputs are the two integer cache codes and pilot matched sums."""
import argparse
import cmath
import json
import math
from pathlib import Path


KERNEL_PATH = Path(__file__).with_name("q2_s17_kernel.json")


def read(kernel, z1, z2, references):
    """Each reference is (acquisition ID, shares_channel, complex matched sum).

    This kernel is calibrated for same-time 2 dB pilots at t=4. Keep acquisition
    IDs when making copies; assign new IDs to independent measurement noise.
    """
    if any(not isinstance(z, int) or not 0 <= z < 16 for z in (z1, z2)):
        raise ValueError("Both cache indices must be integers in 0..15")
    seen = {}
    for identifier, same_channel, matched_sum in references:
        t = complex(matched_sum)
        if not math.isfinite(t.real) or not math.isfinite(t.imag):
            raise ValueError("Non-finite matched sum")
        record = (same_channel, t)
        if identifier in seen and seen[identifier] != record:
            raise ValueError("Conflicting metadata or value for a repeated acquisition ID")
        seen[identifier] = record
    valid = [t for same_channel, t in seen.values() if same_channel]
    if not valid:
        return [0.25]*4
    weighted_t = 2*kernel["rho"]*sum(valid)
    # Stabilize likelihood scale with one common maximum across all g, phase nodes.
    ll = [[(weighted_t*(1j**(-g))*cmath.exp(-1j*ph)).real for g in range(4)]
          for ph in kernel["pilot_phase_nodes"]]
    shift = max(max(row) for row in ll)
    carrier = [sum(w*math.exp(row[g]-shift)
                   for w,row in zip(kernel["weights_given_z2"][z2], ll)) for g in range(4)]
    norm = sum(carrier)
    return [sum(carrier[g]*kernel["u_given_z1"][z1][(k+g)%4] for g in range(4))/norm
            for k in range(4)]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("z1", type=int)
    p.add_argument("z2", type=int)
    p.add_argument("--pilot", nargs=3, action="append", metavar=("ID","REAL","IMAG"), default=[])
    args = p.parse_args()
    kernel = json.loads(KERNEL_PATH.read_text(encoding="utf-8"))
    refs = [(idx, True, complex(float(a),float(b))) for idx,a,b in args.pilot]
    print(json.dumps(read(kernel, args.z1, args.z2, refs)))
```

### Event #8: `item.completed` — `command_execution` (`item_6`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding utf8 'C:/Users/李则徐/.codex/memories/MEMORY.md' | Select-Object -Skip 28 -First 44
Get-Content -Encoding utf8 'C:/Users/李则徐/.codex/memories/rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md' | Select-Object -First 45"
```

**Aggregated Output:**

```text
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

- P4 is a reproducible CPU digital-baseband prototype with C* gauge normalization, complex RMSNorm, Hermitian attention, complex relative-phase RoPE, radial activation, physical loss, nonlinear PA, Doppler/CFO, sparse pilots, and coded LDPC BLER evaluation; it is not real electronic/hardware-native execution. [Task 1]
- Formal P4-D used ten seeds, 600 steps, exact integer accounting, 1024-state sign-flip tests and Holm correction; independent replay verified 150 coded rows, ten checkpoints, 80 factorial arms, 240 gain observations, 253/253 E2E and 100/100 unit/adversarial cases. [Task 1]
- The 2D primary result was negative for learned superiority: at 10 dB/strong PA, EM had 3,266/4,160 block errors versus fixed LMMSE 3,227 and fixed IDD-PA 3,211, and was worse than fixed IDD-PA after Holm (`p=0.0234375`). Narrow 1D ablations found gain normalization/RMSNorm improved BER by 2.615 points and Hermitian attention plus complex RoPE by 1.086 points. [Task 1]
- Gaussian pilot interpolation attenuated long delays; Fourier/per-path phase front-ends supplied nearly all later gains. The decisive next experiment shares physical front-end, pilot information, decoder and inference budget across EM and classical receivers, isolating the learned module. [Task 1]
- Public commit `8380084c6dece64119ab72f9c3c1c212e154fb17` contains 387 new files. `publication.verify_p4` verifies frozen dependencies, historical integrity, counts, hashes, factorial results and test receipts; `--replay` regenerates coded counts and retrains factorial arms. The redacted archive kept 89/92 members byte-identical. [Task 2]

## Failures and how to do differently

- Do not attribute front-end gains to Transformer attention when preprocessing or priors differ; use matched physical preprocessing and compute budgets. Do not claim P4 proved EM-native Transformer superiority. [Task 1]
- Full-projection BF16 exceeded the max-LLR-error target (about 0.202); retain input/physical readout projections in FP32 and use BF16 only inside Transformer blocks (tested max below 0.012). CPU BF16 throughput was about 0.436× FP32. [Task 1]
- `gh auth status`/Git push failed because the local token was invalid; connector APIs published blobs, trees, commit and ref. Keep a later nonzero isolated replay separate from the completed formal receipts. [Task 2]

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
```

### Event #9: `item.completed` — `command_execution` (`item_5`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '"'^#|先编码|4 bit|4-bit|LEMMO|终极'"' docs/LONG_TERM_RESEARCH_PLAN.md docs/MULTIMODAL_INTEGRATION_PLAN.md
rg --files --hidden research/stage4_math_and_design | Select-Object -First 60
Get-Item research/matched_cache/weights/equivariant_b8_s17.npz
foreach ("'$p in @('"'AGENTS.md','research/AGENTS.md','research/stage4_math_and_design/AGENTS.md','research/stage4_math_and_design/M5/AGENTS.md')) { if (Test-Path "'$p) { Get-Content -Encoding utf8 $p }}'
```

**Aggregated Output:**

```text
docs/LONG_TERM_RESEARCH_PLAN.md:1:# EM\-FloatFormer 长期研究计划
docs/LONG_TERM_RESEARCH_PLAN.md:3:## 让 AI 从同一段电磁波形中读出关系、指出依据，并随新证据更新答案
docs/LONG_TERM_RESEARCH_PLAN.md:7:### 总目标与双重验收
docs/LONG_TERM_RESEARCH_PLAN.md:13:**外部起点。** RF-GPT 已能做频谱图问答，MERLIN 已研究低信噪比下的电磁感知与推理，LEMMO 已能从长 I/Q 中按指令抽取证据并描述跨信号关系。LEMMO 使用 Qwen3.5-4B，可用于同底座比较。分别报告双方原生用法，以及在访问权限、缓存精度、训练预算可匹配时的实验。为双方记录重读、上下文和延迟成本。外部方法未支持的题单列，双方均可评分的题才用于性能比较。[RF-GPT](https://arxiv.org/abs/2602.14833) · [MERLIN](https://em-merlin.github.io/) · [LEMMO](https://arxiv.org/html/2611.09476v1)
docs/LONG_TERM_RESEARCH_PLAN.md:15:**最终盲测。** 先编码，后揭题。两种解释均与旧波形相容时，只允许再读取一个已录制的候选参考窗。模型先选窗口并写下各解释预期的观测，再揭示真实波形并评分。对照包括随机窗、最近窗以及由物理模型预测能最大程度减少歧义的窗。提前约定主要指标、可接受的误差差距和采集划分。等缓存与观测预算时，比较答案风险、置信度和证据定位；达到约定准确度时，比较所需缓存和额外读取量。天然观测、人工注入和独立采集分别报告。若现有方法打平，保留已验证的边界与反例，继续寻找下一个确有差异的问题。
docs/LONG_TERM_RESEARCH_PLAN.md:19:### 一、我们现在站在哪里
docs/LONG_TERM_RESEARCH_PLAN.md:35:### 二、我们要争取的贡献
docs/LONG_TERM_RESEARCH_PLAN.md:37:已有 RF\-GPT 用频谱图进行无线信号问答。MERLIN 研究电磁信号感知与推理及低信噪比条件。LEMMO 已报告从最长八百万个复采样的原始 I/Q 中按指令检索证据，并描述跨信号关系。仅凭“输入 I/Q、接上语言模型、能回答问题”不足以建立本项目的原创性。值得争取的贡献是，针对明确的一族物理问题，给出**有限表示保留了什么、何时必然丢失答案、什么额外观测可以恢复答案**的定量结果，并在真实波形上检验这些预测。与最接近模型的同题同信息对比将决定可以提出多强的实证主张。[RF\-GPT](https://arxiv.org/abs/2602.14833) · [MERLIN，IEEE/CVF Conference on Computer Vision and Pattern Recognition 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Shen_MERLIN_Building_Low-SNR_Robust_Multimodal_LLMs_for_Electromagnetic_Signals_CVPR_2026_paper.pdf) · [LEMMO](https://arxiv.org/abs/2611.09476)
docs/LONG_TERM_RESEARCH_PLAN.md:45:### 三、阶段安排
docs/LONG_TERM_RESEARCH_PLAN.md:60:### 四、真实数据怎样用
docs/LONG_TERM_RESEARCH_PLAN.md:70:### 五、实验怎样表明模型读到了波形
docs/LONG_TERM_RESEARCH_PLAN.md:77:### 六、第一轮怎样并行
docs/LONG_TERM_RESEARCH_PLAN.md:93:### 附录 · GitHub 分支快照
docs/MULTIMODAL_INTEGRATION_PLAN.md:1:# EM-FloatFormer 接入已有多模态模型的计划
docs/MULTIMODAL_INTEGRATION_PLAN.md:5:## 用户拿到什么
docs/MULTIMODAL_INTEGRATION_PLAN.md:11:## 3B–27B 开放权重宿主
docs/MULTIMODAL_INTEGRATION_PLAN.md:19:| **4B、9B · Qwen3.5** | 2026-03-02 | 4B 是首个领域对照。已有原生 I/Q 研究 LEMMO 使用同款 4B 语言底座并用 LoRA 适配。优先在相同长度、相同问题和相同可见参考下比较。若要归因到编码方法，还要匹配训练数据、可训练参数和优化预算。4B 与 9B 可比较同代规模趋势。 [4B 权重](https://huggingface.co/Qwen/Qwen3.5-4B) · [9B 权重](https://huggingface.co/Qwen/Qwen3.5-9B) · [LEMMO 论文第三节](https://arxiv.org/html/2611.09476v1) |
docs/MULTIMODAL_INTEGRATION_PLAN.md:28:## 五步与验收
docs/MULTIMODAL_INTEGRATION_PLAN.md:38:## 第一版演示该长什么样
docs/MULTIMODAL_INTEGRATION_PLAN.md:51:## 哪一部分值得投入研究时间
docs/MULTIMODAL_INTEGRATION_PLAN.md:53:视觉模型 BLIP-2 已经展示了冻结宿主、训练小型连接器的技术路线，LEMMO 已经研究指令驱动的原始 I/Q 解读。我们可以直接借用成熟的接入思路。值得投入研究时间的，是测出**冻结的波形表示为后来的追问留下了什么、哪些证据能消除歧义、实际宿主有没有把这部分信息用起来**。第一轮的具体交付应是同一批波形上的解析答案、共享表示答案和宿主答案，三者之间的差距会指出下一步该改哪里。[BLIP-2，Proceedings of the 40th International Conference on Machine Learning，2023](https://proceedings.mlr.press/v202/li23q.html) · [LEMMO，arXiv:2611.09476](https://arxiv.org/abs/2611.09476) · [EM-FloatFormer D 组证明](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/D/PROOF.md)
research/stage4_math_and_design\transcript_utils.py
research/stage4_math_and_design\discussions\M6\round1_events.jsonl
research/stage4_math_and_design\discussions\M6\prompt_r1.md
research/stage4_math_and_design\discussions\M5\round1_events.jsonl
research/stage4_math_and_design\discussions\M5\prompt_r1.md
research/stage4_math_and_design\discussions\M4\round1_events.jsonl
research/stage4_math_and_design\discussions\M4\prompt_r1.md
research/stage4_math_and_design\discussions\M2\prompt_r1.md
research/stage4_math_and_design\discussions\M2\round1_events.jsonl
research/stage4_math_and_design\discussions\M3\round1_events.jsonl
research/stage4_math_and_design\discussions\M3\prompt_r1.md
research/stage4_math_and_design\discussions\M1\round1_events.jsonl
research/stage4_math_and_design\discussions\M1\prompt_r1.md


    Ŀ¼: C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\matched_cache\weights


Mode                 LastWriteTime         Length Name                                                                 
----                 -------------         ------ ----                                                                 
-a----        2026/11/26     19:11           3789 equivariant_b8_s17.npz                                               


```

### Event #10: `item.completed` — `command_execution` (`item_7`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "uv --offline python list --only-installed
uv cache dir
Get-ChildItem -Force research/stage4_math_and_design -Name
rg -n 'EquivariantEvidenceConnector|RefSufficientPool|M1|M3|M4|M6' research/stage4_math_and_design/discussions -g 'prompt_r1.md'"
```

**Aggregated Output:**

```text
error: Failed to initialize cache at `C:\Users\李则徐\AppData\Local\uv\cache`
  Caused by: failed to open file `C:\Users\李则徐\AppData\Local\uv\cache\sdists-v9\.git`: 拒绝访问。 (os error 5)
C:\Users\李则徐\AppData\Local\uv\cache
discussions
M1
M2
M3
M4
M5
M6
__pycache__
transcript_utils.py
research/stage4_math_and_design/discussions\M6\prompt_r1.md:1:# Track M6 (Round 1): First-Principles Zero-Trial-and-Error Decision Theory, Frozen-Host Linearization & Optimal Compute Allocation
research/stage4_math_and_design/discussions\M6\prompt_r1.md:3:You are **Track M6**, added to our parallel mathematical design team specifically to answer a critical meta-scientific & mathematical engineering question from the user:
research/stage4_math_and_design/discussions\M6\prompt_r1.md:8:- **Your Writable Directory**: `research/stage4_math_and_design/M6/` ONLY.
research/stage4_math_and_design/discussions\M6\prompt_r1.md:9:- **FORBIDDEN Edits**: Do NOT modify or delete any files outside `research/stage4_math_and_design/M6/`.
research/stage4_math_and_design/discussions\M6\prompt_r1.md:53:   - Write `research/stage4_math_and_design/M6/verify_m6.py` using `uv run --python 3.12 ...`:
research/stage4_math_and_design/discussions\M6\prompt_r1.md:56:     - Verify exact Modal budget table arithmetic (`<= $10.00 < $12.88`) and save everything to `research/stage4_math_and_design/M6/results_m6.json`.
research/stage4_math_and_design/discussions\M6\prompt_r1.md:58:## 3. Deliverables Required from Track M6
research/stage4_math_and_design/discussions\M6\prompt_r1.md:59:1. `research/stage4_math_and_design/M6/THEOREM_M6.md`
research/stage4_math_and_design/discussions\M6\prompt_r1.md:60:2. `research/stage4_math_and_design/M6/verify_m6.py` (executed cleanly via `uv run`)
research/stage4_math_and_design/discussions\M6\prompt_r1.md:61:3. `research/stage4_math_and_design/M6/results_m6.json`
research/stage4_math_and_design/discussions\M6\prompt_r1.md:62:4. Crisp executive summary in your final message (`reply_r1.md`) giving the exact Stage 3D DPI numbers, Tier-0/1/2 zero-waste checklist, exact compute-cost reduction stats, and pointed meta-architectural questions for M1–M5 Round 2.
research/stage4_math_and_design/discussions\M1\prompt_r1.md:1:# Track M1 (Round 1): Group-Theoretic Phase-Coupled Equivariant Architecture & Extrapolation Theory
research/stage4_math_and_design/discussions\M1\prompt_r1.md:3:You are **Track M1**, part of a 4-track mathematical & architectural multi-agent collaboration preparing **EM-FloatFormer Stage 4** (and fixing the root mathematical bottlenecks observed in **Stage 3D**).
research/stage4_math_and_design/discussions\M1\prompt_r1.md:7:- **Your Writable Directory**: `research/stage4_math_and_design/M1/` ONLY.
research/stage4_math_and_design/discussions\M1\prompt_r1.md:8:- **FORBIDDEN Edits**: Do NOT modify or delete any files outside `research/stage4_math_and_design/M1/` (especially do NOT touch `research/matched_cache`, `research/stage3_ministral_cache`, `research/stage3b_math`, `research/stage3c`, `research/stage3d_qwen_signed_phase`).
research/stage4_math_and_design/discussions\M1\prompt_r1.md:43:3. **Exact Stage 4 Equivariant + Evidence-Coupled Connector Architecture (`EquivariantEvidenceConnector`)**:
research/stage4_math_and_design/discussions\M1\prompt_r1.md:48:   - Explicitly specify clean interfaces for **Track M3 (Q2/Q3_after_pilot `RefSufficientPool`)** and **Track M4 (Real Pluto/CTTC CFO+SNR inputs)**!
research/stage4_math_and_design/discussions\M1\prompt_r1.md:50:   - Write `research/stage4_math_and_design/M1/verify_m1.py` (runnable strictly with `uv run --python 3.12 ...`).
research/stage4_math_and_design/discussions\M1\prompt_r1.md:61:## 3. Deliverables Required from Track M1
research/stage4_math_and_design/discussions\M1\prompt_r1.md:62:1. `research/stage4_math_and_design/M1/THEOREM_M1.md` (comprehensive, mathematically self-contained document with formal Theorems, Proofs, Stage 3D diagnostics, Stage 4 Connector Spec, and interface contract for M3/M4).
research/stage4_math_and_design/discussions\M1\prompt_r1.md:63:2. `research/stage4_math_and_design/M1/verify_m1.py` (executed by you via `uv run` with zero errors).
research/stage4_math_and_design/discussions\M1\prompt_r1.md:64:3. `research/stage4_math_and_design/M1/results_m1.json` (generated by `verify_m1.py`, recording all verified tables, eigenvalues/ranks, equivariance diffs, `G1..G4` extrapolation metrics, and `Pf` verification numbers).
research/stage4_math_and_design/discussions\M1\prompt_r1.md:65:4. A crisp summary in your final message stating exact theorems proved, key numbers from `results_m1.json`, and explicit design/interface questions for M2/M3/M4 Round 2.
research/stage4_math_and_design/discussions\M3\prompt_r1.md:1:# Track M3 (Round 1): Minimal Bayesian Sufficiency for Variable Pilot Multisets (Q2 / Q3_after_pilot) & Idempotent Pooling
research/stage4_math_and_design/discussions\M3\prompt_r1.md:3:You are **Track M3**, part of a 4-track mathematical & architectural multi-agent collaboration preparing **EM-FloatFormer Stage 4** (and diagnosing Stage 3D's Q2 reference-fusion collapse).
research/stage4_math_and_design/discussions\M3\prompt_r1.md:7:- **Your Writable Directory**: `research/stage4_math_and_design/M3/` ONLY.
research/stage4_math_and_design/discussions\M3\prompt_r1.md:8:- **FORBIDDEN Edits**: Do NOT modify or delete any files outside `research/stage4_math_and_design/M3/`.
research/stage4_math_and_design/discussions\M3\prompt_r1.md:52:3. **Design Stage 4's Fixed-Slot `RefSufficientPool` Operator**:
research/stage4_math_and_design/discussions\M3\prompt_r1.md:53:   - Specify an exact, deterministic, differentiable module `RefSufficientPool` (plus compatibility interface with Track M1's `EquivariantEvidenceConnector`) guaranteeing:
research/stage4_math_and_design/discussions\M3\prompt_r1.md:60:   - Write `research/stage4_math_and_design/M3/verify_m3.py` using `uv run --python 3.12 ...`:
research/stage4_math_and_design/discussions\M3\prompt_r1.md:67:     - Save all outputs to `research/stage4_math_and_design/M3/results_m3.json`.
research/stage4_math_and_design/discussions\M3\prompt_r1.md:69:## 3. Deliverables Required from Track M3
research/stage4_math_and_design/discussions\M3\prompt_r1.md:70:1. `research/stage4_math_and_design/M3/THEOREM_M3.md`
research/stage4_math_and_design/discussions\M3\prompt_r1.md:71:2. `research/stage4_math_and_design/M3/verify_m3.py` (executed cleanly via `uv run`)
research/stage4_math_and_design/discussions\M3\prompt_r1.md:72:3. `research/stage4_math_and_design/M3/results_m3.json`
research/stage4_math_and_design/discussions\M3\prompt_r1.md:73:4. Crisp executive summary in your final message (`reply_r1.md`) reporting exact Stage 3D diagnostics, theorems proved, `verify_m3.py` benchmarks, and interface questions for M1/M2/M4 Round 2.
research/stage4_math_and_design/discussions\M5\prompt_r1.md:41:       (b) Every Stage 4→5→6 connector (`EquivariantEvidenceConnector` + `RefSufficientPool` + Stage 5 event head + Stage 6 EIG selector) works identically across `Qwen3.5-4B`, `Qwen3.5-9B`, and `Qwen3.8-27B` with ZERO architectural rewrite!
research/stage4_math_and_design/discussions\M5\prompt_r1.md:57:4. Crisp executive summary in your final message (`reply_r1.md`) reporting exact rate-distortion/EIG theorems, `s17` vs future-proof canonical cache quantitative comparison, and concrete design recommendations for M1/M3/M4/M6 Round 2 so Stage 4 builds directly toward Stage 6 with zero dead ends.
research/stage4_math_and_design/discussions\M2\prompt_r1.md:57:4. Crisp executive summary in your final message (`reply_r1.md`) reporting exact decomposed Stage 3D numbers, the arithmetic vs geometric `Pf` finding, and exact gate/sample-size recommendations for M1/M3/M4 Round 2.
research/stage4_math_and_design/discussions\M4\prompt_r1.md:1:# Track M4 (Round 1): Real-Data Distribution-Shift Bounds, Pluto/CTTC Benchmark Adaptation & Modal Free-Tier Architecture
research/stage4_math_and_design/discussions\M4\prompt_r1.md:3:You are **Track M4**, part of a 4-track mathematical & architectural multi-agent collaboration preparing **EM-FloatFormer Stage 4** (`docs/LONG_TERM_RESEARCH_PLAN.md` Stage 4: real SDR data validation on ADALM-PLUTO 850 MHz + CTTC 433/708/2450 MHz paired with frozen Qwen3.5-4B / fixed encoder alignment).
research/stage4_math_and_design/discussions\M4\prompt_r1.md:7:- **Your Writable Directory**: `research/stage4_math_and_design/M4/` ONLY.
research/stage4_math_and_design/discussions\M4\prompt_r1.md:8:- **FORBIDDEN Edits**: Do NOT modify or delete any files outside `research/stage4_math_and_design/M4/`. (To inspect real-data code on remote git branches, use read-only `git show origin/research/...` — do NOT check out other branches or alter git HEAD!)
research/stage4_math_and_design/discussions\M4\prompt_r1.md:52:     - **Ablation Design on Real Data**: what exact baselines must run on the IDENTICAL real-data test tuples? (Physical Bayes / `B3_mixture` ceiling vs Direct `s17` table vs Frozen LLM `Zero`-arm vs `Naive-Additive` vs `Stage4-Equivariant-Coupled` vs `Stage4+RefSufficientPool`!).
research/stage4_math_and_design/discussions\M4\prompt_r1.md:57:   - Write `research/stage4_math_and_design/M4/verify_m4.py` using `uv run --python 3.12 ...`:
research/stage4_math_and_design/discussions\M4\prompt_r1.md:63:## 3. Deliverables Required from Track M4
research/stage4_math_and_design/discussions\M4\prompt_r1.md:64:1. `research/stage4_math_and_design/M4/THEOREM_M4.md`
research/stage4_math_and_design/discussions\M4\prompt_r1.md:65:2. `research/stage4_math_and_design/M4/verify_m4.py` (executed cleanly via `uv run`)
research/stage4_math_and_design/discussions\M4\prompt_r1.md:66:3. `research/stage4_math_and_design/M4/results_m4.json`
research/stage4_math_and_design/discussions\M4\prompt_r1.md:67:4. Crisp executive summary in your final message (`reply_r1.md`) reporting exact misspecification bounds, recommended `s17` real-data mapping & zero-leakage tuple protocol, exact Modal budget math, and interface questions for M1/M2/M3 Round 2.
```

### Event #11: `item.completed` — `command_execution` (`item_9`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "[Console]::OutputEncoding=[Text.Encoding]::UTF8
Get-Content -Encoding utf8 research/stage3d_qwen_signed_phase/REPORT.md | Select-Object -First 44
Get-Content -Encoding utf8 research/stage3d_qwen_signed_phase/data_stage3d.py | Select-Object -First 280
rg -n 'fit\\(|C1|12\\.2|0\\.012|0\\.504|full_iq' research/matched_cache -g '*.py' -g '*.md'"
```

**Aggregated Output:**

```text
�޷��������ԡ�������ģʽ��֧�ֺ������͵��������á�
����λ�� ��:2 �ַ�: 1
+ [Console]::OutputEncoding=[Text.Encoding]::UTF8
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

"""Deterministic Stage 3D data generators, disjoint seed namespaces, and zero-leakage Chinese prompts."""
from dataclasses import dataclass, replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from research.stage3_ministral_cache.data import MotherBatch, generate_mothers, orbit_key
from research.stage3_ministral_cache.schema import (
    CONDITION_SPECS,
    ConditionSpec,
    N_SAMPLES,
    Q1_CHOICES,
    Q1_PROMPT,
    Q2_CHOICES,
    Q2_PROMPT,
    Q3_CHOICES,
    Q3_PROMPT_INDEP,
    Q3_PROMPT_REPEAT,
    Q3_PROMPT_REPEAT_AFTER_PILOT,
    ROOTS,
    ROTATIONS,
    Reference,
    dedup_references,
    matched_sum,
    sector_of_complex,
)


EQUIVARIANT_B8_S17_SHA256 = "e67d9fbafb90791ab19549de7789798d818286d7f52d7ac566cc39df71e78356"
ARM_NAMES: Tuple[str, ...] = ("phase", "zero")
CONNECTOR_SEEDS: Tuple[int, ...] = (202611, 202622, 202633)
REPLAY_TRAIN_CONDITIONS: Tuple[str, ...] = ("C1_coherent_2db", "C2_discrete_walk", "C3_mid_pilot_cont")
TRAIN_ARMS: Tuple[str, ...] = ("none", "valid", "independent", "fresh_resample")

# Disjoint seed namespaces (strictly isolated from Stage 3 s3_train/s3_dev/s3_test and Stage 3C work_e 316027)
NS_CFO_TRAIN = 930101
NS_CFO_DEV = 930202
NS_DEV_PILOT = 940202
NS_BLIND_G1 = 950401
NS_BLIND_G2 = 950402
NS_BLIND_G3 = 950403
NS_BLIND_G4 = 950404
NS_BLIND_Q2_C4 = 950405
NS_REF_CALIB = 960501

TRAIN_TIMINGS: Tuple[Tuple[int, int, float], ...] = ((2, 2, 4.0), (2, 3, 1.0))
TRAIN_OMEGAS: Tuple[float, ...] = (-0.25, -0.10, 0.0, 0.10, 0.25)
DEV_OMEGAS: Tuple[float, ...] = (-0.18, 0.18)

Q2_BLIND_ARMS: Tuple[str, ...] = (
    "none",
    "valid",
    "duplicate_copy_dedup",
    "duplicate_copy_raw",
    "independent_only",
    "valid_plus_independent",
    "fresh_resample",
)


@dataclass(frozen=True)
class BlindCellSpec:
    cell_id: str
    role: str
    abs_omega: float
    h: int
    ell: int
    t_p: float
    snr_db: float
    sigma_phi: float
    namespace: int
    default_blocks: int

    @property
    def total_delay(self) -> int:
        return int(self.h + self.ell)

    @property
    def cum_angle_abs(self) -> float:
        return float(self.abs_omega * (self.h + self.ell))

    @property
    def cum_diffusion_var(self) -> float:
        return float((self.sigma_phi ** 2) * (self.h + self.ell))


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
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    matched_dir = root / "research" / "matched_cache"
    weight_path = matched_dir / "weights" / "equivariant_b8_s17.npz"
    digest = hashlib.sha256(weight_path.read_bytes()).hexdigest()
    if digest != EQUIVARIANT_B8_S17_SHA256:
        raise RuntimeError(f"s17 SHA-256 mismatch: {digest} != {EQUIVARIANT_B8_S17_SHA256}")
    if str(matched_dir) not in sys.path:
        sys.path.insert(0, str(matched_dir))
    import learned  # type: ignore

    return learned.load(weight_path), digest


def public_phase_slots(omega: float, h: int, ell: int, t_p: float, zero_arm: bool = False) -> np.ndarray:
    """Six public phase slots at t = (h, h + ell, t_p) relative to t1 = 0."""
    if zero_arm:
        return np.zeros(6, dtype=np.float32)
    times = (float(h), float(h + ell), float(t_p))
    return np.array([v for t in times for v in (math.cos(omega * t), math.sin(omega * t))], dtype=np.float32)


def condition_spec_from_params(
    omega: float,
    h: int,
    ell: int,
    t_p: float,
    snr_db: float = 2.0,
    sigma_phi: float = 0.10,
    tag: str = "cfo",
) -> ConditionSpec:
    return ConditionSpec(
        name=tag,
        snr_db=float(snr_db),
        h=int(h),
        ell=int(ell),
        t_p=float(t_p),
        drift_type="cfo_continuous",
        d=0.0,
        sigma_phi=float(sigma_phi),
        omega=float(omega),
        in_support_encoder=False,
        in_support_connector=False,
    )


def format_public_metadata(cond: ConditionSpec) -> str:
    """Zero-leakage public metadata string (excludes condition names, split names, labels, and noise)."""
    if cond.drift_type == "coherent":
        drift_desc = "��ɾ�ֹ(omega=+0.00rad/��,sigma=0.00rad/�̲�,d=0.00/��)"
    elif cond.drift_type == "discrete":
        drift_desc = f"��ɢC4����(omega=+0.00rad/��,sigma=0.00rad/�̲�,d={cond.d:.2f}/��)"
    else:
        drift_desc = f"����Ƶƫ�벼��Ư��(omega={cond.omega:+.2f}rad/��,sigma={cond.sigma_phi:.2f}rad/�̲�,d=0.00/��)"
    return (
        f"�����={cond.snr_db:.1f}dB; "
        f"�������h={cond.h}, �������ell={cond.ell}, ��Ƶʱ��tp={cond.t_p:.1f}; "
        f"�ŵ�={drift_desc}��\n"
        f"�ѱ��벨���빫����λ������<W1><W2><PHASE>��"
    )


def build_stage3d_prompt(
    question_type: str,
    cond: ConditionSpec,
    protocol: str = "repeat_x1",
    references: Sequence[Reference] = (),
    deduplicate: bool = True,
) -> Tuple[str, Tuple[str, ...]]:
    """Construct exact Chinese prompt listing all four candidate choices without any label/ID leakage."""
    if question_type not in ("Q1", "Q2", "Q3", "Q3_after_pilot"):
        raise ValueError(f"Unknown question_type={question_type}")

    meta = format_public_metadata(cond)
    ref_block = ""
    if question_type in ("Q2", "Q3_after_pilot"):
        if len(references) == 0:
            ref_block = "\n����ο����޲ο��۲⡣"
        elif deduplicate:
            seen_ids = set()
            items = []
            for r in references:
                if r.acquisition_id in seen_ids:
                    continue
                seen_ids.add(r.acquisition_id)
                tag = "ͬ�ŵ�" if r.shares_channel else "�����޹�"
                items.append(f"[���={r.acquisition_id},״̬={tag},ʱ��={r.t_p:.1f}]<REF>")
            ref_block = "\n����ο���" + "��".join(items) + f"���Ѱ�Ψһ�ɼ����ȥ�أ���{len(items)}�������ο�����"
        else:
            items = []
            for r in references:
                tag = "ͬ�ŵ�" if r.shares_channel else "�����޹�"
                items.append(f"[���={r.acquisition_id},״̬={tag},ʱ��={r.t_p:.1f}]<REF>")
            ref_block = f"\n����ο���δȥ�أ���{len(items)}������" + "��".join(items) + "��"

    if question_type == "Q1":
        qtext, choices = Q1_PROMPT, Q1_CHOICES
    elif question_type == "Q2":
        qtext, choices = Q2_PROMPT, Q2_CHOICES
    elif question_type == "Q3":
        qtext = Q3_PROMPT_REPEAT if protocol == "repeat_x1" else Q3_PROMPT_INDEP
        choices = Q3_CHOICES
    else:
        qtext, choices = Q3_PROMPT_REPEAT_AFTER_PILOT, Q3_CHOICES

    listed = "\n".join(choices)
    prompt = (
        f"{meta}{ref_block}\n"
        f"���⣺{qtext}\n"
        f"�ĸ���ѡѡ�������˳�򣩣�\n{listed}\n"
        f"��ֱ�Ӵ������������������ܵ�һ������ѡ�����֡�"
    )
    return prompt, choices


def derive_block_seeds(namespace: int, cell_idx: int, blocks: int) -> List[int]:
    """Deterministically derive unique 32-bit block seeds from (namespace, cell_idx, block_id)."""
    return [
        int(np.random.SeedSequence([int(namespace), int(cell_idx), int(b)]).generate_state(1)[0])
        for b in range(int(blocks))
    ]


def generate_paired_cfo_blocks(
    namespace: int,
    cell_idx: int,
    blocks: int,
    h: int,
    ell: int,
    t_p: float,
    omegas: Sequence[float],
    snr_db: float = 2.0,
    sigma_phi: float = 0.10,
    split_prefix: str = "cfo",
research/matched_cache\learned.py:131:def fit(windows, soft_targets, bits, seed, equivariant):
research/matched_cache\train.py:36:    fitted = fit(windows, targets, bits, seed, equivariant=variant == "equivariant")
research/matched_cache\README.md:30:8 比特带旋转约束模型相对同预算手工模型的第一问正确率差为 −0.39 个百分点（按母本配对的点态 95% 区间 −1.27 至 +0.49），第二问为 +0.36 个百分点（−0.49 至 +1.20）。两问的对数损失差分别为 −0.0128（−0.0368 至 +0.0106）和 −0.0013（−0.0190 至 +0.0158）。区间针对这三个既定种子的平均表现。数据支持两者都能守住旋转关系，准确率的差异方向仍不确定。[完整比较与配对区间](results/summary.json)
```

### Event #12: `item.completed` — `command_execution` (`item_8`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '[Console]::OutputEncoding=[Text.Encoding]::UTF8
Get-Content -Encoding utf8 docs/LONG_TERM_RESEARCH_PLAN.md | Select-Object -First 77
Get-Content -Encoding utf8 docs/MULTIMODAL_INTEGRATION_PLAN.md | Select-Object -First 53'
```

**Aggregated Output:**

```text
�޷��������ԡ�������ģʽ��֧�ֺ������͵��������á�
����λ�� ��:2 �ַ�: 1
+ [Console]::OutputEncoding=[Text.Encoding]::UTF8
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
```

### Event #13: `item.completed` — `command_execution` (`item_10`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '12\\.2|0\\.504|0\\.012|G1_main|2 dB|C1' research/stage3d_qwen_signed_phase/REPORT.md research/matched_cache/train.py research/matched_cache/README.md
rg -n '"'^'"\\|[[:space:]]*\\*\\*?[456]|冻结|盲测|Pluto|CTTC' docs/LONG_TERM_RESEARCH_PLAN.md docs/MULTIMODAL_INTEGRATION_PLAN.md | Select-Object -Last 25"
```

**Aggregated Output:**

```text
research/stage3d_qwen_signed_phase/REPORT.md:20:| 同期同母波形 `s17` CPU 物理读出 | **64.62%** (`0.64617`) | **0.9411** (`0.94113`) | `0.5098` | `0.0009` | 真实连接层相对 CPU 物理读出差距：`phase −40.82 pp / -0.5041 nats`；`zero −45.21 pp / -0.5398 nats` |
research/stage3d_qwen_signed_phase/REPORT.md:97:- **训练支持集**：延续 Stage 3 的 `C1 / C2 / C3` 回放（每条件 512 母波形）并按 Work F 加入有符号 CFO 监督：$|\omega| \in \{0, 0.10, 0.25\}$，时延组合 $(h, \ell, t_p) \in \{(2, 2, 4), (2, 3, 1)\}$（训练最大累计转角 $0.25 \times 5 = 1.25\text{ rad}$），每条件 512 母波形（开发集对应 $|\omega| = 0.18$ 及 `C1/C2/C3`，每条件 128 母波形）。
research/stage3d_qwen_signed_phase/REPORT.md:124:在保持每格 `SNR = 2 dB, σ = 0.10`、每条母波形评估 `(±ω) × 3 种旋转 × 3 套种子` 下，完整 `2×2`（单步频偏幅度 `|ω|` × 累计转角 `|ω|(h+ℓ)`）盲测结果如下表（同步报告累计扩散方差 `σ²t`）：
research/stage3d_qwen_signed_phase/REPORT.md:128:| **`G1_main_extrap_both`** | **主检验（双外推）** | `0.40` | `(2, 2, 4)` | **`1.60 rad`** | `0.04` | `131 / 8,384` | `23.80% / 1.4452` | `19.41% / 1.4809` | **`+4.39 pp`** `[+4.11, +4.68]` (`[+4.11, +4.67]`) | **`+0.0357`** `[+0.0330, +0.0383]` (`[+0.0330, +0.0383]`) | `64.62% / 0.9411` (`64.64% / 0.9290`) |
research/stage3d_qwen_signed_phase/REPORT.md:155:- **按各种子细察**：在 `±0.40` 下 `phase` 三套种子的方向增益分别为 `-0.0015 (s202611), +0.0210 (s202622), +0.0099 (s202633)`；`zero` 为 `+0.0018, +0.0129, +0.0004`。此外在均方效应量上，$W^2 \approx 0.00112$ 远超 $M^2 \approx 0.000185$（物理参考为 $M^2 = 0.2016$），表明**宿主仍几乎只按编码波形状态而非频偏符号转移概率**；但即便微弱，`phase` 臂的元数据响应 $M^2$ 确实约为 `zero` 臂的 **1.9 倍**。
research/stage3d_qwen_signed_phase/REPORT.md:216:### 6.3 旧工况 `C1 / C2 / C3` 回放未遗忘验证（`s3_test` × 2 区块 × 3 旋转 = 384 条 × 3 种子）
research/stage3d_qwen_signed_phase/REPORT.md:220:| `C1_coherent_2db` | `Q1 / Q3 / Q2_valid` | `26.48%/1.3764` · **`39.84%/1.2510`** · `36.07%/1.3333` | `29.86%/1.3724` · **`50.09%/1.1876`** · `44.01%/1.2889` | `67.19%/0.8332` · `64.84%/0.9243` · `67.97%/0.8781` |
research/stage3d_qwen_signed_phase/REPORT.md:224:可见回放集未崩溃，且与 `G2/G4` 一致：`zero` 臂在多任务支持集拟合较 `phase` 更深（如 `C1 Q3: 50.09%` vs `39.84%`），再次印证 **6 个连续数值槽在有限 1,280 步微调内略微分散了标准工况上的特征对齐**。
research/stage3d_qwen_signed_phase/REPORT.md:232:- **已全部完成并核验**：Qwen3.5-4B 完整 Smoke 与 Tokenizer 审计、显存与引用环消除基准测试、3 种子 × 2 臂（6 次）配对训练与开发集温度校准、`dev_pilot`（`B_locked=131`）与提交前锁定（`Commit #1`）、全量配对盲测（`G1` + `G2..G4` + 历史 `±0.35`）、Work F（含单模态矛盾探针与 `check_f.py`）、Work G（含离线 `Pf/F0` 严格分离）、Work A Q2 七臂与 `q2_reader.read` 对照、`C1/C2/C3` 遗忘回放、`MANIFEST.sha256` 及干净目录零误差全量重算验证（`max_abs_diff = 0.000e+00`）。
research/stage3d_qwen_signed_phase/REPORT.md:238:1. **指定物理模拟器**：QPSK 四相脉冲、`N_SAMPLES = 4`、`SNR = 2 dB`（及各条件固定 `d / σ_φ`）、连续布朗相位随机游走加线性频偏。
research/matched_cache/README.md:7:第一问读取两个符号的相对相位，第二问在揭示参考后恢复第一符号。两道题、2 dB 信号模型、有效/独立/重复/缺省参考和母本划分均继承[上轮配对基准](../fourphase_paired_questions/README.md)。训练、开发、测试母本数分别为 4,096、256、1,024。每条测试母波形派生三个窗口版本，每个版本配四种参考。12,288 组记录中，同一母本的记录按配对样本处理。独立审计核对了输入样本及其全部四相旋转，训练、开发、测试间未发现复用。
research/matched_cache/README.md:30:8 比特带旋转约束模型相对同预算手工模型的第一问正确率差为 −0.39 个百分点（按母本配对的点态 95% 区间 −1.27 至 +0.49），第二问为 +0.36 个百分点（−0.49 至 +1.20）。两问的对数损失差分别为 −0.0128（−0.0368 至 +0.0106）和 −0.0013（−0.0190 至 +0.0158）。区间针对这三个既定种子的平均表现。数据支持两者都能守住旋转关系，准确率的差异方向仍不确定。[完整比较与配对区间](results/summary.json)
docs/MULTIMODAL_INTEGRATION_PLAN.md:19:| **4B�?B · Qwen3.5** | 2026-03-02 | 4B 是首个领域对照。已有原�?I/Q 研究 LEMMO 使用同款 4B 语言底座并用 LoRA 适配。优先在相同长度、相同问题和相同可见参考下比较。若要归因到编码方法，还要匹配训练数据、可训练参数和优化预算�?B �?9B 可比较同代规模趋势�?[4B 权重](https://huggingface.co/Qwen/Qwen3.5-4B) · [9B 权重](https://huggingface.co/Qwen/Qwen3.5-9B) · [LEMMO 论文第三节](https://arxiv.org/html/2611.09476v1) |
docs/MULTIMODAL_INTEGRATION_PLAN.md:26:正式比较统一冻结电磁编码器，复用相同�?I/Q 缓存、导频、中文追问、留出采集和计分规则，每个宿主训练自己的连接层。规模对照使用相同精度，另记显存与推理时间。对每道题记录单独改变波形或补入参考后，答案是否按物理预期变化。Qwen3.5-9B �?Qwen3.8-27B 的差异同时包含规模和代际训练差异。若研究重点是同代规模趋势，可以另加 Qwen3.5-27B，与 4B�?B 构成一组。[Qwen3.5-27B 权重](https://huggingface.co/Qwen/Qwen3.5-27B)
docs/MULTIMODAL_INTEGRATION_PLAN.md:33:| **2　把波形变成可复用的输�?* | 编码器接�?I、Q 采样和采样率等可用元数据，输出按时间排列的特征向量，每个向量都能对应回一段原始采样。同一波形的编码对不同问题共用。保存缓存预算、量化精度、时间映射和有效观测掩码�?| 先用简单的读出器检查，同一份冻结表示能回答相对相位，也能在补入有效参考后恢复原始相位。以完整 I/Q 和解析相量为参照。比较逐窗丢弃相位后，哪道题先失效�?|
docs/MULTIMODAL_INTEGRATION_PLAN.md:34:| **3　与一个已有模型对�?* | 在提问中为每段波形预留位置，用小型可训练投影层把波形向量映射到宿主词向量的维数，连同文字进入注意力层。首轮冻结已验证的波形编码器和宿主，用波形、问题、答案三元组训练投影层。新一轮追加参考向量时复用原表示�?| 在未用于训练的合成波形上，模型能完成“先回答、再追问、补参考后更新”。平衡不同答案的出现频率，同一句问法配不同波形。只给文字、打乱波形配对或只保留幅度时，逐题核对预期变化。交付一个能实际对话的单宿主原型�?|
docs/MULTIMODAL_INTEGRATION_PLAN.md:35:| **4　在真实波形上验收** | 用公开已知序列定义真值，优先�?Pluto 连续 QPSK 开发，在完整留出的 CTTC 会话或另一次物理采集上测试。在相同片段上对比解析读出、从冻结向量直接读出、宿主语言回答。加入时间轴图与 I/Q 共同决定答案的测试�?| 分别报告自然录制和人工相位干预。记录答案正确率、候选概率、补证前后改对和改错的样例，以及定位证据的时间误差。普通文本与图像问题也跑一组成对回归。若表示可被直接读对而语言回答错，先修对齐；若两者都错，用完�?I/Q 回查输入信息与相干假设�?|
docs/MULTIMODAL_INTEGRATION_PLAN.md:36:| **5　迁移到第二个宿主** | 固定同一波形编码器、预处理、时间坐标、权重和问题集。为另一个模型训练专属的小投影层，用各自的对话模板组织上下文�?| 两个宿主都通过同一套物理干预与实测题，报告各自的误差、每段波形占用的上下文及推理开销。发布共享编码器、一份输入规范及各自的适配权重。结论明确限定在测试过的宿主和波形上�?|
docs/MULTIMODAL_INTEGRATION_PLAN.md:53:视觉模型 BLIP-2 已经展示了冻结宿主、训练小型连接器的技术路线，LEMMO 已经研究指令驱动的原�?I/Q 解读。我们可以直接借用成熟的接入思路。值得投入研究时间的，是测�?*冻结的波形表示为后来的追问留下了什么、哪些证据能消除歧义、实际宿主有没有把这部分信息用起�?*。第一轮的具体交付应是同一批波形上的解析答案、共享表示答案和宿主答案，三者之间的差距会指出下一步该改哪里。[BLIP-2，Proceedings of the 40th International Conference on Machine Learning�?023](https://proceedings.mlr.press/v202/li23q.html) · [LEMMO，arXiv:2611.09476](https://arxiv.org/abs/2611.09476) · [EM-FloatFormer D 组证明](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/D/PROOF.md)
docs/LONG_TERM_RESEARCH_PLAN.md:15:**最终盲测�?* 先编码，后揭题。两种解释均与旧波形相容时，只允许再读取一个已录制的候选参考窗。模型先选窗口并写下各解释预期的观测，再揭示真实波形并评分。对照包括随机窗、最近窗以及由物理模型预测能最大程度减少歧义的窗。提前约定主要指标、可接受的误差差距和采集划分。等缓存与观测预算时，比较答案风险、置信度和证据定位；达到约定准确度时，比较所需缓存和额外读取量。天然观测、人工注入和独立采集分别报告。若现有方法打平，保留已验证的边界与反例，继续寻找下一个确有差异的问题�?
docs/LONG_TERM_RESEARCH_PLAN.md:17:完整交付包含可审查的数学结果、带解析真值和独立采集的盲测集、可接入多个已有 AI 的电磁编码器，以及一段支持多轮追问的真实录制演示。[3B�?7B 宿主与模型接入计划](MULTIMODAL_INTEGRATION_PLAN.md)
docs/LONG_TERM_RESEARCH_PLAN.md:30:|四条实测分支、三种数据来�?       |435 MHz 两条分支使用同一�?742 �?16-QAM OFDM 录制�?38 帧和 240 帧是不同划分。Pluto 提供连续时域 QPSK，CTTC 提供多会�?QPSK。另有宽�?Wi-Fi CSI 文件的格式和重复性审计�?                                                                                                          |在不同信号输入层上分别验证可回答的问题；划分时以会话或物理录制为单位。宽带数据盘点供将来选题�?[435 MHz OFDM](https://github.com/20171806011/EM-FloatFormer/blob/ba39dd5968dbaee7b7a226a9bc78d536278fdee8/research/realdata_sdr_ofdm/REPORT.md) · [压力实验](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/REPORT.md) · [Pluto](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/REPORT.md) · [CTTC](https://github.com/20171806011/EM-FloatFormer/blob/1887aee6de9649bc0dd8a4f05fe3c4542f3f6d77/research/realdata_cttc_qpsk/REPORT.md) · [宽带样本审计](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/DATASET_DISCOVERY.md)|
docs/LONG_TERM_RESEARCH_PLAN.md:50:|2 · �?4�? 周，建立小型判别实验       |“哪一段观测支持我的判断？去掉它，哪一个答案会变？�?                  |用共享母波形制作有效参考、独立参考、重复同一参考和无参考四类配对。给解析方法、只有幅度或逐窗归一化的方法、保留复相位的方法完全相同的输入权限。平衡信噪比、窗位、能量和编号。用精确枚举或可控似然验证理论，冻结问题、划分和评分。输出可�?CPU 复跑的小基准�?                                     |
docs/LONG_TERM_RESEARCH_PLAN.md:51:|3 · �?2�? 个月，共享编码器与首次模型接�?   |“同一份表示能答相对关系、预测留出观测，并在参考到来后重新作答吗？�?          |用有限预算的一个复数编码器配几个简单读出，与解析相�?多假设后验、同预算实数网络等对照。冻结表示后测试新参考位置、相位轨迹与未见问题组合，记录每题相对参照的性能差距和保留信息的成本。先�?Ministral 3B，再�?Qwen3.5-4B。冻结编码器与宿主，为每个宿主训练连接层，按相同问题分别测直接读出和语言回答�?                                  |
docs/LONG_TERM_RESEARCH_PLAN.md:52:|4 · �?3�? 个月，在真实录制上兑现一个预�? |“在陌生的一次采集中，哪段已知波形是有效参考？揭示后，先前的预测应怎样修正？�?     |先用已有 Pluto �?CTTC 的已知序列遮住—揭示，A 次采集设参数，B 次采集检验。CTTC 可留整会话；Pluto 可检查连续时域窗�?35 MHz 网格可检验同题的 OFDM 版本。对缺少真值的问题使用留出导频预测或公开序列定义答案。分别汇报自然录制、基于实测的人工干预与独立来源。交付能从原输入重放的至少一项配对外测。|
docs/LONG_TERM_RESEARCH_PLAN.md:64:|Pluto 850 MHz 连续 QPSK�?,096 帧，每帧 7,560 个过采样复样点，已做匹配滤波|用源数据中可核对的参考序列，留出一个窗预测其相位；揭示该窗后，判断另一个窗的预测是否改善。检验绝对帧号和固定重复周期能解释多少结果。|能检验时域相干性、重复结构及参考更新；跨设备普适性需要其他采集�?                                                                                              |
docs/LONG_TERM_RESEARCH_PLAN.md:65:|CTTC 三频段、三档名义信噪比、九�?QPSK 会话，含原始 ADC 与官方匹配滤波记�?       |整个会话只作测试。在对齐成功的片段上，比较原始与滤波后视图对同一关系问题的回答，以及哪些细节在处理后消失�?            |两种视图来自同一次采集；以会话和来源汇总。已有研究是会话内划分�?                                                                                              |
docs/LONG_TERM_RESEARCH_PLAN.md:68:不请模型回答没有物理定义的问题。例如“两个窗口是否相同发射机”需要发射机身份真值与恰当观测；本计划先问“是否符合一条共享相位轨迹”，并说明替代模型。对一枚外观完全一样、来源又隐藏的错配导频，模型能有多确定取决于相关性线索与错配先验。已知独立参考和隐藏可靠性的参考要分别建模、分别评分。[435 MHz OFDM](https://github.com/20171806011/EM-FloatFormer/blob/ba39dd5968dbaee7b7a226a9bc78d536278fdee8/research/realdata_sdr_ofdm/REPORT.md) · [压力实验](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/REPORT.md) · [Pluto](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/REPORT.md) · [CTTC](https://github.com/20171806011/EM-FloatFormer/blob/1887aee6de9649bc0dd8a4f05fe3c4542f3f6d77/research/realdata_cttc_qpsk/REPORT.md) · [宽带样本审计](https://github.com/20171806011/EM-FloatFormer/blob/5f88a5be984493bcb6aa164d84253b6f29876eef/research/realdata_ofdm_stress_and_theory/DATASET_DISCOVERY.md)
docs/LONG_TERM_RESEARCH_PLAN.md:86:|4 · 实测可行�? |查明 Pluto、CTTC 哪些参考序列、输入视图和会话可配对；435 MHz 记录可检哪类题|“问题—输入—真值—划分”表，加一个可复现样例标识|
docs/LONG_TERM_RESEARCH_PLAN.md:87:|5 · 评价     |固定问题句式和输出格式，制定匹配单窗统计的对照、会话留出与证据扰动计分方法          |所有方案都能使用的一份盲测协议草�?       |
docs/LONG_TERM_RESEARCH_PLAN.md:114:|`research/realdata-pluto-qpsk`                  |Pluto 连续 QPSK           |[ade8333f](https://github.com/20171806011/EM-FloatFormer/blob/ade8333fb68126251a83e484b69b821b5d97bcea/research/realdata_pluto_qpsk/REPORT.md)            |
docs/LONG_TERM_RESEARCH_PLAN.md:115:|`research/realdata-cttc-indoor-qpsk`            |CTTC 三频段九会话 QPSK        |[1887aee6](https://github.com/20171806011/EM-FloatFormer/blob/1887aee6de9649bc0dd8a4f05fe3c4542f3f6d77/research/realdata_cttc_qpsk/REPORT.md)             |
```

### Event #14: `item.completed` — `command_execution` (`item_11`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-ChildItem -Force research/stage4_math_and_design/M5
uv --cache-dir research/stage4_math_and_design/M5/.uv-cache --offline python list --only-installed
Get-Command python, py -All | Select-Object Name,Source
Get-ChildItem 'C:/Users/李则徐/AppData/Local/uv' -Name
Get-ChildItem 'C:/Users/李则徐/AppData/Local/uv/cache' -Name
Get-ChildItem Env: | Where-Object Name -Match '"'^(UV|PYTHON|VIRTUAL)'"'"
```

**Aggregated Output:**

```text
cpython-3.13.0-windows-x86_64-none    C:\Users\李则徐\AppData\Local\Programs\Python\Python313\python.exe
cpython-3.12.7-windows-x86_64-none    C:\qazwsx\python.exe
cpython-3.9.19-windows-x86_64-none    C:\376381\miniconda3\python.exe
cpython-3.9.19-windows-x86_64-none    C:\114514\python.exe

Name             Source                                                        
----             ------                                                        
python.exe       C:\qazwsx\python.exe                                          
python.exe       C:\114514\python.exe                                          
python.exe       C:\376381\miniconda3\python.exe                               
python.exe       C:\Users\������\AppData\Local\Microsoft\WindowsApps\python.exe
py.exe           C:\WINDOWS\py.exe                                             
cache
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
PYTHONIOENCODING                                                               
PYTHONUTF8                                                                     


```

### Event #15: `item.completed` — `command_execution` (`item_12`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "git status --short --untracked-files=all -- research/stage4_math_and_design/M5
rg -n '12\\.2|9290|0\\.5041' research/stage3d_qwen_signed_phase -g '*.md' -g '*.json' -m 2 | Select-Object -First 25
Get-Content research/matched_cache/train.py | Select-Object -First 80
rg -n 'pulse|SNR|COHERENT' research/matched_cache -g '*.py' | Select-Object -First 25"
```

**Aggregated Output:**

```text
research/stage3d_qwen_signed_phase\REPORT.md:20:| 同期同母波形 `s17` CPU 物理读出 | **64.62%** (`0.64617`) | **0.9411** (`0.94113`) | `0.5098` | `0.0009` | 真实连接层相�?CPU 物理读出差距：`phase �?0.82 pp / -0.5041 nats`；`zero �?5.21 pp / -0.5398 nats` |
research/stage3d_qwen_signed_phase\REPORT.md:21:| 同期同母波形 `full_iq` 已知参数全波形解 | **64.64%** (`0.64641`) | **0.9290** (`0.92904`) | `0.5032` | `0.0046` | 物理上限基线 |
research/stage3d_qwen_signed_phase\results\summary_master.json:238:          "nll": 0.9290379865758834,
research/stage3d_qwen_signed_phase\results\summary_master.json:255:          "mean": -0.5041027288195585,
research/stage3d_qwen_signed_phase\results\work_f_host_035_phase.json:2333:        0.18492169290921115,
research/stage3d_qwen_signed_phase\results\work_f_host_035_zero.json:383:        0.22506114243929018
research/stage3d_qwen_signed_phase\results\work_f_host_035_zero.json:1258:        0.28665676929098255,
"""Fit the registered encoder/codebook pairs from fresh, training-only mothers."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "fourphase_paired_questions"))
import paired

from learned import fit, load

SEEDS = (17, 29, 43)
BUDGETS = (4, 8)
VARIANTS = ("plain", "equivariant")


def training_data():
    # Register a new stream without altering published dev/test parameters.
    paired.SPLITS["train"] = (730241, 64)
    windows = np.asarray([window for _, _, pair, _, _ in paired.parents("train") for window in pair])
    targets = np.asarray([paired.likelihood(window) for window in windows])
    assert windows.shape == (8192, 3) and targets.shape == (8192, 4)
    return windows, targets


def name(variant, bits, seed):
    return f"{variant}_b{bits}_s{seed}"


def train_one(setting):
    variant, bits, seed, folder = setting
    windows, targets = training_data()
    fitted = fit(windows, targets, bits, seed, equivariant=variant == "equivariant")
    output = folder / (name(variant, bits, seed) + ".npz")
    fitted.save(output)
    restored = load(output)
    np.testing.assert_array_equal(restored.encode(windows), fitted.encode(windows))
    np.testing.assert_array_equal(restored.table, fitted.table)
    return output.name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "weights")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    settings = [(v, b, s, args.output) for v in VARIANTS for b in BUDGETS for s in SEEDS]
    with ProcessPoolExecutor(max_workers=args.jobs) as workers:
        for checkpoint in workers.map(train_one, settings):
            print(checkpoint, flush=True)


if __name__ == "__main__":
    main()
research/matched_cache\audit.py:102:    w = np.exp(-paired.SNR * np.sum(abs(y[:, :, None] - means)**2, axis=1))
research/matched_cache\evaluate.py:70:    logits = (2 * paired.SNR / np.sqrt(paired.N)) * (amplitude[:, None] * ROOTS.conj()).real
research/matched_cache\physical.py:10:SNR = 10 ** 0.2
research/matched_cache\physical.py:12:NOISE_SD = 1 / math.sqrt(2 * SNR)
```

### Event #16: `item.completed` — `command_execution` (`item_13`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-ChildItem 'C:/Users/李则徐/AppData/Local/uv/cache/wheels-v6' -Recurse -Filter '*2.3.5*' | Select-Object -First 12 -ExpandProperty FullName
Get-ChildItem 'C:/Users/李则徐/AppData/Local/uv/cache/wheels-v6' -Recurse -Filter '*1.17.0*' | Select-Object -First 12 -ExpandProperty FullName
Get-ChildItem C:/qazwsx/Lib/site-packages -Name | Select-String '"'^(numpy|scipy)'"'
Get-ChildItem -Force research/stage4_math_and_design/M5 -Name
uv --version"
```

**Aggregated Output:**

```text
C:\Users\������\AppData\Local\uv\cache\wheels-v6\pypi\numpy\2.3.5-cp312-cp312-win_amd64
C:\Users\������\AppData\Local\uv\cache\wheels-v6\pypi\numpy\2.3.5-cp312-cp312-win_amd64.http
C:\Users\������\AppData\Local\uv\cache\wheels-v6\pypi\numpy\2.3.5-cp312-cp312-win_amd64.msgpack
C:\Users\������\AppData\Local\uv\cache\wheels-v6\pypi\numpy\numpy-2.3.5-cp312-cp312-win_amd64.lock
C:\Users\������\AppData\Local\uv\cache\wheels-v6\pypi\scipy\1.17.0-cp312-cp312-win_amd64
C:\Users\������\AppData\Local\uv\cache\wheels-v6\pypi\scipy\1.17.0-cp312-cp312-win_amd64.http
C:\Users\������\AppData\Local\uv\cache\wheels-v6\pypi\scipy\1.17.0-cp312-cp312-win_amd64.msgpack
C:\Users\������\AppData\Local\uv\cache\wheels-v6\pypi\scipy\scipy-1.17.0-cp312-cp312-win_amd64.lock

numpy
numpy-1.26.4.dist-info
numpydoc
numpydoc-1.7.0.dist-info
scipy
scipy-1.13.1.dist-info
.uv-cache
uv 0.11.21 (5aa65dd7a 2026-06-11 x86_64-pc-windows-msvc)


```

### Event #17: `item.completed` — `command_execution` (`item_14`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$m5 = (Resolve-Path '"'research/stage4_math_and_design/M5').Path
"'$env:UV_CACHE_DIR = Join-Path $m5 '"'.uv-cache'
"'$env:UV_PYTHON_INSTALL_DIR = Join-Path $m5 '"'.uv-python'
"'$env:TMP = $m5
$env:TEMP = $m5
uv run --no-project --offline --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B -c '"'import numpy, scipy; print(numpy.__version__, scipy.__version__)'"
```

**Aggregated Output:**

```text
  × No solution found when resolving `--with` dependencies:
  ╰─▶ Because numpy was not found in the cache and you require numpy==2.3.5,
      we can conclude that your requirements are unsatisfiable.

hint: Packages were unavailable because the network was disabled. When the network is disabled, registry packages may only be read from the cache.
```

### Event #18: `item.completed` — `command_execution` (`item_15`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$base = '"'C:/Users/李则徐/AppData/Local/uv/cache'
Get-ChildItem \""'$base/simple-v21" -Recurse -Filter '"'*numpy*' | Select-Object -First 15 -ExpandProperty FullName
foreach ("'$p in @('"'numpy/2.3.5','scipy/1.17.0')) { "'$a = $p.Split('"'/'); "'$f=Get-Item "$base/wheels-v6/pypi/$($a[0])/$($a[1])-cp312-cp312-win_amd64"; $f | Format-List LinkType,Target,Attributes; Get-ChildItem $f.FullName -Name | Select-Object -First 10 }
Get-ChildItem '"'research/stage4_math_and_design/M5/.uv-cache' -Force -Name"
```

**Aggregated Output:**

```text
C:\Users\������\AppData\Local\uv\cache\simple-v21\pypi\numpy.lock
C:\Users\������\AppData\Local\uv\cache\simple-v21\pypi\numpy.rkyv


LinkType   : 
Target     : {}
Attributes : Archive



2.3.5-cp312-cp312-win_amd64


LinkType   : 
Target     : {}
Attributes : Archive



1.17.0-cp312-cp312-win_amd64
interpreter-v4
sdists-v9
simple-v21
.gitignore
.lock
CACHEDIR.TAG
```

### Event #19: `item.completed` — `command_execution` (`item_16`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$m5 = (Resolve-Path '"'research/stage4_math_and_design/M5').Path
"'$base = '"'C:/Users/李则徐/AppData/Local/uv/cache'
"'$env:UV_CACHE_DIR = Join-Path $m5 '"'.uv-cache'
"'$env:UV_PYTHON_INSTALL_DIR = Join-Path $m5 '"'.uv-python'
"'$env:TMP = $m5
$env:TEMP = $m5
$env:UV_LINK_MODE = '"'copy'
New-Item -ItemType Directory -Force (Join-Path "'$env:UV_CACHE_DIR '"'simple-v21/pypi'),(Join-Path "'$env:UV_CACHE_DIR '"'wheels-v6/pypi') | Out-Null
foreach ("'$name in @('"'numpy','scipy')) { Copy-Item -LiteralPath \""'$base/simple-v21/pypi/$name.rkyv" -Destination (Join-Path $env:UV_CACHE_DIR '"'simple-v21/pypi'); Copy-Item -LiteralPath \""'$base/wheels-v6/pypi/$name" -Destination (Join-Path $env:UV_CACHE_DIR '"'wheels-v6/pypi') -Recurse }
uv run --no-project --offline --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B -c 'import numpy, scipy; print(numpy.__version__, scipy.__version__)'"
```

**Aggregated Output:**

```text
  × Failed to download `scipy==1.17.0`
  ╰─▶ Network connectivity is disabled, but the
      requested data wasn't found in the cache for:
      `https://files.pythonhosted.org/packages/92/ce/672ed546f96d5d41ae78c4b9b02006cedd0b3d6f2bf5bb76ea455c320c28/scipy-1.17.0-cp312-cp312-win_amd64.whl`
```

### Event #20: `item.completed` — `command_execution` (`item_17`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$base = '"'C:/Users/李则徐/AppData/Local/uv/cache'
foreach ("'$p in @('"'numpy/2.3.5','scipy/1.17.0')) { "'$a = $p.Split('"'/'); "'$f=Get-Item "$base/wheels-v6/pypi/$($a[0])/$($a[1])-cp312-cp312-win_amd64"; Get-ChildItem $f.FullName -Force | Format-Table Name,Mode,LinkType,Target,Length; Get-Content -Encoding utf8 $f.FullName -ErrorAction SilentlyContinue }
Get-ChildItem $base/archive-v0 -Recurse -Directory -Filter '"'scipy-1.17.0.dist-info' | Select-Object -First 5 -ExpandProperty FullName
Get-ChildItem "'$base/archive-v0 -Recurse -Directory -Filter '"'numpy-2.3.5.dist-info' | Select-Object -First 5 -ExpandProperty FullName"
```

**Aggregated Output:**

```text

Name                        Mode   LinkType Target Length
----                        ----   -------- ------ ------
2.3.5-cp312-cp312-win_amd64 -a----          {}         27


archive-v0/cSnmXl2iWBpNAXGE

Name                         Mode   LinkType Target Length
----                         ----   -------- ------ ------
1.17.0-cp312-cp312-win_amd64 -a----          {}         27


archive-v0/3Oxde2-Jkv6KGjnQ
C:\Users\������\AppData\Local\uv\cache\archive-v0\3Oxde2-Jkv6KGjnQ\scipy-1.17.0.dist-info
C:\Users\������\AppData\Local\uv\cache\archive-v0\ddofvKiCGHzeQAsn\Lib\site-packages\scipy-1.17.0.dist-info
C:\Users\������\AppData\Local\uv\cache\archive-v0\IZacSOhqoef6F1kn\Lib\site-packages\scipy-1.17.0.dist-info
C:\Users\������\AppData\Local\uv\cache\archive-v0\wrataHacMqZZjJG5\Lib\site-packages\scipy-1.17.0.dist-info
C:\Users\������\AppData\Local\uv\cache\archive-v0\YPuTQaetJCCL75-K\Lib\site-packages\scipy-1.17.0.dist-info
C:\Users\������\AppData\Local\uv\cache\archive-v0\cSnmXl2iWBpNAXGE\numpy-2.3.5.dist-info
C:\Users\������\AppData\Local\uv\cache\archive-v0\ddofvKiCGHzeQAsn\Lib\site-packages\numpy-2.3.5.dist-info
C:\Users\������\AppData\Local\uv\cache\archive-v0\IZacSOhqoef6F1kn\Lib\site-packages\numpy-2.3.5.dist-info
C:\Users\������\AppData\Local\uv\cache\archive-v0\qnqkqxVxMV7Kj5rR\Lib\site-packages\numpy-2.3.5.dist-info
C:\Users\������\AppData\Local\uv\cache\archive-v0\wrataHacMqZZjJG5\Lib\site-packages\numpy-2.3.5.dist-info
```

### Event #21: `item.completed` — `command_execution` (`item_18`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$m5 = (Resolve-Path '"'research/stage4_math_and_design/M5').Path
"'$base = '"'C:/Users/李则徐/AppData/Local/uv/cache'
"'$env:UV_CACHE_DIR = Join-Path $m5 '"'.uv-cache'
"'$env:UV_PYTHON_INSTALL_DIR = Join-Path $m5 '"'.uv-python'
"'$env:TMP = $m5
$env:TEMP = $m5
$env:UV_LINK_MODE = '"'copy'
New-Item -ItemType Directory -Force (Join-Path "'$env:UV_CACHE_DIR '"'archive-v0') | Out-Null
foreach ("'$id in @('"'cSnmXl2iWBpNAXGE','3Oxde2-Jkv6KGjnQ')) { Copy-Item -LiteralPath \""'$base/archive-v0/$id" -Destination (Join-Path $env:UV_CACHE_DIR '"'archive-v0') -Recurse }
uv run --no-project --offline --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B -c 'import numpy, scipy; print(numpy.__version__, scipy.__version__)'"
```

**Aggregated Output:**

```text
Installed 2 packages in 10.26s
2.3.5 1.17.0
```

### Event #22: `item.completed` — `web_search` (`exec-82b758f0-3e51-4e8e-b233-51eba037a0ef`)

```json
{
  "id": "exec-82b758f0-3e51-4e8e-b233-51eba037a0ef",
  "type": "web_search",
  "query": "Blackwell Equivalent Comparisons of Experiments 1953 Annals Mathematical Statistics Project Euclid ...",
  "action": {
    "type": "search",
    "queries": [
      "Blackwell Equivalent Comparisons of Experiments 1953 Annals Mathematical Statistics Project Euclid",
      "Information Theory and Statistics Kullback Leibler 1951 sufficiency Project Euclid",
      "Lindley 1956 measure information experiment Annals mathematical statistics information",
      "LEMMO arxiv 2611.09476 Qwen3.5 4B IQ"
    ]
  },
  "results": [
    {
      "type": "text_result",
      "domain": "bibbase.org",
      "ref_id": "turn0search0",
      "snippet": "On a Measure of the Information Provided by an Experiment. Lindley, D. ... The Annals of Mathematical Statistics, 27(4):986–1005, December, 1956.",
      "title": "On a Measure of the Information Provided by an Experiment",
      "url": "https://bibbase.org/network/publication/lindley-onameasureoftheinformationprovidedbyanexperiment-1956"
    },
    {
      "type": "text_result",
      "domain": "arxiv.cnoctave.top",
      "ref_id": "turn0search12",
      "snippet": "The EM encoder converts the IQ recording into signal tokens, which are projected into the LLM embedding space and passed directly to Qwen3.5-4B together with",
      "title": "LEMMO: A Large Electromagnetic Model leveraging Pretrained LLMs for Million-Sample Signal Understanding",
      "url": "https://arxiv.cnoctave.top/pdf/2611.09476"
    },
    {
      "type": "text_result",
      "domain": "tensorjournal.com",
      "ref_id": "turn0search1",
      "snippet": "The arXiv record is 2611.09476. ... The authors say LEMMO accepts inputs of up to eight million complex samples, counted as complete IQ samples. ...",
      "title": "LEMMO electromagnetic signal model: author-reported gains · Tensor Journal",
      "url": "https://tensorjournal.com/en/research/lemmo-electromagnetic-signal-model-llm"
    },
    {
      "type": "text_result",
      "domain": "www.mathnet.ru",
      "ref_id": "turn0search2",
      "snippet": "Lindley, “On a measure of the information provided by an experiment”, Matematika, 3:3 (1959), 87–104; Ann. ... Statistics, 27 (1956), 986–1005 Image Citation in format",
      "title": "D. V. Lindley, “On a measure of the information provided by an experiment”, Matematika, 3:3 (1959), 87–104; Ann. Math. Statistics, 27 (1956), 986–1005",
      "url": "https://www.mathnet.ru/php/archive.phtml?jrnid=mat&option_lang=eng&paperid=101&wshow=paper"
    },
    {
      "type": "text_result",
      "domain": "celebratio.org",
      "ref_id": "turn0search3",
      "snippet": "AUTHOR = {Blackwell, David}, TITLE = {Comparison of experiments}, ... URL = {http://projecteuclid.org/euclid.bsmsp/1200500222}, ... 1953 ... TITLE = {Equivalent comparisons of experiments}, ... FJOURN",
      "title": "Celebratio Mathematica — Blackwell — Statistics",
      "url": "https://celebratio.org/Blackwell_DH/article/255/"
    },
    {
      "type": "text_result",
      "domain": "cir.nii.ac.jp",
      "ref_id": "turn0search4",
      "snippet": "# Equivalent Comparisons of Experiments ... * David Blackwell ... The Annals of Mathematical Statistics 24 (2), 265-272, 1953-06",
      "title": "Equivalent Comparisons of Experiments",
      "url": "https://cir.nii.ac.jp/crid/1362544419209410176"
    },
    {
      "type": "text_result",
      "domain": "www.ros.hw.ac.uk",
      "ref_id": "turn0search13",
      "snippet": "Suggesting a measure of information provided by an experiment whose objective ... θ, Lindley (1956) exploited Shannon’s information measure.",
      "title": "Inference and Experimental Design for",
      "url": "https://www.ros.hw.ac.uk/bitstream/10399/2341/1/BejanA_0610_macs.pdf"
    },
    {
      "type": "text_result",
      "domain": "www.scienceopen.com",
      "ref_id": "turn0search5",
      "snippet": "# On Information and Sufficiency ... Kullback, R. ... Leibler Publication date Created: March 1951 ... Journal: The Annals of Mathematical Statistics",
      "title": "On Information and Sufficiency",
      "url": "https://www.scienceopen.com/document?vid=3e5c5966-a6a6-4026-99c8-bf9861cb15bb"
    },
    {
      "type": "text_result",
      "domain": "bishtref.com",
      "ref_id": "turn0search6",
      "snippet": "# On Information and Sufficiency ... Kullback R. ... Mathematics Computer science Computational Theory and Mathematics Mathematical economics Econometrics Computability, Logic, AI Algorithms ... Leibl",
      "title": "On Information and Sufficiency",
      "url": "https://bishtref.com/articles/10.1214/aoms/1177729694"
    },
    {
      "type": "text_result",
      "domain": "www.jstor.org",
      "ref_id": "turn0search7",
      "snippet": "Lindley ... The subject matter of mathematical statistics may be divided into two parts, the theory of probability and the theory of inference. ... We",
      "title": "Statistical Inference | JSTOR",
      "url": "https://www.jstor.org/stable/2983721"
    },
    {
      "type": "text_result",
      "domain": "digicoll.lib.berkeley.edu",
      "ref_id": "turn0search8",
      "snippet": "Comparison of Experiments ... Blackwell, David, Author ... Proceedings of the Second Berkeley Symposium on Mathematical Statistics and Probability (July 31 - August 12, 1950)",
      "title": "Comparison of Experiments",
      "url": "https://digicoll.lib.berkeley.edu/record/112749"
    },
    {
      "type": "text_result",
      "domain": "dash.harvard.edu",
      "ref_id": "turn0search14",
      "snippet": "In 1951, Solomon Kullback ... gap between the fields of statistics and information theory, Kullback demonstrated that ... Leibler, “On Information and Sufficiency,” Annals of",
      "title": "A Dream Before the Dawn of the",
      "url": "https://dash.harvard.edu/server/api/core/bitstreams/947f9085-cdbf-4d18-98dd-edd989761c9a/content"
    },
    {
      "type": "text_result",
      "domain": "explore.openaire.eu",
      "ref_id": "turn0search9",
      "snippet": "Project Euclid arrow_drop_down ... Comparison of Experiments description Publication keyboard_double_arrow_right Part of book or chapter of book , Other literature type 31 Dec 1951 Publisher:Universit",
      "title": "Comparison of Experiments",
      "url": "https://explore.openaire.eu/search/result?pid=10.1525%2F9780520411586-009"
    },
    {
      "type": "text_result",
      "domain": "centerconsulting.com",
      "ref_id": "turn0search10",
      "snippet": "“On Information and Sufficiency” by Solomon Kullback and Richard Leibler appeared in The Annals of Mathematical Statistics, volume 22, issue 1, in March 1951 (pages",
      "title": "On Information and Sufficiency | CenterConsulting",
      "url": "https://centerconsulting.com/ai-library/papers/kullback-leibler-divergence"
    },
    {
      "type": "text_result",
      "domain": "www.math.buffalo.edu",
      "ref_id": "turn0search11",
      "snippet": "Blackwell, David Equivalent comparisons of experiments . ... Annals of Mathematics Studies, no. ... The theory of value and the science of decision, a summary",
      "title": "David Blackwell page2, his research",
      "url": "https://www.math.buffalo.edu/mad/PEEPS/blackwell_david2.html"
    },
    {
      "type": "text_result",
      "domain": "upload.wikimedia.org",
      "ref_id": "turn0search15",
      "snippet": "**Lindley–Michetti (Bibliography)** ... V., Binomial sampling schemes and the concept of information, *Biometrika* **44**, 179 (1957). ... M., A significance test for the hypothesis that",
      "title": "Selected bibliography of statistical literature: Supplement, 1958-1960",
      "url": "https://upload.wikimedia.org/wikipedia/commons/8/8a/Selected_bibliography_of_statistical_literature-_Supplement%2C_1958-1960_%28IA_jresv67Bn2p91%29.pdf"
    },
    {
      "type": "text_result",
      "domain": "fileserver-az.core.ac.uk",
      "ref_id": "turn0search16",
      "snippet": "*Science, 330*, 845–851. ... Lindley, D. ... (1956). On a measure of the information provided by an experiment. *The Annals of Mathematical Statistics, 27*, 986–1005.",
      "title": "K. Friston, R.J. Moran, Y. Nagai et al.",
      "url": "https://fileserver-az.core.ac.uk/download/480393614.pdf"
    },
    {
      "type": "text_result",
      "domain": "business.gwu.edu",
      "ref_id": "turn0search17",
      "snippet": "The expected sample information about the parameter, \\(M(\\mathbf{Y};\\Theta)\\) is known as Lindley’s measure (Lindley 1956) and is referred to as the parameter information.",
      "title": "STS0912_008R_051910.DVI",
      "url": "https://business.gwu.edu/sites/g/files/zaxdzs5326/files/downloads/TR-2009-5.pdf"
    },
    {
      "type": "text_result",
      "domain": "en.wikipedia.org",
      "ref_id": "turn0search18",
      "snippet": "In the 1950-1951 academic year, Blackwell took leave from Howard to serve as a visiting professor of statistics at Stanford University, where he continued his",
      "title": "David Blackwell",
      "url": "https://en.wikipedia.org/wiki/David_Blackwell"
    },
    {
      "type": "text_result",
      "domain": "es.wikipedia.org",
      "ref_id": "turn0search19",
      "snippet": "* Blackwell, David (1953). «Equivalent Comparisons of Experiments». ... The Annals of Mathematical Statistics 33 (2): 719-726. doi:10.1214/aoms/1177704593.",
      "title": "David Blackwell",
      "url": "https://es.wikipedia.org/wiki/David_Blackwell"
    },
    {
      "type": "text_result",
      "domain": "en.wikipedia.org",
      "ref_id": "turn0search20",
      "snippet": "In the mathematical subjects of information theory and decision theory, Blackwell's informativeness theorem is an important result related to the ranking of information structures, or",
      "title": "Blackwell's informativeness theorem",
      "url": "https://en.wikipedia.org/wiki/Blackwell%27s_informativeness_theorem"
    },
    {
      "type": "text_result",
      "domain": "en.wikipedia.org",
      "ref_id": "turn0search21",
      "snippet": "Timeline of information theory ... * 1951 – Solomon Kullback and Richard Leibler introduce the Kullback-Leibler divergence",
      "title": "Timeline of information theory",
      "url": "https://en.wikipedia.org/wiki/Timeline_of_information_theory"
    },
    {
      "type": "text_result",
      "domain": "en.wikipedia.org",
      "ref_id": "turn0search22",
      "snippet": "The Kullback-Leibler divergence (or information divergence, information gain, or relative entropy) is a way of comparing two distributions, a \"true\" probability distribution ${\\displaystyle p}$, and",
      "title": "Quantities of information",
      "url": "https://en.wikipedia.org/wiki/Quantities_of_information"
    },
    {
      "type": "text_result",
      "domain": "arxiv.org",
      "ref_id": "turn0academia23",
      "snippet": "Title: Information Theory and Statistics: an overview ... We recall the basic quantities in information theory; entropy, cross-entropy, conditional entropy, mutual information and Kullback-Leibler ris",
      "title": "Information Theory and Statistics: an overview",
      "url": "https://arxiv.org/abs/1511.00860"
    },
    {
      "type": "text_result",
      "domain": "en.wikipedia.org",
      "ref_id": "turn0search24",
      "snippet": "It is now at the intersection of mathematics, statistics and computer science, and has applications in diverse fields ranging from electrical engineering and physics to",
      "title": "Information theory",
      "url": "https://en.wikipedia.org/wiki/Information_theory"
    },
    {
      "type": "text_result",
      "domain": "www.tomshardware.com",
      "ref_id": "turn0news25",
      "snippet": "Described in its most basic terms, Qwenputer is an experimental blend of the stock Qwen 3.5 4B AI LLM with a 32-bit RISC-V CPU compiled",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/3LyZ9V4GGiA7Fi4pxreV8uAf5sMA9vls62xDeeVVIEIHroO5tQWDD5DIGlGbOG8kYdLrJYSjd-YQ_jMfElu4VJcs8eaXH_ULB3IJj3DBPZqeseOrtdant1eL-YrnhCSF",
      "title": "AI model gets a 32-bit RISC-V CPU compiled directly into its weights, then runs DOOM on it",
      "url": "https://www.tomshardware.com/tech-industry/artificial-intelligence/ai-engineer-successfully-compiles-a-32-bit-risc-v-cpu-in-qwen-3-5-4b-llm-inserted-directly-into-its-weights-built-in-cpu-can-run-processor-arithmetic-operations-code-and-of-course-doom"
    },
    {
      "type": "text_result",
      "domain": "www.youtube.com",
      "ref_id": "turn0youtube26",
      "snippet": "In this episode with go over the Kullback-Leibler (KL) divergence paper, \"On Information and Sufficiency\" (1951). ... This concept, rooted in Shannon's information theory (which",
      "title": "Data Science #19 - The Kullback–Leibler divergence paper (1951) - YouTube",
      "url": "https://www.youtube.com/watch?v=eC-SXX3L7YI"
    }
  ]
}
```

### Event #23: `item.completed` — `web_search` (`exec-d3b7e564-06c5-487d-bfbe-7bbec9533fb6`)

```json
{
  "id": "exec-d3b7e564-06c5-487d-bfbe-7bbec9533fb6",
  "type": "web_search",
  "query": "https://arxiv.org/html/2611.09476v1",
  "action": {
    "type": "open_page",
    "url": "https://arxiv.org/html/2611.09476v1"
  },
  "results": [
    {
      "type": "text_result",
      "domain": "arxiv.org",
      "ref_id": "turn1view0",
      "snippet": "Total lines: 551",
      "title": "LEMMO: A Large Electromagnetic Model leveraging Pretrained LLMs for Million-Sample Signal Understanding",
      "url": "https://arxiv.org/html/2611.09476v1"
    },
    {
      "type": "text_result",
      "domain": "doi.org",
      "ref_id": "turn1view1",
      "snippet": "Total lines: 1",
      "title": "projecteuclid.org",
      "url": "https://doi.org/10.1214/aoms/1177729694"
    },
    {
      "type": "text_result",
      "domain": "doi.org",
      "ref_id": "turn1view2",
      "snippet": "Total lines: 1",
      "title": "projecteuclid.org",
      "url": "https://doi.org/10.1214/aoms/1177728069"
    },
    {
      "type": "text_result",
      "ref_id": "turn1view3",
      "title": "Internal Error"
    }
  ]
}
```

### Event #24: `item.completed` — `command_execution` (`item_21`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n 'SIGNAL|SNR|def (signal|observe|parents|wave|likelihood)|N =' research/fourphase_paired_questions/paired.py
Get-Content research/fourphase_paired_questions/paired.py | Select-Object -First 105
rg -n '12\\.2|cpu.*ms' research/stage3d_qwen_signed_phase/REPORT.md research/stage3d_qwen_signed_phase/results/summary_master.json
Get-Content -Encoding utf8 docs/MULTIMODAL_INTEGRATION_PLAN.md | Select-Object -Skip 10 -First 17"
```

**Aggregated Output:**

```text
14:N = 3
15:SNR = 10 ** 0.2  # E_s/N_0 = 2 dB, signal energy per window is one
34:def parents(split):
37:    noise_sd = math.sqrt(1 / (2 * SNR))
39:    def observe(label):
75:def likelihood(win):
77:    logw = [2 * SNR / math.sqrt(N) * (total * s.conjugate()).real for s in ROOTS]
103:    logw = [sum(2 * SNR / math.sqrt(N) * (sum(w) * s.conjugate()).real
"""Paired QPSK questions, Python 3.9+, standard library. Run: python paired.py."""
import argparse
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
import json
import math
import random

Q1 = "第二个窗口的发射符号相对第一个窗口转了几个四分之一圈？"
Q2 = "第一个窗口原来发送的是哪个四相符号？"
ROOTS = (1, 1j, -1, -1j)
U = (0.25,) * 4
N = 3
SNR = 10 ** 0.2  # E_s/N_0 = 2 dB, signal energy per window is one
ROTATIONS = ((0, 0), (0, 1), (1, 1))
ARMS = ("none", "valid", "independent", "duplicate")
SPLITS = {"dev": (410061, 4), "test": (518339, 16)}  # blocks of 64 mothers
VIEWS = ("complex", "amplitude", "local_phase_removed")


@dataclass(frozen=True)
class Reference:
    # equality of source ids marks two renderings of one observation
    source: int
    shares_channel: bool
    samples: tuple


def rotate(window, quarter_turns):
    return tuple(y * ROOTS[quarter_turns % 4] for y in window)


def parents(split):
    seed, blocks = SPLITS[split]
    rng = random.Random(seed)
    noise_sd = math.sqrt(1 / (2 * SNR))

    def observe(label):
        mean = ROOTS[label % 4] / math.sqrt(N)
        return tuple(mean + complex(rng.gauss(0, noise_sd), rng.gauss(0, noise_sd))
                     for _ in range(N))

    for block in range(blocks):
        triples = list(product(range(4), repeat=3))
        rng.shuffle(triples)
        for j, (x1, x2, g) in enumerate(triples):
            y = (observe(x1 + g), observe(x2 + g))
            pilot = observe(g)
            g_other = rng.randrange(4)
            other = rotate(pilot, g_other - g)
            yield (split, block * 64 + j), (x1, x2), y, pilot, other


def reference_views(pilot, other):
    valid = Reference(0, True, pilot)
    return {"none": (), "valid": (valid,), "independent": (Reference(0, False, other),),
            "duplicate": (valid, valid)}


def features(win, view):
    if view == "complex":
        return win
    if view == "amplitude":
        return tuple(abs(z) for z in win)
    if view == "local_phase_removed":
        # One canonical orientation per window, with no absolute-angle side field.
        anchor = next((z for z in win if z != 0), None)
        if anchor is None:
            return win
        return tuple(z * anchor.conjugate() / abs(anchor) for z in win)
    raise ValueError(view)


def likelihood(win):
    total = sum(win)
    logw = [2 * SNR / math.sqrt(N) * (total * s.conjugate()).real for s in ROOTS]
    return normalize_exp(logw)


def normalize_exp(logw):
    a = max(logw)
    weights = [math.exp(v - a) for v in logw]
    return tuple(w / sum(weights) for w in weights)


def diff_prob(left, right):
    # P(right label - left label = k)
    return tuple(sum(left[i] * right[(i + k) % 4] for i in range(4)) for k in range(4))


def cache(data, view):
    # Preserve full observations in the selected representation. No bit cap here.
    return tuple(features(win, view) for win in data)


def read(cache_value, references, view):
    p1, p2 = (likelihood(w) if view == "complex" else U for w in cache_value)
    first_answer = diff_prob(p1, p2)
    shared = {ref.source: features(ref.samples, view) for ref in references if ref.shares_channel}
    if view != "complex" or not shared:
        return first_answer, U
    logw = [sum(2 * SNR / math.sqrt(N) * (sum(w) * s.conjugate()).real
                for w in shared.values()) for s in ROOTS]
    return first_answer, diff_prob(normalize_exp(logw), p1)
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

```

### Event #25: `item.completed` — `agent_message` (`item_22`)

The Stage 3D report separates two gaps on G1. The s17 physical reader was 0.0121 nat behind Full IQ, while the frozen-host phase arm was 0.5041 nat behind the s17 reader. I’ll preserve that distinction in the new benchmark so it can measure cache loss independently of host readout.

### Event #26: `item.completed` — `file_change` (`item_23`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\verify_m5.py",
    "kind": "add"
  }
]
```

### Event #27: `item.completed` — `file_change` (`item_24`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\verify_m5.py",
    "kind": "update"
  }
]
```

### Event #28: `item.completed` — `command_execution` (`item_26`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '2 bit|4 bit|2 比特|4 比特|两比特|四比特|2 位|4 位|先编码' docs/LONG_TERM_RESEARCH_PLAN.md research/abcde_convergence_2026_11_22/D/PROOF.md
Get-Content -Encoding utf8 docs/LONG_TERM_RESEARCH_PLAN.md | Select-Object -Skip 38 -First 19
Get-Content -Encoding utf8 docs/LONG_TERM_RESEARCH_PLAN.md | Select-Object -Skip 69 -First 8
rg -n '12\\.2|latency|耗时' research/stage3d_qwen_signed_phase/REPORT.md | Select-Object -First 14"
```

**Aggregated Output:**

```text
docs/LONG_TERM_RESEARCH_PLAN.md:15:**最终盲测。** 先编码，后揭题。两种解释均与旧波形相容时，只允许再读取一个已录制的候选参考窗。模型先选窗口并写下各解释预期的观测，再揭示真实波形并评分。对照包括随机窗、最近窗以及由物理模型预测能最大程度减少歧义的窗。提前约定主要指标、可接受的误差差距和采集划分。等缓存与观测预算时，比较答案风险、置信度和证据定位；达到约定准确度时，比较所需缓存和额外读取量。天然观测、人工注入和独立采集分别报告。若现有方法打平，保留已验证的边界与反例，继续寻找下一个确有差异的问题。
docs/LONG_TERM_RESEARCH_PLAN.md:31:|11 月 22 日 A–E 会合     |A/B 核对后验读出并定位同一观测被重复利用的问题。C 的连续频偏精化在部分离格条件有益，在跳变上可能有损。D 给出相位跳变后可辨识量和误码下界。E 给出何时值得切换估计器的理论条件，实际 14 帧合成检验比固定连续方法多错 172 位。会合报告中的 435 MHz 数字是对历史逐帧记录的复算。                                                                             |优先延伸 D，借用 A/B 的证据核算与 P7 的预测—揭示设计。频偏精化和门控作为具体问题的工具。 [五组会合](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/README.md) · [D 组证明](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/D/PROOF.md) · [实测输入与复算](https://github.com/20171806011/EM-FloatFormer/blob/e914befb2c03838c5e911e1af55c64230f1f42d6/research/abcde_convergence_2026_11_22/experiments/historical/REAL_INPUTS.md)                                                                                                                                                                                                                                                                      |
docs/LONG_TERM_RESEARCH_PLAN.md:41:**第一题可以小到手算。** 两个独立均匀四相符号 &#40;X\_1,X\_2&#41; 经过同一未知均匀四相旋转 &#40;G&#41;，得到无噪观测 &#40;Y\_k=GX\_k&#41;。现在问二者的相对相位，保存 &#40;R=Y\_2\\overline\{Y\_1\}&#41; 的四种状态即可，恰需 2 比特。稍后接收同一相干区间内单位导频的观测 &#40;P=G&#41;，再问第一个符号的原始四相标签。此时 &#40;X\_1=\\overline P Y\_1&#41;。若第一次只保存 &#40;R&#41;，补入 &#40;P&#41; 后仍有四个等可能答案。若同时保存 &#40;&#40;R,Y\_1&#41;&#41;，两题都能答对，恰需 4 比特。后一项也是下界：固定 &#40;P=1&#41; 时，16 种观测对对应 16 种不同的联合答案，任何固定长度、零误差、不能重读波形的编码都须区分它们。本计划另行枚举了 64 种等可能潜在状态来核对这个新例子。它是后续研究的出发点；真正待解的是有噪、部分相干和有限预算下的推广。
research/abcde_convergence_2026_11_22/D/PROOF.md:38:没有一个固定原始标签位在整个任一类中总能确定。仍可无误恢复旋转轨道编号（16 点分成 4 条轨道，均匀时含 2 bit 信息），例如利用半径并区别两条半径相同的非对角轨道。共同相位还保留后段符号间的比值。一整段只共享一个三选一或四选一的歧义，不能把不同位或不同符号的错误视作独立，也不能把轨道编号的两位等同于原标签的两位。
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

### �塢ʵ����������ģ�Ͷ����˲���

- **�������֡�** �����λ��Բ����������׼ȷ�ʣ�����������Ԥ���ù�һ����������ѡ�����ø��ʶ�����ʧ��У׼�����㶨λ��ʱ�����򸲸����䡣���� BER/BLER ���������񸴺ˡ�
- **��֤���ݡ�** �������õ���ͬ�ɼ���Ϣ���滻����סģ�����������Ĵ��ڣ����滻��ͬ���ȡ������������޹ش����Ƚϴ�ʵ�ʱ仯��������֤�ݵ���ȫ�ظ�����ȷ�ο�����Ӧ����һ�¡������ֳ����ɱ�ʶ���ޣ������Ų��ǩ���ظ��غɺ�����Ԫ���ݡ�
- **�������¡�** ��ͬһ�����ϲ��ż�¼��һ�λش���Ч������ο������ĸı䡢��Щ�𰸱�Ի������Ự�������ЧӦ�����䣬����Խ�����ʱ�䡢������ȡ�ͬ��ʧ�ܺͺ�ѡ�����ѷֵ�������
- **���Ƽ��㡣** ͬʱ�������� I/Q ���գ��Լ�����ͬ����λ����ͬ�������ض����ֹ�ض������µĽ�����ѧϰ���������в��������ӳٺͲ�ѯ�������������ʾ�ܴﵽ���۷��գ������Ǻ������뱣���Ķ��ա�

### ������һ����������
57:| **模型加载耗时与常驻显�?* | 加载 `57.69 s`，常驻权重占显存 `8,021.9 MB` |
102:| �?/ 种子 | 最�?Epoch | 开发筛选分 (`sel_score`) | 校准�?`dev_cfo_q3_nll` | 校准�?`dev_q3_cal_nll` (`T_q3`) | 校准�?`dev_q2_cal_nll` (`T_q2`) | 耗时 (s) / 峰�?VRAM | Checkpoint SHA-256（前 16 位） |
```

### Event #29: `item.completed` — `command_execution` (`item_25`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$m5 = (Resolve-Path '"'research/stage4_math_and_design/M5').Path
"'$env:UV_CACHE_DIR = Join-Path $m5 '"'.uv-cache'
"'$env:UV_PYTHON_INSTALL_DIR = Join-Path $m5 '"'.uv-python'
"'$env:TMP = $m5
$env:TEMP = $m5
$env:UV_LINK_MODE = '"'copy'
uv run --no-project --offline --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M5/verify_m5.py"
```

**Aggregated Output:**

```text
flat_minus3: 2
flat_minus3: 4_s17
flat_minus3: 8
flat_minus3: 12
flat_minus3: 16
flat_minus3: canonical_continuous
flat_minus3: full_iq
flat_plus2: 2
flat_plus2: 4_s17
flat_plus2: 8
flat_plus2: 12
flat_plus2: 16
flat_plus2: canonical_continuous
flat_plus2: full_iq
flat_plus24: 2
Traceback (most recent call last):
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M5\verify_m5.py", line 573, in <module>
    main()
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M5\verify_m5.py", line 523, in main
    profile_result, losses = run_profile(key, profile, tracks, encoder)
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M5\verify_m5.py", line 441, in run_profile
    ll = window_log_likelihood(method, t, residual, packets, bins, a, v, encoder, key)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M5\verify_m5.py", line 207, in window_log_likelihood
    polar = polar_emissions(angle_bits, radial_bits, a[window] / np.sqrt(variance), multiplier)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M5\verify_m5.py", line 185, in polar_emissions
    np.testing.assert_allclose(result.sum(axis=(1, 2)), 1., atol=5e-6, rtol=0)
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M5\.uv-cache\archive-v0\zokrL3rtYqGksyvG\Lib\site-packages\numpy\testing\_private\utils.py", line 1711, in assert_allclose
    assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M5\.uv-cache\archive-v0\zokrL3rtYqGksyvG\Lib\site-packages\numpy\testing\_private\utils.py", line 919, in assert_array_compare
    raise AssertionError(msg)
AssertionError: 
Not equal to tolerance rtol=0, atol=5e-06

Mismatched elements: 128 / 128 (100%)
Max absolute difference among violations: 0.22015162
Max relative difference among violations: 0.22015162
 ACTUAL: array([0.779848, 1.145962, 1.020958, 0.838591, 1.189722, 0.889056,
       0.996481, 1.083478, 0.895392, 1.084164, 0.948433, 1.025932,
       0.988399, 1.005006, 0.997964, 1.000687, 0.9997  , 1.000687,...
 DESIRED: array(1.)
```

### Event #30: `item.completed` — `file_change` (`item_27`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\verify_m5.py",
    "kind": "update"
  }
]
```

### Event #31: `item.completed` — `file_change` (`item_29`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\THEOREM_M5.md",
    "kind": "add"
  }
]
```

### Event #32: `item.completed` — `command_execution` (`item_28`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$m5 = (Resolve-Path '"'research/stage4_math_and_design/M5').Path
"'$env:UV_CACHE_DIR = Join-Path $m5 '"'.uv-cache'
"'$env:UV_PYTHON_INSTALL_DIR = Join-Path $m5 '"'.uv-python'
"'$env:TMP = $m5
$env:TEMP = $m5
$env:UV_LINK_MODE = '"'copy'
uv run --no-project --offline --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M5/verify_m5.py"
```

**Aggregated Output:**

```text
flat_minus3: 2
flat_minus3: 4_s17
flat_minus3: 8
flat_minus3: 12
flat_minus3: 16
flat_minus3: canonical_continuous
flat_minus3: full_iq
flat_plus2: 2
flat_plus2: 4_s17
flat_plus2: 8
flat_plus2: 12
flat_plus2: 16
flat_plus2: canonical_continuous
flat_plus2: full_iq
flat_plus24: 2
flat_plus24: 4_s17
flat_plus24: 8
flat_plus24: 12
flat_plus24: 16
flat_plus24: canonical_continuous
flat_plus24: full_iq
selective_fade: 2
selective_fade: 4_s17
selective_fade: 8
selective_fade: 12
selective_fade: 16
selective_fade: canonical_continuous
selective_fade: full_iq
Wrote results_m5.json
```

### Event #33: `item.completed` — `file_change` (`item_30`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\THEOREM_M5.md",
    "kind": "update"
  }
]
```

### Event #34: `item.completed` — `command_execution` (`item_31`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-ChildItem -Force research/stage4_math_and_design/M5 -Name
Get-Content research/stage3d_qwen_signed_phase/results/summary_master.json | Select-Object -Skip 220 -First 40
rg -n '12\\.2 ms|12\\.2ms|12\\.2 millisecond' research docs -g '*.md'"
```

**Aggregated Output:**

```text
.uv-cache
results_m5.json
THEOREM_M5.md
verify_m5.py
          0.06615776081424937
        ],
        "nll_improv_sd": 0.02167594178554353,
        "nll_improv_range": [
          0.010627005767255858,
          0.048644901678691926
        ]
      },
      "analytic_same_mothers": {
        "direct_s17_table": {
          "acc": 0.6461712786259542,
          "nll": 0.941125824839512,
          "brier": 0.5098317013509959,
          "ece": 0.0009122218241481161
        },
        "full_iq_known_param": {
          "acc": 0.6464098282442748,
          "nll": 0.9290379865758834,
          "brier": 0.5032444510266945,
          "ece": 0.0046012731581605345
        }
      },
      "phase_minus_direct_s17": {
        "acc_gain_t95": {
          "n_blocks": 131,
          "mean": -0.4081849024597116,
          "block_sd": 0.049473726656532394,
          "se": 0.004322539567795168,
          "t_crit_95": 1.9783804054470222,
          "lo95": -0.416736530042407,
          "hi95": -0.3996332748770162
        },
        "nll_improv_t95": {
          "n_blocks": 131,
          "mean": -0.5041027288195585,
          "block_sd": 0.07410397079738873,
          "se": 0.006474493990036962,
          "t_crit_95": 1.9783804054470222,
          "lo95": -0.5169117408646322,
          "hi95": -0.4912937167744849
research\stage4_math_and_design\discussions\M5\prompt_r1.md:38:     - Quantify mathematically WHERE `s17` is near-sufficient (`12.2 ms` vs `0.504 nat`, only `~0.012 nat` below `full_iq` on `G1` synthetic 2 dB `Q3`!) vs WHERE keeping ONLY 4-bit discrete `s17` **permanently destroys essential information** required for Stage 4 real Pluto/CTTC (`SNR` varies `-3..+24 dB`, amplitude fading `>10 dB`, residual DC/spur quality), Stage 5 (distinguishing smooth CFO drift vs discrete jump vs low-SNR noise burst), and Stage 6 (pre-reveal amplitude/phase prediction under rival hypotheses)!
```

### Event #35: `item.completed` — `command_execution` (`item_32`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$j = Get-Content research/stage4_math_and_design/M5/results_m5.json -Raw | ConvertFrom-Json
$j.nll_pareto_table | ConvertTo-Json -Depth 4
$j.paired_nll_16_minus_s17 | ConvertTo-Json -Depth 4
$j.s17_radial_collision_witness | ConvertTo-Json -Depth 4
foreach ($p in $j.per_profile) { $s = $p.metrics.'"'16'.stage6; Write-Output ("'$p.profile); foreach ($cl in @('"'4_s17','16','full_iq')) { "'$m = $p.metrics.PSObject.Properties[$cl].Value; $b = $m.stage6; Write-Output "$cl Stage5 binary $($m.stage5_smooth_vs_jump_given_pair.nll) / $($m.stage5_smooth_vs_jump_given_pair.accuracy), location $($m.stage5_jump_location_given_jump.nll); Stage6 EIG/random/near $($b.max_EIG_window.nll) / $($b.random_window.nll) / $($b.nearest_window.nll); accuracies $($b.max_EIG_window.accuracy) / $($b.random_window.accuracy) / $($b.nearest_window.accuracy); expected and realized gain $($b.max_EIG_window.expected_eig) / $($b.max_EIG_window.observed_nll_gain); true/rival $($b.max_EIG_window.precommitted_true_h_obs_nll) / $($b.max_EIG_window.precommitted_rival_obs_nll); quadrature $($b.quadrature_max_abs_delta_position_0_to_63)" }}'
```

**Aggregated Output:**

```text
{
    "2":  {
              "Q1":  1.145303768860555,
              "Q2":  0.8616165127607907,
              "stage5_model":  1.2709467453612076,
              "stage5_event_and_location":  1.5395649599855026,
              "stage5_smooth_vs_jump_given_pair":  0.6931471805599453,
              "stage5_jump_location_given_jump":  1.0986122886681096,
              "stage6_post_max_eig":  0.46193518486767665
          },
    "4_s17":  {
                  "Q1":  0.9969978902592825,
                  "Q2":  0.7966497961299691,
                  "stage5_model":  1.0413326800512352,
                  "stage5_event_and_location":  1.278765651482162,
                  "stage5_smooth_vs_jump_given_pair":  0.5240112783723827,
                  "stage5_jump_location_given_jump":  0.971068847709512,
                  "stage6_post_max_eig":  0.36770256221497916
              },
    "8":  {
              "Q1":  0.9647869959082289,
              "Q2":  0.7856453862072215,
              "stage5_model":  0.8309072076419806,
              "stage5_event_and_location":  1.061474784681836,
              "stage5_smooth_vs_jump_given_pair":  0.5182574374627927,
              "stage5_jump_location_given_jump":  0.9429903100901134,
              "stage6_post_max_eig":  0.3614416465432886
          },
    "12":  {
               "Q1":  0.9479094296204409,
               "Q2":  0.7536323698096661,
               "stage5_model":  0.7648670644610429,
               "stage5_event_and_location":  0.994391377337026,
               "stage5_smooth_vs_jump_given_pair":  0.5143590299057837,
               "stage5_jump_location_given_jump":  0.9387235002895925,
               "stage6_post_max_eig":  0.3545215292041904
           },
    "16":  {
               "Q1":  0.9455574491340679,
               "Q2":  0.7493153673288835,
               "stage5_model":  0.7374902047506247,
               "stage5_event_and_location":  0.9668770690122719,
               "stage5_smooth_vs_jump_given_pair":  0.5136493818015074,
               "stage5_jump_location_given_jump":  0.9381613539847294,
               "stage6_post_max_eig":  0.3544230046430205
           },
    "canonical_continuous":  {
                                 "Q1":  0.9437871451113418,
                                 "Q2":  0.7424908648343169,
                                 "stage5_model":  0.6640427727112678,
                                 "stage5_event_and_location":  0.8933546592277251,
                                 "stage5_smooth_vs_jump_given_pair":  0.5131090083478068,
                                 "stage5_jump_location_given_jump":  0.9378547051137386,
                                 "stage6_post_max_eig":  0.35381021836071536
                             },
    "full_iq":  {
                    "Q1":  0.9437871451113418,
                    "Q2":  0.7424908648343169,
                    "stage5_model":  0.6640427727112678,
                    "stage5_event_and_location":  0.8933546592277251,
                    "stage5_smooth_vs_jump_given_pair":  0.5131090083478068,
                    "stage5_jump_location_given_jump":  0.9378547051137386,
                    "stage6_post_max_eig":  0.35381021836071536
                }
}
{
    "Q1":  {
               "mean":  -0.05144044112521451,
               "pointwise_95ci":  [
                                      -0.056173050461438286,
                                      -0.04670783178899073
                                  ]
           },
    "Q2":  {
               "mean":  -0.04733442880108564,
               "pointwise_95ci":  [
                                      -0.0520121547647893,
                                      -0.04265670283738198
                                  ]
           },
    "stage5_model":  {
                         "mean":  -0.3038424753006105,
                         "pointwise_95ci":  [
                                                -0.31331226531841627,
                                                -0.2943726852828047
                                            ]
                     },
    "stage5_event_and_location":  {
                                      "mean":  -0.3118885824698902,
                                      "pointwise_95ci":  [
                                                             -0.3214372162247922,
                                                             -0.3023399487149882
                                                         ]
                                  },
    "stage5_smooth_vs_jump_given_pair":  {
                                             "mean":  -0.010361896570875415,
                                             "pointwise_95ci":  [
                                                                    -0.014358563812636213,
                                                                    -0.006365229329114618
                                                                ]
                                         },
    "stage5_jump_location_given_jump":  {
                                            "mean":  -0.03290749372478253,
                                            "pointwise_95ci":  [
                                                                   -0.03899096079480228,
                                                                   -0.02682402665476278
                                                               ]
                                        },
    "stage6_post_max_eig":  {
                                "mean":  -0.013279557571958615,
                                "pointwise_95ci":  [
                                                       -0.017160283348400543,
                                                       -0.009398831795516687
                                                   ]
                            }
}
{
    "code":  1,
    "matched_sum_amplitude":  [
                                  0.02,
                                  0.5833281552047411
                              ],
    "posterior_at_pair":  [
                              [
                                  0.2660946464450768,
                                  0.24974897951639308,
                                  0.23440739452213716,
                                  0.24974897951639308
                              ],
                              [
                                  0.7465178798894453,
                                  0.11749478199083367,
                                  0.018492556128887325,
                                  0.11749478199083367
                              ]
                          ],
    "posterior_tv":  0.48042323344436855,
    "strict_neural_assignment_margins":  [
                                             1.0145228451367705,
                                             0.011926580199638992
                                         ]
}
flat_minus3
4_s17 Stage5 binary 0.6930249891750865 / 0.5185002466699556, location 1.0986964442534988; Stage6 EIG/random/near 0.6450763281741048 / 0.6758540301313506 / 0.6924279683508403; accuracies 0.61328125 / 0.5634765625 / 0.521484375; expected and realized gain 0.04644415972780791 / 0.047987602705330826; true/rival 2.307949700317199 / 2.507299078323119; quadrature 9.66297952070816E-09
16 Stage5 binary 0.6916994017623331 / 0.5150468672915639, location 1.098849547550715; Stage6 EIG/random/near 0.6373868529066015 / 0.6741712799713557 / 0.6928158132608737; accuracies 0.6181640625 / 0.5595703125 / 0.5302734375; expected and realized gain 0.055278224796708554 / 0.05606964062750787; true/rival 2.280561610922397 / 2.518951495657757; quadrature 3.33068588265206E-08
full_iq Stage5 binary 0.6916181856682072 / 0.5185002466699556, location 1.0989576808363593; Stage6 EIG/random/near 0.6365001973740515 / 0.673342647875673 / 0.692469304751175; accuracies 0.6162109375 / 0.56396484375 / 0.5341796875; expected and realized gain 0.055675543303012796 / 0.05669578163941895; true/rival 2.276804031588221 / 2.5178732983657808; quadrature 3.05955746848774E-08
flat_plus2
4_s17 Stage5 binary 0.6912667801595072 / 0.5264677574590952, location 1.096133871999207; Stage6 EIG/random/near 0.5630102756158182 / 0.6260077586115609 / 0.6623498051446626; accuracies 0.7080078125 / 0.64013671875 / 0.5986328125; expected and realized gain 0.12070228071613509 / 0.12526726200666316; true/rival 2.0658305093106204 / 2.6410082080213795; quadrature 4.95652433263327E-08
16 Stage5 binary 0.6802708801501091 / 0.5490856592877768, location 1.0868515699727341; Stage6 EIG/random/near 0.5433659925603104 / 0.6107226834287436 / 0.6444620008173143; accuracies 0.72265625 / 0.65771484375 / 0.6328125; expected and realized gain 0.12352952391180036 / 0.12898317486994318; true/rival 2.051059119764812 / 2.6559921777905067; quadrature 8.02970146901449E-08
full_iq Stage5 binary 0.6789122633744135 / 0.5538979788257941, location 1.0864903981229923; Stage6 EIG/random/near 0.542150660403683 / 0.6092349090327489 / 0.6436824538614732; accuracies 0.72314453125 / 0.65478515625 / 0.6240234375; expected and realized gain 0.12393970992307227 / 0.12976120953391307; true/rival 2.04709594257145 / 2.656353143160639; quadrature 8.22157333046913E-08
flat_plus24
4_s17 Stage5 binary 0.030034659374481118 / 0.9940623453735774, location 0.5897187575961763; Stage6 EIG/random/near 0.023912210703263756 / 0.02879114743547346 / 0.031377191132581014; accuracies 0.99267578125 / 0.99365234375 / 0.99267578125; expected and realized gain 0.012982267420196768 / 0.009345582585233162; true/rival 1.193579888634134 / 3.2299858835692765; quadrature 7.05031595116864E-10
16 Stage5 binary 1.41399772905356E-10 / 1.0, location 0.47667089934742934; Stage6 EIG/random/near 1.81603863891322E-17 / 1.9130747333509E-16 / 4.34222970080531E-16; accuracies 1.0 / 1.0 / 1.0; expected and realized gain 1.53435449847823E-16 / -2.73110527249051E-16; true/rival 0.9691628743148345 / 2.5967144443449017; quadrature 7.39715205843026E-17
full_iq Stage5 binary 8.78287057075929E-11 / 1.0, location 0.47667089934742934; Stage6 EIG/random/near 1.46909394371787E-17 / 1.70273951188871E-16 / 1.27773226027423E-16; accuracies 1.0 / 1.0 / 1.0; expected and realized gain 7.67239294471435E-16 / 6.58110718698703E-16; true/rival 1.0031775481449572 / 2.717381472362958; quadrature 6.1794171942924E-17
selective_fade
4_s17 Stage5 binary 0.6807656133083111 / 0.5712105798575788, location 1.0903010249378382; Stage6 EIG/random/near 0.23881143436672994 / 0.5225432302603794 / 0.579800147265297; accuracies 0.9072265625 / 0.708984375 / 0.703125; expected and realized gain 0.4596843115710118 / 0.44701938017712617; true/rival 0.5317148399545587 / 4.030838392343038; quadrature 6.33579283482177E-07
16 Stage5 binary 0.6819802751413295 / 0.5615462868769074, location 1.0790312140289504; Stage6 EIG/random/near 0.2369391731051702 / 0.5140986804080132 / 0.5713970088345859; accuracies 0.90625 / 0.71630859375 / 0.7021484375; expected and realized gain 0.45416231110649774 / 0.4395346699672849; true/rival 0.5288862511323856 / 4.049154408122931; quadrature 1.63708929273865E-06
full_iq Stage5 binary 0.6812758645633079 / 0.5661241098677517, location 1.0780598519004227; Stage6 EIG/random/near 0.23659001566512708 / 0.5138939548949164 / 0.571497871158102; accuracies 0.9072265625 / 0.71630859375 / 0.703125; expected and realized gain 0.4538563175124225 / 0.4402252849908651; true/rival 0.5289486103238162 / 4.048651954561285; quadrature 1.55017363068222E-06
```

### Event #36: `item.completed` — `agent_message` (`item_33`)

The first complete CPU run shows where the extra bits helped most. The 16-bit cache reduced joint event-and-location NLL by 0.312 nat versus s17. I’ll now check the active-window result specifically on recordings where the old evidence still leaves both rivals plausible.

### Event #37: `item.completed` — `file_change` (`item_34`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\verify_m5.py",
    "kind": "update"
  }
]
```

### Event #38: `item.completed` — `file_change` (`item_36`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\THEOREM_M5.md",
    "kind": "update"
  }
]
```

### Event #39: `item.completed` — `command_execution` (`item_37`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n 'matched controls|physical receiver|matched-control|Local TAN' 'C:/Users/李则徐/.codex/memories/MEMORY.md'
rg -n 'matched preprocessing|nearly all|Group' 'C:/Users/李则徐/.codex/memories/rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md'
Get-Item research/stage4_math_and_design/M5/results_m5.json | Select-Object Length,LastWriteTime
rg -n 'projecteuclid|"'^##'"' research/stage4_math_and_design/M5/THEOREM_M5.md"
```

**Aggregated Output:**

```text
32:applies_to: cwd=C:\Users\李则徐\Documents\ChatGPT\EM-FloatFormer; reuse_rule=Reuse the matched-control, reproducibility, and publication approach for this repository; numerical results, commit, hardware status, and receiver outcomes are P4-specific.
204:# Task Group: Local TAN state-mechanism research, frozen protocol, and trace-pack stop
24:- Follow-up receiver experiments reduced errors on unseen delay distributions, but the pure Fourier/逐径 pilot front-end alone achieved nearly all of the gain. Therefore do not attribute this improvement to Transformer attention without matched-front-end controls.

5:## 1. Minimal sufficiency for the task union
27:### 1.1 Exact Gaussian experiment
39:### 1.2 Noisy generalization of the 2-bit versus 4-bit pair
67:## 2. Exact operational rate–risk frontier
92:### 2.1 Continuous trajectories and identifiable events
98:## 3. Audit of s17 and the shared cache format
108:### 3.1 Bit-exact extension
125:### 3.2 Stage 4→6 connector contract
136:## 4. One-window Bayes EIG
154:Under either true hypothesis, the log likelihood ratio in its direction has the law of \(L\), which proves the scalar formula. Composite trajectories use the full predictive mixture; mean separation alone is insufficient. [Lindley, 1956](https://projecteuclid.org/euclid.aoms/1177728069)
166:## 5. CPU verification
172:### 5.1 Achieved rate–risk points
189:### 5.2 Rerun and interpretation
199:## 6. Round 2 handoff
Length LastWriteTime      
------ -------------      
513326 2026/11/26 19:32:36


```

### Event #40: `item.completed` — `web_search` (`exec-bf17d331-5309-4689-8c54-4e85602f4e2a`)

```json
{
  "id": "exec-bf17d331-5309-4689-8c54-4e85602f4e2a",
  "type": "web_search",
  "query": "Lindley On a Measure of the Information Provided by an Experiment doi 117772 986 1005 Project Euclid",
  "action": {
    "type": "search",
    "query": "Lindley On a Measure of the Information Provided by an Experiment doi 117772 986 1005 Project Euclid"
  },
  "results": [
    {
      "type": "text_result",
      "domain": "www.scilit.com",
      "ref_id": "turn2search0",
      "snippet": "# On a Measure of the Information Provided by an Experiment ... Lindley ... 27 (4) , 986-1005 * DOI:10.1214/aoms/1177728069",
      "title": "On a Measure of the Information Provided by an Experiment | Scilit",
      "url": "https://www.scilit.com/publications/7e5b7d7aa0a06d74704ecbb120af7902"
    },
    {
      "type": "text_result",
      "domain": "itts023d.itts.ttu.edu",
      "ref_id": "turn2search1",
      "snippet": "Name Lindley D (1956) ... Title On a measure of information provided by an experiment ... First page 986 Last page 1005",
      "title": "new DB - Paper - Lindley D (1956)",
      "url": "https://itts023d.itts.ttu.edu/_site/Data/77377"
    },
    {
      "type": "text_result",
      "domain": "www.mathnet.ru",
      "ref_id": "turn2search2",
      "snippet": "Lindley, “On a measure of the information provided by an experiment”, Matematika, 3:3 (1959), 87–104; Ann. ... Statistics, 27 (1956), 986–1005 Image Citation in format",
      "title": "D. V. Lindley, “On a measure of the information provided by an experiment”, Matematika, 3:3 (1959), 87–104; Ann. Math. Statistics, 27 (1956), 986–1005",
      "url": "https://www.mathnet.ru/php/archive.phtml?jrnid=mat&option_lang=eng&paperid=101&wshow=paper"
    },
    {
      "type": "text_result",
      "domain": "bibbase.org",
      "ref_id": "turn2search3",
      "snippet": "On a Measure of the Information Provided by an Experiment. Lindley, D. ... The Annals of Mathematical Statistics, 27(4):986–1005, December, 1956. ... Finally, the measure",
      "title": "On a Measure of the Information Provided by an Experiment",
      "url": "https://bibbase.org/network/publication/lindley-onameasureoftheinformationprovidedbyanexperiment-1956"
    },
    {
      "type": "text_result",
      "domain": "m.mathnet.ru",
      "ref_id": "turn2search4",
      "snippet": "On a measure of the information provided by an experiment ... Lindley^{} ... English version: , 1956, 27, 986–1005",
      "title": "D. V. Lindley, “On a measure of the information provided by an experiment”, Matematika, 1959, Volume 3, Issue 3,Pages <nobr>87–104</nobr>",
      "url": "https://m.mathnet.ru/php/archive.phtml?jrnid=mat&option_lang=eng&paperid=101&wshow=paper"
    },
    {
      "type": "text_result",
      "domain": "onlinelibrary.wiley.com",
      "ref_id": "turn2search5",
      "snippet": "On a measure of information provided by an experiment. ... 27, 986–1005. ... Lindley and A. ... A 145, 117–126.",
      "title": "References",
      "url": "https://onlinelibrary.wiley.com/doi/10.1002/9780470316870.refs"
    },
    {
      "type": "text_result",
      "domain": "colab.ws",
      "ref_id": "turn2search6",
      "snippet": "# On a Measure of the Information Provided by an Experiment ... Lindley D. ... 986-1005. https://doi.org/10.1214/aoms/1177728069.",
      "title": "On a Measure of the Information Provided by an Experiment | CoLab",
      "url": "https://colab.ws/articles/10.1214%2Faoms%2F1177728069"
    },
    {
      "type": "text_result",
      "domain": "www.ivysci.com",
      "ref_id": "turn2search7",
      "snippet": "## On a Measure of the Information Provided by an Experiment ... Lindley DOI: 10.1214/aoms/1177728069 Image: Papermap",
      "title": "On a Measure of the Information Provided by an Experiment · ivySCI",
      "url": "https://www.ivysci.com/en/articles/6054655__On_a_Measure_of_the_Information_Provided_by_an_Experiment"
    },
    {
      "type": "text_result",
      "domain": "www.scirp.org",
      "ref_id": "turn2search8",
      "snippet": "Lindley, “On a Measure of the Information Provided by an Experiment,” The Annals of Mathematical Statistics, Vol. ... 986-1005. doi:10.1214/aoms/1177728069",
      "title": "D. V. Lindley, “On a Measure of the Information Provided by an Experiment,” The Annals of Mathematical Statistics, Vol. 27, No. 4, 1956, pp. 986-1005. doi10.1214/aoms/1177728069 - References - Scientific Research Publishing",
      "url": "https://www.scirp.org/reference/referencespapers?referenceid=244525"
    },
    {
      "type": "text_result",
      "domain": "openreview.net",
      "ref_id": "turn2search12",
      "snippet": "Lindley. On a Measure of the Information Provided by an Experiment. *The Annals of Mathematical Statistics*, 27(4): 986 – 1005, 1956. doi: 10.1214/aoms/1177728069. ... Rae,",
      "title": "D. V. Lindley. On a Measure of the Information Provided by an Experiment. *The Annals of Mathematical Statistics*, 27(4): 986 – 1005, 1956. doi: 10.1214/aoms/1177728069. URL https://doi.org/10.1214/aoms/1177728069.",
      "url": "https://openreview.net/notes/edits/attachment?id=dglx6bIWII&name=pdf"
    },
    {
      "type": "text_result",
      "domain": "www.cambridge.org",
      "ref_id": "turn2search9",
      "snippet": "The purpose of this paper is to construct a theory of the amount of information provided by an experiment which does not rely on what",
      "title": "The theory of Information and statistical inference. I",
      "url": "https://www.cambridge.org/core/journals/journal-of-applied-probability/article/abs/theory-of-information-and-statistical-inference-i/10B8A0A838248DCEC71E09CBEED5B8CD"
    },
    {
      "type": "text_result",
      "domain": "ageconsearch.umn.edu",
      "ref_id": "turn2search13",
      "snippet": "(275) Lindley, D, V, \"Binomial Sampling Schemes and the Con- ... 986-1005, A measure is introduced of the information provided by an ex- ... (277)",
      "title": "The World’s Largest Open Access Agricultural & Applied Economics Digital Library",
      "url": "https://ageconsearch.umn.edu/nanna/record/278877/files/ages840312.pdf?registerDownload=1&version=1&withMetadata=0&withWatermark=0"
    },
    {
      "type": "text_result",
      "domain": "synapsesocial.com",
      "ref_id": "turn2search10",
      "snippet": "# On a Measure of the Information Provided by an Experiment ... Lindley ... * To introduce a formal measure of the information yielded by",
      "title": "On a Measure of the Information Provided by an Experiment",
      "url": "https://synapsesocial.com/papers/69ef64bde63c05d5c9b25529"
    },
    {
      "type": "text_result",
      "domain": "ageconsearch.umn.edu",
      "ref_id": "turn2search14",
      "snippet": "(275) Lindley, D. ... \"On a Measure of the Information Provided by an Experiment.\" ... _ 27(1956): 986–1005.",
      "title": "(274) Lavalle, I. H. \"On Value and Strategic Role of Information in Semi-Normalized Decisions.\" _Op. Res._ 28(1980): 129–38.",
      "url": "https://ageconsearch.umn.edu/record/278877/files/ages840312.pdf"
    },
    {
      "type": "text_result",
      "domain": "alpha.rankless.org",
      "ref_id": "turn2search11",
      "snippet": "You can also color the map by specialization and compare the number of citations received by On a Measure of the Information Provided by an",
      "title": "Rankless | On a Measure of the Information Provided by an Experiment",
      "url": "https://alpha.rankless.org/hit-papers/10.1214/aoms/1177728069"
    },
    {
      "type": "text_result",
      "domain": "bmeder.org",
      "ref_id": "turn2search15",
      "snippet": "Lindley, D. ... On the measure of the information provided by an experiment. *Annals of Mathematical Statistics, 27*, 986–1005. http://dx.doi.org/10.1214/aoms/1177728069 ... On the interpretation of",
      "title": "1296  \nWU, MEDER, FILIMON, AND NELSON\n\nCokely, E.",
      "url": "https://bmeder.org/publication/2017-01-01_wu_2017turtle/2017-01-01_wu_2017turtle.pdf"
    },
    {
      "type": "text_result",
      "domain": "www.ros.hw.ac.uk",
      "ref_id": "turn2search16",
      "snippet": "Suggesting a measure of information provided by an experiment whose objective ... The following is the deﬁnition of the Lindley information measure given for a",
      "title": "Inference and Experimental Design for",
      "url": "https://www.ros.hw.ac.uk/bitstream/10399/2341/1/BejanA_0610_macs.pdf"
    },
    {
      "type": "text_result",
      "domain": "homepages.cwi.nl",
      "ref_id": "turn2search17",
      "snippet": "Lindley, D. ... On a measure of the information provided by an experi- ... *Annals of Mathematical Statistics*, **27**, 986–1005.",
      "title": "Lindley, D. V. (1956). On a measure of the information provided by an experi-",
      "url": "https://homepages.cwi.nl/~pdg/ftp/223.pdf"
    },
    {
      "type": "text_result",
      "domain": "en.wikipedia.org",
      "ref_id": "turn2search18",
      "snippet": "Lindley (1956) noted that the expected utility will then be coordinate-independent and can be written in two forms ... (1956), \"On a measure of information",
      "title": "Bayesian experimental design",
      "url": "https://en.wikipedia.org/wiki/Bayesian_experimental_design"
    },
    {
      "type": "text_result",
      "domain": "arxiv.org",
      "ref_id": "turn2view0",
      "snippet": "Total lines: 551",
      "title": "LEMMO: A Large Electromagnetic Model leveraging Pretrained LLMs for Million-Sample Signal Understanding",
      "url": "https://arxiv.org/html/2611.09476v1"
    }
  ]
}
```

### Event #41: `item.completed` — `file_change` (`item_39`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\THEOREM_M5.md",
    "kind": "update"
  }
]
```

### Event #42: `item.completed` — `command_execution` (`item_35`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$m5 = (Resolve-Path '"'research/stage4_math_and_design/M5').Path
"'$env:UV_CACHE_DIR = Join-Path $m5 '"'.uv-cache'
"'$env:UV_PYTHON_INSTALL_DIR = Join-Path $m5 '"'.uv-python'
"'$env:TMP = $m5
$env:TEMP = $m5
$env:UV_LINK_MODE = '"'copy'
uv run --no-project --offline --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M5/verify_m5.py"
```

**Aggregated Output:**

```text
flat_minus3: 2
flat_minus3: 4_s17
flat_minus3: 8
flat_minus3: 12
flat_minus3: 16
flat_minus3: canonical_continuous
flat_minus3: full_iq
flat_plus2: 2
flat_plus2: 4_s17
flat_plus2: 8
flat_plus2: 12
flat_plus2: 16
flat_plus2: canonical_continuous
flat_plus2: full_iq
flat_plus24: 2
flat_plus24: 4_s17
flat_plus24: 8
flat_plus24: 12
flat_plus24: 16
flat_plus24: canonical_continuous
flat_plus24: full_iq
selective_fade: 2
selective_fade: 4_s17
selective_fade: 8
selective_fade: 12
selective_fade: 16
selective_fade: canonical_continuous
selective_fade: full_iq
Wrote results_m5.json
```

### Event #43: `item.completed` — `command_execution` (`item_40`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$j = Get-Content research/stage4_math_and_design/M5/results_m5.json -Raw | ConvertFrom-Json
$j.protocol | ConvertTo-Json -Depth 5
$j.s17_inner_not_determined_by_matched_sum_and_residual_energy | ConvertTo-Json -Depth 6
foreach ($p in $j.per_profile) { $s = $p.metrics.'"'16'.stage6; Write-Output ("'$p.profile); foreach ($policy in @('"'max_EIG_window','random_window','nearest_window')) { "'$m=$s.PSObject.Properties[$policy].Value; $g=$m.common_ambiguous_subset; Write-Output "$policy n=$($g.n) NLL=$($g.nll) ACC=$($g.accuracy) BEFORE=$($g.nll_before) EIG=$($g.expected_eig) OBSERVED=$($g.observed_nll_gain)" }; $s.paired_ambiguous_nll_max_eig_minus_baseline | ConvertTo-Json -Depth 4 }
$vals = foreach ($p in $j.per_profile) { foreach ($method in @('"'4_s17','16','full_iq')) { "'$s = $p.metrics.PSObject.Properties[$method].Value.stage6; foreach ($policy in @('"'max_EIG_window','random_window','nearest_window')) { "'$m=$s.PSObject.Properties[$policy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m.nll; accuracy=$m.accuracy; e=$m.expected_eig; o=$m.observed_nll_gain} } } }
$vals | Group-Object method,policy | ForEach-Object { $g=$_.Group; [pscustomobject]@{method=$g[0].method;policy=$g[0].policy;nll=($g.nll|Measure-Object -Average).Average;acc=($g.accuracy|Measure-Object -Average).Average;eg=($g.e|Measure-Object -Average).Average;obs=($g.o|Measure-Object -Average).Average} } | Format-Table -AutoSize'
```

**Aggregated Output:**

```text
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
{
    "matched_sum_re_im":  [
                              0.6254985615161947,
                              0.0
                          ],
    "residual_energy":  2.0,
    "codes":  [
                  2,
                  0
              ],
    "iq_re_im":  [
                     [
                         [
                             1.3611317628690986,
                             0.0
                         ],
                         [
                             -0.6388682371309014,
                             0.0
                         ],
                         [
                             0.3611317628690986,
                             0.0
                         ]
                     ],
                     [
                         [
                             -0.6388682371309014,
                             0.0
                         ],
                         [
                             1.3611317628690986,
                             0.0
                         ],
                         [
                             0.3611317628690986,
                             0.0
                         ]
                     ]
                 ]
}
flat_minus3
max_EIG_window n=2048 NLL=0.6373868529066015 ACC=0.6181640625 BEFORE=0.6934564935341094 EIG=0.055278224796708554 OBSERVED=0.05606964062750787
random_window n=2048 NLL=0.6741712799713557 ACC=0.5595703125 BEFORE=0.6934564935341093 EIG=0.021628801501349264 OBSERVED=0.019285213562753603
nearest_window n=2048 NLL=0.6928158132608737 ACC=0.5302734375 BEFORE=0.6934564935341093 EIG=0.005348212560570385 OBSERVED=0.0006406802732355544
{
    "random_window":  {
                          "mean":  -0.036784427064754266,
                          "pointwise_95ci":  [
                                                 -0.04926814047819495,
                                                 -0.02430071365131358
                                             ]
                      },
    "nearest_window":  {
                           "mean":  -0.055428960354272326,
                           "pointwise_95ci":  [
                                                  -0.06923979280997741,
                                                  -0.041618127898567246
                                              ]
                       }
}
flat_plus2
max_EIG_window n=2048 NLL=0.5433659925603104 ACC=0.72265625 BEFORE=0.6723491674302535 EIG=0.12352952391180036 OBSERVED=0.12898317486994318
random_window n=2048 NLL=0.6107226834287436 ACC=0.65771484375 BEFORE=0.6723491674302535 EIG=0.058179053136990594 OBSERVED=0.061626484001509954
nearest_window n=2048 NLL=0.6444620008173143 ACC=0.6328125 BEFORE=0.6723491674302535 EIG=0.027008880917114536 OBSERVED=0.02788716661293916
{
    "random_window":  {
                          "mean":  -0.06735669086843322,
                          "pointwise_95ci":  [
                                                 -0.08675170188950851,
                                                 -0.04796167984735794
                                             ]
                      },
    "nearest_window":  {
                           "mean":  -0.10109600825700402,
                           "pointwise_95ci":  [
                                                  -0.12251405603513224,
                                                  -0.07967796047887579
                                              ]
                       }
}
flat_plus24
max_EIG_window n=0 NLL= ACC= BEFORE= EIG= OBSERVED=
random_window n=0 NLL= ACC= BEFORE= EIG= OBSERVED=
nearest_window n=0 NLL= ACC= BEFORE= EIG= OBSERVED=
selective_fade
max_EIG_window n=2048 NLL=0.2369391731051702 ACC=0.90625 BEFORE=0.6764738430724551 EIG=0.45416231110649774 OBSERVED=0.4395346699672849
random_window n=2048 NLL=0.5140986804080132 ACC=0.71630859375 BEFORE=0.6764738430724551 EIG=0.1582458250172228 OBSERVED=0.1623751626644419
nearest_window n=2048 NLL=0.5713970088345859 ACC=0.7021484375 BEFORE=0.6764738430724551 EIG=0.0864575992833798 OBSERVED=0.10507683423786923
{
    "random_window":  {
                          "mean":  -0.27715950730284306,
                          "pointwise_95ci":  [
                                                 -0.30428509055116754,
                                                 -0.2500339240545186
                                             ]
                      },
    "nearest_window":  {
                           "mean":  -0.3344578357294158,
                           "pointwise_95ci":  [
                                                  -0.3651989669017925,
                                                  -0.3037167045570391
                                              ]
                       }
}
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:6 �ַ�: 268
+ ... icy].Value; [pscustomobject]@{method=$method; policy=$policy; nll=$m. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
```

### Event #44: `item.completed` — `command_execution` (`item_41`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$j = Get-Content research/stage4_math_and_design/M5/results_m5.json -Raw | ConvertFrom-Json
$ps = @('"'max_EIG_window','random_window','nearest_window')
foreach ("'$pol in $ps) { $vals = foreach ($p in $j.per_profile) { $p.metrics.'"'16'.stage6.PSObject.Properties["'$pol].Value }; $n = ($vals.nll | Measure-Object -Average).Average; $a = ($vals.accuracy | Measure-Object -Average).Average; Write-Output "$pol $n $a" }
$gut = foreach ($p in $j.per_profile) { foreach ($m in $p.metrics.PSObject.Properties) { $m.Value.stage6.quadrature_max_abs_delta_position_0_to_63 } }; $gut | Measure-Object -Maximum
$g=$j.per_profile[3].metrics.'"'16'.stage6.max_EIG_window; "'$g.observed_minus_expected_gain | ConvertTo-Json -Depth 3
$g.selected_counts | ConvertTo-Json -Depth 3
$j.finite_pointwise_nondominated_including_rate -join '"', '"
```

**Aggregated Output:**

```text
max_EIG_window 0.354423004643021 0.811767578125
random_window 0.449748160952028 0.7333984375
nearest_window 0.477168705728194 0.71630859375


Count    : 28
Average  : 
Sum      : 
Maximum  : 1.76759396181003E-06
Minimum  : 
Property : 

{
    "mean":  -0.014627641139212704,
    "pointwise_95ci":  [
                           -0.04046436742658255,
                           0.01120908514815714
                       ]
}
{
    "11":  0,
    "1":  0,
    "4":  2048,
    "8":  0
}
2, 4_s17, 8, 12, 16


```

### Event #45: `item.completed` — `command_execution` (`item_42`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$j = Get-Content research/stage4_math_and_design/M5/results_m5.json -Raw | ConvertFrom-Json
foreach ($p in $j.per_profile) { Write-Output $p.profile; foreach ($m in @('"'4_s17','16','full_iq')) { Write-Output "'$m; $p.metrics.PSObject.Properties[$m].Value.stage6.max_EIG_window.selected_counts | ConvertTo-Json -Compress} }'
```

**Aggregated Output:**

```text
flat_minus3
4_s17
{"11":0,"1":0,"4":2030,"8":18}
16
{"11":0,"1":0,"4":2024,"8":24}
full_iq
{"11":0,"1":0,"4":2022,"8":26}
flat_plus2
4_s17
{"11":0,"1":0,"4":2035,"8":13}
16
{"11":2,"1":0,"4":2012,"8":34}
full_iq
{"11":1,"1":0,"4":2014,"8":33}
flat_plus24
4_s17
{"11":0,"1":36,"4":1990,"8":22}
16
{"11":484,"1":166,"4":1398,"8":0}
full_iq
{"11":247,"1":158,"4":1478,"8":165}
selective_fade
4_s17
{"11":0,"1":0,"4":2048,"8":0}
16
{"11":0,"1":0,"4":2048,"8":0}
full_iq
{"11":0,"1":0,"4":2048,"8":0}
```

### Event #46: `item.completed` — `file_change` (`item_43`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\THEOREM_M5.md",
    "kind": "update"
  }
]
```

### Event #47: `item.completed` — `file_change` (`item_44`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\reply_r1.md",
    "kind": "add"
  }
]
```

### Event #48: `item.completed` — `command_execution` (`item_46`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '"'^##|'"\\]\\(' research/stage4_math_and_design/M5/reply_r1.md research/stage4_math_and_design/M5/THEOREM_M5.md
rg -n -m 4 'sigma_phi|repeat_x1|reference' research/stage3d_qwen_signed_phase/data_stage3d.py
git status --short --untracked-files=all -- research/stage4_math_and_design/M5"
```

**Aggregated Output:**

```text
research/stage4_math_and_design/M5/THEOREM_M5.md:3:Round 1 studies a fixed pre-question cache and a single sealed candidate observation. The mathematical claims specify the observation law they require. The CPU experiment uses matched synthetic IQ. Pluto, CTTC, and the host comparison remain separate tests. [Long-term research plan](../../../docs/LONG_TERM_RESEARCH_PLAN.md) · [Integration plan](../../../docs/MULTIMODAL_INTEGRATION_PLAN.md)
research/stage4_math_and_design/M5/THEOREM_M5.md:5:## 1. Minimal sufficiency for the task union
research/stage4_math_and_design/M5/THEOREM_M5.md:25:If \(A,Y_a\) are conditionally independent of old observations given \(U\), \(P(U\mid y)\) suffices for all these readouts. It can be larger than \(\mathcal K\) when distinct latent states give the same answers and predictive mixtures. To preserve all priors and decisions on \(U\), the likelihood-ratio criterion identifies \(y,y'\) iff \(p(y\mid U)/p(y'\mid U)\) is constant in \(U\) on common positive support. Comparing all two-state priors gives the necessity. [Kullback and Leibler, 1951](https://doi.org/10.1214/aoms/1177729694)
research/stage4_math_and_design/M5/THEOREM_M5.md:27:### 1.1 Exact Gaussian experiment
research/stage4_math_and_design/M5/THEOREM_M5.md:39:### 1.2 Noisy generalization of the 2-bit versus 4-bit pair
research/stage4_math_and_design/M5/THEOREM_M5.md:65:This follows by subadditivity and \(H(Z)\le b\log2\), and is attained for integer \(b\le4\) by retaining \(b\) individual bits of \(D,U_1\). Q1 alone needs 2 bits for exact recovery and the pair of queries needs 4. More generally, a finite noiseless experiment needs \(\lceil\log_2 N_q\rceil\) or \(\lceil\log_2 N_\cup\rceil\) bits, where \(N\) counts distinct required kernels on supported old observations. These counts also apply to any finite multihypothesis event and forecast experiment. [Noiseless example in the plan](../../../docs/LONG_TERM_RESEARCH_PLAN.md)
research/stage4_math_and_design/M5/THEOREM_M5.md:67:## 2. Exact operational rate–risk frontier
research/stage4_math_and_design/M5/THEOREM_M5.md:96:### 2.1 Continuous trajectories and identifiable events
research/stage4_math_and_design/M5/THEOREM_M5.md:102:## 3. Audit of s17 and the shared cache format
research/stage4_math_and_design/M5/THEOREM_M5.md:104:The released s17 training generator draws coherent unit-energy QPSK with **additive Gaussian noise** at 2 dB. It uses three complex samples with mean magnitude \(1/\sqrt3\) and fits to analytic noisy-label posteriors. The b8 name means eight bits for the pair. The turn comes from the sum, with a first-nonzero-sample fallback if that sum is zero. The encoder rotates the three samples before the six-input neural network and four-center assignment. The learned inner label is a codebook index. An angular quantizer cannot in general be assumed to produce that index. Stage 3C shifts Q1 by the difference of turns and Q2/Q3-repeat by the first turn. Its Q2 kernel deduplicates acquisition IDs, ignores known independent references, and is calibrated for its pinned pilot time and SNR. The signed-CFO Stage 3D evaluates s17 outside the encoder's coherent training condition. [Encoder and training](../../matched_cache/learned.py) · [Training driver](../../matched_cache/train.py) · [Paired generator](../../fourphase_paired_questions/paired.py) · [Stage 3C quotient](../../stage3c/G/quotient.py) · [Stage 3C Q2 reader](../../stage3c/A/q2_reader.py) · [Stage 3D data](../../stage3d_qwen_signed_phase/data_stage3d.py)
research/stage4_math_and_design/M5/THEOREM_M5.md:106:On Stage 3D G1, physical s17 reaches 64.62% / 0.94113 nat and Full IQ reaches 64.64% / 0.92904 nat. The observed **cache gap** is about 0.01209 nat. The phase host reaches 23.80% / 1.4452 nat, a **readout gap** of 0.5041 nat behind physical s17. This comparison refers to 2 dB Q3-repeat, signed \(\omega=\pm0.40\), target diffusion variance 0.04 and the published decoders. It measures NLL and accuracy. The numbers give no CPU latency measurement. For Bayes readers the exact criterion for \(\epsilon\)-near-sufficiency is \(I(D_q;Y\mid Z,A_q)\le\epsilon\). [Stage 3D G1 report](../../stage3d_qwen_signed_phase/REPORT.md)
research/stage4_math_and_design/M5/THEOREM_M5.md:112:### 3.1 Bit-exact extension
research/stage4_math_and_design/M5/THEOREM_M5.md:129:### 3.2 Stage 4→6 connector contract
research/stage4_math_and_design/M5/THEOREM_M5.md:138:The same numeric modules can serve Ministral 3B and Qwen 4B/9B/27B. Adapt only the host embedding width, special tokens or chat template and language readout calibration. Empirical transfer still requires fitting and testing each adapter. The 9B to Qwen3.8-27B comparison changes generation as well as scale. For LEMMO, report its native instruction-conditioned workflow and a separate matched-access comparison where supported. Use the same old observations, stored/effective precision, reveal allowance, questions and acquisition splits. To attribute a gap to encoding, also match training data, trainable parameters and optimization budget. Report context, latency, old IQ rereads, extra reads and unsupported tasks. LEMMO's published workflow uses Qwen3.5-4B and an instruction-conditioned retrieval stage, with LoRA during joint fine-tuning. [Host plan](../../../docs/MULTIMODAL_INTEGRATION_PLAN.md) · [LEMMO, Sections 3 and B](https://arxiv.org/html/2611.09476v1)
research/stage4_math_and_design/M5/THEOREM_M5.md:140:## 4. One-window Bayes EIG
research/stage4_math_and_design/M5/THEOREM_M5.md:158:Under either true hypothesis, the log likelihood ratio in its direction has the law of \(L\), which proves the scalar formula. Composite trajectories use the full predictive mixture; mean separation alone is insufficient. [Lindley, 1956](https://projecteuclid.org/euclid.aoms/1177728069)
research/stage4_math_and_design/M5/THEOREM_M5.md:164:Under 0–1 loss the optimal success probability is \(\int\max(\pi_0 f_{0a},\pi_1 f_{1a})=(1+\|\pi_0f_{0a}-\pi_1f_{1a}\|_1)/2\). It can rank actions differently from EIG. A sufficient structural condition for dominance for both losses is Blackwell ordering. If a candidate observation can simulate the other by an \(H\)-independent garbling, the informative observation can emulate every policy for the garbled one. Data processing gives strict EIG improvement when \(H\) and the better observation remain dependent after seeing the garbling. Strict accuracy improvement requires the garbling to mix positive-probability predictive regions on opposite sides of the decision boundary. Two simple equal-variance Gaussian experiments are ordered by standardized mean separation: project onto the difference of means to get a scalar \(X=\pm d/2+N(0,1)\); simulate the smaller separation by scaling and adding independent Gaussian noise of the remaining variance. [Blackwell, 1951](https://digicoll.lib.berkeley.edu/record/112749)
research/stage4_math_and_design/M5/THEOREM_M5.md:170:## 5. CPU verification
research/stage4_math_and_design/M5/THEOREM_M5.md:172:[verify_m5.py](verify_m5.py) loads pinned s17 weights and generates three-sample complex windows with two known old pilots and two hidden payload symbols. Q1 reads relative payload phase. Q2 adds a valid late reference at the time origin. The Stage 5 prior gives equal mass to smooth, jump, independent and variance-burst models. It enumerates two smooth slopes, three jump intervals, a reset to independently drawn phase and slope, and a local sixfold noise variance. Old and candidate phases lie on a 128-point circle grid. The simulator uses that finite trajectory law without continuous drift diffusion. Stage 6 separately draws smooth/jump mothers with equal model priors and commits four candidate predictive mixtures before opening the selected recording per policy. A smooth/jump pair can have identical old endpoint means. The model permits post-jump fading, with class-dependent amplitude for some candidates. The maximum old gain swing in the selective fade profile is \(20\log_{10}(1.20/0.23)=14.35\) dB, across old SNRs of \(-3\) to \(+24\) dB.
research/stage4_math_and_design/M5/THEOREM_M5.md:174:Each of four profiles contributes 4,096 mothers to the old tasks and 2,048 other mothers to Stage 6. The s17 emission calibration uses 32,768 samples per canonical phase and distinct gain/variance configuration in a profile, an independent seed stream, C4 symmetry, and a half-count per cell. The canonical reader integrates Gaussian polar cell probabilities and uses the gamma CDF for the residual bins. The exact \((T,E)\) log likelihood was checked against a direct calculation from raw IQ over all simulated windows, phases and noise states. Stage 6 quadrature uses the continuously valued candidate matched sum and verifies an independent finer spacing on the first 64 mothers per method and profile. Candidate noise residuals have a common law at fixed address. The JSON keeps true and rival forecast scores, Bayes posterior NLL and accuracy, expected and observed NLL gains, action frequencies, paired intervals, and examples with the predictive distributions recorded before reveal. Because candidate noise scales and units affect differential log density, the raw observation NLL should be interpreted with the candidate and compared to the rival on the same reveal. [Full results](results_m5.json)
research/stage4_math_and_design/M5/THEOREM_M5.md:176:### 5.1 Achieved rate–risk points
research/stage4_math_and_design/M5/THEOREM_M5.md:193:### 5.2 Sealed one-window selection
research/stage4_math_and_design/M5/THEOREM_M5.md:207:### 5.3 Rerun and interpretation
research/stage4_math_and_design/M5/THEOREM_M5.md:217:## 6. Round 2 handoff
research/stage4_math_and_design/M5/reply_r1.md:5:## Exact results
research/stage4_math_and_design/M5/reply_r1.md:11:## CPU evidence
research/stage4_math_and_design/M5/reply_r1.md:24:The 16-bit cache recovers **0.312 nat** of joint event/location NLL versus s17, with a pointwise paired 95% interval of **0.302–0.321**, and leaves **0.074 nat** to the continuous oracle. Conditional purely smooth-versus-jump NLL improves by **0.0104 nat** pooled and is effectively tied under selective fading. The actual s17 network merges two strict interior points whose four-label posteriors differ by **0.480 total variation**. The G1 Stage 3D result remains a narrow near-sufficiency case. Its physical cache gap is **0.0121 nat**, and the host's additional readout gap is **0.5041 nat**. [Stage 3D report](../../stage3d_qwen_signed_phase/REPORT.md)
research/stage4_math_and_design/M5/reply_r1.md:26:On the 2,048 ambiguous selective-fade mothers, **max EIG gives 90.63% accuracy / 0.2369 NLL**, random gives 71.63% / 0.5141, and nearest gives 70.21% / 0.5714. The committed forecasts imply 0.4542 nat expected gain and yield 0.4395 nat realized gain. Every max-EIG selection in that profile is time 4. A fixed time-4 policy would tie there. The synthetic run tests the value of reference geometry and leaves the incremental gain from recording-specific adaptation for a stronger design. Pluto/CTTC, host transfer and LEMMO performance remain live evaluation questions. [Full results](results_m5.json)
research/stage4_math_and_design/M5/reply_r1.md:28:## Round 2
research/stage4_math_and_design/M5/reply_r1.md:35:[Theorems and interface](THEOREM_M5.md) · [CPU replay](verify_m5.py) · [Raw results](results_m5.json)
30:    dedup_references,
77:    sigma_phi: float
91:        return float((self.sigma_phi ** 2) * (self.h + self.ell))
103:        sigma_phi=0.10,
?? research/stage4_math_and_design/M5/THEOREM_M5.md
?? research/stage4_math_and_design/M5/reply_r1.md
?? research/stage4_math_and_design/M5/results_m5.json
?? research/stage4_math_and_design/M5/verify_m5.py
```

### Event #49: `item.completed` — `command_execution` (`item_45`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$m5=(Resolve-Path '"'research/stage4_math_and_design/M5').Path
"'$r=Get-Content (Join-Path $m5 '"'results_m5.json') -Raw | ConvertFrom-Json
if ("'$r.protocol.python -ne '"'3.12.7' -or "'$r.protocol.numpy -ne '"'2.3.5' -or "'$r.protocol.scipy -ne '"'1.17.0') { throw 'Unexpected replay versions' }
if ("'$r.per_profile.Count -ne 4 -or ($r.finite_pointwise_nondominated_including_rate -join '"',') -ne '2,4_s17,8,12,16') { throw 'Unexpected sweep dimensions' }
foreach ("'$p in $r.per_profile) { foreach ($m in $p.metrics.PSObject.Properties) { $s=$m.Value.stage6; if ($s.commit_sha256 -notmatch '"'"'^[a-f0-9]{64}$'"') { throw 'Missing commitment' }; foreach ("'$policy in @('"'max_EIG_window','nearest_window','random_window')) { "'$run=$s.PSObject.Properties[$policy].Value; $n=($run.selected_counts.PSObject.Properties.Value | Measure-Object -Sum).Sum; if ($n -ne $r.protocol.mothers_per_profile_blind) { throw '"'Unexpected reveal count' } }; if ("'$s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_window.expected_eig,$s.nearest_window.expected_eig)-1e-10) { throw '"'Selected EIG below baseline' } } }
Get-ChildItem "'$m5 -Force | Select-Object Name,Mode,Length
'"'Checked all 28 method/profile commitment rows and 84 policy counts'"
```

**Aggregated Output:**

```text
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 
�޷����÷�����������ģʽ��֧�ֺ������͵ķ������á�
����λ�� ��:6 �ַ�: 476
+ ... t' } }; if ($s.max_EIG_window.expected_eig -lt [Math]::Max($s.random_ ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : MethodInvocationNotSupportedInConstrainedLanguage
 

Name            Mode   Length
----            ----   ------
.uv-cache       d-----       
reply_r1.md     -a---- 4498  
results_m5.json -a---- 548881
THEOREM_M5.md   -a---- 36843 
verify_m5.py    -a---- 31660 
Checked all 28 method/profile commitment rows and 84 policy counts


```

### Event #50: `item.completed` — `command_execution` (`item_47`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$ErrorActionPreference='"'Stop'
"'$m5=(Resolve-Path -LiteralPath '"'research/stage4_math_and_design/M5').Path
"'$r=Get-Content (Join-Path $m5 '"'results_m5.json') -Raw | ConvertFrom-Json
if ("'$r.protocol.python -ne '"'3.12.7' -or "'$r.protocol.numpy -ne '"'2.3.5' -or "'$r.protocol.scipy -ne '"'1.17.0') { throw 'Unexpected replay versions' }
if ("'$r.per_profile.Count -ne 4 -or ($r.finite_pointwise_nondominated_including_rate -join '"',') -ne '2,4_s17,8,12,16') { throw 'Unexpected sweep dimensions' }
foreach ("'$p in $r.per_profile) { foreach ($m in $p.metrics.PSObject.Properties) { $s=$m.Value.stage6; if ($s.commit_sha256 -notmatch '"'"'^[a-f0-9]{64}$'"') { throw 'Missing commitment' }; foreach ("'$policy in @('"'max_EIG_window','nearest_window','random_window')) { "'$run=$s.PSObject.Properties[$policy].Value; $n=($run.selected_counts.PSObject.Properties.Value | Measure-Object -Sum).Sum; if ($n -ne $r.protocol.mothers_per_profile_blind) { throw '"'Unexpected reveal count' }; if ("'$s.max_EIG_window.expected_eig + 1e-10 -lt $run.expected_eig) { throw '"'Selected EIG below baseline' } } } }
'Checked all 28 method/profile commitment rows and 84 policy counts'
"'$requestedCache=Join-Path $m5 '"'.uv-cache'
"'$actualCache=Get-Item -LiteralPath $requestedCache -Force
if ($actualCache.FullName -cne $requestedCache -or $actualCache.Parent.FullName -cne $m5 -or ($actualCache.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw '"'Temporary cache path check failed' }
"'$actualCache.FullName
uv cache clean --cache-dir research/stage4_math_and_design/M5/.uv-cache'
```

**Aggregated Output:**

```text
Checked all 28 method/profile commitment rows and 84 policy counts
C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M5\.uv-cache
Clearing cache at: research/stage4_math_and_design/M5/.uv-cache
Removed 4951 files (295.5MiB)
```

### Event #51: `item.completed` — `command_execution` (`item_48`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$ErrorActionPreference='"'Stop'
"'$m5=(Resolve-Path -LiteralPath '"'research/stage4_math_and_design/M5').Path
foreach ("'$file in @('"'THEOREM_M5.md','reply_r1.md')) { "'$path=Join-Path $m5 $file; $body=Get-Content -LiteralPath $path -Encoding utf8 -Raw; $body=$body.Replace('"'\\(', '"'$'"').Replace('\\)', '"'$'"').Replace('\\[', '"'$$'"').Replace('\\]', '"'$$'"'); Set-Content -LiteralPath "'$path -Value $body -Encoding utf8 -NoNewline }
Get-ChildItem -LiteralPath $m5 -Force | Select-Object Name,Length
rg -n '"'"'^'"\\"'$'"\\"'$|''^##'"' research/stage4_math_and_design/M5/THEOREM_M5.md | Select-Object -First 15
git status --short --untracked-files=all -- research/stage4_math_and_design/M5"
```

**Aggregated Output:**

```text

?? research/stage4_math_and_design/M5/THEOREM_M5.md
?? research/stage4_math_and_design/M5/reply_r1.md
?? research/stage4_math_and_design/M5/results_m5.json
?? research/stage4_math_and_design/M5/verify_m5.py
Name            Length
----            ------
reply_r1.md       4497
results_m5.json 548881
THEOREM_M5.md    36462
verify_m5.py     31660
5:## 1. Minimal sufficiency for the task union
11:$$
19:$$
27:### 1.1 Exact Gaussian experiment
31:$$
33:$$
39:### 1.2 Noisy generalization of the 2-bit versus 4-bit pair
43:$$
46:$$
52:$$
54:$$
61:$$
64:$$
67:## 2. Exact operational rate–risk frontier
71:$$


```

### Event #52: `item.completed` — `agent_message` (`item_49`)

**Round 2 should carry a 16-bit canonical cache with the exact 4-bit s17 prefix.** Keep 12 bits as the cost comparison and continuous complex evidence plus residual energy as the physical oracle.

- **Exact limits.** The minimal shared statistic preserves the answer posteriors and required candidate predictive laws. Under log loss, compression costs exactly the conditional information it discards. The finite-bit frontier depends on the full observation law, including fading, reference SNR, lag, and hypothesis priors. With equal read costs and calibrated predictions, maximum EIG strictly improves expected posterior NLL over a baseline exactly when that baseline sometimes chooses an inferior window. [Lindley, 1956](https://doi.org/10.1214/aoms/1177728069)
- **Cache result.** The pinned CPU replay reduced joint event-and-location NLL from **1.2788** with s17 to **0.9669** at 16 bits, versus **0.8934** for Full IQ. The pooled smooth-versus-jump gain was smaller, at **0.0104 nat**. On the earlier Stage 3D G1 task, the s17 cache gap was **0.0121 nat** and the additional host readout gap was **0.5041 nat**. [Stage 3D report](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage3d_qwen_signed_phase/REPORT.md)
- **Blind-test result.** On 2,048 ambiguous selective-fade recordings, max EIG reached **90.63% accuracy / 0.2369 NLL**, versus **71.63% / 0.5141** for random and **70.21% / 0.5714** for nearest. EIG chose time 4 every time in that profile. A fixed time-4 policy therefore ties it there, leaving the value of recording-specific adaptation to test with competing candidate geometries.

**Round 2 handoff.** M1 should freeze the shared physical interface and legacy prefix. M3 should preserve acquisition IDs and timestamps through reference fusion. M4 should validate gain, noise and residual calibration on held-out Pluto and CTTC captures. M6 should freeze per-query risk margins and a fixed-best-window control before host comparisons. The replay used synthetic IQ with shared nominal calibration. Real-capture performance, host transfer and the LEMMO comparison remain unverified.

[Executive summary](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M5/reply_r1.md) · [Theorems and design](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M5/THEOREM_M5.md) · [CPU verifier](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M5/verify_m5.py) · [Results JSON](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M5/results_m5.json)



### Event #53: `turn.completed`

- Usage: `{"input_tokens": 4373288, "cached_input_tokens": 4232576, "cache_write_input_tokens": 0, "output_tokens": 50960, "reasoning_output_tokens": 9351}`


## Round 1 — Final Output (`reply_r1.md`)

```markdown
**Round 2 should carry a 16-bit canonical cache with the exact 4-bit s17 prefix.** Keep 12 bits as the cost comparison and continuous complex evidence plus residual energy as the physical oracle.

- **Exact limits.** The minimal shared statistic preserves the answer posteriors and required candidate predictive laws. Under log loss, compression costs exactly the conditional information it discards. The finite-bit frontier depends on the full observation law, including fading, reference SNR, lag, and hypothesis priors. With equal read costs and calibrated predictions, maximum EIG strictly improves expected posterior NLL over a baseline exactly when that baseline sometimes chooses an inferior window. [Lindley, 1956](https://doi.org/10.1214/aoms/1177728069)
- **Cache result.** The pinned CPU replay reduced joint event-and-location NLL from **1.2788** with s17 to **0.9669** at 16 bits, versus **0.8934** for Full IQ. The pooled smooth-versus-jump gain was smaller, at **0.0104 nat**. On the earlier Stage 3D G1 task, the s17 cache gap was **0.0121 nat** and the additional host readout gap was **0.5041 nat**. [Stage 3D report](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage3d_qwen_signed_phase/REPORT.md)
- **Blind-test result.** On 2,048 ambiguous selective-fade recordings, max EIG reached **90.63% accuracy / 0.2369 NLL**, versus **71.63% / 0.5141** for random and **70.21% / 0.5714** for nearest. EIG chose time 4 every time in that profile. A fixed time-4 policy therefore ties it there, leaving the value of recording-specific adaptation to test with competing candidate geometries.

**Round 2 handoff.** M1 should freeze the shared physical interface and legacy prefix. M3 should preserve acquisition IDs and timestamps through reference fusion. M4 should validate gain, noise and residual calibration on held-out Pluto and CTTC captures. M6 should freeze per-query risk margins and a fixed-best-window control before host comparisons. The replay used synthetic IQ with shared nominal calibration. Real-capture performance, host transfer and the LEMMO comparison remain unverified.

[Executive summary](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M5/reply_r1.md) · [Theorems and design](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M5/THEOREM_M5.md) · [CPU verifier](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M5/verify_m5.py) · [Results JSON](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M5/results_m5.json)

```


---

## Round 2 — Cross-Review & Convergence Prompt (`prompt_r2.md`)

```markdown
# Track M5 (Round 2): Non-Trivial Multi-Geometry Adaptive EIG, Exact 16-Bit/RefPool Integration & Cross-Host Stage 6 Lock

You are resuming persistent Codex session **Track M5** (`01a2194c-3b55-7153-bc9f-570b6f8c3c3f`) in `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`. Keep edits STRICTLY inside `research/stage4_math_and_design/M5/` and run Python ONLY via `uv run --python 3.12 ...`.

## 1. Cross-Track Findings & Round-1 Critique to Resolve (From M1–M6 & Self-Audit)
1. **Your Own Critical Finding in Round 1 (`fixed_time_4` Tie!)**:
   - In `results_m5.json` Round 1, `max_EIG` crushed `random` (`90.63% vs 71.63%`) and `nearest` (`70.21%`) on selective-fade — **yet time `4` won ALL 2,048 recordings**, meaning a static `fixed_best_on_dev = time 4` rule tied active per-recording `max_EIG`!
   - Wait: **what mathematically forces `max_EIG` to strictly dominate EVERY static `fixed_best_on_dev` window `a_const`?** Obviously whenever **recording-specific old cache `Z` carries conditional information about WHERE the ambiguity lies** (e.g., recording-varying selective deep fades seen via `log_radial(Z)`, OR heterogeneous change-point intervals $\tau$ localized differently per recording by `Z`, OR competing CFO slopes aliased on different candidate subsets per recording)!
2. **Locking the Unified 16-Bit Packet & Causal Real-Data Compatibility with M1, M3, M4**:
   - M4 showed that on real hardware without Stage 3's fixed SNR/amplitude, forcing SNR down to `2 dB` destroys `20 dB` (`0.00% -> 19.89%` SER), while M4's R3 whitened lift `(T, E)` preserves scalar precision cleanly, and M4 proposed **extracting BOTH `4-bit s17` and your `16-bit canonical` packet in Stage 4's zero-GPU CPU shard pass**!
   - Also check M3's `RefSufficientPool`: when revealing ONE candidate window $Y_{a^*}$ in Stage 6, $Y_{a^*}$ enters the same immutable `(acquisition_id, shares_channel, timestamp, K_ref, matched_T)` `RefSufficientPool` contract, handling shared CFO uncertainty $(\hat\omega, v_\omega)$ and correlated diffusion without double-counting!
3. **From M6 & M2 (Stage 6 Single Frozen Query Contract Across 4B / 9B / 27B & LEMMO)**:
   - M6 asked: can you freeze ONE concrete **Stage 6 Ultimate Blind Test Query & Evidence Contract** common to `Qwen3.5-4B`, `Qwen3.5-9B`, `Qwen3.8-27B`, and `LEMMO` under identical pre-question cache bits (`16 bits/window`), candidate read budget (`k = 1` sealed read), pre-reveal forecast commitment scoring, AND M2's paired statistical gate (requiring strict win over `random`, `nearest`, AND **`fixed_best_on_dev`**!)?

## 2. Specific Round-2 Convergence Tasks for Track M5
Update `research/stage4_math_and_design/M5/THEOREM_M5.md`, `verify_m5.py`, and `results_m5.json`:
1. **Prove & Empirically Demonstrate Strict Per-Recording Adaptive EIG Dominance Over `fixed_best_on_dev`**:
   - Prove the exact mutual-information condition $I(a^*(Z); Y_a \text{ utility} \mid \text{marg}) > 0$ under which `max_EIG(Z)` strictly improves expected posterior NLL and accuracy over $\max_{a \in \mathcal{C}} \mathbb{E}_Z[\dots]$ (`fixed_best_on_dev`), i.e., Bayes value of conditional adaptive experiment design!
   - Add a new deterministic benchmark section in `verify_m5.py` (**`multi_geometry_adaptive_suite`** — e.g., varying localized jump intervals + varying candidate-window fading reliability + CFO aliasing across recordings) and verify empirically via paired CIs that **`max_EIG(Z)` strictly beats ALL FOUR controls: `random`, `nearest`, AND `fixed_best_oracle/dev`** on both post-reveal NLL and accuracy!
2. **Harmonize the Exact Bit-Layout of the 16-Bit Packet & `M1/M3/M4` Stage 4→6 Integration Table**:
   - Align byte/bit fields cleanly with M4 (`log_radial`, `resid_quality`, `s17_inner`, `turn`, `canon_angle`, erasure bit when zero/missing), show how Stage 4 CPU extractors populate BOTH `4-bit s17` and `16-bit canonical` views, and lock the Stage 6 `4B / 9B / 27B / LEMMO` frozen benchmark specification!
3. Run updated `verify_m5.py` via `uv run` cleanly and summarize exact numbers (`max_EIG` vs `fixed_best_on_dev`, `random`, `nearest`) in `reply_r2.md`.
```


## Round 2 — Resumed Execution & Reasoning Trace (`round2_events.jsonl`)

### Event #1: `thread.started` (thread_id=`01a2194c-3b55-7153-bc9f-570b6f8c3c3f`)

### Event #2: `turn.started`

### Event #3: `item.completed` — `agent_message` (`item_0`)

I’ll check the other tracks’ interface contracts, then build a Round 2 benchmark where the best candidate changes across recordings. I’ll score max EIG against a dev-locked fixed window and an oracle fixed-window control, with paired intervals for NLL and accuracy.

### Event #4: `item.completed` — `command_execution` (`item_1`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n -i 'EM-FloatFormer|stage4_math|s17' 'C:/Users/李则徐/.codex/memories/MEMORY.md'
rg --files --hidden research/stage4_math_and_design | Where-Object { "'$_ -notmatch '"'events.jsonl' }"
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
research/stage4_math_and_design\transcript_utils.py
research/stage4_math_and_design\M2\verify_m2.py
research/stage4_math_and_design\M2\THEOREM_M2.md
research/stage4_math_and_design\M2\results_m2.json
research/stage4_math_and_design\M2\reply_r1.md
research/stage4_math_and_design\M4\verify_m4.py
research/stage4_math_and_design\M4\THEOREM_M4.md
research/stage4_math_and_design\M4\results_m4.json
research/stage4_math_and_design\M4\reply_r1.md
research/stage4_math_and_design\M1\verify_m1.py
research/stage4_math_and_design\discussions\M6\reply_r1.md
research/stage4_math_and_design\discussions\M6\prompt_r2.md
research/stage4_math_and_design\M1\THEOREM_M1.md
research/stage4_math_and_design\discussions\M6\prompt_r1.md
research/stage4_math_and_design\M1\results_m1.json
research/stage4_math_and_design\M5\verify_m5.py
research/stage4_math_and_design\M5\THEOREM_M5.md
research/stage4_math_and_design\discussions\M3\reply_r1.md
research/stage4_math_and_design\M5\results_m5.json
research/stage4_math_and_design\discussions\M3\prompt_r2.md
research/stage4_math_and_design\discussions\M3\prompt_r1.md
research/stage4_math_and_design\M5\reply_r1.md
research/stage4_math_and_design\M3\results_m3.json
research/stage4_math_and_design\M3\reply_r1.md
research/stage4_math_and_design\M3\THEOREM_M3.md
research/stage4_math_and_design\M3\verify_m3.py
research/stage4_math_and_design\discussions\M5\reply_r1.md
research/stage4_math_and_design\discussions\M5\prompt_r2.md
research/stage4_math_and_design\discussions\M5\prompt_r1.md
research/stage4_math_and_design\discussions\M2\reply_r1.md
research/stage4_math_and_design\discussions\M4\reply_r1.md
research/stage4_math_and_design\discussions\M2\prompt_r2.md
research/stage4_math_and_design\discussions\M4\prompt_r2.md
research/stage4_math_and_design\discussions\M2\prompt_r1.md
research/stage4_math_and_design\discussions\M4\prompt_r1.md
research/stage4_math_and_design\discussions\M1\prompt_r1.md
research/stage4_math_and_design\discussions\M1\prompt_r2.md
research/stage4_math_and_design\discussions\M1\reply_r1.md
research/stage4_math_and_design\M6\verify_m6.py
research/stage4_math_and_design\M6\THEOREM_M6.md
research/stage4_math_and_design\M6\results_m6.json
research/stage4_math_and_design\M6\reply_r1.md
```

### Event #5: `item.completed` — `command_execution` (`item_2`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "git status --short
rg -n '"'^def |''^class |''^##|LAYOUTS|finite_pointwise'"' research/stage4_math_and_design/M5/verify_m5.py research/stage4_math_and_design/M5/THEOREM_M5.md
foreach ("'$p in @('"'AGENTS.md','research/AGENTS.md','research/stage4_math_and_design/AGENTS.md','research/stage4_math_and_design/M5/AGENTS.md')) { if (Test-Path "'$p) { Get-Content -Encoding utf8 $p }}'
```

**Aggregated Output:**

```text
?? research/stage4_math_and_design/
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
research/stage4_math_and_design/M5/verify_m5.py:34:LAYOUTS = {"2": (0, 0, 0), "8": (2, 1, 1), "12": (4, 3, 1), "16": (6, 4, 2), "24": (8, 8, 4)}
research/stage4_math_and_design/M5/verify_m5.py:47:def normal(rng, shape):
research/stage4_math_and_design/M5/verify_m5.py:51:def parameters(snr, amplitude):
research/stage4_math_and_design/M5/verify_m5.py:56:def phase_index(theta):
research/stage4_math_and_design/M5/verify_m5.py:60:def make_tracks():
research/stage4_math_and_design/M5/verify_m5.py:92:def sample_old(rng, count, prior, phase, burst, amplitude, variance, ref_db):
research/stage4_math_and_design/M5/verify_m5.py:104:def features(iq):
research/stage4_math_and_design/M5/verify_m5.py:110:def radial_edges(bits):
research/stage4_math_and_design/M5/verify_m5.py:118:def quality_edges(bits):
research/stage4_math_and_design/M5/verify_m5.py:126:def encode(iq, t, residual, variance, encoder):
research/stage4_math_and_design/M5/verify_m5.py:134:    for method, (angle_bits, radial_bits, quality_bits) in LAYOUTS.items():
research/stage4_math_and_design/M5/verify_m5.py:156:def s17_emissions(encoder, rng, amplitude, variance):
research/stage4_math_and_design/M5/verify_m5.py:166:def polar_emissions(angle_bits, radial_bits, mean_amp, noise_ratio, order=18):
research/stage4_math_and_design/M5/verify_m5.py:198:def window_log_likelihood(method, t, residual, packets, bins, a, v, encoder, key):
research/stage4_math_and_design/M5/verify_m5.py:219:                angle_bits, radial_bits, quality_bits = LAYOUTS[method]
research/stage4_math_and_design/M5/verify_m5.py:227:def infer(windows, old_phase, burst, prior, reference=None, ref_variance=None):
research/stage4_math_and_design/M5/verify_m5.py:251:def score_rows(prob, target):
research/stage4_math_and_design/M5/verify_m5.py:256:def score(loss, hit):
research/stage4_math_and_design/M5/verify_m5.py:260:def paired_ci(diff):
research/stage4_math_and_design/M5/verify_m5.py:266:def components(new_phase, new_gain, branch):
research/stage4_math_and_design/M5/verify_m5.py:282:def density(z, means, variance):
research/stage4_math_and_design/M5/verify_m5.py:286:def integrate_eig(weights, means, variance, step):
research/stage4_math_and_design/M5/verify_m5.py:307:def digest_arrays(*arrays):
research/stage4_math_and_design/M5/verify_m5.py:314:def pick_before_reveal(states, groups, a, v, actions):
research/stage4_math_and_design/M5/verify_m5.py:336:class SealedCandidates:
research/stage4_math_and_design/M5/verify_m5.py:358:def score_revealed(forecasts, eig, actions, receipt, candidates, method, v, truth, ambiguous):
research/stage4_math_and_design/M5/verify_m5.py:434:def run_profile(key, profile, tracks, encoder):
research/stage4_math_and_design/M5/verify_m5.py:526:def collision_witness(encoder):
research/stage4_math_and_design/M5/verify_m5.py:553:def legacy_inner_witness(encoder):
research/stage4_math_and_design/M5/verify_m5.py:575:def main():
research/stage4_math_and_design/M5/verify_m5.py:614:            "cache_extensions_angle_radius_quality_bits": LAYOUTS,
research/stage4_math_and_design/M5/verify_m5.py:623:        "finite_pointwise_nondominated_including_rate": nondominated,
```

### Event #6: `item.completed` — `command_execution` (`item_3`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n -i 'R3|canonical|radial|resid|whiten|candidate|19.89|erasure|s17' research/stage4_math_and_design/M4/THEOREM_M4.md research/stage4_math_and_design/discussions/M4/prompt_r2.md research/stage4_math_and_design/M4/reply_r1.md"
```

**Aggregated Output:**

```text
research/stage4_math_and_design/M4/reply_r1.md:3:**Use the SNR-preserving R3 lift for the first real-data run.** Estimate timing, gain, DC and uncertainty from authorized causal observations. Put the gain-normalized scalar \(T\) along \(e=\mathbf1/\sqrt3\), and fill the two orthogonal directions with fixed-seed independent nominal Gaussian noise. This preserves `matched_sum = T` and makes the three-tap joint KL equal the scalar joint KL. The repeated R2 embedding preserves the scalar but has infinite three-tap KL due to rank deficiency. Use real three-tap R1 and forced 2 dB R3 as diagnostics. On the clean 20 dB surrogate, SNR-preserving R3 gave **0/8,192 symbol errors**, versus **19.89% SER** at forced 2 dB. The R2 and SNR-preserving R3 table SERs agree because the frozen table always selects the matched-sum quadrant. Local codes and NLL can differ. [Theorem and protocol](THEOREM_M4.md) · [CPU results](results_m4.json)
research/stage4_math_and_design/M4/reply_r1.md:5:**Misspecification.** With \(g=1+\delta\), thermal variance ratio \(\lambda=v/(g^2V_0)\), RMS complex units and \(V_0=10^{-0.2}\), scalar conditional KL is \(\lambda-1-\log\lambda+|\mu_s-s|^2/V_0\). The mean \(\mu_s\) contains the coherent sum of gain, DC, ISI, CFO and spur effects. Isolated DC of `0.30` and spur amplitude of `0.26` cost `0.142640` and `0.107139` nats at variance match. Residual `0.020 rad/symbol` at lags 4/25/126 costs `0.010138 / 0.388037 / 5.746671` nats. Correlated taps use the full log-determinant and trace terms. Marginalizing a nuisance subtracts \(I(U;Y\mid S)\) from the average conditional KL. The joint KL upper bounds the excess NLL of the nominal Bayesian posterior. The frozen table has an additional code and calibration term. An independent Rayleigh denominator gives infinite forward Gaussian KL. The centered scalar Student case with three complex residual degrees of freedom gives `0.166667` nats against nominal variance and `0.072132` nats after matching variance. Scale estimation that divides the signal creates a noncentral mixture. Along the verifier's canonical ray, the local code is certified constant up to gain `0.03206`, with a different saturated code certified above `21.89582`. The quadrant remains available for nonzero inputs. [Derivations](THEOREM_M4.md#1-observation-laws-and-losses)
research/stage4_math_and_design/M4/reply_r1.md:7:**Real benchmark.** Lock W1/W2 from one symbol each. Reveal one and then two genuinely observed past pilots without recomputing the cache. Use `15,18` for Pluto and `126,127` for CTTC. Q1 scores the transmitted relative turn; Q2 scores the transmitted symbol and yields SER, bit NLL and posterior-mean EVM; Q3 predicts the future received sector at a physical retransmission. Use the next Pluto copy and the next CTTC loop at the same TX source index. Register 2,240 Pluto frames in 16 blocks, 1,440 CTTC packets in 72 session-loops, and 720 CTTC Q3 pairs within 36 loop pairs. The CPU verifier certifies the archived split, source-bit and local interpolation geometry. Pluto's whole-record synchronization and CTTC's offline full-source lock are excluded from the causal frontend. Real-run qualification requires independently bounded timing plus a raw-support and timestamp audit. Use the raw-I/Q B3 mixture, direct `s17` reader, Qwen Zero, Naive-Additive, Equivariant-Coupled and RefSufficientPool on the registered rows with the same release schedule. [Benchmark design](THEOREM_M4.md#2-hardware-benchmark-and-causality-contract)
research/stage4_math_and_design/M4/reply_r1.md:13:- **M1.** Can the coupled scorer take integer lags, a circular posterior over CFO/jump hypotheses, and real SNR/AGC uncertainty while scoring one canonical host prefix per row? Lock the separate receive-gauge and TX-relabel actions, including the fixed Q2 TX labels and permuted Q3 receive sectors under receive-gauge rotation.
research/stage4_math_and_design/M4/reply_r1.md:15:- **M3.** Can RefSufficientPool consume timestamped unique pilot observations with per-hypothesis precision, matched correlation, residual energy and normalization, and emit a joint innovation when pilots share noise? Specify how the pool avoids counting a pilot already included in the causal state.
research/stage4_math_and_design/discussions/M4/prompt_r2.md:1:# Track M4 (Round 2): Unified A100-Safe Modal Budget, Dual `s17`+`16-bit` Causal Frontend & Full M1/M2/M3 Real-Data Lock
research/stage4_math_and_design/discussions/M4/prompt_r2.md:11:     - M2 proved `1` seed or `3` seeds CANNOT establish population-of-seeds generalization (`>= 10` seeds needed for that!), so Stage 4's real-data + synthetic GPU budget MUST explicitly state a **Fixed-Connector Conditional Scope** OR use `1` seed per each of the `4` matched arms (`Zero`, `Naive-Additive`, `Host-Pure Equivariant+RefPool`, `Physical+HostResidual` -> M1's converged 4-arm ablation!) + optional `2` extra seeds ONLY for the winning M1 `Host-Pure / Physical-Residual` pair if Stage-1 dev pilot clears M6's Tier-2 gate!
research/stage4_math_and_design/discussions/M4/prompt_r2.md:13:2. **Unified Causal Dual-Cache Extraction (`4-bit s17` + `16-bit canonical` M5 Packet) in ONE Zero-GPU CPU Pass**:
research/stage4_math_and_design/discussions/M4/prompt_r2.md:14:   - M5 proved `4-bit s17` saturates along rays (which you ALSO proved independently — `local` saturates to `[0,4,8,12]` above gain `21.90` and gives constant code below `0.032`!), causing Stage 5–6 NLL loss (`0.3119 nats`) and hiding SNR changes across CTTC (`-3..+24 dB`) and Pluto fading (`>10.6 dB`).
research/stage4_math_and_design/discussions/M4/prompt_r2.md:15:   - Since your CPU causal frontend ALREADY computes R3 whitened complex `T`, residual energy `E`, normalized radial SNR `r`, and `equivariant_b8_s17(Y_R3)`, extracting BOTH **`4-bit s17` codes** AND **M5's `16-bit canonical packet` `[2b turn | 2b legacy_inner | 6b canon_angle | 4b log_radial | 2b resid_quality]`** costs **`0` extra GPU time and `< 2 MB` storage**! Incorporate this dual export cleanly into Stage 4's CPU shard schema AND check CPU physical ceilings (`4-bit s17` vs `16-bit canonical`) before launching GPU!
research/stage4_math_and_design/discussions/M4/prompt_r2.md:23:1. Lock the **exact `PhaseContext` + `RefPool` + `Dual 4-bit s17 / 16-bit canonical` CPU Shard Contract** for Pluto (`7,096` frames) and CTTC (`9` captures), explicitly checking causal filtration non-double-counting and receive-gauge vs TX-relabel invariance!
research/stage4_math_and_design/discussions/M4/prompt_r2.md:24:2. Reconcile the **Modal A100-80GB Tier-2 Pilot-Gated Schedule (`<= $9.50` cap, `$0.00` net guard)** combining M1's 4-arm ablation (`Zero`, `Naive-Additive`, `Equivariant-Host-Pure`, `Equivariant-Physical+Residual`), M6's dev pilot futility gate BEFORE full continuation, and `f708_snr10` held-out session transfer! (Explain precisely why M4+M6 pricing rates differed slightly if CPU/RAM sizing differed: `2c/12 GiB` vs `2c/16 GiB`, and lock one exact Modal pricing table!)
research/stage4_math_and_design/discussions/M4/prompt_r2.md:25:3. Extend `verify_m4.py` concisely to test: (i) the unified **`4-bit s17` vs `16-bit canonical` CPU ceiling gap** under Pluto-like fading (`10 dB` dynamic range) & CTTC-like SNR (`-3, +10, +20 dB`) + `0.26` spur; (ii) exact receive-gauge transformation law & non-double-counting causal ledger audit on simulated Pluto/CTTC splits; and (iii) exact reconciled Modal ledger arithmetic. Run via `uv run` and report in `reply_r2.md`.
research/stage4_math_and_design/M4/THEOREM_M4.md:3:Round 1. All logarithms and information quantities use nats. The proposed primary input is a causal, gain-normalized scalar lifted into the frozen `s17` three-tap space at its **physical SNR**. The main real-data contrast is the effect of releasing one or two genuine pilot observations onto a cache whose first two window codes are already fixed.
research/stage4_math_and_design/M4/THEOREM_M4.md:5:The research plan calls for a paired prediction and reveal on actual recordings. Stage 3D defines Q1 as transmitted relative quarter-turn, Q2 as transmitted symbol 1, and Q3 as the received sector of an unrevealed retransmission. The Stage 3D training support reached `|ω(h+ℓ)| = 1.25 rad`. Its physical reader benefited from a second valid pilot while the frozen Qwen connector did not show the same improvement. Freeze the Stage 3D Qwen revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, its tokenization and host fingerprint, and `s17` SHA-256 `e67d9fbafb90791ab19549de7789798d818286d7f52d7ac566cc39df71e78356`. Fit the four connectors with matched budgets and freeze them before scoring. [Research plan](../../../docs/LONG_TERM_RESEARCH_PLAN.md) · [Stage 3D protocol lock](../../stage3d_qwen_signed_phase/PROTOCOL_LOCK.json) · [Stage 3D report](../../stage3d_qwen_signed_phase/REPORT.md)
research/stage4_math_and_design/M4/THEOREM_M4.md:11:Write \(e=\mathbf1_3/\sqrt3\), \(V_0=10^{-0.2}=0.63095734448\), \(S\in\{1,i,-1,-i\}\), and use proper complex noise with \(\mathbb E WW^*=K\) and real per-coordinate variance \(V_0/2\) under the nominal law. `s17` was trained with \(Y_0=eS+W_0,\ W_0\sim\mathcal{CN}_3(0,V_0I)\). The whole window has unit signal energy. Its matched statistic is \(T_0=e^*Y_0\). Each window produces 16 codes, four quadrant turns times four local codes. The frozen lookup table differs from the MLP softmax used to choose the local code. [Frozen encoder](../../matched_cache/learned.py) · [Training waveform law](../../fourphase_paired_questions/paired.py) · [Stage 3 matched sum](../../stage3_ministral_cache/schema.py)
research/stage4_math_and_design/M4/THEOREM_M4.md:17:| **R3**, whitened scalar lift | Let \(B\) have two orthonormal columns orthogonal to \(e\). Set \(Y=eT+B\eta\) with an independent \(\eta\sim\mathcal{CN}_2(0,V_0I)\). This preserves \(T\), and \(D(P_{S,Y}\|P_{0,S,Y})=D(P_{S,T}\|P_{0,S,T})\) by factoring the common independent ancillary law. With nominal scalar mean and variance the synthetic three-tap law is exactly recovered. The ancillary draws carry zero radio evidence and are preassigned in a frozen stream shared by arms. Rotate the realized \(Y\) and all receiver-coordinate evidence together in equivariance replays. |
research/stage4_math_and_design/M4/THEOREM_M4.md:19:For R1, \(T=q^*K^{-1}(R-d)/(\widehat a q^*K^{-1}q)\) is the noise-whitened matched estimate. This follows by expanding the Gaussian likelihood: the only class-dependent observation term is \(2\operatorname{Re}(a^*\bar S q^*K^{-1}(R-d))\). The true variance of \(T\) is \((\widehat a^2 q^*K^{-1}q)^{-1}\); a fixed unitary on the whitened taps can align the steering direction with \(e\). Replacing \(q\) by \(\widehat q\) leaves a component \(\mu_\perp(S)\) orthogonal to the retained direction. With exactly whitened unit covariance and known Gaussian steering,
research/stage4_math_and_design/M4/THEOREM_M4.md:25:The bound follows by comparing the conditional off-axis distribution to independent standard noise, using the KL chain rule. Its upper term vanishes for correct steering. Pulse-shape error, signal-dependent ISI, and improper noise require a fuller law. One may whiten an augmented real vector when the pseudo-covariance is nonzero, using the real Gaussian formula with its factor of \(1/2\).
research/stage4_math_and_design/M4/THEOREM_M4.md:27:The optional forced-calibration control is \(T'=\alpha T+\sqrt{V_0-\alpha^2v}\,Z\), where \(Z\sim\mathcal{CN}(0,1)\), \(\alpha=\min(1,\sqrt{V_0/v})\), and \(v\) is a causal estimate of post-AGC thermal variance. With correct \(v\), this matches the thermal variance, and its conditional Gaussian KL becomes \(|\alpha\mu_S-S|^2/V_0\). When \(v>V_0\), attenuation alone is invertible at continuous input and shrinks the signal mean. When \(v<V_0\), adding noise degrades information. Real noise with random scale, coherent sidebands or ISI retains those effects through this operation. Use the SNR-preserving R3 in the primary comparison, with causal variance and gain-uncertainty summaries available equally to all matched-information decision rules. Use R1, R2, and forced calibration on dev or in a named CPU diagnostic.
research/stage4_math_and_design/M4/THEOREM_M4.md:38:For a scalar in canonical constellation coordinates define \(g=\widehat a/a=1+\delta>0\), \(a=|h|\), \(L=h_{12}+\ell\) in symbols starting at the chosen phase reference, \(\vartheta=\epsilon_\omega L\), \(b=\Delta d/a\), and \(\psi_L=\psi_0+2\pi\Delta f L/f_{\rm sym}\). Include a residual relative complex spur \(r\) as a multiplicative sideband and a fixed conditional neighboring-symbol ISI term \(j_s\) in gain-normalized units. Its known-phase, known-variance law has
research/stage4_math_and_design/M4/THEOREM_M4.md:50:This is conditional on \(U\). A term may be stochastic or correlated with others. Keep the coherent cross terms, or use the triangle inequality to upper bound the squared numerator by \((|\delta|+2|\sin(\vartheta/2)|+r+|b|+|j_s|)^2\). With uniform \(s\), constant \(b\), and \(j_s=0\), the exact average squared shift is \(|e^{i\vartheta}-g+re^{i\psi_L}|^2+|b|^2\). At gain and variance match, isolated \(b=0.3\) costs `0.142640` nats, isolated \(r=0.26\) costs `0.107139`, and isolated CFO costs \(4\sin^2(\epsilon_\omega L/2)/V_0\). At `0.020 rad/symbol`, lags 4, 25 and 126 cost `0.010138`, `0.388037` and `5.746671` nats. A centered independent Gaussian residual CFO with standard deviation \(\sigma_\omega\) gives average phase energy \(2[1-\exp(-L^2\sigma_\omega^2/2)]\). This follows from the characteristic function of the residual. It bounds KL to the fixed phase likelihood after dividing by \(V_0\). The bound does not grow unboundedly with lag. A coherent spur of 0.26 amplitude has power ratio 0.0676 and phase beat about `0.018887 rad/symbol` at a 1503 Hz offset and 500 ksym/s. Represent its phase jointly across windows. The measured sideband ratio alone does not distinguish a multiplicative sideband from an additive tone. For an additive tone, move \(re^{i\psi_L}\) into \(b\) within (2).
research/stage4_math_and_design/M4/THEOREM_M4.md:52:Pluto normalized residual DC can be much larger at low gain for the same absolute subtraction error. For \(\widehat a=a(1+\delta)\), the DC-only term is \(|\Delta d|^2/(\widehat a^2 V_0)\). Its AC-coupling component can depend on the transmitted sequence, which may create conditional mean and covariance terms together. For a proper Gaussian AR(1) approximation to the **total** tap residual with normalized covariance \(K=\lambda V_0 R_\rho\) and \((R_\rho)_{ij}=\rho^{i-j}\) for \(i\ge j\) and Hermitian symmetry, Eq. (1) gives covariance cost
research/stage4_math_and_design/M4/THEOREM_M4.md:72:If the comparator is the frozen `s17` table \(q_{17}\), the exact excess vs full observation Bayes is \(I_P(S;Y\mid Z)+\mathbb E_Z D(P_{S\mid Z}\|q_{17}(\cdot\mid Z))\), with \(Z=\mathrm{s17}(Y)\). The second term contains train-domain calibration and approximation error. Eq. (5) cannot bound it by zero when the input law matches training. For a positive frozen table, its absolute expected NLL change between two laws is at most \(-\log(\min q_{17})\,t\), where \(t\) is total variation after the common mapping and \(t\le\min(1,\sqrt{D/2})\) for a finite joint KL bound \(D\). MAP SER change for a fixed rule is at most \(t\). If the priors on \(S\) coincide, \(|I_P(S;Y)-I_Q(S;Y)|\le\min(\log4, h_2(t)+t\log4)\). Decompose each law into its common mass and its residual of mass \(t\); the conditional entropy of the binary mixture indicator is at most \(h_2(t)\). Apply the same argument to codes by data processing. An unbounded bound in (2) or (4) is compatible with finite code MI and bounded table NLL.
research/stage4_math_and_design/M4/THEOREM_M4.md:89:In the corresponding whitened real coordinates this is a Student distribution with \(2\nu\) degrees of freedom. With \(A=\Sigma_0^{-1}\Sigma\) and \(\nu>1\) the centered KL to the proper Gaussian \(\mathcal{CN}(0,\Sigma_0)\) is exactly
research/stage4_math_and_design/M4/THEOREM_M4.md:98:Rayleigh denominator with \(\mathbb E G^2=1\) is \(\nu=1\). In one complex dimension \(P(|X|^2/\Sigma>q)=(1+q)^{-1}\), so \(\mathbb E|X|^2=\infty\), and forward KL to any nondegenerate Gaussian is infinite. The normalized Student entropy is finite here. More generally an infinite second moment gives infinite KL to a Gaussian by testing bounded truncations of a sufficiently small positive multiple of \(\|X\|^2\) in the variational KL lower bound. If the amplitude estimator itself has a folded Student law, substitute its density \(2f_{t_d}(g/b)/b\) for \(g>0\) in (6), where \(f_{t_d}\) is the standard real Student density and \(b\) its scale. That exact integral is a different scale mixture, with infinite second moment since the folded density is positive at zero. A noisy complex pilot LS amplitude \(|h+E|/a\) with nondegenerate proper Gaussian \(E\) is conditionally Rician. Its radial density has positive linear coefficient at zero, so \(\mathbb E G^{-2}=\infty\) even for nonzero \(h\). If `a` itself is Rayleigh but estimated exactly and both amplitude and noise scale with it, \(G=1\) and the heavy tail argument does not apply. If a new thermal sample and the scale estimator share noise, the exact law uses \(W\mid G\) in (6) and cannot be inferred from the gain histogram alone.
research/stage4_math_and_design/M4/THEOREM_M4.md:100:AGC usually divides the signal too. For \(T=(\mu_s+W)/G\), the exact class-conditional density replaces \(G^2x^*\Sigma^{-1}x\) in (6) with \((Gx-\mu_s)^*\Sigma^{-1}(Gx-\mu_s)\). It is a noncentral scale mixture and generally has no centered Student formula. Absolute Rayleigh fading and fixed receiver noise can also produce a heavy-tailed unconditional noise term even with exact gain estimation, since averaging \(W/a\) over \(a\) invokes the Rayleigh case. A causal guard \(\widehat a\leftarrow\max(\widehat a,a_{\min})\), with floor fixed from training/calibration and carried in the uncertainty flags, makes \(\mathbb E G^{-2}\) finite conditional on \(a\). Fit variance floors, clip estimator outliers as registered on dev, and model or report flags. When estimating a noise scale with \(\nu\) complex residual degrees of freedom independent of the new observation and a correctly specified proper Gaussian linear fit, the normalized residual power estimate has Gamma shape \(\nu\). In a rank-\(p\) complex fit to \(n\) such pilots, \(\nu=n-p\) follows from orthogonal residual projection. Estimated CFO, branch selection, regularization, symbol decisions, correlated pilots, and unknown DC break this pivot unless accounted for. Use a robust likelihood or (4) with measured nuisance uncertainty in those cases.
research/stage4_math_and_design/M4/THEOREM_M4.md:110:For `s17`, along a fixed nonzero canonical ray \(x=cu\), \(c>0\), the sum sector is scale invariant. At low gain \(\tanh(cuW_1+b_1)\to\tanh b_1\), so the local code becomes constant if the winning cross-entropy score has a positive margin. The hidden deviation is at most \(c\|uW_1\|\). Converting it through \(W_2\) and the \(1/2\)-Lipschitz softmax, the bound on pairwise score changes is \((c/2)\|uW_1\|\|W_2\|_{2\to2}\|\log C_i-\log C_j\|\) for centers \(C_i,C_j\). A strict margin greater than this perturbation proves stability. For high gain, neuron \(j\) with \(d_j=(uW_1)_j\ne0\) approaches \(\operatorname{sgn}(d_j)\) with error at most \(2e^{-2(c|d_j|-|b_{1j}|)}\) once \(c|d_j|>|b_{1j}|\); neurons with zero projection stay at \(\tanh b_{1j}\). Apply the same score-margin test to the limiting logits and center choices. This proves radial saturation on each interior ray with strict margin. The four turns can remain informative. At exact all-zero input the encoder chooses its defined tie behavior. `verify_m4.py` finds `[1,5,9,13]` in the low-gain limit and `[0,4,8,12]` on its high-gain ray. Ties, near-orthogonal directions, and random scales near zero preclude a global AGC threshold.
research/stage4_math_and_design/M4/THEOREM_M4.md:114:### 2.1 Candidate rows and time order
research/stage4_math_and_design/M4/THEOREM_M4.md:116:Use time in symbols with an explicit sample-clock conversion and a common fixed transmitter mapping \(e^{-i\pi/4}s_{\rm hardware}\) into \(C_4\). Use dataset-specific Gray lookup in the scorer. For Pluto divide the nominal `0.707` axis levels by their fixed constellation magnitude. Use a corresponding fixed receiver coordinate transform, subtract only the causally available DC estimate, and retain the residual channel angle. Each data window contains observations of **one transmitted symbol**. Record the two codes and supporting provenance, release a reference only after the first forecast is committed, and never revise either code. A Q3 forecast stores both the cache-stage and reveal-stage four-vectors before the first sample of the target stencil is eligible. Collector latency is the maximum raw time in each interpolator stencil, with all clocks and cutoffs materialized.
research/stage4_math_and_design/M4/THEOREM_M4.md:120:| Pluto 850 MHz | One row per frame. Copy 1, \(k_1=30+(f\bmod67)\); \(k_2=k_1+h_{12}\) with \(h_{12}=1\) or 8 using `(f + floor(f/16)) mod 2`. | Two acquisitions at logical indices `15` and `18` of copy 1. Initial state uses sync `0:13` and split-local past, with those two pilot reads excluded from all pre-reveal quantities. | Same payload slot \(k_1\) in the immediately following complete copy, \(L=126\), \(\ell=126-h_{12}\). Score 2,240 candidates with a common lock-failure policy. A copy is a real retransmission; no target-label matching or synthetic replacement. |
research/stage4_math_and_design/M4/THEOREM_M4.md:121:| CTTC 433/708/2450 MHz | One row per packet with ordinal \(j\) and loop \(b\). \(k_1=136+((17j+5b)\bmod48)\); \(k_2=k_1+h_{12}\), using 1 for even \(j+b\), 8 otherwise. | `126,127` of the same packet, both earlier than \(k_1\). Initial state uses `0:6` sync and authorized previous data, excluding both release samples. Estimate their joint covariance. `254,255` remain unavailable until they have physically arrived. | On each capture use disjoint loop pairs `(0,1),(2,3),(4,5),(6,7)` at the same packet source start and same \(k_1\) selected in the earlier loop. \(L=500000\), \(\ell=500000-h_{12}\). Q1/Q2 cover 1,440 packets. Q3 has 720 retransmission candidates and 36 loop-pair groups. |
research/stage4_math_and_design/M4/THEOREM_M4.md:125:The primary samples retain actual fading, residual DC/ISI, noise, clock offset, and any changes present at registered times. A naturally occurring jump is a stratum, with changes flagged by a predeclared causal split-local pilot innovation criterion and a retrospective scorer-only morphology check. A CPU-only cross-frame Q1 diagnostic pairs `(f−1,f)` at `f mod 20 = 0` within each test block, using the earlier frame's \(k_1\) and current frame's \(k_1\) from the table. Its matched control pairs the preceding two interior frames `(f−2,f−1)` in that same block. Use only available sync and authorized pilots and a gap-class prior when the time between Pluto columns is unavailable. This samples recorded batch boundaries by schedule without requiring or asserting a jump at each boundary. Separate source domains and nominal session strata in analysis, keeping capture IDs out of model text/features. Hold `f708_snr10` out of fitted connector weights, temperatures and shared prior tuning to expose one capture transfer; allow its own causal sync and released pilots. Pluto is a blocked same-campaign test. CTTC has one public cyclic TX sequence; input-source disjointness and capture dependence are distinct issues. Use block-paired NLL/SER differences on 16 Pluto groups and 72 CTTC session-loops, group adjacent repeat pairs for CTTC Q3, and report a capture-level sensitivity analysis and source-ordinal sensitivity because loops/captures reuse source positions. Report all registered rows and lock coverage, with the same uniform predictive fallback on failed rows and a secondary locked-only analysis. Avoid treating all packets as independent. Select the Q2 `two−one` proper-loss contrast and large-angle synthetic Q3 coupled-vs-additive contrast before test, with paired intervals and the multiplicity rule frozen with M2.
research/stage4_math_and_design/M4/THEOREM_M4.md:135:**Timing and CFO gates.** The legacy Pluto `fast_blind_frontend` correlates whole captures at lag 2520 and scans two copies for header timing. Even randomized payload and unlabelled probes can enter those sums. CTTC's `anchor_capture` uses whole-source lock candidates at several times and long known-source CFO fits, and extraction includes `offline_cfo_hz`, `h_offline_ref` and powers derived using all TX symbols. None of these are admissible causal inputs. For Pluto use an independently registered trigger or bounded timing prior established from earlier authorized sync, verify the **entire candidate search stencil** fits in the allowed sync envelope, then track only with a prefix of the intended copy. For CTTC reserve a disjoint in-cycle commissioning segment, for example source `[496256,498304)` plus sufficient guard for RRC/padding. Use its whitelisted bits and receiver samples strictly before the first use at a given cutoff to resolve EVEN/ODD and coarse timing/SFO. Use the public clock rates as fixed constants. Require a timing-uncertainty certificate for subsequent local interpolation and renewal, and exclude same-split or cross-session reference observations that arrive later than the prediction. The legacy CTTC role order places the same-loop reference pool after its target range, so only already acquired same-split entries may supply history. Reset state at split boundaries and document commissioning as a distinct authorized acquisition with a disjoint source footprint. If safe synchronization cannot be certified, emit a common fallback and report coverage. Strict raw masking and the full usable row counts are conditional on this gate. A Pluto window with an unconstrained initial timing offset cannot certify strict raw masking from those raw samples alone: for every tentative input position some admissible circular offset places a held-out slot there. An offline alignment-assisted diagnostic may be reported with its timing provenance visible.
research/stage4_math_and_design/M4/THEOREM_M4.md:137:Use causal sync and then revealed pilots for residual CFO. For synchronized observations \(y_p\) and authorized \(s_p\), fit \((H,d,\omega)\) using the pilot-only profile loss
research/stage4_math_and_design/M4/THEOREM_M4.md:143:Within a frozen frequency prior, keep multiple plausible modes and their masses or a causal filter posterior for frequency, phase and possible jump. When rank or timing confidence fails, set the common missingness state and use a declared prior. Without a held-in true rate, \(\epsilon_\omega=\omega_{\rm true}-\widehat\omega\) is latent. A receiver can estimate, propagate and calibrate uncertainty and innovations. For known amplitude and phase design, the efficient information for \(\omega\) with unknown complex gain in proper white noise is \(2|H|^2\sum_p |s_p|^2(t_p-\bar t)^2/v\), with energy-weighted mean time. With unknown DC, project \(iHt_p s_p e^{i\omega t_p}\) onto the whitened orthogonal complement of both nuisance columns and take twice its squared norm. This follows from the Schur complement of the Gaussian Fisher matrix and gives a local unbiased CRLB, not a bound guaranteeing tracking of phase jumps or unwrapping. Irregular gaps and 1-second CTTC prediction require propagating the circular phase law across the actual lag. Never fit from Q3, `HOLDOUT`, full waveform repetitions, payload decisions or injection flags. Freeze real-split provenance and compare hashes after replacing future/probe IQ and all scorer-only columns with sentinels. Outputs, caches and feature hashes must agree. Enumerate the transitive support ledger as well as this mutation test. The CPU replay checks archived geometry and designed local support. It did not run this acquisition-level gate.
research/stage4_math_and_design/M4/THEOREM_M4.md:150:| **Direct `s17` table** | Report raw frozen table for single-window/ Q2-none, and the matched-information likelihood reader using code emissions and the same causal reference statistics for Q1/Q2/Q3. For an exact reader integrate each candidate likelihood over the frozen quantizer cell (by locked simulation or quadrature and an error audit). The frozen table is a training posterior, not an emission likelihood on hardware. A code-center approximation can be reported as such. |
research/stage4_math_and_design/M4/THEOREM_M4.md:153:| **Stage4-Equivariant-Coupled** | Canonicalize the received gauge from the turn in the first valid cached code. Rotate both codes and receiver-coordinate pilot evidence by that gauge, propagate the signed phase law with explicit lag, then map the answer vector back with its registered group action. Under a common receive-chain quarter-turn, Q1 and Q2 TX labels stay fixed and Q3 receiver sectors permute. If testing a TX relabeling, known pilot labels and Q2 labels transform together in a separately specified action. A single canonical host score suffices. Register validity before projection and audit an exact-zero or missing-pivot policy. |
research/stage4_math_and_design/M4/THEOREM_M4.md:154:| **Stage4+RefSufficientPool** | Replace ordered raw-reference tokens by a set likelihood summary on the same pilot observations, with release times and known complex TX values. Deduplicate exact acquisition copies before pooling. With joint proper Gaussian \(y=A_mH+n,\ n\sim\mathcal{CN}(0,K_m)\), sufficient for \(H\) at fixed \(m\) are \(A_m^*K_m^{-1}y\) and \(A_m^*K_m^{-1}A_m\). For comparing candidate \(m\) also retain \(y^*K_m^{-1}y\), normalizers (e.g. \(\log\det K_m\)), and weights/times. Complete the square to see all remaining terms are constant in \(H\). For overlapping/colored blocks use joint covariance or unique innovations. For time-varying states use the corresponding transition and filter state or a bank of likelihood factors. A single summed phasor has no general sufficiency under unknown CFO, shared noise or jumps. |
research/stage4_math_and_design/M4/THEOREM_M4.md:174:- Use BF16 Qwen text-forward, fenced caches including Qwen3.5's attention and recurrent state, batched four candidate likelihoods, and a fixed tokenizer/chat template. Keep exact question and candidate sequence scoring and one host forward per canonicalized row. During connector training, gradients must flow through the frozen host to connector inputs. Share only text-only prelude states when valid. Never reuse post-evidence states across references or arms. Compare KV-cached and uncached logits/gradients on the longest dev template before accepting that scorer. Run initial train smoke with effective batch 32, microbatch 16 with two accumulated steps or microbatch 8 with four if necessary. Match prefix lengths and number of evidence positions to Stage 3D's microbenchmark and resource gate. No extra runtime factor is reserved for online multi-pass group averaging.
research/stage4_math_and_design/M4/THEOREM_M4.md:175:- Stage 3D measured `1.9757 s / optimizer step` and `23.88 ms / logical eval row` on one A100-80GB. Train and eval peaks were `52,972.5 MB` at microbatch 16 and `45,409.9 MB` at eval batch 64. The six 1,280-step runs took `2,884.0–3,217.8 s` including additional work. BF16 weights alone are about 8 GB at 4B parameters. Train activation and batched candidate workspace amplify this. The baseline plan uses A100-80GB for both. Accept the pilot run only if peak allocated/reserved fits a predeclared 65 GiB A100 threshold and sustained times meet `≤2.50 s/step`, `≤0.030 s/row` inclusive of in-batch data feeding. Rerun timing after any microbatch change. A100-40GB, L4 and A10/A10G require a separate shape and kernel check with smaller batches. The measured Stage 3D peaks exceed their VRAM. For a substitution, require equivalent logits, a ≤20 GiB peak on 24 GiB cards, and throughput below `0.030*(0.00074684/(GPU rate + 0.00005284))` seconds per row to match A100 costs, as well as a preapproved wall limit. This is about 81.5 ms on L4 and 62.4 ms on A10, with 2 cores and 12 GiB host RAM. Keep the confirmed A100 path if the smaller card would require quantizing the frozen host or invalidating kernels. [Stage 3D report](../../stage3d_qwen_signed_phase/REPORT.md) · [Modal GPU SKUs](https://modal.com/docs/guide/gpu)
research/stage4_math_and_design/M4/THEOREM_M4.md:178:- Natural real work per arm is `3,680×(Q1 + 3×Q2) + 2,960×(2×Q3) = 20,640` rows, hence `82,560` across arms. Stress adds `(7×140 + 1,080)×4 = 8,240` rows. Dev adds `16,000`. A logical row already includes scoring all four candidate answers. Registered CPU physical arms score the identical rows but consume zero GPU time. Scope and statistical interpretation exclude prompt-level claims about rare jumps or undisclosed source domains that have no registered observations.
research/stage4_math_and_design/M4/THEOREM_M4.md:210:| Surrogate condition | R1 SER | R2 SER | R3 physical SER | R3 calibrated SER | R3 physical vs calibrated NLL |
research/stage4_math_and_design/M4/THEOREM_M4.md:214:| +20 dB, clean gain/DC/sideband | 0.00% | 0.00% | 0.00% | 19.89% | 0.060 vs 0.543 |
research/stage4_math_and_design/M4/THEOREM_M4.md:217:The verifier checks directly that the frozen table chooses the turn's class for all four local codes, which explains SER equality of the scalar R2 and R3 arms. The matched sum and hence the turn coincide under the lift. The local code and calibration can still change. Attenuation at low SNR can improve frozen NLL without improving SER. The high-SNR loss from variance locking occurs when injected noise masks real high-SNR evidence. The observed saturation codes and conditional nuisance curves characterize the frozen weights under the stated draws. Generalization, real timing coverage and throughput remain empirical gates. See the full records for sample counts, scores, transition matrices, budget, manifests and geometry. [Verifier](verify_m4.py) · [Results](results_m4.json)
```

### Event #7: `item.completed` — `command_execution` (`item_5`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n -i 'RefSufficientPool|K_ref|timestamp|acquisition|CFO|diffusion|correl|interface' research/stage4_math_and_design/M3/THEOREM_M3.md research/stage4_math_and_design/discussions/M3/prompt_r2.md research/stage4_math_and_design/M3/verify_m3.py"
```

**Aggregated Output:**

```text
research/stage4_math_and_design/M3/verify_m3.py:55:    acquisition_id: int
research/stage4_math_and_design/M3/verify_m3.py:73:class RefSufficientPool(nn.Module):
research/stage4_math_and_design/M3/verify_m3.py:81:        for ref in sorted(refs, key=lambda x: (x.acquisition_id, x.t_p)):
research/stage4_math_and_design/M3/verify_m3.py:83:                raise ValueError("Expected a finite nonnegative pilot timestamp")
research/stage4_math_and_design/M3/verify_m3.py:84:            prev = by_id.get(ref.acquisition_id)
research/stage4_math_and_design/M3/verify_m3.py:86:                by_id[ref.acquisition_id] = ref
research/stage4_math_and_design/M3/verify_m3.py:92:                raise ValueError(f"Conflicting metadata for acquisition_id={ref.acquisition_id}")
research/stage4_math_and_design/M3/verify_m3.py:94:                raise ValueError(f"Conflicting samples for acquisition_id={ref.acquisition_id}")
research/stage4_math_and_design/M3/verify_m3.py:97:        for acquisition_id in sorted(by_id):
research/stage4_math_and_design/M3/verify_m3.py:98:            ref = by_id[acquisition_id]
research/stage4_math_and_design/M3/verify_m3.py:191:        arrays = [(r, source_samples[r.acquisition_id] if r.shares_channel else mi) for r in refs]
research/stage4_math_and_design/M3/verify_m3.py:193:            (r.acquisition_id, r.shares_channel, vals[i]) for r, vals in arrays
research/stage4_math_and_design/M3/verify_m3.py:239:    train, _ = data.generate_paired_cfo_blocks(data.NS_CFO_TRAIN, 0, blocks, H, 2, TP, (OMEGA,),
research/stage4_math_and_design/M3/verify_m3.py:240:                                               split_prefix="m3_cfo_train")
research/stage4_math_and_design/M3/verify_m3.py:246:    fit = {"namespace": data.NS_CFO_TRAIN, "blind_namespace": data.NS_BLIND_Q2_C4,
research/stage4_math_and_design/M3/verify_m3.py:294:    many = tuple(replace(base, acquisition_id=i, t_p=float(i), variance=(i+1)/RHO)
research/stage4_math_and_design/M3/verify_m3.py:306:    hetero = replace(base, acquisition_id=1, variance=3/RHO, matched_amplitude=0.75)
research/stage4_math_and_design/M3/verify_m3.py:318:            raise AssertionError("Acquisition conflict accepted")
research/stage4_math_and_design/M3/verify_m3.py:417:    pool = RefSufficientPool()
research/stage4_math_and_design/M3/verify_m3.py:430:        # It complements the noisy observed-label average, especially at low diffusion.
research/stage4_math_and_design/M3/verify_m3.py:456:    # One independent low-diffusion follow-up at fixed N after the first noisy estimate.
research/stage4_math_and_design/M3/verify_m3.py:485:    print(f"Low-diffusion follow-up: observed naive gap {followup['naive_minus_exact']['nll']['delta']:+.8f}, "
research/stage4_math_and_design/M3/verify_m3.py:487:    out["independent_low_diffusion_followup"] = followup
research/stage4_math_and_design/M3/verify_m3.py:520:    surrogate_scores = check_pool(RefSufficientPool(), surrogate, blind, pinned)
research/stage4_math_and_design/M3/verify_m3.py:523:    print("Passed paired two-timestamp Brownian checks.", flush=True)
research/stage4_math_and_design/M3/verify_m3.py:529:        "stage3d": archive, "surrogate_fit": fit, "surrogate": surrogate_scores, "multi_timestamp": multi,
research/stage4_math_and_design/discussions/M3/prompt_r2.md:1:# Track M3 (Round 2): Correlated Multi-Clock Pilots, Non-Double-Counting Innovations & Stage 6 Active Mixture Interface
research/stage4_math_and_design/discussions/M3/prompt_r2.md:5:## 1. Cross-Track Demands on `RefSufficientPool` (From M1, M2, M4, M5, M6)
research/stage4_math_and_design/discussions/M3/prompt_r2.md:6:1. **From M4 (NO DOUBLE-COUNTING & Correlated Hardware Noise/State)**:
research/stage4_math_and_design/discussions/M3/prompt_r2.md:8:   - Furthermore, real hardware pilots at nearby lags (e.g., CTTC `126, 127` adjacent symbols through `256`-tap RRC pulse / residual ISI / shared CFO error $\epsilon_\omega$) can share **estimation uncertainty or noise covariance** $K_{\text{ref}}$ AND may use known QPSK pilot roots $d_m \neq 1$!
research/stage4_math_and_design/discussions/M3/prompt_r2.md:9:   - **Crucial Warning from M4**: `RefSufficientPool` MUST NOT double-count observations already conditioned inside M4's $\mathcal{F}^0$ causal state! When adding `(R_1, R_2)` onto $\mathcal{F}^0$, `RefSufficientPool` must emit the **conditional likelihood / innovation factor** given the prior state (and explicitly track which `acquisition_id`s already contributed to $\mathcal{F}^0$!).
research/stage4_math_and_design/discussions/M3/prompt_r2.md:10:2. **From M1 (Unified Likelihood Interface + Shared CFO Uncertainty)**:
research/stage4_math_and_design/discussions/M3/prompt_r2.md:11:   - M1 asked: can `RefSufficientPool` accept per-pilot timestamps (including **$t_p > t_3$**!), known pilot symbols $d_m$, shared CFO uncertainty $(\hat\omega, v_\omega)$ (remember CFO rate error scales phase variance as $v_\omega (t_j - t_i)^2$, creating cross-clock quadratic coupling beyond Brownian diffusion $\sigma_\phi^2 |t_j - t_i|$!), AND return joint factors `canonical_q2[4]`, `canonical_q3[4]`, and optional `joint[4,4]` relative to receiver frame $a_1$ (while keeping exact zero-influence distractors and byte-identical fixed-position prompt tokens)?
research/stage4_math_and_design/discussions/M3/prompt_r2.md:13:   - Before revealing candidate window $a$, M5 computes $f_{ha}(x) = \int p(Y_a = x \mid u, h, z) dP(u \mid z, h)$ and $\text{EIG}(a \mid z) = I(H; Y_a \mid z)$, THEN reveals ONE sealed acquisition $(id_a, t_a, y_a)$ into `RefSufficientPool`! Notice if `RefSufficientPool` represents each candidate hypothesis / phase-node state as clean `(phase_grid, log_weights_by_turn[4, N_nodes], clock_meta)`, then BOTH M5's pre-reveal $\text{EIG}$ integral AND post-reveal Bayes update share ONE exact mathematical engine!
research/stage4_math_and_design/discussions/M3/prompt_r2.md:21:1. **Extend Theorem 2 (Joint Brownian + Latent CFO Uncertainty + Correlated Pilot Noise + Anti-Double-Counting Theorem)**:
research/stage4_math_and_design/discussions/M3/prompt_r2.md:23:   - Derive the augmented Gaussian covariance $(B_{t_j} + \tilde\omega t_j)$ combining Brownian increments $\sigma_\phi^2 \min(t_j, t_k)$ AND shared linear CFO prior variance $v_\omega t_j t_k$ (note rank-1 rank addition cleanly solved by Sherman-Morrison or appending 1 Gaussian-Hermite quadrature node over $\tilde\omega$!).
research/stage4_math_and_design/discussions/M3/prompt_r2.md:24:2. **Freeze the Unified Multi-Stage `RefSufficientPool` Schema** serving **M1 (`EquivariantEvidenceConnector`) + M4 (Pluto `15,18`, CTTC `126,127`, duplicate/missing provenance) + M5 (Stage 6 pre-reveal mixture & EIG)** simultaneously!
research/stage4_math_and_design/discussions/M3/prompt_r2.md:25:3. Add deterministic CPU unit/property verifications inside `verify_m3.py` (keep total runtime concise `< 60s` or reuse cached fast quadrature; verify: (i) zero double-counting invariant against prior state, (ii) non-trivial shared CFO + Brownian multi-clock gain vs naive sum, (iii) byte-exact `uint8`/`float32` representation identity across shuffled/duplicate/distractor/padded `RefPool` inputs, and (iv) Stage 6 pre-reveal predictive marginal + post-reveal Bayes update consistency!). Run via `uv run` and report in `reply_r2.md`.
research/stage4_math_and_design/M3/THEOREM_M3.md:5:The C4 generator sets the continuous drift to zero at the first window. It draws independent Gaussian increments with mean \(\omega\,dt\) and variance \(\sigma_\phi^2dt\) on the grid \(\{0,h,h+\ell,t_p\}\). Its two valid pilots share the same physical phase and have independent complex AWGN. At \(h=\ell=2,\ t_p=4\), the pilot clock equals the withheld-window clock. Each generator call takes one scalar \(t_p\). The schema can carry a different timestamp on each reference, but the seven C4 arms never exercise this case. The released C4 reader accepts only an ID, a channel Boolean and a matched sum. Its fitted encoder emissions and quadrature are tied to \(t_p=4,\ \omega=+0.35,\ \sigma_\phi=0.10,\ \mathrm{SNR}=2\,\mathrm{dB}\). [Generator and arms](../../stage3d_qwen_signed_phase/data_stage3d.py) · [Schema](../../stage3_ministral_cache/schema.py) · [Reader](../../stage3c/A/q2_reader.py) · [Stage 3C derivation](../../stage3c/A/Q2.md)
research/stage4_math_and_design/M3/THEOREM_M3.md:7:A copy has the same acquisition ID and samples. A fresh acquisition has a new ID and independent noise conditional on the shared path. The schema checks duplicate channel flags, timestamps within \(10^{-12}\), and exact sample equality. The weighted pool also checks amplitude and noise variance, which affect the likelihood. Stage 3D's distractor has a separate random quadrant, phase, and noise. Its density cancels when calculating the target posterior. These results condition on public, exogenous acquisition design and correct channel declarations.
research/stage4_math_and_design/M3/THEOREM_M3.md:23:Deduplicate by ID and retain only valid acquisitions. Expanding the density at a common time \(p\) gives
research/stage4_math_and_design/M3/THEOREM_M3.md:56:At \(p=0\) or zero diffusion use the deterministic pilot phase. At \(p=h\) the conditional second-window phase is a point mass. Define \(d\eta_{z_2}(s)\propto dP(\Theta_p=s)\,\mathbb E[m_{z_2}(\Theta_h)\mid\Theta_p=s]\). Then
research/stage4_math_and_design/M3/THEOREM_M3.md:103:**Minimality for the physical likelihood.** A parameter-independent density ratio implies \(\sum_j\Re[(D_j-D'_j)e^{-i(bg+v_j)}]\) is constant. The phase vector has full support, so vary each \(v_j\) independently to force \(D_j=D'_j\). A deterministic zero-time node is resolved by comparing the four \(g\) values. At zero diffusion all residual phases coincide and the derotated sum is sufficient.
research/stage4_math_and_design/M3/THEOREM_M3.md:152:**Position and recurrence.** The prompt renders the ID, timestamp and Boolean as text. Each `<REF>` expands to four learned vectors. The smoke audit records 253 prefix tokens for valid Q2. A second reference adds four embedding vectors and changes the text, moving downstream positions and changing relative distances for attention pairs across the insertion. Distances between two jointly shifted downstream positions stay fixed. DeltaNet consumes additional input and state transitions. The saved blind data lack per-arm prefix traces and hidden states. The training arms already include two fresh pilots, so an unseen length alone is an unsupported explanation. A causal test needs fixed prose, fixed-position payload swaps, and a position-only attention intervention. A separate neutral insertion with a recurrent-state control can test state evolution. [Prompt](../../stage3d_qwen_signed_phase/data_stage3d.py) · [Prefix assembly](../../stage3d_qwen_signed_phase/qwen_connector.py) · [Smoke audit](../../stage3d_qwen_signed_phase/results/smoke_audit.json) · [RoPE](https://arxiv.org/abs/2104.09864) · [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
research/stage4_math_and_design/M3/THEOREM_M3.md:158:## 5. Fixed-slot interface and guarantees
research/stage4_math_and_design/M3/THEOREM_M3.md:160:1. **Validate and canonicalize.** Use \(z_i=4a_i+u_i\). Resolve IDs and check provenance, calibration and exact sample equality before filtering. A production clock should publish one canonical timestamp per physical tick. For schema-tolerance aliases, the prototype sorts timestamps per ID, chooses the minimum, and rejects a spread beyond \(10^{-12}\). Remove independent-channel references before feature evaluation. A conflict-free distractor contributes no activation or gradient. A literal copy of a canonical acquisition ID makes no change. Sort unique IDs before reduction for deterministic sums.
research/stage4_math_and_design/M3/THEOREM_M3.md:167:**Blackwell.** Dropping the new acquisition maps the augmented experiment to the old experiment. There is an explicit garbling even after same-clock pooling. Given the augmented pooled sum \(S_+\) with total precision \(P_+\), the old sum of precision \(P_o\) has conditional law \(\mathcal{CN}((P_o/P_+)S_+,\,P_o(1-P_o/P_+))\), independent of the physical phase. This follows from joint Gaussian receiver noise with covariance \(P_o\) between the two sums. Across clocks, use this garbling within each clock and drop new-only clocks. An exact posterior preserves all decisions about its target, so the Bayes risk of the augmented posterior experiment improves weakly for every loss. The expected log-loss gain is \(I(X_1;T_{\mathrm{new}}\mid Z_1,Z_2,\mathcal R_{\mathrm{old}})\). For the finite-time Brownian Q2 model with informative \(a_{z_1}\) and a fresh finite positive-SNR pilot this is strict. At zero old evidence, the new pilot derivative of the first target moment is nonzero for some \(z_2\), since the unconditional \(e^{iB_{\mathrm{new}}}\) moment is positive. Continuity and positive observation densities give a positive-measure change in the posterior. This applies to the first pilot and subsequent ones at common or distinct clocks. A copy, independent distractor or zero-SNR acquisition gives equality. [Blackwell, *Equivalent Comparisons of Experiments*](https://doi.org/10.1214/aoms/1177729032)
research/stage4_math_and_design/M3/THEOREM_M3.md:181:The script replays the archived physical and host probabilities. A fixed-slot CPU surrogate fits \(a_{z_1}\) on 32,768 distinct `NS_CFO_TRAIN` C4 mothers with quarter-turn tying and a symmetric half-count prior. Its second-window phase emission comes from separate Stage 3C-style importance calibration. The 512 blind labels enter only scoring. The multi-clock experiment uses 16,384 new paired mothers at \(t_p=(1,7)\), \(h=2\), \(\omega=+0.35\) and 2 dB, with common draws across methods and diffusion settings. It compares the full Brownian model, the optimal coarsened-data posterior under the same model, and a coherent common-phase approximation. The first low-diffusion observed-label result was inconclusive. A single independent follow-up at 524,288 mothers resolved that contrast, with the initial result retained. Conditional model NLL in the receipt averages over the full fitted posterior for each observation and complements the independently observed-label NLL. The main experiment uses order 40 per pilot innovation, checks order 64 on the 128 highest-pilot-magnitude cases, and takes the second-window emission from 513 periodic knots with order 20 for the inner bridge. The follow-up uses order 12 per innovation and checks against order 40 on 128 high-magnitude cases. [Script](verify_m3.py) · [Result receipt](results_m3.json) · [Benchmarks and Round 2 questions](reply_r1.md)
```

### Event #8: `item.completed` — `command_execution` (`item_4`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n -i 'paired|multiplicity|gate|holm|bonferroni|fixed.best|LEMMO|3.8|EIG|cache' research/stage4_math_and_design/discussions/M2/prompt_r2.md research/stage4_math_and_design/discussions/M6/prompt_r2.md research/stage4_math_and_design/M6/THEOREM_M6.md research/stage4_math_and_design/M2/THEOREM_M2.md research/stage4_math_and_design/discussions/M1/prompt_r2.md research/stage4_math_and_design/M1/THEOREM_M1.md"
```

**Aggregated Output:**

```text
research/stage4_math_and_design/discussions/M2/prompt_r2.md:1:# Track M2 (Round 2): Reconciling Power vs Modal Free-Credit Scope, Causal Arm Gate & Multi-Claim Hierarchy
research/stage4_math_and_design/discussions/M2/prompt_r2.md:7:   - Your Round 1 analysis proved that across **retraining seeds**, 3 random seeds cap even infinite-block crossed random-seed power at `71.54%` (Acc) / `85.94%` (NLL) under G1 proxy variances (with empirical Monte Carlo ~22–30%), requiring **`>= 10` paired seeds** for population-of-seeds inference!
research/stage4_math_and_design/discussions/M2/prompt_r2.md:10:2. **Causal Isolation Gate for M1's `Host-Pure` vs `Physical-Residual` Arms + M6 DPI**:
research/stage4_math_and_design/discussions/M2/prompt_r2.md:12:   - Lock the exact paired contrasts for M1's 4-arm matrix (`Zero`, `Naive-Additive`, `Host-Pure Equivariant`, `Physical+HostResidual`, plus `Direct S17` & `Full IQ`), including:
research/stage4_math_and_design/discussions/M2/prompt_r2.md:15:     - Pre-registered calibration/temp-fitting scope: separate **dev-locked zero-shot calibration** vs **oracle retrospective sensitivity**, strictly preventing blind recalibration from ever passing a gate!
research/stage4_math_and_design/discussions/M2/prompt_r2.md:18:   - Specify the exact **Holm-Bonferroni / IUT Hierarchy** protecting Family-Wise Error Rate (`alpha = 0.05`) across `(Claim 1: Synthetic Extrapolation + Q2, Claim 2: Pluto, Claim 3: CTTC + held-out f708_snr10)`, handling CTTC's capture-level (`n=9`) vs loop-pair (`n=36`) vs cluster (`n=72`) correlation cleanly (and never claiming `n=72` i.i.d. if source sequence & captures induce session dependence!).
research/stage4_math_and_design/discussions/M2/prompt_r2.md:19:4. **M5's Stage 6 Gate Compatibility**:
research/stage4_math_and_design/discussions/M2/prompt_r2.md:20:   - Include a crisp specification for Stage 6's `max_EIG` vs `{random, nearest, fixed_best_on_dev}` gate (plus `4-bit s17` vs `16-bit canonical` comparison)!
research/stage4_math_and_design/discussions/M2/prompt_r2.md:25:2. Lock the **Exact Stage 4 Formal Preregistered Statistical Gate Table** unifying M1's `Arm 0/1/2` contrasts, M3's `Q2 two-valid > one-valid > none` & exact `0.0` invariance assertions, M4's Pluto/CTTC real-data hierarchy, and M5's `fixed_best_on_dev` Stage-6 gate.
research/stage4_math_and_design/M1/THEOREM_M1.md:17:Here \(B\) is standard Brownian motion, \(G\) is the uniform QPSK channel turn, and \(X_1,X_2\) and the fresh \(X_3\) are independent and uniform. A shared-channel pilot transmits a known root (root 1 in Stage 3D), so \(T_p=i^Ge^{i\phi(t_p)}+\epsilon_p\). Observation noises are independent within a mother. The balanced blocks sample this marginal law, and the two CFO signs share innovations and noises for paired comparison. The posterior below conditions on one mother's observables. [Schema](../../stage3_ministral_cache/schema.py) · [Generator](../../stage3d_qwen_signed_phase/data_stage3d.py)
research/stage4_math_and_design/M1/THEOREM_M1.md:31:**Proof.** The encoder takes the sector of the sum of the three samples, falling back to the first nonzero sample on a zero sum. The half-open convention gives \(\operatorname{sector}(i^rT)=\operatorname{sector}(T)+r\) for nonzero \(T\), including boundary rays. Undoing the selected turn gives the same canonical I/Q features after rotation. The fitted MLP and center assignment therefore return the same inner index. The all-zero window is fixed and cannot carry a nontrivial turn. It has probability zero under the stated noise. A hardware interface can mark it as an erasure. [Encoder](../../matched_cache/learned.py)
research/stage4_math_and_design/M1/THEOREM_M1.md:60:Independent increments yield \(U_n(d_1+d_2)=U_n(d_1)U_n(d_2)\) for constant parameters. Diffusion is an irreversible convolution semigroup for \(\sigma_\phi>0\). To undo a known deterministic drift use the conjugate phasor. Diffusion marginalization continues to use damping.
research/stage4_math_and_design/M1/THEOREM_M1.md:82:**Proof.** Orthogonal projection along the two sector boundaries yields independent normals with the indicated means after standardization. The sign pairs in label order are \(++,-+,--,+-\). Independence gives the four products. Integrate the Brownian angle and use circular observation noise to shift by the source root. The resulting map is a cyclic convolution, diagonalized by the \(C_4\) DFT. Since \(B(\phi+j\pi/2)=R_jB(\phi)\), it gives \(q(z;\theta+j\pi/2,V)=R_jq(z;\theta,V)\) at fixed \(A_z,V,\rho\). \(\square\) [Stage 3D analytical readout](../../stage3_ministral_cache/analytic.py)
research/stage4_math_and_design/M1/THEOREM_M1.md:104:The Stage 3D evaluation angles \(0.80\) and \(1.60\) are close to \(\pi/4\) and \(\pi/2\), respectively. G3 has twice the phase variance of G1 at the same angle; G4 has half the phase variance of G2 at its angle. The verifier also varies only \(\theta\), holding \(V=0.04\), SNR and the W1 code mixture fixed. It computes the code-mass-weighted maximum of the convolved probabilities. This isolates the angular decision effect from sampling and variance differences across cells. The absolute accuracy under source uncertainty follows from the mixture and the measured data.
research/stage4_math_and_design/M1/THEOREM_M1.md:118:The release has 2,184,964 trainable connector parameters and frozen Qwen3.5-4B weights. Its pair MLP already couples codes via \(e_1,e_2,e_1-e_2,e_1e_2\). The phase branch maps six slots at \(h,h+\ell,t_p\) to separate tokens. The chat prompt interleaves the embeddings with text, including the numerical condition metadata. The `zero` arm zeros the six slots while retaining public prompt metadata. The signed-CFO train cells have \(\lvert\omega\rvert\le0.25,\ h+\ell\le5,\ \lvert\theta\rvert\le1.25\). The full Stage 3D run also includes zero-CFO replay C1–C3 with its own timings. The CPU surrogate uses only the ten signed-CFO training settings. [Connector and prefix assembly](../../stage3d_qwen_signed_phase/qwen_connector.py) · [Dataset](../../stage3d_qwen_signed_phase/data_stage3d.py) · [Training construction](../../stage3d_qwen_signed_phase/train_eval_stage3d.py)
research/stage4_math_and_design/M1/THEOREM_M1.md:122:| G1 | 1.60 | 23.80% / 1.4452 | 19.41% / 1.4809 | 64.62% / 0.9411 |
research/stage4_math_and_design/M1/THEOREM_M1.md:123:| G2 | 0.80 | 33.88% / 1.3358 | 36.36% / 1.3081 | 46.78% / 1.0222 |
research/stage4_math_and_design/M1/THEOREM_M1.md:127:Work F gives orientation gain \(+0.00982\) and mean swapped-minus-matched NLL \(+0.01837\) across phase seeds. The phase seed-averaged table has mean-square metadata effect \(1.85378\cdot10^{-4}\), waveform effect \(1.11977\cdot10^{-3}\), and interaction \(6.36253\cdot10^{-5}\). The physical calibration has metadata effect \(0.201603\), over a thousand times the host's mean-square value. Work G has rot01 and rot11 mean maximum coordinate differences of \(0.05914\) and \(0.12627\), with worst tested discrepancy \(0.48167\) at rot01. These measurements establish weak response in the correct signed direction and broken quotient equivariance for the trained host. [Stage 3D summary](../../stage3d_qwen_signed_phase/results/summary_master.json)
research/stage4_math_and_design/M1/THEOREM_M1.md:139:For code pair \(z\), use \(D_z=p_+(z)-p_-(z)\) from `cal_ref` and \(w_z=\min(m_+(z),m_-(z))/\sum_{z'}\min(m_+(z'),m_-(z'))\) from the archived conditional code masses. Compute \(M=\operatorname{diag}(\sqrt w)D\) and the Gram spectrum of \(M^\mathsf TM\). The row sum is zero. In a cyclic orbit of a row, a nonzero frequency-1 component spans a real plane and a nonzero frequency-2 component adds the Nyquist line. The verifier reconstructs the half-count ratios and finds both on the positive-weight orbit of code pair index 0: rows \(0,64,128,192\). Thus the **archived calibration table has rank three**. This is a statement about the finite reference table, with its calibration sampling error.
research/stage4_math_and_design/M1/THEOREM_M1.md:141:| Contrast | Nonzero Gram eigenvalues | Frobenius norm | Stable rank | Effective rank |
research/stage4_math_and_design/M1/THEOREM_M1.md:144:| Qwen phase, mean probability table | \(5.76490\cdot10^{-4},\ 2.43383\cdot10^{-4},\ 1.80399\cdot10^{-4}\) | 0.031627 | 1.73511 | 2.63904 |
research/stage4_math_and_design/M1/THEOREM_M1.md:146:With \(p_i=\lambda_i/\sum_j\lambda_j\), the effective rank is \(\exp(-\sum_i p_i\log p_i)\) and the stable rank is \(\sum_i\lambda_i/\lambda_1\). An arbitrary rank-one matrix would retain at most \(50.19\%\) of the physical Frobenius energy and leave a relative Frobenius residual of \(0.70576\). The host phase contrast has \(3.52\%\) of the physical Frobenius norm and relative Frobenius error \(0.99076\). Work F's orientation gain is its normalized Frobenius projection onto \(M\). Its factorial effect instead uses a paired-mother measure and a signed contrast scaled by \(1/2\). A probability-spectrum rank is different from attention rank or additive-logit rank, since softmax is nonlinear.
research/stage4_math_and_design/M1/THEOREM_M1.md:158:Top-1 accuracy can fall on a complete orbit. Use aligned target 0. At three quarters of the orientations, assign the correctly classified vector \((0.35,0.34,0.16,0.15)\), and at the other quarter assign \((0.01,0.97,0.01,0.01)\). The average is \((0.265,0.4975,0.1225,0.115)\). Accuracy drops from \(75\%\) to zero while NLL drops from \(1.93866\) to \(1.32803\). Work G reproduces these metric directions on its three sampled rotations: NLL \(1.44523\to1.41131\), and accuracy \(23.80\%\to17.05\%\). The paired 95% interval for \(P f\) minus raw accuracy is \([-7.09,-6.41]\) percentage points. The verifier replays the calculation from the released raw score tables and locked temperatures. Symmetry projection alone does not restore the signed physical decision. [Stage 3D projection and metrics](../../stage3d_qwen_signed_phase/recompute_all.py)
research/stage4_math_and_design/M1/THEOREM_M1.md:163:2. **Couple evidence and phase.** Start with canonical source distributions, initializing from the frozen S17 table. Propagate suitable circular moments with \(U_n(d)\) on each edge. For instance, transport \(m_{1,n}\) forward to the repeat target and de-rotate the second-window evidence with the known signed mean, while marginalizing uncertainty with nonnegative diffusion. Feed the real and imaginary components, source probabilities, and physical head features into the **same** trainable token projector. For exact continuous likelihood work, integrate a positive density or use Fourier harmonics appropriate to the sector and pilot maps. A first-moment feature alone cannot preserve a multimodal likelihood.
research/stage4_math_and_design/M1/THEOREM_M1.md:164:3. **Emit packed soft tokens.** Use a fixed 128-wide evidence vector and a 16-wide task embedding. Map through `Linear(144,128)`, GELU, and `Linear(128,1024)`, then reshape to eight 128-wide tokens. Apply a shared `Linear(128,2560)` and `LayerNorm(2560)`. Use a fixed task prompt and canonical candidate order in the existing Qwen four-candidate scoring interface. Train through the input embeddings with host weights frozen. Equivalent items can share an evaluation when they have identical canonical evidence and context.
research/stage4_math_and_design/M1/THEOREM_M1.md:165:4. **Use a bounded physical readout.** For Q3 repeat, the tested CPU implementation interprets canonical host scores as \(\delta=0.15\tanh(\ell_{\rm host})\), then transports the source \(\operatorname{softmax}(\log A_u+\delta)\) through the AWGN/Brownian kernel. A host bypass returns the direct S17 baseline. Q1 reads a relative root difference, Q2 reads a root relative to the common pilot frame, and post-pilot repeat integrates over the target phase conditional on the M3 factors. For an exact transport law as \(\theta\) changes with upstream evidence fixed, keep the source reweighting independent of target transport, or use the bypass. The tested residual reads coupled phase features, so it can change the source adjustment as \(\theta\) changes. The discrete \(H\) law holds for the composite in either case.
research/stage4_math_and_design/M1/THEOREM_M1.md:190:The verifier uses the production loader and fingerprint, regenerates the paired mothers, and matches all blind seeds, code pairs, and repeat targets to the archive. It uses 20,480 signed-CFO training rows, 16 fixed epochs, three optimizer seeds, and a single shared frozen MLP architecture with identical weights and fixed text embedding. A has 6,936 trainable parameters and B has 6,520. B uses a frozen physical prior while A learns from the inputs only. Scores use fractional ties. Paired intervals average signs and rotations within each 64-mother block. The Q1, Q2, and pilot-task tests exercise the quotient lift with synthetic canonical scores. Trainable end-to-end metrics are for Q3 repeat, with Q3 independent evaluated as a uniform control.
research/stage4_math_and_design/M1/THEOREM_M1.md:201:Each B seed exceeds 55% on G1 and G3. B has zero absolute probability defect when reusing canonical representatives, and at most \(5.96\cdot10^{-8}\) on ungrouped batched evaluation. The five task lifts and joint Q2 coordinate check all have zero defect. A breaks equivariance by up to \(0.77851\) but exceeds chance on G1 and G3. The requested A collapse is therefore unreproduced on this surrogate. The Stage 3D Qwen run and the unrotated S17 MAP control both show below-chance accuracy in those cells. B's top-1 parity with the physical-head-only oracle leaves the contribution of the frozen host unresolved.
research/stage4_math_and_design/M1/THEOREM_M1.md:203:- **M2** · Which matched Qwen ablation will isolate quotient sharing, the physical head, and token coupling? Should physical-head-only versus physical-head-plus-host-residual be the primary paired contrast?
research/stage4_math_and_design/discussions/M1/prompt_r2.md:1:# Track M1 (Round 2): Cross-Track Convergence — Pure-Host vs Physical-Residual Ablation, 16-Bit/s17 Dual Cache & Real-Data Group Action
research/stage4_math_and_design/discussions/M1/prompt_r2.md:6:1. **M2 (Stage 3D Forensics)**: On `blind_q3_scores.npz`, `G1` locked `NLL(zero) - NLL(phase) = +0.035650` splits into `+0.036543` scalar temperature mismatch and **`-0.001017` reverse-conditional KL**; after seedwise scalar recalibration, **`zero` beats `phase` across all 5 cells** (`G1: -0.000893`, `G2: -0.027716`, `G3: -0.000430`, `G4: -0.026351`, `H: -0.003695`), despite `0.3631 nats` signed CFO MI available in `s17`! M2's Stage 4 gate requires post-dev-calibration NLL/Acc/NIG ($\eta_{\min}=0.10$), ECE $\le 0.05$, equivariance defect $\le 10^{-5}$, and fixed-text slot interventions.
research/stage4_math_and_design/discussions/M1/prompt_r2.md:9:4. **M5 (Universal Stage 4→6 Cache)**: Proved `4-bit s17` merges amplitude/SNR (`|T|=0.02 vs 0.5833 -> 0.4804 TV`) and loses `0.3119 nats` on Stage 5 joint event+location (`1.2788 -> 0.9669`), AND exhibited two IQ pairs sharing exact `(T, E)` that produce different `s17` inner codes `[2, 0]`. M5 converged on a **16-bit bit-exact canonical extension**: `[2b turn | 2b s17_inner | 6b canon_angle | 4b log_radial | 2b resid_quality]` (recovering `s17` by right-shift while matching `Full IQ` closely in Stage 5–6).
research/stage4_math_and_design/discussions/M1/prompt_r2.md:19:     - **`Arm 1` (`Equivariant-Host-Pure`)**: quotient canonicalization (`canonical_input` + `rotate`) + multiplicative phase-coupled continuous summary fed purely as **soft tokens into the frozen host**, with **NO physical logit bypass** (`q_host` computed strictly from frozen LM / frozen surrogate logits, with isotropic `W_out` geometry)!
research/stage4_math_and_design/discussions/M1/prompt_r2.md:21:   - Implement and run **BOTH `Arm 1 (Host-Pure)` AND `Arm 2 (Physical+Residual)`** in `verify_m1.py` on the SAME frozen CPU surrogate host and identical `|\omega| <= 0.25` training split, testing `G1..G4`, equivariance defect (`<= 1e-6`), and M6's rank/margin candidate-Jacobian diagnostic! (Does `Arm 1 Host-Pure` ALSO extrapolate >55% past $\pi/2$ on `G1/G3`, or does frozen random projection struggle without coupled tokens / sufficient width?)
research/stage4_math_and_design/discussions/M6/prompt_r2.md:3:You are resuming persistent Codex session **Track M6** (`01a2194c-3b55-7c13-89f6-3ed11a482246`) in `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`. Keep edits STRICTLY inside `research/stage4_math_and_design/M6/` and run Python ONLY via `uv run --python 3.12 ...`.
research/stage4_math_and_design/discussions/M6/prompt_r2.md:11:   - **Crucial Synthesis Decision Needed From M6**: Merge M4's VRAM-safe pure `A100-80GB` safety with M6's Tier-2 **1-seed pilot futility gate** (`18%` steps on disjoint dev BEFORE buying full `384`-step × `4`-arm comparison OR continuing `Arm 1 / Arm 2` to extra blocks/seeds!), keeping TOTAL worst-case cap **strictly `<= $9.50`** inside `~$12.88` remaining Modal free credits (`$0.00` net spend guard)!
research/stage4_math_and_design/discussions/M6/prompt_r2.md:14:   - Update your Tier-1 CPU Gate & Stage 4 DAG to explicitly require **BOTH** `Arm 1 (Host-Pure)` AND `Arm 2 (Hybrid)` Jacobian/OOD checks AND counterfactual slot/text controls!
research/stage4_math_and_design/discussions/M6/prompt_r2.md:15:3. **Dual-Cache Zero-GPU Upgrade (`4-bit s17` + `16-bit canonical` M5 Packet) + `RefSufficientPool` Non-Double-Counting (M3+M4)**:
research/stage4_math_and_design/discussions/M6/prompt_r2.md:21:1. **Publish ONE Converged, Contradiction-Free Master Modal Allocation Ledger (`<= $9.50` Hard Cap, 100% A100-80GB Safe)** unifying M4's `52.97 GB / 45.41 GB` VRAM numbers, M1's 4-arm causal matrix, M2's fixed-seed conditional inference scope + dev-disjoint futility stopping, and M5's dual-cache Stage 4→6 export! Explicitly show how much budget goes to Stage 4 Pilot, Arm Comparison, Real Pluto (`2,240`), Real CTTC (`1,440` incl. held-out `f708_snr10`), and Reserve (`>= $3.38` buffer vs `$12.88`!).
research/stage4_math_and_design/discussions/M6/prompt_r2.md:23:3. **Cross-Host & Fair `LEMMO` Execution Law (Stage 6 Readiness)**: Explicitly verify mathematically how fixing the `16-bit canonical` cache + `RefSufficientPool` lets us swap only tiny host adapters (`4B -> 9B -> 27B`) and run exact matched-support `LEMMO` comparison with minimum redundant experiments!
research/stage4_math_and_design/M6/THEOREM_M6.md:7:Let $A\in\{0,\ldots,m-1\}$ be the answer label, $Y$ the permitted raw observations at the time of the question, $Z=T(Y)$ the frozen cache, and $D$ the public condition, question, and evidence available through that time. $D$ includes the revealed references and their provenance where these enter both comparators, plus the host's instruction or the variables needed to render it. Condition on the frozen checkpoint and a known seed, or include the independently assigned seed in $D$. Define
research/stage4_math_and_design/M6/THEOREM_M6.md:36:Each term is nonnegative. A transition preserves Bayes log-loss exactly when its posterior is unchanged almost surely. The host term includes the conversion from hidden states to the restricted candidate score vector. One can insert a hidden-state rung and split that term if the hidden states are recorded. A small output dimension or a low rank in a linearization does not by itself establish positive conditional MI. Even one real number can encode a finite cache injectively. Conversely, an injective prefix can let the host output unusable probabilities. A fixed positive scalar temperature is invertible on centered logits through $L=\tau\Pi\log q_\tau$. It cannot remove a score collision or change the top-1 decision. The difference $R(q_\tau)-R(q_1)$ is signed and measures the effect of the chosen temperature. It differs from the nonnegative calibration term above.
research/stage4_math_and_design/M6/THEOREM_M6.md:40:The verifier replays the real NPZ artifacts. Q3 is synthetic at 2 dB, with paired CFO signs and three rotations per mother. It averages scores over seeds, rather than scoring an ensemble. Full IQ uses known channel parameters and the repository's matched synthetic likelihood. The s17 table is a physical readout and can have its own mismatch to $P(A\mid Z,D)$. In general the observed direct-minus-IQ gap equals the population encoding term plus the difference of the two readouts' KL regrets, up to sampling. If Full IQ is the true posterior under the matched law, the population direct-minus-IQ gap upper-bounds the encoding penalty. If the Full-IQ solver is approximate, include its error in any certified bound. No intervening prefix, hidden-state posterior, or cost-free counterfactual training is available in these archives. The replay localizes the **operational** deficit to the path after the cache, with temperature measured separately. The internal conditional MI terms remain unidentified. If independent validation detects no input leaks, the comparator already witnesses that this cache permits much better G1 decisions.
research/stage4_math_and_design/M6/THEOREM_M6.md:45:| G2 both interpolated | 512 | 46.777% / 1.019346 | 46.777% / 1.022230 | 33.876% / 1.335826 | 36.361% / 1.308067 | .002884 | .313596 | .285837 |
research/stage4_math_and_design/M6/THEOREM_M6.md:47:| G4 CFO extrapolated | 512 | 49.219% / 1.003738 | 47.754% / 1.015851 | 33.290% / 1.334667 | 34.961% / 1.309751 | .012113 | .318816 | .293900 |
research/stage4_math_and_design/M6/THEOREM_M6.md:48:| H_c4_035 | 1,024 | 63.867% / .942628 | 63.770% / .950133 | 25.765% / 1.416987 | 23.172% / 1.435077 | .007505 | .466854 | .484944 |
research/stage4_math_and_design/M6/THEOREM_M6.md:50:For G1 the paired 64-mother block 95% intervals are **[.009178, .014998]** for s17 minus Full IQ and **[.491294, .516912]** for phase minus s17. Phase's observable telescope is .012087838 + .478025740 + .026076989 = .516190567 nats, using unit-temperature host scores for the middle contrast. About **97.66%** of phase's excess over Full IQ lies after the s17 readout when measured at locked temperature. The G1 phase-versus-zero uplift did meet Stage 3D's relative target, even though both arms exceeded the log-loss of a uniform predictor (1.386294). G1's co-rotation worst coordinate defect was **.380019** in phase and near machine precision for the direct readout. Offline 16-way group averaging brought phase NLL to **1.411312** and accuracy to **17.050%**. On the archive's sampled rotations these are distinct endpoints. The all-rotation `H_c4_035` row above is a fresh replay from the archive. Its identity-rotation, both-sign host metrics match Work F. The separate positive-sign Stage 3B lookup anchor in the master JSON is a different comparison.
research/stage4_math_and_design/M6/THEOREM_M6.md:55:| Valid | 66.406% / .819207 | 28.776% / 1.388883 | 26.628% / 1.410690 | .569676 | .591483 |
research/stage4_math_and_design/M6/THEOREM_M6.md:56:| Duplicate, dedup | 66.406% / .819207 | 28.776% / 1.388883 | 26.628% / 1.410690 | .569676 | .591483 |
research/stage4_math_and_design/M6/THEOREM_M6.md:60:| Fresh resample | 74.219% / .663189 | 26.432% / 1.405640 | 25.781% / 1.406397 | .742451 | .743208 |
research/stage4_math_and_design/M6/THEOREM_M6.md:62:\* Duplicate-raw uses the existing deduplicated Work A row as its information-equivalent physical reference. Q2 contains 512 mothers at (+0.35), without a paired Full-IQ score in the NPZ. Work A is a cache/physical reference. The host receives s17 window codes and 12-dimensional continuous reference features. Fresh evidence improves the Work A NLL by **.156017** and worsens phase NLL by **.016758**. The raw duplicate worsens phase accuracy by **2.474 pp** (mean TV **.040377**), while the dedup arm has zero TV. Two tokens for one acquisition and tokens for two genuine acquisitions test different physics. Both the actual length change and the added attention content can matter. A matched padding/mask experiment is needed to identify their separate effects.
research/stage4_math_and_design/M6/THEOREM_M6.md:64:Historical scope. `PROTOCOL_LOCK.json` records six frozen-host checkpoints, 8 epochs, 160 steps per epoch, batch 32, microbatch 16 and dev-fit temperatures. The Work E variance scenario used $q=.35,\rho=.01$, mother NLL SD .5, pilot block SDs .02104014/.01928086, and planned 131 blocks for the joint endpoint gate. Stage 3D's report orders smoke and scorer diagnostics, six trains (2,884–3,218 s each), independent dev variance lock, then blind scoring. The **\$17.12** receipt includes smoke, one OOM/benchmark, all six trains, dev and blind evaluation, and the aborted historical Ministral check. The archival charge was **\$0.00** after allowances. The 150,912 G1 rows are logical predictions per arm from a verified finite table, not 150,912 distinct forwards.
research/stage4_math_and_design/M6/THEOREM_M6.md:70:Set $S=(Z,C,\mathcal R)$, with causal metadata (C) and unique allowed references $\mathcal R$. Under a specified kernel, evaluate (p(a,s)\) by summing or integrating the latent state, then compute $H(A\mid S)=-\sum_{a,s}p(a,s)\log p(a\mid s)$ or its integral. For Q3, tabulate by sign/delay/cache pair and compare the same $D$ in the unrestricted physical solver. For Q2, repeat with none/valid/fresh and counterfactual visibility. Integrate correlated references jointly conditional on latent trajectory. A distinct second acquisition need only be independent conditional on the latent state for multiplication of its likelihood. One physical sample duplicated has one likelihood factor. The entropy drop on reveal is exactly $I(A;R_{\rm new}\mid S)\ge0$ under a common nested law. It is an expectation. Sample-level losses can increase after a valid observation.
research/stage4_math_and_design/M6/THEOREM_M6.md:75:- On finite locked support, form a simultaneous confidence set $\mathcal P$ for the true joint law of $(A,S)$, then use $\inf_{p\in\mathcal P}H_p(A\mid S)$ and $\sup_{p\in\mathcal P}H_p(A\mid S)$. For i.i.d. units, a Bonferroni union of exact binomial intervals for fixed atom probabilities, intersected with the simplex, is one valid construction. For independent clusters, use bounded cluster contributions to each atom and a simultaneous concentration bound. If the support is large the interval may be vacuous. Binning a continuous reference gives an entropy **upper** bound on the richer interface. A lower bound for the binned interface applies when the bins are what the host receives.
research/stage4_math_and_design/M6/THEOREM_M6.md:78:**Proposed decision margins.** On development folds, lock an admissible NLL target $r_*=R(q_{\rm IQ})+.05$ nats and a useful mean reveal improvement of .02 nats against the no-reveal control. Use each contrast at fixed context and comparable label distributions, with simultaneous bounds for multiple gate decisions. A certified $L_S>r_*$ ends the current representation branch. With entropy brackets $[L_0,U_0]$ and $[L_1,U_1]$ before and after reveal, $\max(0,L_0-U_1)\le I(A;R_{\rm new}\mid S)\le U_0-L_1$ for simultaneous valid brackets. An upper bound below .02 ends a **Bayes-gain** claim for that reveal contract. A direct readout whose paired upper regret bound is within .05 and whose reveal lower improvement bound exceeds .02 is a practical existence witness. Otherwise report uncertainty, improve the physical kernel or support, and defer GPU. At final release require independent upper confidence bounds within .05 of both the permitted Full-IQ and direct-cache comparators on the primary cell and a lower confidence bound above .02 for host reveal improvement where a gain was declared. These margins are Round 1 proposals for registration.
research/stage4_math_and_design/M6/THEOREM_M6.md:82:Derive the action from the intervention before defining the model. In synthetic Q3, for $g=(r_1,r_2)\in C_4^2$, coordinate relabeling of cache windows has $z_i\mapsto z_i+4r_i\bmod16$ and answer transport $P_g=\mathrm{roll}(r_1)$ for the repeat target. Rotate reference coordinates consistently under the experiment's physical action. A coordinate change leaves CFO unchanged. A new physical phase jump or fresh measurement can change the answer and must use a separate intervention. For actions with $P(A\mid gx)=P_gP(A\mid x)$, require $q(gx)=P_gq(x)$. At fixed points require stabilizer invariance. This is an exact contract. On any equal-weight pair with target symmetry and probability defect $d=\mathrm{TV}(q(gx),P_gq(x))$, triangle inequality and Pinsker give mean KL regret at least $d^2/2$. Thus some sufficiently large defects also violate a quantitative loss gate, although a nonzero defect by itself proves failure only of the exact contract.
research/stage4_math_and_design/M6/THEOREM_M6.md:92:**Stage 3D interpretation.** The cache-pair MLP sees $z_1,z_2$; a separate phase MLP sees six trigonometric slots; continuous reference features go through a separate MLP per `<REF>`. Signed CFO also appears as text in both arms. A linear fixed-Jacobian map about a common prompt/fixed reference layout yields separate additive score contributions. Qwen's true nonlinear attention can mix those streams and jointly condition its Jacobian. This makes Stage 3D's additive extrapolation risk diagnosable before training, alongside a provable failure of any required symmetry or idempotence whose implementation lacks the corresponding contract. It does not make the observed **.504103** nats or the causal contribution of each suspected mechanism predictable from source alone. Timing, caching and low-precision instability were additional measured risks. A positive CPU joint-readout result establishes feasibility for that readout.
research/stage4_math_and_design/M6/THEOREM_M6.md:96:For development example (i), flatten the fixed-length embeddings $e_i=\mathrm{vec}(E_\theta(x_i))\in\mathbb R^{Kd_{\rm LM}}$, define centered host scores $F_i(e_i)$, and compute the actual candidate Jacobian $J_i=D_eF_i(e_i^0)$. For multi-token candidates differentiate the full sum $\sum_t\log p(w_t\mid w_{<t},e_i)$, including gradients of the log normalizer and the cached context. Use candidate-vector JVPs or VJPs in FP32. With a shared adapter perturbation $\delta$, stack $A_i=J_iD_\theta e_i$ and $d_i=\Pi\log p^*_i-F_i(e_i^0)$ after weighting examples by their locked masses. For the stack (A,d), a frozen tangent can achieve residual $\epsilon$ iff $\inf_{\|\delta\|\le r}\|A\delta-d\|\le\epsilon$; without the norm cap the projection residual is $\|(I-AA^+)d\|$. A rank threshold is necessary only for a declared target subspace. A one-dimensional joint task can succeed through a rank-one interface. A shared adapter may require stacked rank across inputs even if all individual $J_i$ have rank (m-1). Fixed thetas, mask patterns and prompts define which local claim is tested.
research/stage4_math_and_design/M6/THEOREM_M6.md:98:If the stacked score map has Hessian operator norm bounded by (M) over the trust region, Taylor's theorem yields an unavoidable residual ≥ $\|(I-AA^+)d\|-Mr^2/2\$, and the gain cannot exceed $\|A\|r+Mr^2/2$. If $d\in\operatorname{im}A$, smallest retained singular value (s>0) and $\|d\|/s\le r$, select $\delta=A^+d$. The nonlinear residual is at most $M\|d\|^2/(2s^2)$. With softmax, the log-sum-exp Hessian is $\mathrm{diag}(p)-pp^\top$, has norm ≤1, and has curvature ≥(p_{\min}\) on centered vectors when all probabilities on the segment are ≥(p_{\min}\). Taylor's integral gives $p_{\min}\|e\|^2/2\le\mathrm{KL}(p^*\|q)\le\|e\|^2/2$ for centered logit error (e) with positive matched target. Handle absent sectors through a registered probability floor or a separate support analysis. This converts a certified local rank/curvature inequality into an NLL gate. A finite-difference estimate of (M) is diagnostic until justified as an upper bound. With ordinary training outside the certified trust region, the rejection certificate applies only to the local model.
research/stage4_math_and_design/M6/THEOREM_M6.md:100:For least squares in this tangent model, $K=AA^\top$. Gradient flow decays each representable residual eigenmode as $e^{-t\lambda}$; discrete gradient descent multiplies it by $1-\eta\lambda$ for $0<\eta<2/\lambda_{\max}$. The nullspace remains. With fixed train/test Jacobians the linearized test prediction update for minimum-norm interpolation is $A_{\rm test}A_{\rm train}^+d_{\rm train}$. Compare to physical contrasts on a held-out joint CFO/angle grid. A poor result isolates this fixed-tangent extrapolator. A short actual-host test should use locked prompt positions, candidate tokenization, phase signs, matched lengths, and representative physical margins. Use analytic contrast and small CPU probes for the <30 s tier. Full 4B, 9B or 27B Jacobian timings and memory depend on the pinned host and are unmeasured here. A partial host or quantized CPU copy can reject an interface on a structural property shared with the target, or give a heuristic diagnostic. A quantitative negative result transfers only with an error bound on the difference between their operators. [Jacot, Gabriel and Hongler, 2018](https://papers.neurips.cc/paper_files/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html)
research/stage4_math_and_design/M6/THEOREM_M6.md:132:With one-sided alpha $\alpha=.05$, use O'Brien–Fleming-style spending $a(t)=\bar\Phi(z_{1-\alpha}/\sqrt t)$ or predeclare Pocock-style $a(t)=\alpha\log[1+(e-1)t]$ for more early alpha. For the former, $a(.18)=.00005288548$, $b_1=3.876957$; solve $a(t)+\int_{-\infty}^{b_1}\varphi(z)\bar\Phi((b_2-\sqrt tz)/\sqrt{1-t})\,dz=.05$ for $b_2=1.645088$. Refusing continuation can only remove rejection paths if boundaries are held fixed. Componentwise monotonicity in $\mu$ puts the least favorable null at zero. Familywise guarantees require preallocated alpha across multiple promotion hypotheses, or a gatekeeping order; for mandatory jointly successful endpoints, an intersection-union gate can use the full level per component while requiring all successes. [Lan and DeMets, 1983](https://eclass.uoa.gr/modules/document/file.php/MATH301/PracticalSession3/Lan_DeMets_1983.pdf)
research/stage4_math_and_design/M6/THEOREM_M6.md:136:The paired 10,000-candidate replay assumes a 60% structural-null stratum ($\mu=-.8$, CPU pass .10), 15% boundary null ($\mu=0$, pass .90) and 25% working ($\mu=3.6$, pass .99). Pass draws are conditionally independent given stratum. These are planning assumptions, not measured rejection rates for M1–M5. Both policies see the same potential Gaussian scores. Fixed spends the archival \$17.12 envelope for every candidate. Sequential spends a modeled 8.5% if CPU passes and the pilot is bought ($.75\times.18/3+.04$ of fixed, comprising a training share and a pilot overhead share). It pays the remaining fraction on continuation including early efficacy. This scales an archival bill as a scenario; it neither prices an individual seed retrospectively nor assumes a \$17.12 charge in the Stage 4 cap. If $p_s(\mu)=P(\mathrm{CPU\ pass}\mid\mu)$, then
research/stage4_math_and_design/M6/THEOREM_M6.md:160:| Download/reference alignment, causal frontends, dataset audit and weight staging to Volume | CPU | 7,200 | .00006172 | .444384 |
research/stage4_math_and_design/M6/THEOREM_M6.md:161:| Pinned synthetic smoke, cache/version parity and timing | A100-80GB | 450 | .00075572 | .340074 |
research/stage4_math_and_design/M6/THEOREM_M6.md:172:Make dataset acquisition, weight pulls, cache/table computation, causal normalization, provenance linting, calibration folds, and source guard checks CPU jobs. Mount prepared read-only dataset features for evaluation and commit isolated outputs; local CPU reduces cloud seconds where practical. The three synthetic seeds train the one surviving Equivariant+RefPool design. Historical phase/zero archives, physical baselines and exact intervention controls give context. This envelope does not buy six full retrained comparator checkpoints. Any newly trained matched-baseline arm is a separate priced decision. Real evaluation uses those same frozen synthetic-trained weights and frozen candidate schema. If real development data are used to fit preprocessing or a temperature, freeze parameters by source fold and report strictly zero-shot calibration separately where claimed.
research/stage4_math_and_design/M6/THEOREM_M6.md:174:At the historical 1.9757 s/step the 2,707 continuation steps would take ~5,348 s before overhead. Require a receipt within 2.30 s/step, leaving ~524 s for all continuation overhead at that worst allowed rate. The new preprocessing, fused kernels and longest Q2 inputs can change this. At most 6,144 Q3 prefixes cover 4 cells × 2 signs × 256 exact keys × 3 seeds, provided the key includes all prefix state, dtype and batch policy. Q2's seven evidence arms require at most 10,752 prefixes for 512 mothers × 3 seeds. The 1,050 s synthetic scoring line therefore requires 16.1 prefixes/s before overhead. Cache prompts only across identical embeddings, masks, positions, template, candidate scope and attention state. Distinguish KV/linear-state sharing from lookup deduplication. Fix the padding and batch policy and require per-probability and paired endpoint parity on a receipt set before any blind inference. Historical BF16 differences across batch settings reached .067 in probabilities with per-cell argmax agreement as low as .6875. Historical A100 b64 peaked at 45,410 MB on inference, so use smaller fixed batches or prevalidated kernels on 24 GB cards. Keep precision locked across comparable arms.
research/stage4_math_and_design/M6/THEOREM_M6.md:182:    A[4A Causal data manifest and physical controls] --> B[4B Cache risk brackets and collision tests]
research/stage4_math_and_design/M6/THEOREM_M6.md:192:    H --> I[6C Qwen3.8-27B portability]
research/stage4_math_and_design/M6/THEOREM_M6.md:193:    G --> L[6D LEMMO common-support audit]
research/stage4_math_and_design/M6/THEOREM_M6.md:198:| Before allocating GPU | Falsifiable CPU gate and branch decision |
research/stage4_math_and_design/M6/THEOREM_M6.md:201:| 4B cache | Compute synthetic exact/controlled likelihood and causal real-data bounds on dev and support. Pass with the existence witness in §2.1. For a certified insufficiency, test an equal-budget canonical complex cache with identical observable inputs. Stop the branch if both cannot meet the objective. |
research/stage4_math_and_design/M6/THEOREM_M6.md:202:| 4C interface | Enumerate $C_4^2$ on finite cache codes and induced labels, check identity bytes for duplicate, masked, permuted and padded records, and verify genuine fresh effects can survive pooling. Require OOD mixed contrasts and a sufficient singular/margin witness for the design subspace. Check dev Jacobian stack rank before investing in a wider connector. |
research/stage4_math_and_design/M6/THEOREM_M6.md:203:| 4D promotion | CPU freeze the splits, paired estimand, useful margins, tolerances, calibration rule, alpha and maximum allocation. The GPU pilot verifies the timing/VRAM/parity contract and the dev continuation criterion. Release only after independent G1 primary, G2–G4 strata, Q2_C4, three seeds and exact controls. If eligibility fails, archive either the certificate or the prespecified inconclusive outcome. |
research/stage4_math_and_design/M6/THEOREM_M6.md:204:| 4E–4F real | Same locked candidate schema and CPU score oracle on each cohort, the cached sufficient-statistic check, paired source/block plan, valid-vs-none/fresh sign, and independent negative controls. Preserve raw vs official filtered CTTC as views of one source. Log natural vs manipulated cases, evidence location and retained/revised/incorrect counts. A validated explanation of a source-specific negative result can feed 5A as an identifiability constraint. |
research/stage4_math_and_design/M6/THEOREM_M6.md:206:| 6A active read | With history $D_t$ and candidate window $W_j$, Bayes expected log-loss drop is $I(A;W_j\mid D_t)$. Verify the exact-small-case policy through budgeted dynamic programming using conditional revealed data. Skip zero-information reads when they have positive cost. Compare fixed, random, local and global content selection at identical read counts, acquisition permissions, cache bits and context. Gate monotone posterior accounting and cite concrete time spans with uncertainty or abstention. |
research/stage4_math_and_design/M6/THEOREM_M6.md:207:| 6B–6C 4B → 9B → 27B | Freeze the waveform representation, evidence semantics, tasks and read budgets. Pin each host/tokenizer/inference implementation; rerun candidate Jacobian and proof obligations separately, and size memory/throughput before paid compute. Fit only a host-specific adapter on the matched train budget. 4B→9B can inform same-family scale; 9B→3.8-27B also changes family revision and cannot isolate parameter count. Include paired ordinary text/image regression as in the integration plan. Separate follow-on budgets apply. [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
research/stage4_math_and_design/M6/THEOREM_M6.md:208:| 6D LEMMO | CPU lock runnable revision, permissions, train overlap, questions, answer normalization and common-supported endpoints. The LEMMO paper uses Qwen3.5-4B with LoRA and, depending on task, Q16/Q128 or a direct projector. Report the native systems under documented inputs. For causal attribution, also match base revision, physical observations and reveals, sampling and timing, cache/context bits, readable windows, quantization, training data, trainable parameter and optimizer budgets where feasible. Record reruns and latency. If an identical matched run cannot be made, narrow the comparison to the observed common support. [LEMMO paper §3–4 and Appendix B.4](https://arxiv.org/html/2611.09476v1), [LEMMO repository](https://github.com/N1rv4n4/LEMMO) |
research/stage4_math_and_design/M6/THEOREM_M6.md:213:**Why these edges exist.** P1–P3 show strong matched classical baselines, position shortcuts and static-surrogate scope. P4–P6 require separating physical frontends from learned contributions and distinguishing missing from zero. P7–P8 show that large probability updates can yield few corrected blocks and that mismatched past observations can hurt. P9–P11 require state-dependent reference reliability, phase-gauge accounting and counting newly wrong alongside corrected decisions. P12 requires unique evidence accounting and compares identifiable global references against naive locality; plug-in point estimates can lose mixture information. The earlier P12 Stage 5 fusion results scoped neural gains to the tested distribution. Stage 3D adds the strong direct-reader/weak-host distinction, failed raw rotation laws, redundant reference sensitivity, and an expensive numerical parity lesson. The roadmap's anti-loop rule is to predeclare one new question, finish with a success case, a failure case and a changed decision, and archive small BER variations after two rounds. Record the missing observation required by an impossibility theorem. [Long-term plan](../../../docs/LONG_TERM_RESEARCH_PLAN.md), [Multimodal integration plan](../../../docs/MULTIMODAL_INTEGRATION_PLAN.md), [Stage 3D report](../../stage3d_qwen_signed_phase/REPORT.md), [Protocol lock](../../stage3d_qwen_signed_phase/PROTOCOL_LOCK.json)
research/stage4_math_and_design/M2/THEOREM_M2.md:1:# Track M2 | Stage 3D score forensics and the Stage 4 gate
research/stage4_math_and_design/M2/THEOREM_M2.md:3:Round 1. Losses and information are in nats. Positive paired differences favor phase, so the accuracy contrast is phase minus zero and the NLL contrast is zero minus phase. Each saved mother carries its signs and rotations together. Each of the three connector seeds has equal weight. All calibration fitted on blind labels is retrospective.
research/stage4_math_and_design/M2/THEOREM_M2.md:31:For logits \(\ell_i\), inverse temperature \(\beta\ge 0\), and empirical weights \(w_i\), let
research/stage4_math_and_design/M2/THEOREM_M2.md:79:Equality holds exactly when, almost surely, every label of positive posterior mass has the same probability across every aligned member. If the posterior has full support, this is equivariance almost surely. On an arbitrary finite design the exact formula adds \(L(q)-|G|^{-1}\sum_g L(R_gq)\) to the label-weighted Jensen gap. The complete orbits constructed from each saved mother have an invariant design by construction. Saved rotations are an incomplete orbit, so the restriction matters.
research/stage4_math_and_design/M2/THEOREM_M2.md:81:The proposed \(|G|^{-1}\sum_g D_{\rm KL}(Pq\Vert R_gq)\) is \(\sum_y m_y J_y\) pointwise. It corresponds to self-weighting by the pool. It equals the log-loss gain if \(\mathbb E[(p-m)\cdot J]=0\), in particular if \(p=m\). It is not a general equality even under exact invariance. For an invariant binary example, take two equiprobable observations interchanged by a group of order two with identity label action and \(p=(1,0)\) on both. The forecasts are \((0.8,0.2)\) and \((0.4,0.6)\). The pool is \((0.6,0.4)\), and \(J_1=\log(0.6/\sqrt{0.32})\), \(J_2=\log(0.4/\sqrt{0.12})\) differ. The loss gain equals \(J_1\). The reverse-KL mean equals \(0.6J_1+0.4J_2\).
research/stage4_math_and_design/M2/THEOREM_M2.md:99:## 3. Stage 4 frozen statistical gate
research/stage4_math_and_design/M2/THEOREM_M2.md:101:A claim passes when every endpoint for that claim clears its limit. Freeze the candidate, seed handling, allowed observation sets, per-class weighting, code below, multiplicity family, score floor, calibration, and all margins before unblinding. Apply the same three gate parts separately to the synthetic extrapolation claim, the Pluto claim and the CTTC claim. For a claim covering all four synthetic cells, require each cell to pass. Report each failed or indeterminate endpoint. A winner picked from several candidates requires a design split or inclusion in the multiplicity family.
research/stage4_math_and_design/M2/THEOREM_M2.md:103:1. **Absolute and comparative performance.** Use paired lower bounds. Require NLL to beat uniform, the dev-frozen prior, and the matched zero-phase system, with a \(0.02\)-nat increment proposed against matched zero. Require accuracy above analytic chance, a dev-frozen prior classifier, and the matched zero-phase system, with \(+0.02\) absolute over chance and zero proposed as initial margins. With imbalanced classes, also require outperformance of the fixed majority prediction or used prior competitor, and report balanced accuracy as a separately locked estimand. Require valid chance/random-label randomization tails for the accuracy and NLL statistics where the design supplies exchangeability. Choose an oracle with the same information budget as the candidate for the main NIG denominator and disclose a finer physical oracle separately. Propose \(\eta_{\min}=0.10\) and validate \(H(Y)-L(o)>0\) on an independent development design or by a co-primary lower bound. Test the linear contrast \(D_{\eta}=(1-\eta)H(Y)-L(q)+\eta L(o)>0\) rather than dividing two noisy quantities. Use \(H(Y)=\log K\) for a genuinely uniform target law. Otherwise predefine the target distribution or carry uncertainty in \(H(Y)\) by a joint confidence region and minimize the contrast over it. For real data without a defensible Bayes oracle, freeze a physically informed predictive reference, call the resulting metric reference-normalized, and withhold the Bayes-normalized claim. Requiring a learner to outperform a true Bayes oracle under proper loss is infeasible. A positive NIG floor defines a fraction of the attainable improvement. Controlled slot/text interventions form a further prerequisite for a specifically signed-slot mechanism claim. Require improvement with matched versus permuted or flipped slots at fixed text and same \(X\), with seeded training and ablations paired. Use the native receiver outputs for the gate.
research/stage4_math_and_design/M2/THEOREM_M2.md:104:2. **Calibration and structure.** Propose an upper bound of 0.05 on top-label ECE after dev-only calibration with ten frozen bins. Propose a paired Brier improvement of at least 0.005 against the strongest locked trivial or zero-phase Brier competitor, plus a strict improvement over every required comparator. For \(K=4\) uniform Brier is \(1-1/K=0.75\) with the unnormalized multiclass sum. Prelock a probability transformation ensuring finite NLL, for example \(q_\epsilon=(1-K\epsilon)q+\epsilon\) with small fixed \(\epsilon\), and score that forecast consistently. For calibration inference set \(d_j=\mathbb E[1_{\{\operatorname{conf}\in I_j\}}(h_Y-\operatorname{conf})]\); \(\operatorname{ECE}=\sum_j|d_j|=\max_{v\in\{-1,1\}^{10}}v^\top d\). Test all \(2^{10}\) linear upper-margin conditions as an IUT or use a valid joint confidence region for \(d\). Estimating the absolute value first and treating empirical ECE as an unbiased, normal per-mother loss has no exact justification. Require \(\max_{g,x}\|q(gx)-\rho(g)q(x)\|_\infty\le10^{-5}\) and projection/readout idempotence \(\max_x\|P(Pq)(x)-Pq(x)\|_\infty\le10^{-5}\). Verify across the whole finite code domain and all group elements on synthetic. For continuous real inputs distinguish formal architectural guarantees from the observed maximum on prespecified transforms. Test only valid actions with physically correct label permutations. Audit the deployed path and the offline projector separately.
research/stage4_math_and_design/M2/THEOREM_M2.md:105:3. **Hierarchy and multiplicity.** Keep all repeats, rotations, acquisitions, and matched arms inside their mother and higher cluster. Report a frozen-seed conditional estimate and, if claiming generalization to retraining, a separate random-seed inference. Use equal independent cluster weights for the following exact balanced analysis. Pluto supplies 16 units if they are independent captures. For a CTTC new-session claim use nine independent session means, with all eight clusters per session contributing to that session. A conditional claim on the fixed sessions needs the corresponding estimand and defensible within-session cluster independence. For a permutation or label-randomization gate, lock the allowed exchangeability orbit and recompute the same statistic throughout. Repeated/rotated labels move jointly with their mother and never split correlated session units as though independent. A synthetic randomized design with independent label assignments can instead rerandomize the sanctioned generator operation inside the original design. When there is no justified exchangeability, collect a randomized negative control. State whether it tests a sharp random-label null or a mean-effect margin. A permutation p-value \((1+\#\{T_{\rm perm}\ge T_{\rm obs}\})/(R+1)\) with uniform randomizations and ties included is conservative under the sharp design. Use exact enumeration when feasible.
research/stage4_math_and_design/M2/THEOREM_M2.md:109:For an oriented block contrast \(D_{bs}\) let \(D_{bs}=\mu+u_b+\nu_s+e_{bs}\), with independent Gaussian families of variances \(\sigma_u^2,\sigma_\nu^2,\sigma_e^2\). These are the assumptions for the exact parametric calculations. Blocks already aggregate mothers and correlated views. If \(m\) mothers in a block have common variance \(v\) and exchangeable within-block correlation \(\rho\), a block mean contributes \(v[1+(m-1)\rho]/m\); this design effect applies to the particular contrast and variance component being modeled. The Stage 3D Work E lock estimated block SD on a fixed-seed pilot, took the larger SD from pilot upper bound and design-effect scenario and used the conditional noncentral t. It did not price uncertainty across retrains.
research/stage4_math_and_design/M2/THEOREM_M2.md:137:Use the same formulas with \(B=J_0\), \(\sigma_u^2=\sigma_{\rm session}^2+\sigma_{\rm cluster}^2/m\), and \(\sigma_e^2\) replaced by \(\sigma_e^2/m\); an additional session-by-seed effect is another contribution to the collapsed interaction variance. Use planned balanced session-level estimation or a separately validated unequal-weight design. For a fixed-seed claim with independent bounded cluster means and arbitrary within-cluster dependence, a one-sided Hoeffding bound gives an alternative to Gaussian t inference. For mean of \(B\) equal-weight cluster means in \([l,h]\), reject when the empirical excess exceeds \((h-l)\sqrt{\log(1/\alpha)/(2B)}\). For fixed weights \(a_b\) summing to one, replace \(1/B\) by \(\sum_b a_b^2\). Unknown non-Gaussian random seed effects with only three seeds do not admit a generic exact t power calculation. Correlation modeling and null randomization require separate justifications.
research/stage4_math_and_design/M2/THEOREM_M2.md:139:For endpoint nulls \(H_{0j}\) and a joint claim, the null is their union. Valid p-values give \(p_{\rm IUT}=\max_j p_j\), since with at least one true null the event \(p_{\rm IUT}\le a\) is a subset of rejection of that true null. The exact joint rejection probability is the joint tail of the endpoint statistics, equal to the product of marginal powers only under independence. Its universal lower bound under the alternative is \(1-\sum_j(1-\pi_j)\). Across separately reportable claims, sort the IUT p-values and apply Holm's thresholds \(\alpha/(M-i+1)\). If a first true null is rejected, its p-value is at most \(\alpha/m_0\) where \(m_0\) is the number of true claim nulls. The union bound over those true nulls gives strong FWER \(\le\alpha\) under arbitrary dependence. This is a finite-sample guarantee conditional on valid endpoint tests. Under independence with all nulls uniform, the probability of any rejection is exactly \(1-(1-\alpha/M)^M\). [Holm (1979)](https://www.jstor.org/stable/4615733)
research/stage4_math_and_design/M2/THEOREM_M2.md:145:with positive denominator. At fixed \(S\) this is a necessary known-variance bound for the random-seed model. The verifier solves the exact t power and evaluates the crossed chi-square integral for finite counts. It uses a worst Holm first-step threshold \(\alpha=0.05/3\) and marginal power 0.95 to guarantee at least 0.90 power for the two historical paired endpoints if both assumptions and excess effects hold. This target does not imply 0.90 power for the full Stage 4 gate. Full planning requires variance and target excess for every required absolute/calibration endpoint and a joint power calculation or an explicit allocation of endpoint type-II errors. Stage 3D supplies a synthetic variance reference. It cannot measure Pluto or CTTC variability.
research/stage4_math_and_design/M2/THEOREM_M2.md:154:### Q3 Bayes terms and the paired decomposition
research/stage4_math_and_design/M2/THEOREM_M2.md:161:| G4 | 8 | 0.804198 | 0.271874 | 0.179398 | 1.015851 | 1.003738 |
research/stage4_math_and_design/M2/THEOREM_M2.md:164:The independent estimate evaluates \(I(Y;S\mid Z)\) in the joint defined by \(\frac12\,\texttt{cal\_mass}(s,z)\,\texttt{cal\_ref}(y\mid s,z)\). The reference uses 65,536 independent calibration mothers per sign, quarter-turn pooling of conditional counts, and a half-count per class. Blind counts in the eight-block cells are substantially sparser. The independent estimates show available signed information at 0.80 rad as well as 1.60 rad. They quantify a property of the physical decoding problem.
research/stage4_math_and_design/M2/THEOREM_M2.md:173:| G4 | −0.02491655 | +0.00143469 | +0.00443245 | +0.00002490 | −0.03080859 | −0.02635124 |
research/stage4_math_and_design/M2/THEOREM_M2.md:176:**G1.** For phase, the five-term path is \(0.92563349+0.05975673+0.00036415+0.00000192+0.45947227=1.44522855\). The zero path is \(0.92563349+0.09630010+0.00048790+0.00000197+0.45845510=1.48087856\). Scalar mismatch accounts for 102.51% of the locked lead along this path. Before calibration or projection, marginal KLs are \(0.00008076\) and \(0.00035846\), respectively. Raw phase and zero empirical-code NIGs are −0.12794 and −0.20533. The S17 direct reference scores 64.62% versus 23.80% and 19.41%. Phase's lower ECE is consistent with the confidence mismatch. It does not measure the share of the NLL lead. Using the argmax set, the empirical resolutions are 0.01162 and 0.02104 nats for phase and zero. Using the exact score vector, both have 0.46066 nats of in-sample resolution because both distinguish the signed codes. Those scores can encode information that the decoded predictions use poorly. With sign probabilities marginalized conditional on code, the NLL cost is 0.007704 for phase and 0.002672 for zero. The difference, 0.005032 nats, is sensitivity to that specific sign-discarding operation. It is distinct from Bayes MI and from the factorial intervention.
research/stage4_math_and_design/M2/THEOREM_M2.md:178:**G2 and G4.** Raw zero accuracy leads by 2.485 and 1.671 pp. Finite \(T^*\) preserves those rankings and a fitted loss lead of 0.02772 and 0.02635 nats remains. The reverse-conditional term gives zero advantages of 0.03104 and 0.03081 nats after projection and fitting. The global prior and scalar terms are small. Raw zero's coarsened argmax resolution also exceeds phase's by 0.01268 and 0.01877 nats. Projection actually reverses the top-1 ordering in those cells while leaving zero with lower probabilistic loss. Thus the locked reversal is explained at the score level by conditional probability allocation and raw decision ordering. The five saved conditions do not establish an architectural transition at \(\pi/2\). Check the angle effect by holding \(|\omega|\), horizons, and diffusion fixed or by crossing them prospectively with common noise and more independent seeds.
research/stage4_math_and_design/M2/THEOREM_M2.md:188:| G2 | phase | 1.335826 / .022970 | 1.334544 / .011354 | .548, 1.008, .769 |
research/stage4_math_and_design/M2/THEOREM_M2.md:189:| G2 | zero | 1.308067 / .023020 | 1.306828 / .017347 | 1.220, .916, .711 |
research/stage4_math_and_design/M2/THEOREM_M2.md:203:| G1 | phase | 23.80 / 1.445229 | 17.05 / 1.411312 | 16.36 / 1.411586 |
research/stage4_math_and_design/M2/THEOREM_M2.md:205:| G2 | phase | 33.88 / 1.335826 | 40.95 / 1.306349 | 40.43 / 1.299371 |
research/stage4_math_and_design/M2/THEOREM_M2.md:206:| G2 | zero | 36.36 / 1.308067 | 38.31 / 1.276082 | 38.51 / 1.273466 |
research/stage4_math_and_design/M2/THEOREM_M2.md:214:On G1 phase the arithmetic pool makes 8.76% of decisions better and 15.51% worse, the exact weighted hit changes here. Geometric makes 8.28% better and 15.72% worse. G1 phase F0 scores 20.88% / 1.411163. On complete orbits G1 phase's arithmetic gain is 0.03386703, while the reverse-barycenter KL is 0.03430077. For zero the corresponding numbers are 0.04120584 and 0.03488442. Restricting to the three saved rotations adds 0.00004913 and 0.00055046 to the loss gain. Across the tested cells pooled maximum equivariance and idempotence residuals were at floating-point precision. G1 raw maximum probability discrepancies were 0.48167 and 0.41957 for phase and zero. Structural compliance of the offline tables coexists with poor decoding and poor ECE.
research/stage4_math_and_design/M2/THEOREM_M2.md:220:G1 seed-mean orientation gains are +0.00981980 and +0.00504776. Their linear dependence on the probabilities makes them equal to the corresponding gains on the seed-average table up to rounding. NLL is nonlinear. Seed-mean swapped penalties are +0.01836983 and +0.00737184. Evaluating the arithmetic seed-ensemble table yields +0.01615141 and +0.00795537. The physical reference has orientation gain 1 by definition. Phase per-seed gains are −0.00147, +0.02104, +0.00989. The zero-arm gains are +0.00184, +0.01295, +0.00035. An orientation gain is a weighted projection coefficient onto the reference odd direction. It does not measure a fraction of the Bayes information or the Stage 3D arm contrast. The direct readout is a separate physical predictor with access to the known condition.
research/stage4_math_and_design/M2/THEOREM_M2.md:222:On all the saved G1 rotations of the phase checkpoint, a text-only flip raises NLL by +0.00323146 and a slot-only flip by +0.01723982, holding the other channel matched. A joint flip raises it by +0.01857416. The difference in differences is −0.00189712. On rotation zero these are +0.00302035, +0.01679462, +0.01836983, and −0.00144514. The interventions show sensitivity to physically aligned slots in the phase checkpoint, alongside a text response, in the tested models. They compare flipped slots to matched slots within a checkpoint. Training with the phase arm and training with the zero arm is a distinct intervention. A shared-development retraining comparison with a locked recalibration and independent seeds is needed to attribute persistent NLL gains to richer slot use.
research/stage4_math_and_design/M2/THEOREM_M2.md:231:| valid | .819207 | 1.388883 / 1.410690 | 1.383282 / 1.384875 | +.020214 | +.021807 |
research/stage4_math_and_design/M2/THEOREM_M2.md:232:| duplicate dedup | .819207 | 1.388883 / 1.410690 | 1.383282 / 1.384875 | +.020214 | +.021807 |
research/stage4_math_and_design/M2/THEOREM_M2.md:236:| fresh resample | .663189 | 1.405640 / 1.406397 | 1.385318 / 1.382645 | +.003429 | +.000757 |
research/stage4_math_and_design/M2/THEOREM_M2.md:252:At the two planning excesses \(d=(0.03,0.04)\), one-sided \(\alpha=0.05/3\) and marginal power 0.95, the synthetic fixed-seed block minima for accuracy/NLL are G1 7/5, G2 11/7, G3 8/6, G4 11/4 and H 9/6. These are conditional t minima for hypothetical positive effects. Observed G2 and G4 effects favor zero and fail the positive comparative target. G3's accuracy-only random-seed minimum is 42 under its zero-truncated seed variance estimate from eight blocks. Three seeds do not give a viable 95% crossed NLL power design in any cell under the plug-in scenarios. The small conditional minima do not override the frozen synthetic budget or planning from independent upper variance bounds.
research/stage4_math_and_design/M2/THEOREM_M2.md:254:G1-proxy sample sizes and powers follow. CTTC uses session means across eight clusters per session. Its ICC parameter is the assumed fraction of the proxy block variance shared at session level, with no additional session-by-seed effect in this sensitivity run. “Minimum” means the first integer yielding at least 95% marginal power under the specified test and effects.
research/stage4_math_and_design/M2/THEOREM_M2.md:273:For random-seed G1 power the corresponding analytic/Monte Carlo pairs are .22385/.22447 and .30028/.30242. The correlated Holm-IUT MC false-claim family rates range from 0.04002 to 0.04953 under the three simulated null configurations. The proof establishes protection beyond those simulations. ECE and the remaining performance gates need their own pilot variability or a joint design simulation.
research/stage4_math_and_design/M2/THEOREM_M2.md:281:- **M4** should keep the 131-block synthetic budget until independent gate-endpoint pilots justify changes. If claiming robustness across retrains, plan around ten or more paired seeds as a G1-based starting point and recalculate using pilot upper variance bounds. With three seeds, scope the main claim to the frozen connectors. Obtain separate Pluto capture and CTTC session variance pilots and freeze session IDs and permutation strata. Preserve all nine CTTC sessions in inference, and use the power objective for the complete gate before locking the real-data sample size.
research/stage4_math_and_design/M2/THEOREM_M2.md:291:The run used Python 3.12.7, NumPy 2.3.5 and SciPy 1.17.0. The default uv cache was read-only under the workspace sandbox. The run used an offline cache under M2 populated from the already available pinned wheels. The verifier reads the Stage 3D artifacts and writes its result adjacent to the script.
```

### Event #9: `item.completed` — `command_execution` (`item_6`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-Content -Encoding utf8 research/stage4_math_and_design/discussions/M5/prompt_r2.md
uv --version
uv --offline python list --only-installed'
```

**Aggregated Output:**

```text
# Track M5 (Round 2): Non-Trivial Multi-Geometry Adaptive EIG, Exact 16-Bit/RefPool Integration & Cross-Host Stage 6 Lock

You are resuming persistent Codex session **Track M5** (`01a2194c-3b55-7153-bc9f-570b6f8c3c3f`) in `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`. Keep edits STRICTLY inside `research/stage4_math_and_design/M5/` and run Python ONLY via `uv run --python 3.12 ...`.

## 1. Cross-Track Findings & Round-1 Critique to Resolve (From M1–M6 & Self-Audit)
1. **Your Own Critical Finding in Round 1 (`fixed_time_4` Tie!)**:
   - In `results_m5.json` Round 1, `max_EIG` crushed `random` (`90.63% vs 71.63%`) and `nearest` (`70.21%`) on selective-fade — **yet time `4` won ALL 2,048 recordings**, meaning a static `fixed_best_on_dev = time 4` rule tied active per-recording `max_EIG`!
   - Wait: **what mathematically forces `max_EIG` to strictly dominate EVERY static `fixed_best_on_dev` window `a_const`?** Obviously whenever **recording-specific old cache `Z` carries conditional information about WHERE the ambiguity lies** (e.g., recording-varying selective deep fades seen via `log_radial(Z)`, OR heterogeneous change-point intervals $\tau$ localized differently per recording by `Z`, OR competing CFO slopes aliased on different candidate subsets per recording)!
2. **Locking the Unified 16-Bit Packet & Causal Real-Data Compatibility with M1, M3, M4**:
   - M4 showed that on real hardware without Stage 3's fixed SNR/amplitude, forcing SNR down to `2 dB` destroys `20 dB` (`0.00% -> 19.89%` SER), while M4's R3 whitened lift `(T, E)` preserves scalar precision cleanly, and M4 proposed **extracting BOTH `4-bit s17` and your `16-bit canonical` packet in Stage 4's zero-GPU CPU shard pass**!
   - Also check M3's `RefSufficientPool`: when revealing ONE candidate window $Y_{a^*}$ in Stage 6, $Y_{a^*}$ enters the same immutable `(acquisition_id, shares_channel, timestamp, K_ref, matched_T)` `RefSufficientPool` contract, handling shared CFO uncertainty $(\hat\omega, v_\omega)$ and correlated diffusion without double-counting!
3. **From M6 & M2 (Stage 6 Single Frozen Query Contract Across 4B / 9B / 27B & LEMMO)**:
   - M6 asked: can you freeze ONE concrete **Stage 6 Ultimate Blind Test Query & Evidence Contract** common to `Qwen3.5-4B`, `Qwen3.5-9B`, `Qwen3.8-27B`, and `LEMMO` under identical pre-question cache bits (`16 bits/window`), candidate read budget (`k = 1` sealed read), pre-reveal forecast commitment scoring, AND M2's paired statistical gate (requiring strict win over `random`, `nearest`, AND **`fixed_best_on_dev`**!)?

## 2. Specific Round-2 Convergence Tasks for Track M5
Update `research/stage4_math_and_design/M5/THEOREM_M5.md`, `verify_m5.py`, and `results_m5.json`:
1. **Prove & Empirically Demonstrate Strict Per-Recording Adaptive EIG Dominance Over `fixed_best_on_dev`**:
   - Prove the exact mutual-information condition $I(a^*(Z); Y_a \text{ utility} \mid \text{marg}) > 0$ under which `max_EIG(Z)` strictly improves expected posterior NLL and accuracy over $\max_{a \in \mathcal{C}} \mathbb{E}_Z[\dots]$ (`fixed_best_on_dev`), i.e., Bayes value of conditional adaptive experiment design!
   - Add a new deterministic benchmark section in `verify_m5.py` (**`multi_geometry_adaptive_suite`** — e.g., varying localized jump intervals + varying candidate-window fading reliability + CFO aliasing across recordings) and verify empirically via paired CIs that **`max_EIG(Z)` strictly beats ALL FOUR controls: `random`, `nearest`, AND `fixed_best_oracle/dev`** on both post-reveal NLL and accuracy!
2. **Harmonize the Exact Bit-Layout of the 16-Bit Packet & `M1/M3/M4` Stage 4→6 Integration Table**:
   - Align byte/bit fields cleanly with M4 (`log_radial`, `resid_quality`, `s17_inner`, `turn`, `canon_angle`, erasure bit when zero/missing), show how Stage 4 CPU extractors populate BOTH `4-bit s17` and `16-bit canonical` views, and lock the Stage 6 `4B / 9B / 27B / LEMMO` frozen benchmark specification!
3. Run updated `verify_m5.py` via `uv run` cleanly and summarize exact numbers (`max_EIG` vs `fixed_best_on_dev`, `random`, `nearest`) in `reply_r2.md`.
uv 0.11.21 (5aa65dd7a 2026-06-11 x86_64-pc-windows-msvc)
cpython-3.13.0-windows-x86_64-none    C:\Users\李则徐\AppData\Local\Programs\Python\Python313\python.exe
cpython-3.12.7-windows-x86_64-none    C:\qazwsx\python.exe
cpython-3.9.19-windows-x86_64-none    C:\376381\miniconda3\python.exe
cpython-3.9.19-windows-x86_64-none    C:\114514\python.exe
```

### Event #10: `item.completed` — `command_execution` (`item_8`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n -i 'PhaseContext|relabel|immutable|16-bit|host-pure|RefPool' research/stage4_math_and_design/discussions/M1/prompt_r2.md
rg -n -i 'LEMMO|compile|same|Tier|gate|p.iut' research/stage4_math_and_design/discussions/M6/prompt_r2.md research/stage4_math_and_design/discussions/M2/prompt_r2.md | Select-Object -First 23"
```

**Aggregated Output:**

```text
1:# Track M1 (Round 2): Cross-Track Convergence — Pure-Host vs Physical-Residual Ablation, 16-Bit/s17 Dual Cache & Real-Data Group Action
9:4. **M5 (Universal Stage 4→6 Cache)**: Proved `4-bit s17` merges amplitude/SNR (`|T|=0.02 vs 0.5833 -> 0.4804 TV`) and loses `0.3119 nats` on Stage 5 joint event+location (`1.2788 -> 0.9669`), AND exhibited two IQ pairs sharing exact `(T, E)` that produce different `s17` inner codes `[2, 0]`. M5 converged on a **16-bit bit-exact canonical extension**: `[2b turn | 2b s17_inner | 6b canon_angle | 4b log_radial | 2b resid_quality]` (recovering `s17` by right-shift while matching `Full IQ` closely in Stage 5–6).
12:   - M4 requires: (a) clean separation between **receive-gauge rotation** ($r_{\text{rx}}$: `Q1` TX diff invariant, `Q2` TX label invariant, `Q3` RX sector rolls $+r_{\text{rx}}$) and **TX relabeling**, (b) CFO uncertainty variance $v_\omega$ (giving edge damping $e^{-n^2 v_\omega d^2 / 2}$ correlated across edges!), and (c) SNR-preserving R3 lift on real Pluto/CTTC frames.
16:1. **The 4-Arm Causal Ablation Architecture (`Host-Pure` vs `Physical+HostResidual`)**:
19:     - **`Arm 1` (`Equivariant-Host-Pure`)**: quotient canonicalization (`canonical_input` + `rotate`) + multiplicative phase-coupled continuous summary fed purely as **soft tokens into the frozen host**, with **NO physical logit bypass** (`q_host` computed strictly from frozen LM / frozen surrogate logits, with isotropic `W_out` geometry)!
21:   - Implement and run **BOTH `Arm 1 (Host-Pure)` AND `Arm 2 (Physical+Residual)`** in `verify_m1.py` on the SAME frozen CPU surrogate host and identical `|\omega| <= 0.25` training split, testing `G1..G4`, equivariance defect (`<= 1e-6`), and M6's rank/margin candidate-Jacobian diagnostic! (Does `Arm 1 Host-Pure` ALSO extrapolate >55% past $\pi/2$ on `G1/G3`, or does frozen random projection struggle without coupled tokens / sufficient width?)
22:2. **Unified `4-bit s17` / `16-bit canonical` Dual Input Contract**:
23:   - Show cleanly how `EquivariantEvidenceConnector` consumes EITHER `4-bit s17` (`turn, s17_inner`) OR M5's `16-bit canonical packet` (`turn, s17_inner, canon_angle, log_radial, resid_quality`) through one unified canonical evidence embedding + transport layer (keeping total trainable parameters well under `<= 2.2M`, matching Stage 3D's Qwen budget).
research/stage4_math_and_design/discussions/M6/prompt_r2.md:1:# Track M6 (Round 2): Final Converged Tier-0/1/2 Ledger, M1/M4 Reconciliation & Master Stage 4–6 Execution DAG
research/stage4_math_and_design/discussions/M6/prompt_r2.md:11:   - **Crucial Synthesis Decision Needed From M6**: Merge M4's VRAM-safe pure `A100-80GB` safety with M6's Tier-2 **1-seed pilot futility gate** (`18%` steps on disjoint dev BEFORE buying full `384`-step × `4`-arm comparison OR continuing `Arm 1 / Arm 2` to extra blocks/seeds!), keeping TOTAL worst-case cap **strictly `<= $9.50`** inside `~$12.88` remaining Modal free credits (`$0.00` net spend guard)!
research/stage4_math_and_design/discussions/M6/prompt_r2.md:14:   - Update your Tier-1 CPU Gate & Stage 4 DAG to explicitly require **BOTH** `Arm 1 (Host-Pure)` AND `Arm 2 (Hybrid)` Jacobian/OOD checks AND counterfactual slot/text controls!
research/stage4_math_and_design/discussions/M6/prompt_r2.md:22:2. **Lock the Exact Numerical Tier-0 / Tier-1 / Tier-2 Go-No-Go Checklists** (ready for Antigravity to inspect and execute immediately after GitHub commit, with exact pass/fail predicates across every node `4A..6E` incorporating M1–M5 invariants!).
research/stage4_math_and_design/discussions/M6/prompt_r2.md:23:3. **Cross-Host & Fair `LEMMO` Execution Law (Stage 6 Readiness)**: Explicitly verify mathematically how fixing the `16-bit canonical` cache + `RefSufficientPool` lets us swap only tiny host adapters (`4B -> 9B -> 27B`) and run exact matched-support `LEMMO` comparison with minimum redundant experiments!
research/stage4_math_and_design/discussions/M2/prompt_r2.md:1:# Track M2 (Round 2): Reconciling Power vs Modal Free-Credit Scope, Causal Arm Gate & Multi-Claim Hierarchy
research/stage4_math_and_design/discussions/M2/prompt_r2.md:10:2. **Causal Isolation Gate for M1's `Host-Pure` vs `Physical-Residual` Arms + M6 DPI**:
research/stage4_math_and_design/discussions/M2/prompt_r2.md:15:     - Pre-registered calibration/temp-fitting scope: separate **dev-locked zero-shot calibration** vs **oracle retrospective sensitivity**, strictly preventing blind recalibration from ever passing a gate!
research/stage4_math_and_design/discussions/M2/prompt_r2.md:19:4. **M5's Stage 6 Gate Compatibility**:
research/stage4_math_and_design/discussions/M2/prompt_r2.md:20:   - Include a crisp specification for Stage 6's `max_EIG` vs `{random, nearest, fixed_best_on_dev}` gate (plus `4-bit s17` vs `16-bit canonical` comparison)!
research/stage4_math_and_design/discussions/M2/prompt_r2.md:24:1. Formalize the **Scoped Three-Tier Claim Ladder** (Conditional Frozen-Connector Claim `S=1..3` vs Multi-Seed Retrain Claim `S>=10`) + prove conditional validity after M6's Stage-1 dev futility screen + compare power curves for Stage 4's exact `1`-seed and `3`-seed Modal plans on synthetic `32` vs `131` blocks, Pluto `16` blocks, and CTTC `9` captures / `72` clusters.
research/stage4_math_and_design/discussions/M2/prompt_r2.md:25:2. Lock the **Exact Stage 4 Formal Preregistered Statistical Gate Table** unifying M1's `Arm 0/1/2` contrasts, M3's `Q2 two-valid > one-valid > none` & exact `0.0` invariance assertions, M4's Pluto/CTTC real-data hierarchy, and M5's `fixed_best_on_dev` Stage-6 gate.
```

### Event #11: `item.completed` — `command_execution` (`item_7`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding utf8 research/stage4_math_and_design/discussions/M3/prompt_r2.md | Select-Object -First 48
rg -n '"'^def |''^class |K_ref|acquisition_id|already'"' research/stage4_math_and_design/M3/verify_m3.py | Select-Object -First 45"
```

**Aggregated Output:**

```text
# Track M3 (Round 2): Correlated Multi-Clock Pilots, Non-Double-Counting Innovations & Stage 6 Active Mixture Interface

You are resuming persistent Codex session **Track M3** (`01a21948-9063-7e31-91b9-46869eeabbdc`) in `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`. Keep edits STRICTLY inside `research/stage4_math_and_design/M3/` and run Python ONLY via `uv run --python 3.12 ...`.

## 1. Cross-Track Demands on `RefSufficientPool` (From M1, M2, M4, M5, M6)
1. **From M4 (NO DOUBLE-COUNTING & Correlated Hardware Noise/State)**:
   - On Pluto (`15, 18` of Copy 1; sync `0:13` in baseline state $\mathcal{F}^0$) and CTTC (`126, 127` revealed pilots; `0:6` sync in $\mathcal{F}^0$), causal sync and earlier split history have ALREADY formed a causal prior state `(H_0, d_0, omega_hat, P_0, sigma_phi)`!
   - Furthermore, real hardware pilots at nearby lags (e.g., CTTC `126, 127` adjacent symbols through `256`-tap RRC pulse / residual ISI / shared CFO error $\epsilon_\omega$) can share **estimation uncertainty or noise covariance** $K_{\text{ref}}$ AND may use known QPSK pilot roots $d_m \neq 1$!
   - **Crucial Warning from M4**: `RefSufficientPool` MUST NOT double-count observations already conditioned inside M4's $\mathcal{F}^0$ causal state! When adding `(R_1, R_2)` onto $\mathcal{F}^0$, `RefSufficientPool` must emit the **conditional likelihood / innovation factor** given the prior state (and explicitly track which `acquisition_id`s already contributed to $\mathcal{F}^0$!).
2. **From M1 (Unified Likelihood Interface + Shared CFO Uncertainty)**:
   - M1 asked: can `RefSufficientPool` accept per-pilot timestamps (including **$t_p > t_3$**!), known pilot symbols $d_m$, shared CFO uncertainty $(\hat\omega, v_\omega)$ (remember CFO rate error scales phase variance as $v_\omega (t_j - t_i)^2$, creating cross-clock quadratic coupling beyond Brownian diffusion $\sigma_\phi^2 |t_j - t_i|$!), AND return joint factors `canonical_q2[4]`, `canonical_q3[4]`, and optional `joint[4,4]` relative to receiver frame $a_1$ (while keeping exact zero-influence distractors and byte-identical fixed-position prompt tokens)?
3. **From M5 (Stage 6 Ultimate Blind Test EIG & Pre-Reveal Predictive Mixtures)**:
   - Before revealing candidate window $a$, M5 computes $f_{ha}(x) = \int p(Y_a = x \mid u, h, z) dP(u \mid z, h)$ and $\text{EIG}(a \mid z) = I(H; Y_a \mid z)$, THEN reveals ONE sealed acquisition $(id_a, t_a, y_a)$ into `RefSufficientPool`! Notice if `RefSufficientPool` represents each candidate hypothesis / phase-node state as clean `(phase_grid, log_weights_by_turn[4, N_nodes], clock_meta)`, then BOTH M5's pre-reveal $\text{EIG}$ integral AND post-reveal Bayes update share ONE exact mathematical engine!
4. **From M2 & M6 (Idempotence, Byte-Exact Prefix Invariance & Calibrated Comparison)**:
   - Require exact verification that:
     - Raw duplicate IDs, masked `shares_channel=False` distractors, padded empty slots, and permuted input ordering produce **literally byte-identical** fixed-slot tensors / prefix embeddings and `0.0` KL/TV shift;
     - Contrast-check native host logits (`Arm 1`) vs physical+residual (`Arm 2`) vs `q2_reader` posterior under dev-locked calibration.

## 2. Specific Round-2 Convergence Tasks for Track M3
Update `research/stage4_math_and_design/M3/THEOREM_M3.md`, `verify_m3.py`, and `results_m3.json`:
1. **Extend Theorem 2 (Joint Brownian + Latent CFO Uncertainty + Correlated Pilot Noise + Anti-Double-Counting Theorem)**:
   - Derive the exact innovation factorization $p(\mathcal{R} \setminus \mathcal{F}^0 \mid \mathcal{F}^0, G, \boldsymbol{\phi}, \omega)$ when causal history $\mathcal{F}^0$ provides prior $(\hat\omega_0, v_\omega, P_{\phi,0})$ and new pilots have covariance $K_{\text{ref}}$: prove the **exact Schur-complement innovation sufficient statistic** and prove strictly what happens if an observation already in $\mathcal{F}^0$ is accidentally re-counted (`variance underestimation / overconfident false calibration KL`!).
   - Derive the augmented Gaussian covariance $(B_{t_j} + \tilde\omega t_j)$ combining Brownian increments $\sigma_\phi^2 \min(t_j, t_k)$ AND shared linear CFO prior variance $v_\omega t_j t_k$ (note rank-1 rank addition cleanly solved by Sherman-Morrison or appending 1 Gaussian-Hermite quadrature node over $\tilde\omega$!).
2. **Freeze the Unified Multi-Stage `RefSufficientPool` Schema** serving **M1 (`EquivariantEvidenceConnector`) + M4 (Pluto `15,18`, CTTC `126,127`, duplicate/missing provenance) + M5 (Stage 6 pre-reveal mixture & EIG)** simultaneously!
3. Add deterministic CPU unit/property verifications inside `verify_m3.py` (keep total runtime concise `< 60s` or reuse cached fast quadrature; verify: (i) zero double-counting invariant against prior state, (ii) non-trivial shared CFO + Brownian multi-clock gain vs naive sum, (iii) byte-exact `uint8`/`float32` representation identity across shuffled/duplicate/distractor/padded `RefPool` inputs, and (iv) Stage 6 pre-reveal predictive marginal + post-reveal Bayes update consistency!). Run via `uv run` and report in `reply_r2.md`.
42:def rot90(iq, turn):
49:def as_iq(values):
54:class Pilot:
55:    acquisition_id: int
64:class Pooled:
73:class RefSufficientPool(nn.Module):
81:        for ref in sorted(refs, key=lambda x: (x.acquisition_id, x.t_p)):
84:            prev = by_id.get(ref.acquisition_id)
86:                by_id[ref.acquisition_id] = ref
92:                raise ValueError(f"Conflicting metadata for acquisition_id={ref.acquisition_id}")
94:                raise ValueError(f"Conflicting samples for acquisition_id={ref.acquisition_id}")
97:        for acquisition_id in sorted(by_id):
98:            ref = by_id[acquisition_id]
126:class FixedSlotQ2(nn.Module):
152:def metrics(probs, labels):
157:def compare(left, right, labels, block_size=64):
172:def make_arms(m1, m2, mi):
181:def audit_archive(bundle, kernel, summary, smoke):
191:        arrays = [(r, source_samples[r.acquisition_id] if r.shares_channel else mi) for r in refs]
193:            (r.acquisition_id, r.shares_channel, vals[i]) for r, vals in arrays
237:def fit_apparent(encoder):
253:def check_pool(pool, reader, blind, pinned):
294:    many = tuple(replace(base, acquisition_id=i, t_p=float(i), variance=(i+1)/RHO)
306:    hetero = replace(base, acquisition_id=1, variance=3/RHO, matched_amplitude=0.75)
332:def bridge_weights(c1, c2, times, sigma, z2_phase_emission, inner_order=20):
342:def multi_post(s, families, a, times, sigma, z2_phase_emission, order):
375:def multi_draws(encoder, times, n, seed):
385:def multi_evidence(pool, encoder, draws, sigma, times):
400:def multi_test(encoder, a, marginal):
491:def main():
```

### Event #12: `item.completed` — `command_execution` (`item_9`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-Content research/stage4_math_and_design/M5/verify_m5.py | Select-Object -First 165
Get-Content research/stage4_math_and_design/M5/verify_m5.py | Select-Object -Skip 285 -First 74
Get-Content research/stage4_math_and_design/M5/verify_m5.py | Select-Object -Last 70'
```

**Aggregated Output:**

```text
"""Seeded CPU replay for the M5 cache and one-window blind test."""
import hashlib
from itertools import permutations
import json
from math import gcd
import os
from pathlib import Path
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import scipy
from scipy.special import gammainc, logsumexp, xlogy


HERE = Path(__file__).resolve().parent
MATCHED = HERE.parents[2] / "research" / "matched_cache"
sys.path.insert(0, str(MATCHED))
from learned import forward, input_features, load


SEED = 50261126
N_TEST = 4096
N_BLIND = 2048
N_CAL = 32768
PHASES = 128
QUARTER = PHASES // 4
UNIT = np.exp(2j * np.pi * np.arange(PHASES) / PHASES)
OLD = np.array([0, 2, 6, 10])
CANDIDATE = np.array([11, 1, 4, 8])
TAU = np.array([2.5, 4.5, 7.5])
BRANCH_NAMES = ("smooth", "jump_2.5", "jump_4.5", "jump_7.5", "independent", "burst")
LAYOUTS = {"2": (0, 0, 0), "8": (2, 1, 1), "12": (4, 3, 1), "16": (6, 4, 2), "24": (8, 8, 4)}
METHODS = ("2", "4_s17", "8", "12", "16", "canonical_continuous", "full_iq")
EXPECTED_SHA = "e67d9fbafb90791ab19549de7789798d818286d7f52d7ac566cc39df71e78356"


PROFILES = (
    ("flat_minus3", [-3] * 4, [1] * 4, [-3, 0, 2, -1], [1] * 4, -3),
    ("flat_plus2", [2] * 4, [1] * 4, [2, 2, 2, 2], [1] * 4, 2),
    ("flat_plus24", [24] * 4, [1] * 4, [5, 5, 5, 5], [1] * 4, 12),
    ("selective_fade", [14, -3, 3, 24], [1, .23, .42, 1.20], [7, 1, 7, 3], [1, .4, .85, .6], 0),
)


def normal(rng, shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def parameters(snr, amplitude):
    amplitude = np.array(amplitude, dtype=float)
    return amplitude, amplitude**2 / 10**(np.array(snr) / 10)


def phase_index(theta):
    return np.rint(theta * PHASES / (2 * np.pi)).astype(int) % PHASES


def make_tracks():
    slopes = (.18, .34)
    jump = 2.2
    tracks = []
    for omega in slopes:
        tracks.append((0, omega * OLD, omega * CANDIDATE, 1., -1, 1 / 8))
        for label, tau in enumerate(TAU, 1):
            old = (omega - jump / 10) * OLD + jump * (OLD >= tau)
            new = (omega - jump / 10) * CANDIDATE + jump * (CANDIDATE >= tau)
            tracks.append((label, old, new, np.where(CANDIDATE >= tau, .78, 1.), -1, 1 / 24))
        for offset in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
            for omega_second in slopes:
                old = np.where(OLD < 5, omega * OLD, offset + omega_second * (OLD - 5))
                new = np.where(CANDIDATE < 5, omega * CANDIDATE, offset + omega_second * (CANDIDATE - 5))
                tracks.append((4, old, new, 1., -1, 1 / 64))
        for burst in (1, 2):
            tracks.append((5, omega * OLD, omega * CANDIDATE, 1., burst, 1 / 16))

    branch, old_phase, new_phase, gain, burst, prior = [], [], [], [], [], []
    for label, old, new, ratio, burst_window, weight in tracks:
        for g in range(4):
            branch.append(label)
            old_phase.append((phase_index(old) + QUARTER * g) % PHASES)
            new_phase.append((phase_index(new) + QUARTER * g) % PHASES)
            gain.append(np.broadcast_to(ratio, CANDIDATE.shape))
            burst.append(np.where(np.arange(len(OLD)) == burst_window, 6., 1.))
            prior.append(weight / 4)
    prior = np.array(prior)
    np.testing.assert_allclose(prior.sum(), 1.)
    return tuple(map(np.array, (branch, old_phase, new_phase, gain, burst, prior)))


def sample_old(rng, count, prior, phase, burst, amplitude, variance, ref_db):
    state = rng.choice(len(prior), count, p=prior)
    symbols = rng.integers(0, 4, (count, 2))
    shifts = np.zeros((count, len(OLD)), dtype=int)
    shifts[:, 1:3] = QUARTER * symbols
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
    local_angle = (np.angle(t * (1j)**(-turns)) + np.pi / 4) / (np.pi / 2)
    packets = {"4_s17": original}
    bins = {}
    for method, (angle_bits, radial_bits, quality_bits) in LAYOUTS.items():
        angle = np.clip(np.floor(local_angle * 2**angle_bits).astype(int), 0, 2**angle_bits - 1)
        radius = np.searchsorted(radial_edges(radial_bits)[1:-1], abs(t) / np.sqrt(variance))
        quality = np.searchsorted(quality_edges(quality_bits)[1:-1], residual / variance)
        bins[method] = (turns * 2**angle_bits + angle, radius, quality)
        extra = radial_bits + quality_bits
        tail = (angle << extra) | (radius << quality_bits) | quality
        packets[method] = turns if method == "2" else (original << (angle_bits + extra)) | tail
        if method != "2":
            np.testing.assert_array_equal(packets[method] >> (angle_bits + extra), original)
        assert np.all((packets[method] >= 0) & (packets[method] < 2**int(method)))
        prefix = packets[method] if method == "2" else packets[method] >> (angle_bits + extra + 2)
        unpacked = (
            (prefix << angle_bits) | ((packets[method] >> extra) & (2**angle_bits - 1)),
            (packets[method] >> quality_bits) & (2**radial_bits - 1),
            packets[method] & (2**quality_bits - 1),
        )
        np.testing.assert_array_equal(unpacked, bins[method])
        bins[method] = unpacked
    return packets, bins


def s17_emissions(encoder, rng, amplitude, variance):
    """Independent Monte Carlo cell probabilities, with exact C4 transfer."""
    base = np.empty((QUARTER, 16))
    for phase in range(QUARTER):
        y = amplitude * UNIT[phase] / np.sqrt(3) + np.sqrt(variance) * normal(rng, (N_CAL, 3))
        counts = np.bincount(encoder.encode(y), minlength=16)
        base[phase] = (counts + .5) / (N_CAL + 8)
    return np.concatenate([np.roll(base, 4 * g, axis=1) for g in range(4)])


def integrate_eig(weights, means, variance, step):
    # Integrate on a Cartesian midpoint grid in noise-standardized coordinates.
    means = [mu / np.sqrt(variance) for mu in means]
    extent = max(np.max(abs(mu)) for mu in means) + 6.5
    count = int(np.ceil(2 * extent / step))
    step = 2 * extent / count
    axis = -extent + (np.arange(count) + .5) * step
    z = (axis[:, None] + 1j * axis).ravel()
    densities = [np.exp(density(z, mu, 1.)).T for mu in means]
    pi = [weight.sum(axis=1) for weight in weights]
    eig = np.zeros(len(pi[0]))
    for start in range(0, len(eig), 128):
        chunk = slice(start, start + 128)
        joint = [weight[chunk] @ grid for weight, grid in zip(weights, densities)]
        mix = joint[0] + joint[1]
        integrand = sum(xlogy(p, p) - xlogy(p, mix) - xlogy(p, prior[chunk, None])
                        for p, prior in zip(joint, pi))
        eig[chunk] = integrand.sum(axis=1) * step**2
    return eig


def digest_arrays(*arrays):
    digest = hashlib.sha256()
    for data in arrays:
        digest.update(np.ascontiguousarray(data).tobytes())
    return digest.hexdigest()


def pick_before_reveal(states, groups, a, v, actions):
    forecasts = []
    eig = []
    checks = []
    for k, by_class in enumerate(groups):
        joint = [states[:, mask] @ indicator for mask, indicator, _ in by_class]
        means = [a[k] * mean for _, _, mean in by_class]
        forecasts.append((joint, means))
        eig.append(integrate_eig(joint, means, v[k], .40))
        count = 64
        fine = integrate_eig([w[:count] for w in joint], means, v[k], .25)
        checks.append(float(np.max(abs(fine - eig[-1][:count]))))
    eig = np.column_stack(eig)
    assert np.isfinite(eig).all() and eig.min() > -1e-9
    np.testing.assert_array_less(eig, np.log(2) + 1e-8)
    selected = {**actions, "max_EIG_window": eig.argmax(axis=1)}
    rows = np.arange(len(states))
    assert np.all(eig[rows, selected["max_EIG_window"]] >= eig[rows, selected["nearest_window"]])
    receipt = digest_arrays(*selected.values(), v, *(arr for joint, means in forecasts for arr in (*joint, *means)))
    return forecasts, eig, selected, receipt, max(checks)


class SealedCandidates:
    """Record once and release only one selected observation per mother and policy."""

    def __init__(self, iq):
        self._iq = iq
        self._used = set()
        self._committed = {}

    def commit(self, policy, receipt, selected):
        assert receipt and not self._used and policy not in self._committed
        assert selected.shape == (len(self._iq),)
        assert np.all((selected >= 0) & (selected < self._iq.shape[1]))
        self._committed[policy] = (receipt, selected.copy())

    def reveal(self, policy, receipt):
        assert policy not in self._used
        expected, selected = self._committed[policy]
        assert receipt == expected
        self._used.add(policy)
        return self._iq[np.arange(len(selected)), selected]


def score_revealed(forecasts, eig, actions, receipt, candidates, method, v, truth, ambiguous):
    results, rows_by_policy = {}, {}
                            "residual_energy": 2 * delta**2,
                            "codes": encoder.encode(pair).tolist(),
                            "iq_re_im": np.stack((pair.real, pair.imag), axis=-1).tolist()}
    raise AssertionError("Expected the legacy inner index to use more than matched sum and residual energy")


def main():
    checkpoint = MATCHED / "weights" / "equivariant_b8_s17.npz"
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert digest == EXPECTED_SHA
    encoder = load(checkpoint)
    assert encoder.equivariant and len(encoder.centers) == 4
    witness = collision_witness(encoder)
    inner_witness = legacy_inner_witness(encoder)
    tracks = make_tracks()
    all_profiles, all_losses = [], []
    for key, profile in enumerate(PROFILES):
        profile_result, losses = run_profile(key, profile, tracks, encoder)
        all_profiles.append(profile_result)
        all_losses.append(losses)
    tasks = tuple(all_losses[0]["full_iq"])
    pooled = {method: {task: np.concatenate([p[method][task] for p in all_losses]) for task in tasks}
              for method in METHODS}
    comparison = {method: {task: paired_ci(pooled[method][task] - pooled["full_iq"][task]) for task in tasks}
                  for method in METHODS}
    table = {method: {task: float(pooled[method][task].mean()) for task in tasks} for method in METHODS}
    pareto_tasks = ("Q1", "Q2", "stage5_event_and_location", "stage6_post_max_eig")
    rates = np.array([2, 4, 8, 12, 16, np.inf, np.inf])
    objectives = np.array([[table[m][q] for q in pareto_tasks] for m in METHODS])
    finite = rates < np.inf
    nondominated = [
        method for i, method in enumerate(METHODS) if finite[i] and not np.any(
            (rates <= rates[i]) & np.all(objectives <= objectives[i], axis=1)
            & ((rates < rates[i]) | np.any(objectives < objectives[i], axis=1))
        )
    ]
    result = {
        "protocol": {
            "seed": SEED, "python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__,
            "mothers_per_profile_old": N_TEST, "mothers_per_profile_blind": N_BLIND,
            "s17_calibration_samples_per_canonical_phase_per_distinct_gain_variance_within_profile": N_CAL,
            "s17_checkpoint_sha256": digest, "sample_phase_grid": PHASES,
            "old_times": OLD.tolist(), "payload_windows": OLD[1:3].tolist(),
            "candidate_times": CANDIDATE.tolist(), "jump_times": TAU.tolist(),
            "event_location_labels": BRANCH_NAMES, "phase_slope_grid": [.18, .34],
            "cache_extensions_angle_radius_quality_bits": LAYOUTS,
            "compact_reader": "integrated physical cells; packed legacy inner bits available but ignored",
            "full_iq": "distinct raw likelihood verified against canonical (T,residual) factorization",
            "cache_and_candidate_visibility": "all candidate decisions and predictive densities committed before first reveal",
            "common_ambiguity_subset": "old Full IQ smooth posterior in [0.1, 0.9]; evaluation only",
        },
        "s17_radial_collision_witness": witness,
        "s17_inner_not_determined_by_matched_sum_and_residual_energy": inner_witness,
        "nll_pareto_table": table,
        "finite_pointwise_nondominated_including_rate": nondominated,
        "paired_excess_nll_vs_full_iq": comparison,
        "paired_nll_16_minus_s17": {task: paired_ci(pooled["16"][task] - pooled["4_s17"][task]) for task in tasks},
        "paired_profile_nll_16_minus_s17": {
            p["profile"]: {task: paired_ci(loss["16"][task] - loss["4_s17"][task]) for task in tasks}
            for p, loss in zip(all_profiles, all_losses)
        },
        "per_profile": all_profiles,
    }
    output = HERE / "results_m5.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {output.name}", flush=True)


if __name__ == "__main__":
    main()
```

### Event #13: `item.completed` — `agent_message` (`item_10`)

The exact condition is that no single window is optimal almost surely. Mutual information captures that when the optimal window is unique, but ties need a condition based on the full set of optimal windows. I’ll keep M4’s field widths and reserve one radial code for erasure so the packet stays at 16 bits.

### Event #14: `item.completed` — `file_change` (`item_11`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\verify_m5.py",
    "kind": "update"
  }
]
```

### Event #15: `item.completed` — `file_change` (`item_12`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\verify_m5.py",
    "kind": "update"
  }
]
```

### Event #16: `item.completed` — `file_change` (`item_13`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\verify_m5.py",
    "kind": "update"
  }
]
```

### Event #17: `item.completed` — `web_search` (`exec-7c3467d5-75c3-4f70-a8c4-ddbd60400944`)

```json
{
  "id": "exec-7c3467d5-75c3-4f70-a8c4-ddbd60400944",
  "type": "web_search",
  "query": "https://huggingface.co/Qwen/Qwen3.5-4B",
  "action": {
    "type": "open_page",
    "url": "https://huggingface.co/Qwen/Qwen3.5-4B"
  },
  "results": [
    {
      "type": "text_result",
      "domain": "huggingface.co",
      "ref_id": "turn3view0",
      "snippet": "Total lines: 880",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/QGt8PX3wDu7VEvlkrXQpEDBvfvza8D-wqf2Z-bCsWou3TovMSHM9S0SJZ7cWkRxPaq31JZanFYz6SQU3R82fHk7j2wzJv1x9_Dd8FkaozzPf-EO2r7dBPaZfwF8jyMIgRZ2zXf9_TCQftqCoFtaZYw",
      "title": "Qwen/Qwen3.5-4B · Hugging Face",
      "url": "https://huggingface.co/Qwen/Qwen3.5-4B"
    },
    {
      "type": "text_result",
      "domain": "huggingface.co",
      "ref_id": "turn3view1",
      "snippet": "Total lines: 893",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/98MXnU2oFJ-65v5NV6fvxGb0B6Zp45-zCkj2CH6swlqDiOP6HuXEdQvJ4M-6OeWhJNMiinihS-9GFXoupSxW6Na1oYHJ9clfmZQh_WFERKQ_k0KKqL1tzV6YdfSffeR-KKzsJLP6TxQI6rGvzG-bdA",
      "title": "Qwen/Qwen3.5-9B · Hugging Face",
      "url": "https://huggingface.co/Qwen/Qwen3.5-9B"
    },
    {
      "type": "text_result",
      "domain": "huggingface.co",
      "ref_id": "turn3view2",
      "snippet": "Total lines: 735",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/86eSfqszK4drcF7DCv2TZdOkca6Cl8nyUwxN9WfF7fJcbOR7hu1mL8aaDIsdd9YQCZi3MIYp6OWT5S-_dzPoMxvPXMSc0-Kqcbf3zFX4XJ_yGKoHrlb3F_CYXmOAINjFTWkOwyakiIz-MU_3b9ggVQ",
      "title": "Qwen/Qwen3.8-27B · Hugging Face",
      "url": "https://huggingface.co/Qwen/Qwen3.8-27B"
    },
    {
      "type": "text_result",
      "domain": "arxiv.org",
      "ref_id": "turn3view3",
      "snippet": "Total lines: 551",
      "title": "LEMMO: A Large Electromagnetic Model leveraging Pretrained LLMs for Million-Sample Signal Understanding",
      "url": "https://arxiv.org/html/2611.09476v1"
    }
  ]
}
```

### Event #18: `item.completed` — `web_search` (`exec-ba219c92-96de-4f05-bc2f-d2efc21ee17b`)

```json
{
  "id": "exec-ba219c92-96de-4f05-bc2f-d2efc21ee17b",
  "type": "web_search",
  "query": "https://huggingface.co/api/models/Qwen/Qwen3.5-9B",
  "action": {
    "type": "open_page",
    "url": "https://huggingface.co/api/models/Qwen/Qwen3.5-9B"
  },
  "results": [
    {
      "type": "text_result",
      "domain": "huggingface.co",
      "ref_id": "turn4view0",
      "snippet": "Total lines: 1",
      "url": "https://huggingface.co/api/models/Qwen/Qwen3.5-9B"
    },
    {
      "type": "text_result",
      "domain": "huggingface.co",
      "ref_id": "turn4view1",
      "snippet": "Total lines: 3",
      "url": "https://huggingface.co/api/models/Qwen/Qwen3.8-27B"
    },
    {
      "type": "text_result",
      "domain": "huggingface.co",
      "ref_id": "turn4view2",
      "snippet": "Total lines: 314",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/uo9mjRy5CFbCiEc1aduOeuUEXlVdxiWZuTg67Z6yD8pRm7Nwe8fAhpAe3PqF_3QL0WUBuklqG4rqmJ-kw_HNa3e0qcv0TY8Am3WkgyjUg9ufQ7o6Ez_VIX09-Lw-PYCdXoqPaqZLa-c-ImW6-cPKfg",
      "title": "Qwen/Qwen3.5-4B at 851bf6e",
      "url": "https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    },
    {
      "type": "text_result",
      "domain": "github.com",
      "ref_id": "turn4view3",
      "snippet": "Total lines: 281",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/MDaL6QZz6iGOShmODGQyZIa91o5pILpQtWjtukkyXw6xi9rrVqtYrZzaQh5NONu5UzbydMj769RGqu59APGQUVjJuZMXjf1x4VhycDFNV138xuDqHY1oJBOiZhlDFDew5fG5GKOfiWmAyajHpNBXNrgfNKFMF1Kez5vZ0HQufKaQgs4PjY6VHfiBkV9hkrz9",
      "title": "GitHub - N1rv4n4/LEMMO: Implementation of LEMMO, an electromagnetic large model for raw I/Q signal recognition and description · GitHub",
      "url": "https://github.com/N1rv4n4/LEMMO"
    }
  ]
}
```

### Event #19: `item.completed` — `web_search` (`exec-2901277b-b07f-4e55-95de-f34697792513`)

```json
{
  "id": "exec-2901277b-b07f-4e55-95de-f34697792513",
  "type": "web_search",
  "query": "'\"sha\"'",
  "action": {
    "type": "find_in_page",
    "pattern": "\"sha\""
  },
  "results": [
    {
      "type": "text_result",
      "domain": "huggingface.co",
      "ref_id": "turn5view0",
      "snippet": "Total lines: 1",
      "url": "https://huggingface.co/api/models/Qwen/Qwen3.5-9B"
    },
    {
      "type": "text_result",
      "domain": "huggingface.co",
      "ref_id": "turn5view1",
      "snippet": "Total lines: 3",
      "url": "https://huggingface.co/api/models/Qwen/Qwen3.8-27B"
    },
    {
      "type": "text_result",
      "domain": "github.com",
      "ref_id": "turn5view2",
      "snippet": "Total lines: 281",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/GdpXvA_z14YaIeYPSjTd2d1jEsljQusCVQ2G5-TEKMyryckj9MZKx-bOpOPx7ZntVGsDcVaww-1ZndEn9_1Np6qpcq4sg9OY1tPqmUshPUgNYszoSh21LIbdsXbLNhjMh4hO6mpRErHk5cP7y2Fvmkbtr3sRW_JeObXtgSsbjXkamvlR7mNSyGnQdX77EYgx",
      "title": "GitHub - N1rv4n4/LEMMO: Implementation of LEMMO, an electromagnetic large model for raw I/Q signal recognition and description · GitHub",
      "url": "https://github.com/N1rv4n4/LEMMO"
    }
  ]
}
```

### Event #20: `item.completed` — `web_search` (`exec-41fa90ba-1b74-49ab-8604-0073dcc1eb41`)

```json
{
  "id": "exec-41fa90ba-1b74-49ab-8604-0073dcc1eb41",
  "type": "web_search",
  "query": "https://huggingface.co/Qwen/Qwen3.5-9B/commits/main",
  "action": {
    "type": "open_page",
    "url": "https://huggingface.co/Qwen/Qwen3.5-9B/commits/main"
  },
  "results": [
    {
      "type": "text_result",
      "domain": "huggingface.co",
      "ref_id": "turn6view0",
      "snippet": "Total lines: 334",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/zogcalGFUNpzkGvQO31qhCSw7Vj7VgRpW9apcZ5HqgddF1kv-DYubWSoibSl_GkrM44Kv5AzB-QRESGZTMNIrwbvR2km0MU0rED4nZyOoGKvkDr7YyTOfa3qEapFls5pt8Qbjzerux146Tiq04-n2g",
      "title": "Commit history · Qwen/Qwen3.5-9B at main",
      "url": "https://huggingface.co/Qwen/Qwen3.5-9B/commits/main"
    },
    {
      "type": "text_result",
      "domain": "huggingface.co",
      "ref_id": "turn6view1",
      "snippet": "Total lines: 424",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/IqdLh7DwZk9cANNM6tO3PEB43xDmS7GLc0A90Ko1nT6uwfHbg3uDJBTv_CKkmLFU9jFNUuWmbaveIfgPaj3dhmPfJy0sP14RGefCw3MM1d81EU4r-7cNs_OF4JCldQw3Lm7Q2q8kVuWqxVjVeS15cw",
      "title": "Commit history · Qwen/Qwen3.8-27B at main",
      "url": "https://huggingface.co/Qwen/Qwen3.8-27B/commits/main"
    },
    {
      "type": "text_result",
      "domain": "github.com",
      "ref_id": "turn6view2",
      "snippet": "Total lines: 281",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/oXFp54EG8EVky0ITPn5Q4KwmInPfm1Yl8RpcD66NVNiefy36CdsKqQNbtHFHKU9DRXfaX-gghi_MTJPRQVifkpGtk2wyK6CZJd7ZJVLXq2I25aIsE6h_8tIwk8ziKKnwzSgoUtyFMt-N7bSkYqghATIzc8kdO_ftkZ8zBRKcKHK7w58pV9ewnr8Al2q-NFfK",
      "title": "GitHub - N1rv4n4/LEMMO: Implementation of LEMMO, an electromagnetic large model for raw I/Q signal recognition and description · GitHub",
      "url": "https://github.com/N1rv4n4/LEMMO"
    }
  ]
}
```

### Event #21: `item.completed` — `web_search` (`exec-5f94bc4c-3b74-4af0-9387-3297109459d6`)

```json
{
  "id": "exec-5f94bc4c-3b74-4af0-9387-3297109459d6",
  "type": "web_search",
  "query": "https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236",
  "action": {
    "type": "open_page",
    "url": "https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236"
  },
  "results": [
    {
      "type": "text_result",
      "domain": "huggingface.co",
      "ref_id": "turn7view0",
      "snippet": "Total lines: 324",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/ZGA8UuoWYs8zh5jesueyH6REDl359FtlULzt87QMONwuCMSTphUYTXR5e8Itf-3wgoLzS2Vhv_ebhNE1DIl0olstHGUEbhdPdcKmUT-cFuOa7Ojr0R4sSqnT8geYQ-_P5lnqcMKGiZ1H7NWG-UdHNQ",
      "title": "Qwen/Qwen3.5-9B at c202236",
      "url": "https://huggingface.co/Qwen/Qwen3.5-9B/tree/c202236"
    },
    {
      "type": "text_result",
      "ref_id": "turn7view1",
      "title": "Internal Error"
    },
    {
      "type": "text_result",
      "domain": "github.com",
      "ref_id": "turn7view2",
      "snippet": "Total lines: 281",
      "thumbnail_url": "https://images.openai.com/static-rsc-1/LsxLJcEy-MkLS6UHDBgFzVZx8b9Up5hu-n2LR5lBx8b8UPZUQ1GTxPaG3DAKUSrceItRgYwtDaK9mJEzzO_HJTS08mF9395BjUSdqZbpiKgD7jvsF55TU10Hhs67Xl_qwZ3DX_7XWLTVkhpAXVueXWTn5t5upuTJlDjl9NAjTwLwqmng0iJhbnEq31Ch3oBE",
      "title": "GitHub - N1rv4n4/LEMMO: Implementation of LEMMO, an electromagnetic large model for raw I/Q signal recognition and description · GitHub",
      "url": "https://github.com/N1rv4n4/LEMMO"
    }
  ]
}
```

### Event #22: `item.completed` — `command_execution` (`item_20`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$ErrorActionPreference='"'Stop'
foreach ("'$id in @('"'Qwen3.5-9B','Qwen3.8-27B')) { "'$remote=Invoke-RestMethod -Uri "https://huggingface.co/api/models/Qwen/$id" -TimeoutSec 12; Write-Output "$id $($remote.sha)" }
$remote=Invoke-RestMethod -Uri '"'https://api.github.com/repos/N1rv4n4/LEMMO/commits/main' -TimeoutSec 12; Write-Output \"LEMMO "'$($remote.sha)"'
```

**Aggregated Output:**

```text
Qwen3.5-9B c202236235762e1c871ad0ccb60c8ee5ba337b9a
Qwen3.8-27B 37561816e66b5b9153b1b64c10968c18c1e6f4a0
LEMMO ebcc1ac9a6d3a658e62eb73b77b326e3dd2dfc0a
```

### Event #23: `item.completed` — `file_change` (`item_21`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\THEOREM_M5.md",
    "kind": "update"
  }
]
```

### Event #24: `item.completed` — `command_execution` (`item_22`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$m5 = (Resolve-Path -LiteralPath '"'research/stage4_math_and_design/M5').Path
"'$env:UV_CACHE_DIR = Join-Path $m5 '"'.uv-cache'
"'$env:UV_PYTHON_INSTALL_DIR = Join-Path $m5 '"'.uv-python'
"'$env:TMP = $m5
$env:TEMP = $m5
$env:UV_LINK_MODE = '"'copy'
uv run --no-project --offline --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B -c \"import runpy; m=runpy.run_path('research/stage4_math_and_design/M5/verify_m5.py'); enc=m['load'](m['MATCHED'] / 'weights' / 'equivariant_b8_s17.npz'); print(m['wire_check'](enc)); print(m['correlated_innovation_fixture']())\""
```

**Aggregated Output:**

```text
{'wire_id': 'm5-c16-v1-le', 'byte_order': 'little', 'bits_msb_first': [2, 2, 6, 4, 2], 'valid_radial_code': [1, 15], 'erasure_radial_code': 0, 'erased_tail': 0, 'radial_edges_with_infinite_upper_encoded_as_null': [0.0, 0.23935579152928546, 0.5360027779971817, 0.9036539387158784, 1.3593045340150596, 1.924017738212866, 2.623898318388477, 3.491299368807998, 4.566317884224016, 5.898648307306074, 7.549879733383484, 9.596342964447686, 12.132639022018832, 15.276012230002532, 19.17177002025512, None], 'fixture_hex': '1408144800100000'}
{'schur_residual_complex_variance': 0.7252380952380952, 'conditional_log_density_max_abs_error': 6.661338147750939e-16, 'shared_cfo_diffusion_covariance': [[0.06559, 0.08945999999999998, 0.11983999999999997], [0.08945999999999998, 0.23004, 0.30816], [0.11983999999999997, 0.30816, 0.5478400000000001]]}
```

### Event #25: `item.completed` — `file_change` (`item_23`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\THEOREM_M5.md",
    "kind": "update"
  }
]
```

### Event #26: `item.completed` — `command_execution` (`item_14`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$ErrorActionPreference = '"'Stop'
"'$m5 = (Resolve-Path -LiteralPath '"'research/stage4_math_and_design/M5').Path
"'$base = uv cache dir
$env:UV_CACHE_DIR = Join-Path $m5 '"'.uv-cache'
"'$env:UV_PYTHON_INSTALL_DIR = Join-Path $m5 '"'.uv-python'
"'$env:TMP = $m5
$env:TEMP = $m5
$env:UV_LINK_MODE = '"'copy'
New-Item -ItemType Directory -Force (Join-Path "'$env:UV_CACHE_DIR '"'simple-v21/pypi'),(Join-Path "'$env:UV_CACHE_DIR '"'wheels-v6/pypi'),(Join-Path "'$env:UV_CACHE_DIR '"'archive-v0') | Out-Null
foreach ("'$name in @('"'numpy','scipy')) { Copy-Item -LiteralPath \""'$base/simple-v21/pypi/$name.rkyv" -Destination (Join-Path $env:UV_CACHE_DIR '"'simple-v21/pypi'); Copy-Item -LiteralPath \""'$base/wheels-v6/pypi/$name" -Destination (Join-Path $env:UV_CACHE_DIR '"'wheels-v6/pypi') -Recurse }
foreach ("'$id in @('"'cSnmXl2iWBpNAXGE','3Oxde2-Jkv6KGjnQ')) { Copy-Item -LiteralPath \""'$base/archive-v0/$id" -Destination (Join-Path $env:UV_CACHE_DIR '"'archive-v0') -Recurse }
uv run --no-project --offline --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M5/verify_m5.py"
```

**Aggregated Output:**

```text
Installed 2 packages in 4.27s
flat_minus3: 2
flat_minus3: 4_s17
flat_minus3: 8
flat_minus3: 12
flat_minus3: 16
flat_minus3: canonical_continuous
flat_minus3: full_iq
flat_plus2: 2
flat_plus2: 4_s17
flat_plus2: 8
flat_plus2: 12
flat_plus2: 16
flat_plus2: canonical_continuous
flat_plus2: full_iq
flat_plus24: 2
flat_plus24: 4_s17
flat_plus24: 8
flat_plus24: 12
flat_plus24: 16
flat_plus24: canonical_continuous
flat_plus24: full_iq
selective_fade: 2
selective_fade: 4_s17
selective_fade: 8
selective_fade: 12
selective_fade: 16
selective_fade: canonical_continuous
selective_fade: full_iq
adaptive jump_interval
adaptive selective_fade
adaptive cfo_alias
Wrote results_m5.json
```

### Event #27: `item.completed` — `file_change` (`item_24`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\THEOREM_M5.md",
    "kind": "update"
  }
]
```

### Event #28: `item.completed` — `command_execution` (`item_25`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding utf8 'C:/Users/李则徐/.codex/memories/rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md' | Select-Object -Skip 21 -First 5
rg -n '16-bit|K_ref|WIRE_ID' research/stage4_math_and_design/M1 research/stage4_math_and_design/M4 research/stage4_math_and_design/M3 -g 'reply_r2.md' -g '*.py' -m 2"
```

**Aggregated Output:**

```text
- P4-B original benchmark: 23 condition/SNR cells, 7,176 coded blocks per receiver. Original EMFloatFormer2D errored on 4,476 blocks; fixed-prior 2D-LMMSE on 4,307; matched-statistics LMMSE on 2,548.
- Pilot-front-end probe found Gaussian interpolation coherent gains for single-tap delays 0–4 of approximately `1.000, 0.735, 0.291, 0.062, 0.007`; Fourier interpolation nearly exactly reconstructs static integer-delay responses under the current comb-2 pilot geometry.
- Follow-up receiver experiments reduced errors on unseen delay distributions, but the pure Fourier/逐径 pilot front-end alone achieved nearly all of the gain. Therefore do not attribute this improvement to Transformer attention without matched-front-end controls.

## Task 2: Formal P4-C/P4-D confirmation and assessment
research/stage4_math_and_design/M4\verify_m4.py:194:    """M5 16-bit layout. `measured_residual` comes from the radio taps before R3 lift."""
research/stage4_math_and_design/M1\verify_m1.py:101:        raise ValueError("Expected integer 4-bit or 16-bit window words")
```

### Event #29: `item.completed` — `command_execution` (`item_26`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$r=Get-Content -Encoding utf8 research/stage4_math_and_design/M5/results_m5.json -Raw | ConvertFrom-Json
$s=$r.multi_geometry_adaptive_suite
$s.aggregate_post_reveal | ConvertTo-Json -Depth 3
$s.paired_gate.pooled_adaptive_advantages | ConvertTo-Json -Depth 4
foreach ($f in $s.per_family) { $p=$f.policy_metrics.max_EIG_window; Write-Output "$($f.family) active $($p.nll) $($p.accuracy) selections $($p.selected_counts -join '"','), best dev "'$($f.fixed_dev_candidate_index), EIG $($p.expected_eig) observed $($p.realized_gain), max grid delta $($f.max_quadrature_refinement_difference_first_64)" }
$s.paired_gate.oracle_fixed_candidate_per_family_and_endpoint | ConvertTo-Json -Depth 3'
```

**Aggregated Output:**

```text
{
    "max_EIG_window":  {
                           "nll":  0.04830112476069259,
                           "accuracy":  0.98193359375,
                           "n":  12288
                       },
    "random_window":  {
                          "nll":  0.5171324852652104,
                          "accuracy":  0.6504720052083334,
                          "n":  12288
                      },
    "nearest_window":  {
                           "nll":  0.511000762886971,
                           "accuracy":  0.6614583333333334,
                           "n":  12288
                       },
    "fixed_best_on_dev":  {
                              "nll":  0.5100484731297862,
                              "accuracy":  0.6621907552083334,
                              "n":  12288
                          },
    "fixed_best_oracle":  {
                              "nll":  0.5066957171061391,
                              "accuracy":  0.664794921875
                          }
}
{
    "random_window":  {
                          "nll_gain_nats":  {
                                                "gain":  0.4688313605045173,
                                                "simultaneous_95ci":  [
                                                                          0.4573967115573458,
                                                                          0.4802660094516888
                                                                      ]
                                            },
                          "accuracy_gain":  {
                                                "gain":  0.3314615885416667,
                                                "simultaneous_95ci":  [
                                                                          0.31775628541737316,
                                                                          0.3451668916659602
                                                                      ]
                                            }
                      },
    "nearest_window":  {
                           "nll_gain_nats":  {
                                                 "gain":  0.46269963812627807,
                                                 "simultaneous_95ci":  [
                                                                           0.4512772140516309,
                                                                           0.47412206220092523
                                                                       ]
                                             },
                           "accuracy_gain":  {
                                                 "gain":  0.3204752604166667,
                                                 "simultaneous_95ci":  [
                                                                           0.3068424798558248,
                                                                           0.33410804097750857
                                                                       ]
                                             }
                       },
    "fixed_best_on_dev":  {
                              "nll_gain_nats":  {
                                                    "gain":  0.46174734836909326,
                                                    "simultaneous_95ci":  [
                                                                              0.45021668047635005,
                                                                              0.47327801626183646
                                                                          ]
                                                },
                              "accuracy_gain":  {
                                                    "gain":  0.3197428385416667,
                                                    "simultaneous_95ci":  [
                                                                              0.3061608332605485,
                                                                              0.3333248438227849
                                                                          ]
                                                }
                          },
    "fixed_best_oracle":  {
                              "nll_gain_nats":  {
                                                    "gain":  0.45839459234544705,
                                                    "simultaneous_95ci":  [
                                                                              0.4386536989439554,
                                                                              0.4778997778837521
                                                                          ]
                                                },
                              "accuracy_gain":  {
                                                    "gain":  0.317138671875,
                                                    "simultaneous_95ci":  [
                                                                              0.29369139645017944,
                                                                              0.34058594729982056
                                                                          ]
                                                }
                          }
}
jump_interval active 0.04720560714446543 0.98291015625 selections 1001,1017,1066,1012, best dev 3, EIG 0.6434665672208586 observed 0.6459415734154799, max grid delta 1.05497862934456E-06
selective_fade active 0.003709597198110411 0.99853515625 selections 940,1039,1061,1056, best dev 1, EIG 0.6866883558356082 observed 0.6894375833618349, max grid delta 1.99464585382003E-06
cfo_alias active 0.09398816993950196 0.96435546875 selections 1042,1005,1020,1029, best dev 2, EIG 0.5951618676247444 observed 0.5991590106204433, max grid delta 8.40953497172059E-07
{
    "nll_gain_nats":  {
                          "jump_interval":  2,
                          "selective_fade":  1,
                          "cfo_alias":  0
                      },
    "accuracy_gain":  {
                          "jump_interval":  2,
                          "selective_fade":  1,
                          "cfo_alias":  0
                      }
}
```

### Event #30: `item.completed` — `command_execution` (`item_27`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding utf8 research/stage4_math_and_design/M4/verify_m4.py | Select-Object -Skip 175 -First 90
rg -n -i 'erasure|schema|wire|m5-c16|Radial|RefEvidence' research/stage4_math_and_design/M4/THEOREM_M4.md research/stage4_math_and_design/M3/THEOREM_M3.md | Select-Object -Last 18"
```

**Aggregated Output:**

```text
    table = encoder.table[codes]
    uniform_mi, capacity, counts = channel_information(labels, codes, len(encoder.table))
    return {
        'ser': float(np.mean(table.argmax(axis=1) != labels)),
        'frozen_table_nll_nats': float(-np.log(table[np.arange(len(labels)), labels]).mean()),
        'code_uniform_mi_nats': uniform_mi,
        'code_capacity_empirical_bracket_nats': capacity,
        'quadrant_local_counts_by_label': counts,
        'hidden_fraction_abs_gt_0p99': float((abs(hidden) > 0.99).mean()),
        'used_local_codes': int(len(np.unique(codes % len(encoder.centers)))),
    }


def lift(x, ancillary):
    return x[:, None] * E + ancillary @ BASIS.T


def pack_dual_cache(encoder, t, measured_residual, nominal_variance, ancillary):
    """M5 16-bit layout. `measured_residual` comes from the radio taps before R3 lift."""
    if not (np.all(np.isfinite(t)) and np.all(t != 0) and np.all(np.isfinite(measured_residual))
            and np.all(measured_residual >= 0) and np.all(np.isfinite(nominal_variance))
            and np.all(nominal_variance > 0)):
        raise ValueError('Unavailable windows must use the separate validity mask')
    legacy = encoder.encode(lift(t, ancillary)).astype(np.uint16)
    turn = legacy >> 2
    local = (np.angle(t * ROOTS[turn].conj()) + np.pi / 4) / (np.pi / 2)
    angle = np.clip(np.floor(64 * local), 0, 63).astype(np.uint16)
    radius = np.searchsorted(RADIAL_EDGES[1:-1], abs(t) / np.sqrt(nominal_variance)).astype(np.uint16)
    quality = np.searchsorted(QUALITY_EDGES[1:-1], measured_residual / nominal_variance).astype(np.uint16)
    packed = ((legacy << 12) | (angle << 6) | (radius << 2) | quality).astype('<u2')
    np.testing.assert_array_equal(packed >> 12, legacy)
    return packed


def polar_id(packet):
    """Drop only the legacy inner bits from the packed word."""
    return ((packet >> 14) << 12) | (packet & 0x0fff)


def polar_noise_emissions(mean, multiplier, tol_order):
    """P(angle, radius, quality | class) for unit nominal thermal variance."""
    node, weight = np.polynomial.legendre.leggauss(tol_order)
    width = 2 * np.pi / 256
    angle = -np.pi / 4 + (np.arange(256)[:, None] + (1 + node) / 2) * width
    radius = RADIAL_EDGES / np.sqrt(multiplier)
    mean = mean / np.sqrt(multiplier)
    radius = np.clip(radius, max(0, abs(mean) - 11), abs(mean) + 11)
    result = np.zeros((256, 16))
    lateral = 4 * abs(mean) * np.sin((angle - np.angle(mean)) / 2) ** 2
    for i, (lo, hi) in enumerate(zip(radius[:-1], radius[1:])):
        edges = np.linspace(lo, hi, 1 + max(1, math.ceil(hi - lo)))
        dr = np.diff(edges)[:, None] / 2
        r = (edges[1:, None] + edges[:-1, None]) / 2 + dr * node
        radial_weight = (dr * weight * r).ravel()
        distance = (r.ravel() - abs(mean)) ** 2 + lateral[..., None] * r.ravel()
        result[:, i] = (np.exp(-distance) @ radial_weight) @ (weight * width / (2 * np.pi))
    mass_quality = np.diff(gammainc(2, QUALITY_EDGES / multiplier))
    probability = (result[..., None] * mass_quality).ravel()
    np.testing.assert_allclose(probability.sum(), 1, atol=1e-9)
    return probability


def ceiling_readout(log_likelihood, qpsk, burst):
    logp = (log_likelihood - logsumexp(log_likelihood, axis=1, keepdims=True)).reshape(-1, 2, 4)
    index = np.arange(len(qpsk))
    by_tx = logsumexp(logp, axis=1)
    by_burst = logsumexp(logp, axis=2)
    return np.column_stack((-by_tx[index, qpsk], -by_burst[index, burst], -logp[index, burst, qpsk]))


def verify_dual_cache(encoder):
    rng = np.random.default_rng(SEED + 2)
    profiles = [('Pluto_fade', db, .0095 * 10 ** ((db + 2) / 20), 0.0) for db in (-2, 3, 8)]
    profiles += [('CTTC_spur', db, 1.0, 0.26) for db in (-3, 10, 20)]
    multipliers = np.array([1.0, 4.0])
    field = np.arange(1 << 16, dtype=np.uint16)
    state8 = np.arange(8)
    results = []
    max_quadrature_change = 0.0
    for profile, snr_db, gain, spur in profiles:
        nominal = 10 ** (-snr_db / 10)
        mean = np.exp(0.23j) + spur * np.exp(0.8j)
        distributions = {name: [] for name in ('s17', 'polar_quality', 'full16')}
        for multiplier in multipliers:
            v = nominal * multiplier
            t_cal = mean + np.sqrt(v) * noise(rng, 65_536)
            e_cal = v * rng.gamma(2, 1, len(t_cal))
            calibration = pack_dual_cache(encoder, t_cal, e_cal, nominal, np.sqrt(V0) * noise(rng, (len(t_cal), 2)))
            legacy_count = np.bincount(calibration >> 12, minlength=16)
            legacy = (legacy_count + 0.5) / (len(calibration) + 8)
research/stage4_math_and_design/M3/THEOREM_M3.md:5:The C4 generator sets the continuous drift to zero at the first window. It draws independent Gaussian increments with mean \(\omega\,dt\) and variance \(\sigma_\phi^2dt\) on the grid \(\{0,h,h+\ell,t_p\}\). Its two valid pilots share the same physical phase and have independent complex AWGN. At \(h=\ell=2,\ t_p=4\), the pilot clock equals the withheld-window clock. Each generator call takes one scalar \(t_p\). The schema can carry a different timestamp on each reference, but the seven C4 arms never exercise this case. The released C4 reader accepts only an ID, a channel Boolean and a matched sum. Its fitted encoder emissions and quadrature are tied to \(t_p=4,\ \omega=+0.35,\ \sigma_\phi=0.10,\ \mathrm{SNR}=2\,\mathrm{dB}\). [Generator and arms](../../stage3d_qwen_signed_phase/data_stage3d.py) · [Schema](../../stage3_ministral_cache/schema.py) · [Reader](../../stage3c/A/q2_reader.py) · [Stage 3C derivation](../../stage3c/A/Q2.md)
research/stage4_math_and_design/M3/THEOREM_M3.md:7:A copy has the same acquisition ID and samples. A fresh acquisition has a new ID and independent noise conditional on the shared path. The schema checks duplicate channel flags, timestamps within \(10^{-12}\), and exact sample equality. The weighted pool also checks amplitude and noise variance, which affect the likelihood. Stage 3D's distractor has a separate random quadrant, phase, and noise. Its density cancels when calculating the target posterior. These results condition on public, exogenous acquisition design and correct channel declarations.
research/stage4_math_and_design/M3/THEOREM_M3.md:9:Q2 targets the original transmitted \(X_1\). The repeat protocol has \(X_3=X_1\). Q3_after_pilot targets the **sector of the noisy apparent matched sum** of the withheld third observation. This is a different random variable and requires the phase posterior at \(h+\ell\). [Prompt and sector definitions](../../stage3_ministral_cache/schema.py) · [Label construction](../../stage3d_qwen_signed_phase/train_eval_stage3d.py)
research/stage4_math_and_design/M3/THEOREM_M3.md:160:1. **Validate and canonicalize.** Use \(z_i=4a_i+u_i\). Resolve IDs and check provenance, calibration and exact sample equality before filtering. A production clock should publish one canonical timestamp per physical tick. For schema-tolerance aliases, the prototype sorts timestamps per ID, chooses the minimum, and rejects a spread beyond \(10^{-12}\). Remove independent-channel references before feature evaluation. A conflict-free distractor contributes no activation or gradient. A literal copy of a canonical acquisition ID makes no change. Sort unique IDs before reduction for deterministic sums.
research/stage4_math_and_design/M4/THEOREM_M4.md:11:Write \(e=\mathbf1_3/\sqrt3\), \(V_0=10^{-0.2}=0.63095734448\), \(S\in\{1,i,-1,-i\}\), and use proper complex noise with \(\mathbb E WW^*=K\) and real per-coordinate variance \(V_0/2\) under the nominal law. `s17` was trained with \(Y_0=eS+W_0,\ W_0\sim\mathcal{CN}_3(0,V_0I)\). The whole window has unit signal energy. Its matched statistic is \(T_0=e^*Y_0\). Each window produces 16 codes, four quadrant turns times four local codes. The frozen lookup table differs from the MLP softmax used to choose the local code. [Frozen encoder](../../matched_cache/learned.py) · [Training waveform law](../../fourphase_paired_questions/paired.py) · [Stage 3 matched sum](../../stage3_ministral_cache/schema.py)
research/stage4_math_and_design/M4/THEOREM_M4.md:98:Rayleigh denominator with \(\mathbb E G^2=1\) is \(\nu=1\). In one complex dimension \(P(|X|^2/\Sigma>q)=(1+q)^{-1}\), so \(\mathbb E|X|^2=\infty\), and forward KL to any nondegenerate Gaussian is infinite. The normalized Student entropy is finite here. More generally an infinite second moment gives infinite KL to a Gaussian by testing bounded truncations of a sufficiently small positive multiple of \(\|X\|^2\) in the variational KL lower bound. If the amplitude estimator itself has a folded Student law, substitute its density \(2f_{t_d}(g/b)/b\) for \(g>0\) in (6), where \(f_{t_d}\) is the standard real Student density and \(b\) its scale. That exact integral is a different scale mixture, with infinite second moment since the folded density is positive at zero. A noisy complex pilot LS amplitude \(|h+E|/a\) with nondegenerate proper Gaussian \(E\) is conditionally Rician. Its radial density has positive linear coefficient at zero, so \(\mathbb E G^{-2}=\infty\) even for nonzero \(h\). If `a` itself is Rayleigh but estimated exactly and both amplitude and noise scale with it, \(G=1\) and the heavy tail argument does not apply. If a new thermal sample and the scale estimator share noise, the exact law uses \(W\mid G\) in (6) and cannot be inferred from the gain histogram alone.
research/stage4_math_and_design/M4/THEOREM_M4.md:110:For `s17`, along a fixed nonzero canonical ray \(x=cu\), \(c>0\), the sum sector is scale invariant. At low gain \(\tanh(cuW_1+b_1)\to\tanh b_1\), so the local code becomes constant if the winning cross-entropy score has a positive margin. The hidden deviation is at most \(c\|uW_1\|\). Converting it through \(W_2\) and the \(1/2\)-Lipschitz softmax, the bound on pairwise score changes is \((c/2)\|uW_1\|\|W_2\|_{2\to2}\|\log C_i-\log C_j\|\) for centers \(C_i,C_j\). A strict margin greater than this perturbation proves stability. For high gain, neuron \(j\) with \(d_j=(uW_1)_j\ne0\) approaches \(\operatorname{sgn}(d_j)\) with error at most \(2e^{-2(c|d_j|-|b_{1j}|)}\) once \(c|d_j|>|b_{1j}|\); neurons with zero projection stay at \(\tanh b_{1j}\). Apply the same score-margin test to the limiting logits and center choices. This proves radial saturation on each interior ray with strict margin. The four turns can remain informative. At exact all-zero input the encoder chooses its defined tie behavior. `verify_m4.py` finds `[1,5,9,13]` in the low-gain limit and `[0,4,8,12]` on its high-gain ray. Ties, near-orthogonal directions, and random scales near zero preclude a global AGC threshold.
research/stage4_math_and_design/M4/THEOREM_M4.md:158:### 2.4 CPU shard wire contract · `m4.real_dual_cache.v2`
research/stage4_math_and_design/M4/THEOREM_M4.md:160:Process each eligible window once. The CPU worker reads causal raw stencils, applies the frozen template and whitening, serializes both views and computes the matched-information physical reference from the sealed observations. The round-1 split and observation whitelist still apply. `encoder_sha256`, manifest/calibration/quantizer revisions, sample-rate/clock map, arm-independent ancillary RNG version, split/recording version, and schema/covariance units are mandatory manifest fields. Hash the header and every shard. Use little-endian fixed dtypes below. Encode erasure as an explicit flag and use zero-filled payload fields as placeholders. Packet word zero and all four legacy inner labels remain legal on valid rows.
research/stage4_math_and_design/M4/THEOREM_M4.md:173:Pack `((s17 & 15)<<12) | (angle<<6) | (radial<<2) | quality`. The exact M5 bins use the turn from `s17`, canonical angle `arg(T i^(-turn))` and half-open `[-π/4,π/4)` with the encoder's sector boundary rule. Multiply its shifted fraction by 64, floor and clip to `0..63`. Radial (r=|T|/\sqrt{v_{\rm nom}}\) uses internal edges `expm1(j log(25)/16), j=1..15`. Residual (E/v_{\rm nom}) uses `0.75, sqrt(12), 16`. Radial and residual exact edge ties stay below the edge (`searchsorted(side="left")`). The rails saturate. The M5 0.312-nat event/location result is a separate synthetic study. It sets no hardware error bar for this packet. [M5 packet and CPU reader](../M5/THEOREM_M5.md)
research/stage4_math_and_design/M4/THEOREM_M4.md:187:| Common receive gauge \(r_{rx}\) | Multiply all received IQ, matched pilot values, causal complex DC/gain/anchor and the realized ancillary by \(i^{r_{rx}}\). The packed top two bits and first-window anchor `a1` shift by `+r_rx` mod 4. `T` rotates, `E`, radial/quality fields, rate, covariance and TX pilot labels stay invariant. | invariant | invariant | roll `+r_rx` |
research/stage4_math_and_design/M4/THEOREM_M4.md:191:At-valid-input code and output actions are exact permutations. Canonicalization rotates by the inverse observed `a1`; `r_rx` is the applied receive gauge, distinct from the inferred anchor. Never use absolute receiver phase as a TX relabel. Isolated data-window rotation is a separate M1 synthetic intervention requiring the stated synthetic law. Invalid windows and zero phase pivot use the common erasure path when no valid pivot is present. The verifier tests this wire behavior on simulated real-data timelines, and independently checks probability actions of a finite physical likelihood. Final Qwen parity remains a deployment receipt.
```

### Event #31: `item.completed` — `command_execution` (`item_28`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding utf8 research/stage4_math_and_design/M1/verify_m1.py | Select-Object -Skip 84 -First 48
rg -n 'def (pack|extract|decode)|class .*Ref|innovation|K_ref' research/stage4_math_and_design/M3/verify_m3.py | Select-Object -Last 20
rg --files research/stage4_math_and_design | Where-Object { "'$_ -match '"'reply_r2' }"
```

**Aggregated Output:**

```text
        c = torch.special.ndtr(scale * (angle + math.pi / 4).cos())
        s = torch.special.ndtr(scale * (angle + math.pi / 4).sin())
        base = torch.stack((c * s, (1 - c) * s, (1 - c) * (1 - s), c * (1 - s)), dim=-1)
        sector = (self.weights[:, None] * base).sum(dim=-2)
        return torch.stack([sector.roll(j, dims=-1) for j in range(4)], dim=-2)


class EvidencePacket(NamedTuple):
    codes: torch.Tensor
    tail: torch.Tensor
    valid: torch.Tensor


def decode_windows(words, bits, valid=None):
    """Decode the bit-exact S17 prefix and optional invariant M5 fields."""
    if bits not in (4, 16) or words.is_floating_point() or words.is_complex() or words.dtype == torch.bool:
        raise ValueError("Expected integer 4-bit or 16-bit window words")
    words = words.to(torch.int64)
    if words.ndim != 2 or words.shape[-1] != 2 or bool(((words < 0) | (words >= 2**bits)).any()):
        raise ValueError("Expected two in-range words per item")
    valid = torch.ones_like(words, dtype=torch.bool) if valid is None else valid
    if valid.shape != words.shape or valid.dtype != torch.bool:
        raise ValueError("Validity mask must be boolean and match the window words")
    words = torch.where(valid, words, 0)
    if bits == 4:
        tail = torch.zeros((*words.shape, 6), dtype=torch.float32, device=words.device)
        return EvidencePacket(words, tail, valid)
    angle = (((words >> 6) & 63) + 0.5) * (math.pi / 128) - math.pi / 4
    # Summarize bins by log-midpoint, with a separate saturation indicator.
    # Open-ended cells remain intervals in a calibrated likelihood reader.
    rad = (words >> 2) & 15
    qual = words & 3
    log_edges = torch.as_tensor(np.log1p(radial_edges(4)[:-1]), dtype=torch.float32, device=words.device)
    log_edges = torch.cat((log_edges, log_edges.new_tensor([math.log(25)])))
    log_rad = (log_edges[rad] + log_edges[rad + 1]) / 2
    quality_low = torch.as_tensor(np.log1p(quality_edges(2)[:-1]), dtype=torch.float32, device=words.device)[qual]
    tail = torch.stack((angle.cos(), angle.sin(), log_rad, quality_low,
                        (rad == 15) | (qual == 3), torch.ones_like(angle)), dim=-1)
    return EvidencePacket(words >> 12, torch.where(valid[..., None], tail, 0), valid)


class FrozenHost(nn.Module):
    """Frozen MLP and a random regular-simplex candidate projection."""

    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
80:class RefSufficientPool(nn.Module):
606:def check_correlated_innovations():
640:    ordered_ids, innovation = pool.innovation(state, ("pilot-B", "pilot-A"), 7)
642:    innovation_ll = innovation.logpdf([first.matched, second.matched])
643:    factor_error = float(np.max(np.abs(full - old_only - innovation_ll)))
652:    s = innovation.covariance[0]
659:    suff_error = float(np.max(np.abs(kern + common - innovation_ll[0])))
725:    innovations = np.stack(np.meshgrid(x, x, indexing="ij"), -1).reshape(-1, 2)
726:    return mean + innovations @ np.linalg.cholesky(cov).T, (w[:, None]*w).ravel()/(2*np.pi)
941:    innovation = check_correlated_innovations()
949:        "innovation": innovation, "shared_cfo_multiclock": multi, "stage6": stage6,
research/stage4_math_and_design\M2\reply_r2.md
research/stage4_math_and_design\discussions\M2\reply_r2.md
```

### Event #32: `item.completed` — `command_execution` (`item_29`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '"'^class |''^def '"' research/stage4_math_and_design/M3/verify_m3.py | Select-Object -Last 40
Get-Content -Encoding utf8 research/stage4_math_and_design/M4/THEOREM_M4.md | Select-Object -Skip 157 -First 58
rg -n 'RADIAL_EDGES|QUALITY_EDGES' research/stage4_math_and_design/M4/verify_m4.py | Select-Object -First 10"
```

**Aggregated Output:**

```text
49:def rot90(iq, turn):
56:def as_iq(values):
61:class Pilot:
71:class Pooled:
80:class RefSufficientPool(nn.Module):
133:class FixedSlotQ2(nn.Module):
159:def metrics(probs, labels):
164:def compare(left, right, labels, block_size=64):
179:def make_arms(m1, m2, mi):
188:def audit_archive(bundle, kernel, summary, smoke):
244:def fit_apparent(encoder):
260:def check_pool(pool, reader, blind, pinned):
339:def bridge_weights(c1, c2, times, sigma, z2_phase_emission, inner_order=20):
349:def multi_post(s, families, a, times, sigma, z2_phase_emission, order):
382:def multi_draws(encoder, times, n, seed):
392:def multi_evidence(pool, encoder, draws, sigma, times):
407:def multi_test(encoder, a, marginal):
498:def runtime_versions():
503:def rerun_round1(blind, kernel):
530:def normalize_log(log_w):
534:def direct_density(values, mean, covariance):
541:def rejected(action, message):
550:def scenario(clocks, frame=0):
568:def clock_row(name, tick, root=1, available=None, role="reference"):
573:def build_given_history(clocks, y0, frame=0):
581:def q_readout_fixture(pool):
593:def prefix_fixture(state, joint):
606:def check_correlated_innovations():
718:def q_turn_from_law(law, values, weights):
723:def gaussian_nodes(mean, cov, order):
729:def check_cfo_brownian():
807:def check_stage6():
865:def check_pinned_c4_adapter(blind, kernel):
892:def contrast_native_scores(bundle, blind, result_dir):
923:def main():
### 2.4 CPU shard wire contract · `m4.real_dual_cache.v2`

Process each eligible window once. The CPU worker reads causal raw stencils, applies the frozen template and whitening, serializes both views and computes the matched-information physical reference from the sealed observations. The round-1 split and observation whitelist still apply. `encoder_sha256`, manifest/calibration/quantizer revisions, sample-rate/clock map, arm-independent ancillary RNG version, split/recording version, and schema/covariance units are mandatory manifest fields. Hash the header and every shard. Use little-endian fixed dtypes below. Encode erasure as an explicit flag and use zero-filled payload fields as placeholders. Packet word zero and all four legacy inner labels remain legal on valid rows.

| Record | Frozen fields and ownership |
| --- | --- |
| **Window shard** | `packet16 <u2[rows,2]`, `valid u1[rows,2]`, `tick <i8[rows,2]`, `available_sample <i8[rows,2]`; a legacy view is `packet16 >> 12` and an optional dense `u4` nibble stream stores first window in bits `3:0`. GPU-visible input selects this prefix plus the common context and current release only. CPU-only continuous diagnostic fields `T <c16`, `E <f8`, `nominal_variance <f8`, `gain <f8`, clipping/quality-reason flags and typed nuisance diagnostics see the same cutoff. Q3 target IQ and the corresponding packet, if produced for scoring, stay outside the model shard. |
| **PhaseContext** | `t1,t2,t3 <i8` in fixed ticks, `target_valid u1`, `clock_id` and `time_unit=symbol`, `ticks_per_symbol=2^20`. Convert certified matched clock coordinates to ticks once using nearest ties-to-even. Subtract integer `t1` before converting a lag to float64 symbol units. Sub-symbol timing and actual raw availability are also in the provenance record. Pluto uses local frame clocks with the known repeat spacing. CTTC uses the causal sample-rate and clock map, including SFO. Unknown Pluto inter-frame times get a missing clock and cannot feed signed-phase transport. Synthetic uses the same contract with `0,h,h+ℓ`. |
| **PhaseContext nuisance state** | `omega_hat <f8` (alias `omega_rad_per_unit`), `v_omega <f8`, `diffusion_var_per_unit <f8`, `snr_linear <f8[2]`, `estimate_tick <i8`, `estimate_age_ticks <i8`, `phase_anchor_tick <i8`, `anchor_mean/anchor_cov` for allowed gauge-dependent state, `phase_model_id`, `state_digest`, `valid` flags. Sign is always \(T(t)\propto e^{+i\widehat\omega(t-t_1)}\). If deterministic CFO is removed in the frontend, `omega_hat` is the residual in that exact coordinate system. SNR is matched signal energy divided by proper-complex variance. Keep phase/rate correlation or weighted joint hypotheses when fitted. An opaque state digest is audit metadata. |
| **Reference record** | `acquisition_id` is HMAC-SHA256 over immutable dataset/capture identity, physical sample-support and observation kind. Same physical sample stencil maps to one ID across presentations, targets and splits. Record `timestamp <i8`, `available_sample <i8`, support and calibration digests, `shares_channel u1`, `tx_root u1`, `matched_T <c16`, `nominal_variance <f8`, `snr_linear <f8`, and observation cut/release stage. Stable IDs and split/capture/source coordinates stay in the ledger and never enter prompts. `tx_root` is a whitelist known symbol. Sort unique usable references by timestamp and ID, and retain a validity mask. |
| **RefPool innovation inputs and outputs** | `K_ref <c16[2,2]` (raw complex noise within allowed pilot block), `K_ref_B` and a causal joint law or conditional innovation kernel if `B` is correlated. Align the matrix with canonical reference order. A later single-length view uses its principal submatrix and masks empty slots. Pass the shared rate state once via `PhaseContext`, plus anchor prior and state/ledger identifiers. Publish fixed-shape canonical `q2[4]`, `q3[4]` and optional `joint[4,4]` or phase-node likelihood factors with clocks and normalization. The host projector accepts fixed `uint8` masks and canonical float32 numeric features. Canonicalize signed zero at serialization. |
| **Provenance/scorer vault** | Enrolled row ID, physical/source intervals including filter support, safe timing receipt, seen/release/deadline ticks, every parent acquisition ID reaching `F0` or a release, split state digest, label/capture/source strata and target mask. Feature workers see only the known-symbol whitelist and raw permissions. The model consumes neither audit IDs, capture identities, cyclic source addresses, hidden tuning values nor oracle columns. |

To compute physical residual energy, let pre-normalized matched taps have steering \(q\) and nominal residual covariance \(v C\), where \(C\) is the positive-definite shape. Set \(b=q^*C^{-1}q\), whiten with \(C^{-1/2}/\sqrt b\), subtract the causal DC estimate, divide by the causal gain and rotate by a fixed unitary sending the whitened steering vector to \(e\). In these normalized coordinates \(R_w\), use \(T=e^*R_w\), \(E=\|(I-ee^*)R_w\|^2\), and \(v_{\rm nom}=v/(b\widehat a^2)\). Materialize `Y_R3=e T+Bη` with the frozen independent nominal ancillary. The M5 quality bits summarize **measured** \(E\). The artificial orthogonal energy \(\|\eta\|^2\) is irrelevant to radio residual quality. Erase or flag any missing physical support and never synthesize a quality value on its behalf. R3 assumes input coordinates already use the declared fixed receiver convention, with causal derotation recorded if applied.

Pack `((s17 & 15)<<12) | (angle<<6) | (radial<<2) | quality`. The exact M5 bins use the turn from `s17`, canonical angle `arg(T i^(-turn))` and half-open `[-π/4,π/4)` with the encoder's sector boundary rule. Multiply its shifted fraction by 64, floor and clip to `0..63`. Radial (r=|T|/\sqrt{v_{\rm nom}}\) uses internal edges `expm1(j log(25)/16), j=1..15`. Residual (E/v_{\rm nom}) uses `0.75, sqrt(12), 16`. Radial and residual exact edge ties stay below the edge (`searchsorted(side="left")`). The rails saturate. The M5 0.312-nat event/location result is a separate synthetic study. It sets no hardware error bar for this packet. [M5 packet and CPU reader](../M5/THEOREM_M5.md)

The two packed bytes include the old nibble. An upper count for the locked window schedule, even charging all 7,096 Pluto frame slots, all 5,472 CTTC target/reference slots and all 2,060 registered stress rows for three distinct windows, is `3×(7,096+5,472+2,060)=43,884` words or **87,768 bytes**, **109,710 bytes** if the separate nibble stream is materialized. This is an accounting bound for the cache payload, including hypothetical slots for non-task Pluto frames. The real model still reads only eligible split windows. A dense waveform/symbol export, calibration, timestamps, references, and provenance have separate storage costs. The 16-bit quantizer adds zero GPU rows. Measure CPU extraction throughput in the existing CPU cap.

### 2.5 Conditional updates and group actions

Let \(B\) be the authorized observations incorporated in the prior or cached context. Deduplicate and remove every complete acquisition included in the transitive baseline ledger before applying likelihoods. Repeated presentation order, duplicate copies, empty slots and independent distractors must serialize to the same canonical prefix when they carry zero new evidence. Distinct pilots with correlated noise are genuine observations. A different ID on a physically identical sample stencil is a provenance violation. Changes to a repeated ID's fields are errors. Any direct raw read intersecting a blocked/future stencil invalidates that feature bundle. Qualification requires a trusted timing envelope, full read/label provenance and poisoning of withheld columns, as in §2.2. This also covers old sync observations and new pilots that share analog noise filtered upstream.

At fixed latent state, write a proper joint Gaussian for baseline samples \(Y_B\) and new pilots \(Y_R\). Then \(e_R=(Y_R-\mu_R)-K_{RB}K_{BB}^{-1}(Y_B-\mu_B)\) has covariance \(K_{RR}-K_{RB}K_{BB}^{-1}K_{BR}\) after conditioning. The new factor is its normalized complex Gaussian density. Condition the second release on both \(B\) and the first release. Multiplying the two conditional factors equals the joint conditional likelihood. Feeding an old sample updates precision twice under a fictitious independent law. For the actual duplicate its conditional law is a point mass at the existing value, with no update. If an arm is only allowed a function of \(Y_B\), integrate the conditional density over the allowed posterior of \(Y_B\) including any quantizer cells. Giving the restricted cache reader the realized continuous samples in the innovation would change its information budget. If the deployed noise model uses residual independence, validate it on authorized dev data or report the likelihood as approximate. [M3 pooling and likelihood](../M3/THEOREM_M3.md)

At signed offsets \(d_i=t_i-t_1\), a single conditionally Gaussian \(\Delta\omega\) with variance \(v_\omega\) adds \(v_\omega d_i d_j\) to the joint phase covariance. A Brownian two-sided increment referenced at \(t_1\) adds \(D\min(|d_i|,|d_j|)\mathbf1[d_i d_j>0]\) for the zero-anchor covariance case. More generally form the conditional bridge from `phase_anchor_tick` with the full phase/rate covariance. All real reveal clocks precede `t1` physically. The reveal-stage index controls when the observations enter the model and does not determine their RF time. Transport on a single edge has harmonic damping \(e^{-n^2(D|d|+v_\omega d^2)/2}\) in the independent increment case. Joint updates integrate a shared latent CFO or its exact joint covariance. Keep observation covariance separate so CFO uncertainty enters once.

| Action | Observations and state | Q1 | Q2 | Q3 |
| --- | --- | --- | --- | --- |
| Common receive gauge \(r_{rx}\) | Multiply all received IQ, matched pilot values, causal complex DC/gain/anchor and the realized ancillary by \(i^{r_{rx}}\). The packed top two bits and first-window anchor `a1` shift by `+r_rx` mod 4. `T` rotates, `E`, radial/quality fields, rate, covariance and TX pilot labels stay invariant. | invariant | invariant | roll `+r_rx` |
| Global TX relabel \(r_{tx}\) with fixed receive coordinates | `s_k,d_m→i^{r_tx}` times their old values, absorb `i^{-r_tx}` into the channel/prior, RX IQ and packet stay fixed. | invariant | roll `+r_tx` | invariant |
| Both | Compose the actions on their declared fields. | invariant | roll `+r_tx` | roll `+r_rx` |

At-valid-input code and output actions are exact permutations. Canonicalization rotates by the inverse observed `a1`; `r_rx` is the applied receive gauge, distinct from the inferred anchor. Never use absolute receiver phase as a TX relabel. Isolated data-window rotation is a separate M1 synthetic intervention requiring the stated synthetic law. Invalid windows and zero phase pivot use the common erasure path when no valid pivot is present. The verifier tests this wire behavior on simulated real-data timelines, and independently checks probability actions of a finite physical likelihood. Final Qwen parity remains a deployment receipt.

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
38:RADIAL_EDGES = np.r_[np.expm1(np.linspace(0, np.log(25), 17))[:-1], np.inf]
39:QUALITY_EDGES = np.r_[0.0, np.geomspace(0.75, 16, 3), np.inf]
203:    radius = np.searchsorted(RADIAL_EDGES[1:-1], abs(t) / np.sqrt(nominal_variance)).astype(np.uint16)
204:    quality = np.searchsorted(QUALITY_EDGES[1:-1], measured_residual / nominal_variance).astype(np.uint16)
220:    radius = RADIAL_EDGES / np.sqrt(multiplier)
232:    mass_quality = np.diff(gammainc(2, QUALITY_EDGES / multiplier))
```

### Event #33: `item.completed` — `command_execution` (`item_30`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-Content -Encoding utf8 research/stage4_math_and_design/M4/verify_m4.py | Select-Object -First 63
Get-Content -Encoding utf8 research/stage4_math_and_design/M3/verify_m3.py | Select-Object -First 62
rg --files research/stage4_math_and_design/M3
Get-Content -Encoding utf8 research/stage4_math_and_design/M3/verify_m3.py | Select-Object -Skip 803 -First 61'
```

**Aggregated Output:**

```text
"""Replay the M4 CPU checks, with no radio downloads or Modal calls by default."""

import argparse
import ast
from dataclasses import dataclass, replace
from decimal import Decimal
import hashlib
import hmac
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlsplit
from urllib.request import urlopen

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import betaln, digamma, gammainc, gammaln, logsumexp, ndtr, xlogy


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from research.matched_cache.learned import ROOTS, assign, forward, input_features, load

WEIGHT_PATH = ROOT / 'research/matched_cache/weights/equivariant_b8_s17.npz'
LOCK_PATH = ROOT / 'research/stage3d_qwen_signed_phase/PROTOCOL_LOCK.json'
PLUTO_REF = 'origin/research/realdata-pluto-qpsk'
CTTC_REF = 'origin/research/realdata-cttc-indoor-qpsk'
V0 = 10 ** (-0.2)
E = np.ones(3) / np.sqrt(3)
BASIS = np.column_stack(([1, -1, 0], np.array([1, 1, -2]) / np.sqrt(3))) / np.sqrt(2)
SEED = 404_017
HOLDOUT_CAPTURE = 'f708_snr10_QPSK'
RADIAL_EDGES = np.r_[np.expm1(np.linspace(0, np.log(25), 17))[:-1], np.inf]
QUALITY_EDGES = np.r_[0.0, np.geomspace(0.75, 16, 3), np.inf]


def branch_text(ref, folder, filename):
    return subprocess.check_output(
        ['git', 'show', f'{ref}:research/{folder}/{filename}'], cwd=ROOT, encoding='utf-8'
    )


def literal_constants(source, names):
    """Read selected branch literals and NumPy index constructors without importing branch code."""
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = getattr(node.targets[0], 'id', None)
        elif isinstance(node, ast.AnnAssign):
            name = getattr(node.target, 'id', None)
        else:
            continue
        if name in names:
            value = node.value
            if isinstance(value, ast.Call):
                # These are the only constructors used by the requested branch constants.
                function = ast.unparse(value.func)
                if function == 'np.concatenate':
"""CPU verification of the prior-aware pool, with optional full Round 1 replay."""

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy.special import logsumexp, roots_hermitenorm, softmax
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO), str(REPO / "research" / "stage3c" / "A")]

from check_q2 import fit_likelihood, make_bridge
from q2_reader import read
from ref_pool import (
    ClockMeta, GaussianLaw, NodeBank, Observation, PhaseContext,
    RefSufficientPool as ConditionalRefPool, conditional_gaussian, fixed_slots,
    sample_digest, task_readout,
)
from research.matched_cache.learned import load
from research.stage3d_qwen_signed_phase import data_stage3d as data
from research.stage3d_qwen_signed_phase.recompute_all import (
    Q2_LOGICAL_REF_IDX,
    score_vectors,
    work_a_q2_analysis,
)


RHO = 10**0.2
OMEGA = 0.35
H = 2
SIGMA = 0.10
TP = 4.0
CLASSES = np.arange(4)
ROOTS = data.ROOTS
MAX_PILOTS = 64


def rot90(iq, turn):
    """Use sign changes and swaps so the canonical frame is exact in floating point."""
    x, y = iq.unbind(dim=-1)
    orbit = torch.stack((iq, torch.stack((-y, x), -1), -iq, torch.stack((y, -x), -1)), dim=1)
    return orbit[torch.arange(len(iq), device=iq.device), turn % 4]


def as_iq(values):
    return torch.from_numpy(np.stack((values.real, values.imag), axis=-1))


@dataclass(frozen=True)
class Pilot:
    acquisition_id: int
research/stage4_math_and_design/M3\verify_m3.py
research/stage4_math_and_design/M3\THEOREM_M3.md
research/stage4_math_and_design/M3\results_m3.json
research/stage4_math_and_design/M3\reply_r1.md
research/stage4_math_and_design/M3\ref_pool.py
            "metrics": scores, "paired_against_full": comp}


def check_stage6():
    clocks = (clock_row("sync", 0), clock_row("alias", 2, 1j, role="probe"),
              clock_row("probe", 7, -1, 9, role="probe"))
    times, angles, means, k, prior = scenario(clocks)
    # Same conditional law under each hypothesis at the aliased candidate, including calibrated K.
    means[1, ..., :2] = means[0, ..., :2]
    k[1] = k[0]
    y0 = 1.31+0.19j
    weights = normalize_log(prior + GaussianLaw(means[..., :1], k[:, :1, :1]).logpdf([y0]))
    open_phase = angles.copy()
    open_phase[1, :, :2] = open_phase[0, :, :2]
    pool = ConditionalRefPool(NodeBank(("smooth", "jump"), times, open_phase, weights), clocks, means, k)
    state = pool.start((Observation(clocks[0], y0),), 0)
    rejected(lambda: pool.forecast(state, "probe", 7), "unavailable")
    rejected(lambda: pool.forecast(state, "sync", 9), "already consumed")
    # The fixture exposes metadata to the predictor and hides measured IQ until a committed read.
    forecasts = {c.acquisition_id: pool.forecast(state, c.acquisition_id, 9) for c in clocks[1:]}
    eig = {key: forecast.eig(24) for key, forecast in forecasts.items()}
    assert abs(eig["alias"]) < 3e-15 and eig["probe"] > 0.1
    key = max(eig, key=eig.get)
    committed = forecasts[key]
    vault = {"alias": -0.7+0.24j, "probe": 0.60-0.61j}
    reading = Observation(clocks[2], vault.pop(key))
    rejected(lambda: pool.reveal(state, committed, reading, 8), "unavailable")
    rejected(lambda: pool.reveal(state, committed, reading, 10), "commitment")
    after = pool.reveal(state, committed, reading, 9)
    rejected(lambda: pool.reveal(after, committed, reading, 9), "commitment")
    assert pool.update(after, (reading,), 9) is after
    threshold = 4
    assert reading.clock.physical_tick > threshold and len(vault) == 1

    f, m, posterior = committed.log_densities([reading.matched])
    log_h0 = logsumexp(state.log_weights_by_turn, axis=(-2, -1))
    posterior_error = float(np.max(np.abs(posterior-logsumexp(after.log_weights_by_turn, axis=(-2, -1)))))
    z_error = float(abs(m - (after.log_increment-state.log_increment)))
    assert max(posterior_error, z_error) < 2e-14

    axis = np.linspace(-6, 6, 181)
    grid = (axis[:, None]+1j*axis).reshape(-1, 1)
    area = (axis[1]-axis[0])**2
    log_f_grid, log_m_grid, log_post_grid = committed.log_densities(grid)
    mass_h = np.exp(log_f_grid).sum(axis=0)*area
    mass_m = np.exp(log_m_grid).sum()*area
    marginalized = np.sum(np.exp(log_post_grid+log_m_grid[..., None]), axis=0)*area
    grid_eig = np.sum(np.exp(log_post_grid+log_m_grid[..., None])*(log_post_grid-log_h0))*area
    eig_error = abs(grid_eig-eig[key])
    assert np.max(np.abs(mass_h-1)) < 2e-11 and abs(mass_m-1) < 2e-11
    assert np.max(np.abs(marginalized-np.exp(log_h0))) < 2e-11 and eig_error < 3e-5
    return {"candidates": len(forecasts), "eig_nats": eig, "selected": key,
            "post_target_clock": threshold, "selected_physical_tick": reading.clock.physical_tick,
            "last_reveal_tick": reading.clock.available_tick, "observed_reads": 1,
            "hypothesis_density_mass": mass_h.tolist(), "mixture_density_mass": float(mass_m),
            "prior_h": np.exp(log_h0).tolist(), "posterior_h": np.exp(posterior).tolist(),
            "forecast_log_conditional_at_reveal": f.tolist(), "forecast_log_marginal_at_reveal": float(m),
            "pre_post_log_evidence_max_diff": z_error, "pre_post_log_posterior_max_diff": posterior_error,
            "quadrature_vs_grid_eig_diff": float(eig_error)}


```

### Event #34: `item.completed` — `command_execution` (`item_31`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding utf8 research/stage4_math_and_design/M3/ref_pool.py | Select-Object -First 55
rg -n '"'^class |''^    def (start|forecast|reveal|log_densities)'"' research/stage4_math_and_design/M3/ref_pool.py
Get-Content -Encoding utf8 research/stage4_math_and_design/M3/ref_pool.py | Select-Object -Skip 150 -First 75"
```

**Aggregated Output:**

```text
"""Prior-aware RefSufficientPool for proper complex Gaussian observation blocks."""

from dataclasses import dataclass
import hashlib
import math
from operator import index
import struct
from types import MappingProxyType

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp, roots_hermitenorm, xlogy


ROOTS = (1+0j, 1j, -1+0j, -1j)
MAX_REFERENCES = 64
WIRE_VERSION = 2


def readonly(array, dtype):
    out = np.array(array, dtype=dtype, copy=True, order="C")
    out.flags.writeable = False
    return out


def canonical_iq(value, frame):
    # Exact quarter-turns before any general floating-point arithmetic.
    a, b = value.real, value.imag
    return (complex(a, b), complex(b, -a), complex(-a, -b), complex(-b, a))[frame % 4]


def sample_digest(canonical_sample):
    a, b = canonical_sample.real, canonical_sample.imag
    return hashlib.sha256(struct.pack("<dd", a or 0.0, b or 0.0)).hexdigest()


@dataclass(frozen=True)
class PhaseContext:
    anchor_tick: int
    ticks_per_unit: float
    omega_hat: float
    v_omega: float
    diffusion_var_per_unit: float
    p_phi0: float = 0.0
    cov_phi0_omega: float = 0.0
    phi0_hat: float = 0.0

    def path_moments(self, ticks):
        vals = (self.ticks_per_unit, self.omega_hat, self.v_omega, self.diffusion_var_per_unit,
                self.p_phi0, self.cov_phi0_omega, self.phi0_hat)
        if (not np.isfinite(vals).all() or self.ticks_per_unit <= 0
                or min(self.v_omega, self.diffusion_var_per_unit, self.p_phi0) < 0
                or abs(self.cov_phi0_omega) > math.sqrt(self.v_omega*self.p_phi0)):
            raise ValueError("Invalid anchored Brownian/CFO prior")
        # Subtract integer ticks before conversion, preserving short lags at large epochs.
38:class PhaseContext:
66:class ClockMeta:
87:class Observation:
99:class NodeBank:
108:class PoolState:
118:class GaussianLaw:
170:class Forecast:
176:    def log_densities(self, samples):
205:class RefSufficientPool:
237:    def start(self, history, as_of):
335:    def forecast(self, state, acquisition_id, reveal_deadline):
339:    def reveal(self, state, forecast, obs, as_of):
    mean = means[..., new].copy()
    if not len(past):
        return GaussianLaw(mean, knn)
    knp = k[:, new][:, :, past]
    kpp = k[:, past][:, :, past]
    kpn = k[:, past][:, :, new]
    # Missing boundary samples are harmless only for exact zero conditional coupling.
    a = np.linalg.solve(kpp, kpn).conj().swapaxes(-1, -2)
    missing = np.array([value is None for value in seen], dtype=bool)
    if np.any(a[..., missing] != 0):
        raise ValueError("F0 lacks a correlated Gaussian boundary observation")
    y = np.array([0 if value is None else value for value in seen], dtype=np.complex128)
    mean += np.einsum("hjp,hgnp->hgnj", a, y - means[..., past])
    s = knn - a @ kpn
    # Keep the Hermitian structure against solve roundoff.
    return GaussianLaw(mean, (s + s.conj().swapaxes(-1, -2))/2)


@dataclass(frozen=True)
class Forecast:
    state: PoolState
    acquisition_ids: tuple
    law: GaussianLaw
    reveal_deadline: int

    def log_densities(self, samples):
        ell = self.law.logpdf(samples)
        weights = self.state.log_weights_by_turn
        joint = logsumexp(ell + weights, axis=(-2, -1))
        log_prior = logsumexp(weights, axis=(-2, -1))
        marginal = logsumexp(joint, axis=-1)
        with np.errstate(invalid="ignore"):
            conditional = np.where(np.isfinite(log_prior), joint-log_prior, -np.inf)
        return conditional, marginal, joint-marginal[..., None]

    def eig(self, order=24):
        if len(self.acquisition_ids) != 1:
            raise ValueError("The Stage 6 query reads one complex acquisition")
        z, w = roots_hermitenorm(order)
        grid = (z[:, None]+1j*z).ravel()
        gh = (w[:, None]*w).ravel()/(2*np.pi)
        prior = np.exp(logsumexp(self.state.log_weights_by_turn, axis=(-2, -1)))
        log_prior = logsumexp(self.state.log_weights_by_turn, axis=(-2, -1))
        var = self.law.covariance[:, 0, 0].real
        mean = self.law.mean[..., 0]
        nodes = (mean[..., None]+np.sqrt(var[:, None, None, None]/2)*grid).reshape(-1, 1)
        weight = np.exp(self.state.log_weights_by_turn[..., None]) * gh
        _, _, log_post = self.log_densities(nodes)
        p = np.exp(log_post)
        # 0 log 0 is zero, including hypotheses with zero current prior.
        kl = np.sum(xlogy(p, p) - p*np.where(prior > 0, log_prior, 0), axis=-1)
        return float(weight.ravel() @ kl)


class RefSufficientPool:
    """Apply unique observations to an immutable conditional F0 snapshot."""

    def __init__(self, nodes, clocks, means, covariance, receiver_frame=0):
        self.hypothesis_ids = tuple(nodes.hypothesis_ids)
        self.physical_ticks = tuple(nodes.physical_ticks)
        self.phase_grid = readonly(nodes.phase_grid, np.float64)
        self.initial = readonly(nodes.log_weights_by_turn, np.float64)
        self.clocks = tuple(clocks)
        self.indices = MappingProxyType({item.acquisition_id: i for i, item in enumerate(clocks)})
        self.means = readonly(means, np.complex128)
        raw_cov = np.asarray(covariance, dtype=np.complex128)
        h, d = len(self.hypothesis_ids), len(self.clocks)
        if raw_cov.ndim == 2:
            raw_cov = np.broadcast_to(raw_cov, (h, d, d))
        self.covariance = readonly(raw_cov, np.complex128)
        self.frame = receiver_frame % 4
        self._owner = object()
        if (self.initial.ndim != 3 or self.initial.shape[:2] != (h, 4)
                or self.phase_grid.shape != (h, self.initial.shape[-1], len(self.physical_ticks))
                or len(self.indices) != d or len(set(self.hypothesis_ids)) != h
```

### Event #35: `item.completed` — `command_execution` (`item_32`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-Content -Encoding utf8 research/stage4_math_and_design/M3/ref_pool.py | Select-Object -Skip 65 -First 55
Get-Content -Encoding utf8 research/stage4_math_and_design/M3/ref_pool.py | Select-Object -Skip 216 -First 43
Get-Content -Encoding utf8 research/stage4_math_and_design/M3/ref_pool.py | Select-Object -Skip 325 -First 35'
```

**Aggregated Output:**

```text
class ClockMeta:
    acquisition_id: str
    provenance: str
    physical_tick: int
    available_tick: int
    known_root: complex | None
    calibration_id: str
    raw_support: tuple = ()  # optional in fixtures; half-open raw-tick intervals in M4
    role: str = "reference"

    def __post_init__(self):
        if (not self.acquisition_id or not self.provenance or not self.calibration_id
                or not isinstance(self.physical_tick, int) or not isinstance(self.available_tick, int)
                or self.available_tick < self.physical_tick
                or self.known_root not in (*ROOTS, None)
                or self.role not in ("history", "reference", "probe")
                or any(a >= b or b - 1 > self.available_tick for a, b in self.raw_support)):
            raise ValueError("Invalid clock metadata or known QPSK root")


@dataclass(frozen=True)
class Observation:
    clock: ClockMeta
    matched: complex | None  # None only for history retained in the consumption ledger
    shares_channel: bool = True
    digest: str | None = None  # canonical sample digest, mandatory for retired history

    @property
    def acquisition_id(self):
        return self.clock.acquisition_id


@dataclass(frozen=True)
class NodeBank:
    # Columns of phase_grid are physical_ticks, including target clocks if required.
    hypothesis_ids: tuple
    physical_ticks: tuple
    phase_grid: np.ndarray   # (H, N, clocks), real radians before the unknown quarter-turn
    log_weights_by_turn: np.ndarray  # (H, 4, N), globally normalized when entering the pool


@dataclass(frozen=True)
class PoolState:
    owner: object
    base: np.ndarray
    history: tuple  # Every identifier consumed by the F0 snapshot
    added: tuple
    log_weights_by_turn: np.ndarray
    log_increment: float


@dataclass(frozen=True)
class GaussianLaw:
    mean: np.ndarray
    covariance: np.ndarray  # (H, d, d)
        h, d = len(self.hypothesis_ids), len(self.clocks)
        if raw_cov.ndim == 2:
            raw_cov = np.broadcast_to(raw_cov, (h, d, d))
        self.covariance = readonly(raw_cov, np.complex128)
        self.frame = receiver_frame % 4
        self._owner = object()
        if (self.initial.ndim != 3 or self.initial.shape[:2] != (h, 4)
                or self.phase_grid.shape != (h, self.initial.shape[-1], len(self.physical_ticks))
                or len(self.indices) != d or len(set(self.hypothesis_ids)) != h
                or len(set(self.physical_ticks)) != len(self.physical_ticks)
                or self.means.shape != (*self.initial.shape, d)
                or not np.isfinite(logsumexp(self.initial))
                or not np.isfinite(self.phase_grid).all() or not np.isfinite(self.means).all()
                or self.covariance.shape != (h, d, d)
                or not np.isfinite(self.covariance).all()
                or not np.array_equal(self.covariance, self.covariance.conj().swapaxes(-1, -2))
                or any(item.physical_tick not in self.physical_ticks for item in self.clocks)):
            raise ValueError("Inconsistent phase bank or proper complex Gaussian design")
        np.linalg.cholesky(self.covariance)

    def start(self, history, as_of):
        # Initial weights are already conditional on all supplied F0 history.
        old = self._canonical_records(history, as_of, allow_missing=True, allow_outside=True)
        if any(obs.clock.available_tick > as_of for obs in old):
            raise ValueError("F0 contains an unavailable observation")
        base = readonly(self.initial - logsumexp(self.initial), np.float64)
        return PoolState(self._owner, base, old, (), base, 0.0)

    def _validate_state(self, state):
        if state.owner is not self._owner:
            raise ValueError("State belongs to a different Gaussian design")

    def _canonical_records(self, records, as_of, allow_missing=False, allow_outside=False):
        by_id = {}
        for raw in records:
            if raw is None:
                continue  # typed transport padding
            index = self.indices.get(raw.acquisition_id)
            if index is not None and self.clocks[index] != raw.clock:
                raise ValueError(f"Conflicting metadata for {raw.acquisition_id}")
            if index is None and raw.shares_channel and not allow_outside:
                raise ValueError(f"No validated provenance for {raw.acquisition_id}")
            if raw.shares_channel and raw.clock.available_tick > as_of:
            if (index is None or self.clocks[index].role == "history"
                    or self.clocks[index].available_tick > reveal_deadline):
                raise ValueError("Unknown candidate or unavailable by the reveal deadline")
        boundary = tuple(sorted((*state.history, *state.added), key=lambda o: o.acquisition_id))
        if set(ids) & {obs.acquisition_id for obs in boundary}:
            raise ValueError("Candidate is already consumed")
        law = self._conditional(ids, boundary)
        return ids, GaussianLaw(readonly(law.mean, np.complex128), readonly(law.covariance, np.complex128))

    def forecast(self, state, acquisition_id, reveal_deadline):
        ids, law = self.innovation(state, (acquisition_id,), reveal_deadline)
        return Forecast(state, ids, law, reveal_deadline)

    def reveal(self, state, forecast, obs, as_of):
        if (forecast.state is not state or forecast.acquisition_ids != (obs.acquisition_id,)
                or as_of > forecast.reveal_deadline):
            raise ValueError("Reveal differs from the forecast commitment")
        if not obs.shares_channel:
            raise ValueError("Forecast was for a validated same-channel candidate")
        return self.update(state, (obs,), as_of)


def task_readout(state, joint_given_node):
    target = np.asarray(joint_given_node, dtype=np.float64)
    weights = np.exp(state.log_weights_by_turn)
    if (target.shape != (*weights.shape, 4, 4) or not np.isfinite(target).all()
            or np.any(target < 0) or not np.allclose(target.sum(axis=(-2, -1)), 1, atol=2e-12, rtol=0)):
        raise ValueError("Expected normalized Q2/Q3 joint emission at each state")
    joint = np.einsum("hgn,hgnkl->kl", weights, target)
    return joint.sum(1), joint.sum(0), joint


def fixed_slots(canonical_codes, q2, q3):
    """Versioned canonical header + two always-present float32 evidence vectors."""
    if len(canonical_codes) != 3 or any(v not in range(4) for v in canonical_codes):
```

### Event #36: `item.completed` — `agent_message` (`item_33`)

The first Round 2 run cleared all four controls and selected all four candidate IDs in every geometry. The latest M4 shard uses a separate validity mask. I’ll verify a byte-level conversion that preserves its 16-bit field layout and carries erasure inside the Stage 6 word, then compare the reveal update against M3’s reference pool.

### Event #37: `item.completed` — `file_change` (`item_34`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\verify_m5.py",
    "kind": "update"
  }
]
```

### Event #38: `item.completed` — `file_change` (`item_35`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\verify_m5.py",
    "kind": "update"
  }
]
```

### Event #39: `item.completed` — `file_change` (`item_37`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\THEOREM_M5.md",
    "kind": "update"
  }
]
```

### Event #40: `item.completed` — `command_execution` (`item_38`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '"'^## [789]|''^### 7'"\\.1|v0-to-v1|two lowest|M3 shared pool|Stage 6 common' research/stage4_math_and_design/M5/THEOREM_M5.md
rg -n '"'^def (m4_shard_to_wire|check_m3_reveal|multi_geometry_adaptive_suite|simultaneous_intervals)|''^class OneRevealRefPool|ADAPTIVE_'"' research/stage4_math_and_design/M5/verify_m5.py"
```

**Aggregated Output:**

```text
224:## 7. Round 2 · Exact value of adapting to the recording
250:### 7.1 Multi-geometry construction
266:## 8. Round 2 wire lock · m5-c16-v1-le
280:Version 1 keeps M4's field widths and all higher radial edges. The existing M4 Stage 4 shard keeps an out-of-band validity byte and uses all 16 radial codes. Its v0-to-v1 conversion maps valid old radial code 0 to 1, leaves other valid codes alone, and clears the 12-bit tail for unavailable or zero-pivot rows. Thus v1 merges the two lowest v0 cells and the rest of the packet needs no re-quantization. Phase eligibility must already include the availability and zero-pivot check at the causal CPU boundary. The Round 1 rates in §5 use v0. Version 1 is the self-contained wire used in the new multi-geometry run. The M1 v0 decoder must explicitly select the v1 interpretation when given a v1 word, including its erasure mask and merged lowest radial cell. [M4 shard wire contract](../M4/THEOREM_M4.md#24-cpu-shard-wire-contract--m4real_dual_cachev2)
306:The means include $\hat\omega d_i$. Condition the joint law on permitted old evidence before evaluating predictions. Treat shared CFO as one latent variable. If a correlated raw pair $(R_o,R_a)$ is Gaussian conditional on state $u$, the fresh conditional factor has mean $\mu_a+K_{ao}K_{oo}^{-1}(R_o-\mu_o)$ and covariance $K_{aa}-K_{ao}K_{oo}^{-1}K_{oa}$. The old law normalizes the joint exactly once. If only quantized $Z$ retains old support, use $p(R_a\mid Z,u)=\int p(R_a\mid r_o,u)p(dr_o\mid Z,u)$ or an exact equivalent innovation state. Substituting a point estimate of the old observation into the Schur formula is generally an approximation. In Round 2 the candidate thermal innovations are conditionally independent. The vectorized scalar pool checks pre/post likelihood identity, exact deduplication, and neutral prior/distractor IDs. The verifier cross-checks selected actual reveals through M3's shared pool. A separate correlated Gaussian fixture checks the Schur expression against the joint-minus-old log density. The latent CFO/diffusion and hardware noise covariances play distinct roles. [M1 Round 2 brief](../discussions/M1/prompt_r2.md) · [M3 shared pool](../M3/ref_pool.py) · [M4 causal frontend](../M4/THEOREM_M4.md)
308:## 9. Stage 6 common benchmark contract · m5-stage6-blind-v1
43:ADAPTIVE_DEV = 2048
44:ADAPTIVE_TEST = 4096
45:ADAPTIVE_FAMILIES = ("jump_interval", "selective_fade", "cfo_alias")
46:ADAPTIVE_CONTROLS = ("random_window", "nearest_window", "fixed_best_on_dev")
620:def m4_shard_to_wire(words, available):
781:class OneRevealRefPool:
874:def check_m3_reveal(geometry, weights, selection, sample, posterior, log_cond, log_marginal):
906:    dev_words, _, _, _, _ = geometry_split(geometry, encoder, family_id, ADAPTIVE_DEV, 0)
909:    words, truth, state, sealed, random_action = geometry_split(geometry, encoder, family_id, ADAPTIVE_TEST, 1)
969:def simultaneous_intervals(by_family, controls):
1025:def multi_geometry_adaptive_suite(encoder):
1028:    report = simultaneous_intervals(by_family, ADAPTIVE_CONTROLS)
1029:    policies = ("max_EIG_window", *ADAPTIVE_CONTROLS)
1041:            "query_utf8": FROZEN_QUERY, "public_families": ADAPTIVE_FAMILIES,
1051:            "seed": SEED, "rng_namespace": 62, "dev_mothers_per_family": ADAPTIVE_DEV,
1052:            "test_mothers_per_family": ADAPTIVE_TEST,
```

### Event #41: `item.completed` — `command_execution` (`item_39`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n -i 'M5|Stage 6|fixed.best|strict' research/stage4_math_and_design/M2/reply_r2.md research/stage4_math_and_design/M2/THEOREM_M2.md | Select-Object -Last 16
rg -n '"'^##'"' research/stage4_math_and_design/M5/THEOREM_M5.md | Select-Object -Last 10"
```

**Aggregated Output:**

```text
research/stage4_math_and_design/M2/reply_r2.md:21:- **M4/M6** · Use IUT within each of the three claim slots and Holm at 0.05 across synthetic+Q2, Pluto and CTTC. Plan CTTC with **nine capture units** when their conditional independence is defensible. The f708_snr10 result is a held-out cohort guard. Hold the capture out of every fit and calibration. Shared cyclic TX source restricts generalization.
research/stage4_math_and_design/M2/reply_r2.md:22:- **M5** · Require Stage 6 max EIG to beat random, nearest and dev-frozen fixed-best on both post-reveal NLL and accuracy at the same one-read budget. Pair the 16-bit canonical vs 4-bit S17 representation test separately under the same prescribed action.
research/stage4_math_and_design/M2/THEOREM_M2.md:73:Let \(R_gq(x)=\rho(g)^{-1}q(gx)\) for a label permutation representation. Assume the joint law of \((X,Y)\) is invariant under \((x,y)\mapsto(gx,\rho(g)y)\) and all evaluated probabilities are positive. Then \(L(R_gq)=L(q)\). For \(m=Pq=|G|^{-1}\sum_g R_gq\), define \(J_y(x)=\log m_y(x)-|G|^{-1}\sum_g\log (R_gq)_y(x)\). Pointwise \(J_y\ge0\) by strict concavity of \(\log\). Hence
research/stage4_math_and_design/M2/THEOREM_M2.md:79:Equality holds exactly when, almost surely, every label of positive posterior mass has the same probability across every aligned member. If the posterior has full support, this is equivariance almost surely. On an arbitrary finite design the exact formula adds \(L(q)-|G|^{-1}\sum_g L(R_gq)\) to the label-weighted Jensen gap. The complete orbits constructed from each saved mother have an invariant design by construction. Saved rotations are an incomplete orbit, so the restriction matters.
research/stage4_math_and_design/M2/THEOREM_M2.md:97:Over a target distribution, strict accuracy decline is equivalent to \(\mathbb E[(h_y(q)-h_y(Pq))_+]>\mathbb E[(h_y(Pq)-h_y(q))_+]\), including ties. At population level this is the same as \(\mathbb E_X[p(X)\cdot(d(q)-d(Pq))]>0\), where \(d\) is uniform on the maximizer set. For geometric pooling the corresponding competitor margins are averages of \(\log r_y-\log r_k\). Log score depends on the probability assigned to the true class, while top-1 depends on which side of the decision planes the pool lies.
research/stage4_math_and_design/M2/THEOREM_M2.md:104:2. **Calibration and structure.** Propose an upper bound of 0.05 on top-label ECE after dev-only calibration with ten frozen bins. Propose a paired Brier improvement of at least 0.005 against the strongest locked trivial or zero-phase Brier competitor, plus a strict improvement over every required comparator. For \(K=4\) uniform Brier is \(1-1/K=0.75\) with the unnormalized multiclass sum. Prelock a probability transformation ensuring finite NLL, for example \(q_\epsilon=(1-K\epsilon)q+\epsilon\) with small fixed \(\epsilon\), and score that forecast consistently. For calibration inference set \(d_j=\mathbb E[1_{\{\operatorname{conf}\in I_j\}}(h_Y-\operatorname{conf})]\); \(\operatorname{ECE}=\sum_j|d_j|=\max_{v\in\{-1,1\}^{10}}v^\top d\). Test all \(2^{10}\) linear upper-margin conditions as an IUT or use a valid joint confidence region for \(d\). Estimating the absolute value first and treating empirical ECE as an unbiased, normal per-mother loss has no exact justification. Require \(\max_{g,x}\|q(gx)-\rho(g)q(x)\|_\infty\le10^{-5}\) and projection/readout idempotence \(\max_x\|P(Pq)(x)-Pq(x)\|_\infty\le10^{-5}\). Verify across the whole finite code domain and all group elements on synthetic. For continuous real inputs distinguish formal architectural guarantees from the observed maximum on prespecified transforms. Test only valid actions with physically correct label permutations. Audit the deployed path and the offline projector separately.
research/stage4_math_and_design/M2/THEOREM_M2.md:214:On G1 phase the arithmetic pool makes 8.76% of decisions better and 15.51% worse, the exact weighted hit changes here. Geometric makes 8.28% better and 15.72% worse. G1 phase F0 scores 20.88% / 1.411163. On complete orbits G1 phase's arithmetic gain is 0.03386703, while the reverse-barycenter KL is 0.03430077. For zero the corresponding numbers are 0.04120584 and 0.03488442. Restricting to the three saved rotations adds 0.00004913 and 0.00055046 to the loss gain. Across the tested cells pooled maximum equivariance and idempotence residuals were at floating-point precision. G1 raw maximum probability discrepancies were 0.48167 and 0.41957 for phase and zero. Structural compliance of the offline tables coexists with poor decoding and poor ECE.
research/stage4_math_and_design/M2/THEOREM_M2.md:302:| 2. Frozen connector panel | The paired mean loss over the selected \(S=2\) or \(3\) frozen seed pairs satisfies the gate, conditional on the selected checkpoints. | M6 continuation targets \(S=3\). Freeze the panel and average losses or hits within block. Report each seed. A deployed probability ensemble has its own estimand and must be frozen before scoring. Extra survivor-only seeds may be evaluated against deterministic controls. Restrict matched trained-arm contrasts to their common preselected paired seeds. |
research/stage4_math_and_design/M2/THEOREM_M2.md:369:For independent equal-weight Gaussian units with common variance the homogeneous t has \(B-1\) df. In the balanced 32-block synthetic model, fitting eight fixed cell means with common residual variance gives \(32-8=24\) df and the same \(\sigma/\sqrt{32}\) standard error for their average. For unequal allocational counts \(n_h\) with fixed weights \(w_h\), the common-variance standard error is \(\sigma\sqrt{\sum_h w_h^2/n_h}\), with \(B-H\) residual df. For independent Gaussian capture means with fixed heterogeneous means and common variance, the sample variance denominator is noncentral chi-square and independent of the grand mean. It is stochastically larger than the central denominator, giving a conservative upper-tail test at the null boundary. Independent bounded unit means allow the Hoeffding test in §3 for mean margins if Gaussian assumptions are unsupported. Valid random-label p-values require sanctioned assignments under the relevant sharp null. Preserve rotations, repeats, captures and source restrictions in the exchangeability orbit. Use a registered randomized negative control when the existing label design lacks one. A paired sign-flip test needs actual randomization or a justified joint sign-symmetry null. A weak mean inequality alone supplies no such exchangeability.
research/stage4_math_and_design/M2/THEOREM_M2.md:373:## 8. Stage 6 compatibility
research/stage4_math_and_design/M2/THEOREM_M2.md:375:Freeze one task, one pre-question representation, one admissible action set per recording, and exactly one sealed read of equal cost. Give max EIG, uniform random, nearest and fixed best on dev the same available candidates, reader family, calibration opportunity and cutoff. Each policy commits its action and pre-reveal forecast before receiving the read. Seal nonchosen evidence from the policy. The blinded evaluator pairs policies by recording with recorded counterfactual acquisitions when valid. Otherwise preassign an action design with sufficient support and use its registered design-based estimator. Freeze tie breaking. Nearest uses the last authorized anchor with a causal distance rule. Fixed best on dev selects a single constant action (or a prespecified mask-availability fallback rule) by dev-only realized posterior NLL and freezes it. It cannot select based on blind performance or hidden ambiguity.
research/stage4_math_and_design/M2/THEOREM_M2.md:377:For the adaptive acquisition claim, use \(D_L=L_{\rm baseline}-L_{\rm max\,EIG}\) and \(D_A=A_{\rm max\,EIG}-A_{\rm baseline}\) on blinded post-reveal predictions, with both one-sided lower bounds strictly positive for **each** of random, nearest and fixed best on dev. Use one IUT on those six endpoints at the claim's registered level. Stratify or nest trial units at the recording level. A pre-reveal proper-score and seal audit tests commitment and integrity separately. For true post-reveal Bayes NLL with equal read costs, \(V(z,a)=I(Y;R_a\mid Z=z)\). The adaptive-versus-best-static value is
research/stage4_math_and_design/M2/THEOREM_M2.md:381:With a finite common action set this is strict iff every fixed action is strictly suboptimal on a set of positive probability. For varying masks, replace each constant by an eligible frozen availability-only policy and apply the analogous criterion to that policy class. Merely beating random and nearest does not exclude a static optimum. Accuracy has its own Bayes decision value and its own empirical contrast.
research/stage4_math_and_design/M2/THEOREM_M2.md:383:Treat 16-bit canonical versus 4-bit S17 as a distinct representation contrast paired on identical source records, existing evidence, target and prescribed reveal action. Use the same calibrated reader family and development opportunity and count the actual storage bits. Require positive NLL gain and positive accuracy gain if making a joint superiority claim. Require exact S17 prefix reconstruction and all registered symmetry checks. Repeat the full policy comparison at fixed 16-bit budget for an adaptive claim, so representation and action choice can be attributed separately. Full IQ is the diagnostic ceiling after matching allowed causal inputs. Predeclare both Stage 6 claim slots and use Holm within that family if claiming them separately. An optional portfolio-wide 0.05 rule opens the independent Stage 6 confirmatory family only after all three Stage 4 claims pass. If Stage 4 contains a true claim null, reaching Stage 6 already entails a Stage 4 error. If all Stage 4 nulls are false, conditional Stage 6 family control applies after selection. Without this sequential rule, report the families at their separate levels and refrain from a joint 0.05 claim. [M5 action and cache design](../discussions/M5/prompt_r2.md)
170:## 5. CPU verification
176:### 5.1 Achieved rate–risk points
193:### 5.2 Sealed one-window selection
207:### 5.3 Rerun and interpretation
217:## 6. Round 2 handoff
224:## 7. Round 2 · Exact value of adapting to the recording
250:### 7.1 Multi-geometry construction
266:## 8. Round 2 wire lock · m5-c16-v1-le
282:### 8.1 One CPU pass and one innovation path
308:## 9. Stage 6 common benchmark contract · m5-stage6-blind-v1
```

### Event #42: `item.completed` — `file_change` (`item_40`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\verify_m5.py",
    "kind": "update"
  }
]
```

### Event #43: `item.completed` — `file_change` (`item_41`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\THEOREM_M5.md",
    "kind": "update"
  }
]
```

### Event #44: `item.completed` — `command_execution` (`item_43`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Item -LiteralPath 'research/stage4_math_and_design/M5/results_m5.json' | Select-Object Length,LastWriteTime
rg -n 'dev_lock|m3_shared_pool_eight|strict_nll' research/stage4_math_and_design/M5/results_m5.json -m 7"
```

**Aggregated Output:**

```text

15690:      "dev_lock": "maximize mean predicted EIG within each public family; smallest candidate ID breaks ties",
16721:      "strict_nll_and_accuracy_gate_passed": true
Length LastWriteTime      
------ -------------      
586890 2026/11/26 19:54:43


```

### Event #45: `item.completed` — `command_execution` (`item_44`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n 'P4 is|same physical|matched-control|matched-front-end' 'C:/Users/李则徐/.codex/memories/MEMORY.md' 'C:/Users/李则徐/.codex/memories/rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md' | Select-Object -First 6
Get-Item research/stage4_math_and_design/M5/results_m5.json | Select-Object Length,LastWriteTime"
```

**Aggregated Output:**

```text
C:/Users/李则徐/.codex/memories/rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md:24:- Follow-up receiver experiments reduced errors on unseen delay distributions, but the pure Fourier/逐径 pilot front-end alone achieved nearly all of the gain. Therefore do not attribute this improvement to Transformer attention without matched-front-end controls.
C:/Users/李则徐/.codex/memories/rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md:65:Conclusion: P4 made substantial algorithmic progress by implementing and testing a complex-valued, gauge-equivariant, Hermitian-attention Transformer prototype with physical channel simulation and coded-link evaluation. The strongest evidence is that signal symmetries and complex-valued operations were encoded directly into the model and some mechanisms showed reproducible value in controlled 1D ablations. However, the formal 2D result did not establish a learned-Transformer advantage; the largest 2D improvements came from better pilot/channel front-ends. The next decisive experiment is a strictly matched comparison where EM and classical receivers share the same physical front-end, pilot information, decoder, and inference budget, isolating only the learned module.
C:/Users/李则徐/.codex/memories/MEMORY.md:32:applies_to: cwd=C:\Users\李则徐\Documents\ChatGPT\EM-FloatFormer; reuse_rule=Reuse the matched-control, reproducibility, and publication approach for this repository; numerical results, commit, hardware status, and receiver outcomes are P4-specific.
C:/Users/李则徐/.codex/memories/MEMORY.md:61:- P4 is a reproducible CPU digital-baseband prototype with C* gauge normalization, complex RMSNorm, Hermitian attention, complex relative-phase RoPE, radial activation, physical loss, nonlinear PA, Doppler/CFO, sparse pilots, and coded LDPC BLER evaluation; it is not real electronic/hardware-native execution. [Task 1]

Length LastWriteTime      
------ -------------      
586890 2026/11/26 19:54:43


```

### Event #46: `item.completed` — `agent_message` (`item_45`)

The verifier now checks the conversion against M4’s actual packer and replays selected observations through M3’s shared pool. The final sweep also locks each fixed window by realized post-reveal NLL on the separate dev split, matching M2’s gate.

### Event #47: `item.completed` — `command_execution` (`item_46`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding utf8 research/stage4_math_and_design/M3/ref_pool.py | Select-Object -Skip 45 -First 19
Get-ChildItem -LiteralPath 'research/stage4_math_and_design/M5' -Force -Name"
```

**Aggregated Output:**

```text
    phi0_hat: float = 0.0

    def path_moments(self, ticks):
        vals = (self.ticks_per_unit, self.omega_hat, self.v_omega, self.diffusion_var_per_unit,
                self.p_phi0, self.cov_phi0_omega, self.phi0_hat)
        if (not np.isfinite(vals).all() or self.ticks_per_unit <= 0
                or min(self.v_omega, self.diffusion_var_per_unit, self.p_phi0) < 0
                or abs(self.cov_phi0_omega) > math.sqrt(self.v_omega*self.p_phi0)):
            raise ValueError("Invalid anchored Brownian/CFO prior")
        # Subtract integer ticks before conversion, preserving short lags at large epochs.
        t = np.array([index(tick) - self.anchor_tick for tick in ticks], dtype=np.float64) / self.ticks_per_unit
        if np.any(t < 0) or not np.isfinite(t).all():
            raise ValueError("Clock predates the Brownian prediction anchor or has invalid units")
        cov = (self.diffusion_var_per_unit * np.minimum.outer(t, t)
               + self.v_omega * np.outer(t, t)
               + self.p_phi0 + self.cov_phi0_omega*(t[:, None]+t))
        return self.phi0_hat+self.omega_hat*t, cov


.uv-cache
reply_r1.md
results_m5.json
THEOREM_M5.md
verify_m5.py
```

### Event #48: `item.completed` — `file_change` (`item_47`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\THEOREM_M5.md",
    "kind": "update"
  }
]
```

### Event #49: `item.completed` — `command_execution` (`item_36`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$m5 = (Resolve-Path -LiteralPath '"'research/stage4_math_and_design/M5').Path
"'$env:UV_CACHE_DIR = Join-Path $m5 '"'.uv-cache'
"'$env:UV_PYTHON_INSTALL_DIR = Join-Path $m5 '"'.uv-python'
"'$env:TMP = $m5
$env:TEMP = $m5
$env:UV_LINK_MODE = '"'copy'
uv run --no-project --offline --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M5/verify_m5.py"
```

**Aggregated Output:**

```text
adaptive jump_interval
adaptive selective_fade
adaptive cfo_alias
flat_minus3: 2
flat_minus3: 4_s17
flat_minus3: 8
flat_minus3: 12
flat_minus3: 16
flat_minus3: canonical_continuous
flat_minus3: full_iq
flat_plus2: 2
flat_plus2: 4_s17
flat_plus2: 8
flat_plus2: 12
flat_plus2: 16
flat_plus2: canonical_continuous
flat_plus2: full_iq
flat_plus24: 2
flat_plus24: 4_s17
flat_plus24: 8
flat_plus24: 12
flat_plus24: 16
flat_plus24: canonical_continuous
flat_plus24: full_iq
selective_fade: 2
selective_fade: 4_s17
selective_fade: 8
selective_fade: 12
selective_fade: 16
selective_fade: canonical_continuous
selective_fade: full_iq
Wrote results_m5.json
```

### Event #50: `item.completed` — `command_execution` (`item_48`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$r=Get-Content -Encoding utf8 research/stage4_math_and_design/M5/results_m5.json -Raw | ConvertFrom-Json
$r.multi_geometry_adaptive_suite.design.dev_lock
$r.round2_packet.m4_shard_conformance_fixture_words
$r.multi_geometry_adaptive_suite.aggregate_post_reveal | ConvertTo-Json -Depth 3
foreach ($f in $r.multi_geometry_adaptive_suite.per_family) { Write-Output ($f.family + '"' ' + ("'$f.policy_metrics.max_EIG_window.selected_counts -join '"',') + ' M3 err ' + "'$f.m3_shared_pool_eight_selected_rows_max_abs_error) }'
```

**Aggregated Output:**

```text
maximize mean predicted EIG within each public family; smallest candidate ID breaks ties
188
{
    "max_EIG_window":  {
                           "nll":  0.04822777767181902,
                           "accuracy":  0.9820149739583334,
                           "n":  12288
                       },
    "random_window":  {
                          "nll":  0.517222912107759,
                          "accuracy":  0.6495768229166666,
                          "n":  12288
                      },
    "nearest_window":  {
                           "nll":  0.5109821061347279,
                           "accuracy":  0.6605631510416666,
                           "n":  12288
                       },
    "fixed_best_on_dev":  {
                              "nll":  0.5100964867245413,
                              "accuracy":  0.6622721354166666,
                              "n":  12288
                          },
    "fixed_best_oracle":  {
                              "nll":  0.5067152050526097,
                              "accuracy":  0.663818359375
                          }
}
jump_interval 1001,1017,1066,1012 M3 err 3.99680288865056E-15
selective_fade 941,1037,1061,1057 M3 err 3.5527136788005E-15
cfo_alias 1042,1005,1020,1029 M3 err 3.5527136788005E-15
```

### Event #51: `item.completed` — `command_execution` (`item_49`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$r=Get-Content -Encoding utf8 research/stage4_math_and_design/M5/results_m5.json -Raw | ConvertFrom-Json
$s=$r.multi_geometry_adaptive_suite
$s.paired_gate.pooled_adaptive_advantages | ConvertTo-Json -Depth 5
foreach ($f in $s.per_family) { $p=$f.policy_metrics.max_EIG_window; Write-Output "$($f.family) nll=$($p.nll) acc=$($p.accuracy) exp=$($p.expected_eig) gain=$($p.realized_gain) geometryhit=$($f.argmax_matches_latent_geometry_rate) qdelta=$($f.max_quadrature_refinement_difference_first_64)" }
$s.refpool.correlated_gaussian_fixture.conditional_log_density_max_abs_error
foreach ($f in $s.paired_gate.adaptive_vs_every_constant_by_family.PSObject.Properties) { foreach ($m in @('"'nll_gain_nats','accuracy_gain')) { "'$a=foreach ($act in $f.Value.PSObject.Properties) { $act.Value.PSObject.Properties[$m].Value.simultaneous_95ci[0] }; Write-Output "$($f.Name) $m min L=$(( $a | Measure-Object -Minimum).Minimum)" } }'
```

**Aggregated Output:**

```text
{
    "random_window":  {
                          "nll_gain_nats":  {
                                                "gain":  0.46899513443594015,
                                                "simultaneous_95ci":  [
                                                                          0.4575633919073533,
                                                                          0.480426876964527
                                                                      ]
                                            },
                          "accuracy_gain":  {
                                                "gain":  0.3324381510416667,
                                                "simultaneous_95ci":  [
                                                                          0.31874555315377395,
                                                                          0.3461307489295594
                                                                      ]
                                            }
                      },
    "nearest_window":  {
                           "nll_gain_nats":  {
                                                 "gain":  0.4627543284629089,
                                                 "simultaneous_95ci":  [
                                                                           0.4513228683461323,
                                                                           0.4741857885796856
                                                                       ]
                                             },
                           "accuracy_gain":  {
                                                 "gain":  0.3214518229166667,
                                                 "simultaneous_95ci":  [
                                                                           0.30779082889871273,
                                                                           0.33511281693462064
                                                                       ]
                                             }
                       },
    "fixed_best_on_dev":  {
                              "nll_gain_nats":  {
                                                    "gain":  0.46186870905272204,
                                                    "simultaneous_95ci":  [
                                                                              0.45033848933694715,
                                                                              0.47339892876849693
                                                                          ]
                                                },
                              "accuracy_gain":  {
                                                    "gain":  0.3197428385416667,
                                                    "simultaneous_95ci":  [
                                                                              0.3061507036608312,
                                                                              0.3333349734225022
                                                                          ]
                                                }
                          },
    "fixed_best_oracle":  {
                              "nll_gain_nats":  {
                                                    "gain":  0.4584874273807906,
                                                    "simultaneous_95ci":  [
                                                                              0.4388827341988701,
                                                                              0.47798846927517497
                                                                          ]
                                                },
                              "accuracy_gain":  {
                                                    "gain":  0.3181966145833333,
                                                    "simultaneous_95ci":  [
                                                                              0.2947103844376176,
                                                                              0.34168284472904903
                                                                          ]
                                                }
                          }
}
jump_interval nll=0.04742462336745476 acc=0.98291015625 exp=0.6434874393572626 gain=0.6457225571924906 geometryhit=0.998291015625 qdelta=1.07440167229633E-06
selective_fade nll=0.0032445816875599463 acc=0.998779296875 exp=0.6868842140599762 gain=0.6899025988723853 geometryhit=0.998046875 qdelta=1.99464586891906E-06
cfo_alias nll=0.09401412796044234 acc=0.96435546875 exp=0.5951526824567809 gain=0.5991330525995029 geometryhit=0.999755859375 qdelta=8.40953497283081E-07
6.66133814775094E-16
jump_interval nll_gain_nats min L=0.463873052070104
jump_interval accuracy_gain min L=0.32542743507087
selective_fade nll_gain_nats min L=0.434893847577153
selective_fade accuracy_gain min L=0.231978355451679
cfo_alias nll_gain_nats min L=0.417881302949352
cfo_alias accuracy_gain min L=0.326725362790304
```

### Event #52: `item.completed` — `file_change` (`item_50`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\reply_r2.md",
    "kind": "add"
  }
]
```

### Event #53: `item.completed` — `command_execution` (`item_51`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n 'dev_lock|test_mothers_per_family' research/stage4_math_and_design/M5/results_m5.json -m 4
Get-Item -LiteralPath 'research/stage4_math_and_design/M5/results_m5.json' | Select-Object Length,LastWriteTime"
```

**Aggregated Output:**

```text
15698:      "test_mothers_per_family": 4096,
15699:      "dev_lock": "maximize mean predicted EIG within each public family; smallest candidate ID breaks ties",

Length LastWriteTime      
------ -------------      
587817 2026/11/26 20:01:51


```

### Event #54: `item.completed` — `file_change` (`item_52`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\THEOREM_M5.md",
    "kind": "update"
  }
]
```

### Event #55: `item.completed` — `command_execution` (`item_42`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$m5 = (Resolve-Path -LiteralPath '"'research/stage4_math_and_design/M5').Path
"'$env:UV_CACHE_DIR = Join-Path $m5 '"'.uv-cache'
"'$env:UV_PYTHON_INSTALL_DIR = Join-Path $m5 '"'.uv-python'
"'$env:TMP = $m5
$env:TEMP = $m5
$env:UV_LINK_MODE = '"'copy'
uv run --no-project --offline --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M5/verify_m5.py"
```

**Aggregated Output:**

```text
adaptive jump_interval
adaptive selective_fade
adaptive cfo_alias
flat_minus3: 2
flat_minus3: 4_s17
flat_minus3: 8
flat_minus3: 12
flat_minus3: 16
flat_minus3: canonical_continuous
flat_minus3: full_iq
flat_plus2: 2
flat_plus2: 4_s17
flat_plus2: 8
flat_plus2: 12
flat_plus2: 16
flat_plus2: canonical_continuous
flat_plus2: full_iq
flat_plus24: 2
flat_plus24: 4_s17
flat_plus24: 8
flat_plus24: 12
flat_plus24: 16
flat_plus24: canonical_continuous
flat_plus24: full_iq
selective_fade: 2
selective_fade: 4_s17
selective_fade: 8
selective_fade: 12
selective_fade: 16
selective_fade: canonical_continuous
selective_fade: full_iq
Wrote results_m5.json
```

### Event #56: `item.completed` — `command_execution` (`item_53`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$ErrorActionPreference='"'Stop'
"'$r=Get-Content -Encoding utf8 research/stage4_math_and_design/M5/results_m5.json -Raw | ConvertFrom-Json
$s=$r.multi_geometry_adaptive_suite
if ($s.design.dev_lock -notlike '"'minimize realized*' -or "'!$s.paired_gate.strict_nll_and_accuracy_gate_passed -or $s.paired_gate.bonferroni_elementary_intervals -ne 30) { throw '"'Results are not from final design' }
foreach ("'$p in $s.aggregate_post_reveal.PSObject.Properties) { '"'{0}: {1:F8} {2:F6}%' -f "'$p.Name,$p.Value.nll,($p.Value.accuracy*100) }
foreach ($p in $s.paired_gate.pooled_adaptive_advantages.PSObject.Properties) { $n=$p.Value.nll_gain_nats; $a=$p.Value.accuracy_gain; '"'{0}: NLL {1:F5} [{2:F5}, {3:F5}], acc {4:F3} [{5:F3}, {6:F3}] pp' -f "'$p.Name,$n.gain,$n.simultaneous_95ci[0],$n.simultaneous_95ci[1],($a.gain*100),($a.simultaneous_95ci[0]*100),($a.simultaneous_95ci[1]*100) }
foreach ($f in $s.per_family) { $n = $f.dev_post_reveal_nll_by_candidate; if (($n | Measure-Object -Minimum).Minimum -ne $n[$f.fixed_dev_candidate_index]) { throw '"'Incorrect dev lock' }; \"dev "'$($f.family) a=$($f.fixed_dev_candidate_index), NLL $($n -join '"', '); choices "'$($f.policy_metrics.max_EIG_window.selected_counts -join '"', '); M3 error "'$($f.m3_shared_pool_eight_selected_rows_max_abs_error)" }
$r.round2_packet.fixture_hex
$s.refpool.correlated_gaussian_fixture.conditional_log_density_max_abs_error
$s.contract.catalogue_sha256'
```

**Aggregated Output:**

```text
max_EIG_window: 0.04822778 98.201497%
random_window: 0.51722291 64.957682%
nearest_window: 0.51098211 66.056315%
fixed_best_on_dev: 0.51009649 66.227214%
fixed_best_oracle: 0.50671521 66.381836%
random_window: NLL 0.46900 [0.45756, 0.48043], acc 33.244 [31.875, 34.613] pp
nearest_window: NLL 0.46275 [0.45132, 0.47419], acc 32.145 [30.779, 33.511] pp
fixed_best_on_dev: NLL 0.46187 [0.45034, 0.47340], acc 31.974 [30.615, 33.333] pp
fixed_best_oracle: NLL 0.45849 [0.43888, 0.47799], acc 31.820 [29.471, 34.168] pp
Incorrect dev lock
所在位置 行:8 字符: 158
+ ... n[$f.fixed_dev_candidate_index]) { throw 'Incorrect dev lock' }; "dev ...
+                                        ~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : OperationStopped: (Incorrect dev lock:String) [], RuntimeException
    + FullyQualifiedErrorId : Incorrect dev lock
 
```

### Event #57: `item.completed` — `command_execution` (`item_54`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$r=Get-Content -Encoding utf8 research/stage4_math_and_design/M5/results_m5.json -Raw | ConvertFrom-Json
$s=$r.multi_geometry_adaptive_suite
foreach ($f in $s.per_family) { $n=$f.dev_post_reveal_nll_by_candidate; $i=$f.fixed_dev_candidate_index; $mi=($n | Measure-Object -Minimum).Minimum; Write-Output "dev $($f.family) a=$i selected=$($n[$i]) min=$mi NLL $($n -join '"', '), diff="'$($n[$i]-$mi), choices $($f.policy_metrics.max_EIG_window.selected_counts -join '"','), M3 err="'$($f.m3_shared_pool_eight_selected_rows_max_abs_error)" }
$r.round2_packet.fixture_hex
$s.refpool.correlated_gaussian_fixture.conditional_log_density_max_abs_error
$s.contract.catalogue_sha256'
```

**Aggregated Output:**

```text
dev jump_interval a=3 selected=0.5280994408083399 min=0.52809944080834 NLL 0.531845603451522, 0.5315112336212418, 0.5358733600228428, 0.5280994408083399, diff=-0.0000000000000001, choices 1001,1017,1066,1012, M3 err=3.99680288865056E-15
dev selective_fade a=1 selected=0.4454503270749213 min=0.445450327074921 NLL 0.4645845189242045, 0.4454503270749213, 0.4551018252268406, 0.4850648492215489, diff=0.0000000000000003, choices 941,1037,1061,1057, M3 err=3.5527136788005E-15
dev cfo_alias a=2 selected=0.5263324049809344 min=0.526332404980934 NLL 0.5494424365743262, 0.5475045837717645, 0.5263324049809344, 0.5489755988352395, diff=0.0000000000000004, choices 1042,1005,1020,1029, M3 err=3.5527136788005E-15
1008104800100000
6.66133814775094E-16
0127cb34f1451c10cd00554674750893bceb09bfbc4f483ef1780cff67343cdc
```

### Event #58: `item.completed` — `file_change` (`item_55`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M5\\reply_r2.md",
    "kind": "update"
  }
]
```

### Event #59: `item.completed` — `command_execution` (`item_56`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$ErrorActionPreference='"'Stop'
"'$m5=(Resolve-Path -LiteralPath '"'research/stage4_math_and_design/M5').Path
"'$r=Get-Content -Encoding utf8 (Join-Path $m5 '"'results_m5.json') -Raw | ConvertFrom-Json
"'$s=$r.multi_geometry_adaptive_suite
if ($r.protocol.numpy -ne '"'2.3.5' -or "'$r.protocol.scipy -ne '"'1.17.0' -or "'$s.design.dev_lock -notlike '"'minimize realized*' -or "'!$s.paired_gate.strict_nll_and_accuracy_gate_passed) { throw '"'Invalid final replay' }
if ("'$s.contract.bits_per_old_window -ne 16 -or $s.contract.read_budget -ne 1 -or $s.aggregate_post_reveal.max_EIG_window.n -ne 12288 -or $s.per_family.Count -ne 3) { throw '"'Unexpected suite dimensions' }
foreach ("'$f in $s.per_family) { $n=$f.dev_post_reveal_nll_by_candidate; $i=$f.fixed_dev_candidate_index; if ($n[$i] -gt (($n | Measure-Object -Minimum).Minimum+1e-12)) { throw '"'Incorrect dev lock' }; foreach ("'$p in $f.policy_metrics.PSObject.Properties) { if (($p.Value.selected_counts | Measure-Object -Sum).Sum -ne 4096 -or $p.Value.commit_sha256 -notmatch '"'"'^[a-f0-9]{64}$'"') { throw 'Missing commit or reveal count' } } }
foreach ("'$family in $s.paired_gate.adaptive_vs_every_constant_by_family.PSObject.Properties) { foreach ($action in $family.Value.PSObject.Properties) { foreach ($end in $action.Value.PSObject.Properties) { if ($end.Value.simultaneous_95ci[0] -le 0) { throw '"'Fixed-action lower bound did not clear' } } } }
"'$m3 = (Get-FileHash research/stage4_math_and_design/M3/ref_pool.py -Algorithm SHA256).Hash
$m4 = (Get-FileHash research/stage4_math_and_design/M4/verify_m4.py -Algorithm SHA256).Hash
Write-Output "Peer source hashes still match replay: M3 $($m3 -eq $s.refpool.m3_shared_pool_code_sha256), M4 $($m4 -eq $r.round2_packet.m4_shard_code_sha256)"
$requestedCache=Join-Path $m5 '"'.uv-cache'
"'$actualCache=Get-Item -LiteralPath $requestedCache -Force
if ($actualCache.FullName -cne $requestedCache -or $actualCache.Parent.FullName -cne $m5 -or ($actualCache.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw '"'Temporary cache path check failed' }
uv cache clean --cache-dir research/stage4_math_and_design/M5/.uv-cache
if ("'$LASTEXITCODE -ne 0) { throw '"'Cache cleanup failed' }
Get-ChildItem -LiteralPath "'$m5 -Force | Select-Object Name,Length
git status --short --untracked-files=all -- research/stage4_math_and_design/M5'
```

**Aggregated Output:**

```text
Peer source hashes still match replay: M3 True, M4 False
Clearing cache at: research/stage4_math_and_design/M5/.uv-cache
Removed 4948 files (295.5MiB)

?? research/stage4_math_and_design/M5/THEOREM_M5.md
?? research/stage4_math_and_design/M5/reply_r1.md
?? research/stage4_math_and_design/M5/reply_r2.md
?? research/stage4_math_and_design/M5/results_m5.json
?? research/stage4_math_and_design/M5/verify_m5.py
Name            Length
----            ------
reply_r1.md       4497
reply_r2.md       3732
results_m5.json 589536
THEOREM_M5.md    57161
verify_m5.py     58537


```

### Event #60: `item.completed` — `command_execution` (`item_57`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content -Encoding utf8 research/stage4_math_and_design/M4/verify_m4.py | Select-Object -Skip 193 -First 43
rg -n 'm5-c16|merge|erasure|transcode' research/stage4_math_and_design/M4/THEOREM_M4.md research/stage4_math_and_design/M4/verify_m4.py -m 8
Get-Item research/stage4_math_and_design/M4/verify_m4.py | Select-Object LastWriteTime"
```

**Aggregated Output:**

```text
def pack_dual_cache(encoder, t, measured_residual, nominal_variance, ancillary):
    """M5 16-bit layout. `measured_residual` comes from the radio taps before R3 lift."""
    if not (np.all(np.isfinite(t)) and np.all(t != 0) and np.all(np.isfinite(measured_residual))
            and np.all(measured_residual >= 0) and np.all(np.isfinite(nominal_variance))
            and np.all(nominal_variance > 0)):
        raise ValueError('Unavailable windows must use the separate validity mask')
    legacy = encoder.encode(lift(t, ancillary)).astype(np.uint16)
    turn = legacy >> 2
    local = (np.angle(t * ROOTS[turn].conj()) + np.pi / 4) / (np.pi / 2)
    angle = np.clip(np.floor(64 * local), 0, 63).astype(np.uint16)
    radius = np.searchsorted(RADIAL_EDGES[1:-1], abs(t) / np.sqrt(nominal_variance)).astype(np.uint16)
    quality = np.searchsorted(QUALITY_EDGES[1:-1], measured_residual / nominal_variance).astype(np.uint16)
    packed = ((legacy << 12) | (angle << 6) | (radius << 2) | quality).astype('<u2')
    np.testing.assert_array_equal(packed >> 12, legacy)
    return packed


def polar_id(packet):
    """Drop only the legacy inner bits from the packed word."""
    return ((packet >> 14) << 12) | (packet & 0x0fff)


def polar_noise_emissions(mean, multiplier, tol_order):
    """P(angle, radius, quality | class) for unit nominal thermal variance."""
    node, weight = np.polynomial.legendre.leggauss(tol_order)
    width = 2 * np.pi / 256
    angle = -np.pi / 4 + (np.arange(256)[:, None] + (1 + node) / 2) * width
    radius = RADIAL_EDGES / np.sqrt(multiplier)
    mean = mean / np.sqrt(multiplier)
    radius = np.clip(radius, max(0, abs(mean) - 11), abs(mean) + 11)
    result = np.zeros((256, 16))
    lateral = 4 * abs(mean) * np.sin((angle - np.angle(mean)) / 2) ** 2
    for i, (lo, hi) in enumerate(zip(radius[:-1], radius[1:])):
        edges = np.linspace(lo, hi, 1 + max(1, math.ceil(hi - lo)))
        dr = np.diff(edges)[:, None] / 2
        r = (edges[1:, None] + edges[:-1, None]) / 2 + dr * node
        radial_weight = (dr * weight * r).ravel()
        distance = (r.ravel() - abs(mean)) ** 2 + lateral[..., None] * r.ravel()
        result[:, i] = (np.exp(-distance) @ radial_weight) @ (weight * width / (2 * np.pi))
    mass_quality = np.diff(gammainc(2, QUALITY_EDGES / multiplier))
    probability = (result[..., None] * mass_quality).ravel()
    np.testing.assert_allclose(probability.sum(), 1, atol=1e-9)
    return probability
research/stage4_math_and_design/M4/verify_m4.py:789:            'coverage': 'all enrolled rows retained; uniform fallback on common erasure; locked subset secondary',
research/stage4_math_and_design/M4/THEOREM_M4.md:160:Process each eligible window once. The CPU worker reads causal raw stencils, applies the frozen template and whitening, serializes both views and computes the matched-information physical reference from the sealed observations. The round-1 split and observation whitelist still apply. `encoder_sha256`, manifest/calibration/quantizer revisions, sample-rate/clock map, arm-independent ancillary RNG version, split/recording version, and schema/covariance units are mandatory manifest fields. Hash the header and every shard. Use little-endian fixed dtypes below. Encode erasure as an explicit flag and use zero-filled payload fields as placeholders. Packet word zero and all four legacy inner labels remain legal on valid rows.
research/stage4_math_and_design/M4/THEOREM_M4.md:193:Use a distinct TX basis anchor `b_tx`, initialized to class zero in the declared public TX convention and transported with any TX reindexing. Feed the two invariant views \(i^{-a_1}Y\) and \(i^{-b_{tx}}d\), and express any eligible channel state as \(i^{b_{tx}-a_1}H\). Their cancellations follow directly from the table. For canonical output probabilities apply no gather shift for Q1, a `+b_tx` roll for Q2, and `+a1` for Q3. This gives exact output actions for either canonical host arm independently of its numerical scores when canonical inputs agree. Here `r_rx` is the applied receive gauge, distinct from the inferred anchor. Isolated data-window rotation is a separate M1 synthetic intervention requiring the stated synthetic law. Invalid windows and zero phase pivot use the common erasure path when no valid pivot is present. The verifier tests this wire behavior on simulated real-data timelines and independently checks probability actions of a finite physical likelihood. Final Qwen parity remains a deployment receipt.

LastWriteTime      
-------------      
2026/11/26 20:02:54


```

### Event #61: `item.completed` — `command_execution` (`item_58`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$ErrorActionPreference = '"'Stop'
"'$m5 = (Resolve-Path -LiteralPath '"'research/stage4_math_and_design/M5').Path
"'$base = uv cache dir
$env:UV_CACHE_DIR = Join-Path $m5 '"'.uv-cache'
"'$env:UV_PYTHON_INSTALL_DIR = Join-Path $m5 '"'.uv-python'
"'$env:TMP = $m5
$env:TEMP = $m5
$env:UV_LINK_MODE = '"'copy'
New-Item -ItemType Directory -Force (Join-Path "'$env:UV_CACHE_DIR '"'simple-v21/pypi'),(Join-Path "'$env:UV_CACHE_DIR '"'wheels-v6/pypi'),(Join-Path "'$env:UV_CACHE_DIR '"'archive-v0') | Out-Null
foreach ("'$name in @('"'numpy','scipy')) { Copy-Item -LiteralPath \""'$base/simple-v21/pypi/$name.rkyv" -Destination (Join-Path $env:UV_CACHE_DIR '"'simple-v21/pypi'); Copy-Item -LiteralPath \""'$base/wheels-v6/pypi/$name" -Destination (Join-Path $env:UV_CACHE_DIR '"'wheels-v6/pypi') -Recurse }
foreach ("'$id in @('"'cSnmXl2iWBpNAXGE','3Oxde2-Jkv6KGjnQ')) { Copy-Item -LiteralPath \""'$base/archive-v0/$id" -Destination (Join-Path $env:UV_CACHE_DIR '"'archive-v0') -Recurse }
"'$check = @'"'
import json
import runpy

m5 = runpy.run_path('research/stage4_math_and_design/M5/verify_m5.py')
output = m5['HERE'] / 'results_m5.json'
results = json.loads(output.read_text(encoding='utf-8'))
encoder = m5['load'](m5['MATCHED'] / 'weights' / 'equivariant_b8_s17.npz')
latest = m5['wire_check'](encoder)
previous = results['round2_packet']
assert set(latest) == set(previous)
assert all(latest[k] == previous[k] for k in latest if k "'!= '"'m4_shard_code_sha256')
results['round2_packet'] = latest
output.write_text(json.dumps(results, indent=2, allow_nan=False) + '\\n', encoding='utf-8')
print('Rechecked latest M4 packer; all wire fixtures match:', latest['m4_shard_code_sha256'])
'@
uv run --no-project --offline --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B -c "'$check'
```

**Aggregated Output:**

```text
Installed 2 packages in 7.04s
Rechecked latest M4 packer; all wire fixtures match: 5c831617f9bda2a8eb9f73bc6e6ecc7c90229eb6e749982d2dab4aee81ba6b2b
```

### Event #62: `item.completed` — `command_execution` (`item_59`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$ErrorActionPreference='"'Stop'
"'$m5=(Resolve-Path -LiteralPath '"'research/stage4_math_and_design/M5').Path
"'$requestedCache=Join-Path $m5 '"'.uv-cache'
"'$actualCache=Get-Item -LiteralPath $requestedCache -Force
if ($actualCache.FullName -cne $requestedCache -or $actualCache.Parent.FullName -cne $m5 -or ($actualCache.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw '"'Temporary cache path check failed' }
uv cache clean --cache-dir research/stage4_math_and_design/M5/.uv-cache
if ("'$LASTEXITCODE -ne 0) { throw '"'Cache cleanup failed' }
Get-ChildItem -LiteralPath "'$m5 -Force -Name
$r=Get-Content -Encoding utf8 (Join-Path $m5 '"'results_m5.json') -Raw | ConvertFrom-Json
if ("'$r.round2_packet.m4_shard_code_sha256 -ne '"'5c831617f9bda2a8eb9f73bc6e6ecc7c90229eb6e749982d2dab4aee81ba6b2b' -or "'!$r.multi_geometry_adaptive_suite.paired_gate.strict_nll_and_accuracy_gate_passed) { throw '"'Final artifact verification failed' }
'Final artifact verified'"
```

**Aggregated Output:**

```text
Clearing cache at: research/stage4_math_and_design/M5/.uv-cache
Removed 4947 files (295.5MiB)
reply_r1.md
reply_r2.md
results_m5.json
THEOREM_M5.md
verify_m5.py
Final artifact verified
```

### Event #63: `item.completed` — `agent_message` (`item_60`)

**Max EIG beat all four controls on both post-reveal NLL and accuracy** across 12,288 held-out synthetic recordings. All four candidate IDs won within each of the jump, fading, and CFO-alias families, resolving the Round 1 fixed-window tie.

| Policy | NLL (nat) | Accuracy |
|---|---:|---:|
| **Max EIG** | **0.04823** | **98.20%** |
| Random | 0.51722 | 64.96% |
| Nearest | 0.51098 | 66.06% |
| Fixed best on dev | 0.51010 | 66.23% |
| Retrospective fixed oracle | 0.50672 | 66.38% |

Against the dev-locked policy, max EIG gained **0.46187 nat** and **31.97 percentage points**, with simultaneous paired 95% intervals of **[0.45034, 0.47340] nat** and **[30.62, 33.33] points**. All comparisons cleared the paired gate.

The proof gives the exact NLL condition. Adaptation helps when no constant window is optimal almost surely within a positive-measure set of public strata. With unique maximizers, this is equivalent to positive conditional mutual information between the winning action and the utility vector. Accuracy has a separate criterion and was tested directly.

The **16-bit wire format** preserves the s17 prefix and merges M4’s two lowest radial bins to make room for erasure. The pinned CPU replay verified the conversion against M4’s packer and matched M3’s shared pool on 24 selected reveals within \(4.0\times10^{-15}\). The common 4B/9B/27B and LEMMO contract is specified. Host runs, the matched LEMMO adapter, and held-out SDR qualification remain pending.

[Executive summary](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M5/reply_r2.md) · [Proof and contract](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M5/THEOREM_M5.md) · [CPU verifier](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M5/verify_m5.py) · [Results JSON](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M5/results_m5.json)



### Event #64: `turn.completed`

- Usage: `{"input_tokens": 19740415, "cached_input_tokens": 19468288, "cache_write_input_tokens": 0, "output_tokens": 103241, "reasoning_output_tokens": 27719}`


## Round 2 — Final Output (`reply_r2.md`)

```markdown
**Max EIG beat all four controls on both post-reveal NLL and accuracy** across 12,288 held-out synthetic recordings. All four candidate IDs won within each of the jump, fading, and CFO-alias families, resolving the Round 1 fixed-window tie.

| Policy | NLL (nat) | Accuracy |
|---|---:|---:|
| **Max EIG** | **0.04823** | **98.20%** |
| Random | 0.51722 | 64.96% |
| Nearest | 0.51098 | 66.06% |
| Fixed best on dev | 0.51010 | 66.23% |
| Retrospective fixed oracle | 0.50672 | 66.38% |

Against the dev-locked policy, max EIG gained **0.46187 nat** and **31.97 percentage points**, with simultaneous paired 95% intervals of **[0.45034, 0.47340] nat** and **[30.62, 33.33] points**. All comparisons cleared the paired gate.

The proof gives the exact NLL condition. Adaptation helps when no constant window is optimal almost surely within a positive-measure set of public strata. With unique maximizers, this is equivalent to positive conditional mutual information between the winning action and the utility vector. Accuracy has a separate criterion and was tested directly.

The **16-bit wire format** preserves the s17 prefix and merges M4’s two lowest radial bins to make room for erasure. The pinned CPU replay verified the conversion against M4’s packer and matched M3’s shared pool on 24 selected reveals within \(4.0\times10^{-15}\). The common 4B/9B/27B and LEMMO contract is specified. Host runs, the matched LEMMO adapter, and held-out SDR qualification remain pending.

[Executive summary](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M5/reply_r2.md) · [Proof and contract](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M5/THEOREM_M5.md) · [CPU verifier](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M5/verify_m5.py) · [Results JSON](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M5/results_m5.json)

```
