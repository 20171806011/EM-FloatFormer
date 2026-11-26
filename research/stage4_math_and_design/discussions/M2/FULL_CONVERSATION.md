# Full Verbatim Multi-Round Conversation — Track `M2`

- **CLI Runtime**: Codex CLI (`codex exec`)
- **Session / Thread ID**: `01a21948-8b7b-71d0-a50a-d7c7eaf61018`
- **Round 1 Events**: `101` raw JSONL events (`round1_events.jsonl`)
- **Round 2 Events**: `53` raw JSONL events (`round2_events.jsonl`)
- **Completeness Guarantee**: All user prompts (`prompt_r1.md`, `prompt_r2.md`), all completed reasoning/tool/command events, and final responses (`reply_r1.md`, `reply_r2.md`) are included verbatim below without truncation or paraphrasing.

---

## Round 1 — Initial Mathematical Brief (`prompt_r1.md`)

````markdown
# Track M2 (Round 1): Information-Theoretic Bregman/KL Decomposition, Calibration Forensics & Stage 4 Statistical Gate

You are **Track M2**, part of a 4-track mathematical & architectural multi-agent collaboration preparing **EM-FloatFormer Stage 4** (and rigorously dissecting **Stage 3D**'s empirical score arrays).

## 0. Strict Workspace, Environment, and Immutability Rules
- **Repository Root**: `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`
- **Your Writable Directory**: `research/stage4_math_and_design/M2/` ONLY.
- **FORBIDDEN Edits**: Do NOT modify or delete any files outside `research/stage4_math_and_design/M2/`.
- **Python Environment Guardrail**: System/Anaconda `python` crashes with MKL `libiomp5md.dll` conflicts on this machine. **ALWAYS** execute Python scripts via:
  ```powershell
  uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B <script.py>
  ```

## 1. Grounded Empirical Context (Read These Files Read-Only First)
Inspect only the relevant sections of:
1. `research/stage3d_qwen_signed_phase/recompute_all.py` (lines 49–856: exact definitions of `softmax_temperature`, `score_vectors`, `compute_16way_group_avg`, `compute_representative_readout`, `evaluate_cell_paired`, `work_f_sign_probe`, `work_g_rotation_and_controls`, `work_a_q2_analysis`)
2. `research/stage3d_qwen_signed_phase/PROTOCOL_LOCK.json` (lines 37–216: locked temperatures `temp_q3`, `temp_q2`, dev pilot stats, Work E sample size calculation `G=131`)
3. `research/stage3d_qwen_signed_phase/results/summary_master.json` (stats for `G1..G4`, `work_f_primary_040`, `work_g_primary_040`, `work_a_q2_c4`)
4. `research/stage3d_qwen_signed_phase/results/blind_q3_scores.npz` and `blind_q2_scores.npz` (exact saved raw log-likelihoods `(2, 3, 2, 256, 4)` etc. across all Stage 3D arms, seeds, and cells!)
5. `research/stage3c/work_results/E/verify.py` (if helpful for comparing Stage 3C's sample size lock formula)

### Verified Stage 3D Empirical Facts Requiring Forensic Dissection
- `G1_main_extrap_both` (131 blocks = 8,384 mothers):
  - `phase`: Acc `23.80%`, NLL `1.4452`, Brier `0.7845`, ECE `0.1186` (`temp_q3 = 0.5099, 0.8489, 0.6575`).
  - `zero`: Acc `19.41%`, NLL `1.4809`, Brier `0.8010`, ECE `0.1827` (`temp_q3 = 1.0414, 0.7827, 0.6733`).
  - Paired `phase - zero`: `dAcc = +4.39 pp [+4.11, +4.68]`, `dNLL = +0.0357 [+0.0330, +0.0383]` (`pass_both = True` against `lo95 > +0.02`), YET BOTH arms are WORSE than uniform random guessing (`25.00%` Acc, `ln 4 = 1.3863` NLL), while `direct_s17_table` achieves `64.62%` Acc / `0.9411` NLL!
- `G2` & `G4` (cumulative angle `0.80 rad < \pi/2`): `zero` BEATS `phase` (`G2`: `36.36% / 1.3081` vs `33.88% / 1.3358`; `G4`: `34.96% / 1.3098` vs `33.29% / 1.3347`)!
- `Work G` (`G1`): Offline 16-way group averaging `Pf` lowers NLL (`phase 1.4452 -> 1.4113`; `zero 1.4809 -> 1.4391`) yet DROPS Acc (`phase 23.80% -> 17.05%`; `zero 19.41% -> 16.43%`), whereas `F0` drops NLL similarly on phase (`1.4112`) with `20.88%` Acc!
- `Work F` (`G1 ±0.40`): `phase` has `orientation_gain = +0.0098` (only ~1% of ideal `1.0`) and `swapped_minus_matched_nll = +0.0184`, yet `zero` has `orientation_gain = +0.0050` and `swapped_minus_matched_nll = +0.0074` (how can a zero-phase arm have nonzero orientation gain `+0.0050`?! Hint: check how prompt text encodes `omega = ±0.40` even in the `zero` arm, and check `seed_mean` vs `direct`!).

## 2. Mathematical & Forensic Questions You Must Answer Rigorously
1. **Exact Information-Theoretic Bregman / KL Decomposition of `ΔNLL` (Applied to REAL `blind_q3_scores.npz`)**:
   - Let $p^*(Y \mid c)$ be the exact empirical/analytical Bayes oracle posterior (`full_iq_probs` / `direct_table_probs` / empirical class frequencies over codes `(s, z)`). Derive exact algebraic identities decomposing expected cross-entropy and $\Delta \text{NLL} = \text{NLL}(q_0) - \text{NLL}(q_1)$ into:
     1. **Bayes Irreducible Entropy** $H(Y \mid \dots)$,
     2. **Calibration / Temperature-Scaling Mismatch Gap** (oracle optimal scalar $T^*$ per cell vs dev-locked $T_{\text{dev}}$ — e.g. compute exact post-hoc oracle-$T^*$ recalibrated NLL & ECE on `G1..G4`!),
     3. **Global Marginal Prior Bias** ($\text{KL}(\bar{y} \| \bar{q})$),
     4. **Equivariance-Breaking Penalty** (KL gap between raw `p` and group-averaged `Pf`),
     5. **Genuine Signed CFO Mutual Information / Resolution Gain** (how much of `G1`'s `+0.0357` NLL gain is real Bayes resolution vs simply lower overconfidence / lower ECE `0.1186 vs 0.1827` or sign-text response?!), and explain quantitatively why `zero` BEATS `phase` on `G2` (`0.80 rad`) and `G4` (`0.80 rad`)!
   - Run this exact decomposition on **ALL** cells of `research/stage3d_qwen_signed_phase/results/blind_q3_scores.npz` (`G1`, `G2`, `G3`, `G4`, `H_c4_035`) + `blind_q2_scores.npz`!
2. **Exact Convexity Proof & Threshold Characterization of the `Pf` Paradox (NLL Drops While Acc Collapses)**:
   - Under any finite group $G$ acting equivariantly on observations/codes and labels (`\mathbb{Z}_4 \times \mathbb{Z}_4` in EM-FloatFormer):
     - Prove that for strictly proper log score, group symmetrization $P_f = \frac{1}{|G|} \sum_{g \in G} \rho(g)^{-1} q(g \cdot x)$ ALWAYS satisfies $\mathbb{E}[\text{NLL}(P_f)] \le \mathbb{E}[\text{NLL}(q)]$ with exact equality deficit $\frac{1}{|G|} \sum_{g \in G} \text{KL}(P_f \| q^{(g)})$!
     - Construct necessary and sufficient geometric conditions under which top-1 Accuracy strictly DECREASES under `Pf`, and empirically measure on `G1..G4` of `blind_q3_scores.npz` how logit-space geometric averaging (geometric mean / log-probability group averaging followed by normalization!) behaves compared to probability-space `Pf`! (Does geometric group averaging fix or change the Acc/NLL behavior on `G1..G4`? Check!)
3. **Formalizing Stage 4's Uncheatable Three-Part Composite Statistical Gate & Hierarchical Sample-Size / Power Law**:
   - Stage 3D's `dual_practical_threshold` ONLY checked paired `phase - zero > +0.02`, allowing two sub-uniform (`23.80% < 25%`, `1.4452 > ln 4`) predictors to pass!
   - Design the formal **Stage 4 Statistical Gate** covering BOTH synthetic extrapolation and real-data Pluto (`G=16`) / CTTC (`G=72` clusters across `9` sessions):
     1. **Absolute Oracle & Baseline Dominance**: Must beat trivial uniform/zero priors AND achieve strictly positive Bayes Normalized Information Gain $\text{NIG} = \frac{H(Y) - \text{NLL}(q)}{H(Y) - \text{NLL}(\text{oracle})} \ge \eta_{\min}$ AND exceed chance + random-label permutation bounds.
     2. **Calibration & Structural Invariance Gate**: Post-dev-calibration ECE bound, Brier bound, and max equivariance violation / idempotence bound (`\le 10^{-5}`).
     3. **Hierarchical Cluster + Seed Inference**: Derive exact Type-I error and power formulas under the Variance Components model $D_{b, s} = \mu + u_b + \nu_s + e_{b, s}$ (mother correlation inside blocks + multi-seed connector variance `s \in \{1..3\}` + finite clusters $G \in \{16, 72, 131\}$), derive exact Holm-Bonferroni / Intersection-Union (IUT) Type-I protection, and compute required block counts $B$ for Stage 4 synthetic & real-data tests given Stage 3D's measured variance components!
4. **Mandatory CPU Empirical Verification (`verify_m2.py`)**:
   - Write `research/stage4_math_and_design/M2/verify_m2.py` (using `uv run --python 3.12 ...`) that loads REAL `blind_q3_scores.npz` & `blind_q2_scores.npz` (and checks exact agreement with `summary_master.json`), computes the complete KL/Bregman decomposition across all 5 cells (`G1..G4`, `H_c4_035`), compares arithmetic vs geometric (logit) group averaging, runs Monte Carlo + noncentral-$t$ power verification for the variance-components Stage 4 gate, and outputs `research/stage4_math_and_design/M2/results_m2.json`.

## 3. Deliverables Required from Track M2
1. `research/stage4_math_and_design/M2/THEOREM_M2.md`
2. `research/stage4_math_and_design/M2/verify_m2.py` (executed cleanly via `uv run`)
3. `research/stage4_math_and_design/M2/results_m2.json`
4. Crisp executive summary in your final message (`reply_r1.md`) reporting exact decomposed Stage 3D numbers, the arithmetic vs geometric `Pf` finding, and exact gate/sample-size recommendations for M1/M3/M4 Round 2.
````


## Round 1 — Execution & Reasoning Trace (`round1_events.jsonl`)

### Event #1: `thread.started` (thread_id=`01a21948-8b7b-71d0-a50a-d7c7eaf61018`)

### Event #2: `turn.started`

### Event #3: `item.completed` — `agent_message` (`item_0`)

I’ll inspect the Stage 3D scoring code, protocol lock, and saved arrays first. Then I’ll reproduce the published metrics and verify which decomposition terms the artifacts can identify. I’ll keep all writes inside `research/stage4_math_and_design/M2/`.

### Event #4: `item.completed` — `command_execution` (`item_2`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n -i 'EM-FloatFormer|stage3d|stage 4 statistical|readonly-scientific-artifact-audit' 'C:\\Users\\李则徐\\.codex\\memories\\MEMORY.md'"
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
32:applies_to: cwd=C:\Users\李则徐\Documents\ChatGPT\EM-FloatFormer; reuse_rule=Reuse the matched-control, reproducibility, and publication approach for this repository; numerical results, commit, hardware status, and receiver outcomes are P4-specific.
38:- rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md (cwd=\\?\C:\Users\李则徐\Documents\ChatGPT\EM-FloatFormer, rollout_path=\\?\C:\Users\李则徐\.codex\sessions\2026\11\17\rollout-2026-11-17T20-54-26-01a1eb4a-230f-7e02-b32d-21348996e25b.jsonl, updated_at=2026-11-18T16:17:48+00:00, thread_id=01a1eb4a-230f-7e02-b32d-21348996e25b, formal P4-D and assessment completed)
42:- EM-FloatFormer, P4, EMFloatFormer2D, C*-equivariance, Hermitian attention, complex RoPE, sparse DMRS, Doppler, CFO, Rapp PA, LDPC, P4D, Holm, sign-flip
48:- rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md (cwd=\\?\C:\Users\李则徐\Documents\ChatGPT\EM-FloatFormer, rollout_path=\\?\C:\Users\李则徐\.codex\sessions\2026\11\17\rollout-2026-11-17T20-54-26-01a1eb4a-230f-7e02-b32d-21348996e25b.jsonl, updated_at=2026-11-18T16:17:48+00:00, thread_id=01a1eb4a-230f-7e02-b32d-21348996e25b, public replay verified)
52:- 20171806011/EM-FloatFormer, 8380084c6dece64119ab72f9c3c1c212e154fb17, publication.verify_p4, p4d_original_audit.py, public_redactions.json, 150 coded rows, ten weight hashes
76:applies_to: cwd=C:\Users\李则徐\Documents\ChatGPT\EM-FloatFormer; reuse_rule=Reuse the diagnostic and UI verification approach; country eligibility, account status, prices, and dates are time- and account-specific.
82:- rollout_summaries/2026-11-17T21-38-49-uldk-google_checkout_debug_youtube_trial_cancellation.md (cwd=\\?\C:\Users\李则徐\Documents\ChatGPT\EM-FloatFormer, rollout_path=C:\Users\李则徐\.codex\sessions\2026\11\17\rollout-2026-11-17T21-38-49-01a1eb72-c399-7cf2-8270-2061ad0e43de.jsonl, updated_at=2026-11-17T21:57:10+00:00, thread_id=01a1eb72-c399-7cf2-8270-2061ad0e43de, logged-in error not observed)
92:- rollout_summaries/2026-11-17T21-38-49-uldk-google_checkout_debug_youtube_trial_cancellation.md (cwd=\\?\C:\Users\李则徐\Documents\ChatGPT\EM-FloatFormer, rollout_path=C:\Users\李则徐\.codex\sessions\2026\11\17\rollout-2026-11-17T21-38-49-01a1eb72-c399-7cf2-8270-2061ad0e43de.jsonl, updated_at=2026-11-17T21:57:10+00:00, thread_id=01a1eb72-c399-7cf2-8270-2061ad0e43de, cancellation verified)
688:- Related skill: skills/readonly-scientific-artifact-audit/SKILL.md. The reviewed source was `C:\Users\李则徐\Downloads\MomentumConservation\INITIAL_REPORT_2026-10-23.md`, SHA-256 `cc2511a32bc3c50c86f9ef60b36fc6e34e2e166402bd6f7a7ab22b650f1d9058`. Review and snapshot are under `outputs/Initial_Report_二轮评审_2026-10-21.md` and `outputs/INITIAL_REPORT_reviewed_snapshot_v2.md` in the task cwd. Findings reflect static inspection and saved results, not new training or full reproduction. [Task 1]
840:- Related skill: skills/readonly-scientific-artifact-audit/SKILL.md. [Task 3][Task 4]
```

### Event #5: `item.completed` — `command_execution` (`item_1`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "Get-Location; rg --files -g AGENTS.md -g '"'!**/.git/**'"' -g '*M2*' -g '*summary_master*' -g '*blind* scores*'; Get-ChildItem research/stage3d_qwen_signed_phase/results | Select-Object Name,Length; Get-ChildItem research/stage4_math_and_design -Force | Select-Object Name,Mode; Get-Command uv | Select-Object Source"
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
 

research\stage3d_qwen_signed_phase\results\summary_master.json
Path                                                             
----                                                             
C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4

Name   : bench_steady.json
Length : 2412


Name   : blind_q2_scores.npz
Length : 334572


Name   : blind_q3_scores.npz
Length : 2560068


Name   : dev_pilot_lock.json
Length : 2767


Name   : dev_pilot_tables.npz
Length : 48163


Name   : direct_vs_table_parity.json
Length : 1735


Name   : raw_q2_gpu.npz
Length : 261760


Name   : raw_q3_gpu.npz
Length : 336535


Name   : raw_replay_gpu.npz
Length : 222396


Name   : replay_c123_scores.npz
Length : 246607


Name   : smoke_audit.json
Length : 11834


Name   : summary_master.json
Length : 109343


Name   : training_summary.json
Length : 16188


Name   : train_summary_phase_s202611.json
Length : 2487


Name   : train_summary_phase_s202622.json
Length : 2490


Name   : train_summary_phase_s202633.json
Length : 2491


Name   : train_summary_zero_s202611.json
Length : 2486


Name   : train_summary_zero_s202622.json
Length : 2485


Name   : train_summary_zero_s202633.json
Length : 2487


Name   : work_f_host_035_phase.json
Length : 70094


Name   : work_f_host_035_zero.json
Length : 70115


Name : discussions
Mode : d-----


Name : M1
Mode : d-----


Name : M2
Mode : d-----


Name : M3
Mode : d-----


Name : M4
Mode : d-----


Name : __pycache__
Mode : d-----


Name : transcript_utils.py
Mode : -a----


Source : C:\qazwsx\Scripts\uv.exe



```

### Event #6: `item.completed` — `command_execution` (`item_3`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$i=0; Get-Content research/stage3d_qwen_signed_phase/recompute_all.py | ForEach-Object { $i++; if($i -ge 49 -and $i -le 440){ '"'{0}: {1}' -f "'$i,$_ } }'
```

**Aggregated Output:**

```text
49: def softmax_temperature(raw_ll: np.ndarray, temp: float) -> np.ndarray:
50:     scaled = np.asarray(raw_ll, dtype=np.float64) / float(temp)
51:     scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
52:     ex = np.exp(scaled)
53:     return ex / np.sum(ex, axis=-1, keepdims=True)
54: 
55: 
56: def score_vectors(probs: np.ndarray, labels: np.ndarray) -> dict:
57:     """Compute fractional-tie hits, NLL (clipped at 1e-12), Brier, and 10-bin ECE."""
58:     p = np.asarray(probs, dtype=np.float64)
59:     y = np.asarray(labels, dtype=np.int64)
60:     n = len(y)
61:     if p.shape != (n, 4):
62:         raise ValueError(f"Expected {(n, 4)}, got {p.shape}")
63:     max_p = np.max(p, axis=1, keepdims=True)
64:     ties = p == max_p
65:     tie_counts = ties.sum(axis=1)
66:     hits = ties[np.arange(n), y].astype(np.float64) / tie_counts.astype(np.float64)
67:     n_ties = int((tie_counts > 1).sum())
68: 
69:     p_true_raw = p[np.arange(n), y]
70:     n_clipped = int((p_true_raw < 1e-12).sum())
71:     p_true = np.clip(p_true_raw, 1e-12, 1.0)
72:     nll = -np.log(p_true)
73: 
74:     one_hot = np.zeros_like(p)
75:     one_hot[np.arange(n), y] = 1.0
76:     brier = np.sum((p - one_hot) ** 2, axis=1)
77: 
78:     confs = np.max(p, axis=1)
79:     edges = np.linspace(0.0, 1.0, 11)
80:     ece = 0.0
81:     for i in range(10):
82:         mask = (confs >= edges[i]) & (confs <= edges[i + 1]) if i == 0 else (confs > edges[i]) & (confs <= edges[i + 1])
83:         cnt = int(mask.sum())
84:         if cnt > 0:
85:             ece += (cnt / float(n)) * abs(float(hits[mask].mean()) - float(confs[mask].mean()))
86: 
87:     return {
88:         "n": int(n),
89:         "acc": float(hits.mean()),
90:         "nll": float(nll.mean()),
91:         "brier": float(brier.mean()),
92:         "ece": float(ece),
93:         "tie_count": n_ties,
94:         "clipped_count": n_clipped,
95:         "per_item_hit": hits,
96:         "per_item_nll": nll,
97:     }
98: 
99: 
100: def paired_t_ci(block_diffs: np.ndarray) -> dict:
101:     arr = np.asarray(block_diffs, dtype=np.float64)
102:     b = len(arr)
103:     mean = float(arr.mean())
104:     sd = float(arr.std(ddof=1)) if b > 1 else 0.0
105:     se = sd / math.sqrt(b) if b > 0 else 0.0
106:     t_crit = float(student_t.ppf(0.975, df=b - 1)) if b > 1 else 0.0
107:     return {
108:         "n_blocks": int(b),
109:         "mean": mean,
110:         "block_sd": sd,
111:         "se": se,
112:         "t_crit_95": t_crit,
113:         "lo95": mean - t_crit * se,
114:         "hi95": mean + t_crit * se,
115:     }
116: 
117: 
118: def block_bootstrap_ci(block_diffs: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
119:     arr = np.asarray(block_diffs, dtype=np.float64)
120:     b = len(arr)
121:     rng = np.random.default_rng(seed)
122:     idx = rng.integers(0, b, size=(resamples, b))
123:     boot = arr[idx].mean(axis=1)
124:     lo, hi = np.quantile(boot, [0.025, 0.975])
125:     return {
126:         "n_blocks": int(b),
127:         "resamples": int(resamples),
128:         "seed": int(seed),
129:         "mean": float(arr.mean()),
130:         "lo95": float(lo),
131:         "hi95": float(hi),
132:     }
133: 
134: 
135: def compute_16way_group_avg(table_256: np.ndarray) -> np.ndarray:
136:     """Offline Work G 16-way C4 x C4 alignment average on a (..., 256, 4) Q3_repeat table."""
137:     orig_shape = table_256.shape
138:     assert orig_shape[-2:] == (256, 4)
139:     flat = table_256.reshape(-1, 256, 4)
140:     out = np.zeros_like(flat)
141:     for z1 in range(16):
142:         for z2 in range(16):
143:             acc = np.zeros((flat.shape[0], 4), dtype=np.float64)
144:             for a in range(4):
145:                 rz1 = (z1 + 4 * a) % 16
146:                 for b in range(4):
147:                     rz2 = (z2 + 4 * b) % 16
148:                     acc += np.roll(flat[:, 16 * rz1 + rz2, :], -a, axis=-1)
149:             out[:, 16 * z1 + z2, :] = acc / 16.0
150:     return out.reshape(orig_shape)
151: 
152: 
153: def compute_representative_readout(table_256: np.ndarray) -> np.ndarray:
154:     """Offline Work G single-representative F0 equivariant readout using canonical (z1%4, z2%4)."""
155:     orig_shape = table_256.shape
156:     assert orig_shape[-2:] == (256, 4)
157:     flat = table_256.reshape(-1, 256, 4)
158:     out = np.zeros_like(flat)
159:     for z1 in range(16):
160:         turn1, inner1 = divmod(z1, 4)
161:         for z2 in range(16):
162:             _, inner2 = divmod(z2, 4)
163:             out[:, 16 * z1 + z2, :] = np.roll(flat[:, 16 * inner1 + inner2, :], turn1, axis=-1)
164:     return out.reshape(orig_shape)
165: 
166: 
167: def evaluate_cell_paired(
168:     raw_ll_table: np.ndarray,
169:     temps: np.ndarray,
170:     codes: np.ndarray,
171:     targets: np.ndarray,
172:     direct_probs: np.ndarray,
173:     full_iq_probs: np.ndarray,
174:     seeds: Sequence[int],
175:     is_primary: bool = False,
176: ) -> dict:
177:     """Evaluate paired phase vs zero arms over (signs x rotations x mothers) and 3 seeds."""
178:     # raw_ll_table: (2, 3, 2, 256, 4); codes/targets: (2, 3, M)
179:     n_arms, n_seeds, n_signs, _, _ = raw_ll_table.shape
180:     assert n_arms == 2 and n_signs == 2
181:     n_rot = codes.shape[1]
182:     n_mothers = codes.shape[2]
183:     assert n_mothers % BLOCK_SIZE == 0
184:     n_blocks = n_mothers // BLOCK_SIZE
185: 
186:     probs_table = np.zeros_like(raw_ll_table, dtype=np.float64)
187:     for a in range(n_arms):
188:         for s_idx in range(n_seeds):
189:             probs_table[a, s_idx] = softmax_temperature(raw_ll_table[a, s_idx], float(temps[a, s_idx]))
190: 
191:     # Shape (2, 3, 2, n_rot, M)
192:     hits = np.zeros((n_arms, n_seeds, n_signs, n_rot, n_mothers), dtype=np.float64)
193:     nlls = np.zeros((n_arms, n_seeds, n_signs, n_rot, n_mothers), dtype=np.float64)
194:     briers = np.zeros((n_arms, n_seeds, n_signs, n_rot, n_mothers), dtype=np.float64)
195:     tie_counts = np.zeros((n_arms, n_seeds), dtype=int)
196:     clip_counts = np.zeros((n_arms, n_seeds), dtype=int)
197:     all_probs_flat = [[None] * n_seeds for _ in range(n_arms)]
198:     all_lbl_flat = targets.reshape(-1)
199: 
200:     for a in range(n_arms):
201:         for s_idx in range(n_seeds):
202:             p_list = []
203:             for sign_i in range(n_signs):
204:                 p_sr = probs_table[a, s_idx, sign_i][codes[sign_i]]  # (n_rot, M, 4)
205:                 p_list.append(p_sr)
206:                 sc = score_vectors(p_sr.reshape(-1, 4), targets[sign_i].reshape(-1))
207:                 hits[a, s_idx, sign_i] = sc["per_item_hit"].reshape(n_rot, n_mothers)
208:                 nlls[a, s_idx, sign_i] = sc["per_item_nll"].reshape(n_rot, n_mothers)
209:                 oh = np.zeros_like(p_sr)
210:                 np.put_along_axis(oh, targets[sign_i][..., None], 1.0, axis=-1)
211:                 briers[a, s_idx, sign_i] = np.sum((p_sr - oh) ** 2, axis=-1)
212:                 tie_counts[a, s_idx] += sc["tie_count"]
213:                 clip_counts[a, s_idx] += sc["clipped_count"]
214:             all_probs_flat[a][s_idx] = np.stack(p_list, axis=0).reshape(-1, 4)
215: 
216:     # Pooled over signs, rotations, and 3 seeds equally
217:     # Block aggregation: first mother-level average across (seeds, signs, rotations), then 64-mother block means
218:     mother_hit = hits.mean(axis=(1, 2, 3))  # (2, M)
219:     mother_nll = nlls.mean(axis=(1, 2, 3))  # (2, M)
220:     block_hit = mother_hit.reshape(2, n_blocks, BLOCK_SIZE).mean(axis=2)  # (2, B)
221:     block_nll = mother_nll.reshape(2, n_blocks, BLOCK_SIZE).mean(axis=2)  # (2, B)
222: 
223:     acc_gain_blocks = block_hit[0] - block_hit[1]        # phase - zero (>0 is better)
224:     nll_improv_blocks = block_nll[1] - block_nll[0]      # zero - phase (>0 is better)
225:     acc_t_ci = paired_t_ci(acc_gain_blocks)
226:     nll_t_ci = paired_t_ci(nll_improv_blocks)
227:     acc_boot_ci = block_bootstrap_ci(acc_gain_blocks)
228:     nll_boot_ci = block_bootstrap_ci(nll_improv_blocks)
229: 
230:     # Pooled ECE across all (seed x sign x rot x mother) predictions per arm
231:     pooled_arm_stats = {}
232:     for a, aname in enumerate(("phase", "zero")):
233:         flat_p = np.concatenate(all_probs_flat[a], axis=0)
234:         flat_y = np.tile(all_lbl_flat, n_seeds)
235:         sc_pool = score_vectors(flat_p, flat_y)
236:         pooled_arm_stats[aname] = {
237:             "acc": float(hits[a].mean()),
238:             "nll": float(nlls[a].mean()),
239:             "brier": float(briers[a].mean()),
240:             "ece": float(sc_pool["ece"]),
241:             "tie_count": int(tie_counts[a].sum()),
242:             "clipped_count": int(clip_counts[a].sum()),
243:             "by_sign": {
244:                 "-omega": {"acc": float(hits[a, :, 0].mean()), "nll": float(nlls[a, :, 0].mean())},
245:                 "+omega": {"acc": float(hits[a, :, 1].mean()), "nll": float(nlls[a, :, 1].mean())},
246:             },
247:         }
248: 
249:     # Per-seed breakdown
250:     per_seed = []
251:     for s_idx, seed_val in enumerate(seeds):
252:         b_hit_s = hits[:, s_idx].mean(axis=(1, 2)).reshape(2, n_blocks, BLOCK_SIZE).mean(axis=2)
253:         b_nll_s = nlls[:, s_idx].mean(axis=(1, 2)).reshape(2, n_blocks, BLOCK_SIZE).mean(axis=2)
254:         acc_g_s = b_hit_s[0] - b_hit_s[1]
255:         nll_i_s = b_nll_s[1] - b_nll_s[0]
256:         sc_p = score_vectors(all_probs_flat[0][s_idx], all_lbl_flat)
257:         sc_z = score_vectors(all_probs_flat[1][s_idx], all_lbl_flat)
258:         per_seed.append(
259:             {
260:                 "seed": int(seed_val),
261:                 "phase": {"acc": sc_p["acc"], "nll": sc_p["nll"], "brier": sc_p["brier"], "ece": sc_p["ece"]},
262:                 "zero": {"acc": sc_z["acc"], "nll": sc_z["nll"], "brier": sc_z["brier"], "ece": sc_z["ece"]},
263:                 "acc_gain_phase_minus_zero": paired_t_ci(acc_g_s),
264:                 "nll_improv_zero_minus_phase": paired_t_ci(nll_i_s),
265:             }
266:         )
267: 
268:     seed_acc_gains = [x["acc_gain_phase_minus_zero"]["mean"] for x in per_seed]
269:     seed_nll_improv = [x["nll_improv_zero_minus_phase"]["mean"] for x in per_seed]
270: 
271:     # Analytic baselines on exact same mothers
272:     sc_dir = score_vectors(direct_probs.reshape(-1, 4), all_lbl_flat)
273:     sc_iq = score_vectors(full_iq_probs.reshape(-1, 4), all_lbl_flat)
274: 
275:     # Host minus Direct Readout paired block stats
276:     dir_hit = sc_dir["per_item_hit"].reshape(n_signs, n_rot, n_mothers).mean(axis=(0, 1))
277:     dir_nll = sc_dir["per_item_nll"].reshape(n_signs, n_rot, n_mothers).mean(axis=(0, 1))
278:     dir_blk_hit = dir_hit.reshape(n_blocks, BLOCK_SIZE).mean(axis=1)
279:     dir_blk_nll = dir_nll.reshape(n_blocks, BLOCK_SIZE).mean(axis=1)
280: 
281:     res = {
282:         "blocks": int(n_blocks),
283:         "mothers": int(n_mothers),
284:         "logical_rows_per_arm": int(n_seeds * n_signs * n_rot * n_mothers),
285:         "arms": pooled_arm_stats,
286:         "paired_phase_minus_zero": {
287:             "acc_gain_t95": acc_t_ci,
288:             "acc_gain_boot95": acc_boot_ci,
289:             "nll_improv_t95": nll_t_ci,
290:             "nll_improv_boot95": nll_boot_ci,
291:         },
292:         "per_seed": per_seed,
293:         "seed_dispersion": {
294:             "acc_gain_sd": float(np.std(seed_acc_gains, ddof=1)),
295:             "acc_gain_range": [float(min(seed_acc_gains)), float(max(seed_acc_gains))],
296:             "nll_improv_sd": float(np.std(seed_nll_improv, ddof=1)),
297:             "nll_improv_range": [float(min(seed_nll_improv)), float(max(seed_nll_improv))],
298:         },
299:         "analytic_same_mothers": {
300:             "direct_s17_table": {k: sc_dir[k] for k in ("acc", "nll", "brier", "ece")},
301:             "full_iq_known_param": {k: sc_iq[k] for k in ("acc", "nll", "brier", "ece")},
302:         },
303:         "phase_minus_direct_s17": {
304:             "acc_gain_t95": paired_t_ci(block_hit[0] - dir_blk_hit),
305:             "nll_improv_t95": paired_t_ci(dir_blk_nll - block_nll[0]),
306:         },
307:         "zero_minus_direct_s17": {
308:             "acc_gain_t95": paired_t_ci(block_hit[1] - dir_blk_hit),
309:             "nll_improv_t95": paired_t_ci(dir_blk_nll - block_nll[1]),
310:         },
311:     }
312:     if is_primary:
313:         pass_acc = bool(acc_t_ci["lo95"] > 0.02)
314:         pass_nll = bool(nll_t_ci["lo95"] > 0.02)
315:         res["dual_practical_threshold"] = {
316:             "acc_threshold": 0.02,
317:             "nll_threshold": 0.02,
318:             "acc_t95_lo": float(acc_t_ci["lo95"]),
319:             "nll_t95_lo": float(nll_t_ci["lo95"]),
320:             "pass_acc": pass_acc,
321:             "pass_nll": pass_nll,
322:             "pass_both": bool(pass_acc and pass_nll),
323:         }
324:     return res
325: 
326: 
327: def work_f_sign_probe(
328:     raw_ll_table: np.ndarray,
329:     temps: np.ndarray,
330:     codes_rot0: np.ndarray,
331:     targets_rot0: np.ndarray,
332:     cal_ref: np.ndarray,
333:     cal_mass: np.ndarray,
334:     seeds: Sequence[int],
335:     contra_text_flipped: Optional[np.ndarray] = None,
336:     contra_phase_flipped: Optional[np.ndarray] = None,
337: ) -> dict:
338:     """Work F signed CFO probe: 4-cell factorial, swapped-minus-matched NLL, orientation gain, and single-cache spotlight."""
339:     n_arms, n_seeds, _, _, _ = raw_ll_table.shape
340:     n_mothers = codes_rot0.shape[1]
341:     assert n_mothers % BLOCK_SIZE == 0
342:     n_blocks = n_mothers // BLOCK_SIZE
343: 
344:     overlap = np.minimum(cal_mass[0], cal_mass[1])
345:     weights = overlap / overlap.sum()
346:     target_diff = cal_ref[1] - cal_ref[0]  # D(z)
347:     norm_d2 = float(np.sum(weights[:, None] * (target_diff ** 2)))
348:     norm_d = math.sqrt(norm_d2)
349: 
350:     out: Dict[str, object] = {
351:         "mothers": int(n_mothers),
352:         "blocks": int(n_blocks),
353:         "cal_ref_overlap_mass": float(overlap.sum()),
354:         "cal_ref_odd_norm_sq": norm_d2,
355:         "arms": {},
356:     }
357: 
358:     def _analyze_table(tbl_2_256_4: np.ndarray) -> dict:
359:         contrast = tbl_2_256_4[1] - tbl_2_256_4[0]
360:         gain = float(np.sum(weights[:, None] * contrast * target_diff) / norm_d2)
361:         resid_ratio = float(math.sqrt(np.sum(weights[:, None] * ((contrast - gain * target_diff) ** 2)) / norm_d2))
362:         mean_sign_tv = float(0.5 * np.sum(weights[:, None] * np.abs(contrast)))
363:         rot_tvs = []
364:         for z in range(256):
365:             z1, z2 = divmod(z, 16)
366:             rz = 16 * ((z1 + 4) % 16) + z2
367:             disc = tbl_2_256_4[:, rz, :] - np.roll(tbl_2_256_4[:, z, :], 1, axis=-1)
368:             rot_tvs.append(0.5 * np.abs(disc).sum(axis=-1).mean())
369:         mean_rot_tv = float(np.dot(weights, rot_tvs))
370: 
371:         # 4-cell physical factorial: p[world_s, told_t, m, sector]
372:         p = np.stack([[tbl_2_256_4[t][codes_rot0[s]] for t in range(2)] for s in range(2)], axis=0)
373:         m_eff = (p[0, 1] - p[0, 0] + p[1, 1] - p[1, 0]) / 4.0
374:         w_eff = (p[1, 0] - p[0, 0] + p[1, 1] - p[0, 1]) / 4.0
375:         i_eff = (p[1, 1] - p[1, 0] - p[0, 1] + p[0, 0]) / 4.0
376:         factorial_norm = {
377:             "metadata_M_sq": float(np.mean(np.sum(m_eff * m_eff, axis=-1))),
378:             "wave_W_sq": float(np.mean(np.sum(w_eff * w_eff, axis=-1))),
379:             "interaction_I_sq": float(np.mean(np.sum(i_eff * i_eff, axis=-1))),
380:         }
381: 
382:         cell_nll = np.zeros((2, 2), dtype=np.float64)
383:         cell_acc = np.zeros((2, 2), dtype=np.float64)
384:         idx_m = np.arange(n_mothers)
385:         matched_nll_m = np.zeros(n_mothers, dtype=np.float64)
386:         swapped_nll_m = np.zeros(n_mothers, dtype=np.float64)
387:         matched_hit_m = np.zeros(n_mothers, dtype=np.float64)
388:         swapped_hit_m = np.zeros(n_mothers, dtype=np.float64)
389: 
390:         for s in range(2):
391:             y = targets_rot0[s]
392:             for t in range(2):
393:                 pr = p[s, t]
394:                 sc = score_vectors(pr, y)
395:                 cell_nll[s, t] = sc["nll"]
396:                 cell_acc[s, t] = sc["acc"]
397:                 if s == t:
398:                     matched_nll_m += 0.5 * sc["per_item_nll"]
399:                     matched_hit_m += 0.5 * sc["per_item_hit"]
400:                 else:
401:                     swapped_nll_m += 0.5 * sc["per_item_nll"]
402:                     swapped_hit_m += 0.5 * sc["per_item_hit"]
403: 
404:         diff_blocks = (swapped_nll_m - matched_nll_m).reshape(n_blocks, BLOCK_SIZE).mean(axis=1)
405:         return {
406:             "orientation_gain": gain,
407:             "residual_ratio": resid_ratio,
408:             "mean_sign_tv": mean_sign_tv,
409:             "mean_rotation_10_tv": mean_rot_tv,
410:             "matched_acc": float(matched_hit_m.mean()),
411:             "swapped_acc": float(swapped_hit_m.mean()),
412:             "matched_nll": float(matched_nll_m.mean()),
413:             "swapped_nll": float(swapped_nll_m.mean()),
414:             "cell_nll_s_world_t_told": {
415:                 "--": float(cell_nll[0, 0]),
416:                 "-+": float(cell_nll[0, 1]),
417:                 "+-": float(cell_nll[1, 0]),
418:                 "++": float(cell_nll[1, 1]),
419:             },
420:             "cell_acc_s_world_t_told": {
421:                 "--": float(cell_acc[0, 0]),
422:                 "-+": float(cell_acc[0, 1]),
423:                 "+-": float(cell_acc[1, 0]),
424:                 "++": float(cell_acc[1, 1]),
425:             },
426:             "swapped_minus_matched_nll_t95": paired_t_ci(diff_blocks),
427:             "swapped_minus_matched_nll_boot95": block_bootstrap_ci(diff_blocks),
428:             "factorial_mean_sq_norm": factorial_norm,
429:         }
430: 
431:     # Physical reference anchor on same mothers
432:     out["physical_cal_reference"] = _analyze_table(cal_ref)
433: 
434:     for a, aname in enumerate(("phase", "zero")):
435:         seed_tables = np.stack(
436:             [softmax_temperature(raw_ll_table[a, s_idx], float(temps[a, s_idx])) for s_idx in range(n_seeds)],
437:             axis=0,
438:         )
439:         per_seed_res = []
440:         for s_idx, seed_val in enumerate(seeds):
```

### Event #7: `item.completed` — `command_execution` (`item_5`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$i=0; Get-Content research/stage3d_qwen_signed_phase/recompute_all.py | ForEach-Object { $i++; if($i -ge 441 -and $i -le 856){ '"'{0}: {1}' -f "'$i,$_ } }'
```

**Aggregated Output:**

```text
441:             r = _analyze_table(seed_tables[s_idx])
442:             r["seed"] = int(seed_val)
443:             per_seed_res.append(r)
444: 
445:         # Average probabilities across 3 seeds for pooled behavioral table summary + mean of seed stats
446:         mean_tbl = seed_tables.mean(axis=0)
447:         pooled_tbl_res = _analyze_table(mean_tbl)
448:         out["arms"][aname] = {
449:             "seed_mean_orientation_gain": float(np.mean([x["orientation_gain"] for x in per_seed_res])),
450:             "seed_mean_residual_ratio": float(np.mean([x["residual_ratio"] for x in per_seed_res])),
451:             "seed_mean_sign_tv": float(np.mean([x["mean_sign_tv"] for x in per_seed_res])),
452:             "seed_mean_swapped_minus_matched_nll": float(
453:                 np.mean([x["swapped_minus_matched_nll_t95"]["mean"] for x in per_seed_res])
454:             ),
455:             "seed_mean_matched_acc": float(np.mean([x["matched_acc"] for x in per_seed_res])),
456:             "seed_mean_swapped_acc": float(np.mean([x["swapped_acc"] for x in per_seed_res])),
457:             "seed_averaged_table_metrics": pooled_tbl_res,
458:             "per_seed": per_seed_res,
459:         }
460: 
461:     # Single concrete cache spotlight on seed 0 (202611) where phase arm flips its argmax when told +omega vs -omega
462:     p_s0_tbl = softmax_temperature(raw_ll_table[0, 0], float(temps[0, 0]))
463:     z_s0_tbl = softmax_temperature(raw_ll_table[1, 0], float(temps[1, 0]))
464:     # Search real caches appearing in world +1 with opposite argmax under told - vs told +
465:     best_code = None
466:     best_tv = -1.0
467:     for m_idx in range(n_mothers):
468:         c_pos = int(codes_rot0[1, m_idx])
469:         c_neg = int(codes_rot0[0, m_idx])
470:         if c_pos == c_neg:
471:             arg_neg = int(np.argmax(p_s0_tbl[0, c_pos]))
472:             arg_pos = int(np.argmax(p_s0_tbl[1, c_pos]))
473:             if arg_neg != arg_pos and arg_pos == int(targets_rot0[1, m_idx]):
474:                 tv = float(0.5 * np.abs(p_s0_tbl[1, c_pos] - p_s0_tbl[0, c_pos]).sum())
475:                 if tv > best_tv:
476:                     best_tv = tv
477:                     best_code = (m_idx, c_pos)
478:     if best_code is None:
479:         # Fallback: pick highest sign TV cache appearing in G1
480:         for m_idx in range(n_mothers):
481:             c = int(codes_rot0[1, m_idx])
482:             tv = float(0.5 * np.abs(p_s0_tbl[1, c] - p_s0_tbl[0, c]).sum())
483:             if tv > best_tv:
484:                 best_tv = tv
485:                 best_code = (m_idx, c)
486: 
487:     m_spot, c_spot = best_code
488:     z1_sp, z2_sp = divmod(c_spot, 16)
489:     out["concrete_cache_spotlight_seed0"] = {
490:         "mother_idx": int(m_spot),
491:         "cache_code": int(c_spot),
492:         "z1": int(z1_sp),
493:         "z2": int(z2_sp),
494:         "physical_label_neg_world": int(targets_rot0[0, m_spot]),
495:         "physical_label_pos_world": int(targets_rot0[1, m_spot]),
496:         "cal_ref_neg": [float(v) for v in cal_ref[0, c_spot]],
497:         "cal_ref_pos": [float(v) for v in cal_ref[1, c_spot]],
498:         "phase_arm_told_neg": {
499:             "raw_loglik": [float(v) for v in raw_ll_table[0, 0, 0, c_spot]],
500:             "probs": [float(v) for v in p_s0_tbl[0, c_spot]],
501:             "pred": int(np.argmax(p_s0_tbl[0, c_spot])),
502:         },
503:         "phase_arm_told_pos": {
504:             "raw_loglik": [float(v) for v in raw_ll_table[0, 0, 1, c_spot]],
505:             "probs": [float(v) for v in p_s0_tbl[1, c_spot]],
506:             "pred": int(np.argmax(p_s0_tbl[1, c_spot])),
507:         },
508:         "zero_arm_told_neg": {
509:             "raw_loglik": [float(v) for v in raw_ll_table[1, 0, 0, c_spot]],
510:             "probs": [float(v) for v in z_s0_tbl[0, c_spot]],
511:             "pred": int(np.argmax(z_s0_tbl[0, c_spot])),
512:         },
513:         "zero_arm_told_pos": {
514:             "raw_loglik": [float(v) for v in raw_ll_table[1, 0, 1, c_spot]],
515:             "probs": [float(v) for v in z_s0_tbl[1, c_spot]],
516:             "pred": int(np.argmax(z_s0_tbl[1, c_spot])),
517:         },
518:     }
519: 
520:     # Single-modality contradiction probe on phase arm (if supplied)
521:     if contra_text_flipped is not None and contra_phase_flipped is not None:
522:         # contra_* shapes: (3, 2, 256, 4)
523:         # Index s=0/1 corresponds to the PHYSICAL world / reference sign;
524:         # For contra_text_flipped: phase_slots = s, text = 1-s (so compare against matched s vs phase=s)
525:         # For contra_phase_flipped: text = s, phase_slots = 1-s
526:         tf_res = []
527:         pf_res = []
528:         for s_idx, seed_val in enumerate(seeds):
529:             t_val = float(temps[0, s_idx])
530:             p_match = softmax_temperature(raw_ll_table[0, s_idx], t_val)
531:             p_tflip = softmax_temperature(contra_text_flipped[s_idx], t_val)
532:             p_pflip = softmax_temperature(contra_phase_flipped[s_idx], t_val)
533: 
534:             def _contra_stats(p_alt: np.ndarray) -> dict:
535:                 # TV vs matched (same physical sign) and TV vs full swapped (opposite sign)
536:                 tv_vs_matched = []
537:                 tv_vs_full_swap = []
538:                 nll_list = []
539:                 hit_list = []
540:                 for s in range(2):
541:                     pr = p_alt[s][codes_rot0[s]]
542:                     pr_m = p_match[s][codes_rot0[s]]
543:                     pr_sw = p_match[1 - s][codes_rot0[s]]
544:                     tv_vs_matched.append(0.5 * np.abs(pr - pr_m).sum(axis=-1))
545:                     tv_vs_full_swap.append(0.5 * np.abs(pr - pr_sw).sum(axis=-1))
546:                     sc = score_vectors(pr, targets_rot0[s])
547:                     nll_list.append(sc["per_item_nll"])
548:                     hit_list.append(sc["per_item_hit"])
549:                 return {
550:                     "seed": int(seed_val),
551:                     "acc": float(np.mean(hit_list)),
552:                     "nll": float(np.mean(nll_list)),
553:                     "mean_tv_vs_matched_both": float(np.mean(tv_vs_matched)),
554:                     "mean_tv_vs_swapped_both": float(np.mean(tv_vs_full_swap)),
555:                 }
556: 
557:             tf_res.append(_contra_stats(p_tflip))
558:             pf_res.append(_contra_stats(p_pflip))
559:         out["single_modality_contradiction_phase_arm"] = {
560:             "text_flipped_phase_kept": {
561:                 "mean_acc": float(np.mean([x["acc"] for x in tf_res])),
562:                 "mean_nll": float(np.mean([x["nll"] for x in tf_res])),
563:                 "mean_tv_vs_matched": float(np.mean([x["mean_tv_vs_matched_both"] for x in tf_res])),
564:                 "mean_tv_vs_full_swap": float(np.mean([x["mean_tv_vs_swapped_both"] for x in tf_res])),
565:                 "per_seed": tf_res,
566:             },
567:             "phase_flipped_text_kept": {
568:                 "mean_acc": float(np.mean([x["acc"] for x in pf_res])),
569:                 "mean_nll": float(np.mean([x["nll"] for x in pf_res])),
570:                 "mean_tv_vs_matched": float(np.mean([x["mean_tv_vs_matched_both"] for x in pf_res])),
571:                 "mean_tv_vs_full_swap": float(np.mean([x["mean_tv_vs_swapped_both"] for x in pf_res])),
572:                 "per_seed": pf_res,
573:             },
574:         }
575:     return out
576: 
577: 
578: def work_g_rotation_and_controls(
579:     raw_ll_table: np.ndarray,
580:     temps: np.ndarray,
581:     codes: np.ndarray,
582:     targets: np.ndarray,
583:     rot10_codes: np.ndarray,
584:     rot10_targets: np.ndarray,
585:     indep_raw_ll: np.ndarray,
586:     indep_targets: np.ndarray,
587:     shuffled_perm: np.ndarray,
588:     seeds: Sequence[int],
589: ) -> dict:
590:     """Work G raw rotation equivariance, independent_x3, shuffled_Z, and offline 16-way Pf / F0."""
591:     n_arms, n_seeds, n_signs, _, _ = raw_ll_table.shape
592:     n_mothers = codes.shape[2]
593:     n_blocks = n_mothers // BLOCK_SIZE
594:     out: Dict[str, object] = {"arms": {}}
595: 
596:     for a, aname in enumerate(("phase", "zero")):
597:         raw_hits, raw_nlls = [], []
598:         pf_hits, pf_nlls = [], []
599:         f0_hits, f0_nlls = [], []
600:         shuf_hits, shuf_nlls = [], []
601:         indep_hits, indep_nlls = [], []
602:         indep_tvs = []
603: 
604:         rot01_max_abs, rot01_mean_abs, rot01_fail_020 = [], [], []
605:         rot11_max_abs, rot11_mean_abs, rot11_fail_020 = [], [], []
606:         rot10_max_abs, rot10_mean_abs, rot10_fail_020 = [], [], []
607:         worst_rot_dev = None
608: 
609:         for s_idx in range(n_seeds):
610:             t_val = float(temps[a, s_idx])
611:             p_raw = softmax_temperature(raw_ll_table[a, s_idx], t_val)  # (2, 256, 4)
612:             p_pf = compute_16way_group_avg(p_raw)
613:             p_f0 = compute_representative_readout(p_raw)
614:             p_ind = softmax_temperature(indep_raw_ll[a, s_idx], t_val)
615: 
616:             for sign_i in range(n_signs):
617:                 pr_rot = p_raw[sign_i][codes[sign_i]]     # (3, M, 4)
618:                 pf_rot = p_pf[sign_i][codes[sign_i]]
619:                 f0_rot = p_f0[sign_i][codes[sign_i]]
620:                 pr_10 = p_raw[sign_i][rot10_codes[sign_i]]
621: 
622:                 # Rotation checks vs rot=(0,0)
623:                 d01 = np.abs(pr_rot[1] - pr_rot[0]).max(axis=1)
624:                 d11 = np.abs(pr_rot[2] - np.roll(pr_rot[0], 1, axis=1)).max(axis=1)
625:                 d10 = np.abs(pr_10 - np.roll(pr_rot[0], 1, axis=1)).max(axis=1)
626: 
627:                 rot01_max_abs.append(float(d01.max()))
628:                 rot01_mean_abs.append(float(d01.mean()))
629:                 rot01_fail_020.append(float((d01 > 0.20).mean()))
630: 
631:                 rot11_max_abs.append(float(d11.max()))
632:                 rot11_mean_abs.append(float(d11.mean()))
633:                 rot11_fail_020.append(float((d11 > 0.20).mean()))
634: 
635:                 rot10_max_abs.append(float(d10.max()))
636:                 rot10_mean_abs.append(float(d10.mean()))
637:                 rot10_fail_020.append(float((d10 > 0.20).mean()))
638: 
639:                 idx_worst = int(np.argmax(d11))
640:                 if worst_rot_dev is None or float(d11[idx_worst]) > worst_rot_dev["max_abs_diff"]:
641:                     worst_rot_dev = {
642:                         "seed": int(seeds[s_idx]),
643:                         "sign_idx": int(sign_i),
644:                         "mother_idx": idx_worst,
645:                         "code_00": int(codes[sign_i, 0, idx_worst]),
646:                         "code_11": int(codes[sign_i, 2, idx_worst]),
647:                         "probs_00": [float(v) for v in pr_rot[0, idx_worst]],
648:                         "probs_11": [float(v) for v in pr_rot[2, idx_worst]],
649:                         "max_abs_diff": float(d11[idx_worst]),
650:                     }
651: 
652:                 sc_raw = score_vectors(pr_rot.reshape(-1, 4), targets[sign_i].reshape(-1))
653:                 sc_pf = score_vectors(pf_rot.reshape(-1, 4), targets[sign_i].reshape(-1))
654:                 sc_f0 = score_vectors(f0_rot.reshape(-1, 4), targets[sign_i].reshape(-1))
655:                 raw_hits.append(sc_raw["per_item_hit"].reshape(3, n_mothers))
656:                 raw_nlls.append(sc_raw["per_item_nll"].reshape(3, n_mothers))
657:                 pf_hits.append(sc_pf["per_item_hit"].reshape(3, n_mothers))
658:                 pf_nlls.append(sc_pf["per_item_nll"].reshape(3, n_mothers))
659:                 f0_hits.append(sc_f0["per_item_hit"].reshape(3, n_mothers))
660:                 f0_nlls.append(sc_f0["per_item_nll"].reshape(3, n_mothers))
661: 
662:                 # Shuffled_Z control
663:                 shuf_codes = np.take_along_axis(codes[sign_i], shuffled_perm[sign_i], axis=1)
664:                 pr_shuf = p_raw[sign_i][shuf_codes]
665:                 sc_shuf = score_vectors(pr_shuf.reshape(-1, 4), targets[sign_i].reshape(-1))
666:                 shuf_hits.append(sc_shuf["per_item_hit"])
667:                 shuf_nlls.append(sc_shuf["per_item_nll"])
668: 
669:                 # independent_x3 control
670:                 pr_ind = p_ind[sign_i][codes[sign_i]]
671:                 sc_ind = score_vectors(pr_ind.reshape(-1, 4), indep_targets[sign_i].reshape(-1))
672:                 indep_hits.append(sc_ind["per_item_hit"])
673:                 indep_nlls.append(sc_ind["per_item_nll"])
674:                 indep_tvs.append(0.5 * np.abs(pr_ind - 0.25).sum(axis=-1))
675: 
676:         # Block paired gain of offline 16-way group average vs raw host
677:         raw_m_hit = np.stack(raw_hits, axis=0).mean(axis=(0, 1))
678:         raw_m_nll = np.stack(raw_nlls, axis=0).mean(axis=(0, 1))
679:         pf_m_hit = np.stack(pf_hits, axis=0).mean(axis=(0, 1))
680:         pf_m_nll = np.stack(pf_nlls, axis=0).mean(axis=(0, 1))
681:         pf_blk_acc_gain = (pf_m_hit - raw_m_hit).reshape(n_blocks, BLOCK_SIZE).mean(axis=1)
682:         pf_blk_nll_improv = (raw_m_nll - pf_m_nll).reshape(n_blocks, BLOCK_SIZE).mean(axis=1)
683: 
684:         out["arms"][aname] = {
685:             "raw_rotation_checks": {
686:                 "rot01_hold": {
687:                     "mean_max_abs": float(np.mean(rot01_mean_abs)),
688:                     "worst_max_abs": float(np.max(rot01_max_abs)),
689:                     "frac_gt_0p20": float(np.mean(rot01_fail_020)),
690:                 },
691:                 "rot11_roll": {
692:                     "mean_max_abs": float(np.mean(rot11_mean_abs)),
693:                     "worst_max_abs": float(np.max(rot11_max_abs)),
694:                     "frac_gt_0p20": float(np.mean(rot11_fail_020)),
695:                 },
696:                 "rot10_roll": {
697:                     "mean_max_abs": float(np.mean(rot10_mean_abs)),
698:                     "worst_max_abs": float(np.max(rot10_max_abs)),
699:                     "frac_gt_0p20": float(np.mean(rot10_fail_020)),
700:                 },
701:                 "worst_rot11_example": worst_rot_dev,
702:             },
703:             "controls": {
704:                 "independent_x3": {
705:                     "acc": float(np.mean(indep_hits)),
706:                     "nll": float(np.mean(indep_nlls)),
707:                     "mean_tv_from_uniform": float(np.mean(indep_tvs)),
708:                     "max_tv_from_uniform": float(np.max(indep_tvs)),
709:                 },
710:                 "shuffled_Z": {
711:                     "acc": float(np.mean(shuf_hits)),
712:                     "nll": float(np.mean(shuf_nlls)),
713:                 },
714:             },
715:             "raw_host": {"acc": float(np.mean(raw_hits)), "nll": float(np.mean(raw_nlls))},
716:             "offline_group_avg_16way": {
717:                 "acc": float(np.mean(pf_hits)),
718:                 "nll": float(np.mean(pf_nlls)),
719:                 "pf_minus_raw_acc_t95": paired_t_ci(pf_blk_acc_gain),
720:                 "raw_minus_pf_nll_t95": paired_t_ci(pf_blk_nll_improv),
721:             },
722:             "offline_representative_F0": {
723:                 "acc": float(np.mean(f0_hits)),
724:                 "nll": float(np.mean(f0_nlls)),
725:             },
726:         }
727:     return out
728: 
729: 
730: def work_a_q2_analysis(q2_npz, temps_q2: np.ndarray, seeds: Sequence[int]) -> dict:
731:     """Evaluate all 7 Q2 arms on 512 fresh C4 (omega=+0.35, 2dB, tp=4) mothers vs Work A kernel."""
732:     raw_ll = q2_npz["q2__raw_ll"]             # (2, 3, 7, M, 4)
733:     labels = q2_npz["q2__labels"]             # (M,)
734:     z1 = q2_npz["q2__z1"]
735:     z2 = q2_npz["q2__z2"]
736:     work_a_probs = q2_npz["q2__work_a_ref_probs"]  # (6, M, 4)
737:     n_arms, n_seeds, n_q2_arms, n_mothers, _ = raw_ll.shape
738:     assert n_q2_arms == len(Q2_ARMS)
739: 
740:     probs_all = np.zeros_like(raw_ll, dtype=np.float64)
741:     for a in range(n_arms):
742:         for s_idx in range(n_seeds):
743:             probs_all[a, s_idx] = softmax_temperature(raw_ll[a, s_idx], float(temps_q2[a, s_idx]))
744: 
745:     # Physical Work A kernel baselines
746:     work_a_metrics = {}
747:     work_a_per_arm = {}
748:     for arm_name, ref_idx in Q2_LOGICAL_REF_IDX.items():
749:         sc = score_vectors(work_a_probs[ref_idx], labels)
750:         work_a_per_arm[arm_name] = sc
751:         work_a_metrics[arm_name] = {k: sc[k] for k in ("n", "acc", "nll", "brier", "ece")}
752:     work_a_fresh_diff_acc = work_a_per_arm["fresh_resample"]["per_item_hit"] - work_a_per_arm["valid"]["per_item_hit"]
753:     work_a_fresh_diff_nll = work_a_per_arm["fresh_resample"]["per_item_nll"] - work_a_per_arm["valid"]["per_item_nll"]
754:     work_a_metrics["fresh_minus_valid"] = {
755:         "delta_acc": float(work_a_fresh_diff_acc.mean()),
756:         "delta_acc_se": float(work_a_fresh_diff_acc.std(ddof=1) / math.sqrt(n_mothers)),
757:         "delta_nll": float(work_a_fresh_diff_nll.mean()),
758:         "delta_nll_se": float(work_a_fresh_diff_nll.std(ddof=1) / math.sqrt(n_mothers)),
759:     }
760: 
761:     arms_out = {}
762:     for a, aname in enumerate(("phase", "zero")):
763:         by_q2_arm = {}
764:         per_seed_q2 = []
765:         # Pool 3 seeds cleanly
766:         for qi, qarm in enumerate(Q2_ARMS):
767:             flat_p = probs_all[a, :, qi].reshape(-1, 4)
768:             flat_y = np.tile(labels, n_seeds)
769:             sc = score_vectors(flat_p, flat_y)
770:             by_q2_arm[qarm] = {k: sc[k] for k in ("n", "acc", "nll", "brier", "ece", "tie_count")}
771: 
772:         # Paired comparisons across (seed x mother)
773:         def _pair_delta(arm_b: str, arm_a: str) -> dict:
774:             ia = Q2_ARMS.index(arm_a)
775:             ib = Q2_ARMS.index(arm_b)
776:             pa = probs_all[a, :, ia].reshape(-1, 4)
777:             pb = probs_all[a, :, ib].reshape(-1, 4)
778:             ya = np.tile(labels, n_seeds)
779:             sca = score_vectors(pa, ya)
780:             scb = score_vectors(pb, ya)
781:             tv = 0.5 * np.abs(pb - pa).sum(axis=-1)
782:             max_abs = np.max(np.abs(pb - pa), axis=-1)
783:             conf_inc = pb.max(axis=-1) - pa.max(axis=-1)
784:             hit_delta = scb["per_item_hit"] - sca["per_item_hit"]
785:             nll_delta = scb["per_item_nll"] - sca["per_item_nll"]
786:             # Mother-averaged SE across mothers (clustering seeds per mother!)
787:             hit_m = hit_delta.reshape(n_seeds, n_mothers).mean(axis=0)
788:             nll_m = nll_delta.reshape(n_seeds, n_mothers).mean(axis=0)
789:             return {
790:                 "delta_acc": float(hit_m.mean()),
791:                 "delta_acc_mother_se": float(hit_m.std(ddof=1) / math.sqrt(n_mothers)),
792:                 "delta_nll": float(nll_m.mean()),
793:                 "delta_nll_mother_se": float(nll_m.std(ddof=1) / math.sqrt(n_mothers)),
794:                 "mean_tv": float(tv.mean()),
795:                 "max_tv": float(tv.max()),
796:                 "max_abs_prob_diff": float(max_abs.max()),
797:                 "max_conf_increase": float(conf_inc.max()),
798:                 "corrected": int(np.sum(hit_delta > 0)),
799:                 "new_errors": int(np.sum(hit_delta < 0)),
800:             }
801: 
802:         for s_idx, seed_val in enumerate(seeds):
803:             s_dict = {"seed": int(seed_val), "arms": {}}
804:             for qi, qarm in enumerate(Q2_ARMS):
805:                 sc_s = score_vectors(probs_all[a, s_idx, qi], labels)
806:                 s_dict["arms"][qarm] = {k: sc_s[k] for k in ("acc", "nll", "brier", "ece")}
807:             per_seed_q2.append(s_dict)
808: 
809:         arms_out[aname] = {
810:             "by_arm": by_q2_arm,
811:             "paired_contrasts": {
812:                 "valid_minus_none": _pair_delta("valid", "none"),
813:                 "independent_only_minus_none": _pair_delta("independent_only", "none"),
814:                 "duplicate_copy_dedup_minus_valid": _pair_delta("duplicate_copy_dedup", "valid"),
815:                 "duplicate_copy_raw_minus_valid": _pair_delta("duplicate_copy_raw", "valid"),
816:                 "valid_plus_independent_minus_valid": _pair_delta("valid_plus_independent", "valid"),
817:                 "fresh_resample_minus_valid": _pair_delta("fresh_resample", "valid"),
818:             },
819:             "per_seed": per_seed_q2,
820:         }
821: 
822:     # Concrete multi-arm trajectory example from seed 0 where Work A valid->fresh helps and host phase arm corrects
823:     p_s0 = probs_all[0, 0]  # (7, M, 4)
824:     idx_ex = 0
825:     for m_i in range(n_mothers):
826:         pred_none = int(np.argmax(p_s0[Q2_ARMS.index("none"), m_i]))
827:         pred_val = int(np.argmax(p_s0[Q2_ARMS.index("valid"), m_i]))
828:         pred_fresh = int(np.argmax(p_s0[Q2_ARMS.index("fresh_resample"), m_i]))
829:         truth = int(labels[m_i])
830:         if pred_none != truth and pred_val == truth and pred_fresh == truth:
831:             idx_ex = m_i
832:             break
833: 
834:     ex_traj = {
835:         "mother_idx": int(idx_ex),
836:         "z1": int(z1[idx_ex]),
837:         "z2": int(z2[idx_ex]),
838:         "true_x1": int(labels[idx_ex]),
839:         "phase_seed0_probs_by_arm": {
840:             qarm: [float(v) for v in probs_all[0, 0, qi, idx_ex]] for qi, qarm in enumerate(Q2_ARMS)
841:         },
842:         "zero_seed0_probs_by_arm": {
843:             qarm: [float(v) for v in probs_all[1, 0, qi, idx_ex]] for qi, qarm in enumerate(Q2_ARMS)
844:         },
845:         "work_a_ref_probs": {
846:             qarm: [float(v) for v in work_a_probs[ridx, idx_ex]] for qarm, ridx in Q2_LOGICAL_REF_IDX.items()
847:         },
848:     }
849: 
850:     return {
851:         "condition": {"omega": 0.35, "h": 2, "ell": 2, "t_p": 4.0, "snr_db": 2.0, "sigma_phi": 0.10},
852:         "mothers": int(n_mothers),
853:         "work_a_physical_reference": work_a_metrics,
854:         "arms": arms_out,
855:         "concrete_trajectory_example": ex_traj,
856:     }
```

### Event #8: `item.completed` — `command_execution` (`item_4`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$i=0; Get-Content research/stage3d_qwen_signed_phase/PROTOCOL_LOCK.json | ForEach-Object { $i++; if($i -ge 37 -and $i -le 216){ '"'{0}: {1}' -f "'$i,$_ } }; Get-ChildItem research/stage4_math_and_design/M2 -Force; git status --short; @('"'AGENTS.md','research/AGENTS.md','research/stage4_math_and_design/AGENTS.md','research/stage4_math_and_design/M2/AGENTS.md') | ForEach-Object { if(Test-Path -LiteralPath "'$_){ Get-Content -LiteralPath $_ } }'
```

**Aggregated Output:**

```text
37:   "checkpoints": {
38:     "phase_s202611": {
39:       "filename": "connector_phase_s202611.pt",
40:       "checkpoint_sha256": "088d4ad34e1152ff000959a37bf784a8f45e326bdd2e483fdf755f82405f7866",
41:       "best_epoch": 7,
42:       "best_sel_score": 1.3809686587691041,
43:       "temp_q3": 0.5098932035670315,
44:       "temp_q2": 3.9999830531425227,
45:       "dev_q3_cal_nll": 1.375347147854539,
46:       "dev_q2_cal_nll": 1.3866422230228832,
47:       "host_sha256_after": "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d"
48:     },
49:     "phase_s202622": {
50:       "filename": "connector_phase_s202622.pt",
51:       "checkpoint_sha256": "ba275787201abf5f1837bd9ebcc959dab24746f9ffa1a08a5df93b0e8a263ae0",
52:       "best_epoch": 8,
53:       "best_sel_score": 1.338227526868415,
54:       "temp_q3": 0.8489407339257738,
55:       "temp_q2": 1.1069058926498243,
56:       "dev_q3_cal_nll": 1.285269302113726,
57:       "dev_q2_cal_nll": 1.3565327978349748,
58:       "host_sha256_after": "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d"
59:     },
60:     "phase_s202633": {
61:       "filename": "connector_phase_s202633.pt",
62:       "checkpoint_sha256": "75f0022894bde60b6e69673d670604b4f0a8cca7847d8df22339e36ba3ddd92f",
63:       "best_epoch": 8,
64:       "best_sel_score": 1.345658880106861,
65:       "temp_q3": 0.6574859852227614,
66:       "temp_q2": 1.0038161968358967,
67:       "dev_q3_cal_nll": 1.3001861761732214,
68:       "dev_q2_cal_nll": 1.3714957209402086,
69:       "host_sha256_after": "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d"
70:     },
71:     "zero_s202611": {
72:       "filename": "connector_zero_s202611.pt",
73:       "checkpoint_sha256": "3d09eded33b58b928191e54d344381ae141b70473d0409cd660d259ec742b77c",
74:       "best_epoch": 8,
75:       "best_sel_score": 1.360639128458629,
76:       "temp_q3": 1.0413709772961914,
77:       "temp_q2": 1.2209637372368802,
78:       "dev_q3_cal_nll": 1.3349880521106885,
79:       "dev_q2_cal_nll": 1.3793762964672136,
80:       "host_sha256_after": "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d"
81:     },
82:     "zero_s202622": {
83:       "filename": "connector_zero_s202622.pt",
84:       "checkpoint_sha256": "a8eb55bfad0d266cbf7e9dc9712b2d32b162374a39ab2898a7f54fa8724c3e98",
85:       "best_epoch": 8,
86:       "best_sel_score": 1.3124549620599506,
87:       "temp_q3": 0.7826950522516535,
88:       "temp_q2": 1.4118146530519748,
89:       "dev_q3_cal_nll": 1.233350441464049,
90:       "dev_q2_cal_nll": 1.3069782597576265,
91:       "host_sha256_after": "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d"
92:     },
93:     "zero_s202633": {
94:       "filename": "connector_zero_s202633.pt",
95:       "checkpoint_sha256": "ea7aebcc6d53bc7161f07ed38c0d5b21e473ff8b56aa3b5af1db61f80f9eb4b3",
96:       "best_epoch": 8,
97:       "best_sel_score": 1.341565335646975,
98:       "temp_q3": 0.6733062486614652,
99:       "temp_q2": 1.2734733247377854,
100:       "dev_q3_cal_nll": 1.292657943214399,
101:       "dev_q2_cal_nll": 1.3811923802166672,
102:       "host_sha256_after": "f39c9e1b7a5719c1397dcdce42d221215fb20edadb881d2f81e745673636f64d"
103:     }
104:   },
105:   "dev_pilot": {
106:     "namespace": 940202,
107:     "blocks": 16,
108:     "mothers": 1024,
109:     "pilot_block_seeds": [
110:       2338220280,
111:       1372146723,
112:       1251899241,
113:       1019832785,
114:       2632972314,
115:       1698518890,
116:       2353612898,
117:       1409965763,
118:       3844068644,
119:       2792309556,
120:       1248517795,
121:       579468612,
122:       575362531,
123:       3621001698,
124:       787505858,
125:       3687402811
126:     ],
127:     "observed_acc_diff_mean": -0.03352864583333334,
128:     "observed_acc_diff_block_sd": 0.014638601593173158,
129:     "observed_nll_diff_mean": -0.02437774790791658,
130:     "observed_nll_diff_block_sd": 0.013414584590285893,
131:     "chi2_upper_multiplier_df15_alpha05": 1.437305501981677,
132:     "conservative_upper_sd_hit": 0.02104014261118552,
133:     "conservative_upper_sd_nll": 0.019280856238416536,
134:     "work_e_scenario": {
135:       "q": 0.35,
136:       "rho": 0.01,
137:       "nll_mother_sd": 0.5,
138:       "pilot_block_sd": [
139:         0.02104014,
140:         0.01928086
141:       ],
142:       "sd": {
143:         "accuracy": 0.09441431300390846,
144:         "nll": 0.07979465834252315
145:       },
146:       "normal_by_endpoint": {
147:         "accuracy": 129,
148:         "nll": 52
149:       },
150:       "t_by_endpoint": {
151:         "accuracy": 131,
152:         "nll": 54
153:       },
154:       "normal_blocks": 129,
155:       "normal_mothers": 8256,
156:       "t_blocks": 131,
157:       "t_mothers": 8384,
158:       "normal_at_normal": {
159:         "accuracy": {
160:           "power": 0.9504225431012328,
161:           "plugin_half_width": 0.016292625434745873,
162:           "expected_half_width": 0.016292625434745873
163:         },
164:         "nll": {
165:           "power": 0.999905603276502,
166:           "plugin_half_width": 0.013769781706873525,
167:           "expected_half_width": 0.013769781706873525
168:         }
169:       },
170:       "t_at_normal": {
171:         "accuracy": {
172:           "power": 0.9475722738758099,
173:           "plugin_half_width": 0.016448130307161433,
174:           "expected_half_width": 0.016416036730692727
175:         },
176:         "nll": {
177:           "power": 0.9998880532300913,
178:           "plugin_half_width": 0.013901207311426547,
179:           "expected_half_width": 0.013874083288724564
180:         }
181:       },
182:       "t_at_selected": {
183:         "accuracy": {
184:           "power": 0.9505118697757498,
185:           "plugin_half_width": 0.016319693256397413,
186:           "expected_half_width": 0.016288339697408512
187:         },
188:         "nll": {
189:           "power": 0.9999060352925451,
190:           "plugin_half_width": 0.013792658191508557,
191:           "expected_half_width": 0.013766159597728204
192:         }
193:       },
194:       "t_at_previous": {
195:         "accuracy": {
196:           "power": 0.9490618189573753,
197:           "plugin_half_width": 0.01638353419239668,
198:           "expected_half_width": 0.016351814221143575
199:         },
200:         "nll": {
201:           "power": 0.9998974312902781,
202:           "plugin_half_width": 0.013846613630194187,
203:           "expected_half_width": 0.01381980536152976
204:         }
205:       }
206:     },
207:     "b_locked": 131
208:   },
209:   "b_locked": 131,
210:   "blind_seed_manifest_sha256": "b6c075162010a0d6e72bb07cd32a4a01aed24e1abca8be2f949846d258c0f7e7",
211:   "dual_practical_threshold": {
212:     "acc_gain_t95_lo_gt": 0.02,
213:     "nll_improv_t95_lo_gt": 0.02,
214:     "logical_pass": "both"
215:   }
216: }
?? research/stage4_math_and_design/
```

### Event #9: `item.completed` — `command_execution` (`item_6`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '"'^(def |from |import |Q2_|BLOCK_SIZE)|orientation_gain|direct_s17|cal_ref|raw_ll|G1_main|H_c4'"' research/stage3d_qwen_signed_phase/recompute_all.py"
```

**Aggregated Output:**

```text
5:import argparse
6:import json
7:import math
8:from pathlib import Path
9:from typing import Dict, List, Optional, Sequence, Tuple
11:import numpy as np
12:from scipy.stats import t as student_t
15:BLOCK_SIZE = 64
19:    "G1_main_extrap_both",
25:    "G1_main_extrap_both": {"abs_omega": 0.40, "h": 2, "ell": 2, "t_p": 4.0, "cum_angle_abs": 1.60, "sigma2_t": 0.04},
30:Q2_ARMS = (
39:Q2_LOGICAL_REF_IDX = {
49:def softmax_temperature(raw_ll: np.ndarray, temp: float) -> np.ndarray:
50:    scaled = np.asarray(raw_ll, dtype=np.float64) / float(temp)
56:def score_vectors(probs: np.ndarray, labels: np.ndarray) -> dict:
100:def paired_t_ci(block_diffs: np.ndarray) -> dict:
118:def block_bootstrap_ci(block_diffs: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED) -> dict:
135:def compute_16way_group_avg(table_256: np.ndarray) -> np.ndarray:
153:def compute_representative_readout(table_256: np.ndarray) -> np.ndarray:
167:def evaluate_cell_paired(
168:    raw_ll_table: np.ndarray,
178:    # raw_ll_table: (2, 3, 2, 256, 4); codes/targets: (2, 3, M)
179:    n_arms, n_seeds, n_signs, _, _ = raw_ll_table.shape
186:    probs_table = np.zeros_like(raw_ll_table, dtype=np.float64)
189:            probs_table[a, s_idx] = softmax_temperature(raw_ll_table[a, s_idx], float(temps[a, s_idx]))
300:            "direct_s17_table": {k: sc_dir[k] for k in ("acc", "nll", "brier", "ece")},
303:        "phase_minus_direct_s17": {
307:        "zero_minus_direct_s17": {
327:def work_f_sign_probe(
328:    raw_ll_table: np.ndarray,
332:    cal_ref: np.ndarray,
339:    n_arms, n_seeds, _, _, _ = raw_ll_table.shape
346:    target_diff = cal_ref[1] - cal_ref[0]  # D(z)
353:        "cal_ref_overlap_mass": float(overlap.sum()),
354:        "cal_ref_odd_norm_sq": norm_d2,
406:            "orientation_gain": gain,
432:    out["physical_cal_reference"] = _analyze_table(cal_ref)
436:            [softmax_temperature(raw_ll_table[a, s_idx], float(temps[a, s_idx])) for s_idx in range(n_seeds)],
449:            "seed_mean_orientation_gain": float(np.mean([x["orientation_gain"] for x in per_seed_res])),
462:    p_s0_tbl = softmax_temperature(raw_ll_table[0, 0], float(temps[0, 0]))
463:    z_s0_tbl = softmax_temperature(raw_ll_table[1, 0], float(temps[1, 0]))
496:        "cal_ref_neg": [float(v) for v in cal_ref[0, c_spot]],
497:        "cal_ref_pos": [float(v) for v in cal_ref[1, c_spot]],
499:            "raw_loglik": [float(v) for v in raw_ll_table[0, 0, 0, c_spot]],
504:            "raw_loglik": [float(v) for v in raw_ll_table[0, 0, 1, c_spot]],
509:            "raw_loglik": [float(v) for v in raw_ll_table[1, 0, 0, c_spot]],
514:            "raw_loglik": [float(v) for v in raw_ll_table[1, 0, 1, c_spot]],
530:            p_match = softmax_temperature(raw_ll_table[0, s_idx], t_val)
578:def work_g_rotation_and_controls(
579:    raw_ll_table: np.ndarray,
585:    indep_raw_ll: np.ndarray,
591:    n_arms, n_seeds, n_signs, _, _ = raw_ll_table.shape
611:            p_raw = softmax_temperature(raw_ll_table[a, s_idx], t_val)  # (2, 256, 4)
614:            p_ind = softmax_temperature(indep_raw_ll[a, s_idx], t_val)
730:def work_a_q2_analysis(q2_npz, temps_q2: np.ndarray, seeds: Sequence[int]) -> dict:
732:    raw_ll = q2_npz["q2__raw_ll"]             # (2, 3, 7, M, 4)
737:    n_arms, n_seeds, n_q2_arms, n_mothers, _ = raw_ll.shape
740:    probs_all = np.zeros_like(raw_ll, dtype=np.float64)
743:            probs_all[a, s_idx] = softmax_temperature(raw_ll[a, s_idx], float(temps_q2[a, s_idx]))
853:        "work_a_physical_reference": work_a_metrics,
859:def recompute_all(data_dir: Path, verify_against: Optional[Path] = None) -> dict:
873:                raw_ll_table=q3_npz[f"{cid}__raw_ll"],
880:                is_primary=(cid == "G1_main_extrap_both"),
886:        raw_ll_table=q3_npz["G1_main_extrap_both__raw_ll"],
888:        codes_rot0=q3_npz["G1_main_extrap_both__codes"][:, 0, :],
889:        targets_rot0=q3_npz["G1_main_extrap_both__targets"][:, 0, :],
890:        cal_ref=q3_npz["G1_main_extrap_both__cal_ref"],
891:        cal_mass=q3_npz["G1_main_extrap_both__cal_mass"],
893:        contra_text_flipped=q3_npz["G1_main_extrap_both__contra_text_flipped_raw_ll"] if "G1_main_extrap_both__contra_text_flipped_raw_ll" in q3_npz else None,
894:        contra_phase_flipped=q3_npz["G1_main_extrap_both__contra_phase_flipped_raw_ll"] if "G1_main_extrap_both__contra_phase_flipped_raw_ll" in q3_npz else None,
897:        raw_ll_table=q3_npz["H_c4_035__raw_ll"],
899:        codes_rot0=q3_npz["H_c4_035__codes"][:, 0, :],
900:        targets_rot0=q3_npz["H_c4_035__targets"][:, 0, :],
901:        cal_ref=q3_npz["H_c4_035__cal_ref"],
902:        cal_mass=q3_npz["H_c4_035__cal_mass"],
908:        raw_ll_table=q3_npz["G1_main_extrap_both__raw_ll"],
910:        codes=q3_npz["G1_main_extrap_both__codes"],
911:        targets=q3_npz["G1_main_extrap_both__targets"],
912:        rot10_codes=q3_npz["G1_main_extrap_both__rot10_codes"],
913:        rot10_targets=q3_npz["G1_main_extrap_both__rot10_targets"],
914:        indep_raw_ll=q3_npz["G1_main_extrap_both__indep_raw_ll"],
915:        indep_targets=q3_npz["G1_main_extrap_both__indep_targets"],
916:        shuffled_perm=q3_npz["G1_main_extrap_both__shuffled_perm"],
929:            if f"{cname}__q3_raw_ll" not in rp:
934:                raw = rp[f"{cname}__{qk}_raw_ll"]  # (2, 3, N, 4)
945:                arm_res["direct_s17"] = {k: sc_d[k] for k in ("acc", "nll", "brier", "ece")}
967:def _compare_nested(expected, actual, path: str = "root") -> float:
993:def main():
1005:        g1 = summary["cells"]["G1_main_extrap_both"]
```

### Event #10: `item.completed` — `command_execution` (`item_7`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n 'nct|rho|pilot|power' research/stage3c/work_results/E/verify.py; Get-ChildItem -Force research/stage4_math_and_design/M2; uv --version; uv python list --only-installed"
```

**Aggregated Output:**

```text
17:from scipy.stats import chi2, nct, norm, t
36:    power = nct.sf(cutoff, df, excess) if use_t else norm.sf(cutoff - excess)
41:    eligible = COUNTS[(power >= POWER) & (plugin <= width)]
44:    return int(eligible[0]), power, plugin, expected
49:        name: dict(power=float(power[blocks - 2]), plugin_half_width=float(plugin[blocks - 2]),
51:        for name, (_, power, plugin, expected) in curves.items()
55:def plan(q, rho, nll_sd, pilot=None):
56:    deff = 1 + (BLOCK - 1) * rho
58:    if pilot is not None:
59:        sd = np.maximum(sd, pilot)
66:    return dict(q=q, rho=rho, nll_mother_sd=nll_sd, pilot_block_sd=pilot,
83:            # Integrate over the standardized sample variance, independently of nct.
88:            exact = nct.sf(cutoff, df, shift)
96:            rows.append(dict(blocks=blocks, endpoint=name, nct=exact, quadrature=integral,
162:    parser.add_argument("--pilot-block-sd", nargs=2, type=float, metavar=("HIT", "NLL"))
164:    if args.pilot_block_sd and any(not math.isfinite(x) or x <= 0 for x in args.pilot_block_sd):
165:        parser.error("pilot SDs must be positive and finite")
170:    scenarios = [plan(q, rho, 0.50) for q, rho in
177:            assert math.isclose(values["power"], ref["approx_power_against_margin"], abs_tol=1e-12)
180:    if args.pilot_block_sd:
181:        scenarios.append(plan(.35, .01, .50, args.pilot_block_sd))
uv 0.11.21 (5aa65dd7a 2026-06-11 x86_64-pc-windows-msvc)
error: Failed to initialize cache at `C:\Users\李则徐\AppData\Local\uv\cache`
  Caused by: failed to open file `C:\Users\李则徐\AppData\Local\uv\cache\sdists-v9\.git`: 拒绝访问。 (os error 5)
```

### Event #11: `item.completed` — `command_execution` (`item_8`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '"'^(  |    )"(G[1-4]|H_c4|work_|[''^"]+scores|phase|zero|direct|seed_mean)'"' research/stage3d_qwen_signed_phase/results/summary_master.json | Select-Object -First 55; rg -n 'omega.*(text|sign)|zero.*(text|omega)|def .*prompt|direct_table_probs|full_iq_probs' research/stage3d_qwen_signed_phase -g '*.py' | Select-Object -First 80"
```

**Aggregated Output:**

```text
rg: regex parse error:
    (?:^(  |    )(G[1-4]|H_c4|work_|[^]+scores|phase|zero|direct|seed_mean))
                                    ^^^
error: unclosed character class
research/stage3d_qwen_signed_phase\analysis_stage3d.py:130:    full_iq_probs = np.zeros((2, 3, n_m, 4), dtype=np.float64)
research/stage3d_qwen_signed_phase\analysis_stage3d.py:147:            full_iq_probs[s_i, r_i] = solve_full_iq_known_param(v1, v2, (), cond, protocol="repeat_x1")["q3"]
research/stage3d_qwen_signed_phase\analysis_stage3d.py:171:        "direct_table_probs": direct_probs,
research/stage3d_qwen_signed_phase\analysis_stage3d.py:172:        "full_iq_probs": full_iq_probs,
research/stage3d_qwen_signed_phase\recompute_all.py:173:    full_iq_probs: np.ndarray,
research/stage3d_qwen_signed_phase\recompute_all.py:273:    sc_iq = score_vectors(full_iq_probs.reshape(-1, 4), all_lbl_flat)
research/stage3d_qwen_signed_phase\recompute_all.py:877:                direct_probs=q3_npz[f"{cid}__direct_table_probs"],
research/stage3d_qwen_signed_phase\recompute_all.py:878:                full_iq_probs=q3_npz[f"{cid}__full_iq_probs"],
research/stage3d_qwen_signed_phase\data_stage3d.py:209:def build_stage3d_prompt(
research/stage3d_qwen_signed_phase\qwen_connector.py:405:    def format_chat_parts(self, user_prompt: str, extra_history: Optional[List[dict]] = None) -> List[str]:
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:950:                        host, conn, omega=sign * w_abs, h=h, ell=ell, t_p=tp, zero_arm=(arm == "zero"),
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:965:                    host, conn, omega=sign * 0.40, h=2, ell=2, t_p=4.0, zero_arm=(arm == "zero"),
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:971:                        host, conn, omega=-sign * 0.40, h=2, ell=2, t_p=4.0, zero_arm=False,
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:972:                        phase_override_omega=sign * 0.40, batch_size=batch_size, scorer_mode=scorer_mode,
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:976:                        host, conn, omega=sign * 0.40, h=2, ell=2, t_p=4.0, zero_arm=False,
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:977:                        phase_override_omega=-sign * 0.40, batch_size=batch_size, scorer_mode=scorer_mode,
research/stage3d_qwen_signed_phase\tests\test_stage3d_cpu.py:81:def test_zero_leakage_prompts_and_slots():
research/stage3d_qwen_signed_phase\tests\test_stage3d_cpu.py:175:        q3_bundle[f"{cid}__direct_table_probs"] = np.full((2, 3, n_m, 4), 0.25)
research/stage3d_qwen_signed_phase\tests\test_stage3d_cpu.py:176:        q3_bundle[f"{cid}__full_iq_probs"] = np.full((2, 3, n_m, 4), 0.25)
```

### Event #12: `item.completed` — `command_execution` (`item_9`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$r='"'research/stage3d_qwen_signed_phase'; "'$lines=Get-Content "$r/recompute_all.py"; $lines[14..47]; $lines[431..577]; $lines[859..926]; $lines=Get-Content "$r/data_stage3d.py"; $lines[208..294]; $lines=Get-Content "$r/train_eval_stage3d.py"; $lines[939..980]'
```

**Aggregated Output:**

```text
BLOCK_SIZE = 64
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_SEED = 2026311
CELL_ORDER = (
    "G1_main_extrap_both",
    "G2_interp_both",
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

    # Single concrete cache spotlight on seed 0 (202611) where phase arm flips its argmax when told +omega vs -omega
    p_s0_tbl = softmax_temperature(raw_ll_table[0, 0], float(temps[0, 0]))
    z_s0_tbl = softmax_temperature(raw_ll_table[1, 0], float(temps[1, 0]))
    # Search real caches appearing in world +1 with opposite argmax under told - vs told +
    best_code = None
    best_tv = -1.0
    for m_idx in range(n_mothers):
        c_pos = int(codes_rot0[1, m_idx])
        c_neg = int(codes_rot0[0, m_idx])
        if c_pos == c_neg:
            arg_neg = int(np.argmax(p_s0_tbl[0, c_pos]))
            arg_pos = int(np.argmax(p_s0_tbl[1, c_pos]))
            if arg_neg != arg_pos and arg_pos == int(targets_rot0[1, m_idx]):
                tv = float(0.5 * np.abs(p_s0_tbl[1, c_pos] - p_s0_tbl[0, c_pos]).sum())
                if tv > best_tv:
                    best_tv = tv
                    best_code = (m_idx, c_pos)
    if best_code is None:
        # Fallback: pick highest sign TV cache appearing in G1
        for m_idx in range(n_mothers):
            c = int(codes_rot0[1, m_idx])
            tv = float(0.5 * np.abs(p_s0_tbl[1, c] - p_s0_tbl[0, c]).sum())
            if tv > best_tv:
                best_tv = tv
                best_code = (m_idx, c)

    m_spot, c_spot = best_code
    z1_sp, z2_sp = divmod(c_spot, 16)
    out["concrete_cache_spotlight_seed0"] = {
        "mother_idx": int(m_spot),
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


def work_g_rotation_and_controls(
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
            ref_block = "\n补充参考：无参考观测�?
        elif deduplicate:
            seen_ids = set()
            items = []
            for r in references:
                if r.acquisition_id in seen_ids:
                    continue
                seen_ids.add(r.acquisition_id)
                tag = "同信�? if r.shares_channel else "独立无关"
                items.append(f"[编号={r.acquisition_id},状�?{tag},时刻={r.t_p:.1f}]<REF>")
            ref_block = "\n补充参考：" + "�?.join(items) + f"（已按唯一采集编号去重，共{len(items)}条独立参考）�?
        else:
            items = []
            for r in references:
                tag = "同信�? if r.shares_channel else "独立无关"
                items.append(f"[编号={r.acquisition_id},状�?{tag},时刻={r.t_p:.1f}]<REF>")
            ref_block = f"\n补充参考（未去重，共{len(items)}条）�? + "�?.join(items) + "�?

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
        f"问题：{qtext}\n"
        f"四个候选选项（按物理顺序）：\n{listed}\n"
        f"请直接从以上四项中输出最可能的一项完整选项文字�?
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
) -> Tuple[Dict[float, MotherBatch], List[int]]:
    """Generate paired MotherBatches sharing balanced triples, Brownian innovations, and additive noises across omegas."""
    block_seeds = derive_block_seeds(namespace, cell_idx, blocks)
    triples_base = list(itertools.product(range(4), repeat=3))
    snr = 10.0 ** (float(snr_db) / 10.0)
    noise_sd = math.sqrt(1.0 / (2.0 * snr))
    norm = math.sqrt(float(N_SAMPLES))
    times = sorted({0.0, float(h), float(h + ell), float(t_p)})
    m_block = 64
    m_total = int(blocks) * m_block

    x1_parts, x2_parts, g_parts, x3_parts, go_parts = [], [], [], [], []
    z_inc_parts: Dict[float, List[np.ndarray]] = {t: [] for t in times[1:]}
    phi_ind_parts = []
    noise_parts = {k: [] for k in ("y1", "y2", "y3_rep", "y3_ind", "p1", "p2", "pind")}
        ("G4_extrap_freq", 0.40, 1, 1, 2.0),
        ("H_c4_035", 0.35, 2, 2, 4.0),
    ]
    for cid, w_abs, h, ell, tp in eval_cells:
        arr = np.zeros((2, len(CONNECTOR_SEEDS), 2, 256, 4), dtype=np.float64)
        for a_i, arm in enumerate(ARM_NAMES):
            for s_i, seed in enumerate(CONNECTOR_SEEDS):
                conn = conns[(arm, seed)][0]
                for sign_i, sign in enumerate((-1, 1)):
                    arr[a_i, s_i, sign_i] = score_256_cache_table(
                        host, conn, omega=sign * w_abs, h=h, ell=ell, t_p=tp, zero_arm=(arm == "zero"),
                        batch_size=batch_size, scorer_mode=scorer_mode,
                    )
        q3_arrays[f"{cid}__raw_ll"] = arr
        print(f"[BLIND Q3 TABLE] Finished {cid}", flush=True)

    # 2. G1 independent_x3 control + single-modality contradiction probes (on phase arm)
    indep_arr = np.zeros((2, len(CONNECTOR_SEEDS), 2, 256, 4), dtype=np.float64)
    contra_text_arr = np.zeros((len(CONNECTOR_SEEDS), 2, 256, 4), dtype=np.float64)
    contra_phase_arr = np.zeros((len(CONNECTOR_SEEDS), 2, 256, 4), dtype=np.float64)
    for a_i, arm in enumerate(ARM_NAMES):
        for s_i, seed in enumerate(CONNECTOR_SEEDS):
            conn = conns[(arm, seed)][0]
            for sign_i, sign in enumerate((-1, 1)):
                indep_arr[a_i, s_i, sign_i] = score_256_cache_table(
                    host, conn, omega=sign * 0.40, h=2, ell=2, t_p=4.0, zero_arm=(arm == "zero"),
                    protocol="independent_x3", batch_size=batch_size, scorer_mode=scorer_mode,
                )
                if arm == "phase":
                    # contra_text_flipped[s_i, sign_i]: phase slots match sign_i, text flipped to -sign
                    contra_text_arr[s_i, sign_i] = score_256_cache_table(
                        host, conn, omega=-sign * 0.40, h=2, ell=2, t_p=4.0, zero_arm=False,
                        phase_override_omega=sign * 0.40, batch_size=batch_size, scorer_mode=scorer_mode,
                    )
                    # contra_phase_flipped[s_i, sign_i]: text matches sign_i, phase slots flipped to -sign
                    contra_phase_arr[s_i, sign_i] = score_256_cache_table(
                        host, conn, omega=sign * 0.40, h=2, ell=2, t_p=4.0, zero_arm=False,
                        phase_override_omega=-sign * 0.40, batch_size=batch_size, scorer_mode=scorer_mode,
                    )
    q3_arrays["G1_main_extrap_both__indep_raw_ll"] = indep_arr
    q3_arrays["G1_main_extrap_both__contra_text_flipped_raw_ll"] = contra_text_arr
    q3_arrays["G1_main_extrap_both__contra_phase_flipped_raw_ll"] = contra_phase_arr
```

### Event #13: `item.completed` — `command_execution` (`item_10`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$j=Get-Content -Raw research/stage3d_qwen_signed_phase/results/summary_master.json | ConvertFrom-Json; $j.psobject.Properties.Name; $j.cells.psobject.Properties.Name; Get-ChildItem -Force "$env:LOCALAPPDATA/uv"; Get-ChildItem -Force "$env:LOCALAPPDATA/uv/cache" | Select-Object Name,Mode; Get-ChildItem Env:UV*; Get-Command py -ErrorAction SilentlyContinue; rg -n -m 5 '"'control|Holm|reproduc' 'C:/Users/李则徐/.codex/memories/rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md'"
```

**Aggregated Output:**

```text
seeds
locked_temps_q3
locked_temps_q2
cells
work_f_primary_040
work_f_historical_035
work_g_primary_040
work_a_q2_c4
replay_c123_forgetting
historical_c4_035_lookup_anchor
G1_main_extrap_both
G2_interp_both
G3_extrap_angle
G4_extrap_freq


    Ŀ¼: C:\Users\������\AppData\Local\uv


Mode                 LastWriteTime         Length Name                                                                 
----                 -------------         ------ ----                                                                 
d-----        2026/11/26     19:11                cache                                                                

Name : archive-v0
Mode : d-----


Name : builds-v0
Mode : d-----


Name : environments-v2
Mode : d-----


Name : interpreter-v4
Mode : d-----


Name : sdists-v9
Mode : d-----


Name : simple-v21
Mode : d-----


Name : wheels-v6
Mode : d-----


Name : .gitignore
Mode : -a----


Name : .lock
Mode : -a----


Name : CACHEDIR.TAG
Mode : -a----


Path               : C:\WINDOWS\py.exe
Extension          : .exe
Definition         : C:\WINDOWS\py.exe
Source             : C:\WINDOWS\py.exe
Version            : 3.13.150.1013
Visibility         : Public
OutputType         : {System.String}
Name               : py.exe
CommandType        : Application
ModuleName         : 
Module             : 
RemotingCapability : PowerShell
Parameters         : 
ParameterSets      : 
HelpUri            : 
FileVersionInfo    : File:             C:\WINDOWS\py.exe
                     InternalName:     Python Launcher
                     OriginalFilename: py.exe
                     FileVersion:      3.13.0
                     FileDescription:  Python
                     Product:          Python
                     ProductVersion:   3.13.0
                     Debug:            False
                     Patched:          False
                     PreRelease:       False
                     PrivateBuild:     False
                     SpecialBuild:     False
                     Language:         ��������
                     

7:# P4 established a reproducible EM-native Transformer simulation prototype, but not a demonstrated end-to-end performance advantage
24:- Follow-up receiver experiments reduced errors on unseen delay distributions, but the pure Fourier/逐径 pilot front-end alone achieved nearly all of the gain. Therefore do not attribute this improvement to Transformer attention without matched-front-end controls.
31:- Completed formal ten-seed P4-D protocol using seeds `74001–74010`, 600 training steps, batch size 64, exact integer bit/block accounting, exhaustive 1024-state sign-flip tests, and Holm adjustment.
36:- Formal two-dimensional result: EM did not significantly outperform fixed-prior classical baselines in any primary comparison. In-distribution at 10 dB/strong PA, EM had 3,266/4,160 block errors versus fixed LMMSE 3,227 and fixed IDD-PA 3,211; EM was significantly worse than fixed IDD-PA after Holm correction (`p=0.0234375`). Long-delay EM decoded only 22/4,160 blocks successfully, though matched-statistics baselines were much better.
39:- Scientific status: P4 completed a reproducible digital baseband prototype and answered important first-round architecture questions, but it did not prove that the learned 2D EM-native Transformer adds stable value over a well-designed physical receiver. Real hardware/electronic implementation remains untested.


```

### Event #14: `item.completed` — `web_search` (`exec-735b51df-54f3-41bd-bbec-72c552126a2a`)

```json
{
  "id": "exec-735b51df-54f3-41bd-bbec-72c552126a2a",
  "type": "web_search",
  "query": "Holm 1979 A simple sequentially rejective multiple test procedure original paper JSTOR ...",
  "action": {
    "type": "search",
    "queries": [
      "Holm 1979 A simple sequentially rejective multiple test procedure original paper JSTOR",
      "scipy scipy.stats.nct noncentral t official documentation ANOVA two factor random effects variance components NIST orthogonal balanced"
    ]
  },
  "results": [
    {
      "type": "text_result",
      "domain": "ndlsearch.ndl.go.jp",
      "ref_id": "turn0search0",
      "snippet": "A simple sequentially rejective multiple test procedure # A simple sequentially rejective multiple test procedure ... 1979 ... Paper",
      "title": "A simple sequentially rejective multiple test procedure | NDL Search | National Diet Library",
      "url": "https://ndlsearch.ndl.go.jp/en/books/R100000136-I1572543024862166272"
    },
    {
      "type": "text_result",
      "domain": "ndlsearch.ndl.go.jp",
      "ref_id": "turn0search1",
      "snippet": "A simple sequentially rejective multiple test procedure # A simple sequentially rejective multiple test procedure ... 1979",
      "title": "A simple sequentially rejective multiple test procedure | NDLサーチ | 国立国会図書館",
      "url": "https://ndlsearch.ndl.go.jp/books/R100000136-I1572543024862166272"
    },
    {
      "type": "text_result",
      "domain": "bibbase.org",
      "ref_id": "turn0search2",
      "snippet": "This paper presents a simple and widely applicable multiple test procedure of the sequentially rejective type, i.e. hypotheses are rejected one at a time until",
      "title": "A Simple Sequentially Rejective Multiple Test Procedure",
      "url": "https://bibbase.org/network/publication/holm-asimplesequentiallyrejectivemultipletestprocedure-1979"
    },
    {
      "type": "text_result",
      "domain": "www.ime.usp.br",
      "ref_id": "turn0search12",
      "snippet": "A Simple Sequentially Rejective Multiple Test Procedure ... 6, No. 2 (1979), pp. ... JSTOR is a not-for-profit service that helps scholars, researchers, and students",
      "title": "A Simple Sequentially Rejective Multiple Test Procedure",
      "url": "https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf"
    },
    {
      "type": "text_result",
      "domain": "ndlsearch.ndl.go.jp",
      "ref_id": "turn0search3",
      "snippet": "A simple sequentially rejective multiple test procedure # A simple sequentially rejective multiple test procedure ... 1979",
      "title": "A simple sequentially rejective multiple test procedure | NDLサーチ | 国立国会図書館",
      "url": "https://ndlsearch.ndl.go.jp/books/R100000136-I1571698601093721856"
    },
    {
      "type": "text_result",
      "domain": "ndlsearch.ndl.go.jp",
      "ref_id": "turn0search4",
      "snippet": "A simple sequentially rejective Bonferroni test procedure # A simple sequentially rejective Bonferroni test procedure ... 1979 ... Paper",
      "title": "A simple sequentially rejective Bonferroni test procedure | NDL Search | National Diet Library",
      "url": "https://ndlsearch.ndl.go.jp/en/books/R100000136-I1574231874590250496"
    },
    {
      "type": "text_result",
      "domain": "tlooto.com",
      "ref_id": "turn0search5",
      "snippet": "# A Simple Sequentially Rejective Multiple Test Procedure ... This paper presents a simple and widely ap- plicable multiple test procedure of the sequentially rejective",
      "title": "A Simple Sequentially Rejective Multiple Test Procedure | tlooto",
      "url": "https://tlooto.com/paper/150130888732"
    },
    {
      "type": "text_result",
      "domain": "www.tandfonline.com",
      "ref_id": "turn0search6",
      "snippet": "S Holm - Scandinavian journal of statistics, 1979 - JSTOR ... This paper presents a simple and widely applicable multiple test procedure of the sequentially",
      "title": "Holm: A simple sequentially rejective multiple test... - Google Scholar",
      "url": "https://www.tandfonline.com/action/getFTRLinkout?doi=10.1080%2F13546783.2015.1127289&doiOfLink=&linkLocation=Reference&linkSource=FULL_TEXT&linkType=gs&url=http%3A%2F%2Fscholar.google.com%2Fscholar_lookup%3Fhl%3Den%26volume%3D6%26publication_year%3D1979%26pages%3D65-70%26journal%3DScandinavian%2BJournal%2Bof%2BStatistics%26author%3DS.%2BHolm%26title%3DA%2Bsimple%2Bsequentially%2Brejective%2Bmultiple%2Btest%2Bprocedure"
    },
    {
      "type": "text_result",
      "domain": "www.academia.edu",
      "ref_id": "turn0search7",
      "snippet": "http://www.jstor.org This content downloaded from 143.107.183.117 on Wed, 9 Oct 2013 08:15:53 AM All use subject to JSTOR Terms and Conditions Scand J Statist 6:",
      "title": "A Simple Sequentially Rejective Multiple Test Procedure",
      "url": "https://www.academia.edu/77036515/A_Simple_Sequentially_Rejective_Multiple_Test_Procedure"
    },
    {
      "type": "text_result",
      "domain": "scholar.teselagen.com",
      "ref_id": "turn0search8",
      "snippet": "Sture Holm · Scandinavian Journal of Statistics · 1979 This paper presents a simple and widely ap- plicable multiple test procedure of the sequentially rejective",
      "title": "A Simple Sequentially Rejective Multiple Test Procedure | TG Scholar",
      "url": "https://scholar.teselagen.com/works/W2121044470/a-simple-sequentially-rejective-multiple-test-procedure"
    },
    {
      "type": "text_result",
      "domain": "elibrary.unn.edu.ng",
      "ref_id": "turn0search9",
      "snippet": "# A Simple Sequentially Rejective Multiple Test Procedure ... A Simple Sequentially Rejective Multiple Test Procedure - Full text available via OpenAlex",
      "title": "UNN E-Library",
      "url": "https://elibrary.unn.edu.ng/library/international-journals/W2121044470/"
    },
    {
      "type": "text_result",
      "domain": "www.rankless.org",
      "ref_id": "turn0search10",
      "snippet": "This paper presents a simple and widely ap- plicable multiple test procedure of the sequentially rejective type, i.e. hypotheses are rejected one at a tine",
      "title": "Rankless | A Simple Sequentially Rejective Multiple Test Procedure",
      "url": "https://www.rankless.org/hit-papers/10.2307/4615733"
    },
    {
      "type": "text_result",
      "domain": "openalex.org",
      "ref_id": "turn0search11",
      "snippet": "# A Simple Sequentially Rejective Multiple Test Procedure ... Year: 1979 ... Author: Sture Holm",
      "title": "A Simple Sequentially Rejective Multiple Test Procedure | OpenAlex",
      "url": "https://openalex.org/works/W2121044470"
    },
    {
      "type": "text_result",
      "domain": "cdn.clinicaltrials.gov",
      "ref_id": "turn0search13",
      "snippet": "Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. ... *The American Economic Review, 80*(2), 319–323. https://www.jstor.org/stable/2006592",
      "title": "Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70. https://www.jstor.org/stable/4615733",
      "url": "https://cdn.clinicaltrials.gov/large-docs/56/NCT03593356/SAP_005.pdf"
    },
    {
      "type": "text_result",
      "domain": "ijse.padovauniversitypress.it",
      "ref_id": "turn0search14",
      "snippet": "Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. ... JSTOR, http://www.jstor.org/stable/4615733 ... A Review of Social Psychological Research.",
      "title": "Holm, S. (1979). A Simple Sequentially Rejective Multiple Test Procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70. JSTOR, http://www.jstor.org/stable/4615733",
      "url": "https://ijse.padovauniversitypress.it/system/files/papers/IJSE-2024-1-1.pdf"
    },
    {
      "type": "text_result",
      "domain": "pure.iiasa.ac.at",
      "ref_id": "turn0search15",
      "snippet": "Holm, S. (1979). A simple sequentially rejective multiple test procedure.",
      "title": "-14-\n\nHolm, S. (1979). A simple sequentially rejec",
      "url": "https://pure.iiasa.ac.at/id/eprint/2720/1/CP-85-028.pdf"
    },
    {
      "type": "text_result",
      "domain": "api.research-repository.uwa.edu.au",
      "ref_id": "turn0search16",
      "snippet": "Holm, S. (1979). A simple sequentially rejective multiple test procedure. ... Retrieved from http://www.jstor.org/stable/4615733 ... The timing of child physical maltreatment: A cross-domain growth an",
      "title": "Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70. Retrieved from http://www.jstor.org/stable/4615733",
      "url": "https://api.research-repository.uwa.edu.au/ws/portalfiles/portal/18357061/THESIS_DOCTOR_OF_PHILOSOPHY_BELL_Megan_Fiona_2017.pdf"
    },
    {
      "type": "text_result",
      "domain": "en.wikipedia.org",
      "ref_id": "turn0search17",
      "snippet": "Holm's motives for naming his method after Bonferroni are explained in the original paper: \"The use of the Boole inequality within multiple inference theory is",
      "title": "Holm–Bonferroni method",
      "url": "https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method"
    },
    {
      "type": "text_result",
      "domain": "en.wikipedia.org",
      "ref_id": "turn0search18",
      "snippet": "In 1979, Holm proposed the Holm procedure, a stepwise algorithm for controlling the FWER that is at least as powerful as the well-known Bonferroni adjustment.",
      "title": "False discovery rate",
      "url": "https://en.wikipedia.org/wiki/False_discovery_rate"
    },
    {
      "type": "text_result",
      "domain": "pt.wikipedia.org",
      "ref_id": "turn0search19",
      "snippet": "(1979). «A simple sequentially rejective multiple test procedure». ... JSTOR 4615733. ... «Holm's Sequential Bonferroni Procedure» (PDF).",
      "title": "Teste de hipóteses",
      "url": "https://pt.wikipedia.org/wiki/Teste_de_hip%C3%B3teses"
    },
    {
      "type": "text_result",
      "domain": "ru.wikipedia.org",
      "ref_id": "turn0search20",
      "snippet": "Adjusting for Multiple Testing When Reporting Research Results: The Bonferroni vs Holm Methods // American Journal of Public Health, Vol.86, No.5. — 1996. — Май.",
      "title": "Поправка на множественную проверку гипотез",
      "url": "https://ru.wikipedia.org/wiki/%D0%9F%D0%BE%D0%BF%D1%80%D0%B0%D0%B2%D0%BA%D0%B0_%D0%BD%D0%B0_%D0%BC%D0%BD%D0%BE%D0%B6%D0%B5%D1%81%D1%82%D0%B2%D0%B5%D0%BD%D0%BD%D1%83%D1%8E_%D0%BF%D1%80%D0%BE%D0%B2%D0%B5%D1%80%D0%BA%D1%83_%D0%B3%D0%B8%D0%BF%D0%BE%D1%82%D0%B5%D0%B7"
    },
    {
      "type": "text_result",
      "domain": "de.wikipedia.org",
      "ref_id": "turn0search21",
      "snippet": "Holm: A simple sequentially rejective multiple test procedure. ... 6, 1979, S.",
      "title": "Alphafehler-Kumulierung",
      "url": "https://de.wikipedia.org/wiki/Alphafehler-Kumulierung"
    },
    {
      "type": "text_result",
      "domain": "en.wikipedia.org",
      "ref_id": "turn0search22",
      "snippet": "The Holm-Bonferroni method is a special case of a closed test procedure for which each intersection null hypothesis is tested using the simple Bonferroni test.",
      "title": "Closed testing procedure",
      "url": "https://en.wikipedia.org/wiki/Closed_testing_procedure"
    },
    {
      "type": "text_result",
      "domain": "arxiv.org",
      "ref_id": "turn0academia23",
      "snippet": "In this paper, we argue that sequential rejection can be considered equally fundamental as a general principle of multiple testing. We present a general sequentially",
      "title": "The sequential rejection principle of familywise error control",
      "url": "https://arxiv.org/abs/1211.3313"
    },
    {
      "type": "text_result",
      "domain": "arxiv.org",
      "ref_id": "turn0academia24",
      "snippet": "Title: Sequential Tests of Multiple Hypotheses Controlling Type I and II Familywise Error Rates ... Treating each experiment as a hypothesis test and adopting the",
      "title": "Sequential Tests of Multiple Hypotheses Controlling Type I and II Familywise Error Rates",
      "url": "https://arxiv.org/abs/1304.6309"
    }
  ]
}
```

### Event #15: `item.completed` — `command_execution` (`item_13`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$r='"'research/stage3d_qwen_signed_phase'; rg -n -A 22 'def format_public_metadata|def score_256_cache_table|def _zero' "'$r/data_stage3d.py $r/train_eval_stage3d.py; $l=Get-Content "$r/analysis_stage3d.py"; $l[100..176]; $j=Get-Content -Raw "$r/results/summary_master.json" | ConvertFrom-Json; $j.historical_c4_035_lookup_anchor | ConvertTo-Json -Depth 3; $j.work_f_primary_040.arms.phase.per_seed | Select-Object seed,orientation_gain,@{n='"'swap';e={"'$_.swapped_minus_matched_nll_t95.mean}}; $j.work_f_primary_040.arms.zero.per_seed | Select-Object seed,orientation_gain,@{n='"'swap';e={"'$_.swapped_minus_matched_nll_t95.mean}}'
```

**Aggregated Output:**

```text
research/stage3d_qwen_signed_phase/data_stage3d.py:193:def format_public_metadata(cond: ConditionSpec) -> str:
research/stage3d_qwen_signed_phase/data_stage3d.py-194-    """Zero-leakage public metadata string (excludes condition names, split names, labels, and noise)."""
research/stage3d_qwen_signed_phase/data_stage3d.py-195-    if cond.drift_type == "coherent":
research/stage3d_qwen_signed_phase/data_stage3d.py-196-        drift_desc = "相干静止(omega=+0.00rad/步,sigma=0.00rad/√步,d=0.00/步)"
research/stage3d_qwen_signed_phase/data_stage3d.py-197-    elif cond.drift_type == "discrete":
research/stage3d_qwen_signed_phase/data_stage3d.py-198-        drift_desc = f"离散C4游走(omega=+0.00rad/步,sigma=0.00rad/√步,d={cond.d:.2f}/步)"
research/stage3d_qwen_signed_phase/data_stage3d.py-199-    else:
research/stage3d_qwen_signed_phase/data_stage3d.py-200-        drift_desc = f"连续频偏与布朗漂移(omega={cond.omega:+.2f}rad/步,sigma={cond.sigma_phi:.2f}rad/√步,d=0.00/步)"
research/stage3d_qwen_signed_phase/data_stage3d.py-201-    return (
research/stage3d_qwen_signed_phase/data_stage3d.py-202-        f"信噪比={cond.snr_db:.1f}dB; "
research/stage3d_qwen_signed_phase/data_stage3d.py-203-        f"两窗间隔h={cond.h}, 留出间隔ell={cond.ell}, 导频时刻tp={cond.t_p:.1f}; "
research/stage3d_qwen_signed_phase/data_stage3d.py-204-        f"信道={drift_desc}。\n"
research/stage3d_qwen_signed_phase/data_stage3d.py-205-        f"已编码波形与公开相位向量：<W1><W2><PHASE>。"
research/stage3d_qwen_signed_phase/data_stage3d.py-206-    )
research/stage3d_qwen_signed_phase/data_stage3d.py-207-
research/stage3d_qwen_signed_phase/data_stage3d.py-208-
research/stage3d_qwen_signed_phase/data_stage3d.py-209-def build_stage3d_prompt(
research/stage3d_qwen_signed_phase/data_stage3d.py-210-    question_type: str,
research/stage3d_qwen_signed_phase/data_stage3d.py-211-    cond: ConditionSpec,
research/stage3d_qwen_signed_phase/data_stage3d.py-212-    protocol: str = "repeat_x1",
research/stage3d_qwen_signed_phase/data_stage3d.py-213-    references: Sequence[Reference] = (),
research/stage3d_qwen_signed_phase/data_stage3d.py-214-    deduplicate: bool = True,
research/stage3d_qwen_signed_phase/data_stage3d.py-215-) -> Tuple[str, Tuple[str, ...]]:
--
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:837:def score_256_cache_table(
research/stage3d_qwen_signed_phase/train_eval_stage3d.py-838-    host: QwenCacheHost,
research/stage3d_qwen_signed_phase/train_eval_stage3d.py-839-    connector: SignedPhaseCacheConnector,
research/stage3d_qwen_signed_phase/train_eval_stage3d.py-840-    omega: float,
research/stage3d_qwen_signed_phase/train_eval_stage3d.py-841-    h: int,
research/stage3d_qwen_signed_phase/train_eval_stage3d.py-842-    ell: int,
research/stage3d_qwen_signed_phase/train_eval_stage3d.py-843-    t_p: float,
research/stage3d_qwen_signed_phase/train_eval_stage3d.py-844-    zero_arm: bool,
research/stage3d_qwen_signed_phase/train_eval_stage3d.py-845-    protocol: str = "repeat_x1",
research/stage3d_qwen_signed_phase/train_eval_stage3d.py-846-    phase_override_omega: Optional[float] = None,
research/stage3d_qwen_signed_phase/train_eval_stage3d.py-847-    batch_size: int = 64,
research/stage3d_qwen_signed_phase/train_eval_stage3d.py-848-    scorer_mode: str = "kv_cached",
research/stage3d_qwen_signed_phase/train_eval_stage3d.py-849-) -> np.ndarray:
research/stage3d_qwen_signed_phase/train_eval_stage3d.py-850-    """Score all 256 discrete (z1, z2) cache pairs at fixed public metadata; returns raw_ll (256, 4)."""
research/stage3d_qwen_signed_phase/train_eval_stage3d.py-851-    cond = condition_spec_from_params(omega, h, ell, t_p, snr_db=2.0, sigma_phi=0.10, tag="eval")
research/stage3d_qwen_signed_phase/train_eval_stage3d.py-852-    prompt, choices = build_stage3d_prompt("Q3", cond, protocol=protocol)
research/stage3d_qwen_signed_phase/train_eval_stage3d.py-853-    w_slot = omega if phase_override_omega is None else phase_override_omega
research/stage3d_qwen_signed_phase/train_eval_stage3d.py-854-    p_vec = public_phase_slots(w_slot, h, ell, t_p, zero_arm=zero_arm)
research/stage3d_qwen_signed_phase/train_eval_stage3d.py-855-
research/stage3d_qwen_signed_phase/train_eval_stage3d.py-856-    z1_all = np.arange(256, dtype=np.int64) // 16
research/stage3d_qwen_signed_phase/train_eval_stage3d.py-857-    z2_all = np.arange(256, dtype=np.int64) % 16
research/stage3d_qwen_signed_phase/train_eval_stage3d.py-858-    ps_all = np.repeat(p_vec[None, :], 256, axis=0)
research/stage3d_qwen_signed_phase/train_eval_stage3d.py-859-    grp = {
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
{
    "source":  "research/stage3b_math/A/c4_s17_lookup.json",
    "h_c4_pos_acc":  0.6318359375,
    "h_c4_pos_nll":  0.9650571187776844,
    "work_f_check_f_py_035":  {
                                  "phase":  {
                                                "orientation_gain":  0.011111701098819947,
                                                "residual_ratio":  0.036864847496508715,
                                                "mean_sign_TV":  0.026730785475160612,
                                                "mean_rotation_TV":  0.08525621770297791,
                                                "reference_odd_norm_squared":  0.7706897692346777,
                                                "matched_accuracy":  0.223876953125,
                                                "swapped_accuracy":  0.2100830078125,
                                                "swapped_minus_matched_nll":  [
                                                                                  0.018112136041199003,
                                                                                  0.016661060106803696,
                                                                                  0.019498864925645894
                                                                              ],
                                                "factorial_mean_squared_norm":  {
                                                                                    "metadata":  0.00021902479410414665,
                                                                                    "wave":  0.0010807147055390673,
                                                                                    "interaction":  6.2730646255762369E-05
                                                                                }
                                            },
                                  "zero":  {
                                               "orientation_gain":  0.005619790253737205,
                                               "residual_ratio":  0.026739340817530592,
                                               "mean_sign_TV":  0.01921407453500916,
                                               "mean_rotation_TV":  0.09909726683071512,
                                               "reference_odd_norm_squared":  0.7706897692346777,
                                               "matched_accuracy":  0.2158203125,
                                               "swapped_accuracy":  0.216064453125,
                                               "swapped_minus_matched_nll":  [
                                                                                 0.00934451514945835,
                                                                                 0.008375374707456773,
                                                                                 0.010342039645440837
                                                                             ],
                                               "factorial_mean_squared_norm":  {
                                                                                   "metadata":  0.00010600420324573385,
                                                                                   "wave":  0.0010311596081371483,
                                                                                   "interaction":  3.9866612445473744E-05
                                                                               }
                                           }
                              }
}

  seed       orientation_gain                   swap
  ----       ----------------                   ----
202611 -0.0014714446816086089 -0.0028721712359112347
202622   0.021043669725492792    0.04182439063957688
202633   0.009887160059061767     0.0161572721651206
202611  0.0018401095345733241  0.0047811292760272725
202622   0.012949292388971034   0.017931572051473263
202633 0.00035387946319507136 -0.0005971783068895441


```

### Event #16: `item.completed` — `command_execution` (`item_12`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "py -0p; Get-ChildItem 'C:/Users/李则徐/AppData/Local/uv/cache/wheels-v6' -Recurse -File | Where-Object Name -Match 'numpy|scipy' | Select-Object -First 20 FullName; Get-ChildItem 'C:/Users/李则徐/AppData/Local/uv/cache/archive-v0' -Directory | ForEach-Object { Get-ChildItem -LiteralPath "'$_.FullName -Directory | Where-Object Name -match '"'"'^(numpy|scipy)-.*dist-info'"' | Select-Object FullName }; Get-ChildItem 'C:/Users/李则徐/AppData/Roaming/uv/python' -Force -ErrorAction SilentlyContinue"
```

**Aggregated Output:**

```text
 -V:3.13 *        C:\Users\李则徐\AppData\Local\Programs\Python\Python313\python.exe

FullName                                                                                           
--------                                                                                           
C:\Users\������\AppData\Local\uv\cache\wheels-v6\pypi\numpy\numpy-2.3.5-cp312-cp312-win_amd64.lock 
C:\Users\������\AppData\Local\uv\cache\wheels-v6\pypi\numpy\numpy-2.5.5-cp312-cp312-win_amd64.lock 
C:\Users\������\AppData\Local\uv\cache\wheels-v6\pypi\scipy\scipy-1.17.0-cp312-cp312-win_amd64.lock
C:\Users\������\AppData\Local\uv\cache\archive-v0\3Oxde2-Jkv6KGjnQ\scipy-1.17.0.dist-info          
C:\Users\������\AppData\Local\uv\cache\archive-v0\cSnmXl2iWBpNAXGE\numpy-2.3.5.dist-info           
C:\Users\������\AppData\Local\uv\cache\archive-v0\i6_zPlBrmc1oGgj1\numpy-2.5.5.dist-info           


```

### Event #17: `item.completed` — `command_execution` (`item_16`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-ChildItem -Force research/stage4_math_and_design/M2; $l=Get-Content research/stage3d_qwen_signed_phase/analysis_stage3d.py; $l[176..261]; rg -n -A 15 '"'def public_phase_slots' research/stage3d_qwen_signed_phase -g '*.py'; rg -n 'ROTATIONS =|H_c4_035' research/stage3d_qwen_signed_phase/analysis_stage3d.py"
```

**Aggregated Output:**

```text
        "cal_mass": cal_mass,
        "block_seeds": np.asarray(block_seeds, dtype=np.uint64),
    }


def compute_dev_pilot_sample_size(
    pilot_raw_ll: np.ndarray,
    temps_q3: np.ndarray,
    repo_root: Path = REPO_ROOT,
) -> dict:
    """Measure conservative upper block-mean SD on independent dev pilot (16 blocks at |omega|=0.18) and lock B."""
    encoder, _ = load_s17_encoder(repo_root)
    b_map, seeds = generate_paired_cfo_blocks(
        namespace=NS_DEV_PILOT,
        cell_idx=0,
        blocks=16,
        h=2,
        ell=2,
        t_p=4.0,
        omegas=(-0.18, 0.18),
        split_prefix="dev_pilot",
    )
    n_blocks = 16
    n_mothers = n_blocks * BLOCK_SIZE
    codes = np.zeros((2, 3, n_mothers), dtype=np.uint8)
    targets = np.zeros((2, 3, n_mothers), dtype=np.uint8)
    for s_i, w in enumerate((-0.18, 0.18)):
        batch = b_map[w]
        for r_i, (s1, s2) in enumerate(ROTATIONS):
            v1, v2 = batch.rotated_view(s1, s2)
            z1 = np.asarray(encoder.encode(v1), dtype=np.int64)
            z2 = np.asarray(encoder.encode(v2), dtype=np.int64)
            codes[s_i, r_i] = (16 * z1 + z2).astype(np.uint8)
            targets[s_i, r_i] = sector_of_complex(matched_sum(batch.y3_repeat * ROOTS[s1 % 4])).astype(np.uint8)

    n_arms, n_seeds, n_signs, _, _ = pilot_raw_ll.shape
    hits = np.zeros((n_arms, n_seeds, n_signs, 3, n_mothers), dtype=np.float64)
    nlls = np.zeros((n_arms, n_seeds, n_signs, 3, n_mothers), dtype=np.float64)
    for a in range(n_arms):
        for s_idx in range(n_seeds):
            pt = softmax_temperature(pilot_raw_ll[a, s_idx], float(temps_q3[a, s_idx]))
            for sign_i in range(n_signs):
                pr = pt[sign_i][codes[sign_i]]
                sc = score_vectors(pr.reshape(-1, 4), targets[sign_i].reshape(-1))
                hits[a, s_idx, sign_i] = sc["per_item_hit"].reshape(3, n_mothers)
                nlls[a, s_idx, sign_i] = sc["per_item_nll"].reshape(3, n_mothers)

    blk_hit = hits.mean(axis=(1, 2, 3)).reshape(2, n_blocks, BLOCK_SIZE).mean(axis=2)
    blk_nll = nlls.mean(axis=(1, 2, 3)).reshape(2, n_blocks, BLOCK_SIZE).mean(axis=2)
    acc_diff = blk_hit[0] - blk_hit[1]
    nll_diff = blk_nll[1] - blk_nll[0]

    sd_acc_obs = float(acc_diff.std(ddof=1))
    sd_nll_obs = float(nll_diff.std(ddof=1))
    df = n_blocks - 1
    # One-sided 95% upper confidence bound on SD under normal block means
    upper_mult = math.sqrt(df / float(chi2.ppf(0.05, df)))
    sd_acc_upper = max(1e-6, sd_acc_obs * upper_mult)
    sd_nll_upper = max(1e-6, sd_nll_obs * upper_mult)

    verify_e_py = repo_root / "research" / "stage3c" / "work_results" / "E" / "verify.py"
    raw_e = subprocess.check_output(
        [
            sys.executable,
            "-B",
            str(verify_e_py),
            "--pilot-block-sd",
            f"{sd_acc_upper:.8f}",
            f"{sd_nll_upper:.8f}",
        ],
        cwd=repo_root,
    )
    e_json = json.loads(raw_e.decode("utf-8"))
    pilot_scenario = e_json["scenarios"][-1]
    b_locked = max(131, int(pilot_scenario["t_blocks"]))
    return {
        "namespace": NS_DEV_PILOT,
        "blocks": n_blocks,
        "mothers": n_mothers,
        "pilot_block_seeds": seeds,
        "observed_acc_diff_mean": float(acc_diff.mean()),
        "observed_acc_diff_block_sd": sd_acc_obs,
        "observed_nll_diff_mean": float(nll_diff.mean()),
        "observed_nll_diff_block_sd": sd_nll_obs,
        "chi2_upper_multiplier_df15_alpha05": upper_mult,
        "conservative_upper_sd_hit": sd_acc_upper,
research/stage3d_qwen_signed_phase\data_stage3d.py:161:def public_phase_slots(omega: float, h: int, ell: int, t_p: float, zero_arm: bool = False) -> np.ndarray:
research/stage3d_qwen_signed_phase\data_stage3d.py-162-    """Six public phase slots at t = (h, h + ell, t_p) relative to t1 = 0."""
research/stage3d_qwen_signed_phase\data_stage3d.py-163-    if zero_arm:
research/stage3d_qwen_signed_phase\data_stage3d.py-164-        return np.zeros(6, dtype=np.float32)
research/stage3d_qwen_signed_phase\data_stage3d.py-165-    times = (float(h), float(h + ell), float(t_p))
research/stage3d_qwen_signed_phase\data_stage3d.py-166-    return np.array([v for t in times for v in (math.cos(omega * t), math.sin(omega * t))], dtype=np.float32)
research/stage3d_qwen_signed_phase\data_stage3d.py-167-
research/stage3d_qwen_signed_phase\data_stage3d.py-168-
research/stage3d_qwen_signed_phase\data_stage3d.py-169-def condition_spec_from_params(
research/stage3d_qwen_signed_phase\data_stage3d.py-170-    omega: float,
research/stage3d_qwen_signed_phase\data_stage3d.py-171-    h: int,
research/stage3d_qwen_signed_phase\data_stage3d.py-172-    ell: int,
research/stage3d_qwen_signed_phase\data_stage3d.py-173-    t_p: float,
research/stage3d_qwen_signed_phase\data_stage3d.py-174-    snr_db: float = 2.0,
research/stage3d_qwen_signed_phase\data_stage3d.py-175-    sigma_phi: float = 0.10,
research/stage3d_qwen_signed_phase\data_stage3d.py-176-    tag: str = "cfo",
294:        ("H_c4_035", 950406, 16, 0.35, 2, 2, 4.0, 4),
324:    # Historical C4 +0.35 lookup verification on H_c4_035 positive world
332:    h_pos_codes = q3_bundle["H_c4_035__codes"][1].reshape(-1)
333:    h_pos_targets = q3_bundle["H_c4_035__targets"][1].reshape(-1)
342:            [softmax_temperature(q3_bundle["H_c4_035__raw_ll"][a_i, s_i], float(temps_q3[a_i, s_i])) for s_i in range(len(CONNECTOR_SEEDS))],
```

### Event #18: `item.completed` — `command_execution` (`item_15`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$p='"'research/stage4_math_and_design/M2'; "'$env:UV_CACHE_DIR=(Join-Path (Get-Location) "$p/.uv-cache"); uv python list --only-installed; Get-ChildItem -Path '"'C:/Users/李则徐/AppData/Local/uv/cache/environments-v2' -Directory -Recurse -Depth 4 | Where-Object Name -EQ Scripts | Select-Object -First 20 FullName; Get-ChildItem C:/ -Name; Get-ChildItem Env: | Where-Object Name -match 'PYTHON|VIRTUAL|CONDA' | Select-Object Name,Value"
```

**Aggregated Output:**

```text
cpython-3.13.0-windows-x86_64-none    C:\Users\李则徐\AppData\Local\Programs\Python\Python313\python.exe
cpython-3.12.7-windows-x86_64-none    C:\qazwsx\python.exe
cpython-3.9.19-windows-x86_64-none    C:\376381\miniconda3\python.exe
cpython-3.9.19-windows-x86_64-none    C:\114514\python.exe
114514
13381880071
360Downloads
360��ȫ���������
376381
home
inetpub
PerfLogs
Program Files
Program Files (x86)
qazwsx
SillyTavern-Launcher
SoftMgrbcff1feb
Users
var
Windows
�˸�ʽ¼���ļ�
config.ini
DumpStack.log
logUploaderSettings.ini
logUploaderSettings_temp.ini

Name             Value
----             -----
PYTHONIOENCODING utf-8
PYTHONUTF8       1    


```

### Event #19: `item.completed` — `web_search` (`exec-899b55b5-1fc8-43d1-a273-a90cf870810e`)

```json
{
  "id": "exec-899b55b5-1fc8-43d1-a273-a90cf870810e",
  "type": "web_search",
  "query": "https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf",
  "action": {
    "type": "open_page",
    "url": "https://www.ime.usp.br/~abe/lista/pdf4R8xPVzCnX.pdf"
  },
  "results": [
    {
      "type": "text_result",
      "ref_id": "turn1view0",
      "title": "Internal Error"
    },
    {
      "type": "text_result",
      "domain": "docs.scipy.org",
      "ref_id": "turn1view1",
      "snippet": "Total lines: 132",
      "title": "scipy.stats.nct — SciPy v1.18.0 Manual",
      "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html"
    }
  ]
}
```

### Event #20: `item.completed` — `command_execution` (`item_17`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$c="$env:LOCALAPPDATA/uv/cache"; Get-ChildItem "$c/wheels-v6/pypi/numpy","$c/wheels-v6/pypi/scipy" -Force | Select-Object Name,Mode,Length; Get-ChildItem "$c/simple-v21" -Recurse -File | Where-Object Name -Match '"'numpy|scipy' | Select-Object FullName; Get-ChildItem research/stage4_math_and_design/M2 -Force"
```

**Aggregated Output:**

```text

Name                                    Mode   Length
----                                    ----   ------
2.3.5-cp312-cp312-win_amd64             -a----     27
2.3.5-cp312-cp312-win_amd64.http        -a----    645
2.3.5-cp312-cp312-win_amd64.msgpack     -a----    575
2.5.5-cp312-cp312-win_amd64             -a----     27
2.5.5-cp312-cp312-win_amd64.http        -a----    645
2.5.5-cp312-cp312-win_amd64.msgpack     -a----    575
numpy-2.3.5-cp312-cp312-win_amd64.lock  -a----      0
numpy-2.5.5-cp312-cp312-win_amd64.lock  -a----      0
1.17.0-cp312-cp312-win_amd64            -a----     27
1.17.0-cp312-cp312-win_amd64.http       -a----    646
1.17.0-cp312-cp312-win_amd64.msgpack    -a----   1814
scipy-1.17.0-cp312-cp312-win_amd64.lock -a----      0
                                                     
                                                     
                                                     
                                                     
.uv-cache                               d-----       


```

### Event #21: `item.completed` — `command_execution` (`item_18`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$j=Get-Content -Raw research/stage3d_qwen_signed_phase/results/summary_master.json | ConvertFrom-Json; foreach($c in $j.cells.PSObject.Properties){$v=$c.Value; [pscustomobject]@{cell=$c.Name; B=$v.blocks; accphase=$v.arms.phase.acc; acczero=$v.arms.zero.acc; np=$v.arms.phase.nll; nz=$v.arms.zero.nll; direct=$v.analytic_same_mothers.direct_s17_table.nll; iq=$v.analytic_same_mothers.full_iq_known_param.nll; accSD=$v.seed_dispersion.acc_gain_sd;nllSD=$v.seed_dispersion.nll_improv_sd}}; rg -n '"'ROTATIONS =|q2__' research/stage3d_qwen_signed_phase/analysis_stage3d.py | Select-Object -Last 35"
```

**Aggregated Output:**

```text
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:2 �ַ�: 161
+ ... v=$c.Value; [pscustomobject]@{cell=$c.Name; B=$v.blocks; accphase=$v. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:2 �ַ�: 161
+ ... v=$c.Value; [pscustomobject]@{cell=$c.Name; B=$v.blocks; accphase=$v. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:2 �ַ�: 161
+ ... v=$c.Value; [pscustomobject]@{cell=$c.Name; B=$v.blocks; accphase=$v. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:2 �ַ�: 161
+ ... v=$c.Value; [pscustomobject]@{cell=$c.Name; B=$v.blocks; accphase=$v. ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
403:        q2__raw_ll=np.asarray(raw_q2["q2__raw_ll"], dtype=np.float64),
405:        q2__labels=q2_batch.x1.astype(np.uint8),
406:        q2__z1=z1_q2.astype(np.uint8),
407:        q2__z2=z2_q2.astype(np.uint8),
408:        q2__work_a_ref_probs=work_a_ref,
409:        q2__matched_valid1=matched_sum(q2_batch.p_valid1),
410:        q2__matched_valid2=matched_sum(q2_batch.p_valid2),
411:        q2__matched_indep=matched_sum(q2_batch.p_indep),
```

### Event #22: `item.completed` — `agent_message` (`item_20`)

The Stage 3D code confirms that both arms receive signed `omega` in the prompt. The zero arm removes only the phase slots. That explains how Work F can measure a nonzero sign response in the zero arm. The verifier will measure that text response separately from the phase-slot intervention and test the group-averaging identity on complete orbits.

### Event #23: `item.completed` — `command_execution` (`item_19`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$ErrorActionPreference='"'Stop'
"'$m2=Join-Path (Get-Location) '"'research/stage4_math_and_design/M2'
"'$env:UV_CACHE_DIR=Join-Path $m2 '"'.uv-cache'
"'$env:UV_OFFLINE='"'1'
"'$env:UV_LINK_MODE='"'copy'
"'$env:TMP=Join-Path $m2 '"'.uv-cache'
"'$env:TEMP=$env:TMP
$c=Join-Path $env:LOCALAPPDATA '"'uv/cache'
foreach("'$rel in @('"'wheels-v6/pypi/numpy','wheels-v6/pypi/scipy','archive-v0/3Oxde2-Jkv6KGjnQ','archive-v0/cSnmXl2iWBpNAXGE','simple-v21')) {
  "'$dest=Join-Path $env:UV_CACHE_DIR $rel
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
  Copy-Item -LiteralPath (Join-Path $c $rel) -Destination $dest -Recurse -Force
}
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B -c "import sys,numpy,scipy; print(sys.version,numpy.__version__,scipy.__version__)"
'
```

**Aggregated Output:**

```text
Installed 2 packages in 12.58s
3.12.7 | packaged by Anaconda, Inc. | (main, Oct  4 2024, 13:17:27) [MSC v.1929 64 bit (AMD64)] 2.3.5 1.17.0
```

### Event #24: `item.completed` — `file_change` (`item_21`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M2\\verify_m2.py",
    "kind": "add"
  }
]
```

### Event #25: `item.completed` — `command_execution` (`item_22`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-ChildItem -Force research/stage4_math_and_design/M2 | Select-Object Name,Length; $l=Get-Content research/stage3d_qwen_signed_phase/analysis_stage3d.py; $l[320..364]; $l[380..412]; Get-Content research/stage3c/work_results/E/verify.py -TotalCount 76'
```

**Aggregated Output:**

```text

Name         Length
----         ------
.uv-cache          
verify_m2.py 14829 

    np.savez_compressed(out_dir / "blind_q3_scores.npz", **q3_bundle)

    # Historical C4 +0.35 lookup verification on H_c4_035 positive world
    c4_lookup_path = repo_root / "research" / "stage3b_math" / "A" / "c4_s17_lookup.json"
    c4_lookup = json.loads(c4_lookup_path.read_text(encoding="utf-8"))
    cells_16 = [np.asarray(c["q3"], dtype=np.float64) for c in c4_lookup["cells"]]
    q3_pos_tbl = np.zeros((256, 4), dtype=np.float64)
    for k in range(256):
        z1, z2 = divmod(k, 16)
        q3_pos_tbl[k] = np.roll(cells_16[4 * (z1 % 4) + (z2 % 4)], z1 // 4)
    h_pos_codes = q3_bundle["H_c4_035__codes"][1].reshape(-1)
    h_pos_targets = q3_bundle["H_c4_035__targets"][1].reshape(-1)
    h_lookup_probs = q3_pos_tbl[h_pos_codes]
    h_lookup_sc = score_vectors(h_lookup_probs, h_pos_targets)

    # Export Work F schema-compliant +-0.35 tables for phase & zero arms (seed mean) and run check_f.py
    work_f_results = {}
    temps_q3 = q3_bundle["temps_q3"]
    for a_i, aname in enumerate(ARM_NAMES):
        probs_3 = np.stack(
            [softmax_temperature(q3_bundle["H_c4_035__raw_ll"][a_i, s_i], float(temps_q3[a_i, s_i])) for s_i in range(len(CONNECTOR_SEEDS))],
            axis=0,
        )
        mean_tbl = probs_3.mean(axis=0)
        mean_tbl = mean_tbl / mean_tbl.sum(axis=-1, keepdims=True)
        host_json_path = out_dir / f"work_f_host_035_{aname}.json"
        host_json_path.write_text(
            json.dumps(
                {
                    "task": "Q3_repeat",
                    "encoder_sha": enc_sha,
                    "rate_magnitudes": [0.35],
                    "h": 2,
                    "ell": 2,
                    "sigma_phi": 0.1,
                    "snr_db": 2.0,
                    "sign_order": [-1, 1],
                    "pilot": "none",
                    "cache_order": "16*z1+z2",
                    "probabilities": mean_tbl.tolist(),
                },
                indent=2,
            )
            + "\n",
        ell=2,
        t_p=4.0,
        omegas=(0.35,),
        snr_db=2.0,
        sigma_phi=0.10,
        split_prefix="q2_c4",
    )
    q2_batch = q2_map[0.35]
    z1_q2 = np.asarray(encoder.encode(q2_batch.y1), dtype=np.int64)
    z2_q2 = np.asarray(encoder.encode(q2_batch.y2), dtype=np.int64)
    n_q2 = len(q2_batch.x1)

    kernel = json.loads((repo_root / "research" / "stage3c" / "A" / "q2_s17_kernel.json").read_text(encoding="utf-8"))
    work_a_ref = np.zeros((len(Q2_LOGICAL_REF_IDX), n_q2, 4), dtype=np.float64)
    for arm_name, ref_idx in Q2_LOGICAL_REF_IDX.items():
        refs_tuple, _ = build_q2_arm_references(q2_batch, arm_name)
        for row_i in range(n_q2):
            rec = [(r.acquisition_id, bool(r.shares_channel), complex(matched_sum(r.samples[row_i]))) for r in refs_tuple]
            work_a_ref[ref_idx, row_i] = q2_reader_read(kernel, int(z1_q2[row_i]), int(z2_q2[row_i]), rec)

    np.savez_compressed(
        out_dir / "blind_q2_scores.npz",
        q2__raw_ll=np.asarray(raw_q2["q2__raw_ll"], dtype=np.float64),
        temps_q2=np.asarray(raw_q2["temps_q2"], dtype=np.float64),
        q2__labels=q2_batch.x1.astype(np.uint8),
        q2__z1=z1_q2.astype(np.uint8),
        q2__z2=z2_q2.astype(np.uint8),
        q2__work_a_ref_probs=work_a_ref,
        q2__matched_valid1=matched_sum(q2_batch.p_valid1),
        q2__matched_valid2=matched_sum(q2_batch.p_valid2),
        q2__matched_indep=matched_sum(q2_batch.p_indep),
    )

"""Independent CPU check of the frozen Stage 3C E planning calculation."""
import argparse
from dataclasses import replace
import hashlib
import json
import math
import platform
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
import scipy
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.stats import chi2, nct, norm, t


BLOCK = 64
ALPHA = 0.025
POWER = 0.95
SPECS = {"accuracy": (0.05, 0.02, 0.02), "nll": (0.06, 0.02, 0.03)}
COUNTS = np.arange(2, 4097)
SEED = 310327
REPS = 200_000
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from research.stage3c.verify import compare


def curve(sd, gain, margin, width, use_t):
    excess = (gain - margin) * np.sqrt(COUNTS) / sd
    df = COUNTS - 1
    cutoff = t.isf(ALPHA, df) if use_t else norm.isf(ALPHA)
    power = nct.sf(cutoff, df, excess) if use_t else norm.sf(cutoff - excess)
    plugin = cutoff * sd / np.sqrt(COUNTS)
    c4 = np.exp(gammaln((df + 1) / 2) - gammaln(df / 2) + 0.5 * np.log(2 / df))
    expected = plugin * c4 if use_t else plugin
    # Retain the original SD plug-in planning criterion for the width.
    eligible = COUNTS[(power >= POWER) & (plugin <= width)]
    if not eligible.size:
        raise ValueError("Increase the block search range.")
    return int(eligible[0]), power, plugin, expected


def extract(blocks, curves):
    return {
        name: dict(power=float(power[blocks - 2]), plugin_half_width=float(plugin[blocks - 2]),
                   expected_half_width=float(expected[blocks - 2]))
        for name, (_, power, plugin, expected) in curves.items()
    }


def plan(q, rho, nll_sd, pilot=None):
    deff = 1 + (BLOCK - 1) * rho
    sd = np.sqrt(np.array([q, nll_sd**2]) * deff / BLOCK)
    if pilot is not None:
        sd = np.maximum(sd, pilot)
    curves = [
        {name: curve(s, *spec, use_t) for (name, spec), s in zip(SPECS.items(), sd)}
        for use_t in (False, True)
    ]
    requirements = [{name: entry[0] for name, entry in group.items()} for group in curves]
    normal, exact = [max(32, *r.values()) for r in requirements]
    return dict(q=q, rho=rho, nll_mother_sd=nll_sd, pilot_block_sd=pilot,
                sd=dict(zip(SPECS, sd)),
                normal_by_endpoint=requirements[0], t_by_endpoint=requirements[1],
                normal_blocks=normal, normal_mothers=normal * BLOCK,
                t_blocks=exact, t_mothers=exact * BLOCK,
                normal_at_normal=extract(normal, curves[0]), t_at_normal=extract(normal, curves[1]),
                t_at_selected=extract(exact, curves[1]), t_at_previous=extract(exact - 1, curves[1]))


def crosscheck(design):
    rng = np.random.default_rng(SEED)


```

### Event #26: `item.completed` — `command_execution` (`item_23`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "@'



def q2_analysis(bundle, temps):
    raw = bundle[\"q2__raw_ll\"]
    y = bundle[\"q2__labels\"]
    codes = 16 * bundle[\"q2__z1\"].astype(int) + bundle[\"q2__z2\"]
    refs = bundle[\"q2__work_a_ref_probs\"]
    code_count = histogram(codes, y, 256)
    h_y = float(entropy(np.bincount(y, minlength=4) / len(y)))
    h_z = float(np.average(entropy(divide_counts(code_count)), weights=code_count.sum(axis=-1)))
    row_counts = np.broadcast_to(np.eye(4, dtype=int)[y], raw.shape[1:2] + (len(y), 4))
    result = {\"raw_shape\": list(raw.shape), \"H_Y\": h_y, \"H_Y_given_base_codes\": h_z,
              \"base_code_coverage\": int(np.count_nonzero(code_count.sum(axis=1))),
              \"signed_conditional_MI\": None, \"group_averaging\": None, \"controls\": {}}
    for qi, control in enumerate(stage3.Q2_ARMS):
        # The analytical reader deduplicates the physical acquisition in both duplicate controls.
        reference_key = \"duplicate_copy_dedup\" if control == \"duplicate_copy_raw\" else control
        ref = refs[stage3.Q2_LOGICAL_REF_IDX[reference_key]]
        entry = {\"reference_key\": reference_key, \"reference_metrics\": metrics(ref, y)[0], \"arms\": {}}
        for a, arm in enumerate(ARMS):
            logits = raw[a, :, qi]
            dev = np.stack([stage3.softmax_temperature(z, temp) for z, temp in zip(logits, temps[a])])
            opt, opt_info, shared, shared_info = fit_temperatures(logits, row_counts, temps[a])
            dev_metrics = metrics(dev, y)[0]
            opt_metrics = metrics(opt, y)[0]
            ref_dev = posterior_reference(ref, y, dev)
            ref_opt = posterior_reference(ref, y, opt)
            opt_joint = joint_split(row_counts, opt)
            label_marg = np.bincount(y, minlength=4) / len(y)
            avg_ref = ref.mean(axis=0)
            avg_q = opt.mean(axis=(0, 1))
            prior = float(kl(avg_ref, avg_q))
            realized_prior = float(kl(label_marg, avg_q))
            pieces = {
                \"reference_entropy\": ref_opt[\"reference_entropy\"],
                \"temperature_gap\": dev_metrics[\"nll\"] - opt_metrics[\"nll\"],
                \"reference_marginal_prior_kl\": prior,
                \"reference_reverse_conditional_kl\": ref_opt[\"expected_kl\"] - prior,
                \"realized_label_residual\": ref_opt[\"realized_label_residual\"],
            }
            close(sum(pieces.values()), dev_metrics[\"nll\"], \"Q2 additive path\")
            for seed, record in enumerate(opt_info):
                record[\"T_dev\"] = float(temps[a, seed])
                record[\"dev\"] = metrics(dev[seed], y)[0]
                record[\"opt\"] = metrics(opt[seed], y)[0]
            entry[\"arms\"][arm] = {\"dev\": dev_metrics, \"opt\": opt_metrics, \"shared_opt\": metrics(shared, y)[0],
                                   \"T_star_by_seed\": opt_info, \"single_shared_raw_temperature\": shared_info,
                                   \"additive_path\": pieces, \"reference_at_T_dev\": ref_dev, \"reference_at_T_star\": ref_opt,
                                   \"realized_marginal_prior_kl_at_T_star\": realized_prior,
                                   \"realized_row_joint_KL_at_T_star\": opt_joint,
                                   \"forecast_argmax\": forecast_split(dev, row_counts, \"argmax_set\"),
                                   \"forecast_vector\": forecast_split(dev, row_counts, \"vector\"),
                                   \"NIG_vs_Work_A_realized\": (h_y - dev_metrics[\"nll\"]) / (h_y - entry[\"reference_metrics\"][\"nll\"])}
        phase, zero = (entry[\"arms\"][a] for a in ARMS)
        entry[\"delta_nll_zero_minus_phase\"] = {k: zero[\"additive_path\"][k] - v for k, v in phase[\"additive_path\"].items()}
        total = zero[\"dev\"][\"nll\"] - phase[\"dev\"][\"nll\"]
        close(sum(entry[\"delta_nll_zero_minus_phase\"].values()), total, \"Q2 paired delta\")
        entry[\"delta_nll_zero_minus_phase\"][\"total\"] = total
        entry[\"delta_nll_zero_minus_phase\"][\"at_seedwise_T_star\"] = zero[\"opt\"][\"nll\"] - phase[\"opt\"][\"nll\"]
        result[\"controls\"][control] = entry
    return result


def work_f(bundle, name, temps, seeds, primary):
    def get(key):
        return bundle[f\"{name}__{key}\"]

    extras = {k: get(k) for k in (\"contra_text_flipped_raw_ll\", \"contra_phase_flipped_raw_ll\")} if primary else {}
    text, phase = (extras.get(k) for k in (\"contra_text_flipped_raw_ll\", \"contra_phase_flipped_raw_ll\"))
    scores = stage3.work_f_sign_probe(get(\"raw_ll\"), temps, get(\"codes\")[:, 0],
                                      get(\"targets\")[:, 0], get(\"cal_ref\"), get(\"cal_mass\"),
                                      seeds, text, phase)
    if not primary:
        return scores, None
    # Factorial interventions relative to each physical world, evaluated on identical rows.
    raw = get(\"raw_ll\")[0]
    y, codes = get(\"targets\"), get(\"codes\")
    factorial = {}
    for grouping, ix in ((\"rot0\", slice(0, 1)), (\"all_saved_rotations\", slice(None))):
        variants = {\"matched\": raw, \"text_only_flip\": text, \"slot_only_flip\": phase, \"joint_flip\": raw[:, ::-1]}
        f = {}
        for k, z in variants.items():
            pred = np.stack([stage3.softmax_temperature(seed, temp) for seed, temp in zip(z, temps[0])])
            f[k] = metrics(q3_index(pred, codes[:, ix]), y[:, ix])[0]
        m, a, b, d = (f[k][\"nll\"] for k in variants)
        f[\"nll_effects\"] = {\"text_flip_when_slots_matched\": a - m, \"slot_flip_when_text_matched\": b - m,
                            \"difference_in_differences\": d - a - b + m,
                            \"text_main_effect\": (a - m + d - b) / 2,
                            \"slot_main_effect\": (b - m + d - a) / 2}
        factorial[grouping] = f
    return scores, factorial


def read_json(path):
    return json.loads(path.read_text(encoding=\"utf-8\"))


def run():
    result_dir = SOURCE / \"results\"
    summary = read_json(result_dir / \"summary_master.json\")
    lock = read_json(SOURCE / \"PROTOCOL_LOCK.json\")
    q3_result, score_matrices, parity = {}, {}, {}
    with np.load(result_dir / \"blind_q3_scores.npz\", allow_pickle=False) as q3, \\
         np.load(result_dir / \"blind_q2_scores.npz\", allow_pickle=False) as q2:
        seeds = q3[\"seeds\"].tolist()
        close(seeds, summary[\"seeds\"], \"seed IDs\", atol=0)
        t3, t2 = q3[\"temps_q3\"], q2[\"temps_q2\"]
        for q, temp in ((\"q3\", t3), (\"q2\", t2)):
            close(temp, summary[f\"locked_temps_{q}\"], f\"{q} summary temperatures\", atol=0)
            lock_temp = [[lock[\"checkpoints\"][f\"{a}_s{s}\"][f\"temp_{q}\"] for s in seeds] for a in ARMS]
            close(temp, lock_temp, f\"{q} protocol temperatures\", atol=0)
        for name in (*stage3.CELL_ORDER, \"H_c4_035\"):
            q3_result[name], score_matrices[name] = q3_analysis(q3, name, t3)
            if name not in summary[\"cells\"]:
                continue
            keys = (\"raw_ll\", \"codes\", \"targets\", \"direct_table_probs\", \"full_iq_probs\")
            original = stage3.evaluate_cell_paired(q3[f\"{name}__raw_ll\"], t3,
                *(q3[f\"{name}__{k}\"] for k in keys[1:]), seeds, is_primary=name == stage3.CELL_ORDER[0])
            expected = summary[\"cells\"][name]
            parity[name] = check_tree({**stage3.CELL_META[name], **original}, expected, name)
            for arm in ARMS:
                independent = q3_result[name][\"arms\"][arm][\"metrics\"][\"dev\"]
                close(list(independent.values()), [expected[\"arms\"][arm][k] for k in METRICS], name + arm)
            for endpoint, key in ((\"accuracy\", \"acc_gain_t95\"), (\"nll\", \"nll_improv_t95\")):
                block_adj = score_matrices[name][endpoint].mean(axis=1)
                check_tree(stage3.paired_t_ci(block_adj), expected[\"paired_phase_minus_zero\"][key], name + endpoint)
        primary = stage3.CELL_ORDER[0]
        f, f_factorial = work_f(q3, primary, t3, seeds, True)
        hist, _ = work_f(q3, \"H_c4_035\", t3, seeds, False)
        parity[\"work_f_primary_040\"] = check_tree(f, summary[\"work_f_primary_040\"], \"Work F primary\")
        parity[\"work_f_historical_035\"] = check_tree(hist, summary[\"work_f_historical_035\"], \"Work F historical\")
        work_g = stage3.work_g_rotation_and_controls(q3[f\"{primary}__raw_ll\"], t3,
            *(q3[f\"{primary}__{k}\"] for k in (\"codes\", \"targets\", \"rot10_codes\", \"rot10_targets\",
              \"indep_raw_ll\", \"indep_targets\", \"shuffled_perm\")), seeds)
        parity[\"work_g_primary_040\"] = check_tree(work_g, summary[\"work_g_primary_040\"], \"Work G\")
        for a in ARMS:
            for k, field in ((\"arith\", \"offline_group_avg_16way\"), (\"F0\", \"offline_representative_F0\")):
                independent = q3_result[primary][\"arms\"][a][\"metrics\"][k]
                close([independent[\"acc\"], independent[\"nll\"]], [work_g[\"arms\"][a][field][x]
                      for x in (\"acc\", \"nll\")], f\"independent Work G {a} {k}\")
        q2_result = q2_analysis(q2, t2)
        q2_original = stage3.work_a_q2_analysis(q2, t2, seeds)
        parity[\"work_a_q2_c4\"] = check_tree(q2_original, summary[\"work_a_q2_c4\"], \"Q2 Work A\")
        for control, entry in q2_result[\"controls\"].items():
            for a in ARMS:
                replay = q2_original[\"arms\"][a][\"by_arm\"][control]
                close(list(entry[\"arms\"][a][\"dev\"].values()), [replay[k] for k in METRICS], f\"independent Q2 {control} {a}\")
        inventory = {\"q3\": {k: list(q3[k].shape) for k in q3.files},
                     \"q2\": {k: list(q2[k].shape) for k in q2.files}}
    report = {\"runtime\": {\"python\": sys.version.split()[0], \"numpy\": np.__version__, \"scipy\": scipy.__version__},
              \"Q3\": q3_result, \"Q2\": q2_result, \"npz_shapes\": inventory,
              \"summary_checks_max_absolute_difference\": parity,
              \"work_f_primary_040\": f, \"work_f_historical_035\": hist,
              \"work_f_intervention_factorial\": f_factorial,
              \"power\": power_analysis(score_matrices, lock[\"dev_pilot\"])}
    destination = HERE / \"results_m2.json\"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + \"\\n\", encoding=\"utf-8\")
    print(f\"{destination.name} written. Maximum in-scope summary difference: {max(parity.values()):.3g}.\")


if __name__ == \"__main__\":
    run()

'@ | Add-Content -LiteralPath 'research/stage4_math_and_design/M2/verify_m2.py' -Encoding UTF8"
```

### Event #27: `item.completed` — `command_execution` (`item_24`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$path='"'research/stage4_math_and_design/M2/verify_m2.py'
"'$insert=@'"'
def components(block_seed):
    b, s = block_seed.shape
    mean = block_seed.mean()
    row = block_seed.mean(axis=1)
    col = block_seed.mean(axis=0)
    res = block_seed - row[:, None] - col + mean
    ms_block = s * row.var(ddof=1)
    ms_seed = b * col.var(ddof=1)
    ms_error = np.sum(res**2) / ((b - 1) * (s - 1))
    signed = [(ms_block - ms_error) / s, (ms_seed - ms_error) / b, ms_error]
    var_u, var_seed, var_e = np.maximum(signed, 0)
    return {\"B\": b, \"S\": s, \"mean\": float(mean),
            \"block_conditional_sd\": float(row.std(ddof=1)), \"across_seed_means_sd\": float(col.std(ddof=1)),
            \"seed_means\": col.tolist(), \"mean_squares\": [float(v) for v in (ms_block, ms_seed, ms_error)],
            \"untruncated_components\": [float(v) for v in signed],
            \"var_block\": float(var_u), \"var_seed\": float(var_seed), \"var_interaction\": float(var_e),
            \"seed_sd_95_upper\": float(np.sqrt(ms_seed * (s - 1) / (b * chi2.ppf(.05, s - 1))))}


def fixed_power(d, b, s, v, alpha):
    vu, _, ve = v
    df = b - 1
    return nct.sf(t.isf(alpha, df), df, d * np.sqrt(b / (vu + ve / s)))


@lru_cache(maxsize=1024)
def chisq_nodes(df, order):
    x, w = roots_legendre(order)
    return chi2.ppf((x + 1) / 2, df) / df, w / 2


def crossed_parameters(b, s, v):
    vu, vs, ve = v
    a = (vu + ve / s) / b
    c = (vs + ve / b) / s
    return a, c, a + vs / s


def crossed_power(d, b, s, v, alpha, order=64):
    a, c, variance = crossed_parameters(b, s, v)
    crit = t.isf(alpha, min(b - 1, s - 1))
    x, wx = chisq_nodes(b - 1, order)
    y, wy = chisq_nodes(s - 1, order)
    return float(wx @ norm.sf((crit * np.sqrt(a * x[:, None] + c * y) - d) / np.sqrt(variance)) @ wy)


def min_fixed(d, s, v, alpha, target, max_b=4096):
    b = np.arange(2, max_b + 1)
    good = np.flatnonzero(fixed_power(d, b, s, v, alpha) >= target)
    return int(b[good[0]]) if good.size else None


def min_crossed(d, s, v, alpha, target, max_b=4096):
    sd_floor = np.sqrt(v[1] / s)
    known_ceiling = float(norm.sf(norm.isf(alpha) - d / sd_floor)) if sd_floor else 1.0
    limit = float(nct.sf(t.isf(alpha, s - 1), s - 1, d / sd_floor)) if sd_floor else 1.0
    if known_ceiling < target:
        return {\"B\": None, \"known_variance_power_ceiling\": known_ceiling, \"large_B_power\": limit,
                \"status\": \"unattainable_even_with_known_variance\"}
    # With d fixed, the conditional power is at most P(Z + d/sd_floor > k sqrt(r Y)).
    # For thresholds smaller than d use the floor variance. Above d use probability 1/2
    # as an upper bound, independent of b. This sharper bound applies when b >= s.
    if sd_floor:
        z = d / sd_floor
        k = t.isf(alpha, s - 1)
        y, weights = chisq_nodes(s - 1, 256)
        test_ceiling = float(weights @ np.maximum(.5, norm.sf(k * np.sqrt(y) - z)))
    else:
        test_ceiling = 1.0
    for b in range(2, max_b + 1):
        if b >= s and test_ceiling + 0.001 < target:
            break
        if crossed_power(d, b, s, v, alpha) >= target - .001:
            if crossed_power(d, b, s, v, alpha, 192) >= target:
                return {\"B\": b, \"status\": \"attained\", \"known_variance_power_ceiling\": known_ceiling,
                        \"large_B_power\": limit, \"searched_through\": b}
    return {\"B\": None, \"status\": \"no_design_in_search_range\", \"max_b_requested\": max_b,
            \"searched_through\": b - 1 if b >= s and test_ceiling + .001 < target else max_b,
            \"known_variance_power_ceiling\": known_ceiling, \"large_B_power\": limit,
            \"crossed_power_ceiling_for_B_ge_S_numerical\": test_ceiling}


def monte_carlo(v, b, s, d, alpha, rng, reps):
    a, c, variance = crossed_parameters(b, s, v)
    df = b - 1
    z = rng.normal(size=reps)
    xb = rng.chisquare(df, size=reps) / df
    ys = rng.chisquare(s - 1, size=reps) / (s - 1)
    crossed_crit = t.isf(alpha, min(df, s - 1))
    fixed_crit = t.isf(alpha, df)
    crossed_case = crossed_crit * np.sqrt(a * xb + c * ys)
    fixed_case = fixed_crit * np.sqrt(a * xb)
    decisions = (
        (\"fixed\", z * np.sqrt(a), fixed_case, lambda x: fixed_power(x, b, s, v, alpha)),
        (\"crossed\", z * np.sqrt(variance), crossed_case, lambda x: crossed_power(x, b, s, v, alpha, 192)),
        (\"naive_cluster_t_with_random_seeds\", z * np.sqrt(variance), fixed_case,
         lambda x: nct.sf(fixed_crit * np.sqrt(a / variance), df, x / np.sqrt(variance))),
    )
    result = {}
    for label, mean, threshold, power in decisions:
        tests = {}
        for k, effect in ((\"type_I_boundary\", 0.0), (\"power\", d)):
            empirical = float(np.mean(mean + effect > threshold))
            theoretical = float(power(effect))
            mcse = float(np.sqrt(theoretical * (1 - theoretical) / reps))
            assert abs(empirical - theoretical) < 6 * mcse + .00015, (label, b, effect)
            tests[k] = {\"analytic\": theoretical, \"MC\": empirical, \"MC_SE\": mcse}
        result[label] = tests
    assert result[\"crossed\"][\"type_I_boundary\"][\"analytic\"] <= alpha + .0001
    return result


def multiple_testing_mc(rng, reps, alpha):
    # Valid p-values under the true null, with strong shared dependence across claims.
    settings = {}
    for false in (0, 1, 2):
        z = np.sqrt(.6) * rng.normal(size=(reps, 1)) + np.sqrt(.4) * rng.normal(size=(reps, 3, 3))
        p = norm.sf(z)
        p[:, :, 1:] = 0.0  # least-favorable IUT limit when these endpoints have power one
        p[:, :false, :] = 0.0
        p_iut = p.max(axis=-1)
        ix = np.argsort(p_iut, axis=1)
        ordered = np.take_along_axis(p_iut, ix, axis=1)
        rejected = np.logical_and.accumulate(ordered <= alpha / np.arange(3, 0, -1), axis=1)
        fwer = float(np.mean(np.any(rejected & (ix >= false), axis=1)))
        assert fwer <= alpha + .003
        settings[f\"false_claim_nulls_{false}\"] = {\"FWER_MC\": fwer,
            \"IUT_marginal_type_I_MC\": float(np.mean(p_iut[:, false] <= alpha))}
    independent_all_boundary = float(np.mean(np.max(rng.random((reps, 3)), axis=1) <= alpha))
    assert abs(independent_all_boundary - alpha**3) < .0005
    return {\"IUT_at_independent_three_endpoint_boundaries\": {\"exact\": alpha**3, \"MC\": independent_all_boundary},
            \"correlated_Holm_IUT\": settings}


def stage3_lock_check(pilot):
    scenario = pilot[\"work_e_scenario\"]
    curves = {}
    b = np.arange(2, 4097)
    df = b - 1
    alpha = .025
    for i, (k, gain, width) in enumerate(((\"accuracy\", .05, .02), (\"nll\", .06, .03))):
        vm = scenario[\"q\"] if i == 0 else scenario[\"nll_mother_sd\"]**2
        sd = max(np.sqrt(vm * (1 + 63 * scenario[\"rho\"]) / 64), scenario[\"pilot_block_sd\"][i])
        close(sd, scenario[\"sd\"][k], \"Stage 3D pilot SD\")
        crit = t.isf(alpha, df)
        power = nct.sf(crit, df, (gain - .02) * np.sqrt(b) / sd)
        eligible = b[(power >= .95) & (crit * sd / np.sqrt(b) <= width)]
        assert eligible[0] == scenario[\"t_by_endpoint\"][k]
        selected = pilot[\"b_locked\"] - 2
        close(power[selected], scenario[\"t_at_selected\"][k][\"power\"], \"Work E selected NCT power\", atol=2e-9)
        curves[k] = {\"sd\": sd, \"min_B\": int(eligible[0]), \"power_at_131\": float(power[selected])}
    return curves


def power_analysis(matrices, pilot):
    alpha, target, s = .05 / 3, .95, 3
    rng = np.random.default_rng(SEED)
    reps = 200_000
    estimates = {cell: {endpoint: components(d) for endpoint, d in entry.items()} for cell, entry in matrices.items()}
    result = {\"stage3_lock_replay\": stage3_lock_check(pilot), \"planning_alpha\": alpha,
              \"target_marginal_power\": target, \"seed_count\": s, \"MC_seed\": SEED, \"MC_reps\": reps,
              \"variance_components\": estimates, \"scenario_proxies\": {},
              \"multiple_testing_check\": multiple_testing_mc(rng, reps, .05), \"synthetic_cell_plans\": {}}
    g1 = estimates[stage3.CELL_ORDER[0]]
    variance = lambda est: tuple(est[k] for k in (\"var_block\", \"var_seed\", \"var_interaction\"))
    for cell, entry in estimates.items():
        result[\"synthetic_cell_plans\"][cell] = {}
        for endpoint, est in entry.items():
            d = .03 if endpoint == \"accuracy\" else .04
            v = variance(est)
            result[\"synthetic_cell_plans\"][cell][endpoint] = {
                \"effect_above_margin\": d, \"fixed_min_B\": min_fixed(d, s, v, alpha, target),
                \"random_min_B\": min_crossed(d, s, v, alpha, target),
                \"fixed_power_at_131\": float(fixed_power(d, 131, s, v, alpha)),
                \"random_power_at_131\": crossed_power(d, 131, s, v, alpha, 192)}
    designs = [(f\"B{b}_independent\", b, 1, 0) for b in (16, 72, 131)]
    designs.extend((f\"CTTC_9_sessions_x8_rho_{rho:.2f}\", 9, 8, rho) for rho in (0, .1, .25))
    for design, b, per_group, rho in designs:
        design_result = {}
        for ep, est in g1.items():
            vu, vs, ve = variance(est)
            v = (vu * (rho + (1 - rho) / per_group), vs, ve / per_group)
            d = .03 if ep == \"accuracy\" else .04
            ef = max(est[\"mean\"] - .02, 0)
            design_result[ep] = {\"independent_groups\": b, \"clusters_per_group\": per_group,
                \"assumed_block_effect_session_ICC\": rho, \"variance_components_at_group_level\": list(v),
                \"scenarios\": {key: {\"excess\": x, \"fixed_min_groups\": min_fixed(x, s, v, alpha, target) if x else None,
                                      \"random_min_groups\": min_crossed(x, s, v, alpha, target) if x else None,
                                      \"fixed_power_at_available\": float(fixed_power(x, b, s, v, alpha)),
                                      \"random_power_at_available\": crossed_power(x, b, s, v, alpha, 192)}
                              for key, x in ((\"planning\", d), (\"0p02_excess\", .02), (\"G1_observed_excess\", ef))},
                \"MC_at_planning_excess\": monte_carlo(v, b, s, d, alpha, rng, reps)}
        result[\"scenario_proxies\"][design] = design_result
    result[\"G1_more_seeds\"] = {str(num): {ep: {
        \"B131\": crossed_power(.03 if ep == \"accuracy\" else .04, 131, num, variance(est), alpha, 192),
        \"B16\": crossed_power(.03 if ep == \"accuracy\" else .04, 16, num, variance(est), alpha, 192)}
        for ep, est in g1.items()} for num in (3, 5, 6, 8, 10, 12)}
    return result



'@
"'$content=Get-Content -Raw -Encoding UTF8 -LiteralPath $path
$content.Replace('"'def run():',"'$insert+"`n`ndef run():") | Set-Content -Encoding UTF8 -LiteralPath $path'
```

### Event #28: `item.completed` — `agent_message` (`item_26`)

The verifier now includes two distinct inference targets: the three frozen connectors and a population of retrained connectors. The second target retains seed variance even as more test blocks are added. I’ll report both and treat CTTC’s nine sessions as the independent units for claims about new sessions.

### Event #29: `item.completed` — `command_execution` (`item_25`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$ErrorActionPreference='"'Stop'
"'$m2=Join-Path (Get-Location) '"'research/stage4_math_and_design/M2'
"'$p=Join-Path $m2 '"'verify_m2.py'
(Get-Content -Raw -Encoding UTF8 -LiteralPath "'$p).Replace('"'rng.normal(size=(reps, 1))','rng.normal(size=(reps, 1, 1))') | Set-Content -Encoding UTF8 -LiteralPath "'$p
$env:UV_CACHE_DIR=Join-Path $m2 '"'.uv-cache'
"'$env:UV_OFFLINE='"'1'
"'$env:UV_LINK_MODE='"'copy'
"'$env:TMP=$env:UV_CACHE_DIR
$env:TEMP=$env:UV_CACHE_DIR
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M2/verify_m2.py'
```

**Aggregated Output:**

```text
Traceback (most recent call last):
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M2\verify_m2.py", line 660, in <module>
    run()
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M2\verify_m2.py", line 611, in run
    q3_result[name], score_matrices[name] = q3_analysis(q3, name, t3)
                                            ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M2\verify_m2.py", line 268, in q3_analysis
    "reference_loss_split": {k: posterior_reference(v, y, p_rows) for k, v in oracles.items()},
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M2\verify_m2.py", line 203, in posterior_reference
    nll = float(-np.mean(np.log(np.take_along_axis(model, labels[..., None], axis=-1))))
                                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M2\.uv-cache\archive-v0\JiHBfPRTl1ebuKou\Lib\site-packages\numpy\lib\_shape_base_impl.py", line 182, in take_along_axis
    return arr[_make_along_axis_idx(arr_shape, indices, axis)]
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M2\.uv-cache\archive-v0\JiHBfPRTl1ebuKou\Lib\site-packages\numpy\lib\_shape_base_impl.py", line 39, in _make_along_axis_idx
    raise ValueError(
ValueError: `indices` and `arr` must have the same number of dimensions
```

### Event #30: `item.completed` — `command_execution` (`item_27`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "@'
# Track M2 | Stage 3D score forensics and the Stage 4 gate

Round 1. Losses and information are in nats. Positive paired differences favor phase, so the accuracy contrast is phase minus zero and the NLL contrast is zero minus phase. Each saved mother carries its signs and rotations together. Each of the three connector seeds has equal weight. All calibration fitted on blind labels is retrospective.

## 1. Which Bayes risk is being estimated?

For a population law \\(P\\), write \\(p(c)=P(Y=\\cdot\\mid C=c)\\) and \\(L(q)=\\mathbb E[-\\log q_Y(C)]\\). With finite support and positive model probabilities,

\\[
L(q)=H(Y\\mid C)+\\mathbb E_C D_{\\rm KL}\\"'!'"\\left(p(C)\\Vert q(C)\\right).
\\]

This follows by adding and subtracting \\(-\\log p_Y(C)\\) inside the conditional expectation. It is the Bregman decomposition for negative Shannon entropy. If there is a finer observation \\(X\\) and \\(C\\) is a function of \\(X\\), then

\\[
L(q)=H(Y\\mid X)+I(Y;X\\mid C)+\\mathbb E_C D_{\\rm KL}(p(C)\\Vert q(C)).
\\]

The population entropy, the information lost by coarsening, and the conditional forecast error depend on the named observation. In Q3, \\(C=(S,Z)\\) is the physical/told CFO sign and the shared S17 code pair at a fixed condition. Both arms are functions of \\(C\\) and the fitted seed \\(J\\). Under the empirical law \\(J\\) is uniform and independent of \\((C,Y)\\). Therefore \\(H(Y\\mid C,J)=H(Y\\mid C)\\). The same Bayes term cancels exactly in the arm contrast. In particular the Q3 input-resolution difference is zero. Improving a forecast can still use existing information more effectively.

The verifier uses three distinct reference laws.

- **Empirical signed-code posterior.** Entropy of blind counts pooled over the prescribed three rotations, separately by \\((S,Z)\\). It gives an exact replay identity and an in-sample Bayes risk for that empirical law. Estimation and scoring reuse the labels. Sparse codes and especially H make the plug-in conditional MI optimistic as population evidence.
- **S17 direct readout.** A physical predictive baseline using encoder representatives and known condition. Its realized loss is scored on the same mothers. The code does not prove that integrating the population distribution over a quantizer cell equals scoring the representative.
- **Known-parameter full IQ.** A finer physical reference. Its mean soft entropy and realized NLL differ in finite samples. The identity \\(L_{\\rm realized}(q)=\\overline{H(r)}+\\overline{D_{\\rm KL}(r\\Vert q)}+\\overline{\\sum_y(r_y-1_{Y=y})\\log q_y}\\) holds with the sign of the last term as written, for any saved reference \\(r\\). The last term has population conditional mean zero if \\(r\\) is the true posterior on the observation set. The JSON reports it for both references.

The empirical CFO information is \\(I(Y;S\\mid Z)=H(Y\\mid Z)-H(Y\\mid S,Z)\\). The code counts supply both conditional entropies. This is the information available to a Bayes decoder with the sign metadata, independent of which connector is used. Additional diagnostics use the score vector \\(Q\\) as a covariate. Then \\(L(Q)=H(Y)-I(Y;Q)+\\mathbb E_Q D_{\\rm KL}(P(Y\\mid Q)\\Vert Q)\\). With the fixed feature \\(A\\) equal to the argmax set of \\(Q\\), the identity is \\(L(Q)=H(Y\\mid A)-I(Y;Q\\mid A)+\\mathbb E_Q D_{\\rm KL}(P(Y\\mid Q)\\Vert Q)\\). Argmax resolution reports a coarse, model-dependent predictive association. Exact-float score-vector groups can identify a code or even a pilot row, so their empirical resolution can be high for a poor decoder. Neither diagnostic identifies a phase-slot mechanism.

### Scalar calibration and marginal bias

For logits \\(\\ell_i\\), inverse temperature \\(\\beta\\ge 0\\), and empirical weights \\(w_i\\), let

\\[
f(\\beta)=\\sum_i w_i\\big[\\log\\sum_j e"'^{'"\\beta \\ell_{ij}}-\\beta\\ell_{iY_i}\\big].
\\]

Then \\(f'=\\sum_i w_i(\\mathbb E_{q_\\beta(\\cdot\\mid i)}\\ell_{ij}-\\ell_{iY_i})\\) and \\(f''=\\sum_i w_i\\operatorname{Var}_{q_\\beta(\\cdot\\mid i)}\\ell_{ij}\\ge0\\). If \\(f'(0)\\ge0\\), \\(\\beta"'^*=0'"\\) and the optimal scalar forecast is the uniform vector. Otherwise solve \\(f'(\\beta"'^*)=0'"\\) for an interior optimum when present. The gap at the lock is exactly

\\[
f(\\beta_d)-f(\\beta"'^*)=
'"\\mathbb E_i D_{\\rm KL}(q_{\\beta"'^*}('"\\cdot\\mid i)\\Vert q_{\\beta_d}(\\cdot\\mid i))
+(\\beta_d-\\beta"'^*)f'"'(\\beta"'^*).
'"\\]

The boundary derivative term generally remains at \\(T"'^*='"\\infty\\). A finite positive \\(T\\) preserves each row's ranking, so ECE and NLL can change while accuracy stays fixed. At infinity the four-way tie awards fractional accuracy \\(1/4\\). Reported oracle fits are label-optimal scalar temperatures, separately for each arm and seed. The JSON also fits a single common raw temperature per arm and cell. ECE uses the same pooled ten top-confidence bins as Stage 3D. ECE is neither proper nor an additive component of log risk.

For any fixed predictor \\(q\\), define two joints \\(P(c,y)=w(c)p_y(c)\\) and \\(\\widetilde Q(c,y)=w(c)q_y(c)\\), where \\(c\\) includes seed when fitting multiple connectors. The chain rule in reverse order yields

\\[
\\mathbb E_C D_{\\rm KL}(p\\Vert q)
=D_{\\rm KL}(\\bar y\\Vert\\bar q)
+\\mathbb E_{Y\\sim\\bar y}D_{\\rm KL}\\big(P(C\\mid Y)\\Vert\\widetilde Q(C\\mid Y)\\big).
\\]

The marginal term measures global prior bias. The second term measures conditional allocation error, encompassing misranking, miscalibration, and other errors. A learned intercept adjustment need not realize the marginal term as a standalone reduction in NLL.

### The reported additive path

Let \\(q_d\\) use the locked temperatures, \\(q_*\\) use the post-hoc fitted temperatures and \\(Pq_*\\) denote the arithmetic projection. For the empirical Q3 design the reported identity is

\\[
L(q_d)=H(Y\\mid S,Z)+[L(q_d)-L(q_*)]+[L(q_*)-L(Pq_*)]
+D_{\\rm KL}(\\bar y\\Vert\\overline{Pq_*})
+\\mathbb E_Y D_{\\rm KL}(P(C\\mid Y)\\Vert\\widetilde Q_{Pq_*}(C\\mid Y)).
\\]

Take zero minus phase termwise. Empirical \\(H(Y\\mid S,Z)\\) cancels, and the conditional mismatch difference may be signed. This path specifies the order of the two interventions. Temperature and group averaging do not commute. At locked \\(T\\), the JSON additionally gives the raw marginal KL, the arithmetic group gap, and its complete-orbit counterpart. In the actual design the projection bracket includes a distributional correction and can have either sign. A causal allocation of the gain to phase slots and an orthogonal temperature/projection/prior ANOVA do not follow from the log-loss identity.

For Q2 each control has one physical sign, and the host score depends on row-specific references. The file records selected matched reference summaries and physical predictions but has no counterfactual sign or full rotated forecast orbit for each reference acquisition. \\(H(Y\\mid Z)\\) from the no-pilot code is reported as an empirical anchor. The exact Q2 path on every control uses the Work A soft reference \\(r\\) and writes the locked loss as \\(\\overline{H(r)}+\\text{temperature gap}+\\text{reference marginal KL}+\\text{reference conditional KL}+\\text{realized label residual}\\). The last three terms are evaluated at \\(q_*\\). Raw-duplicate and deduplicated controls share the same Work A physical reader under logical acquisition deduplication. A row-ID-only empirical posterior is the point-mass label and gives a tautological zero entropy. Full-vector forecast grouping has a related finite-sample problem. A Q2 CFO MI or group penalty would require additional crossed scores.

## 2. Group averaging and the top-1 loss

Let \\(R_gq(x)=\\rho(g)"'^{-1}q(gx)'"\\) for a label permutation representation. Assume the joint law of \\((X,Y)\\) is invariant under \\((x,y)\\mapsto(gx,\\rho(g)y)\\) and all evaluated probabilities are positive. Then \\(L(R_gq)=L(q)\\). For \\(m=Pq=|G|"'^{-1}'"\\sum_g R_gq\\), define \\(J_y(x)=\\log m_y(x)-|G|"'^{-1}'"\\sum_g\\log (R_gq)_y(x)\\). Pointwise \\(J_y\\ge0\\) by strict concavity of \\(\\log\\). Hence

\\[
L(q)-L(Pq)=\\mathbb E_X\\sum_y p_y(X)J_y(X)\\ge0.
\\]

Equality holds exactly when, almost surely, every label of positive posterior mass has the same probability across every aligned member. If the posterior has full support, this is equivariance almost surely. On an arbitrary finite design the exact formula adds \\(L(q)-|G|"'^{-1}'"\\sum_g L(R_gq)\\) to the label-weighted Jensen gap. The complete orbits constructed from each saved mother have an invariant design by construction. Saved rotations are an incomplete orbit, so the restriction matters.

The proposed \\(|G|"'^{-1}'"\\sum_g D_{\\rm KL}(Pq\\Vert R_gq)\\) is \\(\\sum_y m_y J_y\\) pointwise. It corresponds to self-weighting by the pool. It equals the log-loss gain if \\(\\mathbb E[(p-m)\\cdot J]=0\\), in particular if \\(p=m\\). It is not a general equality even under exact invariance. For an invariant binary example, take two equiprobable observations interchanged by a group of order two with identity label action and \\(p=(1,0)\\) on both. The forecasts are \\((0.8,0.2)\\) and \\((0.4,0.6)\\). The pool is \\((0.6,0.4)\\), and \\(J_1=\\log(0.6/\\sqrt{0.32})\\), \\(J_2=\\log(0.4/\\sqrt{0.12})\\) differ. The loss gain equals \\(J_1\\). The reverse-KL mean equals \\(0.6J_1+0.4J_2\\).

For geometric pooling let \\(h_y=\\exp(|G|"'^{-1}'"\\sum_g\\log(R_gq)_y)\\), \\(Z_h=\\sum_y h_y\\), and \\(q_{\\rm geo}=h/Z_h\\). Componentwise AM–GM gives \\(Z_h\\le1\\), and

\\[
L(q_{\\rm geo})=|G|"'^{-1}'"\\sum_g L(R_gq)+\\mathbb E_X\\log Z_h.
\\]

Thus both pools weakly improve invariant-design log risk over the raw model. Their ordering depends on the label law and the forecasts. Normalization makes the geometric pool the minimizer of \\(\\sum_g D_{\\rm KL}(r\\Vert R_gq)\\) over \\(r\\). It is equivariant, and applying it again returns the same distribution. The arithmetic pool minimizes \\(\\sum_g D_{\\rm KL}(R_gq\\Vert r)\\) and is also equivariant and idempotent.

For each target \\(y\\), the closed top-1 region is the intersection of halfspaces \\(r_y-r_k\\ge0\\) over \\(k\\ne y\\). For the arithmetic path, the exact margin to competitor \\(k\\) at the barycenter is the mean of aligned margins. A unique raw correct decision is lost iff a mean margin is negative or hits a tie boundary. A unique correct decision is gained iff all mean margins are positive after a raw error or tie. The fractional-tie hit function at a row is

\\[
h_y(r)=\\frac{1[\\min_{k\\ne y}(r_y-r_k)\\ge0]}{|\\arg\\max_j r_j|}.
\\]

Over a target distribution, strict accuracy decline is equivalent to \\(\\mathbb E[(h_y(q)-h_y(Pq))_+]>\\mathbb E[(h_y(Pq)-h_y(q))_+]\\), including ties. At population level this is the same as \\(\\mathbb E_X[p(X)\\cdot(d(q)-d(Pq))]>0\\), where \\(d\\) is uniform on the maximizer set. For geometric pooling the corresponding competitor margins are averages of \\(\\log r_y-\\log r_k\\). Log score depends on the probability assigned to the true class, while top-1 depends on which side of the decision planes the pool lies.

## 3. Stage 4 frozen statistical gate

A claim passes when every endpoint for that claim clears its limit. Freeze the candidate, seed handling, allowed observation sets, per-class weighting, code below, multiplicity family, score floor, calibration, and all margins before unblinding. Apply the same three gate parts separately to the synthetic extrapolation claim, the Pluto claim and the CTTC claim. For a claim covering all four synthetic cells, require each cell to pass. Report each failed or indeterminate endpoint. A winner picked from several candidates requires a design split or inclusion in the multiplicity family.

1. **Absolute and comparative performance.** Use paired lower bounds. Require NLL to beat uniform, the dev-frozen prior, and the matched zero-phase system, with a \\(0.02\\)-nat increment proposed against matched zero. Require accuracy above analytic chance, a dev-frozen prior classifier, and the matched zero-phase system, with \\(+0.02\\) absolute over chance and zero proposed as initial margins. With imbalanced classes, also require outperformance of the fixed majority prediction or used prior competitor, and report balanced accuracy as a separately locked estimand. Require valid chance/random-label randomization tails for the accuracy and NLL statistics where the design supplies exchangeability. Choose an oracle with the same information budget as the candidate for the main NIG denominator and disclose a finer physical oracle separately. Propose \\(\\eta_{\\min}=0.10\\) and validate \\(H(Y)-L(o)>0\\) on an independent development design or by a co-primary lower bound. Test the linear contrast \\(D_{\\eta}=(1-\\eta)H(Y)-L(q)+\\eta L(o)>0\\) rather than dividing two noisy quantities. Use \\(H(Y)=\\log K\\) for a genuinely uniform target law. Otherwise predefine the target distribution or carry uncertainty in \\(H(Y)\\) by a joint confidence region and minimize the contrast over it. For real data without a defensible Bayes oracle, freeze a physically informed predictive reference, call the resulting metric reference-normalized, and withhold the Bayes-normalized claim. Requiring a learner to outperform a true Bayes oracle under proper loss is infeasible. A positive NIG floor defines a fraction of the attainable improvement. Controlled slot/text interventions form a further prerequisite for a specifically signed-slot mechanism claim. Require improvement with matched versus permuted or flipped slots at fixed text and same \\(X\\), with seeded training and ablations paired. Use the native receiver outputs for the gate.
2. **Calibration and structure.** Propose an upper bound of 0.05 on top-label ECE after dev-only calibration with ten frozen bins. Propose a paired Brier improvement of at least 0.005 against the strongest locked trivial or zero-phase Brier competitor, plus a strict improvement over every required comparator. For \\(K=4\\) uniform Brier is \\(1-1/K=0.75\\) with the unnormalized multiclass sum. Prelock a probability transformation ensuring finite NLL, for example \\(q_\\epsilon=(1-K\\epsilon)q+\\epsilon\\) with small fixed \\(\\epsilon\\), and score that forecast consistently. For calibration inference set \\(d_j=\\mathbb E[1_{\\{\\operatorname{conf}\\in I_j\\}}(h_Y-\\operatorname{conf})]\\); \\(\\operatorname{ECE}=\\sum_j|d_j|=\\max_{v\\in\\{-1,1\\}"'^{10}}v''^'"\\top d\\). Test all \\(2"'^{10}'"\\) linear upper-margin conditions as an IUT or use a valid joint confidence region for \\(d\\). Estimating the absolute value first and treating empirical ECE as an unbiased, normal per-mother loss has no exact justification. Require \\(\\max_{g,x}\\|q(gx)-\\rho(g)q(x)\\|_\\infty\\le10"'^{-5}'"\\) and projection/readout idempotence \\(\\max_x\\|P(Pq)(x)-Pq(x)\\|_\\infty\\le10"'^{-5}'"\\). Verify across the whole finite code domain and all group elements on synthetic. For continuous real inputs distinguish formal architectural guarantees from the observed maximum on prespecified transforms. Test only valid actions with physically correct label permutations. Audit the deployed path and the offline projector separately.
3. **Hierarchy and multiplicity.** Keep all repeats, rotations, acquisitions, and matched arms inside their mother and higher cluster. Report a frozen-seed conditional estimate and, if claiming generalization to retraining, a separate random-seed inference. Use equal independent cluster weights for the following exact balanced analysis. Pluto supplies 16 units if they are independent captures. For a CTTC new-session claim use nine independent session means, with all eight clusters per session contributing to that session. A conditional claim on the fixed sessions needs the corresponding estimand and defensible within-session cluster independence. For a permutation or label-randomization gate, lock the allowed exchangeability orbit and recompute the same statistic throughout. Repeated/rotated labels move jointly with their mother and never split correlated session units as though independent. A synthetic randomized design with independent label assignments can instead rerandomize the sanctioned generator operation inside the original design. When there is no justified exchangeability, collect a randomized negative control. State whether it tests a sharp random-label null or a mean-effect margin. A permutation p-value \\((1+\\#\\{T_{\\rm perm}\\ge T_{\\rm obs}\\})/(R+1)\\) with uniform randomizations and ties included is conservative under the sharp design. Use exact enumeration when feasible.

### Exact type-I control and power under a balanced Gaussian model

For an oriented block contrast \\(D_{bs}\\) let \\(D_{bs}=\\mu+u_b+\\nu_s+e_{bs}\\), with independent Gaussian families of variances \\(\\sigma_u"'^2,'"\\sigma_\\nu"'^2,'"\\sigma_e"'^2'"\\). These are the assumptions for the exact parametric calculations. Blocks already aggregate mothers and correlated views. If \\(m\\) mothers in a block have common variance \\(v\\) and exchangeable within-block correlation \\(\\rho\\), a block mean contributes \\(v[1+(m-1)\\rho]/m\\); this design effect applies to the particular contrast and variance component being modeled. The Stage 3D Work E lock estimated block SD on a fixed-seed pilot, took the larger SD from pilot upper bound and design-effect scenario and used the conditional noncentral t. It did not price uncertainty across retrains.

For \\(B\\) blocks and \\(S\\) seeds,

\\[
V=\\operatorname{Var}(\\bar D)=\\sigma_u"'^2/B+'"\\sigma_\\nu"'^2/S+'"\\sigma_e"'^2/(BS).
'"\\]

Orthogonal row, column, and residual contrasts yield mutually independent ANOVA terms \\(MS_B/(\\sigma_e"'^2+S'"\\sigma_u"'^2)'"\\sim\\chi"'^2_{B-1}/(B-1)'"\\), \\(MS_S/(\\sigma_e"'^2+B'"\\sigma_\\nu"'^2)'"\\sim\\chi"'^2_{S-1}/(S-1)'"\\), and \\(MS_E/\\sigma_e"'^2'"\\sim\\chi"'^2_{(B-1)(S-1)}/[(B-1)(S-1)]'"\\), independent of \\(\\bar D\\). The corresponding method-of-moments component estimates are \\((MS_B-MS_E)/S,(MS_S-MS_E)/B,MS_E\\). The reported planning plug-in truncates negative estimates to zero and retains the raw estimates. \\(MS_S(S-1)/(B\\chi"'^2_{S-1,0.05})'"\\) is a conservative 95% upper bound for \\(\\sigma_\\nu"'^2'"\\). Three seeds give only two independent seed degrees of freedom. Large \\(B\\) cannot remove the \\(\\sigma_\\nu"'^2/S'"\\) floor.

For the **frozen seeds**, set \\(v_F=\\sigma_u"'^2+'"\\sigma_e"'^2/S'"\\). The ordinary one-sided block t-test has exact boundary size \\(\\alpha\\) for the fixed-seed mean and power \\(1-F_{\\mathrm{nct}}(t_{B-1,1-\\alpha};B-1,d\\sqrt{B/v_F})\\) at true excess \\(d=\\mu-\\delta\\) above the margin, with the mean interpreted conditionally. If one applies this t-test to a population-of-seeds claim, the cutoff for the exact unconditional noncentral t is rescaled by \\(\\sqrt{(v_F/B)/V}\\); the nominal type-I error can then be badly inflated. [SciPy noncentral t definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html)

For **random seeds**, a finite-level test with unknown Gaussian variance components uses
\\[
 W=(MS_B+MS_S)/(BS)=A X+C Y,\\quad
 A=(\\sigma_u"'^2+'"\\sigma_e"'^2/S)/B,'"\\quad C=(\\sigma_\\nu"'^2+'"\\sigma_e"'^2/B)/S,
'"\\]
where \\(X,Y\\) are the independent normalized row and column chi-squares. Reject when \\(\\bar D-\\delta>k\\sqrt W\\) with \\(k=t_{\\min(B-1,S-1),1-\\alpha}\\). At the boundary the exact rejection probability, also the power formula at excess \\(d\\), is
\\[
\\pi(d)=\\mathbb E_{X,Y}\\overline\\Phi\\"'!'"\\left(\\frac{k\\sqrt{A X+C Y}-d}{\\sqrt V}\\right).
\\]
To prove size control, note \\(A+C=V+\\sigma_e"'^2/(BS)'"\\ge V\\) and \\(x\\mapsto\\overline\\Phi(k\\sqrt x)\\) decreases and is convex for \\(x>0\\). Jensen implies \\(\\pi(0)\\le\\frac{A}{A+C}\\Pr(t_{B-1}>k)+\\frac{C}{A+C}\\Pr(t_{S-1}>k)\\le\\alpha\\). The supremum \\(\\alpha\\) is attained as the smaller-df variance component dominates. With known component ratios use the exact nuisance-specific mixture cutoff. The unconstrained method-of-moments estimate \\((MS_B+MS_S-MS_E)/(BS)\\) can be negative and has a signed chi-square mixture law. Plugging it into a t distribution would not be an exact test.

For real data under the same Gaussian model with \\(J_0\\) independent sessions of \\(m\\) nested clusters and separate session and cluster effects, the session means have
\\[
V=\\sigma_{\\rm session}"'^2/J_0+'"\\sigma_{\\rm cluster}"'^2/(J_0m)+'"\\sigma_\\nu"'^2/S+
'"\\sigma_e"'^2/(J_0mS).
'"\\]
Use the same formulas with \\(B=J_0\\), \\(\\sigma_u"'^2='"\\sigma_{\\rm session}"'^2+'"\\sigma_{\\rm cluster}"'^2/m'"\\), and \\(\\sigma_e"'^2'"\\) replaced by \\(\\sigma_e"'^2/m'"\\); an additional session-by-seed effect is another contribution to the collapsed interaction variance. Use planned balanced session-level estimation or a separately validated unequal-weight design. For a fixed-seed claim with independent bounded cluster means and arbitrary within-cluster dependence, a one-sided Hoeffding bound gives an alternative to Gaussian t inference. For mean of \\(B\\) equal-weight cluster means in \\([l,h]\\), reject when the empirical excess exceeds \\((h-l)\\sqrt{\\log(1/\\alpha)/(2B)}\\). For fixed weights \\(a_b\\) summing to one, replace \\(1/B\\) by \\(\\sum_b a_b"'^2'"\\). Unknown non-Gaussian random seed effects with only three seeds do not admit a generic exact t power calculation. Correlation modeling and null randomization require separate justifications.

For endpoint nulls \\(H_{0j}\\) and a joint claim, the null is their union. Valid p-values give \\(p_{\\rm IUT}=\\max_j p_j\\), since with at least one true null the event \\(p_{\\rm IUT}\\le a\\) is a subset of rejection of that true null. The exact joint rejection probability is the joint tail of the endpoint statistics, equal to the product of marginal powers only under independence. Its universal lower bound under the alternative is \\(1-\\sum_j(1-\\pi_j)\\). Across separately reportable claims, sort the IUT p-values and apply Holm's thresholds \\(\\alpha/(M-i+1)\\). If a first true null is rejected, its p-value is at most \\(\\alpha/m_0\\) where \\(m_0\\) is the number of true claim nulls. The union bound over those true nulls gives strong FWER \\(\\le\\alpha\\) under arbitrary dependence. This is a finite-sample guarantee conditional on valid endpoint tests. Under independence with all nulls uniform, the probability of any rejection is exactly \\(1-(1-\\alpha/M)"'^M'"\\). [Holm (1979)](https://www.jstor.org/stable/4615733)

For simple variance planning at known components, if \\(z=z_{1-\\alpha}+z_{1-\\beta}\\), power \\(1-\\beta\\) needs
\\[
B\\ \\ge\\ \\frac{\\sigma_u"'^2+'"\\sigma_e"'^2/S}{d''^2/z''^2-'"\\sigma_\\nu"'^2/S}
'"\\]
with positive denominator. At fixed \\(S\\) this is a necessary known-variance bound for the random-seed model. The verifier solves the exact t power and evaluates the crossed chi-square integral for finite counts. It uses a worst Holm first-step threshold \\(\\alpha=0.05/3\\) and marginal power 0.95 to guarantee at least 0.90 power for the two historical paired endpoints if both assumptions and excess effects hold. This target does not imply 0.90 power for the full Stage 4 gate. Full planning requires variance and target excess for every required absolute/calibration endpoint and a joint power calculation or an explicit allocation of endpoint type-II errors. Stage 3D supplies a synthetic variance reference. It cannot measure Pluto or CTTC variability.


'@ | Set-Content -LiteralPath 'research/stage4_math_and_design/M2/THEOREM_M2.md' -Encoding UTF8"
```

### Event #31: `item.completed` — `command_execution` (`item_29`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n 'argmax resolution|Argmax resolution|math.ceil|NIG|design' research/stage4_math_and_design/M2/THEOREM_M2.md | Select-Object -First 12; git status --short --untracked-files=normal; Get-ChildItem research/stage4_math_and_design/M2 -Force | Select-Object Name,Length"
```

**Aggregated Output:**

```text
27:The empirical CFO information is \(I(Y;S\mid Z)=H(Y\mid Z)-H(Y\mid S,Z)\). The code counts supply both conditional entropies. This is the information available to a Bayes decoder with the sign metadata, independent of which connector is used. Additional diagnostics use the score vector \(Q\) as a covariate. Then \(L(Q)=H(Y)-I(Y;Q)+\mathbb E_Q D_{\rm KL}(P(Y\mid Q)\Vert Q)\). With the fixed feature \(A\) equal to the argmax set of \(Q\), the identity is \(L(Q)=H(Y\mid A)-I(Y;Q\mid A)+\mathbb E_Q D_{\rm KL}(P(Y\mid Q)\Vert Q)\). Argmax resolution reports a coarse, model-dependent predictive association. Exact-float score-vector groups can identify a code or even a pilot row, so their empirical resolution can be high for a poor decoder. Neither diagnostic identifies a phase-slot mechanism.
59:Let \(q_d\) use the locked temperatures, \(q_*\) use the post-hoc fitted temperatures and \(Pq_*\) denote the arithmetic projection. For the empirical Q3 design the reported identity is
67:Take zero minus phase termwise. Empirical \(H(Y\mid S,Z)\) cancels, and the conditional mismatch difference may be signed. This path specifies the order of the two interventions. Temperature and group averaging do not commute. At locked \(T\), the JSON additionally gives the raw marginal KL, the arithmetic group gap, and its complete-orbit counterpart. In the actual design the projection bracket includes a distributional correction and can have either sign. A causal allocation of the gain to phase slots and an orthogonal temperature/projection/prior ANOVA do not follow from the log-loss identity.
79:Equality holds exactly when, almost surely, every label of positive posterior mass has the same probability across every aligned member. If the posterior has full support, this is equivariance almost surely. On an arbitrary finite design the exact formula adds \(L(q)-|G|^{-1}\sum_g L(R_gq)\) to the label-weighted Jensen gap. The complete orbits constructed from each saved mother have an invariant design by construction. Saved rotations are an incomplete orbit, so the restriction matters.
89:Thus both pools weakly improve invariant-design log risk over the raw model. Their ordering depends on the label law and the forecasts. Normalization makes the geometric pool the minimizer of \(\sum_g D_{\rm KL}(r\Vert R_gq)\) over \(r\). It is equivariant, and applying it again returns the same distribution. The arithmetic pool minimizes \(\sum_g D_{\rm KL}(R_gq\Vert r)\) and is also equivariant and idempotent.
101:A claim passes when every endpoint for that claim clears its limit. Freeze the candidate, seed handling, allowed observation sets, per-class weighting, code below, multiplicity family, score floor, calibration, and all margins before unblinding. Apply the same three gate parts separately to the synthetic extrapolation claim, the Pluto claim and the CTTC claim. For a claim covering all four synthetic cells, require each cell to pass. Report each failed or indeterminate endpoint. A winner picked from several candidates requires a design split or inclusion in the multiplicity family.
103:1. **Absolute and comparative performance.** Use paired lower bounds. Require NLL to beat uniform, the dev-frozen prior, and the matched zero-phase system, with a \(0.02\)-nat increment proposed against matched zero. Require accuracy above analytic chance, a dev-frozen prior classifier, and the matched zero-phase system, with \(+0.02\) absolute over chance and zero proposed as initial margins. With imbalanced classes, also require outperformance of the fixed majority prediction or used prior competitor, and report balanced accuracy as a separately locked estimand. Require valid chance/random-label randomization tails for the accuracy and NLL statistics where the design supplies exchangeability. Choose an oracle with the same information budget as the candidate for the main NIG denominator and disclose a finer physical oracle separately. Propose \(\eta_{\min}=0.10\) and validate \(H(Y)-L(o)>0\) on an independent development design or by a co-primary lower bound. Test the linear contrast \(D_{\eta}=(1-\eta)H(Y)-L(q)+\eta L(o)>0\) rather than dividing two noisy quantities. Use \(H(Y)=\log K\) for a genuinely uniform target law. Otherwise predefine the target distribution or carry uncertainty in \(H(Y)\) by a joint confidence region and minimize the contrast over it. For real data without a defensible Bayes oracle, freeze a physically informed predictive reference, call the resulting metric reference-normalized, and withhold the Bayes-normalized claim. Requiring a learner to outperform a true Bayes oracle under proper loss is infeasible. A positive NIG floor defines a fraction of the attainable improvement. Controlled slot/text interventions form a further prerequisite for a specifically signed-slot mechanism claim. Require improvement with matched versus permuted or flipped slots at fixed text and same \(X\), with seeded training and ablations paired. Use the native receiver outputs for the gate.
105:3. **Hierarchy and multiplicity.** Keep all repeats, rotations, acquisitions, and matched arms inside their mother and higher cluster. Report a frozen-seed conditional estimate and, if claiming generalization to retraining, a separate random-seed inference. Use equal independent cluster weights for the following exact balanced analysis. Pluto supplies 16 units if they are independent captures. For a CTTC new-session claim use nine independent session means, with all eight clusters per session contributing to that session. A conditional claim on the fixed sessions needs the corresponding estimand and defensible within-session cluster independence. For a permutation or label-randomization gate, lock the allowed exchangeability orbit and recompute the same statistic throughout. Repeated/rotated labels move jointly with their mother and never split correlated session units as though independent. A synthetic randomized design with independent label assignments can instead rerandomize the sanctioned generator operation inside the original design. When there is no justified exchangeability, collect a randomized negative control. State whether it tests a sharp random-label null or a mean-effect margin. A permutation p-value \((1+\#\{T_{\rm perm}\ge T_{\rm obs}\})/(R+1)\) with uniform randomizations and ties included is conservative under the sharp design. Use exact enumeration when feasible.
109:For an oriented block contrast \(D_{bs}\) let \(D_{bs}=\mu+u_b+\nu_s+e_{bs}\), with independent Gaussian families of variances \(\sigma_u^2,\sigma_\nu^2,\sigma_e^2\). These are the assumptions for the exact parametric calculations. Blocks already aggregate mothers and correlated views. If \(m\) mothers in a block have common variance \(v\) and exchangeable within-block correlation \(\rho\), a block mean contributes \(v[1+(m-1)\rho]/m\); this design effect applies to the particular contrast and variance component being modeled. The Stage 3D Work E lock estimated block SD on a fixed-seed pilot, took the larger SD from pilot upper bound and design-effect scenario and used the conditional noncentral t. It did not price uncertainty across retrains.
137:Use the same formulas with \(B=J_0\), \(\sigma_u^2=\sigma_{\rm session}^2+\sigma_{\rm cluster}^2/m\), and \(\sigma_e^2\) replaced by \(\sigma_e^2/m\); an additional session-by-seed effect is another contribution to the collapsed interaction variance. Use planned balanced session-level estimation or a separately validated unequal-weight design. For a fixed-seed claim with independent bounded cluster means and arbitrary within-cluster dependence, a one-sided Hoeffding bound gives an alternative to Gaussian t inference. For mean of \(B\) equal-weight cluster means in \([l,h]\), reject when the empirical excess exceeds \((h-l)\sqrt{\log(1/\alpha)/(2B)}\). For fixed weights \(a_b\) summing to one, replace \(1/B\) by \(\sum_b a_b^2\). Unknown non-Gaussian random seed effects with only three seeds do not admit a generic exact t power calculation. Correlation modeling and null randomization require separate justifications.
?? research/stage4_math_and_design/

Name          Length
----          ------
.uv-cache           
THEOREM_M2.md 22664 
verify_m2.py  35361 


```

### Event #32: `item.completed` — `command_execution` (`item_28`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$path='"'research/stage4_math_and_design/M2/verify_m2.py'
"'$content=Get-Content -Raw -Encoding UTF8 -LiteralPath $path
$content=$content.Replace('"'    nll = float(-np.mean(np.log(np.take_along_axis(model, labels[..., None], axis=-1))))', \"    label_indices = np.broadcast_to(labels, model.shape[:-1])[..., None]\" + [Environment]::NewLine + '    nll = float(-np.mean(np.log(np.take_along_axis(model, label_indices, axis=-1))))')
"'$content | Set-Content -Encoding UTF8 -LiteralPath $path
$ErrorActionPreference='"'Stop'
"'$m2=Join-Path (Get-Location) '"'research/stage4_math_and_design/M2'
"'$env:UV_CACHE_DIR=Join-Path $m2 '"'.uv-cache'
"'$env:UV_OFFLINE='"'1'
"'$env:UV_LINK_MODE='"'copy'
"'$env:TMP=$env:UV_CACHE_DIR
$env:TEMP=$env:UV_CACHE_DIR
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M2/verify_m2.py'
```

**Aggregated Output:**

```text
Traceback (most recent call last):
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M2\verify_m2.py", line 661, in <module>
    run()
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M2\verify_m2.py", line 640, in run
    q2_result = q2_analysis(q2, t2)
                ^^^^^^^^^^^^^^^^^^^
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M2\verify_m2.py", line 350, in q2_analysis
    "NIG_vs_Work_A_realized": (h_y - dev_metrics["nll"]) / (h_y - entry["reference_metrics"]["nll"])}
                              ~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ZeroDivisionError: float division by zero
```

### Event #33: `item.completed` — `command_execution` (`item_30`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n -A 55 'def calibrate_signed_reference_table' research/stage3d_qwen_signed_phase/analysis_stage3d.py; Get-ChildItem research/stage4_math_and_design/M2/.uv-cache -Force | Select-Object Name"
```

**Aggregated Output:**

```text
52:def calibrate_signed_reference_table(
53-    encoder,
54-    cell_idx: int,
55-    abs_omega: float,
56-    h: int,
57-    ell: int,
58-    t_p: float,
59-    snr_db: float = 2.0,
60-    sigma_phi: float = 0.10,
61-    blocks_per_batch: int = 128,
62-    n_batches: int = 8,
63-) -> Tuple[np.ndarray, np.ndarray]:
64-    """Calibrate quarter-turn pooled 256-code reference table on independent namespace 960501 (65,536 mothers/sign)."""
65-    counts = np.zeros((2, 4, 4, 4), dtype=np.int64)
66-    mass = np.zeros((2, 256), dtype=np.float64)
67-    omegas = (-float(abs_omega), float(abs_omega))
68-    for chunk_i in range(n_batches):
69-        b_map, _ = generate_paired_cfo_blocks(
70-            namespace=NS_REF_CALIB,
71-            cell_idx=cell_idx * 100 + chunk_i,
72-            blocks=blocks_per_batch,
73-            h=h,
74-            ell=ell,
75-            t_p=t_p,
76-            omegas=omegas,
77-            snr_db=snr_db,
78-            sigma_phi=sigma_phi,
79-            split_prefix="ref_cal",
80-        )
81-        for s_i, w in enumerate(omegas):
82-            batch = b_map[w]
83-            z1 = np.asarray(encoder.encode(batch.y1), dtype=np.int64)
84-            z2 = np.asarray(encoder.encode(batch.y2), dtype=np.int64)
85-            codes = 16 * z1 + z2
86-            targets = batch.k3_repeat
87-            np.add.at(counts[s_i], (z1 % 4, z2 % 4, (targets - z1 // 4) % 4), 1)
88-            mass[s_i] += np.bincount(codes, minlength=256)
89-    full = np.zeros((2, 256, 4), dtype=np.float64)
90-    for k in range(256):
91-        z1, z2 = divmod(k, 16)
92-        full[:, k, :] = np.roll(counts[:, z1 % 4, z2 % 4] + 0.5, z1 // 4, axis=-1)
93-    ref_probs = full / full.sum(axis=-1, keepdims=True)
94-    mass_probs = mass / mass.sum(axis=-1, keepdims=True)
95-    return ref_probs, mass_probs
96-
97-
98-def compute_blind_cell_index_and_analytic(
99-    encoder,
100-    cell_id: str,
101-    namespace: int,
102-    blocks: int,
103-    abs_omega: float,
104-    h: int,
105-    ell: int,
106-    t_p: float,
107-    cal_cell_idx: int,

Name           
----           
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


```

### Event #34: `item.completed` — `command_execution` (`item_31`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$path='"'research/stage4_math_and_design/M2/verify_m2.py'
"'$content=Get-Content -Raw -Encoding UTF8 -LiteralPath $path
$old=@'"'
    materialized = stage3.softmax_temperature(raw, 1 / beta) if beta else np.full_like(raw, 0.25)
'@
"'$new=@'"'
    materialized = stage3.softmax_temperature(raw, 1 / beta) if beta else np.full_like(raw, 0.25)
    dev = stage3.softmax_temperature(raw, 1 / dev_beta)
    bregman = float(np.sum(mass[..., 0] * kl(materialized, dev)) / total)
    boundary = (dev_beta - beta) * derivative(beta)
    close(cross_entropy(counts, dev) - cross_entropy(counts, materialized), bregman + boundary,
          \"scalar temperature Bregman identity\")
'@
"'$content=$content.Replace($old,$new)
$content=$content.Replace('"'\"derivative_at_star\": derivative(beta),','\"derivative_at_star\": derivative(beta), \"temperature_bregman_KL\": bregman,'+[Environment]::NewLine+'        \"temperature_boundary_term\": boundary,')
"'$content=$content.Replace('"'    common.pop(\"derivative_at_dev\")','    for field in (\"derivative_at_dev\", \"temperature_bregman_KL\", \"temperature_boundary_term\"):'+[Environment]::NewLine+'        common.pop(field)')
"'$old=@'"'
    result = {\"raw_shape\": list(raw.shape), \"codes_shape\": list(codes.shape), \"blocks\": y.shape[-1] // 64,
'@
"'$new=@'"'
    cal_ref, cal_mass = get(\"cal_ref\"), get(\"cal_mass\")
    cal_joint = cal_ref * cal_mass[..., None] / 2
    cal_unsigned = divide_counts(cal_joint.sum(axis=0))
    cal_signed_h = float(np.sum(cal_mass * entropy(cal_ref)) / 2)
    cal_unsigned_h = float(np.sum(cal_mass.sum(axis=0) * entropy(cal_unsigned)) / 2)
    close(unsigned - signed, np.sum(count.sum(axis=-1) * kl(empirical, divide_counts(count.sum(axis=0)))) / count.sum(),
          \"signed conditional MI\")
    result = {\"raw_shape\": list(raw.shape), \"codes_shape\": list(codes.shape), \"blocks\": y.shape[-1] // 64,
              \"independent_cal_reference\": {\"H_Y_given_signed_code\": cal_signed_h,
                 \"H_Y_given_unsigned_code\": cal_unsigned_h, \"signed_conditional_MI\": cal_unsigned_h - cal_signed_h,
                 \"blind_metrics\": metrics(q3_index(cal_ref[None, ...], codes), y)[0]},
'@
"'$content=$content.Replace($old,$new)
$content=$content.Replace('"'        arm[\"NIG_realized\"] =','        close(arm[\"group_on_complete_orbit\"][\"design_imbalance_nll\"], 0, \"complete orbit invariance\")'+[Environment]::NewLine+'        arm[\"NIG_realized\"] =')
"'$content=$content.Replace('"'        entry = {\"reference_key\": reference_key, \"reference_metrics\": metrics(ref, y)[0], \"arms\": {}}','        entry = {\"reference_key\": reference_key, \"reference_metrics\": metrics(ref, y)[0], \"arms\": {}}'+[Environment]::NewLine+'        denom = h_y - entry[\"reference_metrics\"][\"nll\"]'+[Environment]::NewLine+'        entry[\"NIG_reference_denominator\"] = denom')
"'$content=$content.Replace('"'(h_y - dev_metrics[\"nll\"]) / (h_y - entry[\"reference_metrics\"][\"nll\"])}','(h_y - dev_metrics[\"nll\"]) / denom if denom > 0 else None}')
"'$content | Set-Content -Encoding UTF8 -LiteralPath $path
$ErrorActionPreference='"'Stop'
"'$m2=Join-Path (Get-Location) '"'research/stage4_math_and_design/M2'
"'$env:UV_CACHE_DIR=Join-Path $m2 '"'.uv-cache'
"'$env:UV_OFFLINE='"'1'
"'$env:UV_LINK_MODE='"'copy'
"'$env:TMP=$env:UV_CACHE_DIR
$env:TEMP=$env:UV_CACHE_DIR
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M2/verify_m2.py'
```

**Aggregated Output:**

```text
C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M2\verify_m2.py:237: RuntimeWarning: invalid value encountered in multiply
  close(unsigned - signed, np.sum(count.sum(axis=-1) * kl(empirical, divide_counts(count.sum(axis=0)))) / count.sum(),
Traceback (most recent call last):
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M2\verify_m2.py", line 681, in <module>
    run()
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M2\verify_m2.py", line 632, in run
    q3_result[name], score_matrices[name] = q3_analysis(q3, name, t3)
                                            ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M2\verify_m2.py", line 237, in q3_analysis
    close(unsigned - signed, np.sum(count.sum(axis=-1) * kl(empirical, divide_counts(count.sum(axis=0)))) / count.sum(),
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M2\verify_m2.py", line 29, in close
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=0, err_msg=label)
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M2\.uv-cache\archive-v0\JiHBfPRTl1ebuKou\Lib\site-packages\numpy\testing\_private\utils.py", line 1711, in assert_allclose
    assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M2\.uv-cache\archive-v0\JiHBfPRTl1ebuKou\Lib\site-packages\numpy\testing\_private\utils.py", line 811, in assert_array_compare
    flagged = func_assert_same_pos(x, y, func=isnan, hasval='nan')
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M2\.uv-cache\archive-v0\JiHBfPRTl1ebuKou\Lib\site-packages\numpy\testing\_private\utils.py", line 780, in func_assert_same_pos
    raise AssertionError(msg)
AssertionError: 
Not equal to tolerance rtol=0, atol=2e-11
signed conditional MI
nan location mismatch:
 ACTUAL: array(0.292945)
 DESIRED: array(nan)
```

### Event #35: `item.completed` — `command_execution` (`item_32`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n 'temperature_bregman|complete orbit|signed conditional MI|denom|NIG_vs' research/stage4_math_and_design/M2/verify_m2.py; Get-ChildItem research/stage4_math_and_design/M2 -Force | Select-Object Name,Length"
```

**Aggregated Output:**

```text
104:        "derivative_at_star": derivative(beta), "temperature_bregman_KL": bregman,
112:    for field in ("derivative_at_dev", "temperature_bregman_KL", "temperature_boundary_term"):
238:          "signed conditional MI")
291:        close(arm["group_on_complete_orbit"]["design_imbalance_nll"], 0, "complete orbit invariance")
335:        denom = h_y - entry["reference_metrics"]["nll"]
336:        entry["NIG_reference_denominator"] = denom
370:                                   "NIG_vs_Work_A_realized": (h_y - dev_metrics["nll"]) / denom if denom > 0 else None}

Name          Length
----          ------
.uv-cache           
THEOREM_M2.md 22664 
verify_m2.py  36813 


```

### Event #36: `item.completed` — `command_execution` (`item_33`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$path='"'research/stage4_math_and_design/M2/verify_m2.py'
"'$content=Get-Content -Raw -Encoding UTF8 -LiteralPath $path
$content=$content.Replace('"'    close(unsigned - signed, np.sum(count.sum(axis=-1) * kl(empirical, divide_counts(count.sum(axis=0)))) / count.sum(),', 
    '    code_mass = count.sum(axis=-1)' + [Environment]::NewLine +
    '    active = code_mass > 0' + [Environment]::NewLine +
    '    unsigned_table = np.broadcast_to(divide_counts(count.sum(axis=0)), empirical.shape)' + [Environment]::NewLine +
    '    close(unsigned - signed, np.average(kl(empirical[active], unsigned_table[active]), weights=code_mass[active]),')
"'$content | Set-Content -Encoding UTF8 -LiteralPath $path
$ErrorActionPreference='"'Stop'
"'$m2=Join-Path (Get-Location) '"'research/stage4_math_and_design/M2'
"'$env:UV_CACHE_DIR=Join-Path $m2 '"'.uv-cache'
"'$env:UV_OFFLINE='"'1'
"'$env:UV_LINK_MODE='"'copy'
"'$env:TMP=$env:UV_CACHE_DIR
$env:TEMP=$env:UV_CACHE_DIR
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M2/verify_m2.py'
```

**Aggregated Output:**

```text
results_m2.json written. Maximum in-scope summary difference: 0.
```

### Event #37: `item.completed` — `command_execution` (`item_34`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$j=Get-Content -Raw research/stage4_math_and_design/M2/results_m2.json | ConvertFrom-Json
foreach($c in $j.Q2.controls.psobject.Properties){
 $x=$c.Value; '"'{0} {1:F8}: {2}' -f "'$c.Name,$x.reference_metrics.nll,($x.delta_nll_zero_minus_phase | ConvertTo-Json -Compress)
 foreach($a in @('"'phase','zero')){ "'$v=$x.arms.$a; '"'  {0} raw {1:F8}/{2:F8}/{3:F8}; opt {4:F8}/{5:F8}/{6:F8}; temps {7}' -f "'$a,$v.dev.acc,$v.dev.nll,$v.dev.ece,$v.opt.acc,$v.opt.nll,$v.opt.ece,($v.T_star_by_seed.T_star -join '"', ') }
}
'Q2 H(Y) {0:F8}, H(Y|Z) {1:F8}, codes {2}' -f "'$j.Q2.H_Y,$j.Q2.H_Y_given_base_codes,$j.Q2.base_code_coverage
$p=$j.power
$p.variance_components | ConvertTo-Json -Depth 5 -Compress
$p.synthetic_cell_plans | ConvertTo-Json -Depth 6 -Compress
$p.G1_more_seeds | ConvertTo-Json -Depth 6 -Compress
foreach($c in $p.scenario_proxies.psobject.Properties){
  foreach($ep in @('"'accuracy','nll')){
   "'$v=$c.Value.$ep
   '"'{0} {1} {2} MC {3}' -f "'$c.Name,$ep,($v.scenarios | ConvertTo-Json -Depth 6 -Compress),($v.MC_at_planning_excess | ConvertTo-Json -Depth 5 -Compress)
  }
}'
```

**Aggregated Output:**

```text
none 1.38629436: {"reference_entropy":0.0,"temperature_gap":0.0009531064367116482,"reference_marginal_prior_kl":-1.6454032059631807E-06,"reference_reverse_conditional_kl":-2.1108502757560415E-05,"realized_label_residual":4.5482717117018012E-05,"total":0.0009758352478652643,"at_seedwise_T_star":2.2728811153616135E-05}
  phase raw 0.24804688/1.39284648/0.02295694; opt 0.24674479/1.38627163/0.00406258; temps 3.9648025247968675, infinity, infinity
  zero raw 0.25781250/1.39382232/0.01886379; opt 0.25000000/1.38629436/0.00000000; temps infinity, infinity, infinity
valid 0.81920681: {"reference_entropy":0.0,"temperature_gap":0.020213678961079173,"reference_marginal_prior_kl":0.0002882730363920061,"reference_reverse_conditional_kl":0.001585103176843905,"realized_label_residual":-0.0002797062844093201,"total":0.021807348889905986,"at_seedwise_T_star":0.001593669928826813}
  phase raw 0.28776042/1.38888294/0.03035772; opt 0.28776042/1.38328163/0.02539988; temps 0.8652085356966109, 2.908128403543404, 2.2773727970468105
  zero raw 0.26627604/1.41069029/0.05571145; opt 0.27408854/1.38487530/0.01842628; temps infinity, 8.449271157796604, 2.4502712378555436
duplicate_copy_dedup 0.81920681: {"reference_entropy":0.0,"temperature_gap":0.020213678961079173,"reference_marginal_prior_kl":0.0002882730363920061,"reference_reverse_conditional_kl":0.001585103176843905,"realized_label_residual":-0.0002797062844093201,"total":0.021807348889905986,"at_seedwise_T_star":0.001593669928826813}
  phase raw 0.28776042/1.38888294/0.03035772; opt 0.28776042/1.38328163/0.02539988; temps 0.8652085356966109, 2.908128403543404, 2.2773727970468105
  zero raw 0.26627604/1.41069029/0.05571145; opt 0.27408854/1.38487530/0.01842628; temps infinity, 8.449271157796604, 2.4502712378555436
duplicate_copy_raw 0.81920681: {"reference_entropy":0.0,"temperature_gap":0.017306880657969703,"reference_marginal_prior_kl":4.2747689928401833E-05,"reference_reverse_conditional_kl":0.002795104300267859,"realized_label_residual":-0.0012231411027370243,"total":0.018921591545429273,"at_seedwise_T_star":0.0016147108874595695}
  phase raw 0.26302083/1.39441987/0.02842789; opt 0.27083333/1.38419325/0.00937794; temps infinity, 3.7025338829362013, 3.3847531039348597
  zero raw 0.25781250/1.41334146/0.04628643; opt 0.27213542/1.38580796/0.01587311; temps infinity, 15.880552368353166, 2.6437794207442153
independent_only 1.38629436: {"reference_entropy":0.0,"temperature_gap":0.013030938187352792,"reference_marginal_prior_kl":8.0079987056502766E-06,"reference_reverse_conditional_kl":0.00021766861526371914,"realized_label_residual":-0.0004491704425695353,"total":0.012807444358752917,"at_seedwise_T_star":-0.00022349382859987443}
  phase raw 0.24804688/1.40771417/0.03642444; opt 0.25000000/1.38629436/0.00000000; temps infinity, infinity, infinity
  zero raw 0.24934896/1.42052161/0.04990034; opt 0.25716146/1.38607087/0.00424496; temps infinity, infinity, 3.214102650868034
valid_plus_independent 0.81920681: {"reference_entropy":0.0,"temperature_gap":0.010747711226556156,"reference_marginal_prior_kl":5.329848971552531E-05,"reference_reverse_conditional_kl":-0.0004219690212141636,"realized_label_residual":-0.00032958887638634415,"total":0.01004945181867134,"at_seedwise_T_star":-0.0006982594078848159}
  phase raw 0.24544271/1.40823166/0.04250820; opt 0.25195313/1.38624374/0.00042509; temps infinity, infinity, 9.29829250109447
  zero raw 0.27278646/1.41828111/0.04801281; opt 0.27278646/1.38554548/0.01400351; temps 7.095044906203757, 18.073740775368087, 2.124163092466728
fresh_resample 0.66318900: {"reference_entropy":0.0,"temperature_gap":0.003429119570539374,"reference_marginal_prior_kl":5.3607920829146227E-05,"reference_reverse_conditional_kl":-0.0016212515684325135,"realized_label_residual":-0.0011048463566145639,"total":0.0007566295663214984,"at_seedwise_T_star":-0.0026724900042178756}
  phase raw 0.26432292/1.40563989/0.03483457; opt 0.26432292/1.38531773/0.01012985; temps 3.8321967773369954, 7.929316465149832, 3.7837360764913304
  zero raw 0.25781250/1.40639652/0.05921296; opt 0.25781250/1.38264524/0.01107018; temps 8.109661499774987, 5.098038302891774, 7.537730645527966
Q2 H(Y) 1.38629436, H(Y|Z) 0.80975820, codes 183
{"G1_main_extrap_both":{"accuracy":{"B":131,"S":3,"mean":0.04392626166242578,"block_conditional_sd":0.016436968634732152,"across_seed_means_sd":0.01938021265344532,"seed_means":[0.06615776081424937,0.030593988549618322,0.03502703562340966],"mean_squares":[0.0008105218136975055,0.049202636166551836,0.000582952311272251],"untruncated_components":[7.5856500808418146E-05,0.0003711426248494625,0.000582952311272251],"var_block":7.5856500808418146E-05,"var_seed":0.0003711426248494625,"var_interaction":0.000582952311272251,"seed_sd_95_upper":0.08557132200882683},"nll":{"B":131,"S":3,"mean":0.035650002550707036,"block_conditional_sd":0.015555378401255893,"across_seed_means_sd":0.02167594178554355,"seed_means":[0.04864490167869194,0.047678100206173316,0.010627005767255822],"mean_squares":[0.0007259093916187751,0.06154988525002576,0.00036183827982413995],"untruncated_components":[0.0001213570372648784,0.0004670843280168063,0.00036183827982413995],"var_block":0.0001213570372648784,"var_seed":0.0004670843280168063,"var_interaction":0.00036183827982413995,"seed_sd_95_upper":0.09570787625210025}},"G2_interp_both":{"accuracy":{"B":8,"S":3,"mean":-0.02484809027777778,"block_conditional_sd":0.022759130674656312,"across_seed_means_sd":0.03237309733696349,"seed_means":[-0.060546875,-0.016601562500000007,0.0026041666666666574],"mean_squares":[0.0015539340871982457,0.0083841394495081,0.0007970497090980489],"untruncated_components":[0.0002522947927000656,0.0009483862175512564,0.0007970497090980489],"var_block":0.0002522947927000656,"var_seed":0.0009483862175512564,"var_interaction":0.0007970497090980489,"seed_sd_95_upper":0.1429400588208676},"nll":{"B":8,"S":3,"mean":-0.02775961336159183,"block_conditional_sd":0.021671277716950415,"across_seed_means_sd":0.014093228446933448,"seed_means":[-0.023132501930805738,-0.043584700001079696,-0.01656163815289005],"mean_squares":[0.001408932833655575,0.0015889527044596332,0.00024301373608068432],"untruncated_components":[0.00038863969919163024,0.00016824237104736862,0.00024301373608068432],"var_block":0.00038863969919163024,"var_seed":0.00016824237104736862,"var_interaction":0.00024301373608068432,"seed_sd_95_upper":0.062227190750773716}},"G3_extrap_angle":{"accuracy":{"B":8,"S":3,"mean":0.029405381944444444,"block_conditional_sd":0.014446858075158492,"across_seed_means_sd":0.0074964086151978635,"seed_means":[0.037109375,0.028971354166666668,0.022135416666666664],"mean_squares":[0.0006261351247313165,0.000449569137008102,0.000912661274904927],"untruncated_components":[-9.5508716724536824E-05,-5.7886517237103125E-05,0.000912661274904927],"var_block":0.0,"var_seed":0.0,"var_interaction":0.000912661274904927,"seed_sd_95_upper":0.03309961593258375},"nll":{"B":8,"S":3,"mean":0.03708293509316355,"block_conditional_sd":0.016873732796666648,"across_seed_means_sd":0.021243946460525992,"seed_means":[0.04368909001842264,0.05423894605219398,0.01332076920887404],"mean_squares":[0.0008541685754799111,0.003610442089741558,0.000376570368863823],"untruncated_components":[0.0001591994022053627,0.0004042339651097169,0.000376570368863823],"var_block":0.0001591994022053627,"var_seed":0.0004042339651097169,"var_interaction":0.000376570368863823,"seed_sd_95_upper":0.09380044563075389}},"G4_extrap_freq":{"accuracy":{"B":8,"S":3,"mean":-0.016710069444444444,"block_conditional_sd":0.020744011146093047,"across_seed_means_sd":0.030509366661461167,"seed_means":[-0.04329427083333333,-0.0234375,0.0166015625],"mean_squares":[0.0012909419952876976,0.007446571632667824,0.0014576987614707342],"untruncated_components":[-5.5585588727678869E-05,0.0007486091088996363,0.0014576987614707342],"var_block":0.0,"var_seed":0.0007486091088996363,"var_interaction":0.0014576987614707342,"seed_sd_95_upper":0.13471094902610045},"nll":{"B":8,"S":3,"mean":-0.024916547637425363,"block_conditional_sd":0.0106876153163971,"across_seed_means_sd":0.027126434693930508,"seed_means":[-0.010001754277664393,-0.05622777343300128,-0.008520115201610418],"mean_squares":[0.0003426753634538577,0.0058867476736326136,0.0003605420806454988],"untruncated_components":[-5.9555723972136996E-06,0.0006907756991233893,0.0003605420806454988],"var_block":0.0,"var_seed":0.0006907756991233893,"var_interaction":0.0003605420806454988,"seed_sd_95_upper":0.11977396325076338}},"H_c4_035":{"accuracy":{"B":16,"S":3,"mean":0.025933159722222227,"block_conditional_sd":0.01915205459432223,"across_seed_means_sd":0.013352638150539624,"seed_means":[0.041341145833333336,0.017740885416666668,0.01871744791666667],"mean_squares":[0.0011004035855516976,0.0028526871292679397,0.0006023736647617672],"untruncated_components":[0.00016600997359664347,0.00014064459153163578,0.0006023736647617672],"var_block":0.00016600997359664347,"var_seed":0.00014064459153163578,"var_interaction":0.0006023736647617672,"seed_sd_95_upper":0.05895719099057696},"nll":{"B":16,"S":3,"mean":0.018089816085664654,"block_conditional_sd":0.017279082800419777,"across_seed_means_sd":0.015610432702343682,"seed_means":[0.030086664630508575,0.023742242764931087,0.0004405408615543033],"mean_squares":[0.0008957001072712877,0.003898969746470417,0.00033318238121244416],"untruncated_components":[0.00018750590868628117,0.00022286171032862332,0.00033318238121244416],"var_block":0.00018750590868628117,"var_seed":0.00022286171032862332,"var_interaction":0.00033318238121244416,"seed_sd_95_upper":0.0689262490229641}}}
{"G1_main_extrap_both":{"accuracy":{"effect_above_margin":0.03,"fixed_min_B":7,"random_min_B":{"B":null,"known_variance_power_ceiling":0.715370889001055,"large_B_power":0.23839454201753918,"status":"unattainable_even_with_known_variance"},"fixed_power_at_131":1.0,"random_power_at_131":0.22385039124135897},"nll":{"effect_above_margin":0.04,"fixed_min_B":5,"random_min_B":{"B":null,"known_variance_power_ceiling":0.8594062281784283,"large_B_power":0.3097754540510828,"status":"unattainable_even_with_known_variance"},"fixed_power_at_131":1.0,"random_power_at_131":0.3002791100894329}},"G2_interp_both":{"accuracy":{"effect_above_margin":0.03,"fixed_min_B":11,"random_min_B":{"B":null,"known_variance_power_ceiling":0.3296944449514519,"large_B_power":0.11897970604193757,"status":"unattainable_even_with_known_variance"},"fixed_power_at_131":1.0,"random_power_at_131":0.10884167062269667},"nll":{"effect_above_margin":0.04,"fixed_min_B":7,"random_min_B":{"B":null,"status":"no_design_in_search_range","max_b_requested":4096,"searched_through":2,"known_variance_power_ceiling":0.9993439682884748,"large_B_power":0.6205605885594975,"crossed_power_ceiling_for_B_ge_S_numerical":0.7566271436334422},"fixed_power_at_131":1.0,"random_power_at_131":0.591046605657425}},"G3_extrap_angle":{"accuracy":{"effect_above_margin":0.03,"fixed_min_B":8,"random_min_B":{"B":42,"status":"attained","known_variance_power_ceiling":1.0,"large_B_power":1.0,"searched_through":42},"fixed_power_at_131":1.0,"random_power_at_131":0.999996598315146},"nll":{"effect_above_margin":0.04,"fixed_min_B":6,"random_min_B":{"B":null,"known_variance_power_ceiling":0.9062259206730405,"large_B_power":0.34499554319708337,"status":"unattainable_even_with_known_variance"},"fixed_power_at_131":1.0,"random_power_at_131":0.3326588492782109}},"G4_extrap_freq":{"accuracy":{"effect_above_margin":0.03,"fixed_min_B":11,"random_min_B":{"B":null,"known_variance_power_ceiling":0.4094667782777398,"large_B_power":0.14084962861941103,"status":"unattainable_even_with_known_variance"},"fixed_power_at_131":1.0,"random_power_at_131":0.12776584970207056},"nll":{"effect_above_margin":0.04,"fixed_min_B":4,"random_min_B":{"B":null,"known_variance_power_ceiling":0.6942716996430474,"large_B_power":0.23020489760087653,"status":"unattainable_even_with_known_variance"},"fixed_power_at_131":1.0,"random_power_at_131":0.22647731742833932}},"H_c4_035":{"accuracy":{"effect_above_margin":0.03,"fixed_min_B":9,"random_min_B":{"B":null,"status":"no_design_in_search_range","max_b_requested":4096,"searched_through":2,"known_variance_power_ceiling":0.9878839760168768,"large_B_power":0.48477273609173194,"crossed_power_ceiling_for_B_ge_S_numerical":0.682195907666907},"fixed_power_at_131":1.0,"random_power_at_131":0.44302443011878473},"nll":{"effect_above_margin":0.04,"fixed_min_B":6,"random_min_B":{"B":null,"status":"no_design_in_search_range","max_b_requested":4096,"searched_through":2,"known_variance_power_ceiling":0.9940121917817448,"large_B_power":0.5228240456656998,"crossed_power_ceiling_for_B_ge_S_numerical":0.7024770394427118},"fixed_power_at_131":1.0,"random_power_at_131":0.5040697276443087}}}
{"3":{"accuracy":{"B131":0.22385039124135897,"B16":0.1364997696658935},"nll":{"B131":0.3002791100894329,"B16":0.23444512347289748}},"5":{"accuracy":{"B131":0.6244101407363274,"B16":0.529102320955811},"nll":{"B131":0.7760074903542191,"B16":0.721010063649363}},"6":{"accuracy":{"B131":0.7716656243788191,"B16":0.6846108350552778},"nll":{"B131":0.8982237158613342,"B16":0.8578601882700307}},"8":{"accuracy":{"B131":0.9284444670725485,"B16":0.873456643340691},"nll":{"B131":0.9835829950839948,"B16":0.9684677837177171}},"10":{"accuracy":{"B131":0.9806104397655852,"B16":0.9532072018148796},"nll":{"B131":0.9978402364699044,"B16":0.9935950367068294}},"12":{"accuracy":{"B131":0.9952242346584765,"B16":0.9833172822814217},"nll":{"B131":0.9997492984596659,"B16":0.9987189184647882}}}
B16_independent accuracy {"planning":{"excess":0.03,"fixed_min_groups":7,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.715370889001055,"large_B_power":0.23839454201753918,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":0.9999974815064975,"random_power_at_available":0.1364997696658935},"0p02_excess":{"excess":0.02,"fixed_min_groups":13,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.3707306275129978,"large_B_power":0.13018457417462473,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":0.9906374139683718,"random_power_at_available":0.053431326309220414},"G1_observed_excess":{"excess":0.02392626166242578,"fixed_min_groups":10,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.5092055120294001,"large_B_power":0.16924723379151027,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":0.9993625933257768,"random_power_at_available":0.08045859629456789}} MC {"fixed":{"type_I_boundary":{"analytic":0.01666666666666665,"MC":0.01675,"MC_SE":0.000286259400621961},"power":{"analytic":0.9999974815064975,"MC":0.999995,"MC_SE":3.5485822237296479E-06}},"crossed":{"type_I_boundary":{"analytic":0.0023996590164296634,"MC":0.002185,"MC_SE":0.00010940522503597648},"power":{"analytic":0.1364997696658935,"MC":0.13626,"MC_SE":0.0007676834717090486}},"naive_cluster_t_with_random_seeds":{"type_I_boundary":{"analytic":0.21475999546543334,"MC":0.21498,"MC_SE":0.0009182541582076294},"power":{"analytic":0.9566365212832489,"MC":0.95666,"MC_SE":0.00045542884971383257}}}
B16_independent nll {"planning":{"excess":0.04,"fixed_min_groups":5,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.8594062281784283,"large_B_power":0.3097754540510828,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":0.9999999999997996,"random_power_at_available":0.23444512347289748},"0p02_excess":{"excess":0.02,"fixed_min_groups":12,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.2997238588856113,"large_B_power":0.11080417216654499,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":0.9953737547819306,"random_power_at_available":0.0552173053552394},"G1_observed_excess":{"excess":0.015650002550707035,"fixed_min_groups":17,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.1911096047056438,"large_B_power":0.08048038676617729,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":0.9430342585982612,"random_power_at_available":0.03480276338528375}} MC {"fixed":{"type_I_boundary":{"analytic":0.01666666666666665,"MC":0.01709,"MC_SE":0.000286259400621961},"power":{"analytic":0.9999999999997996,"MC":1.0,"MC_SE":1.0009876521336329E-09}},"crossed":{"type_I_boundary":{"analytic":0.003747897196843079,"MC":0.00379,"MC_SE":0.00013663547239726906},"power":{"analytic":0.23444512347289748,"MC":0.23298,"MC_SE":0.0009473135899813624}},"naive_cluster_t_with_random_seeds":{"type_I_boundary":{"analytic":0.2481931649561898,"MC":0.247005,"MC_SE":0.0009659019562699399},"power":{"analytic":0.9907617388194597,"MC":0.99052,"MC_SE":0.00021392657281063047}}}
B72_independent accuracy {"planning":{"excess":0.03,"fixed_min_groups":7,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.715370889001055,"large_B_power":0.23839454201753918,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":1.0,"random_power_at_available":0.2122409469156132},"0p02_excess":{"excess":0.02,"fixed_min_groups":13,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.3707306275129978,"large_B_power":0.13018457417462473,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":0.9999999999999994,"random_power_at_available":0.10600978671928464},"G1_observed_excess":{"excess":0.02392626166242578,"fixed_min_groups":10,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.5092055120294001,"large_B_power":0.16924723379151027,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":1.0,"random_power_at_available":0.1437241439718856}} MC {"fixed":{"type_I_boundary":{"analytic":0.016666666666666677,"MC":0.016775,"MC_SE":0.0002862594006219612},"power":{"analytic":1.0,"MC":1.0,"MC_SE":0.0}},"crossed":{"type_I_boundary":{"analytic":0.00904592465362831,"MC":0.009035,"MC_SE":0.00021170847763834577},"power":{"analytic":0.2122409469156132,"MC":0.214005,"MC_SE":0.0009143159392900712}},"naive_cluster_t_with_random_seeds":{"type_I_boundary":{"analytic":0.3553647438411226,"MC":0.35619,"MC_SE":0.001070235120606346},"power":{"analytic":0.9888442305543398,"MC":0.9887,"MC_SE":0.00023485440440553258}}}
B72_independent nll {"planning":{"excess":0.04,"fixed_min_groups":5,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.8594062281784283,"large_B_power":0.3097754540510828,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":1.0,"random_power_at_available":0.2925208721145423},"0p02_excess":{"excess":0.02,"fixed_min_groups":12,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.2997238588856113,"large_B_power":0.11080417216654499,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":1.0,"random_power_at_available":0.09421668291547937},"G1_observed_excess":{"excess":0.015650002550707035,"fixed_min_groups":17,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.1911096047056438,"large_B_power":0.08048038676617729,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":0.9999999998175817,"random_power_at_available":0.06552765200602047}} MC {"fixed":{"type_I_boundary":{"analytic":0.016666666666666677,"MC":0.017095,"MC_SE":0.0002862594006219612},"power":{"analytic":1.0,"MC":1.0,"MC_SE":0.0}},"crossed":{"type_I_boundary":{"analytic":0.010635198001237904,"MC":0.01047,"MC_SE":0.00022936968592985832},"power":{"analytic":0.2925208721145423,"MC":0.29176,"MC_SE":0.0010172325483680954}},"naive_cluster_t_with_random_seeds":{"type_I_boundary":{"analytic":0.3766665279259077,"MC":0.376505,"MC_SE":0.0010834870896004002},"power":{"analytic":0.9978569278211881,"MC":0.997995,"MC_SE":0.00010340404780395048}}}
B131_independent accuracy {"planning":{"excess":0.03,"fixed_min_groups":7,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.715370889001055,"large_B_power":0.23839454201753918,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":1.0,"random_power_at_available":0.22385039124135897},"0p02_excess":{"excess":0.02,"fixed_min_groups":13,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.3707306275129978,"large_B_power":0.13018457417462473,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":1.0,"random_power_at_available":0.11629905668226127},"G1_observed_excess":{"excess":0.02392626166242578,"fixed_min_groups":10,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.5092055120294001,"large_B_power":0.16924723379151027,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":1.0,"random_power_at_available":0.1548253323378569}} MC {"fixed":{"type_I_boundary":{"analytic":0.01666666666666666,"MC":0.01654,"MC_SE":0.0002862594006219611},"power":{"analytic":1.0,"MC":1.0,"MC_SE":0.0}},"crossed":{"type_I_boundary":{"analytic":0.011607705502201918,"MC":0.01168,"MC_SE":0.00023950956844326772},"power":{"analytic":0.22385039124135897,"MC":0.22447,"MC_SE":0.0009320445096197109}},"naive_cluster_t_with_random_seeds":{"type_I_boundary":{"analytic":0.39171250889230874,"MC":0.391445,"MC_SE":0.0010914985553577285},"power":{"analytic":0.9917965658665415,"MC":0.99193,"MC_SE":0.00020169454382650616}}}
B131_independent nll {"planning":{"excess":0.04,"fixed_min_groups":5,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.8594062281784283,"large_B_power":0.3097754540510828,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":1.0,"random_power_at_available":0.3002791100894329},"0p02_excess":{"excess":0.02,"fixed_min_groups":12,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.2997238588856113,"large_B_power":0.11080417216654499,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":1.0,"random_power_at_available":0.10131266151964669},"G1_observed_excess":{"excess":0.015650002550707035,"fixed_min_groups":17,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.1911096047056438,"large_B_power":0.08048038676617729,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":1.0,"random_power_at_available":0.07175721649177808}} MC {"fixed":{"type_I_boundary":{"analytic":0.01666666666666666,"MC":0.01716,"MC_SE":0.0002862594006219611},"power":{"analytic":1.0,"MC":1.0,"MC_SE":0.0}},"crossed":{"type_I_boundary":{"analytic":0.01279923115902886,"MC":0.01277,"MC_SE":0.00025135046091828245},"power":{"analytic":0.3002791100894329,"MC":0.30242,"MC_SE":0.0010249672339478252}},"naive_cluster_t_with_random_seeds":{"type_I_boundary":{"analytic":0.40810292408080384,"MC":0.409505,"MC_SE":0.0010989880059343266},"power":{"analytic":0.9984320525364219,"MC":0.99849,"MC_SE":8.8472849064828729E-05}}}
CTTC_9_sessions_x8_rho_0.00 accuracy {"planning":{"excess":0.03,"fixed_min_groups":4,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.715370889001055,"large_B_power":0.23839454201753918,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":1.0,"random_power_at_available":0.2123073818445813},"0p02_excess":{"excess":0.02,"fixed_min_groups":4,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.3707306275129978,"large_B_power":0.13018457417462473,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":0.9999999999159409,"random_power_at_available":0.10635501066188723},"G1_observed_excess":{"excess":0.02392626166242578,"fixed_min_groups":4,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.5092055120294001,"large_B_power":0.16924723379151027,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":0.9999999999999993,"random_power_at_available":0.1439471774381103}} MC {"fixed":{"type_I_boundary":{"analytic":0.016666666666666673,"MC":0.016605,"MC_SE":0.0002862594006219612},"power":{"analytic":1.0,"MC":1.0,"MC_SE":0.0}},"crossed":{"type_I_boundary":{"analytic":0.009397297506861394,"MC":0.00971,"MC_SE":0.00021574276704479494},"power":{"analytic":0.2123073818445813,"MC":0.213745,"MC_SE":0.0009144204652644221}},"naive_cluster_t_with_random_seeds":{"type_I_boundary":{"analytic":0.3356980107410371,"MC":0.336555,"MC_SE":0.001055947101718518},"power":{"analytic":0.9866975327847117,"MC":0.986325,"MC_SE":0.0002561787616223723}}}
CTTC_9_sessions_x8_rho_0.00 nll {"planning":{"excess":0.04,"fixed_min_groups":3,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.8594062281784283,"large_B_power":0.3097754540510828,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":1.0,"random_power_at_available":0.2925081017119681},"0p02_excess":{"excess":0.02,"fixed_min_groups":4,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.2997238588856113,"large_B_power":0.11080417216654499,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":0.9999999999963006,"random_power_at_available":0.09444151896611352},"G1_observed_excess":{"excess":0.015650002550707035,"fixed_min_groups":5,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.1911096047056438,"large_B_power":0.08048038676617729,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":0.9999996608047298,"random_power_at_available":0.06581583160664117}} MC {"fixed":{"type_I_boundary":{"analytic":0.016666666666666673,"MC":0.016405,"MC_SE":0.0002862594006219612},"power":{"analytic":1.0,"MC":1.0,"MC_SE":0.0}},"crossed":{"type_I_boundary":{"analytic":0.010878906328404051,"MC":0.010965,"MC_SE":0.00023195425977444198},"power":{"analytic":0.2925081017119681,"MC":0.292105,"MC_SE":0.0010172195243526075}},"naive_cluster_t_with_random_seeds":{"type_I_boundary":{"analytic":0.3594215612572894,"MC":0.358705,"MC_SE":0.00107293453332592},"power":{"analytic":0.9974300604604283,"MC":0.997315,"MC_SE":0.00011321075369271879}}}
CTTC_9_sessions_x8_rho_0.10 accuracy {"planning":{"excess":0.03,"fixed_min_groups":4,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.715370889001055,"large_B_power":0.23839454201753918,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":1.0,"random_power_at_available":0.20809962363581241},"0p02_excess":{"excess":0.02,"fixed_min_groups":5,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.3707306275129978,"large_B_power":0.13018457417462473,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":0.999999993494671,"random_power_at_available":0.1026524897912729},"G1_observed_excess":{"excess":0.02392626166242578,"fixed_min_groups":4,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.5092055120294001,"large_B_power":0.16924723379151027,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":0.9999999999995671,"random_power_at_available":0.13990581376483033}} MC {"fixed":{"type_I_boundary":{"analytic":0.016666666666666673,"MC":0.01684,"MC_SE":0.0002862594006219612},"power":{"analytic":1.0,"MC":1.0,"MC_SE":0.0}},"crossed":{"type_I_boundary":{"analytic":0.008617234258279602,"MC":0.00898,"MC_SE":0.00020667580327674468},"power":{"analytic":0.20809962363581241,"MC":0.206645,"MC_SE":0.0009077284017767805}},"naive_cluster_t_with_random_seeds":{"type_I_boundary":{"analytic":0.3219681510952768,"MC":0.32241,"MC_SE":0.0010447599264318235},"power":{"analytic":0.9849487690872859,"MC":0.985235,"MC_SE":0.0002722562337277721}}}
CTTC_9_sessions_x8_rho_0.10 nll {"planning":{"excess":0.04,"fixed_min_groups":3,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.8594062281784283,"large_B_power":0.3097754540510828,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":1.0,"random_power_at_available":0.2873176420257712},"0p02_excess":{"excess":0.02,"fixed_min_groups":5,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.2997238588856113,"large_B_power":0.11080417216654499,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":0.9999999916919816,"random_power_at_available":0.08972074446296817},"G1_observed_excess":{"excess":0.015650002550707035,"fixed_min_groups":6,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.1911096047056438,"large_B_power":0.08048038676617729,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":0.9999697139321739,"random_power_at_available":0.06178381324220833}} MC {"fixed":{"type_I_boundary":{"analytic":0.016666666666666673,"MC":0.01659,"MC_SE":0.0002862594006219612},"power":{"analytic":1.0,"MC":1.0,"MC_SE":0.0}},"crossed":{"type_I_boundary":{"analytic":0.009662323447651814,"MC":0.009895,"MC_SE":0.00021873457606474533},"power":{"analytic":0.2873176420257712,"MC":0.287985,"MC_SE":0.001011845380002602}},"naive_cluster_t_with_random_seeds":{"type_I_boundary":{"analytic":0.3385912905472924,"MC":0.338555,"MC_SE":0.0010581758561619412},"power":{"analytic":0.9967906925314396,"MC":0.996835,"MC_SE":0.0001264714950914365}}}
CTTC_9_sessions_x8_rho_0.25 accuracy {"planning":{"excess":0.03,"fixed_min_groups":4,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.715370889001055,"large_B_power":0.23839454201753918,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":0.9999999999999999,"random_power_at_available":0.20187546795574646},"0p02_excess":{"excess":0.02,"fixed_min_groups":5,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.3707306275129978,"large_B_power":0.13018457417462473,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":0.9999995245798564,"random_power_at_available":0.09737927640481825},"G1_observed_excess":{"excess":0.02392626166242578,"fixed_min_groups":4,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.5092055120294001,"large_B_power":0.16924723379151027,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":0.9999999997537985,"random_power_at_available":0.13404998518387687}} MC {"fixed":{"type_I_boundary":{"analytic":0.016666666666666673,"MC":0.01693,"MC_SE":0.0002862594006219612},"power":{"analytic":0.9999999999999999,"MC":1.0,"MC_SE":2.3560804576936211E-11}},"crossed":{"type_I_boundary":{"analytic":0.007625202768767457,"MC":0.0077,"MC_SE":0.00019451296938125537},"power":{"analytic":0.20187546795574646,"MC":0.200875,"MC_SE":0.0008975571385527354}},"naive_cluster_t_with_random_seeds":{"type_I_boundary":{"analytic":0.303990416343727,"MC":0.3023,"MC_SE":0.001028543249248408},"power":{"analytic":0.9822401879982025,"MC":0.982655,"MC_SE":0.0002953337186934329}}}
CTTC_9_sessions_x8_rho_0.25 nll {"planning":{"excess":0.04,"fixed_min_groups":4,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.8594062281784283,"large_B_power":0.3097754540510828,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":1.0,"random_power_at_available":0.27953324578061106},"0p02_excess":{"excess":0.02,"fixed_min_groups":5,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.2997238588856113,"large_B_power":0.11080417216654499,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":0.999996710533198,"random_power_at_available":0.08318157982666655},"G1_observed_excess":{"excess":0.015650002550707035,"fixed_min_groups":6,"random_min_groups":{"B":null,"known_variance_power_ceiling":0.1911096047056438,"large_B_power":0.08048038676617729,"status":"unattainable_even_with_known_variance"},"fixed_power_at_available":0.9990589960466872,"random_power_at_available":0.056356067354542766}} MC {"fixed":{"type_I_boundary":{"analytic":0.016666666666666673,"MC":0.01657,"MC_SE":0.0002862594006219612},"power":{"analytic":1.0,"MC":1.0,"MC_SE":0.0}},"crossed":{"type_I_boundary":{"analytic":0.00820718800334233,"MC":0.00807,"MC_SE":0.00020174030420840706},"power":{"analytic":0.27953324578061106,"MC":0.279675,"MC_SE":0.001003479970612188}},"naive_cluster_t_with_random_seeds":{"type_I_boundary":{"analytic":0.31310588983627624,"MC":0.314785,"MC_SE":0.0010369922651257094},"power":{"analytic":0.9957425240678627,"MC":0.995905,"MC_SE":0.0001455910339070472}}}
```

### Event #38: `item.completed` — `command_execution` (`item_35`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$j=Get-Content -Raw research/stage4_math_and_design/M2/results_m2.json | ConvertFrom-Json
foreach($c in $j.Q3.psobject.Properties){
 $v=$c.Value
 '"'{0} B={1}: HY={2:F8} HS={3:F8} HU={4:F8} I={5:F8} Ical={6:F8}; refs {7:F8} {8:F8} cal {9:F8}; Delta {10}' -f "'$c.Name,$v.blocks,$v.H_Y,$v.H_Y_given_signed_code,$v.H_Y_given_unsigned_code,$v.signed_conditional_MI_plugin,$v.independent_cal_reference.signed_conditional_MI,$v.references.direct_table_probs.nll,$v.references.full_iq_probs.nll,$v.independent_cal_reference.blind_metrics.nll,($v.delta_nll_zero_minus_phase | ConvertTo-Json -Compress)
 foreach($a in @('"'phase','zero')){
 "'$x=$v.arms.$a
 "  $a optT $($x.T_star_by_seed.T_star -join '"', ') shared "'$($x.single_shared_raw_temperature.T_star) pieces $($x.additive_path | ConvertTo-Json -Compress)"
 foreach($k in @('"'dev','opt','arith','geo','F0')){ "'$m=$x.metrics.$k; '"'  {0}: {1:F8} {2:F8} {3:F8} {4:F8}' -f "'$k,$m.acc,$m.nll,$m.brier,$m.ece }
 "  fit res $($x.forecast_argmax.resolution) vector $($x.forecast_vector.resolution) marginal $($x.raw_joint_KL.prior_kl) signben $($x.sign_blind_minus_matched_nll) NIG $($x.NIG_realized.empirical_signed_code)"
 "  group $($x.group_on_saved_design | ConvertTo-Json -Compress)"
 "  complete $($x.group_on_complete_orbit | ConvertTo-Json -Compress) acc $($x.complete_orbit_accuracy | ConvertTo-Json -Compress)"
 "  arith thresholds $($x.group_accuracy_transitions.arith | ConvertTo-Json -Compress) geo $($x.group_accuracy_transitions.geo | ConvertTo-Json -Compress)"
 }
}
$j.work_f_intervention_factorial | ConvertTo-Json -Depth 5 -Compress
foreach($a in @('"'phase','zero')){
 "'$f=$j.work_f_primary_040.arms.$a
 "$a gain=$($f.seed_mean_orientation_gain) swap=$($f.seed_mean_swapped_minus_matched_nll) ensemble swap=$($f.seed_averaged_table_metrics.swapped_minus_matched_nll_t95.mean)"
}'
```

**Aggregated Output:**

```text
G1_main_extrap_both B=131: HY=1.38629239 HS=0.92563349 HU=1.29844896 I=0.37281547 Ical=0.36311152; refs 0.94112582 0.92903799 cal 0.94128711; Delta {"bayes_entropy":0.0,"temperature_gap":0.03654337105822347,"projection_gap_at_T_star":0.00012375083194759995,"projected_marginal_prior_kl":5.181785844854403E-08,"projected_reverse_conditional_kl":-0.0010171711573229159,"total":0.035650002550706716,"at_seedwise_T_star":-0.0008933685075167563}
  phase optT infinity, 10.143939616115972, 5.719906677452968 shared 8.123334042583005 pieces {"bayes_entropy":0.9256334923339425,"temperature_gap":0.059756727627698236,"projection_gap_at_T_star":0.0003641482380882799,"projected_marginal_prior_kl":1.9150328023354035E-06,"projected_reverse_conditional_kl":0.4594722704265394}
  dev: 0.23798638 1.44522855 0.78447619 0.11860426
  opt: 0.23267865 1.38547183 0.74970972 0.02560791
  arith: 0.17050334 1.41131239 0.76959810 0.15043941
  geo: 0.16358540 1.41158582 0.77019422 0.16075634
  F0: 0.20883031 1.41116337 0.76289094 0.12505746
  fit res 0.011617683960125635 vector 0.46065890055141623 marginal 8.07619730466835E-05 signben 0.007703718668528259 NIG -0.12793883001753456
  group {"raw_nll":1.4452285536590708,"arithmetic_nll":1.4113123906537837,"geometric_nll":1.4115858232736398,"randomized_orbit_raw_nll":1.4451794190434344,"label_weighted_arithmetic_jensen_gap":0.033867028389650985,"design_imbalance_nll":4.913461563638144E-05,"reverse_barycenter_kl":0.03430076771192796,"geometric_normalizer_gain":0.033593595769794794,"raw_equivariance_max":0.481667162214782,"arith_equivariance_max":2.2204460492503131E-16,"geo_equivariance_max":2.7755575615628914E-16,"arith_idempotence_max":2.2204460492503131E-16,"geo_idempotence_max":2.7755575615628914E-16}
  complete {"raw_nll":1.4451794190434344,"arithmetic_nll":1.4113123906537839,"geometric_nll":1.4115858232736398,"randomized_orbit_raw_nll":1.4451794190434344,"label_weighted_arithmetic_jensen_gap":0.03386702838965099,"design_imbalance_nll":0.0,"reverse_barycenter_kl":0.03430076771192796,"geometric_normalizer_gain":0.033593595769794794,"raw_equivariance_max":0.481667162214782,"arith_equivariance_max":2.2204460492503131E-16,"geo_equivariance_max":2.7755575615628914E-16,"arith_idempotence_max":2.2204460492503131E-16,"geo_idempotence_max":2.7755575615628914E-16} acc {"raw":0.2377420284669211,"arith":0.17050333969465647,"geo":0.16358540076335878}
  arith thresholds {"improved_fraction":0.08760734732824428,"worsened_fraction":0.15509038379983037,"weighted_gains":0.08760734732824428,"weighted_losses":0.15509038379983037,"delta_acc":-0.06748303647158609} geo {"improved_fraction":0.08284298134011875,"worsened_fraction":0.15724395674300254,"weighted_gains":0.08284298134011875,"weighted_losses":0.15724395674300254,"delta_acc":-0.0744009754028838}
  zero optT infinity, 6.453259912111747, 6.086692402990419 shared 9.231676378670661 pieces {"bayes_entropy":0.9256334923339425,"temperature_gap":0.09630009868592171,"projection_gap_at_T_star":0.00048789907003587984,"projected_marginal_prior_kl":1.9668506607839475E-06,"projected_reverse_conditional_kl":0.45845509926921646}
  dev: 0.19406011 1.48087856 0.80101560 0.18273708
  opt: 0.21080497 1.38457846 0.74937704 0.05122855
  arith: 0.16428117 1.43912226 0.78523643 0.19248299
  geo: 0.16597090 1.44585377 0.78840957 0.19565516
  F0: 0.20139552 1.50902593 0.80875474 0.19003310
  fit res 0.021040194771263288 vector 0.46065890055141623 marginal 0.00035845683715984844 signben 0.002672035414306606 NIG -0.20532798391868154
  group {"raw_nll":1.4808785562097775,"arithmetic_nll":1.4391222583759151,"geometric_nll":1.4458537695701237,"randomized_orbit_raw_nll":1.4803280973227513,"label_weighted_arithmetic_jensen_gap":0.04120583894683642,"design_imbalance_nll":0.0005504588870262062,"reverse_barycenter_kl":0.03488441865389562,"geometric_normalizer_gain":0.03447432775262785,"raw_equivariance_max":0.41957202648281566,"arith_equivariance_max":2.2204460492503131E-16,"geo_equivariance_max":2.7755575615628914E-16,"arith_idempotence_max":2.2204460492503131E-16,"geo_idempotence_max":2.7755575615628914E-16}
  complete {"raw_nll":1.4803280973227517,"arithmetic_nll":1.4391222583759151,"geometric_nll":1.4458537695701237,"randomized_orbit_raw_nll":1.4803280973227513,"label_weighted_arithmetic_jensen_gap":0.04120583894683642,"design_imbalance_nll":4.4408920985006262E-16,"reverse_barycenter_kl":0.03488441865389562,"geometric_normalizer_gain":0.03447432775262785,"raw_equivariance_max":0.41957202648281566,"arith_equivariance_max":2.2204460492503131E-16,"geo_equivariance_max":2.7755575615628914E-16,"arith_idempotence_max":2.2204460492503131E-16,"geo_idempotence_max":2.7755575615628914E-16} acc {"raw":0.1939085361005089,"arith":0.16428117048346055,"geo":0.1659708969465649}
  arith thresholds {"improved_fraction":0.06606499151823579,"worsened_fraction":0.09584393553859202,"weighted_gains":0.06606499151823579,"weighted_losses":0.09584393553859202,"delta_acc":-0.029778944020356236} geo {"improved_fraction":0.06739689355385921,"worsened_fraction":0.0954861111111111,"weighted_gains":0.06739689355385921,"weighted_losses":0.0954861111111111,"delta_acc":-0.028089217557251907}
G2_interp_both B=8: HY=1.38501047 HS=0.82487682 HU=1.11782137 I=0.29294455 Ical=0.17686603; refs 1.02223002 1.01934557 cal 1.02410452; Delta {"bayes_entropy":0.0,"temperature_gap":-4.3507760095984338E-05,"projection_gap_at_T_star":0.003414616417591665,"projected_marginal_prior_kl":-8.8448202253621488E-05,"projected_reverse_conditional_kl":-0.031042273816833943,"total":-0.02775961336159183,"at_seedwise_T_star":-0.0277161056014954}
  phase optT 0.5476449725714686, 1.007599102955738, 0.7688741200513519 shared 0.901667150012624 pieces {"bayes_entropy":0.8248768164658911,"temperature_gap":0.0012820319681758896,"projection_gap_at_T_star":0.022336477738029492,"projected_marginal_prior_kl":0.0011681216632017932,"projected_reverse_conditional_kl":0.4861629044172898}
  dev: 0.33875868 1.33582635 0.72818489 0.02296979
  opt: 0.33875868 1.33454432 0.72700906 0.01135405
  arith: 0.40950521 1.30634903 0.71177432 0.08462899
  geo: 0.40429688 1.29937059 0.70953226 0.07558400
  F0: 0.41406250 1.29978965 0.70797371 0.07225298
  fit res 0.04172632128319487 vector 0.5601336565615118 marginal 0.0011781661244837371 signben 0.0028833254936890818 NIG 0.08780782978966248
  group {"raw_nll":1.3358263522525884,"arithmetic_nll":1.306349031969076,"geometric_nll":1.2993705887088214,"randomized_orbit_raw_nll":1.3342096253029465,"label_weighted_arithmetic_jensen_gap":0.027860593333870667,"design_imbalance_nll":0.001616726949641878,"reverse_barycenter_kl":0.03572560284141581,"geometric_normalizer_gain":0.034839036594125294,"raw_equivariance_max":0.4853085620085492,"arith_equivariance_max":2.2204460492503131E-16,"geo_equivariance_max":2.2204460492503131E-16,"arith_idempotence_max":2.2204460492503131E-16,"geo_idempotence_max":2.7755575615628914E-16}
  complete {"raw_nll":1.3342096253029465,"arithmetic_nll":1.306349031969076,"geometric_nll":1.2993705887088214,"randomized_orbit_raw_nll":1.3342096253029467,"label_weighted_arithmetic_jensen_gap":0.027860593333870667,"design_imbalance_nll":-2.2204460492503131E-16,"reverse_barycenter_kl":0.035725602841415814,"geometric_normalizer_gain":0.034839036594125294,"raw_equivariance_max":0.4853085620085492,"arith_equivariance_max":2.2204460492503131E-16,"geo_equivariance_max":2.2204460492503131E-16,"arith_idempotence_max":2.2204460492503131E-16,"geo_idempotence_max":2.7755575615628914E-16} acc {"raw":0.3367919921875,"arith":0.4095052083333333,"geo":0.404296875}
  arith thresholds {"improved_fraction":0.17903645833333334,"worsened_fraction":0.10828993055555555,"weighted_gains":0.17903645833333334,"weighted_losses":0.10828993055555555,"delta_acc":0.07074652777777778} geo {"improved_fraction":0.17567274305555555,"worsened_fraction":0.1101345486111111,"weighted_gains":0.17567274305555555,"weighted_losses":0.1101345486111111,"delta_acc":0.06553819444444445}
  zero optT 1.220140480826934, 0.9157720286054004, 0.7109111592096737 shared 0.9158789997484711 pieces {"bayes_entropy":0.8248768164658911,"temperature_gap":0.0012385242080799053,"projection_gap_at_T_star":0.025751094155621157,"projected_marginal_prior_kl":0.0010796734609481717,"projected_reverse_conditional_kl":0.4551206306004559}
  dev: 0.36360677 1.30806674 0.71571522 0.02301964
  opt: 0.36360677 1.30682821 0.71470445 0.01734659
  arith: 0.38313802 1.27608216 0.69947045 0.04122962
  geo: 0.38509115 1.27346630 0.69875087 0.04088365
  F0: 0.35319010 1.31320499 0.71834605 0.03493101
  fit res 0.054410583524747036 vector 0.5601336565615118 marginal 0.001318006273765493 signben 0.001632850674722519 NIG 0.137366739589869
  group {"raw_nll":1.3080667388909966,"arithmetic_nll":1.2760821620443754,"geometric_nll":1.2734663024136326,"randomized_orbit_raw_nll":1.308030419135557,"label_weighted_arithmetic_jensen_gap":0.0319482570911816,"design_imbalance_nll":3.6319755439606638E-05,"reverse_barycenter_kl":0.03498977238008491,"geometric_normalizer_gain":0.03456411672192424,"raw_equivariance_max":0.459947882561498,"arith_equivariance_max":2.2204460492503131E-16,"geo_equivariance_max":2.2204460492503131E-16,"arith_idempotence_max":2.7755575615628914E-16,"geo_idempotence_max":2.7755575615628914E-16}
  complete {"raw_nll":1.308030419135557,"arithmetic_nll":1.2760821620443754,"geometric_nll":1.2734663024136326,"randomized_orbit_raw_nll":1.3080304191355567,"label_weighted_arithmetic_jensen_gap":0.03194825709118159,"design_imbalance_nll":2.2204460492503131E-16,"reverse_barycenter_kl":0.03498977238008491,"geometric_normalizer_gain":0.03456411672192425,"raw_equivariance_max":0.459947882561498,"arith_equivariance_max":2.2204460492503131E-16,"geo_equivariance_max":2.2204460492503131E-16,"arith_idempotence_max":2.7755575615628914E-16,"geo_idempotence_max":2.7755575615628914E-16} acc {"raw":0.3673095703125,"arith":0.3831380208333333,"geo":0.3850911458333333}
  arith thresholds {"improved_fraction":0.1052517361111111,"worsened_fraction":0.0857204861111111,"weighted_gains":0.1052517361111111,"weighted_losses":0.0857204861111111,"delta_acc":0.01953125} geo {"improved_fraction":0.10709635416666667,"worsened_fraction":0.08561197916666667,"weighted_gains":0.10709635416666667,"weighted_losses":0.08561197916666667,"delta_acc":0.021484375}
G3_extrap_angle B=8: HY=1.38620246 HS=0.73410981 HU=1.16148495 I=0.42737514 Ical=0.34400953; refs 0.97956751 0.95391566 cal 0.98000534; Delta {"bayes_entropy":0.0,"temperature_gap":0.03751253567167567,"projection_gap_at_T_star":-1.3107488900931941E-05,"projected_marginal_prior_kl":-9.0547063452683929E-07,"projected_reverse_conditional_kl":-0.0004155876189768648,"total":0.037082935093163405,"at_seedwise_T_star":-0.00042960057851226807}
  phase optT 9.890197170190602, 8.24011573458638, 5.041577323580456 shared 6.711458275282907 pieces {"bayes_entropy":0.7341098119424454,"temperature_gap":0.05588106242495816,"projection_gap_at_T_star":3.7299267397594704E-05,"projected_marginal_prior_kl":8.2079317485161951E-05,"projected_reverse_conditional_kl":0.6509369480314624}
  dev: 0.22677951 1.44104720 0.78297508 0.12169534
  opt: 0.22677951 1.38516614 0.74961785 0.03354605
  arith: 0.18391927 1.41022164 0.76852169 0.13335272
  geo: 0.18684896 1.41040923 0.76901636 0.13396185
  F0: 0.19986979 1.41263509 0.76325850 0.13013383
  fit res 0.013460455922685854 vector 0.6520926517390777 marginal 0.00016497805516341169 signben 0.0048416299864915135 NIG -0.08410574349512905
  group {"raw_nll":1.4410472009837487,"arithmetic_nll":1.4102216404289691,"geometric_nll":1.4104092257146195,"randomized_orbit_raw_nll":1.4442602610620527,"label_weighted_arithmetic_jensen_gap":0.03403862063308378,"design_imbalance_nll":-0.0032130600783040197,"reverse_barycenter_kl":0.03459930683688427,"geometric_normalizer_gain":0.03385103534743311,"raw_equivariance_max":0.4468965235309015,"arith_equivariance_max":2.2204460492503131E-16,"geo_equivariance_max":2.7755575615628914E-16,"arith_idempotence_max":2.2204460492503131E-16,"geo_idempotence_max":2.4980018054066022E-16}
  complete {"raw_nll":1.4442602610620527,"arithmetic_nll":1.410221640428969,"geometric_nll":1.41040922571462,"randomized_orbit_raw_nll":1.4442602610620527,"label_weighted_arithmetic_jensen_gap":0.034038620633083765,"design_imbalance_nll":0.0,"reverse_barycenter_kl":0.03459930683688427,"geometric_normalizer_gain":0.03385103534743312,"raw_equivariance_max":0.4468965235309015,"arith_equivariance_max":2.2204460492503131E-16,"geo_equivariance_max":2.7755575615628914E-16,"arith_idempotence_max":2.2204460492503131E-16,"geo_idempotence_max":2.4980018054066022E-16} acc {"raw":0.22878011067708334,"arith":0.18391927083333334,"geo":0.18684895833333334}
  arith thresholds {"improved_fraction":0.10004340277777778,"worsened_fraction":0.14290364583333334,"weighted_gains":0.10004340277777778,"weighted_losses":0.14290364583333334,"delta_acc":-0.04286024305555555} geo {"improved_fraction":0.1025390625,"worsened_fraction":0.14246961805555555,"weighted_gains":0.1025390625,"weighted_losses":0.14246961805555555,"delta_acc":-0.03993055555555555}
  zero optT infinity, 6.841685852170579, 5.996214806087316 shared 8.912911256997479 pieces {"bayes_entropy":0.7341098119424454,"temperature_gap":0.09339359809663383,"projection_gap_at_T_star":2.4191778496662764E-05,"projected_marginal_prior_kl":8.1173846850635112E-05,"projected_reverse_conditional_kl":0.6505213604124855}
  dev: 0.19737413 1.47813014 0.79982050 0.17919568
  opt: 0.20865885 1.38473654 0.74943053 0.05293832
  arith: 0.18196615 1.44157865 0.78536744 0.17112533
  geo: 0.17643229 1.44758381 0.78808039 0.18124521
  F0: 0.21549479 1.52255239 0.81180680 0.18598083
  fit res 0.018436587247149472 vector 0.6520926517390778 marginal 0.000139266362981727 signben 0.002619845852320335 NIG -0.1409733297104717
  group {"raw_nll":1.4781301360769121,"arithmetic_nll":1.441578647952712,"geometric_nll":1.4475838095298807,"randomized_orbit_raw_nll":1.4824564043710984,"label_weighted_arithmetic_jensen_gap":0.040877756418386556,"design_imbalance_nll":-0.004326268294186253,"reverse_barycenter_kl":0.035270157795772236,"geometric_normalizer_gain":0.034872594841217505,"raw_equivariance_max":0.4037575297667021,"arith_equivariance_max":1.6653345369377348E-16,"geo_equivariance_max":2.7755575615628914E-16,"arith_idempotence_max":2.7755575615628914E-16,"geo_idempotence_max":2.7755575615628914E-16}
  complete {"raw_nll":1.4824564043710984,"arithmetic_nll":1.4415786479527117,"geometric_nll":1.4475838095298807,"randomized_orbit_raw_nll":1.4824564043710984,"label_weighted_arithmetic_jensen_gap":0.040877756418386556,"design_imbalance_nll":0.0,"reverse_barycenter_kl":0.035270157795772236,"geometric_normalizer_gain":0.034872594841217505,"raw_equivariance_max":0.4037575297667021,"arith_equivariance_max":1.6653345369377348E-16,"geo_equivariance_max":2.7755575615628914E-16,"arith_idempotence_max":2.7755575615628914E-16,"geo_idempotence_max":2.7755575615628914E-16} acc {"raw":0.19539388020833334,"arith":0.18196614583333334,"geo":0.17643229166666666}
  arith thresholds {"improved_fraction":0.08018663194444445,"worsened_fraction":0.09559461805555555,"weighted_gains":0.08018663194444445,"weighted_losses":0.09559461805555555,"delta_acc":-0.015407986111111112} geo {"improved_fraction":0.0769314236111111,"worsened_fraction":0.0978732638888889,"weighted_gains":0.0769314236111111,"weighted_losses":0.0978732638888889,"delta_acc":-0.020941840277777776}
G4_extrap_freq B=8: HY=1.38621947 HS=0.80419825 HU=1.07607178 I=0.27187353 Ical=0.17939758; refs 1.01585113 1.00373805 cal 1.01678078; Delta {"bayes_entropy":0.0,"temperature_gap":0.0014346934374167564,"projection_gap_at_T_star":0.004432449232268043,"projected_marginal_prior_kl":2.4899929810362487E-05,"projected_reverse_conditional_kl":-0.030808590236920952,"total":-0.024916547637425346,"at_seedwise_T_star":-0.026351241074842102}
  phase optT 0.546526256885924, 0.9450099385487952, 0.7434588299740386 shared 0.856924123822296 pieces {"bayes_entropy":0.8041982508968434,"temperature_gap":0.0006014211493565735,"projection_gap_at_T_star":0.01876261314056893,"projected_marginal_prior_kl":9.5736136157598128E-05,"projected_reverse_conditional_kl":0.5110091906818393}
  dev: 0.33289931 1.33466721 0.72746226 0.02762991
  opt: 0.33289931 1.33406579 0.72679264 0.01839701
  arith: 0.40787760 1.31124748 0.71499256 0.09054945
  geo: 0.40527344 1.30494224 0.71308736 0.08233961
  F0: 0.38867188 1.31302278 0.71527220 0.04986078
  fit res 0.03962599897691321 vector 0.5820212161658299 marginal 0.00012128110845155415 signben 0.003171104511457745 NIG 0.08857452894504007
  group {"raw_nll":1.3346672120047656,"arithmetic_nll":1.3112474752147187,"geometric_nll":1.3049422449892962,"randomized_orbit_raw_nll":1.3384130723171919,"label_weighted_arithmetic_jensen_gap":0.02716559710247299,"design_imbalance_nll":-0.003745860312426297,"reverse_barycenter_kl":0.034247305140643586,"geometric_normalizer_gain":0.0334708273278954,"raw_equivariance_max":0.46493231191062434,"arith_equivariance_max":2.7755575615628914E-16,"geo_equivariance_max":2.7755575615628914E-16,"arith_idempotence_max":2.2204460492503131E-16,"geo_idempotence_max":2.7755575615628914E-16}
  complete {"raw_nll":1.3384130723171919,"arithmetic_nll":1.3112474752147187,"geometric_nll":1.3049422449892962,"randomized_orbit_raw_nll":1.3384130723171919,"label_weighted_arithmetic_jensen_gap":0.027165597102472988,"design_imbalance_nll":0.0,"reverse_barycenter_kl":0.034247305140643586,"geometric_normalizer_gain":0.0334708273278954,"raw_equivariance_max":0.46493231191062434,"arith_equivariance_max":2.7755575615628914E-16,"geo_equivariance_max":2.7755575615628914E-16,"arith_idempotence_max":2.2204460492503131E-16,"geo_idempotence_max":2.7755575615628914E-16} acc {"raw":0.3282674153645833,"arith":0.4078776041666667,"geo":0.4052734375}
  arith thresholds {"improved_fraction":0.1804470486111111,"worsened_fraction":0.10546875,"weighted_gains":0.1804470486111111,"weighted_losses":0.10546875,"delta_acc":0.0749782986111111} geo {"improved_fraction":0.1787109375,"worsened_fraction":0.10633680555555555,"weighted_gains":0.1787109375,"weighted_losses":0.10633680555555555,"delta_acc":0.07237413194444445}
  zero optT 1.5353807161698554, 0.8314786113661982, 0.7619854738460363 shared 0.9380211177526723 pieces {"bayes_entropy":0.8041982508968434,"temperature_gap":0.00203611458677333,"projection_gap_at_T_star":0.023195062372836972,"projected_marginal_prior_kl":0.00012063606596796062,"projected_reverse_conditional_kl":0.4802006004449183}
  dev: 0.34960938 1.30975066 0.71818889 0.03253395
  opt: 0.34960938 1.30771455 0.71650667 0.02183381
  arith: 0.37760417 1.27801842 0.70194089 0.03586607
  geo: 0.37402344 1.27470504 0.70101736 0.04537613
  F0: 0.31445313 1.32843541 0.72862797 0.07458632
  fit res 0.05839357793277089 vector 0.5820212161658298 marginal 8.20554036535803E-05 signben 0.002210758453036199 NIG 0.1313849058614824
  group {"raw_nll":1.3097506643673402,"arithmetic_nll":1.278018421041691,"geometric_nll":1.2747050415115322,"randomized_orbit_raw_nll":1.3093708366329362,"label_weighted_arithmetic_jensen_gap":0.031352415591245034,"design_imbalance_nll":0.00037982773440403683,"reverse_barycenter_kl":0.03506518400668496,"geometric_normalizer_gain":0.0346657951214038,"raw_equivariance_max":0.4412390914189377,"arith_equivariance_max":2.2204460492503131E-16,"geo_equivariance_max":2.4980018054066022E-16,"arith_idempotence_max":2.2204460492503131E-16,"geo_idempotence_max":2.7755575615628914E-16}
  complete {"raw_nll":1.3093708366329362,"arithmetic_nll":1.278018421041691,"geometric_nll":1.2747050415115322,"randomized_orbit_raw_nll":1.3093708366329362,"label_weighted_arithmetic_jensen_gap":0.031352415591245034,"design_imbalance_nll":0.0,"reverse_barycenter_kl":0.03506518400668496,"geometric_normalizer_gain":0.034665795121403796,"raw_equivariance_max":0.4412390914189377,"arith_equivariance_max":2.2204460492503131E-16,"geo_equivariance_max":2.4980018054066022E-16,"arith_idempotence_max":2.2204460492503131E-16,"geo_idempotence_max":2.7755575615628914E-16} acc {"raw":0.3553059895833333,"arith":0.3776041666666667,"geo":0.3740234375}
  arith thresholds {"improved_fraction":0.1228298611111111,"worsened_fraction":0.09483506944444445,"weighted_gains":0.1228298611111111,"weighted_losses":0.09483506944444445,"delta_acc":0.027994791666666668} geo {"improved_fraction":0.1203342013888889,"worsened_fraction":0.0959201388888889,"weighted_gains":0.1203342013888889,"weighted_losses":0.0959201388888889,"delta_acc":0.0244140625}
H_c4_035 B=16: HY=1.38626498 HS=0.82069904 HU=1.24535992 I=0.42466088 Ical=0.35003211; refs 0.95013250 0.94262792 cal 0.95161882; Delta {"bayes_entropy":0.0,"temperature_gap":0.021785036582154804,"projection_gap_at_T_star":0.00023215969485912424,"projected_marginal_prior_kl":-1.4811816173021519E-06,"projected_reverse_conditional_kl":-0.003925899009731726,"total":0.018089816085664623,"at_seedwise_T_star":-0.0036952204964899593}
  phase optT 2.404868311839647, 3.271141832123261, 2.454794908601428 shared 2.9053226728958537 pieces {"bayes_entropy":0.8206990364580872,"temperature_gap":0.036059993862712636,"projection_gap_at_T_star":0.002581465165921948,"projected_marginal_prior_kl":2.8514168702753473E-05,"projected_reverse_conditional_kl":0.557617877957268}
  dev: 0.25764974 1.41698689 0.77081560 0.09674409
  opt: 0.25764974 1.38092689 0.74807777 0.01817478
  arith: 0.22021484 1.38462240 0.75552115 0.09827383
  geo: 0.22737630 1.38233036 0.75517757 0.09441744
  F0: 0.25585938 1.38561962 0.75109604 0.08207649
  fit res 0.013885084889676325 vector 0.5655659439548583 marginal 2.96295299867522E-05 signben 0.007489288734297883 NIG -0.05432064559070981
  group {"raw_nll":1.4169868876126925,"arithmetic_nll":1.384622397761325,"geometric_nll":1.3823303616412217,"randomized_orbit_raw_nll":1.4152309482158438,"label_weighted_arithmetic_jensen_gap":0.030608550454518872,"design_imbalance_nll":0.0017559393968487047,"reverse_barycenter_kl":0.03354632333540399,"geometric_normalizer_gain":0.032900586574622415,"raw_equivariance_max":0.4560943297304916,"arith_equivariance_max":1.6653345369377348E-16,"geo_equivariance_max":2.4980018054066022E-16,"arith_idempotence_max":2.2204460492503131E-16,"geo_idempotence_max":2.2204460492503131E-16}
  complete {"raw_nll":1.4152309482158438,"arithmetic_nll":1.384622397761325,"geometric_nll":1.3823303616412215,"randomized_orbit_raw_nll":1.415230948215844,"label_weighted_arithmetic_jensen_gap":0.03060855045451888,"design_imbalance_nll":-2.2204460492503131E-16,"reverse_barycenter_kl":0.03354632333540399,"geometric_normalizer_gain":0.032900586574622415,"raw_equivariance_max":0.4560943297304916,"arith_equivariance_max":1.6653345369377348E-16,"geo_equivariance_max":2.4980018054066022E-16,"arith_idempotence_max":2.2204460492503131E-16,"geo_idempotence_max":2.2204460492503131E-16} acc {"raw":0.25311279296875,"arith":0.22021484375,"geo":0.22737630208333334}
  arith thresholds {"improved_fraction":0.10894097222222222,"worsened_fraction":0.14637586805555555,"weighted_gains":0.10894097222222222,"weighted_losses":0.14637586805555555,"delta_acc":-0.037434895833333336} geo {"improved_fraction":0.11371527777777778,"worsened_fraction":0.1439887152777778,"weighted_gains":0.11371527777777778,"weighted_losses":0.1439887152777778,"delta_acc":-0.0302734375}
  zero optT 7.968279898462918, 2.8363058950481825, 2.338597600605055 shared 3.146909328642314 pieces {"bayes_entropy":0.8206990364580872,"temperature_gap":0.05784503044486744,"projection_gap_at_T_star":0.002813624860781072,"projected_marginal_prior_kl":2.7032987085451321E-05,"projected_reverse_conditional_kl":0.5536919789475363}
  dev: 0.23171658 1.43507670 0.78075401 0.14290947
  opt: 0.23171658 1.37723167 0.74668004 0.04986322
  arith: 0.21630859 1.39586624 0.76439639 0.13606879
  geo: 0.21630859 1.39905873 0.76633686 0.14085191
  F0: 0.21647135 1.46105650 0.78926762 0.17392367
  fit res 0.018439648633616246 vector 0.5655659439548583 marginal 0.00042470258842014497 signben 0.0024740040658912665 NIG -0.08630598042039767
  group {"raw_nll":1.4350767036983574,"arithmetic_nll":1.3958662424012862,"geometric_nll":1.3990587340433105,"randomized_orbit_raw_nll":1.4345742270622068,"label_weighted_arithmetic_jensen_gap":0.0387079846609206,"design_imbalance_nll":0.0005024766361505506,"reverse_barycenter_kl":0.03593372965756302,"geometric_normalizer_gain":0.035515493018896185,"raw_equivariance_max":0.4157977868542352,"arith_equivariance_max":2.2204460492503131E-16,"geo_equivariance_max":2.2204460492503131E-16,"arith_idempotence_max":2.2204460492503131E-16,"geo_idempotence_max":2.7755575615628914E-16}
  complete {"raw_nll":1.4345742270622068,"arithmetic_nll":1.3958662424012862,"geometric_nll":1.3990587340433105,"randomized_orbit_raw_nll":1.4345742270622066,"label_weighted_arithmetic_jensen_gap":0.0387079846609206,"design_imbalance_nll":2.2204460492503131E-16,"reverse_barycenter_kl":0.035933729657563016,"geometric_normalizer_gain":0.03551549301889619,"raw_equivariance_max":0.4157977868542352,"arith_equivariance_max":2.2204460492503131E-16,"geo_equivariance_max":2.2204460492503131E-16,"arith_idempotence_max":2.2204460492503131E-16,"geo_idempotence_max":2.7755575615628914E-16} acc {"raw":0.23109944661458334,"arith":0.21630859375,"geo":0.21630859375}
  arith thresholds {"improved_fraction":0.08490668402777778,"worsened_fraction":0.1003146701388889,"weighted_gains":0.08490668402777778,"weighted_losses":0.1003146701388889,"delta_acc":-0.015407986111111112} geo {"improved_fraction":0.08490668402777778,"worsened_fraction":0.1003146701388889,"weighted_gains":0.08490668402777778,"weighted_losses":0.1003146701388889,"delta_acc":-0.015407986111111112}
{"rot0":{"matched":{"acc":0.23739662849872772,"nll":1.445489515265846,"brier":0.7847129913561824,"ece":0.11869369800273749},"text_only_flip":{"acc":0.22451494910941475,"nll":1.4485098699808283,"brier":0.7863107472823826,"ece":0.12464894692033401},"slot_only_flip":{"acc":0.21672232824427481,"nll":1.4622841311470864,"brier":0.792218354715795,"ece":0.13248861212115745},"joint_flip":{"acc":0.2076972010178117,"nll":1.463859345788775,"brier":0.7922850951060447,"ece":0.14124895051282013},"nll_effects":{"text_flip_when_slots_matched":0.0030203547149822985,"slot_flip_when_text_matched":0.016794615881240427,"difference_in_differences":-0.0014451400732937714,"text_main_effect":0.0022977846783354128,"slot_main_effect":0.01607204584459354}},"all_saved_rotations":{"matched":{"acc":0.23798637616624257,"nll":1.4452285536590708,"brier":0.7844761922050159,"ece":0.1186042615309493},"text_only_flip":{"acc":0.2249324109414758,"nll":1.4484600087194464,"brier":0.7862087762349014,"ece":0.12424769267715033},"slot_only_flip":{"acc":0.21581451441899915,"nll":1.4624683714290447,"brier":0.7922686287423131,"ece":0.13342532499565965},"joint_flip":{"acc":0.20808153095843934,"nll":1.463802709931582,"brier":0.7922006777743874,"ece":0.14085441538235272},"nll_effects":{"text_flip_when_slots_matched":0.0032314550603755876,"slot_flip_when_text_matched":0.017239817769973964,"difference_in_differences":-0.0018971165578383076,"text_main_effect":0.002282896781456434,"slot_main_effect":0.01629125949105481}}}
phase gain=0.009819795034315315 swap=0.01836983052292875 ensemble swap=0.016151408372142323
zero gain=0.005047760462246476 swap=0.00737184100687033 ensemble swap=0.007955366143453383
```

### Event #39: `item.completed` — `agent_message` (`item_36`)

The verifier reproduced every in-scope published value exactly. On G1, the difference in temperature mismatch is `+0.036543` nats, slightly larger than the locked `+0.035650` phase lead. After fitting one temperature per seed on the blind labels, zero leads by `0.000893` nats. Both group pools lower G1 NLL and reduce accuracy.

### Event #40: `item.completed` — `command_execution` (`item_37`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$path='"'research/stage4_math_and_design/M2/verify_m2.py'
"'$content=Get-Content -Raw -Encoding UTF8 -LiteralPath $path
$old=@'"'
        f = {}
'@
"'$new=@'"'
        f, losses = {}, {}
'@
"'$content=$content.Replace($old,$new)
$old=@'"'
            f[k] = metrics(q3_index(pred, codes[:, ix]), y[:, ix])[0]
'@
"'$new=@'"'
            f[k], item = metrics(q3_index(pred, codes[:, ix]), y[:, ix])
            nar = y[:, ix].shape[1]
            losses[k] = item[\"per_item_nll\"].reshape(len(seeds), 2, nar, -1, 64).mean(axis=(1, 2, 4)).T
'@
"'$content=$content.Replace($old,$new)
$old=@'"'
        factorial[grouping] = f
'@
"'$new=@'"'
        m, a, b, d = (losses[k] for k in variants)
        effect_arrays = {\"text_only_flip\": a - m, \"slot_only_flip\": b - m, \"joint_flip\": d - m,
                         \"difference_in_differences\": d - a - b + m}
        f[\"fixed_seed_paired_block_effects\"] = {k: stage3.paired_t_ci(v.mean(axis=1)) for k, v in effect_arrays.items()}
        f[\"effects_by_seed\"] = {k: v.mean(axis=0).tolist() for k, v in effect_arrays.items()}
        factorial[grouping] = f
'@
"'$content=$content.Replace($old,$new)
$old=@'"'
    return result


def run():
'@
"'$new=@'"'
    result[\"ten_seed_sensitivity\"] = {}
    for name, scenario in result[\"scenario_proxies\"].items():
        result[\"ten_seed_sensitivity\"][name] = {}
        for endpoint, entry in scenario.items():
            excess = entry[\"scenarios\"][\"planning\"][\"excess\"]
            v = entry[\"variance_components_at_group_level\"]
            b = entry[\"independent_groups\"]
            result[\"ten_seed_sensitivity\"][name][endpoint] = {
                \"power_at_available\": crossed_power(excess, b, 10, v, alpha, 192),
                \"random_min_groups\": min_crossed(excess, 10, v, alpha, target),
            }
    return result


def run():
'@
if(-not "'$content.Contains($old)){throw '"'Power integration anchor missing'}
"'$content=$content.Replace($old,$new)
$content | Set-Content -Encoding UTF8 -LiteralPath $path
$ErrorActionPreference='"'Stop'
"'$m2=Join-Path (Get-Location) '"'research/stage4_math_and_design/M2'
"'$env:UV_CACHE_DIR=Join-Path $m2 '"'.uv-cache'
"'$env:UV_OFFLINE='"'1'
"'$env:UV_LINK_MODE='"'copy'
"'$env:TMP=$env:UV_CACHE_DIR
$env:TEMP=$env:UV_CACHE_DIR
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M2/verify_m2.py'
```

**Aggregated Output:**

```text
Power integration anchor missing
����λ�� ��:55 �ַ�: 34
+ ... ot $content.Contains($old)){throw 'Power integration anchor missing'}
+                                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : OperationStopped: (Power integration anchor missing:String) [], RuntimeException
    + FullyQualifiedErrorId : Power integration anchor missing
 
```

### Event #41: `item.completed` — `command_execution` (`item_38`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "@'


## 4. Replayed empirical results

The verifier loaded the real blind bundles. It replayed all published G1–G4 cell entries, primary and historical Work F, Work G, and Work A Q2 with maximum absolute difference zero. It checked the NPZ temperatures against both the summary and the protocol lock. H has a published Work F entry and an older positive-world lookup anchor. Its pooled two-sign metrics below are recomputed from its NPZ arrays. Q3 tables are \\((2,3,2,256,4)\\) per cell; Q2 scores are \\((2,3,7,512,4)\\). NLL clipping never affected the scored true-label probabilities used in the decompositions.

### Q3 Bayes terms and the paired decomposition

| Cell | Blocks | Blind \\(H(Y\\mid S,Z)\\) | Blind \\(I(Y;S\\mid Z)\\) | Independent physical calibration MI | Direct NLL | Full-IQ NLL |
|---|---:|---:|---:|---:|---:|---:|
| G1 | 131 | 0.925633 | 0.372815 | 0.363112 | 0.941126 | 0.929038 |
| G2 | 8 | 0.824877 | 0.292945 | 0.176866 | 1.022230 | 1.019346 |
| G3 | 8 | 0.734110 | 0.427375 | 0.344010 | 0.979568 | 0.953916 |
| G4 | 8 | 0.804198 | 0.271874 | 0.179398 | 1.015851 | 1.003738 |
| H | 16 | 0.820699 | 0.424661 | 0.350032 | 0.950133 | 0.942628 |

The independent estimate evaluates \\(I(Y;S\\mid Z)\\) in the joint defined by \\(\\frac12\\,\\texttt{cal\\_mass}(s,z)\\,\\texttt{cal\\_ref}(y\\mid s,z)\\). The reference uses 65,536 independent calibration mothers per sign, quarter-turn pooling of conditional counts, and a half-count per class. Blind counts in the eight-block cells are substantially sparser. The independent estimates show available signed information at 0.80 rad as well as 1.60 rad. They quantify a property of the physical decoding problem.

The next table decomposes \\(\\Delta L=L_{\\rm zero}-L_{\\rm phase}\\). The Bayes term is zero in every row. Prior is the marginal chain-rule term after fitting then projecting.

| Cell | Locked \\(\\Delta L\\) | Temperature | Projection at \\(T"'^*'"\\) | Prior | Reverse conditional KL | \\(\\Delta L\\) at \\(T"'^*'"\\) |
|---|---:|---:|---:|---:|---:|---:|
| G1 | +0.03565000 | +0.03654337 | +0.00012375 | +0.00000005 | −0.00101717 | −0.00089337 |
| G2 | −0.02775961 | −0.00004351 | +0.00341462 | −0.00008845 | −0.03104227 | −0.02771611 |
| G3 | +0.03708294 | +0.03751254 | −0.00001311 | −0.00000091 | −0.00041559 | −0.00042960 |
| G4 | −0.02491655 | +0.00143469 | +0.00443245 | +0.00002490 | −0.03080859 | −0.02635124 |
| H | +0.01808982 | +0.02178504 | +0.00023216 | −0.00000148 | −0.00392590 | −0.00369522 |

**G1.** For phase, the five-term path is \\(0.92563349+0.05975673+0.00036415+0.00000192+0.45947227=1.44522855\\). The zero path is \\(0.92563349+0.09630010+0.00048790+0.00000197+0.45845510=1.48087856\\). Scalar mismatch accounts for 102.51% of the locked lead along this path. Before calibration or projection, marginal KLs are \\(0.00008076\\) and \\(0.00035846\\), respectively. Raw phase and zero empirical-code NIGs are −0.12794 and −0.20533. The S17 direct reference scores 64.62% versus 23.80% and 19.41%. Phase's lower ECE is consistent with the confidence mismatch. It does not measure the share of the NLL lead. Using the argmax set, the empirical resolutions are 0.01162 and 0.02104 nats for phase and zero. Using the exact score vector, both have 0.46066 nats of in-sample resolution because both distinguish the signed codes. Those scores can encode information that the decoded predictions use poorly. With sign probabilities marginalized conditional on code, the NLL cost is 0.007704 for phase and 0.002672 for zero. The difference, 0.005032 nats, is sensitivity to that specific sign-discarding operation. It is distinct from Bayes MI and from the factorial intervention.

**G2 and G4.** Raw zero accuracy leads by 2.485 and 1.671 pp. Finite \\(T"'^*'"\\) preserves those rankings and a fitted loss lead of 0.02772 and 0.02635 nats remains. The reverse-conditional term gives zero advantages of 0.03104 and 0.03081 nats after projection and fitting. The global prior and scalar terms are small. Raw zero's coarsened argmax resolution also exceeds phase's by 0.01268 and 0.01877 nats. Projection actually reverses the top-1 ordering in those cells while leaving zero with lower probabilistic loss. Thus the locked reversal is explained at the score level by conditional probability allocation and raw decision ordering. The five saved conditions do not establish an architectural transition at \\(\\pi/2\\). Check the angle effect by holding \\(|\\omega|\\), horizons, and diffusion fixed or by crossing them prospectively with common noise and more independent seeds.

### Q3 scalar recalibration

Each cell gives locked NLL / ECE and retrospective seedwise \\(T"'^*'"\\) NLL / ECE. Temperature triples follow seeds 202611, 202622, 202633.

| Cell | Arm | Locked NLL / ECE | Fitted NLL / ECE | \\(T"'^*'"\\) |
|---|---|---|---|---|
| G1 | phase | 1.445229 / .118604 | 1.385472 / .025608 | ∞, 10.144, 5.720 |
| G1 | zero | 1.480879 / .182737 | 1.384578 / .051229 | ∞, 6.453, 6.087 |
| G2 | phase | 1.335826 / .022970 | 1.334544 / .011354 | .548, 1.008, .769 |
| G2 | zero | 1.308067 / .023020 | 1.306828 / .017347 | 1.220, .916, .711 |
| G3 | phase | 1.441047 / .121695 | 1.385166 / .033546 | 9.890, 8.240, 5.042 |
| G3 | zero | 1.478130 / .179196 | 1.384737 / .052938 | ∞, 6.842, 5.996 |
| G4 | phase | 1.334667 / .027630 | 1.334066 / .018397 | .547, .945, .743 |
| G4 | zero | 1.309751 / .032534 | 1.307715 / .021834 | 1.535, .831, .762 |
| H | phase | 1.416987 / .096744 | 1.380927 / .018175 | 2.405, 3.271, 2.455 |
| H | zero | 1.435077 / .142909 | 1.377232 / .049863 | 7.968, 2.836, 2.339 |

### Q3 pooling on the saved test design

Each entry is accuracy (%) / NLL. Both pools reduce NLL in all ten rows. The two 1.60-rad cells lose accuracy with either pool, while the two 0.80-rad cells gain accuracy with either. H follows the accuracy-decline pattern for both arms.

| Cell | Arm | Raw | Arithmetic | Geometric |
|---|---|---|---|---|
| G1 | phase | 23.80 / 1.445229 | 17.05 / 1.411312 | 16.36 / 1.411586 |
| G1 | zero | 19.41 / 1.480879 | 16.43 / 1.439122 | 16.60 / 1.445854 |
| G2 | phase | 33.88 / 1.335826 | 40.95 / 1.306349 | 40.43 / 1.299371 |
| G2 | zero | 36.36 / 1.308067 | 38.31 / 1.276082 | 38.51 / 1.273466 |
| G3 | phase | 22.68 / 1.441047 | 18.39 / 1.410222 | 18.68 / 1.410409 |
| G3 | zero | 19.74 / 1.478130 | 18.20 / 1.441579 | 17.64 / 1.447584 |
| G4 | phase | 33.29 / 1.334667 | 40.79 / 1.311247 | 40.53 / 1.304942 |
| G4 | zero | 34.96 / 1.309751 | 37.76 / 1.278018 | 37.40 / 1.274705 |
| H | phase | 25.76 / 1.416987 | 22.02 / 1.384622 | 22.74 / 1.382330 |
| H | zero | 23.17 / 1.435077 | 21.63 / 1.395866 | 21.63 / 1.399059 |

On G1 phase the arithmetic pool makes 8.76% of decisions better and 15.51% worse, the exact weighted hit changes here. Geometric makes 8.28% better and 15.72% worse. G1 phase F0 scores 20.88% / 1.411163. On complete orbits G1 phase's arithmetic gain is 0.03386703, while the reverse-barycenter KL is 0.03430077. For zero the corresponding numbers are 0.04120584 and 0.03488442. Restricting to the three saved rotations adds 0.00004913 and 0.00055046 to the loss gain. Across the tested cells pooled maximum equivariance and idempotence residuals were at floating-point precision. G1 raw maximum probability discrepancies were 0.48167 and 0.41957 for phase and zero. Structural compliance of the offline tables coexists with poor decoding and poor ECE.

### Work F mechanism and averaging convention

\\(\\omega\\) is formatted with explicit sign in the public prompt. The same condition is used to build the prompt for each connector. Only the six slot entries become zero in zero. At a fixed code, the zero-arm signed text remains the varying input to its deterministic table. The physical world used for evaluation also changes which codes and labels are sampled. Work F quantifies these by comparing the signed tables on the same codes within each world.

G1 seed-mean orientation gains are +0.00981980 and +0.00504776. Their linear dependence on the probabilities makes them equal to the corresponding gains on the seed-average table up to rounding. NLL is nonlinear. Seed-mean swapped penalties are +0.01836983 and +0.00737184. Evaluating the arithmetic seed-ensemble table yields +0.01615141 and +0.00795537. The physical reference has orientation gain 1 by definition. Phase per-seed gains are −0.00147, +0.02104, +0.00989. The zero-arm gains are +0.00184, +0.01295, +0.00035. An orientation gain is a weighted projection coefficient onto the reference odd direction. It does not measure a fraction of the Bayes information or the Stage 3D arm contrast. The direct readout is a separate physical predictor with access to the known condition.

On all the saved G1 rotations of the phase checkpoint, a text-only flip raises NLL by +0.00323146 and a slot-only flip by +0.01723982, holding the other channel matched. A joint flip raises it by +0.01857416. The difference in differences is −0.00189712. On rotation zero these are +0.00302035, +0.01679462, +0.01836983, and −0.00144514. The interventions show sensitivity to physically aligned slots in the phase checkpoint, alongside a text response, in the tested models. They compare flipped slots to matched slots within a checkpoint. Training with the phase arm and training with the zero arm is a distinct intervention. A shared-development retraining comparison with a locked recalibration and independent seeds is needed to attribute persistent NLL gains to richer slot use.

### Q2 controls

\\(H(Y)=\\log4\\), while the base-code in-sample \\(H(Y\\mid Z)=0.809758\\) with 183 occupied codes. The Work A no-reference reader yields uniform on these rows. The gap illustrates the distinction between sparsely conditioned, label-reusing estimates and physical predictive references.

| Control | Work A NLL | Locked phase / zero NLL | Fitted phase / zero NLL | \\(\\Delta\\) temperature | Locked \\(\\Delta\\) |
|---|---:|---|---|---:|---:|
| none | 1.386294 | 1.392846 / 1.393822 | 1.386272 / 1.386294 | +.000953 | +.000976 |
| valid | .819207 | 1.388883 / 1.410690 | 1.383282 / 1.384875 | +.020214 | +.021807 |
| duplicate dedup | .819207 | 1.388883 / 1.410690 | 1.383282 / 1.384875 | +.020214 | +.021807 |
| duplicate raw | .819207 | 1.394420 / 1.413341 | 1.384193 / 1.385808 | +.017307 | +.018922 |
| independent only | 1.386294 | 1.407714 / 1.420522 | 1.386294 / 1.386071 | +.013031 | +.012807 |
| valid plus independent | .819207 | 1.408232 / 1.418281 | 1.386244 / 1.385545 | +.010748 | +.010049 |
| fresh resample | .663189 | 1.405640 / 1.406397 | 1.385318 / 1.382645 | +.003429 | +.000757 |

For valid, the fitted residual phase lead \\(+0.00159367\\) is \\(+0.00028827\\) reference-prior KL, \\(+0.00158510\\) reverse KL, and \\(−0.00027971\\) realized-reference label residual. For fresh resample, the remaining lead reverses to \\(−0.00267249\\). The Work A reference gains 0.156018 from fresh vs valid, while the phase connector is 0.016757 worse at locked \\(T\\). Q2 realized NIG is undefined for the uniform Work A controls because the denominator is zero. All Q2 controls and arms have complete numerical paths, ECE, Brier, scalar temperatures, and soft reference diagnostics in the JSON.


'@ | Add-Content -LiteralPath 'research/stage4_math_and_design/M2/THEOREM_M2.md' -Encoding UTF8"
```

### Event #42: `item.completed` — `command_execution` (`item_39`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$path='"'research/stage4_math_and_design/M2/verify_m2.py'
"'$content=(Get-Content -Raw -Encoding UTF8 -LiteralPath $path) -replace '"'\\r\\n',([string][char]10)
"'$content=$content.Replace('"'        f = {}','        f, losses = {}, {}')
"'$old='"'            f[k] = metrics(q3_index(pred, codes[:, ix]), y[:, ix])[0]'
"'$new=@'"'
            f[k], item = metrics(q3_index(pred, codes[:, ix]), y[:, ix])
            rotations = y[:, ix].shape[1]
            losses[k] = item[\"per_item_nll\"].reshape(len(seeds), 2, rotations, -1, 64).mean(axis=(1, 2, 4)).T
'@
"'$content=$content.Replace($old,$new)
$old='"'        factorial[grouping] = f'
"'$new=@'"'
        m, a, b, d = (losses[k] for k in variants)
        effect_arrays = {\"text_only_flip\": a - m, \"slot_only_flip\": b - m, \"joint_flip\": d - m,
                         \"difference_in_differences\": d - a - b + m}
        f[\"fixed_seed_paired_block_effects\"] = {k: stage3.paired_t_ci(v.mean(axis=1)) for k, v in effect_arrays.items()}
        f[\"effects_by_seed\"] = {k: v.mean(axis=0).tolist() for k, v in effect_arrays.items()}
        factorial[grouping] = f
'@
"'$content=$content.Replace($old,$new)
$old='"'    return result'+(([string][char]10)*3)+'def run():'
"'$new=@'"'
    result[\"ten_seed_sensitivity\"] = {}
    for name, scenario in result[\"scenario_proxies\"].items():
        result[\"ten_seed_sensitivity\"][name] = {}
        for endpoint, entry in scenario.items():
            excess = entry[\"scenarios\"][\"planning\"][\"excess\"]
            v = entry[\"variance_components_at_group_level\"]
            b = entry[\"independent_groups\"]
            result[\"ten_seed_sensitivity\"][name][endpoint] = {
                \"power_at_available\": crossed_power(excess, b, 10, v, alpha, 192),
                \"random_min_groups\": min_crossed(excess, 10, v, alpha, target),
            }
    return result


def run():
'@
if(-not "'$content.Contains($old)){throw '"'Power integration anchor missing'}
"'$content=$content.Replace($old,$new)
$content | Set-Content -Encoding UTF8 -LiteralPath $path
$ErrorActionPreference='"'Stop'
"'$m2=Join-Path (Get-Location) '"'research/stage4_math_and_design/M2'
"'$env:UV_CACHE_DIR=Join-Path $m2 '"'.uv-cache'
"'$env:UV_OFFLINE='"'1'
"'$env:UV_LINK_MODE='"'copy'
"'$env:TMP=$env:UV_CACHE_DIR
$env:TEMP=$env:UV_CACHE_DIR
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M2/verify_m2.py'
```

**Aggregated Output:**

```text
Power integration anchor missing
����λ�� ��:40 �ַ�: 34
+ ... ot $content.Contains($old)){throw 'Power integration anchor missing'}
+                                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : OperationStopped: (Power integration anchor missing:String) [], RuntimeException
    + FullyQualifiedErrorId : Power integration anchor missing
 
```

### Event #43: `item.completed` — `command_execution` (`item_40`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$s=Get-Content research/stage4_math_and_design/M2/verify_m2.py; $s[591..627]; Get-ChildItem research/stage4_math_and_design/M2 -Force | Select-Object Name,Length'
```

**Aggregated Output:**

```text
                "random_power_at_131": crossed_power(d, 131, s, v, alpha, 192)}
    designs = [(f"B{b}_independent", b, 1, 0) for b in (16, 72, 131)]
    designs.extend((f"CTTC_9_sessions_x8_rho_{rho:.2f}", 9, 8, rho) for rho in (0, .1, .25))
    for design, b, per_group, rho in designs:
        design_result = {}
        for ep, est in g1.items():
            vu, vs, ve = variance(est)
            v = (vu * (rho + (1 - rho) / per_group), vs, ve / per_group)
            d = .03 if ep == "accuracy" else .04
            ef = max(est["mean"] - .02, 0)
            design_result[ep] = {"independent_groups": b, "clusters_per_group": per_group,
                "assumed_block_effect_session_ICC": rho, "variance_components_at_group_level": list(v),
                "scenarios": {key: {"excess": x, "fixed_min_groups": min_fixed(x, s, v, alpha, target) if x else None,
                                      "random_min_groups": min_crossed(x, s, v, alpha, target) if x else None,
                                      "fixed_power_at_available": float(fixed_power(x, b, s, v, alpha)),
                                      "random_power_at_available": crossed_power(x, b, s, v, alpha, 192)}
                              for key, x in (("planning", d), ("0p02_excess", .02), ("G1_observed_excess", ef))},
                "MC_at_planning_excess": monte_carlo(v, b, s, d, alpha, rng, reps)}
        result["scenario_proxies"][design] = design_result
    result["G1_more_seeds"] = {str(num): {ep: {
        "B131": crossed_power(.03 if ep == "accuracy" else .04, 131, num, variance(est), alpha, 192),
        "B16": crossed_power(.03 if ep == "accuracy" else .04, 16, num, variance(est), alpha, 192)}
        for ep, est in g1.items()} for num in (3, 5, 6, 8, 10, 12)}
    return result




def run():
    result_dir = SOURCE / "results"
    summary = read_json(result_dir / "summary_master.json")
    lock = read_json(SOURCE / "PROTOCOL_LOCK.json")
    q3_result, score_matrices, parity = {}, {}, {}
    with np.load(result_dir / "blind_q3_scores.npz", allow_pickle=False) as q3, \
         np.load(result_dir / "blind_q2_scores.npz", allow_pickle=False) as q2:
        seeds = q3["seeds"].tolist()
        close(seeds, summary["seeds"], "seed IDs", atol=0)

Name            Length
----            ------
.uv-cache             
results_m2.json 328343
THEOREM_M2.md   33855 
verify_m2.py    36962 


```

### Event #44: `item.completed` — `command_execution` (`item_42`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$j=Get-Content -Raw research/stage4_math_and_design/M2/results_m2.json | ConvertFrom-Json; $j.power.multiple_testing_check | ConvertTo-Json -Depth 5 -Compress; $j.power.stage3_lock_replay | ConvertTo-Json -Depth 5 -Compress; $j.Q3.G1_main_extrap_both.references | ConvertTo-Json -Depth 5 -Compress; $j.Q3.G1_main_extrap_both.arms.phase.reference_loss_split | ConvertTo-Json -Depth 5 -Compress; $j.Q3.G1_main_extrap_both.arms.zero.reference_loss_split | ConvertTo-Json -Depth 5 -Compress; rg -n '"'ten_seed|fixed_seed_paired_block_effects' research/stage4_math_and_design/M2/verify_m2.py"
```

**Aggregated Output:**

```text
{"IUT_at_independent_three_endpoint_boundaries":{"exact":0.00012500000000000003,"MC":8.5E-05},"correlated_Holm_IUT":{"false_claim_nulls_0":{"FWER_MC":0.040015,"IUT_marginal_type_I_MC":0.049655},"false_claim_nulls_1":{"FWER_MC":0.043665,"IUT_marginal_type_I_MC":0.050365},"false_claim_nulls_2":{"FWER_MC":0.049525,"IUT_marginal_type_I_MC":0.049525}}}
{"accuracy":{"sd":0.09441431300390846,"min_B":131,"power_at_131":0.9505118697757498},"nll":{"sd":0.07979465834252315,"min_B":54,"power_at_131":0.9999060352925451}}
{"direct_table_probs":{"acc":0.6461712786259542,"nll":0.941125824839512,"brier":0.5098317013509959,"ece":0.0009122218241481161},"full_iq_probs":{"acc":0.6464098282442748,"nll":0.9290379865758834,"brier":0.5032444510266945,"ece":0.0046012731581605345}}
{"direct_table_probs":{"reference_entropy":0.940950563802668,"expected_kl":0.503163705163483,"expected_nll":1.4441142689661508,"realized_label_residual":0.0011142846929199646,"realized_nll":1.4452285536590708},"full_iq_probs":{"reference_entropy":0.9279404676494312,"expected_kl":0.5161846733285259,"expected_nll":1.444125140977957,"realized_label_residual":0.0011034126811138023,"realized_nll":1.4452285536590708}}
{"direct_table_probs":{"reference_entropy":0.940950563802668,"expected_kl":0.5392769296320912,"expected_nll":1.480227493434759,"realized_label_residual":0.0006510627750184739,"realized_nll":1.4808785562097775},"full_iq_probs":{"reference_entropy":0.9279404676494312,"expected_kl":0.5522895321928175,"expected_nll":1.4802299998422486,"realized_label_residual":0.0006485563675289185,"realized_nll":1.4808785562097775}}
414:        f["fixed_seed_paired_block_effects"] = {k: stage3.paired_t_ci(v.mean(axis=1)) for k, v in effect_arrays.items()}
621:    result["ten_seed_sensitivity"] = {}
623:        result["ten_seed_sensitivity"][name] = {}
628:            result["ten_seed_sensitivity"][name][endpoint] = {
```

### Event #45: `item.completed` — `command_execution` (`item_41`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$ErrorActionPreference='"'Stop'
"'$path='"'research/stage4_math_and_design/M2/verify_m2.py'
"'$nl=[string][char]10
$content=(Get-Content -Raw -Encoding UTF8 -LiteralPath $path) -replace '"'\\r\\n',"'$nl
$content=$content.Replace('"'        f = {}','        f, losses = {}, {}')
"'$old='"'            f[k] = metrics(q3_index(pred, codes[:, ix]), y[:, ix])[0]'
"'$new=@'"'
            f[k], item = metrics(q3_index(pred, codes[:, ix]), y[:, ix])
            rotations = y[:, ix].shape[1]
            losses[k] = item[\"per_item_nll\"].reshape(len(seeds), 2, rotations, -1, 64).mean(axis=(1, 2, 4)).T
'@
"'$content=$content.Replace($old,$new)
$old='"'        factorial[grouping] = f'
"'$new=@'"'
        m, a, b, d = (losses[k] for k in variants)
        effect_arrays = {\"text_only_flip\": a - m, \"slot_only_flip\": b - m, \"joint_flip\": d - m,
                         \"difference_in_differences\": d - a - b + m}
        f[\"fixed_seed_paired_block_effects\"] = {k: stage3.paired_t_ci(v.mean(axis=1)) for k, v in effect_arrays.items()}
        f[\"effects_by_seed\"] = {k: v.mean(axis=0).tolist() for k, v in effect_arrays.items()}
        factorial[grouping] = f
'@
"'$content=$content.Replace($old,$new)
$idx=$content.IndexOf('"'    return result',"'$content.IndexOf('"'    result[\"G1_more_seeds\"]'))
"'$new=@'"'
    result[\"ten_seed_sensitivity\"] = {}
    for name, scenario in result[\"scenario_proxies\"].items():
        result[\"ten_seed_sensitivity\"][name] = {}
        for endpoint, entry in scenario.items():
            excess = entry[\"scenarios\"][\"planning\"][\"excess\"]
            v = entry[\"variance_components_at_group_level\"]
            b = entry[\"independent_groups\"]
            result[\"ten_seed_sensitivity\"][name][endpoint] = {
                \"power_at_available\": crossed_power(excess, b, 10, v, alpha, 192),
                \"random_min_groups\": min_crossed(excess, 10, v, alpha, target),
            }
'@
"'$content=$content.Insert($idx,$new+$nl)
$content=($content -replace '"'\\n{4,}(?=def )',("'$nl*3)).TrimEnd()+$nl
Set-Content -Encoding UTF8 -NoNewline -LiteralPath $path -Value $content
$ErrorActionPreference='"'Stop'
"'$m2=Join-Path (Get-Location) '"'research/stage4_math_and_design/M2'
"'$env:UV_CACHE_DIR=Join-Path $m2 '"'.uv-cache'
"'$env:UV_OFFLINE='"'1'
"'$env:UV_LINK_MODE='"'copy'
"'$env:TMP=$env:UV_CACHE_DIR
$env:TEMP=$env:UV_CACHE_DIR
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M2/verify_m2.py'
```

**Aggregated Output:**

```text
results_m2.json written. Maximum in-scope summary difference: 0.
```

### Event #46: `item.completed` — `command_execution` (`item_43`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$j=Get-Content -Raw research/stage4_math_and_design/M2/results_m2.json | ConvertFrom-Json
foreach($c in $j.power.ten_seed_sensitivity.PSObject.Properties){
  foreach($e in $c.Value.PSObject.Properties){
    '"'{0} {1} {2:F8} min {3}' -f "'$c.Name,$e.Name,$e.Value.power_at_available,$e.Value.random_min_groups.B
  }
}
$j.work_f_intervention_factorial.all_saved_rotations.fixed_seed_paired_block_effects | ConvertTo-Json -Depth 5 -Compress
$j.work_f_intervention_factorial.all_saved_rotations.effects_by_seed | ConvertTo-Json -Depth 5 -Compress
$n=0; Get-Content -Encoding UTF8 '"'C:/Users/李则徐/.codex/memories/MEMORY.md' | ForEach-Object { "'$n++; if($n -ge 29 -and $n -le 42){ '"'{0}: {1}' -f "'$n,$_ } }
$n=0; Get-Content -Encoding UTF8 '"'C:/Users/李则徐/.codex/memories/rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md' | ForEach-Object { "'$n++; if($n -ge 23 -and $n -le 25){ '"'{0}: {1}' -f "'$n,$_ } }'
```

**Aggregated Output:**

```text
B16_independent accuracy 0.95320720 min 15
B16_independent nll 0.99359504 min 7
B72_independent accuracy 0.97822239 min 15
B72_independent nll 0.99752090 min 7
B131_independent accuracy 0.98061044 min 15
B131_independent nll 0.99784024 min 7
CTTC_9_sessions_x8_rho_0.00 accuracy 0.97495542 min 7
CTTC_9_sessions_x8_rho_0.00 nll 0.99698430 min 5
CTTC_9_sessions_x8_rho_0.10 accuracy 0.97308175 min 7
CTTC_9_sessions_x8_rho_0.10 nll 0.99655278 min 5
CTTC_9_sessions_x8_rho_0.25 accuracy 0.97009176 min 7
CTTC_9_sessions_x8_rho_0.25 nll 0.99581211 min 5
{"text_only_flip":{"n_blocks":131,"mean":0.0032314550603758673,"block_sd":0.002474158847532716,"se":0.00021616826219131324,"t_crit_95":1.9783804054470222,"lo95":0.0028037920061770387,"hi95":0.003659118114574696},"slot_only_flip":{"n_blocks":131,"mean":0.017239817769974308,"block_sd":0.003977630503890315,"se":0.0003475271907147657,"t_crit_95":1.9783804054470222,"lo95":0.016552276785504166,"hi95":0.01792735875444445},"joint_flip":{"n_blocks":131,"mean":0.018574156272511484,"block_sd":0.0037406160978871415,"se":0.00032681914591355777,"t_crit_95":1.9783804054470222,"lo95":0.01792758367811117,"hi95":0.0192207288669118},"difference_in_differences":{"n_blocks":131,"mean":-0.001897116557838696,"block_sd":0.00309910172739112,"se":0.00027076977512268514,"t_crit_95":1.9783804054470222,"lo95":-0.002432802175328713,"hi95":-0.001361430940348679}}
{"text_only_flip":[-0.004746692029381965,0.016697771412252357,-0.002256714201742788],"slot_only_flip":[-0.0006413112650926305,0.03521399512438194,0.01714676945063362],"joint_flip":[-0.0028979130944165234,0.042272457025742324,0.016347924886208644],"difference_in_differences":[0.0024900902000580725,-0.009639309510891975,0.0014578696373178156]}
29: # Task Group: EM-FloatFormer P4 scientific assessment and reproducible publication
30: 
31: scope: Assess and publish the EM-native Transformer P4 simulation without overstating learned-model performance.
32: applies_to: cwd=C:\Users\������\Documents\ChatGPT\EM-FloatFormer; reuse_rule=Reuse the matched-control, reproducibility, and publication approach for this repository; numerical results, commit, hardware status, and receiver outcomes are P4-specific.
33: 
34: ## Task 1: Complete and assess P4 EM-native Transformer research, success
35: 
36: ### rollout_summary_files
37: 
38: - rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md (cwd=\\?\C:\Users\������\Documents\ChatGPT\EM-FloatFormer, rollout_path=\\?\C:\Users\������\.codex\sessions\2026\11\17\rollout-2026-11-17T20-54-26-01a1eb4a-230f-7e02-b32d-21348996e25b.jsonl, updated_at=2026-11-18T16:17:48+00:00, thread_id=01a1eb4a-230f-7e02-b32d-21348996e25b, formal P4-D and assessment completed)
39: 
40: ### keywords
41: 
42: - EM-FloatFormer, P4, EMFloatFormer2D, C*-equivariance, Hermitian attention, complex RoPE, sparse DMRS, Doppler, CFO, Rapp PA, LDPC, P4D, Holm, sign-flip
23: - Pilot-front-end probe found Gaussian interpolation coherent gains for single-tap delays 0�C4 of approximately `1.000, 0.735, 0.291, 0.062, 0.007`; Fourier interpolation nearly exactly reconstructs static integer-delay responses under the current comb-2 pilot geometry.
24: - Follow-up receiver experiments reduced errors on unseen delay distributions, but the pure Fourier/�� pilot front-end alone achieved nearly all of the gain. Therefore do not attribute this improvement to Transformer attention without matched-front-end controls.
25: 
```

### Event #47: `item.completed` — `command_execution` (`item_44`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "@'


## 5. Power replay and Round 2 lock proposal

The legacy pilot replay gives endpoint minima 131 and 54 blocks for accuracy and NLL, and powers 0.950512 and 0.999906 at 131. Its scenario SDs are 0.094414 and 0.079795. The crossed G1 means are 0.043926 accuracy and 0.035650 NLL. The fitted variance components on the real block-by-seed matrix are as follows.

| Contrast | Block variance | Seed variance | Interaction variance | Seed SD estimate | 95% upper bound on seed SD |
|---|---:|---:|---:|---:|---:|
| Accuracy | 0.0000758565 | 0.0003711426 | 0.0005829523 | 0.01927 | 0.08557 |
| NLL | 0.0001213570 | 0.0004670843 | 0.0003618383 | 0.02161 | 0.09571 |

At the two planning excesses \\(d=(0.03,0.04)\\), one-sided \\(\\alpha=0.05/3\\) and marginal power 0.95, the synthetic fixed-seed block minima for accuracy/NLL are G1 7/5, G2 11/7, G3 8/6, G4 11/4 and H 9/6. These are conditional t minima for hypothetical positive effects. Observed G2 and G4 effects favor zero and fail the positive comparative target. G3's accuracy-only random-seed minimum is 42 under its zero-truncated seed variance estimate from eight blocks. Three seeds do not give a viable 95% crossed NLL power design in any cell under the plug-in scenarios. The small conditional minima do not override the frozen synthetic budget or planning from independent upper variance bounds.

G1-proxy sample sizes and powers follow. CTTC uses session means across eight clusters per session. Its ICC parameter is the assumed fraction of the proxy block variance shared at session level, with no additional session-by-seed effect in this sensitivity run. “Minimum” means the first integer yielding at least 95% marginal power under the specified test and effects.

| Setting | Available independent units | Three fixed seeds, minimum units accuracy/NLL | Three random seeds, power accuracy/NLL | Ten random seeds, minimum units accuracy/NLL | Ten random seeds, power accuracy/NLL |
|---|---:|---|---|---|---|
| Synthetic G1 proxy | 131 blocks | 7 / 5 | .224 / .300 | 15 / 7 | .981 / .998 |
| Pluto proxy | 16 captures | 7 / 5 | .136 / .234 | 15 / 7 | .953 / .994 |
| 72 independent clusters sensitivity | 72 clusters | 7 / 5 | .212 / .293 | 15 / 7 | .978 / .998 |
| CTTC proxy, ICC .10 | 9 sessions | 4 / 3 | .208 / .287 | 7 / 5 | .973 / .997 |
| CTTC proxy, ICC .25 | 9 sessions | 4 / 4 | .202 / .280 | 7 / 5 | .970 / .996 |

At the actual G1 excesses over \\(0.02\\), the three-fixed-seed G1 or Pluto-proxy minima are 10 and 17 blocks/captures, and the CTTC proxy at ICC .10 gives 4 and 6 sessions. For three random seeds at the planning excesses, the ceiling even with known components and infinitely many blocks is 0.71537 and 0.85941. The conservative finite-level test tends to 0.23839 and 0.30978. These ceilings are consequences of the fitted seed variance and fixed effect-size assumptions. The high potential sampling uncertainty in that variance makes the ten-seed numbers planning proxies. Even with ten seeds, the modeled power statements apply only to the two relative endpoints.

The verifier drew 200,000 Monte Carlo replicates per scenario from the independent normal and chi-square sufficient statistics. G1 at 131 blocks gives the following analytic/Monte Carlo boundary rejection rates at nominal \\(0.01667\\).

| Endpoint | Fixed-seed t | Crossed test | Block-only t with random seeds |
|---|---|---|---|
| Accuracy | .01667 / .01654 | .01161 / .01168 | .39171 / .39145 |
| NLL | .01667 / .01716 | .01280 / .01277 | .40810 / .40951 |

For random-seed G1 power the corresponding analytic/Monte Carlo pairs are .22385/.22447 and .30028/.30242. The correlated Holm-IUT MC false-claim family rates range from 0.04002 to 0.04953 under the three simulated null configurations. The proof establishes protection beyond those simulations. ECE and the remaining performance gates need their own pilot variability or a joint design simulation.

G1's same-mother Work F fixed-seed 95% block intervals for the all-rotation interventions are \\([0.002804,0.003659]\\) for text-only flip and \\([0.016552,0.017927]\\) for slot-only flip. The seed-wise slot effects are \\(-0.000641,+0.035214,+0.017147\\). A pooled interval over mothers cannot establish a stable retraining effect.

### Round 2

- **M1** should lock matched-text phase, text-only, joint-blind and independently permuted phase-slot controls, with a signed-slot intervention at fixed text and matched waveform. Cross \\(|\\omega|\\) and accumulated angle independently where the simulator allows it, preserve the analytic S17 and full-IQ anchors, and keep sign/rotation/mother identifiers together for inference.
- **M3** should score the native equivariant forecast against the raw, arithmetic, geometric and canonical-representative paths on a common development split. Calibrate on development data that span the prospective regimes and freeze the choice before blind scoring. Lock all absolute endpoints before comparing phase and zero. The G1 raw and pooled paths fail absolute accuracy, and the blind-fitted paths have below-chance accuracy and negligible NIG.
- **M4** should keep the 131-block synthetic budget until independent gate-endpoint pilots justify changes. If claiming robustness across retrains, plan around ten or more paired seeds as a G1-based starting point and recalculate using pilot upper variance bounds. With three seeds, scope the main claim to the frozen connectors. Obtain separate Pluto capture and CTTC session variance pilots and freeze session IDs and permutation strata. Preserve all nine CTTC sessions in inference, and use the power objective for the complete gate before locking the real-data sample size.

### Reproduction

Run from the repository root with the pinned environment.

~~~powershell
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M2/verify_m2.py
~~~

The run used Python 3.12.7, NumPy 2.3.5 and SciPy 1.17.0. The default uv cache was read-only under the workspace sandbox. The run used an offline cache under M2 populated from the already available pinned wheels. The verifier reads the Stage 3D artifacts and writes its result adjacent to the script.

'@ | Add-Content -LiteralPath 'research/stage4_math_and_design/M2/THEOREM_M2.md' -Encoding UTF8"
```

### Event #48: `item.completed` — `command_execution` (`item_45`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "@'
# Track M2 | Round 1 reply

**G1's locked phase NLL lead is entirely explained by differential scalar-temperature mismatch along the audited decomposition.** The paired \\(L_{\\rm zero}-L_{\\rm phase}\\) is \\(+0.03565000\\) nats. Subtracting each arm's in-cell temperature regret changes it to \\(−0.00089337\\). The exact ordered split is \\(0\\) shared Bayes term, \\(+0.03654337\\) temperature, \\(+0.00012375\\) projection at the fitted temperatures, \\(+0.00000005\\) prior KL and \\(−0.00101717\\) reverse-conditional KL. Locked G1 phase/zero score 23.80%/19.41% accuracy and 1.445229/1.480879 NLL. Both are below uniform in accuracy and above it in loss. S17 direct scores 64.62% and 0.941126. The blind-fitted NLLs are 1.385472 and 1.384578, with ECE 0.025608 and 0.051229.

| Q3 cell | Locked zero-minus-phase NLL | After seedwise \\(T"'^*'"\\) | Independent physical sign MI |
|---|---:|---:|---:|
| G1 | +0.035650 | −0.000893 | 0.363112 |
| G2 | −0.027760 | −0.027716 | 0.176866 |
| G3 | +0.037083 | −0.000430 | 0.344010 |
| G4 | −0.024917 | −0.026351 | 0.179398 |
| H | +0.018090 | −0.003695 | 0.350032 |

Both Q3 arms observe the same signed metadata and codes. Bayes sign MI is available to both. On G2 and G4 zero retains its advantage under optimal scalar scaling, with \\(0.031042\\) and \\(0.030809\\) lower projected reverse-conditional KL. Work F's seed-mean orientation gain is 0.009820 for phase and 0.005048 for zero. Zero retains signed text. On G1 phase, holding the other path matched, flipping text costs 0.003231 nats and flipping slots costs 0.017240, with slot effects \\(−0.000641,+0.035214,+0.017147\\) across seeds. The within-checkpoint slot response does not establish a stable gain across retrains. On Q2 valid, recalibration reduces the phase lead from 0.021807 to 0.001594. The fresh-resample lead reverses from +0.000757 to −0.002672. All seven Q2 controls are in the full report.

**Geometric pooling preserves the failure pattern.** G1 arithmetic/geometric give phase 17.05%/16.36% at NLL 1.411312/1.411586 and zero 16.43%/16.60% at 1.439122/1.445854. G3 also loses accuracy under both pools. G2 and G4 improve accuracy and NLL with both. On complete orbits the exact arithmetic loss gain is label-weighted Jensen. For G1 zero it is 0.04120584 versus 0.03488442 for the requested reverse-KL expression. The latter uses the pooled forecast as the label law.

**Stage 4 gate.** Require absolute accuracy/NLL against uniform, dev prior and matched zero, a preregistered random-label test, and a positive linearized NIG contrast with proposed \\(\\eta_{\\min}=0.10\\). Require dev-only ECE below 0.05, paired Brier improvement of at least 0.005 against the strongest locked baseline, and maximum equivariance/idempotence error at most \\(10"'^{-5}'"\\). Require all endpoints within each claim and apply Holm across separate synthetic, Pluto and CTTC claims. Add a fixed-text slot intervention for claims about signed-slot use.

- **M1** should cross accumulated angle and frequency, with matched text, slots, code, waveform and physical references in the intervention design.
- **M3** should compare native forecasts with arithmetic, geometric and representative readouts on common development data and lock calibration before blind evaluation. G1 fails the absolute gate under the raw and projected scores.
- **M4** should retain 131 synthetic blocks pending full-gate pilots. With G1 variance and 3 pp/0.04 nats true excess above the margins, the historical pair needs 7/5 independent blocks for 95% marginal power conditional on the three fixed seeds. Across retrains, three seeds cap even known-variance power at 71.54%/85.94%, regardless of block count. Ten paired seeds give proxy minima of 15/7 independent blocks or captures, or 7/5 CTTC sessions of eight clusters at proxy session ICC 0.10. At the observed G1 excess, the fixed-seed capture proxy needs 10/17. Treat Pluto's 16 captures and CTTC's nine sessions as separate design populations, measure their variance on independent pilots, and power every gate endpoint before locking real-data enrollment.

The pinned CPU run exited 0. The in-scope Stage 3D summary replay has maximum absolute difference 0. [Theorem and full results](THEOREM_M2.md) · [Verifier](verify_m2.py) · [Numerical results](results_m2.json)

'@ | Set-Content -LiteralPath 'research/stage4_math_and_design/M2/reply_r1.md' -Encoding UTF8"
```

### Event #49: `item.completed` — `agent_message` (`item_46`)

The power replay is complete. With G1 variance and the planned effects, three random seeds limit power even if the test set grows without bound. Ten paired seeds clear the two-endpoint target in the proxy designs. Pluto and CTTC still need separate variance pilots before those counts can be used to lock the full gate.

### Event #50: `item.completed` — `command_execution` (`item_47`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$j=Get-Content -Raw -Encoding UTF8 research/stage4_math_and_design/M2/results_m2.json | ConvertFrom-Json
$j.runtime | ConvertTo-Json -Compress
$j.summary_checks_max_absolute_difference | ConvertTo-Json -Compress
'"'Q3 cells {0}, Q2 controls {1}' -f "'$j.Q3.PSObject.Properties.Count.Count,$j.Q2.controls.PSObject.Properties.Count.Count
rg -n '"'"'^#|In G1|Half'"' research/stage4_math_and_design/M2/THEOREM_M2.md research/stage4_math_and_design/M2/reply_r1.md
git diff --name-only"
```

**Aggregated Output:**

```text
{"python":"3.12.7","numpy":"2.3.5","scipy":"1.17.0"}
{"G1_main_extrap_both":0.0,"G2_interp_both":0.0,"G3_extrap_angle":0.0,"G4_extrap_freq":0.0,"work_f_primary_040":0.0,"work_f_historical_035":0.0,"work_g_primary_040":0.0,"work_a_q2_c4":0.0}
Q3 cells 5, Q2 controls 7
research/stage4_math_and_design/M2/reply_r1.md:1:# Track M2 | Round 1 reply
research/stage4_math_and_design/M2/THEOREM_M2.md:1:# Track M2 | Stage 3D score forensics and the Stage 4 gate
research/stage4_math_and_design/M2/THEOREM_M2.md:5:## 1. Which Bayes risk is being estimated?
research/stage4_math_and_design/M2/THEOREM_M2.md:29:### Scalar calibration and marginal bias
research/stage4_math_and_design/M2/THEOREM_M2.md:57:### The reported additive path
research/stage4_math_and_design/M2/THEOREM_M2.md:71:## 2. Group averaging and the top-1 loss
research/stage4_math_and_design/M2/THEOREM_M2.md:99:## 3. Stage 4 frozen statistical gate
research/stage4_math_and_design/M2/THEOREM_M2.md:107:### Exact type-I control and power under a balanced Gaussian model
research/stage4_math_and_design/M2/THEOREM_M2.md:150:## 4. Replayed empirical results
research/stage4_math_and_design/M2/THEOREM_M2.md:154:### Q3 Bayes terms and the paired decomposition
research/stage4_math_and_design/M2/THEOREM_M2.md:180:### Q3 scalar recalibration
research/stage4_math_and_design/M2/THEOREM_M2.md:197:### Q3 pooling on the saved test design
research/stage4_math_and_design/M2/THEOREM_M2.md:216:### Work F mechanism and averaging convention
research/stage4_math_and_design/M2/THEOREM_M2.md:224:### Q2 controls
research/stage4_math_and_design/M2/THEOREM_M2.md:243:## 5. Power replay and Round 2 lock proposal
research/stage4_math_and_design/M2/THEOREM_M2.md:277:### Round 2
research/stage4_math_and_design/M2/THEOREM_M2.md:283:### Reproduction
```

### Event #51: `item.completed` — `command_execution` (`item_48`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$m2=(Resolve-Path -LiteralPath research/stage4_math_and_design/M2).Path
$cache=(Resolve-Path -LiteralPath (Join-Path $m2 '"'.uv-cache')).Path
\"Expected cache root: "'$(Join-Path $m2 '"'.uv-cache')\"
\"Actual cache root: "'$cache"
Get-ChildItem -LiteralPath $cache -Recurse -Force -Attributes ReparsePoint | Select-Object -First 10 FullName,Attributes
Get-ChildItem -LiteralPath $m2 -Force | Select-Object Name,Length'
```

**Aggregated Output:**

```text
Expected cache root: C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M2\.uv-cache
Actual cache root: C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M2\.uv-cache

Name            Length
----            ------
.uv-cache             
reply_r1.md     4303  
results_m2.json 336609
THEOREM_M2.md   40042 
verify_m2.py    38143 


```

### Event #52: `item.completed` — `command_execution` (`item_49`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'uv cache clean --cache-dir research/stage4_math_and_design/M2/.uv-cache'
```

**Aggregated Output:**

```text
Clearing cache at: research/stage4_math_and_design/M2/.uv-cache
Removed 5161 files (365.0MiB)
```

### Event #53: `item.completed` — `command_execution` (`item_50`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-ChildItem -LiteralPath research/stage4_math_and_design/M2 -Force | Select-Object Name,Length; git ls-files --others --exclude-standard -- research/stage4_math_and_design/M2; git diff --name-only'
```

**Aggregated Output:**

```text

research/stage4_math_and_design/M2/THEOREM_M2.md
research/stage4_math_and_design/M2/reply_r1.md
research/stage4_math_and_design/M2/results_m2.json
research/stage4_math_and_design/M2/verify_m2.py
Name            Length
----            ------
reply_r1.md       4303
results_m2.json 336609
THEOREM_M2.md    40042
verify_m2.py     38143


```

### Event #54: `item.completed` — `agent_message` (`item_51`)

Track M2 Round 1 is complete. The pinned CPU verifier exited 0 and reproduced every in-scope Stage 3D summary value exactly. It analyzed all five Q3 cells and all seven Q2 controls.

**G1’s locked phase lead of +0.035650 nats reverses after scalar recalibration.** The exact zero-minus-phase split is `0` shared Bayes entropy, `+0.03654337` temperature mismatch, `+0.00012375` projection, `+0.00000005` prior KL, and `−0.00101717` reverse-conditional KL. After fitting temperatures on the blind labels, zero leads by `0.000893` nats. Both locked arms score below 25% accuracy, while the S17 direct readout achieves **64.62% accuracy and 0.941126 NLL**.

| Q3 cell | Locked zero-minus-phase NLL | After seedwise scalar fit |
|---|---:|---:|
| G1 | +0.035650 | −0.000893 |
| G2 | −0.027760 | −0.027716 |
| G3 | +0.037083 | −0.000430 |
| G4 | −0.024917 | −0.026351 |
| H | +0.018090 | −0.003695 |

G2 and G4 retain zero’s advantage after recalibration. On G1, the independent physical reference estimates **0.363112 nats** of signed CFO information available to both arms. The zero arm still receives signed `omega` in text. In the phase checkpoint, flipping the slots at fixed text costs `0.017240` nats on average, with substantial variation across seeds. Q2 supplies neither a second sign nor complete rotated forecast orbits, so its decomposition uses the saved Work A references.

**Geometric pooling preserves the G1 accuracy collapse.** Arithmetic and geometric pooling move phase accuracy from **23.80%** to **17.05%** and **16.36%**, while lowering NLL to **1.411312** and **1.411586**. Both pools improve accuracy in G2 and G4 and reduce it in G1 and G3. On invariant orbits, the exact arithmetic log-loss gain is the **label-weighted Jensen gap**. The proposed reverse-KL expression weights the same gap by the pooled forecast and generally gives a different value.

**Round 2.** M1 should lock fixed-text slot interventions and cross frequency with accumulated angle. M3 should compare native and pooled forecasts under dev-only calibration, with absolute NIG, accuracy, Brier, ECE, and structural gates. M4 should retain **131 synthetic blocks** pending full-gate pilots and apply Holm across the synthetic, Pluto, and CTTC claims. Under the G1 variance proxy, three random seeds cap even known-variance power at **71.54% accuracy / 85.94% NLL** for the two planned contrasts. Ten paired seeds give proxy minima of **15/7 independent blocks or captures** or **7/5 CTTC sessions**. Those counts cover the two relative endpoints. The full gate requires separate Pluto and CTTC variance pilots.

[Round 1 reply](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M2/reply_r1.md) · [Theorems and full report](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M2/THEOREM_M2.md) · [CPU verifier](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M2/verify_m2.py) · [Numerical results](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M2/results_m2.json)



### Event #55: `turn.completed`

- Usage: `{"input_tokens": 6901007, "cached_input_tokens": 6764544, "cache_write_input_tokens": 0, "output_tokens": 42086, "reasoning_output_tokens": 6772}`


## Round 1 — Final Output (`reply_r1.md`)

```markdown
Track M2 Round 1 is complete. The pinned CPU verifier exited 0 and reproduced every in-scope Stage 3D summary value exactly. It analyzed all five Q3 cells and all seven Q2 controls.

**G1’s locked phase lead of +0.035650 nats reverses after scalar recalibration.** The exact zero-minus-phase split is `0` shared Bayes entropy, `+0.03654337` temperature mismatch, `+0.00012375` projection, `+0.00000005` prior KL, and `−0.00101717` reverse-conditional KL. After fitting temperatures on the blind labels, zero leads by `0.000893` nats. Both locked arms score below 25% accuracy, while the S17 direct readout achieves **64.62% accuracy and 0.941126 NLL**.

| Q3 cell | Locked zero-minus-phase NLL | After seedwise scalar fit |
|---|---:|---:|
| G1 | +0.035650 | −0.000893 |
| G2 | −0.027760 | −0.027716 |
| G3 | +0.037083 | −0.000430 |
| G4 | −0.024917 | −0.026351 |
| H | +0.018090 | −0.003695 |

G2 and G4 retain zero’s advantage after recalibration. On G1, the independent physical reference estimates **0.363112 nats** of signed CFO information available to both arms. The zero arm still receives signed `omega` in text. In the phase checkpoint, flipping the slots at fixed text costs `0.017240` nats on average, with substantial variation across seeds. Q2 supplies neither a second sign nor complete rotated forecast orbits, so its decomposition uses the saved Work A references.

**Geometric pooling preserves the G1 accuracy collapse.** Arithmetic and geometric pooling move phase accuracy from **23.80%** to **17.05%** and **16.36%**, while lowering NLL to **1.411312** and **1.411586**. Both pools improve accuracy in G2 and G4 and reduce it in G1 and G3. On invariant orbits, the exact arithmetic log-loss gain is the **label-weighted Jensen gap**. The proposed reverse-KL expression weights the same gap by the pooled forecast and generally gives a different value.

**Round 2.** M1 should lock fixed-text slot interventions and cross frequency with accumulated angle. M3 should compare native and pooled forecasts under dev-only calibration, with absolute NIG, accuracy, Brier, ECE, and structural gates. M4 should retain **131 synthetic blocks** pending full-gate pilots and apply Holm across the synthetic, Pluto, and CTTC claims. Under the G1 variance proxy, three random seeds cap even known-variance power at **71.54% accuracy / 85.94% NLL** for the two planned contrasts. Ten paired seeds give proxy minima of **15/7 independent blocks or captures** or **7/5 CTTC sessions**. Those counts cover the two relative endpoints. The full gate requires separate Pluto and CTTC variance pilots.

[Round 1 reply](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M2/reply_r1.md) · [Theorems and full report](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M2/THEOREM_M2.md) · [CPU verifier](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M2/verify_m2.py) · [Numerical results](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M2/results_m2.json)

```


---

## Round 2 — Cross-Review & Convergence Prompt (`prompt_r2.md`)

```markdown
# Track M2 (Round 2): Reconciling Power vs Modal Free-Credit Scope, Causal Arm Gate & Multi-Claim Hierarchy

You are resuming persistent Codex session **Track M2** (`01a21948-8b7b-71d0-a50a-d7c7eaf61018`) in `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`. Keep edits STRICTLY inside `research/stage4_math_and_design/M2/` and execute Python ONLY via `uv run --python 3.12 ...`.

## 1. Critical Cross-Track Findings & Tensions to Resolve (From M1, M3, M4, M5, M6)
1. **The Biggest Statistical-Budget Tension (M2 vs M4 & M6)**:
   - Your Round 1 analysis proved that across **retraining seeds**, 3 random seeds cap even infinite-block crossed random-seed power at `71.54%` (Acc) / `85.94%` (NLL) under G1 proxy variances (with empirical Monte Carlo ~22–30%), requiring **`>= 10` paired seeds** for population-of-seeds inference!
   - Meanwhile, M4 (`$8.44` total on A100-80GB) and M6 (`$9.50` cap against `$12.88` remaining Modal free credits, `Billed Cost: $0.00`) CANNOT afford 10 seeds (`10 × $2.50 = $25+`!). M4 specifically proposed `1` seed (`202611`) mixed synthetic+real fitting (`384` steps, `$2.87`) while M6 proposed a `1`-seed pilot (`<= 173/960` steps, `$0.49`) continuing up to `3` seeds (`$5.10`) ONLY if pilot clears futility!
   - **You must resolve this cleanly**: Define the exact mathematical distinction between **Fixed-Connector / Conditional Seed Claim** (`S=1` or `S=3` frozen seeds, valid conditional $t$-/permutation-inference over independent blind blocks/captures) vs **Random-Seed Retraining Claim** (strictly prohibited below sufficient `S`, or scoped cleanly!). Also address: under M6's 1-seed futility stop ($t=0.18$ on disjoint dev), prove exact conditional Type-I control when the blind test uses disjoint blind blocks/sessions!
2. **Causal Isolation Gate for M1's `Host-Pure` vs `Physical-Residual` Arms + M6 DPI**:
   - M1's Round 1 `B` achieved `64.62%` via a physical S17 head + tiny host residual. Both you and M6 flagged: that doesn't by itself prove frozen Qwen reads coupled evidence!
   - Lock the exact paired contrasts for M1's 4-arm matrix (`Zero`, `Naive-Additive`, `Host-Pure Equivariant`, `Physical+HostResidual`, plus `Direct S17` & `Full IQ`), including:
     - When `Host-Pure` claims LLM phase-coupled extrapolation (`Arm 1` vs `Zero` & `Naive-Additive` + dev-calibrated NIG $\eta_{\min} \ge 0.10$),
     - When `Physical+HostResidual` claims residual value over `Direct Physical` (`Arm 2` vs `Direct Physical` on blind NLL/Acc with matched-text slot interventions!),
     - Pre-registered calibration/temp-fitting scope: separate **dev-locked zero-shot calibration** vs **oracle retrospective sensitivity**, strictly preventing blind recalibration from ever passing a gate!
3. **M3 (`RefSufficientPool`) & M4 (Pluto / CTTC) Real-Data Inference Structure**:
   - M4 locked: Pluto (`2,240` test frames in `16` chronological blocks, `140` stress frames, pilots `15, 18`), CTTC (`1,440` packets across `9` captures × `8` loops = `72` session-loops, `720` Q3 loop-pair groups, held-out `f708_snr10` capture, shared cyclic TX source `[334_096, 455_000)`!), and zero-permutation Q2 `valid / duplicate_raw / valid+indep / fresh` checks.
   - Specify the exact **Holm-Bonferroni / IUT Hierarchy** protecting Family-Wise Error Rate (`alpha = 0.05`) across `(Claim 1: Synthetic Extrapolation + Q2, Claim 2: Pluto, Claim 3: CTTC + held-out f708_snr10)`, handling CTTC's capture-level (`n=9`) vs loop-pair (`n=36`) vs cluster (`n=72`) correlation cleanly (and never claiming `n=72` i.i.d. if source sequence & captures induce session dependence!).
4. **M5's Stage 6 Gate Compatibility**:
   - Include a crisp specification for Stage 6's `max_EIG` vs `{random, nearest, fixed_best_on_dev}` gate (plus `4-bit s17` vs `16-bit canonical` comparison)!

## 2. Specific Round-2 Convergence Tasks for Track M2
Update `research/stage4_math_and_design/M2/THEOREM_M2.md`, `verify_m2.py`, and `results_m2.json`:
1. Formalize the **Scoped Three-Tier Claim Ladder** (Conditional Frozen-Connector Claim `S=1..3` vs Multi-Seed Retrain Claim `S>=10`) + prove conditional validity after M6's Stage-1 dev futility screen + compare power curves for Stage 4's exact `1`-seed and `3`-seed Modal plans on synthetic `32` vs `131` blocks, Pluto `16` blocks, and CTTC `9` captures / `72` clusters.
2. Lock the **Exact Stage 4 Formal Preregistered Statistical Gate Table** unifying M1's `Arm 0/1/2` contrasts, M3's `Q2 two-valid > one-valid > none` & exact `0.0` invariance assertions, M4's Pluto/CTTC real-data hierarchy, and M5's `fixed_best_on_dev` Stage-6 gate.
3. Add a fast deterministic numerical check in `verify_m2.py` verifying: (a) conditional Type-I error under M6's dev-disjoint futility selection, and (b) synthetic `32`-block (M4 budget) vs `131`-block power & Pluto/CTTC conditional fixed-seed power curves (`results_m2.json`). Run via `uv run` and summarize cleanly in `reply_r2.md`.
```


## Round 2 — Resumed Execution & Reasoning Trace (`round2_events.jsonl`)

### Event #1: `thread.started` (thread_id=`01a21948-8b7b-71d0-a50a-d7c7eaf61018`)

### Event #2: `turn.started`

### Event #3: `item.completed` — `agent_message` (`item_0`)

I’ll read the cross-track locks and proposals, then update the M2 gate and verifier. The central question is what the one- and three-seed Modal plans can establish about frozen connectors, and what would require a separate retraining study.

### Event #4: `item.completed` — `web_search` (`exec-d5698c44-a597-48d1-b736-5c3a08f6235e`)

```json
{
  "id": "exec-d5698c44-a597-48d1-b736-5c3a08f6235e",
  "type": "web_search",
  "query": "https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.stats.nct.html",
  "action": {
    "type": "open_page",
    "url": "https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.stats.nct.html"
  },
  "results": [
    {
      "type": "text_result",
      "domain": "docs.scipy.org",
      "ref_id": "turn2view0",
      "snippet": "Total lines: 132",
      "title": "scipy.stats.nct — SciPy v1.17.0 Manual",
      "url": "https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.stats.nct.html"
    },
    {
      "type": "text_result",
      "domain": "www.jstor.org",
      "ref_id": "turn2view1",
      "snippet": "Total lines: 1",
      "title": "A Simple Sequentially Rejective Multiple Test Procedure on JSTOR",
      "url": "https://www.jstor.org/stable/4615733"
    }
  ]
}
```

### Event #5: `item.completed` — `command_execution` (`item_2`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n -i 'Track M2|EM-FloatFormer|Modal' 'C:/Users/李则徐/.codex/memories/MEMORY.md' | Select-Object -First 28; rg -n '"'^def |''^## '"' research/stage4_math_and_design/M2/verify_m2.py research/stage4_math_and_design/M2/THEOREM_M2.md"
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
research/stage4_math_and_design/M2/verify_m2.py:28:def close(actual, expected, label, atol=2e-11):
research/stage4_math_and_design/M2/verify_m2.py:33:def check_tree(actual, expected, label):
research/stage4_math_and_design/M2/verify_m2.py:46:def divide_counts(counts):
research/stage4_math_and_design/M2/verify_m2.py:51:def histogram(index, labels, size):
research/stage4_math_and_design/M2/verify_m2.py:56:def entropy(prob):
research/stage4_math_and_design/M2/verify_m2.py:60:def kl(p, q):
research/stage4_math_and_design/M2/verify_m2.py:64:def metrics(probs, labels):
research/stage4_math_and_design/M2/verify_m2.py:72:def cross_entropy(counts, prob):
research/stage4_math_and_design/M2/verify_m2.py:76:def calibration_fit(raw, counts, dev_beta):
research/stage4_math_and_design/M2/verify_m2.py:109:def fit_temperatures(raw, counts, temps):
research/stage4_math_and_design/M2/verify_m2.py:117:def joint_split(counts, pred):
research/stage4_math_and_design/M2/verify_m2.py:137:def forecast_split(table, counts, kind):
research/stage4_math_and_design/M2/verify_m2.py:156:def rotate_codes(codes, a, b):
research/stage4_math_and_design/M2/verify_m2.py:161:def align(table, a, b):
research/stage4_math_and_design/M2/verify_m2.py:165:def project(table):
research/stage4_math_and_design/M2/verify_m2.py:169:def group_diagnostics(probs, arith, geo, counts):
research/stage4_math_and_design/M2/verify_m2.py:197:def paired_breakdown(raw_hit, other_hit):
research/stage4_math_and_design/M2/verify_m2.py:204:def q3_index(table, codes):
research/stage4_math_and_design/M2/verify_m2.py:208:def posterior_reference(prob, labels, model):
research/stage4_math_and_design/M2/verify_m2.py:220:def q3_analysis(bundle, name, temps):
research/stage4_math_and_design/M2/verify_m2.py:320:def q2_analysis(bundle, temps):
research/stage4_math_and_design/M2/verify_m2.py:383:def work_f(bundle, name, temps, seeds, primary):
research/stage4_math_and_design/M2/verify_m2.py:420:def read_json(path):
research/stage4_math_and_design/M2/verify_m2.py:424:def components(block_seed):
research/stage4_math_and_design/M2/verify_m2.py:443:def fixed_power(d, b, s, v, alpha):
research/stage4_math_and_design/M2/verify_m2.py:450:def chisq_nodes(df, order):
research/stage4_math_and_design/M2/verify_m2.py:455:def crossed_parameters(b, s, v):
research/stage4_math_and_design/M2/verify_m2.py:462:def crossed_power(d, b, s, v, alpha, order=64):
research/stage4_math_and_design/M2/verify_m2.py:470:def min_fixed(d, s, v, alpha, target, max_b=4096):
research/stage4_math_and_design/M2/verify_m2.py:476:def min_crossed(d, s, v, alpha, target, max_b=4096):
research/stage4_math_and_design/M2/verify_m2.py:506:def monte_carlo(v, b, s, d, alpha, rng, reps):
research/stage4_math_and_design/M2/verify_m2.py:536:def multiple_testing_mc(rng, reps, alpha):
research/stage4_math_and_design/M2/verify_m2.py:558:def stage3_lock_check(pilot):
research/stage4_math_and_design/M2/verify_m2.py:578:def power_analysis(matrices, pilot):
research/stage4_math_and_design/M2/verify_m2.py:635:def run():
research/stage4_math_and_design/M2/THEOREM_M2.md:5:## 1. Which Bayes risk is being estimated?
research/stage4_math_and_design/M2/THEOREM_M2.md:71:## 2. Group averaging and the top-1 loss
research/stage4_math_and_design/M2/THEOREM_M2.md:99:## 3. Stage 4 frozen statistical gate
research/stage4_math_and_design/M2/THEOREM_M2.md:150:## 4. Replayed empirical results
research/stage4_math_and_design/M2/THEOREM_M2.md:243:## 5. Power replay and Round 2 lock proposal
```

### Event #6: `item.completed` — `command_execution` (`item_3`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg --files research/stage4_math_and_design; git status --short; Get-Command uv | Select-Object Source; @('AGENTS.md','research/AGENTS.md','research/stage4_math_and_design/AGENTS.md','research/stage4_math_and_design/M2/AGENTS.md') | ForEach-Object { if(Test-Path -LiteralPath "'$_){ Get-Content -LiteralPath $_ } }'
```

**Aggregated Output:**

```text
research/stage4_math_and_design\transcript_utils.py
research/stage4_math_and_design\M2\verify_m2.py
research/stage4_math_and_design\M2\THEOREM_M2.md
research/stage4_math_and_design\M2\results_m2.json
research/stage4_math_and_design\M2\reply_r1.md
research/stage4_math_and_design\M6\verify_m6.py
research/stage4_math_and_design\M6\THEOREM_M6.md
research/stage4_math_and_design\M6\results_m6.json
research/stage4_math_and_design\M6\reply_r1.md
research/stage4_math_and_design\M1\verify_m1.py
research/stage4_math_and_design\M1\THEOREM_M1.md
research/stage4_math_and_design\M1\results_m1.json
research/stage4_math_and_design\M5\verify_m5.py
research/stage4_math_and_design\M5\THEOREM_M5.md
research/stage4_math_and_design\M5\results_m5.json
research/stage4_math_and_design\M5\reply_r1.md
research/stage4_math_and_design\M4\verify_m4.py
research/stage4_math_and_design\M4\THEOREM_M4.md
research/stage4_math_and_design\M4\results_m4.json
research/stage4_math_and_design\M4\reply_r1.md
research/stage4_math_and_design\M3\verify_m3.py
research/stage4_math_and_design\M3\THEOREM_M3.md
research/stage4_math_and_design\M3\results_m3.json
research/stage4_math_and_design\M3\reply_r1.md
research/stage4_math_and_design\discussions\M6\round2_events.jsonl
research/stage4_math_and_design\discussions\M6\round1_events.jsonl
research/stage4_math_and_design\discussions\M6\reply_r1.md
research/stage4_math_and_design\discussions\M6\prompt_r2.md
research/stage4_math_and_design\discussions\M6\prompt_r1.md
research/stage4_math_and_design\discussions\M4\round2_events.jsonl
research/stage4_math_and_design\discussions\M4\round1_events.jsonl
research/stage4_math_and_design\discussions\M4\reply_r1.md
research/stage4_math_and_design\discussions\M4\prompt_r2.md
research/stage4_math_and_design\discussions\M4\prompt_r1.md
research/stage4_math_and_design\discussions\M1\round2_events.jsonl
research/stage4_math_and_design\discussions\M1\round1_events.jsonl
research/stage4_math_and_design\discussions\M5\prompt_r1.md
research/stage4_math_and_design\discussions\M1\reply_r1.md
research/stage4_math_and_design\discussions\M1\prompt_r2.md
research/stage4_math_and_design\discussions\M5\prompt_r2.md
research/stage4_math_and_design\discussions\M1\prompt_r1.md
research/stage4_math_and_design\discussions\M5\round2_events.jsonl
research/stage4_math_and_design\discussions\M5\round1_events.jsonl
research/stage4_math_and_design\discussions\M5\reply_r1.md
research/stage4_math_and_design\discussions\M2\prompt_r2.md
research/stage4_math_and_design\discussions\M2\prompt_r1.md
research/stage4_math_and_design\discussions\M2\reply_r1.md
research/stage4_math_and_design\discussions\M3\reply_r1.md
research/stage4_math_and_design\discussions\M3\prompt_r2.md
research/stage4_math_and_design\discussions\M3\prompt_r1.md
research/stage4_math_and_design\discussions\M3\round1_events.jsonl
research/stage4_math_and_design\discussions\M3\round2_events.jsonl
research/stage4_math_and_design\discussions\M2\round2_events.jsonl
research/stage4_math_and_design\discussions\M2\round1_events.jsonl
?? research/stage4_math_and_design/

Source                  
------                  
C:\qazwsx\Scripts\uv.exe


```

### Event #7: `item.completed` — `command_execution` (`item_4`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$d='"'research/stage4_math_and_design'; rg -n -i 'Arm [012]|Host.Pure|Physical.Residual|naive|same.*text|gate|cross|zero' \""'$d/M1/THEOREM_M1.md" | Select-Object -Last 75; rg -n -i '"'two.valid|one.valid|max.*diff|0\\.0|duplicate|independent|IUT|gate' \""'$d/M3/THEOREM_M3.md" | Select-Object -Last 65'
```

**Aggregated Output:**

```text
31:**Proof.** The encoder takes the sector of the sum of the three samples, falling back to the first nonzero sample on a zero sum. The half-open convention gives \(\operatorname{sector}(i^rT)=\operatorname{sector}(T)+r\) for nonzero \(T\), including boundary rays. Undoing the selected turn gives the same canonical I/Q features after rotation. The fitted MLP and center assignment therefore return the same inner index. The all-zero window is fixed and cannot carry a nontrivial turn. It has probability zero under the stated noise. A hardware interface can mark it as an erasure. [Encoder](../../matched_cache/learned.py)
45:**Proof.** Every code orbit intersects the canonical slice once. Applying equivariance to that representative proves necessity. The shift rule is additive, so \(R_jR_k=R_{j+k}\) proves sufficiency. A deterministic frozen host can be part of \(F\), because its inputs are identical across the orbit. Normalize in the canonical order and gather by integer indices to obtain bitwise equality for identical canonical computations. With an explicit common pilot frame, canonicalize that evidence and use the extended Q2 shift. For Q3 independent, return exact quarters. \(\square\) [Stage 3C canonicalization and readout](../../stage3c/G/quotient.py)
60:Independent increments yield \(U_n(d_1+d_2)=U_n(d_1)U_n(d_2)\) for constant parameters. Diffusion is an irreversible convolution semigroup for \(\sigma_\phi>0\). To undo a known deterministic drift use the conjugate phasor. Diffusion marginalization continues to use damping.
104:The Stage 3D evaluation angles \(0.80\) and \(1.60\) are close to \(\pi/4\) and \(\pi/2\), respectively. G3 has twice the phase variance of G1 at the same angle; G4 has half the phase variance of G2 at its angle. The verifier also varies only \(\theta\), holding \(V=0.04\), SNR and the W1 code mixture fixed. It computes the code-mass-weighted maximum of the convolved probabilities. This isolates the angular decision effect from sampling and variance differences across cells. The absolute accuracy under source uncertainty follows from the mixture and the measured data.
106:The unrotated S17 MAP decision uses \(\arg\max_jA_z(j)\) at every \(\theta\). Its conditional hit rate at \(1.60\) is the corresponding unshifted coordinate of \(q(z;1.60,V)\). A universal below-25% claim is false. A uniform fixed decision is exactly at chance. A fixed code-dependent decision chosen to suit an informative target angle can beat chance there, even as a degree-zero polynomial. Further, the logits \(\log q_k(z;\theta,V)\) are continuous on a compact angle interval for finite SNR. Polynomial logits can approximate them uniformly and yield the same decisions wherever the approximation error is less than half the logit margin. They can also interpolate any finite angle grid exactly. Restricted training gives no assurance that extrapolation finds those coefficients. The chance comparison concerns specific estimators.
118:The release has 2,184,964 trainable connector parameters and frozen Qwen3.5-4B weights. Its pair MLP already couples codes via \(e_1,e_2,e_1-e_2,e_1e_2\). The phase branch maps six slots at \(h,h+\ell,t_p\) to separate tokens. The chat prompt interleaves the embeddings with text, including the numerical condition metadata. The `zero` arm zeros the six slots while retaining public prompt metadata. The signed-CFO train cells have \(\lvert\omega\rvert\le0.25,\ h+\ell\le5,\ \lvert\theta\rvert\le1.25\). The full Stage 3D run also includes zero-CFO replay C1–C3 with its own timings. The CPU surrogate uses only the ten signed-CFO training settings. [Connector and prefix assembly](../../stage3d_qwen_signed_phase/qwen_connector.py) · [Dataset](../../stage3d_qwen_signed_phase/data_stage3d.py) · [Training construction](../../stage3d_qwen_signed_phase/train_eval_stage3d.py)
120:| Blind cell | Cumulative angle magnitude | Qwen phase acc / NLL | Qwen zero acc / NLL | Direct S17 acc / NLL |
127:Work F gives orientation gain \(+0.00982\) and mean swapped-minus-matched NLL \(+0.01837\) across phase seeds. The phase seed-averaged table has mean-square metadata effect \(1.85378\cdot10^{-4}\), waveform effect \(1.11977\cdot10^{-3}\), and interaction \(6.36253\cdot10^{-5}\). The physical calibration has metadata effect \(0.201603\), over a thousand times the host's mean-square value. Work G has rot01 and rot11 mean maximum coordinate differences of \(0.05914\) and \(0.12627\), with worst tested discrepancy \(0.48167\) at rot01. These measurements establish weak response in the correct signed direction and broken quotient equivariance for the trained host. [Stage 3D summary](../../stage3d_qwen_signed_phase/results/summary_master.json)
131:Suppose centered logits satisfy \(L(z,\phi)=U(z)+V(\phi)\). The signed phase contrast across codes has rank at most one and the mixed finite difference in \((z,\phi)\) vanishes. If the same readout is exactly \(H\)-equivariant for a task with a surjective output shift, the contrast is zero.
133:**Proof.** Every row of \(L(z,\phi_+)-L(z,\phi_-)\) is the same vector \(V(\phi_+)-V(\phi_-)\). The mixed difference subtracts two identical rows. Equivariance under all four rolls then makes the common row invariant, hence constant across classes. Centering removes that constant. Multiplicative transport has code-dependent signed contrasts when the states carry differing moments affected by phase. \(\square\)
139:For code pair \(z\), use \(D_z=p_+(z)-p_-(z)\) from `cal_ref` and \(w_z=\min(m_+(z),m_-(z))/\sum_{z'}\min(m_+(z'),m_-(z'))\) from the archived conditional code masses. Compute \(M=\operatorname{diag}(\sqrt w)D\) and the Gram spectrum of \(M^\mathsf TM\). The row sum is zero. In a cyclic orbit of a row, a nonzero frequency-1 component spans a real plane and a nonzero frequency-2 component adds the Nyquist line. The verifier reconstructs the half-count ratios and finds both on the positive-weight orbit of code pair index 0: rows \(0,64,128,192\). Thus the **archived calibration table has rank three**. This is a statement about the finite reference table, with its calibration sampling error.
141:| Contrast | Nonzero Gram eigenvalues | Frobenius norm | Stable rank | Effective rank |
158:Top-1 accuracy can fall on a complete orbit. Use aligned target 0. At three quarters of the orientations, assign the correctly classified vector \((0.35,0.34,0.16,0.15)\), and at the other quarter assign \((0.01,0.97,0.01,0.01)\). The average is \((0.265,0.4975,0.1225,0.115)\). Accuracy drops from \(75\%\) to zero while NLL drops from \(1.93866\) to \(1.32803\). Work G reproduces these metric directions on its three sampled rotations: NLL \(1.44523\to1.41131\), and accuracy \(23.80\%\to17.05\%\). The paired 95% interval for \(P f\) minus raw accuracy is \([-7.09,-6.41]\) percentage points. The verifier replays the calculation from the released raw score tables and locked temperatures. Symmetry projection alone does not restore the signed physical decision. [Stage 3D projection and metrics](../../stage3d_qwen_signed_phase/recompute_all.py)
163:2. **Couple evidence and phase.** Start with canonical source distributions, initializing from the frozen S17 table. Propagate suitable circular moments with \(U_n(d)\) on each edge. For instance, transport \(m_{1,n}\) forward to the repeat target and de-rotate the second-window evidence with the known signed mean, while marginalizing uncertainty with nonnegative diffusion. Feed the real and imaginary components, source probabilities, and physical head features into the **same** trainable token projector. For exact continuous likelihood work, integrate a positive density or use Fourier harmonics appropriate to the sector and pilot maps. A first-moment feature alone cannot preserve a multimodal likelihood.
166:5. **Normalize once and lift.** Gather canonical probabilities into the physical choice order. For bitwise table parity, evaluate each distinct representative once at fixed context and reuse its canonical vector. Independently evaluated FP32 rows may differ by roundoff due to batch shape or kernels. Apply the same construction for candidate logits when used. Unrestricted free-text generation is outside the guarantee.
173:- Return `valid_mask[B]`, `pilot_frame_turn[B]` (zero under the fixed external frame), and `log_likelihood(phase_nodes, channel_turns) -> [...,4]` or equivalent factors grouped by timestamp. Equal-time independent AWGN pilots with known roots \(d_i\) reduce to \(S_p=\sum_i \rho_i T_{p,i}\overline{d_i}\) and \(\log L_p(g,\phi)=2\Re[S_p i^{-g}e^{-i\phi}]+C\). Retain relative likelihood scales across phase nodes and channel turns. Differently timed pilots use a shared Brownian chain in chronological order, which handles \(t_p>t_3\) without assigning a winding number to a wrapped observation.
179:- Supply an optional posterior on shared CFO and its estimate age or covariance if available. For independent Gaussian rate uncertainty of variance \(v_\omega\), the single-edge attenuation includes \(e^{-n^2v_\omega d^2/2}\). The same uncertainty is correlated across edges. Keep a shared latent or a joint law for multi-window and pilot inference. A rate uncertainty uses \(d^2\), whereas Brownian phase variance uses \(d\).
201:Each B seed exceeds 55% on G1 and G3. B has zero absolute probability defect when reusing canonical representatives, and at most \(5.96\cdot10^{-8}\) on ungrouped batched evaluation. The five task lifts and joint Q2 coordinate check all have zero defect. A breaks equivariance by up to \(0.77851\) but exceeds chance on G1 and G3. The requested A collapse is therefore unreproduced on this surrogate. The Stage 3D Qwen run and the unrotated S17 MAP control both show below-chance accuracy in those cells. B's top-1 parity with the physical-head-only oracle leaves the contribution of the frozen host unresolved.
205:- **M4** · Which CFO sign, time base, and residual convention does the hardware maintain, and can it provide uncertainty for the same CFO latent across all edges?
5:The C4 generator sets the continuous drift to zero at the first window. It draws independent Gaussian increments with mean \(\omega\,dt\) and variance \(\sigma_\phi^2dt\) on the grid \(\{0,h,h+\ell,t_p\}\). Its two valid pilots share the same physical phase and have independent complex AWGN. At \(h=\ell=2,\ t_p=4\), the pilot clock equals the withheld-window clock. Each generator call takes one scalar \(t_p\). The schema can carry a different timestamp on each reference, but the seven C4 arms never exercise this case. The released C4 reader accepts only an ID, a channel Boolean and a matched sum. Its fitted encoder emissions and quadrature are tied to \(t_p=4,\ \omega=+0.35,\ \sigma_\phi=0.10,\ \mathrm{SNR}=2\,\mathrm{dB}\). [Generator and arms](../../stage3d_qwen_signed_phase/data_stage3d.py) · [Schema](../../stage3_ministral_cache/schema.py) · [Reader](../../stage3c/A/q2_reader.py) · [Stage 3C derivation](../../stage3c/A/Q2.md)
7:A copy has the same acquisition ID and samples. A fresh acquisition has a new ID and independent noise conditional on the shared path. The schema checks duplicate channel flags, timestamps within \(10^{-12}\), and exact sample equality. The weighted pool also checks amplitude and noise variance, which affect the likelihood. Stage 3D's distractor has a separate random quadrant, phase, and noise. Its density cancels when calculating the target posterior. These results condition on public, exogenous acquisition design and correct channel declarations.
13:Write \(b=\pi/2\), \(\Theta_t=\varphi_0+\omega t+B_t\), \(B_t=\sigma_\phi W_t\). The physical carrier angle is \(bG+\Theta_t\). C4 takes \(\varphi_0=0\) in its chosen frame. With independent uniform \(X_1,X_2,G\), the variables \(U=X_1+G,\ V=X_2+G,\ G\) are independent and uniform modulo four. For \(n_m\) pilot samples of per-sample amplitude \(A_m>0\), define
23:Deduplicate by ID and retain only valid acquisitions. Expanding the density at a common time \(p\) gives
37:For S17 at the pinned initial phase, the circulant mapping is invertible in the underlying Gaussian model. In the first-window turn-zero region, \(\Re T>|\Im T|\) almost surely. On any code cell of positive probability within that region, the density for \(U=0\) exceeds the density for \(U=2\) pointwise because \(\Re T>0\). Also, the sum of densities for \(U=0,2\) exceeds the sum for \(U=1,3\) by the strict increase of \(\cosh\) in \(|\Re T|\) versus \(|\Im T|\). Integrate over the code cell to obtain the nonzero first and second harmonics of \(a_{z_1}\); the third is the conjugate of the first. The other turns just permute the same reasoning. Thus the normalized carrier posterior is itself a minimal Q2 statistic. The argument identifies that statistic even in the Brownian case, without using a global injectivity assumption for \(S\mapsto (L_g/\sum L_g)_g\). The first phase moment is nonzero for at least one \(z_2\) since its unconditional value has magnitude \(e^{-\sigma_\phi^2p/2}\).
83:For positive distinct clocks put \(C_{jk}=\sigma_\phi^2\min(t_j,t_k)\), \(d_j=t_j-t_{j-1}\), \(t_0=v_0=0\). Independent innovations give the full-rank density
101:Use the Q2 convolution above. Gauss–Hermite quadrature in independent innovations evaluates this integral for short histories. Brownian independent increments also permit forward/backward integration with wrapped Gaussian transitions on the circle. A Fourier calculation integrates each multidimensional harmonic with multiplier \(e^{-n^\top C n/2}\), augmenting the path with \(h\) if needed. Finite grids, finite Fourier expansions, quadrature, and estimation of encoder emissions introduce distinct errors.
103:**Minimality for the physical likelihood.** A parameter-independent density ratio implies \(\sum_j\Re[(D_j-D'_j)e^{-i(bg+v_j)}]\) is constant. The phase vector has full support, so vary each \(v_j\) independently to force \(D_j=D'_j\). A deterministic zero-time node is resolved by comparing the four \(g\) values. At zero diffusion all residual phases coincide and the derotated sum is sufficient.
132:Normalize over \(k,q\). Summing over \(q\) gives Q2 and summing over \(k\) gives Q3_after_pilot. The second-window emission and third-window sector emission generally require the same joint bridge. The carrier-only vector \(L_g\) need not retain the pilot update of \(B_\tau\). Coincident pilot and prediction clocks share the latent phase, with independent observation noises. In a causal receiver, only pilots available by the prediction deadline enter the calculation.
144:| duplicate copy, dedup | 66.41 / 0.8192 | 28.78 / 1.3889 |
145:| duplicate copy, raw | 66.41 / 0.8192 | 26.30 / 1.3944 |
146:| independent only | 25.00 / 1.3863 | 24.80 / 1.4077 |
147:| valid + independent | 66.41 / 0.8192 | 24.54 / 1.4082 |
150:Relative to valid, raw copy costs 2.47 pp, the distractor costs 4.23 pp, and fresh costs 2.34 pp. Appending the distractor changes the phase host posterior by 0.04030 mean total variation and adds 0.01935 nats. A fresh pilot changes it by 0.03727 mean TV and adds 0.01676 nats. The zero host drops from 26.63% to 25.78% on fresh. Work A gains 40 correct decisions net and reduces NLL by 0.15602 on fresh. The verification reconstructs the raw duplicate arm with the physical reader, since the archive has six physical arms for seven host arms. [Results](results_m3.json) · [Archived scoring rules](../../stage3d_qwen_signed_phase/recompute_all.py)
154:**Coordinates and filtering.** `build_ref_feature` includes \((\Re T,\Im T,|T|,\Re y_0,\Im y_0,\ldots,\Re y_2,\Im y_2,c,t_p,\mathrm{SNR}_{dB}/10)\). Four global I/Q pairs rotate together. Magnitude and metadata are invariant. `ref_mlp` and window embeddings lack quarter-turn tying, the explicit multiplication by \(i^{-\mathrm{turn}(z_1)}e^{-i\omega t_p}\), and numeric pooling. The learned mapping therefore relies on cross-token interactions for these operations, with no guarantee that it recovers them. The saved scores do not establish an expressivity impossibility. Deduplication is prompt/feature-list filtering when enabled. The raw duplicate bypasses it. Independent-channel embeddings reach the host along with their status tags, so the flag alone supplies no zero-influence guarantee. [Reference features](../../stage3_ministral_cache/ministral_connector.py) · [Projection](../../stage3d_qwen_signed_phase/qwen_connector.py)
156:The archived first valid matched sums have RMS magnitude 1.27735. Their global I/Q pair moves by RMS 1.80644 under a joint quarter-turn because \(|iT-T|^2=2|T|^2\). The canonicalized pool had zero movement for that action. The normalized absolute cross moments of the raw matched sums are 0.62189 for the two valid pilots and 0.00420 for valid and independent, consistent with the generator's common signal and separately drawn distractor. [Result receipt](results_m3.json)
160:1. **Validate and canonicalize.** Use \(z_i=4a_i+u_i\). Resolve IDs and check provenance, calibration and exact sample equality before filtering. A production clock should publish one canonical timestamp per physical tick. For schema-tolerance aliases, the prototype sorts timestamps per ID, chooses the minimum, and rejects a spread beyond \(10^{-12}\). Remove independent-channel references before feature evaluation. A conflict-free distractor contributes no activation or gradient. A literal copy of a canonical acquisition ID makes no change. Sort unique IDs before reduction for deterministic sums.
167:**Blackwell.** Dropping the new acquisition maps the augmented experiment to the old experiment. There is an explicit garbling even after same-clock pooling. Given the augmented pooled sum \(S_+\) with total precision \(P_+\), the old sum of precision \(P_o\) has conditional law \(\mathcal{CN}((P_o/P_+)S_+,\,P_o(1-P_o/P_+))\), independent of the physical phase. This follows from joint Gaussian receiver noise with covariance \(P_o\) between the two sums. Across clocks, use this garbling within each clock and drop new-only clocks. An exact posterior preserves all decisions about its target, so the Bayes risk of the augmented posterior experiment improves weakly for every loss. The expected log-loss gain is \(I(X_1;T_{\mathrm{new}}\mid Z_1,Z_2,\mathcal R_{\mathrm{old}})\). For the finite-time Brownian Q2 model with informative \(a_{z_1}\) and a fresh finite positive-SNR pilot this is strict. At zero old evidence, the new pilot derivative of the first target moment is nonzero for some \(z_2\), since the unconditional \(e^{iB_{\mathrm{new}}}\) moment is positive. Continuity and positive observation densities give a positive-measure change in the posterior. This applies to the first pilot and subsequent ones at common or distinct clocks. A copy, independent distractor or zero-SNR acquisition gives equality. [Blackwell, *Equivalent Comparisons of Experiments*](https://doi.org/10.1214/aoms/1177729032)
181:The script replays the archived physical and host probabilities. A fixed-slot CPU surrogate fits \(a_{z_1}\) on 32,768 distinct `NS_CFO_TRAIN` C4 mothers with quarter-turn tying and a symmetric half-count prior. Its second-window phase emission comes from separate Stage 3C-style importance calibration. The 512 blind labels enter only scoring. The multi-clock experiment uses 16,384 new paired mothers at \(t_p=(1,7)\), \(h=2\), \(\omega=+0.35\) and 2 dB, with common draws across methods and diffusion settings. It compares the full Brownian model, the optimal coarsened-data posterior under the same model, and a coherent common-phase approximation. The first low-diffusion observed-label result was inconclusive. A single independent follow-up at 524,288 mothers resolved that contrast, with the initial result retained. Conditional model NLL in the receipt averages over the full fitted posterior for each observation and complements the independently observed-label NLL. The main experiment uses order 40 per pilot innovation, checks order 64 on the 128 highest-pilot-magnitude cases, and takes the second-window emission from 513 periodic knots with order 20 for the inner bridge. The follow-up uses order 12 per innovation and checks against order 40 on 128 high-magnitude cases. [Script](verify_m3.py) · [Result receipt](results_m3.json) · [Benchmarks and Round 2 questions](reply_r1.md)
```

### Event #8: `item.completed` — `command_execution` (`item_5`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$d='"'research/stage4_math_and_design'; rg -n -i 'f708|334.?096|455.?000|2,240|1440|1,440|140|pair|pilot|384|8\\.44|202611|budget' \""'$d/M4/THEOREM_M4.md" | Select-Object -Last 78; rg -n -i '"'0\\.18|173|dev|9\\.50|12\\.88|5\\.10|0\\.49|DPI|futility|scope' \""'$d/M6/THEOREM_M6.md" | Select-Object -Last 75'
```

**Aggregated Output:**

```text
3:Round 1. All logarithms and information quantities use nats. The proposed primary input is a causal, gain-normalized scalar lifted into the frozen `s17` three-tap space at its **physical SNR**. The main real-data contrast is the effect of releasing one or two genuine pilot observations onto a cache whose first two window codes are already fixed.
5:The research plan calls for a paired prediction and reveal on actual recordings. Stage 3D defines Q1 as transmitted relative quarter-turn, Q2 as transmitted symbol 1, and Q3 as the received sector of an unrevealed retransmission. The Stage 3D training support reached `|ω(h+ℓ)| = 1.25 rad`. Its physical reader benefited from a second valid pilot while the frozen Qwen connector did not show the same improvement. Freeze the Stage 3D Qwen revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, its tokenization and host fingerprint, and `s17` SHA-256 `e67d9fbafb90791ab19549de7789798d818286d7f52d7ac566cc39df71e78356`. Fit the four connectors with matched budgets and freeze them before scoring. [Research plan](../../../docs/LONG_TERM_RESEARCH_PLAN.md) · [Stage 3D protocol lock](../../stage3d_qwen_signed_phase/PROTOCOL_LOCK.json) · [Stage 3D report](../../stage3d_qwen_signed_phase/REPORT.md)
11:Write \(e=\mathbf1_3/\sqrt3\), \(V_0=10^{-0.2}=0.63095734448\), \(S\in\{1,i,-1,-i\}\), and use proper complex noise with \(\mathbb E WW^*=K\) and real per-coordinate variance \(V_0/2\) under the nominal law. `s17` was trained with \(Y_0=eS+W_0,\ W_0\sim\mathcal{CN}_3(0,V_0I)\). The whole window has unit signal energy. Its matched statistic is \(T_0=e^*Y_0\). Each window produces 16 codes, four quadrant turns times four local codes. The frozen lookup table differs from the MLP softmax used to choose the local code. [Frozen encoder](../../matched_cache/learned.py) · [Training waveform law](../../fourphase_paired_questions/paired.py) · [Stage 3 matched sum](../../stage3_ministral_cache/schema.py)
98:Rayleigh denominator with \(\mathbb E G^2=1\) is \(\nu=1\). In one complex dimension \(P(|X|^2/\Sigma>q)=(1+q)^{-1}\), so \(\mathbb E|X|^2=\infty\), and forward KL to any nondegenerate Gaussian is infinite. The normalized Student entropy is finite here. More generally an infinite second moment gives infinite KL to a Gaussian by testing bounded truncations of a sufficiently small positive multiple of \(\|X\|^2\) in the variational KL lower bound. If the amplitude estimator itself has a folded Student law, substitute its density \(2f_{t_d}(g/b)/b\) for \(g>0\) in (6), where \(f_{t_d}\) is the standard real Student density and \(b\) its scale. That exact integral is a different scale mixture, with infinite second moment since the folded density is positive at zero. A noisy complex pilot LS amplitude \(|h+E|/a\) with nondegenerate proper Gaussian \(E\) is conditionally Rician. Its radial density has positive linear coefficient at zero, so \(\mathbb E G^{-2}=\infty\) even for nonzero \(h\). If `a` itself is Rayleigh but estimated exactly and both amplitude and noise scale with it, \(G=1\) and the heavy tail argument does not apply. If a new thermal sample and the scale estimator share noise, the exact law uses \(W\mid G\) in (6) and cannot be inferred from the gain histogram alone.
100:AGC usually divides the signal too. For \(T=(\mu_s+W)/G\), the exact class-conditional density replaces \(G^2x^*\Sigma^{-1}x\) in (6) with \((Gx-\mu_s)^*\Sigma^{-1}(Gx-\mu_s)\). It is a noncentral scale mixture and generally has no centered Student formula. Absolute Rayleigh fading and fixed receiver noise can also produce a heavy-tailed unconditional noise term even with exact gain estimation, since averaging \(W/a\) over \(a\) invokes the Rayleigh case. A causal guard \(\widehat a\leftarrow\max(\widehat a,a_{\min})\), with floor fixed from training/calibration and carried in the uncertainty flags, makes \(\mathbb E G^{-2}\) finite conditional on \(a\). Fit variance floors, clip estimator outliers as registered on dev, and model or report flags. When estimating a noise scale with \(\nu\) complex residual degrees of freedom independent of the new observation and a correctly specified proper Gaussian linear fit, the normalized residual power estimate has Gamma shape \(\nu\). In a rank-\(p\) complex fit to \(n\) such pilots, \(\nu=n-p\) follows from orthogonal residual projection. Estimated CFO, branch selection, regularization, symbol decisions, correlated pilots, and unknown DC break this pivot unless accounted for. Use a robust likelihood or (4) with measured nuisance uncertainty in those cases.
110:For `s17`, along a fixed nonzero canonical ray \(x=cu\), \(c>0\), the sum sector is scale invariant. At low gain \(\tanh(cuW_1+b_1)\to\tanh b_1\), so the local code becomes constant if the winning cross-entropy score has a positive margin. The hidden deviation is at most \(c\|uW_1\|\). Converting it through \(W_2\) and the \(1/2\)-Lipschitz softmax, the bound on pairwise score changes is \((c/2)\|uW_1\|\|W_2\|_{2\to2}\|\log C_i-\log C_j\|\) for centers \(C_i,C_j\). A strict margin greater than this perturbation proves stability. For high gain, neuron \(j\) with \(d_j=(uW_1)_j\ne0\) approaches \(\operatorname{sgn}(d_j)\) with error at most \(2e^{-2(c|d_j|-|b_{1j}|)}\) once \(c|d_j|>|b_{1j}|\); neurons with zero projection stay at \(\tanh b_{1j}\). Apply the same score-margin test to the limiting logits and center choices. This proves radial saturation on each interior ray with strict margin. The four turns can remain informative. At exact all-zero input the encoder chooses its defined tie behavior. `verify_m4.py` finds `[1,5,9,13]` in the low-gain limit and `[0,4,8,12]` on its high-gain ray. Ties, near-orthogonal directions, and random scales near zero preclude a global AGC threshold.
120:| Pluto 850 MHz | One row per frame. Copy 1, \(k_1=30+(f\bmod67)\); \(k_2=k_1+h_{12}\) with \(h_{12}=1\) or 8 using `(f + floor(f/16)) mod 2`. | Two acquisitions at logical indices `15` and `18` of copy 1. Initial state uses sync `0:13` and split-local past, with those two pilot reads excluded from all pre-reveal quantities. | Same payload slot \(k_1\) in the immediately following complete copy, \(L=126\), \(\ell=126-h_{12}\). Score 2,240 candidates with a common lock-failure policy. A copy is a real retransmission; no target-label matching or synthetic replacement. |
121:| CTTC 433/708/2450 MHz | One row per packet with ordinal \(j\) and loop \(b\). \(k_1=136+((17j+5b)\bmod48)\); \(k_2=k_1+h_{12}\), using 1 for even \(j+b\), 8 otherwise. | `126,127` of the same packet, both earlier than \(k_1\). Initial state uses `0:6` sync and authorized previous data, excluding both release samples. Estimate their joint covariance. `254,255` remain unavailable until they have physically arrived. | On each capture use disjoint loop pairs `(0,1),(2,3),(4,5),(6,7)` at the same packet source start and same \(k_1\) selected in the earlier loop. \(L=500000\), \(\ell=500000-h_{12}\). Q1/Q2 cover 1,440 packets. Q3 has 720 retransmission candidates and 36 loop-pair groups. |
125:The primary samples retain actual fading, residual DC/ISI, noise, clock offset, and any changes present at registered times. A naturally occurring jump is a stratum, with changes flagged by a predeclared causal split-local pilot innovation criterion and a retrospective scorer-only morphology check. A CPU-only cross-frame Q1 diagnostic pairs `(f−1,f)` at `f mod 20 = 0` within each test block, using the earlier frame's \(k_1\) and current frame's \(k_1\) from the table. Its matched control pairs the preceding two interior frames `(f−2,f−1)` in that same block. Use only available sync and authorized pilots and a gap-class prior when the time between Pluto columns is unavailable. This samples recorded batch boundaries by schedule without requiring or asserting a jump at each boundary. Separate source domains and nominal session strata in analysis, keeping capture IDs out of model text/features. Hold `f708_snr10` out of fitted connector weights, temperatures and shared prior tuning to expose one capture transfer; allow its own causal sync and released pilots. Pluto is a blocked same-campaign test. CTTC has one public cyclic TX sequence; input-source disjointness and capture dependence are distinct issues. Use block-paired NLL/SER differences on 16 Pluto groups and 72 CTTC session-loops, group adjacent repeat pairs for CTTC Q3, and report a capture-level sensitivity analysis and source-ordinal sensitivity because loops/captures reuse source positions. Report all registered rows and lock coverage, with the same uniform predictive fallback on failed rows and a secondary locked-only analysis. Avoid treating all packets as independent. Select the Q2 `two−one` proper-loss contrast and large-angle synthetic Q3 coupled-vs-additive contrast before test, with paired intervals and the multiplicity rule frozen with M2.
127:Keep the natural recordings as the primary results. Recreate seven Pluto stress recipes on 140 frame IDs `range(4800,7040,16)` each, and all 1,080 CTTC stress IDs from the branch identity rule (`i mod 4 != 3`, round-robin modes). Score a single two-reference Q2 vector per registered stress row in the GPU budget. For perturbation strength use frozen train-scale or authorized pre-cutoff sync/pilot-only power. Pluto’s legacy injection measures power across the full frame, and CTTC’s legacy injection uses `offline_sig_pow` derived with full TX truth. Their historical raw scores stay available as historical comparators. Hide exact injected CFO, noise power and jumps in the scoring vault and keep RNG independent of payload labels. Real-only CFO uncertainty is estimated from observed pilot evidence. [Pluto data and acquisition](https://data.mendeley.com/datasets/k72b4pnc5g/1) · [CTTC data and acquisition](https://zenodo.org/records/4895133)
131:**Split proof.** Pluto uses frame intervals `[0,3200)`, `[3300,4500)`, `[4600,7040)` including the first 200 warmup frames of each split. The 100-frame gaps imply pairwise empty intersections. All copies of a frame are inseparable across splits. CTTC uses six disjoint cyclic source ranges of length-500,000 circumference, including `train target [0,160000)`, `train ref [164096,200000)`, `dev target [204096,290000)`, `dev ref [294096,330000)`, `test target [334096,455000)`, `test ref [459096,495904)`. The boundary gaps are at least 4,096 symbols, including the wrap. The verifier regenerates starts `lo + i floor((hi−lo−256)/n) + 128`, enumerates both bit pairings' union and asserts disjoint bit footprints across split and role. On the archived nine anchors the 5,472 target/reference acquisitions fit inside their files, have distinct IDs, and the nearest receive ranges remain 16,481 samples apart after 255-sample RRC support expansion. This verifies the stored geometry, including 72 test groups. Acquisition edges must be rechecked on fresh causal alignment. Source sequence reuse within a split creates statistical dependence; the proof addresses explicit identity and support leakage.
133:**Filtration proof.** The released model payload comprises timestamped W1/W2 entries and a causal state with nuisance parameters estimated only from authenticated training/allowed sync and prior pilots. Each state node records raw-support intervals, truth whitelist, availability time and acquisition IDs of its parents. Acquisition truth is delivered as individual permitted known symbols; the feature worker has no access to `Tx_data` payload or the full CTTC source sequence. Build an independent scorer vault containing all targets, withheld probes, archival oracle fields, future IQ and injected parameters. Model projections strip frame, capture, packet and source ordinals, plus provenance fields that could identify cyclic TX labels. Prompt content carries lags, units and predeclared uncertainty summaries. Auditable salted observation IDs stay outside the prompt and are used for exact de-duplication. Denote the allowed baseline information by \(\mathcal F^0\). Its closure under deterministic functions and independent lift randomness contains the committed cache. Updating only by the union with released past pilot observations yields \(\mathcal F^r=\mathcal F^0\vee\sigma(R_1,\ldots,R_r)\). A listed input lies in this filtration by induction on the provenance graph. Intersection checks of all authorized raw read supports with actual withheld-symbol read stencils, and the inequality on every raw cutoff, certify direct observation masking once the time geometry is established. The verifier checks disjoint logical indices and local cubic-stencil examples over the registered ranges. Adjacent samples can retain analog contributions through real ISI, matched filtering or DC memory. Physical dependence is part of the channel model and cannot be excluded by an index proof.
137:Use causal sync and then revealed pilots for residual CFO. For synchronized observations \(y_p\) and authorized \(s_p\), fit \((H,d,\omega)\) using the pilot-only profile loss
151:| **Frozen Qwen Zero** | Same host, W1/W2 codes, admissible public metadata, identical reference observations and reveal schedule. Set phase-coupling numeric slots to zero. Fit the zero control connector with the same steps and optimization budget. |
153:| **Stage4-Equivariant-Coupled** | Canonicalize the received gauge from the turn in the first valid cached code. Rotate both codes and receiver-coordinate pilot evidence by that gauge, propagate the signed phase law with explicit lag, then map the answer vector back with its registered group action. Under a common receive-chain quarter-turn, Q1 and Q2 TX labels stay fixed and Q3 receiver sectors permute. If testing a TX relabeling, known pilot labels and Q2 labels transform together in a separately specified action. A single canonical host score suffices. Register validity before projection and audit an exact-zero or missing-pivot policy. |
154:| **Stage4+RefSufficientPool** | Replace ordered raw-reference tokens by a set likelihood summary on the same pilot observations, with release times and known complex TX values. Deduplicate exact acquisition copies before pooling. With joint proper Gaussian \(y=A_mH+n,\ n\sim\mathcal{CN}(0,K_m)\), sufficient for \(H\) at fixed \(m\) are \(A_m^*K_m^{-1}y\) and \(A_m^*K_m^{-1}A_m\). For comparing candidate \(m\) also retain \(y^*K_m^{-1}y\), normalizers (e.g. \(\log\det K_m\)), and weights/times. Complete the square to see all remaining terms are constant in \(H\). For overlapping/colored blocks use joint covariance or unique innovations. For time-varying states use the corresponding transition and filter state or a bank of likelihood factors. A single summed phasor has no general sufficiency under unknown CFO, shared noise or jumps. |
156:No legitimate arm counts the same observation twice. If shared state already conditions on a pilot, score the next observation using its conditional innovation given that state. If causal metadata contains the same estimates, the model receives them as a representation of evidence already used. The Q2 none/one/two protocol above keeps reveal pilots out of the baseline fit. Identity invariance under duplicate copy can be checked via byte-equality before scoring within existing row budget. A full independent-reference mismatch or separate multi-seed host reruns need additional evaluation rows. CPU bootstraps and true-likelihood integration are independent of host training.
170:Use one CPU ingestion function, with a content-addressed `raw/model` cache committed before scheduling dependents. Reuse a verified snapshot of the historical Qwen revision when possible, or fetch only that pinned revision. A second CPU function streams Pluto frames and CTTC `file_2.dat` memmaps into train/dev/test feature shards. It computes split-local frontend state, per-stage pilot inputs and CPU physical baselines. Use separate reader mounts for scorer-only material and GPU-visible caches. Store frozen host scores, checkpoint and shard manifests on a results Volume, and export compact paired rows for independent CPU replay. Stage default time caps cover image/container boot and CPU dispatch overhead via the explicit reserve lines. Use only per-second compute Functions in this envelope. Modal charges CPU and memory by the larger of request and actual consumption. Enforce `cpu=(c,c)` and `memory=(1024g,1024g)` for the paid caps in the table, with an economical low-CPU-memory model loader. [Modal CPU and memory](https://modal.com/docs/guide/resources)
175:- Stage 3D measured `1.9757 s / optimizer step` and `23.88 ms / logical eval row` on one A100-80GB. Train and eval peaks were `52,972.5 MB` at microbatch 16 and `45,409.9 MB` at eval batch 64. The six 1,280-step runs took `2,884.0–3,217.8 s` including additional work. BF16 weights alone are about 8 GB at 4B parameters. Train activation and batched candidate workspace amplify this. The baseline plan uses A100-80GB for both. Accept the pilot run only if peak allocated/reserved fits a predeclared 65 GiB A100 threshold and sustained times meet `≤2.50 s/step`, `≤0.030 s/row` inclusive of in-batch data feeding. Rerun timing after any microbatch change. A100-40GB, L4 and A10/A10G require a separate shape and kernel check with smaller batches. The measured Stage 3D peaks exceed their VRAM. For a substitution, require equivalent logits, a ≤20 GiB peak on 24 GiB cards, and throughput below `0.030*(0.00074684/(GPU rate + 0.00005284))` seconds per row to match A100 costs, as well as a preapproved wall limit. This is about 81.5 ms on L4 and 62.4 ms on A10, with 2 cores and 12 GiB host RAM. Keep the confirmed A100 path if the smaller card would require quantizing the frozen host or invalidating kernels. [Stage 3D report](../../stage3d_qwen_signed_phase/REPORT.md) · [Modal GPU SKUs](https://modal.com/docs/guide/gpu)
176:- Train four matched connectors with one seed (`202611`), `384×32=12,288` logical examples per arm, mixing equal synthetic/real examples across the steps, using a frozen sampling manifest. Stage 3D locked the base training scorer and LR `0.0015`; use its temperature protocol on disjoint dev data and cap the extra dev scoring at 4,000 rows per arm. This is a single-seed resource-bounded replication tier. Freeze common architecture/parameter scales, prompt length cap and calibration before blind scoring.
177:- Synthetic blind extrapolation has 32 blocks × 64 mothers and holds cumulative phase beyond `π/2`, e.g. `θ=±{1.70,2.10,2.50,2.90}` rad, four independently seeded blocks per signed setting. Use lag families such as `(h,ℓ)=(2,2),(2,3)` counterbalanced within block, including both signs and unseen magnitudes. Use the same mother order and true signed synthetic condition metadata for all arms. Fit on the Stage 3D support or narrower, with separate seeds; known synthetic metadata is an intervention. Each synthetic mother contributes Q3 pre/post and Q2 `none/one/two`, for `40,960` logical rows across four arms. Include CPU first and second pilot likelihood ceilings and score-matched RefPool diagnostics. Estimate angles from allowed sync/pilot evidence for real data.
178:- Natural real work per arm is `3,680×(Q1 + 3×Q2) + 2,960×(2×Q3) = 20,640` rows, hence `82,560` across arms. Stress adds `(7×140 + 1,080)×4 = 8,240` rows. Dev adds `16,000`. A logical row already includes scoring all four candidate answers. Registered CPU physical arms score the identical rows but consume zero GPU time. Scope and statistical interpretation exclude prompt-level claims about rare jumps or undisclosed source domains that have no registered observations.
180:### 3.3 Budget arithmetic and zero-bill gate
182:Public Modal list prices checked for this round are A100-80GB `$2.4984/h`, A100-40GB `$2.0988/h`, L4 `$0.7992/h` and A10 `$1.1016/h`, plus CPU `$0.04716/physical-core-h` and memory `$0.007992/GiB-h`. The A100 Function cost with 2 physical cores and 12 GiB is `$0.00074684/s` or `$2.688624/h`. A10 is the pricing label and `A10G` appears in Modal examples. The main schedule budgets the A100 and assumes no savings from smaller devices. [Modal pricing](https://modal.com/pricing)
186:| A100 connector fit, `4×384×2.50` | 3,840 | 2.86786560 |
196:| **Total**, 2.8538889 A100 hours and 3 CPU function hours | | **8.44073416** |
200:Require account-visible `$0.00` out-of-pocket gating and sufficient remaining compute and volume/egress quotas before launch. On Modal Starter a workspace spend limit caps net charges separately from the workspace gross usage limit. Verify that the owner has set a net `$0.00` limit or an equivalent hard guard for the live workspace, and a gross cap covering the reserved stage amount within available credits. Use dashboard reading and control-plane caps as well as a local ledger because other jobs may consume credits and usage can lag. Hosted model APIs are a distinct charge; this run uses a frozen self-hosted model on Functions. No account state was verified or changed and no Modal workload was started in Round 1. [Modal budgets](https://modal.com/docs/guide/budgets) · [Modal billing](https://modal.com/docs/guide/billing)
217:The verifier checks directly that the frozen table chooses the turn's class for all four local codes, which explains SER equality of the scalar R2 and R3 arms. The matched sum and hence the turn coincide under the lift. The local code and calibration can still change. Attenuation at low SNR can improve frozen NLL without improving SER. The high-SNR loss from variance locking occurs when injected noise masks real high-SNR evidence. The observed saturation codes and conditional nuisance curves characterize the frozen weights under the stated draws. Generalization, real timing coverage and throughput remain empirical gates. See the full records for sample counts, scores, transition matrices, budget, manifests and geometry. [Verifier](verify_m4.py) · [Results](results_m4.json)
3:**Decision.** Verify the evidence contract and run the physical readout before training the connector. Make rotation and reference handling part of the interface. Use the actual frozen host's candidate-score Jacobian to scope a local adaptation claim. Buy one bounded pilot only after these checks pass. Use independent blind data for the final decision. The proposed Stage 4 compute cap is **\$9.50**, leaving **\$3.38** of the historically reported \$12.88. Round 1 used local CPU only.
27:Also $R(q_\tau)=H_3+\mathbb E D_{\rm KL}(p_3\Vert q_\tau)$. The sum telescopes to
50:For G1 the paired 64-mother block 95% intervals are **[.009178, .014998]** for s17 minus Full IQ and **[.491294, .516912]** for phase minus s17. Phase's observable telescope is .012087838 + .478025740 + .026076989 = .516190567 nats, using unit-temperature host scores for the middle contrast. About **97.66%** of phase's excess over Full IQ lies after the s17 readout when measured at locked temperature. The G1 phase-versus-zero uplift did meet Stage 3D's relative target, even though both arms exceeded the log-loss of a uniform predictor (1.386294). G1's co-rotation worst coordinate defect was **.380019** in phase and near machine precision for the direct readout. Offline 16-way group averaging brought phase NLL to **1.411312** and accuracy to **17.050%**. On the archive's sampled rotations these are distinct endpoints. The all-rotation `H_c4_035` row above is a fresh replay from the archive. Its identity-rotation, both-sign host metrics match Work F. The separate positive-sign Stage 3B lookup anchor in the master JSON is a different comparison.
64:Historical scope. `PROTOCOL_LOCK.json` records six frozen-host checkpoints, 8 epochs, 160 steps per epoch, batch 32, microbatch 16 and dev-fit temperatures. The Work E variance scenario used $q=.35,\rho=.01$, mother NLL SD .5, pilot block SDs .02104014/.01928086, and planned 131 blocks for the joint endpoint gate. Stage 3D's report orders smoke and scorer diagnostics, six trains (2,884–3,218 s each), independent dev variance lock, then blind scoring. The **\$17.12** receipt includes smoke, one OOM/benchmark, all six trains, dev and blind evaluation, and the aborted historical Ministral check. The archival charge was **\$0.00** after allowances. The 150,912 G1 rows are logical predictions per arm from a verified finite table, not 150,912 distinct forwards.
78:**Proposed decision margins.** On development folds, lock an admissible NLL target $r_*=R(q_{\rm IQ})+.05$ nats and a useful mean reveal improvement of .02 nats against the no-reveal control. Use each contrast at fixed context and comparable label distributions, with simultaneous bounds for multiple gate decisions. A certified $L_S>r_*$ ends the current representation branch. With entropy brackets $[L_0,U_0]$ and $[L_1,U_1]$ before and after reveal, $\max(0,L_0-U_1)\le I(A;R_{\rm new}\mid S)\le U_0-L_1$ for simultaneous valid brackets. An upper bound below .02 ends a **Bayes-gain** claim for that reveal contract. A direct readout whose paired upper regret bound is within .05 and whose reveal lower improvement bound exceeds .02 is a practical existence witness. Otherwise report uncertainty, improve the physical kernel or support, and defer GPU. At final release require independent upper confidence bounds within .05 of both the permitted Full-IQ and direct-cache comparators on the primary cell and a lower confidence bound above .02 for host reveal improvement where a gain was declared. These margins are Round 1 proposals for registration.
96:For development example (i), flatten the fixed-length embeddings $e_i=\mathrm{vec}(E_\theta(x_i))\in\mathbb R^{Kd_{\rm LM}}$, define centered host scores $F_i(e_i)$, and compute the actual candidate Jacobian $J_i=D_eF_i(e_i^0)$. For multi-token candidates differentiate the full sum $\sum_t\log p(w_t\mid w_{<t},e_i)$, including gradients of the log normalizer and the cached context. Use candidate-vector JVPs or VJPs in FP32. With a shared adapter perturbation $\delta$, stack $A_i=J_iD_\theta e_i$ and $d_i=\Pi\log p^*_i-F_i(e_i^0)$ after weighting examples by their locked masses. For the stack (A,d), a frozen tangent can achieve residual $\epsilon$ iff $\inf_{\|\delta\|\le r}\|A\delta-d\|\le\epsilon$; without the norm cap the projection residual is $\|(I-AA^+)d\|$. A rank threshold is necessary only for a declared target subspace. A one-dimensional joint task can succeed through a rank-one interface. A shared adapter may require stacked rank across inputs even if all individual $J_i$ have rank (m-1). Fixed thetas, mask patterns and prompts define which local claim is tested.
134:Under the planning prior $\mu\sim N(0,4)$, $\mu\mid Z_1=z\sim N(v\sqrt tz,v)$ with $v=(1/4+t)^{-1}$, from completing the square. The predictive distribution for $Z_2$ has mean $\sqrt tz[1+(1-t)v]$ and variance $1-t+(1-t)^2v$. Continue if predictive $P(Z_2\ge b_2\mid z)\ge .10$, equivalently $z\ge -0.270438$; early efficacy proceeds to the full rollout for independent validation. The exact least favorable Type-I rate is **.050000** with nonbinding futility and **.044695** with modeled binding futility and no CPU screen. These are probability calculations under the specified Gaussian law; the Monte Carlo has sampling error.
152:**Actual Stage 4 release.** Use the pilot primarily for engineering, variance, and predictive futility on disjoint development mothers, sessions or blocks. Freeze pilot/final coupling and calibrate on external development simulations before using the two-look efficacy boundaries. If that law cannot be justified, allow only futility at the first look and apply a single valid, independent final test on blind blocks after all three seeds and scoring settings are fixed. Use new synthetic mother namespaces, disjoint from the Stage 3D diagnostic archives, and hold out all physical records used for the claimed final test. Conditional on the pilot and model selection, the blind test must be valid for the declared null. Its unconditional false-promotion rate is at most its level by iterated expectation. Power after futility, particularly with seed-to-seed heterogeneity, still needs development calibration. Full release also requires passing physical margins and negative controls. Count mothers or recording clusters as independent units under the registered assumptions. Use randomization inference where the experimental randomization supports it, or a declared block model with sensitivity to correlation. For Pluto, 16 temporal blocks support inference within the capture when justified. For CTTC, report 72 nested blocks and session-level (9-unit) sensitivity. A sign-flip test is exact for justified randomization or a stated joint sign-symmetry null. Mean-zero shifts alone supply neither condition.
156:The arithmetic charges compute by billed seconds using public base rates A100-80GB .000694/s, L4 .000222/s and A10 .000306/s. All cloud lines include a planned ceiling of **2 physical cores (4 vCPU) and 16 GiB** at .0000131/core/s and .00000222/GiB/s. These are resource ceilings for the estimate. Confirm actual measured consumption fits before extending the pilot. Use standard preemptible Functions, one container at a time, no region premium, and budget cold start/scale-down in the seconds shown. The \$12.88 is a historical planning figure. Verify account usage, free storage, egress and pricing before launch. [Modal pricing](https://modal.com/pricing), [Resource controls](https://modal.com/docs/guide/resources)
158:| Work | Device | Max billed seconds | Rate incl. CPU/RAM | Cost cap |
162:| Synthetic pilot, 1 seed ≤173 of 960 steps (18.02%) | A100-80GB | 650 | .00075572 | .491218 |
163:| Resume seed 1 and train seeds 2 and 3, ≤2,707 steps total | A100-80GB | 6,750 | .00075572 | 5.101110 |
170:| **Authorization cap / buffer vs \$12.88** | | | | **9.50 / 3.38** |
172:Make dataset acquisition, weight pulls, cache/table computation, causal normalization, provenance linting, calibration folds, and source guard checks CPU jobs. Mount prepared read-only dataset features for evaluation and commit isolated outputs; local CPU reduces cloud seconds where practical. The three synthetic seeds train the one surviving Equivariant+RefPool design. Historical phase/zero archives, physical baselines and exact intervention controls give context. This envelope does not buy six full retrained comparator checkpoints. Any newly trained matched-baseline arm is a separate priced decision. Real evaluation uses those same frozen synthetic-trained weights and frozen candidate schema. If real development data are used to fit preprocessing or a temperature, freeze parameters by source fold and report strictly zero-shot calibration separately where claimed.
174:At the historical 1.9757 s/step the 2,707 continuation steps would take ~5,348 s before overhead. Require a receipt within 2.30 s/step, leaving ~524 s for all continuation overhead at that worst allowed rate. The new preprocessing, fused kernels and longest Q2 inputs can change this. At most 6,144 Q3 prefixes cover 4 cells × 2 signs × 256 exact keys × 3 seeds, provided the key includes all prefix state, dtype and batch policy. Q2's seven evidence arms require at most 10,752 prefixes for 512 mothers × 3 seeds. The 1,050 s synthetic scoring line therefore requires 16.1 prefixes/s before overhead. Cache prompts only across identical embeddings, masks, positions, template, candidate scope and attention state. Distinguish KV/linear-state sharing from lookup deduplication. Fix the padding and batch policy and require per-probability and paired endpoint parity on a receipt set before any blind inference. Historical BF16 differences across batch settings reached .067 in probabilities with per-cell argmax agreement as low as .6875. Historical A100 b64 peaked at 45,410 MB on inference, so use smaller fixed batches or prevalidated kernels on 24 GB cards. Keep precision locked across comparable arms.
176:The real rows cap at one query per packet/frame for up to seven evidence arms × three seeds. Without dedup, Pluto requires 47,040 prefixes (≥20.46/s within 2,300 s) and CTTC 30,240 (≥17.79/s within 1,700 s). Reserve startup separately within each row before setting a required steady rate. These rates, and VRAM on L4/A10G, require a parity-and-throughput witness under the actual workload. Use part of the relevant row for an unblinded developer smoke. Select the lowest-cost parity-passing device whose worst-case authorized count fits. Recompute cost after any failed receipt. Substitution allowance covers price difference at equal seconds, while a first device's already billed time still counts. A100 fallback is possible only for a separately metered subset within remaining cost/time. If one resource proof fails, preserve the CPU-only evidence and mark that GPU row incomplete. Neither the time cap nor the historical A100 rate mathematically guarantees real workload completion. Keep per-invocation deadlines, minimal scale-down, charge for billed teardown and any retries, and precharge each next shard at its bounded worst-case dollars against a single serial ledger. Account for already running and settling time before dispatch. Modal dashboard budget enforcement can lag and availability varies by plan. [Volume staging](https://modal.com/docs/guide/volumes), [Modal budgets](https://modal.com/docs/guide/budgets)
201:| 4B cache | Compute synthetic exact/controlled likelihood and causal real-data bounds on dev and support. Pass with the existence witness in §2.1. For a certified insufficiency, test an equal-budget canonical complex cache with identical observable inputs. Stop the branch if both cannot meet the objective. |
202:| 4C interface | Enumerate $C_4^2$ on finite cache codes and induced labels, check identity bytes for duplicate, masked, permuted and padded records, and verify genuine fresh effects can survive pooling. Require OOD mixed contrasts and a sufficient singular/margin witness for the design subspace. Check dev Jacobian stack rank before investing in a wider connector. |
203:| 4D promotion | CPU freeze the splits, paired estimand, useful margins, tolerances, calibration rule, alpha and maximum allocation. The GPU pilot verifies the timing/VRAM/parity contract and the dev continuation criterion. Release only after independent G1 primary, G2–G4 strata, Q2_C4, three seeds and exact controls. If eligibility fails, archive either the certificate or the prespecified inconclusive outcome. |
211:This graph has the topological order 4A, 4B, 4C, 4D, 4E, 4F, 5A, 5B, 6A, 6B, 6C, 6D, 6E. Preregister branch handling and assign new versions to future rounds so a failure cannot lead back into the same blind set. CPU checks remain useful when they return an equivalence class, an invalid split or a model mismatch. They certify scoped impossibility or feasibility; actual hardware throughput and independently held-out gain still need measurement.
213:**Why these edges exist.** P1–P3 show strong matched classical baselines, position shortcuts and static-surrogate scope. P4–P6 require separating physical frontends from learned contributions and distinguishing missing from zero. P7–P8 show that large probability updates can yield few corrected blocks and that mismatched past observations can hurt. P9–P11 require state-dependent reference reliability, phase-gauge accounting and counting newly wrong alongside corrected decisions. P12 requires unique evidence accounting and compares identifiable global references against naive locality; plug-in point estimates can lose mixture information. The earlier P12 Stage 5 fusion results scoped neural gains to the tested distribution. Stage 3D adds the strong direct-reader/weak-host distinction, failed raw rotation laws, redundant reference sensitivity, and an expensive numerical parity lesson. The roadmap's anti-loop rule is to predeclare one new question, finish with a success case, a failure case and a changed decision, and archive small BER variations after two rounds. Record the missing observation required by an impossibility theorem. [Long-term plan](../../../docs/LONG_TERM_RESEARCH_PLAN.md), [Multimodal integration plan](../../../docs/MULTIMODAL_INTEGRATION_PLAN.md), [Stage 3D report](../../stage3d_qwen_signed_phase/REPORT.md), [Protocol lock](../../stage3d_qwen_signed_phase/PROTOCOL_LOCK.json)
```

### Event #9: `item.completed` — `command_execution` (`item_6`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$d='"'research/stage4_math_and_design'; rg -n -i 'fixed_best|max.EIG|nearest|random|4.bit|16.bit|gate|pre.reg' \""'$d/M5/THEOREM_M5.md" | Select-Object -Last 80; Get-ChildItem research/stage4_math_and_design/M* -Filter '"'*r2*' -Recurse | Select-Object FullName; uv cache dir; rg -n -m 3 'matched.*control|seeds' 'C:/Users/李则徐/.codex/memories/rollout_summaries/2026-11-17T20-54-26-dFGP-p4_em_native_transformer_assessment_and_github_publication.md'"
```

**Aggregated Output:**

```text
39:### 1.2 Noisy generalization of the 2-bit versus 4-bit pair
98:If $\phi(t+d)-\phi(t)=\omega d+\sigma_\phi\sqrt d\,N\pmod{2\pi}$, the $n$-th circular moment propagates by $e^{-n^2\sigma_\phi^2d/2}e^{in\omega d}$. A jump multiplies the moment by $e^{inJ}$ after $\tau$. An independent uniform reset sets every nonzero circular moment to zero. Combine pilot likelihoods at their time nodes and compute $p(H,\tau\mid Z,A)$ by marginalizing state, slope, diffusion, jump amplitude, gain, and noise under each model. A noise burst can widen a likelihood while leaving its mean unchanged. Under the Gaussian model the residual energy carries additional variance evidence whenever there are residual degrees of freedom.
108:For a decoded matched sum $\hat T$, a conservative bound makes the precision tradeoff explicit. Let $e_U$ be the error of the surrogate log likelihood up to an additive constant shared across states. Conditional posterior KL to the surrogate is $E_p e_U+\log E_p e^{-e_U}\le(\sup_U e_U-\inf_U e_U)^2/8$ by Hoeffding's lemma. Its expected value upper-bounds the Bayes task loss, by data processing and the optimality of the cell decoder. The Gaussian mean terms give $e_U=2\sum_k \Re[(T_k-\hat T_k)^*m_k(U)]/v_k$ at fixed variance. Coarse rounding can therefore be more costly at higher precision. If angle error is at most $h/2$ and radial error at most $\epsilon_r$ on a bounded unsaturated bin, $|T-\hat T|\le\epsilon_r+|T|h/2$. Saturation and covariance terms must be handled separately.
125:Use the same boundary convention as the historical encoder, define a missing/invalid state, and compute the prefix in the encoding pass. Right-shifting by the extension length recovers s17 exactly for every supported input. This also preserves the original reader and its edge cases. Lossy polar coordinates alone cannot guarantee recovery of a learned label computed from the full raw sample vector. If a real-data adapter lacks a valid historical triplet, mark the legacy interface unavailable. For a Bayes-optimal decoder, an extended packet can always emulate the legacy reader. A strict gain for a query requires a refinement that divides some old cell into distinct conditional laws of positive mass. The benchmark reader uses the physical fields and leaves the legacy inner index unused, while still counting its bits. Its results therefore quantify an achievable decoder rather than the optimal legacy-plus-physical reader. The 12- and 16-bit residual thresholds are not nested. The 24-bit layout is proposed and has no risk estimate in this round.
162:**Strict dominance.** For a randomized baseline $r(a\mid z)$, the conditional expected posterior-NLL advantage is exactly $\sum_a r(a\mid z)[V^*(z)-V(a,z)]$. For nearest it is $V^*(z)-V(a_{\rm near},z)$. Each baseline is strictly worse in expectation iff it assigns positive probability to a strictly inferior action on a positive-measure set of caches. Uniform random needs one inferior action there. Nearest must itself be inferior. These statements assume the true predictive law and equal read costs. With a certified uniform evaluation error $\max_a|\hat V(a,z)-V(a,z)|\le\epsilon$, the estimated maximizer incurs at most $2\epsilon$ regret in true EIG, by applying the error bound at both maximizers.
167:- **Anti-aliasing.** For known pilot symbols, phases at all old times agree for two simple slopes with shared anchored phase at $t_0$ when $\Delta\omega(t_k-t_0)\in2\pi\mathbb Z$. A known-symbol candidate that violates this relation gives positive EIG at positive finite precision and interior weights. With old times $0,2,4$, slope difference $\pi$, and alternatives at 6 and 1, the first candidate remains aliased and the second gives opposite phase. From the anchor at 4, nearest chooses 6. A uniform unknown QPSK symbol at the candidate also masks differences in $(\pi/2)\mathbb Z$, so the candidate needs a known symbol or a cross-window constraint. Posterior phase diffusion and uncertain association can erase otherwise useful separation.
168:- **Smooth versus jump.** Two trajectories can match at $0,D$ when the smooth slope is $\omega$ and the jump model uses $J$ at $\tau$ and slope $\omega-J/D$. Their phase difference at a candidate $t$ is $J(1\{\tau\le t\}-t/D)$. At fixed calibrated noise, an interior window can have strictly larger $2|\alpha|\left|\sin[J(1\{\tau\le t\}-t/D)/2]\right|/\sqrt v$ than nearest. For simple rivals this gives strict dominance by the Gaussian ordering above. With uncertain $\tau$ and fading, integrate the mixtures.
189:The 16-bit gain over s17 is 0.3119 nat for event and location, paired pointwise 95% interval $[0.3023,0.3214]$. The remaining gap to Full IQ is 0.0735 nat. For the conditional smooth-versus-jump task the pooled NLL is 0.5240 (s17), 0.5136 (16 bit), and 0.5131 (Full IQ). For localization given a true jump it is 0.9711, 0.9382, and 0.9379, respectively. The recovery varies with regime. At +24 dB conditional binary NLL falls from 0.0300 with s17 to nearly zero with 16 bits. Under selective fading it is 0.6808 versus 0.6820, and both the 16-bit and continuous results are around 0.68. The observed low-SNR conditional localization remains near $\log3$ for all readers. Thus the larger joint event gain should not be read as an equally large binary-discrimination gain.
195:With the 16-bit cache, EIG, random, and nearest attain pooled NLLs 0.3544, 0.4497 and 0.4772, and accuracies 81.18%, 73.34% and 71.63%. The comparison below conditions on the same ambiguous-case stratum for all methods, defined for evaluation only by the old Full IQ smooth posterior in $[0.1,0.9]$. All 2,048 cases qualify in each of the three listed profiles. The +24 dB profile has no qualifying cases and shows a ceiling tie between policies at 16 bits.
197:| Ambiguous profile | Max EIG NLL / accuracy | Random NLL / accuracy | Nearest NLL / accuracy |
203:In selective fade, max EIG gains 0.2772 nat over random, paired pointwise 95% interval $[0.2500,0.3043]$, and 0.3345 nat over nearest, interval $[0.3037,0.3652]$. Expected EIG is 0.4542 nat and realized gain is 0.4395 nat. Their realized-minus-expected difference has a paired interval $[-0.0405,0.0112]$. The already committed true and rival predictive observation NLLs are 0.5289 and 4.0492 in the same observation units. The largest checked change in EIG under grid refinement is $1.77\times 10^{-6}$ nat.
219:- **M1.** Freeze the factored physical evidence interface, 16-bit candidate encoding, and exact legacy prefix. Preserve C4 transformation laws on candidate probabilities, and transform dynamics and reference factors consistently. Use matched old IQ to measure the separate cache and host readout gaps. Keep the continuous physical decoder as an oracle.
222:- **M6.** Freeze risk margins by query and capture stratum, candidate-access rules, and held-out proof of sealing before training larger hosts. Compare 12 and 16 bits with all storage and read costs accounted for. Include the continuous ceiling, ambiguous-case coverage, a fixed-best development baseline, forecast calibration, step-by-step I/O and latency. Preserve unidentifiable and tied cases in the report.
C:\Users\李则徐\AppData\Local\uv\cache
24:- Follow-up receiver experiments reduced errors on unseen delay distributions, but the pure Fourier/逐径 pilot front-end alone achieved nearly all of the gain. Therefore do not attribute this improvement to Transformer attention without matched-front-end controls.
31:- Completed formal ten-seed P4-D protocol using seeds `74001–74010`, 600 training steps, batch size 64, exact integer bit/block accounting, exhaustive 1024-state sign-flip tests, and Holm adjustment.
```

### Event #10: `item.completed` — `command_execution` (`item_7`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$d='"'research/stage4_math_and_design'; rg -n 'f708|334.?096|455.?000|2,240|1,440|140|384|8\\.44|202611|36 |72 |15, 18|14, 17|loop.pair' \""'$d/M4/THEOREM_M4.md" "$d/M4/reply_r1.md"; rg -n '"'0\\.18|\\.18|0\\.49|prior|futility' \""'$d/M6/THEOREM_M6.md" | Select-Object -First 20'
```

**Aggregated Output:**

```text
research/stage4_math_and_design/M4/reply_r1.md:7:**Real benchmark.** Lock W1/W2 from one symbol each. Reveal one and then two genuinely observed past pilots without recomputing the cache. Use `15,18` for Pluto and `126,127` for CTTC. Q1 scores the transmitted relative turn; Q2 scores the transmitted symbol and yields SER, bit NLL and posterior-mean EVM; Q3 predicts the future received sector at a physical retransmission. Use the next Pluto copy and the next CTTC loop at the same TX source index. Register 2,240 Pluto frames in 16 blocks, 1,440 CTTC packets in 72 session-loops, and 720 CTTC Q3 pairs within 36 loop pairs. The CPU verifier certifies the archived split, source-bit and local interpolation geometry. Pluto's whole-record synchronization and CTTC's offline full-source lock are excluded from the causal frontend. Real-run qualification requires independently bounded timing plus a raw-support and timestamp audit. Use the raw-I/Q B3 mixture, direct `s17` reader, Qwen Zero, Naive-Additive, Equivariant-Coupled and RefSufficientPool on the registered rows with the same release schedule. [Benchmark design](THEOREM_M4.md#2-hardware-benchmark-and-causality-contract)
research/stage4_math_and_design/M4/reply_r1.md:9:**Modal envelope.** The pinned downloads are `2,789,188,515` bytes, plus the Qwen snapshot. The cap is **$8.44073416**, including 10,274 A100-80GB seconds, metered host CPU/RAM and retry reserve, plus `$0.76770000` CPU-only work. That leaves **$4.43926584** against the historical `$12.88` balance. It covers four single-seed connector fits, `82,560` raw real rows, `8,240` stress rows, `40,960` synthetic extrapolation rows and `16,000` dev rows. Start only after the dev runtime and causal gates pass and the workspace confirms live credits and a `$0.00` net spend guard. No paid or free Modal jobs were dispatched in this round. [Budget](THEOREM_M4.md#3-frozen-compute-envelope-and-dispatch) · [Modal pricing](https://modal.com/pricing) · [Modal spend limits](https://modal.com/docs/guide/budgets)
research/stage4_math_and_design/M4/reply_r1.md:14:- **M2.** Can we lock Q2 `two−one` NLL and synthetic Q3 coupled-vs-additive NLL as the primary contrast family, with lock failures included, paired cluster intervals, and a predeclared capture/source sensitivity analysis? The 384-step single-seed budget supports a conditional seed result.
research/stage4_math_and_design/M4/THEOREM_M4.md:120:| Pluto 850 MHz | One row per frame. Copy 1, \(k_1=30+(f\bmod67)\); \(k_2=k_1+h_{12}\) with \(h_{12}=1\) or 8 using `(f + floor(f/16)) mod 2`. | Two acquisitions at logical indices `15` and `18` of copy 1. Initial state uses sync `0:13` and split-local past, with those two pilot reads excluded from all pre-reveal quantities. | Same payload slot \(k_1\) in the immediately following complete copy, \(L=126\), \(\ell=126-h_{12}\). Score 2,240 candidates with a common lock-failure policy. A copy is a real retransmission; no target-label matching or synthetic replacement. |
research/stage4_math_and_design/M4/THEOREM_M4.md:121:| CTTC 433/708/2450 MHz | One row per packet with ordinal \(j\) and loop \(b\). \(k_1=136+((17j+5b)\bmod48)\); \(k_2=k_1+h_{12}\), using 1 for even \(j+b\), 8 otherwise. | `126,127` of the same packet, both earlier than \(k_1\). Initial state uses `0:6` sync and authorized previous data, excluding both release samples. Estimate their joint covariance. `254,255` remain unavailable until they have physically arrived. | On each capture use disjoint loop pairs `(0,1),(2,3),(4,5),(6,7)` at the same packet source start and same \(k_1\) selected in the earlier loop. \(L=500000\), \(\ell=500000-h_{12}\). Q1/Q2 cover 1,440 packets. Q3 has 720 retransmission candidates and 36 loop-pair groups. |
research/stage4_math_and_design/M4/THEOREM_M4.md:125:The primary samples retain actual fading, residual DC/ISI, noise, clock offset, and any changes present at registered times. A naturally occurring jump is a stratum, with changes flagged by a predeclared causal split-local pilot innovation criterion and a retrospective scorer-only morphology check. A CPU-only cross-frame Q1 diagnostic pairs `(f−1,f)` at `f mod 20 = 0` within each test block, using the earlier frame's \(k_1\) and current frame's \(k_1\) from the table. Its matched control pairs the preceding two interior frames `(f−2,f−1)` in that same block. Use only available sync and authorized pilots and a gap-class prior when the time between Pluto columns is unavailable. This samples recorded batch boundaries by schedule without requiring or asserting a jump at each boundary. Separate source domains and nominal session strata in analysis, keeping capture IDs out of model text/features. Hold `f708_snr10` out of fitted connector weights, temperatures and shared prior tuning to expose one capture transfer; allow its own causal sync and released pilots. Pluto is a blocked same-campaign test. CTTC has one public cyclic TX sequence; input-source disjointness and capture dependence are distinct issues. Use block-paired NLL/SER differences on 16 Pluto groups and 72 CTTC session-loops, group adjacent repeat pairs for CTTC Q3, and report a capture-level sensitivity analysis and source-ordinal sensitivity because loops/captures reuse source positions. Report all registered rows and lock coverage, with the same uniform predictive fallback on failed rows and a secondary locked-only analysis. Avoid treating all packets as independent. Select the Q2 `two−one` proper-loss contrast and large-angle synthetic Q3 coupled-vs-additive contrast before test, with paired intervals and the multiplicity rule frozen with M2.
research/stage4_math_and_design/M4/THEOREM_M4.md:127:Keep the natural recordings as the primary results. Recreate seven Pluto stress recipes on 140 frame IDs `range(4800,7040,16)` each, and all 1,080 CTTC stress IDs from the branch identity rule (`i mod 4 != 3`, round-robin modes). Score a single two-reference Q2 vector per registered stress row in the GPU budget. For perturbation strength use frozen train-scale or authorized pre-cutoff sync/pilot-only power. Pluto’s legacy injection measures power across the full frame, and CTTC’s legacy injection uses `offline_sig_pow` derived with full TX truth. Their historical raw scores stay available as historical comparators. Hide exact injected CFO, noise power and jumps in the scoring vault and keep RNG independent of payload labels. Real-only CFO uncertainty is estimated from observed pilot evidence. [Pluto data and acquisition](https://data.mendeley.com/datasets/k72b4pnc5g/1) · [CTTC data and acquisition](https://zenodo.org/records/4895133)
research/stage4_math_and_design/M4/THEOREM_M4.md:131:**Split proof.** Pluto uses frame intervals `[0,3200)`, `[3300,4500)`, `[4600,7040)` including the first 200 warmup frames of each split. The 100-frame gaps imply pairwise empty intersections. All copies of a frame are inseparable across splits. CTTC uses six disjoint cyclic source ranges of length-500,000 circumference, including `train target [0,160000)`, `train ref [164096,200000)`, `dev target [204096,290000)`, `dev ref [294096,330000)`, `test target [334096,455000)`, `test ref [459096,495904)`. The boundary gaps are at least 4,096 symbols, including the wrap. The verifier regenerates starts `lo + i floor((hi−lo−256)/n) + 128`, enumerates both bit pairings' union and asserts disjoint bit footprints across split and role. On the archived nine anchors the 5,472 target/reference acquisitions fit inside their files, have distinct IDs, and the nearest receive ranges remain 16,481 samples apart after 255-sample RRC support expansion. This verifies the stored geometry, including 72 test groups. Acquisition edges must be rechecked on fresh causal alignment. Source sequence reuse within a split creates statistical dependence; the proof addresses explicit identity and support leakage.
research/stage4_math_and_design/M4/THEOREM_M4.md:176:- Train four matched connectors with one seed (`202611`), `384×32=12,288` logical examples per arm, mixing equal synthetic/real examples across the steps, using a frozen sampling manifest. Stage 3D locked the base training scorer and LR `0.0015`; use its temperature protocol on disjoint dev data and cap the extra dev scoring at 4,000 rows per arm. This is a single-seed resource-bounded replication tier. Freeze common architecture/parameter scales, prompt length cap and calibration before blind scoring.
research/stage4_math_and_design/M4/THEOREM_M4.md:178:- Natural real work per arm is `3,680×(Q1 + 3×Q2) + 2,960×(2×Q3) = 20,640` rows, hence `82,560` across arms. Stress adds `(7×140 + 1,080)×4 = 8,240` rows. Dev adds `16,000`. A logical row already includes scoring all four candidate answers. Registered CPU physical arms score the identical rows but consume zero GPU time. Scope and statistical interpretation exclude prompt-level claims about rare jumps or undisclosed source domains that have no registered observations.
research/stage4_math_and_design/M4/THEOREM_M4.md:186:| A100 connector fit, `4×384×2.50` | 3,840 | 2.86786560 |
research/stage4_math_and_design/M4/THEOREM_M4.md:189:| A100 synthetic, `ceil(40,960×0.030)` | 1,229 | 0.91786636 |
research/stage4_math_and_design/M4/THEOREM_M4.md:196:| **Total**, 2.8538889 A100 hours and 3 CPU function hours | | **8.44073416** |
76:- A collision class with fixed $S=s$ and at least masses $a,b>0$ of two distinct labels contributes at least $(a+b)h_2(a/(a+b))$ to conditional entropy, where $h_2(u)=-u\log u-(1-u)\log(1-u)$. Restrict to the two-label subevent and condition on its indicator. Exact equivalence of two hypotheses with equal prior forces at least $\log 2$ nats and error at least $1/2$ within that subproblem. If their distributions are only close, equal-prior optimal binary error is $\tfrac12(1-\mathrm{TV}(P_0,P_1))$, from the likelihood-ratio decision rule. This yields a CPU impossibility bound when TV can be bounded above.
126:This is optimal for specified priors, utilities, transition likelihoods and caps. Greedy information per dollar is at most a heuristic when checks are complementary. In a differentiable separable allocation with concave information utilities $v_j(n_j)$, constant per-unit prices $c_j$ and a binding budget, KKT gives $v'_j(n_j)/c_j=\lambda$ for interior arms, with only arms above this price at the origin receiving effort. For $v_j=-w_j\sigma_j^2/n_j$, the allocation is $n_j\propto\sqrt{w_j\sigma_j^2/c_j}$, with independent clusters and a fixed variance estimand. Use mandatory cohort minima and integer reallocation, especially for nine CTTC sessions. No universal zero-waste guarantee follows without a specified law. In a nonidentifiable case the valuable output is the collision and the minimum extra observation, even when no GPU trial follows.
130:Use a preregistered directional **primary** score $\mu>0$ in final-standard-error units above the practical margin. In the canonical independent-increment Gaussian experiment, $Z_1\sim N(\sqrt t\mu,1)$ and $Z_2=\sqrt tZ_1+\sqrt{1-t}V\$, $V\sim N(\sqrt{1-t}\mu,1)$ independent. $t=.18$ is an information fraction. A one-seed pilot at 15–20% of optimizer steps does not, by itself, have this information fraction or this joint distribution.
132:With one-sided alpha $\alpha=.05$, use O'Brien–Fleming-style spending $a(t)=\bar\Phi(z_{1-\alpha}/\sqrt t)$ or predeclare Pocock-style $a(t)=\alpha\log[1+(e-1)t]$ for more early alpha. For the former, $a(.18)=.00005288548$, $b_1=3.876957$; solve $a(t)+\int_{-\infty}^{b_1}\varphi(z)\bar\Phi((b_2-\sqrt tz)/\sqrt{1-t})\,dz=.05$ for $b_2=1.645088$. Refusing continuation can only remove rejection paths if boundaries are held fixed. Componentwise monotonicity in $\mu$ puts the least favorable null at zero. Familywise guarantees require preallocated alpha across multiple promotion hypotheses, or a gatekeeping order; for mandatory jointly successful endpoints, an intersection-union gate can use the full level per component while requiring all successes. [Lan and DeMets, 1983](https://eclass.uoa.gr/modules/document/file.php/MATH301/PracticalSession3/Lan_DeMets_1983.pdf)
134:Under the planning prior $\mu\sim N(0,4)$, $\mu\mid Z_1=z\sim N(v\sqrt tz,v)$ with $v=(1/4+t)^{-1}$, from completing the square. The predictive distribution for $Z_2$ has mean $\sqrt tz[1+(1-t)v]$ and variance $1-t+(1-t)^2v$. Continue if predictive $P(Z_2\ge b_2\mid z)\ge .10$, equivalently $z\ge -0.270438$; early efficacy proceeds to the full rollout for independent validation. The exact least favorable Type-I rate is **.050000** with nonbinding futility and **.044695** with modeled binding futility and no CPU screen. These are probability calculations under the specified Gaussian law; the Monte Carlo has sampling error.
136:The paired 10,000-candidate replay assumes a 60% structural-null stratum ($\mu=-.8$, CPU pass .10), 15% boundary null ($\mu=0$, pass .90) and 25% working ($\mu=3.6$, pass .99). Pass draws are conditionally independent given stratum. These are planning assumptions, not measured rejection rates for M1–M5. Both policies see the same potential Gaussian scores. Fixed spends the archival \$17.12 envelope for every candidate. Sequential spends a modeled 8.5% if CPU passes and the pilot is bought ($.75\times.18/3+.04$ of fixed, comprising a training share and a pilot overhead share). It pays the remaining fraction on continuation including early efficacy. This scales an archival bill as a scenario; it neither prices an individual seed retrospectively nor assumes a \$17.12 charge in the Stage 4 cap. If $p_s(\mu)=P(\mathrm{CPU\ pass}\mid\mu)$, then
152:**Actual Stage 4 release.** Use the pilot primarily for engineering, variance, and predictive futility on disjoint development mothers, sessions or blocks. Freeze pilot/final coupling and calibrate on external development simulations before using the two-look efficacy boundaries. If that law cannot be justified, allow only futility at the first look and apply a single valid, independent final test on blind blocks after all three seeds and scoring settings are fixed. Use new synthetic mother namespaces, disjoint from the Stage 3D diagnostic archives, and hold out all physical records used for the claimed final test. Conditional on the pilot and model selection, the blind test must be valid for the declared null. Its unconditional false-promotion rate is at most its level by iterated expectation. Power after futility, particularly with seed-to-seed heterogeneity, still needs development calibration. Full release also requires passing physical margins and negative controls. Count mothers or recording clusters as independent units under the registered assumptions. Use randomization inference where the experimental randomization supports it, or a declared block model with sensitivity to correlation. For Pluto, 16 temporal blocks support inference within the capture when justified. For CTTC, report 72 nested blocks and session-level (9-unit) sensitivity. A sign-flip test is exact for justified randomization or a stated joint sign-symmetry null. Mean-zero shifts alone supply neither condition.
```

### Event #11: `item.completed` — `command_execution` (`item_8`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$d='"'research/stage4_math_and_design'; rg -n -i 'arm 0|arm 1|arm 2|host-pure|naive-additive|physical.residual|0\\.0|invariance|near|fixed_best' \""'$d/discussions/M1/prompt_r2.md" "$d/discussions/M3/prompt_r2.md" "$d/discussions/M4/prompt_r2.md" "$d/discussions/M5/prompt_r2.md"; Get-Content "$d/M2/verify_m2.py" | Select-Object -Skip 570 -First 143'
```

**Aggregated Output:**

```text
research/stage4_math_and_design/discussions/M5/prompt_r2.md:7:   - In `results_m5.json` Round 1, `max_EIG` crushed `random` (`90.63% vs 71.63%`) and `nearest` (`70.21%`) on selective-fade — **yet time `4` won ALL 2,048 recordings**, meaning a static `fixed_best_on_dev = time 4` rule tied active per-recording `max_EIG`!
research/stage4_math_and_design/discussions/M5/prompt_r2.md:8:   - Wait: **what mathematically forces `max_EIG` to strictly dominate EVERY static `fixed_best_on_dev` window `a_const`?** Obviously whenever **recording-specific old cache `Z` carries conditional information about WHERE the ambiguity lies** (e.g., recording-varying selective deep fades seen via `log_radial(Z)`, OR heterogeneous change-point intervals $\tau$ localized differently per recording by `Z`, OR competing CFO slopes aliased on different candidate subsets per recording)!
research/stage4_math_and_design/discussions/M5/prompt_r2.md:10:   - M4 showed that on real hardware without Stage 3's fixed SNR/amplitude, forcing SNR down to `2 dB` destroys `20 dB` (`0.00% -> 19.89%` SER), while M4's R3 whitened lift `(T, E)` preserves scalar precision cleanly, and M4 proposed **extracting BOTH `4-bit s17` and your `16-bit canonical` packet in Stage 4's zero-GPU CPU shard pass**!
research/stage4_math_and_design/discussions/M5/prompt_r2.md:13:   - M6 asked: can you freeze ONE concrete **Stage 6 Ultimate Blind Test Query & Evidence Contract** common to `Qwen3.5-4B`, `Qwen3.5-9B`, `Qwen3.8-27B`, and `LEMMO` under identical pre-question cache bits (`16 bits/window`), candidate read budget (`k = 1` sealed read), pre-reveal forecast commitment scoring, AND M2's paired statistical gate (requiring strict win over `random`, `nearest`, AND **`fixed_best_on_dev`**!)?
research/stage4_math_and_design/discussions/M5/prompt_r2.md:17:1. **Prove & Empirically Demonstrate Strict Per-Recording Adaptive EIG Dominance Over `fixed_best_on_dev`**:
research/stage4_math_and_design/discussions/M5/prompt_r2.md:18:   - Prove the exact mutual-information condition $I(a^*(Z); Y_a \text{ utility} \mid \text{marg}) > 0$ under which `max_EIG(Z)` strictly improves expected posterior NLL and accuracy over $\max_{a \in \mathcal{C}} \mathbb{E}_Z[\dots]$ (`fixed_best_on_dev`), i.e., Bayes value of conditional adaptive experiment design!
research/stage4_math_and_design/discussions/M5/prompt_r2.md:19:   - Add a new deterministic benchmark section in `verify_m5.py` (**`multi_geometry_adaptive_suite`** — e.g., varying localized jump intervals + varying candidate-window fading reliability + CFO aliasing across recordings) and verify empirically via paired CIs that **`max_EIG(Z)` strictly beats ALL FOUR controls: `random`, `nearest`, AND `fixed_best_oracle/dev`** on both post-reveal NLL and accuracy!
research/stage4_math_and_design/discussions/M5/prompt_r2.md:22:3. Run updated `verify_m5.py` via `uv run` cleanly and summarize exact numbers (`max_EIG` vs `fixed_best_on_dev`, `random`, `nearest`) in `reply_r2.md`.
research/stage4_math_and_design/discussions/M4/prompt_r2.md:11:     - M2 proved `1` seed or `3` seeds CANNOT establish population-of-seeds generalization (`>= 10` seeds needed for that!), so Stage 4's real-data + synthetic GPU budget MUST explicitly state a **Fixed-Connector Conditional Scope** OR use `1` seed per each of the `4` matched arms (`Zero`, `Naive-Additive`, `Host-Pure Equivariant+RefPool`, `Physical+HostResidual` -> M1's converged 4-arm ablation!) + optional `2` extra seeds ONLY for the winning M1 `Host-Pure / Physical-Residual` pair if Stage-1 dev pilot clears M6's Tier-2 gate!
research/stage4_math_and_design/discussions/M4/prompt_r2.md:12:   - **Resolve this cleanly with exact USD arithmetic (`<= $9.50` cap inside `$12.88` remaining credits)**: Put A100-80GB (`$2.4984/h + cpu/ram = $0.00074684/s` or M6's rate check — wait: standardize the EXACT Modal pricing formula!) as the verified VRAM-safe default (`no unverified L4 assumption unless a $0.10 micro-batch parity smoke proves <= 20 GB!`), reconcile M6's `$0.49` 1-seed pilot gate cleanly BEFORE spending full arm budgets, and check how zero-shot `f708_snr10` holdout fits inside the same row count!
research/stage4_math_and_design/discussions/M4/prompt_r2.md:14:   - M5 proved `4-bit s17` saturates along rays (which you ALSO proved independently — `local` saturates to `[0,4,8,12]` above gain `21.90` and gives constant code below `0.032`!), causing Stage 5–6 NLL loss (`0.3119 nats`) and hiding SNR changes across CTTC (`-3..+24 dB`) and Pluto fading (`>10.6 dB`).
research/stage4_math_and_design/discussions/M4/prompt_r2.md:23:1. Lock the **exact `PhaseContext` + `RefPool` + `Dual 4-bit s17 / 16-bit canonical` CPU Shard Contract** for Pluto (`7,096` frames) and CTTC (`9` captures), explicitly checking causal filtration non-double-counting and receive-gauge vs TX-relabel invariance!
research/stage4_math_and_design/discussions/M4/prompt_r2.md:24:2. Reconcile the **Modal A100-80GB Tier-2 Pilot-Gated Schedule (`<= $9.50` cap, `$0.00` net guard)** combining M1's 4-arm ablation (`Zero`, `Naive-Additive`, `Equivariant-Host-Pure`, `Equivariant-Physical+Residual`), M6's dev pilot futility gate BEFORE full continuation, and `f708_snr10` held-out session transfer! (Explain precisely why M4+M6 pricing rates differed slightly if CPU/RAM sizing differed: `2c/12 GiB` vs `2c/16 GiB`, and lock one exact Modal pricing table!)
research/stage4_math_and_design/discussions/M3/prompt_r2.md:8:   - Furthermore, real hardware pilots at nearby lags (e.g., CTTC `126, 127` adjacent symbols through `256`-tap RRC pulse / residual ISI / shared CFO error $\epsilon_\omega$) can share **estimation uncertainty or noise covariance** $K_{\text{ref}}$ AND may use known QPSK pilot roots $d_m \neq 1$!
research/stage4_math_and_design/discussions/M3/prompt_r2.md:14:4. **From M2 & M6 (Idempotence, Byte-Exact Prefix Invariance & Calibrated Comparison)**:
research/stage4_math_and_design/discussions/M3/prompt_r2.md:16:     - Raw duplicate IDs, masked `shares_channel=False` distractors, padded empty slots, and permuted input ordering produce **literally byte-identical** fixed-slot tensors / prefix embeddings and `0.0` KL/TV shift;
research/stage4_math_and_design/discussions/M3/prompt_r2.md:17:     - Contrast-check native host logits (`Arm 1`) vs physical+residual (`Arm 2`) vs `q2_reader` posterior under dev-locked calibration.
research/stage4_math_and_design/discussions/M3/prompt_r2.md:23:   - Derive the augmented Gaussian covariance $(B_{t_j} + \tilde\omega t_j)$ combining Brownian increments $\sigma_\phi^2 \min(t_j, t_k)$ AND shared linear CFO prior variance $v_\omega t_j t_k$ (note rank-1 rank addition cleanly solved by Sherman-Morrison or appending 1 Gaussian-Hermite quadrature node over $\tilde\omega$!).
research/stage4_math_and_design/discussions/M1/prompt_r2.md:1:# Track M1 (Round 2): Cross-Track Convergence — Pure-Host vs Physical-Residual Ablation, 16-Bit/s17 Dual Cache & Real-Data Group Action
research/stage4_math_and_design/discussions/M1/prompt_r2.md:6:1. **M2 (Stage 3D Forensics)**: On `blind_q3_scores.npz`, `G1` locked `NLL(zero) - NLL(phase) = +0.035650` splits into `+0.036543` scalar temperature mismatch and **`-0.001017` reverse-conditional KL**; after seedwise scalar recalibration, **`zero` beats `phase` across all 5 cells** (`G1: -0.000893`, `G2: -0.027716`, `G3: -0.000430`, `G4: -0.026351`, `H: -0.003695`), despite `0.3631 nats` signed CFO MI available in `s17`! M2's Stage 4 gate requires post-dev-calibration NLL/Acc/NIG ($\eta_{\min}=0.10$), ECE $\le 0.05$, equivariance defect $\le 10^{-5}$, and fixed-text slot interventions.
research/stage4_math_and_design/discussions/M1/prompt_r2.md:7:2. **M6 (DPI Loss Ladder & Host Jacobian)**: On `G1`, `Full IQ` (`0.9290`) $\to$ `direct_s17` (`0.9411`, `+0.0121`) $\to$ `Qwen phase unit-T` (`1.4192`, `+0.4780`) $\to$ `locked-T` (`1.4452`, `+0.0261`) shows **97.66% of excess NLL occurs after `s17`**. M6 derived the frozen-host candidate-score alignment matrix $M_{\text{cand}} = \Pi C_{\text{cand}} E_{\text{vocab}} D_{\text{answer}} B$ and asks how to test the true frozen host rather than merely bypassing it via a standalone head.
research/stage4_math_and_design/discussions/M1/prompt_r2.md:9:4. **M5 (Universal Stage 4→6 Cache)**: Proved `4-bit s17` merges amplitude/SNR (`|T|=0.02 vs 0.5833 -> 0.4804 TV`) and loses `0.3119 nats` on Stage 5 joint event+location (`1.2788 -> 0.9669`), AND exhibited two IQ pairs sharing exact `(T, E)` that produce different `s17` inner codes `[2, 0]`. M5 converged on a **16-bit bit-exact canonical extension**: `[2b turn | 2b s17_inner | 6b canon_angle | 4b log_radial | 2b resid_quality]` (recovering `s17` by right-shift while matching `Full IQ` closely in Stage 5–6).
research/stage4_math_and_design/discussions/M1/prompt_r2.md:16:1. **The 4-Arm Causal Ablation Architecture (`Host-Pure` vs `Physical+HostResidual`)**:
research/stage4_math_and_design/discussions/M1/prompt_r2.md:18:     - **`Arm 0` (Stage 3D Baseline)**: non-quotient separate `<W1><W2><PHASE>`, no bypass.
research/stage4_math_and_design/discussions/M1/prompt_r2.md:19:     - **`Arm 1` (`Equivariant-Host-Pure`)**: quotient canonicalization (`canonical_input` + `rotate`) + multiplicative phase-coupled continuous summary fed purely as **soft tokens into the frozen host**, with **NO physical logit bypass** (`q_host` computed strictly from frozen LM / frozen surrogate logits, with isotropic `W_out` geometry)!
research/stage4_math_and_design/discussions/M1/prompt_r2.md:20:     - **`Arm 2` (`Equivariant-Physical-Residual`)**: canonical physical prior + bounded trainable host residual (`delta = gamma * tanh(ell_host)`), plus fixed-text counterfactual slot-swap and shuffled-slot diagnostics isolating host contribution beyond the analytic bypass!
research/stage4_math_and_design/discussions/M1/prompt_r2.md:21:   - Implement and run **BOTH `Arm 1 (Host-Pure)` AND `Arm 2 (Physical+Residual)`** in `verify_m1.py` on the SAME frozen CPU surrogate host and identical `|\omega| <= 0.25` training split, testing `G1..G4`, equivariance defect (`<= 1e-6`), and M6's rank/margin candidate-Jacobian diagnostic! (Does `Arm 1 Host-Pure` ALSO extrapolate >55% past $\pi/2$ on `G1/G3`, or does frozen random projection struggle without coupled tokens / sufficient width?)
        assert eligible[0] == scenario["t_by_endpoint"][k]
        selected = pilot["b_locked"] - 2
        close(power[selected], scenario["t_at_selected"][k]["power"], "Work E selected NCT power", atol=2e-9)
        curves[k] = {"sd": sd, "min_B": int(eligible[0]), "power_at_131": float(power[selected])}
    return curves


def power_analysis(matrices, pilot):
    alpha, target, s = .05 / 3, .95, 3
    rng = np.random.default_rng(SEED)
    reps = 200_000
    estimates = {cell: {endpoint: components(d) for endpoint, d in entry.items()} for cell, entry in matrices.items()}
    result = {"stage3_lock_replay": stage3_lock_check(pilot), "planning_alpha": alpha,
              "target_marginal_power": target, "seed_count": s, "MC_seed": SEED, "MC_reps": reps,
              "variance_components": estimates, "scenario_proxies": {},
              "multiple_testing_check": multiple_testing_mc(rng, reps, .05), "synthetic_cell_plans": {}}
    g1 = estimates[stage3.CELL_ORDER[0]]
    variance = lambda est: tuple(est[k] for k in ("var_block", "var_seed", "var_interaction"))
    for cell, entry in estimates.items():
        result["synthetic_cell_plans"][cell] = {}
        for endpoint, est in entry.items():
            d = .03 if endpoint == "accuracy" else .04
            v = variance(est)
            result["synthetic_cell_plans"][cell][endpoint] = {
                "effect_above_margin": d, "fixed_min_B": min_fixed(d, s, v, alpha, target),
                "random_min_B": min_crossed(d, s, v, alpha, target),
                "fixed_power_at_131": float(fixed_power(d, 131, s, v, alpha)),
                "random_power_at_131": crossed_power(d, 131, s, v, alpha, 192)}
    designs = [(f"B{b}_independent", b, 1, 0) for b in (16, 72, 131)]
    designs.extend((f"CTTC_9_sessions_x8_rho_{rho:.2f}", 9, 8, rho) for rho in (0, .1, .25))
    for design, b, per_group, rho in designs:
        design_result = {}
        for ep, est in g1.items():
            vu, vs, ve = variance(est)
            v = (vu * (rho + (1 - rho) / per_group), vs, ve / per_group)
            d = .03 if ep == "accuracy" else .04
            ef = max(est["mean"] - .02, 0)
            design_result[ep] = {"independent_groups": b, "clusters_per_group": per_group,
                "assumed_block_effect_session_ICC": rho, "variance_components_at_group_level": list(v),
                "scenarios": {key: {"excess": x, "fixed_min_groups": min_fixed(x, s, v, alpha, target) if x else None,
                                      "random_min_groups": min_crossed(x, s, v, alpha, target) if x else None,
                                      "fixed_power_at_available": float(fixed_power(x, b, s, v, alpha)),
                                      "random_power_at_available": crossed_power(x, b, s, v, alpha, 192)}
                              for key, x in (("planning", d), ("0p02_excess", .02), ("G1_observed_excess", ef))},
                "MC_at_planning_excess": monte_carlo(v, b, s, d, alpha, rng, reps)}
        result["scenario_proxies"][design] = design_result
    result["G1_more_seeds"] = {str(num): {ep: {
        "B131": crossed_power(.03 if ep == "accuracy" else .04, 131, num, variance(est), alpha, 192),
        "B16": crossed_power(.03 if ep == "accuracy" else .04, 16, num, variance(est), alpha, 192)}
        for ep, est in g1.items()} for num in (3, 5, 6, 8, 10, 12)}
    result["ten_seed_sensitivity"] = {}
    for name, scenario in result["scenario_proxies"].items():
        result["ten_seed_sensitivity"][name] = {}
        for endpoint, entry in scenario.items():
            excess = entry["scenarios"]["planning"]["excess"]
            v = entry["variance_components_at_group_level"]
            b = entry["independent_groups"]
            result["ten_seed_sensitivity"][name][endpoint] = {
                "power_at_available": crossed_power(excess, b, 10, v, alpha, 192),
                "random_min_groups": min_crossed(excess, 10, v, alpha, target),
            }
    return result


def run():
    result_dir = SOURCE / "results"
    summary = read_json(result_dir / "summary_master.json")
    lock = read_json(SOURCE / "PROTOCOL_LOCK.json")
    q3_result, score_matrices, parity = {}, {}, {}
    with np.load(result_dir / "blind_q3_scores.npz", allow_pickle=False) as q3, \
         np.load(result_dir / "blind_q2_scores.npz", allow_pickle=False) as q2:
        seeds = q3["seeds"].tolist()
        close(seeds, summary["seeds"], "seed IDs", atol=0)
        t3, t2 = q3["temps_q3"], q2["temps_q2"]
        for q, temp in (("q3", t3), ("q2", t2)):
            close(temp, summary[f"locked_temps_{q}"], f"{q} summary temperatures", atol=0)
            lock_temp = [[lock["checkpoints"][f"{a}_s{s}"][f"temp_{q}"] for s in seeds] for a in ARMS]
            close(temp, lock_temp, f"{q} protocol temperatures", atol=0)
        for name in (*stage3.CELL_ORDER, "H_c4_035"):
            q3_result[name], score_matrices[name] = q3_analysis(q3, name, t3)
            if name not in summary["cells"]:
                continue
            keys = ("raw_ll", "codes", "targets", "direct_table_probs", "full_iq_probs")
            original = stage3.evaluate_cell_paired(q3[f"{name}__raw_ll"], t3,
                *(q3[f"{name}__{k}"] for k in keys[1:]), seeds, is_primary=name == stage3.CELL_ORDER[0])
            expected = summary["cells"][name]
            parity[name] = check_tree({**stage3.CELL_META[name], **original}, expected, name)
            for arm in ARMS:
                independent = q3_result[name]["arms"][arm]["metrics"]["dev"]
                close(list(independent.values()), [expected["arms"][arm][k] for k in METRICS], name + arm)
            for endpoint, key in (("accuracy", "acc_gain_t95"), ("nll", "nll_improv_t95")):
                block_adj = score_matrices[name][endpoint].mean(axis=1)
                check_tree(stage3.paired_t_ci(block_adj), expected["paired_phase_minus_zero"][key], name + endpoint)
        primary = stage3.CELL_ORDER[0]
        f, f_factorial = work_f(q3, primary, t3, seeds, True)
        hist, _ = work_f(q3, "H_c4_035", t3, seeds, False)
        parity["work_f_primary_040"] = check_tree(f, summary["work_f_primary_040"], "Work F primary")
        parity["work_f_historical_035"] = check_tree(hist, summary["work_f_historical_035"], "Work F historical")
        work_g = stage3.work_g_rotation_and_controls(q3[f"{primary}__raw_ll"], t3,
            *(q3[f"{primary}__{k}"] for k in ("codes", "targets", "rot10_codes", "rot10_targets",
              "indep_raw_ll", "indep_targets", "shuffled_perm")), seeds)
        parity["work_g_primary_040"] = check_tree(work_g, summary["work_g_primary_040"], "Work G")
        for a in ARMS:
            for k, field in (("arith", "offline_group_avg_16way"), ("F0", "offline_representative_F0")):
                independent = q3_result[primary]["arms"][a]["metrics"][k]
                close([independent["acc"], independent["nll"]], [work_g["arms"][a][field][x]
                      for x in ("acc", "nll")], f"independent Work G {a} {k}")
        q2_result = q2_analysis(q2, t2)
        q2_original = stage3.work_a_q2_analysis(q2, t2, seeds)
        parity["work_a_q2_c4"] = check_tree(q2_original, summary["work_a_q2_c4"], "Q2 Work A")
        for control, entry in q2_result["controls"].items():
            for a in ARMS:
                replay = q2_original["arms"][a]["by_arm"][control]
                close(list(entry["arms"][a]["dev"].values()), [replay[k] for k in METRICS], f"independent Q2 {control} {a}")
        inventory = {"q3": {k: list(q3[k].shape) for k in q3.files},
                     "q2": {k: list(q2[k].shape) for k in q2.files}}
    report = {"runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
              "Q3": q3_result, "Q2": q2_result, "npz_shapes": inventory,
              "summary_checks_max_absolute_difference": parity,
              "work_f_primary_040": f, "work_f_historical_035": hist,
              "work_f_intervention_factorial": f_factorial,
              "power": power_analysis(score_matrices, lock["dev_pilot"])}
    destination = HERE / "results_m2.json"
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{destination.name} written. Maximum in-scope summary difference: {max(parity.values()):.3g}.")


if __name__ == "__main__":
    run()
```

### Event #12: `item.completed` — `command_execution` (`item_9`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '32 |131|C4|40,960|claim|hold' research/stage4_math_and_design/M4/THEOREM_M4.md | Select-Object -Last 36; rg -n 'f708|futility' research/stage4_math_and_design/M4/results_m4.json research/stage4_math_and_design/M6/results_m6.json | Select-Object -First 20"
```

**Aggregated Output:**

```text
110:For `s17`, along a fixed nonzero canonical ray \(x=cu\), \(c>0\), the sum sector is scale invariant. At low gain \(\tanh(cuW_1+b_1)\to\tanh b_1\), so the local code becomes constant if the winning cross-entropy score has a positive margin. The hidden deviation is at most \(c\|uW_1\|\). Converting it through \(W_2\) and the \(1/2\)-Lipschitz softmax, the bound on pairwise score changes is \((c/2)\|uW_1\|\|W_2\|_{2\to2}\|\log C_i-\log C_j\|\) for centers \(C_i,C_j\). A strict margin greater than this perturbation proves stability. For high gain, neuron \(j\) with \(d_j=(uW_1)_j\ne0\) approaches \(\operatorname{sgn}(d_j)\) with error at most \(2e^{-2(c|d_j|-|b_{1j}|)}\) once \(c|d_j|>|b_{1j}|\); neurons with zero projection stay at \(\tanh b_{1j}\). Apply the same score-margin test to the limiting logits and center choices. This proves radial saturation on each interior ray with strict margin. The four turns can remain informative. At exact all-zero input the encoder chooses its defined tie behavior. `verify_m4.py` finds `[1,5,9,13]` in the low-gain limit and `[0,4,8,12]` on its high-gain ray. Ties, near-orthogonal directions, and random scales near zero preclude a global AGC threshold.
175:- Stage 3D measured `1.9757 s / optimizer step` and `23.88 ms / logical eval row` on one A100-80GB. Train and eval peaks were `52,972.5 MB` at microbatch 16 and `45,409.9 MB` at eval batch 64. The six 1,280-step runs took `2,884.0–3,217.8 s` including additional work. BF16 weights alone are about 8 GB at 4B parameters. Train activation and batched candidate workspace amplify this. The baseline plan uses A100-80GB for both. Accept the pilot run only if peak allocated/reserved fits a predeclared 65 GiB A100 threshold and sustained times meet `≤2.50 s/step`, `≤0.030 s/row` inclusive of in-batch data feeding. Rerun timing after any microbatch change. A100-40GB, L4 and A10/A10G require a separate shape and kernel check with smaller batches. The measured Stage 3D peaks exceed their VRAM. For a substitution, require equivalent logits, a ≤20 GiB peak on 24 GiB cards, and throughput below `0.030*(0.00074684/(GPU rate + 0.00005284))` seconds per row to match A100 costs, as well as a preapproved wall limit. This is about 81.5 ms on L4 and 62.4 ms on A10, with 2 cores and 12 GiB host RAM. Keep the confirmed A100 path if the smaller card would require quantizing the frozen host or invalidating kernels. [Stage 3D report](../../stage3d_qwen_signed_phase/REPORT.md) · [Modal GPU SKUs](https://modal.com/docs/guide/gpu)
177:- Synthetic blind extrapolation has 32 blocks × 64 mothers and holds cumulative phase beyond `π/2`, e.g. `θ=±{1.70,2.10,2.50,2.90}` rad, four independently seeded blocks per signed setting. Use lag families such as `(h,ℓ)=(2,2),(2,3)` counterbalanced within block, including both signs and unseen magnitudes. Use the same mother order and true signed synthetic condition metadata for all arms. Fit on the Stage 3D support or narrower, with separate seeds; known synthetic metadata is an intervention. Each synthetic mother contributes Q3 pre/post and Q2 `none/one/two`, for `40,960` logical rows across four arms. Include CPU first and second pilot likelihood ceilings and score-matched RefPool diagnostics. Estimate angles from allowed sync/pilot evidence for real data.
178:- Natural real work per arm is `3,680×(Q1 + 3×Q2) + 2,960×(2×Q3) = 20,640` rows, hence `82,560` across arms. Stress adds `(7×140 + 1,080)×4 = 8,240` rows. Dev adds `16,000`. A logical row already includes scoring all four candidate answers. Registered CPU physical arms score the identical rows but consume zero GPU time. Scope and statistical interpretation exclude prompt-level claims about rare jumps or undisclosed source domains that have no registered observations.
188:| A100 stress, `ceil(8,240×0.030)` | 248 | 0.18521632 |
189:| A100 synthetic, `ceil(40,960×0.030)` | 1,229 | 0.91786636 |
213:| +2 dB, clean gain/DC/sideband | 20.74% | 19.98% | 19.98% | 19.98% | 0.532 vs 0.532 |
research/stage4_math_and_design/M6/results_m6.json:682:    "predictive_futility_threshold": 0.1,
research/stage4_math_and_design/M6/results_m6.json:686:    "futility_z": -0.2704380422162607,
research/stage4_math_and_design/M6/results_m6.json:714:        "futility_count": 348,
research/stage4_math_and_design/M6/results_m6.json:735:        "futility_count": 545,
research/stage4_math_and_design/M6/results_m6.json:756:        "futility_count": 93,
research/stage4_math_and_design/M4/results_m4.json:62694:          "filename": "f708_snr20_QPSK.tar.gz",
research/stage4_math_and_design/M4/results_m4.json:62696:          "capture_id": "f708_snr20_QPSK",
research/stage4_math_and_design/M4/results_m4.json:62716:          "download_url": "https://zenodo.org/api/records/4895133/files/f708_snr20_QPSK.tar.gz/content"
research/stage4_math_and_design/M4/results_m4.json:62719:          "filename": "f708_snr10_QPSK.tar.gz",
research/stage4_math_and_design/M4/results_m4.json:62721:          "capture_id": "f708_snr10_QPSK",
research/stage4_math_and_design/M4/results_m4.json:62741:          "download_url": "https://zenodo.org/api/records/4895133/files/f708_snr10_QPSK.tar.gz/content"
research/stage4_math_and_design/M4/results_m4.json:62744:          "filename": "f708_snr0_QPSK.tar.gz",
research/stage4_math_and_design/M4/results_m4.json:62746:          "capture_id": "f708_snr0_QPSK",
research/stage4_math_and_design/M4/results_m4.json:62766:          "download_url": "https://zenodo.org/api/records/4895133/files/f708_snr0_QPSK.tar.gz/content"
```

### Event #13: `item.completed` — `command_execution` (`item_10`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$j=Get-Content -Raw research/stage4_math_and_design/M2/results_m2.json | ConvertFrom-Json; $j.power.variance_components.G1_main_extrap_both | ConvertTo-Json -Depth 5; git status --short --untracked-files=normal; uv python list --only-installed'
```

**Aggregated Output:**

```text
{
    "accuracy":  {
                     "B":  131,
                     "S":  3,
                     "mean":  0.04392626166242578,
                     "block_conditional_sd":  0.016436968634732152,
                     "across_seed_means_sd":  0.01938021265344532,
                     "seed_means":  [
                                        0.06615776081424937,
                                        0.030593988549618322,
                                        0.03502703562340966
                                    ],
                     "mean_squares":  [
                                          0.0008105218136975055,
                                          0.049202636166551836,
                                          0.000582952311272251
                                      ],
                     "untruncated_components":  [
                                                    7.5856500808418146E-05,
                                                    0.0003711426248494625,
                                                    0.000582952311272251
                                                ],
                     "var_block":  7.5856500808418146E-05,
                     "var_seed":  0.0003711426248494625,
                     "var_interaction":  0.000582952311272251,
                     "seed_sd_95_upper":  0.08557132200882683
                 },
    "nll":  {
                "B":  131,
                "S":  3,
                "mean":  0.035650002550707036,
                "block_conditional_sd":  0.015555378401255893,
                "across_seed_means_sd":  0.02167594178554355,
                "seed_means":  [
                                   0.04864490167869194,
                                   0.047678100206173316,
                                   0.010627005767255822
                               ],
                "mean_squares":  [
                                     0.0007259093916187751,
                                     0.06154988525002576,
                                     0.00036183827982413995
                                 ],
                "untruncated_components":  [
                                               0.0001213570372648784,
                                               0.0004670843280168063,
                                               0.00036183827982413995
                                           ],
                "var_block":  0.0001213570372648784,
                "var_seed":  0.0004670843280168063,
                "var_interaction":  0.00036183827982413995,
                "seed_sd_95_upper":  0.09570787625210025
            }
}
?? research/stage4_math_and_design/
cpython-3.13.0-windows-x86_64-none    C:\Users\李则徐\AppData\Local\Programs\Python\Python313\python.exe
cpython-3.12.7-windows-x86_64-none    C:\qazwsx\python.exe
cpython-3.9.19-windows-x86_64-none    C:\376381\miniconda3\python.exe
cpython-3.9.19-windows-x86_64-none    C:\114514\python.exe
```

### Event #14: `item.completed` — `agent_message` (`item_11`)

M4’s plan budgets four matched arms at one seed. M6’s continuation budget covers up to three seeds for the surviving design, without automatically funding retrained comparators. I’ll make matched comparator availability a condition of each causal claim, so extra seeds on one arm cannot broaden what the comparison establishes.

### Event #15: `item.completed` — `command_execution` (`item_12`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$ErrorActionPreference='"'Stop'
"'$m2=Join-Path (Get-Location) '"'research/stage4_math_and_design/M2'
"'$env:UV_CACHE_DIR=Join-Path $m2 '"'.uv-cache-r2'
"'$env:UV_OFFLINE='"'1'
"'$env:UV_NO_PROJECT='"'1'
"'$env:UV_LINK_MODE='"'copy'
"'$env:TMP=$env:UV_CACHE_DIR
$env:TEMP=$env:UV_CACHE_DIR
$c=Join-Path $env:LOCALAPPDATA '"'uv/cache'
foreach("'$rel in @('"'wheels-v6/pypi/numpy','wheels-v6/pypi/scipy','archive-v0/3Oxde2-Jkv6KGjnQ','archive-v0/cSnmXl2iWBpNAXGE','simple-v21')) {
  "'$dest=Join-Path $env:UV_CACHE_DIR $rel
  New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
  Copy-Item -LiteralPath (Join-Path $c $rel) -Destination $dest -Recurse -Force
}
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B -c "import sys,numpy,scipy; print(sys.version.split()[0],numpy.__version__,scipy.__version__)"
'
```

**Aggregated Output:**

```text
Installed 2 packages in 4.73s
3.12.7 2.3.5 1.17.0
```

### Event #16: `item.completed` — `command_execution` (`item_13`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$ErrorActionPreference='"'Stop'
"'$path='"'research/stage4_math_and_design/M2/verify_m2.py'
"'$content=Get-Content -Raw -Encoding UTF8 -LiteralPath $path
$insertion=@'"'
def dev_futility_check():
    \"\"\"Independent blind null with dev-only continuation and a dev-chosen plan.\"\"\"
    information_proxy = .18
    prior_var = 4.0
    prediction_cut = .10
    legacy_cutoff = -.2704380422162607
    alpha_levels = (.05 / 3, .05)
    posterior_var = 1 / (1 / prior_var + information_proxy)
    posterior_slope = np.sqrt(information_proxy) * posterior_var

    # The final standardized score is independent of the dev statistic given the parameter.
    independent_cutoff = ((norm.isf(alpha_levels[-1]) -
                           norm.isf(prediction_cut) * np.sqrt(1 + posterior_var)) / posterior_slope)
    rng = np.random.default_rng(SEED + 1)
    reps = 400_000
    dev = rng.standard_normal(reps)
    numerator = rng.standard_normal(reps)
    # Dev alone chooses the evaluation plan in this check. Generate an exact pivot
    # for each chosen design without observing any blind training or scores.
    use_three = dev >= .8
    df = np.where(use_three, 130, 24)
    blind = numerator / np.sqrt(rng.chisquare(df) / df)
    policies = {}
    for label, cutoff in ((\"M6_legacy_cutoff_used_as_dev_only_screen\", legacy_cutoff),
                          (\"independent_normal_predictive_example\", independent_cutoff)):
        go = dev >= cutoff
        prob_go = float(norm.sf(cutoff))
        row = {\"dev_Z_threshold\": float(cutoff), \"prob_continue_at_dev_boundary\": prob_go,
               \"simulated_continue_count\": int(go.sum()), \"alpha_checks\": {}}
        for alpha in alpha_levels:
            reject = blind > t.isf(alpha, df)
            conditional_rate = float(np.mean(reject[go]))
            marginal_rate = float(np.mean(reject & go))
            mc_se = float(np.sqrt(alpha * (1 - alpha) / go.sum()))
            assert abs(conditional_rate - alpha) <= 6 * mc_se
            last = {}
            for s, plan_rows in ((1, ~use_three), (3, use_three)):
                idx = go & plan_rows
                rate = float(np.mean(reject[idx]))
                branch_se = np.sqrt(alpha * (1 - alpha) / idx.sum())
                assert abs(rate - alpha) <= 6 * branch_se
                last[str(s)] = {\"simulated_tested\": int(idx.sum()), \"size_MC\": rate,
                                \"size_exact\": float(t.sf(t.isf(alpha, df[idx][0]), df[idx][0]))}
                close(last[str(s)][\"size_exact\"], alpha, \"conditional plan null size\")
            row[\"alpha_checks\"][str(alpha)] = {
                \"conditional_size_exact\": alpha, \"conditional_size_MC\": conditional_rate,
                \"conditional_size_MC_SE\": mc_se, \"joint_false_promotion_exact\": alpha * prob_go,
                \"joint_false_promotion_MC\": marginal_rate,
                \"joint_false_promotion_at_blind_boundary\": alpha * prob_go, \"by_selected_seed_count\": last,
                \"invalid_same_score_screen_conditional_size\": min(alpha, prob_go) / prob_go,
            }
            assert abs(marginal_rate - alpha * prob_go) <= 6 * np.sqrt(alpha * prob_go * (1 - alpha * prob_go) / reps)
        policies[label] = row
    return {\"seed\": SEED + 1, \"reps\": reps, \"dev_information_proxy_for_planning\": information_proxy,
            \"prior_variance\": prior_var, \"predictive_continue_probability_threshold\": prediction_cut,
            \"predictive_calculation_final_alpha\": alpha_levels[-1], \"policies\": policies,
            \"size_boundary_assumptions\": \"Gaussian t pivots conditional on each locked plan, independent dev and blind innovations.\"}


def conditional_modal_power(primary, pilot):
    \"\"\"G1 proxy curves for two historical relative endpoints under fixed connectors.\"\"\"
    alpha = .05 / 3
    target = .95
    excesses = [0, .005, .01, .015, .02, .025, .03, .04, .05, .06, .08, .10]
    result = {\"first_step_Holm_alpha\": alpha, \"per_endpoint_target\": target,
              \"planning_margin_against_zero\": .02, \"common_excess_grid\": excesses,
              \"target_LCB_excess_units\": {\"accuracy\": \"proportion\", \"nll\": \"nats\"},
              \"inputs\": {}, \"designs\": {}}
    designs = [
        (\"synthetic_32_homogeneous_proxy\", 32, 31, 1, 0),
        (\"synthetic_32_eight_balanced_strata\", 32, 24, 1, 0),
        (\"synthetic_131_homogeneous_proxy\", 131, 130, 1, 0),
        (\"Pluto_16_block_conditional_proxy\", 16, 15, 1, 0),
    ]
    for groups_per_capture, name in ((8, \"Q2_loops\"), (4, \"Q3_loop_pairs\")):
        designs.extend((f\"CTTC_9_captures_{name}_ICC_{rho:.2f}\", 9, 8, groups_per_capture, rho)
                       for rho in (0, .10, .25))
    for name, b, df, groups_per_unit, rho in designs:
        entry = {\"independent_test_units\": b, \"student_t_df\": df, \"groups_per_unit\": groups_per_unit,
                 \"group_effect_session_ICC\": rho, \"endpoints\": {}}
        cutoff = t.isf(alpha, df)
        standardized_target = brentq(lambda nc: nct.sf(cutoff, df, nc) - target, 0, 30)
        close(nct.sf(cutoff, df, 0), alpha, \"conditional power at boundary\")
        for endpoint, block_seed in primary.items():
            vc = components(block_seed)
            v_block, v_seed, v_interaction = (vc[k] for k in (\"var_block\", \"var_seed\", \"var_interaction\"))
            plan_excess = .03 if endpoint == \"accuracy\" else .04
            pilot_sd = pilot[\"work_e_scenario\"][\"sd\"][endpoint]
            result[\"inputs\"][endpoint] = {
                \"vc_stage3d_G1\": [v_block, v_seed, v_interaction],
                \"stage3_work_E_conservative_block_sd\": pilot_sd,
                \"observed_seed_202611_block_sd\": float(np.std(block_seed[:, 0], ddof=1)),
                \"observed_seed_202611_effect_above_margin\": float(block_seed[:, 0].mean() - .02),
                \"observed_three_seed_bundle_effect_above_margin\": float(block_seed.mean() - .02),
                \"assumed_planning_effect_above_margin\": plan_excess,
            }
            data = {}
            for s in (1, 3):
                frozen = block_seed[:, :s].mean(axis=1)
                obs_excess = float(frozen.mean() - .02)
                deff = rho + (1 - rho) / groups_per_unit
                # ICC belongs to the common group effect in the crossed proxy.
                proxy_sd = np.sqrt(v_block * deff + v_interaction / (s * groups_per_unit))
                # Work E supplies only an aggregate envelope, with no seed/ICC decomposition.
                work_e_sd = pilot_sd * np.sqrt(deff)
                readings = {}
                for label, sd in ((\"G1_variance_components_proxy\", proxy_sd),
                                  (\"legacy_Work_E_high_variance_sensitivity\", work_e_sd)):
                    scale = np.sqrt(b) / sd
                    curve = nct.sf(cutoff, df, np.asarray(excesses) * scale)
                    assert np.all(np.diff(curve) >= -1e-12)
                    min_excess = standardized_target / scale
                    close(nct.sf(cutoff, df, min_excess * scale), target, \"power threshold inversion\")
                    if df == b - 1 and label == \"G1_variance_components_proxy\":
                        collapsed = (v_block * deff, v_seed, v_interaction / groups_per_unit)
                        close(nct.sf(cutoff, df, plan_excess * scale),
                              fixed_power(plan_excess, b, s, collapsed, alpha), \"conditional power reuse\")
                    readings[label] = {\"unit_sd\": float(sd), \"power_curve\": curve.tolist(),
                                       \"power_at_planning_excess\": float(nct.sf(cutoff, df, plan_excess * scale)),
                                       \"power_at_historical_observed_excess\": float(nct.sf(cutoff, df, obs_excess * scale)),
                                       \"minimum_excess_for_95_percent_power\": float(min_excess)}
                data[str(s)] = {\"historical_effect_above_margin_for_frozen_set\": obs_excess, **readings}
            entry[\"endpoints\"][endpoint] = data
        result[\"designs\"][name] = entry
    return result



'@
"'$report='"'              \"power\": power_analysis(score_matrices, lock[\"dev_pilot\"])}'
if(-not "'$content.Contains($report)){throw '"'Missing report anchor'}
"'$content=$content.Replace('"'def run():',"'$insertion+([string][char]10)+'"'def run():')
"'$updated=@'"'
              \"power\": power_analysis(score_matrices, lock[\"dev_pilot\"]),
              \"round2\": {\"dev_disjoint_futility\": dev_futility_check(),
                         \"conditional_modal_power\": conditional_modal_power(
                             score_matrices[stage3.CELL_ORDER[0]], lock[\"dev_pilot\"])}}
'@
"'$content=$content.Replace($report,$updated)
Set-Content -Encoding UTF8 -NoNewline -LiteralPath $path -Value $content
'
```

### Event #17: `item.completed` — `command_execution` (`item_14`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$p='"'research/stage4_math_and_design/M2/verify_m2.py'
"'$content=Get-Content -Raw -Encoding UTF8 -LiteralPath $p
$content=$content.Replace('"'\"joint_false_promotion_MC\": marginal_rate,','\"joint_false_promotion_MC\": marginal_rate, \"by_selected_seed_count\": last,')
"'$content=$content -replace '"'(?m)"'^'"\\s*\"joint_false_promotion_at_blind_boundary\": alpha \\* prob_go, \"by_selected_seed_count\": last,\\r?\\n',''
Set-Content -LiteralPath "'$p -Encoding UTF8 -NoNewline -Value $content
$ErrorActionPreference='"'Stop'
"'$m2=Join-Path (Get-Location) '"'research/stage4_math_and_design/M2'
"'$env:UV_CACHE_DIR=Join-Path $m2 '"'.uv-cache-r2'
"'$env:UV_OFFLINE='"'1'
"'$env:UV_NO_PROJECT='"'1'
"'$env:UV_LINK_MODE='"'copy'
"'$env:TMP=$env:UV_CACHE_DIR
$env:TEMP=$env:UV_CACHE_DIR
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M2/verify_m2.py'
```

**Aggregated Output:**

```text
results_m2.json written. Maximum in-scope summary difference: 0.
```

### Event #18: `item.completed` — `command_execution` (`item_15`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "@'


## 6. Round 2 claim scope and budget lock

This section sets the Round 2 promotion rules. Sections 1–5 retain the retrospective Stage 3D replay and its original power assumptions. Freeze the applicable branch, training mixture, checkpoints, comparators, row manifest, scoring and weights before Stage 4 blind scoring. Use fresh synthetic generator namespaces. Stage 3D serves as development evidence for this protocol.

| Tier | Admissible conclusion | Eligibility |
|---|---|---|
| 1. Single frozen connector per trained arm | At seed 202611, these checkpoint/readout pairs satisfy the gate on the registered target population, conditional on training and dev selection. | M4's four matched 384-step fits and one prespecified blind evaluation after dev eligibility. An incomplete M6 pilot gives engineering evidence. A one-seed confirmatory fallback requires its complete arm roster, trained checkpoints and plan to be selected on dev before blind access. |
| 2. Frozen connector panel | The paired mean loss over the selected \\(S=2\\) or \\(3\\) frozen seed pairs satisfies the gate, conditional on the selected checkpoints. | M6 continuation targets \\(S=3\\). Freeze the panel and average losses or hits within block. Report each seed. A deployed probability ensemble has its own estimand and must be frozen before scoring. Extra survivor-only seeds may be evaluated against deterministic controls. Restrict matched trained-arm contrasts to their common preselected paired seeds. |
| 3. Population of retrains | The locked training recipe has expected benefit over fresh training runs and the registered blind sampling process. | Require at least ten independently paired retrains and adequate prospective all-endpoint power under the crossed model or a validated alternative. Ten is the present protocol floor based on G1, with no universal power guarantee. Fewer seeds are ineligible for Tier 3 under this protocol. |

M4 estimates \\(\\"'$8.44073416'"\\), including four single-seed connector fits at \\(\\"'$2.86786560'"\\). M6's alternative envelope is \\(\\"'$9.50'"\\) against the historical \\(\\"'$12.88'"\\) balance, including \\(\\"'$0.491218'"\\) for the \\(\\le173/960\\)-step pilot and \\(\\"'$5.101110'"\\) for continuation to three seeds. These are alternative envelopes. Reprice any combined plan against the single \\(\\"'$9.50'"\\) ceiling before release. Mixed synthetic/real fitting and synthetic-only fitting with real transfer are distinct frozen recipes. Both budget routes target zero net spend under the account guard. This verification used local CPU. [M4 compute envelope](../M4/THEOREM_M4.md#3-frozen-compute-envelope-and-dispatch) · [M6 pilot and budget](../M6/THEOREM_M6.md)

### Conditional validity after the 18% dev screen

Let \\(\\mathcal F\\) contain training and dev data and randomness, pilot decision, selected checkpoints, seed panel, calibration, comparator selection, margins, sampling and scoring plans. Set \\(A=\\{\\text{continue and eligible}\\}\\in\\mathcal F\\). The selected-path target is the conditional expected blind gain \\(\\mu_{\\mathcal F}\\). Under the balanced additive model from §3,
\\[
 \\mu_{\\mathcal F}=\\mu+\\overline{\\nu}_{\\mathcal F},\\qquad
 \\operatorname{Var}(\\bar D\\mid\\mathcal F)=(\\sigma_u"'^2+'"\\sigma_e"'^2/S)/B.
'"\\]
This assumes conditional independence of blind block innovations from the development selection. The frozen seed effects remain in the conditional mean. Crossed inference includes \\(\\sigma_\\nu"'^2/S'"\\) in the sampling variance. Dev selection can also change the sampled seed population, so a Tier 3 design must lock that population and account for specified training failures.

Require \\(P(p_{\\rm blind}\\le a\\mid\\mathcal F=f)\\le a\\) at every eligible \\(f\\) with \\(\\mu_f\\le\\delta\\). The one-sided Student test attains equality at the boundary under its Gaussian and independence conditions. A sharp-null randomization test uses its justified conditional exchangeability orbit. Give a stopped claim \\(p=1\\). For any \\(a<1\\), iterated expectation yields
\\[
 P(A,\\ \\mu_{\\mathcal F}\\le\\delta,\\ p_{\\rm blind}\\le a)
 \\le a P(A,\\ \\mu_{\\mathcal F}\\le\\delta).
\\]
With exact continuous boundary p-values this gives conditional size \\(a\\) among boundary-null survivors and joint false-promotion probability \\(aP(A)\\). Conditional Holm over the true selected-path claims then gives unconditional FWER at most 0.05. Dev choices may govern futility and select a fully specified plan. The confirmatory rejection is made once on the blind data. All three claim slots remain in the family.

M6's \\(t=0.18\\) is a planning input. The training fraction \\(173/960\\) does not determine information time. The \\(-0.270438\\) cutoff came from the correlated two-look model and may be frozen as a dev-only heuristic, with a single confirmatory blind look. For an independent alternative planning model, take \\(Z_d\\mid\\mu\\sim N(\\sqrt t\\mu,1)\\), \\(Z_b\\mid\\mu\\sim N(\\mu,1)\\) with independent innovations and prior \\(\\mu\\sim N(0,4)\\). The conditional mean and variance of \\(\\mu\\) are \\(\\sqrt t\\,z/(t+1/4)\\) and \\(1/(t+1/4)\\). The cutoff for predicted \\(P(Z_b>z_{.95}\\mid Z_d)\\ge0.10\\) is
\\[
 \\big(z_{.95}-z_{.90}\\sqrt{1+1/(t+1/4)}\\big)(t+1/4)/\\sqrt t.
\\]
Either cutoff obeys the conditional proof if the blind test is valid. The threshold controls resource allocation. Its usefulness and projected continuation power require dev calibration.

Disjoint row IDs are one part of the split. The proof also needs conditional test validity when dev and blind share a campaign, hardware state or deterministic TX source. Hold out the entire designated recording where required and justify independence given genuinely fixed conditioning variables. Outcome-linked score reports, variance, pathology and troubleshooting on blind rows can compromise selection. Keep timing and parity smoke tests on dev. Blind scorer failures follow the frozen fallback or stop rule. In the counterexample that uses the same standardized test score to screen and test, conditional error among survivors is \\(\\min(a,P(A))/P(A)\\).

## 7. Round 2 formal gate table

Let \\(q_X\\) be a frozen dev-calibrated forecast. Orient paired differences as \\(D_L(X,Y)=L(q_Y)-L(q_X)\\) and \\(D_A(X,Y)=A(q_X)-A(q_Y)\\) on identical cases. \\({\\rm LB}(D)>d\\) means a one-sided lower confidence bound above the margin at the claim's tested Holm level. Use equivalent component p-values in the testing tree. Every applicable row is mandatory. Exact controls and fixed-cohort guards get p-value 0 on pass and 1 on failure for IUT bookkeeping, with no population inference attached to the guard.

### Arm roles and causal contrasts

The four priced arms are **matched Zero**, **matched Naive-Additive**, **Arm 1 Host-Pure Equivariant**, and **Arm 2 Physical+HostResidual**. M1's **Arm 0** names the legacy separate-token Stage 3D-style reference. Identify it with the matched Zero comparator only if it satisfies the matching contract. Otherwise retain legacy Arm 0 as contextual replay and use a separate matched Zero implementation. Direct S17/Direct Physical and known-parameter Full IQ are analytic references. Full IQ is a finer-information synthetic diagnostic. Real raw-I/Q references obey the causal filtration and may be approximate.

| Gate row | Exact contrast or eligibility control |
|---|---|
| Arm matching | Match allowed observations and deadlines, frozen host, signed text, candidate strings, quotient/lift when isolating fusion, calibration opportunity, score floor, training manifest, optimizer and step budget. Zero uses the prescribed coupled-phase input ablation. Naive-Additive shares the inputs with a centered additive code/phase score in canonical coordinates. Verify its mixed code-by-phase term is zero. Changes to the quotient, head or training recipe create a package comparison. A fusion attribution requires the matched ablation. |
| Common absolute gate | Require NLL improvement \\(>0\\) over uniform and the dev prior. Require accuracy \\(>1/K+0.02\\) and improvement \\(>0\\) over the frozen prior classifier. Test both skill statistics against a justified locked random-label null. Require paired Brier improvement \\(>0.005\\) over the dev-selected lowest-Brier baseline from uniform, dev prior and matched Zero, and \\(>0\\) over every required baseline in that set apart from the candidate. Require a valid blind ECE upper bound \\(<0.05\\) on ten frozen top-label bins. Require native equivariance and projection idempotence max defect \\(\\le10"'^{-5}'"\\) on the declared group support. |
| NIG | Lock a same-information oracle or independently fitted physical reference, a positive denominator and \\({\\rm LB}(0.90H(Y)-L(q_X)+0.10L(q_{\\rm ref}))>0\\). A balanced synthetic target law has \\(H(Y)=\\log K\\). For real data predeclare the target class law or carry entropy uncertainty jointly. A causal approximate reader yields reference-normalized wording. An indefensible denominator makes the gate ineligible. |
| Arm 1 host-pure extrapolation | Against each of Zero and Naive-Additive require \\({\\rm LB}(D_L)>0.02\\) nats and \\({\\rm LB}(D_A)>0.02\\) on the locked extrapolation mixture, plus common absolute/NIG gates. Score the frozen host's candidate outputs with quotient alignment and scalar calibration. Coupled physical features can enter the soft tokens. Verify the output graph and absence of a direct physical logit path. |
| Arm 1 fixed-text intervention | At the same checkpoint, waveform/cache, signed text, scorer and scalar, require \\({\\rm LB}(D_L(\\mathrm{matched},\\mathrm{slot\\ swap}))>0\\) and \\({\\rm LB}(D_L(\\mathrm{matched},\\mathrm{slot\\ shuffle}))>0\\). The signed swap flips the slots. The shuffle is a separately seeded, frozen mapping of informative slot values within prespecified eligible strata, without labels. Keep signs/rotations with their mothers. Audit canonical centered code-by-phase mixed contrasts with the additive arm as the zero-interaction control. This establishes a response to the supplied slot. Waveform-based CFO estimation is a different hypothesis. |
| Arm 2 host residual value | With \\(q_2=\\operatorname{softmax}((\\ell_{\\rm phys}+\\gamma\\tanh\\ell_{\\rm host})/T_2)\\), require \\({\\rm LB}(D_L)>0.02\\) nats and \\({\\rm LB}(D_A)>0\\) against independently dev-calibrated Direct Physical with identical evidence and representation. Also require \\({\\rm LB}(D_L)>0\\) versus the same-stack residual-off forecast \\(\\operatorname{softmax}(\\ell_{\\rm phys}/T_2)\\). Apply both slot interventions only in the host residual branch. Require positive NLL lower bounds for both while \\(\\ell_{\\rm phys}\\) is bit-identical and \\(\\gamma,T_2\\) remain fixed. Apply common absolute/NIG gates to confirmatory Arm 2 release. This tests incremental host value over the specified physical reader. Arm 1 evidence must come from Arm 1. |
| Calibration | Freeze one dev-fitted scalar per checkpoint/arm and task or predeclared stratum, using the same predeclared search range and objective. Hash the calibrator, priors and adapter with the scores. Reuse the treatment's scalar for its ablations. A synthetic-to-real zero-shot route transfers the synthetic-dev fit and permits only registered causal sample state. A real-dev fit is a separate real-calibrated route and excludes the held-out capture. Choose the confirmatory route on dev. Oracle blind temperature and calibration sensitivity are retrospective diagnostics. |
| Q2 gain | Under the same frozen RefSufficientPool scorer require \\({\\rm LB}(L_{\\rm one}-L_{\\rm two})>0.02\\) nats and \\({\\rm LB}(L_{\\rm none}-L_{\\rm one})>0\\). One uses the registered first past release. Two includes the distinct second valid release. Q2 two-valid clears its absolute and reference-normalized gates. Report Direct Q2, accuracy/SER, bit NLL and EVM as secondary comparisons. |
| Q2 exact controls | Valid must give byte-identical preprocessing, fixed slots, masks, positions, candidate context and deterministic scorer outputs after appending a raw duplicate ID, independent-channel record or empty padding, or permuting the reference list. Require 0.0 max probability change, TV and KL for each pair. Verify canonical reuse at the real API boundary where used. Validate ID and samples before deduplication and fail conflicting IDs. A genuinely fresh acquisition with a new ID enters the two-valid efficacy test; superiority is expected across cases. |

With metadata \\(M\\) and authorized references \\(R\\) held fixed, a deterministic waveform cache \\(Z\\) and host forecast \\(Q\\) satisfy \\(I(Y;Q\\mid R,M)\\le I(Y;Z\\mid R,M)\\le I(Y;W\\mid R,M)\\). The Stage 3D loss ladder compares finite readouts and includes their regret. Conditional entropy gaps require Bayes readouts at each named information set. The residual-off and fixed-physical interventions compare architectures at the same information set. [M1 arm design](../discussions/M1/prompt_r2.md) · [M3 reference contract](../discussions/M3/prompt_r2.md) · [M6 DPI](../M6/THEOREM_M6.md)

'@ | Add-Content -LiteralPath 'research/stage4_math_and_design/M2/THEOREM_M2.md' -Encoding UTF8"
```

### Event #19: `item.completed` — `command_execution` (`item_16`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "@'


### Three-claim testing tree

Freeze one primary route and checkpoint panel per claim before blind access. Claim 1 defaults to Arm 1. A preregistered Arm 2 substitution requires the complete residual-value, absolute, Q2 and structural endpoints and yields an Arm 2 claim. Both arms can instead be required jointly by taking the maximum over both full endpoint lists. Independent confirmatory releases for either arm require extra hypothesis slots and revised multiplicity before acquisition.

| Claim | Mandatory endpoints and scope | Unit |
|---|---|---|
| 1. Synthetic extrapolation + Q2 | Require the locked extrapolation route's arm-specific, absolute, NIG, calibration and structural endpoints. Also require the Q2 ordered gain, absolute and exact controls. Use the M4 32-block equal-weight mixture of eight signed large-angle settings with four blocks per setting, or preregister a 131-block design with the same target mixture weights and the exact per-stratum weights/counts. Region-wide efficacy means the prespecified average. Per-setting claims are descriptive unless included in the IUT. The 131-block homogeneous G1 curve below is a power proxy, rather than a replay of that stratified design. | Independent generator blocks. Keep a mother's repeats and controls paired. Use the fixed-stratum weighted contrast where applicable. |
| 2. Pluto | Require the common absolute/calibration/structure and ordered Q2 gate on the natural rows. Release pilots 15 and 18. Include every registered lock failure with the common uniform fallback. Report lock coverage, stress results on 140 base frames, and Q1/Q3 diagnostics. | 2,240 frames in 16 chronological blocks from one campaign, with copies nested. Condition on campaign. Exact inference needs a valid conditional block model or a valid design randomization. If shared temporal dependence invalidates the model, report the campaign descriptively. |
| 3. CTTC + f708_snr10 | Require the same paired natural-data Q2, absolute and exact controls, pilots 126 and 127, and the single-capture transfer guard on f708_snr10. Require \\(L_{\\rm two}<L_{\\rm one}<L_{\\rm none}\\) and \\(L_{\\rm two}<L_{\\rm prior}\\) as fixed-cohort point inequalities there, complete reporting and zero integrity failures. Exclude this capture from checkpoint fitting, calibration and nuisance/reader/prior/hyperparameter tuning. Its causal sync and authorized pilots remain permitted. | 1,440 packets in 72 loops nested in nine captures, and 720 Q3 pairs in 36 disjoint loop-pair groups nested in the same captures. Average paired effects within capture, then weight captures equally. Report the held-out capture separately, plus leave-out results for the other eight. The holdout's p=0/1 bookkeeping is a cohort guard, with no standalone population test. |

Real primary Q2 uses paired symbol NLL for its ordered-evidence contrasts. Q1 relative-turn, real Q3 retransmission and stress metrics are secondary unless a full confirmatory endpoint set is appended to the relevant slot before scoring. Preserve every failed-lock row and apply the same fallback rules across arms. Validate identities across the regular and fallback paths. At each release, audit raw support and timestamp using the locked causal alignment and without consulting target labels or unreleased references.

CTTC replays one public cyclic TX sequence. Disjoint roles/splits prevent direct observation reuse. The common fixed test range \\([334096,455000)\\) can still induce source-ordinal dependence across captures. Scope the primary cohort estimand to the registered captures and fixed source. Justify independent capture innovations given the locked conditioning before nine-unit inference. If common residual randomness remains across captures and there is no valid design-based alternative, report descriptive results. Show source-ordinal aggregation, capture leave-one-out and capture-by-source sensitivity. The 36/72 nested groups contribute to capture means. They do not supply independent capture or new-source draws. [M4 benchmark and split](../M4/THEOREM_M4.md#2-hardware-benchmark-and-causality-contract)

For independent equal-weight Gaussian units with common variance the homogeneous t has \\(B-1\\) df. In the balanced 32-block synthetic model, fitting eight fixed cell means with common residual variance gives \\(32-8=24\\) df and the same \\(\\sigma/\\sqrt{32}\\) standard error for their average. For unequal allocational counts \\(n_h\\) with fixed weights \\(w_h\\), the common-variance standard error is \\(\\sigma\\sqrt{\\sum_h w_h"'^2/n_h}'"\\), with \\(B-H\\) residual df. For independent Gaussian capture means with fixed heterogeneous means and common variance, the sample variance denominator is noncentral chi-square and independent of the grand mean. It is stochastically larger than the central denominator, giving a conservative upper-tail test at the null boundary. Independent bounded unit means allow the Hoeffding test in §3 for mean margins if Gaussian assumptions are unsupported. Valid random-label p-values require sanctioned assignments under the relevant sharp null. Preserve rotations, repeats, captures and source restrictions in the exchangeability orbit. Use a registered randomized negative control when the existing label design lacks one. A paired sign-flip test needs actual randomization or a justified joint sign-symmetry null. A weak mean inequality alone supplies no such exchangeability.

For each claim \\(j\\), put \\(p_j=\\max_{k\\in K_j}p_{jk}\\) over mandatory tests and guards. The claim null is their union, so \\(p_j\\) is super-uniform whenever any endpoint null is true, conditional on the dev lock. Apply Holm to \\((p_1,p_2,p_3)\\) at \\(0.05/3,\\ 0.05/2,\\ 0.05\\), stopping at the first failure. Conditional on \\(\\mathcal F\\), the first rejected true claim, if any, meets a threshold at most \\(0.05/m_0\\), where \\(m_0\\) counts true claims. The union bound gives conditional FWER at most 0.05 under arbitrary claim dependence. Averaging over \\(\\mathcal F\\) retains the bound. Keep discontinued, ineligible and missing claims as p=1 in their original slots. An IUT divides no alpha inside a claim. Reporting a set of intervals with simultaneous coverage is a separate requirement.

## 8. Stage 6 compatibility

Freeze one task, one pre-question representation, one admissible action set per recording, and exactly one sealed read of equal cost. Give max EIG, uniform random, nearest and fixed best on dev the same available candidates, reader family, calibration opportunity and cutoff. Each policy commits its action and pre-reveal forecast before receiving the read. Seal nonchosen evidence from the policy. The blinded evaluator pairs policies by recording with recorded counterfactual acquisitions when valid. Otherwise preassign an action design with sufficient support and use its registered design-based estimator. Freeze tie breaking. Nearest uses the last authorized anchor with a causal distance rule. Fixed best on dev selects a single constant action (or a prespecified mask-availability fallback rule) by dev-only realized posterior NLL and freezes it. It cannot select based on blind performance or hidden ambiguity.

For the adaptive acquisition claim, use \\(D_L=L_{\\rm baseline}-L_{\\rm max\\,EIG}\\) and \\(D_A=A_{\\rm max\\,EIG}-A_{\\rm baseline}\\) on blinded post-reveal predictions, with both one-sided lower bounds strictly positive for **each** of random, nearest and fixed best on dev. Use one IUT on those six endpoints at the claim's registered level. Stratify or nest trial units at the recording level. A pre-reveal proper-score and seal audit tests commitment and integrity separately. For true post-reveal Bayes NLL with equal read costs, \\(V(z,a)=I(Y;R_a\\mid Z=z)\\). The adaptive-versus-best-static value is
\\[
 \\mathbb E_Z\\max_a V(Z,a)-\\max_a\\mathbb E_Z V(Z,a).
\\]
With a finite common action set this is strict iff every fixed action is strictly suboptimal on a set of positive probability. For varying masks, replace each constant by an eligible frozen availability-only policy and apply the analogous criterion to that policy class. Merely beating random and nearest does not exclude a static optimum. Accuracy has its own Bayes decision value and its own empirical contrast.

Treat 16-bit canonical versus 4-bit S17 as a distinct representation contrast paired on identical source records, existing evidence, target and prescribed reveal action. Use the same calibrated reader family and development opportunity and count the actual storage bits. Require positive NLL gain and positive accuracy gain if making a joint superiority claim. Require exact S17 prefix reconstruction and all registered symmetry checks. Repeat the full policy comparison at fixed 16-bit budget for an adaptive claim, so representation and action choice can be attributed separately. Full IQ is the diagnostic ceiling after matching allowed causal inputs. Predeclare both Stage 6 claim slots and use Holm within that family if claiming them separately. An optional portfolio-wide 0.05 rule opens the independent Stage 6 confirmatory family only after all three Stage 4 claims pass. If Stage 4 contains a true claim null, reaching Stage 6 already entails a Stage 4 error. If all Stage 4 nulls are false, conditional Stage 6 family control applies after selection. Without this sequential rule, report the families at their separate levels and refrain from a joint 0.05 claim. [M5 action and cache design](../discussions/M5/prompt_r2.md)

'@ | Add-Content -LiteralPath 'research/stage4_math_and_design/M2/THEOREM_M2.md' -Encoding UTF8"
```

### Event #20: `item.completed` — `command_execution` (`item_18`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$r=Get-Content -Raw -Encoding UTF8 '"'research/stage4_math_and_design/M2/results_m2.json' | ConvertFrom-Json
"'$r.round2.dev_disjoint_futility | ConvertTo-Json -Depth 12 -Compress
$r.round2.conditional_modal_power.inputs | ConvertTo-Json -Depth 6 -Compress
foreach($d in $r.round2.conditional_modal_power.designs.PSObject.Properties){
  $v=$d.Value
  $a=$v.endpoints.accuracy
  $n=$v.endpoints.nll
  "{0} {1}/{2} G1 plan acc {3:P3} {4:P3} NLL {5:P3} {6:P3} | d95 acc {7:F5} {8:F5} NLL {9:F5} {10:F5} | Work E plan acc {11:P3} NLL {12:P3} | G1 d=.01 acc {13:P3} {14:P3} NLL {15:P3} {16:P3}" -f $d.Name,$v.independent_test_units,$v.student_t_df,$a.'"'1'.G1_variance_components_proxy.power_at_planning_excess,"'$a.'"'3'.G1_variance_components_proxy.power_at_planning_excess,"'$n.'"'1'.G1_variance_components_proxy.power_at_planning_excess,"'$n.'"'3'.G1_variance_components_proxy.power_at_planning_excess,"'$a.'"'1'.G1_variance_components_proxy.minimum_excess_for_95_percent_power,"'$a.'"'3'.G1_variance_components_proxy.minimum_excess_for_95_percent_power,"'$n.'"'1'.G1_variance_components_proxy.minimum_excess_for_95_percent_power,"'$n.'"'3'.G1_variance_components_proxy.minimum_excess_for_95_percent_power,"'$a.'"'1'.legacy_Work_E_high_variance_sensitivity.power_at_planning_excess,"'$n.'"'1'.legacy_Work_E_high_variance_sensitivity.power_at_planning_excess,"'$a.'"'1'.G1_variance_components_proxy.power_curve[2],"'$a.'"'3'.G1_variance_components_proxy.power_curve[2],"'$n.'"'1'.G1_variance_components_proxy.power_curve[2],"'$n.'"'3'.G1_variance_components_proxy.power_curve[2]
}
"
```

**Aggregated Output:**

```text
{"seed":2026413,"reps":400000,"dev_information_proxy_for_planning":0.18,"prior_variance":4.0,"predictive_continue_probability_threshold":0.1,"predictive_calculation_final_alpha":0.05,"policies":{"M6_legacy_cutoff_used_as_dev_only_screen":{"dev_Z_threshold":-0.2704380422162607,"prob_continue_at_dev_boundary":0.6065883617134366,"simulated_continue_count":242598,"alpha_checks":{"0.016666666666666666":{"conditional_size_exact":0.016666666666666666,"conditional_size_MC":0.016290323910337266,"conditional_size_MC_SE":0.0002599148823858683,"joint_false_promotion_exact":0.010109806028557275,"joint_false_promotion_MC":0.00988,"by_selected_seed_count":{"1":{"simulated_tested":157817,"size_MC":0.01603122604028717,"size_exact":0.016666666666666677},"3":{"simulated_tested":84781,"size_MC":0.01677262594213326,"size_exact":0.01666666666666666}},"invalid_same_score_screen_conditional_size":0.02747607392200562},"0.05":{"conditional_size_exact":0.05,"conditional_size_MC":0.050255979027032376,"conditional_size_MC_SE":0.0004424897313880418,"joint_false_promotion_exact":0.030329418085671828,"joint_false_promotion_MC":0.03048,"by_selected_seed_count":{"1":{"simulated_tested":157817,"size_MC":0.05000095046794705,"size_exact":0.05},"3":{"simulated_tested":84781,"size_MC":0.050730706172373526,"size_exact":0.04999999999999997}},"invalid_same_score_screen_conditional_size":0.08242822176601687}}},"independent_normal_predictive_example":{"dev_Z_threshold":-0.7015648862983981,"prob_continue_at_dev_boundary":0.7585247219431419,"simulated_continue_count":303543,"alpha_checks":{"0.016666666666666666":{"conditional_size_exact":0.016666666666666666,"conditional_size_MC":0.016600613422151064,"conditional_size_MC_SE":0.0002323617530814924,"joint_false_promotion_exact":0.012642078699052366,"joint_false_promotion_MC":0.0125975,"by_selected_seed_count":{"1":{"simulated_tested":218762,"size_MC":0.016533950137592453,"size_exact":0.016666666666666677},"3":{"simulated_tested":84781,"size_MC":0.01677262594213326,"size_exact":0.01666666666666666}},"invalid_same_score_screen_conditional_size":0.021972476551549994},"0.05":{"conditional_size_exact":0.05,"conditional_size_MC":0.05043107566308563,"conditional_size_MC_SE":0.0003955821565971026,"joint_false_promotion_exact":0.0379262360971571,"joint_false_promotion_MC":0.03827,"by_selected_seed_count":{"1":{"simulated_tested":218762,"size_MC":0.05031495415108657,"size_exact":0.05},"3":{"simulated_tested":84781,"size_MC":0.050730706172373526,"size_exact":0.04999999999999997}},"invalid_same_score_screen_conditional_size":0.06591742965464999}}}},"size_boundary_assumptions":"Gaussian t pivots conditional on each locked plan, independent dev and blind innovations."}
{"accuracy":{"vc_stage3d_G1":[7.5856500808418146E-05,0.0003711426248494625,0.000582952311272251],"stage3_work_E_conservative_block_sd":0.09441431300390846,"observed_seed_202611_block_sd":0.02883420477582685,"observed_seed_202611_effect_above_margin":0.04615776081424937,"observed_three_seed_bundle_effect_above_margin":0.02392626166242578,"assumed_planning_effect_above_margin":0.03},"nll":{"vc_stage3d_G1":[0.0001213570372648784,0.0004670843280168063,0.00036183827982413995],"stage3_work_E_conservative_block_sd":0.07979465834252315,"observed_seed_202611_block_sd":0.0171797858611482,"observed_seed_202611_effect_above_margin":0.02864490167869194,"observed_three_seed_bundle_effect_above_margin":0.015650002550707035,"assumed_planning_effect_above_margin":0.04}}
synthetic_32_homogeneous_proxy 32/31 G1 plan acc 99.999% 100.000% NLL 100.000% 100.000% | d95 acc 0.01778 0.01139 NLL 0.01523 0.01078 | Work E plan acc 34.604% NLL 72.682% | G1 d=.01 acc 49.811% 88.217% NLL 63.716% 91.520%
synthetic_32_eight_balanced_strata 32/24 G1 plan acc 99.998% 100.000% NLL 100.000% 100.000% | d95 acc 0.01799 0.01152 NLL 0.01541 0.01090 | Work E plan acc 33.907% NLL 71.652% | G1 d=.01 acc 48.874% 87.454% NLL 62.673% 90.879%
synthetic_131_homogeneous_proxy 131/130 G1 plan acc 100.000% 100.000% NLL 100.000% 100.000% | d95 acc 0.00854 0.00547 NLL 0.00731 0.00517 | Work E plan acc 93.015% NLL 99.981% | G1 d=.01 acc 98.905% 100.000% NLL 99.879% 100.000%
Pluto_16_block_conditional_proxy 16/15 G1 plan acc 98.521% 100.000% NLL 100.000% 100.000% | d95 acc 0.02627 0.01682 NLL 0.02250 0.01592 | Work E plan acc 17.079% NLL 39.220% | G1 d=.01 acc 24.647% 54.812% NLL 32.832% 59.785%
CTTC_9_captures_Q2_loops_ICC_0.00 9/8 G1 plan acc 100.000% 100.000% NLL 100.000% 100.000% | d95 acc 0.01345 0.00861 NLL 0.01152 0.00815 | Work E plan acc 57.273% NLL 93.130% | G1 d=.01 acc 75.673% 98.731% NLL 87.672% 99.335%
CTTC_9_captures_Q2_loops_ICC_0.10 9/8 G1 plan acc 100.000% 100.000% NLL 100.000% 100.000% | d95 acc 0.01398 0.00942 NLL 0.01249 0.00947 | Work E plan acc 36.368% NLL 74.516% | G1 d=.01 acc 72.256% 96.931% NLL 81.814% 96.777%
CTTC_9_captures_Q2_loops_ICC_0.25 9/8 G1 plan acc 100.000% 100.000% NLL 100.000% 100.000% | d95 acc 0.01474 0.01052 NLL 0.01382 0.01117 | Work E plan acc 23.414% NLL 52.892% | G1 d=.01 acc 67.457% 92.837% NLL 73.290% 89.601%
CTTC_9_captures_Q3_loop_pairs_ICC_0.00 9/8 G1 plan acc 99.991% 100.000% NLL 100.000% 100.000% | d95 acc 0.01902 0.01218 NLL 0.01629 0.01153 | Work E plan acc 31.351% NLL 67.207% | G1 d=.01 acc 45.232% 83.739% NLL 58.373% 87.621%
CTTC_9_captures_Q3_loop_pairs_ICC_0.10 9/8 G1 plan acc 99.987% 100.000% NLL 100.000% 100.000% | d95 acc 0.01934 0.01268 NLL 0.01689 0.01236 | Work E plan acc 24.643% NLL 55.337% | G1 d=.01 acc 43.914% 80.593% NLL 55.144% 82.603%
CTTC_9_captures_Q3_loop_pairs_ICC_0.25 9/8 G1 plan acc 99.978% 100.000% NLL 100.000% 100.000% | d95 acc 0.01982 0.01340 NLL 0.01776 0.01352 | Work E plan acc 18.855% NLL 43.093% | G1 d=.01 acc 42.067% 75.982% NLL 50.835% 75.203%
```

### Event #21: `item.completed` — `command_execution` (`item_17`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '"'^## |''^### '"' research/stage4_math_and_design/M2/THEOREM_M2.md | Select-Object -Last 18; Get-ChildItem -LiteralPath research/stage4_math_and_design/M2 -Force | Select-Object Name,Length; git diff --name-only"
```

**Aggregated Output:**

```text
71:## 2. Group averaging and the top-1 loss
99:## 3. Stage 4 frozen statistical gate
107:### Exact type-I control and power under a balanced Gaussian model
150:## 4. Replayed empirical results
154:### Q3 Bayes terms and the paired decomposition
180:### Q3 scalar recalibration
197:### Q3 pooling on the saved test design
216:### Work F mechanism and averaging convention
224:### Q2 controls
243:## 5. Power replay and Round 2 lock proposal
277:### Round 2
283:### Reproduction
295:## 6. Round 2 claim scope and budget lock
307:### Conditional validity after the 18% dev screen
331:## 7. Round 2 formal gate table
335:### Arm roles and causal contrasts
355:### Three-claim testing tree
373:## 8. Stage 6 compatibility

Name            Length
----            ------
.uv-cache-r2          
reply_r1.md     4303  
results_m2.json 419996
THEOREM_M2.md   62360 
verify_m2.py    46159 


```

### Event #22: `item.completed` — `agent_message` (`item_19`)

The 400,000-trial check agrees with the conditional proof. With M6’s original cutoff used only on dev, the exact false-rejection rate among continuations is 1.667% at the first Holm threshold. The power sensitivity also matters: under the older high-variance planning assumptions, 32 synthetic blocks give 33.9% accuracy power, versus 93.0% at 131.

### Event #23: `item.completed` — `command_execution` (`item_21`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "@'


## 9. Round 2 verification and conditional power

The replay exercises two independently generated dev/blind statistics, with a dev-only illustrative switch between 24-df single-seed and 130-df three-seed blind pivots. The switch is a test fixture for selection validity. The actual cost route must be decided and frozen by the pilot rule. At the M6 legacy cutoff, the analytic boundary continuation probability is 0.606588. The recalculated independent predictive example uses dev cutoff −0.701565 and continues with probability 0.758525 at its dev boundary. Both admit a final exact-size blind test.

| Dev-only screen | Blind level | Conditional size exact / MC | Joint false promotion exact / MC |
|---|---:|---:|---:|
| M6 cutoff −0.270438 | .016667 | .016667 / .016290 | .010110 / .009880 |
| M6 cutoff −0.270438 | .05 | .050000 / .050256 | .030329 / .030480 |
| Independent predictive example | .016667 | .016667 / .016601 | .012642 / .012598 |
| Independent predictive example | .05 | .050000 / .050431 | .037926 / .038270 |

The run uses 400,000 deterministic seeded draws, checks both selected branches against their exact t boundary size and repeats all Round 1 replay assertions. The numbers illustrate the conditional theorem. They are not estimates of practical continuation on unseen Stage 4 pilots.

For power, let \\(d=\\mu_{\\mathcal F}-\\delta\\) and \\(s_F\\) be the conditional per-unit SD. Use
\\[
 P_{\\rm pass}(d)=\\Pr\\"'!'"\\left[t_{\\mathrm{df}}\\big(d\\sqrt B/s_F\\big)>t_{\\mathrm{df},1-a}\\right].
\\]
The receipt gives the full excess grid, an inversion at 95% marginal power, and the power at the historical G1 effects. It uses \\(a=0.05/3\\), the conservative first Holm level. It reports conditional power among eligible runs. Joint promotion power also depends on continuation and the full endpoint law. The M4 32-block row uses 24 residual df. The 131-block row is the homogeneous G1 proxy with 130 df. Compute a new variance and df from the exact stratified 131-block allocation if that option is locked. [SciPy noncentral t definition](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.stats.nct.html)

For CTTC with \\(m\\) loops or pairs per capture, use the sensitivity proxy \\(s_F"'^2='"\\sigma_u"'^2['"\\rho+(1-\\rho)/m]+\\sigma_e"'^2/(mS)'"\\), after conditioning on the frozen seed panel. The model uses \\(m=8\\) for Q2 and \\(m=4\\) for Q3 and retains eight df. An extra session-by-seed term would add a component to \\(s_F"'^2'"\\). Use actual real dev covariance for deployment planning. The main table assumes 0.10 for the fraction of common group variance shared within captures. The JSON covers 0, 0.10 and 0.25, a direct historical-panel SD sensitivity, and the legacy Work E high-variance sensitivity. The latter uses a single aggregate variance envelope for both seed counts because Work E never decomposed it.

| Design | Blind df | G1 proxy accuracy power at +1 pp excess, S=1 / S=3 | G1 proxy NLL power at +0.01 nat excess, S=1 / S=3 | Excess needed for 95% power, acc pp S=1 / S=3 | Excess needed for 95% power, NLL nats S=1 / S=3 |
|---|---:|---:|---:|---:|---:|
| Synthetic 32, eight fixed strata | 24 | 48.87% / 87.45% | 62.67% / 90.88% | 1.799 / 1.152 | .01541 / .01090 |
| Synthetic 131, G1 homogeneous proxy | 130 | 98.91% / ≈100% | 99.88% / ≈100% | .854 / .547 | .00731 / .00517 |
| Pluto 16 chronological blocks | 15 | 24.65% / 54.81% | 32.83% / 59.79% | 2.627 / 1.682 | .02250 / .01592 |
| CTTC nine captures, eight Q2 loops, \\(\\rho=.10\\) | 8 | 72.26% / 96.93% | 81.81% / 96.78% | 1.398 / .942 | .01249 / .00947 |
| CTTC nine captures, four Q3 loop-pair groups, \\(\\rho=.10\\) | 8 | 43.91% / 80.59% | 55.14% / 82.60% | 1.934 / 1.268 | .01689 / .01236 |

The old Work E variance envelope gives only 33.91%/71.65% accuracy/NLL power on 32 blocks and 93.01%/99.98% on 131 at the old planned excesses of +3 pp and +0.04 nat, using the new first Holm level. Pluto under that transferred envelope gives 17.08%/39.22%. These comparisons explain why the 32-block plan can produce a scoped confirmatory result if it passes yet deserves a wider uncertainty forecast. The G1 component proxy and the Work E envelope both originate in synthetic Phase-vs-Zero. Neither measures Stage 4's candidate-vs-additive, Arm 2, Q2 real, ECE, or random-label test power. Prospectively acquire development variance and meaningful effect targets for all required endpoints. If a typed gate remains infeasible under its budget, report its outcome as inconclusive or descriptive and retain the Holm slot.

For the random-label checks, archive the assignment law and compute the upper-tail randomization p-value by the exact legal assignment orbit or the plus-one Monte Carlo rule in §3. A diagnostic shuffle of deterministic real labels is insufficient without a valid exchangeability argument. The existing split proof and CPU replay do not establish real-capture exchangeability or Stage 4 gate completion.

'@ | Add-Content -LiteralPath 'research/stage4_math_and_design/M2/THEOREM_M2.md' -Encoding UTF8"
```

### Event #24: `item.completed` — `command_execution` (`item_20`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$ErrorActionPreference='"'Stop'
"'$path='"'research/stage4_math_and_design/M2/verify_m2.py'
"'$content=Get-Content -Raw -Encoding UTF8 -LiteralPath $path
$content=$content.Replace('"'\"predictive_calculation_final_alpha\": alpha_levels[-1], \"policies\": policies,','\"predictive_calculation_final_alpha\": alpha_levels[-1], \"policies\": policies,'+[Environment]::NewLine+'            \"illustrative_plan_selection\": {\"three_seed_dev_Z_cutoff\": .8, \"one_seed_df\": 24, \"three_seed_df\": 130},')
"'$old='"'                for label, sd in ((\"G1_variance_components_proxy\", proxy_sd),'
"'$new=@'"'
                panel_sd = frozen.std(ddof=1) * np.sqrt(deff)
                for label, sd in ((\"G1_variance_components_proxy\", proxy_sd),
                                  (\"historical_frozen_panel_aggregate_ICC_sensitivity\", panel_sd),
'@
if(-not "'$content.Contains($old)){throw '"'Missing power curve anchor'}
"'$content=$content.Replace($old,$new)
Set-Content -Encoding UTF8 -NoNewline -LiteralPath $path -Value $content
$ErrorActionPreference='"'Stop'
"'$m2=Join-Path (Get-Location) '"'research/stage4_math_and_design/M2'
"'$env:UV_CACHE_DIR=Join-Path $m2 '"'.uv-cache-r2'
"'$env:UV_OFFLINE='"'1'
"'$env:UV_NO_PROJECT='"'1'
"'$env:UV_LINK_MODE='"'copy'
"'$env:TMP=$env:UV_CACHE_DIR
$env:TEMP=$env:UV_CACHE_DIR
uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 python -B research/stage4_math_and_design/M2/verify_m2.py'
```

**Aggregated Output:**

```text
results_m2.json written. Maximum in-scope summary difference: 0.
```

### Event #25: `item.completed` — `command_execution` (`item_22`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "@'
# Track M2 | Round 2 reply

**The Modal plans support fixed-connector claims.** M4's four matched fits at seed 202611 qualify for Tier 1 once the blind gate passes. M6's up-to-three selected seeds qualify as a frozen panel under Tier 2. Additional seeds on one arm alone do not add matched trained-arm comparisons. Tier 3 requires at least ten independently paired retrains **and** prospective power for all mandatory endpoints. More blind blocks improve conditional inference about frozen models while leaving the retraining variance floor intact.

**Use the pilot only for dev futility.** Conditional on the training, dev screen, selected checkpoints, calibration and testing plan, the blind p-value remains valid under the selected-path null. With the M6 cutoff −0.270438 and an independent final test at the first Holm level, the exact false-positive rate among continuations is **1.667%** and the joint rate including stops is **1.011%**. The 400,000-trial seeded check gives **1.629%** and **0.988%**. This requires valid blind units after conditioning, including when the recordings or TX design have shared structure. The 18% step target is a budget screen. It supplies no efficacy look.

| Conditional proxy at +1 pp accuracy / +0.01 nat NLL beyond the margin | Acc power, S=1 / S=3 | NLL power, S=1 / S=3 |
|---|---:|---:|
| 32 synthetic blocks, eight strata | 48.87% / 87.45% | 62.67% / 90.88% |
| 131 synthetic blocks, homogeneous G1 proxy | 98.91% / ≈100% | 99.88% / ≈100% |
| Pluto 16 temporal blocks | 24.65% / 54.81% | 32.83% / 59.79% |
| CTTC nine captures, 72 Q2 loops, proxy ICC .10 | 72.26% / 96.93% | 81.81% / 96.78% |
| CTTC nine captures, 36 Q3 groups, proxy ICC .10 | 43.91% / 80.59% | 55.14% / 82.60% |

These are G1 variance proxies for two historical relative endpoints. At the older Work E conservative variance envelope and planned +3 pp/+0.04 nats, 32 blocks give **33.91%/71.65%** and 131 give **93.01%/99.98%**. Use independent pilot variance for the full Stage 4 gate and for each real-data claim.

**Gate handoff.**

- **M1** · Arm 1 must beat matched Zero and Naive-Additive by more than 0.02 nats and 2 pp, clear dev-locked NIG 0.10, and pass fixed-text slot interventions. Arm 2 must beat Direct Physical on NLL and accuracy and beat its own residual-off control. Hold physical logits fixed in both slot interventions. Arm 0 is the legacy reference unless the implementation meets the matched-Zero contract.
- **M3** · Require the paired Q2 NLL ordering two-valid \\(>\\) one-valid \\(>\\) none in utility, and exact 0.0 output change for copies, distractors, padding and list order. Check the actual scorer path. Blind temperature fits stay retrospective.
- **M4/M6** · Use IUT within each of the three claim slots and Holm at 0.05 across synthetic+Q2, Pluto and CTTC. Plan CTTC with **nine capture units** when their conditional independence is defensible. The f708_snr10 result is a held-out cohort guard. Hold the capture out of every fit and calibration. Shared cyclic TX source restricts generalization.
- **M5** · Require Stage 6 max EIG to beat random, nearest and dev-frozen fixed-best on both post-reveal NLL and accuracy at the same one-read budget. Pair the 16-bit canonical vs 4-bit S17 representation test separately under the same prescribed action.

The pinned CPU verifier exited 0. The Stage 3D replay still has maximum in-scope summary difference zero. [Gate and proof](THEOREM_M2.md) · [Verifier](verify_m2.py) · [Numerical results](results_m2.json)

'@ | Set-Content -LiteralPath 'research/stage4_math_and_design/M2/reply_r2.md' -Encoding UTF8"
```

### Event #26: `item.completed` — `command_execution` (`item_24`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$expected=Join-Path (Resolve-Path -LiteralPath '"'research/stage4_math_and_design/M2').Path '.uv-cache-r2'
"'$cache=(Resolve-Path -LiteralPath $expected).Path
if($cache -cne $expected){throw '"'Unexpected cache path'}
if((Get-Item -LiteralPath "'$cache -Force).Attributes -band [IO.FileAttributes]::ReparsePoint){throw '"'Unexpected cache link'}
\"Verified isolated repository cache: "'$cache"
Get-ChildItem -LiteralPath $cache -Force -Recurse -Attributes ReparsePoint | Select-Object -First 4 FullName
'
```

**Aggregated Output:**

```text
Verified isolated repository cache: C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M2\.uv-cache-r2
```

### Event #27: `item.completed` — `command_execution` (`item_23`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$r=Get-Content -Raw -Encoding UTF8 research/stage4_math_and_design/M2/results_m2.json | ConvertFrom-Json
$r.runtime | ConvertTo-Json -Compress
$r.summary_checks_max_absolute_difference | ConvertTo-Json -Compress
$r.round2.dev_disjoint_futility.illustrative_plan_selection | ConvertTo-Json -Compress
$r.round2.conditional_modal_power.designs.synthetic_32_eight_balanced_strata.endpoints.accuracy.'"'1'.historical_frozen_panel_aggregate_ICC_sensitivity | ConvertTo-Json -Depth 4 -Compress
'Q3 {0}, Q2 {1}, round2 designs {2}' -f "'$r.Q3.PSObject.Properties.Count.Count,$r.Q2.controls.PSObject.Properties.Count.Count,$r.round2.conditional_modal_power.designs.PSObject.Properties.Count.Count
rg -n '"'"'^## [6-9]|''^### Three-claim|''^def dev_futility_check|''^def conditional_modal_power'"' research/stage4_math_and_design/M2/THEOREM_M2.md research/stage4_math_and_design/M2/verify_m2.py
git diff --name-only"
```

**Aggregated Output:**

```text
{"python":"3.12.7","numpy":"2.3.5","scipy":"1.17.0"}
{"G1_main_extrap_both":0.0,"G2_interp_both":0.0,"G3_extrap_angle":0.0,"G4_extrap_freq":0.0,"work_f_primary_040":0.0,"work_f_historical_035":0.0,"work_g_primary_040":0.0,"work_a_q2_c4":0.0}
{"three_seed_dev_Z_cutoff":0.8,"one_seed_df":24,"three_seed_df":130}
{"unit_sd":0.02883420477582685,"power_curve":[0.016666666666666677,0.11650056397038171,0.3979909615470512,0.7499391201451486,0.9458511334461015,0.9943975351549901,0.9997352441127835,0.9999999481197003,0.9999999999996251,1.0,1.0,1.0],"power_at_planning_excess":0.9997352441127835,"power_at_historical_observed_excess":0.9999999999475577,"minimum_excess_for_95_percent_power":0.02020926105862834}
Q3 5, Q2 7, round2 designs 10
research/stage4_math_and_design/M2/verify_m2.py:635:def dev_futility_check():
research/stage4_math_and_design/M2/verify_m2.py:694:def conditional_modal_power(primary, pilot):
research/stage4_math_and_design/M2/THEOREM_M2.md:295:## 6. Round 2 claim scope and budget lock
research/stage4_math_and_design/M2/THEOREM_M2.md:331:## 7. Round 2 formal gate table
research/stage4_math_and_design/M2/THEOREM_M2.md:355:### Three-claim testing tree
research/stage4_math_and_design/M2/THEOREM_M2.md:373:## 8. Stage 6 compatibility
research/stage4_math_and_design/M2/THEOREM_M2.md:387:## 9. Round 2 verification and conditional power
```

### Event #28: `item.completed` — `command_execution` (`item_25`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$ErrorActionPreference='"'Stop'
"'$p='"'research/stage4_math_and_design/M2/THEOREM_M2.md'
"'$c=Get-Content -Raw -Encoding UTF8 -LiteralPath $p
$c=$c.Replace('"'The shuffle is a separately seeded, frozen mapping of informative slot values within prespecified eligible strata, without labels.','The shuffle is a separately seeded, frozen mapping of slot values across signs within prespecified absolute-CFO, lag, SNR and availability strata, without labels. Record its effective mismatch fraction and the locked treatment of strata that cannot be shuffled.')
"'$c=$c.Replace('"'Lock a same-information oracle or independently fitted physical reference, a positive denominator and','Lock a same-information oracle or independently fitted physical reference. Establish a positive denominator analytically or with a valid paired lower bound as an additional IUT endpoint, and require')
Set-Content -LiteralPath "'$p -Encoding UTF8 -NoNewline -Value $c
uv cache clean --cache-dir research/stage4_math_and_design/M2/.uv-cache-r2
if($LASTEXITCODE -ne 0){throw '"'Failed to clean the temporary M2 cache'}
Get-ChildItem -LiteralPath research/stage4_math_and_design/M2 -Force | Select-Object Name,Length
git diff --name-only
"
```

**Aggregated Output:**

```text
Clearing cache at: research/stage4_math_and_design/M2/.uv-cache-r2
Removed 5158 files (365.0MiB)

Name            Length
----            ------
reply_r1.md       4303
reply_r2.md       3520
results_m2.json 453753
THEOREM_M2.md    67511
verify_m2.py     46438


```

### Event #29: `item.completed` — `agent_message` (`item_26`)

**Round 2 locks the Modal plans to fixed-connector claims.** M4’s one-seed plan can support a claim about the named checkpoints. M6’s continuation can support a claim about the frozen three-seed panel, with trained-arm comparisons limited to common paired seeds. A retraining-population claim requires **at least ten paired seeds and prospective power for the full gate**.

The dev-only futility screen preserves blind Type-I control when the blind test is valid conditional on the frozen plan. At the first Holm threshold, M6’s cutoff gives **1.667%** exact false rejection among continuations and **1.011%** joint false promotion including stops. The 400,000-trial check returned **1.629%** and **0.988%**.

| Design | Accuracy power, 1 / 3 seeds | NLL power, 1 / 3 seeds |
|---|---:|---:|
| Synthetic, 32 blocks | 48.9% / 87.5% | 62.7% / 90.9% |
| Synthetic, 131 blocks | 98.9% / ≈100% | 99.9% / ≈100% |
| Pluto, 16 blocks | 24.6% / 54.8% | 32.8% / 59.8% |
| CTTC Q2, nine captures | 72.3% / 96.9% | 81.8% / 96.8% |
| CTTC Q3, nine captures | 43.9% / 80.6% | 55.1% / 82.6% |

These are G1-based planning proxies at **+1 pp accuracy and +0.01 nat NLL above the margin**. Pluto and CTTC still require their own variance and dependence checks.

The gate requires **Host-Pure** to beat matched Zero and Naive-Additive and pass fixed-text slot interventions. **Physical+HostResidual** must beat Direct Physical and its own residual-off forecast. Blind recalibration is retrospective only. Q2 requires ordered gains from distinct valid acquisitions and exact `0.0` changes for duplicates, distractors, padding and list permutations. Use IUT within each claim and Holm across synthetic+Q2, Pluto and CTTC. CTTC inference uses capture units when defensible, with `f708_snr10` as a held-out cohort guard. Stage 6 requires `max_EIG` to beat random, nearest and dev-frozen fixed-best under the same one-read budget, with a separate 16-bit versus 4-bit comparison.

The pinned CPU verifier exited 0 and reproduced the in-scope Stage 3D summary with maximum difference **0**.

[Round 2 reply](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M2/reply_r2.md) · [Gate and proofs](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M2/THEOREM_M2.md:295) · [Verifier](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M2/verify_m2.py:635) · [Numerical results](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M2/results_m2.json)



### Event #30: `turn.completed`

- Usage: `{"input_tokens": 11541996, "cached_input_tokens": 11342720, "cache_write_input_tokens": 0, "output_tokens": 69062, "reasoning_output_tokens": 13787}`


## Round 2 — Final Output (`reply_r2.md`)

```markdown
**Round 2 locks the Modal plans to fixed-connector claims.** M4’s one-seed plan can support a claim about the named checkpoints. M6’s continuation can support a claim about the frozen three-seed panel, with trained-arm comparisons limited to common paired seeds. A retraining-population claim requires **at least ten paired seeds and prospective power for the full gate**.

The dev-only futility screen preserves blind Type-I control when the blind test is valid conditional on the frozen plan. At the first Holm threshold, M6’s cutoff gives **1.667%** exact false rejection among continuations and **1.011%** joint false promotion including stops. The 400,000-trial check returned **1.629%** and **0.988%**.

| Design | Accuracy power, 1 / 3 seeds | NLL power, 1 / 3 seeds |
|---|---:|---:|
| Synthetic, 32 blocks | 48.9% / 87.5% | 62.7% / 90.9% |
| Synthetic, 131 blocks | 98.9% / ≈100% | 99.9% / ≈100% |
| Pluto, 16 blocks | 24.6% / 54.8% | 32.8% / 59.8% |
| CTTC Q2, nine captures | 72.3% / 96.9% | 81.8% / 96.8% |
| CTTC Q3, nine captures | 43.9% / 80.6% | 55.1% / 82.6% |

These are G1-based planning proxies at **+1 pp accuracy and +0.01 nat NLL above the margin**. Pluto and CTTC still require their own variance and dependence checks.

The gate requires **Host-Pure** to beat matched Zero and Naive-Additive and pass fixed-text slot interventions. **Physical+HostResidual** must beat Direct Physical and its own residual-off forecast. Blind recalibration is retrospective only. Q2 requires ordered gains from distinct valid acquisitions and exact `0.0` changes for duplicates, distractors, padding and list permutations. Use IUT within each claim and Holm across synthetic+Q2, Pluto and CTTC. CTTC inference uses capture units when defensible, with `f708_snr10` as a held-out cohort guard. Stage 6 requires `max_EIG` to beat random, nearest and dev-frozen fixed-best under the same one-read budget, with a separate 16-bit versus 4-bit comparison.

The pinned CPU verifier exited 0 and reproduced the in-scope Stage 3D summary with maximum difference **0**.

[Round 2 reply](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M2/reply_r2.md) · [Gate and proofs](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M2/THEOREM_M2.md:295) · [Verifier](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M2/verify_m2.py:635) · [Numerical results](C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M2/results_m2.json)

```
