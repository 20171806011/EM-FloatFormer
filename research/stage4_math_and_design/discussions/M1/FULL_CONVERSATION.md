# Full Verbatim Multi-Round Conversation — Track `M1`

- **CLI Runtime**: Codex CLI (`codex exec`)
- **Session / Thread ID**: `01a21948-8b7b-7ec0-bb98-65a487a5ce22`
- **Round 1 Events**: `105` raw JSONL events (`round1_events.jsonl`)
- **Round 2 Events**: `71` raw JSONL events (`round2_events.jsonl`)
- **Completeness Guarantee**: All user prompts (`prompt_r1.md`, `prompt_r2.md`), all completed reasoning/tool/command events, and final responses (`reply_r1.md`, `reply_r2.md`) are included verbatim below without truncation or paraphrasing.

---

## Round 1 — Initial Mathematical Brief (`prompt_r1.md`)

````markdown
# Track M1 (Round 1): Group-Theoretic Phase-Coupled Equivariant Architecture & Extrapolation Theory

You are **Track M1**, part of a 4-track mathematical & architectural multi-agent collaboration preparing **EM-FloatFormer Stage 4** (and fixing the root mathematical bottlenecks observed in **Stage 3D**).

## 0. Strict Workspace, Environment, and Immutability Rules
- **Repository Root**: `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`
- **Your Writable Directory**: `research/stage4_math_and_design/M1/` ONLY.
- **FORBIDDEN Edits**: Do NOT modify or delete any files outside `research/stage4_math_and_design/M1/` (especially do NOT touch `research/matched_cache`, `research/stage3_ministral_cache`, `research/stage3b_math`, `research/stage3c`, `research/stage3d_qwen_signed_phase`).
- **Python Environment Guardrail**: System/Anaconda `python` crashes with MKL `libiomp5md.dll` conflicts on this machine. **ALWAYS** execute Python scripts via:
  ```powershell
  uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B <script.py>
  ```
  (Omit `--with torch` if only numpy/scipy are needed; test with `uv run` so all scripts run cleanly and reproducibly.)

## 1. Grounded Empirical Context (Read These Files Read-Only First)
Inspect only the relevant slices of these files inside the repo:
1. `research/matched_cache/learned.py` (`equivariant_b8_s17`, `input_features`, `fit`, `Learned.encode`)
2. `research/stage3c/G/quotient.py` (`canonical_input`, `rotate`, `one_call`)
3. `research/stage3_ministral_cache/schema.py` (`Q1`, `Q2`, `Q3` choices/prompts, `matched_sum`, `sector_of_complex`, `ROOTS`)
4. `research/stage3d_qwen_signed_phase/data_stage3d.py` (`public_phase_slots`, `TRAIN_TIMINGS`, `TRAIN_OMEGAS`, `DEV_OMEGAS`, `BLIND_CELLS`, `generate_paired_cfo_blocks`)
5. `research/stage3d_qwen_signed_phase/qwen_connector.py` (`SignedPhaseCacheConnector`, `build_prefix_embeds`, `format_chat_parts`)
6. `research/stage3d_qwen_signed_phase/recompute_all.py` (`compute_16way_group_avg`, `compute_representative_readout`, `work_g_rotation_and_controls`)
7. `research/stage3d_qwen_signed_phase/results/summary_master.json` (inspect summary stats for `G1..G4`, `work_f_primary_040`, `work_g_primary_040`)

### Verified Stage 3D Empirical Facts to Explain & Overcome
- Stage 3D trained `SignedPhaseCacheConnector` (2,184,964 trainable params, `<W1><W2><PHASE>` concatenated as additive/separate embedding blocks before frozen `Qwen/Qwen3.5-4B`) strictly on `|\omega| <= 0.25` (`TRAIN_OMEGAS = (-0.25, -0.10, 0.0, 0.10, 0.25)`) and `h+\ell <= 5` (`TRAIN_TIMINGS = ((2,2,4.0), (2,3,1.0))`), i.e. cumulative angle `|\omega|(h+\ell) <= 1.25 rad < \pi/2`.
- On blind 2x2 CFO test grid (`snr_db = 2.0`, `sigma_phi = 0.10`):
  - `G2_interp_both` (`|\omega|=0.20, h=\ell=2, |\theta|=0.80 rad`): `phase` Acc `33.88%` (NLL `1.3358`) vs `zero` Acc `36.36%` (NLL `1.3081`), `direct_s17_table` Acc `46.78%` (NLL `1.0222`). Notice `zero` beats `phase`!
  - `G4_extrap_freq` (`|\omega|=0.40, h=\ell=1, |\theta|=0.80 rad`): `phase` Acc `33.29%` (NLL `1.3347`) vs `zero` Acc `34.96%` (NLL `1.3098`), `direct_s17_table` Acc `47.75%` (NLL `1.0159`). Again `zero` beats `phase`!
  - `G1_main_extrap_both` (`|\omega|=0.40, h=\ell=2, |\theta|=1.60 rad`): `phase` Acc `23.80%` (NLL `1.4452`) vs `zero` Acc `19.41%` (NLL `1.4809`), while `direct_s17_table` JUMPS UP to `64.62%` (NLL `0.9411`)!
  - `G3_extrap_angle` (`|\omega|=0.20, h=\ell=4, |\theta|=1.60 rad`): `phase` Acc `22.68%` (NLL `1.4410`) vs `zero` Acc `19.74%` (NLL `1.4781`), while `direct_s17_table` JUMPS UP to `62.11%` (NLL `0.9796`)!
- **Work F** (`G1 ±0.40`): `phase` `orientation_gain = +0.0098`, `swapped_minus_matched_nll = +0.0184`, factorial `metadata_M_sq = 1.854e-4 << wave_W_sq = 1.120e-3` (whereas physical `cal_ref` has `M_sq = 0.2016`).
- **Work G** (`G1`): Raw host violates $\mathbb{Z}_4$-equivariance (`rot01` mean max-abs diff `0.0591`, `rot11` `0.1263`, worst `0.4817`); post-hoc offline 16-way group averaging `Pf` lowers NLL (`1.4452 -> 1.4113`) yet DROPS Acc (`23.80% -> 17.05%`)!

## 2. Mathematical Questions You Must Answer Rigorously
1. **Exact $(\mathbb{Z}_4 \times \mathbb{Z}_4) \rtimes S^1$ Decomposition & Why `direct_s17` Jumps to 64.6% While Stage 3D Collapses <25%**:
   - Prove the exact symmetries of `(z1, z2) = (4 a + u, 4 b + v)` (`a, b \in \mathbb{Z}_4`, `u, v \in \{0..3\}` canonical inner states) across **all tasks** (`Q1`, `Q2`, `Q3_repeat`, `Q3_indep`, `Q3_after_pilot`) under continuous CFO $\omega$ and Brownian phase diffusion $\sigma_\phi$.
   - Derive mathematically why `direct_s17_table` exhibits non-monotonic "boundary dip at $|\theta| \approx \pi/4$" (`~47%` at `0.80 rad`) and "quarter-turn resonance peak near $|\theta| \approx \pi/2$" (`~64.6%` at `1.60 rad`), and prove why ANY estimator restricted to unrotated sector predictions or local polynomial extrapolation in raw logits MUST fall below random `25%` at $|\theta| = 1.60 > \pi/2$.
2. **Structural Impossibility / Failure Proof of Additive Token-Concatenation Connector (`SignedPhaseCacheConnector`)**:
   - Analyze `SignedPhaseCacheConnector`: `(e_w1(z1,z2), e_w2(z1,z2), e_phase(cos(omega*t), sin(omega*t)))` injected as separate tokens into a frozen LLM.
   - Prove the exact structural failure modes: (i) broken $\mathbb{Z}_4$ quotient equivariance (16x sample inefficiency & directional collapse), (ii) rank-1 / bilinear blind spot when phase slots live at separate attention positions from discrete cache embeddings inside a **frozen** LLM (cannot synthesize exact `exp(i * omega * t)` group rotation across unseen cumulative phase $|\theta| \in (\pi/2, \pi]$ when trained only on $|\theta| \le 1.25$), and (iii) why naive post-hoc `Pf` probability averaging improves convex NLL by Jensen's inequality yet pushes top-1 Accuracy DOWN from `23.80% -> 17.05%`!
   - Quantify the exact effective rank / Frobenius spectral mismatch on the Stage 3D physical reference (`cal_ref` of `G1`)!
3. **Exact Stage 4 Equivariant + Evidence-Coupled Connector Architecture (`EquivariantEvidenceConnector`)**:
   - Design an end-to-end differentiable, frozen-LLM-compatible connector that:
     - Enforces **exact** `0.000e+00` $\mathbb{Z}_4$-equivariance on all tasks by quotient canonicalization (`canonical_input` + cyclic logit rotation `rotate` or equivariant spectral mixing on canonical `(u, v)` + explicit task conditioning so `Q3_indep` remains flat uniform!),
     - Directly models continuous $SO(2) = S^1$ phase evolution via exact complex/phasor equivariant features $(\cos(\omega \Delta t), \sin(\omega \Delta t), \exp(-\frac{1}{2}\sigma_\phi^2 \Delta t), \dots)$ **multiplicatively coupled** to canonical inner-state evidence **inside the trainable connector** before emitting soft tokens (and optionally via a lightweight equivariant logit residual / direct physical-evidence head or frozen-LM equivariant readout),
     - Uses parameter budget comparable to or smaller than Stage 3D (`<= 2.2M` params, fast convergence inside Modal free credits).
   - Explicitly specify clean interfaces for **Track M3 (Q2/Q3_after_pilot `RefSufficientPool`)** and **Track M4 (Real Pluto/CTTC CFO+SNR inputs)**!
4. **Mandatory CPU Empirical Verification (`verify_m1.py`)**:
   - Write `research/stage4_math_and_design/M1/verify_m1.py` (runnable strictly with `uv run --python 3.12 ...`).
   - Load the REAL `equivariant_b8_s17.npz` encoder (`load_s17_encoder`) and generate REAL synthetic paired CFO data via `generate_paired_cfo_blocks` for `TRAIN_OMEGAS/TIMINGS` (`|\omega| <= 0.25, h+\ell <= 5`) and `G1..G4`.
   - Build a clean frozen-host CPU surrogate (e.g. frozen random Transformer/MLP linear readout host with frozen token embeddings + tiny trainable connector trained ONLY on `|\omega| <= 0.25, h+\ell <= 5`) comparing:
     (A) Stage 3D-style Additive Uncoupled Connector (`AdditivePhaseBaseline`),
     (B) Stage 4 Equivariant + Phase-Coupled Evidence Connector (`EquivariantPhaseCoupled`), and
     (C) Exact Analytical Quotient Oracle (`direct_s17`).
   - Verify empirically:
     - Exact $\mathbb{Z}_4$ equivariance defect on `B` (`<= 1e-6`, ideally `0.0`) vs broken equivariance on `A`;
     - Exact `G1..G4` extrapolation: `B` must cleanly extrapolate past $\pi/2$ on `G1` & `G3` (achieving **> 55% Acc** and matching `direct_s17` closely, while `A` collapses around `<= 25%`!);
     - Reproduce and verify the `Pf` NLL-improves / Acc-drops lemma numerically!

## 3. Deliverables Required from Track M1
1. `research/stage4_math_and_design/M1/THEOREM_M1.md` (comprehensive, mathematically self-contained document with formal Theorems, Proofs, Stage 3D diagnostics, Stage 4 Connector Spec, and interface contract for M3/M4).
2. `research/stage4_math_and_design/M1/verify_m1.py` (executed by you via `uv run` with zero errors).
3. `research/stage4_math_and_design/M1/results_m1.json` (generated by `verify_m1.py`, recording all verified tables, eigenvalues/ranks, equivariance diffs, `G1..G4` extrapolation metrics, and `Pf` verification numbers).
4. A crisp summary in your final message stating exact theorems proved, key numbers from `results_m1.json`, and explicit design/interface questions for M2/M3/M4 Round 2.
````


## Round 1 — Execution & Reasoning Trace (`round1_events.jsonl`)

### Event #1: `thread.started` (thread_id=`01a21948-8b7b-7ec0-bb98-65a487a5ce22`)

### Event #2: `turn.started`

### Event #3: `item.completed` — `agent_message` (`item_0`)

I’ll read the specified Stage 3D slices first, then build the theory and CPU verification in `research/stage4_math_and_design/M1/`. I’ll distinguish what the architecture guarantees from what the Stage 3D measurements and surrogate runs can establish.

### Event #4: `item.completed` — `command_execution` (`item_2`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n '"'^(class |def |[A-Z][A-Z0-9_]+'"\\s*=)' research/matched_cache/learned.py research/stage3c/G/quotient.py research/stage3_ministral_cache/schema.py research/stage3d_qwen_signed_phase/{data_stage3d,qwen_connector,recompute_all}.py"
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
 
����λ�� ��:2 �ַ�: 199
+ ... e/schema.py research/stage3d_qwen_signed_phase/{data_stage3d,qwen_con ...
+                                                                 ~
�����б���ȱ�ٲ�����
    + CategoryInfo          : ParserError: (:) [], ParentContainsErrorRecordException
    + FullyQualifiedErrorId : MissingArgument
 
```

### Event #5: `item.completed` — `command_execution` (`item_1`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "rg -n -i 'EM-FloatFormer|stage3d|signed.phase|s17|matched_cache' 'C:/Users/李则徐/.codex/memories/MEMORY.md'"
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
```

### Event #6: `item.completed` — `command_execution` (`item_3`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -Command "Get-Location; git status --short; rg --files --hidden -g AGENTS.md -g '"'!**/.git/**'"' -g '"'!**/.venv/**'"'; Get-Command uv"
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

### Event #7: `item.completed` — `command_execution` (`item_5`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n '"'^(class |def |[A-Z][A-Z0-9_]+'"\\s*=)' research/matched_cache/learned.py research/stage3c/G/quotient.py research/stage3_ministral_cache/schema.py research/stage3d_qwen_signed_phase/data_stage3d.py research/stage3d_qwen_signed_phase/qwen_connector.py research/stage3d_qwen_signed_phase/recompute_all.py"
```

**Aggregated Output:**

```text
research/matched_cache/learned.py:7:ROOTS = np.array([1, 1j, -1, -1j])
research/matched_cache/learned.py:8:LABELS = np.arange(4)
research/matched_cache/learned.py:11:def input_features(windows, equivariant):
research/matched_cache/learned.py:33:def forward(x, weights):
research/matched_cache/learned.py:41:def fit_network(x, target, rng):
research/matched_cache/learned.py:72:def assign(posterior, centers):
research/matched_cache/learned.py:77:def fit_centers(prediction, count, rng):
research/matched_cache/learned.py:107:class Learned:
research/matched_cache/learned.py:125:def load(path):
research/matched_cache/learned.py:131:def fit(windows, soft_targets, bits, seed, equivariant):
research/stage3d_qwen_signed_phase/qwen_connector.py:24:MODEL_ID_DEFAULT = "Qwen/Qwen3.5-4B"
research/stage3d_qwen_signed_phase/qwen_connector.py:25:PINNED_REVISION_DEFAULT = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
research/stage3d_qwen_signed_phase/qwen_connector.py:26:SYSTEM_PROMPT = (
research/stage3d_qwen_signed_phase/qwen_connector.py:30:MARKER_REGEX = re.compile(r"(<W1>|<W2>|<PHASE>|<REF>)")
research/stage3d_qwen_signed_phase/qwen_connector.py:33:class SignedPhaseCacheConnector(nn.Module):
research/stage3d_qwen_signed_phase/qwen_connector.py:144:def _get_oop_linear_attention_layer_cls():
research/stage3d_qwen_signed_phase/qwen_connector.py:192:def _make_differentiable_qwen35_cache(lm_config):
research/stage3d_qwen_signed_phase/qwen_connector.py:205:def _clone_expand_qwen35_cache(pref_cache, lm_config, repeats: int):
research/stage3d_qwen_signed_phase/qwen_connector.py:233:class QwenCacheHost:
research/stage3d_qwen_signed_phase/data_stage3d.py:36:EQUIVARIANT_B8_S17_SHA256 = "e67d9fbafb90791ab19549de7789798d818286d7f52d7ac566cc39df71e78356"
research/stage3d_qwen_signed_phase/data_stage3d.py:43:NS_CFO_TRAIN = 930101
research/stage3d_qwen_signed_phase/data_stage3d.py:44:NS_CFO_DEV = 930202
research/stage3d_qwen_signed_phase/data_stage3d.py:45:NS_DEV_PILOT = 940202
research/stage3d_qwen_signed_phase/data_stage3d.py:46:NS_BLIND_G1 = 950401
research/stage3d_qwen_signed_phase/data_stage3d.py:47:NS_BLIND_G2 = 950402
research/stage3d_qwen_signed_phase/data_stage3d.py:48:NS_BLIND_G3 = 950403
research/stage3d_qwen_signed_phase/data_stage3d.py:49:NS_BLIND_G4 = 950404
research/stage3d_qwen_signed_phase/data_stage3d.py:50:NS_BLIND_Q2_C4 = 950405
research/stage3d_qwen_signed_phase/data_stage3d.py:51:NS_REF_CALIB = 960501
research/stage3d_qwen_signed_phase/data_stage3d.py:69:class BlindCellSpec:
research/stage3d_qwen_signed_phase/data_stage3d.py:146:def load_s17_encoder(repo_root: Optional[Path] = None):
research/stage3d_qwen_signed_phase/data_stage3d.py:161:def public_phase_slots(omega: float, h: int, ell: int, t_p: float, zero_arm: bool = False) -> np.ndarray:
research/stage3d_qwen_signed_phase/data_stage3d.py:169:def condition_spec_from_params(
research/stage3d_qwen_signed_phase/data_stage3d.py:193:def format_public_metadata(cond: ConditionSpec) -> str:
research/stage3d_qwen_signed_phase/data_stage3d.py:209:def build_stage3d_prompt(
research/stage3d_qwen_signed_phase/data_stage3d.py:262:def derive_block_seeds(namespace: int, cell_idx: int, blocks: int) -> List[int]:
research/stage3d_qwen_signed_phase/data_stage3d.py:270:def generate_paired_cfo_blocks(
research/stage3d_qwen_signed_phase/data_stage3d.py:375:def build_q2_arm_references(batch: MotherBatch, arm: str, coord_rot: int = 0) -> Tuple[Tuple[Reference, ...], bool]:
research/stage3d_qwen_signed_phase/data_stage3d.py:405:def build_blind_seed_manifest(g1_blocks: int = 131) -> dict:
research/stage3d_qwen_signed_phase/data_stage3d.py:445:def verify_stage3d_orbit_isolation(max_blocks: int = 2) -> Dict[str, object]:
research/stage3d_qwen_signed_phase/recompute_all.py:15:BLOCK_SIZE = 64
research/stage3d_qwen_signed_phase/recompute_all.py:16:BOOTSTRAP_RESAMPLES = 10000
research/stage3d_qwen_signed_phase/recompute_all.py:17:BOOTSTRAP_SEED = 2026311
research/stage3d_qwen_signed_phase/recompute_all.py:18:CELL_ORDER = (
research/stage3d_qwen_signed_phase/recompute_all.py:24:CELL_META = {
research/stage3d_qwen_signed_phase/recompute_all.py:30:Q2_ARMS = (
research/stage3d_qwen_signed_phase/recompute_all.py:39:Q2_LOGICAL_REF_IDX = {
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
research/stage3d_qwen_signed_phase/recompute_all.py:967:def _compare_nested(expected, actual, path: str = "root") -> float:
research/stage3d_qwen_signed_phase/recompute_all.py:993:def main():
research/stage3_ministral_cache/schema.py:7:ROOTS = np.array([1.0 + 0j, 1j, -1.0 + 0j, -1j], dtype=np.complex128)
research/stage3_ministral_cache/schema.py:8:N_SAMPLES = 3
research/stage3_ministral_cache/schema.py:9:TWO_DB_SNR = 10.0 ** 0.2
research/stage3_ministral_cache/schema.py:14:Q1_PROMPT = "第二个窗口的发射符号相对第一个窗口转了几个四分之一圈？"
research/stage3_ministral_cache/schema.py:17:Q2_PROMPT = "第一个窗口原来发送的是哪个四相符号？"
research/stage3_ministral_cache/schema.py:20:Q3_PROMPT_REPEAT = (
research/stage3_ministral_cache/schema.py:24:Q3_PROMPT_INDEP = (
research/stage3_ministral_cache/schema.py:28:Q3_PROMPT_REPEAT_AFTER_PILOT = (
research/stage3_ministral_cache/schema.py:41:class Reference:
research/stage3_ministral_cache/schema.py:51:class ConditionSpec:
research/stage3_ministral_cache/schema.py:152:def sector_of_complex(z: np.ndarray) -> np.ndarray:
research/stage3_ministral_cache/schema.py:164:def matched_sum(windows: np.ndarray) -> np.ndarray:
research/stage3_ministral_cache/schema.py:170:def dedup_references(references: Sequence[Reference]) -> Tuple[List[Reference], int, int, bool]:
research/stage3_ministral_cache/schema.py:198:def build_query_prompt(
research/stage3c/G/quotient.py:8:def canonical_input(z1, z2, task):
research/stage3c/G/quotient.py:23:def rotate(values, shift):
research/stage3c/G/quotient.py:29:def one_call(reader, z1, z2, task, context):
```

### Event #8: `item.completed` — `command_execution` (`item_4`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-ChildItem -Force research/stage4_math_and_design; rg --files research/stage3d_qwen_signed_phase research/matched_cache | Select-String 'summary|s17|recompute|work_[fg]'"
```

**Aggregated Output:**

```text


    Ŀ¼: C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design


Mode                 LastWriteTime         Length Name                                                                 
----                 -------------         ------ ----                                                                 
d-----        2026/11/26     19:11                discussions                                                          
d-----        2026/11/26     19:11                M1                                                                   
d-----        2026/11/26     19:11                M2                                                                   
d-----        2026/11/26     19:11                M3                                                                   
d-----        2026/11/26     19:11                M4                                                                   
d-----        2026/11/26     19:13                __pycache__                                                          
-a----        2026/11/26     19:13           8800 transcript_utils.py                                                  

IgnoreCase : True
LineNumber : 11
Line       : research/matched_cache\results\test\equivariant_b4_s17.json
Filename   : InputStream
Path       : InputStream
Pattern    : summary|s17|recompute|work_[fg]
Context    : 
Matches    : {0}


IgnoreCase : True
LineNumber : 12
Line       : research/matched_cache\results\summary.json
Filename   : InputStream
Path       : InputStream
Pattern    : summary|s17|recompute|work_[fg]
Context    : 
Matches    : {0}


IgnoreCase : True
LineNumber : 17
Line       : research/matched_cache\results\test\plain_b8_s17.json
Filename   : InputStream
Path       : InputStream
Pattern    : summary|s17|recompute|work_[fg]
Context    : 
Matches    : {0}


IgnoreCase : True
LineNumber : 20
Line       : research/matched_cache\results\test\plain_b4_s17.json
Filename   : InputStream
Path       : InputStream
Pattern    : summary|s17|recompute|work_[fg]
Context    : 
Matches    : {0}


IgnoreCase : True
LineNumber : 25
Line       : research/matched_cache\results\test\equivariant_b8_s17.json
Filename   : InputStream
Path       : InputStream
Pattern    : summary|s17|recompute|work_[fg]
Context    : 
Matches    : {0}


IgnoreCase : True
LineNumber : 35
Line       : research/stage3d_qwen_signed_phase\recompute_all.py
Filename   : InputStream
Path       : InputStream
Pattern    : summary|s17|recompute|work_[fg]
Context    : 
Matches    : {0}


IgnoreCase : True
LineNumber : 40
Line       : research/matched_cache\results\dev\equivariant_b8_s17.json
Filename   : InputStream
Path       : InputStream
Pattern    : summary|s17|recompute|work_[fg]
Context    : 
Matches    : {0}


IgnoreCase : True
LineNumber : 43
Line       : research/matched_cache\results\dev\equivariant_b4_s17.json
Filename   : InputStream
Path       : InputStream
Pattern    : summary|s17|recompute|work_[fg]
Context    : 
Matches    : {0}


IgnoreCase : True
LineNumber : 48
Line       : research/matched_cache\results\dev\plain_b4_s17.json
Filename   : InputStream
Path       : InputStream
Pattern    : summary|s17|recompute|work_[fg]
Context    : 
Matches    : {0}


IgnoreCase : True
LineNumber : 50
Line       : research/matched_cache\results\dev\plain_b8_s17.json
Filename   : InputStream
Path       : InputStream
Pattern    : summary|s17|recompute|work_[fg]
Context    : 
Matches    : {0}


IgnoreCase : True
LineNumber : 54
Line       : research/stage3d_qwen_signed_phase\results\work_f_host_035_zero.json
Filename   : InputStream
Path       : InputStream
Pattern    : summary|s17|recompute|work_[fg]
Context    : 
Matches    : {0}


IgnoreCase : True
LineNumber : 55
Line       : research/stage3d_qwen_signed_phase\results\work_f_host_035_phase.json
Filename   : InputStream
Path       : InputStream
Pattern    : summary|s17|recompute|work_[fg]
Context    : 
Matches    : {0}


IgnoreCase : True
LineNumber : 56
Line       : research/stage3d_qwen_signed_phase\results\train_summary_zero_s202633.json
Filename   : InputStream
Path       : InputStream
Pattern    : summary|s17|recompute|work_[fg]
Context    : 
Matches    : {0}


IgnoreCase : True
LineNumber : 57
Line       : research/stage3d_qwen_signed_phase\results\train_summary_zero_s202622.json
Filename   : InputStream
Path       : InputStream
Pattern    : summary|s17|recompute|work_[fg]
Context    : 
Matches    : {0}


IgnoreCase : True
LineNumber : 58
Line       : research/stage3d_qwen_signed_phase\results\train_summary_zero_s202611.json
Filename   : InputStream
Path       : InputStream
Pattern    : summary|s17|recompute|work_[fg]
Context    : 
Matches    : {0}


IgnoreCase : True
LineNumber : 59
Line       : research/stage3d_qwen_signed_phase\results\train_summary_phase_s202633.json
Filename   : InputStream
Path       : InputStream
Pattern    : summary|s17|recompute|work_[fg]
Context    : 
Matches    : {0}


IgnoreCase : True
LineNumber : 60
Line       : research/stage3d_qwen_signed_phase\results\train_summary_phase_s202622.json
Filename   : InputStream
Path       : InputStream
Pattern    : summary|s17|recompute|work_[fg]
Context    : 
Matches    : {0}


IgnoreCase : True
LineNumber : 61
Line       : research/stage3d_qwen_signed_phase\results\train_summary_phase_s202611.json
Filename   : InputStream
Path       : InputStream
Pattern    : summary|s17|recompute|work_[fg]
Context    : 
Matches    : {0}


IgnoreCase : True
LineNumber : 62
Line       : research/stage3d_qwen_signed_phase\results\training_summary.json
Filename   : InputStream
Path       : InputStream
Pattern    : summary|s17|recompute|work_[fg]
Context    : 
Matches    : {0}


IgnoreCase : True
LineNumber : 63
Line       : research/stage3d_qwen_signed_phase\results\summary_master.json
Filename   : InputStream
Path       : InputStream
Pattern    : summary|s17|recompute|work_[fg]
Context    : 
Matches    : {0}



```

### Event #9: `item.completed` — `command_execution` (`item_7`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$p=Get-Content research/stage3d_qwen_signed_phase/qwen_connector.py; $p[23..142]; rg -n -A 47 '"'def format_chat_parts|def build_prefix_embeds' research/stage3d_qwen_signed_phase/qwen_connector.py"
```

**Aggregated Output:**

```text
MODEL_ID_DEFAULT = "Qwen/Qwen3.5-4B"
PINNED_REVISION_DEFAULT = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SYSTEM_PROMPT = (
    "你是电磁波形分析助手。仅根据已编码波形向量、公开相位向量、参考导频与问题描述�?
    "直接选择最符合条件的一项�?
)
MARKER_REGEX = re.compile(r"(<W1>|<W2>|<PHASE>|<REF>)")


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


_OOP_LA_CLS = None


405:    def format_chat_parts(self, user_prompt: str, extra_history: Optional[List[dict]] = None) -> List[str]:
406-        """Render using official Qwen3.5 chat_template with thinking disabled, then split on modality markers."""
407-        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
408-        if extra_history:
409-            messages.extend(extra_history)
410-        messages.append({"role": "user", "content": user_prompt})
411-        rendered = self.tokenizer.apply_chat_template(
412-            messages,
413-            tokenize=False,
414-            add_generation_prompt=True,
415-            enable_thinking=False,
416-        )
417-        parts = MARKER_REGEX.split(rendered)
418-        return [p for p in parts if p != ""]
419-
420-    def _make_position_ids(self, pos_1d: torch.Tensor, batch_size: int) -> torch.Tensor:
421-        """Build explicit position_ids avoiding stale self.model.rope_deltas."""
422-        # pos_1d: (L,) or (B, L)
423-        if pos_1d.ndim == 1:
424-            p2 = pos_1d.unsqueeze(0).expand(batch_size, -1)
425-        else:
426-            p2 = pos_1d
427-        if self._is_qwen35_mrope:
428-            # 4 axes: (text, temporal, height, width) -> identical 1D positions for text/connector embeddings
429-            return p2.unsqueeze(0).expand(4, batch_size, -1)
430-        return p2
431-
432:    def build_prefix_embeds(
433-        self,
434-        connector: Optional[SignedPhaseCacheConnector],
435-        user_prompt: str,
436-        z1_batch: Optional[torch.Tensor],
437-        z2_batch: Optional[torch.Tensor],
438-        phase_slots_batch: Optional[torch.Tensor] = None,
439-        ref_feats_list: Sequence[torch.Tensor] = (),
440-        text_only_mode: bool = False,
441-        extra_history: Optional[List[dict]] = None,
442-    ) -> torch.Tensor:
443-        """Assemble `(B, L, D)` prefix embeddings with zero leakage."""
444-        parts = self.format_chat_parts(user_prompt, extra_history=extra_history)
445-        embed_layer = self.lm.get_input_embeddings()
446-        batch_size = int(z1_batch.shape[0]) if z1_batch is not None else 1
447-
448-        if text_only_mode or connector is None:
449-            e_pref = torch.zeros((batch_size, 0, self.hidden_size), device=self.device, dtype=self.dtype)
450-            e_w1 = torch.zeros((batch_size, 0, self.hidden_size), device=self.device, dtype=self.dtype)
451-            e_w2 = torch.zeros((batch_size, 0, self.hidden_size), device=self.device, dtype=self.dtype)
452-            e_phase = torch.zeros((batch_size, 0, self.hidden_size), device=self.device, dtype=self.dtype)
453-            e_refs = [
454-                torch.zeros((batch_size, 0, self.hidden_size), device=self.device, dtype=self.dtype)
455-                for _ in ref_feats_list
456-            ]
457-        else:
458-            e_pref = connector.encode_prefix(batch_size).to(self.dtype)
459-            w1, w2 = connector.encode_windows(z1_batch.to(self.device), z2_batch.to(self.device))
460-            e_w1 = w1.to(self.dtype)
461-            e_w2 = w2.to(self.dtype)
462-            if phase_slots_batch is None:
463-                ps = torch.zeros((batch_size, 6), device=self.device, dtype=torch.float32)
464-            else:
465-                ps = phase_slots_batch.to(self.device, dtype=torch.float32)
466-            e_phase = connector.encode_phase(ps).to(self.dtype)
467-            e_refs = [connector.encode_refs(rf.to(self.device, dtype=torch.float32)).to(self.dtype) for rf in ref_feats_list]
468-
469-        chunks: List[torch.Tensor] = []
470-        ref_ptr = 0
471-        prefix_inserted = False
472-        for seg in parts:
473-            if seg == "<W1>":
474-                if not text_only_mode:
475-                    if not prefix_inserted:
476-                        chunks.append(e_pref)
477-                        prefix_inserted = True
478-                    chunks.append(e_w1)
479-            elif seg == "<W2>":
```

### Event #10: `item.completed` — `command_execution` (`item_8`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-Content research/matched_cache/learned.py; Get-Content research/stage3c/G/quotient.py; $p=Get-Content research/stage3_ministral_cache/schema.py; $p[0..65]; $p[149..168]'
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
        # Half-open quadrant [-45°,45°), and its three quarter-turn shifts.
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
"""Shared schemas, canonical QPSK sectors, prompts, and reference deduplication."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np


ROOTS = np.array([1.0 + 0j, 1j, -1.0 + 0j, -1j], dtype=np.complex128)
N_SAMPLES = 3
TWO_DB_SNR = 10.0 ** 0.2
ROTATIONS: Tuple[Tuple[int, int], ...] = ((0, 0), (0, 1), (1, 1))
ARMS: Tuple[str, ...] = ("none", "valid", "independent", "duplicate_copy", "fresh_resample")
PROTOCOLS: Tuple[str, ...] = ("repeat_x1", "independent_x3")

Q1_PROMPT = "第二个窗口的发射符号相对第一个窗口转了几个四分之一圈？"
Q1_CHOICES: Tuple[str, ...] = ("0圈（0°�?, "1圈（+90°�?, "2圈（180°�?, "3圈（+270°�?)

Q2_PROMPT = "第一个窗口原来发送的是哪个四相符号？"
Q2_CHOICES: Tuple[str, ...] = ("符号0�?�?, "符号1（i�?, "符号2�?1�?, "符号3�?i�?)

Q3_PROMPT_REPEAT = (
    "协议规定第三个窗口重发第一个窗口的符号，且已知当前时间间隔�?
    "在揭示第三个窗口观测前，其接收相位最可能落入哪个四分扇区�?
)
Q3_PROMPT_INDEP = (
    "协议规定第三个窗口发送与前两窗独立均匀的新四相符号�?
    "在揭示第三个窗口观测前，其接收相位最可能落入哪个四分扇区�?
)
Q3_PROMPT_REPEAT_AFTER_PILOT = (
    "已揭示上述参考导频；协议规定第三个窗口重发第一个窗口的符号�?
    "综合原缓存与导频后，第三个窗口接收相位最可能落入哪个四分扇区�?
)
Q3_CHOICES: Tuple[str, ...] = (
    "扇区0�?45°�?45°�?,
    "扇区1�?45°�?135°�?,
    "扇区2�?135°�?135°�?,
    "扇区3�?135°�?45°�?,
)


@dataclass(frozen=True)
class Reference:
    """Reference observation with explicit acquisition ID and channel-sharing declaration."""
    acquisition_id: int
    shares_channel: bool
    samples: np.ndarray  # shape (..., 3) or (3,) complex128
    t_p: float = 0.0
    snr_db: float = 2.0


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    snr_db: float
    h: int
    ell: int
    t_p: float
    drift_type: str  # "coherent" | "discrete" | "continuous" | "cfo_continuous"
    d: float = 0.0
    sigma_phi: float = 0.0
    omega: float = 0.0
    in_support_encoder: bool = False
    in_support_connector: bool = False
    question_order: str = "standard"

    @property
    def snr(self) -> float:


def sector_of_complex(z: np.ndarray) -> np.ndarray:
    """Canonical half-open QPSK sectors matching research/matched_cache/learned.py."""
    z_arr = np.asarray(z, dtype=np.complex128)
    u = z_arr.real + z_arr.imag
    v = z_arr.imag - z_arr.real
    out = np.zeros(z_arr.shape, dtype=np.int64)
    out[(u > 0.0) & (v >= 0.0)] = 1
    out[(u <= 0.0) & (v > 0.0)] = 2
    out[(u < 0.0) & (v <= 0.0)] = 3
    return out


def matched_sum(windows: np.ndarray) -> np.ndarray:
    """Orthogonal matched-filter statistic T = sum(y) / sqrt(3)."""
    arr = np.asarray(windows, dtype=np.complex128)
    return arr.sum(axis=-1) / np.sqrt(float(N_SAMPLES))


```

### Event #11: `item.completed` — `command_execution` (`item_6`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$p=Get-Content research/stage3d_qwen_signed_phase/data_stage3d.py; $p[0..168]; $p[258..374]'
```

**Aggregated Output:**

```text
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

    for b_idx, seed in enumerate(block_seeds):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(m_block)
        arr_tr = np.asarray([triples_base[int(p)] for p in perm], dtype=np.int64)
        x1_parts.append(arr_tr[:, 0])
        x2_parts.append(arr_tr[:, 1])
        g_parts.append(arr_tr[:, 2])
        x3_parts.append(rng.integers(4, size=m_block, dtype=np.int64))
        go_parts.append(rng.integers(4, size=m_block, dtype=np.int64))
        for t_cur in times[1:]:
            z_inc_parts[t_cur].append(rng.normal(scale=1.0, size=m_block))
        phi_ind_parts.append(rng.uniform(-math.pi, math.pi, size=m_block))
        for key in noise_parts:
            noise_parts[key].append(
                rng.normal(scale=noise_sd, size=(m_block, N_SAMPLES))
                + 1j * rng.normal(scale=noise_sd, size=(m_block, N_SAMPLES))
            )

    x1 = np.concatenate(x1_parts, axis=0)
    x2 = np.concatenate(x2_parts, axis=0)
    g = np.concatenate(g_parts, axis=0)
    x3_indep = np.concatenate(x3_parts, axis=0)
    g_other = np.concatenate(go_parts, axis=0)
    phi_indep = np.concatenate(phi_ind_parts, axis=0)
    z_incs = {t: np.concatenate(v, axis=0) for t, v in z_inc_parts.items()}
    noises = {k: np.concatenate(v, axis=0) for k, v in noise_parts.items()}

    out: Dict[float, MotherBatch] = {}
    for w_raw in omegas:
        omega = float(w_raw)
        cond_tag = f"{split_prefix}_w{omega:+.2f}_h{h}_l{ell}_tp{t_p:.1f}"
        cond = condition_spec_from_params(omega, h, ell, t_p, snr_db=snr_db, sigma_phi=sigma_phi, tag=cond_tag)
        cont_map = {times[0]: np.zeros(m_total, dtype=np.float64)}
        for t_prev, t_cur in zip(times[:-1], times[1:]):
            dt = float(t_cur - t_prev)
            mean = omega * dt
            sd_p = sigma_phi * math.sqrt(max(0.0, dt))
            cont_map[t_cur] = cont_map[t_prev] + mean + (sd_p * z_incs[t_cur] if sd_p > 0.0 else 0.0)
        phi_map = {t: cont_map[t] - cont_map[0.0] for t in times}
        phi_2 = phi_map[float(h)]
        phi_3 = phi_map[float(h + ell)]
        phi_p = phi_map[float(t_p)]

        def obs(u_labels: np.ndarray, phi: np.ndarray, n_arr: np.ndarray) -> np.ndarray:
            sig = (ROOTS[u_labels % 4] * np.exp(1j * phi)) / norm
            return sig[:, None] + n_arr

        y1 = obs(x1 + g, phi_map[0.0], noises["y1"])
        y2 = obs(x2 + g, phi_2, noises["y2"])
        y3_repeat = obs(x1 + g, phi_3, noises["y3_rep"])
        y3_indep = obs(x3_indep + g, phi_3, noises["y3_ind"])
        p_valid1 = obs(g, phi_p, noises["p1"])
        p_valid2 = obs(g, phi_p, noises["p2"])
        p_indep = obs(g_other, phi_indep, noises["pind"])
        mother_ids = [f"{cond_tag}:{i}" for i in range(m_total)]

        out[omega] = MotherBatch(
            split=split_prefix,
            condition=cond,
            mother_ids=mother_ids,
            x1=x1,
            x2=x2,
            g=g,
            x3_indep=x3_indep,
            y1=y1,
            y2=y2,
            y3_repeat=y3_repeat,
            y3_indep=y3_indep,
            p_valid1=p_valid1,
            p_valid2=p_valid2,
            p_indep=p_indep,
            phi_2=phi_2,
            phi_3=phi_3,
            phi_p=phi_p,
        )
    return out, block_seeds


def build_q2_arm_references(batch: MotherBatch, arm: str, coord_rot: int = 0) -> Tuple[Tuple[Reference, ...], bool]:
```

### Event #12: `item.completed` — `command_execution` (`item_9`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$p=Get-Content research/stage3d_qwen_signed_phase/recompute_all.py; $p[134..166]; $p[577..728]; $s=Get-Content research/stage3d_qwen_signed_phase/results/summary_master.json -Raw | ConvertFrom-Json; $s.PSObject.Properties.Name; Get-ChildItem -Force research/stage4_math_and_design/M1'
```

**Aggregated Output:**

```text
def compute_16way_group_avg(table_256: np.ndarray) -> np.ndarray:
    """Offline Work G 16-way C4 x C4 alignment average on a (..., 256, 4) Q3_repeat table."""
    orig_shape = table_256.shape
    assert orig_shape[-2:] == (256, 4)
    flat = table_256.reshape(-1, 256, 4)
    out = np.zeros_like(flat)
    for z1 in range(16):
        for z2 in range(16):
            acc = np.zeros((flat.shape[0], 4), dtype=np.float64)
            for a in range(4):
                rz1 = (z1 + 4 * a) % 16
                for b in range(4):
                    rz2 = (z2 + 4 * b) % 16
                    acc += np.roll(flat[:, 16 * rz1 + rz2, :], -a, axis=-1)
            out[:, 16 * z1 + z2, :] = acc / 16.0
    return out.reshape(orig_shape)


def compute_representative_readout(table_256: np.ndarray) -> np.ndarray:
    """Offline Work G single-representative F0 equivariant readout using canonical (z1%4, z2%4)."""
    orig_shape = table_256.shape
    assert orig_shape[-2:] == (256, 4)
    flat = table_256.reshape(-1, 256, 4)
    out = np.zeros_like(flat)
    for z1 in range(16):
        turn1, inner1 = divmod(z1, 4)
        for z2 in range(16):
            _, inner2 = divmod(z2, 4)
            out[:, 16 * z1 + z2, :] = np.roll(flat[:, 16 * inner1 + inner2, :], turn1, axis=-1)
    return out.reshape(orig_shape)


def evaluate_cell_paired(
def work_g_rotation_and_controls(
    raw_ll_table: np.ndarray,
    temps: np.ndarray,
    codes: np.ndarray,
    targets: np.ndarray,
    rot10_codes: np.ndarray,
    rot10_targets: np.ndarray,
    indep_raw_ll: np.ndarray,
    indep_targets: np.ndarray,
    shuffled_perm: np.ndarray,
    seeds: Sequence[int],
) -> dict:
    """Work G raw rotation equivariance, independent_x3, shuffled_Z, and offline 16-way Pf / F0."""
    n_arms, n_seeds, n_signs, _, _ = raw_ll_table.shape
    n_mothers = codes.shape[2]
    n_blocks = n_mothers // BLOCK_SIZE
    out: Dict[str, object] = {"arms": {}}

    for a, aname in enumerate(("phase", "zero")):
        raw_hits, raw_nlls = [], []
        pf_hits, pf_nlls = [], []
        f0_hits, f0_nlls = [], []
        shuf_hits, shuf_nlls = [], []
        indep_hits, indep_nlls = [], []
        indep_tvs = []

        rot01_max_abs, rot01_mean_abs, rot01_fail_020 = [], [], []
        rot11_max_abs, rot11_mean_abs, rot11_fail_020 = [], [], []
        rot10_max_abs, rot10_mean_abs, rot10_fail_020 = [], [], []
        worst_rot_dev = None

        for s_idx in range(n_seeds):
            t_val = float(temps[a, s_idx])
            p_raw = softmax_temperature(raw_ll_table[a, s_idx], t_val)  # (2, 256, 4)
            p_pf = compute_16way_group_avg(p_raw)
            p_f0 = compute_representative_readout(p_raw)
            p_ind = softmax_temperature(indep_raw_ll[a, s_idx], t_val)

            for sign_i in range(n_signs):
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

                idx_worst = int(np.argmax(d11))
                if worst_rot_dev is None or float(d11[idx_worst]) > worst_rot_dev["max_abs_diff"]:
                    worst_rot_dev = {
                        "seed": int(seeds[s_idx]),
                        "sign_idx": int(sign_i),
                        "mother_idx": idx_worst,
                        "code_00": int(codes[sign_i, 0, idx_worst]),
                        "code_11": int(codes[sign_i, 2, idx_worst]),
                        "probs_00": [float(v) for v in pr_rot[0, idx_worst]],
                        "probs_11": [float(v) for v in pr_rot[2, idx_worst]],
                        "max_abs_diff": float(d11[idx_worst]),
                    }

                sc_raw = score_vectors(pr_rot.reshape(-1, 4), targets[sign_i].reshape(-1))
                sc_pf = score_vectors(pf_rot.reshape(-1, 4), targets[sign_i].reshape(-1))
                sc_f0 = score_vectors(f0_rot.reshape(-1, 4), targets[sign_i].reshape(-1))
                raw_hits.append(sc_raw["per_item_hit"].reshape(3, n_mothers))
                raw_nlls.append(sc_raw["per_item_nll"].reshape(3, n_mothers))
                pf_hits.append(sc_pf["per_item_hit"].reshape(3, n_mothers))
                pf_nlls.append(sc_pf["per_item_nll"].reshape(3, n_mothers))
                f0_hits.append(sc_f0["per_item_hit"].reshape(3, n_mothers))
                f0_nlls.append(sc_f0["per_item_nll"].reshape(3, n_mothers))

                # Shuffled_Z control
                shuf_codes = np.take_along_axis(codes[sign_i], shuffled_perm[sign_i], axis=1)
                pr_shuf = p_raw[sign_i][shuf_codes]
                sc_shuf = score_vectors(pr_shuf.reshape(-1, 4), targets[sign_i].reshape(-1))
                shuf_hits.append(sc_shuf["per_item_hit"])
                shuf_nlls.append(sc_shuf["per_item_nll"])

                # independent_x3 control
                pr_ind = p_ind[sign_i][codes[sign_i]]
                sc_ind = score_vectors(pr_ind.reshape(-1, 4), indep_targets[sign_i].reshape(-1))
                indep_hits.append(sc_ind["per_item_hit"])
                indep_nlls.append(sc_ind["per_item_nll"])
                indep_tvs.append(0.5 * np.abs(pr_ind - 0.25).sum(axis=-1))

        # Block paired gain of offline 16-way group average vs raw host
        raw_m_hit = np.stack(raw_hits, axis=0).mean(axis=(0, 1))
        raw_m_nll = np.stack(raw_nlls, axis=0).mean(axis=(0, 1))
        pf_m_hit = np.stack(pf_hits, axis=0).mean(axis=(0, 1))
        pf_m_nll = np.stack(pf_nlls, axis=0).mean(axis=(0, 1))
        pf_blk_acc_gain = (pf_m_hit - raw_m_hit).reshape(n_blocks, BLOCK_SIZE).mean(axis=1)
        pf_blk_nll_improv = (raw_m_nll - pf_m_nll).reshape(n_blocks, BLOCK_SIZE).mean(axis=1)

        out["arms"][aname] = {
            "raw_rotation_checks": {
                "rot01_hold": {
                    "mean_max_abs": float(np.mean(rot01_mean_abs)),
                    "worst_max_abs": float(np.max(rot01_max_abs)),
                    "frac_gt_0p20": float(np.mean(rot01_fail_020)),
                },
                "rot11_roll": {
                    "mean_max_abs": float(np.mean(rot11_mean_abs)),
                    "worst_max_abs": float(np.max(rot11_max_abs)),
                    "frac_gt_0p20": float(np.mean(rot11_fail_020)),
                },
                "rot10_roll": {
                    "mean_max_abs": float(np.mean(rot10_mean_abs)),
                    "worst_max_abs": float(np.max(rot10_max_abs)),
                    "frac_gt_0p20": float(np.mean(rot10_fail_020)),
                },
                "worst_rot11_example": worst_rot_dev,
            },
            "controls": {
                "independent_x3": {
                    "acc": float(np.mean(indep_hits)),
                    "nll": float(np.mean(indep_nlls)),
                    "mean_tv_from_uniform": float(np.mean(indep_tvs)),
                    "max_tv_from_uniform": float(np.max(indep_tvs)),
                },
                "shuffled_Z": {
                    "acc": float(np.mean(shuf_hits)),
                    "nll": float(np.mean(shuf_nlls)),
                },
            },
            "raw_host": {"acc": float(np.mean(raw_hits)), "nll": float(np.mean(raw_nlls))},
            "offline_group_avg_16way": {
                "acc": float(np.mean(pf_hits)),
                "nll": float(np.mean(pf_nlls)),
                "pf_minus_raw_acc_t95": paired_t_ci(pf_blk_acc_gain),
                "raw_minus_pf_nll_t95": paired_t_ci(pf_blk_nll_improv),
            },
            "offline_representative_F0": {
                "acc": float(np.mean(f0_hits)),
                "nll": float(np.mean(f0_nlls)),
            },
        }
    return out


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
```

### Event #13: `item.completed` — `command_execution` (`item_10`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n 'direct_s17|cal_ref|factorial|metadata_M_sq|cell_idx|rotation' research/stage3d_qwen_signed_phase/recompute_all.py research/stage3d_qwen_signed_phase/data_stage3d.py; rg -n 'def .*direct|def .*cal_ref|direct_s17_table|def .*phase' research/stage3d_qwen_signed_phase -g '*.py'"
```

**Aggregated Output:**

```text
research/stage3d_qwen_signed_phase/data_stage3d.py:262:def derive_block_seeds(namespace: int, cell_idx: int, blocks: int) -> List[int]:
research/stage3d_qwen_signed_phase/data_stage3d.py:263:    """Deterministically derive unique 32-bit block seeds from (namespace, cell_idx, block_id)."""
research/stage3d_qwen_signed_phase/data_stage3d.py:265:        int(np.random.SeedSequence([int(namespace), int(cell_idx), int(b)]).generate_state(1)[0])
research/stage3d_qwen_signed_phase/data_stage3d.py:272:    cell_idx: int,
research/stage3d_qwen_signed_phase/data_stage3d.py:283:    block_seeds = derive_block_seeds(namespace, cell_idx, blocks)
research/stage3d_qwen_signed_phase/recompute_all.py:177:    """Evaluate paired phase vs zero arms over (signs x rotations x mothers) and 3 seeds."""
research/stage3d_qwen_signed_phase/recompute_all.py:216:    # Pooled over signs, rotations, and 3 seeds equally
research/stage3d_qwen_signed_phase/recompute_all.py:217:    # Block aggregation: first mother-level average across (seeds, signs, rotations), then 64-mother block means
research/stage3d_qwen_signed_phase/recompute_all.py:300:            "direct_s17_table": {k: sc_dir[k] for k in ("acc", "nll", "brier", "ece")},
research/stage3d_qwen_signed_phase/recompute_all.py:303:        "phase_minus_direct_s17": {
research/stage3d_qwen_signed_phase/recompute_all.py:307:        "zero_minus_direct_s17": {
research/stage3d_qwen_signed_phase/recompute_all.py:332:    cal_ref: np.ndarray,
research/stage3d_qwen_signed_phase/recompute_all.py:338:    """Work F signed CFO probe: 4-cell factorial, swapped-minus-matched NLL, orientation gain, and single-cache spotlight."""
research/stage3d_qwen_signed_phase/recompute_all.py:346:    target_diff = cal_ref[1] - cal_ref[0]  # D(z)
research/stage3d_qwen_signed_phase/recompute_all.py:353:        "cal_ref_overlap_mass": float(overlap.sum()),
research/stage3d_qwen_signed_phase/recompute_all.py:354:        "cal_ref_odd_norm_sq": norm_d2,
research/stage3d_qwen_signed_phase/recompute_all.py:371:        # 4-cell physical factorial: p[world_s, told_t, m, sector]
research/stage3d_qwen_signed_phase/recompute_all.py:376:        factorial_norm = {
research/stage3d_qwen_signed_phase/recompute_all.py:377:            "metadata_M_sq": float(np.mean(np.sum(m_eff * m_eff, axis=-1))),
research/stage3d_qwen_signed_phase/recompute_all.py:409:            "mean_rotation_10_tv": mean_rot_tv,
research/stage3d_qwen_signed_phase/recompute_all.py:428:            "factorial_mean_sq_norm": factorial_norm,
research/stage3d_qwen_signed_phase/recompute_all.py:432:    out["physical_cal_reference"] = _analyze_table(cal_ref)
research/stage3d_qwen_signed_phase/recompute_all.py:496:        "cal_ref_neg": [float(v) for v in cal_ref[0, c_spot]],
research/stage3d_qwen_signed_phase/recompute_all.py:497:        "cal_ref_pos": [float(v) for v in cal_ref[1, c_spot]],
research/stage3d_qwen_signed_phase/recompute_all.py:578:def work_g_rotation_and_controls(
research/stage3d_qwen_signed_phase/recompute_all.py:590:    """Work G raw rotation equivariance, independent_x3, shuffled_Z, and offline 16-way Pf / F0."""
research/stage3d_qwen_signed_phase/recompute_all.py:685:            "raw_rotation_checks": {
research/stage3d_qwen_signed_phase/recompute_all.py:853:        "work_a_physical_reference": work_a_metrics,
research/stage3d_qwen_signed_phase/recompute_all.py:890:        cal_ref=q3_npz["G1_main_extrap_both__cal_ref"],
research/stage3d_qwen_signed_phase/recompute_all.py:901:        cal_ref=q3_npz["H_c4_035__cal_ref"],
research/stage3d_qwen_signed_phase/recompute_all.py:907:    work_g = work_g_rotation_and_controls(
research/stage3d_qwen_signed_phase/recompute_all.py:945:                arm_res["direct_s17"] = {k: sc_d[k] for k in ("acc", "nll", "brier", "ece")}
research/stage3d_qwen_signed_phase\data_stage3d.py:161:def public_phase_slots(omega: float, h: int, ell: int, t_p: float, zero_arm: bool = False) -> np.ndarray:
research/stage3d_qwen_signed_phase\modal_runner.py:546:        def _cmp_tbl(direct_ll, ref_ll):
research/stage3d_qwen_signed_phase\recompute_all.py:300:            "direct_s17_table": {k: sc_dir[k] for k in ("acc", "nll", "brier", "ece")},
research/stage3d_qwen_signed_phase\qwen_connector.py:112:    def encode_phase(self, phase_slots: torch.Tensor) -> torch.Tensor:
research/stage3d_qwen_signed_phase\tests\test_stage3d_cpu.py:148:def test_recompute_all_clean_directory_roundtrip(tmp_path: Path):
```

### Event #14: `item.completed` — `command_execution` (`item_11`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'uv --version; uv cache dir; uv python list --only-installed; Get-ChildItem -Force -Name | Where-Object {$_ -match '"'AGENTS|uv|pyproject'}; Get-ChildItem -Force research/stage4_math_and_design/M1 -Name"
```

**Aggregated Output:**

```text
uv 0.11.21 (5aa65dd7a 2026-06-11 x86_64-pc-windows-msvc)
C:\Users\李则徐\AppData\Local\uv\cache
error: Failed to initialize cache at `C:\Users\李则徐\AppData\Local\uv\cache`
  Caused by: failed to open file `C:\Users\李则徐\AppData\Local\uv\cache\sdists-v9\.git`: 拒绝访问。 (os error 5)
```

### Event #15: `item.completed` — `command_execution` (`item_12`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'rg --files --hidden research/stage3d_qwen_signed_phase | Where-Object {$_ -match '"'\\.(npz|npy|json)"'$'"' -and "'$_ -notmatch '"'train_summary'}; "'$s=Get-Content research/stage3d_qwen_signed_phase/results/summary_master.json -Raw | ConvertFrom-Json; $s.cells.PSObject.Properties.Name; $s.cells.G1_main_extrap_both.PSObject.Properties.Name; $s.work_f_primary_040.PSObject.Properties.Name; $s.work_g_primary_040.arms.phase | ConvertTo-Json -Depth 5'
```

**Aggregated Output:**

```text
research/stage3d_qwen_signed_phase\blind_seed_manifest.json
research/stage3d_qwen_signed_phase\PROTOCOL_LOCK.json
research/stage3d_qwen_signed_phase\results\dev_pilot_lock.json
research/stage3d_qwen_signed_phase\results\bench_steady.json
research/stage3d_qwen_signed_phase\results\smoke_audit.json
research/stage3d_qwen_signed_phase\results\direct_vs_table_parity.json
research/stage3d_qwen_signed_phase\results\summary_master.json
research/stage3d_qwen_signed_phase\results\training_summary.json
research/stage3d_qwen_signed_phase\results\work_f_host_035_zero.json
research/stage3d_qwen_signed_phase\results\work_f_host_035_phase.json
G1_main_extrap_both
G2_interp_both
G3_extrap_angle
G4_extrap_freq
abs_omega
h
ell
t_p
cum_angle_abs
sigma2_t
blocks
mothers
logical_rows_per_arm
arms
paired_phase_minus_zero
per_seed
seed_dispersion
analytic_same_mothers
phase_minus_direct_s17
zero_minus_direct_s17
dual_practical_threshold
mothers
blocks
cal_ref_overlap_mass
cal_ref_odd_norm_sq
arms
physical_cal_reference
concrete_cache_spotlight_seed0
single_modality_contradiction_phase_arm
{
    "raw_rotation_checks":  {
                                "rot01_hold":  {
                                                   "mean_max_abs":  0.05913836761169119,
                                                   "worst_max_abs":  0.481667162214782,
                                                   "frac_gt_0p20":  0.05106949745547074
                                               },
                                "rot11_roll":  {
                                                   "mean_max_abs":  0.12627380352018333,
                                                   "worst_max_abs":  0.3800186428066911,
                                                   "frac_gt_0p20":  0.1544409987277354
                                               },
                                "rot10_roll":  {
                                                   "mean_max_abs":  0.123705202350481,
                                                   "worst_max_abs":  0.37518911347679684,
                                                   "frac_gt_0p20":  0.1513199745547074
                                               },
                                "worst_rot11_example":  {
                                                            "seed":  202622,
                                                            "sign_idx":  1,
                                                            "mother_idx":  39,
                                                            "code_00":  198,
                                                            "code_11":  10,
                                                            "probs_00":  [
                                                                             0.13948502604132854,
                                                                             0.10035237681789953,
                                                                             0.16885326578249082,
                                                                             0.5913093313582811
                                                                         ],
                                                            "probs_11":  [
                                                                             0.21129068855159,
                                                                             0.33100097499342845,
                                                                             0.2749615734448838,
                                                                             0.18274676301009782
                                                                         ],
                                                            "max_abs_diff":  0.3800186428066911
                                                        }
                            },
    "controls":  {
                     "independent_x3":  {
                                            "acc":  0.24909881255301103,
                                            "nll":  1.4237211668639687,
                                            "mean_tv_from_uniform":  0.09471768685807422,
                                            "max_tv_from_uniform":  0.29350237446691374
                                        },
                     "shuffled_Z":  {
                                        "acc":  0.25200779262086515,
                                        "nll":  1.4600696208746997
                                    }
                 },
    "raw_host":  {
                     "acc":  0.23798637616624257,
                     "nll":  1.4452285536590708
                 },
    "offline_group_avg_16way":  {
                                    "acc":  0.17050333969465647,
                                    "nll":  1.4113123906537837,
                                    "pf_minus_raw_acc_t95":  {
                                                                 "n_blocks":  131,
                                                                 "mean":  -0.0674830364715861,
                                                                 "block_sd":  0.01981303079493022,
                                                                 "se":  0.0017310725380280602,
                                                                 "t_crit_95":  1.9783804054470222,
                                                                 "lo95":  -0.07090775646122827,
                                                                 "hi95":  -0.06405831648194393
                                                             },
                                    "raw_minus_pf_nll_t95":  {
                                                                 "n_blocks":  131,
                                                                 "mean":  0.0339161630052869,
                                                                 "block_sd":  0.008374058906737819,
                                                                 "se":  0.0007316449237535328,
                                                                 "t_crit_95":  1.9783804054470222,
                                                                 "lo95":  0.03246869102438813,
                                                                 "hi95":  0.03536363498618567
                                                             }
                                },
    "offline_representative_F0":  {
                                      "acc":  0.2088303117048346,
                                      "nll":  1.411163365759204
                                  }
}
```

### Event #16: `item.completed` — `command_execution` (`item_14`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n 'cal_ref|direct_s17|cfo_train|enumerate\\(BLIND' research/stage3d_qwen_signed_phase/modal_runner.py; rg -n 'def .*q3|def .*table|def .*likelihood' research/stage3_ministral_cache research/stage3c -g '*.py' | Select-Object -First 72; "'$p=Get-Content research/stage3d_qwen_signed_phase/recompute_all.py; $p[47..98]; $p[337..432]; $p[857..880]'
```

**Aggregated Output:**

```text
research/stage3c\A\check_q2.py:30:def fit_likelihood(model, count, seed):
research/stage3_ministral_cache\analytic.py:101:def iq_likelihood(windows: np.ndarray, snr: float) -> np.ndarray:
research/stage3_ministral_cache\analytic.py:148:def _pilot_c4_likelihood(
research/stage3_ministral_cache\analytic.py:162:def _window2_phase_likelihood(t2: np.ndarray, snr: float, phi_grid: np.ndarray) -> np.ndarray:
research/stage3_ministral_cache\analytic.py:171:def readout_from_table(
research/stage3c\F\check_f.py:62:def validate(table):
research/stage3c\F\check_f.py:68:def fixed_cache(table, reference, weights):
research/stage3c\F\check_f.py:90:def evaluate_physical(table, cubes):
research/stage3c\G\check_g.py:22:def check(table):
research/stage3c\work_results\A\verify.py:19:def posterior(table, z1, weights, phase, rho, pilots, separate=False):
research/stage3c\work_results\F\verify_f.py:76:def audit(table, reference, weights, cache, target, draws):
research/stage3c\work_results\D\audit.py:64:def verify_orbits(table, decisions, read_pair):
research/stage3_ministral_cache\verify_theory.py:303:    def eval_full_iq_q3_and_pilot(t1, t2, grid):

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
```

### Event #17: `item.completed` — `command_execution` (`item_13`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-ChildItem -Force -Name; Get-ChildItem -Force $env:LOCALAPPDATA/uv; Get-Command python -All | Select-Object -ExpandProperty Source; Get-ChildItem Env: | Where-Object {$_.Name -match '"'"'^(UV_|VIRTUAL_ENV|CONDA_PREFIX)'"'}; rg --files --hidden -g '*q3*.npz' -g '*stage3d*.npz' -g '"'!**/.git/**'"' research"
```

**Aggregated Output:**

```text
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


    Ŀ¼: C:\Users\������\AppData\Local\uv


Mode                 LastWriteTime         Length Name                                                                 
----                 -------------         ------ ----                                                                 
d-----        2026/11/26     19:11                cache                                                                
C:\qazwsx\python.exe
C:\114514\python.exe
C:\376381\miniconda3\python.exe
C:\Users\������\AppData\Local\Microsoft\WindowsApps\python.exe
research\stage3d_qwen_signed_phase\results\blind_q3_scores.npz
research\stage3d_qwen_signed_phase\results\raw_q3_gpu.npz


```

### Event #18: `item.completed` — `command_execution` (`item_15`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-Content research/stage3_ministral_cache/analytic.py -TotalCount 260; $s=Get-Content research/stage3d_qwen_signed_phase/results/summary_master.json -Raw | ConvertFrom-Json; $s.work_f_primary_040.arms.phase | ConvertTo-Json -Depth 3; $s.work_f_primary_040.physical_cal_reference | ConvertTo-Json -Depth 3'
```

**Aggregated Output:**

```text
"""Exact analytic conditional solvers (Discrete C4 & Continuous Circular Quadrature) for Q1, Q2, Q3."""
import math
from typing import Optional, Sequence, Tuple
import numpy as np
from scipy.special import erfc, ndtr

from research.stage3_ministral_cache.schema import (
    ConditionSpec,
    N_SAMPLES,
    ROOTS,
    Reference,
    dedup_references,
    matched_sum,
)


QUAD_NODES_DEFAULT = 128


def awgn_sector_kernel(snr: float) -> np.ndarray:
    q = 0.5 * float(erfc(math.sqrt(snr / 2.0)))
    return np.array([(1.0 - q) ** 2, q * (1.0 - q), q ** 2, q * (1.0 - q)], dtype=np.float64)


def cont_sector_probs(theta: np.ndarray, snr: float) -> np.ndarray:
    scale = math.sqrt(2.0 * snr)
    p_plus = ndtr(scale * np.cos(theta + math.pi / 4.0))
    q_plus = ndtr(scale * np.sin(theta + math.pi / 4.0))
    return np.stack(
        [
            p_plus * q_plus,
            (1.0 - p_plus) * q_plus,
            (1.0 - p_plus) * (1.0 - q_plus),
            p_plus * (1.0 - q_plus),
        ],
        axis=-1,
    )


def discrete_walk_dist(d: float, steps: int) -> np.ndarray:
    k = 1.0 - 4.0 * d / 3.0
    c = 1.0 if steps <= 0 else (k ** steps)
    p = np.full(4, (1.0 - c) / 4.0, dtype=np.float64)
    p[0] += c
    return p


def wrapped_gaussian_weights(mean: float, variance: float, grid: int = QUAD_NODES_DEFAULT) -> Tuple[np.ndarray, np.ndarray]:
    phi = (2.0 * np.pi / grid) * np.arange(grid, dtype=np.float64) - np.pi
    if variance <= 0.0:
        w = np.zeros(grid, dtype=np.float64)
        idx = int(np.argmin(np.abs(((phi - mean + np.pi) % (2.0 * np.pi)) - np.pi)))
        w[idx] = 1.0
        # For exact deterministic nonzero phase mean (e.g. omega*t with sigma=0),
        # return exact angle `[mean]` if caller checks variance==0, else grid.
        return phi, w
    if not np.isfinite(variance):
        return phi, np.full(grid, 1.0 / grid, dtype=np.float64)
    sd = math.sqrt(variance)
    n_img = max(1, math.ceil((10.0 * sd + np.pi) / (2.0 * np.pi)))
    disp = ((phi - mean + np.pi) % (2.0 * np.pi)) - np.pi
    w = np.zeros(grid, dtype=np.float64)
    for m in range(-n_img, n_img + 1):
        w += np.exp(-0.5 * ((disp + 2.0 * np.pi * m) / sd) ** 2)
    w /= w.sum()
    return phi, w


def wrapped_sector_quad_probs(mean: float, variance: float) -> np.ndarray:
    """Exact mass of wrapped N(mean, variance) inside each QPSK sector [-pi/4, pi/4) + k*pi/2."""
    if variance <= 0.0:
        out = np.zeros(4, dtype=np.float64)
        z = complex(math.cos(mean), math.sin(mean))
        u = z.real + z.imag
        v = z.imag - z.real
        if u > 0 and v >= 0:
            out[1] = 1.0
        elif u <= 0 and v > 0:
            out[2] = 1.0
        elif u < 0 and v <= 0:
            out[3] = 1.0
        else:
            out[0] = 1.0
        return out
    sd = math.sqrt(variance)
    n_img = max(1, math.ceil((10.0 * sd + np.pi) / (2.0 * np.pi)))
    out = np.zeros(4, dtype=np.float64)
    for k in range(4):
        lo_base = -0.25 * math.pi + 0.5 * math.pi * k
        hi_base = lo_base + 0.5 * math.pi
        mass = 0.0
        for m in range(-n_img, n_img + 1):
            lo = lo_base + 2.0 * math.pi * m
            hi = hi_base + 2.0 * math.pi * m
            mass += float(ndtr((hi - mean) / sd) - ndtr((lo - mean) / sd))
        out[k] = mass
    out /= out.sum()
    return out


def iq_likelihood(windows: np.ndarray, snr: float) -> np.ndarray:
    """Exact 4-class QPSK posterior a_j(u) from 3-sample complex windows (M, 3)."""
    t = matched_sum(windows)
    logw = 2.0 * snr * (t[:, None] * ROOTS[None, :].conj()).real
    logw -= logw.max(axis=1, keepdims=True)
    w = np.exp(logw)
    return w / np.sort(w, axis=1).sum(axis=1, keepdims=True)


def diff_posterior(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Exact sorted-sum circular difference P(right - left = k mod 4)."""
    sums = []
    for k in range(4):
        prod = left * np.roll(right, -k, axis=-1)
        sums.append(np.sort(prod, axis=-1).sum(axis=-1))
    return np.stack(sums, axis=-1)


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.maximum(arr, 1e-300)
    return arr / np.sort(arr, axis=-1).sum(axis=-1, keepdims=True)


def _gaussian_transition_matrix(
    phi_from: np.ndarray,
    phi_to: np.ndarray,
    mean_shift: float,
    variance: float,
) -> np.ndarray:
    """Matrix K[i, j] proportional to wrapped N(phi_to[i] - phi_from[j] - mean_shift; variance)."""
    if variance <= 0.0:
        # Nearest-node delta fallback when dt=0 or sigma=0
        mat = np.zeros((len(phi_to), len(phi_from)), dtype=np.float64)
        for j, pf in enumerate(phi_from):
            disp = np.abs(((phi_to - pf - mean_shift + np.pi) % (2.0 * np.pi)) - np.pi)
            mat[int(np.argmin(disp)), j] = 1.0
        return mat
    sd = math.sqrt(variance)
    n_img = max(1, math.ceil((10.0 * sd + np.pi) / (2.0 * np.pi)))
    disp = ((phi_to[:, None] - phi_from[None, :] - mean_shift + np.pi) % (2.0 * np.pi)) - np.pi
    mat = np.zeros_like(disp)
    for m in range(-n_img, n_img + 1):
        mat += np.exp(-0.5 * ((disp + 2.0 * np.pi * m) / sd) ** 2)
    mat /= mat.sum(axis=0, keepdims=True)
    return mat


def _pilot_c4_likelihood(
    t_pilot_sum: Optional[np.ndarray],
    snr: float,
    phi_grid: np.ndarray,
) -> Optional[np.ndarray]:
    """L_p(phi_p, g) for pilot sum T_p, shape (M, G, 4), or None if no valid pilots."""
    if t_pilot_sum is None:
        return None
    phases = ROOTS[None, :] * np.exp(1j * phi_grid)[:, None]  # (G, 4)
    lw = 2.0 * snr * np.real(t_pilot_sum[:, None, None] * np.conj(phases)[None, :, :])  # (M, G, 4)
    lw -= np.max(lw, axis=(1, 2), keepdims=True)
    return np.exp(lw)


def _window2_phase_likelihood(t2: np.ndarray, snr: float, phi_grid: np.ndarray) -> np.ndarray:
    """C4-periodic likelihood L_2(phi_2) = (1/4) sum_v exp(2*snr*Re(T_2 * conj(s_v) * exp(-i*phi_2)))."""
    phases = ROOTS[None, :] * np.exp(1j * phi_grid)[:, None]  # (G, 4)
    lw = 2.0 * snr * np.real(t2[:, None, None] * np.conj(phases)[None, :, :])  # (M, G, 4)
    m_lw = np.max(lw, axis=2, keepdims=True)
    lik = np.exp(lw - m_lw).sum(axis=2) * np.exp(m_lw[:, :, 0] - np.max(lw, axis=(1, 2))[:, None])
    return 0.25 * lik  # (M, G)


def readout_from_table(
    table_z1: np.ndarray,
    table_z2: np.ndarray,
    references: Sequence[Reference],
    cond: ConditionSpec,
    protocol: str = "repeat_x1",
    grid: int = QUAD_NODES_DEFAULT,
    allow_double_count_unsafe: bool = False,
) -> dict:
    """Exact readout conditioned ONLY on discrete cache posteriors `(a1, a2)` + deduped refs."""
    a1 = np.asarray(table_z1, dtype=np.float64)
    a2 = np.asarray(table_z2, dtype=np.float64)
    n_obs = len(a1)
    snr = cond.snr

    if not allow_double_count_unsafe:
        valid_refs, _, _, _ = dedup_references(references)
    else:
        valid_refs = [r for r in references if r.shares_channel]

    if valid_refs:
        t_p_sum = sum(matched_sum(r.samples) for r in valid_refs)
    else:
        t_p_sum = None

    if cond.drift_type == "coherent":
        q1 = diff_posterior(a1, a2)
        if t_p_sum is None:
            q2 = np.full((n_obs, 4), 0.25, dtype=np.float64)
        else:
            lw_p = 2.0 * snr * np.real(t_p_sum[:, None] * ROOTS[None, :].conj())
            lw_p -= lw_p.max(axis=1, keepdims=True)
            bp = np.exp(lw_p)
            bp /= np.sort(bp, axis=1).sum(axis=1, keepdims=True)
            q2 = diff_posterior(bp, a1)
        if protocol == "repeat_x1":
            dk = awgn_sector_kernel(snr)
            kmat = np.array([np.roll(dk, w) for w in range(4)], dtype=np.float64)
            q3 = a1 @ kmat
            mu3 = a1 @ ROOTS
        elif protocol == "independent_x3":
            q3 = np.full((n_obs, 4), 0.25, dtype=np.float64)
            mu3 = np.zeros(n_obs, dtype=np.complex128)
        else:
            raise ValueError(protocol)
        q3_after = q3.copy()

    elif cond.drift_type == "discrete":
        k_rel = 1.0 - 4.0 * cond.d / 3.0
        ch = 1.0 if cond.h == 0 else (k_rel ** cond.h)
        chl = 1.0 if (cond.h + cond.ell) == 0 else (k_rel ** (cond.h + cond.ell))
        q1_raw = diff_posterior(a1, a2)
        q1 = ch * q1_raw + (1.0 - ch) / 4.0

        if t_p_sum is None:
            q2 = np.full((n_obs, 4), 0.25, dtype=np.float64)
        else:
            p_0tp = discrete_walk_dist(cond.d, int(round(cond.t_p)))
            lw_p = 2.0 * snr * np.real(t_p_sum[:, None] * ROOTS[None, :].conj())
            lw_p -= lw_p.max(axis=1, keepdims=True)
            bp_raw = np.exp(lw_p)
            bp_raw /= np.sort(bp_raw, axis=1).sum(axis=1, keepdims=True)
            # Propagate by p_0tp: P(g_p - g = w) -> bp[g] = sum_w p_0tp[w] * bp_raw[(g+w)%4]
            bp = np.stack([sum(p_0tp[w] * bp_raw[:, (g + w) % 4] for w in range(4)) for g in range(4)], axis=1)
            bp /= np.sort(bp, axis=1).sum(axis=1, keepdims=True)
            q2 = diff_posterior(bp, a1)

        if protocol == "repeat_x1":
            dk = awgn_sector_kernel(snr)
            pu3 = chl * a1 + (1.0 - chl) / 4.0
            kmat = np.array([np.roll(dk, w) for w in range(4)], dtype=np.float64)
            q3 = pu3 @ kmat
            mu3 = chl * (a1 @ ROOTS)
        elif protocol == "independent_x3":
            q3 = np.full((n_obs, 4), 0.25, dtype=np.float64)
            mu3 = np.zeros(n_obs, dtype=np.complex128)
        else:
            raise ValueError(protocol)
        q3_after = q3.copy()

    else:
        # continuous Brownian (+ omega)
        var2 = float(cond.h) * (cond.sigma_phi ** 2)
        var3 = float(cond.h + cond.ell) * (cond.sigma_phi ** 2)
        varp = float(cond.t_p) * (cond.sigma_phi ** 2)
        mean2 = cond.omega * float(cond.h)
        mean3 = cond.omega * float(cond.h + cond.ell)
        meanp = cond.omega * float(cond.t_p)

        # Q1: phase rotation weights of a2 shifted by phi_2
{
    "seed_mean_orientation_gain":  0.009819795034315315,
    "seed_mean_residual_ratio":  0.05485821545817657,
    "seed_mean_sign_tv":  0.04003317991781616,
    "seed_mean_swapped_minus_matched_nll":  0.01836983052292875,
    "seed_mean_matched_acc":  0.23739662849872775,
    "seed_mean_swapped_acc":  0.20769720101781167,
    "seed_averaged_table_metrics":  {
                                        "orientation_gain":  0.009819795034315315,
                                        "residual_ratio":  0.033809123399957446,
                                        "mean_sign_tv":  0.025698790712428962,
                                        "mean_rotation_10_tv":  0.08307747554299151,
                                        "matched_acc":  0.18189408396946566,
                                        "swapped_acc":  0.18010496183206107,
                                        "matched_nll":  1.4147270812818875,
                                        "swapped_nll":  1.4308784896540296,
                                        "cell_nll_s_world_t_told":  {
                                                                        "--":  1.4238574581597658,
                                                                        "-+":  1.442908222764981,
                                                                        "+-":  1.4188487565430787,
                                                                        "++":  1.4055967044040094
                                                                    },
                                        "cell_acc_s_world_t_told":  {
                                                                        "--":  0.19537213740458015,
                                                                        "-+":  0.19119751908396945,
                                                                        "+-":  0.16901240458015268,
                                                                        "++":  0.16841603053435114
                                                                    },
                                        "swapped_minus_matched_nll_t95":  {
                                                                              "n_blocks":  131,
                                                                              "mean":  0.016151408372142323,
                                                                              "block_sd":  0.005366196339988422,
                                                                              "se":  0.0004688467511087499,
                                                                              "t_crit_95":  1.9783804054470222,
                                                                              "lo95":  0.015223851146591276,
                                                                              "hi95":  0.017078965597693372
                                                                          },
                                        "swapped_minus_matched_nll_boot95":  {
                                                                                 "n_blocks":  131,
                                                                                 "resamples":  10000,
                                                                                 "seed":  2026311,
                                                                                 "mean":  0.016151408372142323,
                                                                                 "lo95":  0.015252125141685327,
                                                                                 "hi95":  0.017072632295022542
                                                                             },
                                        "factorial_mean_sq_norm":  {
                                                                       "metadata_M_sq":  0.00018537805471633675,
                                                                       "wave_W_sq":  0.001119772417564818,
                                                                       "interaction_I_sq":  6.3625303969710041E-05
                                                                   }
                                    },
    "per_seed":  [
                     {
                         "orientation_gain":  -0.0014714446816086089,
                         "residual_ratio":  0.06187061738575668,
                         "mean_sign_tv":  0.04416703262776987,
                         "mean_rotation_10_tv":  0.07111297937771423,
                         "matched_acc":  0.2637762404580153,
                         "swapped_acc":  0.24015982824427481,
                         "matched_nll":  1.3953781588415262,
                         "swapped_nll":  1.3925059876056152,
                         "cell_nll_s_world_t_told":  {
                                                         "--":  1.3969048539674522,
                                                         "-+":  1.3927726795721096,
                                                         "+-":  1.3922392956391203,
                                                         "++":  1.3938514637156003
                                                     },
                         "cell_acc_s_world_t_told":  {
                                                         "--":  0.2526240458015267,
                                                         "-+":  0.23079675572519084,
                                                         "+-":  0.24952290076335878,
                                                         "++":  0.2749284351145038
                                                     },
                         "swapped_minus_matched_nll_t95":  {
                                                               "n_blocks":  131,
                                                               "mean":  -0.0028721712359112347,
                                                               "block_sd":  0.010204875000990796,
                                                               "se":  0.0008916040686084472,
                                                               "t_crit_95":  1.9783804054470222,
                                                               "lo95":  -0.004636103254663029,
                                                               "hi95":  -0.0011082392171594404
                                                           },
                         "swapped_minus_matched_nll_boot95":  {
                                                                  "n_blocks":  131,
                                                                  "resamples":  10000,
                                                                  "seed":  2026311,
                                                                  "mean":  -0.0028721712359112347,
                                                                  "lo95":  -0.0045970141698002875,
                                                                  "hi95":  -0.001101467741930932
                                                              },
                         "factorial_mean_sq_norm":  {
                                                        "metadata_M_sq":  0.0005168668016449173,
                                                        "wave_W_sq":  0.0008460693957756728,
                                                        "interaction_I_sq":  0.00025319832928274094
                                                    },
                         "seed":  202611
                     },
                     {
                         "orientation_gain":  0.021043669725492792,
                         "residual_ratio":  0.054284693290079525,
                         "mean_sign_tv":  0.04143398575874327,
                         "mean_rotation_10_tv":  0.17702509924593504,
                         "matched_acc":  0.2109375,
                         "swapped_acc":  0.1868439885496183,
                         "matched_nll":  1.4831983487121236,
                         "swapped_nll":  1.5250227393517006,
                         "cell_nll_s_world_t_told":  {
                                                         "--":  1.4571247741400677,
                                                         "-+":  1.515881641651842,
                                                         "+-":  1.5341638370515591,
                                                         "++":  1.5092719232841796
                                                     },
                         "cell_acc_s_world_t_told":  {
                                                         "--":  0.25656011450381677,
                                                         "-+":  0.21827290076335878,
                                                         "+-":  0.15541507633587787,
                                                         "++":  0.1653148854961832
                                                     },
                         "swapped_minus_matched_nll_t95":  {
                                                               "n_blocks":  131,
                                                               "mean":  0.04182439063957688,
                                                               "block_sd":  0.009588744494020364,
                                                               "se":  0.0008377724962711783,
                                                               "t_crit_95":  1.9783804054470222,
                                                               "lo95":  0.04016695794873154,
                                                               "hi95":  0.04348182333042222
                                                           },
                         "swapped_minus_matched_nll_boot95":  {
                                                                  "n_blocks":  131,
                                                                  "resamples":  10000,
                                                                  "seed":  2026311,
                                                                  "mean":  0.04182439063957688,
                                                                  "lo95":  0.040214628810498844,
                                                                  "hi95":  0.04346096571259359
                                                              },
                         "factorial_mean_sq_norm":  {
                                                        "metadata_M_sq":  0.0005099856833317363,
                                                        "wave_W_sq":  0.004967892858831495,
                                                        "interaction_I_sq":  0.00017117050311773901
                                                    },
                         "seed":  202622
                     },
                     {
                         "orientation_gain":  0.009887160059061767,
                         "residual_ratio":  0.04841933569869349,
                         "mean_sign_tv":  0.034498521366935334,
                         "mean_rotation_10_tv":  0.17719567418333743,
                         "matched_acc":  0.23747614503816794,
                         "swapped_acc":  0.19608778625954199,
                         "matched_nll":  1.4578920382438887,
                         "swapped_nll":  1.4740493104090095,
                         "cell_nll_s_world_t_told":  {
                                                         "--":  1.5139464289535944,
                                                         "-+":  1.5279577251349157,
                                                         "+-":  1.4201408956831034,
                                                         "++":  1.4018376475341834
                                                     },
                         "cell_acc_s_world_t_told":  {
                                                         "--":  0.23723759541984732,
                                                         "-+":  0.17986641221374045,
                                                         "+-":  0.21230916030534353,
                                                         "++":  0.23771469465648856
                                                     },
                         "swapped_minus_matched_nll_t95":  {
                                                               "n_blocks":  131,
                                                               "mean":  0.0161572721651206,
                                                               "block_sd":  0.008305731943857772,
                                                               "se":  0.0007256751692887703,
                                                               "t_crit_95":  1.9783804054470222,
                                                               "lo95":  0.014721610629480247,
                                                               "hi95":  0.017592933700760954
                                                           },
                         "swapped_minus_matched_nll_boot95":  {
                                                                  "n_blocks":  131,
                                                                  "resamples":  10000,
                                                                  "seed":  2026311,
                                                                  "mean":  0.0161572721651206,
                                                                  "lo95":  0.014744847891171175,
                                                                  "hi95":  0.017592068840072975
                                                              },
                         "factorial_mean_sq_norm":  {
                                                        "metadata_M_sq":  0.00031739929000029163,
                                                        "wave_W_sq":  0.0040141403690335245,
                                                        "interaction_I_sq":  0.00017054258057424908
                                                    },
                         "seed":  202633
                     }
                 ]
}
{
    "orientation_gain":  1.0,
    "residual_ratio":  0.0,
    "mean_sign_tv":  0.6943259442324701,
    "mean_rotation_10_tv":  0.0,
    "matched_acc":  0.6461712786259542,
    "swapped_acc":  0.03625954198473282,
    "matched_nll":  0.94128711026503,
    "swapped_nll":  2.996389774546989,
    "cell_nll_s_world_t_told":  {
                                    "--":  0.9394448567112544,
                                    "-+":  3.002745926720927,
                                    "+-":  2.9900336223730517,
                                    "++":  0.9431293638188055
                                },
    "cell_acc_s_world_t_told":  {
                                    "--":  0.6474236641221374,
                                    "-+":  0.03661736641221374,
                                    "+-":  0.03590171755725191,
                                    "++":  0.644918893129771
                                },
    "swapped_minus_matched_nll_t95":  {
                                          "n_blocks":  131,
                                          "mean":  2.055102664281959,
                                          "block_sd":  0.17310562053315998,
                                          "se":  0.015124308289064813,
                                          "t_crit_95":  1.9783804054470222,
                                          "lo95":  2.025181029116933,
                                          "hi95":  2.0850242994469848
                                      },
    "swapped_minus_matched_nll_boot95":  {
                                             "n_blocks":  131,
                                             "resamples":  10000,
                                             "seed":  2026311,
                                             "mean":  2.055102664281959,
                                             "lo95":  2.0256386804496875,
                                             "hi95":  2.084100689943572
                                         },
    "factorial_mean_sq_norm":  {
                                   "metadata_M_sq":  0.20160318171766897,
                                   "wave_W_sq":  2.332636364430212E-05,
                                   "interaction_I_sq":  2.8427318619844037E-05
                               }
}
```

### Event #19: `item.completed` — `command_execution` (`item_16`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n --no-ignore -g '*.py' 'cal_ref|direct_table_probs|NS_CFO_TRAIN|generate_paired_cfo_blocks' research/stage3d_qwen_signed_phase; Get-Item research/stage3d_qwen_signed_phase/results/blind_q3_scores.npz; Get-ChildItem -Force (uv cache dir); Get-ChildItem -Force 'C:/Users/李则徐/AppData/Roaming/uv/python'"
```

**Aggregated Output:**

```text
research/stage3d_qwen_signed_phase\analysis_stage3d.py:35:    generate_paired_cfo_blocks,
research/stage3d_qwen_signed_phase\analysis_stage3d.py:69:        b_map, _ = generate_paired_cfo_blocks(
research/stage3d_qwen_signed_phase\analysis_stage3d.py:111:    """Generate paired blind codes/targets, analytic s17 & full_iq probs, and independent cal_ref."""
research/stage3d_qwen_signed_phase\analysis_stage3d.py:113:    b_map, block_seeds = generate_paired_cfo_blocks(
research/stage3d_qwen_signed_phase\analysis_stage3d.py:157:    cal_ref, cal_mass = calibrate_signed_reference_table(
research/stage3d_qwen_signed_phase\analysis_stage3d.py:171:        "direct_table_probs": direct_probs,
research/stage3d_qwen_signed_phase\analysis_stage3d.py:176:        "cal_ref": cal_ref,
research/stage3d_qwen_signed_phase\analysis_stage3d.py:189:    b_map, seeds = generate_paired_cfo_blocks(
research/stage3d_qwen_signed_phase\analysis_stage3d.py:376:    q2_map, _ = generate_paired_cfo_blocks(
research/stage3d_qwen_signed_phase\data_stage3d.py:43:NS_CFO_TRAIN = 930101
research/stage3d_qwen_signed_phase\data_stage3d.py:270:def generate_paired_cfo_blocks(
research/stage3d_qwen_signed_phase\data_stage3d.py:459:        tr_map, _ = generate_paired_cfo_blocks(NS_CFO_TRAIN, t_idx, max_blocks, h, ell, tp, TRAIN_OMEGAS, split_prefix="cfo_train")
research/stage3d_qwen_signed_phase\data_stage3d.py:461:        dv_map, _ = generate_paired_cfo_blocks(NS_CFO_DEV, t_idx, max_blocks, h, ell, tp, DEV_OMEGAS, split_prefix="cfo_dev")
research/stage3d_qwen_signed_phase\data_stage3d.py:464:    pilot_map, _ = generate_paired_cfo_blocks(NS_DEV_PILOT, 0, max_blocks, 2, 2, 4.0, DEV_OMEGAS, split_prefix="dev_pilot")
research/stage3d_qwen_signed_phase\data_stage3d.py:468:        bmap, _ = generate_paired_cfo_blocks(
research/stage3d_qwen_signed_phase\data_stage3d.py:473:    q2_map, _ = generate_paired_cfo_blocks(NS_BLIND_Q2_C4, 0, max_blocks, 2, 2, 4.0, (0.35,), split_prefix="q2_c4")
research/stage3d_qwen_signed_phase\recompute_all.py:332:    cal_ref: np.ndarray,
research/stage3d_qwen_signed_phase\recompute_all.py:346:    target_diff = cal_ref[1] - cal_ref[0]  # D(z)
research/stage3d_qwen_signed_phase\recompute_all.py:353:        "cal_ref_overlap_mass": float(overlap.sum()),
research/stage3d_qwen_signed_phase\recompute_all.py:354:        "cal_ref_odd_norm_sq": norm_d2,
research/stage3d_qwen_signed_phase\recompute_all.py:432:    out["physical_cal_reference"] = _analyze_table(cal_ref)
research/stage3d_qwen_signed_phase\recompute_all.py:496:        "cal_ref_neg": [float(v) for v in cal_ref[0, c_spot]],
research/stage3d_qwen_signed_phase\recompute_all.py:497:        "cal_ref_pos": [float(v) for v in cal_ref[1, c_spot]],
research/stage3d_qwen_signed_phase\recompute_all.py:853:        "work_a_physical_reference": work_a_metrics,
research/stage3d_qwen_signed_phase\recompute_all.py:877:                direct_probs=q3_npz[f"{cid}__direct_table_probs"],
research/stage3d_qwen_signed_phase\recompute_all.py:890:        cal_ref=q3_npz["G1_main_extrap_both__cal_ref"],
research/stage3d_qwen_signed_phase\recompute_all.py:901:        cal_ref=q3_npz["H_c4_035__cal_ref"],
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:34:    NS_CFO_TRAIN,
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:44:    generate_paired_cfo_blocks,
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:196:        tr_map, _ = generate_paired_cfo_blocks(
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:197:            NS_CFO_TRAIN, t_idx, train_blocks, h, ell, tp, TRAIN_OMEGAS, split_prefix="cfo_train"
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:202:        dv_map, _ = generate_paired_cfo_blocks(
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:261:    dv_cfo_map, _ = generate_paired_cfo_blocks(NS_CFO_DEV, 0, 1, 2, 2, 4.0, (0.18,), split_prefix="smoke_cfo")
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:985:    g1_sample, _ = generate_paired_cfo_blocks(
research/stage3d_qwen_signed_phase\train_eval_stage3d.py:1064:    q2_map, _ = generate_paired_cfo_blocks(
research/stage3d_qwen_signed_phase\modal_runner.py:373:        generate_paired_cfo_blocks,
research/stage3d_qwen_signed_phase\modal_runner.py:397:    dv_map, _ = generate_paired_cfo_blocks(NS_CFO_DEV, 0, 1, 2, 2, 4.0, (0.18,), split_prefix="bench")
research/stage3d_qwen_signed_phase\modal_runner.py:526:        g1_map, _ = generate_paired_cfo_blocks(NS_BLIND_G1, 0, 1, 2, 2, 4.0, (0.40,), split_prefix="g1_parity_bench")
research/stage3d_qwen_signed_phase\tests\test_stage3d_cpu.py:24:    generate_paired_cfo_blocks,
research/stage3d_qwen_signed_phase\tests\test_stage3d_cpu.py:50:    bmap, seeds = generate_paired_cfo_blocks(NS_BLIND_G1, 0, 2, 2, 2, 4.0, (-0.40, 0.40), split_prefix="G1_test")
research/stage3d_qwen_signed_phase\tests\test_stage3d_cpu.py:103:    bmap, _ = generate_paired_cfo_blocks(950405, 0, 1, 2, 2, 4.0, (0.35,))
research/stage3d_qwen_signed_phase\tests\test_stage3d_cpu.py:158:        cal_ref = rng.dirichlet(np.ones(4), size=(2, 256))
research/stage3d_qwen_signed_phase\tests\test_stage3d_cpu.py:160:        # Make phase arm match cal_ref closely and zero arm nearly uniform -> strong positive gain
research/stage3d_qwen_signed_phase\tests\test_stage3d_cpu.py:162:        raw_ll[0] = np.log(np.clip(cal_ref, 1e-4, 1.0))[None, ...] + 0.01 * rng.normal(size=(3, 2, 256, 4))
research/stage3d_qwen_signed_phase\tests\test_stage3d_cpu.py:169:                    targets[s_i, r_i, m_i] = int(np.argmax(cal_ref[s_i, codes[s_i, r_i, m_i]]))
research/stage3d_qwen_signed_phase\tests\test_stage3d_cpu.py:173:        q3_bundle[f"{cid}__cal_ref"] = cal_ref
research/stage3d_qwen_signed_phase\tests\test_stage3d_cpu.py:175:        q3_bundle[f"{cid}__direct_table_probs"] = np.full((2, 3, n_m, 4), 0.25)


    Ŀ¼: C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage3d_qwen_signed_phase\results


Mode                 LastWriteTime         Length Name                                                                 
----                 -------------         ------ ----                                                                 
-a----        2026/11/26     19:11        2560068 blind_q3_scores.npz                                                  
Get-ChildItem : �Ҳ���·����C:\Users\李则徐\AppData\Local\uv\cache������Ϊ��·�������ڡ�
����λ�� ��:2 �ַ�: 207
+ ... ults/blind_q3_scores.npz; Get-ChildItem -Force (uv cache dir); Get-Ch ...
+                               ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : ObjectNotFound: (C:\Users\李则徐\AppData\Local\uv\cache:String) [Get-ChildItem], ItemN
otFo    undException
    + FullyQualifiedErrorId : PathNotFound,Microsoft.PowerShell.Commands.GetChildItemCommand
 
Get-ChildItem : �Ҳ���·����C:\Users\������\AppData\Roaming\uv\python������Ϊ��·�������ڡ�
����λ�� ��:2 �ַ�: 244
+ ... cache dir); Get-ChildItem -Force 'C:/Users/������/AppData/Roaming/uv/pyt ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : ObjectNotFound: (C:\Users\������\AppData\Roaming\uv\python:String) [Get-ChildItem], ItemN
ot    FoundException
    + FullyQualifiedErrorId : PathNotFound,Microsoft.PowerShell.Commands.GetChildItemCommand
 


```

### Event #20: `item.completed` — `command_execution` (`item_18`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$p=Get-Content research/stage3d_qwen_signed_phase/analysis_stage3d.py; $p[0..184]; $p=Get-Content research/stage3_ministral_cache/analytic.py; $p[252..362]'
```

**Aggregated Output:**

```text
"""Stage 3D CPU physical calibration, analytic baselines, dev-pilot sample-size lock, and master bundling."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import chi2

from research.stage3_ministral_cache.analytic import (
    readout_from_table,
    solve_full_iq_known_param,
)
from research.stage3_ministral_cache.data import generate_mothers
from research.stage3_ministral_cache.schema import (
    ROOTS,
    ROTATIONS,
    matched_sum,
    sector_of_complex,
)
from research.stage3c.A.q2_reader import read as q2_reader_read
from research.stage3d_qwen_signed_phase.data_stage3d import (
    ARM_NAMES,
    BLIND_CELLS,
    CONNECTOR_SEEDS,
    NS_BLIND_Q2_C4,
    NS_DEV_PILOT,
    NS_REF_CALIB,
    Q2_BLIND_ARMS,
    REPLAY_TRAIN_CONDITIONS,
    build_q2_arm_references,
    generate_paired_cfo_blocks,
    load_s17_encoder,
)
from research.stage3d_qwen_signed_phase.recompute_all import (
    BLOCK_SIZE,
    CELL_ORDER,
    Q2_LOGICAL_REF_IDX,
    recompute_all,
    score_vectors,
    softmax_temperature,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE3D_DIR = Path(__file__).resolve().parent


def calibrate_signed_reference_table(
    encoder,
    cell_idx: int,
    abs_omega: float,
    h: int,
    ell: int,
    t_p: float,
    snr_db: float = 2.0,
    sigma_phi: float = 0.10,
    blocks_per_batch: int = 128,
    n_batches: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calibrate quarter-turn pooled 256-code reference table on independent namespace 960501 (65,536 mothers/sign)."""
    counts = np.zeros((2, 4, 4, 4), dtype=np.int64)
    mass = np.zeros((2, 256), dtype=np.float64)
    omegas = (-float(abs_omega), float(abs_omega))
    for chunk_i in range(n_batches):
        b_map, _ = generate_paired_cfo_blocks(
            namespace=NS_REF_CALIB,
            cell_idx=cell_idx * 100 + chunk_i,
            blocks=blocks_per_batch,
            h=h,
            ell=ell,
            t_p=t_p,
            omegas=omegas,
            snr_db=snr_db,
            sigma_phi=sigma_phi,
            split_prefix="ref_cal",
        )
        for s_i, w in enumerate(omegas):
            batch = b_map[w]
            z1 = np.asarray(encoder.encode(batch.y1), dtype=np.int64)
            z2 = np.asarray(encoder.encode(batch.y2), dtype=np.int64)
            codes = 16 * z1 + z2
            targets = batch.k3_repeat
            np.add.at(counts[s_i], (z1 % 4, z2 % 4, (targets - z1 // 4) % 4), 1)
            mass[s_i] += np.bincount(codes, minlength=256)
    full = np.zeros((2, 256, 4), dtype=np.float64)
    for k in range(256):
        z1, z2 = divmod(k, 16)
        full[:, k, :] = np.roll(counts[:, z1 % 4, z2 % 4] + 0.5, z1 // 4, axis=-1)
    ref_probs = full / full.sum(axis=-1, keepdims=True)
    mass_probs = mass / mass.sum(axis=-1, keepdims=True)
    return ref_probs, mass_probs


def compute_blind_cell_index_and_analytic(
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


def compute_dev_pilot_sample_size(
    pilot_raw_ll: np.ndarray,
    temps_q3: np.ndarray,
    repo_root: Path = REPO_ROOT,
        var2 = float(cond.h) * (cond.sigma_phi ** 2)
        var3 = float(cond.h + cond.ell) * (cond.sigma_phi ** 2)
        varp = float(cond.t_p) * (cond.sigma_phi ** 2)
        mean2 = cond.omega * float(cond.h)
        mean3 = cond.omega * float(cond.h + cond.ell)
        meanp = cond.omega * float(cond.t_p)

        # Q1: phase rotation weights of a2 shifted by phi_2
        w_quad2 = wrapped_sector_quad_probs(mean2, var2)  # mass in quadrant k for phi_2
        # Pr(u2_apparent = v) -> true u2 distribution rotated back: sum_k w_quad2[k] * a2[:, (u2+k)%4]
        a2_unshifted = sum(w_quad2[k] * np.roll(a2, -k, axis=1) for k in range(4))
        q1 = _normalize_rows(diff_posterior(a1, a2_unshifted))

        # Q2: continuous wrapped-Gaussian pilot prior at t_p
        if t_p_sum is None:
            q2 = np.full((n_obs, 4), 0.25, dtype=np.float64)
        else:
            phi_g, w_p = wrapped_gaussian_weights(meanp, varp, grid=grid)
            # For exact zero-var deterministic offset (omega*t_p)
            if varp <= 0.0:
                dir_p = ROOTS * np.exp(1j * meanp)
                lw = 2.0 * snr * np.real(t_p_sum[:, None] * np.conj(dir_p)[None, :])
                lw -= lw.max(axis=1, keepdims=True)
                bp = np.exp(lw)
                bp /= np.sort(bp, axis=1).sum(axis=1, keepdims=True)
            else:
                lp = _pilot_c4_likelihood(t_p_sum, snr, phi_g)  # (M, G, 4)
                bp = np.einsum("g,mgk->mk", w_p, lp)
                bp /= np.sort(bp, axis=1).sum(axis=1, keepdims=True)
            q2 = _normalize_rows(diff_posterior(bp, a1))

        # Q3 pre-reveal
        if protocol == "repeat_x1":
            phi_g, w_3 = wrapped_gaussian_weights(mean3, var3, grid=grid)
            if var3 <= 0.0:
                theta_u = 0.5 * math.pi * np.arange(4, dtype=np.float64) + mean3
                mc_u = cont_sector_probs(theta_u, snr)  # (4, 4)
                q3 = a1 @ mc_u
                mu3 = (a1 @ ROOTS) * np.exp(1j * mean3)
            else:
                theta_grid = 0.5 * math.pi * np.arange(4, dtype=np.float64)[:, None] + phi_g[None, :]
                mc_grid = cont_sector_probs(theta_grid, snr)  # (4, G, 4)
                q3 = np.einsum("mu,ugk,g->mk", a1, mc_grid, w_3)
                mu3 = (a1 @ ROOTS) * complex(w_3 @ np.exp(1j * phi_g))
        elif protocol == "independent_x3":
            q3 = np.full((n_obs, 4), 0.25, dtype=np.float64)
            mu3 = np.zeros(n_obs, dtype=np.complex128)
        else:
            raise ValueError(protocol)
        q3_after = _normalize_rows(q3.copy())
        # Note: if pilot reveals phase phi_p and t_p == t_3 (e.g. C6 with finite drift or general),
        # for table readout (which only observes pilot sum) under continuous drift t_p and t_3 correlated,
        # compute exact conditional posterior p(phi_3 | P) if t_p > 0 and sigma_phi > 0!
        if protocol == "repeat_x1" and t_p_sum is not None and cond.sigma_phi > 0.0 and cond.t_p > 0.0:
            t_share = min(float(cond.t_p), float(cond.h + cond.ell))
            phi_p, w_p = wrapped_gaussian_weights(meanp, varp, grid=grid)
            lp = _pilot_c4_likelihood(t_p_sum, snr, phi_p)  # (M, G, 4)
            lik_p_marginal = 0.25 * lp.sum(axis=2)  # (M, G)
            post_phip = _normalize_rows(lik_p_marginal * w_p[None, :])
            if abs(float(cond.h + cond.ell) - float(cond.t_p)) < 1e-12:
                post_phi3 = post_phip
            elif float(cond.h + cond.ell) > float(cond.t_p):
                dt = float(cond.h + cond.ell) - float(cond.t_p)
                trans = _gaussian_transition_matrix(phi_p, phi_p, cond.omega * dt, dt * cond.sigma_phi ** 2)
                post_phi3 = post_phip @ trans.T
            else:
                # Brownian bridge from phi_p at t_p back to t_3 < t_p
                alpha = float(cond.h + cond.ell) / float(cond.t_p)
                var_br = alpha * (1.0 - alpha) * varp
                # Note E[phi_3 | phi_p] = alpha * phi_p
                disp = ((phi_p[:, None] - alpha * phi_p[None, :] + math.pi) % (2.0 * math.pi)) - math.pi
                sd_br = math.sqrt(max(var_br, 1e-12))
                n_img = max(1, math.ceil((10.0 * sd_br + math.pi) / (2.0 * math.pi)))
                trans = sum(np.exp(-0.5 * ((disp + 2.0 * math.pi * m) / sd_br) ** 2) for m in range(-n_img, n_img + 1))
                trans /= trans.sum(axis=0, keepdims=True)
                post_phi3 = post_phip @ trans.T
            theta_grid = 0.5 * math.pi * np.arange(4, dtype=np.float64)[:, None] + phi_p[None, :]
            mc_grid = cont_sector_probs(theta_grid, snr)
            q3_after = _normalize_rows(np.einsum("mu,ugk,mg->mk", a1, mc_grid, post_phi3))

    return {
        "q1": _normalize_rows(q1),
        "q2": _normalize_rows(q2),
        "q3": _normalize_rows(q3),
        "q3_after_pilot": _normalize_rows(q3_after),
        "mu3": mu3,
    }


def solve_full_iq_known_param(
    y1: np.ndarray,
    y2: np.ndarray,
    references: Sequence[Reference],
    cond: ConditionSpec,
    protocol: str = "repeat_x1",
    grid: int = QUAD_NODES_DEFAULT,
    allow_double_count_unsafe: bool = False,
) -> dict:
    """Analytic conditional Bayesian optimal baseline using UNQUANTIZED full complex I/Q and KNOWN cond params."""
    snr = cond.snr
    a1 = iq_likelihood(y1, snr)
    a2 = iq_likelihood(y2, snr)
    if cond.drift_type in ("coherent", "discrete"):
        # For coherent & discrete C4, a1 and a2 are exact sufficient statistics of (Y1, Y2)!
        return readout_from_table(
            a1,
            a2,
            references,
            cond,
            protocol=protocol,
            grid=grid,
```

### Event #21: `item.completed` — `command_execution` (`item_17`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-ChildItem -Force $env:LOCALAPPDATA/uv/cache | Select-Object Name; Get-ChildItem -Force C:/qazwsx | Select-Object Name; Get-ChildItem -Force C:/114514 | Select-Object Name; Get-ChildItem -Force $env:LOCALAPPDATA/uv/cache/archive-v0 | Select-Object -First 12 Name; Get-ChildItem -Force $env:LOCALAPPDATA/uv/cache/environments-v2 | Select-Object -First 12 Name'
```

**Aggregated Output:**

```text

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
conda-meta                                   
condabin                                     
DLLs                                         
envs                                         
etc                                          
include                                      
Lib                                          
Library                                      
libs                                         
Menu                                         
pkgs                                         
Scripts                                      
share                                        
shell                                        
Tools                                        
.condarc                                     
api-ms-win-core-console-l1-1-0.dll           
api-ms-win-core-console-l1-2-0.dll           
api-ms-win-core-datetime-l1-1-0.dll          
api-ms-win-core-debug-l1-1-0.dll             
api-ms-win-core-errorhandling-l1-1-0.dll     
api-ms-win-core-fibers-l1-1-0.dll            
api-ms-win-core-file-l1-1-0.dll              
api-ms-win-core-file-l1-2-0.dll              
api-ms-win-core-file-l2-1-0.dll              
api-ms-win-core-handle-l1-1-0.dll            
api-ms-win-core-heap-l1-1-0.dll              
api-ms-win-core-interlocked-l1-1-0.dll       
api-ms-win-core-libraryloader-l1-1-0.dll     
api-ms-win-core-localization-l1-2-0.dll      
api-ms-win-core-memory-l1-1-0.dll            
api-ms-win-core-namedpipe-l1-1-0.dll         
api-ms-win-core-processenvironment-l1-1-0.dll
api-ms-win-core-processthreads-l1-1-0.dll    
api-ms-win-core-processthreads-l1-1-1.dll    
api-ms-win-core-profile-l1-1-0.dll           
api-ms-win-core-rtlsupport-l1-1-0.dll        
api-ms-win-core-string-l1-1-0.dll            
api-ms-win-core-synch-l1-1-0.dll             
api-ms-win-core-synch-l1-2-0.dll             
api-ms-win-core-sysinfo-l1-1-0.dll           
api-ms-win-core-timezone-l1-1-0.dll          
api-ms-win-core-util-l1-1-0.dll              
api-ms-win-crt-conio-l1-1-0.dll              
api-ms-win-crt-convert-l1-1-0.dll            
api-ms-win-crt-environment-l1-1-0.dll        
api-ms-win-crt-filesystem-l1-1-0.dll         
api-ms-win-crt-heap-l1-1-0.dll               
api-ms-win-crt-locale-l1-1-0.dll             
api-ms-win-crt-math-l1-1-0.dll               
api-ms-win-crt-multibyte-l1-1-0.dll          
api-ms-win-crt-private-l1-1-0.dll            
api-ms-win-crt-process-l1-1-0.dll            
api-ms-win-crt-runtime-l1-1-0.dll            
api-ms-win-crt-stdio-l1-1-0.dll              
api-ms-win-crt-string-l1-1-0.dll             
api-ms-win-crt-time-l1-1-0.dll               
api-ms-win-crt-utility-l1-1-0.dll            
concrt140.dll                                
cwp.py                                       
LICENSE_PYTHON.txt                           
msvcp140.dll                                 
msvcp140_1.dll                               
msvcp140_2.dll                               
msvcp140_atomic_wait.dll                     
msvcp140_codecvt_ids.dll                     
pre_uninstall.bat                            
pylupdate5.bat                               
pyrcc5.bat                                   
python.exe                                   
python.pdb                                   
python3.dll                                  
python312.dll                                
python312.pdb                                
pythonw.exe                                  
pythonw.pdb                                  
pyuic5.bat                                   
ucrtbase.dll                                 
Uninstall-Anaconda3.exe                      
vcamp140.dll                                 
vccorlib140.dll                              
vcomp140.dll                                 
vcruntime140.dll                             
vcruntime140_1.dll                           
vcruntime140_threads.dll                     
xlwings32-0.32.1.dll                         
xlwings64-0.32.1.dll                         
zlib.dll                                     
_conda.exe                                   
conda-meta                                   
condabin                                     
DLLs                                         
envs                                         
etc                                          
include                                      
Lib                                          
Library                                      
libs                                         
Menu                                         
pkgs                                         
Scripts                                      
share                                        
shell                                        
tcl                                          
Tools                                        
.nonadmin                                    
api-ms-win-core-console-l1-1-0.dll           
api-ms-win-core-console-l1-2-0.dll           
api-ms-win-core-datetime-l1-1-0.dll          
api-ms-win-core-debug-l1-1-0.dll             
api-ms-win-core-errorhandling-l1-1-0.dll     
api-ms-win-core-fibers-l1-1-0.dll            
api-ms-win-core-file-l1-1-0.dll              
api-ms-win-core-file-l1-2-0.dll              
api-ms-win-core-file-l2-1-0.dll              
api-ms-win-core-handle-l1-1-0.dll            
api-ms-win-core-heap-l1-1-0.dll              
api-ms-win-core-interlocked-l1-1-0.dll       
api-ms-win-core-libraryloader-l1-1-0.dll     
api-ms-win-core-localization-l1-2-0.dll      
api-ms-win-core-memory-l1-1-0.dll            
api-ms-win-core-namedpipe-l1-1-0.dll         
api-ms-win-core-processenvironment-l1-1-0.dll
api-ms-win-core-processthreads-l1-1-0.dll    
api-ms-win-core-processthreads-l1-1-1.dll    
api-ms-win-core-profile-l1-1-0.dll           
api-ms-win-core-rtlsupport-l1-1-0.dll        
api-ms-win-core-string-l1-1-0.dll            
api-ms-win-core-synch-l1-1-0.dll             
api-ms-win-core-synch-l1-2-0.dll             
api-ms-win-core-sysinfo-l1-1-0.dll           
api-ms-win-core-timezone-l1-1-0.dll          
api-ms-win-core-util-l1-1-0.dll              
api-ms-win-crt-conio-l1-1-0.dll              
api-ms-win-crt-convert-l1-1-0.dll            
api-ms-win-crt-environment-l1-1-0.dll        
api-ms-win-crt-filesystem-l1-1-0.dll         
api-ms-win-crt-heap-l1-1-0.dll               
api-ms-win-crt-locale-l1-1-0.dll             
api-ms-win-crt-math-l1-1-0.dll               
api-ms-win-crt-multibyte-l1-1-0.dll          
api-ms-win-crt-private-l1-1-0.dll            
api-ms-win-crt-process-l1-1-0.dll            
api-ms-win-crt-runtime-l1-1-0.dll            
api-ms-win-crt-stdio-l1-1-0.dll              
api-ms-win-crt-string-l1-1-0.dll             
api-ms-win-crt-time-l1-1-0.dll               
api-ms-win-crt-utility-l1-1-0.dll            
concrt140.dll                                
cwp.py                                       
LICENSE_PYTHON.txt                           
msvcp140.dll                                 
msvcp140_1.dll                               
msvcp140_2.dll                               
msvcp140_atomic_wait.dll                     
msvcp140_codecvt_ids.dll                     
pre_uninstall.bat                            
python.exe                                   
python.pdb                                   
python3.dll                                  
python39.dll                                 
python39.pdb                                 
pythonw.exe                                  
pythonw.pdb                                  
ucrtbase.dll                                 
Uninstall-Miniconda3.exe                     
vccorlib140.dll                              
vcomp140.dll                                 
vcruntime140.dll                             
vcruntime140_1.dll                           
vcruntime140_threads.dll                     
venvlauncher.exe                             
venvwlauncher.exe                            
zlib.dll                                     
_conda.exe                                   
-3Eg0_w5zGyPhAVD                             
-INXTsup9SBW2nnw                             
-yFVWlDN8f_PSvOm                             
0kEkF80-N13tUw4M                             
0xUSRPAHLRdjp9oW                             
12oJCSOQV4GMwgHu                             
1SmvAy6YA_w5SdKx                             
1uFPYbapY1T1Qpz9                             
1v_fiJekykmZ15YB                             
1W0Yzc8ULbKjuVDZ                             
2H0sRTjw_eCmh8sk                             
2mCZH8WvQjPMoK1v                             
1093dab72d22fcbe                             


```

### Event #22: `item.completed` — `command_execution` (`item_20`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$s=Get-Content research/stage3d_qwen_signed_phase/results/summary_master.json -Raw | ConvertFrom-Json; $s.cells.PSObject.Properties | ForEach-Object { $_.Name; $_.Value.analytic_same_mothers | ConvertTo-Json -Depth 3 -Compress }; $p=Get-Content research/stage3d_qwen_signed_phase/data_stage3d.py; $p[208..259]; $p=Get-Content research/stage3_ministral_cache/data.py; $p[0..99]'
```

**Aggregated Output:**

```text
G1_main_extrap_both
{"direct_s17_table":{"acc":0.6461712786259542,"nll":0.941125824839512,"brier":0.5098317013509959,"ece":0.0009122218241481161},"full_iq_known_param":{"acc":0.6464098282442748,"nll":0.9290379865758834,"brier":0.5032444510266945,"ece":0.0046012731581605345}}
G2_interp_both
{"direct_s17_table":{"acc":0.4677734375,"nll":1.0222300150191364,"brier":0.5962492861161929,"ece":0.005695706517643107},"full_iq_known_param":{"acc":0.4677734375,"nll":1.0193455737210775,"brier":0.5949481802702071,"ece":0.01074199100752489}}
G3_extrap_angle
{"direct_s17_table":{"acc":0.62109375,"nll":0.9795675052936387,"brier":0.5312420968465116,"ece":0.02410018087291977},"full_iq_known_param":{"acc":0.619140625,"nll":0.953915663195823,"brier":0.5166361748255862,"ece":0.03584787000730208}}
G4_extrap_freq
{"direct_s17_table":{"acc":0.4775390625,"nll":1.0158511271590178,"brier":0.5936743155765972,"ece":0.015298223750495409},"full_iq_known_param":{"acc":0.4921875,"nll":1.0037380488119603,"brier":0.5896128349446631,"ece":0.02572273653326804}}
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

"""Deterministic mother-waveform dataset generator and cross-split orbit collision audit."""
from dataclasses import dataclass
import itertools
import math
from pathlib import Path
import struct
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from research.stage3_ministral_cache.schema import (
    CONDITION_SPECS,
    ConditionSpec,
    N_SAMPLES,
    ROOTS,
    Reference,
    matched_sum,
    sector_of_complex,
)


SPLIT_SEEDS: Dict[str, Tuple[int, int]] = {
    "s3_train": (840101, 16),  # 1,024 mothers
    "s3_dev": (840202, 4),     # 256 mothers
    "s3_test": (840303, 16),   # 1,024 mothers
}

PARENT_PARTITIONS: Dict[str, Tuple[int, int]] = {
    "train": (730241, 64),
    "dev": (410061, 4),
    "test": (518339, 16),
}


@dataclass
class MotherBatch:
    split: str
    condition: ConditionSpec
    mother_ids: List[str]
    x1: np.ndarray            # (M,) int64 in 0..3
    x2: np.ndarray            # (M,) int64 in 0..3
    g: np.ndarray             # (M,) int64 in 0..3
    x3_indep: np.ndarray      # (M,) int64 in 0..3
    y1: np.ndarray            # (M, 3) complex128
    y2: np.ndarray            # (M, 3) complex128
    y3_repeat: np.ndarray     # (M, 3) complex128
    y3_indep: np.ndarray      # (M, 3) complex128
    p_valid1: np.ndarray      # (M, 3) complex128
    p_valid2: np.ndarray      # (M, 3) complex128 (fresh resample of valid pilot)
    p_indep: np.ndarray       # (M, 3) complex128 (unrelated channel pilot)
    phi_2: np.ndarray         # (M,) float64 continuous shift or discrete*pi/2
    phi_3: np.ndarray         # (M,) float64
    phi_p: np.ndarray         # (M,) float64

    @property
    def k3_repeat(self) -> np.ndarray:
        return sector_of_complex(matched_sum(self.y3_repeat))

    @property
    def k3_indep(self) -> np.ndarray:
        return sector_of_complex(matched_sum(self.y3_indep))

    @property
    def t3_repeat(self) -> np.ndarray:
        return matched_sum(self.y3_repeat)

    @property
    def t3_indep(self) -> np.ndarray:
        return matched_sum(self.y3_indep)

    def rotated_view(self, s1: int, s2: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.y1 * ROOTS[s1 % 4], self.y2 * ROOTS[s2 % 4]

    def references_for_arm(self, arm: str, coord_rot: int = 0) -> Tuple[Reference, ...]:
        r = ROOTS[coord_rot % 4]
        tp = self.condition.t_p
        snr_db = self.condition.snr_db
        if arm == "none":
            return ()
        if arm == "valid":
            return (Reference(0, True, self.p_valid1 * r, tp, snr_db),)
        if arm == "independent":
            return (Reference(0, False, self.p_indep * r, tp, snr_db),)
        if arm == "duplicate_copy":
            ref0 = Reference(0, True, self.p_valid1 * r, tp, snr_db)
            return (ref0, ref0)
        if arm == "fresh_resample":
            return (
                Reference(0, True, self.p_valid1 * r, tp, snr_db),
                Reference(1, True, self.p_valid2 * r, tp, snr_db),
            )
        raise ValueError(f"Unknown arm={arm}")


def _sample_discrete_jumps(rng: np.random.Generator, d: float, steps: int, size: int) -> np.ndarray:
    if steps <= 0 or d == 0.0:
        return np.zeros(size, dtype=np.int64)
    # Use cumulative mod-4 Markov chain step-by-step
    acc = np.zeros(size, dtype=np.int64)
```

### Event #23: `item.completed` — `command_execution` (`item_19`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$d=Join-Path $env:LOCALAPPDATA '"'uv/cache'; Get-ChildItem -Force -LiteralPath (Join-Path "'$d '"'environments-v2'); Get-ChildItem -Directory -LiteralPath (Join-Path "'$d '"'archive-v0') | ForEach-Object { Get-ChildItem -Directory -LiteralPath "'$_.FullName | Where-Object {$_.Name -match '"'"'^(numpy-2.3.5|scipy-1.17.0|torch-.*'"\\.dist-info)"'$'"'} | Select-Object FullName }; Get-ChildItem -LiteralPath 'C:/qazwsx/envs' | Select-Object Name"
```

**Aggregated Output:**

```text


    Ŀ¼: C:\Users\������\AppData\Local\uv\cache\environments-v2


Mode                 LastWriteTime         Length Name                                                                 
----                 -------------         ------ ----                                                                 
d-----        2026/11/26     19:11                1093dab72d22fcbe                                                     

FullName : C:\Users\������\AppData\Local\uv\cache\archive-v0\l-bbeSyNZh-_fB5p\torch-2.12.1.dist-info


Name : my_project_env


Name : .conda_envs_dir_test



```

### Event #24: `item.completed` — `command_execution` (`item_21`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$p=Get-Content research/stage3d_qwen_signed_phase/data_stage3d.py; $p[192..208]; rg -n '"'training|cfo_train_blocks|train_blocks' research/stage3d_qwen_signed_phase/PROTOCOL_LOCK.json research/stage3d_qwen_signed_phase/train_eval_stage3d.py | Select-Object -First 32; git ls-files -- research/stage3d_qwen_signed_phase/results/blind_q3_scores.npz"
```

**Aggregated Output:**

```text
def format_public_metadata(cond: ConditionSpec) -> str:
    """Zero-leakage public metadata string (excludes condition names, split names, labels, and noise)."""
    if cond.drift_type == "coherent":
        drift_desc = "相干静止(omega=+0.00rad/�?sigma=0.00rad/√步,d=0.00/�?"
    elif cond.drift_type == "discrete":
        drift_desc = f"离散C4游走(omega=+0.00rad/�?sigma=0.00rad/√步,d={cond.d:.2f}/�?"
    else:
        drift_desc = f"连续频偏与布朗漂�?omega={cond.omega:+.2f}rad/�?sigma={cond.sigma_phi:.2f}rad/√步,d=0.00/�?"
    return (
        f"信噪�?{cond.snr_db:.1f}dB; "
        f"两窗间隔h={cond.h}, 留出间隔ell={cond.ell}, 导频时刻tp={cond.t_p:.1f}; "
        f"信道={drift_desc}。\n"
        f"已编码波形与公开相位向量�?W1><W2><PHASE>�?
    )


def build_stage3d_prompt(
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:1:"""GPU training, dev calibration, smoke verification, and blind evaluation routines for Stage 3D."""
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:182:    train_blocks: int = 8,
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:191:        _append_batch_groups(tr_groups, generate_mothers("s3_train", cname, max_blocks=train_blocks), encoder, zero_arm, "replay")
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:197:            NS_CFO_TRAIN, t_idx, train_blocks, h, ell, tp, TRAIN_OMEGAS, split_prefix="cfo_train"
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:357:    # 5. Microbatch throughput & VRAM sweep on training + inference
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:543:    train_blocks: int = 8,
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:568:        train_blocks=train_blocks,
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:630:                assert torch.isfinite(loss), f"Non-finite training loss in {arm}/s{seed} ep={ep}"
research/stage3d_qwen_signed_phase/train_eval_stage3d.py:778:    assert host_sha_after == host.host_sha256, "Host mutated during training!"
research/stage3d_qwen_signed_phase/results/blind_q3_scores.npz
```

### Event #25: `item.completed` — `command_execution` (`item_22`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$d=Join-Path $env:LOCALAPPDATA '"'uv/cache/environments-v2/1093dab72d22fcbe'; Get-ChildItem -Force -LiteralPath "'$d | Select-Object Name; Get-ChildItem -Force -LiteralPath $d -Depth 3 -Filter pyvenv.cfg | ForEach-Object { $_.FullName; Get-Content -LiteralPath $_.FullName }; Get-ChildItem -Directory -LiteralPath (Join-Path $env:LOCALAPPDATA '"'uv/cache/archive-v0') | ForEach-Object {Get-ChildItem -Directory -LiteralPath "'$_.FullName | Where-Object {$_.Name -match '"'"'^(numpy-|scipy-|torch-)'"'} | Select-Object FullName}; Get-ChildItem -Name -Force 'C:/qazwsx/envs/my_project_env/Lib/site-packages' | Where-Object {"'$_ -match '"'"'^(numpy-|scipy-|torch-)'"'}"
```

**Aggregated Output:**

```text

Name            
----            
27fef6b82521218f
44669b92cf8db157
76580b3e21535888
7829de497d478757
7d6c70a4e0ec676a
8ab8c010e9f0c738
a4a0d4059abfef7e
fff2ded241c488c1
                
                
                
                


```

### Event #26: `item.completed` — `command_execution` (`item_23`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$d=Join-Path $env:LOCALAPPDATA '"'uv/cache/environments-v2/1093dab72d22fcbe'; Get-ChildItem -LiteralPath "'$d | ForEach-Object { $_.FullName; Get-Item -LiteralPath $_.FullName | Format-List LinkType,Target,Attributes; Get-ChildItem -LiteralPath $_.FullName -Name | Select-Object -First 8 }; $p=Join-Path $env:LOCALAPPDATA '"'uv/cache/archive-v0'; Get-ChildItem -Directory -LiteralPath "'$p | ForEach-Object {Get-ChildItem -Directory -LiteralPath $_.FullName | Where-Object {$_.Name -match '"'"'^(numpy-|scipy-|torch-)'"'} | ForEach-Object {"'$_.FullName}}'
```

**Aggregated Output:**

```text
C:\Users\������\AppData\Local\uv\cache\environments-v2\1093dab72d22fcbe\27fef6b82521218f


LinkType   : 
Target     : {}
Attributes : Archive



27fef6b82521218f
C:\Users\������\AppData\Local\uv\cache\environments-v2\1093dab72d22fcbe\44669b92cf8db157


LinkType   : 
Target     : {}
Attributes : Archive



44669b92cf8db157
C:\Users\������\AppData\Local\uv\cache\environments-v2\1093dab72d22fcbe\76580b3e21535888


LinkType   : 
Target     : {}
Attributes : Archive



76580b3e21535888
C:\Users\������\AppData\Local\uv\cache\environments-v2\1093dab72d22fcbe\7829de497d478757


LinkType   : 
Target     : {}
Attributes : Archive



7829de497d478757
C:\Users\������\AppData\Local\uv\cache\environments-v2\1093dab72d22fcbe\7d6c70a4e0ec676a


LinkType   : 
Target     : {}
Attributes : Archive



7d6c70a4e0ec676a
C:\Users\������\AppData\Local\uv\cache\environments-v2\1093dab72d22fcbe\8ab8c010e9f0c738


LinkType   : 
Target     : {}
Attributes : Archive



8ab8c010e9f0c738
C:\Users\������\AppData\Local\uv\cache\environments-v2\1093dab72d22fcbe\a4a0d4059abfef7e


LinkType   : 
Target     : {}
Attributes : Archive



a4a0d4059abfef7e
C:\Users\������\AppData\Local\uv\cache\environments-v2\1093dab72d22fcbe\fff2ded241c488c1


LinkType   : 
Target     : {}
Attributes : Archive



fff2ded241c488c1
C:\Users\������\AppData\Local\uv\cache\archive-v0\3Oxde2-Jkv6KGjnQ\scipy-1.17.0.dist-info
C:\Users\������\AppData\Local\uv\cache\archive-v0\cSnmXl2iWBpNAXGE\numpy-2.3.5.dist-info
C:\Users\������\AppData\Local\uv\cache\archive-v0\i6_zPlBrmc1oGgj1\numpy-2.5.5.dist-info
C:\Users\������\AppData\Local\uv\cache\archive-v0\l-bbeSyNZh-_fB5p\torch-2.12.1.dist-info
```

### Event #27: `item.completed` — `command_execution` (`item_24`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$d=Join-Path $env:LOCALAPPDATA '"'uv/cache'; Get-ChildItem -LiteralPath (Join-Path "'$d '"'wheels-v6') -Name; Get-ChildItem -LiteralPath (Join-Path "'$d '"'simple-v21') -Name; Get-ChildItem -LiteralPath (Join-Path "'$d '"'archive-v0') -Directory | ForEach-Object {Get-ChildItem -LiteralPath "'$_.FullName -Directory -Filter '"'*.dist-info' | ForEach-Object { '{0} {1}' -f "'$_.Parent.Name,$_.Name }}; Get-ChildItem -Path C:/qazwsx/envs/my_project_env -Name; Get-ChildItem -LiteralPath (Join-Path $d '"'sdists-v9') -Force -Name"
```

**Aggregated Output:**

```text
pypi
pypi
-3Eg0_w5zGyPhAVD filelock-4.3.1.dist-info
-INXTsup9SBW2nnw jinja2-3.1.6.dist-info
-yFVWlDN8f_PSvOm jsonschema_specifications-2025.9.1.dist-info
0kEkF80-N13tUw4M pytest-9.1.1.dist-info
0xUSRPAHLRdjp9oW huggingface_hub-2.4.0.dist-info
12oJCSOQV4GMwgHu typing_inspection-0.4.5.dist-info
1SmvAy6YA_w5SdKx idna-3.20.dist-info
1uFPYbapY1T1Qpz9 fsspec-2026.9.0.dist-info
1v_fiJekykmZ15YB idna-3.19.dist-info
1W0Yzc8ULbKjuVDZ pydantic_core-2.50.4.dist-info
2H0sRTjw_eCmh8sk httpcore-1.0.9.dist-info
2rcofYqL1tRIBjqM referencing-0.37.0.dist-info
2UiMWEna5g3a7mnm setuptools-84.2.1.dist-info
3Oxde2-Jkv6KGjnQ scipy-1.17.0.dist-info
3wUvENWnx4snPG-6 azure_core-1.41.0.dist-info
3yh5YxBGAZ7EKNMV pyyaml-6.0.4.dist-info
3ytU4TS7z5iR9B1o pydantic_settings-2.15.0.dist-info
56XI-j9XrVnWM9Dh cryptography-50.0.1.dist-info
5PfgZIw8oWGQeUQg yarl-1.25.1.dist-info
5Vd1elZEW_X1q0fV anyio-4.15.1.dist-info
6JCnrcOQFUae-bks markupsafe-3.0.3.dist-info
6mSdmXuSld5vI1Zh packaging-26.3.dist-info
6W7Mxvd2I5oM9NZo iniconfig-2.3.0.dist-info
7PFJPmC9Xat-I635 psycopg-3.3.6.dist-info
7q3YKZs83n85kOJ3 markdown_it_py-4.2.0.dist-info
90ShW96ru-ZvA4Ka bcrypt-5.0.0.dist-info
9gsQ0T9CaGaWpM0o starlette-1.6.0.dist-info
9SjBS1AZnkF2JDj1 aiohappyeyeballs-2.7.1.dist-info
9vEhfDzLzB9UT1-J psycopg_binary-3.3.6.dist-info
9YQY_nZFQYHnFXtW uvicorn-0.52.4.dist-info
A4L3DGaVK4XPPR2B transformers-5.24.1.dist-info
abSAHPKYBKEBNWfr propcache-0.5.4.dist-info
aCAhhzepGZw7egib click-8.5.0.dist-info
Bj8VGUhuUrsbwYQg pycparser-3.0.dist-info
cSnmXl2iWBpNAXGE numpy-2.3.5.dist-info
cwrpe1AjGHyI2_gb typing_extensions-4.16.0.dist-info
CYTaZpIZS3OHdVlU azure_ai_projects-2.7.0.dist-info
dFVBWYGKpHl4VsNc markdown_it_py-4.3.0.dist-info
DpcKqdpPiYvgTQio typer-0.27.3.dist-info
DWPzT2jdRxPBzMKX pluggy-1.7.0.dist-info
d_6f1tYs0n5Kp6dw web_exploration_swarm-0.1.0.dist-info
e8qPhHGsqrG2VBkO frozenlist-1.8.0.dist-info
EjlrxW1vt2n4E_Zy tokenizers-0.23.3.dist-info
EkMxwaEyyu94iEYV httpcore2-2.13.0.dist-info
Ex519kRx8LyaaWSY rpds_py-2026.6.3.dist-info
EY3pQ0Mp0YWIj69t charset_normalizer-3.5.1.dist-info
EzOaACCIhQc0olV4 mdurl-0.1.2.dist-info
fB6Tq2NbRLrUMgYD uvicorn-0.53.0.dist-info
fGr8TdEDo4rpslOk psutil-7.2.2.dist-info
FqjKYQeN-jLkmoBm packaging-26.4.dist-info
fqY_x9ViVdktj-A4 sympy-1.14.0.dist-info
GG3u21Fd2tY-A36F rich-15.0.0.dist-info
gxNQq7Xirzx575jt attrs-26.1.0.dist-info
gZwqqxonl5VRPmzk pydantic-2.13.5.dist-info
HCFJl0zAFSrhStFU azure_storage_blob-12.30.2.dist-info
HMNS93CZ9VsRjoIb sniffio-1.3.1.dist-info
HNdcIQB23kIfe6Jl hf_xet-1.8.0.dist-info
i6_zPlBrmc1oGgj1 numpy-2.5.5.dist-info
iEan70G4iv4DQIs- email_validator-2.3.0.dist-info
ifyyhSqg0eUOlgvO truststore-0.10.4.dist-info
iG0KkwrJTDYZxqk0 sqlalchemy-2.0.54.dist-info
Ikd8ZAUHZlpz3Dxm comtypes-1.4.16.dist-info
iS4n0tqGZl9y1iiB aiohttp-3.14.5.dist-info
IYz-uAhyOg9BpQX6 colorama-0.4.6.dist-info
Jb5rO3w7XZ5ESE_S multidict-7.2.1.dist-info
JfqcIgHK1RzyRhoJ jiter-0.17.0.dist-info
JhAj3hHYPhbZN7dN pluggy-1.6.0.dist-info
jjwcAyQOycIQ1KMR iniconfig-2.3.1.dist-info
k4asR5j2Ej4BfAdg mpmath-1.3.0.dist-info
KGHJxYgFFfAAv80r requests-2.34.2.dist-info
kmKGQ6Otd9MXCL96 websockets-17.1.dist-info
KSA8IuYDEQgfAsPM python_dotenv-1.2.3.dist-info
kSCnL09rXM1W6rWS azure_identity-1.25.3.dist-info
l-bbeSyNZh-_fB5p torch-2.12.1.dist-info
LEAfNjBc3f0fogoN urllib3-2.8.0.dist-info
LeBFdodBQkFZbYuu setuptools-81.0.0.dist-info
Lm--KtUkgcfHhyLH aiosignal-1.4.0.dist-info
lT-9ccG8dj9G9OAh shellingham-1.5.4.dist-info
mdRltH95YC52pGL6 pyyaml-6.0.3.dist-info
MgHTeOX_zz2sKLjK msal-1.39.0.dist-info
Miseba8O4VwVj0nC pytest_asyncio-1.4.1.post1.dist-info
MlCgSeRRTJJDqWTJ greenlet-3.5.5.dist-info
NKW9Ac6dKG8vvuC4 dnspython-2.8.0.dist-info
noC29VL6ydprAsAz mako-1.4.1.dist-info
NyMM_xBP26ecldgk isodate-0.7.2.dist-info
OZOPdRF8iiOpU5NI certifi-2026.10.23.dist-info
PG1NQZGN3le2b3iP jsonschema-4.26.0.dist-info
PVYMSEf2wvdIKpnF alembic-1.20.0.dist-info
QbHpeUpwUlMIHxD8 certifi-2026.7.22.dist-info
qk-XGijnztOkvV7k setuptools-84.0.0.dist-info
qKe0tCdNaO_ikACt sqlalchemy-2.0.52.dist-info
qlCCwK6KdJ5lIrkx typing_inspection-0.4.4.dist-info
QnF-cSwg3ZyLDdcm annotated_types-0.8.0.dist-info
r5xmeIXRYfPvXmRL pygments-2.21.0.dist-info
RASdK5DVvQdT2S82 pyjwt-2.14.0.dist-info
RmfBwt5ZunAnJm8D watchfiles-1.2.0.dist-info
R_AaBz_KNzIYcH31 fastapi-0.141.1.dist-info
sa6WrcM_noqJGYRj pydantic-2.14.3.dist-info
SBMxmgz2ENWBMRkC pytest-9.2.0.dist-info
sE9z_LoBecf-wi5q click-8.5.1.dist-info
slWYru8ngjFUELlr pycryptodome-3.23.0.dist-info
SQluAP6D7kM3zAKB pydantic_core-2.46.5.dist-info
SU_uXNtSmT_kQTgH regex-2026.11.25.dist-info
tnLjP6lODONriCsL h11-0.16.0.dist-info
TUBOw2KgHhj5hV52 greenlet-3.5.6.dist-info
uAdUx8vmB5pnn0Pq httpx-0.28.1.dist-info
V9W27WqcxfPs08zZ continuity-0.1.0.dist-info
vgfZWT2mRI4WaCIq ruff-0.16.8.dist-info
Vqk4-stqIOqEeUEo networkx-3.7.dist-info
W9tY5vyAPoqejRFG cffi-2.0.0.dist-info
wd1VbUJDzJ5LkUe1 markupsafe-3.0.4.dist-info
wgZ10WoAZHIWl_8s anyio-4.16.0.dist-info
WPdV-PrJSzWbieEN anyio-4.14.2.dist-info
xH5rm0sTWorkXOsr httpx2-2.13.0.dist-info
xnNee54dvcH18Acp tqdm-4.70.1.dist-info
xrWuBev81OODqN_Y msal_extensions-1.3.1.dist-info
XvsPanRaidTE4lgC openai-3.16.2.dist-info
xwcs_6WgtsaNDhA6 tzdata-2026.4.dist-info
yeAZvqeknOh8PqsR uiautomation-2.0.29.dist-info
YRfIhXjd6GdsDL2j httptools-0.8.0.dist-info
Y_N-k_SSHV5WTuQn annotated_doc-0.0.5.dist-info
zboF2DshbeUiR457 safetensors-0.8.0.dist-info
_ftBthdCeU-egQZk typer-0.29.0.dist-info
conda-meta
DLLs
etc
include
Lib
Library
libs
Scripts
share
Tools
api-ms-win-core-console-l1-1-0.dll
api-ms-win-core-console-l1-2-0.dll
api-ms-win-core-datetime-l1-1-0.dll
api-ms-win-core-debug-l1-1-0.dll
api-ms-win-core-errorhandling-l1-1-0.dll
api-ms-win-core-fibers-l1-1-0.dll
api-ms-win-core-file-l1-1-0.dll
api-ms-win-core-file-l1-2-0.dll
api-ms-win-core-file-l2-1-0.dll
api-ms-win-core-handle-l1-1-0.dll
api-ms-win-core-heap-l1-1-0.dll
api-ms-win-core-interlocked-l1-1-0.dll
api-ms-win-core-libraryloader-l1-1-0.dll
api-ms-win-core-localization-l1-2-0.dll
api-ms-win-core-memory-l1-1-0.dll
api-ms-win-core-namedpipe-l1-1-0.dll
api-ms-win-core-processenvironment-l1-1-0.dll
api-ms-win-core-processthreads-l1-1-0.dll
api-ms-win-core-processthreads-l1-1-1.dll
api-ms-win-core-profile-l1-1-0.dll
api-ms-win-core-rtlsupport-l1-1-0.dll
api-ms-win-core-string-l1-1-0.dll
api-ms-win-core-synch-l1-1-0.dll
api-ms-win-core-synch-l1-2-0.dll
api-ms-win-core-sysinfo-l1-1-0.dll
api-ms-win-core-timezone-l1-1-0.dll
api-ms-win-core-util-l1-1-0.dll
api-ms-win-crt-conio-l1-1-0.dll
api-ms-win-crt-convert-l1-1-0.dll
api-ms-win-crt-environment-l1-1-0.dll
api-ms-win-crt-filesystem-l1-1-0.dll
api-ms-win-crt-heap-l1-1-0.dll
api-ms-win-crt-locale-l1-1-0.dll
api-ms-win-crt-math-l1-1-0.dll
api-ms-win-crt-multibyte-l1-1-0.dll
api-ms-win-crt-private-l1-1-0.dll
api-ms-win-crt-process-l1-1-0.dll
api-ms-win-crt-runtime-l1-1-0.dll
api-ms-win-crt-stdio-l1-1-0.dll
api-ms-win-crt-string-l1-1-0.dll
api-ms-win-crt-time-l1-1-0.dll
api-ms-win-crt-utility-l1-1-0.dll
concrt140.dll
LICENSE_PYTHON.txt
msvcp140.dll
msvcp140_1.dll
msvcp140_2.dll
msvcp140_atomic_wait.dll
msvcp140_codecvt_ids.dll
python.exe
python.pdb
python3.dll
python311.dll
python311.pdb
pythonw.exe
pythonw.pdb
ucrtbase.dll
vcamp140.dll
vccorlib140.dll
vcruntime140.dll
vcruntime140_1.dll
vcruntime140_threads.dll
zlib.dll
editable
.git
.gitignore
```

### Event #28: `item.completed` — `agent_message` (`item_25`)

The source confirms exact quarter-turn equivariance for the S17 codes, with the pilot coordinate held fixed in the quotient tests. Separate token positions leave the phase interaction to the host, but frozen attention can still compute interactions. The saved score bundle is available, so I can recompute the projection metrics and measure the physical and host spectra directly.

### Event #29: `item.completed` — `command_execution` (`item_26`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'where.exe uv; where.exe python; Get-ChildItem -Name C:/ | Where-Object {$_ -match '"'python|uv'}; Get-ChildItem -Path "'$env:LOCALAPPDATA/Programs/Python -Name -ErrorAction SilentlyContinue; Get-ChildItem -Path $env:APPDATA/uv -Name -ErrorAction SilentlyContinue; Get-ChildItem -Path (Join-Path $env:LOCALAPPDATA '"'uv/cache/interpreter-v4') -Recurse -Name | Select-Object -First 20"
```

**Aggregated Output:**

```text
C:\qazwsx\Scripts\uv.exe
C:\qazwsx\python.exe
C:\114514\python.exe
C:\376381\miniconda3\python.exe
C:\Users\������\AppData\Local\Microsoft\WindowsApps\python.exe
Python313
tools
08cc7f641f8f98d9
0de9bff94b993286
2f8a8d68d05c453f
64dcec958efe0fa4
e9e5012c70b2fce1
08cc7f641f8f98d9\4f3e0e6caa48264e.msgpack
0de9bff94b993286\00ea6f00532cc82d.msgpack
0de9bff94b993286\4f3e0e6caa48264e.msgpack
0de9bff94b993286\5dddf0f874944d71.msgpack
0de9bff94b993286\a2658311c8299a64.msgpack
0de9bff94b993286\fefca76762d8a91a.msgpack
2f8a8d68d05c453f\4f3e0e6caa48264e.msgpack
64dcec958efe0fa4\4f3e0e6caa48264e.msgpack
64dcec958efe0fa4\cc864c9b62c92ed8.msgpack
e9e5012c70b2fce1\4f3e0e6caa48264e.msgpack
e9e5012c70b2fce1\4fa82a53748a7cd5.msgpack
e9e5012c70b2fce1\624cae97aa1c8ac3.msgpack
e9e5012c70b2fce1\6367655b63c438be.msgpack
e9e5012c70b2fce1\74c787f825b5a79f.msgpack
e9e5012c70b2fce1\9f43e16fed0a30ae.msgpack
```

### Event #30: `item.completed` — `command_execution` (`item_27`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$d=Join-Path $env:LOCALAPPDATA '"'uv/cache'; "'$p=Join-Path $d '"'wheels-v6/pypi'; Get-ChildItem -LiteralPath "'$p -Name | Where-Object {$_ -match '"'"'^(numpy|scipy|torch|networkx|sympy|mpmath|setuptools|filelock|fsspec|jinja2|markupsafe|typing-extensions)$'"'}; Get-ChildItem -LiteralPath (Join-Path "'$p '"'torch') -Name; Get-ChildItem -LiteralPath (Join-Path "'$p '"'numpy') -Name; Get-ChildItem -LiteralPath (Join-Path "'$d '"'simple-v21/pypi') -Name | Select-Object -First 12; Get-Content -LiteralPath (Join-Path "'$d '"'archive-v0/l-bbeSyNZh-_fB5p/torch-2.12.1.dist-info/METADATA') | Select-String '"'^Requires-Dist'"'"
```

**Aggregated Output:**

```text
filelock
fsspec
jinja2
markupsafe
mpmath
networkx
numpy
scipy
setuptools
sympy
torch
typing-extensions
2.12.1-cp312-cp312-win_amd64
2.12.1-cp312-cp312-win_amd64.http
2.12.1-cp312-cp312-win_amd64.msgpack
torch-2.12.1-cp312-cp312-win_amd64.lock
2.3.5-cp312-cp312-win_amd64
2.3.5-cp312-cp312-win_amd64.http
2.3.5-cp312-cp312-win_amd64.msgpack
2.5.5-cp312-cp312-win_amd64
2.5.5-cp312-cp312-win_amd64.http
2.5.5-cp312-cp312-win_amd64.msgpack
numpy-2.3.5-cp312-cp312-win_amd64.lock
numpy-2.5.5-cp312-cp312-win_amd64.lock
aiohappyeyeballs.lock
aiohappyeyeballs.rkyv
aiohttp.lock
aiohttp.rkyv
aiosignal.lock
aiosignal.rkyv
alembic.lock
alembic.rkyv
annotated-doc.lock
annotated-doc.rkyv
annotated-types.lock
annotated-types.rkyv

Requires-Dist: filelock
Requires-Dist: typing-extensions>=4.10.0
Requires-Dist: setuptools<82
Requires-Dist: sympy>=1.13.3
Requires-Dist: networkx>=2.5.1
Requires-Dist: jinja2
Requires-Dist: fsspec>=0.8.5
Requires-Dist: cuda-toolkit[cudart,cufft,cufile,cupti,curand,cusolver,cusparse,nvjitlink,nvrtc,nvtx]==13.0.2; platform_
system == "Linux"
Requires-Dist: nvidia-cublas<=13.1.1.3,>=13.1.0.3; platform_system == "Linux"
Requires-Dist: cuda-bindings<14,>=13.0.3; platform_system == "Linux"
Requires-Dist: nvidia-cudnn-cu13==9.20.0.48; platform_system == "Linux"
Requires-Dist: nvidia-cusparselt-cu13==0.8.1; platform_system == "Linux"
Requires-Dist: nvidia-nccl-cu13==2.29.7; platform_system == "Linux"
Requires-Dist: nvidia-nvshmem-cu13==3.4.5; platform_system == "Linux"
Requires-Dist: triton==3.7.1; platform_system == "Linux"
Requires-Dist: opt-einsum>=3.3; extra == "opt-einsum"
Requires-Dist: optree>=0.13.0; extra == "optree"
Requires-Dist: pyyaml; extra == "pyyaml"


```

### Event #31: `item.completed` — `web_search` (`exec-db1c42aa-05cd-4708-9e6f-190d8ff2df9d`)

```json
{
  "id": "exec-db1c42aa-05cd-4708-9e6f-190d8ff2df9d",
  "type": "web_search",
  "query": "Cohen Welling Group Equivariant Convolutional Networks ICML 2016 PMLR group equivariance representation ...",
  "action": {
    "type": "search",
    "queries": [
      "Cohen Welling Group Equivariant Convolutional Networks ICML 2016 PMLR group equivariance representation",
      "Vaswani attention is all you need scaled dot product attention q k arxiv 2017"
    ]
  },
  "results": [
    {
      "type": "text_result",
      "domain": "proceedings.mlr.press",
      "ref_id": "turn0search0",
      "snippet": "`@InProceedings{pmlr-v48-cohenc16, title = {Group Equivariant Convolutional Networks}, author = {Cohen, Taco and Welling, Max}, booktitle = {Proceedings of The 33rd International Conference on Machine",
      "title": "Group Equivariant Convolutional Networks",
      "url": "https://proceedings.mlr.press/v48/cohenc16.html"
    },
    {
      "type": "text_result",
      "domain": "arxiv.org",
      "ref_id": "turn0academia12",
      "snippet": "Title: Attention Is All You Need Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. ... On the WMT 2014 English-to-French",
      "title": "Attention Is All You Need",
      "url": "https://arxiv.org/abs/1706.03762"
    },
    {
      "type": "text_result",
      "domain": "proceedings.neurips.cc",
      "ref_id": "turn0search13",
      "snippet": "Left, “Scaled Dot-Product Attention”: Q and K enter MatMul; the result passes through Scale, Mask (opt.), SoftMax, and a second MatMul, with V entering that",
      "title": "Attention is All you Need",
      "url": "https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf?src_trk=em66da689f075e00.591058311350402508"
    },
    {
      "type": "text_result",
      "domain": "s10251.pcdn.co",
      "ref_id": "turn0search14",
      "snippet": "Scaled Dot-Product Attention ... Dot-product attention is identical to our algorithm, except for the scaling factor ... To counteract this effect, we scale the dot",
      "title": "Attention Is All You Need",
      "url": "https://s10251.pcdn.co/pdf/2017-vaswani-transformer.pdf"
    },
    {
      "type": "text_result",
      "domain": "proceedings.mlr.press",
      "ref_id": "turn0search15",
      "snippet": "Cohen ... Max Welling ... We introduce Group equivariant Convolutional Neural Networks (G-CNNs), a natural generalization of convolutional neural networks that reduces sample complexity by",
      "title": "Group Equivariant Convolutional Networks",
      "url": "https://proceedings.mlr.press/v48/cohenc16.pdf"
    },
    {
      "type": "text_result",
      "domain": "tamnd.github.io",
      "ref_id": "turn0search1",
      "snippet": "Image: Figure 2(left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel. ... Dot-product attention is identical to our algorithm,",
      "title": "Attention Is All You Need in English",
      "url": "https://tamnd.github.io/papers-reader/p/vaswani-2017-attention/en/"
    },
    {
      "type": "text_result",
      "domain": "papers.neurips.cc",
      "ref_id": "turn0search16",
      "snippet": "Left diagram title: “Scaled Dot-Product Attention” ... - Flow: Q and K into MatMul → Scale → Mask (opt.) ... Dot-product attention is identical to",
      "title": "Attention is All you Need",
      "url": "https://papers.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf"
    },
    {
      "type": "text_result",
      "domain": "proceedings.mlr.press",
      "ref_id": "turn0search17",
      "snippet": "Welling, Group Equivariant Convolutional Networks. Proceedings of the International Conference on Machine Learning (ICML), 2016”. ... Equivariance Derivations ... The ordinary planar correlation \\(\\st",
      "title": "Group Equivariant Convolutional Networks",
      "url": "https://proceedings.mlr.press/v48/cohenc16-supp.pdf"
    },
    {
      "type": "text_result",
      "domain": "proceedings.neurips.cc",
      "ref_id": "turn0search18",
      "snippet": "## Scaled Dot-Product Attention and Multi-Head Attention ... Dot-product attention is identical to our algorithm, except for the scaling factor of \\(\\frac{1}{\\sqrt{d_k}}\\). ... Instead of",
      "title": "Attention is All you Need",
      "url": "https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf?amp%3Butm_campaign=news&amp%3Butm_medium=nationaltribune"
    },
    {
      "type": "text_result",
      "domain": "www.gabormelli.com",
      "ref_id": "turn0search2",
      "snippet": "We call our particular attention “Scaled Dot-Product Attention” (Figure 2). ... [math]\\displaystyle{ Attention(Q,K,V)=\\mathrm{softmax}\\left(\\dfrac{QK_T}{\\sqrt{d_k}}\\right)V }[/math] | (1) ... Dot-prod",
      "title": "2017 AttentionisallYouNeed - GM-RKB",
      "url": "https://www.gabormelli.com/RKB/Vaswani_et_al._2017"
    },
    {
      "type": "text_result",
      "domain": "zeo1122.github.io",
      "ref_id": "turn0search3",
      "snippet": "# Attention Is All You Need ... Image: Figure 1 from Vaswani et al. showing the Transformer model architecture Figure 1. ... ### Scaled Dot-Product",
      "title": "Attention Is All You Need - Jeo Jeon",
      "url": "https://zeo1122.github.io/notes/vaswani-2017-transformer/"
    },
    {
      "type": "text_result",
      "domain": "arxiv.org",
      "ref_id": "turn0academia19",
      "snippet": "Cohen, Max Welling Date: Wed Feb 24 16:17:15 2016 We introduce Group equivariant Convolutional Neural Networks (G-CNNs), a natural generalization of convolutional neural networks that",
      "title": "Group Equivariant Convolutional Networks",
      "url": "https://arxiv.org/abs/1602.07576"
    },
    {
      "type": "text_result",
      "domain": "bibtex.github.io",
      "ref_id": "turn0search4",
      "snippet": "## Taco Cohen, Max Welling Group Equivariant Convolutional Networks ... @inproceedings{ICML-2016-CohenW,",
      "title": "BibSLEIGH — Group Equivariant Convolutional Networks",
      "url": "https://bibtex.github.io/ICML-2016-CohenW.html"
    },
    {
      "type": "text_result",
      "domain": "www.june.kim",
      "ref_id": "turn0search5",
      "snippet": "# Attention Is All You Need Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin · 2017",
      "title": "Vaswani et al. 2017 — Attention Is All You Need",
      "url": "https://www.june.kim/reading/cognitive-architecture/vaswani-2017/"
    },
    {
      "type": "text_result",
      "domain": "dblp.dagstuhl.de",
      "ref_id": "turn0search6",
      "snippet": "# \"Group Equivariant Convolutional Networks.\" Taco Cohen, Max Welling (2016) ... ICML 2016: 2990-2999",
      "title": "dblp: Group Equivariant Convolutional Networks.",
      "url": "https://dblp.dagstuhl.de/rec/conf/icml/CohenW16.html"
    },
    {
      "type": "text_result",
      "domain": "www.gabormelli.com",
      "ref_id": "turn0search7",
      "snippet": "We call our particular attention “Scaled Dot-Product Attention” (Figure 2). ... [math]\\displaystyle{ Attention(Q,K,V)=\\mathrm{softmax}\\left(\\dfrac{QK_T}{\\sqrt{d_k}}\\right)V }[/math] | (1) ... Dot-prod",
      "title": "2017 AttentionisallYouNeed - GM-RKB",
      "url": "https://www.gabormelli.com/RKB/Attention_Is_All_You_Need"
    },
    {
      "type": "text_result",
      "domain": "mlanthology.org",
      "ref_id": "turn0search8",
      "snippet": "`Cohen and Welling. \"Group Equivariant Convolutional Networks.\" ... International Conference on Machine Learning, 2016.](https://mlanthology.org/icml/2016/cohen2016icml-group/)`",
      "title": "Group Equivariant Convolutional Networks",
      "url": "https://mlanthology.org/icml/2016/cohen2016icml-group/"
    },
    {
      "type": "text_result",
      "domain": "utensil.github.io",
      "ref_id": "turn0search9",
      "snippet": "Scaled dot-product attention [vaswani2017attention, sec. ... For \\(Q\\in \\mathbb R^{T_q\\times d_h}\\), \\(K\\in \\mathbb R^{T_k\\times d_h}\\), and \\(V\\in \\mathbb R^{T_k\\times d_v}\\), scaled dot-product atte",
      "title": "Decoder-only Transformer architecture",
      "url": "https://utensil.github.io/forest/ftip-000K/"
    },
    {
      "type": "text_result",
      "domain": "sitemap.tomrochette.com",
      "ref_id": "turn0search10",
      "snippet": "# Ashish Vaswani - Attention Is All You Need (2017) ... #### 3.2.1 Scaled Dot-Product Attention ... * We compute the dot products of the",
      "title": "Ashish Vaswani - Attention Is All You Need (2017) · tomrochette.com",
      "url": "https://sitemap.tomrochette.com/machine-learning/papers/ashish-vaswani-attention-is-all-you-need/"
    },
    {
      "type": "text_result",
      "domain": "rumblr.com",
      "ref_id": "turn0search11",
      "snippet": "Figure 2 (left) of the paper, redrawn: scaled dot-product attention. ... Every word scored against every word, hence the n² cost. | n × n",
      "title": "Attention Is All You Need, Annotated",
      "url": "https://rumblr.com/papers/primer/attention-is-all-you-need"
    },
    {
      "type": "text_result",
      "domain": "www.youtube.com",
      "ref_id": "turn0youtube22",
      "snippet": "Last lecture the model had no choices: q = k = v = x. ... Query, key, value: what am I looking for, what do",
      "title": "Q, K, V intuition - self attention with trainable weights | The attention family series - Lecture 6 - YouTube",
      "url": "https://www.youtube.com/watch?v=wshXBhGItkM"
    },
    {
      "type": "text_result",
      "domain": "www.youtube.com",
      "ref_id": "turn0youtube23",
      "snippet": "# Attention Is All You Need ... Published: 2017-11-28, 813532 views, 17000 likes ... On the WMT 2014 English-to-French translation task, our model establishes a",
      "title": "Attention Is All You Need - YouTube",
      "url": "https://www.youtube.com/watch?v=iDulhoQ2pro"
    }
  ]
}
```

### Event #32: `item.completed` — `command_execution` (`item_28`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$src=Join-Path $env:LOCALAPPDATA '"'uv/cache'; "'$dst=Join-Path (Get-Location) '"'research/stage4_math_and_design/M1/.uv-cache'; "'$packages=@('"'numpy','scipy','torch','networkx','sympy','mpmath','setuptools','filelock','fsspec','jinja2','markupsafe','typing-extensions'); "'$archive=@('"'3Oxde2-Jkv6KGjnQ','cSnmXl2iWBpNAXGE','l-bbeSyNZh-_fB5p','Vqk4-stqIOqEeUEo','fqY_x9ViVdktj-A4','k4asR5j2Ej4BfAdg','LeBFdodBQkFZbYuu','-3Eg0_w5zGyPhAVD','1uFPYbapY1T1Qpz9','-INXTsup9SBW2nnw','6JCnrcOQFUae-bks','wd1VbUJDzJ5LkUe1','cwrpe1AjGHyI2_gb'); foreach ("'$sub in @('"'archive-v0','wheels-v6/pypi','simple-v21/pypi')) { New-Item -ItemType Directory -Force -Path (Join-Path "'$dst $sub) | Out-Null }; foreach ($p in $packages) { Copy-Item -Recurse -LiteralPath (Join-Path $src "wheels-v6/pypi/$p") -Destination (Join-Path $dst '"'wheels-v6/pypi'); Copy-Item -LiteralPath (Join-Path "'$src "simple-v21/pypi/$p.rkyv") -Destination (Join-Path $dst '"'simple-v21/pypi') }; foreach ("'$id in $archive) { Copy-Item -Recurse -LiteralPath (Join-Path $src "archive-v0/$id") -Destination (Join-Path $dst '"'archive-v0') }; "'$env:UV_CACHE_DIR=$dst; $env:UV_OFFLINE='"'1'; "'$env:UV_LINK_MODE='"'hardlink'; "'$env:UV_NO_PROJECT='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B -c 'import numpy, scipy, torch, sys; print(sys.executable, numpy.__version__, scipy.__version__, torch.__version__)'"
```

**Aggregated Output:**

```text
error: Failed to install: torch-2.12.1-cp312-cp312-win_amd64.whl (torch==2.12.1)
  Caused by: 拒绝访问。 (os error 5) at path "C:\\WINDOWS\\TEMP\\.tmpXQpSPf"
```

### Event #33: `item.completed` — `file_change` (`item_30`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M1\\verify_m1.py",
    "kind": "add"
  }
]
```

### Event #34: `item.completed` — `command_execution` (`item_32`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$p=Get-Content research/stage3_ministral_cache/schema.py -Encoding utf8; $p[85..150]; Get-ChildItem research/stage4_math_and_design/M1 -Force | Select-Object Name; rg -n '"'cal_ref|work_g' research/stage3d_qwen_signed_phase/results/summary_master.json | Select-Object -First 8"
```

**Aggregated Output:**

```text
        t_p=6.0,
        drift_type="discrete",
        d=0.08,
        in_support_encoder=False,
        in_support_connector=True,
    ),
    "C3_mid_pilot_cont": ConditionSpec(
        name="C3_mid_pilot_cont",
        snr_db=2.0,
        h=4,
        ell=2,
        t_p=2.0,
        drift_type="continuous",
        sigma_phi=0.25,
        in_support_encoder=False,
        in_support_connector=True,
    ),
    "C4_cfo_ramp": ConditionSpec(
        name="C4_cfo_ramp",
        snr_db=2.0,
        h=2,
        ell=2,
        t_p=4.0,
        drift_type="cfo_continuous",
        sigma_phi=0.10,
        omega=0.35,
        in_support_encoder=False,
        in_support_connector=False,
    ),
    "C5_snr_0db": ConditionSpec(
        name="C5_snr_0db",
        snr_db=0.0,
        h=1,
        ell=2,
        t_p=3.0,
        drift_type="continuous",
        sigma_phi=0.20,
        in_support_encoder=False,
        in_support_connector=False,
    ),
    "C5_snr_6db": ConditionSpec(
        name="C5_snr_6db",
        snr_db=6.0,
        h=1,
        ell=2,
        t_p=3.0,
        drift_type="continuous",
        sigma_phi=0.20,
        in_support_encoder=False,
        in_support_connector=False,
    ),
    "C6_decoherent_order": ConditionSpec(
        name="C6_decoherent_order",
        snr_db=2.0,
        h=2,
        ell=2,
        t_p=4.0,
        drift_type="continuous",
        sigma_phi=3.0,
        in_support_encoder=False,
        in_support_connector=False,
        question_order="unseen_q3_first",
    ),
}



Name        
----        
.uv-cache   
.uv-tmp     
verify_m1.py
1053:    "cal_ref_overlap_mass": 0.9676971435546875,
1054:    "cal_ref_odd_norm_sq": 0.8070056852144443,
1435:    "physical_cal_reference": {
1486:      "cal_ref_neg": [
1492:      "cal_ref_pos": [
1623:    "cal_ref_overlap_mass": 0.9600372314453125,
1624:    "cal_ref_odd_norm_sq": 0.7719518825306924,
2005:    "physical_cal_reference": {


```

### Event #35: `item.completed` — `file_change` (`item_33`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M1\\verify_m1.py",
    "kind": "update"
  }
]
```

### Event #36: `item.completed` — `command_execution` (`item_34`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-Process uv,python -ErrorAction SilentlyContinue | Select-Object Id,CPU,StartTime,Path; Get-ChildItem -Force research/stage4_math_and_design/M1/.uv-tmp | Select-Object Name'
```

**Aggregated Output:**

```text

   Id       CPU StartTime           Path                                                                               
   --       --- ---------           ----                                                                               
 4048 15.609375 2026/11/26 19:22:21 C:\qazwsx\python.exe                                                               
 7020  0.015625 2026/11/26 19:22:21 C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage...
 7940   0.96875 2026/11/26 14:21:21 C:\qazwsx\python.exe                                                               
15784  0.546875 2026/11/26 14:21:21 C:\qazwsx\python.exe                                                               
22696  0.421875 2026/11/26 14:21:22 C:\qazwsx\python.exe                                                               
17436   30.8125 2026/11/26 19:21:55 C:\qazwsx\Scripts\uv.exe                                                           


```

### Event #37: `item.completed` — `command_execution` (`item_31`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$m1=Join-Path (Get-Location) '"'research/stage4_math_and_design/M1'; "'$env:UV_CACHE_DIR=Join-Path $m1 '"'.uv-cache'; "'$env:UV_OFFLINE='"'1'; "'$env:UV_LINK_MODE='"'hardlink'; "'$env:UV_NO_PROJECT='"'1'; "'$env:TEMP=Join-Path $m1 '"'.uv-tmp'; "'$env:TMP=$env:TEMP; New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B research/stage4_math_and_design/M1/verify_m1.py'
```

**Aggregated Output:**

```text
Installed 12 packages in 24.77s
Trained both CPU connectors for seed 202611
Trained both CPU connectors for seed 202622
Trained both CPU connectors for seed 202633
Traceback (most recent call last):
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M1\verify_m1.py", line 503, in <module>
    }
    ^^
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M1\verify_m1.py", line 452, in main
    
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M1\verify_m1.py", line 323, in stage3d_diagnostics
    def exact_row(z):
             ^^^^^^^^
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage3d_qwen_signed_phase\recompute_all.py", line 442, in work_f_sign_probe
    r["seed"] = int(seed_val)
                ^^^^^^^^^^^^^
TypeError: only length-1 arrays can be converted to Python scalars
```

### Event #38: `item.completed` — `command_execution` (`item_35`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$p=Get-Content research/stage3d_qwen_signed_phase/recompute_all.py; $p[326..338]; $p[881..915]; rg -n '"'def paired_t_ci' research/stage3d_qwen_signed_phase/recompute_all.py"
```

**Aggregated Output:**

```text
def work_f_sign_probe(
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
100:def paired_t_ci(block_diffs: np.ndarray) -> dict:
```

### Event #39: `item.completed` — `file_change` (`item_36`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M1\\verify_m1.py",
    "kind": "update"
  }
]
```

### Event #40: `item.completed` — `command_execution` (`item_37`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$m1=Join-Path (Get-Location) '"'research/stage4_math_and_design/M1'; "'$env:UV_CACHE_DIR=Join-Path $m1 '"'.uv-cache'; "'$env:UV_OFFLINE='"'1'; "'$env:UV_LINK_MODE='"'hardlink'; "'$env:UV_NO_PROJECT='"'1'; "'$env:TEMP=Join-Path $m1 '"'.uv-tmp'; "'$env:TMP=$env:TEMP; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B research/stage4_math_and_design/M1/verify_m1.py'
```

**Aggregated Output:**

```text
Trained both CPU connectors for seed 202611
Trained both CPU connectors for seed 202622
Trained both CPU connectors for seed 202633
G1_main_extrap_both A=62.40%, B=64.62% C=64.62%
G2_interp_both A=44.59%, B=46.78% C=46.78%
G3_extrap_angle A=59.06%, B=62.11% C=62.11%
G4_extrap_freq A=46.10%, B=47.75% C=47.75%
C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M1\results_m1.json
{"B_all_cell_defects_at_most_1e_6": true, "A_breaks_equivariance": true, "B_G1_G3_each_seed_above_55_pct": true, "B_all_cell_means_within_3pp_of_C": true, "A_G1_G3_mean_at_most_25_pct": false}
```

### Event #41: `item.completed` — `file_change` (`item_38`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M1\\THEOREM_M1.md",
    "kind": "add"
  }
]
```

### Event #42: `item.completed` — `agent_message` (`item_39`)

The CPU run finished. B matched the direct S17 accuracy on all four blind cells and had zero measured equivariance defect. A broke equivariance but scored 62.40% on G1 and 59.06% on G3, so the surrogate did not reproduce Qwen’s below-chance collapse. I’ll record both results and the limits they put on the structural claim.

### Event #43: `item.completed` — `command_execution` (`item_40`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$s=Get-Content -Raw research/stage4_math_and_design/M1/results_m1.json | ConvertFrom-Json; $s.runtime | ConvertTo-Json -Compress; $s.train.trainable_parameters | ConvertTo-Json -Compress; $s.cells.PSObject.Properties | ForEach-Object { $q=$_.Value; [pscustomobject]@{cell=$_.Name; A_nll=$q.models.A_AdditivePhaseBaseline.mean.nll; B_nll=$q.models.B_EquivariantPhaseCoupled.mean.nll; C_nll=$q.C_direct_s17.nll; A_defect=$q.models.A_AdditivePhaseBaseline.maximum_equivariance_defect; B_defect=$q.models.B_EquivariantPhaseCoupled.maximum_equivariance_defect; unrotated_acc=$q.unrotated_s17_MAP.acc; parity=$q.C_archive_max_abs_probability_difference; B_acc=$q.models.B_EquivariantPhaseCoupled.mean.acc} }; $spec=$s.work_f_overlap_weighted_signed_contrast_spectra; $spec.physical | ConvertTo-Json -Compress; $spec.physical_calibration_rank_certificate | ConvertTo-Json -Compress; $spec.host.phase | Select-Object seed_mean_table_contrast,orientation_gain,relative_frobenius_norm,relative_frobenius_error_to_physical | ConvertTo-Json -Depth 6 -Compress; $s.boundary_scan_fixed_variance_0p04 | ConvertTo-Json -Depth 4 -Compress; $s.pf_complete_orbit_counterexample | ConvertTo-Json -Depth 4 -Compress'
```

**Aggregated Output:**

```text
{"python":"3.12.7","numpy":"2.3.5","scipy":"1.17.0","torch":"2.12.1+cpu","device":"cpu"}
{"A_AdditivePhaseBaseline":6936,"B_EquivariantPhaseCoupled":6520}
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:2 �ַ�: 250
+ ... q=$_.Value; [pscustomobject]@{cell=$_.Name; A_nll=$q.models.A_Additiv ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:2 �ַ�: 250
+ ... q=$_.Value; [pscustomobject]@{cell=$_.Name; A_nll=$q.models.A_Additiv ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:2 �ַ�: 250
+ ... q=$_.Value; [pscustomobject]@{cell=$_.Name; A_nll=$q.models.A_Additiv ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
�޷���ֵת��Ϊ���͡�System.Management.Automation.LanguagePrimitives+InternalPSCustomObject����������ģʽ��֧�ֺ������͡�
����λ�� ��:2 �ַ�: 250
+ ... q=$_.Value; [pscustomobject]@{cell=$_.Name; A_nll=$q.models.A_Additiv ...
+                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidArgument: (:) []��RuntimeException
    + FullyQualifiedErrorId : ConversionSupportedOnlyToCoreTypes
 
{"gram_eigenvalues_descending":[0.4050346258308556,0.4019090499509318,6.2009432656674588E-05,4.2847742231648343E-33],"singular_values_descending":[0.6364233071084493,0.6339629720661387,0.007874606825529424,6.5458186830715334E-17],"rank_svd":3,"rank_threshold":3.6176476618746238E-14,"stable_rank":1.9924362850682635,"effective_rank_entropy_of_gram_energy":2.001488615297034,"frobenius_norm":0.8983349515712077,"best_rank1_relative_frobenius_residual":0.705763340572848}
{"rank":3,"code_index":0,"positive_weight_orbit_code_indices":[0,64,128,192],"frequency_1_real_and_imag":["-442738/144881269","210759050/144881269"],"frequency_2":"-2038912/144881269"}
{"seed_mean_table_contrast":{"gram_eigenvalues_descending":[0.0005764897303738236,0.00024338327274281586,0.00018039859969125804,7.230340519615827E-34],"singular_values_descending":[0.024010200548388254,0.015600745903411666,0.01343125458366634,2.6889292515080844E-17],"rank_svd":3,"rank_threshold":1.3648218867040358E-15,"stable_rank":1.7351074097352495,"effective_rank_entropy_of_gram_energy":2.639038789091652,"frobenius_norm":0.03162707072758869,"best_rank1_relative_frobenius_residual":0.6508969222086347},"orientation_gain":0.009819795034315315,"relative_frobenius_norm":0.03520632328985112,"relative_frobenius_error_to_physical":0.9907572331963864}
{"0.000000000000":{"center_source_kernel":[0.7798549203838425,0.10446110952590283,0.011222860564351787,0.10446110952590286],"expected_oracle_accuracy":0.6462637675301874,"expected_unrotated_MAP_accuracy":0.6462637675301874},"0.785398163397":{"center_source_kernel":[0.4796623164398741,0.47966231643987406,0.020337683560125885,0.020337683560125892],"expected_oracle_accuracy":0.4572037538366993,"expected_unrotated_MAP_accuracy":0.4329418940590925},"0.800000000000":{"center_source_kernel":[0.47049061220840915,0.4888157071871575,0.020799190426089312,0.019894490178343976],"expected_oracle_accuracy":0.4610655626944147,"expected_unrotated_MAP_accuracy":0.42640162261378656},"1.570796326795":{"center_source_kernel":[0.10446110952590287,0.7798549203838425,0.10446110952590285,0.011222860564351792],"expected_oracle_accuracy":0.6462637675301874,"expected_unrotated_MAP_accuracy":0.15987048870669165},"1.600000000000":{"center_source_kernel":[0.09751432162990256,0.7793739197343108,0.11187981774631278,0.011231940889473793],"expected_oracle_accuracy":0.6459458430971585,"expected_unrotated_MAP_accuracy":0.15432105180761824},"3.141592653590":{"center_source_kernel":[0.011222860564351785,0.10446110952590289,0.7798549203838425,0.10446110952590279],"expected_oracle_accuracy":0.6462637675301874,"expected_unrotated_MAP_accuracy":0.03663762932848194}}
{"raw":{"acc":0.75,"nll":1.9386591398710313,"brier":0.9199499999999998,"tie_count":0},"pf":{"acc":0.0,"nll":1.3280254529959152,"brier":0.8159624999999997,"tie_count":0},"pf_defect":5.5511151231257827E-17,"pf_idempotence_residual":1.1102230246251565E-16,"F0_defect":0.0,"aligned_projected_probabilities":[0.26499999999999996,0.4974999999999999,0.12249999999999998,0.11499999999999998]}
```

### Event #44: `item.completed` — `command_execution` (`item_41`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$s=Get-Content -Raw research/stage4_math_and_design/M1/results_m1.json | ConvertFrom-Json; foreach ($p in $s.cells.PSObject.Properties) { $q=$p.Value; $a=$q.models.A_AdditivePhaseBaseline; $b=$q.models.B_EquivariantPhaseCoupled; '"'{0} A {1:F6}/{2:F6} B {3:F6}/{4:F6} C {5:F6}/{6:F6} A_defect {7:F7} B_defect {8} unrotated {9:F6} parity {10:E4}' -f "'$p.Name,$a.mean.acc,$a.mean.nll,$b.mean.acc,$b.mean.nll,$q.C_direct_s17.acc,$q.C_direct_s17.nll,$a.maximum_equivariance_defect,$b.maximum_equivariance_defect,$q.unrotated_s17_MAP.acc,$q.C_archive_max_abs_probability_difference }; $s.work_g_recomputed.arms.phase.offline_group_avg_16way | ConvertTo-Json -Depth 4 -Compress; $s.all_task_quotient_lift_defects | ConvertTo-Json -Compress'
```

**Aggregated Output:**

```text
G1_main_extrap_both A 0.623980/1.043227 B 0.646171/0.941775 C 0.646171/0.941126 A_defect 0.7785088 B_defect 5.96046447753906E-08 unrotated 0.152493 parity 1.1102E-015
G2_interp_both A 0.445855/1.052723 B 0.467773/1.021871 C 0.467773/1.022230 A_defect 0.6180046 B_defect 2.98023223876953E-08 unrotated 0.417969 parity 8.3267E-016
G3_extrap_angle A 0.590603/1.118583 B 0.621094/0.979565 C 0.621094/0.979568 A_defect 0.7453497 B_defect 5.96046447753906E-08 unrotated 0.166016 parity 1.2212E-015
G4_extrap_freq A 0.461046/1.074825 B 0.477539/1.016728 C 0.477539/1.015851 A_defect 0.7018841 B_defect 5.96046447753906E-08 unrotated 0.430664 parity 7.7716E-016
{"acc":0.17050333969465647,"nll":1.4113123906537837,"pf_minus_raw_acc_t95":{"n_blocks":131,"mean":-0.0674830364715861,"block_sd":0.01981303079493022,"se":0.0017310725380280602,"t_crit_95":1.9783804054470222,"lo95":-0.07090775646122827,"hi95":-0.06405831648194393},"raw_minus_pf_nll_t95":{"n_blocks":131,"mean":0.0339161630052869,"block_sd":0.008374058906737819,"se":0.0007316449237535328,"t_crit_95":1.9783804054470222,"lo95":0.03246869102438813,"hi95":0.03536363498618567}}
{"q1":0.0,"q2":0.0,"q3_repeat":0.0,"q3_indep":0.0,"q3_after_pilot":0.0,"q2_joint_coordinate":0.0}
```

### Event #45: `item.completed` — `file_change` (`item_42`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M1\\verify_m1.py",
    "kind": "update"
  }
]
```

### Event #46: `item.completed` — `command_execution` (`item_43`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$m1=Join-Path (Get-Location) '"'research/stage4_math_and_design/M1'; "'$env:UV_CACHE_DIR=Join-Path $m1 '"'.uv-cache'; "'$env:UV_OFFLINE='"'1'; "'$env:UV_LINK_MODE='"'hardlink'; "'$env:UV_NO_PROJECT='"'1'; "'$env:TEMP=Join-Path $m1 '"'.uv-tmp'; "'$env:TMP=$env:TEMP; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B research/stage4_math_and_design/M1/verify_m1.py'
```

**Aggregated Output:**

```text
Trained both CPU connectors for seed 202611
Trained both CPU connectors for seed 202622
Trained both CPU connectors for seed 202633
G1_main_extrap_both A=62.40%, B=64.62% C=64.62%
G2_interp_both A=44.59%, B=46.78% C=46.78%
G3_extrap_angle A=59.06%, B=62.11% C=62.11%
G4_extrap_freq A=46.10%, B=47.75% C=47.75%
C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M1\results_m1.json
{"B_all_grouped_cell_defects_equal_zero": true, "B_all_cell_defects_at_most_1e_6": true, "B_all_ungrouped_cell_defects_at_most_1e_6": true, "A_breaks_equivariance": true, "B_G1_G3_each_seed_above_55_pct": true, "B_all_cell_means_within_3pp_of_C": true, "A_G1_G3_mean_at_most_25_pct": false}
```

### Event #47: `item.completed` — `file_change` (`item_44`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M1\\THEOREM_M1.md",
    "kind": "update"
  }
]
```

### Event #48: `item.completed` — `command_execution` (`item_45`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$s=Get-Content -Raw research/stage4_math_and_design/M1/results_m1.json | ConvertFrom-Json; foreach ($p in $s.cells.PSObject.Properties) { $q=$p.Value; $a=$q.models.A_AdditivePhaseBaseline; $b=$q.models.B_EquivariantPhaseCoupled; '"'{0} B grouped {1} ungrouped {2:E4} delta {3:E4} NLL {4:F8}; independent TV {5}' -f "'$p.Name,$b.maximum_equivariance_defect,$b.ungrouped_batch_maximum_equivariance_defect,$b.maximum_grouped_vs_ungrouped_probability_difference,$b.mean.nll,$q.q3_indep_zero_tv.max_total_variation_from_uniform }; git status --short --untracked-files=normal; Get-ChildItem -Force research/stage4_math_and_design/M1 | Select-Object Name,Length'
```

**Aggregated Output:**

```text
G1_main_extrap_both B grouped 0.0 ungrouped 5.9605E-008 delta 5.9605E-008 NLL 0.94177453; independent TV 0.0
G2_interp_both B grouped 0.0 ungrouped 2.9802E-008 delta 2.9802E-008 NLL 1.02187054; independent TV 0.0
G3_extrap_angle B grouped 0.0 ungrouped 5.9605E-008 delta 5.9605E-008 NLL 0.97956544; independent TV 0.0
G4_extrap_freq B grouped 0.0 ungrouped 5.9605E-008 delta 5.9605E-008 NLL 1.01672820; independent TV 0.0
?? research/stage4_math_and_design/

Name            Length
----            ------
.uv-cache             
.uv-tmp               
results_m1.json 59462 
THEOREM_M1.md   28564 
verify_m1.py    27710 


```

### Event #49: `item.completed` — `file_change` (`item_46`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M1\\THEOREM_M1.md",
    "kind": "update"
  }
]
```

### Event #50: `item.completed` — `command_execution` (`item_47`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$m1=(Resolve-Path -LiteralPath '"'research/stage4_math_and_design/M1').ProviderPath; foreach ("'$name in @('"'.uv-cache','.uv-tmp')) { "'$item=Get-Item -LiteralPath (Join-Path $m1 $name) -Force; if ($item.Parent.FullName -ne $m1 -or ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw '"'Unexpected cache path' }; "'$item.FullName; Get-ChildItem -Force -LiteralPath $item.FullName | Select-Object Name }; git status --short --untracked-files=normal -- research/stage4_math_and_design/M1; rg -n '"'"'^#'"' research/stage4_math_and_design/M1/THEOREM_M1.md"
```

**Aggregated Output:**

```text
C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M1\.uv-cache

?? research/stage4_math_and_design/M1/
1:# M1 Round 1 — Quotient equivariance and signed phase transport
7:## 1. Exact discrete symmetry
19:### Theorem 1 — Encoder and conditional choice actions
37:### Theorem 2 — Quotient factorization
49:## 2. Continuous phase transport and the sector dip
51:### Theorem 3 — Circular phase evolution before the sector map
64:### Theorem 4 — Circulant Q3 pushforward
94:### Corollary — Why \(\pi/4\) and \(\pi/2\) give different accuracies
116:## 3. Stage 3D diagnostics and the scope of separation
129:### Theorem 5 — Additive centered-logit obstruction
137:#### Work F spectrum
148:### Theorem 6 — Reynolds averaging and NLL
160:## 4. Stage 4 `EquivariantEvidenceConnector`
170:### M3 interface — `RefSufficientPool`
176:### M4 interface — `PhaseContext` for Pluto and CTTC
182:## 5. CPU verification and Round 2 decisions
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
C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M1\.uv-tmp
torchinductor_������


```

### Event #51: `item.completed` — `command_execution` (`item_48`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'uv cache clean --cache-dir research/stage4_math_and_design/M1/.uv-cache; Get-ChildItem -Force -Recurse -LiteralPath research/stage4_math_and_design/M1/.uv-tmp | Select-Object -First 12 FullName,PSIsContainer'
```

**Aggregated Output:**

```text
Clearing cache at: research/stage4_math_and_design/M1/.uv-cache
Removed 34670 files (1.2GiB)

FullName                                                                                                               
--------                                                                                                               
C:\Users\������\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M1\.uv-tmp\torchin...


```

### Event #52: `item.completed` — `command_execution` (`item_49`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '[Console]::OutputEncoding=[System.Text.Encoding]::UTF8; Get-ChildItem -Force -Recurse -LiteralPath research/stage4_math_and_design/M1/.uv-tmp | ForEach-Object { $_.Name }; Get-ChildItem -Force research/stage4_math_and_design/M1 | Select-Object Name,Length; git status --short --untracked-files=all -- research/stage4_math_and_design/M1; git diff --name-only; rg -n '"'reuse the matched-control|Reuse the matched-control' 'C:/Users/李则徐/.codex/memories/MEMORY.md'"
```

**Aggregated Output:**

```text
�޷��������ԡ�������ģʽ��֧�ֺ������͵��������á�
����λ�� ��:2 �ַ�: 1
+ [Console]::OutputEncoding=[System.Text.Encoding]::UTF8; Get-ChildItem ...
+ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : InvalidOperation: (:) []��RuntimeException
    + FullyQualifiedErrorId : PropertySetterNotSupportedInConstrainedLanguage
 
torchinductor_������

?? research/stage4_math_and_design/M1/THEOREM_M1.md
?? research/stage4_math_and_design/M1/results_m1.json
?? research/stage4_math_and_design/M1/verify_m1.py
32:applies_to: cwd=C:\Users\李则徐\Documents\ChatGPT\EM-FloatFormer; reuse_rule=Reuse the matched-control, reproducibility, and publication approach for this repository; numerical results, commit, hardware status, and receiver outcomes are P4-specific.
Name            Length
----            ------
.uv-tmp               
results_m1.json 59462 
THEOREM_M1.md   28756 
verify_m1.py    27710 


```

### Event #53: `item.completed` — `command_execution` (`item_50`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Remove-Item -LiteralPath 'research/stage4_math_and_design/M1/.uv-tmp/torchinductor_李则徐' -ErrorAction Stop"
```

### Event #54: `item.completed` — `command_execution` (`item_51`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Remove-Item -LiteralPath 'research/stage4_math_and_design/M1/.uv-tmp' -ErrorAction Stop; Get-ChildItem -Force research/stage4_math_and_design/M1 | Select-Object Name,Length; git status --short --untracked-files=all -- research/stage4_math_and_design/M1"
```

**Aggregated Output:**

```text

?? research/stage4_math_and_design/M1/THEOREM_M1.md
?? research/stage4_math_and_design/M1/results_m1.json
?? research/stage4_math_and_design/M1/verify_m1.py
Name            Length
----            ------
results_m1.json  59462
THEOREM_M1.md    28756
verify_m1.py     27710


```

### Event #55: `item.completed` — `agent_message` (`item_52`)

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



### Event #56: `turn.completed`

- Usage: `{"input_tokens": 4669111, "cached_input_tokens": 4448000, "cache_write_input_tokens": 0, "output_tokens": 41091, "reasoning_output_tokens": 6881}`


## Round 1 — Final Output (`reply_r1.md`)

```markdown
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

```


---

## Round 2 — Cross-Review & Convergence Prompt (`prompt_r2.md`)

```markdown
# Track M1 (Round 2): Cross-Track Convergence — Pure-Host vs Physical-Residual Ablation, 16-Bit/s17 Dual Cache & Real-Data Group Action

You are resuming persistent Codex session **Track M1** (`01a21948-8b7b-7ec0-bb98-65a487a5ce22`) in `C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4`. Keep edits STRICTLY inside `research/stage4_math_and_design/M1/` and run all Python scripts ONLY via `uv run --python 3.12 ...`.

## 1. Cross-Track Summary From Round 1 (M2, M3, M4, M5, M6)
1. **M2 (Stage 3D Forensics)**: On `blind_q3_scores.npz`, `G1` locked `NLL(zero) - NLL(phase) = +0.035650` splits into `+0.036543` scalar temperature mismatch and **`-0.001017` reverse-conditional KL**; after seedwise scalar recalibration, **`zero` beats `phase` across all 5 cells** (`G1: -0.000893`, `G2: -0.027716`, `G3: -0.000430`, `G4: -0.026351`, `H: -0.003695`), despite `0.3631 nats` signed CFO MI available in `s17`! M2's Stage 4 gate requires post-dev-calibration NLL/Acc/NIG ($\eta_{\min}=0.10$), ECE $\le 0.05$, equivariance defect $\le 10^{-5}$, and fixed-text slot interventions.
2. **M6 (DPI Loss Ladder & Host Jacobian)**: On `G1`, `Full IQ` (`0.9290`) $\to$ `direct_s17` (`0.9411`, `+0.0121`) $\to$ `Qwen phase unit-T` (`1.4192`, `+0.4780`) $\to$ `locked-T` (`1.4452`, `+0.0261`) shows **97.66% of excess NLL occurs after `s17`**. M6 derived the frozen-host candidate-score alignment matrix $M_{\text{cand}} = \Pi C_{\text{cand}} E_{\text{vocab}} D_{\text{answer}} B$ and asks how to test the true frozen host rather than merely bypassing it via a standalone head.
3. **M1 Self-Critique + M2/M6 Challenge**: In your Round 1 `B_EquivariantPhaseCoupled`, using the physical S17 prior + residual `delta = 0.15 * tanh(ell_host)` reached `64.62%` (parity with `direct_s17`), leaving unresolved whether frozen host attention learned anything beyond the physical prior! Also naive `A` reached `62.40%` on a tiny toy MLP, so we need a clean causal isolation between **quotient equivariance**, **multiplicative phase coupling**, **coupled tokens into frozen host (`NO direct logit bypass`)**, and **physical prior + bounded host residual**.
4. **M5 (Universal Stage 4→6 Cache)**: Proved `4-bit s17` merges amplitude/SNR (`|T|=0.02 vs 0.5833 -> 0.4804 TV`) and loses `0.3119 nats` on Stage 5 joint event+location (`1.2788 -> 0.9669`), AND exhibited two IQ pairs sharing exact `(T, E)` that produce different `s17` inner codes `[2, 0]`. M5 converged on a **16-bit bit-exact canonical extension**: `[2b turn | 2b s17_inner | 6b canon_angle | 4b log_radial | 2b resid_quality]` (recovering `s17` by right-shift while matching `Full IQ` closely in Stage 5–6).
5. **M3 & M4 (Interfaces)**:
   - M3 emits `canonical_q2[4]`, `canonical_q3[4]`, `joint[4,4]`, plus `a1` reference frame, validated `valid_mask`, and supports multi-clock Brownian likelihood + `t_p > t_3`.
   - M4 requires: (a) clean separation between **receive-gauge rotation** ($r_{\text{rx}}$: `Q1` TX diff invariant, `Q2` TX label invariant, `Q3` RX sector rolls $+r_{\text{rx}}$) and **TX relabeling**, (b) CFO uncertainty variance $v_\omega$ (giving edge damping $e^{-n^2 v_\omega d^2 / 2}$ correlated across edges!), and (c) SNR-preserving R3 lift on real Pluto/CTTC frames.

## 2. Specific Round-2 Convergence Tasks for Track M1
Update `research/stage4_math_and_design/M1/THEOREM_M1.md` (append/integrate a clear `## Round 2 Convergence`), `verify_m1.py`, and `results_m1.json` to resolve:
1. **The 4-Arm Causal Ablation Architecture (`Host-Pure` vs `Physical+HostResidual`)**:
   - Explicitly formalize inside `EquivariantEvidenceConnector`:
     - **`Arm 0` (Stage 3D Baseline)**: non-quotient separate `<W1><W2><PHASE>`, no bypass.
     - **`Arm 1` (`Equivariant-Host-Pure`)**: quotient canonicalization (`canonical_input` + `rotate`) + multiplicative phase-coupled continuous summary fed purely as **soft tokens into the frozen host**, with **NO physical logit bypass** (`q_host` computed strictly from frozen LM / frozen surrogate logits, with isotropic `W_out` geometry)!
     - **`Arm 2` (`Equivariant-Physical-Residual`)**: canonical physical prior + bounded trainable host residual (`delta = gamma * tanh(ell_host)`), plus fixed-text counterfactual slot-swap and shuffled-slot diagnostics isolating host contribution beyond the analytic bypass!
   - Implement and run **BOTH `Arm 1 (Host-Pure)` AND `Arm 2 (Physical+Residual)`** in `verify_m1.py` on the SAME frozen CPU surrogate host and identical `|\omega| <= 0.25` training split, testing `G1..G4`, equivariance defect (`<= 1e-6`), and M6's rank/margin candidate-Jacobian diagnostic! (Does `Arm 1 Host-Pure` ALSO extrapolate >55% past $\pi/2$ on `G1/G3`, or does frozen random projection struggle without coupled tokens / sufficient width?)
2. **Unified `4-bit s17` / `16-bit canonical` Dual Input Contract**:
   - Show cleanly how `EquivariantEvidenceConnector` consumes EITHER `4-bit s17` (`turn, s17_inner`) OR M5's `16-bit canonical packet` (`turn, s17_inner, canon_angle, log_radial, resid_quality`) through one unified canonical evidence embedding + transport layer (keeping total trainable parameters well under `<= 2.2M`, matching Stage 3D's Qwen budget).
3. **Locked Gauge & Uncertainty Contract with M3 & M4**:
   - Write down the exact permutation table distinguishing: (i) **synthetic/canonical per-window shift** `(r1, r2)`, (ii) **real hardware common receive-gauge turn** `r_rx` (affecting Q1/Q2/Q3_rx), and (iii) **joint TX+RX transform**, plus exact incorporation of M4's shared CFO latent `(omega_hat, v_omega)` (via Gaussian-Hermite over $\omega$ or analytical coupled variance $n^2(v_\phi d + v_\omega d^2)$ with shared edge correlation) and M3's `joint[4,4]` factor!
4. Run updated `verify_m1.py` via `uv run`, confirm exit 0 and updated `results_m1.json`, and summarize converged decisions in your final reply (`reply_r2.md`).
```


## Round 2 — Resumed Execution & Reasoning Trace (`round2_events.jsonl`)

### Event #1: `thread.started` (thread_id=`01a21948-8b7b-7ec0-bb98-65a487a5ce22`)

### Event #2: `turn.started`

### Event #3: `item.completed` — `agent_message` (`item_0`)

I’ll read the cross-track interface notes, then add the host-only and residual arms on the same frozen CPU host. I’ll keep the analytic readout as a separate control so the comparison can measure what the host contributes.

### Event #4: `item.completed` — `command_execution` (`item_2`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg --files --hidden research/stage4_math_and_design -g '"'!**/__pycache__/**'"' -g '"'!**/.uv-cache/**'"'; rg --files --hidden -g AGENTS.md -g '"'!**/.git/**'"'; rg -n '"'^(class|def|##)'"' research/stage4_math_and_design/M1/verify_m1.py research/stage4_math_and_design/M1/THEOREM_M1.md"
```

**Aggregated Output:**

```text
research/stage4_math_and_design\transcript_utils.py
research/stage4_math_and_design\M6\verify_m6.py
research/stage4_math_and_design\M6\THEOREM_M6.md
research/stage4_math_and_design\M6\results_m6.json
research/stage4_math_and_design\M6\reply_r1.md
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
research/stage4_math_and_design\M2\reply_r1.md
research/stage4_math_and_design\M2\results_m2.json
research/stage4_math_and_design\M2\verify_m2.py
research/stage4_math_and_design\M2\THEOREM_M2.md
research/stage4_math_and_design\M1\verify_m1.py
research/stage4_math_and_design\M1\THEOREM_M1.md
research/stage4_math_and_design\M1\results_m1.json
research/stage4_math_and_design\discussions\M6\round2_events.jsonl
research/stage4_math_and_design\discussions\M6\round1_events.jsonl
research/stage4_math_and_design\discussions\M6\reply_r1.md
research/stage4_math_and_design\discussions\M6\prompt_r2.md
research/stage4_math_and_design\discussions\M6\prompt_r1.md
research/stage4_math_and_design\discussions\M5\round2_events.jsonl
research/stage4_math_and_design\discussions\M5\round1_events.jsonl
research/stage4_math_and_design\discussions\M5\reply_r1.md
research/stage4_math_and_design\discussions\M5\prompt_r2.md
research/stage4_math_and_design\discussions\M5\prompt_r1.md
research/stage4_math_and_design\discussions\M2\round2_events.jsonl
research/stage4_math_and_design\discussions\M2\round1_events.jsonl
research/stage4_math_and_design\discussions\M2\reply_r1.md
research/stage4_math_and_design\discussions\M2\prompt_r2.md
research/stage4_math_and_design\discussions\M2\prompt_r1.md
research/stage4_math_and_design\discussions\M4\round2_events.jsonl
research/stage4_math_and_design\discussions\M4\round1_events.jsonl
research/stage4_math_and_design\discussions\M4\reply_r1.md
research/stage4_math_and_design\discussions\M4\prompt_r2.md
research/stage4_math_and_design\discussions\M4\prompt_r1.md
research/stage4_math_and_design\discussions\M3\prompt_r1.md
research/stage4_math_and_design\discussions\M3\prompt_r2.md
research/stage4_math_and_design\discussions\M3\round2_events.jsonl
research/stage4_math_and_design\discussions\M3\round1_events.jsonl
research/stage4_math_and_design\discussions\M3\reply_r1.md
research/stage4_math_and_design\discussions\M1\prompt_r2.md
research/stage4_math_and_design\discussions\M1\prompt_r1.md
research/stage4_math_and_design\discussions\M1\reply_r1.md
research/stage4_math_and_design\discussions\M1\round1_events.jsonl
research/stage4_math_and_design\discussions\M1\round2_events.jsonl
research/stage4_math_and_design/M1/THEOREM_M1.md:7:## 1. Exact discrete symmetry
research/stage4_math_and_design/M1/THEOREM_M1.md:19:### Theorem 1 — Encoder and conditional choice actions
research/stage4_math_and_design/M1/THEOREM_M1.md:37:### Theorem 2 — Quotient factorization
research/stage4_math_and_design/M1/THEOREM_M1.md:49:## 2. Continuous phase transport and the sector dip
research/stage4_math_and_design/M1/THEOREM_M1.md:51:### Theorem 3 — Circular phase evolution before the sector map
research/stage4_math_and_design/M1/THEOREM_M1.md:64:### Theorem 4 — Circulant Q3 pushforward
research/stage4_math_and_design/M1/THEOREM_M1.md:94:### Corollary — Why \(\pi/4\) and \(\pi/2\) give different accuracies
research/stage4_math_and_design/M1/THEOREM_M1.md:116:## 3. Stage 3D diagnostics and the scope of separation
research/stage4_math_and_design/M1/THEOREM_M1.md:129:### Theorem 5 — Additive centered-logit obstruction
research/stage4_math_and_design/M1/THEOREM_M1.md:137:#### Work F spectrum
research/stage4_math_and_design/M1/THEOREM_M1.md:148:### Theorem 6 — Reynolds averaging and NLL
research/stage4_math_and_design/M1/THEOREM_M1.md:160:## 4. Stage 4 `EquivariantEvidenceConnector`
research/stage4_math_and_design/M1/THEOREM_M1.md:170:### M3 interface — `RefSufficientPool`
research/stage4_math_and_design/M1/THEOREM_M1.md:176:### M4 interface — `PhaseContext` for Pluto and CTTC
research/stage4_math_and_design/M1/THEOREM_M1.md:182:## 5. CPU verification and Round 2 decisions
research/stage4_math_and_design/M1/verify_m1.py:40:def lift_scores(canonical, codes, task, pilot_turn=0):
research/stage4_math_and_design/M1/verify_m1.py:57:class SectorTransport(nn.Module):
research/stage4_math_and_design/M1/verify_m1.py:76:class FrozenHost(nn.Module):
research/stage4_math_and_design/M1/verify_m1.py:93:class AdditivePhaseBaseline(nn.Module):
research/stage4_math_and_design/M1/verify_m1.py:111:class EquivariantPhaseCoupled(nn.Module):
research/stage4_math_and_design/M1/verify_m1.py:156:def metadata(cond):
research/stage4_math_and_design/M1/verify_m1.py:162:def physical_kernel(transport, theta, variance, snr_db):
research/stage4_math_and_design/M1/verify_m1.py:166:def condition_kernel(transport, cond):
research/stage4_math_and_design/M1/verify_m1.py:171:def encode_view(encoder, mother, r1, r2):
research/stage4_math_and_design/M1/verify_m1.py:176:def train_data(encoder, transport):
research/stage4_math_and_design/M1/verify_m1.py:198:def fit(model, training, seed):
research/stage4_math_and_design/M1/verify_m1.py:215:def count_trainable(model):
research/stage4_math_and_design/M1/verify_m1.py:219:def scores(p, labels):
research/stage4_math_and_design/M1/verify_m1.py:224:def orbit_defect(p, shift_rule=lambda r, s: r):
research/stage4_math_and_design/M1/verify_m1.py:234:def blind_data(cell, encoder, transport, archive, cid, train_seeds):
research/stage4_math_and_design/M1/verify_m1.py:272:def physical_oracle(prior, kernels):
research/stage4_math_and_design/M1/verify_m1.py:279:def lookup(table, indices):
research/stage4_math_and_design/M1/verify_m1.py:283:def predict_table(model, meta, kernel):
research/stage4_math_and_design/M1/verify_m1.py:290:def predict_quotient_table(model, meta, kernel):
research/stage4_math_and_design/M1/verify_m1.py:294:def logit_lift_checks():
research/stage4_math_and_design/M1/verify_m1.py:314:def spectrum(matrix):
research/stage4_math_and_design/M1/verify_m1.py:333:def cal_rank_certificate(cal, overlap):
research/stage4_math_and_design/M1/verify_m1.py:357:def stage3d_diagnostics(archive, saved):
research/stage4_math_and_design/M1/verify_m1.py:401:def pf_counterexample():
research/stage4_math_and_design/M1/verify_m1.py:418:def boundary_scan(prior, g1_indices, transport):
research/stage4_math_and_design/M1/verify_m1.py:436:def benchmark_cell(cid, data, encoder, archive, saved, models):
research/stage4_math_and_design/M1/verify_m1.py:491:def main():
```

### Event #5: `item.completed` — `command_execution` (`item_1`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n -i 'EM-FloatFormer|Stage 4|Track M1|s17' 'C:/Users/李则徐/.codex/memories/MEMORY.md'; git status --short"
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
?? research/stage4_math_and_design/
```

### Event #6: `item.completed` — `command_execution` (`item_4`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n -i -C 2 '16.bit|bit layout|canon_angle|log_radial|resid_quality|def .*16' research/stage4_math_and_design/M5/THEOREM_M5.md research/stage4_math_and_design/M5/verify_m5.py"
```

**Aggregated Output:**

```text
research/stage4_math_and_design/M5/THEOREM_M5.md-123-| 24 | 2 | 2 | 8 | 8 | 4 |
research/stage4_math_and_design/M5/THEOREM_M5.md-124-
research/stage4_math_and_design/M5/THEOREM_M5.md:125:Use the same boundary convention as the historical encoder, define a missing/invalid state, and compute the prefix in the encoding pass. Right-shifting by the extension length recovers s17 exactly for every supported input. This also preserves the original reader and its edge cases. Lossy polar coordinates alone cannot guarantee recovery of a learned label computed from the full raw sample vector. If a real-data adapter lacks a valid historical triplet, mark the legacy interface unavailable. For a Bayes-optimal decoder, an extended packet can always emulate the legacy reader. A strict gain for a query requires a refinement that divides some old cell into distinct conditional laws of positive mass. The benchmark reader uses the physical fields and leaves the legacy inner index unused, while still counting its bits. Its results therefore quantify an achievable decoder rather than the optimal legacy-plus-physical reader. The 12- and 16-bit residual thresholds are not nested. The 24-bit layout is proposed and has no risk estimate in this round.
research/stage4_math_and_design/M5/THEOREM_M5.md-126-
research/stage4_math_and_design/M5/THEOREM_M5.md-127-Version the header with time origin/sample rate, template, gain and noise convention, calibration, quantizer edges, format ID, and capture/sample-span mapping. Record whether quality indicates clipping, noise, or a typed nuisance projection such as DC/spur. When required include irregular addresses, masks and per-window calibration bits in the budget. The CPU benchmark holds the address map and nominal amplitude/noise profile common to all readers and counts only the cache payload.
--
research/stage4_math_and_design/M5/THEOREM_M5.md-187-| Continuous canonical / Full IQ | 0.9438 | 0.7425 | 0.8934 | 0.3538 |
research/stage4_math_and_design/M5/THEOREM_M5.md-188-
research/stage4_math_and_design/M5/THEOREM_M5.md:189:The 16-bit gain over s17 is 0.3119 nat for event and location, paired pointwise 95% interval $[0.3023,0.3214]$. The remaining gap to Full IQ is 0.0735 nat. For the conditional smooth-versus-jump task the pooled NLL is 0.5240 (s17), 0.5136 (16 bit), and 0.5131 (Full IQ). For localization given a true jump it is 0.9711, 0.9382, and 0.9379, respectively. The recovery varies with regime. At +24 dB conditional binary NLL falls from 0.0300 with s17 to nearly zero with 16 bits. Under selective fading it is 0.6808 versus 0.6820, and both the 16-bit and continuous results are around 0.68. The observed low-SNR conditional localization remains near $\log3$ for all readers. Thus the larger joint event gain should not be read as an equally large binary-discrimination gain.
research/stage4_math_and_design/M5/THEOREM_M5.md-190-
research/stage4_math_and_design/M5/THEOREM_M5.md-191-The actual s17 encoder emits code 1 for two real, in-cell matched sums of magnitude 0.0200 and 0.5833 in the 2 dB Gaussian witness. The exact four-label posterior changes by 0.4804 total variation. The JSON also includes two IQ vectors sharing their matched sum and orthogonal energy that give distinct s17 codes. This establishes the need for an explicit legacy prefix even with a continuous canonical Gaussian cache.
--
research/stage4_math_and_design/M5/THEOREM_M5.md-193-### 5.2 Sealed one-window selection
research/stage4_math_and_design/M5/THEOREM_M5.md-194-
research/stage4_math_and_design/M5/THEOREM_M5.md:195:With the 16-bit cache, EIG, random, and nearest attain pooled NLLs 0.3544, 0.4497 and 0.4772, and accuracies 81.18%, 73.34% and 71.63%. The comparison below conditions on the same ambiguous-case stratum for all methods, defined for evaluation only by the old Full IQ smooth posterior in $[0.1,0.9]$. All 2,048 cases qualify in each of the three listed profiles. The +24 dB profile has no qualifying cases and shows a ceiling tie between policies at 16 bits.
research/stage4_math_and_design/M5/THEOREM_M5.md-196-
research/stage4_math_and_design/M5/THEOREM_M5.md-197-| Ambiguous profile | Max EIG NLL / accuracy | Random NLL / accuracy | Nearest NLL / accuracy |
--
research/stage4_math_and_design/M5/THEOREM_M5.md-217-## 6. Round 2 handoff
research/stage4_math_and_design/M5/THEOREM_M5.md-218-
research/stage4_math_and_design/M5/THEOREM_M5.md:219:- **M1.** Freeze the factored physical evidence interface, 16-bit candidate encoding, and exact legacy prefix. Preserve C4 transformation laws on candidate probabilities, and transform dynamics and reference factors consistently. Use matched old IQ to measure the separate cache and host readout gaps. Keep the continuous physical decoder as an oracle.
research/stage4_math_and_design/M5/THEOREM_M5.md-220-- **M3.** Freeze acquisition ID, timestamp, association probability, calibration, and noise-dependence fields. Verify idempotence for reused sources. Compare co-temporal pooling and cross-time messages against the joint likelihood. Keep the posterior mixture available for pre-reveal forecasts.
research/stage4_math_and_design/M5/THEOREM_M5.md-221-- **M4.** Use session- or acquisition-disjoint tuples. Calibrate SNR, gain and quality from allowed old data or a separate calibration split. Stratify natural, injected, and independent acquisitions, including low SNR and high fading. Test typed DC/spur projections when a scalar residual fails. Use a separately labeled oracle-calibration arm where needed.
research/stage4_math_and_design/M5/THEOREM_M5.md:222:- **M6.** Freeze risk margins by query and capture stratum, candidate-access rules, and held-out proof of sealing before training larger hosts. Compare 12 and 16 bits with all storage and read costs accounted for. Include the continuous ceiling, ambiguous-case coverage, a fixed-best development baseline, forecast calibration, step-by-step I/O and latency. Preserve unidentifiable and tied cases in the report.
```

### Event #7: `item.completed` — `command_execution` (`item_3`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n -i -C 2 'canonical_q2|canonical_q3|joint\\[|reference frame|valid_mask|a1\\b' research/stage4_math_and_design/M3/THEOREM_M3.md"
```

**Aggregated Output:**

```text
160-1. **Validate and canonicalize.** Use \(z_i=4a_i+u_i\). Resolve IDs and check provenance, calibration and exact sample equality before filtering. A production clock should publish one canonical timestamp per physical tick. For schema-tolerance aliases, the prototype sorts timestamps per ID, chooses the minimum, and rejects a spread beyond \(10^{-12}\). Remove independent-channel references before feature evaluation. A conflict-free distractor contributes no activation or gradient. A literal copy of a canonical acquisition ID makes no change. Sort unique IDs before reduction for deterministic sums.
161-2. **Keep exact evidence upstream.** Use sign swaps for the integer rotation \(i^{-a_1}\) on I/Q before floating-point reduction. Apply continuous derotation once per clock. Pack `Pooled(iq[B,64,2], times, precision[64], gap[B], family1[B], family2[B])` with zero padding excluded from the reader. The prototype batches one shared public design. A production wrapper can batch by shared design or use a padded per-row mask. Declare \(\omega,\sigma_\phi,h,\tau\) and the noise model as public inputs. The continuous path is differentiable for fixed metadata. ID groupings, channel flags, and clock equality are discrete.
162:3. **Expose fixed task evidence to M1.** Compute `canonical_q2[4]`, `canonical_q3[4]`, and optionally `joint[4,4]` from the reader. Provide a separate frame \(a_1\) for external sector relabeling. Use one always-present task evidence marker and fixed projection, with identical prose and no raw IDs, counts, flags, or clock lists. A multitask prompt can reserve both markers. The CPU prototype returns a single four-component Q2 slot. A projection to Qwen dimensions and an intervention on the frozen host remain separate integration work.
163-4. **Use the model posterior for guaranteed decisions.** Arbitrary host scores or trainable residual logits can change posterior calibration and rankings even at fixed length. Keep the physical posterior available as the decision output and the frozen-host readout as an experimental comparator.
164-
```

### Event #8: `item.completed` — `command_execution` (`item_5`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n -i -C 2 'receive.gauge|tx relabel|joint tx|r3|correlat|pluto|omega_hat' research/stage4_math_and_design/M4/THEOREM_M4.md"
```

**Aggregated Output:**

```text
13-| View | Law and consequence |
14-| --- | --- |
15:| **R1**, one symbol at three sub-symbol times | Use three fractional-time samples of an already matched-filtered stream, e.g. offsets `(-2,0,+2)` at Pluto SPS20 and `(-1,0,+1)` at CTTC SPS4. The vector \(R=aqS+d+N\) has a pulse/timing steering vector \(q\) and colored \(K\); concatenating adjacent transmitted symbols sees a different target model. Normalize gain, estimate \(q,K\) on authorized calibration observations, and preserve R1 as a diagnostic. Projecting a known single-symbol Gaussian steering model is sufficient. Actual ISI can make off-axis samples informative. |
16-| **R2**, repeated scalar | Put \(Y=eT\) with \(T=(\tilde y-\widehat d)/\widehat a\), so `matched_sum(Y)=T`. Repetition, and any deterministic linear interpolation of a single scalar along \(e\), retain its scalar information but produce a rank-one complex law. \(D(P_Y\|P_{Y_0})=+\infty\) even if \(T\) has the nominal marginal. A scalar-domain KL remains meaningful. |
17:| **R3**, whitened scalar lift | Let \(B\) have two orthonormal columns orthogonal to \(e\). Set \(Y=eT+B\eta\) with an independent \(\eta\sim\mathcal{CN}_2(0,V_0I)\). This preserves \(T\), and \(D(P_{S,Y}\|P_{0,S,Y})=D(P_{S,T}\|P_{0,S,T})\) by factoring the common independent ancillary law. With nominal scalar mean and variance the synthetic three-tap law is exactly recovered. The ancillary draws carry zero radio evidence and are preassigned in a frozen stream shared by arms. Rotate the realized \(Y\) and all receiver-coordinate evidence together in equivariance replays. |
18-
19-For R1, \(T=q^*K^{-1}(R-d)/(\widehat a q^*K^{-1}q)\) is the noise-whitened matched estimate. This follows by expanding the Gaussian likelihood: the only class-dependent observation term is \(2\operatorname{Re}(a^*\bar S q^*K^{-1}(R-d))\). The true variance of \(T\) is \((\widehat a^2 q^*K^{-1}q)^{-1}\); a fixed unitary on the whitened taps can align the steering direction with \(e\). Replacing \(q\) by \(\widehat q\) leaves a component \(\mu_\perp(S)\) orthogonal to the retained direction. With exactly whitened unit covariance and known Gaussian steering,
--
25-The bound follows by comparing the conditional off-axis distribution to independent standard noise, using the KL chain rule. Its upper term vanishes for correct steering. Pulse-shape error, signal-dependent ISI, and improper noise require a fuller law. One may whiten an augmented real vector when the pseudo-covariance is nonzero, using the real Gaussian formula with its factor of \(1/2\).
26-
27:The optional forced-calibration control is \(T'=\alpha T+\sqrt{V_0-\alpha^2v}\,Z\), where \(Z\sim\mathcal{CN}(0,1)\), \(\alpha=\min(1,\sqrt{V_0/v})\), and \(v\) is a causal estimate of post-AGC thermal variance. With correct \(v\), this matches the thermal variance, and its conditional Gaussian KL becomes \(|\alpha\mu_S-S|^2/V_0\). When \(v>V_0\), attenuation alone is invertible at continuous input and shrinks the signal mean. When \(v<V_0\), adding noise degrades information. Real noise with random scale, coherent sidebands or ISI retains those effects through this operation. Use the SNR-preserving R3 in the primary comparison, with causal variance and gain-uncertainty summaries available equally to all matched-information decision rules. Use R1, R2, and forced calibration on dev or in a named CPU diagnostic.
28-
29-### 1.2 Conditional KL and perturbation terms
--
48-\]
49-
50:This is conditional on \(U\). A term may be stochastic or correlated with others. Keep the coherent cross terms, or use the triangle inequality to upper bound the squared numerator by \((|\delta|+2|\sin(\vartheta/2)|+r+|b|+|j_s|)^2\). With uniform \(s\), constant \(b\), and \(j_s=0\), the exact average squared shift is \(|e^{i\vartheta}-g+re^{i\psi_L}|^2+|b|^2\). At gain and variance match, isolated \(b=0.3\) costs `0.142640` nats, isolated \(r=0.26\) costs `0.107139`, and isolated CFO costs \(4\sin^2(\epsilon_\omega L/2)/V_0\). At `0.020 rad/symbol`, lags 4, 25 and 126 cost `0.010138`, `0.388037` and `5.746671` nats. A centered independent Gaussian residual CFO with standard deviation \(\sigma_\omega\) gives average phase energy \(2[1-\exp(-L^2\sigma_\omega^2/2)]\). This follows from the characteristic function of the residual. It bounds KL to the fixed phase likelihood after dividing by \(V_0\). The bound does not grow unboundedly with lag. A coherent spur of 0.26 amplitude has power ratio 0.0676 and phase beat about `0.018887 rad/symbol` at a 1503 Hz offset and 500 ksym/s. Represent its phase jointly across windows. The measured sideband ratio alone does not distinguish a multiplicative sideband from an additive tone. For an additive tone, move \(re^{i\psi_L}\) into \(b\) within (2).
51-
52:Pluto normalized residual DC can be much larger at low gain for the same absolute subtraction error. For \(\widehat a=a(1+\delta)\), the DC-only term is \(|\Delta d|^2/(\widehat a^2 V_0)\). Its AC-coupling component can depend on the transmitted sequence, which may create conditional mean and covariance terms together. For a proper Gaussian AR(1) approximation to the **total** tap residual with normalized covariance \(K=\lambda V_0 R_\rho\) and \((R_\rho)_{ij}=\rho^{i-j}\) for \(i\ge j\) and Hermitian symmetry, Eq. (1) gives covariance cost
53-
54-\[
--
56-\]
57-
58:The determinant follows by factoring each successive innovation with variance \(1-|\rho|^2\). If ISI contributes only a portion of colored variance, use \(K=(v_{\rm AWGN}I+v_{\rm ISI}R_\rho)/g^2\) in (1). A single measured correlation coefficient without a full joint law or eigenvalue floor cannot determine KL. Discrete neighboring symbols yield a finite mixture if those symbols are marginalized. Overlapping symbol windows and common random gain or CFO can induce extra dependence even with white thermal noise. Stack the full covariance or use the joint likelihood; summing isolated-window KL is exact only under the appropriate factorization.
59-
60-For a Gaussian nuisance mixture with the same class prior, including a class-dependent nuisance law if needed, the exact identity is
--
98-Rayleigh denominator with \(\mathbb E G^2=1\) is \(\nu=1\). In one complex dimension \(P(|X|^2/\Sigma>q)=(1+q)^{-1}\), so \(\mathbb E|X|^2=\infty\), and forward KL to any nondegenerate Gaussian is infinite. The normalized Student entropy is finite here. More generally an infinite second moment gives infinite KL to a Gaussian by testing bounded truncations of a sufficiently small positive multiple of \(\|X\|^2\) in the variational KL lower bound. If the amplitude estimator itself has a folded Student law, substitute its density \(2f_{t_d}(g/b)/b\) for \(g>0\) in (6), where \(f_{t_d}\) is the standard real Student density and \(b\) its scale. That exact integral is a different scale mixture, with infinite second moment since the folded density is positive at zero. A noisy complex pilot LS amplitude \(|h+E|/a\) with nondegenerate proper Gaussian \(E\) is conditionally Rician. Its radial density has positive linear coefficient at zero, so \(\mathbb E G^{-2}=\infty\) even for nonzero \(h\). If `a` itself is Rayleigh but estimated exactly and both amplitude and noise scale with it, \(G=1\) and the heavy tail argument does not apply. If a new thermal sample and the scale estimator share noise, the exact law uses \(W\mid G\) in (6) and cannot be inferred from the gain histogram alone.
99-
100:AGC usually divides the signal too. For \(T=(\mu_s+W)/G\), the exact class-conditional density replaces \(G^2x^*\Sigma^{-1}x\) in (6) with \((Gx-\mu_s)^*\Sigma^{-1}(Gx-\mu_s)\). It is a noncentral scale mixture and generally has no centered Student formula. Absolute Rayleigh fading and fixed receiver noise can also produce a heavy-tailed unconditional noise term even with exact gain estimation, since averaging \(W/a\) over \(a\) invokes the Rayleigh case. A causal guard \(\widehat a\leftarrow\max(\widehat a,a_{\min})\), with floor fixed from training/calibration and carried in the uncertainty flags, makes \(\mathbb E G^{-2}\) finite conditional on \(a\). Fit variance floors, clip estimator outliers as registered on dev, and model or report flags. When estimating a noise scale with \(\nu\) complex residual degrees of freedom independent of the new observation and a correctly specified proper Gaussian linear fit, the normalized residual power estimate has Gamma shape \(\nu\). In a rank-\(p\) complex fit to \(n\) such pilots, \(\nu=n-p\) follows from orthogonal residual projection. Estimated CFO, branch selection, regularization, symbol decisions, correlated pilots, and unknown DC break this pivot unless accounted for. Use a robust likelihood or (4) with measured nuisance uncertainty in those cases.
101-
102-For a finite label alphabet, the unrestricted scalar posterior for any specified mixture is obtained by normalizing those four densities. Its uniform-input MI is exactly
--
114-### 2.1 Candidate rows and time order
115-
116:Use time in symbols with an explicit sample-clock conversion and a common fixed transmitter mapping \(e^{-i\pi/4}s_{\rm hardware}\) into \(C_4\). Use dataset-specific Gray lookup in the scorer. For Pluto divide the nominal `0.707` axis levels by their fixed constellation magnitude. Use a corresponding fixed receiver coordinate transform, subtract only the causally available DC estimate, and retain the residual channel angle. Each data window contains observations of **one transmitted symbol**. Record the two codes and supporting provenance, release a reference only after the first forecast is committed, and never revise either code. A Q3 forecast stores both the cache-stage and reveal-stage four-vectors before the first sample of the target stencil is eligible. Collector latency is the maximum raw time in each interpolator stencil, with all clocks and cutoffs materialized.
117-
118-| Dataset | Q1 and Q2 window schedule | Revealed references | Q3 target |
119-| --- | --- | --- | --- |
120:| Pluto 850 MHz | One row per frame. Copy 1, \(k_1=30+(f\bmod67)\); \(k_2=k_1+h_{12}\) with \(h_{12}=1\) or 8 using `(f + floor(f/16)) mod 2`. | Two acquisitions at logical indices `15` and `18` of copy 1. Initial state uses sync `0:13` and split-local past, with those two pilot reads excluded from all pre-reveal quantities. | Same payload slot \(k_1\) in the immediately following complete copy, \(L=126\), \(\ell=126-h_{12}\). Score 2,240 candidates with a common lock-failure policy. A copy is a real retransmission; no target-label matching or synthetic replacement. |
121-| CTTC 433/708/2450 MHz | One row per packet with ordinal \(j\) and loop \(b\). \(k_1=136+((17j+5b)\bmod48)\); \(k_2=k_1+h_{12}\), using 1 for even \(j+b\), 8 otherwise. | `126,127` of the same packet, both earlier than \(k_1\). Initial state uses `0:6` sync and authorized previous data, excluding both release samples. Estimate their joint covariance. `254,255` remain unavailable until they have physically arrived. | On each capture use disjoint loop pairs `(0,1),(2,3),(4,5),(6,7)` at the same packet source start and same \(k_1\) selected in the earlier loop. \(L=500000\), \(\ell=500000-h_{12}\). Q1/Q2 cover 1,440 packets. Q3 has 720 retransmission candidates and 36 loop-pair groups. |
122-
123:Q1 label is \((x_2-x_1)\bmod4\) from the transmitter vault. Q2 predicts \(x_1\) in the absolute transmitter convention with `none`, `one`, `two`; the second reference is an actual sample at a different time with its own known symbol and possible correlation. Q3 predicts the sector of the future measured scalar \(e^{-i\pi/4}\tilde y_3\) in a fixed receiver phase convention. A DC-centered target is also defensible if the subtraction is frozen at the initial cutoff for every arm; use the raw-sector definition above as the primary. Q3 can be near uniform at the CTTC one-second lag. A fresh unrelated packet with a different source index cannot be called a known retransmission. For Q2, the same four-vector supplies symbol NLL, MAP SER and per-bit NLL from \(P(b_v=j)=\sum_s p(s)\mathbf1[b_v(s)=j]\). Define posterior-mean RMS EVM as \(\sqrt{\sum_k|\sum_s p_k(s)s-s_k|^2/\sum_k|s_k|^2}\) on the same rows. Conventional equalized-IQ EVM, when an arm returns physical \(\widehat s_k\), is a separate named endpoint with the same denominator. Q1/Q3 classification is scored on its own target, with Q2 anchoring receiver metrics rather than pretending the Q3 label is a TX bit.
124-
125:The primary samples retain actual fading, residual DC/ISI, noise, clock offset, and any changes present at registered times. A naturally occurring jump is a stratum, with changes flagged by a predeclared causal split-local pilot innovation criterion and a retrospective scorer-only morphology check. A CPU-only cross-frame Q1 diagnostic pairs `(f−1,f)` at `f mod 20 = 0` within each test block, using the earlier frame's \(k_1\) and current frame's \(k_1\) from the table. Its matched control pairs the preceding two interior frames `(f−2,f−1)` in that same block. Use only available sync and authorized pilots and a gap-class prior when the time between Pluto columns is unavailable. This samples recorded batch boundaries by schedule without requiring or asserting a jump at each boundary. Separate source domains and nominal session strata in analysis, keeping capture IDs out of model text/features. Hold `f708_snr10` out of fitted connector weights, temperatures and shared prior tuning to expose one capture transfer; allow its own causal sync and released pilots. Pluto is a blocked same-campaign test. CTTC has one public cyclic TX sequence; input-source disjointness and capture dependence are distinct issues. Use block-paired NLL/SER differences on 16 Pluto groups and 72 CTTC session-loops, group adjacent repeat pairs for CTTC Q3, and report a capture-level sensitivity analysis and source-ordinal sensitivity because loops/captures reuse source positions. Report all registered rows and lock coverage, with the same uniform predictive fallback on failed rows and a secondary locked-only analysis. Avoid treating all packets as independent. Select the Q2 `two−one` proper-loss contrast and large-angle synthetic Q3 coupled-vs-additive contrast before test, with paired intervals and the multiplicity rule frozen with M2.
126-
127:Keep the natural recordings as the primary results. Recreate seven Pluto stress recipes on 140 frame IDs `range(4800,7040,16)` each, and all 1,080 CTTC stress IDs from the branch identity rule (`i mod 4 != 3`, round-robin modes). Score a single two-reference Q2 vector per registered stress row in the GPU budget. For perturbation strength use frozen train-scale or authorized pre-cutoff sync/pilot-only power. Pluto’s legacy injection measures power across the full frame, and CTTC’s legacy injection uses `offline_sig_pow` derived with full TX truth. Their historical raw scores stay available as historical comparators. Hide exact injected CFO, noise power and jumps in the scoring vault and keep RNG independent of payload labels. Real-only CFO uncertainty is estimated from observed pilot evidence. [Pluto data and acquisition](https://data.mendeley.com/datasets/k72b4pnc5g/1) · [CTTC data and acquisition](https://zenodo.org/records/4895133)
128-
129-### 2.2 What zero leakage requires and what was verified
130-
131:**Split proof.** Pluto uses frame intervals `[0,3200)`, `[3300,4500)`, `[4600,7040)` including the first 200 warmup frames of each split. The 100-frame gaps imply pairwise empty intersections. All copies of a frame are inseparable across splits. CTTC uses six disjoint cyclic source ranges of length-500,000 circumference, including `train target [0,160000)`, `train ref [164096,200000)`, `dev target [204096,290000)`, `dev ref [294096,330000)`, `test target [334096,455000)`, `test ref [459096,495904)`. The boundary gaps are at least 4,096 symbols, including the wrap. The verifier regenerates starts `lo + i floor((hi−lo−256)/n) + 128`, enumerates both bit pairings' union and asserts disjoint bit footprints across split and role. On the archived nine anchors the 5,472 target/reference acquisitions fit inside their files, have distinct IDs, and the nearest receive ranges remain 16,481 samples apart after 255-sample RRC support expansion. This verifies the stored geometry, including 72 test groups. Acquisition edges must be rechecked on fresh causal alignment. Source sequence reuse within a split creates statistical dependence; the proof addresses explicit identity and support leakage.
132-
133-**Filtration proof.** The released model payload comprises timestamped W1/W2 entries and a causal state with nuisance parameters estimated only from authenticated training/allowed sync and prior pilots. Each state node records raw-support intervals, truth whitelist, availability time and acquisition IDs of its parents. Acquisition truth is delivered as individual permitted known symbols; the feature worker has no access to `Tx_data` payload or the full CTTC source sequence. Build an independent scorer vault containing all targets, withheld probes, archival oracle fields, future IQ and injected parameters. Model projections strip frame, capture, packet and source ordinals, plus provenance fields that could identify cyclic TX labels. Prompt content carries lags, units and predeclared uncertainty summaries. Auditable salted observation IDs stay outside the prompt and are used for exact de-duplication. Denote the allowed baseline information by \(\mathcal F^0\). Its closure under deterministic functions and independent lift randomness contains the committed cache. Updating only by the union with released past pilot observations yields \(\mathcal F^r=\mathcal F^0\vee\sigma(R_1,\ldots,R_r)\). A listed input lies in this filtration by induction on the provenance graph. Intersection checks of all authorized raw read supports with actual withheld-symbol read stencils, and the inequality on every raw cutoff, certify direct observation masking once the time geometry is established. The verifier checks disjoint logical indices and local cubic-stencil examples over the registered ranges. Adjacent samples can retain analog contributions through real ISI, matched filtering or DC memory. Physical dependence is part of the channel model and cannot be excluded by an index proof.
134-
135:**Timing and CFO gates.** The legacy Pluto `fast_blind_frontend` correlates whole captures at lag 2520 and scans two copies for header timing. Even randomized payload and unlabelled probes can enter those sums. CTTC's `anchor_capture` uses whole-source lock candidates at several times and long known-source CFO fits, and extraction includes `offline_cfo_hz`, `h_offline_ref` and powers derived using all TX symbols. None of these are admissible causal inputs. For Pluto use an independently registered trigger or bounded timing prior established from earlier authorized sync, verify the **entire candidate search stencil** fits in the allowed sync envelope, then track only with a prefix of the intended copy. For CTTC reserve a disjoint in-cycle commissioning segment, for example source `[496256,498304)` plus sufficient guard for RRC/padding. Use its whitelisted bits and receiver samples strictly before the first use at a given cutoff to resolve EVEN/ODD and coarse timing/SFO. Use the public clock rates as fixed constants. Require a timing-uncertainty certificate for subsequent local interpolation and renewal, and exclude same-split or cross-session reference observations that arrive later than the prediction. The legacy CTTC role order places the same-loop reference pool after its target range, so only already acquired same-split entries may supply history. Reset state at split boundaries and document commissioning as a distinct authorized acquisition with a disjoint source footprint. If safe synchronization cannot be certified, emit a common fallback and report coverage. Strict raw masking and the full usable row counts are conditional on this gate. A Pluto window with an unconstrained initial timing offset cannot certify strict raw masking from those raw samples alone: for every tentative input position some admissible circular offset places a held-out slot there. An offline alignment-assisted diagnostic may be reported with its timing provenance visible.
136-
137-Use causal sync and then revealed pilots for residual CFO. For synchronized observations \(y_p\) and authorized \(s_p\), fit \((H,d,\omega)\) using the pilot-only profile loss
--
151-| **Frozen Qwen Zero** | Same host, W1/W2 codes, admissible public metadata, identical reference observations and reveal schedule. Set phase-coupling numeric slots to zero. Fit the zero control connector with the same steps and optimization budget. |
152-| **Naive-Additive** | Encode the same signed estimated phase and reference features through an additive real token slot or logit correction, with no enforced covariance of the correction. Hold parameter scale, reference code/features, losses, training rows and steps fixed. |
153:| **Stage4-Equivariant-Coupled** | Canonicalize the received gauge from the turn in the first valid cached code. Rotate both codes and receiver-coordinate pilot evidence by that gauge, propagate the signed phase law with explicit lag, then map the answer vector back with its registered group action. Under a common receive-chain quarter-turn, Q1 and Q2 TX labels stay fixed and Q3 receiver sectors permute. If testing a TX relabeling, known pilot labels and Q2 labels transform together in a separately specified action. A single canonical host score suffices. Register validity before projection and audit an exact-zero or missing-pivot policy. |
154-| **Stage4+RefSufficientPool** | Replace ordered raw-reference tokens by a set likelihood summary on the same pilot observations, with release times and known complex TX values. Deduplicate exact acquisition copies before pooling. With joint proper Gaussian \(y=A_mH+n,\ n\sim\mathcal{CN}(0,K_m)\), sufficient for \(H\) at fixed \(m\) are \(A_m^*K_m^{-1}y\) and \(A_m^*K_m^{-1}A_m\). For comparing candidate \(m\) also retain \(y^*K_m^{-1}y\), normalizers (e.g. \(\log\det K_m\)), and weights/times. Complete the square to see all remaining terms are constant in \(H\). For overlapping/colored blocks use joint covariance or unique innovations. For time-varying states use the corresponding transition and filter state or a bank of likelihood factors. A single summed phasor has no general sufficiency under unknown CFO, shared noise or jumps. |
155-
--
162-| Resource | Request and verification |
163-| --- | --- |
164:| Pluto | `GET https://data.mendeley.com/public-api/datasets/k72b4pnc5g?version=1`, then `GET https://data.mendeley.com/public-api/datasets/k72b4pnc5g/files?folder_id=root&version=1`. Use `content_details.download_url` with size and SHA-256 checked against the pinned downloaded manifest. `QPSKlink.mat` is MATLAB v5, **822,762,822 B**. Reader arrays and label vault go to distinct outputs. |
165-| CTTC | `GET https://zenodo.org/api/records/4895133/files`, with file bytes at `/api/records/4895133/files/{filename}/content`. Nine QPSK captures contribute **1,807,824,953 B**. Including README and sources, the selected archives total **1,966,425,693 B**. Verify bytes and official MD5 and pinned SHA-256 for each, then extract and verify only the authorized named files. `file_2.dat` at `176,015,264 B` × 9 plus configs and the 1,000,000-byte QPSK source totals **1,585,139,947 B** extracted for this plan. |
166-| Combined | **2,789,188,515 B** downloaded before model weights. The proposed 24 GiB envelope covers raw and selected extracted files, compact caches, model snapshots, and results. Pin the imported dataset bytes in a single ingestion job and use commit/reload between jobs. Keep volume use and compact result egress within verified aggregate Starter quotas. |
--
168-The exact download URLs, file sizes, hashes and archived manifest parity are in [results_m4.json](results_m4.json). The verifier can probe metadata with `--check-http` and still complete offline. Ingestion should verify before exposing an artifact to any downstream job, and allowlist regular files when extracting archives. Scientific sources and response fields come from the named branch downloader versions and the public archive pages. [Mendeley record](https://data.mendeley.com/datasets/k72b4pnc5g/1) · [Zenodo record](https://zenodo.org/records/4895133) · [Modal Volumes](https://modal.com/docs/guide/volumes)
169-
170:Use one CPU ingestion function, with a content-addressed `raw/model` cache committed before scheduling dependents. Reuse a verified snapshot of the historical Qwen revision when possible, or fetch only that pinned revision. A second CPU function streams Pluto frames and CTTC `file_2.dat` memmaps into train/dev/test feature shards. It computes split-local frontend state, per-stage pilot inputs and CPU physical baselines. Use separate reader mounts for scorer-only material and GPU-visible caches. Store frozen host scores, checkpoint and shard manifests on a results Volume, and export compact paired rows for independent CPU replay. Stage default time caps cover image/container boot and CPU dispatch overhead via the explicit reserve lines. Use only per-second compute Functions in this envelope. Modal charges CPU and memory by the larger of request and actual consumption. Enforce `cpu=(c,c)` and `memory=(1024g,1024g)` for the paid caps in the table, with an economical low-CPU-memory model loader. [Modal CPU and memory](https://modal.com/docs/guide/resources)
171-
172-### 3.2 Model and scored workload
--
208-The script loads the exact frozen weights and the archived branch source via read-only `git show`. It uses 8,192 examples per cell in the 168-cell mapping factorial and 32,768 examples per nuisance-curve point. This is a synthetic, matched-filter surrogate with a specified steering vector and colored proper Gaussian covariance. It verifies KL against quadrature or sampled log-density ratios, estimates scalar Bayes MI and empirical code capacity, and scores actual encoder codes and the frozen lookup table. At one full complex Gaussian point, (1) gives `1.196255` nats versus `1.198460 ± 0.004144` Monte Carlo (one standard error). A bounded random gain mixture with equiprobable denominators `0.7,1.3` has an estimated joint KL of `0.185511` nats after marginalization. Its scalar MI is `0.873286` nats with hidden gain and `0.889764` with observed gain. The centered scalar Student noise experiment has about `0.587033` nats of scalar MI for \(\nu=1\) with infinite forward Gaussian KL. All sweep inputs are generated surrogates. The optional metadata requests were refused by the local network. Historical manifest comparisons passed.
209-
210:| Surrogate condition | R1 SER | R2 SER | R3 physical SER | R3 calibrated SER | R3 physical vs calibrated NLL |
211-| --- | ---: | ---: | ---: | ---: | ---: |
212-| −2 dB, clean gain/DC/sideband | 39.06% | 37.56% | 37.56% | 37.56% | 1.129 vs 1.008 |
--
215-| +20 dB, +30% AGC, DC 0.30 and spur 0.26 | 0.00% | 0.00% | 0.00% | 32.54% | 0.239 vs 0.879 |
216-
217:The verifier checks directly that the frozen table chooses the turn's class for all four local codes, which explains SER equality of the scalar R2 and R3 arms. The matched sum and hence the turn coincide under the lift. The local code and calibration can still change. Attenuation at low SNR can improve frozen NLL without improving SER. The high-SNR loss from variance locking occurs when injected noise masks real high-SNR evidence. The observed saturation codes and conditional nuisance curves characterize the frozen weights under the stated draws. Generalization, real timing coverage and throughput remain empirical gates. See the full records for sample counts, scores, transition matrices, budget, manifests and geometry. [Verifier](verify_m4.py) · [Results](results_m4.json)
```

### Event #9: `item.completed` — `command_execution` (`item_6`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n -i -C 2 'M_\\{|margins|svd|candidate.*jacob|eta_min|slot.*swap|NIG|ECE' research/stage4_math_and_design/M6/THEOREM_M6.md research/stage4_math_and_design/M2/THEOREM_M2.md; uv --version; uv python list --only-installed"
```

**Aggregated Output:**

```text
research/stage4_math_and_design/M2/THEOREM_M2.md-43-\]
research/stage4_math_and_design/M2/THEOREM_M2.md-44-
research/stage4_math_and_design/M2/THEOREM_M2.md:45:The boundary derivative term generally remains at \(T^*=\infty\). A finite positive \(T\) preserves each row's ranking, so ECE and NLL can change while accuracy stays fixed. At infinity the four-way tie awards fractional accuracy \(1/4\). Reported oracle fits are label-optimal scalar temperatures, separately for each arm and seed. The JSON also fits a single common raw temperature per arm and cell. ECE uses the same pooled ten top-confidence bins as Stage 3D. ECE is neither proper nor an additive component of log risk.
research/stage4_math_and_design/M2/THEOREM_M2.md-46-
research/stage4_math_and_design/M2/THEOREM_M2.md-47-For any fixed predictor \(q\), define two joints \(P(c,y)=w(c)p_y(c)\) and \(\widetilde Q(c,y)=w(c)q_y(c)\), where \(c\) includes seed when fitting multiple connectors. The chain rule in reverse order yields
--
research/stage4_math_and_design/M2/THEOREM_M2.md-89-Thus both pools weakly improve invariant-design log risk over the raw model. Their ordering depends on the label law and the forecasts. Normalization makes the geometric pool the minimizer of \(\sum_g D_{\rm KL}(r\Vert R_gq)\) over \(r\). It is equivariant, and applying it again returns the same distribution. The arithmetic pool minimizes \(\sum_g D_{\rm KL}(R_gq\Vert r)\) and is also equivariant and idempotent.
research/stage4_math_and_design/M2/THEOREM_M2.md-90-
research/stage4_math_and_design/M2/THEOREM_M2.md:91:For each target \(y\), the closed top-1 region is the intersection of halfspaces \(r_y-r_k\ge0\) over \(k\ne y\). For the arithmetic path, the exact margin to competitor \(k\) at the barycenter is the mean of aligned margins. A unique raw correct decision is lost iff a mean margin is negative or hits a tie boundary. A unique correct decision is gained iff all mean margins are positive after a raw error or tie. The fractional-tie hit function at a row is
research/stage4_math_and_design/M2/THEOREM_M2.md-92-
research/stage4_math_and_design/M2/THEOREM_M2.md-93-\[
--
research/stage4_math_and_design/M2/THEOREM_M2.md-95-\]
research/stage4_math_and_design/M2/THEOREM_M2.md-96-
research/stage4_math_and_design/M2/THEOREM_M2.md:97:Over a target distribution, strict accuracy decline is equivalent to \(\mathbb E[(h_y(q)-h_y(Pq))_+]>\mathbb E[(h_y(Pq)-h_y(q))_+]\), including ties. At population level this is the same as \(\mathbb E_X[p(X)\cdot(d(q)-d(Pq))]>0\), where \(d\) is uniform on the maximizer set. For geometric pooling the corresponding competitor margins are averages of \(\log r_y-\log r_k\). Log score depends on the probability assigned to the true class, while top-1 depends on which side of the decision planes the pool lies.
research/stage4_math_and_design/M2/THEOREM_M2.md-98-
research/stage4_math_and_design/M2/THEOREM_M2.md-99-## 3. Stage 4 frozen statistical gate
research/stage4_math_and_design/M2/THEOREM_M2.md-100-
research/stage4_math_and_design/M2/THEOREM_M2.md:101:A claim passes when every endpoint for that claim clears its limit. Freeze the candidate, seed handling, allowed observation sets, per-class weighting, code below, multiplicity family, score floor, calibration, and all margins before unblinding. Apply the same three gate parts separately to the synthetic extrapolation claim, the Pluto claim and the CTTC claim. For a claim covering all four synthetic cells, require each cell to pass. Report each failed or indeterminate endpoint. A winner picked from several candidates requires a design split or inclusion in the multiplicity family.
research/stage4_math_and_design/M2/THEOREM_M2.md-102-
research/stage4_math_and_design/M2/THEOREM_M2.md:103:1. **Absolute and comparative performance.** Use paired lower bounds. Require NLL to beat uniform, the dev-frozen prior, and the matched zero-phase system, with a \(0.02\)-nat increment proposed against matched zero. Require accuracy above analytic chance, a dev-frozen prior classifier, and the matched zero-phase system, with \(+0.02\) absolute over chance and zero proposed as initial margins. With imbalanced classes, also require outperformance of the fixed majority prediction or used prior competitor, and report balanced accuracy as a separately locked estimand. Require valid chance/random-label randomization tails for the accuracy and NLL statistics where the design supplies exchangeability. Choose an oracle with the same information budget as the candidate for the main NIG denominator and disclose a finer physical oracle separately. Propose \(\eta_{\min}=0.10\) and validate \(H(Y)-L(o)>0\) on an independent development design or by a co-primary lower bound. Test the linear contrast \(D_{\eta}=(1-\eta)H(Y)-L(q)+\eta L(o)>0\) rather than dividing two noisy quantities. Use \(H(Y)=\log K\) for a genuinely uniform target law. Otherwise predefine the target distribution or carry uncertainty in \(H(Y)\) by a joint confidence region and minimize the contrast over it. For real data without a defensible Bayes oracle, freeze a physically informed predictive reference, call the resulting metric reference-normalized, and withhold the Bayes-normalized claim. Requiring a learner to outperform a true Bayes oracle under proper loss is infeasible. A positive NIG floor defines a fraction of the attainable improvement. Controlled slot/text interventions form a further prerequisite for a specifically signed-slot mechanism claim. Require improvement with matched versus permuted or flipped slots at fixed text and same \(X\), with seeded training and ablations paired. Use the native receiver outputs for the gate.
research/stage4_math_and_design/M2/THEOREM_M2.md:104:2. **Calibration and structure.** Propose an upper bound of 0.05 on top-label ECE after dev-only calibration with ten frozen bins. Propose a paired Brier improvement of at least 0.005 against the strongest locked trivial or zero-phase Brier competitor, plus a strict improvement over every required comparator. For \(K=4\) uniform Brier is \(1-1/K=0.75\) with the unnormalized multiclass sum. Prelock a probability transformation ensuring finite NLL, for example \(q_\epsilon=(1-K\epsilon)q+\epsilon\) with small fixed \(\epsilon\), and score that forecast consistently. For calibration inference set \(d_j=\mathbb E[1_{\{\operatorname{conf}\in I_j\}}(h_Y-\operatorname{conf})]\); \(\operatorname{ECE}=\sum_j|d_j|=\max_{v\in\{-1,1\}^{10}}v^\top d\). Test all \(2^{10}\) linear upper-margin conditions as an IUT or use a valid joint confidence region for \(d\). Estimating the absolute value first and treating empirical ECE as an unbiased, normal per-mother loss has no exact justification. Require \(\max_{g,x}\|q(gx)-\rho(g)q(x)\|_\infty\le10^{-5}\) and projection/readout idempotence \(\max_x\|P(Pq)(x)-Pq(x)\|_\infty\le10^{-5}\). Verify across the whole finite code domain and all group elements on synthetic. For continuous real inputs distinguish formal architectural guarantees from the observed maximum on prespecified transforms. Test only valid actions with physically correct label permutations. Audit the deployed path and the offline projector separately.
research/stage4_math_and_design/M2/THEOREM_M2.md-105-3. **Hierarchy and multiplicity.** Keep all repeats, rotations, acquisitions, and matched arms inside their mother and higher cluster. Report a frozen-seed conditional estimate and, if claiming generalization to retraining, a separate random-seed inference. Use equal independent cluster weights for the following exact balanced analysis. Pluto supplies 16 units if they are independent captures. For a CTTC new-session claim use nine independent session means, with all eight clusters per session contributing to that session. A conditional claim on the fixed sessions needs the corresponding estimand and defensible within-session cluster independence. For a permutation or label-randomization gate, lock the allowed exchangeability orbit and recompute the same statistic throughout. Repeated/rotated labels move jointly with their mother and never split correlated session units as though independent. A synthetic randomized design with independent label assignments can instead rerandomize the sanctioned generator operation inside the original design. When there is no justified exchangeability, collect a randomized negative control. State whether it tests a sharp random-label null or a mean-effect margin. A permutation p-value \((1+\#\{T_{\rm perm}\ge T_{\rm obs}\})/(R+1)\) with uniform randomizations and ties included is conservative under the sharp design. Use exact enumeration when feasible.
research/stage4_math_and_design/M2/THEOREM_M2.md-106-
--
research/stage4_math_and_design/M2/THEOREM_M2.md-143-B\ \ge\ \frac{\sigma_u^2+\sigma_e^2/S}{d^2/z^2-\sigma_\nu^2/S}
research/stage4_math_and_design/M2/THEOREM_M2.md-144-\]
research/stage4_math_and_design/M2/THEOREM_M2.md:145:with positive denominator. At fixed \(S\) this is a necessary known-variance bound for the random-seed model. The verifier solves the exact t power and evaluates the crossed chi-square integral for finite counts. It uses a worst Holm first-step threshold \(\alpha=0.05/3\) and marginal power 0.95 to guarantee at least 0.90 power for the two historical paired endpoints if both assumptions and excess effects hold. This target does not imply 0.90 power for the full Stage 4 gate. Full planning requires variance and target excess for every required absolute/calibration endpoint and a joint power calculation or an explicit allocation of endpoint type-II errors. Stage 3D supplies a synthetic variance reference. It cannot measure Pluto or CTTC variability.
research/stage4_math_and_design/M2/THEOREM_M2.md-146-
research/stage4_math_and_design/M2/THEOREM_M2.md-147-
--
research/stage4_math_and_design/M2/THEOREM_M2.md-174-| H | +0.01808982 | +0.02178504 | +0.00023216 | −0.00000148 | −0.00392590 | −0.00369522 |
research/stage4_math_and_design/M2/THEOREM_M2.md-175-
research/stage4_math_and_design/M2/THEOREM_M2.md:176:**G1.** For phase, the five-term path is \(0.92563349+0.05975673+0.00036415+0.00000192+0.45947227=1.44522855\). The zero path is \(0.92563349+0.09630010+0.00048790+0.00000197+0.45845510=1.48087856\). Scalar mismatch accounts for 102.51% of the locked lead along this path. Before calibration or projection, marginal KLs are \(0.00008076\) and \(0.00035846\), respectively. Raw phase and zero empirical-code NIGs are −0.12794 and −0.20533. The S17 direct reference scores 64.62% versus 23.80% and 19.41%. Phase's lower ECE is consistent with the confidence mismatch. It does not measure the share of the NLL lead. Using the argmax set, the empirical resolutions are 0.01162 and 0.02104 nats for phase and zero. Using the exact score vector, both have 0.46066 nats of in-sample resolution because both distinguish the signed codes. Those scores can encode information that the decoded predictions use poorly. With sign probabilities marginalized conditional on code, the NLL cost is 0.007704 for phase and 0.002672 for zero. The difference, 0.005032 nats, is sensitivity to that specific sign-discarding operation. It is distinct from Bayes MI and from the factorial intervention.
research/stage4_math_and_design/M2/THEOREM_M2.md-177-
research/stage4_math_and_design/M2/THEOREM_M2.md-178-**G2 and G4.** Raw zero accuracy leads by 2.485 and 1.671 pp. Finite \(T^*\) preserves those rankings and a fitted loss lead of 0.02772 and 0.02635 nats remains. The reverse-conditional term gives zero advantages of 0.03104 and 0.03081 nats after projection and fitting. The global prior and scalar terms are small. Raw zero's coarsened argmax resolution also exceeds phase's by 0.01268 and 0.01877 nats. Projection actually reverses the top-1 ordering in those cells while leaving zero with lower probabilistic loss. Thus the locked reversal is explained at the score level by conditional probability allocation and raw decision ordering. The five saved conditions do not establish an architectural transition at \(\pi/2\). Check the angle effect by holding \(|\omega|\), horizons, and diffusion fixed or by crossing them prospectively with common noise and more independent seeds.
--
research/stage4_math_and_design/M2/THEOREM_M2.md-180-### Q3 scalar recalibration
research/stage4_math_and_design/M2/THEOREM_M2.md-181-
research/stage4_math_and_design/M2/THEOREM_M2.md:182:Each cell gives locked NLL / ECE and retrospective seedwise \(T^*\) NLL / ECE. Temperature triples follow seeds 202611, 202622, 202633.
research/stage4_math_and_design/M2/THEOREM_M2.md-183-
research/stage4_math_and_design/M2/THEOREM_M2.md:184:| Cell | Arm | Locked NLL / ECE | Fitted NLL / ECE | \(T^*\) |
research/stage4_math_and_design/M2/THEOREM_M2.md-185-|---|---|---|---|---|
research/stage4_math_and_design/M2/THEOREM_M2.md-186-| G1 | phase | 1.445229 / .118604 | 1.385472 / .025608 | ∞, 10.144, 5.720 |
--
research/stage4_math_and_design/M2/THEOREM_M2.md-212-| H | zero | 23.17 / 1.435077 | 21.63 / 1.395866 | 21.63 / 1.399059 |
research/stage4_math_and_design/M2/THEOREM_M2.md-213-
research/stage4_math_and_design/M2/THEOREM_M2.md:214:On G1 phase the arithmetic pool makes 8.76% of decisions better and 15.51% worse, the exact weighted hit changes here. Geometric makes 8.28% better and 15.72% worse. G1 phase F0 scores 20.88% / 1.411163. On complete orbits G1 phase's arithmetic gain is 0.03386703, while the reverse-barycenter KL is 0.03430077. For zero the corresponding numbers are 0.04120584 and 0.03488442. Restricting to the three saved rotations adds 0.00004913 and 0.00055046 to the loss gain. Across the tested cells pooled maximum equivariance and idempotence residuals were at floating-point precision. G1 raw maximum probability discrepancies were 0.48167 and 0.41957 for phase and zero. Structural compliance of the offline tables coexists with poor decoding and poor ECE.
research/stage4_math_and_design/M2/THEOREM_M2.md-215-
research/stage4_math_and_design/M2/THEOREM_M2.md-216-### Work F mechanism and averaging convention
--
research/stage4_math_and_design/M2/THEOREM_M2.md-236-| fresh resample | .663189 | 1.405640 / 1.406397 | 1.385318 / 1.382645 | +.003429 | +.000757 |
research/stage4_math_and_design/M2/THEOREM_M2.md-237-
research/stage4_math_and_design/M2/THEOREM_M2.md:238:For valid, the fitted residual phase lead \(+0.00159367\) is \(+0.00028827\) reference-prior KL, \(+0.00158510\) reverse KL, and \(−0.00027971\) realized-reference label residual. For fresh resample, the remaining lead reverses to \(−0.00267249\). The Work A reference gains 0.156018 from fresh vs valid, while the phase connector is 0.016757 worse at locked \(T\). Q2 realized NIG is undefined for the uniform Work A controls because the denominator is zero. All Q2 controls and arms have complete numerical paths, ECE, Brier, scalar temperatures, and soft reference diagnostics in the JSON.
research/stage4_math_and_design/M2/THEOREM_M2.md-239-
research/stage4_math_and_design/M2/THEOREM_M2.md-240-
--
research/stage4_math_and_design/M2/THEOREM_M2.md-271-| NLL | .01667 / .01716 | .01280 / .01277 | .40810 / .40951 |
research/stage4_math_and_design/M2/THEOREM_M2.md-272-
research/stage4_math_and_design/M2/THEOREM_M2.md:273:For random-seed G1 power the corresponding analytic/Monte Carlo pairs are .22385/.22447 and .30028/.30242. The correlated Holm-IUT MC false-claim family rates range from 0.04002 to 0.04953 under the three simulated null configurations. The proof establishes protection beyond those simulations. ECE and the remaining performance gates need their own pilot variability or a joint design simulation.
research/stage4_math_and_design/M2/THEOREM_M2.md-274-
research/stage4_math_and_design/M2/THEOREM_M2.md-275-G1's same-mother Work F fixed-seed 95% block intervals for the all-rotation interventions are \([0.002804,0.003659]\) for text-only flip and \([0.016552,0.017927]\) for slot-only flip. The seed-wise slot effects are \(-0.000641,+0.035214,+0.017147\). A pooled interval over mothers cannot establish a stable retraining effect.
--
research/stage4_math_and_design/M2/THEOREM_M2.md-278-
research/stage4_math_and_design/M2/THEOREM_M2.md-279-- **M1** should lock matched-text phase, text-only, joint-blind and independently permuted phase-slot controls, with a signed-slot intervention at fixed text and matched waveform. Cross \(|\omega|\) and accumulated angle independently where the simulator allows it, preserve the analytic S17 and full-IQ anchors, and keep sign/rotation/mother identifiers together for inference.
research/stage4_math_and_design/M2/THEOREM_M2.md:280:- **M3** should score the native equivariant forecast against the raw, arithmetic, geometric and canonical-representative paths on a common development split. Calibrate on development data that span the prospective regimes and freeze the choice before blind scoring. Lock all absolute endpoints before comparing phase and zero. The G1 raw and pooled paths fail absolute accuracy, and the blind-fitted paths have below-chance accuracy and negligible NIG.
research/stage4_math_and_design/M2/THEOREM_M2.md-281-- **M4** should keep the 131-block synthetic budget until independent gate-endpoint pilots justify changes. If claiming robustness across retrains, plan around ten or more paired seeds as a G1-based starting point and recalculate using pilot upper variance bounds. With three seeds, scope the main claim to the frozen connectors. Obtain separate Pluto capture and CTTC session variance pilots and freeze session IDs and permutation strata. Preserve all nine CTTC sessions in inference, and use the power objective for the complete gate before locking the real-data sample size.
research/stage4_math_and_design/M2/THEOREM_M2.md-282-
--
research/stage4_math_and_design/M6/THEOREM_M6.md-1-# M6 Round 1 • CPU certificates and a capped Stage 4 experiment
research/stage4_math_and_design/M6/THEOREM_M6.md-2-
research/stage4_math_and_design/M6/THEOREM_M6.md:3:**Decision.** Verify the evidence contract and run the physical readout before training the connector. Make rotation and reference handling part of the interface. Use the actual frozen host's candidate-score Jacobian to scope a local adaptation claim. Buy one bounded pilot only after these checks pass. Use independent blind data for the final decision. The proposed Stage 4 compute cap is **\$9.50**, leaving **\$3.38** of the historically reported \$12.88. Round 1 used local CPU only.
research/stage4_math_and_design/M6/THEOREM_M6.md-4-
research/stage4_math_and_design/M6/THEOREM_M6.md-5-## 1. The exact population loss ladder
--
research/stage4_math_and_design/M6/THEOREM_M6.md-60-| Fresh resample | 74.219% / .663189 | 26.432% / 1.405640 | 25.781% / 1.406397 | .742451 | .743208 |
research/stage4_math_and_design/M6/THEOREM_M6.md-61-
research/stage4_math_and_design/M6/THEOREM_M6.md:62:\* Duplicate-raw uses the existing deduplicated Work A row as its information-equivalent physical reference. Q2 contains 512 mothers at (+0.35), without a paired Full-IQ score in the NPZ. Work A is a cache/physical reference. The host receives s17 window codes and 12-dimensional continuous reference features. Fresh evidence improves the Work A NLL by **.156017** and worsens phase NLL by **.016758**. The raw duplicate worsens phase accuracy by **2.474 pp** (mean TV **.040377**), while the dedup arm has zero TV. Two tokens for one acquisition and tokens for two genuine acquisitions test different physics. Both the actual length change and the added attention content can matter. A matched padding/mask experiment is needed to identify their separate effects.
research/stage4_math_and_design/M6/THEOREM_M6.md-63-
research/stage4_math_and_design/M6/THEOREM_M6.md:64:Historical scope. `PROTOCOL_LOCK.json` records six frozen-host checkpoints, 8 epochs, 160 steps per epoch, batch 32, microbatch 16 and dev-fit temperatures. The Work E variance scenario used $q=.35,\rho=.01$, mother NLL SD .5, pilot block SDs .02104014/.01928086, and planned 131 blocks for the joint endpoint gate. Stage 3D's report orders smoke and scorer diagnostics, six trains (2,884–3,218 s each), independent dev variance lock, then blind scoring. The **\$17.12** receipt includes smoke, one OOM/benchmark, all six trains, dev and blind evaluation, and the aborted historical Ministral check. The archival charge was **\$0.00** after allowances. The 150,912 G1 rows are logical predictions per arm from a verified finite table, not 150,912 distinct forwards.
research/stage4_math_and_design/M6/THEOREM_M6.md-65-
research/stage4_math_and_design/M6/THEOREM_M6.md-66-## 2. Tier 0 • analytic rejection before any host work
--
research/stage4_math_and_design/M6/THEOREM_M6.md-68-### 2.1 Bayes-risk brackets for each evidence state
research/stage4_math_and_design/M6/THEOREM_M6.md-69-
research/stage4_math_and_design/M6/THEOREM_M6.md:70:Set $S=(Z,C,\mathcal R)$, with causal metadata (C) and unique allowed references $\mathcal R$. Under a specified kernel, evaluate (p(a,s)\) by summing or integrating the latent state, then compute $H(A\mid S)=-\sum_{a,s}p(a,s)\log p(a\mid s)$ or its integral. For Q3, tabulate by sign/delay/cache pair and compare the same $D$ in the unrestricted physical solver. For Q2, repeat with none/valid/fresh and counterfactual visibility. Integrate correlated references jointly conditional on latent trajectory. A distinct second acquisition need only be independent conditional on the latent state for multiplication of its likelihood. One physical sample duplicated has one likelihood factor. The entropy drop on reveal is exactly $I(A;R_{\rm new}\mid S)\ge0$ under a common nested law. It is an expectation. Sample-level losses can increase after a valid observation.
research/stage4_math_and_design/M6/THEOREM_M6.md-71-
research/stage4_math_and_design/M6/THEOREM_M6.md-72-For Pluto and CTTC, a causal physical $q_S$ fitted off-fold provides $H(A\mid S)\le R(q_S)$, hence $I(A;S)\ge H(A)-R(q_S)$. This is a one-sided variational bound on MI. Poor learned/table scores alone leave the opposite bound unknown. Obtain a lower risk bound by one of the following valid routes.
research/stage4_math_and_design/M6/THEOREM_M6.md-73-
research/stage4_math_and_design/M6/THEOREM_M6.md-74-- Use a certified physical observation model with $S=f(Y,D)$, and a lower bound on $H(A\mid Y,D)$. Alternatively add independent-known oracle information (O) to form the lower bound $H(A\mid S,O)$. Both require actual posterior evaluation or a bound on its error.
research/stage4_math_and_design/M6/THEOREM_M6.md:75:- On finite locked support, form a simultaneous confidence set $\mathcal P$ for the true joint law of $(A,S)$, then use $\inf_{p\in\mathcal P}H_p(A\mid S)$ and $\sup_{p\in\mathcal P}H_p(A\mid S)$. For i.i.d. units, a Bonferroni union of exact binomial intervals for fixed atom probabilities, intersected with the simplex, is one valid construction. For independent clusters, use bounded cluster contributions to each atom and a simultaneous concentration bound. If the support is large the interval may be vacuous. Binning a continuous reference gives an entropy **upper** bound on the richer interface. A lower bound for the binned interface applies when the bins are what the host receives.
research/stage4_math_and_design/M6/THEOREM_M6.md-76-- A collision class with fixed $S=s$ and at least masses $a,b>0$ of two distinct labels contributes at least $(a+b)h_2(a/(a+b))$ to conditional entropy, where $h_2(u)=-u\log u-(1-u)\log(1-u)$. Restrict to the two-label subevent and condition on its indicator. Exact equivalence of two hypotheses with equal prior forces at least $\log 2$ nats and error at least $1/2$ within that subproblem. If their distributions are only close, equal-prior optimal binary error is $\tfrac12(1-\mathrm{TV}(P_0,P_1))$, from the likelihood-ratio decision rule. This yields a CPU impossibility bound when TV can be bounded above.
research/stage4_math_and_design/M6/THEOREM_M6.md-77-
research/stage4_math_and_design/M6/THEOREM_M6.md:78:**Proposed decision margins.** On development folds, lock an admissible NLL target $r_*=R(q_{\rm IQ})+.05$ nats and a useful mean reveal improvement of .02 nats against the no-reveal control. Use each contrast at fixed context and comparable label distributions, with simultaneous bounds for multiple gate decisions. A certified $L_S>r_*$ ends the current representation branch. With entropy brackets $[L_0,U_0]$ and $[L_1,U_1]$ before and after reveal, $\max(0,L_0-U_1)\le I(A;R_{\rm new}\mid S)\le U_0-L_1$ for simultaneous valid brackets. An upper bound below .02 ends a **Bayes-gain** claim for that reveal contract. A direct readout whose paired upper regret bound is within .05 and whose reveal lower improvement bound exceeds .02 is a practical existence witness. Otherwise report uncertainty, improve the physical kernel or support, and defer GPU. At final release require independent upper confidence bounds within .05 of both the permitted Full-IQ and direct-cache comparators on the primary cell and a lower confidence bound above .02 for host reveal improvement where a gain was declared. These margins are Round 1 proposals for registration.
research/stage4_math_and_design/M6/THEOREM_M6.md-79-
research/stage4_math_and_design/M6/THEOREM_M6.md-80-### 2.2 Exact group action and reference algebra
--
research/stage4_math_and_design/M6/THEOREM_M6.md-94-### 3.1 Rank, curvature and optimization certificates
research/stage4_math_and_design/M6/THEOREM_M6.md-95-
research/stage4_math_and_design/M6/THEOREM_M6.md:96:For development example (i), flatten the fixed-length embeddings $e_i=\mathrm{vec}(E_\theta(x_i))\in\mathbb R^{Kd_{\rm LM}}$, define centered host scores $F_i(e_i)$, and compute the actual candidate Jacobian $J_i=D_eF_i(e_i^0)$. For multi-token candidates differentiate the full sum $\sum_t\log p(w_t\mid w_{<t},e_i)$, including gradients of the log normalizer and the cached context. Use candidate-vector JVPs or VJPs in FP32. With a shared adapter perturbation $\delta$, stack $A_i=J_iD_\theta e_i$ and $d_i=\Pi\log p^*_i-F_i(e_i^0)$ after weighting examples by their locked masses. For the stack (A,d), a frozen tangent can achieve residual $\epsilon$ iff $\inf_{\|\delta\|\le r}\|A\delta-d\|\le\epsilon$; without the norm cap the projection residual is $\|(I-AA^+)d\|$. A rank threshold is necessary only for a declared target subspace. A one-dimensional joint task can succeed through a rank-one interface. A shared adapter may require stacked rank across inputs even if all individual $J_i$ have rank (m-1). Fixed thetas, mask patterns and prompts define which local claim is tested.
research/stage4_math_and_design/M6/THEOREM_M6.md-97-
research/stage4_math_and_design/M6/THEOREM_M6.md-98-If the stacked score map has Hessian operator norm bounded by (M) over the trust region, Taylor's theorem yields an unavoidable residual ≥ $\|(I-AA^+)d\|-Mr^2/2\$, and the gain cannot exceed $\|A\|r+Mr^2/2$. If $d\in\operatorname{im}A$, smallest retained singular value (s>0) and $\|d\|/s\le r$, select $\delta=A^+d$. The nonlinear residual is at most $M\|d\|^2/(2s^2)$. With softmax, the log-sum-exp Hessian is $\mathrm{diag}(p)-pp^\top$, has norm ≤1, and has curvature ≥(p_{\min}\) on centered vectors when all probabilities on the segment are ≥(p_{\min}\). Taylor's integral gives $p_{\min}\|e\|^2/2\le\mathrm{KL}(p^*\|q)\le\|e\|^2/2$ for centered logit error (e) with positive matched target. Handle absent sectors through a registered probability floor or a separate support analysis. This converts a certified local rank/curvature inequality into an NLL gate. A finite-difference estimate of (M) is diagnostic until justified as an upper bound. With ordinary training outside the certified trust region, the rejection certificate applies only to the local model.
research/stage4_math_and_design/M6/THEOREM_M6.md-99-
research/stage4_math_and_design/M6/THEOREM_M6.md:100:For least squares in this tangent model, $K=AA^\top$. Gradient flow decays each representable residual eigenmode as $e^{-t\lambda}$; discrete gradient descent multiplies it by $1-\eta\lambda$ for $0<\eta<2/\lambda_{\max}$. The nullspace remains. With fixed train/test Jacobians the linearized test prediction update for minimum-norm interpolation is $A_{\rm test}A_{\rm train}^+d_{\rm train}$. Compare to physical contrasts on a held-out joint CFO/angle grid. A poor result isolates this fixed-tangent extrapolator. A short actual-host test should use locked prompt positions, candidate tokenization, phase signs, matched lengths, and representative physical margins. Use analytic contrast and small CPU probes for the <30 s tier. Full 4B, 9B or 27B Jacobian timings and memory depend on the pinned host and are unmeasured here. A partial host or quantized CPU copy can reject an interface on a structural property shared with the target, or give a heuristic diagnostic. A quantitative negative result transfers only with an error bound on the difference between their operators. [Jacot, Gabriel and Hongler, 2018](https://papers.neurips.cc/paper_files/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html)
research/stage4_math_and_design/M6/THEOREM_M6.md-101-
research/stage4_math_and_design/M6/THEOREM_M6.md-102-### 3.2 Geometry of $W_{\rm out}$ and vocabulary readout
--
research/stage4_math_and_design/M6/THEOREM_M6.md-110-The constant direction vanishes and the radial multiplier before learned gains is $\varepsilon/s^3$. On the slice $\{\mathbf1,v\}^\perp$, singular scales lie between $\min|\gamma|/s$ and $\max|\gamma|/s$. RMSNorm uses $P=I,v=x$, retaining the radial attenuation. Check normalization at the projector and inside the host. If a residual map is $x\mapsto x+f_i(x)$ and $\|Df_i\|\le\eta_i<1$, the full-state residual Jacobian scales lie in $[1-\eta_i,1+\eta_i]$ by the triangle inequalities. Products give bounds across a sequence of such maps. They alone do not control routing from the injected slots to the answer token. With no routing, answer Jacobians can vanish even when all residual maps are invertible.
research/stage4_math_and_design/M6/THEOREM_M6.md-111-
research/stage4_math_and_design/M6/THEOREM_M6.md:112:For single distinct answer tokens, let $C_{\rm cand}$ select vocabulary rows and $E_{\rm vocab}\$ be the **actual unembedding**. If $D_{\rm answer}$ is the full derivative from injected slots to the relevant normalized answer hidden state, the effective matrix is $M_{\rm cand}=\Pi C_{\rm cand}E_{\rm vocab}D_{\rm answer}B$. For multi-token answers use $J_iB$ instead. A sufficient condition for preserving every evidence contrast at linear order is $r\le m-1,\ \sigma_{\min}(M_{\rm cand})\ge s_*>0,\ \sigma_{\max}(M_{\rm cand})\le S_*$ on that subspace. In an admissible tangent basis (U), if $J_iU=L\Sigma V^\top$ has $r$ useful singular values, choose $B=UV_r\Sigma_r^{-1}\beta$. Then $J_iB=\beta L_r$ is isotropic **in candidate-score space**, and $\|B\|\le\beta/\sigma_r(J_iU)$. Constrain embedding energy and Jacobian error. One shared (B) needs these bounds across the declared prompts/positions or validated uniform stack constraints, not just one anchor. Full rank of $W_{\rm out}$ can hold while (B) lies in the nullspace of the candidate readout.
research/stage4_math_and_design/M6/THEOREM_M6.md-113-
research/stage4_math_and_design/M6/THEOREM_M6.md-114-For intended class (a) against rival (b), verify baseline centered margin $m^0_{ab}$ and compare $m^0_{ab}+(e_a-e_b)^\top J_iBu$. A positive margin after subtracting $\sqrt2\$ times the Taylor/rounding error bound guarantees local preservation of that decision. Upper-bound logit excursion as well to prevent saturation, preserving the curvature and calibrated probability budget. A large singular value from a generic vocabulary analysis can correspond to directions orthogonal to all answer contrasts. Freeze tokenization and check candidate lengths, boundaries, order, truth-score parity and the reference-count contract before trusting this geometry.
--
research/stage4_math_and_design/M6/THEOREM_M6.md-124-$$
research/stage4_math_and_design/M6/THEOREM_M6.md-125-
research/stage4_math_and_design/M6/THEOREM_M6.md:126:This is optimal for specified priors, utilities, transition likelihoods and caps. Greedy information per dollar is at most a heuristic when checks are complementary. In a differentiable separable allocation with concave information utilities $v_j(n_j)$, constant per-unit prices $c_j$ and a binding budget, KKT gives $v'_j(n_j)/c_j=\lambda$ for interior arms, with only arms above this price at the origin receiving effort. For $v_j=-w_j\sigma_j^2/n_j$, the allocation is $n_j\propto\sqrt{w_j\sigma_j^2/c_j}$, with independent clusters and a fixed variance estimand. Use mandatory cohort minima and integer reallocation, especially for nine CTTC sessions. No universal zero-waste guarantee follows without a specified law. In a nonidentifiable case the valuable output is the collision and the minimum extra observation, even when no GPU trial follows.
research/stage4_math_and_design/M6/THEOREM_M6.md-127-
research/stage4_math_and_design/M6/THEOREM_M6.md-128-### 4.2 Two-look design and what the Monte Carlo verifies
--
research/stage4_math_and_design/M6/THEOREM_M6.md-150-Sequential exact-binomial MC intervals are **[2.716%, 4.671%]** on the boundary-null stratum and **[92.227%, 94.230%]** on the working stratum. The modeled 90% power needs drift **≥3.267657** at .99 CPU sensitivity. At .95 sensitivity the same working stratum drops below 90% power. If every candidate is working, the analytic saving drops to about **4.27%**. These are selection and effect-size sensitivities, not a transfer claim. Savings come from declining expensive branches, not more favorable final testing.
research/stage4_math_and_design/M6/THEOREM_M6.md-151-
research/stage4_math_and_design/M6/THEOREM_M6.md:152:**Actual Stage 4 release.** Use the pilot primarily for engineering, variance, and predictive futility on disjoint development mothers, sessions or blocks. Freeze pilot/final coupling and calibrate on external development simulations before using the two-look efficacy boundaries. If that law cannot be justified, allow only futility at the first look and apply a single valid, independent final test on blind blocks after all three seeds and scoring settings are fixed. Use new synthetic mother namespaces, disjoint from the Stage 3D diagnostic archives, and hold out all physical records used for the claimed final test. Conditional on the pilot and model selection, the blind test must be valid for the declared null. Its unconditional false-promotion rate is at most its level by iterated expectation. Power after futility, particularly with seed-to-seed heterogeneity, still needs development calibration. Full release also requires passing physical margins and negative controls. Count mothers or recording clusters as independent units under the registered assumptions. Use randomization inference where the experimental randomization supports it, or a declared block model with sensitivity to correlation. For Pluto, 16 temporal blocks support inference within the capture when justified. For CTTC, report 72 nested blocks and session-level (9-unit) sensitivity. A sign-flip test is exact for justified randomization or a stated joint sign-symmetry null. Mean-zero shifts alone supply neither condition.
research/stage4_math_and_design/M6/THEOREM_M6.md-153-
research/stage4_math_and_design/M6/THEOREM_M6.md-154-### 4.3 Stage 4 Modal plan in US dollars
--
research/stage4_math_and_design/M6/THEOREM_M6.md-172-Make dataset acquisition, weight pulls, cache/table computation, causal normalization, provenance linting, calibration folds, and source guard checks CPU jobs. Mount prepared read-only dataset features for evaluation and commit isolated outputs; local CPU reduces cloud seconds where practical. The three synthetic seeds train the one surviving Equivariant+RefPool design. Historical phase/zero archives, physical baselines and exact intervention controls give context. This envelope does not buy six full retrained comparator checkpoints. Any newly trained matched-baseline arm is a separate priced decision. Real evaluation uses those same frozen synthetic-trained weights and frozen candidate schema. If real development data are used to fit preprocessing or a temperature, freeze parameters by source fold and report strictly zero-shot calibration separately where claimed.
research/stage4_math_and_design/M6/THEOREM_M6.md-173-
research/stage4_math_and_design/M6/THEOREM_M6.md:174:At the historical 1.9757 s/step the 2,707 continuation steps would take ~5,348 s before overhead. Require a receipt within 2.30 s/step, leaving ~524 s for all continuation overhead at that worst allowed rate. The new preprocessing, fused kernels and longest Q2 inputs can change this. At most 6,144 Q3 prefixes cover 4 cells × 2 signs × 256 exact keys × 3 seeds, provided the key includes all prefix state, dtype and batch policy. Q2's seven evidence arms require at most 10,752 prefixes for 512 mothers × 3 seeds. The 1,050 s synthetic scoring line therefore requires 16.1 prefixes/s before overhead. Cache prompts only across identical embeddings, masks, positions, template, candidate scope and attention state. Distinguish KV/linear-state sharing from lookup deduplication. Fix the padding and batch policy and require per-probability and paired endpoint parity on a receipt set before any blind inference. Historical BF16 differences across batch settings reached .067 in probabilities with per-cell argmax agreement as low as .6875. Historical A100 b64 peaked at 45,410 MB on inference, so use smaller fixed batches or prevalidated kernels on 24 GB cards. Keep precision locked across comparable arms.
research/stage4_math_and_design/M6/THEOREM_M6.md-175-
research/stage4_math_and_design/M6/THEOREM_M6.md:176:The real rows cap at one query per packet/frame for up to seven evidence arms × three seeds. Without dedup, Pluto requires 47,040 prefixes (≥20.46/s within 2,300 s) and CTTC 30,240 (≥17.79/s within 1,700 s). Reserve startup separately within each row before setting a required steady rate. These rates, and VRAM on L4/A10G, require a parity-and-throughput witness under the actual workload. Use part of the relevant row for an unblinded developer smoke. Select the lowest-cost parity-passing device whose worst-case authorized count fits. Recompute cost after any failed receipt. Substitution allowance covers price difference at equal seconds, while a first device's already billed time still counts. A100 fallback is possible only for a separately metered subset within remaining cost/time. If one resource proof fails, preserve the CPU-only evidence and mark that GPU row incomplete. Neither the time cap nor the historical A100 rate mathematically guarantees real workload completion. Keep per-invocation deadlines, minimal scale-down, charge for billed teardown and any retries, and precharge each next shard at its bounded worst-case dollars against a single serial ledger. Account for already running and settling time before dispatch. Modal dashboard budget enforcement can lag and availability varies by plan. [Volume staging](https://modal.com/docs/guide/volumes), [Modal budgets](https://modal.com/docs/guide/budgets)
research/stage4_math_and_design/M6/THEOREM_M6.md-177-
research/stage4_math_and_design/M6/THEOREM_M6.md-178-## 5. Stage 4 → 6 critical path
--
research/stage4_math_and_design/M6/THEOREM_M6.md-201-| 4B cache | Compute synthetic exact/controlled likelihood and causal real-data bounds on dev and support. Pass with the existence witness in §2.1. For a certified insufficiency, test an equal-budget canonical complex cache with identical observable inputs. Stop the branch if both cannot meet the objective. |
research/stage4_math_and_design/M6/THEOREM_M6.md-202-| 4C interface | Enumerate $C_4^2$ on finite cache codes and induced labels, check identity bytes for duplicate, masked, permuted and padded records, and verify genuine fresh effects can survive pooling. Require OOD mixed contrasts and a sufficient singular/margin witness for the design subspace. Check dev Jacobian stack rank before investing in a wider connector. |
research/stage4_math_and_design/M6/THEOREM_M6.md:203:| 4D promotion | CPU freeze the splits, paired estimand, useful margins, tolerances, calibration rule, alpha and maximum allocation. The GPU pilot verifies the timing/VRAM/parity contract and the dev continuation criterion. Release only after independent G1 primary, G2–G4 strata, Q2_C4, three seeds and exact controls. If eligibility fails, archive either the certificate or the prespecified inconclusive outcome. |
research/stage4_math_and_design/M6/THEOREM_M6.md-204-| 4E–4F real | Same locked candidate schema and CPU score oracle on each cohort, the cached sufficient-statistic check, paired source/block plan, valid-vs-none/fresh sign, and independent negative controls. Preserve raw vs official filtered CTTC as views of one source. Log natural vs manipulated cases, evidence location and retained/revised/incorrect counts. A validated explanation of a source-specific negative result can feed 5A as an identifiability constraint. |
research/stage4_math_and_design/M6/THEOREM_M6.md-205-| 5A–5B event/trajectory | Use matched marginals for smooth drift, discrete jump and independent processes. Enumerate equivalence classes and compare likelihoods, bounding TV/LR for competing trajectories. A pair with the same observable law cannot be localized more finely than its indistinguishable interval without new evidence. Freeze interval coverage/resolution, causal evidence spans, proper scores and real held-out-pilot contracts. Synthetic interventions can validate jump recovery. Use natural truth only where externally established. |
research/stage4_math_and_design/M6/THEOREM_M6.md-206-| 6A active read | With history $D_t$ and candidate window $W_j$, Bayes expected log-loss drop is $I(A;W_j\mid D_t)$. Verify the exact-small-case policy through budgeted dynamic programming using conditional revealed data. Skip zero-information reads when they have positive cost. Compare fixed, random, local and global content selection at identical read counts, acquisition permissions, cache bits and context. Gate monotone posterior accounting and cite concrete time spans with uncertainty or abstention. |
research/stage4_math_and_design/M6/THEOREM_M6.md:207:| 6B–6C 4B → 9B → 27B | Freeze the waveform representation, evidence semantics, tasks and read budgets. Pin each host/tokenizer/inference implementation; rerun candidate Jacobian and proof obligations separately, and size memory/throughput before paid compute. Fit only a host-specific adapter on the matched train budget. 4B→9B can inform same-family scale; 9B→3.8-27B also changes family revision and cannot isolate parameter count. Include paired ordinary text/image regression as in the integration plan. Separate follow-on budgets apply. [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B), [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) |
research/stage4_math_and_design/M6/THEOREM_M6.md-208-| 6D LEMMO | CPU lock runnable revision, permissions, train overlap, questions, answer normalization and common-supported endpoints. The LEMMO paper uses Qwen3.5-4B with LoRA and, depending on task, Q16/Q128 or a direct projector. Report the native systems under documented inputs. For causal attribution, also match base revision, physical observations and reveals, sampling and timing, cache/context bits, readable windows, quantization, training data, trainable parameter and optimizer budgets where feasible. Record reruns and latency. If an identical matched run cannot be made, narrow the comparison to the observed common support. [LEMMO paper §3–4 and Appendix B.4](https://arxiv.org/html/2611.09476v1), [LEMMO repository](https://github.com/N1rv4n4/LEMMO) |
research/stage4_math_and_design/M6/THEOREM_M6.md-209-| 6E ultimate blind | CPU seal independent physical recordings and unseen question combinations. Verify time/evidence alignment, cluster count and attainable precision, multi-turn reveal provenance, baselines and matched scoring. Run once under a locked ledger and keep unverifiable tasks as explicit abstentions. |
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
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$p=Get-Content research/stage4_math_and_design/M5/THEOREM_M5.md -Encoding UTF8; $p[109..146]; rg -n '"'encode|pack|quant' research/stage4_math_and_design/M5/verify_m5.py"
```

**Aggregated Output:**

```text
**What s17 permanently merges.** Opposite means have likelihood ratio $\exp(4\Re[T^*\alpha]/v)$. Two interior points of one s17 cell with distinct projection on $\alpha$ give different likelihood ratios. A strict neural assignment margin ensures an open neighborhood around each point inside the same cell. Gaussian positivity and strict propriety then imply positive Bayes loss when a query separates these posteriors, including an informative phase pilot under the conditions of §1.2. The experiment verifies such a collision with the pinned weights. The same argument holds when an event or forecast contrast varies within a cell. In the variance experiment a circular pure-noise window has a uniform 2-bit turn at every variance, whereas radial or residual energy gives positive class information. s17 can retain some quality information through its learned index, but cannot preserve a continuously varying likelihood ratio. Bounds for the actual Pluto and CTTC acquisitions require their measured calibration and held-out data. The $-3$ to $+24$ dB and $>10$ dB gain variations in this report are synthetic stress conditions.

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
126:def encode(iq, t, residual, variance, encoder):
127:    original = encoder.encode(iq.reshape(-1, 3)).reshape(t.shape)
132:    packets = {"4_s17": original}
141:        packets[method] = turns if method == "2" else (original << (angle_bits + extra)) | tail
143:            np.testing.assert_array_equal(packets[method] >> (angle_bits + extra), original)
144:        assert np.all((packets[method] >= 0) & (packets[method] < 2**int(method)))
145:        prefix = packets[method] if method == "2" else packets[method] >> (angle_bits + extra + 2)
146:        unpacked = (
147:            (prefix << angle_bits) | ((packets[method] >> extra) & (2**angle_bits - 1)),
148:            (packets[method] >> quality_bits) & (2**radial_bits - 1),
149:            packets[method] & (2**quality_bits - 1),
151:        np.testing.assert_array_equal(unpacked, bins[method])
152:        bins[method] = unpacked
153:    return packets, bins
156:def s17_emissions(encoder, rng, amplitude, variance):
161:        counts = np.bincount(encoder.encode(y), minlength=16)
167:    """Integrate the complex Gaussian probability of each packed polar cell."""
198:def window_log_likelihood(method, t, residual, packets, bins, a, v, encoder, key):
215:                    emissions[condition] = s17_emissions(encoder, rng, *condition)
217:                values[:, window, noise_id] = np.log(emission[:, packets[method][:, window]].T)
434:def run_profile(key, profile, tracks, encoder):
455:    packets, bins = encode(all_iq, t, residual, v, encoder)
474:        ll = window_log_likelihood(method, t, residual, packets, bins, a, v, encoder, key)
526:def collision_witness(encoder):
529:    encoded = encoder.encode(np.repeat(amplitudes[:, None] / np.sqrt(3), 3, axis=1))
531:    for code in np.unique(encoded):
532:        group = amplitudes[encoded == code]
542:    _, soft = forward(canonical, encoder.weights)
543:    assignment = np.sort(soft @ np.log(encoder.centers).T, axis=1)
549:    np.testing.assert_array_equal(encoder.encode((selected[:, None] + perturb).reshape(-1, 3)), best["code"])
553:def legacy_inner_witness(encoder):
561:                codes = encoder.encode(vectors)
570:                            "codes": encoder.encode(pair).tolist(),
579:    encoder = load(checkpoint)
580:    assert encoder.equivariant and len(encoder.centers) == 4
581:    witness = collision_witness(encoder)
582:    inner_witness = legacy_inner_witness(encoder)
586:        profile_result, losses = run_profile(key, profile, tracks, encoder)
615:            "compact_reader": "integrated physical cells; packed legacy inner bits available but ignored",
```

### Event #11: `item.completed` — `command_execution` (`item_7`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$p=Get-Content research/stage4_math_and_design/M6/THEOREM_M6.md -Encoding UTF8; $p[74..142]'
```

**Aggregated Output:**

```text
- On finite locked support, form a simultaneous confidence set $\mathcal P$ for the true joint law of $(A,S)$, then use $\inf_{p\in\mathcal P}H_p(A\mid S)$ and $\sup_{p\in\mathcal P}H_p(A\mid S)$. For i.i.d. units, a Bonferroni union of exact binomial intervals for fixed atom probabilities, intersected with the simplex, is one valid construction. For independent clusters, use bounded cluster contributions to each atom and a simultaneous concentration bound. If the support is large the interval may be vacuous. Binning a continuous reference gives an entropy **upper** bound on the richer interface. A lower bound for the binned interface applies when the bins are what the host receives.
- A collision class with fixed $S=s$ and at least masses $a,b>0$ of two distinct labels contributes at least $(a+b)h_2(a/(a+b))$ to conditional entropy, where $h_2(u)=-u\log u-(1-u)\log(1-u)$. Restrict to the two-label subevent and condition on its indicator. Exact equivalence of two hypotheses with equal prior forces at least $\log 2$ nats and error at least $1/2$ within that subproblem. If their distributions are only close, equal-prior optimal binary error is $\tfrac12(1-\mathrm{TV}(P_0,P_1))$, from the likelihood-ratio decision rule. This yields a CPU impossibility bound when TV can be bounded above.

**Proposed decision margins.** On development folds, lock an admissible NLL target $r_*=R(q_{\rm IQ})+.05$ nats and a useful mean reveal improvement of .02 nats against the no-reveal control. Use each contrast at fixed context and comparable label distributions, with simultaneous bounds for multiple gate decisions. A certified $L_S>r_*$ ends the current representation branch. With entropy brackets $[L_0,U_0]$ and $[L_1,U_1]$ before and after reveal, $\max(0,L_0-U_1)\le I(A;R_{\rm new}\mid S)\le U_0-L_1$ for simultaneous valid brackets. An upper bound below .02 ends a **Bayes-gain** claim for that reveal contract. A direct readout whose paired upper regret bound is within .05 and whose reveal lower improvement bound exceeds .02 is a practical existence witness. Otherwise report uncertainty, improve the physical kernel or support, and defer GPU. At final release require independent upper confidence bounds within .05 of both the permitted Full-IQ and direct-cache comparators on the primary cell and a lower confidence bound above .02 for host reveal improvement where a gain was declared. These margins are Round 1 proposals for registration.

### 2.2 Exact group action and reference algebra

Derive the action from the intervention before defining the model. In synthetic Q3, for $g=(r_1,r_2)\in C_4^2$, coordinate relabeling of cache windows has $z_i\mapsto z_i+4r_i\bmod16$ and answer transport $P_g=\mathrm{roll}(r_1)$ for the repeat target. Rotate reference coordinates consistently under the experiment's physical action. A coordinate change leaves CFO unchanged. A new physical phase jump or fresh measurement can change the answer and must use a separate intervention. For actions with $P(A\mid gx)=P_gP(A\mid x)$, require $q(gx)=P_gq(x)$. At fixed points require stabilizer invariance. This is an exact contract. On any equal-weight pair with target symmetry and probability defect $d=\mathrm{TV}(q(gx),P_gq(x))$, triangle inequality and Pinsker give mean KL regret at least $d^2/2$. Thus some sufficiently large defects also violate a quantitative loss gate, although a nonzero defect by itself proves failure only of the exact contract.

Define `RefPool` on a **set of acquisition IDs** with authenticated provenance and role masks, before prompt construction. Same ID/same sample reduces to one record, and inconsistent copies of an ID invalidate the input under the locked contract. Each allowed acquisition contributes its conditional likelihood factor once, or enters a joint dependence model once. Excluded IDs and irrelevant padding are removed before aggregation. An independent distractor is Bayes-ignorable only under $A\perp R_d\mid Z,C,\mathcal R_{\rm valid}$, including any conditioning induced by selection. For natural unknown relevance, retain a latent relevance posterior. An oracle `valid` flag is admissible only if shared and available at reveal. Implement a fixed-length pooled representation with typed missing/valid slots, fixed positions and masks, including Qwen's recurrent state semantics. Equal semantic inputs must produce identical bytes for embeddings, positions, masks and prompt. The contract implies token-count invariance to duplicate, masked, or padded records. Genuine evidence must remain capable of changing the posterior.

Two exact ways to enforce output symmetry are (i) score a canonical representative and transport to the original coordinates, projecting on the stabilizer when needed, or (ii) average transported probabilities $q_{\rm eq}(x)=|G|^{-1}\sum_g P_g^{-1}q(gx)$. Reindexing proves covariance in (ii). Jensen gives expected NLL no worse under the full invariant population law. Top-1 accuracy has no analogous Jensen bound. Prefer (i) for fixed inference cost where the action and canonicalization are valid. If the host must itself produce native equivariant answers, a frontend equivariance proof alone is insufficient. Report native and transported scores when both are evaluated. Real hardware phase imbalance or label conventions can break a physical action, so use deliberate coordinate relabeling and score natural and injected perturbations separately.

## 3. Tier 1 • CPU geometry and the frozen host

**Additive obstruction.** For centered scores $f(z,\phi)=a(z)+b(\phi)$, the mixed rectangle $f(z,\phi)-f(z',\phi)-f(z,\phi')+f(z',\phi')$ is identically zero. For complex evidence (u(z)) and phase $\phi=\omega t$, score $\Re[u(z)e^{i\phi}\bar r_a]$ for class (a). The physical rectangle equals $\Re[(u(z)-u(z'))(e^{i\phi}-e^{i\phi'})\bar r_a]$, generically nonzero. At least one corner must have error at least one fourth of the rectangle norm by the triangle inequality. Build $u\cos\phi,u\sin\phi$ into the connector, or transport complex evidence first and apply a joint score. The rotation law then extrapolates for all $\omega t$ where the physical model holds. The verified toy probe fits on $|\omega|\le .25$ and evaluates at $\omega=\pm .40,t=4$ ($|\phi|=1.60>\pi/2$). Additive vs coupled OOD centered-logit RMSE is **.935161 vs .000000000000000738**, with rectangle L2 norm **3.998294** and subsecond execution. A smaller randomized frozen host can test software geometry; it has no guaranteed approximation to Qwen. A flexible nonlinear function plus a smooth bump supported outside the training interval can fit the same train data and give any OOD value. Some inductive bias or explicit mechanistic law is therefore indispensable for certified extrapolation.

**Stage 3D interpretation.** The cache-pair MLP sees $z_1,z_2$; a separate phase MLP sees six trigonometric slots; continuous reference features go through a separate MLP per `<REF>`. Signed CFO also appears as text in both arms. A linear fixed-Jacobian map about a common prompt/fixed reference layout yields separate additive score contributions. Qwen's true nonlinear attention can mix those streams and jointly condition its Jacobian. This makes Stage 3D's additive extrapolation risk diagnosable before training, alongside a provable failure of any required symmetry or idempotence whose implementation lacks the corresponding contract. It does not make the observed **.504103** nats or the causal contribution of each suspected mechanism predictable from source alone. Timing, caching and low-precision instability were additional measured risks. A positive CPU joint-readout result establishes feasibility for that readout.

### 3.1 Rank, curvature and optimization certificates

For development example (i), flatten the fixed-length embeddings $e_i=\mathrm{vec}(E_\theta(x_i))\in\mathbb R^{Kd_{\rm LM}}$, define centered host scores $F_i(e_i)$, and compute the actual candidate Jacobian $J_i=D_eF_i(e_i^0)$. For multi-token candidates differentiate the full sum $\sum_t\log p(w_t\mid w_{<t},e_i)$, including gradients of the log normalizer and the cached context. Use candidate-vector JVPs or VJPs in FP32. With a shared adapter perturbation $\delta$, stack $A_i=J_iD_\theta e_i$ and $d_i=\Pi\log p^*_i-F_i(e_i^0)$ after weighting examples by their locked masses. For the stack (A,d), a frozen tangent can achieve residual $\epsilon$ iff $\inf_{\|\delta\|\le r}\|A\delta-d\|\le\epsilon$; without the norm cap the projection residual is $\|(I-AA^+)d\|$. A rank threshold is necessary only for a declared target subspace. A one-dimensional joint task can succeed through a rank-one interface. A shared adapter may require stacked rank across inputs even if all individual $J_i$ have rank (m-1). Fixed thetas, mask patterns and prompts define which local claim is tested.

If the stacked score map has Hessian operator norm bounded by (M) over the trust region, Taylor's theorem yields an unavoidable residual ≥ $\|(I-AA^+)d\|-Mr^2/2\$, and the gain cannot exceed $\|A\|r+Mr^2/2$. If $d\in\operatorname{im}A$, smallest retained singular value (s>0) and $\|d\|/s\le r$, select $\delta=A^+d$. The nonlinear residual is at most $M\|d\|^2/(2s^2)$. With softmax, the log-sum-exp Hessian is $\mathrm{diag}(p)-pp^\top$, has norm ≤1, and has curvature ≥(p_{\min}\) on centered vectors when all probabilities on the segment are ≥(p_{\min}\). Taylor's integral gives $p_{\min}\|e\|^2/2\le\mathrm{KL}(p^*\|q)\le\|e\|^2/2$ for centered logit error (e) with positive matched target. Handle absent sectors through a registered probability floor or a separate support analysis. This converts a certified local rank/curvature inequality into an NLL gate. A finite-difference estimate of (M) is diagnostic until justified as an upper bound. With ordinary training outside the certified trust region, the rejection certificate applies only to the local model.

For least squares in this tangent model, $K=AA^\top$. Gradient flow decays each representable residual eigenmode as $e^{-t\lambda}$; discrete gradient descent multiplies it by $1-\eta\lambda$ for $0<\eta<2/\lambda_{\max}$. The nullspace remains. With fixed train/test Jacobians the linearized test prediction update for minimum-norm interpolation is $A_{\rm test}A_{\rm train}^+d_{\rm train}$. Compare to physical contrasts on a held-out joint CFO/angle grid. A poor result isolates this fixed-tangent extrapolator. A short actual-host test should use locked prompt positions, candidate tokenization, phase signs, matched lengths, and representative physical margins. Use analytic contrast and small CPU probes for the <30 s tier. Full 4B, 9B or 27B Jacobian timings and memory depend on the pinned host and are unmeasured here. A partial host or quantized CPU copy can reject an interface on a structural property shared with the target, or give a heuristic diagnostic. A quantitative negative result transfers only with an error bound on the difference between their operators. [Jacot, Gabriel and Hongler, 2018](https://papers.neurips.cc/paper_files/paper/2018/hash/5a4be1fa34e62bb8a6ec6b91d2462f5a-Abstract.html)

### 3.2 Geometry of $W_{\rm out}$ and vocabulary readout

Use column vectors and map physical coordinates $u\in\mathbb R^r$ into an embedding perturbation (Bu), where $B=W_{\rm out}^\top Q$ in one slot and (Q) selects the physical subspace in $d_{\rm hid}$. For multiple slots stack the corresponding blocks. $B^\top B=\beta^2I$ gives an isotropic frame in embedding space. It does not guarantee candidate-score alignment. For standard LayerNorm at (x), $v=Px$, $P=I-\mathbf1\mathbf1^\top/d\$, and $s=(\|v\|^2/d+\varepsilon)^{1/2}$, its derivative is

$$
D\mathrm{LN}_x=\mathrm{diag}(\gamma)\big(P/s-vv^\top/(ds^3)\big).
$$

The constant direction vanishes and the radial multiplier before learned gains is $\varepsilon/s^3$. On the slice $\{\mathbf1,v\}^\perp$, singular scales lie between $\min|\gamma|/s$ and $\max|\gamma|/s$. RMSNorm uses $P=I,v=x$, retaining the radial attenuation. Check normalization at the projector and inside the host. If a residual map is $x\mapsto x+f_i(x)$ and $\|Df_i\|\le\eta_i<1$, the full-state residual Jacobian scales lie in $[1-\eta_i,1+\eta_i]$ by the triangle inequalities. Products give bounds across a sequence of such maps. They alone do not control routing from the injected slots to the answer token. With no routing, answer Jacobians can vanish even when all residual maps are invertible.

For single distinct answer tokens, let $C_{\rm cand}$ select vocabulary rows and $E_{\rm vocab}\$ be the **actual unembedding**. If $D_{\rm answer}$ is the full derivative from injected slots to the relevant normalized answer hidden state, the effective matrix is $M_{\rm cand}=\Pi C_{\rm cand}E_{\rm vocab}D_{\rm answer}B$. For multi-token answers use $J_iB$ instead. A sufficient condition for preserving every evidence contrast at linear order is $r\le m-1,\ \sigma_{\min}(M_{\rm cand})\ge s_*>0,\ \sigma_{\max}(M_{\rm cand})\le S_*$ on that subspace. In an admissible tangent basis (U), if $J_iU=L\Sigma V^\top$ has $r$ useful singular values, choose $B=UV_r\Sigma_r^{-1}\beta$. Then $J_iB=\beta L_r$ is isotropic **in candidate-score space**, and $\|B\|\le\beta/\sigma_r(J_iU)$. Constrain embedding energy and Jacobian error. One shared (B) needs these bounds across the declared prompts/positions or validated uniform stack constraints, not just one anchor. Full rank of $W_{\rm out}$ can hold while (B) lies in the nullspace of the candidate readout.

For intended class (a) against rival (b), verify baseline centered margin $m^0_{ab}$ and compare $m^0_{ab}+(e_a-e_b)^\top J_iBu$. A positive margin after subtracting $\sqrt2\$ times the Taylor/rounding error bound guarantees local preservation of that decision. Upper-bound logit excursion as well to prevent saturation, preserving the curvature and calibrated probability budget. A large singular value from a generic vocabulary analysis can correspond to directions orthogonal to all answer contrasts. Freeze tokenization and check candidate lengths, boundaries, order, truth-score parity and the reference-count contract before trusting this geometry.

## 4. Tier 2 • sequential decisions and an exact cost envelope

### 4.1 VoI and continuation

Let $\xi$ encode the true effects, feasibility and dataset transfer, $D_i$ the current evidence, and $b_i$ the remaining authorized budget. With final utility $u(a,\xi)$, define $U(D)=\max_a\mathbb E[u(a,\xi)\mid D]$ over promote, stop or archive, including a cost for erroneous promotion. For a candidate check (j) with new observation $X_j$ and cost $c_j$, one-step $\mathrm{VoI}_j=\mathbb E[U(D,X_j)\mid D]-U(D)$. A check worth less than its priced cost should not be purchased as a one-step action. With dependencies and horizon, the exact finite decision recursion is

$$
V(D,b)=\max\{U(D),\max_{j\;\mathrm{eligible},\,c_j\le b}[-c_j+\mathbb E V(D,X_j,b-c_j)]\}.
$$

This is optimal for specified priors, utilities, transition likelihoods and caps. Greedy information per dollar is at most a heuristic when checks are complementary. In a differentiable separable allocation with concave information utilities $v_j(n_j)$, constant per-unit prices $c_j$ and a binding budget, KKT gives $v'_j(n_j)/c_j=\lambda$ for interior arms, with only arms above this price at the origin receiving effort. For $v_j=-w_j\sigma_j^2/n_j$, the allocation is $n_j\propto\sqrt{w_j\sigma_j^2/c_j}$, with independent clusters and a fixed variance estimand. Use mandatory cohort minima and integer reallocation, especially for nine CTTC sessions. No universal zero-waste guarantee follows without a specified law. In a nonidentifiable case the valuable output is the collision and the minimum extra observation, even when no GPU trial follows.

### 4.2 Two-look design and what the Monte Carlo verifies

Use a preregistered directional **primary** score $\mu>0$ in final-standard-error units above the practical margin. In the canonical independent-increment Gaussian experiment, $Z_1\sim N(\sqrt t\mu,1)$ and $Z_2=\sqrt tZ_1+\sqrt{1-t}V\$, $V\sim N(\sqrt{1-t}\mu,1)$ independent. $t=.18$ is an information fraction. A one-seed pilot at 15–20% of optimizer steps does not, by itself, have this information fraction or this joint distribution.

With one-sided alpha $\alpha=.05$, use O'Brien–Fleming-style spending $a(t)=\bar\Phi(z_{1-\alpha}/\sqrt t)$ or predeclare Pocock-style $a(t)=\alpha\log[1+(e-1)t]$ for more early alpha. For the former, $a(.18)=.00005288548$, $b_1=3.876957$; solve $a(t)+\int_{-\infty}^{b_1}\varphi(z)\bar\Phi((b_2-\sqrt tz)/\sqrt{1-t})\,dz=.05$ for $b_2=1.645088$. Refusing continuation can only remove rejection paths if boundaries are held fixed. Componentwise monotonicity in $\mu$ puts the least favorable null at zero. Familywise guarantees require preallocated alpha across multiple promotion hypotheses, or a gatekeeping order; for mandatory jointly successful endpoints, an intersection-union gate can use the full level per component while requiring all successes. [Lan and DeMets, 1983](https://eclass.uoa.gr/modules/document/file.php/MATH301/PracticalSession3/Lan_DeMets_1983.pdf)

Under the planning prior $\mu\sim N(0,4)$, $\mu\mid Z_1=z\sim N(v\sqrt tz,v)$ with $v=(1/4+t)^{-1}$, from completing the square. The predictive distribution for $Z_2$ has mean $\sqrt tz[1+(1-t)v]$ and variance $1-t+(1-t)^2v$. Continue if predictive $P(Z_2\ge b_2\mid z)\ge .10$, equivalently $z\ge -0.270438$; early efficacy proceeds to the full rollout for independent validation. The exact least favorable Type-I rate is **.050000** with nonbinding futility and **.044695** with modeled binding futility and no CPU screen. These are probability calculations under the specified Gaussian law; the Monte Carlo has sampling error.

The paired 10,000-candidate replay assumes a 60% structural-null stratum ($\mu=-.8$, CPU pass .10), 15% boundary null ($\mu=0$, pass .90) and 25% working ($\mu=3.6$, pass .99). Pass draws are conditionally independent given stratum. These are planning assumptions, not measured rejection rates for M1–M5. Both policies see the same potential Gaussian scores. Fixed spends the archival \$17.12 envelope for every candidate. Sequential spends a modeled 8.5% if CPU passes and the pilot is bought ($.75\times.18/3+.04$ of fixed, comprising a training share and a pilot overhead share). It pays the remaining fraction on continuation including early efficacy. This scales an archival bill as a scenario; it neither prices an individual seed retrospectively nor assumes a \$17.12 charge in the Stage 4 cap. If $p_s(\mu)=P(\mathrm{CPU\ pass}\mid\mu)$, then

$$
\mathbb E[C_{\rm seq}\mid\mu]=C_Fp_s(\mu)[f+(1-f)\bar\Phi(z_f-\sqrt t\mu)].
$$

| Outcome | Fixed MC | Sequential MC | Fixed analytic | Sequential analytic |
|---|---:|---:|---:|---:|
```

### Event #12: `item.completed` — `command_execution` (`item_9`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n 'joint|valid_mask|def .*read|return ' research/stage4_math_and_design/M3/verify_m3.py | Select-Object -First 85; "'$p=Get-Content research/stage4_math_and_design/M1/verify_m1.py; $p[0..220]; $p[430..575]'
```

**Aggregated Output:**

```text
46:    return orbit[torch.arange(len(iq), device=iq.device), turn % 4]
50:    return torch.from_numpy(np.stack((values.real, values.imag), axis=-1))
123:        return Pooled(torch.stack(clock_iq, dim=1), times, power, (t2 - t1) % 4, z1 % 4, z2 % 4)
138:            return evidence.iq[:, :1, :1].expand(-1, 1, 4) + 0.25
149:        return (g[:, None, :] * a).sum(dim=-1).unsqueeze(1)
154:    return {key: row[key] for key in ("n", "acc", "nll", "brier", "ece")}
169:    return result
176:    return dict(none=(), valid=(p1,), duplicate_copy_dedup=(p1, p1),
234:    return replay_info, (torch.from_numpy(z1), torch.from_numpy(z2), y, arms, measured)
250:    return a, fit
253:def check_pool(pool, reader, blind, pinned):
324:    return {"metrics": scores, "fresh_minus_valid": compare(fitted["valid"], fitted["fresh_resample"], y),
339:    return np.einsum("qnf,n->qf", values, weights / np.sqrt(2*np.pi))
365:        # The naive model uses the correct joint (B_1, Z2) marginal.
372:    return out
382:    return labels, x2, g, unit_walk, noise, z1
397:    return sums, families, labels
411:        return np.stack([np.interp(folded, angle_grid, emission_grid[:, f])
488:    return out
521:    print("Passed blind fixed-slot and disjoint-calibration checks.", flush=True)
"""CPU verification of the Stage 4 M1 quotient and phase transport."""

import json
import math
from fractions import Fraction
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from research.stage3_ministral_cache.schema import ROOTS, ROTATIONS, matched_sum, sector_of_complex
from research.stage3d_qwen_signed_phase.data_stage3d import (
    BLIND_CELLS, DEV_OMEGAS, NS_CFO_TRAIN, TRAIN_OMEGAS, TRAIN_TIMINGS,
    generate_paired_cfo_blocks, load_s17_encoder, public_phase_slots,
)
from research.stage3d_qwen_signed_phase.recompute_all import (
    compute_16way_group_avg, compute_representative_readout, paired_t_ci,
    score_vectors, softmax_temperature, work_f_sign_probe, work_g_rotation_and_controls,
)

STAGE3D_RESULTS = ROOT / "research/stage3d_qwen_signed_phase/results"
MODEL_SEEDS = (202611, 202622, 202633)
TRAIN_BLOCKS = 32
EPOCHS = 16
BATCH_SIZE = 256
LEARNING_RATE = 0.01
WIDTH = 24
RESIDUAL_LIMIT = 0.15
TASKS = ("q1", "q2", "q3_repeat", "q3_indep", "q3_after_pilot")


def lift_scores(canonical, codes, task, pilot_turn=0):
    """Map normalized canonical scores into the task's physical option order."""
    a, b = (codes // 4).unbind(dim=-1)
    if task == "q1":
        shift = b - a
    elif task == "q2":
        shift = a - pilot_turn
    elif task in ("q3_repeat", "q3_after_pilot"):
        shift = a
    elif task == "q3_indep":
        return torch.full_like(canonical, 0.25)
    else:
        raise ValueError(f"Unknown task {task}")
    index = (torch.arange(4, device=canonical.device) - shift[:, None]) % 4
    return canonical.gather(-1, index)


class SectorTransport(nn.Module):
    """Integrate the AWGN sector map against the signed Brownian propagator."""

    def __init__(self):
        super().__init__()
        x, w = np.polynomial.hermite.hermgauss(64)
        self.register_buffer("nodes", torch.from_numpy(x))
        self.register_buffer("weights", torch.from_numpy(w / math.sqrt(math.pi)))

    def forward(self, theta, variance, snr_db):
        angle = theta[..., None] + (2.0 * variance[..., None]).sqrt() * self.nodes
        scale = (2.0 * (10.0 ** (snr_db[..., None] / 10.0))).sqrt()
        c = torch.special.ndtr(scale * (angle + math.pi / 4).cos())
        s = torch.special.ndtr(scale * (angle + math.pi / 4).sin())
        base = torch.stack((c * s, (1 - c) * s, (1 - c) * (1 - s), c * (1 - s)), dim=-1)
        sector = (self.weights[:, None] * base).sum(dim=-2)
        return torch.stack([sector.roll(j, dims=-1) for j in range(4)], dim=-2)


class FrozenHost(nn.Module):
    """One shared frozen nonlinear host with a fixed text embedding."""

    def __init__(self, seed):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.register_buffer("text", torch.randn(1, 1, WIDTH) * 0.2)
            self.reader = nn.Sequential(nn.Linear(4 * WIDTH, 64), nn.GELU(), nn.Linear(64, 4))
        self.requires_grad_(False)
        self.eval()

    def forward(self, tokens):
        text = self.text.expand(len(tokens), -1, -1)
        return self.reader(torch.cat((tokens, text), dim=1).flatten(1))


class AdditivePhaseBaseline(nn.Module):
    """Joint code-only token heads and a separate public-metadata token head."""

    def __init__(self, host):
        super().__init__()
        self.host = host
        self.code_embedding = nn.Embedding(16, 16)
        self.windows = nn.Sequential(nn.Linear(64, 48), nn.GELU(), nn.Linear(48, 2 * WIDTH))
        self.phase = nn.Sequential(nn.Linear(12, 32), nn.GELU(), nn.Linear(32, WIDTH))

    def forward(self, codes, meta, kernel):
        first, second = self.code_embedding(codes).unbind(dim=1)
        paired = torch.cat((first, second, first - second, first * second), dim=-1)
        windows = self.windows(paired).reshape(-1, 2, WIDTH)
        phase = self.phase(meta).unsqueeze(1)
        return self.host(torch.cat((windows, phase), dim=1)).softmax(dim=-1)


class EquivariantPhaseCoupled(nn.Module):
    """S17-prior adapter with phase-coupled tokens and a physical output head."""

    def __init__(self, host, prior):
        super().__init__()
        self.host = host
        self.register_buffer("prior", torch.as_tensor(prior[:4], dtype=torch.float32))
        powers = ROOTS[None, :] ** np.arange(1, 4)[:, None]
        self.register_buffer("fourier_basis", torch.from_numpy(powers).to(torch.complex64))
        self.inner_embedding = nn.Embedding(4, 16)
        self.tokens = nn.Sequential(nn.Linear(60, 48), nn.GELU(), nn.Linear(48, 3 * WIDTH))

    def canonical_readout(self, inner_codes, meta, kernel):
        first, second = inner_codes.unbind(dim=1)
        a1, a2 = self.prior[first], self.prior[second]
        n = torch.arange(1, 4, device=meta.device)
        h, ell, _, omega, sigma, _ = meta[:, 6:].unbind(dim=1)
        dt = torch.stack((h + ell, h), dim=-1)
        signed_dt = dt * meta.new_tensor([1.0, -1.0])
        mean = omega[:, None, None] * signed_dt[..., None]
        decay = torch.exp(-0.5 * sigma[:, None, None].square() * dt[..., None] * n.square())
        phasors = decay * torch.exp(1j * mean * n)
        source = torch.stack((a1, a2), dim=1).to(torch.complex64) @ self.fourier_basis.T
        coupled = torch.view_as_real(phasors * source).flatten(1)
        inner = self.inner_embedding(inner_codes).flatten(1)
        physical = torch.bmm(a1[:, None, :], kernel).squeeze(1)
        features = torch.cat((inner, coupled, a1, a2, physical, meta[:, 6:10]), dim=-1)
        tokens = self.tokens(features).reshape(-1, 3, WIDTH)
        delta = RESIDUAL_LIMIT * self.host(tokens).tanh()
        source_posterior = (a1.log() + delta).softmax(dim=-1)
        return torch.bmm(source_posterior[:, None, :], kernel).squeeze(1)

    def forward(self, codes, meta, kernel):
        canonical = self.canonical_readout(codes % 4, meta, kernel)
        return lift_scores(canonical, codes, "q3_repeat")

    def orbit_table(self, meta, kernel):
        """Evaluate each representative once at this context before lifting."""
        canonical = self.canonical_readout(
            INNER_CODES, meta.expand(16, -1), kernel.expand(16, -1, -1),
        )
        inner_index = (ALL_CODES[:, 0] % 4) * 4 + ALL_CODES[:, 1] % 4
        return lift_scores(canonical[inner_index], ALL_CODES, "q3_repeat")


def metadata(cond):
    slots = public_phase_slots(cond.omega, cond.h, cond.ell, cond.t_p)
    scalars = (cond.h, cond.ell, cond.t_p, cond.omega, cond.sigma_phi, cond.snr_db)
    return torch.tensor([*slots, *scalars], dtype=torch.float32)


def physical_kernel(transport, theta, variance, snr_db):
    return transport(*(torch.as_tensor(x, dtype=torch.float64) for x in (theta, variance, snr_db)))


def condition_kernel(transport, cond):
    dt = cond.h + cond.ell
    return physical_kernel(transport, cond.omega * dt, dt * cond.sigma_phi**2, cond.snr_db)


def encode_view(encoder, mother, r1, r2):
    y1, y2 = mother.rotated_view(r1, r2)
    return np.column_stack((encoder.encode(y1), encoder.encode(y2)))


def train_data(encoder, transport):
    columns = ([], [], [], [])
    seeds = []
    for i, (h, ell, tp) in enumerate(TRAIN_TIMINGS):
        paired, block_seeds = generate_paired_cfo_blocks(
            NS_CFO_TRAIN, i, TRAIN_BLOCKS, h, ell, tp, TRAIN_OMEGAS, split_prefix="cfo_train",
        )
        seeds.extend(block_seeds)
        for omega in TRAIN_OMEGAS:
            mother = paired[omega]
            n = len(mother.x1)
            values = (
                torch.from_numpy(encode_view(encoder, mother, 0, 0)),
                metadata(mother.condition).expand(n, -1),
                condition_kernel(transport, mother.condition).float().expand(n, -1, -1),
                torch.from_numpy(mother.k3_repeat),
            )
            for column, value in zip(columns, values):
                column.append(value)
    return tuple(torch.cat(column) for column in columns), seeds


def fit(model, training, seed):
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATE,
    )
    rng = torch.Generator().manual_seed(seed)
    n = len(training[-1])
    for _ in range(EPOCHS):
        for index in torch.randperm(n, generator=rng).split(BATCH_SIZE):
            z, meta, kernel, y = (column[index] for column in training)
            loss = F.nll_loss(model(z, meta, kernel).clamp_min(1e-12).log(), y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    assert torch.isfinite(loss)
    model.eval()


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def scores(p, labels):
    scored = score_vectors(np.asarray(p).reshape(-1, 4), np.asarray(labels).ravel())
    return {k: scored[k] for k in ("acc", "nll", "brier", "tie_count")}
            "expected_unrotated_MAP_accuracy": float(weights @ probs[np.arange(4), unrotated]),
        }
    return out


def benchmark_cell(cid, data, encoder, archive, saved, models):
    indices, labels, independent, meta, kernels = data
    oracle = physical_oracle(encoder.table, kernels)
    expected = archive[f"{cid}__direct_table_probs"]
    oracle_on_mothers = lookup(oracle, indices)
    parity = float(np.max(np.abs(oracle_on_mothers - expected)))
    np.testing.assert_allclose(oracle_on_mothers, expected, atol=1e-9, rtol=0)
    oracle_score = scores(oracle_on_mothers, labels)
    np.testing.assert_allclose(oracle_score["acc"], saved["cells"][cid]["analytic_same_mothers"]["direct_s17_table"]["acc"])
    unrotated_prior = encoder.table[(indices // 16)]
    uniform = lift_scores(torch.zeros_like(torch.from_numpy(oracle[0])), ALL_CODES, "q3_indep")[None]
    uniform = np.repeat(uniform.numpy(), len(meta), axis=0)
    uniform_on_mothers = lookup(uniform, indices)
    result = {
        "mothers_per_sign": labels.shape[-1], "C_direct_s17": oracle_score,
        "C_archive_max_abs_probability_difference": parity, "C_equivariance_defect": orbit_defect(oracle),
        "unrotated_s17_MAP": scores(unrotated_prior, labels), "models": {},
        "q3_indep_zero_tv": {
            **scores(uniform_on_mothers, independent),
            "max_total_variation_from_uniform": float(np.max(0.5 * np.abs(uniform_on_mothers - 0.25).sum(axis=-1))),
        },
    }
    for name, arms in models.items():
        raw_batch_tables = [predict_table(arm, meta, kernels) for arm in arms]
        quotient = isinstance(arms[0], EquivariantPhaseCoupled)
        model_tables = [predict_quotient_table(arm, meta, kernels) for arm in arms] if quotient else raw_batch_tables
        metrics = [scores(lookup(tbl, indices), labels) for tbl in model_tables]
        defects = [orbit_defect(tbl) for tbl in model_tables]
        mean_hit = np.mean([
            score_vectors(lookup(tbl, indices).reshape(-1, 4), labels.ravel())["per_item_hit"].reshape(labels.shape)
            for tbl in model_tables
        ], axis=0)
        oracle_hit = score_vectors(oracle_on_mothers.reshape(-1, 4), labels.ravel())["per_item_hit"].reshape(labels.shape)
        block_gaps = (mean_hit - oracle_hit).mean(axis=(0, 1)).reshape(-1, 64).mean(axis=1)
        result["models"][name] = {
            "mean": {key: float(np.mean([row[key] for row in metrics])) for key in ("acc", "nll", "brier")},
            "per_seed": {str(seed): {**row, "equivariance_defect": defect}
                         for seed, row, defect in zip(MODEL_SEEDS, metrics, defects)},
            "maximum_equivariance_defect": max(defects),
            "accuracy_minus_oracle_paired_block_t95": paired_t_ci(block_gaps),
        }
        if quotient:
            result["models"][name]["ungrouped_batch_maximum_equivariance_defect"] = max(
                orbit_defect(tbl) for tbl in raw_batch_tables
            )
            result["models"][name]["maximum_grouped_vs_ungrouped_probability_difference"] = max(
                float(np.abs(p - q).max()) for p, q in zip(model_tables, raw_batch_tables)
            )
    return result


ALL_CODES = torch.cartesian_prod(torch.arange(16), torch.arange(16))
INNER_CODES = torch.cartesian_prod(torch.arange(4), torch.arange(4))


def main():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    encoder, digest = load_s17_encoder(ROOT)
    transport = SectorTransport()
    archive_file = STAGE3D_RESULTS / "blind_q3_scores.npz"
    saved = json.loads((STAGE3D_RESULTS / "summary_master.json").read_text(encoding="utf-8"))
    training, train_seeds = train_data(encoder, transport)
    models = {"A_AdditivePhaseBaseline": [], "B_EquivariantPhaseCoupled": []}
    for seed in MODEL_SEEDS:
        host = FrozenHost(123_457)  # Same frozen weights and text on both arms and every seed.
        original_host = {key: value.clone() for key, value in host.state_dict().items()}
        for key, make in zip(models, (AdditivePhaseBaseline, EquivariantPhaseCoupled)):
            torch.manual_seed(seed)
            ctor_args = (host, encoder.table) if make is EquivariantPhaseCoupled else (host,)
            arm = make(*ctor_args)
            fit(arm, training, seed)
            models[key].append(arm)
        assert all(torch.equal(value, original_host[key]) for key, value in host.state_dict().items())
        print(f"Trained both CPU connectors for seed {seed}", flush=True)

    with np.load(archive_file, allow_pickle=False) as archive, torch.no_grad():
        work_f, work_g, mismatch = stage3d_diagnostics(archive, saved)
        cells = {}
        for cid, cell in BLIND_CELLS.items():
            data = blind_data(cell, encoder, transport, archive, cid, train_seeds)
            cells[cid] = benchmark_cell(cid, data, encoder, archive, saved, models)
            if cell.namespace == BLIND_CELLS["G1_main_extrap_both"].namespace:
                scan = boundary_scan(encoder.table, data[0], transport)
            print(cid, ", ".join(
                f"{k[0]}={100 * v['mean']['acc']:.2f}%"
                for k, v in cells[cid]["models"].items()
            ), f"C={100 * cells[cid]['C_direct_s17']['acc']:.2f}%", flush=True)

    a, b = models
    extrapolation = [cells[cid] for cid in ("G1_main_extrap_both", "G3_extrap_angle")]
    checks = {
        "B_all_grouped_cell_defects_equal_zero": all(row["models"][b]["maximum_equivariance_defect"] == 0.0
                                                    for row in cells.values()),
        "B_all_cell_defects_at_most_1e_6": all(row["models"][b]["maximum_equivariance_defect"] <= 1e-6
                                            for row in cells.values()),
        "B_all_ungrouped_cell_defects_at_most_1e_6": all(
            row["models"][b]["ungrouped_batch_maximum_equivariance_defect"] <= 1e-6
            for row in cells.values()
        ),
        "A_breaks_equivariance": all(row["models"][a]["maximum_equivariance_defect"] > 1e-6
                                     for row in cells.values()),
        "B_G1_G3_each_seed_above_55_pct": all(v["acc"] > 0.55 for row in extrapolation
                                             for v in row["models"][b]["per_seed"].values()),
        "B_all_cell_means_within_3pp_of_C": all(abs(row["models"][b]["mean"]["acc"]
                                                  - row["C_direct_s17"]["acc"]) <= 0.03
                                               for row in cells.values()),
        "A_G1_G3_mean_at_most_25_pct": all(row["models"][a]["mean"]["acc"] <= 0.25 for row in extrapolation),
    }
    out = {
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "torch": torch.__version__, "device": "cpu"},
        "train": {"s17_sha256": digest, "namespace": NS_CFO_TRAIN, "blocks_per_timing": TRAIN_BLOCKS,
                  "rows": len(training[-1]), "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                  "learning_rate": LEARNING_RATE, "timings": TRAIN_TIMINGS, "omegas": TRAIN_OMEGAS,
                  "dev_omegas_unused": DEV_OMEGAS, "model_seeds": MODEL_SEEDS, "host_seed": 123_457,
                  "trainable_parameters": {k: count_trainable(v[0]) for k, v in models.items()},
                  "B_log_prior_residual_limit": RESIDUAL_LIMIT},
        "checks": checks, "all_task_quotient_lift_defects": logit_lift_checks(),
        "cells": cells, "boundary_scan_fixed_variance_0p04": scan,
        "stage3d_saved_cells": {cid: {"arms": value["arms"], "analytic_same_mothers": value["analytic_same_mothers"]}
                                for cid, value in saved["cells"].items()},
        "work_f_recomputed": work_f, "work_g_recomputed": work_g,
        "work_f_overlap_weighted_signed_contrast_spectra": mismatch,
        "pf_complete_orbit_counterexample": pf_counterexample(),
    }
    output = Path(__file__).with_name("results_m1.json")
    output.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(output, flush=True)
    print(json.dumps(checks), flush=True)
    assert all(value for key, value in checks.items() if key != "A_G1_G3_mean_at_most_25_pct")


if __name__ == "__main__":
    main()
```

### Event #13: `item.completed` — `command_execution` (`item_11`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n -C 5 'Q2.*invariant|Q2.*roll|q_2|p_2|joint posterior|joint\\[4,4\\]|a_1' research/stage4_math_and_design/M3/THEOREM_M3.md | Select-Object -First 180; "'$p=Get-Content research/stage4_math_and_design/M5/verify_m5.py; $p[27..71]; $p[123..155]'
```

**Aggregated Output:**

```text
119-
120-There is also an optimal **coarsened-data** Bayes comparator. Conditional on \(v,g\), \(D_{\rm sum}\) is complex Gaussian with mean \(i^g\sum_jP_je^{iv_j}\) and variance \(\sum_jP_j\). Phase marginalization retains the squared modulus of that conditional mean in the exponent. A coherent single-clock model adds a likelihood approximation on top of discarding the individual clock sums.
121-
122-### Q3_after_pilot posterior
123-
124:Add the withheld clock \(\tau=h+\ell\). Let \(H_q(u,s)\) be the probability that \(\alpha_3 i^u e^{is}+\epsilon_3\) falls in apparent sector \(q\), given \(u,s\), using the withheld-window noise law. For Gaussian noise it is a two-dimensional sector integral. With \(\mathcal W_g(v)=\exp\{2\sum_j\Re[D_je^{-i(bg+v_j)}]\}\), the unnormalized joint posterior is
125-
126-\[
127-J(k,q)=\sum_g a_{z_1}(k+g)\,
128-\mathbb E\!\left[m_{z_2}(\Theta_h)\mathcal W_g(B_{t_1},\ldots,B_{t_K})
129-H_q(k+g,\Theta_\tau)\right].
--
156-The archived first valid matched sums have RMS magnitude 1.27735. Their global I/Q pair moves by RMS 1.80644 under a joint quarter-turn because \(|iT-T|^2=2|T|^2\). The canonicalized pool had zero movement for that action. The normalized absolute cross moments of the raw matched sums are 0.62189 for the two valid pilots and 0.00420 for valid and independent, consistent with the generator's common signal and separately drawn distractor. [Result receipt](results_m3.json)
157-
158-## 5. Fixed-slot interface and guarantees
159-
160-1. **Validate and canonicalize.** Use \(z_i=4a_i+u_i\). Resolve IDs and check provenance, calibration and exact sample equality before filtering. A production clock should publish one canonical timestamp per physical tick. For schema-tolerance aliases, the prototype sorts timestamps per ID, chooses the minimum, and rejects a spread beyond \(10^{-12}\). Remove independent-channel references before feature evaluation. A conflict-free distractor contributes no activation or gradient. A literal copy of a canonical acquisition ID makes no change. Sort unique IDs before reduction for deterministic sums.
161:2. **Keep exact evidence upstream.** Use sign swaps for the integer rotation \(i^{-a_1}\) on I/Q before floating-point reduction. Apply continuous derotation once per clock. Pack `Pooled(iq[B,64,2], times, precision[64], gap[B], family1[B], family2[B])` with zero padding excluded from the reader. The prototype batches one shared public design. A production wrapper can batch by shared design or use a padded per-row mask. Declare \(\omega,\sigma_\phi,h,\tau\) and the noise model as public inputs. The continuous path is differentiable for fixed metadata. ID groupings, channel flags, and clock equality are discrete.
162:3. **Expose fixed task evidence to M1.** Compute `canonical_q2[4]`, `canonical_q3[4]`, and optionally `joint[4,4]` from the reader. Provide a separate frame \(a_1\) for external sector relabeling. Use one always-present task evidence marker and fixed projection, with identical prose and no raw IDs, counts, flags, or clock lists. A multitask prompt can reserve both markers. The CPU prototype returns a single four-component Q2 slot. A projection to Qwen dimensions and an intervention on the frozen host remain separate integration work.
163-4. **Use the model posterior for guaranteed decisions.** Arbitrary host scores or trainable residual logits can change posterior calibration and rankings even at fixed length. Keep the physical posterior available as the decision output and the frozen-host readout as an experimental comparator.
164-
165:**Rotation.** Under the joint coordinate action, \(a_i\mapsto a_i+r,\ G\mapsto G+r\), and \(X_1,u_1,u_2\) are invariant. Thus \(i^{-a_1}T_m\) and \(a_2-a_1\) are unchanged. Use \(G-a_1,U-a_1\) in the canonical kernel. Q2 is invariant and Q3 global sector probabilities permute by \(r\) through an explicit external index rotation. The exact-zero CPU verification covers Q2 and pooled evidence in all four frames. Qwen outputs were replayed from the archive.
166-
167-**Blackwell.** Dropping the new acquisition maps the augmented experiment to the old experiment. There is an explicit garbling even after same-clock pooling. Given the augmented pooled sum \(S_+\) with total precision \(P_+\), the old sum of precision \(P_o\) has conditional law \(\mathcal{CN}((P_o/P_+)S_+,\,P_o(1-P_o/P_+))\), independent of the physical phase. This follows from joint Gaussian receiver noise with covariance \(P_o\) between the two sums. Across clocks, use this garbling within each clock and drop new-only clocks. An exact posterior preserves all decisions about its target, so the Bayes risk of the augmented posterior experiment improves weakly for every loss. The expected log-loss gain is \(I(X_1;T_{\mathrm{new}}\mid Z_1,Z_2,\mathcal R_{\mathrm{old}})\). For the finite-time Brownian Q2 model with informative \(a_{z_1}\) and a fresh finite positive-SNR pilot this is strict. At zero old evidence, the new pilot derivative of the first target moment is nonzero for some \(z_2\), since the unconditional \(e^{iB_{\mathrm{new}}}\) moment is positive. Continuity and positive observation densities give a positive-measure change in the posterior. This applies to the first pilot and subsequent ones at common or distinct clocks. A copy, independent distractor or zero-SNR acquisition gives equality. [Blackwell, *Equivalent Comparisons of Experiments*](https://doi.org/10.1214/aoms/1177729032)
168-
169-For Q2 with these same nondegeneracy conditions, strict population MAP improvement also follows. At zero old evidence, choose a new pilot value giving a nonuniform posterior. Quarter-turning **only that new pilot** produces the four label rotations of that posterior, because the old likelihood is constant in \(G\) there. The intersection of the four sets of maximizers is empty. The strict gaps excluding each label from at least one such maximizer set persist in neighborhoods of the new values and of zero old evidence. The conditional new pilot has positive density in every neighborhood. Therefore the expected new maximum posterior probability strictly exceeds the maximum of the conditional mean on a positive-measure set of old observations. This strengthens the sample-level claim to a population statement for the specified model. Accuracy can still go down on individual mothers or a finite test set, and the result supplies no guarantee for a learned host. For Q3 alone, strict log-loss improvement requires its conditional sector distribution to change.
170-
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
```

### Event #14: `item.completed` — `command_execution` (`item_12`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "rg -n -A 15 'def radial_edges|def quality_edges|def .*summary' research/stage4_math_and_design/M5/verify_m5.py; "'$p=Get-Content research/stage4_math_and_design/M3/THEOREM_M3.md -Encoding UTF8; $p[120..142]; $p=Get-Content research/stage4_math_and_design/M6/THEOREM_M6.md -Encoding UTF8; $p[115..130]'
```

**Aggregated Output:**

```text
110:def radial_edges(bits):
111-    if bits == 0:
112-        return np.array([0., np.inf])
113-    edges = np.expm1(np.linspace(0, np.log(25), 2**bits + 1))
114-    edges[-1] = np.inf
115-    return edges
116-
117-
118:def quality_edges(bits):
119-    if bits == 0:
120-        return np.array([0., np.inf])
121-    if bits == 1:
122-        return np.array([0., 4., np.inf])
123-    return np.r_[0., np.geomspace(.75, 16., 2**bits - 1), np.inf]
124-
125-
126-def encode(iq, t, residual, variance, encoder):
127-    original = encoder.encode(iq.reshape(-1, 3)).reshape(t.shape)
128-    _, turns = input_features(iq.reshape(-1, 3), True)
129-    turns = turns.reshape(t.shape)
130-    np.testing.assert_array_equal(original // 4, turns)
131-    local_angle = (np.angle(t * (1j)**(-turns)) + np.pi / 4) / (np.pi / 2)
132-    packets = {"4_s17": original}
133-    bins = {}

### Q3_after_pilot posterior

Add the withheld clock \(\tau=h+\ell\). Let \(H_q(u,s)\) be the probability that \(\alpha_3 i^u e^{is}+\epsilon_3\) falls in apparent sector \(q\), given \(u,s\), using the withheld-window noise law. For Gaussian noise it is a two-dimensional sector integral. With \(\mathcal W_g(v)=\exp\{2\sum_j\Re[D_je^{-i(bg+v_j)}]\}\), the unnormalized joint posterior is

\[
J(k,q)=\sum_g a_{z_1}(k+g)\,
\mathbb E\!\left[m_{z_2}(\Theta_h)\mathcal W_g(B_{t_1},\ldots,B_{t_K})
H_q(k+g,\Theta_\tau)\right].
\]

Normalize over \(k,q\). Summing over \(q\) gives Q2 and summing over \(k\) gives Q3_after_pilot. The second-window emission and third-window sector emission generally require the same joint bridge. The carrier-only vector \(L_g\) need not retain the pilot update of \(B_\tau\). Coincident pilot and prediction clocks share the latent phase, with independent observation noises. In a causal receiver, only pilots available by the prediction deadline enter the calculation.

The normalized entries of \(J\) give an exact minimal Bayesian statistic for this joint target at fixed data codes and design. For the pair of transmitted symbols in the repeat protocol, the posterior is supported on \(X_3=X_1\) and the four Q2 probabilities suffice.

## 4. Stage 3D reference-fusion diagnosis

The replay uses all 512 mothers and three saved host seeds with locked temperatures. Accuracy is in percent and NLL in nats.

| Arm | Work A Acc / NLL | Qwen phase Acc / NLL |
|---|---:|---:|
| none | 25.00 / 1.3863 | 24.80 / 1.3928 |
| valid | 66.41 / 0.8192 | 28.78 / 1.3889 |
## 4. Tier 2 • sequential decisions and an exact cost envelope

### 4.1 VoI and continuation

Let $\xi$ encode the true effects, feasibility and dataset transfer, $D_i$ the current evidence, and $b_i$ the remaining authorized budget. With final utility $u(a,\xi)$, define $U(D)=\max_a\mathbb E[u(a,\xi)\mid D]$ over promote, stop or archive, including a cost for erroneous promotion. For a candidate check (j) with new observation $X_j$ and cost $c_j$, one-step $\mathrm{VoI}_j=\mathbb E[U(D,X_j)\mid D]-U(D)$. A check worth less than its priced cost should not be purchased as a one-step action. With dependencies and horizon, the exact finite decision recursion is

$$
V(D,b)=\max\{U(D),\max_{j\;\mathrm{eligible},\,c_j\le b}[-c_j+\mathbb E V(D,X_j,b-c_j)]\}.
$$

This is optimal for specified priors, utilities, transition likelihoods and caps. Greedy information per dollar is at most a heuristic when checks are complementary. In a differentiable separable allocation with concave information utilities $v_j(n_j)$, constant per-unit prices $c_j$ and a binding budget, KKT gives $v'_j(n_j)/c_j=\lambda$ for interior arms, with only arms above this price at the origin receiving effort. For $v_j=-w_j\sigma_j^2/n_j$, the allocation is $n_j\propto\sqrt{w_j\sigma_j^2/c_j}$, with independent clusters and a fixed variance estimand. Use mandatory cohort minima and integer reallocation, especially for nine CTTC sessions. No universal zero-waste guarantee follows without a specified law. In a nonidentifiable case the valuable output is the collision and the minimum extra observation, even when no GPU trial follows.

### 4.2 Two-look design and what the Monte Carlo verifies

Use a preregistered directional **primary** score $\mu>0$ in final-standard-error units above the practical margin. In the canonical independent-increment Gaussian experiment, $Z_1\sim N(\sqrt t\mu,1)$ and $Z_2=\sqrt tZ_1+\sqrt{1-t}V\$, $V\sim N(\sqrt{1-t}\mu,1)$ independent. $t=.18$ is an information fraction. A one-seed pilot at 15–20% of optimizer steps does not, by itself, have this information fraction or this joint distribution.

```

### Event #15: `item.completed` — `command_execution` (`item_10`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$src=Join-Path $env:LOCALAPPDATA '"'uv/cache'; "'$dst=Join-Path (Get-Location) '"'research/stage4_math_and_design/M1/.uv-cache'; "'$packages=@('"'numpy','scipy','torch','networkx','sympy','mpmath','setuptools','filelock','fsspec','jinja2','markupsafe','typing-extensions'); "'$archive=@('"'3Oxde2-Jkv6KGjnQ','cSnmXl2iWBpNAXGE','l-bbeSyNZh-_fB5p','Vqk4-stqIOqEeUEo','fqY_x9ViVdktj-A4','k4asR5j2Ej4BfAdg','LeBFdodBQkFZbYuu','-3Eg0_w5zGyPhAVD','1uFPYbapY1T1Qpz9','-INXTsup9SBW2nnw','6JCnrcOQFUae-bks','wd1VbUJDzJ5LkUe1','cwrpe1AjGHyI2_gb'); foreach ("'$sub in @('"'archive-v0','wheels-v6/pypi','simple-v21/pypi')) { New-Item -ItemType Directory -Force -Path (Join-Path "'$dst $sub) | Out-Null }; foreach ($p in $packages) { Copy-Item -Recurse -LiteralPath (Join-Path $src "wheels-v6/pypi/$p") -Destination (Join-Path $dst '"'wheels-v6/pypi'); Copy-Item -LiteralPath (Join-Path "'$src "simple-v21/pypi/$p.rkyv") -Destination (Join-Path $dst '"'simple-v21/pypi') }; foreach ("'$id in $archive) { Copy-Item -Recurse -LiteralPath (Join-Path $src "archive-v0/$id") -Destination (Join-Path $dst '"'archive-v0') }; "'$env:UV_CACHE_DIR=$dst; $env:UV_OFFLINE='"'1'; "'$env:UV_LINK_MODE='"'hardlink'; "'$env:UV_NO_PROJECT='"'1'; "'$env:TEMP=Join-Path (Split-Path $dst) '"'.uv-tmp'; "'$env:TMP=$env:TEMP; New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B -c '"'import numpy, scipy, torch, sys; print(sys.version.split()[0], numpy.__version__, scipy.__version__, torch.__version__)'"
```

**Aggregated Output:**

```text
Installed 12 packages in 9.94s
3.12.7 2.3.5 1.17.0 2.12.1+cpu
```

### Event #16: `item.completed` — `file_change` (`item_13`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M1\\verify_m1.py",
    "kind": "update"
  }
]
```

### Event #17: `item.completed` — `file_change` (`item_14`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M1\\verify_m1.py",
    "kind": "update"
  }
]
```

### Event #18: `item.completed` — `file_change` (`item_15`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M1\\verify_m1.py",
    "kind": "update"
  }
]
```

### Event #19: `item.completed` — `agent_message` (`item_16`)

I found two interface details that affect the comparison. M3’s canonical Q2 scores are already in the TX label frame under a common receive-gauge change, and M5’s 16-bit packet preserves S17 in its top four bits. I’ve set up matched controls for quotient sharing and phase coupling, with the physical-only arm separate.

### Event #20: `item.completed` — `file_change` (`item_17`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M1\\verify_m1.py",
    "kind": "update"
  }
]
```

### Event #21: `item.completed` — `command_execution` (`item_18`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Content research/stage4_math_and_design/M5/verify_m5.py -TotalCount 26; rg -n '"'^(class |def |TASKS)'"' research/stage4_math_and_design/M1/verify_m1.py; Get-ChildItem -Force research/stage4_math_and_design/M1 | Select-Object Name,Length"
```

**Aggregated Output:**

```text
"""Seeded CPU replay for the M5 cache and one-window blind test."""
from dataclasses import dataclass
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
from scipy.stats import t as student_t


HERE = Path(__file__).resolve().parent
MATCHED = HERE.parents[2] / "research" / "matched_cache"
sys.path.insert(0, str(MATCHED))
from learned import forward, input_features, load


SEED = 50261126
N_TEST = 4096
46:TASKS = ("q1", "q2", "q3_repeat", "q3_indep", "q3_after_pilot")
56:def lift_scores(canonical, codes, task, pilot_turn=0, receive_gauge=False, tx_chart_turn=0):
73:class SectorTransport(nn.Module):
92:class EvidencePacket(NamedTuple):
98:def decode_windows(words, bits, valid=None):
126:class FrozenHost(nn.Module):
152:class EquivariantEvidenceConnector(nn.Module):
232:def metadata(cond, rate_var=0.0):
238:def physical_kernel(transport, theta, variance, snr_db):
242:def condition_kernel(transport, cond, rate_var=0.0):
247:def encode_view(encoder, mother, r1, r2):
252:def draw_split(encoder, transport, namespace, blocks, omegas, split):
274:def fit(model, training, seed):
291:def count_trainable(model):
295:def scores(p, labels):
300:def orbit_defect(p, shift_rule=lambda r, s: r):
310:def blind_data(cell, encoder, transport, archive, cid, train_seeds):
348:def physical_oracle(prior, kernels):
355:def lookup(table, indices):
359:def predict_table(model, meta, kernel):
366:def predict_orbit_tables(model, meta, kernel, slots=None, gamma=RESIDUAL_LIMIT):
371:def logit_lift_checks():
391:def spectrum(matrix):
410:def cal_rank_certificate(cal, overlap):
434:def stage3d_diagnostics(archive, saved):
478:def pf_counterexample():
495:def boundary_scan(prior, g1_indices, transport):
513:def lock_temperature(model, dev):
530:def apply_temperature(probabilities, temperature):
534:def paired_change(before, after, y):
548:def mean_scores(probabilities, targets, oracle_nll):
556:def benchmark_cell(cid, data, encoder, archive, saved, models, calibrations):
656:def centered(values):
660:def jacobian_readout(model, meta, target, *, stack=False):
722:def candidate_diagnostics(models, contexts, prior, host):
742:def joint_phase_moment(edges, harmonics, rate_mean, phase_var, rate_var):
754:def pilot_joint_marginals(joint, valid, no_pilot_q3):
761:def interface_checks(encoder, model, residual_model, transport):
866:def main():

Name            Length
----            ------
.uv-cache             
.uv-tmp               
results_m1.json 59462 
THEOREM_M1.md   28756 
verify_m1.py    51143 


```

### Event #22: `item.completed` — `file_change` (`item_20`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M1\\THEOREM_M1.md",
    "kind": "update"
  }
]
```

### Event #23: `item.completed` — `command_execution` (`item_21`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command "Get-Process python,uv -ErrorAction SilentlyContinue | Sort-Object StartTime -Descending | Select-Object -First 8 Id,CPU,StartTime,Path; git status --short --untracked-files=normal -- research/stage4_math_and_design/M1; rg -n '"'^## Round 2|''^### 6'"' research/stage4_math_and_design/M1/THEOREM_M1.md"
```

**Aggregated Output:**

```text

?? research/stage4_math_and_design/M1/
207:## Round 2 Convergence
211:### 6.1 Four primary arms and the causal contrasts
230:### 6.2 Candidate geometry and transfer to the true frozen host
245:### 6.3 Dual cache contract
258:### 6.4 Gauge and M3 joint factor
280:### 6.5 Shared CFO, correlated phase, and R3
   Id      CPU StartTime           Path                                                                                
   --      --- ---------           ----                                                                                
28336 82.09375 2026/11/26 19:51:48 C:\qazwsx\python.exe                                                                
22944 0.015625 2026/11/26 19:51:48 C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_ma...
15796  0.21875 2026/11/26 19:51:47 C:\qazwsx\Scripts\uv.exe                                                            
14228 202.6875 2026/11/26 19:49:40 C:\qazwsx\python.exe                                                                
29904        0 2026/11/26 19:49:40 C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_ma...
 6204 6.484375 2026/11/26 19:49:34 C:\qazwsx\Scripts\uv.exe                                                            
22696 0.421875 2026/11/26 14:21:22 C:\qazwsx\python.exe                                                                
15784 0.546875 2026/11/26 14:21:21 C:\qazwsx\python.exe                                                                


```

### Event #24: `item.completed` — `command_execution` (`item_22`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$m1=Join-Path (Get-Location) '"'research/stage4_math_and_design/M1'; "'$env:UV_CACHE_DIR=Join-Path $m1 '"'.uv-cache'; "'$env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:TEMP=Join-Path $m1 '"'.uv-tmp'; "'$env:TMP=$env:TEMP; $env:OPENBLAS_NUM_THREADS='"'1'; "'$env:OMP_NUM_THREADS='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B -c 'import torch; from research.stage4_math_and_design.M1 import verify_m1 as m; torch.set_num_threads(1); enc,_=m.load_s17_encoder(m.ROOT); host=m.FrozenHost(m.HOST_SEED); a=m.EquivariantEvidenceConnector(host,enc.table,quotient=True,coupled=True); b=m.EquivariantEvidenceConnector(host,enc.table,quotient=True,coupled=True,residual=True); print(m.interface_checks(enc,a,b,m.SectorTransport())); split,_=m.draw_split(enc,m.SectorTransport(),m.NS_CFO_DEV,1,m.DEV_OMEGAS,\"smoke\"); print(m.candidate_diagnostics({k:[a] for k in (m.ARM0,m.ARM1,m.ARM2)},{\"dev\":(split[1][::64],split[2][::64].double())},enc.table,host)[\"contexts\"][\"dev\"][m.ARM1][\"202611\"][\"rank_min\"])'"
```

**Aggregated Output:**

```text
Traceback (most recent call last):
  File "<string>", line 1, in <module>
NameError: name 'smoke' is not defined
{'all_65536_prefixes_and_four_turns_checked': True, 'host_pure_probabilities_independent_of_kernel_even_if_nan': True, 'm3_marginals_and_missing_fallback_checked': True, 'm3_rx_tx_action_max_probability_defect': {'q1': 0.0, 'q2': 0.0, 'q3_repeat': 0.0, 'q3_indep': 0.0, 'q3_after_pilot': 0.0}, 'synthetic_r3_matched_filter_max_abs_difference': 9.036560719766055e-16, 'synthetic_r3_packet_model_rx_gauge_defects': {'arm1_equivariant_host_pure': {'4': 0.0, '16': 0.0}, 'arm2_equivariant_physical_residual': {'4': 0.0, '16': 0.0}}, 'hardware_capture_scored': False, 'extended_packet_weights_trained': False, 'shared_cfo_joint_moment': {'edges': [(0, 2), (2, 4), (0, 4)], 'harmonics': [1, -2, 1], 'covariance': [[0.18, 0.16, 0.34], [0.16, 0.18, 0.34], [0.34, 0.34, 0.68]], 'closed_form_re_im': [np.float64(0.7868196039421212), np.float64(0.3898288042922555)], 'gauss_hermite_abs_error': 1.900395703982448e-15, 'spuriously_independent_rate_abs_error': 0.4150823626093332}}
```

### Event #25: `item.completed` — `command_execution` (`item_23`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$m1=Join-Path (Get-Location) '"'research/stage4_math_and_design/M1'; "'$env:UV_CACHE_DIR=Join-Path $m1 '"'.uv-cache'; "'$env:UV_OFFLINE='"'1'; "'$env:UV_NO_PROJECT='"'1'; "'$env:TEMP=Join-Path $m1 '"'.uv-tmp'; "'$env:TMP=$env:TEMP; $env:OPENBLAS_NUM_THREADS='"'1'; "'$env:OMP_NUM_THREADS='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B -c 'import torch; from research.stage4_math_and_design.M1 import verify_m1 as m; torch.set_num_threads(1); enc,_=m.load_s17_encoder(m.ROOT); host=m.FrozenHost(m.HOST_SEED); a=m.EquivariantEvidenceConnector(host,enc.table,quotient=True,coupled=True); split,_=m.draw_split(enc,m.SectorTransport(),m.NS_CFO_DEV,1,m.DEV_OMEGAS,str(0)); meta,k=split[1][::64],split[2][::64].double(); i=(m.INNER_CODES @ torch.tensor([16,1])).numpy(); j=m.jacobian_readout(a,meta,m.physical_oracle(enc.table,k)[:,i],stack=True); print(j.get(next(iter(dict(rank_min=1)))), j.get(next(iter(dict(shared_last_projection_tangent=1)))).get(next(iter(dict(rank=1)))))'"
```

**Aggregated Output:**

```text
3 192
```

### Event #26: `item.completed` — `agent_message` (`item_24`)

The interface checks passed for all 65,536 packed values and all four quarter turns. In the shared-CFO test, Gauss–Hermite quadrature matched the joint moment formula to \(1.9\times10^{-15}\). Treating the rate uncertainty as independent across edges changed that same moment by 0.415. The trained-arm comparison is still running.

### Event #27: `item.completed` — `command_execution` (`item_25`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command 'Get-Item research/stage4_math_and_design/M1/results_m1.json | Select-Object Length,LastWriteTime; Get-Process -Id 28336 -ErrorAction SilentlyContinue | Select-Object CPU,StartTime; Get-ChildItem -Force research/stage4_math_and_design/M1/.uv-tmp | Select-Object Name'
```

**Aggregated Output:**

```text

Length LastWriteTime      
------ -------------      
 59462 2026/11/26 19:34:14
                          
                          


```

### Event #28: `item.completed` — `command_execution` (`item_19`)

- Status: `failed`, Exit Code: `1`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$m1=Join-Path (Get-Location) '"'research/stage4_math_and_design/M1'; "'$env:UV_CACHE_DIR=Join-Path $m1 '"'.uv-cache'; "'$env:UV_OFFLINE='"'1'; "'$env:UV_LINK_MODE='"'hardlink'; "'$env:UV_NO_PROJECT='"'1'; "'$env:TEMP=Join-Path $m1 '"'.uv-tmp'; "'$env:TMP=$env:TEMP; $env:OPENBLAS_NUM_THREADS='"'1'; "'$env:OMP_NUM_THREADS='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B research/stage4_math_and_design/M1/verify_m1.py"
```

**Aggregated Output:**

```text
Trained the six matched CPU connectors for seed 202611
Trained the six matched CPU connectors for seed 202622
Trained the six matched CPU connectors for seed 202633
G1_main_extrap_both Arm 0=62.16%, Arm 1=64.60%, Arm 2=64.62% Arm 3=64.62%
G2_interp_both Arm 0=45.21%, Arm 1=46.81%, Arm 2=46.78% Arm 3=46.78%
G3_extrap_angle Arm 0=58.80%, Arm 1=61.91%, Arm 2=62.11% Arm 3=62.11%
G4_extrap_freq Arm 0=45.18%, Arm 1=46.97%, Arm 2=47.75% Arm 3=47.75%
Traceback (most recent call last):
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M1\verify_m1.py", line 976, in <module>
    main()
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M1\verify_m1.py", line 915, in main
    geometry = candidate_diagnostics(models, contexts, encoder.table, host)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M1\verify_m1.py", line 727, in candidate_diagnostics
    np.testing.assert_allclose(host.w_out.numpy() @ host.w_out.numpy().T, projector, atol=1e-7)
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M1\.uv-cache\archive-v0\ilw4ovp7bGVkAiqx\Lib\site-packages\numpy\testing\_private\utils.py", line 1711, in assert_allclose
    assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
  File "C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M1\.uv-cache\archive-v0\ilw4ovp7bGVkAiqx\Lib\site-packages\numpy\testing\_private\utils.py", line 919, in assert_array_compare
    raise AssertionError(msg)
AssertionError: 
Not equal to tolerance rtol=1e-07, atol=1e-07

Mismatched elements: 2 / 16 (12.5%)
Max absolute difference among violations: 1.78813934e-07
Max relative difference among violations: 2.38418579e-07
 ACTUAL: array([[ 0.75, -0.25, -0.25, -0.25],
       [-0.25,  0.75, -0.25, -0.25],
       [-0.25, -0.25,  0.75, -0.25],
       [-0.25, -0.25, -0.25,  0.75]], dtype=float32)
 DESIRED: array([[ 0.75, -0.25, -0.25, -0.25],
       [-0.25,  0.75, -0.25, -0.25],
       [-0.25, -0.25,  0.75, -0.25],
       [-0.25, -0.25, -0.25,  0.75]])
```

### Event #29: `item.completed` — `file_change` (`item_26`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M1\\verify_m1.py",
    "kind": "update"
  }
]
```

### Event #30: `item.completed` — `agent_message` (`item_28`)

The first full pass put Host-Pure above 55% on both extrapolation cells. The run stopped at an isotropy check that used a tighter tolerance than its FP32 matrix multiplication supported. I changed the check to measure the same frozen weights in FP64 and started the full rerun. Training and scoring are unchanged.

### Event #31: `item.completed` — `file_change` (`item_29`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M1\\THEOREM_M1.md",
    "kind": "update"
  }
]
```

### Event #32: `item.completed` — `command_execution` (`item_27`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$m1=Join-Path (Get-Location) '"'research/stage4_math_and_design/M1'; "'$env:UV_CACHE_DIR=Join-Path $m1 '"'.uv-cache'; "'$env:UV_OFFLINE='"'1'; "'$env:UV_LINK_MODE='"'hardlink'; "'$env:UV_NO_PROJECT='"'1'; "'$env:TEMP=Join-Path $m1 '"'.uv-tmp'; "'$env:TMP=$env:TEMP; $env:OPENBLAS_NUM_THREADS='"'1'; "'$env:OMP_NUM_THREADS='"'1'; uv run --python 3.12 --with numpy==2.3.5 --with scipy==1.17.0 --with torch python -B research/stage4_math_and_design/M1/verify_m1.py"
```

**Aggregated Output:**

```text
Trained the six matched CPU connectors for seed 202611
Trained the six matched CPU connectors for seed 202622
Trained the six matched CPU connectors for seed 202633
G1_main_extrap_both Arm 0=62.16%, Arm 1=64.60%, Arm 2=64.62% Arm 3=64.62%
G2_interp_both Arm 0=45.21%, Arm 1=46.81%, Arm 2=46.78% Arm 3=46.78%
G3_extrap_angle Arm 0=58.80%, Arm 1=61.91%, Arm 2=62.11% Arm 3=62.11%
G4_extrap_freq Arm 0=45.18%, Arm 1=46.97%, Arm 2=47.75% Arm 3=47.75%
C:\Users\李则徐\.gemini\antigravity\scratch\EM-FloatFormer-stage4\research\stage4_math_and_design\M1\results_m1.json
{"arm1_arm2_grouped_defect_zero": true, "arm1_arm2_ungrouped_defect_at_most_1e_6": true, "host_frozen_and_same_trainable_parameter_count": true}
{"arm0_separate_nonquotient": {"G1_G3_seedmean_above_55_pct_in_both": true, "G1_G3_each_seed_above_55_pct_in_both": true, "all_cell_mean_ece_point_below_0p05": false}, "arm1_equivariant_host_pure": {"G1_G3_seedmean_above_55_pct_in_both": true, "G1_G3_each_seed_above_55_pct_in_both": true, "all_cell_mean_ece_point_below_0p05": true}, "arm2_equivariant_physical_residual": {"G1_G3_seedmean_above_55_pct_in_both": true, "G1_G3_each_seed_above_55_pct_in_both": true, "all_cell_mean_ece_point_below_0p05": true}}
```

### Event #33: `item.completed` — `command_execution` (`item_30`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$s=Get-Content -Raw research/stage4_math_and_design/M1/results_m1.json | ConvertFrom-Json
'"'Runtime: ' + ("'$s.runtime | ConvertTo-Json -Compress)
'"'Params: ' + ("'$s.train.trainable_parameters | ConvertTo-Json -Compress)
foreach ($cell in $s.cells.PSObject.Properties) {
  $cell.Name
  $v=$cell.Value
  foreach($m in $v.models.PSObject.Properties) {
    $r=$m.Value; $swap=$r.fixed_text_interventions.sign_swap.dev_cal_intervention_minus_matched.nll; $shuf=$r.fixed_text_interventions.balanced_shuffle.dev_cal_intervention_minus_matched.nll
    '"'{0} acc {1:F7} NLL {2:F7} ECE {3:F7} NIG {4:F6}; rawNLL {5:F7}; defect {6:G5} ung {7:G5}; swap {8:F6} [{9:F6},{10:F6}] shuffle {11:F6} [{12:F6},{13:F6}]' -f "'$m.Name,$r.dev_calibrated.acc,$r.dev_calibrated.nll,$r.dev_calibrated.ece,$r.dev_calibrated.reference_normalized_nig,$r.unit_T.nll,$r.maximum_equivariance_defect,$r.ungrouped_batch_maximum_equivariance_defect,$swap.mean,$swap.lo95,$swap.hi95,$shuf.mean,$shuf.lo95,$shuf.hi95
  }
  $v.C_direct_s17 | ConvertTo-Json -Compress
  foreach($contrast in $v.paired_contrasts.PSObject.Properties) {
    $c=$contrast.Value.dev_calibrated.nll; '"'{0} NLL {1:F7} [{2:F7},{3:F7}]' -f "'$contrast.Name,$c.mean,$c.lo95,$c.hi95
  }
  $b=$v.models.arm2_equivariant_physical_residual.bypass_diagnostics
  '"'Bypass: ' + ("'$b | ConvertTo-Json -Depth 4 -Compress)
}
$c=$s.candidate_jacobian
'"'Geometry global: ' + "'$c.host_out_projection_gram_max_abs_defect
foreach($n in $c.contexts.PSObject.Properties) {
  foreach($arm in $n.Value.PSObject.Properties) {
    foreach($run in $arm.Value.PSObject.Properties) {
      $j=$run.Value; $m=$j.shared_last_projection_tangent
      '"'{0} {1} {2} rank {3} min third {4:F5} cand. margin {5:F5} oracle margin {6:F5} agreement {7:F5} min D {8:F5} stuck {9:F5} stack rank {10} residual {11:F5}' -f "'$n.Name,$arm.Name,$run.Name,$j.rank_min,$j.minimum_third_singular_value,$j.native_margin_median_against_oracle_runner_up,$j.oracle_margin_median,$j.native_top1_agreement_with_oracle,$j.median_min_token_norm_to_fit_centered_target_linearly,$j.fraction_oracle_pairwise_margin_unreachable_in_linearized_ball,$m.rank,$m.relative_projection_residual
    }
  }
}
$s.interface_checks | ConvertTo-Json -Depth 8 -Compress'
```

**Aggregated Output:**

```text
Runtime: {"python":"3.12.7","numpy":"2.3.5","scipy":"1.17.0","torch":"2.12.1+cpu","device":"cpu"}
Params: {"arm0_separate_nonquotient":9064,"arm1_equivariant_host_pure":9064,"arm2_equivariant_physical_residual":9064,"control_quotient_separate":9064,"control_nonquotient_coupled":9064,"control_arm1_zero_slots_refit":9064}
G1_main_extrap_both
arm0_separate_nonquotient acc 0.6216139 NLL 1.0409368 ECE 0.0644920 NIG 0.775791; rawNLL 1.0450969; defect 0.84783 ung ; swap 1.612978 [1.588183,1.637772] shuffle 0.806734 [0.780798,0.832670]
arm1_equivariant_host_pure acc 0.6459924 NLL 0.9785287 ECE 0.0401738 NIG 0.915980; rawNLL 0.9797828; defect 0 ung 1.4901E-07; swap 2.017169 [1.987428,2.046911] shuffle 1.007212 [0.975569,1.038855]
arm2_equivariant_physical_residual acc 0.6461713 NLL 0.9422187 ECE 0.0069021 NIG 0.997545; rawNLL 0.9421678; defect 0 ung 5.9605E-08; swap 0.000050 [-0.000338,0.000439] shuffle 0.000206 [-0.000067,0.000478]
control_quotient_separate acc 0.6440045 NLL 0.9699287 ECE 0.0303667 NIG 0.935299; rawNLL 0.9705047; defect 0 ung 1.7881E-07; swap  [,] shuffle  [,]
control_nonquotient_coupled acc 0.6205404 NLL 1.0465698 ECE 0.0536501 NIG 0.763137; rawNLL 1.0542255; defect 0.74647 ung ; swap  [,] shuffle  [,]
control_arm1_zero_slots_refit acc 0.1538446 NLL 1.4948033 ECE 0.2888795 NIG -0.243748; rawNLL 1.5326185; defect 0 ung 1.4901E-07; swap  [,] shuffle  [,]
{"acc":0.6461712786259542,"nll":0.9411258248395115,"brier":0.5098317013509958,"ece":0.0009122218241484292,"tie_count":0}
arm1_minus_arm0 NLL -0.0624080 [-0.0656004,-0.0592157]
quotient_effect_with_separate NLL -0.0710081 [-0.0745501,-0.0674661]
quotient_effect_with_coupling NLL -0.0680410 [-0.0713516,-0.0647305]
coupling_effect_without_quotient NLL 0.0056330 [0.0036603,0.0076057]
coupling_effect_with_quotient NLL 0.0086001 [0.0073251,0.0098750]
arm1_minus_zero_slots_refit NLL -0.5162745 [-0.5286195,-0.5039296]
arm2_minus_arm1 NLL -0.0363101 [-0.0401116,-0.0325085]
arm1_minus_arm3 NLL 0.0374029 [0.0335280,0.0412779]
arm2_minus_arm3 NLL 0.0010929 [0.0004960,0.0016897]
Bypass: {"gamma_zero_max_abs_diff_from_arm3":4.9458963591320071E-08,"gamma_zero_slot_swap_defect":0.0,"unit_T_host_increment_arm2_minus_gamma_zero":{"acc":{"n_blocks":131,"mean":0.0,"block_sd":0.0,"se":0.0,"t_crit_95":1.9783804054470222,"lo95":0.0,"hi95":0.0},"nll":{"n_blocks":131,"mean":0.0010420152023473386,"block_sd":0.0032324930635073226,"se":0.0002824242302715014,"t_crit_95":1.9783804054470222,"lo95":0.00048327263915474256,"hi95":0.0016007577655399348}},"common_dev_T_host_increment_arm2_minus_gamma_zero":{"acc":{"n_blocks":131,"mean":0.0,"block_sd":0.0,"se":0.0,"t_crit_95":1.9783804054470222,"lo95":0.0,"hi95":0.0},"nll":{"n_blocks":131,"mean":0.0008519445436962199,"block_sd":0.003323662217388485,"se":0.000290389716230334,"t_crit_95":1.9783804054470222,"lo95":0.000277443219162806,"hi95":0.0014264458682296337}}}
G2_interp_both
arm0_separate_nonquotient acc 0.4521484 NLL 1.0538422 ECE 0.0358398 NIG 0.913169; rawNLL 1.0528925; defect 0.71217 ung ; swap 0.818174 [0.754032,0.882316] shuffle 0.407973 [0.350878,0.465069]
arm1_equivariant_host_pure acc 0.4680990 NLL 1.0234294 ECE 0.0168679 NIG 0.996706; rawNLL 1.0256090; defect 0 ung 1.1921E-07; swap 0.977818 [0.894259,1.061378] shuffle 0.491277 [0.423441,0.559113]
arm2_equivariant_physical_residual acc 0.4677734 NLL 1.0217442 ECE 0.0036378 NIG 1.001334; rawNLL 1.0223616; defect 0 ung 2.9802E-08; swap 0.000260 [-0.001127,0.001647] shuffle -0.000065 [-0.001558,0.001428]
control_quotient_separate acc 0.4687500 NLL 1.0247256 ECE 0.0202830 NIG 0.993145; rawNLL 1.0262365; defect 0 ung 1.1921E-07; swap  [,] shuffle  [,]
control_nonquotient_coupled acc 0.4563802 NLL 1.0557779 ECE 0.0393125 NIG 0.907852; rawNLL 1.0549389; defect 0.61003 ung ; swap  [,] shuffle  [,]
control_arm1_zero_slots_refit acc 0.4186198 NLL 1.2207380 ECE 0.0270822 NIG 0.454745; rawNLL 1.2213207; defect 0 ung 8.9407E-08; swap  [,] shuffle  [,]
{"acc":0.4677734375,"nll":1.022230015019136,"brier":0.5962492861161929,"ece":0.005695706517642937,"tie_count":0}
arm1_minus_arm0 NLL -0.0304128 [-0.0478076,-0.0130180]
quotient_effect_with_separate NLL -0.0291167 [-0.0465182,-0.0117151]
quotient_effect_with_coupling NLL -0.0323485 [-0.0508782,-0.0138187]
coupling_effect_without_quotient NLL 0.0019357 [-0.0018517,0.0057231]
coupling_effect_with_quotient NLL -0.0012962 [-0.0045481,0.0019558]
arm1_minus_zero_slots_refit NLL -0.1973086 [-0.2268325,-0.1677846]
arm2_minus_arm1 NLL -0.0016852 [-0.0067005,0.0033301]
arm1_minus_arm3 NLL 0.0011994 [-0.0045516,0.0069504]
arm2_minus_arm3 NLL -0.0004858 [-0.0022470,0.0012754]
Bypass: {"gamma_zero_max_abs_diff_from_arm3":3.011788557438777E-08,"gamma_zero_slot_swap_defect":0.0,"unit_T_host_increment_arm2_minus_gamma_zero":{"acc":{"n_blocks":8,"mean":0.0,"block_sd":0.0,"se":0.0,"t_crit_95":2.364624251592784,"lo95":0.0,"hi95":0.0},"nll":{"n_blocks":8,"mean":0.00013158851523597013,"block_sd":0.002150744756382806,"se":0.0007604031009198455,"t_crit_95":2.364624251592784,"lo95":-0.0016664790981854517,"hi95":0.0019296561286573921}},"common_dev_T_host_increment_arm2_minus_gamma_zero":{"acc":{"n_blocks":8,"mean":0.0,"block_sd":0.0,"se":0.0,"t_crit_95":2.364624251592784,"lo95":0.0,"hi95":0.0},"nll":{"n_blocks":8,"mean":-4.4550369871760741E-05,"block_sd":0.0022255765545323874,"se":0.0007868601368798215,"t_crit_95":2.364624251592784,"lo95":-0.0019051789321494043,"hi95":0.0018160781924058827}}}
G3_extrap_angle
arm0_separate_nonquotient acc 0.5879991 NLL 1.0760618 ECE 0.0422636 NIG 0.762754; rawNLL 1.0831384; defect 0.84238 ung ; swap 1.508179 [1.417909,1.598449] shuffle 0.760704 [0.604095,0.917312]
arm1_equivariant_host_pure acc 0.6191406 NLL 1.0130140 ECE 0.0350438 NIG 0.917767; rawNLL 1.0129485; defect 0 ung 1.1921E-07; swap 1.922893 [1.811427,2.034360] shuffle 0.967661 [0.790604,1.144717]
arm2_equivariant_physical_residual acc 0.6210938 NLL 0.9809814 ECE 0.0234208 NIG 0.996524; rawNLL 0.9810010; defect 0 ung 2.9802E-08; swap 0.000058 [-0.001867,0.001983] shuffle -0.000155 [-0.001502,0.001192]
control_quotient_separate acc 0.6145833 NLL 1.0059892 ECE 0.0395200 NIG 0.935038; rawNLL 1.0053924; defect 0 ung 1.7881E-07; swap  [,] shuffle  [,]
control_nonquotient_coupled acc 0.5882161 NLL 1.0864937 ECE 0.0310247 NIG 0.737106; rawNLL 1.0978529; defect 0.75586 ung ; swap  [,] shuffle  [,]
control_arm1_zero_slots_refit acc 0.1679688 NLL 1.5157842 ECE 0.2751883 NIG -0.318370; rawNLL 1.5561059; defect 0 ung 1.1921E-07; swap  [,] shuffle  [,]
{"acc":0.62109375,"nll":0.979567505293638,"brier":0.5312420968465117,"ece":0.024100180872919062,"tie_count":0}
arm1_minus_arm0 NLL -0.0630478 [-0.0893938,-0.0367018]
quotient_effect_with_separate NLL -0.0700726 [-0.1014446,-0.0387005]
quotient_effect_with_coupling NLL -0.0734798 [-0.0986029,-0.0483567]
coupling_effect_without_quotient NLL 0.0104320 [-0.0011367,0.0220006]
coupling_effect_with_quotient NLL 0.0070248 [-0.0005730,0.0146225]
arm1_minus_zero_slots_refit NLL -0.5027702 [-0.5632402,-0.4423002]
arm2_minus_arm1 NLL -0.0320326 [-0.0542561,-0.0098091]
arm1_minus_arm3 NLL 0.0334465 [0.0086121,0.0582809]
arm2_minus_arm3 NLL 0.0014139 [-0.0021363,0.0049641]
Bypass: {"gamma_zero_max_abs_diff_from_arm3":4.2513572551250434E-08,"gamma_zero_slot_swap_defect":0.0,"unit_T_host_increment_arm2_minus_gamma_zero":{"acc":{"n_blocks":8,"mean":0.0,"block_sd":0.0,"se":0.0,"t_crit_95":2.364624251592784,"lo95":0.0,"hi95":0.0},"nll":{"n_blocks":8,"mean":0.0014334563565199627,"block_sd":0.004362790100593656,"se":0.0015424792325116569,"t_crit_95":2.364624251592784,"lo95":-0.002213927444255326,"hi95":0.005080840157295251}},"common_dev_T_host_increment_arm2_minus_gamma_zero":{"acc":{"n_blocks":8,"mean":0.0,"block_sd":0.0,"se":0.0,"t_crit_95":2.364624251592784,"lo95":0.0,"hi95":0.0},"nll":{"n_blocks":8,"mean":0.001256741537557464,"block_sd":0.004497286855967658,"se":0.0015900310163979295,"t_crit_95":2.364624251592784,"lo95":-0.0025030843646018036,"hi95":0.005016567439716731}}}
G4_extrap_freq
arm0_separate_nonquotient acc 0.4518229 NLL 1.0579015 ECE 0.0445385 NIG 0.886486; rawNLL 1.0569756; defect 0.71426 ung ; swap 0.793399 [0.707544,0.879254] shuffle 0.407901 [0.319725,0.496077]
arm1_equivariant_host_pure acc 0.4697266 NLL 1.0209700 ECE 0.0118538 NIG 0.986182; rawNLL 1.0235266; defect 0 ung 1.1921E-07; swap 0.943717 [0.839952,1.047483] shuffle 0.483211 [0.382159,0.584263]
arm2_equivariant_physical_residual acc 0.4775391 NLL 1.0163873 ECE 0.0131332 NIG 0.998553; rawNLL 1.0172553; defect 0 ung 5.9605E-08; swap 0.000108 [-0.001613,0.001829] shuffle 0.000017 [-0.000826,0.000860]
control_quotient_separate acc 0.4713542 NLL 1.0233628 ECE 0.0159611 NIG 0.979723; rawNLL 1.0252381; defect 0 ung 1.1921E-07; swap  [,] shuffle  [,]
control_nonquotient_coupled acc 0.4471571 NLL 1.0594234 ECE 0.0402962 NIG 0.882378; rawNLL 1.0584480; defect 0.60794 ung ; swap  [,] shuffle  [,]
control_arm1_zero_slots_refit acc 0.4313151 NLL 1.1972751 ECE 0.0177365 NIG 0.510252; rawNLL 1.1937216; defect 0 ung 8.9407E-08; swap  [,] shuffle  [,]
{"acc":0.4775390625,"nll":1.0158511271590174,"brier":0.5936743155765972,"ece":0.015298223750495322,"tie_count":0}
arm1_minus_arm0 NLL -0.0369315 [-0.0483743,-0.0254887]
quotient_effect_with_separate NLL -0.0345387 [-0.0448077,-0.0242697]
quotient_effect_with_coupling NLL -0.0384534 [-0.0492796,-0.0276272]
coupling_effect_without_quotient NLL 0.0015219 [-0.0026127,0.0056566]
coupling_effect_with_quotient NLL -0.0023928 [-0.0051954,0.0004099]
arm1_minus_zero_slots_refit NLL -0.1763051 [-0.2133014,-0.1393087]
arm2_minus_arm1 NLL -0.0045827 [-0.0127402,0.0035748]
arm1_minus_arm3 NLL 0.0051189 [-0.0045935,0.0148312]
arm2_minus_arm3 NLL 0.0005362 [-0.0013787,0.0024511]
Bypass: {"gamma_zero_max_abs_diff_from_arm3":5.174014922060266E-08,"gamma_zero_slot_swap_defect":0.0,"unit_T_host_increment_arm2_minus_gamma_zero":{"acc":{"n_blocks":8,"mean":0.0,"block_sd":0.0,"se":0.0,"t_crit_95":2.364624251592784,"lo95":0.0,"hi95":0.0},"nll":{"n_blocks":8,"mean":0.0014041913073501308,"block_sd":0.0024902224063118113,"se":0.0008804265750828818,"t_crit_95":2.364624251592784,"lo95":-0.0006776867238376266,"hi95":0.0034860693385378883}},"common_dev_T_host_increment_arm2_minus_gamma_zero":{"acc":{"n_blocks":8,"mean":0.0,"block_sd":0.0,"se":0.0,"t_crit_95":2.364624251592784,"lo95":0.0,"hi95":0.0},"nll":{"n_blocks":8,"mean":0.001263071521109141,"block_sd":0.002574633438348566,"se":0.0009102703816629539,"t_crit_95":2.364624251592784,"lo95":-0.000889375898877699,"hi95":0.003415518941095981}}}
Geometry global: 7.67417140945525E-09
dev arm0_separate_nonquotient 202611 rank 3 min third 0.20293 cand. margin 0.31218 oracle margin 0.16577 agreement 0.60938 min D 2.53979 stuck 0.26563 stack rank 192 residual 0.00000
dev arm0_separate_nonquotient 202622 rank 3 min third 0.21684 cand. margin 0.46553 oracle margin 0.16577 agreement 0.70313 min D 2.33207 stuck 0.28125 stack rank 192 residual 0.00000
dev arm0_separate_nonquotient 202633 rank 3 min third 0.22540 cand. margin 0.25499 oracle margin 0.16577 agreement 0.59375 min D 3.15542 stuck 0.28125 stack rank 192 residual 0.00000
dev arm1_equivariant_host_pure 202611 rank 3 min third 0.16949 cand. margin 0.21705 oracle margin 0.16577 agreement 0.89063 min D 1.27775 stuck 0.09375 stack rank 192 residual 0.00000
dev arm1_equivariant_host_pure 202622 rank 3 min third 0.20077 cand. margin 0.20744 oracle margin 0.16577 agreement 0.81250 min D 1.16934 stuck 0.17188 stack rank 192 residual 0.00000
dev arm1_equivariant_host_pure 202633 rank 3 min third 0.17277 cand. margin 0.22787 oracle margin 0.16577 agreement 0.79688 min D 1.37786 stuck 0.15625 stack rank 192 residual 0.00000
dev arm2_equivariant_physical_residual 202611 rank 3 min third 0.17492 cand. margin -0.19410 oracle margin 0.16577 agreement 0.14063 min D 31.07692 stuck 0.50000 stack rank 192 residual 0.00000
dev arm2_equivariant_physical_residual 202622 rank 3 min third 0.16453 cand. margin -0.54609 oracle margin 0.16577 agreement 0.12500 min D 18.55004 stuck 0.53125 stack rank 192 residual 0.00000
dev arm2_equivariant_physical_residual 202633 rank 3 min third 0.19531 cand. margin -0.40470 oracle margin 0.16577 agreement 0.25000 min D 40.26434 stuck 0.54688 stack rank 192 residual 0.00000
G1_main_extrap_both arm0_separate_nonquotient 202611 rank 3 min third 0.21164 cand. margin 1.02808 oracle margin 0.69314 agreement 0.84375 min D 3.98495 stuck 0.12500 stack rank  residual 
G1_main_extrap_both arm0_separate_nonquotient 202622 rank 3 min third 0.23162 cand. margin 0.81500 oracle margin 0.69314 agreement 0.90625 min D 2.92559 stuck 0.03125 stack rank  residual 
G1_main_extrap_both arm0_separate_nonquotient 202633 rank 3 min third 0.25259 cand. margin 1.16841 oracle margin 0.69314 agreement 0.96875 min D 3.20008 stuck 0.00000 stack rank  residual 
G1_main_extrap_both arm1_equivariant_host_pure 202611 rank 3 min third 0.17951 cand. margin 0.52175 oracle margin 0.69314 agreement 0.96875 min D 2.52216 stuck 0.03125 stack rank  residual 
G1_main_extrap_both arm1_equivariant_host_pure 202622 rank 3 min third 0.20035 cand. margin 0.78451 oracle margin 0.69314 agreement 1.00000 min D 2.02262 stuck 0.00000 stack rank  residual 
G1_main_extrap_both arm1_equivariant_host_pure 202633 rank 3 min third 0.17730 cand. margin 0.52722 oracle margin 0.69314 agreement 1.00000 min D 2.40500 stuck 0.00000 stack rank  residual 
G1_main_extrap_both arm2_equivariant_physical_residual 202611 rank 3 min third 0.17428 cand. margin -1.27085 oracle margin 0.69314 agreement 0.34375 min D 28.88785 stuck 0.62500 stack rank  residual 
G1_main_extrap_both arm2_equivariant_physical_residual 202622 rank 3 min third 0.17916 cand. margin -0.21397 oracle margin 0.69314 agreement 0.18750 min D 16.59308 stuck 0.53125 stack rank  residual 
G1_main_extrap_both arm2_equivariant_physical_residual 202633 rank 3 min third 0.19975 cand. margin 0.15959 oracle margin 0.69314 agreement 0.34375 min D 41.61187 stuck 0.46875 stack rank  residual 
G2_interp_both arm0_separate_nonquotient 202611 rank 3 min third 0.20583 cand. margin 0.06639 oracle margin 0.14947 agreement 0.50000 min D 2.68119 stuck 0.40625 stack rank  residual 
G2_interp_both arm0_separate_nonquotient 202622 rank 3 min third 0.22162 cand. margin 0.50745 oracle margin 0.14947 agreement 0.68750 min D 2.24364 stuck 0.25000 stack rank  residual 
G2_interp_both arm0_separate_nonquotient 202633 rank 3 min third 0.23114 cand. margin 0.36436 oracle margin 0.14947 agreement 0.68750 min D 3.21122 stuck 0.21875 stack rank  residual 
G2_interp_both arm1_equivariant_host_pure 202611 rank 3 min third 0.17340 cand. margin 0.25398 oracle margin 0.14947 agreement 0.75000 min D 1.26319 stuck 0.15625 stack rank  residual 
G2_interp_both arm1_equivariant_host_pure 202622 rank 3 min third 0.20261 cand. margin 0.22630 oracle margin 0.14947 agreement 0.93750 min D 1.09097 stuck 0.06250 stack rank  residual 
G2_interp_both arm1_equivariant_host_pure 202633 rank 3 min third 0.17439 cand. margin 0.17867 oracle margin 0.14947 agreement 0.84375 min D 1.25126 stuck 0.15625 stack rank  residual 
G2_interp_both arm2_equivariant_physical_residual 202611 rank 3 min third 0.17512 cand. margin 0.43600 oracle margin 0.14947 agreement 0.21875 min D 30.85882 stuck 0.40625 stack rank  residual 
G2_interp_both arm2_equivariant_physical_residual 202622 rank 3 min third 0.16731 cand. margin 0.47587 oracle margin 0.14947 agreement 0.18750 min D 18.46892 stuck 0.43750 stack rank  residual 
G2_interp_both arm2_equivariant_physical_residual 202633 rank 3 min third 0.19674 cand. margin 3.56771 oracle margin 0.14947 agreement 0.34375 min D 39.75449 stuck 0.37500 stack rank  residual 
G3_extrap_angle arm0_separate_nonquotient 202611 rank 3 min third 0.21309 cand. margin 0.99219 oracle margin 0.66617 agreement 0.84375 min D 3.80120 stuck 0.12500 stack rank  residual 
G3_extrap_angle arm0_separate_nonquotient 202622 rank 3 min third 0.23146 cand. margin 0.82283 oracle margin 0.66617 agreement 0.93750 min D 3.06111 stuck 0.03125 stack rank  residual 
G3_extrap_angle arm0_separate_nonquotient 202633 rank 3 min third 0.25249 cand. margin 1.20007 oracle margin 0.66617 agreement 0.96875 min D 3.27557 stuck 0.00000 stack rank  residual 
G3_extrap_angle arm1_equivariant_host_pure 202611 rank 3 min third 0.18871 cand. margin 0.56519 oracle margin 0.66617 agreement 0.93750 min D 2.55130 stuck 0.03125 stack rank  residual 
G3_extrap_angle arm1_equivariant_host_pure 202622 rank 3 min third 0.20195 cand. margin 0.78925 oracle margin 0.66617 agreement 1.00000 min D 2.04848 stuck 0.00000 stack rank  residual 
G3_extrap_angle arm1_equivariant_host_pure 202633 rank 3 min third 0.18482 cand. margin 0.52640 oracle margin 0.66617 agreement 1.00000 min D 2.20288 stuck 0.00000 stack rank  residual 
G3_extrap_angle arm2_equivariant_physical_residual 202611 rank 3 min third 0.17234 cand. margin -1.23101 oracle margin 0.66617 agreement 0.34375 min D 28.36920 stuck 0.62500 stack rank  residual 
G3_extrap_angle arm2_equivariant_physical_residual 202622 rank 3 min third 0.17722 cand. margin -0.17100 oracle margin 0.66617 agreement 0.18750 min D 16.90780 stuck 0.53125 stack rank  residual 
G3_extrap_angle arm2_equivariant_physical_residual 202633 rank 3 min third 0.19861 cand. margin 0.18921 oracle margin 0.66617 agreement 0.34375 min D 40.92911 stuck 0.46875 stack rank  residual 
G4_extrap_freq arm0_separate_nonquotient 202611 rank 3 min third 0.20825 cand. margin 0.06775 oracle margin 0.14963 agreement 0.50000 min D 2.78158 stuck 0.40625 stack rank  residual 
G4_extrap_freq arm0_separate_nonquotient 202622 rank 3 min third 0.22120 cand. margin 0.46895 oracle margin 0.14963 agreement 0.68750 min D 2.35759 stuck 0.25000 stack rank  residual 
G4_extrap_freq arm0_separate_nonquotient 202633 rank 3 min third 0.22939 cand. margin 0.35092 oracle margin 0.14963 agreement 0.65625 min D 3.22692 stuck 0.21875 stack rank  residual 
G4_extrap_freq arm1_equivariant_host_pure 202611 rank 3 min third 0.17011 cand. margin 0.25637 oracle margin 0.14963 agreement 0.78125 min D 1.24527 stuck 0.15625 stack rank  residual 
G4_extrap_freq arm1_equivariant_host_pure 202622 rank 3 min third 0.20004 cand. margin 0.20794 oracle margin 0.14963 agreement 0.93750 min D 1.18307 stuck 0.06250 stack rank  residual 
G4_extrap_freq arm1_equivariant_host_pure 202633 rank 3 min third 0.17157 cand. margin 0.18371 oracle margin 0.14963 agreement 0.81250 min D 1.35892 stuck 0.12500 stack rank  residual 
G4_extrap_freq arm2_equivariant_physical_residual 202611 rank 3 min third 0.17564 cand. margin 0.46936 oracle margin 0.14963 agreement 0.21875 min D 30.86271 stuck 0.43750 stack rank  residual 
G4_extrap_freq arm2_equivariant_physical_residual 202622 rank 3 min third 0.16883 cand. margin 0.45205 oracle margin 0.14963 agreement 0.18750 min D 18.63591 stuck 0.43750 stack rank  residual 
G4_extrap_freq arm2_equivariant_physical_residual 202633 rank 3 min third 0.19726 cand. margin 3.59973 oracle margin 0.14963 agreement 0.37500 min D 39.88152 stuck 0.37500 stack rank  residual 
{"all_65536_prefixes_and_four_turns_checked":true,"host_pure_probabilities_independent_of_kernel_even_if_nan":true,"m3_marginals_and_missing_fallback_checked":true,"m3_rx_tx_action_max_probability_defect":{"q1":0.0,"q2":0.0,"q3_repeat":0.0,"q3_indep":0.0,"q3_after_pilot":0.0},"synthetic_r3_matched_filter_max_abs_difference":9.0365607197660547E-16,"synthetic_r3_packet_model_rx_gauge_defects":{"arm1_equivariant_host_pure":{"4":0.0,"16":0.0},"arm2_equivariant_physical_residual":{"4":0.0,"16":0.0}},"hardware_capture_scored":false,"extended_packet_weights_trained":false,"shared_cfo_joint_moment":{"edges":[[0,2],[2,4],[0,4]],"harmonics":[1,-2,1],"covariance":[[0.18,0.16,0.34],[0.16,0.18,0.34],[0.34,0.34,0.68]],"closed_form_re_im":[0.7868196039421212,0.3898288042922555],"gauss_hermite_abs_error":1.9003957039824482E-15,"spuriously_independent_rate_abs_error":0.4150823626093332}}
```

### Event #34: `item.completed` — `agent_message` (`item_31`)

The rerun exited 0. The controls sharpen the result. Quotient sharing improved NLL in every cell, while adding the explicit products increased G1 NLL by 0.0086 nats with the quotient held fixed. Host-Pure was sensitive to signed-slot swaps. The bounded residual showed no resolved slot effect and slightly worsened G1 NLL against physical-only.

### Event #35: `item.completed` — `file_change` (`item_32`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M1\\THEOREM_M1.md",
    "kind": "update"
  }
]
```

### Event #36: `item.completed` — `file_change` (`item_33`)

```json
[
  {
    "path": "C:\\Users\\李则徐\\.gemini\\antigravity\\scratch\\EM-FloatFormer-stage4\\research\\stage4_math_and_design\\M1\\reply_r2.md",
    "kind": "add"
  }
]
```

### Event #37: `item.completed` — `command_execution` (`item_34`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$root=(Resolve-Path -LiteralPath '"'research/stage4_math_and_design/M1').ProviderPath; "'$cache=Get-Item -Force -LiteralPath '"'research/stage4_math_and_design/M1/.uv-cache'; if ("'$cache.Parent.FullName -ne $root -or ($cache.Attributes -band [IO.FileAttributes]::ReparsePoint)) { throw '"'Unexpected uv cache path' }; uv cache clean --cache-dir research/stage4_math_and_design/M1/.uv-cache; Get-ChildItem -Force -Recurse -LiteralPath research/stage4_math_and_design/M1/.uv-tmp | Select-Object Name,PSIsContainer"
```

**Aggregated Output:**

```text
Clearing cache at: research/stage4_math_and_design/M1/.uv-cache
Removed 34670 files (1.2GiB)

Name              PSIsContainer
----              -------------
torchinductor_李则徐          True


```

### Event #38: `item.completed` — `command_execution` (`item_35`)

- Status: `completed`, Exit Code: `0`

**Command:**

```powershell
"C:\\WINDOWS\\System32\\WindowsPowerShell\\v1.0\\powershell.exe" -NoProfile -Command '$tmp=Get-Item -Force -LiteralPath '"'research/stage4_math_and_design/M1/.uv-tmp/torchinductor_李则徐'; if (@(Get-ChildItem -Force -LiteralPath "'$tmp.FullName).Count -ne 0) { throw '"'Unexpected temporary files' }; Remove-Item -LiteralPath 'research/stage4_math_and_design/M1/.uv-tmp/torchinductor_李则徐' -ErrorAction Stop; Remove-Item -LiteralPath 'research/stage4_math_and_design/M1/.uv-tmp' -ErrorAction Stop; "'$root='"'research/stage4_math_and_design/M1'; Get-ChildItem -Force -LiteralPath "'$root | Select-Object Name,Length; git status --short --untracked-files=all -- research/stage4_math_and_design/M1; $j=Get-Content -Raw -LiteralPath (Join-Path $root '"'results_m1.json') | ConvertFrom-Json; if ("'$j.round -ne 2 -or @($j.cells.PSObject.Properties).Count -ne 4 -or $j.interface_checks.hardware_capture_scored) { throw '"'Round 2 receipt failed' }; "'$j.verified_structural_checks | ConvertTo-Json -Compress; rg -n '"'"'^## Round 2 Convergence|''^### 6.6|''^# M1 Round 2 handoff'"' research/stage4_math_and_design/M1/THEOREM_M1.md research/stage4_math_and_design/M1/reply_r2.md"
```

**Aggregated Output:**

```text

?? research/stage4_math_and_design/M1/THEOREM_M1.md
?? research/stage4_math_and_design/M1/reply_r2.md
?? research/stage4_math_and_design/M1/results_m1.json
?? research/stage4_math_and_design/M1/verify_m1.py
research/stage4_math_and_design/M1/reply_r2.md:1:# M1 Round 2 handoff
research/stage4_math_and_design/M1/THEOREM_M1.md:207:## Round 2 Convergence
research/stage4_math_and_design/M1/THEOREM_M1.md:295:### 6.6 Round 2 verification results
Name            Length
----            ------
reply_r2.md       2705
results_m1.json 344932
THEOREM_M1.md    48818
verify_m1.py     50874
{"arm1_arm2_grouped_defect_zero":true,"arm1_arm2_ungrouped_defect_at_most_1e_6":true,"host_frozen_and_same_trainable_parameter_count":true}


```

### Event #39: `item.completed` — `agent_message` (`item_36`)

**Host-Pure cleared 55% on both extrapolation cells in every seed**, reaching **64.60% on G1** and **61.91% on G3** using frozen-host scores alone. Host-Pure and Physical+Residual both had **zero equivariance defect** with canonical reuse. Their worst ungrouped defects were \(1.49\times10^{-7}\) and \(5.96\times10^{-8}\).

The matched controls resolve three design decisions:

- **Keep quotient sharing.** It improved NLL in all four cells at both coupling settings. Adding explicit multiplicative terms at fixed quotient increased G1 NLL by **0.00860 nats**, so carry the quotient-separate comparator into the Qwen test.
- **Keep physical-only as Arm 3.** Host-Pure used the signed slots: swapping them at fixed text increased G1 NLL by **2.0172 nats**. The bounded residual left top-1 unchanged across all cells and increased G1 NLL by **0.00109 nats** against physical-only.
- **Use the locked cache and gauge contract.** The 16-bit packet preserves S17 bit-exactly. A common receive turn fixes Q1 and Q2 and rolls Q3 receiver sectors. M3’s joint factor keeps Q2 in the TX frame, and CFO uncertainty remains shared across clocks. These contracts passed synthetic checks for both packet formats. Extended-field predictive gains, Qwen margins, and Pluto/CTTC performance remain unmeasured.

The updated verifier exited **0** under the prescribed `uv run` command. [Round 2 handoff](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M1/reply_r2.md>) · [Theory and contracts](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M1/THEOREM_M1.md:207>) · [Results and candidate-Jacobian diagnostics](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M1/results_m1.json>)



### Event #40: `turn.completed`

- Usage: `{"input_tokens": 12566320, "cached_input_tokens": 12260096, "cache_write_input_tokens": 0, "output_tokens": 77884, "reasoning_output_tokens": 17187}`


## Round 2 — Final Output (`reply_r2.md`)

```markdown
**Host-Pure cleared 55% on both extrapolation cells in every seed**, reaching **64.60% on G1** and **61.91% on G3** using frozen-host scores alone. Host-Pure and Physical+Residual both had **zero equivariance defect** with canonical reuse. Their worst ungrouped defects were \(1.49\times10^{-7}\) and \(5.96\times10^{-8}\).

The matched controls resolve three design decisions:

- **Keep quotient sharing.** It improved NLL in all four cells at both coupling settings. Adding explicit multiplicative terms at fixed quotient increased G1 NLL by **0.00860 nats**, so carry the quotient-separate comparator into the Qwen test.
- **Keep physical-only as Arm 3.** Host-Pure used the signed slots: swapping them at fixed text increased G1 NLL by **2.0172 nats**. The bounded residual left top-1 unchanged across all cells and increased G1 NLL by **0.00109 nats** against physical-only.
- **Use the locked cache and gauge contract.** The 16-bit packet preserves S17 bit-exactly. A common receive turn fixes Q1 and Q2 and rolls Q3 receiver sectors. M3’s joint factor keeps Q2 in the TX frame, and CFO uncertainty remains shared across clocks. These contracts passed synthetic checks for both packet formats. Extended-field predictive gains, Qwen margins, and Pluto/CTTC performance remain unmeasured.

The updated verifier exited **0** under the prescribed `uv run` command. [Round 2 handoff](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M1/reply_r2.md>) · [Theory and contracts](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M1/THEOREM_M1.md:207>) · [Results and candidate-Jacobian diagnostics](<C:/Users/李则徐/.gemini/antigravity/scratch/EM-FloatFormer-stage4/research/stage4_math_and_design/M1/results_m1.json>)

```
